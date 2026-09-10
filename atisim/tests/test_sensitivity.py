"""Tests for atisim/sensitivity.py -- phase S0 of the sensitivity study.

WHAT THESE ASSERT, AND WHY EACH ONE IS HERE. Every check in this file guards one
of the three obstacles the module's docstring names. They are written as
negative controls wherever a negative control exists, because the failure mode
of a sensitivity study is not a crash: it is a plausible number computed about
the wrong thing.

  - The jnp plant matrices are a SECOND PATH to something `validation` already
    computes. Two paths that compute the same thing drift apart, so their
    agreement is asserted rather than assumed -- exactly as `vortex_viz._measure`
    exists so the three Fig. 8 points cannot.
  - The implicit trim Jacobian has a free negative control: a LATERAL derivative
    cannot move a wings-level longitudinal trim, so its trim sensitivity must be
    an EXACT zero. A small number there would mean the differentiation is
    reaching something it should not.
  - The eigenvalue perturbation is checked against a central difference on the
    real plant, which is the only comparison that tests the formula rather than
    restating it.

Design: docs/superpowers/specs/2026-09-10-model-sensitivity-analysis-design.md.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import atisim  # noqa: F401  -- enables x64
from atisim import sensitivity, trim, validation
from atisim.aircraft import REGISTRY
from atisim.units import FT2M

# Caughey's 747 power approach: the condition session 11 swept, and therefore the
# only condition at which this project holds an independent sensitivity answer.
CAUGHEY_V = 279.1 * FT2M
CAUGHEY_H = 0.0


@pytest.fixture(scope="module")
def approach():
    ac = REGISTRY["boeing747_approach"]
    x, res = trim.trim(jnp.array(CAUGHEY_V), jnp.array(CAUGHEY_H), ac)
    return ac, x, res


# --------------------------------------------------------------------------
# The second path, asserted against the first
# --------------------------------------------------------------------------


def test_the_jnp_longitudinal_matrix_is_bit_identical_to_validations(approach):
    """Exact equality, not a tolerance.

    `sensitivity.longitudinal_matrix_jnp` exists only because
    `validation.longitudinal_matrix` ends in `np.asarray` and takes `alpha`
    through `np.cos`, so the AD chain stops there. It is the same arithmetic in
    the same order, so the claim is bit-identity -- and a tolerance would admit a
    second path that had quietly become a different model.
    """
    ac, x, _ = approach
    a, de, thr = float(x[0]), float(x[1]), float(x[2])
    A_validation = validation.longitudinal_matrix(ac, a, de, thr, CAUGHEY_V, CAUGHEY_H)
    A_jnp = np.asarray(
        sensitivity.longitudinal_matrix_jnp(ac, x[0], x[1], x[2], CAUGHEY_V, CAUGHEY_H)
    )
    assert np.array_equal(A_validation, A_jnp), (
        f"max |diff| {np.abs(A_validation - A_jnp).max():.3e}"
    )


def test_the_jnp_lateral_matrix_reproduces_validations_lateral_modes(approach):
    """The lateral twin has no matrix to compare against -- so compare the MODES.

    `validation.lateral_modes` builds its matrix inline and returns only the
    modes, so the second path is checked through the three quantities that path
    exists to produce.
    """
    ac, x, _ = approach
    a, de, thr = float(x[0]), float(x[1]), float(x[2])
    (dr_wn, dr_zeta), roll_tau, spiral_tau = validation.lateral_modes(
        ac, a, de, thr, CAUGHEY_V, CAUGHEY_H
    )
    A = np.asarray(
        sensitivity.lateral_matrix_jnp(ac, x[0], x[1], x[2], CAUGHEY_V, CAUGHEY_H)
    )
    lam = np.linalg.eigvals(A)
    osc = [l for l in lam if l.imag > 1e-9]
    reals = sorted((-1.0 / l.real for l in lam if abs(l.imag) <= 1e-9), key=abs)
    assert len(osc) == 1
    assert abs(osc[0]) == pytest.approx(dr_wn, rel=1e-12)
    assert -osc[0].real / abs(osc[0]) == pytest.approx(dr_zeta, rel=1e-12)
    assert reals[0] == pytest.approx(roll_tau, rel=1e-12)
    assert reals[1] == pytest.approx(spiral_tau, rel=1e-12)


# --------------------------------------------------------------------------
# The trim solve, and its free negative control
# --------------------------------------------------------------------------


def test_a_lateral_derivative_cannot_move_a_wings_level_trim(approach):
    """EXACT zero, and it is the cheapest check that the AD reaches the right thing.

    The trim residual is [udot, wdot, qdot] at beta = 0, wings level, no body
    rates. No lateral coefficient appears in any of the three. A small non-zero
    here would mean the differentiation is picking up something the trim does not
    actually depend on, and every longitudinal number downstream would be
    suspect for the same reason.
    """
    ac, x, _ = approach
    J = sensitivity.implicit_trim_jacobian(
        ac, CAUGHEY_V, CAUGHEY_H, fields=sensitivity.LATERAL_FIELDS, x=x
    )
    worst = max(float(np.abs(J[f]).max()) for f in sensitivity.LATERAL_FIELDS)
    assert worst == 0.0, f"a lateral derivative moved the trim by {worst:.3e}"


def test_thrust_moves_only_the_throttle(approach):
    """A second exact-zero control, and it is a statement about ASSUMPTIONS C5.

    Thrust acts along body x through the CG with no moment, so more available
    thrust changes the throttle FRACTION needed and nothing else: alpha and
    elevator must not move at all. If C5 ever gains a thrust moment, this test
    is the one that should go red first.
    """
    ac, x, _ = approach
    J = sensitivity.implicit_trim_jacobian(
        ac, CAUGHEY_V, CAUGHEY_H, fields=("max_thrust",), x=x
    )["max_thrust"]
    assert J[0] == 0.0 and J[1] == 0.0, f"thrust moved alpha/elevator: {J}"
    assert J[2] != 0.0, "more thrust must reduce the throttle fraction"


@pytest.mark.parametrize("field", ["CD0", "CLa", "Cma", "mass"])
def test_the_implicit_trim_jacobian_matches_a_difference_on_the_real_solver(
    approach, field
):
    """The implicit function theorem, against differentiating the solver itself.

    This is the check that the theorem was applied to the right residual with the
    right sign. The tolerance is the central difference's own, not the model's.
    """
    ac, x, _ = approach
    ad = sensitivity.implicit_trim_jacobian(
        ac, CAUGHEY_V, CAUGHEY_H, fields=(field,), x=x
    )[field]

    p0 = float(getattr(ac, field))
    h = abs(p0) * 1e-6
    got = []
    for s in (+1.0, -1.0):
        swept = ac._replace(**{field: jnp.array(p0 + s * h)})
        xx, _ = trim.trim(jnp.array(CAUGHEY_V), jnp.array(CAUGHEY_H), swept)
        got.append(np.asarray(xx))
    fd = (got[0] - got[1]) / (2.0 * h)

    np.testing.assert_allclose(ad, fd, rtol=1e-4, atol=1e-12)


# --------------------------------------------------------------------------
# The eigenvalue perturbation
# --------------------------------------------------------------------------


def test_a_zero_perturbation_moves_no_eigenvalue(approach):
    """The degenerate case, asserted so the einsum's index order cannot be wrong.

    dA = 0 must give dlambda = 0 exactly, and the eigenvalues returned must be
    the matrix's own.
    """
    ac, x, _ = approach
    A = np.asarray(
        sensitivity.longitudinal_matrix_jnp(ac, x[0], x[1], x[2], CAUGHEY_V, CAUGHEY_H)
    )
    lam, dlam = sensitivity.eigenvalue_sensitivity(A, np.zeros_like(A))
    np.testing.assert_allclose(np.sort_complex(lam), np.sort_complex(np.linalg.eigvals(A)))
    assert np.max(np.abs(dlam)) < 1e-12


def test_the_eigenvalue_perturbation_matches_a_difference_on_the_real_plant(approach):
    """First-order perturbation against re-solving the eigenproblem.

    Perturb the plant matrix along a direction the model actually produces --
    dA/dCma -- and compare the predicted eigenvalue shift with the one a re-solve
    gives. Restating the formula would not test it; re-solving does.
    """
    ac, x, _ = approach
    A, dA, _, _ = sensitivity.plant_matrix_sensitivity(
        ac, CAUGHEY_V, CAUGHEY_H, axis="longitudinal", fields=("Cma",)
    )
    direction = dA["Cma"]
    lam, dlam = sensitivity.eigenvalue_sensitivity(A, direction)

    eps = 1e-7
    up = np.sort_complex(np.linalg.eigvals(A + eps * direction))
    down = np.sort_complex(np.linalg.eigvals(A - eps * direction))
    fd = (up - down) / (2.0 * eps)
    # `np.argsort` on complex sorts lexicographically -- real part, then
    # imaginary -- which is exactly what `np.sort_complex` above does, so this
    # is what puts the perturbation result in the same order as the re-solve.
    np.testing.assert_allclose(fd, dlam[np.argsort(lam)], rtol=1e-5, atol=1e-9)


def test_eigenvalue_separation_is_reported_and_positive(approach):
    """The number that says whether a first-order perturbation may be quoted."""
    ac, x, _ = approach
    A = np.asarray(
        sensitivity.longitudinal_matrix_jnp(ac, x[0], x[1], x[2], CAUGHEY_V, CAUGHEY_H)
    )
    assert sensitivity.eigenvalue_separation(A) > 0.1


# --------------------------------------------------------------------------
# The parameters that may not be varied alone
# --------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["inertia", "inertia_inv", "AR", "S", "b"])
def test_an_algebraically_coupled_field_is_refused(field):
    """`Aircraft` is not a set of independent parameters, and this is the guard.

    `inertia_inv` is the inverse of `inertia`; `AR` is b^2/S. A `jacfwd` over the
    whole tuple happily returns a derivative with respect to each member alone,
    and that derivative describes an airframe whose stored quantities contradict
    each other. Refusing is a declared modelling statement; silently screening
    them would be an invented one.
    """
    with pytest.raises(ValueError, match="cannot be varied alone"):
        sensitivity.implicit_trim_jacobian(
            REGISTRY["boeing747_approach"], CAUGHEY_V, CAUGHEY_H, fields=(field,)
        )


def test_an_unknown_field_is_refused():
    with pytest.raises(ValueError, match="not fields of Aircraft"):
        sensitivity.implicit_trim_jacobian(
            REGISTRY["boeing747_approach"], CAUGHEY_V, CAUGHEY_H, fields=("Cxyz",)
        )


def test_the_independent_and_coupled_sets_do_not_overlap():
    overlap = set(sensitivity.INDEPENDENT_FIELDS) & set(sensitivity.COUPLED_FIELDS)
    assert overlap == set(), f"a field is both independent and coupled: {overlap}"
    for f in sensitivity.INDEPENDENT_FIELDS:
        assert f in REGISTRY["boeing747_approach"]._fields


# --------------------------------------------------------------------------
# Mode selection, which is where a study starts reporting the wrong mode
# --------------------------------------------------------------------------


def test_picking_a_mode_fails_loudly_when_the_predicate_is_ambiguous(approach):
    """Two oscillatory roots and a predicate that matches both is an ERROR.

    The longitudinal plant has a phugoid and a short period, both oscillatory. A
    `[0]` there would silently return whichever `np.linalg.eig` happened to order
    first, which is not a defined mode.
    """
    ac, _, _ = approach
    modes, _, _, _, _, _ = sensitivity.mode_sensitivity(
        ac, CAUGHEY_V, CAUGHEY_H, axis="longitudinal", fields=("Cma",)
    )
    with pytest.raises(ValueError, match="2 roots matched"):
        sensitivity.pick(modes["Cma"], sensitivity.oscillatory, "the oscillatory one")

    ordered = sensitivity.oscillatory_modes(modes["Cma"])
    assert len(ordered) == 2
    assert ordered[0].wn < ordered[1].wn


def test_elasticity_is_nan_rather_than_large_where_it_is_undefined():
    """A zero base value must not sort to the top of a ranking."""
    assert sensitivity.elasticity(2.0, 3.0, 6.0) == pytest.approx(1.0)
    assert np.isnan(sensitivity.elasticity(2.0, 3.0, 0.0))
    assert np.isnan(sensitivity.elasticity(2.0, 0.0, 6.0))


# --------------------------------------------------------------------------
# S1: the falsification against session 11
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,field,quantity,sign",
    [
        ("phugoid zeta vs CD0", "CD0", "ph_zeta", +1.0),
        ("short-period wn^2 vs Cma", "Cma", "sp_wn2", +1.0),
        ("1/tau_roll vs |Clp|", "Clp", "inv_tau_roll", -1.0),
        ("dutch-roll wn^2 vs Cnb", "Cnb", "dr_wn2", +1.0),
    ],
)
def test_the_ad_screen_matches_a_central_difference_on_the_same_quantity(
    name, field, quantity, sign
):
    """Phase S1's real gate: the MACHINERY, against a difference at the same point.

    Section 4's session-11 slopes are LEAST-SQUARES fits over wide ranges, so
    comparing an AD tangent against them tests affineness and machinery at once
    and cannot separate them. This test isolates the machinery: same base point,
    same quantity, central difference on the shipped `validation` path. Section 4
    carries the other comparison and what its residual turned out to be.

    `sign` carries the |Clp| convention: the published relation is in |Clp| and
    Clp is negative, so d/d|Clp| = -d/dClp.
    """
    ac = REGISTRY["boeing747_approach"]
    axis = "longitudinal" if quantity in ("ph_zeta", "sp_wn2") else "lateral"
    modes, _, _, _, _, _ = sensitivity.mode_sensitivity(
        ac, CAUGHEY_V, CAUGHEY_H, axis=axis, fields=(field,)
    )

    def extract_ad(ms):
        if quantity == "ph_zeta":
            return sensitivity.oscillatory_modes(ms)[0].dzeta
        if quantity == "sp_wn2":
            m = sensitivity.oscillatory_modes(ms)[-1]
            return 2.0 * m.wn * m.dwn
        if quantity == "dr_wn2":
            m = sensitivity.oscillatory_modes(ms)[0]
            return 2.0 * m.wn * m.dwn
        # 1/tau_roll = -Re(lambda), and the fast real root is the roll mode
        return -sensitivity.real_modes(ms)[0].dlam.real

    def evaluate(a):
        x, _ = trim.trim(jnp.array(CAUGHEY_V), jnp.array(CAUGHEY_H), a)
        al, de, thr = float(x[0]), float(x[1]), float(x[2])
        if quantity == "ph_zeta":
            return validation.longitudinal_modes(a, al, de, thr, CAUGHEY_V, CAUGHEY_H)[0][1]
        if quantity == "sp_wn2":
            return validation.longitudinal_modes(a, al, de, thr, CAUGHEY_V, CAUGHEY_H)[-1][0] ** 2
        if quantity == "dr_wn2":
            return validation.lateral_modes(a, al, de, thr, CAUGHEY_V, CAUGHEY_H)[0][0] ** 2
        return 1.0 / validation.lateral_modes(a, al, de, thr, CAUGHEY_V, CAUGHEY_H)[1]

    p0 = float(getattr(ac, field))
    h = abs(p0) * 1e-5
    fd = (evaluate(ac._replace(**{field: jnp.array(p0 + h)}))
          - evaluate(ac._replace(**{field: jnp.array(p0 - h)}))) / (2.0 * h)

    ad = sign * extract_ad(modes[field])
    fd = sign * fd
    assert ad == pytest.approx(fd, rel=1e-6), f"{name}: AD {ad}, central difference {fd}"


# --------------------------------------------------------------------------
# S2: the differentiable load path, and the trim primitive under it
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mehta():
    """The headline run's configuration, short enough to be a test.

    2 s rather than the headline's 47 s: this file asserts that the two paths
    AGREE, which a short run tests exactly as well as a long one and 24x faster.
    `scripts/sensitivity_load.py` flies the real thing.
    """
    from atisim import wind
    from atisim.aircraft import CRUISE

    ac = REGISTRY["boeing747"]
    V = CRUISE["boeing747"]["airspeed"]
    H = wind.MEHTA_HANNIBAL_ALTITUDE
    array = wind.mehta_hannibal_array(H)

    def field(p):
        return wind.vortex_wind(p, array)

    r0 = float(array.r0)
    start = float(array.north.min()) - 12.0 * r0
    return dict(ac=ac, V=V, H=H, field=field, model=wind.field_model(field),
                start_north=start, dt=0.01, n_steps=200)


def test_the_differentiable_load_is_bit_identical_to_measure(mehta):
    """The S0 gate for the load path. Exact equality, not a tolerance.

    `vortex_viz._measure` ends in `np.asarray(jax.vmap(analyse)(...))`, so the
    Encounter path cannot be differentiated and `load_history` is a SECOND path
    to the same channel. If the two ever part, every elasticity taken through it
    is about a load this project does not report. A tolerance here would admit
    exactly that, so the claim is bit-identity.
    """
    from atisim import vortex_viz

    seconds = mehta["n_steps"] * mehta["dt"]
    enc = vortex_viz.fly_in_moving_air(
        mehta["ac"], mehta["field"], mehta["V"], mehta["H"], label="gate",
        start_north=mehta["start_north"], seconds=seconds, dt=mehta["dt"],
        window=(-1e12, 1e12), window_name="whole run")

    n_z, north = sensitivity.load_history(
        mehta["ac"], mehta["model"], mehta["V"], mehta["H"],
        start_north=mehta["start_north"], n_steps=mehta["n_steps"],
        dt=mehta["dt"])

    assert np.array_equal(np.asarray(n_z), enc.n_z), (
        f"worst |diff| {np.abs(np.asarray(n_z) - enc.n_z).max():.3e}")
    assert np.array_equal(np.asarray(north), enc.north)


def test_the_trim_primitive_agrees_with_differentiating_through_the_solver(mehta):
    """The implicit JVP against the unrolled Newton loop -- the design's own check.

    `solved_trim` attaches the implicit-function-theorem derivative by hand.
    Differentiating straight through `trim.trim`'s 40 unrolled Newton steps is
    the independent answer: it is what the naive thing would have done, it is
    correct once converged, and it costs 40 linear solves per tangent instead of
    one. They must agree.
    """
    ac, V, H = mehta["ac"], jnp.array(mehta["V"]), jnp.array(mehta["H"])
    zeros = jax.tree.map(jnp.zeros_like, ac)

    for field in ("CLa", "Cma", "CD0", "mass"):
        seed = zeros._replace(**{field: jnp.ones_like(getattr(ac, field))})
        _, implicit = jax.jvp(lambda a: sensitivity.solved_trim(V, H, a),
                              (ac,), (seed,))
        _, unrolled = jax.jvp(lambda a: trim.trim(V, H, a)[0], (ac,), (seed,))
        np.testing.assert_allclose(np.asarray(implicit), np.asarray(unrolled),
                                   rtol=1e-9, atol=1e-14,
                                   err_msg=f"{field}: implicit vs unrolled trim")


def test_the_trim_primitive_matches_the_jacobian_table(mehta):
    """`solved_trim`'s JVP and `implicit_trim_jacobian` are the same derivative.

    One is a differentiable function and the other is a table; they are built
    from the same two blocks and a divergence would mean one of them has the
    sign or the solve the wrong way round.
    """
    ac, V, H = mehta["ac"], jnp.array(mehta["V"]), jnp.array(mehta["H"])
    table = sensitivity.implicit_trim_jacobian(ac, mehta["V"], mehta["H"],
                                              fields=("CLa", "Cma"))
    zeros = jax.tree.map(jnp.zeros_like, ac)
    for field in ("CLa", "Cma"):
        seed = zeros._replace(**{field: jnp.ones_like(getattr(ac, field))})
        _, jvp = jax.jvp(lambda a: sensitivity.solved_trim(V, H, a), (ac,), (seed,))
        np.testing.assert_allclose(np.asarray(jvp), table[field],
                                   rtol=1e-12, atol=1e-15)


def test_a_lateral_derivative_cannot_reach_the_vortex_load(mehta):
    """EXACT zero, and it is a MEASUREMENT of ASSUMPTIONS E10 rather than a check.

    Mehta's array is a function of along-track distance alone, so no wing strip
    sees a different gust and no lateral coefficient can reach `n_z`. The screen
    reports these as exactly 0.000000 and the distinction from "small" is the
    whole point: a zero here means NOT USED, and the register's E10 row is the
    reason.
    """
    Q, grad, elas, _ = sensitivity.load_elasticities(
        mehta["ac"], mehta["model"], mehta["V"], mehta["H"],
        start_north=mehta["start_north"], n_steps=mehta["n_steps"],
        dt=mehta["dt"], window=(-1e12, 1e12),
        fields=sensitivity.LATERAL_FIELDS)
    worst = max(abs(v) for v in grad.values())
    assert worst == 0.0, f"a lateral derivative moved the vortex load by {worst:.3e}"


def test_the_load_elasticity_matches_a_central_difference(mehta):
    """S2's gate: the AD screen against a difference on the same quantity.

    Taken on a short run so the extremes cannot wander between the two
    evaluations -- `max - min` is differentiable almost everywhere, and this test
    is about the machinery, not about that caveat.
    `scripts/sensitivity_load.py` section C is where the caveat is measured.
    """
    window = (-1e12, 1e12)
    Q, grad, elas, extra = sensitivity.load_elasticities(
        mehta["ac"], mehta["model"], mehta["V"], mehta["H"],
        start_north=mehta["start_north"], n_steps=mehta["n_steps"],
        dt=mehta["dt"], window=window, fields=("CLa", "mass"))

    for field in ("CLa", "mass"):
        p0 = float(getattr(mehta["ac"], field))
        h = abs(p0) * 1e-6
        vals = []
        for s in (+1.0, -1.0):
            swept = mehta["ac"]._replace(**{field: jnp.array(p0 + s * h)})
            n_z, _ = sensitivity.load_history(
                swept, mehta["model"], mehta["V"], mehta["H"],
                start_north=mehta["start_north"], n_steps=mehta["n_steps"],
                dt=mehta["dt"])
            vals.append(float(sensitivity.peak_to_peak(
                n_z, jnp.asarray(extra["mask"]))))
        fd = (vals[0] - vals[1]) / (2.0 * h)
        assert grad[field] == pytest.approx(fd, rel=1e-5), (
            f"{field}: AD {grad[field]}, central difference {fd}")


def test_the_field_vector_round_trips():
    """`with_field_vector(ac, f, field_vector(ac, f))` must be the same aircraft."""
    ac = REGISTRY["boeing747"]
    fields = ("CLa", "Cma", "mass", "CD0")
    back = sensitivity.with_field_vector(ac, fields,
                                        sensitivity.field_vector(ac, fields))
    for f in fields:
        assert float(getattr(back, f)) == float(getattr(ac, f))


def test_peak_to_peak_respects_its_mask():
    """A sample outside the mask must not be able to set either extreme."""
    n_z = jnp.array([0.0, 5.0, 1.0, -3.0, 0.5])
    mask = jnp.array([False, False, True, False, True])
    assert float(sensitivity.peak_to_peak(n_z, mask)) == pytest.approx(0.5)
    allm = jnp.ones_like(mask, dtype=bool)
    assert float(sensitivity.peak_to_peak(n_z, allm)) == pytest.approx(8.0)
