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
