"""Tier 1 and 2 validation: analytic laws, and published worked examples."""

import jax.numpy as jnp
import numpy as np
import pytest

import atisim  # noqa: F401
from atisim import trim, validation
from atisim.aircraft import REGISTRY
from atisim.units import FT2M

# Caughey Eq. (5.48): M = 0.25 at sea level. CR-2144 Table IX-2's header says
# 165 KTAS = 278.49 ft/s, a 0.2% difference recorded in the design spec. Checks
# against Caughey run at Caughey's speed.
CAUGHEY_V = 279.1 * FT2M


def _approach_trim(V=CAUGHEY_V):
    ac = REGISTRY["boeing747_approach"]
    x, res = trim.trim(jnp.array(V), jnp.array(0.0), ac)
    return ac, float(x[0]), float(x[1]), float(x[2]), res


def _approach_A(imperial=True):
    ac, alpha, de, thr, _ = _approach_trim()
    A = validation.to_stability_axes(
        validation.longitudinal_matrix(ac, alpha, de, thr, CAUGHEY_V, 0.0), alpha
    )
    return validation.to_imperial_matrix(A) if imperial else A


def _sp_wn(a, alpha, de, thr):
    return validation.longitudinal_modes(a, alpha, de, thr, CAUGHEY_V, 0.0)[-1][0]


def _ph_zeta(a, alpha, de, thr):
    return validation.longitudinal_modes(a, alpha, de, thr, CAUGHEY_V, 0.0)[0][1]


# --------------------------------------------------------------------------
# The axis transform, asserted before anything relies on it
# --------------------------------------------------------------------------


def test_the_stability_axis_transform_is_a_similarity_transform():
    """Every element moves; no eigenvalue does.

    This is what licenses comparing a body-axis matrix against a published
    stability-axis one, so it is asserted before it is used.
    """
    ac, alpha, de, thr, _ = _approach_trim()
    A_body = validation.longitudinal_matrix(ac, alpha, de, thr, CAUGHEY_V, 0.0)
    A_stab = validation.to_stability_axes(A_body, alpha)

    assert not np.allclose(A_body, A_stab, atol=1e-6), "the transform did nothing"
    np.testing.assert_allclose(
        np.sort_complex(np.linalg.eigvals(A_body)),
        np.sort_complex(np.linalg.eigvals(A_stab)),
        atol=1e-8,
    )


def test_every_reference_carries_its_citation():
    """PROJECT.md's first rule, enforced mechanically.

    'Flag, never invent. Every number carries the table it came from.' A
    reference without a source string is exactly the failure that rule exists to
    prevent, and it is cheap to make impossible.
    """
    assert validation.REFERENCES, "the reference table is empty"
    for name, ref in validation.REFERENCES.items():
        assert ref.source and len(ref.source) > 20, f"{name} has no usable citation"
        assert np.isfinite(ref.value), f"{name} has a non-finite value"


# --------------------------------------------------------------------------
# Tier 2: Caughey's published worked example
# --------------------------------------------------------------------------


def test_the_plant_matrix_matches_caugheys_where_the_model_has_the_terms():
    """Tier 2: same source data, an outside implementation, published intermediates.

    Covers ONLY the elements carrying no alpha-dot content. A[1,1], A[1,2], A[2,1]
    and A[2,2] all do, and are the next test's business -- asserting them here too
    would give the same element two homes and two different tolerances.

    Measured, after the stability-axis rotation:
        A[0,0] Xu      -0.02094 vs -0.02120   1.2%
        A[0,1] Xw       0.04632 vs  0.04660   0.6%
        A[1,0] Zu      -0.22851 vs -0.22290   2.5%
    """
    _, _, _, _, res = _approach_trim()
    assert float(jnp.linalg.norm(res)) < 1e-10
    A, C = _approach_A(), validation.CAUGHEY_A

    assert A[0, 0] == pytest.approx(C[0, 0], rel=0.02)
    assert A[0, 1] == pytest.approx(C[0, 1], rel=0.01)
    assert A[1, 0] == pytest.approx(C[1, 0], rel=0.03)

    # Kinematic and gravity elements are structural, so they are exact.
    assert A[0, 3] == pytest.approx(C[0, 3], rel=1e-4)
    assert A[1, 3] == pytest.approx(0.0, abs=1e-9)
    assert A[3, 2] == pytest.approx(1.0, abs=1e-12)


def test_the_omitted_alpha_dot_terms_are_recoverable():
    """Section 5's attribution, turned from a claim into arithmetic.

    Every element that disagrees by more than 3% carries a derivative this model
    excludes by design (aero.py is alpha/q/de only). Caughey's Eq. (5.44) gives
    the exact form of each:

        A[1,1] = Zw / (1 - Zwdot)
        A[1,2] = (u0 + Zq) / (1 - Zwdot)
        A[2,1] = Mw + Mwdot * Zw / (1 - Zwdot)  ->  A_model[2,1] + Mwdot * C[1,1]
        A[2,2] = Mq + (u0 + Zq) * Mwdot / (1 - Zwdot)  ->  A_model[2,2] + Mwdot * C[1,2]

    Restoring them from Caughey's own tabulated CL_alphadot = 6.7 and
    Cm_alphadot = -3.2 recovers his published values. That is a much stronger
    statement than "attributed": the residual is reconstructed, not just explained.

    Tolerances on the two M-row rows are looser because Mwdot is published to ONE
    significant figure (-0.0002), which bounds the reconstruction independently of
    anything this model does.
    """
    A, C = _approach_A(), validation.CAUGHEY_A
    Zwdot = validation.REFERENCES["747pa_Zwdot"].value
    Mwdot = validation.REFERENCES["747pa_Mwdot"].value

    assert A[1, 1] / (1.0 - Zwdot) == pytest.approx(C[1, 1], rel=0.005)
    assert A[1, 2] / (1.0 - Zwdot) == pytest.approx(C[1, 2], rel=0.005)

    # This model's A[2,2] IS Caughey's raw Eq. (5.51) Mq, before his alpha-dot term.
    assert A[2, 2] == pytest.approx(validation.REFERENCES["747pa_Mq"].value, rel=0.01)

    assert A[2, 1] + Mwdot * C[1, 1] == pytest.approx(C[2, 1], rel=0.05)
    assert A[2, 2] + Mwdot * C[1, 2] == pytest.approx(C[2, 2], rel=0.05)


def test_the_approach_modes_match_caugheys_published_roots():
    """The end-to-end statement: units, trim, dynamics and jacfwd in four numbers."""
    ac, alpha, de, thr, _ = _approach_trim()
    (ph_wn, ph_z), (sp_wn, sp_z) = validation.longitudinal_modes(
        ac, alpha, de, thr, CAUGHEY_V, 0.0
    )
    R = validation.REFERENCES
    assert ph_wn == pytest.approx(R["747pa_phugoid_wn"].value, rel=0.01)
    assert ph_z == pytest.approx(R["747pa_phugoid_zeta"].value, rel=0.05)
    assert sp_wn == pytest.approx(R["747pa_short_period_wn"].value, rel=0.02)
    assert sp_z == pytest.approx(R["747pa_short_period_zeta"].value, rel=0.06)


def test_the_mode_error_is_an_order_of_magnitude_smaller_on_approach():
    """The finding that reframes section 5.

    Section 4 records the cruise phugoid at 17.8% and short-period zeta at 11.5%
    from CR-2144, attributed to derivatives the aero form excludes. The same code
    with the same exclusions reads 0.4% and 5.5% at the power-approach point. So
    the gap is not a fixed modelling deficit -- it is CONDITION-DEPENDENT, and it
    bites at M 0.8 / 40,000 ft where compressibility drives the Mach content of
    Xu and Zu.

    Asserted as a RATIO of the two errors rather than a bound on one of them: a
    bound on the approach error alone is already implied by the tolerances in
    test_the_approach_modes_match_caugheys_published_roots, so it would assert
    nothing new.
    """
    R = validation.REFERENCES
    ac, alpha, de, thr, _ = _approach_trim()
    (ph_wn, _), (_, sp_z) = validation.longitudinal_modes(
        ac, alpha, de, thr, CAUGHEY_V, 0.0
    )

    def rel_err(model, ref):
        return abs(model - ref) / abs(ref)

    approach_ph = rel_err(ph_wn, R["747pa_phugoid_wn"].value)
    approach_sp = rel_err(sp_z, R["747pa_short_period_zeta"].value)
    cruise_ph = rel_err(
        R["747cruise_phugoid_wn_model"].value, R["747cruise_phugoid_wn_ref"].value
    )
    cruise_sp = rel_err(
        R["747cruise_short_period_zeta_model"].value,
        R["747cruise_short_period_zeta_ref"].value,
    )

    assert cruise_ph / approach_ph > 10.0, (
        f"phugoid: cruise {cruise_ph:.4f} vs approach {approach_ph:.4f}"
    )
    assert cruise_sp / approach_sp > 1.5, (
        f"short-period zeta: cruise {cruise_sp:.4f} vs approach {approach_sp:.4f}"
    )


# --------------------------------------------------------------------------
# Tier 1: coefficient sweeps against analytic laws
# --------------------------------------------------------------------------


def test_the_sweep_helper_refuses_a_physically_absurd_trim():
    """A sweep sample that silently trims to nonsense reports modes for something
    that is not a flight condition, which looks like a physics finding and is not.

    The interesting part is that the RESIDUAL does not catch this. At CLa = 0.1
    the Newton solve converges to a residual of 1.6e-15 -- at alpha = -633
    degrees, because CL = CL0 + CLa*alpha is linear and a huge alpha compensates a
    small CLa. Convergence and sense are different questions, so `sweep` asks both.
    """
    ac, _, _, _, _ = _approach_trim()
    swept = ac._replace(CLa=jnp.array(0.1))
    _, residual = trim.trim(jnp.array(CAUGHEY_V), jnp.array(0.0), swept)
    assert float(jnp.linalg.norm(residual)) < validation.TRIM_RESIDUAL_LIMIT, (
        "this case is only interesting because it DOES converge"
    )

    with pytest.raises(RuntimeError, match="linear-aero range"):
        validation.sweep(ac, "CLa", [0.1], lambda a, al, de, th: 0.0, CAUGHEY_V, 0.0)


def test_the_phugoid_frequency_follows_the_lanchester_law():
    """Tier 1: a relation with no aerodynamic coefficient in it at all.

    Lanchester's phugoid approximation is wn = sqrt(2) g / u0 -- no derivative, no
    area, no mass. It cannot be satisfied by accident, and no source's vintage can
    affect it.

    Caughey states his approximate analysis "over predicts the undamped natural
    frequency by about 20 per cent". Measured: 0.163028 against 0.13391, a ratio
    of 1.2174. Matching the published SIZE of the approximation's error is far
    tighter than matching its trend.
    """
    from atisim.atmosphere import G0

    ac, alpha, de, thr, _ = _approach_trim()
    (ph_wn, _), _ = validation.longitudinal_modes(ac, alpha, de, thr, CAUGHEY_V, 0.0)
    lanchester = np.sqrt(2.0) * float(G0) / CAUGHEY_V
    assert lanchester / ph_wn == pytest.approx(1.22, rel=0.03)


def test_the_phugoid_damping_follows_the_lift_to_drag_law():
    """Lanchester again: zeta = 1 / (sqrt(2) L/D).

    Caughey states the approximation "over predicts the damping ratio by a factor
    of almost 5". Measured: 0.0651 against 0.01329, a ratio of 4.9.
    """
    ac, alpha, de, thr, _ = _approach_trim()
    (_, ph_zeta), _ = validation.longitudinal_modes(ac, alpha, de, thr, CAUGHEY_V, 0.0)
    R = validation.REFERENCES
    L_over_D = R["747pa_CL"].value / R["747pa_CD"].value
    lanchester = 1.0 / (np.sqrt(2.0) * L_over_D)
    assert lanchester / ph_zeta == pytest.approx(4.9, rel=0.06)


def test_more_drag_damps_the_phugoid_linearly_in_CD0():
    """The sweep itself: change one coefficient, get the predicted change.

    The textbook relation is zeta = 1/(sqrt(2) L/D), which at fixed CL makes zeta
    LINEAR in CD0 with slope 1/(sqrt(2) CL) = 0.637. Measured: linear to 0.31%
    over a 3x range, slope 0.770.

    So the FORM of the law holds tightly and the COEFFICIENT is 21% high. Both
    halves are the finding. Note this is a different statement from
    test_the_phugoid_damping_follows_the_lift_to_drag_law, which compares the
    absolute value at one point and reproduces Caughey's factor-of-5.
    """
    ac, _, _, _, _ = _approach_trim()
    base = float(ac.CD0)
    cd0s = base * np.array([1.0, 1.5, 2.0, 3.0])

    zetas = validation.sweep(ac, "CD0", cd0s, _ph_zeta, CAUGHEY_V, 0.0)
    assert np.all(np.diff(zetas) > 0), "more drag must damp the phugoid"

    slope, _, worst = validation.affine_fit(cd0s, zetas)
    assert worst < 0.01, f"zeta is not linear in CD0: worst residual {worst}"
    theory = 1.0 / (np.sqrt(2.0) * validation.REFERENCES["747pa_CL"].value)
    assert slope / theory == pytest.approx(1.21, rel=0.05)


def test_the_short_period_frequency_is_affine_in_pitch_stiffness():
    """wn_sp^2 = Z_alpha M_q/u0 - M_alpha, so wn^2 is AFFINE in Cm_alpha.

    Not proportional to -Cm_alpha, and the difference matters: the intercept is
    the Z_alpha M_q/u0 term, which does NOT vanish at the neutral point. Measured
    over Cm_alpha in [-1.26, -0.1]: affine to 1.68%, slope -0.4225, intercept
    +0.2737.

    An earlier version of this test asserted wn -> 0 as Cm_alpha -> 0. That was
    wrong physics, and the model was right: a short period whose frequency
    vanished at the neutral point would be missing the pitch-damping term.
    """
    ac, _, _, _, _ = _approach_trim()
    base = validation.REFERENCES["747pa_Cma"].value
    cmas = np.array([base, -0.9, -0.6, -0.3, -0.1])

    wns = validation.sweep(ac, "Cma", cmas, _sp_wn, CAUGHEY_V, 0.0)
    assert np.all(np.diff(wns) < 0), f"wn must fall as Cma -> 0, got {wns}"

    slope, intercept, worst = validation.affine_fit(cmas, wns**2)
    assert worst < 0.03, f"wn^2 is not affine in Cma: worst residual {worst}"
    assert slope < 0.0
    assert intercept > 0.2, (
        f"intercept {intercept} should carry Z_alpha M_q/u0, not vanish"
    )


def test_the_model_goes_statically_unstable_exactly_at_zero_pitch_stiffness():
    """The neutral point, which is where the sharpest check in flight dynamics is.

    Cm_alpha = 0 IS the definition of the neutral point, so the longitudinal
    system must be neutrally stable there and divergent beyond. Measured: the
    largest real part is 0.00000 at Cm_alpha = 0 and +0.0475 at +0.1. Nothing was
    tuned to make that land on zero -- it falls out of the derivative chain.
    """
    ac, _, _, _, _ = _approach_trim()

    def max_real_root(cma):
        swept = ac._replace(Cma=jnp.array(float(cma)))
        x, _ = trim.trim(jnp.array(CAUGHEY_V), jnp.array(0.0), swept)
        A = validation.longitudinal_matrix(
            swept, float(x[0]), float(x[1]), float(x[2]), CAUGHEY_V, 0.0
        )
        return max(lam.real for lam in np.linalg.eigvals(A))

    assert max_real_root(-0.1) < 0.0, "should still be stable inside the neutral point"
    assert max_real_root(0.0) == pytest.approx(0.0, abs=1e-6), "neutral point"
    assert max_real_root(0.1) > 0.01, "should diverge past the neutral point"


def test_the_roll_rate_root_is_affine_in_roll_damping():
    """tau_roll = -1/L_p is an approximation; 1/tau affine in |Clp| is the truth.

    L_p is proportional to Clp, so the approximation makes 1/tau PROPORTIONAL to
    |Clp| with no intercept. Measured: affine to 1.73%, slope 2.069, and an
    intercept of +0.302 that the approximation has no room for.

    The intercept is roll-yaw coupling. The 747's Ixz is 870,050 slug-ft^2 and
    PROJECT.md section 2 already notes it "is not negligible for the 747", so the
    roll root is not the pure -L_p the two-term approximation gives.
    """
    ac, _, _, _, _ = _approach_trim()
    clps = np.array([-0.30, -0.45, -0.60, -0.75])

    def roll_tau(a, alpha, de, thr):
        return validation.lateral_modes(a, alpha, de, thr, CAUGHEY_V, 0.0)[1]

    taus = validation.sweep(ac, "Clp", clps, roll_tau, CAUGHEY_V, 0.0)
    assert np.all(np.diff(taus) < 0), "more roll damping must shorten tau"

    slope, intercept, worst = validation.affine_fit(np.abs(clps), 1.0 / taus)
    assert worst < 0.03, f"1/tau is not affine in |Clp|: worst residual {worst}"
    assert slope > 0.0
    assert intercept > 0.1, f"intercept {intercept} is the Ixz coupling"


def test_the_dutch_roll_frequency_is_affine_in_weathercock_stability():
    """wn_dr^2 is AFFINE in Cnb, not proportional to it.

    The square-root law wn ~ sqrt(N_beta) would make wn^2 proportional to Cnb with
    no intercept, so wn/sqrt(Cnb) would be flat. It is not: measured 2.385, 1.969,
    1.727, 1.597 across a 8x range. What IS flat is the slope of wn^2 against Cnb
    -- affine to 0.56%, slope 2.105, intercept +0.266.

    The intercept is the Y_beta/u0 side-force term, which does not scale with Cnb.
    Asserting the square-root law would have been asserting the approximation
    rather than the model.
    """
    ac, _, _, _, _ = _approach_trim()
    cnbs = np.array([0.075, 0.150, 0.300, 0.600])

    def dr_wn(a, alpha, de, thr):
        return validation.lateral_modes(a, alpha, de, thr, CAUGHEY_V, 0.0)[0][0]

    wns = validation.sweep(ac, "Cnb", cnbs, dr_wn, CAUGHEY_V, 0.0)
    assert np.all(np.diff(wns) > 0), f"wn must rise with Cnb, got {wns}"

    slope, intercept, worst = validation.affine_fit(cnbs, wns**2)
    assert worst < 0.02, f"wn^2 is not affine in Cnb: worst residual {worst}"
    assert slope == pytest.approx(2.105, rel=0.05)
    assert intercept > 0.1, f"intercept {intercept} is the Y_beta/u0 term"
