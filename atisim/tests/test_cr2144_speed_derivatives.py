"""CR-2144's 747 Mach derivatives: the engine seam, the digitisation, the retest.

Three kinds of test, in the order a reader should trust them.

VERIFICATION. `Aircraft.mach_deriv_ref / CL_M / CD_M / Cm_M` is new physics in
`aero.coefficients`. Undeclared it must move nothing; declared it must be exactly
the first-order increment it claims; and through the engine's own jacfwd
linearisation it must reproduce CR-2144 Appendix A's (M/2) C_XM, C_NM, C_mM terms
(printed pp. A-16, A-17) to round-off. None of this depends on any digitised
number.

THE DIGITISATION, against the source's own tables. CR-2144 printed pp. 220-222
were read by hand; Table IX-4 (printed p. 230) independently implies a value at
every circled flight condition. Asserted as bands.

THE RETEST. The FC9 set declared on a COPY of the 747, re-trimmed and linearised,
against Table IX-5 (printed p. 231). Asserted as bands and orderings, per
CLAUDE.md rule 6. The registry entry itself does not declare the set; see the
last verification test.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from atisim import aero
from atisim import cr2144_mach as cm
from atisim.aircraft import CRUISE, REGISTRY
from atisim.atmosphere import density, speed_of_sound
from atisim.state import Controls
from atisim.trim import trim
from atisim.validation import longitudinal_matrix

AC = REGISTRY["boeing747"]
V, H = CRUISE["boeing747"]["airspeed"], CRUISE["boeing747"]["altitude"]
CONTROLS = Controls(elevator=jnp.array(0.01), aileron=jnp.array(0.02),
                    rudder=jnp.array(-0.01), throttle=jnp.array(0.5))
OMEGA = jnp.array([0.01, -0.02, 0.005])
# Magnitudes of the FC9 set, used where only the arithmetic is under test.
SET = dict(CL_M=jnp.array(0.13), CD_M=jnp.array(0.025), Cm_M=jnp.array(0.175))


def _coefficients(ac, vel, a_sound):
    return np.array([float(x) for x in aero.coefficients(vel, OMEGA, CONTROLS, ac, a_sound)])


# ===========================================================================
# 1. Verification of the seam
# ===========================================================================

@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_undeclared_mach_derivatives_add_an_exact_zero(name):
    """A negative reference Mach switches the terms off whatever the
    coefficients hold. That the SHIPPED path is also unmoved from before the
    seam existed is what the frozen bit-pins assert --
    test_vortex_viz.FIG8_VORTEX and test_verification's single-bit test."""
    ac = REGISTRY[name]
    a_sound = speed_of_sound(jnp.array(CRUISE[name]["altitude"]))
    vel = jnp.array([CRUISE[name]["airspeed"] * 1.07, 3.0, 9.0])
    loaded = ac._replace(CL_M=jnp.array(0.3), CD_M=jnp.array(0.05), Cm_M=jnp.array(-0.2))
    assert float(loaded.mach_deriv_ref) < 0.0
    assert np.array_equal(_coefficients(loaded, vel, a_sound), _coefficients(ac, vel, a_sound))


def test_a_declared_set_is_an_exact_first_order_increment_in_mach():
    """Zero at the reference Mach, C_M * dM away from it, and in CL, CD and Cm
    only -- at the same alpha, beta and rates, so nothing else can differ."""
    a_sound = speed_of_sound(jnp.array(H))
    vel_ref = jnp.array([V, 0.0, 0.08 * V])
    m_ref = float(jnp.linalg.norm(vel_ref) / a_sound)
    declared = AC._replace(mach_deriv_ref=jnp.array(m_ref), **SET)

    assert _coefficients(declared, vel_ref, a_sound) == pytest.approx(
        _coefficients(AC, vel_ref, a_sound), rel=1e-12)

    vel = vel_ref * 1.05
    dM = float(jnp.linalg.norm(vel) / a_sound) - m_ref
    got = _coefficients(declared, vel, a_sound) - _coefficients(AC, vel, a_sound)
    CL_M, CD_M, Cm_M = (float(SET[k]) for k in ("CL_M", "CD_M", "Cm_M"))
    # (CL, CD, CY, Cl, Cm, Cn)
    assert got == pytest.approx([CL_M * dM, CD_M * dM, 0.0, 0.0, Cm_M * dM, 0.0], abs=1e-12)


def test_the_increment_does_not_reach_induced_or_wave_drag():
    """CR-2144's C_DM is the TOTAL drag slope. A lift increment feeding CL**2
    would add 2 CL CL_M/(pi e AR) to it -- about 0.010 on the 747 against a
    sourced 0.025 -- so declaring CL_M alone must leave CD exactly unmoved."""
    a_sound = speed_of_sound(jnp.array(H))
    vel = jnp.array([V * 1.05, 0.0, 0.08 * V * 1.05])
    lift_only = AC._replace(mach_deriv_ref=jnp.array(0.80), CL_M=SET["CL_M"])
    assert _coefficients(lift_only, vel, a_sound)[1] == _coefficients(AC, vel, a_sound)[1]


def test_the_engine_linearisation_carries_appendix_a_mach_terms_exactly():
    """THE VERIFICATION THAT MATTERS. Declared at the trim Mach, the trim is
    untouched, so the difference between the two jacfwd plant matrices is the
    Mach content and nothing else -- and it must equal CR-2144 Appendix A's
    terms, built independently in `cr2144_mach.mach_increment`, in all six
    elements they occupy and zero elsewhere."""
    x, _ = trim(jnp.array(V), jnp.array(H), AC)
    alpha, de, th = (float(v) for v in x)
    a_s = float(speed_of_sound(jnp.array(H)))
    declared = AC._replace(mach_deriv_ref=jnp.array(V / a_s), **SET)
    x2, _ = trim(jnp.array(V), jnp.array(H), declared)
    assert np.allclose(np.asarray(x2), np.asarray(x), rtol=0.0, atol=1e-10)

    dA = (longitudinal_matrix(declared, alpha, de, th, V, H)
          - longitudinal_matrix(AC, alpha, de, th, V, H))
    expected = cm.mach_increment(
        alpha, V, float(density(jnp.array(H))), a_s, float(AC.mass),
        float(np.asarray(AC.inertia)[1, 1]), float(AC.S), float(AC.c),
        *(float(SET[k]) for k in ("CL_M", "CD_M", "Cm_M")))
    assert np.count_nonzero(expected) == 6
    assert np.allclose(dA, expected, rtol=1e-9, atol=1e-15), (dA, expected)


def test_no_registry_entry_declares_mach_derivatives_yet():
    """A DECISION, pinned so that taking it is visible. Declaring the digitised
    FC9 set on `boeing747` moves the shipped phugoid, every Fig. 8 pin and the
    CAT headline, and needs the thrust-moment and drag-rise caveats PROJECT.md
    section 4 records. If you declare it, re-capture those deliberately, record
    what moved, and replace this test with one that pins the declared values."""
    for name, ac in REGISTRY.items():
        assert float(ac.mach_deriv_ref) < 0.0, name


# ===========================================================================
# 2. The digitisation, against CR-2144's own tables
# ===========================================================================

def test_only_hand_placed_points_are_carried_and_nothing_is_extrapolated():
    """The Engauge CSV exports read 105 for Cm_M and -47 for CL_M because they
    extrapolate. The tracked table must hold the placed points only: 295 of
    them, CL_M's 40,000 ft curve without the nine SL points copied into it
    (so starting above M 0.65, where no 40,000 ft condition could fly), and
    NaN for any reading outside a curve's own span."""
    curves = cm.curves()
    assert sum(c.mach.size for c in curves.values()) == 295
    assert set(q for q, _ in curves) == set(cm.QUANTITIES)
    for c in curves.values():
        assert 0.0 < c.mach[0] and c.mach[-1] < 1.0 and np.all(np.diff(c.mach) > 0.0)
    assert curves[("cl_m", "40K")].mach[0] > 0.65
    assert ("cd_m", "SL") not in curves  # p. 222 draws no SL drag-rise curve
    c = curves[("cm_m", "40K")]
    assert np.isnan(cm.value("cm_m", "40K", c.mach[0] - 1e-3))
    assert np.isnan(cm.value("cm_m", "40K", c.mach[-1] + 1e-3))


# Measured session 30 (scripts/cr2144_speed_derivatives.py, section 3): the
# worst |digitised - Table IX-4 implied| / |table| over every circled condition
# the curve spans. The bands sit above those, not at them.
SMOOTH_CURVE_BAND = {          # measured worst
    "cm_q": 0.015,             # 0.9%, 8 conditions
    "cm_alpha": 0.015,         # 0.7%, 6 conditions
    "cl_alpha": 0.025,         # 1.5%, 8 conditions
    "cm_alpha_dot": 0.06,      # 4.3%, 8 conditions
}


def _residuals(quantity, thrust=True):
    out = []
    for fc, (alt, mach, *_rest) in cm.IX3.items():
        dig = cm.value(quantity, alt, mach)
        if np.isfinite(dig):
            table = cm.backsolve(fc, thrust=thrust)[quantity]
            out.append((fc, dig - table, table))
    return out


@pytest.mark.parametrize("quantity,band", sorted(SMOOTH_CURVE_BAND.items()))
def test_the_smooth_curves_agree_with_table_ix4_at_every_circled_condition(quantity, band):
    rows = _residuals(quantity)
    assert len(rows) >= 6
    worst = max(abs(r) / abs(t) for _, r, t in rows)
    assert worst < band, [(fc, round(r, 4)) for fc, r, _ in rows]


def test_cm_M_agrees_with_table_ix4_only_once_the_thrust_moment_is_trimmed():
    """The finding that fixes how Appendix A is read. Table IX-3 puts the thrust
    line 10 ft from the CG, so trim C_m is not zero and M_u carries it. With it,
    seven circled points agree to 0.011 (RMS 0.0060); without it, RMS 0.0382.
    An ordering by a factor of four, not a value."""
    with_thrust = np.array([r for _, r, _ in _residuals("cm_m", thrust=True)])
    without = np.array([r for _, r, _ in _residuals("cm_m", thrust=False)])
    assert with_thrust.size == 7
    assert np.abs(with_thrust).max() < 0.015
    assert 4.0 * np.sqrt(np.mean(with_thrust**2)) < np.sqrt(np.mean(without**2))


# ===========================================================================
# 3. The retest: the FC9 set on a copy of the 747, against Table IX-5
# ===========================================================================

def _korn_lock_slope(ac, alpha, mach):
    import jax

    CL = float(ac.CL0) + float(ac.CLa) * alpha
    return float(jax.grad(lambda m: aero.wave_drag(m, jnp.array(CL), ac))(jnp.array(mach)))


def _errors(ac):
    x, _ = trim(jnp.array(V), jnp.array(H), ac)
    alpha, de, th = (float(v) for v in x)
    return cm.errors_vs_ix5(cm.modes(longitudinal_matrix(ac, alpha, de, th, V, H)))


def _declared(CL_M, CD_M_total, Cm_M):
    x, _ = trim(jnp.array(V), jnp.array(H), AC)
    slope = _korn_lock_slope(AC, float(x[0]), V / float(speed_of_sound(jnp.array(H))))
    return AC._replace(mach_deriv_ref=jnp.array(0.800), CL_M=jnp.array(CL_M),
                       CD_M=jnp.array(CD_M_total - slope), Cm_M=jnp.array(Cm_M))


@pytest.fixture(scope="module")
def retest():
    d = {q: cm.value(q, "40K", 0.800) for q in ("cl_m", "cd_m", "cm_m")}
    b9 = cm.backsolve(9)
    return dict(
        shipped=_errors(AC),
        sourced=_errors(_declared(d["cl_m"], d["cd_m"], d["cm_m"])),
        korn_lock_kept=_errors(AC._replace(mach_deriv_ref=jnp.array(0.800),
                                           CL_M=jnp.array(d["cl_m"]), Cm_M=jnp.array(d["cm_m"]))),
        table=_errors(_declared(b9["cl_m"], b9["cd_m"], b9["cm_m"])),
        thrust_compensated=_errors(_declared(d["cl_m"], d["cd_m"],
                                             d["cm_m"] + b9["Cm_trim"] / 0.400)),
    )


def test_the_digitised_speed_derivatives_close_most_of_the_phugoid_gap(retest):
    """THE RETEST. Shipped: phugoid wn -18.1%, zeta +13.2%. With the digitised
    FC9 CL_M, Cm_M and a sourced TOTAL drag Mach slope: +4.05% and +4.55%,
    measured session 30. Bands, not values: both errors are now under a
    third of their shipped size, and the frequency error has changed sign --
    it overshoots, which the next two tests explain."""
    s, e = retest["shipped"], retest["sourced"]
    assert s["ph_wn"] < -0.15 and s["ph_z"] > 0.10
    assert 0.02 < e["ph_wn"] < 0.07
    assert 0.01 < e["ph_z"] < 0.08
    assert abs(e["ph_wn"]) < abs(s["ph_wn"]) / 3.0 and abs(e["ph_z"]) < abs(s["ph_z"]) / 2.0


def test_the_speed_derivatives_leave_the_short_period_alone(retest):
    """Speed derivatives are a phugoid effect. Measured: -1.17% -> -1.34% and
    -11.35% -> -11.50%, both under 0.2 points. The short-period damping gap is
    the alpha-dot family's, as test_audit_regression already attributes."""
    for k in ("sp_wn", "sp_z"):
        assert abs(retest["sourced"][k] - retest["shipped"][k]) < 0.005, k


def test_the_overshoot_is_not_the_reading_it_is_the_missing_thrust_moment(retest):
    """Two statements. Table IX-4's OWN implied set overshoots the same way
    (+3.86% / +5.26% against the digitised +4.05% / +4.55%), so the residual is
    not digitisation error. And adding back the trim C_m the engine cannot have
    -- it places thrust through the CG -- takes both to under 1% (+0.07% /
    +0.64%). DIAGNOSTIC ONLY: the compensated Cm_M is not a sourced number."""
    e, t, c = retest["sourced"], retest["table"], retest["thrust_compensated"]
    assert abs(e["ph_wn"] - t["ph_wn"]) < 0.005 and abs(e["ph_z"] - t["ph_z"]) < 0.015
    assert abs(c["ph_wn"]) < 0.01 and abs(c["ph_z"]) < 0.015


def test_keeping_the_models_own_drag_rise_makes_phugoid_damping_worse(retest):
    """The partial-correction trap, and why CD_M is declared net of Korn/Lock.
    The engine's wave-drag slope at FC9 is 0.0477 per Mach against a sourced
    total of 0.025-0.028. Adding CL_M and Cm_M on top of it takes phugoid
    damping from +13.2% to +21.3% -- worse than shipping nothing."""
    assert retest["korn_lock_kept"]["ph_z"] > retest["shipped"]["ph_z"]
    # The drag slope does not touch the frequency. Not to 1e-6 though: the set
    # is declared at the TABLE's M 0.800 and the engine trims at M 0.7995, so a
    # different CD_M shifts the trim throttle slightly -- measured 1.7e-6.
    assert retest["korn_lock_kept"]["ph_wn"] == pytest.approx(retest["sourced"]["ph_wn"], abs=1e-4)


def test_the_improvement_survives_the_reading_uncertainty():
    """PRICED, not assumed. Every hand-placed point on the three 40,000 ft
    speed-derivative curves re-read with pixel scatter and an axis-calibration
    error on the sheets' own scales (`cr2144_mach.perturbed`). Session 30's
    N = 4000 run gave phugoid wn 5-95% of [+1.1, +6.9]% at 1 px and
    [-5.7, +9.5]% at 3 px, against -18.1% shipped. Checked here at N = 400."""
    x, _ = trim(jnp.array(V), jnp.array(H), AC)
    alpha, de, th = (float(v) for v in x)
    A0 = longitudinal_matrix(AC, alpha, de, th, V, H)
    a_s = float(speed_of_sound(jnp.array(H)))
    slope = _korn_lock_slope(AC, alpha, V / a_s)
    rho, Iyy = float(density(jnp.array(H))), float(np.asarray(AC.inertia)[1, 1])
    rng = np.random.default_rng(20260915)
    for sigma, lo, hi in ((1.0, -0.01, 0.09), (3.0, -0.12, 0.12)):
        wn = []
        for _ in range(400):
            v = [cm.value(q, "40K", 0.800,
                          curve=cm.perturbed(cm.curves()[(q, "40K")], rng, sigma))
                 for q in ("cl_m", "cd_m", "cm_m")]
            dA = cm.mach_increment(alpha, V, rho, a_s, float(AC.mass), Iyy, float(AC.S),
                                   float(AC.c), v[0], v[1] - slope, v[2])
            wn.append(cm.errors_vs_ix5(cm.modes(A0 + dA))["ph_wn"])
        p5, p95 = np.percentile(wn, [5, 95])
        assert lo < p5 and p95 < hi, (sigma, p5, p95)
