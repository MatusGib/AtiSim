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

THE RETEST. `boeing747` DECLARES the FC9 set and CR-2144's thrust line, so the
comparisons below are against `BARE` -- the entry without the line and with the
seam shut, which is the 747 as it was before session 30 -- and the speed set as
first declared, without the line. Bands and orderings, per CLAUDE.md rule 6.
"""

import math

import jax.numpy as jnp
import numpy as np
import pytest

from atisim import aero
from atisim import cr2144_mach as cm
from atisim.aircraft import CRUISE, REGISTRY, boeing747_without_thrust_line
from atisim.atmosphere import density, speed_of_sound
from atisim.state import Controls
from atisim.trim import trim
from atisim.validation import longitudinal_matrix

AC = REGISTRY["boeing747"]
# The entry with the seam shut and no thrust line: the 747 as it stood before
# session 30 declared CR-2144's Mach derivatives and thrust line on it. Every
# "what did declaring do" comparison here is against THIS, not against a
# different aeroplane.
BARE = boeing747_without_thrust_line()._replace(mach_deriv_ref=jnp.array(-1.0), CL_M=jnp.array(0.0),
                   CD_M=jnp.array(0.0), Cm_M=jnp.array(0.0))
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
    coefficients hold. That the UNDECLARED path is also unmoved from before the
    seam existed is what the frozen bit-pins assert --
    test_vortex_viz.FIG8_VORTEX and test_verification's single-bit test."""
    ac = REGISTRY[name]._replace(mach_deriv_ref=jnp.array(-1.0))
    a_sound = speed_of_sound(jnp.array(CRUISE[name]["altitude"]))
    vel = jnp.array([CRUISE[name]["airspeed"] * 1.07, 3.0, 9.0])
    loaded = ac._replace(CL_M=jnp.array(0.3), CD_M=jnp.array(0.05), Cm_M=jnp.array(-0.2))
    assert np.array_equal(_coefficients(loaded, vel, a_sound), _coefficients(ac, vel, a_sound))


def test_a_declared_set_is_an_exact_first_order_increment_in_mach():
    """Zero at the reference Mach, C_M * dM away from it, and in CL, CD and Cm
    only -- at the same alpha, beta and rates, so nothing else can differ."""
    a_sound = speed_of_sound(jnp.array(H))
    vel_ref = jnp.array([V, 0.0, 0.08 * V])
    m_ref = float(jnp.linalg.norm(vel_ref) / a_sound)
    declared = BARE._replace(mach_deriv_ref=jnp.array(m_ref), **SET)

    assert _coefficients(declared, vel_ref, a_sound) == pytest.approx(
        _coefficients(BARE, vel_ref, a_sound), rel=1e-12)

    vel = vel_ref * 1.05
    dM = float(jnp.linalg.norm(vel) / a_sound) - m_ref
    got = _coefficients(declared, vel, a_sound) - _coefficients(BARE, vel, a_sound)
    CL_M, CD_M, Cm_M = (float(SET[k]) for k in ("CL_M", "CD_M", "Cm_M"))
    # (CL, CD, CY, Cl, Cm, Cn)
    assert got == pytest.approx([CL_M * dM, CD_M * dM, 0.0, 0.0, Cm_M * dM, 0.0], abs=1e-12)


def test_the_increment_does_not_reach_induced_or_wave_drag():
    """CR-2144's C_DM is the TOTAL drag slope. A lift increment feeding CL**2
    would add 2 CL CL_M/(pi e AR) to it -- about 0.010 on the 747 against a
    sourced 0.025 -- so declaring CL_M alone must leave CD exactly unmoved."""
    a_sound = speed_of_sound(jnp.array(H))
    vel = jnp.array([V * 1.05, 0.0, 0.08 * V * 1.05])
    lift_only = BARE._replace(mach_deriv_ref=jnp.array(0.80), CL_M=SET["CL_M"])
    assert _coefficients(lift_only, vel, a_sound)[1] == _coefficients(BARE, vel, a_sound)[1]


def test_the_engine_linearisation_carries_appendix_a_mach_terms_exactly():
    """THE VERIFICATION THAT MATTERS. Declared at the trim Mach, the trim is
    untouched, so the difference between the two jacfwd plant matrices is the
    Mach content and nothing else -- and it must equal CR-2144 Appendix A's
    terms, built independently in `cr2144_mach.mach_increment`, in all six
    elements they occupy and zero elsewhere."""
    x, _ = trim(jnp.array(V), jnp.array(H), BARE)
    alpha, de, th = (float(v) for v in x)
    a_s = float(speed_of_sound(jnp.array(H)))
    declared = BARE._replace(mach_deriv_ref=jnp.array(V / a_s), **SET)
    x2, _ = trim(jnp.array(V), jnp.array(H), declared)
    assert np.allclose(np.asarray(x2), np.asarray(x), rtol=0.0, atol=1e-10)

    dA = (longitudinal_matrix(declared, alpha, de, th, V, H)
          - longitudinal_matrix(BARE, alpha, de, th, V, H))
    expected = cm.mach_increment(
        alpha, V, float(density(jnp.array(H))), a_s, float(BARE.mass),
        float(np.asarray(BARE.inertia)[1, 1]), float(BARE.S), float(BARE.c),
        *(float(SET[k]) for k in ("CL_M", "CD_M", "Cm_M")))
    assert np.count_nonzero(expected) == 6
    assert np.allclose(dA, expected, rtol=1e-9, atol=1e-15), (dA, expected)


def test_the_747_declares_the_digitised_fc9_set_and_nothing_else_does():
    """THE DECISION, TAKEN. `boeing747` declares the set; every other entry
    leaves the seam shut.

    Two things are pinned. The reference Mach and the two directly SOURCED
    values are the digitised ones. And `CD_M` is DERIVED, not sourced: the field
    is the sourced total minus this model's own Korn/Lock slope, so what must be
    right is the TOTAL, asserted against the sourced value rather than against
    the field.
    """
    assert float(AC.mach_deriv_ref) == pytest.approx(0.800, abs=1e-12)
    assert float(AC.CL_M) == pytest.approx(cm.value("cl_m", "40K", 0.800), abs=5e-5)
    assert float(AC.Cm_M) == pytest.approx(cm.value("cm_m", "40K", 0.800), abs=5e-5)

    # The Korn/Lock slope is lift-dependent, so it is taken at the TRIM CL the
    # entry was built on, not at alpha = 0 where CL is just CL0. Since the thrust
    # line was declared that is CR-2144's AERODYNAMIC trim lift -- weight less the
    # thrust line's share -- not W/qS: at W/qS the total reads 0.0284 against the
    # sourced 0.0251, which is this test being out of date, not the entry.
    _, _, _, q, adeg = cm.IX3[9]
    a_xi = math.radians(adeg) + cm.XI_RAD
    thrust = cm.backsolve(9)["CD"] * q * cm.S_FT2 / math.cos(a_xi)
    CL_trim = (cm.W_LB - thrust * math.sin(a_xi)) / (q * cm.S_FT2)
    m_crit = float(aero.drag_divergence_mach(jnp.array(CL_trim), AC)) - aero._MDD_OFFSET
    total = float(AC.CD_M) + 80.0 * max(0.800 - m_crit, 0.0) ** 3
    assert total == pytest.approx(cm.value("cd_m", "40K", 0.800), abs=2e-3)
    assert float(AC.CD_M) < 0.0, "the net field is negative: Korn/Lock is the steeper of the two"

    for name, ac in REGISTRY.items():
        if name != "boeing747":
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
# 3. The retest: what declaring the set did, against Table IX-5
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
    """A copy carrying a sourced TOTAL drag Mach slope, net of Korn/Lock at the
    bare entry's own trim. The registry entry does the same subtraction from its
    own construction, so the two differ slightly and the test below says so."""
    x, _ = trim(jnp.array(V), jnp.array(H), BARE)
    slope = _korn_lock_slope(BARE, float(x[0]), V / float(speed_of_sound(jnp.array(H))))
    return BARE._replace(mach_deriv_ref=jnp.array(0.800), CL_M=jnp.array(CL_M),
                         CD_M=jnp.array(CD_M_total - slope), Cm_M=jnp.array(Cm_M))


@pytest.fixture(scope="module")
def retest():
    d = {q: cm.value(q, "40K", 0.800) for q in ("cl_m", "cd_m", "cm_m")}
    b9 = cm.backsolve(9)
    return dict(
        bare=_errors(BARE),
        # the speed set as declared first, thrust through the CG (commit 0d84eae)
        speed_only=_errors(boeing747_without_thrust_line()),
        shipped=_errors(AC),
        line_only=_errors(AC._replace(mach_deriv_ref=jnp.array(-1.0))),
        copy=_errors(_declared(d["cl_m"], d["cd_m"], d["cm_m"])),
        korn_lock_kept=_errors(BARE._replace(mach_deriv_ref=jnp.array(0.800),
                                             CL_M=jnp.array(d["cl_m"]),
                                             Cm_M=jnp.array(d["cm_m"]))),
        table=_errors(_declared(b9["cl_m"], b9["cd_m"], b9["cm_m"])),
        thrust_compensated=_errors(_declared(d["cl_m"], d["cd_m"],
                                             d["cm_m"] + b9["Cm_trim"] / 0.400)),
    )


def test_declaring_the_set_closed_most_of_the_phugoid_gap(retest):
    """THE RETEST. Bare: phugoid wn -18.1%, zeta +13.2%. Declared: both under a
    third and a half of that respectively, and the frequency error has changed
    sign -- it overshoots, which the next tests explain. Bands, not values.

    Runs on the speed set as first declared, thrust through the CG. Since the
    thrust line was declared too, the SHIPPED entry is the one below that closes
    the overshoot; this claim is about the speed set, and is kept on it."""
    b, s = retest["bare"], retest["speed_only"]
    assert b["ph_wn"] < -0.15 and b["ph_z"] > 0.10
    assert 0.02 < s["ph_wn"] < 0.07
    assert 0.01 < s["ph_z"] < 0.09
    assert abs(s["ph_wn"]) < abs(b["ph_wn"]) / 3.0 and abs(s["ph_z"]) < abs(b["ph_z"]) / 2.0


def test_the_shipped_entry_and_the_curve_read_at_trim_agree(retest):
    """The registry subtracts Korn/Lock's slope at its OWN construction point
    (M 0.800, CL = W/qS); `_declared` subtracts it at the engine's trim
    (M 0.7995, CL from CL0 + CLa*alpha). That is a real difference of about 3%
    of the slope, and it must stay small enough not to matter: under a point of
    phugoid damping, and nothing on the frequency."""
    # Both without the thrust line: the copy is built on BARE.
    assert retest["speed_only"]["ph_wn"] == pytest.approx(retest["copy"]["ph_wn"], abs=0.005)
    assert retest["speed_only"]["ph_z"] == pytest.approx(retest["copy"]["ph_z"], abs=0.02)


def test_the_speed_derivatives_leave_the_short_period_alone(retest):
    """Speed derivatives are a phugoid effect. The short-period pair moves by
    under 0.2 points; its damping gap is the alpha-dot family's, as
    test_audit_regression already attributes."""
    for k in ("sp_wn", "sp_z"):
        assert abs(retest["speed_only"][k] - retest["bare"][k]) < 0.005, k
        assert abs(retest["shipped"][k] - retest["bare"][k]) < 0.005, k


def test_the_overshoot_is_not_the_reading_it_is_the_missing_thrust_moment(retest):
    """Two statements. Table IX-4's OWN implied set overshoots the same way, so
    the residual is not digitisation error. And adding back the trim C_m the
    engine cannot have -- it places thrust through the CG -- takes both phugoid
    errors under 1%. DIAGNOSTIC ONLY: the compensated Cm_M is not sourced.

    Kept on the speed-only entry, where it was established; the test below is
    the sourced version of the diagnostic."""
    s, t, c = retest["speed_only"], retest["table"], retest["thrust_compensated"]
    assert abs(s["ph_wn"] - t["ph_wn"]) < 0.01 and abs(s["ph_z"] - t["ph_z"]) < 0.02
    assert abs(c["ph_wn"]) < 0.01 and abs(c["ph_z"]) < 0.015


def test_the_declared_thrust_line_closes_most_of_the_overshoot(retest):
    """THE SOURCED FIX, AT THE REVISED ARM. `boeing747` declares a thrust line
    5.70 ft below the CG (NASA CR-114494's revised engine pitching arms) and
    2.5 deg up (Table IX-3), with its trim referenced to it. Measured: phugoid
    wn +1.69%, zeta +2.83% against Table IX-5, where the speed set alone read
    +4.05% / +3.45%.

    CR-2144's OWN 10 ft closes it further (-0.05% / +1.13%) and its tables
    agree with 10 ft better. The revised arm is declared anyway: an arm is not
    chosen by the answer it gives. The second half asserts that cost exists,
    so a later edit cannot quietly swap the arm for the better number."""
    s, so = retest["shipped"], retest["speed_only"]
    assert 0.0 < s["ph_wn"] < 0.03 and 0.0 < s["ph_z"] < 0.05
    assert s["ph_wn"] < so["ph_wn"] / 2.0 and s["ph_z"] < so["ph_z"]
    assert abs(float(AC.thrust_arm) / 0.3048 - 5.70) < 1e-9

    ten_ft = AC._replace(
        thrust_arm=jnp.array(10.0 * 0.3048),
        Cm0=AC.Cm0 - _trim_thrust_moment_coefficient() * (10.0 - 5.70) / 5.70)
    t = _errors(ten_ft)
    assert abs(t["ph_wn"]) < 0.01 < s["ph_wn"]


def _trim_thrust_moment_coefficient():
    """T LTH / (qS c) at FC9 for the declared arm, from CR-2144's own trim."""
    _, _, _, q, adeg = cm.IX3[9]
    thrust = cm.backsolve(9)["CD"] * q * cm.S_FT2 / math.cos(math.radians(adeg) + cm.XI_RAD)
    return thrust * 5.70 / (q * cm.S_FT2 * cm.CBAR_FT)


def test_the_thrust_line_needs_the_speed_derivatives(retest):
    """NEITHER HALF WORKS ALONE. The line on the entry with its speed seam shut
    takes the phugoid frequency from -18.1% to -21.1% (at 5.70 ft): the M_u term
    it adds is the one CR-2144 pairs with Cm_M, and without Cm_M it pulls the
    wrong way."""
    assert retest["line_only"]["ph_wn"] < retest["bare"]["ph_wn"] - 0.02


def test_keeping_the_models_own_drag_rise_makes_phugoid_damping_worse(retest):
    """The partial-correction trap, and why CD_M is declared net of Korn/Lock.
    The engine's wave-drag slope at FC9 is 0.0477 per Mach against a sourced
    total of 0.025-0.028. Declaring CL_M and Cm_M on top of it takes phugoid
    damping to about +21% -- worse than declaring nothing at all."""
    assert retest["korn_lock_kept"]["ph_z"] > retest["bare"]["ph_z"]
    # The drag slope does not touch the frequency. Not to 1e-6 though: the set
    # is declared at the TABLE's M 0.800 and the engine trims at M 0.7995, so a
    # different CD_M shifts the trim throttle slightly.
    assert retest["korn_lock_kept"]["ph_wn"] == pytest.approx(retest["copy"]["ph_wn"], abs=1e-4)


def test_the_improvement_survives_the_reading_uncertainty():
    """PRICED, not assumed. Every hand-placed point on the three 40,000 ft
    speed-derivative curves re-read with pixel scatter and an axis-calibration
    error on the sheets' own scales (`cr2144_mach.perturbed`). Session 30's
    N = 4000 run gave phugoid wn 5-95% of [+1.1, +6.9]% at 1 px and
    [-5.7, +9.5]% at 3 px, against -18.1% bare. Checked here at N = 400."""
    x, _ = trim(jnp.array(V), jnp.array(H), BARE)
    alpha, de, th = (float(v) for v in x)
    A0 = longitudinal_matrix(BARE, alpha, de, th, V, H)
    a_s = float(speed_of_sound(jnp.array(H)))
    slope = _korn_lock_slope(BARE, alpha, V / a_s)
    rho, Iyy = float(density(jnp.array(H))), float(np.asarray(BARE.inertia)[1, 1])
    rng = np.random.default_rng(20260915)
    for sigma, lo, hi in ((1.0, -0.01, 0.09), (3.0, -0.12, 0.12)):
        wn = []
        for _ in range(400):
            v = [cm.value(q, "40K", 0.800,
                          curve=cm.perturbed(cm.curves()[(q, "40K")], rng, sigma))
                 for q in ("cl_m", "cd_m", "cm_m")]
            dA = cm.mach_increment(alpha, V, rho, a_s, float(BARE.mass), Iyy, float(BARE.S),
                                   float(BARE.c), v[0], v[1] - slope, v[2])
            wn.append(cm.errors_vs_ix5(cm.modes(A0 + dA))["ph_wn"])
        p5, p95 = np.percentile(wn, [5, 95])
        assert lo < p5 and p95 < hi, (sigma, p5, p95)
