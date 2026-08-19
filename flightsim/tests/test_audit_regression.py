"""Permanent fixtures from the physics audit (see AUDIT.md, INVENTORY.md, NOTATION.md).

Two kinds of test live here, and the distinction is the point.

**Verifications.** A number read out of a source document, or an algebraic
identity re-derived by hand, pinned so a future change cannot quietly break it.

**Bounds on known flaws.** The audit's most valuable outputs are not the
transcriptions that were right; they are the measured sizes of the things that
are wrong. Those are what will drift. A flaw whose magnitude is pinned is a
documented flaw; a flaw whose magnitude is free is an undocumented one waiting
to happen. Each is labelled KNOWN FLAW, with the size the audit measured.

Sources, both in the repo:
  CR-2144  refs/NASA-CR-2144.pdf   -- Heffley & Jewell, NASA CR-2144, Dec 1972.
           Section IX page offset: PDF page = printed page + 6.
  FD2e     Flight_Dynamics_-_Second_Edition.pdf -- Stengel, 2nd ed., Princeton
           UP 2022, ISBN 9780691220253.

Nothing here reads a PDF at test time. The values below were read from rendered
images of those pages during the audit and are transcribed with their printed
page numbers, so a reader can check any one of them by hand.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flightsim import airframe, provenance, wind
from flightsim.aero import aero_forces_moments, air_data
from flightsim.aircraft import CRUISE, REGISTRY, _unprime
from flightsim.atmosphere import G0, density
from flightsim.state import Controls, State, euler_to_quat, quat_to_dcm
from flightsim.trim import trim
from flightsim.units import DEG2RAD, FT2M
from flightsim.validation import (
    longitudinal_matrix,
    to_imperial_matrix,
    to_stability_axes,
)

# ---------------------------------------------------------------------------
# CR-2144 Table IX-2, printed p.217 (PDF p.223)
# "B-747 Power Approach Configuration Non-Dimensional Derivatives"
# h = sea level, VTo = 165 KTAS, alpha0 = 5.7 deg, delta_s = -2.1 deg
# ---------------------------------------------------------------------------
TABLE_IX2 = {
    "CL0_implied": 1.11, "CD_implied": 0.102,
    "CLa": 5.70, "CDa_implied": 0.66, "Cma": -1.26,
    "CLq": 5.4, "Cmq": -20.8, "CLde": 0.338, "Cmde": -1.34,
    "CYb": -0.96, "Clb": -0.221, "Cnb": 0.150,
    "Clp": -0.45, "Cnp": -0.121, "Clr": 0.101, "Cnr": -0.30,
    "Clda": 0.0461, "Cnda": 0.0064,
    "CYdr": 0.175, "Cldr": 0.007, "Cndr": -0.109,
}
# Excluded by the model's alpha/q/de form, but tabulated:
IX2_EXCLUDED = {"CLad": -6.7, "Cmad": -3.2, "CLM": -0.81, "CmM": 0.27}

# CR-2144 Table IX-3, printed p.229 -- B-747 dimensional/mass/flight condition
IX3_FC9 = dict(H="40K", M=0.800, VTo=774.0, W=636636.0, cg_mgc=0.250,
               Ix=0.182e8, Iy=0.331e8, Iz=0.497e8, Ixz=970056.0,
               Q=177.0, alpha_deg=4.60, gamma_deg=0.0, LXP=86.0, LZP=-10.0)
IX3_FC2 = dict(H="SL", M=0.249, VTo=278.0, W=564032.0, cg_mgc=0.250,
               Ix=0.142e8, Iy=0.323e8, Iz=0.454e8, Ixz=870050.0,
               Q=92.2, alpha_deg=5.70, gamma_deg=0.0, LXP=86.0, LZP=-10.0)
IX3_HEADER = dict(S=5500.0, b=195.68, c=27.31)

# CR-2144 Table IX-4, printed p.230, "(BODY AXIS SYSTEM)"
IX4_FC9 = dict(Xu=-0.00276, Zu=-0.0650, Mu=0.000193, Xw=0.0389, Zw=-0.317,
               Mw=-0.00105, Zwd=0.00556, Zq=-5.16, Mwd=-0.000116, Mq=-0.339,
               Xde=1.44, Zde=-17.9, Mde=-1.16)
IX4_FC2 = dict(Xu=-0.0108, Zu=-0.150, Mu=0.000181, Xw=0.106, Zw=-0.613,
               Mw=-0.00193, Zwd=0.0338, Zq=-7.58, Mwd=-0.000240, Mq=-0.437,
               Xde=0.971, Zde=-9.73, Mde=-0.574)

# CR-2144 Table IX-8, printed p.234, "(BODY AXIS SYSTEM)", PRIMED
IX8_FC9 = dict(Yv=-0.0558, Yb=-43.2, Lb=-3.05, Nb=0.598, Lp=-0.465, Np=-0.0316,
               Lr=0.388, Nr=-0.115, Yca=0.0, Lca=0.143, Nca=0.00775,
               Ycr=0.00729, Lcr=0.153, Ncr=-0.475)

# CR-2144 Table IX-5, printed p.231, "Bare Airframe (BODY AXIS SYSTEM)".
# The longitudinal denominator, flight condition 9. Read at 600 dpi -- at 200
# dpi Z(DET)1 can be misread as carrying a leading minus; it does not, and
# Table IX-6's identical denominator confirms it.
IX5_FC9_DENOM = dict(phugoid_zeta=0.0489, phugoid_wn=0.0673,
                     sp_zeta=0.387, sp_wn=0.964)

B747 = REGISTRY["boeing747"]
B747PA = REGISTRY["boeing747_approach"]
G = 32.174  # ft/s^2, the value CR-2144's own arithmetic uses


# ===========================================================================
# 1. Transcription -- every number the approach set claims to take verbatim
# ===========================================================================

@pytest.mark.parametrize("name,expected", sorted(
    (k, v) for k, v in TABLE_IX2.items() if not k.endswith("_implied")))
def test_the_approach_set_is_table_ix2_verbatim(name, expected):
    """CR-2144 Table IX-2, printed p.217. All 21 directly-stored derivatives.

    Table IX-2 is already non-dimensional, so there is no conversion chain to
    get wrong here -- this is pure transcription, and it is exact.
    """
    assert float(getattr(B747PA, name)) == pytest.approx(expected, rel=1e-12)


def test_the_approach_trim_reference_reproduces_table_ix2_CL_and_CD():
    """CL0 and Cm0 are referenced to alpha0 = 5.7 deg, so CL(alpha0) must be the
    tabulated 1.11 and CD(alpha0) the tabulated 0.102, exactly by construction."""
    a0 = 5.7 * DEG2RAD
    CL = float(B747PA.CL0) + float(B747PA.CLa) * a0
    CD = float(B747PA.CD0) + CL**2 / (math.pi * float(B747PA.e) * float(B747PA.AR))
    assert CL == pytest.approx(TABLE_IX2["CL0_implied"], rel=1e-10)
    assert CD == pytest.approx(TABLE_IX2["CD_implied"], rel=1e-10)


def test_the_747_reference_geometry_is_table_ix3s_header():
    """S = 5500 sq ft, b = 195.68 ft, cbar = 27.31 ft, printed p.229."""
    for ac in (B747, B747PA):
        assert float(ac.S) == pytest.approx(IX3_HEADER["S"] * FT2M**2, rel=1e-12)
        assert float(ac.b) == pytest.approx(IX3_HEADER["b"] * FT2M, rel=1e-12)
        assert float(ac.c) == pytest.approx(IX3_HEADER["c"] * FT2M, rel=1e-12)


@pytest.mark.parametrize("ac,fc", [(B747, IX3_FC9), (B747PA, IX3_FC2)])
def test_mass_and_inertia_are_table_ix3s_own_flight_condition(ac, fc):
    """Flight condition 9 (cruise) and 2 (power approach), printed p.229.

    Figure IX-1's Power Approach block disagrees with Table IX-3 by up to 6% on
    the approach inertias; the code takes IX-3 and says why. This pins IX-3.
    """
    slug_ft2 = 14.5939029372064 * FT2M**2
    assert float(ac.mass) == pytest.approx(fc["W"] * 0.45359237, rel=1e-12)
    I = np.asarray(ac.inertia)
    assert I[0, 0] == pytest.approx(fc["Ix"] * slug_ft2, rel=1e-9)
    assert I[1, 1] == pytest.approx(fc["Iy"] * slug_ft2, rel=1e-9)
    assert I[2, 2] == pytest.approx(fc["Iz"] * slug_ft2, rel=1e-9)
    # Ixz enters the tensor NEGATED -- the usual positive-forward-up convention.
    assert I[0, 2] == pytest.approx(-fc["Ixz"] * slug_ft2, rel=1e-9)
    assert I[2, 0] == I[0, 2]
    assert I[0, 1] == 0.0 and I[1, 2] == 0.0


def test_the_cruise_set_inverts_table_ix4s_own_dimensional_derivatives():
    """Round-trip: the code's non-dimensional cruise set, pushed back through
    the same relations, must return Table IX-4 flight condition 9 (printed
    p.230). This pins the seven values actually transcribed, INCLUDING
    Mq = -0.339 -- the session-5 fix, which the source confirms.
    """
    m = IX3_FC9["W"] / G
    qS = IX3_FC9["Q"] * IX3_HEADER["S"]
    c, U0, Iy = IX3_HEADER["c"], IX3_FC9["VTo"], IX3_FC9["Iy"]
    got = dict(
        Zw=-(float(B747.CLa) + 0.043) * qS / (m * U0),
        Zq=-float(B747.CLq) * qS * c / (2.0 * m * U0),
        Zde=-float(B747.CLde) * qS / m,
        Mw=float(B747.Cma) * qS * c / (Iy * U0),
        Mq=float(B747.Cmq) * qS * c * c / (2.0 * Iy * U0),
        Mde=float(B747.Cmde) * qS * c / Iy,
    )
    for k, v in got.items():
        assert v == pytest.approx(IX4_FC9[k], rel=2e-3), f"{k}: {v} vs {IX4_FC9[k]}"


def test_unprime_is_cr2144_appendix_a_page_a18_exactly():
    """Appendix A, printed A-18:
        L' = (L_b + Ixz*N_b/Ix) * G,  N' = (N_b + Ixz*L_b/Iz) * G,
        G  = 1 / (1 - Ixz^2/(Ix*Iz))
    `aircraft._unprime` inverts exactly this. Round-trip it on Table IX-8's own
    primed values and the identity must return them.
    """
    Ix, Iz, Ixz = IX3_FC9["Ix"], IX3_FC9["Iz"], IX3_FC9["Ixz"]
    g = 1.0 / (1.0 - Ixz**2 / (Ix * Iz))
    for key in ("b", "p", "r", "ca", "cr"):
        Lp, Np = IX8_FC9["L" + key], IX8_FC9["N" + key]
        L, N = (float(v) for v in _unprime(Lp, Np, Ix, Iz, Ixz))
        assert (L + Ixz * N / Ix) * g == pytest.approx(Lp, rel=1e-10)
        assert (N + Ixz * L / Iz) * g == pytest.approx(Np, rel=1e-10)


def test_the_lateral_chain_reproduces_table_ix8_flight_condition_9():
    """The code's lateral non-dimensional set, pushed back through CR-2144
    Appendix A A-17/A-18, must return Table IX-8's primed values (printed
    p.234). Appendix A's lateral relations carry no W0 term, so this is exact.
    """
    m, qS = IX3_FC9["W"] / G, IX3_FC9["Q"] * IX3_HEADER["S"]
    b, U0 = IX3_HEADER["b"], IX3_FC9["VTo"]
    Ix, Iz, Ixz = IX3_FC9["Ix"], IX3_FC9["Iz"], IX3_FC9["Ixz"]
    g = 1.0 / (1.0 - Ixz**2 / (Ix * Iz))
    raw = {
        "b": (float(B747.Clb) * qS * b / Ix, float(B747.Cnb) * qS * b / Iz),
        "p": (float(B747.Clp) * qS * b * b / (2 * U0 * Ix),
              float(B747.Cnp) * qS * b * b / (2 * U0 * Iz)),
        "r": (float(B747.Clr) * qS * b * b / (2 * U0 * Ix),
              float(B747.Cnr) * qS * b * b / (2 * U0 * Iz)),
        "ca": (float(B747.Clda) * qS * b / Ix, float(B747.Cnda) * qS * b / Iz),
        "cr": (float(B747.Cldr) * qS * b / Ix, float(B747.Cndr) * qS * b / Iz),
    }
    for key, (L, N) in raw.items():
        assert (L + Ixz * N / Ix) * g == pytest.approx(IX8_FC9["L" + key], rel=2e-3)
        assert (N + Ixz * L / Iz) * g == pytest.approx(IX8_FC9["N" + key], rel=2e-3)
    assert float(B747.CYb) * qS / m == pytest.approx(IX8_FC9["Yb"], rel=2e-3)


# ===========================================================================
# 2. The conversion chain, closed against CR-2144's own two tables
# ===========================================================================

def _chain_from_dimensional(fc, ix4, CL, CD):
    """The relations aircraft.py actually uses, applied to a flight condition."""
    m, qS = fc["W"] / G, fc["Q"] * IX3_HEADER["S"]
    c, Iy, VTo = IX3_HEADER["c"], fc["Iy"], fc["VTo"]
    CLa = -ix4["Zw"] * m * VTo / qS - CD
    return dict(
        CLa=CLa,
        CLq=-ix4["Zq"] * 2.0 * m * VTo / (qS * c),
        CLde=-ix4["Zde"] * m / qS,
        Cma=ix4["Mw"] * Iy * VTo / (qS * c),
        Cmq=ix4["Mq"] * 2.0 * Iy * VTo / (qS * c * c),
        Cmde=ix4["Mde"] * Iy / (qS * c),
        CDa=(-ix4["Xw"] * m * VTo / qS + CD * math.sin(fc["alpha_deg"] * DEG2RAD)
             + CLa * math.sin(fc["alpha_deg"] * DEG2RAD)
             + CL * math.cos(fc["alpha_deg"] * DEG2RAD))
            / math.cos(fc["alpha_deg"] * DEG2RAD),
    )


# Measured by the audit. CR-2144 publishes BOTH a dimensional (IX-4) and a
# non-dimensional (IX-2) set for flight condition 2, so the chain can be closed
# against the source's own arithmetic. The project uses IX-4 for cruise and
# IX-2 for approach and never crosses them, so this loop had never been run.
CHAIN_CLOSURE_FC2 = {           # code relation vs Table IX-2, worst |rel err|
    "CLa": 0.016, "CDa": 0.022, "Cma": 0.007,
    "CLq": 0.013, "Cmq": 0.003, "CLde": 0.005, "Cmde": 0.001,
}


@pytest.mark.parametrize("name,bound", sorted(CHAIN_CLOSURE_FC2.items()))
def test_the_conversion_chain_closes_against_cr2144s_own_two_tables(name, bound):
    """Run aircraft.py's OWN relations on Table IX-4 flight condition 2 and
    compare against Table IX-2, which tabulates the same condition
    non-dimensionally. Worst disagreement 2.2% (CDa); everything else <= 1.6%.

    This is the strongest available check on the CRUISE set, because cruise has
    no non-dimensional table to compare against -- the relations are the same.
    """
    got = _chain_from_dimensional(IX3_FC2, IX4_FC2, TABLE_IX2["CL0_implied"],
                                  TABLE_IX2["CD_implied"])
    ref = {**TABLE_IX2, "CDa": TABLE_IX2["CDa_implied"]}[name]
    err = abs(got[name] - ref) / abs(ref)
    assert err <= bound, f"{name}: chain gives {got[name]:.5f} vs IX-2 {ref}, {err:.2%}"


def test_appendix_a_defines_no_Zq_but_the_assumed_relation_cross_checks():
    """KNOWN GAP, bounded. CR-2144 Appendix A section 5 defines X_u, X_w, X_de,
    Z_u, Z_w, Z_wdot, Z_de, M_u, M_w, M_wdot, M_alpha, M_alphadot, M_q, M_de,
    T_u and the whole lateral set (printed A-16..A-18) -- but NO Z_q, despite
    Table IX-4 printing a ZQ row. `aircraft.py` assumes the M_q analogue.

    It cannot be verified from the definition, but it CAN be cross-checked:
    push Table IX-4 FC2's ZQ = -7.58 through it and Table IX-2's own CLq = 5.4
    must come back. It does, to 1.3%.
    """
    got = _chain_from_dimensional(IX3_FC2, IX4_FC2, 1.11, 0.102)["CLq"]
    assert got == pytest.approx(TABLE_IX2["CLq"], rel=0.013)


# KNOWN FLAW, bounded. CR-2144 Appendix A (printed A-16/A-17) defines
#   X_w = (rho S U0/2m)[-C_Xa - 2(W0/U0)(C_X + (M/2)C_XM)]
#   Z_w = (rho S U0/2m)[-C_Na - 2(W0/U0)(C_N + (M/2)C_NM)]
#   M_w = (rho S c U0/2Iy)[ C_ma + (2W0/U0)(C_m + (M/2)C_mM)]
# with U0 = VTo*cos(alpha0).  aircraft.py inverts these as if the bracket were
# the first term alone and as if U0 = VTo.  Measured cost at CRUISE, with the
# unknown Mach term set to zero (the cruise C_LM is a plot, not a table):
APPENDIX_A_OMISSION_AT_CRUISE = {"CLa": 0.0115, "CDa": 0.0113, "Cma": 0.0032}


@pytest.mark.parametrize("name,expected", sorted(APPENDIX_A_OMISSION_AT_CRUISE.items()))
def test_the_appendix_a_W0_omission_stays_within_its_measured_bound(name, expected):
    """KNOWN FLAW, bounded at ~1.2%. If this test moves, either the conversion
    chain changed or the trim reference did -- both are things a reader of
    PROJECT.md section 4 would want to know about.
    """
    fc, ix4 = IX3_FC9, IX4_FC9
    m, qS = fc["W"] / G, fc["Q"] * IX3_HEADER["S"]
    c, Iy, VTo = IX3_HEADER["c"], fc["Iy"], fc["VTo"]
    a0 = fc["alpha_deg"] * DEG2RAD
    ca, sa = math.cos(a0), math.sin(a0)
    U0, W0 = VTo * ca, VTo * sa
    rho = 2.0 * fc["Q"] / VTo**2
    CL, CD = fc["W"] / qS, 0.043
    C_N, C_X = CL * ca + CD * sa, CD * ca - CL * sa
    f = 2.0 * m / (rho * IX3_HEADER["S"] * U0)

    code = _chain_from_dimensional(fc, ix4, CL, CD)
    # Appendix A gives C_Na and C_Xa; C_L and C_D follow from the exact pair
    #   C_Na =  CLa*ca - CL*sa + CDa*sa + CD*ca
    #   C_Xa =  CDa*ca - CD*sa - CLa*sa - CL*ca
    # which is a 2x2 linear system. Solving it (rather than substituting a
    # provisional CLa into the CDa expression) removes an ordering ambiguity
    # that shifts CDa by ~0.4%.
    C_Na = -ix4["Zw"] * f - 2.0 * (W0 / U0) * C_N
    C_Xa = -ix4["Xw"] * f - 2.0 * (W0 / U0) * C_X
    sol = np.linalg.solve(np.array([[ca, sa], [-sa, ca]]),
                          np.array([C_Na + CL * sa - CD * ca,
                                    C_Xa + CD * sa + CL * ca]))
    appA = dict(
        CLa=sol[0], CDa=sol[1],
        Cma=ix4["Mw"] * 2.0 * Iy / (rho * IX3_HEADER["S"] * c * U0),
    )
    err = (code[name] - appA[name]) / abs(appA[name])
    assert err == pytest.approx(expected, abs=0.002), (
        f"{name}: code {code[name]:.5f} vs Appendix-A-literal {appA[name]:.5f} "
        f"= {err:+.2%} (audit measured {expected:+.2%})"
    )


# ===========================================================================
# 3. Mode comparison and its attribution
# ===========================================================================

def _cruise_modes():
    V, H = CRUISE["boeing747"]["airspeed"], CRUISE["boeing747"]["altitude"]
    x, _ = trim(jnp.array(V), jnp.array(H), B747)
    A = longitudinal_matrix(B747, float(x[0]), float(x[1]), float(x[2]), V, H)
    ev = np.linalg.eigvals(A)
    osc = sorted((abs(l), -l.real / abs(l)) for l in ev if l.imag > 1e-9)
    return dict(phugoid_wn=osc[0][0], phugoid_zeta=osc[0][1],
                sp_wn=osc[1][0], sp_zeta=osc[1][1])


# KNOWN FLAW, bounded. CR-2144 Table IX-5's denominator publishes FOUR factors
# for flight condition 9; PROJECT.md section 4 compares TWO. All four are
# pinned here with the errors the audit measured.
MODE_ERRORS_VS_IX5 = {"phugoid_wn": -0.178, "phugoid_zeta": +0.144,
                      "sp_wn": -0.014, "sp_zeta": -0.115}


@pytest.mark.parametrize("name,expected", sorted(MODE_ERRORS_VS_IX5.items()))
def test_all_four_published_longitudinal_factors_are_compared(name, expected):
    """CR-2144 Table IX-5, printed p.231, flight condition 9 denominator:
    Z(DET)1 = .0489, W(DET)1 = .0673, Z(DET)2 = .387, W(DET)2 = .964.

    PROJECT.md section 4 reports only W(DET)1 and Z(DET)2. Phugoid zeta (+14.4%)
    and short-period wn (-1.4%) are equally available and were never compared.
    """
    got = _cruise_modes()[name]
    err = (got - IX5_FC9_DENOM[name]) / IX5_FC9_DENOM[name]
    assert err == pytest.approx(expected, abs=0.01), (
        f"{name}: model {got:.5f} vs Table IX-5 {IX5_FC9_DENOM[name]}, "
        f"{err:+.1%} (audit measured {expected:+.1%})"
    )


def _reconstruct(use_speed, use_alphadot, xu_override=None):
    """The 4-state longitudinal system built from CR-2144 Table IX-4 FC9's own
    dimensional derivatives. Uses NO engine code, so it is an independent
    instrument for attributing the engine's mode errors."""
    ix4 = IX4_FC9
    xu, zu, mu = ((ix4["Xu"], ix4["Zu"], ix4["Mu"]) if use_speed else (0.0, 0.0, 0.0))
    if xu_override is not None:
        xu = xu_override
    zwd, mwd = ((ix4["Zwd"], ix4["Mwd"]) if use_alphadot else (0.0, 0.0))
    d = 1.0 - zwd
    th0 = IX3_FC9["alpha_deg"] * DEG2RAD
    U0 = IX3_FC9["VTo"] * math.cos(th0)
    A = np.array([
        [xu, ix4["Xw"], 0.0, -G * math.cos(th0)],
        [zu / d, ix4["Zw"] / d, (U0 + ix4["Zq"]) / d, -G * math.sin(th0) / d],
        [mu + mwd * zu / d, ix4["Mw"] + mwd * ix4["Zw"] / d,
         ix4["Mq"] + mwd * (U0 + ix4["Zq"]) / d, -mwd * G * math.sin(th0) / d],
        [0.0, 0.0, 1.0, 0.0]])
    ev = np.linalg.eigvals(A)
    return sorted((abs(l), -l.real / abs(l)) for l in ev if l.imag > 1e-9)


def test_the_phugoid_frequency_error_is_the_omitted_speed_derivatives():
    """ATTRIBUTION, demonstrated. Restoring Xu, Zu, Mu alone -- and nothing else
    -- takes the phugoid frequency to within 0.3% of CR-2144's published 0.0673.
    Without them CR-2144's own derivatives give no oscillatory phugoid at all,
    so the mode's very existence in the engine comes from its implicit Xu/Zu.
    """
    assert len(_reconstruct(False, False)) < 2, "expected no oscillatory phugoid"
    wn = _reconstruct(True, False)[0][0]
    assert wn == pytest.approx(IX5_FC9_DENOM["phugoid_wn"], rel=0.01)


def test_the_short_period_damping_error_is_the_omitted_alpha_dot_derivatives():
    """ATTRIBUTION, demonstrated. Restoring Zwdot, Mwdot alone takes short-period
    damping from 0.3446 (-10.9%) to 0.3912, within 1.1% of the published 0.387.
    """
    assert _reconstruct(False, False)[-1][1] == pytest.approx(0.3446, abs=0.002)
    assert _reconstruct(False, True)[-1][1] == pytest.approx(
        IX5_FC9_DENOM["sp_zeta"], rel=0.02)


def test_the_two_attributions_are_orthogonal():
    """The two causes are independent, which is why they are two findings and
    not one. Speed derivatives barely move short-period damping; alpha-dot
    derivatives leave the phugoid non-oscillatory entirely.
    """
    base_sp = _reconstruct(False, False)[-1][1]
    assert _reconstruct(True, False)[-1][1] == pytest.approx(base_sp, rel=0.01)
    assert len(_reconstruct(False, True)) < 2


def _patched_engine_modes(speed, alphadot):
    """The ENGINE's own cruise plant matrix with the omitted derivative families
    restored from CR-2144 Table IX-4 FC9.

    This is the decisive form of the attribution, and it is stronger than the
    standalone reconstruction above. `_reconstruct` shows what CR-2144's OWN
    derivatives imply; it does not show that the omissions explain THE ENGINE,
    which already carries implicit Xu/Zu from dynamic-pressure variation and a
    `d(udot)/dq = -w0` Coriolis term the source's linear model has no row for.
    Patching the engine's own matrix closes that gap.
    """
    V, H = CRUISE["boeing747"]["airspeed"], CRUISE["boeing747"]["altitude"]
    x, _ = trim(jnp.array(V), jnp.array(H), B747)
    A = to_imperial_matrix(longitudinal_matrix(
        B747, float(x[0]), float(x[1]), float(x[2]), V, H))
    if speed:
        A[0, 0], A[1, 0], A[2, 0] = IX4_FC9["Xu"], IX4_FC9["Zu"], IX4_FC9["Mu"]
    if alphadot:
        d = 1.0 - IX4_FC9["Zwd"]
        A[1, :] = A[1, :] / d
        A[2, :] = A[2, :] + IX4_FC9["Mwd"] * A[1, :]
    ev = np.linalg.eigvals(A)
    osc = sorted((abs(l), -l.real / abs(l)) for l in ev if l.imag > 1e-9)
    return dict(phugoid_wn=osc[0][0], phugoid_zeta=osc[0][1],
                sp_wn=osc[1][0], sp_zeta=osc[1][1])


def test_restoring_both_omitted_families_closes_ALL_FOUR_modes_to_one_percent():
    """ATTRIBUTION, complete. This is the audit's central positive result.

    Restoring CR-2144's speed derivatives and alpha-dot terms to the ENGINE's
    own plant matrix takes every one of the four published Table IX-5 factors
    to within 1% -- from -17.8%, +14.4%, -1.4%, -11.5%.

    So the whole of the 747 cruise mode discrepancy is the two omissions
    PROJECT.md section 5 names, and nothing else. The engine's aerodynamic data,
    conversion chain, trim solve and eigen-extraction are all exonerated.
    """
    got = _patched_engine_modes(speed=True, alphadot=True)
    for name, ref in IX5_FC9_DENOM.items():
        assert abs(got[name] - ref) / ref < 0.01, (
            f"{name}: {got[name]:.5f} vs {ref} = {100*(got[name]-ref)/ref:+.1f}%")


def test_the_phugoid_frequency_is_the_speed_derivatives_alone():
    """Speed derivatives alone: phugoid wn -17.8% -> +0.3%, while short-period
    damping does NOT improve (-11.5% -> -12.0%). One cause, one mode."""
    got = _patched_engine_modes(speed=True, alphadot=False)
    assert abs(got["phugoid_wn"] - 0.0673) / 0.0673 < 0.01
    assert got["sp_zeta"] < 0.345, "short-period damping must NOT be fixed by this"


def test_the_short_period_damping_is_the_alpha_dot_terms_alone():
    """Alpha-dot terms alone: short-period zeta -11.5% -> +0.6%, while the
    phugoid frequency does not move at all. The other half of the pair."""
    base = _cruise_modes()
    got = _patched_engine_modes(speed=False, alphadot=True)
    assert abs(got["sp_zeta"] - 0.387) / 0.387 < 0.01
    assert got["phugoid_wn"] == pytest.approx(base["phugoid_wn"], rel=1e-6)


def test_phugoid_damping_needs_BOTH_families_which_is_why_it_looked_unattributable():
    """Phugoid damping is the one factor neither omission explains alone:
    +14.4% shipped, +3.7% with speed derivatives, +13.0% with alpha-dot terms,
    +0.5% with both.

    A reconstruction built from Table IX-4's derivatives ALONE gets this wrong
    (it returns a negative zeta), because it discards the engine's own
    `d(udot)/dq = -w0` Coriolis term and the X_q its drag polar produces --
    neither of which CR-2144's linear model carries. That is why the attribution
    had to be done on the engine's matrix rather than on a clean-room rebuild,
    and it is recorded here so the next auditor does not repeat the detour.
    """
    shipped = _cruise_modes()["phugoid_zeta"]
    only_speed = _patched_engine_modes(True, False)["phugoid_zeta"]
    only_adot = _patched_engine_modes(False, True)["phugoid_zeta"]
    both = _patched_engine_modes(True, True)["phugoid_zeta"]
    ref = IX5_FC9_DENOM["phugoid_zeta"]
    err = lambda v: abs(v - ref) / ref  # noqa: E731
    assert err(shipped) > 0.10
    assert err(only_speed) > 0.02 and err(only_speed) < err(shipped)
    assert err(only_adot) > 0.10
    assert err(both) < 0.01
    # and the clean-room reconstruction really does get the sign wrong
    assert _reconstruct(True, True)[0][1] < 0.0


def test_the_engine_carries_an_Xq_that_cr2144_does_not_model():
    """UNDOCUMENTED MODELLING CHOICE, quantified.

    aero.coefficients builds CD from the TOTAL CL, which includes CLq*qhat. So a
    pitch rate changes induced drag and the engine has a stability-axis
    A[0,2] = X_q. CR-2144 tabulates no C_Dq and its linear model has none.
    Measured: the engine's X_q equals the parabolic polar's own prediction to
    1%, which is what identifies the mechanism rather than merely noting the
    element is non-zero.
    """
    V, H = CRUISE["boeing747_approach"]["airspeed"], CRUISE["boeing747_approach"]["altitude"]
    x, _ = trim(jnp.array(V), jnp.array(H), B747PA)
    a = float(x[0])
    A = to_imperial_matrix(to_stability_axes(
        longitudinal_matrix(B747PA, a, float(x[1]), float(x[2]), V, H), a))
    rho = float(density(jnp.array(H)))
    qbar = 0.5 * rho * V * V
    CL = float(B747PA.mass) * 9.80665 / (qbar * float(B747PA.S))
    dCD_dqhat = 2 * CL * float(B747PA.CLq) / (math.pi * float(B747PA.e) * float(B747PA.AR))
    predicted = (-qbar * float(B747PA.S) * dCD_dqhat * (float(B747PA.c) / (2 * V))
                 / float(B747PA.mass) / FT2M)
    assert A[0, 2] != 0.0
    assert A[0, 2] == pytest.approx(predicted, rel=0.02)


# ===========================================================================
# 4. Assumption B4 -- the sensor-offset bound the register says cannot exist
# ===========================================================================

# CR-2144 Table IX-3, printed p.229: LXP = 86.0 ft, LZP = -10.0 ft, EVERY flight
# condition; and Table IX-5 publishes N(AZP/DE), normal acceleration at the
# pilot station. ASSUMPTIONS.md B4 says "Bound: none". It has one.
SENSOR_ARM_M = np.array([86.0 * FT2M, 0.0, -10.0 * FT2M])


def _sensor_correction(t, q):
    """dn_z at a body-frame offset r for a symmetric longitudinal run, where
    omega = (0,q,0):  a_p = a_cg + omega_dot x r + omega x (omega x r)."""
    qd = np.gradient(np.asarray(q), float(t[1] - t[0]))
    rx, rz = SENSOR_ARM_M[0], SENSOR_ARM_M[2]
    return (rx * qd - rz * np.asarray(q) ** 2) / 9.80665


def test_the_accelerometer_offset_is_bounded_for_the_manoeuvring_fig8_point():
    """PREVIOUSLY UNBOUNDED, NOW BOUNDED at 8.3% of the load excursion."""
    from flightsim import vortex_viz
    V, H = CRUISE["boeing747"]["airspeed"], CRUISE["boeing747"]["altitude"]
    enc = vortex_viz.manoeuvre(B747, V, H, label="pushdown",
                               elevator_step=8.926 * DEG2RAD, hold=6.609,
                               seconds=12.609, dt=0.01)
    w = np.asarray(enc.window)
    nz = np.asarray(enc.n_z)
    d = _sensor_correction(enc.t, enc.q)
    i = int(np.argmax(np.abs(nz[w] - nz[0])))
    frac = abs(d[w][i] / (nz[w] - nz[0])[i])
    assert frac == pytest.approx(0.083, abs=0.02), (
        f"pilot-station correction is {frac:.1%} of the CG excursion")


def test_the_sensor_offset_bites_hardest_on_the_vortex_not_the_manoeuvre():
    """The finding ASSUMPTIONS.md B4 does not anticipate: it names the pitch
    manoeuvre as the worrying case, but the Parks vortex -- the project's
    headline result -- carries a THREE TIMES larger relative correction, because
    it drives a large rapid qdot against a small load excursion.
    """
    from flightsim import vortex_viz
    V, H = CRUISE["boeing747"]["airspeed"], CRUISE["boeing747"]["altitude"]
    hb = wind.PARKS_CASES["hannibal"]
    arr = wind.VortexArray(north=jnp.array([0.0, hb["spacing"]]),
                           down=jnp.array([-H, -H]),
                           r0=jnp.array(hb["r0"]), v0=jnp.array(hb["v0"]))
    enc = vortex_viz.fly(B747, lambda p: wind.vortex_wind(p, arr), V, H,
                         label="vortex", start_north=-40 * hb["r0"], seconds=40.0,
                         dt=0.01, window=(-hb["r0"], hb["r0"]), window_name="core")
    w = np.asarray(enc.window)
    nz = np.asarray(enc.n_z)
    d = _sensor_correction(enc.t, enc.q)
    i = int(np.argmax(np.abs(nz[w] - nz[0])))
    frac = abs(d[w][i] / (nz[w] - nz[0])[i])
    assert frac == pytest.approx(0.264, abs=0.04)
    assert frac > 0.15, "the vortex correction is the large one; see AUDIT.md section 3"


# ===========================================================================
# 5. Analytic identities re-derived during the audit
# ===========================================================================

def test_oseguera_and_bowles_four_constants_are_mutually_derivable():
    """wind.py stores four constants from Oseguera & Bowles 1988 and claims they
    are re-derivable from the paper's own equations rather than being four
    restatements of one number. Re-derived here.
    """
    from scipy.optimize import brentq
    # peak outflow: d/dx[(1 - e^{-x^2})/x] = 0  <=>  e^{-x^2}(2x^2 + 1) = 1.
    # The exact root is 1.120906; the paper prints 1.1212. The 2.6e-4 gap is the
    # paper's own iteration/rounding, and it is inconsequential -- the quantity
    # is stationary there, so u_max changes by under 1e-7. Recorded, not hidden.
    x = brentq(lambda x: math.exp(-x * x) * (2 * x * x + 1) - 1.0, 0.5, 3.0)
    assert x == pytest.approx(wind.MICROBURST_PEAK_RADIUS_RATIO, abs=4e-4)
    assert x == pytest.approx(1.120906, abs=1e-6)
    # z_m/z* = ln(z*/eps) / (z*/eps - 1); exact 0.219629 vs the paper's 0.22
    ratio = wind.MICROBURST_ZSTAR_OVER_EPS
    assert math.log(ratio) / (ratio - 1.0) == pytest.approx(
        wind.MICROBURST_ZM_OVER_ZSTAR, abs=5e-4)
    # u_max = 0.2357 * lambda * R follows from the other three
    zm = wind.MICROBURST_ZM_OVER_ZSTAR
    shape = math.exp(-zm) - math.exp(-zm * ratio)
    assert 0.5 * (1 - math.exp(-x * x)) / x * shape == pytest.approx(
        wind.MICROBURST_UMAX_COEFF, abs=5e-5)


def test_the_microburst_satisfies_continuity_numerically():
    """The paper's own headline property, asserted rather than trusted."""
    burst = wind.microburst(u_max=19.0, radius=1000.0, z_m=100.0)
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(300):
        p = jnp.array([rng.uniform(-2500, 2500), rng.uniform(-2500, 2500),
                       -rng.uniform(30.0, 1200.0)])
        jac = np.asarray(jnp.stack([
            (wind.microburst_wind(p.at[i].add(0.05), burst)
             - wind.microburst_wind(p.at[i].add(-0.05), burst)) / 0.1
            for i in range(3)], axis=1))
        worst = max(worst, abs(np.trace(jac)))
    assert worst < 1e-6, f"max |div| = {worst:.3e}"


def test_the_parks_vortex_is_solid_body_inside_and_irrotational_outside():
    """Parks et al.'s own stated property: "a rotational (solid-body) core
    embedded in an irrotational flow". Both halves, plus divergence-free
    everywhere, checked against the implementation."""
    hb = wind.PARKS_CASES["hannibal"]
    r0, v0 = hb["r0"], hb["v0"]
    arr = wind.VortexArray(north=jnp.array([0.0]), down=jnp.array([0.0]),
                           r0=jnp.array(r0), v0=jnp.array(v0))
    def jac(p):
        return np.asarray(jnp.stack([
            (wind.vortex_wind(p.at[i].add(0.01), arr)
             - wind.vortex_wind(p.at[i].add(-0.01), arr)) / 0.02
            for i in range(3)], axis=1))
    inside = jac(jnp.array([0.3 * r0, 0.0, -0.4 * r0]))
    outside = jac(jnp.array([2.0 * r0, 0.0, -1.5 * r0]))
    # vorticity about east: dw_north/d(down) - ... in the (north, down) plane
    vort_in = inside[2, 0] - inside[0, 2]
    vort_out = outside[2, 0] - outside[0, 2]
    assert abs(np.trace(inside)) < 1e-8 and abs(np.trace(outside)) < 1e-8
    assert abs(vort_in) == pytest.approx(2.0 * v0 / r0, rel=1e-3)
    assert abs(vort_out) < 1e-3 * (v0 / r0)


def test_the_korn_lock_offset_is_the_drag_divergence_definition():
    """aero._MDD_OFFSET claims to be fixed by dCD/dM = 0.1 at M_dd with
    CD_wave = 20(M - M_crit)^4. Re-derived: 80*d^3 = 0.1."""
    from flightsim.aero import _MDD_OFFSET
    assert 80.0 * _MDD_OFFSET**3 == pytest.approx(0.1, rel=1e-12)


@pytest.mark.parametrize("axis,expected_index", [(0, 0), (1, 1), (2, 2)])
def test_gust_rates_recover_a_rigid_rotation_of_the_air_mass(axis, expected_index):
    """FD2e eqs. 3.4-48, 3.4-50, 3.4-52 (printed p.216), re-derived rather than
    transcribed: for an air mass rotating rigidly at Omega, `gust_rates` must
    return exactly Omega, on ALL THREE axes independently. Testing one axis
    cannot distinguish a sign convention from a transposition.
    """
    omega_air = np.zeros(3)
    omega_air[axis] = 0.37
    field = lambda p: jnp.cross(jnp.array(omega_air), p)  # noqa: E731
    quat = euler_to_quat(jnp.array(0.0), jnp.array(0.0), jnp.array(0.0))
    got = np.asarray(wind.gust_rates(jnp.array([120.0, -80.0, -3000.0]), quat, field))
    assert got[expected_index] == pytest.approx(omega_air[axis], rel=1e-8)
    assert np.abs(np.delete(got, expected_index)).max() < 1e-8


def test_fd2e_equation_3_4_49_is_wrong_and_the_code_does_not_use_it():
    """FD2e prints, for the aircraft's own rotation (printed p.216):
        eq. 3.4-48  dw/dy = +p        eq. 3.4-49  dv/dz = +p
    A rigid rotation gives v = r*x - p*z, hence dv/dz = -p. Equation 3.4-49
    contradicts the book's own 3.4-48. Re-derived here so the claim in
    NOTATION.md section 7.4 rests on arithmetic rather than on a citation.
    """
    p = 0.41
    omega = np.array([p, 0.0, 0.0])
    r = np.array([13.0, -7.0, 5.0])
    v = np.cross(omega, r)  # velocity of a body-fixed point
    eps = 1e-6
    dv_dz = (np.cross(omega, r + np.array([0, 0, eps]))[1]
             - np.cross(omega, r - np.array([0, 0, eps]))[1]) / (2 * eps)
    dw_dy = (np.cross(omega, r + np.array([0, eps, 0]))[2]
             - np.cross(omega, r - np.array([0, eps, 0]))[2]) / (2 * eps)
    assert dw_dy == pytest.approx(+p, rel=1e-6)   # eq. 3.4-48 is right
    assert dv_dz == pytest.approx(-p, rel=1e-6)   # eq. 3.4-49 says +p; it is wrong
    assert v is not None


def test_fd2e_equation_3_4_55_is_wrong_by_a_factor_of_minus_two():
    """FD2e eq. 3.4-55 (printed p.217): dM = -(dM_q)(w_wx - u_wz).
    For an air mass in rigid pitch rotation at q_a, w_wx = -q_a and u_wz = +q_a,
    so the book gives +2*M_q*q_a. The correct increment is -M_q*q_a. Ratio -2.
    """
    q_a = 0.29
    omega_air = np.array([0.0, q_a, 0.0])
    eps = 1e-6
    base = np.array([11.0, 4.0, -6.0])
    w_wx = (np.cross(omega_air, base + np.array([eps, 0, 0]))[2]
            - np.cross(omega_air, base - np.array([eps, 0, 0]))[2]) / (2 * eps)
    u_wz = (np.cross(omega_air, base + np.array([0, 0, eps]))[0]
            - np.cross(omega_air, base - np.array([0, 0, eps]))[0]) / (2 * eps)
    assert w_wx == pytest.approx(-q_a, rel=1e-6)
    assert u_wz == pytest.approx(+q_a, rel=1e-6)
    book = -(w_wx - u_wz)          # per unit M_q
    correct = -q_a                 # per unit M_q: aero sees omega_rel = -q_a
    assert book / correct == pytest.approx(-2.0, rel=1e-6)


def test_lift_is_perpendicular_to_the_relative_wind_at_nonzero_alpha_AND_beta():
    """The asymmetric case. `aero.aero_forces_moments` rotates wind-axis lift and
    drag into body axes through alpha and beta; a form that is only correct at
    beta = 0 would survive every symmetric test in the suite.
    """
    vel_rel = jnp.array([210.0, 31.0, 24.0])   # both alpha and beta non-zero
    V, alpha, beta = air_data(vel_rel)
    assert abs(float(alpha)) > 0.1 and abs(float(beta)) > 0.1
    controls = Controls(elevator=jnp.array(0.0), aileron=jnp.array(0.0),
                        rudder=jnp.array(0.0), throttle=jnp.array(0.0))
    ac = B747PA._replace(CD0=jnp.array(0.0), e=jnp.array(1e9),
                         CYb=jnp.array(0.0), CYdr=jnp.array(0.0))
    # a_sound huge so the Korn/Lock wave-drag term is identically zero too;
    # with CD0, induced and wave drag all removed the whole force IS the lift.
    force, _ = aero_forces_moments(vel_rel, jnp.zeros(3), controls, ac,
                                   jnp.array(1.225), jnp.array(1.0e6))
    # with drag and side force removed, the whole force is lift
    # 1e-9 is round-off on a 1.7e7 N force, not a slack threshold: a form that
    # was only correct at beta = 0 would leave a residual of order sin(beta),
    # i.e. ~0.14 -- eight orders of magnitude above this bound.
    assert abs(float(jnp.dot(force, vel_rel))) / (
        float(jnp.linalg.norm(force)) * float(V)) < 1e-9


def test_drag_is_antiparallel_to_the_relative_wind_at_nonzero_alpha_and_beta():
    vel_rel = jnp.array([190.0, -27.0, 19.0])
    controls = Controls(elevator=jnp.array(0.0), aileron=jnp.array(0.0),
                        rudder=jnp.array(0.0), throttle=jnp.array(0.0))
    ac = B747PA._replace(CL0=jnp.array(0.0), CLa=jnp.array(0.0),
                         CLq=jnp.array(0.0), CLde=jnp.array(0.0),
                         CYb=jnp.array(0.0), CYdr=jnp.array(0.0),
                         e=jnp.array(1e9))
    force, _ = aero_forces_moments(vel_rel, jnp.zeros(3), controls, ac,
                                   jnp.array(1.225), jnp.array(340.0))
    cosang = float(jnp.dot(force, vel_rel)) / (
        float(jnp.linalg.norm(force)) * float(jnp.linalg.norm(vel_rel)))
    assert cosang == pytest.approx(-1.0, abs=1e-12)


def test_the_dcm_really_maps_body_to_ned_for_a_general_attitude():
    """All three Euler angles non-zero -- a transposed DCM survives any test
    with a single non-zero angle."""
    phi, theta, psi = 0.31, -0.22, 0.77
    q = euler_to_quat(jnp.array(phi), jnp.array(theta), jnp.array(psi))
    cr, sr, cp, sp, cy, sy = (math.cos(phi), math.sin(phi), math.cos(theta),
                              math.sin(theta), math.cos(psi), math.sin(psi))
    expected = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr]])
    assert np.allclose(np.asarray(quat_to_dcm(q)), expected, atol=1e-12)


# ===========================================================================
# 5b. The strip calibration at the station count PRODUCTION actually uses
# ===========================================================================

# Measured by the audit. Pure trapezoidal quadrature error on the elliptic
# chord, whose sqrt has infinite slope at the tips. Identical for every
# aircraft, which is what identifies it as quadrature rather than data.
STRIP_CLP_RATIO_BY_N = {9: 0.826435, 19: 0.946285, 51: 0.988155,
                        201: 0.998507, 2001: 0.999953}


@pytest.mark.parametrize("n,ratio", sorted(STRIP_CLP_RATIO_BY_N.items()))
@pytest.mark.parametrize("name", ["boeing747", "boeing747_approach"])
def test_the_strip_integral_only_reproduces_Clp_in_the_continuum_limit(name, n, ratio):
    """KNOWN FLAW, and it is a live one.

    `airframe.calibrated_lift_slope` pins a0 = -8*Clp so that a rigid roll rate
    through the strip integral returns the tabulated Clp EXACTLY. Two docstrings
    say "exactly". It is exact only as the station count goes to infinity.

    `loads.strip_model` builds its stations from `airframe.stations(ac)`, i.e.
    the default `N_SPAN = 9`, where the integral returns 82.6% of Clp -- a
    -17.4% systematic understatement of the strip path's own effect.

    Every existing test of this identity overrides the count: test_airframe.py
    uses 2001, test_loads.py uses 201, test_wind.py's helper defaults to 2001.
    None exercises 9. Independently found by audit Agent C and confirmed here.

    `test_the_default_station_count_has_converged` does NOT cover this: its own
    docstring says it measures "the fitted RATES", i.e. `sampled_rates`, a
    linear slope fit that converges immediately. The strip integral is a
    different quantity with a sqrt-singular integrand and converges far slower.
    """
    ac = REGISTRY[name]
    st = airframe.stations(ac, n_span=n, n_lon=9)
    got = float(wind.strip_clp_from_rate(ac, st, jnp.array(1.0)))
    assert got / float(ac.Clp) == pytest.approx(ratio, abs=2e-5)


def test_the_production_station_count_is_the_one_that_is_wrong():
    """States the defect as a single fact, so it cannot be lost among the
    parametrised rows above. If N_SPAN is ever raised, this test fails and the
    ratio table above must be re-measured -- which is the correct outcome."""
    assert airframe.N_SPAN == 9
    ac = REGISTRY["boeing747"]
    st = airframe.stations(ac)          # exactly what loads.strip_model builds
    got = float(wind.strip_clp_from_rate(ac, st, jnp.array(1.0)))
    assert abs(got / float(ac.Clp) - 1.0) > 0.15, (
        "the strip calibration now holds at the production station count; "
        "AUDIT.md's finding is fixed and this test should be replaced by the "
        "exactness assertion the docstrings already claim")


# ===========================================================================
# 5c. Two defects found by the audit that no shipped test covers
# ===========================================================================

def test_lateral_modes_orders_roll_and_spiral_by_speed_not_by_sign():
    """REPAIRED. `validation.lateral_modes` used to end with

        reals.sort()  # roll subsidence is fast (small tau), spiral is slow

    whose reasoning was right and whose implementation was not: `sort()` orders
    the SIGNED time constants, so an UNSTABLE spiral (positive eigenvalue, hence
    negative tau) sorted to the front and came back as `roll_tau`. It now sorts
    by `abs`, which is what "fast" means.

    The Cherokee is the case that shows it: its spiral root is +0.01938 (tau
    -51.59 s) and its roll subsidence is -2.78143 (tau +0.3595 s). Three of the
    four registry aircraft have a stable spiral and were never affected -- the
    two sorts agree exactly when every value is positive, which is the property
    that makes this repair provably surgical, asserted below.

    Nothing the project publishes moved -- no Cherokee lateral mode is quoted
    anywhere -- but the Cherokee is the microburst aircraft, so the next lateral
    analysis to be run would have hit it. Found by audit Agent A; repaired in
    the remediation pass.
    """
    from flightsim.validation import lateral_modes
    ac = REGISTRY["cherokee"]
    V, H = CRUISE["cherokee"]["airspeed"], CRUISE["cherokee"]["altitude"]
    x, _ = trim(jnp.array(V), jnp.array(H), ac)
    _, roll_tau, spiral_tau = lateral_modes(
        ac, float(x[0]), float(x[1]), float(x[2]), V, H)
    # the spiral really is unstable for this aircraft -- that is what made the
    # signed sort go wrong, and it is still true after the repair
    assert spiral_tau < 0.0, "the Cherokee's spiral is unstable; tau is negative"
    # the fast mode is now returned as the fast mode
    assert roll_tau == pytest.approx(0.3595, rel=0.02)
    assert spiral_tau == pytest.approx(-51.59, rel=0.02)
    assert abs(roll_tau) < abs(spiral_tau), "roll subsidence must be the fast mode"
    # the three stable-spiral aircraft are unaffected
    for name in ("boeing747", "boeing747_approach", "cessna172"):
        a = REGISTRY[name]
        v, h = CRUISE[name]["airspeed"], CRUISE[name]["altitude"]
        xx, _ = trim(jnp.array(v), jnp.array(h), a)
        _, rt, st = lateral_modes(a, float(xx[0]), float(xx[1]), float(xx[2]), v, h)
        assert 0.0 < rt < st, f"{name}: roll {rt} should be fast and positive"


def _aero_power(ac, V, alt, q, de):
    """F_aero.v + M_aero.omega in still air with the throttle shut.

    For this EOM that IS dE/dt: gravity is conservative and enters E as
    potential energy, and with no wind and no thrust the aerodynamic wrench is
    the only other force. A passive airframe must have it <= 0 everywhere.
    """
    from flightsim.atmosphere import speed_of_sound
    rho, a = density(jnp.array(alt)), speed_of_sound(jnp.array(alt))
    vel = jnp.array([V, 0.0, 0.0])
    om = jnp.array([0.0, q, 0.0])
    c = Controls(elevator=jnp.array(de), aileron=jnp.array(0.0),
                 rudder=jnp.array(0.0), throttle=jnp.array(0.0))
    F, M = aero_forces_moments(vel, om, c, ac, rho, a)
    return float(jnp.dot(F, vel) + jnp.dot(M, om))


def test_the_model_admits_states_that_CREATE_energy_in_still_air():
    """KNOWN FLAW, bounded, and outside the declared envelope.

    The aero build-up has no `CD_de` and no explicit `CD_q`; drag responds to
    elevator and pitch rate only through the induced term on total CL. So the
    control-moment power `Cmde*de*q` has no matching drag channel, and at a
    large enough pitch rate it beats the `Cmq` damping power. The aerodynamics
    then do POSITIVE work on the aircraft with the throttle shut.

    Found by audit Agent E; the state below reproduces its figure exactly.
    Bounded by the two tests that follow, and those bounds are what make this a
    documented flaw rather than an alarm.
    """
    ac = REGISTRY["cherokee"]
    alt = CRUISE["cherokee"]["altitude"]
    p = _aero_power(ac, 75.0, alt, -8.256, 25.0 * DEG2RAD)
    assert p > 0.0, "no positive-power state; if this is fixed, see AUDIT.md"
    assert p == pytest.approx(56.965e3, rel=0.02)


def test_only_the_cherokee_has_a_positive_power_region_and_it_needs_201_deg_per_sec():
    """The bound on the finding above, and it is what keeps it harmless.

    At their own cruise conditions, with the elevator at its own limit and the
    pitch rate swept to 30 rad/s, the Cessna and BOTH 747s have no
    positive-power point at all. The Cherokee's threshold is |q| = 3.51 rad/s =
    201 deg/s, implying a 16 m/s wingtip vertical velocity against a 50 m/s
    airspeed -- a tip incidence perturbation of ~18 deg, far outside the linear
    range PROJECT.md section 7 caps at 10-12 deg.

    So the region exists in the coefficient model and lies outside the envelope
    the model declares for itself.
    """
    for name in ("cessna172", "boeing747", "boeing747_approach"):
        ac = REGISTRY[name]
        V, alt = CRUISE[name]["airspeed"], CRUISE[name]["altitude"]
        de = float(ac.elevator_limit)
        worst = max(_aero_power(ac, V, alt, q, de)
                    for q in np.linspace(0.0, -30.0, 301))
        assert worst < 0.0, f"{name} now has a positive-power state"

    ac = REGISTRY["cherokee"]
    V, alt = CRUISE["cherokee"]["airspeed"], CRUISE["cherokee"]["altitude"]
    de = float(ac.elevator_limit)
    # The positive-power region is a BAND, not a half-line: at large |q| the
    # Cmq damping power (~q^2) beats the control-moment power (~q) again. So
    # scan for the first sign change and bisect inside that bracket.
    grid = np.linspace(0.0, -30.0, 301)
    power = np.array([_aero_power(ac, V, alt, q, de) for q in grid])
    first = int(np.argmax(power > 0.0))
    assert power[first] > 0.0, "the Cherokee no longer has a positive-power band"
    lo, hi = grid[first - 1], grid[first]
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if _aero_power(ac, V, alt, mid, de) < 0 else (lo, mid)
    qstar = 0.5 * (lo + hi)
    assert qstar == pytest.approx(-3.510, abs=0.01)
    assert abs(qstar) * float(ac.b) / 2.0 == pytest.approx(16.0, abs=0.5)


def test_a_fixed_control_pull_from_trim_does_not_reach_the_energy_gaining_region():
    """The other half, and it is a DISAGREEMENT recorded rather than resolved.

    Agent E reports reaching the positive-power region from trim on plain full
    elevator with the throttle shut, gaining +691 J. This run does not: full
    elevator, throttle 0, 6 s at dt = 0.002 reaches |q| = 3.63 rad/s -- past the
    3.51 threshold -- and gains energy at ZERO of 2999 steps, because by then
    |alpha| has reached 119 deg and drag dominates.

    Agent E's own wider search agrees with this run rather than with its own
    headline: it reports P_aero < 0 at 0 of 80,000 randomised reachable states,
    and max(E - E0) = +0 across ~120 still-air runs including five deliberate
    energy-pumping feedback laws.

    What is pinned here is the conservative claim: no energy is gained on this
    trajectory. If that changes, the model has become reachably nonphysical and
    the bound in AUDIT.md no longer holds.
    """
    from flightsim.integrate import init_sim, rollout
    from flightsim.trim import trimmed_controls, trimmed_state
    ac = REGISTRY["cherokee"]
    V, alt = CRUISE["cherokee"]["airspeed"], CRUISE["cherokee"]["altitude"]
    x, _ = trim(jnp.array(V), jnp.array(alt), ac)
    state = trimmed_state(x[0], jnp.array(V), jnp.array(alt))
    controls = trimmed_controls(jnp.array(-float(ac.elevator_limit)), jnp.array(0.0))
    dt, n = 0.002, 3000
    _, traj = rollout(init_sim(state, jax.random.PRNGKey(0)),
                      controls, jnp.array(dt), ac, n)
    pos, vb, om = (np.asarray(traj.pos_ned), np.asarray(traj.vel_body),
                   np.asarray(traj.omega))
    m, g = float(ac.mass), 9.80665
    E = 0.5 * m * np.sum(vb**2, axis=1) + m * g * (-pos[:, 2])
    assert np.abs(om[:, 1]).max() > 3.51, "the run must reach the threshold rate"
    assert (np.diff(E) > 0).sum() == 0, "energy was gained on a reachable trajectory"


def test_the_wind_hold_costs_the_headline_figure_more_than_E4_bounds_it():
    """KNOWN FLAW, re-bounded. Corrects the magnitude in ASSUMPTIONS.md E4.

    E4 bounds the once-per-step wind hold by an h-vs-h/2 POSITION difference
    (0.0169 m -> 0.0024 m/s of gust, "~1e-4 relative") and concludes "No result
    the project quotes is affected."

    That measures the wrong quantity. h-vs-h/2 is the DISCRETISATION error with
    the hold still in place; the SCHEME error is hold-vs-per-stage at the same
    dt. Measured on the number `scripts/vortex.py` actually prints, the in-core
    Fig-8 d(theta):

        dt = 0.02   hold 2.2596 deg   per-stage 2.2230 deg   -1.62%
        dt = 0.01   hold 2.2400 deg   per-stage 2.2216 deg   -0.82%

    So the cost at the production step is ~0.8%, not ~1e-4 -- about 80x the
    register's figure, and it halves with dt as an O(h) error must.

    It changes no CONCLUSION: PROJECT.md section 5 caps vortex claims at
    orderings and puts +-25% bands on the identified parameters. What it does
    mean is that the quoted 2.240 deg is not good to four significant figures --
    its last two digits are scheme-dependent. Found by audit Agent D, confirmed
    independently here.
    """
    from flightsim.dynamics import derivatives
    from flightsim.integrate import rk4_step
    from flightsim.state import quat_normalize, quat_to_euler
    from flightsim.trim import trimmed_controls, trimmed_state
    from flightsim import vortex_viz

    V, H = CRUISE["boeing747"]["airspeed"], CRUISE["boeing747"]["altitude"]
    hb = wind.PARKS_CASES["hannibal"]
    r0 = hb["r0"]
    arr = wind.VortexArray(north=jnp.array([0.0, hb["spacing"]]),
                           down=jnp.array([-H, -H]),
                           r0=jnp.array(r0), v0=jnp.array(hb["v0"]))
    field = lambda p: wind.vortex_wind(p, arr)  # noqa: E731
    dt, seconds = 0.02, 40.0

    enc = vortex_viz.fly(B747, field, V, H, label="v", start_north=-40 * r0,
                         seconds=seconds, dt=dt, window=(-r0, r0),
                         window_name="core")
    w = np.asarray(enc.window)
    th = np.asarray(enc.theta)
    held = math.degrees(th[w].max() - th[w].min())

    x, _ = trim(jnp.array(V), jnp.array(H), B747)
    state = trimmed_state(x[0], jnp.array(V), jnp.array(H))
    state = state._replace(pos_ned=state.pos_ned.at[0].set(-40.0 * r0))
    controls = trimmed_controls(x[1], x[2])

    def body(s, _):
        def f(y):
            return derivatives(y, controls, B747, field(y.pos_ned),
                               wind.gust_rates(y.pos_ned, y.quat, field))
        ns = rk4_step(f, s, jnp.array(dt))
        ns = ns._replace(quat=quat_normalize(ns.quat))
        return ns, (ns.pos_ned[0], quat_to_euler(ns.quat)[1])

    _, (north, theta) = jax.lax.scan(body, state, None,
                                     length=int(round(seconds / dt)))
    north, theta = np.asarray(north), np.asarray(theta)
    wp = np.abs(north) <= r0
    per_stage = math.degrees(theta[wp].max() - theta[wp].min())

    rel = (per_stage - held) / held
    assert held == pytest.approx(2.2596, abs=0.005)
    assert per_stage == pytest.approx(2.2230, abs=0.005)
    assert rel == pytest.approx(-0.0162, abs=0.004)
    assert abs(rel) > 1e-3, (
        "the wind hold now costs less than 0.1% at dt=0.02; ASSUMPTIONS.md E4's "
        "~1e-4 bound may have become correct and this test should be re-measured")


def test_the_longitudinal_station_set_is_one_sided_and_biases_the_pitch_secant():
    """KNOWN FLAW, newly bounded. Qualifies ASSUMPTIONS.md section E2.

    `airframe.stations` returns `longitudinal = linspace(-arm, 0, N)`: every
    station is AFT of the CG, so the set's centroid is at x = -16.75 m, not 0.
    A least-squares slope over a one-sided set estimates the derivative at the
    set's CENTROID, so `sampled_rates`' pitch channel is a BACKWARD secant
    carrying an O(arm/2 * f'') bias -- not the symmetric O(arm^2 f''') error a
    centred set would give. The span set IS centred, so roll is unaffected.

    Consequence for the register's E2 profile, which is measured at POSITIVE
    fractions of r0 only: its headline claim that the correction is "exactly
    zero inside the core, not merely small" holds on the DOWNSTREAM half of a
    core traverse and not on the upstream half. At -0.99 r0 the CG is inside the
    core while the tail, 33.5 m behind it, is outside -- so the fit straddles
    the gradient discontinuity and the correction is 1.80 V0/r0, not zero.

    Found by audit Agent D, confirmed here. The DEFAULT wind path uses
    `gust_rates` (the tangent), not `sampled_rates`, so nothing the project
    publishes moves; what changes is the size of the error the register
    attributes to the point model on the upstream half.
    """
    from flightsim.state import euler_to_quat
    ac = REGISTRY["boeing747"]
    st = airframe.stations(ac)
    lon = np.asarray(st.longitudinal)
    assert lon.max() == pytest.approx(0.0, abs=1e-12)
    assert lon.min() < -30.0
    assert lon.mean() == pytest.approx(-16.749, abs=0.01)      # one-sided
    assert np.asarray(st.span).mean() == pytest.approx(0.0, abs=1e-12)  # centred

    hb = wind.PARKS_CASES["hannibal"]
    r0, v0 = hb["r0"], hb["v0"]
    arr = wind.VortexArray(north=jnp.array([0.0]), down=jnp.array([0.0]),
                           r0=jnp.array(r0), v0=jnp.array(v0))
    field = lambda p: wind.vortex_wind(p, arr)  # noqa: E731
    quat = euler_to_quat(jnp.array(0.0), jnp.array(0.0), jnp.array(0.0))

    def correction(frac):
        p = jnp.array([frac * r0, 0.0, -1.0])
        t = float(wind.gust_rates(p, quat, field)[1])
        s = float(wind.sampled_rates(p, quat, field, st)[1])
        return abs(s - t) / (v0 / r0)

    # downstream half, inside the core: the register's "exactly zero"
    assert correction(+0.50) < 1e-9
    assert correction(+0.99) < 1e-9
    # upstream half, inside the core: NOT zero
    assert correction(-0.99) == pytest.approx(1.80, abs=0.05)
    # and the boundary values are asymmetric for the same reason
    assert correction(+1.00) == pytest.approx(2.00, abs=0.02)
    assert correction(-1.00) == pytest.approx(0.156, abs=0.02)


def test_trim_returns_absurd_roots_from_plausible_guesses_on_real_aircraft():
    """KNOWN FLAW, and wider than PROJECT.md section 5 records.

    Section 5 documents `trim.trim` converging to absurd roots for DEGENERATE
    coefficients (CLa = 1e-4). This shows it happens on UNMODIFIED registry data
    at the aircraft's own cruise condition, purely from the starting guess:
    `trim` is a fixed 40-iteration Newton in `lax.scan` with no convergence
    test, no bounds and no reporting, so it returns whatever root it lands on.

    `trim.is_physical` catches these, and since the remediation pass it checks
    elevator and throttle as well as alpha. That makes the gate wider; it does
    not make `trim` report anything. THE FINDING PINNED HERE IS ABOUT `trim`,
    not about the gate: an unbounded fixed-iteration Newton that returns a root
    hundreds of degrees away in silence is still what it does, and a caller who
    never asks `is_physical` still never learns.
    """
    from flightsim.trim import is_physical
    ac = REGISTRY["boeing747"]
    V, H = CRUISE["boeing747"]["airspeed"], CRUISE["boeing747"]["altitude"]
    absurd = []
    for a0 in np.radians([-60.0, -30.0, 30.0, 60.0]):
        for de in np.radians([-20.0, 20.0]):
            for th in (0.1, 0.9):
                x, r = trim(jnp.array(V), jnp.array(H), ac,
                            guess=jnp.array([a0, de, th]))
                if float(jnp.linalg.norm(r)) < 1e-9 and not is_physical(x, ac):
                    absurd.append(np.degrees(float(x[0])))
    assert absurd, (
        "no absurd-but-converged root found; if trim now rejects them this "
        "test should be replaced by the positive assertion")
    assert max(abs(a) for a in absurd) > 90.0


# ===========================================================================
# 6. Enforce the claim PROJECT.md makes and test_provenance.py does not
# ===========================================================================

# PROJECT.md section 2 and provenance.py both used to say "a constant with no
# ledger entry fails the build", and no test walked the source modules to check
# it; every test in test_provenance.py iterates LEDGER itself. This is the
# missing direction. Both documents have since been corrected to describe what
# is actually enforced, which is this test.
#
# The set below is the audit's baseline, pinned: these module-level constants
# carry NO ledger entry. The test fails if a NEW one appears, which is what the
# corrected claim promises.
#
# IT MAY ONLY EVER SHRINK. Widening it to admit a new constant is the one move
# that would make this check meaningless, so `test_the_recorded_baseline_has_no
# _stale_entries` fails if a name here has since been ledgered and not removed.
# Seven were removed in the remediation pass -- the four Oseguera & Bowles
# microburst constants, whose paper IS held in refs/, plus three DECLARED
# numbers whose sensitivity was already written at the constant.
KNOWN_UNLEDGERED = {
    ("aero", "_MDD_OFFSET"),
    ("airframe", "TAIL_ARM_BAND"),
    ("airframe", "N_SPAN"),
    ("airframe", "N_LON"),
    ("airframe", "_SENSITIVITY_TAPER"),
    ("trim", "INITIAL_GUESS"),
    ("wind", "UPDRAFT_W0"),
    ("wind", "UPDRAFT_SECONDS"),
    ("atmosphere", "T0"), ("atmosphere", "P0"), ("atmosphere", "RHO0"),
    ("atmosphere", "LAPSE"), ("atmosphere", "H_TROPOPAUSE"),
    ("atmosphere", "T_TROPOPAUSE"), ("atmosphere", "G0"),
    ("atmosphere", "R_AIR"), ("atmosphere", "GAMMA"),
    ("atmosphere", "P_TROPOPAUSE"),
}


def _module_constants():
    """Module-level numeric constants ASSIGNED IN each physics module.

    Parsed from the source with `ast` rather than read off the module object,
    so a name imported from elsewhere (`aero.RHO0`, `trim.G0`, `wind.FT2M`) is
    attributed to the module that defines it and is not double-counted.
    """
    import ast
    import pathlib
    found = set()
    root = pathlib.Path(__file__).resolve().parents[1]
    for name in ("aero", "airframe", "atmosphere", "trim", "wind"):
        tree = ast.parse((root / f"{name}.py").read_text(encoding="utf-8"))
        for node in tree.body:
            targets = ([node.target] if isinstance(node, ast.AnnAssign)
                       else getattr(node, "targets", []))
            for t in targets:
                if not isinstance(t, ast.Name):
                    continue
                if _is_numeric_literal(node.value):
                    found.add((name, t.id))
    return found


def _is_numeric_literal(node):
    """A constant expression built only from numeric literals and arithmetic."""
    import ast
    if node is None:
        return False
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant):
            if not isinstance(sub.value, (int, float)) or isinstance(sub.value, bool):
                return False
        elif isinstance(sub, (ast.BinOp, ast.UnaryOp, ast.Tuple, ast.Load,
                              ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow,
                              ast.USub, ast.UAdd, ast.Name)):
            continue
        else:
            return False
    return any(isinstance(sub, ast.Constant) for sub in ast.walk(node))


def test_the_provenance_ledger_does_not_cover_the_source_modules():
    """THE MISSING DIRECTION, and now the thing the documents actually claim.

    PROJECT.md section 2 and provenance.py's own docstring both used to state
    that "a constant added without a ledger entry fails the build" while nothing
    enforced the source -> ledger direction: every test in test_provenance.py
    iterates LEDGER.items() or indexes known keys, so the entries were checked
    for internal consistency and nothing checked COVERAGE.

    This test is that direction. A NEW unledgered module-level constant in one
    of the five modules below fails it. Both documents have been rewritten to
    describe exactly this and no more, so the claim and the check now agree.
    Closing the remaining gap means adding ledger entries and shrinking
    KNOWN_UNLEDGERED -- never widening it.
    """
    ledgered = {k.split(".", 1)[-1] for k in provenance.LEDGER}
    uncovered = {(m, a) for (m, a) in _module_constants() if a not in ledgered}
    new = uncovered - KNOWN_UNLEDGERED
    assert not new, (
        f"constants with no provenance ledger entry that the audit did not "
        f"record: {sorted(new)}. PROJECT.md section 2 says this must fail the "
        f"build; add a provenance.LEDGER entry.")


def test_the_recorded_baseline_has_no_stale_entries():
    """KNOWN_UNLEDGERED may only ever shrink, so it must not accumulate names
    that have since been ledgered. Without this, the honest half of the repair --
    removing a name when its entry is added -- would be optional, and the set
    would drift into a list of things that were once true.

    Only that direction is asserted. The set also carries three names
    `_module_constants` cannot see -- `T_TROPOPAUSE` and `P_TROPOPAUSE` are
    built from other names with no literal in them, `INITIAL_GUESS` is a
    `jnp.array` call -- and those are the audit's record of what it found by
    hand, not a claim about what the parser reaches.
    """
    ledgered = {k.split(".", 1)[-1] for k in provenance.LEDGER}
    stale = {(m, a) for (m, a) in KNOWN_UNLEDGERED if a in ledgered}
    assert not stale, (
        f"these are in the ledger and still recorded as unledgered: "
        f"{sorted(stale)}. Remove them from KNOWN_UNLEDGERED.")


def test_the_ledger_covers_the_constants_it_does_claim():
    """The other direction, which test_provenance.py already asserts for the
    graph but not for the airframe module: every DERIVED chain must bottom out
    in something SOURCED or DECLARED."""
    for name, entry in provenance.LEDGER.items():
        seen, stack = set(), [name]
        bottoms = set()
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            e = provenance.LEDGER[cur]
            if not e.inputs:
                bottoms.add(e.category)
            stack.extend(e.inputs)
        assert bottoms & {"SOURCED", "DECLARED"}, (
            f"{name} bottoms out in {bottoms}, not in SOURCED or DECLARED")


def test_the_derived_tail_arm_is_still_the_ratio_the_ledger_states():
    """provenance.LEDGER['b747.l_eff'] states 4.0241 chords = 109.90 ft."""
    arm = float(airframe.effective_tail_arm(B747))
    assert arm == pytest.approx(4.0241, abs=5e-4)
    assert arm * float(B747.c) / FT2M == pytest.approx(109.90, abs=0.05)


# ===========================================================================
# PHASE 2E -- FALSIFICATION
#
# The energy seam, the absent ground, the airspeed floor, the silent NaN, and
# the two tail arms. See AUDIT.md section 2.5 and findings 33-41, and
# ASSUMPTIONS_AUDIT.md U15-U19.
#
# Every number below was re-derived by this auditor before it was written down;
# where it disagrees with audit_evidence/E-falsification.md, AUDIT.md section 3b
# records why.
# ===========================================================================

CHEROKEE = REGISTRY["cherokee"]
CESSNA = REGISTRY["cessna172"]

# The reproduction state for the energy violation. Cherokee at its own cruise
# altitude, elevator at its 25 deg limit, throttle shut, still air.
_E1_ALTITUDE = 1499.6
_E1_AIRSPEED = 75.0
_E1_PITCH_RATE = -8.256
_E1_ELEVATOR = 0.4363


def _still_air_power(ac, V, q, de, altitude):
    """F.v + M.omega for the aero build-up alone: throttle shut, no wind.

    In motionless air the fluid starts at rest, so a quasi-steady model of a
    body moving through it can only LOSE energy. This must be <= 0.
    """
    from flightsim.atmosphere import speed_of_sound

    vel = jnp.array([V, 0.0, 0.0])
    omega = jnp.array([0.0, q, 0.0])
    controls = Controls(
        elevator=jnp.array(de), aileron=jnp.array(0.0),
        rudder=jnp.array(0.0), throttle=jnp.array(0.0),
    )
    force, moment = aero_forces_moments(
        vel, omega, controls, ac, density(altitude), speed_of_sound(altitude)
    )
    return float(force @ vel + moment @ omega)


def test_the_model_creates_mechanical_energy_in_still_air():
    """KNOWN FLAW, bounded, and it is a real violation of energy conservation.

    WRITTEN TO FAIL WHEN FIXED. If the lift-tilt term of AUDIT.md section 2.5 is
    ever restored, this test goes red and should be REPLACED by an assertion
    that aerodynamic power is non-positive everywhere -- not loosened.

    In motionless, uniform air with the throttle shut there is no energy source
    but the aircraft itself, so P_aero = F.v + M.omega must be <= 0. It is not.
    The elevator's pitching moment does work on the airframe with nothing
    opposing it, because the model applies every lift increment perpendicular to
    the relative wind AT THE CG rather than at the surface producing it.
    """
    power = _still_air_power(
        CHEROKEE, _E1_AIRSPEED, _E1_PITCH_RATE, _E1_ELEVATOR, _E1_ALTITUDE
    )
    assert power == pytest.approx(56927.7, rel=1e-3), (
        "the energy-creation state moved; AUDIT.md section 2.5 measured "
        f"+56927.7 W and this is {power:.1f} W")
    assert power > 0, "if this is now <= 0 the flaw is fixed -- replace this test"


def test_the_pitch_damping_channel_alone_is_always_dissipative():
    """VERIFICATION. The damping term cannot be the culprit, and this pins why.

    The Cmq contribution to moment power is qbar*S*c*Cmq*(c/2V)*q^2, which is
    negative-definite in q because Cmq < 0. So the sign change in
    `test_the_model_creates_mechanical_energy_in_still_air` is attributable to
    the CONTROL term and not to the rate term. Checked on all four aircraft.
    """
    for name in REGISTRY:
        ac = REGISTRY[name]
        assert float(ac.Cmq) < 0.0, f"{name} has Cmq >= 0"
        for q in (-8.0, -1.0, 1.0, 8.0):
            q_hat = q * float(ac.c) / (2.0 * 100.0)
            assert float(ac.Cmq) * q_hat * q <= 0.0


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_the_elevator_moment_power_is_cancelled_by_its_own_lift_tilt(name):
    """VERIFICATION -- an algebraic identity, and the audit's attribution.

    A surface at arm l sees a local relative wind tilted by eps = q*l/V. Its
    lift, tilted by eps, acquires a streamwise component dL*sin(eps) whose power
    is dL*eps*V to first order. The elevator's own two derivatives fix its arm,
    l_de = -(Cmde/CLde)*c, and substituting gives

        P_tilt = qbar*S*CLde*de * (q*l_de/V) * V
               = -qbar*S*c*Cmde*de*q
               = -(elevator moment power)

    exactly. That is the term `aero.aero_forces_moments` omits, and restoring it
    is what takes the violating region to zero (next test). The identity needs
    NO new constant, which is why it supersedes the missing-CD_de explanation
    Phase 2E proposed -- see AUDIT.md section 3b.
    """
    ac = REGISTRY[name]
    rng = np.random.default_rng(20260817)
    worst = 0.0
    for _ in range(200):
        V = float(rng.uniform(40.0, 280.0))
        q = float(rng.uniform(-8.0, 8.0))
        de = float(rng.uniform(-0.4, 0.4))
        qbar = 0.5 * 1.0 * V**2
        moment_power = qbar * float(ac.S) * float(ac.c) * float(ac.Cmde) * de * q
        arm = -float(ac.Cmde) / float(ac.CLde) * float(ac.c)
        tilt_power = (qbar * float(ac.S) * float(ac.CLde) * de) * (q * arm / V) * V
        if abs(moment_power) > 1e-6:
            worst = max(worst, abs(moment_power + tilt_power) / abs(moment_power))
    assert worst < 1e-14, f"the cancellation is no longer exact: {worst:.3e}"


def test_restoring_the_lift_tilt_removes_the_energy_violation():
    """KNOWN FLAW, bounded -- and this is the demonstration behind the bound.

    Over a (V, q, elevator) grid the shipped model has 1190 states with positive
    aerodynamic power. Adding the one omitted tilt term takes that to EXACTLY
    zero. The audit reports this as a demonstrated attribution rather than a
    plausible cause because of this test.
    """
    ac = CHEROKEE
    shipped = restored = 0
    for V in np.linspace(0.4 * 50.0, 1.6 * 50.0, 40):
        for q in np.linspace(-10.0, 10.0, 80):
            for de in (-float(ac.elevator_limit), float(ac.elevator_limit)):
                p = _still_air_power(ac, float(V), float(q), de, _E1_ALTITUDE)
                qbar = 0.5 * float(density(_E1_ALTITUDE)) * V**2
                arm = -float(ac.Cmde) / float(ac.CLde) * float(ac.c)
                tilt = (qbar * float(ac.S) * float(ac.CLde) * de) * (q * arm / V) * V
                shipped += p > 0
                restored += (p + tilt) > 0
    assert shipped == 1190, f"the violating region moved: {shipped} points"
    assert restored == 0, (
        f"restoring the lift tilt left {restored} violating states; the "
        "attribution in AUDIT.md section 2.5 no longer holds")


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_aerodynamic_power_is_dissipative_inside_the_declared_envelope(name):
    """BOUND on the energy flaw: it is unreachable within the declared scope.

    PROJECT.md section 7 puts the linear-aero ceiling at |alpha| ~ 10-12 deg.
    Inside |alpha| <= 12 deg, |q| <= 0.5 rad/s and V in [0.5, 1.6] Vcruise, with
    the elevator free to its limits, aerodynamic power is strictly negative --
    0 positive states out of 2000 per aircraft. This is what makes the violation
    a bounded flaw rather than a live defect in anything the project reports.
    """
    from flightsim.atmosphere import speed_of_sound

    ac = REGISTRY[name]
    Vc = float(CRUISE[name]["airspeed"])
    altitude = float(CRUISE[name]["altitude"])
    rng = np.random.default_rng(4242)
    rho, a_sound = density(altitude), speed_of_sound(altitude)
    worst = -np.inf
    for _ in range(2000):
        V = float(rng.uniform(0.5 * Vc, 1.6 * Vc))
        alpha = float(rng.uniform(-math.radians(12), math.radians(12)))
        q = float(rng.uniform(-0.5, 0.5))
        de = float(rng.uniform(-float(ac.elevator_limit), float(ac.elevator_limit)))
        vel = jnp.array([V * math.cos(alpha), 0.0, V * math.sin(alpha)])
        omega = jnp.array([0.0, q, 0.0])
        controls = Controls(jnp.array(de), jnp.array(0.0), jnp.array(0.0),
                            jnp.array(0.0))
        force, moment = aero_forces_moments(vel, omega, controls, ac, rho, a_sound)
        worst = max(worst, float(force @ vel + moment @ omega))
    assert worst < 0.0, (
        f"{name} gains energy inside the declared envelope: {worst:+.1f} W. "
        "The bound in AUDIT.md section 2.5 no longer holds.")


# ---------------------------------------------------------------------------
# The two tail arms. AUDIT.md finding 35, ASSUMPTIONS_AUDIT.md U19.
# ---------------------------------------------------------------------------

# l/c from the RATE pair and from the CONTROL pair. Same geometry, two
# independent routes, both already in the registry.
TAIL_ARM_BY_ROUTE = {
    "boeing747": (4.0241, 3.9694),
    "boeing747_approach": (3.8519, 3.9645),
    "cherokee": (1.2802, 2.5621),
    "cessna172": (0.8558, 2.4681),
}


@pytest.mark.parametrize("name", sorted(TAIL_ARM_BY_ROUTE))
def test_the_two_independent_tail_arm_estimates(name):
    """VERIFICATION for the CR-2144 pair, KNOWN FLAW for the light pair.

    `airframe.effective_tail_arm` takes l/c = -Cmq/CLq. The control pair gives
    the same geometry independently as l/c = -Cmde/CLde. Nothing in the project
    compares them, and the comparison is free.

    The two CR-2144 aircraft agree to 1.4% and 2.9% -- an independent
    corroboration of that transcription which the project does not currently
    claim. The two aircraft whose cited source file does not exist in this
    repository (AUDIT.md finding 20) disagree by factors of 2.0 and 2.9.
    """
    ac = REGISTRY[name]
    rate = -float(ac.Cmq) / float(ac.CLq)
    control = -float(ac.Cmde) / float(ac.CLde)
    expected_rate, expected_control = TAIL_ARM_BY_ROUTE[name]
    assert rate == pytest.approx(expected_rate, abs=5e-4)
    assert control == pytest.approx(expected_control, abs=5e-4)


def test_the_tail_arm_routes_agree_only_where_the_source_is_held():
    """KNOWN FLAW. The disagreement tracks source availability exactly.

    This is the only internal evidence the audit found bearing on the quality of
    the Cherokee and Cessna derivative sets, whose cited source
    (`aircraft_data_validated.py`) is not in the repository.
    """
    for name in ("boeing747", "boeing747_approach"):
        rate, control = TAIL_ARM_BY_ROUTE[name]
        assert abs(control / rate - 1.0) < 0.03, (
            f"{name}: the two CR-2144-sourced routes have diverged")
    for name in ("cherokee", "cessna172"):
        rate, control = TAIL_ARM_BY_ROUTE[name]
        assert control / rate > 1.9, (
            f"{name}: the routes now agree -- if the derivative data was "
            "corrected, update AUDIT.md finding 35 and ASSUMPTIONS_AUDIT.md U19")


# ---------------------------------------------------------------------------
# The airspeed floor. AUDIT.md finding 40, ASSUMPTIONS_AUDIT.md U17.
# ---------------------------------------------------------------------------

# The force each aircraft used to make out of still air, at its own cruise
# altitude, before the floor was confined to the quantities that divide by V.
# Kept as the size of the defect that was repaired, not as an expectation.
FORCE_AT_ZERO_AIRSPEED_N_BEFORE_REPAIR = {
    "boeing747": 19.8620, "boeing747_approach": 170.7342,
    "cherokee": 4.2959, "cessna172": 1.4431,
}


@pytest.mark.parametrize("name", sorted(FORCE_AT_ZERO_AIRSPEED_N_BEFORE_REPAIR))
def test_aerodynamic_force_is_exactly_zero_at_exactly_zero_airspeed(name):
    """REPAIRED. `aero.V_MIN = 1.0` used to floor V before `qbar` was formed, so
    a stationary aircraft reported dynamic pressure it did not have and the
    residual CL0 term produced a force out of still air -- 1.44 N to 170.73 N,
    depending on aircraft and altitude. Free fall therefore did not read n_z = 0.

    `qbar` is now built from the true `norm(vel_rel)`. The floor remains where
    something divides by V: beta and the three non-dimensional rates.

    EXACTLY zero is asserted rather than approximately zero. qbar is a factor of
    every term in both the force and the moment, so at V = 0 the whole build-up
    is multiplied by an exact zero and nothing is left to round.
    """
    from flightsim.atmosphere import speed_of_sound

    ac = REGISTRY[name]
    altitude = float(CRUISE[name]["altitude"])
    force, moment = aero_forces_moments(
        jnp.zeros(3), jnp.zeros(3),
        Controls(jnp.array(0.0), jnp.array(0.0), jnp.array(0.0), jnp.array(0.0)),
        ac, density(altitude), speed_of_sound(altitude),
    )
    assert float(jnp.linalg.norm(force)) == 0.0
    assert float(jnp.linalg.norm(moment)) == 0.0
    # and the defect it replaced was real, at the size the audit measured
    assert FORCE_AT_ZERO_AIRSPEED_N_BEFORE_REPAIR[name] > 1.0


def test_free_fall_reads_exactly_zero_load_factor_with_the_aerodynamics_LIVE():
    """The consequence the previous test exists for, at the level a user sees.

    Before the repair, zeroing the aero coefficients gave exactly -0.0 while the
    live aircraft gave +7.0e-6 to +4.0e-4 -- which is what localised the defect
    to the floor rather than to `load_factor`. Both routes now agree exactly, so
    the localisation no longer has anything to separate.
    """
    from flightsim.dynamics import load_factor

    controls = Controls(jnp.array(0.0), jnp.array(0.0), jnp.array(0.0), jnp.array(0.0))
    for name in sorted(REGISTRY):
        ac = REGISTRY[name]
        altitude = float(CRUISE[name]["altitude"])
        state = State(
            pos_ned=jnp.array([0.0, 0.0, -altitude]), vel_body=jnp.zeros(3),
            quat=euler_to_quat(jnp.array(0.0), jnp.array(0.0), jnp.array(0.0)),
            omega=jnp.zeros(3),
        )
        zeroed = ac._replace(
            CL0=jnp.array(0.0), CLa=jnp.array(0.0), CLq=jnp.array(0.0),
            CLde=jnp.array(0.0), CD0=jnp.array(0.0),
        )
        live = float(load_factor(state, controls, ac, jnp.zeros(3), jnp.zeros(3)))
        off = float(load_factor(state, controls, zeroed, jnp.zeros(3), jnp.zeros(3)))
        assert live == 0.0, f"{name}: aero-live free fall reads n_z = {live!r}"
        assert off == 0.0, f"{name}: aero-zeroed free fall reads n_z = {off!r}"
        assert math.copysign(1.0, live) == math.copysign(1.0, off) == -1.0


def test_the_airspeed_floor_no_longer_reaches_dynamic_pressure():
    """REPAIRED, and this is the assertion that makes the repair surgical.

    `jnp.maximum(x, 1.0)` returns x EXACTLY for x >= 1, so confining the floor
    to the quantities that divide by V cannot move anything at or above 1 m/s.
    That is the whole safety argument for the change, so it is asserted rather
    than reasoned about: below the floor the force now follows the true V^2, and
    at 1 m/s and above the two expressions coincide bit for bit.

    `air_data` still REPORTS the floored airspeed -- that is what keeps alpha,
    beta and the rates finite at rest, it is what `sensors.sense` shows on the
    ASI, and `test_aero.test_airspeed_floor_prevents_nan_at_zero_velocity` pins
    it. The repair separated the reported airspeed from the force path; it did
    not remove the floor.
    """
    from flightsim.aero import V_MIN
    from flightsim.atmosphere import speed_of_sound

    ac = REGISTRY["boeing747"]
    altitude = float(CRUISE["boeing747"]["altitude"])
    rho, a_sound = density(altitude), speed_of_sound(altitude)
    controls = Controls(jnp.array(0.05), jnp.array(0.0), jnp.array(0.0), jnp.array(0.0))

    def force(V):
        f, _ = aero_forces_moments(
            jnp.array([V, 0.0, 0.0]), jnp.zeros(3), controls, ac, rho, a_sound)
        return float(jnp.linalg.norm(f))

    # the reported airspeed is still floored -- unchanged, and still the guard
    for V in (0.9, 0.5, 0.1):
        reported, _, _ = air_data(jnp.array([V, 0.0, 0.0]))
        assert float(reported) == pytest.approx(V_MIN)

    # but the force below the floor now follows the TRUE dynamic pressure. The
    # coefficients are constant along this line (alpha = beta = 0, no rates), so
    # the ratio is exactly (V/V_ref)^2 rather than approximately so.
    reference = force(1.0)
    for V in (0.9, 0.5, 0.1):
        assert force(V) / reference == pytest.approx(V**2, rel=1e-12), (
            f"at {V} m/s the force no longer scales with the true V^2")

    # and above the floor the change is bit-for-bit invisible
    assert force(0.0) == 0.0
    for V in (1.0, 1.0 + 2**-52, 2.0, 50.0, 265.0):
        floored = float(jnp.maximum(jnp.array(V), V_MIN))
        assert floored == V, f"the floor binds at {V}, which it must not"


# ---------------------------------------------------------------------------
# The silent NaN. AUDIT.md finding 38, ASSUMPTIONS_AUDIT.md U18.
# ---------------------------------------------------------------------------


def test_a_control_channel_with_zero_authority_returns_nan_in_silence():
    """KNOWN FLAW. A declared modelling choice with an undeclared consequence.

    The Cessna's CYdr = Cldr = Cndr = 0 zeroes the rudder column of any control
    Jacobian, so the matrix is singular and `jnp.linalg.solve` returns non-finite
    entries WITHOUT raising. `trim` does not carry rudder as an unknown so it is
    unaffected; a steady-turn solver would be, and nothing warns it.

    WRITTEN TO FAIL WHEN FIXED: if the Cessna ever gains rudder derivatives, or
    the solvers gain a singularity check, this goes red.

    NOTE ON THE GUARD THAT DOES EXIST. `conftest.py` sets `jax_debug_nans`, so
    under pytest this NaN raises a `FloatingPointError` rather than propagating.
    That is a real mitigation and the audit records it -- but it is a TEST-TIME
    setting only; nothing sets it for `scripts/` or for a library caller. This
    test therefore turns it off deliberately, to measure the behaviour a
    non-test caller actually gets.
    """
    from flightsim.dynamics import derivatives
    from flightsim.trim import trimmed_state

    ac = CESSNA
    assert float(ac.CYdr) == 0.0 and float(ac.Cldr) == 0.0 and float(ac.Cndr) == 0.0

    def angular_accel(u):
        state = trimmed_state(jnp.array(0.05), jnp.array(60.0), jnp.array(2500.0))
        return derivatives(
            state, Controls(u[0], u[1], u[2], jnp.array(0.5)), ac,
            jnp.zeros(3), jnp.zeros(3),
        ).omega

    jacobian = jax.jacfwd(angular_accel)(jnp.zeros(3))
    assert float(jnp.linalg.det(jacobian)) == 0.0

    jax.config.update("jax_debug_nans", False)
    try:
        solution = jnp.linalg.solve(jacobian, jnp.ones(3))
        finite = bool(jnp.all(jnp.isfinite(solution)))
    finally:
        jax.config.update("jax_debug_nans", True)
    assert not finite, (
        "the singular solve now returns finite values -- if a guard was added, "
        "replace this test with one asserting the guard fires")


# ---------------------------------------------------------------------------
# The absent ground. AUDIT.md finding 39, ASSUMPTIONS_AUDIT.md U16.
# ---------------------------------------------------------------------------


def test_the_integrator_has_no_ground_plane():
    """KNOWN FLAW, undocumented in the register, guarded only at one call site.

    A 747-approach released at 300 m in a 20 deg nose-down attitude crosses
    h = 0 and keeps integrating, finite throughout, to several hundred metres
    below sea level. `scripts/microburst.py` truncates its own run at one
    wingspan of clearance with a documented rationale -- that guard is per
    script, not in the engine.

    WRITTEN TO FAIL WHEN FIXED: adding a ground plane makes this go red.
    """
    from flightsim.integrate import init_sim, rollout
    from flightsim.trim import trimmed_controls, trimmed_state
    from flightsim.wind import zero_wind

    ac = B747PA
    solution, _ = trim(jnp.array(85.2), jnp.array(300.0), ac)
    state = trimmed_state(solution[0], jnp.array(85.2), jnp.array(300.0))
    state = state._replace(
        quat=euler_to_quat(jnp.array(0.0), jnp.array(-0.35), jnp.array(0.0))
    )
    sim = init_sim(state, jnp.zeros(2, dtype=jnp.uint32))
    _, trajectory = rollout(
        sim, trimmed_controls(solution[1], jnp.array(0.0)), 0.01, ac, 6000,
        zero_wind,
    )
    altitude = -np.asarray(trajectory.pos_ned)[:, 2]
    assert np.isfinite(altitude).all(), "the run went non-finite, not through"
    assert altitude.min() < -100.0, (
        f"lowest altitude {altitude.min():.1f} m -- if a ground plane was "
        "added, replace this test with one asserting it stops the run")


def test_the_atmosphere_extrapolates_below_sea_level_without_limit():
    """KNOWN FLAW, out of scope but reachable because of the previous test.

    `atmosphere.py` documents "0 to 20 km". Below zero the troposphere lapse
    continues without clamp or warning, and the previous test shows runs go
    there.
    """
    from flightsim.atmosphere import temperature

    assert float(temperature(-5000.0)) == pytest.approx(320.65, abs=0.01)
    assert float(density(-5000.0)) == pytest.approx(1.9305, rel=1e-3)
    assert float(temperature(-50000.0)) == pytest.approx(613.15, abs=0.01)
    assert float(density(-50000.0)) == pytest.approx(30.468, rel=1e-3)


# ---------------------------------------------------------------------------
# is_physical. AUDIT.md finding 37, ASSUMPTIONS_AUDIT.md U7.
# ---------------------------------------------------------------------------


def test_is_physical_rejects_trims_that_no_aircraft_could_fly():
    """REPAIRED. `trim.is_physical` used to check |alpha| <= 15 deg and nothing
    else -- not throttle, not elevator, not the residual. Finding 23 found
    absurd roots from a guess grid; this was a plain (V, h) sweep at the shipped
    initial guess, i.e. what an ordinary caller would do, and it endorsed
    319/640 of the 747's grid, 22/640 of the approach's, 284/640 of the
    Cherokee's and 364/640 of the Cessna's.

    The pinned example is the sharpest: the 747 at 471.8 m/s converges to
    alpha = -0.57 deg -- comfortably inside the alpha gate -- on a throttle of
    567. It is now rejected.

    The positive control lives in `test_trim.py`
    (`test_every_real_aircraft_trims_to_a_physical_solution`) and must stay
    green: a gate that rejects everything would pass this test and be useless.
    Both halves are asserted here too, so the pair cannot drift apart.
    """
    from flightsim.trim import is_physical

    solution, _ = trim(jnp.array(471.8), jnp.array(11579.0), B747)
    alpha, elevator, throttle = (float(v) for v in solution)
    # the root has not moved, and it is still inside the alpha gate on its own
    assert abs(math.degrees(alpha)) < 15.0, "the alpha gate no longer passes"
    assert throttle > 100.0, f"throttle {throttle:.1f} -- the pinned root moved"
    # ...and the widened gate rejects it
    assert not is_physical(solution, B747), (
        "is_physical endorses a trim demanding 567x full thrust again")

    # the two new conditions, isolated: each must be able to reject on its own
    at_cruise, _ = trim(jnp.array(CRUISE["boeing747"]["airspeed"]),
                        jnp.array(CRUISE["boeing747"]["altitude"]), B747)
    assert is_physical(at_cruise, B747), "the positive control has gone red"
    over_throttle = at_cruise.at[2].set(1.0 + 1e-9)
    under_throttle = at_cruise.at[2].set(-1e-9)
    past_the_stops = at_cruise.at[1].set(float(B747.elevator_limit) * 1.0001)
    assert not is_physical(over_throttle, B747), "throttle > 1 endorsed"
    assert not is_physical(under_throttle, B747), "negative throttle endorsed"
    assert not is_physical(past_the_stops, B747), "elevator past the stops endorsed"
    # and the boundaries themselves are inside, so a legitimate trim at the
    # limit is not rejected by an off-by-one
    assert is_physical(at_cruise.at[2].set(1.0), B747)
    assert is_physical(at_cruise.at[2].set(0.0), B747)
    assert is_physical(at_cruise.at[1].set(float(B747.elevator_limit)), B747)


def test_every_registry_aircraft_still_passes_the_widened_gate():
    """The positive control for the widened gate, at the level the audit
    measured the defect: every aircraft at its own cruise condition, all three
    unknowns checked. `test_trim.py` asserts the same thing and asserted it
    before the gate was widened; this asserts it survived the widening."""
    from flightsim.trim import is_physical

    for name in sorted(REGISTRY):
        ac = REGISTRY[name]
        x, r = trim(jnp.array(CRUISE[name]["airspeed"]),
                    jnp.array(CRUISE[name]["altitude"]), ac)
        assert float(jnp.linalg.norm(r)) < 1e-9, f"{name} did not converge"
        assert is_physical(x, ac), (
            f"{name}: alpha {math.degrees(float(x[0])):.2f} deg, elevator "
            f"{math.degrees(float(x[1])):.2f} deg, throttle {float(x[2]):.3f}")


# ---------------------------------------------------------------------------
# The order of the scheme. AUDIT.md finding 36, sharpening ASSUMPTIONS.md E4.
# ---------------------------------------------------------------------------


def test_a_wind_field_with_no_spatial_gradient_keeps_fourth_order():
    """VERIFICATION -- and it isolates the variable behind ASSUMPTIONS.md E4.

    E4 says the once-per-step wind hold makes the scheme first order "in a
    spatially varying wind field", measured against still air. That leaves open
    whether the cost comes from the wind or from its gradient. A constant uniform
    field has wind but no gradient, and it keeps FOURTH order: 3.9875 against
    3.9873 in still air, where the lee wave gives 0.99 and the updraft 1.00.

    So the hold costs three orders if and only if the field varies in space,
    which is what makes E4's verdict -- right for a stochastic field, wrong only
    for a deterministic spatial one -- exactly the right way round.
    """
    from flightsim import verification
    from flightsim.wind import field_model

    dts = np.array([1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32])
    uniform = field_model(lambda pos_ned: jnp.array([12.0, 5.0, -2.0]))
    _, order = verification.fixed_control_refinement(
        B747, CRUISE["boeing747"]["airspeed"], CRUISE["boeing747"]["altitude"],
        dts, dt_ref=1.0 / 1024.0, wind_model=uniform,
    )
    assert float(order) == pytest.approx(4.0, abs=0.05), (
        f"a gradient-free field no longer keeps fourth order: {float(order)}")


# ---------------------------------------------------------------------------
# The heading-rotation term. AUDIT.md finding 34, repaired in the remediation
# pass. `wind.along_track_shear` differentiated the wind field with the track
# direction held FIXED, so the along-track wind of a turning aircraft lost the
# part that changes because the along-track direction itself rotates.
# ---------------------------------------------------------------------------


def _straight_track_shear(pos_ned, vel_ned, field):
    """Proctor et al. Eq. (4) alone -- the expression the repair replaced.

    Kept verbatim so the reduction below compares against the real thing rather
    than against a paraphrase of it.
    """
    track = vel_ned[:2]
    heading = track / jnp.maximum(jnp.linalg.norm(track), 1e-9)

    def u_x(p):
        return jnp.dot(field(p)[:2], heading)

    return jnp.dot(jax.grad(u_x)(pos_ned), vel_ned)


def _parks_field():
    """The two-core Parks array the project's vortex runs use."""
    case = wind.PARKS_CASES["hannibal"]
    array = wind.VortexArray(
        north=jnp.array([0.0, case["spacing"]]),
        down=jnp.array([-12192.0, -12192.0]),
        r0=jnp.array(case["r0"]), v0=jnp.array(case["v0"]))
    return (lambda p: wind.vortex_wind(p, array)), case


def _circular_track(p0, psi0, turn_rate, speed):
    """A constant-speed, constant-rate ground track passing through p0 at t = 0."""
    def position(t):
        psi = psi0 + turn_rate * t
        radius = speed / turn_rate
        return jnp.array([
            p0[0] + radius * (jnp.sin(psi) - jnp.sin(psi0)),
            p0[1] - radius * (jnp.cos(psi) - jnp.cos(psi0)),
            p0[2],
        ])

    def velocity(t):
        psi = psi0 + turn_rate * t
        return jnp.array([speed * jnp.cos(psi), speed * jnp.sin(psi), 0.0])

    return position, velocity, jax.jacfwd(velocity)


def test_along_track_shear_reduces_to_proctor_eq_4_when_the_track_is_straight():
    """THE CHECK THAT THE EXTENSION IS AN EXTENSION.

    Proctor et al. Eq. (4) is written for a straight track. Adding the heading
    rotation is therefore a derivation, not a transcription, and what makes it a
    derivation FROM the paper rather than a replacement OF it is that it
    collapses back to Eq. (4) exactly when psi_dot = 0.

    Asserted as bit equality, not approximate agreement: with zero acceleration
    the added product is an exact zero and what remains is character for
    character the expression this replaced.
    """
    field, _ = _parks_field()
    velocities = [
        jnp.array([235.92, 0.0, 0.0]),        # due north, as the runs fly
        jnp.array([0.0, 235.92, 0.0]),        # due east
        jnp.array([160.0, 160.0, -8.0]),      # climbing, off-axis
    ]
    positions = [
        jnp.array([0.0, 0.0, -12192.0 + 182.88]),
        jnp.array([182.88, 0.0, -12192.0]),
        jnp.array([400.0, 120.0, -12192.0 + 90.0]),
    ]
    for pos in positions:
        for vel in velocities:
            straight = float(_straight_track_shear(pos, vel, field))
            extended = float(wind.along_track_shear(pos, vel, jnp.zeros(3), field))
            assert extended == straight, (
                f"at {pos} on {vel} the zero-turn case moved: "
                f"{extended!r} vs {straight!r}")


def test_along_track_shear_carries_the_heading_rotation_of_a_turning_aircraft():
    """REPAIRED, against a truth obtained independently of the implementation.

    `U_x(t)` is evaluated along a prescribed circular ground track -- a turn at
    a constant rate -- and differentiated in t. That is the definition of the
    quantity, computed without reference to how `along_track_shear` decomposes
    it, so agreement is evidence rather than a restatement.

    The sample point is one core radius above the first Parks core, where a
    Rankine vortex's tangential velocity is fully horizontal and equal to v0.
    Heading east, the whole of it is cross-track.
    """
    field, case = _parks_field()
    speed = float(CRUISE["boeing747"]["airspeed"])
    p0 = jnp.array([0.0, 0.0, -12192.0 + case["r0"]])

    for turn_deg in (3.0, 1.0):
        position, velocity, acceleration = _circular_track(
            p0, math.radians(90.0), math.radians(turn_deg), speed)

        def u_x(t):
            p, v = position(t), velocity(t)
            return jnp.dot(field(p)[:2], v[:2] / jnp.linalg.norm(v[:2]))

        exact = float(jax.grad(u_x)(0.0))
        got = float(wind.along_track_shear(
            position(0.0), velocity(0.0), acceleration(0.0), field))
        assert got == pytest.approx(exact, abs=1e-15), (
            f"{turn_deg} deg/s: {got} against the differentiated truth {exact}")

        # and this term is the whole of what the straight-track form missed
        straight = float(_straight_track_shear(position(0.0), velocity(0.0), field))
        assert straight == pytest.approx(0.0, abs=1e-12), (
            "the frozen-heading expression should see nothing at all here")

    # SIZE. At a standard-rate turn the omission was the entire FAA 1 km
    # alerting threshold. Rebuilt from the source constant rather than pinned:
    # dF = v0 * psi_dot / g, with v0 = 85 ft/s from Parks et al.
    rate = math.radians(3.0)
    position, velocity, acceleration = _circular_track(
        p0, math.radians(90.0), rate, speed)
    got = float(wind.along_track_shear(
        position(0.0), velocity(0.0), acceleration(0.0), field))
    delta_f = abs(got) / G0
    single_core = case["v0"] * rate / G0
    assert single_core == pytest.approx(0.1383, abs=5e-4)
    assert delta_f == pytest.approx(0.1423, abs=5e-4), (
        f"the standard-rate turn now costs dF = {delta_f:.4f}")
    assert delta_f > single_core > 0.1, "still above the FAA 0.1 threshold"


def test_the_shipped_runs_fly_a_genuinely_straight_track():
    """WHY THE REPAIR MOVED NOTHING, measured rather than assumed.

    Every field in `wind.py` has zero east wind on the north axis and every run
    the project reports is flown due north, so the ground track never turns and
    the new term is identically zero. That is a property of the RUNS rather than
    of the approximation, and it is why nothing the project quotes moved.

    A short leg only. The full runs are 30,000+ steps each and were measured end
    to end during the remediation pass: psi_dot was exactly 0.0 at all 77,036
    samples of the two lee-wave legs and the microburst penetration, and the
    printed output of both scripts was byte-identical across the change.
    """
    from flightsim import dynamics, integrate, trim

    ac = REGISTRY["cherokee"]
    burst = wind.microburst(u_max=19.03, radius=1000.0, z_m=150.0)
    field = lambda p: wind.microburst_wind(p, burst)  # noqa: E731
    peak = wind.MICROBURST_PEAK_RADIUS_RATIO * 1000.0

    x, _ = trim.trim(jnp.array(50.0), jnp.array(300.0), ac)
    controls = trim.trimmed_controls(x[1], x[2])
    state = trim.trimmed_state(x[0], jnp.array(50.0), jnp.array(300.0))
    state = state._replace(pos_ned=jnp.array([-3.0 * peak, 0.0, -300.0]))
    _, hist = integrate.rollout(
        integrate.init_sim(state, jnp.zeros(2, dtype=jnp.uint32)),
        controls, jnp.array(0.05), ac, 400, wind_model=wind.field_model(field))

    def sample(pos_ned, vel_body, quat, omega):
        s = State(pos_ned=pos_ned, vel_body=vel_body, quat=quat, omega=omega)
        wind_ned = field(pos_ned)
        gust = wind.gust_rates(pos_ned, quat, field)
        d = dynamics.derivatives(s, controls, ac, wind_ned, gust)
        accel_ned = quat_to_dcm(quat) @ (d.vel_body + jnp.cross(omega, vel_body))
        vel_ned = quat_to_dcm(quat) @ vel_body
        return jnp.array([
            wind.along_track_shear(pos_ned, vel_ned, accel_ned, field),
            _straight_track_shear(pos_ned, vel_ned, field),
            accel_ned[1],
        ])

    rows = np.asarray(jax.vmap(sample)(
        hist.pos_ned, hist.vel_body, hist.quat, hist.omega))
    assert np.abs(rows[:, 2]).max() == 0.0, "the run developed an east acceleration"
    assert np.array_equal(rows[:, 0], rows[:, 1]), (
        "the repaired and straight-track expressions disagree on a straight run")


# ---------------------------------------------------------------------------
# The empty superposition. ASSUMPTIONS_AUDIT.md U14's parenthesis.
# ---------------------------------------------------------------------------


def test_superposing_no_fields_returns_a_field_and_not_the_integer_zero():
    """REPAIRED. `sum(...)` over an empty generator is the Python int 0, so
    `wind.superpose()` returned a value where every caller expects a callable.
    Reachable from `superpose(*chosen)` whenever the filter that built `chosen`
    selected nothing.

    The empty sum is now the zero field, which is the identity the function's
    own algebra requires: superposing it onto anything must return that thing.
    """
    empty = wind.superpose()
    assert callable(empty), "superpose() still returns a value rather than a field"
    for pos in (jnp.zeros(3), jnp.array([1200.0, -450.0, -12192.0])):
        got = empty(pos)
        assert got.shape == (3,)
        assert float(jnp.linalg.norm(got)) == 0.0

    # and it really is the identity: adding it changes nothing, bit for bit
    field, _ = _parks_field()
    combined = wind.superpose(field, empty)
    for pos in (jnp.array([90.0, 0.0, -12192.0]), jnp.array([300.0, 40.0, -12100.0])):
        assert np.array_equal(np.asarray(combined(pos)), np.asarray(field(pos)))
