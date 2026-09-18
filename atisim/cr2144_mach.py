"""CR-2144's 747 Mach-derivative curves, digitised by hand, and what checks them.

SOURCE. Heffley & Jewell, "Aircraft Handling Qualities Data", NASA CR-2144,
December 1972 -- `refs/NASA-CR-2144.pdf`, outside git. Section IX's three
B-747 sheets captioned "636600 lb / .25 c / Flexible", each carrying SL
(solid), 20,000 ft (dashed) and 40,000 ft (dash-dot) curves against Mach:

    printed p. 220  (PDF index 225)  CL_alpha, CD_alpha              per rad
    printed p. 221  (PDF index 226)  Cm_alpha, Cm_alphadot, Cm_q     per rad
    printed p. 222  (PDF index 227)  CL_M, CD_M, Cm_M                per Mach

THE DATA is `data/cr2144_p220_222_digitised.csv`: read by hand in Engauge
Digitizer, one .dig file per quantity, and regenerated from those files by
`scripts/cr2144_speed_derivatives.py --dig-dir`, which applies each file's own
three-point axis calibration. **The eight .dig originals are tracked beside it
in `data/cr2144_dig/`**, each carrying the page crop it was traced on, so the
placement of every point can be re-read or corrected rather than only its
result being trusted. That is where `--dig-dir` looks by default. ONLY THE HAND-PLACED POINTS ARE KEPT. The Engauge
CSV exports that came with them are not used: they put every curve on one
shared Mach grid and extrapolate each one past its drawn end, which is where a
40,000 ft Cm_M of 105 and a sea-level CL_M of -47 come from. Two cleaning rules,
both reported by the script: a point copied verbatim from the SL curve into
another curve is dropped (nine in CL_M's 40,000 ft curve, at Mach numbers no
40,000 ft condition flies), and an exact repeat within a curve is dropped.

THE CHECK. The circled numbers on every curve are CR-2144's flight conditions
3-10, and Table IX-4 prints the dimensional derivatives at each. Appendix A
(printed pp. A-16, A-17) relates the two, so every circled point has a value the
curve must pass through that does not depend on anyone's reading of the plot.
`backsolve` inverts those relations. It needs two things the tables do not
print, and both are stated rather than hidden:

  * TRIM THRUST. Table IX-3 gives a thrust moment arm LTH = 10.0 ft and a thrust
    inclination XI = 2.50 deg at every condition, so the aerodynamic C_m at trim
    is NOT zero -- it balances the thrust moment -- and Appendix A's M_u and M_w
    both contain it. Leaving it out mis-states Cm_M by up to 0.07; including it
    brings seven circled points to within 0.011. The sign of LTH is therefore
    fixed by the tables themselves rather than assumed. T_u = 0 is DECLARED:
    Table IX-4's XU, ZU, MU are the starred forms, which would carry it.
  * TRIM DRAG. SOURCED at 40,000 ft from Figure IX-6 (0.0747 at M 0.70 and
    0.0427 at M 0.90 as digitised in test_drag_polar; 0.043 at M 0.80, the value
    `aircraft._boeing_747` cites). At SL and 20,000 ft it is DECLARED, taken from
    the FC9 drag polar, and the script reports what +-0.01 of it moves.
"""

import csv
import math
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import numpy as np
from scipy.interpolate import PchipInterpolator

DATA = Path(__file__).parent / "data" / "cr2144_p220_222_digitised.csv"

QUANTITIES = ("cl_alpha", "cd_alpha", "cm_alpha", "cm_alpha_dot", "cm_q",
              "cl_m", "cd_m", "cm_m")
PRINTED_PAGE = {"cl_alpha": 220, "cd_alpha": 220, "cm_alpha": 221,
                "cm_alpha_dot": 221, "cm_q": 221, "cl_m": 222, "cd_m": 222,
                "cm_m": 222}


class Curve(NamedTuple):
    """One altitude's curve: the hand-placed points, sorted by Mach, and the
    sheet's own scale -- graph units per screen pixel, and the span between
    the calibration points -- which is what a reading error is measured in."""

    mach: np.ndarray
    value: np.ndarray
    mach_per_px: float
    value_per_px: float
    mach_span: float
    value_span: float


@lru_cache(maxsize=1)
def curves() -> dict:
    """{(quantity, altitude): Curve}, altitude one of "SL", "20K", "40K"."""
    rows = {}
    with open(DATA, newline="") as f:
        for r in csv.DictReader(f):
            rows.setdefault((r["quantity"], r["altitude"]), []).append(r)
    out = {}
    for key, rs in rows.items():
        mach = np.array([float(r["mach"]) for r in rs])
        val = np.array([float(r["value"]) for r in rs])
        order = np.argsort(mach, kind="stable")
        r0 = rs[0]
        out[key] = Curve(mach[order], val[order], float(r0["mach_per_px"]),
                         float(r0["value_per_px"]), float(r0["mach_span"]),
                         float(r0["value_span"]))
    return out


def value(quantity: str, altitude: str, mach: float, scheme: str = "linear",
          curve: Curve | None = None) -> float:
    """The curve read at `mach`, or NaN outside the range its points span.

    Never extrapolated: a curve read beyond its last hand-placed point is the
    failure the Engauge exports make, and a NaN is the honest answer there.
    """
    c = curve if curve is not None else curves().get((quantity, altitude))
    if c is None or not (c.mach[0] <= mach <= c.mach[-1]):
        return float("nan")
    if scheme == "linear":
        return float(np.interp(mach, c.mach, c.value))
    if scheme == "pchip":
        return float(PchipInterpolator(c.mach, c.value)(mach))
    raise ValueError(f"unknown scheme {scheme!r}")


AUTO_DIR = Path(__file__).parent.parent / "Reference_papers" / "CR-2144" / "csv"

# The automated trace names files by printed symbol and altitude in feet; the
# hand reading uses snake_case quantities and SL / 20K / 40K. cm_q is absent
# from the hand reading, so it is not mapped.
AUTO_NAME = {"cl_alpha": "p220_CL_alpha", "cd_alpha": "p220_CD_alpha",
             "cm_alpha": "p221_Cm_alpha", "cm_alpha_dot": "p221_Cm_adot",
             "cl_m": "p222_CL_M", "cd_m": "p222_CD_M", "cm_m": "p222_Cm_M"}
AUTO_ALT = {"SL": "SL", "20K": "20000ft", "40K": "40000ft"}


def automated_curves(csv_dir: Path = AUTO_DIR) -> dict:
    """{(quantity, altitude): (mach, value)} from the automated pp. 218-228
    trace, for the curves the hand reading also has. Lines starting '#' are the
    file's provenance header and are skipped."""
    out = {}
    for q, stem in AUTO_NAME.items():
        for alt, suffix in AUTO_ALT.items():
            p = Path(csv_dir) / f"{stem}_{suffix}.csv"
            if not p.exists():
                continue
            lines = [ln for ln in p.read_text(encoding="utf-8").splitlines()
                     if ln and not ln.startswith("#")]
            data = np.array([[float(x) for x in ln.split(",")] for ln in lines[1:]])
            order = np.argsort(data[:, 0], kind="stable")
            out[(q, alt)] = (data[order, 0], data[order, 1])
    return out


class CrossCheck(NamedTuple):
    quantity: str
    altitude: str
    n: int                 # hand points compared
    median_pct_fs: float   # median |hand - automated|, % of the panel's full scale
    max_pct_fs: float
    median_px: float       # the same median, in the hand sheet's own pixels
    bias_pct_fs: float     # signed median (hand - automated), % of full scale
    best_altitude: str     # the automated altitude these hand points sit closest to


def crosscheck(max_gap: float = 0.01, csv_dir: Path = AUTO_DIR) -> list:
    """Every hand-placed point, against the automated trace at the same Mach.

    A point is compared only where the automated trace has a sample within
    `max_gap` Mach on BOTH sides. Its README records short gaps where curves
    cross; interpolating across one would compare the hand reading against a
    straight line the automated trace never drew.
    """
    hand, auto = curves(), automated_curves(csv_dir)

    def residual(c: Curve, am: np.ndarray, av: np.ndarray) -> np.ndarray:
        i = np.clip(np.searchsorted(am, c.mach), 1, len(am) - 1)
        inside = (c.mach >= am[0]) & (c.mach <= am[-1])
        near = (c.mach - am[i - 1] <= max_gap) & (am[i] - c.mach <= max_gap)
        ok = inside & near
        return c.value[ok] - np.interp(c.mach[ok], am, av)

    rows = []
    for (q, alt), c in sorted(hand.items()):
        if (q, alt) not in auto:
            continue
        r = residual(c, *auto[(q, alt)])
        if r.size == 0:
            continue
        nearest = {}
        for a in AUTO_ALT:
            if (q, a) in auto:
                ra = residual(c, *auto[(q, a)])
                if ra.size:
                    nearest[a] = float(np.median(np.abs(ra)))
        fs = c.value_span
        rows.append(CrossCheck(
            q, alt, int(r.size),
            100.0 * float(np.median(np.abs(r))) / fs,
            100.0 * float(np.max(np.abs(r))) / fs,
            float(np.median(np.abs(r))) / c.value_per_px,
            100.0 * float(np.median(r)) / fs,
            min(nearest, key=nearest.get)))
    return rows


def perturbed(c: Curve, rng: np.random.Generator, sigma_px: float) -> Curve:
    """One Monte Carlo re-reading of a curve, in the sheet's own pixels.

    Two error sources, both of size `sigma_px`: every point scattered
    independently in both screen directions (placing a cursor on an ink line a
    few pixels wide), and a systematic calibration error common to the whole
    curve -- an offset and a scale on each axis, from misplacing the axis
    points the same distance.
    """
    n = c.mach.size
    m_off, v_off = rng.normal(0.0, sigma_px, 2) * (c.mach_per_px, c.value_per_px)
    m_rel = rng.normal(0.0, sigma_px) * c.mach_per_px / c.mach_span
    v_rel = rng.normal(0.0, sigma_px) * c.value_per_px / c.value_span
    mach = (c.mach + rng.normal(0.0, sigma_px, n) * c.mach_per_px + m_off) * (1.0 + m_rel)
    val = (c.value + rng.normal(0.0, sigma_px, n) * c.value_per_px + v_off) * (1.0 + v_rel)
    order = np.argsort(mach, kind="stable")
    mach, val = mach[order], val[order]
    keep = np.concatenate([[True], np.diff(mach) > 1e-9])
    return c._replace(mach=mach[keep], value=val[keep])


# ---------------------------------------------------------------------------
# CR-2144 Table IX-3, printed p. 229: altitude, Mach, VTo (ft/s), dynamic
# pressure Q (lb/ft^2), alpha (deg). Identical at all eight: W = 636,636 lb,
# Iy = 0.331e8 slug ft^2, LTH = 10.0 ft, XI = 2.50 deg, S = 5500 ft^2,
# cbar = 27.31 ft. Read at 300 dpi.
# ---------------------------------------------------------------------------
IX3 = {
    3: ("SL", 0.450, 502.0, 300.0, 3.10),
    4: ("SL", 0.650, 726.0, 626.0, 0.00),
    5: ("20K", 0.500, 518.0, 170.0, 6.80),
    6: ("20K", 0.650, 674.0, 288.0, 2.50),
    7: ("20K", 0.800, 830.0, 436.0, 0.00),
    8: ("40K", 0.700, 678.0, 135.0, 7.30),
    9: ("40K", 0.800, 774.0, 177.0, 4.60),
    10: ("40K", 0.900, 871.0, 224.0, 2.40),
}
W_LB, IY_SLUGFT2, S_FT2, CBAR_FT, LTH_FT = 636636.0, 0.331e8, 5500.0, 27.31, 10.0
XI_RAD = math.radians(2.50)
G_FTS2 = 32.174  # the value CR-2144's own arithmetic uses

# CR-2144 Table IX-4, printed p. 230, "(BODY AXIS SYSTEM)", read at 300 dpi.
# XU, ZU, MU are the table's STARRED forms. At 150 dpi FC10's MU reads -.523E-4
# and FC8's XU carries a minus sign; at 300 dpi they are -.623E-4 and +.00187.
IX4_FIELDS = ("Xu", "Zu", "Mu", "Xw", "Zw", "Mw", "Zwd", "Zq", "Mwd", "Mq")
IX4 = {
    3: (-.00499, -.0807, .000146, .0743, -.736, -.00262, .0297, -10.4, -.000221, -.699),
    4: (-.00777, -.126, -.000199, .0345, -.963, -.00239, .0293, -12.8, -.000228, -.925),
    5: (-.00247, -.0679, .000247, .0782, -.433, -.00170, .0157, -6.39, -.000125, -.421),
    6: (-.00280, -.0832, .885e-4, .0482, -.539, -.00190, .0156, -8.09, -.000155, -.535),
    7: (-.00643, -.0941, -.000222, .0253, -.624, -.00153, .0144, -9.98, -.000212, -.659),
    8: (.00187, -.0696, .000259, .0263, -.292, -.00101, .00704, -4.32, -.905e-4, -.284),
    9: (-.00276, -.0650, .000193, .0389, -.317, -.00105, .00556, -5.16, -.000116, -.339),
    10: (-.0200, -.0424, -.623e-4, .0159, -.401, -.00190, .00614, -6.71, -.000160, -.401),
}

# Figure IX-6, 40,000 ft: M 0.70 and 0.90 from test_drag_polar.FIGURE_IX6_40KFT;
# M 0.80 is the 0.043 `aircraft._boeing_747` was built on.
FIG_IX6_CD_40K = {8: 0.0747, 9: 0.043, 10: 0.0427}

# CR-2144 Table IX-5, printed p. 231, FC9 bare-airframe denominator.
IX5_FC9 = dict(ph_wn=0.0673, ph_z=0.0489, sp_wn=0.964, sp_z=0.387)


def trim_drag(fc: int) -> float:
    """Trim C_D: SOURCED at 40,000 ft, DECLARED (the FC9 polar) elsewhere."""
    if fc in FIG_IX6_CD_40K:
        return FIG_IX6_CD_40K[fc]
    import jax.numpy as jnp

    from atisim import aero
    from atisim.aircraft import REGISTRY

    ac = REGISTRY["boeing747"]
    _, mach, _, q, _ = IX3[fc]
    CL = W_LB / (q * S_FT2)
    return (float(ac.CD0) + CL**2 / (math.pi * float(ac.e) * float(ac.AR))
            + float(aero.wave_drag(jnp.array(mach), jnp.array(CL), ac)))


def backsolve(fc: int, CD: float | None = None, thrust: bool = True) -> dict:
    """Table IX-4 at flight condition `fc`, inverted through Appendix A.

    Printed pp. A-16 and A-17, body axis, with r = W0/U0 and mu = M/2:

        X_u = (rho S U0/m)  [-mu C_XM - C_X + (r/2) C_Xa]
        X_w = (rho S U0/2m) [-C_Xa - 2r (C_X + mu C_XM)]
        Z_u, Z_w  the same with N for X
        M_u = (rho S c U0/Iy)  [ mu C_mM + C_m - (r/2) C_ma]
        M_w = (rho S c U0/2Iy) [ C_ma + 2r (C_m + mu C_mM)]
        M_q = (rho S c^2 VTo/4Iy) C_mq,  M_wdot = (rho S c^2/4Iy)(U0/VTo) C_madot

    Each (u, w) pair is two equations in (C_.a, C_.M) and solves in closed form.
    C_X = -X/qS positive aft and C_N positive up (p. A-14); CL_M and CD_M follow
    by rotating through alpha, which is exact at constant alpha.
    """
    _, mach, VT, Q, adeg = IX3[fc]
    Xu, Zu, Mu, Xw, Zw, Mw, Zwd, Zq, Mwd, Mq = IX4[fc]
    m = W_LB / G_FTS2
    rho = 2.0 * Q / VT**2
    a = math.radians(adeg)
    ca, sa = math.cos(a), math.sin(a)
    U0, r, mu, qS = VT * ca, math.tan(a), mach / 2.0, Q * S_FT2
    if CD is None:
        CD = trim_drag(fc)
    T = CD * qS / math.cos(a + XI_RAD) if thrust else 0.0
    CL = (W_LB - T * math.sin(a + XI_RAD)) / qS
    CN, CX = CL * ca + CD * sa, CD * ca - CL * sa
    Cm = -T * LTH_FT / (qS * CBAR_FT)
    Kf, Km = rho * S_FT2 * U0 / m, rho * S_FT2 * CBAR_FT * U0 / IY_SLUGFT2
    k = mu * (1.0 + r * r)
    CmM = (Mu + r * Mw) / (Km * k) - Cm / mu
    CNM = -(Zu + r * Zw) / (Kf * k) - CN / mu
    CXM = -(Xu + r * Xw) / (Kf * k) - CX / mu
    Cma = 2.0 * Mw / Km - 2.0 * r * (Cm + mu * CmM)
    CNa = -2.0 * Zw / Kf - 2.0 * r * (CN + mu * CNM)
    CXa = -2.0 * Xw / Kf - 2.0 * r * (CX + mu * CXM)
    CLa, CDa = np.linalg.solve(np.array([[ca, sa], [-sa, ca]]),
                               np.array([CNa + CL * sa - CD * ca, CXa + CD * sa + CL * ca]))
    return dict(
        cl_m=CNM * ca - CXM * sa, cd_m=CNM * sa + CXM * ca, cm_m=CmM,
        cl_alpha=float(CLa), cd_alpha=float(CDa), cm_alpha=Cma,
        cm_q=4.0 * IY_SLUGFT2 * Mq / (rho * S_FT2 * CBAR_FT**2 * VT),
        cm_alpha_dot=4.0 * IY_SLUGFT2 * Mwd * VT / (rho * S_FT2 * CBAR_FT**2 * U0),
        CD=CD, Cm_trim=Cm)


def mach_increment(alpha, V, rho, a_sound, mass, Iyy, S, c, CL_M, CD_M, Cm_M):
    """Appendix A's Mach terms as a body-axis [u, w, q, theta] matrix increment.

    Any consistent units. This is what declaring `Aircraft.CL_M/CD_M/Cm_M` at
    the trim Mach adds to `validation.longitudinal_matrix`, and
    test_cr2144_speed_derivatives asserts the two agree; the Monte Carlo below
    uses this form because it needs no re-trace per sample.
    """
    ca, sa = math.cos(alpha), math.sin(alpha)
    U0, r, mu = V * ca, sa / ca, V / a_sound / 2.0
    Kf, Km = rho * S * U0 / mass, rho * S * c * U0 / Iyy
    CXM, CNM = CD_M * ca - CL_M * sa, CL_M * ca + CD_M * sa
    dA = np.zeros((4, 4))
    dA[0, 0], dA[1, 0], dA[2, 0] = -Kf * mu * CXM, -Kf * mu * CNM, Km * mu * Cm_M
    dA[0, 1], dA[1, 1], dA[2, 1] = r * dA[0, 0], r * dA[1, 0], r * dA[2, 0]
    return dA


def modes(A) -> dict:
    """Phugoid and short-period (wn, zeta) from a longitudinal plant matrix."""
    from atisim.validation import modes_from_matrix

    (pw, pz), (sw, sz) = modes_from_matrix(A)
    return dict(ph_wn=pw, ph_z=pz, sp_wn=sw, sp_z=sz)


def errors_vs_ix5(m: dict) -> dict:
    """Signed fractional error of each mode against Table IX-5 FC9."""
    return {k: (v - IX5_FC9[k]) / IX5_FC9[k] for k, v in m.items()}
