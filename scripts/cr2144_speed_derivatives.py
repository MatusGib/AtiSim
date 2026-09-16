"""CR-2144's 747 speed derivatives, digitised: extract, check, retest, and price the reading.

SOURCE. NASA CR-2144, Heffley & Jewell, "Aircraft Handling Qualities Data",
December 1972, `refs/NASA-CR-2144.pdf` (outside git). Section IX, the three B-747
Mach sheets captioned "636600 lb / .25 c / Flexible": printed p. 220 (CL_alpha,
CD_alpha), p. 221 (Cm_alpha, Cm_alphadot, Cm_q) and p. 222 (CL_M, CD_M, Cm_M),
PDF indices 225-227. Checked against Tables IX-3, IX-4 and IX-5 (printed
pp. 229-231) through Appendix A (printed pp. A-14, A-16, A-17).

The digitisation itself was done by hand in Engauge Digitizer, one .dig file per
quantity. This script answers five questions about it:

  1  EXTRACT (--dig-dir)  Read the HAND-PLACED points out of the .dig files through
     each file's own three-point axis calibration, drop the two kinds of artefact
     the files carry, and compare with -- or with --write, regenerate --
     `atisim/data/cr2144_p220_222_digitised.csv`.
  2  AUDIT THE EXPORTS (--csv-dir)  Say whether each Engauge CSV column is the curve
     its header claims, and how many of its rows are extrapolation.
  3  CHECK  Every circled flight condition 3-10 has a value Table IX-4 implies. Read
     the curves there and report the residuals.
  4  RETEST  Declare the FC9 values on the 747 through `Aircraft.CL_M/CD_M/Cm_M`,
     re-trim, linearise with the engine's own jacfwd, and compare all four
     longitudinal modes with Table IX-5 -- the improvement PROJECT.md section 4
     attributed in analysis to the speed derivatives.
  5  PRICE  How much of that retest is the hand reading: interpolation scheme,
     leave-one-out, a pixel-level Monte Carlo on the sheets' own scales, and the
     CHECK residuals propagated as an empirical error.
  (--headline)  Fly Mehta's five-vortex Hannibal field with the declared set and
     report how far the headline load moves. Analysis only; no registry change.

Run from the worktree root with the tree on the path (PROJECT.md section 10):
    PYTHONPATH=<worktree> python scripts/cr2144_speed_derivatives.py \\
        --dig-dir <folder of .dig files> [--csv-dir <same>] [--mc 4000] [--headline]
"""

import argparse
import csv
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import atisim

print(f"atisim imported from: {atisim.__file__}")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from scipy.interpolate import Akima1DInterpolator, CubicSpline, PchipInterpolator  # noqa: E402

from atisim import aero  # noqa: E402
from atisim import cr2144_mach as cm  # noqa: E402
from atisim.aircraft import CRUISE, REGISTRY  # noqa: E402
from atisim.atmosphere import density, speed_of_sound  # noqa: E402
from atisim.trim import trim  # noqa: E402
from atisim.validation import longitudinal_matrix  # noqa: E402

DIG_DIR = Path(atisim.__file__).parent / "data" / "cr2144_dig"
STEM = {"cl_alpha": "cl_alpha", "cd_alpha": "cd_alpha", "cm_alpha": "cm_alpha",
        "cm_alpha_dot": "cm_alpha_dot", "cm_q": "cm_q", "cl_m": "cl_m", "cd_m": "cd_m",
        "cm_m": "cm_m"}
ALT = {"SL": "SL", "Sl": "SL", "20,000": "20K", "40,000": "40K"}
FIELDS = ("quantity", "altitude", "mach", "value", "mach_per_px", "value_per_px",
          "mach_span", "value_span")
SPEED = ("cl_m", "cd_m", "cm_m")
MODES = ("ph_wn", "ph_z", "sp_wn", "sp_z")


# ---------------------------------------------------------------------------
# 1. Extract
# ---------------------------------------------------------------------------

def read_dig(path: Path):
    """Hand-placed points through the file's own axis calibration, cleaned.

    Engauge stores graph coordinates only for its three axis points; every curve
    point carries SCREEN pixels. The affine map through the three axis points is
    exact for linear axes and absorbs any small rotation of the scan.
    """
    root = ET.parse(path).getroot()
    axes, raw = None, {}
    for curve in root.iter("Curve"):
        pts = []
        for p in curve.iter("Point"):
            s, g = p.find("PositionScreen"), p.find("PositionGraph")
            pts.append((int(p.get("Ordinal")), float(s.get("X")), float(s.get("Y")),
                        None if g is None else (float(g.get("X")), float(g.get("Y")))))
        if curve.get("CurveName") == "Axes":
            axes = pts
        else:
            raw[ALT[curve.get("CurveName")]] = sorted(pts)
    S = np.array([[sx, sy, 1.0] for _, sx, sy, _ in axes])
    G = np.array([g for *_, g in axes])
    T = np.linalg.solve(S, G)
    scale = dict(mach_per_px=math.hypot(T[0, 0], T[1, 0]),
                 value_per_px=math.hypot(T[0, 1], T[1, 1]),
                 mach_span=float(np.ptp(G[:, 0])), value_span=float(np.ptp(G[:, 1])))
    screens = {k: {(sx, sy) for _, sx, sy, _ in v} for k, v in raw.items()}
    out, report = {}, {}
    for alt, pts in raw.items():
        others = set().union(*[screens[o] for o in screens if o != alt])
        keep, seen, copied, repeats = [], set(), 0, 0
        for _, sx, sy, _ in pts:
            if alt != "SL" and (sx, sy) in others:
                copied += 1
                continue
            if (sx, sy) in seen:
                repeats += 1
                continue
            seen.add((sx, sy))
            keep.append([sx, sy, 1.0])
        xy = np.array(keep) @ T
        xy = xy[np.argsort(xy[:, 0], kind="stable")]
        out[alt] = xy
        report[alt] = (len(pts), copied, repeats)
    return out, scale, report


def extract(dig_dir: Path):
    rows = []
    print("\n== 1. EXTRACT: hand-placed points only ==")
    for path in sorted(dig_dir.glob("*.dig")):
        quantity = STEM[path.stem.lower()]
        pts, scale, report = read_dig(path)
        for alt in ("SL", "20K", "40K"):
            if alt not in pts:
                continue
            n, copied, repeats = report[alt]
            xy = pts[alt]
            print(f"  {path.name:17s} {alt:3s} {n:3d} placed, {copied} copied from SL dropped, "
                  f"{repeats} repeats dropped -> {len(xy):3d}  M {xy[0,0]:.4f}-{xy[-1,0]:.4f}  "
                  f"(p. {cm.PRINTED_PAGE[quantity]}, {scale['mach_per_px']:.2e} M/px, "
                  f"{scale['value_per_px']:.2e}/px)")
            for m, v in xy:
                rows.append(dict(quantity=quantity, altitude=alt, mach=f"{m:.6f}",
                                 value=f"{v:.6f}", **{k: f"{scale[k]:.6e}" for k in scale}))
    return rows


def compare_or_write(rows, write: bool):
    if write:
        cm.DATA.parent.mkdir(exist_ok=True)
        with open(cm.DATA, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        cm.curves.cache_clear()
        print(f"  wrote {len(rows)} points to {cm.DATA}")
        return
    with open(cm.DATA, newline="") as f:
        tracked = list(csv.DictReader(f))
    same = len(tracked) == len(rows) and all(
        a["quantity"] == b["quantity"] and a["altitude"] == b["altitude"]
        and abs(float(a["mach"]) - float(b["mach"])) < 1e-9
        and abs(float(a["value"]) - float(b["value"])) < 1e-9 for a, b in zip(tracked, rows))
    print(f"  tracked {cm.DATA.name}: {'IDENTICAL to' if same else 'DIFFERS from'} the .dig files")


# ---------------------------------------------------------------------------
# 2. Audit the Engauge CSV exports
# ---------------------------------------------------------------------------

def audit_exports(csv_dir: Path):
    print("\n== 2. AUDIT THE EXPORTS: is each column the curve its header names? ==")
    for path in sorted(csv_dir.glob("*.csv")):
        quantity = STEM[path.stem.lower()]
        lines = path.read_text().strip().splitlines()
        # The header is "x,SL,40,000,20,000": the thousands separators split it
        # into six fields over four data columns, so it cannot be parsed as CSV.
        # The order is taken from the header's words and confirmed against the
        # .dig curves below rather than trusted.
        order = (["SL"] if lines[0].split(",")[1] in ("SL", "Sl") else []) + ["40K", "20K"]
        data = np.array([[float(t) for t in ln.split(",")] for ln in lines[1:]])
        parts = []
        for j, alt in enumerate(order, start=1):
            c = cm.curves()[(quantity, alt)]
            hits = [abs(data[i, j] - v) for m, v in zip(c.mach, c.value)
                    for i in np.where(np.abs(data[:, 0] - m) < 2e-5)[0][:1]]
            inside = (data[:, 0] >= c.mach[0] - 1e-6) & (data[:, 0] <= c.mach[-1] + 1e-6)
            outside = ~inside & np.isfinite(data[:, j])
            worst = float(np.abs(data[outside, j]).max()) if outside.any() else 0.0
            match = f"{max(hits):.1e} at {len(hits)}/{c.mach.size} points" if hits else "no shared Mach"
            parts.append(f"{alt}: {match}; {outside.sum()} extrapolated rows, worst |y| {worst:.3g}")
        print(f"  {path.name:17s} " + " | ".join(parts))


# ---------------------------------------------------------------------------
# 3. Check against Table IX-4
# ---------------------------------------------------------------------------

def check():
    print("\n== 3. CHECK: the curve at each circled condition against Table IX-4 via Appendix A ==")
    print("  (table value with trim thrust; [no thrust]; +-spread for C_D +-0.01; digitised linear | pchip)")
    resid = {q: [] for q in cm.QUANTITIES}
    resid_nothrust = []
    for fc, (alt, mach, *_rest) in cm.IX3.items():
        b, b0 = cm.backsolve(fc), cm.backsolve(fc, thrust=False)
        hi, lo = cm.backsolve(fc, CD=b["CD"] + 0.01), cm.backsolve(fc, CD=b["CD"] - 0.01)
        src = "Fig. IX-6" if fc in cm.FIG_IX6_CD_40K else "DECLARED polar"
        print(f"  FC{fc:<2d} {alt:3s} M {mach:.3f}  C_D {b['CD']:.4f} ({src})  trim C_m {b['Cm_trim']:+.4f}")
        for q in cm.QUANTITIES:
            lin, pch = cm.value(q, alt, mach), cm.value(q, alt, mach, "pchip")
            spread = max(abs(hi[q] - b[q]), abs(lo[q] - b[q]))
            tail = ""
            if math.isfinite(lin):
                resid[q].append((fc, lin - b[q], b[q]))
                tail = f"  resid {lin - b[q]:+.4f}"
                if q == "cm_m":
                    resid_nothrust.append(lin - b0[q])
            print(f"      {q:13s} {b[q]:+9.4f} [{b0[q]:+9.4f}] +-{spread:.4f}   "
                  f"{lin:+9.4f} | {pch:+9.4f}{tail}")
    print("\n  summary, digitised (linear) minus table:")
    for q in cm.QUANTITIES:
        r = np.array([x[1] for x in resid[q]])
        rel = np.array([x[1] / abs(x[2]) for x in resid[q]])
        print(f"    {q:13s} n={r.size}  RMS {np.sqrt(np.mean(r**2)):.4f}  max|r| {np.abs(r).max():.4f}  "
              f"max|r|/|table| {np.abs(rel).max():.1%}  FCs {[x[0] for x in resid[q]]}")
    rn = np.array(resid_nothrust)
    rt = np.array([x[1] for x in resid["cm_m"]])
    print(f"    cm_m RMS with trim thrust {np.sqrt(np.mean(rt**2)):.4f} vs WITHOUT {np.sqrt(np.mean(rn**2)):.4f}"
          " -- the thrust-moment term is what makes table and figure agree")
    return resid


# ---------------------------------------------------------------------------
# 4. Retest in the engine
# ---------------------------------------------------------------------------

AC = REGISTRY["boeing747"]
# `boeing747` DECLARES the FC9 set since session 30, so the baseline for every
# "what did declaring do" row is this: the same entry with the seam shut, which
# is the 747 as it stood before.
BARE = AC._replace(mach_deriv_ref=jnp.array(-1.0), CL_M=jnp.array(0.0),
                   CD_M=jnp.array(0.0), Cm_M=jnp.array(0.0))
V, H = CRUISE["boeing747"]["airspeed"], CRUISE["boeing747"]["altitude"]
M_REF = cm.IX3[9][1]


def engine_modes(ac):
    x, _ = trim(jnp.array(V), jnp.array(H), ac)
    alpha, de, th = (float(v) for v in x)
    return cm.modes(longitudinal_matrix(ac, alpha, de, th, V, H)), alpha


def korn_lock_slope(ac, alpha, mach):
    CL = float(ac.CL0) + float(ac.CLa) * alpha
    return float(jax.grad(lambda m: aero.wave_drag(m, jnp.array(CL), ac))(jnp.array(mach)))


def declare(CL_M, CD_M_total, Cm_M, base=BARE, **extra):
    """The 747 with a SOURCED total drag Mach slope: the field carries the source
    value minus the Korn/Lock slope the model already has (see the field)."""
    _, alpha = engine_modes(base)
    M0 = V / float(speed_of_sound(jnp.array(H)))
    return base._replace(mach_deriv_ref=jnp.array(M_REF), CL_M=jnp.array(CL_M),
                         CD_M=jnp.array(CD_M_total - korn_lock_slope(base, alpha, M0)),
                         Cm_M=jnp.array(Cm_M), **extra)


def fmt(e):
    return "  ".join(f"{k} {100 * e[k]:+7.2f}%" for k in MODES)


def retest():
    print("\n== 4. RETEST: FC9 set declared on the 747, re-trimmed, linearised by jacfwd ==")
    d = {q: cm.value(q, "40K", M_REF) for q in cm.QUANTITIES}
    print("  digitised at M 0.800, 40,000 ft (linear): " +
          "  ".join(f"{q} {d[q]:+.4f}" for q in cm.QUANTITIES))
    bare, alpha = engine_modes(BARE)
    M0 = V / float(speed_of_sound(jnp.array(H)))
    kl = korn_lock_slope(BARE, alpha, M0)
    b9 = cm.backsolve(9)
    print(f"  engine trim M {M0:.5f}; Korn/Lock wave-drag slope there {kl:.5f} per Mach "
          f"(sourced total: digitised {d['cd_m']:.4f}, Table IX-4 {b9['cd_m']:.4f})")
    print(f"  shipped entry declares CL_M {float(AC.CL_M):.4f}, Cm_M {float(AC.Cm_M):.4f}, "
          f"CD_M {float(AC.CD_M):+.4f} net (total {float(AC.CD_M) + kl:.4f} at the flown trim)")
    rows = {"bare entry, seam shut (the 747 before session 30)": bare}
    rows["SHIPPED boeing747, FC9 set declared"] = engine_modes(AC)[0]
    sourced = declare(d["cl_m"], d["cd_m"], d["cm_m"])
    rows["  a copy, CD_M net of Korn/Lock at the flown trim"] = engine_modes(sourced)[0]
    rows["  ...Korn/Lock slope kept as CD_M"] = engine_modes(
        BARE._replace(mach_deriv_ref=jnp.array(M_REF), CL_M=jnp.array(d["cl_m"]),
                      Cm_M=jnp.array(d["cm_m"])))[0]
    rows["  ...digitised CD_M ADDED to Korn/Lock"] = engine_modes(
        BARE._replace(mach_deriv_ref=jnp.array(M_REF), CL_M=jnp.array(d["cl_m"]),
                      CD_M=jnp.array(d["cd_m"]), Cm_M=jnp.array(d["cm_m"])))[0]
    cm_eff = d["cm_m"] + b9["Cm_trim"] / (M_REF / 2.0)
    rows[f"  ...DIAGNOSTIC Cm_M + C_m,trim/(M/2) = {cm_eff:.4f}"] = engine_modes(
        declare(d["cl_m"], d["cd_m"], cm_eff))[0]
    rows["IX-4 implied (table back-solve) set"] = engine_modes(
        declare(b9["cl_m"], b9["cd_m"], b9["cm_m"]))[0]
    # Alpha-dot family, analysis only: digitised Cm_alphadot, and CL_alphadot from
    # Table IX-4's Zwd (p. 221 does not plot it). Appendix A: Zwd = -(rho S c/4m)(U0/VTo) CN_adot.
    rho_t, U0_t = 2 * 177.0 / 774.0**2, 774.0 * math.cos(math.radians(4.6))
    cl_adot = -cm.IX4[9][6] * 4 * (cm.W_LB / cm.G_FTS2) / (rho_t * cm.S_FT2 * cm.CBAR_FT) * 774.0 / U0_t
    rows[f"speed + alpha-dot (Cm_adot {d['cm_alpha_dot']:.3f} dig, CL_adot {cl_adot:.2f} IX-4)"] = engine_modes(
        declare(d["cl_m"], d["cd_m"], d["cm_m"], Cmadot=jnp.array(d["cm_alpha_dot"]),
                CLadot=jnp.array(cl_adot)))[0]
    # AUDIT.md section 2.3's instrument, for comparison: Table IX-4's own starred
    # Xu, Zu, Mu written straight into the engine's imperial plant matrix.
    from atisim.validation import to_imperial_matrix

    x, _ = trim(jnp.array(V), jnp.array(H), BARE)
    A = to_imperial_matrix(longitudinal_matrix(BARE, *(float(v) for v in x), V, H))
    A[0, 0], A[1, 0], A[2, 0] = cm.IX4[9][0], cm.IX4[9][1], cm.IX4[9][2]
    rows["(AUDIT 2.3: IX-4 Xu*, Zu*, Mu* substituted into the matrix)"] = cm.modes(A)
    for label, m in rows.items():
        print(f"  {label:58s} {fmt(cm.errors_vs_ix5(m))}")
    return d, bare, rows


# ---------------------------------------------------------------------------
# 5. Price the reading
# ---------------------------------------------------------------------------

def price(d, resid, n_mc: int, seed: int):
    print("\n== 5. PRICE: how much of the retest is the hand reading ==")
    x, _ = trim(jnp.array(V), jnp.array(H), BARE)
    alpha, de, th = (float(v) for v in x)
    A0 = longitudinal_matrix(BARE, alpha, de, th, V, H)
    a_s = float(speed_of_sound(jnp.array(H)))
    kl = korn_lock_slope(BARE, alpha, V / a_s)
    rho = float(density(jnp.array(H)))
    Iyy = float(np.asarray(AC.inertia)[1, 1])

    def err(clm, cdm, cmm):
        dA = cm.mach_increment(alpha, V, rho, a_s, float(AC.mass), Iyy, float(AC.S), float(AC.c),
                               clm, cdm - kl, cmm)
        return cm.errors_vs_ix5(cm.modes(A0 + dA))

    base = err(d["cl_m"], d["cd_m"], d["cm_m"])
    print(f"  central (linear read, analytic Appendix-A increment): {fmt(base)}")
    env = {k: [base[k], base[k]] for k in MODES}

    def widen(e):
        for k in MODES:
            env[k][0], env[k][1] = min(env[k][0], e[k]), max(env[k][1], e[k])

    print("\n  (a) interpolation scheme through the same points")
    schemes = {"linear": lambda c: np.interp(M_REF, c.mach, c.value),
               "pchip": lambda c: PchipInterpolator(c.mach, c.value)(M_REF),
               "natural cubic": lambda c: CubicSpline(c.mach, c.value, bc_type="natural")(M_REF),
               "akima": lambda c: Akima1DInterpolator(c.mach, c.value)(M_REF)}
    for name, f in schemes.items():
        v = {q: float(f(cm.curves()[(q, "40K")])) for q in SPEED}
        e = err(*(v[q] for q in SPEED))
        widen(e)
        print(f"    {name:14s} " + " ".join(f"{q} {v[q]:+.4f}" for q in SPEED) + f"   {fmt(e)}")

    print("\n  (b) leave one point out, within M 0.70-0.90 (pchip)")
    for q in SPEED:
        c = cm.curves()[(q, "40K")]
        for i in np.where((c.mach >= 0.70) & (c.mach <= 0.90))[0]:
            keep = np.arange(c.mach.size) != i
            c2 = c._replace(mach=c.mach[keep], value=c.value[keep])
            v = dict(d)
            v[q] = cm.value(q, "40K", M_REF, "pchip", curve=c2)
            if not math.isfinite(v[q]):
                continue
            e = err(v["cl_m"], v["cd_m"], v["cm_m"])
            widen(e)
            print(f"    drop {q} point at M {c.mach[i]:.4f}: {q} {v[q]:+.4f}   {fmt(e)}")

    print(f"\n  (c) Monte Carlo on the sheets' own pixels, N = {n_mc}, seed {seed}")
    rng = np.random.default_rng(seed)
    mc = {}
    for sigma in (1.0, 2.0, 3.0):
        vals, errs = [], []
        for _ in range(n_mc):
            v = {q: cm.value(q, "40K", M_REF, curve=cm.perturbed(cm.curves()[(q, "40K")], rng, sigma))
                 for q in SPEED}
            if not all(math.isfinite(v[q]) for q in SPEED):
                continue
            vals.append([v[q] for q in SPEED])
            errs.append([err(*vals[-1])[k] for k in MODES])
        vals, errs = np.array(vals), np.array(errs)
        mc[sigma] = errs
        pv = np.percentile(vals, [5, 95], axis=0)
        pe = np.percentile(errs, [5, 50, 95], axis=0)
        print(f"    sigma {sigma:.0f} px ({len(errs)} valid): " +
              " ".join(f"{q} [{pv[0,i]:+.4f},{pv[1,i]:+.4f}]" for i, q in enumerate(SPEED)))
        print("        mode error 5/50/95%: " + "  ".join(
            f"{k} [{100*pe[0,j]:+.2f} {100*pe[1,j]:+.2f} {100*pe[2,j]:+.2f}]" for j, k in enumerate(MODES)))
        if sigma == 3.0:
            for j, k in enumerate(MODES):
                env[k][0], env[k][1] = min(env[k][0], pe[0, j]), max(env[k][1], pe[2, j])

    print("\n  (d) the CHECK residuals as an empirical reading error, +-2 RMS (other conditions only)")
    oat = {}
    for q in SPEED:
        others = np.array([r for fc, r, _ in resid[q] if fc != 9])
        rms = float(np.sqrt(np.mean(others**2))) if others.size else float("nan")
        for sign in (-2.0, 2.0):
            v = dict(d)
            v[q] = d[q] + sign * rms
            e = err(v["cl_m"], v["cd_m"], v["cm_m"])
            widen(e)
            oat.setdefault(q, []).append(e)
        print(f"    {q}: RMS {rms:.4f} over {others.size} conditions -> "
              f"ph_wn {100*oat[q][0]['ph_wn']:+.2f}..{100*oat[q][1]['ph_wn']:+.2f}%  "
              f"ph_z {100*oat[q][0]['ph_z']:+.2f}..{100*oat[q][1]['ph_z']:+.2f}%")

    print("\n  (e) one at a time, change in mode error per +0.01 of each derivative")
    for i, q in enumerate(SPEED):
        v = [d[s] + (0.01 if j == i else 0.0) for j, s in enumerate(SPEED)]
        e = err(*v)
        print(f"    {q}: " + "  ".join(f"{k} {100*(e[k]-base[k]):+.3f}" for k in MODES))

    print("\n  ENVELOPE over (a), (b), (c at 3 px, 5-95%) and (d):")
    print("    " + "  ".join(f"{k} [{100*env[k][0]:+.2f}, {100*env[k][1]:+.2f}]%" for k in MODES))
    return env, mc


# ---------------------------------------------------------------------------
# Headline consequence
# ---------------------------------------------------------------------------

def headline(d, dt: float):
    from atisim import vortex_viz, wind

    print(f"\n== HEADLINE: Mehta's Hannibal array at 37,000 ft, dt {dt} -- analysis only ==")
    Hm = wind.MEHTA_HANNIBAL_ALTITUDE
    array = wind.mehta_hannibal_array(Hm)
    field = lambda p: wind.vortex_wind(p, array)  # noqa: E731
    r0 = float(array.r0)
    x0, x1 = float(array.north.min()), float(array.north.max())
    start = x0 - 12.0 * r0
    seconds = (x1 + 12.0 * r0 - start) / V
    out = {}
    for label, ac in (("bare, before session 30", BARE), ("SHIPPED, FC9 set declared", AC)):
        enc = vortex_viz.fly_in_moving_air(ac, field, V, Hm, label=label, start_north=start,
                                           seconds=seconds, dt=dt, window=(x0 - 2 * r0, x1 + 2 * r0),
                                           window_name="array +- 2 r0")
        w = np.asarray(enc.window)
        nz, th = np.asarray(enc.n_z)[w], np.degrees(np.asarray(enc.theta)[w])
        out[label] = (float(nz.max()), float(nz.min()), float(nz.max() - nz.min()), float(np.ptp(th)))
        print(f"  {label:28s} n_z {out[label][1]:+.4f} .. {out[label][0]:+.4f}  "
              f"peak-to-peak {out[label][2]:.4f} g  ({out[label][2]/2.70:.1%} of 2.70 g)  pitch ptp {out[label][3]:.3f} deg")
    a, b = out["bare, before session 30"], out["SHIPPED, FC9 set declared"]
    print(f"  change: peak-to-peak {100*(b[2]-a[2])/a[2]:+.2f}%, pitch {100*(b[3]-a[3])/a[3]:+.2f}%")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dig-dir", type=Path, nargs="?", const=DIG_DIR,
                   help=f"folder holding the eight Engauge .dig files; bare flag uses {DIG_DIR}")
    p.add_argument("--write", action="store_true", help="regenerate the tracked CSV from --dig-dir")
    p.add_argument("--csv-dir", type=Path, help="folder holding Engauge's CSV exports, to audit them")
    p.add_argument("--mc", type=int, default=4000, help="Monte Carlo draws per pixel level")
    p.add_argument("--seed", type=int, default=20260915)
    p.add_argument("--headline", action="store_true", help="also fly the Mehta encounter")
    p.add_argument("--dt", type=float, default=0.01)
    args = p.parse_args()

    if args.dig_dir:
        compare_or_write(extract(args.dig_dir), args.write)
    if args.csv_dir:
        audit_exports(args.csv_dir)
    resid = check()
    d, _, _ = retest()
    price(d, resid, args.mc, args.seed)
    if args.headline:
        headline(d, args.dt)


if __name__ == "__main__":
    main()
