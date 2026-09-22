"""Stewart's measured C_Nalpha(M) against this model's Prandtl-Glauert treatment.

WHY THIS EXISTS. PROJECT.md section 4 records the Prandtl-Glauert machinery as
PRESENT AND INERT: `Aircraft.pg_mach_ref` is -1.0 on every registry entry, which
the code reads as undeclared and makes the factor exactly 1, so every aircraft
flies at M 0.70-0.90 with NO compressibility correction to its lift slope. That
was known. What was missing was any outside measurement of what being inert
costs, because every Prandtl-Glauert claim in this record had been checked
against the model's own algebra. Stewart's Table 1 is such a measurement: a
published B-757 normal-force slope over M 0.242-0.851.

HOW THE TRIM CONFOUND IS BEATEN, which is the only reason this is a comparison
and not a scatter plot. At fixed altitude, raising dynamic pressure raises Mach
AND lowers the trim angle of attack, so a column of Stewart's table mixes the
two and cannot isolate either. But trim lift is C_L = W/(qS): at FIXED q and
fixed weight the trim C_L is fixed too. Stewart tabulated at roughly matched
dynamic pressures across his five altitudes, so reading DOWN a matched-q column
varies Mach at near-constant trim C_L. The residual q spread per column is
reported beside every result and is what limits the claim.

THE 40 kft ROW IS EXCLUDED from the column comparison and reported separately.
Its dynamic-pressure grid is not matched to the others -- 113/136/161/186 psf
against ~86/132/204/288 -- so including it would reintroduce the confound the
columns exist to remove. Excluded, the 0-30 kft columns match to 4.5-10.7% in q;
included, three of them exceed 30%.

WHAT IS COMPARED. Per column, the measured C_Nalpha ratio between its lowest and
highest Mach, against the 2-D Prandtl-Glauert ratio sqrt(1-M_lo^2)/sqrt(1-M_hi^2)
-- which is the exact factor `aero.py` would apply if an entry declared a
reference Mach. The ratio of those two is what the model's form gets wrong.

WHAT THIS IS NOT. C_N is total normal force and CLa is lift, the aircraft is a
B-757 and none of this project's entries is one, and twenty points carrying a
few per cent of residual trim variation cannot resolve a functional form to
better than the scatter reported. This measures a SHAPE against a SHAPE and is
quoted that way, per rule 6.

Run: PYTHONPATH=<abs worktree root> .venv/Scripts/python.exe \\
         scripts/stewart_mach_comparison.py
"""

import csv
import math
from pathlib import Path

import atisim

print(f"atisim imported from: {atisim.__file__}")

from atisim.aircraft import REGISTRY  # noqa: E402

DATA = Path(atisim.__file__).parent / "data" / "stewart_tm2003_212666_table1.csv"
COLUMN_ALTITUDES = (0.0, 10.0, 20.0, 30.0)  # 40 kft excluded, see docstring
S_757_FT2 = 1951.0
W_757_LB = 180000.0  # Stewart printed p. 4, the nominal the table is built at
PG_MACH_MAX = 0.9


def prandtl_glauert(m_lo, m_hi):
    """Exactly the factor aero.py applies, referenced from m_lo to m_hi."""
    lo = math.sqrt(max(1.0 - min(m_lo, PG_MACH_MAX) ** 2, 0.0))
    hi = math.sqrt(max(1.0 - min(m_hi, PG_MACH_MAX) ** 2, 1e-6))
    return lo / hi


def load():
    by = {}
    with open(DATA, newline="") as f:
        for r in csv.DictReader(f):
            if r["coefficient"] != "CNalpha":
                continue
            by.setdefault(float(r["altitude_kft"]), []).append(
                (float(r["q_psf"]), float(r["mach"]), float(r["value"])))
    for k in by:
        by[k].sort()
    return by


def main():
    by = load()

    print("\nCHECK  the inertness this comparison is about")
    inert = []
    for name, ac in REGISTRY.items():
        ref = float(ac.pg_mach_ref)
        inert.append((name, ref))
    live = [n for n, r in inert if r >= 0.0]
    print(f"   {len(inert)} registry entries, {len(live)} declare a reference "
          f"Mach -> Prandtl-Glauert factor is exactly 1 on "
          f"{len(inert) - len(live)} of them")
    if live:
        print(f"   FAIL  these now declare one and the framing below is stale: {live}")

    print(f"\nA  MACH AT NEAR-CONSTANT TRIM, reading down matched-q columns "
          f"({', '.join(f'{a:.0f}' for a in COLUMN_ALTITUDES)} kft)")
    print(f"   {'col':>3} {'q spread':>9} {'C_L trim':>9} {'Mach':>13} "
          f"{'C_Nalpha':>15} {'measured':>9} {'Prandtl-G':>10} {'ratio':>7}")
    rows = []
    for col in range(5):
        entries = [(a, *by[a][col]) for a in COLUMN_ALTITUDES if len(by[a]) > col]
        if len(entries) < 4:
            continue
        qs = [e[1] for e in entries]
        spread = (max(qs) - min(qs)) / min(qs)
        lo, hi = entries[0], entries[-1]
        meas = hi[3] / lo[3]
        pg = prandtl_glauert(lo[2], hi[2])
        cl = W_757_LB / (sum(qs) / len(qs) * S_757_FT2)
        rows.append((col, spread, cl, lo, hi, meas, pg))
        print(f"   {col:3d} {100 * spread:8.1f}% {cl:9.3f} "
              f"{lo[2]:.3f}->{hi[2]:.3f} {lo[3]:6.3f}->{hi[3]:6.3f} "
              f"{100 * (meas - 1):+8.1f}% {100 * (pg - 1):+9.1f}% "
              f"{meas / pg:7.3f}")

    print("\n   'ratio' is measured over Prandtl-Glauert: 1.000 would mean the "
          "model's\n   form is right. Every column is BELOW 1, so the form "
          "over-predicts throughout.")

    print(f"\nB  WHERE THE RISE ACTUALLY IS -- every 0-30 kft point, pooled by "
          f"Mach band")
    pts = sorted((m, v) for a in COLUMN_ALTITUDES for _, m, v in by[a])
    bands = ((0.20, 0.45), (0.45, 0.60), (0.60, 0.72), (0.72, 0.90))
    print(f"   {'Mach band':>14} {'n':>3} {'mean C_Nalpha':>14}")
    means = []
    for b0, b1 in bands:
        vals = [v for m, v in pts if b0 <= m < b1]
        if not vals:
            continue
        means.append((f"{b0:.2f}-{b1:.2f}", len(vals), sum(vals) / len(vals)))
        print(f"   {b0:.2f}-{b1:.2f}    {len(vals):3d} "
              f"{sum(vals) / len(vals):14.3f}")
    if len(means) >= 2:
        flat = means[1][2] / means[0][2]
        steep = means[-1][2] / means[1][2]
        print(f"\n   below M 0.60 the slope is FLAT to {100 * (flat - 1):+.1f}%; "
              f"from there to the top band it rises {100 * (steep - 1):+.1f}%.")
        print("   Prandtl-Glauert has no such knee -- it rises smoothly from "
              "M = 0 and\n   keeps rising toward M = 1, which is the shape "
              "disagreement, not a size one.")

    print("\nC  THE 40 kft ROW, held out of A because its q grid is not matched")
    for q, m, v in by[40.0]:
        print(f"   q {q:7.2f} psf  M {m:.3f}  C_Nalpha {v:6.3f}")
    print("   Reported, not used. Its q runs 113-186 psf against the others' "
          "86-415.")

    print("\nD  WHAT THE INERT FACTOR COSTS -- AN IMPLICATION, NOT A MEASUREMENT")
    low = [v for m, v in pts if m < 0.60]
    cruise = [v for m, v in pts if m >= 0.72]
    rise = (sum(cruise) / len(cruise)) / (sum(low) / len(low)) - 1.0
    print(f"   Both registry transports fly M 0.70-0.90, where this data puts "
          f"the lift\n   slope {100 * rise:+.1f}% above its sub-M-0.60 value "
          f"({sum(low) / len(low):.3f} -> {sum(cruise) / len(cruise):.3f}). "
          f"The model applies\n   no correction at all, so it flies the "
          f"low-Mach slope at cruise Mach.")
    ELASTICITY = 0.692  # CLa on the headline load, section 4, session 29
    print(f"\n   CLa's elasticity on the headline load is +{ELASTICITY} "
          f"(section 4, session 29),\n   so a {100 * rise:+.1f}% lift-slope "
          f"error implies roughly "
          f"{100 * ELASTICITY * rise:+.1f}% on the load,\n   against section "
          f"5's recorded 32% Hannibal shortfall.")
    print("\n   THREE REASONS THIS IS AN IMPLICATION AND NOT A CORRECTION, and "
          "none is small:")
    print("   1  that elasticity was measured over +/-1-5% and section 4 says "
          "so; this\n      extrapolates it five-fold and a tangent is not "
          "licensed that far.")
    print("   2  the elasticity is for the 747's Hannibal peak load, which is "
          "not the\n      quantity this Mach shape was read from.")
    print("   3  C_N is total normal force on a B-757; CLa is lift on aircraft "
          "that are\n      not one. The SHAPE transfers; the level does not.")
    print("\n   What it does support: the Mach axis is worth declaring, it "
          "points the same\n   way as the known shortfall, and a flat "
          "Prandtl-Glauert is the wrong form to\n   declare it with -- see A "
          "and B.")


if __name__ == "__main__":
    main()
