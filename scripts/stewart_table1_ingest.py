"""NASA/TM-2003-212666 Table 1: a published B-757 normal-force set, ingested.

SOURCE. E. C. Stewart, "Description of a Normal-Force In-Situ Turbulence
Algorithm for Airplanes", NASA/TM-2003-212666, December 2003, NTRS 20040021314.
A NASA work. Held outside git (`docs/DEVELOPMENT.md` rule 5); pass `--pdf`.
Table 1 "Aerodynamic parameters", printed pp. 12-13, extracts from the TEXT
LAYER. Three sub-tables -- C_Nalpha, C_Nq, C_Ndelta, all per radian -- each
tabulated against dynamic pressure at 0, 10, 20, 30 and 40 kft pressure
altitude, for the clean configuration at a nominal 180,000 lb (printed p. 4).

WHY THIS AND NOT THE FLIGHT ILLUSTRATION, which is what design phase S1 went
looking for. The paper's severe-turbulence encounter CANNOT serve as a
validation target for this project, for three independent reasons, and the
third is the paper's own:

  1  ALTITUDE. 33,000 ft is 2,000 ft below `boeing747.valid_altitude`'s floor,
     so `checks.recovery_band` refuses the run. PROJECT.md 5.3's band objection.
  2  CIRCULARITY. Stewart's "measured" vertical gust is not an independent
     observation of the atmosphere. Eq. (4) obtains it by dividing the measured
     normal acceleration by the airplane's OWN C_Nalpha, plus small elevator,
     pitch-rate and attitude corrections. Driving this project's model with
     that gust and comparing the load back would test whether two lift-curve
     slopes agree, wearing the costume of a time-history validation.
  3  THE AUTHOR SAYS SO. Printed p. 7: the measurements "were combined
     asynchronously on a common data bus and are, therefore, not representative
     of an operational data stream. The results that are shown herein are,
     therefore, only for illustrative purposes." He repeats it twice more.

So the figures are not digitised and no number is taken from them. TABLE 1 IS
FREE OF ALL THREE OBJECTIONS: it is tabulated, it is the airplane rather than
the encounter, and the author states no caveat over it.

WHAT IT IS GOOD FOR. It is an INDEPENDENT published lift-curve slope for a
transport aircraft across altitude and dynamic pressure -- and through the
document's own Eq. (8) each column carries a Mach number, so C_Nalpha(M) falls
out. That is a first-hand external check on this project's compressibility
treatment (`ASSUMPTIONS` C3, `pg_mach_ref`), which until now had none: every
Prandtl-Glauert claim in PROJECT.md is checked against the model's own algebra
rather than against a published set. NOTE THE CONFOUND before using it: at
fixed altitude and weight a higher q is also a lower trim angle of attack, so a
column mixes Mach with trim-alpha nonlinearity, and Stewart tabulates the total
NORMAL force, not lift. Both are stated in PROJECT.md section 4 beside the
comparison, and neither is resolved here.

CHECKS, none of which sets a parameter:
  1  SHAPE. 15 blocks -- 5 altitudes x 3 coefficients -- in document order.
  2  THE SHARED GRID. The dynamic-pressure grid must be IDENTICAL across all
     three coefficients at a given altitude. It is printed three times, so a
     misparse of any one of them shows up here.
  3  ALTITUDES are 0, 10, 20, 30, 40 kft, and 40 kft carries 4 columns where
     the others carry 5.
  4  MACH. Reconstructing M from q and h through the paper's own Eq. (8) must
     land every column in 0.2 < M < 0.95. A block read in the wrong order
     fails this immediately.

Run: PYTHONPATH=<abs worktree root> .venv/Scripts/python.exe \\
         scripts/stewart_table1_ingest.py --pdf <refs/NASA-TM-2003-212666-...pdf> [--write]
"""

import argparse
import csv
import math
import re
from pathlib import Path

import atisim

print(f"atisim imported from: {atisim.__file__}")

import fitz  # noqa: E402

DATA = Path(atisim.__file__).parent / "data" / "stewart_tm2003_212666_table1.csv"

COEFFS = ("CNalpha", "CNq", "CNdelta")  # document order, printed pp. 12-13
ALTITUDES_KFT = (0.0, 10.0, 20.0, 30.0, 40.0)
FIELDS = ("coefficient", "altitude_kft", "q_psf", "value", "mach")

_BLOCK = re.compile(r"^(\d+)\s+kft$")
_NUM = re.compile(r"^-?\d+(?:\.\d+)?$")


def dynamic_pressure_ratio(h_ft):
    """The paper's own Eq. (8), printed p. 8. Pressure ratio eta at h_ft."""
    if h_ft <= 36089.0:
        return (1.0 - 6.87535e-6 * h_ft) ** 5.2561
    return 0.22336 * math.exp(-4.80634e-5 * (h_ft - 36089.0))


def mach_from_q(q_psf, h_ft):
    """Invert q = 1481 * eta * M^2, Eq. (8)."""
    return math.sqrt(q_psf / (1481.0 * dynamic_pressure_ratio(h_ft)))


def parse(doc):
    """Table 1 spans two pages; each block is 'N kft' then n grid then n values."""
    lines = []
    for page in range(doc.page_count):
        t = doc[page].get_text()
        if not ("Aerodynamic parameters" in t or "Table 1--concluded" in t
                or ("Table 1" in t and "concluded" in t)):
            continue
        # Each page opens with its own printed page number on a line of its
        # own, and 12 and 13 would otherwise be read as table cells -- the
        # second page's number lands inside the 40 kft block that straddles
        # the page break, where it made the cell count odd.
        page_lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
        lines += page_lines[1:]
    blocks = []
    idx = [i for i, ln in enumerate(lines) if _BLOCK.match(ln)]
    for start, end in zip(idx, idx[1:] + [len(lines)]):
        alt = float(_BLOCK.match(lines[start]).group(1))
        nums = []
        for ln in lines[start + 1:end]:
            if _NUM.match(ln):
                nums.append(float(ln))
            elif nums:
                break
        if len(nums) % 2:
            raise SystemExit(f"{alt} kft: odd number of cells {nums}")
        half = len(nums) // 2
        blocks.append((alt, nums[:half], nums[half:]))
    return blocks


def rows_from(blocks):
    if len(blocks) != len(COEFFS) * len(ALTITUDES_KFT):
        raise SystemExit(f"expected 15 blocks, parsed {len(blocks)}")
    out = []
    for i, (alt, grid, vals) in enumerate(blocks):
        coeff = COEFFS[i // len(ALTITUDES_KFT)]
        for q, v in zip(grid, vals):
            out.append(dict(zip(FIELDS, (coeff, alt, q, v,
                                         mach_from_q(q, alt * 1000.0)))))
    return out


def check(blocks, rows):
    fails = []
    for k, coeff in enumerate(COEFFS):
        alts = [blocks[k * 5 + j][0] for j in range(5)]
        if tuple(alts) != ALTITUDES_KFT:
            fails.append(f"3 ALTITUDES for {coeff}: {alts}")
    for j, alt in enumerate(ALTITUDES_KFT):
        grids = [blocks[k * 5 + j][1] for k in range(len(COEFFS))]
        if any(g != grids[0] for g in grids[1:]):
            fails.append(f"2 SHARED GRID at {alt} kft: {grids}")
        want = 4 if alt == 40.0 else 5
        if len(grids[0]) != want:
            fails.append(f"3 WIDTH at {alt} kft: {len(grids[0])}, expected {want}")
    bad = [(r["altitude_kft"], r["q_psf"], round(r["mach"], 3)) for r in rows
           if not 0.2 < r["mach"] < 0.95]
    if bad:
        fails.append(f"4 MACH outside 0.2-0.95: {sorted(set(bad))}")
    for f in fails:
        print(f"  FAIL  {f}")
    if not fails:
        print(f"  all four checks pass on {len(rows)} rows")
    return not fails


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pdf", type=Path, required=True)
    p.add_argument("--write", action="store_true", help=f"write {DATA}")
    args = p.parse_args()

    doc = fitz.open(args.pdf)
    blocks = parse(doc)
    rows = rows_from(blocks)
    print(f"\nparsed {len(blocks)} blocks, {len(rows)} rows\n\nCHECKS")
    ok = check(blocks, rows)

    print("\nC_Nalpha AGAINST MACH, which is what makes this worth holding")
    print(f"  {'alt':>6} {'q_psf':>8} {'Mach':>7} {'CNalpha':>9} {'1/sqrt(1-M^2)':>14}")
    for r in rows:
        if r["coefficient"] != "CNalpha":
            continue
        pg = 1.0 / math.sqrt(1.0 - r["mach"] ** 2)
        print(f"  {r['altitude_kft']:6.0f} {r['q_psf']:8.2f} {r['mach']:7.3f} "
              f"{r['value']:9.4f} {pg:14.4f}")

    if args.write:
        DATA.parent.mkdir(parents=True, exist_ok=True)
        with open(DATA, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            for r in rows:
                w.writerow({**r, "mach": f"{r['mach']:.6f}"})
        print(f"\nwrote {len(rows)} rows -> {DATA}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
