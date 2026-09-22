"""NASA/TM-2012-217337 Table 1: the TPAWS significant turbulence events, ingested.

SOURCE. D. W. Hamilton, F. H. Proctor, N. N. Ahmad, "Flight Tests of the
Turbulence Prediction and Warning System (TPAWS)", NASA/TM-2012-217337, February
2012, NTRS 20120003172 -- a work of the US Government, "Public Use Permitted".
Held outside git (`docs/DEVELOPMENT.md` rule 5); pass its path with `--pdf`.
Table 1, "Summary of Significant Turbulence Events", printed p. 8 (PDF page
index 11). The table EXTRACTS FROM THE TEXT LAYER: no digitisation, no reading
uncertainty, no calibration step.

WHY. The turbulence response validation design, phase S0
(`docs/design/specs/2026-09-21-turbulence-response-validation-design.md`). The
aircraft is NASA Langley's B-757 ARIES and THE WEIGHT IS TABULATED FOR EVERY ROW
-- the exact quantity whose absence makes PROJECT.md 5.19 structurally
impossible for Hannibal. Phase S2 then takes the peak factor from these rows.

WHAT THE COLUMNS ARE, read off the document rather than assumed, because the
peak factor S2 computes is a ratio of two of them and rule 2 forbids guessing
which window each is over.

  sigma_dn_g   printed p. 5 defines the RMS of normal load acceleration as a
               function of time over a SHIFTING / SLIDING window, tau = 5 s
               ("5 seconds has been chosen"). A single number per event is
               therefore a reduction of that running quantity, and printed
               p. 125 names the reduction in so many words -- "the peak
               sigma_dn". Table 1's column group header, "Peak In Situ
               Turbulence (g's)", governs all three load columns.
               SO: the MAXIMUM OVER THE ENCOUNTER OF A 5 s SLIDING-WINDOW RMS.
               It is NOT a whole-record sigma, and the two are not
               interchangeable: a running maximum is >= the whole-record value,
               so a peak factor built on this denominator is SMALLER than one
               built on a whole-record sigma.
  dn_max_g     the encounter's extreme normal load, printed p. 7: "the peak
  dn_min_g     normal load acceleration". Figure 2, printed p. 6, plots this
               against sigma_dn over 606 encounters "assuming a 5 second window"
               (from Bowles and Buck 2009).
  weight_klb   aircraft weight. tas_ms is true airspeed, m/s.
  wind_up_ms   peak vertical wind, both signs, m/s.
  reflectivity_dbz   peak radar reflectivity factor for the event.
  long_pulse   the printed asterisk: "Events where the radar was in long pulse
               mode, and which are not included in radar scoring."

ALTITUDE IS NOT ALWAYS A SCALAR, and this is where a previous parse went wrong.
Two of the 53 rows print a RANGE -- 232-05 "31 to 35" and 235-05* "22 to 19"
(a descent, so printed high-to-low, not low-to-high). They are carried as
`altitude_kft_first` and `altitude_kft_last` exactly as printed, with the printed
string kept beside them in `altitude_kft_raw`; for the other 51 rows the two are
equal. Nothing is invented and no range is collapsed to a midpoint.

CHECKS, and none of them sets a parameter. Each is the document's own statement
about its own table, so a parse that drifts fails here rather than silently.
  1  ROW COUNT. Printed p. 7: "A summary of the key in situ parameters from 53
     significant turbulence events is given in Table 1." Printed p. 5 agrees:
     "ARIES tallied 53 encounters with CIT."
  2  THE SIGNIFICANCE CRITERION. Printed p. 7: "An event was classified as
     significant turbulence if sigma_dn >= 0.2 g." Every row must satisfy it.
  3  SIGN DISCIPLINE. dn_max > 0 > dn_min and wind_up > 0 > wind_down, per row.
  4  REFLECTIVITY BAND. Printed p. 7: "peak values for each event ranging from
     0 to 40 dBz".

Run: PYTHONPATH=<abs worktree root> .venv/Scripts/python.exe \\
         scripts/tpaws_table1_ingest.py --pdf <refs/NASA-TM-2012-217337-...pdf> [--write]
"""

import argparse
import csv
import re
from pathlib import Path

import atisim

print(f"atisim imported from: {atisim.__file__}")

import fitz  # noqa: E402

DATA = Path(atisim.__file__).parent / "data" / "tpaws_tm2012_217337_table1.csv"

CAPTION = "Summary of Significant Turbulence Events"
N_EVENTS_PRINTED = 53  # printed p. 7, the document's own count
SIGMA_SIGNIFICANT_G = 0.2  # printed p. 7, the document's own criterion
DBZ_BAND = (0.0, 40.0)  # printed p. 7

FIELDS = (
    "event", "altitude_kft_raw", "altitude_kft_first", "altitude_kft_last",
    "weight_klb", "tas_ms", "sigma_dn_g", "dn_max_g", "dn_min_g",
    "wind_up_ms", "wind_down_ms", "reflectivity_dbz", "long_pulse",
)

_EVENT = re.compile(r"^(\d{3}-\d{2})(\*?)$")
_RANGE = re.compile(r"^(\d+)\s+to\s+(\d+)$")


def _lines(doc, page):
    return [ln.strip() for ln in doc[page].get_text().splitlines() if ln.strip()]


def find_table_page(doc):
    """The page carrying Table 1, located by its caption rather than by index."""
    pages = [i for i in range(doc.page_count)
             if CAPTION in doc[i].get_text()
             and sum(bool(_EVENT.match(ln)) for ln in _lines(doc, i)) > 10]
    if len(pages) != 1:
        raise SystemExit(f"expected exactly one Table 1 page, found {pages}")
    return pages[0]


def parse(doc, page):
    """Each cell sits on its own line, so a row is an event id plus nine cells."""
    lines = _lines(doc, page)
    rows = []
    for i, line in enumerate(lines):
        m = _EVENT.match(line)
        if not m:
            continue
        cells = lines[i + 1:i + 10]
        if len(cells) != 9:
            raise SystemExit(f"{m.group(1)}: expected 9 cells, got {cells}")
        alt_raw = cells[0]
        r = _RANGE.match(alt_raw)
        first, last = ((float(r.group(1)), float(r.group(2))) if r
                       else (float(alt_raw), float(alt_raw)))
        weight, tas, sigma, dnmax, dnmin, wup, wdn, dbz = (float(c) for c in cells[1:])
        rows.append(dict(zip(FIELDS, (
            m.group(1), alt_raw, first, last, weight, tas, sigma, dnmax, dnmin,
            wup, wdn, dbz, bool(m.group(2))))))
    return rows


def check(rows):
    """The document's own statements about its own table. A failure is a parse bug."""
    fails = []
    if len(rows) != N_EVENTS_PRINTED:
        fails.append(f"1 ROW COUNT: {len(rows)} parsed, printed p.7 says {N_EVENTS_PRINTED}")
    bad = [r["event"] for r in rows if r["sigma_dn_g"] < SIGMA_SIGNIFICANT_G]
    if bad:
        fails.append(f"2 SIGNIFICANCE: sigma below {SIGMA_SIGNIFICANT_G} g in {bad}")
    bad = [r["event"] for r in rows
           if not (r["dn_max_g"] > 0 > r["dn_min_g"]
                   and r["wind_up_ms"] > 0 > r["wind_down_ms"])]
    if bad:
        fails.append(f"3 SIGNS: extremes not straddling zero in {bad}")
    bad = [r["event"] for r in rows
           if not DBZ_BAND[0] <= r["reflectivity_dbz"] <= DBZ_BAND[1]]
    if bad:
        fails.append(f"4 REFLECTIVITY: outside {DBZ_BAND} dBz in {bad}")
    for f in fails:
        print(f"  FAIL  {f}")
    if not fails:
        print(f"  all four checks pass on {len(rows)} rows")
    return not fails


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pdf", type=Path, required=True,
                   help="NASA/TM-2012-217337, held outside git")
    p.add_argument("--write", action="store_true", help=f"write {DATA}")
    args = p.parse_args()

    doc = fitz.open(args.pdf)
    page = find_table_page(doc)
    print(f"\nTable 1 on PDF page index {page}, printed page {_lines(doc, page)[0]}")

    rows = parse(doc, page)
    print("\nCHECKS")
    ok = check(rows)

    print(f"\nRANGES ACROSS {len(rows)} ROWS")
    alts = [v for r in rows for v in (r["altitude_kft_first"], r["altitude_kft_last"])]
    print(f"  {'altitude_kft':<19} {min(alts):6.0f} - {max(alts):6.0f}  kft")
    for key, unit in (("weight_klb", "klb"), ("tas_ms", "m/s"), ("sigma_dn_g", "g"),
                      ("dn_max_g", "g"), ("dn_min_g", "g"),
                      ("wind_up_ms", "m/s"), ("wind_down_ms", "m/s")):
        lo = min(r[key] for r in rows)
        hi = max(r[key] for r in rows)
        print(f"  {key:<19} {lo:6.2f} - {hi:6.2f}  {unit}")
    ranged = [r["event"] for r in rows
              if r["altitude_kft_first"] != r["altitude_kft_last"]]
    print(f"  rows with a RANGED altitude: {ranged}")
    print(f"  rows flagged long-pulse:     {sum(r['long_pulse'] for r in rows)}")

    if args.write:
        DATA.parent.mkdir(parents=True, exist_ok=True)
        with open(DATA, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {len(rows)} rows -> {DATA}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
