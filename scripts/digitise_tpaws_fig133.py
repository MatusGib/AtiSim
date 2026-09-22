"""TPAWS Figure 133: the measured peak sigma_u and sigma_w, read as VECTORS.

SOURCE. NASA/TM-2012-217337 (Hamilton, Proctor & Ahmad, Feb 2012, NTRS
20120003172), Figure 133, printed p. 128 (PDF page index 131). Caption:
"Corresponding peak values of sigma_u and sigma_w peaks for all 2002 turbulence
events. Computed from in situ 20 Hz wind data assuming a 5 second window."
Held outside git; pass `--pdf`.

THIS IS NOT A DIGITISATION IN THIS PROJECT'S USUAL SENSE, and the distinction
matters for what the numbers are worth. Every other figure in `atisim/data/`
was read off a raster at 300-600 dpi, with a pixel error budget. This figure is
BORN DIGITAL: the chart is vector art, each marker is a filled four-segment
path, and `page.get_drawings()` returns their exact coordinates. The axes are
calibrated on the printed tick labels' own text boxes, which are also vector.
So there is no reading uncertainty, only the author's own plotting precision,
and the calibration residual below measures what remains.

WHAT IT SUPPLIES. sigma_w per event -- the one term the gust-response-factor
test needs and the only one neither Table 1 nor Stewart provides. With it,
PROJECT.md section 7's discriminating test closes on measured data.

THE POPULATION IS NOT TABLE 1'S, and this is the caveat that governs every use.
This figure carries **78** points for "all 2002 turbulence events". Table 1
carries **53 significant** events across BOTH campaigns, of which 49 are 2002 --
the same 49 the text scores on printed p. 5. So Figure 133's set CONTAINS
Table 1's 2002 subset and adds ~29 events that did not meet the sigma_dn >= 0.2 g
significance criterion. Pairing a marker to a Table 1 row is impossible: the
scatter is unlabelled. Any ratio of the two therefore has to be BRACKETED --
all 78 against the significant subset's 49 -- and both ends reported. Rank
matching the top 49 is not arbitrary: sigma_dn scales with sigma_w, so the
significance cut maps to a sigma_w cut, but it is a DECLARED choice and the
bracket is what is quoted.

CHECKS, none of which sets a parameter:
  1  CALIBRATION. The eleven printed tick labels per axis must fit a straight
     line. The residual is reported in DATA units and must be < 0.01, which on
     vector text is a formality -- and that is the point: it is what says the
     axes were read and not assumed.
  2  INSIDE THE BOX. Every marker must land within the printed 0-10 range on
     both axes.
  3  THE PAPER'S OWN ANISOTROPY CLAIM. Printed p. 127-128: "Isotropy would
     reveal a one to one relationship between the sigma_u and sigma_w peaks.
     However, the figure shows a bias towards higher values" of sigma_w. So a
     clear majority of markers must lie ABOVE the drawn 1:1 diagonal. This is
     the check that would catch swapped axes, which no range test can.
  4  COUNT. 78 markers, which is the figure's own content and is asserted so a
     re-extraction that loses some fails here.

Run: PYTHONPATH=<abs worktree root> .venv/Scripts/python.exe \\
         scripts/digitise_tpaws_fig133.py --pdf <refs/NASA-TM-2012-217337-...pdf> [--write]
"""

import argparse
import csv
from pathlib import Path

import numpy as np

import atisim

print(f"atisim imported from: {atisim.__file__}")

import fitz  # noqa: E402

DATA = Path(atisim.__file__).parent / "data" / "tpaws_fig133_sigma_uw.csv"

PAGE_HINT = "Corresponding peak values of"
MARKER_RGB = (0.9137254953384399, 0.15292592346668243, 0.15292592346668243)
N_MARKERS = 78
FIELDS = ("sigma_u_ms", "sigma_w_ms")


def find_page(doc):
    hits = [i for i in range(doc.page_count) if PAGE_HINT in doc[i].get_text()]
    if len(hits) != 1:
        raise SystemExit(f"expected one Figure 133 page, found {hits}")
    return hits[0]


def _axis(page, lo_y, hi_y, horizontal):
    """Fit tick VALUE against tick POSITION from the printed labels themselves."""
    pts = []
    for b in page.get_text("dict")["blocks"]:
        for line in b.get("lines", []):
            for s in line["spans"]:
                t = s["text"].strip()
                bb = s["bbox"]
                mid_y = (bb[1] + bb[3]) / 2.0
                if not (lo_y < mid_y < hi_y) or not t.isdigit():
                    continue
                pos = (bb[0] + bb[2]) / 2.0 if horizontal else mid_y
                pts.append((float(t), pos))
    pts.sort()
    if len(pts) != 11:
        raise SystemExit(f"expected 11 tick labels, found {[p[0] for p in pts]}")
    vals = np.array([v for v, _ in pts])
    pos = np.array([p for _, p in pts])
    slope, intercept = np.polyfit(pos, vals, 1)
    resid = float(np.max(np.abs(np.polyval([slope, intercept], pos) - vals)))
    return slope, intercept, resid


def markers(page):
    out = []
    for it in page.get_drawings():
        f = it.get("fill")
        if f and all(abs(a - b) < 1e-3 for a, b in zip(f, MARKER_RGB)):
            r = it["rect"]
            out.append(((r.x0 + r.x1) / 2.0, (r.y0 + r.y1) / 2.0))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pdf", type=Path, required=True)
    p.add_argument("--write", action="store_true", help=f"write {DATA}")
    args = p.parse_args()

    doc = fitz.open(args.pdf)
    page_no = find_page(doc)
    page = doc[page_no]
    print(f"\nFigure 133 on PDF page index {page_no}")

    xs, xi, xr = _axis(page, 286.0, 300.0, horizontal=True)
    ys, yi, yr = _axis(page, 120.0, 290.0, horizontal=False)
    raw = markers(page)
    sig_u = np.array([xs * x + xi for x, _ in raw])
    sig_w = np.array([ys * y + yi for _, y in raw])

    print("\nCHECKS")
    fails = []
    print(f"  1 CALIBRATION residual: sigma_u axis {xr:.2e}, "
          f"sigma_w axis {yr:.2e} data units")
    if max(xr, yr) >= 0.01:
        fails.append(f"1 CALIBRATION: residual {max(xr, yr):.3f} >= 0.01")
    if len(raw) != N_MARKERS:
        fails.append(f"4 COUNT: {len(raw)} markers, expected {N_MARKERS}")
    outside = [(round(a, 2), round(b, 2)) for a, b in zip(sig_u, sig_w)
               if not (0.0 <= a <= 10.0 and 0.0 <= b <= 10.0)]
    if outside:
        fails.append(f"2 INSIDE THE BOX: {outside}")
    above = int(np.sum(sig_w > sig_u))
    if above <= len(raw) * 0.5:
        fails.append(f"3 ANISOTROPY: only {above}/{len(raw)} above the 1:1 line")
    for f in fails:
        print(f"  FAIL  {f}")
    if not fails:
        print(f"  2 all {len(raw)} markers inside the printed 0-10 box")
        print(f"  3 ANISOTROPY: {above}/{len(raw)} = {100 * above / len(raw):.0f}% "
              f"lie above the 1:1 line, which is the paper's own claim")
        print(f"  4 COUNT: {len(raw)} markers")

    print(f"\nsigma_w (m/s): mean {sig_w.mean():.3f}  median "
          f"{np.median(sig_w):.3f}  sd {sig_w.std(ddof=1):.3f}  "
          f"range {sig_w.min():.2f}-{sig_w.max():.2f}")
    print(f"sigma_u (m/s): mean {sig_u.mean():.3f}  median "
          f"{np.median(sig_u):.3f}  range {sig_u.min():.2f}-{sig_u.max():.2f}")
    top49 = np.sort(sig_w)[-49:]
    print(f"\nthe TOP 49 by sigma_w, the rank-matched stand-in for Table 1's "
          f"2002 significant subset:\n  mean {top49.mean():.3f}  "
          f"range {top49.min():.2f}-{top49.max():.2f}")

    if args.write:
        DATA.parent.mkdir(parents=True, exist_ok=True)
        with open(DATA, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(FIELDS)
            for a, b in zip(sig_u, sig_w):
                w.writerow([f"{a:.4f}", f"{b:.4f}"])
        print(f"\nwrote {len(raw)} rows -> {DATA}")
    raise SystemExit(0 if not fails else 1)


if __name__ == "__main__":
    main()
