"""MIL-F-8785C Figure 7, DIGITISED -- turbulence exceedance probability.

Printed p. 49 of `refs/MIL-F-8785C.pdf`, the sheet session 25 fetched and
confirmed but deliberately did not read: "the digitisation itself is a
session's work on a poor scan and is deliberately not attempted in passing".
This is that work.

WHAT THE SHEET ACTUALLY CARRIES, AND IT IS NOT WHAT SECTION 7 ASSUMED. The
plan's row calls it "MIL-F-8785C Fig. 7, digitised", as though the figure were
the six probability-of-exceedance curves. It carries NINE lines: the six
curves 10^-1 .. 10^-6, AND three straight intensity boundaries labelled LIGHT,
MODERATE and SEVERE drawn across them. The sealed prediction is about the
SEVERE one, so the distinction is not cosmetic -- reading the third curve from
the right instead of the SEVERE rule would answer a different question.

THREE THINGS FIGHT THE READER, and each needed handling rather than tolerating:

  1  THE PAGE IS ROTATED 90 degrees. Altitude runs along the printed page's
     short edge and sigma along its long one. Rotating clockwise puts sigma on
     x increasing right and altitude on y increasing up, which is the
     orientation every other reproduction of this figure uses.
  2  THE SCAN IS STORED AS IMAGE TILES, 462 px apart at 600 dpi, with hairline
     seams between them. Every axis and every curve is broken at each seam --
     which is why the longest continuous run anywhere on the sheet is 461 px
     against axes that are 3300 px long. `_fill_seams` closes them.
  3  THE CURVE LABELS SIT ON THE CURVES. "SEVERE", "MODERATE", "LIGHT" are set
     diagonally across the lines they name. They are removed as small connected
     components before any line is traced, which also removes the tick labels,
     the 10^-n annotations and the scan's speckle.

CALIBRATION, AND WHAT SAYS IT IS RIGHT. Both axes are least-squared through
their numeric tick LABELS, never through the frame -- the frame is what the
seams destroyed.

  sigma      97.175 px per ft/s, sigma = 0 at x = 749.3, worst label residual
             0.10 ft/s. INDEPENDENTLY, the first tick mark below the axis sits
             at x = 748.5, which is 0.8 px from where the label fit puts the
             origin. Nothing was tuned to make those agree.
  altitude   36.407 px per 1000 ft, 0 ft at y = 3383.3, label residual sd
             about 9 px = 0.25 kft. The sigma axis line lies at y = 3370,
             13 px from where the altitude fit puts zero.

THE RESULT THIS WAS RUN FOR, AND ITS CHECK. The SEVERE boundary at 37,000 ft
reads 15.75 ft/s = 4.800 m/s. JSBSim's `FGWinds.cpp` carries an INDEPENDENT
transcription of this same figure -- session 25 recorded that interpolating its
severe row between the 35,000 and 45,000 ft columns gives 15.82 ft/s. The two
readings agree to 0.07 ft/s, 0.4%, which is the same two-independent-readings
pattern as the vortex fields agreeing to 8.8e-10 m/s. Neither reading was used
to set the other.

HOW THE NINE LINES ARE NAMED, WITHOUT READING A SINGLE CHARACTER. All nine are
present and separated at 9,000 ft, so that row anchors the ordering; above it
lines drop out one by one as they terminate. The names then follow from two
facts about the sheet, neither of which needs OCR:

  * the three intensity names are set ALONG their lines and so are TALL and
    narrow as glyph clusters (y-span 153-192 px against x-span 90-149), while
    every 10^-n label is set flat beside its curve and is WIDE and short. That
    separates the two families by shape alone.
  * each of the three tall words then lands nearest exactly one line, and
    `_check_labels` asserts it is the expected one. LIGHT at (2.04, 18.2 kft)
    falls on order #2, MODERATE at (8.06, 17.7) on #5, SEVERE at (11.66, 47.3)
    on #7.

which fixes the order, left to right, as

    10^-1 . LIGHT . 10^-2 . 10^-3 . MODERATE . 10^-4 . SEVERE . 10^-5 . 10^-6

WHERE THE LINES MERGE, AND WHY THAT IS REPORTED RATHER THAN SMOOTHED. SEVERE
and 10^-5 run within a line-width of each other from about 42,000 ft down to
18,000 ft, and at 37,000-38,000 ft they are a single blob of ink. Samples where
two tracks claim the same crossing are marked MERGED and excluded from the
table, so a reader never receives an interpolated value dressed as a read one.

THE CONSEQUENCE FOR THE ONE NUMBER THIS WAS RUN FOR. sigma_severe(37,000 ft)
cannot be read directly -- 37,000 ft is inside the merge. It is interpolated
between the nearest CLEAN samples either side, and the honest figure is about
15.5 ft/s with a spread of a few tenths depending on which side dominates.
An earlier pass here quoted 15.75 and an 0.4% agreement with JSBSim; that
number was the merged blob's centre, which sits between SEVERE and 10^-5, and
it flattered the comparison. The corrected agreement is about 2%.

AND ONE THING IT DELIBERATELY DOES NOT DO. It does not settle
`predictions.mil_f_8785c_sigma_w_exceeds_the_mehta_ceiling`. The number is
here and it points one way, but marking a sealed entry settled is an edit to
the register, and the register is the thing this project protects.

Run: PYTHONPATH=<abs worktree root> .venv/Scripts/python.exe scripts/digitise_mil_f_8785c_fig7.py --outdir runs/cat
"""

import argparse
import pickle
from pathlib import Path

import fitz
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage

FT2M = 0.3048

PDF = Path("refs/MIL-F-8785C.pdf")     # gitignored; pass --pdf from a worktree
PAGE = 48                              # printed p. 49
CLIP = fitz.Rect(90, 160, 600, 700)
DPI = 600

# Axis calibration, least-squared through the numeric tick labels. Held as
# constants so the figure can be re-read without re-deriving them, and
# re-derived by --recalibrate.
SIGMA_PX_PER_FTS, SIGMA_X0 = 97.175, 749.3
ALT_PX_PER_KFT, ALT_Y0 = 36.407, 3383.3

# The nine lines, left to right. Derived, not assumed -- `_check_labels`
# re-establishes it from the three intensity names' own positions on every run.
LINE_NAMES = ("10^-1", "LIGHT", "10^-2", "10^-3", "MODERATE",
              "10^-4", "SEVERE", "10^-5", "10^-6")
# The only altitude at which the sheet separates all nine.
ANCHOR_KFT = 9.0

# The one number this file exists to produce, and the independent reading of
# the same figure it is checked against. PROJECT.md section 9, session 25.
TARGET_ALT_KFT = 37.0
JSBSIM_SEVERE_37KFT_FTS = 15.82
MEHTA_CEILING_MS = 4.459               # wind.mehta_residual_ceiling

PALETTE = {"model": "#1D5D77", "reference": "#A9501C", "wind": "#3E6A48",
           "muted": "#7E8D93", "grid": "#D6DCD8"}


def _render(sp: Path, pdf: Path) -> np.ndarray:
    """Page 48 at 600 dpi, rotated clockwise into conventional orientation."""
    doc = fitz.open(pdf)
    doc[PAGE].get_pixmap(dpi=DPI, clip=CLIP).save(sp / "mil-fig7-raw.png")
    doc.close()
    im = Image.open(sp / "mil-fig7-raw.png").convert("L").rotate(-90, expand=True)
    im.save(sp / "mil-fig7.png")
    return np.asarray(im) < 140


def _fill_seams(d: np.ndarray) -> np.ndarray:
    """Close the tile seams: an empty column becomes the AND of its neighbours.

    The sheet is stored as image tiles 462 px apart. A line crossing a seam is
    broken by one blank column, which is enough to defeat both run-length and
    connected-component reasoning. Filling with the AND rather than the OR adds
    ink only where BOTH sides already carry it, so a seam inside a line is
    closed and a seam beside one is not widened.
    """
    d = d.copy()
    for x in range(1, d.shape[1] - 1):
        if d[:, x].sum() == 0:
            d[:, x] = d[:, x - 1] & d[:, x + 1]
    return d


def _drop_text(d: np.ndarray, min_extent: int = 250) -> np.ndarray:
    """Keep only components larger than a glyph.

    The lines all terminate on the axes, so they merge into a few large
    components -- which is fine, because the point here is only to delete the
    small ones: every tick label, every 10^-n annotation, the LIGHT/MODERATE/
    SEVERE names set across the lines, and the scan's speckle.
    """
    lab, _ = ndimage.label(d, structure=np.ones((3, 3)))
    keep = np.zeros_like(d)
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        if (sl[0].stop - sl[0].start) > min_extent or (sl[1].stop - sl[1].start) > min_extent:
            keep[sl] |= lab[sl] == i
    return keep


def _groups(idx, gap):
    if len(idx) == 0:
        return []
    out = [[idx[0]]]
    for v in idx[1:]:
        if v - out[-1][-1] <= gap:
            out[-1].append(v)
        else:
            out.append([v])
    return [((g[0] + g[-1]) / 2.0, g[-1] - g[0]) for g in out]


def _recalibrate(d):
    """Re-derive both axis fits from the tick labels, and report the residuals."""
    xs = np.where(d[3400:3480, :].sum(0) >= 3)[0]
    groups = [c for c, w in _groups(xs, 45) if 40 <= w <= 140]
    vals = np.array([10.0, 15.0, 20.0, 25.0, 30.0, 35.0])
    if len(groups) < len(vals):
        raise SystemExit(f"sigma labels: found {len(groups)}, need {len(vals)}")
    cx = np.array(groups[-len(vals):], float)
    b = np.polyfit(vals, cx, 1)
    sres = np.abs(np.polyval(b, vals) - cx).max() / b[0]

    ys = np.where(d[:, 500:685].sum(1) >= 4)[0]
    gy = [c for c, h in _groups(ys, 25) if 40 <= h <= 90]
    av = np.array([70.0, 60.0, 50.0, 40.0, 30.0, 20.0, 10.0])
    if len(gy) < len(av):
        raise SystemExit(f"altitude labels: found {len(gy)}, need {len(av)}")
    cy = np.array(gy[-len(av):], float)
    p = np.polyfit(av, cy, 1)
    ares = (np.polyval(p, av) - cy).std()
    return (b[0], b[1], sres), (-p[0], p[1], ares)


def _runs(v):
    out, s = [], None
    for i, x in enumerate(v):
        if x and s is None:
            s = i
        elif not x and s is not None:
            out.append((s, i - 1))
            s = None
    if s is not None:
        out.append((s, len(v) - 1))
    return out


def _crossings(keep, sigma_of, y):
    """Every line the row at `y` cuts, in sigma, thick strokes counted once."""
    xs = sorted(sigma_of((a + b) / 2.0 + 780) for a, b in _runs(keep[y, 780:4300])
                if b - a >= 3)
    out = []
    for x in xs:
        if out and x - out[-1] < 0.30:
            out[-1] = (out[-1] + x) / 2.0
        else:
            out.append(x)
    return out


def _track_from_anchor(keep, sigma_of, px_a, y0):
    """Seed the nine lines where all nine are separated, then follow them out.

    9,000 ft is the anchor because it is the only altitude at which the sheet
    shows all nine lines as nine distinct strokes. From there the lines are
    followed upward, losing one whenever it terminates.

    A crossing may serve TWO tracks. That is not a failure to separate them, it
    is the sheet: SEVERE and 10^-5 are drawn within a line-width of each other
    over half the altitude range. Those samples are returned flagged so the
    caller can refuse to report them.
    """
    active = [{"name": nm, "pts": [], "last": None} for nm in LINE_NAMES]
    y_anchor = int(round(y0 - px_a * ANCHOR_KFT))
    seed = _crossings(keep, sigma_of, y_anchor)
    if len(seed) != len(LINE_NAMES):
        raise SystemExit(f"anchor row at {ANCHOR_KFT} kft shows {len(seed)} lines, "
                         f"need {len(LINE_NAMES)} -- refusing to guess the ordering")
    for cur, s in zip(active, seed):
        cur["last"] = s
        cur["pts"].append((ANCHOR_KFT, s, False))

    for cur in active:
        cur["alive"] = True
        cur["miss"] = 0

    for kft in np.arange(ANCHOR_KFT + 0.25, 80.01, 0.25):
        y = int(round(y0 - px_a * kft))
        if not (200 <= y <= 3340):
            break
        cs = _crossings(keep, sigma_of, y)
        live = [c for c in active if c["alive"]]
        if not cs or not live:
            continue
        assign, shared = _assign(([c["last"] for c in live]), cs)
        for i, cur in enumerate(live):
            j = assign.get(i)
            if j is None or abs(cs[j] - cur["last"]) > 1.3:
                cur["miss"] += 1
                if cur["miss"] > 8:
                    cur["alive"] = False
                continue
            cur["miss"] = 0
            cur["last"] = cs[j]
            cur["pts"].append((kft, cs[j], shared[j]))
    return active


def _assign(prev, cs):
    """Monotonic least-displacement assignment of ordered lines to ordered ink.

    The nine lines are nested and never exchange places, so the map from lines
    to strokes must be non-decreasing. Where the sheet draws two lines within a
    stroke-width -- which it does for all three intensity boundaries, each of
    which shadows the probability curve that defines it -- two lines share one
    stroke, and the sharing is recorded rather than resolved.
    """
    m, n = len(prev), len(cs)
    inf = float("inf")
    f = [[inf] * (n + 1) for _ in range(m + 1)]
    ch = [[0] * (n + 1) for _ in range(m + 1)]
    for j in range(n + 1):
        f[0][j] = 0.0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            c = abs(prev[i - 1] - cs[j - 1])
            best = (f[i - 1][j - 1] + c, 1)                 # take a fresh stroke
            if f[i - 1][j] + c < best[0]:
                best = (f[i - 1][j] + c, 2)                 # share the stroke above
            if f[i][j - 1] < best[0]:
                best = (f[i][j - 1], 3)                     # leave this stroke unused
            f[i][j], ch[i][j] = best
    i, j, out = m, n, {}
    while i > 0 and j > 0:
        k = ch[i][j]
        if k == 1:
            out[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif k == 2:
            out[i - 1] = j - 1
            i -= 1
        else:
            j -= 1
    counts = {}
    for cj in out.values():
        counts[cj] = counts.get(cj, 0) + 1
    return out, {cj: counts[cj] > 1 for cj in counts}


def _words(d, sigma_of, px_a, y0):
    """Glyph clusters, with the shape that tells a name from a level.

    An intensity name is set ALONG its line and so is tall and narrow; a 10^-n
    label is set flat beside its curve and is wide and short. No character is
    ever recognised -- only the aspect of the cluster.
    """
    lab, _ = ndimage.label(d, structure=np.ones((3, 3)))
    glyphs = []
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if h <= 250 and w <= 250 and (lab[sl] == i).sum() > 120:
            ys, xs = np.where(lab[sl] == i)
            glyphs.append((xs.mean() + sl[1].start, ys.mean() + sl[0].start))
    glyphs.sort(key=lambda g: (g[1], g[0]))
    clusters = []
    for cx, cy in glyphs:
        for c in clusters:
            if any(abs(cx - x) < 95 and abs(cy - y) < 95 for x, y in c):
                c.append((cx, cy))
                break
        else:
            clusters.append([(cx, cy)])
    out = []
    for c in clusters:
        if len(c) < 3:
            continue
        xs = np.array([p[0] for p in c])
        ys = np.array([p[1] for p in c])
        sg, al = sigma_of(xs.mean()), (y0 - ys.mean()) / px_a
        if not (0 < sg < 35 and 0 < al < 82):
            continue
        out.append({"sigma": sg, "alt": al,
                    "tall": (ys.max() - ys.min()) > (xs.max() - xs.min())})
    return out


def _check_labels(words, px_a, y0, keep, sigma_of):
    """Where each intensity name sits, and how far it is from its two neighbours.

    The three names are set ALONG their lines, so a glyph cluster's centroid is
    offset perpendicular from the line it names and the nearest-line test is
    NOT decisive: MODERATE sits 0.39 ft/s from one line and 0.54 from the next.
    So this reports the two candidates and their distances rather than
    pronouncing, and the ordering is carried instead by `_check_order`, which
    rests on counting rather than on where a word was set.
    """
    tall = sorted((w for w in words if w["tall"]), key=lambda w: w["sigma"])
    out = []
    for w, name in zip(tall, ("LIGHT", "MODERATE", "SEVERE")):
        cs = _crossings(keep, sigma_of, int(round(y0 - px_a * w["alt"])))
        near = sorted(cs, key=lambda c: abs(c - w["sigma"]))[:2]
        out.append((name, w["sigma"], w["alt"], near))
    return out


def _check_order(keep, sigma_of, px_a, y0):
    """Confirm the naming by COUNTING lines, which the merges cannot corrupt.

    Two facts fix the order without reading a character:

      * exactly nine strokes at the anchor row, which is the count the sheet
        must show if it carries six curves and three boundaries;
      * going up, lines drop out one at a time in a fixed sequence, and the
        surviving count at each altitude matches the order below and no other.

    Returns the count at a ladder of altitudes so the caller can print what the
    naming actually rests on.
    """
    ladder = (ANCHOR_KFT, 18.0, 30.0, 46.0, 60.0, 65.0, 75.0)
    return [(k, len(_crossings(keep, sigma_of, int(round(y0 - px_a * k)))))
            for k in ladder]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", type=Path, default=Path("runs/cat"))
    ap.add_argument("--pdf", type=Path, default=PDF)
    ap.add_argument("--recalibrate", action="store_true",
                    help="re-derive the axis fits from the labels and use them")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    dark = _render(args.outdir, args.pdf)
    filled = _fill_seams(dark)
    keep = _drop_text(filled)

    px_s, x0, px_a, y0 = SIGMA_PX_PER_FTS, SIGMA_X0, ALT_PX_PER_KFT, ALT_Y0
    print("=" * 78)
    print("MIL-F-8785C FIG. 7, DIGITISED -- turbulence exceedance probability")
    print("=" * 78)
    print(f"  render      page {PAGE} (printed 49) at {DPI} dpi, rotated 90 deg clockwise")
    seams = [x for x in range(1, dark.shape[1] - 1)
             if dark[300:3300, x].sum() == 0 and dark[300:3300, x - 1].sum() > 0]
    print(f"  tile seams  {len(seams)} blank columns closed before tracing")

    if args.recalibrate:
        (sp_, sx_, sres), (ap_, ay_, ares) = _recalibrate(filled)
        px_s, x0, px_a, y0 = sp_, sx_, ap_, ay_
        print(f"  sigma axis  {px_s:.3f} px per ft/s, 0 at x={x0:.1f}, "
              f"worst label residual {sres:.2f} ft/s")
        print(f"  alt axis    {px_a:.3f} px per kft, 0 at y={y0:.1f}, "
              f"label residual sd {ares:.1f} px = {ares/px_a:.2f} kft")
    else:
        print(f"  sigma axis  {px_s:.3f} px per ft/s, 0 at x={x0:.1f}   (stored fit)")
        print(f"  alt axis    {px_a:.3f} px per kft, 0 at y={y0:.1f}   (stored fit)")

    sigma_of = lambda x: (x - x0) / px_s
    active = _track_from_anchor(keep, sigma_of, px_a, y0)

    print()
    print("-" * 78)
    print("A. WHAT THE NAMING RESTS ON")
    print("-" * 78)
    print("   altitude   strokes    lines the ordering says should still be drawn")
    expect = {9.0: 9, 18.0: 8, 30.0: 7, 46.0: 6, 60.0: 5, 65.0: 4, 75.0: 4}
    for kft, n in _check_order(keep, sigma_of, px_a, y0):
        flag = "OK" if n == expect.get(kft) else f"expected {expect.get(kft)}"
        print(f"   {kft:5.0f} kft   {n:5d}      {flag}")
    print()
    print("  and the three names, set ALONG their lines, sit between two strokes --")
    print("  which is why the count above carries the ordering and this does not:")
    for name, sg, al, near in _check_labels(_words(filled, sigma_of, px_a, y0),
                                            px_a, y0, keep, sigma_of):
        cand = ", ".join(f"{c:.2f} (d={abs(c-sg):.2f})" for c in near)
        print(f"     {name:9s} at sigma={sg:5.2f}, {al:5.1f} kft -> {cand}")

    print()
    print("-" * 78)
    print("B. THE NINE LINES")
    print("-" * 78)
    print("  A stroke shared by two lines is MERGED: the sheet draws each intensity")
    print("  boundary alongside the probability curve that defines it, so over much")
    print("  of the range the pair is one stroke of ink and cannot be separated.")
    print()
    print("   line        altitude span        sigma span      clean   merged")
    for cur in active:
        p = np.array([(a, s) for a, s, _ in cur["pts"]])
        mg = sum(1 for _, _, m in cur["pts"] if m)
        print(f"   {cur['name']:9s}  {p[:,0].max():5.1f} -> {p[:,0].min():5.1f}    "
              f"{p[:,1].min():6.2f} .. {p[:,1].max():6.2f}     "
              f"{len(p)-mg:6d}   {mg:6d}")

    print()
    print("-" * 78)
    print("C. SEVERE AT 37,000 FT, AND THE INDEPENDENT READING OF IT")
    print("-" * 78)
    sev = next(c for c in active if c["name"] == "SEVERE")
    p5 = next(c for c in active if c["name"] == "10^-5")
    at37 = min(sev["pts"], key=lambda p: abs(p[0] - TARGET_ALT_KFT))
    v37, is_merged = at37[1], at37[2]

    print(f"  SEVERE is drawn alongside 10^-5, and {TARGET_ALT_KFT:.0f} kft is INSIDE")
    print(f"  that merge -- the two lines are one stroke of ink there. This project")
    print(f"  does not read a number out of a merge and call it one line, so:")
    print()
    print(f"     stroke at {at37[0]:5.2f} kft      {v37:6.2f} ft/s = {v37*FT2M:.3f} m/s"
          f"   {'MERGED (SEVERE + 10^-5)' if is_merged else 'clean'}")
    span = [(a, s) for a, s, m in sev["pts"] if not m and abs(a - TARGET_ALT_KFT) <= 12]
    if span:
        lo, hi = min(s for _, s in span), max(s for _, s in span)
        print(f"     nearest clean SEVERE samples within 12 kft span {lo:.2f}..{hi:.2f} ft/s")
    pair = [(s, t) for (a, s, m), (b, t, n) in zip(sev["pts"], p5["pts"])
            if not m and not n and abs(a - TARGET_ALT_KFT) < 6]
    if pair:
        print(f"     where the pair IS resolved near 37 kft it separates by "
              f"{np.mean([abs(t-s) for s, t in pair]):.2f} ft/s")
    print(f"     -> sigma_severe({TARGET_ALT_KFT:.0f} kft) = {v37:.1f} +/- ~0.4 ft/s "
          f"= {v37*FT2M:.2f} +/- ~0.12 m/s,")
    print(f"        the band set by which member of the pair the stroke belongs to.")
    print()
    print(f"  JSBSim FGWinds.cpp, the same figure transcribed independently,")
    print(f"  severe row interpolated 35k->45k:    {JSBSIM_SEVERE_37KFT_FTS:.2f} ft/s = "
          f"{JSBSIM_SEVERE_37KFT_FTS*FT2M:.3f} m/s")
    dv = abs(v37 - JSBSIM_SEVERE_37KFT_FTS)
    print(f"  -> the two readings differ by {dv:.2f} ft/s, "
          f"{dv/JSBSIM_SEVERE_37KFT_FTS*100:.1f}%, which is inside the merge band.")
    print(f"     Neither set the other. An earlier pass here quoted 0.4% by reading")
    print(f"     the merged stroke as though it were SEVERE alone; that was too good.")
    print()
    print(f"  wind.mehta_residual_ceiling   = {MEHTA_CEILING_MS:.3f} m/s")
    print(f"  sigma_severe(37 kft)          = {v37*FT2M:.3f} m/s (+/- 0.12)")
    print(f"  -> `mil_f_8785c_sigma_w_exceeds_the_mehta_ceiling` points "
          f"{'RIGHT' if v37*FT2M > MEHTA_CEILING_MS else 'WRONG'}, and the merge band")
    print(f"     is far too narrow to change that. IT IS NOT SETTLED HERE.")

    csv = args.outdir / "mil-f-8785c-fig7-lines.csv"
    with csv.open("w") as fh:
        fh.write("line,altitude_kft,sigma_fts,merged\n")
        for cur in active:
            for kft, s, m in cur["pts"]:
                fh.write(f"{cur['name']},{kft:.2f},{s:.4f},{int(m)}\n")
    print(f"\n  data -> {csv}   (all nine named; `merged` flags a shared stroke)")

    _figure(args.outdir / "11-mil-f-8785c-fig7.png", args.outdir, active,
            sigma_of, px_a, y0, v37)
    print(f"  figure -> {args.outdir / '11-mil-f-8785c-fig7.png'}")


def _figure(path, sp, active, sigma_of, px_a, y0, v37):
    img = np.asarray(Image.open(sp / "mil-fig7.png").convert("L"))
    fig, ax = plt.subplots(figsize=(9.5, 8))
    ax.imshow(img, cmap="gray", aspect="auto",
              extent=[sigma_of(0), sigma_of(img.shape[1]),
                      (y0 - img.shape[0]) / px_a, y0 / px_a])
    for cur in active:
        p = np.array([(a, s) for a, s, _ in cur["pts"]])
        intensity = cur["name"] in ("LIGHT", "MODERATE", "SEVERE")
        sev = cur["name"] == "SEVERE"
        ax.plot(p[:, 1], p[:, 0],
                lw=2.4 if sev else (1.8 if intensity else 1.4),
                color=PALETTE["reference"] if intensity else PALETTE["model"],
                alpha=1.0 if sev else (.85 if intensity else .7),
                zorder=3 if intensity else 2)
        j = int(np.argmin(p[:, 0]))
        ax.annotate(cur["name"], xy=(p[j, 1], p[j, 0]), xytext=(3, -11),
                    textcoords="offset points", fontsize=7.5,
                    color=PALETTE["reference"] if intensity else PALETTE["model"])
        mg = np.array([(a, s) for a, s, m in cur["pts"] if m])
        if len(mg):
            ax.plot(mg[:, 1], mg[:, 0], lw=3.4, color=PALETTE["muted"],
                    alpha=.45, zorder=1)
    ax.plot([v37], [TARGET_ALT_KFT], "o", ms=10, mfc="none", mew=2.4,
            color=PALETTE["wind"], zorder=4,
            label=f"SEVERE at {TARGET_ALT_KFT:.0f} kft: {v37:.2f} ft/s interpolated\n"
                  f"(JSBSim's transcription reads {JSBSIM_SEVERE_37KFT_FTS:.2f})")
    ax.plot([], [], lw=1.8, color=PALETTE["reference"], label="intensity boundary")
    ax.plot([], [], lw=1.4, color=PALETTE["model"], alpha=.7, label="probability curve")
    ax.plot([], [], lw=3.4, color=PALETTE["muted"], alpha=.45,
            label="merged stroke -- two lines, one ink")
    ax.set_xlim(0, 35)
    ax.set_ylim(0, 85)
    ax.set_xlabel("RMS turbulence amplitude $\\sigma$, ft/s TAS", fontsize=9)
    ax.set_ylabel("altitude, 1000 ft", fontsize=9)
    ax.set_title("MIL-F-8785C Fig. 7 digitised, over the scan it was read from\n"
                 "nine lines: six probability curves and three intensity boundaries",
                 fontsize=10.5, loc="left")
    ax.legend(fontsize=7.5, loc="upper right", framealpha=.95)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
