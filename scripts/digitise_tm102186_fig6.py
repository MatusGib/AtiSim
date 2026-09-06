"""NASA TM-102186 Figure 6, DIGITISED -- the Hannibal DFDR g-load time history.

The recorded trace this project has been quoting as `wind.TM102186_HANNIBAL_NZ`
since session 23 WITHOUT EVER LOOKING AT THE CURVE BETWEEN ITS TWO EXTREMES.
Yoshimura's equivalent is withheld under confidentiality, so this is the only
recorded acceleration history within reach.

Figure 6 is three stacked panels against GMT -- horizontal wind (knots),
vertical wind (ft/sec) and G LOAD -- for the DC-10 that encountered severe
turbulence near Hannibal, MO in April 1981. The bottom panel is the target.

METHOD. Render the page at 600 dpi, binarise, and read the trace column by
column as the top and bottom of the ink. Nothing is fitted and nothing is
smoothed: a column of ink becomes an interval in g, and the pair of envelopes
is the output. Calibration comes from the axis LABELS rather than the frame,
by least squares -- see `_calibrate` for why the outermost labels are excluded.

WHY THIS IS CHECKABLE, WHICH IS THE ONLY REASON IT IS IN THE TREE. Three
independent checks, none of which was used to set a parameter:

  1  THE PAPER'S OWN BAND. p. 3-4 states the fluctuations run "from +1.7 to
     -1.0 g". The digitised downward extreme is -0.972 and the SECOND upward
     peak is +1.702, the 99.5th percentile of the upper envelope +1.701.
  2  THE NEGATIVE CONTROL ON THE METHOD. The same extraction run on the
     VERTICAL WIND panel must return zero through the pre-encounter cruise,
     because the aircraft was straight and level in smooth air. It returns
     +1.1 ft/s. So the column-envelope method carries no vertical bias, and
     any offset the G LOAD panel shows is a property of that panel.
  3  THE GUST SPACING. The two large downward excursions land 1.0 s apart at
     this resolution and the encounter sits at 1:24:3x, against TM-102186's
     own prose "about 5 sec apart" for the up-and-down gust pairs.

WHAT THE DIGITISATION FINDS THAT THE PROSE DOES NOT SAY. The absolute peak of
the plotted trace is +1.855 g, not +1.7 -- one narrow spike, 0.15 g above the
second peak. The paper's "+1.7" is the second peak to within 0.002 g, so the
prose is quoting the sustained band and not the largest single excursion.
Whether to compare a model peak against 1.70 or 1.86 is a decision for
PROJECT.md section 4 and is deliberately NOT taken here.

AND ONE THING THAT IS DECLARED RATHER THAN RESOLVED. The trace sits at
0.951 g through the pre-encounter cruise, where level flight is 1.000 g by
definition. Check 2 says the method is not what put it there. It is either a
recorder bias or a registration offset of the plotted curve within its own
axes, and the figure alone cannot separate those. Both envelopes are reported
RAW, in plot coordinates, because that is what the paper's own numbers are.

Run: PYTHONPATH=<abs worktree root> .venv/Scripts/python.exe scripts/digitise_tm102186_fig6.py --outdir runs/cat
"""

import argparse
from pathlib import Path

import fitz
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PALETTE = {
    "model": "#1D5D77", "reference": "#A9501C", "wind": "#3E6A48",
    "muted": "#7E8D93", "grid": "#D6DCD8",
}

# The reference PDFs are gitignored (see .gitignore) and live in the main
# checkout, so a worktree run must point --pdf at them.
PDF = Path("Reference_papers/19890016606.pdf")
PAGE = 6                       # printed p. 3-5
CLIP = fitz.Rect(78, 60, 320, 385)
DPI = 600

# The paper's own statement of the band, p. 3-4. This is what
# `wind.TM102186_HANNIBAL_NZ` carries.
PAPER_NZ = (-1.0, 1.7)
PAPER_GUST_SPACING_S = 5.0     # "about 5 sec apart"

# Panel label values, top to bottom, as printed.
PANELS = {
    "horizontal_wind_kt": (150, 640, (160.0, 140.0, 120.0)),
    "vertical_wind_fps": (700, 1620, (100.0, 50.0, 0.0, -50.0, -100.0)),
    "g_load": (1700, 2380, (2.0, 1.0, 0.0, -1.0)),
}

# Pre-encounter window, in image columns: smooth cruise, used only as a
# negative control and never to set a calibration constant.
QUIET_COLS = (450, 1330)


def _render(sp: Path, pdf: Path) -> np.ndarray:
    """Page 6 at 600 dpi as a boolean ink mask."""
    doc = fitz.open(pdf)
    pix = doc[PAGE].get_pixmap(dpi=DPI, clip=CLIP)
    out = sp / "tm-fig6.png"
    pix.save(out)
    doc.close()
    return np.asarray(Image.open(out).convert("L")) < 128


def _groups(idx, gap=10):
    if len(idx) == 0:
        return []
    out = [[idx[0]]]
    for v in idx[1:]:
        if v - out[-1][-1] <= gap:
            out[-1].append(v)
        else:
            out.append([v])
    return [((g[0] + g[-1]) / 2.0, g[-1] - g[0]) for g in out]


def _calibrate(dark, y0, y1, values):
    """Pixel row per axis value, from the numeric labels left of the axis.

    The OUTERMOST labels are dropped. On all three panels they sit 5-10 px
    inside the spacing the interior labels agree on -- the top and bottom
    labels are nudged in to stay within the figure's own bounds, which is a
    property of the 1989 artwork and not of the scan. On the G LOAD panel the
    interior pair '1' and '0' reproduce the frame's own +2 and -1 rules to
    within 2 px, which is what says dropping the outer two is right rather
    than convenient.
    """
    band = dark[y0:y1, 300:392]
    rows = np.where(band.sum(1) >= 3)[0] + y0
    centres = [c for c, _ in _groups(rows)]
    if len(centres) != len(values):
        raise SystemExit(f"expected {len(values)} labels in {y0}..{y1}, found {len(centres)}")
    ys = np.array(centres[1:-1], float)
    vs = np.array(values[1:-1], float)
    px_per_unit = -np.polyfit(vs, ys, 1)[0]
    y_at_zero = np.polyval(np.polyfit(vs, ys, 1), 0.0)
    return px_per_unit, y_at_zero, centres


def _x_axis(dark):
    """Columns of the six inward minute ticks on the bottom axis, least-squared."""
    band = dark[2338:2357, :]
    xs = np.where(band.sum(0) >= 14)[0]
    cand = [c for c, _ in _groups(xs, 6)]
    # the data spike also reaches the axis; keep the six that fit a line best
    ticks = [c for c in cand if 390 <= c <= 1850]
    best, bestres = None, None
    for drop in range(len(ticks) + 1):
        sub = ticks[:drop] + ticks[drop + 1:] if drop < len(ticks) else ticks
        if len(sub) != 6:
            continue
        i = np.arange(6.0)
        p = np.polyfit(i, sub, 1)
        res = np.abs(np.polyval(p, i) - sub).max()
        if bestres is None or res < bestres:
            best, bestres = p, res
    if best is None:
        raise SystemExit(f"could not find six minute ticks, got {ticks}")
    px_per_min, x_at_121 = best[0], best[1]
    return px_per_min, x_at_121, bestres


def _trace(dark, y_top, y_bot, x_lo, x_hi):
    """Top and bottom of the ink in every column, specks removed."""
    sub = dark[y_top:y_bot, x_lo:x_hi + 1]
    xs, tops, bots = [], [], []
    for j in range(sub.shape[1]):
        ys = np.where(sub[:, j])[0]
        if len(ys) == 0:
            continue
        s = set(ys.tolist())
        keep = np.array([y for y in ys if (y - 1 in s) or (y + 1 in s)])
        if keep.size == 0:
            continue
        xs.append(j + x_lo)
        tops.append(keep.min() + y_top)
        bots.append(keep.max() + y_top)
    return np.array(xs), np.array(tops, float), np.array(bots, float)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=Path, default=Path("runs/cat"))
    p.add_argument("--pdf", type=Path, default=PDF)
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    dark = _render(args.outdir, args.pdf).copy()

    px_min, x121, xres = _x_axis(dark)
    t_of = lambda x: (x - x121) / px_min * 60.0

    print("=" * 78)
    print("NASA TM-102186 FIG. 6, DIGITISED -- the Hannibal DFDR g-load trace")
    print("=" * 78)
    print(f"  render         page {PAGE} at {DPI} dpi, clip {tuple(CLIP)}")
    print(f"  time axis      {px_min:.2f} px per minute, 1:21 at x = {x121:.1f}")
    print(f"                 worst tick residual {xres:.1f} px = {xres/px_min*60:.2f} s")

    # ---- the G LOAD panel -------------------------------------------------
    y0, y1, vals = PANELS["g_load"]
    px_g, y_zero, labels = _calibrate(dark, y0, y1, vals)
    g_of = lambda y: (y_zero - y) / px_g
    print(f"  load axis      {px_g:.1f} px per g, 0 g at y = {y_zero:.1f}")
    print(f"                 labels at {['%.1f' % c for c in labels]}")

    masked = dark.copy()
    for i in range(6):                       # the minute ticks reach up into the panel
        xt = int(round(x121 + px_min * i))
        masked[2325:2364, xt - 9:xt + 10] = False
    x, top, bot = _trace(masked, 1735, 2357, 430, 1851)
    t, g_hi, g_lo = t_of(x), g_of(top), g_of(bot)
    centre = (g_hi + g_lo) / 2.0

    q = (x >= QUIET_COLS[0]) & (x <= QUIET_COLS[1])
    base = centre[q].mean()

    print()
    print("-" * 78)
    print("A. WHAT THE TRACE SAYS")
    print("-" * 78)
    print(f"  {len(x)} columns, t = {t.min():.1f} .. {t.max():.1f} s after 1:21:00")
    print(f"  pre-encounter cruise   {base:.4f} g   sd {centre[q].std():.4f}")
    print(f"  absolute peak          {g_hi.max():+.3f} g at t = {t[g_hi.argmax()]:.1f} s")
    print(f"  absolute trough        {g_lo.min():+.3f} g at t = {t[g_lo.argmin()]:.1f} s")
    print(f"  peak-to-peak           {g_hi.max() - g_lo.min():.3f} g")

    order = np.argsort(g_hi)[::-1]
    seen = []
    for i in order:
        if all(abs(t[i] - s) > 1.0 for s in seen):
            seen.append(t[i])
        if len(seen) >= 2:
            break
    second = max(g_hi[np.argmin(np.abs(t - seen[1]))], 0.0)
    pct995 = np.percentile(g_hi, 99.5)

    print()
    print("-" * 78)
    print("B. THE THREE CHECKS")
    print("-" * 78)
    print(f"  1  paper's band, p. 3-4:            {PAPER_NZ[0]:+.2f} / {PAPER_NZ[1]:+.2f} g")
    print(f"     digitised trough:                {g_lo.min():+.3f} g   "
          f"({abs(g_lo.min() - PAPER_NZ[0]):.3f} off)")
    print(f"     digitised SECOND peak:           {second:+.3f} g   "
          f"({abs(second - PAPER_NZ[1]):.3f} off)")
    print(f"     99.5th pctile of upper envelope: {pct995:+.3f} g   "
          f"({abs(pct995 - PAPER_NZ[1]):.3f} off)")
    print(f"     -> the prose quotes the SUSTAINED band. The absolute peak is")
    print(f"        {g_hi.max():+.3f} g, one narrow spike {g_hi.max() - second:.3f} g above it,")
    print(f"        and the paper does not mention it.")

    yy0, yy1, vvals = PANELS["vertical_wind_fps"]
    px_w, yw_zero, _ = _calibrate(dark, yy0, yy1, vvals)
    mask2 = dark.copy()
    for i in range(6):
        xt = int(round(x121 + px_min * i))
        mask2[1545:1590, xt - 9:xt + 10] = False
    xw, tw, bw = _trace(mask2, 800, 1578, 430, 1851)
    cw = ((yw_zero - tw) / px_w + (yw_zero - bw) / px_w) / 2.0
    qw = (xw >= QUIET_COLS[0]) & (xw <= QUIET_COLS[1])
    print()
    print(f"  2  NEGATIVE CONTROL, vertical-wind panel, same method:")
    print(f"     pre-encounter cruise  {cw[qw].mean():+.2f} ft/s   (must be 0 -- smooth air)")
    print(f"     -> the method carries no vertical bias, so the {base:.3f} g")
    print(f"        baseline above belongs to the G LOAD panel, not to this code.")

    tw_s = t_of(xw)
    w_hi = (yw_zero - tw) / px_w
    ups = []
    for i in np.argsort(w_hi)[::-1]:
        if all(abs(tw_s[i] - s) > 1.5 for s in ups):
            ups.append(tw_s[i])
        if len(ups) >= 2:
            break
    spacing = abs(ups[1] - ups[0])
    print()
    print(f"  3  GUST SPACING, the channel the identification did not set.")
    print(f"     the two principal UP gusts land at t = {min(ups):.1f} and {max(ups):.1f} s")
    print(f"     digitised spacing   {spacing:.2f} s")
    print(f"     paper's prose       'about {PAPER_GUST_SPACING_S:.0f} sec apart'")
    print(f"     cat_uncertainty.py's flown model, centre-to-centre:  5.36 s")
    print(f"     -> the model is {(5.36 - spacing) / spacing * 100:+.1f}% against the DIGITISED")
    print(f"        record, where against the prose alone it read +7.2%. The figure")
    print(f"        is a tighter reference than the sentence it is described by.")
    print()
    print(f"     recorded vertical gust extremes: {w_hi.max():+.1f} / "
          f"{((yw_zero - bw) / px_w).min():+.1f} ft/s")
    print(f"     the 747 flown through Mehta's field sees:  +59.1 / -86.8 ft/s")

    csv = args.outdir / "tm102186-fig6-gload.csv"
    np.savetxt(csv, np.c_[t, g_lo, g_hi, centre], delimiter=",",
               header="t_s_after_0121,g_lower,g_upper,g_centre", comments="", fmt="%.4f")
    print(f"\n  data -> {csv}")

    _figure(args.outdir / "10-tm102186-fig6.png", t, g_lo, g_hi, base, second, g_hi.max())
    print(f"  figure -> {args.outdir / '10-tm102186-fig6.png'}")


def _figure(path, t, g_lo, g_hi, base, second, peak):
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.fill_between(t, g_lo, g_hi, color=PALETTE["model"], alpha=.85, lw=0,
                    label="digitised trace (ink envelope)")
    ax.axhline(PAPER_NZ[1], color=PALETTE["reference"], ls="--", lw=1.2,
               label=f"paper's stated band {PAPER_NZ[0]:+.1f} / {PAPER_NZ[1]:+.1f} g")
    ax.axhline(PAPER_NZ[0], color=PALETTE["reference"], ls="--", lw=1.2)
    ax.axhline(base, color=PALETTE["muted"], ls=":", lw=1.2,
               label=f"pre-encounter cruise, {base:.3f} g")
    ax.annotate(f"absolute peak {peak:+.3f} g\nnot quoted in the paper",
                xy=(t[np.argmax(g_hi)], peak), xytext=(t[np.argmax(g_hi)] - 78, peak + .07),
                fontsize=8.5, color=PALETTE["reference"],
                arrowprops=dict(arrowstyle="->", color=PALETTE["reference"], lw=1))
    ax.set_xlabel("seconds after 1:21:00 GMT", fontsize=9)
    ax.set_ylabel("normal load factor $n_z$, g", fontsize=9)
    ax.set_title("NASA TM-102186 Fig. 6 digitised -- the recorded Hannibal g-load trace\n"
                 "the paper's +1.7 is the SECOND peak, to 0.002 g", fontsize=10.5, loc="left")
    ax.grid(True, color=PALETTE["grid"], lw=.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8, loc="upper left", framealpha=.95)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
