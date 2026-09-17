"""The Hannibal HORIZONTAL wind: digitised, and marked against the field AtiSim flies.

SOURCE. NASA TM-102186, Wingrove, Bach & Schultz, "Analysis of Severe Atmospheric
Disturbances From Airline Flight Records", June 1989 -- `Reference_papers/
19890016606.pdf`, outside git. Figure 7, printed p. 3-5 (PDF page index 6),
"Vortex-array model for a severe turbulence encounter over Hannibal, MO, April
1981": a HORIZONTAL WIND panel (knots) and a VERTICAL WIND panel (ft/s), each with
ACTUAL (dotted, from the DC-10's flight recorder) and MODEL (solid), over
DISTANCE -25,000 to +25,000 ft. It is the same five-vortex fit Mehta 1987 draws
as his Fig. 9, redrawn cleanly, and its parameters are the ones in
`wind.MEHTA_HANNIBAL_*`.

WHY THE HORIZONTAL PANEL. A core the same distance above or below the flight
path gives the SAME vertical wind; only the horizontal perturbation changes sign.
So nothing the project had compared -- vertical-wind extremes, gust spacing,
load -- could tell which side of the path a core sits on. `wind.py`'s sign of z
was argued from Mehta's prose. The along-track gust changes dynamic pressure on
any 747, and since session 30 declared CR-2144's speed derivatives `Cm_M` also
turns it into pitch, so the side matters.

METHOD. Render at the scan's native 300 dpi, binarise, and separate the two
curves by SHAPE: the solid MODEL line is a few long ink components, the dotted
ACTUAL curve is many small dots. Calibration is least squares: y through each
panel's printed tick LABELS, x through the TICK MARKS each DISTANCE label names
(a label's glyph centre is not its tick), never through the frame. Nothing is fitted to the
curves and nothing is smoothed. `--overlay` writes the classification over the
scan so it can be looked at rather than trusted.

THREE CHECKS, in the order they must pass:

  1  CALIBRATION CONTROL. The digitised VERTICAL model must reproduce AtiSim's
     own vertical wind along the path. That component is already pinned by
     other tests, so a failure here is the digitisation, not the field.
  2  THE SIGN TEST. The digitised HORIZONTAL model against AtiSim's horizontal
     perturbation plus Mehta's 149.8 kt bias -- as transcribed, and with every
     core's z flipped. One must fit and the other must not.
  3  THE FIELD AGAINST THE RECORD. ACTUAL minus MODEL, horizontal: how much
     along-track gust the fitted field does not carry.

Run: PYTHONPATH=<abs worktree root> .venv/Scripts/python.exe \\
         scripts/digitise_hannibal_horizontal_wind.py --pdf <Reference_papers/19890016606.pdf> \\
         --outdir runs/cat [--overlay] [--write]
"""

import argparse
import csv
from pathlib import Path

import atisim

print(f"atisim imported from: {atisim.__file__}")

import fitz  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

from atisim import wind  # noqa: E402
from atisim.units import FT2M  # noqa: E402

KT2MS = 0.514444
BIAS_KT = 149.8  # Mehta 1987 p. 30, b_xy of the five-vortex solution
PAGE = 6
CLIP = fitz.Rect(300, 100, 590, 380)  # Fig. 7, in page points
DPI = 300
DATA = Path(atisim.__file__).parent / "data" / "tm102186_fig7_winds.csv"

# Panel geometry at 300 dpi inside CLIP, found by scripts' own exploration
# (long vertical ink runs): the left axis of each wind panel and its extent.
# The CALIBRATION does not use these -- it uses the tick labels -- they only
# bound where to look.
PANELS = {
    # The ACTUAL/MODEL legend sits inside the horizontal panel's top-left
    # corner; its sample line and the word MODEL were read as model curve until
    # this box excluded them (seen on the --overlay, not inferred).
    "horizontal": dict(y0=120, y1=400, labels=(175.0, 150.0, 125.0, 100.0),
                       exclude=((230, 500, 40, 152),)),
    "vertical": dict(y0=445, y1=725, labels=(100.0, 0.0, -100.0), exclude=()),
}
# Tick marks stand ~10 px proud of each axis and were read as data dots.
DOT_MIN_AXIS_GAP = 18
# A solid-line piece is long; a chain of touching data dots is compact. Chains
# were read as model line where the dotted curve crosses the solid one.
LINE_MIN_PIXELS, LINE_MIN_EXTENT = 250, 40
X_LABEL_BAND = (1022, 1062)  # the DISTANCE tick labels under the altitude panel
X_LABELS = tuple(float(v) for v in range(-25, 30, 5))  # thousands of ft


def render(pdf: Path) -> np.ndarray:
    doc = fitz.open(pdf)
    pix = doc[PAGE].get_pixmap(dpi=DPI, clip=CLIP, colorspace=fitz.csGRAY)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)


def components(mask):
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    objs = ndimage.find_objects(lab)
    sizes = ndimage.sum(mask, lab, index=np.arange(1, n + 1))
    return lab, objs, sizes


def axis_column(ink, y0, y1):
    """The panel's left axis: the column with the longest vertical ink run."""
    best, bx = 0, None
    for x in range(ink.shape[1] // 2):
        col = ink[y0:y1, x].astype(int)
        d = np.diff(np.concatenate([[0], col, [0]]))
        s, e = np.where(d == 1)[0], np.where(d == -1)[0]
        if s.size and (e - s).max() > best:
            best, bx = (e - s).max(), x
    return bx, best


def label_centres(ink, lab, objs, band_y, band_x, axis, n_expected):
    """Centres of the tick-label glyph clusters, clustered along `axis`."""
    boxes = []
    for i, sl in enumerate(objs):
        (ya, yb), (xa, xb) = (sl[0].start, sl[0].stop), (sl[1].start, sl[1].stop)
        if band_y[0] <= ya and yb <= band_y[1] and band_x[0] <= xa and xb <= band_x[1]:
            if 20 <= (lab[sl] == i + 1).sum() <= 800:
                boxes.append((ya, yb, xa, xb))
    # Clustered on the blank space BETWEEN glyph edges, not on glyph centres:
    # centres chain, so the "5" of -25 sits close enough to the minus of -20 to
    # fuse two labels. Inside one label, glyphs (a minus sign included) are a
    # few px apart; neighbouring labels have 30 px or more of white between them.
    lo, hi = (0, 1) if axis == 0 else (2, 3)
    boxes.sort(key=lambda b: b[lo])
    clusters, gap, edge = [], 15, None
    for b in boxes:
        if clusters and b[lo] - edge < gap:
            clusters[-1].append(b)
            edge = max(edge, b[hi])
        else:
            clusters.append([b])
            edge = b[hi]
    centres = []
    for c in clusters:
        if axis == 0:
            centres.append((min(b[0] for b in c) + max(b[1] for b in c)) / 2)
        else:
            centres.append((min(b[2] for b in c) + max(b[3] for b in c)) / 2)
    if len(centres) != n_expected:
        raise RuntimeError(f"found {len(centres)} label clusters, expected {n_expected}: {centres}")
    return np.array(centres)


def fit(px, values):
    A = np.vstack([np.ones_like(px), px]).T
    coef, *_ = np.linalg.lstsq(A, values, rcond=None)
    resid = values - A @ coef
    return coef, float(np.abs(resid).max()), resid


def extract(img):
    ink = img < 128
    lab, objs, sizes = components(ink)
    out = {}
    # x calibration. The DISTANCE labels place each tick to about +-0.5 kft -- a
    # label's glyph centre is not its tick, and a minus sign widens it -- and on
    # the horizontal wind's dip half a kft is worth ~10 kt. So the fit is made
    # through the TICK MARKS standing on the altitude panel's baseline, and the
    # labels are kept only to name them and to cross-check.
    xc = label_centres(ink, lab, objs, X_LABEL_BAND, (150, ink.shape[1]), axis=1,
                       n_expected=len(X_LABELS))
    _, _, xr_lab = fit(xc, np.array(X_LABELS))
    baseline = X_LABEL_BAND[0] - 5          # the axis line sits just above the labels
    rows = ink[baseline - 16:baseline - 2, :]
    tick_cols = np.nonzero(rows.sum(axis=0) >= 8)[0]
    ticks, run = [], [tick_cols[0]]
    for c in tick_cols[1:]:
        if c - run[-1] <= 1:
            run.append(c)
        else:
            ticks.append(np.mean(run))
            run = [c]
    ticks.append(np.mean(run))
    # keep, for each label, the tick nearest its centre
    ticks = np.array(ticks)
    xt = np.array([ticks[np.argmin(np.abs(ticks - c))] for c in xc])
    xcoef, xres, xr = fit(xt, np.array(X_LABELS))
    out["x_cal"] = (xcoef, xres)
    out["x_labels"] = list(zip(X_LABELS, xt, xr))
    out["x_label_resid"] = xr_lab
    for name, p in PANELS.items():
        ax, run = axis_column(ink, p["y0"], p["y1"])
        yc = label_centres(ink, lab, objs, (p["y0"] - 15, p["y1"] + 15), (ax - 60, ax - 4),
                           axis=0, n_expected=len(p["labels"]))
        ycoef, yres, yr = fit(yc, np.array(p["labels"]))
        # plot area: right of the axis and its ticks, inside the panel
        region = np.zeros_like(ink)
        region[p["y0"]:p["y1"], ax + 10:] = True
        for x0, x1, y0, y1 in p["exclude"]:
            region[y0:y1, x0:x1] = False
        pink = ink & region
        plab, pobjs, psizes = components(pink)
        line = np.zeros_like(ink)
        dots = []
        for i, sl in enumerate(pobjs):
            h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
            if psizes[i] >= LINE_MIN_PIXELS or max(h, w) > LINE_MIN_EXTENT:
                line |= plab == i + 1
            elif psizes[i] >= 6:
                ys, xs = np.nonzero(plab[sl] == i + 1)
                cx = xs.mean() + sl[1].start
                if cx >= ax + DOT_MIN_AXIS_GAP:
                    dots.append((cx, ys.mean() + sl[0].start))
        cols = []
        for x in range(ax + 10, ink.shape[1]):
            rows = np.nonzero(line[:, x])[0]
            if rows.size:
                cols.append((x, rows.min(), rows.max(), rows.mean()))
        out[name] = dict(axis=(ax, run), y_cal=(ycoef, yres), cols=cols, dots=dots,
                         line_mask=line, y_labels=list(zip(p["labels"], yc, yr)))
    return out


def to_data(xcoef, ycoef, x_px, y_px):
    return xcoef[0] + xcoef[1] * np.asarray(x_px), ycoef[0] + ycoef[1] * np.asarray(y_px)


def atisim_winds(x_kft, flip_z=False):
    """AtiSim's field along Mehta's straight path at 37,000 ft: horizontal
    perturbation + Mehta's bias in knots, vertical wind (up) in ft/s."""
    alt = wind.MEHTA_HANNIBAL_ALTITUDE
    array = wind.mehta_hannibal_array(alt)
    if flip_z:  # core at alt - z  ->  alt + z
        array = array._replace(down=-2.0 * alt - array.down)
    pos = jnp.stack([jnp.asarray(x_kft) * 1000.0 * FT2M, jnp.zeros(len(x_kft)),
                     jnp.full(len(x_kft), -alt)], axis=1)
    w = np.asarray(jax.vmap(lambda p: wind.vortex_wind(p, array))(pos))
    return w[:, 0] / KT2MS + BIAS_KT, -w[:, 2] / FT2M


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, default=Path("runs/cat"))
    ap.add_argument("--overlay", action="store_true")
    ap.add_argument("--write", action="store_true", help="write the tracked CSV of digitised points")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    img = render(args.pdf)
    ex = extract(img)
    xcoef, xres = ex["x_cal"]
    print(f"\nx calibration through TICK MARKS: {1 / xcoef[1]:.3f} px per 1000 ft, "
          f"worst tick residual {xres:.3f} kft (label centres alone: "
          f"{np.abs(ex['x_label_resid']).max():.3f} kft)")
    print("   " + "  ".join(f"{v:+.0f}@{c:.1f}px({r:+.2f})" for v, c, r in ex["x_labels"]))

    rows_out = []
    for name in ("vertical", "horizontal"):
        p = ex[name]
        ycoef, yres = p["y_cal"]
        unit = "kt" if name == "horizontal" else "ft/s"
        print(f"\n== {name.upper()} panel: axis x={p['axis'][0]} (run {p['axis'][1]} px); "
              f"{abs(1 / ycoef[1]):.3f} px per {unit}, worst label residual {yres:.3f} {unit}")
        print("   labels: " + "  ".join(f"{v:+.0f}@{c:.1f}px({r:+.2f})" for v, c, r in p["y_labels"]))
        cols = np.array(p["cols"], dtype=float)
        steep = (cols[:, 2] - cols[:, 1]) > 14
        xk, ym = to_data(xcoef, ycoef, cols[:, 0], cols[:, 3])
        flat = ~steep
        dots = np.array(p["dots"], dtype=float) if p["dots"] else np.zeros((0, 2))
        dx, dy = to_data(xcoef, ycoef, dots[:, 0], dots[:, 1]) if dots.size else (np.array([]), np.array([]))
        print(f"   MODEL: {len(cols)} columns ({steep.sum()} steep), x {xk.min():.2f}..{xk.max():.2f} kft;"
              f"   ACTUAL: {len(dots)} dots")
        idx = 0 if name == "horizontal" else 1
        for flip in (False, True):
            ours = atisim_winds(xk, flip)[idx]
            r = ym[flat] - ours[flat]
            print(f"   digitised MODEL - AtiSim ({'z FLIPPED' if flip else 'z as transcribed'}): "
                  f"RMS {np.sqrt(np.mean(r**2)):.2f} {unit}, mean {r.mean():+.2f}, "
                  f"median |r| {np.median(np.abs(r)):.2f}, 95% |r| {np.percentile(np.abs(r), 95):.2f}")
        m_all = atisim_winds(xk)[idx]
        for lo, hi in ((-25, -18), (18, 26)):
            sel = flat & (xk >= lo) & (xk < hi)
            print(f"   digitised MODEL - AtiSim in the far field {lo:+d}..{hi:+d} kft: "
                  f"mean {np.mean(ym[sel] - m_all[sel]):+.2f} {unit} over {sel.sum()} columns")
        if dots.size:
            ours_d = atisim_winds(dx)[idx]
            r = dy - ours_d
            print(f"   ACTUAL - AtiSim: RMS {np.sqrt(np.mean(r**2)):.2f} {unit}, mean {r.mean():+.2f}, "
                  f"max {r.max():+.2f} at {dx[np.argmax(r)]:+.1f} kft, min {r.min():+.2f} at {dx[np.argmin(r)]:+.1f} kft")
            for lo, hi in ((-25, -5), (-5, 6), (6, 26)):
                sel = (dx >= lo) & (dx < hi)
                if sel.any():
                    print(f"      {lo:+d}..{hi:+d} kft: ACTUAL - AtiSim mean {r[sel].mean():+.2f} {unit} "
                          f"({sel.sum()} dots)")
        # the extremes, where there are any to compare
        if name == "horizontal":
            for lo, hi, kind in ((-10, -4, "min"), (1.5, 5, "min"), (8, 18, "max")):
                sel = flat & (xk >= lo) & (xk < hi)
                f = np.argmin if kind == "min" else np.argmax
                i_d = f(ym[sel])
                xs = np.linspace(lo, hi, 2001)
                ours = atisim_winds(xs)[0]
                i_o = f(ours)
                txt = f"      {kind} in {lo:+}..{hi:+} kft: MODEL {ym[sel][i_d]:.1f} kt at {xk[sel][i_d]:+.2f}; " \
                      f"AtiSim {ours[i_o]:.1f} kt at {xs[i_o]:+.2f}"
                dsel = (dx >= lo) & (dx < hi)
                if dsel.any():
                    j = f(dy[dsel])
                    txt += f"; ACTUAL {dy[dsel][j]:.1f} kt at {dx[dsel][j]:+.2f}"
                print(txt)
        for x, yv in zip(xk[flat], ym[flat]):
            rows_out.append(dict(panel=name, curve="model", x_kft=f"{x:.4f}", value=f"{yv:.4f}"))
        for x, yv in zip(dx, dy):
            rows_out.append(dict(panel=name, curve="actual", x_kft=f"{x:.4f}", value=f"{yv:.4f}"))

    if args.overlay:
        from PIL import Image
        rgb = np.stack([img] * 3, axis=-1).copy()
        for name in PANELS:
            rgb[ex[name]["line_mask"]] = (220, 40, 40)
            for x, y in ex[name]["dots"]:
                rgb[int(round(y)) - 2:int(round(y)) + 3, int(round(x)) - 2:int(round(x)) + 3] = (30, 90, 220)
        out = args.outdir / "tm102186-fig7-classified.png"
        Image.fromarray(rgb).save(out)
        print(f"\noverlay -> {out}")

    if args.write:
        DATA.parent.mkdir(exist_ok=True)
        with open(DATA, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=("panel", "curve", "x_kft", "value"))
            w.writeheader()
            w.writerows(rows_out)
        print(f"wrote {len(rows_out)} points -> {DATA}")


if __name__ == "__main__":
    main()
