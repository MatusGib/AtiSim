"""Parks et al. 1985 Fig. 6: the DC-10's ALTITUDE through the Hannibal encounter, digitised.

SOURCE. E. K. Parks, R. C. Wingrove, R. E. Bach Jr., R. S. Mehta, "Identification
of Vortex-Induced Clear Air Turbulence Using Airline Flight Records", J. Aircraft
22(2), Feb 1985, pp. 124-129, DOI 10.2514/3.45095 -- `Reference_papers/parks-1985-
identification-of-vortex-induced-clear-air-turbulence-JA22-2.pdf`, outside git.
Figure 6, printed p. 127 (PDF page index 3), "Time-history data from the
turbulence encounter near Hannibal, Mo., April 3, 1981": seven stacked panels
against GMT 1:21-1:26 -- horizontal wind, vertical wind, normal acceleration,
pitch, true airspeed, static air temperature and ALTITUDE.

WHY. PROJECT.md section 8 (session 30): the simulated 747 climbs ~500-600 ft
before Mehta's cores 3 and 4 and passes above both, where Mehta placed them just
above the DC-10's path. Flown that way the Hannibal headline reads 64.5% of the
record; with the fitted wind replayed at the nominal altitude, 75.0%. Which is
right depends on whether the DC-10 itself climbed through the pair.

THE ALTITUDE PANEL HOLDS TWO TRACES, and they are not the same quantity.
  solid   the MEASURED (barometric) altitude. p. 127: its fluctuations "indicate
          localized variations in the flowfield such as might be expected in the
          vicinity of vortices" -- pressure, not height.
  dashed  "ESTIMATE", the altitude estimated from inertial measurements. p. 126:
          integrating the accelerations gives inertial velocity and position,
          with biases set by matching x, y to radar and z to the measured
          altitude, and the winds are computed along that path. So the dashed
          trace is the path the vortex identification was made along.

METHOD. Render at the scan's native 300 dpi, binarise. Panels are found by their
left axis lines. The scan is skewed ~0.2 deg, so every panel is calibrated on its
OWN tick marks: x through its minute ticks (1:21-1:26), y through its left-axis
ticks at the printed label values. Nothing is fitted to a trace and nothing is
smoothed. On the altitude panel the solid trace is the one long ink component
and the dashed trace is the short dash components where the two part; the
ESTIMATE label and its leader line are removed first, the leader by a straight-
line fit. Load and vertical-wind panels are read as the top and bottom of the ink
per column, as `digitise_tm102186_fig6.py` reads the same record.

CHECKS, none used to set a parameter:
  1  THE PAPER'S OWN BAND. p. 127: normal acceleration "from +1.7 to -1.0 gs".
  2  CRUISE ALTITUDE. p. 126: "cruising ... at 37,000 ft". The pre-encounter
     barometric trace must sit there.
  3  ONE CLOCK ACROSS PANELS. The load minimum and the vertical-wind minimum,
     read on separately calibrated panels, must fall in the same second.

TIME TO DISTANCE. Mehta's cores are placed in feet along the path; the figure is
in seconds. The conversion anchors the record's deepest downdraft (the vertical-
wind minimum here) to where TM-102186 Fig. 7 draws the MEASURED vertical-wind
minimum (`atisim/data/tm102186_fig7_winds.csv`), and spans a DECLARED speed band:
air-relative (the true airspeed read here) to ground speed (plus Mehta's 149.8 kt
bias x cos 31 deg). Which one Mehta's distance axis is, no source held here says.

Run: PYTHONPATH=<abs worktree root> .venv/Scripts/python.exe \\
         scripts/digitise_parks_fig6_altitude.py --pdf <Reference_papers/parks-1985-...pdf> \\
         --outdir runs/cat [--overlay] [--write]
"""

import argparse
import csv
import math
from pathlib import Path

import atisim

print(f"atisim imported from: {atisim.__file__}")

import fitz  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

from atisim import wind  # noqa: E402

PAGE = 3  # printed p. 127
CLIP = fitz.Rect(300, 40, 619, 440)
DPI = 300  # the page image is 2584 px over 619 pt: the scan's own resolution
DATA = Path(atisim.__file__).parent / "data" / "parks1985_fig6_altitude.csv"
FIG7 = Path(atisim.__file__).parent / "data" / "tm102186_fig7_winds.csv"

# Panels top to bottom, with the label values printed on each left axis.
PANELS = (
    ("horizontal_wind", (160.0, 140.0, 120.0)),
    ("vertical_wind", (100.0, 50.0, 0.0, -50.0, -100.0)),
    ("normal_acceleration", (2.0, 1.0, 0.0, -1.0)),
    ("pitch", (10.0, 5.0, 0.0)),
    ("true_airspeed", (520.0, 480.0, 440.0)),
    ("static_temperature", (-50.0, -60.0)),
    ("altitude", (37400.0, 37200.0, 37000.0)),
)
AXIS_BAND = (329, 342)  # image columns holding every panel's left axis line
PAPER_NZ = (-1.0, 1.7)  # p. 127
PAPER_ALTITUDE_FT = 37000.0  # p. 126
KT2FTS = 1.68781
BIAS_KT = 149.8  # Mehta 1987 p. 30, b_xy

# Where the ESTIMATE label and leader can be, in (seconds after 1:21, feet). Only
# a search region: the leader itself is found by a line fit inside it, and the
# overlay shows what was removed.
LABEL_REGION = (243.0, 320.0, 37270.0, 37420.0)
LEADER_REGION = (218.0, 262.0, 37090.0, 37400.0)
DASH_REGION = (180.0, 245.0, 36900.0, 37380.0)


def render(pdf: Path) -> np.ndarray:
    doc = fitz.open(pdf)
    pix = doc[PAGE].get_pixmap(dpi=DPI, clip=CLIP, colorspace=fitz.csGRAY)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    doc.close()
    return img


def runs(mask_1d, min_len=1, gap=0):
    """(start, end) inclusive of True runs, bridging gaps of <= `gap`."""
    idx = np.nonzero(mask_1d)[0]
    if idx.size == 0:
        return []
    out = [[idx[0], idx[0]]]
    for i in idx[1:]:
        if i - out[-1][1] <= gap + 1:
            out[-1][1] = i
        else:
            out.append([i, i])
    return [(a, b) for a, b in out if b - a + 1 >= min_len]


def find_panels(ink):
    rows = ink[:, AXIS_BAND[0]:AXIS_BAND[1]].any(axis=1)
    segs = runs(rows, min_len=45, gap=2)
    if len(segs) != len(PANELS):
        raise SystemExit(f"expected {len(PANELS)} axis segments, found {len(segs)}: {segs}")
    panels = {}
    for (name, labels), (top, bot) in zip(PANELS, segs):
        cols = np.nonzero(ink[top:bot + 1, AXIS_BAND[0]:AXIS_BAND[1]].sum(axis=0) > 0.6 * (bot - top))[0]
        axis_x = int(AXIS_BAND[0] + cols.max())  # right edge of the axis line
        # the bottom axis. The skew steps the line across 2-3 rows, so it is found
        # as the rows near the segment's foot carrying a long total of ink
        axis_rows = [r for r in range(bot - 5, bot + 6) if ink[r, axis_x + 5:axis_x + 480].sum() >= 150]
        r0, r1 = min(axis_rows), max(axis_rows)
        right = int(axis_x + np.nonzero(ink[r0:r1 + 1, axis_x:].sum(axis=0) >= 1)[0].max())
        panels[name] = dict(labels=labels, top=top, bot=bot, axis_x=axis_x, axis_rows=(r0, r1), right=right)
    return panels


def y_calibration(ink, p):
    """Left-axis tick rows at the printed label values, least squares.

    The ticks point INTO the plot: a short run just right of the axis line that
    stops within a few pixels, where a trace touching the axis would carry on.
    The lowest label sits on the bottom axis line itself, so its row is that
    line's centre."""
    x = p["axis_x"]
    near = ink[:, x + 1:x + 5].sum(axis=1) >= 3
    far = ink[:, x + 8:x + 12].any(axis=1)
    rows = near & ~far
    rows[:p["top"] - 3] = False
    rows[p["axis_rows"][0] - 2:] = False
    cand = [(a + b) / 2.0 for a, b in runs(rows, gap=1)]
    labels = p["labels"]
    bottom = (p["axis_rows"][0] + p["axis_rows"][1]) / 2.0
    # Prior only for MATCHING ticks to labels: the top label at the top of the
    # axis line, the bottom one on the bottom axis. A trace that meets the axis
    # hides its tick (true airspeed at 480 kt), so a label may go unmatched;
    # the top label then falls back to the axis line's own end.
    prior = np.polyfit([labels[0], labels[-1]], [p["top"], bottom], 1)
    pairs = [(labels[-1], bottom)]
    for v in labels[:-1]:
        y_pred = np.polyval(prior, v)
        hits = [c for c in cand if abs(c - y_pred) <= 4.0]
        if hits:
            pairs.append((v, min(hits, key=lambda c: abs(c - y_pred))))
        elif v == labels[0]:
            pairs.append((v, float(p["top"])))
    vs, ys = map(np.array, zip(*pairs))
    coef = np.polyfit(vs, ys, 1)
    return coef, float(np.abs(np.polyval(coef, vs) - ys).max()), sorted(ys.tolist())


def x_calibration(ink, p, below):
    """Minute ticks on the bottom axis, least squares. Upper panels draw them up
    into the plot; the altitude panel hangs them detached below its axis. On the
    upper panels 1:21 is the left axis line itself, so only 1:22-1:26 are used,
    and a trace crossing a tick can hide it, so any four or more will do."""
    r0, r1 = p["axis_rows"]
    band = ink[r1 + 2:r1 + 20] if below else ink[r0 - 11:r0 - 1]
    cand = [(a + b) / 2.0 for a, b in runs(band.sum(axis=0) >= 5, gap=1) if b - a <= 5]
    minutes = range(0, 6) if below else range(1, 6)
    coef = (97.0, float(p["axis_x"]))  # matching prior: ~97 px per minute from the axis
    for _ in range(2):
        pairs = []
        for m in minutes:
            x_pred = coef[1] + coef[0] * m
            hits = [c for c in cand if abs(c - x_pred) <= 6.0]
            if hits:
                pairs.append((m, min(hits, key=lambda c: abs(c - x_pred))))
        if len(pairs) < 4:
            raise SystemExit(f"x ticks: matched {pairs} from candidates {cand}")
        ms, xs = map(np.array, zip(*pairs))
        coef = tuple(np.polyfit(ms, xs, 1))
    res = float(np.abs(np.polyval(coef, ms) - xs).max())
    return np.array(coef), res, pairs


def plot_box(ink, p, below_extra=0, above_extra=6):
    """The panel's plot area with the axis line and tick marks removed."""
    r0, r1 = p["axis_rows"]
    top, bot = p["top"] - above_extra, r1 + 1 + below_extra
    x0, x1 = p["axis_x"] + 3, p["right"] - 2
    box = ink[top:bot, x0:x1].copy()
    box[r0 - top:r1 + 1 - top, :] = False
    return box, top, x0


# ---------------------------------------------------------------------------
# Reading the table. Everything below works on the tracked CSV, or on the rows
# main() is about to write, so the script, the tests and the flight figures use
# one set of numbers.
# ---------------------------------------------------------------------------

RUN_START_KFT = (wind.MEHTA_HANNIBAL_X_FT[0] - 12.0 * 500.5) / 1000.0  # where the 747's run starts


def table(rows):
    out = {}
    for panel, curve, t, v in rows:
        out.setdefault((panel, curve), []).append((float(t), float(v)))
    return {k: np.array(sorted(v)) for k, v in out.items()}


def load(path=DATA):
    with open(path, newline="") as f:
        return table((r["panel"], r["curve"], r["t_s_after_0121"], r["value"]) for r in csv.DictReader(f))


def barometric(d):
    """The solid trace, with the columns it spends hidden under the 37,000 ft axis line."""
    a = np.concatenate([d[k] for k in (("altitude", "barometric"), ("altitude", "barometric_on_axis")) if k in d])
    return a[np.argsort(a[:, 0])]


def estimate_at(d, t):
    """The inertial estimate: the dashes where it parts from the solid trace, and
    the solid trace itself where the two are drawn as one line."""
    est, baro = d[("altitude", "inertial_estimate")], barometric(d)
    if est[0, 0] - 1.0 <= t <= est[-1, 0] + 1.0:
        return float(np.interp(t, est[:, 0], est[:, 1]))
    return float(np.interp(t, baro[:, 0], baro[:, 1]))


def anchors(d):
    nz_lo, nz_hi = d[("normal_acceleration", "ink_bottom")], d[("normal_acceleration", "ink_top")]
    wz_lo = d[("vertical_wind", "ink_bottom")]
    tas_lo, tas_hi = d[("true_airspeed", "ink_bottom")], d[("true_airspeed", "ink_top")]
    quiet = tas_lo[:, 0] < 180.0  # 1:21-1:24, before the encounter
    return dict(
        t_nz_min=float(nz_lo[np.argmin(nz_lo[:, 1]), 0]), nz_min=float(nz_lo[:, 1].min()),
        nz_max=float(nz_hi[:, 1].max()), t_wz_min=float(wz_lo[np.argmin(wz_lo[:, 1]), 0]),
        wz_min=float(wz_lo[:, 1].min()),
        tas_cruise=float(np.median((tas_lo[quiet, 1] + tas_hi[quiet, 1]) / 2.0)))


def fig7_anchor():
    """(x kft, value) of TM-102186 Fig. 7's MEASURED vertical-wind minimum."""
    with open(FIG7, newline="") as f:
        dots = [r for r in csv.DictReader(f) if r["panel"] == "vertical" and r["curve"] == "actual"]
    r = min(dots, key=lambda r: float(r["value"]))
    return float(r["x_kft"]), float(r["value"])


def speeds(tas_kt):
    """DECLARED band for Mehta's distance axis: air-relative, or ground (plus his
    bias wind's along-path component). No source held says which it is."""
    cos_psi = math.cos(math.radians(wind.MEHTA_HANNIBAL_PSI_DEG))
    return {"air-relative": tas_kt * KT2FTS, "ground": (tas_kt + BIAS_KT * cos_psi) * KT2FTS}


def clock(d, speed_fts):
    """Seconds after 1:21:00 at which the DC-10 was at `x_kft` on Mehta's axis."""
    t0, (x0, _) = anchors(d)["t_wz_min"], fig7_anchor()
    return lambda x_kft: t0 + (np.asarray(x_kft) - x0) * 1000.0 / speed_fts


def dc10_path(d, speed_fts, x_kft):
    """The DC-10's inertially estimated altitude minus 37,000 ft, on Mehta's axis."""
    t_at = clock(d, speed_fts)
    return np.array([estimate_at(d, float(t)) for t in t_at(x_kft)]) - PAPER_ALTITUDE_FT


def figure(d, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from hannibal_along_track_wind import BASE, C_ATISIM, C_BEFORE, C_MEASURED, INK2, MUTED, style
    style()
    a = anchors(d)
    baro, est = barometric(d), d[("altitude", "inertial_estimate")]
    nz_lo, nz_hi = d[("normal_acceleration", "ink_bottom")], d[("normal_acceleration", "ink_top")]
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(10, 6.0), dpi=150, sharex=True,
                                 gridspec_kw=dict(height_ratios=(2.2, 1.0)))
    tas = a["tas_cruise"]
    for (kind, v), alpha in zip(speeds(tas).items(), (0.10, 0.18)):
        t_at = clock(d, v)
        t_cores = t_at(np.array(wind.MEHTA_HANNIBAL_X_FT) / 1000.0)
        ax.axvspan(t_cores[2], t_cores[3], color=MUTED, alpha=alpha, lw=0)
        bx.axvspan(t_cores[2], t_cores[3], color=MUTED, alpha=alpha, lw=0)
    ax.plot(baro[:, 0], baro[:, 1], color=C_BEFORE, lw=1.6, label="Measured (barometric) altitude")
    ax.plot(est[:, 0], est[:, 1], "o", ms=2.6, color=C_ATISIM, label="Altitude estimated from inertial measurements")
    ax.axhline(PAPER_ALTITUDE_FT, color=BASE, lw=1.0)
    ax.set_ylabel("Altitude (feet)")
    ax.set_title("The DC-10 held 37,000 ft through the two central vortices, and climbed after them")
    ax.legend(loc="upper left")
    ax.annotate("passing cores 3 and 4\n(two readings of the distance scale)",
                xy=(clock(d, speeds(tas)["ground"])(-0.343), 37330), xytext=(-8, 0),
                textcoords="offset points", ha="right", color=INK2, fontsize=8)
    bx.fill_between(nz_lo[:, 0], nz_lo[:, 1], np.interp(nz_lo[:, 0], nz_hi[:, 0], nz_hi[:, 1]),
                    color=C_MEASURED, lw=0.6, edgecolor=C_MEASURED)
    bx.set_ylabel("Load (g)")
    bx.set_xlabel("Seconds after 1:21:00 GMT, 3 April 1981")
    ax.set_xlim(150, 300)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, default=Path("runs/cat"))
    ap.add_argument("--overlay", action="store_true")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    img = render(args.pdf)
    ink = img < 140
    panels = find_panels(ink)
    print(f"\nrender: page index {PAGE} at {DPI} dpi, {img.shape[1]}x{img.shape[0]} px")
    for name, p in panels.items():
        ycoef, yres, yticks = y_calibration(ink, p)
        xcoef, xres, _ = x_calibration(ink, p, below=(name == "altitude"))
        p.update(ycoef=ycoef, yres=yres, xcoef=xcoef, xres=xres)
        unit = abs(ycoef[0])
        print(f"  {name:<20} axis x {p['axis_x']}, rows {p['top']}..{p['bot']}; "
              f"{1 / unit:.3f} units per px (worst y tick {yres:.2f} px); "
              f"{xcoef[0]:.2f} px per minute, 1:21 at x {xcoef[1]:.1f} (worst x tick {xres:.2f} px)\n"
              f"  {'':<20} y ticks at rows {[round(float(v), 1) for v in yticks]}")

    def t_of(p, x):
        return (x - p["xcoef"][1]) / p["xcoef"][0] * 60.0

    def v_of(p, y):
        return (y - p["ycoef"][1]) / p["ycoef"][0]

    out_rows = []
    classes = {}

    # ---------------------------------------------------------------- load, vertical wind, airspeed
    for name in ("normal_acceleration", "vertical_wind", "true_airspeed"):
        p = panels[name]
        box, top, x0 = plot_box(ink, p)
        for m in range(0, 6):  # tick marks rise into the plot
            xc = int(round(p["xcoef"][1] + p["xcoef"][0] * m)) - x0
            box[p["axis_rows"][0] - 12 - top:p["axis_rows"][0] - top, max(xc - 3, 0):xc + 4] = False
        lab, n = ndimage.label(box, structure=np.ones((3, 3)))
        sizes = ndimage.sum(box, lab, range(1, n + 1))
        keep = np.isin(lab, 1 + np.nonzero(sizes >= 15)[0])
        ts, lo, hi = [], [], []
        for j in range(box.shape[1]):
            ys = np.nonzero(keep[:, j])[0]
            if ys.size == 0:
                continue
            ts.append(t_of(p, x0 + j))
            hi.append(v_of(p, top + ys.min()))
            lo.append(v_of(p, top + ys.max()))
        ts, lo, hi = map(np.array, (ts, lo, hi))
        classes[name] = (keep, top, x0)
        for t, a, b in zip(ts, lo, hi):
            out_rows.append((name, "ink_bottom", t, a))
            out_rows.append((name, "ink_top", t, b))

    # ---------------------------------------------------------------- altitude
    p = panels["altitude"]
    # the climb after the encounter overshoots the 37,400 ft tick, so the box runs above the axis
    box, top, x0 = plot_box(ink, p, below_extra=16, above_extra=30)
    r0, r1 = p["axis_rows"]
    for m in range(0, 6):
        xc = int(round(p["xcoef"][1] + p["xcoef"][0] * m)) - x0
        box[r1 + 1 - top:r1 + 20 - top, max(xc - 3, 0):xc + 4] = False  # the ticks hanging below
        box[r0 - 8 - top:r0 - top, max(xc - 2, 0):xc + 3] = False  # and their stubs above the axis
    H, W = box.shape
    tt = t_of(p, x0 + np.arange(W))
    vv = v_of(p, top + np.arange(H))

    def region(r):
        cols = (tt >= r[0]) & (tt <= r[1])
        rows = (vv >= r[2]) & (vv <= r[3])
        return np.outer(rows, cols)

    # the ESTIMATE glyphs: small components wholly inside the label region
    lab, n = ndimage.label(box, structure=np.ones((3, 3)))
    objs = ndimage.find_objects(lab)
    lab_region = region(LABEL_REGION)
    glyph = np.zeros_like(box)
    for k, sl in enumerate(objs, start=1):
        comp = lab[sl] == k
        if comp.sum() < 400 and lab_region[sl][comp].all():
            glyph[sl] |= comp

    # the leader: a straight line fitted to the ink in its region that is not glyph
    lr = region(LEADER_REGION) & box & ~glyph
    ys, xs = np.nonzero(lr)
    best = None
    for ang in np.radians(np.arange(30.0, 70.0, 0.5)):
        # lines of the form  y = y0 - tan(ang) * (x - x_ref)  in image rows/cols
        d = ys + np.tan(ang) * xs
        hist, edges = np.histogram(d, bins=np.arange(d.min(), d.max() + 2, 1.0))
        k = hist.argmax()
        if best is None or hist[k] > best[0]:
            best = (hist[k], ang, (edges[k] + edges[k + 1]) / 2.0)
    votes, ang, d0 = best
    on_line = np.abs(ys + np.tan(ang) * xs - d0) <= 2.5
    leader = np.zeros_like(box)
    yy, xx = np.mgrid[0:H, 0:W]
    # perpendicular distance, in pixels, from the fitted leader line
    to_leader = np.abs(yy + np.tan(ang) * xx - d0) * math.cos(ang)
    if on_line.sum() >= 40:
        lx = xs[on_line]
        span = (xx >= lx.min() - 1) & (xx <= lx.max() + 1)
        # the scan's anti-aliasing leaves a 1 px fringe just outside the stroke
        leader = (to_leader <= 2.5) & span & box
    print(f"\n  ESTIMATE label: {int(glyph.sum())} px removed; leader line at {math.degrees(ang):.1f} deg, "
          f"{int(on_line.sum())} px on the fitted line, {int(leader.sum())} px removed")

    # What is left is the solid trace -- cut into pieces where the leader crossed
    # it and where its spikes thin to nothing -- the dashes, and specks. A dash is
    # a short, low component inside the region where the two traces part; the
    # solid trace is every other component of real size.
    clean = box & ~glyph & ~leader
    lab, n = ndimage.label(clean, structure=np.ones((3, 3)))
    dash_region = region(DASH_REGION)
    solid = np.zeros_like(clean)
    dashes = np.zeros_like(clean)
    for k, sl in enumerate(ndimage.find_objects(lab), start=1):
        comp = lab[sl] == k
        size = int(comp.sum())
        h = sl[0].stop - sl[0].start
        if leader.any() and to_leader[sl][comp].max() <= 4.0 and (leader[sl].any() or size < 20):
            leader[sl] |= comp  # a sliver of the leader's own stroke, not a dash
        elif 3 <= size <= 120 and h <= 14 and dash_region[sl][comp].all():
            dashes[sl] |= comp
        elif size >= 20:
            solid[sl] |= comp
    specks = clean & ~solid & ~dashes
    print(f"  solid trace {int(solid.sum())} px; {int(ndimage.label(dashes, structure=np.ones((3, 3)))[1])} dash "
          f"components, {int(dashes.sum())} px; {int(specks.sum())} px of specks unused")

    axis_v = v_of(p, (p["axis_rows"][0] + p["axis_rows"][1]) / 2.0)
    inked = {j: vv[np.nonzero(solid[:, j])[0]].mean() for j in range(W) if solid[:, j].any()}
    cols = np.array(sorted(inked))
    baro = []
    for j in range(W):
        if j in inked:
            baro.append((tt[j], inked[j], 0))
        elif 0.0 < tt[j] < 300.0 and cols.size:
            # An empty column is the trace hidden under the axis line only if it
            # meets the axis on both sides; otherwise it is a gap (where the
            # leader crossed it) and is left out, to be interpolated.
            left, right = cols[cols < j], cols[cols > j]
            if left.size and right.size and all(abs(inked[k] - axis_v) <= 15.0 for k in (left[-1], right[0])):
                baro.append((tt[j], axis_v, 1))
    baro = np.array(baro)
    est = []
    for j in range(W):
        ys_ = np.nonzero(dashes[:, j])[0]
        if ys_.size:
            est.append((tt[j], vv[ys_].mean()))
    est = np.array(est)
    for t, c, on_axis in baro:
        out_rows.append(("altitude", "barometric_on_axis" if on_axis else "barometric", t, c))
    for t, c in est:
        out_rows.append(("altitude", "inertial_estimate", t, c))

    # ---------------------------------------------------------------- checks
    d = table(out_rows)
    a = anchors(d)
    baro = barometric(d)
    est = d[("altitude", "inertial_estimate")]
    print("\nCHECKS")
    print(f"  1  normal acceleration extremes {a['nz_min']:+.2f} / {a['nz_max']:+.2f} g"
          f"   (paper, p. 127: {PAPER_NZ[0]:+.1f} / {PAPER_NZ[1]:+.1f})")
    pre = baro[(baro[:, 0] >= 0.0) & (baro[:, 0] < 180.0)]
    print(f"  2  pre-encounter barometric altitude, 1:21-1:24: mean {pre[:, 1].mean():.0f} ft, "
          f"range {pre[:, 1].min():.0f}..{pre[:, 1].max():.0f}   (paper, p. 126: {PAPER_ALTITUDE_FT:.0f})")
    print(f"  3  load minimum at t {a['t_nz_min']:.1f} s; vertical-wind minimum at t {a['t_wz_min']:.1f} s "
          f"(differ by {abs(a['t_nz_min'] - a['t_wz_min']):.1f} s; one column is "
          f"{60.0 / panels['altitude']['xcoef'][0]:.2f} s)")

    # ---------------------------------------------------------------- what the altitude says
    print("\nTHE ALTITUDE TRACES")
    print(f"  dashed estimate parts from the solid trace over t {est[0, 0]:.1f}..{est[-1, 0]:.1f} s")
    i = np.argmin(est[:, 1])
    print(f"  estimate minimum {est[i, 1]:.0f} ft at t {est[i, 0]:.1f} s")
    j = np.argmax(baro[:, 1])
    print(f"  barometric maximum {baro[j, 1]:.0f} ft at t {baro[j, 0]:.1f} s "
          f"({baro[j, 0] - a['t_wz_min']:+.1f} s after the vertical-wind minimum)")

    x_anchor_kft, w_anchor = fig7_anchor()
    print(f"\nTIME TO DISTANCE: the vertical-wind minimum (t {a['t_wz_min']:.1f} s) is TM-102186 Fig. 7's "
          f"measured minimum, {w_anchor:+.0f} ft/s at x {x_anchor_kft:+.2f} kft")
    print(f"  pre-encounter true airspeed read here {a['tas_cruise']:.0f} kt")
    for kind, v in speeds(a["tas_cruise"]).items():
        t_at = clock(d, v)
        base = estimate_at(d, t_at(RUN_START_KFT))
        print(f"  {kind}, {v:.0f} ft/s:")
        print(f"    747 run start x {RUN_START_KFT:+.1f} kft -> t {t_at(RUN_START_KFT):.1f} s: estimate {base:.0f} ft")
        for c, (xc, zc) in enumerate(zip(wind.MEHTA_HANNIBAL_X_FT, wind.MEHTA_HANNIBAL_Z_FT), start=1):
            t = t_at(xc / 1000.0)
            e = estimate_at(d, t)
            print(f"    core {c} x {xc / 1000:+6.2f} kft -> t {t:6.1f} s: estimate {e:6.0f} ft "
                  f"({e - base:+4.0f} since run start; above core {e - PAPER_ALTITUDE_FT + zc:+5.0f} ft); "
                  f"barometric {float(np.interp(t, baro[:, 0], baro[:, 1])):6.0f}")
        print(f"    barometric maximum at x {x_anchor_kft + (baro[j, 0] - a['t_wz_min']) * v / 1000:+.1f} kft")

    figure(d, args.outdir / "parks-fig6-altitude.png")
    print(f"\nfigure -> {args.outdir / 'parks-fig6-altitude.png'}")

    if args.overlay:
        from PIL import Image
        rgb = np.stack([img] * 3, axis=-1).copy()
        paint = {
            "solid": (solid, (220, 40, 40)), "dashes": (dashes, (30, 110, 230)),
            "glyph": (glyph, (150, 150, 150)), "leader": (leader, (240, 160, 0)),
        }
        for mask, colour in paint.values():
            yy, xx = np.nonzero(mask)
            rgb[yy + top, xx + x0] = colour
        for name in ("normal_acceleration", "vertical_wind", "true_airspeed"):
            keep, ktop, kx0 = classes[name]
            yy, xx = np.nonzero(keep)
            rgb[yy + ktop, xx + kx0] = (220, 40, 40)
        out = args.outdir / "parks-fig6-classified.png"
        Image.fromarray(rgb).save(out)
        print(f"\noverlay -> {out}")

    if args.write:
        DATA.parent.mkdir(parents=True, exist_ok=True)
        with open(DATA, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["panel", "curve", "t_s_after_0121", "value"])
            for r in out_rows:
                w.writerow([r[0], r[1], f"{r[2]:.3f}", f"{r[3]:.4f}"])
        print(f"wrote {len(out_rows)} rows -> {DATA}")


if __name__ == "__main__":
    main()
