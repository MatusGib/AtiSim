"""What the along-track wind does to the 747 at Hannibal, and what the fitted field leaves out.

Reads `atisim/data/tm102186_fig7_winds.csv`, written by
`scripts/digitise_hannibal_horizontal_wind.py` from NASA TM-102186 Fig. 7 (printed
p. 3-5): the horizontal and vertical winds the DC-10 met over Hannibal, MO, as
MEASURED (dotted in the figure) and as Mehta 1987's five-vortex MODEL (solid).

Parks et al. 1985 p. 125 puts the DC-10 "cruising in an easterly direction" with
the jet stream at 240 deg, so the ambient wind blew ~30 deg off the path from
BEHIND -- Mehta's psi = 31 deg -- and a positive horizontal perturbation is a
TAILWIND gust, which is the direction `wind.vortex_wind` adds it in. What the
point form does NOT do is project it: it puts the whole perturbation along the
path, where only cos(31 deg) = 0.857 of it lies.

Five fields, each flown by the 747 with its Mach-derivative seam SHUT (the
airframe before session 30, thrust through the CG) and as SHIPPED (CR-2144's
speed derivatives and thrust line declared), through Mehta's array at 37,000 ft with fixed controls, dt 0.01 and
`vortex_viz.fly_mehta`'s window:

  A  as flown           `wind.vortex_wind`, unchanged
  B  projected          horizontal perturbation x cos(psi)
  C  no along-track     horizontal perturbation removed
  D  B + recorded miss  plus (measured - model) horizontal wind x cos(psi),
                        interpolated along the path between digitised points
  E  wrong side         every core's z flipped -- for scale only
  F  replayed           A's field evaluated at the NOMINAL altitude, so the 747
                        meets exactly the wind the fit says the DC-10 met,
                        whatever its own climb
  G  replayed + miss    D's field, replayed the same way

F IS THE HEADLINE FORM since session 30 (`wind.on_identified_path`, used by
`scripts/cat_validation.py`): Parks 1985 Fig. 6 shows the DC-10 held its
altitude through cores 3 and 4 (`scripts/digitise_parks_fig6_altitude.py`).

WHY F AND G EXIST. Cores 3 and 4 sit 94 ft and 254 ft above the nominal path,
and the updrafts before them lift the 747 by ~500 ft, so it passes ABOVE both
where the straight path passes below -- and the horizontal wind flips sign
across a core while the vertical does not. Mehta placed those cores relative to
the DC-10's ACTUAL flown path, so flying a second aircraft's own excursion
through them counts an altitude change twice. F and G remove that.

Run: PYTHONPATH=<abs worktree root> .venv/Scripts/python.exe \\
         scripts/hannibal_along_track_wind.py --outdir runs/cat
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from digitise_hannibal_horizontal_wind import BIAS_KT, DATA, KT2MS, atisim_winds  # noqa: E402

import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from atisim import viz, vortex_viz, wind  # noqa: E402
from atisim.aircraft import CRUISE, REGISTRY, boeing747_without_thrust_line  # noqa: E402
from atisim.units import FT2M, RAD2DEG  # noqa: E402

COS_PSI = math.cos(math.radians(wind.MEHTA_HANNIBAL_PSI_DEG))
RECORD_PTP = 2.70  # g, TM-102186's +1.7 / -1.0 band

# Report palette -- the validated reference instance (dataviz references/palette.md)
SURF, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
C_ATISIM, C_MEHTA, C_MEASURED = "#2a78d6", "#eb6834", "#1baf7a"
C_BEFORE = "#b9b7ae"  # the de-emphasised "before" in before/after pairs


def style():
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10,
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
        "axes.edgecolor": BASE, "axes.linewidth": 1.0, "axes.labelcolor": INK2,
        "axes.titlecolor": INK, "axes.titlesize": 12, "axes.titleweight": "semibold",
        "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "legend.frameon": False, "legend.labelcolor": INK2, "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    })


def load():
    rows = list(csv.DictReader(open(DATA)))
    out = {}
    for r in rows:
        out.setdefault((r["panel"], r["curve"]), []).append((float(r["x_kft"]), float(r["value"])))
    return {k: np.array(sorted(v)) for k, v in out.items()}


def fields(recorded_miss):
    alt = wind.MEHTA_HANNIBAL_ALTITUDE
    arr = wind.mehta_hannibal_array(alt)
    flipped = arr._replace(down=-2.0 * alt - arr.down)
    xs_m = jnp.asarray(recorded_miss[:, 0] * 1000.0 * FT2M)
    miss_ms = jnp.asarray(recorded_miss[:, 1] * KT2MS * COS_PSI)
    lo, hi = float(xs_m[0]), float(xs_m[-1])

    def as_flown(p):
        return wind.vortex_wind(p, arr)

    def projected(p):
        return wind.vortex_wind(p, arr).at[0].multiply(COS_PSI)

    def no_along(p):
        return wind.vortex_wind(p, arr).at[0].set(0.0)

    def with_miss(p):
        extra = jnp.where((p[0] >= lo) & (p[0] <= hi), jnp.interp(p[0], xs_m, miss_ms), 0.0)
        return projected(p).at[0].add(extra)

    def wrong_side(p):
        return wind.vortex_wind(p, flipped)

    def on_path(field):
        # the wind at this distance along the path, at the NOMINAL altitude
        return lambda p: field(jnp.array([p[0], p[1], -alt]))

    return {
        "A": ("Wind as flown until now", as_flown),
        "B": ("Horizontal gust projected onto the path", projected),
        "C": ("Horizontal gust removed", no_along),
        "D": ("Projected, plus the horizontal wind the fit misses", with_miss),
        "E": ("Cores on the wrong side of the path", wrong_side),
        "F": ("Fitted wind replayed along the path (the headline)", on_path(as_flown)),
        "G": ("Replayed, projected, plus the wind the fit misses", on_path(with_miss)),
    }


def fly(ac, field):
    V = CRUISE["boeing747"]["airspeed"]
    H = wind.MEHTA_HANNIBAL_ALTITUDE
    r0 = float(wind.MEHTA_HANNIBAL_R0)
    x0 = float(wind.MEHTA_HANNIBAL_X_FT[0]) * FT2M
    x1 = float(wind.MEHTA_HANNIBAL_X_FT[-1]) * FT2M
    start = x0 - 12.0 * r0
    enc = vortex_viz.fly_in_moving_air(ac, field, V, H, label="along-track", start_north=start,
                                       seconds=(x1 + 12.0 * r0 - start) / V, dt=0.01,
                                       window=(x0 - 2.0 * r0, x1 + 2.0 * r0), window_name="array")
    w = np.asarray(enc.window)
    d = viz.derived(enc.log)
    speed = np.asarray(d.airspeed)
    if speed.size == w.size + 1:
        speed = speed[1:]
    nz, th = np.asarray(enc.n_z), np.asarray(enc.theta) * RAD2DEG
    return dict(
        nz_min=float(nz[w].min()), nz_max=float(nz[w].max()), ptp=float(nz[w].max() - nz[w].min()),
        pitch_ptp=float(np.ptp(th[w])), alpha_peak=float(np.abs(np.asarray(enc.alpha_air)[w]).max() * RAD2DEG),
        speed_min=float(speed[w].min()), speed_max=float(speed[w].max()),
        north_kft=(np.asarray(enc.north) / FT2M / 1000.0).tolist(), nz=nz.tolist(),
        speed_kt=(speed / KT2MS).tolist(), window=w.tolist(),
        alt_dev_ft=((np.asarray(enc.altitude) - H) / FT2M).tolist())


def rms(a):
    return float(np.sqrt(np.mean(np.asarray(a) ** 2)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", type=Path, default=Path("runs/cat"))
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    style()
    d = load()

    hm, ha = d[("horizontal", "model")], d[("horizontal", "actual")]
    vm, va = d[("vertical", "model")], d[("vertical", "actual")]
    xs = np.linspace(-24.0, 25.0, 1961)
    h_ours, v_ours = atisim_winds(xs)
    h_flip, _ = atisim_winds(xs, flip_z=True)
    fit_h = rms(hm[:, 1] - atisim_winds(hm[:, 0])[0])
    fit_h_flip = rms(hm[:, 1] - atisim_winds(hm[:, 0], flip_z=True)[0])
    fit_v = rms(vm[:, 1] - atisim_winds(vm[:, 0])[1])
    miss = np.column_stack([ha[:, 0], ha[:, 1] - atisim_winds(ha[:, 0])[0]])
    print(f"horizontal: AtiSim vs Mehta's drawn model RMS {fit_h:.2f} kt; cores flipped {fit_h_flip:.2f} kt")
    print(f"vertical control: RMS {fit_v:.2f} ft/s; measured-minus-model horizontal RMS {rms(miss[:, 1]):.2f} kt")

    # ---------------------------------------------------------------- flights
    runs = {}
    bare = boeing747_without_thrust_line()._replace(mach_deriv_ref=jnp.array(-1.0))
    shipped = REGISTRY["boeing747"]
    for key, (label, field) in fields(miss).items():
        for name, ac in (("bare", bare), ("shipped", shipped)):
            r = fly(ac, field)
            runs[f"{key}/{name}"] = r
            print(f"  {key} {name:8s} n_z {r['nz_min']:+.3f}..{r['nz_max']:+.3f}  ptp {r['ptp']:.4f} g "
                  f"({100 * r['ptp'] / RECORD_PTP:.1f}% of record)  pitch ptp {r['pitch_ptp']:.2f} deg  "
                  f"airspeed {r['speed_min'] / KT2MS:.1f}..{r['speed_max'] / KT2MS:.1f} kt  "
                  f"|alpha| {r['alpha_peak']:.2f} deg   [{label}]", flush=True)
    print("\n  where the 747 passes each core (its altitude change plus Mehta's z):")
    for key in ("A", "F"):
        for name in ("bare", "shipped"):
            r = runs[f"{key}/{name}"]
            north, alt_dev = np.array(r["north_kft"]), np.array(r["alt_dev_ft"])
            parts = []
            for k, (xc, zc) in enumerate(zip(wind.MEHTA_HANNIBAL_X_FT, wind.MEHTA_HANNIBAL_Z_FT), start=1):
                i = int(np.argmin(np.abs(north - xc / 1000.0)))
                parts.append(f"core {k} {alt_dev[i] + zc:+.0f} ft")
            print(f"    {key} {name:8s} climb {alt_dev.min():+.0f}..{alt_dev.max():+.0f} ft; "
                  f"height above cores: " + ", ".join(parts))
    summary = {k: {kk: vv for kk, vv in v.items() if not isinstance(vv, list)} for k, v in runs.items()}
    summary["fits"] = dict(horizontal_rms_kt=fit_h, horizontal_flipped_rms_kt=fit_h_flip,
                           vertical_rms_fts=fit_v, measured_minus_model_rms_kt=rms(miss[:, 1]))
    (args.outdir / "hannibal-along-track.json").write_text(json.dumps(summary, indent=2))

    # ---------------------------------------------------------------- figures
    def axes_common(ax, ylabel):
        ax.set_xlabel("Distance along the flight path (thousand feet)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(-25, 25)

    def gaps_broken(a, gap):
        # the reading skips near-vertical strokes and hidden dots; draw a gap, not a bridge
        a = a.copy()
        cut = np.nonzero(np.diff(a[:, 0]) > gap)[0] + 1
        return np.insert(a, cut, np.nan, axis=0)

    hm_line, vm_line = gaps_broken(hm, 0.25), gaps_broken(vm, 0.25)

    # H1 -- horizontal wind: measured, Mehta's model, AtiSim
    fig, ax = plt.subplots(figsize=(9, 4.4), dpi=150)
    ax.plot(ha[:, 0], ha[:, 1], "o", ms=4, color=C_MEASURED, mec=SURF, mew=0.8, label="Measured by the DC-10's recorder")
    ax.plot(hm_line[:, 0], hm_line[:, 1], color=C_MEHTA, label="Mehta's five-vortex model, as drawn")
    ax.plot(xs, h_ours, color=C_ATISIM, lw=1.6, label="The wind AtiSim flies")
    ax.set_title("The horizontal wind over Hannibal: measured, modelled, and simulated")
    axes_common(ax, "Horizontal wind (knots)")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(args.outdir / "h1-horizontal-wind.png")
    plt.close(fig)

    # H2 -- which side of the path
    fig, ax = plt.subplots(figsize=(9, 4.4), dpi=150)
    ax.plot(xs, h_flip, color=C_BEFORE, label=f"Cores on the other side: misses by {fit_h_flip:.1f} kt")
    ax.plot(hm_line[:, 0], hm_line[:, 1], color=C_MEHTA, label="Mehta's model, as drawn")
    ax.plot(xs, h_ours, color=C_ATISIM, lw=1.6, label=f"AtiSim as built: within {fit_h:.1f} kt")
    ax.set_title("Which side of the flight path the vortices sit on")
    axes_common(ax, "Horizontal wind (knots)")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(args.outdir / "h2-which-side.png")
    plt.close(fig)

    # H3 -- vertical wind, the calibration control
    fig, ax = plt.subplots(figsize=(9, 4.4), dpi=150)
    ax.plot(va[:, 0], va[:, 1], "o", ms=4, color=C_MEASURED, mec=SURF, mew=0.8, label="Measured by the DC-10's recorder")
    ax.plot(vm_line[:, 0], vm_line[:, 1], color=C_MEHTA, label="Mehta's model, as drawn")
    ax.plot(xs, v_ours, color=C_ATISIM, lw=1.6, label=f"The wind AtiSim flies (within {fit_v:.1f} ft/s of the drawing)")
    ax.set_title("The vertical wind: the check that the reading itself is right")
    axes_common(ax, "Vertical wind (feet per second)")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(args.outdir / "h3-vertical-control.png")
    plt.close(fig)

    # H4 -- the along-track miss
    fig, ax = plt.subplots(figsize=(9, 4.0), dpi=150)
    ax.axhline(0.0, color=BASE, lw=1.0)
    miss_line = gaps_broken(miss, 1.5)
    ax.plot(miss_line[:, 0], miss_line[:, 1], "-o", ms=4, lw=1.4, color=C_MEASURED, mec=SURF, mew=0.8)
    for lo, hi, below in ((-25, -5, False), (-5, 6, True), (6, 26, False)):
        sel = (miss[:, 0] >= lo) & (miss[:, 0] < hi)
        m = miss[sel, 1].mean()
        a, b = max(lo, -24.5), min(hi, 25)
        ax.plot([a, b], [m, m], color=INK2, lw=1.0)
        # the middle label sits under the left of its line, clear of the dip at +3 kft
        ax.annotate(f"average {m:+.1f} kt", xy=(a, m) if below else ((a + b) / 2, m),
                    xytext=(4, -14) if below else (0, 8), textcoords="offset points",
                    ha="left" if below else "center", color=INK, fontsize=9)
    ax.set_title("How much stronger or weaker the real horizontal wind was than the model")
    axes_common(ax, "Measured minus model (knots)")
    fig.tight_layout()
    fig.savefig(args.outdir / "h4-along-track-miss.png")
    plt.close(fig)

    # H5 -- flight results, two small multiples
    keys = ["A", "B", "C", "D", "E", "F", "G"]
    labels = [fields(miss)[k][0] for k in keys]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), dpi=150, sharey=True)
    y = np.arange(len(keys))[::-1]
    for ax, metric, title, fmt in ((axes[0], "ptp", "Peak-to-peak load, % of the recorded 2.70 g", "{:.1f}%"),
                                   (axes[1], "pitch_ptp", "Peak-to-peak pitch (degrees)", "{:.2f}°")):
        for off, name, color, lab in ((0.19, "bare", C_BEFORE, "Before session 30 (no speed derivatives)"),
                                      (-0.19, "shipped", C_ATISIM, "Now (speed derivatives declared)")):
            vals = [runs[f"{k}/{name}"][metric] * (100.0 / RECORD_PTP if metric == "ptp" else 1.0) for k in keys]
            ax.barh(y + off, vals, height=0.34, color=color, label=lab)
            for yy, v in zip(y + off, vals):
                ax.text(v, yy, " " + fmt.format(v), va="center", ha="left", color=INK2, fontsize=8)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, max(ax.get_xlim()[1], 1) * 1.15)
    axes[0].set_yticks(y, labels)
    axes[0].legend(loc="lower center", bbox_to_anchor=(1.0, -0.24), ncol=2)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.06)
    fig.savefig(args.outdir / "h5-flight-results.png")
    plt.close(fig)

    # H7 -- side view: where the 747 flies relative to the cores, to scale in feet
    from matplotlib.patches import Ellipse
    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=150)
    r0_ft = float(wind.MEHTA_HANNIBAL_R0) / FT2M
    for xc, zc in zip(wind.MEHTA_HANNIBAL_X_FT, wind.MEHTA_HANNIBAL_Z_FT):
        # z is the aircraft's height above the core, so the core sits at -z
        ax.add_patch(Ellipse((xc / 1000.0, -zc), 2 * r0_ft / 1000.0, 2 * r0_ft,
                             facecolor=GRID, edgecolor=MUTED, lw=1.0))
    ax.axhline(0.0, color=C_MEHTA, lw=2.0, label="The path the vortex fit was made along")
    for name, color, lab in (("bare", C_BEFORE, "747 before session 30"),
                             ("shipped", C_ATISIM, "747 as shipped")):
        r = runs[f"A/{name}"]
        ax.plot(r["north_kft"], r["alt_dev_ft"], color=color, label=f"{lab}: where it actually flies")
    # The DC-10's own path: Parks 1985 Fig. 6's inertial altitude estimate, put on
    # Mehta's axis under both readings of its distance scale (the digitiser's docstring)
    import digitise_parks_fig6_altitude as parks
    rec = parks.load()
    xk = np.linspace(-15.0, 15.0, 301)
    for (kind, v), ls in zip(reversed(parks.speeds(parks.anchors(rec)["tas_cruise"]).items()), ("-", ":")):
        ax.plot(xk, parks.dc10_path(rec, v, xk), color=C_MEASURED, ls=ls, lw=2.0 if ls == "-" else 1.6,
                label="The DC-10's own path (Parks 1985 Fig. 6)" if kind == "ground"
                else "   the same, on an air-relative distance scale")
    ax.set_xlim(-15, 15)
    ax.set_ylim(-2400, 2500)
    ax.set_xlabel("Distance along the flight path (thousand feet)")
    ax.set_ylabel("Height relative to 37,000 ft (feet)")
    ax.set_title("The simulated 747 climbs over the two central cores; the DC-10 stayed beneath them")
    ax.legend(loc="lower left")
    ax.annotate("vortex core, to scale (1,000 ft across)", xy=(-6.669 + r0_ft / 1000.0, 1836),
                xytext=(6, 0), textcoords="offset points", ha="left", va="center", color=INK2, fontsize=8)
    ax.annotate("core 1 is 3,500 ft above, off the chart", xy=(-14.8, 2500), xytext=(0, -12),
                textcoords="offset points", ha="left", color=INK2, fontsize=8)
    fig.tight_layout()
    fig.savefig(args.outdir / "h7-side-view.png")
    plt.close(fig)

    # H6 -- airspeed and load along the path, shipped 747, A against D
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.2), dpi=150, sharex=True)
    for key, color, lab in (("A", C_BEFORE, "Wind as flown until now"),
                            ("D", C_ATISIM, "With the recorded horizontal wind added")):
        r = runs[f"{key}/shipped"]
        axes[0].plot(r["north_kft"], r["speed_kt"], color=color, label=lab)
        axes[1].plot(r["north_kft"], r["nz"], color=color, label=lab)
    for v, txt in ((1.7, "recorded +1.7 g"), (-1.0, "recorded −1.0 g")):
        axes[1].axhline(v, color=MUTED, lw=1.0)
        axes[1].annotate(txt, xy=(-24.5, v), xytext=(0, 3), textcoords="offset points", color=INK2, fontsize=8)
    axes[0].set_ylabel("Airspeed (knots)")
    axes[1].set_ylabel("Load factor (g)")
    axes[0].set_title("The 747 as shipped: airspeed and load through the vortex field")
    axes[0].legend(loc="lower left")
    axes_common(axes[1], "Load factor (g)")
    fig.tight_layout()
    fig.savefig(args.outdir / "h6-airspeed-and-load.png")
    plt.close(fig)
    print(f"figures -> {args.outdir}")


if __name__ == "__main__":
    main()
