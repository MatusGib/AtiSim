"""Figures for session 30's report on CR-2144's speed derivatives.

Draws four figures from the tracked digitisation (`atisim/data/
cr2144_p220_222_digitised.csv`) and the analysis in `scripts/cr2144_speed_derivatives.py`,
whose own functions are imported rather than restated, so a figure cannot drift
from the numbers PROJECT.md section 4 quotes:

  s1  the hand-read curves against the values CR-2144's Table IX-4 implies
  s2  the four longitudinal modes against Table IX-5: before, with the speed
      derivatives alone, and with CR-2144's thrust line as well
  s3  how far each source of reading error could move the phugoid result
  s4  the load through Mehta's Hannibal field, before and after

Run: PYTHONPATH=<abs worktree root> .venv/Scripts/python.exe \\
         scripts/cr2144_report_figures.py --outdir runs/cat
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import cr2144_speed_derivatives as sd  # noqa: E402  (prints which atisim it imported)
from hannibal_along_track_wind import (  # noqa: E402
    BASE, C_ATISIM, C_BEFORE, C_MEASURED, C_MEHTA, INK, INK2, MUTED, RECORD_PTP, SURF, fly, style)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from atisim import cr2144_mach as cm  # noqa: E402
from atisim import wind  # noqa: E402

C_SPEED_ONLY = "#9ec3ee"  # a lighter step of the AtiSim blue: the intermediate state
ALT_COLOUR = {"40K": C_ATISIM, "20K": C_MEHTA, "SL": C_MEASURED}
ALT_NAME = {"40K": "40,000 ft (the cruise)", "20K": "20,000 ft", "SL": "sea level"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", type=Path, default=Path("runs/cat"))
    ap.add_argument("--mc", type=int, default=2000)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    style()

    # s1 -- the curves and the table values they must pass through
    panels = (("cm_m", "Pitching moment change with Mach"), ("cl_m", "Lift change with Mach"),
              ("cd_m", "Drag change with Mach"))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), dpi=150)
    for ax, (q, title) in zip(axes, panels):
        for alt in ("SL", "20K", "40K"):
            c = cm.curves().get((q, alt))
            if c is None:
                continue
            ax.plot(c.mach, c.value, color=ALT_COLOUR[alt], lw=1.8)
            for fc, (a, mach, *_rest) in cm.IX3.items():
                if a == alt and np.isfinite(cm.value(q, alt, mach)):
                    ax.plot(mach, cm.backsolve(fc)[q], "o", ms=7, color=ALT_COLOUR[alt], mec=SURF, mew=1.5)
        ax.axvline(0.80, color=MUTED, lw=1.0)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Mach number")
    axes[1].annotate("M 0.80: the cruise\ncondition used", xy=(0.80, axes[1].get_ylim()[0]),
                     xytext=(-6, 8), textcoords="offset points", ha="right", color=INK2, fontsize=8)
    handles = [Line2D([], [], color=ALT_COLOUR[a], lw=1.8, label=f"Read by hand, {ALT_NAME[a]}")
               for a in ("40K", "20K", "SL")]
    handles.append(Line2D([], [], ls="", marker="o", ms=7, color=INK2, mec=SURF,
                          label="Value implied by CR-2144's own table"))
    fig.legend(handles=handles, loc="lower center", ncol=4)
    fig.suptitle("The hand-read curves pass through the values the source's tables imply",
                 x=0.01, ha="left", color=INK, fontsize=12, fontweight="semibold")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(args.outdir / "s1-curves-against-tables.png")
    plt.close(fig)

    # s2 -- modes before and after
    resid = sd.check()
    d, bare, rows = sd.retest()
    before = cm.errors_vs_ix5(bare)
    speed_only = cm.errors_vs_ix5(rows["FC9 set declared, thrust through the CG (0d84eae)"])
    after = cm.errors_vs_ix5(rows["SHIPPED boeing747, FC9 set + thrust line"])
    names = (("ph_wn", "Slow oscillation (phugoid): frequency"), ("ph_z", "Slow oscillation (phugoid): damping"),
             ("sp_wn", "Fast oscillation (short period): frequency"), ("sp_z", "Fast oscillation (short period): damping"))
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=150)
    y = np.arange(len(names))[::-1]
    for off, vals, color, lab in ((0.26, before, C_BEFORE, "Before session 30"),
                                  (0.0, speed_only, C_SPEED_ONLY, "Speed derivatives only"),
                                  (-0.26, after, C_ATISIM, "Speed derivatives and thrust line (now)")):
        v = [100 * vals[k] for k, _ in names]
        ax.barh(y + off, v, height=0.24, color=color, label=lab)
        for yy, val in zip(y + off, v):
            label = f" {val:+.2f}% " if abs(val) < 1.0 else f" {val:+.1f}% "  # -0.05 is not -0.1
            ax.text(val, yy, label, va="center", ha="left" if val >= 0 else "right", color=INK2, fontsize=8)
    ax.axvline(0, color=BASE, lw=1.0)
    ax.set_yticks(y, [n for _, n in names])
    ax.grid(axis="y", visible=False)
    ax.set_xlim(-24, 20)
    ax.set_xlabel("Error against the value CR-2144 publishes (%)")
    ax.set_title("How closely the simulated 747 oscillates like the published 747")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(args.outdir / "s2-modes-before-after.png")
    plt.close(fig)

    # s3 -- what each source of reading error could do
    _, _, ranges = sd.price(d, resid, args.mc, 20260915)
    order = (("interpolation", "Choice of curve through the points"),
             ("leave-one-out", "Leaving out any one hand-placed point"),
             ("check residuals", "Mismatch seen at the other flight conditions"),
             ("monte carlo 1 px", "Random misplacement of 1 pixel (90% of cases)"),
             ("monte carlo 3 px", "Random misplacement of 3 pixels (90% of cases)"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), dpi=150, sharey=True)
    for ax, (k, title, was) in zip(axes, (("ph_wn", "Phugoid frequency error (%)", before["ph_wn"]),
                                          ("ph_z", "Phugoid damping error (%)", before["ph_z"]))):
        yy = np.arange(len(order))[::-1]
        for yv, (src, _) in zip(yy, order):
            lo, hi = (100 * v for v in ranges[src][k])
            ax.plot([lo, hi], [yv, yv], color=C_ATISIM, lw=7, solid_capstyle="round")
        ax.axvline(100 * after[k], color=C_ATISIM, lw=1.0)
        ax.axvline(0, color=BASE, lw=1.0)
        ax.axvline(100 * was, color=C_BEFORE, lw=2.0)
        ax.annotate(f"before: {100 * was:+.1f}%", xy=(100 * was, len(order) - 0.45), xytext=(4, 0),
                    textcoords="offset points", color=INK2, fontsize=8)
        ax.annotate(f"after: {100 * after[k]:+.1f}%", xy=(100 * after[k], -0.6), xytext=(4, 0),
                    textcoords="offset points", color=INK2, fontsize=8)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", visible=False)
        ax.set_ylim(-0.9, len(order) - 0.2)
    axes[0].set_yticks(np.arange(len(order))[::-1], [lab for _, lab in order])
    fig.suptitle("How far the hand reading could move the answer",
                 x=0.01, ha="left", color=INK, fontsize=12, fontweight="semibold")
    fig.tight_layout()
    fig.savefig(args.outdir / "s3-reading-uncertainty.png")
    plt.close(fig)

    # s4 -- the headline load, before and after
    alt = wind.MEHTA_HANNIBAL_ALTITUDE
    arr = wind.mehta_hannibal_array(alt)
    field = lambda p: wind.vortex_wind(p, arr)  # noqa: E731
    fig, ax = plt.subplots(figsize=(10, 4.4), dpi=150)
    for ac, color, lab in ((sd.BARE, C_BEFORE, "Before session 30"),
                           (sd.AC, C_ATISIM, "Now (speed derivatives and thrust line)")):
        r = fly(ac, field)
        ax.plot(r["north_kft"], r["nz"], color=color,
                label=f"{lab}: {r['ptp']:.2f} g peak to peak, {100 * r['ptp'] / RECORD_PTP:.1f}% of the record")
    for v, txt in ((1.7, "recorded +1.7 g"), (-1.0, "recorded −1.0 g")):
        ax.axhline(v, color=MUTED, lw=1.0)
        ax.annotate(txt, xy=(18.2, v), xytext=(0, 3), textcoords="offset points", ha="right",
                    color=INK2, fontsize=8)
    ax.set_ylim(-1.3, 1.95)
    ax.set_xlim(-18.5, 18.5)
    ax.set_xlabel("Distance along the flight path (thousand feet)")
    ax.set_ylabel("Load factor (g)")
    ax.set_title("The load the 747 feels through the Hannibal vortex field")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(args.outdir / "s4-headline-load.png")
    plt.close(fig)
    print(f"figures -> {args.outdir}")


if __name__ == "__main__":
    main()
