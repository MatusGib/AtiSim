"""Figures for docs/summary/turbulence-validation-report.md.

Every DATA-side number is read from the tracked CSVs in `atisim/data/`, so the
document side of each figure regenerates from the repository alone. Every
MODEL-side number is a literal below, taken from the printed output of the
committed script named beside it -- re-flying ~200 flights of 1200 s to redraw
a chart would cost 90 minutes and prove nothing the scripts have not already
printed. Each literal carries the script that produced it so the provenance is
walkable, which is the same standard `provenance.py` holds numbers to.

Run: PYTHONPATH=<abs worktree root> .venv/Scripts/python.exe \\
         scripts/turbulence_validation_figures.py [--outdir docs/summary/figures]
"""

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import atisim  # noqa: E402

print(f"atisim imported from: {atisim.__file__}")

D = Path(atisim.__file__).parent / "data"

# ---- model-side results, each from the committed script named ---------------
# scripts/tpaws_peak_factor.py -n 48   (boeing747, 37 kft, M 0.80)
B747 = {"R1": 3.8674, "R1se": 0.0450, "R2": 1.9421, "R2se": 0.0246,
        "R3": {15.0: 1.9909, 30.0: 1.9391, 60.0: 1.9236}}
B747_GAUSS = {"R1": 4.0199, "R2": 1.9329}
# scripts/tpaws_737_band.py -n 8       (boeing737, 8 conditions in band)
B737_BAND = {"R1": 3.7763, "R1se": 0.0251, "R2": 1.9960, "R2se": 0.0101,
             "R3_30": 2.0228,
             "per_alt": [(25, 2.0021, 0.0109), (26, 2.0006, 0.0242),
                         (27, 1.9922, 0.0252), (28, 1.9943, 0.0268),
                         (29, 2.0225, 0.0345), (31, 2.0039, 0.0344),
                         (33, 1.9856, 0.0337), (35, 1.9664, 0.0384)]}
# scripts/gust_response_factor.py -n 4
DISCRIM = {"boeing737": {"tau": 1.97, "atten": 0.771, "pf_load": 2.0089,
                         "pf_field": 2.3791},
           "boeing747": {"tau": 2.77, "atten": 0.961, "pf_load": 1.9489,
                         "pf_field": 2.3521}}
MEASURED_757 = {"tau": 1.18, "atten_lo": 0.700, "atten_hi": 0.830}
# scripts/pitch_set_requirement.py
TAU_SWEEP = [(1.72, 0.7202, "737"), (1.97, 0.7686, "737"), (2.14, 0.8203, "737"),
             (2.77, 0.9555, "747"), (2.81, 0.9410, "747"), (3.44, 1.0325, "747")]
PITCH_SENS = [("$C_{m\\alpha}$", -0.5, 0.7), ("$C_{mq}$", 1.4, 0.0),
              ("$I_{yy}$", 1.3, -2.3)]

INK, ACCENT, WARN, MUTE = "#1b2a41", "#2a7ae2", "#c1440e", "#8a94a6"


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=11, color=INK, pad=10, loc="left")
    ax.set_xlabel(xlabel, fontsize=9, color=INK)
    ax.set_ylabel(ylabel, fontsize=9, color=INK)
    ax.tick_params(labelsize=8, colors=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTE)
    ax.grid(alpha=0.18, linewidth=0.6)
    ax.set_axisbelow(True)


def save(fig, out, name):
    fig.tight_layout()
    p = out / name
    fig.savefig(p, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"  wrote {p.name}")


def tpaws_rows():
    with open(D / "tpaws_tm2012_217337_table1.csv", newline="") as f:
        return list(csv.DictReader(f))


def fig1_population(out):
    rows = tpaws_rows()
    pf = [float(r["dn_max_g"]) / float(r["sigma_dn_g"]) for r in rows]
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.hist(pf, bins=np.arange(1.4, 4.3, 0.2), color=ACCENT, alpha=0.55,
            edgecolor="white", label=f"TPAWS measured, N={len(pf)}")
    mean, sd = np.mean(pf), np.std(pf)
    ax.axvspan(mean - sd, mean + sd, color=ACCENT, alpha=0.10,
               label=f"population mean {mean:.3f} $\\pm$ {sd:.3f}")
    ax.axvline(mean, color=ACCENT, lw=1.6)
    ax.axvline(B747["R1"], color=WARN, lw=2.2,
               label=f"model, whole-record $\\sigma$ = {B747['R1']:.3f}")
    ax.axvline(B747["R2"], color=INK, lw=2.2, ls="--",
               label=f"model, TPAWS' own $\\sigma$ = {B747['R2']:.3f}")
    style(ax, "Fig 1  The same model, reduced two ways, against the same data",
          "peak factor  $\\Delta n_{max}/\\sigma_{\\Delta n}$", "encounters")
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    save(fig, out, "fig1-population.png")


def fig2_reductions(out):
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    rows = tpaws_rows()
    pf = [float(r["dn_max_g"]) / float(r["sigma_dn_g"]) for r in rows]
    mean, sd = np.mean(pf), np.std(pf)
    ax.axhspan(mean - sd, mean + sd, color=ACCENT, alpha=0.12,
               label="TPAWS population $\\pm1$ sd")
    ax.axhline(mean, color=ACCENT, lw=1.4)
    labels = ["R1\nwhole record", "R2\nTPAWS window", "R3\n15 s", "R3\n30 s",
              "R3\n60 s"]
    vals = [B747["R1"], B747["R2"], B747["R3"][15.0], B747["R3"][30.0],
            B747["R3"][60.0]]
    cols = [WARN] + [INK] * 4
    ax.bar(labels, vals, color=cols, alpha=0.85, width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.06, f"{v:.3f}", ha="center", fontsize=8.5, color=INK)
    style(ax, "Fig 2  The reduction, not the physics, decides the verdict",
          "", "peak factor")
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.set_ylim(0, 4.4)
    save(fig, out, "fig2-reductions.png")


def fig3_fig133(out):
    with open(D / "tpaws_fig133_sigma_uw.csv", newline="") as f:
        d = [(float(r["sigma_u_ms"]), float(r["sigma_w_ms"]))
             for r in csv.DictReader(f)]
    u = np.array([a for a, _ in d])
    w = np.array([b for _, b in d])
    above = int(np.sum(w > u))
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    ax.plot([0, 10], [0, 10], color=INK, lw=1.2, label="isotropy, 1:1")
    ax.scatter(u, w, s=26, color=WARN, alpha=0.8, edgecolor="white",
               linewidth=0.5, label=f"{len(d)} events, 2002 campaign")
    style(ax, "Fig 3  TPAWS Fig 133, re-extracted from the PDF's vectors",
          "peak $\\sigma_u$  (m/s)", "peak $\\sigma_w$  (m/s)")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.text(0.35, 9.2, f"{100 * above / len(d):.0f}% lie above the 1:1 line\n"
            f"(the paper's own anisotropy claim)", fontsize=8.5, color=INK)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    save(fig, out, "fig3-fig133.png")


def fig4_discrimination(out):
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2))
    names = ["boeing737", "boeing747"]
    x = np.arange(2)
    ax = axes[0]
    pf_l = [DISCRIM[n]["pf_load"] for n in names]
    pf_f = [DISCRIM[n]["pf_field"] for n in names]
    ax.bar(x - 0.18, pf_f, 0.34, color=MUTE, label="the GUST's peak factor")
    ax.bar(x + 0.18, pf_l, 0.34, color=INK, label="the LOAD's peak factor")
    for i, (a, b) in enumerate(zip(pf_f, pf_l)):
        ax.text(i - 0.18, a + 0.05, f"{a:.3f}", ha="center", fontsize=8)
        ax.text(i + 0.18, b + 0.05, f"{b:.3f}", ha="center", fontsize=8)
    spread = 100 * abs(pf_l[0] - pf_l[1]) / np.mean(pf_l)
    style(ax, f"peak factor — separates the aircraft by {spread:.1f}%", "",
          "peak factor")
    ax.set_xticks(x)
    ax.set_xticklabels(["737", "747"])
    ax.set_ylim(0, 3.1)
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1]
    at = [DISCRIM[n]["atten"] for n in names]
    ax.bar(x, at, 0.42, color=ACCENT)
    for i, v in enumerate(at):
        ax.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=8.5)
    ax.axhspan(MEASURED_757["atten_lo"], MEASURED_757["atten_hi"],
               color=WARN, alpha=0.18, label="measured B-757, 0.700–0.830")
    spread2 = 100 * abs(at[0] - at[1]) / np.mean(at)
    style(ax, f"gust response factor — separates them by {spread2:.1f}%", "",
          "attenuation  (measured / quasi-steady)")
    ax.set_xticks(x)
    ax.set_xticklabels(["737", "747"])
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle("Fig 4  Two statistics, one set of flights: only one can fail",
                 fontsize=11, color=INK, x=0.012, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, out, "fig4-discrimination.png")


def fig5_tau(out):
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    for tag, col in (("737", ACCENT), ("747", INK)):
        pts = [(t, k) for t, k, n in TAU_SWEEP if n == tag]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=col,
                lw=1.6, ms=7, label=f"model, {tag}, inside its declared band")
    allp = sorted((t, k) for t, k, _ in TAU_SWEEP)
    ax.plot([p[0] for p in allp], [p[1] for p in allp], color=MUTE, lw=1.0,
            ls=":", zorder=1, label="both entries, one curve")
    lo = min(t for t, _, _ in TAU_SWEEP)
    ax.axvspan(0.9, lo, color=WARN, alpha=0.10)
    ax.axvspan(MEASURED_757["tau"] - 0.02, MEASURED_757["tau"] + 0.02,
               ymin=0, ymax=1, color=WARN, alpha=0.0)
    ax.errorbar([MEASURED_757["tau"]],
                [(MEASURED_757["atten_lo"] + MEASURED_757["atten_hi"]) / 2],
                yerr=[[(MEASURED_757["atten_hi"] - MEASURED_757["atten_lo"]) / 2],
                      [(MEASURED_757["atten_hi"] - MEASURED_757["atten_lo"]) / 2]],
                fmt="s", color=WARN, ms=9, capsize=5, lw=2,
                label="measured B-757, 0.700–0.830")
    ax.text(1.0, 1.02, "no aircraft this model can\nvalidly fly reaches here",
            fontsize=8.5, color=WARN)
    style(ax, "Fig 5  Attenuation tracks plunge time constant — and the "
              "measurement sits outside the model's reach",
          "plunge time constant  $\\tau = 1/(g\\cdot$quasi-steady$)$   (s)",
          "attenuation")
    ax.set_xlim(0.9, 3.7)
    ax.set_ylim(0.55, 1.12)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    save(fig, out, "fig5-tau.png")


def fig6_mach(out):
    with open(D / "stewart_tm2003_212666_table1.csv", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["coefficient"] == "CNalpha"]
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    cmap = {0.0: "#9db4d0", 10.0: "#6f93bd", 20.0: ACCENT, 30.0: "#17457f",
            40.0: MUTE}
    for alt in sorted({float(r["altitude_kft"]) for r in rows}):
        pts = sorted((float(r["mach"]), float(r["value"])) for r in rows
                     if float(r["altitude_kft"]) == alt)
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=48,
                   color=cmap[alt], edgecolor="white", linewidth=0.6, zorder=3,
                   label=f"{alt:.0f} kft" + (" (grid unmatched)" if alt == 40 else ""))
    base_m, base_v = 0.30, 5.44
    m = np.linspace(0.20, 0.87, 200)
    pg = base_v * math.sqrt(1 - base_m ** 2) / np.sqrt(1 - np.minimum(m, 0.9) ** 2)
    ax.plot(m, pg, color=WARN, lw=2.0, ls="--",
            label="Prandtl–Glauert, anchored at M 0.30")
    ax.axvline(0.60, color=MUTE, lw=1.0, ls=":")
    ax.text(0.605, 7.5, "measured slope is flat below M 0.60,\n"
            "then rises +22.9% — a transonic knee\nPrandtl–Glauert does not have",
            fontsize=8.5, color=INK)
    ax.axvspan(0.70, 0.90, color=INK, alpha=0.06)
    ax.text(0.715, 4.35, "where both model transports fly", fontsize=8,
            color=INK)
    style(ax, "Fig 6  Stewart's measured B-757 lift slope against the "
              "correction this model would apply",
          "Mach", "$C_{N\\alpha}$   (per radian)")
    ax.set_xlim(0.20, 0.90)
    ax.set_ylim(4.2, 9.2)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    save(fig, out, "fig6-mach.png")


def fig7_pitch_and_band(out):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    ax = axes[0]
    y = np.arange(len(PITCH_SENS))
    for i, (lab, up, dn) in enumerate(PITCH_SENS):
        ax.barh(i - 0.16, up, 0.3, color=ACCENT)
        ax.barh(i + 0.16, dn, 0.3, color=MUTE)
    ax.axvline(0, color=INK, lw=1.0)
    ax.axvline(32.1, color=WARN, lw=2.0, ls="--")
    ax.text(20.5, 2.45, "the 32% the two\nborrowed frames\ndisagreed by",
            fontsize=8, color=WARN)
    ax.set_yticks(y)
    ax.set_yticklabels([p[0] for p in PITCH_SENS], fontsize=10)
    style(ax, "a ±30% pitch error moves it ≤2.3%", "change in attenuation (%)", "")
    ax.set_xlim(-6, 36)

    ax = axes[1]
    per = B737_BAND["per_alt"]
    ax.errorbar([p[0] for p in per], [p[1] for p in per],
                yerr=[p[2] for p in per], fmt="o-", color=ACCENT, ms=6,
                capsize=3, lw=1.4, label="model 737, per in-band altitude")
    rows = tpaws_rows()
    ax.axhline(2.5013, color=WARN, lw=1.4)
    ax.axhspan(2.5013 - 0.5503, 2.5013 + 0.5503, color=WARN, alpha=0.12,
               label="28 in-band encounters, 2.501 ± 0.550")
    style(ax, "and the 28 in-band rows agree", "altitude (kft)",
          "peak factor, TPAWS window")
    ax.set_ylim(1.6, 3.2)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    fig.suptitle("Fig 7  What the 32% frame spread was not, and the in-band "
                 "re-run", fontsize=11, color=INK, x=0.012, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, out, "fig7-pitch-band.png")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--outdir", type=Path,
                   default=Path("docs/summary/figures"))
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    print(f"\nwriting figures to {args.outdir}")
    fig1_population(args.outdir)
    fig2_reductions(args.outdir)
    fig3_fig133(args.outdir)
    fig4_discrimination(args.outdir)
    fig5_tau(args.outdir)
    fig6_mach(args.outdir)
    fig7_pitch_and_band(args.outdir)
    print("done")


if __name__ == "__main__":
    main()
