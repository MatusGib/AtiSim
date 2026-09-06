"""Generate the presentation figures from LIVE model calls.

Every number plotted here is recomputed from `atisim` at run time -- nothing is
transcribed from a document or from PROJECT.md. Reference values are the
literature values already cited in the repository, and each is labelled with the
table it came from.

Run from the worktree root with PYTHONPATH set to the absolute worktree root:

    PYTHONPATH=<abs worktree root> .venv/Scripts/python.exe presentation_package/scripts/make_figures.py

Writes into presentation_package/figures/.
"""

import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

import atisim
print("atisim imported from:", atisim.__file__)

from atisim import trim
from atisim.validation import longitudinal_modes, lateral_modes

sys.path.insert(0, "atisim/tests")
from test_cr2144_modes import (  # noqa: E402
    AC, V, H, _augmented_longitudinal_modes,
    CR2144_PHUGOID, CR2144_SHORT_PERIOD,
    CR2144_DUTCH_ROLL_WN, CR2144_DUTCH_ROLL_ZETA,
    CR2144_ROLL_TAU, CR2144_SPIRAL_TAU,
)
from test_drag_polar import FIGURE_IX6_40KFT, _sim_cd, _NEAR_FIT_TOL, _OFF_FIT_TOL  # noqa: E402

OUT = "presentation_package/figures"
BLUE, RED, GREY = "#1f77b4", "#d62728", "#888888"


def _trim_point():
    x, _ = trim.trim(jnp.array(V), jnp.array(H), AC)
    return float(x[0]), float(x[1]), float(x[2])


def figure_modes():
    """Figures A and B -- the 747 modes against NASA CR-2144."""
    alpha, elevator, throttle = _trim_point()
    ph, sp = longitudinal_modes(AC, alpha, elevator, throttle, V, H)
    dr, roll_tau, spiral_tau = lateral_modes(AC, alpha, elevator, throttle, V, H)
    aph, asp = _augmented_longitudinal_modes()

    # IMPORTANT: only the four LATERAL modes are asserted against CR-2144
    # (test_lateral_modes_match_cr2144_tables_ix9_and_ix10, rel= as below).
    # The longitudinal tolerances in that file are REGRESSION guards against
    # the sim's own recorded values, not agreement bounds against CR-2144, so
    # plotting them on this axis would misrepresent them. They are shown with
    # no tolerance mark, and the attributed gap is Figure B's subject.
    rows = [
        ("Dutch roll $\\omega_n$", dr[0], CR2144_DUTCH_ROLL_WN, 2.0, "lateral"),
        ("Roll time const.", roll_tau, CR2144_ROLL_TAU, 5.0, "lateral"),
        ("Spiral time const.", spiral_tau, CR2144_SPIRAL_TAU, 2.0, "lateral"),
        ("Dutch roll $\\zeta$", dr[1], CR2144_DUTCH_ROLL_ZETA, 10.0, "lateral"),
        ("Short period $\\omega_n$", sp[0], CR2144_SHORT_PERIOD[0], None, "longitudinal"),
        ("Short period $\\zeta$", sp[1], CR2144_SHORT_PERIOD[1], None, "longitudinal"),
        ("Phugoid $\\zeta$", ph[1], CR2144_PHUGOID[1], None, "longitudinal"),
        ("Phugoid $\\omega_n$", ph[0], CR2144_PHUGOID[0], None, "longitudinal"),
    ]
    labels = [r[0] for r in rows]
    errs = [100.0 * (r[1] - r[2]) / r[2] for r in rows]
    tols = [r[3] for r in rows]
    cols = [BLUE if r[4] == "lateral" else RED for r in rows]

    fig, ax = plt.subplots(figsize=(10.6, 5.8))
    y = np.arange(len(rows))
    ax.barh(y, errs, color=cols, zorder=3, height=0.6)
    for i, t in enumerate(tols):
        if t is None:
            continue
        ax.plot([-t, -t], [i - 0.36, i + 0.36], color=GREY, lw=1.6, zorder=4)
        ax.plot([t, t], [i - 0.36, i + 0.36], color=GREY, lw=1.6, zorder=4)
    for i, e in enumerate(errs):
        ax.text(e + (0.6 if e >= 0 else -0.6), i, "%+.2f%%" % e,
                va="center", ha="left" if e >= 0 else "right", fontsize=9)
    ax.axvline(0, color="k", lw=0.9)
    ax.axhline(3.5, color="k", lw=0.8, ls="--", alpha=0.5)
    ax.text(-23.4, 1.5, "LATERAL\nasserted against\nCR-2144 Table IX-9\n(grey ticks = tolerance)",
            fontsize=8.5, color=BLUE, va="center")
    ax.text(-23.4, 5.5, "LONGITUDINAL\nno CR-2144 tolerance\nasserted - the gap is\nattributed, see Fig. B",
            fontsize=8.5, color=RED, va="center")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("signed error vs NASA CR-2144 (%)")
    ax.set_title("747 flight-dynamics modes vs NASA CR-2144 Tables IX-5 / IX-9\n"
                 "FC9: 40,000 ft, M 0.80.  Simulator output, recomputed live.", fontsize=11)
    ax.grid(axis="x", alpha=0.3, zorder=0)
    ax.set_xlim(-26, 20)
    fig.tight_layout()
    fig.savefig(OUT + "/fig-A-747-modes-vs-NASA-CR-2144.png", dpi=150)
    plt.close(fig)
    print("wrote fig-A-747-modes-vs-NASA-CR-2144.png")

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    names = ["Phugoid $\\omega_n$", "Phugoid $\\zeta$",
             "Short period $\\omega_n$", "Short period $\\zeta$"]
    sim = [ph[0], ph[1], sp[0], sp[1]]
    aug = [aph[0], aph[1], asp[0], asp[1]]
    ref = [CR2144_PHUGOID[0], CR2144_PHUGOID[1],
           CR2144_SHORT_PERIOD[0], CR2144_SHORT_PERIOD[1]]
    se = [100.0 * (s - r) / r for s, r in zip(sim, ref)]
    ae = [100.0 * (a - r) / r for a, r in zip(aug, ref)]
    x = np.arange(4)
    w = 0.36
    ax.bar(x - w / 2, se, w, color=RED, zorder=3,
           label="the simulator (aero form is $\\alpha$, q, $\\delta_e$ only)")
    ax.bar(x + w / 2, ae, w, color=BLUE, zorder=3,
           label="augmented textbook model (CR-2144 Table IX-4 full set)")
    for xi, v in zip(x - w / 2, se):
        ax.text(xi, v + (0.7 if v >= 0 else -1.8), "%+.1f%%" % v, ha="center", fontsize=9)
    for xi, v in zip(x + w / 2, ae):
        ax.text(xi, v + (0.7 if v >= 0 else -1.8), "%+.2f%%" % v, ha="center", fontsize=9)
    ax.axhline(0, color="k", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("signed error vs CR-2144 Table IX-5 (%)")
    ax.set_title("Where the longitudinal gap comes from, and what closes it\n"
                 "Restoring CR-2144's own excluded $X_u$, $Z_u$, $M_u$, $Z_{\\dot w}$, $M_{\\dot w}$ "
                 "closes all four to under 1.2%.", fontsize=11)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_ylim(-24, 20)
    fig.tight_layout()
    fig.savefig(OUT + "/fig-B-phugoid-gap-attribution.png", dpi=150)
    plt.close(fig)
    print("wrote fig-B-phugoid-gap-attribution.png")


def figure_drag():
    """Figure C -- the 747 drag polar against CR-2144 Figure IX-6."""
    machs = sorted(FIGURE_IX6_40KFT)
    figure_cd = [FIGURE_IX6_40KFT[m] for m in machs]
    sim_cd = [_sim_cd(m) for m in machs]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.0, 7.0), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(machs, figure_cd, "o--", color=GREY,
             label="NASA CR-2144 Fig. IX-6, 40,000 ft (digitised)")
    ax1.plot(machs, sim_cd, "s-", color=BLUE, label="AtiSim $C_D$ at trim (live)")
    ax1.axvspan(0.75, 0.88, color=BLUE, alpha=0.08)
    ax1.text(0.815, max(figure_cd) * 0.95, "near-fit band", ha="center", fontsize=9, color=BLUE)
    ax1.axvline(0.80, color=RED, ls=":", lw=1.4)
    ax1.text(0.806, max(figure_cd) * 0.78,
             "M 0.80 - the single point\n$C_{D0}$ and $e$ were back-solved\nfrom. Agreement here is circular.",
             fontsize=8, color=RED, va="top")
    ax1.set_ylabel("$C_D$")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.set_title("747 drag polar off its single fitted point\n"
                  "Nine points digitised from CR-2144 Figure IX-6's 40,000 ft curve", fontsize=11)

    resid = [s - f for s, f in zip(sim_cd, figure_cd)]
    cols = [BLUE if 0.75 <= m <= 0.88 else "#ff7f0e" for m in machs]
    ax2.bar(machs, resid, width=0.012, color=cols, zorder=3)
    ax2.axhline(0, color="k", lw=0.9)
    for tol, c in ((_NEAR_FIT_TOL, BLUE), (_OFF_FIT_TOL, "#ff7f0e")):
        ax2.axhline(tol, color=c, ls="--", lw=1.0, alpha=0.75)
        ax2.axhline(-tol, color=c, ls="--", lw=1.0, alpha=0.75)
    ax2.text(0.884, _OFF_FIT_TOL * 0.60, "off-fit tolerance $\\pm$0.020", fontsize=8, color="#ff7f0e")
    ax2.text(0.700, _NEAR_FIT_TOL * 1.30, "near-fit tolerance $\\pm$0.004", fontsize=8, color=BLUE)
    ax2.set_xlabel("Mach")
    ax2.set_ylabel("sim $-$ figure")
    ax2.grid(axis="y", alpha=0.3, zorder=0)
    fig.tight_layout()
    fig.savefig(OUT + "/fig-C-747-drag-polar-vs-CR-2144-IX6.png", dpi=150)
    plt.close(fig)
    print("wrote fig-C-747-drag-polar-vs-CR-2144-IX6.png")


if __name__ == "__main__":
    figure_modes()
    figure_drag()
    print("done")
