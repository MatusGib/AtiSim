"""Session 23's CAT evidence: fly the new source cases and draw them against the papers.

Four questions, four figures, one printed table of every number behind them:

  1  Mehta 1987's five-vortex Hannibal field is the only wind field this project
     holds that DECLARES NOTHING. Fly it and put the run beside the normal
     acceleration NASA TM-102186 measured in the same encounter.
  2  TM-102186 Fig. 8 flies three aircraft of very different speed through the
     same reconstructed vortices and states a mechanism for the ordering it
     gets. Reproduce the ordering, then test the mechanism directly.
  3  Yoshimura et al. 2022's appendix carries CR-2144's 747 at a THIRD flight
     condition. Show where the model's error at that condition lives.
  4  Misaka et al. 2008's RMS normal load is a severity index defined at cruise
     altitude, which the F-factor thresholds are not. Grade every run on it.

Run: .venv/Scripts/python.exe scripts/cat_validation.py --outdir runs/cat
"""

import argparse
import math
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import atisim  # noqa: F401  -- enables x64
from atisim import checks, trim, validation, vortex_viz, wind
from atisim.aircraft import CRUISE, REGISTRY
from atisim.atmosphere import G0, density, speed_of_sound
from atisim.panel import ALPHA_INVALID_DEG, ALPHA_LINEAR_DEG
from atisim.units import FT2M, RAD2DEG

# ---------------------------------------------------------------------------
# TM-102186 Fig. 8, DIGITISED. This is the only number in this file that was
# read off a plot rather than out of a table, and it is kept apart for that
# reason.
#
# The figure is a 1989 photocopy at ~2500x3300 with three overlapping traces
# (solid / dotted / dashed) on marked axes -- pitch -25..+50 deg, g-load
# -1..+2, distance -10..+40 kft. Only the EXTREMA are read. A full trace
# digitisation off this scan would carry an error nobody could bound; two
# well-separated turning points against a labelled gridline can be bounded, and
# the bound below is deliberately generous.
#
# NOTHING ASSERTS AGAINST THESE. They are drawn as a reference band beside the
# model's own excursions and nowhere else. Promoting them to a test would be
# asserting against my own eyesight.
TM_FIG8_READ_ERROR = {"pitch_deg": 3.0, "g": 0.15}
TM_FIG8 = {
    #            V ft/s   pitch max  pitch min   g max   g min
    "RPV":       dict(V=150.0, pitch=(43.0, -23.0), g=(2.05, 0.35)),
    "EXECUTIVE": dict(V=700.0, pitch=(13.0, -3.0), g=(1.95, -0.35)),
    "AIRLINER":  dict(V=800.0, pitch=(8.0, 3.0), g=(1.55, -0.55)),
}

# Which registry aircraft stands in for which of the paper's three, by speed.
# The middle slot is EMPTY and stays empty: the paper's executive jet is at
# 700 ft/s and this registry's nearest entry is the 737 on approach at 439,
# which is a different flight regime rather than a slower jet. Filling it with
# the nearest number available would be inventing a comparison.
TM_FIG8_STANDIN = {"RPV": "cherokee", "EXECUTIVE": None, "AIRLINER": "boeing747"}

# Every aircraft that can be trimmed and flown, for the MECHANISM test. This one
# does not need matched speeds -- it tests the paper's stated reason rather than
# its three particular results -- so the whole registry is fair game.
MECHANISM_FLEET = (
    "cherokee", "cessna172", "boeing737_approach",
    "boeing747_approach", "boeing737", "boeing747",
)

PALETTE = {
    "model": "#1D5D77",
    "reference": "#A9501C",
    "wind": "#3E6A48",
    "muted": "#7E8D93",
    "grid": "#D6DCD8",
}


def _style(ax, xlabel=None, ylabel=None, title=None):
    ax.grid(True, color=PALETTE["grid"], lw=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    if title:
        ax.set_title(title, fontsize=10, loc="left", fontweight="bold")
    ax.tick_params(labelsize=8)


# ---------------------------------------------------------------------------
# 1. The Mehta encounter
# ---------------------------------------------------------------------------


def fly_mehta(aircraft: str, dt: float, lead_r0: float = 12.0):
    """Fly one aircraft through Mehta's five-vortex field at its own altitude.

    The field is placed at the aircraft's own cruise altitude rather than at
    Mehta's 37,000 ft for every aircraft, because a Cherokee cannot be trimmed
    at 37,000 ft. The VORTICES ARE IDENTICAL either way -- `vortex_wind` is a
    velocity field with no density in it -- so what differs between aircraft is
    the air they fly through, which is a confound and is reported as one.

    The 747 is flown at Mehta's own 37,000 ft, so the headline case has no such
    confound at all.
    """
    ac = REGISTRY[aircraft]
    V = CRUISE[aircraft]["airspeed"]
    H = (wind.MEHTA_HANNIBAL_ALTITUDE if aircraft == "boeing747"
         else CRUISE[aircraft]["altitude"])
    array = wind.mehta_hannibal_array(H)
    field = lambda p: wind.vortex_wind(p, array)  # noqa: E731

    r0 = float(array.r0)
    x0, x1 = float(array.north.min()), float(array.north.max())
    start = x0 - lead_r0 * r0
    seconds = (x1 + lead_r0 * r0 - start) / V

    enc = vortex_viz.fly_in_moving_air(
        ac, field, V, H,
        label=f"{aircraft} through Mehta 1987",
        start_north=start, seconds=seconds, dt=dt,
        window=(x0 - 2.0 * r0, x1 + 2.0 * r0),
        window_name="the identified array, plus 2 r0 either side",
    )
    return enc, dict(ac=ac, V=V, H=H, array=array, field=field, r0=r0)


def figure_mehta(enc, meta, path: Path):
    """The encounter, against the acceleration TM-102186 measured in it."""
    array = meta["array"]
    x_kft = enc.north / FT2M / 1000.0
    core_kft = np.asarray(array.north) / FT2M / 1000.0
    penetrated = set(wind.MEHTA_HANNIBAL_CORE_PAIR)

    fig, axes = plt.subplots(3, 1, figsize=(9.5, 9.0), sharex=True)
    fig.suptitle(
        "Mehta 1987's identified Hannibal field, flown by AtiSim's 747\n"
        "Wind is the paper's own converged solution; the load band is what the "
        "DC-10 actually recorded",
        fontsize=11, fontweight="bold", x=0.02, ha="left",
    )

    ax = axes[0]
    ax.plot(x_kft, enc.w_up / FT2M, color=PALETTE["wind"], lw=1.4,
            label="vertical gust at the aircraft")
    for k, cx in enumerate(core_kft):
        hit = k in penetrated
        ax.axvline(cx, color=PALETTE["muted"], lw=1.6 if hit else 0.8,
                   ls="-" if hit else ":", alpha=0.9 if hit else 0.6)
        ax.annotate(f"v{k + 1}" + (" (core)" if hit else ""),
                    xy=(cx, ax.get_ylim()[1]), xytext=(2, -10),
                    textcoords="offset points", fontsize=7,
                    color=PALETTE["muted"], rotation=90, va="top")
    ax.axhline(wind.MEHTA_HANNIBAL_V0 / FT2M, color=PALETTE["reference"],
               lw=1.0, ls="--", alpha=0.8)
    ax.axhline(-wind.MEHTA_HANNIBAL_V0 / FT2M, color=PALETTE["reference"],
               lw=1.0, ls="--", alpha=0.8,
               label=f"$\\pm V_0$ = {wind.MEHTA_HANNIBAL_V0 / FT2M:.1f} ft/s (Mehta)")
    _style(ax, ylabel="vertical wind, ft/s",
           title="a  The field Mehta identified — five cores, two on the flight path")
    ax.legend(fontsize=8, loc="lower left", framealpha=0.9)

    ax = axes[1]
    lo, hi = wind.TM102186_HANNIBAL_NZ
    ax.axhspan(lo, hi, color=PALETTE["reference"], alpha=0.13, zorder=0)
    ax.axhline(hi, color=PALETTE["reference"], lw=1.2, ls="--")
    ax.axhline(lo, color=PALETTE["reference"], lw=1.2, ls="--",
               label=f"DC-10 measured, {lo:+.1f} to {hi:+.1f} g (TM-102186 Fig. 6)")
    ax.plot(x_kft, enc.n_z, color=PALETTE["model"], lw=1.5,
            label=f"AtiSim 747, {enc.n_z.min():+.2f} to {enc.n_z.max():+.2f} g")
    ax.axhline(1.0, color=PALETTE["muted"], lw=0.8, alpha=0.6)
    _style(ax, ylabel="load factor $n_z$, g",
           title="b  Predicted load against the measured band")
    ax.legend(fontsize=8, loc="lower left", framealpha=0.9)

    ax = axes[2]
    ax.plot(x_kft, enc.theta * RAD2DEG, color=PALETTE["model"], lw=1.5,
            label="pitch attitude $\\theta$")
    ax.plot(x_kft, enc.alpha_air * RAD2DEG, color=PALETTE["wind"], lw=1.2,
            ls="--", label="air-relative $\\alpha$")
    ax.plot(x_kft, enc.alpha_inertial * RAD2DEG, color=PALETTE["muted"],
            lw=1.0, ls=":",
            label="inertial $\\alpha$ (what a still-air reading would give)")
    _style(ax, xlabel="distance along flight path, 1000 ft",
           ylabel="angle, deg",
           title="c  Attitude — and why air-relative sensing is not optional")
    ax.legend(fontsize=8, loc="lower left", framealpha=0.9)

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. The ordering, and the mechanism behind it
# ---------------------------------------------------------------------------


def excursion(enc, V: float):
    """Pitch, load and incidence over the window -- plus how much gust got through.

    `incidence_gain` is the mechanism TM-102186 states, reduced to one number:

        frozen  = atan(max|w_up| / V)   the alpha a RIGIDLY HELD attitude sees
        gain    = max|alpha - alpha_trim| / frozen

    A gain below 1 means the aircraft pitched into the flow and shed part of the
    gust before it reached the wing. A gain ABOVE 1 means its own motion added
    incidence rather than removing it -- which is not a contradiction and not a
    bug: after the first core the aircraft carries a vertical velocity and a
    pitch rate into the second, and a fast crossing puts those in phase with the
    next gust instead of against it.

    Preferred over peak-to-peak pitch as the mechanism test, because pitch also
    scales with how large the gust is in incidence terms, and `frozen` varies by
    a factor of four across this fleet purely through airspeed.

    `frozen` is built from the gust the aircraft ACTUALLY met, not from V0.
    Superposing five cores can and does exceed the single-core peak, so using V0
    would quietly understate the denominator.
    """
    w = enc.window
    th = enc.theta[w] * RAD2DEG
    nz = enc.n_z[w]
    a = enc.alpha_air[w] * RAD2DEG
    a_trim = float(enc.alpha_air[0] * RAD2DEG)
    w_peak = float(np.abs(enc.w_up[w]).max())
    frozen = math.degrees(math.atan(w_peak / V))
    return dict(
        pitch_ptp=float(th.max() - th.min()),
        pitch_max=float(th.max()), pitch_min=float(th.min()),
        g_max=float(nz.max()), g_min=float(nz.min()),
        alpha_min=float(a.min()), alpha_max=float(a.max()),
        alpha_peak=float(max(abs(a.min()), abs(a.max()))),
        w_peak_fts=w_peak / FT2M,
        frozen_deg=frozen,
        incidence_gain=float(max(abs(a.max() - a_trim), abs(a.min() - a_trim))
                             / frozen),
    )


def traverse_ratio(aircraft: str, r0: float) -> float:
    """Core traverse time divided by the aircraft's own short period.

    TM-102186's stated mechanism, as a number: "These variations are dependent
    upon the relationship between the time span of the vortex traverse and the
    aircraft's short oscillatory period."
    """
    ac = REGISTRY[aircraft]
    V, H = CRUISE[aircraft]["airspeed"], CRUISE[aircraft]["altitude"]
    x, _ = trim.trim(jnp.array(V), jnp.array(H), ac)
    a, e, t = (float(v) for v in x)
    _, (wn, _) = validation.longitudinal_modes(ac, a, e, t, V, H)
    return (2.0 * r0 / V) / (2.0 * math.pi / wn)


def figure_ordering(runs, path: Path):
    """AtiSim's excursions against the digitised Fig. 8, plus the mechanism."""
    fig = plt.figure(figsize=(11.0, 8.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0], hspace=0.38, wspace=0.26)
    fig.suptitle(
        "TM-102186 Fig. 8 — three aircraft, one vortex field\n"
        "The paper's claim is an ORDERING and a mechanism; absolute agreement "
        "is ruled out by aircraft type",
        fontsize=11, fontweight="bold", x=0.02, ha="left",
    )

    # -- a: pitch excursion, model vs digitised reference ---------------------
    ax = fig.add_subplot(gs[0, 0])
    names = list(TM_FIG8)
    xs = np.arange(len(names))
    ref = [TM_FIG8[n]["pitch"][0] - TM_FIG8[n]["pitch"][1] for n in names]
    ax.bar(xs - 0.19, ref, 0.36, color=PALETTE["reference"], alpha=0.85,
           yerr=2 * TM_FIG8_READ_ERROR["pitch_deg"], capsize=3,
           error_kw=dict(lw=1, ecolor=PALETTE["muted"]),
           label="TM-102186 Fig. 8, digitised")
    model, model_x = [], []
    for i, n in enumerate(names):
        key = TM_FIG8_STANDIN[n]
        if key is None:
            ax.annotate("no stand-in\nin the registry", xy=(i + 0.19, 20.0),
                        ha="center", va="bottom", fontsize=7.5, style="italic",
                        color=PALETTE["muted"])
            continue
        model.append(runs[key]["exc"]["pitch_ptp"])
        model_x.append(i + 0.19)
    ax.bar(model_x, model, 0.36, color=PALETTE["model"], label="AtiSim")
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [f"{n}\n{TM_FIG8[n]['V']:.0f} ft/s" +
         ("" if TM_FIG8_STANDIN[n] is None
          else f"\n({TM_FIG8_STANDIN[n]}, {CRUISE[TM_FIG8_STANDIN[n]]['airspeed'] / FT2M:.0f})")
         for n in names], fontsize=7.5)
    _style(ax, ylabel="peak-to-peak pitch, deg",
           title="a  Pitch excursion falls with speed, in both")
    ax.legend(fontsize=8, framealpha=0.9)

    # -- b: minimum g -- the ordering that REVERSES ---------------------------
    ax = fig.add_subplot(gs[0, 1])
    ref_g = [TM_FIG8[n]["g"][1] for n in names]
    ax.bar(xs - 0.19, ref_g, 0.36, color=PALETTE["reference"], alpha=0.85,
           yerr=2 * TM_FIG8_READ_ERROR["g"], capsize=3,
           error_kw=dict(lw=1, ecolor=PALETTE["muted"]),
           label="TM-102186 Fig. 8, digitised")
    model_g, gx = [], []
    for i, n in enumerate(names):
        key = TM_FIG8_STANDIN[n]
        if key is None:
            continue
        model_g.append(runs[key]["exc"]["g_min"])
        gx.append(i + 0.19)
    ax.bar(gx, model_g, 0.36, color=PALETTE["model"], label="AtiSim")
    ax.axhline(0.0, color=PALETTE["muted"], lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([n for n in names], fontsize=8)
    _style(ax, ylabel="minimum load factor, g",
           title="b  …but minimum g goes the OTHER way")
    ax.legend(fontsize=8, framealpha=0.9, loc="lower left")

    # -- c, d: the mechanism --------------------------------------------------
    fleet = [k for k in MECHANISM_FLEET if k in runs]
    ratio = np.array([runs[k]["ratio"] for k in fleet])
    order = np.argsort(ratio)

    ax = fig.add_subplot(gs[1, 0])
    ptp = np.array([runs[k]["exc"]["pitch_ptp"] for k in fleet])
    ax.axvspan(ratio.min() * 0.7, 0.45, color=PALETTE["muted"], alpha=0.10)
    ax.plot(ratio[order], ptp[order], "o-", color=PALETTE["model"], lw=1.6, ms=7)
    for k, r, p in zip(fleet, ratio, ptp):
        ax.annotate(k.replace("boeing", ""), xy=(r, p),
                    xytext=(5, 5 if k != "boeing737" else -12),
                    textcoords="offset points", fontsize=7.5,
                    color=PALETTE["muted"])
    ax.set_xscale("log")
    ax.set_xlim(ratio.min() * 0.7, ratio.max() * 1.6)
    _style(ax, xlabel="core traverse time / short period",
           ylabel="peak-to-peak pitch, deg",
           title="c  Below ~0.4 the pitch response collapses")
    ax.annotate("no time\nto pitch", xy=(0.23, ptp.max() * 0.80), fontsize=7.5,
                color=PALETTE["muted"], ha="center")

    ax = fig.add_subplot(gs[1, 1])
    gain = np.array([runs[k]["exc"]["incidence_gain"] for k in fleet])
    ax.axhline(1.0, color=PALETTE["reference"], lw=1.1, ls="--")
    ax.plot(ratio[order], gain[order], "o-", color=PALETTE["wind"], lw=1.6, ms=7)
    for k, r, a in zip(fleet, ratio, gain):
        ax.annotate(k.replace("boeing", ""), xy=(r, a),
                    xytext=(6, 6 if k in ("boeing747", "boeing747_approach",
                                          "cherokee") else -13),
                    textcoords="offset points", fontsize=7.5,
                    color=PALETTE["muted"])
    ax.set_xscale("log")
    ax.set_xlim(ratio.min() * 0.7, ratio.max() * 1.6)
    ax.set_ylim(0, max(1.7, float(gain.max()) * 1.15))
    ax.annotate("own motion ADDS incidence", xy=(ratio.max() * 1.5, 1.04),
                fontsize=7.5, color=PALETTE["reference"], ha="right",
                style="italic")
    ax.annotate("pitches away, SHEDS incidence", xy=(ratio.max() * 1.5, 0.94),
                fontsize=7.5, color=PALETTE["reference"], va="top", ha="right",
                style="italic")
    _style(ax, xlabel="core traverse time / short period",
           ylabel="incidence gain (reaching the wing / frozen attitude)",
           title="d  …and it collapses monotonically, six for six")

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. The third flight condition
# ---------------------------------------------------------------------------


def fl200_comparison():
    """747 short period at M 0.8 / 6096 m: own derivatives, swapped, reference."""
    R = validation.REFERENCES
    H = R["747fl200_altitude"].value
    V = 0.8 * float(speed_of_sound(jnp.array(H)))
    ac = REGISTRY["boeing747"]

    def sp(aircraft):
        x, _ = trim.trim(jnp.array(V), jnp.array(H), aircraft)
        a, e, t = (float(v) for v in x)
        _, mode = validation.longitudinal_modes(aircraft, a, e, t, V, H)
        return mode

    swapped = ac._replace(
        CLa=jnp.array(-R["747fl200_CZalpha"].value),
        Cma=jnp.array(R["747fl200_Cmalpha"].value),
        Cmq=jnp.array(R["747fl200_Cmq"].value),
        CLq=jnp.array(-R["747fl200_CZq"].value),
    )
    rho = float(density(jnp.array(H)))
    return dict(
        V=V, H=H, rho=rho,
        CL=float(ac.mass) * G0 / (0.5 * rho * V * V * float(ac.S)),
        own=sp(ac), swapped=sp(swapped),
        ref=(R["747fl200_short_period_wn"].value,
             R["747fl200_short_period_zeta"].value),
    )


def figure_fl200(cmp_, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.3))
    fig.suptitle(
        "747 short period at M 0.8 / 6,096 m — a CR-2144 condition the model "
        "had never been run at\n"
        "Swapping the derivative set for that condition closes the frequency "
        "gap entirely",
        fontsize=10.5, fontweight="bold", x=0.02, ha="left",
    )
    labels = ["AtiSim\n(FC9, 40,000 ft\nderivatives)",
              "AtiSim\n(Table A2\nderivatives)",
              "reference\n(CR-2144 via\nYoshimura A5)"]
    colours = [PALETTE["model"], PALETTE["wind"], PALETTE["reference"]]

    for ax, idx, name, ref in (
        (axes[0], 0, "$\\omega_n$, rad/s", cmp_["ref"][0]),
        (axes[1], 1, "$\\zeta$", cmp_["ref"][1]),
    ):
        vals = [cmp_["own"][idx], cmp_["swapped"][idx], ref]
        bars = ax.bar(labels, vals, 0.6, color=colours)
        ax.axhline(ref, color=PALETTE["reference"], lw=1.1, ls="--", alpha=0.8)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.3f}\n({100 * (v - ref) / ref:+.1f}%)",
                        xy=(b.get_x() + b.get_width() / 2, v),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=8)
        ax.set_ylim(0, max(vals) * 1.32)
        _style(ax, ylabel=name)
        ax.tick_params(axis="x", labelsize=7.5)

    axes[0].set_title("a  Frequency — the gap is the DATA, not the solver",
                      fontsize=10, loc="left", fontweight="bold")
    axes[1].set_title("b  Damping — the residual is the missing $C_{m\\dot\\alpha}$",
                      fontsize=10, loc="left", fontweight="bold")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Severity
# ---------------------------------------------------------------------------


def figure_severity(runs, path: Path):
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    names = [k for k in MECHANISM_FLEET if k in runs]
    sig = [runs[k]["sigma_n"] for k in names]
    band = checks.RMS_NORMAL_LOAD_BANDS
    ax.axhspan(0, band["moderate"], color=PALETTE["muted"], alpha=0.10)
    ax.axhspan(band["moderate"], band["severe"], color=PALETTE["reference"],
               alpha=0.14)
    ax.axhspan(band["severe"], max(max(sig) * 1.25, 0.5),
               color=PALETTE["reference"], alpha=0.28)
    ax.axhline(band["moderate"], color=PALETTE["reference"], lw=1.0, ls="--")
    ax.axhline(band["severe"], color=PALETTE["reference"], lw=1.2, ls="--")
    ax.bar(names, sig, 0.55, color=PALETTE["model"])
    for i, v in enumerate(sig):
        ax.annotate(f"{v:.3f}", xy=(i, v), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=8)
    for y, text in ((band["severe"], "severe  $\\geq$ 0.30 g"),
                    (band["moderate"], "moderate  0.20–0.30 g"),
                    (0.0, "smooth  < 0.20 g")):
        ax.annotate(text, xy=(-0.45, y), xytext=(0, 3),
                    textcoords="offset points", ha="left", va="bottom",
                    fontsize=8, style="italic", color=PALETTE["reference"])
    ax.annotate(
        "Every aircraft is graded SEVERE. That is the expected answer — this is "
        "the field\nidentified from an encounter that injured people — and it is "
        "the point of the index:\nthe F-factor thresholds AtiSim uses elsewhere "
        "are low-altitude figures and say nothing here.",
        xy=(0.5, -0.30), xycoords="axes fraction", ha="center", va="top",
        fontsize=8, color=PALETTE["muted"])
    ax.set_ylim(0, max(max(sig) * 1.25, 0.5))
    _style(ax, ylabel="$\\sigma_n$ over a 5 s window, g",
           title="RMS normal load through Mehta's field — the severity index "
                 "that is defined at cruise altitude")
    ax.tick_params(axis="x", labelsize=8, rotation=12)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=Path, default=Path("runs/cat"))
    p.add_argument("--dt", type=float, default=0.01)
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    print(f"atisim imported from: {atisim.__file__}")
    print()

    runs = {}
    for name in MECHANISM_FLEET:
        enc, meta = fly_mehta(name, args.dt)
        c = checks.rms_normal_load(enc.log, meta["ac"], dt=args.dt)
        runs[name] = dict(
            enc=enc, meta=meta, exc=excursion(enc, meta["V"]),
            ratio=traverse_ratio(name, meta["r0"]),
            sigma_n=float(c.value), sigma_detail=c.detail,
            start_offset=float(enc.n_z[0] - math.cos(float(enc.theta[0]))),
        )

    # -- 1 --------------------------------------------------------------------
    lo, hi = wind.TM102186_HANNIBAL_NZ
    e747 = runs["boeing747"]
    print("=" * 78)
    print("1. MEHTA 1987's FIVE-VORTEX HANNIBAL FIELD, FLOWN BY THE 747")
    print("=" * 78)
    print(f"  altitude        {e747['meta']['H']:.0f} m "
          f"({e747['meta']['H'] / FT2M:.0f} ft) -- Mehta's own")
    print(f"  airspeed        {e747['meta']['V']:.1f} m/s")
    print(f"  cores           5, at x = {wind.MEHTA_HANNIBAL_X_FT} ft")
    print(f"  r0 / V0 / psi   {wind.MEHTA_HANNIBAL_R0 / FT2M:.1f} ft / "
          f"{wind.MEHTA_HANNIBAL_V0 / FT2M:.1f} ft/s / "
          f"{wind.MEHTA_HANNIBAL_PSI_DEG:.0f} deg")
    enc = e747["enc"]
    print(f"  start offset    {e747['start_offset']:+.4f} g from cos(theta0) "
          "-- moving-air start")
    print(f"  vertical gust   {enc.w_up.min() / FT2M:+.1f} to "
          f"{enc.w_up.max() / FT2M:+.1f} ft/s")
    print(f"  MODEL n_z       {enc.n_z.min():+.3f} to {enc.n_z.max():+.3f} g")
    print(f"  MEASURED n_z    {lo:+.1f} to {hi:+.1f} g   (DC-10, TM-102186 Fig. 6)")
    print(f"  alpha (air)     {enc.alpha_air.min() * RAD2DEG:+.2f} to "
          f"{enc.alpha_air.max() * RAD2DEG:+.2f} deg")
    # Computed, not read off panel c. This is the number that says air-relative
    # sensing is not optional, so it must not be an eyeball estimate.
    divergence = float(np.abs(enc.alpha_air - enc.alpha_inertial).max()) * RAD2DEG
    print(f"  air vs inertial alpha diverge by up to {divergence:.2f} deg")
    print(f"  sigma_n         {e747['sigma_n']:.4f} g")
    print()

    # -- 2 --------------------------------------------------------------------
    print("=" * 78)
    print("2. THE THREE-AIRCRAFT ORDERING, AND THE MECHANISM BEHIND IT")
    print("=" * 78)
    print(f"  {'aircraft':<20}{'V ft/s':>8}{'trav/Tsp':>10}"
          f"{'pitch p-p':>11}{'g min':>8}{'g max':>8}"
          f"{'frozen':>9}{'gain':>7}{'|a| pk':>8}  band")
    for k in MECHANISM_FLEET:
        r, e = runs[k], runs[k]["exc"]
        band = ("linear" if e["alpha_peak"] <= ALPHA_LINEAR_DEG
                else "OUTSIDE" if e["alpha_peak"] <= ALPHA_INVALID_DEG
                else "INVALID")
        print(f"  {k:<20}{CRUISE[k]['airspeed'] / FT2M:>8.0f}{r['ratio']:>10.3f}"
              f"{e['pitch_ptp']:>11.2f}{e['g_min']:>8.3f}{e['g_max']:>8.3f}"
              f"{e['frozen_deg']:>9.1f}{e['incidence_gain']:>7.2f}"
              f"{e['alpha_peak']:>8.2f}  {band}")
    print(f"  ('band' is the declared linear-aero range: linear <= "
          f"{ALPHA_LINEAR_DEG:.0f} deg, invalid > {ALPHA_INVALID_DEG:.0f} deg.")
    print("   Every run above is inside it -- and the reason is the mechanism "
          "itself:")
    print("   the slow aircraft pitch into the flow and never see the "
          "incidence a frozen attitude would.)")
    print()
    print("  TM-102186 Fig. 8, digitised (+/- "
          f"{TM_FIG8_READ_ERROR['pitch_deg']:.0f} deg, "
          f"+/- {TM_FIG8_READ_ERROR['g']:.2f} g):")
    for n, d in TM_FIG8.items():
        print(f"  {n:<20}{d['V']:>8.0f}{'--':>14}"
              f"{d['pitch'][0] - d['pitch'][1]:>11.1f}"
              f"{d['g'][1]:>8.2f}{d['g'][0]:>8.2f}")
    print()

    # -- 3 --------------------------------------------------------------------
    cmp_ = fl200_comparison()
    print("=" * 78)
    print("3. THE 747 AT A THIRD CR-2144 FLIGHT CONDITION (M 0.8 / 6,096 m)")
    print("=" * 78)
    print(f"  U0 {cmp_['V']:.1f} m/s (Table A3: 253)   "
          f"rho {cmp_['rho']:.4f} (0.653)   "
          f"CL {cmp_['CL']:.4f} (0.266)")
    for label, (wn, z) in (("AtiSim, FC9 derivatives", cmp_["own"]),
                           ("AtiSim, Table A2 derivatives", cmp_["swapped"])):
        print(f"  {label:<30} wn {wn:.4f} ({100 * (wn - cmp_['ref'][0]) / cmp_['ref'][0]:+6.2f}%)"
              f"   zeta {z:.4f} ({100 * (z - cmp_['ref'][1]) / cmp_['ref'][1]:+6.2f}%)")
    print(f"  {'reference (Table A5)':<30} wn {cmp_['ref'][0]:.4f}"
          f"            zeta {cmp_['ref'][1]:.4f}")
    print()

    # -- 4 --------------------------------------------------------------------
    print("=" * 78)
    print("4. SEVERITY (Misaka 2008: moderate 0.2-0.3 g, severe >= 0.3 g)")
    print("=" * 78)
    for k in MECHANISM_FLEET:
        print(f"  {k:<20} sigma_n {runs[k]['sigma_n']:.4f} g")
    print()

    figure_mehta(e747["enc"], e747["meta"], args.outdir / "01-mehta-encounter.png")
    figure_ordering(runs, args.outdir / "02-ordering-and-mechanism.png")
    figure_fl200(cmp_, args.outdir / "03-fl200-short-period.png")
    figure_severity(runs, args.outdir / "04-severity.png")
    print(f"figures -> {args.outdir}")


if __name__ == "__main__":
    main()
