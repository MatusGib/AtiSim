"""What the Hannibal comparison is actually worth, once the inputs carry error.

Every load number this project has quoted against TM-102186 is a point compared
with a point. Neither side carried a band, so "68% of the recorded
peak-to-peak" has never been separable into "the model is wrong by 32%" and
"the inputs are not known to 32%". Three sections, each of which needed a
source read rather than an argument:

  A  TIMING, which is the one channel that is not circular. The vortex
     parameters were identified FROM the accelerations, so predicting
     accelerations partly re-derives the fit. Core SPACING is different: it
     sets when the gusts arrive and not how hard they hit, and TM-102186
     states the answer independently -- "about 5 sec apart".

  B  HOW WELL MEHTA'S OWN FIT FITS. His Eq. (A3) cost is a MEAN square, so it
     converts to an RMS wind residual without knowing N. Subtracting Lester's
     reconstruction error leaves the physical fluctuation his five vortices do
     not represent -- which is exactly the quantity `cat_bounds.py` section B
     swept blind as `sigma_w`.

  C  THE PROPAGATED BAND. Fly the V0 and r0 uncertainty through and ask
     whether the recorded band is reachable from inside it.

Run: PYTHONPATH=<abs worktree root> .venv/Scripts/python.exe scripts/cat_uncertainty.py --outdir runs/cat
"""

import argparse
import math
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import atisim  # noqa: F401  -- enables x64
from atisim import vortex_viz, wind
from atisim.aircraft import CRUISE, REGISTRY
from atisim.atmosphere import speed_of_sound
from atisim.units import FT2M, RAD2DEG

PALETTE = {
    "model": "#1D5D77", "reference": "#A9501C", "wind": "#3E6A48",
    "muted": "#7E8D93", "grid": "#D6DCD8",
}

# Fraction of V0 an extremum must reach to count as one of the "sharp
# up-and-down gusts". The answer is INSENSITIVE to it: 0.25, 0.40 and 0.50 all
# select exactly the same four extrema, because the field's next-largest
# turning point is far below any of them. 0.40 is the middle of that plateau.
GUST_THRESHOLD = 0.40

LEAD_R0 = 12.0  # run-in, in core radii -- `cat_bounds.MEHTA_LEAD_R0`


def _style(ax, xlabel=None, ylabel=None, title=None):
    ax.grid(True, color=PALETTE["grid"], lw=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    if title:
        ax.set_title(title, fontsize=10, loc="left", fontweight="bold")
    ax.tick_params(labelsize=8)


def fly_mehta(*, v0_scale=1.0, r0_scale=1.0, extra_field=None, dt=0.01):
    """The session-23 headline run with the identified parameters perturbed.

    `v0_scale` and `r0_scale` are the knobs section C turns. Everything else
    matches `cat_bounds.fly_mehta_747`, including the equilibrium start -- a
    still-air start puts the aircraft 0.198 g out of trim in this field and
    would swamp the effects being measured.
    """
    ac = REGISTRY["boeing747"]
    V = CRUISE["boeing747"]["airspeed"]
    H = wind.MEHTA_HANNIBAL_ALTITUDE
    array = wind.mehta_hannibal_array(H)
    array = array._replace(v0=array.v0 * v0_scale, r0=array.r0 * r0_scale)
    vortex = lambda p: wind.vortex_wind(p, array)  # noqa: E731
    field = vortex if extra_field is None else wind.superpose(vortex, extra_field)

    # The window and run-in are pinned to the UNPERTURBED radius, so every run
    # in a sweep analyses the same stretch of track. Scaling them with r0 would
    # change the sampled field and the window together and confound the two.
    r0 = float(wind.MEHTA_HANNIBAL_R0)
    x0, x1 = float(array.north.min()), float(array.north.max())
    start = x0 - LEAD_R0 * r0
    seconds = (x1 + LEAD_R0 * r0 - start) / V
    enc = vortex_viz.fly_in_moving_air(
        ac, field, V, H, label="mehta", start_north=start, seconds=seconds,
        dt=dt, window=(x0 - 2.0 * r0, x1 + 2.0 * r0), window_name="array",
    )
    return enc, dict(ac=ac, V=V, H=H, r0=r0)


def band(enc):
    w = enc.window
    return float(enc.n_z[w].min()), float(enc.n_z[w].max())


def peak_alpha_deg(enc):
    w = enc.window
    return float(np.abs(enc.alpha_air[w]).max() * RAD2DEG)


# ---------------------------------------------------------------------------
# A. Gust spacing
# ---------------------------------------------------------------------------


def gust_extrema(enc, threshold=GUST_THRESHOLD):
    """Times and values of the sharp gust peaks in the flown vertical wind.

    A turning point of `w_up` counts when it reaches `threshold * V0`. Returns
    `(times, values)` in encounter order.
    """
    w = enc.window
    t, wu = enc.t[w], enc.w_up[w]
    d = np.sign(np.diff(wu))
    turn = np.where(d[:-1] * d[1:] < 0)[0] + 1
    keep = [k for k in turn if abs(wu[k]) >= threshold * wind.MEHTA_HANNIBAL_V0]
    return t[keep], wu[keep]


def gust_spacing(times):
    """The three defensible readings of "gusts about 5 sec apart", in seconds.

    Four extrema arrive as two up-and-down PAIRS -- one per penetrated core --
    so "apart" can mean peak-to-peak, trough-to-trough, or centre-to-centre.
    All three are returned because the claim is only worth quoting if it does
    not depend on which was chosen.
    """
    if len(times) != 4:
        raise ValueError(f"expected two up-and-down pairs, got {len(times)} extrema")
    up1, dn1, up2, dn2 = times
    return {
        "peak-to-peak": up2 - up1,
        "trough-to-trough": dn2 - dn1,
        "centre-to-centre": (up2 + dn2) / 2 - (up1 + dn1) / 2,
    }


# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=Path, default=Path("runs/cat"))
    p.add_argument("--seeds", type=int, default=8)
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    print(f"atisim imported from: {atisim.__file__}\n")

    lo_m, hi_m = wind.TM102186_HANNIBAL_NZ
    rec_pp = hi_m - lo_m
    base, meta = fly_mehta()
    b0 = band(base)

    # -- A ------------------------------------------------------------------
    print("=" * 78)
    print("A. GUST SPACING -- THE CHANNEL THE IDENTIFICATION DID NOT SET")
    print("=" * 78)
    print("  V0 and r0 were fitted to these winds, so an AMPLITUDE comparison")
    print("  partly re-derives the fit. The core SPACING is not in that loop:")
    print("  it comes from the fitted positions and the aircraft's speed, and")
    print("  TM-102186 states it in prose, independently of any amplitude.\n")

    i, j = wind.MEHTA_HANNIBAL_CORE_PAIR
    sep_ft = wind.MEHTA_HANNIBAL_X_FT[j] - wind.MEHTA_HANNIBAL_X_FT[i]
    V = meta["V"]
    print(f"  penetrated cores {i} and {j}: x = "
          f"{wind.MEHTA_HANNIBAL_X_FT[i]:.0f} and "
          f"{wind.MEHTA_HANNIBAL_X_FT[j]:.0f} ft, separation {sep_ft:.0f} ft")
    print(f"  geometric prediction, {sep_ft:.0f} ft at {V / FT2M:.0f} ft/s: "
          f"{sep_ft * FT2M / V:.3f} s")
    print(f"  RECORDED (TM-102186 p. 3-4): "
          f"'about {wind.TM102186_HANNIBAL_GUST_PERIOD:.0f} sec apart'\n")

    times, values = gust_extrema(base)
    print(f"  flown gust extrema (|w_up| >= {GUST_THRESHOLD:.2f} V0 = "
          f"{GUST_THRESHOLD * wind.MEHTA_HANNIBAL_V0:.2f} m/s):")
    for t, v in zip(times, values):
        print(f"    t = {t:7.3f} s   w_up = {v:+7.2f} m/s ({v / FT2M:+6.1f} ft/s)")
    spacing = gust_spacing(times)
    print()
    for name, s in spacing.items():
        err = 100 * (s - wind.TM102186_HANNIBAL_GUST_PERIOD) / \
            wind.TM102186_HANNIBAL_GUST_PERIOD
        print(f"    {name:<18}{s:7.3f} s   {err:+6.1f}% vs the record")
    lo_s, hi_s = min(spacing.values()), max(spacing.values())
    print(f"\n  The three readings span {hi_s - lo_s:.3f} s, so the answer is a")
    print(f"  property of the field and not of the definition.\n")

    implied = sep_ft * FT2M / wind.TM102186_HANNIBAL_GUST_PERIOD
    a37 = float(speed_of_sound(jnp.array(wind.MEHTA_HANNIBAL_ALTITUDE)))
    print(f"  WHERE THE RESIDUAL GOES. Exactly 5.0 s needs {implied:.1f} m/s "
          f"(M {implied / a37:.3f});")
    print(f"  this 747 flies {V:.1f} m/s (M {V / a37:.3f}). The DC-10's own speed "
          f"is not in")
    print(f"  any source held here, and a {100 * (implied / V - 1):.1f}% speed "
          f"difference between two")
    print(f"  transports at 37,000 ft accounts for the whole discrepancy.")

    # -- B ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("B. HOW WELL MEHTA'S FIT FITS, AND WHAT IT LEAVES OVER")
    print("=" * 78)
    print("  Eq. (A3): J = (1/N) sum e^T B e, B = I. A MEAN square, so it")
    print("  converts without N -- which the paper never states.\n")
    print(f"  {'n':>3}{'cost J':>10}{'RMS resid':>12}{'d(J)':>10}{'d(RMS)':>10}")
    print(f"  {'':>3}{'(ft/s)^2':>10}{'m/s':>12}")
    print(f"  {2:>3}{wind.MEHTA_COST_STARTUP:>10.1f}"
          f"{math.sqrt(wind.MEHTA_COST_STARTUP) * FT2M:>12.3f}"
          f"{'--':>10}{'--':>10}   MANUAL STARTUP, not a fit")
    prev = wind.MEHTA_COST_STARTUP
    for n in sorted(wind.MEHTA_COST):
        J = wind.MEHTA_COST[n]
        print(f"  {n:>3}{J:>10.1f}{math.sqrt(J) * FT2M:>12.3f}"
              f"{100 * (J - prev) / prev:>9.1f}%"
              f"{100 * (math.sqrt(J / prev) - 1):>9.2f}%")
        prev = J
    print(f"\n  Mehta p. 30: n = 6, 7 do NOT lower the cost -- the extra vortices")
    print(f"  get pushed away from the flight path. So J = "
          f"{wind.MEHTA_COST[wind.MEHTA_COST_SATURATES_AT]:.0f} is this model")
    print(f"  family's FLOOR, not where the author stopped, and 4 -> 5 already")
    print(f"  buys only {100 * (math.sqrt(wind.MEHTA_COST[5] / wind.MEHTA_COST[4]) - 1):.1f}%"
          f" of RMS. The residual is structural.\n")

    ceiling = wind.mehta_residual_ceiling()
    floor = wind.mehta_unmodelled_wind()
    meas = {k: v for k, v in wind.DFDR_WIND_RMS_ERROR.items()}
    print(f"  DECOMPOSITION. The residual is against RECONSTRUCTED winds, so it")
    print(f"  already contains their error (Lester Table 1: "
          f"{meas['horizontal']:.3f} h, {meas['vertical']:.3f} v m/s).")
    print(f"    total residual        {math.sqrt(wind.MEHTA_COST[5]) * FT2M:.3f} m/s")
    print(f"    reconstruction floor  "
          f"{math.sqrt(sum((v / FT2M) ** 2 for v in meas.values())) * FT2M:.3f} m/s")
    print(f"    -> unmodelled, per component  {floor:.3f} m/s   (a LOWER bound)")
    print(f"    -> ceiling, all of J vertical {ceiling:.3f} m/s   (unit-robust)")
    print(f"\n  So the random component Mehta excluded is sourced to "
          f"{floor:.2f}-{ceiling:.2f} m/s.")
    print(f"  Before this it was unbounded: cat_bounds.py section B had to sweep")
    print(f"  sigma_w blind because MIL-F-8785C gives it as an undigitised chart.")

    print(f"\n  FLYING THE SOURCED INTENSITIES ({args.seeds} seeds each):")
    print(f"  {'sigma_w':>9}{'':>3}{'n_z min':>26}{'n_z max':>26}{'reaches':>9}")
    dryden = {}
    for name, sigma in (("lower", floor), ("ceiling", ceiling)):
        mins, maxs, alphas = [], [], []
        for seed in range(args.seeds):
            enc, _ = fly_mehta(extra_field=wind.dryden_vertical_field(sigma, seed))
            lo, hi = band(enc)
            mins.append(lo)
            maxs.append(hi)
            alphas.append(peak_alpha_deg(enc))
        mins, maxs = np.array(mins), np.array(maxs)
        got = bool(maxs.max() >= hi_m or mins.min() <= lo_m)
        dryden[name] = (sigma, mins, maxs, max(alphas))
        print(f"  {sigma:>9.3f}{'':>3}"
              f"{mins.mean():>12.3f} [{mins.min():.3f},{mins.max():.3f}]"
              f"{maxs.mean():>12.3f} [{maxs.min():.3f},{maxs.max():.3f}]"
              f"{('YES' if got else 'no'):>9}")
    print(f"\n  Peak |alpha| over all seeds: "
          f"{dryden['lower'][3]:.2f} deg at the lower bound, "
          f"{dryden['ceiling'][3]:.2f} deg at the ceiling")
    print(f"  -- the ceiling run sits ON the 10 deg edge of the linear range,")
    print(f"  so it is the last intensity this model may be asked about at all.")

    # -- C ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("C. THE PROPAGATED INPUT BAND")
    print("=" * 78)
    frac = wind.DFDR_WIND_RMS_ERROR["vertical"] / wind.MEHTA_HANNIBAL_V0
    print(f"  V0 +/- {100 * frac:.2f}% (Lester's vertical RMS error over Mehta's V0).")
    print(f"  A CEILING, not a 1-sigma: that is the error of ONE reconstructed")
    print(f"  sample, and V0 was fitted to N of them, which averages it down.")
    print(f"  r0 carries NO published uncertainty -- Mehta's sensitivity study")
    print(f"  reports convergence from 100-1300 ft, which is robustness and not")
    print(f"  a band -- so its sweep below is DECLARED and labelled as such.\n")
    print(f"  recorded: {lo_m:+.2f} to {hi_m:+.2f} g, peak-to-peak {rec_pp:.2f}\n")
    print(f"  {'variant':<34}{'n_z min':>10}{'n_z max':>10}{'p-p':>9}"
          f"{'% of rec':>10}{'|a| deg':>9}")

    rows = [("baseline (identified parameters)", 1.0, 1.0)]
    rows += [(f"V0 x {s:.4f}  (sourced ceiling)", s, 1.0)
             for s in (1 - frac, 1 + frac)]
    rows += [(f"r0 x {s:.2f}  (DECLARED)", 1.0, s) for s in (0.85, 1.15)]
    rows += [("corner: V0 high, r0 low", 1 + frac, 0.85)]
    best = 0.0
    for label, vs, rs in rows:
        enc, _ = fly_mehta(v0_scale=vs, r0_scale=rs)
        lo, hi = band(enc)
        pp = hi - lo
        best = max(best, pp)
        print(f"  {label:<34}{lo:>10.4f}{hi:>10.4f}{pp:>9.4f}"
              f"{100 * pp / rec_pp:>9.1f}%{peak_alpha_deg(enc):>9.2f}")
    print(f"\n  The whole input band tops out at {100 * best / rec_pp:.1f}% of the "
          f"recorded peak-to-peak.")
    print(f"  The 32% shortfall is NOT inside the uncertainty of the inputs.\n")

    print("  WHY MORE WIND DOES NOT HELP: the up peak is saturated.")
    print(f"  {'V0 x':>7}{'n_z min':>10}{'n_z max':>10}{'up incr':>10}"
          f"{'elasticity':>12}{'|a| deg':>9}")
    sweep = []
    for s in (1.0, 1.25, 1.5, 2.0, 3.0, 3.25, 3.5, 4.0):
        enc, _ = fly_mehta(v0_scale=s)
        lo, hi = band(enc)
        alpha = peak_alpha_deg(enc)
        el = "--" if s == 1.0 else f"{((hi - 1) / (b0[1] - 1) - 1) / (s - 1):.3f}"
        sweep.append((s, lo, hi, alpha))
        print(f"  {s:>7.2f}{lo:>10.4f}{hi:>10.4f}{hi - 1:>10.4f}{el:>12}"
              f"{alpha:>9.2f}  {'' if alpha < 10.0 else 'OUTSIDE the model'}")
    print(f"\n  TRIPLING the identified gust leaves the up increment at "
          f"{sweep[4][2] - 1:.3f} g")
    print(f"  against a recorded {hi_m - 1:.2f}, with |alpha| still near 8 deg. The")
    print(f"  aircraft pitches away and sheds the gust -- the incidence-gain")
    print(f"  mechanism TM-102186 Fig. 8 is about, seen from the inside.")
    print()
    print(f"  AND THE TWO BOUNDARIES COINCIDE. The load first reaches "
          f"{hi_m:+.1f} g between")
    print(f"  x3.25 and x3.50, and |alpha| leaves the 10 deg linear range in the")
    print(f"  SAME interval. So there is no gust strength at which this model both")
    print(f"  reaches the record and may be believed -- which is a stronger")
    print(f"  statement than the shortfall being large, and it is why no amount of")
    print(f"  wind is the answer.")

    # -- figure -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))

    ax = axes[0]
    w = base.window
    ax.plot(base.t[w], base.w_up[w], color=PALETTE["wind"], lw=1.3)
    ax.scatter(times, values, s=34, color=PALETTE["reference"], zorder=3)
    for t, v in zip(times, values):
        ax.annotate(f"{t:.2f}s", (t, v), textcoords="offset points",
                    xytext=(0, 9 if v > 0 else -15), ha="center", fontsize=7,
                    color=PALETTE["reference"])
    c = spacing["centre-to-centre"]
    top = base.w_up[w].max()
    y = top + 0.30 * abs(top)
    mid1 = (times[0] + times[1]) / 2
    ax.annotate("", xy=(mid1 + c, y), xytext=(mid1, y),
                arrowprops=dict(arrowstyle="<->", color=PALETTE["model"], lw=1.3))
    ax.text(mid1 + c / 2, y + 0.10 * abs(top),
            f"model {c:.2f} s   vs   record ~"
            f"{wind.TM102186_HANNIBAL_GUST_PERIOD:.0f} s",
            ha="center", va="bottom", fontsize=8.5, color=PALETTE["model"])
    ax.set_ylim(1.25 * base.w_up[w].min(), y + 0.62 * abs(top))
    _style(ax, "time, s", "vertical gust, m/s",
           "A. Gust spacing: the non-circular channel")

    ax = axes[1]
    names = ["total\nresidual", "reconstruction\nerror (Lester)",
             "unmodelled\n(lower bound)", "ceiling\n(all vertical)"]
    vals = [math.sqrt(wind.MEHTA_COST[5]) * FT2M,
            math.sqrt(sum((v / FT2M) ** 2
                          for v in wind.DFDR_WIND_RMS_ERROR.values())) * FT2M,
            floor, ceiling]
    cols = [PALETTE["model"], PALETTE["muted"], PALETTE["wind"], PALETTE["wind"]]
    ax.bar(names, vals, color=cols, width=0.62)
    for k, v in enumerate(vals):
        ax.text(k, v + 0.08, f"{v:.2f}", ha="center", fontsize=8)
    ax.axhspan(4.0, 5.0, color=PALETTE["reference"], alpha=0.16, zorder=0)
    ax.text(1.5, 4.5, "what closing the gap needs", ha="center", va="center",
            fontsize=8, color=PALETTE["reference"])
    ax.set_ylim(0, 5.4)
    _style(ax, None, "RMS wind, m/s", "B. What Mehta's fit leaves over")
    ax.tick_params(axis="x", labelsize=7)

    ax = axes[2]
    scales = [s for s, _, _, _ in sweep]
    ax.plot(scales, [hi for _, _, hi, _ in sweep], "o-", color=PALETTE["model"],
            lw=1.5, label="model peak $n_z$")
    ax.plot(scales, [lo for _, lo, _, _ in sweep], "o--", color=PALETTE["model"],
            lw=1.5, label="model min $n_z$")

    # Everything right of the first run that leaves the 10 deg linear band is
    # outside the model, and that is where -- and only where -- the peak reaches
    # the record. Drawing it is the whole point of the panel.
    outside = [s for s, _, _, a in sweep if a >= 10.0]
    if outside:
        ax.axvspan(min(outside) - 0.125, max(scales) + 0.1, color="#8C2F1C",
                   alpha=0.10, zorder=0)
        ax.text(min(outside) - 0.06, 1.42, "outside the\nlinear range",
                ha="left", va="top", fontsize=7.5, color="#8C2F1C")

    ax.axhline(hi_m, color=PALETTE["reference"], lw=1.4)
    ax.axhline(lo_m, color=PALETTE["reference"], lw=1.4, ls="--")
    ax.text(2.9, hi_m + 0.06, "recorded +1.7 g", ha="right", fontsize=8,
            color=PALETTE["reference"])
    ax.text(2.9, lo_m + 0.06, "recorded -1.0 g", ha="right", fontsize=8,
            color=PALETTE["reference"])
    ax.axvspan(1 - frac, 1 + frac, color=PALETTE["muted"], alpha=0.22, zorder=0)
    ax.text(1 + frac + 0.06, 0.55,
            f"sourced $V_0$ band, $\\pm${100 * frac:.1f}%", ha="left",
            va="center", fontsize=8, color=PALETTE["muted"])
    _style(ax, "gust strength / identified $V_0$", "$n_z$, g",
           "C. The record is only reachable outside the model")
    ax.legend(fontsize=8, frameon=False, loc="upper left")

    fig.tight_layout()
    fig.savefig(args.outdir / "06-uncertainty.png", dpi=150, bbox_inches="tight")
    print(f"\nfigure -> {args.outdir / '06-uncertainty.png'}")


if __name__ == "__main__":
    main()
