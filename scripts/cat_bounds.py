"""What the modelling choices cost on the CAT runs, and one case the 747 can own.

Three questions the session-23 evidence pass left open, each of which needed a
run rather than an argument:

  A  Does the 68% load shortfall on the Mehta run come from the model's own
     approximations? Bounds the point-sampled gust and the step size.
  B  Mehta says his fit represents "the sudden, violent, and periodic
     disturbances ... and not the small, random fluctuations". Superpose the
     missing random part and find what intensity closes the gap.
  C  Every load comparison so far is a 747 flown against a DC-10 record. Lester
     et al. 1989 recorded a B-747 -- the type this project models -- so fly it
     and remove the aircraft-type confound.

Run: PYTHONPATH=<abs worktree root> .venv/Scripts/python.exe scripts/cat_bounds.py --outdir runs/cat
"""

import argparse
import math
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import atisim  # noqa: F401  -- enables x64
from atisim import airframe, checks, loads, trim, vortex_viz, wind
from atisim.aircraft import CRUISE, REGISTRY
from atisim.atmosphere import speed_of_sound
from atisim.units import FT2M, RAD2DEG

PALETTE = {
    "model": "#1D5D77", "reference": "#A9501C", "wind": "#3E6A48",
    "muted": "#7E8D93", "grid": "#D6DCD8",
}


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


# ---------------------------------------------------------------------------
# The Mehta run, parameterised by everything this script wants to vary.
# ---------------------------------------------------------------------------

MEHTA_LEAD_R0 = 12.0


def fly_mehta_747(*, dt=0.01, extra_field=None, sampled=False, strip=False):
    """The session-23 headline run, with one knob turned at a time."""
    ac = REGISTRY["boeing747"]
    V = CRUISE["boeing747"]["airspeed"]
    H = wind.MEHTA_HANNIBAL_ALTITUDE
    array = wind.mehta_hannibal_array(H)
    vortex = lambda p: wind.vortex_wind(p, array)  # noqa: E731
    field = vortex if extra_field is None else wind.superpose(vortex, extra_field)

    r0 = float(array.r0)
    x0, x1 = float(array.north.min()), float(array.north.max())
    start = x0 - MEHTA_LEAD_R0 * r0
    seconds = (x1 + MEHTA_LEAD_R0 * r0 - start) / V

    kw = {}
    if sampled:
        kw["wind_model"] = wind.sampled_field_model(field, airframe.stations(ac))
    if strip:
        kw["load_model"] = loads.strip_model(field, ac)

    enc = _fly(ac, field, V, H, start, seconds, dt, x0, x1, r0, **kw)
    return enc, dict(ac=ac, V=V, H=H, r0=r0, field=field)


def _fly(ac, field, V, H, start, seconds, dt, x0, x1, r0, **kw):
    """`fly_in_moving_air`, plus the two overrides it does not take directly."""
    if not kw:
        return vortex_viz.fly_in_moving_air(
            ac, field, V, H, label="mehta", start_north=start, seconds=seconds,
            dt=dt, window=(x0 - 2.0 * r0, x1 + 2.0 * r0), window_name="array",
        )
    # The overriding paths need the state built the same way -- in equilibrium
    # with the wind already present -- or the comparison measures the start
    # transient instead of the thing being varied.
    from atisim.state import quat_to_dcm

    x, _ = trim.trim(jnp.array(V), jnp.array(H), ac)
    controls = trim.trimmed_controls(x[1], x[2])
    state = trim.trimmed_state(jnp.array(float(x[0])), jnp.array(V), jnp.array(H))
    state = state._replace(pos_ned=jnp.array([start, 0.0, -H]))
    dcm = quat_to_dcm(state.quat)
    state = state._replace(vel_body=state.vel_body + dcm.T @ field(state.pos_ned))
    return vortex_viz.fly_from_state(
        ac, field, state, controls, label="mehta", seconds=seconds, dt=dt,
        window=(x0 - 2.0 * r0, x1 + 2.0 * r0), window_name="array", **kw,
    )


def band(enc):
    w = enc.window
    return float(enc.n_z[w].min()), float(enc.n_z[w].max())


def ptp_pitch(enc):
    w = enc.window
    return float((enc.theta[w].max() - enc.theta[w].min()) * RAD2DEG)


# ---------------------------------------------------------------------------
# C. The Greenland 747
# ---------------------------------------------------------------------------


def fly_greenland(w0, *, dt=0.01, wavelength=None, cycles=3.0):
    """The 747 through a lee wave at Lester's altitude and wavelength.

    Lester et al. give the WAVELENGTH (22 km) and the aircraft's response
    (+2.7/-1.0 g, a 300 m altitude gain) but NOT a wave amplitude, so `w0` is
    the caller's and the script sweeps it rather than picking one.
    """
    ac = REGISTRY["boeing747"]
    H = wind.LESTER_GREENLAND_ALTITUDE
    lam = wind.LESTER_LEE_WAVE_WAVELENGTH if wavelength is None else wavelength
    # M 0.80, the Mach this derivative set is tabulated at, at Lester's altitude.
    V = 0.80 * float(speed_of_sound(jnp.array(H)))
    wave = wind.LeeWave(w0=jnp.array(w0), wavelength=jnp.array(lam),
                        north=jnp.array(0.0))
    field = lambda p: wind.lee_wave_wind(p, wave)  # noqa: E731

    seconds = cycles * lam / V
    enc = vortex_viz.fly_in_moving_air(
        ac, field, V, H, label=f"greenland w0={w0}", start_north=-0.25 * lam,
        seconds=seconds, dt=dt, window=(-0.25 * lam, cycles * lam),
        window_name="the wave train",
    )
    return enc, dict(ac=ac, V=V, H=H, lam=lam)


# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=Path, default=Path("runs/cat"))
    p.add_argument("--seeds", type=int, default=8)
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    print(f"atisim imported from: {atisim.__file__}\n")

    lo_m, hi_m = wind.TM102186_HANNIBAL_NZ

    # -- A ------------------------------------------------------------------
    print("=" * 78)
    print("A. WHAT THE MODELLING CHOICES COST ON THE MEHTA RUN")
    print("=" * 78)
    base, meta = fly_mehta_747()
    b0 = band(base)
    print("  Changes are quoted on the INCREMENT from 1 g, which is the quantity")
    print("  the gust produces -- a fraction of n_z itself would flatter the")
    print("  negative excursion, where n_z is small and its increment is not.\n")
    print(f"  {'variant':<34}{'n_z min':>10}{'n_z max':>10}{'pitch p-p':>11}"
          f"{'d(up)':>9}{'d(down)':>9}")
    print(f"  {'baseline (tangent, point loads)':<34}{b0[0]:>10.4f}{b0[1]:>10.4f}"
          f"{ptp_pitch(base):>11.2f}{'--':>9}{'--':>9}")

    for label, kw in (("gust rates fitted across airframe", dict(sampled=True)),
                      ("strip-integrated loads", dict(strip=True))):
        enc, _ = fly_mehta_747(**kw)
        b = band(enc)
        up = 100 * ((b[1] - 1.0) - (b0[1] - 1.0)) / abs(b0[1] - 1.0)
        dn = 100 * ((b[0] - 1.0) - (b0[0] - 1.0)) / abs(b0[0] - 1.0)
        print(f"  {label:<34}{b[0]:>10.4f}{b[1]:>10.4f}{ptp_pitch(enc):>11.2f}"
              f"{up:>8.2f}%{dn:>8.2f}%")

    print()
    print("  step-size convergence (peak load):")
    prev = None
    for dt in (0.02, 0.01, 0.005, 0.0025):
        enc, _ = fly_mehta_747(dt=dt)
        b = band(enc)
        d = "--" if prev is None else f"{100 * (b[1] - prev) / abs(prev):+.3f}%"
        print(f"    dt {dt:<8.4f} n_z {b[0]:+.5f} to {b[1]:+.5f}   change {d}")
        prev = b[1]
    print()

    # -- B ------------------------------------------------------------------
    print("=" * 78)
    print("B. THE RANDOM COMPONENT MEHTA'S FIT EXCLUDES")
    print("=" * 78)
    print(f"  Dryden L_w = {wind.DRYDEN_LW:.1f} m (MIL-F-8785C, above 2000 ft)")
    print(f"  sigma_w is NOT sourced -- the spec gives it as a chart this project"
          f" has not digitised.")
    print(f"  Measured target: the DC-10 recorded {lo_m:+.1f} to {hi_m:+.1f} g.\n")
    print(f"  {'sigma_w':>8}{'n_z min':>26}{'n_z max':>26}{'reaches':>10}")
    print(f"  {'m/s':>8}{'mean [min,max] over seeds':>26}"
          f"{'mean [min,max] over seeds':>26}{'band?':>10}")
    sweep = []
    for sigma in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
        mins, maxs = [], []
        n = 1 if sigma == 0.0 else args.seeds
        for seed in range(n):
            extra = (None if sigma == 0.0
                     else wind.dryden_vertical_field(sigma, seed))
            enc, _ = fly_mehta_747(extra_field=extra)
            b = band(enc)
            mins.append(b[0])
            maxs.append(b[1])
        mins, maxs = np.array(mins), np.array(maxs)
        got = bool((maxs.max() >= hi_m) or (mins.min() <= lo_m))
        sweep.append((sigma, mins, maxs))
        print(f"  {sigma:>8.1f}{mins.mean():>12.3f} [{mins.min():.3f},{mins.max():.3f}]"
              f"{maxs.mean():>12.3f} [{maxs.min():.3f},{maxs.max():.3f}]"
              f"{('YES' if got else 'no'):>10}")
    print()

    # -- C ------------------------------------------------------------------
    print("=" * 78)
    print("C. LESTER'S GREENLAND 747 -- THE TYPE THIS PROJECT ACTUALLY MODELS")
    print("=" * 78)
    lo_g, hi_g = wind.LESTER_GREENLAND_NZ
    print(f"  altitude {wind.LESTER_GREENLAND_ALTITUDE:.0f} m "
          f"({wind.LESTER_GREENLAND_ALTITUDE / FT2M:.0f} ft), "
          f"wavelength {wind.LESTER_LEE_WAVE_WAVELENGTH / 1000:.0f} km")
    print(f"  MEASURED: {lo_g:+.1f} to {hi_g:+.1f} g, and a "
          f"{wind.LESTER_GREENLAND_ALTITUDE_GAIN:.0f} m altitude gain")
    print(f"  Doyle et al. amplitudes, for scale: "
          f"{wind.LEE_WAVE_AMPLITUDE['north']:.0f} and "
          f"{wind.LEE_WAVE_AMPLITUDE['south']:.0f} m/s (a DIFFERENT campaign, "
          f"the Sierra Nevada)\n")
    print(f"  {'w0 m/s':>8}{'n_z min':>10}{'n_z max':>10}{'|dn| peak':>11}"
          f"{'climb m':>10}{'band widths out':>17}")
    grid, bands = [], {}
    for w0 in (3.0, 6.0, 12.0, 20.0, 30.0):
        enc, gm = fly_greenland(w0)
        b = band(enc)
        climb = float(enc.altitude.max() - enc.altitude.min())
        dn = max(abs(b[0] - 1.0), abs(b[1] - 1.0))
        # The band this entry declares is 35,000-45,000 ft; Lester's aircraft
        # was at 33,000, so EVERY run here is outside it. Reported per
        # amplitude rather than once, because the violent runs leave it far
        # further than the mild ones and the inversion below uses a mild one.
        bands[w0] = checks.recovery_band(enc.log, gm["ac"])
        grid.append((w0, dn, climb, b))
        print(f"  {w0:>8.1f}{b[0]:>10.3f}{b[1]:>10.3f}{dn:>11.3f}{climb:>10.1f}"
              f"{bands[w0].value:>17.2f}")
    print("\n  RECOVERY BAND, on the 6 m/s run the inversion below uses:")
    print(f"    {bands[6.0].detail}")
    print("    Lester's aircraft was at 33,000 ft and this entry is tabulated at")
    print("    40,000, so the excursion is real. aircraft.py's interpolation puts")
    print("    the short-period frequency error at that altitude at about 5%.")

    # The response is linear in w0 for small perturbations, so invert it.
    w0_ref, dn_ref = grid[1][0], grid[1][1]
    need = max(abs(lo_g - 1.0), abs(hi_g - 1.0))
    print(f"\n  Linear inversion from the {w0_ref:.0f} m/s run "
          f"(|dn| {dn_ref:.3f} g):")
    print(f"    reaching the recorded |dn| of {need:.1f} g needs "
          f"w0 ~ {w0_ref * need / dn_ref:.1f} m/s")
    print(f"    that is {w0_ref * need / dn_ref / wind.LEE_WAVE_AMPLITUDE['south']:.1f}x "
          f"Doyle's largest measured amplitude")
    # The same inversion on the OTHER observation Lester records.
    climbs = np.array([g[2] for g in grid])
    w0s_a = np.array([g[0] for g in grid])
    w0_climb = float(np.interp(wind.LESTER_GREENLAND_ALTITUDE_GAIN, climbs, w0s_a))
    print(f"    but the recorded {wind.LESTER_GREENLAND_ALTITUDE_GAIN:.0f} m altitude "
          f"gain needs only w0 ~ {w0_climb:.1f} m/s")
    print(f"    -> the two observations imply amplitudes "
          f"{w0_ref * need / dn_ref / w0_climb:.0f}x apart, so ONE smooth wave "
          f"cannot produce both")

    # -- figure -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    ax = axes[0]
    sig = [s for s, _, _ in sweep]
    mx = [m.mean() for _, _, m in sweep]
    mn = [m.mean() for _, m, _ in sweep]
    ax.fill_between(sig, [m.min() for _, _, m in sweep],
                    [m.max() for _, _, m in sweep], color=PALETTE["model"],
                    alpha=0.18, lw=0)
    ax.fill_between(sig, [m.min() for _, m, _ in sweep],
                    [m.max() for _, m, _ in sweep], color=PALETTE["model"],
                    alpha=0.18, lw=0)
    ax.plot(sig, mx, "o-", color=PALETTE["model"], lw=1.6, ms=5, label="model peak")
    ax.plot(sig, mn, "o-", color=PALETTE["model"], lw=1.6, ms=5)
    ax.axhline(hi_m, color=PALETTE["reference"], lw=1.2, ls="--")
    ax.axhline(lo_m, color=PALETTE["reference"], lw=1.2, ls="--",
               label=f"DC-10 measured {lo_m:+.1f} / {hi_m:+.1f} g")
    _style(ax, xlabel="Dryden $\\sigma_w$, m/s  (NOT sourced — swept)",
           ylabel="load factor extremes, g",
           title="a  How much random gust would close the gap")
    ax.legend(fontsize=8, loc="center left", framealpha=0.9)
    ax.annotate("upper extreme reaches the record\nat $\\sigma_w \\approx$ 4–5 m/s",
                xy=(3.1, 1.80), fontsize=8, color=PALETTE["muted"])
    ax.annotate("lower extreme never does — the spread\nwidens, the mean does not move",
                xy=(0.5, -0.90), fontsize=8, color=PALETTE["muted"])

    ax = axes[1]
    w0s = [g[0] for g in grid]
    dns = [g[1] for g in grid]
    ax.plot(w0s, dns, "o-", color=PALETTE["wind"], lw=1.6, ms=6, label="model")
    ax.axhline(need, color=PALETTE["reference"], lw=1.2, ls="--",
               label=f"Lester recorded $|\\Delta n|$ = {need:.1f} g")
    for a, lbl in ((wind.LEE_WAVE_AMPLITUDE["north"], "Doyle N"),
                   (wind.LEE_WAVE_AMPLITUDE["south"], "Doyle S")):
        ax.axvline(a, color=PALETTE["muted"], lw=0.9, ls=":")
        ax.annotate(lbl, xy=(a, max(dns) * 0.94), fontsize=7.5, rotation=90,
                    color=PALETTE["muted"], ha="right", va="top")
    ax.axvline(w0_climb, color=PALETTE["model"], lw=1.2, ls="-.")
    ax.annotate(f"the recorded 300 m climb\nimplies only {w0_climb:.0f} m/s",
                xy=(w0_climb + 1.2, need * 0.36), fontsize=8,
                color=PALETTE["model"])
    ax.annotate(
        f"the recorded {need:.0f} g needs ~{w0_ref * need / dn_ref:.0f} m/s —\n"
        f"{w0_ref * need / dn_ref / w0_climb:.0f}$\\times$ further right than "
        "this axis goes",
        xy=(8.5, need * 0.90), fontsize=8, color=PALETTE["reference"])
    _style(ax, xlabel="lee-wave amplitude $w_0$, m/s",
           ylabel="peak $|\\Delta n|$, g",
           title="b  Greenland — one wave cannot produce both observations")
    ax.legend(fontsize=8, loc="lower right", framealpha=0.95)
    fig.savefig(args.outdir / "05-bounds.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nfigure -> {args.outdir / '05-bounds.png'}")


if __name__ == "__main__":
    main()
