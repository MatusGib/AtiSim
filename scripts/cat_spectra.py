"""Change what counts as agreement: response SPECTRA and load EXCEEDANCE.

Every comparison in PROJECT.md section 4 matches a PEAK from a single
encounter. That is one realisation of a random process, and its error bar does
not shrink no matter how much more work is done. This script asks the two
questions that do get better with effort, and neither needs a source the project
does not already hold.

  A  THE HEADLINE RUN AS A SPECTRUM. Mehta's five-core Hannibal field, flown as
     always -- but reported as where the response energy sits rather than as how
     large its largest excursion was. The array is not uniform, so it forces the
     aircraft at four different core-passage frequencies at once; the aircraft
     has one short period. A spectrum is the only statistic that can say which
     of those the load followed.

  B  YOSHIMURA'S PROTOCOL, RUN. Yoshimura et al. 2023 (GRL 50, e2022GL101286)
     validate a CAT simulation by comparing the FREQUENCY SPECTRUM of vertical
     acceleration between virtual and recorded flights, and by checking the peak
     lands near the aircraft's own natural frequency -- 0.14 Hz for their B787.
     Their Fig. 6b is the average of 151 virtual flights of 100 s each. This
     limb is that protocol on this project's aircraft and this project's only
     stochastic field: N virtual flights through `wind.dryden_field` at the
     SOURCED sigma range, spectra averaged over the ensemble.

     It settles the sealed prediction `the_dryden_response_peaks_at_the_short_
     period`, which was written before this script had ever been run.

  C  THE LOAD EXCEEDANCE CURVE. The same ensemble, counted rather than peaked:
     upcrossings per second of each load level, both signs, with N in the
     denominator. This is the statistic whose error bar shrinks.

WHAT THIS SCRIPT CANNOT DO, SAID HERE RATHER THAN DISCOVERED AT THE FIGURE. It
cannot overlay a PUBLISHED exceedance curve, because this project holds none --
TM-102186 gives Hannibal's recorded load as a two-number band and Wingrove &
Bach 1994's Table 2 gives twelve single incidents, neither of which is a rate.
Nor can it overlay a published SPECTRUM: no digitised acceleration history
exists here, and Yoshimura's own is in a figshare dataset the project does not
hold. So limb B compares the model against the aircraft's OWN dynamics, which
is the half of Yoshimura's protocol that is available, and the missing half is
named in PROJECT.md section 5 as an acquisition rather than quietly dropped.

Run: PYTHONPATH=<abs worktree root> .venv/Scripts/python.exe scripts/cat_spectra.py --outdir runs/cat
"""

import argparse
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import atisim  # noqa: F401  -- enables x64
from atisim import predictions, response, trim, validation, vortex_viz, wind
from atisim.aircraft import CRUISE, REGISTRY

PALETTE = {
    "model": "#1D5D77", "reference": "#A9501C", "wind": "#3E6A48",
    "muted": "#7E8D93", "grid": "#D6DCD8",
}

AIRCRAFT = "boeing747"
MEHTA_LEAD_R0 = 12.0  # r0, scripts/cat_bounds.py's lead-in and its reasoning

# DECLARED, not sourced. The Dryden runs start in equilibrium with the
# translational wind but not with its gradient (see `fly_in_moving_air`), so the
# first seconds carry a start transient that belongs to the initial condition
# rather than to the turbulence. Discarding a fixed head of every record is the
# cheapest defence; 20 s is about three short-period periods at this condition.
SETTLE_SECONDS = 20.0

# Yoshimura's own record length, Fig. 6: "computed for 100 s for each wind
# field". Kept because it is the source's, and because it is comfortably under
# one phugoid period here (114 s) -- a longer record would let the phugoid
# develop and make the record non-stationary in exactly the band the peak search
# is trying to step over.
RECORD_SECONDS = 100.0

YOSHIMURA_B787_HZ = 0.14  # their estimated natural frequency, for context only


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


def short_period(ac, V, H):
    """(f_hz, zeta) of the aircraft's own short period at this condition."""
    x, _ = trim.trim(jnp.array(V), jnp.array(H), ac)
    modes = validation.longitudinal_modes(
        ac, float(x[0]), float(x[1]), float(x[2]), V, H)
    wn, zeta = modes[-1]  # sorted low-to-high wn; the short period is the high one
    return float(wn) / (2.0 * np.pi), float(zeta)


def fly_mehta(ac, V, H, dt):
    """Limb A's run -- the section-4 headline, unchanged."""
    array = wind.mehta_hannibal_array(H)
    field = lambda p: wind.vortex_wind(p, array)  # noqa: E731
    r0 = float(array.r0)
    x0, x1 = float(array.north.min()), float(array.north.max())
    start = x0 - MEHTA_LEAD_R0 * r0
    seconds = (x1 + MEHTA_LEAD_R0 * r0 - start) / V
    enc = vortex_viz.fly_in_moving_air(
        ac, field, V, H, label="mehta", start_north=start, seconds=seconds,
        dt=dt, window=(x0 - 2.0 * r0, x1 + 2.0 * r0), window_name="array",
    )
    return enc, array


def fly_dryden(ac, V, H, sigma, seed, dt):
    """One virtual flight through a frozen Dryden realisation.

    `dryden_field` rather than `dryden_vertical_field`: the three-component form
    landed in session 24 and this is the first run to use it, so the lateral
    channels see a real input. It still has no spanwise variation -- its own
    docstring says so -- which is why this limb is a LONGITUDINAL statistic.
    """
    field = wind.dryden_field(sigma, seed)
    seconds = SETTLE_SECONDS + RECORD_SECONDS
    return vortex_viz.fly_in_moving_air(
        ac, field, V, H, label=f"dryden-{seed}", start_north=0.0,
        seconds=seconds, dt=dt,
        # The whole record is the analysis window here; there is no core to
        # centre on. `_measure` needs one, so it gets the run.
        window=(-1e12, 1e12), window_name="whole run",
    )


def tail(enc, dt):
    """The record with the start transient dropped, as a plain array."""
    return np.asarray(enc.n_z)[int(round(SETTLE_SECONDS / dt)):]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=Path, default=Path("runs/cat"))
    p.add_argument("--seeds", type=int, default=32)
    p.add_argument("--dt", type=float, default=0.02)
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    print(f"atisim imported from: {atisim.__file__}\n")

    ac = REGISTRY[AIRCRAFT]
    V = CRUISE[AIRCRAFT]["airspeed"]
    H = wind.MEHTA_HANNIBAL_ALTITUDE
    f_sp, zeta_sp = short_period(ac, V, H)
    lo_s, hi_s = wind.mehta_unmodelled_wind(), wind.mehta_residual_ceiling()

    print("=" * 78)
    print("RESPONSE SPECTRA AND LOAD EXCEEDANCE -- phase 2")
    print("=" * 78)
    print(f"  {AIRCRAFT} at {H:.0f} m, {V:.1f} m/s, dt {args.dt} s")
    print(f"  short period: f = {f_sp:.4f} Hz (wn = {f_sp * 2 * np.pi:.4f} "
          f"rad/s), zeta = {zeta_sp:.4f}")
    print(f"  peak searched above response.PHUGOID_FLOOR_HZ = "
          f"{response.PHUGOID_FLOOR_HZ} Hz (DECLARED)\n")

    # -- A. the headline run as a spectrum ----------------------------------
    print("-" * 78)
    print("A. MEHTA'S HANNIBAL FIELD, AS A SPECTRUM")
    print("-" * 78)
    enc, array = fly_mehta(ac, V, H, args.dt)
    n_z = np.asarray(enc.n_z)
    f_a, psd_a = response.spectrum(n_z, args.dt)
    peak_a = response.peak_frequency(f_a, psd_a)
    duration = (n_z.size - 1) * args.dt
    spacings = np.diff(np.asarray(array.north))
    passage = V / spacings

    print(f"  record {duration:.1f} s, {n_z.size} samples, "
          f"bin {f_a[1] - f_a[0]:.4f} Hz")
    print(f"  the array forces at FOUR frequencies, not one:")
    for s, hz in zip(spacings, passage):
        print(f"    {s:8.1f} m core spacing -> {V / hz:5.2f} s -> {hz:.4f} Hz")
    print(f"  mean core passage: {passage.mean():.4f} Hz")
    print(f"  response peak:     {peak_a:.4f} Hz")
    print(f"    vs short period  {f_sp:.4f} Hz  ->  {100 * (peak_a / f_sp - 1):+.1f}%")
    print(f"    vs mean passage  {passage.mean():.4f} Hz  ->  "
          f"{100 * (peak_a / passage.mean() - 1):+.1f}%")
    print("  RESOLUTION CAVEAT: the two candidates are "
          f"{abs(f_sp - passage.mean()) / (f_a[1] - f_a[0]):.1f} bins apart on "
          "this record.")
    print("  A transient of a few cycles cannot resolve them better than that,")
    print("  which is why limb B uses a stationary field and an ensemble.\n")

    # -- B. Yoshimura's protocol --------------------------------------------
    print("-" * 78)
    print("B. YOSHIMURA'S PROTOCOL: N VIRTUAL FLIGHTS, SPECTRA AVERAGED")
    print("-" * 78)
    print(f"  {args.seeds} flights x {RECORD_SECONDS:.0f} s through "
          f"wind.dryden_field (Yoshimura: 151 x 100 s)")
    print(f"  first {SETTLE_SECONDS:.0f} s of each record discarded (DECLARED)")
    print(f"  sigma over the SOURCED range {lo_s:.3f}-{hi_s:.3f} m/s "
          "(mehta_unmodelled_wind .. mehta_residual_ceiling)\n")

    ensembles = {}
    for sigma in (lo_s, hi_s):
        records = [tail(fly_dryden(ac, V, H, sigma, seed, args.dt), args.dt)
                   for seed in range(args.seeds)]
        spectra = [response.spectrum(r, args.dt) for r in records]
        f_b = spectra[0][0]
        mean_psd = np.mean([s[1] for s in spectra], axis=0)
        peak_b = response.peak_frequency(f_b, mean_psd)
        peaks = np.array([response.peak_frequency(f_b, s[1]) for s in spectra])
        rms = np.array([r.std() for r in records])
        ensembles[sigma] = dict(f=f_b, psd=mean_psd, peak=peak_b,
                                peaks=peaks, records=records, rms=rms)
        print(f"  sigma = {sigma:.3f} m/s")
        print(f"    ensemble-averaged peak   {peak_b:.4f} Hz  "
              f"({100 * (peak_b / f_sp - 1):+.1f}% of the short period)")
        print(f"    per-flight peaks         median {np.median(peaks):.4f} Hz, "
              f"[{peaks.min():.4f}, {peaks.max():.4f}]")
        print(f"    n_z rms                  {rms.mean():.4f} g "
              f"[{rms.min():.4f}, {rms.max():.4f}]")

    # The sealed prediction, settled here.
    pred = predictions.BY_NAME["the_dryden_response_peaks_at_the_short_period"]
    band = (0.131, 0.197)
    inside = {s: band[0] <= e["peak"] <= band[1] for s, e in ensembles.items()}
    print(f"\n  SEALED PREDICTION `{pred.name}` (sealed at {pred.sealed_at})")
    print(f"    predicted band {band[0]}-{band[1]} Hz, i.e. +/-20% of "
          f"{f_sp:.4f} Hz")
    for sigma, ok in inside.items():
        print(f"    sigma {sigma:.3f}: peak {ensembles[sigma]['peak']:.4f} Hz "
              f"-> {'INSIDE' if ok else 'OUTSIDE -- FALSIFIED'}")
    print(f"    Yoshimura's B787 sat at {YOSHIMURA_B787_HZ} Hz; this 747 sits "
          f"at {f_sp:.4f} Hz.")
    print("    Different aircraft, same statement: the airframe picks the")
    print("    frequency, and the turbulence supplies the energy.\n")

    # -- C. the exceedance curve --------------------------------------------
    print("-" * 78)
    print("C. LOAD EXCEEDANCE OVER THE ENSEMBLE")
    print("-" * 78)
    levels = np.arange(0.02, 0.42, 0.02)  # g of increment above/below 1 g
    curves = {}
    for sigma, e in ensembles.items():
        pooled = np.concatenate(e["records"])
        n = len(e["records"])
        seconds = sum((r.size - 1) for r in e["records"]) * args.dt
        # Every record is the same length, so the ensemble rate is the mean of
        # the per-record rates -- which is what puts the N in the denominator.
        up = np.mean([response.exceedance(r, args.dt, 1.0 + levels)
                      for r in e["records"]], axis=0)
        # The negative side is the upcrossings of the REFLECTED history against
        # the reflected levels: dn < -y happens exactly when -n_z upcrosses
        # -(1 - y). Reflecting rather than writing a second counter keeps one
        # tested primitive doing both jobs.
        down = np.mean([response.exceedance(-r, args.dt, -(1.0 - levels))
                        for r in e["records"]], axis=0)
        curves[sigma] = (up, down)
        print(f"  sigma = {sigma:.3f} m/s   "
              f"{n} flights, {seconds:.0f} s of record, "
              f"peak |dn| {np.abs(pooled - 1.0).max():.4f} g")
        print(f"    {'level, g':>10}{'up, /s':>12}{'down, /s':>12}"
              f"{'up, /1000 km':>15}")
        for i, lv in enumerate(levels):
            if up[i] == 0 and down[i] == 0:
                break
            print(f"    {lv:>10.2f}{up[i]:>12.5f}{down[i]:>12.5f}"
                  f"{up[i] * 1e6 / V:>15.2f}")
        both = (up > 0) & (down > 0)
        if both.any():
            ratio = up[both] / down[both]
            print(f"    up/down ratio where both are non-zero: "
                  f"{ratio.min():.3f} to {ratio.max():.3f}")
        print("    Section 5: linear lift is exactly odd-symmetric in the")
        print("    gust, so a ratio away from 1 is the trim state and the")
        print("    finite sample, not an asymmetry the model can represent.\n")

    print("  WHAT IS AND IS NOT ESTABLISHED BY THIS.")
    print("  The curve is the model's own, with N in the denominator, and it")
    print("  is the first statistic here that improves with more flying. It is")
    print("  NOT yet a comparison: no published exceedance curve is held. The")
    print("  document that would make it one is named in PROJECT.md section 5.")

    figure(args.outdir / "09-spectra.png", f_a, psd_a, peak_a, ensembles,
           curves, levels, f_sp, passage, args.seeds)
    print(f"\nfigure -> {args.outdir / '09-spectra.png'}")


def figure(path, f_a, psd_a, peak_a, ensembles, curves, levels, f_sp, passage,
           seeds):
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))

    ax = axes[0]
    ax.loglog(f_a[1:], psd_a[1:], color=PALETTE["model"], lw=1.0)
    ax.axvline(f_sp, color=PALETTE["reference"], lw=1.2, ls="--",
               label=f"short period {f_sp:.3f} Hz")
    for hz in passage:
        ax.axvline(hz, color=PALETTE["wind"], lw=0.8, ls=":", alpha=0.8)
    ax.axvline(passage.mean(), color=PALETTE["wind"], lw=1.2, ls="-.",
               label="core passage (4 spacings)")
    ax.plot([peak_a], [psd_a[np.argmin(np.abs(f_a - peak_a))]], "o",
            color=PALETTE["model"], ms=6, label=f"peak {peak_a:.3f} Hz")
    _style(ax, "frequency, Hz", "$n_z$ PSD, g$^2$/Hz",
           "A. Mehta's array, one transient")
    ax.set_xlim(0.01, 5.0)
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")

    ax = axes[1]
    for (sigma, e), style in zip(ensembles.items(), ("-", "--")):
        ax.loglog(e["f"][1:], e["psd"][1:], style, color=PALETTE["model"],
                  lw=1.1, label=f"$\\sigma$ = {sigma:.2f} m/s, "
                                f"peak {e['peak']:.3f} Hz")
    ax.axvline(f_sp, color=PALETTE["reference"], lw=1.2, ls="--",
               label=f"short period {f_sp:.3f} Hz")
    ax.axvspan(0.131, 0.197, color=PALETTE["reference"], alpha=0.10,
               label="sealed prediction band")
    _style(ax, "frequency, Hz", "$n_z$ PSD, g$^2$/Hz",
           f"B. Dryden ensemble, {seeds} flights averaged")
    ax.set_xlim(0.01, 5.0)
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")

    ax = axes[2]
    for (sigma, (up, down)), style in zip(curves.items(), ("-", "--")):
        ax.semilogy(levels, np.where(up > 0, up, np.nan), style,
                    color=PALETTE["model"], lw=1.2,
                    label=f"$\\sigma$ = {sigma:.2f} m/s, up")
        ax.semilogy(levels, np.where(down > 0, down, np.nan), style,
                    color=PALETTE["wind"], lw=1.2,
                    label=f"$\\sigma$ = {sigma:.2f} m/s, down")
    _style(ax, "load increment $|\\Delta n|$, g", "upcrossings per second",
           "C. Exceedance, both signs")
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")

    fig.suptitle("Response spectra and load exceedance -- statistics with an N "
                 "in the denominator", fontsize=11, fontweight="bold",
                 x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
