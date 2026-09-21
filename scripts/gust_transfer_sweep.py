"""V1 -- the gust-to-load transfer function, measured against an exact answer.

Design: docs/design/specs/2026-09-21-turbulence-response-validation-design.md,
phase V1. **It is the falsification step.** Every turbulence number this project
has published came out of one time-domain path -- trim, RK4 through a frozen
field, `wind.field_model` composing three gust channels, `vortex_viz` measuring
n_z -- and nothing had ever compared that path against a result computed a
different way. This script does, at twelve frequencies spanning two decades, in
amplitude AND phase.

Phase is not optional and the reason is specific: an amplitude-only comparison
passes with the sign of `Cmq` reversed, because the resonant gain depends on
|zeta| and the phase depends on its sign. Half of what this is looking for is
invisible without it.

The reference is `atisim.gust.gust_transfer`, which solves
`H = C (i omega I - A)^-1 B + D` on the linearisation of the same dynamics. It
is NOT an independent aerodynamic model and must not be quoted as one -- see
`atisim/gust.py`'s own docstring for what is and is not independent here.

Limbs:

  --sweep       twelve log-spaced frequencies, 0.01 to 1 Hz. The headline.
  --amplitudes  the same frequency at four gust amplitudes, which is how the
                claim "this is not limited by the aerodynamic nonlinearity"
                is checked rather than asserted.
  --refine      the same frequency at four step sizes, with and without the RK4
                wind hold. This is the limb that found PROJECT.md section 6(i).

Run: PYTHONPATH=<abs worktree root> .venv/bin/python scripts/gust_transfer_sweep.py --sweep
"""

import argparse

import numpy as np

import atisim  # noqa: F401  -- enables x64
from atisim import gust, wind
from atisim.aircraft import REGISTRY
from atisim.atmosphere import speed_of_sound

AIRCRAFT = "boeing747"

# The condition every CAT run in this project is flown at: Mehta's Hannibal
# altitude at M 0.80, which is inside `boeing747.valid_altitude` [35, 45] kft
# and inside `valid_mach` [0.70, 0.90]. Not CRUISE["boeing747"] (40,000 ft),
# so that V1 verifies the path at the condition the turbulence results use.
MACH = 0.80

# 0.01 to 1 Hz. The lower end is below the phugoid (0.0125 Hz) and the upper is
# six times the short period (0.164 Hz), so the sweep brackets both resonances
# and the quasi-steady tail beyond them.
SWEEP_HZ = np.geomspace(0.01, 1.0, 12)

# The gate. 0.5% in amplitude is the design's; the design did not say what 0.5%
# of a PHASE means, and a percentage of an angle depends on where the branch cut
# is put, so the phase gate is stated as an ABSOLUTE 0.5 deg here. That is a
# definition being pinned down before the measurement, not a tolerance being
# moved after one -- nothing had been run when it was written, and the measured
# phase errors came in four orders of magnitude inside it.
AMPLITUDE_GATE = 0.005   # fraction
PHASE_GATE_DEG = 0.5


def condition():
    H = float(wind.MEHTA_HANNIBAL_ALTITUDE)
    return REGISTRY[AIRCRAFT], MACH * float(speed_of_sound(H)), H


def _row(ac, V, H, f_hz, **kw):
    r = gust.measure_gust_transfer(ac, V, H, V / f_hz, **kw)
    theory = gust.gust_transfer(
        ac, V, H, r["spatial_frequency"], ground_speed=r["ground_speed"])
    r["theory"] = theory
    r["amplitude_error"] = abs(r["H"]) / abs(theory) - 1.0
    r["phase_error_deg"] = float(np.degrees(np.angle(r["H"] / theory)))
    return r


def _print(rows, title):
    print(f"\n{title}")
    print(f"{'f (Hz)':>8} {'lambda (m)':>11} {'|H| meas':>11} {'|H| theory':>11} "
          f"{'amp err':>10} {'phase err':>11} {'resid/amp':>10} {'cond':>8}")
    for r in rows:
        print(f"{r['frequency_hz']:8.4f} {r['wavelength']:11.1f} "
              f"{abs(r['H']):11.6f} {abs(r['theory']):11.6f} "
              f"{100 * r['amplitude_error']:+9.5f}% {r['phase_error_deg']:+10.5f}d "
              f"{r['load_residual'] / r['load_amplitude']:10.2e} "
              f"{r['load_cond']:8.1f}")


def _verdict(rows):
    worst_a = max(abs(r["amplitude_error"]) for r in rows)
    worst_p = max(abs(r["phase_error_deg"]) for r in rows)
    print(f"\n  worst amplitude error {100 * worst_a:.5f}% "
          f"against a {100 * AMPLITUDE_GATE:.1f}% gate")
    print(f"  worst phase error     {worst_p:.5f} deg "
          f"against a {PHASE_GATE_DEG:.1f} deg gate")
    passed = worst_a <= AMPLITUDE_GATE and worst_p <= PHASE_GATE_DEG
    print(f"  V1 {'PASSES' if passed else 'FAILS'}")
    return passed


def sweep(ac, V, H):
    rows = [_row(ac, V, H, f) for f in SWEEP_HZ]
    _print(rows, f"V1 sweep -- {AIRCRAFT}, M {MACH}, {H / 0.3048:.0f} ft")
    _verdict(rows)
    return rows


def amplitudes(ac, V, H, f_hz=0.1686):
    print(f"\nAmplitude sweep at {f_hz:.4f} Hz (the short period)")
    print(f"{'a (m/s)':>9} {'amp err':>11} {'phase err':>11} "
          f"{'d(alpha) p-p':>13} {'resid/amp':>10}")
    for a in (2.0, 0.5, 0.125, 0.03125):
        r = _row(ac, V, H, f_hz, amplitude=a)
        print(f"{a:9.5f} {100 * r['amplitude_error']:+10.5f}% "
              f"{r['phase_error_deg']:+10.5f}d {r['alpha_excursion_deg']:13.4f} "
              f"{r['load_residual'] / r['load_amplitude']:10.2e}")


def refine(ac, V, H, f_hz=0.1686):
    print(f"\nStep-size refinement at {f_hz:.4f} Hz, wind HELD vs STAGE-SAMPLED")
    print(f"{'dt (s)':>9} {'held amp err':>14} {'held phase':>12} "
          f"{'sampled amp err':>17} {'sampled phase':>15}")
    for spp in (50.0, 100.0, 200.0, 400.0):
        kw = dict(samples_per_period=spp, dt_max=1.0, dt_min=1e-4,
                  seed_steady=False)
        held = _row(ac, V, H, f_hz, stage_sampled=False, **kw)
        samp = _row(ac, V, H, f_hz, stage_sampled=True, **kw)
        print(f"{held['dt']:9.5f} {100 * held['amplitude_error']:+13.5f}% "
              f"{held['phase_error_deg']:+11.5f}d "
              f"{100 * samp['amplitude_error']:+16.5f}% "
              f"{samp['phase_error_deg']:+14.5f}d")
    print("\n  The held column HALVES with dt -- first order, which is the wind "
          "hold's\n  signature and not the response's. PROJECT.md section 6(i).")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--amplitudes", action="store_true")
    p.add_argument("--refine", action="store_true")
    args = p.parse_args()
    if not (args.sweep or args.amplitudes or args.refine):
        args.sweep = True

    ac, V, H = condition()
    print(f"{AIRCRAFT} at M {MACH}, {H / 0.3048:.0f} ft: V = {V:.3f} m/s")
    print(f"plant eigenvalues: {np.round(gust.plant_eigenvalues(ac, V, H), 6)}")
    if args.sweep:
        sweep(ac, V, H)
    if args.amplitudes:
        amplitudes(ac, V, H)
    if args.refine:
        refine(ac, V, H)


if __name__ == "__main__":
    main()
