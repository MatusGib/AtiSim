"""Phases S2 and S3: the headline CAT gust load, differentiated and then swept.

WHAT THIS MEASURES. `PROJECT.md` section 1 quotes the headline load as a bare
per cent of a recorded peak-to-peak. That number has never been separable into
"the model is wrong by 32%" and "the inputs are not known to 32%".
`scripts/cat_uncertainty.py` did the WIND half. This is the AIRCRAFT half: which
of the 747's own coefficients the load rests on, and how far the answer moves per
per cent of each.

THE RUN IS THE HEADLINE RUN, unchanged: `boeing747` through Mehta's five-core
Hannibal array at 37,000 ft, started in moving air at a 12 r0 lead, measured over
the array plus 2 r0 either side -- the same configuration
`scripts/cat_validation.py:fly_mehta` flies.

WHAT A PEAK-TO-PEAK DERIVATIVE IS, AND IS NOT. `max - min` is differentiable
almost everywhere and not everywhere: its gradient is the gradient of the two
samples that happen to be the extremes, and it JUMPS when an extreme moves from
one core to another. Section C measures how far the tangent survives rather than
assuming it does, and reports every excursion at which an extreme moved. That is
the whole reason phase S3 exists and is not an optional confirmation of S2.

Run: PYTHONPATH=<abs worktree root> python scripts/sensitivity_load.py
"""

import argparse
import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import atisim  # noqa: F401  -- enables x64
from atisim import sensitivity, vortex_viz, wind
from atisim.aircraft import CRUISE, REGISTRY

# TM-102186's recorded peak-to-peak for the Hannibal encounter, -1.0 to +1.7 g.
# `wind.TM102186_HANNIBAL_NZ` carries the pair; this is the span it implies.
RECORDED_PEAK_TO_PEAK = 2.70

# The excursions section C sweeps. Chosen to bracket the range over which a
# published derivative might plausibly be wrong: 1% is a transcription, 25% is a
# different aeroplane.
EXCURSIONS = (0.01, 0.05, 0.10, 0.25)


def mehta_setup(dt: float = 0.01, lead_r0: float = 12.0):
    """The headline run's configuration, read off `cat_validation.fly_mehta`."""
    ac = REGISTRY["boeing747"]
    V = CRUISE["boeing747"]["airspeed"]
    H = wind.MEHTA_HANNIBAL_ALTITUDE
    array = wind.mehta_hannibal_array(H)

    def field(p):
        return wind.vortex_wind(p, array)

    r0 = float(array.r0)
    x0, x1 = float(array.north.min()), float(array.north.max())
    start = x0 - lead_r0 * r0
    seconds = (x1 + lead_r0 * r0 - start) / V
    return dict(ac=ac, V=V, H=H, array=array, field=field,
                model=wind.field_model(field), r0=r0,
                start_north=start, seconds=seconds, dt=dt,
                n_steps=int(round(seconds / dt)),
                window=(x0 - 2.0 * r0, x1 + 2.0 * r0))


def section_a(cfg, out):
    """S0's remaining gate: the differentiable path against `_measure`, sample for sample."""
    print("\n" + "=" * 78)
    print("A. THE GATE -- the differentiable load against vortex_viz._measure")
    print("=" * 78)
    print("The Encounter path ends in np.asarray, so this is a SECOND path to the")
    print("same number. Two paths that compute the same thing drift apart unless")
    print("something asserts they do not. The claim is bit-identity.\n")

    enc = vortex_viz.fly_in_moving_air(
        cfg["ac"], cfg["field"], cfg["V"], cfg["H"], label="mehta headline",
        start_north=cfg["start_north"], seconds=cfg["seconds"], dt=cfg["dt"],
        window=cfg["window"], window_name="the identified array, plus 2 r0 either side")

    n_z, north = sensitivity.load_history(
        cfg["ac"], cfg["model"], cfg["V"], cfg["H"],
        start_north=cfg["start_north"], n_steps=cfg["n_steps"], dt=cfg["dt"])
    n_z, north = np.asarray(n_z), np.asarray(north)

    identical = bool(np.array_equal(n_z, enc.n_z))
    worst = float(np.abs(n_z - enc.n_z).max())
    mask = (north >= cfg["window"][0]) & (north <= cfg["window"][1])
    ptp = float(n_z[mask].max() - n_z[mask].min())
    ref = float(enc.n_z[enc.window].max() - enc.n_z[enc.window].min())

    print(f"  samples                       {len(n_z)} over {cfg['seconds']:.2f} s at dt = {cfg['dt']}")
    print(f"  n_z worst absolute difference {worst:.3e}   bit-identical: {identical}")
    print(f"  window mask identical         {bool(np.array_equal(mask, enc.window))}")
    print(f"  peak-to-peak                  {ptp:.12f} g   (_measure: {ref:.12f} g)")
    print(f"  against TM-102186's {RECORDED_PEAK_TO_PEAK} g   {100*ptp/RECORDED_PEAK_TO_PEAK:.2f}%")
    print(f"\n  GATE: {'PASS -- bit-identical' if identical else 'FAIL'} (requirement was 1e-12)")
    out["gate"] = dict(bit_identical=identical, worst=worst, peak_to_peak=ptp,
                       per_cent_of_record=100*ptp/RECORDED_PEAK_TO_PEAK)
    return ptp


def section_b(cfg, out):
    """S2: the AD screen over every independent coefficient, in one pass."""
    print("\n" + "=" * 78)
    print("B. THE SCREEN -- elasticity of the headline peak-to-peak load")
    print("=" * 78)

    t0 = time.time()
    Q, grad, elas, extra = sensitivity.load_elasticities(
        cfg["ac"], cfg["model"], cfg["V"], cfg["H"],
        start_north=cfg["start_north"], n_steps=cfg["n_steps"], dt=cfg["dt"],
        window=cfg["window"])
    took = time.time() - t0

    print(f"  Q = {Q:.9f} g, {100*Q/RECORDED_PEAK_TO_PEAK:.2f}% of the record."
          f"  {len(sensitivity.INDEPENDENT_FIELDS)} tangents in {took:.1f} s\n")
    print(f"  the two extremes sit at samples {extra['argmin']} and {extra['argmax']}"
          f" of {cfg['n_steps']},")
    print(f"  at north {extra['north'][extra['argmin']]:+.0f} m and"
          f" {extra['north'][extra['argmax']]:+.0f} m -- the gradient of a peak")
    print("  belongs to whichever sample IS the peak, so section C checks these move.\n")

    live = [(f, v) for f, v in elas.items() if np.isfinite(v) and abs(v) > 1e-12]
    live.sort(key=lambda kv: -abs(kv[1]))
    print(f"  {'field':15s} {'elasticity':>11s} {'gradient':>14s} {'base value':>14s}")
    for f, v in live:
        print(f"  {f:15s} {v:+11.5f} {grad[f]:+14.5e} {extra['values'][f]:+14.6g}")

    inert = sorted(f for f, v in elas.items() if np.isfinite(v) and abs(v) <= 1e-12)
    print(f"\n  EXACTLY ZERO ({len(inert)}): {', '.join(inert)}")
    print("  Every lateral derivative is here, and that is a MEASUREMENT of")
    print("  ASSUMPTIONS E10: Mehta's array is a function of along-track distance")
    print("  alone, so no lateral coefficient can reach the load. Not 'small' -- zero.")

    out["screen"] = dict(Q=Q, gradients=grad, elasticities=elas,
                         argmin=extra["argmin"], argmax=extra["argmax"],
                         values=extra["values"], seconds=took)
    return Q, elas, extra


def section_c(cfg, Q0, elas, extra, out, top_k=6):
    """S3: does the tangent survive a finite excursion, and do the extremes move?"""
    print("\n" + "=" * 78)
    print("C. THE CONFIRM -- how far the tangent survives, and where it breaks")
    print("=" * 78)
    print("A tangent predicts dQ/Q = elasticity * dp/p. Actual/predicted is 1.000")
    print("exactly where the local gradient is the whole answer. The `moved` column")
    print("is the one that matters for a PEAK: it says an extreme changed sample.\n")

    live = sorted([(f, v) for f, v in elas.items()
                   if np.isfinite(v) and abs(v) > 1e-6], key=lambda kv: -abs(kv[1]))
    fields = [f for f, _ in live[:top_k]]
    base_extremes = (extra["argmin"], extra["argmax"])

    print(f"  {'field':14s} {'excursion':>10s} {'Q, g':>10s} {'actual %':>10s}"
          f" {'tangent %':>10s} {'act/tan':>9s}  extremes")
    for f in fields:
        p0 = float(getattr(cfg["ac"], f))
        for frac in EXCURSIONS:
            for sgn in (+1.0, -1.0):
                swept = cfg["ac"]._replace(**{f: jnp.array(p0 * (1.0 + sgn * frac))})
                n_z, north = sensitivity.load_history(
                    swept, cfg["model"], cfg["V"], cfg["H"],
                    start_north=cfg["start_north"], n_steps=cfg["n_steps"],
                    dt=cfg["dt"])
                n_z, north = np.asarray(n_z), np.asarray(north)
                mask = extra["mask"]
                inside = np.where(mask, n_z, -np.inf)
                outside = np.where(mask, n_z, np.inf)
                Q = float(inside.max() - outside.min())
                extremes = (int(outside.argmin()), int(inside.argmax()))
                actual = 100.0 * (Q - Q0) / Q0
                tangent = 100.0 * elas[f] * sgn * frac
                ratio = actual / tangent if tangent != 0 else float("nan")
                moved = "" if extremes == base_extremes else \
                    f"MOVED {base_extremes}->{extremes}"
                out["confirm"].append(dict(field=f, frac=sgn*frac, Q=Q,
                                           actual_pct=actual, tangent_pct=tangent,
                                           ratio=ratio, extremes=extremes,
                                           moved=extremes != base_extremes))
                print(f"  {f:14s} {sgn*frac:+10.2%} {Q:10.5f} {actual:+10.4f}"
                      f" {tangent:+10.4f} {ratio:9.4f}  {moved}")
        print()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--top", type=int, default=6)
    args = ap.parse_args()

    print(f"atisim imported from: {atisim.__file__}")
    print("PROJECT.md section 10: if that is not the tree you edited, stop here.")

    cfg = mehta_setup(dt=args.dt)
    print(f"\nthe headline run: boeing747, Mehta five-core Hannibal array at "
          f"{cfg['H']:.0f} m,\n  {cfg['V']:.2f} m/s, {cfg['seconds']:.2f} s, "
          f"{cfg['n_steps']} steps, window "
          f"{cfg['window'][0]:.0f}..{cfg['window'][1]:.0f} m")

    out = {"tree": atisim.__file__, "confirm": [], "dt": args.dt}
    Q0 = section_a(cfg, out)
    Q, elas, extra = section_b(cfg, out)
    section_c(cfg, Q, elas, extra, out, top_k=args.top)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=1, default=float))
        print(f"written: {args.json}")


if __name__ == "__main__":
    main()
