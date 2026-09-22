"""Does the gust response attenuation actually need a 757 pitch set?

WHY THIS EXISTS. PROJECT.md section 4 records that a B-757 built on a borrowed
frame moves 32% depending on whose frame, and concluded the test needed "a 757
pitch set and inertia". A search for one came up empty -- NTRS, this project's
local reference library and JSBSim's 61 shipped models all have nothing; the
search is recorded in section 4. So before treating that as a blocking
acquisition, this asks whether a pitch set is what the 32% was made of.

A  PITCH SENSITIVITY. Cma, Cmq and Iyy are perturbed +/-30% one at a time on
   `boeing747`, which is FULLY SOURCED, so nothing here rests on a borrowed
   field. If a +/-30% error moves the attenuation by only a few percent, the
   pitch set is not the acquisition it looked like, and the 32% came from the
   rest of the assembly -- CL0, Cm0, the drag polar, the engine, the trim
   state -- rather than from the pitch derivatives.

B  DOES THE ATTENUATION COLLAPSE ONTO THE PLUNGE TIME CONSTANT? tau =
   1/(g * quasi) is swept by flying the two sourced aircraft across their own
   declared bands. If both land on ONE curve, then the measured B-757 -- whose
   tau is computed from fully sourced quantities -- can be placed on it with no
   757 entry, no borrowed dynamics and no DECLARED structure at all. That is
   the frame-independent version of the test section 4 could not settle.

WHAT THIS CANNOT DO, stated because the answer is tempting. The model's
declared bands do not reach the B-757's tau of 1.18 s: the lowest reachable
here is `boeing737` at 25 kft, tau 1.72 s, and going lower means leaving the
band `checks.recovery_band` exists to enforce. Reading the curve at 1.18 s is
EXTRAPOLATION, which section 7(e) says to refuse. THE GAP IS THE RESULT, not an
inconvenience to be interpolated away.

A NOTE ON WHAT `CLa` MEANS HERE. `Aircraft.pg_mach_ref` is -1.0 on every
registry entry, which the code reads as undeclared and makes the
Prandtl-Glauert factor exactly 1. So both aircraft below fly at their raw lift
slope with no compressibility correction, and the quasi-steady denominator used
here is the one the model actually flies. Section 4 records that machinery as
PRESENT AND INERT; what is new is that Stewart's Table 1 now prices what being
inert costs.

Run: PYTHONPATH=<abs worktree root> .venv/Scripts/python.exe -u \\
         scripts/pitch_set_requirement.py
"""

import numpy as np

import atisim  # noqa: F401  -- enables x64

print(f"atisim imported from: {atisim.__file__}")

import jax.numpy as jnp  # noqa: E402

from atisim import vortex_viz, wind  # noqa: E402
from atisim.aircraft import REGISTRY  # noqa: E402
from atisim.atmosphere import density, speed_of_sound  # noqa: E402

FT2M, G = 0.3048, 9.80665
DT, SECONDS, HEAD = 0.02, 1200.0, 100.0
SEEDS = range(3)

# Each aircraft is swept only inside its OWN declared band, per rule 6 and
# because `checks.recovery_band` is what makes a derivative set mean anything.
SWEEP = (("boeing747", ((35.0, 0.72), (37.0, 0.80), (43.0, 0.86))),
         ("boeing737", ((25.0, 0.70), (30.0, 0.75), (35.0, 0.85))))
MEASURED_757 = (1.18, 0.700, 0.830)  # tau, and the bracket from section 4


def run(ac, alt_kft, mach, sigma_w):
    """Plunge time constant, sigma_nz/sigma_w, and the attenuation between."""
    H = alt_kft * 1000.0 * FT2M
    V = mach * float(speed_of_sound(H))
    sn, sw = [], []
    for seed in SEEDS:
        enc = vortex_viz.fly_in_moving_air(
            ac, wind.dryden_vertical_field(sigma_w, seed), V, H,
            label=f"{alt_kft:.0f}kft", start_north=0.0, seconds=SECONDS,
            dt=DT, window=(-np.inf, np.inf), window_name="whole run",
            stage_sampled=True)
        keep = enc.t >= HEAD
        d = np.asarray(enc.n_z[keep])
        w = np.asarray(enc.w_up[keep])
        sn.append(float((d - d.mean()).std()))
        sw.append(float((w - w.mean()).std()))
    qz = (float(density(H)) * V * float(ac.S) * float(ac.CLa)
          / (2.0 * float(ac.mass) * G))
    ratio = float(np.mean(sn) / np.mean(sw))
    return 1.0 / (G * qz), ratio, ratio / qz


def main():
    sigma_w = float(wind.mehta_residual_ceiling())
    base = REGISTRY["boeing747"]
    tau0, _, k0 = run(base, 37.0, 0.80, sigma_w)

    print(f"\nA  PITCH SENSITIVITY on boeing747, fully sourced, "
          f"N = {len(list(SEEDS))}")
    print(f"   baseline: tau {tau0:.2f} s, attenuation {k0:.4f}\n")
    print(f"   {'perturbation':20} {'attenuation':>12} {'change':>9}")
    worst = 0.0
    for field in ("Cma", "Cmq"):
        for f in (0.7, 1.3):
            ac = base._replace(
                **{field: jnp.asarray(float(getattr(base, field)) * f)})
            _, _, k = run(ac, 37.0, 0.80, sigma_w)
            worst = max(worst, abs(k / k0 - 1.0))
            print(f"   {field} x{f:<15.1f} {k:12.4f} {100 * (k / k0 - 1):+8.1f}%")
    for f in (0.7, 1.3):
        inertia = np.asarray(base.inertia).copy()
        inertia[1, 1] *= f
        ac = base._replace(inertia=jnp.asarray(inertia),
                           inertia_inv=jnp.linalg.inv(jnp.asarray(inertia)))
        _, _, k = run(ac, 37.0, 0.80, sigma_w)
        worst = max(worst, abs(k / k0 - 1.0))
        print(f"   Iyy x{f:<15.1f} {k:12.4f} {100 * (k / k0 - 1):+8.1f}%")
    print(f"\n   WORST move from a +/-30% pitch error: {100 * worst:.1f}%, "
          f"against the 32% frame spread section 4 reports.")

    print(f"\nB  ATTENUATION AGAINST PLUNGE TIME CONSTANT, sourced aircraft "
          f"inside their own declared bands")
    print(f"   {'aircraft':12} {'alt':>5} {'M':>5} {'tau (s)':>8} "
          f"{'attenuation':>12}")
    pts = []
    for name, cases in SWEEP:
        for alt, mach in cases:
            tau, _, k = run(REGISTRY[name], alt, mach, sigma_w)
            pts.append((tau, k, name))
            print(f"   {name:12} {alt:5.0f} {mach:5.2f} {tau:8.2f} "
                  f"{k:12.4f}", flush=True)

    pts.sort()
    mono = all(b[1] >= a[1] - 0.02 for a, b in zip(pts, pts[1:]))
    mixed = len({n for _, _, n in pts}) > 1
    print(f"\n   ordered by tau: "
          + ", ".join(f"{t:.2f}->{k:.3f}" for t, k, _ in pts))
    print(f"   monotone in tau across BOTH aircraft: {mono}; "
          f"the two entries interleave: {mixed}")

    tau757, lo, hi = MEASURED_757
    print(f"\n   the measured B-757 sits at tau {tau757:.2f} s, attenuation "
          f"{lo:.3f}-{hi:.3f}")
    print(f"   the lowest tau reachable inside a declared band is "
          f"{pts[0][0]:.2f} s ({pts[0][2]})")
    print(f"   so the measured point is OUTSIDE the swept range by "
          f"{pts[0][0] - tau757:.2f} s and reading the curve there would be "
          f"EXTRAPOLATION. Not done.")


if __name__ == "__main__":
    main()
