"""V4 -- bounding the unsteady lag on the GUST's arrival.

Design: docs/design/specs/2026-09-21-turbulence-response-validation-design.md,
phase V4. It exists because of a gap that survived twelve sessions of assumption
auditing:

    `docs/ASSUMPTIONS.md` C2 is titled "Aerodynamics are quasi-steady: no alpha-dot
    OR UNSTEADY LAG" and calls itself the best-characterised assumption in the
    project -- and every number in its bound is about alpha-dot, the lag on the
    aircraft's OWN motion, which `Cmadot` carries. The lag on the GUST's arrival
    was bounded nowhere, and `Sears`, `Kussner`, `Wagner` and `Theodorsen`
    appeared nowhere in the tree.

They are two different problems. Theodorsen's is an aerofoil pitching and
plunging in still air; Sears' is an aerofoil held still while a sinusoidal gust
pattern convects past it. This project models the first and not the second.

WHAT IT IS WORTH, WHICH IS THE ONLY REASON TO RAISE IT. The attenuation acts to
REDUCE simulated load. `PROJECT.md` §5 records the model falling SHORT of the
Hannibal encounter, so this term widens the shortfall it would have to explain
rather than closing it -- which is the direction that says it is not being
raised for convenience. Limb C prices that directly against the turbulence
result, not just at a frequency.

Run: PYTHONPATH=<abs worktree root> .venv/Scripts/python.exe scripts/gust_lag_bound.py
"""

import numpy as np

import atisim  # noqa: F401  -- enables x64
from atisim import gust, trim, validation, wind
from atisim.aircraft import REGISTRY
from atisim.atmosphere import speed_of_sound
import jax.numpy as jnp

AIRCRAFT = "boeing747"
MACH = 0.80


def condition():
    H = float(wind.MEHTA_HANNIBAL_ALTITUDE)
    return REGISTRY[AIRCRAFT], MACH * float(speed_of_sound(H)), H


def main():
    ac, V, H = condition()
    c = float(ac.c)
    x, _ = trim.trim(jnp.array(V), jnp.array(H), ac)
    modes = validation.longitudinal_modes(
        ac, float(x[0]), float(x[1]), float(x[2]), V, H)
    wn_sp = float(modes[-1][0])
    r0 = float(wind.PARKS_CASES["hannibal"]["r0"])

    print(f"{AIRCRAFT} at M {MACH}, {H / 0.3048:.0f} ft: "
          f"V = {V:.3f} m/s, c = {c:.4f} m")
    print("\nA  Sears' function at the frequencies this project actually forces")
    print(f"{'what':38} {'omega (rad/s)':>14} {'k':>9} {'|S|':>8} "
          f"{'arg S':>9} {'lift lost':>10}")
    cases = [
        ("short period", wn_sp),
        ("Parks core passage, V/r0", V / r0),
        ("Dryden peak response (0.187 Hz)", 2.0 * np.pi * 0.1874),
        ("Dryden scale length, V/L_w", V / float(wind.DRYDEN_LW)),
        ("shortest component, 20 m", 2.0 * np.pi * V / 20.0),
    ]
    for name, omega in cases:
        k = gust.reduced_frequency(omega, c, V)
        S = gust.sears(k)
        print(f"{name:38} {omega:14.4f} {k:9.5f} {abs(S):8.4f} "
              f"{np.degrees(np.angle(S)):8.3f}d {100 * (1 - abs(S)):9.2f}%")

    print("\nB  what the attenuation is against, for scale")
    print("   ASSUMPTIONS E2 prices the POINT-GUST approximation at 4.4% on the")
    print("   same encounter. The gust lag is the same order and was unbounded.")

    print("\nC  the same term priced on the turbulence result rather than at a frequency")
    lo = 2.0 * np.pi / 40_000.0
    hi = 2.0 * np.pi / 20.0
    Omega = np.geomspace(lo, hi, 4000)
    Phi = np.asarray(wind.dryden_spectrum(jnp.asarray(Omega), 1.0), dtype=float)
    Hmag2 = np.abs(gust.gust_transfer(ac, V, H, Omega)) ** 2
    Smag2 = np.abs(gust.sears(gust.reduced_frequency(Omega * V, c, V))) ** 2
    bare = np.trapezoid(Hmag2 * Phi, Omega)
    lagged = np.trapezoid(Smag2 * Hmag2 * Phi, Omega)
    print(f"   Abar without Sears            {np.sqrt(bare):.6f} g per (m/s)")
    print(f"   Abar with Sears applied to H  {np.sqrt(lagged):.6f} g per (m/s)")
    print(f"   sigma_nz would fall by        {100 * (1 - np.sqrt(lagged / bare)):.2f}%")
    print("\n   That is a BOUND, not a correction: applying |S| to |H| is the")
    print("   attenuation a thin aerofoil would suffer, and this is a whole")
    print("   aircraft with a tail in the same gust. It says the size of the")
    print("   term and its SIGN, which is what C12 needed and did not have.")


if __name__ == "__main__":
    main()
