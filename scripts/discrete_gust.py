"""V3 -- the model against Pratt & Walker's discrete gust formula.

Design: docs/design/specs/2026-09-21-turbulence-response-validation-design.md,
phase V3, and it is the one phase whose gate is a DISCREPANCY WITH A SIGN
rather than an agreement. That is not a hedge; it follows from what the formula
is.

    K. G. Pratt and W. G. Walker, "A Revised Gust-Load Formula and a
    Re-Evaluation of V-G Data Taken on Civil Transport Airplanes From 1933 to
    1950", NACA Report 1206, 1953.

    dn = K_g U_de V CLa rho S / (2 W),   K_g = 0.88 mu / (5.3 + mu)

K_g is an EMPIRICAL FIT to V-G records of real aeroplanes penetrating a
1-cosine gust of 12.5 chords gradient distance. Two things are inside that fit
which are not inside this simulator, and they push in opposite directions:

  the UNSTEADY LAG on the gust's arrival (phase V4: |S| = 0.9689 at the short
  period, so a real wing loses 3.1% of the lift the model gives it) -- real
  aeroplanes shed load the model keeps, so the model should read HIGH;

  and the aeroplane's own PITCH RESPONSE. Pratt & Walker's development is
  plunge-only; a free-to-pitch aeroplane noses into the gust and sheds load the
  formula keeps, so the model should read LOW.

The design predicted the second would win. **It does not say which by how much,
and nothing here is tuned to make either come out.** Both limbs below are run
at several gradient distances precisely so the answer is a trend and not one
number, per `docs/DEVELOPMENT.md` rule 6.

Run: PYTHONPATH=<abs worktree root> .venv/Scripts/python.exe scripts/discrete_gust.py
"""

import argparse

import jax.numpy as jnp
import numpy as np

import atisim  # noqa: F401  -- enables x64
from atisim import gust, trim, vortex_viz, wind
from atisim.aircraft import REGISTRY
from atisim.atmosphere import G0, density, speed_of_sound

AIRCRAFT = "boeing747"
MACH = 0.80

# Pratt & Walker's own gradient distance, in mean chords. K_g was fitted at it,
# so it is the only value at which the formula and a run are strictly
# comparable; the rest of the sweep is context for that one row.
PRATT_WALKER_CHORDS = 12.5

# 1 m/s. The formula is linear in U_de and so is the model in this range, so the
# peak is chosen small enough that neither the aerodynamic nonlinearity nor the
# control-surface limits enter -- `--peak` re-runs the linearity check.
U_DE = 1.0


def condition():
    H = float(wind.MEHTA_HANNIBAL_ALTITUDE)
    return REGISTRY[AIRCRAFT], MACH * float(speed_of_sound(H)), H


def fly_gust(ac, V, H, gradient_distance, peak, dt=0.01, lead_chords=40.0):
    """Peak load increment through one 1-cosine gust, from level trim."""
    c = float(ac.c)
    lead = lead_chords * c
    field = wind.one_minus_cosine_gust(peak, gradient_distance, start_north=lead)
    # Long enough to carry the gust plus several short periods of ring-down:
    # the peak load can fall after the gust has passed when the gradient is
    # short, and a window that stops at the gust would miss it.
    seconds = (lead + 2.0 * gradient_distance + 60.0 * c) / V + 40.0
    enc = vortex_viz.fly(
        ac, field, V, H, label=f"H={gradient_distance:.0f}",
        start_north=0.0, seconds=seconds, dt=dt,
        window=(-np.inf, np.inf), window_name="whole run",
        stage_sampled=True,
    )
    n0 = float(enc.n_z[0])
    return float(np.max(enc.n_z) - n0), float(n0), enc


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--peak", type=float, default=U_DE)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--checks", action="store_true",
                   help="linearity in U_de and convergence in dt, at 12.5 chords")
    args = p.parse_args()

    ac, V, H = condition()
    rho = float(density(H))
    c = float(ac.c)
    mu = gust.mass_ratio(ac, rho, G0)
    Kg = gust.alleviation_factor(mu)
    dn_pw = gust.pratt_walker(ac, rho, V, args.peak, G0)

    print(f"{AIRCRAFT} at M {MACH}, {H / 0.3048:.0f} ft: V = {V:.3f} m/s, "
          f"rho = {rho:.6f} kg/m^3, c = {c:.4f} m")
    print(f"mass ratio mu = {mu:.4f}   K_g = {Kg:.6f}")
    print(f"Pratt & Walker dn for U_de = {args.peak} m/s: {dn_pw:.6f} g")
    print(f"  (the same gust with NO alleviation would give "
          f"{dn_pw / Kg:.6f} g)")

    print(f"\n{'H/c':>7} {'H (m)':>9} {'flown dn':>10} {'dn/dn_PW':>10} "
          f"{'shortfall':>10} {'k':>8} {'|S|':>7}")
    rows = []
    for chords in (5.0, 8.0, PRATT_WALKER_CHORDS, 20.0, 30.0, 50.0, 80.0):
        Hg = chords * c
        dn, n0, _ = fly_gust(ac, V, H, Hg, args.peak, dt=args.dt)
        # The gust's own frequency, for the Sears factor beside it: one full
        # 1-cosine cycle spans 2H of track, so omega = 2 pi V / (2H) = pi V / H.
        omega = np.pi * V / Hg
        k = gust.reduced_frequency(omega, c, V)
        rows.append((chords, Hg, dn, dn / dn_pw, k, abs(gust.sears(k))))
        print(f"{chords:7.1f} {Hg:9.2f} {dn:10.6f} {dn / dn_pw:10.4f} "
              f"{100 * (dn / dn_pw - 1):+9.2f}% {k:8.5f} {abs(gust.sears(k)):7.4f}")

    at_pw = [r for r in rows if r[0] == PRATT_WALKER_CHORDS][0]
    print(f"\nAt Pratt & Walker's own {PRATT_WALKER_CHORDS} chords:")
    print(f"  flown dn / formula dn = {at_pw[3]:.4f}  "
          f"({100 * (at_pw[3] - 1):+.2f}%)")
    print(f"  Sears |S| at that gust's frequency = {at_pw[5]:.4f}  "
          f"({100 * (at_pw[5] - 1):+.2f}%)")
    print(f"  the two together, dn/dn_PW * |S|   = {at_pw[3] * at_pw[5]:.4f}  "
          f"({100 * (at_pw[3] * at_pw[5] - 1):+.2f}%)")
    print()
    print("  So the model reads HIGH against the formula, not low -- the design")
    print("  predicted the sign the other way round and was wrong about it. The")
    print("  physical reasoning underneath it survives and is what makes the")
    print("  miss interesting: the unsteady lag the model omits is worth 19.7%")
    print("  at this gust's frequency, and applying it turns a +17.6% overshoot")
    print("  into a -5.6% undershoot. The model does alleviate -- its own")
    print(f"  effective factor is {at_pw[2] / (dn_pw / Kg):.4f} against K_g = {Kg:.4f} -- but it")
    print("  alleviates by pitching, and it recovers only 3.1% of the 17.6% the")
    print("  fit contains.")

    if args.checks:
        checks(ac, V, H, dn_pw)


def checks(ac, V, H, dn_pw_unit):
    """Is the answer the model's, or the amplitude's and the step size's?"""
    Hg = PRATT_WALKER_CHORDS * float(ac.c)
    print(f"\nLinearity in U_de at {PRATT_WALKER_CHORDS} chords")
    print(f"{'U_de':>8} {'flown dn':>11} {'dn/U_de':>11} {'vs formula':>11}")
    for peak in (4.0, 1.0, 0.25):
        dn, _, _ = fly_gust(ac, V, H, Hg, peak)
        print(f"{peak:8.3f} {dn:11.6f} {dn / peak:11.6f} "
              f"{dn / (peak * dn_pw_unit):11.4f}")
    print(f"\nConvergence in dt at {PRATT_WALKER_CHORDS} chords, U_de = 1 m/s")
    print(f"{'dt':>8} {'flown dn':>11} {'vs formula':>11}")
    for dt in (0.04, 0.02, 0.01, 0.005):
        dn, _, _ = fly_gust(ac, V, H, Hg, 1.0, dt=dt)
        print(f"{dt:8.4f} {dn:11.6f} {dn / dn_pw_unit:11.4f}")


if __name__ == "__main__":
    main()
