"""V2 -- the PSD identity, and the three errors it separates.

Design: docs/design/specs/2026-09-21-turbulence-response-validation-design.md,
phase V2. V1 established that the simulator reproduces its own linearisation's
transfer function one frequency at a time. This asks the next question: does
the RANDOM-PROCESS statistic follow from the same H?

    sigma_nz^2 = integral |H(Omega)|^2 Phi_w(Omega) dOmega

Three things can go wrong in that one line and the design's point is that this
run separates them rather than reporting their sum:

  THE FACTOR OF TWO. `wind.dryden_spectrum` is ONE-SIDED. Integrating it as a
  two-sided spectrum inflates sigma_nz by sqrt(2) -- 41%, which is larger than
  any discrepancy this project is trying to explain, and would be attributed to
  physics. Nothing here can hide it: a 41% miss is not a tolerance question.

  THE LOG-GRID RIEMANN TRUNCATION. `dryden_vertical_field` is 400 components
  between 20 m and 40 km, and their summed power is 1.63% BELOW sigma_w^2 at
  the default settings. So a realisation does not carry the variance its
  argument names, and comparing a flown run against `Abar * sigma_w` charges
  the model for the grid's shortfall. `gust.realisation_mean_square_ratio`
  computes what THAT realisation exactly contains, and limb B compares against
  it instead.

  ENSEMBLE CONVERGENCE. One realisation is one sample. The gate is stated
  against the ensemble's own standard error with N beside it, per
  `docs/DEVELOPMENT.md` rule 6.

WHY THE VERTICAL-ONLY FIELD. `wind.dryden_field` carries all three components
and the u component moves n_z too, through dynamic pressure -- a channel H does
not contain. Feeding it here would compare a three-channel run against a
one-channel theory and call the difference an error.

Run: PYTHONPATH=<abs worktree root> .venv/bin/python scripts/gust_psd_identity.py
"""

import argparse

import numpy as np

import atisim  # noqa: F401  -- enables x64
from atisim import gust, vortex_viz, wind
from atisim.aircraft import REGISTRY
from atisim.atmosphere import speed_of_sound

AIRCRAFT = "boeing747"
MACH = 0.80

# 1200 s is seven periods of the 40 km longest component at this speed. Shorter
# records do not contain the low-wavenumber end of the field they were built
# from, so their sigma is a sample of a process the theory is not describing.
RECORD_SECONDS = 1200.0

# Discarded head. `fly_in_moving_air` starts the aircraft in equilibrium with
# the TRANSLATIONAL wind but not with its gradient, so a small start transient
# survives. Measured: moving this between 0 and 600 s moves sigma_nz by 0.3%,
# which is inside the ensemble's own spread -- so it is a declared choice that
# the answer is not sensitive to, and `--head` re-runs the check.
SETTLE_SECONDS = 100.0

# Stage-sampled, and legitimately so: a Dryden realisation is a FROZEN field,
# a pure function of position, so re-evaluating it per RK4 stage returns the
# same field rather than re-drawing one. PROJECT.md section 6(i) is why this is
# passed explicitly rather than left to the default.
DT = 0.02


def condition():
    H = float(wind.MEHTA_HANNIBAL_ALTITUDE)
    return REGISTRY[AIRCRAFT], MACH * float(speed_of_sound(H)), H


def fly_one(ac, V, H, sigma_w, seed, seconds, dt, head):
    field = wind.dryden_vertical_field(sigma_w, seed)
    enc = vortex_viz.fly_in_moving_air(
        ac, field, V, H, label=f"seed {seed}", start_north=0.0,
        seconds=seconds, dt=dt, window=(-np.inf, np.inf),
        window_name="whole run", stage_sampled=True,
    )
    keep = enc.t >= head
    return float(np.std(enc.n_z[keep])), float(np.std(enc.w_up[keep]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-n", "--seeds", type=int, default=24)
    p.add_argument("--seconds", type=float, default=RECORD_SECONDS)
    p.add_argument("--dt", type=float, default=DT)
    p.add_argument("--head", type=float, default=SETTLE_SECONDS)
    args = p.parse_args()

    ac, V, H = condition()
    sigma_w = float(wind.mehta_residual_ceiling())

    abar2 = gust.mean_square_ratio(ac, V, H, wind.dryden_spectrum)
    abar = np.sqrt(abar2)
    print(f"{AIRCRAFT} at M {MACH}, {H / 0.3048:.0f} ft, "
          f"sigma_w = {sigma_w:.4f} m/s (wind.mehta_residual_ceiling)")
    print(f"\nA  the continuous identity")
    print(f"   Abar = sqrt(integral |H|^2 Phi dOmega) = {abar:.6f} g per (m/s)")
    print(f"   predicted sigma_nz = Abar sigma_w      = {abar * sigma_w:.6f} g")
    print(f"   the TWO-SIDED mistake would give         {abar * sigma_w * np.sqrt(2):.6f} g"
          f"  (+41.4%)")

    rows = []
    for seed in range(args.seeds):
        s_n, s_w = fly_one(ac, V, H, sigma_w, seed, args.seconds, args.dt, args.head)
        exact_n2, exact_w2 = gust.realisation_mean_square_ratio(
            ac, V, H, sigma_w, seed)
        rows.append((seed, s_n, s_w, np.sqrt(exact_n2), np.sqrt(exact_w2)))

    flown_n = np.array([r[1] for r in rows])
    flown_w = np.array([r[2] for r in rows])
    exact_n = np.array([r[3] for r in rows])
    exact_w = np.array([r[4] for r in rows])
    n = len(rows)

    def mean_se(x):
        return float(np.mean(x)), float(np.std(x, ddof=1) / np.sqrt(len(x)))

    m_n, se_n = mean_se(flown_n)
    m_r, se_r = mean_se(flown_n / flown_w)
    m_xr, _ = mean_se(exact_n / exact_w)

    print(f"\nB  the ensemble, N = {n}, {args.seconds:.0f} s each at dt = {args.dt}, "
          f"first {args.head:.0f} s discarded")
    print(f"   flown sigma_nz                 {m_n:.6f} +/- {se_n:.6f} g")
    print(f"   predicted, continuous identity {abar * sigma_w:.6f} g"
          f"   -> {100 * (m_n / (abar * sigma_w) - 1):+.3f}%"
          f"  ({abs(m_n - abar * sigma_w) / se_n:.2f} se)")
    print(f"   predicted, this realisation    {np.mean(exact_n):.6f} g"
          f"   -> {100 * (m_n / np.mean(exact_n) - 1):+.3f}%"
          f"  ({abs(m_n - np.mean(exact_n)) / se_n:.2f} se)")

    print(f"\nC  the ratio, which divides the realisation's own variance out")
    print(f"   flown sigma_nz / sigma_w,record   {m_r:.6f} +/- {se_r:.6f} g per (m/s)")
    print(f"   Abar, continuous                  {abar:.6f}"
          f"   -> {100 * (m_r / abar - 1):+.3f}%  ({abs(m_r - abar) / se_r:.2f} se)")
    print(f"   exact for these realisations      {m_xr:.6f}"
          f"   -> {100 * (m_r / m_xr - 1):+.3f}%  ({abs(m_r - m_xr) / se_r:.2f} se)")

    print(f"\nD  the grid, priced on its own")
    print(f"   sigma_w the field was asked for   {sigma_w:.6f} m/s")
    print(f"   sigma_w the 400 components carry  {np.mean(exact_w):.6f} m/s"
          f"   ({100 * (np.mean(exact_w) / sigma_w - 1):+.3f}%)")
    print(f"   sigma_w the records realise       {np.mean(flown_w):.6f} m/s"
          f"   ({100 * (np.mean(flown_w) / sigma_w - 1):+.3f}%)")


if __name__ == "__main__":
    main()
