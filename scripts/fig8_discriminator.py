"""At what turbulence intensity does Fig. 8's PITCH axis stop discriminating?

THE QUESTION, AND WHY IT IS ABOUT SOMEONE ELSE'S METHOD. Wingrove & Bach 1994
Fig. 8 separates three causes of a severe-turbulence report -- vortex CAT,
convective updraft, and manoeuvring -- on two axes: pitch excursion and maximum
negative load change. Their three category values are 1.4, 6.2 and 12.0 deg of
pitch inside ONE load band, -2.01 to -1.69 g. So the chart's discriminating power
is mostly in PITCH, and a vortex's 1.4 deg is small not because a vortex is weak
but because it is FAST: a Hannibal core traverse is 0.235 of this 747's short
period, which is impulsive, where a 20 s updraft is 3.026 of one and quasi-steady.

Session 23d found that raising sigma_w across the sourced range collapsed the
vortex-updraft PITCH gap from 0.886 to 0.060 deg while the LOAD gap survived at
0.343 g -- i.e. the pitch axis stops working under realistic background
turbulence. It called that result MARGINAL, and it was right to, for a reason
this script exists to fix.

*** THE STATISTIC SESSION 23d USED MOVES WITH N, AND THAT IS THE WRONG WAY
ROUND. *** It reported the GAP BETWEEN THE EXTREMES of two 16-seed clouds. The
extremes of a distribution spread as more samples are drawn, so that gap shrinks
towards zero with effort whatever the truth is: collecting more evidence makes
"the clouds do not overlap" HARDER to say. A conclusion whose statistic degrades
with sample size is not a conclusion.

WHAT REPLACES IT. Two statistics whose EXPECTATION does not depend on N, so more
seeds shrink the error bar instead of moving the answer:

  AUC = P(updraft pitch > vortex pitch) over independently drawn pairs -- the
  Mann-Whitney U statistic normalised by N^2. 1.0 is perfect separation, 0.5 is
  none. Unpaired on purpose: a real diagnosis has ONE record and no matched
  control, so the question is whether two independent draws can be told apart.

  Cohen's d, the standardised mean difference, as a second scale-free reading.

  The extremes gap is still computed, and section A shows it moving with N while
  the other two hold. That comparison IS the argument for replacing it.

WHAT THIS CANNOT ANSWER, AND IT BOUNDS THE SWEEP. ASSUMPTIONS.md E11 records that
fixed-control runs leave their trim condition, and section 1's envelope caps the
linear aero at |alpha| = 10 deg. Peak |alpha| is therefore measured on every run
and reported per sigma, and any sigma whose runs leave the envelope is marked --
its AUC is printed but must not be used, per CLAUDE.md rule 6.

Run: PYTHONPATH=<abs worktree root> python -m scripts.fig8_discriminator
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import atisim  # noqa: F401  -- enables x64
from atisim import vortex_viz, wind
from atisim.aircraft import CRUISE, REGISTRY
from atisim.units import RAD2DEG
from scripts.cat_ensemble import AIRCRAFT, fly_updraft, fly_vortex

# Section 1's linear-aero ceiling: green below 10 deg, amber to 12.
ALPHA_GREEN_DEG = 10.0

# The sourced sigma_w range, session 23c, from Mehta's own fit residual. Values
# outside it are EXTRAPOLATION and are marked as such wherever they are printed.
SOURCED_LO = wind.mehta_unmodelled_wind()
SOURCED_HI = wind.mehta_residual_ceiling()


def auc(a: np.ndarray, b: np.ndarray) -> float:
    """P(b > a) over all N*M pairs, ties counted as a half.

    The Mann-Whitney U statistic divided by N*M. Computed directly rather than
    through a rank sum because N is small enough that the N^2 comparison is free
    and the direct form cannot get the tie convention wrong.
    """
    diff = b[None, :] - a[:, None]
    return float((np.sum(diff > 0) + 0.5 * np.sum(diff == 0)) / diff.size)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Standardised mean difference, pooled sd. NaN where both clouds are flat."""
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((b.mean() - a.mean()) / pooled) if pooled > 0 else float("nan")


def extremes_gap(a: np.ndarray, b: np.ndarray) -> float:
    """Session 23d's statistic: min(b) - max(a). Kept to show it moving with N."""
    return float(b.min() - a.max())


def auc_interval(a, b, *, boot=2000, seed=0):
    """Percentile bootstrap CI on the AUC, resampling seeds within each cloud."""
    rng = np.random.default_rng(seed)
    draws = [auc(rng.choice(a, len(a), replace=True),
                 rng.choice(b, len(b), replace=True)) for _ in range(boot)]
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def peak_alpha_deg(enc) -> float:
    """Largest |alpha| in the analysis window, degrees, air-relative."""
    return float(np.abs(enc.alpha_air[enc.window]).max()) * RAD2DEG


def fly_grid(ac, V, H, sigmas, seeds, dt):
    """One flight pair per (sigma, seed). Returns nested dicts of arrays."""
    out = {}
    for sigma in sigmas:
        rows = {"vortex": [], "updraft": [], "alpha": []}
        t0 = time.time()
        for seed in seeds:
            extra = wind.dryden_vertical_field(sigma, seed)
            v = fly_vortex(ac, V, H, extra, dt)
            u = fly_updraft(ac, V, H, extra, dt)
            rows["vortex"].append(vortex_viz.fig8_point(v))
            rows["updraft"].append(vortex_viz.fig8_point(u))
            rows["alpha"].append((peak_alpha_deg(v), peak_alpha_deg(u)))
        out[sigma] = {k: np.array(x) for k, x in rows.items()}
        out[sigma]["seconds"] = time.time() - t0
        print(f"    sigma {sigma:5.3f}  {len(seeds)} seeds in "
              f"{out[sigma]['seconds']:5.1f} s")
    return out


def section_a(grid, sigma, seeds, out):
    """Does the statistic move with N? Subsampled from the grid, no extra flights."""
    print("\n" + "=" * 78)
    print(f"A. THE STATISTIC ITSELF -- all three at N = 8, 16, 32, {len(seeds)}")
    print("=" * 78)
    print(f"sigma_w = {sigma:.3f} m/s. Nested subsamples of ONE grid, so the")
    print("difference between rows is the sample size and nothing else.\n")
    v = grid[sigma]["vortex"][:, 0]
    u = grid[sigma]["updraft"][:, 0]
    print(f"  {'N':>4s} {'extremes gap, deg':>18s} {'AUC':>8s} {'95% CI':>16s} {'Cohen d':>9s}")
    rows = []
    for n in (8, 16, 32, len(seeds)):
        if n > len(seeds):
            continue
        a, b = v[:n], u[:n]
        lo, hi = auc_interval(a, b)
        rows.append(dict(n=n, gap=extremes_gap(a, b), auc=auc(a, b),
                         ci=(lo, hi), d=cohens_d(a, b)))
        print(f"  {n:>4d} {extremes_gap(a, b):>18.4f} {auc(a, b):>8.4f} "
              f"[{lo:.3f}, {hi:.3f}] {cohens_d(a, b):>9.3f}")
    out["section_a"] = dict(sigma=sigma, rows=rows)
    first, last = rows[0], rows[-1]
    print(f"\n  gap      {first['gap']:+.4f} -> {last['gap']:+.4f} deg "
          f"over {first['n']} -> {last['n']} seeds")
    print(f"  AUC      {first['auc']:.4f} -> {last['auc']:.4f}, and its CI narrows "
          f"from {first['ci'][1]-first['ci'][0]:.3f} to {last['ci'][1]-last['ci'][0]:.3f}")
    print("  The gap is a statement about the sample; the AUC is a statement about")
    print("  the distributions. Only the second is what session 23d meant to make.")


def section_b(grid, sigmas, seeds, out):
    """AUC on both axes across the sigma sweep, with the envelope marked."""
    print("\n" + "=" * 78)
    print("B. THE SWEEP -- pitch and load discrimination against intensity")
    print("=" * 78)
    print(f"N = {len(seeds)} seeds per sigma. The SOURCED range is "
          f"{SOURCED_LO:.3f}-{SOURCED_HI:.3f} m/s (session 23c, from Mehta's own")
    print("fit residual); anything outside it is extrapolation and is marked EXTRAP.")
    print(f"peak |alpha| is the largest over BOTH categories; section 1 caps the")
    print(f"linear aero at {ALPHA_GREEN_DEG:.0f} deg and a run past it is marked OUT.\n")
    print(f"  {'sigma':>6s} {'AUC pitch':>10s} {'95% CI':>16s} {'AUC load':>9s} "
          f"{'95% CI':>16s} {'d pitch':>8s} {'d load':>7s} {'|a| max':>8s}  flags")
    rows = []
    for sigma in sigmas:
        g = grid[sigma]
        vp, up = g["vortex"][:, 0], g["updraft"][:, 0]
        vl, ul = g["vortex"][:, 1], g["updraft"][:, 1]
        # BOTH axes are auc(vortex, updraft) = P(updraft > vortex), so 1.0 means
        # "perfectly separable" on each. That is the right order for load as well
        # as pitch, and it is worth saying why rather than leaving it to be
        # re-derived: a vortex reaches a MORE NEGATIVE load than an updraft, so
        # the updraft's signed load is the larger of the two and the same
        # argument order gives the same meaning. An earlier version reversed the
        # load pair and read AUC 0.0000 for a perfectly separated axis.
        a_pitch, a_load = auc(vp, up), auc(vl, ul)
        cp, cl = auc_interval(vp, up), auc_interval(vl, ul)
        alpha = float(g["alpha"].max())
        flags = []
        # The endpoints are printed rounded, so compare with a tolerance rather
        # than exactly -- a sourced endpoint marked EXTRAP would be a lie about
        # its own provenance.
        if not (SOURCED_LO - 1e-3 <= sigma <= SOURCED_HI + 1e-3):
            flags.append("EXTRAP")
        if alpha > ALPHA_GREEN_DEG:
            flags.append("OUT-OF-ENVELOPE")
        rows.append(dict(sigma=sigma, auc_pitch=a_pitch, ci_pitch=cp,
                         auc_load=a_load, ci_load=cl, d_pitch=cohens_d(vp, up),
                         d_load=cohens_d(vl, ul), alpha=alpha, flags=flags,
                         gap_pitch=extremes_gap(vp, up),
                         gap_load=extremes_gap(vl, ul),
                         overlap_pitch=int(np.sum(up[None, :] <= vp[:, None])),
                         overlap_load=int(np.sum(ul[None, :] <= vl[:, None])),
                         pairs=len(vp) * len(up)))
        print(f"  {sigma:6.3f} {a_pitch:10.4f} [{cp[0]:.3f}, {cp[1]:.3f}] "
              f"{a_load:9.4f} [{cl[0]:.3f}, {cl[1]:.3f}] "
              f"{cohens_d(vp, up):8.3f} {cohens_d(vl, ul):7.3f} {alpha:7.2f}d  "
              f"{' '.join(flags)}")
    print(f"\n  AUC is over {rows[0]['pairs']} independent pairs per sigma. Where it")
    print("  reads exactly 1.0000 the bootstrap CI degenerates to [1, 1]: that is")
    print("  'no overlapping pair was observed', not 'separation is certain'. The")
    print("  overlap column below is the honest form of the same statement.\n")
    print(f"  {'sigma':>6s} {'overlapping pairs, pitch':>26s} {'load':>12s}")
    for r in rows:
        print(f"  {r['sigma']:6.3f} {r['overlap_pitch']:>13d} / {r['pairs']:<10d}"
              f" {r['overlap_load']:>5d} / {r['pairs']:<6d}")
    out["section_b"] = rows
    return rows


def section_c(rows, out):
    """Where the pitch axis fails, and whether the load axis goes with it."""
    print("\n" + "=" * 78)
    print("C. WHERE THE PITCH AXIS FAILS")
    print("=" * 78)
    usable = [r for r in rows if "OUT-OF-ENVELOPE" not in r["flags"]]
    sourced = [r for r in usable if "EXTRAP" not in r["flags"]]
    if not usable:
        print("  no sigma produced a run inside the envelope; nothing may be said")
        return
    print("  Reading only the rows inside section 1's envelope:\n")
    for label, subset in (("sourced range", sourced), ("all in-envelope", usable)):
        if not subset:
            continue
        lo, hi = subset[0], subset[-1]
        print(f"  {label:18s} sigma {lo['sigma']:.3f} -> {hi['sigma']:.3f}")
        print(f"    AUC pitch        {lo['auc_pitch']:.4f} -> {hi['auc_pitch']:.4f}")
        print(f"    AUC load         {lo['auc_load']:.4f} -> {hi['auc_load']:.4f}")
        print(f"    23d's gap, pitch {lo['gap_pitch']:+.4f} -> {hi['gap_pitch']:+.4f} deg")
    # The crossing, by linear interpolation between the bracketing rows.
    for level in (0.95, 0.90, 0.80):
        crossing = None
        for a, b in zip(usable, usable[1:]):
            if (a["auc_pitch"] - level) * (b["auc_pitch"] - level) <= 0 and \
               a["auc_pitch"] != b["auc_pitch"]:
                f = (a["auc_pitch"] - level) / (a["auc_pitch"] - b["auc_pitch"])
                crossing = a["sigma"] + f * (b["sigma"] - a["sigma"])
                break
        where = f"sigma = {crossing:.3f} m/s" if crossing else "not crossed in range"
        print(f"  AUC pitch falls through {level:.2f} at  {where}")
    out["section_c"] = dict(
        sourced=[r["sigma"] for r in sourced],
        in_envelope=[r["sigma"] for r in usable])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=64)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--sigmas", type=float, nargs="*", default=None)
    args = ap.parse_args()

    print(f"atisim imported from: {atisim.__file__}")
    ac = REGISTRY[AIRCRAFT]
    V, H = CRUISE[AIRCRAFT]["airspeed"], CRUISE[AIRCRAFT]["altitude"]
    sigmas = args.sigmas or [1.0, SOURCED_LO, 3.0, 3.75, SOURCED_HI, 5.5]
    sigmas = sorted(round(s, 4) for s in sigmas)
    seeds = list(range(args.seeds))

    print(f"\n  {AIRCRAFT} at {V:.2f} m/s, {H:.0f} m, dt {args.dt}")
    print(f"  vortex: PARKS_CASES['hannibal'], first core; updraft: Wingrove w0")
    print(f"  {len(sigmas)} intensities x {len(seeds)} seeds x 2 categories = "
          f"{len(sigmas)*len(seeds)*2} flights\n")

    grid = fly_grid(ac, V, H, sigmas, seeds, args.dt)
    out = {"tree": atisim.__file__, "seeds": len(seeds), "dt": args.dt,
           "sigmas": sigmas, "sourced": [SOURCED_LO, SOURCED_HI]}
    section_a(grid, sigmas[min(1, len(sigmas) - 1)], seeds, out)
    rows = section_b(grid, sigmas, seeds, out)
    section_c(rows, out)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(out)
        payload["grid"] = {str(s): {k: v.tolist() for k, v in g.items()
                                    if isinstance(v, np.ndarray)}
                           for s, g in grid.items()}
        args.json.write_text(json.dumps(payload, indent=1, default=float))
        print(f"\nwritten: {args.json}")


if __name__ == "__main__":
    main()
