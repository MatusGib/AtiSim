"""Phases S5 and S6: the same screen on an ENSEMBLE statistic, and the budget.

WHY RUN THE SCREEN TWICE. `scripts/sensitivity_load.py` ranks the coefficients
against a PEAK from one deterministic encounter -- one realisation of a random
process, whose error bar does not shrink with effort, and whose gradient belongs
to whichever sample happens to be the extreme. Section 4 added a statistic that
does not have those properties in session 25: the `n_z` rms over a Dryden
ensemble. If the ranking is a property of the aeroplane it should survive the
change of statistic. If it is a property of the peak it should not. Nothing in
this project has ever checked.

The protocol is session 25's, unchanged: `wind.dryden_field` at the two ends of
the sourced sigma range, 100 s flights with the first 20 s discarded, started in
moving air. Fewer seeds than `cat_spectra.py` uses, because each seed is a fresh
field closure and therefore a fresh trace -- the count is printed with the result
and the elasticities are reported per seed as well as pooled, so the spread is
visible rather than averaged away.

THEN THE BUDGET. Section 6 combines everything the study has priced into one
bracket on the headline load, in two forms: a linear sum, which needs no
independence claim, and an RSS, which needs one and says so.

Run: PYTHONPATH=<abs worktree root> python scripts/sensitivity_ensemble.py
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import atisim  # noqa: F401  -- enables x64
from atisim import sensitivity, wind
from atisim.aircraft import CRUISE, REGISTRY
from scripts.sensitivity_load import RECORDED_PEAK_TO_PEAK, mehta_setup

SETTLE_SECONDS = 20.0
RECORD_SECONDS = 100.0

# The fields screened here. The longitudinal set plus the sideslip derivative,
# because `dryden_field`'s v component is a real lateral input -- SIDESLIP, not
# a rolling gust, since the field still has no spanwise variation.
ENSEMBLE_FIELDS = sensitivity.LONGITUDINAL_FIELDS + (
    # The wave-drag trio, which `LONGITUDINAL_FIELDS` does not carry and which
    # the peak screen ranks THIRD. Omitting them here would have made the two
    # rankings incomparable in exactly the place they most needed comparing.
    "kappa_airfoil", "sweep", "t_over_c",
    # `dryden_field`'s v component is a real lateral input -- SIDESLIP, not a
    # rolling gust, since the field still has no spanwise variation -- so the
    # three sideslip-driven derivatives can reach the answer here where they
    # were EXACTLY zero on the vortex.
    "CYb", "Cnb", "Clb")


def rms_elasticities(ac, sigma, seed, V, H, dt, fields):
    """Elasticity of the recorded-window `n_z` rms w.r.t. each field, one seed."""
    field = wind.dryden_field(sigma, seed)
    model = wind.field_model(field)
    n_total = int(round((SETTLE_SECONDS + RECORD_SECONDS) / dt))
    n_settle = int(round(SETTLE_SECONDS / dt))

    def quantity(vec):
        n_z, _ = sensitivity.load_history(
            sensitivity.with_field_vector(ac, fields, vec), model, V, H,
            start_north=0.0, n_steps=n_total, dt=dt)
        rec = n_z[n_settle:]
        # rms of the FLUCTUATION, not of n_z: the mean is ~cos(theta) and
        # carries the trim, which is not what a gust statistic is about.
        return jnp.sqrt(jnp.mean((rec - jnp.mean(rec)) ** 2))

    base = sensitivity.field_vector(ac, fields)
    Q = float(quantity(base))
    grad = np.asarray(jax.jacfwd(quantity)(base))
    return Q, {f: sensitivity.elasticity(float(grad[i]), float(getattr(ac, f)), Q)
               for i, f in enumerate(fields)}


def section_e(out, seeds, dt):
    """S5: the ranking, against an rms instead of a peak."""
    print("\n" + "=" * 78)
    print("E. THE SAME SCREEN ON AN ENSEMBLE rms -- does the ranking survive?")
    print("=" * 78)

    ac = REGISTRY["boeing747"]
    V, H = CRUISE["boeing747"]["airspeed"], CRUISE["boeing747"]["altitude"]
    lo, hi = wind.mehta_unmodelled_wind(), wind.mehta_residual_ceiling()
    print(f"  sigma over the SOURCED range {lo:.3f}-{hi:.3f} m/s, "
          f"{len(seeds)} seeds x {RECORD_SECONDS:.0f} s "
          f"(first {SETTLE_SECONDS:.0f} s discarded), dt {dt}")
    print(f"  boeing747 at {V:.2f} m/s, {H:.0f} m\n")

    for sigma in (lo, hi):
        per_seed, Qs = [], []
        for seed in seeds:
            Q, e = rms_elasticities(ac, sigma, seed, V, H, dt, ENSEMBLE_FIELDS)
            per_seed.append(e)
            Qs.append(Q)
        pooled = {f: float(np.mean([e[f] for e in per_seed]))
                  for f in ENSEMBLE_FIELDS}
        spread = {f: float(np.std([e[f] for e in per_seed]))
                  for f in ENSEMBLE_FIELDS}
        out["ensemble"].append(dict(sigma=sigma, rms=Qs, pooled=pooled,
                                    spread=spread, seeds=list(seeds)))
        print(f"  sigma = {sigma:.3f} m/s   n_z rms {np.mean(Qs):.5f} g "
              f"[{min(Qs):.5f}, {max(Qs):.5f}] over {len(seeds)} seeds")
        rows = sorted(pooled.items(), key=lambda kv: -abs(kv[1]))
        print(f"    {'field':14s} {'elasticity':>11s} {'sd over seeds':>14s}")
        for f, v in rows:
            if abs(v) < 1e-6:
                continue
            print(f"    {f:14s} {v:+11.5f} {spread[f]:14.5f}")
        print()


def section_f(out, peak_elas):
    """S6: everything the study has priced, in one bracket on the headline."""
    print("\n" + "=" * 78)
    print("F. THE BUDGET -- what the headline load's band actually is")
    print("=" * 78)
    print("Every row is a per-cent of the headline peak-to-peak. A row enters")
    print("only where a source states a band or the register states a measured")
    print("cost; everything else is DECLARED fixed and named as such.\n")

    rows = [
        # (label, low %, high %, kind, provenance)
        ("C3 Mach, derivatives frozen", -5.041, 0.0, "measured",
         "session 29, this study -- PG on at the tabulation Mach"),
        ("E12 Hannibal core radius", -4.26, 0.0, "measured",
         "session 26, four papers disagree; PROJECT.md section 8"),
        ("V0 +/- 8.45% (SOURCED ceiling)", -4.28, +2.56, "sourced",
         "session 23c cat_uncertainty.py, Lester's vertical RMS over Mehta's V0"),
        ("r0 +/- 15% (DECLARED)", -2.70, +5.24, "declared",
         "session 23c -- Mehta's convergence study, not a knowledge statement"),
        ("F1 step size", 0.0, +0.25, "measured",
         "session 29 -- and it is sample placement, not integration order"),
        ("E4 wind held across RK4 stages", -0.053, 0.0, "measured",
         "session 29, this study"),
        ("A2 constant g vs g(z)", 0.0, +0.383, "measured",
         "sessions 12 and 23, cited -- g itself, an upper bound on the load term"),
    ]
    print(f"  {'source of error':34s} {'low %':>8s} {'high %':>8s}  {'kind':9s} provenance")
    for label, lo, hi, kind, prov in rows:
        print(f"  {label:34s} {lo:+8.3f} {hi:+8.3f}  {kind:9s} {prov}")

    lo_lin = sum(r[1] for r in rows)
    hi_lin = sum(r[2] for r in rows)
    lo_rss = -np.sqrt(sum(r[1] ** 2 for r in rows))
    hi_rss = +np.sqrt(sum(r[2] ** 2 for r in rows))

    base = out["base_pct"]
    print(f"\n  base: {base:.2f}% of TM-102186's {RECORDED_PEAK_TO_PEAK} g")
    print(f"  LINEAR SUM   {lo_lin:+.2f}% / {hi_lin:+.2f}%"
          f"  ->  {base*(1+lo_lin/100):.2f}% .. {base*(1+hi_lin/100):.2f}%")
    print(f"  RSS          {lo_rss:+.2f}% / {hi_rss:+.2f}%"
          f"  ->  {base*(1+lo_rss/100):.2f}% .. {base*(1+hi_rss/100):.2f}%")
    print("\n  NOT IN THE BUDGET, AND WHY. The tabulation Mach is not an unknown:")
    print("  section 4 records that this 747 flies M 0.800, 'which is the Mach its")
    print("  derivative set is tabulated at'. So the +3.04% / -15.85% measured for a")
    print("  reference Mach of 0.75 / 0.85 is a SENSITIVITY, not an error band, and")
    print("  putting it in the sum would have inflated the bracket with an")
    print("  uncertainty the source does not have. It is reported because it says")
    print("  how sharply C3 depends on a number nobody has had to defend before:")
    print("  a 0.05 error in the tabulation Mach would be worth three times the")
    print("  whole rest of the budget.\n")
    print("  The RSS needs the rows to be INDEPENDENT, which is not established")
    print("  and is declared here rather than assumed quietly. The linear sum")
    print("  needs no such claim and is the bracket to quote.")
    print(f"\n  THE SHORTFALL SURVIVES THE WHOLE BAND: the top of the linear sum")
    print(f"  is {base*(1+hi_lin/100):.1f}% of the record, against 100%. Session 23c")
    print("  reached the same conclusion from the WIND inputs alone and topped out")
    print("  at 72.7%; adding the AIRCRAFT's own priced errors does not change it.")
    out["budget"] = dict(rows=[dict(label=r[0], lo=r[1], hi=r[2], kind=r[3],
                                    provenance=r[4]) for r in rows],
                         linear=(lo_lin, hi_lin), rss=(float(lo_rss), float(hi_rss)),
                         base_pct=base,
                         linear_band=(base*(1+lo_lin/100), base*(1+hi_lin/100)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--dt", type=float, default=0.02)
    args = ap.parse_args()

    print(f"atisim imported from: {atisim.__file__}")
    cfg = mehta_setup()
    n_z, north = sensitivity.load_history(
        cfg["ac"], cfg["model"], cfg["V"], cfg["H"],
        start_north=cfg["start_north"], n_steps=cfg["n_steps"], dt=cfg["dt"])
    n_z, north = np.asarray(n_z), np.asarray(north)
    mask = (north >= cfg["window"][0]) & (north <= cfg["window"][1])
    base = float(n_z[mask].max() - n_z[mask].min())

    out = {"tree": atisim.__file__, "ensemble": [],
           "base": base, "base_pct": 100.0 * base / RECORDED_PEAK_TO_PEAK}

    section_e(out, tuple(range(args.seeds)), args.dt)
    section_f(out, None)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=1, default=float))
        print(f"\nwritten: {args.json}")


if __name__ == "__main__":
    main()
