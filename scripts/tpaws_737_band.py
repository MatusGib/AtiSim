"""The TPAWS peak factor against the aircraft whose declared band the data sits in.

WHY THIS EXISTS. S2 compared TPAWS' encounters against `boeing747`, whose
`valid_altitude` is 35,000-45,000 ft. That admits **3 of the 53 rows**, and
three points cannot set a band -- the design said so itself and fell back on
the whole 15-35 kft population, out of band, with the justification restated
every time it was quoted. `boeing737` declares **25,000-35,000 ft, M 0.68-0.88**
(`aircraft.py:1486`), which is where the data actually is: **28 of the 53 rows
sit inside BOTH that altitude band and that Mach band**. This flies the
comparison there.

WHAT IS AND IS NOT LICENSED BY THIS. TPAWS' aircraft is a **B-757** and
`boeing737` is not one, so nothing here licenses a comparison of load LEVEL.
The peak factor is compared because it is a property of the response process --
how a lightly damped oscillator's extreme relates to its own rms -- which is
the same argument the design used to quote an out-of-band population, and it is
the argument that has to survive for this to mean anything. Per rule 6 this is
an ordering-and-band comparison and never an absolute one.

A SECOND CAVEAT, which belongs beside every number below. The `boeing737`
entry's longitudinal set is **fitted to JSBSim over 25,000-35,000 ft**, which it
reproduces to 1.8% there (`aircraft.py`'s own docstring). So this is a
model calibrated against another code, not against flight. What that buys is a
short period in roughly the right place; what it does not buy is independence.

THE TARGET MOVES WHEN THE BAND IS APPLIED, and that is itself a finding. Over
all 53 rows the up peak factor is 2.3831 +/- 0.4942 (population sd). Over the
28 in band for this aircraft it is **2.5013 +/- 0.5503** -- the in-band
encounters are the higher-altitude ones and they are peakier. The gate below
uses the 28-row number, because comparing against a population the aircraft
could not have flown is the thing this script exists to stop doing.

PRE-SPECIFIED BEFORE THE RUN, and the reductions are IMPORTED from
`tpaws_peak_factor.py` rather than restated, so they are the same code S2 ran
and cannot drift from it:

  conditions  the 8 distinct in-band ALTITUDES among the 28 rows, each flown at
              the MEAN MACH of the in-band rows at that altitude. A DECLARED
              grouping: the model does not distinguish 25 kft M 0.691 from
              25 kft M 0.710, and pretending otherwise would invent resolution.
  N           8 seeds per condition, 64 flights. R3 pools segments, so even at
              N = 8 each condition carries ~290 thirty-second encounters.
  record      1200 s at dt = 0.02, first 100 s discarded -- identical to S2.
  field       `wind.dryden_vertical_field` at `wind.mehta_residual_ceiling()`,
              the construction and intensity S2 used.
  reductions  R1 whole-record, R2 TPAWS' own window, R3 at 15/30/60 s.
  GATE        does the ensemble mean under R2 land inside 2.5013 +/- 0.5503,
              and where does it fall? Reported per condition as well as pooled,
              because whether the peak factor moves across the band is the
              question that decides if one number was ever the right summary.

CHECKS, none of which sets a parameter:
  1  BAND. Every flown condition must pass `checks.recovery_band` for this
     aircraft. A condition that does not is not flown.
  2  INTENSITY INVARIANCE. The peak factor is a normalised ratio and should not
     move with sigma_w. One condition is re-flown at 0.5x and 2x, and R2 must
     agree within 5%. If it does not, the comparison is measuring intensity and
     the gate above is meaningless.
  3  POPULATION. The in-band selection must be exactly 28 rows.

Run: PYTHONPATH=<abs worktree root> .venv/Scripts/python.exe -u \\
         scripts/tpaws_737_band.py [-n 8]
"""

import argparse
import csv
import statistics as st
import sys
from pathlib import Path

import numpy as np

import atisim  # noqa: F401  -- enables x64

print(f"atisim imported from: {atisim.__file__}")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from atisim import checks, vortex_viz, wind  # noqa: E402
from atisim.aircraft import REGISTRY  # noqa: E402
from atisim.atmosphere import speed_of_sound  # noqa: E402
from tpaws_peak_factor import ENCOUNTER_SECONDS, mean_se, reductions  # noqa: E402

AIRCRAFT = "boeing737"
RECORD_SECONDS = 1200.0
SETTLE_SECONDS = 100.0
DT = 0.02
SEEDS = 8
FT2M = 0.3048

DATA = Path(atisim.__file__).parent / "data" / "tpaws_tm2012_217337_table1.csv"
N_IN_BAND = 28


def in_band_rows(ac):
    """The TPAWS rows inside BOTH declared bands, with each row's own Mach."""
    lo_ft, hi_ft = (float(v) / FT2M / 1000.0 for v in ac.valid_altitude)
    m_lo, m_hi = (float(v) for v in ac.valid_mach)
    out = []
    with open(DATA, newline="") as f:
        for r in csv.DictReader(f):
            lo = min(float(r["altitude_kft_first"]), float(r["altitude_kft_last"]))
            hi = max(float(r["altitude_kft_first"]), float(r["altitude_kft_last"]))
            if not (lo >= lo_ft and hi <= hi_ft):
                continue
            mach = float(r["tas_ms"]) / float(speed_of_sound(lo * 1000.0 * FT2M))
            if not m_lo <= mach <= m_hi:
                continue
            out.append({
                "event": r["event"], "alt_kft": lo, "mach": mach,
                "up": float(r["dn_max_g"]) / float(r["sigma_dn_g"]),
                "down": abs(float(r["dn_min_g"])) / float(r["sigma_dn_g"]),
            })
    return out


def conditions_from(rows):
    """One condition per distinct in-band altitude, at that altitude's mean Mach."""
    alts = sorted({r["alt_kft"] for r in rows})
    return [(a, st.mean([r["mach"] for r in rows if r["alt_kft"] == a]),
             sum(1 for r in rows if r["alt_kft"] == a)) for a in alts]


def fly(ac, alt_kft, mach, sigma_w, seeds, seconds, dt, head):
    """Returns the three reductions plus the WORST recovery-band excursion.

    The band check is run on the flown trajectory rather than on the commanded
    condition, because the aeroplane is free to drift in altitude over 1200 s
    and it is where it actually went that decides whether the derivatives were
    recovered there. `Check.value` is in band widths: 0.0 is inside.
    """
    H = alt_kft * 1000.0 * FT2M
    V = mach * float(speed_of_sound(H))
    r1s, r2s, r3s = [], [], {s: [] for s in ENCOUNTER_SECONDS}
    worst_band = 0.0
    for seed in seeds:
        enc = vortex_viz.fly_in_moving_air(
            ac, wind.dryden_vertical_field(sigma_w, seed), V, H,
            label=f"{alt_kft:.0f}kft seed {seed}", start_north=0.0,
            seconds=seconds, dt=dt, window=(-np.inf, np.inf),
            window_name="whole run", stage_sampled=True,
        )
        band = checks.recovery_band(enc.log, ac)
        worst_band = max(worst_band, float(band.value))
        n_z = enc.n_z[enc.t >= head]
        d = np.asarray(n_z) - float(np.mean(n_z))
        r1, r2, r3 = reductions(d, dt)
        r1s.append(r1)
        r2s.append(r2)
        for s in ENCOUNTER_SECONDS:
            r3s[s].extend(r3[s])
    return (np.array(r1s), np.array(r2s),
            {s: np.array(v) for s, v in r3s.items()}, worst_band)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("-n", "--seeds", type=int, default=SEEDS)
    p.add_argument("--seconds", type=float, default=RECORD_SECONDS)
    p.add_argument("--dt", type=float, default=DT)
    p.add_argument("--head", type=float, default=SETTLE_SECONDS)
    args = p.parse_args()

    ac = REGISTRY[AIRCRAFT]
    rows = in_band_rows(ac)
    conds = conditions_from(rows)
    up = [r["up"] for r in rows]
    target, target_sd = st.mean(up), st.pstdev(up)

    print(f"\nTARGET -- the {len(rows)} TPAWS rows inside {AIRCRAFT}'s declared bands")
    print(f"   up peak factor   mean {target:.4f}  sd(pop) {target_sd:.4f}  "
          f"range {min(up):.2f}-{max(up):.2f}")
    print(f"   down             mean {st.mean([r['down'] for r in rows]):.4f}")
    print("   for contrast, all 53 rows: mean 2.3831  sd(pop) 0.4942 "
          "-- the in-band subset is PEAKIER")

    print("\nCHECK 3, POPULATION")
    if len(rows) != N_IN_BAND:
        print(f"  FAIL  {len(rows)} rows in band, expected {N_IN_BAND}")
        raise SystemExit(1)
    print(f"  {len(rows)} rows in band over {len(conds)} distinct altitudes")

    sigma_w = float(wind.mehta_residual_ceiling())
    print(f"\nMODEL -- {AIRCRAFT}, sigma_w = {sigma_w:.4f} m/s, N = {args.seeds} "
          f"per condition, {args.seconds:.0f} s at dt = {args.dt}")
    print(f"   {'alt':>5} {'Mach':>6} {'rows':>5} {'R1 whole':>16} "
          f"{'R2 TPAWS':>16} {'R3 30s':>16} {'band':>7}")

    pooled = {"r1": [], "r2": [], "r3": []}
    worst_band = 0.0
    for alt, mach, nrows in conds:
        r1, r2, r3, band = fly(ac, alt, mach, sigma_w, range(args.seeds),
                               args.seconds, args.dt, args.head)
        worst_band = max(worst_band, band)
        m1, s1 = mean_se(r1)
        m2, s2 = mean_se(r2)
        m3, s3 = mean_se(r3[30.0])
        pooled["r1"] += list(r1)
        pooled["r2"] += list(r2)
        pooled["r3"] += list(r3[30.0])
        print(f"   {alt:5.0f} {mach:6.3f} {nrows:5d} {m1:9.4f}+/-{s1:.4f} "
              f"{m2:9.4f}+/-{s2:.4f} {m3:9.4f}+/-{s3:.4f} {band:7.3f}",
              flush=True)

    verdict = ("INSIDE" if worst_band <= 0.0 else
               "OUTSIDE -- the derivatives were not recovered where this flew")
    print(f"\n  CHECK 1, BAND: worst recovery_band excursion over every flown "
          f"trajectory = {worst_band:.3f} band widths ({verdict})")

    print("\n  CHECK 2, INTENSITY INVARIANCE at the median condition")
    alt, mach, _ = conds[len(conds) // 2]
    got = {}
    for scale in (0.5, 1.0, 2.0):
        _, r2, _, _ = fly(ac, alt, mach, sigma_w * scale, range(2),
                          args.seconds, args.dt, args.head)
        got[scale] = float(r2.mean())
        print(f"     sigma_w x{scale:<4} R2 = {got[scale]:.4f}")
    spread = max(abs(got[s] / got[1.0] - 1.0) for s in (0.5, 2.0))
    print(f"     worst departure from the x1 value: {spread * 100:.2f}% "
          f"({'PASSES' if spread < 0.05 else 'FAILS'} the 5% gate)")

    print(f"\nPOOLED over {len(conds)} conditions x {args.seeds} seeds")
    for key, label in (("r1", "R1 whole-record"), ("r2", "R2 TPAWS window"),
                       ("r3", "R3 30 s encounters")):
        a = np.array(pooled[key])
        m, s = mean_se(a)
        z = (m - target) / target_sd
        inside = "INSIDE" if abs(z) <= 1.0 else "outside"
        print(f"   {label:20} {m:8.4f}+/-{s:.4f}   {z:+.2f} population sd "
              f"-> {inside} {target:.4f}+/-{target_sd:.4f}")


if __name__ == "__main__":
    main()
