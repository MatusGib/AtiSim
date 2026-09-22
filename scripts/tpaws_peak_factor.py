"""S2 -- the peak factor, both sides, ON TPAWS' OWN WINDOW DEFINITION.

Design: docs/design/specs/2026-09-21-turbulence-response-validation-design.md,
phase S2. Settles `predictions.the_model_peak_factor_lands_below_tpaws`, sealed
at db4eadf before the document was in reach.

THE HAZARD THE PREDICTION MADE A PRECONDITION, and what the document says.
The seal reads: "Whether Delta_n_max is the peak within that same window or over
the whole encounter must be READ OFF THE DOCUMENT. If it cannot be established
from the document, this prediction is abandoned rather than settled either way."
It can be established, from three places in NASA/TM-2012-217337:

  printed p. 5   sigma_dn is defined as a function of TIME over a "shifting /
                 sliding window", tau = 5 s, and the printed equation subtracts
                 the mean OVER THAT SAME WINDOW -- so it is a running standard
                 deviation, not a running RMS about the record mean.
  printed p. 125 "the peak sigma_dn" -- the tabulated scalar is the MAXIMUM over
                 the encounter of that running quantity.
  printed p. 8   Table 1's column group header, "Peak In Situ Turbulence (g's)",
                 governs all three load columns; Delta_n_max and Delta_n_min are
                 the encounter's extremes (printed p. 7, "the peak normal load
                 acceleration").

So TPAWS' peak factor is  max|Delta_n| over the encounter  /  max over the
encounter of a 5 s running standard deviation.  V5 measured the model's on a
WHOLE-RECORD sigma. Those are different statistics and the difference has a
SIGN: a running maximum is >= the whole-record value, so TPAWS' denominator is
the larger one and its peak factor is the smaller. Comparing V5's 3.8674 against
TPAWS' mean directly would credit the model with a gap the definitions create.

A SECOND MISMATCH THE DESIGN DID NOT ANTICIPATE, recorded because it is not
removable by any reduction. TPAWS' encounters last "several seconds to a minute"
(printed p. 7); this project's records are RECORD_SECONDS = 1200 s. Both the
extreme and the running maximum grow with record length, so record length is a
third quantity the comparison is sensitive to, and R3 below is the only
reduction that controls it.

THE REDUCTIONS ARE PRE-SPECIFIED HERE, BEFORE THE RUN, and all of them are
computed from the SAME flights in one pass. That is deliberate: V5 recorded that
increasing N after seeing a marginal result and stopping when it clears is
optional stopping, and choosing a reduction after seeing which one settles the
prediction is the same fault wearing a different hat. N is the audit script's
own sealed default, 48 per arm, not a number picked here.

  R1  WHOLE-RECORD.   max(d) / std(d) over the whole retained record.
                      V5's definition, carried for continuity.
  R2  TPAWS WINDOW.   max(d) / max(running 5 s std of d), over the whole
                      retained record. Fixes the denominator, not the duration.
  R3  TPAWS ENCOUNTER. the retained record chopped into non-overlapping
                      segments of L seconds, each treated as one encounter:
                      max(d_seg) / max(running 5 s std within that segment).
                      Segments pool across seeds into a population the way
                      TPAWS' 53 encounters do. L = 15, 30 and 60 s brackets
                      the document's "several seconds to a minute"; L is
                      DECLARED and no single value of it is privileged.

d is the deviation of n_z from its whole-record mean throughout, because TPAWS'
Delta_n is a load increment about 1 g and not a per-encounter anomaly.

THE TPAWS SIDE IS RE-DERIVED FROM THE DOCUMENT, as the seal requires -- from
`atisim/data/tpaws_tm2012_217337_table1.csv` (phase S0), which has 53 rows. The
seal's own 2.386 / 0.506 were transcribed from the design document, which
carried 51: its parser dropped the two rows whose altitude prints as a range.

Run: PYTHONPATH=<abs worktree root> .venv/Scripts/python.exe -u \\
         scripts/tpaws_peak_factor.py [-n 48] [--tpaws-only]
"""

import argparse
import csv
from pathlib import Path

import numpy as np

import atisim  # noqa: F401  -- enables x64

print(f"atisim imported from: {atisim.__file__}")

from atisim import predictions, vortex_viz, wind  # noqa: E402
from atisim.aircraft import REGISTRY  # noqa: E402
from atisim.atmosphere import speed_of_sound  # noqa: E402

AIRCRAFT = "boeing747"
MACH = 0.80

# All three match scripts/dryden_realisation_audit.py, so the R1 column here is
# comparable to V5's number rather than merely similar to it.
RECORD_SECONDS = 1200.0
SETTLE_SECONDS = 100.0
DT = 0.02
ENSEMBLE = 48

TAU_SECONDS = 5.0  # TPAWS' own window, printed p. 5
ENCOUNTER_SECONDS = (15.0, 30.0, 60.0)  # DECLARED; printed p. 7's duration band

DATA = Path(atisim.__file__).parent / "data" / "tpaws_tm2012_217337_table1.csv"


def condition():
    H = float(wind.MEHTA_HANNIBAL_ALTITUDE)
    return REGISTRY[AIRCRAFT], MACH * float(speed_of_sound(H)), H


def running_std(d, win):
    """The printed p. 5 equation: std over a sliding window of `win` samples.

    Mean of the square minus the square of the mean, both over the SAME window,
    which is what the document's inner integral makes it. Clipped at zero before
    the root because cancellation can put the variance a few ulp below it.
    """
    if win > d.size:
        raise ValueError(f"window {win} longer than record {d.size}")
    c1 = np.concatenate(([0.0], np.cumsum(d)))
    c2 = np.concatenate(([0.0], np.cumsum(d * d)))
    m = (c1[win:] - c1[:-win]) / win
    m2 = (c2[win:] - c2[:-win]) / win
    return np.sqrt(np.clip(m2 - m * m, 0.0, None))


def reductions(d, dt):
    """R1, R2 and R3(L) for one record. Returns (r1, r2, {L: [per-segment...]})."""
    win = int(round(TAU_SECONDS / dt))
    r1 = float(np.max(d) / np.std(d))
    r2 = float(np.max(d) / np.max(running_std(d, win)))
    r3 = {}
    for seconds in ENCOUNTER_SECONDS:
        n = int(round(seconds / dt))
        segs = d[:d.size - d.size % n].reshape(-1, n)
        r3[seconds] = [float(np.max(s) / np.max(running_std(s, win))) for s in segs]
    return r1, r2, r3


def fly_arm(ac, V, H, sigma_w, maker, seeds, seconds, dt, head):
    """One construction's ensemble. Flights match dryden_realisation_audit.py."""
    r1s, r2s, r3s = [], [], {s: [] for s in ENCOUNTER_SECONDS}
    for seed in seeds:
        enc = vortex_viz.fly_in_moving_air(
            ac, maker(sigma_w, seed), V, H, label=f"seed {seed}", start_north=0.0,
            seconds=seconds, dt=dt, window=(-np.inf, np.inf),
            window_name="whole run", stage_sampled=True,
        )
        n_z = enc.n_z[enc.t >= head]
        d = np.asarray(n_z) - float(np.mean(n_z))
        r1, r2, r3 = reductions(d, dt)
        r1s.append(r1)
        r2s.append(r2)
        for s in ENCOUNTER_SECONDS:
            r3s[s].extend(r3[s])
        print(f"      seed {seed:>3}  R1 {r1:6.3f}  R2 {r2:6.3f}", flush=True)
    return np.array(r1s), np.array(r2s), {s: np.array(v) for s, v in r3s.items()}


def tpaws_side():
    rows = list(csv.DictReader(open(DATA)))
    up = np.array([float(r["dn_max_g"]) / float(r["sigma_dn_g"]) for r in rows])
    return rows, up


def mean_se(a):
    return float(a.mean()), float(a.std(ddof=1) / np.sqrt(a.size))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("-n", "--seeds", type=int, default=ENSEMBLE)
    p.add_argument("--seconds", type=float, default=RECORD_SECONDS)
    p.add_argument("--dt", type=float, default=DT)
    p.add_argument("--head", type=float, default=SETTLE_SECONDS)
    p.add_argument("--tpaws-only", action="store_true",
                   help="the document side alone; no flights")
    args = p.parse_args()

    rows, up = tpaws_side()
    print(f"\nTPAWS SIDE -- {DATA.name}, {len(rows)} rows")
    print(f"   peak factor dn_max/sigma : mean {up.mean():.4f}  "
          f"sd(pop) {up.std():.4f}  range {up.min():.2f}-{up.max():.2f}")
    print("   sigma is the PEAK of a 5 s running std (printed pp. 5, 125);")
    print("   the extreme is the encounter's (printed pp. 7, 8).")
    print(f"   the seal transcribed mean 2.386 / sd 0.506 from a 51-row parse.")
    if args.tpaws_only:
        return

    ac, V, H = condition()
    sigma_w = float(wind.mehta_residual_ceiling())
    print(f"\nMODEL SIDE -- {AIRCRAFT} at M {MACH}, {H / 0.3048:.0f} ft, "
          f"sigma_w = {sigma_w:.4f} m/s, N = {args.seeds} per arm, "
          f"{args.seconds:.0f} s records at dt = {args.dt}")

    arms = {}
    for name, maker in (("Shinozuka", wind.dryden_vertical_field),
                        ("Gaussian", wind.gaussian_vertical_field)):
        print(f"\n   {name}:")
        arms[name] = fly_arm(ac, V, H, sigma_w, maker, range(args.seeds),
                             args.seconds, args.dt, args.head)

    print(f"\nRESULTS, N = {args.seeds} per arm")
    print(f"   {'arm':10} {'R1 whole-record':>20} {'R2 TPAWS window':>20}")
    for name, (r1, r2, _) in arms.items():
        m1, s1 = mean_se(r1)
        m2, s2 = mean_se(r2)
        print(f"   {name:10} {m1:12.4f}+/-{s1:.4f} {m2:12.4f}+/-{s2:.4f}")

    print(f"\n   R3, TPAWS ENCOUNTER, segments pooled across seeds")
    print(f"   {'arm':10} {'L (s)':>7} {'segments':>10} {'peak factor':>20}")
    for name, (_, _, r3) in arms.items():
        for seconds in ENCOUNTER_SECONDS:
            m, s = mean_se(r3[seconds])
            print(f"   {name:10} {seconds:7.0f} {r3[seconds].size:10d} "
                  f"{m:12.4f}+/-{s:.4f}")

    pred = predictions.BY_NAME["the_model_peak_factor_lands_below_tpaws"]
    threshold = float(up.mean())
    print(f"\n   {pred.name} (sealed {pred.sealed_at}, status {pred.status}):")
    print(f"      threshold, re-derived from the document: {threshold:.4f} "
          f"(the seal transcribed 2.386)")
    sh = arms["Shinozuka"]
    for label, value in (("R1 whole-record", sh[0].mean()),
                         ("R2 TPAWS window", sh[1].mean())) + tuple(
            (f"R3 L={s:.0f}s", sh[2][s].mean()) for s in ENCOUNTER_SECONDS):
        verdict = "BELOW -> right" if value < threshold else "at or above -> WRONG"
        print(f"      {label:18} {value:7.4f}   {verdict}")


if __name__ == "__main__":
    main()
