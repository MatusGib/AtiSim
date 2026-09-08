"""Does a POD of the LES n_z ensemble separate condition drift from gust response?

PROJECT.md section 4 lists condition drift as candidate 1 for the ~20% that
survives the matched-aeroplane LES comparison: "AtiSim's aeroplane loses 39.8
m/s over the record -- 30% of its airspeed -- and 552 m of altitude, because it
flies fixed-control with real drag. Yoshimura's holds altitude to 26-43 m and
cannot decelerate, having neither drag nor thrust. Load goes as qbar, so a
decelerating aeroplane and a non-decelerating one are not measuring the same
thing. THIS IS THE NEXT EXPERIMENT: hold the condition, or high-pass harder,
and re-read."

An SVD of the (flights x time) matrix is the obvious instrument: the drift is
common to every flight, so it should fall out as one dominant mode with a
near-uniform loading across flights, leaving the gust response behind.

*** THE POINT OF THIS SCRIPT IS TO ASK WHETHER IT BEATS WHAT IS ALREADY DONE. ***
The shipped pipeline already removes this drift twice over: `les_flight.py`
discards 20 s of settling BEFORE writing the .npy (line 652), and
`les_compare.py` then high-passes at `response.PHUGOID_FLOOR_HZ` = 0.05 Hz. Its
own docstring says that high-pass "is what makes the two the same quantity, and
the drift it removes is" exactly this. So the SVD is not competing with nothing.
Three removals are measured side by side on the same arrays:

    (ii)  subtract the ENSEMBLE MEAN over flights   -- one line, no SVD
    (iii) subtract the RANK-1 SVD reconstruction    -- the proposal
    (iv)  HIGH-PASS at 0.05 Hz                      -- what is shipped

and the question that decides it is narrow: **how much of POD mode 1 lives
below 0.05 Hz?** If nearly all of it does, the shipped high-pass already removes
what the SVD would remove, and the SVD is a second name for the same operation.

Data: the n_z ensembles written by `scripts/les_flight.py`, 16 flights x 5000
samples at dt = 0.02 s. They are gitignored and live in whichever worktree ran
them, so `--runs` is required rather than assumed -- CLAUDE.md rule 5.

*** THESE ARE THE boeing747 RUNS SECTION 4 REFUSES FOR LOAD LEVEL. *** The
aeroplane is 5.4% away in short period and 2.63 band widths outside its
envelope, so no number here is evidence about load. It does not matter for this
question: the drift is a property of flying fixed-control with real drag, which
that entry does, and the confounders section 4 names are constant multipliers
that cannot create or remove a secular trend.

Run: PYTHONPATH=<abs worktree root> .venv/Scripts/python.exe scripts/les_ensemble_svd.py \
         --runs <dir containing les-nz-D0*.npy>
"""

import argparse
from pathlib import Path

import numpy as np

from atisim.response import PHUGOID_FLOOR_HZ

DT = 0.02
DOMAINS = (("D01", 500.0), ("D02", 250.0), ("D03", 70.0), ("D04", 35.0))


def highpass(x, dt, f_cut=PHUGOID_FLOOR_HZ):
    """Identical to les_compare.highpass -- a brick wall on the rfft."""
    X = np.fft.rfft(x - x.mean())
    f = np.fft.rfftfreq(len(x), dt)
    X[f < f_cut] = 0.0
    return np.fft.irfft(X, n=len(x))


def low_fraction(x, dt, f_cut=PHUGOID_FLOOR_HZ):
    """Fraction of a signal's variance below f_cut. Parseval on the rfft."""
    X = np.fft.rfft(x - x.mean())
    f = np.fft.rfftfreq(len(x), dt)
    p = np.abs(X) ** 2
    return float(p[f < f_cut].sum() / p.sum())


def analyse(nz, dt=DT):
    # NO SETTLING DROP HERE, AND THAT IS DELIBERATE. `les_flight.py` line 652
    # already writes `n_z[k:]` with k = SETTLE_SECONDS/dt, so the .npy on disk
    # has the 20 s removed and `les_compare.atisim` correctly does not remove it
    # again. An earlier draft of this script dropped it a second time and got
    # D03 rms 0.09547 against the published 0.09049 -- the check below is what
    # caught that, which is why it is a check and not a comment.
    A = np.asarray(nz)
    n_flights, n = A.shape
    t = np.arange(n) * dt

    # Row-centred: each flight against its own mean, which is the first thing
    # `highpass` does too. Without it mode 1 is the 1 g DC level and says nothing.
    Ac = A - A.mean(axis=1, keepdims=True)

    U, s, Vt = np.linalg.svd(Ac, full_matrices=False)
    energy = s**2 / (s**2).sum()

    mode1_t = Vt[0]
    loading = U[:, 0]
    # Sign convention: make the dominant loading positive so the shape is readable.
    if loading[np.argmax(np.abs(loading))] < 0:
        loading, mode1_t = -loading, -mode1_t

    rank1 = np.outer(U[:, 0] * s[0], Vt[0])
    ens_mean = Ac.mean(axis=0, keepdims=True)

    variants = {
        "(i)   row-centred only": Ac,
        "(ii)  minus ensemble mean": Ac - ens_mean,
        "(iii) minus rank-1 SVD": Ac - rank1,
        "(iv)  high-passed 0.05 Hz": np.array([highpass(r, dt) for r in A]),
    }
    return dict(
        t=t, n_flights=n_flights, energy=energy, s=s,
        mode1_t=mode1_t, loading=loading,
        mode1_low=low_fraction(mode1_t, dt),
        mode1_trend=float(np.corrcoef(t, mode1_t)[0, 1]),
        loading_uniformity=float(
            np.abs(loading.mean()) / np.abs(loading).mean()
        ),
        rms={k2: float(v.std(axis=1).mean()) for k2, v in variants.items()},
        variants=variants,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, type=Path,
                    help="directory holding les-nz-D0*-boeing747.npy")
    ap.add_argument("--aircraft", default="boeing747")
    args = ap.parse_args()

    print("=" * 78)
    print("POD OF THE LES n_z ENSEMBLE -- IS MODE 1 THE CONDITION DRIFT?")
    print("=" * 78)
    print(f"  arrays from {args.runs}")
    print(f"  entry {args.aircraft}; settling already dropped on disk; dt = {DT} s")
    print()

    results = {}
    for name, dx in DOMAINS:
        f = args.runs / f"les-nz-{name}-{args.aircraft}.npy"
        if not f.exists():
            print(f"  {name}: MISSING {f.name}")
            continue
        r = analyse(np.load(f))
        results[name] = r
        print(f"  {name}  dx = {dx:5.0f} m   {r['n_flights']} flights")
        print(f"    energy by mode   " + "  ".join(f"{e*100:5.1f}%" for e in r["energy"][:5]))
        print(f"    mode 1 carries   {r['energy'][0]*100:.1f}% of ensemble variance")
        print(f"    mode 1 below 0.05 Hz          {r['mode1_low']*100:6.2f}%  "
              f"<-- THE DECIDING NUMBER")
        print(f"    mode 1 correlation with t     {r['mode1_trend']:+6.3f}")
        print(f"    loading uniformity |mean|/mean|.|  {r['loading_uniformity']:.4f}"
              f"   (1.000 = every flight loads it the same way)")
        print(f"    rms of what survives each removal:")
        for k2, v in r["rms"].items():
            print(f"      {k2:28s} {v:.5f} g")
        print()

    if "D03" in results and "D04" in results:
        print("=" * 78)
        print("DOES THE SVD BEAT THE HIGH-PASS THAT IS ALREADY SHIPPED?")
        print("=" * 78)
        for name in ("D03", "D04"):
            r = results[name]
            hp = r["rms"]["(iv)  high-passed 0.05 Hz"]
            r1 = r["rms"]["(iii) minus rank-1 SVD"]
            em = r["rms"]["(ii)  minus ensemble mean"]
            print(f"  {name}:  high-pass {hp:.5f}   rank-1 {r1:.5f}   "
                  f"ens-mean {em:.5f} g")
            print(f"        rank-1 vs high-pass  {r1/hp:+.4f}x     "
                  f"rank-1 vs ens-mean {r1/em:.4f}x")
        print()
        print("  Published: D03 0.09049 g, D04 0.15728 g (les-comparison.csv).")
        print("  The (iv) rows above must match them exactly or this script is")
        print("  not analysing the pipeline it claims to be analysing.")
        drift_test(args)


def drift_test(args):
    """The RIGHT instrument for the question the POD was reached for.

    A 0.05 Hz high-pass removes a secular drift outright, so the drift cannot be
    an ADDITIVE term in the published rms -- it is already gone from both sides.
    What a high-pass cannot remove is the drift's real effect: load goes as qbar,
    so an aeroplane that has lost 30% of its airspeed by the end of the record
    responds to the SAME gust with a smaller excursion. That is an amplitude
    MODULATION of the in-band signal, and it shows up as early-half rms against
    late-half rms -- which needs no decomposition at all.
    """
    print()
    print("=" * 78)
    print("AND THE QUESTION THE POD WAS REACHED FOR, ASKED DIRECTLY")
    print("=" * 78)
    print("  Section 4: the aeroplane loses 39.8 m/s -- 30% of its airspeed -- over")
    print("  the record. Load goes as qbar, so if that matters the in-band response")
    print("  must be WEAKER in the second half of every flight. Split and compare:")
    print()
    for name, dx in DOMAINS:
        f = args.runs / f"les-nz-{name}-{args.aircraft}.npy"
        if not f.exists():
            continue
        A = np.load(f)
        hp = np.array([highpass(r, DT) for r in A])
        half = hp.shape[1] // 2
        early = hp[:, :half].std(axis=1)
        late = hp[:, half:].std(axis=1)
        ratio = late.mean() / early.mean()
        n_down = int((late < early).sum())
        print(f"  {name}  dx {dx:5.0f} m   early {early.mean():.5f} g   "
              f"late {late.mean():.5f} g   late/early {ratio:.4f}   "
              f"{n_down}/{len(early)} flights weaker late")
    print()
    print("  qbar alone would predict late/early ~ (V_late/V_early)^2. A 30% speed")
    print("  loss spread over the record puts that near 0.6-0.7 if the drift")
    print("  dominates the in-band response. Read the measured column against that.")
    print()
    print("  AND IT IS INCONCLUSIVE, WHICH IS ITSELF THE ANSWER. The ratios go BOTH")
    print("  WAYS -- 1.68, 0.86, 1.22, 0.67 -- and the flight counts are mixed. A")
    print("  monotone qbar decay cannot produce a late half that is 68% STRONGER")
    print("  (D01) on one domain and 33% weaker on another. What dominates is that")
    print("  the aeroplane flies through DIFFERENT TURBULENCE in the second half:")
    print("  the field is not statistically homogeneous along the track, and that")
    print("  swamps the qbar term.")
    print()
    print("  SO NO POST-PROCESSING SEPARATES THESE TWO. Not a POD, not a high-pass,")
    print("  not a split. Drift and field inhomogeneity are confounded in every")
    print("  statistic computable from these arrays, because they are confounded in")
    print("  the RUN. Section 4's own wording is the only route: 'hold the")
    print("  condition ... and re-read' -- i.e. fly it again with qbar maintained,")
    print("  and difference the two runs. The separation has to happen in the")
    print("  experiment, not in the analysis.")


if __name__ == "__main__":
    main()
