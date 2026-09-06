"""FALSIFICATION TEST: is the LES load discrepancy just our frozen lift slope?

PROJECT.md section 4 ("The frozen lift-curve slope: 47-68% of the LES
discrepancy, measured") made a prediction and named the run that would kill it.
That section's heading used to end "...and why it probably IS the LES
discrepancy". THIS SCRIPT IS WHY IT NO LONGER DOES.

THE RESULT, so nobody re-runs it to find out. D03, 16 flights:

    baseline boeing747   rms 0.0905 g  ratio 1.427  short period 0.1647 Hz
    full PG correction   rms 0.0777 g  ratio 1.225  short period 0.1272 Hz   47% closed
    lift-only            rms 0.0720 g  ratio 1.135  short period 0.1568 Hz   68% closed
    pure CL_alpha linearity would have given rms 0.0594 g, ratio 0.937

The baseline reproduces the weekly worktree's 0.09049 exactly. The frozen slope
is the DOMINANT identified contributor -- 47-68% depending on whether the
resonance is held fixed -- and it is OUR error, not Yoshimura's. It does NOT
explain the discrepancy on its own: 1.14-1.23 survives the best correction.

Note the full PG correction does WORSE than lift-only, because scaling C_m_alpha
drops the short period into more energetic turbulence and adds load back. A more
complete compressibility correction makes the agreement worse. That is a finding
about how gust load is set here, not an artefact.

THE CLAIM. `boeing747` is linearised at M 0.80 and `aero.py` carries Mach only
into `wave_drag`, so `CL_alpha` is used unchanged at the LES condition of
M 0.406. Prandtl-Glauert says a slope tabulated at M 0.80 is 1.523x too large
there. Gust load goes linearly as `CL_alpha`. The measured AtiSim/Yoshimura rms
ratio on the two domains that resolve the turbulence is 1.427 (D03) and 1.420
(D04). Those numbers are close enough that the discrepancy may be entirely ours.

THE TEST. Fly the identical field, path, condition and seed set with the six
longitudinal derivatives that Prandtl-Glauert scales -- CLa, CLq, CLde, Cma,
Cmq, Cmde -- multiplied by beta(0.80)/beta(0.406) = 0.6565, and see whether the
ratio collapses toward 1.

WHAT WOULD FALSIFY THE CLAIM. If the corrected ratio stays near 1.42, the
discrepancy is NOT the frozen slope and something else is wrong -- the aircraft
mismatch, the field reader, or one of the two codes.

WHAT THIS IS NOT. The corrected aircraft is a SENSITIVITY CASE, not a validated
entry. The correction is first-order thin-aerofoil compressibility, DECLARED,
not a derivative set read from any table. It is not added to `REGISTRY` on disk
and no result from it belongs in section 4 as a property of the 747. Note also
that scaling `Cma` moves the short period, which is itself part of what the
comparison measures -- both frequencies are printed so the reader can see it.

Run (D03 and D04 are the resolved domains; D01/D02 do not resolve the gusts):

    PYTHONPATH=<abs worktree root> .venv/Scripts/python.exe \
        scripts/les_mach_test.py --domain D04 --flights 16
"""

import argparse
import sys

import numpy as np
import jax.numpy as jnp

import atisim  # noqa: F401  -- enables x64 before any array is made
from atisim.aircraft import REGISTRY, CRUISE
from atisim.atmosphere import speed_of_sound

# The two Mach numbers the correction spans.
M_TABULATED = 0.80      # where CR-2144 FC9 linearised the 747
LES_ALT = 3000.0        # m, Yoshimura's flight level
LES_SPEED = 133.5       # m/s, their airspeed

# Prandtl-Glauert scales lift-derived derivatives as 1/sqrt(1 - M^2). These six
# are the longitudinal ones that come from lift on the wing and tail. CD0 is
# deliberately NOT scaled: it is friction and form drag, not lift.
PG_SCALED = ("CLa", "CLq", "CLde", "Cma", "Cmq", "Cmde")

# The LIFT-ONLY set exists because the full PG correction is not a
# single-variable experiment: scaling Cma moves the SHORT PERIOD, and where the
# short period sits in the gust spectrum is itself part of what the LES
# comparison measures. Measured session 27: the full correction drops the short
# period 0.1647 -> 0.1272 Hz, toward the energetic low-frequency end of the LES
# spectrum, which pushes the load back UP and partly cancels the lift
# reduction. Scaling only the lift derivatives leaves the pitch stiffness -- and
# so the resonance -- where it was, which separates the two effects.
# It is NOT more physical than the full correction. It is more DIAGNOSTIC.
LIFT_ONLY = ("CLa", "CLq", "CLde")


def pg_factor(m_from: float, m_to: float) -> float:
    """Value at `m_to` divided by value at `m_from`, thin-aerofoil PG."""
    beta = lambda m: np.sqrt(1.0 - m * m)  # noqa: E731
    return float(beta(m_from) / beta(m_to))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domain", default="D04", choices=("D01", "D02", "D03", "D04"))
    ap.add_argument("--flights", type=int, default=16)
    ap.add_argument("--outdir", default="runs/cat")
    ap.add_argument("--lift-only", action="store_true",
                    help="scale only CLa/CLq/CLde, leaving pitch stiffness and "
                         "so the short period untouched -- separates the lift "
                         "effect from the resonance shift")
    ap.add_argument("--baseline", action="store_true",
                    help="fly the UNCORRECTED 747, to reproduce the number "
                         "being tested in this tree before changing anything")
    args = ap.parse_args()

    print("atisim imported from:", atisim.__file__)

    a = float(speed_of_sound(jnp.array(LES_ALT)))
    m_les = LES_SPEED / a
    k = pg_factor(M_TABULATED, m_les)
    print(f"\nLES condition: {LES_ALT:.0f} m, {LES_SPEED} m/s -> M {m_les:.4f}")
    print(f"derivatives tabulated at M {M_TABULATED:.2f}")
    print(f"Prandtl-Glauert factor applied to {', '.join(PG_SCALED)}: {k:.5f}"
          f"   (i.e. 1/{1.0 / k:.4f})")

    base = REGISTRY["boeing747"]
    if args.baseline:
        name = "boeing747"
        print("\n--baseline: flying the UNCORRECTED entry.")
    else:
        fields = LIFT_ONLY if args.lift_only else PG_SCALED
        name = "boeing747_liftonly" if args.lift_only else "boeing747_pgtest"
        which = "LIFT ONLY" if args.lift_only else "FULL PG SET"
        print(f"\nscaling {which}: {', '.join(fields)}")
        scaled = {f: jnp.asarray(getattr(base, f)) * k for f in fields}
        for f in fields:
            print(f"    {f:6s} {float(getattr(base, f)):+10.4f} -> "
                  f"{float(scaled[f]):+10.4f}")
        # DECLARED sensitivity case, injected for this process only. It is not
        # written to aircraft.py and does not survive the run.
        REGISTRY[name] = base._replace(**scaled)
        CRUISE[name] = dict(CRUISE["boeing747"])

    # Hand off to the real runner so the field reader, the path, the filtering
    # and the checks are byte-identical to the run being tested.
    from les_flight import main as les_main  # noqa: E402
    sys.argv = ["les_flight.py", "--domain", args.domain,
                "--flights", str(args.flights), "--aircraft", name,
                "--outdir", args.outdir]
    les_main()


if __name__ == "__main__":
    sys.path.insert(0, "scripts")
    main()
