"""Freeze one trajectory from the FLAT-EARTH model, before the Earth is added.

Run ONCE, before any of the WGS-84 work lands. After Task 8 changes `State`
this script will no longer run, and that is fine -- its output is checked in.
Task 13 compares the FLAT configuration of the new plant against this file.

The comparison it enables is NOT a bit-identity test. See the design doc
section 8: an ECEF state cannot reproduce a NED state bit-for-bit. What the
file supports is the weaker and honest claim -- that the difference is
round-off and does not grow with the run.
"""

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from atisim.aircraft import CRUISE, REGISTRY          # noqa: E402
from atisim.integrate import init_sim, logged_rollout  # noqa: E402
from atisim.trim import trim, trimmed_controls, trimmed_state  # noqa: E402

import jax  # noqa: E402

OUT = ROOT / "atisim" / "tests" / "data" / "pre_earth_trajectory.npz"

DT = 0.02
N_STEPS = 1000   # 20 s, the same window the JSBSim layer comparison uses


def main():
    # NOTE the key and the accessor: the registry keys are `boeing747`, not
    # `b747`, and CRUISE values are plain dicts, not attribute objects.
    ac = REGISTRY["boeing747"]
    airspeed = CRUISE["boeing747"]["airspeed"]     # 235.9152 m/s
    altitude = CRUISE["boeing747"]["altitude"]     # 12192.0 m

    x, residual = trim(airspeed, altitude, ac)
    assert np.max(np.abs(np.asarray(residual))) < 1e-9, "pre-Earth trim did not converge"
    alpha, elevator, throttle = (float(v) for v in x)

    sim = init_sim(
        trimmed_state(jnp.asarray(alpha), jnp.asarray(airspeed), jnp.asarray(altitude)),
        jax.random.PRNGKey(0),
    )
    controls = trimmed_controls(jnp.asarray(elevator), jnp.asarray(throttle))
    _, log = logged_rollout(sim, controls, DT, ac, N_STEPS)

    np.savez_compressed(
        OUT,
        dt=DT, n_steps=N_STEPS,
        airspeed=float(airspeed), altitude=float(altitude),
        trim=np.asarray(x, dtype=float),
        pos_ned=np.asarray(log.state.pos_ned, dtype=float),
        vel_body=np.asarray(log.state.vel_body, dtype=float),
        quat=np.asarray(log.state.quat, dtype=float),
        omega=np.asarray(log.state.omega, dtype=float),
    )
    print(f"wrote {OUT.relative_to(ROOT)}  ({N_STEPS} samples)")


if __name__ == "__main__":
    main()
