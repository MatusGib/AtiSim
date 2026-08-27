"""Every channel the UI plots, computed once, from the recorded run.

Separate from `figures.py` on purpose. The standing rule is that the framework
imports `atisim` and never the reverse, and that everything displayed is
computed by a tested function -- so the arithmetic lives here, the Plotly calls
live next door, and the Dash app does layout and nothing else.

**Nothing here re-evaluates the wind model.** Every channel is built from the
`wind_ned` and `omega_gust` columns the run actually flew. Re-evaluating is exact
for a deterministic field and silently wrong for a stochastic one, which would
split the PRNG key again and return a different realisation -- `SimState`'s own
docstring warns about it, and `vortex_viz._measure` re-invokes the model only
because it receives a bare `State` trajectory rather than a logged one. An
artifact has no such excuse.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from atisim import checks, dynamics, earth, viz
from atisim.aero import air_data
from atisim.aircraft import Aircraft
from atisim.units import RAD2DEG


class Series(NamedTuple):
    """Plot-ready channels. Degrees where a reader thinks in degrees.

    `alpha_inertial_deg` is here for one reason: overlaid on `alpha_deg` it is
    the single most direct picture of the air-relative/inertial confusion that
    lived in this repo for three sessions. In a Parks encounter the two differ by
    up to 7 degrees; in still air they coincide exactly, which is why no still-air
    test could see the bug.
    """

    t: np.ndarray  # s
    north: np.ndarray  # m, LOCAL NED from the run anchor
    east: np.ndarray  # m, LOCAL NED from the run anchor
    altitude: np.ndarray  # m, GEODETIC -- not -pos_d, which is a tangent-plane height
    w_up: np.ndarray  # m/s, vertical gust POSITIVE UP (= -wind_d)
    q_gust_deg: np.ndarray  # deg/s, SIM TRUTH -- no instrument can sense it
    alpha_deg: np.ndarray  # deg, AIR-RELATIVE
    alpha_inertial_deg: np.ndarray  # deg, what a ground-relative reading gives
    beta_deg: np.ndarray  # deg, AIR-RELATIVE
    theta_deg: np.ndarray  # deg, inertial
    q_deg: np.ndarray  # deg/s, inertial
    n_z: np.ndarray  # g, body-normal
    elevator_deg: np.ndarray  # deg
    airspeed: np.ndarray  # m/s, true


def build(
    traj: viz.Trajectory, ac: Aircraft, earth_model: earth.EarthModel
) -> Series:
    """Derive every plotted channel from one recorded run.

    The anchor rides along on `traj`, so the ground track and the altitude are
    referred to the frame the run was actually flown in. `earth_model` does not
    -- it is a statement about the plant rather than about the record -- and
    `checks.load_factor_series` below re-inverts the plant, so it is asked for.
    No default: `FLAT` and `WGS84_J2` give different load factors.
    """
    air = viz.derived(traj)  # air-relative, from the RECORDED wind

    vel_body = jnp.asarray(traj.vel_body)
    _, alpha_inertial, _ = jax.vmap(air_data)(vel_body)
    track = viz.pos_ned(traj)

    return Series(
        t=np.asarray(traj.t),
        north=track[:, 0],
        east=track[:, 1],
        altitude=air.altitude,
        w_up=-np.asarray(traj.wind_ned)[:, 2],
        q_gust_deg=np.asarray(traj.omega_gust)[:, 1] * RAD2DEG,
        alpha_deg=air.alpha * RAD2DEG,
        alpha_inertial_deg=np.asarray(alpha_inertial) * RAD2DEG,
        beta_deg=air.beta * RAD2DEG,
        theta_deg=air.theta * RAD2DEG,
        q_deg=np.asarray(traj.omega)[:, 1] * RAD2DEG,
        n_z=checks.load_factor_series(traj, ac, earth_model),
        elevator_deg=np.asarray(traj.controls)[:, 0] * RAD2DEG,
        airspeed=air.airspeed,
    )


def envelope(x: np.ndarray, y: np.ndarray, target: int) -> tuple[np.ndarray, np.ndarray]:
    """Decimate to about `target` points KEEPING BOTH EXTREMES of every bucket.

    **Never stride-decimate a channel whose peak is the result.** Measured on the
    canonical vortex run: taking every 16th sample reports a peak load factor of
    -1.164 g against the true -1.235 g, a 5.7% error on the number PROJECT.md
    section 4 headlines. Min/max envelope decimation loses 0.0000% at every level
    tested -- 2000, 1000, 500, 250 and 100 points -- because the peak is by
    definition an extreme and this keeps every bucket's extremes.

    And the stride error cannot be bounded by using more points: it is 4.07% at
    500, 5.72% at 250 and 4.07% at 100. That is aliasing luck, not a convergent
    approximation, so there is no "enough samples" threshold to pick.

    Points come back in original index order, so x stays monotone and a line plot
    is still a line rather than a zigzag between buckets.
    """
    n = len(x)
    if n <= target or target < 2:
        return x, y
    bucket = max(1, n // (target // 2))
    keep = []
    for start in range(0, n, bucket):
        chunk = y[start:start + bucket]
        if len(chunk) == 0:
            continue
        lo, hi = start + int(np.argmin(chunk)), start + int(np.argmax(chunk))
        keep.extend((lo, hi) if lo <= hi else (hi, lo))
    keep = np.unique(np.array(keep + [0, n - 1], dtype=int))
    return x[keep], y[keep]
