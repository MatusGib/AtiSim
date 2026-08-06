"""What the aircraft's instruments report.

This module exists because of a specific bug class. A controller or a display
that wants "airspeed" almost always reaches for `state.vel_body`, which is
*inertial*. In still air that is correct and the mistake is invisible. Under a
wind field it silently becomes groundspeed, and an angle of attack taken the
same way is off by up to 7 degrees in a vortex encounter -- measured, not
estimated. Both defects lived in this repo for three sessions precisely because
still-air tests cannot see them.

The fix is to make the right quantity the only one on offer. `sense` is the sole
supported way to ask the aircraft what it is doing, and it cannot be called
without saying what the air is doing.

**Which quantities are air-relative, and which are not, is physics, not taste.**

    airspeed, alpha, beta   pitot tube and flow vanes. AIR-RELATIVE. These
                            measure the flow over the airframe, so a gust
                            changes them immediately and completely.
    phi, theta, psi         inertial platform. INERTIAL. An attitude gyro is
                            unaware of the air it is flying through.
    p, q, r                 rate gyros. INERTIAL. A gyro measures the airframe's
                            own rotation. `omega_gust` is a gradient across the
                            span and chord -- it changes the flow the wings see,
                            not the rate the gyro reads, which is why
                            `aero.py` gets `omega - omega_gust` and a controller
                            does not.
    altitude                barometric. Treated as geometric here; the
                            atmosphere model has no pressure-error term.

Getting the rate group wrong in the other direction is the classic
overcorrection: feeding `omega - omega_gust` to a rate-damping loop makes the
autopilot chase a gust gradient no gyro can see.
"""

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from flightsim.aero import air_data
from flightsim.dynamics import relative_velocity
from flightsim.state import State, quat_to_euler

STILL_AIR = jnp.zeros(3)


class AirData(NamedTuple):
    """One sensor set. Air-relative where a real sensor would be, inertial elsewhere."""

    airspeed: Array  # m/s, true airspeed -- AIR-RELATIVE
    alpha: Array  # rad -- AIR-RELATIVE
    beta: Array  # rad -- AIR-RELATIVE
    phi: Array  # rad, inertial
    theta: Array  # rad, inertial
    psi: Array  # rad, inertial
    p: Array  # rad/s, inertial
    q: Array  # rad/s, inertial
    r: Array  # rad/s, inertial
    altitude: Array  # m


def sense(state: State, wind_ned: Array = STILL_AIR) -> AirData:
    """Read the instruments.

    `wind_ned` defaults to still air so that every still-air caller reads
    exactly as before, but it is a parameter rather than an assumption: passing
    the wrong wind is now a visible mistake instead of an invisible one.
    """
    airspeed, alpha, beta = air_data(relative_velocity(state.vel_body, state.quat, wind_ned))
    phi, theta, psi = quat_to_euler(state.quat)
    p, q, r = state.omega
    return AirData(
        airspeed=airspeed,
        alpha=alpha,
        beta=beta,
        phi=phi,
        theta=theta,
        psi=psi,
        p=p,
        q=q,
        r=r,
        altitude=-state.pos_ned[2],
    )
