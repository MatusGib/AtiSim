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
    vertical_speed          barometric VSI. INERTIAL. It reads the rate of
                            change of geometric height, so an updraft that
                            CARRIES the aircraft up is read -- but through the
                            trajectory, on the next step. The wind vector itself
                            never enters the calculation.

Getting the rate group wrong in the other direction is the classic
overcorrection: feeding `omega - omega_gust` to a rate-damping loop makes the
autopilot chase a gust gradient no gyro can see.

**The accelerometer package is a SEPARATE function.** `n_x, n_y, n_z` are a
specific force, which is a force over a mass, so they need the controls and the
aircraft -- and the air-data computer has neither. A real aircraft has two boxes
here, and so does this module. Making `sense` depend on `dynamics` to fold them
in would be worse than a second function.

**Both readers are jitted**, which is a performance decision with a measured
reason rather than a reflex. Eager JAX dispatches every operation separately, and
`sense` is about twenty tiny operations on 3- and 4-element arrays, so the Python
dispatch overhead swamps the arithmetic completely: 6.56 ms eager against 0.028
ms jitted, a factor of 234. The live panel calls `sense` once per physics step
and `accelerometers` once per frame, so eager dispatch alone was costing about
44 ms of every 73 ms frame -- more than the whole instrument panel took to draw.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from atisim.aero import air_data
from atisim.aircraft import Aircraft
from atisim.dynamics import relative_velocity, specific_force
from atisim.state import Controls, State, quat_to_dcm, quat_to_euler

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
    vertical_speed: Array  # m/s, positive UP, inertial


@jax.jit
def sense(state: State, wind_ned: Array = STILL_AIR) -> AirData:
    """Read the instruments.

    `wind_ned` defaults to still air so that every still-air caller reads
    exactly as before, but it is a parameter rather than an assumption: passing
    the wrong wind is now a visible mistake instead of an invisible one.
    """
    airspeed, alpha, beta = air_data(relative_velocity(state.vel_body, state.quat, wind_ned))
    phi, theta, psi = quat_to_euler(state.quat)
    p, q, r = state.omega
    # Inertial, and deliberately not built from `relative_velocity`: the VSI
    # reads how fast the airframe is actually changing height, not how fast it
    # is moving through the air mass.
    vel_ned = quat_to_dcm(state.quat) @ state.vel_body
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
        vertical_speed=-vel_ned[2],
    )


class Accelerations(NamedTuple):
    """The accelerometer package. Body axes, in g.

    NOTE THE SIGN ASYMMETRY, which is inherited rather than a mistake: `n_z`
    follows the load-factor convention and reads +1 in level flight, so it is
    the NEGATED z component of the specific force. `n_x` and `n_y` are the raw
    components, positive forward and positive right.

    These are neither air-relative nor inertial in the sense the table above
    uses. They are a FORCE over a mass, so a gust moves them -- by changing the
    flow the wings see, which changes the lift.
    """

    n_x: Array  # g, positive FORWARD
    n_y: Array  # g, positive RIGHT
    n_z: Array  # g, positive UP-ish: +1 in level flight


@jax.jit
def accelerometers(
    state: State,
    controls: Controls,
    ac: Aircraft,
    wind_ned: Array = STILL_AIR,
    omega_gust: Array = STILL_AIR,
) -> Accelerations:
    """Read the accelerometer package.

    Separate from `sense` for the reason given in the module docstring: a
    specific force needs a mass and a set of deflections, and an air-data
    computer has neither.
    """
    n = specific_force(state, controls, ac, wind_ned, omega_gust)
    return Accelerations(n_x=n[0], n_y=n[1], n_z=-n[2])
