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
                            atmosphere model has no pressure-error term. It is
                            now the GEODETIC height, `state.altitude`, not
                            `-pos_ned[2]`: the tangent plane falls away from the
                            ellipsoid as d^2/2R, which is 785 m at 100 km of
                            ground track, and an altimeter does not read that.
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

**Both readers take the run's `anchor`, and neither has a default for it.** The
state is ECEF now, so nothing in here can be read without saying where on the
Earth the aircraft is: the local vertical that the attitude, the VSI and the
altimeter are all referred to is a function of position. `earth.py` deliberately
ships no default anchor, and inventing one here would put a silently chosen
latitude under every readout.
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from atisim import earth
from atisim.aero import air_data
from atisim.aircraft import Aircraft
from atisim.dynamics import specific_force
from atisim.state import (
    Controls,
    State,
    altitude as geodetic_altitude,
    dcm_body_to_ned,
    matrix_to_euler,
)

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
def sense(state: State, anchor: earth.Anchor, wind_ned: Array = STILL_AIR) -> AirData:
    """Read the instruments.

    `wind_ned` defaults to still air so that every still-air caller reads
    exactly as before, but it is a parameter rather than an assumption: passing
    the wrong wind is now a visible mistake instead of an invisible one.
    `anchor` has NO default, because a local vertical picked for the caller is
    exactly the invisible mistake this module exists to prevent.

    ONE body -> NED matrix is formed and used three times. Every quantity below
    that is not already in body axes -- the incidence angles, the Euler set, the
    NED velocity -- is referred to the local frame AT THE AIRCRAFT, which is not
    the anchor's frame once the two are any distance apart. Going through a
    quaternion for `dynamics.relative_velocity_ned` and back would be a matrix ->
    quat -> matrix round trip for arithmetic that is one line; the frame it
    documents is asserted by using the named `dcm_body_to_ned` instead.
    """
    dcm_b2n = dcm_body_to_ned(state, anchor)
    # AIR-RELATIVE, exactly `dynamics.relative_velocity_ned` with the matrix
    # already in hand -- and in the LOCAL frame, which is what makes `wind_ned`
    # mean what its name says.
    vel_rel = state.vel_body - dcm_b2n.T @ wind_ned
    airspeed, alpha, beta = air_data(vel_rel)
    phi, theta, psi = matrix_to_euler(dcm_b2n)
    p, q, r = state.omega
    # Inertial, and deliberately not built from `vel_rel`: the VSI reads how
    # fast the airframe is actually changing height, not how fast it is moving
    # through the air mass.
    vel_ned = dcm_b2n @ state.vel_body
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
        # GEODETIC, not -pos_ned[2]. The two differ by 785 m at 100 km of ground
        # track, and it is the geodetic height an altimeter is calibrated to.
        altitude=geodetic_altitude(state, anchor),
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


@partial(jax.jit, static_argnames=("earth_model",))
def accelerometers(
    state: State,
    controls: Controls,
    ac: Aircraft,
    anchor: earth.Anchor,
    earth_model: earth.EarthModel,
    wind_ned: Array = STILL_AIR,
    omega_gust: Array = STILL_AIR,
) -> Accelerations:
    """Read the accelerometer package.

    Separate from `sense` for the reason given in the module docstring: a
    specific force needs a mass and a set of deflections, and an air-data
    computer has neither.

    `earth_model` is a STATIC argument, matching `dynamics` and `integrate`: it
    is a hashable NamedTuple of str/float, so its branches resolve at trace time
    and the jit this module's docstring measures is unaffected.
    """
    n = specific_force(
        state, controls, ac, wind_ned, omega_gust, anchor, earth_model
    )
    return Accelerations(n_x=n[0], n_y=n[1], n_z=-n[2])
