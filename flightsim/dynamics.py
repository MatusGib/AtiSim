"""6-DOF Newton-Euler equations of motion.

    vdot     = F/m + g_body - omega x v
    omegadot = I^-1 (M - omega x (I omega))
    posdot   = DCM v
    quatdot  = 0.5 q (x) [0, omega]

The Coriolis terms use the *inertial* velocity and angular rate. Only the
aerodynamics see air-relative quantities.
"""

import jax.numpy as jnp
from jax import Array

from flightsim.aero import aero_forces_moments, thrust_force
from flightsim.aircraft import Aircraft
from flightsim.atmosphere import G0, density, speed_of_sound
from flightsim.state import Controls, State, quat_derivative, quat_to_dcm


def relative_velocity(vel_body: Array, quat: Array, wind_ned: Array) -> Array:
    """Body-axis velocity relative to the surrounding air mass."""
    dcm = quat_to_dcm(quat)  # body -> NED
    return vel_body - dcm.T @ wind_ned


def derivatives(
    state: State,
    controls: Controls,
    ac: Aircraft,
    wind_ned: Array,
    omega_gust: Array,
) -> State:
    """State derivative. Returns a State whose fields are time derivatives."""
    dcm = quat_to_dcm(state.quat)

    vel_rel = relative_velocity(state.vel_body, state.quat, wind_ned)
    omega_rel = state.omega - omega_gust

    altitude = -state.pos_ned[2]
    rho = density(altitude)
    force, moment = aero_forces_moments(
        vel_rel, omega_rel, controls, ac, rho, speed_of_sound(altitude)
    )
    force = force + thrust_force(controls, ac, rho)

    gravity_body = dcm.T @ jnp.array([0.0, 0.0, G0])

    accel = force / ac.mass + gravity_body - jnp.cross(state.omega, state.vel_body)
    omega_dot = ac.inertia_inv @ (
        moment - jnp.cross(state.omega, ac.inertia @ state.omega)
    )

    return State(
        pos_ned=dcm @ state.vel_body,
        vel_body=accel,
        quat=quat_derivative(state.quat, state.omega),
        omega=omega_dot,
    )


def specific_force(
    state: State,
    controls: Controls,
    ac: Aircraft,
    wind_ned: Array,
    omega_gust: Array,
) -> Array:
    """Body-axis specific force in g: what a three-axis accelerometer at the CG reads.

    Specific force is the aerodynamic plus propulsive force over mass and
    excludes gravity. `derivatives` computes exactly that quantity as
    `force / ac.mass` and then discards it inside the sum at the top of this
    module, so it is recovered here by inverting that sum:

        a_spec = vdot_body - g_body + omega x vel_body

    Inverting rather than recomputing `force / mass` is deliberate. It cannot
    silently disagree with the plant if a force term is ever added to
    `derivatives`, because it inverts whatever `derivatives` actually did.

    Signs are body axes throughout: +x forward, +y right, +z down. Note that
    `load_factor` NEGATES the z component, because the load-factor convention is
    +1 in level flight while a_spec[2] is negative there.
    """
    d = derivatives(state, controls, ac, wind_ned, omega_gust)
    gravity_body = quat_to_dcm(state.quat).T @ jnp.array([0.0, 0.0, G0])
    return (d.vel_body - gravity_body + jnp.cross(state.omega, state.vel_body)) / G0


def load_factor(
    state: State,
    controls: Controls,
    ac: Aircraft,
    wind_ned: Array,
    omega_gust: Array,
) -> Array:
    """Normal load factor n_z. +1 in level flight, 0 in free fall.

    Body-normal, not flight-path-normal: this is the quantity Wingrove & Bach's
    Fig. 8 is built from, since DFDR "normal acceleration" is what an
    accelerometer reads. In trimmed level flight it is cos(theta), not 1.

    Needed because the turbulence work's headline comparison is stated in load
    factor, and nothing in the package produced it before.
    """
    return -specific_force(state, controls, ac, wind_ned, omega_gust)[2]
