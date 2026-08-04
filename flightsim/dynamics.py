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
