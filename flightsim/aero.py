"""Aerodynamic force and moment build-up.

The one rule that matters: this module never sees inertial velocity. It is given
velocity and angular rate *relative to the air mass*, so turbulence is a matter
of what the caller passes in, not a change to anything here.

Sign conventions:
  alpha  positive nose-up relative to the relative wind
  beta   positive nose-left (relative wind from the right)
  de     positive trailing-edge down, so CLde > 0 and Cmde < 0
  da     positive gives positive (right-wing-down) roll, so Clda > 0
  dr     positive trailing-edge left, so CYdr > 0 and Cndr < 0
"""

import jax.numpy as jnp
from jax import Array

from flightsim.aircraft import Aircraft
from flightsim.atmosphere import RHO0
from flightsim.state import Controls

# Airspeed floor. Purely a NaN guard for the zero-velocity case -- alpha, beta
# and the non-dimensional rates all divide by V. Far below any flight speed, so
# it never binds in normal operation.
V_MIN = 1.0  # m/s


def air_data(vel_rel: Array) -> tuple[Array, Array, Array]:
    """(true airspeed, alpha, beta) from body-axis velocity relative to air."""
    u, v, w = vel_rel
    V = jnp.maximum(jnp.linalg.norm(vel_rel), V_MIN)
    alpha = jnp.arctan2(w, u)
    beta = jnp.arcsin(jnp.clip(v / V, -1.0, 1.0))
    return V, alpha, beta


def coefficients(
    vel_rel: Array, omega_rel: Array, controls: Controls, ac: Aircraft
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """(CL, CD, CY, Cl, Cm, Cn). Lift and drag wind-axis, the rest body-axis."""
    V, alpha, beta = air_data(vel_rel)
    p, q, r = omega_rel

    # Non-dimensional rates. Span for the lateral pair, chord for pitch.
    p_hat = p * ac.b / (2.0 * V)
    q_hat = q * ac.c / (2.0 * V)
    r_hat = r * ac.b / (2.0 * V)

    de, da, dr = controls.elevator, controls.aileron, controls.rudder

    CL = ac.CL0 + ac.CLa * alpha + ac.CLq * q_hat + ac.CLde * de
    Cm = ac.Cm0 + ac.Cma * alpha + ac.Cmq * q_hat + ac.Cmde * de
    CD = ac.CD0 + CL**2 / (jnp.pi * ac.e * ac.AR)

    CY = ac.CYb * beta + ac.CYp * p_hat + ac.CYr * r_hat + ac.CYdr * dr
    Cl = (
        ac.Clb * beta
        + ac.Clp * p_hat
        + ac.Clr * r_hat
        + ac.Clda * da
        + ac.Cldr * dr
    )
    Cn = (
        ac.Cnb * beta
        + ac.Cnp * p_hat
        + ac.Cnr * r_hat
        + ac.Cnda * da
        + ac.Cndr * dr
    )
    return CL, CD, CY, Cl, Cm, Cn


def aero_forces_moments(
    vel_rel: Array,
    omega_rel: Array,
    controls: Controls,
    ac: Aircraft,
    rho: Array,
) -> tuple[Array, Array]:
    """Body-axis aerodynamic force (N) and moment (N.m)."""
    V, alpha, beta = air_data(vel_rel)
    qbar = 0.5 * rho * V**2
    CL, CD, CY, Cl, Cm, Cn = coefficients(vel_rel, omega_rel, controls, ac)

    lift = qbar * ac.S * CL
    drag = qbar * ac.S * CD
    side = qbar * ac.S * CY

    # Lift and drag are wind-axis; rotate into body axes through alpha and beta.
    # Side force is already body-axis, as the lateral derivatives are defined.
    ca, sa = jnp.cos(alpha), jnp.sin(alpha)
    cb, sb = jnp.cos(beta), jnp.sin(beta)
    force = jnp.array(
        [
            -drag * ca * cb + lift * sa,
            -drag * sb + side,
            -drag * sa * cb - lift * ca,
        ]
    )

    moment = qbar * ac.S * jnp.array([ac.b * Cl, ac.c * Cm, ac.b * Cn])
    return force, moment


def thrust_force(controls: Controls, ac: Aircraft, rho: Array) -> Array:
    """Body-axis thrust, assumed aligned with the body x axis.

    Deliberately not an engine model: throttle times maximum thrust, with a
    density lapse. The exponent is 1 for a normally-aspirated piston and around
    0.7-0.8 for a high-bypass turbofan.
    """
    magnitude = controls.throttle * ac.max_thrust * (rho / RHO0) ** ac.thrust_lapse
    return jnp.array([magnitude, 0.0, 0.0])
