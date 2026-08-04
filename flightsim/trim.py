"""Steady level flight trim.

Three unknowns -- angle of attack, elevator, throttle -- for wings-level flight
at a target true airspeed and altitude, with gamma = 0 (so theta = alpha),
beta = 0 and no body rates. The residual is the resulting [udot, wdot, qdot],
all three of which must vanish.

Newton with a forward-mode Jacobian. Pure JAX, so it is jittable and vmappable
over a grid of (V, h) if that is ever wanted. Every run starts from here: an
untrimmed start accelerates or climbs away for no reason and makes autopilot
tuning meaningless.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

from flightsim.aircraft import Aircraft
from flightsim.dynamics import derivatives
from flightsim.state import Controls, State, euler_to_quat


def trimmed_state(alpha: Array, airspeed: Array, altitude: Array) -> State:
    """Wings-level state at the given alpha, flying level (theta = alpha)."""
    return State(
        pos_ned=jnp.array([0.0, 0.0, -altitude]),
        vel_body=airspeed * jnp.array([jnp.cos(alpha), 0.0, jnp.sin(alpha)]),
        quat=euler_to_quat(jnp.array(0.0), alpha, jnp.array(0.0)),
        omega=jnp.zeros(3),
    )


def trimmed_controls(elevator: Array, throttle: Array) -> Controls:
    return Controls(
        elevator=elevator,
        aileron=jnp.array(0.0),
        rudder=jnp.array(0.0),
        throttle=throttle,
    )


def residual(x: Array, airspeed: Array, altitude: Array, ac: Aircraft) -> Array:
    """[udot, wdot, qdot] for the candidate trim vector [alpha, elevator, throttle]."""
    alpha, elevator, throttle = x
    d = derivatives(
        trimmed_state(alpha, airspeed, altitude),
        trimmed_controls(elevator, throttle),
        ac,
        jnp.zeros(3),
        jnp.zeros(3),
    )
    return jnp.array([d.vel_body[0], d.vel_body[2], d.omega[1]])


@partial(jax.jit, static_argnames=("iterations",))
def trim(
    airspeed: Array,
    altitude: Array,
    ac: Aircraft,
    guess: Array = None,
    iterations: int = 40,
) -> tuple[Array, Array]:
    """Solve for [alpha, elevator, throttle]. Returns (solution, final residual)."""
    x0 = jnp.array([0.05, 0.0, 0.5]) if guess is None else guess

    def step(x, _):
        r = residual(x, airspeed, altitude, ac)
        jacobian = jax.jacfwd(residual)(x, airspeed, altitude, ac)
        return x - jnp.linalg.solve(jacobian, r), None

    x, _ = jax.lax.scan(step, x0, None, length=iterations)
    return x, residual(x, airspeed, altitude, ac)
