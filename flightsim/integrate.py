"""Fixed-step RK4 integration and scan-based rollout.

The carry is a `SimState`: rigid-body state, wind-model state, and PRNG key
together. Bundling them means `lax.scan` has one carry and `vmap` has one thing
to batch, which is what makes a Monte Carlo ensemble over turbulence
realisations a one-line change later.

Wind is sampled once per step and held constant across the four RK4 stages.
This is the standard treatment for Dryden and von Karman turbulence and avoids
splitting four keys per step.
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from flightsim.aircraft import Aircraft
from flightsim.dynamics import derivatives
from flightsim.state import Controls, State, quat_normalize
from flightsim.wind import WindState, zero_wind, zero_wind_state


class SimState(NamedTuple):
    """Rigid-body state, wind-model state, PRNG key, and the wind last applied.

    `wind_ned` and `omega_gust` are an OUTPUT cache, not model state: they record
    what the previous `step` actually blew on the aircraft. They are here so that
    a controller or a recorder can sense the air without calling the wind model a
    second time -- which would cost a duplicate evaluation and, for a stochastic
    model, split the key twice and yield a different realisation from the one the
    aircraft actually flew through.

    Consumers therefore see the wind from one step ago. At 50 Hz that is 20 ms of
    lag, which is what a real sensor gives you anyway.
    """

    state: State
    wind: WindState
    key: Array
    wind_ned: Array  # (3,) m/s NED, applied by the previous step
    omega_gust: Array  # (3,) rad/s body, applied by the previous step


def init_sim(state: State, key: Array) -> SimState:
    return SimState(
        state=state,
        wind=zero_wind_state(),
        key=key,
        wind_ned=jnp.zeros(3),
        omega_gust=jnp.zeros(3),
    )


def _axpy(x, y, a):
    """x + a*y over a pytree."""
    return jax.tree.map(lambda xi, yi: xi + a * yi, x, y)


def rk4_step(f, x, dt):
    """One classical RK4 stage set on a pytree state.

    Split out of `step` so the stage weights can be verified against a problem
    with a closed-form solution -- `step` is welded to `derivatives`, and
    conservation drift cannot distinguish a fourth-order scheme from a
    second-order one. `_axpy` maps over a pytree and an array is a leaf, so this
    also runs unchanged on a plain array right-hand side, which is what
    flightsim/verification.py uses.
    """
    k1 = f(x)
    k2 = f(_axpy(x, k1, dt / 2))
    k3 = f(_axpy(x, k2, dt / 2))
    k4 = f(_axpy(x, k3, dt))
    increment = jax.tree.map(
        lambda a, b, c, d: (a + 2.0 * b + 2.0 * c + d) / 6.0, k1, k2, k3, k4
    )
    return _axpy(x, increment, dt)


@partial(jax.jit, static_argnames=("wind_model",))
def step(
    sim: SimState,
    controls: Controls,
    dt: Array,
    ac: Aircraft,
    wind_model=zero_wind,
) -> SimState:
    """One RK4 step. The PRNG key is threaded through the wind model."""
    wind_ned, omega_gust, wind_state, key = wind_model(sim.wind, sim.state, sim.key, dt)

    def f(s: State) -> State:
        return derivatives(s, controls, ac, wind_ned, omega_gust)

    new_state = rk4_step(f, sim.state, dt)
    new_state = new_state._replace(quat=quat_normalize(new_state.quat))

    return SimState(
        state=new_state,
        wind=wind_state,
        key=key,
        wind_ned=wind_ned,
        omega_gust=omega_gust,
    )


@partial(jax.jit, static_argnames=("n_steps", "wind_model"))
def rollout(
    sim: SimState,
    controls: Controls,
    dt: Array,
    ac: Aircraft,
    n_steps: int,
    wind_model=zero_wind,
) -> tuple[SimState, State]:
    """Run n_steps with fixed controls.

    Returns the final SimState and the state trajectory, the latter stacked
    along a leading time axis of length n_steps.
    """

    def body(carry: SimState, _) -> tuple[SimState, State]:
        carry = step(carry, controls, dt, ac, wind_model=wind_model)
        return carry, carry.state

    return jax.lax.scan(body, sim, None, length=n_steps)


def batch_sim(state: State, keys: Array) -> SimState:
    """Replicate one initial state across a batch of PRNG keys.

    The batch axis exists from day one even at size 1, so that Monte Carlo over
    turbulence realisations needs no restructuring.
    """
    n = keys.shape[0]
    tiled = jax.tree.map(lambda x: jnp.broadcast_to(x, (n,) + x.shape), state)
    winds = jax.tree.map(lambda x: jnp.broadcast_to(x, (n,) + x.shape), zero_wind_state())
    return SimState(
        state=tiled,
        wind=winds,
        key=keys,
        wind_ned=jnp.zeros((n, 3)),
        omega_gust=jnp.zeros((n, 3)),
    )


def batched_rollout(
    sim: SimState,
    controls: Controls,
    dt: Array,
    ac: Aircraft,
    n_steps: int,
    wind_model=zero_wind,
) -> tuple[SimState, State]:
    """rollout vmapped over the leading batch axis of `sim`."""
    return jax.vmap(
        lambda s: rollout(s, controls, dt, ac, n_steps, wind_model=wind_model)
    )(sim)
