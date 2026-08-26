"""Fixed-step RK4 integration and scan-based rollout.

The carry is a `SimState`: rigid-body state, wind-model state, and PRNG key
together. Bundling them means `lax.scan` has one carry and `vmap` has one thing
to batch, which is what makes a Monte Carlo ensemble over turbulence
realisations a one-line change later.

Wind is sampled once per step and held constant across the four RK4 stages.

THIS IS A DESIGN CHOICE, NOT A CITATION. Earlier text here called it "the
standard treatment for Dryden and von Karman turbulence"; no source saying so
exists anywhere in this repository and none was found when one was looked for.
That matters more than an ordinary uncited line would, because this is the sole
stated justification for a choice that costs the scheme three orders of accuracy
in a spatially varying field (ASSUMPTIONS.md E4). So here is the reasoning
instead, with no appeal to authority:

  A Dryden or von Karman field is a STOCHASTIC PROCESS whose realisation is
  drawn from a PRNG key, not a function that can be evaluated twice at the same
  argument and give the same answer. Re-sampling it inside the RK4 stages means
  drawing four times per step, which changes the realisation the aircraft flies
  through as a function of the step size -- so refining dt would no longer be
  refining the same problem, and a convergence study would be measuring the
  noise process rather than the integrator. Holding one draw per step keeps the
  realisation a property of the key alone, which is what makes an ensemble over
  keys mean anything and what lets a batch of keys vary only the stochastic part
  of a composed field.

  For a DETERMINISTIC spatial field the same argument does not apply and the
  hold is simply first-order-accurate where per-stage sampling would be fourth.
  That cost is measured, in E4, rather than argued about: 1.05 against 3.99 in
  still air, and 4.05 with the hold removed.

It also avoids splitting four keys per step, which is a cost rather than a
justification.
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from atisim.aircraft import Aircraft
from atisim.dynamics import derivatives
from atisim.loads import CoeffIncrement, zero_increment
from atisim.state import Controls, State, quat_normalize
from atisim.wind import WindState, zero_wind, zero_wind_state


class SimState(NamedTuple):
    """Rigid-body state, wind-model state, PRNG key, and the loads last applied.

    `wind_ned`, `omega_gust` and `increment` are an OUTPUT cache, not model
    state: they record what the previous `step` actually blew on the aircraft.
    They are here so that a controller or a recorder can sense the air without
    calling the wind or load model a second time -- which would cost a duplicate
    evaluation and, for a stochastic model, split the key twice and yield a
    different realisation from the one the aircraft actually flew through.

    Consumers therefore see the wind from one step ago. At 50 Hz that is 20 ms of
    lag, which is what a real sensor gives you anyway.
    """

    state: State
    wind: WindState
    key: Array
    wind_ned: Array  # (3,) m/s NED, applied by the previous step
    omega_gust: Array  # (3,) rad/s body, applied by the previous step
    increment: CoeffIncrement  # coefficients applied by the previous step


def init_sim(state: State, key: Array) -> SimState:
    return SimState(
        state=state,
        wind=zero_wind_state(),
        key=key,
        wind_ned=jnp.zeros(3),
        omega_gust=jnp.zeros(3),
        increment=zero_increment(),
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
    atisim/verification.py uses.
    """
    k1 = f(x)
    k2 = f(_axpy(x, k1, dt / 2))
    k3 = f(_axpy(x, k2, dt / 2))
    k4 = f(_axpy(x, k3, dt))
    increment = jax.tree.map(
        lambda a, b, c, d: (a + 2.0 * b + 2.0 * c + d) / 6.0, k1, k2, k3, k4
    )
    return _axpy(x, increment, dt)


@partial(jax.jit, static_argnames=("wind_model", "load_model", "stage_sampled"))
def step(
    sim: SimState,
    controls: Controls,
    dt: Array,
    ac: Aircraft,
    wind_model=zero_wind,
    load_model=None,
    stage_sampled: bool = False,
) -> SimState:
    """One RK4 step. The PRNG key is threaded through the wind model.

    `load_model` maps a `State` to a `loads.CoeffIncrement`. It is sampled once
    per step and held across the four RK4 stages, exactly as the wind is and for
    the same reason. Omitting it gives an exact zero increment, which makes the
    step bit-identical to one taken before this feature existed.

    Two names for the one increment, deliberately. The CACHED value is always a
    `CoeffIncrement`, because `lax.scan` needs the carry's pytree structure to be
    the same on every iteration and a `None` leaf would change it. The value
    handed to `derivatives` is `None` when no load model was given, so the
    default path adds nothing at all rather than adding four exact zeros.
    `load_model` is static, so this branch is resolved at trace time.
    """
    # The wind contract gained an OPTIONAL fifth return, the wind-induced
    # angle-of-attack rate. A model that does not produce one is unchanged and
    # gets exactly 0.0, so `zero_wind` and every model written before this stay
    # bit-identical rather than merely equivalent. The tuple length is static, so
    # this branch resolves at trace time and costs nothing under jit.
    produced = wind_model(sim.wind, sim.state, sim.key, dt)
    if len(produced) == 5:
        wind_ned, omega_gust, wind_state, key, alphadot_gust = produced
    else:
        wind_ned, omega_gust, wind_state, key = produced
        alphadot_gust = 0.0

    applied = zero_increment() if load_model is None else load_model(sim.state)
    increment = None if load_model is None else applied

    # *** THE WIND HOLD, AND HOW TO TURN IT OFF -- session 23. ***
    #
    # Holding one wind sample across all four RK4 stages is an O(h) perturbation
    # of the right-hand side inside the step, so the scheme is FIRST ORDER
    # through a spatially varying field however good the stage weights are:
    # measured 1.05 held against 4.05 re-sampled, and it costs 0.82% of the
    # headline in-core d(theta) at the published dt (ASSUMPTIONS E4).
    #
    # Re-sampling is correct ONLY for a model that is a pure function of
    # position. For a stochastic field it would make the realisation depend on
    # the step size, so a convergence study would measure the noise instead of
    # the integrator. `wind.field_model` therefore MARKS the models that are
    # safe, and nothing else is re-sampled.
    #
    # DEFAULT IS STILL THE HOLD. Flipping it would move every published
    # deterministic-field result again, so it is opt-in per call and the switch
    # is measured rather than assumed -- see
    # test_integrate.test_stage_sampling_restores_fourth_order.
    stage_field = getattr(wind_model, "field", None) if stage_sampled else None

    if stage_field is None:
        def f(s: State) -> State:
            return derivatives(s, controls, ac, wind_ned, omega_gust,
                               increment=increment, alphadot_gust=alphadot_gust)
    else:
        from atisim.wind import gust_alphadot as _gust_alphadot
        from atisim.wind import gust_rates as _gust_rates

        def f(s: State) -> State:
            w = stage_field(s.pos_ned)
            og = _gust_rates(s.pos_ned, s.quat, stage_field)
            ad = _gust_alphadot(s.pos_ned, s.quat, s.vel_body, stage_field)
            return derivatives(s, controls, ac, w, og,
                               increment=increment, alphadot_gust=ad)

    new_state = rk4_step(f, sim.state, dt)
    new_state = new_state._replace(quat=quat_normalize(new_state.quat))

    return SimState(
        state=new_state,
        wind=wind_state,
        key=key,
        wind_ned=wind_ned,
        omega_gust=omega_gust,
        increment=applied,
    )


@partial(jax.jit, static_argnames=("n_steps", "wind_model", "load_model", "stage_sampled"))
def rollout(
    sim: SimState,
    controls: Controls,
    dt: Array,
    ac: Aircraft,
    n_steps: int,
    wind_model=zero_wind,
    load_model=None,
    stage_sampled: bool = False,
) -> tuple[SimState, State]:
    """Run n_steps with fixed controls.

    Returns the final SimState and the state trajectory, the latter stacked
    along a leading time axis of length n_steps.
    """

    def body(carry: SimState, _) -> tuple[SimState, State]:
        carry = step(carry, controls, dt, ac, wind_model=wind_model,
                     load_model=load_model, stage_sampled=stage_sampled)
        return carry, carry.state

    return jax.lax.scan(body, sim, None, length=n_steps)


@partial(jax.jit, static_argnames=("n_steps", "wind_model", "load_model", "stage_sampled"))
def logged_rollout(
    sim: SimState,
    controls: Controls,
    dt: Array,
    ac: Aircraft,
    n_steps: int,
    wind_model=zero_wind,
    load_model=None,
    stage_sampled: bool = False,
) -> tuple[SimState, SimState]:
    """`rollout`, but the whole `SimState` is stacked rather than just the state.

    `rollout` emits `carry.state`, which discards `wind_ned`, `omega_gust` and
    `increment` -- so the wind a run actually flew through is unrecoverable from
    its output, and no analysis script could write a self-describing run.
    Re-evaluating the model afterwards is NOT the same thing: it is exact for a
    deterministic field and returns a different realisation for a stochastic one,
    which is the trap `SimState`'s own docstring warns about.

    Separate from `rollout` rather than replacing it, because `rollout`'s narrower
    output is what `lax.scan` stacks in the hot path and every existing caller
    wants a `State` trajectory. `test_integrate.py` asserts the two produce
    bit-identical states.
    """

    def body(carry: SimState, _) -> tuple[SimState, SimState]:
        carry = step(carry, controls, dt, ac, wind_model=wind_model,
                     load_model=load_model, stage_sampled=stage_sampled)
        return carry, carry

    return jax.lax.scan(body, sim, None, length=n_steps)


def trajectory_from_log(t, log: SimState, controls_hist: Controls, mode: int = 0):
    """Turn a `logged_rollout` output into a `viz.Trajectory`.

    Lives here rather than in `viz.py` so that `viz` keeps its standing property
    of needing no simulator to read a run -- this function is on the writing side,
    where the simulator is already in scope.
    """
    import numpy as np

    from atisim.viz import Trajectory

    return Trajectory(
        t=np.asarray(t, dtype=float),
        pos_ned=np.asarray(log.state.pos_ned, dtype=float),
        vel_body=np.asarray(log.state.vel_body, dtype=float),
        quat=np.asarray(log.state.quat, dtype=float),
        omega=np.asarray(log.state.omega, dtype=float),
        controls=np.stack([np.asarray(c, dtype=float) for c in controls_hist], axis=1),
        mode=np.full(len(t), int(mode), dtype=int),
        wind_ned=np.asarray(log.wind_ned, dtype=float),
        omega_gust=np.asarray(log.omega_gust, dtype=float),
    )


def batch_sim(state: State, keys: Array) -> SimState:
    """Replicate one initial state across a batch of PRNG keys.

    The batch axis exists from day one even at size 1, so that Monte Carlo over
    turbulence realisations needs no restructuring.
    """
    n = keys.shape[0]
    tiled = jax.tree.map(lambda x: jnp.broadcast_to(x, (n,) + x.shape), state)
    winds = jax.tree.map(lambda x: jnp.broadcast_to(x, (n,) + x.shape), zero_wind_state())
    increments = jax.tree.map(
        lambda x: jnp.broadcast_to(x, (n,) + x.shape), zero_increment()
    )
    return SimState(
        state=tiled,
        wind=winds,
        key=keys,
        wind_ned=jnp.zeros((n, 3)),
        omega_gust=jnp.zeros((n, 3)),
        increment=increments,
    )


def batched_rollout(
    sim: SimState,
    controls: Controls,
    dt: Array,
    ac: Aircraft,
    n_steps: int,
    wind_model=zero_wind,
    load_model=None,
) -> tuple[SimState, State]:
    """rollout vmapped over the leading batch axis of `sim`."""
    return jax.vmap(
        lambda s: rollout(
            s, controls, dt, ac, n_steps, wind_model=wind_model, load_model=load_model
        )
    )(sim)
