from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from atisim import earth, integrate, loads
from atisim.state import Array, Controls, State, euler_to_quat, state_from_ned
from atisim.state import pos_ned as state_pos_ned

CRUISE_CONTROLS = Controls(
    elevator=jnp.array(-0.02),
    aileron=jnp.array(0.0),
    rudder=jnp.array(0.0),
    throttle=jnp.array(0.4),
)

# 47N is the latitude the rest of this project's Earth-rotation work uses. Each
# anchor sits AT its own flight altitude, so every state below is built at
# `pos_ned = 0` and is level flight where its name says it is. Two of them
# because this file flies at two altitudes.
ANCHOR = earth.anchor_at(np.radians(47.0), 0.0, 1000.0)
ANCHOR_2000 = earth.anchor_at(np.radians(47.0), 0.0, 2000.0)
EARTH = earth.WGS84_J2


def initial_state(u=50.0, anchor=ANCHOR) -> State:
    """At the anchor, so the flight altitude is `anchor.h` and not an argument.

    It used to be `pos_ned=[0, 0, -altitude]` against an implied sea-level
    origin. That reading no longer exists: position is an ECEF offset from a
    run anchor, and altitude is geodetic.
    """
    return state_from_ned(
        jnp.zeros(3),
        jnp.array([u, 0.0, 0.0]),
        euler_to_quat(jnp.array(0.0), jnp.array(0.05), jnp.array(0.0)),
        jnp.zeros(3),
        anchor,
    )


def test_rollout_shapes_and_finiteness(test_aircraft):
    sim = integrate.init_sim(initial_state(), jax.random.PRNGKey(0))
    final, history = integrate.rollout(
        sim, CRUISE_CONTROLS, jnp.array(0.02), test_aircraft, 500, ANCHOR, EARTH
    )
    assert history.pos_ecef.shape == (500, 3)
    assert history.quat.shape == (500, 4)
    for field in final.state:
        assert np.isfinite(np.asarray(field)).all()


def test_quaternion_norm_holds_over_a_long_rollout(test_aircraft):
    """1e5 steps: the design spec's own stated norm-stability horizon."""
    sim = integrate.init_sim(initial_state(), jax.random.PRNGKey(0))
    _, history = integrate.rollout(
        sim, CRUISE_CONTROLS, jnp.array(0.02), test_aircraft, 100_000, ANCHOR, EARTH
    )
    norms = np.linalg.norm(np.asarray(history.quat), axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-12)


def test_rk4_is_fourth_order(test_aircraft):
    """Halving dt must cut the error by roughly 16.

    Measured on WGS84_J2 rather than FLAT, and it survives the move to five
    digits: the ratio is 16.66547 on WGS84_J2 against 16.66557 on FLAT. The
    Coriolis, centrifugal and J2 terms are smooth functions of the state, so
    they change the trajectory without touching the order of the scheme -- which
    is what that agreement measures, and it is the reason this test did not need
    FLAT to keep testing what it was written to test.

    `pos_ecef` being an OFFSET from the anchor is what keeps this measurable at
    all -- an absolute ECEF coordinate would carry a 512x coarser ulp into the
    difference (ASSUMPTIONS.md F4) and put a floor under the fine run.
    """
    duration = 4.0

    def final_state(dt):
        n = int(round(duration / dt))
        sim = integrate.init_sim(initial_state(), jax.random.PRNGKey(0))
        final, _ = integrate.rollout(
            sim, CRUISE_CONTROLS, jnp.array(dt), test_aircraft, n, ANCHOR, EARTH
        )
        return np.concatenate(
            [np.asarray(f).ravel() for f in final.state]
        )

    reference = final_state(0.0025)
    coarse = np.linalg.norm(final_state(0.04) - reference)
    fine = np.linalg.norm(final_state(0.02) - reference)
    ratio = coarse / fine
    assert 10.0 < ratio < 20.0, f"observed order ratio {ratio}"


def test_logged_rollout_gives_the_same_states_as_rollout(test_aircraft):
    """Bit-identical, because it is the same `step` scanned with a wider output.

    `rollout` emits `carry.state`, so the wind and gust a run actually flew
    through are unrecoverable from it -- which is why no analysis script could
    write a run artifact. `logged_rollout` emits the whole `SimState`. If the two
    ever disagreed, an artifact would describe a different run from the figure
    drawn beside it.
    """
    import jax.numpy as jnp

    from atisim import wind

    def shear(wind_state, state, key, dt):
        # The wind field's frame is the anchor's local NED, so the state has to
        # be rotated into it. `state.pos_ecef[0]` is an ECEF component and would
        # have been a different axis entirely.
        del dt
        north = state_pos_ned(state, ANCHOR)[0]
        return jnp.array([0.0, 0.0, -0.001 * north]), jnp.zeros(3), wind_state, key

    sim = integrate.init_sim(initial_state(), jax.random.PRNGKey(0))
    _, plain = integrate.rollout(
        sim, CRUISE_CONTROLS, jnp.array(0.02), test_aircraft, 200, ANCHOR, EARTH,
        wind_model=shear,
    )
    _, logged = integrate.logged_rollout(
        sim, CRUISE_CONTROLS, jnp.array(0.02), test_aircraft, 200, ANCHOR, EARTH,
        wind_model=shear,
    )

    for field in State._fields:
        assert np.array_equal(
            np.asarray(getattr(logged.state, field)), np.asarray(getattr(plain, field))
        ), field
    # And it carries the thing rollout throws away.
    assert np.asarray(logged.wind_ned).shape == (200, 3)
    assert np.abs(np.asarray(logged.wind_ned)).max() > 0.0


def test_step_matches_a_manual_rk4_stage_sequence(test_aircraft):
    """Guard against a mis-weighted Butcher tableau."""
    from atisim.dynamics import derivatives

    dt = 0.02
    x = initial_state()
    f = lambda s: derivatives(  # noqa: E731
        s, CRUISE_CONTROLS, test_aircraft, jnp.zeros(3), jnp.zeros(3), ANCHOR, EARTH
    )
    add = lambda a, b, s: jax.tree.map(lambda p, q: p + s * q, a, b)

    k1 = f(x)
    k2 = f(add(x, k1, dt / 2))
    k3 = f(add(x, k2, dt / 2))
    k4 = f(add(x, k3, dt))
    expected = jax.tree.map(
        lambda a, b, c, d, e: a + dt * (b + 2 * c + 2 * d + e) / 6, x, k1, k2, k3, k4
    )

    sim = integrate.init_sim(x, jax.random.PRNGKey(0))
    got = integrate.step(
        sim, CRUISE_CONTROLS, jnp.array(dt), test_aircraft, ANCHOR, EARTH
    ).state

    np.testing.assert_allclose(np.asarray(got.pos_ecef), np.asarray(expected.pos_ecef), atol=1e-12)
    np.testing.assert_allclose(np.asarray(got.vel_body), np.asarray(expected.vel_body), atol=1e-12)
    np.testing.assert_allclose(np.asarray(got.omega), np.asarray(expected.omega), atol=1e-12)


# --- batching: the Monte Carlo path ---


def test_vmap_batch_of_one_matches_unbatched(test_aircraft):
    x0 = initial_state()
    dt, n = jnp.array(0.02), 300

    single, single_hist = integrate.rollout(
        integrate.init_sim(x0, jax.random.PRNGKey(7)), CRUISE_CONTROLS, dt,
        test_aircraft, n, ANCHOR, EARTH,
    )
    batch = integrate.batch_sim(x0, jax.random.PRNGKey(7)[None, :])
    batched, batched_hist = integrate.batched_rollout(
        batch, CRUISE_CONTROLS, dt, test_aircraft, n, ANCHOR, EARTH
    )

    assert batched_hist.pos_ecef.shape == (1, n, 3)
    np.testing.assert_allclose(
        np.asarray(batched_hist.pos_ecef[0]), np.asarray(single_hist.pos_ecef), atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(batched.state.vel_body[0]), np.asarray(single.state.vel_body), atol=1e-12
    )


def test_batch_of_many_keys_is_identical_under_zero_wind(test_aircraft):
    """Different keys, still air: every realisation must agree exactly."""
    keys = jax.random.split(jax.random.PRNGKey(0), 8)
    batch = integrate.batch_sim(initial_state(), keys)
    _, hist = integrate.batched_rollout(
        batch, CRUISE_CONTROLS, jnp.array(0.02), test_aircraft, 200, ANCHOR, EARTH
    )
    spread = np.asarray(hist.pos_ecef).std(axis=0).max()
    assert spread == pytest.approx(0.0, abs=1e-12)


# --- the hooks that exist for turbulence ---


class FilterState(NamedTuple):
    """Stands in for a Dryden shaping-filter state."""

    accumulated: Array


def counting_wind(wind_state, state, key, dt):
    """A fake turbulence model: carries filter state and consumes the key."""
    key, subkey = jax.random.split(key)
    draw = jax.random.normal(subkey)
    return (
        jnp.array([draw, 0.0, 0.0]),
        jnp.zeros(3),
        FilterState(accumulated=wind_state.accumulated + 1.0),
        key,
    )


def test_wind_state_and_key_are_threaded_through_the_scan(test_aircraft):
    """The two hooks the whole design exists for, exercised end to end."""
    key0 = jax.random.PRNGKey(3)
    sim = integrate.SimState(
        state=initial_state(),
        wind=FilterState(accumulated=jnp.array(0.0)),
        key=key0,
        wind_ned=jnp.zeros(3),
        omega_gust=jnp.zeros(3),
        increment=loads.zero_increment(),
    )
    final, _ = integrate.rollout(
        sim, CRUISE_CONTROLS, jnp.array(0.02), test_aircraft, 250, ANCHOR, EARTH,
        wind_model=counting_wind,
    )
    # Filter state advanced once per step, not once per RK4 stage.
    assert float(final.wind.accumulated) == 250.0
    # The key was actually consumed rather than passed along untouched.
    assert not np.array_equal(np.asarray(final.key), np.asarray(key0))


def test_different_keys_diverge_once_wind_is_stochastic(test_aircraft):
    """With a real wind model, a batch of keys must give distinct trajectories.

    This is the Monte Carlo ensemble the PRNG plumbing exists to enable.
    """
    keys = jax.random.split(jax.random.PRNGKey(0), 8)
    n = keys.shape[0]
    tiled = jax.tree.map(
        lambda x: jnp.broadcast_to(x, (n,) + x.shape), initial_state()
    )
    sim = integrate.SimState(
        state=tiled,
        wind=FilterState(accumulated=jnp.zeros(n)),
        key=keys,
        wind_ned=jnp.zeros((n, 3)),
        omega_gust=jnp.zeros((n, 3)),
        increment=jax.tree.map(
            lambda x: jnp.broadcast_to(x, (n,) + x.shape), loads.zero_increment()
        ),
    )
    _, hist = jax.vmap(
        lambda s: integrate.rollout(
            s, CRUISE_CONTROLS, jnp.array(0.02), test_aircraft, 200, ANCHOR, EARTH,
            wind_model=counting_wind,
        )
    )(sim)
    spread = np.asarray(hist.pos_ecef).std(axis=0).max()
    assert spread > 1e-6


def test_wind_is_sampled_once_per_step_not_per_rk4_stage(test_aircraft):
    """Four stages per step must not mean four filter updates."""
    sim = integrate.SimState(
        state=initial_state(),
        wind=FilterState(accumulated=jnp.array(0.0)),
        key=jax.random.PRNGKey(1),
        wind_ned=jnp.zeros(3),
        omega_gust=jnp.zeros(3),
        increment=loads.zero_increment(),
    )
    after_one = integrate.step(
        sim, CRUISE_CONTROLS, jnp.array(0.02), test_aircraft, ANCHOR, EARTH,
        wind_model=counting_wind,
    )
    assert float(after_one.wind.accumulated) == 1.0


# --- the load seam: an optional strip model, off by default ----------------


def test_omitting_the_load_model_is_bit_identical_to_today(test_aircraft):
    """The gate that protects every frozen baseline. A run that does not ask for
    strip loads must produce exactly the trajectory it produced before this
    feature existed -- not nearly, exactly."""
    state = state_from_ned(
        jnp.zeros(3),
        jnp.array([60.0, 0.0, 2.0]),
        euler_to_quat(jnp.array(0.0), jnp.array(0.03), jnp.array(0.0)),
        jnp.zeros(3),
        ANCHOR_2000,
    )
    controls = Controls(
        elevator=jnp.array(0.02), aileron=jnp.array(0.0),
        rudder=jnp.array(0.0), throttle=jnp.array(0.5),
    )
    sim = integrate.init_sim(state, jax.random.PRNGKey(0))

    plain, _ = integrate.rollout(
        sim, controls, jnp.array(0.02), test_aircraft, 100, ANCHOR_2000, EARTH
    )
    explicit, _ = integrate.rollout(
        sim, controls, jnp.array(0.02), test_aircraft, 100, ANCHOR_2000, EARTH,
        load_model=None,
    )
    assert np.array_equal(
        np.asarray(plain.state.pos_ecef), np.asarray(explicit.state.pos_ecef)
    )
    assert np.array_equal(np.asarray(plain.state.quat), np.asarray(explicit.state.quat))


def test_the_applied_increment_is_cached_on_the_sim_state(test_aircraft):
    """Same reason wind_ned and omega_gust are cached: a recorder or controller
    must be able to see what was actually applied without calling the model a
    second time. Consumers see one step of lag, which is what a real sensor
    gives anyway."""
    state = state_from_ned(
        jnp.zeros(3),
        jnp.array([60.0, 0.0, 0.0]),
        euler_to_quat(jnp.array(0.0), jnp.array(0.0), jnp.array(0.0)),
        jnp.zeros(3),
        ANCHOR_2000,
    )
    controls = Controls(
        elevator=jnp.array(0.0), aileron=jnp.array(0.0),
        rudder=jnp.array(0.0), throttle=jnp.array(0.5),
    )
    sim = integrate.init_sim(state, jax.random.PRNGKey(0))
    assert isinstance(sim.increment, loads.CoeffIncrement)
    assert float(sim.increment.Cl) == 0.0

    stepped = integrate.step(
        sim, controls, jnp.array(0.02), test_aircraft, ANCHOR_2000, EARTH,
        load_model=lambda s: loads.zero_increment()._replace(Cl=jnp.array(0.005)),
    )
    assert float(stepped.increment.Cl) == pytest.approx(0.005)
