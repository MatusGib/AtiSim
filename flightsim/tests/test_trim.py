import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import root

from flightsim import integrate, trim
from flightsim.aircraft import CRUISE, REGISTRY
from flightsim.units import RAD2DEG

AC = REGISTRY["boeing747"]
V = CRUISE["boeing747"]["airspeed"]
H = CRUISE["boeing747"]["altitude"]


def test_trim_converges():
    _, residual = trim.trim(jnp.array(V), jnp.array(H), AC)
    assert float(jnp.linalg.norm(residual)) < 1e-10


def test_trim_reproduces_the_source_flight_condition():
    """CR-2144 Table IX-3 tabulates alpha = 4.60 deg at flight condition 9.

    Recovering it means the atmosphere, unit conversions, mass, geometry and
    lift build-up are mutually consistent -- lift really does equal weight at
    ISA density for 40,000 ft and 774 ft/s.
    """
    x, _ = trim.trim(jnp.array(V), jnp.array(H), AC)
    assert float(x[0]) * RAD2DEG == pytest.approx(4.60, abs=0.1)


def test_trim_controls_are_physically_sensible():
    x, _ = trim.trim(jnp.array(V), jnp.array(H), AC)
    _, elevator, throttle = (float(v) for v in x)
    assert 0.0 < throttle < 1.0
    assert abs(elevator) < float(AC.elevator_limit)


def test_trim_matches_scipy():
    """Guard against a bug in the hand-rolled Newton loop."""
    x, _ = trim.trim(jnp.array(V), jnp.array(H), AC)

    def f(x_np):
        return np.asarray(trim.residual(jnp.array(x_np), jnp.array(V), jnp.array(H), AC))

    sol = root(f, np.array([0.05, 0.0, 0.5]), tol=1e-12)
    assert sol.success
    np.testing.assert_allclose(np.asarray(x), sol.x, atol=1e-8)


def test_trimmed_flight_holds_altitude_and_airspeed():
    """60 s with fixed controls from trim.

    Trim is an exact fixed point of the ODE, so the tolerance here is tight on
    purpose: any drift at all means the integrator or the state construction
    disagrees with the residual the solver drove to zero.
    """
    x, _ = trim.trim(jnp.array(V), jnp.array(H), AC)
    state = trim.trimmed_state(x[0], jnp.array(V), jnp.array(H))
    controls = trim.trimmed_controls(x[1], x[2])
    dt = 0.02
    _, hist = integrate.rollout(
        integrate.init_sim(state, jax.random.PRNGKey(0)),
        controls,
        jnp.array(dt),
        AC,
        int(60.0 / dt),
    )
    altitude = -np.asarray(hist.pos_ned)[:, 2]
    airspeed = np.linalg.norm(np.asarray(hist.vel_body), axis=1)
    assert np.abs(altitude - H).max() < 1e-6
    assert np.abs(airspeed - V).max() < 1e-6
    # It must still be flying, not merely frozen.
    assert np.asarray(hist.pos_ned)[-1, 0] == pytest.approx(V * 60.0, rel=1e-9)


def test_trim_converges_away_from_the_reference_condition():
    for airspeed in (200.0, 220.0, 250.0):
        _, residual = trim.trim(jnp.array(airspeed), jnp.array(H), AC)
        assert float(jnp.linalg.norm(residual)) < 1e-9, airspeed


def test_slower_flight_needs_more_alpha():
    slow, _ = trim.trim(jnp.array(200.0), jnp.array(H), AC)
    fast, _ = trim.trim(jnp.array(250.0), jnp.array(H), AC)
    assert float(slow[0]) > float(fast[0])
