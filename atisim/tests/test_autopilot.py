import jax
import jax.numpy as jnp
import numpy as np
import pytest

from atisim import autopilot as ap_mod
from atisim import integrate, trim
from atisim.autopilot import _accumulate as _acc
from atisim.aero import air_data
from atisim.aircraft import CRUISE, REGISTRY
from atisim.sensors import sense
from atisim.state import quat_to_euler
from atisim.units import RAD2DEG

AC = REGISTRY["boeing747"]
GAINS = ap_mod.GAINS["boeing747"]
V = CRUISE["boeing747"]["airspeed"]
H = CRUISE["boeing747"]["altitude"]
DT = 0.02


def trimmed():
    x, _ = trim.trim(jnp.array(V), jnp.array(H), AC)
    return (
        trim.trimmed_state(x[0], jnp.array(V), jnp.array(H)),
        trim.trimmed_controls(x[1], x[2]),
    )


def fly(targets, seconds, gains=GAINS):
    state, controls = trimmed()
    ap = ap_mod.engage(sense(state), controls, targets, gains, AC)
    sim = integrate.init_sim(state, jax.random.PRNGKey(0))
    (_, _), (hist, ctrl) = ap_mod.closed_loop_rollout(
        sim, ap, targets, gains, jnp.array(DT), AC, int(seconds / DT)
    )
    return hist, ctrl


def hold_targets():
    return ap_mod.Targets(
        altitude=jnp.array(H), heading=jnp.array(0.0), airspeed=jnp.array(V)
    )


# --- bumpless transfer: the ten lines worth writing ---


def test_engagement_reproduces_the_current_controls_exactly():
    state, controls = trimmed()
    targets = hold_targets()
    ap = ap_mod.engage(sense(state), controls, targets, GAINS, AC)
    out, _ = ap_mod.autopilot(sense(state), ap, targets, GAINS, AC, jnp.array(DT))
    for field, current in zip(out, controls):
        assert float(field) == pytest.approx(float(current), abs=1e-12)


def test_engagement_from_a_non_trim_deflection_still_matches():
    """Bumpless transfer must work from any hand-flown position, not just trim."""
    state, _ = trimmed()
    controls = trim.trimmed_controls(jnp.array(0.06), jnp.array(0.9))
    targets = hold_targets()
    ap = ap_mod.engage(sense(state), controls, targets, GAINS, AC)
    out, _ = ap_mod.autopilot(sense(state), ap, targets, GAINS, AC, jnp.array(DT))
    assert float(out.elevator) == pytest.approx(0.06, abs=1e-12)
    assert float(out.throttle) == pytest.approx(0.9, abs=1e-12)


def test_engaged_at_trim_the_aircraft_does_not_move():
    hist, ctrl = fly(hold_targets(), 30.0)
    altitude = -np.asarray(hist.pos_ned)[:, 2]
    assert np.abs(altitude - H).max() < 1.0
    assert np.abs(np.asarray(ctrl.elevator) - float(ctrl.elevator[0])).max() < 1e-3


def test_disengagement_hands_back_the_current_deflections():
    """APState.controls is what the pilot inherits; it must be the live output."""
    state, controls = trimmed()
    targets = hold_targets()
    ap = ap_mod.engage(sense(state), controls, targets, GAINS, AC)
    out, ap2 = ap_mod.autopilot(sense(state), ap, targets, GAINS, AC, jnp.array(DT))
    for handback, live in zip(ap2.controls, out):
        assert float(handback) == float(live)


# --- capture ---


def test_altitude_capture():
    targets = ap_mod.Targets(
        altitude=jnp.array(H + 300.0), heading=jnp.array(0.0), airspeed=jnp.array(V)
    )
    hist, _ = fly(targets, 300.0)
    altitude = -np.asarray(hist.pos_ned)[:, 2]
    assert abs(altitude[-1] - (H + 300.0)) < 10.0
    assert altitude.max() < H + 320.0  # no large overshoot


def test_heading_capture_stays_coordinated():
    targets = ap_mod.Targets(
        altitude=jnp.array(H),
        heading=jnp.array(np.deg2rad(30.0)),
        airspeed=jnp.array(V),
    )
    hist, _ = fly(targets, 300.0)
    euler = np.asarray(jax.vmap(quat_to_euler)(hist.quat))
    beta = np.asarray(jax.vmap(lambda v: air_data(v)[2])(hist.vel_body))
    assert abs(euler[-1, 2] * RAD2DEG - 30.0) < 1.0
    assert np.abs(beta).max() * RAD2DEG < 1.0  # rudder is coordinating the turn
    assert np.abs(euler[:, 0]).max() <= float(GAINS.phi_limit) + 0.02


def test_airspeed_capture_holds_altitude():
    targets = ap_mod.Targets(
        altitude=jnp.array(H), heading=jnp.array(0.0), airspeed=jnp.array(V - 15.0)
    )
    hist, _ = fly(targets, 400.0)
    speed = np.linalg.norm(np.asarray(hist.vel_body), axis=1)
    altitude = -np.asarray(hist.pos_ned)[:, 2]
    assert abs(speed[-1] - (V - 15.0)) < 1.0
    assert np.abs(altitude - H).max() < 40.0


def test_heading_wrap_takes_the_short_way_round():
    """A target 10 deg west of north from a heading of 0 must turn left."""
    targets = ap_mod.Targets(
        altitude=jnp.array(H),
        heading=jnp.array(np.deg2rad(350.0)),
        airspeed=jnp.array(V),
    )
    hist, _ = fly(targets, 60.0)
    roll = np.asarray(jax.vmap(quat_to_euler)(hist.quat))[:, 0]
    assert roll.min() < -0.01  # banked left
    assert roll.max() < 0.01


def test_wrap_pi():
    # +pi and -pi are the same heading, so the boundary case is asserted on
    # magnitude rather than sign.
    assert abs(float(ap_mod.wrap_pi(jnp.array(3.0 * np.pi)))) == pytest.approx(np.pi)
    assert float(ap_mod.wrap_pi(jnp.array(0.3))) == pytest.approx(0.3)
    assert float(ap_mod.wrap_pi(jnp.array(2.0 * np.pi + 0.3))) == pytest.approx(0.3)
    assert float(ap_mod.wrap_pi(jnp.array(np.deg2rad(350.0)))) == pytest.approx(
        np.deg2rad(-10.0)
    )
    assert all(
        -np.pi <= float(ap_mod.wrap_pi(jnp.array(a))) <= np.pi
        for a in np.linspace(-20.0, 20.0, 101)
    )


# --- limits ---


def test_surfaces_respect_deflection_limits():
    targets = ap_mod.Targets(
        altitude=jnp.array(H + 8000.0),  # unreachable, drives full saturation
        heading=jnp.array(np.deg2rad(170.0)),
        airspeed=jnp.array(V),
    )
    _, ctrl = fly(targets, 200.0)
    assert np.abs(np.asarray(ctrl.elevator)).max() <= float(AC.elevator_limit) + 1e-9
    assert np.abs(np.asarray(ctrl.aileron)).max() <= float(AC.aileron_limit) + 1e-9
    assert np.abs(np.asarray(ctrl.rudder)).max() <= float(AC.rudder_limit) + 1e-9
    throttle = np.asarray(ctrl.throttle)
    assert throttle.min() >= -1e-9 and throttle.max() <= 1.0 + 1e-9


def test_surface_rate_limits_are_respected():
    targets = ap_mod.Targets(
        altitude=jnp.array(H + 8000.0),
        heading=jnp.array(np.deg2rad(170.0)),
        airspeed=jnp.array(V),
    )
    _, ctrl = fly(targets, 100.0)
    for name in ("elevator", "aileron", "rudder"):
        rate = np.diff(np.asarray(getattr(ctrl, name))) / DT
        assert np.abs(rate).max() <= float(GAINS.surface_rate) + 1e-9, name
    throttle_rate = np.diff(np.asarray(ctrl.throttle)) / DT
    assert np.abs(throttle_rate).max() <= float(GAINS.throttle_rate) + 1e-9


def test_accumulate_freezes_when_saturated_but_still_unwinds():
    dt, limit, bound = jnp.array(0.1), jnp.array(1.0), jnp.array(100.0)
    acc = jnp.array(5.0)
    # Saturated high and the error pushes further out: frozen.
    frozen = _acc(acc, jnp.array(2.0), dt, jnp.array(1.5), limit, bound)
    assert float(frozen) == 5.0
    # Saturated high but the error has reversed: must unwind, not stay stuck.
    unwinding = _acc(acc, jnp.array(-2.0), dt, jnp.array(1.5), limit, bound)
    assert float(unwinding) < 5.0
    # Unsaturated: ordinary integration.
    normal = _acc(acc, jnp.array(2.0), dt, jnp.array(0.5), limit, bound)
    assert float(normal) == pytest.approx(5.2)


def test_accumulate_bounds_the_state_not_just_the_command():
    """The state bound is what caps recovery time after a long saturation.

    Conditional integration alone leaves the accumulator free to sit at an
    enormous value, which then commands full deflection when the target moves.
    """
    dt, limit, bound = jnp.array(1.0), jnp.array(1.0), jnp.array(10.0)
    acc = jnp.array(0.0)
    for _ in range(500):  # long run with the command well inside its limit
        acc = _acc(acc, jnp.array(50.0), dt, jnp.array(0.0), limit, bound)
    assert float(acc) == pytest.approx(10.0)


def test_anti_windup_recovers_promptly_from_an_unreachable_target():
    """Chase an impossible altitude, then ask for the original one back.

    Without anti-windup the altitude integrator accumulates for the whole first
    leg and the aircraft sails past the recovered target.
    """
    state, controls = trimmed()
    unreachable = ap_mod.Targets(
        altitude=jnp.array(H + 8000.0), heading=jnp.array(0.0), airspeed=jnp.array(V)
    )
    ap = ap_mod.engage(sense(state), controls, unreachable, GAINS, AC)
    sim = integrate.init_sim(state, jax.random.PRNGKey(0))
    (sim, ap), _ = ap_mod.closed_loop_rollout(
        sim, ap, unreachable, GAINS, jnp.array(DT), AC, int(400.0 / DT)
    )
    reached = -float(sim.state.pos_ned[2])

    back = ap_mod.Targets(
        altitude=jnp.array(reached), heading=jnp.array(0.0), airspeed=jnp.array(V)
    )
    (_, _), (hist, _) = ap_mod.closed_loop_rollout(
        sim, ap, back, GAINS, jnp.array(DT), AC, int(300.0 / DT)
    )
    altitude = -np.asarray(hist.pos_ned)[:, 2]
    assert abs(altitude[-1] - reached) < 60.0
    assert np.abs(altitude - reached).max() < 400.0


# --- the batched path ---


def test_closed_loop_rollout_vmaps_over_keys():
    state, controls = trimmed()
    targets = hold_targets()
    ap = ap_mod.engage(sense(state), controls, targets, GAINS, AC)
    keys = jax.random.split(jax.random.PRNGKey(0), 4)
    sims = integrate.batch_sim(state, keys)
    aps = jax.tree.map(lambda x: jnp.broadcast_to(x, (4,) + jnp.shape(x)), ap)

    (_, _), (hist, _) = jax.vmap(
        lambda s, a: ap_mod.closed_loop_rollout(
            s, a, targets, GAINS, jnp.array(DT), AC, 500
        )
    )(sims, aps)
    assert hist.pos_ned.shape == (4, 500, 3)
    assert np.asarray(hist.pos_ned).std(axis=0).max() == pytest.approx(0.0, abs=1e-12)
