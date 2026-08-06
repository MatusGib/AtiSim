"""Tests for the air-relative sensing fix -- latent bugs (a), (b), (c), (d).

Every test here flies through a NON-ZERO wind field. That is the whole point:
the four defects these cover survived three sessions and a 209-test suite
because still air cannot distinguish airspeed from groundspeed, and every test
in the project was still air.

A steady horizontal wind is used rather than a vortex because it makes the
expected answer exact rather than approximate: in a W m/s headwind, airspeed is
groundspeed plus W, to machine precision, forever.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flightsim import autopilot as ap_mod
from flightsim import integrate, trim, viz
from flightsim.aero import air_data
from flightsim.aircraft import CRUISE, REGISTRY
from flightsim.sensors import sense
from flightsim.state import State, quat_to_euler

AC = REGISTRY["boeing747"]
V = CRUISE["boeing747"]["airspeed"]
H = CRUISE["boeing747"]["altitude"]
DT = 0.02
HEADWIND = 25.0  # m/s


def steady(wind_ned):
    """A constant wind field. Hashable by identity, so it works as a static arg."""
    vector = jnp.asarray(wind_ned, dtype=float)

    def model(wind_state, state, key, dt):
        del state, dt
        return vector, jnp.zeros(3), wind_state, key

    return model


# Flying north, so a NED wind of -W blows the air southward: a headwind.
HEADWIND_MODEL = steady([-HEADWIND, 0.0, 0.0])
UPDRAFT_MODEL = steady([0.0, 0.0, -10.0])  # NED down is negative up


@pytest.fixture(scope="module")
def trimmed():
    x, _ = trim.trim(jnp.array(V), jnp.array(H), AC)
    return (
        trim.trimmed_state(x[0], jnp.array(V), jnp.array(H)),
        trim.trimmed_controls(x[1], x[2]),
        float(x[0]),
    )


# --- sense() itself ---------------------------------------------------------


def test_still_air_sensing_is_unchanged(trimmed):
    """The fix must not move any still-air number, or every baseline is invalid."""
    state, _, _ = trimmed
    air = sense(state)
    speed, alpha, beta = air_data(state.vel_body)
    assert float(air.airspeed) == pytest.approx(float(speed), abs=1e-12)
    assert float(air.alpha) == pytest.approx(float(alpha), abs=1e-12)
    assert float(air.beta) == pytest.approx(float(beta), abs=1e-12)


def test_a_headwind_raises_airspeed_above_groundspeed(trimmed):
    state, _, _ = trimmed
    groundspeed = float(jnp.linalg.norm(state.vel_body))
    air = sense(state, jnp.array([-HEADWIND, 0.0, 0.0]))
    assert float(air.airspeed) == pytest.approx(groundspeed + HEADWIND, rel=1e-6)


def test_a_tailwind_lowers_it(trimmed):
    state, _, _ = trimmed
    groundspeed = float(jnp.linalg.norm(state.vel_body))
    air = sense(state, jnp.array([+HEADWIND, 0.0, 0.0]))
    assert float(air.airspeed) == pytest.approx(groundspeed - HEADWIND, rel=1e-6)


def test_an_updraft_changes_alpha_and_nothing_inertial(trimmed):
    """The defining signature of the bug: alpha moves, attitude does not.

    An updraft raises incidence -- the air arrives from below. At 236 m/s a
    10 m/s updraft is atan(10/236) = 2.4 deg, which is what this measures.
    """
    state, _, _ = trimmed
    still = sense(state)
    gusted = sense(state, jnp.array([0.0, 0.0, -10.0]))
    assert float(gusted.alpha) - float(still.alpha) == pytest.approx(
        np.arctan2(10.0, float(state.vel_body[0])), rel=0.02
    )
    # Attitude and rates come from an IMU and cannot know about the air.
    for field in ("phi", "theta", "psi", "p", "q", "r", "altitude"):
        assert float(getattr(gusted, field)) == float(getattr(still, field))


def test_body_rates_are_inertial_not_gust_relative(trimmed):
    """The classic overcorrection. A rate gyro measures the airframe's own
    rotation; omega_gust is a gradient across the span that changes the flow the
    wings see, not the rate the gyro reads. Feeding omega - omega_gust to a
    damping loop makes the autopilot chase a gust it cannot sense.
    """
    state, _, _ = trimmed
    spun = state._replace(omega=jnp.array([0.1, 0.2, 0.3]))
    air = sense(spun, jnp.array([5.0, -3.0, 2.0]))
    assert (float(air.p), float(air.q), float(air.r)) == (0.1, 0.2, 0.3)


# --- bug (a): the autopilot regulated groundspeed ---------------------------


def test_the_autopilot_holds_airspeed_not_groundspeed(trimmed):
    """Fly the speed loop into a 25 m/s headwind.

    Correct behaviour: airspeed settles on the target, so GROUNDSPEED settles
    25 m/s below it. The old code did the opposite -- it held groundspeed at the
    target and let true airspeed sit 25 m/s high, which at cruise Mach is the
    difference between cruising and approaching buffet.
    """
    state, controls, _ = trimmed
    targets = ap_mod.Targets(
        altitude=jnp.array(H), heading=jnp.array(0.0), airspeed=jnp.array(V)
    )
    ap = ap_mod.engage(sense(state), controls, targets, ap_mod.GAINS["boeing747"], AC)
    (final, _), (hist, _) = ap_mod.closed_loop_rollout(
        integrate.init_sim(state, jax.random.PRNGKey(0)),
        ap, targets, ap_mod.GAINS["boeing747"], jnp.array(DT), AC,
        int(600.0 / DT), wind_model=HEADWIND_MODEL,
    )
    groundspeed = float(jnp.linalg.norm(final.state.vel_body))
    airspeed = float(sense(final.state, final.wind_ned).airspeed)
    assert airspeed == pytest.approx(V, abs=1.5)
    assert groundspeed == pytest.approx(V - HEADWIND, abs=1.5)


# --- bug (c): the run records the wind that was applied ----------------------


def test_simstate_carries_the_wind_the_step_applied(trimmed):
    state, controls, _ = trimmed
    sim = integrate.init_sim(state, jax.random.PRNGKey(0))
    assert np.allclose(np.asarray(sim.wind_ned), 0.0)  # nothing applied yet
    sim = integrate.step(sim, controls, jnp.array(DT), AC, wind_model=HEADWIND_MODEL)
    assert np.asarray(sim.wind_ned) == pytest.approx([-HEADWIND, 0.0, 0.0])


def test_trajectory_records_wind_and_round_trips(trimmed, tmp_path):
    state, controls, _ = trimmed
    sim = integrate.init_sim(state, jax.random.PRNGKey(0))
    recorder = viz.Recorder()
    recorder.append(0.0, sim, controls, 0)
    for i in range(5):
        sim = integrate.step(sim, controls, jnp.array(DT), AC, wind_model=HEADWIND_MODEL)
        recorder.append((i + 1) * DT, sim, controls, 0)
    traj = recorder.trajectory()
    assert traj.wind_ned.shape == (6, 3)
    assert traj.wind_ned[-1] == pytest.approx([-HEADWIND, 0.0, 0.0])

    path = tmp_path / "wind.npz"
    viz.save(traj, path)
    again = viz.load(path)
    assert np.array_equal(again.wind_ned, traj.wind_ned)
    assert np.array_equal(again.omega_gust, traj.omega_gust)


def test_a_file_written_before_wind_existed_still_loads(trimmed, tmp_path):
    """Old .npz have no wind columns. They were all flown in still air, because
    nothing else was possible, so loading them as still air is honest.
    """
    state, controls, _ = trimmed
    sim = integrate.init_sim(state, jax.random.PRNGKey(0))
    recorder = viz.Recorder()
    for i in range(3):
        recorder.append(i * DT, sim, controls, 0)
    traj = recorder.trajectory()

    legacy = {k: v for k, v in traj._asdict().items()
              if k not in ("wind_ned", "omega_gust")}
    path = tmp_path / "legacy.npz"
    np.savez_compressed(path, **legacy)

    loaded = viz.load(path)
    assert loaded.wind_ned.shape == (3, 3)
    assert np.allclose(loaded.wind_ned, 0.0)
    assert np.array_equal(loaded.pos_ned, traj.pos_ned)


# --- bug (b): the test that could not go red --------------------------------


def test_derived_is_air_relative_and_this_test_can_fail(trimmed):
    """The replacement for test_derived_agrees_with_the_aero_module.

    That test fed viz and aero the SAME input and compared them, so both sides
    moved together and it stayed green while reporting a ground-relative angle.
    This one computes the expected answer from the recorded wind independently,
    so reverting derived() to inertial velocity turns it red.
    """
    state, controls, _ = trimmed
    sim = integrate.init_sim(state, jax.random.PRNGKey(0))
    recorder = viz.Recorder()
    for i in range(40):
        sim = integrate.step(sim, controls, jnp.array(DT), AC, wind_model=UPDRAFT_MODEL)
        recorder.append(i * DT, sim, controls, 0)
    traj = recorder.trajectory()
    d = viz.derived(traj)

    # Independent expectation: air-relative alpha from the logged wind.
    inertial = np.asarray(jax.vmap(lambda v: air_data(v)[1])(jnp.asarray(traj.vel_body)))
    assert np.abs(np.degrees(d.alpha - inertial)).max() > 1.0, (
        "derived() is reporting the inertial angle -- bug (b) has returned"
    )
    for i in (0, len(traj.t) // 2, -1):
        expected = sense(
            State(
                pos_ned=jnp.asarray(traj.pos_ned[i]),
                vel_body=jnp.asarray(traj.vel_body[i]),
                quat=jnp.asarray(traj.quat[i]),
                omega=jnp.asarray(traj.omega[i]),
            ),
            jnp.asarray(traj.wind_ned[i]),
        )
        assert d.alpha[i] == pytest.approx(float(expected.alpha), abs=1e-12)
        assert d.airspeed[i] == pytest.approx(float(expected.airspeed), abs=1e-9)


def test_attitude_is_untouched_by_the_wind_correction(trimmed):
    """Only the air-relative group moves. If theta changed, the fix went too far."""
    state, controls, _ = trimmed
    sim = integrate.init_sim(state, jax.random.PRNGKey(0))
    recorder = viz.Recorder()
    for i in range(20):
        sim = integrate.step(sim, controls, jnp.array(DT), AC, wind_model=UPDRAFT_MODEL)
        recorder.append(i * DT, sim, controls, 0)
    traj = recorder.trajectory()
    d = viz.derived(traj)
    euler = np.asarray(jax.vmap(quat_to_euler)(jnp.asarray(traj.quat)))
    assert np.abs(d.theta - euler[:, 1]).max() < 1e-12
    assert np.abs(d.phi - euler[:, 0]).max() < 1e-12


# --- bug (d): the transcription ---------------------------------------------


def test_the_747_pitch_damping_matches_cr2144_table_ix_4():
    """Mq is -0.339. It was transcribed as -0.330 and cost 1.1 points of
    short-period damping match. Cmq is the only place it reaches the model.
    """
    # Cmq is linear in Mq, so the shipped value pins the transcription.
    # -0.330 gives Cmq -23.288; -0.339 gives -23.923.
    assert float(REGISTRY["boeing747"].Cmq) == pytest.approx(-23.923, rel=1e-3)


# --- vertical speed and the accelerometer package ---------------------------


def test_vertical_speed_is_inertial_and_matches_the_ned_velocity(trimmed):
    """Computed independently from the quaternion, not by calling the same helper.

    A VSI fed anything else -- body w, or the down component of the AIR-relative
    velocity -- passes any still-air level test. This one is rolled, pitched and
    yawed in a wind, so neither substitution survives.
    """
    from flightsim.state import euler_to_quat, quat_to_dcm

    state, _, _ = trimmed
    tilted = state._replace(
        quat=euler_to_quat(jnp.array(0.3), jnp.array(0.15), jnp.array(0.7))
    )
    air = sense(tilted, jnp.array([5.0, -3.0, 2.0]))

    vel_ned = quat_to_dcm(tilted.quat) @ tilted.vel_body
    assert float(air.vertical_speed) == pytest.approx(-float(vel_ned[2]), abs=1e-12)
    # Body w would be a different number entirely; make sure we did not get it.
    assert abs(float(air.vertical_speed) + float(tilted.vel_body[2])) > 1.0


def test_vertical_speed_ignores_the_wind_argument_itself(trimmed):
    """It is INERTIAL.

    A barometric VSI reads the rate of change of geometric height, so an updraft
    that CARRIES the aircraft up is read -- through the trajectory, on the next
    step. But the instantaneous wind vector must not enter the calculation, or
    the VSI would show a climb the aircraft is not making.
    """
    state, _, _ = trimmed
    still = sense(state, jnp.zeros(3))
    blown = sense(state, jnp.array([0.0, 0.0, -12.0]))
    assert float(blown.vertical_speed) == pytest.approx(float(still.vertical_speed), abs=1e-12)


def test_accelerometers_report_specific_force_with_the_load_factor_sign(trimmed):
    from flightsim import dynamics
    from flightsim.sensors import accelerometers

    state, controls, _ = trimmed
    n = accelerometers(state, controls, AC, jnp.zeros(3), jnp.zeros(3))
    raw = dynamics.specific_force(state, controls, AC, jnp.zeros(3), jnp.zeros(3))

    assert float(n.n_x) == pytest.approx(float(raw[0]), abs=1e-15)
    assert float(n.n_y) == pytest.approx(float(raw[1]), abs=1e-15)
    assert float(n.n_z) == pytest.approx(-float(raw[2]), abs=1e-15)
    # Trimmed level flight: n_z is cos(theta) (see test_dynamics), lateral quiet.
    assert float(n.n_z) == pytest.approx(0.9967, abs=1e-3)
    assert abs(float(n.n_y)) < 1e-6


def test_accelerometers_see_a_gust_because_it_changes_the_aerodynamic_force(trimmed):
    """The accelerometer package is not air-relative or inertial -- it reads a
    FORCE, and a gust changes the force by changing the flow over the wings."""
    from flightsim.sensors import accelerometers

    state, controls, _ = trimmed
    still = accelerometers(state, controls, AC, jnp.zeros(3), jnp.zeros(3))
    gusted = accelerometers(state, controls, AC, jnp.array([0.0, 0.0, -10.0]), jnp.zeros(3))
    assert float(gusted.n_z) > float(still.n_z) + 0.05
