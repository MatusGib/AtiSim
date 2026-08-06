"""Headless tests for the visuals.

Agg supports `copy_from_bbox`/`restore_region`, so `FuncAnimation(blit=True)`
runs for real here -- these are not just "does it import" tests. Keyboard input
is driven by synthesising the same events the GUI backends deliver.
"""

import matplotlib

matplotlib.use("Agg")  # before flightsim.viz imports pyplot

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from matplotlib.animation import FuncAnimation
from matplotlib.backend_bases import KeyEvent

import matplotlib.pyplot as plt

from flightsim import autopilot as ap_mod
from flightsim import integrate, manual as man, trim, viz
from flightsim.aero import air_data
from flightsim.aircraft import CRUISE, REGISTRY
from flightsim.manual import Mode
from flightsim.sensors import sense
from flightsim.state import Controls, quat_to_euler
from flightsim.viz import PITCH_SPAN_DEG, _horizon_frame, _ladder

AC = REGISTRY["boeing747"]
GAINS = ap_mod.GAINS["boeing747"]
MGAINS = man.MANUAL_GAINS["boeing747"]
V = CRUISE["boeing747"]["airspeed"]
H = CRUISE["boeing747"]["altitude"]
DT = 0.02


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def trimmed():
    x, _ = trim.trim(jnp.array(V), jnp.array(H), AC)
    return (
        trim.trimmed_state(x[0], jnp.array(V), jnp.array(H)),
        trim.trimmed_controls(x[1], x[2]),
    )


@pytest.fixture
def targets():
    return ap_mod.Targets(
        altitude=jnp.array(H), heading=jnp.array(0.0), airspeed=jnp.array(V)
    )


@pytest.fixture
def live(trimmed, targets):
    """Panel plus LiveSim, stepping a fixed 1/fps of sim time per frame."""
    state, controls = trimmed
    ctl = man.start(sense(state), controls, targets, GAINS, AC)
    sim = integrate.init_sim(state, jax.random.PRNGKey(0))
    panel = viz.Panel(targets, window=20.0, fps=20.0)
    return viz.LiveSim(
        sim, ctl, targets, GAINS, MGAINS, AC, panel, dt=DT, real_time=False
    )


def press(panel, key):
    KeyEvent("key_press_event", panel.fig.canvas, key)._process()


def release(panel, key):
    KeyEvent("key_release_event", panel.fig.canvas, key)._process()


# --- the artificial horizon geometry ---------------------------------------


def test_horizon_is_level_and_centred_when_the_aircraft_is():
    centre, along, sky = _horizon_frame(0.0, 0.0)
    assert centre == pytest.approx([0.0, 0.0])
    assert along == pytest.approx([1.0, 0.0])
    assert sky == pytest.approx([0.0, 1.0])


def test_right_bank_puts_the_ground_on_the_right():
    """Right wing down: the sky rotates to the left of the pilot's view."""
    _, _, sky = _horizon_frame(np.deg2rad(90.0), 0.0)
    assert sky == pytest.approx([-1.0, 0.0], abs=1e-12)
    _, _, sky = _horizon_frame(np.deg2rad(30.0), 0.0)
    assert sky[0] < 0.0 and sky[1] > 0.0


def test_left_bank_puts_the_ground_on_the_left():
    _, _, sky = _horizon_frame(np.deg2rad(-30.0), 0.0)
    assert sky[0] > 0.0 and sky[1] > 0.0


def test_nose_up_moves_the_horizon_down_the_ball():
    centre, _, _ = _horizon_frame(0.0, +PITCH_SPAN_DEG)
    assert centre == pytest.approx([0.0, -1.0])
    centre, _, _ = _horizon_frame(0.0, -PITCH_SPAN_DEG)
    assert centre == pytest.approx([0.0, +1.0])


def test_the_ladder_rung_for_the_current_pitch_sits_on_the_centre():
    centre, along, sky = _horizon_frame(0.0, 10.0)
    xs, ys = _ladder(centre, along, sky, 10.0)
    rung = [(x, y) for x, y in zip(xs, ys) if np.isfinite(x)]
    assert min(abs(y) for _, y in rung) == pytest.approx(0.0, abs=1e-12)


# --- keyboard ---------------------------------------------------------------


def test_held_keys_become_a_stick_position(live):
    panel = live.panel
    assert panel.pilot_input() == man.NEUTRAL
    press(panel, "up")
    press(panel, "right")
    press(panel, "=")
    pilot = panel.pilot_input()
    assert (pilot.pitch, pilot.roll, pilot.throttle) == (1.0, 1.0, 1.0)
    release(panel, "up")
    assert panel.pilot_input().pitch == 0.0


def test_opposite_keys_cancel_and_unknown_keys_are_ignored(live):
    panel = live.panel
    press(panel, "left")
    press(panel, "right")
    press(panel, "ctrl+z")
    assert panel.pilot_input() == man.NEUTRAL


def test_the_autopilot_key_is_edge_triggered(live):
    """Auto-repeat must not toggle the mode once per repeated press."""
    panel = live.panel
    press(panel, "a")
    press(panel, "a")
    press(panel, "a")
    assert panel.take_toggle_request() is True
    assert panel.take_toggle_request() is False


def test_pressing_a_switches_mode_in_the_running_loop(live):
    assert live.ctl.mode is Mode.MANUAL
    press(live.panel, "a")
    live.frame()
    assert live.ctl.mode is Mode.AUTOPILOT
    live.frame()
    assert live.ctl.mode is Mode.AUTOPILOT  # and stays there
    press(live.panel, "a")
    live.frame()
    assert live.ctl.mode is Mode.MANUAL


def test_the_stick_reaches_the_plant(live):
    """A key press has to travel through the panel, the controller and the plant."""
    before = float(quat_to_euler(live.sim.state.quat)[1])
    press(live.panel, "down")  # stick back
    for _ in range(60):
        live.frame()
    assert float(quat_to_euler(live.sim.state.quat)[1]) > before + np.deg2rad(1.0)
    assert float(live.controls.elevator) < 0.0  # trailing edge up


# --- physics/render decoupling ---------------------------------------------


def test_a_frame_runs_whole_physics_steps_to_catch_up(live):
    """Fractions of a step are carried to the next frame, never run short."""
    assert live.advance(5.5 * DT) == 5
    assert live.t == pytest.approx(5 * DT)
    # Ten more half-frames are five more whole steps. Which call each one lands
    # on is a matter of a few ulp in the accumulator and is not asserted; that
    # the carry is neither dropped nor rounded up is the property that matters.
    assert sum(live.advance(0.5 * DT) for _ in range(10)) == 5
    assert live.t == pytest.approx(10 * DT)


def test_a_long_stall_drops_its_backlog_instead_of_chasing_the_clock(live):
    ran = live.advance(60.0)
    assert ran == live.max_steps_per_frame
    assert live._backlog == 0.0
    assert live.advance(DT) == 1  # and the next frame is normal again


def test_the_recorder_logs_physics_steps_not_frames(live):
    frames = 10
    for _ in range(frames):
        live.frame()
    steps = round(live.t / DT)
    assert steps == round(frames / live.panel.fps / DT)  # 20 fps, 50 Hz -> 25
    assert steps > frames  # the physics outruns the rendering, which is the point
    assert len(live.recorder._rows) == steps + 1  # plus the initial state


class FakeAnimation:
    """Just the two attributes _retime touches on a real FuncAnimation."""

    def __init__(self, interval):
        self._interval = float(interval)
        self.event_source = None


def test_the_render_delay_is_corrected_towards_the_target_period(live):
    """`interval` is the gap between frames, so it must absorb the draw cost."""
    target = 1000.0 / live.panel.fps  # 50 ms at 20 fps
    live.animation = FakeAnimation(target)

    live._retime(0.075)  # frames arriving at 75 ms: shorten the delay
    assert live.animation._interval < target
    slow = live.animation._interval

    live._retime(0.010)  # arriving too fast: lengthen it again
    assert live.animation._interval > slow

    # It can never exceed the target -- that would be slower than asked for
    # even with a free draw -- nor drop below a millisecond.
    for _ in range(50):
        live._retime(0.001)
    assert live.animation._interval == pytest.approx(target)
    for _ in range(50):
        live._retime(10.0)
    assert live.animation._interval == pytest.approx(1.0)


def test_retiming_is_skipped_when_nothing_is_driving_the_animation(live):
    live.animation = None
    live._retime(0.075)  # must not raise; the headless path has no timer
    for _ in range(3):
        live.frame()
    assert live.t > 0.0


# --- the blitted animation --------------------------------------------------


def test_the_animation_runs_blitted_and_moves_its_artists(live):
    panel = live.panel
    animation = FuncAnimation(
        panel.fig,
        live.frame,
        init_func=panel.init,
        interval=50,
        blit=True,
        cache_frame_data=False,
    )
    panel.fig.canvas.draw()
    press(panel, "right")
    for _ in range(60):
        animation._step()

    assert live.t > 0.0
    # The trace has caught up with the ring buffer and the aircraft is at its origin.
    trace = np.asarray(panel.trace._verts3d)
    assert trace.shape == (3, len(panel._t))
    assert np.isfinite(trace).any()
    assert panel.trace_now._verts3d[0] == pytest.approx([0.0])
    # The horizon has rolled: the ground quad is no longer axis-aligned.
    assert float(quat_to_euler(live.sim.state.quat)[0]) > np.deg2rad(1.0)
    assert not np.allclose(panel.ground.get_xy()[0, 1], panel.ground.get_xy()[1, 1])
    # Every strip is inside its fixed window, which is what makes blitting legal.
    for line, readout, _ in panel.strips:
        x, _y = line.get_data()
        finite = x[np.isfinite(x)]
        assert finite.min() >= -panel.window - 1e-9 and finite.max() <= 1e-9
        assert readout.get_text().strip() != ""
    assert "MANUAL" in panel.status.get_text()


def test_axis_limits_never_move_because_blitting_would_not_notice(live):
    panel = live.panel
    before = [
        (ax.get_xlim(), ax.get_ylim())
        for ax in (panel.ax_trace, panel.ax_horizon, *[ln.axes for ln, _, _ in panel.strips])
    ]
    press(panel, "down")
    press(panel, "=")
    for _ in range(80):
        live.frame()
    after = [
        (ax.get_xlim(), ax.get_ylim())
        for ax in (panel.ax_trace, panel.ax_horizon, *[ln.axes for ln, _, _ in panel.strips])
    ]
    assert before == after


# --- the trajectory log -----------------------------------------------------


def test_trajectory_round_trips_through_npz(live, tmp_path):
    for _ in range(20):
        live.frame()
    traj = live.trajectory()
    path = tmp_path / "run.npz"
    viz.save(traj, path)
    again = viz.load(path)

    assert again._fields == traj._fields
    for saved, loaded in zip(traj, again):
        assert np.array_equal(saved, loaded)


def test_the_log_has_one_row_per_step_and_the_documented_shapes(live):
    for _ in range(5):
        live.frame()
    traj = live.trajectory()
    n = len(traj.t)
    assert traj.pos_ned.shape == (n, 3)
    assert traj.vel_body.shape == (n, 3)
    assert traj.quat.shape == (n, 4)
    assert traj.omega.shape == (n, 3)
    assert traj.controls.shape == (n, len(Controls._fields))
    assert traj.mode.shape == (n,)
    assert np.diff(traj.t) == pytest.approx(DT)


def test_derived_agrees_with_the_aero_module(live):
    press(live.panel, "right")
    for _ in range(40):
        live.frame()
    traj = live.trajectory()
    d = viz.derived(traj)

    for index in (0, len(traj.t) // 2, -1):
        speed, alpha, beta = air_data(jnp.asarray(traj.vel_body[index]))
        phi, theta, psi = quat_to_euler(jnp.asarray(traj.quat[index]))
        assert d.airspeed[index] == pytest.approx(float(speed))
        assert d.alpha[index] == pytest.approx(float(alpha))
        assert d.beta[index] == pytest.approx(float(beta))
        assert d.phi[index] == pytest.approx(float(phi))
        assert d.theta[index] == pytest.approx(float(theta))
        assert d.psi[index] == pytest.approx(float(psi))
        assert d.altitude[index] == pytest.approx(-traj.pos_ned[index, 2])


def test_a_saved_run_can_be_replotted_without_the_simulator(live, tmp_path):
    """The whole point of the .npz: analysis with no aircraft or solver in sight."""
    press(live.panel, "a")
    for _ in range(30):
        live.frame()
    path = tmp_path / "run.npz"
    viz.save(live.trajectory(), path)

    figure = viz.post_flight(viz.load(path), title="replay")
    assert len(figure.axes) >= 5
    titles = {ax.get_title() for ax in figure.axes}
    assert {
        "ground track",
        "altitude profile",
        "airspeed and incidence",
        "control deflections",
        "mode timeline",
    } <= titles


def test_post_flight_shows_both_modes_on_the_timeline(live):
    for _ in range(10):
        live.frame()
    press(live.panel, "a")
    for _ in range(10):
        live.frame()
    traj = live.trajectory()
    assert set(np.unique(traj.mode)) == {int(Mode.MANUAL), int(Mode.AUTOPILOT)}

    figure = viz.post_flight(traj)
    timeline = next(ax for ax in figure.axes if ax.get_title() == "mode timeline")
    assert [t.get_text() for t in timeline.get_yticklabels()] == ["MANUAL", "AUTOPILOT"]


# --- the wind hook in the live loop -----------------------------------------


def _fly_live(trimmed, targets, frames=40, **kwargs):
    state, controls = trimmed
    ctl = man.start(sense(state), controls, targets, GAINS, AC)
    sim = integrate.init_sim(state, jax.random.PRNGKey(0))
    panel = viz.Panel(targets, window=20.0, fps=20.0)
    live = viz.LiveSim(
        sim, ctl, targets, GAINS, MGAINS, AC, panel, dt=DT, real_time=False, **kwargs
    )
    for _ in range(frames):
        live.frame()
    return np.asarray(live.trajectory().pos_ned)


def test_a_live_run_through_a_wind_field_differs_from_still_air(trimmed, targets):
    """The wind must actually reach the plant.

    This is the test that stops the whole feature being a parameter that is
    accepted and ignored -- the exact shape of latent bugs (a) and (b), which
    survived three sessions because nothing in the project flew through a
    non-zero field.
    """
    from flightsim import wind as wind_mod

    column = wind_mod.UpdraftColumn(
        north=jnp.array(0.0), east=jnp.array(0.0), w0=jnp.array(20.0),
        radius=jnp.array(3000.0), sharpness=jnp.array(6.0),
    )
    blown = _fly_live(
        trimmed, targets,
        wind_model=wind_mod.field_model(lambda p: wind_mod.updraft_wind(p, column)),
    )
    still = _fly_live(trimmed, targets)
    assert abs(blown[-1, 2] - still[-1, 2]) > 1.0  # metres of altitude


def test_the_live_loop_in_still_air_is_untouched_by_the_wind_plumbing(trimmed, targets):
    """Mirrors PROJECT.md section 4's zero-strength-wind row: adding the hook
    must not perturb the default path by one bit."""
    from flightsim import wind as wind_mod

    assert np.array_equal(
        _fly_live(trimmed, targets),
        _fly_live(trimmed, targets, wind_model=wind_mod.zero_wind),
    )
