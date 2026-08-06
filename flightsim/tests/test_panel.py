"""Headless tests for the live cockpit.

Agg supports `copy_from_bbox`/`restore_region`, so `FuncAnimation(blit=True)`
runs for real here -- these are not just "does it import" tests. Keyboard input
is driven by synthesising the same events the GUI backends deliver.
"""

import matplotlib

matplotlib.use("Agg")  # before flightsim.panel imports pyplot

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from matplotlib.animation import FuncAnimation
from matplotlib.backend_bases import KeyEvent

import matplotlib.pyplot as plt

from flightsim import autopilot as ap_mod
from flightsim import integrate, manual as man, panel as panel_mod, trim
from flightsim.aircraft import CRUISE, REGISTRY
from flightsim.manual import Mode
from flightsim.panel import PITCH_SPAN_DEG, _horizon_frame, _ladder
from flightsim.sensors import AirData, Accelerations, accelerometers, sense
from flightsim.state import Controls, quat_to_euler

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
    panel = panel_mod.Panel(targets, window=20.0, fps=20.0)
    return panel_mod.LiveSim(
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


def test_held_keys_become_a_stick_demand(live):
    """`key_demand` is where the keys are ASKING the stick to go.

    Split from `pilot_input` when the stick started ramping: the demand is
    instantaneous, the position that reaches the aircraft is not.
    """
    panel = live.panel
    assert panel.key_demand()["pitch"] == 0.0
    press(panel, "up")
    press(panel, "right")
    press(panel, "=")
    demand = panel.key_demand()
    assert (demand["pitch"], demand["roll"], demand["throttle"]) == (1.0, 1.0, 1.0)
    release(panel, "up")
    assert panel.key_demand()["pitch"] == 0.0


def test_opposite_keys_cancel_and_unknown_keys_are_ignored(live):
    panel = live.panel
    press(panel, "left")
    press(panel, "right")
    press(panel, "ctrl+z")
    assert all(value == 0.0 for value in panel.key_demand().values())


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
    # The horizon has rolled: the ground quad is no longer axis-aligned.
    assert float(quat_to_euler(live.sim.state.quat)[0]) > np.deg2rad(1.0)
    assert not np.allclose(panel.ground.get_xy()[0, 1], panel.ground.get_xy()[1, 1])
    # Every strip is inside its fixed window, which is what makes blitting legal.
    for line, readout, _ in panel.strips:
        x, _y = line.get_data()
        finite = x[np.isfinite(x)]
        assert finite.min() >= -panel.window - 1e-9 and finite.max() <= 1e-9
        assert readout.get_text().strip() != ""
    # Every tape's scale stays inside its own axes -- a tape works by moving the
    # ticks, so an off-scale tick would be drawn outside the cached background.
    for tape in (panel.asi, panel.alt, panel.hdg):
        for tick in tape.ticks:
            data = np.concatenate([tick.get_xdata(), tick.get_ydata()])
            assert np.all(np.abs(data) <= max(tape.span, 1.0) + 1e-9)
        assert tape.box.get_text().strip() != ""
    assert "MANUAL" in panel.status.get_text()
    assert "still air" in panel.status.get_text()


def test_axis_limits_never_move_because_blitting_would_not_notice(live):
    """Every axes on the panel, not just the ones that existed when this was
    written -- the whole point is that a NEW instrument cannot autoscale either."""
    panel = live.panel
    axes = list(panel.fig.axes)
    assert len(axes) >= 10  # T, overlay, strips, status
    before = [(ax.get_xlim(), ax.get_ylim()) for ax in axes]
    press(panel, "down")
    press(panel, "=")
    for _ in range(80):
        live.frame()
    after = [(ax.get_xlim(), ax.get_ylim()) for ax in axes]
    assert before == after


# --- the wind hook in the live loop -----------------------------------------


def _fly_live(trimmed, targets, frames=40, **kwargs):
    state, controls = trimmed
    ctl = man.start(sense(state), controls, targets, GAINS, AC)
    sim = integrate.init_sim(state, jax.random.PRNGKey(0))
    panel = panel_mod.Panel(targets, window=20.0, fps=20.0)
    live = panel_mod.LiveSim(
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


# --- the instruments, driven directly ---------------------------------------
#
# Each one takes a Readout and nothing else, which is the point of the unit
# shape: a gauge can be poked without standing up a figure or flying anything.


def a_readout(**overrides):
    """A Readout with everything quiet, for moving one input at a time."""
    air = AirData(
        airspeed=jnp.array(236.0), alpha=jnp.array(0.05), beta=jnp.array(0.0),
        phi=jnp.array(0.0), theta=jnp.array(0.08), psi=jnp.array(0.0),
        p=jnp.array(0.0), q=jnp.array(0.0), r=jnp.array(0.0),
        altitude=jnp.array(12192.0), vertical_speed=jnp.array(0.0),
    )
    quiet = Controls(*[jnp.array(0.0)] * 4)
    base = dict(
        t=0.0, air=air,
        accel=Accelerations(n_x=jnp.array(0.0), n_y=jnp.array(0.0), n_z=jnp.array(1.0)),
        controls=quiet, reference=quiet, mode=Mode.MANUAL,
        wind_ned=np.zeros(3), omega_gust=np.zeros(3), field=None,
    )
    base.update(overrides)
    return panel_mod.Readout(**base)


def test_the_vsi_needle_moves_up_in_a_climb_and_down_in_a_descent(targets):
    p = panel_mod.Panel(targets, window=20.0, fps=20.0, aircraft_name="boeing747")
    air = a_readout().air

    p.vsi.update(a_readout(air=air._replace(vertical_speed=jnp.array(+8.0))))
    climbing = p.vsi.needle.get_ydata()[-1]
    p.vsi.update(a_readout(air=air._replace(vertical_speed=jnp.array(-8.0))))
    descending = p.vsi.needle.get_ydata()[-1]

    assert climbing > 0.0 > descending
    assert climbing == pytest.approx(-descending)


def test_the_vsi_clips_off_scale_rather_than_moving_its_axes(targets):
    p = panel_mod.Panel(targets, window=20.0, fps=20.0, aircraft_name="boeing747")
    air = a_readout().air
    p.vsi.update(a_readout(air=air._replace(vertical_speed=jnp.array(500.0))))
    assert p.vsi.needle.get_ydata()[-1] == pytest.approx(1.0)
    assert "500" in p.vsi.readout.get_text()  # the NUMBER is still right


def test_the_alpha_gauge_names_the_band_the_model_is_in(targets):
    """PROJECT.md section 7's ceiling, on the panel instead of in a footnote."""
    p = panel_mod.Panel(targets, window=20.0, fps=20.0, aircraft_name="boeing747")
    air = a_readout().air

    p.alpha_gauge.update(a_readout(air=air._replace(alpha=jnp.deg2rad(4.0))))
    assert p.alpha_gauge.state() == "linear"
    p.alpha_gauge.update(a_readout(air=air._replace(alpha=jnp.deg2rad(11.0))))
    assert p.alpha_gauge.state() == "marginal"
    p.alpha_gauge.update(a_readout(air=air._replace(alpha=jnp.deg2rad(14.0))))
    assert p.alpha_gauge.state() == "invalid"
    assert "invalid" in p.alpha_gauge.readout.get_text()


def test_the_load_factor_gauge_holds_the_peak_excursion(targets):
    """In an encounter the excursion IS the result, and it is over in a second."""
    p = panel_mod.Panel(targets, window=20.0, fps=20.0, aircraft_name="boeing747")
    for n_z in (1.0, 2.4, 1.1, 0.2, 1.0):
        p.nz_gauge.update(a_readout(accel=Accelerations(
            n_x=jnp.array(0.0), n_y=jnp.array(0.0), n_z=jnp.array(n_z)
        )))
    assert p.nz_gauge.peak_high == pytest.approx(2.4)
    assert p.nz_gauge.peak_low == pytest.approx(0.2)


def test_the_gust_gauge_rows_are_p_q_r_top_to_bottom(targets):
    p = panel_mod.Panel(targets, window=20.0, fps=20.0, aircraft_name="boeing747")
    p.gust_gauge.update(a_readout(omega_gust=np.array([0.10, 0.05, -0.02])))
    # Bar 0 is the bottom row and is labelled r; omega_gust is ordered (p, q, r).
    assert p.gust_gauge.bars[0].get_xdata()[-1] == pytest.approx(-0.02)
    assert p.gust_gauge.bars[1].get_xdata()[-1] == pytest.approx(0.05)
    assert p.gust_gauge.bars[2].get_xdata()[-1] == pytest.approx(0.10)


def test_the_wind_arrow_points_the_way_the_air_is_moving(targets):
    p = panel_mod.Panel(targets, window=20.0, fps=20.0, aircraft_name="boeing747")
    p.wind_gauge.update(a_readout(wind_ned=np.array([0.0, 20.0, 0.0])))  # blowing east
    x, y = p.wind_gauge.arrow.get_xdata()[-1], p.wind_gauge.arrow.get_ydata()[-1]
    assert x > 0.0 and abs(y) < 1e-9  # screen x is east
    assert "20.0" in p.wind_gauge.readout.get_text()


# --- the slip ball, which is the one that pins the physics -------------------


def test_the_ball_indicates_the_rudder_that_would_reduce_the_sideslip(live):
    """A slip ball is a pendulum: it reads lateral specific force, not beta.

    Hold right rudder from trim. The aircraft develops sideslip, and the ball
    must fall to the side whose pedal removes it -- "step on the ball". Wiring
    beta into the ball, or getting the sign backwards, fails here.
    """
    press(live.panel, ".")  # right pedal
    for _ in range(60):
        live.frame()

    air = sense(live.sim.state, live.sim.wind_ned)
    accel = accelerometers(
        live.sim.state, live.controls, live.ac, live.sim.wind_ned, live.sim.omega_gust
    )
    assert abs(float(air.beta)) > np.deg2rad(0.5)  # there IS a sideslip to indicate
    assert live.panel.slip.offset == pytest.approx(
        -float(accel.n_y) / panel_mod.SLIP_SPAN, abs=1e-9
    )
    # Step on the ball: the indicated pedal opposes the rudder being held.
    assert live.panel.slip.offset < 0.0


def test_the_ball_and_beta_are_different_numbers(live):
    """So a display quietly fed beta would not read the same thing.

    They agree in steady coordinated flight, which is exactly why this test
    holds a rudder input instead of flying straight.
    """
    press(live.panel, ".")
    for _ in range(60):
        live.frame()

    air = sense(live.sim.state, live.sim.wind_ned)
    accel = accelerometers(
        live.sim.state, live.controls, live.ac, live.sim.wind_ned, live.sim.omega_gust
    )
    # Both in their own natural units; the point is that neither is a scaling of
    # the other, so no choice of full scale makes the ball equal to beta.
    assert abs(float(air.beta)) > np.deg2rad(0.5)
    assert abs(float(accel.n_y)) > 1e-3
    assert float(air.beta) * float(accel.n_y) < 0.0  # they even disagree in sign


# --- the ramped stick --------------------------------------------------------


def test_a_one_step_tap_gives_less_than_full_deflection(live):
    """The whole point of ramping: a tap is now a small input, not a full one."""
    press(live.panel, "up")
    live.panel.step_stick(DT)
    assert 0.0 < live.panel.pilot_input().pitch < 1.0


def test_holding_a_key_reaches_full_travel(live):
    press(live.panel, "up")
    for _ in range(int(1.0 / DT)):
        live.panel.step_stick(DT)
    assert live.panel.pilot_input().pitch == pytest.approx(1.0)


def test_releasing_springs_the_stick_back_to_centre(live):
    press(live.panel, "up")
    for _ in range(int(1.0 / DT)):
        live.panel.step_stick(DT)
    release(live.panel, "up")
    for _ in range(int(1.0 / DT)):
        live.panel.step_stick(DT)
    assert live.panel.pilot_input().pitch == pytest.approx(0.0)


def test_the_throttle_and_trim_are_not_ramped_because_manual_integrates_them(live):
    """A ramp on top of an integrator is just a second lag."""
    press(live.panel, "=")
    press(live.panel, "]")
    assert live.panel.pilot_input().throttle == 1.0
    assert live.panel.pilot_input().trim == 1.0


def test_the_stick_ramps_per_physics_step_not_per_frame(trimmed, targets):
    """Same total physics steps, different frame grouping, same stick position.

    Stepping the ramp in `frame` instead of inside `advance` would make the
    control feel depend on the render rate -- and the render rate is corrected
    against the wall clock, so it is not even constant.
    """
    def fly(steps_per_frame, frames):
        state, controls = trimmed
        ctl = man.start(sense(state), controls, targets, GAINS, AC)
        sim = integrate.init_sim(state, jax.random.PRNGKey(0))
        p = panel_mod.Panel(targets, window=20.0, fps=20.0)
        run = panel_mod.LiveSim(
            sim, ctl, targets, GAINS, MGAINS, AC, p, dt=DT, real_time=False
        )
        press(p, "up")
        for _ in range(frames):
            run.advance(steps_per_frame * DT)
        return p.pilot_input().pitch

    assert fly(1, 10) == pytest.approx(fly(10, 1))
    assert 0.0 < fly(1, 10) < 1.0


# --- trim --------------------------------------------------------------------


def test_trim_here_holds_the_surfaces_the_stick_was_holding(trimmed, targets):
    """The reason trim exists: trim out the pressure, let go, stay put.

    It has to be pressed while the stick is STILL HELD -- that is what "trim out
    this pressure" means. Once the stick has sprung back the surfaces are already
    at the reference, so trim-here is correctly a no-op, which is what a first
    version of this test discovered the hard way.

    The same manoeuvre is flown twice and trimmed only once, so the comparison is
    against the untrimmed aircraft rather than against a tolerance somebody chose.
    """
    def fly(trim_it):
        state, controls = trimmed
        ctl = man.start(sense(state), controls, targets, GAINS, AC)
        sim = integrate.init_sim(state, jax.random.PRNGKey(0))
        p = panel_mod.Panel(targets, window=20.0, fps=20.0)
        run = panel_mod.LiveSim(
            sim, ctl, targets, GAINS, MGAINS, AC, p, dt=DT, real_time=False
        )
        press(p, "up")  # stick forward: push the nose over, and HOLD it
        for _ in range(60):
            run.frame()
        if trim_it:
            run.request_trim_here()
            run.frame()
        release(p, "up")
        for _ in range(100):
            run.frame()
        return (
            float(run.controls.elevator),
            float(quat_to_euler(run.sim.state.quat)[1]),
            float(run.ctl.manual.reference.elevator),
        )

    held_elev, held_theta, held_ref = fly(trim_it=True)
    stale_elev, stale_theta, stale_ref = fly(trim_it=False)

    # Trimmed: the released stick centres to the deflection that was being held.
    assert held_ref != pytest.approx(stale_ref)
    assert held_elev == pytest.approx(held_ref, abs=1e-6)
    # Untrimmed: it springs back to the stale handover value and the nose comes up.
    assert stale_elev == pytest.approx(stale_ref, abs=1e-6)
    assert held_theta < stale_theta


def test_trim_here_is_a_no_op_under_the_autopilot(live):
    """`toggle` already reseeds the reference on the way out, so there is
    nothing left for trim-here to do -- and it must not disturb the autopilot."""
    press(live.panel, "a")
    live.frame()
    before = live.ctl
    live.request_trim_here()
    live.frame()
    assert live.ctl.mode is Mode.AUTOPILOT
    assert float(live.ctl.manual.reference.elevator) == pytest.approx(
        float(before.manual.reference.elevator)
    )


# --- where the wind field is -------------------------------------------------


def test_the_vortex_range_is_a_north_distance_with_no_bearing():
    """The cores are infinite east-west lines, so there is no bearing to report
    and no way to miss one by turning."""
    from flightsim import wind as wind_mod
    from flightsim.state import State

    array = wind_mod.VortexArray(
        north=jnp.array([1000.0, 2000.0]), down=jnp.array([-12192.0, -12192.0]),
        r0=jnp.array(183.0), v0=jnp.array(25.9),
    )
    ranger = panel_mod.vortex_range(array, label="vortex test")

    def at(north):
        return ranger(State(
            pos_ned=jnp.array([north, 0.0, -12192.0]),
            vel_body=jnp.array([236.0, 0.0, 0.0]),
            quat=jnp.array([1.0, 0.0, 0.0, 0.0]),
            omega=jnp.zeros(3),
        ))

    assert at(1000.0).distance == pytest.approx(0.0, abs=1e-9)
    assert at(0.0).distance == pytest.approx(1000.0, abs=1e-9)
    assert at(500.0).distance < at(0.0).distance
    assert at(1500.0).distance == pytest.approx(500.0, abs=1e-9)  # the second core
    assert at(0.0).bearing is None
    assert at(0.0).closing == pytest.approx(236.0, abs=1e-6)
    assert "core 1" in at(0.0).label and "core 2" in at(1500.0).label


def test_the_updraft_range_has_a_bearing_because_a_column_is_a_point():
    from flightsim import wind as wind_mod
    from flightsim.state import State

    column = wind_mod.UpdraftColumn(
        north=jnp.array(0.0), east=jnp.array(3000.0), w0=jnp.array(24.0),
        radius=jnp.array(2400.0), sharpness=jnp.array(6.0),
    )
    ranger = panel_mod.updraft_range(column, label="updraft")
    got = ranger(State(
        pos_ned=jnp.array([0.0, 0.0, -12192.0]),
        vel_body=jnp.array([236.0, 0.0, 0.0]),
        quat=jnp.array([1.0, 0.0, 0.0, 0.0]),
        omega=jnp.zeros(3),
    ))
    assert got.distance == pytest.approx(3000.0, abs=1e-9)
    assert np.degrees(got.bearing) == pytest.approx(90.0, abs=1e-6)  # due east
    assert got.closing == pytest.approx(0.0, abs=1e-9)  # flying north, not closing


# --- flying a cited field, end to end ----------------------------------------
#
# `field_ahead` is what scripts/fly.py actually calls, so these exercise the
# real path rather than a copy of it. A test that rebuilt the field itself would
# pass happily while fly.py placed it somewhere else.


def test_field_ahead_puts_the_first_core_where_the_lead_in_says(targets):
    model, ranger, note = panel_mod.field_ahead(
        "hannibal", airspeed=V, altitude=H, lead_in=40.0
    )
    from flightsim.state import State
    from flightsim.wind import PARKS_CASES

    expected = 40.0 * PARKS_CASES["hannibal"]["r0"]
    got = ranger(State(
        pos_ned=jnp.array([0.0, 0.0, -H]),
        vel_body=jnp.array([V, 0.0, 0.0]),
        quat=jnp.array([1.0, 0.0, 0.0, 0.0]),
        omega=jnp.zeros(3),
    ))
    assert got.distance == pytest.approx(expected, rel=1e-9)
    assert got.bearing is None
    assert "hannibal" in note and "core" in note


def test_still_air_asks_for_no_field_at_all(targets):
    model, ranger, note = panel_mod.field_ahead("none", airspeed=V, altitude=H)
    assert ranger is None
    assert note == "still air"


def test_an_unknown_field_is_refused_rather_than_silently_still_air():
    with pytest.raises(ValueError, match="unknown wind field"):
        panel_mod.field_ahead("hurricane", airspeed=V, altitude=H)


def test_flying_the_parks_array_closes_the_range_and_moves_the_gust_bars(trimmed, targets):
    """The whole feature, end to end: place the field, fly at it, watch it arrive.

    A short lead-in is used so the encounter fits in a test. That is BELOW the
    ~12 core radii PROJECT.md section 9 warns about, so the aircraft starts out
    of equilibrium and no number from this run means anything -- which is fine,
    because what is asserted is that the machinery connects, not what the
    encounter measures.
    """
    model, ranger, _ = panel_mod.field_ahead(
        "hannibal", airspeed=V, altitude=H, lead_in=14.0
    )
    state, controls = trimmed
    ctl = man.start(sense(state), controls, targets, GAINS, AC)
    sim = integrate.init_sim(state, jax.random.PRNGKey(0))
    p = panel_mod.Panel(targets, window=20.0, fps=20.0)
    run = panel_mod.LiveSim(
        sim, ctl, targets, GAINS, MGAINS, AC, p,
        dt=DT, real_time=False, wind_model=model, field_range=ranger,
    )

    start = ranger(run.sim.state).distance
    gust_seen = 0.0
    for _ in range(120):
        run.frame()
        gust_seen = max(gust_seen, float(np.abs(run.sim.omega_gust).max()))

    assert ranger(run.sim.state).distance < start  # the range closed
    assert gust_seen > 1e-4  # the rotational gust reached the aircraft
    assert "vortex hannibal core" in p.status.get_text()
    assert "closing" in p.status.get_text()
    # And the panel showed the wind rather than reporting still air.
    assert np.abs(run.sim.wind_ned).max() > 1e-3
    assert len(p.wind_gauge.arrow.get_xdata()) > 0  # the arrow is drawn
