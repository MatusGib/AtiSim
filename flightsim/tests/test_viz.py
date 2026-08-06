"""Headless tests for the trajectory log and the post-flight figure.

The live cockpit's own tests are in test_panel.py. What is left here is the part
that must work with no simulator in sight: a `.npz` written, loaded and replotted.
"""

import matplotlib

matplotlib.use("Agg")  # before flightsim.panel imports pyplot

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from matplotlib.backend_bases import KeyEvent

import matplotlib.pyplot as plt

from flightsim import autopilot as ap_mod
from flightsim import integrate, manual as man, panel as panel_mod, trim, viz
from flightsim.aero import air_data
from flightsim.aircraft import CRUISE, REGISTRY
from flightsim.manual import Mode
from flightsim.sensors import sense
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


