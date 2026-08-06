"""Live instrument panel and post-flight plots.

Three things live here: a `.npz` trajectory log, a blitted live panel driven by
`FuncAnimation`, and a post-flight figure.

**Physics and rendering are decoupled.** The animation timer asks for a frame at
roughly 20 fps; `LiveSim.advance` then runs however many fixed 50 Hz physics
steps are needed to catch up with the wall clock. The physics step never changes
size, because a variable step would make a run unreproducible and would put the
integrator error at the mercy of a garbage collection pause. A frame that falls
badly behind drops its backlog rather than trying to run it off, which is the
usual guard against a slow frame turning into a slower one. The render delay is
corrected from the measured frame period (`LiveSim._retime`), because
matplotlib's `interval` is the gap between frames rather than the period.

**Everything on the panel is a pre-allocated artist updated with `set_data`, and
the animation blits.** That constrains the design in one way worth naming:
blitting redraws the animated artists over a cached background, so no axis limit
on the panel may ever change. The consequences are deliberate:

  - The rolling strips are windowed on time-before-now, so the data slides
    through fixed axes instead of the axes chasing the data. Their y-limits are
    centred on the targets, which are fixed for a run.
  - The 3D trace is plotted relative to the aircraft's current position, so the
    box never has to move. It is a trailing ribbon with the aircraft at the
    origin. The absolute path is in the post-flight ground track, which is
    where you would actually measure it.
  - The aircraft symbol and pitch ladder are animated artists despite never
    moving, because the horizon polygons are drawn over the whole axes and would
    otherwise bury anything left in the background.

Values outside a strip's window clip against the edge; the numeric readout beside
each strip is always correct, so nothing is ever silently wrong, only off-scale.
"""

import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import Array
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon

from flightsim import manual as man
from flightsim.aircraft import Aircraft
from flightsim.autopilot import Gains, Targets, wrap_pi
from flightsim.integrate import SimState, step
from flightsim.manual import Controller, ManualGains, Mode, PilotInput
from flightsim.sensors import sense
from flightsim.state import Controls, State
from flightsim.units import RAD2DEG

# ---------------------------------------------------------------------------
# Trajectory log
# ---------------------------------------------------------------------------


class Trajectory(NamedTuple):
    """A flown run, as numpy. Controls are stored in `Controls._fields` order.

    `wind_ned` and `omega_gust` are recorded because without them a saved run
    cannot be corrected even in principle: every incidence angle in it would be
    ground-relative with no way to recover the air-relative one. They are last so
    that `.npz` files written before they existed still load -- see `load`.
    """

    t: np.ndarray  # (n,) s
    pos_ned: np.ndarray  # (n, 3) m
    vel_body: np.ndarray  # (n, 3) m/s
    quat: np.ndarray  # (n, 4)
    omega: np.ndarray  # (n, 3) rad/s
    controls: np.ndarray  # (n, 4)
    mode: np.ndarray  # (n,) Mode
    wind_ned: np.ndarray  # (n, 3) m/s NED, the wind actually applied
    omega_gust: np.ndarray  # (n, 3) rad/s body, the gust rate actually applied


def save(traj: Trajectory, path) -> None:
    """Write a run to .npz so it can be re-analysed without re-flying it.

    This matters more than it looks: comparing turbulence realisations means
    comparing runs against each other, and re-flying to change a plot loses the
    realisation unless the PRNG key is also pinned.
    """
    np.savez_compressed(path, **traj._asdict())


def load(path) -> Trajectory:
    """Read a run back.

    Files written before wind was recorded load as still air rather than being
    rejected. That is honest for those files -- every run predating the wind
    columns was flown in still air because nothing else was possible -- and it
    keeps them analysable instead of stranding them.
    """
    with np.load(path) as data:
        fields = {name: data[name] for name in Trajectory._fields if name in data}
        n = len(fields["t"])
        for name in ("wind_ned", "omega_gust"):
            fields.setdefault(name, np.zeros((n, 3)))
        return Trajectory(**fields)


class Recorder:
    """Full-rate log. One row per physics step, not per rendered frame."""

    def __init__(self) -> None:
        self._rows: list[tuple] = []

    def append(
        self,
        t: float,
        sim: SimState,
        controls: Controls,
        mode: Mode,
    ) -> None:
        """Log one physics step.

        Takes the whole `SimState` rather than just `State` so the wind that was
        actually applied is recorded alongside the motion it caused. Passing only
        the rigid-body state is what made saved runs uncorrectable.
        """
        state = sim.state
        self._rows.append(
            (
                t,
                np.asarray(state.pos_ned, dtype=float),
                np.asarray(state.vel_body, dtype=float),
                np.asarray(state.quat, dtype=float),
                np.asarray(state.omega, dtype=float),
                np.array([float(c) for c in controls]),
                int(mode),
                np.asarray(sim.wind_ned, dtype=float),
                np.asarray(sim.omega_gust, dtype=float),
            )
        )

    def trajectory(self) -> Trajectory:
        columns = list(zip(*self._rows))
        return Trajectory(
            t=np.array(columns[0]),
            pos_ned=np.stack(columns[1]),
            vel_body=np.stack(columns[2]),
            quat=np.stack(columns[3]),
            omega=np.stack(columns[4]),
            controls=np.stack(columns[5]),
            mode=np.array(columns[6], dtype=int),
            wind_ned=np.stack(columns[7]),
            omega_gust=np.stack(columns[8]),
        )


class Derived(NamedTuple):
    """Air-relative incidence and inertial attitude, over a whole trajectory."""

    airspeed: np.ndarray  # m/s, AIR-RELATIVE
    alpha: np.ndarray  # rad, AIR-RELATIVE
    beta: np.ndarray  # rad, AIR-RELATIVE
    phi: np.ndarray  # rad, inertial
    theta: np.ndarray  # rad, inertial
    psi: np.ndarray  # rad, inertial
    altitude: np.ndarray  # m


def derived(traj: Trajectory) -> Derived:
    """Sensed air data for a whole trajectory, in one vmap.

    Uses the recorded wind, so the incidence angles are air-relative. Before the
    wind was recorded these were ground-relative and wrong by up to 7 deg in a
    vortex encounter, and no still-air test could see it.
    """
    states = State(
        pos_ned=jnp.asarray(traj.pos_ned),
        vel_body=jnp.asarray(traj.vel_body),
        quat=jnp.asarray(traj.quat),
        omega=jnp.asarray(traj.omega),
    )
    air = jax.vmap(sense)(states, jnp.asarray(traj.wind_ned))
    return Derived(
        airspeed=np.asarray(air.airspeed),
        alpha=np.asarray(air.alpha),
        beta=np.asarray(air.beta),
        phi=np.asarray(air.phi),
        theta=np.asarray(air.theta),
        psi=np.asarray(air.psi),
        altitude=np.asarray(air.altitude),
    )


# ---------------------------------------------------------------------------
# Live panel
# ---------------------------------------------------------------------------

# Held keys. Signs follow flightsim.manual: the arrows are a centre stick, so
# "up" is stick forward and pitches the nose down.
KEYMAP = {
    "up": ("pitch", +1.0),
    "down": ("pitch", -1.0),
    "right": ("roll", +1.0),
    "left": ("roll", -1.0),
    ".": ("yaw", +1.0),
    ",": ("yaw", -1.0),
    "=": ("throttle", +1.0),
    "-": ("throttle", -1.0),
}
TOGGLE_KEY = "a"

HELP = (
    "arrows  stick (up = nose down)     , .  rudder\n"
    "-  =    throttle                    a   autopilot"
)

# Instrument-panel colours.
_BG = "#101418"
_FG = "#c8d0d8"
_SKY = "#2c6fb5"
_GROUND = "#7a5230"
_SYMBOL = "#ffd24a"
_TRACE = "#4ec9b0"

PITCH_SPAN_DEG = 30.0  # degrees from the horizon to the top of the horizon ball
_HORIZON_L = 4.0  # half-length of the sky/ground quads, in axes units


def _push(buffer: np.ndarray, value) -> None:
    """Shift a ring buffer left by one and write the newest sample at the end."""
    buffer[:-1] = buffer[1:]
    buffer[-1] = value


def _horizon_frame(phi: float, theta_deg: float):
    """Screen-frame horizon: (centre point, along-horizon unit, sky-side unit).

    Rolling right by phi rotates the world anticlockwise in the pilot's view, so
    the sky normal is the +phi rotation of screen-up. Pitching up pushes the
    horizon down the ball, hence the minus sign on the offset.
    """
    along = np.array([np.cos(phi), np.sin(phi)])
    sky = np.array([-np.sin(phi), np.cos(phi)])
    centre = -(theta_deg / PITCH_SPAN_DEG) * sky
    return centre, along, sky


def _quad(centre, along, normal, length, depth) -> np.ndarray:
    return np.array(
        [
            centre - length * along,
            centre + length * along,
            centre + length * along + depth * normal,
            centre - length * along + depth * normal,
        ]
    )


def _ladder(centre, along, sky, theta_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Pitch ladder as one polyline with NaN breaks between rungs."""
    xs, ys = [], []
    for mark in (-30, -20, -10, 10, 20, 30):
        offset = centre + ((mark - theta_deg) / PITCH_SPAN_DEG) * sky
        half = 0.18
        for end in (offset - half * along, offset + half * along):
            xs.append(end[0])
            ys.append(end[1])
        xs.append(np.nan)
        ys.append(np.nan)
    return np.array(xs), np.array(ys)


# Fixed aircraft symbol: two wings and a centre pip, NaN-separated.
_SYMBOL_X = np.array([-0.45, -0.15, np.nan, -0.03, 0.03, np.nan, 0.15, 0.45])
_SYMBOL_Y = np.array([0.0, 0.0, np.nan, 0.0, 0.0, np.nan, 0.0, 0.0])


class Panel:
    """The figure, its pre-allocated artists, and the held-keys set.

    The panel owns the keyboard because it owns the canvas that generates the
    events. It does not own the aircraft: a key press only updates a set, and
    `LiveSim` asks for the resulting stick position once per physics step.
    """

    def __init__(self, targets: Targets, *, window: float = 60.0, fps: float = 20.0):
        self.targets = targets
        self.window = window
        self.fps = fps
        self.held: set[str] = set()
        self._toggle_requested = False

        target_speed = float(targets.airspeed)
        target_altitude = float(targets.altitude)
        target_heading = float(targets.heading) * RAD2DEG

        n = int(window * fps) + 1
        self._t = np.full(n, np.nan)
        self._pos = np.full((n, 3), np.nan)
        self._speed = np.full(n, np.nan)
        self._altitude = np.full(n, np.nan)
        self._heading = np.full(n, np.nan)

        self.fig = plt.figure(figsize=(14.0, 8.0), facecolor=_BG)
        if getattr(self.fig.canvas, "manager", None) is not None:
            self.fig.canvas.manager.set_window_title("flightsim")
        grid = self.fig.add_gridspec(
            3,
            3,
            width_ratios=(1.3, 0.95, 0.85),
            height_ratios=(1.0, 1.0, 0.5),
            hspace=0.45,
            wspace=0.38,
        )

        self._build_trace(grid, target_speed)
        self._build_horizon(grid)
        self.strips = [
            self._build_strip(grid[0, 2], "TAS  m/s", target_speed, 40.0),
            self._build_strip(grid[1, 2], "ALT  m", target_altitude, 600.0),
            self._build_strip(grid[2, 2], "HDG  deg", target_heading, 90.0, xlabel=True),
        ]
        self._build_status(grid)
        self._connect_keys()

    # -- construction ------------------------------------------------------

    def _style(self, ax) -> None:
        ax.set_facecolor(_BG)
        ax.tick_params(colors=_FG, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#39424b")

    def _build_trace(self, grid, target_speed: float) -> None:
        # The box is sized to the distance covered in one window, so the trail
        # just fills it at the target speed.
        reach = max(target_speed * self.window / 1000.0, 1.0)  # km
        self.ax_trace = self.fig.add_subplot(grid[0:2, 0], projection="3d")
        ax = self.ax_trace
        ax.set_facecolor(_BG)
        ax.set_xlim(-reach, reach)
        ax.set_ylim(-reach, reach)
        ax.set_zlim(-max(reach / 10.0, 0.3), max(reach / 10.0, 0.3))
        ax.set_xlabel("east  km", color=_FG, fontsize=8)
        ax.set_ylabel("north  km", color=_FG, fontsize=8)
        ax.set_zlabel("up  km", color=_FG, fontsize=8)
        ax.tick_params(colors=_FG, labelsize=7)
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.set_pane_color((0.06, 0.08, 0.10, 1.0))
        ax.set_title(
            f"trajectory  (last {self.window:.0f} s, relative to aircraft)",
            color=_FG,
            fontsize=9,
        )
        ax.view_init(elev=24.0, azim=-58.0)
        # The box has to be +-reach in both horizontal axes, because the trail
        # can lie in any direction, but a trail only ever runs one way from the
        # aircraft and so uses half of it. Zoom compensates rather than shrinking
        # the box, which would clip a straight run at the window length.
        ax.set_box_aspect((1.0, 1.0, 0.55), zoom=1.15)
        (self.trace,) = ax.plot([], [], [], color=_TRACE, lw=1.4, animated=True)
        (self.trace_now,) = ax.plot(
            [], [], [], marker="o", color=_SYMBOL, ms=5, animated=True
        )

    def _build_horizon(self, grid) -> None:
        self.ax_horizon = self.fig.add_subplot(grid[0:2, 1])
        ax = self.ax_horizon
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor(_BG)
        for spine in ax.spines.values():
            spine.set_color("#39424b")
        ax.set_title("attitude", color=_FG, fontsize=9)

        self.sky = Polygon(np.zeros((4, 2)), closed=True, color=_SKY, animated=True)
        self.ground = Polygon(
            np.zeros((4, 2)), closed=True, color=_GROUND, animated=True
        )
        ax.add_patch(self.sky)
        ax.add_patch(self.ground)
        (self.horizon_line,) = ax.plot([], [], color="white", lw=1.6, animated=True)
        (self.ladder,) = ax.plot([], [], color="white", lw=1.0, animated=True)
        (self.symbol,) = ax.plot(
            _SYMBOL_X, _SYMBOL_Y, color=_SYMBOL, lw=2.4, animated=True
        )

    def _build_strip(
        self, cell, label: str, centre: float, span: float, *, xlabel: bool = False
    ):
        ax = self.fig.add_subplot(cell)
        self._style(ax)
        ax.set_xlim(-self.window, 0.0)
        ax.set_ylim(centre - span, centre + span)
        ax.set_xticks([-self.window, -self.window / 2.0, 0.0])
        ax.set_ylabel(label, color=_FG, fontsize=8)
        ax.axhline(centre, color="#5a6a78", lw=0.9, ls="--")  # target, background
        ax.grid(color="#232b33", lw=0.6)
        if xlabel:
            ax.set_xlabel("seconds before now", color=_FG, fontsize=8)
        (line,) = ax.plot([], [], color=_TRACE, lw=1.3, animated=True)
        readout = ax.text(
            0.02,
            0.90,
            "",
            transform=ax.transAxes,
            color=_SYMBOL,
            fontsize=9,
            family="monospace",
            va="top",
            animated=True,
        )
        return line, readout, centre

    def _build_status(self, grid) -> None:
        ax = self.fig.add_subplot(grid[2, 0:2])
        ax.set_axis_off()
        ax.set_facecolor(_BG)
        ax.text(
            0.0,
            0.02,
            HELP,
            transform=ax.transAxes,
            color="#7d8a96",
            fontsize=9,
            family="monospace",
            va="bottom",
        )
        self.status = ax.text(
            0.0,
            0.98,
            "",
            transform=ax.transAxes,
            color=_FG,
            fontsize=10,
            family="monospace",
            va="top",
            animated=True,
        )

    # -- keyboard ----------------------------------------------------------

    def _connect_keys(self) -> None:
        # matplotlib binds the arrow keys to pan/zoom history by default, which
        # would fight the stick. Drop the default handler for this figure only,
        # rather than mutating the global rcParams keymaps.
        manager = getattr(self.fig.canvas, "manager", None)
        handler_id = getattr(manager, "key_press_handler_id", None)
        if handler_id is not None:
            self.fig.canvas.mpl_disconnect(handler_id)
        self.fig.canvas.mpl_connect("key_press_event", self.on_press)
        self.fig.canvas.mpl_connect("key_release_event", self.on_release)

    def on_press(self, event) -> None:
        if event.key is None:
            return
        if event.key == TOGGLE_KEY:
            self._toggle_requested = True
        self.held.add(event.key)

    def on_release(self, event) -> None:
        self.held.discard(event.key)

    def take_toggle_request(self) -> bool:
        """Edge-triggered: a press is consumed once, however long the key is held."""
        requested, self._toggle_requested = self._toggle_requested, False
        return requested

    def pilot_input(self) -> PilotInput:
        axes = {"pitch": 0.0, "roll": 0.0, "yaw": 0.0, "throttle": 0.0}
        for key in self.held:
            if key in KEYMAP:
                axis, sign = KEYMAP[key]
                axes[axis] += sign
        # Opposite keys held together cancel; the clip is for a stuck repeat.
        return PilotInput(**{k: float(np.clip(v, -1.0, 1.0)) for k, v in axes.items()})

    # -- drawing -----------------------------------------------------------

    @property
    def artists(self) -> list:
        # Order matters under blitting: within the horizon axes the artists are
        # drawn in this order, so the polygons must come before the symbol.
        return [
            self.trace,
            self.trace_now,
            self.sky,
            self.ground,
            self.horizon_line,
            self.ladder,
            self.symbol,
            self.status,
            *[artist for line, readout, _ in self.strips for artist in (line, readout)],
        ]

    def init(self) -> list:
        return self.artists

    def update(
        self, t: float, sim: SimState, controls: Controls, mode: Mode
    ) -> list:
        # Sensed from the wind the last step applied, so the readouts are
        # air-relative under a wind field rather than quietly ground-relative.
        air = sense(sim.state, sim.wind_ned)
        speed, alpha, beta = float(air.airspeed), float(air.alpha), float(air.beta)
        phi, theta, psi = float(air.phi), float(air.theta), float(air.psi)
        altitude = float(air.altitude)
        state = sim.state
        target_heading = float(self.targets.heading)
        # Unwrapped about the target so a heading either side of north does not
        # jump the width of the strip.
        heading = (
            target_heading + float(wrap_pi(jnp.array(psi - target_heading)))
        ) * RAD2DEG

        _push(self._t, t)
        _push(self._pos, np.asarray(state.pos_ned, dtype=float))
        _push(self._speed, speed)
        _push(self._altitude, altitude)
        _push(self._heading, heading)

        relative = (self._pos - self._pos[-1]) / 1000.0
        self.trace.set_data_3d(relative[:, 1], relative[:, 0], -relative[:, 2])
        self.trace_now.set_data_3d([0.0], [0.0], [0.0])

        theta_deg = theta * RAD2DEG
        centre, along, sky = _horizon_frame(phi, theta_deg)
        self.sky.set_xy(_quad(centre, along, sky, _HORIZON_L, 2.0 * _HORIZON_L))
        self.ground.set_xy(_quad(centre, along, sky, _HORIZON_L, -2.0 * _HORIZON_L))
        self.horizon_line.set_data(
            [centre[0] - _HORIZON_L * along[0], centre[0] + _HORIZON_L * along[0]],
            [centre[1] - _HORIZON_L * along[1], centre[1] + _HORIZON_L * along[1]],
        )
        self.ladder.set_data(*_ladder(centre, along, sky, theta_deg))

        elapsed = self._t - t
        for (line, readout, _), values, fmt in zip(
            self.strips,
            (self._speed, self._altitude, self._heading),
            ("{:8.1f}", "{:8.0f}", "{:8.1f}"),
        ):
            line.set_data(elapsed, values)
            readout.set_text(fmt.format(values[-1]))

        self.status.set_text(
            f"{Mode(mode).name:<10s} t {t:7.1f} s\n"
            f"TAS {speed:7.1f} m/s   ALT {altitude:8.0f} m   "
            f"HDG {np.degrees(psi) % 360.0:5.1f} deg\n"
            f"pitch {theta_deg:+6.1f}   bank {phi * RAD2DEG:+6.1f}   "
            f"aoa {alpha * RAD2DEG:+5.1f}   beta {beta * RAD2DEG:+5.1f}\n"
            f"elev {float(controls.elevator) * RAD2DEG:+6.2f}   "
            f"ail {float(controls.aileron) * RAD2DEG:+6.2f}   "
            f"rud {float(controls.rudder) * RAD2DEG:+6.2f}   "
            f"thr {float(controls.throttle):5.3f}"
        )
        return self.artists


class LiveSim:
    """Drives the plant from the panel's keyboard and records every step."""

    def __init__(
        self,
        sim: SimState,
        ctl: Controller,
        targets: Targets,
        gains: Gains,
        mgains: ManualGains,
        ac: Aircraft,
        panel: Panel,
        *,
        dt: float = 0.02,
        real_time: bool = True,
        max_steps_per_frame: int = 20,
    ):
        self.sim = sim
        self.ctl = ctl
        self.targets = targets
        self.gains = gains
        self.mgains = mgains
        self.ac = ac
        self.panel = panel
        self.dt = dt
        self.real_time = real_time
        self.max_steps_per_frame = max_steps_per_frame

        self.t = 0.0
        self.controls = man.current_controls(ctl)
        self.recorder = Recorder()
        self.recorder.append(self.t, sim, self.controls, ctl.mode)
        self._backlog = 0.0
        self._last_wall = None
        # Set by run_live once FuncAnimation exists; see _retime.
        self.animation = None

    def advance(self, seconds: float) -> int:
        """Run whole physics steps until caught up. Returns how many it ran."""
        self._backlog += seconds
        steps = 0
        while self._backlog >= self.dt and steps < self.max_steps_per_frame:
            air = sense(self.sim.state, self.sim.wind_ned)
            if self.panel.take_toggle_request():
                self.ctl = man.toggle(
                    self.ctl, air, self.targets, self.gains, self.ac
                )
            self.controls, self.ctl = man.update(
                self.ctl,
                air,
                self.panel.pilot_input(),
                self.targets,
                self.gains,
                self.mgains,
                self.ac,
                self.dt,
            )
            self.sim = step(self.sim, self.controls, self.dt, self.ac)
            self.t += self.dt
            self._backlog -= self.dt
            steps += 1
            self.recorder.append(self.t, self.sim, self.controls, self.ctl.mode)
        if steps == self.max_steps_per_frame:
            # Drop the backlog. Trying to run it off makes the next frame later
            # still, and the sim ends up chasing a clock it cannot catch.
            self._backlog = 0.0
        return steps

    def _retime(self, period: float) -> None:
        """Hold the achieved render rate at `fps`.

        matplotlib's `interval` is the delay *between* frames, not the frame
        period, so the rate you actually get is `interval` plus however long the
        frame took to draw. On this panel that draw is over 20 ms -- six axes
        means six region restores and six canvas blits -- so a nominal 20 fps
        comes out at about 13. Correcting the delay from the measured period is
        what makes `fps` mean frames per second on a machine other than the one
        it was tuned on.

        It has to be `Animation._interval`: `TimedAnimation._step` rewrites
        `event_source.interval` from it after every frame, so setting the timer
        alone lasts exactly one frame. matplotlib exposes no public setter for
        the rate of a running animation. If that private name ever goes away the
        rate simply stops being corrected, which is what it does today anyway.
        """
        animation = self.animation
        if animation is None or not hasattr(animation, "_interval"):
            return
        target = 1000.0 / self.panel.fps
        animation._interval = float(
            np.clip(animation._interval + target - 1000.0 * period, 1.0, target)
        )
        if animation.event_source is not None:
            animation.event_source.interval = animation._interval

    def frame(self, _=None) -> list:
        now = time.perf_counter()
        if self._last_wall is not None:
            self._retime(now - self._last_wall)
        if self.real_time:
            elapsed = 0.0 if self._last_wall is None else now - self._last_wall
        else:
            elapsed = 1.0 / self.panel.fps
        self._last_wall = now
        self.advance(elapsed)
        return self.panel.update(self.t, self.sim, self.controls, self.ctl.mode)

    def trajectory(self) -> Trajectory:
        return self.recorder.trajectory()


def run_live(
    sim: SimState,
    ctl: Controller,
    targets: Targets,
    gains: Gains,
    mgains: ManualGains,
    ac: Aircraft,
    *,
    dt: float = 0.02,
    fps: float = 20.0,
    window: float = 60.0,
) -> Trajectory:
    """Fly interactively until the window is closed, then return the run."""
    # Warm the jit caches before the window opens. Each of these compiles on
    # first call, which is otherwise a half-second freeze on the first frame --
    # and since that frame's backlog is dropped, it comes straight out of the
    # sim's real-time budget. Both controllers are warmed, so the first press of
    # `a` does not stall either. Everything here is pure; the results are
    # discarded and `toggle` returns a new controller rather than mutating one.
    warm_air = sense(sim.state, sim.wind_ned)
    warm, _ = man.update(ctl, warm_air, man.NEUTRAL, targets, gains, mgains, ac, dt)
    other = man.toggle(ctl, warm_air, targets, gains, ac)
    man.update(other, warm_air, man.NEUTRAL, targets, gains, mgains, ac, dt)
    step(sim, warm, dt, ac)

    panel = Panel(targets, window=window, fps=fps)
    live = LiveSim(sim, ctl, targets, gains, mgains, ac, panel, dt=dt)
    animation = FuncAnimation(
        panel.fig,
        live.frame,
        init_func=panel.init,
        interval=int(1000.0 / fps),
        blit=True,
        cache_frame_data=False,
    )
    live.animation = animation
    # FuncAnimation is only weakly held by the figure; without a strong
    # reference here it is collected and the panel silently freezes.
    panel.fig._flightsim_animation = animation
    plt.show()
    return live.trajectory()


# ---------------------------------------------------------------------------
# Post-flight
# ---------------------------------------------------------------------------


def post_flight(traj: Trajectory, *, title: str = "post-flight"):
    """Ground track, altitude, airspeed/alpha/beta, deflections, mode timeline."""
    d = derived(traj)
    t = traj.t
    autopilot_engaged = traj.mode == int(Mode.AUTOPILOT)

    fig = plt.figure(figsize=(14.0, 9.0))
    fig.suptitle(title)
    grid = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.22)

    ax = fig.add_subplot(grid[0:2, 0])
    east, north = traj.pos_ned[:, 1] / 1000.0, traj.pos_ned[:, 0] / 1000.0
    ax.plot(east, north, color="0.6", lw=1.2, label="track")
    ax.plot(
        np.ma.masked_where(~autopilot_engaged, east),
        np.ma.masked_where(~autopilot_engaged, north),
        color="C0",
        lw=1.6,
        label="autopilot",
    )
    ax.plot(east[0], north[0], "o", color="C2", ms=6, label="start")
    ax.plot(east[-1], north[-1], "s", color="C3", ms=6, label="end")
    ax.set_xlabel("east  km")
    ax.set_ylabel("north  km")
    ax.set_title("ground track")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(grid[0, 1])
    ax.plot(t, d.altitude, color="C0")
    ax.set_ylabel("altitude  m")
    ax.set_title("altitude profile")
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(grid[1, 1])
    ax.plot(t, d.airspeed, color="C0", label="TAS")
    ax.set_ylabel("TAS  m/s", color="C0")
    ax.tick_params(axis="y", labelcolor="C0")
    ax.grid(alpha=0.3)
    twin = ax.twinx()
    twin.plot(t, d.alpha * RAD2DEG, color="C1", lw=1.0, label="alpha")
    twin.plot(t, d.beta * RAD2DEG, color="C3", lw=1.0, label="beta")
    twin.set_ylabel("alpha, beta  deg")
    twin.legend(fontsize=8, loc="upper right")
    ax.set_title("airspeed and incidence")

    ax = fig.add_subplot(grid[2, 0])
    for index, name in enumerate(Controls._fields[:3]):
        ax.plot(t, traj.controls[:, index] * RAD2DEG, lw=1.0, label=name)
    ax.set_ylabel("deflection  deg")
    ax.set_xlabel("time  s")
    ax.grid(alpha=0.3)
    twin = ax.twinx()
    twin.plot(t, traj.controls[:, 3], color="0.4", lw=1.0, label="throttle")
    twin.set_ylabel("throttle")
    twin.set_ylim(0.0, 1.0)
    handles = ax.get_legend_handles_labels()[0] + twin.get_legend_handles_labels()[0]
    ax.legend(handles, [h.get_label() for h in handles], fontsize=8, ncol=2)
    ax.set_title("control deflections")

    ax = fig.add_subplot(grid[2, 1])
    ax.step(t, traj.mode, where="post", color="C0", lw=1.4)
    ax.fill_between(t, 0, 1, where=autopilot_engaged, step="post", alpha=0.2, color="C0")
    ax.set_yticks([0, 1])
    ax.set_yticklabels([Mode.MANUAL.name, Mode.AUTOPILOT.name])
    ax.set_ylim(-0.2, 1.2)
    ax.set_xlabel("time  s")
    ax.set_title("mode timeline")
    ax.grid(alpha=0.3)

    return fig
