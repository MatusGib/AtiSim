"""The live instrument panel and the loop that drives it.

The layout is the basic T -- airspeed left, attitude centre, altitude right,
heading below -- with a flight-test overlay beside it. The T is not decoration:
it has been the standard arrangement since the RAF specified it in 1937, and the
point of it is that the scan is the same in every aircraft.

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

  - A TAPE works by sliding the SCALE past a fixed pointer. The value stays at
    the centre of the axes and the tick marks and their labels move, so the
    limits never have to. That is what lets a tape live on a blitted panel.
  - The rolling strips are windowed on time-before-now, so the data slides
    through fixed axes instead of the axes chasing the data. Their y-limits are
    centred on the targets, which are fixed for a run.
  - Every gauge has a fixed full scale, chosen per aircraft where the right
    scale depends on the aircraft. Values outside it clip against the edge and
    the numeric readout stays correct, so nothing is ever silently wrong, only
    off-scale.
  - The aircraft symbol and pitch ladder are animated artists despite never
    moving, because the horizon polygons are drawn over the whole axes and would
    otherwise bury anything left in the background.

**What is sensed and what is truth.** Everything on the cockpit side comes
through `sensors.sense` and `sensors.accelerometers`. The wind readout is
legitimate cockpit information -- with no sensor noise, ground velocity minus air
velocity IS the wind, so a real air-data/INS pair could compute it. The gust RATE
is not: `omega_gust` is a gradient across the span and chord, no instrument can
sense it, and it is labelled SIM TRUTH on the panel for that reason.
"""

import time
from typing import NamedTuple

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon

from flightsim import manual as man
from flightsim.aircraft import Aircraft
from flightsim.autopilot import Gains, Targets, wrap_pi
from flightsim.integrate import SimState, step
from flightsim.manual import Controller, ManualGains, Mode, PilotInput
from flightsim.sensors import AirData, Accelerations, accelerometers, sense
from flightsim.state import Controls, quat_to_dcm
from flightsim.units import RAD2DEG
from flightsim.viz import Recorder, Trajectory
from flightsim.wind import (
    PARKS_CASES,
    UPDRAFT_SECONDS,
    UPDRAFT_W0,
    UpdraftColumn,
    VortexArray,
    field_model,
    updraft_wind,
    vortex_wind,
    zero_wind,
)

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
    "]": ("trim", +1.0),  # nose up
    "[": ("trim", -1.0),  # nose down
}
TOGGLE_KEY = "a"
TRIM_HERE_KEY = "t"

HELP = (
    "arrows  stick (up = nose down)\n"
    ",  .    rudder\n"
    "-  =    throttle\n"
    "[  ]    trim (nose down / up)\n"
    "t       trim here\n"
    "a       autopilot"
)

# Instrument-panel colours.
_BG = "#101418"
_PANEL = "#161b21"
_FG = "#c8d0d8"
_SKY = "#2c6fb5"
_GROUND = "#7a5230"
_SYMBOL = "#ffd24a"
_TRACE = "#4ec9b0"
_TRUTH = "#c04a3a"
_EDGE = "#39424b"

PITCH_SPAN_DEG = 30.0  # degrees from the horizon to the top of the horizon ball
_HORIZON_L = 4.0  # half-length of the sky/ground quads, in axes units

# ---------------------------------------------------------------------------
# Declared display constants
#
# None of these come from a source. They are display choices, named here rather
# than buried in a call so that changing one is a visible decision.
# ---------------------------------------------------------------------------

# PROJECT.md section 7: "any encounter driving alpha past ~10-12 deg reports lift
# the sources say is not there". That is a statement about THIS MODEL's linear
# aero, not a stall table -- aero.py is CL = CL0 + CLa*alpha with no stall at
# all, and the only aircraft in the project with nonlinear data is out of scope.
# The band exists so that a run leaving the model's valid range says so on the
# panel, instead of in a footnote nobody reads until afterwards.
ALPHA_LINEAR_DEG = 10.0
ALPHA_INVALID_DEG = 12.0
ALPHA_SPAN_DEG = 15.0

# g, full scale of the load-factor gauge. The low end is DECLARED to clear the
# Fig. 8 load band under either reading of it (PROJECT.md section 8): read as an
# increment the band's trough is about -1.01 g absolute, read as an absolute load
# it is -2.01 g. A gauge that clips inside the band it exists to display would
# read "off scale" for the whole manoeuvring case. Was (-1.0, 3.0).
NZ_RANGE = (-2.5, 3.0)
SLIP_SPAN = 0.30  # g of lateral specific force at full ball deflection
WIND_SPAN = 40.0  # m/s at full arrow length
GUST_SPAN = 0.15  # rad/s, full scale of the gust-rate bars

# Vertical-speed full scale. Per aircraft, because the 747 at cruise trades
# altitude for speed an order faster than a light aircraft does.
VSI_SPAN: dict[str, float] = {"boeing747": 20.0, "cherokee": 10.0, "cessna172": 10.0}

# Tape scales: (full span either side of the value, gap between labelled ticks).
TAPE_AIRSPEED = (40.0, 10.0)
TAPE_ALTITUDE = (600.0, 200.0)
TAPE_HEADING = (40.0, 10.0)  # degrees


# ---------------------------------------------------------------------------
# What one frame shows
# ---------------------------------------------------------------------------


class FieldRange(NamedTuple):
    """Where the wind field is, in the terms its own geometry supports.

    `bearing` is None for a `VortexArray`, and that is not an omission. Its cores
    are infinite line vortices running east-west -- `wind.vortex_wind` reads only
    `pos_ned[0]` and `pos_ned[2]`, and the induced wind has no east component at
    all. A bearing to a line is meaningless, so none is reported rather than one
    being invented. It also means you cannot miss a core by turning.
    """

    label: str
    distance: float  # m. North distance to a line vortex; slant range to a column.
    closing: float  # m/s, rate of change of `distance`
    bearing: float | None  # rad, None for a line vortex


class Readout(NamedTuple):
    """One frame's worth of everything the panel shows.

    Exists so that every instrument has the same update signature and can be
    driven directly in a test without standing up a figure.
    """

    t: float
    air: AirData
    accel: Accelerations
    controls: Controls
    reference: Controls  # the stick's centring point, for the trim readout
    mode: Mode
    wind_ned: np.ndarray
    omega_gust: np.ndarray
    field: FieldRange | None  # None in still air


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


def _style(ax, title: str = "") -> None:
    ax.set_facecolor(_PANEL)
    ax.tick_params(colors=_FG, labelsize=7)
    for spine in ax.spines.values():
        spine.set_color(_EDGE)
    if title:
        ax.set_title(title, color=_FG, fontsize=8)


# ---------------------------------------------------------------------------
# Instruments
#
# Each one pre-allocates its artists, fixes its axis limits once, and exposes
# `artists` and `update(Readout)`. Nothing here computes physics: everything
# arrives already sensed.
# ---------------------------------------------------------------------------


class Tape:
    """A vertical scale that slides past a fixed pointer.

    Blitting forbids moving an axis limit, so the VALUE stays fixed at the centre
    and the SCALE moves: tick positions and tick label text are animated artists
    inside axes whose limits never change. A fixed number of ticks is
    pre-allocated for the same reason.
    """

    N_TICKS = 7

    def __init__(self, ax, label, span, step, fmt, source):
        self.ax, self.span, self.step, self.fmt, self.source = ax, span, step, fmt, source
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-span, span)
        ax.set_xticks([])
        ax.set_yticks([])
        _style(ax, label)
        ax.axhline(0.0, color=_EDGE, lw=0.8)
        self.ticks = [
            ax.plot([], [], color=_FG, lw=0.8, animated=True)[0]
            for _ in range(self.N_TICKS)
        ]
        # Ticks and their labels live in the right-hand half, the current-value
        # box in the left. They are kept apart rather than layered, because the
        # tick nearest the value is exactly the one the box would sit on.
        self.tick_labels = [
            ax.text(
                0.70, 0.0, "", color=_FG, fontsize=7, family="monospace",
                ha="right", va="center", animated=True,
            )
            for _ in range(self.N_TICKS)
        ]
        self.box = ax.text(
            0.30, 0.0, "", color=_SYMBOL, fontsize=9, family="monospace",
            ha="center", va="center", animated=True,
            bbox=dict(facecolor=_BG, edgecolor=_SYMBOL, boxstyle="square,pad=0.25"),
        )

    @property
    def artists(self) -> list:
        return [*self.ticks, *self.tick_labels, self.box]

    def update(self, r: Readout) -> None:
        value = self.source(r)
        centre = round(value / self.step) * self.step
        first = centre - (self.N_TICKS // 2) * self.step
        for index, (tick, text) in enumerate(zip(self.ticks, self.tick_labels)):
            mark = first + index * self.step
            offset = mark - value
            y = float(np.clip(offset, -self.span, self.span))
            # Off the end of the scale, or underneath the value box. A real PFD
            # box covers the tape it sits on; hiding that one tick is what the
            # instrument does, not a fudge to avoid overlapping text.
            visible = abs(offset) <= self.span and abs(offset) > 0.35 * self.step
            tick.set_data([0.74, 0.92] if visible else [], [y, y] if visible else [])
            text.set_position((0.70, y))
            text.set_text(self.fmt.format(mark) if visible else "")
        self.box.set_text(self.fmt.format(value))


class HeadingTape(Tape):
    """The same idea laid on its side, under the stem of the T.

    The value is unwrapped about the target so that a heading either side of
    north does not jump the width of the tape.
    """

    def __init__(self, ax, span, step, fmt, source):
        self.ax, self.span, self.step, self.fmt, self.source = ax, span, step, fmt, source
        ax.set_xlim(-span, span)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks([])
        ax.set_yticks([])
        _style(ax)
        self.ticks = [
            ax.plot([], [], color=_FG, lw=0.8, animated=True)[0]
            for _ in range(self.N_TICKS)
        ]
        self.tick_labels = [
            ax.text(
                0.0, 0.30, "", color=_FG, fontsize=7, family="monospace",
                ha="center", va="top", animated=True,
            )
            for _ in range(self.N_TICKS)
        ]
        self.box = ax.text(
            0.0, 0.78, "", color=_SYMBOL, fontsize=9, family="monospace",
            ha="center", va="center", animated=True,
            bbox=dict(facecolor=_BG, edgecolor=_SYMBOL, boxstyle="square,pad=0.25"),
        )

    def update(self, r: Readout) -> None:
        value = self.source(r)
        centre = round(value / self.step) * self.step
        first = centre - (self.N_TICKS // 2) * self.step
        for index, (tick, text) in enumerate(zip(self.ticks, self.tick_labels)):
            mark = first + index * self.step
            x = float(np.clip(mark - value, -self.span, self.span))
            visible = abs(mark - value) <= self.span
            tick.set_data([x, x] if visible else [], [0.36, 0.52] if visible else [])
            text.set_position((x, 0.30))
            text.set_text(self.fmt.format(mark % 360.0) if visible else "")
        # The box sits above the ticks here rather than on them, so nothing is
        # covered and no tick has to be suppressed.
        self.box.set_text(self.fmt.format(value % 360.0))


class VSI:
    """Vertical speed as a needle on a fixed scale. Full scale is per aircraft."""

    def __init__(self, ax, span: float):
        self.span = span
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_xticks([])
        ax.set_yticks([])
        _style(ax, "VS")
        ax.axhline(0.0, color="#5a6a78", lw=0.9)
        for fraction in (-0.5, 0.5):
            ax.axhline(fraction, color=_EDGE, lw=0.6)
        # A horizontal needle that slides up and down the scale. Drawing it as a
        # line from the centre to the value reads as a diagonal, which is not
        # what a VSI looks like and is harder to read at a glance.
        (self.needle,) = ax.plot([], [], color=_TRACE, lw=3.0, animated=True)
        self.readout = ax.text(
            0.0, -0.90, "", color=_SYMBOL, fontsize=8, family="monospace",
            ha="center", animated=True,
        )

    @property
    def artists(self) -> list:
        return [self.needle, self.readout]

    def update(self, r: Readout) -> None:
        vs = float(r.air.vertical_speed)
        y = float(np.clip(vs / self.span, -1.0, 1.0))
        self.needle.set_data([-0.7, 0.7], [y, y])
        self.readout.set_text(f"{vs:+5.1f}")


class SlipBall:
    """Lateral specific force, drawn where a PFD puts it: at the top of the ball.

    A slip ball is a bead in a curved tube, so it reads n_y and NOT beta. The two
    agree in steady coordinated flight and part company everywhere interesting,
    which in a turbulence simulator is everywhere that matters. A vane measures
    beta; a pendulum measures a force.

    Sign: the bead settles where the tube's normal force supplies the aircraft's
    specific force. The tube curves upward, so a bead displaced to +y is pushed
    back toward -y -- meaning a specific force to the RIGHT puts the ball LEFT.
    Hence the negation. It is pinned by
    test_the_ball_indicates_the_rudder_that_would_reduce_the_sideslip, because a
    derivation is not evidence.

    Simplification, stated rather than hidden: a real PFD hangs the slip index
    under the roll pointer, so it rotates with bank. This one sits at a fixed
    screen position, which is the older inclinometer presentation and is
    unambiguous to read.
    """

    def __init__(self, ax, y: float = 0.84, half_width: float = 0.15):
        self.y, self.half_width = y, half_width
        self.offset = 0.0
        (self.cage,) = ax.plot(
            [-half_width, -half_width, np.nan, half_width, half_width],
            [y - 0.05, y + 0.05, np.nan, y - 0.05, y + 0.05],
            color=_FG, lw=1.2, animated=True,
        )
        (self.ball,) = ax.plot([], [], marker="o", ms=7, color=_SYMBOL, animated=True)

    @property
    def artists(self) -> list:
        return [self.cage, self.ball]

    def update(self, r: Readout) -> None:
        self.offset = float(np.clip(-float(r.accel.n_y) / SLIP_SPAN, -1.0, 1.0))
        self.ball.set_data([self.offset * self.half_width], [self.y])


class IncidenceMarker:
    """Where the relative wind is coming from, in body axes: (-alpha, +beta).

    Deliberately NOT called a flight path vector. A HUD's FPV is earth-referenced
    and rotates with bank; this is the body-axis incidence pair the vanes
    actually measure, drawn on the same angular scale as the pitch ladder. Naming
    it after what it is costs nothing and stops it being read as something the
    sensors do not provide.
    """

    def __init__(self, ax):
        (self.marker,) = ax.plot(
            [], [], marker="o", ms=9, mfc="none", mec=_TRACE, mew=1.6, animated=True
        )
        (self.wings,) = ax.plot([], [], color=_TRACE, lw=1.6, animated=True)

    @property
    def artists(self) -> list:
        return [self.marker, self.wings]

    def update(self, r: Readout) -> None:
        x = float(r.air.beta) * RAD2DEG / PITCH_SPAN_DEG
        y = -float(r.air.alpha) * RAD2DEG / PITCH_SPAN_DEG
        x = float(np.clip(x, -0.9, 0.9))
        y = float(np.clip(y, -0.9, 0.9))
        self.marker.set_data([x], [y])
        self.wings.set_data(
            [x - 0.16, x - 0.07, np.nan, x + 0.07, x + 0.16], [y, y, np.nan, y, y]
        )


class AlphaGauge:
    """Air-relative alpha against the DECLARED linear-aero ceiling."""

    def __init__(self, ax):
        # Symmetric, because the band is a statement about |alpha|: aero.py is
        # CL0 + CLa*alpha with no stall, exactly odd-symmetric, so a pushdown to
        # -14 deg is as far outside the model as a pull-up to +14.
        ax.set_xlim(-ALPHA_SPAN_DEG, ALPHA_SPAN_DEG)
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([])
        _style(ax, "alpha  air-relative")
        ax.axvspan(-ALPHA_LINEAR_DEG, ALPHA_LINEAR_DEG, color="#2f5f42")
        for sign in (-1.0, 1.0):
            ax.axvspan(sign * ALPHA_LINEAR_DEG, sign * ALPHA_INVALID_DEG, color="#6d5423")
            ax.axvspan(sign * ALPHA_INVALID_DEG, sign * ALPHA_SPAN_DEG, color="#5f2a24")
        (self.needle,) = ax.plot([], [], color="white", lw=2.4, animated=True)
        self.readout = ax.text(
            0.03, 0.88, "", transform=ax.transAxes, color=_SYMBOL,
            fontsize=8, family="monospace", va="top", animated=True,
        )
        self._deg = 0.0

    @property
    def artists(self) -> list:
        return [self.needle, self.readout]

    def state(self) -> str:
        """Which band the model is in. `invalid` means the run proves nothing."""
        magnitude = abs(self._deg)
        if magnitude >= ALPHA_INVALID_DEG:
            return "invalid"
        if magnitude >= ALPHA_LINEAR_DEG:
            return "marginal"
        return "linear"

    def update(self, r: Readout) -> None:
        self._deg = float(r.air.alpha) * RAD2DEG
        x = float(np.clip(self._deg, -ALPHA_SPAN_DEG, ALPHA_SPAN_DEG))
        self.needle.set_data([x, x], [0.0, 1.0])
        self.readout.set_text(f"{self._deg:+5.1f} deg  {self.state()}")


class LoadFactorGauge:
    """n_z with a peak hold, because in an encounter the excursion IS the result."""

    def __init__(self, ax):
        ax.set_xlim(*NZ_RANGE)
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([])
        _style(ax, "load factor  n_z")
        ax.axvline(1.0, color="#5a6a78", lw=0.9, ls="--")
        (self.needle,) = ax.plot([], [], color=_TRACE, lw=2.4, animated=True)
        self.readout = ax.text(
            0.03, 0.88, "", transform=ax.transAxes, color=_SYMBOL,
            fontsize=8, family="monospace", va="top", animated=True,
        )
        self.peak_high = 1.0
        self.peak_low = 1.0

    @property
    def artists(self) -> list:
        return [self.needle, self.readout]

    def update(self, r: Readout) -> None:
        n_z = float(r.accel.n_z)
        self.peak_high = max(self.peak_high, n_z)
        self.peak_low = min(self.peak_low, n_z)
        x = float(np.clip(n_z, *NZ_RANGE))
        self.needle.set_data([x, x], [0.0, 1.0])
        self.readout.set_text(
            f"{n_z:+5.2f} g  peak {self.peak_low:+.2f}/{self.peak_high:+.2f}"
        )


class WindGauge:
    """Applied wind as an arrow, with the groundspeed it implies.

    Legitimate cockpit information rather than a cheat: with no sensor noise,
    ground velocity minus air velocity IS the wind, so a real air-data and
    inertial pair could compute exactly this. Screen x is east, screen y north.
    """

    def __init__(self, ax):
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.15, 1.15)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        _style(ax, "wind")
        ax.plot(
            np.cos(np.linspace(0, 2 * np.pi, 64)),
            np.sin(np.linspace(0, 2 * np.pi, 64)),
            color=_EDGE, lw=0.8,
        )
        (self.arrow,) = ax.plot([], [], color=_TRACE, lw=2.0, animated=True)
        self.readout = ax.text(
            0.0, -1.02, "", color=_SYMBOL, fontsize=8, family="monospace",
            ha="center", va="top", animated=True,
        )

    @property
    def artists(self) -> list:
        return [self.arrow, self.readout]

    def update(self, r: Readout) -> None:
        north, east = float(r.wind_ned[0]), float(r.wind_ned[1])
        speed = float(np.hypot(north, east))
        if speed > 1e-9:
            scale = min(speed / WIND_SPAN, 1.0)
            self.arrow.set_data([0.0, scale * east / speed], [0.0, scale * north / speed])
        else:
            self.arrow.set_data([], [])
        down = float(r.wind_ned[2])
        self.readout.set_text(f"{speed:4.1f} m/s  up {-down:+5.1f}")


class GustGauge:
    """The three body-axis gust rates. SIM TRUTH -- no instrument senses these.

    `omega_gust` is a gradient across the span and chord. A rate gyro measures
    the airframe's own rotation and cannot see it, which is exactly why aero.py
    is handed `omega - omega_gust` and a controller is not. Drawing it without
    saying so would present a number no aircraft has as an instrument reading.
    """

    def __init__(self, ax):
        ax.set_xlim(-GUST_SPAN, GUST_SPAN)
        ax.set_ylim(-0.6, 2.6)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["r", "q", "p"])
        _style(ax)
        ax.set_title("gust rate   SIM TRUTH", color=_TRUTH, fontsize=8)
        ax.axvline(0.0, color="#5a6a78", lw=0.9)
        self.bars = [
            ax.plot(
                [], [], color=_TRACE, lw=6.0, solid_capstyle="butt", animated=True
            )[0]
            for _ in range(3)
        ]

    @property
    def artists(self) -> list:
        return self.bars

    def update(self, r: Readout) -> None:
        # Bar 0 is the bottom row, labelled r; omega_gust is ordered (p, q, r).
        for index, bar in enumerate(self.bars):
            value = float(np.clip(r.omega_gust[2 - index], -GUST_SPAN, GUST_SPAN))
            bar.set_data([0.0, value], [index, index])


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

# How fast a hand moves a spring-centred stick: full travel in 0.4 s. A DECLARED
# figure, not a measured one. It lives here rather than in ManualGains because it
# is a property of the input device and not of the aircraft, which keeps
# manual.manual a pure function of stick position.
STICK_RATE = 2.5  # per second


class Stick:
    """Three ramped surface axes.

    Held keys ramp toward the demand and released keys spring back to centre,
    both at STICK_RATE. Without this a keyboard is a switch: a tap commands full
    travel for as long as it is held, which is why the per-aircraft authorities
    in manual.py are geared so far down.

    Stepped once per PHYSICS step, never per frame -- stepping it per frame would
    make the control feel depend on the render rate, and the render rate is
    corrected against the wall clock, so it is not even constant.

    Throttle and trim are deliberately absent. They are rate-integrating keys
    whose demand passes straight to `manual`, which integrates it; ramping them
    would put a lag on top of an integrator.
    """

    AXES = ("pitch", "roll", "yaw")

    def __init__(self) -> None:
        self.position = {axis: 0.0 for axis in self.AXES}

    def step(self, demand: dict, dt: float) -> None:
        move = STICK_RATE * dt
        for axis in self.AXES:
            delta = demand[axis] - self.position[axis]
            self.position[axis] += float(np.clip(delta, -move, move))


class Panel:
    """The figure, its pre-allocated artists, and the held-keys set.

    The panel owns the keyboard because it owns the canvas that generates the
    events. It does not own the aircraft: a key press only updates a set, and
    `LiveSim` asks for the resulting stick position once per physics step.
    """

    def __init__(
        self,
        targets: Targets,
        *,
        window: float = 60.0,
        fps: float = 20.0,
        aircraft_name: str = "boeing747",
    ):
        self.targets = targets
        self.window = window
        self.fps = fps
        self.aircraft_name = aircraft_name
        self.held: set[str] = set()
        self.stick = Stick()
        self._toggle_requested = False
        self._trim_here_requested = False

        target_speed = float(targets.airspeed)
        target_altitude = float(targets.altitude)
        target_heading = float(targets.heading) * RAD2DEG

        n = int(window * fps) + 1
        self._t = np.full(n, np.nan)
        self._speed = np.full(n, np.nan)
        self._altitude = np.full(n, np.nan)
        self._heading = np.full(n, np.nan)

        self.fig = plt.figure(figsize=(15.0, 8.5), facecolor=_BG)
        if getattr(self.fig.canvas, "manager", None) is not None:
            self.fig.canvas.manager.set_window_title("flightsim")
        grid = self.fig.add_gridspec(
            3,
            6,
            width_ratios=(0.30, 1.05, 0.30, 0.17, 0.70, 0.92),
            height_ratios=(1.00, 0.16, 0.22),
            hspace=0.40,
            wspace=0.42,
            left=0.035,
            right=0.965,
            top=0.94,
            bottom=0.04,
        )

        span, tick = TAPE_AIRSPEED
        self.asi = Tape(
            self.fig.add_subplot(grid[0, 0]), "TAS m/s", span, tick, "{:.0f}",
            lambda r: float(r.air.airspeed),
        )
        self._build_horizon(grid[0, 1])
        span, tick = TAPE_ALTITUDE
        self.alt = Tape(
            self.fig.add_subplot(grid[0, 2]), "ALT m", span, tick, "{:.0f}",
            lambda r: float(r.air.altitude),
        )
        self.vsi = VSI(self.fig.add_subplot(grid[0, 3]), VSI_SPAN[aircraft_name])
        span, tick = TAPE_HEADING
        self.hdg = HeadingTape(
            self.fig.add_subplot(grid[1, 0:4]), span, tick, "{:03.0f}",
            self._unwrapped_heading,
        )

        overlay = grid[0:2, 4].subgridspec(4, 1, hspace=0.60)
        self.nz_gauge = LoadFactorGauge(self.fig.add_subplot(overlay[0]))
        self.alpha_gauge = AlphaGauge(self.fig.add_subplot(overlay[1]))
        self.wind_gauge = WindGauge(self.fig.add_subplot(overlay[2]))
        self.gust_gauge = GustGauge(self.fig.add_subplot(overlay[3]))

        cells = grid[0:2, 5].subgridspec(3, 1, hspace=0.38)
        self.strips = [
            self._build_strip(cells[0], "TAS  m/s", target_speed, 40.0),
            self._build_strip(cells[1], "ALT  m", target_altitude, 600.0),
            self._build_strip(cells[2], "HDG  deg", target_heading, 90.0, xlabel=True),
        ]
        self._build_status(grid)
        self._connect_keys()

        self.instruments = [
            self.asi, self.alt, self.vsi, self.hdg, self.slip, self.incidence,
            self.nz_gauge, self.alpha_gauge, self.wind_gauge, self.gust_gauge,
        ]

    # -- construction ------------------------------------------------------

    def _unwrapped_heading(self, r: Readout) -> float:
        """Heading unwrapped about the target, so either side of north is smooth."""
        target = float(self.targets.heading)
        return (target + float(wrap_pi(jnp.array(float(r.air.psi) - target)))) * RAD2DEG

    def _build_horizon(self, cell) -> None:
        self.ax_horizon = self.fig.add_subplot(cell)
        ax = self.ax_horizon
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor(_BG)
        for spine in ax.spines.values():
            spine.set_color(_EDGE)
        ax.set_title("attitude", color=_FG, fontsize=9)

        self.sky = Polygon(np.zeros((4, 2)), closed=True, color=_SKY, animated=True)
        self.ground = Polygon(
            np.zeros((4, 2)), closed=True, color=_GROUND, animated=True
        )
        ax.add_patch(self.sky)
        ax.add_patch(self.ground)
        (self.horizon_line,) = ax.plot([], [], color="white", lw=1.6, animated=True)
        (self.ladder,) = ax.plot([], [], color="white", lw=1.0, animated=True)
        self.incidence = IncidenceMarker(ax)
        (self.symbol,) = ax.plot(
            _SYMBOL_X, _SYMBOL_Y, color=_SYMBOL, lw=2.4, animated=True
        )
        self.slip = SlipBall(ax)

    def _build_strip(
        self, cell, label: str, centre: float, span: float, *, xlabel: bool = False
    ):
        ax = self.fig.add_subplot(cell)
        _style(ax)
        ax.set_xlim(-self.window, 0.0)
        ax.set_ylim(centre - span, centre + span)
        ax.set_xticks([-self.window, -self.window / 2.0, 0.0])
        # Labels and ticks on the RIGHT: the strips sit at the right-hand edge of
        # the figure, so a left-hand label lands on top of the overlay gauges.
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        ax.set_ylabel(label, color=_FG, fontsize=8)
        ax.axhline(centre, color="#5a6a78", lw=0.9, ls="--")  # target, background
        ax.grid(color="#232b33", lw=0.6)
        if xlabel:
            ax.set_xlabel("seconds before now", color=_FG, fontsize=8)
        (line,) = ax.plot([], [], color=_TRACE, lw=1.3, animated=True)
        readout = ax.text(
            0.02, 0.90, "", transform=ax.transAxes, color=_SYMBOL, fontsize=9,
            family="monospace", va="top", animated=True,
        )
        return line, readout, centre

    def _build_status(self, grid) -> None:
        ax = self.fig.add_subplot(grid[2, 0:5])
        ax.set_axis_off()
        ax.set_facecolor(_BG)
        self.status = ax.text(
            0.0, 0.98, "", transform=ax.transAxes, color=_FG, fontsize=10,
            family="monospace", va="top", animated=True,
        )
        help_ax = self.fig.add_subplot(grid[2, 5])
        help_ax.set_axis_off()
        help_ax.set_facecolor(_BG)
        help_ax.text(
            0.0, 0.98, HELP, transform=help_ax.transAxes, color="#7d8a96",
            fontsize=8, family="monospace", va="top",
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
        if event.key == TRIM_HERE_KEY:
            self._trim_here_requested = True
        self.held.add(event.key)

    def on_release(self, event) -> None:
        self.held.discard(event.key)

    def take_toggle_request(self) -> bool:
        """Edge-triggered: a press is consumed once, however long the key is held."""
        requested, self._toggle_requested = self._toggle_requested, False
        return requested

    def take_trim_here_request(self) -> bool:
        """Edge-triggered, exactly as the autopilot toggle is."""
        requested, self._trim_here_requested = self._trim_here_requested, False
        return requested

    def key_demand(self) -> dict:
        """Where the keys are asking each axis to go, BEFORE the stick ramp."""
        axes = {"pitch": 0.0, "roll": 0.0, "yaw": 0.0, "throttle": 0.0, "trim": 0.0}
        for key in self.held:
            if key in KEYMAP:
                axis, sign = KEYMAP[key]
                axes[axis] += sign
        # Opposite keys held together cancel; the clip is for a stuck repeat.
        return {k: float(np.clip(v, -1.0, 1.0)) for k, v in axes.items()}

    def step_stick(self, dt: float) -> None:
        """Advance the ramp by one PHYSICS step."""
        self.stick.step(self.key_demand(), dt)

    def pilot_input(self) -> PilotInput:
        """The stick as the aircraft sees it: ramped surfaces, raw rate keys."""
        demand = self.key_demand()
        return PilotInput(
            pitch=self.stick.position["pitch"],
            roll=self.stick.position["roll"],
            yaw=self.stick.position["yaw"],
            throttle=demand["throttle"],
            trim=demand["trim"],
        )

    # -- drawing -----------------------------------------------------------

    @property
    def artists(self) -> list:
        # Order matters under blitting: within the horizon axes the artists are
        # drawn in this order, so the polygons must come before anything on top.
        return [
            self.sky,
            self.ground,
            self.horizon_line,
            self.ladder,
            *self.incidence.artists,
            self.symbol,
            *self.slip.artists,
            *self.asi.artists,
            *self.alt.artists,
            *self.vsi.artists,
            *self.hdg.artists,
            *self.nz_gauge.artists,
            *self.alpha_gauge.artists,
            *self.wind_gauge.artists,
            *self.gust_gauge.artists,
            self.status,
            *[artist for line, readout, _ in self.strips for artist in (line, readout)],
        ]

    def init(self) -> list:
        return self.artists

    def update(
        self,
        t: float,
        sim: SimState,
        controls: Controls,
        mode: Mode,
        accel: Accelerations,
        reference: Controls,
        field: FieldRange | None = None,
    ) -> list:
        # Sensed from the wind the last step applied, so the readouts are
        # air-relative under a wind field rather than quietly ground-relative.
        air = sense(sim.state, sim.wind_ned)
        r = Readout(
            t=t,
            air=air,
            accel=accel,
            controls=controls,
            reference=reference,
            mode=mode,
            wind_ned=np.asarray(sim.wind_ned, dtype=float),
            omega_gust=np.asarray(sim.omega_gust, dtype=float),
            field=field,
        )

        for instrument in self.instruments:
            instrument.update(r)
        self._draw_horizon(r)
        self._draw_strips(r)
        self._draw_status(r)
        return self.artists

    def _draw_horizon(self, r: Readout) -> None:
        theta_deg = float(r.air.theta) * RAD2DEG
        centre, along, sky = _horizon_frame(float(r.air.phi), theta_deg)
        self.sky.set_xy(_quad(centre, along, sky, _HORIZON_L, 2.0 * _HORIZON_L))
        self.ground.set_xy(_quad(centre, along, sky, _HORIZON_L, -2.0 * _HORIZON_L))
        self.horizon_line.set_data(
            [centre[0] - _HORIZON_L * along[0], centre[0] + _HORIZON_L * along[0]],
            [centre[1] - _HORIZON_L * along[1], centre[1] + _HORIZON_L * along[1]],
        )
        self.ladder.set_data(*_ladder(centre, along, sky, theta_deg))

    def _draw_strips(self, r: Readout) -> None:
        _push(self._t, r.t)
        _push(self._speed, float(r.air.airspeed))
        _push(self._altitude, float(r.air.altitude))
        _push(self._heading, self._unwrapped_heading(r))

        elapsed = self._t - r.t
        for (line, readout, _), values, fmt in zip(
            self.strips,
            (self._speed, self._altitude, self._heading),
            ("{:8.1f}", "{:8.0f}", "{:8.1f}"),
        ):
            line.set_data(elapsed, values)
            readout.set_text(fmt.format(values[-1]))

    def _draw_status(self, r: Readout) -> None:
        if r.field is None:
            field_line = "still air"
        elif r.field.bearing is None:
            field_line = (
                f"{r.field.label}   {r.field.distance:7.0f} m   "
                f"closing {r.field.closing:+5.0f} m/s"
            )
        else:
            field_line = (
                f"{r.field.label}   {r.field.distance:7.0f} m   "
                f"brg {np.degrees(r.field.bearing) % 360.0:5.1f}"
            )
        self.status.set_text(
            f"{Mode(r.mode).name:<10s} t {r.t:7.1f} s     {field_line}\n"
            f"TAS {float(r.air.airspeed):7.1f} m/s   "
            f"ALT {float(r.air.altitude):8.0f} m   "
            f"VS {float(r.air.vertical_speed):+6.1f} m/s   "
            f"HDG {np.degrees(float(r.air.psi)) % 360.0:5.1f} deg\n"
            f"pitch {float(r.air.theta) * RAD2DEG:+6.1f}   "
            f"bank {float(r.air.phi) * RAD2DEG:+6.1f}   "
            f"aoa {float(r.air.alpha) * RAD2DEG:+5.1f}   "
            f"beta {float(r.air.beta) * RAD2DEG:+5.1f}   "
            f"n_z {float(r.accel.n_z):+5.2f}\n"
            f"elev {float(r.controls.elevator) * RAD2DEG:+6.2f}   "
            f"ail {float(r.controls.aileron) * RAD2DEG:+6.2f}   "
            f"rud {float(r.controls.rudder) * RAD2DEG:+6.2f}   "
            f"thr {float(r.controls.throttle):5.3f}   "
            f"trim {float(r.reference.elevator) * RAD2DEG:+6.2f}"
        )


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
        wind_model=zero_wind,
        field_range=None,
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
        self.wind_model = wind_model
        # An optional `State -> FieldRange`, built by the caller alongside the
        # field. Kept separate from the wind model because the panel needs the
        # geometry in a form the model does not carry, and a model with no
        # meaningful "where is it" -- a future Dryden layer -- can pass None
        # rather than being forced to invent one.
        self.field_range = field_range

        self.t = 0.0
        self.controls = man.current_controls(ctl)
        self.recorder = Recorder()
        self.recorder.append(self.t, sim, self.controls, ctl.mode)
        self._backlog = 0.0
        self._last_wall = None
        # Set by run_live once FuncAnimation exists; see _retime.
        self.animation = None

    def request_trim_here(self) -> None:
        """Ask for a trim-here on the next physics step, as the key would."""
        self.panel._trim_here_requested = True

    def advance(self, seconds: float) -> int:
        """Run whole physics steps until caught up. Returns how many it ran."""
        self._backlog += seconds
        steps = 0
        while self._backlog >= self.dt and steps < self.max_steps_per_frame:
            air = sense(self.sim.state, self.sim.wind_ned)
            if self.panel.take_toggle_request():
                self.ctl = man.toggle(self.ctl, air, self.targets, self.gains, self.ac)
            if self.panel.take_trim_here_request():
                self.ctl = man.trim_here(self.ctl)
            # Ramp the stick on the PHYSICS clock, not the render clock.
            self.panel.step_stick(self.dt)
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
            self.sim = step(self.sim, self.controls, self.dt, self.ac, self.wind_model)
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
        frame took to draw. On this panel that draw is over 20 ms -- a dozen axes
        means a dozen region restores and a dozen canvas blits -- so a nominal
        20 fps comes out well short. Correcting the delay from the measured
        period is what makes `fps` mean frames per second on a machine other than
        the one it was tuned on.

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
        accel = accelerometers(
            self.sim.state, self.controls, self.ac,
            self.sim.wind_ned, self.sim.omega_gust,
        )
        field = None if self.field_range is None else self.field_range(self.sim.state)
        return self.panel.update(
            self.t, self.sim, self.controls, self.ctl.mode,
            accel, self.ctl.manual.reference, field,
        )

    def trajectory(self) -> Trajectory:
        return self.recorder.trajectory()


# ---------------------------------------------------------------------------
# Where a wind field is, in the terms its own geometry supports
# ---------------------------------------------------------------------------


def vortex_range(array, *, label: str):
    """Distance to the next VortexArray core ahead, as a `State -> FieldRange`.

    The cores are infinite line vortices running east-west: `wind.vortex_wind`
    reads only `pos_ned[0]` and `pos_ned[2]`, and the induced wind has no east
    component. So the distance is a perpendicular north distance whatever the
    heading, and there is no bearing -- you cannot miss a line by turning away
    from it.
    """
    cores = np.sort(np.asarray(array.north, dtype=float))

    def ranged(state) -> FieldRange:
        north = float(state.pos_ned[0])
        ahead = cores[cores >= north - 1e-9]
        target = float(ahead[0]) if len(ahead) else float(cores[-1])
        index = int(np.searchsorted(cores, target)) + 1
        vel_ned = np.asarray(quat_to_dcm(state.quat) @ state.vel_body, dtype=float)
        return FieldRange(
            label=f"{label} core {index}",
            distance=target - north,
            closing=float(vel_ned[0]),
            bearing=None,
        )

    return ranged


def updraft_range(column, *, label: str):
    """Range and bearing to an UpdraftColumn, which IS a point and so has both."""
    north0, east0 = float(column.north), float(column.east)

    def ranged(state) -> FieldRange:
        north, east = float(state.pos_ned[0]), float(state.pos_ned[1])
        dn, de = north0 - north, east0 - east
        distance = float(np.hypot(dn, de))
        vel_ned = np.asarray(quat_to_dcm(state.quat) @ state.vel_body, dtype=float)
        closing = (
            float((dn * vel_ned[0] + de * vel_ned[1]) / distance)
            if distance > 1e-9
            else 0.0
        )
        return FieldRange(
            label=label,
            distance=distance,
            closing=closing,
            bearing=float(np.arctan2(de, dn)),
        )

    return ranged


def field_ahead(
    name: str,
    *,
    airspeed: float,
    altitude: float,
    lead_in: float = 40.0,
    sharpness: float = 6.0,
):
    """Place a cited wind field ahead of the origin: `(wind_model, range, note)`.

    The FIELD is offset rather than the aircraft being started behind it, which
    is the other way round from `scripts/vortex.py`. Physically identical --
    only the relative geometry enters a position-only field -- but every run
    still begins at the origin, so ground tracks and saved `.npz` files stay
    comparable and nothing about `trimmed_state` or `init_sim` changes.

    Placement and the range callable come out of the same object on purpose. Two
    functions each deciding for themselves where the field is, is how a readout
    ends up confidently pointing at somewhere the aircraft is not.

    `lead_in` is in core radii, matching vortex.py. Below about 12 the 1/r far
    field launches the aircraft out of equilibrium (PROJECT.md section 9,
    session 3). `sharpness` is a DECLARED modelling parameter, not source data.
    """
    if name == "none":
        return zero_wind, None, "still air"

    if name in PARKS_CASES:
        case = PARKS_CASES[name]
        lead = lead_in * case["r0"]
        array = VortexArray(
            north=jnp.array([lead, lead + case["spacing"]]),
            down=jnp.array([-altitude, -altitude]),
            r0=jnp.array(case["r0"]),
            v0=jnp.array(case["v0"]),
        )
        note = (
            f"vortex array ({name}): r0 {case['r0']:.0f} m, V0 {case['v0']:.1f} m/s, "
            f"first core {lead:.0f} m ahead ({lead / airspeed:.0f} s at cruise)"
        )
        return (
            field_model(lambda p: vortex_wind(p, array)),
            vortex_range(array, label=f"vortex {name}"),
            note,
        )

    if name == "updraft":
        # The paper's 20 s is a TRAVERSE time, so it fixes a diameter only once
        # a flight speed is chosen -- hence the radius depending on airspeed.
        radius = 0.5 * UPDRAFT_SECONDS * airspeed
        # `lead_in` cannot mean the same thing here as it does for a vortex. A
        # column radius is over a dozen vortex core radii, so 40 of them would be
        # 90 km -- six minutes of flying before anything happened. It is read as
        # tenths of a radius beyond the edge instead, which puts the default 40
        # four radii clear. Stated rather than hidden in the arithmetic.
        lead = radius * (1.0 + lead_in / 10.0)
        column = UpdraftColumn(
            north=jnp.array(lead),
            east=jnp.array(0.0),
            w0=jnp.array(UPDRAFT_W0),
            radius=jnp.array(radius),
            sharpness=jnp.array(sharpness),
        )
        note = (
            f"updraft column: w0 {UPDRAFT_W0:.1f} m/s, radius {radius:.0f} m, "
            f"sharpness {sharpness:.1f} (DECLARED), centre {lead:.0f} m ahead"
        )
        return (
            field_model(lambda p: updraft_wind(p, column)),
            updraft_range(column, label="updraft column"),
            note,
        )

    raise ValueError(f"unknown wind field {name!r}")


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
    wind_model=zero_wind,
    field_range=None,
    aircraft_name: str = "boeing747",
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
    # Warmed with the ACTUAL wind model, not the default. `step` takes
    # wind_model as a static argument, so a different model is a different
    # compilation -- warming zero_wind here would leave the real one to compile
    # inside the first frame, which is precisely what this warm-up exists to
    # prevent.
    step(sim, warm, dt, ac, wind_model)
    accelerometers(sim.state, warm, ac, sim.wind_ned, sim.omega_gust)

    panel = Panel(targets, window=window, fps=fps, aircraft_name=aircraft_name)
    live = LiveSim(
        sim, ctl, targets, gains, mgains, ac, panel,
        dt=dt, wind_model=wind_model, field_range=field_range,
    )
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
