"""Fly a test point from a browser: the live loop of `panel`, with no figure.

The loop is the one the matplotlib panel flies (`scripts/fly.py`), unchanged:
whole 50 Hz physics steps catching up with the wall clock, the stick ramped per
physics step, the autopilot and trim-here keys edge-triggered (`panel.LiveSim`).
Only the two ends differ. The held keys arrive from the app's cockpit page
(`panel.Keys`) instead of a figure's key events, and each poll returns one
frame of plain numbers for the page to draw instead of updating artists.

The test points are the fields `panel.field_ahead` places ahead of the
aircraft. Every one is flown by the 747 at its cruise condition, the aircraft
the Parks and Wingrove cases are valid for (a 747/DC-10-class case). Parks'
two vortex pairs were identified from DC-10 flight records; the model flying
them here is the 747, and the card says so.

Nothing here needs the `ui` extra: a test flies a `Flight` directly.
"""

import threading
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from atisim import autopilot as ap_mod
from atisim import checks, integrate, manual as man, panel, trim
from atisim.aircraft import CRUISE, REGISTRY
from atisim.manual import Mode
from atisim.sensors import accelerometers, sense
from atisim.state import State, quat_to_euler
from atisim.units import RAD2DEG
from atisim.wind import PARKS_CASES, UPDRAFT_SECONDS, UPDRAFT_W0

DT = 0.02  # s, the physics step scripts/fly.py flies
AIRCRAFT = "boeing747"


class TestPoint(NamedTuple):
    key: str
    title: str
    where: str  # where and when the source met it, or "" for still air
    summary: str  # one plain sentence for someone who does not know the code
    field: str  # a `panel.field_ahead` name
    source: str


_H = PARKS_CASES["hannibal"]
_M = PARKS_CASES["morton"]
TEST_POINTS: dict[str, TestPoint] = {
    tp.key: tp for tp in (
        TestPoint(
            "calm", "Calm air", "",
            "Still air at cruise. Get the feel of the controls, and try the autopilot.",
            "none", "no wind field",
        ),
        TestPoint(
            "hannibal", "Vortex pair, Hannibal",
            "Hannibal, Missouri, 3 April 1981, 37,000 ft",
            f"Two rolling air vortices {_H['r0'] * 2:.0f} m across, "
            f"{_H['spacing']:.0f} m apart. A DC-10 met them in severe turbulence.",
            "hannibal",
            "Parks, Wingrove, Bach & Mehta 1985, J. Aircraft 22(2) 124-129, case 1",
        ),
        TestPoint(
            "morton", "Vortex pair, Morton",
            "Morton, Wyoming, 16 July 1982, 39,000 ft",
            f"A smaller, tighter pair: {_M['r0'] * 2:.0f} m across, "
            f"{_M['spacing']:.0f} m apart. Also a DC-10, also severe.",
            "morton",
            "Parks, Wingrove, Bach & Mehta 1985, J. Aircraft 22(2) 124-129, case 2",
        ),
        TestPoint(
            "updraft", "Thunderstorm updraft",
            "near Bermuda, 12 October 1983",
            f"A column of air rising at {UPDRAFT_W0:.0f} m/s that takes "
            f"{UPDRAFT_SECONDS:.0f} s to cross. How sharp its edges are is a "
            "declared choice, not source data.",
            "updraft",
            "Wingrove & Bach 1994, J. Aircraft 31(4) 753-760, p. 756",
        ),
    )
}

# The page's scales, from the panel's own declared display constants, so the
# browser and the matplotlib panel show the same thing.
LIMITS = {
    "alpha_linear_deg": panel.ALPHA_LINEAR_DEG,
    "alpha_invalid_deg": panel.ALPHA_INVALID_DEG,
    "nz_range": list(panel.NZ_RANGE),
    "vsi_span": panel.VSI_SPAN[AIRCRAFT],
    "wind_span": panel.WIND_SPAN,
    "slip_span": panel.SLIP_SPAN,
    "gust_span": panel.GUST_SPAN,
}

# One wind model per test point for the life of the process. `integrate.step`
# takes the model as a static argument, so a fresh closure per flight would
# compile the step again on every flight.
_fields: dict[str, tuple] = {}
_fields_lock = threading.Lock()


def _cruise():
    return CRUISE[AIRCRAFT]["airspeed"], CRUISE[AIRCRAFT]["altitude"]


def _field(key: str):
    with _fields_lock:
        if key not in _fields:
            airspeed, altitude = _cruise()
            _fields[key] = panel.field_ahead(
                TEST_POINTS[key].field, airspeed=airspeed, altitude=altitude)
        return _fields[key]


def gust_profile(key: str, seconds: float = 70.0, n: int = 281) -> dict:
    """The vertical wind a straight, level flight at cruise meets, against time.

    Evaluated from the placed field itself, so the card's sketch of a test
    point is the field the flight will meet, not a drawing of it. Up is
    positive. The aircraft's own response is not in it: nothing is flown.
    """
    airspeed, altitude = _cruise()
    wind_model = _field(key)[0]
    t = np.linspace(0.0, seconds, n)
    level = State(
        pos_ned=jnp.zeros(3), vel_body=jnp.array([airspeed, 0.0, 0.0]),
        quat=jnp.array([1.0, 0.0, 0.0, 0.0]), omega=jnp.zeros(3),
    )

    def up(north):
        state = level._replace(pos_ned=jnp.array([north, 0.0, -altitude]))
        return -wind_model(None, state, None, DT)[0][2]

    w = jax.vmap(up)(jnp.asarray(t * airspeed))
    return {"t": t.round(3).tolist(), "w_up": np.asarray(w).round(3).tolist()}


class Flight:
    """One test point, flown from held keys. Thread-safe: one poll at a time."""

    def __init__(self, key: str, *, clock=time.perf_counter):
        self.test_point = TEST_POINTS[key]
        self.ac = REGISTRY[AIRCRAFT]
        gains, mgains = ap_mod.GAINS[AIRCRAFT], man.MANUAL_GAINS[AIRCRAFT]
        airspeed, altitude = _cruise()
        x, _ = trim.trim(jnp.array(airspeed), jnp.array(altitude), self.ac)
        state = trim.trimmed_state(x[0], jnp.array(airspeed), jnp.array(altitude))
        controls = trim.trimmed_controls(x[1], x[2])
        self.targets = ap_mod.Targets(
            altitude=jnp.array(altitude), heading=jnp.array(0.0),
            airspeed=jnp.array(airspeed))
        ctl = man.start(sense(state), controls, self.targets, gains, self.ac)
        sim = integrate.init_sim(state, jax.random.PRNGKey(0))
        wind_model, field_range, self.note = _field(key)
        panel.warm_up(sim, ctl, self.targets, gains, mgains, self.ac,
                      dt=DT, wind_model=wind_model)
        self.keys = panel.Keys()
        self.live = panel.LiveSim(
            sim, ctl, self.targets, gains, mgains, self.ac, self.keys,
            dt=DT, wind_model=wind_model, field_range=field_range)
        self.lock = threading.Lock()
        self._clock = clock
        self._last = None  # None: paused, or not started
        self._presses = {"a": 0, "t": 0}

    def info(self) -> dict:
        """What does not change during the flight."""
        tp = self.test_point
        return {
            "test_point": tp._asdict(),
            "aircraft": AIRCRAFT,
            "note": self.note,
            "dt": DT,
            "targets": {"airspeed": float(self.targets.airspeed),
                        "altitude": float(self.targets.altitude),
                        "heading": float(self.targets.heading) * RAD2DEG},
            "limits": LIMITS,
        }

    def pause(self) -> None:
        """Stop the clock: the next poll starts it again without a jump."""
        with self.lock:
            self._last = None

    def poll(self, held=(), presses=None) -> dict:
        """Fly up to now with these keys held, then return one frame.

        `presses` counts each edge-triggered key ("a", "t") since the flight
        began. A count is idempotent where a press event is not: a poll that is
        sent twice, or lost, cannot toggle the autopilot twice or not at all.
        """
        with self.lock:
            self.keys.held = {k for k in held if k in panel.KEYMAP}
            for key, count in (presses or {}).items():
                if key in self._presses and int(count) > self._presses[key]:
                    self._presses[key] = int(count)
                    self.keys.press(key)
            now = self._clock()
            self.live.advance(0.0 if self._last is None else now - self._last)
            self._last = now
            return self.frame()

    def frame(self) -> dict:
        live = self.live
        sim = live.sim
        air = sense(sim.state, sim.wind_ned)
        acc = accelerometers(sim.state, live.controls, self.ac, sim.wind_ned,
                             sim.omega_gust)
        field = None if live.field_range is None else live.field_range(sim.state)
        wind = np.asarray(sim.wind_ned, dtype=float)
        return {
            "t": round(live.t, 3),
            "mode": Mode(live.ctl.mode).name.lower(),
            "airspeed": float(air.airspeed),
            "altitude": float(air.altitude),
            "vs": float(air.vertical_speed),
            "heading": float(air.psi) * RAD2DEG % 360.0,
            "pitch": float(air.theta) * RAD2DEG,
            "bank": float(air.phi) * RAD2DEG,
            "alpha": float(air.alpha) * RAD2DEG,
            "beta": float(air.beta) * RAD2DEG,
            "nz": float(acc.n_z),
            "ny": float(acc.n_y),
            "wind": {"north": wind[0], "east": wind[1], "up": -wind[2]},
            "gust": np.asarray(sim.omega_gust, dtype=float).tolist(),
            "stick": dict(self.keys.stick.position),
            "throttle": float(live.controls.throttle),
            "trim": float(live.ctl.manual.reference.elevator) * RAD2DEG,
            "field": None if field is None else {
                "label": field.label, "distance": field.distance,
                "closing": field.closing,
                "bearing": None if field.bearing is None
                else float(np.degrees(field.bearing)) % 360.0,
            },
        }

    def summary(self, max_points: int = 4000) -> dict:
        """The debrief: what the flight did, from every physics step."""
        with self.lock:
            traj = self.live.trajectory()
        t = np.asarray(traj.t)
        nz = checks.load_factor_series(traj, self.ac)
        euler = np.asarray(jax.vmap(quat_to_euler)(jnp.asarray(traj.quat))) * RAD2DEG
        altitude = -np.asarray(traj.pos_ned[:, 2])
        states = State(pos_ned=jnp.asarray(traj.pos_ned),
                       vel_body=jnp.asarray(traj.vel_body),
                       quat=jnp.asarray(traj.quat), omega=jnp.asarray(traj.omega))
        alpha = np.asarray(jax.vmap(sense)(states, jnp.asarray(traj.wind_ned)).alpha)
        alpha = alpha * RAD2DEG
        hi, lo = int(nz.argmax()), int(nz.argmin())
        a_peak = int(np.abs(alpha).argmax())
        severity = checks.rms_normal_load(traj, self.ac, dt=DT)
        keep = np.unique(np.concatenate((
            np.arange(0, t.size, max(1, -(-t.size // max_points))), [hi, lo, t.size - 1])))
        return {
            "duration": float(t[-1]),
            "nz_max": {"value": float(nz[hi]), "t": float(t[hi])},
            "nz_min": {"value": float(nz[lo]), "t": float(t[lo])},
            "bank_max": float(np.abs(euler[:, 0]).max()),
            "altitude_change": {"min": float(altitude.min() - altitude[0]),
                                "max": float(altitude.max() - altitude[0]),
                                "end": float(altitude[-1] - altitude[0])},
            "alpha_max": {"value": float(alpha[a_peak]), "t": float(t[a_peak]),
                          "band": _alpha_band(float(abs(alpha[a_peak])))},
            "severity": severity.as_dict(),
            "severity_band": checks.severity_band(severity.value),
            "recovery": checks.recovery_band(traj, self.ac).as_dict(),
            "events": self._events(traj),
            "series": {
                "t": t[keep].round(3).tolist(),
                "nz": nz[keep].round(4).tolist(),
                "altitude": (altitude[keep] - altitude[0]).round(2).tolist(),
                "bank": euler[keep, 0].round(2).tolist(),
                "pitch": euler[keep, 1].round(2).tolist(),
                "autopilot": (np.asarray(traj.mode)[keep] == Mode.AUTOPILOT).tolist(),
            },
        }

    def _events(self, traj) -> list[dict]:
        """Where the flight met the field, sample-exact.

        A vortex core is passed where the next-core label changes, or, for the
        last core, where its north distance goes through zero. An updraft
        column is a point, so its event is the closest approach.
        """
        field_range = self.live.field_range
        if field_range is None:
            return []
        t = np.asarray(traj.t)
        cache: dict[int, panel.FieldRange] = {}

        def at(i):  # field_range is a Python call per sample: evaluate sparingly
            if i not in cache:
                cache[i] = field_range(State(
                    pos_ned=traj.pos_ned[i], vel_body=traj.vel_body[i],
                    quat=traj.quat[i], omega=traj.omega[i]))
            return cache[i]

        # A coarse pass every STRIDE samples, then every sample inside the one
        # stride where something happened. Nothing in these fields changes
        # twice within 0.2 s at cruise speed: a core is hundreds of metres wide.
        stride = 10
        coarse = list(range(0, t.size, stride)) + [t.size - 1]
        if at(0).bearing is not None:
            i = min(coarse, key=lambda k: at(k).distance)
            i = min(range(max(0, i - stride), min(t.size, i + stride + 1)),
                    key=lambda k: at(k).distance)
            if i in (0, t.size - 1):
                return []  # not reached, or still closing when the flight ended
            return [{"t": float(t[i]), "label": "closest to the column centre"}]

        def passed(i, j):
            before, now = at(i), at(j)
            return now.label != before.label or before.distance >= 0.0 > now.distance

        events = []
        for i, j in zip(coarse, coarse[1:]):
            if passed(i, j):
                for k in range(i + 1, j + 1):
                    if passed(k - 1, k):
                        events.append({"t": float(t[k]),
                                       "label": f"passed {at(k - 1).label}"})
        return events


def _alpha_band(alpha_deg: float) -> str:
    """The panel's alpha gauge words: where the linear aerodynamics hold."""
    if alpha_deg < panel.ALPHA_LINEAR_DEG:
        return "linear"
    return "marginal" if alpha_deg < panel.ALPHA_INVALID_DEG else "invalid"
