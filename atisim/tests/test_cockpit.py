"""The browser cockpit's engine: `cockpit.Flight` flies `panel.LiveSim` from the
held keys a page sends. No browser and no Dash: a fake clock drives the flight.
"""

import json
import math

import numpy as np
import pytest

from atisim import cockpit
from atisim.aircraft import CRUISE
from atisim.wind import PARKS_CASES

V = CRUISE[cockpit.AIRCRAFT]["airspeed"]


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def fly(key="calm"):
    clock = Clock()
    return cockpit.Flight(key, clock=clock), clock


def fly_to(flight, clock, t_end, step=0.4, **poll):
    """Poll every `step` seconds of wall clock until the flight reaches t_end."""
    frame = flight.poll(**poll)
    while frame["t"] < t_end - 1e-9:
        clock.now += step
        frame = flight.poll(**poll)
    return frame


def test_a_poll_flies_whole_physics_steps_up_to_the_clock():
    flight, clock = fly()
    assert flight.poll()["t"] == 0.0  # the first poll starts the clock
    clock.now = 0.101  # a whole step is flown only once its 20 ms have passed
    assert flight.poll()["t"] == pytest.approx(0.1)  # five 20 ms steps


def test_a_pause_stops_the_clock_so_the_next_poll_does_not_jump():
    flight, clock = fly()
    flight.poll()
    clock.now = 0.101
    flight.poll()
    flight.pause()
    clock.now = 60.0  # a minute on the pause screen
    assert flight.poll()["t"] == pytest.approx(0.1)
    clock.now = 60.04
    assert flight.poll()["t"] == pytest.approx(0.14)


def test_a_press_count_toggles_the_autopilot_once_however_often_it_is_sent():
    flight, clock = fly()
    flight.poll()
    for _ in range(3):  # the same count, sent again: a retried or duplicated poll
        clock.now += 0.04
        assert flight.poll(presses={"a": 1})["mode"] == "autopilot"
    clock.now += 0.04
    assert flight.poll(presses={"a": 2})["mode"] == "manual"


def test_held_keys_reach_the_stick_and_unknown_keys_are_dropped():
    flight, clock = fly()
    flight.poll()
    clock.now = 0.2
    frame = flight.poll(held=["down", "not-a-key"])
    assert flight.keys.held == {"down"}
    assert frame["stick"]["pitch"] < 0.0  # "down" is stick back: panel.KEYMAP


def test_a_frame_is_json_ready_and_names_the_field_ahead():
    flight, _ = fly("hannibal")
    frame = flight.poll()
    json.dumps(frame)
    json.dumps(flight.info())
    lead = 40.0 * PARKS_CASES["hannibal"]["r0"]  # panel.field_ahead's default lead-in
    assert frame["field"]["distance"] == pytest.approx(lead)
    assert frame["field"]["bearing"] is None  # a line vortex has none


def test_the_gust_sketch_is_the_placed_field():
    assert not np.any(cockpit.gust_profile("calm")["w_up"])
    p = cockpit.gust_profile("hannibal")
    t, w = np.asarray(p["t"]), np.asarray(p["w_up"])
    first_core = 40.0 * PARKS_CASES["hannibal"]["r0"] / V
    # The vortex's vertical wind is antisymmetric about its core: the strongest
    # up and down lie either side of it, within a core radius in time.
    r0_s = PARKS_CASES["hannibal"]["r0"] / V
    near = np.abs(t - first_core) < 3.0
    assert abs(t[near][np.argmax(w[near])] - first_core) < r0_s + 0.3
    assert abs(t[near][np.argmin(w[near])] - first_core) < r0_s + 0.3


def test_the_debrief_passes_both_hannibal_cores_where_the_field_puts_them():
    flight, clock = fly("hannibal")
    fly_to(flight, clock, 42.0, presses={"a": 1})  # on the autopilot, straight through
    summary = flight.summary()
    json.dumps(summary)
    case = PARKS_CASES["hannibal"]
    lead = 40.0 * case["r0"]
    times = [e["t"] for e in summary["events"]]
    assert len(times) == 2
    # Held at cruise by the autopilot, so the ground speed is close to V.
    assert times[0] == pytest.approx(lead / V, abs=0.5)
    assert times[1] - times[0] == pytest.approx(case["spacing"] / V, abs=0.3)
    assert summary["nz_max"]["value"] > 1.2 and summary["nz_min"]["value"] < 0.8
    assert summary["severity_band"] in ("moderate", "severe")
    assert all(summary["series"]["autopilot"][1:])


def test_a_flight_is_saved_as_a_run_that_engineering_mode_reads(tmp_path):
    pytest.importorskip("pyarrow")
    from atisim.analysis import artifact

    flight, clock = fly("hannibal")
    fly_to(flight, clock, 42.0, presses={"a": 1})
    path = cockpit.save_run(flight, tmp_path)
    back = artifact.read_run(path)
    assert back.meta["flown_by_hand"]["test_point"] == "hannibal"
    # No spec: nothing can fly a hand-flown run again.
    assert not (path / "spec.json").exists()
    # The field is recorded where it was placed: rebuilt, it gives the wind flown.
    wind = next(c for c in back.checks if c["name"] == "recorded wind")
    assert wind["passed"], wind
    # A second save is a second run, never an overwrite.
    assert cockpit.save_run(flight, tmp_path) != path


def test_the_debrief_carries_the_paper_record_for_hannibal_and_none_in_calm_air():
    flight, clock = fly("hannibal")
    fly_to(flight, clock, 42.0, presses={"a": 1})
    record = flight.summary()["record"]
    assert record["trace"] is not None and record["bands"]
    calm, clock = fly("calm")
    fly_to(calm, clock, 5.0)
    assert calm.summary()["record"] is None


def test_the_severity_band_follows_the_sourced_thresholds():
    from atisim import checks

    assert checks.severity_band(math.nan) == "too short to rate"
    assert checks.severity_band(0.1) == "smooth"
    assert checks.severity_band(0.2) == "moderate"
    assert checks.severity_band(0.3) == "severe"
