# Free-air flying interface — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the live flying path into a basic-T cockpit with a flight-test overlay, flyable through a cited wind field, with a proportional stick and working pitch trim.

**Architecture:** Two new sensor groups in `sensors.py` fed by a factored `dynamics.specific_force`; the live cockpit extracted from `viz.py` into `atisim/panel.py` and rebuilt as small instrument units around a `Readout` bundle; `wind_model` threaded through `run_live`/`LiveSim` into `integrate.step`; a ramped `Stick` in the panel and a trim axis in `manual.py`.

**Tech Stack:** Python 3.10, JAX (x64), matplotlib (Agg in tests, TkAgg live), pytest.

**Spec:** `docs/superpowers/specs/2026-08-06-free-air-flying-interface-design.md`

**Run everything from the project root**, not from the worktree, or the editable install resolves to the main checkout (PROJECT.md §10). Test command throughout:

```bash
.venv/Scripts/python.exe -m pytest atisim/tests -q
```

---

## File structure

| File | Change | Responsibility |
|---|---|---|
| `atisim/dynamics.py` | modify | `specific_force` added; `load_factor` becomes a one-line wrapper |
| `atisim/sensors.py` | modify | `AirData.vertical_speed`; new `Accelerations` + `accelerometers` |
| `atisim/wind.py` | modify | cited case constants move in from `scripts/vortex.py` |
| `atisim/panel.py` | **create** | live cockpit, instrument units, `Stick`, `LiveSim`, `run_live` |
| `atisim/viz.py` | modify | keeps `Trajectory`, `Recorder`, `save`/`load`, `derived`, `post_flight` only |
| `atisim/manual.py` | modify | `PilotInput.trim`, `ManualGains.trim_rate`, `trim_here` |
| `scripts/fly.py` | modify | `--wind`, `--lead-in`, `--sharpness`; builds the field and the range callable |
| `scripts/vortex.py` | modify | imports the case constants instead of defining them |
| `atisim/tests/test_panel.py` | **create** | live-panel tests moved from `test_viz.py`, plus the new ones |
| `atisim/tests/test_viz.py` | modify | keeps log / `derived` / `post_flight` tests |
| `atisim/tests/test_dynamics.py` | modify | `specific_force` force check |
| `atisim/tests/test_sensors.py` | modify | vertical speed, accelerometers |
| `atisim/tests/test_manual.py` | modify | trim axis, `trim_here` |

**Off limits** (PROJECT.md §4 validated baseline): `test_conservation.py`, `test_cr2144_modes.py`, `test_drag_polar.py`, `test_navion.py`, `test_trim.py`. If one of these fails, stop and report — do not adjust a tolerance.

---

## Task 1: `dynamics.specific_force`

**Files:**
- Modify: `atisim/dynamics.py:62-90`
- Test: `atisim/tests/test_dynamics.py`

- [ ] **Step 1: Write the failing test**

Append to `atisim/tests/test_dynamics.py`. This mirrors the existing
`test_load_factor_matches_the_aerodynamic_and_thrust_force_directly` but asserts all three
components, so it pins `n_x` and `n_y`, which nothing has ever checked.

```python
def test_specific_force_matches_the_forces_in_all_three_axes(test_aircraft):
    """All three components, recomputed from aero + thrust rather than from the same call.

    load_factor only ever pinned the z component. n_x and n_y are new, and n_y is
    about to drive the slip indicator, so a wrong sign there would be a display
    that is confidently backwards.
    """
    from atisim.aero import aero_forces_moments, thrust_force
    from atisim.atmosphere import density, speed_of_sound

    s = level_state(u=60.0, altitude=2000.0)._replace(omega=jnp.array([0.1, 0.2, -0.05]))
    controls = Controls(
        elevator=jnp.array(0.1), aileron=jnp.array(-0.05),
        rudder=jnp.array(0.02), throttle=jnp.array(0.6),
    )
    altitude = -s.pos_ned[2]
    force, _ = aero_forces_moments(
        s.vel_body, s.omega, controls, test_aircraft,
        density(altitude), speed_of_sound(altitude),
    )
    force = force + thrust_force(controls, test_aircraft, density(altitude))
    expected = np.asarray(force / test_aircraft.mass) / G0

    got = np.asarray(
        dynamics.specific_force(s, controls, test_aircraft, jnp.zeros(3), jnp.zeros(3))
    )
    assert got == pytest.approx(expected, abs=1e-12)


def test_load_factor_is_the_negated_z_component_of_specific_force(test_aircraft):
    """The wrapper must not quietly change sign or scale."""
    s = level_state(u=60.0, altitude=2000.0)._replace(omega=jnp.array([0.1, 0.2, -0.05]))
    controls = Controls(
        elevator=jnp.array(0.1), aileron=jnp.array(-0.05),
        rudder=jnp.array(0.02), throttle=jnp.array(0.6),
    )
    n = dynamics.specific_force(s, controls, test_aircraft, jnp.zeros(3), jnp.zeros(3))
    n_z = dynamics.load_factor(s, controls, test_aircraft, jnp.zeros(3), jnp.zeros(3))
    assert float(n_z) == pytest.approx(-float(n[2]), abs=1e-15)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/Scripts/python.exe -m pytest atisim/tests/test_dynamics.py -q -k specific_force
```

Expected: FAIL, `AttributeError: module 'atisim.dynamics' has no attribute 'specific_force'`.

- [ ] **Step 3: Implement**

Replace `atisim/dynamics.py:62-90` with the following. The long docstring moves to
`specific_force` because that is where the inversion argument now lives; `load_factor` keeps a short
one naming its convention.

```python
def specific_force(
    state: State,
    controls: Controls,
    ac: Aircraft,
    wind_ned: Array,
    omega_gust: Array,
) -> Array:
    """Body-axis specific force in g: what a three-axis accelerometer at the CG reads.

    Specific force is the aerodynamic plus propulsive force over mass and
    excludes gravity. `derivatives` computes exactly that as `force / ac.mass`
    and then discards it inside the sum at the top of this module, so it is
    recovered here by inverting that sum:

        a_spec = vdot_body - g_body + omega x vel_body

    Inverting rather than recomputing `force / mass` is deliberate. It cannot
    silently disagree with the plant if a force term is ever added to
    `derivatives`, because it inverts whatever `derivatives` actually did.

    Signs are body axes throughout: +x forward, +y right, +z down. Note that
    `load_factor` NEGATES the z component, because the load factor convention is
    +1 in level flight while a_spec[2] is negative there.
    """
    d = derivatives(state, controls, ac, wind_ned, omega_gust)
    gravity_body = quat_to_dcm(state.quat).T @ jnp.array([0.0, 0.0, G0])
    return (d.vel_body - gravity_body + jnp.cross(state.omega, state.vel_body)) / G0


def load_factor(
    state: State,
    controls: Controls,
    ac: Aircraft,
    wind_ned: Array,
    omega_gust: Array,
) -> Array:
    """Normal load factor n_z. +1 in level flight, 0 in free fall.

    Body-normal, not flight-path-normal: this is the quantity Wingrove & Bach's
    Fig. 8 is built from, since DFDR "normal acceleration" is what an
    accelerometer reads. In trimmed level flight it is cos(theta), not 1.
    """
    return -specific_force(state, controls, ac, wind_ned, omega_gust)[2]
```

- [ ] **Step 4: Run the tests to verify they pass, including the two existing ones unchanged**

```bash
.venv/Scripts/python.exe -m pytest atisim/tests/test_dynamics.py -q
```

Expected: PASS. The two pre-existing `test_load_factor_*` tests must pass **with no edits** — that
is the regression guard on the factoring.

- [ ] **Step 5: Commit**

```bash
git add atisim/dynamics.py atisim/tests/test_dynamics.py
git commit -m "Factor load_factor into specific_force"
```

---

## Task 2: Vertical speed and the accelerometer package

**Files:**
- Modify: `atisim/sensors.py`
- Test: `atisim/tests/test_sensors.py`

- [ ] **Step 1: Write the failing tests**

Append to `atisim/tests/test_sensors.py`.

```python
def test_vertical_speed_is_inertial_and_matches_the_ned_velocity(trimmed):
    """Computed independently from the quaternion, not by calling the same helper.

    A VSI fed anything other than vertical speed -- body w, or the negated
    down-velocity of the AIR-relative vector -- passes any still-air, level test.
    This one climbs and rolls so neither substitution survives.
    """
    state, _ = trimmed
    climbing = state._replace(
        quat=euler_to_quat(jnp.array(0.3), jnp.array(0.15), jnp.array(0.7))
    )
    air = sense(climbing, jnp.array([5.0, -3.0, 2.0]))

    vel_ned = quat_to_dcm(climbing.quat) @ climbing.vel_body
    assert float(air.vertical_speed) == pytest.approx(-float(vel_ned[2]), abs=1e-12)
    assert float(air.vertical_speed) != pytest.approx(-float(climbing.vel_body[2]), abs=1e-3)


def test_vertical_speed_ignores_the_wind_because_a_baro_vsi_would(trimmed):
    """It is INERTIAL. An updraft that carries the aircraft up changes its
    geometric height, and that is what the instrument sees -- but the wind
    argument itself must not enter the calculation."""
    state, _ = trimmed
    still = sense(state, jnp.zeros(3))
    blown = sense(state, jnp.array([0.0, 0.0, -12.0]))
    assert float(blown.vertical_speed) == pytest.approx(float(still.vertical_speed), abs=1e-12)


def test_accelerometers_report_the_specific_force_with_load_factor_sign(trimmed):
    state, controls = trimmed
    ac = REGISTRY["boeing747"]
    n = sensors.accelerometers(state, controls, ac, jnp.zeros(3), jnp.zeros(3))
    raw = dynamics.specific_force(state, controls, ac, jnp.zeros(3), jnp.zeros(3))

    assert float(n.n_x) == pytest.approx(float(raw[0]), abs=1e-15)
    assert float(n.n_y) == pytest.approx(float(raw[1]), abs=1e-15)
    assert float(n.n_z) == pytest.approx(-float(raw[2]), abs=1e-15)
    # Level flight: n_z is cos(theta), and the lateral axis is quiet.
    assert float(n.n_z) == pytest.approx(0.9967, abs=1e-3)
    assert abs(float(n.n_y)) < 1e-6
```

Add whatever imports the file is missing at the top: `from atisim import dynamics, sensors`,
`from atisim.aircraft import REGISTRY`, `from atisim.state import euler_to_quat, quat_to_dcm`.

- [ ] **Step 2: Run to verify they fail**

```bash
.venv/Scripts/python.exe -m pytest atisim/tests/test_sensors.py -q -k "vertical_speed or accelerometers"
```

Expected: FAIL — `AirData` has no `vertical_speed`, `sensors` has no `accelerometers`.

- [ ] **Step 3: Implement**

In `atisim/sensors.py`, extend the module docstring's sensor table with two entries:

```
    vertical_speed          barometric VSI. INERTIAL. It measures the rate of
                            change of static pressure, and an updraft that
                            carries the aircraft up does change its geometric
                            height -- so the wind never enters the calculation,
                            but its effect on the trajectory is still read.
    n_x, n_y, n_z           accelerometer package. Specific force, so gravity is
                            excluded by construction. A SEPARATE function,
                            because it needs the controls and the aircraft and
                            `sense` has neither.
```

Add the import and the field:

```python
from atisim.dynamics import relative_velocity, specific_force
from atisim.state import State, quat_to_dcm, quat_to_euler


class AirData(NamedTuple):
    """One sensor set. Air-relative where a real sensor would be, inertial elsewhere."""

    airspeed: Array  # m/s, true airspeed -- AIR-RELATIVE
    alpha: Array  # rad -- AIR-RELATIVE
    beta: Array  # rad -- AIR-RELATIVE
    phi: Array  # rad, inertial
    theta: Array  # rad, inertial
    psi: Array  # rad, inertial
    p: Array  # rad/s, inertial
    q: Array  # rad/s, inertial
    r: Array  # rad/s, inertial
    altitude: Array  # m
    vertical_speed: Array  # m/s, positive UP, inertial
```

In `sense`, before the return, and add the field last so every existing keyword call site is
untouched:

```python
    vel_ned = quat_to_dcm(state.quat) @ state.vel_body
```
```python
        altitude=-state.pos_ned[2],
        vertical_speed=-vel_ned[2],
    )
```

Append the second sensor group:

```python
class Accelerations(NamedTuple):
    """The accelerometer package. Body axes, in g.

    NOTE THE SIGN ASYMMETRY, which is inherited and not a mistake: `n_z` follows
    the load-factor convention and reads +1 in level flight, so it is the
    NEGATED z component of the specific force. `n_x` and `n_y` are the raw
    components, positive forward and positive right.
    """

    n_x: Array  # g, positive FORWARD
    n_y: Array  # g, positive RIGHT
    n_z: Array  # g, positive UP-ish: +1 in level flight


def accelerometers(
    state: State,
    controls: Controls,
    ac: Aircraft,
    wind_ned: Array = STILL_AIR,
    omega_gust: Array = STILL_AIR,
) -> Accelerations:
    """Read the accelerometer package.

    Separate from `sense` because it needs the controls and the aircraft: a
    specific force is a force over a mass, and the air-data computer knows
    neither. Making it a second function is better than making `sense` depend on
    `dynamics`.
    """
    n = specific_force(state, controls, ac, wind_ned, omega_gust)
    return Accelerations(n_x=n[0], n_y=n[1], n_z=-n[2])
```

Add `from atisim.aircraft import Aircraft` and `from atisim.state import Controls` to the
imports.

- [ ] **Step 4: Run the full suite**

```bash
.venv/Scripts/python.exe -m pytest atisim/tests -q
```

Expected: PASS. `test_still_air_sensing_is_unchanged` must still hold to atol 1e-12 — it constructs
`AirData` positionally nowhere, so appending a field is safe. If it fails, a call site is building
`AirData` positionally and must be found.

- [ ] **Step 5: Commit**

```bash
git add atisim/sensors.py atisim/tests/test_sensors.py
git commit -m "Add vertical speed and the accelerometer package to sensors"
```

---

## Task 3: Move the cited case data into `wind.py`

**Files:**
- Modify: `atisim/wind.py`, `scripts/vortex.py`

- [ ] **Step 1: Add the constants to `wind.py`**

Immediately after the `VortexArray` class, so the identified values sit beside the type they
parameterise. `FT2M` must be imported from `atisim.units` — check whether it already is.

```python
# The two cases Parks et al. 1985 identifies, J. Aircraft 22(2) pp. 127-128.
# Both were DC-10s near the tropopause. Kept here rather than in a script so
# that fly.py and vortex.py cannot drift apart on a sourced number.
#
#   Case 1, Hannibal MO,  37,000 ft: r0 = 600 ft, V0 = 85 ft/s, spacing 3500 ft
#   Case 2, Morton WY,    39,000 ft: r0 = 450 ft, V0 = 70 ft/s, spacing 3200 ft
PARKS_CASES: dict[str, dict[str, float]] = {
    "hannibal": {"r0": 600.0 * FT2M, "v0": 85.0 * FT2M, "spacing": 3500.0 * FT2M},
    "morton": {"r0": 450.0 * FT2M, "v0": 70.0 * FT2M, "spacing": 3200.0 * FT2M},
}
```

And beside `UpdraftColumn`:

```python
# Wingrove & Bach 1994 p. 756, Bermuda 12 Oct 1983: over 80 ft/s, 20 s traverse.
UPDRAFT_W0 = 80.0 * FT2M  # m/s
UPDRAFT_SECONDS = 20.0  # s, a TRAVERSE time -- it fixes diameter only with a speed
```

- [ ] **Step 2: Point `scripts/vortex.py` at them**

Delete the `CASES`, `UPDRAFT_W0` and `UPDRAFT_SECONDS` definitions (currently at
`scripts/vortex.py:22-29`) and their comments, and import instead:

```python
from atisim.wind import PARKS_CASES as CASES, UPDRAFT_SECONDS, UPDRAFT_W0
```

`FT2M` may now be unused in `scripts/vortex.py`. Remove it from the import line if so, and nothing
else — this is the only orphan the change creates.

- [ ] **Step 3: Verify the sourced result is unmoved**

```bash
.venv/Scripts/python.exe scripts/vortex.py --case hannibal --png runs/plan-task3.png
```

Expected: runs to completion and prints a first-core Δθ of **2.24 deg**, matching PROJECT.md §4. Any
other number means the constants were transcribed wrong. Delete `runs/plan-task3.png` afterwards.

- [ ] **Step 4: Run the suite**

```bash
.venv/Scripts/python.exe -m pytest atisim/tests -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add atisim/wind.py scripts/vortex.py
git commit -m "Move the cited vortex and updraft constants into wind.py"
```

---

## Task 4: Thread the wind model through the live loop

**Files:**
- Modify: `atisim/viz.py` (`LiveSim.__init__`, `LiveSim.advance`, `run_live`)
- Test: `atisim/tests/test_viz.py`

Done before the panel split so that the split (Task 5) is a pure move with no behaviour change.

- [ ] **Step 1: Write the failing tests**

Append to `atisim/tests/test_viz.py`.

```python
def test_a_live_run_through_a_wind_field_differs_from_still_air(trimmed, targets):
    """The wind must reach the plant.

    This is the test that stops the whole feature being a parameter that is
    accepted and ignored -- the exact shape of latent bugs (a) and (b), which
    survived three sessions because nothing flew through a non-zero field.
    """
    from atisim import wind as wind_mod

    def fly(wind_model):
        state, controls = trimmed
        ctl = man.start(sense(state), controls, targets, GAINS, AC)
        sim = integrate.init_sim(state, jax.random.PRNGKey(0))
        panel = viz.Panel(targets, window=20.0, fps=20.0)
        live = viz.LiveSim(
            sim, ctl, targets, GAINS, MGAINS, AC, panel,
            dt=DT, real_time=False, wind_model=wind_model,
        )
        for _ in range(40):
            live.frame()
        return np.asarray(live.trajectory().pos_ned)

    column = wind_mod.UpdraftColumn(
        north=jnp.array(0.0), east=jnp.array(0.0), w0=jnp.array(20.0),
        radius=jnp.array(3000.0), sharpness=jnp.array(6.0),
    )
    blown = fly(wind_mod.field_model(lambda p: wind_mod.updraft_wind(p, column)))
    still = fly(wind_mod.zero_wind)

    assert abs(blown[-1, 2] - still[-1, 2]) > 1.0  # metres of altitude


def test_the_live_loop_in_still_air_is_untouched_by_the_wind_plumbing(live):
    """Mirrors PROJECT.md section 4's zero-strength-wind row: adding the hook
    must not perturb the default path by one bit."""
    for _ in range(40):
        live.frame()
    blown = np.asarray(live.trajectory().pos_ned)

    state, controls = live.recorder._rows[0][1], None  # unused; rebuild from trim below
    assert np.isfinite(blown).all()
```

Replace that second test's body with a real comparison against an explicit `zero_wind` model, which
is the actual invariant:

```python
def test_the_live_loop_in_still_air_is_untouched_by_the_wind_plumbing(trimmed, targets):
    from atisim import wind as wind_mod

    def fly(**kwargs):
        state, controls = trimmed
        ctl = man.start(sense(state), controls, targets, GAINS, AC)
        sim = integrate.init_sim(state, jax.random.PRNGKey(0))
        panel = viz.Panel(targets, window=20.0, fps=20.0)
        live = viz.LiveSim(
            sim, ctl, targets, GAINS, MGAINS, AC, panel,
            dt=DT, real_time=False, **kwargs,
        )
        for _ in range(40):
            live.frame()
        return np.asarray(live.trajectory().pos_ned)

    assert np.array_equal(fly(), fly(wind_model=wind_mod.zero_wind))
```

- [ ] **Step 2: Run to verify they fail**

```bash
.venv/Scripts/python.exe -m pytest atisim/tests/test_viz.py -q -k "wind_field or wind_plumbing"
```

Expected: FAIL — `LiveSim.__init__` got an unexpected keyword argument `wind_model`.

- [ ] **Step 3: Implement**

In `atisim/viz.py`, add the import:

```python
from atisim.wind import zero_wind
```

`LiveSim.__init__` gains two keyword arguments after `max_steps_per_frame`:

```python
        wind_model=zero_wind,
        field_range=None,
```
```python
        self.wind_model = wind_model
        self.field_range = field_range
```

`LiveSim.advance`, the `step` call:

```python
            self.sim = step(self.sim, self.controls, self.dt, self.ac, self.wind_model)
```

`run_live` gains the same two keyword arguments and passes them on. **The warm-up must warm the real
model**:

```python
    step(sim, warm, dt, ac, wind_model)
```

with a comment saying why:

```python
    # Warmed with the ACTUAL wind model, not the default. `step` takes
    # wind_model as a static argument, so a different model is a different
    # compilation -- warming zero_wind here would leave the real one to compile
    # inside the first frame, whose backlog is then dropped straight out of the
    # real-time budget.
```

- [ ] **Step 4: Run the suite**

```bash
.venv/Scripts/python.exe -m pytest atisim/tests -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add atisim/viz.py atisim/tests/test_viz.py
git commit -m "Thread a wind model through the live flying loop"
```

---

## Task 5: Split the live panel into `atisim/panel.py`

A **pure move**. No behaviour changes, no renames, no reformatting. Doing this separately is what
makes Task 6's diff readable: if the panel breaks, it broke in the re-layout, not in the move.

**Files:**
- Create: `atisim/panel.py`, `atisim/tests/test_panel.py`
- Modify: `atisim/viz.py`, `atisim/tests/test_viz.py`, `scripts/fly.py`

- [ ] **Step 1: Create `atisim/panel.py`**

Move, verbatim: `KEYMAP`, `TOGGLE_KEY`, `HELP`, the `_BG`/`_FG`/`_SKY`/`_GROUND`/`_SYMBOL`/`_TRACE`
colour constants, `PITCH_SPAN_DEG`, `_HORIZON_L`, `_push`, `_horizon_frame`, `_quad`, `_ladder`,
`_SYMBOL_X`, `_SYMBOL_Y`, `Panel`, `LiveSim`, `run_live` (currently `viz.py:193-712`).

Module docstring: move the two paragraphs of `viz.py`'s docstring that describe the decoupled physics
and the blitting constraint — they describe this module, not the log.

Imports it needs:

```python
import time

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon

from atisim import manual as man
from atisim.aircraft import Aircraft
from atisim.autopilot import Gains, Targets, wrap_pi
from atisim.integrate import SimState, step
from atisim.manual import Controller, ManualGains, Mode, PilotInput
from atisim.sensors import sense
from atisim.state import Controls
from atisim.units import RAD2DEG
from atisim.viz import Recorder, Trajectory
from atisim.wind import zero_wind
```

- [ ] **Step 2: Strip `viz.py` back**

Delete everything moved. `viz.py` keeps its first docstring paragraph (rewritten to describe the log
and the post-flight figure only), `Trajectory`, `save`, `load`, `Recorder`, `Derived`, `derived`,
`post_flight`. Remove the imports that are now unused: `time`, `FuncAnimation`, `Polygon`,
`SimState`, `step`, `Controller`, `ManualGains`, `PilotInput`, `manual as man`, `Aircraft`, `Gains`,
`wrap_pi`, `State` if unused, `zero_wind`.

`viz.py` still needs: `jax`, `jnp`, `plt`, `np`, `Mode`, `sense`, `State`, `Controls`, `RAD2DEG`.

- [ ] **Step 3: Update `scripts/fly.py`**

```python
from atisim import integrate, manual as man, panel as panel_mod, trim, viz
```
```python
traj = panel_mod.run_live(
    sim, ctl, targets, gains, mgains, ac, dt=args.dt, fps=args.fps, window=args.window
)
```

- [ ] **Step 4: Split the tests**

Create `atisim/tests/test_panel.py` with the module docstring, the `matplotlib.use("Agg")`
preamble, the fixtures (`_close_figures`, `trimmed`, `targets`, `live`), `press`, `release`, and
these tests moved verbatim from `test_viz.py`: the five horizon-geometry tests, the four keyboard
tests, `test_the_stick_reaches_the_plant`, the three `LiveSim` timing tests, the two `_retime` tests
plus `FakeAnimation`, `test_the_animation_runs_blitted_and_moves_its_artists`,
`test_axis_limits_never_move_because_blitting_would_not_notice`, and the two wind tests from Task 4.

Change their imports from `viz.Panel` / `viz.LiveSim` / `viz.PITCH_SPAN_DEG` to the `panel` module.

`test_viz.py` keeps: `test_trajectory_round_trips_through_npz`,
`test_the_log_has_one_row_per_step_and_the_documented_shapes`,
`test_derived_agrees_with_the_aero_module`, `test_a_saved_run_can_be_replotted_without_the_simulator`,
`test_post_flight_shows_both_modes_on_the_timeline`. It still needs a `live` fixture, so it imports
`panel` too.

- [ ] **Step 5: Run the suite and check the count**

```bash
.venv/Scripts/python.exe -m pytest atisim/tests -q
```

Expected: PASS, and the collected count must be **exactly what it was before the move** — a pure move
neither adds nor loses a test. Note the number.

- [ ] **Step 6: Commit**

```bash
git add atisim/panel.py atisim/viz.py atisim/tests/test_panel.py atisim/tests/test_viz.py scripts/fly.py
git commit -m "Move the live cockpit into atisim/panel.py"
```

---

## Task 6: Re-lay-out the panel as a basic T with a test overlay

**Files:**
- Modify: `atisim/panel.py`
- Test: `atisim/tests/test_panel.py`

- [ ] **Step 1: Add the `Readout` and `FieldRange` types**

In `atisim/panel.py`, above `Panel`:

```python
class FieldRange(NamedTuple):
    """Where the wind field is, in the terms its own geometry supports.

    `bearing` is None for a VortexArray, and that is not an omission: the cores
    are infinite line vortices running east-west (`vortex_wind` reads only
    pos_ned[0] and pos_ned[2], and the induced wind has no east component). A
    bearing to a line is meaningless, so none is reported rather than one being
    invented.
    """

    label: str
    distance: float  # m. North distance to a line vortex; slant range to a column.
    closing: float  # m/s, rate of change of `distance`
    bearing: float | None  # rad, None for a line vortex


class Readout(NamedTuple):
    """One frame's worth of everything the panel shows.

    Exists so every instrument has the same update signature and can be tested
    without building a figure.
    """

    t: float
    air: AirData
    accel: Accelerations
    controls: Controls
    reference: Controls  # the stick's centring point, for the trim readout
    mode: Mode
    wind_ned: np.ndarray
    omega_gust: np.ndarray
    field: FieldRange | None
```

- [ ] **Step 2: Add the declared display constants**

```python
# --- declared display constants --------------------------------------------
#
# None of these come from a source. They are display choices, named here rather
# than buried in a call so that changing one is a visible decision.

# PROJECT.md section 7: "any encounter driving alpha past ~10-12 deg reports
# lift the sources say is not there". That is a statement about this model's
# linear aero, NOT a stall table -- aero.py is CL = CL0 + CLa*alpha with no
# stall, and the only aircraft in the project with nonlinear data is out of
# scope. The band exists so a run that leaves the model's valid range says so on
# the panel instead of in a footnote.
ALPHA_LINEAR_DEG = 10.0
ALPHA_INVALID_DEG = 12.0
ALPHA_SPAN_DEG = 15.0

NZ_RANGE = (-1.0, 3.0)  # g
WIND_SPAN = 40.0  # m/s
GUST_SPAN = 0.15  # rad/s

# Vertical-speed full scale, per aircraft. The 747 at cruise trades altitude for
# speed far faster than a light aircraft does.
VSI_SPAN: dict[str, float] = {"boeing747": 20.0, "cherokee": 10.0, "cessna172": 10.0}
```

- [ ] **Step 3: Write the failing instrument tests**

Append to `atisim/tests/test_panel.py`. These drive instruments directly, which is the point of
the unit shape.

```python
def readout_at(**overrides):
    """A Readout with everything quiet, for poking one instrument at a time."""
    air = AirData(
        airspeed=jnp.array(236.0), alpha=jnp.array(0.05), beta=jnp.array(0.0),
        phi=jnp.array(0.0), theta=jnp.array(0.08), psi=jnp.array(0.0),
        p=jnp.array(0.0), q=jnp.array(0.0), r=jnp.array(0.0),
        altitude=jnp.array(12192.0), vertical_speed=jnp.array(0.0),
    )
    base = dict(
        t=0.0, air=air,
        accel=panel_mod.Accelerations(
            n_x=jnp.array(0.0), n_y=jnp.array(0.0), n_z=jnp.array(1.0)
        ),
        controls=Controls(*[jnp.array(0.0)] * 4),
        reference=Controls(*[jnp.array(0.0)] * 4),
        mode=Mode.MANUAL,
        wind_ned=np.zeros(3), omega_gust=np.zeros(3), field=None,
    )
    base.update(overrides)
    if "air" in overrides:
        base["air"] = overrides["air"]
    return panel_mod.Readout(**base)


def test_the_vsi_needle_moves_up_in_a_climb_and_down_in_a_descent(targets):
    p = panel_mod.Panel(targets, window=20.0, fps=20.0, aircraft_name="boeing747")
    air = readout_at().air

    p.vsi.update(readout_at(air=air._replace(vertical_speed=jnp.array(+8.0))))
    climbing = p.vsi.needle.get_ydata()[-1]
    p.vsi.update(readout_at(air=air._replace(vertical_speed=jnp.array(-8.0))))
    descending = p.vsi.needle.get_ydata()[-1]

    assert climbing > 0.0 > descending
    assert climbing == pytest.approx(-descending)


def test_the_alpha_band_turns_red_past_the_declared_ceiling(targets):
    p = panel_mod.Panel(targets, window=20.0, fps=20.0, aircraft_name="boeing747")
    air = readout_at().air

    p.alpha_gauge.update(readout_at(air=air._replace(alpha=jnp.deg2rad(4.0))))
    assert p.alpha_gauge.state() == "linear"
    p.alpha_gauge.update(readout_at(air=air._replace(alpha=jnp.deg2rad(11.0))))
    assert p.alpha_gauge.state() == "marginal"
    p.alpha_gauge.update(readout_at(air=air._replace(alpha=jnp.deg2rad(14.0))))
    assert p.alpha_gauge.state() == "invalid"


def test_the_load_factor_gauge_holds_the_peak_excursion(targets):
    p = panel_mod.Panel(targets, window=20.0, fps=20.0, aircraft_name="boeing747")
    for n_z in (1.0, 2.4, 1.1, 0.2):
        p.nz_gauge.update(
            readout_at(accel=panel_mod.Accelerations(
                n_x=jnp.array(0.0), n_y=jnp.array(0.0), n_z=jnp.array(n_z)
            ))
        )
    assert p.nz_gauge.peak_high == pytest.approx(2.4)
    assert p.nz_gauge.peak_low == pytest.approx(0.2)
```

- [ ] **Step 4: Write the slip-ball test — the one that pins the physics**

```python
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
    offset = live.panel.slip.offset  # what the instrument draws

    assert abs(float(air.beta)) > np.deg2rad(0.5)  # there IS a sideslip to indicate
    assert offset == pytest.approx(-float(accel.n_y) / panel_mod.SLIP_SPAN, abs=1e-12)
    # Step on the ball: the indicated pedal is the one opposing the rudder held.
    assert offset < 0.0
```

The final assertion's sign is the spec's derivation (ball displaces along `-n_y`; right rudder gives
`n_y > 0`, so the ball goes left, i.e. negative). **If it fails, do not flip the assertion to match
the code** — check `specific_force`'s y sign against `test_specific_force_matches_the_forces_in_all_three_axes`
first, and only then flip `SLIP_SPAN`'s sign convention, recording the correction in PROJECT.md §9.

- [ ] **Step 5: Run to verify they fail**

```bash
.venv/Scripts/python.exe -m pytest atisim/tests/test_panel.py -q -k "vsi or alpha_band or load_factor_gauge or ball"
```

Expected: FAIL — `Panel` has no `aircraft_name`, no `vsi`, no `alpha_gauge`, no `nz_gauge`, no `slip`.

- [ ] **Step 6: Implement the instrument units**

Each instrument is a small class with `artists`, `update(r: Readout)`, and fixed limits set in
`__init__`. The tape is the novel one; write it first and the rest follow its shape.

```python
class Tape:
    """A vertical scale that slides past a fixed pointer.

    Blitting forbids moving an axis limit, so the VALUE stays fixed at the
    centre and the SCALE moves: tick positions and tick label text are animated
    artists inside axes whose limits never change. That is what lets a tape live
    on a blitted panel at all.
    """

    N_TICKS = 7

    def __init__(self, ax, label: str, span: float, step: float, fmt: str):
        self.ax, self.span, self.step, self.fmt = ax, span, step, fmt
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-span, span)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor(_BG)
        for spine in ax.spines.values():
            spine.set_color("#39424b")
        ax.set_title(label, color=_FG, fontsize=8)
        self.ticks = [
            ax.plot([], [], color=_FG, lw=0.8, animated=True)[0]
            for _ in range(self.N_TICKS)
        ]
        self.tick_labels = [
            ax.text(0.30, 0.0, "", color=_FG, fontsize=7, family="monospace",
                    ha="right", va="center", animated=True)
            for _ in range(self.N_TICKS)
        ]
        self.box = ax.text(
            0.5, 0.0, "", color=_SYMBOL, fontsize=10, family="monospace",
            ha="center", va="center", animated=True,
            bbox=dict(facecolor=_BG, edgecolor=_SYMBOL, boxstyle="square,pad=0.25"),
        )

    @property
    def artists(self) -> list:
        return [*self.ticks, *self.tick_labels, self.box]

    def show(self, value: float) -> None:
        centre = round(value / self.step) * self.step
        first = centre - (self.N_TICKS // 2) * self.step
        for index, (tick, text) in enumerate(zip(self.ticks, self.tick_labels)):
            mark = first + index * self.step
            y = mark - value
            tick.set_data([0.34, 0.46], [y, y])
            text.set_position((0.30, y))
            text.set_text(self.fmt.format(mark))
        self.box.set_text(self.fmt.format(value))
```

The airspeed and altitude tapes are `Tape` instances updated from `r.air.airspeed` and
`r.air.altitude`. The heading tape is the same idea rotated: subclass or parameterise on axis. Use a
horizontal variant `HeadingTape` with the same `show` logic on x instead of y, and wrap the value
about the target with the existing `wrap_pi` so a heading either side of north does not jump.

```python
class VSI:
    """Vertical speed as a needle on a fixed arc. Span is per-aircraft."""

    def __init__(self, ax, span: float):
        self.span = span
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor(_BG)
        for spine in ax.spines.values():
            spine.set_color("#39424b")
        ax.set_title("VS", color=_FG, fontsize=8)
        ax.axhline(0.0, color="#5a6a78", lw=0.9)
        (self.needle,) = ax.plot([], [], color=_TRACE, lw=2.0, animated=True)
        self.readout = ax.text(
            0.0, -0.88, "", color=_SYMBOL, fontsize=9, family="monospace",
            ha="center", animated=True,
        )

    @property
    def artists(self) -> list:
        return [self.needle, self.readout]

    def update(self, r: Readout) -> None:
        vs = float(r.air.vertical_speed)
        y = float(np.clip(vs / self.span, -1.0, 1.0))
        self.needle.set_data([-0.5, 0.5], [0.0, y])
        self.readout.set_text(f"{vs:+5.1f}")
```

```python
SLIP_SPAN = 0.30  # g of lateral specific force at full ball deflection. Declared.


class SlipBall:
    """Lateral specific force, drawn where a PFD puts it: under the roll pointer.

    A ball is a bead in a curved tube. It reads n_y, NOT beta -- the two agree in
    steady coordinated flight and part company everywhere interesting, which in
    a turbulence simulator is everywhere that matters.

    The bead settles where the tube's normal force supplies the aircraft's
    specific force. The tube curves upward, so a bead displaced to +y is pushed
    toward -y; a specific force to the right therefore puts the ball LEFT. Hence
    the negation, which is pinned by
    test_the_ball_indicates_the_rudder_that_would_reduce_the_sideslip.
    """

    def __init__(self, ax, y: float = 0.80, half_width: float = 0.16):
        self.y, self.half_width = y, half_width
        self.offset = 0.0
        (self.cage,) = ax.plot(
            [-half_width, -half_width, np.nan, half_width, half_width],
            [y - 0.05, y + 0.05, np.nan, y - 0.05, y + 0.05],
            color=_FG, lw=1.0, animated=True,
        )
        (self.ball,) = ax.plot(
            [], [], marker="o", ms=6, color=_SYMBOL, animated=True
        )

    @property
    def artists(self) -> list:
        return [self.cage, self.ball]

    def update(self, r: Readout) -> None:
        self.offset = float(np.clip(-float(r.accel.n_y) / SLIP_SPAN, -1.0, 1.0))
        self.ball.set_data([self.offset * self.half_width], [self.y])
```

```python
class AlphaGauge:
    """Air-relative alpha against the DECLARED linear-aero ceiling."""

    def __init__(self, ax):
        ax.set_xlim(0.0, ALPHA_SPAN_DEG)
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([])
        ax.set_facecolor(_BG)
        ax.tick_params(colors=_FG, labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#39424b")
        ax.set_title("alpha  air-relative", color=_FG, fontsize=8)
        ax.axvspan(0.0, ALPHA_LINEAR_DEG, color="#2f5f42")
        ax.axvspan(ALPHA_LINEAR_DEG, ALPHA_INVALID_DEG, color="#6d5423")
        ax.axvspan(ALPHA_INVALID_DEG, ALPHA_SPAN_DEG, color="#5f2a24")
        (self.needle,) = ax.plot([], [], color="white", lw=2.4, animated=True)
        self.readout = ax.text(
            0.04, 0.80, "", transform=ax.transAxes, color=_SYMBOL,
            fontsize=9, family="monospace", va="top", animated=True,
        )
        self._deg = 0.0

    @property
    def artists(self) -> list:
        return [self.needle, self.readout]

    def state(self) -> str:
        if self._deg >= ALPHA_INVALID_DEG:
            return "invalid"
        if self._deg >= ALPHA_LINEAR_DEG:
            return "marginal"
        return "linear"

    def update(self, r: Readout) -> None:
        self._deg = float(r.air.alpha) * RAD2DEG
        x = float(np.clip(self._deg, 0.0, ALPHA_SPAN_DEG))
        self.needle.set_data([x, x], [0.0, 1.0])
        self.readout.set_text(f"{self._deg:+5.1f} deg  {self.state()}")
```

```python
class LoadFactorGauge:
    """n_z with a peak-excursion hold, because the excursion is the measurement."""

    def __init__(self, ax):
        ax.set_xlim(*NZ_RANGE)
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([])
        ax.set_facecolor(_BG)
        ax.tick_params(colors=_FG, labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#39424b")
        ax.set_title("load factor  n_z", color=_FG, fontsize=8)
        ax.axvline(1.0, color="#5a6a78", lw=0.9, ls="--")
        (self.needle,) = ax.plot([], [], color=_TRACE, lw=2.4, animated=True)
        self.readout = ax.text(
            0.04, 0.80, "", transform=ax.transAxes, color=_SYMBOL,
            fontsize=9, family="monospace", va="top", animated=True,
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
            f"{n_z:+5.2f} g   peak {self.peak_low:+.2f} / {self.peak_high:+.2f}"
        )
```

```python
class WindGauge:
    """Applied wind as an arrow, plus the groundspeed it implies.

    Legitimate cockpit information: with no sensor noise, ground velocity minus
    air velocity IS the wind, so a real air-data/INS pair could compute this.
    The gust RATE below it is different and is labelled accordingly.
    """

    def __init__(self, ax):
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor(_BG)
        for spine in ax.spines.values():
            spine.set_color("#39424b")
        ax.set_title("wind", color=_FG, fontsize=8)
        (self.arrow,) = ax.plot([], [], color=_TRACE, lw=2.0, animated=True)
        self.readout = ax.text(
            0.0, -0.92, "", color=_SYMBOL, fontsize=8, family="monospace",
            ha="center", animated=True,
        )

    @property
    def artists(self) -> list:
        return [self.arrow, self.readout]

    def update(self, r: Readout) -> None:
        north, east = float(r.wind_ned[0]), float(r.wind_ned[1])
        speed = float(np.hypot(north, east))
        scale = min(speed / WIND_SPAN, 1.0)
        # Screen x is east, screen y is north.
        if speed > 1e-9:
            self.arrow.set_data([0.0, scale * east / speed], [0.0, scale * north / speed])
        else:
            self.arrow.set_data([], [])
        self.readout.set_text(f"{speed:5.1f} m/s")
```

```python
class GustGauge:
    """The three body-axis gust rates. SIM TRUTH -- no instrument senses these.

    omega_gust is a gradient across the span and chord. A rate gyro measures the
    airframe's own rotation and cannot see it, which is exactly why aero.py gets
    `omega - omega_gust` and a controller does not. Presenting it as an
    instrument reading would be a lie, so it is labelled.
    """

    def __init__(self, ax):
        ax.set_xlim(-GUST_SPAN, GUST_SPAN)
        ax.set_ylim(-0.5, 2.5)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["r", "q", "p"], color=_FG, fontsize=7)
        ax.set_facecolor(_BG)
        ax.tick_params(colors=_FG, labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#39424b")
        ax.set_title("gust rate   SIM TRUTH", color="#c04a3a", fontsize=8)
        ax.axvline(0.0, color="#5a6a78", lw=0.9)
        self.bars = [
            ax.plot([], [], color=_TRACE, lw=5.0, solid_capstyle="butt", animated=True)[0]
            for _ in range(3)
        ]

    @property
    def artists(self) -> list:
        return self.bars

    def update(self, r: Readout) -> None:
        for index, bar in enumerate(self.bars):
            value = float(np.clip(r.omega_gust[2 - index], -GUST_SPAN, GUST_SPAN))
            bar.set_data([0.0, value], [index, index])
```

- [ ] **Step 7: Rebuild `Panel`**

`Panel.__init__` gains `aircraft_name: str = "boeing747"` (it needs it for `VSI_SPAN`). Replace
`_build_trace` with nothing — the 3D trace is dropped — and lay out:

```python
        grid = self.fig.add_gridspec(
            3, 6,
            width_ratios=(0.30, 1.05, 0.30, 0.17, 0.70, 0.92),
            height_ratios=(1.00, 0.18, 0.34),
            hspace=0.35, wspace=0.30,
        )
        self.asi = Tape(self.fig.add_subplot(grid[0, 0]), "TAS m/s", 40.0, 10.0, "{:.0f}")
        self._build_horizon(grid[0, 1])          # + roll pointer, slip ball, FPV
        self.alt = Tape(self.fig.add_subplot(grid[0, 2]), "ALT m", 600.0, 200.0, "{:.0f}")
        self.vsi = VSI(self.fig.add_subplot(grid[0, 3]), VSI_SPAN[aircraft_name])
        self.hdg = HeadingTape(self.fig.add_subplot(grid[1, 0:4]), float(targets.heading))

        overlay = grid[0:2, 4].subgridspec(4, 1, hspace=0.55)
        self.nz_gauge = LoadFactorGauge(self.fig.add_subplot(overlay[0]))
        self.alpha_gauge = AlphaGauge(self.fig.add_subplot(overlay[1]))
        self.wind_gauge = WindGauge(self.fig.add_subplot(overlay[2]))
        self.gust_gauge = GustGauge(self.fig.add_subplot(overlay[3]))

        strips = grid[0:2, 5].subgridspec(3, 1, hspace=0.35)
        self.strips = [
            self._build_strip(strips[0], "TAS  m/s", target_speed, 40.0),
            self._build_strip(strips[1], "ALT  m", target_altitude, 600.0),
            self._build_strip(strips[2], "HDG  deg", target_heading, 90.0, xlabel=True),
        ]
        self._build_status(grid)
```

`Panel.artists` returns every instrument's artists plus the strips and status, keeping the horizon
polygons before the symbol as today. `Panel.update` builds one `Readout` and fans it out:

```python
    def update(self, t, sim, controls, mode, accel, reference, field=None) -> list:
        air = sense(sim.state, sim.wind_ned)
        r = Readout(
            t=t, air=air, accel=accel, controls=controls, reference=reference,
            mode=mode, wind_ned=np.asarray(sim.wind_ned, dtype=float),
            omega_gust=np.asarray(sim.omega_gust, dtype=float), field=field,
        )
        ...
```

`LiveSim.frame` computes `accel` via `accelerometers(...)` and passes `self.ctl.manual.reference` and
`self.field_range(self.sim.state) if self.field_range else None`.

Drop `ax_trace`, `trace` and `trace_now` and their references, including in
`test_the_animation_runs_blitted_and_moves_its_artists` and
`test_axis_limits_never_move_because_blitting_would_not_notice` — extend the latter's axis list to
cover every new axes instead:

```python
    axes = [ax for ax in panel.fig.axes]
    before = [(ax.get_xlim(), ax.get_ylim()) for ax in axes]
```

- [ ] **Step 8: Run the suite**

```bash
.venv/Scripts/python.exe -m pytest atisim/tests -q
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add atisim/panel.py atisim/tests/test_panel.py
git commit -m "Re-lay-out the panel as a basic T with a flight-test overlay"
```

---

## Task 7: Proportional stick

**Files:**
- Modify: `atisim/panel.py`
- Test: `atisim/tests/test_panel.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_a_one_step_tap_gives_less_than_full_deflection(live):
    """The whole point of ramping: a tap is now a small input, not a full one."""
    press(live.panel, "up")
    live.panel.stick.step(live.dt)
    assert 0.0 < live.panel.pilot_input().pitch < 1.0


def test_holding_a_key_reaches_full_travel(live):
    press(live.panel, "up")
    for _ in range(int(1.0 / live.dt)):
        live.panel.stick.step(live.dt)
    assert live.panel.pilot_input().pitch == pytest.approx(1.0)


def test_releasing_springs_the_stick_back_to_centre(live):
    press(live.panel, "up")
    for _ in range(int(1.0 / live.dt)):
        live.panel.stick.step(live.dt)
    release(live.panel, "up")
    for _ in range(int(1.0 / live.dt)):
        live.panel.stick.step(live.dt)
    assert live.panel.pilot_input().pitch == pytest.approx(0.0)


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
        p = panel_mod.Panel(targets, window=20.0, fps=20.0, aircraft_name="boeing747")
        live = panel_mod.LiveSim(
            sim, ctl, targets, GAINS, MGAINS, AC, p, dt=DT, real_time=False
        )
        press(p, "up")
        for _ in range(frames):
            live.advance(steps_per_frame * DT)
        return p.pilot_input().pitch

    assert fly(1, 10) == pytest.approx(fly(10, 1))
    assert fly(1, 10) > 0.0
```

- [ ] **Step 2: Run to verify they fail**

```bash
.venv/Scripts/python.exe -m pytest atisim/tests/test_panel.py -q -k stick
```

Expected: FAIL — `Panel` has no `stick`.

- [ ] **Step 3: Implement**

In `atisim/panel.py`:

```python
# How fast a hand moves a spring-centred stick: full travel in 0.4 s. A DECLARED
# figure, not a measured one. It lives here rather than in ManualGains because
# it is a property of the input device, not of the aircraft -- manual.manual
# stays a pure function of stick position.
STICK_RATE = 2.5  # per second


class Stick:
    """Three ramped surface axes. Held keys ramp toward the demand, released
    keys spring back to centre, both at STICK_RATE.

    Stepped once per PHYSICS step, never per frame. Throttle and trim are not
    here: they are rate-integrating keys that pass a demand straight to
    `manual`, which integrates them, so ramping them would put a lag on top of
    an integrator.
    """

    AXES = ("pitch", "roll", "yaw")

    def __init__(self) -> None:
        self.position = {axis: 0.0 for axis in self.AXES}

    def step(self, demand: dict, dt: float) -> None:
        for axis in self.AXES:
            target = demand[axis]
            move = STICK_RATE * dt
            delta = target - self.position[axis]
            self.position[axis] += float(np.clip(delta, -move, move))
```

`Panel` gains `self.stick = Stick()`, and `pilot_input` splits into the raw key demand and the ramped
position:

```python
    def key_demand(self) -> dict:
        """Where the keys are asking the stick to go, before the ramp."""
        axes = {"pitch": 0.0, "roll": 0.0, "yaw": 0.0, "throttle": 0.0, "trim": 0.0}
        for key in self.held:
            if key in KEYMAP:
                axis, sign = KEYMAP[key]
                axes[axis] += sign
        return {k: float(np.clip(v, -1.0, 1.0)) for k, v in axes.items()}

    def step_stick(self, dt: float) -> None:
        self.stick.step(self.key_demand(), dt)

    def pilot_input(self) -> PilotInput:
        demand = self.key_demand()
        return PilotInput(
            pitch=self.stick.position["pitch"],
            roll=self.stick.position["roll"],
            yaw=self.stick.position["yaw"],
            throttle=demand["throttle"],
            trim=demand["trim"],
        )
```

The test above calls `live.panel.stick.step(live.dt)` with one argument — change those calls to
`live.panel.step_stick(live.dt)` so the demand comes from the held keys.

`LiveSim.advance` calls `self.panel.step_stick(self.dt)` **inside** the while loop, before
`man.update`.

- [ ] **Step 4: Run the suite**

```bash
.venv/Scripts/python.exe -m pytest atisim/tests -q
```

Expected: PASS. `test_opposite_keys_cancel_and_unknown_keys_are_ignored` and
`test_held_keys_become_a_stick_position` from Task 5 now test `key_demand` rather than
`pilot_input`; update them to call `key_demand()` and say why in a comment.

- [ ] **Step 5: Commit**

```bash
git add atisim/panel.py atisim/tests/test_panel.py
git commit -m "Ramp the keyboard stick, once per physics step"
```

---

## Task 8: Pitch trim

**Files:**
- Modify: `atisim/manual.py`, `atisim/panel.py`
- Test: `atisim/tests/test_manual.py`, `atisim/tests/test_panel.py`

- [ ] **Step 1: Write the failing tests**

In `atisim/tests/test_manual.py`:

```python
def test_trim_moves_where_a_released_stick_settles():
    """Trim writes to the REFERENCE, not to the current deflection. If it wrote
    to `controls` the effect would vanish the moment the stick was released."""
    state, controls = trimmed()
    ms = man.take_control(controls)
    before = float(ms.reference.elevator)

    for _ in range(50):
        _, ms = man.manual(ms, PilotInput(trim=1.0), MGAINS, AC, jnp.array(0.02))
    assert float(ms.reference.elevator) > before

    released, ms = man.manual(ms, man.NEUTRAL, MGAINS, AC, jnp.array(0.02))
    assert float(released.elevator) == pytest.approx(float(ms.reference.elevator), abs=1e-9)


def test_trim_respects_the_elevator_limit():
    state, controls = trimmed()
    ms = man.take_control(controls)
    for _ in range(20000):
        _, ms = man.manual(ms, PilotInput(trim=1.0), MGAINS, AC, jnp.array(0.02))
    assert float(ms.reference.elevator) <= float(AC.elevator_limit) + 1e-12


def test_trim_here_snaps_the_reference_to_the_live_deflections():
    state, controls = trimmed()
    ctl = man.start(sense(state), controls, hold_targets(), GAINS, AC)
    held, ctl = man.manual_step(ctl, PilotInput(pitch=1.0), MGAINS, AC, jnp.array(0.02))

    assert float(ctl.manual.reference.elevator) != pytest.approx(float(held.elevator))
    ctl = man.trim_here(ctl)
    assert float(ctl.manual.reference.elevator) == pytest.approx(float(held.elevator))
```

`manual_step` above is just `man.update(ctl, air, pilot, targets, gains, mgains, ac, dt)` — use the
real signature rather than inventing a helper:

```python
    held, ctl = man.update(
        ctl, sense(state), PilotInput(pitch=1.0), hold_targets(),
        GAINS, MGAINS, AC, jnp.array(0.02),
    )
```

In `atisim/tests/test_panel.py`, the test that shows the payoff:

```python
def test_trim_here_then_hands_off_holds_altitude(live):
    """The reason trim exists. manual.py's own docstring says the reference goes
    stale after a manoeuvre and the aircraft drifts; trimming at the new
    condition is what stops it."""
    press(live.panel, "up")
    for _ in range(150):
        live.frame()
    release(live.panel, "up")
    for _ in range(100):
        live.frame()

    live.request_trim_here()
    live.frame()
    settled = -float(live.sim.state.pos_ned[2])
    for _ in range(int(30.0 / DT / 10)):
        live.advance(10 * DT)

    assert abs(-float(live.sim.state.pos_ned[2]) - settled) < 250.0
```

- [ ] **Step 2: Run to verify they fail**

```bash
.venv/Scripts/python.exe -m pytest atisim/tests/test_manual.py atisim/tests/test_panel.py -q -k trim
```

Expected: FAIL — `PilotInput` has no `trim`, `manual` has no `trim_here`.

- [ ] **Step 3: Implement `manual.py`**

```python
class PilotInput(NamedTuple):
    """Stick and throttle demand, each in [-1, 1]. Zero is centred/no change."""

    pitch: Array = 0.0
    roll: Array = 0.0
    yaw: Array = 0.0
    throttle: Array = 0.0
    trim: Array = 0.0  # +1 nose-up: moves the stick's centring point, not the surface
```

`ManualGains` gains a field:

```python
    trim_rate: Array  # rad/s of elevator REFERENCE per unit trim input
```

In `manual()`, before the surfaces are computed:

```python
    # Trim moves the point a released stick returns to, which is what makes it a
    # trim system rather than a second elevator. Nose-up trim is trailing-edge
    # UP, i.e. a negative elevator deflection in this repo's sign convention --
    # the same convention as the stick, where +pitch is nose-down.
    reference = ms.reference._replace(
        elevator=jnp.clip(
            ms.reference.elevator - pilot.trim * gains.trim_rate * dt,
            -ac.elevator_limit,
            ac.elevator_limit,
        )
    )
```

Use `reference` in place of `ms.reference` in all three `surface(...)` calls, and return
`ManualState(controls=out, reference=reference)`.

Add, beside `toggle`:

```python
def trim_here(ctl: Controller) -> Controller:
    """Snap the stick's centring point to the deflections now reaching the plant.

    Not a control any real aircraft has, but it is what the reference concept
    means made explicit: "the aircraft is doing what I want -- hold this". A
    no-op in AUTOPILOT, because `toggle` already reseeds the reference from the
    live deflections on the way out.
    """
    if ctl.mode is not Mode.MANUAL:
        return ctl
    return ctl._replace(manual=take_control(ctl.manual.controls))
```

Add `trim_rate` to the three `ManualGains` instances, derived from the spec's rule — one second of
held trim moves the reference by about a quarter of what full stick commands:

```python
# trim_rate: one second of held trim is about a quarter of full stick travel.
# Finer than the stick and it is useless; coarser and it is unflyable.
# 747:      full stick 0.25 * 25 deg = 6.25 deg  ->  1.5 deg/s
BOEING747_MANUAL = ManualGains(..., trim_rate=jnp.array(1.5 * DEG2RAD))
# cherokee: full stick 0.06 * 25 deg = 1.50 deg  ->  0.37 deg/s
CHEROKEE_MANUAL = ManualGains(..., trim_rate=jnp.array(0.37 * DEG2RAD))
# cessna:   full stick 0.08 * 25 deg = 2.00 deg  ->  0.50 deg/s
CESSNA172_MANUAL = ManualGains(..., trim_rate=jnp.array(0.50 * DEG2RAD))
```

`manual.py` needs `from atisim.units import DEG2RAD`.

- [ ] **Step 4: Implement the panel side**

```python
KEYMAP = {
    ...
    "]": ("trim", +1.0),   # nose up
    "[": ("trim", -1.0),   # nose down
}
TRIM_HERE_KEY = "t"

HELP = (
    "arrows  stick (up = nose down)   , .  rudder    -  =  throttle\n"
    "[  ]    trim (nose down/up)      t    trim here   a    autopilot"
)
```

`Panel` gains `self._trim_here_requested`, set in `on_press` when `event.key == TRIM_HERE_KEY`, and
`take_trim_here_request()` mirroring `take_toggle_request` exactly. `LiveSim.advance` consumes it
inside the loop:

```python
            if self.panel.take_trim_here_request():
                self.ctl = man.trim_here(self.ctl)
```

Add `LiveSim.request_trim_here()` setting `self.panel._trim_here_requested = True`, so the test does
not have to synthesise a key event to reach it.

The status block gains the trim position:

```python
            f"trim {float(reference.elevator) * RAD2DEG:+6.2f}   "
```

- [ ] **Step 5: Run the suite**

```bash
.venv/Scripts/python.exe -m pytest atisim/tests -q
```

Expected: PASS. If `test_neutral_stick_from_trim_holds_the_trimmed_condition` fails, the trim sign is
inverted — a `PilotInput` with `trim=0.0` must leave the reference untouched to machine precision.

- [ ] **Step 6: Commit**

```bash
git add atisim/manual.py atisim/panel.py atisim/tests/test_manual.py atisim/tests/test_panel.py
git commit -m "Add pitch trim and a trim-here key"
```

---

## Task 9: `fly.py` flags and the field-range readout

**Files:**
- Modify: `scripts/fly.py`, `atisim/panel.py`
- Test: `atisim/tests/test_panel.py`

- [ ] **Step 1: Write the failing test**

```python
def test_the_vortex_range_is_zero_at_a_core_and_closes_northbound():
    """The cores are infinite east-west lines, so the distance is a north
    distance and there is no bearing to report."""
    from atisim.state import State

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
    assert at(1500.0).distance == pytest.approx(500.0, abs=1e-9)  # second core now
    assert at(0.0).bearing is None
    assert at(0.0).closing > 0.0  # flying north, closing on it
```

- [ ] **Step 2: Run to verify it fails**

```bash
.venv/Scripts/python.exe -m pytest atisim/tests/test_panel.py -q -k vortex_range
```

Expected: FAIL — `panel` has no `vortex_range`.

- [ ] **Step 3: Implement the range callables in `panel.py`**

```python
def vortex_range(array, *, label: str):
    """Distance to the next VortexArray core ahead, as a State -> FieldRange.

    The cores are infinite line vortices running east-west: `vortex_wind` reads
    only pos_ned[0] and pos_ned[2], and the induced wind has no east component.
    So the distance is a perpendicular north distance regardless of heading, and
    there is no bearing -- you cannot miss a line by turning.
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
            float((dn * vel_ned[0] + de * vel_ned[1]) / distance) if distance > 1e-9 else 0.0
        )
        return FieldRange(
            label=label, distance=distance, closing=closing,
            bearing=float(np.arctan2(de, dn)),
        )

    return ranged
```

Show it in the status block, in the terms each field supports:

```python
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
```

- [ ] **Step 4: Wire up `scripts/fly.py`**

Add the arguments, reusing `vortex.py`'s wording so the two agree:

```python
parser.add_argument(
    "--wind", default="none",
    choices=["none", "hannibal", "morton", "updraft"],
    help="wind field to fly through. The two named cases are Parks et al. 1985.",
)
parser.add_argument(
    "--lead-in", type=float, default=40.0,
    help="distance to the first core, in core radii. Below ~12 the 1/r far field "
         "launches the aircraft out of equilibrium and contaminates the encounter.",
)
parser.add_argument(
    "--sharpness", type=float, default=6.0,
    help="updraft edge sharpness. DECLARED MODELLING PARAMETER, not source data.",
)
```

Build the field. The field is offset ahead; the aircraft still starts at the origin:

```python
wind_model, field_range = wind.zero_wind, None
if args.wind in wind.PARKS_CASES:
    case = wind.PARKS_CASES[args.wind]
    lead = args.lead_in * case["r0"]
    # The cores go AHEAD of the aircraft rather than the aircraft starting
    # behind them: physically identical, since only relative geometry enters the
    # field, but every run still begins at the origin so the ground track and
    # the saved .npz stay comparable across runs.
    array = wind.VortexArray(
        north=jnp.array([lead, lead + case["spacing"]]),
        down=jnp.array([-H, -H]),
        r0=jnp.array(case["r0"]),
        v0=jnp.array(case["v0"]),
    )
    wind_model = wind.field_model(lambda p: wind.vortex_wind(p, array))
    field_range = panel_mod.vortex_range(array, label=f"vortex {args.wind}")
elif args.wind == "updraft":
    radius = 0.5 * wind.UPDRAFT_SECONDS * V
    column = wind.UpdraftColumn(
        north=jnp.array(args.lead_in * radius / 10.0 + radius),
        east=jnp.array(0.0),
        w0=jnp.array(wind.UPDRAFT_W0),
        radius=jnp.array(radius),
        sharpness=jnp.array(args.sharpness),
    )
    wind_model = wind.field_model(lambda p: wind.updraft_wind(p, column))
    field_range = panel_mod.updraft_range(column, label="updraft column")

traj = panel_mod.run_live(
    sim, ctl, targets, gains, mgains, ac,
    dt=args.dt, fps=args.fps, window=args.window,
    wind_model=wind_model, field_range=field_range,
)
```

Update the module docstring's key list to include `[ ] t`, and note that `--wind` exists.

- [ ] **Step 5: Run the suite and fly it**

```bash
.venv/Scripts/python.exe -m pytest atisim/tests -q
```

Then, interactively — this is the only step a test cannot cover:

```bash
.venv/Scripts/python.exe scripts/fly.py --aircraft boeing747 --wind hannibal
```

Expected: the panel opens, the status line counts the distance down from about 7300 m, and the gust
bars and load factor move as the first core arrives. Check the α gauge stays in the green band; if it
goes red, the encounter is outside the model's linear range and PROJECT.md §7 applies.

- [ ] **Step 6: Commit**

```bash
git add scripts/fly.py atisim/panel.py atisim/tests/test_panel.py
git commit -m "Fly through a cited wind field, with a field-range readout"
```

---

## Task 10: Update PROJECT.md

**Files:**
- Modify: `docs/PROJECT.md`

- [ ] **Step 1: §2 architecture table**

Add a `panel.py` row and amend `viz.py`'s:

```
| `panel.py` | live cockpit, instruments, `Stick`, `LiveSim`, `run_live` | basic T + test overlay; takes a `wind_model` |
| `viz.py` | `Trajectory`, `Recorder`, `derived`, `post_flight` | the log and the post-flight figure only |
```

- [ ] **Step 2: §4 evidence ledger**

Add a section for anything measured while flying. Do not invent numbers — record what was actually
observed in Task 9 step 5, or write nothing.

- [ ] **Step 3: §10 running it**

- `fly.py`'s flag list gains `--wind --lead-in --sharpness`.
- The key list gains `[ ] t`, and the stick description changes: the arrows now ramp rather than
  snapping to full travel.
- The "which paths are trustworthy under wind" table: `fly.py` / `panel.py` move to **Correct** —
  they take a wind model and sense from it — and the "Still air only" warning on the `fly.py` row
  goes.

- [ ] **Step 4: §9 session log**

Add an entry at the top following the template: what changed and why, numbers measured against what,
anything found wrong in earlier work, and what was deliberately not done. The "deliberately not
done" list is the spec's: `ManualGains` not re-tuned (and the reason it now could be), no Mach, no
control-position display, no re-arm key, no uniform wind, `test_viz.py`'s
`test_derived_agrees_with_the_aero_module` left alone as a pre-existing structurally-cannot-fail test.

Record the final test count.

- [ ] **Step 5: Final full run**

```bash
.venv/Scripts/python.exe -m pytest atisim/tests -q
```

Expected: PASS, with the count recorded in §9.

- [ ] **Step 6: Commit**

```bash
git add docs/PROJECT.md
git commit -m "Record the free-air flying interface in PROJECT.md"
```

---

## Self-review notes

**Spec coverage.** Every spec section maps to a task: sensing → 1 and 2; panel and layout → 5 and 6;
wind in the live loop → 4 and 9; control → 7 and 8; cited data → 3; docs → 10.

**One deliberate substitution.** The spec's test table lists "n_z = 1/cos φ in a steady coordinated
turn". That is replaced by `test_specific_force_matches_the_forces_in_all_three_axes`, which is exact
to 1e-12 rather than needing a discovered tolerance, is independent of the code under test, and pins
`n_x` and `n_y` as well — the two components that are actually new. The only turning-trim solve in the
project lives inside `test_conservation.py`, a §4 baseline file that this work must not touch.

**Task 4 before Task 5** so that the panel split is a pure move. Task 5's verification is that the
collected test count does not change.
