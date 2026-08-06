# Free-air flying interface — Design Spec

Date: 2026-08-06

## Purpose

Turn the live flying path from a still-air demonstration into an instrument you can fly research
with. Three things are wrong with it today, and they compound:

1. **`run_live` takes no wind model.** `LiveSim.advance` calls `step(sim, controls, dt, ac)` and gets
   the `zero_wind` default. The Parks vortex array and the Wingrove updraft — the only things the
   project is building toward — cannot be hand-flown at all. PROJECT.md §10's "still air only" note
   reads as the session-5 sensing bug; this plumbing gap is separate and still open.
2. **The panel is missing half the basic T.** No vertical speed, no slip indicator. Both are
   six-pack instruments, and without a rate cue you hold altitude by chasing the altimeter.
3. **Nothing on the panel reports the physics the project measures.** `dynamics.load_factor` exists
   and `vortex_viz.py` uses it; the live panel does not. Air-relative α is a number in a text block
   with no indication of where PROJECT.md §7's linear-aero ceiling lies.

The end state: a basic-T cockpit with a flight-test overlay, flyable through a cited wind field,
with a proportional stick and working pitch trim.

Out of scope, by decision: nonlinear aero, out-the-window view, Mach display, control-position
display, joystick/gamepad input, re-tuning `ManualGains`, and anything touching the Cessna.

## Decisions taken during brainstorming

| Question | Decision |
|---|---|
| What is the interface for | **Cockpit core plus test overlay.** Basic-T primary instruments, flight-test quantities alongside. |
| Panel layout | **Re-laid-out.** The 3D trace is dropped; the three time-history strips stay. |
| Live wind | **Plumbed, with a display.** `--wind` on `fly.py`, wind and gust rate on the panel. |
| Input | **Keyboard, proportional, plus trim.** Held keys ramp; pitch trim on two keys plus a trim-here key. |
| `ManualGains` | **Not re-tuned this session.** The ramp delivers fine control on its own; re-tuning is hand work verifiable only by flying, and `test_manual.py`'s response tests are too loose to pin it. |
| Code structure | **New `flightsim/panel.py`.** `viz.py` is 791 lines doing three jobs; this work would take it past 1100. |
| Slip indicator | **Lateral specific force `n_y`, not β.** A ball is a pendulum. See below. |
| AoA red band | **A declared modelling constant**, cited to PROJECT.md §7, not to a source table. |

### Why the slip indicator is not β

A slip ball is a bead in a curved tube — it measures lateral specific force, not sideslip angle. A
vane measures sideslip. The two agree in steady coordinated flight and diverge everywhere else, so
wiring β into a ball is the same shape of mistake as §6(a): correct in the easy case, quietly wrong
in the interesting one. This is a turbulence simulator; the interesting case is the one that matters.

## Sensing

Two sensor groups, because a real aircraft has two boxes. The air-data computer knows nothing about
mass or control deflection; the accelerometer package needs both.

```python
# dynamics.py — load_factor factored, not replaced
def specific_force(state, controls, ac, wind_ned, omega_gust) -> Array   # (3,) body axes, in g
def load_factor(state, controls, ac, wind_ned, omega_gust) -> Array      # = -specific_force(...)[2]
```

`load_factor` keeps its docstring and its reasoning about inverting `derivatives` rather than
recomputing `force / mass` — that argument now lives in `specific_force` and `load_factor` becomes
one line. Both existing tests in `test_dynamics.py` must pass **unchanged**.

```python
# sensors.py
class AirData(NamedTuple):
    ...                       # unchanged fields, in order
    vertical_speed: Array     # NEW. m/s, positive UP. INERTIAL.

class Accelerations(NamedTuple):
    n_x: Array                # g, body axes
    n_y: Array
    n_z: Array

def accelerometers(state, controls, ac, wind_ned, omega_gust) -> Accelerations
```

`vertical_speed` is appended, so positional construction in `sense` is the only call site that
changes and every `air.field` consumer is untouched. It is **inertial**: a barometric VSI measures
the rate of change of static pressure, and an updraft that carries the aircraft up does change its
geometric height. It comes from `quat_to_dcm(state.quat) @ state.vel_body`, negated for up.

`vertical_speed` belongs in `AirData` because `sense(state, wind_ned)` can compute it. The
accelerations do not, because they need `controls` and `ac`, and dragging `dynamics` into `sense`
to get them would be worse than a second function.

### Ball sign convention

Display deflection is **`-n_y`**. The bead settles where the tube's normal force supplies the
aircraft's specific force; the tube curves upward, so a bead displaced to +y is pushed toward −y.
A specific force to the right therefore puts the ball left.

This derivation is checked, not trusted. The test asserts the pilot-facing rule — the ball indicates
the rudder that would reduce the sideslip, "step on the ball" — so if the sign is backwards the test
goes red during implementation and the constant flips.

## Panel

`flightsim/panel.py` takes the live cockpit and `LiveSim`. `viz.py` keeps `Trajectory`, `Recorder`,
`save`/`load`, `derived` and `post_flight` — the log and the post-flight figure, which is one job.

### Layout

```
GridSpec(3, 6)
  width_ratios  = (0.30, 1.05, 0.30, 0.17, 0.70, 0.92)     ASI  ADI  ALT  VSI  overlay  strips
  height_ratios = (1.00, 0.18, 0.34)                       main  heading  status

  [0, 0] airspeed tape          [0, 1] attitude          [0, 2] altitude tape   [0, 3] VSI
  [1, 0:4] heading tape
  [0:2, 4] test overlay, nested 4x1:  n_z | alpha | wind | gust rate
  [0:2, 5] time strips, nested 3x1:   TAS | ALT | HDG
  [2, 0:5] status block         [2, 5] key help
```

The attitude cell carries the horizon, the pitch ladder, the aircraft symbol, the roll pointer with
the slip index beneath it (PFD convention), and a flight path vector at `(-β, -α)` from the symbol —
air-relative, which is the point.

### Blitting

The existing constraint holds and `test_axis_limits_never_move_because_blitting_would_not_notice`
still guards it. Tapes are compatible with it: the *scale* slides past a fixed pointer, so tick
positions and tick label text are animated artists and the axis limits never move. Every tape
pre-allocates a fixed number of tick artists.

Fixed spans are declared choices in one table beside `MANUAL_GAINS`, shared by a tape and its strip
so the two never disagree about scale:

| Instrument | Span | Centre |
|---|---|---|
| airspeed | ±40 m/s | `targets.airspeed` (as today) |
| altitude | ±600 m | `targets.altitude` (as today) |
| heading | ±90° | `targets.heading` (as today) |
| VSI | ±20 m/s (747), ±10 m/s (cherokee, cessna172) | 0 |
| `n_z` | −1 to +3 g | 1.0 marked |
| α | 0 to 15° | green to 10, amber 10–12, red beyond |
| wind | 0 to 40 m/s, direction on a full circle | — |
| gust rate | ±0.15 rad/s | 0 |

Values outside a span clip against the edge and the numeric readout stays correct — the existing
rule, unchanged. The α and gust spans are declared; the rest follow the targets or the existing
panel.

### Instrument units

One shape for all of them, so a gauge can be tested without building a figure:

```python
class Readout(NamedTuple):
    """One frame's worth of everything the panel shows."""
    t: float
    air: AirData
    accel: Accelerations
    controls: Controls
    reference: Controls        # the stick's centring point, for the trim readout
    mode: Mode
    wind_ned: np.ndarray
    omega_gust: np.ndarray
    field: FieldRange | None   # None when flying still air

class FieldRange(NamedTuple):
    """Where the wind field is, in the terms its own geometry supports."""
    label: str                 # "vortex hannibal core 1", "updraft column"
    distance: float            # m. North distance for a line vortex; slant range for a column.
    closing: float             # m/s, rate of change of `distance`
    bearing: float | None      # rad. None for a line vortex — see below.

# each instrument
def build(ax, ...) -> Instrument         # pre-allocates artists, sets fixed limits
Instrument.artists -> list
Instrument.update(r: Readout) -> None    # set_data only
```

`Panel` becomes a layout plus a list of instruments, and `Panel.update` builds one `Readout` and
fans it out.

### The overlay

- **`n_z`** — bar with 1.0 marked and a peak-excursion hold. The y-axis of Wingrove & Bach Fig. 8,
  which nothing in the live path has ever produced.
- **α, air-relative** — bar with a green/amber/red band. `ALPHA_LINEAR_DEG = 10.0` and
  `ALPHA_INVALID_DEG = 12.0`, **declared modelling constants** cited to PROJECT.md §7's statement
  that "any encounter driving α past ~10-12 deg reports lift the sources say is not there", not to
  any source table. They are named as declarations in the same way `--sharpness` is.
- **Wind** — magnitude, direction and the groundspeed it implies. Legitimate cockpit information:
  with no sensor noise, ground velocity minus air velocity *is* the wind, so a real aircraft could
  compute this.
- **Gust rate** — `p/q/r` bars, labelled **sim truth**. `omega_gust` is a gradient across the span
  and chord. No instrument can sense it, so it is never presented as though one could.

## Wind in the live loop

```python
def run_live(sim, ctl, targets, gains, mgains, ac, *, dt, fps, window,
             wind_model=zero_wind, field_range=None) -> Trajectory
```

`wind_model` is threaded to `LiveSim`, which passes it to `step(sim, controls, dt, ac,
self.wind_model)`. `field_range` is a separate, optional `State -> FieldRange` callable built by
`fly.py` alongside the field — separate because the panel needs the geometry in a form the wind model
does not carry, and because a wind model with no meaningful "where is it" (a future Dryden layer) can
pass `None` rather than being forced to invent one.

**`step` has `static_argnames=("wind_model",)`.** `run_live`'s jit warm-up must warm with the actual
model object, not the default — otherwise the first frame compiles the real model, stalls, and drops
its backlog straight out of the real-time budget. That is the whole reason the warm-up exists.

`fly.py` gains `--wind {none,hannibal,morton,updraft}`, `--lead-in` and `--sharpness`, with
`vortex.py`'s meanings and warnings. Below ~12 core radii the 1/r far field launches the aircraft out
of equilibrium (§9 session 3), and `--sharpness` is a declared parameter, not source data.

**The field is offset ahead; the aircraft is not moved.** `vortex.py` starts the aircraft at
`-lead`; here the cores go at `lead` and `lead + spacing` instead. Physically identical — only
relative geometry enters the field — but every run still begins at the origin, so nothing about
`trimmed_state`, `init_sim`, the recorder or the ground track changes.

### The field-range readout

The two fields have different geometry and get different readouts, because pretending otherwise
would be wrong:

- **`VortexArray` cores are infinite line vortices running east–west.** `vortex_wind` reads only
  `pos_ned[0]` and `pos_ned[2]`, and the induced wind has no east component. You cannot miss a core
  by turning. The readout is the north distance to the next core ahead plus the closure rate. There
  is no bearing to report.
- **`UpdraftColumn` is a point** at `(north, east)` with a radius, so range and bearing are correct
  for it.

## Control

### Proportional stick

A `Stick` in `panel.py` holds three ramped axes. Held keys ramp toward ±1 and released keys spring
back to 0, both at `STICK_RATE = 2.5` per second — full travel in 0.4 s, a **declared** figure for
how fast a hand moves a stick. It lives with the keyboard because it is a property of the input
device, not of the aircraft, and `manual.manual` stays a pure function of stick position.

**It is stepped once per physics step, not once per frame.** Stepping it in `LiveSim.frame` would
make the control feel depend on the render rate, and the render rate is corrected against wall clock.
`LiveSim.advance` steps it inside the loop.

**Only the three surface axes ramp.** Throttle and trim are rate-integrating keys, not stick
positions: they already pass a ±1 demand straight through to `manual()`, which integrates it at
`throttle_rate` / `trim_rate`. Ramping them would put a second lag on top of an integrator. The
throttle is not sprung either, because a lever stays where it is left, and trim is not sprung for the
same reason — a trim wheel that recentred itself would be worse than no trim at all.

### Trim

```python
class PilotInput(NamedTuple):
    pitch; roll; yaw; throttle
    trim: Array = 0.0          # NEW, [-1, 1]

class ManualGains(NamedTuple):
    ...
    trim_rate: Array           # NEW, rad/s of elevator reference

def trim_here(ctl: Controller) -> Controller   # snap reference to the live deflections
```

`manual()` moves `ManualState.reference.elevator` by `pilot.trim * trim_rate * dt`, clipped to
`ac.elevator_limit`, and the elevator surface is then computed from the updated reference. This is
the honest fix for what the module's own docstring already admits: "It is not a trim system: after a
manoeuvre the reference is stale and the aircraft will drift."

`trim_here` is edge-triggered and consumed exactly as the autopilot toggle is. It is a no-op in
`AUTOPILOT` mode, because `toggle` already reseeds the reference from the live deflections on the
way out.

**Rule for choosing `trim_rate`**, so the three numbers are derived rather than invented: one second
of held trim moves the elevator reference by about a quarter of what full stick commands. Trim finer
than the stick is useless; trim coarser than the stick is unflyable.

| Aircraft | full stick | `trim_rate` |
|---|---|---|
| boeing747 | 0.25 × 25° = 6.25° | 0.027 rad/s (1.5 °/s) |
| cherokee | 0.06 × 25° = 1.50° | 0.0065 rad/s (0.37 °/s) |
| cessna172 | 0.08 × 25° = 2.00° | 0.0087 rad/s (0.50 °/s) |

Keys: `[` nose-down, `]` nose-up, `t` trim here. Trim position shows in the status block, which is
what makes the reference concept self-teaching.

## Cited data

`CASES`, `UPDRAFT_W0` and `UPDRAFT_SECONDS` move from `scripts/vortex.py` into `wind.py`, carrying
their Parks 1985 and Wingrove & Bach 1994 citations. Both scripts import them.

A script is not importable, so `fly.py` would otherwise have to restate identified source parameters
— and a number restated in two places is a number that will eventually disagree with itself. This is
the one file the work does not strictly force us to touch, and it is touched for that reason alone.

## Tests

Applying §6's lesson: **a test that cannot fail is not a test.** Every item below names what breaks
it.

| Test | Goes red when |
|---|---|
| A live run through a non-zero field differs from still air | `wind_model` is accepted and silently not applied — the (a)/(b)-shaped risk, and the one that would make this whole feature a no-op |
| `run_live` with no wind model is bit-identical to today | the split or the re-layout perturbs the plant |
| The ball indicates the rudder that reduces the sideslip | β is wired into the ball, or the sign is backwards |
| `n_z` = 1/cos φ in a steady coordinated turn | `specific_force` breaks `load_factor`'s meaning |
| `vertical_speed` matches −(dcm @ vel_body)[2], computed independently | the VSI is fed something that is not vertical speed |
| The ramp reaches the same position in 1 frame or 10 frames of equal total steps | the stick is stepped per frame |
| A one-step tap gives less than full deflection | the ramp is not applied at all |
| Trim moves where a released stick settles | trim writes to `controls` instead of `reference` |
| Trim-here, then 30 s hands-off, holds altitude in band | the reference is not actually snapped |
| Distance-to-core is zero at a core and decreases monotonically northbound | the field offset and the readout disagree |
| Axis limits never move (existing test, extended over the new axes) | a tape autoscales |
| Still-air sensing unchanged to atol 1e-12 (existing) | `AirData` gains a field wrongly |

Test files: `test_panel.py` is new and takes the live-panel half of `test_viz.py`. `test_viz.py`
keeps `Trajectory`, `derived` and `post_flight`. `test_sensors.py`, `test_dynamics.py` and
`test_manual.py` gain cases.

**§4's validated baseline is off limits**: `test_conservation.py`, `test_cr2144_modes.py`,
`test_drag_polar.py`, `test_navion.py`, `test_trim.py`. All are still-air statements and none of this
work touches the derivative chain or the integrator. If one moves, something real broke — stop and
say so rather than adjusting it.

## Order of work

Steps 1–3 are independent; 4 and 5 are independent of each other.

```
1. dynamics.specific_force, load_factor as a wrapper   -> verify: both existing test_dynamics
                                                                  tests pass UNCHANGED; new
                                                                  n_z = 1/cos(phi) turn test
2. AirData.vertical_speed, sensors.accelerometers      -> verify: VSI against an independently
                                                                  computed dcm @ vel_body;
                                                                  still-air sensing to 1e-12
3. Cited case data into wind.py, vortex.py imports     -> verify: scripts/vortex.py --case
                                                                  hannibal still gives PROJECT.md
                                                                  §4's first-core dtheta 2.24 deg
4. wind_model through run_live / LiveSim, no display   -> verify: live run through a field differs
                                                                  from still air; still air
                                                                  bit-identical to today
5. panel.py split, panel moved unchanged               -> verify: the moved live tests pass with
                                                                  NO assertion changes
6. Re-layout to the basic T, new instruments           -> verify: axis limits never move; one unit
                                                                  test per instrument
7. Ramped stick                                        -> verify: per-physics-step test; one-step
                                                                  tap under full deflection
8. Trim axis and trim-here                             -> verify: released stick settles at the new
                                                                  reference; trim-here then 30 s
                                                                  hands-off holds altitude
9. fly.py flags and the field-range readout            -> verify: distance zero at a core,
                                                                  monotone northbound
10. PROJECT.md                                         -> verify: sections 2, 4, 9, 10 all edited
```

Step 5 is deliberately a pure move with no behaviour change, so that step 6's diff is only the
re-layout. Doing both at once would make a regression in either indistinguishable from the other.

## Explicitly not doing

Recorded so it is not rediscovered:

- **`ManualGains` not re-tuned.** The ramp makes higher authorities available for the first time —
  a tap is now a small input, which is why they were geared down. Re-tuning is the next step if
  flying shows it is needed, and it wants tighter response tests than the current ±2°/±45° bounds.
- **No Mach, no control-position display, no re-arm key, no uniform-wind option.** Considered and
  not chosen.
- **No changes to `trim.py`, `aero.py`, `autopilot.py`** or any §4 baseline test file.
- **No out-the-window view.** The Cessna stays out of scope per §5.
- **`test_viz.py`'s `test_derived_agrees_with_the_aero_module` is left alone.** It is the
  structurally-cannot-fail test §6(b) says was replaced; the replacement did land, in
  `test_sensors.py:196`, but the original was never deleted. Flagged, not removed — it is not this
  work's mess.

## Open questions

- **Panel density.** Six columns in a 14×8 figure is dense. If the tapes come out unreadable the
  figure grows rather than the layout shedding an instrument, since every instrument here was chosen
  deliberately.
- **`STICK_RATE` and the three `trim_rate` values are declared, not derived from anything measurable.**
  They are feel numbers. The rule for choosing `trim_rate` is written down so a change is
  a change to the rule, not a fresh guess.
- **Whether the strips duplicate the tapes.** They do not: a tape is the current value against a
  scale, a strip is 60 s of trend, and a phugoid is invisible on the first and obvious on the second.
  If the panel is too dense the strips are still the thing to keep, being the flight-test instrument.

## Escalation rules

Carried from PROJECT.md:

- **Flag, never invent.** `ALPHA_LINEAR_DEG`, `ALPHA_INVALID_DEG`, `STICK_RATE`, `trim_rate` and the
  per-aircraft display spans are declared modelling choices and must say so where they are defined.
- **Do not edit a tolerance to make a test pass.** If a §4 baseline moves, stop.
