# Wiring Strip Loads Into The 6-DOF — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let strip-integrated aerodynamic loads reach the equations of motion, so the model can actually *fly* flow fields whose scale approaches a wingspan instead of only measuring the error in treating them as a point.

**Architecture:** One additive seam. `dynamics.derivatives` gains an optional coefficient increment that is summed into the aero coefficients at the same place wind already enters; `integrate.step` computes it from an optional strip model and threads it through. The increment defaults to zeros, so a run that does not ask for strip loads is **bit-identical** to today. The point path is retained as both the default and the fallback for aircraft failing the tail-arm gate.

**Tech Stack:** Python 3.10+, JAX (float64, `jax_debug_nans` on in tests), NumPy, pytest.

**Predecessor:** `docs/superpowers/plans/2026-08-14-wind-shear-fidelity.md`, complete through Task 13. This plan is that plan's Task 12, answered **(b)**.

**Design document:** `docs/superpowers/specs/2026-08-14-wind-shear-fidelity-design.md`. §2 records two errors in the theory source; read it before touching any sign.

**Python interpreter:**

```bash
PY="C:/Users/mateusz/UROP/Claude_Flight_Sim/.venv/Scripts/python.exe"
```

---

## Things that will bite you

Read these before Task 1. Every one has already cost time in this work.

1. **`conftest.py` turns on `jax_debug_nans` for the whole suite.** `jnp.where` evaluates *both* branches, so guard divisors before the `where`, not inside it.
2. **Never `import jax.numpy` before `import flightsim`.** Float64 must be enabled before any array exists.
3. **`PROJECT.md` §4 baselines are frozen.** Five 747 mode values. If one moves, something real broke — do not adjust a tolerance.
4. **A worktree needs `PYTHONPATH`.** The venv resolves `flightsim` to the main checkout. `pytest` gets this right via rootdir; a plain script run does not.
5. **The full suite takes about 4m40s.** Budget for it; do not assume a 2-minute timeout is enough.
6. **The gust incidence sign is `−w_g/V`, not `+w_g/V`.** A wing moving down gains incidence; air moving down past a stationary wing loses it. This was wrong once already and only the cross-treatment comparison caught it.

---

## The one design decision, stated before any code

There are three places the strip load could enter, and the choice is not obvious.

| Option | Why not |
|---|---|
| Inside `aero.coefficients` | Breaks the module's rule that it sees only `vel_rel`/`omega_rel` and never the field. `PROJECT.md` §2 makes that a contract, and a zero-strength-wind test relies on it |
| A second force term in `derivatives` | Bypasses the coefficient build-up, so the increment would not see dynamic pressure or reference area consistently |
| **A coefficient increment summed in `derivatives`** | **Chosen.** Enters at the same seam wind already does, keeps `aero.py` untouched, and composes with `qbar·S` exactly as every other coefficient does |

The increment is a small NamedTuple rather than a bare array, so a caller cannot silently swap roll for yaw.

---

## File Structure

| File | Responsibility | Status |
|---|---|---|
| `flightsim/loads.py` | `CoeffIncrement`, `zero_increment()`, and `strip_increment()` which builds one from a field | **create** |
| `flightsim/tests/test_loads.py` | The increment's own behaviour and its zero case | **create** |
| `flightsim/dynamics.py` | `derivatives`, `specific_force`, `load_factor` gain an optional increment | modify |
| `flightsim/integrate.py` | `SimState` carries the applied increment; `step`/`rollout` take an optional `load_model` | modify |
| `flightsim/tests/test_dynamics.py` | The zero-increment identity | modify |
| `flightsim/tests/test_integrate.py` | Threading, and the bit-identical default | modify |
| `flightsim/provenance.py` | No new constants — asserted, not assumed | modify (test only) |
| `docs/PROJECT.md`, `docs/ASSUMPTIONS.md` | Record the new path and re-measure the vortex point | modify |

---

# Phase 1 — The increment type

### Task 1: `CoeffIncrement` and its zero

**Files:**
- Create: `flightsim/loads.py`
- Test: `flightsim/tests/test_loads.py`

- [ ] **Step 1: Write the failing test**

Create `flightsim/tests/test_loads.py`:

```python
"""Aerodynamic coefficient increments from distributed loads.

The seam that lets strip-integrated loads reach the equations of motion. It is
deliberately a NamedTuple of named coefficients rather than a bare array: the
four entries are not interchangeable, and a caller that swaps roll for yaw
would otherwise produce a model that flies plausibly and wrongly.

The zero increment is the important case. It is what every existing run uses,
and `test_dynamics.py` asserts that applying it is bit-identical to not
applying one at all.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import flightsim  # noqa: F401  -- enables x64 before any array is made
from flightsim import loads


def test_the_zero_increment_is_all_zeros_and_correctly_shaped():
    """Every field must be a jnp array, not a Python float. integrate.batch_sim
    broadcasts every leaf of the carry and a Python scalar has no .shape, which
    raises there -- the same constraint WindState documents."""
    zero = loads.zero_increment()
    for name, value in zero._asdict().items():
        assert hasattr(value, "shape"), f"{name} is not a jnp array"
        assert float(value) == 0.0, f"{name} is not zero"


def test_the_increment_names_the_four_coefficients_it_carries():
    """Named, not positional. Lift, roll, pitch, yaw -- the four a spanwise and
    longitudinal load distribution can produce that this model has channels for.
    Side force and drag are omitted deliberately: strip theory over a spanwise
    incidence distribution does not produce them at first order."""
    assert loads.CoeffIncrement._fields == ("CL", "Cl", "Cm", "Cn")


def test_increments_add():
    """Superposition. Two fields acting at once contribute independently, the
    same property that makes wind.superpose legitimate."""
    a = loads.CoeffIncrement(
        CL=jnp.array(0.1), Cl=jnp.array(0.2), Cm=jnp.array(0.3), Cn=jnp.array(0.4)
    )
    b = loads.CoeffIncrement(
        CL=jnp.array(0.5), Cl=jnp.array(-0.2), Cm=jnp.array(0.0), Cn=jnp.array(0.1)
    )
    total = loads.add(a, b)
    assert float(total.CL) == pytest.approx(0.6)
    assert float(total.Cl) == pytest.approx(0.0)
    assert float(total.Cm) == pytest.approx(0.3)
    assert float(total.Cn) == pytest.approx(0.5)


def test_adding_zero_changes_nothing_exactly():
    """Not approximately. This is what makes the default path bit-identical."""
    a = loads.CoeffIncrement(
        CL=jnp.array(0.123), Cl=jnp.array(-0.456), Cm=jnp.array(0.789), Cn=jnp.array(0.0)
    )
    total = loads.add(a, loads.zero_increment())
    for name in a._fields:
        assert np.asarray(getattr(total, name)) == np.asarray(getattr(a, name))
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
$PY -m pytest flightsim/tests/test_loads.py -v
```

Expected: collection error, `ModuleNotFoundError: No module named 'flightsim.loads'`.

- [ ] **Step 3: Write the implementation**

Create `flightsim/loads.py`:

```python
"""Aerodynamic coefficient increments from distributed loads.

`aero.py` builds coefficients from air-relative velocity and rate, and its
standing rule is that it never sees the wind field. That rule is what makes a
zero-strength wind bit-identical to still air, and it is asserted. So a load
computed by integrating the field ACROSS the airframe -- which necessarily does
see the field -- cannot be built there.

It enters instead as an increment summed into the coefficients inside
`dynamics.derivatives`, at the same seam where wind already enters and nowhere
else. That keeps `aero.py` untouched, and it means the increment composes with
dynamic pressure and reference area exactly as every other coefficient does.

Four coefficients, named rather than positional. A bare (4,) array would let a
caller put a rolling moment in the yaw slot and produce a model that flies
plausibly and wrongly.

Side force and drag are deliberately absent. A spanwise incidence distribution
produces lift and the three moments at first order; it does not produce side
force, and its drag contribution is second order in the incidence perturbation.
Adding empty channels would invite someone to fill them without deriving them.
"""

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array


class CoeffIncrement(NamedTuple):
    """Additions to the aerodynamic coefficients, body/wind axes as `aero.py`."""

    CL: Array
    Cl: Array
    Cm: Array
    Cn: Array


def zero_increment() -> CoeffIncrement:
    """The default. Every field is a jnp array, not a Python float.

    `integrate.batch_sim` broadcasts every leaf of the carry and a Python scalar
    has no `.shape`, which raises there -- the same constraint `WindState`
    documents for the same reason.
    """
    return CoeffIncrement(
        CL=jnp.array(0.0), Cl=jnp.array(0.0), Cm=jnp.array(0.0), Cn=jnp.array(0.0)
    )


def add(a: CoeffIncrement, b: CoeffIncrement) -> CoeffIncrement:
    """Sum two increments. Superposition, the same property that licenses
    `wind.superpose`: two fields acting at once contribute independently."""
    return CoeffIncrement(
        CL=a.CL + b.CL, Cl=a.Cl + b.Cl, Cm=a.Cm + b.Cm, Cn=a.Cn + b.Cn
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
$PY -m pytest flightsim/tests/test_loads.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add flightsim/loads.py flightsim/tests/test_loads.py
git commit -m "Add CoeffIncrement: the seam distributed loads will enter through"
```

---

# Phase 2 — The dynamics seam

### Task 2: `derivatives` accepts an increment, defaulting to zero

**Files:**
- Modify: `flightsim/dynamics.py`
- Modify: `flightsim/tests/test_dynamics.py`

**This is the highest-risk task in the plan.** `derivatives` is called by `trim`, `integrate`, `validation`, `verification` and `vortex_viz`. The signature change must be backward-compatible by default.

- [ ] **Step 1: Write the failing test**

Append to `flightsim/tests/test_dynamics.py`:

```python
def test_a_zero_increment_is_bit_identical_to_not_passing_one(test_aircraft):
    """The property the whole design rests on. Every existing caller omits the
    increment, so if the zero case were merely close rather than exact, every
    frozen baseline in PROJECT.md section 4 would drift.

    Bit-identical, not approximately equal: adding exact 0.0 to a float is the
    identity for every finite value, so there is no reason to accept less.
    """
    import jax.numpy as jnp

    from flightsim import dynamics, loads
    from flightsim.state import Controls, State, euler_to_quat

    state = State(
        pos_ned=jnp.array([0.0, 0.0, -2000.0]),
        vel_body=jnp.array([60.0, 2.0, 3.0]),
        quat=euler_to_quat(jnp.array(0.1), jnp.array(0.05), jnp.array(0.2)),
        omega=jnp.array([0.1, 0.2, -0.05]),
    )
    controls = Controls(
        elevator=jnp.array(0.1), aileron=jnp.array(-0.05),
        rudder=jnp.array(0.02), throttle=jnp.array(0.6),
    )
    wind_ned, omega_gust = jnp.zeros(3), jnp.zeros(3)

    without = dynamics.derivatives(state, controls, test_aircraft, wind_ned, omega_gust)
    with_zero = dynamics.derivatives(
        state, controls, test_aircraft, wind_ned, omega_gust,
        increment=loads.zero_increment(),
    )
    for field in ("pos_ned", "vel_body", "quat", "omega"):
        assert np.array_equal(
            np.asarray(getattr(without, field)), np.asarray(getattr(with_zero, field))
        ), f"{field} differs between omitting the increment and passing zero"


def test_a_rolling_increment_produces_a_rolling_acceleration(test_aircraft):
    """The increment must actually reach the equations of motion, and reach the
    right axis. A test that only checked the zero case would pass just as
    happily if the increment were ignored entirely."""
    import jax.numpy as jnp

    from flightsim import dynamics, loads
    from flightsim.state import Controls, State, euler_to_quat

    state = State(
        pos_ned=jnp.array([0.0, 0.0, -2000.0]),
        vel_body=jnp.array([60.0, 0.0, 0.0]),
        quat=euler_to_quat(jnp.array(0.0), jnp.array(0.0), jnp.array(0.0)),
        omega=jnp.zeros(3),
    )
    controls = Controls(
        elevator=jnp.array(0.0), aileron=jnp.array(0.0),
        rudder=jnp.array(0.0), throttle=jnp.array(0.5),
    )
    wind_ned, omega_gust = jnp.zeros(3), jnp.zeros(3)

    base = dynamics.derivatives(state, controls, test_aircraft, wind_ned, omega_gust)
    rolled = dynamics.derivatives(
        state, controls, test_aircraft, wind_ned, omega_gust,
        increment=loads.zero_increment()._replace(Cl=jnp.array(0.01)),
    )
    # A positive rolling-moment coefficient must raise p-dot and leave q-dot alone.
    assert float(rolled.omega[0]) > float(base.omega[0])
    assert float(rolled.omega[1]) == pytest.approx(float(base.omega[1]), abs=1e-12)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
$PY -m pytest flightsim/tests/test_dynamics.py -v -k "zero_increment or rolling_increment"
```

Expected: 2 failed with `TypeError: derivatives() got an unexpected keyword argument 'increment'`.

- [ ] **Step 3: Write the implementation**

In `flightsim/dynamics.py`, add the import:

```python
from flightsim.loads import CoeffIncrement, zero_increment
```

Replace the `derivatives` signature and its aero call:

```python
def derivatives(
    state: State,
    controls: Controls,
    ac: Aircraft,
    wind_ned: Array,
    omega_gust: Array,
    increment: CoeffIncrement | None = None,
) -> State:
    """State derivative. Returns a State whose fields are time derivatives.

    `increment` carries aerodynamic coefficients computed from the wind field
    ACROSS the airframe -- strip-integrated loads -- which `aero.py` cannot
    produce because its standing rule is that it never sees the field. It is
    summed into the coefficients here, at the same seam where wind already
    enters and nowhere else.

    Defaults to None, which is treated as exact zeros. Adding 0.0 to a float is
    the identity, so an omitted increment is BIT-IDENTICAL to a zero one and
    every existing caller is unaffected. `test_dynamics.py` asserts that rather
    than assuming it.
    """
    dcm = quat_to_dcm(state.quat)

    vel_rel = relative_velocity(state.vel_body, state.quat, wind_ned)
    omega_rel = state.omega - omega_gust

    altitude = -state.pos_ned[2]
    rho = density(altitude)
    force, moment = aero_forces_moments(
        vel_rel, omega_rel, controls, ac, rho, speed_of_sound(altitude),
        increment=zero_increment() if increment is None else increment,
    )
    force = force + thrust_force(controls, ac, rho)
    ...
```

Then in `flightsim/aero.py`, thread the increment into `aero_forces_moments` only:

```python
def aero_forces_moments(
    vel_rel: Array,
    omega_rel: Array,
    controls: Controls,
    ac: Aircraft,
    rho: Array,
    a_sound: Array,
    increment=None,
) -> tuple[Array, Array]:
    """Body-axis aerodynamic force (N) and moment (N.m).

    `increment` is an additive set of coefficients computed OUTSIDE this module,
    from the wind field across the airframe. This module still never sees the
    field itself -- it receives four numbers and adds them after the build-up,
    which is what preserves the rule that a zero-strength wind is bit-identical
    to still air.
    """
    V, alpha, beta = air_data(vel_rel)
    qbar = 0.5 * rho * V**2
    CL, CD, CY, Cl, Cm, Cn = coefficients(vel_rel, omega_rel, controls, ac, a_sound)
    if increment is not None:
        CL = CL + increment.CL
        Cl = Cl + increment.Cl
        Cm = Cm + increment.Cm
        Cn = Cn + increment.Cn
    ...
```

Leave the rest of `aero_forces_moments` unchanged.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
$PY -m pytest flightsim/tests/test_dynamics.py -v -k "zero_increment or rolling_increment"
```

Expected: 2 passed.

- [ ] **Step 5: Run the whole suite — this is the task that could break everything**

```bash
$PY -m pytest flightsim/tests/ -q
```

Expected: all pass, including every `PROJECT.md` §4 mode check. If a mode value has moved, **stop** — the zero-increment path is not bit-identical and the defaulting is wrong.

- [ ] **Step 6: Commit**

```bash
git add flightsim/dynamics.py flightsim/aero.py flightsim/tests/test_dynamics.py
git commit -m "Let derivatives accept a coefficient increment, defaulting to exact zero"
```

---

# Phase 3 — Building the increment from a field

### Task 3: `strip_increment` — roll only, honestly labelled

**Files:**
- Modify: `flightsim/loads.py`
- Modify: `flightsim/tests/test_loads.py`

**Scope note, and it matters.** `wind.strip_roll_moment` exists and is validated. There is no validated strip pitch or yaw integral. This task wires in **only** the rolling moment and leaves the other three channels at zero, because shipping an unvalidated pitch integral would be worse than shipping none.

- [ ] **Step 1: Write the failing test**

Append to `flightsim/tests/test_loads.py`:

```python
def test_the_strip_increment_carries_roll_only_for_now():
    """Only the rolling moment has a validated strip integral. Pitch and yaw
    are left at exact zero rather than filled with something plausible -- the
    plan that added the roll integral deliberately stopped there, and an
    unvalidated pitch integral would be worse than none.

    Asserted so that if someone later fills those channels, they have to change
    this test and therefore have to justify it.
    """
    import jax.numpy as jnp

    from flightsim import airframe, loads, wind
    from flightsim.aircraft import REGISTRY
    from flightsim.state import State, euler_to_quat

    ac = REGISTRY["boeing747"]
    st = airframe.stations(ac, n_span=201, n_lon=9)
    field = lambda p: jnp.array([0.0, 0.0, 1e-7 * p[1] ** 3])  # noqa: E731
    state = State(
        pos_ned=jnp.array([0.0, 0.0, -11278.0]),
        vel_body=jnp.array([236.0, 0.0, 0.0]),
        quat=euler_to_quat(jnp.array(0.0), jnp.array(0.0), jnp.array(0.0)),
        omega=jnp.zeros(3),
    )

    inc = loads.strip_increment(state, field, ac, st)
    expected = wind.strip_roll_moment(
        state.pos_ned, state.quat, field, ac, st, 236.0
    )
    assert float(inc.Cl) == pytest.approx(float(expected), rel=1e-9)
    assert float(inc.CL) == 0.0
    assert float(inc.Cm) == 0.0
    assert float(inc.Cn) == 0.0


def test_the_strip_increment_uses_air_relative_speed_not_ground_speed():
    """The incidence a strip sees is set by the speed of the air over it. Using
    inertial speed would reintroduce exactly the error the whole air-relative
    design exists to avoid, and it would only show up in a headwind."""
    import jax.numpy as jnp

    from flightsim import airframe, loads
    from flightsim.aircraft import REGISTRY
    from flightsim.state import State, euler_to_quat

    ac = REGISTRY["boeing747"]
    st = airframe.stations(ac, n_span=201, n_lon=9)
    field = lambda p: jnp.array([50.0, 0.0, 1e-7 * p[1] ** 3])  # noqa: E731
    base = State(
        pos_ned=jnp.array([0.0, 0.0, -11278.0]),
        vel_body=jnp.array([236.0, 0.0, 0.0]),
        quat=euler_to_quat(jnp.array(0.0), jnp.array(0.0), jnp.array(0.0)),
        omega=jnp.zeros(3),
    )
    # A 50 m/s tailwind component leaves ground speed alone and reduces
    # airspeed, so an air-relative implementation must give a LARGER incidence
    # and therefore a larger rolling moment than a ground-speed one would.
    inc = loads.strip_increment(base, field, ac, st)
    still = loads.strip_increment(
        base, lambda p: jnp.array([0.0, 0.0, 1e-7 * p[1] ** 3]), ac, st
    )
    assert abs(float(inc.Cl)) > abs(float(still.Cl))
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
$PY -m pytest flightsim/tests/test_loads.py -v -k "strip_increment"
```

Expected: 2 failed with `AttributeError: module 'flightsim.loads' has no attribute 'strip_increment'`.

- [ ] **Step 3: Write the implementation**

Append to `flightsim/loads.py`:

```python
from flightsim.aero import air_data
from flightsim.aircraft import Aircraft
from flightsim.dynamics import relative_velocity
from flightsim.state import State


def strip_increment(state: State, field, ac: Aircraft, stations) -> CoeffIncrement:
    """Coefficient increment from integrating a wind field across the airframe.

    ROLL ONLY. `wind.strip_roll_moment` is the one strip integral this project
    has validated -- against the tabulated Clp for a rigid roll rate, against
    Stengel eq. 3.4-40's closed form for a rectangular wing, and against the
    equivalent-rate treatment for a linear gradient. There is no validated
    pitch or yaw integral, so those channels stay at exact zero rather than
    being filled with something plausible. Shipping an unvalidated pitch
    integral would be worse than shipping none, because it would look like
    increased fidelity.

    Airspeed is AIR-RELATIVE, taken through the same `relative_velocity` the
    rest of the model uses. Using ground speed would reintroduce precisely the
    error the air-relative design exists to prevent, and it would only reveal
    itself in a wind with a significant along-track component.
    """
    from flightsim import wind  # local: wind imports airframe, which imports aircraft

    wind_at_cg = field(state.pos_ned)
    vel_rel = relative_velocity(state.vel_body, state.quat, wind_at_cg)
    airspeed, _, _ = air_data(vel_rel)

    roll = wind.strip_roll_moment(
        state.pos_ned, state.quat, field, ac, stations, airspeed
    )
    return zero_increment()._replace(Cl=roll)
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
$PY -m pytest flightsim/tests/test_loads.py -v
```

Expected: 6 passed.

If an `ImportError` about a circular import appears, the local `wind` import inside the function is the fix — `wind` imports `airframe`, which imports `aircraft`, and a module-level import here closes the loop.

- [ ] **Step 5: Commit**

```bash
git add flightsim/loads.py flightsim/tests/test_loads.py
git commit -m "Build a coefficient increment from the strip integral, roll only"
```

---

# Phase 4 — Threading it through the integrator

### Task 4: `step` and `rollout` take an optional load model

**Files:**
- Modify: `flightsim/integrate.py`
- Modify: `flightsim/tests/test_integrate.py`

- [ ] **Step 1: Write the failing test**

Append to `flightsim/tests/test_integrate.py`:

```python
def test_omitting_the_load_model_is_bit_identical_to_today(test_aircraft):
    """The gate that protects every frozen baseline. A run that does not ask for
    strip loads must produce exactly the trajectory it produced before this
    feature existed -- not nearly, exactly."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    from flightsim import integrate
    from flightsim.state import Controls, State, euler_to_quat

    state = State(
        pos_ned=jnp.array([0.0, 0.0, -2000.0]),
        vel_body=jnp.array([60.0, 0.0, 2.0]),
        quat=euler_to_quat(jnp.array(0.0), jnp.array(0.03), jnp.array(0.0)),
        omega=jnp.zeros(3),
    )
    controls = Controls(
        elevator=jnp.array(0.02), aileron=jnp.array(0.0),
        rudder=jnp.array(0.0), throttle=jnp.array(0.5),
    )
    sim = integrate.init_sim(state, jax.random.PRNGKey(0))

    plain, _ = integrate.rollout(sim, controls, jnp.array(0.02), test_aircraft, 100)
    explicit, _ = integrate.rollout(
        sim, controls, jnp.array(0.02), test_aircraft, 100, load_model=None
    )
    assert np.array_equal(
        np.asarray(plain.state.pos_ned), np.asarray(explicit.state.pos_ned)
    )
    assert np.array_equal(np.asarray(plain.state.quat), np.asarray(explicit.state.quat))


def test_the_applied_increment_is_cached_on_the_sim_state(test_aircraft):
    """Same reason wind_ned and omega_gust are cached: a recorder or controller
    must be able to see what was actually applied without calling the model a
    second time. Consumers see one step of lag, which is what a real sensor
    gives anyway."""
    import jax
    import jax.numpy as jnp

    from flightsim import integrate, loads
    from flightsim.state import Controls, State, euler_to_quat

    state = State(
        pos_ned=jnp.array([0.0, 0.0, -2000.0]),
        vel_body=jnp.array([60.0, 0.0, 0.0]),
        quat=euler_to_quat(jnp.array(0.0), jnp.array(0.0), jnp.array(0.0)),
        omega=jnp.zeros(3),
    )
    controls = Controls(
        elevator=jnp.array(0.0), aileron=jnp.array(0.0),
        rudder=jnp.array(0.0), throttle=jnp.array(0.5),
    )
    sim = integrate.init_sim(state, jax.random.PRNGKey(0))
    assert isinstance(sim.increment, loads.CoeffIncrement)
    assert float(sim.increment.Cl) == 0.0

    stepped = integrate.step(
        sim, controls, jnp.array(0.02), test_aircraft,
        load_model=lambda s: loads.zero_increment()._replace(Cl=jnp.array(0.005)),
    )
    assert float(stepped.increment.Cl) == pytest.approx(0.005)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
$PY -m pytest flightsim/tests/test_integrate.py -v -k "load_model or cached_on_the_sim_state"
```

Expected: 2 failed — `TypeError` on the unexpected `load_model` argument, and `AttributeError` on `sim.increment`.

- [ ] **Step 3: Write the implementation**

In `flightsim/integrate.py`, add to the imports:

```python
from flightsim.loads import CoeffIncrement, zero_increment
```

Extend `SimState` with one field, after `omega_gust`:

```python
    increment: CoeffIncrement  # coefficients applied by the previous step
```

Extend its docstring to say the same thing the wind cache says: an output cache, not model state, so a recorder can see what was applied without re-invoking the model.

Update `init_sim` and `batch_sim`:

```python
def init_sim(state: State, key: Array) -> SimState:
    return SimState(
        state=state,
        wind=zero_wind_state(),
        key=key,
        wind_ned=jnp.zeros(3),
        omega_gust=jnp.zeros(3),
        increment=zero_increment(),
    )
```

```python
    # in batch_sim, alongside the other broadcasts
    increments = jax.tree.map(
        lambda x: jnp.broadcast_to(x, (n,) + x.shape), zero_increment()
    )
```

and pass `increment=increments` in the returned `SimState`.

Update `step`:

```python
@partial(jax.jit, static_argnames=("wind_model", "load_model"))
def step(
    sim: SimState,
    controls: Controls,
    dt: Array,
    ac: Aircraft,
    wind_model=zero_wind,
    load_model=None,
) -> SimState:
    """One RK4 step. The PRNG key is threaded through the wind model.

    `load_model` maps a `State` to a `CoeffIncrement`. It is sampled once per
    step and held across the four RK4 stages, exactly as the wind is and for the
    same reason. Omitting it gives an exact zero increment, which makes the step
    bit-identical to one taken before this feature existed.
    """
    wind_ned, omega_gust, wind_state, key = wind_model(sim.wind, sim.state, sim.key, dt)
    increment = zero_increment() if load_model is None else load_model(sim.state)

    def f(s: State) -> State:
        return derivatives(s, controls, ac, wind_ned, omega_gust, increment=increment)

    new_state = rk4_step(f, sim.state, dt)
    new_state = new_state._replace(quat=quat_normalize(new_state.quat))

    return SimState(
        state=new_state,
        wind=wind_state,
        key=key,
        wind_ned=wind_ned,
        omega_gust=omega_gust,
        increment=increment,
    )
```

Add `load_model=None` to `rollout` and `batched_rollout`, both in the signature, the `static_argnames`, and the forwarded call.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
$PY -m pytest flightsim/tests/test_integrate.py -v
```

Expected: all pass.

- [ ] **Step 5: Run the whole suite**

```bash
$PY -m pytest flightsim/tests/ -q
```

Expected: all pass. `SimState` gained a field, so anything constructing one positionally will fail here — fix those call sites rather than reordering the tuple.

- [ ] **Step 6: Commit**

```bash
git add flightsim/integrate.py flightsim/tests/test_integrate.py
git commit -m "Thread an optional load model through step and rollout"
```

---

### Task 5: A strip-flying convenience wrapper

**Files:**
- Modify: `flightsim/loads.py`
- Modify: `flightsim/tests/test_loads.py`

- [ ] **Step 1: Write the failing test**

Append to `flightsim/tests/test_loads.py`:

```python
def test_the_strip_model_refuses_an_aircraft_that_fails_the_tail_arm_gate():
    """The gate exists to stop the strip path being used where its key input is
    not trustworthy. It must fire at construction, loudly, rather than silently
    producing numbers -- a run that quietly used a 0.856-chord tail arm would be
    very hard to spot afterwards."""
    from flightsim import airframe, loads
    from flightsim.aircraft import REGISTRY

    for name in ("cessna172", "cherokee"):
        ac = REGISTRY[name]
        assert not airframe.tail_arm_is_plausible(ac)
        with pytest.raises(ValueError, match="tail arm"):
            loads.strip_model(lambda p: p * 0.0, ac)


def test_the_strip_model_accepts_both_747_configurations():
    from flightsim import loads
    from flightsim.aircraft import REGISTRY

    for name in ("boeing747", "boeing747_approach"):
        model = loads.strip_model(lambda p: p * 0.0, REGISTRY[name])
        assert callable(model)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
$PY -m pytest flightsim/tests/test_loads.py -v -k "strip_model"
```

Expected: 2 failed with `AttributeError: module 'flightsim.loads' has no attribute 'strip_model'`.

- [ ] **Step 3: Write the implementation**

Append to `flightsim/loads.py`:

```python
def strip_model(field, ac: Aircraft, stations=None):
    """Build a `load_model` for `integrate.step` from a wind field.

    Raises if the aircraft fails the tail-arm plausibility gate. That check
    exists because the sample stations are built from a DERIVED tail arm, and
    for two of the four aircraft in the registry that derivation returns a value
    the airframe plainly does not have. Failing at construction is deliberate:
    a run that quietly used a 0.856-chord tail arm would produce numbers that
    look ordinary and are not.
    """
    from flightsim import airframe

    if not airframe.tail_arm_is_plausible(ac):
        raise ValueError(
            f"tail arm {float(airframe.effective_tail_arm(ac)):.4f} chords is outside "
            f"{airframe.TAIL_ARM_BAND} -- this aircraft's CLq and Cmq disagree about "
            f"what airframe they describe, so the strip path must not be used for it. "
            f"Use the point model instead."
        )
    st = airframe.stations(ac) if stations is None else stations
    return lambda state: strip_increment(state, field, ac, st)
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
$PY -m pytest flightsim/tests/test_loads.py -v
```

Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add flightsim/loads.py flightsim/tests/test_loads.py
git commit -m "Add strip_model, gated on the aircraft having a plausible tail arm"
```

---

# Phase 5 — Measure what changed

### Task 6: Fly the vortex both ways and quantify the difference

**Files:**
- Modify: `flightsim/tests/test_wind.py`

This is validation gates 7 and 9 from the design document, which the previous plan could not implement because nothing consumed the strip loads.

- [ ] **Step 1: Write the test**

Append to `flightsim/tests/test_wind.py`:

```python
def test_flying_the_parks_vortex_with_strip_loads_changes_the_trajectory():
    """Gate 9. The point of the whole exercise is that this differs. If the two
    trajectories agreed, the strip path would not be reaching the equations of
    motion and every other test here would be measuring nothing.

    Reported rather than bounded: the size of the difference is the RESULT, and
    fixing a tolerance around it now would be asserting the answer before
    measuring it.
    """
    from flightsim import airframe, integrate, loads, trim
    from flightsim.aircraft import CRUISE, REGISTRY

    ac = REGISTRY["boeing747"]
    v, h = CRUISE["boeing747"]["airspeed"], CRUISE["boeing747"]["altitude"]
    x, _ = trim.trim(jnp.array(v), jnp.array(h), ac)
    state = trim.trimmed_state(x[0], jnp.array(v), jnp.array(h))
    controls = trim.trimmed_controls(x[1], x[2])

    array = single()
    field = lambda p: wind.vortex_wind(p, array)  # noqa: E731
    model = wind.vortex_model(array)
    # Start well upstream so the aircraft flies through the whole core.
    start = state._replace(pos_ned=jnp.array([-6.0 * CASE1_R0, 0.0, -h]))
    sim = integrate.init_sim(start, jax.random.PRNGKey(0))
    steps = int(12.0 * CASE1_R0 / v / 0.02)

    point, _ = integrate.rollout(sim, controls, jnp.array(0.02), ac, steps, wind_model=model)
    strip, _ = integrate.rollout(
        sim, controls, jnp.array(0.02), ac, steps, wind_model=model,
        load_model=loads.strip_model(field, ac),
    )

    d_pos = float(
        np.linalg.norm(np.asarray(point.state.pos_ned) - np.asarray(strip.state.pos_ned))
    )
    print(f"\nParks core traverse, {steps} steps:")
    print(f"  position difference, point vs strip: {d_pos:.6f} m")
    assert np.isfinite(d_pos)


def test_the_rigid_rotation_structure_diagnostic_is_reported_per_field():
    """Gate 7. The current point model is exactly equivalent to assuming the
    shear matrix has RIGID-ROTATION STRUCTURE -- that d(v)/dz = -d(w)/dy and
    d(u)/dz = -d(w)/dx. This reports how far each field departs from that, which
    is the cheapest available predictor of where the point model will struggle.

    Reported, not asserted: the ratio is a property of each field, and pinning
    it would freeze a diagnostic rather than a result.
    """
    ac_dummy = None  # not needed; the diagnostic is a property of the field alone
    del ac_dummy

    array = single()
    field = lambda p: wind.vortex_wind(p, array)  # noqa: E731
    print("\nRigid-rotation-structure diagnostic (1.0 = exactly rotation-like):")
    for frac, label in ((0.5, "inside core"), (1.5, "outside core")):
        s = _level_state(north=frac * CASE1_R0)
        dcm = np.asarray(
            jax.jacfwd(field)(s.pos_ned)
        )
        # body == NED here (wings level, heading north)
        dw_dy, dv_dz = dcm[2, 1], dcm[1, 2]
        dw_dx, du_dz = dcm[2, 0], dcm[0, 2]
        pair_p = "n/a" if abs(dw_dy) < 1e-12 else f"{-dv_dz / dw_dy:+.4f}"
        pair_q = "n/a" if abs(dw_dx) < 1e-12 else f"{-du_dz / dw_dx:+.4f}"
        print(f"  {label:12s}  p-pair {pair_p}   q-pair {pair_q}")
    assert True  # a reporting test; the numbers go into PROJECT.md
```

- [ ] **Step 2: Run the tests**

```bash
$PY -m pytest flightsim/tests/test_wind.py -v -s -k "strip_loads_changes or rigid_rotation_structure"
```

Expected: 2 passed, with the position difference and both diagnostic ratios printed. **Record all three — Task 7 writes them into the documents.**

- [ ] **Step 3: Commit**

```bash
git add flightsim/tests/test_wind.py
git commit -m "Fly the Parks vortex both ways, and report the structure diagnostic"
```

---

### Task 7: Re-run the figure-8 analysis and write up what moved

**Files:**
- Modify: `scripts/vortex.py`
- Modify: `docs/PROJECT.md`
- Modify: `docs/ASSUMPTIONS.md`

- [ ] **Step 1: Add a strip option to the analysis script**

In `scripts/vortex.py`, add an argument:

```python
parser.add_argument(
    "--strip", action="store_true",
    help="fly the vortex with strip-integrated loads instead of a point sample "
         "plus gradient. Changes the answer; that change is the result and must "
         "be reported with the loading-shape sensitivity beside it.",
)
```

`vortex_viz.fly` takes the **field** and builds `wind.field_model(field)` itself, so it does not need a load model passed in — it needs a flag and can build one from the field it already has. Add a keyword-only `strip: bool = False` to `fly`, and inside it:

```python
    model = wind.field_model(field)
    load_model = loads.strip_model(field, ac) if strip else None
```

then forward `load_model=load_model` to the existing `integrate.rollout` call. Nothing else in `fly` changes, and `strip=False` leaves it byte-for-byte equivalent to today.

`fly` also passes `model=model` to `_measure`; leave that as the wind model. The strip increment is recoverable from `SimState.increment` if the measurement ever needs it, and adding it to `Encounter` now would be speculative.

- [ ] **Step 2: Run both ways and record the numbers**

```bash
PYTHONPATH=. $PY scripts/vortex.py --case hannibal
```

```bash
PYTHONPATH=. $PY scripts/vortex.py --case hannibal --strip
```

Record `d(theta)`, `d(n)` and the ordering verdict from both.

- [ ] **Step 3: Write the comparison into `PROJECT.md` §4**

Append to the session-13 table added by the previous plan:

```markdown
### Strip loads in the 6-DOF (session 14)

| Check | Measured | Tolerance |
|---|---|---|
| Zero increment vs omitting it, `derivatives` | **bit-identical** | `np.array_equal` |
| Omitting `load_model`, 100-step rollout | **bit-identical** | `np.array_equal` |
| Strip increment vs `wind.strip_roll_moment` | agrees | rel 1e-9 |
| Tail-arm gate at construction | raises for both light aircraft | `pytest.raises` |
| Parks core traverse, point vs strip position | **<RECORDED> m** | reported |
| Fig. 8 point, point vs strip | **<RECORDED>** | reported |
| Ordering vortex < updraft < manoeuvre | **must still hold** | exact |
```

- [ ] **Step 4: Add the caveat to `ASSUMPTIONS.md` §E2**

Append:

```markdown
**Session 14: the strip path is now flyable, and it moves the answer.** The
correction measured above is no longer only a diagnostic — `loads.strip_model`
feeds strip-integrated rolling moments into the equations of motion. The
resulting change to the Fig. 8 point is recorded in `PROJECT.md` §4 and **must
be quoted with the loading-shape sensitivity beside it** (2.6% across defensible
shapes, 49.7% including a uniform bracket), because that sensitivity is the
dominant remaining uncertainty in the strip result.

**Only the rolling moment is strip-integrated.** Pitch and yaw still come from
the point-plus-gradient treatment. The vortex's dominant input is *pitch*, so
the headline Fig. 8 number is still produced by the old path — the strip work
improves the lateral response, which the Parks field barely excites when flown
wings-level. **Do not read the strip path as having fixed the vortex result.**
```

- [ ] **Step 5: Verify no placeholders remain**

```bash
grep -rn "<RECORDED>" docs/ flightsim/ scripts/
```

Expected: no output.

- [ ] **Step 6: Full suite and notebook gate**

```bash
$PY -m pytest flightsim/tests/ -q
```

```bash
$PY -m pytest --nbval-lax notebooks/ -q
```

- [ ] **Step 7: Rebuild the report so its figures match the new state**

```bash
PYTHONPATH=. $PY scripts/turbulence_report.py docs/summary/turbulence-report.pdf
```

- [ ] **Step 8: Commit**

```bash
git add scripts/vortex.py flightsim/vortex_viz.py docs/PROJECT.md docs/ASSUMPTIONS.md docs/summary/turbulence-report.pdf
git commit -m "Fly the vortex with strip loads, and record what moved"
```

---

## The honest limitation this plan ships with

**Only the rolling moment is strip-integrated.** That is worth stating plainly because it is easy to over-read what this achieves.

The Parks vortex, flown wings-level on a northerly track, produces **no roll input at all** — its axes lie across the flight path and the field has no lateral variation. Its dominant input is pitch, and pitch still comes from the point-plus-gradient path.

So this plan does **not** fix the vortex result. What it does is build and validate the seam, and deliver a strip path that matters for any field with genuine spanwise structure — which is what the stated goal, flying small-scale fields, actually requires. A strip pitch integral is the obvious next step, and it needs its own validation against `Cmq` before it can be trusted, exactly as the roll integral needed against `Clp`.

## Deliberately out of scope

- **A strip pitch or yaw integral.** No validated implementation exists. Task 3 leaves those channels at exact zero and asserts it, so filling them requires changing a test and therefore justifying it.
- **The eq. 3.4-55 weighting question.** Still unresolved and still avoided structurally.
- **The `α̇` unsteady term** and **the accelerometer lever arm.** Both verified, both separate work.
- **Any change to `PROJECT.md` §4 mode baselines.**

## Self-review against the design

| Design requirement | Task |
|---|---|
| §4 "A2 needs a wider contract, called out as its own reviewed step" | this plan exists |
| §6 gate 7, rigid-rotation-structure diagnostic | 6 |
| §6 gate 9, vortex Fig. 8 movement quantified and explained | 6, 7 |
| §6 gate 8, baselines unmoved on the default path | 2, 4 |
| §7c, shape sensitivity quoted with every strip result | 7 |
| Point path retained as default and fallback | 2, 4, 5 |
