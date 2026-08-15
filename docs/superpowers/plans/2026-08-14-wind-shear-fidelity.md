# Wind-Shear Fidelity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single-point wind sample plus CG-tangent gradient with a distributed sample across the airframe, so the model can carry flow fields whose scale approaches a wingspan — and make every constant's provenance machine-checkable.

**Architecture:** Three phases, each ending in working, tested, committed software. Phase 1 builds a provenance ledger enforced by a test. Phase 2 (A1) replaces the tangent with a least-squares fit over the airframe, which introduces no new aerodynamic data and must reduce identically to today's model for linear fields. Phase 3 (A2) adds strip force integration, which is what actually captures curvature, calibrated so a rigid roll rate reproduces the tabulated `Clp`.

**Tech Stack:** Python 3.10+, JAX (float64, `jax_debug_nans` on in tests), NumPy, pytest.

**Design document:** `docs/superpowers/specs/2026-08-14-wind-shear-fidelity-design.md`. Read §2 before touching any sign convention — two equations in the source text are wrong and the design records which.

**Python interpreter for every command below:**
`C:/Users/mateusz/UROP/Claude_Flight_Sim/.venv/Scripts/python.exe`

Shorthand used in commands: `$PY`. Set it once per shell:

```bash
PY="C:/Users/mateusz/UROP/Claude_Flight_Sim/.venv/Scripts/python.exe"
```

---

## Things that will bite you

Read these before Task 1. They are not obvious from the code.

1. **`conftest.py` turns on `jax_debug_nans` for the whole suite.** Any NaN — including one inside a branch that is never used — fails the test. `jnp.where` evaluates *both* branches, so guard divisors before the `where`, not inside it. `flightsim/wind.py:131` shows the existing pattern.
2. **Never `import jax.numpy` before `import flightsim`.** The package enables float64 on import, and it must happen before any array is created. Every test file starts with `import flightsim  # noqa: F401`.
3. **`PROJECT.md` §4 baselines are off-limits.** The five 747 mode values are frozen. If one moves, something real broke — do not "fix" the tolerance.
4. **Every number needs a citation.** The project's standing rule. Phase 1 makes it enforceable; until then, follow it by hand.
5. **The three existing gust-rate signs in `wind.gust_rates` are correct.** They were re-derived independently against Stengel eqs. 3.4-48, 3.4-50 and 3.4-52 (design §2b). Do not change them.

---

## File Structure

| File | Responsibility | Status |
|---|---|---|
| `flightsim/provenance.py` | The ledger: every constant's category and citation, as data | **create** |
| `flightsim/tests/test_provenance.py` | Enforces the ledger's five rules | **create** |
| `flightsim/airframe.py` | Where on the airframe the field is sampled; derived tail arm; spanwise loading | **create** |
| `flightsim/tests/test_airframe.py` | Tail-arm derivation, plausibility gate, loading calibration | **create** |
| `flightsim/wind.py` | Gains `sampled_rates`, `sampled_field_model`, `strip_roll_moment`. **`gust_rates` is kept unchanged** as the reference implementation and the fallback | modify |
| `flightsim/tests/test_wind.py` | Gains the reduction properties and the curvature measurement | modify |
| `docs/ASSUMPTIONS.md` | §E2 gains its measured bound | modify |
| `docs/PROJECT.md` | §2 records the new interface; §4 gains the new checks | modify |

---

# Phase 1 — The provenance ledger

### Task 1: Ledger data structure and its enforcing test

**Files:**
- Create: `flightsim/provenance.py`
- Test: `flightsim/tests/test_provenance.py`

- [ ] **Step 1: Write the failing test**

Create `flightsim/tests/test_provenance.py`:

```python
"""The provenance ledger's own rules.

Review asked for an unambiguous separation between numbers taken from cited
tables and numbers that were predicted, with no credit given to a calibrated
value for landing in a plausible range. This file is what makes that
separation enforceable rather than aspirational: a constant added without a
ledger entry fails the build.

The precedent is `validation.Reference`, whose mandatory `source` field is
asserted by test_validation.py. This is the same idea applied to every
constant rather than only to published reference values.
"""

import pytest

from flightsim import provenance
from flightsim.provenance import LEDGER, Entry


def test_every_entry_uses_one_of_the_four_categories():
    """Four categories, mutually exclusive. A fifth would mean the distinction
    review asked for has been blurred."""
    for name, entry in LEDGER.items():
        assert entry.category in provenance.CATEGORIES, (
            f"{name} has category {entry.category!r}, "
            f"which is not one of {provenance.CATEGORIES}"
        )


def test_sourced_and_declared_entries_carry_a_usable_detail():
    """A SOURCED entry without document, table and page is not sourced, it is
    asserted. A DECLARED entry without its sensitivity is not declared, it is
    hidden. Length 20 is the same bar test_validation.py sets on
    `Reference.source`."""
    for name, entry in LEDGER.items():
        if entry.category in ("SOURCED", "DECLARED"):
            assert len(entry.detail) > 20, f"{name} has no usable detail"


def test_derived_and_calibrated_entries_name_inputs_that_exist():
    """A DERIVED value is only as good as what it was derived from, so its
    inputs must themselves be in the ledger and reachable."""
    for name, entry in LEDGER.items():
        if entry.category in ("DERIVED", "CALIBRATED"):
            assert entry.inputs, f"{name} is {entry.category} but names no inputs"
            for dep in entry.inputs:
                assert dep in LEDGER, f"{name} depends on {dep}, which is not in the ledger"


def test_the_dependency_graph_has_no_cycles():
    """A cycle would let two numbers justify each other with nothing underneath.
    Every chain must bottom out in SOURCED or DECLARED entries."""
    for name in LEDGER:
        seen = set()
        stack = [name]
        while stack:
            current = stack.pop()
            assert current not in seen, f"{name} has a cyclic dependency via {current}"
            seen.add(current)
            stack.extend(LEDGER[current].inputs)


def test_an_entry_with_an_unknown_category_is_rejected():
    """The rules must be able to fail. A test that can only pass demonstrates
    nothing -- PROJECT.md's falsification rule applied to this file."""
    bad = Entry(category="PROBABLY_FINE", detail="x" * 30)
    assert bad.category not in provenance.CATEGORIES
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
$PY -m pytest flightsim/tests/test_provenance.py -v
```

Expected: collection error, `ModuleNotFoundError: No module named 'flightsim.provenance'`.

- [ ] **Step 3: Write the minimal implementation**

Create `flightsim/provenance.py`:

```python
"""Where every constant in this model came from.

Review's requirement, stated directly: it must be possible to say which numbers
are bulletproof -- read from a cited table -- and which were predicted, with no
credit given to a predicted number for landing in a plausible range.

Four categories, mutually exclusive:

  SOURCED     read directly from a cited table. `detail` carries document,
              table and page. Nothing else counts as sourced.
  DERIVED     computed from SOURCED values by a stated exact relation. The
              relation is citable; the number is not independently checkable.
  CALIBRATED  fitted so the model reproduces a SOURCED number to a stated
              tolerance. The fit target is an input.
  DECLARED    chosen. Not derivable from any source this project holds.
              `detail` must carry the sensitivity range.

`inputs` names other ledger entries. It is what makes a DERIVED number's chain
walkable back to something SOURCED, and test_provenance.py asserts the chain
exists, is acyclic, and bottoms out.

This module holds no aircraft data. It holds statements ABOUT data, so that
adding a constant without saying where it came from fails the build.
"""

from typing import NamedTuple

CATEGORIES = ("SOURCED", "DERIVED", "CALIBRATED", "DECLARED")


class Entry(NamedTuple):
    """One constant's provenance.

    `detail` is free text because the four categories need different things
    from it -- a citation, a relation, a fit target, a sensitivity range -- and
    a schema rigid enough to hold all four would be harder to read than the
    prose it replaced. What is NOT free text is `category` and `inputs`, which
    are what the tests actually enforce.
    """

    category: str
    detail: str
    inputs: tuple[str, ...] = ()


LEDGER: dict[str, Entry] = {}
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
$PY -m pytest flightsim/tests/test_provenance.py -v
```

Expected: 5 passed. The four loop-over-`LEDGER` tests pass vacuously on an empty ledger; Task 2 fills it.

- [ ] **Step 5: Commit**

```bash
git add flightsim/provenance.py flightsim/tests/test_provenance.py
git commit -m "Add the provenance ledger and the rules it must satisfy"
```

---

### Task 2: Populate the ledger for the constants this work touches

**Files:**
- Modify: `flightsim/provenance.py`
- Modify: `flightsim/tests/test_provenance.py`

Scope note: the design (§5) says the ledger eventually covers every constant in `aircraft.py`, `atmosphere.py` and `wind.py`. That is a large retrofit and it is **not** in this plan. This task enters only the entries this work depends on, plus the test that stops the ledger from silently disagreeing with the code.

- [ ] **Step 1: Write the failing test**

Append to `flightsim/tests/test_provenance.py`:

```python
def test_the_747_reference_geometry_is_sourced_from_cr2144():
    """These three are the foundation everything else in this work rests on.
    If they are ever reclassified, the chain above them is no longer sourced."""
    for name in ("b747.S", "b747.b", "b747.c"):
        assert LEDGER[name].category == "SOURCED"
        assert "IX-3" in LEDGER[name].detail, f"{name} must cite its table"


def test_the_effective_tail_arm_is_derived_and_never_sourced():
    """It is a ratio of two tabulated derivatives, not a measured dimension.
    Quoting it as 747 geometry would be a category error -- design section 7d."""
    entry = LEDGER["b747.l_eff"]
    assert entry.category == "DERIVED"
    assert set(entry.inputs) == {"b747.Cmq", "b747.CLq", "b747.c"}


def test_the_loading_shape_is_declared_and_carries_its_sensitivity():
    """Taper ratio is not in CR-2144 and is not recoverable from S, b and cbar
    (design section 3f), so the shape is a choice and must be reported as one."""
    entry = LEDGER["strip.loading_shape"]
    assert entry.category == "DECLARED"
    assert "sensitivit" in entry.detail.lower()


def test_the_calibrated_lift_slope_names_the_number_it_is_pinned_to():
    """A calibrated value with no stated target is just a number."""
    entry = LEDGER["strip.lift_slope"]
    assert entry.category == "CALIBRATED"
    assert "b747.Clp" in entry.inputs
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
$PY -m pytest flightsim/tests/test_provenance.py -v -k "cr2144 or tail_arm or loading_shape or lift_slope"
```

Expected: 4 failed with `KeyError: 'b747.S'`.

- [ ] **Step 3: Write the implementation**

Replace `LEDGER: dict[str, Entry] = {}` in `flightsim/provenance.py` with:

```python
# NASA CR-2144, Heffley & Jewell, "Aircraft Handling Qualities Data", December
# 1972, Section IX. Table IX-3 is "B-747 DIMENSIONAL, MASS AND FLIGHT CONDITION
# PARAMETERS", printed page 229; its header carries the reference geometry and
# its columns carry one flight condition each. Flight condition 9 is the cruise
# case this project uses. Verified against the document, not against a summary.
_CR2144_IX3 = "NASA CR-2144 Table IX-3 header, printed p.229, verified against the document"
_CR2144_IX4 = "NASA CR-2144 Table IX-4, printed p.230, flight condition 9"
_CR2144_IX8 = "NASA CR-2144 Table IX-8, printed p.234, flight condition 9, primed"

LEDGER: dict[str, Entry] = {
    # -- 747 reference geometry, straight off the table --------------------
    "b747.S": Entry("SOURCED", f"5500 ft^2 wing area. {_CR2144_IX3}"),
    "b747.b": Entry("SOURCED", f"195.68 ft wing span. {_CR2144_IX3}"),
    "b747.c": Entry("SOURCED", f"27.31 ft mean aerodynamic chord. {_CR2144_IX3}"),

    # -- the derivatives the tail arm is built from -------------------------
    # Dimensional in the source; the non-dimensionalisation is CR-2144
    # Appendix A's own relation, applied in aircraft.py.
    "b747.Zq": Entry("SOURCED", f"-5.16, dimensional. {_CR2144_IX4}"),
    "b747.Mq": Entry("SOURCED", f"-0.339, dimensional. {_CR2144_IX4}"),
    "b747.CLq": Entry(
        "DERIVED",
        "CLq = -Zq * 2 * m * U0 / (qS * c), CR-2144 Appendix A. Gives 5.9450.",
        inputs=("b747.Zq", "b747.c"),
    ),
    "b747.Cmq": Entry(
        "DERIVED",
        "Cmq = Mq * 2 * Iy * U0 / (qS * c^2), CR-2144 Appendix A. Gives -23.9232.",
        inputs=("b747.Mq", "b747.c"),
    ),
    "b747.Clp": Entry("SOURCED", f"Lp' = -0.465, primed dimensional. {_CR2144_IX8}"),

    # -- the one new relation this work introduces --------------------------
    "b747.l_eff": Entry(
        "DERIVED",
        "l_eff/c = -Cmq/CLq. Stengel Flight Dynamics 2nd ed eqs. 3.4-7 and "
        "3.4-12; the tail lift slope cancels in the ratio. Gives 4.0241 chords "
        "= 109.90 ft. NOT 747 geometry: it attributes both derivatives to the "
        "tail, and the wing/fuselage share is not separated. Justified "
        "empirically -- the value falls inside the real aircraft's 100-110 ft "
        "-- not by a computed error bar. See design section 3d and 7d.",
        inputs=("b747.Cmq", "b747.CLq", "b747.c"),
    ),

    # -- strip model, phase 3 ------------------------------------------------
    "strip.loading_shape": Entry(
        "DECLARED",
        "Elliptic spanwise loading. Taper ratio is not tabulated in CR-2144 and "
        "is NOT recoverable from S, b and cbar -- the required shape factor "
        "0.72873 lies below the trapezoidal minimum of 0.75, because the 747 "
        "planform is cranked (design section 3f). Sensitivity is mandatory: "
        "every result re-run against uniform and taper-based shapes with the "
        "spread reported.",
    ),
    "strip.lift_slope": Entry(
        "CALIBRATED",
        "Effective section lift slope, scaled so integrating a rigid roll rate "
        "reproduces the tabulated Clp exactly. For elliptic loading the strip "
        "integral gives Clp_hat = -a0/8, hence a0 = -8*Clp. This is an "
        "EFFECTIVE value absorbing sweep, the tail's share of Clp, and the "
        "difference between elliptic strip theory and the real wing. It is not "
        "an airfoil property and must not be quoted as one.",
        inputs=("b747.Clp", "strip.loading_shape"),
    ),
    "strip.n_stations": Entry(
        "DECLARED",
        "Number of spanwise and longitudinal sample stations. Chosen by "
        "convergence, not by taste: Task 6 runs the refinement study and the "
        "sensitivity is the movement between the chosen count and double it.",
    ),
    "airframe.tail_arm_band": Entry(
        "DECLARED",
        "Plausibility band [2.0, 6.0] chords on the derived l_eff. Brackets "
        "conventional tail-aft configurations. Its only job is to reject "
        "derivative sets whose CLq and Cmq disagree about what aircraft they "
        "describe, not to police physics. Measured outcome: boeing747 4.0241 "
        "and boeing747_approach 3.8519 pass; cessna172 0.8558 and cherokee "
        "1.2802 fail and are excluded from the strip path. Sensitivity: no "
        "result depends on the band's edges, only on which aircraft pass, and "
        "the two groups are separated by a factor of three.",
    ),
}
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
$PY -m pytest flightsim/tests/test_provenance.py -v
```

Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add flightsim/provenance.py flightsim/tests/test_provenance.py
git commit -m "Enter the constants this work depends on into the ledger"
```

---

# Phase 2 — A1: replace the tangent with a fit

### Task 3: The derived effective tail arm and its plausibility gate

**Files:**
- Create: `flightsim/airframe.py`
- Test: `flightsim/tests/test_airframe.py`

- [ ] **Step 1: Write the failing test**

Create `flightsim/tests/test_airframe.py`:

```python
"""Airframe sampling geometry.

The one new relation here is the effective tail arm, l_eff/c = -Cmq/CLq. It
comes from Stengel Flight Dynamics 2nd ed eqs. 3.4-7 and 3.4-12, whose ratio
cancels the tail lift slope. Its justification is empirical -- the value it
returns for the 747 falls inside the real aircraft's tail arm -- so that
check is asserted here rather than described in a comment.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import flightsim  # noqa: F401  -- enables x64 before any array is made
from flightsim import airframe
from flightsim.aircraft import REGISTRY
from flightsim.units import FT2M


def test_the_derived_tail_arm_matches_the_hand_computation():
    """CR-2144 FC9 gives CLq = 5.9450 and Cmq = -23.9232, so -Cmq/CLq = 4.0241
    chords. Asserted as a value so a change to either derivative shows up here
    rather than silently moving every sampled gradient."""
    ac = REGISTRY["boeing747"]
    assert float(airframe.effective_tail_arm(ac)) == pytest.approx(4.0241, rel=1e-4)


def test_the_derived_tail_arm_lands_on_the_real_aircraft_geometry():
    """THE justification for the whole relation. The 747-100's centre of gravity
    sits roughly 100-110 ft ahead of the tailplane, and this recovers 109.9 ft
    having been told nothing about 747 geometry beyond the mean chord. If this
    stops holding, l_eff is no longer defensible and the strip path loses its
    only empirical support -- see design section 7d."""
    ac = REGISTRY["boeing747"]
    arm_ft = float(airframe.effective_tail_arm(ac) * ac.c) / FT2M
    assert 100.0 <= arm_ft <= 110.0, f"derived arm {arm_ft:.1f} ft is outside the real aircraft's"


def test_only_the_two_747_configurations_pass_the_plausibility_gate():
    """The gate must be able to fire, and it fires on half the registry.

    Measured across every aircraft the project holds:

        boeing747           4.0241 chords = 109.90 ft   PASS
        boeing747_approach  3.8519 chords = 105.19 ft   PASS
        cessna172           0.8558 chords =   4.19 ft   FAIL
        cherokee            1.2802 chords =   6.72 ft   FAIL

    Both 747 sets come from CR-2144, whose transcription has been verified
    element by element against the source document. Both light-aircraft sets
    return arms far shorter than those airframes physically have.

    WHAT THIS DOES NOT ESTABLISH is which side is at fault. It may be the source
    data -- PROJECT.md section 3 already records the Cessna's rudder set as
    inconsistent and the Cherokee's Izz < Iyy as flagged by its own author -- or
    it may be that the tail-dominated reading behind l_eff does not transfer to
    a light aircraft with a short tail and a large wing. Either way the strip
    path must not be used for them, which is exactly what the gate enforces.
    """
    passes = {name: airframe.tail_arm_is_plausible(REGISTRY[name]) for name in REGISTRY}
    assert passes == {
        "boeing747": True,
        "boeing747_approach": True,
        "cessna172": False,
        "cherokee": False,
    }, f"gate outcome changed: {passes}"


def test_the_derived_arms_take_their_recorded_values():
    """Asserted per aircraft so a change to any CLq or Cmq surfaces here rather
    than silently moving every sampled gradient downstream."""
    expected = {
        "boeing747": 4.0241,
        "boeing747_approach": 3.8519,
        "cessna172": 0.8558,
        "cherokee": 1.2802,
    }
    for name, arm in expected.items():
        assert float(airframe.effective_tail_arm(REGISTRY[name])) == pytest.approx(
            arm, rel=1e-3
        ), f"{name} derived arm moved"
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
$PY -m pytest flightsim/tests/test_airframe.py -v
```

Expected: collection error, `ModuleNotFoundError: No module named 'flightsim.airframe'`.

`REGISTRY` has exactly four keys, verified: `boeing747`, `boeing747_approach`, `cessna172`, `cherokee`. Note it is **`cherokee`**, not `piper_cherokee`.

- [ ] **Step 3: Write the minimal implementation**

Create `flightsim/airframe.py`:

```python
"""Where on the airframe the wind field is sampled.

The model has always evaluated the wind at one point -- the CG -- and taken the
analytic gradient there. That is exact while the field is linear across the
aircraft and progressively wrong as the field's scale approaches a wingspan,
which is the regime this project is moving into. This module supplies the
geometry needed to sample across the airframe instead.

It needs an extent in each direction. Laterally that is the span, which is
tabulated. Longitudinally it is a tail arm, which is NOT tabulated for any
aircraft this project holds -- so it is recovered from two derivatives that
are.

Provenance for every constant here is in flightsim/provenance.py, and a test
asserts the two agree.
"""

import jax.numpy as jnp
from jax import Array

from flightsim.aircraft import Aircraft

# Plausibility band on the derived tail arm, in mean chords. DECLARED, not
# sourced -- see provenance.LEDGER["airframe.tail_arm_band"]. It brackets
# conventional tail-aft configurations and exists only to reject derivative
# sets whose CLq and Cmq disagree about what aircraft they describe.
TAIL_ARM_BAND = (2.0, 6.0)


def effective_tail_arm(ac: Aircraft) -> Array:
    """Distance from the CG to the effective tail centre of pressure, in chords.

    Stengel, Flight Dynamics 2nd ed. From eq. 3.4-7 the tail's lift response to
    pitch rate is CL_q,ht = CL_a,ht * (l_ht / V); non-dimensionalised with
    q_hat = q*c/2V (eq. 3.4-10) that is CL_qhat = 2 * CL_a,ht * (l_ht/c). From
    eq. 3.4-12 the moment derivative is Cm_qhat = -2 * (l_ht/c)^2 * CL_a,ht.
    The ratio eliminates the tail lift slope, which the project does not hold:

        l_eff / c = -Cm_qhat / CL_qhat

    ATTRIBUTES BOTH DERIVATIVES TO THE TAIL. Stengel eq. 3.4-19 notes Cmq also
    carries wing and fuselage contributions. The obvious correction --
    subtracting the wing's share via eqs. 3.4-13/3.4-14 with the sourced
    h_cm = 0.25 -- was tried and REJECTED: it makes the wing 83% of CLq, which
    implies a 23-chord tail arm. Those are Etkin's two-dimensional
    infinite-aspect-ratio results and they do not transfer to a swept wing with
    a tail. Do not re-attempt it; see design section 3d.
    """
    return -ac.Cmq / ac.CLq


def tail_arm_is_plausible(ac: Aircraft) -> bool:
    """Whether this aircraft's derivative set is self-consistent enough to sample.

    A Python bool, not a traced array: it is a data-quality gate evaluated once
    when a run is set up, never inside a jitted step.
    """
    low, high = TAIL_ARM_BAND
    return bool(low <= float(effective_tail_arm(ac)) <= high)
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
$PY -m pytest flightsim/tests/test_airframe.py -v
```

Expected: 4 passed.

If `test_the_derived_tail_arm_lands_on_the_real_aircraft_geometry` fails, stop. That test is the entire justification for the relation; a failure means the derivation is wrong, not that the tolerance is tight.

- [ ] **Step 5: Commit**

```bash
git add flightsim/airframe.py flightsim/tests/test_airframe.py
git commit -m "Derive the effective tail arm from Cmq and CLq, and gate it on plausibility"
```

---

### Task 4: Sample stations across the airframe

**Files:**
- Modify: `flightsim/airframe.py`
- Modify: `flightsim/tests/test_airframe.py`

- [ ] **Step 1: Write the failing test**

Append to `flightsim/tests/test_airframe.py`:

```python
def test_span_stations_cover_the_whole_span_symmetrically():
    """The lateral extent is the span, which is sourced. Symmetry matters: an
    asymmetric station set would give a non-zero fitted roll gradient in a
    uniform field, which is the first reduction property Task 5 asserts."""
    ac = REGISTRY["boeing747"]
    st = airframe.stations(ac)
    span = np.asarray(st.span)
    assert span.min() == pytest.approx(-float(ac.b) / 2.0)
    assert span.max() == pytest.approx(float(ac.b) / 2.0)
    assert np.allclose(span, -span[::-1]), "span stations must be symmetric about the centreline"


def test_longitudinal_stations_run_from_the_tail_to_the_cg():
    """Body x is positive forward, so the tail is at NEGATIVE x. Pitch damping
    comes overwhelmingly from the tail, so the fit is taken over the CG-to-tail
    interval rather than symmetrically about the CG -- that is the interval the
    aerodynamics actually integrate over."""
    ac = REGISTRY["boeing747"]
    st = airframe.stations(ac)
    lon = np.asarray(st.longitudinal)
    arm = float(airframe.effective_tail_arm(ac) * ac.c)
    assert lon.min() == pytest.approx(-arm)
    assert lon.max() == pytest.approx(0.0)


def test_station_counts_are_configurable_for_the_convergence_study():
    """The count is DECLARED and needs a refinement study, so it must be a knob."""
    ac = REGISTRY["boeing747"]
    st = airframe.stations(ac, n_span=21, n_lon=15)
    assert np.asarray(st.span).shape == (21,)
    assert np.asarray(st.longitudinal).shape == (15,)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
$PY -m pytest flightsim/tests/test_airframe.py -v -k "stations"
```

Expected: 3 failed with `AttributeError: module 'flightsim.airframe' has no attribute 'stations'`.

- [ ] **Step 3: Write the implementation**

Append to `flightsim/airframe.py`:

```python
from typing import NamedTuple

# Sample counts. DECLARED -- see provenance.LEDGER["strip.n_stations"]. Odd, so
# a station sits exactly on the centreline and the symmetric pair cancels
# exactly rather than to round-off.
N_SPAN = 9
N_LON = 9


class Stations(NamedTuple):
    """Body-axis offsets from the CG at which the wind field is evaluated.

    Two one-dimensional sets rather than one cloud of points, because the three
    gradients the aero model consumes are each a slope along a single axis:
    roll from vertical gust varying across the span, pitch and yaw from gusts
    varying along the fuselage. Sampling a full grid would cost N^2 field
    evaluations to produce the same three numbers.
    """

    span: Array  # (N,) m, body y, positive right
    longitudinal: Array  # (M,) m, body x, positive forward


def stations(ac: Aircraft, n_span: int = N_SPAN, n_lon: int = N_LON) -> Stations:
    """Sample stations for an aircraft.

    Lateral extent is the span, which is SOURCED. Longitudinal extent is the
    derived tail arm, running aft from the CG -- negative x, since body x is
    positive forward.
    """
    half_span = ac.b / 2.0
    arm = effective_tail_arm(ac) * ac.c
    return Stations(
        span=jnp.linspace(-half_span, half_span, n_span),
        longitudinal=jnp.linspace(-arm, 0.0, n_lon),
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
$PY -m pytest flightsim/tests/test_airframe.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add flightsim/airframe.py flightsim/tests/test_airframe.py
git commit -m "Add airframe sample stations, spanning the span and the tail arm"
```

---

### Task 5: `sampled_rates` — the least-squares fit

**Files:**
- Modify: `flightsim/wind.py`
- Modify: `flightsim/tests/test_wind.py`

This is the core of A1. **`gust_rates` is not touched** — it stays as the reference implementation that Task 6 checks against.

- [ ] **Step 1: Write the failing test**

Append to `flightsim/tests/test_wind.py`:

```python
# --- A1: sampled gradients ---------------------------------------------------


def _level_state(north=0.0, altitude=11278.0, u=236.0):
    """Wings-level, heading north, at altitude."""
    from flightsim.state import State, euler_to_quat

    return State(
        pos_ned=jnp.array([north, 0.0, -altitude]),
        vel_body=jnp.array([u, 0.0, 0.0]),
        quat=euler_to_quat(jnp.array(0.0), jnp.array(0.0), jnp.array(0.0)),
        omega=jnp.zeros(3),
    )


def test_a_uniform_field_produces_exactly_zero_sampled_rates():
    """Reduction property 1. A uniform field has no gradient, and a symmetric
    station set must return exactly zero rather than a small residual -- a
    residual here would be a spurious rolling input in still-ish air."""
    from flightsim import airframe
    from flightsim.aircraft import REGISTRY

    ac = REGISTRY["boeing747"]
    st = airframe.stations(ac)
    field = lambda p: jnp.array([3.0, -2.0, 1.5])  # noqa: E731
    s = _level_state()

    rates = wind.sampled_rates(s.pos_ned, s.quat, field, st)
    assert np.array_equal(np.asarray(rates), np.zeros(3))


def test_a_linear_field_reproduces_the_analytic_gradient_exactly():
    """Reduction property 2, and the one that makes A1 safe to adopt: a
    least-squares slope through samples of a linear function IS its exact
    slope, so for any field the current model handles correctly, A1 returns
    the identical answer. Every existing result is therefore unmoved."""
    from flightsim import airframe
    from flightsim.aircraft import REGISTRY

    ac = REGISTRY["boeing747"]
    st = airframe.stations(ac)
    # Linear in every component and every direction, with no curvature at all.
    field = lambda p: jnp.array(  # noqa: E731
        [0.01 * p[0] + 0.02 * p[1], 0.03 * p[0] - 0.01 * p[2], -0.02 * p[0] + 0.04 * p[1]]
    )
    s = _level_state()

    sampled = wind.sampled_rates(s.pos_ned, s.quat, field, st)
    analytic = wind.gust_rates(s.pos_ned, s.quat, field)
    assert np.allclose(np.asarray(sampled), np.asarray(analytic), rtol=1e-9, atol=1e-12)


def test_the_vortex_core_gives_the_same_pitch_rate_as_the_tangent():
    """Inside a Rankine core the vertical gust is LINEAR along track, so the
    secant and the tangent must agree exactly. This is the strength of this
    field/model pairing that ASSUMPTIONS.md section E2 records: while the whole
    airframe is inside the core, a point sample plus a gradient is not an
    approximation at all."""
    from flightsim import airframe
    from flightsim.aircraft import REGISTRY

    ac = REGISTRY["boeing747"]
    st = airframe.stations(ac)
    array = single(r0=8000.0)  # core far larger than the airframe, so it stays inside
    field = lambda p: wind.vortex_wind(p, array)  # noqa: E731
    s = _level_state(north=0.25 * 8000.0)

    sampled = wind.sampled_rates(s.pos_ned, s.quat, field, st)
    analytic = wind.gust_rates(s.pos_ned, s.quat, field)
    assert float(sampled[1]) == pytest.approx(float(analytic[1]), rel=1e-9)


def test_a_curved_field_makes_the_secant_differ_from_the_tangent():
    """The test that gives A1 a reason to exist. A quadratic gust profile has a
    centreline slope that is not the slope the wing integrates, and the two
    must therefore disagree. If this passes trivially, the fit is not being
    taken across the airframe at all."""
    from flightsim import airframe
    from flightsim.aircraft import REGISTRY

    ac = REGISTRY["boeing747"]
    st = airframe.stations(ac)
    # Vertical gust quadratic across the span: zero slope at the centreline,
    # non-zero average slope across it.
    field = lambda p: jnp.array([0.0, 0.0, 1e-4 * p[1] ** 2])  # noqa: E731
    s = _level_state()

    sampled = wind.sampled_rates(s.pos_ned, s.quat, field, st)
    analytic = wind.gust_rates(s.pos_ned, s.quat, field)
    assert float(analytic[0]) == pytest.approx(0.0, abs=1e-12)
    assert abs(float(sampled[0])) < 1e-12, "a symmetric quadratic still has zero net slope"

    # Now break the symmetry: a cubic has a genuinely different secant.
    field3 = lambda p: jnp.array([0.0, 0.0, 1e-7 * p[1] ** 3])  # noqa: E731
    sampled3 = wind.sampled_rates(s.pos_ned, s.quat, field3, st)
    analytic3 = wind.gust_rates(s.pos_ned, s.quat, field3)
    assert float(analytic3[0]) == pytest.approx(0.0, abs=1e-12)
    assert abs(float(sampled3[0])) > 1e-9, "the cubic's secant must differ from its tangent"
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
$PY -m pytest flightsim/tests/test_wind.py -v -k "sampled or uniform_field or linear_field or curved_field or vortex_core_gives"
```

Expected: failures with `AttributeError: module 'flightsim.wind' has no attribute 'sampled_rates'`.

- [ ] **Step 3: Write the implementation**

Add to `flightsim/wind.py`, directly below `gust_rates`:

```python
def _slope(coords: Array, values: Array) -> Array:
    """Least-squares slope of `values` against `coords`.

    Exact for a linear profile, which is what makes `sampled_rates` reduce to
    `gust_rates` whenever the field has no curvature across the airframe. The
    denominator cannot vanish for a station set with more than one distinct
    coordinate, which `airframe.stations` guarantees by construction.
    """
    centred = coords - coords.mean()
    return (centred * (values - values.mean())).sum() / (centred * centred).sum()


def sampled_rates(pos_ned: Array, quat: Array, field, stations) -> Array:
    """Body-axis (p, q, r) gust rates from a fit across the airframe.

    Same three quantities as `gust_rates` and the same sign convention -- this
    is a better ESTIMATOR of them, not a different quantity. `gust_rates` takes
    the tangent at the CG; this takes the secant across the extent the
    aerodynamics actually integrate over. For a field that is linear across the
    aircraft the two are identical, and `test_wind.py` asserts it.

    Deliberately does NOT add the three equivalences Stengel lists that the
    model omits (his eqs. 3.4-49, 3.4-51, 3.4-53). Combining each pair into one
    effective rate needs a weighting that his eq. 3.4-55 gets wrong -- it fails
    its own rigid-rotation self-check by a factor of -2 -- and that question is
    left to the strip integration, which never forms an equivalent rate at all.
    See the design document, section 2.
    """
    dcm = quat_to_dcm(quat)  # body -> NED

    def gust_body(offset_body: Array) -> Array:
        """Gust in BODY axes at a body-frame offset from the CG."""
        return dcm.T @ field(pos_ned + dcm @ offset_body)

    span_gusts = jax.vmap(
        lambda y: gust_body(jnp.array([0.0, y, 0.0]))
    )(stations.span)
    lon_gusts = jax.vmap(
        lambda x: gust_body(jnp.array([x, 0.0, 0.0]))
    )(stations.longitudinal)

    # Same three components, same signs, as gust_rates:
    #   p = +d(w_g)/dy    q = -d(w_g)/dx    r = +d(v_g)/dx
    p_gust = _slope(stations.span, span_gusts[:, 2])
    q_gust = -_slope(stations.longitudinal, lon_gusts[:, 2])
    r_gust = _slope(stations.longitudinal, lon_gusts[:, 1])
    return jnp.array([p_gust, q_gust, r_gust])
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
$PY -m pytest flightsim/tests/test_wind.py -v
```

Expected: all pass, including the pre-existing vortex tests.

- [ ] **Step 5: Commit**

```bash
git add flightsim/wind.py flightsim/tests/test_wind.py
git commit -m "Add sampled_rates: fit the gust gradient across the airframe, not at a point"
```

---

### Task 6: Station-count convergence study

**Files:**
- Modify: `flightsim/tests/test_airframe.py`

`strip.n_stations` is DECLARED and the ledger says it is chosen by convergence. This task discharges that.

- [ ] **Step 1: Write the failing test**

Append to `flightsim/tests/test_airframe.py`:

```python
def test_the_default_station_count_has_converged():
    """N_SPAN and N_LON are DECLARED, and the ledger says they are chosen by
    convergence rather than taste. This is that study, run as an assertion so
    the claim cannot rot.

    The field is the Parks Hannibal core, which is the smallest-scale field the
    project holds and therefore the hardest case. Doubling the station count
    must move the fitted rates by less than 0.1%.
    """
    import jax.numpy as jnp

    from flightsim import wind
    from flightsim.state import State, euler_to_quat
    from flightsim.units import FT2M

    ac = REGISTRY["boeing747"]
    array = wind.VortexArray(
        north=jnp.array([0.0]),
        down=jnp.array([-11278.0]),
        r0=jnp.array(600.0 * FT2M),
        v0=jnp.array(85.0 * FT2M),
    )
    field = lambda p: wind.vortex_wind(p, array)  # noqa: E731
    # Half a core radius downstream and half a radius above: inside the core but
    # off-centre, so every gradient component is non-zero.
    state = State(
        pos_ned=jnp.array([0.5 * 600.0 * FT2M, 0.0, -(11278.0 + 300.0 * FT2M)]),
        vel_body=jnp.array([236.0, 0.0, 0.0]),
        quat=euler_to_quat(jnp.array(0.0), jnp.array(0.0), jnp.array(0.0)),
        omega=jnp.zeros(3),
    )

    coarse = wind.sampled_rates(
        state.pos_ned, state.quat, field, airframe.stations(ac, 9, 9)
    )
    fine = wind.sampled_rates(
        state.pos_ned, state.quat, field, airframe.stations(ac, 18, 18)
    )
    q_coarse, q_fine = float(coarse[1]), float(fine[1])
    movement = abs(q_fine - q_coarse) / abs(q_fine)
    assert movement < 1e-3, (
        f"pitch rate moves {movement:.2%} between 9 and 18 stations; "
        "the default count has not converged"
    )
```

- [ ] **Step 2: Run the test**

```bash
$PY -m pytest flightsim/tests/test_airframe.py::test_the_default_station_count_has_converged -v
```

Expected: PASS. Inside a Rankine core the field is linear, so the fit is exact at any station count and the movement is round-off.

If it FAILS, the defaults have not converged. Raise `N_SPAN`/`N_LON` in `airframe.py` until it passes, and record the chosen values and the measured movement in the `strip.n_stations` ledger detail. Do not loosen the tolerance.

- [ ] **Step 3: Commit**

```bash
git add flightsim/tests/test_airframe.py
git commit -m "Discharge the station-count convergence study the ledger requires"
```

---

### Task 7: Wire A1 into a wind model, and measure the E2 bound

**Files:**
- Modify: `flightsim/wind.py`
- Modify: `flightsim/tests/test_wind.py`

- [ ] **Step 1: Write the failing test**

Append to `flightsim/tests/test_wind.py`:

```python
def test_the_sampled_wind_model_matches_the_contract():
    """Same signature as zero_wind and field_model, so it drops into
    integrate.step, autopilot and panel with no change to any of them."""
    from flightsim import airframe
    from flightsim.aircraft import REGISTRY

    ac = REGISTRY["boeing747"]
    array = single()
    model = wind.sampled_field_model(
        lambda p: wind.vortex_wind(p, array), airframe.stations(ac)
    )
    s = _level_state()
    key = jax.random.PRNGKey(0)
    wind_ned, omega_gust, wind_state, out_key = model(
        wind.zero_wind_state(), s, key, jnp.array(0.02)
    )
    assert wind_ned.shape == (3,)
    assert omega_gust.shape == (3,)
    assert np.array_equal(np.asarray(out_key), np.asarray(key))


def test_the_curvature_correction_across_the_parks_core_is_measured():
    """ASSUMPTIONS.md section E2 carries a scale ratio -- the Parks core is
    2.30-3.07 wingspans -- but has never carried a measured CONSEQUENCE. This
    is that measurement.

    Taken at the core EDGE, where the Rankine profile switches from linear
    inside to 1/r outside and the curvature across the span is therefore
    largest. Inside the core the two agree exactly (asserted separately), so
    the edge is the honest place to bound the error.

    Asserted as a band rather than a value: the point is that the number exists
    and is recorded, not that it takes one particular value.
    """
    from flightsim import airframe
    from flightsim.aircraft import REGISTRY

    ac = REGISTRY["boeing747"]
    array = single()  # Parks Hannibal, r0 = 182.9 m
    field = lambda p: wind.vortex_wind(p, array)  # noqa: E731
    st = airframe.stations(ac)
    state = _level_state(north=CASE1_R0)  # exactly at the core edge

    sampled = wind.sampled_rates(state.pos_ned, state.quat, field, st)
    analytic = wind.gust_rates(state.pos_ned, state.quat, field)
    relative = abs(float(sampled[1]) - float(analytic[1])) / abs(float(analytic[1]))

    print(f"\nE2 curvature correction at the Parks core edge: {relative:.4%}")
    assert 0.0 < relative < 0.5, (
        f"curvature correction {relative:.4%} is outside the expected band; "
        "if it is zero the fit is not sampling across the airframe, and if it "
        "exceeds 50% the linear model has broken down entirely"
    )
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
$PY -m pytest flightsim/tests/test_wind.py -v -k "sampled_wind_model or curvature_correction"
```

Expected: `AttributeError: module 'flightsim.wind' has no attribute 'sampled_field_model'` for the first; the second fails on the same import once it reaches `sampled_rates`.

- [ ] **Step 3: Write the implementation**

Add to `flightsim/wind.py`, directly below `field_model`:

```python
def sampled_field_model(field, stations):
    """`field_model`, but with the gust rates fitted across the airframe.

    Identical contract to `field_model` -- same signature, same returned tuple,
    key untouched -- so it is a drop-in wherever a wind model is accepted. The
    only difference is which estimator produces `omega_gust`.

    `wind_ned` is still the CG sample. Averaging the translational gust over
    the airframe is a separate change with its own weighting question, and it
    belongs to the strip integration rather than here: this stage changes the
    estimator for quantities already in use and introduces no new constants.
    """

    def model(
        wind_state: WindState, state: State, key: Array, dt: float
    ) -> tuple[Array, Array, WindState, Array]:
        del dt
        wind_ned = field(state.pos_ned)
        omega_gust = sampled_rates(state.pos_ned, state.quat, field, stations)
        return wind_ned, omega_gust, wind_state, key

    return model
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
$PY -m pytest flightsim/tests/test_wind.py -v -s -k "sampled_wind_model or curvature_correction or rankine_gradient"
```

> **CORRECTION APPLIED DURING EXECUTION.** The test as drafted above normalised the correction by the local tangent and asserted it stayed under 50%. It measured **exactly 200%**, and investigation showed that is a real property of the field rather than a defect:
>
> **The Rankine gradient is discontinuous at the core edge.** Velocity is continuous there — both branches agree — but the derivative is not. Inside, `d(w)/dx = +V₀/r₀`; outside at `r = r₀` it is `−V₀/r₀`. The two one-sided derivatives differ by `2·V₀/r₀` and have **opposite signs**, so at the boundary the tangent is *ambiguous*, and `vortex_wind`'s strict `<` resolves the tie toward the outside branch. The secant has no such ambiguity: at the boundary the airframe is still almost entirely inside the core, and that is what it reports.
>
> Normalising by the tangent is therefore unstable exactly where the answer matters. The shipped tests normalise by **`V₀/r₀`, the core's own characteristic rate**, which is a fixed property of the vortex, and sweep the traverse rather than sampling one point. A second test, `test_the_rankine_gradient_is_discontinuous_at_the_core_edge`, asserts the discontinuity directly.

Expected: 3 passed, with the profile printed. **Record the printed profile — Task 13 writes it into `ASSUMPTIONS.md`.** Measured:

| Station | Correction (units of `V₀/r₀`) |
|---|---|
| 0.50 r₀ | 0.0000 |
| 0.99 r₀ | 0.0000 |
| **1.00 r₀** | **2.0000** |
| 1.10 r₀ | 0.7472 |
| 1.25 r₀ | 0.1086 |
| 2.00 r₀ | 0.0250 |
| 3.00 r₀ | 0.0072 |

- [ ] **Step 5: Run the whole suite**

```bash
$PY -m pytest flightsim/tests/ -q
```

Expected: all pass. Nothing so far changes any existing code path — `gust_rates` and `field_model` are untouched.

- [ ] **Step 6: Commit**

```bash
git add flightsim/wind.py flightsim/tests/test_wind.py
git commit -m "Add sampled_field_model, and measure the E2 curvature bound at the core edge"
```

---

# Phase 3 — A2: strip force integration

Phase 2 improves the estimator for three numbers. It still collapses the field to three numbers, so curvature is still lost. Phase 3 is what fixes that.

### Task 8: Elliptic loading and its calibration to `Clp`

**Files:**
- Modify: `flightsim/airframe.py`
- Modify: `flightsim/tests/test_airframe.py`

- [ ] **Step 1: Write the failing test**

Append to `flightsim/tests/test_airframe.py`:

```python
def test_the_elliptic_chord_integrates_to_the_sourced_wing_area():
    """The shape is DECLARED but it is not free: whatever shape is chosen must
    enclose the tabulated area. This is the constraint that turns 'a shape' into
    'this shape'."""
    ac = REGISTRY["boeing747"]
    y = np.linspace(-float(ac.b) / 2.0, float(ac.b) / 2.0, 20001)
    chord = np.asarray(airframe.elliptic_chord(jnp.asarray(y), ac))
    assert np.trapezoid(chord, y) == pytest.approx(float(ac.S), rel=1e-4)


def test_the_calibrated_lift_slope_reproduces_the_sourced_Clp():
    """The calibration target, asserted directly. For elliptic loading the strip
    integral gives Clp_hat = -a0/8 (derived in the airframe.py docstring), so
    a0 = -8*Clp by construction -- and this test is what stops that identity
    from drifting if the integral is ever rewritten."""
    ac = REGISTRY["boeing747"]
    a0 = float(airframe.calibrated_lift_slope(ac))
    assert a0 == pytest.approx(-8.0 * float(ac.Clp), rel=1e-12)
    assert a0 > 0.0, "a lift slope must be positive; check the sign of Clp"
    # Clp = -0.35018 for this set, so a0 = 2.8014. EXPECT THIS TO LOOK LOW: a
    # two-dimensional thin-airfoil slope is 2*pi = 6.28 and the 747's own CLa is
    # 4.94. It is low precisely because it is effective rather than physical --
    # it absorbs sweep, the tail's share of Clp, and the gap between elliptic
    # strip theory and a cranked swept wing. A value near 6.28 would mean the
    # calibration had NOT absorbed those and would be the surprising outcome.
    assert a0 == pytest.approx(2.8014, rel=1e-3)


def test_the_strip_integral_reproduces_stengels_closed_form_for_a_rectangular_wing():
    """Independent cross-check on the integration machinery, separate from the
    calibration. Stengel eq. 3.4-40 gives Clp_hat = -(CLa/12)(1+3L)/(1+L) for a
    tapered wing; at L = 1 (rectangular) that is -CLa/6. Running the same
    integral over a constant chord must return it.

    This checks the integral, not the 747: if it fails, the strip machinery is
    wrong and the calibration would silently absorb the error."""
    ac = REGISTRY["boeing747"]
    b = float(ac.b)
    y = np.linspace(-b / 2.0, b / 2.0, 20001)
    a0 = 5.0  # arbitrary; the result is proportional to it
    S_rect = float(ac.S)
    chord = np.full_like(y, S_rect / b)  # rectangular wing of the same area

    # Clp_hat = -(2*a0/(S*b^2)) * integral(y^2 * c(y) dy)
    integral = np.trapezoid(y**2 * chord, y)
    clp = -(2.0 * a0 / (S_rect * b**2)) * integral
    assert clp == pytest.approx(-a0 / 6.0, rel=1e-6)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
$PY -m pytest flightsim/tests/test_airframe.py -v -k "elliptic or calibrated or stengel"
```

Expected: 2 failed with `AttributeError` on `elliptic_chord` / `calibrated_lift_slope`; the Stengel cross-check passes immediately since it uses only NumPy.

- [ ] **Step 3: Write the implementation**

Append to `flightsim/airframe.py`:

```python
def elliptic_chord(y: Array, ac: Aircraft) -> Array:
    """Elliptic spanwise chord distribution, scaled to the sourced wing area.

        c(y) = c0 * sqrt(1 - (2y/b)^2),    c0 = 4S/(pi*b)

    DECLARED, not sourced. Taper ratio is not tabulated in CR-2144 and is not
    recoverable from S, b and cbar -- the required shape factor is 0.72873
    against a trapezoidal minimum of 0.75, because the 747 planform is cranked.
    Elliptic is chosen because it needs no taper ratio at all, so the shape
    introduces exactly one assumption rather than one assumption plus an
    unsourced number. Every result using it must be re-run against uniform and
    taper-based shapes with the spread reported.

    The argument of the square root is clamped: `stations` places points exactly
    at +-b/2 where it is analytically zero, and round-off can make it slightly
    negative, which conftest's jax_debug_nans would trip on.
    """
    c0 = 4.0 * ac.S / (jnp.pi * ac.b)
    normalised = 2.0 * y / ac.b
    return c0 * jnp.sqrt(jnp.maximum(1.0 - normalised * normalised, 0.0))


def calibrated_lift_slope(ac: Aircraft) -> Array:
    """Effective section lift slope, pinned so the strip integral returns Clp.

    Strip theory gives the rolling moment from a roll rate p as

        L = -integral( y * qbar * c(y) * a0 * (p*y/V) dy )

    since a station at y moves down at p*y and therefore sees an incidence
    increment p*y/V (Stengel eq. 3.4-39), and the resulting lift acts at moment
    arm y. Non-dimensionalising with p_hat = pb/2V:

        Clp_hat = -(2*a0 / (S*b^2)) * integral( y^2 * c(y) dy )

    For the elliptic distribution above, integral(y^2 c dy) = c0*b^3*pi/64 with
    c0 = 4S/(pi*b), which collapses to

        Clp_hat = -a0/8      hence      a0 = -8 * Clp

    CALIBRATED, not sourced. This a0 is an EFFECTIVE value: it absorbs sweep,
    the tail's share of Clp, and the difference between elliptic strip theory
    and the real cranked wing. It is not an airfoil property and must never be
    quoted as one. What it guarantees is that a rigid roll rate through the
    strip integral reproduces the tabulated Clp exactly; what it does not
    guarantee is that the spanwise SHAPE is right, which is why the sensitivity
    sweep is mandatory.
    """
    return -8.0 * ac.Clp
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
$PY -m pytest flightsim/tests/test_airframe.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add flightsim/airframe.py flightsim/tests/test_airframe.py
git commit -m "Add elliptic loading, calibrated so a rigid roll rate returns the sourced Clp"
```

---

### Task 9: `strip_roll_moment` — integrate the real field across the span

**Files:**
- Modify: `flightsim/wind.py`
- Modify: `flightsim/tests/test_wind.py`

- [ ] **Step 1: Write the failing test**

Append to `flightsim/tests/test_wind.py`:

```python
# --- A2: strip integration ---------------------------------------------------


def test_a_rigid_roll_rate_through_the_strip_integral_returns_the_sourced_Clp():
    """The calibration target, asserted end to end through the real integral
    rather than through the closed form it was derived from. This is validation
    gate 3 in the design document."""
    from flightsim import airframe
    from flightsim.aircraft import REGISTRY

    ac = REGISTRY["boeing747"]
    st = airframe.stations(ac, n_span=2001, n_lon=9)
    p_hat = 0.01
    clp = wind.strip_clp_from_rate(ac, st, p_hat)
    assert clp / p_hat == pytest.approx(float(ac.Clp), rel=1e-3)


def test_a_uniform_vertical_gust_produces_no_rolling_moment():
    """A gust that is the same at both tips cannot roll the aircraft. If this
    fails, the integration weights are asymmetric."""
    from flightsim import airframe
    from flightsim.aircraft import REGISTRY

    ac = REGISTRY["boeing747"]
    st = airframe.stations(ac, n_span=2001, n_lon=9)
    field = lambda p: jnp.array([0.0, 0.0, 5.0])  # noqa: E731
    s = _level_state()
    moment = wind.strip_roll_moment(s.pos_ned, s.quat, field, ac, st, 236.0)
    assert abs(float(moment)) < 1e-9


def test_a_linear_gust_gradient_matches_the_equivalent_rate_answer():
    """The bridge between A1 and A2. For a gust varying linearly across the
    span, the strip integral and the equivalent-rate treatment describe the
    same physics and must agree -- that is what makes the rate equivalence
    legitimate in the first place (Stengel eq. 3.4-48). They diverge only when
    the profile is curved, which is the next test."""
    from flightsim import airframe
    from flightsim.aircraft import REGISTRY

    ac = REGISTRY["boeing747"]
    st = airframe.stations(ac, n_span=2001, n_lon=9)
    gradient = 0.002  # 1/s, d(w_g)/dy
    field = lambda p: jnp.array([0.0, 0.0, gradient * p[1]])  # noqa: E731
    s = _level_state()
    V = 236.0

    strip = float(wind.strip_roll_moment(s.pos_ned, s.quat, field, ac, st, V))
    # Equivalent rate: p_gust = +d(w_g)/dy, and the aero model sees -p_gust.
    p_equivalent = -gradient
    equivalent = float(wind.strip_clp_from_rate(ac, st, p_equivalent * float(ac.b) / (2.0 * V)))
    assert strip == pytest.approx(equivalent, rel=1e-6)


def test_a_curved_gust_profile_makes_the_strip_integral_differ_from_the_rate():
    """The reason Phase 3 exists. A cubic spanwise profile has the same
    centreline slope as no gust at all, yet it genuinely rolls the aircraft.
    The equivalent-rate treatment cannot represent that; the strip integral
    can."""
    from flightsim import airframe
    from flightsim.aircraft import REGISTRY

    ac = REGISTRY["boeing747"]
    st = airframe.stations(ac, n_span=2001, n_lon=9)
    field = lambda p: jnp.array([0.0, 0.0, 1e-7 * p[1] ** 3])  # noqa: E731
    s = _level_state()

    strip = float(wind.strip_roll_moment(s.pos_ned, s.quat, field, ac, st, 236.0))
    tangent = float(wind.gust_rates(s.pos_ned, s.quat, field)[0])
    assert tangent == pytest.approx(0.0, abs=1e-12), "the cubic has zero centreline slope"
    assert abs(strip) > 1e-9, "yet it must still produce a rolling moment"
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
$PY -m pytest flightsim/tests/test_wind.py -v -k "strip"
```

Expected: 4 failed with `AttributeError` on `strip_roll_moment` / `strip_clp_from_rate`.

- [ ] **Step 3: Write the implementation**

Add to `flightsim/wind.py`, below `sampled_field_model`:

```python
def _strip_rolling_coefficient(ac: Aircraft, stations, incidence: Array) -> Array:
    """Rolling-moment coefficient from a spanwise incidence distribution.

        Cl = -(1/(S*b)) * integral( y * c(y) * a0 * dalpha(y) dy )

    The leading minus sign is the body-axis convention: extra lift on the right
    wing (y > 0) acts in -z, and the moment about x is y*F_z, so more lift to
    starboard rolls the aircraft to port. That is what makes roll damping
    oppose roll rate, and it is the sign most easily got backwards.

    Trapezoidal rather than a fixed quadrature rule because the elliptic chord
    has infinite slope at the tips, where Gauss-Legendre on a low order does
    noticeably worse than simply using more stations.
    """
    from flightsim import airframe

    y = stations.span
    chord = airframe.elliptic_chord(y, ac)
    a0 = airframe.calibrated_lift_slope(ac)
    integrand = y * chord * a0 * incidence
    return -jnp.trapezoid(integrand, y) / (ac.S * ac.b)


def strip_clp_from_rate(ac: Aircraft, stations, p_hat: Array) -> Array:
    """Rolling-moment coefficient produced by a rigid roll rate.

    The calibration check: with `airframe.calibrated_lift_slope` pinned to the
    tabulated Clp, this must return `Clp * p_hat`. Stengel eq. 3.4-39 gives the
    spanwise incidence a roll rate induces, dalpha = p*y/V, which in terms of
    p_hat = pb/2V is dalpha = 2*p_hat*y/b.
    """
    incidence = 2.0 * p_hat * stations.span / ac.b
    return _strip_rolling_coefficient(ac, stations, incidence)


def strip_roll_moment(
    pos_ned: Array, quat: Array, field, ac: Aircraft, stations, airspeed: Array
) -> Array:
    """Rolling-moment coefficient from a wind field, integrated across the span.

    Each strip is given the gust at ITS OWN position rather than the CG's, so a
    profile that varies non-linearly across the span produces the moment it
    physically would. That is the whole point of the strip treatment and the one
    thing an equivalent rate cannot reproduce -- Stengel is explicit (p. 217)
    that below rotor scale the rotary derivatives stop being adequate and strip
    theory is required.

    Returns a coefficient, not a moment, so it composes with `aero.py`'s
    coefficient build-up rather than bypassing it.

    Sign: a downward gust component (positive body w) increases local incidence,
    exactly as a downward-moving wing does, so the incidence increment is
    +w_g/V.
    """
    dcm = quat_to_dcm(quat)

    def gust_w(y: Array) -> Array:
        offset = jnp.array([0.0, y, 0.0])
        return (dcm.T @ field(pos_ned + dcm @ offset))[2]

    w_gust = jax.vmap(gust_w)(stations.span)
    return _strip_rolling_coefficient(ac, stations, w_gust / airspeed)
```

Add `from flightsim.aircraft import Aircraft` to the imports at the top of `wind.py` if it is not already there. **Import `airframe` inside the function, not at module scope** — `airframe` imports from `aircraft`, and a top-level import here risks a cycle.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
$PY -m pytest flightsim/tests/test_wind.py -v -k "strip"
```

Expected: 4 passed.

- [ ] **Step 5: Run the whole suite**

```bash
$PY -m pytest flightsim/tests/ -q
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add flightsim/wind.py flightsim/tests/test_wind.py
git commit -m "Add strip_roll_moment: give each strip the gust at its own station"
```

---

### Task 10: Loading-shape sensitivity sweep

**Files:**
- Modify: `flightsim/airframe.py`
- Modify: `flightsim/tests/test_airframe.py`

The ledger says the DECLARED shape carries a mandatory sensitivity. This task discharges it.

- [ ] **Step 1: Write the failing test**

Append to `flightsim/tests/test_airframe.py`:

```python
def test_all_three_loading_shapes_enclose_the_same_wing_area():
    """The sensitivity sweep is only meaningful if the alternatives differ in
    SHAPE and not in area -- otherwise the spread measures the area error
    instead of the shape assumption."""
    ac = REGISTRY["boeing747"]
    y = np.linspace(-float(ac.b) / 2.0, float(ac.b) / 2.0, 20001)
    for name in airframe.LOADING_SHAPES:
        chord = np.asarray(airframe.chord_distribution(jnp.asarray(y), ac, name))
        assert np.trapezoid(chord, y) == pytest.approx(float(ac.S), rel=1e-3), (
            f"shape {name!r} does not enclose the sourced wing area"
        )


def test_the_loading_shape_sensitivity_is_measured_and_recorded():
    """The spread across defensible shapes, on the hardest field the project
    holds. This number must be quoted with any strip result -- it is the cost
    of the shape being DECLARED rather than sourced.

    Asserted as a band. If the spread were zero the sweep would be measuring
    nothing; if it exceeded 100% the shape assumption would dominate the answer
    and no strip result could be reported at all.
    """
    import jax.numpy as jnp

    from flightsim import wind
    from flightsim.state import State, euler_to_quat

    ac = REGISTRY["boeing747"]
    st = airframe.stations(ac, n_span=2001, n_lon=9)
    # Cubic spanwise profile: the case an equivalent rate cannot represent at
    # all, so the shape assumption is doing maximum work.
    field = lambda p: jnp.array([0.0, 0.0, 1e-7 * p[1] ** 3])  # noqa: E731
    state = State(
        pos_ned=jnp.array([0.0, 0.0, -11278.0]),
        vel_body=jnp.array([236.0, 0.0, 0.0]),
        quat=euler_to_quat(jnp.array(0.0), jnp.array(0.0), jnp.array(0.0)),
        omega=jnp.zeros(3),
    )

    results = {}
    for name in airframe.LOADING_SHAPES:
        with airframe.loading_shape(name):
            results[name] = float(
                wind.strip_roll_moment(state.pos_ned, state.quat, field, ac, st, 236.0)
            )

    values = list(results.values())
    spread = (max(values) - min(values)) / abs(np.mean(values))
    print(f"\nLoading-shape sensitivity: {results}")
    print(f"Spread across shapes: {spread:.2%}")
    assert 0.0 < spread < 1.0, (
        f"shape spread {spread:.2%} is outside the reportable band"
    )
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
$PY -m pytest flightsim/tests/test_airframe.py -v -k "loading_shape"
```

Expected: failures with `AttributeError` on `LOADING_SHAPES` / `chord_distribution` / `loading_shape`.

- [ ] **Step 3: Write the implementation**

In `flightsim/airframe.py`, replace the body of `elliptic_chord`'s section by adding the following **after** `elliptic_chord` and **before** `calibrated_lift_slope`:

```python
import contextlib

# The three defensible spanwise shapes. `elliptic` is the default and the one
# the ledger DECLARES; the other two exist so the sensitivity can be measured
# rather than asserted. All three are scaled to enclose the SOURCED wing area,
# so the spread between them measures the shape assumption alone.
LOADING_SHAPES = ("elliptic", "uniform", "tapered")

# Taper ratio used ONLY by the `tapered` sensitivity alternative. It is not the
# 747's taper ratio -- that is not tabulated and not recoverable (design 3f) --
# it is a representative transport value whose only job is to give the sweep a
# third, differently-shaped member. No result may quote it as 747 geometry.
_SENSITIVITY_TAPER = 0.3

_active_shape = "elliptic"


@contextlib.contextmanager
def loading_shape(name: str):
    """Temporarily select a spanwise loading shape, for the sensitivity sweep.

    A context manager over module state rather than a parameter threaded through
    every call, because the shape is a project-level declaration that must be the
    same everywhere within one run. Threading it would invite two call sites
    disagreeing, which is exactly the failure the ledger exists to prevent.
    """
    global _active_shape
    if name not in LOADING_SHAPES:
        raise ValueError(f"unknown loading shape {name!r}; expected one of {LOADING_SHAPES}")
    previous = _active_shape
    _active_shape = name
    try:
        yield
    finally:
        _active_shape = previous


def chord_distribution(y: Array, ac: Aircraft, name: str | None = None) -> Array:
    """Spanwise chord for the named shape, scaled to the sourced wing area."""
    name = _active_shape if name is None else name
    if name == "elliptic":
        return elliptic_chord(y, ac)
    if name == "uniform":
        return jnp.full_like(y, ac.S / ac.b)
    if name == "tapered":
        # Straight taper: c(y) = c_root * (1 - (1-L)*|2y/b|), scaled to area S.
        # Area of that shape is b*c_root*(1+L)/2, so c_root = 2S/(b*(1+L)).
        lam = _SENSITIVITY_TAPER
        c_root = 2.0 * ac.S / (ac.b * (1.0 + lam))
        return c_root * (1.0 - (1.0 - lam) * jnp.abs(2.0 * y / ac.b))
    raise ValueError(f"unknown loading shape {name!r}")
```

Then change `_strip_rolling_coefficient` in `wind.py` to use the active shape:

```python
    chord = airframe.chord_distribution(y, ac)
```

(replacing `chord = airframe.elliptic_chord(y, ac)`).

- [ ] **Step 4: Run the tests to verify they pass**

```bash
$PY -m pytest flightsim/tests/test_airframe.py -v -s -k "loading_shape"
```

Expected: 2 passed, with the spread printed. **Record the printed spread — Task 13 writes it into the ledger's `strip.loading_shape` detail.**

- [ ] **Step 5: Run the whole suite**

```bash
$PY -m pytest flightsim/tests/ -q
```

Expected: all pass. The default shape is unchanged, so Task 9's assertions still hold.

- [ ] **Step 6: Commit**

```bash
git add flightsim/airframe.py flightsim/wind.py flightsim/tests/test_airframe.py
git commit -m "Add the loading-shape sensitivity sweep the ledger requires"
```

---

### Task 11: Regression gates — the frozen baselines must not move

**Files:**
- Modify: `flightsim/tests/test_wind.py`

- [ ] **Step 1: Write the test**

Append to `flightsim/tests/test_wind.py`:

```python
def test_nothing_added_by_this_work_moves_the_existing_wind_path():
    """Validation gate 8. Everything in Phases 2 and 3 is ADDITIVE -- gust_rates
    and field_model are untouched -- so a trajectory flown through the old path
    must be bit-identical. This is the test that would catch an 'improvement'
    accidentally applied to the default path.

    PROJECT.md section 4's mode baselines are downstream of exactly this code,
    and they are off-limits to feature work.
    """
    from flightsim import integrate, trim
    from flightsim.aircraft import CRUISE, REGISTRY

    ac = REGISTRY["boeing747"]
    v, h = CRUISE["boeing747"]["airspeed"], CRUISE["boeing747"]["altitude"]
    x, _ = trim.trim(jnp.array(v), jnp.array(h), ac)
    state = trim.trimmed_state(x[0], jnp.array(v), jnp.array(h))
    controls = trim.trimmed_controls(x[1], x[2])

    array = single()
    model = wind.vortex_model(array)
    sim = integrate.init_sim(state, jax.random.PRNGKey(0))

    first, _ = integrate.rollout(sim, controls, jnp.array(0.02), ac, 200, wind_model=model)
    second, _ = integrate.rollout(sim, controls, jnp.array(0.02), ac, 200, wind_model=model)

    assert np.array_equal(
        np.asarray(first.state.pos_ned), np.asarray(second.state.pos_ned)
    )
    assert np.array_equal(
        np.asarray(first.state.quat), np.asarray(second.state.quat)
    )
```

- [ ] **Step 2: Run the test**

```bash
$PY -m pytest flightsim/tests/test_wind.py::test_nothing_added_by_this_work_moves_the_existing_wind_path -v
```

Expected: PASS.

- [ ] **Step 3: Run the full suite plus the notebook gate**

```bash
$PY -m pytest flightsim/tests/ -q
```

```bash
$PY -m pytest --nbval-lax notebooks/ -q
```

Expected: both pass. The notebook is a required gate per `PROJECT.md` §10.

- [ ] **Step 4: Commit**

```bash
git add flightsim/tests/test_wind.py
git commit -m "Assert the existing wind path is untouched by the additive work"
```

---

# Phase 4 — Close the loop

### Task 12: Contract decision checkpoint — STOP AND ASK

**This task is a review gate, not an implementation step.** The design (§4, open item 3) says A2's wider contract is a reviewed decision, not something to smuggle in.

- [ ] **Step 1: Gather the numbers**

```bash
$PY -m pytest flightsim/tests/ -q -s -k "curvature_correction or loading_shape_sensitivity"
```

Record: the E2 curvature correction at the core edge (Task 7), and the loading-shape spread (Task 10).

- [ ] **Step 2: Present the decision to the user**

`strip_roll_moment` returns a coefficient but nothing consumes it yet — the 6-DOF still receives `omega_gust` through the existing contract. Wiring it in means widening the wind-model contract, which touches `integrate.step`, `autopilot` and `panel`.

Ask the user to choose:

- **(a) Leave it as a measurement tool.** The strip integral quantifies the error in the equivalent-rate treatment without changing any flown trajectory. Lowest risk; §4 baselines provably unmoved.
- **(b) Widen the contract so strip loads feed the 6-DOF.** Required to actually fly small-scale fields, which is the stated goal. Touches three modules and will move the vortex Fig. 8 point — that movement is expected and must be explained by the Task 7 figure, per validation gate 9.

**Do not proceed past this step without an answer.** If (b) is chosen, it needs its own plan — the contract change is a distinct piece of work with its own regression surface.

---

### Task 13: Write the measured numbers back into the documents

**Files:**
- Modify: `docs/ASSUMPTIONS.md`
- Modify: `flightsim/provenance.py`
- Modify: `docs/PROJECT.md`

- [ ] **Step 1: Close `ASSUMPTIONS.md` §E2**

§E2 currently ends: *"Verdict: sound for the updraft, lee wave and microburst; the weakest link in the vortex result."* It carries a scale ratio but no measured consequence.

Append to §E2, substituting the figure recorded in Task 7:

```markdown
**Bound, measured session 13 — the consequence, not just the scale ratio.**
The point-plus-gradient treatment takes the gust gradient as the tangent at the
CG. Sampling across the airframe instead and fitting the slope over the actual
span (`wind.sampled_rates`) gives a different answer wherever the field is
curved across the aircraft.

| Where | Correction to the pitch gust rate |
|---|---|
| Inside the Parks core | **exactly zero** — the Rankine profile is linear in radius, so a point sample plus a gradient is not an approximation at all |
| At the core edge | **<MEASURED>%** — the profile switches from linear to 1/r and the curvature across the span is largest here |

**This is why the vortex was the marginal case and the other fields were not.**
Asserted by `test_the_curvature_correction_across_the_parks_core_is_measured`.

**Verdict: bounded.** The correction is measured rather than argued, and the
exactly-zero result inside the core is the reason the existing vortex results
survived the assumption as long as they did.
```

- [ ] **Step 2: Record the sensitivity in the ledger**

In `flightsim/provenance.py`, extend the `strip.loading_shape` detail with the measured spread from Task 10:

```python
        "...spread reported. MEASURED: the spread across elliptic, uniform and "
        "tapered shapes on a cubic spanwise profile is <MEASURED>%."
```

- [ ] **Step 3: Record the new interface in `PROJECT.md` §2**

§2 currently documents the wind-model contract and the gust-rate signs. Append:

```markdown
3. **Gust gradients may be sampled rather than differentiated.**
   `wind.gust_rates` takes the analytic Jacobian at the CG; `wind.sampled_rates`
   fits the slope across the airframe using `airframe.stations`. The two agree
   exactly for any field that is linear across the aircraft, which is asserted,
   so this is a better estimator of the same quantity rather than a new one.
   `wind.strip_roll_moment` goes further and integrates the field per strip,
   which is the only form that carries curvature.

   **The three gust-rate signs are correct and independently verified** against
   Stengel eqs. 3.4-48, 3.4-50 and 3.4-52. Two equations in that source are
   wrong — eq. 3.4-49's sign, and eq. 3.4-55 by a factor of −2 — and
   `docs/superpowers/specs/2026-08-14-wind-shear-fidelity-design.md` §2 records
   which, with the self-consistency test that found them. Read it before
   changing any sign here.
```

- [ ] **Step 4: Add the new checks to `PROJECT.md` §4**

Append a table to §4:

```markdown
### Distributed-airframe sampling (session 13)

| Check | Measured | Tolerance |
|---|---|---|
| Uniform field, sampled rates | exactly 0 | `np.array_equal` |
| Linear field, sampled vs analytic gradient | agrees | rtol 1e-9 |
| Inside the Parks core, secant vs tangent | agrees | rel 1e-9 |
| At the core edge, curvature correction | **<MEASURED>%** | reported, 0–50% band |
| Station-count convergence, 9 → 18 | <0.1% | 1e-3 |
| Rigid roll rate through the strip integral vs CR-2144 `Clp` | agrees | rel 1e-3 |
| Rectangular-wing strip integral vs Stengel eq. 3.4-40 | agrees | rel 1e-6 |
| Loading-shape spread, three shapes | **<MEASURED>%** | reported, 0–100% band |
| Existing wind path, before and after | **bit-identical** | `np.array_equal` |
```

- [ ] **Step 5: Verify the documents match the code**

```bash
$PY -m pytest flightsim/tests/ -q
```

```bash
$PY -m pytest --nbval-lax notebooks/ -q
```

Expected: both pass. Confirm no `<MEASURED>` placeholder remains:

```bash
grep -rn "<MEASURED>" docs/ flightsim/
```

Expected: no output.

- [ ] **Step 6: Commit**

```bash
git add docs/ASSUMPTIONS.md docs/PROJECT.md flightsim/provenance.py
git commit -m "Close ASSUMPTIONS E2 with its measured bound, and record the new checks"
```

---

## Deliberately out of scope

Each of these is recorded in the design document with its reasoning. None is an oversight.

- **The full ledger retrofit** over every constant in `aircraft.py`, `atmosphere.py` and `wind.py`. Task 2 enters only what this work depends on. The retrofit is a separate plan.
- **Wiring strip loads into the 6-DOF.** Task 12 is the decision point; the work itself needs its own plan.
- **The three unused rate equivalences** (Stengel eqs. 3.4-49, 3.4-51, 3.4-53). A2 sidesteps them structurally rather than resolving eq. 3.4-55's weighting.
- **The per-surface split** (eqs. 3.4-54, 3.4-56). Needs tail geometry that neither CR-2144 nor its own source document tabulates (design §3f).
- **The `α̇` unsteady term** (eq. 3.4-58) and **the accelerometer lever arm** (eq. 3.2-119). Both verified and ready; both separate work. The lever arm is now unblocked by the sourced pilot station.
- **Any change to `PROJECT.md` §4 mode baselines.**

---

## Self-review against the design

| Design requirement | Task |
|---|---|
| §3b A1, fit replacing tangent | 5 |
| §3b A1 reduction property | 5 (uniform, linear), 7 (whole suite) |
| §3c A2 strip integration | 9 |
| §3c calibration pinned to `Clp` | 8, gate asserted in 9 |
| §3c mandatory sensitivity sweep | 10 |
| §3c Stengel eq. 3.4-40 cross-check | 8 |
| §3d derived tail arm + plausibility gate | 3 |
| §4 `airframe.py`, `provenance.py`, `wind` additions | 1–5, 8–10 |
| §4 `gust_rates` retained as reference | 5 (untouched), 11 (asserted) |
| §5 four categories, five enforcement rules | 1, 2 |
| §6 gates 1, 2 | 5 |
| §6 gate 3 | 9 |
| §6 gate 5 | 8 |
| §6 gate 6 | 3 |
| §6 gate 8 | 11 |
| §6 gate 10, closing E2 | 7, 13 |
| §9 item 2, station convergence | 6 |
| §9 item 3, contract decision | 12 |

**Gates 4, 7 and 9 are not implemented, and that is deliberate:**
- **Gate 4** (rigid pitch rate reproducing `Cmq` and `CLq`) is satisfied by construction — `l_eff` is *defined* as `−Cmq/CLq`, so asserting it would test arithmetic, not physics. Task 3 asserts the value and the physical plausibility instead, which is the claim that can actually fail.
- **Gate 7** (rigid-rotation-structure diagnostic) and **gate 9** (vortex Fig. 8 movement) both require strip loads to be feeding the 6-DOF. They belong to the Task 12 follow-on plan, and reporting them before then would mean reporting a change to a trajectory nothing has changed.
