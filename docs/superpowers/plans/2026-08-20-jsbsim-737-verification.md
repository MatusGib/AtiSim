# JSBSim 737 Cross-Code Verification — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify flightsim's aero build-up, trim solver, linearisation and integrator against JSBSim 1.3.1 by driving both engines with the same coefficients at the same conditions, and report the result as a PDF.

**Architecture:** A generator script drives JSBSim and freezes its outputs into a reference XML. Tests read only that XML, so the suite never imports `jsbsim`. A `_boeing_737()` registry entry is built from derivatives recovered by finite-differencing the running engine — not read from `737.xml`, because JSBSim applies forces at the AERORP and takes moments about the CG, which changes `Cma` from −0.6 to −1.064.

**Tech Stack:** JAX, numpy, scipy, matplotlib (PdfPages), pytest, JSBSim 1.3.1 Python extension.

**Spec:** `docs/superpowers/specs/2026-08-20-jsbsim-737-verification-design.md`

---

## File Structure

| File | Responsibility |
|---|---|
| `flightsim/aircraft.py` (modify) | `mach_ram` field; `_boeing_737()`; `REGISTRY`/`CRUISE` entries |
| `flightsim/aero.py` (modify, ~line 198) | Apply the Mach ram factor to thrust |
| `flightsim/dynamics.py` (modify, ~line 234) | Same factor in the available-thrust path |
| `flightsim/jsbsim_ref.py` (create) | Parse the reference XML into typed records. **No `jsbsim` import.** Shared by tests and report |
| `scripts/gen_jsbsim_reference.py` (create) | Drive JSBSim, write the XML. **The only file importing `jsbsim`** |
| `flightsim/tests/data/jsbsim_737_reference.xml` (create) | Frozen reference data |
| `flightsim/tests/test_jsbsim_737.py` (create) | The four comparison layers |
| `scripts/jsbsim_report.py` (create) | Build the results PDF |
| `pyproject.toml` (modify) | `ref = ["jsbsim"]` extra |

`jsbsim_ref.py` is split out from the test file so the report and the tests read the reference
through one parser rather than two, and so a schema change breaks in one place.

---

### Task 1: Mach ram factor in the thrust model

**Files:**
- Modify: `flightsim/aircraft.py` (the `Aircraft` NamedTuple; the propulsion comment at ~line 70)
- Modify: `flightsim/aero.py:198`
- Modify: `flightsim/dynamics.py:234`
- Modify: `flightsim/tests/test_conservation.py:26`
- Test: `flightsim/tests/test_aero.py`

Measured basis (spec, "Thrust model"): the CFM56's `MilThrust` table at M=0 tracks `(ρ/ρ₀)^1.0`
to 0.6% up to 30,000 ft, so the altitude law is already right. The gap is Mach: at 30,000 ft the
ratio to M=0 runs 1.000 / 0.954 / 0.960 / 1.016 / 1.121 / 1.277 at M = 0 / 0.2 / 0.4 / 0.6 / 0.8 / 1.0.

- [ ] **Step 1: Write the failing tests**

Append to `flightsim/tests/test_aero.py`:

```python
def test_mach_ram_defaults_to_neutral(test_aircraft):
    """A fixture built without mach_ram must behave exactly as before."""
    assert float(test_aircraft.mach_ram) == 0.0


def test_mach_ram_raises_thrust_with_mach():
    """thrust = throttle * Fmax * (rho/rho0)^n * (1 + mach_ram * M^2)."""
    from flightsim.tests.conftest import make_test_aircraft
    ac = make_test_aircraft()._replace(
        mach_ram=jnp.array(0.2), thrust_lapse=jnp.array(0.0)
    )
    controls = Controls(
        elevator=jnp.array(0.0), aileron=jnp.array(0.0),
        rudder=jnp.array(0.0), throttle=jnp.array(1.0),
    )
    a_sound = jnp.array(300.0)
    still = aero.thrust_body(jnp.array([0.0, 0.0, 0.0]), controls, ac, a_sound)
    moving = aero.thrust_body(jnp.array([150.0, 0.0, 0.0]), controls, ac, a_sound)
    # M = 0.5 -> factor 1 + 0.2 * 0.25 = 1.05
    assert float(moving[0]) == pytest.approx(1.05 * float(still[0]), rel=1e-12)
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest flightsim/tests/test_aero.py -k mach_ram -v`
Expected: FAIL — `AttributeError: 'Aircraft' object has no attribute 'mach_ram'`.

Note: the second test names `aero.thrust_body`. Read `flightsim/aero.py` around line 198 first and
use whatever that function is actually called; adjust the test to the real name before running.

- [ ] **Step 3: Add the field**

In `flightsim/aircraft.py`, update the propulsion comment and add the field **last** in the
`Aircraft` NamedTuple (a defaulted field must follow all non-defaulted ones):

```python
    # Propulsion: thrust = throttle * max_thrust * (rho/rho0)^thrust_lapse
    #                      * (1 + mach_ram * M^2)
    # The ram factor is the compressor's recovery of forward speed. It defaults
    # to zero so every aircraft defined before it is bit-for-bit unchanged; only
    # the 737, whose reference engine has a measured +12% at M 0.8, sets it.
    max_thrust: Array  # N, sea-level static
    thrust_lapse: Array  # density-ratio exponent
    ...
    rudder_limit: Array

    mach_ram: Array = jnp.array(0.0)
```

- [ ] **Step 4: Apply it in both thrust paths**

`flightsim/aero.py:198` becomes:

```python
    mach = V / a_sound
    magnitude = (
        controls.throttle * ac.max_thrust * (rho / RHO0) ** ac.thrust_lapse
        * (1.0 + ac.mach_ram * mach**2)
    )
```

Read the surrounding lines first — `V`, `rho` and `a_sound` must already be in scope, and `mach`
may already be computed. Do not introduce a second Mach.

`flightsim/dynamics.py:234` becomes:

```python
    available = (
        ac.max_thrust * (density(altitude) / RHO0) ** ac.thrust_lapse
        * (1.0 + ac.mach_ram * mach**2)
    )
```

If Mach is not in scope there, derive it from the same airspeed that function already uses; if that
function has no airspeed at all, leave it alone and record why in the commit message.

- [ ] **Step 5: Add the field name to the conservation field list**

`flightsim/tests/test_conservation.py:26` — append `"mach_ram"` to the tuple of field names.

- [ ] **Step 6: Run the new tests, then the whole suite**

Run: `python -m pytest flightsim/tests/test_aero.py -k mach_ram -v`
Expected: PASS.

Run: `python -m pytest flightsim/tests -q`
Expected: PASS, same count as before plus 2. **Any pre-existing test that changes value is a
failure of the neutral default and must be fixed, not accepted.**

- [ ] **Step 7: Commit**

```bash
git add flightsim/aircraft.py flightsim/aero.py flightsim/dynamics.py flightsim/tests/test_aero.py flightsim/tests/test_conservation.py
git commit -m "Give thrust the Mach ram term it was missing"
```

---

### Task 2: Reference XML parser

**Files:**
- Create: `flightsim/jsbsim_ref.py`
- Test: `flightsim/tests/test_jsbsim_737.py` (first tests only)

Written before the generator so the schema is fixed by its reader.

- [ ] **Step 1: Define the schema and parser**

`flightsim/jsbsim_ref.py`:

```python
"""Reader for the frozen JSBSim reference data.

Deliberately does NOT import jsbsim. The reference is generated once by
scripts/gen_jsbsim_reference.py and checked in, so the test suite runs on a
machine with no JSBSim installed and reference drift shows up in git diff.

See docs/superpowers/specs/2026-08-20-jsbsim-737-verification-design.md.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import NamedTuple

import numpy as np

REFERENCE = Path(__file__).parent / "tests" / "data" / "jsbsim_737_reference.xml"


class Condition(NamedTuple):
    name: str
    altitude_m: float        # what JSBSim was run at
    matched_altitude_m: float  # what flightsim must be run at, for equal density
    density: float           # kg/m^3, JSBSim's
    sound_speed: float       # m/s, JSBSim's
    airspeed: float          # m/s true
    mass: float              # kg
    inertia: np.ndarray      # (3,3) kg m^2


class Trim(NamedTuple):
    mode: str
    alpha: float
    elevator: float
    throttle: float
    thrust: float            # N, total
    bank: float


class SweepPoint(NamedTuple):
    alpha: float
    beta: float
    rates: np.ndarray        # (3,) rad/s, body, relative to air
    controls: np.ndarray     # (3,) rad: elevator, aileron, rudder
    coefficients: np.ndarray  # (6,) CL, CD, CY, Cl, Cm, Cn


class Linearization(NamedTuple):
    A: np.ndarray            # (12,12)
    B: np.ndarray            # (12,4)
    x0: np.ndarray
    u0: np.ndarray


class TrajectorySample(NamedTuple):
    t: float
    vel_body: np.ndarray     # (3,) m/s
    omega: np.ndarray        # (3,) rad/s
    euler: np.ndarray        # (3,) rad
    altitude: float          # m
    controls: np.ndarray     # (3,) rad
    thrust: float            # N


class Reference(NamedTuple):
    jsbsim_version: str
    jsbsim_commit: str
    condition: dict[str, Condition]
    trim: dict[str, Trim]
    derivatives: dict[str, float]
    absent: dict[str, float]
    sweep: list[SweepPoint]
    linearization: Linearization
    trajectory: dict[str, list[TrajectorySample]]
    tolerances: dict[str, float]
    tolerance_derivations: dict[str, str]


def _floats(text: str) -> np.ndarray:
    return np.array([float(v) for v in text.split()])


def load(path: Path = REFERENCE) -> Reference:
    root = ET.parse(path).getroot()
    prov = root.find("provenance")

    conditions = {}
    for c in root.findall("condition"):
        conditions[c.get("name")] = Condition(
            name=c.get("name"),
            altitude_m=float(c.findtext("altitude_m")),
            matched_altitude_m=float(c.findtext("matched_altitude_m")),
            density=float(c.findtext("density")),
            sound_speed=float(c.findtext("sound_speed")),
            airspeed=float(c.findtext("airspeed")),
            mass=float(c.findtext("mass")),
            inertia=_floats(c.findtext("inertia")).reshape(3, 3),
        )

    trims = {}
    for t in root.findall("trim"):
        trims[t.get("mode")] = Trim(
            mode=t.get("mode"),
            alpha=float(t.findtext("alpha")),
            elevator=float(t.findtext("elevator")),
            throttle=float(t.findtext("throttle")),
            thrust=float(t.findtext("thrust")),
            bank=float(t.findtext("bank")),
        )

    derivatives = {d.get("name"): float(d.get("value"))
                   for d in root.findall("derivatives/derivative")}
    absent = {d.get("name"): float(d.get("measured"))
              for d in root.findall("absent_derivatives/derivative")}

    sweep = [
        SweepPoint(
            alpha=float(p.get("alpha")),
            beta=float(p.get("beta")),
            rates=_floats(p.get("rates")),
            controls=_floats(p.get("controls")),
            coefficients=_floats(p.get("coefficients")),
        )
        for p in root.findall("sweep/point")
    ]

    lin = root.find("linearization")
    linearization = Linearization(
        A=_floats(lin.findtext("A")).reshape(12, 12),
        B=_floats(lin.findtext("B")).reshape(12, 4),
        x0=_floats(lin.findtext("x0")),
        u0=_floats(lin.findtext("u0")),
    )

    trajectory = {}
    for tr in root.findall("trajectory"):
        trajectory[tr.get("name")] = [
            TrajectorySample(
                t=float(s.get("t")),
                vel_body=_floats(s.get("vel_body")),
                omega=_floats(s.get("omega")),
                euler=_floats(s.get("euler")),
                altitude=float(s.get("altitude")),
                controls=_floats(s.get("controls")),
                thrust=float(s.get("thrust")),
            )
            for s in tr.findall("sample")
        ]

    tolerances, derivations = {}, {}
    for t in root.findall("tolerances/tolerance"):
        tolerances[t.get("name")] = float(t.get("value"))
        derivations[t.get("name")] = t.get("derivation")

    return Reference(
        jsbsim_version=prov.findtext("jsbsim_version"),
        jsbsim_commit=prov.findtext("jsbsim_commit"),
        condition=conditions, trim=trims, derivatives=derivatives, absent=absent,
        sweep=sweep, linearization=linearization, trajectory=trajectory,
        tolerances=tolerances, tolerance_derivations=derivations,
    )
```

- [ ] **Step 2: Commit the parser**

```bash
git add flightsim/jsbsim_ref.py
git commit -m "Read the frozen reference through one parser, not two"
```

---

### Task 3: The generator

**Files:**
- Create: `scripts/gen_jsbsim_reference.py`
- Modify: `pyproject.toml`
- Creates: `flightsim/tests/data/jsbsim_737_reference.xml`

The only file importing `jsbsim`. Structure it as small functions so re-recovering at a different
condition is a parameter change (spec, Limitation 1 mitigation).

- [ ] **Step 1: Add the extra to `pyproject.toml`**

```toml
# The JSBSim cross-code reference generator. A SEPARATE extra for the same
# reason as `ui`: scripts/gen_jsbsim_reference.py is the only file that needs
# it, the frozen XML it writes is checked in, and the test suite reads that XML
# rather than importing jsbsim. No test requires this to be installed.
ref = ["jsbsim"]
```

- [ ] **Step 2: Write the generator**

Requirements, each of which is a spec clause:

1. `matched_altitude(rho_target)` — `scipy.optimize.brentq` on
   `flightsim.atmosphere.density(h) - rho_target`, bracket ±300 m. **Assert the residual is
   below 1e-10 relative** (success criterion 8).
2. `trim(mode)` — `simulation/do_simple_trim` for mode 0 (longitudinal) and mode 5 (turn).
3. `recover_derivatives()` — central differences about trim. Perturb α, β, p, q, r, δe, δa, δr;
   non-dimensionalise by JSBSim's own q̄, S, b, c̄. **Read back** the actual α/β/rates/surfaces
   after each run and difference against those, never against the commanded values — the yaw
   damper moves the rudder on its own.
4. **Assert the six absent derivatives measure zero**: `CLq`, `CYp`, `CYr`, `CYdr`, `Cnp`, `Cnda`
   (success criterion 4). Tolerance 1e-9.
5. `sweep()` — a grid over α ∈ [0°, 4°], β ∈ [−3°, 3°], each rate and each surface, ~60 points.
6. `linearization()` — trigger `simulation/do_linearization`, parse the `.sce`, and **re-derive
   the state ordering** from the structural rows: find the row that is `θ̇ = q` (a single 1.0),
   the row where `ψ̇ = r/cos θ`, and the row `ḣ = vt(θ − α)`. Raise if the derived ordering is not
   `[vt, α, θ, q, β, φ, p, ψ, r, lat, lon, h]` (success criterion 5).
7. `trajectory(name, surface_history)` — 20 s at dt = 1/120. Each step, pre-compensate
   `fcs/rudder-cmd-norm` by `fcs/yaw-damper-final` so the *surface* follows the commanded history,
   and **assert `fcs/rudder-pos-rad` is within 1e-6 of target**. Two cases: `elevator_doublet`
   and `rudder_kick`. Set `simulation/gravity-model` to constant-g; record the Coriolis
   contribution by running the same case at latitude 0° and 47° and storing the max state
   difference.
8. Write the XML with `%.12e` for every float so regeneration is byte-stable, and sort all
   dict-derived elements by name.

- [ ] **Step 3: Generate and inspect**

Run: `python scripts/gen_jsbsim_reference.py`
Expected: writes `flightsim/tests/data/jsbsim_737_reference.xml`, prints the recovered derivative
table, and prints `density match residual: <1e-10`.

Sanity-check against the values measured during brainstorming before trusting anything:
`Cma ≈ −1.0637`, `CLa ≈ +4.3478`, `CLde ≈ +0.2000`, `Cmde ≈ −0.8710`, trim `α ≈ 1.9516°`,
`δe ≈ −0.0516 rad`.

- [ ] **Step 4: Verify byte-stability**

```bash
cp flightsim/tests/data/jsbsim_737_reference.xml /tmp/ref1.xml
python scripts/gen_jsbsim_reference.py
diff /tmp/ref1.xml flightsim/tests/data/jsbsim_737_reference.xml && echo "byte-stable"
```
Expected: `byte-stable` (success criterion 2).

- [ ] **Step 5: Commit**

```bash
git add scripts/gen_jsbsim_reference.py flightsim/tests/data/jsbsim_737_reference.xml pyproject.toml
git commit -m "Recover the 737 from the running engine, not from its XML"
```

---

### Task 4: The 737 registry entry

**Files:**
- Modify: `flightsim/aircraft.py`
- Test: `flightsim/tests/test_jsbsim_737.py`

- [ ] **Step 1: Write the failing test**

```python
def test_737_matches_reference_derivatives():
    """Every derivative in the registry entry equals the recovered reference."""
    ref = jsbsim_ref.load()
    ac = REGISTRY["boeing737"]
    for name, expected in ref.derivatives.items():
        if not hasattr(ac, name):
            continue
        assert float(getattr(ac, name)) == pytest.approx(expected, rel=1e-9), name


def test_737_zeroes_the_derivatives_jsbsim_lacks():
    ac = REGISTRY["boeing737"]
    for name in ("CLq", "CYp", "CYr", "CYdr", "Cnp", "Cnda"):
        assert float(getattr(ac, name)) == 0.0, name
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest flightsim/tests/test_jsbsim_737.py -k 737_matches -v`
Expected: FAIL — `KeyError: 'boeing737'`.

- [ ] **Step 3: Write `_boeing_737()`**

Follow the existing per-aircraft style exactly: literal numbers, conversions at the point of
definition, a comment naming each number's origin. The docstring **must** carry Limitation 1
(success criterion 7). Skeleton — fill every number from the generator's printed table:

```python
def _boeing_737() -> Aircraft:
    """Boeing 737, linearised from JSBSim 1.3.1's own 737 at cruise.

    *** NOT A QUALIFIED SOURCE, AND NOT VALID AWAY FROM CRUISE. ***

    The JSBSim 737 declares itself built from public data "and guesses", for
    "educational and entertainment purposes only". Nothing here supports any
    claim about a real 737. It exists to verify this project's solver against an
    independent engine fed the same coefficients -- see
    docs/superpowers/specs/2026-08-20-jsbsim-737-verification-design.md.

    VALIDITY BAND: linearised about 30,000 ft, M 0.78, alpha 1.95 deg. Unlike
    every other entry in REGISTRY -- which are linear derivative sets valid
    across the ordinary linear range -- this one is a local fit to a NONLINEAR
    model and degrades away from that point:

      - JSBSim's CL(alpha) peaks at 1.20 near 13 deg and falls. CL0 + CLa*alpha
        keeps climbing, so above ~10 deg this entry has NO stall behaviour and
        reports lift the source model does not have.
      - CD0 is the alpha = 1.95 deg value of a table running 0.021 to 0.042 over
        0 to 15 deg, so drag is progressively under-predicted off-cruise.
      - mach_ram is fitted over M 0.6-1.0 and is wrong at low speed.

    Nothing in the code prevents use outside this band. To use another
    condition, re-run scripts/gen_jsbsim_reference.py at that condition.

    Derivatives are recovered by finite-differencing the RUNNING engine, not read
    from 737.xml, because JSBSim applies aero forces at the AERORP (x = 625 in)
    and takes moments about the CG (x = 610.8 in). That 0.096 cbar offset makes
    the effective Cma -1.0637 where the XML constant says -0.6.
    """
```

Then `REGISTRY["boeing737"] = _boeing_737()` and:

```python
    # JSBSim 737 cruise: the condition the derivatives were recovered at. The
    # altitude is the DENSITY-MATCHED one, 43 ft below JSBSim's 30,000 ft,
    # because flightsim's ISA uses geometric altitude where the standard uses
    # geopotential. Running at a nominal 30,000 ft would put a 0.16% bias on
    # every force. See the spec, "Matching the input conditions".
    "boeing737": {"altitude": <matched_altitude_m>, "airspeed": <V_si>},
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest flightsim/tests/test_jsbsim_737.py -v`
Expected: PASS.

- [ ] **Step 5: Confirm nothing else moved**

Run: `python -m pytest flightsim/tests -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add flightsim/aircraft.py flightsim/tests/test_jsbsim_737.py
git commit -m "Add the 737, and say plainly where it stops being valid"
```

---

### Task 5: Layer 1 — coefficients

**Files:**
- Modify: `flightsim/tests/test_jsbsim_737.py`

- [ ] **Step 1: Write the test**

```python
def test_layer1_coefficients_match_across_the_sweep():
    """flightsim's build-up against JSBSim's, at identical read-back states."""
    ref = jsbsim_ref.load()
    cond = ref.condition["cruise"]
    ac = REGISTRY["boeing737"]
    worst = {}
    for pt in ref.sweep:
        V = cond.airspeed
        vel = jnp.array([
            V * jnp.cos(pt.alpha) * jnp.cos(pt.beta),
            V * jnp.sin(pt.beta),
            V * jnp.sin(pt.alpha) * jnp.cos(pt.beta),
        ])
        controls = Controls(
            elevator=jnp.array(pt.controls[0]), aileron=jnp.array(pt.controls[1]),
            rudder=jnp.array(pt.controls[2]), throttle=jnp.array(0.0),
        )
        got = aero.coefficients(vel, jnp.array(pt.rates), controls, ac,
                                jnp.array(cond.sound_speed))
        for name, g, e in zip("CL CD CY Cl Cm Cn".split(), got, pt.coefficients):
            worst[name] = max(worst.get(name, 0.0), abs(float(g) - e))
    for name, err in worst.items():
        assert err <= ref.tolerances[f"layer1_{name}"], (
            f"{name}: {err:.3e} > {ref.tolerances[f'layer1_{name}']:.3e} "
            f"({ref.tolerance_derivations[f'layer1_{name}']})"
        )
```

Check `aero.coefficients`'s real signature and argument order before running; the call above is
from the module docstring, not verified.

- [ ] **Step 2: Run**

Run: `python -m pytest flightsim/tests/test_jsbsim_737.py -k layer1 -v`

If it fails, **do not widen the tolerance.** Localise: a constant offset in one coefficient is a
sign or reference-point error; growth with |α| is the linearisation residual and should already be
in the derived tolerance; growth with a rate is a non-dimensionalisation error (`b/2V` vs `c/2V`).

- [ ] **Step 3: Commit**

```bash
git add flightsim/tests/test_jsbsim_737.py
git commit -m "Compare the build-up coefficient by coefficient"
```

---

### Task 6: Layer 2 — trim

**Files:**
- Modify: `flightsim/tests/test_jsbsim_737.py`

- [ ] **Step 1: Write the test**

```python
@pytest.mark.parametrize("mode", ["longitudinal", "turn"])
def test_layer2_trim_matches(mode):
    ref = jsbsim_ref.load()
    cond, expected = ref.condition["cruise"], ref.trim[mode]
    ac = REGISTRY["boeing737"]
    got = trim.trim(ac, airspeed=cond.airspeed, altitude=cond.matched_altitude_m,
                    bank=expected.bank)
    assert float(got.alpha) == pytest.approx(expected.alpha,
                                             abs=ref.tolerances["layer2_alpha"])
    assert float(got.elevator) == pytest.approx(expected.elevator,
                                                abs=ref.tolerances["layer2_elevator"])
    assert float(got.thrust) == pytest.approx(expected.thrust,
                                              rel=ref.tolerances["layer2_thrust"])
```

Read `trim.trim`'s real signature and return type first and adapt; it may not take `bank` or
return a `.thrust`. If it cannot do a banked trim, keep the longitudinal case, mark the turn case
`xfail` with the reason, and record it as a finding rather than deleting it.

- [ ] **Step 2: Run, then commit**

```bash
python -m pytest flightsim/tests/test_jsbsim_737.py -k layer2 -v
git add flightsim/tests/test_jsbsim_737.py
git commit -m "Trim both engines at the same point and compare"
```

---

### Task 7: Layer 3 — modes

**Files:**
- Modify: `flightsim/tests/test_jsbsim_737.py`

- [ ] **Step 1: Write the test**

```python
def test_layer3_longitudinal_eigenvalues_match():
    """JSBSim's [vt, alpha, theta, q] block against ours, on eigenvalues."""
    ref = jsbsim_ref.load()
    cond, tr = ref.condition["cruise"], ref.trim["longitudinal"]
    ac = REGISTRY["boeing737"]
    A_js = ref.linearization.A[:4, :4]
    A_fs = validation.longitudinal_matrix(
        ac, alpha=tr.alpha, elevator=tr.elevator, throttle=tr.throttle,
        V=cond.airspeed, H=cond.matched_altitude_m,
    )
    got = np.sort_complex(np.linalg.eigvals(_uw_to_vt_alpha(A_fs, cond.airspeed, tr.alpha)))
    want = np.sort_complex(np.linalg.eigvals(A_js))
    assert np.allclose(got, want, rtol=ref.tolerances["layer3_longitudinal"])
```

`_uw_to_vt_alpha` is the similarity transform from `[u, w, q, θ]` to `[vt, α, θ, q]`: for small
perturbations `δvt = cos α₀ δu + sin α₀ δw` and `δα = (−sin α₀ δu + cos α₀ δw)/V₀`, plus the row/
column swap putting θ before q. Write it in the test file, and assert it preserves eigenvalues on
a random matrix first — a wrong transform silently produces plausible numbers.

- [ ] **Step 2: Add the same for the lateral block** `A[4:9, 4:9]` against the lateral matrix,
      comparing Dutch roll, roll and spiral.

- [ ] **Step 3: Run, then commit**

```bash
python -m pytest flightsim/tests/test_jsbsim_737.py -k layer3 -v
git add flightsim/tests/test_jsbsim_737.py
git commit -m "Compare the linearisations where the modes live"
```

---

### Task 8: Layer 4 — trajectory

**Files:**
- Modify: `flightsim/tests/test_jsbsim_737.py`

- [ ] **Step 1: Write the test**

```python
@pytest.mark.parametrize("case", ["elevator_doublet", "rudder_kick"])
def test_layer4_trajectory_tracks(case):
    """Integrate 20 s on the same prescribed surface history and bound the drift."""
    ref = jsbsim_ref.load()
    cond = ref.condition["cruise"]
    ac = REGISTRY["boeing737"]
    samples = ref.trajectory[case]
    state = _state_from(samples[0], cond)
    worst_v = worst_w = 0.0
    for prev, cur in zip(samples, samples[1:]):
        state = integrate.step(state, _controls_from(prev), ac, cur.t - prev.t)
        worst_v = max(worst_v, abs(float(state.vel_body[0]) - cur.vel_body[0]))
        worst_w = max(worst_w, abs(float(state.omega[1]) - cur.omega[1]))
    assert worst_v <= ref.tolerances["layer4_velocity"]
    assert worst_w <= ref.tolerances["layer4_rate"]
```

Sample spacing in the XML is coarser than JSBSim's dt; sub-step `integrate.step` to the project's
own dt between samples rather than taking one giant step, or the comparison measures step size
rather than physics.

- [ ] **Step 2: Run, then commit**

```bash
python -m pytest flightsim/tests/test_jsbsim_737.py -k layer4 -v
git add flightsim/tests/test_jsbsim_737.py
git commit -m "Fly both engines through the same doublet"
```

---

### Task 9: The results PDF

**Files:**
- Create: `scripts/jsbsim_report.py`
- Creates: `docs/summary/jsbsim-737-report.pdf`

Follow `scripts/turbulence_report.py`: `matplotlib.use("Agg")`, `PdfPages`, A4 `(8.27, 11.69)`,
the same ink/muted/blue/teal/amber palette, output path from `sys.argv[1]`.

**The rule from `scripts/summary.py` applies: every number is read from the reference XML or
computed at build time by calling the project's own code. Nothing typed from memory.**

- [ ] **Step 1: Build the pages**

1. **Title + claim.** What is and is not claimed; JSBSim version and commit from the XML.
2. **Input conditions.** The atmosphere error table before and after density matching; the
   sign-convention table. Both computed live, not transcribed.
3. **Derivative recovery.** Recovered values against `737.xml` constants, with the AERORP offset
   column. Bar chart of the `Cma` −0.6 vs −1.064 difference.
4. **Layer 1.** Per-coefficient error across the sweep; error against α with the tolerance band.
5. **Layer 2.** Trim comparison, both modes.
6. **Layer 3.** Both engines' eigenvalues on the complex plane, annotated by mode.
7. **Layer 4.** State overlays with the divergence envelope; the Coriolis contribution as a number.
8. **Thrust.** The ram fit against the CFM56 table, and the `(ρ/ρ₀)^1.0` altitude agreement.
9. **Limitations.** All seven, verbatim in substance from the spec, Limitation 1 first and boxed.

- [ ] **Step 2: Build and check**

Run: `python scripts/jsbsim_report.py docs/summary/jsbsim-737-report.pdf`
Expected: a PDF with every page populated and no empty axes.

- [ ] **Step 3: Commit**

```bash
git add scripts/jsbsim_report.py docs/summary/jsbsim-737-report.pdf
git commit -m "Report what the comparison actually found"
```

---

### Task 10: Close out

- [ ] **Step 1: Full suite with JSBSim importable**

Run: `python -m pytest flightsim/tests -q`

- [ ] **Step 2: Prove the suite does not need JSBSim** (success criterion 1)

```bash
python -c "import sys; sys.modules['jsbsim']=None; import pytest; sys.exit(pytest.main(['flightsim/tests','-q']))"
```
Expected: PASS. If anything imports `jsbsim`, this fails — that is the point.

- [ ] **Step 3: Record findings in `docs/PROJECT.md`**

Add the measured agreement per layer to the evidence ledger, and record as findings: the
geopotential/geometric atmosphere defect (with its spawned task), and any disagreement documented
rather than fixed (spec, Limitation 5).

- [ ] **Step 4: Final commit**

```bash
git add docs/PROJECT.md
git commit -m "Put the JSBSim comparison in the evidence ledger"
```

---

## Self-review

**Spec coverage:** claim/non-claim → Task 4 docstring + Task 9 page 1. Derivative recovery from the
engine → Task 3. Six absent derivatives → Tasks 3, 4. `e` = 0.9666 → Task 4. Thrust ram → Task 1.
Four layers → Tasks 5–8. Read-back protocol → Task 3. Density matching → Tasks 3, 4. State-order
re-derivation → Task 3. Tolerance derivations in the XML → Tasks 2, 3, 5. Limitation 1 in three
places → Task 4 (docstring, `CRUISE`) and Task 3 (XML). PDF → Task 9. All nine success criteria
have a step.

**Placeholders:** the numeric literals in Task 4 are marked `<...>` deliberately — they come from
Task 3's output and cannot be known before it runs. Task 3's step 3 lists the brainstormed values
to check them against, so they are verifiable rather than invented.

**Type consistency:** `jsbsim_ref.load()` returns `Reference`; every later task reads
`ref.condition["cruise"]`, `ref.trim[mode]`, `ref.derivatives`, `ref.sweep`, `ref.linearization`,
`ref.trajectory[case]`, `ref.tolerances`, `ref.tolerance_derivations` — all defined in Task 2.
Tolerance keys are `layer1_<coef>`, `layer2_<quantity>`, `layer3_<axis>`, `layer4_<quantity>`,
used consistently in Tasks 3 and 5–8.

**Known soft spots**, flagged rather than papered over: Tasks 5–8 call `aero.coefficients`,
`trim.trim`, `validation.longitudinal_matrix` and `integrate.step` with signatures taken from
module docstrings rather than verified call sites. Each of those steps says to read the real
signature first. If `trim.trim` cannot do a banked trim, Task 6 says to `xfail` the turn case and
record it, not delete it.
