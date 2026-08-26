# WGS-84 and Earth Rotation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace AtiSim's flat, non-rotating Earth with the WGS-84 ellipsoid, J2 gravity and Earth rotation, in the formulation JSBSim uses, verified term-by-term against the JSBSim binary.

**Architecture:** A new `atisim/earth.py` holds WGS-84 geodesy, gravity and an `EarthModel` config. `State` keeps its 13-element shape but changes meaning: position becomes an ECEF offset from a run `Anchor`, the quaternion becomes body→ECEF, and `omega` becomes the body rate relative to ECEF. Local NED becomes a derived *view* — which is why the wind fields, which take NED positions, need no changes at all.

**Tech Stack:** Python 3.10, JAX (float64), NumPy, pytest. JSBSim 1.3.1 is used only by `scripts/` to regenerate frozen XML reference data; nothing under `atisim/` may import it.

**Design:** `docs/superpowers/specs/2026-08-26-wgs84-earth-rotation-design.md`

---

## Before you start: the two facts that will waste your day

**1. Check which tree you imported.** Worktrees share the main checkout's `.venv`, and the editable install can silently resolve `atisim` to the *wrong tree* — a passing suite against code you did not change. Run this from the worktree root before believing any result:

```bash
../../../.venv/Scripts/python.exe -c "import atisim; print(atisim.__file__)"
```

It must print a path inside this worktree. If it does not, stop. See PROJECT.md §10.

**2. Two interpreters, deliberately.** The venv has pytest but **no jsbsim** — that is intended, because the suite must run anywhere. Only Task 5 needs JSBSim, and it uses the system interpreter.

| Purpose | Command prefix (run from the worktree root) |
|---|---|
| Tests, everything under `atisim/` | `../../../.venv/Scripts/python.exe -m pytest` |
| Reference generation only (Task 5) | `C:/Users/mateusz/AppData/Local/Programs/Python/Python310/python.exe` |

Always `-m pytest`, never the bare `pytest` console script — the console script resolves `atisim` to the main checkout.

**Baseline measured before any change: `724 passed, 1 skipped in 705s (11m45s)`.** The full suite is slow, so per-task steps below target specific files. Run the full suite only at the checkpoints that ask for it.

---

## File Structure

| File | Responsibility |
|---|---|
| **Create** `atisim/earth.py` | WGS-84 constants, geodesy, gravity, `EarthModel`, `Anchor`. Pure functions; imports nothing from `atisim` except `units` and `atmosphere.G0`. |
| **Create** `atisim/earth_ref.py` | Reader for the frozen Earth reference XML. Must not import `jsbsim`. |
| **Create** `scripts/gen_jsbsim_earth_reference.py` | The only new file that imports `jsbsim`. Writes frozen XML. |
| **Create** `atisim/tests/data/jsbsim_earth_reference.xml` | Frozen reference, checked in. |
| **Create** `atisim/tests/test_earth.py` | Geodesy/gravity unit tests + term-by-term vs the frozen reference + the FLAT floor. |
| **Modify** `atisim/state.py` | New `State` meaning, `Anchor`-aware accessors, `dcm_to_quat`, `quat_to_dcm` renamed to force a loud break. |
| **Modify** `atisim/dynamics.py` | The four equations of the design §2. |
| **Modify** `atisim/integrate.py` | Thread `anchor` and `earth` through `step`/`rollout`. |
| **Modify** `atisim/trim.py` | Six unknowns, `beta = 0`, six residuals. |
| **Modify** `atisim/verification.py` | Closed-form rotating-Earth invariants (tier 0). |
| **Modify** `atisim/sensors.py`, `atisim/checks.py`, `atisim/viz.py`, `atisim/panel.py`, `atisim/wind.py`, `atisim/validation.py`, `atisim/vortex_viz.py` | Migrate to the NED accessors. |
| **Modify** `atisim/provenance.py` | Ledger entries for the five WGS-84 constants. |
| **Modify** `docs/ASSUMPTIONS.md`, `docs/PROJECT.md`, `audit/INVENTORY.md`, `audit/NOTATION.md` | Retire A1/A2, rewrite A3, re-measure F4, re-measure §4. |

### The blast radius, measured rather than estimated

The spec says "43 files read `pos_ned`". That is true and misleading. Measured:

- **3 sites** outside wind fields derive altitude from position: `dynamics.py:63`, `sensors.py:110`, `state.py:35`. These change.
- **Every other `pos_ned` use passes it INTO a wind field**, and wind fields keep taking NED positions. They do not change.
- **40+ sites call `quat_to_dcm`**, which currently means body→NED and would silently become body→ECEF. This is the real hazard, and Task 7 forces it to fail loudly.

---

## Task 0: Freeze a pre-Earth trajectory — DO THIS FIRST

**Files:**
- Create: `scripts/freeze_pre_earth_trajectory.py`
- Create: `atisim/tests/data/pre_earth_trajectory.npz` (generated)

Task 13 compares `FLAT` against the model as it exists **today**. Once Task 8 changes `State`, today's model is gone and cannot be re-run. This must therefore happen before any other task.

- [ ] **Step 1: Write the freezer**

Create `scripts/freeze_pre_earth_trajectory.py`:

```python
"""Freeze one trajectory from the FLAT-EARTH model, before the Earth is added.

Run ONCE, before any of the WGS-84 work lands. After Task 8 changes `State`
this script will no longer run, and that is fine -- its output is checked in.
Task 13 compares the FLAT configuration of the new plant against this file.

The comparison it enables is NOT a bit-identity test. See the design doc
section 8: an ECEF state cannot reproduce a NED state bit-for-bit. What the
file supports is the weaker and honest claim -- that the difference is
round-off and does not grow with the run.
"""

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from atisim.aircraft import CRUISE, REGISTRY          # noqa: E402
from atisim.integrate import init_sim, logged_rollout  # noqa: E402
from atisim.trim import trim, trimmed_controls, trimmed_state  # noqa: E402

import jax  # noqa: E402

OUT = ROOT / "atisim" / "tests" / "data" / "pre_earth_trajectory.npz"

DT = 0.02
N_STEPS = 1000   # 20 s, the same window the JSBSim layer comparison uses


def main():
    # NOTE the key and the accessor: the registry keys are `boeing747`, not
    # `b747`, and CRUISE values are plain dicts, not attribute objects.
    ac = REGISTRY["boeing747"]
    airspeed = CRUISE["boeing747"]["airspeed"]     # 235.9152 m/s
    altitude = CRUISE["boeing747"]["altitude"]     # 12192.0 m

    x, residual = trim(airspeed, altitude, ac)
    assert np.max(np.abs(np.asarray(residual))) < 1e-9, "pre-Earth trim did not converge"
    alpha, elevator, throttle = (float(v) for v in x)

    sim = init_sim(
        trimmed_state(jnp.asarray(alpha), jnp.asarray(airspeed), jnp.asarray(altitude)),
        jax.random.PRNGKey(0),
    )
    controls = trimmed_controls(jnp.asarray(elevator), jnp.asarray(throttle))
    _, log = logged_rollout(sim, controls, DT, ac, N_STEPS)

    np.savez_compressed(
        OUT,
        dt=DT, n_steps=N_STEPS,
        airspeed=float(airspeed), altitude=float(altitude),
        trim=np.asarray(x, dtype=float),
        pos_ned=np.asarray(log.state.pos_ned, dtype=float),
        vel_body=np.asarray(log.state.vel_body, dtype=float),
        quat=np.asarray(log.state.quat, dtype=float),
        omega=np.asarray(log.state.omega, dtype=float),
    )
    print(f"wrote {OUT.relative_to(ROOT)}  ({N_STEPS} samples)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `../../../.venv/Scripts/python.exe scripts/freeze_pre_earth_trajectory.py`
Expected: `wrote atisim/tests/data/pre_earth_trajectory.npz  (1000 samples)`

- [ ] **Step 3: Commit**

```bash
git add scripts/freeze_pre_earth_trajectory.py atisim/tests/data/pre_earth_trajectory.npz
git commit -m "Freeze a flat-Earth trajectory before the Earth model replaces it"
```

---

## Task 1: WGS-84 constants

**Files:**
- Create: `atisim/earth.py`
- Test: `atisim/tests/test_earth.py`

- [ ] **Step 1: Write the failing test**

```python
"""WGS-84 geodesy, gravity, and the rotating-Earth frames.

Every constant here was recovered from the running JSBSim binary, not quoted
from a table -- see the design doc section 2. The tests below assert the
DEFINING values; the term-by-term agreement with JSBSim is asserted separately
against the frozen reference.
"""

import numpy as np
import pytest

import atisim  # noqa: F401  -- enables x64 before any array is made
from atisim import earth


def test_the_defining_wgs84_constants_are_exact():
    """a and f are DEFINING; b and e2 are derived from them and never quoted.

    JSBSim reports a semi-minor axis of 6356752.314186 m, which differs from
    a(1-f) by 5.87e-5 m. That is JSBSim's internal storage in feet
    round-tripping, not a different ellipsoid, so this model carries the
    defining pair and derives the rest. Asserted here so nobody "fixes" b to
    JSBSim's reported value later.
    """
    assert earth.A_WGS84 == 6378137.0
    assert earth.F_WGS84 == 1.0 / 298.257223563
    assert earth.GM_WGS84 == 3.986004418e14
    assert earth.J2_WGS84 == 1.08262982e-3
    assert earth.OMEGA_WGS84 == 7.292115e-5

    assert earth.B_WGS84 == pytest.approx(6356752.314245179, abs=1e-6)
    assert earth.E2_WGS84 == pytest.approx(earth.F_WGS84 * (2.0 - earth.F_WGS84), rel=1e-15)
    # The recorded difference against JSBSim's reported value, so a future
    # reader meets it as a number rather than as a surprise.
    assert abs(earth.B_WGS84 - 6356752.314186481) == pytest.approx(5.87e-5, rel=0.02)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `../../../.venv/Scripts/python.exe -m pytest atisim/tests/test_earth.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'atisim.earth'`

- [ ] **Step 3: Write minimal implementation**

Create `atisim/earth.py`:

```python
"""WGS-84 geodesy, gravity, and the frames of a rotating Earth.

Conventions, fixed everywhere in this module:

  ECEF     Earth-centred Earth-fixed. x through (0 lat, 0 lon), z through the
           north pole. Rotates with the Earth at OMEGA_WGS84.
  NED      Local North-East-Down at a stated geodetic latitude/longitude.
  Latitude is GEODETIC unless a name says `geocentric`. This is not a
           preference: JSBSim's local frame uses geodetic latitude, verified to
           machine precision, and using geocentric instead costs 65 m/s in
           v_north at 47 degrees.

EVERY CONSTANT HERE WAS RECOVERED FROM THE RUNNING JSBSim BINARY (v1.3.1,
GitHub build 1837, commit 3b25f25e) rather than transcribed. JSBSim ships as a
compiled wheel with no C++ source, so a transcription could not have been
checked. The recovery and its residuals are in
docs/superpowers/specs/2026-08-26-wgs84-earth-rotation-design.md section 2.
"""

import jax
import jax.numpy as jnp
from jax import Array

# -- Defining constants. WGS-84 defines `a` and `f`; everything else follows. --
A_WGS84 = 6378137.0             # m, semi-major axis
F_WGS84 = 1.0 / 298.257223563   # flattening
GM_WGS84 = 3.986004418e14       # m^3/s^2, geocentric gravitational constant
J2_WGS84 = 1.08262982e-3        # second dynamic form factor
OMEGA_WGS84 = 7.292115e-5       # rad/s, Earth rotation rate

# -- Derived. NEVER quote these from a table; they follow from the pair above. --
B_WGS84 = A_WGS84 * (1.0 - F_WGS84)          # m, semi-minor axis
E2_WGS84 = F_WGS84 * (2.0 - F_WGS84)         # first eccentricity squared
EP2_WGS84 = E2_WGS84 / (1.0 - E2_WGS84)      # second eccentricity squared
```

- [ ] **Step 4: Run test to verify it passes**

Run: `../../../.venv/Scripts/python.exe -m pytest atisim/tests/test_earth.py -q`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add atisim/earth.py atisim/tests/test_earth.py
git commit -m "Add the WGS-84 defining constants, with b derived rather than quoted"
```

---

## Task 2: Geodetic and ECEF conversion

**Files:**
- Modify: `atisim/earth.py`
- Test: `atisim/tests/test_earth.py`

- [ ] **Step 1: Write the failing test**

Append to `atisim/tests/test_earth.py`:

```python
def test_geodetic_ecef_roundtrip_reaches_the_float64_floor_including_the_poles():
    """Bowring at a FIXED 3 iterations, measured over the whole latitude range.

    Fixed rather than converged because it must stay jittable and
    differentiable. Three is chosen because it reaches the float64 floor: the
    ulp of an ECEF coordinate is 9.3e-10 m, and measured worst round-trip error
    is 3.5e-6 m at 1 iteration, 1.4e-8 at 2, and 3.7e-9 at 3. A fourth buys
    nothing.

    The poles are INCLUDED. The textbook altitude form h = p/cos(lat) - N is
    singular there; this module uses the non-singular
    h = p cos(lat) + z sin(lat) - a sqrt(1 - e2 sin^2 lat) instead, and this
    test is what holds it to that.
    """
    worst_lat = worst_h = 0.0
    for lat in np.radians(np.linspace(-90.0, 90.0, 721)):
        for h in (0.0, 5.0e3, 12192.0, 2.0e4):
            r = earth.geodetic_to_ecef(lat, 0.7, h)
            back_lat, back_lon, back_h = earth.ecef_to_geodetic(r)
            worst_lat = max(worst_lat, abs(float(back_lat) - lat))
            worst_h = max(worst_h, abs(float(back_h) - h))
            assert float(back_lon) == pytest.approx(0.7, abs=1e-12)

    assert worst_h < 1.0e-8, f"altitude round-trip {worst_h:.2e} m"
    assert np.degrees(worst_lat) * 3600 < 1.0e-9, (
        f"latitude round-trip {np.degrees(worst_lat) * 3600:.2e} arcsec"
    )


def test_geodetic_to_ecef_puts_the_reference_points_where_they_belong():
    """Three points whose ECEF coordinates are the defining constants themselves."""
    equator = earth.geodetic_to_ecef(0.0, 0.0, 0.0)
    assert np.allclose(np.asarray(equator), [earth.A_WGS84, 0.0, 0.0], atol=1e-6)

    quarter = earth.geodetic_to_ecef(0.0, np.pi / 2, 0.0)
    assert np.allclose(np.asarray(quarter), [0.0, earth.A_WGS84, 0.0], atol=1e-6)

    pole = earth.geodetic_to_ecef(np.pi / 2, 0.0, 0.0)
    assert np.allclose(np.asarray(pole), [0.0, 0.0, earth.B_WGS84], atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `../../../.venv/Scripts/python.exe -m pytest atisim/tests/test_earth.py -q`
Expected: FAIL — `AttributeError: module 'atisim.earth' has no attribute 'geodetic_to_ecef'`

- [ ] **Step 3: Write minimal implementation**

Append to `atisim/earth.py`:

```python
# Bowring's method, at a FIXED iteration count so it stays jittable and
# differentiable. Three reaches the float64 floor -- measured 3.5e-6 m at one
# iteration, 1.4e-8 at two, 3.7e-9 at three, against a 9.3e-10 m ECEF ulp.
BOWRING_ITERATIONS = 3


def geodetic_to_ecef(lat: Array, lon: Array, h: Array) -> Array:
    """Geodetic (rad, rad, m) -> ECEF (3,) m."""
    s, c = jnp.sin(lat), jnp.cos(lat)
    n = A_WGS84 / jnp.sqrt(1.0 - E2_WGS84 * s * s)
    return jnp.array([
        (n + h) * c * jnp.cos(lon),
        (n + h) * c * jnp.sin(lon),
        (n * (1.0 - E2_WGS84) + h) * s,
    ])


def ecef_to_geodetic(r: Array) -> tuple[Array, Array, Array]:
    """ECEF (3,) m -> (lat, lon, h) in rad, rad, m. Latitude is GEODETIC.

    Altitude uses h = p cos(lat) + z sin(lat) - a sqrt(1 - e2 sin^2 lat), which
    is exact and NON-SINGULAR. The textbook h = p/cos(lat) - N divides by zero
    at the poles; under jit that is a silent inf rather than an exception, so
    the singular form is not used at all rather than guarded.
    """
    x, y, z = r[0], r[1], r[2]
    lon = jnp.arctan2(y, x)
    p = jnp.sqrt(x * x + y * y)

    theta = jnp.arctan2(z * A_WGS84, p * B_WGS84)
    lat = jnp.arctan2(
        z + EP2_WGS84 * B_WGS84 * jnp.sin(theta) ** 3,
        p - E2_WGS84 * A_WGS84 * jnp.cos(theta) ** 3,
    )

    def refine(lat, _):
        n = A_WGS84 / jnp.sqrt(1.0 - E2_WGS84 * jnp.sin(lat) ** 2)
        return jnp.arctan2(z + E2_WGS84 * n * jnp.sin(lat), p), None

    lat, _ = jax.lax.scan(refine, lat, None, length=BOWRING_ITERATIONS - 1)

    s, c = jnp.sin(lat), jnp.cos(lat)
    h = p * c + z * s - A_WGS84 * jnp.sqrt(1.0 - E2_WGS84 * s * s)
    return lat, lon, h
```

- [ ] **Step 4: Run test to verify it passes**

Run: `../../../.venv/Scripts/python.exe -m pytest atisim/tests/test_earth.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add atisim/earth.py atisim/tests/test_earth.py
git commit -m "Add WGS-84 geodetic/ECEF conversion, pole-safe and at the float64 floor"
```

---

## Task 3: The ECEF-to-NED matrix

**Files:**
- Modify: `atisim/earth.py`
- Test: `atisim/tests/test_earth.py`

- [ ] **Step 1: Write the failing test**

Append to `atisim/tests/test_earth.py`:

```python
def test_ecef_to_ned_matrix_is_orthonormal_and_points_down_along_the_ellipsoid_normal():
    """Down must be the GEODETIC normal, not the geocentric radius.

    These differ by up to 0.19 degrees at 45 latitude. Using the geocentric
    direction costs 65 m/s in v_north at 47 degrees, measured against JSBSim,
    which is why this is asserted rather than assumed.
    """
    for lat_deg in (-89.0, -45.0, 0.0, 12.5, 47.0, 89.0):
        lat, lon = np.radians(lat_deg), 0.4
        m = np.asarray(earth.ecef_to_ned_matrix(lat, lon))

        assert np.allclose(m @ m.T, np.eye(3), atol=1e-13), f"not orthonormal at {lat_deg}"
        assert np.linalg.det(m) == pytest.approx(1.0, abs=1e-13)

        # `down` must be the inward geodetic normal: moving 1 m along -down
        # from the surface must raise geodetic altitude by exactly 1 m.
        surface = earth.geodetic_to_ecef(lat, lon, 0.0)
        up_ecef = -m[2]
        _, _, h = earth.ecef_to_geodetic(surface + np.asarray(up_ecef))
        assert float(h) == pytest.approx(1.0, abs=1e-6), f"down is not the normal at {lat_deg}"

        # North at the pole-facing side must raise latitude.
        north_ecef = m[0]
        lat2, _, _ = earth.ecef_to_geodetic(surface + np.asarray(north_ecef))
        assert float(lat2) > lat
```

- [ ] **Step 2: Run test to verify it fails**

Run: `../../../.venv/Scripts/python.exe -m pytest atisim/tests/test_earth.py -q`
Expected: FAIL — `AttributeError: module 'atisim.earth' has no attribute 'ecef_to_ned_matrix'`

- [ ] **Step 3: Write minimal implementation**

Append to `atisim/earth.py`:

```python
def ecef_to_ned_matrix(lat: Array, lon: Array) -> Array:
    """ECEF -> local NED rotation. v_ned = ecef_to_ned_matrix(lat, lon) @ v_ecef.

    `lat` is GEODETIC. Verified against JSBSim: rotating its ECEF velocity by
    this matrix at the geodetic latitude reproduces its own
    velocities/v-north|east|down to machine precision, and at the geocentric
    latitude it does not.
    """
    sf, cf = jnp.sin(lat), jnp.cos(lat)
    sl, cl = jnp.sin(lon), jnp.cos(lon)
    return jnp.array([
        [-sf * cl, -sf * sl, cf],
        [-sl, cl, jnp.zeros_like(lat)],
        [-cf * cl, -cf * sl, -sf],
    ])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `../../../.venv/Scripts/python.exe -m pytest atisim/tests/test_earth.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add atisim/earth.py atisim/tests/test_earth.py
git commit -m "Add the ECEF-to-NED matrix, on geodetic latitude"
```

---

## Task 4: Gravity, EarthModel and Anchor

**Files:**
- Modify: `atisim/earth.py`
- Test: `atisim/tests/test_earth.py`

- [ ] **Step 1: Write the failing test**

Append to `atisim/tests/test_earth.py`:

```python
def test_j2_gravity_matches_the_values_read_off_the_jsbsim_binary():
    """The two anchor values, measured from JSBSim v1.3.1 at sea level.

    These are GRAVITATION, not apparent gravity: JSBSim reports 9.8142 at the
    equator, not the 9.7803 that includes the centrifugal term. The centrifugal
    term is a SEPARATE term in the equation of motion, and folding it into
    gravity here would double-count it.
    """
    equator = earth.gravitation(earth.geodetic_to_ecef(0.0, 0.0, 0.0), earth.WGS84_J2)
    pole = earth.gravitation(earth.geodetic_to_ecef(np.pi / 2, 0.0, 0.0), earth.WGS84_J2)

    assert float(np.linalg.norm(np.asarray(equator))) == pytest.approx(
        9.814197353250055, rel=1e-11
    )
    assert float(np.linalg.norm(np.asarray(pole))) == pytest.approx(
        9.832066846743299, rel=1e-9
    )
    # It points inward.
    assert float(np.asarray(equator)[0]) < 0.0
    assert float(np.asarray(pole)[2]) < 0.0


def test_j2_is_what_separates_the_two_gravity_models():
    """The inverse-square model is JSBSim's `gravity-model = 0`, and it differs.

    Carried so a test can isolate what J2 alone is worth. At the pole the two
    differ by roughly 3 J2 = 0.32%, which is far above anything this project
    calls agreement, so the choice of model is not a detail.
    """
    r = earth.geodetic_to_ecef(np.pi / 2, 0.0, 0.0)
    j2 = float(np.linalg.norm(np.asarray(earth.gravitation(r, earth.WGS84_J2))))
    inv = float(np.linalg.norm(np.asarray(earth.gravitation(r, earth.WGS84_INVERSE_SQUARE))))
    assert abs(j2 / inv - 1.0) == pytest.approx(3.0 * earth.J2_WGS84 * (earth.A_WGS84 / earth.B_WGS84) ** 2, rel=0.02)


def test_flat_is_constant_g_along_the_local_vertical_and_does_not_rotate():
    """FLAT is a CONFIGURATION of the one plant, not a second implementation."""
    from atisim.atmosphere import G0

    r = earth.geodetic_to_ecef(np.radians(47.0), 0.3, 9144.0)
    g = np.asarray(earth.gravitation(r, earth.FLAT))
    assert float(np.linalg.norm(g)) == pytest.approx(G0, rel=1e-14)
    # Along the inward radius: FLAT is spherical, so down is -r_hat.
    assert np.allclose(g / np.linalg.norm(g), -np.asarray(r) / np.linalg.norm(np.asarray(r)), atol=1e-14)
    assert earth.FLAT.rotation_rate == 0.0
    assert earth.WGS84_J2.rotation_rate == earth.OMEGA_WGS84


def test_an_anchor_carries_its_own_ecef_position_and_frame():
    anchor = earth.anchor_at(np.radians(47.0), np.radians(11.0), 0.0)
    assert np.allclose(
        np.asarray(anchor.r_ecef),
        np.asarray(earth.geodetic_to_ecef(np.radians(47.0), np.radians(11.0), 0.0)),
        atol=1e-9,
    )
    m = np.asarray(anchor.T_e2l)
    assert np.allclose(m @ m.T, np.eye(3), atol=1e-13)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `../../../.venv/Scripts/python.exe -m pytest atisim/tests/test_earth.py -q`
Expected: FAIL — `AttributeError: module 'atisim.earth' has no attribute 'gravitation'`

- [ ] **Step 3: Write minimal implementation**

Append to `atisim/earth.py`:

```python
from typing import NamedTuple

from atisim.atmosphere import G0


class EarthModel(NamedTuple):
    """Which Earth the plant is flying over.

    Carried as a STATIC argument through `integrate.step`, so every branch below
    resolves at trace time and costs nothing under jit. That is also why the
    fields are plain str/float rather than arrays -- a NamedTuple of those is
    hashable, and a static argument must be.

    `gravity` is one of "j2", "inverse_square", "constant".
    """

    gravity: str
    rotation_rate: float
    ellipsoidal: bool


WGS84_J2 = EarthModel("j2", OMEGA_WGS84, True)
WGS84_INVERSE_SQUARE = EarthModel("inverse_square", OMEGA_WGS84, True)
# FLAT: non-rotating, spherical, constant g along the local vertical. It is a
# CONFIGURATION of this one plant, not a second implementation retained
# alongside. It does NOT reproduce the pre-Earth model bit-for-bit -- see the
# design doc section 8 and ASSUMPTIONS.md F4.
FLAT = EarthModel("constant", 0.0, False)


def gravitation(r_ecef: Array, model: EarthModel) -> Array:
    """Gravitational acceleration in the ECEF frame, (3,) m/s^2.

    GRAVITATION, NOT APPARENT GRAVITY. The centrifugal term belongs to the
    equation of motion in `dynamics.derivatives` and is not folded in here.
    JSBSim makes the same split, which is why it reports 9.8142 m/s^2 at the
    equator rather than 9.7803.
    """
    radius = jnp.linalg.norm(r_ecef)
    direction = r_ecef / radius

    if model.gravity == "constant":
        return -G0 * direction

    magnitude = GM_WGS84 / (radius * radius)
    if model.gravity == "inverse_square":
        return -magnitude * direction

    if model.gravity != "j2":
        raise ValueError(f"unknown gravity model {model.gravity!r}")

    # J2 zonal harmonic. `sin_gc` is the sine of the GEOCENTRIC latitude, which
    # is what a spherical-harmonic expansion is written in -- unlike the local
    # frame, which is geodetic.
    sin_gc = r_ecef[2] / radius
    common = 1.5 * J2_WGS84 * (A_WGS84 / radius) ** 2
    horizontal = 1.0 + common * (1.0 - 5.0 * sin_gc * sin_gc)
    vertical = 1.0 + common * (3.0 - 5.0 * sin_gc * sin_gc)
    return -magnitude * jnp.array([
        direction[0] * horizontal,
        direction[1] * horizontal,
        direction[2] * vertical,
    ])


class Anchor(NamedTuple):
    """A run's geodetic origin: where local NED is pinned.

    A traced pytree, not a static argument, because it carries arrays. It is
    constant for the life of a run.

    THERE IS DELIBERATELY NO DEFAULT ANCHOR. A default would silently pick a
    latitude, and latitude now changes the answer -- the Coriolis term, the
    trimmed bank angle and the gravity magnitude all depend on it. Every caller
    states where it is flying.
    """

    lat: Array
    lon: Array
    h: Array
    r_ecef: Array   # (3,)
    T_e2l: Array    # (3,3), ECEF -> NED at (lat, lon)


def anchor_at(lat: Array, lon: Array, h: Array) -> Anchor:
    lat, lon, h = jnp.asarray(lat), jnp.asarray(lon), jnp.asarray(h)
    return Anchor(
        lat=lat, lon=lon, h=h,
        r_ecef=geodetic_to_ecef(lat, lon, h),
        T_e2l=ecef_to_ned_matrix(lat, lon),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `../../../.venv/Scripts/python.exe -m pytest atisim/tests/test_earth.py -q`
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
git add atisim/earth.py atisim/tests/test_earth.py
git commit -m "Add J2 and inverse-square gravity, the EarthModel config, and the run Anchor"
```

---

## Task 5: Provenance entries for the five constants

**Files:**
- Modify: `atisim/provenance.py`
- Test: `atisim/tests/test_earth.py`

- [ ] **Step 1: Write the failing test**

Append to `atisim/tests/test_earth.py`:

```python
def test_every_wgs84_constant_is_in_the_provenance_ledger():
    """SOURCED, with the JSBSim recovery named as the CHECK, not as the source.

    The distinction matters and the ledger's own rules enforce it: SOURCED means
    read from a cited table. WGS-84 is that table. The agreement with JSBSim is
    evidence that this model and that one read the same table, which is a
    verification, not a provenance.
    """
    from atisim.provenance import LEDGER

    for name in ("earth.a", "earth.f", "earth.GM", "earth.J2", "earth.omega"):
        assert name in LEDGER, f"{name} missing from the provenance ledger"
        assert LEDGER[name].category == "SOURCED"
        assert "WGS-84" in LEDGER[name].detail

    for name in ("earth.b", "earth.e2"):
        assert LEDGER[name].category == "DERIVED"
        assert LEDGER[name].inputs, f"{name} claims DERIVED with no inputs"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `../../../.venv/Scripts/python.exe -m pytest atisim/tests/test_earth.py::test_every_wgs84_constant_is_in_the_provenance_ledger -q`
Expected: FAIL — `AssertionError: earth.a missing from the provenance ledger`

- [ ] **Step 3: Write minimal implementation**

In `atisim/provenance.py`, add before the closing brace of `LEDGER`:

```python
    # -- WGS-84. The ellipsoid, gravity field and rotation rate. -----------
    # SOURCED against the standard; the JSBSim agreement recorded in `detail`
    # is a CHECK that both implementations read the same standard, not the
    # source itself.
    "earth.a": Entry(
        "SOURCED",
        "6378137.0 m semi-major axis. WGS-84 defining parameter. Recovered from "
        "JSBSim v1.3.1 build 1837 inertial/sea-level-radius_ft at the equator, exact",
    ),
    "earth.f": Entry(
        "SOURCED",
        "1/298.257223563 flattening. WGS-84 defining parameter. Recovered from the "
        "JSBSim polar sea-level radius, 8e-10 relative",
    ),
    "earth.GM": Entry(
        "SOURCED",
        "3.986004418e14 m^3/s^2 geocentric gravitational constant. WGS-84. Solved "
        "from JSBSim's equatorial and polar gravity, 4.1e-13 relative",
    ),
    "earth.J2": Entry(
        "SOURCED",
        "1.08262982e-3 second dynamic form factor. WGS-84. Solved from the same "
        "pair, 1.3e-11 relative",
    ),
    "earth.omega": Entry(
        "SOURCED",
        "7.292115e-5 rad/s Earth rotation rate. WGS-84. Recovered from JSBSim's "
        "d(position/epa-rad)/dt over 10 s, exact",
    ),
    "earth.b": Entry(
        "DERIVED",
        "a(1-f) = 6356752.314245 m. NOT taken from JSBSim, which reports "
        "6356752.314186 m -- a 5.87e-5 m difference that is its internal storage "
        "in feet round-tripping, not a different ellipsoid",
        ("earth.a", "earth.f"),
    ),
    "earth.e2": Entry(
        "DERIVED", "f(2-f), first eccentricity squared", ("earth.f",)
    ),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `../../../.venv/Scripts/python.exe -m pytest atisim/tests/test_earth.py atisim/tests/test_provenance.py -q`
Expected: PASS — the new test plus every existing provenance test (the ledger's acyclicity and chain checks must still hold)

- [ ] **Step 5: Commit**

```bash
git add atisim/provenance.py atisim/tests/test_earth.py
git commit -m "Put the WGS-84 constants in the provenance ledger"
```

---

## Task 6: The frozen JSBSim Earth reference

**Files:**
- Create: `scripts/gen_jsbsim_earth_reference.py`
- Create: `atisim/tests/data/jsbsim_earth_reference.xml` (generated)
- Create: `atisim/earth_ref.py`

- [ ] **Step 1: Write the generator**

Create `scripts/gen_jsbsim_earth_reference.py`:

```python
"""Freeze the JSBSim Earth-model reference: geodesy, gravity, and every EOM term.

Run with an interpreter that has JSBSim installed. The project venv deliberately
does NOT, so the suite runs anywhere:

    C:/Users/mateusz/AppData/Local/Programs/Python/Python310/python.exe \
        scripts/gen_jsbsim_earth_reference.py

Writes atisim/tests/data/jsbsim_earth_reference.xml.

THE RULE FROM gen_jsbsim_reference.py APPLIES HERE AND IS NOT OPTIONAL:
READ BACK, NEVER ASSUME. Feeding a NOMINAL altitude into the gravity formula
instead of the position JSBSim actually reached puts the agreement at 2.4e-6
relative; reading the ECEF position back off the engine puts it at 3.6e-13.
That factor of ten million is the whole difference between a verification and a
coincidence, and it was measured while writing the design.

This is the fourth and last file under scripts/ that imports jsbsim. Nothing
under atisim/ may import it.
"""

import sys
from pathlib import Path

import jsbsim
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from atisim.units import FT2M, LBF2N, SLUG2KG, SLUG_FT2_TO_KG_M2  # noqa: E402

OUT = ROOT / "atisim" / "tests" / "data" / "jsbsim_earth_reference.xml"

# The probe grid. Latitude spans both hemispheres because a sign error in the
# NED matrix or in J2 is antisymmetric and would cancel on one hemisphere alone.
PROBES = [
    # (lat_deg, lon_deg, alt_ft, psi_deg, phi_deg, aileron, rudder)
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (47.0, 11.0, 30000.0, 30.0, 12.0, 0.0, 0.0),
    (-33.0, 151.0, 20000.0, 240.0, -8.0, 0.0, 0.0),
    (60.0, -120.0, 5000.0, 90.0, 0.0, 0.6, 0.3),
    (-70.0, 45.0, 40000.0, 180.0, 25.0, 0.6, 0.3),
    (10.0, -60.0, 10000.0, 315.0, -20.0, 0.3, -0.4),
]

SETTLE_STEPS = 240   # 2 s at 120 Hz, so the FCS and engines are not in a transient


def f(x):
    return f"{float(x):.12e}"


def vec(a):
    return " ".join(f(v) for v in np.asarray(a).ravel())


def euler_to_matrix(phi, theta, psi):
    """Local NED -> body, 3-2-1. Rebuilt here rather than imported from atisim,
    so the reference cannot inherit a convention error from the code under test.
    """
    cp, sp = np.cos(phi), np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    cy, sy = np.cos(psi), np.sin(psi)
    return np.array([
        [ct * cy, ct * sy, -st],
        [sp * st * cy - cp * sy, sp * st * sy + cp * cy, sp * ct],
        [cp * st * cy + sp * sy, cp * st * sy - sp * cy, cp * ct],
    ])


def ecef_to_ned(lat, lon):
    sf, cf, sl, cl = np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon)
    return np.array([[-sf * cl, -sf * sl, cf], [-sl, cl, 0.0], [-cf * cl, -cf * sl, -sf]])


def probe(fdm_root, lat_deg, lon_deg, alt_ft, psi_deg, phi_deg, aileron, rudder):
    fdm = jsbsim.FGFDMExec(fdm_root)
    fdm.set_debug_level(0)
    fdm.load_model("737")
    fdm["ic/lat-geod-deg"] = lat_deg
    fdm["ic/long-gc-deg"] = lon_deg
    fdm["ic/h-sl-ft"] = alt_ft
    fdm["ic/vc-kts"] = 280.0
    fdm["ic/psi-true-deg"] = psi_deg
    fdm["ic/phi-deg"] = phi_deg
    fdm.run_ic()
    fdm["fcs/aileron-cmd-norm"] = aileron
    fdm["fcs/rudder-cmd-norm"] = rudder
    for _ in range(SETTLE_STEPS):
        fdm.run()

    g = fdm.__getitem__

    # -- everything READ BACK from the engine, never assumed ---------------
    lat = g("position/lat-geod-rad")
    lon = g("position/long-gc-rad")
    r_ecef = np.array([g(f"position/ecef-{c}-ft") for c in "xyz"]) * FT2M
    phi, theta, psi = (g("attitude/phi-rad"), g("attitude/theta-rad"), g("attitude/psi-rad"))

    mass = g("inertia/mass-slugs") * SLUG2KG
    inertia = np.array([
        [g("inertia/ixx-slugs_ft2"), -g("inertia/ixy-slugs_ft2"), g("inertia/ixz-slugs_ft2")],
        [-g("inertia/ixy-slugs_ft2"), g("inertia/iyy-slugs_ft2"), -g("inertia/iyz-slugs_ft2")],
        [g("inertia/ixz-slugs_ft2"), -g("inertia/iyz-slugs_ft2"), g("inertia/izz-slugs_ft2")],
    ]) * SLUG_FT2_TO_KG_M2
    # NOTE THE +Ixz IN THE OFF-DIAGONAL. Measured: the rotational equation closes
    # to 2.4e-14 with +Ixz and is out by 2.3e-3 with -Ixz.

    force = np.array([
        g(f"forces/fb{c}-aero-lbs") + g(f"forces/fb{c}-prop-lbs") + g(f"forces/fb{c}-gear-lbs")
        for c in "xyz"
    ]) * LBF2N
    weight = np.array([g(f"forces/fb{c}-weight-lbs") for c in "xyz"]) * LBF2N
    moment = np.array([
        g("moments/l-total-lbsft"), g("moments/m-total-lbsft"), g("moments/n-total-lbsft")
    ]) * LBF2N * FT2M

    vel_body = np.array([g(f"velocities/{c}-fps") for c in "uvw"]) * FT2M
    vel_ecef = np.array([g(f"velocities/ecef-{c}-fps") for c in "xyz"]) * FT2M
    vel_ned = np.array([
        g("velocities/v-north-fps"), g("velocities/v-east-fps"), g("velocities/v-down-fps")
    ]) * FT2M
    pqr = np.array([g(f"velocities/{c}-rad_sec") for c in "pqr"])
    pqri = np.array([g(f"velocities/{c}i-rad_sec") for c in ("p", "q", "r")])
    uvwdot = np.array([g(f"accelerations/{c}dot-ft_sec2") for c in "uvw"]) * FT2M
    pqrdot = np.array([g(f"accelerations/{c}dot-rad_sec2") for c in "pqr"])
    pqridot = np.array([g(f"accelerations/{c}idot-rad_sec2") for c in ("p", "q", "r")])

    return dict(
        lat=lat, lon=lon, r_ecef=r_ecef, euler=[phi, theta, psi],
        t_l2b=euler_to_matrix(phi, theta, psi), t_e2l=ecef_to_ned(lat, lon),
        mass=mass, inertia=inertia, force=force, weight=weight, moment=moment,
        gravity_magnitude=g("accelerations/gravity-ft_sec2") * FT2M,
        vel_body=vel_body, vel_ecef=vel_ecef, vel_ned=vel_ned,
        pqr=pqr, pqri=pqri, uvwdot=uvwdot, pqrdot=pqrdot, pqridot=pqridot,
    )


def main():
    root = str(Path(jsbsim.__file__).parent)
    lines = ['<?xml version="1.0" encoding="utf-8"?>', "<jsbsim_earth_reference>"]
    lines.append(f"  <version>{jsbsim.FGFDMExec(root).get_version()}</version>")
    for spec in PROBES:
        p = probe(root, *spec)
        lines.append(f'  <probe lat_deg="{f(spec[0])}" lon_deg="{f(spec[1])}" alt_ft="{f(spec[2])}">')
        for key in ("r_ecef", "euler", "force", "weight", "moment", "vel_body",
                    "vel_ecef", "vel_ned", "pqr", "pqri", "uvwdot", "pqrdot", "pqridot"):
            lines.append(f"    <{key}>{vec(p[key])}</{key}>")
        lines.append(f"    <t_l2b>{vec(p['t_l2b'])}</t_l2b>")
        lines.append(f"    <t_e2l>{vec(p['t_e2l'])}</t_e2l>")
        lines.append(f"    <inertia>{vec(p['inertia'])}</inertia>")
        lines.append(f"    <lat>{f(p['lat'])}</lat>")
        lines.append(f"    <lon>{f(p['lon'])}</lon>")
        lines.append(f"    <mass>{f(p['mass'])}</mass>")
        lines.append(f"    <gravity_magnitude>{f(p['gravity_magnitude'])}</gravity_magnitude>")
        lines.append("  </probe>")
    lines.append("</jsbsim_earth_reference>")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT.relative_to(ROOT)}  ({len(lines)} lines, {len(PROBES)} probes)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Generate the reference**

Run:

```bash
C:/Users/mateusz/AppData/Local/Programs/Python/Python310/python.exe scripts/gen_jsbsim_earth_reference.py
```

Expected: `wrote atisim/tests/data/jsbsim_earth_reference.xml  (... lines, 6 probes)`

- [ ] **Step 3: Write the reader**

Create `atisim/earth_ref.py`:

```python
"""Reader for the frozen JSBSim Earth reference.

Deliberately does NOT import jsbsim, and nothing that imports this module may.
Same rule as jsbsim_ref.py and for the same reason: the suite must run on a
machine with no JSBSim, and reference drift must show up in git diff rather
than as a mysterious change in test results.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import NamedTuple

import numpy as np

REFERENCE = Path(__file__).parent / "tests" / "data" / "jsbsim_earth_reference.xml"


class Probe(NamedTuple):
    lat: float
    lon: float
    r_ecef: np.ndarray       # (3,) m
    euler: np.ndarray        # (3,) rad, phi theta psi
    t_l2b: np.ndarray        # (3,3) local NED -> body
    t_e2l: np.ndarray        # (3,3) ECEF -> local NED
    mass: float              # kg
    inertia: np.ndarray      # (3,3) kg m^2, +Ixz convention
    force: np.ndarray        # (3,) N, body, EXCLUDING weight
    weight: np.ndarray       # (3,) N, body
    moment: np.ndarray       # (3,) N m, body
    gravity_magnitude: float # m/s^2
    vel_body: np.ndarray     # (3,) m/s, ECEF-relative, body axes
    vel_ecef: np.ndarray     # (3,) m/s, ECEF frame
    vel_ned: np.ndarray      # (3,) m/s
    pqr: np.ndarray          # (3,) rad/s, body rel ECEF
    pqri: np.ndarray         # (3,) rad/s, body rel ECI
    uvwdot: np.ndarray       # (3,) m/s^2
    pqrdot: np.ndarray       # (3,) rad/s^2
    pqridot: np.ndarray      # (3,) rad/s^2


def _v(node, name, shape=None):
    text = node.find(name).text
    a = np.array([float(x) for x in text.split()])
    return a.reshape(shape) if shape else a


def load(path: Path = REFERENCE) -> list[Probe]:
    root = ET.parse(path).getroot()
    out = []
    for node in root.findall("probe"):
        out.append(Probe(
            lat=float(node.find("lat").text),
            lon=float(node.find("lon").text),
            r_ecef=_v(node, "r_ecef"),
            euler=_v(node, "euler"),
            t_l2b=_v(node, "t_l2b", (3, 3)),
            t_e2l=_v(node, "t_e2l", (3, 3)),
            mass=float(node.find("mass").text),
            inertia=_v(node, "inertia", (3, 3)),
            force=_v(node, "force"),
            weight=_v(node, "weight"),
            moment=_v(node, "moment"),
            gravity_magnitude=float(node.find("gravity_magnitude").text),
            vel_body=_v(node, "vel_body"),
            vel_ecef=_v(node, "vel_ecef"),
            vel_ned=_v(node, "vel_ned"),
            pqr=_v(node, "pqr"),
            pqri=_v(node, "pqri"),
            uvwdot=_v(node, "uvwdot"),
            pqrdot=_v(node, "pqrdot"),
            pqridot=_v(node, "pqridot"),
        ))
    return out
```

- [ ] **Step 4: Verify the reader round-trips**

Run:

```bash
../../../.venv/Scripts/python.exe -c "from atisim import earth_ref; p=earth_ref.load(); print(len(p),'probes'); print(p[1].r_ecef, p[1].gravity_magnitude)"
```

Expected: `6 probes` and a plausible ECEF triple with gravity near 9.79

- [ ] **Step 5: Commit**

```bash
git add scripts/gen_jsbsim_earth_reference.py atisim/earth_ref.py atisim/tests/data/jsbsim_earth_reference.xml
git commit -m "Freeze a JSBSim Earth reference over six probes, read back from the engine"
```

---

## Task 7: Term-by-term agreement with JSBSim

**Files:**
- Modify: `atisim/tests/test_earth.py`

This is verification check 1 of the design. It asserts the geodesy and gravity now; the EOM rows are added in Task 10 once `derivatives` exists.

- [ ] **Step 1: Write the failing test**

Append to `atisim/tests/test_earth.py`:

```python
from atisim import earth_ref


def _probes():
    return earth_ref.load()


def test_our_geodesy_reproduces_jsbsims_own_position_and_frame():
    """Round-trip JSBSim's ECEF through our geodesy and back to its own frame.

    This is the check that would have failed had we used geocentric latitude:
    the t_e2l comparison is the one that costs 65 m/s in v_north at 47 degrees.
    """
    for p in _probes():
        lat, lon, h = earth.ecef_to_geodetic(p.r_ecef)
        assert float(lat) == pytest.approx(p.lat, abs=1e-12)
        assert float(lon) == pytest.approx(p.lon, abs=1e-12)

        back = earth.geodetic_to_ecef(lat, lon, h)
        assert np.allclose(np.asarray(back), p.r_ecef, atol=1e-6)

        assert np.allclose(np.asarray(earth.ecef_to_ned_matrix(lat, lon)), p.t_e2l, atol=1e-12)

        # And the frame actually maps JSBSim's ECEF velocity onto its own NED.
        assert np.allclose(p.t_e2l @ p.vel_ecef, p.vel_ned, atol=1e-9)


def test_our_j2_gravity_reproduces_jsbsims_at_every_probe():
    """3.6e-13 relative when the position is read back from the engine.

    Feeding a nominal altitude instead puts this at 2.4e-6. The tolerance is set
    tight enough that the nominal-altitude mistake fails it.
    """
    for p in _probes():
        ours = np.asarray(earth.gravitation(p.r_ecef, earth.WGS84_J2))
        assert float(np.linalg.norm(ours)) == pytest.approx(p.gravity_magnitude, rel=1e-10)

        # Direction too, via JSBSim's own weight force.
        theirs_body = p.weight / p.mass
        t_b2e = (p.t_l2b @ p.t_e2l).T
        assert np.allclose(t_b2e @ theirs_body, ours, atol=1e-7)
```

- [ ] **Step 2: Run test to verify it fails, then passes**

Run: `../../../.venv/Scripts/python.exe -m pytest atisim/tests/test_earth.py -q`
Expected: PASS. If `test_our_geodesy_...` fails on `t_e2l`, the latitude convention is wrong — re-read design §2.

- [ ] **Step 3: Commit**

```bash
git add atisim/tests/test_earth.py
git commit -m "Assert our geodesy and gravity term-by-term against the frozen JSBSim reference"
```

---

## Task 8: The State, and forcing a loud break

**Files:**
- Modify: `atisim/state.py`
- Test: `atisim/tests/test_state.py`

**The hazard this task exists to defuse:** `quat_to_dcm(state.quat)` currently means *body→NED*. Under the new state it would mean *body→ECEF* — a different matrix that is still a valid rotation, so every one of the 40+ call sites would keep running and quietly return wrong answers. **Renaming the function turns all of them into import errors.** Do not skip this.

- [ ] **Step 1: Write the failing test**

Append to `atisim/tests/test_state.py`:

```python
import numpy as np
import pytest

from atisim import earth


def test_quat_to_dcm_no_longer_exists_under_its_old_name():
    """The rename is the safety mechanism, not cosmetics.

    `quat_to_dcm(state.quat)` meant body->NED and now means body->ECEF. Both are
    valid rotation matrices, so every one of the 40-odd call sites would have
    kept running and returned a wrong answer silently. Deleting the name makes
    each one an ImportError instead. If this test ever fails because someone
    re-added the alias, the aliasing is the bug.
    """
    import atisim.state as st

    assert not hasattr(st, "quat_to_dcm"), (
        "quat_to_dcm was re-added; it is ambiguous about frames now and each "
        "call site must say body_to_ecef or body_to_ned"
    )
    assert hasattr(st, "quat_to_matrix")


def test_dcm_to_quat_round_trips_including_the_trace_minus_one_case():
    """Shepperd, branch-free. The naive trace form fails near trace = -1."""
    import atisim.state as st

    rng = np.random.default_rng(0)
    for _ in range(200):
        q = rng.normal(size=4)
        q = q / np.linalg.norm(q)
        m = st.quat_to_matrix(jnp.asarray(q))
        back = np.asarray(st.dcm_to_quat(m))
        # q and -q are the same rotation.
        if np.dot(back, q) < 0:
            back = -back
        assert np.allclose(back, q, atol=1e-12)

    # The adversarial case: 180 degrees about y, where trace = -1.
    m = jnp.asarray(np.diag([-1.0, 1.0, -1.0]))
    back = st.dcm_to_quat(m)
    assert np.allclose(np.asarray(st.quat_to_matrix(back)), np.diag([-1.0, 1.0, -1.0]), atol=1e-12)


def test_pos_ned_is_a_rotation_of_the_offset_not_a_difference_of_large_numbers():
    """This is what preserves F4's round-off floor.

    An absolute ECEF coordinate has a 9.3e-10 m ulp against 1.8e-12 m for a
    12 km NED altitude -- 512x. Forming pos_ned by DIFFERENCING two absolute
    positions would inherit the coarse one. Rotating the stored offset does not.
    """
    import atisim.state as st

    anchor = earth.anchor_at(np.radians(47.0), np.radians(11.0), 0.0)
    offset = jnp.asarray([1234.5, -678.25, -9144.0])
    state = st.State(
        pos_ecef=anchor.T_e2l.T @ offset,
        vel_body=jnp.zeros(3),
        quat=jnp.asarray([1.0, 0.0, 0.0, 0.0]),
        omega=jnp.zeros(3),
    )
    assert np.allclose(np.asarray(st.pos_ned(state, anchor)), np.asarray(offset), atol=1e-9)


def test_altitude_is_geodetic_and_diverges_from_the_tangent_plane_with_range():
    """-pos_ned[2] is no longer altitude, and the difference is not small.

    The tangent plane falls away from the ellipsoid as d^2/2R: about 7.8 m at
    10 km of ground track and 785 m at 100 km. Anything still reading
    -pos_ned[2] as altitude is a bug this asserts the existence of.
    """
    import atisim.state as st

    anchor = earth.anchor_at(0.0, 0.0, 0.0)
    for distance, expected in ((10.0e3, 7.8), (100.0e3, 785.0)):
        offset = jnp.asarray([distance, 0.0, 0.0])   # level in the tangent plane
        state = st.State(
            pos_ecef=anchor.T_e2l.T @ offset,
            vel_body=jnp.zeros(3), quat=jnp.asarray([1.0, 0.0, 0.0, 0.0]), omega=jnp.zeros(3),
        )
        drop = -float(st.altitude(state, anchor))
        assert drop == pytest.approx(expected, rel=0.05), (
            f"tangent-plane drop at {distance/1e3:.0f} km is {drop:.1f} m, expected ~{expected}"
        )
```

Add `import jax.numpy as jnp` at the top of `test_state.py` if not already present.

- [ ] **Step 2: Run test to verify it fails**

Run: `../../../.venv/Scripts/python.exe -m pytest atisim/tests/test_state.py -q`
Expected: FAIL — `quat_to_dcm was re-added` / `AttributeError: 'atisim.state' has no attribute 'quat_to_matrix'`

- [ ] **Step 3: Rewrite `atisim/state.py`**

Replace the module docstring and the `State`/`altitude`/`quat_to_dcm` definitions:

```python
"""Simulation state and quaternion utilities.

Conventions, fixed everywhere in this package:

  Frames   ECEF is the propagation frame. LOCAL NED is a derived VIEW, taken at
           the aircraft's own geodetic position -- not at the anchor's, because
           the two diverge with range and the whole point of carrying an
           ellipsoid is to be right at range.
  Position `pos_ecef` is an OFFSET from the run anchor, not an absolute ECEF
           coordinate. This is the one deliberate deviation from JSBSim's
           FGPropagate, and it is numerical, not physical: an absolute
           coordinate's ulp is 512x coarser and would bind the convergence
           studies. See ASSUMPTIONS.md F4.
  Quat     [w, x, y, z], unit norm, rotates BODY vectors into ECEF.
  Omega    body rate relative to ECEF (JSBSim's vPQR), body axes.
  Euler    3-2-1 (yaw psi, pitch theta, roll phi) in the LOCAL NED frame.
           Display only -- the state carries the quaternion.
  Altitude GEODETIC, via `altitude(state, anchor)`. It is NOT -pos_ned[2] any
           more, and the two differ by 785 m at 100 km of ground track.

`quat_to_dcm` HAS BEEN DELETED, deliberately. It meant body->NED and would now
mean body->ECEF; both are valid rotations, so every call site would have kept
running and returned a wrong answer. The generic `quat_to_matrix` is
frame-agnostic, and the two named helpers say which frame they mean.
"""

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from atisim import earth


class State(NamedTuple):
    """13-element rigid-body state, on a rotating Earth."""

    pos_ecef: Array  # (3,) m, OFFSET from the run anchor
    vel_body: Array  # (3,) m/s, ECEF-relative, body axes (JSBSim vUVW)
    quat: Array      # (4,) w x y z, body -> ECEF
    omega: Array     # (3,) rad/s, body rate relative to ECEF (JSBSim vPQR)


class Controls(NamedTuple):
    elevator: Array  # rad, positive trailing-edge down
    aileron: Array   # rad, positive right-roll command
    rudder: Array    # rad, positive trailing-edge left
    throttle: Array  # 0-1


def quat_to_matrix(q: Array) -> Array:
    """Rotation matrix from a quaternion. FRAME-AGNOSTIC on purpose.

    For a state quaternion this is body -> ECEF. Use `dcm_body_to_ned` when you
    want the local frame, and never assume which one you have.
    """
    w, x, y, z = q
    return jnp.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def dcm_to_quat(m: Array) -> Array:
    """Quaternion [w,x,y,z] from a rotation matrix. Branch-free Shepperd.

    All four candidates are computed and the one with the largest denominator is
    selected, which is the numerically stable choice: the naive trace form loses
    all its precision near trace = -1. Branch-free because it runs under jit.
    """
    c = jnp.array([
        1.0 + m[0, 0] + m[1, 1] + m[2, 2],
        1.0 + m[0, 0] - m[1, 1] - m[2, 2],
        1.0 - m[0, 0] + m[1, 1] - m[2, 2],
        1.0 - m[0, 0] - m[1, 1] + m[2, 2],
    ])
    i = jnp.argmax(c)
    s = jnp.sqrt(jnp.maximum(c[i], 1e-300)) * 2.0
    candidates = jnp.stack([
        jnp.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]),
        jnp.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]),
        jnp.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]),
        jnp.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]),
    ])
    return quat_normalize(candidates[i])


def absolute_ecef(state: State, anchor: earth.Anchor) -> Array:
    """The aircraft's ABSOLUTE ECEF position. Formed only where geodesy needs it."""
    return anchor.r_ecef + state.pos_ecef


def geodetic(state: State, anchor: earth.Anchor) -> tuple[Array, Array, Array]:
    """(lat, lon, h) of the aircraft. Latitude is geodetic."""
    return earth.ecef_to_geodetic(absolute_ecef(state, anchor))


def altitude(state: State, anchor: earth.Anchor) -> Array:
    """GEODETIC altitude, m. Not -pos_ned[2] -- see the module docstring."""
    return geodetic(state, anchor)[2]


def pos_ned(state: State, anchor: earth.Anchor) -> Array:
    """Local NED offset from the anchor, m.

    A pure ROTATION of the stored offset. Never a difference of two absolute
    ECEF coordinates -- that would inherit the 512x coarser ulp and undo the
    reason position is stored as an offset at all.
    """
    return anchor.T_e2l @ state.pos_ecef


def dcm_body_to_ned(state: State, anchor: earth.Anchor) -> Array:
    """Body -> local NED, at the AIRCRAFT's position, not the anchor's.

    The distinction is the point of the ellipsoid: the local vertical at the
    aircraft is not the local vertical at the anchor once they are any distance
    apart, and using the anchor's frame here would put that error straight into
    the Euler angles and every attitude loop.
    """
    lat, lon, _ = geodetic(state, anchor)
    return earth.ecef_to_ned_matrix(lat, lon) @ quat_to_matrix(state.quat)


def quat_to_euler_ned(state: State, anchor: earth.Anchor) -> Array:
    """[phi, theta, psi] in rad, in the local NED frame. Display only."""
    return matrix_to_euler(dcm_body_to_ned(state, anchor))


def matrix_to_euler(dcm: Array) -> Array:
    """[phi, theta, psi] from a body -> NED matrix. Display only."""
    phi = jnp.arctan2(dcm[2, 1], dcm[2, 2])
    # Clip guards asin against a norm error of a few ulp at +-90 deg pitch.
    theta = -jnp.arcsin(jnp.clip(dcm[2, 0], -1.0, 1.0))
    psi = jnp.arctan2(dcm[1, 0], dcm[0, 0])
    return jnp.array([phi, theta, psi])


def state_from_ned(pos_ned_, vel_body, quat_ned, omega, anchor: earth.Anchor) -> State:
    """Build a State from the local-NED description most call sites already use.

    Exists so migrating the existing call sites is mechanical rather than a
    rewrite: they keep describing the aircraft in the terms they always did, and
    this converts once.

    `quat_ned` is body -> NED and `omega` is the body rate relative to ECEF.
    """
    pos_ecef = anchor.T_e2l.T @ jnp.asarray(pos_ned_)
    lat, lon, _ = earth.ecef_to_geodetic(anchor.r_ecef + pos_ecef)
    t_e2l = earth.ecef_to_ned_matrix(lat, lon)
    dcm_b2e = t_e2l.T @ quat_to_matrix(jnp.asarray(quat_ned))
    return State(
        pos_ecef=pos_ecef,
        vel_body=jnp.asarray(vel_body),
        quat=dcm_to_quat(dcm_b2e),
        omega=jnp.asarray(omega),
    )
```

Keep `quat_normalize`, `euler_to_quat` and `quat_derivative` exactly as they are. Delete `quat_to_dcm` and the old `quat_to_euler` and `altitude`.

- [ ] **Step 4: Run test to verify it passes**

Run: `../../../.venv/Scripts/python.exe -m pytest atisim/tests/test_state.py -q`
Expected: PASS for the new tests. **Other modules are now broken on import — that is the intended loud failure and Task 12 fixes them.**

- [ ] **Step 5: Commit**

```bash
git add atisim/state.py atisim/tests/test_state.py
git commit -m "Move State to ECEF, and delete quat_to_dcm so every frame assumption breaks loudly"
```

---

## Task 9: The equations of motion

**Files:**
- Modify: `atisim/dynamics.py`
- Test: `atisim/tests/test_earth.py`

- [ ] **Step 1: Write the failing test**

Append to `atisim/tests/test_earth.py`:

```python
def test_derivatives_reproduce_every_jsbsim_eom_term():
    """Verification check 1, the EOM rows. This is the heart of the work.

    Each probe carries JSBSim's own forces, moments, inertia, velocities and
    rates, plus the accelerations it computed from them. Feeding OUR equations
    the same inputs must give the same accelerations.

    Tolerances are tight because the design measured these relations closing to
    machine zero. A loose tolerance here would pass with a wrong sign on a small
    term, which is exactly the failure the design's frame-transfer probe hit.
    """
    import jax.numpy as jnp

    from atisim import earth as E
    from atisim.dynamics import earth_acceleration_terms

    for p in _probes():
        t_b2e = (p.t_l2b @ p.t_e2l).T
        omega_earth_body = t_b2e.T @ np.array([0.0, 0.0, E.OMEGA_WGS84])

        # The frame-transfer relation, which JSBSim states directly.
        assert np.allclose(p.pqri - p.pqr, omega_earth_body, atol=1e-12)

        accel, omega_dot = earth_acceleration_terms(
            force=jnp.asarray(p.force),
            moment=jnp.asarray(p.moment),
            mass=p.mass,
            inertia=jnp.asarray(p.inertia),
            inertia_inv=jnp.asarray(np.linalg.inv(p.inertia)),
            vel_body=jnp.asarray(p.vel_body),
            omega=jnp.asarray(p.pqr),
            quat_body_to_ecef=jnp.asarray(_quat_from(t_b2e)),
            r_ecef=jnp.asarray(p.r_ecef),
            model=E.WGS84_J2,
        )
        assert np.allclose(np.asarray(accel), p.uvwdot, atol=1e-9), (
            f"translational EOM at lat {np.degrees(p.lat):.1f}: "
            f"{np.asarray(accel)} vs {p.uvwdot}"
        )
        assert np.allclose(np.asarray(omega_dot), p.pqrdot, atol=1e-12), (
            f"rotational EOM at lat {np.degrees(p.lat):.1f}: "
            f"{np.asarray(omega_dot)} vs {p.pqrdot}"
        )


def _quat_from(m):
    """numpy Shepperd, so the test does not depend on the code under test."""
    c = np.array([
        1 + m[0, 0] + m[1, 1] + m[2, 2], 1 + m[0, 0] - m[1, 1] - m[2, 2],
        1 - m[0, 0] + m[1, 1] - m[2, 2], 1 - m[0, 0] - m[1, 1] + m[2, 2],
    ])
    i = int(np.argmax(c)); s = np.sqrt(c[i]) * 2.0
    q = [
        np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]),
        np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]),
        np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]),
        np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]),
    ][i]
    return q / np.linalg.norm(q)


def test_the_frame_transfer_sign_is_plus_and_a_wings_level_probe_cannot_tell():
    """The design's own near-miss, pinned so it cannot come back.

    Near wings-level, `omega_be x Omega_b` is about 1e-5 and the WRONG sign's
    residual is the same order -- both signs "fit". Only the high-rate probes
    separate them: measured 6.9e-18 for plus against 1.9e-5 for minus. This test
    asserts that the high-rate probes are what does the separating, so nobody
    later trims the probe grid down to level flight and destroys the evidence.
    """
    high_rate = [p for p in _probes() if np.linalg.norm(p.pqr) > 0.05]
    assert high_rate, "the probe grid no longer contains a high-body-rate case"

    for p in high_rate:
        t_b2e = (p.t_l2b @ p.t_e2l).T
        omega_earth_body = t_b2e.T @ np.array([0.0, 0.0, earth.OMEGA_WGS84])
        term = np.cross(p.pqr, omega_earth_body)
        assert np.allclose(p.pqridot + term, p.pqrdot, atol=1e-14)
        assert not np.allclose(p.pqridot - term, p.pqrdot, atol=1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `../../../.venv/Scripts/python.exe -m pytest atisim/tests/test_earth.py -q`
Expected: FAIL — `ImportError: cannot import name 'earth_acceleration_terms'`

- [ ] **Step 3: Write the implementation**

In `atisim/dynamics.py`, first fix the imports — the old ones will not resolve, because `quat_to_dcm` no longer exists:

```python
from atisim import earth
from atisim.state import Controls, State, quat_derivative, quat_to_matrix
```

`G0` is no longer used by `derivatives` or `specific_force`; leave the `atmosphere` import only if `thrust_authority` still needs it (it does).

Then add this function and rewrite `derivatives` around it:

```python
def earth_acceleration_terms(force, moment, mass, inertia, inertia_inv,
                             vel_body, omega, quat_body_to_ecef, r_ecef, model):
    """The rotating-Earth equations of motion, isolated so they can be tested alone.

    Established empirically from JSBSim v1.3.1, not transcribed -- the residuals
    are in the design doc section 2. `force` EXCLUDES gravity; gravity is added
    here so the caller cannot double-count it with the centrifugal term.

        vdot    = F/m + g - (omega_be + 2 Omega_b) x v - [Omega x (Omega x r)]_b
        wdot_be = I^-1 (M - omega_bi x I omega_bi) + omega_be x Omega_b

    THE PLUS IN THE SECOND LINE IS MEASURED, NOT DERIVED-AND-HOPED. A
    wings-level probe cannot distinguish it from a minus.
    """
    t_b2e = quat_to_matrix(quat_body_to_ecef)
    omega_ecef = jnp.array([0.0, 0.0, model.rotation_rate])
    omega_earth_body = t_b2e.T @ omega_ecef

    gravity_body = t_b2e.T @ earth.gravitation(r_ecef, model)
    centrifugal_body = t_b2e.T @ jnp.cross(omega_ecef, jnp.cross(omega_ecef, r_ecef))

    accel = (
        force / mass
        + gravity_body
        - jnp.cross(omega + 2.0 * omega_earth_body, vel_body)
        - centrifugal_body
    )

    omega_bi = omega + omega_earth_body
    omega_dot_bi = inertia_inv @ (moment - jnp.cross(omega_bi, inertia @ omega_bi))
    omega_dot_be = omega_dot_bi + jnp.cross(omega, omega_earth_body)
    return accel, omega_dot_be
```

Then rewrite `derivatives` to take `anchor` and `earth_model`, compute `dcm_b2n` from the aircraft's own position, feed `pos_ned` and geodetic altitude to the existing wind/aero path, and return the new `State`:

```python
def derivatives(state, controls, ac, wind_ned, omega_gust, anchor, earth_model,
                increment=None, alphadot_gust=0.0):
    t_b2e = quat_to_matrix(state.quat)
    r_ecef = anchor.r_ecef + state.pos_ecef
    lat, lon, h = earth.ecef_to_geodetic(r_ecef)
    dcm_b2n = earth.ecef_to_ned_matrix(lat, lon) @ t_b2e

    vel_rel = state.vel_body - dcm_b2n.T @ wind_ned
    omega_rel = state.omega - omega_gust

    rho = density(h)
    a_sound = speed_of_sound(h)
    mach = jnp.linalg.norm(vel_rel) / a_sound
    thrust = thrust_force(controls, ac, rho, mach)

    def accelerate(alphadot):
        f, m = aero_forces_moments(
            vel_rel, omega_rel, controls, ac, rho, a_sound,
            increment=increment, alphadot_gust=alphadot,
        )
        f = f + thrust
        a, wdot = earth_acceleration_terms(
            f, m, ac.mass, ac.inertia, ac.inertia_inv,
            state.vel_body, state.omega, state.quat, r_ecef, earth_model,
        )
        return f, m, a, wdot

    # The alphadot loop is unchanged; only what `accelerate` returns has grown.
    _, _, accel_open, _ = accelerate(alphadot_gust)
    u_rel, w_rel = vel_rel[0], vel_rel[2]
    denominator = jnp.maximum(u_rel**2 + w_rel**2, V_MIN**2)
    alphadot_aircraft = (u_rel * accel_open[2] - w_rel * accel_open[0]) / denominator

    _, _, accel, omega_dot = accelerate(alphadot_gust + alphadot_aircraft)

    return State(
        pos_ecef=t_b2e @ state.vel_body,
        vel_body=accel,
        quat=quat_derivative(state.quat, state.omega),
        omega=omega_dot,
    )
```

Update `specific_force` to take `anchor`/`earth_model` and to build `gravity_body` from `earth.gravitation` rather than `G0`, keeping its standing property of INVERTING the sum rather than recomputing it.

- [ ] **Step 4: Run test to verify it passes**

Run: `../../../.venv/Scripts/python.exe -m pytest atisim/tests/test_earth.py -q`
Expected: PASS — all EOM rows agree at 1e-9 (translational) and 1e-12 (rotational)

- [ ] **Step 5: Commit**

```bash
git add atisim/dynamics.py atisim/tests/test_earth.py
git commit -m "Put JSBSim's rotating-Earth equations of motion into derivatives"
```

---

## Task 10: Closed-form invariants (tier 0)

**Files:**
- Modify: `atisim/verification.py`
- Test: `atisim/tests/test_verification.py`

These exist because a comparison cannot catch an error the two engines **share**. A sign convention both get wrong is invisible to Task 9 and visible here.

- [ ] **Step 1: Write the failing test**

Append to `atisim/tests/test_verification.py`:

```python
def test_a_mass_at_rest_on_the_rotating_earth_feels_apparent_gravity():
    """The centrifugal and gravitational terms must combine correctly.

    A body at rest ON the rotating Earth has zero ECEF velocity, so Coriolis
    vanishes and only gravitation and centrifugal remain. Their sum is APPARENT
    gravity -- which is the quantity JSBSim does NOT report, because it reports
    gravitation. That is exactly why this check cannot be made against JSBSim.

    The two failure modes this catches, and how far off they land:

        centrifugal MISSING      9.81420 at the equator, 0.35% high
        centrifugal SIGN FLIPPED 9.84811 at the equator, 0.69% high

    Both are three orders above the 1e-9 tolerance below.
    """
    from atisim import earth, verification

    # These are the J2-TRUNCATED model's own values, measured. See the next
    # test for why they are not the WGS-84 normal-gravity constants.
    assert verification.apparent_gravity_at_rest(0.0, earth.WGS84_J2) == pytest.approx(
        9.7802816473, rel=1e-9
    )
    assert verification.apparent_gravity_at_rest(
        np.radians(90.0), earth.WGS84_J2
    ) == pytest.approx(9.8320668466, rel=1e-9)


def test_the_j2_truncation_costs_at_most_1_2e_5_against_exact_normal_gravity():
    """What truncating at J2 is worth, as a NUMBER rather than an omission.

    WGS-84 normal gravity is the Somigliana formula and carries J4 and J6;
    JSBSim truncates at J2 and so does this model, deliberately, because
    matching JSBSim is the point. The gap is therefore a real modelling error
    that no JSBSim comparison can see, and it belongs in ASSUMPTIONS.md rather
    than in a docstring.

    Measured: 4.5e-6 relative at the equator, 4.9e-6 at 45 degrees, 1.2e-5 at
    the pole. That is 1.2e-4 m/s^2 -- three orders below the 0.0035 g Coriolis
    term this whole change is about, so it does not threaten anything the
    project reports, and it is bounded rather than unbounded.
    """
    from atisim import earth, verification

    a, b, e2 = earth.A_WGS84, earth.B_WGS84, earth.E2_WGS84
    ge, gp = 9.7803253359, 9.8321849378        # WGS-84 normal gravity, defining
    k = (b * gp - a * ge) / (a * ge)

    worst = 0.0
    for lat_deg in (0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0):
        s2 = np.sin(np.radians(lat_deg)) ** 2
        somigliana = ge * (1.0 + k * s2) / np.sqrt(1.0 - e2 * s2)
        ours = verification.apparent_gravity_at_rest(np.radians(lat_deg), earth.WGS84_J2)
        worst = max(worst, abs(ours / somigliana - 1.0))

    assert worst < 1.3e-5, f"J2 truncation now costs {worst:.2e}, was 1.2e-5"


def test_switching_the_earth_off_recovers_the_non_rotating_equations():
    """FLAT must remove every Earth term, not merely shrink them."""
    from atisim import earth, verification

    assert verification.apparent_gravity_at_rest(np.radians(45.0), earth.FLAT) == pytest.approx(
        G0, rel=1e-14
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `../../../.venv/Scripts/python.exe -m pytest atisim/tests/test_verification.py -k rotating -q`
Expected: FAIL — `AttributeError: module 'atisim.verification' has no attribute 'apparent_gravity_at_rest'`

- [ ] **Step 3: Write the implementation**

Append to `atisim/verification.py`:

```python
def apparent_gravity_at_rest(lat, model):
    """|gravitation + centrifugal| for a body at rest on the ellipsoid, m/s^2.

    TIER 0: this takes no aircraft data. A failure here is a defect in the core.

    It is the check a JSBSim comparison cannot make. JSBSim reports GRAVITATION
    and carries the centrifugal term separately in its equations of motion, so
    a comparison against it is blind to the centrifugal term being absent or
    sign-flipped -- both engines would simply agree on the wrong thing.

    This is NOT held to the WGS-84 normal-gravity constants, deliberately. Those
    come from the Somigliana formula and carry J4 and J6; this model truncates
    at J2 because JSBSim does. The gap is 4.5e-6 to 1.2e-5 relative and is
    asserted separately as a bounded cost rather than hidden inside a loose
    tolerance here.
    """
    import jax.numpy as jnp

    from atisim import earth

    r = earth.geodetic_to_ecef(jnp.asarray(lat), 0.0, 0.0)
    omega = jnp.array([0.0, 0.0, model.rotation_rate])
    apparent = earth.gravitation(r, model) - jnp.cross(omega, jnp.cross(omega, r))
    return float(jnp.linalg.norm(apparent))
```

> **Sign note for the implementer.** Apparent gravity is `g_gravitation − Ω×(Ω×r)`, and the equation of motion in Task 9 carries that same term with that same sign. If the equator comes out at **9.84811** the sign is flipped; if it comes out at **9.81420** the term is missing entirely. Fix the sign, never the expected value.

- [ ] **Step 4: Run test to verify it passes**

Run: `../../../.venv/Scripts/python.exe -m pytest atisim/tests/test_verification.py -k "rotating or earth_off" -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add atisim/verification.py atisim/tests/test_verification.py
git commit -m "Add the closed-form gravity invariants a JSBSim comparison cannot make"
```

---

## Task 11: Trim on six unknowns

**Files:**
- Modify: `atisim/trim.py`
- Test: `atisim/tests/test_trim.py`

- [ ] **Step 1: Write the failing test**

Append to `atisim/tests/test_trim.py`:

```python
def test_trim_zeroes_all_six_residuals_on_a_rotating_earth():
    """Wings-level level flight is not an equilibrium once the Earth turns.

    The count is easy to get wrong and the design got it wrong once: six
    residuals against alpha, beta, phi, elevator, aileron, rudder, throttle is
    SEVEN freedoms for six equations, a one-parameter family. It is closed by
    imposing beta = 0.

    `aileron` is in the set and is not optional: with beta = 0 and a deflected
    rudder, Cl is non-zero through Cl_dr, so the roll residual cannot vanish
    without it.
    """
    from atisim import earth
    from atisim.aircraft import CRUISE, REGISTRY
    from atisim.trim import trim

    ac = REGISTRY["boeing747"]
    airspeed = CRUISE["boeing747"]["airspeed"]
    altitude = CRUISE["boeing747"]["altitude"]
    anchor = earth.anchor_at(np.radians(47.0), 0.0, altitude)

    solution, residual = trim(
        airspeed, altitude, ac, anchor, earth.WGS84_J2, heading=np.radians(30.0),
    )
    assert np.max(np.abs(np.asarray(residual))) < 1e-9, (
        f"six-residual trim did not converge: {np.asarray(residual)}"
    )


def test_the_trimmed_bank_angle_is_measured_not_assumed():
    """The design ESTIMATED 0.2 deg from atan(0.0035). This measures it.

    Recorded rather than asserted tightly, because the estimate was never a
    prediction. What IS asserted is the physics that must hold: the bank is
    non-zero on a rotating Earth, it is zero when the Earth is switched off, and
    it reverses sign in the opposite hemisphere.
    """
    from atisim import earth
    from atisim.aircraft import CRUISE, REGISTRY
    from atisim.trim import trim

    ac = REGISTRY["boeing747"]
    airspeed = CRUISE["boeing747"]["airspeed"]
    altitude = CRUISE["boeing747"]["altitude"]

    def bank(lat_deg, model):
        anchor = earth.anchor_at(np.radians(lat_deg), 0.0, altitude)
        x, _ = trim(airspeed, altitude, ac, anchor, model, heading=np.radians(90.0))
        return float(np.asarray(x)[3])

    north = bank(47.0, earth.WGS84_J2)
    south = bank(-47.0, earth.WGS84_J2)
    flat = bank(47.0, earth.FLAT)

    assert abs(north) > 1e-5, "no bank on a rotating Earth"
    assert abs(flat) < 1e-12, "bank survived switching the Earth off"
    assert np.sign(north) == -np.sign(south), "bank does not reverse across the equator"
    print(f"\ntrimmed bank: {np.degrees(north):.4f} deg at 47N, "
          f"{np.degrees(south):.4f} deg at 47S")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `../../../.venv/Scripts/python.exe -m pytest atisim/tests/test_trim.py -k six_residuals -q`
Expected: FAIL — `TypeError: trim() takes 3 positional arguments but 5 were given`

- [ ] **Step 3: Write the implementation**

Rewrite `atisim/trim.py`'s `trimmed_state`, `residual`, `trim` and `is_physical`:

```python
def trimmed_state(alpha, phi, airspeed, altitude, anchor, heading):
    """Level flight at the given alpha and bank, with beta = 0.

    The level-flight constraint is NOT theta = alpha any more. With beta = 0 and
    gamma = 0 it is tan(theta) = cos(phi) tan(alpha), which reduces to the old
    form at phi = 0. At the ~0.2 deg of bank this trim produces the two differ
    by well under a microradian, but the exact form is used because nothing
    downstream would reveal it if it were wrong.
    """
    theta = jnp.arctan(jnp.cos(phi) * jnp.tan(alpha))
    quat_ned = euler_to_quat(phi, theta, heading)
    vel_body = airspeed * jnp.array([jnp.cos(alpha), 0.0, jnp.sin(alpha)])
    return state_from_ned(
        jnp.array([0.0, 0.0, 0.0]), vel_body, quat_ned, jnp.zeros(3), anchor
    )


def residual(x, airspeed, altitude, ac, anchor, earth_model, heading):
    """All six: [udot, vdot, wdot, pdot, qdot, rdot]."""
    alpha, elevator, throttle, phi, aileron, rudder = x
    d = derivatives(
        trimmed_state(alpha, phi, airspeed, altitude, anchor, heading),
        Controls(elevator=elevator, aileron=aileron, rudder=rudder, throttle=throttle),
        ac, jnp.zeros(3), jnp.zeros(3), anchor, earth_model,
    )
    return jnp.concatenate([d.vel_body, d.omega])


INITIAL_GUESS = jnp.array([0.05, 0.0, 0.5, 0.0, 0.0, 0.0])


@partial(jax.jit, static_argnames=("iterations", "earth_model"))
def trim(airspeed, altitude, ac, anchor, earth_model, heading=0.0,
         guess=None, iterations=40):
    """Solve for [alpha, elevator, throttle, phi, aileron, rudder].

    Six unknowns, six residuals, square. `heading` matters now: the Coriolis
    term depends on which way the aircraft is pointing, so a trim is a function
    of latitude AND heading, which it was not before.
    """
    x0 = INITIAL_GUESS if guess is None else guess

    def step(x, _):
        r = residual(x, airspeed, altitude, ac, anchor, earth_model, heading)
        jacobian = jax.jacfwd(residual)(x, airspeed, altitude, ac, anchor, earth_model, heading)
        return x - jnp.linalg.solve(jacobian, r), None

    x, _ = jax.lax.scan(step, x0, None, length=iterations)
    return x, residual(x, airspeed, altitude, ac, anchor, earth_model, heading)


def is_physical(x, ac) -> bool:
    """As before, extended to the three new unknowns."""
    alpha, elevator, throttle, phi, aileron, rudder = (float(v) for v in x)
    return bool(
        abs(alpha) <= ALPHA_LIMIT
        and abs(elevator) <= float(ac.elevator_limit)
        and 0.0 <= throttle <= 1.0
        and abs(phi) <= BANK_LIMIT
        and abs(aileron) <= float(ac.aileron_limit)
        and abs(rudder) <= float(ac.rudder_limit)
    )
```

Add near `ALPHA_LIMIT`:

```python
# A steady-flight trim on a rotating Earth banks by a fraction of a degree. Five
# degrees is far beyond that and far below a turn, so it separates "trimmed" from
# "solved into a banked turn" without rejecting any legitimate solution.
BANK_LIMIT = math.radians(5.0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `../../../.venv/Scripts/python.exe -m pytest atisim/tests/test_trim.py -q -s`
Expected: PASS, and the printed trimmed bank angle recorded for the ledger

- [ ] **Step 5: Commit**

```bash
git add atisim/trim.py atisim/tests/test_trim.py
git commit -m "Trim six unknowns against six residuals, with beta = 0 closing the system"
```

---

## Task 12: Migrate the consumers

**Files:** `atisim/integrate.py`, `atisim/sensors.py`, `atisim/checks.py`, `atisim/wind.py`, `atisim/viz.py`, `atisim/panel.py`, `atisim/validation.py`, `atisim/vortex_viz.py`, `atisim/verification.py`, `atisim/loads.py`, and `scripts/*.py`

This is the wide task. Work file by file; the compiler is your worklist.

- [ ] **Step 1: Get the worklist**

Run:

```bash
../../../.venv/Scripts/python.exe -m pytest atisim/tests -q 2>&1 | grep -E "ImportError|AttributeError|TypeError" | sort -u
```

Every `cannot import name 'quat_to_dcm'` is a site that was silently assuming body→NED.

- [ ] **Step 2: Apply the mechanical rules**

| Old | New | Note |
|---|---|---|
| `quat_to_dcm(state.quat)` used as body→NED | `dcm_body_to_ned(state, anchor)` | the common case |
| `quat_to_dcm(q)` on a bare quaternion of unknown frame | `quat_to_matrix(q)` | only where the frame is genuinely generic |
| `-state.pos_ned[2]` as altitude | `altitude(state, anchor)` | **geodetic now** |
| `state.pos_ned` fed to a wind field | `pos_ned(state, anchor)` | wind fields themselves are unchanged |
| `State(pos_ned=..., quat=...)` | `state_from_ned(..., anchor)` | keeps call sites in the terms they already use |
| `quat_to_euler(state.quat)` | `quat_to_euler_ned(state, anchor)` | |

**`atisim/wind.py` needs no change to any field.** Its `-pos_ned[2]` at lines 219 and 625 are heights within the field's own local frame, which is correct and stays. Only the `quat_to_dcm` calls at 265, 299, 722 and 862 change, to `quat_to_matrix` where they act on a bare quaternion.

- [ ] **Step 3: Thread anchor and earth through `integrate.py`**

`step` and `rollout` gain `anchor` and `earth_model`; add `"earth_model"` to `static_argnames` (it is a hashable NamedTuple of str/float/bool). `SimState` is unchanged.

- [ ] **Step 4: Run the full suite**

Run: `../../../.venv/Scripts/python.exe -m pytest -q`
Expected: green. Baseline was `724 passed, 1 skipped`; the count will have grown by the new tests. **Any test that now fails on a NUMBER rather than an import is a real result — record it for Task 14, do not adjust the tolerance.**

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "Migrate every consumer to the anchor-aware NED accessors"
```

---

## Task 13: FLAT against the pre-change model, and the 737 floor

**Files:**
- Modify: `atisim/tests/test_earth.py`, `atisim/tests/test_jsbsim_737_layers.py`
- Modify: `docs/ASSUMPTIONS.md` (F4)

- [ ] **Step 1: Write the FLAT floor test**

Append to `atisim/tests/test_earth.py`:

```python
def test_flat_reproduces_the_pre_earth_model_to_a_measured_round_off_floor():
    """NOT a bit-identity test, and the difference is the point.

    Bit-identity is unreachable through an ECEF state: pos_ned is a rotation of
    an offset accumulated in ECEF, so the arithmetic differs even where the
    physics does not. The design doc section 8 says so and this is where the
    claim is actually cashed out.

    What CAN be asserted, and is:

      1. the divergence is small in absolute terms;
      2. it is ROUND-OFF, not physics -- it must not grow linearly with the run,
         which is what a real dynamical difference would do.

    (2) is the load-bearing half. A missing Earth term would look like a small
    number at 1 s and a large one at 20 s; round-off does not.

    The number this prints is what ASSUMPTIONS.md F4 records.
    """
    import jax
    import jax.numpy as jnp

    from atisim import earth as E
    from atisim import state as st
    from atisim.aircraft import REGISTRY
    from atisim.integrate import init_sim, logged_rollout
    from atisim.trim import trimmed_controls

    frozen = np.load(Path(__file__).parent / "data" / "pre_earth_trajectory.npz")
    ac = REGISTRY["boeing747"]
    dt, n_steps = float(frozen["dt"]), int(frozen["n_steps"])
    alpha, elevator, throttle = (float(v) for v in frozen["trim"])
    airspeed, alt = float(frozen["airspeed"]), float(frozen["altitude"])

    # Same initial condition, expressed in the new state. The anchor's latitude
    # is irrelevant under FLAT -- which is itself worth knowing, and the
    # hemisphere check in test_trim asserts it independently.
    anchor = E.anchor_at(0.0, 0.0, alt)
    start = st.state_from_ned(
        jnp.array([0.0, 0.0, -alt]),
        airspeed * jnp.array([np.cos(alpha), 0.0, np.sin(alpha)]),
        st.euler_to_quat(jnp.array(0.0), jnp.array(alpha), jnp.array(0.0)),
        jnp.zeros(3),
        anchor,
    )
    sim = init_sim(start, jax.random.PRNGKey(0))
    _, log = logged_rollout(
        sim, trimmed_controls(jnp.array(elevator), jnp.array(throttle)),
        dt, ac, n_steps, anchor=anchor, earth_model=E.FLAT,
    )

    ours = np.asarray(jax.vmap(lambda s: st.pos_ned(s, anchor))(log.state))
    theirs = frozen["pos_ned"]
    divergence = np.linalg.norm(ours - theirs, axis=1)

    early = float(divergence[n_steps // 10])     # 2 s
    late = float(divergence[-1])                 # 20 s
    print(f"\nFLAT vs pre-Earth: {early:.3e} m at 2 s, {late:.3e} m at 20 s")

    assert late < 1.0e-3, f"FLAT diverges by {late:.3e} m, which is not round-off"
    # Round-off accumulates far more slowly than a dynamical difference, which
    # would grow at least linearly -- a 10x longer run would then be 10x worse.
    assert late < 10.0 * max(early, 1e-12), (
        f"divergence grew from {early:.3e} to {late:.3e} m, which is the "
        "signature of a model difference rather than round-off"
    )
```

Add `from pathlib import Path` at the top of `test_earth.py`.

> **If this fails on the growth check**, an Earth term has survived into `FLAT` — check `model.rotation_rate` really is 0.0 and that `gravitation` takes the `"constant"` branch. Do not relax the factor of 10.

- [ ] **Step 2: Turn the 737 allowance into an assertion**

In `test_jsbsim_737_layers.py::test_layer4_what_survives_the_hold_is_two_residuals_and_no_more`, the transverse bound is currently set against a floor the docstring says AtiSim "cannot reproduce even in principle". Replace that clause and assert the floor is now **matched**, not merely allowed for.

- [ ] **Step 3: Run both**

Run: `../../../.venv/Scripts/python.exe -m pytest atisim/tests/test_earth.py atisim/tests/test_jsbsim_737_layers.py -q -s`
Expected: PASS, with the measured floor printed

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "Measure the FLAT round-off floor, and assert the 737 Earth-rotation floor"
```

---

## Task 14: Re-measure the ledger and rewrite the assumption register

**Files:** `docs/PROJECT.md`, `docs/ASSUMPTIONS.md`, `audit/INVENTORY.md`, `audit/NOTATION.md`

- [ ] **Step 1: Re-measure §4's evidence ledger**

Run the analysis scripts and record the new numbers. **Supersede rows, never delete them** — that is PROJECT.md's standing rule.

```bash
../../../.venv/Scripts/python.exe scripts/sanity.py
../../../.venv/Scripts/python.exe scripts/summary.py
../../../.venv/Scripts/python.exe scripts/jsbsim_report.py
```

- [ ] **Step 2: Rewrite the assumption register**

| Entry | Action |
|---|---|
| A1 flat, non-rotating Earth | **Retire.** Replace with what is now modelled and what is still not (no polar motion, no nutation, no time-varying gravity). |
| A2 constant gravity | **Retire.** J2 gravitation is now modelled; record what the change was worth against §4's old numbers. |
| A3 geopotential altitude | **Rewrite.** Geodetic altitude now exists, so state whether `atmosphere.py` is fed geometric or geopotential and what the residual difference is. |
| F4 round-off floor | **Re-measure** at the new position magnitudes, and record the FLAT floor from Task 13. |
| **New** | The anchor-relative position deviation from FGPropagate, with the 512x ulp measurement. |
| **New** | Trim is now a function of latitude and heading. |
| **New** | **Gravity is truncated at J2**, because JSBSim's is. Measured cost against exact WGS-84 normal gravity: 4.5e-6 relative at the equator, 1.2e-5 at the pole — 1.2e-4 m/s², three orders below the Coriolis term this change is about. No JSBSim comparison can see this, which is why it needs its own entry. |
| **New** | No polar motion, no nutation, no time-varying gravity. Unmeasured and stated as such: all three are far below J2 truncation over a flight, and none is modelled by JSBSim either. |

- [ ] **Step 3: Update the audit documents**

`audit/INVENTORY.md` R2 changes from "implicit, by omission" to an explicit cited model. `audit/NOTATION.md`'s frame table gains ECEF and the anchor.

- [ ] **Step 4: Full suite and notebook gate**

Run: `../../../.venv/Scripts/python.exe -m pytest -q`
Expected: green

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "Re-measure the evidence ledger on the rotating WGS-84 Earth"
```

---

## Self-review notes

**Spec coverage.** §2 formulation → Tasks 9, 10. §3 `earth.py` → Tasks 1–4. §4 state → Task 8. §5 dynamics → Task 9. §6 wind unchanged → Task 12 step 2. §7 trim → Task 11. §8 FLAT → Tasks 4, 13. §9 verification checks 1–4 → Tasks 7, 9, 10, 13. §10 costs → Task 14. §11–12 → the staging above.

**Task 0 exists because of an ordering bug in the spec.** §8 asks FLAT to be compared against "the pre-change model", but Task 8 destroys that model — once `State` is ECEF there is nothing left to compare against. The trajectory has to be frozen first, so Task 0 does it and everything else depends on it.

**Three API errors were caught by running against the real code rather than assuming:** the registry keys are `boeing747`, not `b747`; `CRUISE` values are plain dicts, so it is `CRUISE["boeing747"]["airspeed"]`, not `condition.airspeed`; and the J2-truncated apparent gravity is **not** the WGS-84 normal-gravity constant — the first version of Task 10 asserted `9.7803267715` at `rel=1e-6` and would have failed at 4.6e-6.

**Two places the plan corrects the spec:**

1. The spec says the blast radius is 43 files. Measured, only **3** sites outside wind fields derive altitude from position; the rest pass `pos_ned` into wind fields, which do not change. The real hazard is the 40+ `quat_to_dcm` sites, which the spec did not identify at all — Task 8 defuses it by deleting the name.
2. The spec's `dcm_body_to_ned` was implicitly at the anchor. It must be at the **aircraft's own position**, or the local vertical error goes straight into the Euler angles. Task 8 fixes this.

**Two numbers deliberately left unmeasured until execution**, flagged rather than invented: the trimmed bank angle (Task 11 prints it; the design's 0.2° is an estimate from `atan(0.0035)`, not a prediction) and the FLAT round-off floor (Task 13 measures it; the design's ~3.6e-8 m is F4 scaled by the ulp ratio).
