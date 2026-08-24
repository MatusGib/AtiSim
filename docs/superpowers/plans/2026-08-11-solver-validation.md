# Solver Validation Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish that the solver is arithmetically correct and that known coefficient changes produce known results, before any further modelling layer is added.

**Architecture:** Two new flat modules beside the existing ones — `atisim/verification.py` for checks that depend on no aircraft data at all, and `atisim/validation.py` for checks against analytic laws and published worked examples. A Jupyter notebook is a thin front end that imports both and plots; it contains no arithmetic, and is executed by `nbval` as a required gate so it cannot drift. Design spec: `docs/superpowers/specs/2026-08-11-solver-validation-design.md`.

**Tech Stack:** Python 3.10, JAX (float64 via `atisim/__init__.py`), NumPy, SciPy, pytest, Jupyter + nbval.

**Revision:** rev 2, after plan review. Every fix below is verified, not assumed — see "What review changed".

---

## What review changed

| Was | Is | Verified how |
|---|---|---|
| Task 6 ω₀ = (0.9, 0.05, 0.4) | **(0.6, 0.0, 0.9)** | ω₀ as planned put the motion on the wrong elliptic branch: L² = 5.330e6 < 2T·I₂ = 7.835e6 gives m = 5.5653, and `ellipj` returns **NaN** for m > 1 without `jax_debug_nans` seeing it (NumPy path). ω₂(0) = a₂·sn(0) = 0 identically, so a non-zero middle component was incompatible regardless. The replacement gives m = 0.4928, rate = 0.5782 rad/s, a₁ = 0.600000 and a₃ = 0.900000 exactly, and matches an independent RK4 of Euler's equations to **9.2e-15**. |
| Task 6 grid 601 points | **2001** | central-difference residual 9.27e-7 against `atol=1e-6` is an 8% margin and scales as h²; 2001 points gives 8.3e-8. |
| Task 4 wind [7, −3, **2**] | **[7, −3, 0]** | a vertical component drops the blown aircraft 40 m over 20 s; density is a function of altitude, so the forces differ and `atol=1e-11` cannot hold. |
| Task 2 `rk4_step_array` | **deleted** | `_axpy` maps over a pytree and an array is a leaf, so `integrate.rk4_step` runs unchanged on a plain array (confirmed: harmonic oscillator, 40 steps, error 1.04e-7). A separate copy would have tested a duplicate of the stage weights, defeating Task 1. |
| Task 1 self-referential test | **committed hash constant** | comparing `rk4_step` against an inline re-derivation of itself would pass on an extraction that changed the arithmetic. |
| Task 10 | **rewritten** | `ph_err < 0.0178` is weaker than Task 9's `rel=0.01`, so it asserted nothing new. Now computes both conditions and asserts the ratio. |
| §5 wording "excluded Xu, Zu, Mu" | **corrected** | the model *has* Xu and Zu — they fall out of dynamic-pressure variation, and Task 9's own A[0,0] within 1.2% of Caughey's Xu proves it. What is excluded is their **Mach content**, which is why the gap opens at M 0.8 and closes on approach. |

Also fixed: `to_imperial_matrix` missing two elements; A[2,1] reconstructed rather than given a loose tolerance; uncited constants moved into `REFERENCES`; no global `addopts`; `torque_free_period` dropped as unused; `sweep` used for all four sweeps instead of one; trim residual asserted per sweep sample.

---

## Conventions for every task

**Working directory** is the worktree root. **Every** command needs `PYTHONPATH=.` or the editable install resolves to the main checkout instead (PROJECT.md §10, "Environment notes"). The interpreter lives in the main checkout:

```bash
PYTHONPATH=. "C:/Users/mateusz/UROP/Claude_Flight_Sim/.venv/Scripts/python.exe" -m pytest atisim/tests -q
```

Shorthand below: `$PY` means `"C:/Users/mateusz/UROP/Claude_Flight_Sim/.venv/Scripts/python.exe"`.

**Two project rules that override normal practice:**

1. **Do not edit a tolerance to make a test pass.** If `test_conservation.py`, `test_cr2144_modes.py`, `test_drag_polar.py`, `test_navion.py` or `test_trim.py` moves, something real broke — stop and report.
2. **Flag, never invent.** Every reference number carries its table, in the code, as a string.

`conftest.py` enables `jax_debug_nans` for the suite. Note it guards **JAX** only: Task 6's reference solution runs through SciPy/NumPy and can produce NaN silently, which is why that task asserts its own domain.

---

## File Structure

| File | Responsibility |
|---|---|
| `atisim/integrate.py` (modify) | extract `rk4_step` so the stage weights can be tested on a problem with a closed-form solution |
| `atisim/trim.py` (modify) | expose the Newton start point as a module constant |
| `atisim/verification.py` (create) | tier 0: no aircraft data. Order of accuracy, Galilean invariance, torque-free rigid body, Newton convergence |
| `atisim/validation.py` (create) | tiers 1–2: plant matrices, axis transforms, published references, coefficient sweeps |
| `atisim/tests/modes.py` (modify) | becomes a re-export so `test_cr2144_modes.py` and `test_navion.py` are untouched |
| `atisim/tests/test_verification.py` (create) | tests for `verification.py` |
| `atisim/tests/test_validation.py` (create) | tests for `validation.py` |
| `notebooks/solver-validation.ipynb` (create) | narrative and figures only |
| `pyproject.toml` (modify) | add `jupyter`, `nbval` to the dev extra |
| `docs/PROJECT.md` (modify) | §3, §4, §5, §7, §9, §10 |

---

## Task 1: Extract the RK4 stage weights

`integrate.step` inlines RK4 around `derivatives`, so the stage weights cannot be exercised on a problem whose exact answer is known. Extract them. The refactor must change nothing, and the guard is a trajectory hash captured **before** the change and pinned as a constant afterwards.

**Files:**
- Modify: `atisim/integrate.py:57-94`
- Test: `atisim/tests/test_verification.py`

- [ ] **Step 1: Capture the pre-refactor hash**

```bash
PYTHONPATH=. $PY -c "
import jax, jax.numpy as jnp, numpy as np, hashlib
import atisim
from atisim import trim, integrate
from atisim.aircraft import REGISTRY, CRUISE
ac = REGISTRY['boeing747']
V, H = CRUISE['boeing747']['airspeed'], CRUISE['boeing747']['altitude']
x, _ = trim.trim(jnp.array(V), jnp.array(H), ac)
st = trim.trimmed_state(x[0], jnp.array(V), jnp.array(H))
ct = trim.trimmed_controls(x[1] + 0.02, x[2])
sim = integrate.init_sim(st, jax.random.PRNGKey(0))
_, traj = integrate.rollout(sim, ct, jnp.array(0.02), ac, 500)
print(hashlib.sha256(np.asarray(traj.vel_body).tobytes()).hexdigest())
"
```

**Write the printed hash down.** It goes into the test in Step 4 as a literal. This is the only thing in the task that can detect an extraction which changed the arithmetic — a test that re-derives RK4 inline and compares would be comparing the new code against itself.

- [ ] **Step 2: Extract `rk4_step`**

In `atisim/integrate.py`, insert after `_axpy` (line 59):

```python
def rk4_step(f, x, dt):
    """One classical RK4 stage set on a pytree state.

    Split out of `step` so the stage weights can be verified against a problem
    with a closed-form solution -- `step` is welded to `derivatives`, and
    conservation drift cannot distinguish a fourth-order scheme from a
    second-order one. `_axpy` maps over a pytree and an array is a leaf, so this
    also runs unchanged on a plain array right-hand side, which is what
    atisim/verification.py uses.
    """
    k1 = f(x)
    k2 = f(_axpy(x, k1, dt / 2))
    k3 = f(_axpy(x, k2, dt / 2))
    k4 = f(_axpy(x, k3, dt))
    increment = jax.tree.map(
        lambda a, b, c, d: (a + 2.0 * b + 2.0 * c + d) / 6.0, k1, k2, k3, k4
    )
    return _axpy(x, increment, dt)
```

Replace lines 76-85 of `step` (from `x = sim.state` through `new_state = _axpy(x, increment, dt)`) with:

```python
    new_state = rk4_step(f, sim.state, dt)
```

Leave the `quat_normalize` line and the `return SimState(...)` exactly as they are.

- [ ] **Step 3: Re-run the hash command from Step 1**

Expected: **the identical hash**. If it differs, revert and redo — the extraction changed the arithmetic.

- [ ] **Step 4: Pin the hash in a test**

Create `atisim/tests/test_verification.py`:

```python
"""Tier-0 verification: checks that depend on no aircraft data at all.

PROJECT.md section 4 is almost entirely validation -- a measured quantity against
a published one for one aircraft. These are the other kind: if one of them fails,
the arithmetic is wrong and no source can say otherwise.

Note that conftest.py's jax_debug_nans guards JAX only. Task 6's reference
solution goes through SciPy and asserts its own domain instead.
"""

import hashlib
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import atisim  # noqa: F401  -- enables x64 before any array is made
from atisim import integrate, trim, verification
from atisim.aircraft import CRUISE, REGISTRY

# Captured from the integrator BEFORE rk4_step was extracted from `step`
# (plan Task 1, Step 1). This is the whole guard on that refactor: a test that
# re-derives the stage weights inline would compare the new code against itself
# and pass on an extraction that changed the arithmetic.
PRE_REFACTOR_VEL_HASH = "PASTE_THE_HASH_FROM_STEP_1_HERE"


def _fixed_control_rollout(dt, n_steps, d_elevator=0.02):
    """747 at cruise trim with the elevator off trim, so something happens."""
    ac = REGISTRY["boeing747"]
    V, H = CRUISE["boeing747"]["airspeed"], CRUISE["boeing747"]["altitude"]
    x, _ = trim.trim(jnp.array(V), jnp.array(H), ac)
    state = trim.trimmed_state(x[0], jnp.array(V), jnp.array(H))
    controls = trim.trimmed_controls(x[1] + d_elevator, x[2])
    sim = integrate.init_sim(state, jax.random.PRNGKey(0))
    return integrate.rollout(sim, controls, jnp.array(dt), ac, n_steps)


def test_extracting_rk4_step_did_not_move_a_single_bit():
    """Bit-identity against a hash taken before the refactor.

    The same instrument PROJECT.md section 4 uses for the zero-wind path. A
    tolerance would not do: the claim is that the extraction was arithmetic
    neutral, and any tolerance admits an extraction that was not.
    """
    _, traj = _fixed_control_rollout(0.02, 500)
    got = hashlib.sha256(np.asarray(traj.vel_body).tobytes()).hexdigest()
    assert got == PRE_REFACTOR_VEL_HASH
```

Replace `PASTE_THE_HASH_FROM_STEP_1_HERE` with the actual hash.

- [ ] **Step 5: Run it**

```bash
PYTHONPATH=. $PY -m pytest atisim/tests/test_verification.py -q
```
Expected: 1 passed. (`verification` is imported but unused until Task 2; if that import fails, comment it out and restore it in Task 2.)

- [ ] **Step 6: Full suite — nothing may move**

```bash
PYTHONPATH=. $PY -m pytest atisim/tests -q
```
Expected: 296 passed, 1 skipped, plus the new test.

- [ ] **Step 7: Commit**

```bash
git add atisim/integrate.py atisim/tests/test_verification.py
git commit -m "Split the RK4 stage weights out of step, bit-identically"
```

---

## Task 2: Observed order of accuracy on a manufactured solution

**Files:**
- Create: `atisim/verification.py`
- Test: `atisim/tests/test_verification.py`

- [ ] **Step 1: Write the failing test**

Append to `atisim/tests/test_verification.py`:

```python
def test_rk4_is_fourth_order_on_a_problem_with_a_closed_form():
    """The single biggest gap in the project's evidence.

    PROJECT.md section 4 asserts angular momentum barely drifts. An integrator
    can conserve beautifully and still be second-order: drift measures a
    symmetry, not an order. Nothing asserts the order, so a mis-weighted stage
    would pass every existing test at dt = 0.02 and quietly degrade every result
    taken at a larger step.

    Harmonic oscillator, xdot = [[0, 1], [-1, 0]] x, exact solution a rotation.
    Calls integrate.rk4_step DIRECTLY -- not a copy of it -- which is the entire
    point of Task 1's extraction.
    """
    def f(x):
        return jnp.array([x[1], -x[0]])

    x0 = jnp.array([1.0, 0.0])
    t_end = 2.0

    def final(dt):
        x = x0
        for _ in range(int(round(t_end / dt))):
            x = integrate.rk4_step(f, x, dt)
        return np.asarray(x)

    exact = np.array([np.cos(t_end), -np.sin(t_end)])
    dts = np.array([0.2, 0.1, 0.05, 0.025])
    errors = np.array([np.linalg.norm(final(dt) - exact) for dt in dts])
    slope = verification.fitted_order(dts, errors)
    assert slope == pytest.approx(4.0, abs=0.05), f"observed order {slope}"
```

- [ ] **Step 2: Run it and watch it fail**

```bash
PYTHONPATH=. $PY -m pytest atisim/tests/test_verification.py -q -k fourth_order
```
Expected: FAIL — `ModuleNotFoundError: No module named 'atisim.verification'`.

- [ ] **Step 3: Create the module**

Create `atisim/verification.py`:

```python
"""Tier-0 verification: is the arithmetic right?

Nothing here takes an aircraft's published data as a reference. These checks ask
whether the integrator, the trim solve and the rigid-body equations are correct
as mathematics -- a different question from whether the aerodynamic coefficients
describe a real aeroplane, and the one that has to be settled first. A failure
here is a defect in the core.

The split is the standard verification/validation one (Roache; AIAA G-077).
PROJECT.md section 4 mixes them; the design spec says why separating them is most
of the value.
"""

import numpy as np


def fitted_order(dts, errors):
    """Observed order of accuracy: the slope of log(error) against log(dt).

    Least squares over the whole sequence rather than a two-point ratio, so one
    noisy refinement cannot carry the answer.
    """
    dts = np.asarray(dts, dtype=float)
    errors = np.asarray(errors, dtype=float)
    if np.any(errors <= 0.0):
        raise ValueError("an error is zero or negative; the sequence is saturated")
    slope, _ = np.polyfit(np.log(dts), np.log(errors), 1)
    return float(slope)
```

- [ ] **Step 4: Run it and watch it pass**

```bash
PYTHONPATH=. $PY -m pytest atisim/tests/test_verification.py -q
```
Expected: 2 passed. The slope is **3.99982** on this sequence, so `abs=0.05` has ample margin.

**If the slope is not 4.0, stop.** That is the finding this plan exists to look for. Report the measured slope; do not adjust the tolerance.

- [ ] **Step 5: Commit**

```bash
git add atisim/verification.py atisim/tests/test_verification.py
git commit -m "Assert RK4 is fourth-order, which nothing did before"
```

---

## Task 3: Observed order of accuracy on the real 6-DOF

The manufactured case isolates the stage weights. It cannot see a wind sample or control update applied at the wrong stage — the seam turbulence will lean on (PROJECT.md §2: "wind sampled once per step, held across the four stages").

**Files:**
- Test: `atisim/tests/test_verification.py`

- [ ] **Step 1: Write the test**

```python
def test_the_six_dof_rollout_is_fourth_order():
    """Same claim, through the real dynamics.

    The reference is generated at the smallest step rather than analytically, so
    it carries its own error. The fit uses only steps at least 16x the reference
    step, which puts the reference's error at 16^-4 = 1/65536 of the smallest
    error being fitted -- comfortably below the fit's own scatter.
    """
    t_end = 4.0
    dt_ref = 1.0 / 2048.0
    _, ref = _fixed_control_rollout(dt_ref, int(round(t_end / dt_ref)))
    ref_end = np.asarray(ref.pos_ned[-1])

    dts = np.array([1.0 / 16, 1.0 / 32, 1.0 / 64, 1.0 / 128])
    errors = []
    for dt in dts:
        _, traj = _fixed_control_rollout(float(dt), int(round(t_end / dt)))
        errors.append(np.linalg.norm(np.asarray(traj.pos_ned[-1]) - ref_end))

    slope = verification.fitted_order(dts, np.array(errors))
    assert slope == pytest.approx(4.0, abs=0.15), f"observed order {slope}, errors {errors}"
```

- [ ] **Step 2: Run it**

```bash
PYTHONPATH=. $PY -m pytest atisim/tests/test_verification.py -q -k six_dof
```
Expected: PASS. Each distinct `dt` recompiles the jitted `rollout` (`n_steps` is static), so this takes tens of seconds. Do not shorten the sequence to speed it up.

- [ ] **Step 3: Commit**

```bash
git add atisim/tests/test_verification.py
git commit -m "Assert the order of accuracy through the real dynamics too"
```

---

## Task 4: Galilean invariance under a uniform horizontal wind

**Files:**
- Test: `atisim/tests/test_verification.py`

- [ ] **Step 1: Write the test**

```python
def test_a_uniform_horizontal_wind_only_translates_the_trajectory():
    """Galilean invariance -- the assertion the zero-wind test cannot make.

    A uniform wind W is a change of inertial frame. Fly the same aircraft with
    its ground velocity offset by W and the aerodynamics see an identical
    relative flow, so attitude and body rates must be untouched and position must
    differ by exactly W*t.

    PROJECT.md section 2 names putting vel_rel into the Coriolis term as a
    classic gust-modelling error. This test catches exactly that, and it is
    invisible in still air, which is why every test in the project was blind to
    it until now.

    SCOPE, stated because it is narrower than it looks: the wind is STEADY, so
    its material derivative is zero and this test CANNOT detect the other error
    section 2 names -- a spurious -m dW/dt term. That needs a time-varying field
    and a different assertion than invariance; it is recorded as an open question
    rather than claimed here.

    The wind is HORIZONTAL. A vertical component would leave the blown aircraft
    at a different altitude (2 m/s over 20 s is 40 m), and density is a function
    of altitude, so the two aircraft would not see the same dynamic pressure and
    the invariance would not hold to any tolerance worth asserting.
    """
    W = jnp.array([7.0, -3.0, 0.0])  # m/s NED, horizontal by necessity

    def uniform_wind(wind_state, state, key, dt):
        return W, jnp.zeros(3), wind_state, key

    ac = REGISTRY["boeing747"]
    V, H = CRUISE["boeing747"]["airspeed"], CRUISE["boeing747"]["altitude"]
    x, _ = trim.trim(jnp.array(V), jnp.array(H), ac)
    state = trim.trimmed_state(x[0], jnp.array(V), jnp.array(H))
    controls = trim.trimmed_controls(x[1] + 0.01, x[2])

    from atisim.state import quat_to_dcm

    dcm = quat_to_dcm(state.quat)  # body -> NED
    shifted = state._replace(vel_body=state.vel_body + dcm.T @ W)

    dt, n = jnp.array(0.02), 1000
    _, still = integrate.rollout(
        integrate.init_sim(state, jax.random.PRNGKey(0)), controls, dt, ac, n
    )
    _, blown = integrate.rollout(
        integrate.init_sim(shifted, jax.random.PRNGKey(0)),
        controls, dt, ac, n, wind_model=uniform_wind,
    )

    np.testing.assert_allclose(np.asarray(blown.quat), np.asarray(still.quat), atol=1e-11)
    np.testing.assert_allclose(np.asarray(blown.omega), np.asarray(still.omega), atol=1e-11)

    t = np.arange(1, n + 1) * float(dt)
    expected = np.asarray(still.pos_ned) + t[:, None] * np.asarray(W)
    np.testing.assert_allclose(np.asarray(blown.pos_ned), expected, atol=1e-6)
```

- [ ] **Step 2: Run it**

```bash
PYTHONPATH=. $PY -m pytest atisim/tests/test_verification.py -q -k galilean or horizontal_wind
```

If `-k` with `or` is awkward in the shell, run the file.

Expected: PASS. If it **fails**, that is a real finding about the wind path — report it, do not work around it.

- [ ] **Step 3: Commit**

```bash
git add atisim/tests/test_verification.py
git commit -m "Assert Galilean invariance, which still air could never show"
```

---

## Task 5: Newton trim convergence is quadratic

**Files:**
- Modify: `atisim/trim.py:97`, `atisim/verification.py`
- Test: `atisim/tests/test_verification.py`

**Verified before writing this task:** `trim.trim`'s scan body (`atisim/trim.py:99-104`) is the undamped Newton update `x - solve(jacfwd(residual), residual)` — no damping, no least squares. So the history below reproduces the solver rather than a different algorithm.

- [ ] **Step 1: Expose the start point**

In `atisim/trim.py`, above `trim`, add:

```python
# The Newton start point. A module constant rather than a literal inside `trim`
# so that verification.py measures the convergence of the actual solver instead
# of a hand-copied guess that could drift away from it.
INITIAL_GUESS = jnp.array([0.05, 0.0, 0.5])
```

and change line 97 to:

```python
    x0 = INITIAL_GUESS if guess is None else guess
```

- [ ] **Step 2: Write the failing test**

```python
def test_the_trim_solve_converges_quadratically():
    """A Newton solve that merely converges may have a wrong Jacobian.

    Quadratic convergence -- the residual exponent roughly doubling each step --
    is the signature that jacfwd differentiates the same function the residual
    evaluates. A finite-difference or stale Jacobian still converges, linearly,
    and the final residual alone cannot tell them apart.
    """
    ac = REGISTRY["boeing747"]
    V, H = CRUISE["boeing747"]["airspeed"], CRUISE["boeing747"]["altitude"]
    history = verification.newton_residual_history(jnp.array(V), jnp.array(H), ac, 6)
    assert history[0] > 1.0, "the start point is already converged; pick a worse one"

    # takewhile, not a filter: the residual can dip below float64 resolution and
    # come back up, and a filter would splice non-contiguous iterations into a
    # sequence that was never actually walked.
    usable = np.array(list(itertools.takewhile(lambda r: r > 1e-13, history)))
    assert len(usable) >= 3, f"converged too fast to measure: {history}"

    e = np.log10(usable)
    ratios = (e[2:] - e[1:-1]) / (e[1:-1] - e[:-2])
    assert np.max(ratios) > 1.6, f"convergence looks linear, ratios {ratios}"
```

- [ ] **Step 3: Run it and watch it fail**

```bash
PYTHONPATH=. $PY -m pytest atisim/tests/test_verification.py -q -k quadratic
```
Expected: FAIL — no attribute `newton_residual_history`.

- [ ] **Step 4: Implement**

Append to `atisim/verification.py`:

```python
def newton_residual_history(airspeed, altitude, ac, iterations=6):
    """Residual norm after each Newton iteration, from trim.py's own start point.

    `trim.trim` runs a fixed iteration count inside `lax.scan` and returns only
    the final answer, so the convergence rate is not observable through it. This
    repeats the same update -- verified against trim.py:99-104 as the identical
    undamped Newton step -- and keeps every iterate.
    """
    import jax
    import jax.numpy as jnp

    from atisim.trim import INITIAL_GUESS, residual

    x = INITIAL_GUESS
    history = [float(jnp.linalg.norm(residual(x, airspeed, altitude, ac)))]
    for _ in range(iterations):
        r = residual(x, airspeed, altitude, ac)
        jacobian = jax.jacfwd(residual)(x, airspeed, altitude, ac)
        x = x - jnp.linalg.solve(jacobian, r)
        history.append(float(jnp.linalg.norm(residual(x, airspeed, altitude, ac))))
    return np.array(history)
```

- [ ] **Step 5: Run the file, then the whole suite**

```bash
PYTHONPATH=. $PY -m pytest atisim/tests/test_verification.py -q
PYTHONPATH=. $PY -m pytest atisim/tests/test_trim.py -q
```
Expected: both green. `test_trim.py` is a §4 baseline file and `trim.py` was just edited — if it moves, revert.

- [ ] **Step 6: Commit**

```bash
git add atisim/trim.py atisim/verification.py atisim/tests/test_verification.py
git commit -m "Assert the trim Newton solve converges quadratically"
```

---

## Task 6: Torque-free rigid body against the analytic solution

PROJECT.md §4 records angular-momentum drift of 5.7e-13. That says the integrator is conservative, not that it is right — a scheme can conserve H exactly and traverse the polhode at the wrong rate.

**Files:**
- Modify: `atisim/verification.py`
- Test: `atisim/tests/test_verification.py`

- [ ] **Step 1: Implement the closed form, with its domain asserted**

Append to `atisim/verification.py`:

```python
def torque_free_omega(I1, I2, I3, omega0, t):
    """Exact torque-free rotation of an asymmetric rigid body. Returns (3, len(t)).

    Landau & Lifshitz, *Mechanics*, section 37, for I1 < I2 < I3 with the rotation
    nearer the I3 axis. That branch requires L^2 > 2*T*I2; on the other side of
    the separatrix the elliptic modulus exceeds 1 and `scipy.special.ellipj`
    returns NaN. conftest.py's jax_debug_nans does NOT see that -- this path is
    NumPy -- so the domain is asserted here rather than left to be discovered.

    Note omega_2(0) = a2*sn(0) = 0 identically, so this branch can only represent
    an initial rate whose MIDDLE component is zero.
    """
    from scipy.special import ellipj

    I = np.array([I1, I2, I3], dtype=float)
    if not (I1 < I2 < I3):
        raise ValueError("this form assumes I1 < I2 < I3")
    w0 = np.asarray(omega0, dtype=float)
    if w0[1] != 0.0:
        raise ValueError("omega_2(0) is identically zero on this branch")
    L2 = float(np.sum((I * w0) ** 2))
    twoT = float(np.sum(I * w0**2))
    if L2 <= twoT * I2:
        raise ValueError(
            f"L^2 = {L2:.4e} <= 2T*I2 = {twoT * I2:.4e}: wrong side of the "
            "separatrix, the elliptic modulus would exceed 1"
        )

    a1 = np.sqrt((twoT * I3 - L2) / (I1 * (I3 - I1)))
    a2 = np.sqrt((twoT * I3 - L2) / (I2 * (I3 - I2)))
    a3 = np.sqrt((L2 - twoT * I1) / (I3 * (I3 - I1)))
    rate = np.sqrt((I3 - I2) * (L2 - twoT * I1) / (I1 * I2 * I3))
    m = ((I2 - I1) * (twoT * I3 - L2)) / ((I3 - I2) * (L2 - twoT * I1))

    sn, cn, dn, _ = ellipj(rate * np.asarray(t, dtype=float), m)
    return np.vstack([a1 * cn, a2 * sn, a3 * dn])
```

- [ ] **Step 2: Write both tests**

```python
# Principal-axis inertias, well separated so the elliptic modulus is not near 0
# or 1. Not an aircraft -- this test has no aircraft in it.
_I1, _I2, _I3 = 1420.0, 4070.0, 4780.0
# Middle component MUST be zero (see torque_free_omega). This pair gives
# m = 0.4928, rate = 0.5782 rad/s, and a1, a3 equal to 0.6 and 0.9 exactly.
_OMEGA0 = np.array([0.6, 0.0, 0.9])


def test_the_analytic_torque_free_solution_solves_eulers_equations():
    """Check the reference before using it as one.

    A mis-stated closed form would otherwise surface as a phantom integrator
    defect in the next test. 2001 points, not 601: the central-difference
    residual scales as h^2 and 601 gives 9.27e-7 against a 1e-6 tolerance, an 8%
    margin that would fail on any tightening.
    """
    t = np.linspace(0.0, 3.0, 2001)
    w = verification.torque_free_omega(_I1, _I2, _I3, _OMEGA0, t)
    I = np.array([_I1, _I2, _I3])[:, None]
    dwdt = np.gradient(w, t, axis=1)
    euler = -np.cross(w.T, (I * w).T).T / I
    np.testing.assert_allclose(dwdt[:, 5:-5], euler[:, 5:-5], atol=1e-6)


def test_the_analytic_solution_rejects_the_wrong_branch():
    """The domain guard, exercised. Without it this returns silent NaN."""
    with pytest.raises(ValueError, match="separatrix"):
        verification.torque_free_omega(
            _I1, _I2, _I3, np.array([0.9, 0.0, 0.05]), np.array([0.0, 1.0])
        )


def test_the_integrator_reproduces_torque_free_rotation():
    """Conservation is not correctness.

    Section 4 records angular-momentum drift of 5.7e-13 over 60,000 steps. This
    checks the trajectory instead of the invariant.
    """
    from atisim.aircraft import inertia_tensor
    from atisim.state import State, euler_to_quat
    from atisim.tests.conftest import make_test_aircraft

    inertia = inertia_tensor(_I1, _I2, _I3, 0.0)
    zeroed = dict(
        CL0=0.0, CLa=0.0, CLq=0.0, CLde=0.0, Cm0=0.0, Cma=0.0, Cmq=0.0, Cmde=0.0,
        CD0=0.0, CYb=0.0, CYp=0.0, CYr=0.0, CYdr=0.0, Clb=0.0, Clp=0.0, Clr=0.0,
        Clda=0.0, Cldr=0.0, Cnb=0.0, Cnp=0.0, Cnr=0.0, Cnda=0.0, Cndr=0.0,
        max_thrust=0.0,
    )
    ac = make_test_aircraft()._replace(
        inertia=inertia,
        inertia_inv=jnp.linalg.inv(inertia),
        **{k: jnp.array(v) for k, v in zeroed.items()},
    )
    state = State(
        pos_ned=jnp.array([0.0, 0.0, -3000.0]),
        vel_body=jnp.array([60.0, 0.0, 0.0]),
        quat=euler_to_quat(jnp.array(0.0), jnp.array(0.0), jnp.array(0.0)),
        omega=jnp.array(_OMEGA0),
    )
    controls = trim.trimmed_controls(jnp.array(0.0), jnp.array(0.0))
    dt, n = 0.002, 1500
    _, traj = integrate.rollout(
        integrate.init_sim(state, jax.random.PRNGKey(0)), controls, jnp.array(dt), ac, n
    )

    t = np.arange(1, n + 1) * dt
    exact = verification.torque_free_omega(_I1, _I2, _I3, _OMEGA0, t)
    np.testing.assert_allclose(np.asarray(traj.omega).T, exact, atol=1e-8)
```

- [ ] **Step 3: Run them**

```bash
PYTHONPATH=. $PY -m pytest atisim/tests/test_verification.py -q -k torque_free
```
Expected: 3 passed.

If the **first** fails, the closed form is mis-stated — fix `torque_free_omega`, not the integrator. If only the **third** fails, that is a genuine integrator finding: report it.

- [ ] **Step 4: Commit**

```bash
git add atisim/verification.py atisim/tests/test_verification.py
git commit -m "Check torque-free rotation against its closed form, branch guarded"
```

---

## Task 7: Promote the linearisation into the package

`atisim/tests/modes.py` holds the only linearisation in the project and lives inside the test package, so the notebook cannot import it and Family C cannot reach the plant matrix — only the modes derived from it.

**Files:**
- Create: `atisim/validation.py`
- Modify: `atisim/tests/modes.py`
- Test: `atisim/tests/test_validation.py`

- [ ] **Step 1: Create `atisim/validation.py`**

Move both functions from `atisim/tests/modes.py` verbatim — including the docstring reasoning about the `r*cos(phi)*tan(theta0)` term — and add a matrix layer beneath.

```python
"""Validation against analytic laws and published worked examples.

Tier 1 checks a coefficient sweep against a closed-form relation; tier 2 checks
the model against a source that published both its inputs and its own computed
outputs. Neither can be invalidated by a source's age: if CR-2144's derivatives
were 10% from the real aeroplane, this model must still reproduce CR-2144's own
transfer-function factors from CR-2144's own derivatives. See the design spec,
"Source qualification".

Tier 0 -- the checks needing no aircraft data at all -- is atisim/verification.py.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from atisim.aircraft import Aircraft
from atisim.dynamics import derivatives
from atisim.state import Controls, State, euler_to_quat
from atisim.units import FT2M


def longitudinal_matrix(ac, alpha, elevator, throttle, V, H):
    """Body-axis plant matrix in [u, w, q, theta], by jacfwd of the real dynamics.

    Body axes, so theta0 = alpha0 and w0 = V sin(alpha0) are both non-zero. Most
    textbook longitudinal matrices are quoted in STABILITY axes, where Theta0 = 0
    -- see `to_stability_axes`, which is what makes them comparable at all.
    """
    u0, w0 = V * np.cos(alpha), V * np.sin(alpha)
    controls = Controls(
        elevator=jnp.array(elevator), aileron=jnp.array(0.0),
        rudder=jnp.array(0.0), throttle=jnp.array(throttle),
    )

    def f(x):
        u, w, q, theta = x
        state = State(
            pos_ned=jnp.array([0.0, 0.0, -H]),
            vel_body=jnp.array([u, 0.0, w]),
            quat=euler_to_quat(jnp.array(0.0), theta, jnp.array(0.0)),
            omega=jnp.array([0.0, q, 0.0]),
        )
        d = derivatives(state, controls, ac, jnp.zeros(3), jnp.zeros(3))
        return jnp.array([d.vel_body[0], d.vel_body[2], d.omega[1], q])

    return np.asarray(jax.jacfwd(f)(jnp.array([u0, w0, 0.0, alpha])))


def to_stability_axes(A, alpha):
    """Rotate a [u, w, q, theta] plant matrix from body to stability axes.

    Stability axes are body axes turned by the trim angle of attack, so the
    perturbation velocities mix and q and theta do not. That is a similarity
    transform: every element moves, the eigenvalues do not. Both halves are
    asserted in test_validation.py.

    Needed because published matrices state Theta0 = 0, which is a stability-axis
    statement -- compared raw against a body-axis matrix the (2, 4) element reads
    -g sin(alpha0) against a published 0 and looks like a defect.
    """
    ca, sa = np.cos(alpha), np.sin(alpha)
    T = np.array([[ca, sa, 0.0, 0.0],
                  [-sa, ca, 0.0, 0.0],
                  [0.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0, 1.0]])
    return T @ A @ np.linalg.inv(T)


def to_imperial_matrix(A):
    """A [u, w, q, theta] plant matrix from SI into ft/s-rad units.

    The state mixes dimensions, so each element converts differently. The five
    rate elements (0,0) (0,1) (1,0) (1,1) (2,2) are 1/s and unchanged; (3,2) is
    dimensionless. The rest:

        A[0,2]  m/s     -> ft/s      (du_dot/dq)
        A[0,3]  m/s^2   -> ft/s^2    (du_dot/dtheta)
        A[1,2]  m/s     -> ft/s      (dw_dot/dq)
        A[1,3]  m/s^2   -> ft/s^2    (dw_dot/dtheta)
        A[2,0]  1/(m.s) -> 1/(ft.s)  (dq_dot/du)
        A[2,1]  1/(m.s) -> 1/(ft.s)  (dq_dot/dw)
    """
    out = np.array(A, dtype=float, copy=True)
    for i, j in ((0, 2), (0, 3), (1, 2), (1, 3)):
        out[i, j] /= FT2M
    for i, j in ((2, 0), (2, 1)):
        out[i, j] *= FT2M
    return out


def modes_from_matrix(A):
    """(wn, zeta) for every oscillatory root, sorted low-to-high wn."""
    eig = np.linalg.eigvals(A)
    return sorted((abs(lam), -lam.real / abs(lam)) for lam in eig if lam.imag > 1e-9)


def longitudinal_modes(ac, alpha, elevator, throttle, V, H):
    """Phugoid and short-period (wn, zeta), sorted low-to-high wn."""
    return modes_from_matrix(longitudinal_matrix(ac, alpha, elevator, throttle, V, H))
```

Then append `lateral_modes` from `atisim/tests/modes.py`, **unchanged**, docstring included.

- [ ] **Step 2: Replace `atisim/tests/modes.py` with a re-export**

```python
"""Kept as an import shim.

The linearisation moved to `atisim.validation` so the notebook and the
validation module can reach the plant matrix itself, not only the modes derived
from it. `test_cr2144_modes.py` and `test_navion.py` import from here and are
deliberately untouched.
"""

from atisim.validation import lateral_modes, longitudinal_modes  # noqa: F401
```

- [ ] **Step 3: Run the WHOLE suite**

```bash
PYTHONPATH=. $PY -m pytest atisim/tests -q
```

Not just the two obvious files. This is the change most likely to perturb the suite, and `test_cr2144_modes.py` / `test_navion.py` are §4 baseline files: if any value moves, revert and find out why.

- [ ] **Step 4: Write the similarity-transform test**

Create `atisim/tests/test_validation.py`:

```python
"""Tier 1 and 2 validation: analytic laws, and published worked examples."""

import jax.numpy as jnp
import numpy as np
import pytest

import atisim  # noqa: F401
from atisim import trim, validation
from atisim.aircraft import CRUISE, REGISTRY
from atisim.units import FT2M

# Caughey Eq. (5.48): M = 0.25 at sea level. CR-2144 Table IX-2's header says
# 165 KTAS = 278.49 ft/s, a 0.2% difference recorded in the design spec. Checks
# against Caughey run at Caughey's speed.
CAUGHEY_V = 279.1 * FT2M


def _approach_trim(V=CAUGHEY_V):
    ac = REGISTRY["boeing747_approach"]
    x, res = trim.trim(jnp.array(V), jnp.array(0.0), ac)
    return ac, float(x[0]), float(x[1]), float(x[2]), res


def _approach_A(imperial=True):
    ac, alpha, de, thr, _ = _approach_trim()
    A = validation.to_stability_axes(
        validation.longitudinal_matrix(ac, alpha, de, thr, CAUGHEY_V, 0.0), alpha
    )
    return validation.to_imperial_matrix(A) if imperial else A


def test_the_stability_axis_transform_is_a_similarity_transform():
    """Every element moves; no eigenvalue does.

    This is what licenses comparing a body-axis matrix against a published
    stability-axis one, so it is asserted before it is used.
    """
    ac, alpha, de, thr, _ = _approach_trim()
    A_body = validation.longitudinal_matrix(ac, alpha, de, thr, CAUGHEY_V, 0.0)
    A_stab = validation.to_stability_axes(A_body, alpha)

    assert not np.allclose(A_body, A_stab, atol=1e-6), "the transform did nothing"
    np.testing.assert_allclose(
        np.sort_complex(np.linalg.eigvals(A_body)),
        np.sort_complex(np.linalg.eigvals(A_stab)),
        atol=1e-8,
    )
```

- [ ] **Step 5: Run it**

```bash
PYTHONPATH=. $PY -m pytest atisim/tests/test_validation.py -q
```
Expected: 1 passed.

- [ ] **Step 6: Commit**

```bash
git add atisim/validation.py atisim/tests/modes.py atisim/tests/test_validation.py
git commit -m "Promote the linearisation out of the test package, and add the axis transform"
```

---

## Task 8: The published reference table

Everything a test compares against lives here with its citation. Nothing is a bare literal in a test file.

**Files:**
- Modify: `atisim/validation.py`
- Test: `atisim/tests/test_validation.py`

- [ ] **Step 1: Write the failing test**

```python
def test_every_reference_carries_its_citation():
    """PROJECT.md's first rule, enforced mechanically.

    'Flag, never invent. Every number carries the table it came from.' A
    reference without a source string is exactly the failure that rule exists to
    prevent, and it is cheap to make impossible.
    """
    assert validation.REFERENCES, "the reference table is empty"
    for name, ref in validation.REFERENCES.items():
        assert ref.source and len(ref.source) > 20, f"{name} has no usable citation"
        assert np.isfinite(ref.value), f"{name} has a non-finite value"
```

- [ ] **Step 2: Run it and watch it fail**

```bash
PYTHONPATH=. $PY -m pytest atisim/tests/test_validation.py -q -k citation
```
Expected: FAIL — no attribute `REFERENCES`.

- [ ] **Step 3: Implement**

Append to `atisim/validation.py`:

```python
class Reference(NamedTuple):
    """A published number and the document it came from.

    `source` is not optional and not a comment. PROJECT.md's standing rule is
    that a figure without its table has broken the project, and a NamedTuple
    field is the cheapest way to make an uncited number unconstructible.
    """

    value: float
    source: str


# D. A. Caughey, "Introduction to Aircraft Stability and Control", Cornell
# MAE 5070 course notes, Chapter 5. Its Eq. (5.48)-(5.50) cite Heffley & Jewell,
# NASA CR-2144 -- the same document this project transcribed for
# `boeing747_approach` -- so Caughey is NOT an independent data source. He is an
# independent IMPLEMENTATION: he formed the dimensional derivatives, the plant
# matrix, the characteristic polynomial and the roots himself and published all
# four. For checking a solver, that is the useful kind of independence.
_C = "Caughey, Cornell MAE 5070 notes, Ch. 5, {}"
_IX2 = "NASA CR-2144, Heffley & Jewell 1972, Table IX-2 (747 power approach)"
_IX5 = "NASA CR-2144 Table IX-5 via PROJECT.md section 4, '747 modes vs CR-2144'"

REFERENCES = {
    # Eq. (5.54): the roots of the power-approach longitudinal system.
    "747pa_short_period_wn": Reference(0.88178, _C.format("Eq. (5.54)")),
    "747pa_short_period_zeta": Reference(0.62546, _C.format("Eq. (5.54)")),
    "747pa_phugoid_wn": Reference(0.13391, _C.format("Eq. (5.54)")),
    "747pa_phugoid_zeta": Reference(0.01329, _C.format("Eq. (5.54)")),
    # Eq. (5.51): dimensional derivatives, ft-s-rad. Mq is the one this model can
    # be expected to match exactly -- it has no alpha-dot content. Zwdot and Mwdot
    # are the two this model excludes by design, and are what the residuals in
    # test_the_omitted_alpha_dot_terms_are_recoverable are reconstructed from.
    # Mwdot is quoted to ONE significant figure, which bounds how well any
    # reconstruction using it can do -- see that test's tolerances.
    "747pa_Mq": Reference(-0.4381, _C.format("Eq. (5.51)")),
    "747pa_Zwdot": Reference(-0.0341, _C.format("Eq. (5.51)")),
    "747pa_Mwdot": Reference(-0.0002, _C.format("Eq. (5.51)")),
    # Table IX-2 as transcribed into aircraft.boeing747_approach.
    "747pa_CL": Reference(1.11, _IX2),
    "747pa_CD": Reference(0.102, _IX2),
    "747pa_Cma": Reference(-1.26, _IX2),
    # The CRUISE comparison, for the condition-dependence test. Model and
    # reference both, so the error can be recomputed rather than hard-coded.
    "747cruise_phugoid_wn_ref": Reference(0.0673, _IX5),
    "747cruise_phugoid_wn_model": Reference(0.0553, _IX5),
    "747cruise_short_period_zeta_ref": Reference(0.387, _IX5),
    "747cruise_short_period_zeta_model": Reference(0.3425, _IX5),
}

# Caughey Eq. (5.52), the [u, w, q, theta] plant matrix in ft/s and radians.
# Stability axes: he states Theta0 = 0, which is only true there.
CAUGHEY_A = np.array([
    [-0.0212,  0.0466,   0.000,   -32.174],
    [-0.2229, -0.5839, 262.472,     0.0],
    [ 0.0001, -0.0018,  -0.5015,    0.0],
    [ 0.0,     0.0,      1.0,       0.0],
])
```

- [ ] **Step 4: Run it and watch it pass**

```bash
PYTHONPATH=. $PY -m pytest atisim/tests/test_validation.py -q
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add atisim/validation.py atisim/tests/test_validation.py
git commit -m "Add the published reference table, citations enforced by a test"
```

---

## Task 9: Reproduce Caughey's plant matrix

Three tests with **disjoint** responsibilities: elements the model contains, elements it does not, and the roots. No element is asserted twice.

**Files:**
- Test: `atisim/tests/test_validation.py`

- [ ] **Step 1: Write the tests**

```python
def test_the_plant_matrix_matches_caugheys_where_the_model_has_the_terms():
    """Tier 2: same source data, an outside implementation, published intermediates.

    Covers ONLY the elements carrying no alpha-dot content. A[1,1], A[1,2], A[2,1]
    and A[2,2] all do, and are the next test's business -- asserting them here too
    would give the same element two homes and two different tolerances.

    Measured during planning, after the stability-axis rotation:
        A[0,0] Xu      -0.02094 vs -0.02120   1.2%
        A[0,1] Xw       0.04632 vs  0.04660   0.6%
        A[1,0] Zu      -0.22851 vs -0.22290   2.5%
    """
    _, _, _, _, res = _approach_trim()
    assert float(jnp.linalg.norm(res)) < 1e-10
    A, C = _approach_A(), validation.CAUGHEY_A

    assert A[0, 0] == pytest.approx(C[0, 0], rel=0.02)
    assert A[0, 1] == pytest.approx(C[0, 1], rel=0.01)
    assert A[1, 0] == pytest.approx(C[1, 0], rel=0.03)

    # Kinematic and gravity elements are structural, so they are exact.
    assert A[0, 3] == pytest.approx(C[0, 3], rel=1e-4)
    assert A[1, 3] == pytest.approx(0.0, abs=1e-9)
    assert A[3, 2] == pytest.approx(1.0, abs=1e-12)


def test_the_omitted_alpha_dot_terms_are_recoverable():
    """Section 5's attribution, turned from a claim into arithmetic.

    Every element that disagrees by more than 3% carries a derivative this model
    excludes by design (aero.py is alpha/q/de only). Caughey's Eq. (5.44) gives
    the exact form of each:

        A[1,1] = Zw / (1 - Zwdot)
        A[1,2] = (u0 + Zq) / (1 - Zwdot)
        A[2,1] = Mw + Mwdot * Zw / (1 - Zwdot)  ->  A_model[2,1] + Mwdot * C[1,1]
        A[2,2] = Mq + (u0 + Zq) * Mwdot / (1 - Zwdot)  ->  A_model[2,2] + Mwdot * C[1,2]

    Restoring them from Caughey's own tabulated CL_alphadot = 6.7 and
    Cm_alphadot = -3.2 recovers his published values. That is a much stronger
    statement than "attributed": the residual is reconstructed, not just explained.

    Tolerances on the two M-row rows are looser because Mwdot is published to ONE
    significant figure (-0.0002), which bounds the reconstruction independently of
    anything this model does.
    """
    A, C = _approach_A(), validation.CAUGHEY_A
    Zwdot = validation.REFERENCES["747pa_Zwdot"].value
    Mwdot = validation.REFERENCES["747pa_Mwdot"].value

    assert A[1, 1] / (1.0 - Zwdot) == pytest.approx(C[1, 1], rel=0.005)
    assert A[1, 2] / (1.0 - Zwdot) == pytest.approx(C[1, 2], rel=0.005)

    # This model's A[2,2] IS Caughey's raw Eq. (5.51) Mq, before his alpha-dot term.
    assert A[2, 2] == pytest.approx(validation.REFERENCES["747pa_Mq"].value, rel=0.01)

    assert A[2, 1] + Mwdot * C[1, 1] == pytest.approx(C[2, 1], rel=0.05)
    assert A[2, 2] + Mwdot * C[1, 2] == pytest.approx(C[2, 2], rel=0.05)


def test_the_approach_modes_match_caugheys_published_roots():
    """The end-to-end statement: units, trim, dynamics and jacfwd in four numbers."""
    ac, alpha, de, thr, _ = _approach_trim()
    (ph_wn, ph_z), (sp_wn, sp_z) = validation.longitudinal_modes(
        ac, alpha, de, thr, CAUGHEY_V, 0.0
    )
    R = validation.REFERENCES
    assert ph_wn == pytest.approx(R["747pa_phugoid_wn"].value, rel=0.01)
    assert ph_z == pytest.approx(R["747pa_phugoid_zeta"].value, rel=0.05)
    assert sp_wn == pytest.approx(R["747pa_short_period_wn"].value, rel=0.02)
    assert sp_z == pytest.approx(R["747pa_short_period_zeta"].value, rel=0.06)
```

- [ ] **Step 2: Run them**

```bash
PYTHONPATH=. $PY -m pytest atisim/tests/test_validation.py -q
```
Expected: 5 passed. Any tolerance that fails is a finding — report the measured value, do not widen.

- [ ] **Step 3: Commit**

```bash
git add atisim/tests/test_validation.py
git commit -m "Reproduce Caughey's plant matrix, and reconstruct the omitted terms"
```

---

## Task 10: The gap is condition-dependent

Rewritten after review: the previous version asserted `ph_err < 0.0178`, which Task 9's `rel=0.01` already implies, so it added nothing. This computes **both** conditions and asserts their ratio, which is the actual claim.

**Files:**
- Test: `atisim/tests/test_validation.py`

- [ ] **Step 1: Write the test**

```python
def test_the_mode_error_is_an_order_of_magnitude_smaller_on_approach():
    """The finding that reframes section 5.

    Section 4 records the cruise phugoid at 17.8% and short-period zeta at 11.5%
    from CR-2144, attributed to derivatives the aero form excludes. The same code
    with the same exclusions reads 0.4% and 5.5% at the power-approach point. So
    the gap is not a fixed modelling deficit -- it is CONDITION-DEPENDENT, and it
    bites at M 0.8 / 40,000 ft where compressibility drives the Mach content of
    Xu, Zu and Mu.

    Asserted as a RATIO of the two errors rather than a bound on one of them: a
    bound on the approach error alone is already implied by the tolerances in
    test_the_approach_modes_match_caugheys_published_roots, so it would assert
    nothing new.
    """
    R = validation.REFERENCES
    ac, alpha, de, thr, _ = _approach_trim()
    (ph_wn, _), (_, sp_z) = validation.longitudinal_modes(
        ac, alpha, de, thr, CAUGHEY_V, 0.0
    )

    def rel_err(model, ref):
        return abs(model - ref) / abs(ref)

    approach_ph = rel_err(ph_wn, R["747pa_phugoid_wn"].value)
    approach_sp = rel_err(sp_z, R["747pa_short_period_zeta"].value)
    cruise_ph = rel_err(
        R["747cruise_phugoid_wn_model"].value, R["747cruise_phugoid_wn_ref"].value
    )
    cruise_sp = rel_err(
        R["747cruise_short_period_zeta_model"].value,
        R["747cruise_short_period_zeta_ref"].value,
    )

    assert cruise_ph / approach_ph > 10.0, (
        f"phugoid: cruise {cruise_ph:.4f} vs approach {approach_ph:.4f}"
    )
    assert cruise_sp / approach_sp > 1.5, (
        f"short-period zeta: cruise {cruise_sp:.4f} vs approach {approach_sp:.4f}"
    )
```

- [ ] **Step 2: Run it**

```bash
PYTHONPATH=. $PY -m pytest atisim/tests/test_validation.py -q -k condition
```
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add atisim/tests/test_validation.py
git commit -m "Pin the phugoid gap as condition-dependent, as a ratio"
```

---

## Task 11: The sweep helper

One helper, used by every sweep. It re-trims per sample and asserts convergence, because a sweep that silently fails to trim reports modes for an aircraft that is not in equilibrium.

**Files:**
- Modify: `atisim/validation.py`
- Test: `atisim/tests/test_validation.py`

- [ ] **Step 1: Implement**

Append to `atisim/validation.py`:

```python
TRIM_RESIDUAL_LIMIT = 1e-9


def sweep(ac, field, values, quantity, V, H):
    """Vary one coefficient and report a scalar per value.

    `Aircraft` is a NamedTuple, so `_replace` gives an independent airframe per
    sample with no mutation and no copy discipline to get wrong.

    The aircraft is RE-TRIMMED at every sample, because changing a derivative
    moves the trim point and comparing modes across different trims would
    confound the two. The residual is asserted per sample rather than assumed:
    3x CD0 and a near-neutral Cma are exactly where the Newton solve stops
    converging, and an unconverged trim yields plausible-looking nonsense.

    `quantity` takes (aircraft, alpha, elevator, throttle) and returns a float.
    """
    from atisim.trim import trim as solve_trim

    out = []
    for v in values:
        swept = ac._replace(**{field: jnp.array(float(v))})
        x, res = solve_trim(jnp.array(V), jnp.array(H), swept)
        residual_norm = float(jnp.linalg.norm(res))
        if residual_norm > TRIM_RESIDUAL_LIMIT:
            raise RuntimeError(
                f"{field}={v} did not trim: residual {residual_norm:.3e}"
            )
        out.append(quantity(swept, float(x[0]), float(x[1]), float(x[2])))
    return np.array(out)
```

- [ ] **Step 2: Write its test**

```python
def test_the_sweep_helper_refuses_an_unconverged_trim():
    """A sweep that silently fails to trim reports modes for an aircraft that is
    not in equilibrium, which looks like a physics finding and is not one."""
    ac, _, _, _, _ = _approach_trim()
    with pytest.raises(RuntimeError, match="did not trim"):
        # CL_alpha near zero cannot hold level flight at any angle of attack.
        validation.sweep(
            ac, "CLa", [1e-4], lambda a, al, de, th: 0.0, CAUGHEY_V, 0.0
        )
```

- [ ] **Step 3: Run it**

```bash
PYTHONPATH=. $PY -m pytest atisim/tests/test_validation.py -q -k unconverged
```
Expected: PASS. If the solve happens to converge for `CLa=1e-4`, pick a value that does not and say which in the docstring — the point is that the guard fires, not the particular number.

- [ ] **Step 4: Commit**

```bash
git add atisim/validation.py atisim/tests/test_validation.py
git commit -m "Add the sweep helper, with per-sample trim convergence asserted"
```

---

## Task 12: Coefficient sweeps against the analytic laws

The review request read literally: change a coefficient, get a known result. The Lanchester rows are strongest because the size of the approximation's **error** is itself published.

**Files:**
- Test: `atisim/tests/test_validation.py`

- [ ] **Step 1: Write the Lanchester tests**

```python
def _sp_wn(a, alpha, de, thr):
    return validation.longitudinal_modes(a, alpha, de, thr, CAUGHEY_V, 0.0)[-1][0]


def _ph_zeta(a, alpha, de, thr):
    return validation.longitudinal_modes(a, alpha, de, thr, CAUGHEY_V, 0.0)[0][1]


def test_the_phugoid_frequency_follows_the_lanchester_law():
    """Tier 1: a relation with no aerodynamic coefficient in it at all.

    Lanchester's phugoid approximation is wn = sqrt(2) g / u0 -- no derivative, no
    area, no mass. It cannot be satisfied by accident, and no source's vintage can
    affect it.

    Caughey states his approximate analysis "over predicts the undamped natural
    frequency by about 20 per cent". Measured: 0.163028 against 0.13391, a ratio
    of 1.2174. Matching the published SIZE of the approximation's error is far
    tighter than matching its trend.
    """
    from atisim.atmosphere import G0

    ac, alpha, de, thr, _ = _approach_trim()
    (ph_wn, _), _ = validation.longitudinal_modes(ac, alpha, de, thr, CAUGHEY_V, 0.0)
    lanchester = np.sqrt(2.0) * float(G0) / CAUGHEY_V
    assert lanchester / ph_wn == pytest.approx(1.22, rel=0.03)


def test_the_phugoid_damping_follows_the_lift_to_drag_law():
    """Lanchester again: zeta = 1 / (sqrt(2) L/D).

    Caughey states the approximation "over predicts the damping ratio by a factor
    of almost 5". Measured: 0.0651 against 0.01329, a ratio of 4.9.
    """
    ac, alpha, de, thr, _ = _approach_trim()
    (_, ph_zeta), _ = validation.longitudinal_modes(ac, alpha, de, thr, CAUGHEY_V, 0.0)
    R = validation.REFERENCES
    L_over_D = R["747pa_CL"].value / R["747pa_CD"].value
    lanchester = 1.0 / (np.sqrt(2.0) * L_over_D)
    assert lanchester / ph_zeta == pytest.approx(4.9, rel=0.06)


def test_worsening_the_drag_polar_damps_the_phugoid_as_the_law_says():
    """The sweep itself: change one coefficient, get the predicted change.

    zeta_phugoid is inversely proportional to L/D, so raising CD0 must raise it,
    and the product zeta * (L/D) must stay put across the sweep even though
    neither factor does.
    """
    ac, _, _, _, _ = _approach_trim()
    R = validation.REFERENCES
    base = float(ac.CD0)
    factors = np.array([1.0, 1.5, 2.0, 3.0])

    zetas = validation.sweep(
        ac, "CD0", base * factors, _ph_zeta, CAUGHEY_V, 0.0
    )
    CL = R["747pa_CL"].value
    l_over_ds = np.array([
        CL / (base * k + CL**2 / (np.pi * float(ac.e) * float(ac.AR)))
        for k in factors
    ])

    assert np.all(np.diff(zetas) > 0), "more drag must damp the phugoid"
    product = zetas * l_over_ds
    assert product.max() / product.min() < 1.15, (
        f"zeta * L/D should be near constant, got {product}"
    )
```

- [ ] **Step 2: Write the remaining sweeps**

```python
def test_reducing_pitch_stiffness_lowers_the_short_period_frequency():
    """The neutral point, approached from the stable side.

    Short-period wn goes as sqrt(-M_alpha), so driving Cm_alpha towards zero must
    drive wn towards zero. A model that does not lose its short period near the
    neutral point does not have a short period, it has a curve fit.
    """
    ac, _, _, _, _ = _approach_trim()
    base = validation.REFERENCES["747pa_Cma"].value
    cmas = np.array([base, -0.9, -0.6, -0.3, -0.1])

    wns = validation.sweep(ac, "Cma", cmas, _sp_wn, CAUGHEY_V, 0.0)
    assert np.all(np.diff(wns) < 0), f"wn must fall as Cma -> 0, got {wns}"
    assert wns[0] / wns[-1] == pytest.approx(np.sqrt(base / -0.1), rel=0.35)


def test_the_roll_time_constant_tracks_one_over_Clp():
    """tau_roll = -1/L_p, and L_p is proportional to Clp, so tau * |Clp| is flat."""
    ac, _, _, _, _ = _approach_trim()
    clps = np.array([-0.30, -0.45, -0.60, -0.75])

    def roll_tau(a, alpha, de, thr):
        return validation.lateral_modes(a, alpha, de, thr, CAUGHEY_V, 0.0)[1]

    taus = validation.sweep(ac, "Clp", clps, roll_tau, CAUGHEY_V, 0.0)
    product = taus * np.abs(clps)
    assert product.max() / product.min() < 1.05, (
        f"tau_roll * |Clp| should be constant, got {product}"
    )


def test_the_dutch_roll_frequency_rises_with_weathercock_stability():
    """wn_dutch is dominated by sqrt(N_beta), but NOT proportional to it.

    The full expression carries a Y_beta/u0 term that does not scale with Cnb, so
    wn/sqrt(Cnb) is not flat and asserting that it is would be asserting the
    approximation rather than the model. What IS reliable is the ordering and the
    rough magnitude: quadrupling Cnb should roughly double wn, falling somewhat
    short of the factor of 2 the pure square-root law gives.

    Record the measured ratio in PROJECT.md section 4 -- how far short it falls is
    a measurement of the neglected term, which is the useful output here.
    """
    ac, _, _, _, _ = _approach_trim()
    cnbs = np.array([0.075, 0.150, 0.300])

    def dr_wn(a, alpha, de, thr):
        return validation.lateral_modes(a, alpha, de, thr, CAUGHEY_V, 0.0)[0][0]

    wns = validation.sweep(ac, "Cnb", cnbs, dr_wn, CAUGHEY_V, 0.0)
    assert np.all(np.diff(wns) > 0), f"wn must rise with Cnb, got {wns}"
    assert 1.5 < wns[-1] / wns[0] < 2.1, (
        f"4x Cnb should roughly double wn (sqrt law gives 2.0), got {wns[-1] / wns[0]}"
    )
```

- [ ] **Step 3: Run them**

```bash
PYTHONPATH=. $PY -m pytest atisim/tests/test_validation.py -q
```
Expected: all pass.

Two known risks, both to be **reported rather than tuned around**:
- The Cma sweep may lose its oscillatory short period before −0.1, making `[-1]` the wrong root. If so, shorten the sequence to the values that keep one and say so in the docstring — the divergence is the expected physics.
- The Dutch-roll band is the widest guess in the plan. If the measured ratio falls outside 1.5–2.1, record it and set the band around the measurement with the Yβ/u₀ reasoning in the docstring.

- [ ] **Step 4: Commit**

```bash
git add atisim/tests/test_validation.py
git commit -m "Sweep drag, pitch stiffness, roll damping and weathercock stability"
```

---

## Task 13: The notebook

**Files:**
- Create: `notebooks/solver-validation.ipynb`
- Modify: `pyproject.toml`

- [ ] **Step 1: Add the dependencies**

In `pyproject.toml`:

```toml
[project.optional-dependencies]
dev = ["pytest", "pymupdf", "jupyter", "nbval"]
```

```bash
"C:/Users/mateusz/UROP/Claude_Flight_Sim/.venv/Scripts/python.exe" -m pip install jupyter nbval
```

- [ ] **Step 2: Build the notebook**

Create `notebooks/solver-validation.ipynb` with these cells in order. **Every cell imports and calls — no arithmetic written inline**, so nothing here can disagree with the suite.

1. *Markdown.* Title, the verification/validation split, and the source-qualification tier table from the design spec. State plainly that this validates the solver, not fidelity to a real 747.
2. *Code.* `import atisim; from atisim import validation, verification, trim, integrate; from atisim.aircraft import REGISTRY, CRUISE`.
3. *Markdown.* "Tier 0 — does the arithmetic work?"
4. *Code.* The manufactured-solution refinement: print `verification.fitted_order(...)`, plot log error against log dt with a reference slope-4 line.
5. *Code.* `verification.newton_residual_history(...)` as a table.
6. *Code.* `verification.torque_free_omega(...)` against the integrated trajectory, three panels.
7. *Markdown.* "Tier 1 — does a known coefficient change give the known result?"
8. *Code.* The CD0 sweep via `validation.sweep`: ζ_phugoid against L/D, and the product against sample index to show it flat.
9. *Code.* The Cmα sweep via `validation.sweep`: short-period ωn against Cmα with `sqrt(-Cma)` overlaid.
10. *Markdown.* "Tier 2 — against a published worked example."
11. *Code.* The stability-axis plant matrix beside `validation.CAUGHEY_A`, with a per-element relative-difference column. **This is why `to_imperial_matrix` must convert all six dimensional elements** — A[0,2] and A[1,3] compare against Caughey zeros in the tests, so an incomplete conversion fails nothing there but misleads here.
12. *Code.* The four modes against `validation.REFERENCES`, each row printing `ref.source`.
13. *Markdown.* Closing: what passed, what the residuals are attributable to, and what is **not** claimed.

- [ ] **Step 3: Run the notebook**

```bash
PYTHONPATH=. $PY -m pytest --nbval-lax notebooks/ -q
```
Expected: all cells execute without error.

`--nbval-lax` checks that cells run, not that their output text matches: figures and float formatting would otherwise fail on cosmetic changes. The **numbers** are pinned by `test_validation.py`, not by the notebook.

- [ ] **Step 4: Register it as a gate — but do NOT set a global `addopts`**

The project has **no CI configuration** (no `.github/`), so "executed in CI" means "a documented gate command that must pass". Add to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["atisim/tests"]
```

`pyproject.toml` currently has no `[tool.pytest.ini_options]` table — verified — so this adds rather than duplicates. A duplicate table is a TOML parse error.

**Do not** put `addopts = "--nbval-lax"` there. It would make every `pytest` invocation fail with "unrecognized arguments" wherever nbval is not installed, including any install of the package without the dev extra.

The notebook runs as its own command, documented in §10 (Task 14) as a required second gate alongside the suite:

```bash
PYTHONPATH=. $PY -m pytest --nbval-lax notebooks/ -q
```

- [ ] **Step 5: Run both gates**

```bash
PYTHONPATH=. $PY -m pytest -q
PYTHONPATH=. $PY -m pytest --nbval-lax notebooks/ -q
```
Expected: both green.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml notebooks/solver-validation.ipynb
git commit -m "Add the validation notebook and its gate command"
```

---

## Task 14: Record it in PROJECT.md

**Files:**
- Modify: `docs/PROJECT.md` §3, §4, §5, §7, §9, §10

- [ ] **Step 1: §4 — a verification subsection**

Add above "The validated baseline", titled "Verification — is the arithmetic right? (session 11)": measured RK4 order (manufactured and 6-DOF), Galilean invariance, Newton convergence ratios, torque-free agreement, and the Dutch-roll ratio from Task 12. **Use the values the tests actually printed**, not the ones in this plan.

- [ ] **Step 2: §4 — a validation subsection**

"The 747 approach against an independent implementation (session 11)": the four modes against Caughey, the plant-matrix element table, and the reconstructed α̇ residuals.

- [ ] **Step 3: §5 — correct the attribution, do not just extend it**

The existing entry reads *"CR-2144 Table IX-4's `Xu, Zu, Mu, Żw, Ṁw` are deliberately excluded"*. **That wording is imprecise and this session has the evidence to fix it:**

- The model **has** Xu and Zu. They fall out of dynamic-pressure variation — drag and lift both go as V² — and Task 9 measures A[0,0] within **1.2%** of Caughey's Xu without any Xu having been entered anywhere. What is excluded is their **Mach content** (CXu, CZu, from CL_M and CD_M), which is why the gap opens at M 0.8 and closes at M 0.25.
- Mu genuinely is ≈ 0 in the model, since Cm = 0 at trim and there is no Cm_M term — consistent with Caughey's A[2,0] = 0.0001, which is his Mẇ·Zu rather than an Mu.
- Zẇ and Mẇ are the two genuinely absent terms, and are now **reconstructed to within the source's own one-significant-figure Mẇ**, not merely attributed.

Rewrite the entry accordingly, and add the new attributed gap: **CR-2144's 747 data is the flexible airframe** — §IX's plots are labelled "Flexible" and `dynamics.py` integrates a rigid body.

- [ ] **Step 4: §3 — two rows**

- Caughey / Cornell MAE 5070: supplies an independent *implementation* of CR-2144's 747 approach case. Known gap — **not an independent dataset**; it cites CR-2144.
- The 279.1 vs 278.49 ft/s speed discrepancy, under CR-2144's known-gap column.

- [ ] **Step 5: §7 — the plan block**

Mark the validation work done. Record what it did **not** cover: Jetstar and C-5A (CR-2144 Tables VII-1 and X-1, both power-approach non-dimensional sets, body and stability axes respectively) remain the cheapest next aircraft; the 747's other flight conditions remain unread because CR-2144 pp. 229–236 are scanned line-printer output.

- [ ] **Step 6: §8 — one new open question**

The Galilean test uses a **steady** wind, whose material derivative is zero, so it cannot detect the spurious `−m·dW/dt` term §2 also warns about. That needs a time-varying field and an assertion other than invariance. Record it as open rather than letting §4 imply both errors are covered.

- [ ] **Step 7: §9 — a session entry**

Follow the template at the bottom of §9. Lead with what review asked for and what the answer turned out to be. State plainly anything that failed.

- [ ] **Step 8: §10 — the entry points**

Add the notebook gate command, the two new test files, and note that `testpaths` is now set explicitly.

- [ ] **Step 9: Final run**

```bash
PYTHONPATH=. $PY -m pytest -q
PYTHONPATH=. $PY -m pytest --nbval-lax notebooks/ -q
```
Expected: both green. Record the test count and runtime for §4.

- [ ] **Step 10: Commit**

```bash
git add docs/PROJECT.md
git commit -m "Record the verification pass, and correct section 5's attribution"
```

---

## Self-review notes

- **Spec coverage.** Family A → Tasks 2–6. Family B → Tasks 11–12. Family C → Tasks 8–10. Module architecture → Tasks 2, 7, 8. Notebook → Task 13. Findings → Task 14.
- **Deferred, with reason.** The spec flags checking whether Etkin & Reid publishes an independent **cruise** worked example. That needs the physical book, so it stays an open question in §8 rather than a task.
- **Naming.** `longitudinal_matrix` → `to_stability_axes` → `to_imperial_matrix` → `modes_from_matrix` is the fixed order of application throughout. There is exactly one RK4 implementation, `integrate.rk4_step`, called by both `step` and the verification tests.
- **No element asserted twice.** Task 9's first test covers A[0,0], A[0,1], A[1,0] and the structural elements; its second covers A[1,1], A[1,2], A[2,1], A[2,2]. Disjoint.
- **The task most likely to bite** is now Task 12's Dutch-roll band, which is a guess rather than a measurement. Task 6's closed form, previously the riskiest, is verified against an independent RK4 to 9.2e-15 and carries a domain guard.
