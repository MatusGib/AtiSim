# Solver Validation Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish that the solver is arithmetically correct and that known coefficient changes produce known results, before any further modelling layer is added.

**Architecture:** Two new flat modules beside the existing ones — `flightsim/verification.py` for checks that depend on no aircraft data at all, and `flightsim/validation.py` for checks against analytic laws and published worked examples. A Jupyter notebook is a thin front end that imports both and plots; it contains no arithmetic, and `nbval` executes it in the suite so it cannot drift. Design spec: `docs/superpowers/specs/2026-08-11-solver-validation-design.md`.

**Tech Stack:** Python 3.10, JAX (float64 via `flightsim/__init__.py`), NumPy, SciPy, pytest, Jupyter + nbval.

---

## Conventions for every task

**Working directory** is the worktree root. **Every** command below needs `PYTHONPATH=.` or the editable install resolves to the main checkout instead (PROJECT.md §10, "Environment notes").

The interpreter lives in the main checkout, not the worktree:

```bash
PYTHONPATH=. "C:/Users/mateusz/UROP/Claude_Flight_Sim/.venv/Scripts/python.exe" -m pytest flightsim/tests -q
```

Shorthand used below: `$PY` means `"C:/Users/mateusz/UROP/Claude_Flight_Sim/.venv/Scripts/python.exe"`.

**Two project rules that override normal practice, from PROJECT.md:**

1. **Do not edit a tolerance to make a test pass.** If `test_conservation.py`, `test_cr2144_modes.py`, `test_drag_polar.py`, `test_navion.py` or `test_trim.py` moves, something real broke — stop and report, do not adjust.
2. **Flag, never invent.** Every reference number carries the table it came from, in the code, as a string.

`conftest.py` turns on `jax_debug_nans` for the whole suite. A NaN raises rather than propagating; if a new test raises there, the physics setup is wrong, not the guard.

---

## File Structure

| File | Responsibility |
|---|---|
| `flightsim/integrate.py` (modify) | extract `rk4_step` so the stage weights can be tested on a problem with a closed-form solution |
| `flightsim/verification.py` (create) | tier 0: no aircraft data. Order of accuracy, Galilean invariance, torque-free rigid body, Newton convergence |
| `flightsim/validation.py` (create) | tiers 1–2: plant matrices, axis transforms, published references, coefficient sweeps |
| `flightsim/tests/modes.py` (modify) | becomes a two-line re-export so `test_cr2144_modes.py` and `test_navion.py` are untouched |
| `flightsim/tests/test_verification.py` (create) | tests for `verification.py` |
| `flightsim/tests/test_validation.py` (create) | tests for `validation.py` |
| `notebooks/solver-validation.ipynb` (create) | narrative and figures only |
| `pyproject.toml` (modify) | add `jupyter`, `nbval` to the dev extra |
| `docs/PROJECT.md` (modify) | §3, §4, §5, §7, §10 |

---

## Task 1: Extract the RK4 stage weights

`integrate.step` inlines RK4 around `derivatives`, so the stage weights cannot be exercised on a problem whose exact answer is known. Extract them. The refactor must change nothing, and the project's own favourite instrument proves it: bit-identical trajectories.

**Files:**
- Modify: `flightsim/integrate.py:57-94`
- Test: `flightsim/tests/test_verification.py`

- [ ] **Step 1: Capture the pre-refactor trajectory**

Run this and keep the printed hash — it is the baseline for Step 5.

```bash
PYTHONPATH=. $PY -c "
import jax, jax.numpy as jnp, numpy as np, hashlib
import flightsim
from flightsim import trim, integrate
from flightsim.aircraft import REGISTRY, CRUISE
ac = REGISTRY['boeing747']
V, H = CRUISE['boeing747']['airspeed'], CRUISE['boeing747']['altitude']
x, _ = trim.trim(jnp.array(V), jnp.array(H), ac)
st = trim.trimmed_state(x[0], jnp.array(V), jnp.array(H))
ct = trim.trimmed_controls(x[1] + 0.02, x[2])
sim = integrate.init_sim(st, jax.random.PRNGKey(0))
_, traj = integrate.rollout(sim, ct, jnp.array(0.02), ac, 500)
a = np.asarray(traj.vel_body)
print(hashlib.sha256(a.tobytes()).hexdigest())
"
```

- [ ] **Step 2: Extract `rk4_step`**

In `flightsim/integrate.py`, insert after `_axpy` (line 59) and rewrite the body of `step` to call it. Keep `_axpy` where it is.

```python
def rk4_step(f, x, dt):
    """One classical RK4 stage set on a pytree state.

    Split out of `step` so the stage weights can be verified against a problem
    with a closed-form solution -- `step` is welded to `derivatives`, and
    conservation drift cannot distinguish a fourth-order scheme from a
    second-order one. See flightsim/verification.py.
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

Then replace lines 76-85 of `step` (from `x = sim.state` through `new_state = _axpy(x, increment, dt)`) with:

```python
    new_state = rk4_step(f, sim.state, dt)
```

Leave the `quat_normalize` line and the `return SimState(...)` exactly as they are.

- [ ] **Step 3: Re-run the trajectory hash**

Run the Step 1 command again.
Expected: **the identical hash**. If it differs, the extraction changed the arithmetic — revert and redo.

- [ ] **Step 4: Write the regression test**

Create `flightsim/tests/test_verification.py`:

```python
"""Tier-0 verification: checks that depend on no aircraft data at all.

PROJECT.md section 4 is almost entirely validation -- a measured quantity against
a published one for one aircraft. These are the other kind: if one of them fails,
the arithmetic is wrong and no source can say otherwise.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import flightsim  # noqa: F401  -- enables x64 before any array is made
from flightsim import integrate, trim, verification
from flightsim.aircraft import CRUISE, REGISTRY


def _fixed_control_rollout(dt, n_steps):
    """747 at cruise trim with the elevator off trim, so something happens."""
    ac = REGISTRY["boeing747"]
    V, H = CRUISE["boeing747"]["airspeed"], CRUISE["boeing747"]["altitude"]
    x, _ = trim.trim(jnp.array(V), jnp.array(H), ac)
    state = trim.trimmed_state(x[0], jnp.array(V), jnp.array(H))
    controls = trim.trimmed_controls(x[1] + 0.02, x[2])
    sim = integrate.init_sim(state, jax.random.PRNGKey(0))
    return integrate.rollout(sim, controls, jnp.array(dt), ac, n_steps)


def test_rk4_step_reproduces_the_inline_integrator():
    """The Task 1 extraction must be arithmetic-neutral.

    Asserted as bit-identity rather than a tolerance, which is the same
    instrument PROJECT.md section 4 uses for the zero-wind path: a refactor that
    moves the last bit has changed the integrator.
    """
    ac = REGISTRY["boeing747"]
    V, H = CRUISE["boeing747"]["airspeed"], CRUISE["boeing747"]["altitude"]
    x, _ = trim.trim(jnp.array(V), jnp.array(H), ac)
    state = trim.trimmed_state(x[0], jnp.array(V), jnp.array(H))
    controls = trim.trimmed_controls(x[1], x[2])

    def f(s):
        from flightsim.dynamics import derivatives
        return derivatives(s, controls, ac, jnp.zeros(3), jnp.zeros(3))

    dt = jnp.array(0.02)
    manual = integrate._axpy(
        state,
        jax.tree.map(
            lambda a, b, c, d: (a + 2.0 * b + 2.0 * c + d) / 6.0,
            f(state),
            f(integrate._axpy(state, f(state), dt / 2)),
            f(integrate._axpy(state, f(integrate._axpy(state, f(state), dt / 2)), dt / 2)),
            f(integrate._axpy(
                state,
                f(integrate._axpy(state, f(integrate._axpy(state, f(state), dt / 2)), dt / 2)),
                dt,
            )),
        ),
        dt,
    )
    viaHelper = integrate.rk4_step(f, state, dt)
    assert np.array_equal(np.asarray(manual.vel_body), np.asarray(viaHelper.vel_body))
    assert np.array_equal(np.asarray(manual.omega), np.asarray(viaHelper.omega))
```

- [ ] **Step 5: Run it**

```bash
PYTHONPATH=. $PY -m pytest flightsim/tests/test_verification.py -q
```
Expected: 1 passed. (`verification` is imported but unused so far; that is fine — the module is created in Task 2. If the import fails, comment it out until Task 2 and restore it then.)

- [ ] **Step 6: Run the full suite — nothing may move**

```bash
PYTHONPATH=. $PY -m pytest flightsim/tests -q
```
Expected: 296 passed, 1 skipped (plus the new test). **No baseline test may change.**

- [ ] **Step 7: Commit**

```bash
git add flightsim/integrate.py flightsim/tests/test_verification.py
git commit -m "Split the RK4 stage weights out of step, bit-identically"
```

---

## Task 2: Observed order of accuracy on a manufactured solution

**Files:**
- Create: `flightsim/verification.py`
- Test: `flightsim/tests/test_verification.py`

- [ ] **Step 1: Write the failing test**

Append to `flightsim/tests/test_verification.py`:

```python
def test_rk4_is_fourth_order_on_a_problem_with_a_closed_form():
    """The single biggest gap in the project's evidence.

    PROJECT.md section 4 asserts that angular momentum barely drifts. An
    integrator can conserve beautifully and still be second-order: drift measures
    a symmetry, not an order. Nothing in the project asserts the order, so a
    mis-weighted stage would pass every existing test at dt = 0.02 and quietly
    degrade every result taken at a larger step.

    Harmonic oscillator, xdot = [[0, 1], [-1, 0]] x, exact solution a rotation.
    """
    def f(x):
        return jnp.array([x[1], -x[0]])

    x0 = jnp.array([1.0, 0.0])
    t_end = 2.0

    def final(dt):
        n = int(round(t_end / dt))
        x = x0
        for _ in range(n):
            x = verification.rk4_step_array(f, x, dt)
        return np.asarray(x)

    exact = np.array([np.cos(t_end), -np.sin(t_end)])
    dts = np.array([0.2, 0.1, 0.05, 0.025])
    errors = np.array([np.linalg.norm(final(dt) - exact) for dt in dts])
    slope = verification.fitted_order(dts, errors)
    assert slope == pytest.approx(4.0, abs=0.05), f"observed order {slope}"
```

- [ ] **Step 2: Run it and watch it fail**

```bash
PYTHONPATH=. $PY -m pytest flightsim/tests/test_verification.py::test_rk4_is_fourth_order_on_a_problem_with_a_closed_form -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'flightsim.verification'`.

- [ ] **Step 3: Create the module**

Create `flightsim/verification.py`:

```python
"""Tier-0 verification: is the arithmetic right?

Nothing here takes an aircraft's published data as a reference. These checks ask
whether the integrator, the trim solve and the rigid-body equations are correct
as mathematics -- which is a different question from whether the aerodynamic
coefficients describe a real aeroplane, and is the question that has to be
settled first. A failure here is a defect in the core.

The split is the standard verification/validation one (Roache; AIAA G-077).
PROJECT.md section 4 mixes them; see the design spec for why separating them is
most of the value.
"""

import jax
import jax.numpy as jnp
import numpy as np

from flightsim.integrate import _axpy


def rk4_step_array(f, x, dt):
    """`integrate.rk4_step` for a plain array state.

    Identical arithmetic -- `_axpy` maps over a pytree, and an array is a leaf --
    but callable with a hand-written right-hand side whose exact solution is
    known. This is what lets the stage weights be checked independently of
    `dynamics.derivatives`.
    """
    k1 = f(x)
    k2 = f(_axpy(x, k1, dt / 2))
    k3 = f(_axpy(x, k2, dt / 2))
    k4 = f(_axpy(x, k3, dt))
    increment = jax.tree.map(
        lambda a, b, c, d: (a + 2.0 * b + 2.0 * c + d) / 6.0, k1, k2, k3, k4
    )
    return _axpy(x, increment, dt)


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
PYTHONPATH=. $PY -m pytest flightsim/tests/test_verification.py -q
```
Expected: 2 passed.

**If the slope is not 4.0, stop.** That is the finding this whole plan exists to look for. Report the measured slope, do not adjust the tolerance.

- [ ] **Step 5: Commit**

```bash
git add flightsim/verification.py flightsim/tests/test_verification.py
git commit -m "Assert RK4 is fourth-order, which nothing did before"
```

---

## Task 3: Observed order of accuracy on the real 6-DOF

The manufactured case isolates the stage weights. It cannot see a wind sample or a control update applied at the wrong stage — which is exactly the seam turbulence will lean on (PROJECT.md §2: "wind sampled once per step, held across the four stages").

**Files:**
- Test: `flightsim/tests/test_verification.py`

- [ ] **Step 1: Write the failing test**

```python
def test_the_six_dof_rollout_is_fourth_order():
    """Same claim, through the real dynamics.

    The reference is generated at the smallest step in the sequence rather than
    analytically, so it carries its own error. The fit therefore uses only steps
    at least 8x the reference step, which keeps the reference's error at most
    8^-4 = 1/4096 of the smallest error being fitted.
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
    assert slope == pytest.approx(4.0, abs=0.15), (
        f"observed order {slope}, errors {errors}"
    )
```

- [ ] **Step 2: Run it**

```bash
PYTHONPATH=. $PY -m pytest flightsim/tests/test_verification.py::test_the_six_dof_rollout_is_fourth_order -q
```
Expected: PASS. Each distinct `dt` forces a recompile of the jitted `rollout` (`n_steps` is static), so this test takes tens of seconds. That is expected; do not reduce the sequence to speed it up.

If the slope lands near 4 but outside `abs=0.15`, widen **only** if the errors span less than two decades — otherwise it is a real finding. Record the measured value either way.

- [ ] **Step 3: Commit**

```bash
git add flightsim/tests/test_verification.py
git commit -m "Assert the order of accuracy through the real dynamics too"
```

---

## Task 4: Galilean invariance under a uniform wind

PROJECT.md §2 names substituting `vel_rel` into the Coriolis term as a classic gust-modelling error, and says the code deliberately does not. Nothing asserts it. The existing zero-wind bit-identity test cannot: with no wind, the correct and incorrect forms are the same number.

**Files:**
- Test: `flightsim/tests/test_verification.py`

- [ ] **Step 1: Write the failing test**

```python
def test_a_uniform_wind_only_translates_the_trajectory():
    """Galilean invariance, the assertion the zero-wind test cannot make.

    A uniform wind W is a change of inertial frame. Fly the same aircraft with
    its ground velocity offset by W and the aerodynamics see an identical
    relative flow, so attitude and body rates must be untouched and the position
    must differ by exactly W*t.

    PROJECT.md section 2: putting vel_rel into the Coriolis term breaks this, and
    adding an explicit -m dW/dt term double-counts. Both are invisible in still
    air, which is why every test in the project was blind to them until now.
    """
    W = jnp.array([7.0, -3.0, 2.0])  # m/s NED

    def uniform_wind(wind_state, state, key, dt):
        return W, jnp.zeros(3), wind_state, key

    ac = REGISTRY["boeing747"]
    V, H = CRUISE["boeing747"]["airspeed"], CRUISE["boeing747"]["altitude"]
    x, _ = trim.trim(jnp.array(V), jnp.array(H), ac)
    state = trim.trimmed_state(x[0], jnp.array(V), jnp.array(H))
    controls = trim.trimmed_controls(x[1] + 0.01, x[2])

    from flightsim.state import quat_to_dcm

    dcm = quat_to_dcm(state.quat)          # body -> NED
    shifted = state._replace(vel_body=state.vel_body + dcm.T @ W)

    dt, n = jnp.array(0.02), 1000
    _, still = integrate.rollout(
        integrate.init_sim(state, jax.random.PRNGKey(0)), controls, dt, ac, n
    )
    _, blown = integrate.rollout(
        integrate.init_sim(shifted, jax.random.PRNGKey(0)),
        controls, dt, ac, n, wind_model=uniform_wind,
    )

    # Attitude and rates must be untouched.
    np.testing.assert_allclose(
        np.asarray(blown.quat), np.asarray(still.quat), atol=1e-11
    )
    np.testing.assert_allclose(
        np.asarray(blown.omega), np.asarray(still.omega), atol=1e-11
    )

    # Position must differ by exactly W*t.
    t = np.arange(1, n + 1) * float(dt)
    expected = np.asarray(still.pos_ned) + t[:, None] * np.asarray(W)
    np.testing.assert_allclose(np.asarray(blown.pos_ned), expected, atol=1e-6)
```

- [ ] **Step 2: Run it**

```bash
PYTHONPATH=. $PY -m pytest flightsim/tests/test_verification.py::test_a_uniform_wind_only_translates_the_trajectory -q
```
Expected: PASS.

If `quat_to_dcm` is not the exported name, check `flightsim/state.py` for the body-to-NED direction-cosine helper and use that; do not write a second one.

If this **fails**, it is a real finding about the wind path and must be reported before continuing, not worked around.

- [ ] **Step 3: Commit**

```bash
git add flightsim/tests/test_verification.py
git commit -m "Assert Galilean invariance, which still air could never show"
```

---

## Task 5: Newton trim convergence is quadratic

**Files:**
- Modify: `flightsim/verification.py`
- Test: `flightsim/tests/test_verification.py`

- [ ] **Step 1: Write the failing test**

```python
def test_the_trim_solve_converges_quadratically():
    """A Newton solve that is merely converging may have a wrong Jacobian.

    Quadratic convergence -- the residual exponent roughly doubling each step --
    is the signature that jacfwd is differentiating the same function the
    residual evaluates. A finite-difference or stale Jacobian still converges,
    linearly, and the final residual alone cannot tell the two apart.
    """
    ac = REGISTRY["boeing747"]
    V, H = CRUISE["boeing747"]["airspeed"], CRUISE["boeing747"]["altitude"]
    history = verification.newton_residual_history(
        jnp.array(V), jnp.array(H), ac, iterations=6
    )
    assert history[0] > 1.0, "the start point is already converged; pick a worse one"

    # Find the last step before float64 saturates, then check the doubling.
    usable = [r for r in history if r > 1e-13]
    assert len(usable) >= 3, f"converged too fast to measure: {history}"
    e = np.log10(np.array(usable))
    ratios = (e[2:] - e[1:-1]) / (e[1:-1] - e[:-2])
    assert np.max(ratios) > 1.6, f"convergence looks linear, ratios {ratios}"
```

- [ ] **Step 2: Run it and watch it fail**

```bash
PYTHONPATH=. $PY -m pytest flightsim/tests/test_verification.py::test_the_trim_solve_converges_quadratically -q
```
Expected: FAIL — `AttributeError: module 'flightsim.verification' has no attribute 'newton_residual_history'`.

- [ ] **Step 3: Implement**

Append to `flightsim/verification.py`:

```python
def newton_residual_history(airspeed, altitude, ac, iterations=6):
    """Residual norm after each Newton iteration, from trim.py's own start point.

    `trim.trim` runs a fixed iteration count inside `lax.scan` and returns only
    the final answer, so the convergence rate is not observable through it. This
    repeats the same update with the same initial guess and keeps every step.
    """
    from flightsim.trim import residual

    x = jnp.array([0.05, 0.0, 0.5])  # trim.trim's own default guess
    history = [float(jnp.linalg.norm(residual(x, airspeed, altitude, ac)))]
    for _ in range(iterations):
        r = residual(x, airspeed, altitude, ac)
        jacobian = jax.jacfwd(residual)(x, airspeed, altitude, ac)
        x = x - jnp.linalg.solve(jacobian, r)
        history.append(float(jnp.linalg.norm(residual(x, airspeed, altitude, ac))))
    return np.array(history)
```

- [ ] **Step 4: Run it and watch it pass**

```bash
PYTHONPATH=. $PY -m pytest flightsim/tests/test_verification.py -q
```
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add flightsim/verification.py flightsim/tests/test_verification.py
git commit -m "Assert the trim Newton solve converges quadratically"
```

---

## Task 6: Torque-free rigid body against the analytic solution

PROJECT.md §4 records that angular momentum drifts by 5.7e-13. That says the integrator is conservative, not that it is right. Euler's equations for a torque-free asymmetric body have a closed-form solution in Jacobi elliptic functions, so "right" is checkable.

**Files:**
- Modify: `flightsim/verification.py`
- Test: `flightsim/tests/test_verification.py`

- [ ] **Step 1: Implement the closed form, and verify it before trusting it**

Append to `flightsim/verification.py`:

```python
def torque_free_omega(I1, I2, I3, omega0, t):
    """Exact torque-free rotation of an asymmetric rigid body.

    Landau & Lifshitz, *Mechanics*, section 37, for I1 < I2 < I3 with the
    rotation nearest the I3 axis (2*T*I3 > L^2 > 2*T*I1). Returns (3, len(t)).

    This is a reference solution, so it is checked against Euler's equations by
    `test_the_analytic_torque_free_solution_solves_eulers_equations` before any
    test relies on it -- a mis-stated closed form would otherwise look like an
    integrator defect.
    """
    from scipy.special import ellipj

    I = np.array([I1, I2, I3], dtype=float)
    w0 = np.asarray(omega0, dtype=float)
    L2 = float(np.sum((I * w0) ** 2))
    twoT = float(np.sum(I * w0**2))

    a1 = np.sqrt((twoT * I3 - L2) / (I1 * (I3 - I1)))
    a2 = np.sqrt((twoT * I3 - L2) / (I2 * (I3 - I2)))
    a3 = np.sqrt((L2 - twoT * I1) / (I3 * (I3 - I1)))
    rate = np.sqrt((I3 - I2) * (L2 - twoT * I1) / (I1 * I2 * I3))
    m = ((I2 - I1) * (twoT * I3 - L2)) / ((I3 - I2) * (L2 - twoT * I1))

    sn, cn, dn, _ = ellipj(rate * np.asarray(t, dtype=float), m)
    return np.vstack([a1 * cn, a2 * sn, a3 * dn]), rate, m


def torque_free_period(I1, I2, I3, omega0):
    """Period of the elliptic functions above, 4K(m)/rate."""
    from scipy.special import ellipk

    _, rate, m = torque_free_omega(I1, I2, I3, omega0, np.array([0.0]))
    return float(4.0 * ellipk(m) / rate)
```

- [ ] **Step 2: Write the test that the closed form is actually a solution**

```python
# Principal-axis inertias, deliberately well separated so the elliptic modulus
# is not near 0 or 1. Not an aircraft -- this test has no aircraft in it.
_I1, _I2, _I3 = 1420.0, 4070.0, 4780.0
_OMEGA0 = np.array([0.9, 0.05, 0.4])


def test_the_analytic_torque_free_solution_solves_eulers_equations():
    """Check the reference before using it as one."""
    t = np.linspace(0.0, 3.0, 601)
    w, _, _ = verification.torque_free_omega(_I1, _I2, _I3, _OMEGA0, t)
    I = np.array([_I1, _I2, _I3])[:, None]
    # Central difference of the closed form vs Euler's equations.
    dwdt = np.gradient(w, t, axis=1)
    euler = -np.cross(w.T, (I * w).T).T / I
    np.testing.assert_allclose(dwdt[:, 5:-5], euler[:, 5:-5], atol=1e-6)


def test_the_integrator_reproduces_torque_free_rotation():
    """Conservation is not correctness.

    PROJECT.md section 4 records angular-momentum drift of 5.7e-13 over 60,000
    steps. A scheme can conserve H exactly and still traverse the polhode at the
    wrong rate. This checks the trajectory, not the invariant.
    """
    from flightsim.aircraft import inertia_tensor
    from flightsim.state import State, euler_to_quat
    from flightsim.tests.conftest import make_test_aircraft

    ac = make_test_aircraft()._replace(
        inertia=inertia_tensor(_I1, _I2, _I3, 0.0),
        inertia_inv=jnp.linalg.inv(inertia_tensor(_I1, _I2, _I3, 0.0)),
        CL0=jnp.array(0.0), CLa=jnp.array(0.0), CLq=jnp.array(0.0), CLde=jnp.array(0.0),
        Cm0=jnp.array(0.0), Cma=jnp.array(0.0), Cmq=jnp.array(0.0), Cmde=jnp.array(0.0),
        CD0=jnp.array(0.0),
        CYb=jnp.array(0.0), CYp=jnp.array(0.0), CYr=jnp.array(0.0), CYdr=jnp.array(0.0),
        Clb=jnp.array(0.0), Clp=jnp.array(0.0), Clr=jnp.array(0.0),
        Clda=jnp.array(0.0), Cldr=jnp.array(0.0),
        Cnb=jnp.array(0.0), Cnp=jnp.array(0.0), Cnr=jnp.array(0.0),
        Cnda=jnp.array(0.0), Cndr=jnp.array(0.0),
        max_thrust=jnp.array(0.0),
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
        integrate.init_sim(state, jax.random.PRNGKey(0)),
        controls, jnp.array(dt), ac, n,
    )

    t = np.arange(1, n + 1) * dt
    exact, _, _ = verification.torque_free_omega(_I1, _I2, _I3, _OMEGA0, t)
    np.testing.assert_allclose(np.asarray(traj.omega).T, exact, atol=1e-8)
```

- [ ] **Step 3: Run both**

```bash
PYTHONPATH=. $PY -m pytest flightsim/tests/test_verification.py -q -k torque_free
```
Expected: 2 passed.

If the *first* test fails, the closed form is mis-stated — fix `torque_free_omega`, not the integrator. If only the *second* fails, that is a genuine integrator finding: report it.

If the aero-zeroing above trips `jax_debug_nans` (airspeed enters `air_data` as a divisor), confirm `vel_body` stays well above `aero.V_MIN`; with zero drag and zero thrust it is constant, so it will.

- [ ] **Step 4: Commit**

```bash
git add flightsim/verification.py flightsim/tests/test_verification.py
git commit -m "Check torque-free rotation against its closed form, not just its invariants"
```

---

## Task 7: Promote the linearisation into the package

`flightsim/tests/modes.py` holds the only linearisation in the project and lives inside the test package, so the notebook cannot import it and Family C cannot reach the plant matrix itself — only the modes derived from it. Move it, keeping the old import path working so no existing test changes.

**Files:**
- Create: `flightsim/validation.py`
- Modify: `flightsim/tests/modes.py`
- Test: `flightsim/tests/test_validation.py`

- [ ] **Step 1: Create `flightsim/validation.py`**

Move the two functions from `flightsim/tests/modes.py` verbatim — including their docstrings, which carry the reasoning about the `r*cos(phi)*tan(theta0)` term — and add a matrix-returning layer beneath them.

```python
"""Validation against analytic laws and published worked examples.

Tier 1 checks a coefficient sweep against a closed-form relation; tier 2 checks
the model against a source that published both its inputs and its own computed
outputs. Neither can be invalidated by a source's age: if CR-2144's derivatives
were 10% from the real aeroplane, this model must still reproduce CR-2144's own
transfer-function factors from CR-2144's own derivatives. See the design spec,
"Source qualification".

Tier 0 -- the checks that need no aircraft data at all -- is flightsim/verification.py.
"""

import jax
import jax.numpy as jnp
import numpy as np

from flightsim.aircraft import Aircraft
from flightsim.dynamics import derivatives
from flightsim.state import Controls, State, euler_to_quat


def longitudinal_matrix(ac, alpha, elevator, throttle, V, H):
    """Body-axis plant matrix in [u, w, q, theta], by jacfwd of the real dynamics.

    Body axes, so theta0 = alpha0 and w0 = V sin(alpha0) are both non-zero. Most
    textbook longitudinal matrices are quoted in STABILITY axes, where Theta0 = 0
    -- see `to_stability_axes`, which is what makes them comparable.
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
    transform: every element moves, the eigenvalues do not. Both halves of that
    are asserted in test_validation.py.

    Needed because published matrices state Theta0 = 0, which is a stability-axis
    statement -- compared raw against a body-axis matrix, the (2, 4) element reads
    -g sin(alpha0) against a published 0 and looks like a defect.
    """
    ca, sa = np.cos(alpha), np.sin(alpha)
    T = np.array([[ca, sa, 0.0, 0.0],
                  [-sa, ca, 0.0, 0.0],
                  [0.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0, 1.0]])
    return T @ A @ np.linalg.inv(T)


def modes_from_matrix(A):
    """(wn, zeta) for every oscillatory root, sorted low-to-high wn."""
    eig = np.linalg.eigvals(A)
    return sorted((abs(lam), -lam.real / abs(lam)) for lam in eig if lam.imag > 1e-9)
```

Then append the existing `longitudinal_modes` and `lateral_modes` from `flightsim/tests/modes.py`, changing only the body of `longitudinal_modes` to reuse the new pieces:

```python
def longitudinal_modes(ac, alpha, elevator, throttle, V, H):
    """Phugoid and short-period (wn, zeta), sorted low-to-high wn."""
    return modes_from_matrix(longitudinal_matrix(ac, alpha, elevator, throttle, V, H))
```

`lateral_modes` moves across unchanged, docstring included.

- [ ] **Step 2: Replace `flightsim/tests/modes.py` with a re-export**

```python
"""Kept as an import shim.

The linearisation moved to `flightsim.validation` so the notebook and the
validation module can reach the plant matrix itself, not only the modes derived
from it. `test_cr2144_modes.py` and `test_navion.py` import from here and are
deliberately untouched.
"""

from flightsim.validation import lateral_modes, longitudinal_modes  # noqa: F401
```

- [ ] **Step 3: Run the tests that depend on it — they must not move**

```bash
PYTHONPATH=. $PY -m pytest flightsim/tests/test_cr2144_modes.py flightsim/tests/test_navion.py -q
```
Expected: all pass, unchanged. These are §4 baseline files; if any value moves, revert and find out why.

- [ ] **Step 4: Write the similarity-transform test**

Create `flightsim/tests/test_validation.py`:

```python
"""Tier 1 and 2 validation: analytic laws, and published worked examples."""

import jax.numpy as jnp
import numpy as np
import pytest

import flightsim  # noqa: F401
from flightsim import trim, validation
from flightsim.aircraft import CRUISE, REGISTRY
from flightsim.units import FT2M

# Caughey Eq. (5.48): M = 0.25 at sea level. CR-2144 Table IX-2's header says
# 165 KTAS = 278.49 ft/s, a 0.2% difference, recorded in the design spec. Checks
# against Caughey run at Caughey's speed.
CAUGHEY_V = 279.1 * FT2M


def _approach_trim(V=CAUGHEY_V):
    ac = REGISTRY["boeing747_approach"]
    x, res = trim.trim(jnp.array(V), jnp.array(0.0), ac)
    return ac, float(x[0]), float(x[1]), float(x[2]), res


def test_the_stability_axis_transform_is_a_similarity_transform():
    """Every element moves; no eigenvalue does.

    This is what licenses comparing a body-axis matrix against a published
    stability-axis one at all, so it is asserted before it is used.
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
PYTHONPATH=. $PY -m pytest flightsim/tests/test_validation.py -q
```
Expected: 1 passed.

- [ ] **Step 6: Commit**

```bash
git add flightsim/validation.py flightsim/tests/modes.py flightsim/tests/test_validation.py
git commit -m "Promote the linearisation out of the test package, and add the axis transform"
```

---

## Task 8: The published reference table

**Files:**
- Modify: `flightsim/validation.py`
- Test: `flightsim/tests/test_validation.py`

- [ ] **Step 1: Write the failing test**

Append to `flightsim/tests/test_validation.py`:

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
PYTHONPATH=. $PY -m pytest flightsim/tests/test_validation.py::test_every_reference_carries_its_citation -q
```
Expected: FAIL — `AttributeError: module 'flightsim.validation' has no attribute 'REFERENCES'`.

- [ ] **Step 3: Implement**

Append to `flightsim/validation.py`:

```python
from typing import NamedTuple


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
# four. For checking a solver that is the useful kind of independence.
_CAUGHEY = "Caughey, Cornell MAE 5070 notes, Ch. 5, {}"

REFERENCES = {
    "747pa_short_period_wn": Reference(0.88178, _CAUGHEY.format("Eq. (5.54)")),
    "747pa_short_period_zeta": Reference(0.62546, _CAUGHEY.format("Eq. (5.54)")),
    "747pa_phugoid_wn": Reference(0.13391, _CAUGHEY.format("Eq. (5.54)")),
    "747pa_phugoid_zeta": Reference(0.01329, _CAUGHEY.format("Eq. (5.54)")),
    # Eq. (5.51): the dimensional derivatives, ft-s-rad. Mq is the one this
    # model can be expected to match exactly -- it has no alpha-dot content.
    "747pa_Mq": Reference(-0.4381, _CAUGHEY.format("Eq. (5.51)")),
    # Eq. (5.51) again: the two derivatives this model excludes by design, and
    # which the residuals in test_the_omitted_alpha_dot_terms_account_for_the_gap
    # are reconstructed from.
    "747pa_Zwdot": Reference(-0.0341, _CAUGHEY.format("Eq. (5.51)")),
    "747pa_Mwdot": Reference(-0.0002, _CAUGHEY.format("Eq. (5.51)")),
}

# Caughey Eq. (5.52), the [u, w, q, theta] plant matrix in ft/s and radians.
# Stability axes: he states Theta0 = 0, which is only true there.
CAUGHEY_A = np.array([
    [-0.0212,  0.0466,   0.000,   -32.174],
    [-0.2229, -0.5839, 262.472,     0.0],
    [ 0.0001, -0.0018,  -0.5015,    0.0],
    [ 0.0,     0.0,      1.0,       0.0],
])


def to_imperial_matrix(A):
    """A [u, w, q, theta] plant matrix from SI into Caughey's ft/s-rad units.

    The state mixes dimensions, so each element converts differently:
      A[0,0] A[0,1] A[1,0] A[1,1] A[2,2]   1/s          unchanged
      A[0,3]                               m/s^2 -> ft/s^2
      A[1,2]                               m/s   -> ft/s
      A[2,0] A[2,1]                        1/(m.s) -> 1/(ft.s)
    """
    out = np.array(A, dtype=float, copy=True)
    out[0, 3] /= FT2M
    out[1, 2] /= FT2M
    out[2, 0] *= FT2M
    out[2, 1] *= FT2M
    return out
```

Add `from flightsim.units import FT2M` to the module's imports.

- [ ] **Step 4: Run it and watch it pass**

```bash
PYTHONPATH=. $PY -m pytest flightsim/tests/test_validation.py -q
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add flightsim/validation.py flightsim/tests/test_validation.py
git commit -m "Add the published reference table, citations enforced by a test"
```

---

## Task 9: Reproduce Caughey's plant matrix element by element

The expected numbers below were measured during planning. They are targets, not tolerances to be fitted to.

**Files:**
- Test: `flightsim/tests/test_validation.py`

- [ ] **Step 1: Write the failing test**

```python
def test_the_plant_matrix_matches_caugheys_where_the_model_has_the_terms():
    """Tier 2: same source data, an outside implementation, published intermediates.

    Every element this model contains matches to 3.4% or better once the matrix is
    rotated into Caughey's stability axes. Measured during planning:

        A[0,0] Xu        -0.02094 vs -0.02120   1.2%
        A[0,1] Xw         0.04632 vs  0.04660   0.6%
        A[0,3] -g cos     -32.174 vs -32.174    0.000%
        A[1,3] -g sin     0.00000 vs  0.0       exact
        A[3,2]            1.00000 vs  1.0       exact

    A[1,2] and A[2,2] are excluded here and checked in the next test, because they
    carry the alpha-dot derivatives this model omits by design.
    """
    ac, alpha, de, thr, res = _approach_trim()
    assert float(jnp.linalg.norm(res)) < 1e-10

    A = validation.to_imperial_matrix(
        validation.to_stability_axes(
            validation.longitudinal_matrix(ac, alpha, de, thr, CAUGHEY_V, 0.0), alpha
        )
    )
    C = validation.CAUGHEY_A

    for (i, j), tol in {
        (0, 0): 0.02, (0, 1): 0.01, (1, 0): 0.03, (1, 1): 0.04, (2, 1): 0.09,
    }.items():
        assert A[i, j] == pytest.approx(C[i, j], rel=tol), f"A[{i},{j}]"

    # The kinematic and gravity elements are structural, so they are exact.
    assert A[0, 3] == pytest.approx(C[0, 3], rel=1e-4)
    assert A[1, 3] == pytest.approx(0.0, abs=1e-9)
    assert A[3, 2] == pytest.approx(1.0, abs=1e-12)


def test_the_omitted_alpha_dot_terms_account_for_the_gap_exactly():
    """Section 5's attribution, turned from a claim into arithmetic.

    Two elements disagree by more than 3%, and both carry derivatives this model
    excludes by design (aero.py is alpha/q/de only). Caughey's Z row is divided
    throughout by (1 - Zwdot); restoring that divisor from his own tabulated
    CL_alphadot = 6.7 recovers the published values to well under 1%. His A[2,2]
    is this model's Mq plus (u0 + Zq) Mwdot / (1 - Zwdot), from Cm_alphadot = -3.2.

    That is a much stronger statement than "attributed": the residual is not
    merely explained, it is reconstructed from the source's own numbers.
    """
    ac, alpha, de, thr, _ = _approach_trim()
    A = validation.to_imperial_matrix(
        validation.to_stability_axes(
            validation.longitudinal_matrix(ac, alpha, de, thr, CAUGHEY_V, 0.0), alpha
        )
    )
    C = validation.CAUGHEY_A
    Zwdot = validation.REFERENCES["747pa_Zwdot"].value
    Mwdot = validation.REFERENCES["747pa_Mwdot"].value

    # Z row: divide by (1 - Zwdot) and the published values appear.
    assert A[1, 1] / (1.0 - Zwdot) == pytest.approx(C[1, 1], rel=0.005)
    assert A[1, 2] / (1.0 - Zwdot) == pytest.approx(C[1, 2], rel=0.005)

    # Mq: this model's value IS Caughey's raw Eq. (5.51) Mq.
    assert A[2, 2] == pytest.approx(validation.REFERENCES["747pa_Mq"].value, rel=0.01)
    # ...and adding the omitted term reaches his A[2,2].
    reconstructed = A[2, 2] + C[1, 2] * Mwdot
    assert reconstructed == pytest.approx(C[2, 2], rel=0.02)


def test_the_approach_modes_match_caugheys_published_roots():
    """The end-to-end statement: units, trim, dynamics and jacfwd in one number."""
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
PYTHONPATH=. $PY -m pytest flightsim/tests/test_validation.py -q
```
Expected: 5 passed.

Any tolerance above that fails is a finding. **Do not widen it** — report the measured value against the target in the docstring.

- [ ] **Step 3: Commit**

```bash
git add flightsim/tests/test_validation.py
git commit -m "Reproduce Caughey's plant matrix, and reconstruct the omitted terms"
```

---

## Task 10: The gap is condition-dependent

**Files:**
- Test: `flightsim/tests/test_validation.py`

- [ ] **Step 1: Write the test**

```python
def test_the_mode_error_is_far_smaller_on_approach_than_at_cruise():
    """The finding that reframes section 5.

    Section 4 records the cruise phugoid 17.8% and short-period zeta 11.5% from
    CR-2144, attributed to the excluded Xu, Zu, Mu, Zwdot, Mwdot. The same code
    with the same omissions reads 0.4% and 5.5% at the power-approach point. So
    the gap is not a fixed modelling deficit; it is condition-dependent, and it
    bites at M 0.8 / 40,000 ft where compressibility drives Mu.

    Asserted as an ordering, per section 5's rule that this project asserts bands
    and orderings rather than values wherever the comparison is across sources.
    """
    ac, alpha, de, thr, _ = _approach_trim()
    (ph_wn, _), (_, sp_z) = validation.longitudinal_modes(
        ac, alpha, de, thr, CAUGHEY_V, 0.0
    )
    R = validation.REFERENCES
    ph_err = abs(ph_wn - R["747pa_phugoid_wn"].value) / R["747pa_phugoid_wn"].value
    sp_err = abs(sp_z - R["747pa_short_period_zeta"].value) / R["747pa_short_period_zeta"].value

    # PROJECT.md section 4, "747 modes vs CR-2144", cruise column.
    CRUISE_PHUGOID_ERR, CRUISE_SP_ZETA_ERR = 0.178, 0.115
    assert ph_err < CRUISE_PHUGOID_ERR / 10.0, (
        f"approach phugoid error {ph_err:.4f} vs cruise {CRUISE_PHUGOID_ERR}"
    )
    assert sp_err < CRUISE_SP_ZETA_ERR
```

- [ ] **Step 2: Run it**

```bash
PYTHONPATH=. $PY -m pytest flightsim/tests/test_validation.py::test_the_mode_error_is_far_smaller_on_approach_than_at_cruise -q
```
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add flightsim/tests/test_validation.py
git commit -m "Pin the phugoid gap as condition-dependent, not a fixed deficit"
```

---

## Task 11: Coefficient sweeps against the Lanchester laws

This is the review request read literally: change a coefficient, get a known result. The Lanchester approximations are the strongest available, because the size of their *error* is itself published.

**Files:**
- Modify: `flightsim/validation.py`
- Test: `flightsim/tests/test_validation.py`

- [ ] **Step 1: Write the failing test**

```python
def test_the_phugoid_frequency_follows_the_lanchester_law():
    """Tier 1: a relation with no aerodynamic coefficient in it at all.

    Lanchester's phugoid approximation is wn = sqrt(2) g / u0. It contains no
    derivative, no area, no mass -- so it cannot be satisfied by accident, and no
    source's vintage can affect it.

    Caughey states his approximate analysis "over predicts the undamped natural
    frequency by about 20 per cent" at this condition. Measured during planning:
    0.1630 against 0.1334, a ratio of 1.22. Matching the published SIZE of the
    approximation's error is a far tighter claim than matching its trend.
    """
    from flightsim.atmosphere import G0

    ac, alpha, de, thr, _ = _approach_trim()
    (ph_wn, _), _ = validation.longitudinal_modes(ac, alpha, de, thr, CAUGHEY_V, 0.0)
    lanchester = np.sqrt(2.0) * float(G0) / CAUGHEY_V
    assert lanchester / ph_wn == pytest.approx(1.22, rel=0.03)


def test_the_phugoid_damping_follows_the_lift_to_drag_law():
    """Lanchester again: zeta = 1 / (sqrt(2) L/D).

    Caughey states the approximation "over predicts the damping ratio by a factor
    of almost 5". Measured during planning: 0.0651 against 0.01329, a ratio of 4.9.
    """
    ac, alpha, de, thr, _ = _approach_trim()
    (_, ph_zeta), _ = validation.longitudinal_modes(ac, alpha, de, thr, CAUGHEY_V, 0.0)

    # CL and CD at the trimmed condition, from Table IX-2 as transcribed.
    L_over_D = 1.11 / 0.102
    lanchester = 1.0 / (np.sqrt(2.0) * L_over_D)
    assert lanchester / ph_zeta == pytest.approx(4.9, rel=0.06)


def test_worsening_the_drag_polar_damps_the_phugoid_as_the_law_says():
    """The sweep itself: change one coefficient, get the predicted change.

    zeta_phugoid is inversely proportional to L/D, so raising CD0 must raise it,
    and the product zeta * (L/D) must stay put across the sweep even though
    neither factor does.
    """
    ac, alpha, de, thr, _ = _approach_trim()
    base_CD0 = float(ac.CD0)
    factors = np.array([1.0, 1.5, 2.0, 3.0])

    zetas, l_over_ds = [], []
    for k in factors:
        swept = ac._replace(CD0=jnp.array(base_CD0 * k))
        x, _ = trim.trim(jnp.array(CAUGHEY_V), jnp.array(0.0), swept)
        (_, z), _ = validation.longitudinal_modes(
            swept, float(x[0]), float(x[1]), float(x[2]), CAUGHEY_V, 0.0
        )
        CD = float(swept.CD0) + 1.11**2 / (np.pi * float(swept.e) * float(swept.AR))
        zetas.append(z)
        l_over_ds.append(1.11 / CD)

    zetas, l_over_ds = np.array(zetas), np.array(l_over_ds)
    assert np.all(np.diff(zetas) > 0), "more drag must damp the phugoid"
    product = zetas * l_over_ds
    assert product.max() / product.min() < 1.15, (
        f"zeta * L/D should be near constant, got {product}"
    )
```

- [ ] **Step 2: Run them**

```bash
PYTHONPATH=. $PY -m pytest flightsim/tests/test_validation.py -q
```
Expected: all pass.

The two ratio assertions (1.22 and 4.9) are the load-bearing ones. If either misses, report the measured ratio against Caughey's stated "about 20 per cent" and "a factor of almost 5" — do not widen `rel`.

- [ ] **Step 3: Commit**

```bash
git add flightsim/tests/test_validation.py
git commit -m "Check the phugoid against Lanchester, error size included"
```

---

## Task 12: The remaining sweeps

**Files:**
- Modify: `flightsim/validation.py`
- Test: `flightsim/tests/test_validation.py`

- [ ] **Step 1: Add the sweep helper**

Append to `flightsim/validation.py`:

```python
def sweep(ac, field, values, quantity, V, H):
    """Vary one coefficient and report a scalar for each value.

    `Aircraft` is a NamedTuple, so `_replace` gives an independent airframe per
    sample with no mutation and no copy discipline to get wrong. The aircraft is
    re-trimmed at every sample, because changing a derivative moves the trim
    point and comparing modes across different trims would confound the two.

    `quantity` takes (aircraft, alpha, elevator, throttle) and returns a float.
    """
    from flightsim.trim import trim as solve_trim

    out = []
    for v in values:
        swept = ac._replace(**{field: jnp.array(float(v))})
        x, _ = solve_trim(jnp.array(V), jnp.array(H), swept)
        out.append(quantity(swept, float(x[0]), float(x[1]), float(x[2])))
    return np.array(out)
```

- [ ] **Step 2: Write the tests**

```python
def test_reducing_pitch_stiffness_lowers_the_short_period_frequency_to_zero():
    """The neutral point, approached from the stable side.

    Short-period wn goes as sqrt(-M_alpha), so driving Cm_alpha towards zero must
    drive wn towards zero. This is the sharpest sanity check in flight dynamics:
    a model that does not lose its short period at the neutral point does not have
    a short period, it has a curve fit.
    """
    ac, _, _, _, _ = _approach_trim()
    cmas = np.array([-1.26, -0.9, -0.6, -0.3, -0.1])

    def sp_wn(a, alpha, de, thr):
        modes = validation.longitudinal_modes(a, alpha, de, thr, CAUGHEY_V, 0.0)
        return modes[-1][0]

    wns = validation.sweep(ac, "Cma", cmas, sp_wn, CAUGHEY_V, 0.0)
    assert np.all(np.diff(wns) < 0), f"wn must fall as Cma -> 0, got {wns}"
    # sqrt law: halving |Cma| should drop wn by about sqrt(2).
    assert wns[0] / wns[-1] == pytest.approx(np.sqrt(1.26 / 0.1), rel=0.35)


def test_the_roll_time_constant_tracks_one_over_Clp():
    """tau_roll = -1/L_p, and L_p is proportional to Clp, so tau * Clp is constant."""
    ac, alpha, de, thr, _ = _approach_trim()
    clps = np.array([-0.30, -0.45, -0.60, -0.75])

    taus = []
    for clp in clps:
        swept = ac._replace(Clp=jnp.array(float(clp)))
        x, _ = trim.trim(jnp.array(CAUGHEY_V), jnp.array(0.0), swept)
        _, roll_tau, _ = validation.lateral_modes(
            swept, float(x[0]), float(x[1]), float(x[2]), CAUGHEY_V, 0.0
        )
        taus.append(roll_tau)

    product = np.array(taus) * np.abs(clps)
    assert product.max() / product.min() < 1.05, (
        f"tau_roll * |Clp| should be constant, got {product}"
    )


def test_the_dutch_roll_frequency_tracks_the_square_root_of_Cnb():
    """wn_dutch ~ sqrt(N_beta), and N_beta is proportional to Cnb."""
    ac, _, _, _, _ = _approach_trim()
    cnbs = np.array([0.075, 0.150, 0.300])

    wns = []
    for cnb in cnbs:
        swept = ac._replace(Cnb=jnp.array(float(cnb)))
        x, _ = trim.trim(jnp.array(CAUGHEY_V), jnp.array(0.0), swept)
        dr, _, _ = validation.lateral_modes(
            swept, float(x[0]), float(x[1]), float(x[2]), CAUGHEY_V, 0.0
        )
        wns.append(dr[0])

    ratio = np.array(wns) / np.sqrt(cnbs)
    assert ratio.max() / ratio.min() < 1.10, f"wn/sqrt(Cnb) should be flat: {ratio}"
```

- [ ] **Step 3: Run them**

```bash
PYTHONPATH=. $PY -m pytest flightsim/tests/test_validation.py -q
```
Expected: all pass.

The Cma sweep may produce a real (non-oscillatory) pair before Cma reaches −0.1, which makes `modes[-1]` an index error. If so, shorten the sequence to the values that keep an oscillatory short period and say so in the docstring — the divergence itself is the expected physics, not a failure.

- [ ] **Step 4: Commit**

```bash
git add flightsim/validation.py flightsim/tests/test_validation.py
git commit -m "Sweep pitch stiffness, roll damping and weathercock stability"
```

---

## Task 13: The notebook, executed in the suite

**Files:**
- Create: `notebooks/solver-validation.ipynb`
- Modify: `pyproject.toml:8-9`

- [ ] **Step 1: Add the dependencies**

In `pyproject.toml`, change the dev extra:

```toml
[project.optional-dependencies]
dev = ["pytest", "pymupdf", "jupyter", "nbval"]
```

Install:

```bash
"C:/Users/mateusz/UROP/Claude_Flight_Sim/.venv/Scripts/python.exe" -m pip install jupyter nbval
```

- [ ] **Step 2: Build the notebook**

Create `notebooks/solver-validation.ipynb` with these cells, in order. **Every cell must import and call — no arithmetic written inline**, so that nothing here can disagree with the test suite.

1. *Markdown.* Title, the verification/validation split, and the source-qualification tier table from the design spec. State plainly that this validates the solver, not fidelity to a real 747.
2. *Code.* `import flightsim; from flightsim import validation, verification, trim; from flightsim.aircraft import REGISTRY, CRUISE`.
3. *Markdown.* "Tier 0 — does the arithmetic work?"
4. *Code.* Run the manufactured-solution refinement, print the fitted order, and plot log error against log dt with a reference slope-4 line.
5. *Code.* Print the Newton residual history as a table.
6. *Markdown.* "Tier 1 — does a known coefficient change give the known result?"
7. *Code.* The CD0 sweep: plot ζ_phugoid against L/D, and the product against sample index to show it flat.
8. *Code.* The Cmα sweep: plot short-period ωn against Cmα with `sqrt(-Cma)` overlaid.
9. *Markdown.* "Tier 2 — against a published worked example."
10. *Code.* Print the model's stability-axis plant matrix beside `validation.CAUGHEY_A`, with a per-element relative-difference column.
11. *Code.* Print the four modes against `validation.REFERENCES`, each row carrying `ref.source`.
12. *Markdown.* Closing: what passed, what the residuals are attributable to, and what is **not** claimed.

- [ ] **Step 3: Run the notebook under nbval**

```bash
PYTHONPATH=. $PY -m pytest --nbval-lax notebooks/solver-validation.ipynb -q
```
Expected: all cells execute without error.

`--nbval-lax` checks that cells run rather than that their output text matches exactly, which is right here: the figures and float formatting would otherwise make the notebook fail on cosmetic changes. The *numbers* are pinned by `test_validation.py`, not by the notebook.

- [ ] **Step 4: Wire it into the default suite**

Append to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
addopts = "--nbval-lax"
testpaths = ["flightsim/tests", "notebooks"]
```

- [ ] **Step 5: Run everything**

```bash
PYTHONPATH=. $PY -m pytest -q
```
Expected: the whole suite plus the notebook, all passing.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml notebooks/solver-validation.ipynb
git commit -m "Add the validation notebook, executed by the suite so it cannot rot"
```

---

## Task 14: Record it in PROJECT.md

**Files:**
- Modify: `docs/PROJECT.md` §3, §4, §5, §7, §9, §10

- [ ] **Step 1: §4 — a new verification subsection**

Add above "The validated baseline", titled "Verification — is the arithmetic right? (session 11)", with the measured RK4 order (manufactured and 6-DOF), Galilean invariance, Newton convergence ratios, and torque-free agreement. Use the values the tests actually printed, not the ones in this plan.

- [ ] **Step 2: §4 — a new validation subsection**

"The 747 approach against an independent implementation (session 11)": the four modes against Caughey, the plant-matrix element table, and the two reconstructed α̇ residuals.

- [ ] **Step 3: §5 — three edits**

- New attributed gap: **CR-2144's 747 derivatives are the flexible airframe.** Section IX's plots are labelled "Flexible"; `dynamics.py` integrates a rigid body.
- Amend the phugoid/short-period entry: the offsets are **condition-dependent**, 0.4%/5.5% on approach against 17.8%/11.5% at cruise, with Caughey as the outside reference.
- Note that the excluded α̇ derivatives are now **reconstructed to <1%**, not merely attributed.

- [ ] **Step 4: §3 — two rows**

- Caughey / Cornell MAE 5070 as a source: supplies an independent implementation of CR-2144's 747 approach case; known gap — **not an independent dataset**, it cites CR-2144.
- Note the 279.1 vs 278.49 ft/s speed discrepancy under CR-2144's known-gap column.

- [ ] **Step 5: §7 — the plan block**

Mark the validation work done and record what it did **not** cover: Jetstar and C-5A (CR-2144 Tables VII-1 and X-1, both power-approach non-dimensional sets, body and stability axes respectively) remain the cheapest next aircraft, and the 747's other flight conditions remain unread because CR-2144 pp. 229–236 are scanned line-printer output.

- [ ] **Step 6: §9 — a session entry**

Follow the template at the bottom of §9. Lead with what review asked for and what the answer turned out to be. State plainly anything that failed.

- [ ] **Step 7: §10 — the entry points**

Add the notebook and the new test files, and note that `pytest` now collects `notebooks/` as well.

- [ ] **Step 8: Final full run**

```bash
PYTHONPATH=. $PY -m pytest -q
```
Expected: green. Record the test count and runtime for §4.

- [ ] **Step 9: Commit**

```bash
git add docs/PROJECT.md
git commit -m "Record the verification pass, and what it found"
```

---

## Self-review notes

- **Spec coverage.** Family A → Tasks 2–6. Family B → Tasks 11–12. Family C → Tasks 8–10. Module architecture → Tasks 2, 7, 8. Notebook + nbval → Task 13. Findings for PROJECT.md → Task 14. The spec's `observed_order` two-reference requirement is Tasks 2 and 3.
- **Spec item deliberately deferred:** the spec flags checking whether Etkin & Reid publishes an independent **cruise** worked example. That needs the physical book, so it stays an open question in §8 rather than a task here.
- **Naming consistency.** `rk4_step` (integrate, pytree) and `rk4_step_array` (verification, plain array) are deliberately different functions with the same arithmetic; `longitudinal_matrix` → `to_stability_axes` → `to_imperial_matrix` → `modes_from_matrix` is the fixed order of application throughout.
- **The one task most likely to bite:** Task 6's closed form. Its first test checks the reference against Euler's equations before the second test uses it, so a mis-stated formula reports itself as a bad reference rather than as an integrator defect.
