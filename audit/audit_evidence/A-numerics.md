# Agent A — Numerics

Evidence only. **No causal attribution.** Every number below was produced by running
the code, not read from a docstring.

**Interpreter:** `C:/Users/mateusz/UROP/Claude_Flight_Sim/.venv/Scripts/python.exe`
**Versions:** Python 3.10.11, jax 0.6.2, jaxlib 0.6.2, numpy 2.2.6, scipy 1.15.3,
device `CpuDevice(id=0)` (no GPU in this environment).

**Scripts:** `audit_evidence/scripts/a1..a11*.py`
**Captured output:** `audit_evidence/logs/a*.log`

**Nothing under `flightsim/` or `docs/` was modified.** No tolerance, test or
reference value was touched. Where I needed a variant of shipped code (an
`integrate.step` without the renormalisation line) I wrote a **local copy** in my
own script and said so.

Unless stated otherwise the test case is `REGISTRY["boeing747"]` at
`CRUISE["boeing747"]` = 235.9152 m/s TAS, 12192 m (40,000 ft), trimmed by
`trim.trim`, with the elevator perturbed +0.02 rad so something happens.

---

## Summary table

| # | Question | Answer measured |
|---|---|---|
| 1 | Is `rk4_step` classical RK4? | **Yes.** Stability polynomial matches `1+z+z²/2+z³/6+z⁴/24` to 4.3e-13; quadrature exact through t³ with exactly Simpson's t⁴ error. Observed order 3.9998 (oscillator), 3.9891 (6-DOF still air) |
| 2 | Richardson on full sim output | Still air: order exists, p→4.00, exact solution recovered. Smooth field: order exists, **p→1.00**. Rankine core: **no order exists**; extrapolated "exact" moves 6 mm in pN / 7 mm in pD depending on the assumed p |
| — | Round-off floor | **Confirmed ~3.6e-11 – 9.4e-11 m** at 40,000 ft. Project's 7e-11 m is right |
| 3 | Quaternion renormalisation | Removes 2.1e-13 of norm error at dt=0.02 (**bounded, not secular**) but **1.7e-5 at dt=0.1 over 1e5 steps (secular, linear in n)**. Removing it moves the trajectory: 3.3e-7 m at dt=0.02/1e5 steps, **233 m at dt=0.1/1e5 steps** |
| 4 | Conservation / invariance | Torque-free \|H\| drift 5.70e-13 (600 s @ dt=0.01). Galilean invariance to 2.8e-11 m for horizontal wind, **fails by 3.08 m for a 2 m/s vertical wind** (documented scope). Rotational invariance about NED-z holds to 1.1e-9 m with the wind field rotated too. Energy conserved to 1.5e-11 relative, bounded |
| 5 | Trim | Converges in **2–3 iterations**, `iterations=40` is 13–20× more than needed. **NOT unique:** 2–3 physically distinct roots per aircraft on real registry data at cruise; **9–14 % of a 2090-point guess grid lands on an absurd root** (α ≈ −118°, elevator 86°, throttle −194) |
| 6 | float64 | Set in `flightsim/__init__.py:10` before any array. A plain `import flightsim` **does** get x64. float32 costs ~1e-7 rel on modes (phugoid ζ 1.0e-6, Dutch roll ζ 1.8e-7) and **499 m of position after 2000 s** |
| 7 | jit vs eager | **Not bit-identical**, but at the last ulp: 1.6e-17 on trim, 4.5e-13 m on a 1000-step rollout, 1.4e-17 on the plant matrix, and the extracted modes are bit-identical floats |
| 8 | Other | `validation.lateral_modes` **swaps roll and spiral for the Cherokee** (unstable spiral). `trim.trim` has no bound on elevator or throttle. `is_physical` checks α only. `boeing747_approach` cruises **below** its own `minimum_drag_speed` |

---

## 1. Integrator order

### 1a. The Butcher weights, read from the code

`flightsim/integrate.py:65-82` is

```python
k1 = f(x)
k2 = f(_axpy(x, k1, dt / 2))
k3 = f(_axpy(x, k2, dt / 2))
k4 = f(_axpy(x, k3, dt))
increment = jax.tree.map(lambda a, b, c, d: (a + 2.0*b + 2.0*c + d) / 6.0, k1, k2, k3, k4)
return _axpy(x, increment, dt)
```

i.e. c = [0, ½, ½, 1], b = [1, 2, 2, 1]/6, lower-triangular A = [½, ½, 1] on the
sub-diagonal. That **is** the classical tableau.

### 1b. The Butcher weights, measured by probing the function

Two independent probes (`a1_order.py`):

**Stability polynomial.** For y' = λy, RK4's amplification is
R(z) = 1 + z + z²/2 + z³/6 + z⁴/24. Fitting a degree-4 polynomial to
`rk4_step(lambda y: y, 1.0, z)` over z ∈ [0.05, 0.35]:

```
fitted R(z) coefficients : [1.  1.  0.5  0.16666667  0.04166667]
classical RK4 exact      : [1.  1.  0.5  0.16666667  0.04166667]
max |diff|               : 4.340972026284362e-13
```

**Quadrature identity.** Integrating a clock alongside tᵏ (so `rk4_step` computes
∫₀¹ tᵏ dt in one step) isolates the b-weights and c-nodes:

| k | rk4 | exact | error |
|---|---|---|---|
| 0 | 1.0000000000000000 | 1.0000000000000000 | 0 |
| 1 | 0.5000000000000000 | 0.5000000000000000 | 0 |
| 2 | 0.3333333333333333 | 0.3333333333333333 | 0 |
| 3 | 0.2500000000000000 | 0.2500000000000000 | 0 |
| 4 | 0.2083333333333333 | 0.2000000000000000 | **+8.333e-03 = +1/120** |
| 5 | 0.1875000000000000 | 0.1666666666666667 | +2.083e-02 |

Exact through t³ and off by exactly Simpson's rule error (5/24 − 1/5 = 1/120) at
t⁴. That pins b = [1,2,2,1]/6 with c = [0,½,½,1] and no other tableau.

### 1c. Observed order on a problem with a closed form

`verification.oscillator_refinement`, ẋ = [[0,1],[−1,0]]x, exact solution a rotation:

| dt | error | pairwise order |
|---|---|---|
| 0.2 | 2.665603e-05 | |
| 0.1 | 1.666501e-06 | 3.9996 |
| 0.05 | 1.041641e-07 | 3.9999 |
| 0.025 | 6.510376e-09 | 4.0000 |

**Fitted order 3.999819351358985.**

Two more problems, driven through `rk4_step` with no project code path at all
(all step sizes chosen to divide t_end = 1 exactly):

y' = y, exact e:

| dt | error | pairwise order |
|---|---|---|
| 0.5 | 9.356371e-04 | |
| 0.25 | 7.188926e-05 | 3.7021 |
| 0.125 | 4.984042e-06 | 3.8504 |
| 0.0625 | 3.281185e-07 | 3.9250 |
| 0.03125 | 2.104785e-08 | 3.9625 |

y' = −y³, exact 1/√(1+2t) (nonlinear):

| dt | error | pairwise order |
|---|---|---|
| 0.25 | 1.878679e-05 | |
| 0.125 | 1.431722e-07 | 7.0358 |
| 0.0625 | 1.324969e-08 | 3.4337 |
| 0.03125 | 1.378638e-09 | 3.2646 |
| 0.015625 | 1.012780e-10 | 3.7669 |

The 7.04 and the sag to 3.26 are a zero crossing in the leading error
coefficient for this particular problem, not a property of the scheme; reported
as measured. The oscillator result (3.9998) is the clean one.

### 1d. Observed order of the FULL 6-DOF rollout

`verification.fixed_control_refinement`, 747 cruise, 4 s, dt_ref = 1/1024:

```
6-DOF still air            order = 3.989127   (PROJECT.md records 3.98913)
  errors = [5.68971293e-05 3.62320033e-06 2.27634168e-07 1.42195329e-08]
```

**Reproduces PROJECT.md's published 3.98913 to all six digits.**

Pairwise orders over the wide sweep the test's docstring quotes (my values vs the
docstring's):

| window | measured here | docstring |
|---|---|---|
| 1/4 → 1/8 | **+3.9730** | 3.973 |
| 1/8 → 1/16 | **+3.9925** | 3.993 |
| 1/16 → 1/32 | **+4.0008** | 4.008 |
| 1/32 → 1/64 | +4.0460 | 4.167 |
| 1/64 → 1/128 | +5.8532 | 3.420 |
| 1/128 → 1/256 | **−1.5634** | −0.685 |
| fitted over the wide window | **3.7406** | 3.82 |

The first three agree exactly. The last three are inside the round-off floor
(§2c) and differ between platforms, which is itself the evidence that they are
round-off and not discretisation.

---

## 2. Richardson extrapolation on FULL SIMULATION OUTPUT

`a2_richardson.py` / `a2b_richardson_table.py`. 747 at cruise, 4 s, the state at
t = 4 s taken from `integrate.rollout`. Refinement ratio 2, so
p_obs = log₂((f_h − f_{h/2}) / (f_{h/2} − f_{h/4})), and
f_exact = f_{h/4} + (f_{h/4} − f_{h/2})/(2^p − 1). **The reference is the
extrapolation, not a fine-step run** — so this does not inherit the fine run's
own error, which matters a great deal in case (b).

### (a) Still air

Triples on pN and pD (all values metres):

| h | comp | f_h | f_h/2 | f_h/4 | p_obs |
|---|---|---|---|---|---|
| 1/4 | pN | 944.47862200832719 | 944.47861007560800 | 944.47860935184303 | **4.0433** |
| 1/4 | pD | −12184.56752312271419 | −12184.56747120214823 | −12184.56746788461169 | **3.9681** |
| 1/8 | pN | 944.47861007560800 | 944.47860935184303 | 944.47860930728302 | **4.0217** |
| 1/8 | pD | −12184.56747120214823 | −12184.56746788461169 | −12184.56746767590084 | **3.9905** |
| 1/16 | pN | | | | **4.0103** |
| 1/16 | pD | | | | **3.9972** |
| 1/32 | pN | | | | **4.0097** |
| 1/32 | pD | | | | **3.9621** |
| 1/64 | pN | | | | 4.2205 |
| 1/64 | pD | | | | 4.2048 |
| 1/128 | pN | | | | **1.3857** ← floor |
| 1/128 | pD | | | | **nan** ← floor |
| 1/256 | pD | | | | **−1.2801** ← floor |

**An order exists: p → 4.00** in the asymptotic range, and the triples stop
meaning anything below dt = 1/64.

**Richardson-extrapolated exact solution (p = 4, three finest grids):**
`pos_ned = [944.4786093043, 0.0000000000, −12184.5674676620]` m

Error at each dt against that extrapolated exact:

| dt | \|pos − exact\| [m] | order | × above the 7e-11 floor |
|---|---|---|---|
| 1/4 | 5.689713e-05 | | 8.13e+05 |
| 1/8 | 3.623199e-06 | 3.9730 | 5.18e+04 |
| 1/16 | 2.276324e-07 | 3.9925 | 3252 |
| 1/32 | 1.421778e-08 | 4.0009 | **203.1** |
| 1/64 | 8.590563e-10 | 4.0488 | 12.27 |
| 1/128 | 1.489298e-11 | 5.8500 | **0.213 — BELOW the floor** |
| 1/256 | 4.582863e-11 | **−1.6216** | 0.655 |
| 1/512 | 3.281298e-11 | 0.4820 | 0.469 |
| 1/1024 | 1.822539e-12 | 4.1702 | 0.026 |

**Above the floor?** Yes for dt ≥ 1/32 — the smallest fitted error is 203× the
floor, which is exactly what the test docstring claims. No for dt ≤ 1/128.

### (b) Smooth spatial wind field

`wind.LeeWave(w0 = 25 m/s, wavelength = 1200 m)` through the real
`wind.field_model` path (so `omega_gust` is the analytic gradient).

| h | comp | f_h | f_h/2 | f_h/4 | p_obs |
|---|---|---|---|---|---|
| 1/4 | pD | −12198.93463880849 | −12199.86349226990 | −12200.29708303769 | **1.0991** |
| 1/8 | pD | −12199.86349226990 | −12200.29708303769 | −12200.50573143383 | **1.0553** |
| 1/16 | pD | −12200.29708303769 | −12200.50573143383 | −12200.60796399174 | **1.0292** |
| 1/32 | pD | −12200.50573143383 | −12200.60796399174 | −12200.65855052647 | **1.0150** |
| 1/64 | pD | | | | **1.0076** |
| 1/128 | pD | | | | **1.0038** |
| 1/256 | pD | | | | **1.0019** |
| 1/256 | pN | | | | **0.9981** |

**An order exists, and it is 1, not 4.** p_obs converges monotonically to 1.00.

**Richardson-extrapolated exact solution (p = 1):**
`pos_ned = [943.2466404865, 0.0000000000, −12200.7087868896]` m

| dt | \|pos − exact\| [m] | order | × above the 7e-11 floor |
|---|---|---|---|
| 1/4 | 1.785791e+00 | | 2.55e+10 |
| 1/8 | 8.521594e-01 | 1.0674 | 1.22e+10 |
| 1/16 | 4.154308e-01 | 1.0365 | 5.94e+09 |
| 1/32 | 2.049974e-01 | 1.0190 | 2.93e+09 |
| 1/64 | 1.018141e-01 | 1.0097 | 1.45e+09 |
| 1/128 | 5.073703e-02 | 1.0048 | 7.25e+08 |
| 1/256 | 2.532793e-02 | 1.0023 | 3.62e+08 |
| 1/512 | 1.265584e-02 | 1.0009 | 1.81e+08 |
| 1/1024 | 6.327921e-03 | 1.0000 | 9.04e+07 |

**Above the floor by eight orders of magnitude at every dt.** Nothing here is
round-off.

**A measurement method note, recorded because it changes the numbers.** The
project's own `fixed_control_refinement` measures the error against a run at
dt_ref = 1/1024. For a first-order scheme that reference carries 6.3e-03 m of its
own error, which is 33 % of the dt = 1/512 error. The pairwise orders computed
that way therefore *rise* at the fine end (1.0730, 1.0479, 1.0421, 1.0570,
1.1044, 1.2249, 1.5862) instead of converging to 1. The Richardson triples above,
which use no reference at all, converge cleanly to 1.0019. Both are reported;
the project's fitted window (1/4 … 1/32, giving **1.053691**, reproducing the
test docstring's 1.0537 exactly) is not affected.

### (c) Rankine vortex core crossing

`wind.PARKS_CASES["hannibal"]`, one core at the aircraft's altitude,
r0 = 182.8800 m, v0 = 25.9080 m/s, start_north = −365.7600 m, so the core edge is
crossed at t ≈ 0.775 s and t ≈ 2.326 s. `d_elevator = 0.0`.

Triples:

| h | comp | p_obs |
|---|---|---|
| 1/4 | pN | **−0.4444** |
| 1/4 | pD | 0.6832 |
| 1/8 | pN | 1.7091 |
| 1/8 | pD | 1.3274 |
| 1/16 | pN | **nan** (sign flip in the difference) |
| 1/16 | pD | **nan** |
| 1/32 | pN | 1.5755 |
| 1/32 | pD | **nan** |
| 1/64 | pN | **nan** |
| 1/64 | pD | **−4.6763** |
| 1/128 | pN | 0.9823 |
| 1/128 | pD | 1.0002 |
| 1/256 | pN | **−0.3765** |
| 1/256 | pD | 3.0452 |

**No order exists.** Half the triples return nan because the difference changes
sign, and the ones that return a number return a different number each time.

The extrapolated "exact" solution from the three finest grids depends on the
order you assume, which is the definition of an extrapolation that means nothing:

| assumed p | exact pD [m] | exact pN [m] |
|---|---|---|
| 0.5 | −12185.526378 | 580.604912 |
| 1.0 | −12185.522055 | 580.606148 |
| 2.0 | −12185.520017 | 580.606730 |
| 4.0 | −12185.519202 | 580.606963 |

7.2 mm of spread in pD and 2.1 mm in pN purely from the choice of p. Separately,
the Richardson triples taken at different h give exact-pD values of −12184.29,
−12185.28 and −12185.49 m — **1.2 m** of spread.

Errors against the p = 1 extrapolation (only to show the shape; the reference is
not trustworthy):

| dt | \|pos − exact\| [m] | order |
|---|---|---|
| 1/4 | 3.036548e+00 | |
| 1/8 | 1.426875e+00 | 1.0896 |
| 1/16 | 4.242487e-01 | 1.7499 |
| 1/32 | 3.067406e-02 | 3.7898 |
| 1/64 | **7.169942e-02** | **−1.2249** ← refining made it worse |
| 1/128 | 6.959444e-02 | 0.0430 |
| 1/256 | 1.915000e-02 | 1.8616 |
| 1/512 | 6.358801e-03 | 1.5905 |
| 1/1024 | 3.179400e-03 | 1.0000 |

**Above the floor by 8–10 orders of magnitude at every dt** (4.5e+07 × the floor
at the finest). So the non-monotonicity is not round-off. Reproduced through the
project's own `fixed_control_refinement` with its own window
(dts 1/16…1/128, dt_ref 1/2048): errors `[0.43525628 0.03960198 0.08265195
0.08063046]`, monotone = **False**, min = 0.0396 m ≫ 1e-4 — the test's own
assertions hold.

### 2c. The round-off floor, measured directly

Four independent lines of evidence (`a3_floor_quat.py`, `a3b_floor_scaling.py`):

**(i) Machine resolution at the coordinate magnitude.**
```
np.spacing(12184.567)        = 1.818989e-12 m   (1 ulp, float64)
eps*|pD| = 2.22e-16*12184.6  = 2.705517e-12 m   <- the project's "2.7e-12" figure
np.spacing(944.4786)         = 1.136868e-13 m
```
Over n = 4096 steps (dt = 1/1024, 4 s), a random walk of half-ulps is
**5.821e-11 m**; the worst case is 3.7e-09 m.

**(ii) An exact invariance that isolates round-off.** In still air `pN` and `pE`
do not enter `derivatives` at all (only `pD`, through `density`). So offsetting
the start north by D must translate the answer by exactly D; any residue is pure
round-off. At dt = 1/1024, n = 4096:

| D [m] | ulp(D) | residue [m] | residue / ulp(D) |
|---|---|---|---|
| 1e+03 | 1.1369e-13 | 4.4338e-12 | 39.0 |
| **1.2192e+04** | 1.8190e-12 | **1.4438e-11** | 7.9 |
| 1e+05 | 1.4552e-11 | 4.0211e-10 | 27.6 |
| 1e+06 | 1.1642e-10 | 3.3377e-09 | 28.7 |
| 1e+07 | 1.8626e-09 | 3.8572e-08 | 20.7 |

The floor scales with the **magnitude of the accumulator**, at 8–39 ulp over 4096
steps.

**(iii) The refinement sequence bottoming out**, against a dt = 1/2048 reference:

| dt | \|pos − ref\| [m] |
|---|---|
| 1/64 | 9.384318e-10 |
| 1/128 | **8.264024e-11** |
| 1/256 | **3.644386e-11** |
| 1/512 | **4.913166e-11** |
| 1/1024 | **8.010817e-11** |

**Observed floor 3.6e-11 – 9.4e-11 m. The project's stated ~7e-11 m at 40,000 ft
is correct.**

**(iv) Perturbation amplification.** Perturbing the start north by 1 ulp
(1.137e-13 m) gives an output \|Δpos\| of exactly 1.137e-13 m over 512 steps —
no amplification. And the same call twice is bit-identical (`identical bits: True`).

---

## 3. Quaternion norm drift, with and without renormalisation

`integrate.step:116` is `new_state = new_state._replace(quat=quat_normalize(new_state.quat))`.
I wrote a **local copy of `step` in my own script** with that one line removed
(`a3_floor_quat.step_no_renorm`, `a3b_floor_scaling.step_no_renorm`), jitted the
same way. `flightsim/` was not touched.

### 3a. 747 at cruise, longitudinally + laterally excited (δe +0.02, δa +0.02), dt = 0.02

| steps | renorm ON: ‖q‖−1 | renorm OFF: ‖q‖−1 | \|Δpos\| [m] | \|Δquat\| | \|Δvel\| | \|Δω\| |
|---|---|---|---|---|---|---|
| 1 | 0 | 3.108624e-15 | 0 | 3.111e-15 | 0 | 0 |
| 10 | 0 | 3.508305e-14 | 1.062e-18 | 3.511e-14 | 1.789e-18 | 3.388e-21 |
| 100 | 0 | 2.193801e-13 | 1.273e-11 | 2.195e-13 | 3.130e-13 | 6.888e-17 |
| **1e3** | 0 | **2.224887e-13** | 4.290e-11 | 2.227e-13 | 1.081e-12 | 1.282e-15 |
| **1e4** | 0 | **2.218226e-13** | 2.245e-08 | 2.890e-13 | 2.024e-12 | 3.979e-15 |
| **1e5** | 0 | **2.129408e-13** | **3.301e-07** | 7.523e-12 | 6.198e-12 | 1.538e-14 |

**What renormalisation removes at dt = 0.02 is 2.1e-13 of norm error, and that
error is BOUNDED, not secular** — it saturates by 100 steps and does not grow
between 1e3 and 1e5 steps. `quat_normalize` returns `‖q‖−1 = 0.0` exactly.

**Does removing it change the trajectory measurably? Yes, but far below physical
significance at this dt:** 3.3e-7 m of position after 1e5 steps (2000 s).

### 3b. Torque-free tumbling at ‖ω‖ = 1.082 rad/s — the case that makes it secular

dt = 0.02:

| steps | ON ‖q‖−1 | OFF ‖q‖−1 | \|Δquat\| | \|Δω\| | \|Δpos\| [m] |
|---|---|---|---|---|---|
| 1 | 0 | −1.676e-14 | 1.677e-14 | 0 | 0 |
| 100 | 0 | 4.005e-11 | 4.005e-11 | 0 | 3.846e-09 |
| 1e3 | 2.220e-16 | −1.573e-12 | 1.573e-12 | 0 | 7.676e-08 |
| 1e4 | 0 | −7.277e-11 | 7.277e-11 | 0 | 6.788e-07 |
| **1e5** | 0 | **−1.092e-09** | 1.092e-09 | 0 | **5.768e-04** |

dt = 0.1:

| steps | ON ‖q‖−1 | OFF ‖q‖−1 | \|Δquat\| | \|Δω\| | \|Δpos\| [m] |
|---|---|---|---|---|---|
| 1 | 0 | −2.599e-10 | 2.599e-10 | 0 | 0 |
| 10 | 2.220e-16 | 5.611e-09 | 5.611e-09 | 0 | 9.272e-08 |
| 100 | 0 | 1.794e-08 | 1.794e-08 | 0 | 1.929e-05 |
| 1e3 | 0 | −1.409e-07 | 1.409e-07 | 0 | 7.053e-04 |
| 1e4 | 0 | −1.683e-06 | 1.683e-06 | 0 | 2.809e-01 |
| **1e5** | 0 | **−1.714e-05** | 1.714e-05 | 0 | **2.333e+02** |

At dt = 0.1 the drift **is** secular — 1.7e-10 per step, accumulating linearly to
1.7e-05 over 1e5 steps — and the trajectory difference reaches **233 m**.
Note `|Δω| = 0` identically: with no aerodynamics ω does not depend on q, so the
whole effect enters through the DCM.

### 3c. Why the norm error reaches the position

`state.quat_to_dcm` does **not** normalise. For a quaternion of norm 1+d
(measured on a general attitude, φ=0.3, θ=−0.2, ψ=0.7):

| d = ‖q‖−1 | det(DCM) | ‖Dv‖/‖v‖ | max\|DᵀD − I\| |
|---|---|---|---|
| 0 | 1.00000000000000 | 1.00000000000000 | 2.220e-16 |
| 1e-12 | 1.00000000000124 | 1.00000000000036 | 1.229e-12 |
| 1e-09 | 1.00000000124250 | 1.00000000035570 | 1.229e-09 |
| 1e-06 | 1.00000124250562 | 1.00000035570452 | 1.229e-06 |
| 1e-05 | 1.00001242533579 | 1.00000355711955 | 1.229e-05 |
| 1e-03 | 1.00124561125848 | 1.00035653010942 | 1.232e-03 |

Orthogonality error ≈ 1.23 d. Recorded as a measurement, not an explanation.

---

## 4. Conservation and invariance

### 4a. Torque-free angular momentum and rotational KE

Reproducing `test_conservation.test_free_rigid_body_conserves_angular_momentum_and_energy`
exactly (synthetic aircraft, Ixz = 200, every aero coefficient zeroed,
ω₀ = [0.5, 0.3, −0.2]):

| dt | T [s] | n | \|L\| drift (tol 1e-11) | dir drift [deg] (tol 3e-5) | KE drift (tol 2e-11) |
|---|---|---|---|---|---|
| **0.01** | **600** | **60000** | **5.6958e-13** | **1.4788e-06** | **9.3205e-13** |
| 0.02 | 600 | 30000 | 1.8334e-11 | 1.4788e-06 | 2.9341e-11 |
| 0.05 | 600 | 12000 | 1.7865e-09 | 1.4788e-06 | 2.8071e-09 |
| 0.01 | 6000 | 600000 | 5.7209e-12 | 1.4788e-06 | 8.8871e-12 |

PROJECT.md line 180 records 5.7e-13 / 1.5e-6 deg / 9.3e-13 — **all three
reproduce exactly.**

**Recorded discrepancy in a comment, not in the code:**
`test_conservation.py:82` says `assert mag_drift < 1e-11  # measured ~5.7e-14`.
The measured value is **5.6958e-13**, ten times the inline comment. PROJECT.md
line 180 has the correct 5.7e-13. The KE comment (`~9.3e-13`) is right. The
assertion passes either way; only the comment is off by a decade.

**dt sensitivity of the drift:** 0.05 → 0.02 → 0.01 gives 1.79e-9 → 1.83e-11 →
5.70e-13, ratios of 97.6 and 32.2 for dt ratios of 2.5 and 2 (2.5⁴ = 39, 2⁴ = 16).
Steeper than h⁴; recorded as measured. The **1e-11 tolerance in the test is only
1.8× above the drift at dt = 0.02** — the test is dt-fragile.

Independent run at ω₀ = [0.6, 0, 0.9], I = diag(1420, 4070, 4780), dt = 0.02,
6000 s / 300000 steps: max \|H\|/H₀ − 1 = **1.126312e-09**, max \|T/T₀ − 1\| =
**2.544548e-09**, inertial-H direction drift 5.484e-06 kg·m²/s on H₀ = 4385.6.

### 4b. Galilean invariance under a uniform steady wind

Same aircraft flown twice, once in still air and once with its ground velocity
offset by W and a uniform wind model returning W. 1000 steps at dt = 0.02.

| W [m/s NED] | max \|Δquat\| | max \|Δω\| [rad/s] | max \|pos − (pos_still + W t)\| [m] |
|---|---|---|---|
| [7, −3, 0] | 2.726118e-15 | 6.917210e-16 | **2.819434e-11** |
| [70, −30, 0] | 3.022756e-14 | 7.433941e-15 | **2.264642e-10** |
| **[0, 0, 2]** | **8.849986e-04** | **1.632554e-04** | **3.077563e+00** |

Horizontal wind: invariance holds to the round-off floor, and holds at 10×
strength (the residue grows by exactly 8×, i.e. with \|W\|). Vertical wind:
**fails by 3.08 m**, which is what the test's own docstring predicts (density is a
function of altitude, so the two aircraft do not see the same dynamic pressure).
Recorded as a measurement of the documented scope limit, not as a defect.

### 4c. Rotational invariance

Gravity is along NED +z, so **only rotations about z are a symmetry**. Rotating
the whole configuration (position by Rz, attitude by qz⊗q, body velocity and body
rates unchanged) must rotate the trajectory by Rz and leave body-frame quantities
untouched. 747 at cruise, aileron +0.015 and rudder −0.01 so the lateral channel
is excited, 40 s at dt = 0.02, on a position of magnitude 1.219e+04 m:

| ψ [rad] | max \|Δvel_body\| [m/s] | max \|Δω\| [rad/s] | max \|pos − R pos\| [m] | max \|quat − qz q\| |
|---|---|---|---|---|
| 0.7 | 5.684342e-14 | 2.498002e-16 | **6.808477e-09** | 1.831868e-15 |
| 2.5 | 5.684342e-14 | 3.261280e-16 | **1.906119e-08** | 9.992007e-16 |
| −1.9 | 1.705303e-13 | 3.747003e-16 | **1.398314e-08** | 3.913536e-15 |

**Control that the test is not vacuous:** the same rotation applied about NED x
(which is *not* a symmetry, because gravity points along z) gives
max \|Δvel_body\| = **9.197289e+00 m/s**.

**With the wind field rotated too** — two Rankine vortices at
`north = [300, 1000] m`, `down = [−H, −H+200] m`, field replaced by
`Rz · field(Rzᵀ p)` — 30 s at dt = 0.02:

| ψ [rad] | max \|Δvel_body\| | max \|Δω\| | max \|pos − R pos\| [m] | max \|quat − qz q\| |
|---|---|---|---|---|
| 0.7 | 2.4880e-11 m/s | 1.9995e-13 rad/s | **1.0597e-09** | 1.2795e-13 |
| −2.1 | 1.5857e-10 m/s | 5.8106e-13 rad/s | **2.4751e-09** | 3.9136e-13 |

Rotational invariance holds with a spatial wind field, including through the
`gust_rates` jacobian.

### 4d. Energy over a long horizon

**Trimmed and powered, still air, 500000 steps = 10000 s:**
`dE/E0 = +0.000000e+00`, altitude 12192.0000 → 12192.0000 m, \|V\| 235.915200 →
235.915200 m/s, max \|E−E0\|/E0 over the run = **0.000000e+00**. The trim residual
is 1.9e-20, so the state is an exact fixed point of the (vel, ω, quat) subsystem
and pD never moves — this says the fixed point is exactly held, and nothing about
the integrator's energy behaviour.

**A case where energy is a real invariant.** Drag-free (CD0 = 0, e = 1e12),
thrust-free, moment-free aircraft, lift only. Lift in body axes is
[L sin α, 0, −L cos α] and vel_rel is V[cos α cos β, sin β, sin α cos β]; the dot
product is exactly zero, so lift does no work and KE + PE must be conserved.
Initial state perturbed off trim so the flight path oscillates over 121.5 m of
altitude:

| dt | n | T [s] | max\|E−E0\|/E0 | final dE/E0 |
|---|---|---|---|---|
| 0.02 | 50000 | 1000 | **1.4593e-11** | +1.4304e-11 |
| 0.02 | 500000 | 10000 | **1.4593e-11** | +1.4304e-11 |
| 0.002 | 500000 | 1000 | **5.7259e-13** | −5.7259e-13 |
| 0.1 | 100000 | 10000 | **1.0068e-08** | +1.0068e-08 |

**Bounded, not secular** — identical at 1000 s and 10000 s. The dt scaling
0.1 → 0.02 is 1.0068e-08 / 1.4593e-11 = **690**, against 5⁴ = 625, i.e. O(dt⁴).

**Power-off descent, recorded as observed:** throttle 0, 100000 steps = 2000 s
takes the aircraft from 12192.0000 m to **−3994.3118 m** (below sea level) with
\|V\| 235.90 → 97.59 m/s. `atmosphere.density` extrapolates the troposphere
formula there without complaint: at h = −3994 m, T = 314.111 K, p = 159450.4 Pa,
ρ = 1.768400 kg/m³. The formula has T → 0 at h = −44330.8 m.

---

## 5. Trim convergence and uniqueness

### 5a. Convergence rate

`verification.newton_residual_history` from `trim.INITIAL_GUESS = [0.05, 0.0, 0.5]`:

`|residual|` after each iteration:

| iter | boeing747 | boeing747_approach | cherokee | cessna172 |
|---|---|---|---|---|
| 0 | 2.297995e+00 | 2.582178e+00 | 4.411529e+00 | 3.869704e+00 |
| 1 | 4.277288e-02 | 8.423831e-02 | 1.210741e-01 | 4.627013e-02 |
| 2 | **8.335234e-08** | **8.787578e-07** | **2.899691e-06** | **2.146039e-12** |
| 3 | **5.333706e-15** | **3.568316e-15** | **5.555017e-15** | 2.414396e-16 |
| 4 | 1.779906e-15 | 1.790203e-15 | 1.778227e-15 | 1.665335e-16 |
| 5 | 1.776359e-15 | 4.461263e-18 | 1.205633e-16 | 5.551115e-17 |
| 6 | 1.779824e-15 | 1.858860e-19 | 8.803722e-17 | 5.551115e-17 |
| 7–10 | ~1.78e-15 (flat) | **0.000000e+00** | 8.80e-17 (flat) | 5.55e-17 (flat) |

Log-residual exponent ratios (2 = quadratic): boeing747 **3.3004**,
boeing747_approach **3.3513**, cherokee **2.9591**, cessna172 **5.3755**. All
are ≥ 2, and they are inflated *above* 2 because the residual crashes into the
float64 floor inside the measured window — the shape of a Newton solve whose
Jacobian is the exact derivative of the residual it evaluates.
`test_the_trim_solve_converges_quadratically` asserts max ratio > 1.6 and passes.

**How many iterations does `trim.trim` actually need?** It runs a fixed 40 inside
`lax.scan`:

| iterations= | boeing747 alpha | boeing747 \|res\| | cherokee alpha | cherokee \|res\| |
|---|---|---|---|---|
| 1 | +0.080963596021 | 4.277e-02 | +0.000461544786 | 1.211e-01 |
| 2 | +0.080917499320 | 8.335e-08 | +0.000216056252 | 2.900e-06 |
| **3** | **+0.080917499226** | 5.334e-15 | **+0.000216050230** | 5.540e-15 |
| 4 | +0.080917499226 | 3.331e-16 | +0.000216050230 | 1.776e-15 |
| 5 | +0.080917499226 | 2.920e-19 | +0.000216050230 | 1.709e-16 |
| **6 … 200** | +0.080917499226 | **1.933e-20** | +0.000216050230 | **9.543e-18** |

**The solution is converged to the last printed bit at iteration 3 for every
registry aircraft; the residual bottoms out by iteration 6.** The fixed 40 is
7–13× more than needed. `iterations=200` returns the identical answer, so the
extra iterations neither help nor diverge — they are pure cost.

**Jacobian conditioning at the nominal root** (`jacfwd(trim.residual)`):

| aircraft | cond(J) |
|---|---|
| boeing747 | 8.5374e+01 |
| cherokee | 5.4216e+01 |
| cessna172 | 1.3981e+02 |

Well conditioned. Row 3 of J has a structural zero in the throttle column
(thrust produces no pitching moment in this model), which is not a rank problem.

### 5b. UNIQUENESS — swept over 2090 initial guesses per aircraft

Grid: 19 α ∈ [−90°, +90°] × 11 δe ∈ [−0.6, +0.6] rad × 10 throttle ∈ [−1, +5],
each aircraft at its own CRUISE condition, `iterations=40`. Roots clustered after
folding α into (−π, π] (`trimmed_state` uses cos α / sin α / `euler_to_quat`, so
α and α + 2πk are the *same physical state* and must not be counted twice).

**Every one of the 2090 starts converged** (`|residual| < 1e-9`, no NaN) for all
four aircraft.

| aircraft | distinct roots | basin of the physical root | basin of absurd roots |
|---|---|---|---|
| boeing747 | **2** | 1870 / 2090 (89.5 %) | **220 / 2090 (10.5 %)** |
| boeing747_approach | **2** | 1980 / 2090 (94.7 %) | **110 / 2090 (5.3 %)** |
| cherokee | **3** | 1910 / 2090 (91.4 %) | **180 / 2090 (8.6 %)** |
| cessna172 | **3** | 1810 / 2090 (86.6 %) | **280 / 2090 (13.4 %)** |

The roots, with the residual re-evaluated independently at the folded α:

**boeing747** (V = 235.915 m/s, H = 12192 m)

| root | α [deg] | δe [rad] | throttle | basin | \|residual\| | verdict |
|---|---|---|---|---|---|---|
| 0 | **+4.6362** | −0.000443 (−0.025°) | **+0.737647** | 1870 | 1.421e-14 | PHYSICAL |
| 1 | **−118.1008** | **+1.500365 (+85.965°)** | **−193.686533** | 220 | 3.163e-14 | ABSURD: α > 15°; \|δe\| > 0.436 rad limit; throttle ∉ [0,1] |

**boeing747_approach** (V = 84.883 m/s, H = 0 m)

| root | α [deg] | δe [rad] | throttle | basin | \|residual\| | verdict |
|---|---|---|---|---|---|---|
| 0 | **+5.6219** | +0.001282 (+0.073°) | **+0.296129** | 1980 | 2.665e-14 | PHYSICAL |
| 1 | **−116.0927** | **+1.998777 (+114.521°)** | **−37.123364** | 110 | 8.926e-15 | ABSURD |

A further root at α = **+179.3399°, throttle = −4124.4661** was reached from the
α = −75° start in the boundary probe (outside the clustering grid).

**cherokee** (V = 50 m/s, H = 1499.6 m)

| root | α [deg] | δe [rad] | throttle | basin | \|residual\| | verdict |
|---|---|---|---|---|---|---|
| 0 | **+0.0124** | −0.000067 | **+0.658771** | 1910 | 1.620e-14 | PHYSICAL |
| 1 | **+149.5490** | **−0.809132 (−46.360°)** | **−241.812685** | 150 | 3.081e-13 | ABSURD |
| 2 | **−130.1754** | +0.704311 (+40.354°) | **−139.638268** | 30 | 2.403e-13 | ABSURD |

**cessna172** (V = 60 m/s, H = 1524 m)

| root | α [deg] | δe [rad] | throttle | basin | \|residual\| | verdict |
|---|---|---|---|---|---|---|
| 0 | **+1.8171** | −0.035072 (−2.009°) | **+0.820063** | 1810 | 1.607e-14 | PHYSICAL |
| 1 | **−114.9094** | **+1.390704 (+79.681°)** | **−290.676438** | 260 | 1.381e-13 | ABSURD |
| 2 | **+154.9861** | **−1.905978 (−109.205°)** | **−818.005775** | 20 | 1.419e-12 | ABSURD |

**Basins.** The physical root's basin is a wide contiguous interval around the
default guess. Sweeping only α with δe = 0, throttle = 0.5:

- boeing747: the physical root is reached for **every guess α from −55° to +90°**;
  −90°, −85°, −80° and −60° go to the absurd root. The nearest absurd basin to
  the default guess (α = 2.865°) is at **−60°**.
- boeing747_approach: physical for −70° and −60° … +90°; absurd at −90°, −85°,
  −80°, −75° (a *third* root), −65°.
- cessna172: physical for −70…−55°, −45…+50°, +65…+70°; **absurd at −50°**, and
  from +55° upward.
- cherokee: physical for −85°, −75°, −60° … +70°, +90°; absurd at −90°, −80°,
  −65°, +75°, +80°, +85°. The basin structure is **interleaved** at the extremes.

**All absurd roots converge to machine precision** (1e-14 … 1e-12) and pass any
residual test. `trim.is_physical` catches every one of them **via the α bound
alone**. Note that it checks α only: the elevator deflections of 86°, 114.5°,
−109° all exceed the airframe's own `elevator_limit = ±0.436 rad`, and the
throttles of −37 … −818 are outside [0, 1], and neither is checked anywhere.

`trim.trim` called with its default guess reaches the physical root for all four
aircraft (`is_physical = True` in every case).

**Recorded scope difference from the project's own account.** PROJECT.md §7
records absurd roots for **degenerate coefficients** (CLa = 1e-4 gives −632.1°).
The measurement above is with the **unmodified registry data at each aircraft's
own cruise condition** and still finds 2–3 distinct roots and 5–13 % of a wide
guess grid landing on an absurd one.

---

## 6. Floating point

### 6a. Where x64 is set, and whether it is set in time

`flightsim/__init__.py`, lines 8–10, and nowhere else in the package:

```python
import jax

jax.config.update("jax_enable_x64", True)
```

`flightsim/tests/conftest.py:7` has `import flightsim  # noqa: F401  -- enables x64
before any array is made`, and line 13 adds `jax.config.update("jax_debug_nans", True)`.

Measured:

```
x64 BEFORE import flightsim: False
x64 AFTER  import flightsim: True
dtype of jnp.zeros(3):       float64
```

**Does a user outside pytest get x64? YES.** Both `import flightsim` and
`from flightsim import integrate` (which triggers the package `__init__`) turn it
on, and any array created afterwards is float64.

**The one ordering hazard, measured:**
```
a = jnp.ones(3)      # before importing flightsim
  -> float32
import flightsim     # x64 now True
b = jnp.ones(3)      -> float64
a is still           -> float32
```
An array made before the import stays float32 and silently down-promotes anything
it touches. Nothing in `flightsim` can prevent this; recorded as a measurement.

**Related observation:** `trim.INITIAL_GUESS` is a module-level `jnp.array`
created at import time, so its dtype is frozen there. Calling `trim.trim` after
flipping x64 off raises
`TypeError: scan body function carry input and carry output must have equal types
... input carry x has type float64[3] but the corresponding output carry component
has type float32[3]`. In a genuinely-float32 process this would not arise; it is
recorded because it is what makes a mid-process precision experiment need an
explicit `guess=`.

### 6b. What float32 costs

x64 turned off, every `Aircraft` leaf cast to float32, the trim solved in float32
with a float32 guess, and the plant matrix built by `jacfwd` in float32
(`A_lon dtype float32`, then `np.linalg.eigvals` on a float32 array → complex64).
float32 eps = 1.19e-07.

**Modes (the lightly damped ones are the ones asked for):**

| aircraft | mode | float64 | float32 | rel err | abs err |
|---|---|---|---|---|---|
| **boeing747** | **phugoid ζ** | **0.0559555008** | **0.0559555590** | **1.041e-06** | **+5.826e-08** |
| boeing747 | phugoid ωn | 0.0553189351 | 0.0553189293 | 1.058e-07 | |
| **boeing747** | **Dutch roll ζ** | **0.0360852417** | **0.0360852480** | **1.756e-07** | **+6.335e-09** |
| boeing747 | Dutch roll ωn | 0.9432021594 | 0.9432021976 | 4.047e-08 | |
| boeing747 | short-period ζ | 0.3425258602 | 0.3425258994 | 1.144e-07 | +3.920e-08 |
| boeing747 | short-period ωn | 0.9507725037 | 0.9507725239 | 2.124e-08 | |
| boeing747 | roll τ | 1.79536570 s | 1.79536545 s | 1.384e-07 | |
| boeing747 | spiral τ | 138.04240755 s | 138.04238892 s | 1.350e-07 | |
| **747 approach** | **phugoid ζ** | **0.0126848560** | **0.0126848817** | **2.026e-06** | **+2.571e-08** |
| 747 approach | short-period ζ | 0.5911203154 | 0.5911203027 | 2.158e-08 | |
| cherokee | phugoid ζ | 0.0660113530 | 0.0660113543 | 2.012e-08 | |
| cessna172 | phugoid ζ | 0.0665101668 | 0.0665101632 | 5.397e-08 | |

**Every error is between 1e-8 and 2e-6 relative, i.e. 0.1× to 17× float32 eps.**
The 747 phugoid ζ = 0.056 loses ~6 significant figures, ending at 5.8e-08
absolute. `cond(A_lon)` is 7.97e+05 for the 747, 5.76e+04, 7.71e+03, 1.31e+04 for
the others, and comes out identical to 7 digits in both precisions.

**Trim in float32:**

| aircraft | Δα [rad] | Δδe | Δthrottle | float32 \|residual\| | float64 \|residual\| |
|---|---|---|---|---|---|
| boeing747 | −2.907e-08 | +2.425e-08 | −1.002e-07 | **2.3024e-12** | 1.9330e-20 |
| 747 approach | −7.903e-09 | +2.551e-08 | +4.715e-08 | **9.3596e-12** | 5.8794e-20 |
| cherokee | −1.081e-08 | +3.354e-09 | −6.571e-08 | **1.5600e-08** | 9.5426e-18 |
| cessna172 | +7.722e-10 | −5.020e-09 | −3.688e-08 | **9.6712e-07** | 1.7830e-15 |

The *solution* is accurate to 1e-8 rad in α; the *residual* stalls 8–13 orders
higher. Any caller that gates on `TRIM_RESIDUAL_LIMIT = 1e-9`
(`validation.sweep`) would **reject the Cessna's perfectly good float32 trim**
(9.67e-07 > 1e-09).

**Rollout in float32**, same initial condition and controls, 747 cruise, dt = 0.02:

| steps | t [s] | \|Δpos\| [m] | \|Δvel\| [m/s] | \|Δquat\| | \|Δω\| | f32 ‖q‖−1 |
|---|---|---|---|---|---|---|
| 1000 | 20 | 4.392857e-03 | 1.752786e-04 | 2.112e-07 | 2.291e-08 | +8.537e-08 |
| 10000 | 200 | 2.609719e-01 | 7.893458e-04 | 5.264e-07 | 2.838e-07 | +1.179e-08 |
| 50000 | 1000 | 6.148903e-01 | 1.523770e-04 | 4.856e-06 | 1.179e-07 | −2.129e-08 |
| **100000** | **2000** | **4.985854e+02** | 8.570398e-03 | 7.309e-06 | 3.041e-06 | +3.471e-08 |

float32 ulp at pD ≈ 12192 m is **9.7656e-04 m**, and at pN ≈ 5.07e+05 m it is
0.031 m. The 499 m at 2000 s is a phugoid phase divergence, not a per-step error.
The quaternion norm stays at 1e-8 – 1e-7 (sub-ulp of 1.0 in float32) **because
`quat_normalize` runs every step**; with renormalisation removed in float32 the
norm error would start at 1e-7 per step.

---

## 7. Does `jit` change results?

`jax.disable_jit()` against the ordinary jitted path (`a7_jit.py`). Compared by
SHA-256 of the raw bytes, then by max absolute difference.

| what | bit-identical? | max \|diff\| |
|---|---|---|
| `trim.trim` solution x | **No** | 1.645e-17 (‖Δx‖) |
| `trim.trim` residual | **No** | 1.780e-15 |
| one `integrate.step`, `pos_ned` | Yes | 0 |
| one `integrate.step`, `vel_body` | Yes | 0 |
| one `integrate.step`, `quat` | Yes | 0 |
| one `integrate.step`, `omega` | **No** | 5.421e-20 |
| 1000-step rollout, `pos_ned` | **No** | **4.547e-13 m** |
| 1000-step rollout, `vel_body` | **No** | 3.553e-15 |
| 1000-step rollout, `quat` | **No** | 1.110e-16 |
| 1000-step rollout, `omega` | **No** | 1.128e-17 |
| 1000-step rollout through the lee-wave field, `pos_ned` | **No** | 4.547e-13 m |
| 1000-step rollout through the lee-wave field, `vel_body` | **No** | 3.997e-14 |
| `validation.longitudinal_matrix` (A_lon) | **No** | 1.388e-17 |
| `longitudinal_modes` (ωn, ζ) | **Yes** — every float compares `==` | 0 |
| `lateral_modes` (Dutch roll, roll τ, spiral τ) | **Yes** — tuple compares `==` | 0 |

Final positions printed to 8 decimals are indistinguishable:
`jit [4809.41371101, 0, −11899.72775483]`,
`eager [4809.41371101, 0, −11899.72775483]`.

**So jit does move bits, at the last-ulp level, and the 1000-step amplification
(4.5e-13 m on a 12000 m coordinate, 4e-17 relative) sits at the round-off floor
measured in §2c.** Note `test_extracting_rk4_step_did_not_move_a_single_bit`
compares a hash of the **jitted** path, and that hash still matches on this
platform (§10); it would not match an eager run.

**Three internal paths that ARE bit-identical:**

| comparison | result |
|---|---|
| `rollout` vs `logged_rollout` final state | **identical** |
| `rollout` vs a hand Python loop of `step` | **identical**, max \|Δpos\| = 0 |
| `batched_rollout` (vmap, batch 4) member 0 vs single `rollout` | **identical**, max \|Δpos\| = 0 |

---

## 8. Other numerically load-bearing findings

### 8a. `validation.lateral_modes` mislabels roll and spiral when the spiral is unstable

`flightsim/validation.py:159-160`:

```python
reals.sort()  # roll subsidence is fast (small tau), spiral is slow
roll_tau, spiral_tau = reals[0], reals[1]
```

`reals` holds `−1/λ` for the two real lateral roots. The sort is on the **signed**
τ, so a **negative** τ (unstable root) always sorts first and is labelled "roll".

Measured over the registry:

| aircraft | real lateral eigenvalues | τ = −1/λ | `lateral_modes` returns | fastest \|τ\| |
|---|---|---|---|---|
| boeing747 | [−0.55698959, −0.00724415] | [1.795366, 138.042408] | roll 1.795366, spiral 138.042408 | 1.795366 ✔ |
| 747 approach | [−1.21269133, −0.03677222] | [0.824612, 27.194444] | roll 0.824612, spiral 27.194444 | 0.824612 ✔ |
| **cherokee** | **[−2.78142966, +0.01938326]** | **[0.359527, −51.5909]** | **roll −51.590900, spiral 0.359527** | **0.359527 ✘** |
| cessna172 | [−11.34124338, −0.06005444] | [0.088174, 16.651559] | roll 0.088174, spiral 16.651559 | 0.088174 ✔ |

**The Cherokee's spiral root is unstable (λ = +0.0194 /s, τ = −51.6 s) and the
function returns it as `roll_tau`, while returning the true roll subsidence
(0.3595 s) as `spiral_tau`.** No shipped test exercises the Cherokee's lateral
modes, so the suite does not go red. `test_cr2144_modes.py:66` asserts
`spiral_tau > 0` for the 747 approach and `test_navion.py:154-156` asserts
`0.03 < roll_tau < 1.0` and `abs(spiral_tau) > 15.0` — both aircraft have a
stable spiral, so neither test can see this.

### 8b. Peak load factor, the project's headline quantity, vs dt

`logged_rollout` + `dynamics.load_factor`, exactly as `vortex_viz._measure` does.
Parks 5-vortex array (hannibal), 747 cruise, fixed controls, 60 s:

| dt | n_z max | n_z min | err vs 1/4096 ref (max) | err (min) |
|---|---|---|---|---|
| 0.05 | 1.994110871 | −0.317773344 | −2.955e-02 | +1.983e-02 |
| **0.02** | 2.010722452 | −0.313627015 | **−1.294e-02** | +2.398e-02 |
| **0.01** ← `scripts/vortex.py` default | 2.014794797 | −0.333736925 | **−8.870e-03** | +3.869e-03 |
| 0.005 | 2.015275772 | −0.333712562 | −8.389e-03 | +3.893e-03 |
| 0.0025 | 2.019358087 | −0.333691211 | −4.306e-03 | +3.915e-03 |
| 0.00125 | 2.021781523 | −0.336205943 | −1.883e-03 | +1.400e-03 |
| 1/4096 | **2.023664370** | **−0.337605886** | | |

At the production dt = 0.01 the peak n_z is low by **0.0089 g**, about **0.9 % of
the 1.02 g peak excursion**, and refining from 0.01 to 0.005 improves it by only
5 % (−8.870e-03 → −8.389e-03).

Same measurement on the **smooth** lee-wave field (no kink), which isolates the
O(dt) hold:

| dt | n_z max | err vs ref |
|---|---|---|
| 0.05 | 1.028010891 | −1.940e-04 |
| 0.02 | 1.028126604 | −7.828e-05 |
| 0.01 | 1.028166733 | −3.815e-05 |
| 0.005 | 1.028186277 | −1.860e-05 |
| 1/4096 | 1.028204880 | |

Clean halving (first order), and negligible in absolute terms.

### 8c. Full-state error at the timesteps the project's scripts actually use

`scripts/fly.py` and `scripts/checkpoint.py` use dt = 0.02;
`scripts/leewave.py`, `scripts/microburst.py`, `scripts/vortex.py` use dt = 0.01.
Reference dt = 1/4096, 60 s, 747 cruise, sourced field amplitudes:

**Lee wave (Doyle, w0 = 6 m/s, λ = 25 km):**

| dt | \|Δpos\| [m] | \|Δvel\| [m/s] | \|Δquat\| | \|Δω\| |
|---|---|---|---|---|
| 0.02 | 6.56076e-02 | 4.24347e-03 | 4.27114e-06 | 9.56445e-07 |
| **0.01** | **3.23919e-02** | 2.09473e-03 | 2.10851e-06 | 4.82128e-07 |
| 0.005 | 1.57891e-02 | 1.02096e-03 | 1.02771e-06 | 2.37424e-07 |
| 0.0025 | 7.48890e-03 | 4.84232e-04 | 4.87436e-07 | 1.13185e-07 |

Clean first-order halving.

**Parks vortex array (hannibal, 5 cores):**

| dt | \|Δpos\| [m] | \|Δvel\| [m/s] | \|Δquat\| | \|Δω\| |
|---|---|---|---|---|
| 0.02 | 1.48526e+00 | 3.30237e-02 | 1.27191e-04 | 1.62259e-05 |
| **0.01** | **5.38899e-01** | 8.50507e-03 | 4.78007e-05 | 3.92764e-06 |
| **0.005** | **6.08485e-01** ← *worse* | 8.44534e-03 | 5.43380e-05 | 3.96175e-06 |
| 0.0025 | 2.41959e-01 | 4.18858e-03 | 2.00256e-05 | 1.81727e-06 |

Non-monotone at production settings.

**Updraft column (Wingrove & Bach, w0 = 24.4 m/s, r = 2400 m, sharpness 4):**

| dt | \|Δpos\| [m] |
|---|---|
| 0.02 | 7.56769e-02 |
| 0.01 | 3.73789e-02 |
| 0.005 | 1.82237e-02 |
| 0.0025 | 8.64457e-03 |

### 8d. The Rankine kink, measured

`ASSUMPTIONS.md` E2 claims the one-sided derivatives differ by 2·V0/r0 with
opposite signs. Measured by `jacfwd` either side of r = r0 (± 1e-6 m):

```
d(w_horiz)/d(z) inside  = -0.14166667 1/s
d(w_horiz)/d(z) outside = +0.14166667 1/s
jump                    = +0.28333333 1/s ;  2*v0/r0 = 0.28333333 1/s
```

**Exactly 2·V0/r0, with opposite signs.** The claim reproduces to 8 digits.

### 8e. `trim.trim` has no bounds; `is_physical` checks α only

`trim.ALPHA_LIMIT` = 15.0 deg. `is_physical` tests `abs(x[0])` and nothing else.
The airframes carry `elevator_limit`, `aileron_limit`, `rudder_limit` (all
±0.436 rad for the registry) and the throttle is physical on [0, 1]; **neither is
bounded in the solve nor checked afterwards.** Every absurd root in §5b violates
both. `is_physical` also uses `bool(float(...))`, so it cannot be called under
`jit`/`vmap` — which the docstring states.

Separately, `is_physical` would **reject a legitimately converged root reported as
α + 360°** (e.g. the 747's +4.6362° also appears at +364.6362° from some starts),
because it does not fold α into (−π, π].

### 8f. `minimum_drag_speed` grid resolution and where V_md sits

`jnp.linspace(20, 400, 4000)` → resolution **0.095024 m/s**, i.e. the returned
V_md is quantised to ±0.05 m/s.

| aircraft | V_md [m/s] | cruise [m/s] | margin |
|---|---|---|---|
| boeing747 | 233.423356 | 235.915 | +2.49 |
| **boeing747_approach** | **97.064266** | **84.883** | **−12.18 (BELOW V_md)** |
| cherokee | 47.271818 | 50.000 | +2.73 |
| cessna172 | 38.434609 | 60.000 | +21.57 |

`boeing747_approach` at its declared condition is **on the back side of the drag
curve**, which `trim.minimum_drag_speed`'s own docstring says is where the
throttle/elevator loop pairing stops being the right assignment. Recorded as a
measurement; whether it matters is a physics question, not a numerics one.

### 8g. Smaller observations

- `verification.fitted_order` raises `ValueError: an error is zero or negative;
  the sequence is saturated` on a zero error — verified. So a saturated refinement
  fails loudly rather than returning a wrong slope. Good.
- `validation.modes_from_matrix` selects oscillatory roots with
  `lam.imag > 1e-9`, an **absolute** threshold in rad/s. A mode below 1e-9 rad/s
  would be silently reported as two real roots, and `lateral_modes` would then
  index `reals[0], reals[1]` on a list of four and mislabel; with fewer than two
  reals it would `IndexError`. Not triggered by any registry aircraft.
- `aero.V_MIN = 1.0 m/s` floors the airspeed in `air_data`, so α, β and all
  non-dimensional rates are guarded against V → 0. Never binds at flight speeds.
- The simulation is **deterministic**: the same call twice returns bit-identical
  results (`identical bits: True`).
- `verification.free_fall_through_a_swinging_wind` (747, aero zeroed, dt = 0.02,
  n = 300): `max_position_error = 3.979039e-12 m` against a closed form,
  `peak_wind = 29.461817 m/s`, `peak_dwdt = 88.385519 m/s²`, `elapsed = 6.0 s`.
  The test's 1e-9 m bound holds with 250× margin.

---

## 9. Reproduction of the project's own published numbers

| claim | source | measured here | verdict |
|---|---|---|---|
| RK4 order, oscillator = 4.00 ± 0.05 | test_verification | **3.999819** | ✔ |
| 6-DOF still-air order = 3.98913 | PROJECT.md §4, l.555 | **3.989127** | ✔ exact |
| smooth-field order = 1.0537 | test_verification docstring | **1.053691** | ✔ exact |
| Rankine core: non-monotone, min > 1e-4 | test_verification | errors `[0.4353, 0.0396, 0.0827, 0.0806]`, monotone = False, min = 0.0396 | ✔ |
| pairwise 1/4→1/8 = 3.973 | test docstring | **3.9730** | ✔ |
| pairwise 1/8→1/16 = 3.993 | test docstring | **3.9925** | ✔ |
| pairwise 1/16→1/32 = 4.008 | test docstring | **4.0008** | ✔ |
| pairwise 1/32→1/64 = 4.167 | test docstring | 4.0460 | ✘ platform variance (round-off region) |
| pairwise 1/64→1/128 = 3.420 | test docstring | 5.8532 | ✘ same |
| pairwise 1/128→1/256 = −0.685 | test docstring / PROJECT.md l.1392 | −1.5634 | ✘ same (sign and conclusion agree) |
| wide-window fitted slope = 3.82 | test docstring | 3.7406 | ✘ same |
| round-off floor ≈ 7e-11 m at 40,000 ft | PROJECT.md l.618, ASSUMPTIONS F4 | **3.6e-11 – 9.4e-11 m** | ✔ |
| pos_ned resolved to 2.7e-12 m | PROJECT.md l.1390 | eps·\|pD\| = **2.7055e-12 m** (true ulp is 1.819e-12) | ✔ |
| angular-momentum drift 5.7e-13 / 60,000 steps | PROJECT.md l.180 | **5.6958e-13** | ✔ |
| direction drift 1.5e-6 deg | PROJECT.md l.181 | **1.4788e-06 deg** | ✔ |
| rotational KE drift 9.3e-13 | PROJECT.md l.182 | **9.3205e-13** | ✔ |
| `# measured ~5.7e-14` | test_conservation.py:82 comment | **5.6958e-13** | ✘ **comment off by 10×** (PROJECT.md is right) |
| Rankine kink = 2·V0/r0, opposite signs | ASSUMPTIONS E2 | 0.28333333 vs 0.28333333 | ✔ |
| `PRE_REFACTOR_VEL_HASH` bit-identity | test_verification | `bbc0323e…6aa4` **MATCH: True** | ✔ |
| free-fall through swinging wind < 1e-9 m | test_verification | 3.979e-12 m | ✔ |

---

## 10. Test suite status

`./.venv/Scripts/python.exe -m pytest flightsim/tests -q -x --no-header`

```
444 passed, 1 skipped in 460.98s (0:07:40)
```

That run collected **445 tests**. Scope note: an untracked
`flightsim/tests/test_audit_regression.py` (873 lines, 73 further tests) appeared
in the working tree **during** this session — it is not mine and it landed after
my collection, so the 444/1 result covers the pre-existing suite only. Current
collection is 518 with it, 445 without it.

**The pre-existing suite is green on this machine.** No tolerance, test or
reference value was modified by me. Every finding above that is a *defect*
(§8a, the roll/spiral swap; §5b, the multiple trim roots; §9, the 10× comment
slip) is therefore something the suite does not currently assert against, not a
test I broke.

---

## 11. What I could NOT check, and why

1. **GPU / TPU behaviour.** Only `CpuDevice(id=0)` is available here. Every
   bit-identity result in §7 and §9 (including `PRE_REFACTOR_VEL_HASH`) is a
   CPU/XLA-CPU statement and says nothing about another backend, where fused
   multiply-add and different reduction orders would almost certainly move the
   hash.
2. **A genuinely float32-from-process-start run.** `flightsim/__init__.py`
   unconditionally sets x64 True, so I flipped it off mid-process and cast the
   registry down (§6b). That reproduces float32 *arithmetic* faithfully but not
   a float32 *process*: constants folded at import time (notably
   `trim.INITIAL_GUESS` and `atmosphere.P_TROPOPAUSE`) were computed in float64
   and then rounded, whereas a true float32 process would compute them in
   float32. The difference is at most 1 ulp in each constant; I did not bound its
   effect on the modes.
3. **Whether the absurd trim roots are reachable from any *realistic* automated
   caller.** I swept `guess=` directly. `minimum_drag_speed`, `validation.sweep`
   and the autopilot all use the default guess or a nearby one, and I did not
   enumerate every call site to check none of them can wander into an absurd
   basin. The default guess reaches the physical root for all four aircraft.
4. **Order of accuracy of the *closed-loop* (autopilot) rollout.** Everything
   above is fixed-control. `autopilot.closed_loop_rollout` adds a discrete
   controller update, which is a second per-step hold and could impose its own
   order limit. Not measured.
5. **Turbulence / stochastic wind.** `wind.WindState` is empty and every field
   shipped is deterministic, so I could not measure whether the once-per-step
   wind sample behaves correctly for a shaped-noise model — the model does not
   exist yet.
6. **Whether the 1e-11 tolerance in `test_conservation` is safe across
   platforms.** I measured the drift at dt = 0.02 to be 1.83e-11, i.e. *above*
   the tolerance, but the test runs at dt = 0.01 where it is 5.70e-13. I did not
   attempt to establish platform variance for the dt = 0.01 case.
7. **`loads.py` / strip-integration numerics.** `strip_roll_moment` and
   `_strip_rolling_coefficient` use `jnp.trapezoid` over `airframe.stations`;
   I did not measure the quadrature convergence of the strip integration, which
   is a separate order-of-accuracy question from the time integrator.
8. **Long-horizon behaviour past 1e5 steps** for the quaternion experiments.
   1e5 steps per configuration, run as a Python loop so the intermediate norms
   could be read, already cost most of the wall-clock budget.
9. **`float64` vs `longdouble`.** No independent higher-precision reference
    exists in this environment, so every "exact" value in §2 is a Richardson
    extrapolation of float64 runs, not an independently computed truth.

---

## Appendix — how to re-run

```
cd C:/Users/mateusz/UROP/Claude_Flight_Sim
./.venv/Scripts/python.exe audit_evidence/scripts/a1_order.py
./.venv/Scripts/python.exe audit_evidence/scripts/a2_richardson.py
./.venv/Scripts/python.exe audit_evidence/scripts/a2b_richardson_table.py
./.venv/Scripts/python.exe audit_evidence/scripts/a3_floor_quat.py       # slow, ~10 min
./.venv/Scripts/python.exe audit_evidence/scripts/a3b_floor_scaling.py   # slow, ~15 min
./.venv/Scripts/python.exe audit_evidence/scripts/a4_conservation.py     # slow
./.venv/Scripts/python.exe audit_evidence/scripts/a5_trim.py
./.venv/Scripts/python.exe audit_evidence/scripts/a5b_uniqueness.py
./.venv/Scripts/python.exe audit_evidence/scripts/a6_float32.py
./.venv/Scripts/python.exe audit_evidence/scripts/a6b_f32_rollout.py
./.venv/Scripts/python.exe audit_evidence/scripts/a7_jit.py              # slow (disable_jit)
./.venv/Scripts/python.exe audit_evidence/scripts/a8_misc.py
./.venv/Scripts/python.exe audit_evidence/scripts/a9_workingdt.py
./.venv/Scripts/python.exe audit_evidence/scripts/a10_repro.py
./.venv/Scripts/python.exe audit_evidence/scripts/a11_loadfactor.py
```

`a6_float32.py` and `a6b_f32_rollout.py` flip `jax_enable_x64` off partway
through and must be run as their own process.
