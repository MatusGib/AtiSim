# Agent D — Wind fields: evidence

Scope: `flightsim/wind.py` (four analytic fields, `gust_rates`, `sampled_rates`,
`strip_roll_moment`, `superpose`, `field_model`, `sampled_field_model`,
`along_track_shear`) and the RK4 wind-sampling seam in `integrate.step`.

**Evidence only. No causal attribution, no verdicts.** Every number below was
measured with `.venv/Scripts/python.exe` at `jax_enable_x64 = True`. Nothing
under `flightsim/` or `docs/` was modified; the per-stage RK4 variant lives in
`audit_evidence/D-scripts/d3_rk4_seam.py`.

Baseline: `pytest flightsim/tests/test_wind.py test_microburst.py
test_lee_wave.py` → **54 passed**. Every finding below is an addition to a
green suite, not a failing test.

Scripts, all runnable from the repo root
(`./.venv/Scripts/python.exe audit_evidence/D-scripts/<name>`):

| script | what it measures |
| --- | --- |
| `wcommon.py` | the four fields built exactly as `scripts/vortex.py`, `leewave.py`, `microburst.py` build them; random position/quaternion samplers |
| `d1_gradients.py` | §1 — `gust_rates` vs jacrev, explicit jacfwd, 5 central-difference steps, hand-derived closed forms |
| `d2_admissibility.py` | §2 — divergence/vorticity on grids; the four Oseguera–Bowles constants |
| `d3_rk4_seam.py` | §3 — the per-stage `step` variant; order ladder; final-state cost |
| `d3b_order_isolate.py` | §3 — order isolation across four RHSs of increasing non-smoothness |
| `d3c_headline_cost.py` | §3 — hold vs per-stage on the four project encounters, whole-run channels |
| `d3d_fig8_window.py` | §3 — the same, in the WINDOWED Fig-8 coordinate `scripts/vortex.py` prints |
| `d4_edges.py` | §4 — every edge/singularity probe (value, jacfwd, jacrev, hessian, `gust_rates`, `along_track_shear`) |
| `d4b_guards.py` | §4 — do the guards earn their keep; what the clamps cost |
| `d567.py` | §5, §6, §7 |
| `d67b.py` | §6 curving path at exposing headings; §7 the one-sided station set |
| `d8_closure.py` | rigid-rotation sign self-check; heading rate in the project's own runs |
| `d9_source.py` | the microburst against its primary source, `refs/NASA-TM-100632` |

---

## 1. Analytic gradient cross-check

`gust_rates` builds `jac_ned = jax.jacfwd(field)(pos_ned)` and contracts it as
`dcm.T @ jac_ned @ dcm`, picking `[grad[2,1], -grad[2,0], grad[1,0]]`. Because
that *is* jacfwd, a jacfwd comparison only tests the contraction and the picks,
not the derivative. Four genuinely independent references were used: **jacrev**
(a different AD mode), **central finite differences at five step sizes**,
**hand-derived closed-form Jacobians** for all four fields, and a
**rigid-rotation identity** that tests the sign convention without any
derivative at all.

400 randomised (position, quaternion) pairs per field. Quaternions are uniform
on SO(3) (Shoemake). Positions cover the regime each field is flown in.
Relative error is `‖a−b‖ / max(‖b‖, 1e-3·max‖rates‖)` — the floor is tied to the
sample's *largest* rate so a point where the field happens to be flat cannot
manufacture a large relative error.

Each cell is **median / max**.

| comparison | vortex (Hannibal, 2 cores) | updraft (w0 24.4, R 2360, p=6) | lee wave (w0 6, L 25 km) | microburst (19.03, R 1000, z_m 150) |
| --- | --- | --- | --- | --- |
| jacrev | 1.48e-16 / 2.53e-15 | 8.38e-17 / 1.12e-15 | 1.27e-16 / 4.80e-16 | 1.75e-16 / 1.83e-15 |
| jacfwd (explicit) | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| central FD h=1e-1 | 4.71e-08 / 4.90e-07 | 7.48e-09 / 3.23e-07 | 1.09e-10 / 4.69e-10 | 4.71e-09 / 3.44e-06 |
| central FD h=1e-2 | 4.63e-10 / 4.83e-09 | 7.24e-11 / 3.23e-09 | 8.67e-11 / 1.27e-09 | 5.08e-11 / 3.44e-08 |
| central FD h=1e-3 | 1.48e-10 / 7.45e-10 | 1.71e-10 / 6.30e-09 | 8.30e-10 / 4.32e-08 | 8.51e-11 / 7.67e-10 |
| central FD h=1e-4 | 5.00e-09 / 1.87e-08 | 1.68e-09 / 7.70e-08 | 6.60e-09 / 3.95e-07 | 7.49e-10 / 6.28e-09 |
| central FD h=1e-5 | 1.81e-08 / 1.50e-07 | 1.91e-08 / 4.83e-07 | 5.28e-08 / 3.91e-06 | 7.20e-09 / 7.60e-08 |
| **hand-derived** | **1.79e-16 / 5.52e-15** | **1.84e-16 / 7.76e-15** | **0 / 3.80e-16** | **1.83e-16 / 1.08e-15** |
| hand vs AD, full 3×3 Jacobian (Frobenius) | 1.50e-16 / 5.55e-15 | 2.13e-16 / 4.16e-14 | 0 / 2.21e-16 | 1.80e-16 / 1.66e-15 |

Sample scale: median/max `|gust_rates|` = 1.56e-2 / 1.45e-1 (vortex),
1.25e-3 / 2.27e-2 (updraft), 4.30e-4 / 1.48e-3 (lee wave), 7.91e-3 / 1.84e-1
rad/s (microburst).

The FD column bottoms out at h≈1e-3 and rises again at h=1e-5 — the classic
truncation/round-off V. That the minimum is 1e-10 and not 1e-16 is the FD
method's own floor, not a discrepancy.

### 1a. Hand algebra (shown, as requested)

**Parks vortex** — one core at `(n_c, d_c)`, with `l = x − n_c`, `d = d_c − z`,
`r² = l² + d²`. `dl/dx = 1`, `dd/dz = −1`, `dr²/dx = 2l`, `dr²/dz = −2d`.

Outside (`r ≥ r0`), `W_N = V0 r0 d / r²` and `W_D = V0 r0 l / r²`:

```
∂W_N/∂x = −2 V0 r0 l d / r⁴
∂W_N/∂z =  V0 r0 (2d² − r²) / r⁴
∂W_D/∂x =  V0 r0 (r² − 2l²) / r⁴
∂W_D/∂z =  2 V0 r0 l d / r⁴
```

Inside (`r < r0`), `W_N = V0 d / r0`, `W_D = V0 l / r0`:
`∂W_N/∂z = −V0/r0`, `∂W_D/∂x = +V0/r0`, the rest zero.
Every `∂/∂y` is identically zero — this field has no east component and no east
dependence.

Trace check: outside, `∂W_N/∂x + ∂W_D/∂z = −2V0r0ld/r⁴ + 2V0r0ld/r⁴ = 0`;
inside, `0 + 0 = 0`. Divergence-free in both branches, analytically.
Vorticity `ω_y = ∂W_N/∂z − ∂W_D/∂x`: inside `= −2V0/r0` (solid body), outside
`= V0r0(2d²−r² − r² + 2l²)/r⁴ = 0` (irrotational). That is the Rankine
structure, derived rather than asserted.

**Lee wave** — `W = [0, 0, w0 cos(φ)]`, `φ = 2π(x − n0)/L`. The only non-zero
Jacobian entry is `∂W_D/∂x = −w0 (2π/L) sin(φ)`. Divergence `= ∂W_D/∂z = 0`.

**Updraft** (also done, though not requested) — `∂W_D/∂x = w0 p s^{p−1} e^{−s^p}
· dx/(ρR)` with `s = ρ/R`; `∂W_D/∂z = 0`.

**Microburst** (also done) — full 3×3 in `d1_gradients.py:jac_mb_hand`, including
`da/dz = −1` for `a > 0` and the `g(s) = (1−e^{−s})/s`, `g'(s) = (s e^{−s} −
(1−e^{−s}))/s²` pair.

### 1b. Independent sign check: the rigid-rotation identity

If the air moves as a rigid body at `ω` about the CG, the body-axis gust field
is `w_g(ρ) = ω × ρ`, whose components give `dv/dx = r`, `dw/dy = p`,
`dw/dx = −q`. `gust_rates` must therefore return `ω` exactly. Over 200 random
`(ω, attitude, position)` triples:

| estimator | worst relative error |
| --- | --- |
| `gust_rates` | 1.57e-15 |
| `sampled_rates` | 2.85e-14 |

This tests both signs and all three component picks without touching a
derivative reference, and it is consistent with `omega_rel = omega − omega_gust`
in `dynamics.derivatives`.

**Nothing was found wrong in §1.** All four fields' gradients are correct to
machine precision by four independent routes.

---

## 2. Physical admissibility

### 2a. Divergence and vorticity on grids

Divergence is the trace of the analytic Jacobian; it is reported both absolutely
and relative to `max |∂W_i/∂x_j|` at the same point, so it is judged against the
gradients it is a cancellation of.

| field | grid pts | max \|div\| (1/s) | max \|div\| / max\|∂J\| | max \|curl\| (1/s) | median \|curl\| |
| --- | --- | --- | --- | --- | --- |
| vortex (2 cores) | 1323 | 2.78e-17 | 3.81e-15 | 2.833e-01 | 3.5e-18 |
| updraft (p=6) | 1323 | 0 | 0 | 2.281e-02 | 3.7e-11 |
| lee wave | 189 | 0 | 0 | 1.434e-03 | 8.9e-04 |
| microburst | 2205 | 1.39e-17 | 1.47e-15 | 3.930e-01 | 1.29e-02 |

Independent central-difference divergence (h = 0.5 m, 200 pts): max 9.1e-7
(vortex), 0 (updraft), 0 (lee wave), 1.0e-6 (microburst) — i.e. the FD residual,
not a physical divergence.

**All four fields are incompressible to machine precision.** Two of them
trivially so: the updraft and the lee wave have only a vertical component with no
vertical dependence, so `∇·W = ∂W_D/∂D ≡ 0` identically. That is admissible but
it is *structurally* divergence-free rather than a property that could have
failed — an infinite column with no entrainment and no return flow. The vortex
and the microburst are the two where the cancellation is non-trivial, and both
cancel to 1e-15 relative.

Rankine structure, single Hannibal core (`ω_y`, 1/s):

| r/r0 | 0.00 | 0.25 | 0.50 | 0.75 | 0.99 | **1.000** | 1.001 | 1.5 | 3.0 | 10.0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ω_y | −0.283333 | −0.283333 | −0.283333 | −0.283333 | −0.283333 | **−2.8e-17** | +2.8e-17 | +1.4e-17 | +3.5e-18 | −4.3e-19 |

`−2V0/r0 = −0.2833333`. Solid-body inside to all digits, irrotational outside to
round-off, and at `r = r0` **exactly** the strict `<` picks the outside branch
(see §4a).

### 2b. The microburst against its primary source

`refs/NASA-TM-100632-Oseguera-Bowles-1988.pdf` is in the repo, so the
transcription was checked against the paper directly rather than only for
self-consistency.

Paper eqs. (5)/(6), p. 4:

```
u = (λR²/2r)[1 − e^{−(r/R)²}](e^{−z/z*} − e^{−z/ε})
w = −λ e^{−(r/R)²}[ε(e^{−z/ε} − 1) − z*(e^{−z/z*} − 1)]
```

The module's `w_up = −λ e^{−s}[z*(1 − e^{−z/z*}) − ε(1 − e^{−z/ε})]` is eq. (6)
with the minus sign distributed — algebraically identical. The horizontal
component, refactored as `0.5 λ · (1−e^{−s})/s · shape · n`, is eq. (5) times the
direction cosine `n/r`, also identical. Numerically, over 12 000 (r, z) points
against a literal transcription of (5) and (6):

| component | worst relative error |
| --- | --- |
| `u` | 1.59e-11 (the 1/r form's own cancellation near the axis) |
| `w` | 4.16e-15 |

**Continuity.** In cylindrical form, `(1/r)∂(ru)/∂r = λ e^{−(r/R)²}
[e^{−z/z*} − e^{−z/ε}]` and `∂w/∂z = −λ e^{−(r/R)²}[e^{−z/z*} − e^{−z/ε}]`, so
the sum is identically zero. Numerically the worst `|div| / max|∂J|` over a
60×40 (r, z) grid is **6.59e-15**. The paper's claim, and the module's, hold.

**The four constants.** The paper's two stationarity equations (p. 4) are
`2(r/R)² = e^{(r/R)²} − 1` and `z_m/z* = ln(z*/ε)/((z*/ε) − 1)`. (The scan of the
first prints `e^{−(r/R)²} − 1`, which has no positive root; the `+` sign is
forced and is equivalent to the module docstring's `e^{−x²}(2x²+1) = 1`.)

| constant | module value | exact value | rel. discrepancy | measured on the shipped field |
| --- | --- | --- | --- | --- |
| `r/R` at peak outflow | 1.1212 | **1.1209064228** | 2.62e-04 | 1.1209099 |
| `z_m/z*` | 0.22 | **0.2196285778** = ln(12.5)/11.5 | 1.69e-03 | 0.2196277 |
| `z*/ε` | 12.5 | **12.4694690** (the root given `z_m/z* = 0.22`) | 2.45e-03 | 12.5 exactly (constructor identity) |
| `u_max/(λR)` | 0.2357 | **0.2356745** | 1.08e-04 | 0.2356743 |

Consequences for `wind.microburst(u_max=19.03, radius=1000, z_m=150)`:

| requested | delivered | rel. error |
| --- | --- | --- |
| `u_max` 19.03 m/s | 19.027924 m/s | 1.09e-04 |
| `z_m` 150.0 m | 149.746178 m | 1.69e-03 |
| peak radius 1121.2 m | 1120.910 m | 2.59e-04 |

All four constants are mutually consistent to within their own quoted precision.
The largest deliverable error is **0.17 % in the altitude of peak outflow**.

**What the paper actually says about the four, which differs from the module's
docstring.** TM-100632 p. 4: *"It was also noted that the ratio z_m/z* = 0.22"* —
an empirical TASS observation. p. 5: *"Recalling that z_m/z* = 0.22, the values
1.1212 and 12.5 were obtained from iteration for the ratios r/R and z*/ε."* So
in the source, `0.22` is an **input** and `12.5` and `0.2357` are its
**consequences**; only `1.1212` is independent of it. The module's docstring
states the reverse dependency (*"0.22 is ln(12.5)/11.5"*) and concludes the four
are *"a genuine cross-check rather than four restatements of one number"*. As a
check on the transcription of eqs. 5 and 6 the four constants are real and they
pass; as a claim of four independent facts about the world they are one
empirical ratio plus two of its consequences plus one independent root.

**The paper's fifth relation.** `w_max = λ z*(e^{−z_h/z*} − 0.92)` is the
large-`z` limit of eq. (6) on the axis (since `1 − 1/12.5 = 0.92`). Measured
against the shipped field on the axis:

| z (m) | model `w_up` | paper form | diff |
| --- | --- | --- | --- |
| 100 | −3.8098 | −3.1058 | 7.04e-01 |
| 300 | −15.2095 | −15.1915 | 1.80e-02 |
| 600 | −27.8117 | −27.8116 | 7.36e-05 |
| 1000 | −37.9455 | −37.9455 | 4.81e-08 |

Consistent: the two agree once `e^{−z/ε}` is negligible (ε = 54.5 m here).

**The paper's fourth parameter is not in the module.** TM-100632 p. 5:
*"Different shears can be modeled by specifying four parameters … 1) a
characteristic horizontal dimension; 2) maximum wind velocity; 3) altitude of
maximum outflow; and 4) **depth of outflow**."* `wind.microburst()` takes three.
`z_h` never enters eqs. (5)/(6), so nothing is mis-computed — but it is where
the paper stops the model (*"the maximum vertical wind is located at r = 0 and
z = z_h, by definition"*), and without it the field has no ceiling. On the axis
with the project's default parameters:

| z (m) | 50 | 150 | 300 | 500 | 1000 | 2000 | 5000 | 1e6 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| \|w_up\| (m/s) | 1.25 | 6.75 | 15.21 | 24.21 | 37.95 | 47.72 | 50.61 | 50.64 |
| \|w_up\| (kt) | 2.4 | 13.1 | 29.6 | 47.1 | 73.8 | 92.8 | 98.4 | 98.5 |

The asymptote is `λ(z* − ε) = 50.64 m/s = 98.5 kt`, approached monotonically.
The paper's own example has 21 kt of downflow at `z_m` and 29 kt at `z_h`; the
shipped field reaches 29 kt at **z = 294.4 m**, so `scripts/microburst.py`'s
default 300 m AGL penetration sits essentially exactly at the model's own depth
of outflow. Nothing in `wind.py` refuses a caller who asks for 5 km.

Consistency check on the paper's other statement, *"this is about 89 percent of
the radius of peak outflow"*: `1/1.1212 = 0.8919`, `1/1.120906 = 0.8921`. Both
round to 89 %.

---

## 3. The RK4 sampling seam

`integrate.step` calls `wind_model` once and closes over the result in `f`, so
all four RK4 stages see `w(x_n)` rather than `w(x(t))`. The per-stage variant
(scratch only) is:

```python
@partial(jax.jit, static_argnames=("wind_model",))
def step_perstage(sim, controls, dt, ac, wind_model):
    def f(s):
        w, og, _, _ = wind_model(sim.wind, s, sim.key, dt)
        return derivatives(s, controls, ac, w, og, increment=None)
    new_state = integrate.rk4_step(f, sim.state, dt)
    new_state = new_state._replace(quat=quat_normalize(new_state.quat))
    ...
```

Everything else — `rk4_step`, `derivatives`, the quaternion renormalisation — is
the shipped code. Reference solutions are per-stage RK4 at `dt = T/262144`.

### 3a. Observed order of accuracy

Error is `‖pos_ned(T) − pos_ref(T)‖`. `p` is the local log₂ ratio.

| RHS | HELD order | PER-STAGE order | reference self-consistency |
| --- | --- | --- | --- |
| zero wind (baseline) | — (identical, 1.0e-9 floor) | — | 1.7e-10 m |
| lee wave (C^∞) | **1.00, 1.00, 1.00, 1.00, 1.00** | **3.97, 3.45** then floor | 3.7e-10 m |
| updraft p=6 (C^∞) | **1.00, 1.00, 1.00, 1.00, 1.00** | **4.10, 4.05, 4.02, 3.95** | 9.6e-11 m |
| vortex, path 2 r0 *above* the cores (analytic on the path) | **1.00, 1.01, 1.02, 1.00, 1.00** | **4.09, 4.06, 4.23** | 2.9e-10 m |
| vortex, path *through* the core (C1 kink) | erratic: 2.47, −0.42, 2.99, −0.57, 1.76 | erratic: 0.21, 1.44, 1.11, 2.66, −1.41 | **1.4e-04 m** |

Sample table (lee wave, `T = 3.0 s`):

| n | dt (s) | HELD err (m) | p | PER-STAGE err (m) | p |
| --- | --- | --- | --- | --- | --- |
| 32 | 0.09375 | 1.0961e-02 | — | 8.0615e-08 | — |
| 64 | 0.04688 | 5.4940e-03 | 1.00 | 5.1427e-09 | 3.97 |
| 128 | 0.02344 | 2.7504e-03 | 1.00 | 4.7213e-10 | 3.45 |
| 256 | 0.01172 | 1.3760e-03 | 1.00 | 1.8839e-10 | 1.33 |
| 512 | 0.00586 | 6.8822e-04 | 1.00 | 1.7810e-10 | 0.08 |
| 1024 | 0.00293 | 3.4416e-04 | 1.00 | 1.6538e-10 | 0.11 |

**Measured: the hold reduces the scheme from fourth order to exactly first
order** whenever any wind field is switched on. It is order 1.00 to two
decimals over five successive halvings, on three different smooth fields. The
per-stage variant recovers 4.0–4.1 on all three. Independently, the Parks core
traverse shows **neither** scheme achieving a clean order, and the reference
itself is only self-consistent to 1.4e-4 m — the C1 kink at `r = r0` (§4a) caps
the attainable order on that specific path for both schemes.

Two smoothness features of the plant were checked and cleared as *not* the
cause: cruise Mach is 0.7995 against `m_crit` 0.7143, so `wave_drag`'s
`max(M − m_crit, 0)⁴` is on its smooth (C3) side throughout; and cruise altitude
12 192 m is above the 11 000 m tropopause, so `atmosphere` is on its smooth
upper branch.

### 3b. What the hold costs the project's own encounters

Held vs per-stage, same `dt`, same everything else. "excursion" is the run's own
range in that channel.

**Parks Hannibal traverse, lead-in 40 r0, 40.2 s — dt = 0.02 (as requested):**

| quantity | run's excursion | hold error | fraction |
| --- | --- | --- | --- |
| max \|Δpos\| | — | 1.29e-01 m | — |
| θ | 8.3263 deg | 4.861e-02 deg | 5.84e-03 |
| n_z | 2.0776 g | 7.417e-03 g | 3.57e-03 |
| altitude | 43.2939 m | 1.277e-01 m | 2.95e-03 |
| max \|Δq\| | 7.224e-02 rad/s | 1.367e-03 rad/s | 1.89e-02 |

**The headline numbers `scripts/vortex.py` actually prints** — `vortex_viz.fig8_point`
restricted to the declared analysis window (`north ∈ [−r0, r0]` for the vortex,
`[−radius, radius]` for the column):

| encounter | dt | pts in window | d(θ) held | d(θ) per-stage | rel | d(n) held | d(n) per-stage | rel |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Parks vortex | 0.02 | 76 | 2.259586° | 2.222956° | **1.62e-02** | −1.239344 g | −1.232250 g | **5.72e-03** |
| Parks vortex | 0.01 | 153 | 2.239956° | 2.221648° | 8.17e-03 | −1.235217 g | −1.231469 g | 3.03e-03 |
| Wingrove updraft | 0.02 | 976 | 4.359837° | 4.354903° | 1.13e-03 | −0.114513 g | −0.112696 g | **1.59e-02** |
| Wingrove updraft | 0.01 | 1952 | 4.362332° | 4.359879° | 5.62e-04 | −0.113605 g | −0.112696 g | 8.00e-03 |

**At the requested dt = 0.02 the hold moves the Parks Hannibal Fig-8 pitch
coordinate by 1.6 % and its load coordinate by 0.6 %.** Every one of these eight
relative errors halves when `dt` halves, which is the first-order signature
showing up directly in the published quantities.

For reference, the same two statistics taken over the WHOLE run rather than the
window: d(θ) 8.32626° vs 8.28157° (5.37e-03) and d(n) −1.23934 vs −1.23225 g
(5.72e-03) at dt = 0.02.

**All four encounters, `max |Δ| / run's own excursion` (whole run, per-sample):**

| encounter | dt | θ | n_z | altitude | whole-run d(n) rel |
| --- | --- | --- | --- | --- | --- |
| Parks vortex | 0.02 | 5.84e-03 | 3.57e-03 | 2.95e-03 | 5.72e-03 |
| Parks vortex | 0.01 | 5.70e-03 | 2.70e-03 | 1.21e-02 | 3.03e-03 |
| Wingrove updraft p=6 | 0.02 | 1.89e-03 | 6.93e-03 | 4.52e-04 | 1.60e-02 |
| Wingrove updraft p=6 | 0.01 | 9.44e-04 | 3.48e-03 | 2.26e-04 | 8.06e-03 |
| Doyle lee wave, 3 λ | 0.02 | 3.20e-04 | 6.81e-04 | 3.41e-04 | 6.83e-04 |
| Doyle lee wave, 3 λ | 0.01 | 1.60e-04 | 3.40e-04 | 1.70e-04 | 3.43e-04 |
| Oseguera–Bowles microburst | 0.02 | 1.49e-03 | 1.60e-02 | 2.78e-03 | 1.83e-02 |
| Oseguera–Bowles microburst | 0.01 | 7.44e-04 | 8.08e-03 | 1.39e-03 | 9.19e-03 |

The three smooth-field rows halve exactly when `dt` halves, confirming the
first-order behaviour end-to-end in the analysed quantities. **The largest cost
is in the load-factor channel: 1.6–1.8 % of the microburst's and updraft's own
Fig-8 `d(n)` at dt = 0.02.**

Placed next to the discretisation error the run already carries (both schemes vs
a `dt → 0` reference, Parks run):

| dt | held \|pos err\| | per-stage \|pos err\| | ratio |
| --- | --- | --- | --- |
| 0.02 | 5.238e-01 m | 4.174e-01 m | 1.25 |
| 0.01 | 4.432e-01 m | 2.581e-01 m | 1.72 |

On this particular path the kink error is comparable to the hold error, which is
why the ratio is only 1.25–1.7 rather than the orders-of-magnitude gap the
smooth fields show.

---

## 4. Edges and singularities

Probe set per point: value, `jacfwd`, `jacrev`, `jacfwd(jacfwd)` (Hessian),
`gust_rates`, `along_track_shear`. NaN/Inf flagged on any of them.

### 4a. Parks vortex, core boundary r = r0

Approach along +north with `d = 0` (`r0 = 182.88 m`, `V0 = 25.908 m/s`):

| r − r0 | `w_D` | `∂W_D/∂N` | `ω_y` |
| --- | --- | --- | --- |
| +1e−6 | +25.907999858 | −1.416666651e-01 | +2.8e-17 |
| +1e−12 | +25.908000000 | −1.416666667e-01 | −5.6e-17 |
| **0** | **+25.908000000** | **−1.416666667e-01** | **−2.8e-17** |
| −1e−12 | +25.908000000 | **+1.416666667e-01** | −2.833333e-01 |
| −1e−6 | +25.907999858 | +1.416666667e-01 | −2.833333e-01 |

- **C0: continuous.** Both branches give `V0 d/r0 = V0 r0 d/r²` on the circle
  `r = r0`; checked at three points on that circle to 12 digits.
- **C1: discontinuous, and it is a sign reversal.** One-sided closed forms:
  inside `+V0/r0 = +1.41667e-01`, outside `−V0/r0 = −1.41667e-01`. Jump
  `2V0/r0 = 2.8333e-01 1/s`.
- **At `r = r0` exactly, JAX takes the OUTSIDE branch** (`inside = r2 < r0**2` is
  False). So `q_gust` at the boundary is the irrotational value, `ω_y = 0`, not
  the solid-body `−2V0/r0`. This is already asserted by
  `test_the_rankine_gradient_is_discontinuous_at_the_core_edge`; measured here
  independently and it agrees.
- **Which branch you get at the nominal boundary is decided by round-off in the
  position, not by the code.** On the vertical approach (`l = 0`, `d = r0`, i.e.
  `pos_ned[2] = −12192.0 + 182.88`) the measured `∂W_N/∂D` is `+1.41667e-01` at
  `d = r0 + 1e-6` (outside) but `−1.41667e-01` at `d = r0` (inside) — the
  opposite of the along-track case. Cause, verified directly: the north approach
  puts `above = 0.0` and `along = 182.88` exactly, so `r2 == r0**2` bit-for-bit
  and the strict `<` picks *outside*; the vertical approach forms
  `above = −12192.0 − (−12009.12)` and loses `7.96e-13 m` to cancellation, so
  `r2 < r0**2` is True and it picks *inside*. At 12 km altitude "exactly on the
  core boundary" is not a representable state, and the sign of a 0.283 1/s gust
  rate turns on the last bit of the position.

**Vortex core centre `r = 0`:** value `[0,0,0]`, all six probes clean, no NaN.
The `r2_safe = where(r2 < r0², r0², r2)` guard holds the outside branch's divisor
at `r0² > 0`, so the unused branch is finite and its tangent is finite.

**Far field:** `r = 1e6 r0` → `|w| = 2.59e-05`; `r = 1e12 r0` → `2.59e-11`. No
overflow, decays as 1/r as expected.

**Far field is not "still air", and the run knows it.** The 2-core array decays
only as 1/r, so the trimmed initial condition is never in equilibrium:

| lead-in | 4 r0 | 8 r0 | 12 r0 | 20 r0 | **40 r0 (the default)** | 80 r0 |
| --- | --- | --- | --- | --- | --- | --- |
| \|w\| (m/s) | 9.112 | 5.111 | 3.612 | 2.298 | **1.213** | 0.626 |
| \|ω_gust\| (rad/s) | 1.03e-2 | 2.95e-3 | 1.43e-3 | 5.66e-4 | **1.56e-4** | 4.14e-5 |

`scripts/vortex.py`'s 40 r0 default still starts the aircraft in a 1.21 m/s
downdraft.

### 4b. Updraft column, axis and `sharpness < 2`

No NaN or Inf at any probe, at any sharpness in {0.1, 0.5, 1, 1.5, 2, 3, 6}, at
r = 0 exactly, 1e-12 m, 1e-6 m, 1 m, or 1e9 m.

**The `jnp.maximum(offset/radius, 1e-12)` guard does earn its keep.** An
unclamped scratch copy, `jacfwd` at r = 0:

| sharpness | unclamped row 2 of the Jacobian | clamped |
| --- | --- | --- |
| 0.5 | `[inf, inf, nan]` | `[0, 0, 0]` |
| 1.0 | `[0.0051661, 0.0051661, 0]` | `[0, 0, 0]` |
| 2.0 | `[0, 0, 0]` | `[0, 0, 0]` |
| 6.0 | `[0, 0, 0]` | `[0, 0, 0]` |

At `sharpness = 1` the unclamped form is finite but *arbitrary*: it is the
`hypot`-at-zero tie rule (`jax.grad(hypot)(0,0) = [0.5, 0.5]`) leaking a
diagonal direction into an axisymmetric cusp. The clamp returns zero, which is
the defensible subgradient. So the guard is a correctness improvement, not only
a NaN mask — the docstring understates it.

**What the clamp costs, and the near-axis gradient it hides:**

| sharpness | clamp cost on `w(0)` | \|p_gust\| at r=1 m | at r=10 m | max \|p_gust\| over r (1/s) |
| --- | --- | --- | --- | --- |
| 0.1 | **6.115e-02** | 7.08e-01 | 7.91e-02 | **5.83e+07** |
| 0.5 | 1.000e-06 | 2.46e-01 | 7.44e-02 | **5.03e+03** |
| 1.0 | 1.000e-12 | 1.03e-02 | 1.03e-02 | 1.03e-02 |
| 1.5 | 0 | 3.19e-04 | 1.01e-03 | 7.68e-03 |
| 2.0 | 0 | 8.76e-06 | 8.76e-05 | 8.70e-03 |
| 3.0 | 0 | 5.57e-09 | 5.57e-07 | 1.19e-02 |
| **6.0 (the project's value)** | **0** | 8.47e-19 | 8.47e-14 | **1.77e-02** |

Two findings: (i) for `sharpness < 0.5` the clamp is **silent clamping that
changes physics** — it removes 6.1 % of the peak updraft at `sharpness = 0.1`;
(ii) for `sharpness < 1` the near-axis pitching/rolling gust rate is unbounded
as `r → 0` (5.8e7 rad/s at `p = 0.1`, 5.0e3 at `p = 0.5`), because
`d(s^p)/ds = p s^{p−1} → ∞`. Both are outside the project's declared value of
6.0, where the clamp costs exactly zero and the peak rate is a sane 1.8e-2 rad/s.

### 4c. Microburst — axis, ground plane, below it, far field

No NaN or Inf at any probe: axis at z = 150 and z = 0, r = 1e-9 m, either side of
the `scaled > 1e-8` guard, the ground plane, z = −1 m, −1000 m, −1e6 m,
r = 1e6 and 1e9 m, and z = 1e5 m.

**The `jnp.maximum(−pos_ned[2], 0.0)` tie at exactly z = 0 halves the vertical
derivative:**

| altitude | `∂W_N/∂D` | `∂W_D/∂D` |
| --- | --- | --- |
| +1e−6 m | −4.345269066e-01 | −3.88e-10 |
| +1e−12 m | −4.345269152e-01 | −3.85e-16 |
| **0 (exactly)** | **−2.172634576e-01** | +1.28e-18 |
| −1e−12 m | 0 | 0 |
| −1e−6 m | 0 | 0 |

The one-sided-from-above closed form is `−0.5 λ g(s) n · dshp/da = −4.3452692e-01`,
so the value returned at exactly `z = 0` is **exactly half** of it. This is
`lax.max`'s balanced-tie JVP (`jax.grad(jnp.maximum(x,0))(0) = 0.5`, verified
directly). It affects a measure-zero set and, with a level attitude, does not
reach `gust_rates` at all (the three picks are `∂w_g/∂y`, `∂w_g/∂x`, `∂v_g/∂x`,
none of which is the z-column); at a non-level attitude the `dcm.T J dcm`
contraction mixes it in.

**Below ground the field is exactly zero, not "the ground value held".** The
docstring says *"Holding the ground value keeps such a run finite"* — the ground
value *is* zero, so the statement is true, but the effect is that an aircraft
below `z = 0` flies into perfectly still air with exactly zero gust rate:

| z (m) | +50 | +10 | +1 | +0.1 | 0 | −0.1 | −1 | −10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| w_N | +13.640 | +3.940 | +0.430 | +0.043 | 0 | 0 | 0 | 0 |
| \|ω_gust\| | 7.97e-4 | 4.07e-5 | 4.32e-7 | 4.34e-9 | 0 | 0 | 0 | 0 |

C0 across `z = 0` (both components vanish there); the **gradient** steps to zero,
so the gust rate is C1-discontinuous at the ground plane.

**The `scaled > 1e-8` removable-singularity guard is C0 to machine precision.**
Comparing the two branches with `expm1` (a naive `(1−exp(−s))/s` check is swamped
by its own cancellation and falsely reports a 1e-8 jump):

| scaled | series `1 − s/2` | `−expm1(−s)/s` | \|diff\| |
| --- | --- | --- | --- |
| 1e−6 | 0.9999995000000000 | 0.9999995000001666 | 1.67e-13 |
| 2e−8 | 0.9999999899999999 | 0.9999999900000002 | 2.22e-16 |
| **1e−8 (the switch)** | 0.9999999950000000 | 0.9999999950000000 | **0** |
| 1e−9 | 0.9999999995000000 | 0.9999999995000000 | 0 |

### 4d. Lee wave

Clean at north = 0, 1e9 m and 1e15 m (the phase stays exact because `2π x/L`
with L = 25 km keeps `x/L` in a benign range even at 1e15). **`wavelength = 0`
produces NaN in the value and every derivative** — a degenerate input with no
guard, not reachable from any project script.

### 4e. The `jnp.where` unused-branch trap, checked branch by branch

`jnp.where` blocks a NaN *value* in the forward pass, but its transpose sends a
zero cotangent into the unused branch, so `0 × inf` can still surface in reverse
mode. Each guard was checked with `jacrev` **and** with `jax.grad` of a scalar
reduction (the shape `along_track_shear` actually uses).

| guard | admissible parameters | degenerate parameters |
| --- | --- | --- |
| `vortex_wind`: `r2_safe = where(r2 < r0², r0², r2)` | **clean.** The unused inside branch is `V0·d/r0`, finite for `r0 > 0`; the unused outside branch divides by `r2_safe ≥ r0² > 0`. jacfwd, jacrev, Hessian, `gust_rates`, `along_track_shear` all clean at r = 0 and r = r0. | **`r0 = 0` leaks.** At the core centre the *value* is NaN. Off-axis the value and `jacfwd` are fine but **`jacrev` and `along_track_shear` return NaN** — the unused `V0·d/r0` branch differentiates to `inf`, and the where-transpose multiplies it by a zero cotangent. This is the classic trap, and it exists, but only at `r0 = 0`. |
| `updraft_wind`: `scaled = maximum(offset/radius, 1e-12)` | **clean at every sharpness tested**, including 0.1 and 0.5 where the unclamped form is `[inf, inf, nan]`. `jnp.maximum`'s JVP *selects* rather than multiplies, so the infinite tangent is discarded, and the clamped `scaled` has zero derivative w.r.t. `offset` inside the clamp so nothing multiplies it downstream. | n/a |
| `microburst_wind`: `safe = where(scaled > 1e-8, scaled, 1.0)` | **clean.** `safe` is never zero, so `−expm1(−safe)/safe` and its derivative are finite in the unused branch. Verified with `jacrev` and `grad(Σw²)` at r = 0 exactly. | n/a |
| `microburst_wind`: `altitude = maximum(−z, 0)` | **clean** — no NaN above, at, or below ground; but see 4c for the halved tie derivative. | n/a |

Also checked: differentiating w.r.t. the *field parameters* rather than position
(a sensitivity study would). `d/dr0`, `d/dv0` at r = 0, r = r0, inside and
outside: all finite. `d/d(sharpness)` at the axis and at r = 1000 m for
sharpness 0.5, 2, 6: all finite.

**Things I tried and failed to break:** NaN at the vortex core centre with
`r0 > 0`; NaN on the updraft axis at any sharpness down to 0.1; NaN on the
microburst axis, at the ground, or 1 000 km below it; overflow in any far field;
reverse-mode leakage through any of the four guards at admissible parameters;
second-order AD (Hessians) at every one of those points.

---

## 5. Superposition

| check | result |
| --- | --- |
| `superpose(v, u, l)` vs a hand-written `v(p)+u(p)+l(p)`, 500 pts | max diff 0, **bit-identical** |
| `wind_ned` of the superposed field vs the sum of the parts | max diff 0, **bit-identical** |
| `omega_gust` of the superposed field vs the sum of the parts, 500 (pos, quat) | max **relative** 6.51e-16, not bit-identical |
| `strip_roll_moment` of the superposed field vs the sum | max diff 8.02e-18 on a Cl scale of 4.39e-03 |
| `superpose(one_field)` | exact |
| `superpose()` (no fields) | returns Python `int 0`, not a 3-vector — would fail downstream. No guard. |

`omega_gust` is not bit-identical because `jacfwd` of a summed field is not the
same floating-point sequence as summing three Jacobians; 6.5e-16 relative is
round-off.

**Where superposition is not legitimate.**

1. **The microburst's ground boundary condition.** The microburst is the only
   field here with a ground plane, and it is the property the paper's
   introduction singles out. Superposing anything with a non-zero vertical
   component at `z = 0` destroys it:

   | r (m) | 0 | 500 | 1121 | 3000 |
   | --- | --- | --- | --- | --- |
   | microburst `w_D` at z = 0 | 0 | 0 | 0 | 0 |
   | + a 3 m/s updraft | −3.0000 | −3.0000 | −3.0000 | −3.0000 |

   i.e. 3 m/s of flow through the ground everywhere. Two microbursts superpose
   fine (both vanish at z = 0); a microburst plus any of the other three does
   not. `superpose`'s docstring cites Parks for the legitimacy of superposition
   and does not record this exception.

2. **The aerodynamic response is not linear in the wind**, even though the fields
   are. `aero.py` is linear in α, but `α = arctan2(w_rel, u_rel)` is not linear
   in the wind, and `CD ∝ CL²` is quadratic. Testing
   `F(w1+w2) − F(0)` against `[F(w1) − F(0)] + [F(w2) − F(0)]` at the 747 cruise
   condition:

   | gust pair | non-additivity | absolute |
   | --- | --- | --- |
   | vortex core peak (−25.9 m/s) + lee-wave crest (+6 m/s) | 2.50 % | 47.1 kN of 1888 kN |
   | updraft peak (−24.4) + lee-wave crest (+6) | 2.57 % | 44.6 kN of 1739 kN |
   | two 10 m/s vertical gusts | 1.61 % | 30.5 kN of 1897 kN |

   The module docstring says *"aero.py sees only vel_rel and omega_rel, so
   summing fields is exact within the model's own linearisation"*. That is true
   as literally written — the **fields** sum exactly — but the **response** to a
   superposed field is not the sum of the responses, at the 1.6–2.6 % level for
   gusts of this size.

---

## 6. `along_track_shear`

### 6a. Sign convention, end to end

Test field `u_N = 0.01 x` (a tailwind growing toward +north), 747 cruise:

| case | `U_x` | `dU_x/dt` | `F` |
| --- | --- | --- | --- |
| flying north into a growing tailwind | +10.000 | +2.35915 | +0.24057 |
| flying south (same field: a headwind that weakens) | −10.000 | +2.35915 | +0.24057 |
| flying north, descending 5 m/s | +10.000 | +2.35915 | +0.24057 |
| flying east, across the gradient | 0 | 0 | 0 |
| pure `w_up = +6 m/s` updraft, no shear | — | 0 | **−0.02543** |

`U_x` is positive for a tailwind, as `dynamics.f_factor`'s Eq. (3) requires; a
tailwind that grows ahead of the aircraft gives `F > 0` (hazardous) and an
updraft gives `F < 0` (favourable). **Sign convention verified end to end,
including the case that most easily goes wrong — flying the opposite way through
the same field also gives `F > 0`, which is right, because a weakening headwind
is the same energy loss as a growing tailwind.**

### 6b. The constant-heading approximation on a curving path

The exact rate is `dU_x/dt = (∇U_x · v) + u_h · dĥ/dt`. The code computes only
the first term, because `heading` is captured as a constant before `jax.grad`.
The omitted term is exactly `−u_⊥ ψ̇`, with `u_⊥` the **cross-track** horizontal
wind. Verified directly (`u_⊥ = +13.3319 m/s`, `ψ̇ = 3 deg/s` →
predicted `−0.69806`, measured `+0.69806` with the sign of the definition):

Measured by finite-differencing `U_x(t)` along a prescribed constant-speed
circular ground track:

| case | ψ̇ | exact | code | omitted | ΔF |
| --- | --- | --- | --- | --- | --- |
| Parks vortex, 0.5 r0 above a core, heading **east** | 1 °/s | 0.23269 | −0.00000 | 0.23269 | 0.0237 |
| Parks vortex, 0.5 r0 above a core, heading **east** | 3 °/s | 0.69806 | −0.00000 | 0.69806 | **0.0712** |
| microburst peak outflow, 45° off the radius | 3 °/s | 2.70646 | 2.00197 | 0.70449 | **0.0718** |
| microburst peak outflow, heading east on the north axis | 3 °/s | 3.00814 | 4.00444 | −0.99630 | **0.1016** |
| microburst r = 2000 m, heading east | 3 °/s | 0.96003 | 1.72632 | −0.76629 | 0.0781 |
| lee wave / updraft, any heading | any | 0 | 0 | 0 | 0 |

Worst case over 300 sampled positions × the worst heading, at a standard-rate
(3 °/s) turn:

| field | max \|omitted\| (m/s²) | ΔF |
| --- | --- | --- |
| vortex (Hannibal, 2 cores) | 1.36802 | **0.13950** |
| microburst | 0.98395 | **0.10033** |
| updraft | 0 | 0 |
| lee wave | 0 | 0 |

So the approximation can cost **more than the FAA's whole 0.1 alerting
threshold** at a standard-rate turn — but `f_factor`'s own docstring already
declines to use that threshold at 12 km, so the right comparison is against
`thrust_authority`, which for the 747 at 40 kft is small.

The updraft and lee wave have *no horizontal component at all* in this module,
so the omitted term is identically zero for them by construction.

### 6c. But the error is exactly zero in every run the project actually makes

| run | max \|ψ̇\| (deg/s) | max \|u_⊥\| (m/s) | max omitted ΔF |
| --- | --- | --- | --- |
| Parks Hannibal vortex | 0 | 0 | 0 |
| Wingrove updraft p=6 | 0 | 0 | 0 |
| Doyle lee wave | 0 | 0 | 0 |
| Oseguera–Bowles microburst | 0 | 0 | 0 |

All four fields have zero east wind on the north axis and every project run is
entered northbound at zero bank, so the ground track never leaves the meridian.
Vertical-plane curvature does not matter either: `heading` is built from
`vel_ned[:2]` only, so a pull-up leaves it exactly constant. **This is a real
approximation with a real bound, currently exercised at exactly zero amplitude.**

### 6d. The `1e-9` ground-track floor

| `vel_ned` | result |
| --- | --- |
| `[1e-3, 0, 0]` | −2.143791e-09 |
| `[1e-12, 0, 0]` | −2.143791e-21 (1000× too small — below the floor, `heading` stops being a unit vector) |
| `[0, 0, −5]` | 0 |
| `[0, 0, 0]` | 0 |

No NaN. Below `|track| = 1e-9 m/s` the "unit" heading is silently scaled, so the
answer is silently scaled too. For a purely vertical flight path the function
returns exactly 0, which also silently drops Proctor Eq. (4)'s
vertical-shear × ascent-rate term. Both are unreachable at any flying speed.

---

## 7. `sampled_rates` vs `gust_rates`

`airframe.stations(747)`: span `−29.82 … +29.82 m` (9 pts), longitudinal
`−33.497 … 0 m` (9 pts).

### 7a. The claimed exact agreement for a linear field

50 random linear fields `A p + b` × 50 random attitudes: **worst relative
difference 1.15e-13.** Not bit-identical — the secant is a least-squares
reduction and the tangent is an AD contraction — but exact to round-off, which
is what the docstring claims. The rigid-rotation check in §1b gives 2.85e-14 for
`sampled_rates` on the same footing.

### 7b. Disagreement across the Parks core

Normalised by `V0/r0 = 0.141667 1/s`, single Hannibal core, level attitude:

| north/r0 | tangent q | secant q | \|diff\| |
| --- | --- | --- | --- |
| −3.00 | 0.11111 | 0.10469 | 0.00642 |
| −2.00 | 0.25000 | 0.22891 | 0.02109 |
| −1.25 | 0.64000 | 0.55752 | 0.08248 |
| −1.10 | 0.82645 | 0.70737 | 0.11907 |
| **−1.00** | 1.00000 | 0.84363 | **0.15637** |
| **−0.99** | −1.00000 | 0.80085 | **1.80085** |
| −0.50 … +0.99 | −1.00000 | −1.00000 | **0** |
| **+1.00** | 1.00000 | −1.00000 | **2.00000** |
| **+1.10** | 0.82645 | 0.07928 | **0.74717** |
| +1.25 | 0.64000 | 0.74865 | 0.10865 |
| +2.00 | 0.25000 | 0.27504 | 0.02504 |
| +3.00 | 0.11111 | 0.11831 | 0.00719 |

Dense sweep, −4 r0 to +4 r0: **max 2.00000 V0/r0 = 0.28333 rad/s of q_gust, at
north/r0 = +1.0000** — the full sign reversal, consistent with
`test_the_curvature_correction_across_the_parks_core_is_measured`.

Across the other three fields, over 400 random (position, attitude) per field:

| field | peak \|tangent\| (rad/s) | max \|secant − tangent\| | % of peak | median % |
| --- | --- | --- | --- | --- |
| vortex (2 cores) | 1.470e-01 | 2.301e-01 | 156.5 % | 0.40 % |
| microburst | 1.978e-01 | 5.384e-02 | 27.2 % | 0.07 % |
| updraft p=6 | 2.270e-02 | 6.628e-04 | 2.9 % | 0.12 % |
| lee wave | 1.481e-03 | 6.262e-06 | 0.42 % | 0.04 % |

### 7c. The longitudinal station set is one-sided — a systematic bias, not just a secant

`airframe.stations` gives `longitudinal = linspace(−33.497, 0, 9)`: **every
station is aft of the CG**, so their centroid is `x = −16.749 m`. A
least-squares slope over a one-sided set estimates the derivative at the set's
**centroid**, so `sampled_rates`' longitudinal channel is a *backward* secant
carrying an `O(arm/2 · f'')` bias, not the symmetric `O(arm² f''')` error a
centred secant would have.

Verified with a quadratic field `w_D(x) = c x²`, whose exact tangent at the CG is
zero:

| c | tangent q | secant q | predicted `−2 c x_centroid` |
| --- | --- | --- | --- |
| 1e−4 | −0.000000e+00 | +3.349707e-03 | +3.349707e-03 |
| 1e−3 | −0.000000e+00 | +3.349707e-02 | +3.349707e-02 |

Exact to all printed digits. This also explains the strong **upstream/downstream
asymmetry** in 7b: at `+1.0 r0` (downstream edge) the whole airframe is inside
the core and the correction is the full 2.00 V0/r0; at `−1.0 r0` (upstream edge)
the whole airframe is outside and the correction is only 0.156 V0/r0. And at
`−0.99 r0` — *inside* the core, where
`test_the_curvature_correction_across_the_parks_core_is_measured` asserts
`profile[0.99] < 1e-12` — the correction is **1.80 V0/r0**, because the test's
`_pitch_rates_at` samples positive fractions only. The docstring's summary
*"inside the core exactly 0"* holds on the downstream side and not on the
upstream one.

For scale: the core radius is 182.88 m = 3.07 half-spans (b = 59.64 m), i.e. a
core **diameter** of 6.13 spans; the tail arm is 33.50 m = 0.092 core diameters.
Morton case: `2 r0 = 274.32 m`, 4.60 spans.

---

## What I could not check

1. **Three of the four sources are not in the repo.** `refs/` holds Doyle et al.
   2011, Proctor et al. 2000, Oseguera & Bowles 1988, Caughey 2011 and
   CR-2144 — but **not** Parks et al. 1985 (J. Aircraft 22(2) 124–129) nor
   Wingrove & Bach 1994 (J. Aircraft 31(4) 753–760). So I verified the
   *microburst* transcription against its primary source and could not verify:
   - the Parks Rankine equations (3)–(6), the `r0`/`V0`/spacing values for
     Hannibal and Morton, or the claim that Parks builds arrays by linear
     superposition;
   - `UPDRAFT_W0 = 80 ft/s` and `UPDRAFT_SECONDS = 20 s` from Wingrove & Bach,
     or the Fig. 8 reference values.

   Everything I report about those two fields is *internal* consistency
   (gradients, divergence, Rankine structure), not agreement with a source.

2. **I did not audit the Doyle et al. lee-wave quotations** even though the PDF
   is present — the amplitudes (3/6 m/s zero-to-peak) and the 25 km declared
   wavelength are provenance questions rather than wind-field physics, and belong
   to whichever agent holds §5 of `PROJECT.md`.

3. **The `1.1212` discrepancy's origin.** The exact root of the paper's own
   r-derivative equation is 1.1209064; the paper prints 1.1212. I cannot tell
   whether that is the paper's iteration tolerance or a typesetting error,
   because the scanned equation also has a sign that cannot be right as printed
   (`e^{−(r/R)²} − 1`, which has no positive root) and the errata sheet's
   correction text is illegible in this scan for the equation it names.

4. **Whether the wind hold is "the standard treatment for Dryden and von Karman
   turbulence"** (`integrate.py` module docstring). That is a claim about
   practice in other simulators, not something measurable here, and no source for
   it is cited. What *is* measured is that the hold costs three orders of
   accuracy for the deterministic fields this module actually ships — for which
   the stochastic-key argument in the docstring does not apply, since
   `field_model` returns the key untouched.

5. **Coupled/interaction effects with the load path.** All §3 runs used
   `load_model=None`. I did not repeat the order study with
   `loads.strip_model`, which is also sampled once per step and held.

6. **Anything about `strip_roll_moment` beyond linearity.** I verified it
   superposes exactly; I did not audit the strip integral, the elliptic chord, or
   the `calibrated_lift_slope` — those are `airframe.py` and belong elsewhere.

7. **float32.** Every measurement is at `jax_enable_x64 = True`, which is what
   `flightsim/__init__.py` sets. I did not check whether the guards' thresholds
   (`1e-12` in `updraft_wind`, `1e-8` in `microburst_wind`, `1e-9` in
   `along_track_shear`) remain appropriate at single precision; `1e-12` is below
   float32 epsilon and would behave differently.

8. **How often the round-off branch flip in §4a actually bites in a flown run.**
   I established that at cruise altitude the boundary state is not representable
   and the branch is decided by the last bit of `pos_ned`. I did not instrument a
   trajectory to count how many RK4 stage evaluations land within one ulp of
   `r = r0`, nor whether any of them do — a 40 s traverse at dt = 0.02 makes 8036
   stage evaluations and the boundary is a measure-zero set, so the expected
   count is zero, but I did not verify that.

9. **The microburst `z_h` ceiling's effect on a flown run.** I measured that the
   field has no ceiling and reaches a 98.5 kt on-axis asymptote, and that the
   default 300 m AGL penetration sits at the model's own depth of outflow. I did
   not fly a run above `z_h` to quantify what a caller would get wrong, because
   choosing a "correct" answer to compare against would require a source the
   model is not fitted to.
