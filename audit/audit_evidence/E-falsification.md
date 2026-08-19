# Agent E — Falsification

**Brief:** actively try to produce nonphysical behaviour. Report what broke and,
equally, what resisted.

**Evidence only.** Nothing below attributes a cause. Where two findings look
related I say so and stop there; attribution is Phase 3's job.

**Scope labelling.** Every finding carries one of:

- `[MODEL]` — nonphysical behaviour inside the declared scope
  (turbulence-encounter analysis of a rigid fixed-wing aircraft, seconds to
  minutes).
- `[OUT-OF-SCOPE]` — the model behaving badly where it never claimed to work.
  Recorded because nothing in the code stops a caller going there.

**Reproduction.** All scripts are in `audit_evidence/E-falsification-scripts/`.
Run with `C:/Users/mateusz/UROP/Claude_Flight_Sim/.venv/Scripts/python.exe`
from inside that directory (`common.py` puts the repo on `sys.path`). Nothing
under `flightsim/` or `docs/` was modified; no tolerance, test or reference
value was touched.

**Conventions used throughout.** Total mechanical energy is

```
E = 0.5*m*|v_body|^2 + 0.5*omega.I.omega + m*g0*h,   h = -pos_ned[2]
```

For the EOM in `dynamics.py` this satisfies, exactly and with gravity
cancelling identically,

```
dE/dt = F_aero.v_body + M_aero.omega        (throttle 0)
```

Verified numerically against a finite difference to 2.3e-6 relative
(`a01b_power.py`). In still, uniform, motionless air with the throttle shut,
that quantity must never be positive: there is no energy source.

---

# WHAT BROKE

## E-1 `[MODEL]` Mechanical energy is created in still air with the throttle shut

**Severity: this is the headline finding.** The aerodynamic build-up delivers
net positive power to the airframe in motionless, uniform air.

### Decisive reproduction — `a01f_repro.py`

Cherokee, registry coefficients untouched, at its own cruise altitude
(1499.6 m):

| quantity | value |
|---|---|
| `vel_body` | `[75.0, 0, 0]` m/s |
| `omega` | `[0, -8.256, 0]` rad/s |
| elevator | `+0.4363` rad (= `elevator_limit`, 25 deg) |
| throttle | `0` |
| `wind_ned`, `omega_gust` | `0` |

```
F_aero = [ -2334.37,  0,  -19535.38]  N
M_aero = [     0,  -28105.90,      0]  N.m
P_aero = F.v + M.omega = +56964.9 W        <-- must be <= 0
```

Integrating from that state:

| dt | E(0) | E(peak) | gain | mean rate |
|---|---|---|---|---|
| 1e-4 s | 19153330.38 J | 19154536.53 J | **+1206.15 J** | +60.9 kW |
| 1e-3 s | 19153330.38 J | 19154536.25 J | **+1205.87 J** | +60.3 kW |
| 1e-2 s | 19153330.38 J | 19154536.32 J | **+1205.93 J** | +60.3 kW |

The gain is **dt-converged to 4 significant figures**, so it is the model, not
the integrator. Breakdown at the peak: `dKE = -2626 J`, `dPE = -6 J`,
`dRKE = +3838 J` — the energy lands in rotation.

### Which channel supplies it — `a01d_cherokee.py`

At that state, `Cm0 = 0`, `alpha = 0`, so

| term | value |
|---|---|
| `Cmq * q_hat` (damping) | +0.64939 |
| `Cmde * de` (control) | −1.04290 |
| **Cm** | **−0.39352** |

Split of the moment power `qbar*S*c*Cm*q`:

| contribution | power |
|---|---|
| `Cmq` damping alone | **−382.9 kW** (correctly dissipative) |
| `Cmde` control alone | **+615.0 kW** |

The control-moment power term wins. `CD` in `aero.py:93-98` is
`CD0 + CL^2/(pi e AR) + wave_drag`: there is **no `CD_de` term**, so a deflected
elevator generates a pitching moment that does work on the airframe while
incurring no drag penalty of its own.

### Is it reachable from a trimmed start? Yes — `a01g_reachable.py`, `a01f_repro.py`

Threshold pitch rate for `P_aero > 0`, Cherokee, elevator at its limit:

| V (m/s) | first q with P>0 |
|---|---|
| 30 | −2.109 rad/s (−121 deg/s) |
| **50 (cruise)** | **−3.514 rad/s (−201 deg/s)** |
| 75 | −5.268 rad/s (−302 deg/s) |

A plain fixed-control rollout from the Cherokee's own trim, throttle 0, full
elevator, reaches **max |q| = 3.677 rad/s (211 deg/s)** — past the threshold.

Energy actually created in fixed-control runs from trim (90 s, dt = 0.002 s,
throttle 0, still air). "Max rise" is the largest increase in `E` from any
local minimum, i.e. energy the model manufactured:

| aircraft | input | max rise | rise/E0 | peak dE/dt |
|---|---|---|---|---|
| boeing747 | any of 6 tested | **0 J** | 0 | negative |
| boeing747_approach | any of 6 tested | **0 J** | 0 | negative |
| cherokee | elevator +limit | 520.6 J | 3.0e-5 | +12.2 kW |
| cherokee | elev+ail+rud all +limit | **691.1 J** | 4.0e-5 | **+14.7 kW** |
| cessna172 | elevator+aileron +limit | 73.5 J | 4.2e-6 | +2.6 kW |

**Worst energy drift per unit time in still air, throttle 0, from a trimmed
start with a control input a pilot can make: +14.7 kW (Cherokee), sustained
long enough to create 691 J.**

### Bound on the consequence

In every one of ~120 still-air runs, `max(E − E0) = +0` exactly. The creation
is always superimposed on a larger drag loss, so total energy never rises above
where it started. The violation is real but bounded: it is a **local** energy
source that never wins globally in anything I could construct (see NB-1).

### Scan of the positive-power region — `a01b_power.py`, `a01c_power_corner.py`

Over a 90x120x5 grid in (V, q, de) at cruise altitude:

| aircraft | positive-power region found? |
|---|---|
| boeing747 | **none** anywhere in the scan |
| boeing747_approach | **none** anywhere in the scan |
| cherokee | 5079 grid points, V in [1.0, 75.0] m/s, max **+56.96 kW** |
| cessna172 | none at full elevator; +2.6 kW reached dynamically with aileron |

Both 747s resisted entirely. The two light aircraft did not.

---

## E-2 `[MODEL]` `omega_gust` steps discontinuously at the Parks vortex core edge

`a06_vortexedge.py`, `a056_wind2.py`

The Rankine core matches in **value** at `r = r0` but not in **gradient**.
`field_model` builds `omega_gust` from `jax.jacfwd` of the field
(`wind.py:528`), so the rotational gust handed to the aircraft **sign-flips**
across the boundary.

Measured either side of `r0` at 1e-7 relative offset:

| case | \|W jump\| | \|dW/dx jump\| | as multiple of 2·v0/r0 |
|---|---|---|---|
| hannibal, radial | 2.5e-13 m/s | 0.2833333 1/s | **1.0000** |
| hannibal, tangential | 2.5e-13 m/s | 0.2833333 1/s | **1.0000** |
| hannibal, 45 deg | 1.8e-13 m/s | 0.1416667 1/s | 0.5000 |
| morton, radial | 2.1e-13 m/s | 0.3111111 1/s | **1.0000** |
| morton, tangential | 2.1e-13 m/s | 0.3111111 1/s | **1.0000** |

The jump is exactly `2*v0/r0` — the core vorticity. On a tangential traverse
(aircraft level with the core, the geometry the source's traces come from):

```
q_gust inside  = -0.1416667 rad/s
q_gust outside = +0.1416666 rad/s
JUMP           =  0.2833333 rad/s  (16.23 deg/s)   [hannibal]
JUMP           =  0.3111111 rad/s  (17.83 deg/s)   [morton]
```

Sweeping `l/r0` through 1.0 for a 747 at cruise:

| l/r0 | q_gust (rad/s) |
|---|---|
| 0.99999 | −0.1416667 |
| 1.00000 | **+0.1416667** |
| 1.00001 | +0.1416638 |

Consequence at the aircraft — 747 at cruise, hannibal core:

```
STEP in aerodynamic moment across the edge:  +4.2683e+06 N.m in pitch
STEP in angular acceleration:                +0.095109 rad/s^2 in qdot
```

At dt = 0.005 s the full 0.28283 rad/s step lands between two consecutive
samples (measured on an actual traverse). The core traverse itself lasts
1550 ms at 235.9 m/s, so this is not an edge case of the geometry — it is the
encounter the project is built for.

**Note on interpretation.** A Rankine vortex is *supposed* to have a vorticity
discontinuity at the core edge; the code reproduces the idealisation
faithfully. What is a finding is the **magnitude of the resulting step in
`omega_gust` and in the moment**, and its interaction with E-3.

---

## E-3 `[MODEL]` RK4 collapses from 4th order to 1st order in any spatially varying wind field

`a06c_order.py`. Proper Richardson: error estimated as `|x(dt) − x(dt/2)|`, so
the reference does not contaminate the finest entry. 747 at cruise, 20 s,
perturbed rates `[0.02, 0.03, −0.01]`.

| wind field | ratio (should be 16 for RK4) | implied order |
|---|---|---|
| **(a) none (`zero_wind`)** | 16.04, 16.02, 16.02 | **4.00, 4.00, 4.00** |
| **(b) constant uniform (12,5,−2) m/s** | 16.40, 16.15, 15.50 | **4.04, 4.01, 3.95** |
| **(c) lee wave** (smooth, C-infinity) | 2.00, 2.00, 2.00, 2.00, 2.00 | **1.00 x5** |
| **(d) updraft column** (smooth, Gaussian) | 2.00, 2.00, 2.00, 2.00, 2.00 | **1.00 x5** |
| **(e) Parks vortex through the core** | 24.79, 0.16, 2.00, 41.40, 1.19 | **no order at all** |

Case (c) and (d) hold order 1.00 flat across five successive halvings — that is
a clean first-order result, not noise. Case (e) is non-monotone: the error at
dt = 0.005 s (1.92e-01) is **larger** than at dt = 0.01 s (3.84e-01 → 1.92e-01
is a decrease, but 0.02 s gives 6.17e-02, smaller than both). No refinement
schedule gives a trustworthy answer through the core.

Cases (a) and (b) isolate the variable cleanly: the order is intact with no
wind and with a wind of zero spatial gradient, and lost with any spatial
variation. `integrate.py:8-10` states wind is "sampled once per step and held
constant across the four RK4 stages" and calls this "the standard treatment for
Dryden and von Karman turbulence". I record that those are **temporal**
stochastic models, whereas every field exercised here is a **spatial** one; I
do not attribute the order loss.

Practical size: at dt = 0.01 s over 20 s through an updraft column, the
truncation error is 1.27e-2 in the max state component — roughly 800x what the
same step gives in still air.

---

## E-4 `[MODEL]` `trim` returns converged-but-absurd roots next to the real envelope, and `is_physical` passes non-flight conditions

`a03_trim.py`. 640 (V, h) points per aircraft, shipped coefficients untouched.
**All 640 converge to |residual| < 1e-8 for every aircraft.**

| aircraft | absurd roots | \|alpha\|>15 deg | throttle outside [0,1] | \|de\|>limit |
|---|---|---|---|---|
| boeing747 | 504/640 | 116 | 490 | 68 |
| boeing747_approach | 327/640 | 277 | 292 | 192 |
| cherokee | 563/640 | 191 | 561 | 72 |
| cessna172 | 575/640 | 156 | 568 | 101 |

**How close to the real envelope the absurd roots start** — nearest to each
aircraft's own cruise speed:

| aircraft | V | h | result | residual | `is_physical` |
|---|---|---|---|---|---|
| boeing747 | 232.0 (0.98·Vc) | 15000 m | throttle **1.416**, alpha 9.4 deg | — | **True** |
| boeing747_approach | 83.5 (0.98·Vc) | 6000 m | alpha 15.8 deg | 5.8e-21 | False |
| cherokee | 49.2 (0.98·Vc) | 6000 m | throttle **1.121**, alpha 4.5 deg | — | **True** |
| cessna172 | 59.0 (0.98·Vc) | 9000 m | throttle **1.303**, alpha 6.5 deg | — | **True** |

**Two percent below cruise speed** is enough. Worst roots reached:

| aircraft | V | h | alpha | throttle | residual |
|---|---|---|---|---|---|
| cherokee | 19.3 | 12000 | **−98214 deg** | 11.705 | 3.29e-13 |
| boeing747_approach | 45.5 | 20000 | **+11787 deg** | −30.728 | 3.45e-14 |
| cessna172 | 16.4 | 20000 | −7834 deg | 80.786 | 5.54e-14 |
| boeing747 | 11.8 | 20000 | +990 deg | −30.306 | 9.24e-15 |

Throttle over the absurd roots ranges to **[−151.3, +20529.8]**.

### `is_physical`'s blind spot, quantified

`trim.is_physical` checks `|alpha| <= 15 deg` and nothing else — not throttle,
not elevator, not even the residual. Roots that **pass** while requiring more
than 100% throttle or more than full elevator:

| aircraft | count | example |
|---|---|---|
| boeing747 | 471 | V=328.3, h=0: alpha −2.35 deg (passes), throttle **1.013** |
| boeing747_approach | 41 | V=254.7, h=0: alpha −4.77 deg (passes), throttle **1.015** |
| cherokee | 432 | V=79.3, h=0: alpha −4.57 deg (passes), throttle **1.015** |
| cessna172 | 496 | V=70.0, h=0: alpha 0.53 deg (passes), throttle **1.020** |

Cross-envelope (`a07b_extreme.py`): the Cherokee and Cessna both trim at the
747's 235.9 m/s with `is_physical = True` and throttle **8.634** and **11.007**
respectively.

### Negative altitude — `[OUT-OF-SCOPE]` but silent

All four aircraft trim happily at h = −20000 m with `is_physical = True`
(e.g. cherokee: alpha −5.73 deg, throttle 0.378, residual 2.0e-15) in air the
atmosphere model reports as 5.98 kg/m^3.

---

## E-5 `[MODEL]` Structurally singular Jacobians return NaN with no error raised

`a04_singular.py`, `a03_trim.py`

### cessna172 has an inert rudder

`CYdr = Cldr = Cndr = 0.0` (a declared modelling choice, `aircraft.py`
comment). The consequence is that the rudder column of any control Jacobian is
identically zero:

```
d(pdot,qdot,rdot)/d(de,da,dr) =
[[  0.        109.30327326   0.  ]
 [-29.13949662   0.           0.  ]
 [  0.         -2.49973932   0.  ]]

det  = 0.000e+00      cond = inf
jnp.linalg.solve(J, [1,1,1]) = [nan nan inf]
```

Any Newton solve carrying rudder as an unknown for this aircraft inverts a
singular matrix and returns NaN silently. My own steady-turn solver hit exactly
this (`a10b_turn.py` / `a10c_turn.py` give `nan` for cessna172 at every bank
angle, including **zero** bank).

### `trim` with zero thrust authority

`max_thrust = 0` makes the throttle column of `trim`'s Jacobian identically
zero. `trim` returns `[nan, nan, nan]` with residual `nan` — no exception, no
flag. This is a degenerate-coefficient case, not a shipped one.

Other degenerate coefficients all returned finite, converged, and mostly
`is_physical = True` roots:

| perturbation | alpha | throttle | `is_physical` |
|---|---|---|---|
| `CLa` sign flipped to −5.73 | −1.63 deg | 0.8209 | **True** |
| `Cma` flipped to +0.81 (statically unstable) | +1.61 deg | 0.8195 | **True** |
| `Cmde -> 0` (no pitch control at all) | −1.05 deg | 0.8204 | **True** |
| `e -> 1e-4` | +0.43 deg | **417.9** | **True** |
| `S -> 0.001 m^2` | 1169.96 deg | 7.93 | False |
| `mass -> 1e6 kg` | 809.30 deg | 7408 | False |

A sign-flipped lift-curve slope and a total loss of pitch control both produce
a root that `is_physical` endorses.

---

## E-6 `[MODEL]` / `[OUT-OF-SCOPE]` There is no ground, and runs reach it routinely

`a99_loose.py`, `a056_wind2.py`

Lowest altitude reached by plain **fixed-control** rollouts from each
aircraft's own trim (300 s, four control settings):

| aircraft | lowest altitude |
|---|---|
| boeing747 | +6521.1 m (stayed above) |
| boeing747_approach | **−3177.9 m** |
| cherokee | **−485.8 m** |
| cessna172 | **−11328.5 m** |

### The microburst case specifically

747-approach released at 300 m into a 25 m/s microburst on fixed trim controls:

```
min altitude          = -152.7 m
crosses h = 0 at      t = 17.16 s
  sink rate           = +32.65 m/s  (6427 ft/min)
  airspeed            = 108.8 m/s
THE RUN CONTINUES to t = 200 s: h = -152.7 m, V = 108.6 m/s, all finite
```

The aircraft arrives at the ground at 6400 ft/min and 212 kt and the simulation
reports a steady level cruise 153 m below sea level.

### The microburst ground clamp itself works

Below `z = 0` the field is identically zero and stays finite arbitrarily far
down (checked to h = −100 km). The field is **C0** across the ground plane
(max |dW| between samples 0.00025 m apart = 1.86e-4 m/s) but has a **C1 kink**:
`dW/dz` goes from `[-0.7425, 0, -2.05e-6]` just above to exactly `[0, 0, 0]`
just below — a 0.742 1/s gradient step. `omega_gust` is unaffected at level
attitude (it goes to zero smoothly: 1.15e-12 rad/s at h = 0.001 m), because
`gust_rates` reads only vertical-wind gradients.

### The atmosphere below sea level `[OUT-OF-SCOPE]`

`atmosphere.py` extrapolates the troposphere lapse downward without limit:

| h | T | rho |
|---|---|---|
| −1000 m | 294.65 K | 1.347 |
| −5000 m | 320.65 K | 1.930 |
| −20000 m | 418.15 K | 5.975 |
| −50000 m | **613.15 K** | **30.468** |

No NaN, no clamp, no warning. At the −152.7 m the microburst run reaches, the
error is small (rho +1.47% vs sea level, T 289.14 K) — so the ground-penetration
finding is about the missing ground, not about the atmosphere.

---

## E-7 `[OUT-OF-SCOPE]` The atmosphere is silently wrong above its stated 20 km ceiling

`a04_singular.py`. The module docstring says "ISA standard atmosphere, 0 to
20 km". Above 11 km the code holds `T = T_TROPOPAUSE = 216.65 K` **forever** —
there is no upper bound and no stratospheric warming layer.

| h | model T | ISA T | model rho | ISA rho | rho error | model a | a error |
|---|---|---|---|---|---|---|---|
| 20 km | 216.65 K | 216.65 K | 0.0880347 | 0.088035 | −0.0004% | 295.07 | 0% |
| **30 km** | 216.65 K | 226.65 K | 0.0181895 | 0.018410 | **−1.20%** | 295.07 | **−2.2%** |
| **47 km** | 216.65 K | 270.65 K | 0.00124625 | 0.0014275 | **−12.70%** | 295.07 | −19.9% |
| 100 km | 216.65 K | — | 2.92e-07 | — | — | 295.07 | — |

It returns finite, plausible-looking numbers all the way to 1000 km
(rho = 6.8e-69). Nothing raises, warns, or clamps. The speed-of-sound error
matters more than the density error because it feeds Mach and therefore the
wave-drag term.

*(ISA reference values quoted from the standard 1976/ICAO layer definitions.
I could not verify them against a source file in this repo, so treat the
comparison column as `unverified` — the model's own behaviour, the constant
216.65 K to arbitrary altitude, is directly measured and is the finding.)*

---

## E-8 `[MODEL]` The `V_MIN = 1.0` airspeed floor does bind, and its docstring says it does not

`a04_singular.py`, `a99_loose.py`

`aero.py:22-24`: *"Far below any flight speed, so it never binds in normal
operation."*

Lowest |V| reached over 21 fixed-control runs x 120 s per aircraft (full
deflections, large initial rates, steep attitudes — no state injection):

| aircraft | lowest \|V\| | binds? |
|---|---|---|
| boeing747 | **0.865 m/s** | **yes** |
| boeing747_approach | 1.721 m/s | no |
| cherokee | **0.659 m/s** | **yes** |
| cessna172 | **0.518 m/s** | **yes** |

Three of four aircraft cross it. What happens below the floor:

- **Dynamic pressure is overstated** by `(V_MIN/V)^2`: 3.7x at the Cessna's
  0.518 m/s, up to 100x at 0.1 m/s.
- **Non-dimensional rates are understated** by `V/V_MIN`, because
  `p_hat = p*b/(2V)` uses the same floored `V`.
- **Aerodynamic force is nonzero at exactly zero airspeed.** At V = 0 the
  Cessna reports `|F| = 1.520 N` (`Fz = −1.495 N`), identical at V = 0,
  1e-9, 0.5 and 0.9 m/s. That is 0.148 milli-g of lift out of stationary air.

Consequence on `load_factor` at V = 0 (`a10_loadfactor.py`, folded into
`a04_singular.py`): free fall does **not** give exactly zero.

| aircraft | n_z at V = 0 |
|---|---|
| boeing747 | +2.108e-05 |
| boeing747_approach | +5.026e-05 |
| cherokee | **+3.431e-04** |
| cessna172 | +1.195e-04 |

With the aero coefficients zeroed, `n_z` is exactly `-0.0` for all four, so the
offset is entirely the `CL0` term surviving the airspeed floor.

---

## E-9 `[MODEL]` No stall: lift keeps rising past the source's own measured CLmax

`a07b_extreme.py`. Cessna 172 against `CESSNA172_TABLES`, the aircraft's own
source data, which the module comment already flags as unusable by this model.
Quantified here:

| alpha | model CL | table CL | ratio | model CD | table CD | model L/W |
|---|---|---|---|---|---|---|
| 0 deg | 0.1660 | 0.1480 | 1.12 | 0.0306 | 0.0300 | 0.50 |
| 10 deg | 1.1660 | 1.1950 | 0.98 | 0.0900 | 0.0930 | 3.52 |
| 19.5 deg | 2.1160 | 1.8890 | 1.12 | 0.2290 | 0.1840 | 6.38 |
| **30 deg** | **3.1660** | — (past table) | **1.68x CLmax** | 0.4764 | — | **9.55** |
| 60 deg | 6.1660 | — | 3.26x CLmax | 1.7362 | — | 18.59 |
| 90 deg | 9.1660 | — | 4.85x CLmax | 4.6239 | — | 27.64 |

Source CLmax = 1.889 at 19.5 deg; a real wing loses lift past it. All four
aircraft at alpha = 30 deg:

| aircraft | CL | CD | L/W |
|---|---|---|---|
| boeing747 | 2.8458 | 2.0099 | 4.31 g |
| boeing747_approach | 3.5275 | 0.6897 | 3.17 g |
| cherokee | 2.9881 | 0.8753 | 5.55 g |
| cessna172 | 3.1660 | 0.4764 | **9.53 g** |

Extreme attitudes are worse: at alpha = 180 deg (backwards flight, 60 m/s) the
Cessna reports `F = [+2.648e6, 0, +5.889e5] N` — **259 g**. `arctan2` gives the
right alpha; the linear `CL = CL0 + CLa*alpha` then does the rest.

This is a declared limitation of linear aero, but the module comment estimates
"alpha excursions of order 20 deg" for the intended vortex work, which is
already past CLmax.

---

## E-10 minor / `[OUT-OF-SCOPE]`

| # | finding | evidence |
|---|---|---|
| a | `quat_to_euler` at **exactly** theta = ±90 deg returns `phi = psi = 0` instead of preserving `phi + psi`. At 89.999 and 90.001 deg it is correct (mod 360). Measure-zero and display-only. | `a04_singular.py` |
| b | A zero-length input quaternion `[0,0,0,0]` propagates NaN with no guard. | `a99_loose.py` |
| c | `jit` vs `disable_jit` on one `step` differs by up to 1.1e-16 (747-approach), consistent with FMA/reassociation. | `a02_longhorizon.py` |
| d | 747 at max throttle in a vertical dive from 12 km reaches **Mach 1.035**, where the wave-drag law `20*(M − M_crit)^4` (anchored on drag divergence) is far outside Lock's range. Still finite. | `a07b_extreme.py` |
| e | Cessna 172 at max throttle in a vertical dive ends at **h = −1332.9 m** and Mach 0.763 (Vne is roughly 84 m/s; it reaches 236.8 m/s). | `a07b_extreme.py` |

---

# WHAT I COULD NOT BREAK

These are the load-bearing negatives. Each records what was tried, how hard,
and the margin.

## NB-1 Total energy never once exceeded its initial value in still air

**Tried hardest here.** Approximately 120 rollouts across:

- All four registry aircraft from their own trim.
- Throttle 0, elevator at trim, all controls zero, full +/− elevator, full
  aileron, full rudder, and all three at their limits simultaneously.
- Initial rates to `[3.0, 2.0, 1.5]` rad/s.
- Initial attitudes at theta = ±1.2 to ±1.4 rad, phi = pi.
- Horizons 60–300 s at dt = 0.002–0.01 s.
- **Five deliberate energy-pumping feedback laws** (`a07_pump2.py`), designed
  from the E-1 mechanism to make `Cmde*de*q` positive every step:
  `de = −K*q` for K in {1, 5, 20, 100, 1000}; bang-bang `−lim*sign(q)`;
  bang-bang `+lim*sign(q)`; `±lim` alternating every step; and the same trick
  on the roll axis via aileron. 40 runs of 120 s, plus 600 s on the strongest.
- **Full-deflection reversals at the sample rate** with throttle 0
  (`a07d_rev.py`), at 25, 50, 100, 200, 500 and 1000 Hz, elevator alone and
  elevator+aileron+rudder together — 32 runs.

**Result: `max(E − E0) = +0` in every single one.** `Emax/E0 = 1.00000` for all
40 pumping runs. The 600 s run on the strongest pump had **0 of 29999 steps**
with `dE > 0`.

Margin: the strongest local creation found (+14.7 kW, E-1) is set against drag
losses of 12.9 MW (747), 7.2 MW (747-approach) and 49 kW (Cherokee) at trim —
a factor of 3.3 for the Cherokee, ~1000 for the 747s. The Cherokee's margin is
the thin one.

## NB-2 Aerodynamic power is strictly negative throughout the reachable envelope

`a01b_power.py`. 20000 randomised states per aircraft (80000 total) with
V in [0.5, 1.6]·Vcruise, alpha in ±0.25 rad, beta in ±0.20 rad,
omega in ±0.35 rad/s, and all three controls randomised to their limits:

| aircraft | states with P > 0 | max P | worst as fraction of W·V |
|---|---|---|---|
| boeing747 | **0 / 20000** | −1.022e+06 W | −0.00153 |
| boeing747_approach | **0 / 20000** | −1.059e+06 W | −0.00497 |
| cherokee | **0 / 20000** | −2912 W | −0.00545 |
| cessna172 | **0 / 20000** | −5090 W | −0.00829 |

Inside the linear-aero envelope the model is strictly dissipative with margin.
E-1 required leaving it (|q| > 2 rad/s).

Both 747s additionally resisted the full corner scan (90x120x5 grid over
V in [1, 1.5·Vc], q in ±10 rad/s, elevator at ±limit): **no positive-power point
exists anywhere in that space** for either.

## NB-3 Long horizons: nothing diverges, drifts, or loses the quaternion

`a02_longhorizon.py`. **1,000,000 steps (10,000 s) per aircraft at trim:**

| aircraft | max \|\|q\|−1\| | dV | altitude drift | max \|omega\| |
|---|---|---|---|---|
| boeing747 | **0.000e+00** | 0.0000 m/s | **0 m** | 0.000e+00 |
| boeing747_approach | **0.000e+00** | 0.0000 m/s | **0 m** | 1.064e-19 |
| cherokee | **0.000e+00** | 0.0000 m/s | **0 m** | 0.000e+00 |
| cessna172 | **0.000e+00** | 0.0000 m/s | **0 m** | 4.516e-17 |

The trim point is held to **machine precision over 10,000 simulated seconds** —
the state literally does not change. Trim residuals are 1.9e-20 to 1.8e-15.

**500,000 steps (5000 s) with a large initial rate perturbation
`[0.3, 0.2, −0.15]` rad/s:** all finite, quaternion norm deviation
**2.22e-16** (one ulp) for all four, no runaway in V.

100,000 steps at `omega0 = [2.0, 1.5, −1.0]` rad/s: post-normalisation
`||q|−1|` max **2.220e-16** for all four.

Quaternion handling is robust to abuse: a deliberately denormalised initial
quaternion at scale 0.01, 0.5, 1.5 and **10.0** is recovered to
`|q| = 1.0000000000` after **one step**, with final airspeed within 0.26% of
the correct value.

## NB-4 `lax.scan` rollout is bit-identical to a manual Python loop

`a02_longhorizon.py`. 500 steps, all four aircraft:

| comparison | max abs diff | bit-identical |
|---|---|---|
| `rollout` vs manual loop of `step` | **0.000e+00** on all of pos/vel/quat/omega | **yes** |
| `rollout` vs `logged_rollout` | **0.000e+00** | **yes** |

No finding. (Only `jit` vs `disable_jit` differs, at 1e-16 — E-10c.)

## NB-5 `specific_force` and `load_factor` are kinematically exact

`a10c_turn.py`. A proper 7-unknown steady **level** turn
(`[alpha, theta, de, da, dr, throttle, psidot]`, beta = 0, residual
`[udot,vdot,wdot,pdot,qdot,rdot,hdot] = 0`, converged to 1e-15 to 1e-16):

| aircraft | phi | \|a_spec\|/g | sqrt(1+(V·psidot/g)^2) | match |
|---|---|---|---|---|
| boeing747 | 60 deg | 1.96417 | 1.96417 | **exact to 5 dp** |
| boeing747_approach | 60 deg | 1.92207 | 1.92207 | **exact to 5 dp** |
| cherokee | 60 deg | 1.98213 | 1.98213 | **exact to 5 dp** |
| boeing747 | 45 deg | 1.40626 | 1.40626 | **exact to 5 dp** |
| cherokee | 30 deg | 1.15282 | 1.15282 | **exact to 5 dp** |

The specific force matches pure turn kinematics at every bank angle tested.

Two caveats that are **not** failures:

- `n_z` is 1–4% below `1/cos(phi)` because it is **body-normal**, as
  `load_factor`'s docstring states, and the aircraft carries pitch attitude.
- `V·psidot/g` is 1–4% below `tan(phi)` because `beta = 0` is not the same as
  zero side force in this model (the rudder contributes `CYdr*dr`). The
  Cherokee, whose turn needs least rudder, is closest: 1.711 vs 1.732 (1.2%).

Trimmed level flight gives `n_z = cos(theta)` to **2.2e-16** for all four,
exactly as documented. Ballistic free fall with the aero zeroed gives `n_z`
= exactly `−0.0` for all four. (The V=0 offset is E-8, a `V_MIN` artefact, not
a `load_factor` error.)

**My first two attempts at this test were wrong, not the model.** A rotational-
equilibrium-only solve (`a10b_turn.py`) gave `n_z ~ 1.0` at 60 deg bank; a
6-unknown solve without the `hdot = 0` constraint gave a steady *descending
spiral* (gamma = −9.6 deg) and `n_z = 1.96`. Only the constrained level turn is
a valid test. Recording this because a careless version of this check would
have produced a spurious finding.

## NB-6 Wind-field superposition is exact and every field is divergence-free

`a056_wind2.py`. Four fields superposed (two Parks arrays of 5 cores each at
different spacings and altitudes, an updraft column, a lee wave), 2000
randomised points over a 12 km cube:

```
max |W_sum - sum W| = 3.553e-15 m/s      (exact linearity)
max |J_sum - sum J| = 0.000e+00 1/s      (exact, including the Jacobian)
```

Divergence (incompressibility), same 2000 points:

| field | max \|div W\| | peak \|W\| |
|---|---|---|
| vortex array (Hannibal) | 6.94e-18 1/s | 26.4 m/s |
| vortex array (Morton) | 1.39e-17 1/s | 19.3 m/s |
| updraft column | **0.00e+00** | 24.4 m/s |
| lee wave | **0.00e+00** | 6.0 m/s |
| microburst | 8.67e-18 1/s | 41.3 m/s |
| all four superposed | 1.39e-17 1/s | 32.6 m/s |

**Two overlapping vortex arrays** (co-located, 1 m apart, same case): peak |W|
doubles to 66.96 m/s from 33.55, stays finite everywhere, and divergence stays
at 2.78e-17 1/s. Non-finite `W` or `omega_gust` at **0 / 2000** points.

## NB-7 The Parks vortex *value* is continuous at the core edge

Distinct from E-2, which is about the gradient. `|W jump|` across `r = r0` is
2.5e-13 m/s (hannibal) and 2.1e-13 m/s (morton) — machine precision. The
`r2_safe` clamp in `wind.py:133` does not leak into the value, and there is no
NaN at the core centre.

## NB-8 Singular geometries survive integration

`a04_singular.py`. Exactly-vertical flight (theta = +90 deg) integrated for
3000 steps at dt = 0.01 s:

| aircraft | finite | max quat norm deviation |
|---|---|---|
| all four | **True** | **2.220e-16** |

Pure vertical relative velocity (alpha = ±90 deg), pure sideslip (beta = 90
deg), and backwards flight all return finite forces (nonphysically large — see
E-9 — but finite). The DCM stays orthonormal to 6.7e-16 through gimbal lock.

## NB-9 Wind-field guards all hold

`a99_loose.py`. `updraft_wind` on the column axis (r = 0, 1e-15, 1e-9 m) is
finite in both value and Jacobian, including for **fractional** `sharpness`
(0.5, 1.0, 1.5) where `d(r^s)/dr` is singular at the origin — the `1e-12` clamp
does its job. `along_track_shear` with a zero ground track returns 0.0, finite.
`microburst_wind` on the axis and below ground: finite everywhere tested,
including h = −100 km.

## NB-10 Sample-rate control reversals create nothing

`a07d_rev.py`. Elevator slammed to opposite limits **every single step** at
25 Hz through 1000 Hz, throttle 0, and again with aileron and rudder reversing
in phase: `max(E − E0) = +0` and `max rise from a local minimum = 0` in all 32
runs. All finite. Peak |q| falls smoothly with dt (1.68 → 0.19 rad/s for the
Cessna) with no resonance or blow-up.

*(An earlier version of this test left the throttle at trim and appeared to show
energy gain. That was engine work, not a violation. Corrected.)*

---

# ATTACKS I DID NOT GET TO

Honest coverage gaps. None of these were attempted; absence of a finding here
means absence of a test.

1. **Stochastic wind models.** Every field exercised is deterministic. The
   `WindState` / PRNG-key threading in `integrate.step` and the Dryden/von
   Karman path that `integrate.py`'s docstring cites were never driven. E-3's
   order result therefore says nothing about the stochastic case.
2. **`sampled_field_model` and the strip integration.** `wind.sampled_rates`,
   `loads.strip_increment`, `loads.strip_model` and `airframe.stations` were
   not attacked at all. The `CoeffIncrement` path into `load_factor` (which
   `dynamics.py` claims carries only `CL`, not `Cl`/`Cm`/`Cn`) is untested here.
3. **The autopilot.** `autopilot.closed_loop_rollout` and the loop-pairing
   boundary at `minimum_drag_speed` were not attacked. My pumping laws were
   hand-rolled, not the shipped controller.
4. **`batched_rollout` / `vmap` consistency.** Not checked against the serial
   path.
5. **float32.** `jax_enable_x64` is on repo-wide (`__init__.py`); I never turned
   it off. Every precision number above is float64.
6. **Lateral-directional long-horizon mode content.** I perturbed the lateral
   axis and confirmed nothing diverges, but did not analyse spiral or Dutch-roll
   behaviour over the 500k-step runs.
7. **`sensors.py`.** Untouched.
8. **Component-wise Richardson.** E-3 uses a max-abs norm over the whole state.
   Per-component orders (which could differ between position and quaternion)
   were not extracted.
9. **Multi-core vortex arrays where cores overlap within one r0.** I tested two
   arrays 1 m apart, but not the case where a single aircraft is inside two
   cores at once.
10. **Energy accounting with a wind field present.** E-1's energy argument only
    holds in still air; I did not construct the correct energy budget for a
    moving air mass, so nothing above tests conservation during an actual
    turbulence encounter.

---

# CROSS-REFERENCES WORTH ONE LOOK IN PHASE 3

Stated as observations, not causal claims.

- **E-2 and E-3(e) touch the same geometry.** The vortex core edge carries a
  0.283 rad/s step in `omega_gust`, and dt refinement through that same core
  has no convergence order. Whether one explains the other is not established
  here.
- **E-3(c,d) is independent of E-2.** The lee wave and the updraft column are
  C-infinity and still converge at exactly order 1.00, so first-order behaviour
  is present without any discontinuity.
- **E-1 and E-9 both live in `aero.py`'s coefficient build-up** — one in the
  moment channel (`Cmde` with no matching drag), one in the force channel
  (linear `CL` with no stall). Both light aircraft show E-1; both 747s do not.
- **E-5's two instances share a shape**: a control channel with identically
  zero authority makes a Jacobian column vanish, and `jnp.linalg.solve`
  returns NaN rather than raising.
- **E-4 and E-6 share a symptom**: `is_physical` returns `True` for trim
  solutions at 20 km below sea level, and nothing in the integrator objects to
  flying there.
