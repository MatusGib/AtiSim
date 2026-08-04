# JAX Flight Simulator — Design Spec

Date: 2026-08-04

## Purpose

A 6-DOF fixed-wing flight dynamics core in JAX with three aircraft, a cascaded PID autopilot, and
matplotlib visuals. This is a **foundation for later turbulence modelling**. Correctness and
extensibility outrank features.

Out of scope: turbulence, wind shear, gust models, landing gear, stall hysteresis, transonic aero,
engine dynamics, mode-validation suites.

## The two non-negotiable interfaces

These exist so turbulence is a drop-in later rather than a core rewrite.

**1. Aero forces use velocity relative to the air mass.** Angle of attack is never computed from
inertial velocity.

```python
def relative_velocity(vel_body, quat, wind_ned):
    dcm = quat_to_dcm(quat)              # body -> NED
    return vel_body - dcm.T @ wind_ned

def aero_forces_moments(vel_rel, omega_rel, controls, ac, rho): ...
```

`omega_rel = omega - omega_gust` is a parameter from day one, defaulting to zero. Dryden and von
Karman produce both a wind vector and angular-rate perturbations `(p_g, q_g, r_g)`; both hooks must
already exist.

**2. A PRNG key is threaded through `step`.** Turbulence is stochastic, so Monte Carlo ensembles
over turbulence realisations are the eventual goal. That is `vmap` over a batch of keys — but only
if the key is plumbed through from the start. Splitting keys inside a jitted scan is awkward to
retrofit.

```python
def step(state, controls, key, dt, ac) -> (State, key)
```

## Decisions taken during brainstorming

| Question | Decision |
|---|---|
| Third aircraft | **Cessna 172.** Not single-seat, but the best-documented light GA dataset after the Navion and the lowest-risk option. |
| Aero data sourcing | **Web-research each set.** Every derivative carries a traceable citation. Missing derivatives get flagged, never invented. |
| Delivery | **Checkpoint at trim.** Steps 1–5, then report trim residuals and 60 s hold before building autopilot/manual/visuals. |
| Float precision | **`jax_enable_x64 = True`.** float32 is marginal for a Newton trim solve to ~1e-8 and for quaternion norm stability over 1e5 steps. |

## Module structure

```
flightsim/
  units.py        # conversion constants ONLY; no logic
  state.py        # State/Controls NamedTuples, quaternion utilities, x64 config
  atmosphere.py   # ISA density/temperature/pressure vs altitude
  aero.py         # derivative build-up; takes vel_rel, omega_rel
  dynamics.py     # 6-DOF Newton-Euler
  wind.py         # zero-wind stub; turbulence goes here later
  integrate.py    # RK4 step + lax.scan rollout, key threaded
  aircraft.py     # three definitions + Aircraft NamedTuple
  trim.py         # Newton solve for steady level flight
  autopilot.py    # cascaded PID (second pass)
  viz.py          # live animation + post-flight plots (second pass)
  tests/
```

## Data types

All `NamedTuple`, therefore JAX pytrees for free — `jit`/`vmap` work without `static_argnums`, and
`vmap` over aircraft is available if wanted later.

```python
class State(NamedTuple):     # 13 floats
    pos_ned:  Array  # (3,) m, NED from origin; altitude = -pos_ned[2]
    vel_body: Array  # (3,) m/s, u, v, w
    quat:     Array  # (4,) w,x,y,z, body -> NED
    omega:    Array  # (3,) rad/s, p, q, r

class Controls(NamedTuple):
    elevator: Array  # rad
    aileron:  Array  # rad
    rudder:   Array  # rad
    throttle: Array  # 0-1

class Aircraft(NamedTuple):
    # geometry, mass, inertia tensor (including Ixz), all derivatives.
    # SI, per-radian, body-axis. Always.
```

## Aero build-up

From `vel_rel`: `V = |vel_rel|`, `alpha = atan2(w, u)`, `beta = asin(v / V)`,
`qbar = 0.5 * rho * V^2`. Rates non-dimensionalised as `p_hat = p*b/(2V)`, `q_hat = q*c/(2V)`,
`r_hat = r*b/(2V)` using `omega_rel`.

Longitudinal:
```
CL = CL0 + CLa*alpha + CLq*q_hat + CLde*de
Cm = Cm0 + Cma*alpha + Cmq*q_hat + Cmde*de
CD = CD0 + CL^2 / (pi * e * AR)
```

Lateral:
```
CY = CYb*beta + CYp*p_hat + CYr*r_hat + CYdr*dr
Cl = Clb*beta + Clp*p_hat + Clr*r_hat + Clda*da + Cldr*dr
Cn = Cnb*beta + Cnp*p_hat + Cnr*r_hat + Cnda*da + Cndr*dr
```

Lift and drag are wind-axis; rotate to body through `alpha` and `beta` before summing forces.

**Known modelling call:** sources give `CD0` and `CDu` but rarely a full `CD(alpha)` expansion, so
drag uses the parabolic polar above. Oswald efficiency `e` is read from the source where stated and
back-solved from the cited cruise L/D where not. Any aircraft where `e` is back-solved rather than
read is flagged in a comment in `aircraft.py`.

Thrust is `throttle * max_thrust`, with a density-ratio lapse for the 747. The engine is not
modelled.

## Dynamics

Newton-Euler with the full inertia tensor including `Ixz`:

```
vdot   = F_body/m - omega x vel_body
omegadot = I_inv @ (M_body - omega x (I @ omega))
```

`I_inv` is precomputed at load. Gravity enters body axes as `dcm.T @ [0, 0, g]`. Quaternion
kinematics from `omega`; Euler angles are derived for display only.

## Integration and the jit boundary

Fixed-step RK4 at 50–100 Hz. **Wind is sampled once per step, not once per RK4 stage** — held
constant across the step. This is the standard treatment for Dryden/von Karman and avoids four key
splits per step. The quaternion is renormalised after each completed step.

Two drivers sit on one `step`:

- `rollout(...)` — `lax.scan` with the autopilot *inside* the loop, `jit`ed, `vmap`ped over a
  leading batch axis of PRNG keys. The Monte Carlo path. Exists from day one at batch size 1.
- Live interactive loop — plain Python at 50 Hz calling the same jitted `step`. Second pass.

Same physics on both paths; no duplication.

## Units and axes

`units.py` contains named constants and nothing else: `FT2M`, `LB2KG`, `SLUG_FT2_TO_KG_M2`,
`DEG2RAD`, `KT2MS`, `LBF2N`, and so on. Conversion factors are never inlined anywhere else.

Aircraft definitions apply conversions at the point of definition, each with a comment naming the
source table, then store SI / per-radian / body-axis only.

Stability-to-body axis conversion is a rotation about *y* by `alpha_trim`, applied at load:
```
Cl_body = Cl_s*cos(a) - Cn_s*sin(a)
Cn_body = Cn_s*cos(a) + Cl_s*sin(a)
```
and correspondingly for the inertia tensor. CR-2144 Appendix B specifies which convention its
derivatives use; that determines whether the conversion is applied.

## Trim

Newton on three unknowns — `[alpha, elevator, throttle]` — for wings-level steady flight at target
`(V, h)`, with `theta = alpha` (gamma = 0), `beta = 0`, `omega = 0`. Residual is `[udot, wdot, qdot]`.
Jacobian via `jax.jacfwd`, solved with `jnp.linalg.solve`. Pure JAX, so jittable and vmappable over
a (V, h) grid. No new dependency.

A test cross-checks one case against `scipy.optimize.root`, so a bug in the hand-rolled Newton loop
cannot hide.

Every run starts trimmed.

## Autopilot (second pass)

Stateless functional PID: `autopilot(state, ap_state, targets) -> (controls, ap_state)`. Two
cascade levels only:

- Inner: pitch attitude hold, roll attitude hold.
- Outer: altitude → pitch command; heading → bank command; airspeed → throttle.
- Rudder from sideslip feedback for turn coordination.

Anti-windup on integrators; rate and deflection limits applied before the plant. Gains per aircraft,
hand-tuned against the trimmed model. No LQR, no gradient-based tuning.

Bumpless transfer: on engagement, seed integrators from current control positions; on disengagement,
hand current deflections back.

## Visuals (second pass)

Live: `FuncAnimation` with `mpl_connect('key_press_event')` and a held-keys set. Arrow keys
pitch/roll, `-`/`=` throttle, `a` toggles autopilot. Physics at 50 Hz, render ~20 fps, decoupled.
Blitting and `set_data` on pre-allocated artists.

Panel: 3D trajectory trace, artificial horizon, rolling strips for airspeed/altitude/heading.

Post-flight: ground track, altitude profile, airspeed/alpha/beta, control deflections, mode
timeline. Trajectories saved to `.npz` so runs are re-analysable without re-flying.

## Tests

- Quaternion round-trip and norm stability over 1e5 steps.
- DCM orthonormality.
- ISA atmosphere against table values at 0 / 5 000 / 11 000 m.
- Unit conversions.
- Zero-wind relative velocity equals the inertial-velocity case.
- `vmap` batch-of-1 rollout matches the unbatched rollout.
- Trim converges for all three aircraft; trimmed flight holds altitude.
- `jax_debug_nans` enabled across the suite — NaNs inside `jit` are otherwise silent.

## Order of work

1. State, quaternions, atmosphere, units.
2. Aero + dynamics, wind and PRNG hooks in place (zero wind).
3. RK4 + scan rollout, jit and vmap.
4. Aircraft data: 747, Navion, C172.
5. Trim solver. **← checkpoint: report trim residuals and 60 s drift, then stop.**
6. Autopilot.
7. Manual control and switching.
8. Visuals and plots.

## Checkpoint deliverable (end of step 5)

A table of, per aircraft: trim residuals, and altitude/airspeed drift over 60 s from the trimmed
state with fixed controls. A sign or axis error is visible here and nowhere cheaper.

## Escalation rules

- A derivative missing from a source → ask whether to estimate or switch source. Never invent a
  plausible number.
- Trimmed flight will not hold steady → report the residuals. Do not loosen the tolerance.

## Environment

Python 3.10.11, `.venv` in the project root. jax (CPU), numpy, scipy, matplotlib, pytest.
