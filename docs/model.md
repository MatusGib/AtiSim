# Model and assumptions

## The model

**State.** `State` holds the position in NED (`pos_ned`), body-axis velocity (`vel_body`:
u, v, w), attitude quaternion (`quat`: w, x, y, z, rotating body to NED) and body rates
(`omega`: p, q, r). `Controls` holds the elevator, aileron and rudder (rad) and the throttle
(0–1).

**Dynamics.** `dynamics.derivatives` evaluates the Newton–Euler equations for a rigid body in
body axes. Gravity follows the inverse-square law with altitude, and the atmosphere is the
two-layer International Standard Atmosphere, read on geopotential altitude.

**Aerodynamics.** `aero.coefficients` builds the force and moment coefficients from stability
derivatives. Lift is linear in α, except for aircraft that carry a CL(α) table, and includes
pitch-rate, α̇ and control terms. Drag is a parabolic polar with a Korn wave-drag rise.
Prandtl–Glauert compressibility scales the longitudinal lift-slope family. Thrust acts along
the aircraft's thrust line, which runs along body x through the CG unless the aircraft declares
its own line (the 747 does).

**Wind coupling.** Wind reaches the aerodynamics only through the air-relative velocity and
rate:

```text
vel_rel   = vel_body − Cᵀ · wind_ned
omega_rel = omega − omega_gust
```

The Coriolis, gyroscopic and kinematic terms keep the **inertial** velocity and rates. A gust
changes the flow the wings see, not the airframe's velocity over the ground. Putting `vel_rel`
into the Coriolis term breaks Galilean invariance, and adding an explicit `−m·dW/dt` term
counts the gust twice. The solver-validation notebook tests for both errors.

**Integration.** `integrate.step` takes one fixed RK4 step. It samples the wind once per step
and holds it across the four stages, and re-normalises the quaternion afterwards. A PRNG key is
threaded through every step, so an ensemble varies only its stochastic part.

**Trim and linearisation.** `trim.trim` solves for angle of attack, elevator and throttle in
steady level flight by Newton iteration. `validation.longitudinal_matrix` and `lateral_modes`
linearise about that point with `jax.jacfwd`.

## Assumptions

Every modelling assumption is listed below with its effect and, where one exists, a measured
bound. The code refers to them by these identifiers.

### A. Frames and Earth

| | Assumption | Effect |
|---|---|---|
| A1 | Flat, non-rotating Earth; NED is inertial | Coriolis 0.0035 g and transport 0.0009 g at 747 cruise; 6.9 m of position error over 20 s. Sound for encounters of seconds to minutes and a few kilometres. Gravity does not vary with latitude (0.53% between equator and pole). |
| A4 | There is no ground | Nothing stops a trajectory at h = 0, and the atmosphere keeps extrapolating below it. Truncate runs at your own clearance, as `scripts/microburst.py` does. |

### B. Mass and structure

| | Assumption | Effect |
|---|---|---|
| B1 | The airframe is rigid | CR-2144's derivatives are for the **flexible** airframe. Comparing two CR-2144 conditions at the same Mach shows every coefficient softening at 2.48× the dynamic pressure, which is the direction aeroelastic relief predicts. This is the most significant open assumption, so make no claims about structural response. |
| B2 | Constant mass and inertia; no fuel burn | Fuel burn is about 2.9 kg/s at 747 cruise, which is negligible over an encounter. |
| B3 | Symmetric aircraft: Ixy = Iyz = 0 | Every source tabulates only Ixz, which the model carries. |
| B4 | The accelerometer is at the CG | A real flight-recorder accelerometer also reads rotational terms. This is one reason load comparisons with recorded data stay as orderings. |
| B5 | The 747's slug mass uses g₀ = 32.174 ft/s² | 1.5 × 10⁻⁶ relative error, for this aircraft only. |

### C. Aerodynamics

| | Assumption | Effect |
|---|---|---|
| C1 | Lift is linear in α unless the aircraft carries a CL(α) table | **No stall** on the linear entries. Results past about 10° are outside the model. The 737 entries carry JSBSim's table. |
| C2 | Quasi-steady aerodynamics: no unsteady lag | The α̇ derivatives are carried where the source tabulates them (the 747's `Cmadot`). Leaving `Cmadot` out predicts, and produces, a −12% short-period damping error. |
| C3 | Stability derivatives are constant across the envelope | The 747 carries CR-2144's Mach derivatives. Every other variation with α and altitude is frozen, so quote the excursion with every result. |
| C4 | Parabolic drag polar plus a Korn wave-drag rise | Within 0.004 in CD near the fit point, and up to 0.014 below Mach 0.75. |
| C5 | No engine dynamics | Thrust responds instantly. Spool-up is needed before any result that depends on a powered recovery. |
| C6 | Control surfaces move instantly | A real 747 elevator takes about 0.24 s for a full manoeuvre deflection, 3.7% of the short period. |
| C7 | Drag responds to the total CL, including the pitch-rate and control terms | The model carries an `X_q` that CR-2144 does not model; the engine's value agrees with the polar's prediction to 0.9%. |
| C8 | Wave drag acts on the total CL | A small, measured addition at cruise. |
| C9 | Every lift increment acts at the CG's relative wind | This violates energy conservation at extreme pitch rates (84–201°/s), but no state inside the envelope shows it. |
| C10 | Aileron deflection is a compound control treated as one angle | This matches the definition in the source's derivatives. Do not compare `aileron_limit` with the travel of a single surface. |
| C11 | Two routes to the tail arm (`Cmq/CLq` and `Cmδe/CLδe`) disagree | 1.4–2.9% on the 747, and 2.0–2.9× on the light aircraft. Do not adjust either derivative to make them agree. |

### D. Atmosphere

| | Assumption | Effect |
|---|---|---|
| D1 | ISA exactly, dry air, two layers to 20 km | The wind fields are the weather in this model, and each comes from a source that states its own conditions. |
| D2 | The atmosphere has no ceiling and no floor | Temperature holds at 216.65 K above 11 km, and the lapse rate continues below sea level. |

### E. Wind and turbulence

| | Assumption | Effect |
|---|---|---|
| E1 | One-way coupling: the aircraft does not disturb the air | Sound for atmospheric fields. It does **not** hold for wake-vortex encounters. |
| E2 | The aircraft is a point for translational gusts, with first-order gust rates | Sound when gusts are larger than about three wingspans. For the Hannibal cores (2.3–3.1 spans), sampling across the airframe moves the load increment by at most 4.4%. |
| E3 | Frozen fields: wind depends on position, not time | Sound for vortices, updrafts, microbursts and lee waves, whose evolution is slow against a traverse. |
| E4 | Wind is sampled once per step and held across the RK4 stages | This keeps a stochastic realisation independent of the time step, but makes the scheme first order in a spatially varying field. At the published step it moves the in-core pitch change by 0.82%. |
| E5 | Superposition is exact, except for the microburst's ground condition | Adding any field with a non-zero ground value to a microburst puts wind through the ground. |
| E6 | The microburst's outflow depth is not modelled | The on-axis downdraft grows with altitude to an asymptote. Fly microbursts only in the low-altitude band the default parameters target. |
| E7 | The along-track shear index needs the caller's turn rate | Worth ΔF = 0.14 in a standard-rate turn, and zero in straight flight. |
| E8 | The longitudinal sampling stations all lie aft of the CG | The sampled pitch rate is a backward difference, so it is biased on the upstream half of a vortex core. |
| E9 | The Rankine core boundary is resolved to the outside | The two branches agree to about 10⁻¹³ at the boundary, so the choice cannot be seen in any value. |
| E10 | Spanwise variation needs a line-vortex field | Every other field is a function of along-track distance, so it cannot roll the aircraft. `line_vortex_wind` varies across the span, but the lateral response it produces is not validated. |
| E11 | A response spectrum assumes a stationary record | With fixed controls, the flight condition drifts over a 100 s record: up to 13% of airspeed at the upper turbulence intensity. Report the drift with the spectrum. |
| E12 | The published descriptions of the Hannibal vortex disagree | Each set of core radius, strength and spacing is used exactly as its paper gives it, and sets are never mixed. Taking the radius from one paper and the strength and spacing from another moves the peak-to-peak load by about 4%. |

### F. Numerics

| | Assumption | Effect |
|---|---|---|
| F1 | Fixed-step RK4 | The observed order of accuracy is 3.99982 on a closed-form problem, and 3.989 through the full dynamics. |
| F2 | The quaternion is re-normalised every step | The norm error is 1.1 × 10⁻¹⁶ after 3,000 steps, so the projection removes essentially nothing. |
| F3 | float64 throughout | A Newton trim to 10⁻¹⁰ and the quaternion norm over 10⁵ steps are both marginal in float32. |
| F4 | There is a round-off floor on trajectory differences | At 40,000 ft the discretisation error reaches round-off near 7 × 10⁻¹¹ m. Refining past dt = 1/128 s makes the answer worse. |
| F5 | The strip integral uses 9 spanwise stations | It returns 82.6% of its own calibration target, and needs about 56 stations for 1%. This affects rolling loads only. |
| F6 | The spanwise loading shape is process-global state | This is correct single-threaded. Check it before calling the strip path from more than one thread. |
| F7 | A control with zero authority makes a Newton solve return NaN silently | This applies only to solvers that trim with rudder as an unknown; none ships. |
