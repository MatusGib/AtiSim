# INVENTORY — every physical claim the engine makes

Phase 1 of the audit defined in `AUDIT_PROMPT.md`. Built by reading the code,
not the documentation. Where a claim appears only in a docstring it is recorded
as a **claim**, never as evidence.

Statuses here are deliberately absent — this document says *what is asserted*,
`AUDIT.md` says whether it holds.

---

## 0. Map of the engine (Phase 0)

```
                      units.py          (conversion factors, no logic)
                          |
      atmosphere.py  <----+----> aircraft.py ---- provenance.py (ledger, data-about-data)
       (ISA, G0, rho)         (4 airframes, derivative
            |                  transcription + back-solves)
            |                        |
            +----> aero.py <---------+          airframe.py (tail arm, stations,
            |   (coefficients, wind-axis            |         spanwise loading)
            |    -> body-axis rotation,             |
            |    thrust)  *** never sees            v
            |    inertial velocity ***          wind.py (4 fields, gust_rates,
            |        |                           sampled_rates, strip_roll_moment,
            |        v                           superpose, field_model)
            +--> dynamics.py  <------ loads.py (CoeffIncrement, strip_model)
                 (Newton-Euler, specific_force,        |
                  load_factor, f_factor,               |
                  thrust_authority)                    |
                  *** the ONLY place wind enters ***   |
                        |                              |
                        v                              v
                   integrate.py  (rk4_step, step, rollout, logged_rollout,
                        |          batched_rollout; SimState carry;
                        |          WIND SAMPLED ONCE PER STEP)
        +---------------+----------------+-------------------+
        v               v                v                   v
    trim.py       autopilot.py      sensors.py          checks.py
  (Newton, still  (cascaded PID,   (AirData, the only   (per-run gates,
   air only)       air-relative)    supported way to     tripwires, reports)
                                    ask what the
        verification.py (tier 0)    aircraft is doing)  analysis/, apps/, viz.py,
        validation.py   (tiers 1-2)                     vortex_viz.py, panel.py
```

**Where physics decisions are actually made — six places, and only six:**

| # | Decision point | File:function |
|---|---|---|
| 1 | The equations of motion themselves | `dynamics.derivatives` |
| 2 | What the air does to the aircraft (coefficient build-up + wind→body rotation) | `aero.coefficients`, `aero.aero_forces_moments` |
| 3 | What the numbers describing the airframe are | `aircraft._boeing_747` and siblings |
| 4 | What the air is doing (the four fields, and the gradient→rate map) | `wind.py` |
| 5 | How wind couples to the integrator (the once-per-step hold) | `integrate.step` |
| 6 | Where on the airframe the field is sampled | `airframe.py` + `loads.strip_increment` |

Everything else — autopilot, panel, viz, analysis, apps, checks — consumes
physics and does not decide it. `checks.py` is the one borderline case: it
computes energy closure and divergence, which are physics *statements*, but it
never feeds the plant.

**The one structural rule the architecture is built around:** `aero.py` is
handed `vel_rel` and `omega_rel` and never the inertial velocity, so the wind
enters at exactly one seam. This is load-bearing for almost every claim below.

---

## 1. Rigid-body core

| # | Claim | Where | Kind |
|---|---|---|---|
| R1 | State is 13 elements: `pos_ned`(3), `vel_body`(3), `quat`(4), `omega`(3) | `state.State` | structural |
| R2 | NED is inertial; flat, non-rotating Earth; no Earth-rate or transport terms | `dynamics.derivatives` (by omission) | **implicit** |
| R3 | Body axes x-fwd, y-right, z-down; NED z down; altitude = `-pos_ned[2]` | `state.py` docstring + `state.altitude` | convention |
| R4 | Quaternion is `[w,x,y,z]`, unit, rotates **body→NED** | `state.quat_to_dcm` | convention |
| R5 | `v̇ = F/m + g_body − ω×v` — Coriolis uses **inertial** v and ω, never air-relative | `dynamics.derivatives:72` | equation of motion |
| R6 | `ω̇ = I⁻¹(M − ω×Iω)` — full Euler equation, no linearisation | `dynamics.derivatives:73` | equation of motion |
| R7 | `ṗos_ned = DCM·v_body` | `dynamics.derivatives:78` | kinematics |
| R8 | `q̇ = ½ q ⊗ [0,ω]` | `state.quat_derivative` | kinematics |
| R9 | Gravity is a constant `G0 = 9.80665` in NED z, rotated into body | `dynamics.derivatives:70`, `atmosphere.G0` | **assumption** |
| R10 | Inertia tensor is constant, symmetric, `Ixy = Iyz = 0`, `Ixz` enters **negated** | `aircraft.inertia_tensor` | assumption |
| R11 | Mass is constant; no fuel burn | `Aircraft` pytree | **implicit** |
| R12 | Integration is classical RK4, fixed step | `integrate.rk4_step` | numerics |
| R13 | Quaternion is renormalised after every step | `integrate.step:116` | numerics |
| R14 | float64 throughout, enabled at import | `flightsim/__init__.py` | numerics |
| R15 | No explicit `−m·dW/dt` body force — wind enters only through `vel_rel` | `dynamics.derivatives` (by omission) | **deliberate omission, claimed correct** |
| R16 | Euler angles are display-only; 3-2-1 sequence; `arcsin` clipped at ±90° | `state.quat_to_euler` | convention |

**Not present, and nowhere declared:** no gyroscopic engine-rotor term; no
apparent-mass/added-mass term; no structural degrees of freedom; no
CG-offset term between the inertia reference and the aerodynamic moment
reference.

## 2. Aerodynamic model

| # | Claim | Where | Kind |
|---|---|---|---|
| A1 | `CL = CL0 + CLa·α + CLq·q̂ + CLde·δe` — linear in α, **no stall, no α̇** | `aero.coefficients:91` | model form |
| A2 | `Cm = Cm0 + Cma·α + Cmq·q̂ + Cmde·δe` | `aero.coefficients:92` | model form |
| A3 | `CD = CD0 + CL²/(π e AR) + CD_wave` — parabolic polar on the **total** CL | `aero.coefficients:94` | model form |
| A4 | `CY, Cl, Cn` linear in β, p̂, r̂, δa, δr | `aero.coefficients:100-114` | model form |
| A5 | `q̂ = q·c/(2V)`; `p̂ = p·b/(2V)`; `r̂ = r·b/(2V)` — chord for pitch, span for the lateral pair | `aero.coefficients:85-87` | convention |
| A6 | `α = atan2(w,u)`, `β = asin(v/V)` | `aero.air_data` | convention |
| A7 | Wave drag `= 20(M − M_crit)⁴`, Lock's fourth-power law | `aero.wave_drag` | model form |
| A8 | `M_crit = M_dd − (0.1/80)^⅓`, anchored on `dCD/dM = 0.1` at drag divergence | `aero._MDD_OFFSET` | derived constant |
| A9 | Korn equation `M_dd = κ/cosΛ − (t/c)/cos²Λ − CL/(10cos³Λ)` | `aero.drag_divergence_mach` | model form |
| A10 | Lift and drag are wind-axis; side force is body-axis; rotation into body through α and β | `aero.aero_forces_moments:157` | convention |
| A11 | Moments are `q̄S·[b·Cl, c·Cm, b·Cn]` | `aero.aero_forces_moments:165` | convention |
| A12 | Thrust is `throttle × max_thrust × (ρ/ρ₀)^lapse`, along body x, through the CG | `aero.thrust_force` | model form |
| A13 | Airspeed floor `V_MIN = 1.0 m/s` guards the divisions | `aero.V_MIN` | numerics |
| A14 | Derivatives are constant over the whole envelope — one linearisation point per aircraft | `Aircraft` pytree (by omission) | **implicit** |
| A15 | An external `CoeffIncrement` may be added to CL, Cl, Cm, Cn after build-up | `aero.aero_forces_moments:143` | seam |

**Omissions that are physics, visible only in what is absent:** no `CLα̇`/`Cmα̇`;
no `CXu`/`CZu`/`Cmu` Mach content; no `CYp`/`CYr` for any aircraft; no rudder
set at all for the Cessna; no `CD` dependence on β or δe; no thrust pitching
moment; no engine spool dynamics; no control-surface rate limit or actuator lag;
no ground effect; no landing gear; no terrain; no stall; no buffet; no Reynolds
dependence; no aeroelastic degrees of freedom.

## 3. Aircraft data and its provenance

| # | Claim | Where |
|---|---|---|
| C1 | `boeing747` is CR-2144 Table IX-3 flight condition 9: 40,000 ft, M 0.80, W 636,636 lb | `aircraft._boeing_747` |
| C2 | Its longitudinal derivatives come from Table IX-4 dimensional values, inverted through CR-2144 Appendix A | `aircraft._boeing_747:263-268` |
| C3 | Its lateral derivatives come from Table IX-8 **primed** values, un-primed first | `aircraft._unprime` |
| C4 | Tables IX-4 and IX-8 are body-axis, so no stability→body rotation is applied | comment at `aircraft.py:216` |
| C5 | `CD_trim = 0.043` is a **chart read** off Figure IX-6 with ±0.003 | `aircraft.py:238` |
| C6 | `e` is back-solved from `Xw`; `CD0` is the remainder of `CD_trim` | `aircraft.py:290-291` |
| C7 | Sweep 37.5°, t/c 0.09, κ 0.87 are **declared**, not sourced | `aircraft.py:282-284` |
| C8 | `max_thrust` = 4 × 43,500 lbf; lapse exponent 0.8 — declared, "CR-2144 models no engine" | `aircraft.py:344-348` |
| C9 | Control deflection limits 25/20/25° are declared | `aircraft.py:349-351` |
| C10 | `boeing747_approach` is Table IX-2 verbatim, already non-dimensional | `aircraft._boeing_747_approach` |
| C11 | Its mass/inertia are Table IX-3 flight condition 2, chosen over Figure IX-1 in a stated source conflict | `aircraft.py:389-399` |
| C12 | Its `e` and `CD0` are solved exactly from `CD`, `CDα`, `CL`, `CLα` with no chart read | `aircraft.py:434-436` |
| C13 | `cherokee` is McCormick's worked example, dimensional, converted through `FlightCondition` | `aircraft._cherokee_pa28_180` |
| C14 | Its `Izz < Iyy`, flagged by the source's own author and carried as given | comment at `aircraft.py:511` |
| C15 | `cessna172` is Roskam/DATCOM via PyFME, non-dimensional, fitted over α ≤ 10° | `aircraft._cessna_172` |
| C16 | The Cessna's entire rudder set is **zeroed**, deliberately | `aircraft.py:755, 764, 769` |
| C17 | `CYp = CYr = 0` for every aircraft, because no source tabulates them | four places in `aircraft.py` |
| C18 | Each aircraft is trimmed at its own `CRUISE` condition; `CL0 = CL_trim − CLa·α₀`, `Cm0 = −Cma·α₀` | `aircraft.py:272-273, 426-427` |
| C19 | The provenance ledger classifies constants SOURCED / DERIVED / CALIBRATED / DECLARED | `provenance.LEDGER` |
| C20 | *Claim:* "a constant with no ledger entry fails the build" | `docs/PROJECT.md` §2 — **a claim about the code, testable** |

## 4. Atmosphere

| # | Claim | Where |
|---|---|---|
| T1 | ISA, two layers, 0–20 km; troposphere lapse −0.0065 K/m to 11 km, then isothermal 216.65 K | `atmosphere.py` |
| T2 | `p = p₀(T/T₀)^(−g/(L·R))` below the tropopause, exponential above | `atmosphere.pressure` |
| T3 | `ρ = p/(R·T)`, `a = √(γRT)`, dry air, `R = 287.05287`, `γ = 1.4` | `atmosphere.py` |
| T4 | Altitude is geopotential, not geometric; stated error 0.17% at 11 km | module docstring |
| T5 | No weather deviation, no humidity, no temperature offset | by omission |
| T6 | Behaviour above 20 km is **undefined but not guarded** — the isothermal branch simply continues | by omission |

## 5. Wind fields

| # | Claim | Where |
|---|---|---|
| W1 | Parks Rankine vortex: solid-body core inside `r₀`, irrotational `1/r` outside, continuous at `r₀` | `wind.vortex_wind` |
| W2 | Arrays are built by **linear superposition** | `wind.vortex_wind` vmap-sum |
| W3 | `Δψ = 0` is hard-coded — vortex axes perpendicular to track | module comment |
| W4 | Identified cases: Hannibal r₀ 600 ft, V₀ 85 ft/s, spacing 3500 ft; Morton 450/70/3200 | `wind.PARKS_CASES` |
| W5 | Updraft column is super-Gaussian in radius, purely vertical, no altitude variation | `wind.updraft_wind` |
| W6 | `sharpness` is **DECLARED**, not sourced, and dominates the load factor | `UpdraftColumn` docstring |
| W7 | Updraft peak 80 ft/s and 20 s traverse are sourced from Wingrove & Bach 1994 | `wind.UPDRAFT_W0`, `UPDRAFT_SECONDS` |
| W8 | Lee wave is `w = −w₀cos(2π(N−N₀)/λ)`, purely vertical, no altitude variation, divergence-free | `wind.lee_wave_wind` |
| W9 | Lee-wave amplitudes 3.0 / 6.0 m/s zero-to-peak are Doyle et al.'s two legs | `wind.LEE_WAVE_AMPLITUDE` |
| W10 | Wavelength 25 km is **DECLARED** — the source's band is tropospheric | `wind.LEE_WAVE_WAVELENGTH` |
| W11 | Microburst is Oseguera & Bowles Eqs. (5)–(6), an axisymmetric stagnation flow satisfying continuity exactly | `wind.microburst_wind` |
| W12 | Its four constants (1.1212, 0.22, 12.5, 0.2357) are re-derivable from the equations | module comment |
| W13 | The microburst is the only field with a ground; below it the value is **clamped**, not extrapolated | `wind.microburst_wind:457` |
| W14 | `p_g = +∂w_g/∂y`, `q_g = −∂w_g/∂x`, `r_g = +∂v_g/∂x`, body axes | `wind.gust_rates` |
| W15 | `omega_rel = omega − omega_gust` | `dynamics.derivatives:60` |
| W16 | `field_model` derives `omega_gust` from the **analytic Jacobian at the CG** — a first-order correction | `wind.field_model` |
| W17 | `sampled_rates` fits the slope across span and tail arm; identical to `gust_rates` for a linear field | `wind.sampled_rates` |
| W18 | `strip_roll_moment` integrates the field per strip; gust incidence is `−w_g/V` | `wind.strip_roll_moment` |
| W19 | Every field is **frozen** — a function of position only, never of time | `field_model` signature |
| W20 | Fields compose by summation, and that is exact within the model's linearisation | `wind.superpose` |
| W21 | `along_track_shear` = `∇(W·ĥ)·v_ned`, heading held constant in the gradient; the `∂/∂t` term is zero for steady fields | `wind.along_track_shear` |
| W22 | One-way coupling: the aircraft never disturbs the air | whole module, by omission |

## 6. The RK4 / wind seam

| # | Claim | Where |
|---|---|---|
| S1 | Wind is sampled **once per step** and held across all four RK4 stages | `integrate.step:108` |
| S2 | So is the load-model increment | `integrate.step:109` |
| S3 | The claimed justification is "the standard treatment for Dryden and von Kármán" | `integrate.py` module docstring — **a claim** |
| S4 | `SimState.wind_ned`/`omega_gust`/`increment` are an **output cache**, one step stale | `SimState` docstring |
| S5 | Omitting `load_model` passes `None`, so the default path performs no arithmetic at all | `integrate.step:110` |
| S6 | A deterministic wind model returns the PRNG key untouched, so an ensemble meets the same field | `wind.field_model` |
| S7 | `n_steps`, `wind_model`, `load_model` are `static_argnames` — each distinct value recompiles | `integrate.py` decorators |

## 7. Trim, modes and derived quantities

| # | Claim | Where |
|---|---|---|
| M1 | Trim is wings-level, γ = 0 (so θ = α), β = 0, no body rates; residual is `[u̇, ẇ, q̇]` | `trim.residual` |
| M2 | Newton with `jacfwd`, **fixed 40 iterations**, no line search, no convergence test | `trim.trim` |
| M3 | Trim is still-air by construction | `trim.residual` passes zeros |
| M4 | A converged residual does not imply a flight condition; `is_physical` bounds \|α\| at 15° **separately** | `trim.is_physical` |
| M5 | `minimum_drag_speed` sweeps the real drag model rather than using the parabolic closed form | `trim.minimum_drag_speed` |
| M6 | Longitudinal modes come from `jacfwd` of the real nonlinear dynamics in `[u,w,q,θ]`, body axes | `validation.longitudinal_matrix` |
| M7 | Lateral modes are a 4-state `[v,p,r,φ]` reduction holding u, w, θ fixed | `validation.lateral_modes` |
| M8 | `φ̇ = p + r·cosφ·tanθ₀` — the exact kinematic term, which matters ~30% for the spiral root | `validation.lateral_modes:147` |
| M9 | `to_stability_axes` is a similarity transform: every element moves, no eigenvalue does | `validation.to_stability_axes` |
| M10 | `specific_force` **inverts** `derivatives`' force sum rather than recomputing it | `dynamics.specific_force` |
| M11 | `load_factor = −specific_force[2]`, body-normal, `cos θ` in trimmed level flight | `dynamics.load_factor` |
| M12 | F-factor `= U̇ₓ/g − w/Vₐ`, positive hazardous | `dynamics.f_factor` |
| M13 | The hazard metric is the **1 km forward average**, not the instantaneous value | `dynamics.average_f_factor` |
| M14 | Thrust authority `= ((T_avail − D)/W, −D/W)`, with `D = trim_throttle × T_avail` | `dynamics.thrust_authority` |

## 8. Airframe sampling and distributed loads

| # | Claim | Where |
|---|---|---|
| L1 | Effective tail arm `= −Cmq/CLq` chords; the tail lift slope cancels | `airframe.effective_tail_arm` |
| L2 | This **attributes both derivatives entirely to the tail**; the wing/fuselage share is not separated | same docstring |
| L3 | A plausibility band [2, 6] chords gates which aircraft may use the strip path | `airframe.TAIL_ARM_BAND` |
| L4 | 9 span and 9 longitudinal stations, odd so the centreline cancels exactly | `airframe.N_SPAN/N_LON` |
| L5 | Spanwise loading is **DECLARED elliptic**; taper ratio is not recoverable from S, b, c̄ | `airframe.chord_distribution` |
| L6 | Section lift slope is **CALIBRATED**: `a₀ = −8·Clp`, so the strip integral reproduces `Clp` exactly | `airframe.calibrated_lift_slope` |
| L7 | Strip integration supplies **roll only**; pitch and yaw stay at exact zero | `loads.strip_increment` |
| L8 | Only `CL` can reach `specific_force`, because it inverts the force sum; `Cl`, `Cm`, `Cn` cannot | `dynamics.specific_force` docstring |
| L9 | Strip airspeed is air-relative, not ground speed | `loads.strip_increment:91` |

---

## Reconciliation against the prompt's checklist

### What the prompt named, and what I found

| Prompt item | Found | Note |
|---|---|---|
| Quaternion state, RK4, frame/axis transforms | yes | R1–R16 |
| Stability and control derivatives, drag polar, trim solver | yes | A1–A15, C1–C20, M1–M5 |
| Parks / Wingrove / Doyle / Oseguera fields, analytic gradients, `omega_gust` | yes | W1–W22 |
| The `field_model(field)` contract and the RK4 sampling seam | yes | S1–S7 |
| Mode predictions: phugoid, short period, dutch roll, spiral, roll subsidence | yes | M6–M9 |
| Atmosphere and its consistency with the above | yes | T1–T6 |

Nothing on the prompt's list is absent from the engine.

### What I found that the prompt did not name

These are the ones worth the reader's attention, because they were not
anticipated:

1. **The `−ω×v` Coriolis term is a load-bearing physics decision, not
   bookkeeping.** It produces a `∂u̇/∂q = −w₀` element (−62.6 ft/s at cruise)
   that CR-2144's own linear model does not have. The engine is *more*
   complete than its reference here, and no comparison in the project accounts
   for it.
2. **`specific_force` inverts the plant rather than recomputing the force.**
   That is a structural claim with a physical consequence — it is *why* a
   rolling-moment increment provably cannot reach the load factor (L8). No
   checklist item covers "quantities recovered by inversion".
3. **The moment reference point is nowhere in the code.** Derivatives, the
   inertia tensor and `specific_force` all silently assume one common origin.
   CR-2144 Table IX-3 does state it (0.250 MGC for both flight conditions used),
   so this is checkable — but the engine records it nowhere.
4. **The engine has no CG-to-sensor offset**, while CR-2144 Table IX-3 supplies
   `LXP = 86.0 ft`, `LZP = −10.0 ft` — the pilot-station offsets. The
   accelerometer assumption (`ASSUMPTIONS` B4) is declared *unboundable*; the
   source supplies the arm needed to bound it.
5. **`trim.trim` has no convergence test at all** — a fixed 40-iteration
   `lax.scan`. Convergence is asserted only by callers that happen to check the
   residual. `is_physical` is opt-in and is not called by `trim`.
6. **Module-global mutable state**: `airframe._active_shape` is set by a context
   manager. A physics parameter that is process-global is a claim about
   thread-safety and about run reproducibility that nothing tests.
7. **`static_argnames` on `n_steps`/`wind_model`/`load_model`** makes recompiles
   depend on Python object identity of a closure. Two structurally identical
   wind models are different cache keys; the same field rebuilt is a fresh
   compile. This is a performance claim, but it also means "the same field"
   is not a well-defined notion to the cache.
8. **The atmosphere has no ceiling guard** (T6). Above 20 km the isothermal
   branch continues silently; the module docstring's "0 to 20 km" is not enforced.
9. **The updraft column and the lee wave are divergence-free but not solutions
   of anything.** Both are purely vertical with no compensating horizontal flow.
   Divergence-free is necessary, not sufficient — neither satisfies a momentum
   equation, and neither is claimed to.
10. **`wave_drag` acts on the *total* CL**, including the `CLq·q̂` and `CLde·δe`
    contributions. So elevator deflection changes wave drag instantaneously.
    Defensible, but it is a modelling decision nobody declared.
11. **Circularity by construction in the strip calibration** (L6 with the test
    that the strip integral reproduces `Clp`). The code's docstring is honest
    about it; `PROJECT.md` §4 lists it in a table of measurements without the
    label.

### What the prompt named that I could not find as a distinct claim

- **"Conservation properties."** The engine asserts none. There is no energy
  or momentum invariant in the code; conservation appears only as *measured
  drift* in tests. That is the honest position for a system with
  non-conservative forces, but it means "conservation" is not a claim the
  engine makes — it is a claim the test suite makes about the integrator.

---

## The prompt's Phase 1 is under-specified in one way, and it matters

The prompt asks for "an implicit assumption visible only in what the code
omits" and calls it the hardest category. It is — but the harder category is
one level up: **an assumption visible only in what the code's *comparisons*
omit.** The clearest instance found here is that `PROJECT.md` §4's 747 cruise
mode table compares phugoid **ωn** and short-period **ζ** against CR-2144 and
stops there. CR-2144 Table IX-5 flight condition 9 publishes **four**
longitudinal denominator factors, not two: `Z(DET)1 = .0489`, `W(DET)1 = .0673`,
`Z(DET)2 = .387`, `W(DET)2 = .964`. The two the project never compares are
phugoid **ζ** (engine 0.0560 against 0.0489, **+14.4%**) and short-period **ωn**
(engine 0.9508 against 0.964, −1.4%). The first of those is the same order as
the two errors the project does report, and it appears nowhere in the project's
record. An inventory built only from the code cannot surface that; it took
building the comparison the project did not build.

*(Two process notes, recorded because both produced a wrong answer first.
(1) An earlier pass read `Z(DET)1` as `−.0489` off a 200 dpi render and briefly
concluded that the engine and its source disagreed about the *sign* of phugoid
stability. Re-rendering at 600 dpi showed no minus sign — and showed that minus
signs elsewhere in the same column render clearly at that magnification. (2) A
clean-room rebuild from Table IX-4's derivatives then suggested phugoid damping
was unattributable; it is not, and the rebuild was the thing at fault. Both
claims were withdrawn before reaching `AUDIT.md`; see `AUDIT.md` §2.3.)*
