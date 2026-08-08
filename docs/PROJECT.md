# JAX Flight Simulator — project record

A 6-DOF fixed-wing flight-dynamics core in JAX, built as a foundation for turbulence
modelling. This document is the standing record: what exists, what is validated, what is
known-broken, and what happens next.

**Last updated:** session 9 (the microburst, and the 1 km averaged F-factor).

**To run any of it, see §10.**

---

## How to update this document

This file is split into **stable** sections (§1–§5) and **volatile** ones (§6–§9). A
session normally edits only the volatile ones.

| When you… | Edit |
|---|---|
| finish any session | §9 session log — add an entry at the top |
| land a new module or change a public API | §2 architecture **and** §10 running it |
| add or change a script, flag or key binding | §10 running it |
| measure a number against a source | §4 evidence ledger — never delete a row, supersede it |
| find a gap the source cannot fill | §5 gaps — say which source failed and why |
| find a bug that does not fail a test | §6 latent bugs |
| complete or re-order planned work | §7 plan |
| discover something that changes the approach | §8 open questions |

Two rules carried from `CLAUDE.md` and enforced throughout the code:

- **Flag, never invent.** Every number carries the table it came from. A parameter the
  source does not supply is named as a declared modelling choice, not given a plausible
  default. If you add a number here without a citation, you have broken the project.
- **Do not edit a tolerance to make a test pass.** §4's "validated baseline" files are
  off-limits to feature work; if one moves, something real broke.

---

## 1. What this is

Quaternion state, fixed-step RK4, `lax.scan` rollout, `jit` + `vmap` over PRNG keys.
Three aircraft, a cascaded PID autopilot, manual control, matplotlib visuals, and a
turbulence layer under construction. Float64 throughout (`jax_enable_x64`, set before any
array is created — a Newton trim solve to 1e-10 and quaternion norm stability over 1e5
steps are both marginal in float32).

The original design spec is `docs/superpowers/specs/2026-08-04-jax-flight-sim-design.md`.
It remains accurate on architecture. **It is stale in one respect: it lists turbulence as
out of scope, which is no longer true** — that was always the intended destination, and
the two "non-negotiable interfaces" it names exist precisely so turbulence could be added
without a core rewrite. Both have now been exercised and both held.

## 2. Architecture

| Module | Responsibility | Notes |
|---|---|---|
| `units.py` | conversion constants only | no logic; factors are never inlined elsewhere |
| `state.py` | `State`/`Controls`, quaternion utilities | NED inertial, body x-fwd/y-right/z-down; quat is `[w,x,y,z]`, body→NED |
| `atmosphere.py` | ISA to 20 km | two layers — the 747 cruise sits above the tropopause |
| `aero.py` | coefficient build-up | **takes `vel_rel`/`omega_rel` only; never sees inertial velocity** |
| `dynamics.py` | 6-DOF Newton-Euler, `load_factor`, `f_factor`, `average_f_factor`, `thrust_authority` | wind enters here and nowhere else |
| `wind.py` | wind fields and composition | vortex array, updraft column, lee wave, microburst, `superpose`, `field_model`, `along_track_shear` |
| `integrate.py` | RK4 `step`, `rollout`, batched rollout | wind sampled once per step, held across the four stages |
| `aircraft.py` | three aircraft + `REGISTRY`/`CRUISE` | every derivative cites its source table; `FlightCondition` + `from_dimensional_*` do the conversions |
| `sensors.py` | `AirData`, `sense(state, wind_ned)` | **the only supported way to ask what the aircraft is doing**; air-relative where a real sensor is |
| `trim.py` | Newton solve for steady level flight | still-air by construction, and must stay so |
| `autopilot.py` | cascaded PID | per-aircraft gains; bumpless engage |
| `manual.py` | manual control, mode switching, pitch trim | trim moves the stick's centring point, never `controls` |
| `panel.py` | live cockpit, instruments, `Stick`, `LiveSim`, `run_live` | basic T + test overlay; takes a `wind_model` and a `field_range` |
| `viz.py` | `Trajectory`, `Recorder`, `derived`, `post_flight` | the log and the post-flight figure only; no simulator needed to read a run |
| `vortex_viz.py` | encounter analysis and the Fig. 8 figure | air-relative throughout; deliberately separate from `viz.py`. `fly` for a wind field with fixed controls, `manoeuvre` for an elevator schedule at zero wind; both go through `_measure`, so the three Fig. 8 points cannot drift apart |

### The two interfaces turbulence depends on

1. **Aero is air-relative.** `dynamics.py` forms `vel_rel = vel_body - dcm.T @ wind_ned`
   and `omega_rel = omega - omega_gust`, then passes only those to `aero.py`. The
   Coriolis, gyroscopic and kinematic terms deliberately keep the **inertial** velocity
   and rate — a gust changes the flow the wings see, not the airframe's ground velocity.
   Substituting `vel_rel` into the Coriolis term breaks Galilean invariance; adding an
   explicit `-m·dW/dt` term double-counts. Both are classic gust-modelling errors.
2. **A PRNG key is threaded through `step`.** Deterministic components return it
   untouched, so a batch of keys varies only the stochastic part — every member of an
   ensemble meets the same vortex at the same place.

### Wind model contract

```python
wind_model(wind_state, state, key, dt) -> (wind_ned, omega_gust, wind_state, key)
```
`wind_ned` is NED; **`omega_gust` is body-axis** (it is subtracted from `state.omega`).
`state.pos_ned` is available, so a spatial field needs no signature change.
`field_model(field)` wraps any position-only field and derives `omega_gust` from its
analytic gradient, so a component cannot contribute a translational gust while silently
omitting its rotational one.

Gust-rate signs, derived from the repo's own conventions:
`p_gust = +∂w_g/∂y`, `q_gust = −∂w_g/∂x`, `r_gust = +∂v_g/∂x`.

## 3. Sources

| Source | Supplied | Known gap |
|---|---|---|
| NASA CR-2144 (Heffley & Jewell 1972), §IX | 747 geometry, inertia, dimensional derivatives, transfer-function factors, drag figure | no non-dimensional cruise set; **no buffet-onset data at all** |
| McCormick (via a worked example) | Cherokee PA-28-180 dimensional derivatives | no second source for the lateral set; `Izz < Iyy` flagged by its own author |
| Roskam / USAF DATCOM via PyFME | Cessna 172 non-dimensional tables | rudder derivatives omitted and inconsistent — the whole rudder set is zeroed |
| Nelson / Etkin / McRuer | Navion per-radian derivatives | no extractable published mode table was found; tests assert ranges, not values |
| **Parks, Wingrove, Bach & Mehta 1985**, J. Aircraft 22(2) 124–129 | **the vortex model** — Rankine core, array by superposition, and identified parameters | α is *inferred* from accelerometers through an assumed aero model — see §5 |
| **Wingrove & Bach 1994**, J. Aircraft 31(4) 753–760 | updraft magnitudes/duration, g-load statistics, the Fig. 8 discriminator | never identifies an aircraft type; no updraft edge gradient; no lateral data |
| **Doyle, Jiang, Smith & Grubišić 2011**, *Mon. Wea. Rev.* 139, 3–23, DOI 10.1175/2010MWR3466.1 | **the lee-wave amplitudes** — T-REX Gulfstream V over the Sierra Nevada, IOP 4 primary wave, 6 and 12 m/s crest-to-trough | gives a **tropospheric** wavelength band (20–35 km) and says stratospheric ones are shorter **without a number** — see §5 |
| **Proctor, Hinton & Bowles 2000**, 9th Conf. Aviation Range & Aerospace Meteorology, paper 7.7, 482–487 | **the F-factor** — Eq. (3) `F = U̇ₓ/g − w/Vₐ`, Eq. (4) for the shear term, Eq. (7) for the **1 km average**, the `F > (T−D)/W` thrust criterion, the 0.1/0.13 thresholds, and F = 0.2–0.36 in real accidents | its thresholds are **low-altitude** (§4.1 bounds the threat below 500 m) **and jet-transport only** — it states the scale and threshold "are yet to be determined" for piston aircraft |
| **Oseguera & Bowles 1988**, NASA TM-100632 | **the microburst** — Eqs. (5)–(6), an axisymmetric stagnation flow satisfying continuity, with four stated constants (r/R = 1.1212, z_m/z* = 0.22, z*/ε = 12.5, u_max = 0.2357λR) | the example's `R` is legible only in a scanned figure, so the downdraft radius is declared inside the 1–4 km band Wilson et al. use to define a microburst |
| MIL-F-8785C | (not yet used) Dryden spectra | σ above 2000 ft is a **chart read**, not a formula — must be digitised |

### The vortex model, as cited

Parks §"Vortex Modeling", Eqs. (3)–(6): *"a rotational (solid-body) core embedded in an
irrotational flow"*, axis horizontal and perpendicular to the wind vector, with
`r = (ℓ²cos²Δψ + d²)^½`:

| | horizontal `w_xy` | vertical `w_z` |
|---|---|---|
| outside (`r ≥ r₀`) | `V₀r₀d/r²` | `−V₀r₀ℓcosΔψ/r²` |
| inside (`r < r₀`) | `V₀d/r₀` | `−V₀ℓcosΔψ/r₀` |

Arrays are linear superposition. Identified cases, both DC-10s near the tropopause:

| Case | Altitude | r₀ | V₀ | Spacing | Spacing/diameter |
|---|---|---|---|---|---|
| 1 Hannibal MO | 37,000 ft | 600 ft | 85 ft/s | 3500 ft | 2.92 |
| 2 Morton WY | 39,000 ft | 450 ft | 70 ft/s | 3200 ft | 3.56 |

Parks checks that ratio against Scorer's theoretical 2.7 — which is what turned the array
spacing from a free parameter into a cited one.

> **Note a source conflict:** Wingrove & Bach 1994 Fig. 4 gives Hannibal's core diameter
> as 1000 ft; Parks 1985 gives r₀ = 600 ft, i.e. **1200 ft**, and its abstract states the
> range "900 to 1200 ft". Parks is the primary identification source and is used here.

## 4. Evidence ledger

Every figure below is measured, with the tolerance the test asserts.

### Integrator and rigid body

| Check | Measured | Tolerance |
|---|---|---|
| Angular-momentum magnitude drift, 600 s / 60,000 steps | 5.7e-13 | 1e-11 |
| Angular-momentum direction drift | 1.5e-6 deg | 3e-5 deg |
| Rotational KE drift | 9.3e-13 | 2e-11 |
| Quaternion norm, 1e5 steps | holds | atol 1e-12 |
| Coordinated turn vs `g·tanφ/V`, 25.4° bank | 1.12% | 3% |
| Zero-strength wind vs still air, 2000 steps | **bit-identical**, max diff 0.0 | `np.array_equal` |

### 747 modes vs CR-2144

| Mode | Model | Reference | Error |
|---|---|---|---|
| Dutch roll ωn | 0.943 | 0.947 rad/s | 0.4% |
| Dutch roll ζ | 0.0361 | 0.0349 | 3.4% |
| Roll τ | 1.795 s | 1.779 s | 0.9% |
| Spiral τ | 138.0 s | 137.0 s | 0.8% |
| Phugoid ωn (as shipped) | 0.0553 | 0.0673 rad/s | 17.8% — attributed, §5 |
| Short-period ζ (as shipped) | 0.3425 | 0.387 | 11.5% — attributed, §5 |
| Phugoid / short period (augmented model) | — | — | ~1% |

Superseded by the session-5 `Mq` fix (§6d), kept per §4's rule: short-period ζ read
**0.338 / 12.6%** and phugoid ωn **0.0554** while `Mq` was −0.330. Short-period ωn moved
0.9493 → 0.9508.

The same fix moved the encounter table below, because `Mq` is pitch damping and those runs
are open-loop pitch responses: first-core Δθ **2.20 → 2.24 deg**, updraft Δθ
**4.39 → 4.37 deg**. Superseded values recorded here; no conclusion in §5 or the Fig. 8
mechanism changes, since both are orderings rather than values.

### Air-relative sensing (session 5)

| Check | Measured | Tolerance |
|---|---|---|
| Still-air sensing unchanged by the fix | exact | atol 1e-12 |
| Airspeed in a 25 m/s headwind vs groundspeed + 25 | exact | rel 1e-6 |
| Δα from a 10 m/s updraft at 236 m/s | 2.43 deg = atan(10/236) | rel 0.02 |
| Autopilot into a 25 m/s headwind: airspeed on target | yes | ±1.5 m/s |
| …and groundspeed 25 m/s below it | yes | ±1.5 m/s |
| Attitude/rates unmoved by the wind correction | exact | atol 1e-12 |
| Bugs (a)+(b) re-introduced → their tests go red | 2 failed, 219 passed | — |

### The live flying interface (session 6)

| Check | Measured | Tolerance |
|---|---|---|
| Live loop in still air, with and without an explicit `zero_wind` | **bit-identical** | `np.array_equal` |
| Live loop through an updraft vs still air, 40 frames | 1.0 m of altitude | > 1.0 m |
| `vortex.py` unmoved by the constants move: first-core Δθ | 2.240 deg | was 2.24 |
| …updraft Δθ / peak load | 4.366 deg / −1.235 g | were 4.37 / −1.23 |
| Slip ball vs β under held rudder | **opposite signs** | product < 0 |
| Stick ramp: 10 steps in 1 frame vs in 10 frames | identical | `approx` |
| One second of held trim, all three aircraft | 0.25 × full stick | rel 0.05 |
| Suite | 256 tests, 207 s | — |

The bit-identical row is the one that matters: it is the same statement §4 already
makes about a zero-strength wind model, applied to the live loop, and it is what says
the wind hook did not perturb the default path.

### Panel frame rate (session 7)

Medians of 120 runs, Agg backend, 12-core machine, 7 Aug 2026. The "before" column is
the panel as session 6 shipped it; the "after" column is the same panel with `sense`
and `accelerometers` jitted.

| Item | Before | After |
|---|---|---|
| jitted RK4 step | 0.114 ms | — |
| `sense` | 6.56 ms eager | 0.028 ms jitted |
| `accelerometers` | 10.70 ms eager | ~0.03 ms jitted |
| blit, 14 axes / 72 artists | 28.6 ms | 30.1 ms |
| whole frame | 73.0 ms | 37.8 ms |
| achieved rate | 13.7 fps | 26.4 fps |

Session 6 shipped the re-laid-out panel **below** its own 20 fps target and did not
know it, because the target is asserted nowhere and the only measurement on record
(§10's 19.9 fps) predated the re-layout. The two sensing calls were the cost: both ran
eagerly once per frame, and together they were 17.3 ms of a 73.0 ms frame. Blitting is
now the floor — 30.1 ms of 37.8 — and it did not improve, which is the expected result
of jitting something that was never the bottleneck's neighbour.

### Vortex, updraft and manoeuvre encounters (747 at CR-2144 FC9)

| Quantity | Measured | Reference |
|---|---|---|
| Gust spacing, Parks Case 1 | 4.52 s | "about 5 s apart" |
| In-core Δθ, first core | 2.24 deg | Fig. 8 vortex ≈1.4 deg |
| In-core Δθ, second core | 4.17 deg | response builds through the array |
| Whole-run Δθ | 8.33 deg | the phugoid, **not** the encounter |
| Peak load excursion, vortex | −1.23 g | — |
| In-column Δθ, updraft (sharpness 6) | 4.37 deg | paper states 5.2 deg; Fig. 8 cluster 6.2 |
| Updraft Δθ across sharpness 2→10 | 3.63 → 5.34 deg | the declared parameter's influence |
| Air-relative vs inertial α, peak difference | 7.0 deg | — |
| corr(n_z, α) air-relative / inertial | 0.9990 / 0.5572 | — |

### The manoeuvring case (session 7)

Zero wind. Elevator pulse of one short period, **declared**; the deflection is
**bisected**, not chosen, so the sourced quantity is the load and the angle is an output.

| Quantity | Measured | Reference |
|---|---|---|
| Elevator to reach the band | 8.926 deg from trim | derived by bisection, tol 1e-5 rad |
| Load excursion, in-pulse | −1.900 g | Fig. 8 band −2.01…−1.69, **increment** reading |
| Δθ, in-pulse | 30.37 deg | Fig. 8 manoeuvring 12.0 |
| Peak \|α\| in-pulse | 10.31 deg | **marginal** — §7's 10–12 deg amber band |
| n_z at the first sample | 0.9967 | the trimmed value, i.e. the lead-in worked |
| Δθ, whole run vs in-pulse | 30.74 vs 30.37 deg | 1.2% — the window barely matters here |
| Fig. 8 pitch ordering, model | 2.24 < 4.37 < 30.37 | paper 1.4 < 6.2 < 12.0 — **ordering holds** |
| Suite | 260 tests, 104 s | was 256, 126 s on the same machine this session |
| Suite (session 8) | 270 tests, 167 s | the lee wave added 10 |
| Suite (session 9) | 284 tests, 272 s | the microburst added 11, the averaged index 3 |

Two of those rows are the result and the rest are the guard. **The ordering holds**, which
is the only claim §5 permits. **The absolute values do not agree** and are not meant to:
30.37 against 12.0 is 2.5×, in a comparison whose reference aircraft the paper never
identifies.

The whole-run/in-pulse row is worth keeping for contrast: the vortex moves 2.24 → 8.33 deg
between the two windows and the manoeuvre moves 30.37 → 30.74. A manoeuvre is bounded —
the elevator comes back — so the window rule barely bites. For a vortex the aircraft is
left ringing and it bites hard. That asymmetry is why §8's window rule had to be stated
before the third point could be computed rather than after.

### The Fig. 8 mechanism

747 short period is 6.609 s undamped. A **Hannibal** core traverse is 1.550 s, 0.235 of
that (impulsive); a 20 s updraft is 3.026 (quasi-steady). **That 12.9× separation in
non-dimensional encounter duration is what separates the two weather categories**, and it
is a rigid-body timescale effect requiring no nonlinear aerodynamics — which is why this
model reproduces the clustering while it can never reproduce the ±g asymmetry.

> **Superseded, session 7: this said 17×, and 17× is the wrong case.** It is Morton's
> ratio — r₀ = 137.16 m gives a 1.163 s traverse and 20/1.163 = 17.2 — while every run,
> figure and ledger row in this project uses Hannibal, whose r₀ = 182.88 m gives 1.550 s
> and 12.9. Nothing downstream moves: the discriminator is an ordering claim and both
> ratios are an order of magnitude. Corrected because §8's window table now states the
> durations to four figures and a reader would otherwise find them contradicting this
> paragraph.

**The manoeuvre is not a third point on this scale, and that is the point.** Its pulse is
6.609 s — 1.000 short periods, *between* the vortex and the updraft — yet it lands at
30.37° of pitch, far right of both. So duration does not order the three categories, and
timescale is not "the whole discriminator" once the third one exists. It separates the two
**weather** categories. The manoeuvre separates for a different reason: the elevator is
moving, so pitch follows the stick rather than the air. That is precisely the distinction
Wingrove & Bach's chart was drawn to make, and it is why `vortex_viz.fly` holds its
controls fixed.

### Mountain lee wave and the F-factor (session 8)

747 at CR-2144 FC9, fixed controls, three wavelengths flown. The thrust envelope is
**recomputed**, not taken on trust from session 2 — it agrees.

| Quantity | Measured | Reference |
|---|---|---|
| (T−D)/W, full throttle | **+0.0234** | session 2 recorded +0.023 — confirmed |
| (T−D)/W, idle | **−0.0657** | session 2 recorded −0.066 — confirmed |
| Peak F, north leg (w₀ 3.0 m/s) | **+0.01291** | **within** thrust authority |
| Peak F, south leg (w₀ 6.0 m/s) | **+0.02621** | **exceeds** +0.0234 — unrecoverable by thrust |
| Critical amplitude, F = full throttle | **w₀ = 5.51 m/s** | Doyle's two legs are 3.0 and 6.0 |
| Shear term, `U̇ₓ/g` | **0.0 exactly** | zero by construction — see §5 |
| Minimum airspeed, south leg | 226.9 m/s | from 235.9 — this is why F beats `w₀/V` |

Superseded, same session: the south leg first read **+0.02623**, from a run that opened on
a wave **crest** rather than a zero crossing — 6 m/s of updraft, about 1.5° of α out of
equilibrium before the first sample. Corrected to a zero-crossing start. The defect moved
the answer by 8e-5 and changed no conclusion, but it is the same shape as the too-short
vortex lead-in in §9 session 3 and is recorded rather than quietly repaired. Peak F also
drifts about 3% across six wavelengths, because with fixed controls the aircraft never
reaches a periodic steady state; the quoted figure is the run maximum.

**The result, and it is sharper than step 8 asked for.** §7 step 8 wanted "F exceeds the
measured envelope". It does — but not for both of the *same paper's two flight legs*. The
critical amplitude, 5.51 m/s, falls **between** Doyle et al.'s northern (3.0) and southern
(6.0) primary-wave amplitudes, measured on one aircraft on one day 50 km apart. So the
honest statement is not "a lee wave defeats a 747" but **"the threshold sits inside the
observed range"**, which is a much more useful thing to know and was not knowable before
the envelope and the field were in the same place.

Note the peak F exceeds the naive `w₀/V` = 0.02543: flown, it reaches 0.02623, because the
aircraft *slows* in the downdraft and F is inversely proportional to airspeed. The
encounter makes itself slightly worse, and only flying it shows that.

### Microburst penetration (session 9)

Cherokee, 50 m/s, **300 m AGL**, fixed controls, straight through the axis. Oseguera &
Bowles field at the paper's own 37 kt peak outflow. **The hazard metric is the 1 km
average F (Eq. 7), not the instantaneous value** — that is the FAA's metric and the one
this project should have been using all along.

| Quantity | Measured | Reference |
|---|---|---|
| Peak instantaneous F | +0.2326 | not the metric — see below |
| **Peak 1 km average F** | **+0.1929** | the metric that counts |
| …its shear term `U̇ₓ/g` | +0.1342 | **the lee wave's was exactly zero** |
| …its vertical term `−w/Vₐ` | +0.1456 | the two are comparable here |
| Cherokee thrust authority at 300 m | +0.0784 | **exceeded 2.5×** |
| FAA jet-transport hazard / must-alert | 0.10 / 0.13 | 1.9× hazardous — *for scale only* |
| F in real accidents | 0.2 – 0.36 | this run sits just below that band |
| Outflow strength that first beats the Cherokee | u_max = 7.73 m/s (15 kt) | far below anything called a microburst |
| Ground contact | t = 95.5 s, +383 m past the axis | entered 300 m up, never reached the far side |

**Why this one matters more than the lee wave.** The lee-wave field is purely vertical, so
`U̇ₓ` was identically zero and only half of Eq. (3) was ever exercised. A microburst has a
horizontal outflow, and here the shear term (+0.134) is the same size as the vertical one
(+0.146). The index is now tested on both its legs rather than one.

**The instantaneous/average distinction is not cosmetic.** Peak instantaneous F is +0.2326
against an averaged +0.1929 — 21% higher. The paper is blunt about why the average is the
right quantity: peaks "over small length scales... are quickly followed by negative values",
which an aircraft experiences as turbulence rather than as a trajectory loss. A 100 m spike
of F = 0.5 averages to 0.05 over a kilometre, and a test asserts exactly that.

**The Cherokee cannot survive any microburst worth the name.** F scales linearly with the
field, so its +0.0784 of authority is first exceeded at 7.73 m/s of peak outflow — well
below the 10 m/s of divergence Wilson et al. require before an outflow is even *called* a
microburst. Unlike the lee wave, where the threshold fell inside the observed range, here
it falls below the bottom of it.

### The validated baseline — do not touch these tolerances

`test_conservation.py`, `test_cr2144_modes.py`, `test_drag_polar.py`, `test_navion.py`,
`test_trim.py`. All are still-air statements; no reading of "turbulence landed" makes any
of them stale. If one moves, the derivative chain or the integrator changed.

## 5. Attributed gaps and structural impossibilities

**Attributed — understood, documented, not bugs:**

- **Phugoid and short-period offsets.** The sim's aero form is α/q/δe only; CR-2144 Table
  IX-4's `Xu, Zu, Mu, Żw, Ṁw` are deliberately excluded. A second linear model built for
  mode extraction only, restoring them, closes both to ~1% of the reference.
- **Drag polar away from its fitted point.** `CD0` and `e` were back-solved from a single
  reading. Residuals are within 0.004 near the fit, up to 0.014 below M 0.75 (parabolic
  polar misses the induced rise) and 0.006 above M 0.88 (Korn law extrapolating past its
  single anchor).
- **Vortex parameter uncertainty inherited from the source.** Parks derives α from
  accelerometers *"together with a knowledge of the aircraft's aerodynamic
  characteristics"* — so there are two layers of modelling between the raw DFDR data and
  the identified r₀/V₀. A 1° α error maps to 4.12 m/s of wind, 27% of a 50 ft/s peak.
  Treat the identified parameters as order-of-magnitude with roughly ±25% bands.

- **The lee wave carries no horizontal perturbation, so half the F-factor is missing.**
  `wind.LeeWave` is purely vertical and constant in altitude. That is divergence-free, so
  it is an admissible flow rather than a convenience — but a real lee wave also has a
  horizontal velocity perturbation, in quadrature with the vertical one, with amplitude
  ratio `m/k` (vertical to horizontal wavenumber). Building it needs a Brunt–Väisälä
  frequency and an ambient cross-mountain wind speed at 12 km, and **Doyle et al. supplies
  neither** — the paper gives wave amplitudes and a tropospheric wavelength band, not a
  stratification profile. So `U̇ₓ/g` is exactly zero here and the reported F is the
  vertical term alone.
  Two things follow, and they point opposite ways. The omitted term is **in quadrature**,
  so it peaks where the vertical term vanishes and vice versa — the *location* of peak F
  would move but the peak *magnitude* would not simply double. Against that, an
  order-of-magnitude estimate with a plausible `N` and ambient wind puts the shear term
  **larger** than the vertical one, so the true hazard is probably understated. The
  measured result is therefore a **lower bound**, and is quoted as one.
- **The wavelength is declared, not sourced.** Doyle et al.'s 20–35 km is tropospheric and
  the same paragraph warns "shorter wavelengths are apparent in the stratosphere" without
  quantifying them. 25 km is the middle of the band the paper *does* give. It does not
  move the F-factor peak at all — with no horizontal perturbation F is `−w/Vₐ`,
  independent of wavelength — but it sets the encounter duration and the pitching gust
  rate, so anything depending on those must say which value was used.

- **The 747 cannot be flown into a microburst, and no source held here changes that.**
  Its only derivative set is CR-2144 flight condition 9 — Mach 0.8 at 40,000 ft. A
  microburst is a sub-500 m phenomenon met at approach speed in a landing configuration.
  Using cruise derivatives there would be a larger extrapolation than anything else in
  this project, and it would be invisible in the output: the numbers would look
  reasonable. So the microburst work flies the **Cherokee**, whose 50 m/s cruise is a
  modest extrapolation to 300 m, and the cost of that choice is stated in the next entry.
  CR-2144 does contain other flight conditions; adding an approach set for the 747 is the
  fix, and it is a data-entry job rather than a modelling one.
- **The FAA windshear thresholds do not apply to the aircraft this project can fly there.**
  Proctor et al. state plainly that the 0.1 hazard and 0.13 must-alert figures, and the
  1 km averaging scale itself, were established for jet transports and "are yet to be
  determined" for piston aircraft. The Cherokee is piston. So those numbers are printed
  for scale and the **verdict is always `F > (T−D)/W`**, which is that paper's own
  criterion and needs nobody's certification basis. Note this is a *different* reason from
  the lee wave's, where the thresholds failed on altitude rather than aircraft class.
- **There is no ground.** No terrain, no landing gear, no ground effect, no stall. A
  microburst run therefore ends when the aircraft descends within one wingspan of the
  surface, because below that the integration is arithmetic rather than physics — left to
  itself the model bounces and climbs away, which reads as a survival and is not one.

**Structurally impossible — cannot be fixed from any source currently held:**

- **The ±g asymmetry.** Both papers attribute it to stall buffet. `aero.py` is
  `CL = CL0 + CLa·α`, exactly odd-symmetric in Δα, so an up-gust and an equal down-gust
  give equal and opposite load increments to machine precision. The only aircraft in the
  project with nonlinear data is the Cessna, which is out of scope; **CR-2144 provides no
  buffet-onset table for the 747**. Reproducing this needs a source the project does not
  have. Do not promise it.
- **Absolute agreement with the papers' g-loads.** Wingrove & Bach never identifies an
  aircraft type; Parks' two cases are DC-10s at 37–39 kft against this project's 747 at
  40 kft with roughly 0.8× the wing loading. Every load comparison is order-of-magnitude
  or clustering. Assert bands and orderings, never values.
- **Half of the Fig. 8 load band is unreachable inside the linear range.** Read as an
  *absolute* load factor, the band's −1.9 g needs about 13.8° of elevator from trim and
  drives |α| to roughly 18.5° — half again past the 12° where §7 says this model reports
  lift the sources deny. Read as an *increment* it needs 8.926° and |α| 10.31°, which is
  marginal but flyable. §8 records the decision to fly the increment; what belongs *here*
  is that the choice was not free. This is the same ceiling as the ±g asymmetry seen from
  the other side: `CL = CL0 + CLa·α` has no stall, so the only way to reach a large
  negative load is a large negative α, and there is no aerodynamic mechanism to get there
  sooner. A run flown to the absolute reading would not be a harder test of the model, it
  would be outside it, and would prove nothing.

## 6. Latent bugs — (a)–(d) fixed in session 5, (e) in session 7

All five are closed. Kept here rather than deleted because the *shape* of (a) and (b) is
the thing worth remembering: both survived three sessions and a 209-test suite because
every test in the project was still air, and still air cannot distinguish airspeed from
groundspeed.

**(a) `autopilot.py` sensed inertial airspeed.** FIXED. `autopilot` and `engage` now take
`sensors.AirData` rather than `State`, so they cannot be handed inertial velocity — there
is no `vel_body` in scope to misuse. Verified by
`test_the_autopilot_holds_airspeed_not_groundspeed`: into a 25 m/s headwind the loop now
settles airspeed on target with groundspeed 25 m/s below, where it previously did the
reverse.

**(b) `viz.derived` computed incidence from inertial velocity.** FIXED. It now uses the
recorded wind. The old `test_derived_agrees_with_the_aero_module` fed both sides the same
input and structurally could not fail; the replacement,
`test_derived_is_air_relative_and_this_test_can_fail`, computes the expectation
independently. **Verified by re-introducing the bug**: exactly that test and (a)'s went
red, and nothing else moved.

**(c) `viz.Trajectory` recorded no wind.** FIXED, by recording rather than replay.
`SimState` now carries the wind the previous step applied, and `Recorder.append` takes the
whole `SimState`, so a run is self-describing. Replay was the cheaper option but needs the
caller to reconstruct the exact model and key; recording cannot be got wrong later.
`load` defaults the two new columns to zero, so `.npz` written before they existed still
open — honestly, since those runs were all still air.

**(d) 747 `Mq` transcribed as −0.330.** FIXED to −0.339. Short-period damping error
12.6% → 11.5%; everything else moved in the fourth decimal or not at all.

**(e) `panel.AlphaGauge` was one-sided.** FIXED in session 7. Introduced by session 6's
own re-layout: `set_xlim(0, 15)`, a needle clipped to `[0, 15]`, and a `state()` comparing
**signed** degrees. At α = −16° it pegged the needle at zero and reported `linear`.

`aero.py` is `CL = CL0 + CLa·α`, exactly odd-symmetric in Δα — the same property §5 blames
for the ±g asymmetry being unreachable — so **|α| is what decides validity, not α**. The
gauge could not see half of its own invalid range.

It is (a) and (b)'s shape one more time: right in the easy case. Every test drove the gauge
positive, because level flight and a pull-up both do; nothing pushed. Found by asking what
the manoeuvring case (§7 step 5) would display, not by a test failing — a pushdown drives
α negative, so the instrument would have said `linear` throughout precisely the run whose
entire job is to report whether the model stayed in range. Two tests now pin it: the band
by magnitude, and the needle position, because fixing `state()` alone would have left the
needle still lying.

### What made (a) and (b) invisible

A still-air test suite cannot catch an air-relative/inertial confusion, because in still
air the two are the same number. Any future quantity with an air-relative and an inertial
form needs at least one test that flies through a non-zero wind field —
`test_sensors.py` exists for exactly that and nothing else.

## 7. Plan

```
1. [DONE] Rankine vortex array, cited to Parks 1985       -> verify: source's own three
                                                                    stated properties
2. [DONE] Updraft column, declared edge sharpness         -> verify: Δθ(updraft) > 1.5×Δθ(vortex)
3. [DONE] load_factor + air-relative analysis figure      -> verify: corr(n_z, α_air) > 0.999
7. [DONE] Fix latent bugs (a)-(d)                         -> verify: DONE, by re-introducing
   (was step 7; done early because 4 and 6 both                     (a) and (b) and watching
    depend on the sensing being right)                              exactly their tests go red
4. Dryden background layer                                -> verify: sample σ to rel 0.10;
   (needs MIL-F-8785C Fig. 7 σ at 40 kft DIGITISED;                AR(1) pole = exp(−V·dt/L)
    forces init_sim/batch_sim to be parameterised)
5. [DONE] Manoeuvring case: elevator pushdown to        -> verify: DONE, 30.37 deg against
   Δn = −1.9 g at zero wind, elevator bisected                     the updraft's 4.37 and the
                                                                   vortex's 2.24; ordering holds
6. Fig. 8 with ensemble error bars                        -> verify: vortex/updraft/manoeuvre
   (vmap over keys; deterministic parts see the same field)        ordering holds across the ensemble
9. [DONE] Microburst + averaged F-factor                   -> verify: DONE, 1 km average F
   (not in the original plan; the lee wave                          +0.193 against the Cherokee's
    left half of the F-factor untested)                             +0.078 of thrust, and the
                                                                    shear term is no longer zero
8. [DONE] Mountain lee wave + F-factor                     -> verify: DONE, and sharper than
                                                                    asked: peak F +0.0262 on
                                                                    Doyle's south leg exceeds
                                                                    +0.0234, +0.0129 on the
                                                                    north leg does not
```

Step 5 is done, so **step 6 now waits only on step 4** (Dryden), which is where the
ensemble spread would come from — the three deterministic points have no spread by
construction, since every member of a batch meets the same field.

### Extensibility: what the next wind model will cost

Everything so far is a **deterministic, position-only field**: `field_model(field)` wraps
`pos_ned -> wind_ned` and derives `omega_gust` from the analytic gradient. Dryden is not
that, and the gap is where the work is.

| | Deterministic field (vortex, updraft, lee wave) | Stochastic filter (Dryden, von Kármán) |
|---|---|---|
| Depends on | position only | its own previous output |
| Needs `WindState` | no — it is an empty tuple today | **yes**, one filter state per axis |
| Needs the key | no — returns it untouched | **yes**, splits it every step |
| `omega_gust` from | analytic gradient of the field | its own separate shaping filter |
| Ensemble meaning | every member meets the same field | every member is a different realisation |

Three things must change before Dryden lands, none of them large but all of them structural:

1. **`WindState` has to carry filter states**, and `init_sim`/`batch_sim` must be
   parameterised to seed them. Today they hard-code `zero_wind_state()`, so a stateful
   model cannot be initialised at all. This is the actual blocker.
2. **`omega_gust` needs its own filter.** `field_model`'s analytic-gradient trick has no
   equivalent for a stochastic field; MIL-F-8785C gives separate rate spectra, and reusing
   the translational filter would be wrong.
3. **The ensemble contract needs stating in a test.** "Deterministic components return the
   key untouched, so every member meets the same vortex" is currently true by construction
   and asserted nowhere. It stops being true for free the moment a stochastic layer is
   superposed with a deterministic one.

Adding another *deterministic* field — mountain lee wave, microburst, wake vortex from a
preceding aircraft — needs none of this. Write the field function, wrap it in
`field_model`, done. That path is genuinely extensible today.

### Extensibility: what the next encounter category will cost

Session 7 added the first encounter that is **not** a wind field, and the shape of that
change is the reusable part. There are now two ways to disturb the aircraft, and they are
siblings rather than one general mechanism:

| | `fly` | `manoeuvre` |
|---|---|---|
| Excitation | a wind field | an elevator schedule |
| Controls | fixed, by design | time-varying, necessarily |
| Wind | the field under test | zero |
| Rollout | `integrate.rollout` | its own `lax.scan` |
| Window from | north position | time |
| Analysis | **`_measure`, shared** | **`_measure`, shared** |

The last row is the load-bearing one. A category is a *mask plus a history*; everything
downstream — air-relative α, load factor, the Fig. 8 coordinate — happens in one place.
So a fourth category costs: excite the aircraft however it must be excited, build a
boolean window from the rule below, hand both to `_measure`. Nothing in `fig8_point` or
the figure needs to know which kind it is.

**The window rule, which is what makes the categories comparable at all:** the window is
the *disturbance's own extent*. The core for a vortex, the column for an updraft, the
pulse for a manoeuvre. Stated in §8 with the measurements that forced it. Without it,
categories are not commensurable and the discriminator means nothing — Δθ over a
badly-chosen window moves by a factor of three.

**Resist generalising `fly`.** A schedule that happens to be constant is a strictly larger
surface than a constant, and `fly`'s fixed controls are a *physical* statement — the
discriminator separates turbulence from manoeuvring by whether pitch correlates with
elevator, so an autopilot or a moving stick in the turbulence cases would blur exactly the
distinction being measured.

**`elevator_for_load` is more general than its name.** It is an inverse solve — "what
input produces this response?" — done as a bisection over a vmapped rollout, and it is why
the manoeuvre's deflection is *derived* rather than chosen. Any future "fly to a stated
condition" (a target roll rate, a target rate of descent) is the same three lines with a
different scalar extracted from the history. That is the difference between a model that
reaches a source's number and one that was handed it.

### Extensibility: what the next aircraft will cost

`FlightCondition` + `from_dimensional_longitudinal/lateral/controls` (session 5) are the
shared conversion path. They exist because the same algebra was hand-transcribed three
times and produced a real bug the third time — the Cessna's control derivatives recovered
at the wrong dynamic pressure, 25% high across all four.

Adding an aircraft whose source gives **dimensional** derivatives is now: state the source's
own `FlightCondition`, call the three helpers, fill in propulsion and limits, add gains.
Whose source gives **non-dimensional** derivatives (Navion, Cessna): skip the helpers
entirely.

What is still per-aircraft and unavoidable: the drag polar. `CD0` and `e` are back-solved
differently for every aircraft — from `Xw` for the 747 and Cherokee, by least squares on a
table for the Cessna — because no source states them. That is source variety, not missing
abstraction, and pushing it into a data file would hide the derivation rather than share it.

**Data-file definitions were considered and deferred.** The blocker is that the per-number
provenance comments and the back-solve logic are the most valuable part of `aircraft.py`,
and TOML expresses neither. The sequence that would work: keep factoring derivation into
tested helpers until a definition is *only* citations plus literal numbers, then the data
file is a mechanical translation. Not before.

### Extensibility: the ceiling nobody should walk into

`aero.py` is `CL = CL0 + CLa·α`, linear, with no stall — and by decision it stays that way.
Three consequences, stated here so they are not rediscovered:

- **The ±g asymmetry cannot be reproduced.** It is exactly odd-symmetric in Δα, so an
  up-gust and an equal down-gust give equal and opposite load increments to machine
  precision. §5 already calls this structurally impossible; the decision to stay linear
  makes it permanent, not merely pending.
- **The Cessna's stall tables stay unused.** `CESSNA172_TABLES` runs to CLmax 1.889 at
  19.5 deg and nothing reads it.
- **Any encounter driving α past ~10-12 deg reports lift the sources say is not there.**
  Parks-scale vortices do exactly this: the source's own note estimates α excursions of
  order 20 deg. Analysis windows must therefore stay in the linear range, and a run that
  leaves it is not evidence of anything.

If that ceiling ever needs lifting, the seam is `aero.coefficients` — swap it for a
protocol with a linear and a table implementation. That was the option not taken.

## 8. Open questions

- **Row spacing beyond two cores.** Parks identifies two significant vortices per case.
  Whether Mehta 1987 (*JGCD* 10, 27–31, DOI 10.2514/3.20176) uses a longer periodic train
  is unconfirmed — it is paywalled and was not retrieved.
- ~~**The Fig. 8 load-band convention.**~~ **DECIDED, session 7: read as an increment.**
  Still not resolvable from the paper's text — what forced the decision is that the two
  readings land on opposite sides of *this model's* validity boundary, so it stopped being
  harmless the moment a manoeuvre had to be flown to the band. Measured, 747 at CR-2144
  FC9, elevator pulse held one short period:

  | Reading | Elevator from trim | \|α\| max | Verdict |
  |---|---|---|---|
  | increment, Δn = −1.9 g | 8.926° (bisected) | 10.31° | **marginal** — flyable, amber band |
  | absolute, n_z = −1.9 g | ≈13.8° | ≈18.5° | **outside** — see §5 |

  The increment reading is flown. The absolute one is reported as out of reach, which is a
  finding about the model's ceiling rather than a dodge — §5 carries it.
- ~~**Which window is canonical for Fig. 8.**~~ **DECIDED, session 7: the window is the
  disturbance's own extent.** This is what the two existing points were already doing; it
  had never been stated as a rule, so the third point had nothing to follow. The vortex
  window is the core (±r₀), the updraft window is the column (±radius), and the manoeuvre
  window is the elevator pulse. In seconds, at the 747's 235.9 m/s, against a 6.609 s
  short period:

  | Encounter | Extent | Window | in short periods |
  |---|---|---|---|
  | vortex | first core, ±182.88 m | 1.550 s | 0.235 |
  | manoeuvre | elevator pulse, **declared** | 6.609 s | 1.000 |
  | updraft | column, ±2359 m | 20.0 s | 3.026 |

  The rule matters because Δθ is the one quantity with no natural bound: Δn and \|α\| both
  saturate 4 s into a held elevator and never move again, while Δθ keeps growing at about
  3.6°/s for as long as the elevator is held. A "manoeuvre" measured over a 12 s hold reads
  43°, and it reads that because it has stopped being a manoeuvre and become a descent.
  The window still prints in the figure's provenance footer, and now so does the rule.
- **What sets the manoeuvre's pulse length.** Decided *as* a declared parameter rather than
  resolved: one short period, `--pushdown-seconds`, in the same sense `--sharpness` is
  declared. The paper constrains the load, not how the pilot got there. It is bounded
  below by the ~4 s the load excursion needs to develop and unbounded above, so it is a
  choice; one short period puts the manoeuvre *between* the other two in duration, which
  is what makes the third cluster's separation attributable to the elevator rather than to
  timescale. Δθ is 25° at the shortest defensible hold and 30° at this one, so the choice
  moves the number without moving the conclusion.
- **Suite runtime is not currently measurable.** The same untouched tests (187 at the time,
  221 by session 5) have run in 53 s and 164 s on the same machine. Session 6 saw 126–207 s
  across runs of the same suite. Re-measure on a quiet machine before treating any timing
  as a baseline.
- **Whether the panel holds 20 fps on an interactive backend.** The headless half of this
  is now answered and is in §4: the re-layout did **not** hold 20 fps — 13.7 fps on Agg —
  and jitting `sense`/`accelerometers` took it to 26.4. What remains open is narrower.
  Agg is not TkAgg, which has a window manager and a real compositor in the loop, so the
  interactive rate has still not been re-taken since the re-layout.

## 9. Session log

### Session 9 — the microburst, and the metric session 8 got wrong

Chosen as the next field because of what it would *test*, not because it was next on a
list. The lee wave is purely vertical, so `U̇ₓ` was identically zero and **half of the
F-factor had never been exercised**. A microburst has a horizontal outflow, and in the
flown result the shear term (+0.134) comes out the same size as the vertical one (+0.146).

The field is Oseguera & Bowles 1988 (NASA TM-100632), the standard analytic microburst —
and Bowles also wrote the F-factor, so the field and the index it is measured with come
from the same group. **The 1988 scan OCRs badly**, so the equations were reconstructed and
then checked against four constants the paper states independently: peak outflow at
r/R = 1.1212 solves `exp(−x²)(2x²+1) = 1`; z_m/z* = 0.22 is `ln(12.5)/11.5 = 0.2196`;
u_max = 0.2357λR is the product of those two; and the paper's `w_max = λz*(e^(−z_h/z*) −
0.92)` is the vertical equation with `ε = z*/12.5` substituted, which is where 0.92 comes
from — `1 − 1/12.5`. Four different consequences of the same two shaping functions, so a
mis-transcription could not have satisfied all of them. That is what makes the
reconstruction trustworthy rather than merely plausible.

**Session 8 measured the wrong quantity, and this session's source says so explicitly.**
The F-factor's hazard metric is the **1 km average** (Proctor et al. Eq. 7), not the
instantaneous value: peaks "over small length scales... are quickly followed by negative
values", felt as turbulence rather than as a loss of flight path. `average_f_factor` now
implements it, and a test asserts that a 100 m spike of F = 0.5 averages to 0.05. The lee
wave was re-reported with it and **barely moved** — +0.02621 to +0.02614, because a 25 km
wave and a 1 km window is `sin(x)/x` at 99.7% — so session 8's conclusion stands. It was
still the wrong quantity, and on this session's field the gap is 21%.

**Two defects found in this session's own work, both by auditing rather than by tests:**

- The microburst's outflow was written as the paper writes it, `(λR²/2r)[1 − e^(−(r/R)²)]`,
  which is 0/0 on the axis. Guarding the radius made the **value** right and the
  **gradient** wrong — and `field_model` differentiates the field to get `omega_gust`, so
  anything flying through the core would have been handed a silently wrong rotational
  gust. Fixed by factoring the direction cosine back in, leaving a function of r² with a
  removable singularity. Caught only because the continuity test evaluates *on the axis*.
- With fixed controls the aeroplane descends, reaches 6 m, and **climbs away again**.
  There is no terrain, no gear and no ground effect in this model, so that is arithmetic,
  not a survival. Runs now stop at one wingspan.

**The result.** Cherokee at 300 m, 1 km average F = **+0.1929** against **+0.0784** of
thrust authority — exceeded 2.5×, 1.9× the FAA jet-transport hazard threshold, and just
below the 0.2–0.36 band the paper reports for real accidents. Ground contact 95.5 s in,
383 m past the axis; it never reaches the far side. Scaling linearly, the Cherokee's
authority is first beaten at 7.73 m/s of peak outflow — **below the 10 m/s of divergence
Wilson et al. require before an outflow is called a microburst at all.** Where the lee
wave's threshold fell *inside* the observed range, this one falls below the bottom of it.

**The aircraft choice is a finding, not a convenience.** The 747 could not be flown here:
its only derivative set is Mach 0.8 at 40,000 ft and a microburst is met below 500 m at
approach speed. §5 records that, and that the FAA thresholds are jet-transport-only so the
verdict is always the aircraft's own `(T−D)/W`.

**Deliberately not done:** no approach-configuration 747 (CR-2144 has other flight
conditions; adding one is data entry, and it would let this be re-flown on the aircraft
class the thresholds were written for). No ground model. The summary PDF is unchanged.

284 tests.

### Session 8 — the mountain lee wave, and what a 747 can do about it

§7 step 8, and it went the way session 3 went: **retrieve the source, then write the
model**. Two were needed and they do different jobs — Doyle et al. 2011 for the *field*,
Proctor/Hinton/Bowles 2000 for the *index*. Both are now in §3.

Doyle et al. was chosen over any textbook lee-wave treatment for one reason: **altitude**.
Its Gulfstream V flew legs at 11.3 and 13.1 km over the Sierra Nevada during T-REX, and
this project's 747 cruises at 12.192 km, between them. Everything else in `wind.py` is
DC-10-class data near the tropopause, so the whole module stays altitude-comparable rather
than mixing a low-level wave model into high-altitude work.

**The result is sharper than the step asked for.** §7 wanted "F exceeds the measured
+0.023/−0.066 envelope". It does — but not for both of the *same paper's two flight legs*,
flown by one aircraft on one day 50 km apart. The critical amplitude is **w₀ = 5.51 m/s**,
and Doyle's legs are 3.0 and 6.0. So the honest statement is not "a lee wave defeats a
747" but **the hazard threshold sits inside the observed range** — which is the more
useful claim and was not knowable until the envelope and the field were in one place.

**A defect found in this session's own work, by auditing it rather than by a test.** The
first version of `leewave.py` started the run on a wave **crest** while its own comment
claimed it opened "in undisturbed-mean air": 6 m/s of updraft, about 1.5° of α out of trim
before the first sample. That is exactly the §9 session-3 vortex lead-in defect wearing a
different hat, and a periodic field makes it easy to walk into because there *is* no
undisturbed region to lead in through — the nearest equivalent is a zero crossing, which
is what it now uses. It moved peak F by 8e-5 and changed nothing, which is precisely why
it would have survived: the number it produced was not wrong enough to look wrong.

The envelope itself was **recomputed rather than trusted**: session 2 recorded
+0.023/−0.066 with no derivation on the record, and `dynamics.thrust_authority` now
reproduces +0.0234/−0.0657 from the trim solution, with a test. In trimmed level flight
T = D, so the drag *is* the trim thrust and the envelope is just how far the throttle can
travel either way over the weight — which is why it needs a trim solve and not a drag
model.

Two things worth keeping:

- **Flying it matters.** Peak F comes out at 0.02623 against a naive `w₀/V` of 0.02543,
  because the aircraft *slows* in the downdraft and F goes as 1/Vₐ. The encounter makes
  itself slightly worse, and only integrating it shows that.
- **The FAA's 0.1 threshold is deliberately not used.** Proctor et al. §4.1 bounds the
  windshear threat to below 500 m, since higher up an aircraft has potential energy to
  trade. What transfers to 12 km is the *index* and the paper's own `F > (T−D)/W`
  criterion, not a number calibrated for approach. Quoting 0.1 here would have been the
  easy mistake and it is flagged in the code.

**The honest gap, in §5:** the field is purely vertical, so `U̇ₓ/g` is exactly zero and
only half of Eq. (3) is exercised. A real lee wave has a horizontal perturbation in
quadrature with the vertical one, needing a stratification and an ambient wind speed
Doyle et al. does not give. The omitted term peaks where the vertical one vanishes, so the
peak *location* would move rather than the peak simply doubling — but an order-of-magnitude
estimate puts it **larger** than the vertical term, so **the measured F is a lower bound
and is quoted as one**. The wavelength is declared for the same reason: the paper's
20–35 km is tropospheric and it says stratospheric wavelengths are shorter without saying
how much.

**The summary PDF gained a page**, "A second result: when the engines are not enough",
drawing the thrust-authority band to scale with both legs' F on it. Writing it turned up
a second class of defect, and this one had been there for sessions: **four of the five
page cross-references in the PDF were wrong.** "The reason is on page 7" pointed at the
panel page, "the cluster diagram discussed on page 6" pointed at the vortex page, and so
on. They were typed-in prose numbers in a *generated* document, and inserting the lee-wave
page shifted one of them from off-by-one to off-by-two.

Fixed structurally rather than by retyping: `PAGE_ORDER` names the pages, `pageno()` looks
them up, every reference is now an f-string, and `check_pagination()` aborts the build if
the emitted count and the declared order disagree. Verified by deleting a name and
watching it fail. This is the same lesson as session 7's stale test counts, one level up:
**a generated document is only as undriftable as the fraction of it that is generated**,
and prose numbers referring to the document's own structure are the most fragile kind
because nothing renders an error.

**Deliberately not done:** the lee wave is not on the Fig. 8 discriminator. It would cost
almost nothing — it is a deterministic field, so `vortex_viz.fly` takes it directly — but
Fig. 8 is a *pitch-and-load* clustering chart and the lee-wave result is an *energy* one,
so putting it there would imply a comparison the paper does not make.

270 tests.

### Session 7 — frame rate, the summary PDF, and the third Fig. 8 cluster

Two pieces of work landed *after* session 6's §9 entry was written, so the record was
self-contradictory when this session opened: §8 asked whether the panel still held 20 fps,
and the answer had already been measured. It had not. **13.7 fps on Agg**, against a 20 fps
target the code sets and no test asserts. `sense` and `accelerometers` were running eagerly
once per frame — 6.56 ms and 10.70 ms of a 73.0 ms frame. Jitting both took the frame to
37.8 ms, **26.4 fps**. Full table in §4; §10's 19.9 fps is superseded, not deleted.

The remaining cost is blitting, 30.1 ms of 37.8, and it did **not** improve — which is
what should happen when you jit something that was never the bottleneck's neighbour. It is
recorded rather than fixed: the panel is now comfortably above its target.

`scripts/summary.py` generates the plain-English summary PDF rather than it being written
by hand, so it cannot drift from the code — page 6's vortex figures call `wind.vortex_wind`
and the aircraft table reads `CRUISE`.

**The α gauge was one-sided**, and it was found by asking what the manoeuvring case would
*display* rather than by any test failing. §6(e) has the detail. It is (a) and (b)'s shape
a third time — correct in the easy case — and it would have lied during precisely the run
built next.

**Two open questions were closed before any code was written**, which is the part of this
session worth copying. Both had sat in §8 as "not resolvable from the paper's text", and
both still are; what changed is that they became decidable on *the model's own evidence*:

- The Fig. 8 load band's two readings turn out to land on **opposite sides of this model's
  validity boundary** — 10.31° of |α| as an increment, ≈18.5° as an absolute load. That
  makes the choice forced rather than arbitrary. §5 now carries the unreachability of the
  absolute reading next to the ±g asymmetry, which it shares a cause with: no stall means
  no way to reach a big negative load except a big negative α.
- The analysis window is **the disturbance's own extent** — which is what the vortex and
  updraft points were already doing, unstated. Writing it down was the whole difficulty:
  the third point had no rule to follow. The measurement that forced it is that Δn and
  |α| both **saturate 4 s into a held elevator** and never move again, while Δθ grows at
  3.6°/s for as long as it is held. A "manoeuvre" measured over 12 s reads 43° because by
  then it is a descent.

**The third cluster is in**, and `vortex_viz.fly` could not do it: it flies with fixed
controls by design, and a manoeuvre needs a time-varying elevator. `manoeuvre` is a
sibling rather than a generalisation of `fly` — a schedule that happens to be constant is
a strictly larger surface than a constant — and both now go through `_measure`, so the
three Fig. 8 coordinates are computed by one piece of code rather than two that could
drift. The elevator angle is **bisected** to reach the band, not chosen, so the sourced
quantity is the load. Result: 2.24 < 4.37 < 30.37 deg against the paper's 1.4 < 6.2 <
12.0. **The ordering holds; the values do not agree and §5 forbids claiming they should.**

Found while building it, and it would have been an invisible 0.08 g error: `fig8_point`
measures the load excursion from `enc.n_z[0]`, the run's own first sample. For a vortex
that is the trimmed value because the run has a long lead-in. Step the elevator at t=0 and
it is not — the first sample is already loaded. `manoeuvre` therefore has a lead-in at
trim for the same reason the vortex run has 40 core radii, and §4 records `n_z[0] = 0.9967`
as the check that it worked.

The figure's "manoeuvring: NOT MODELLED" annotation is gone because the slot is filled,
and the discriminator panel's x-axis is now scaled to the data rather than to the paper's
range — cropping to the reference would have hidden the 2.5× disagreement instead of
showing it.

**The summary PDF's claim about itself was too strong.** §10 said it "cannot drift from the
code". Its *computed* parts cannot; its prose and summary statistics are literals, and the
test and line counts had already drifted (256/4,300 against an actual 260/4,600). Corrected
by hand and the claim in §10 narrowed to what is actually true. A generated document is
only as undriftable as the fraction of it that is generated.

**The summary PDF now carries all three categories**, and building that page turned up a
number that had been wrong since session 6. Its timing chart said the two weather events
differ **seventeen-fold** in duration, and so did §4. **17× is Morton's ratio**, not
Hannibal's: r₀ = 137.16 m gives a 1.163 s traverse and 20/1.163 = 17.2, while every run,
figure and ledger row in this project uses Hannibal, whose r₀ = 182.88 m gives 1.550 s and
**12.9**. Corrected in both, superseded rather than deleted in §4. Nothing downstream
moves — the discriminator is an ordering claim and both ratios are the same order — but
§8's new window table states these durations to four figures, and a reader would have
found them contradicting §4's own sentence.

Writing that page also forced a sharper statement of the mechanism than §4 had. **Timing is
not "the whole discriminator" once the third category exists.** The pushdown's pulse is
6.609 s, *between* the vortex's 1.550 and the updraft's 20.0, yet it lands furthest right
of the three. So duration does not order the categories. It separates the two **weather**
ones; the manoeuvre separates because the elevator is moving and pitch follows the stick
rather than the air — which is the distinction Wingrove & Bach's chart was drawn to make,
and the reason `fly` holds its controls fixed.

**Deliberately not done:** `_trace_stack` still hard-codes the elevator
trace to ±5°, which would clip if the manoeuvre were ever made the figure's *primary*
encounter; it is not — `scripts/vortex.py` keeps the vortex primary — so this is flagged,
not fixed. Dryden (§7 step 4) is untouched and still blocked on digitising MIL-F-8785C
Fig. 7. `ManualGains` still not re-tuned. The interactive TkAgg frame rate still not
re-taken.

`test_viz.py::test_derived_agrees_with_the_aero_module` is **deleted**, after being flagged
in two session logs. Every assertion in it re-derived the implementation, so none could go
red; the replacement in `test_sensors.py` computes its expectation from the recorded wind
independently. What went with it: nothing else asserts `viz.derived`'s field wiring or its
altitude sign convention. That is a real if small loss, recorded here rather than left to
be discovered.

260 tests.

### Session 6 — the free-air flying interface
`run_live` took no wind model at all. `LiveSim.advance` called `step(sim, controls, dt,
ac)` and got the `zero_wind` default, so the Parks array and the Wingrove updraft — the
only things this project is building toward — could not be hand-flown. §10's "still air
only" note read as the session-5 sensing bug; this was a separate gap and it was still
open. It is now threaded through, with `--wind {none,hannibal,morton,updraft}` on
`fly.py`.

**The verification that mattered** was the same shape as session 5's: the guard test was
re-run with the wind model accepted but not applied, and the blown and still-air runs came
out at *identical* altitude while exactly that one test went red. An accepted-and-ignored
parameter is precisely how (a) and (b) survived three sessions.

The panel was re-laid-out as a basic T with a flight-test overlay. The 3D trace is gone;
it was the largest cell and told a pilot the least. New: VSI, slip ball, α against the
declared §7 band, load factor with a peak hold, wind, gust rate. Two decisions worth
keeping:

- **The slip ball reads lateral specific force, not β.** A ball is a pendulum. The two
  agree in steady coordinated flight and part company everywhere interesting — under held
  rudder they come out with *opposite signs*, which is what the test asserts. Wiring β into
  a ball would have been the same shape of mistake as §6(a): right in the easy case.
- **`omega_gust` is labelled SIM TRUTH.** It is a gradient across the span and chord and no
  instrument can sense it. The translational wind is not labelled, because with no sensor
  noise ground velocity minus air velocity *is* the wind and a real aircraft could compute
  it.

`dynamics.load_factor` was factored into `specific_force` returning all three components;
its two existing tests pass unchanged, which is the regression guard. `AirData` gained
`vertical_speed`; the accelerometer package is a *separate* function because a specific
force needs a mass and a set of deflections and an air-data computer has neither.

The stick now ramps, and there is pitch trim plus a trim-here key. The three `trim_rate`
values are derived from one stated rule — one second of trim is a quarter of full stick —
and a test asserts the rule, so the next person cannot quietly pick a fourth number.

`viz.py` was split: `panel.py` takes the live cockpit, `viz.py` keeps the log and the
post-flight figure. The split was landed as a **pure move** in its own commit, verified by
the collected test count not changing, so the re-layout's diff is only the re-layout.

Two things found in this session's own work, both by rendering the panel rather than by a
test: four layout defects (help text off the edge, strip labels over the overlay gauges, a
VSI drawn as a diagonal, a third of the figure empty), and a first trim test that asserted
the wrong thing — it trimmed *after* the stick had centred, where trim-here is correctly a
no-op. The test was rewritten, not the code.

**Deliberately not done:** `ManualGains` not re-tuned. The ramp makes higher authorities
available for the first time — they were geared down because a keyboard snapped to full
travel — but re-tuning is hand work verifiable only by flying, and `test_manual.py`'s
response bounds (±2°/±45°) are far too loose to pin it. No Mach, no control-position
display, no re-arm key, no uniform-wind option: considered and not chosen. `test_viz.py`'s
`test_derived_agrees_with_the_aero_module` is left alone — it is the
structurally-cannot-fail test §6(b) says was replaced, and the replacement did land in
`test_sensors.py`, but the original was never deleted. Flagged, not this work's mess.

**Unchanged, and still the Dryden blocker:** `init_sim`/`batch_sim` still hard-code
`zero_wind_state()`. Threading a wind *model* through the live loop does not touch that,
so §7's three structural changes stand exactly as written.

256 tests.

### Session 5 — the four latent bugs, and the extensibility seams
Fixed (a)-(d). (a) and (b) were one root cause, so they got one fix: `sensors.AirData`
and `sense(state, wind_ned)`, with `autopilot`/`engage`/`manual.update`/`viz.derived` now
taking the sensor set instead of `State`. That is deliberately the invasive option — with
no `vel_body` in scope there is nothing left to misuse. Which quantities are air-relative
is physics and is documented in `sensors.py`: pitot and vanes yes, IMU and rate gyros no.
Feeding `omega - omega_gust` to a rate-damping loop is the overcorrection, and there is a
test pinning against it.

(c) was fixed by recording rather than the replay §6 originally suggested: `SimState`
carries the wind the previous step applied, `Recorder` takes the whole `SimState`, and
`load` defaults the new columns so old `.npz` still open. Replay was cheaper but needs the
caller to reconstruct the model and key correctly every time.

(d) was one character. Short-period ζ error 12.6% → 11.5%; the affected ledger rows are
superseded in §4, not deleted.

**The verification that mattered:** re-introduced (a) and (b) and confirmed exactly their
two tests went red while the other 219 stayed green. §7 asked for "test_viz's derived test
can actually go red" and the honest answer was that it never could — it fed both sides the
same input. It is replaced rather than repaired.

Also added `FlightCondition` and the `from_dimensional_*` helpers, and retrofitted the
Cherokee and Cessna onto them. They exist because the Cessna bug was a wrong dynamic
pressure, and bundling the condition with the conversion leaves no argument to get wrong.

Deliberately not done: nonlinear aero. Asked and declined, so the ±g asymmetry is now
permanently out of reach rather than pending — recorded in §7 so it is not rediscovered.
Data-file aircraft definitions also deferred, with the condition for revisiting written
down.

221 tests.

### Session 4 — usage record
No code changed. Added §10 because nothing in this document said how to *run* any of it:
the architecture table names modules, not entry points, and the four scripts had their
usage only in their own docstrings. Recorded every script, flag, key binding and library
entry point, and which of them are trustworthy under wind (most are not — §6(a) and (b)
mean the live panel and the autopilot both mis-sense under a wind field, so `scripts/
vortex.py` and `vortex_viz.py` are the only air-relative analysis path).

Corrected a stale count in §8: 187 → 209 collected tests.

Deliberately not done: the Cessna was left exactly as it is (§5 keeps it out of scope) and
no attempt was made to reconcile the design spec, which stays a historical document per §1.

### Session 3 — vortex model, updraft, analysis figure
Retrieved Parks 1985 and used it to replace every assumed part of the vortex model with a
cited one: Rankine profile, superposed arrays, and a spacing that Parks itself checks
against Scorer. Added the updraft column with edge sharpness as a declared parameter.
Added `dynamics.load_factor`, `vortex_viz.py` and `scripts/vortex.py`.

Two defects found in the previous session's own work:
- The committed vortex test used a −6·r₀ lead-in, which starts the aircraft 0.20 g out of
  equilibrium because the 1/r far field has not died away. First-core Δθ was understated
  by 15% (1.89 vs a converged 2.20 deg). The initial load factor is now asserted.
- `n_z` in trimmed level flight is `cos θ = 0.9967`, not 1.0 — the body-normal
  accelerometer reads `g·cos θ`. The first version of the test asserted the wrong physics.

### Session 2 — turbulence design
Design pass over turbulence options and analysis methods. Established that a zero-strength
wind model is bit-identical to still air, that the rotational gust from a Wingrove-scale
vortex exceeds the 747's full aileron authority by ~1.5×, and that the mountain-wave
thrust-authority result (+0.023/−0.066 against F-factor ±0.2) is already true from
existing data with no wind model at all. Surfaced latent bugs (a)–(c).

### Session 1 — validation pass
Converted ad hoc checkpoint prints into 18 asserted tolerance-bound tests. Found no
lateral bug — the suspected Dutch-roll damping gap did not exist. Established that a
previously reported 103 s spiral was a linearisation artifact (a dropped
`r·cosφ·tanθ₀` term), confirmed by a nonlinear decay fit giving 138.05 s. Attributed the
phugoid/short-period gap to the excluded speed and α̇ derivatives.

### Template for a new entry
```
### Session N — one-line theme
What changed and why. Numbers measured, with what they were compared against.
Anything found to be wrong in earlier work, stated plainly.
What was deliberately not done.
```

---

## 10. Running it

Python 3.10.11, `.venv` in the project root. All commands are run **from the project
root**; the scripts import `flightsim` from the editable install, not from `scripts/`.

### The five entry points

| Command | What it does |
|---|---|
| `.venv/Scripts/python.exe -m pytest flightsim/tests -q` | 284 tests. The first thing to run and the only complete statement of what works. |
| `.venv/Scripts/python.exe scripts/checkpoint.py` | 747 only, no flags. Trim residuals, 60 s fixed-control hold, longitudinal modes against CR-2144 Table IX-5. |
| `.venv/Scripts/python.exe scripts/tune.py --aircraft cherokee` | Autopilot step responses for one aircraft. Exits non-zero on failure, so it is usable as a gate. |
| `.venv/Scripts/python.exe scripts/fly.py --aircraft cherokee --save runs/a.npz` | Interactive flight, basic-T cockpit plus a flight-test overlay. |
| `.venv/Scripts/python.exe scripts/fly.py --wind hannibal` | The same, hand-flown into the Parks vortex array. The panel counts the range down. |
| `.venv/Scripts/python.exe scripts/vortex.py --case hannibal --png runs/v.png` | Flies the 747 through the Parks vortex array, the Wingrove updraft, and an elevator pushdown, and draws the analysis figure with all three Fig. 8 categories. This is the turbulence path. Prints each point's Δθ, Δn and peak \|α\| with its band, then whether the ordering holds. |
| `.venv/Scripts/python.exe scripts/microburst.py --png runs/mb.png` | Flies the Cherokee through an Oseguera & Bowles microburst at 300 m and reports the 1 km average F against its thrust authority. Cuts the run at one wingspan above the ground and says so. |
| `.venv/Scripts/python.exe scripts/leewave.py --png runs/lw.png` | Flies the 747 through a Doyle et al. lee wave and compares the Bowles F-factor against the aircraft's own `(T−D)/W`. Prints both of the source's flight legs and which of them the engines can cover. |
| `.venv/Scripts/python.exe scripts/analyse.py runs/a.npz` | Replays a saved `.npz`. Accepts several files; `--png DIR` writes instead of showing. |
| `.venv/Scripts/python.exe scripts/summary.py docs/summary/flightsim-summary.pdf docs/summary/panel.png` | Rebuilds the plain-English summary PDF (14 pages). The parts that are *computed* cannot drift from the code — the vortex figures call `wind.vortex_wind`, and the aircraft table reads `CRUISE`. **The prose and the summary statistics are literals and can**: the test and line counts were stale by session 7, and four page cross-references were wrong by session 8. The page numbers are now generated from `PAGE_ORDER` with a build-time count check; the statistics are still literals. Re-run it after anything that changes those. |

Flags: `tune.py` takes `--aircraft` only. `fly.py` takes `--aircraft --autopilot --save
--dt --fps --window --seed --wind --lead-in --sharpness`. `vortex.py` takes `--case
{hannibal,morton} --aircraft --dt --lead-in --sharpness --pushdown-seconds --png`.
`--lead-in` below ~12 core radii contaminates the first core (§9 session 3);
`leewave.py` takes `--aircraft --dt --wavelength --waves --png`. `microburst.py` takes
`--aircraft --altitude --dt --u-max --radius --z-m --png`.
`--sharpness`, `--pushdown-seconds` and `--wavelength` are declared modelling parameters,
not source data
— the papers fix the updraft's magnitude and duration but not its edge, and fix the load
the pilot reached but not how long they held it. Both scripts read the case constants from
`wind.PARKS_CASES`, so they cannot disagree about a sourced number. The manoeuvre's
elevator angle is **not** a flag: it is bisected to land on the Fig. 8 load band, so the
sourced quantity is the load and the deflection is an output.

### Flying it

    arrows   centre stick: up is stick forward, so up pitches the nose DOWN
    ,  .     rudder left/right
    -  =     throttle down/up
    [  ]     pitch trim, nose down/up
    t        trim here — hold the deflections the stick is holding now
    a        toggle manual/autopilot

The stick **ramps** rather than snapping to full travel: a held key reaches the stop in
0.4 s and a released one springs back at the same rate, so a tap is a small input. That
rate is a declared figure in `panel.py`, not a measured one, and it is stepped on the
physics clock — stepping it per frame would make the feel depend on the render rate.

Releasing a surface axis returns it to `ManualState.reference`, and **trim is what moves
that reference**. Without trimming, the reference is whatever the surfaces were doing at
the last mode handover, so after a manoeuvre it is stale and the aircraft drifts. `t`
trims to what the stick is holding *right now*, so it must be pressed while the stick is
still held — once the stick has centred, the surfaces are already at the reference and
trim-here correctly does nothing. The throttle stays where it is left, because a lever
does, and trim is not sprung either.

Close the window to end the flight — the post-flight figure opens afterwards, and
`--save` writes the `.npz` first.

### What is on the panel

Basic T: airspeed and altitude tapes flanking the attitude ball, VSI beside the altitude,
heading tape below. The slip ball is at the top of the ball and reads **lateral specific
force, not β** — a ball is a pendulum, and the two quantities agree only in steady
coordinated flight. The teal marker on the ball is the body-axis **incidence** pair
(−α, +β); it is deliberately not called a flight path vector, which is earth-referenced
and would rotate with bank.

The overlay is the flight-test half: load factor with a peak hold, air-relative α against
the **declared** linear-aero band (green to ±10°, amber to ±12°, red beyond — PROJECT.md
§7, not a stall table; the band is **symmetric** because `aero.py` is odd-symmetric in α,
so a pushdown leaves the model exactly as far as an equal pull-up — see §6(e)), the
applied wind, and the gust rate labelled **SIM TRUTH** because
`omega_gust` is a span-wise gradient and no instrument can sense it. Under `--wind` the
status line carries the range to the field: a north distance and a closure rate for a
vortex array, whose cores are infinite east–west lines and therefore have no bearing, and
a range and bearing for an updraft column, which is a point.

Physics runs at a fixed 50 Hz regardless of frame rate; rendering targets 20 fps and
measures itself to hold that (matplotlib's `interval` is the gap between frames, not the
period). Current measurement is **26.4 fps headless on Agg** (§4, session 7).

Superseded, kept per §4's rule: "Measured on TkAgg: 19.9 fps, real-time ratio 0.9994, no
drift over 15 s" was taken **before** the session-6 re-layout replaced a 3D axes with a
dozen 2D ones, and is no longer a statement about this panel. The interactive rate has
not been re-taken since (§8).

### Which paths are trustworthy under wind

This is the part that is easy to get wrong, because most of the tooling predates the wind
model and silently assumes still air.

| Path | Under wind |
|---|---|
| `scripts/vortex.py`, `vortex_viz.py` | **Correct.** Air-relative throughout, by construction. |
| `integrate.step`, `dynamics`, `aero` | **Correct.** The core has always been air-relative. |
| `autopilot.py`, `manual.py` | **Correct since session 5.** Take `AirData`; the speed loop holds true airspeed. |
| `scripts/fly.py`, `panel.py` | **Correct since session 6.** `run_live` takes a `wind_model` and the panel senses from the wind the last step applied. Before that the live path could not fly through a field at all. |
| `viz.py` `Derived` and `post_flight` | **Correct since session 5.** Sensed from the recorded wind. |
| `viz.Trajectory` / saved `.npz` | Records the applied wind. Files written before session 5 load as still air. |
| `trim.py` | Still-air by construction and must stay so. Not a defect. |

Ask for state through `sensors.sense(state, wind_ned)`. Reaching into `state.vel_body` for
"airspeed" is the bug that took three sessions to find — see §6.

### Aircraft

`--aircraft` accepts `boeing747`, `cherokee`, `cessna172`. The 747 is the only one with
modes validated against a source (§4) and the only one used for turbulence work. The
Cherokee is the validated light aircraft. **The Cessna is out of scope** (§5): its rudder
set is zeroed because the source omits it, so its turns are uncoordinated. It trims, flies
and passes its tests, but no result should be quoted from it.

### Library use, without any script

```python
import flightsim                                   # enables x64 — import first
from flightsim import trim, integrate, autopilot as ap
from flightsim.aircraft import REGISTRY, CRUISE

ac = REGISTRY["boeing747"]
V, H = CRUISE["boeing747"]["airspeed"], CRUISE["boeing747"]["altitude"]
x, residual = trim.trim(jnp.array(V), jnp.array(H), ac)         # [alpha, elevator, throttle]
state    = trim.trimmed_state(x[0], jnp.array(V), jnp.array(H))
controls = trim.trimmed_controls(x[1], x[2])
sim      = integrate.init_sim(state, jax.random.PRNGKey(0))
```

- Open loop: `integrate.rollout(sim, controls, dt, ac, n_steps, wind_model=...)`.
- Closed loop: `ap.engage(...)` then `ap.closed_loop_rollout(...)` — autopilot inside `lax.scan`.
- Ensembles: `integrate.batch_sim(state, keys)` then `jax.vmap` the rollout. Deterministic
  wind components return the key untouched, so every member meets the same field.
- Wind: build with `wind.VortexArray` / `wind.UpdraftColumn`, wrap with
  `wind.field_model(...)`, combine with `wind.superpose(...)`, pass as `wind_model=`.
- `trim.minimum_drag_speed(ac, altitude)` bounds the autopilot: below V_md the
  throttle-to-speed, elevator-to-altitude pairing inverts and no gain set repairs it. The
  Cherokee cruises 2.7 m/s above it, the 747 2.5 m/s.

### Environment notes

The `.venv` lives in the project root and only there — running from a git worktree needs
`PYTHONPATH` set to the worktree, or the editable install resolves to the main checkout.
Live flight needs an interactive matplotlib backend (TkAgg is the default here); the test
suite forces Agg and drives the animation, blitting and key events for real.
