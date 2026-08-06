# JAX Flight Simulator — project record

A 6-DOF fixed-wing flight-dynamics core in JAX, built as a foundation for turbulence
modelling. This document is the standing record: what exists, what is validated, what is
known-broken, and what happens next.

**Last updated:** session 5 (latent bugs (a)-(d) fixed; extensibility seams).

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
| `dynamics.py` | 6-DOF Newton-Euler, `load_factor` | wind enters here and nowhere else |
| `wind.py` | wind fields and composition | vortex array, updraft column, `superpose`, `field_model` |
| `integrate.py` | RK4 `step`, `rollout`, batched rollout | wind sampled once per step, held across the four stages |
| `aircraft.py` | three aircraft + `REGISTRY`/`CRUISE` | every derivative cites its source table; `FlightCondition` + `from_dimensional_*` do the conversions |
| `sensors.py` | `AirData`, `sense(state, wind_ned)` | **the only supported way to ask what the aircraft is doing**; air-relative where a real sensor is |
| `trim.py` | Newton solve for steady level flight | still-air by construction, and must stay so |
| `autopilot.py` | cascaded PID | per-aircraft gains; bumpless engage |
| `manual.py` | manual control and mode switching | |
| `viz.py` | live panel, `Trajectory`, `post_flight` | air-relative since session 5; `Trajectory` records the applied wind |
| `vortex_viz.py` | turbulence encounter analysis and figure | air-relative throughout; deliberately separate from `viz.py` |

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

### Vortex and updraft encounters (747 at CR-2144 FC9)

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

### The Fig. 8 mechanism

747 short period is 6.6 s undamped. A 1.5 s vortex traverse is ~0.2 of that (impulsive);
a 20 s updraft is ~3 (quasi-steady). **That 17× separation in non-dimensional encounter
duration is the whole discriminator**, and it is a rigid-body timescale effect requiring
no nonlinear aerodynamics — which is why this model reproduces the clustering while it
can never reproduce the ±g asymmetry.

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

## 6. Latent bugs — all four fixed in session 5

All four are closed. Kept here rather than deleted because the *shape* of (a) and (b) is
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
5. Manoeuvring case: elevator pushdown to Δn ≈ −1.9 g     -> verify: third Fig. 8 cluster
   at zero wind                                                    separates from the other two
6. Fig. 8 with ensemble error bars                        -> verify: vortex/updraft/manoeuvre
   (vmap over keys; deterministic parts see the same field)        ordering holds across the ensemble
8. Mountain lee wave + F-factor                           -> verify: F exceeds the measured
                                                                    +0.023/−0.066 thrust envelope
```

Steps 4 and 5 are independent and either may go first. Step 6 needs both.

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
- **The Fig. 8 load-band convention.** Whether the paper's "−1.7 to −2.0 g" is absolute
  load factor or an increment is not resolvable from the text. The model's in-core
  excursion of −1.23 g sits between the two readings.
- **Which window is canonical for Fig. 8.** First core (2.20 deg), second core (4.17) and
  whole array (6.75) give three different answers, and only the first core separates
  cleanly from the updraft. It is currently an explicit argument, printed in the figure's
  provenance footer.
- **Suite runtime is not currently measurable.** The same untouched tests (187 at the time,
  209 now) have run in 53 s and 164 s on the same machine. Re-measure on a quiet machine
  before treating any timing as a baseline.

## 9. Session log

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
| `.venv/Scripts/python.exe -m pytest flightsim/tests -q` | 209 tests. The first thing to run and the only complete statement of what works. |
| `.venv/Scripts/python.exe scripts/checkpoint.py` | 747 only, no flags. Trim residuals, 60 s fixed-control hold, longitudinal modes against CR-2144 Table IX-5. |
| `.venv/Scripts/python.exe scripts/tune.py --aircraft cherokee` | Autopilot step responses for one aircraft. Exits non-zero on failure, so it is usable as a gate. |
| `.venv/Scripts/python.exe scripts/fly.py --aircraft cherokee --save runs/a.npz` | Interactive flight. **Still air only — see the warning below.** |
| `.venv/Scripts/python.exe scripts/vortex.py --case hannibal --png runs/v.png` | Flies the 747 through the Parks vortex array and the Wingrove updraft, draws the analysis figure. This is the turbulence path. |
| `.venv/Scripts/python.exe scripts/analyse.py runs/a.npz` | Replays a saved `.npz`. Accepts several files; `--png DIR` writes instead of showing. |

Flags: `tune.py` takes `--aircraft` only. `fly.py` takes `--aircraft --autopilot --save
--dt --fps --window --seed`. `vortex.py` takes `--case {hannibal,morton} --aircraft --dt
--lead-in --sharpness --png`. `--lead-in` below ~12 core radii contaminates the first core
(§9 session 3); `--sharpness` is a declared modelling parameter, not source data.

### Flying it

Arrows are a spring-centred centre stick, so **up is stick forward and pitches the nose
down**. `,` and `.` are rudder, `-` and `=` throttle, `a` toggles the autopilot. Releasing
a surface axis returns it to the deflection held at the last mode handover, not to zero;
the throttle stays where it is left, because a lever does. Close the window to end the
flight — the post-flight figure opens afterwards, and `--save` writes the `.npz` first.

Physics runs at a fixed 50 Hz regardless of frame rate; rendering targets 20 fps and
measures itself to hold that (matplotlib's `interval` is the gap between frames, not the
period). Measured on TkAgg: 19.9 fps, real-time ratio 0.9994, no drift over 15 s.

### Which paths are trustworthy under wind

This is the part that is easy to get wrong, because most of the tooling predates the wind
model and silently assumes still air.

| Path | Under wind |
|---|---|
| `scripts/vortex.py`, `vortex_viz.py` | **Correct.** Air-relative throughout, by construction. |
| `integrate.step`, `dynamics`, `aero` | **Correct.** The core has always been air-relative. |
| `autopilot.py`, `manual.py` | **Correct since session 5.** Take `AirData`; the speed loop holds true airspeed. |
| `viz.py` live panel and `Derived` | **Correct since session 5.** Sensed from the recorded wind. |
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
