# JAX Flight Simulator — project record

A 6-DOF fixed-wing flight-dynamics core in JAX, built as a foundation for turbulence
modelling. This document is the standing record: what exists, what is validated, what is
known-broken, and what happens next.

**Last updated:** session 11 (verifying the solver, and correcting what §5 claimed).

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

Three rules carried from `CLAUDE.md` and enforced throughout the code:

- **Check that the tree you are testing is the tree you edited.** Every worktree shares the
  main checkout's `.venv`, whose editable install maps `flightsim` to the **main checkout**
  for the life of the install. Nothing warns you when that mapping wins — the tests import,
  collect and pass, against code you did not change, and the result reads exactly like a
  real one. Print `flightsim.__file__` before believing any run. §10 carries the one-line
  check and the cases it catches.
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
| `verification.py` | **tier 0** — `fitted_order`, `oscillator_refinement`, `fixed_control_refinement`, `newton_residual_history`, `torque_free_omega`, `without_aerodynamics`, `free_fall_through_a_swinging_wind` | takes **no aircraft data as a reference**; a failure here is a defect in the core. Every check lives here rather than inside its test, so the notebook runs the same code the suite asserts on |
| `validation.py` | **tiers 1–2** — `longitudinal_matrix`, `to_stability_axes`, `to_imperial_matrix`, `longitudinal_modes`, `lateral_modes`, `Reference`/`REFERENCES`, `CAUGHEY_A`, `sweep`, `affine_fit` | the linearisation lives here, not in `tests/modes.py`, which is now a re-export. Every reference number carries its citation as a `Reference.source` field, enforced by a test |
| **`docs/ASSUMPTIONS.md`** | not code — the **assumption register**: what the model assumes, why, and a measured bound on each | this document records what has been *measured*; that one records what has been *assumed*. Read it before quoting any result to better than ~0.5%, before flying far from a trim point, and before adding a wind field whose scale approaches a wingspan |
| **`provenance.py`** | the **ledger**: a constant's category and citation, as data — SOURCED / DERIVED / CALIBRATED / DECLARED | `test_provenance.py` enforces the entries' internal consistency; coverage is enforced separately and only over five modules' module-level constants — see §2's point 4, which corrects what this row used to claim. Answers "which numbers are bulletproof?" as a query rather than a memory |
| **`airframe.py`** | where on the airframe the field is sampled: derived tail arm, sample stations, spanwise loading | the tail arm is DERIVED from `Cmq`/`CLq`, never sourced; the loading shape is DECLARED and carries a measured sensitivity |
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
| **`checks.py`** | **tier 3 — RUN checks**: `quaternion_norm`, `field_divergence`, `energy_closure`, `energy_residual_profile`, `trimmed_start`, `alpha_band`, `lateral_symmetry`, `recorded_wind_matches_field`, `run_checks` | `verification`/`validation` ask whether the MODEL is right, once, in the suite. This asks whether ONE RUN is sensible, every time one is flown. Each check carries a `kind`: **gate** (can and does fail), **tripwire** (has never fired — renders as a number and the word, never a green tick), **report** (a number with no honest threshold). Every check has a **negative control** in `test_checks.py` |
| **`analysis/`** | `artifact.py` (run artifacts: Parquet + `meta.json` + `checks.json`, and `rebuild_field`), `series.py` (every plotted channel), `figures.py` (pure Plotly figures) | needs the **`ui` extra**. Imports `flightsim`, never the reverse. Nothing in `flightsim/` proper imports it, so the simulator and every script keep working without it |
| **`apps/`** | `sweep.py` — the Dash analysis UI | **the only package that imports Dash, and it computes nothing.** It never runs the simulator either: `n_steps` is a `static_argname`, so every distinct dt pays a fresh 0.6–0.9 s compile and a panel whose contents depend on machine warmth is not a check |
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

**These three signs are independently verified** against Stengel, *Flight Dynamics* 2nd ed.,
eqs. 3.4-48, 3.4-50 and 3.4-52, by re-deriving them from `v_rel = v_cg + ω×r − w_g(r)` rather
than transcribing them. **Two equations in that source are wrong** — eq. 3.4-49's sign, and
eq. 3.4-55 by a factor of −2 — and
`docs/superpowers/specs/2026-08-14-wind-shear-fidelity-design.md` §2 records which, with the
rigid-rotation self-consistency test that found them. **Read it before changing any sign here.**

3. **Gust gradients may be sampled rather than differentiated.** `wind.gust_rates` takes the
   analytic Jacobian at the CG; `wind.sampled_rates` fits the slope across the airframe using
   `airframe.stations`. The two agree *exactly* for any field that is linear across the
   aircraft, which is asserted — so this is a better estimator of the same quantity, not a new
   one. `wind.strip_roll_moment` goes further and integrates the field per strip, which is the
   only form that carries curvature.

   **`field_model` remains the default and still uses the tangent.** A test asserts the two
   disagree at the vortex core edge, because that is what proves the default has not been
   switched over — §4's frozen baselines sit downstream of it.

4. **Constants carry their provenance, and the ledger's reach is bounded.**
   `flightsim/provenance.py` classifies each entry as SOURCED, DERIVED, CALIBRATED or
   DECLARED, and `test_provenance.py` enforces that DERIVED chains name inputs that exist,
   are acyclic, and bottom out in something sourced. **This document used to add "a
   constant added without a ledger entry fails the build", and that was not true**: every
   test in `test_provenance.py` iterates the ledger against itself, so nothing checked
   coverage. The audit measured it — roughly 346 non-trivial numeric literals across nine
   physics modules against 13 entries, about **2%**. The missing direction now exists as
   `test_audit_regression.py::test_the_provenance_ledger_does_not_cover_the_source_modules`,
   and the true statement is narrower: **a NEW module-level constant in `aero`, `airframe`,
   `atmosphere`, `trim` or `wind`, added without a ledger entry, fails the build.**
   Constants inside functions, the other four physics modules, and the aircraft data in
   `aircraft.py` are not covered. The recorded baseline may only ever shrink — a second
   test fails if a name is ledgered and left in it.

## 3. Sources

| Source | Supplied | Known gap |
|---|---|---|
| NASA CR-2144 (Heffley & Jewell 1972), §IX | 747 geometry, inertia, dimensional derivatives, transfer-function factors, drag figure; **and Table IX-2, a complete non-dimensional POWER-APPROACH set** | no non-dimensional *cruise* set; **no buffet-onset data at all**; Figure IX-1 and Table IX-3 disagree on the approach inertias by up to 6% (§4) |
| McCormick (via a worked example) | Cherokee PA-28-180 dimensional derivatives | no second source for the lateral set; `Izz < Iyy` flagged by its own author |
| Roskam / USAF DATCOM via PyFME | Cessna 172 non-dimensional tables | rudder derivatives omitted and inconsistent — the whole rudder set is zeroed |
| Nelson / Etkin / McRuer | Navion per-radian derivatives | no extractable published mode table was found; tests assert ranges, not values |
| **Parks, Wingrove, Bach & Mehta 1985**, J. Aircraft 22(2) 124–129 | **the vortex model** — Rankine core, array by superposition, and identified parameters | α is *inferred* from accelerometers through an assumed aero model — see §5 |
| **Wingrove & Bach 1994**, J. Aircraft 31(4) 753–760 | updraft magnitudes/duration, g-load statistics, the Fig. 8 discriminator | never identifies an aircraft type; no updraft edge gradient; no lateral data |
| **Doyle, Jiang, Smith & Grubišić 2011**, *Mon. Wea. Rev.* 139, 3–23, DOI 10.1175/2010MWR3466.1 | **the lee-wave amplitudes** — T-REX Gulfstream V over the Sierra Nevada, IOP 4 primary wave, 6 and 12 m/s crest-to-trough | gives a **tropospheric** wavelength band (20–35 km) and says stratospheric ones are shorter **without a number** — see §5 |
| **Proctor, Hinton & Bowles 2000**, 9th Conf. Aviation Range & Aerospace Meteorology, paper 7.7, 482–487 | **the F-factor** — Eq. (3) `F = U̇ₓ/g − w/Vₐ`, Eq. (4) for the shear term, Eq. (7) for the **1 km average**, the `F > (T−D)/W` thrust criterion, the 0.1/0.13 thresholds, and F = 0.2–0.36 in real accidents | its thresholds are **low-altitude** (§4.1 bounds the threat below 500 m) **and jet-transport only** — it states the scale and threshold "are yet to be determined" for piston aircraft |
| **Oseguera & Bowles 1988**, NASA TM-100632 | **the microburst** — Eqs. (5)–(6), an axisymmetric stagnation flow satisfying continuity, with four stated constants (r/R = 1.1212, z_m/z* = 0.22, z*/ε = 12.5, u_max = 0.2357λR) | the example's `R` is legible only in a scanned figure, so the downdraft radius is declared inside the 1–4 km band Wilson et al. use to define a microburst |
| MIL-F-8785C | (not yet used) Dryden spectra | σ above 2000 ft is a **chart read**, not a formula — must be digitised |
| **Caughey, *Introduction to Aircraft Stability and Control*, Cornell MAE 5070 notes, Ch. 5** | an **independent implementation** of CR-2144's 747 power-approach case: dimensional derivatives Eq. (5.51), plant matrix Eq. (5.52), characteristic polynomial (5.53), roots (5.54) | **not an independent dataset** — its Eq. (5.48)–(5.50) cite CR-2144, the same document §IX comes from. Same inputs, different code. Also states V = 279.1 ft/s (M 0.25 at sea level) where Table IX-2's header says 165 KTAS = 278.49 ft/s, a 0.2% difference |

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

### Distributed-airframe sampling (session 13)

| Check | Measured | Tolerance |
|---|---|---|
| Uniform field, sampled rates | **exactly 0** | `np.array_equal` |
| Linear field, sampled vs analytic gradient | agrees | rtol 1e-9 |
| Inside the Parks core, secant vs tangent | agrees | rel 1e-9 |
| At the core edge, one-sided derivatives | differ by **2·`V₀/r₀`**, opposite signs | rel 1e-6 |
| Curvature correction, 1.25 r₀ / 2.0 r₀ | **0.109 / 0.025** `V₀/r₀` | reported |
| Station-count convergence, 9 → 18 | below 0.1% | 1e-3 |
| Derived tail arm, 747 | **4.0241 c̄ = 109.90 ft** | inside the real 100–110 ft |
| Plausibility gate | passes both 747 sets, rejects both light-aircraft sets | exact |
| Rigid roll rate through the strip integral vs CR-2144 `Clp` | agrees | rel 1e-3 |
| Rectangular-wing strip integral vs Stengel eq. 3.4-40 | agrees | rel 1e-6 |
| Elliptic closed form `∫y²c dy = c₀b³π/64` | agrees | rel 1e-5 |
| Linear gust gradient, strip vs equivalent rate | agrees | rel 1e-6 |
| Loading-shape spread, elliptic vs tapered | **2.6%** | reported |
| …including uniform as a bracket | **49.7%** | reported |
| Existing wind path, before and after | **reproducible, and still the tangent** | `np.array_equal` |

Suite: **358 passed, 1 skipped**, up from 342 with nothing broken.

### Strip loads in the 6-DOF (session 14)

| Check | Measured | Tolerance |
|---|---|---|
| Zero increment vs omitting it, `derivatives` | **bit-identical** | `np.array_equal` |
| Omitting `load_model`, 100-step rollout | **bit-identical** | `np.array_equal` |
| Strip increment vs `wind.strip_roll_moment` | agrees | rel 1e-9 |
| Strip increment under a 50 m/s tailwind | uses air-relative speed, not ground speed | strict inequality |
| Tail-arm gate at construction | raises for both light aircraft | `pytest.raises` |
| Parks core traverse, point vs strip position | **0.000000 m** (`Cl` = 1.8e-19) | reported |
| Cubic spanwise gust, point vs strip position | **0.187463 m** (`Cl` = 1.335e-05) | must be > 0 |
| Fig. 8 vortex point, point vs strip | **unchanged: d(θ) 2.240°, d(n) −1.235 g** | reported |
| Fig. 8 updraft point, point vs strip | **unchanged: d(θ) 4.366°, d(n) −0.114 g** | reported |
| Ordering vortex < updraft < manoeuvre | **HOLDS on both paths** | exact |
| Rigid-rotation structure, inside the Parks core | **q-pair +1.0000**, p-pair n/a | reported |
| …outside the core | **q-pair −1.0000**, p-pair n/a | reported |
| `CL` increment through `load_factor` | raises it | strict inequality |
| `Cl` increment through `load_factor` | **exactly no effect** — it enters the moment, not the force | exact |
| Zero increment vs omitting it, `specific_force` | **bit-identical** | `np.array_equal` |
| Load model reaching the measured `n_z` in `_measure` | changes it | `not allclose` |
| Omitting the load model in `fly` | **bit-identical** `n_z` | `np.array_equal` |

Suite: **377 passed, 1 skipped**, up from 358 with nothing broken.

> **Precision note added by the remediation pass.** The Δθ figures above are quoted to
> four significant figures and are only good to three. They are unchanged as
> measurements — nothing moved them — but the once-per-step wind hold costs the in-core
> Δθ **0.82%** at the published dt (§4, "The ORDER half of the wind seam", and
> `ASSUMPTIONS.md` E4, whose bound this pass corrected by ~80×). Read them as 2.24° and
> 4.37°. The comparison this table is making — point path against strip path — is
> unaffected, because both paths carry the same hold.

**Read the two zeros together, because they have one cause.** The strip path
changes neither turbulence encounter, and that is a property of the two fields
rather than a defect in the seam. The Parks vortex has no east variation and
the updraft column is axisymmetric about an axis the aircraft flies straight
through, so in both cases every strip on the span sees the same vertical gust
and the antisymmetric roll integral cancels. The diagnostic reports the same
fact from the other side: `p-pair n/a` means `∂w/∂y` is identically zero.

**What the `q-pair` measures, stated carefully.** `gust_rates` reads three
entries out of the 3×3 body-frame velocity-gradient tensor: `p_g = +∂w/∂y`,
`q_g = −∂w/∂x`, `r_g = +∂v/∂x`. A rigid rotation of the air mass has a
*skew-symmetric* gradient tensor, which has exactly three free parameters — so
when the tensor is skew, those three numbers capture the field's entire
first-order structure with nothing left over. The q-pair reports
`−(∂u/∂z)/(∂w/∂x)`, which is `+1` iff the (x,z) block is skew.

Inside the core it is **+1.0000**: Rankine solid-body rotation, tensor skew,
three numbers sufficient. Outside it is **−1.0000**: irrotational, in Parks'
own terms — the tensor is *symmetric*, pure **strain**, zero vorticity, and the
model has no channel for strain at all.

The implication that holds is one-directional: q-pair `= +1` ⇒ rigid rotation
⇒ the field is linear in position ⇒ the point treatment is exact. That is
independently why the curvature correction is exactly `0.0000` at 0.50 r₀ and
0.99 r₀ in the session-13 table. **The converse does not hold**, and a strip
*pitch* integral would not address the `−1` region: a pitch integral gives each
longitudinal station the gust at its own `x` instead of fitting one slope, so
it fixes curvature in `w(x)` and never reads `∂u/∂z` at all. Two different
failures. That irrotational and curved coincide outside the core is a property
of the Rankine profile, not a theorem.

The 0.187 m cubic case is the positive control. It is the only evidence here
that the seam reaches the equations of motion at all; without it every number
in this table would be satisfied by a `load_model` that was computed and
discarded.

### The remediation repairs (session 16)

Seven code changes, and the measurement that shows each was surgical. Every one is a
repair to a defect the audit found; none is a change to the model's physics or its data.
**No aerodynamic derivative was added, removed or altered.**

| Check | Measured | Tolerance |
|---|---|---|
| **Whole-model regression: every mode and trim of all four aircraft, pre- vs post-remediation** | **42 of 44 scalars bit-identical**; the 2 that moved are the Cherokee's `roll_tau`/`spiral_tau`, the same two values swapped into the correct slots | `==` on the hex repr |
| 747 phugoid ωn / ζ, short-period ωn / ζ across the whole pass | **bit-identical** (0.055319, 0.055956, 0.950773, 0.342526) | `==` |
| `lateral_modes`, three stable-spiral aircraft, signed sort vs `key=abs` | **bit-identical** | `==` |
| `lateral_modes`, Cherokee (unstable spiral) | roll **0.3595 s**, spiral **−51.59 s** — was returning them swapped | rel 0.02 |
| Aerodynamic force **and** moment at exactly V = 0, all four aircraft | **exactly 0.0** — was 1.44 N (Cessna) to 170.73 N (747-approach) | `== 0.0` |
| `load_factor` in free fall, aerodynamics **live** | **−0.0** exactly, matching the aero-zeroed control — was +7.0e-6 to +4.0e-4 | `== 0.0` |
| Force/moment/coefficients at ‖v‖ ≥ 1 m/s, floor confined to the divisions | **81 of 81 sampled states bit-identical** | `==` on the hex repr |
| `is_physical`, four registry aircraft at their own cruise conditions | all **pass** (the positive control) | `bool` |
| `is_physical`, pinned 747 root at V = 471.8 m/s (throttle 567) | now **rejected** | `bool` |
| `along_track_shear` vs `dU_x/dt` differentiated along a prescribed circular track | **0 to 5.6e-17** | abs 1e-15 |
| `along_track_shear` with `accel_ned = 0` vs the straight-track expression it replaced | **bit-identical** — the reduction to Proctor Eq. (4) | `==` |
| ψ̇ along both lee-wave legs and the microburst penetration, 77,036 samples | **identically 0.0**, so the new term contributes nothing to any published run | `== 0.0` |
| `scripts/leewave.py` and `scripts/microburst.py` printed output across the change | **byte-identical** | `diff` |
| Heading-rotation term, standard-rate turn one core radius above a Parks core | **ΔF = 0.1423** (single-core closed form `v₀ψ̇/g` = 0.1383) | abs 5e-4 |
| `superpose()` with no fields | returns the zero field; superposing it is **bit-identical** to not superposing | `np.array_equal` |
| Angular-momentum drift, re-measured for the comment that was wrong by a decade | **5.695769e-13** | 1e-11 |

**The whole-model regression row is the one that matters** and it is the reason the mode
table above is untouched. It was taken by extracting the tree at the commit before the
remediation, running the same probe against both, and comparing hex representations —
not by re-reading the numbers and finding them similar.

**What was deliberately NOT repaired** is recorded in `docs/ASSUMPTIONS.md`, not here,
because a bounded flaw left in place with its size stated is an assumption rather than a
measurement: C9 (the lift-tilt energy seam), F5 (the strip quadrature at 9 stations), E6,
E8, E9, F7, C10 and C11. `AUDIT.md` §1 carries the per-finding status.

**One repair was made, measured and reverted**: `_B747_G`. See `ASSUMPTIONS.md` B5 — it
moved 19 quantities on the 747 by ≤ 4.1e-6 and broke two bit-exact guards, and the flaw
is smaller than the fix.

### 747 modes vs CR-2144

| Mode | Model | Reference | Error |
|---|---|---|---|
| Dutch roll ωn | 0.943 | 0.947 rad/s | 0.4% |
| Dutch roll ζ | 0.0361 | 0.0349 | 3.4% |
| Roll τ | 1.795 s | 1.779 s | 0.9% |
| Spiral τ | 138.0 s | 137.0 s | 0.8% |
| Phugoid ωn (as shipped) | 0.0553 | 0.0673 rad/s | 17.8% — attributed, §5 |
| Phugoid ζ (as shipped) | 0.0560 | 0.0489 | 14.4% — attributed, §5 |
| Short-period ωn (as shipped) | 0.9508 | 0.964 rad/s | 1.4% — attributed, §5 |
| Short-period ζ (as shipped) | 0.3425 | 0.387 | 11.5% — attributed, §5 |
| Phugoid / short period (augmented model) | — | — | ~1% |

**All four published longitudinal factors are now listed.** CR-2144 Table IX-5's
denominator publishes four for FC9 and this table compared two of them until the
remediation pass — and the two it omitted were the two that look worse, phugoid ζ at
+14.4% and short-period ωn at −1.4%. Reference values are the denominator block on
printed **p.231**, `Z(DET)1 = .0489`, `W(DET)1 = .0673`, `Z(DET)2 = .387`,
`W(DET)2 = .964`, read at 600 dpi and re-read independently when the rows were added.

**THE MODEL DID NOT CHANGE — the comparison did.** Every "as shipped" figure above is
the engine's own value, unchanged: the remediation pass added two ROWS, not two
derivatives. `Aircraft` still carries no speed derivative (`Xu, Zu, Mu`) and no α̇
derivative (`Zẇ, Mẇ`), and §5 still declares both families out of scope. Read the row
labels literally.

That is measured, not asserted. All four are **bit-identical** across the remediation —
see the whole-model regression row in "The remediation repairs" above, which extracted
the tree at the preceding commit and compared hex representations. The pass did change
code, in seven places; none of it is on the path that produces these four numbers.

The ≤1% figures below are an **attribution, computed in analysis**, and are not a state
this code can be run in. `AUDIT.md` §2.3 patches the engine's own cruise plant matrix
with those two families, taken from CR-2144 Table IX-4 FC9, one family at a time, and
**every one of the four closes to ≤1% when both are restored**: phugoid ωn +0.3%,
phugoid ζ +0.5%, short-period ωn −0.9%, short-period ζ +0.1%. The speed derivatives alone
fix the phugoid frequency and make short-period damping slightly worse; the α̇ derivatives
alone fix short-period damping and leave the phugoid frequency exactly unmoved; phugoid
damping needs both, which is why it is attributable to neither.

**Why that strengthens the position rather than weakening it.** The two rows added are
the two that look worse, so the table now shows a larger worst-case error than it did —
and it simultaneously shows that the whole cruise mode discrepancy is the two documented
omissions and nothing else. The aerodynamic data, the conversion chain, the trim solve
and the eigen-extraction are all exonerated by it. Restoring the two families in the
engine is a **feature with its own design and re-measurement**, not an error correction,
and deliberately did not happen here.

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
| Suite (session 10) | 296 tests + 1 skipped, 193 s | the approach 747 added 12 |

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

### The 747 power-approach set (session 10)

CR-2144 Table IX-2, sea level, 165 KTAS, α₀ = 5.7°, 20° flaps, gear up, 1.4 Vs; mass and
inertia from Table IX-3 flight condition 2. **The table is already non-dimensional**, so
this set involves no conversion chain at all — unlike the cruise set, which is recovered
from dimensional derivatives.

| Check | Measured | Source |
|---|---|---|
| Trim residual at the tabulated condition | 6.5e-20 | — |
| Trim α | 5.62° | Table IX-3 F/C 2 states 5.70° |
| W/qS at 165 KTAS sea level | 1.1123 | Table IX-2 states **CL = 1.11** |
| CD rebuilt at α₀ | 0.102 | Table IX-2, exact by construction |
| dCD/dα rebuilt at α₀ | 0.66 | Table IX-2, exact by construction |
| CD0 / e (solved, not read) | 0.0377 / 0.877 | cruise CD0 carries ±0.003 from a chart read; this does not |
| Minimum-drag speed | 97.06 m/s | **12.2 m/s above the 84.88 m/s approach speed** |

**Two source conflicts, both recorded rather than smoothed over.**

1. **Figure IX-1 and Table IX-3 disagree on the approach inertias.** The figure's Power
   Approach block gives 13.7/30.5/43.1/0.825 ×10⁶ slug-ft²; Table IX-3 column 2 gives
   14.2/32.3/45.4/0.870 — up to **6%** larger. **Table IX-3 is used**, because it is the
   table the derivatives were computed at: its Q = 92.2 psf, VTO = 165 KTAS and
   ALPHA = 5.70° all match Table IX-2's header exactly, and the cruise set already reads
   flight condition 9 from it. For *cruise* the two sources agree exactly (18.2e6,
   970056), so the disagreement is specific to the approach configuration.
2. **The weights differ in the last two digits** the same way: Figure IX-1 rounds to
   564,000 and 636,600 lb, Table IX-3 gives 564,032 and 636,636. The existing cruise set's
   636,636 was checked against this and is **correct**, not a transcription slip.

**The approach point sits below minimum-drag speed, and that is not an error.** 1.4 Vs at
max landing weight comes out 12.2 m/s below V_md — the back side of the drag curve, which
is where an airliner on final actually is. It is also the reason a windshear encounter is
lethal on approach and merely uncomfortable at cruise. The consequence is recorded: the
autopilot's throttle-to-speed / elevator-to-altitude pairing is inverted for this entry,
so it holds trim but is not to be trusted through a large speed excursion. All microburst
work flies it **open loop**, which sidesteps the question entirely.

### Microburst on the aircraft class the thresholds were written for (session 10)

The same Oseguera & Bowles field, now flown by the 747 in power-approach configuration.

| Quantity | 747 approach | Cherokee | Note |
|---|---|---|---|
| Thrust authority (T−D)/W | **+0.2094** | +0.0784 | Proctor et al. quote ~0.15 for a 4-engine jet at max **takeoff** weight; this is max **landing** weight, hence more |
| Peak 1 km average F | **+0.2835** | +0.1929 | |
| …shear term | +0.2430 | +0.1342 | scales with airspeed, as Eq. (4) says it must |
| …vertical term | +0.1453 | +0.1456 | |
| Exceeded by | **1.4×** | 2.5× | |
| FAA hazard / must-alert | **2.8× / 2.2×** | *not applicable* | **the thresholds apply to the jet and not to the piston aircraft** |
| Real-accident band 0.2–0.36 | **inside it** | below it | |
| Ground contact | 51.5 s, +441 m | 95.5 s, +383 m | neither reaches the far side |

**This is what adding the approach set bought.** The Cherokee result could only ever be
compared against the aircraft's own `(T−D)/W`, because Proctor et al. state the FAA scale
and threshold "are yet to be determined" for piston aircraft. The 747 in power-approach
configuration *is* the class Lewis et al. studied, so the same run now carries a
certification-grade verdict as well as a physical one — and it lands **inside** the band
the paper reports for real microburst accidents.

Note the jet has **2.7× the thrust authority** of the light aircraft and is still beaten,
by 1.4×. More engine does not buy immunity; it buys a smaller multiple.

### Verification — is the arithmetic right? (session 11)

Every row here takes **no aircraft data as a reference**. §4 was almost entirely
validation before this: a measured quantity against a published one for one aircraft.
Conservation drift was the only entry of the other kind, and drift measures a symmetry
rather than an order — a scheme can conserve angular momentum to 5.7e-13 and still be
second order when it claims to be fourth.

| Check | Measured | Tolerance |
|---|---|---|
| RK4 observed order, harmonic oscillator (exact solution known) | **3.99982** | 4.00 ± 0.05 |
| RK4 observed order, real 6-DOF vs fine-step reference | **3.98913** | 4.00 ± 0.05 |
| Galilean invariance, uniform horizontal wind: quaternion and rates | exact | atol 1e-11 |
| …and position differs by exactly W·t | exact | atol 1e-6 |
| **No `−m·dW/dt` body force** (session 12): zero-aero free fall through a wind swinging to 29.46 m/s, peak \|dW/dt\| = **88.39 m/s² (9.01 g)**, vs `p₀ + v₀t + ½gt²` | **3.98e-12 m**, 300 steps | atol 1e-9 |
| …and the same experiment with a `−m·dW/dt` term injected into `step` (session 13) | **13.33 m** — falsified | must fail |
| …and one step is independent of the cached previous wind, full 747 aero | **bit-identical** | equality |
| Trim Newton convergence ratio (log-residual exponent) | > 1.6, i.e. quadratic | > 1.6 |
| Torque-free asymmetric body vs Jacobi elliptic closed form, 1500 steps | agrees | atol 1e-8 |
| …the closed form itself vs Euler's equations | 8.3e-8 | atol 1e-6 |
| `rk4_step` extraction from `step` | **bit-identical**, sha256 pinned | equality |

### The ORDER half of the wind seam (session 15)

Session 12 closed E4's **body-force** half — no spurious `−m·dW/dt` term. The
**order** half was left open, and `ASSUMPTIONS.md` §E4 said plainly why: the
order-of-accuracy test flies in still air, so it cannot see the once-per-step
wind hold. `verification.fixed_control_refinement` now takes a `wind_model`, and
the answer is **not 4**.

| Check | Measured | Tolerance |
|---|---|---|
| Observed order, still air (the control, same window) | **3.9891** | 4.00 ± 0.05 |
| Observed order, C∞ field (lee wave, 1.2 km, 25 m/s) | **1.0537** | 1.00 ± 0.10 |
| …its pairwise orders | 1.073 / 1.048 / 1.042 | reported |
| **…with the hold removed** (wind re-sampled per RK4 stage) | **4.0542** | must fail |
| Observed order across a Rankine core traverse | **non-monotone** | asserted non-monotone |
| …its errors, dt 1/16 → 1/128 | 0.435 / 0.0396 / 0.0827 / 0.0806 m | > 1e-4, i.e. off the floor |

**This is a property of the scheme, not a defect.** `integrate.step` samples the
wind once per step and holds it across the four stages. Session 2 justified that
as *"the standard treatment for Dryden and von Karman"* — **an appeal to
authority with no citation behind it**, which the remediation pass replaced with
the actual argument in `integrate.py`'s module docstring: a Dryden field is a
stochastic process drawn from a key, so re-sampling it per stage makes the
realisation depend on the step size and a convergence study would then be
measuring the noise rather than the integrator. Correct for a stochastic field,
on that reasoning rather than on a source. For a field varying in **space** it is an O(h)
perturbation of the right-hand side inside the step, so the scheme is **first
order** however good the stage weights are. The falsification is what makes that
attribution rather than assertion: re-sampling the wind at each stage restores
**4.0542**.

**What it costs the project's own results, re-measured with the right instrument.**
The figure this paragraph used to give was **0.0169 m** of h-vs-h/2 position difference
inside the first core, worth **0.0024 m/s of gust out of a ~26 m/s peak**, about 1e-4
relative. **That measures the wrong quantity.** h-vs-h/2 is the DISCRETISATION error with
the hold still in place; the SCHEME error is hold-vs-per-stage at the SAME dt. On the
number `scripts/vortex.py` actually prints — the in-core Fig-8 Δθ:

| dt | wind held | wind re-sampled per stage | cost |
|---|---|---|---|
| 0.02 | 2.2596° | 2.2230° | **−1.62%** |
| **0.01 (published)** | **2.2400°** | **2.2216°** | **−0.82%** |
| 0.005 | 2.2271° | 2.2179° | −0.41% |

So the cost at the production step is **~0.8%, not ~1e-4 — about 80× the figure this
paragraph carried** — and it halves with dt, as an O(h) error must. Pinned by
`test_the_wind_hold_costs_the_headline_figure_more_than_E4_bounds_it`.

**No conclusion changes**, because §5 caps the vortex claims at orderings and puts ±25%
bands on the identified parameters, and 0.8% sits far inside both. What does change is
the precision claim: **Δθ = 2.240° is not good to four significant figures** — its last
two digits are scheme-dependent. Quote it as 2.24°.

**The Rankine row is a second, separate mechanism.** `vortex_wind` switches
branches at `r = r₀`, where §E2 records the one-sided derivatives differ by
`2·V₀/r₀` with opposite signs, so the right-hand side is C⁰ but not C¹ there.
RK4 across a kink has an error depending on where the step grid lands relative
to the crossing, so refining dt does not monotonically improve the answer. The
core test **passes with the falsification probe still in**, which is what says
the two mechanisms are independent. **Anything reporting a fitted order across a
core traverse is reporting an artefact** — a least-squares slope through that
sequence returns 0.62 and describes nothing.

**The 6-DOF error floor is round-off, and it bites earlier than expected.** The first
fitted window read **3.82** and the cause was the measurement, not the integrator. The
747 cruises at 40,000 ft, so `pos_ned` is about [944, 0, −12184] and float64 resolves it
to 2.7e-12 m. Measured pairwise orders across a wide sweep:

| dt | 1/4→1/8 | 1/8→1/16 | 1/16→1/32 | 1/32→1/64 | 1/64→1/128 | 1/128→1/256 |
|---|---|---|---|---|---|---|
| order | 3.973 | 3.993 | 4.008 | 4.167 | 3.420 | **−0.685** |

so the error bottoms out near **7e-11 m at dt = 1/128** and refining past it makes the
answer *worse*. The fitted window stops at 1/32, 203× above the floor. Anything
measuring a difference of trajectories at 40,000 ft has this ceiling.

### What constant gravity costs, and why it stays (session 12)

`ASSUMPTIONS.md` §A2 records that `G0 = 9.80665` is **+0.383% high** at the 747's cruise
altitude. Session 11 reasoned from Lanchester that this threatened every sub-0.5% claim.
This is the measurement that was made instead of the change: `flightsim.dynamics.G0`
replaced by `g(h) = g₀(R/(R+h))²`, the aircraft **re-trimmed**, and all five modes
recomputed. Tolerances are the ones each mode is actually asserted to in
`test_cr2144_modes.py`.

| Mode, 747 at cruise | g = 9.80665 | g(h) = 9.76922 | Movement | Tolerance | Consumed |
|---|---|---|---|---|---|
| phugoid ωn | 0.055319 | 0.055109 | **−0.3798%** | 5% | 7.6% |
| phugoid ζ | 0.055956 | 0.055654 | −0.5385% | 10% | 5.4% |
| short period ωn | 0.950773 | 0.950775 | **+0.0002%** | 3% | 0.01% |
| short period ζ | 0.342526 | 0.342510 | −0.0046% | 5% | 0.1% |
| Dutch roll ωn | 0.943202 | 0.942458 | −0.0788% | 2% | 3.9% |
| Dutch roll ζ | 0.036085 | 0.035931 | −0.4288% | 10% | 4.3% |
| roll τ | 1.795366 | 1.794285 | −0.0602% | 5% | 1.2% |
| spiral τ | 138.0424 | 138.1187 | +0.0552% | 2% | 2.8% |
| trim α | 4.6362° | 4.6059° | −0.6535% | — | — |

**The control is the approach 747 at sea level, where every quantity moves by exactly
0.0000%** — `g(0) = g₀` identically, so the experiment is measuring altitude and nothing
else. `aircraft.py`'s two `G0` uses are deliberately not patched: they are in the Navion and
Cessna transcription paths, where the conversion must use the g the *source* used, and
neither 747 is built through them.

**Decision: `g(h)` is not modelled.** The worst movement consumes 7.6% of its tolerance,
`G0` is imported by four modules, and every result the project quotes is at one altitude
per aircraft — so a constant g is *exactly* right per run and the bias exists only for
comparisons across altitudes, which the project does not make. Revisit if that changes.
§5 carries what the measurement corrected in the reasoning.

### Coefficient sensitivity — known change, known result (session 11)

747 power approach, re-trimmed at every sample. **Every one of these is affine with a
non-zero intercept, and the intercept is the term the textbook approximation drops.**
Three of the four laws originally planned were the wrong functional form.

| Sweep | Law that holds | Slope | Intercept | Worst residual |
|---|---|---|---|---|
| CD0 → ζ_phugoid | linear in CD0 | 0.76994 | −0.0162 | **0.31%** |
| Cmα → ωn_sp² | affine in Cmα | −0.42251 | +0.27366 | **1.68%** |
| \|Clp\| → 1/τ_roll | affine in \|Clp\| | 2.06872 | +0.30240 | **1.73%** |
| Cnβ → ωn_dr² | affine in Cnβ | 2.10545 | +0.26626 | **0.56%** |

- ζ_phugoid's slope against the textbook `1/(√2·CL)` = 0.637 is **1.21×**: the *form*
  holds tightly, the *coefficient* is 21% high.
- ωn_sp² does **not** vanish at the neutral point. The intercept is `Zα·Mq/u₀`, which
  survives there. A test asserting ωn → 0 at Cmα = 0 was written first and was wrong
  physics; the model was right.
- 1/τ_roll's intercept is Ixz roll–yaw coupling — §2 already says Ixz "is not negligible
  for the 747", so the roll root is not the pure −L_p of the two-term approximation.
- ωn_dr's intercept is the Yβ/u₀ term. ωn/√Cnβ is **not** flat: 2.385 → 1.597 over 8×.

**The neutral point is exact.** Largest real root **0.00000** at Cmα = 0, −ve inside,
**+0.0475** at Cmα = +0.1. Nothing was tuned to put it there.

### The 747 approach against an independent implementation (session 11)

Caughey (§3) works CR-2144's own power-approach data and publishes every intermediate.
Comparison requires a **stability-axis rotation**: he states Θ₀ = 0, true only there,
while the model linearises in body axes where θ₀ = α₀ = 5.57° and w₀ ≠ 0. A rotation by
α₀ is a similarity transform — asserted to move every element and no eigenvalue (1e-8).

| Element | Body axes | Stability axes | Caughey Eq. (5.52) | rel |
|---|---|---|---|---|
| A[0,0] Xu | −0.00883 | **−0.02094** | −0.02120 | 1.2% |
| A[0,1] Xw | 0.10434 | **0.04632** | 0.04660 | 0.6% |
| A[1,0] Zu | −0.17049 | **−0.22851** | −0.22290 | 2.5% |
| A[0,3] −g cos Θ₀ | −32.022 | **−32.174** | −32.174 | **0.000%** |
| A[1,3] −g sin Θ₀ | −0.952 | **0.00000** | 0.0 | **exact** |

**The elements that disagree are reconstructed, not merely attributed.** Caughey's Z row
carries a factor `1/(1 − Zẇ)` with Zẇ = −0.0341 from the CLα̇ = 6.7 this model excludes:

| Reconstruction | Result | Caughey | rel |
|---|---|---|---|
| A[1,1] / (1 − Zẇ) | 0.58374 | 0.58390 | **0.03%** |
| A[1,2] / (1 − Zẇ) | 262.48 | 262.472 | **0.003%** |
| A[2,2] (raw) vs his Eq. (5.51) Mq | −0.4381 | −0.4381 | **exact** |
| A[2,2] + (u₀+Zq)·Mẇ | −0.4906 | −0.5015 | 2.2% ‡ |

‡ Mẇ is published to one significant figure (−0.0002), which bounds this independently
of anything the model does.

| Mode | Model | Caughey Eq. (5.54) | Error |
|---|---|---|---|
| Phugoid ωn | 0.1334 | 0.13391 | **0.4%** |
| Phugoid ζ | 0.01289 | 0.01329 | 3.0% |
| Short-period ωn | 0.8961 | 0.88178 | 1.6% |
| Short-period ζ | 0.5911 | 0.62546 | 5.5% |

Trim residual 1.8e-15. **Against the cruise column's 17.8% and 11.5% for the same two
modes with the same omissions, this is 45× better from changing nothing but the flight
condition** — see §5, which this corrects.

Both Lanchester approximations reproduce the *size* of their own published error:
ωn 0.163028 vs 0.13391 is 1.217× against Caughey's stated "about 20 per cent", and
ζ 0.0651 vs 0.01329 is 4.9× against his "a factor of almost 5".

### Cross-code verification against JSBSim (session 17)

The first comparison against another **executing** 6-DOF implementation rather than a
published table. JSBSim 1.3.1 (build 1837, commit `3b25f25e`) is driven headless and its
737 is used as an **engine**, never as a dataset: `737.xml` says of itself that it was
built from public data "and guesses", validated only to the extent that it "seems to fly
right", and is for "educational and entertainment purposes only". Feeding two codes the
same coefficients makes the dataset's quality irrelevant — a disagreement is a defect in
one of the two implementations. Design:
`docs/superpowers/specs/2026-08-20-jsbsim-737-verification-design.md`.

**Derivatives are recovered by finite-differencing the running engine, not read from the
XML, and the difference is not cosmetic.** JSBSim applies aero forces at the AERORP
(x = 625 in) and takes moments about the CG (x = 610.8 in):

| | 737.xml | engine | why |
|---|---|---|---|
| Cmα | −0.600 | **−1.0637** | 77%; `CLa × 0.096 c̄`. Transcribing gives 56% of the right pitch stiffness |
| Clβ | −0.090 | −0.1440 | side force 4.925 ft above the CG — closes to 0.15% |
| Cnβ | +0.260 | +0.2730 | side force 1.183 ft aft of the CG — closes to 0.17% |
| CLα | +4.3478 | +4.3478 | **exact** — no offset effect on lift |
| Cmq | −27.0 | **−43.000** | = Cmq + Cmα̇ (−16); α̇ = q here, so they fold with coefficient one |

| Layer | Result |
|---|---|
| 1 build-up | CL **5.5e-9**, CY **1.6e-14**, Cl 7.0e-6, Cn 1.7e-6 |
| 1 CD, Cm | differ; **predicted** from 737.xml's own tables to 3.6e-4, and to **2.9e-10** at the sideslip points |
| 2 trim | α 1.980° vs 1.965°, δe −0.05311 vs −0.05192 rad, thrust **+1.04%** |
| 3 short period | ωn **0.040%**, ζ **0.019%** |
| 3 phugoid | ωn 3.33%, ζ 1.13% |
| 3 lateral | Dutch roll ζ 1.79%, roll TC 3.22%, spiral TC **0.50%** — after the yaw-damper correction below |
| 4 trajectory | elevator doublet **0.410 m/s** over 20 s against a 0.409 m/s Earth-rotation floor; rudder kick 1.52 m/s |

**No defect was found in flightsim.** Every disagreement traces to a documented model
difference with a measured magnitude.

**Three findings worth keeping.**

*JSBSim's `do_linearization` is closed-loop and nothing says so.* Its FCS is inside the
exported model. Against the bare airframe the lateral comparison reads as catastrophic —
Dutch roll ζ 0.101 against 0.344, spiral **127.6 s against 16.7 s, a 664% disagreement** —
and the natural conclusion is that this project's lateral dynamics are broken. The yaw
damper feeds yaw rate to the rudder with unit gain above M 0.11 geared by 0.35 rad, adding
`ΔCnr = Cndr × 0.35 × 2V/b = −1.147` against a bare Cnr of −0.350. Folding it in gives
ζ 0.338 and spiral 16.61 s. Pitch and roll have no feedback, which is why the short period
needed no correction at all.

*flightsim's ISA uses geometric altitude where the standard uses geopotential.* Density runs
0.159% low at 30,000 ft and 0.368% at 40,000 — a same-signed bias on every force in every
layer, since q̄ ∝ ρ. Predicted temperature errors match measured ones to four decimal places.
Neutralised for the comparison by matching on **density rather than altitude** (43.22 ft
lower, agreeing to 1e-16); the underlying defect is pre-existing and filed rather than fixed
here. **Anything altitude-dependent in this ledger carries it.**

*Scripts run from a git worktree import the wrong tree.* `flightsim` is installed editable
against the main checkout, so `python scripts/foo.py` from a worktree silently runs the other
tree's code — no error, wrong answers. pytest is immune because it puts its rootdir first,
which is precisely why a green suite did not catch it.

The thrust model gained a **Mach ram term** (`+12.1%` from M 0 to M 0.8 on the CFM56, an 11%
error at cruise that would read as a drag defect). The field defaults to neutral, so the 747,
747-approach, Cherokee, Cessna and the synthetic fixture are **bit-for-bit unchanged** —
asserted, not inspected.

Report: `docs/summary/jsbsim-737-report.pdf`.

### The validated baseline — do not touch these tolerances

`test_conservation.py`, `test_cr2144_modes.py`, `test_drag_polar.py`, `test_navion.py`,
`test_trim.py`. All are still-air statements; no reading of "turbulence landed" makes any
of them stale. If one moves, the derivative chain or the integrator changed.

## 5. Attributed gaps and structural impossibilities

> **`docs/ASSUMPTIONS.md` is the companion to this section.** §5 lists gaps found by
> comparing against sources; that file lists what the model *assumes* before any comparison
> happens, with a measured bound on each. Two of its entries are load-bearing enough to be
> repeated here.

**Attributed — understood, documented, not bugs:**

- **The Parks vortex core is 2.3–3.1 wingspans, and the gust field is sampled at a point.**
  The wind is evaluated at `pos_ned` and `field_model` derives `omega_gust` from the
  analytic gradient — a **first-order** correction for variation across the airframe. That
  is comfortable for every field in the project except the one the headline result uses:

  | Field | Scale | In 747 spans (59.64 m) |
  |---|---|---|
  | Parks Hannibal core radius | 182.9 m | **3.07** |
  | Parks Morton core radius | 137.2 m | **2.30** |
  | Wingrove updraft radius | 2359 m | 39.6 |
  | Doyle lee wave, quarter wavelength | 6250 m | 104.8 |
  | Oseguera microburst radius | 1000 m | 109.8 (Cherokee spans) |

  At 2–3 spans the linear-gradient correction is doing real work rather than tidying up,
  and second-order variation across the span is not represented at all. This compounds with
  the ±25% parameter band below and with §2's note that the rotational gust already exceeds
  the 747's full aileron authority by ~1.5×. **Vortex conclusions stay orderings** — which
  this section already required for a different reason.

- **Gravity is constant at 9.80665 m/s², which is +0.383% high at the 747's cruise
  altitude.** True `g(h) = g₀(R/(R+h))²` is 9.76922 at 12,192 m. **Session 12 measured what
  that costs and decided not to model it**; §4 carries the table and `ASSUMPTIONS.md` §A2
  the reasoning. Lanchester's `ωn_phugoid = √2·g/u₀` predicts a **1:1** mapping and the
  measurement confirms it to three figures — phugoid ωn moves −0.3798% against a −0.3816%
  change in g.

  ~~It is the same order as the tightest agreements in §4, so no claim below ~0.5% at
  altitude is safe until `g(h)` is modelled.~~ **That was session 11's reasoning and the
  measurement contradicts it.** The 1:1 mapping is the phugoid's alone: the short period is
  immune (+0.0002%) and the lateral modes — which are where §4's tightest agreements are —
  move only **0.055–0.079%**, five to fifteen times smaller than the agreements they were
  feared to threaten. Their sensitivity is indirect, through a trim α that falls 0.65%, not
  through a gravity term in the lateral equations. The corrected rule: **a sub-0.5% claim
  at altitude is unsafe for the phugoid and safe for the other four modes.** At sea level —
  the approach 747 — the error is exactly zero, and the measured movement there is exactly
  0.0000%, which is the control on the whole experiment.

- **Phugoid and short-period offsets.** ~~The sim's aero form is α/q/δe only; CR-2144
  Table IX-4's `Xu, Zu, Mu, Żw, Ṁw` are deliberately excluded.~~ **That wording was wrong
  and session 11 has the measurement to fix it.** The model **has** Xu and Zu: they fall
  out of dynamic-pressure variation, since lift and drag both go as V², and the
  stability-axis A[0,0] lands within **1.2%** of Caughey's Xu with no Xu entered anywhere
  in `aircraft.py`. `Mu` genuinely is ≈ 0, since Cm = 0 at trim and there is no Cm_M term
  — consistent with Caughey's A[2,0] = 0.0001 being his `Ṁw·Zu`, not an Mu.

  What is excluded is the **Mach content** of those derivatives (CXu, CZu, from CL_M and
  CD_M), plus **Żw and Ṁw** which are genuinely absent. That distinction is the
  explanation, because it predicts what §4 measures: **the offsets are
  condition-dependent.** Phugoid ωn is off by 17.8% at M 0.80 / 40,000 ft and **0.4%** at
  M 0.25 / sea level, same code, same omissions. Compressibility is what drives the
  missing terms, and there is none at M 0.25.

  Żw and Ṁw are now **reconstructed rather than attributed** — restoring Caughey's own
  CLα̇ = 6.7 recovers his published A[1,1] to 0.03% (§4). A second linear model built for
  mode extraction only, restoring all of them, closes both modes to ~1% of the reference.

- **CR-2144's 747 derivatives are the FLEXIBLE airframe.** Section IX's derivative plots
  are labelled "Flexible" — they carry aeroelastic corrections — and `dynamics.py`
  integrates a rigid body. This is a genuine model/data mismatch and it is larger in
  consequence than the document's 1972 date, which threatens nothing that is only ever
  compared against the document's own arithmetic (§3, and the design spec's
  "Source qualification"). Not quantified: doing so needs a rigid derivative set the
  project does not hold.

- **`trim.trim` converges to physically absurd roots for degenerate coefficients.**
  `CL = CL0 + CLa·α` is linear, so a huge α compensates a small CLa and Newton reaches a
  root that satisfies the residual to machine precision and is not a flight condition.
  Convergence and sense are different questions. Found by a sweep guard failing to fire.

  **Two numbers in the session-11 wording were wrong, corrected session 12.** The −633°
  was attributed to CLa = 0.1; it is **CLa = 1e-4**. Measured, `boeing747_approach` at
  85 m/s and sea level: CLa = 0.1 gives **−272.7°** at residual 2.3e-15, CLa = 1e-4 gives
  **−632.1°** at 5.7e-15. And "every real aircraft trims at 5–6°" was wrong in the other
  direction — the registry spans **0.01° (Cherokee) to 5.62°** at its own cruise
  conditions, which is what makes the 15° bound non-binding on legitimate data.

  **The angle itself is not reproducible, and only the phenomenon is.** The far root is
  chaotically sensitive to the start conditions: same aircraft, same CLa = 1e-4, sea level,
  **85.0 m/s gives −632.1° and 84.9 m/s gives −4232.1°**. So no specific angle is asserted
  anywhere — the test asserts converged-and-absurd, which is the stable fact. This is why
  quoting one in §5 produced two wrong numbers in the first place.

  The bound now lives in **`trim.is_physical`** rather than in `validation.sweep`, which is
  where session 11 put it. The defect is in `trim.trim` — it returns the absurd root and
  says nothing — so every other caller was equally exposed. It is not folded into `trim`
  itself because `trim` is jitted and vmapped (`minimum_drag_speed`) and therefore cannot
  raise. Nothing in the project's own results is affected; any future parameter study must
  check the angle, not just the residual.
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

- ~~**The 747 cannot be flown into a microburst.**~~ **CLOSED, session 10.** It was true
  while the only derivative set was flight condition 9, Mach 0.8 at 40,000 ft. CR-2144
  Table IX-2 turned out to hold a complete **non-dimensional power-approach set** —
  the module's own header had said so since session 1 without anyone acting on it — so
  `boeing747_approach` now exists and the microburst runs on the aircraft class the
  thresholds were written for. §4 has the numbers.
- **The FAA windshear thresholds apply to the jet and not to the light aircraft.**
  Proctor et al. state plainly that the 0.1 hazard and 0.13 must-alert figures, and the
  1 km averaging scale itself, were established for jet transports and "are yet to be
  determined" for piston aircraft. So `scripts/microburst.py` decides per aircraft:
  the 747 gets a verdict against them, the Cherokee gets them printed for scale only. In
  both cases the **physical verdict is `F > (T−D)/W`**, which is that paper's own
  criterion and needs nobody's certification basis. Note this is a *different* reason from
  the lee wave's, where the thresholds failed on altitude rather than aircraft class.
- **The approach 747 flies below its minimum-drag speed**, by 12.2 m/s, because 1.4 Vs at
  max landing weight is on the back side of the drag curve — which is where an airliner on
  final is. The autopilot's loop pairing is therefore inverted for that entry. It holds
  trim, but no gain set repairs the pairing, so it is flown open loop for all analysis.
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

Step 10 (session 11, not in the original plan): **verify the solver before adding layers.**
DONE — §4's three new subsections. Prompted by external review, which asked for validity to
be established first and for an interface where a known coefficient change gives a known
result. What it did **not** cover, deliberately:

- **No new aircraft.** CR-2144 documents **ten** — NT-33A, F-104A, F-4C, X-15, HL-10,
  **Jetstar**, **CV-880M**, B-747, **C-5A**, XB-70A — each with derivatives,
  transfer-function factors and handling-qualities parameters under the same Appendix
  A/B/C conventions this code already implements. The cheapest next two are the
  **Jetstar** (Table VII-1, power approach, non-dimensional, **body axis**) and the
  **C-5A** (Table X-1, same form, **stability axis**, so `stability_to_body` applies) —
  both in the identical form to Table IX-2, which session 10 noted "involves no conversion
  chain at all".
- **No further 747 flight conditions.** CR-2144 pp. 229–236 are scanned line-printer
  output whose text layer OCRs to noise, and the cruise non-dimensional derivatives are
  published as **plots against Mach**, not tables — which is why session 1 had to recover
  them from the dimensional set. Reading more is the same error-prone eye-work as adding a
  new aircraft.

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

1. **`init_sim`/`batch_sim` must be parameterised to seed a filter state.** Today they
   hard-code `zero_wind_state()`, so a stateful model cannot be initialised **through them**.
   This is the actual blocker, and session 12 narrowed it: it is the *seeding* that is
   missing, not the *carrying*.

   `step` threads `wind_state` opaquely and never interprets it, so a model already brings
   its own state type — `test_integrate.py`'s `FilterState` has done so since session 2, and
   session 12's `_Clock` carries a time field the same way. **So `wind.WindState` does not
   need to grow fields, and the wind-model signature does not change.** Constructing the
   `SimState` directly is the workaround until `init_sim` takes a seed; that is a two-line
   change to one function rather than a structural one. Growing `WindState` a `t` field
   *now* would also break `FilterState`, since `step` would have to `_replace` a field that
   a bring-your-own state does not have.
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
- ~~**The `−m·dW/dt` gust error is still untested.**~~ **CLOSED, session 12.** §2 names two
  classic gust-modelling mistakes: substituting `vel_rel` into the Coriolis term, and
  adding an explicit `−m·dW/dt`. Session 11's Galilean test catches the first and, as it
  said, could not catch the second. §4 now carries both new rows.

  What forced the design is worth keeping, because the obvious test is wrong. Offsetting
  the start state by `W(0)` and demanding the rates match still air — the steady test's own
  instrument — asserts **false physics**. With `ṽ_b = v_b − CᵀW(t)` the air-relative body
  velocity, `ṽ̇_b = F(ṽ_b,ω)/m + g_b − ω×ṽ_b − Cᵀ Ẇ`: the air-relative state obeys the
  still-air equation **plus** `−Cᵀ Ẇ`. That term is exactly what makes a time-varying wind
  something other than a change of inertial frame, so the two runs must diverge and an
  invariance assertion cannot be the instrument. What works is a **closed form** — zero the
  aerodynamics and the thrust, and free fall is the exact answer while the wind has no
  legitimate route into the equations at all, so any dependence on it is the spurious term
  and nothing else.

  Both tests were checked by **injecting the bug**, per §3's rule that a check which can
  only pass shows nothing. Both fail on it by orders of magnitude; the Galilean test
  *passes* with the bug still in (2.7e-15 on quaternion against its 1e-11 tolerance) once
  the wind cache is seeded consistently, so its blindness is measured and not merely
  argued. `docs/ASSUMPTIONS.md` §E4 carries the detail.
- **Whether Etkin & Reid publishes an independent CRUISE worked example.** Caughey covers
  the M 0.25 approach point, where the model's error is 0.4%. The interesting condition is
  M 0.80 / 40,000 ft, where it is 17.8%, and there the only reference is CR-2144's own
  transfer-function factors. Etkin & Reid (3rd ed., 1996) is Caughey's reference [1] and
  may carry a worked cruise case; **not verified**, needs the physical book. It would give
  a second implementation exactly where the gap is largest.
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

### Session 16 — remediation: fixing what the audit found, and not fixing the rest

Code and documents in commit `4e925c4`; the audit documents, the regression tests and
the regenerated PDF in the commit after it.

The audit produced 41 findings. This session acted on them under `FIX_PROMPT.md`, whose
standing rule is the audit's own: *a known, bounded, honestly-recorded flaw is a pass; an
unexamined assumption presented as fine is a failure.* It follows that **"fix" is not
always the right response to a finding**, and the most important thing this session did was
decide which were which.

**Ten repairs, each with its own measurement rather than "the tests pass".**

| # | Repair | Verification that it was surgical |
|---|---|---|
| 21 | `validation.lateral_modes` sorted **signed** time constants, so an unstable spiral came back as `roll_tau`. Now `sort(key=abs)`. | Cherokee returns roll 0.3595 s and spiral −51.59 s, the same two numbers in the right slots. The three stable-spiral aircraft are **bit-identical**, which is what `sort()` and `sort(key=abs)` must be when every value is positive |
| 40 | `V_MIN` floored `V` before `qbar`, so a stationary airframe made 1.4–171 N out of still air. `qbar` and Mach now use the true airspeed; the floor stays on β and the three rates. | Force and moment at V = 0 are **exactly** zero on all four aircraft; free fall reads `n_z = −0.0` with the aerodynamics live; and **81/81 sampled states at ‖v‖ ≥ 1 m/s are bit-identical**, which is the whole safety argument — `jnp.maximum(x, 1)` returns `x` exactly for `x ≥ 1` |
| 37 | `trim.is_physical` checked α and nothing else, endorsing up to 364/640 of a (V, h) sweep on throttle outside [0,1] or elevator past the stops. Now `is_physical(x, ac)`. | The pinned 747 root at 471.8 m/s — α = −0.57° on throttle 567 — is rejected; all four aircraft still pass at their own cruise conditions, which is the positive control `test_trim.py` already asserted |
| 30 | `along_track_shear` held the track direction fixed, dropping the heading-rotation term. Derived, with the reduction to Proctor Eq. (4) at ψ̇ = 0 shown in the docstring. | Reproduces the differentiated truth along a prescribed circular track to **0–5.6e-17**; ΔF = **0.1423** at a standard-rate turn, the FAA threshold entire. **ψ̇ was identically 0.0 at all 77,036 samples** of the runs on record and both scripts' printed output was byte-identical |
| 17 | `aircraft.py` cited Table IX-1's `CLδe` as .396; the table reads **.356**. | Printed p.216 re-read at 500 dpi in this session. The recovered 0.3638 agrees to **2.2%**, not the 8.1% the comment implied |
| 8 | `PROJECT.md` §4 compared 2 of the 4 published longitudinal factors. | All four now listed, from p.231 re-read at 600 dpi. The two that were missing are the two that look worse — and adding them **strengthens** the position, because §2.3's attribution closes all four to ≤1% |
| 11 | "A constant with no ledger entry fails the build" was false — nothing checked coverage. | Both documents now state what is enforced: a **new** module-level constant in one of five modules. Seven ledger entries added, `KNOWN_UNLEDGERED` shrunk to match, and a second test now fails if a ledgered name is left in it |
| 27 | `ASSUMPTIONS.md` E4 bounded the wind hold with the wrong instrument, by ~80×. | Re-measured hold-vs-per-stage at the same dt: **−1.62%** at dt 0.02, **−0.82%** at the published 0.01, **−0.41%** at 0.005. Halving with dt is what identifies it as O(h) |
| — | `integrate.py` justified the wind hold as "the standard treatment for Dryden and von Kármán". **No such source exists in this repository and none was found.** | Replaced with the actual argument — a stochastic field is drawn from a key, so per-stage sampling makes the realisation depend on dt — with no appeal to authority |
| 29 | `superpose()` with no arguments returned the integer `0`. | Returns the zero field, and superposing it is bit-identical to not superposing |

Also repaired: `test_conservation.py`'s comment was wrong by a decade (5.6958e-13, not
5.7e-14); `wind.py` had Oseguera & Bowles' four constants' dependency backwards, which the
paper's own wording settles — 0.22 is the TASS input and 12.5 was iterated *from* it; and
the Cherokee and Cessna source notes now say **`unverifiable — source not available`**
rather than citing a file that is not in the repository.

**One repair was made, measured, and reverted.** `_B747_G = 32.174` breaks `units.py`'s
no-inlined-factor rule and truncates g₀ by 1.5e-6. Going through `LB2KG / SLUG2KG` moved 19
quantities on the 747 and nothing on any other aircraft, all ≤ 4.1e-6, with **every number
§4 quotes unchanged at its quoted precision** — and broke two bit-exact
arithmetic-neutrality guards on earlier refactors. Re-pinning those would have spent the
guarantee they exist to provide on a violation that moves no result. **The flaw is smaller
than the fix.** Recorded as `ASSUMPTIONS.md` B5 and at the constant.

**Eight findings were real, measured, and deliberately left alone**, because the correct
form is not established by any source this project holds. Changing them would convert an
honest documented limitation into a fabricated certainty, which is strictly worse than the
flaw. They are now declared in the register: the **lift-tilt energy seam** (C9 — the fix is
exact to 1.9e-16 and would still be wrong to apply, because the tilt belongs to every lift
channel and the only available arm rests on an attribution this project already rejected);
the **strip quadrature at 9 stations** (F5 — 82.6%, convergence order measured at **1.50**,
and the decision to change neither `N_SPAN` nor `a₀` recorded in `provenance.py` so it is
not re-taken silently); the **microburst's unmodelled `z_h`** (E6); the **silent NaN on a
singular control Jacobian** (F7); the **one-sided longitudinal station set** (E8); the **two
tail arms that disagree by 2–2.9×** (C11 — and which of them is wrong cannot be determined,
because the split is exactly the sourcing split); the **vortex core branch tie** (E9); and
the **compound `δa`** (C10). Four more assumptions the code made and the register never
declared are now A4, C7, C8 and D2.

**Verification.** Pre-existing suite **444 passed, 1 skipped** — the baseline, unmoved.
`test_audit_regression.py` grew from 109 to 116: three fail-when-fixed tests were replaced
by the positive assertions their own messages named, and new tests pin the heading-rotation
derivation, its reduction to Eq. (4), the straightness of the shipped runs, the empty
superposition, and the widened trim gate. **No tolerance, reference value or assertion was
weakened anywhere** — the only assertions removed from the whole repository are the seven
that constituted those three tests.


### Session 15 — the order half of the wind seam, and the analysis UI

Two things, and the first came out of designing the second.

**The integrator is first order, not fourth, in a spatially varying wind field.**
Found while measuring for the UI design: the energy-closure residual converged at
order 1.00 and the explanation first written for it — trapezoidal quadrature
across the core kink — was wrong. The trajectory itself is first order. §4 has
the table; the cause is the once-per-step wind hold, which `integrate.py` has
documented as deliberate since session 2 and which `ASSUMPTIONS.md` §E4 said in
as many words the still-air order test could not see. Session 12 closed E4's
body-force half; this closes the order half.

`verification.fixed_control_refinement` now takes a `wind_model`, which is what
its own docstring always claimed it was for — it promised to catch "a wind sample
applied at the wrong RK4 stage" while having no way to pass a wind model.
**Falsified by injecting the fix**: re-sampling the wind at each stage restores
4.0542, so the 1.05 is attributable to the hold and to nothing else. Bounded and
harmless at the step sizes in use — 0.0024 m/s of gust error in the Parks core,
~1e-4 relative — so nothing §4 quotes moves.

**The analysis UI.** `checks.py`, `analysis/` and `apps/` (§2), fed by run
artifacts from `scripts/vortex.py --artifacts`. `integrate.logged_rollout` is the
one core addition: `rollout` emits `carry.state` and therefore discards the wind
a run actually flew, so no analysis script could write a self-describing run.
`vortex_viz.fly` now uses it and carries the log on the `Encounter`; a test pins
the headline pair to exact equality with the pre-change values, and it did not
move a bit.

**Three things the work found in itself, all recorded rather than repaired
quietly:**

- **A 14 r₀ lead-in starts the aircraft 0.100 g out of equilibrium**, and
  `trimmed_start` caught it in its own test fixture on first use — the same
  defect §9 session 3 records at −6 r₀. But the check as first written was also
  wrong: at `scripts/vortex.py`'s own **40 r₀ it is still 0.0384 g**, because the
  Parks far field is 1/r and never dies away. There is no trimmed start in a
  vortex, so demanding one would fail the project's canonical run and loosening
  the tolerance until it passed would be choosing a number to make a check
  succeed. It is now a **report** carrying the offset as a fraction of the run's
  own peak excursion (3.1% at 40 r₀ against 16% at −6 r₀), and a **gate** only in
  still air, where cos θ₀ is genuinely the right answer.
- **The α gate condemned the project's own published result.** Collapsing three
  declared bands into a boolean as `band == "linear"` failed the manoeuvring
  Fig. 8 point at |α| 10.31°, which is amber and which §4 reports. The gate now
  answers "does this run prove anything?" and condemns **INVALID** alone, with
  the band word carried separately for the UI to colour.
- **Two browser measurements of `uirevision` disagreed with each other**, because
  the synthetic drag fired a `plotly_relayout` and populated `_preGUI` without
  actually moving the camera — so the mechanism looked broken when the probe was.
  The app therefore does not rely on it: `figures.apply_camera` restores the
  camera explicitly from `relayoutData`, which is deterministic, unit-tested
  without a browser, and verified end-to-end.
- **The 3D panel drew a flight path through an empty field, and looked fine
  doing it.** The field grid was centred on the trajectory's midpoint, which for
  a 40 r₀ lead-in is ~2.5 km upstream of both cores — where the Parks field is
  **exactly irrotational** — so the vorticity isosurface was computed over a
  region of zero vorticity. `vortex_viz._field_panel` had already written the
  rule: *"Zoom to the structure, not the run."* The grid is now centred from the
  artifact's field spec; `isomin` reads 0.255 against 0.9·2V₀/r₀ = 0.255 and the
  isosurface peaks at the in-core 0.28333 s⁻¹. **Found by reading a number off
  the rendered page, not by a test** — which is the same way §9 session 7 found
  the one-sided α gauge.

Suite **434 passed, 1 skipped** in 339 s, up from 377. Nothing in §4's validated
baseline moved.

**Deliberately not done:** no comparison driver (Galilean, strip-vs-point and the
h/2 Richardson panel all need a second run and belong to a driver that writes its
own artifact); no multi-run sweep table; `leewave.py` and `microburst.py` do not
write artifacts yet — the schema and `rebuild_field` already cover their fields,
so it is the same wiring as `vortex.py` and nothing yet needs it.

### Session 13 — putting session 12's work where the notebook can see it

Session 12 closed E4 but left its two checks **inside the test file**, unlike every other
tier-0 check, which is a `verification.py` function driven by a thin test. The consequence
was not cosmetic. `ASSUMPTIONS.md` states the protocol — *add the computation to
`verification.py`, assert it in a test, then add a notebook cell* — and session 12 did only
the middle step, so **the notebook still told a reader the `−m·dW/dt` term could not be
detected.** The review deliverable was misstating the project's status.

Extracted `without_aerodynamics` and `free_fall_through_a_swinging_wind` into
`verification.py`, returning a `FreeFallResult` the test asserts on and the notebook prints.
The notebook is now 13 cells; its summary moves the seam from "known gaps" to "established"
and picks up the g(h) decision.

**Having the experiment report its own numbers immediately found a wrong one.** The comment
claimed `|W0|` = 30.5 m/s and peak `|dW/dt|` = 91 m/s² (9.3 g). It is
`√(18²+20²+12²)` = **29.4618 m/s** and **88.3855 m/s² (9.01 g)**. The slip had reached
§4, §9 and `ASSUMPTIONS.md` E4. All corrected, and the test now asserts both figures rather
than trusting a comment — which is the entire argument for computing a number where a test
can see it.

Two other numbers are now sharper. The free-fall agreement is **3.98e-12 m**, not merely
"under 1e-9". And the falsification is quantified: with the spurious term injected,
the same figure is **13.33 m**, ten orders of magnitude above the bound.

The experiment now runs on the **747** rather than `conftest.make_test_aircraft`, whose own
docstring says it is synthetic and "must never be used for results" — which a notebook is.
Free fall is independent of mass and airframe, so the choice cannot flatter the result.

322 tests + 1 skipped and the 13-cell notebook, both green. No test count change: the
existing test was rewritten as a driver, not duplicated.

### Session 12 — closing the two actionable assumptions, and what measuring changed

Session 11's register ended with two entries marked new and actionable: **E4**, the
untested `−m·dW/dt` gust seam, and **A2**, constant gravity. Both are now closed, and in
**both cases the measurement contradicted the reasoning that raised them**. That is the
theme worth carrying forward.

**E4 — the gust seam.** The plan proposed catching a spurious `−m·dW/dt` term by offsetting
the start state by `W(0)` and demanding the rates match still air. That asserts false
physics: the air-relative velocity obeys the still-air equation *plus* `−CᵀẆ`, so the two
runs must diverge, and no tolerance could have been chosen honestly. The instrument that
works is a **closed form** — zero the aerodynamics and the thrust and free fall is the exact
answer, while the wind has no legitimate route into the equations at all, so any dependence
on it is the spurious term. Matches `p₀ + v₀t + ½gt²` to 3.98e-12 m through a wind swinging
to 29.46 m/s at peak `|dW/dt|` = 88.39 m/s². A second test keeps the full 747 aero and
varies only the cached previous wind: bit-identical. Both were falsified by injecting the
bug; the Galilean test **passes with the bug still in**, so its documented blindness is now
measured. (Session 12 recorded 91 m/s² here from an arithmetic slip; session 13 corrected
it by having the experiment report the figure rather than a comment assert it.)

**A2 — constant gravity.** Session 11 reasoned from Lanchester that a 0.383% gravity error
threatened every agreement below 0.5%. Measured, the 1:1 mapping is the **phugoid's alone**
(−0.3798%, confirming Lanchester to three figures); the short period is immune (+0.0002%)
and the lateral modes — where §4's tightest agreements actually are — move 0.055–0.079%,
five to fifteen times smaller than feared. Their sensitivity is *indirect*, through a trim α
that falls 0.65%. Worst tolerance consumption is 7.6%, so **`g(h)` is not modelled** and the
entry closes on the bound. Sea level moves exactly 0.0000%, which is the control.

**Two wrong numbers found in §5 while writing a test.** The absurd-trim example attributed
−633° to `CLa = 0.1`; it is `CLa = 1e-4`, and 0.1 gives −272.7°. "Every real aircraft trims
at 5–6°" is wrong the other way — the registry spans 0.01° to 5.62°. Worse, the angle is
**not reproducible at all**: same aircraft and CLa, 85.0 m/s gives −632.1° and 84.9 m/s
gives −4232.1°. Nothing asserts an angle now, only converged-and-absurd.

`trim.is_physical` moved the |α| bound out of `validation.sweep`, where session 11 put it,
into the module whose function actually has the defect. **`WindState` did not need a time
field** — `step` threads the wind state opaquely, so a model brings its own type, and adding
`t` would have broken `test_integrate.py`'s `FilterState`. §7 records that Dryden therefore
costs no signature change; the blocker is `init_sim` seeding, not `WindState` carrying.

**322 tests + 1 skipped and the 12-cell notebook, both green.** Four new tests: two on the
gust seam, two on `is_physical` (one of which is the positive control, without which
`is_physical` could simply return `False` always). Nothing in the validated baseline moved.
Note the preceding commit's message says 320 — that count was taken from a background run
that predated the two trim tests in the same commit; the correct figure there is 322.

### Session 11 — verifying the solver, and correcting what §5 claimed

External review asked for the solver's validity to be established **before** more layers,
and for an interface where a known coefficient change gives a known result. The project
could not meet that: §4 was almost entirely validation rows, with next to nothing saying
the arithmetic is right independent of any aircraft's data.

Review also asked whether a 1972 source is itself a source of error. The answer is that it
is not, for what is actually being checked — if CR-2144's derivatives were 10% from the
real aeroplane, this model must **still** reproduce CR-2144's own transfer-function factors
from CR-2144's own derivatives. The real risks are different and are now recorded: the data
is the **flexible** airframe against a rigid-body model (§5, new), the scan, and the
small-perturbation range. The design spec ranks every check by how much it depends on any
source at all.

**Three things were found, and none of them was the failure being looked for.**

1. **`aero.py`'s exclusions were misdescribed, and had been since session 1.** §5 said
   `Xu, Zu, Mu, Żw, Ṁw` are excluded. The model **has** Xu and Zu — they fall out of
   dynamic-pressure variation, and A[0,0] lands within 1.2% of Caughey's Xu with no Xu
   entered anywhere. What is missing is their **Mach content**, which is why the offsets
   are **condition-dependent**: 17.8% at M 0.80 and **0.4%** at M 0.25, same code.
2. **The 6-DOF order-of-accuracy fit read 3.82, and the measurement was at fault.** At
   40,000 ft `pos_ned` carries a 12,184 m altitude that float64 resolves to 2.7e-12 m, so
   the error floors near 7e-11 m and past dt = 1/128 refining makes it *worse* — pairwise
   order **−0.685** at 1/256. Fitted in the asymptotic range it is **3.98913**.
3. **`trim.trim` converges to absurd roots.** CLa = 0.1 gives α = −633° at a residual of
   1.6e-15. A residual check detects non-convergence and cannot detect nonsense.

**Three of the four coefficient sweeps were the wrong functional form, and the model was
right each time.** Every relation turned out affine with a non-zero intercept, and the
intercept is the term the textbook approximation drops — `Zα·Mq/u₀` for the short period,
`Yβ/u₀` for Dutch roll, Ixz coupling for roll. A test asserting ωn_sp → 0 at the neutral
point was written first; that is wrong physics. What *is* exact is the neutral point
itself: largest real root **0.00000** at Cmα = 0, and nothing was tuned to put it there.

**What the tier-2 comparison bought.** Caughey's Cornell notes work CR-2144's own approach
data and publish every intermediate, so this is an independent *implementation* rather than
an independent dataset — same inputs, different code. It needed a stability-axis rotation
(he states Θ₀ = 0, true only there) which is asserted to be a similarity transform first.
After it, every element the model contains matches, and the two that do not are
**reconstructed** from his own α̇ derivatives to 0.03%. §5's attribution stops being an
attribution.

**Deliberately not done:** no new aircraft, and no further 747 flight conditions — §7 says
why, and records that CR-2144 holds nine more airframes including the Jetstar and C-5A.
The notebook is a thin front end over tested code; it holds no arithmetic, so nothing it
displays can drift from the suite.

318 tests, was 296. Nothing in the validated baseline moved.

### Session 10 — the 747 power-approach set, and what it was hiding

§5 said the 747 could not be flown into a microburst because its only derivative set was
Mach 0.8 at 40,000 ft, and that adding an approach set would be "data entry rather than
modelling". Both halves turned out to be true, and the data was closer to hand than that
implies: **`aircraft.py`'s own header had said since session 1 that CR-2144 tabulates
non-dimensional derivatives for the landing and power-approach configurations.** It was
written down as a *reason the cruise set needed converting* and never read as an
opportunity. Table IX-2 is a complete, already-non-dimensional set — no conversion chain,
no primed-to-unprimed lateral algebra, no chart read.

**Two source conflicts came out of cross-checking it, and one nearly shipped.**

- **Figure IX-1 and Table IX-3 disagree on the approach inertias by up to 6%.** The figure
  says 13.7/30.5/43.1/0.825 ×10⁶ slug-ft²; Table IX-3's flight condition 2 says
  14.2/32.3/45.4/0.870. Figure IX-1 was transcribed first and would have gone in unnoticed.
  Table IX-3 is used, because it is the table the derivatives were *computed* at — its
  Q = 92.2 psf, VTO = 165 KTAS and ALPHA = 5.70° all match Table IX-2's header exactly.
  For **cruise** the two agree exactly, so this is specific to the approach configuration.
- The weights differ in the last two digits the same way (564,000 vs 564,032; 636,600 vs
  636,636). This let the **existing cruise weight be checked**: 636,636 is Table IX-3's
  value and is correct, not the transcription slip it briefly looked like.

**A finding that is physics rather than bookkeeping:** the approach point sits **12.2 m/s
below minimum-drag speed**. `test_cruise_is_above_the_minimum_drag_speed` had asserted the
opposite for every registry entry — correctly, while every entry was a cruise point. Rather
than weaken it, it now skips the approach entry and a second test asserts the *inverse*
with the reason: 1.4 Vs at max landing weight is the back side of the drag curve, which is
where an airliner on final actually is, and is why windshear is lethal on approach and
merely uncomfortable at cruise. The autopilot's loop pairing is inverted there; all
analysis flies it open loop.

**What it bought.** The microburst re-flown as a jet transport: 1 km average F = **+0.2835**
against **+0.2094** of thrust authority — beaten by 1.4×, **2.8× the FAA hazard threshold
that now legitimately applies**, and **inside** the 0.2–0.36 band the paper reports for
real accidents. The Cherokee's answer could only ever be measured against its own thrust,
because Proctor et al. say the FAA scale was never established for piston aircraft;
`scripts/microburst.py` now decides that per aircraft instead of disclaiming it globally.

The jet has **2.7× the light aircraft's thrust authority and is still beaten**. More engine
does not buy immunity, it buys a smaller multiple.

**Deliberately not done:** the landing configuration (Table IX-1, 131 KTAS, 30° flaps,
gear down) is not added — it is the same job again and nothing yet needs it. The approach
gains are re-scaled from cruise by dynamic pressure and pass the engage-and-hold test, but
they are **not hand-tuned** and §9 session 6's point stands: that needs a human flying it.
The summary PDF is unchanged.

296 tests.

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

### Before you trust a run, check which tree it imported

This is the one failure mode here that produces **no error message at all.** Worktrees do
not get their own `.venv`; they share the main checkout's, and that editable install's
finder maps `flightsim` to the **main checkout's** `flightsim/` permanently. The finder is
*appended* to `sys.meta_path`, so it is reached only once `sys.path` has already failed —
and whether `sys.path` succeeds depends on how the process was started. Measured from a
worktree, all four rows:

| how it is run | what `import flightsim` resolves to |
|---|---|
| `.venv/Scripts/python.exe -m pytest`, cwd = **worktree root** | **the worktree.** `-m` puts cwd on `sys.path` first |
| `pytest` / `pytest.exe`, cwd = worktree root | **the main checkout.** The console script does not put cwd on `sys.path`, and `flightsim/tests/conftest.py` imports `flightsim` before pytest's own insertion helps. `sys.path[0]` *is* the worktree by the time a test body runs, which is why this one looks fine and is not |
| anything, cwd = **any other directory** — `notebooks/`, `scripts/` | **the main checkout** |
| any of the above with `PYTHONPATH` set to the **absolute** worktree root | **the worktree** |

So the documented `.venv/Scripts/python.exe -m pytest -q` is safe from the worktree root,
and **nothing else in that table is.** The check costs one line, run from the directory you
are about to run the suite from:

```
.venv/Scripts/python.exe -c "import flightsim; print(flightsim.__file__)"
```

If that path is not the tree you edited, everything downstream is about someone else's
code: a passing suite, a green notebook gate, a sanity ladder that agrees with itself, and
a measurement that lands in §4 under false provenance. A change that is *absent* from the
tree under test fails in the safest possible way — the old behaviour is asserted and
passes — which is precisely why it survives review.

### The entry points

The count is deliberately out of this heading: it read "five" over an eleven-row table for
several sessions, which is the drift §4's rules exist to prevent.

| Command | What it does |
|---|---|
| `.venv/Scripts/python.exe -m pytest -q` | 322 tests, 1 skipped. The first thing to run and the only complete statement of what works. `testpaths` is set in `pyproject.toml`, so the bare command collects `flightsim/tests`. |
| `.venv/Scripts/python.exe scripts/sanity.py` | **The ladder, for a reader who does not yet trust the model.** Twelve cases from degenerate inputs upward — zero the wind, zero a coefficient so a motion becomes impossible, then signs, then hand-computable numbers, then structural properties. Every expected value is derived by hand in the source and printed beside the model's answer, so it is read rather than trusted. Ends with the item 08 convention probe, which is a measurement rather than a pass/fail. |
| `.venv/Scripts/python.exe -m pytest --nbval-lax notebooks/ -q` | **The second gate.** Executes `notebooks/solver-validation.ipynb` so it cannot rot. Needs the `dev` extra (`jupyter`, `nbval`). Deliberately *not* in `testpaths` and `--nbval-lax` is deliberately *not* in `addopts`: that would make every `pytest` run fail with "unrecognized arguments" wherever nbval is absent. **Run it from a worktree with an ABSOLUTE `PYTHONPATH`** — nbval starts the kernel with its cwd in `notebooks/`, so a relative `PYTHONPATH=.` resolves to the wrong directory and `flightsim` silently loads from the main checkout. |
| `.venv/Scripts/python.exe scripts/checkpoint.py` | 747 only, no flags. Trim residuals, 60 s fixed-control hold, longitudinal modes against CR-2144 Table IX-5. |
| `.venv/Scripts/python.exe scripts/tune.py --aircraft cherokee` | Autopilot step responses for one aircraft. Exits non-zero on failure, so it is usable as a gate. |
| `.venv/Scripts/python.exe scripts/fly.py --aircraft cherokee --save runs/a.npz` | Interactive flight, basic-T cockpit plus a flight-test overlay. |
| `.venv/Scripts/python.exe scripts/fly.py --wind hannibal` | The same, hand-flown into the Parks vortex array. The panel counts the range down. |
| `.venv/Scripts/python.exe scripts/vortex.py --case hannibal --png runs/v.png` | Flies the 747 through the Parks vortex array, the Wingrove updraft, and an elevator pushdown, and draws the analysis figure with all three Fig. 8 categories. This is the turbulence path. Prints each point's Δθ, Δn and peak \|α\| with its band, then whether the ordering holds. |
| `.venv/Scripts/python.exe scripts/microburst.py --png runs/mb.png` | Flies the Cherokee through an Oseguera & Bowles microburst at 300 m and reports the 1 km average F against its thrust authority. Cuts the run at one wingspan above the ground and says so. |
| `.venv/Scripts/python.exe scripts/leewave.py --png runs/lw.png` | Flies the 747 through a Doyle et al. lee wave and compares the Bowles F-factor against the aircraft's own `(T−D)/W`. Prints both of the source's flight legs and which of them the engines can cover. |
| `.venv/Scripts/python.exe scripts/analyse.py runs/a.npz` | Replays a saved `.npz`. Accepts several files; `--png DIR` writes instead of showing. |
| `.venv/Scripts/python.exe scripts/vortex.py --artifacts runs/analysis` | The same run, **also written as a run artifact per encounter** — Parquet series plus `meta.json` and `checks.json`. Prints `checks ok` or names the checks that failed. Needs the `ui` extra. Until this flag existed, every number in §4's encounter tables came from a run that did not survive the script that produced it. |
| `.venv/Scripts/python.exe -m flightsim.apps.sweep runs/analysis` | **The analysis UI.** Sweep view per run — provenance header with caveats, check badges, the causal strip stack, the 3D field with the trajectory through it, Fig. 8, and n_z-vs-α — plus a shared time cursor: click any strip and every panel, the 3D marker and the readout move to that sample together. Needs the `ui` extra (`pip install -e .[ui]`). |
| `.venv/Scripts/python.exe scripts/summary.py docs/summary/flightsim-summary.pdf docs/summary/panel.png` | Rebuilds the plain-English summary PDF (14 pages). The parts that are *computed* cannot drift from the code — the vortex figures call `wind.vortex_wind`, and the aircraft table reads `CRUISE`. **The prose and the summary statistics are literals and can**: the test and line counts were stale by session 7, and four page cross-references were wrong by session 8. The page numbers are now generated from `PAGE_ORDER` with a build-time count check; the statistics are still literals. Re-run it after anything that changes those. |

### The documents, and which question each answers

| Document | Answers |
|---|---|
| **`docs/PROJECT.md`** (this file) | what exists, what is **measured**, what is known-broken, what happens next |
| **`docs/ASSUMPTIONS.md`** | what is **assumed** before any measurement, with a bound on each. Read before quoting a result to better than ~0.5%, before flying far from a trim point, or before adding a wind field whose scale approaches a wingspan. Its closing section explains what the notebook does and does not demonstrate |
| `docs/superpowers/plans/2026-08-11-close-the-verification-gaps.md` | the current plan, plus a full handover of session 11 for a session that was not there |
| `docs/superpowers/specs/2026-08-11-solver-validation-design.md` | why the verification/validation split, and the source-qualification tiers that answer "is a 1972 document a source of error" |

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

`--aircraft` accepts `boeing747`, `boeing747_approach`, `cherokee`, `cessna172`. The 747 is the only one with
modes validated against a source (§4) and the only one used for turbulence work.
`boeing747_approach` is the same airframe at CR-2144's power-approach point (sea level,
165 KTAS, 20° flaps, gear up) and exists for low-altitude windshear work; it sits below
V_md, so fly it **open loop** (§4, §5). The
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
