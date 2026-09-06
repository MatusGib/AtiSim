# AtiSim — project record

A 6-DOF fixed-wing flight-dynamics core in JAX, built as a foundation for turbulence
modelling. This document is the standing record: what exists, what is validated, what is
known-broken, and what happens next.

**Last updated:** session 27 (the recorded trace digitised at last, and it says the *wind* is 12% light; a second sealed prediction settled RIGHT; the DC-10 wing loading found unpinnable, withdrawing session 26's sign; the LES comparison audited and its numbers refused, because the frozen lift-curve slope at M 0.41 predicts the 1.42x discrepancy it reports).

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

> **`CLAUDE.md` at the repository root is the enforcing document, and rule 1 there is the
> one this file depends on: NOTHING IS DONE UNTIL IT IS IN THIS FILE.** Session 27 found
> two completed digitisations stranded in another worktree — untracked scripts, gitignored
> outputs — while that worktree's copy of this document still listed both as open work. A
> measurement that is not written down here will be paid for twice. Before ending a session,
> add the §9 entry, `git add` the scripts you wrote, and record what you deliberately did
> *not* do. Until session 27 that `CLAUDE.md` did not exist, though this line referenced it.

Rules carried from `CLAUDE.md` and enforced throughout the code:

- **Check that the tree you are testing is the tree you edited.** Every worktree shares the
  main checkout's `.venv`, whose editable install maps `atisim` to the **main checkout**
  for the life of the install. Nothing warns you when that mapping wins — the tests import,
  collect and pass, against code you did not change, and the result reads exactly like a
  real one. Print `atisim.__file__` before believing any run. §10 carries the one-line
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

### The validation claim — what this model may and may not be used for

**Added session 24 (phase 0).** Everything the project has measured lives in §4, which is
long and grows by session. This is the one paragraph a reader needs before using any of
it, and it is deliberately narrower than what the evidence might be stretched to support.

> **AtiSim is validated as a comparative and mechanistic tool for the LONGITUDINAL gust
> response of a rigid transport aircraft at cruise.** Within the envelope below it
> reproduces published response orderings, the mechanisms behind them, and the linear
> modes of its own source data. **It is not a validated absolute-load predictor**, and
> until session 24 it could not represent lateral turbulence at all.

**The envelope the claim is made inside.** Outside any one of these the claim does not
transfer, and §4 will not have measured it:

| | Validated range | Set by |
|---|---|---|
| Aircraft | 747-class transport; `boeing747`, `boeing747_jsbsim` | the only entries with a measured recovery band |
| Mach | **0.70 – 0.90** | `aircraft.valid_mach`, edges measured not assumed |
| Altitude | **35,000 – 45,000 ft** (`boeing747`); **35,000 – 41,000 ft** (`boeing747_jsbsim`) | `aircraft.valid_altitude`. The two differ and the narrower one governs any statement made about both — do not quote the union |
| Incidence | **\|α\| < 10°** green, 12° amber | `panel.ALPHA_LINEAR_DEG`; the aero has no stall |
| Gust length scale | **> 3 wingspans** | ASSUMPTIONS E2; the Parks core at 2.3–3.1 spans is the marginal case and is called out as such |
| Axis | **longitudinal only** | see §5 — no field varied across the span before session 24 |

**What "comparative and mechanistic" buys, and it is more than it sounds.** The
questions this model answers with evidence behind it are the ones most engineering
questions actually are: *which encounter is worse, how does the response scale with
airspeed, what happens to the load if the aircraft is slower, does the ordering hold.*
§4 records six-for-six monotone agreement on TM-102186's stated mechanism and both
halves of its three-aircraft ordering — including the counter-intuitive half.

**Session 25 adds a frequency-domain member to that list, and it is a different kind of
claim.** The `n_z` response spectrum of the headline run peaks 0.23 of a bin from the
aircraft's own short period and 1.64 bins from the mean core-passage frequency of the
array forcing it, and over a Dryden ensemble the response carries **3× more** energy at
the short period than at 0.05 Hz where the **input** carries more. That is a statement
about **coupling** — which frequencies the airframe selects out of what it is given —
and no comparison of peaks could have made it. It is still a comparative claim: it is
tested against the aircraft's own linearised dynamics, not against a recorded spectrum,
because none is held. §5 says what would change that.

**What it does not buy.** Any statement of the form *"the load will be X g"*. The
headline load comparison reaches **68%** of a recorded peak-to-peak — 67.1% is the
like-for-like *translational* figure quoted against JSBSim and this line used to carry it
by mistake — and §5 records what is left to explain it. **Session 27 unsettled that
attribution twice over**, and §5 now says so: the wing-loading sign that made *the
aircraft* the surviving explanation rests on a ratio that is not pinned, and the digitised
record shows the *wind* the model flies is ~12% weaker than the wind the record carries.
Absolute agreement was ruled structurally out of reach in session 7 and nothing since has
changed that.

**Three claims a reader could reasonably infer and should not.**

1. **Structural fidelity.** The derivatives are a flexible airframe's; the solver
   integrates a rigid body (ASSUMPTIONS B1). Never assert it.
2. **Behaviour away from trim.** Derivatives are frozen across the envelope
   (ASSUMPTIONS C3, unbounded). Quote the excursion with every result.
3. **Lateral behaviour, from the lateral MODES.** Dutch roll agrees to 0.4% *as an
   eigenvalue*. That is a statement about the linearisation, not about what the aircraft
   does when a gust rolls it — which nothing measured until session 24, and which is
   still not validated against any source.


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
| `predictions.py` | sealed predictions | the register of claims made BEFORE their answer is available. Not imported by any model code and deliberately not a source of numbers: nothing here may be quoted as evidence FOR the model. `test_predictions.py` enforces the rules |
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
| **`analysis/`** | `artifact.py` (run artifacts: Parquet + `meta.json` + `checks.json`, and `rebuild_field`), `series.py` (every plotted channel), `figures.py` (pure Plotly figures) | needs the **`ui` extra**. Imports `atisim`, never the reverse. Nothing in `atisim/` proper imports it, so the simulator and every script keep working without it |
| **`apps/`** | `sweep.py` — the Dash analysis UI | **the only package that imports Dash, and it computes nothing.** It never runs the simulator either: `n_steps` is a `static_argname`, so every distinct dt pays a fresh 0.6–0.9 s compile and a panel whose contents depend on machine warmth is not a check |
| `vortex_viz.py` | encounter analysis and the Fig. 8 figure | air-relative throughout; deliberately separate from `viz.py`. `fly` for a wind field with fixed controls, `manoeuvre` for an elevator schedule at zero wind; both go through `_measure`, so the three Fig. 8 points cannot drift apart |
| **`response.py`** | **tier 3 — RUN statistics**: `spectrum`, `peak_frequency`, `exceedance` | added session 25 (phase 2). A run as a SPECTRUM and as a RATE, rather than as a peak. Numpy, takes a sampled history, same standing as `checks.py` — nothing here is jitted or differentiated. Note the name collision worth keeping straight: `wind.dryden_spectrum` is an INPUT spectrum, this is the RESPONSE. Every unit check in `test_response.py` is against a signal whose answer is closed-form |

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
   `atisim/provenance.py` classifies each entry as SOURCED, DERIVED, CALIBRATED or
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
| **Parks, Wingrove, Bach & Mehta 1985**, J. Aircraft **22**(2) 124–129, DOI 10.2514/3.45095 — **HELD, session 26**, `Reference_papers/parks-1985-identification-of-vortex-induced-clear-air-turbulence-JA22-2.pdf` | **the vortex model and both identified cases, first-hand.** Rankine core, array by superposition, Eqs. (3)–(6). Case 1 Hannibal p. 127: r₀ 600 ft, V₀ 85 ft/s, spacing 3500 ft. Case 2 Morton p. 128: 450, 70, 3200. **Both DC-10s.** The abstract gives core *diameters* 900–1200 ft, which settles radius-versus-diameter. His own assessment of the two fits, p. 128, is in `wind.PARKS_FIT_QUALITY` | α is *inferred* from accelerometers through an assumed aero model — see §5. Gives **no** aircraft weight or wing area, deferring aerodynamic characteristics to a 1982 SFTE symposium paper. **Case 2 is the weaker fit and he says so**: "not as good as … case 1", attributed to "strong mountain wave activity which influences the short-period wind pattern" |
| **Wingrove & Bach 1994**, J. Aircraft 31(4) 753–760 | updraft magnitudes/duration, g-load statistics, the Fig. 8 discriminator | never identifies an aircraft type; no updraft edge gradient; no lateral data |
| **Doyle, Jiang, Smith & Grubišić 2011**, *Mon. Wea. Rev.* 139, 3–23, DOI 10.1175/2010MWR3466.1 | **the lee-wave amplitudes** — T-REX Gulfstream V over the Sierra Nevada, IOP 4 primary wave, 6 and 12 m/s crest-to-trough | gives a **tropospheric** wavelength band (20–35 km) and says stratospheric ones are shorter **without a number** — see §5 |
| **Proctor, Hinton & Bowles 2000**, 9th Conf. Aviation Range & Aerospace Meteorology, paper 7.7, 482–487 | **the F-factor** — Eq. (3) `F = U̇ₓ/g − w/Vₐ`, Eq. (4) for the shear term, Eq. (7) for the **1 km average**, the `F > (T−D)/W` thrust criterion, the 0.1/0.13 thresholds, and F = 0.2–0.36 in real accidents | its thresholds are **low-altitude** (§4.1 bounds the threat below 500 m) **and jet-transport only** — it states the scale and threshold "are yet to be determined" for piston aircraft |
| **Oseguera & Bowles 1988**, NASA TM-100632 | **the microburst** — Eqs. (5)–(6), an axisymmetric stagnation flow satisfying continuity, with four stated constants (r/R = 1.1212, z_m/z* = 0.22, z*/ε = 12.5, u_max = 0.2357λR) | the example's `R` is legible only in a scanned figure, so the downdraft radius is declared inside the 1–4 km band Wilson et al. use to define a microburst |
| **MIL-F-8785C** — **now held, `refs/MIL-F-8785C.pdf`, session 25** (5 Nov 1980, 95 pp., everyspec.com) | **the Dryden spectral forms, first-hand.** §3.7.1.2 "Turbulence model (Dryden form)", printed p. 47, prints all three components, and both implemented forms match it **verbatim** — including that the spec gives `v` and `w` identical right-hand sides, which this project had inferred from isotropy and now cites. Also `L_w` = 1750 ft above 2000 ft | **σ_w is still un-digitised.** Figure 7, *"Turbulence exceedance probability"*, printed p. 49, is confirmed present — a rotated scan, axes RMS turbulence amplitude σ (ft/sec TAS) against altitude, curves 10⁻¹ to 10⁻⁶ banded LIGHT/MODERATE/SEVERE. **Holding the document closed the FORMS, not the intensity**; every use still sweeps σ_w and reports what value the result implies. §4's σ_w ≈ 4–5 m/s is an implication, not a validated intensity |
| **Caughey, *Introduction to Aircraft Stability and Control*, Cornell MAE 5070 notes, Ch. 5** | an **independent implementation** of CR-2144's 747 power-approach case: dimensional derivatives Eq. (5.51), plant matrix Eq. (5.52), characteristic polynomial (5.53), roots (5.54) | **not an independent dataset** — its Eq. (5.48)–(5.50) cite CR-2144, the same document §IX comes from. Same inputs, different code. Also states V = 279.1 ft/s (M 0.25 at sea level) where Table IX-2's header says 165 KTAS = 278.49 ft/s, a 0.2% difference |
| **Mehta 1987**, *J. Guidance, Control & Dynamics* 10(1) 27–31 (AIAA 84-2083) | **the only wind field in the project that declares nothing**: the converged five-vortex Hannibal solution — five core positions, `r₀` = 500.5 ft, `V₀` = 86.8 ft/s, ψ = 31°, altitude, bias and trend terms. Also the identification method behind Parks, and the cost at each array size | the fit is to DFDR-derived winds, so it inherits their reconstruction error (bounded by Lester below). States the encounter as **July** 1981 where two NASA documents say April |
| **Wingrove, Bach & Schultz 1989**, NASA TM-102186 | the Hannibal encounter's **measured** normal acceleration (+1.7 to −1.0 g, gusts ~5 s apart); the vortex-array model in words (1,000 ft diameter, 87 ft/s, 3,400 ft spacing); **Fig. 8's three-aircraft simulation** at V = 150 / 700 / 800 ft/s and the mechanism it states | Fig. 8's exact wind field is not recoverable from the paper, so only orderings and excursion ratios can be compared. **Quotes Schultz 1990's Table 1 *initial estimates* as if they were his converged DFW results** — see §5 |
| **Lester, Sen & Bach 1989**, *Mon. Wea. Rev.* 117 1103–1107 | **Table 1: the RMS error of a DFDR-plus-radar wind reconstruction** (2.449 m/s horizontal, 2.236 m/s vertical at V = 250 m/s); a **B-747** mountain-wave encounter at 33,000 ft, +2.7/−1.0 g, 1,000 ft altitude gain; a **measured 22 km lee-wave wavelength** ~1 km above the tropopause | one case, over Greenland rather than the Sierra Nevada that `LEE_WAVE_AMPLITUDE` comes from; no ATC radar fixes, so the track was initialised from the pilot's log and a six-minute mean was removed from the derived vertical velocity |
| **Bach & Parks 1987**, J. Aircraft **24**(11) 789–792 — **HELD, session 26**, `Reference_papers/bach-parks-1987-angle-of-attack-estimation-JA24-11.pdf` | **the error budget on the identification this project's fields rest on.** Eq. (2) gives `C_L` from body-axis accelerations and thrust over `QS`, so **mass and wing area enter only as the ratio `m/S`**. Eq. (4)'s error analysis: the AOA estimate moves **about 0.05° for a 1% error in acceleration**, and "also about 0.05 deg for a 1% error in the lift coefficient" | **contains no DC-10.** Its two validation cases are an **L-1011** and a **B-747SP**, so it does not help §7's acquisition #1. It bounds the input, not the airframe |
| **Ashburn, Waco & Melvin 1970**, AFFDL-TR-70-101 (HICAT), AD878415 — **HELD, session 26**, `Reference_papers/AFFDL-TR-70-101-Ashburn-Waco-Melvin-1970-HICAT-AD878415.pdf` | **measured** high-altitude turbulence: probability densities and **exceedance curves of RMS gust velocity**, from U-2 flights | **its band is 45,000–70,000 ft and this project flies 33,000–41,000.** Against MIL-F-8785C Fig. 7 in that band it is an *extrapolated* check, and possibly not an independent one — the report compares itself against **MIL-A-8861A** and against **Steiner's NASA U-2** data, so the high-altitude end of Fig. 7 may descend from the same aircraft. Settling that needs the spec's Background Information and User Guide (ADA119421), which is **not held** |
| **Misaka, Obayashi & Endo 2008**, *J. Aircraft* 45(4) 1217–1229 | **the RMS normal load severity index** — `σ_n` over a moving 5 s average, moderate 0.2–0.3 g, severe ≥ 0.3 g (attributed there to Hamilton & Proctor). Defined at cruise altitude, which the F-factor thresholds are not | its own Figs. 26–27 show `σ_n` tracks the *trend* of measured acceleration and misses the peaks, by construction of the 5 s window |
| **Yoshimura et al. 2022**, *J. Appl. Meteor. Climatol.* 61 503–519 | Tables A2/A3/A5: a **third CR-2144 747 flight condition** — M 0.8 at 6,096 m — with a complete non-dimensional longitudinal set including `C_mα̇`, the flight condition, and the short-period pair (`ω_n` 1.29, `ζ` 0.57) | **not an independent dataset** — Table A2 is attributed to Heffley & Jewell, i.e. CR-2144 again. Same standing as Caughey. Its own conclusion misreads Table A5's `s⁻¹` as `Hz` — see §5 |

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

> **~~Note a source conflict:~~ ~~RESOLVED, session 23 in favour of 500 ft.~~
> **REVERSED, SESSION 26: PARKS IS IN HAND AND THE TABLE ABOVE IS HIS.**
>
> The 500 ft reading is real and is Wingrove & Bach 1994 Fig. 4's, which gives this vortex
> a **1000 ft diameter**. Session 23 adopted it on the reasoning that Fig. 4 was the source
> actually held while "Parks 1985 has never been obtained". Parks 1985 has now been
> obtained — J. Aircraft **22**(2) 124–129, DOI 10.2514/3.45095 — and says r₀ = 600 ft
> with V₀ = 85 ft/s and 3500 ft of spacing, p. 127.
>
> **His abstract settles radius-versus-diameter without any inference.** p. 124: *"the
> vortex cores had diameters in the range of 900 to 1200 ft with tangential velocities in
> the range of 70 to 85 ft/s"* — 900 = 2×450 and 1200 = 2×600. Session 22 had to deduce
> that Fig. 4's column was a diameter from Morton happening to agree; the primary source
> states it.
>
> **And the Scorer check reads differently from the paper than from memory.** p. 129 gives
> Scorer's ratio as *"of the order of 2.7"* and says that for these two cases *"the ratio
> of spacing to core diameter ranged from about 2.9 to 3.5"* — **a range across both
> cases**. This document previously said "Parks quotes 2.92"; he does not. 2.92 is our
> arithmetic on his Hannibal numbers, and that it lands inside his printed range is the
> check. It reproduces at 600 ft and does not at 500 ft, where Hannibal would give 3.50 and
> the lower end of his range would be unreachable.
>
> **~~What remains a hybrid, deliberately.~~ THE HYBRID IS GONE.** From session 22 to 25
> `PARKS_CASES["hannibal"]` paired Fig. 4's *radius* (500 ft) with Parks' *strength*
> (85 ft/s) and Parks' *spacing* (3500 ft) — a vortex no paper states, kept because §4
> baselines sat downstream of it. Session 26 restored Parks' triple and re-measured every
> baseline that moved rather than leaving the hybrid in place; `docs/ASSUMPTIONS.md` **E12**
> carries the four lineages and what the disagreement is worth.
>
> **What the correction cost, and it is in the uncomfortable direction.** JSBSim's Hannibal
> peak-to-peak `n_z` falls from 1.8969 g to **1.8162 g**, i.e. from 70.3% to **67.3%** of
> the recorded 2.70 g. A solid-body core has `dw/dx = V₀/r₀`, so a larger core at the same
> tangential velocity is a *gentler* gradient. **The load shortfall §5 records got worse.**
>
> **Both cases are DC-10s** — p. 127 and p. 128 — which was not previously recorded and
> doubles what a DC-10 derivative set buys: one set serves both validation cases.
>
> **The two fits are not equally good, and Parks says so.** Case 1 shows *"reasonably good
> agreement"*; case 2 *"some agreement, but not as good as that shown previously for case
> 1"*, which he attributes to *"strong mountain wave activity which influences the
> short-period wind pattern"*. Morton therefore carries a contaminant in exactly the band a
> vortex-passage comparison measures. **Quote Hannibal as the primary case and Morton as
> support**, and say which when they disagree. `wind.PARKS_FIT_QUALITY` carries both
> assessments.
>
> **The array spacing is confirmed, and its convention identified.** Mehta's two
> core-penetrating vortices are 4,104 ft apart *along the flight path*. The vortex lines
> run perpendicular to the wind, 31° off that path, so the separation measured
> perpendicular to the lines is 4104·cos31° = 3,518 ft, or 3,522 ft with the 160 ft
> vertical offset in quadrature — **0.6% from Parks' 3,500 ft**, from an entirely separate
> fit. So 3,500 ft is real and it is a **perpendicular** spacing. Read as an along-track
> spacing it would place the cores 15% too close, and nothing else in the project would
> catch that.

## 4. Evidence ledger

Every figure below is measured, with the tolerance the test asserts.

### The LES comparison is NOT like-for-like — input audit, session 27

**The runs exist and their numbers are not admitted here.** `scripts/les_flight.py` and
`les_compare.py` were run in the weekly worktree over all four LES domains and two aircraft,
and report AtiSim/Yoshimura rms ratios of **1.068, 1.827, 1.427, 1.420** (D01…D04). **No row
of §4 quotes them**, because an audit of the two codes' inputs finds differences large enough
to account for a ratio of that size without any code disagreement at all.

**Yoshimura's own aircraft, recovered from their own source** — `flightsim-data/src/fs.f90`
lines 252–265 in the figshare extract, which carries their **complete dimensional derivative
set**. Built into their own `A_lon` exactly as their code builds it:

| | from their `fs.f90` |
|---|---|
| short period | **ω_n 0.9023 rad/s = 0.1436 Hz**, ζ = 0.621 |
| phugoid | ω_n 0.0880 rad/s = 0.0140 Hz, ζ = 0.012 |
| airspeed `U_0` | 133.0 m/s |
| `θ_0` | **0** — they do not trim |
| `Z_δe`, `M_δe` | **0** — no control surfaces at all |
| `I_x`, `I_z`, `I_xz` | 1.5752e6, 3.8109e6, 7.5789e4 (× 9.81) |

**0.1436 Hz against the paper's stated 0.14 Hz** — so their prose is confirmed by their code,
and this project now holds the frequency as SOURCED rather than quoted.

**The mismatches, in the order they matter.**

| # | Input | Yoshimura | AtiSim as flown | Consequence |
|---|---|---|---|---|
| 1 | **Aircraft** | short period **0.1436 Hz** | `boeing747` 0.1513 Hz (**+5.4%**); `boeing737_approach` 0.2627 Hz (**+82.9%**) | Neither is their aeroplane. The resonance the whole comparison turns on sits in a different place |
| 2 | **Envelope** | n/a — frozen linear model, no envelope | `boeing747` at **M 0.31–0.50, 2,661–3,209 m** against a validated M 0.70–0.90 / 10,668–13,716 m. `checks.recovery_band` **fails at 2.63 band widths** | The 747 rows are **extrapolation**, not measurement. Its derivatives were linearised at M 0.80 / 40 kft and are frozen (ASSUMPTIONS C3, unbounded) |
| 3 | **Trim** | `θ_0 = 0`, untrimmed | trims to **α 4.92°** | Different starting attitude, and AtiSim's α excursion reaches **11.81°** — past the 10° linear band |
| 4 | **Model class** | 9-state **linear**, frozen derivatives | nonlinear 6-DOF | The irreducible difference, and the *point* of a cross-code check — but only once 1–3 are removed |
| 5 | Time step | 1/128 s | 0.02 s | Minor; bounded elsewhere at ~0.14% |
| 6 | `g` | 9.81 | 9.80665 | 0.034%, negligible |

**And one thing wrong inside their own model, which matters for anyone reading it.** Their
lateral block is commented **`!Lateral/Directional derivatives >>> B747`** while the
longitudinal set gives a 0.1436 Hz short period and inertias about 0.63× the 747's — i.e.
**their aeroplane appears to be a B787 longitudinally and a B747 laterally.** This project's
own rule against crossing sources says what to do with that: quote their **longitudinal**
comparison, which is what the load is, and do not use their lateral response as a reference.

**Why the 747/737 choice is a genuine dilemma rather than an oversight.** The LES tops out at
**5,215 m** and cannot reach 747 cruise at 11,278 m, so the comparison must happen at ~3,000 m.
At that condition the only registry entry *inside* its own validated envelope is
`boeing737_approach` — whose short period is **83% away from theirs**. The entry closest in
frequency, `boeing747`, is the one 2.63 band widths outside its envelope. **You cannot
currently match both the aeroplane and the envelope, and the runs on disk chose to match
neither cleanly.**

**The gate is now open: the four B787 numbers are obtained, and three of them are checked
against Yoshimura's own derivatives rather than merely looked up.**

| | value | provenance |
|---|---|---|
| `m` | **215,910 kg** (476,000 lb, MTOW) | SOURCED — Piano 787-8 analysis; **and confirmed by the check below** |
| `S` | **325.3 m²** (3,501.39 ft², **trapezoidal reference**) | SOURCED, same. *The definition matters*: Wimpress 3,870 ft² and Piano gross 4,028 ft² also exist and differ by 15% |
| `c̄` | **6.437 m** (21.12 ft, trapezoidal MAC) | SOURCED, same |
| `I_yy` | **≈ 2.34e7 kg·m² ± 12.7%** | **DECLARED — and it does not matter, see below** |

**The mass and area are not assumed, they are pinned by their own `Z_a`.** Inverting
`Z_α = −(C_Lα + C_D)·q̄·S/m` at their condition:

| mass | implied `C_Lα` (trapezoidal S) | verdict |
|---|---|---|
| **MTOW 476,000 lb** | **4.847 /rad** | physical, and within 2% of this project's 747 (4.944) |
| mid (OEW+MTOW)/2 | 3.634 /rad | too low for a swept transport |
| OEW 239,200 lb | 2.421 /rad | unphysical |

**So their aeroplane is a 787-8 at or near maximum takeoff weight**, and that is a conclusion
drawn from their derivative rather than an assumption fed into it. Every other coefficient
recovers physical too: `C_mα` −0.76…−0.86, `C_mq` −29…−33, `C_lβ` −0.2515, `C_nβ` +0.1596 —
all the same sign and order as the 747's.

**`I_yy` is unobservable and therefore free.** `C_mα` and `I_yy` enter the dynamics only as
their product `M_α`, which is SOURCED. Verified numerically by round-trip: `I_yy` from 1.0e7
to 4.0e7 moves `C_mα` from −0.346 to −1.383 and recovers **`M_α = −0.582000` and
`M_q = −0.537000` exactly, every time** (`_evidence/b787_consistency_check.log`). So the one
number that could not be sourced is the one number that cannot affect the answer. **Declare it,
state the range, and move on** — do not go looking for a published 787 pitch inertia.

**What remains is implementation, not acquisition:** build the registry entry, give it a
`valid_mach` / `valid_altitude` band around *its own* condition, and re-run the LES limb with
both codes flying the same aeroplane. **Note the one term AtiSim still will not carry:
Yoshimura's `M_α̇ = −0.137`** — the same `M_ẇ` class the 747's phugoid gap is attributed to
(§4, "Where the longitudinal gap comes from"). Quote that as a known, attributed difference
rather than discovering it again afterwards.

### What the LES comparison IS good for: resolution, not level — session 27

The audit above refuses the LES *load* comparison. **One thing survives it intact, and it is
worth more than the number that was refused.**

Both confounders — the wrong aeroplane and the frozen `C_Lα` — are **constant multipliers on
AtiSim's side across all four domains**: same aircraft, same condition, same Mach, only the
field changes. So they **cancel exactly** in each code's own resolution ratio. Normalising each
code to its own finest domain:

| domain | dx | AtiSim / D04 | Yoshimura / D04 | disagreement |
|---|---|---|---|---|
| D01 | 500 m | 0.0475 | 0.0632 | −24.8% |
| D02 | 250 m | 0.1157 | 0.0899 | +28.7% |
| **D03** | **70 m** | **0.5753** | **0.5724** | **+0.5%** |
| D04 | 35 m | 1.000 | 1.000 | — |

**On the two domains that resolve the turbulence the two codes agree on the resolution
scaling to 0.5% — AtiSim grows ×1.7381 from 70 m to 35 m, Yoshimura ×1.7469 — while their
absolute levels differ by 42%.** That is the project's own "comparative, not absolute" claim,
demonstrated for the first time **against an independent code on a field that was not
identified from the accelerations it is asked to predict.** Everything §1 says this model is
for, and nothing it says it is not.

**Three usable statements, and they are the ones to quote.**

1. **The load is NOT grid-converged at 35 m.** It is still growing **×1.74 per halving of the
   grid**, and *both codes agree it is*. Any CAT load computed from an LES coarser than this is
   an underestimate of unknown size — a result about **the meteorology's requirements**, not
   about either aeroplane.
2. **Below ~70 m the comparison degrades and the codes stop agreeing** (−24.8%, +28.7% at 250
   and 500 m). So 500 m and 250 m LES **cannot** drive an aircraft-load calculation, and there
   are now two independent codes saying where that floor is.
3. **The field reader is validated infrastructure**: +0.978 / −0.968 / −0.935 against
   Yoshimura's own sampled wind on u/v/w, registration residual measured at 1–2 cells (under
   150 m) and deliberately not fed back. It is reusable for any published LES.

**And it reaches Yoshimura's own conclusion by a route their arithmetic cannot.** Their paper
argues for fine grid spacing from a resonant wavelength of 200 m — computed from the 1.29 Hz
that §5 shows is a misread `s⁻¹`, where the correct figure is ~1,257 m and would argue the
opposite way. **The measurement above supports fine grids anyway, empirically, and without
touching that calculation.** Being right for a reason the source did not have is the most this
comparison can currently claim, and it is a real claim.

**What it may NOT be used for**, until a `boeing787_yoshimura` entry exists: any statement
about absolute load, any statement that the two codes agree or disagree *in level*, anything
about the 747 at that condition, and anything lateral.

### The frozen lift-curve slope, and why it probably IS the LES discrepancy — session 27

**`ASSUMPTIONS.md` C3 says derivatives are frozen across the envelope and calls the Mach axis
UNBOUNDED. The LES runs are the largest Mach excursion in the project's history, and nothing
in them accounts for it.**

| | cruise, where the 747 was linearised | the LES condition |
|---|---|---|
| altitude | 12,192 m | 3,000 m |
| airspeed | 235.9 m/s | 133.5 m/s |
| **Mach** | **0.7995** | **0.4063** |
| dynamic pressure | 8,392 Pa | 8,101 Pa — **ratio 0.965** |

**The `q̄` axis is essentially matched.** That matters, because `q̄` is the axis session 23
*bounded* (23.5% of `ω_n` per 2.48× of `q̄`). Here it is 1.04×, so that bound is not the
issue. **What moves is Mach, by ΔM = −0.393 — and the largest excursion C3 previously
recorded is the lee wave's ΔM = −0.031. This is thirteen times larger than anything the
assumption had been stress-tested against.**

`aero.py` carries Mach **only** into `wave_drag`. There is no Prandtl–Glauert factor and no
compressibility correction on `C_Lα`, `C_mα` or anything else — they are literally constant.
So the 747 flies the LES field with its **M 0.80** lift-curve slope:

- Prandtl–Glauert `1/√(1−M²)`: **1.6649** at M 0.80, **1.0944** at M 0.406
- ⇒ a slope tabulated at M 0.80 is **1.521× too large** at M 0.406
- `C_Lα` carried: **4.944 /rad**; PG-consistent value there: **3.250 /rad**

**Gust load goes linearly as `C_Lα`, so this alone predicts AtiSim/Yoshimura ≈ 1.52×.**
Measured on the runs on disk: **D03 1.427, D04 1.420** — the two domains that actually resolve
the turbulence. **The discrepancy the LES comparison reports is the size the frozen derivative
predicts, and Yoshimura's model does not share the error because theirs is linearised at
their own condition.**

**This is a mechanism, not yet a measurement**, and it is falsifiable in one run: rescale the
747's `C_Lα` by the PG ratio, re-fly D03/D04, and the ratio should collapse toward 1. **Do that
before attributing anything in the LES comparison to either code.** It is also the cheapest
Mach-axis bound the project has ever had within reach — C3 has wanted one since session 12 and
declined it because it needed chart reads off a poor scan; this needs no chart at all.

**Until then:** the LES runs stand as a capability demonstration and a reader for their field —
the field reader itself is independently validated, correlating **+0.978 / −0.968 / −0.935**
against Yoshimura's own sampled wind on all three components, with the registration residual
measured (1–2 cells, under 150 m) and **deliberately not fed back**. That part is sound and
reusable. The **load comparison** is not yet a comparison.

### The recorded trace, digitised at last — session 27

**The project quoted `wind.TM102186_HANNIBAL_NZ` = −1.0 / +1.7 g for four sessions without
ever looking at the curve between those two numbers.** `scripts/digitise_tm102186_fig6.py`
reads TM-102186 Figure 6's G LOAD panel out of `Reference_papers/19890016606.pdf` at 600 dpi,
column by column, as the top and bottom of the ink. Nothing is fitted and nothing smoothed.
1,420 columns over t = 6.9 … 301.2 s.

| Quantity | Digitised | The paper's own prose | Agreement |
|---|---|---|---|
| downward extreme | **−0.972 g** | −1.0 g | 0.028 g |
| **second** upward peak | **+1.702 g** | +1.7 g | **0.002 g** |
| 99.5th pctile, upper envelope | +1.701 g | +1.7 g | 0.001 g |
| **absolute** upward peak | **+1.855 g** | *not mentioned* | — |
| absolute peak-to-peak | **2.827 g** | 2.70 g implied by the band | — |
| pre-encounter cruise datum | **0.9513 g** (sd 0.0100) | 1.000 g by definition | **unresolved, declared** |

**Three checks, none of which set a parameter.** (1) The paper's band, above — the prose
quotes the *sustained* band, and the absolute peak is a single narrow spike 0.153 g above it
that the paper does not mention. (2) **A negative control on the method**: the same
column-envelope extraction run on the *vertical wind* panel must return zero through the
pre-encounter cruise, because the aircraft was straight and level in smooth air. It returns
**+1.17 ft/s**. So the method carries no vertical bias and the 0.951 g datum belongs to the
G LOAD panel, not to this code. (3) **Gust spacing**, the one channel Mehta's identification
did not set.

**Two results that change what §5 may claim.**

**(a) The model's wind is weaker than the recorded wind.** The digitised vertical-wind panel
reaches **+64.8 / −98.3 ft/s**. The 747 flown through Mehta's five-vortex field sees
**+59.1 / −86.8 ft/s** — **11.7% weaker on the down gust, 8.8% on the up.** Mehta's field is
a smooth five-vortex *fit* to that record, so under-shooting its extremes is expected; what
is new is that the size of the under-shoot is now **measured** rather than assumed away. In
the linear range load goes with gust, so this is a previously-unattributed piece of the 32%
shortfall, and it points at **the wind**, which §5 had stopped naming as a candidate.

**(b) The spacing comparison gets tighter, and the model gets worse against it.** The two
principal up-gusts land at t = 209.3 and 214.9 s, i.e. **5.60 s apart**. `cat_uncertainty.py`'s
flown model gives 5.36 s centre-to-centre. Against the digitised figure that is **−4.3%**;
against the prose *"about 5 sec apart"* alone it read **+7.2%**. **The figure is a tighter
reference than the sentence describing it**, and the sign flips.

**One thing declared rather than resolved.** The trace sits at 0.9513 g through level cruise
where the definition says 1.000. Check (2) rules out the method. It is either a recorder bias
or a registration offset of the plotted curve inside its own axes, and **the figure alone
cannot separate those**. Both envelopes are therefore reported RAW, in plot coordinates,
because that is what the paper's own numbers are. **Do not "correct" the trace to 1.0 g** —
that would be fitting the record to the model.

**Which denominator to quote.** Three are now defensible and they are not interchangeable:
2.70 g (the prose band, and what §4's headline row has always used), 2.674 g (the digitised
*sustained* band) and 2.827 g (the digitised *absolute* peak-to-peak). The 747's 1.839 g is
**68.1%**, **68.8%** and **65.0%** of them respectively. **§4's headline row keeps 2.70 g**
so the number does not silently move, and this row records the other two beside it.

### MIL-F-8785C Figure 7, and a second sealed prediction settled — session 27

`scripts/digitise_mil_f_8785c_fig7.py` reads Figure 7 out of `refs/MIL-F-8785C.pdf` p. 49.
At 37,000 ft the severe curve gives **σ_w = 15.7 ft/s = 4.80 m/s**.

**It is a band, not a point, and the reason is in the figure.** SEVERE and the 10⁻⁵
exceedance curve are drawn as **one stroke of ink** at that altitude, so the value is
**4.80 ± ~0.12 m/s** depending which member the stroke belongs to. This project does not read
a number out of a merge and call it one line.

**Independently cross-checked.** JSBSim's `FGWinds.cpp` transcribes the same figure; its
severe row interpolated to 37 kft gives **4.822 m/s** — 0.5% away, inside the merge band, and
neither reading set the other. *An earlier pass here quoted 0.4% by reading the merged stroke
as though it were SEVERE alone; that was too good and is superseded.*

**This settles `mil_f_8785c_sigma_w_exceeds_the_mehta_ceiling`: RIGHT**, by 0.34 m/s against
`wind.mehta_residual_ceiling` = 4.459 m/s, with the whole merge band clearing it. **Two of
three sealed predictions are now settled and both were right.** What it buys is what the
sealed reasoning reserved: the ~4–5 m/s closure is **physically available** at the
specification's severe level. It is **not** evidence that Hannibal contained it.

### The Dryden ensemble at Yoshimura's own sample count — session 27

Session 25 ran 32 flights where Yoshimura et al. 2023 used **151**, and §5 carried the
under-sampling as a caveat. `scripts/cat_spectra.py --seeds 151` closes it. 151 × 100 s at
each intensity, 15,097 s of record apiece.

| σ_w, m/s | peak at **N = 32** | peak at **N = 151** | vs short period (0.1640 Hz) |
|---|---|---|---|
| 2.108 (`mehta_unmodelled_wind`) | 0.1400 Hz | **0.1700 Hz** | **+3.6%** |
| 4.459 (`mehta_residual_ceiling`) | 0.1700 Hz | **0.1700 Hz** | **+3.6%** |

**The intensity-dependence was under-sampling, and it is gone.** At N = 32 the low-σ peak sat
0.1400 Hz — 14.7% *below* the short period and less than one bin inside the sealed band's
lower edge — and session 25 read that as the falling Dryden input pulling the resonance down.
At N = 151 **both intensities land on the same 0.1700 Hz**, per-flight median 0.1700 in both
cases. So the honest reading is that the N = 32 low-σ figure was noise, not a competing
mechanism, and the resonance is **cleaner** than the settled prediction's outcome suggests.

**The settled prediction's `outcome` is deliberately NOT edited.** It records 0.1400 / 0.1700
at N = 32, which is what was measured when it was settled, and `predictions.py` rule 1 exists
precisely so a settled entry is not improved after the fact. **It stands as written; this row
supersedes it as a measurement, per §4's own rule.** The verdict is unchanged either way —
RIGHT, at both intensities, and now with room to spare.

Half-power band 0.1200–0.2300 Hz at σ = 2.108 and 0.1200–0.2400 at 4.459. Peak |Δn| 0.3698 g
and 0.8378 g. The σ ratio is 2.1156 against an `n_z` rms ratio that tracks it, so the response
is linear in intensity across the sourced range — which is the other thing the two σ values
were there to measure.

### The CAT source pass (session 23)

Everything here comes from `scripts/cat_validation.py`, whose printed report and four
figures are the artefact. Run it as:

```
PYTHONPATH=<worktree root> .venv/Scripts/python.exe scripts/cat_validation.py --outdir runs/cat
```

**`PYTHONPATH` is not optional** — `python scripts/…` resolves `atisim` to the **main
checkout**, which §10 documents and which this script caught on its first run by printing
`atisim.__file__` before doing anything else.

#### Mehta 1987's five-vortex field, flown by the 747 at Mehta's own 37,000 ft

| Quantity | Model | Source | Note |
|---|---|---|---|
| vertical gust peak | −86.8 / +59.1 ft/s | `V₀` = 86.8 ft/s | the negative peak lands on `−V₀` exactly, which is the Rankine core signature |
| load factor `n_z` | **−0.398 to +1.441 g** | **−1.0 to +1.7 g** (DC-10, TM-102186 Fig. 6) | inside the measured band, at **68%** of its peak-to-peak |
| air-relative α | −5.98 to +7.86° | — | inside the 10° linear band |
| air- vs inertial α | diverge by up to **6.85°** | — | the reason `vortex_viz` exists separately from `viz`. Computed by the script, not read off the figure |
| start out of trim | **−0.0001 g** | — | `fly_in_moving_air`; the still-air start was **+0.198 g** out and inflated peak load by 12% |
| `σ_n` | **0.6394 g** | severe ≥ 0.3 g (Misaka) | severe, on a field identified from an encounter that injured people |

**Absolute agreement is not claimed and is not available** — DC-10 against 747, roughly
0.8× the wing loading, §5 rules it structurally out. What is claimed is that the model
lands **inside** the measured band and reaches about two thirds of it.

#### TM-102186 Fig. 8 — the ordering, and the mechanism

Six aircraft through the same field, at their own cruise conditions. `trav/T_sp` is core
traverse time over the aircraft's own short period, which is the paper's stated mechanism.

| aircraft | V ft/s | `trav/T_sp` | pitch p-p | `n_z` min | incidence gain | \|α\| peak |
|---|---|---|---|---|---|---|
| cherokee | 164 | 4.024 | 46.98° | +0.403 | 0.21 | 5.44° |
| cessna172 | 197 | 4.046 | 74.92° | +0.413 | 0.18 | 5.51° |
| boeing737_approach | 439 | 0.599 | 17.17° | −0.136 | 0.54 | 7.06° |
| boeing747_approach | 278 | 0.512 | 40.08° | +0.070 | 0.56 | 9.78° |
| boeing737 | 776 | 0.363 | 7.35° | −0.374 | 1.00 | 5.17° |
| boeing747 | 774 | 0.196 | 6.70° | −0.398 | 1.49 | 7.86° |

TM-102186 Fig. 8, **digitised** (±3°, ±0.15 g — extrema only, off a 1989 photocopy;
nothing asserts against these):

| | V ft/s | pitch p-p | `n_z` min | `n_z` max |
|---|---|---|---|---|
| RPV | 150 | 66.0° | +0.35 | 2.05 |
| EXECUTIVE | 700 | 16.0° | −0.35 | 1.95 |
| AIRLINER | 800 | 5.0° | −0.55 | 1.55 |

**Three results, in increasing order of how much they are worth.**

1. **The pitch ordering reproduces.** Slow aircraft pitch hugely, fast ones barely.
   Cherokee 46.98° against the RPV's 66.0°; 747 6.70° against the AIRLINER's 5.0°, inside
   the ±3° read error. The middle slot is empty — the registry has nothing at 700 ft/s and
   the nearest entry is a different flight regime, so it was left empty rather than filled.
2. **The `n_z` minimum ordering REVERSES, in both.** Reference: RPV +0.35, EXECUTIVE
   −0.35, AIRLINER −0.55. Model: cherokee +0.403, 737 −0.374, 747 −0.398. The aircraft
   that pitches least eats the most load. This is the non-obvious half of the paper's
   claim and the model gets it, with the cherokee/RPV pair agreeing inside the read error.
3. **The mechanism itself is monotonic, six for six.** *Incidence gain* is the α that
   actually reached the wing divided by the α a rigidly-held attitude would have seen
   (`atan(max|w_up|/V)`). Against `trav/T_sp` it falls **1.49 → 1.00 → 0.56 → 0.54 → 0.21
   → 0.18** with no inversions. Above 1 the aircraft's own motion *adds* incidence; below
   it, the aircraft pitches away and sheds the gust. Peak-to-peak pitch does **not**
   collapse as cleanly, because it also scales with how large the gust is in incidence
   terms, and that varies fourfold across the fleet through airspeed alone.

**Every run stays inside the 10° linear band — and the mechanism is why.** A Cherokee at
50 m/s meets a gust worth 25.6° of incidence at a frozen attitude and sees 5.44°, because
it has time to pitch into the flow. Without that relief the low-speed runs would be
outside the model's validity and the test could not be run at all.

**Confound, stated:** each aircraft flies the field at its own cruise altitude, so density
varies with the aircraft. The vortex field itself does not — `vortex_wind` is a velocity
field with no density in it — and the 747 flies Mehta's own 37,000 ft, so the headline
case carries no such confound.

#### Where the 32% load shortfall is NOT (session 23 follow-up)

`scripts/cat_bounds.py` §A. The Mehta run reaches 68% of the DC-10's recorded
peak-to-peak. Three candidate causes inside the model, each turned one at a time:

| Variant | `n_z` min | `n_z` max | Δ(up increment) | Δ(down increment) |
|---|---|---|---|---|
| baseline — tangent gust rates, point loads | −0.3976 | 1.4410 | — | — |
| gust rates fitted across the airframe | −0.3361 | 1.4336 | **−1.67%** | **+4.40%** |
| strip-integrated loads | −0.3976 | 1.4410 | **0.00%** | **0.00%** |

Step size, peak load: dt 0.02 → 0.01 moves it **−0.085%**, → 0.005 **−0.032%**,
→ 0.0025 **−0.021%**. Converged.

**Together these bound the model's own approximations at about 4.4% of a 32%
gap.** The shortfall is not numerics and not the point-sampled gust.

**The strip result is exactly zero, and that is informative rather than a
null.** `loads.strip_increment` is roll-only and `vortex_wind` has no `y`
dependence at all, so integrating across the span *must* return the point value.
The point-gust assumption on this field can only be probed through the
longitudinal gust RATES, which is what the sampled-rate row does — and it is the
**down** excursion that moves, not the peak.

#### What the random component Mehta excludes is worth (session 23 follow-up)

`scripts/cat_bounds.py` §B. Mehta states his fit represents "the sudden,
violent, and periodic disturbances … and not the small, random fluctuations that
are part of the overall turbulence", so the field flown is a *smoothed* version
of the air the DC-10 met. A Dryden layer (`wind.dryden_vertical_field`, MIL-F-8785C
form, `L_w` = 1750 ft sourced, **σ_w not sourced and therefore swept**) was
superposed, six seeds per point:

| σ_w m/s | `n_z` min, mean [min, max] | `n_z` max, mean [min, max] | reaches the band? |
|---|---|---|---|
| 0 | −0.398 | 1.441 | no |
| 2 | −0.421 [−0.500, −0.343] | 1.527 [1.440, 1.561] | no |
| 4 | −0.423 [−0.613, −0.234] | 1.642 [1.456, 1.737] | **yes** |
| 6 | −0.415 [−0.730, −0.148] | 1.797 [1.643, 1.952] | **yes** |

**σ_w ≈ 4–5 m/s closes the upper extreme. Nothing in this sweep closes the lower
one** — at σ_w = 6 the mean minimum is −0.415 against a recorded −1.0, and even
the most extreme of six seeds reaches only −0.730. The spread widens; the mean
barely moves, because the down-peak is set by the vortex core and the Dryden
layer only adds variance around it.

**Do not read σ_w ≈ 4 m/s as a validated intensity.** It is the value the gap
implies, and MIL-F-8785C gives σ_w as a chart against altitude and exceedance
probability that §3 records as un-digitised. Comparing the two is the next step,
not a step already taken.

> **Session 23c bounded it from the other direction, and the news is not good for
> this explanation.** Mehta's own fit residual puts the unmodelled random
> component at **2.11–4.46 m/s**, and 4.46 is the ceiling — it requires the
> *entire* residual to be vertical, unmodelled, and free of reconstruction
> error. So σ_w ≈ 4–5 m/s is not merely un-validated, it sits **at or above the
> top of what the source data permit.** Flown at the sourced lower bound the run
> does not reach the band at all. See the next subsection.

#### The Hannibal comparison with error bars on both sides (session 23c)

Everything here comes from `scripts/cat_uncertainty.py`, figure `06-uncertainty.png`.
Until this pass every load number quoted against TM-102186 was a point compared
with a point, so "68% of the recorded peak-to-peak" could not be split into *the
model is 32% wrong* and *the inputs are not known to 32%*.

**The standing problem this addresses.** The vortex parameters were identified
**from** the recorded accelerations, through Parks' and Mehta's aircraft model.
Predicting those accelerations back therefore tests the composition of two
aircraft models, not this one alone. Two ways out are taken below: a channel
outside that loop, and a bound on what the loop's inputs are worth.

##### A. Gust spacing — the channel the identification did not set

TM-102186 p. 3–4, in **prose**, not a figure: *"sharp up-and-down gusts about
5 sec apart"*. Spacing is set by the fitted core **positions** and the aircraft's
speed; it is independent of the fitted **strengths**, and that independence is
measured rather than asserted — the field is exactly linear in `V₀`, so scaling
`V₀` by 0.5, 1.5 or 3.0 moves every wind value and **not one turning point**.

| Reading of "apart" | Model | vs record |
|---|---|---|
| peak-to-peak | 5.290 s | **+5.8%** |
| centre-to-centre | 5.360 s | **+7.2%** |
| trough-to-trough | 5.430 s | **+8.6%** |

The three span 0.14 s, so the answer is a property of the field and not of the
definition. Geometric prediction from the 4,104 ft core separation at this 747's
774 ft/s is 5.302 s, and the flown run adds 1.1% to it.

**The residual is a speed proxy, not a field error.** Exactly 5.0 s needs
**250.2 m/s (M 0.848)**; this 747 flies **235.9 m/s (M 0.800)**, which is the
Mach its derivative set is tabulated at. The DC-10's own true airspeed is in no
source held here, and 6% between two transports at 37,000 ft is unremarkable.
The model is **slow**, which is the direction that mismatch requires — a fast
model would have needed explaining.

##### B. What Mehta's own fit leaves over

Mehta's Appendix Eq. (A3) is **`J = (1/N) Σ eᵀ B e`**, with `B` the identity for
every cost he quotes. **The `1/N` is the load-bearing part**: `J` is a *mean*
square, so it converts to an RMS wind residual without `N` — which the paper
never states and no source here supplies. Read as a *sum* it is uninterpretable:
over any plausible `N` the implied residual lands well below the error of the
winds being fitted — a factor of **4.1 even at `N` = 30**, and 6.0 at `N` = 65 —
which no honest fit can do.

| n | cost `J` (ft/s)² | RMS residual | Δ`J` | Δ RMS |
|---|---|---|---|---|
| 2 | 482 | 6.692 m/s | — | — |
| 2 | 355 | 5.743 m/s | −26.3% | −14.18% |
| 3 | 303 | 5.306 m/s | −14.6% | −7.61% |
| 4 | 226 | 4.582 m/s | −25.4% | −13.64% |
| 5 | **214** | **4.459 m/s** | −5.3% | −2.69% |

The first row is the **manual startup estimate**, not a fit — see §3, where this
document previously paired it with a converged cost as one "history".

**J = 214 is a floor, not a stopping point.** Mehta p. 30: *"Further increases in
the number of vortices (n = 6,7, etc.) do not result in decreases in the cost. In
fact, the algorithm 'pushes' the extra vortices away from the flight path"*. The
fifth vortex already buys only 2.7% of RMS. **So the residual bounds the field
FORM** — what a Rankine array cannot represent about this record — rather than
one author's patience.

Decomposing it against Lester's reconstruction error, since Mehta fitted
*reconstructed* winds and his residual therefore contains theirs:

| | RMS wind |
|---|---|
| total residual, `√J` | 4.459 m/s |
| reconstruction floor, Lester Table 1 RSS | 3.316 m/s |
| **unmodelled, per component** (quadrature) | **2.108 m/s** — a *lower* bound |
| **ceiling**, all of `J` vertical and unmodelled | **4.459 m/s** — unit-robust |

Both caveats on the lower bound push the same way, **up**: independence is
assumed where Mehta fits bias and trend terms explicitly, and Lester's case had
no ATC radar fixes where Hannibal did. The ceiling survives any unit convention
for the horizontal term because `B = I` makes both squares non-negative.

**Flown, 8 seeds each:**

| σ_w | `n_z` min, mean [min, max] | `n_z` max, mean [min, max] | reaches band? | peak \|α\| |
|---|---|---|---|---|
| 2.108 (sourced lower) | −0.394 [−0.506, −0.246] | 1.529 [1.441, 1.568] | **no** | 8.85° |
| 4.459 (ceiling) | −0.367 [−0.641, −0.031] | 1.663 [1.459, 1.790] | yes | 9.97° |

**This weakens session 23b's Dryden reading rather than confirming it.** The
intensity that closes the gap is the intensity at the absolute ceiling of what
the sources permit, and the ceiling run sits **on** the 10° edge of the linear
range — the last intensity this model may be asked about at all. What the pass
*does* buy is that σ_w went from unbounded to **2.11–4.46 m/s** without
digitising anything.

##### C. The propagated input band

`V₀ ± 8.45%` is sourced (Lester's vertical RMS over Mehta's `V₀`) and is a
**ceiling, not a 1σ** — it is the error of one reconstructed sample, and `V₀` was
fitted to `N` of them. `r₀ ± 15%` is **declared**: Mehta's sensitivity study
reports convergence from initial guesses of 100–1300 ft, which is a statement
about his algorithm, not about how well `r₀` is known.

| Variant | `n_z` min | `n_z` max | p-p | % of recorded | \|α\| |
|---|---|---|---|---|---|
| baseline (identified) | −0.3976 | 1.4410 | 1.8385 | 68.1% | 7.86° |
| `V₀ × 0.9155` (sourced) | −0.3312 | 1.4287 | 1.7599 | 65.2% | 7.72° |
| `V₀ × 1.0845` (sourced) | −0.4374 | 1.4482 | 1.8855 | 69.8% | 7.89° |
| `r₀ × 0.85` (declared) | −0.4582 | 1.4765 | 1.9348 | 71.7% | 8.01° |
| `r₀ × 1.15` (declared) | −0.3418 | 1.4470 | 1.7888 | 66.3% | 7.74° |
| **corner: `V₀` high, `r₀` low** | −0.5065 | 1.4565 | 1.9631 | **72.7%** | 8.15° |

**The whole input band tops out at 72.7%. The 32% shortfall is not inside the
uncertainty of the inputs.**

##### Why no amount of wind helps: the peak load is saturated

| `V₀ ×` | `n_z` min | `n_z` max | up increment | elasticity | \|α\| | |
|---|---|---|---|---|---|---|
| 1.00 | −0.3976 | 1.4410 | 0.4410 | — | 7.86° | |
| 1.25 | −0.4422 | 1.4567 | 0.4567 | 0.143 | 7.76° | |
| 1.50 | −0.3372 | 1.4465 | 0.4465 | 0.025 | 7.97° | |
| 2.00 | −0.1429 | 1.3820 | 0.3820 | −0.134 | 7.98° | |
| 3.00 | +0.0820 | 1.5257 | 0.5257 | 0.096 | 8.13° | |
| **3.25** | −0.2002 | **1.6416** | 0.6416 | 0.202 | **9.11°** | short, **inside** |
| **3.50** | −0.5839 | **1.7862** | 0.7862 | 0.313 | **10.34°** | reaches, **outside** |
| 4.00 | −1.9049 | 2.2065 | 1.2065 | 0.579 | 13.93° | outside |

**Tripling the identified gust leaves the up-increment at 0.526 g against a
recorded 0.70, with peak |α| still near 8°.** A peak that tracked the gust would
have elasticity ≈ 1; this one is under 0.15 and changes sign. The aircraft
pitches away and sheds the gust — TM-102186 Fig. 8's incidence-gain mechanism
seen from the inside.

**And the two boundaries coincide, which is the sharp form of the result.** The
peak first reaches +1.7 g between **×3.25 and ×3.50**, and |α| leaves the 10°
linear range in the **same interval**. So there is no gust strength at which this
model both reaches the record and may be believed. That is a stronger statement
than "the shortfall is large": it **excludes amplitude** as the explanation
rather than merely bounding it. Anything past the crossing is past the 12° where
§7 says this aero reports lift the sources deny — not a harder test, a different
model.

**What all three sections together leave.** The shortfall is not numerics (0.14%
over 8× `dt`, session 23b), not the point-sampled gust (≤4.4%), not the strip path
(0.00%), **not the identified parameters (≤72.7% at the favourable corner), and
not gust amplitude at any strength inside the linear range.** The random
component is bounded to 2.11–4.46 m/s and only its ceiling reaches. What is left
is the aircraft: a DC-10 record flown by a 747, and no buffet-onset data — which
§5 already lists as structurally out of reach from the sources held.

#### Lester's Greenland 747 — one wave cannot produce both observations

`scripts/cat_bounds.py` §C. The first load comparison in the project flown by
the aircraft type the record is *of*: a B-747, at Lester's 33,000 ft, through a
22 km lee wave.

| `w0` m/s | `n_z` min | `n_z` max | peak \|Δn\| | climb m | band widths out |
|---|---|---|---|---|---|
| 3 | 0.955 | 1.052 | 0.052 | 223 | 0.25 |
| **6** (Doyle's largest) | 0.911 | 1.106 | **0.106** | **451** | 0.29 |
| 12 | 0.826 | 1.222 | 0.222 | 915 | 0.39 |
| 30 | 0.602 | 1.610 | 0.610 | 2367 | 0.69 |

Lester records **two** things about the same event, and the model inverts each:

- the **300 m altitude gain** needs `w0` ≈ **4.0 m/s** — squarely inside Doyle's
  measured 3–6 m/s;
- the **+2.7/−1.0 g** needs `w0` ≈ **113 m/s**, which is 18.8× Doyle's largest
  and not a physical lee-wave amplitude.

**The two imply amplitudes 28× apart, so one smooth wave cannot produce both.**
The slow, large-scale response is reproduced at a plausible amplitude; the fast
one is not reproduced at any. That is a statement about the WIND MODEL, not the
flight dynamics — and it is exactly what the source says happened, since Lester
reads the flight-level windspeed collapse at the point of largest vertical
motion as a critical level from *overturning* waves. The loads came from the
wave breaking down, which `wind.LeeWave` does not contain.

**Caveat, reported by the run itself.** The band declared on `boeing747` this
session puts 33,000 ft outside it: at `w0` = 6 m/s the run is 0.29 band widths
out, on the altitude axis only (Mach 0.764–0.835 is inside). `aircraft.py`'s
interpolation puts the short-period frequency error there at about 5%. The guard
fired on its first real use.

#### The 747 at a third CR-2144 flight condition, M 0.8 / 6,096 m

Condition reproduced before any derivative is used: `U₀` 252.8 m/s against Table A3's 253,
`ρ` 0.6527 against 0.653, and **`C_L` 0.2657 against 0.266** — the last from mass, wing
area and the ISA atmosphere alone, using no derivative from either document.

| | `ω_n` | vs ref | `ζ` | vs ref |
|---|---|---|---|---|
| AtiSim, FC9 (40,000 ft) derivatives | 1.5926 | **+23.5%** | 0.4723 | −17.1% |
| AtiSim, Table A2 (6,096 m) derivatives | 1.2823 | **−0.6%** | 0.5031 | −11.7% |
| reference, Table A5 | 1.2900 | — | 0.5700 | — |

**The frequency error is entirely the derivative set.** Swap the coefficients for the ones
CR-2144 tabulates at *that* condition, change nothing else, and 23.5% becomes 0.6%.

**The residual damping error is entirely the missing `C_mα̇`.** Deleting `M_α̇ = −0.176`
from the reference's own damping formula predicts −12.2%; the model measures −11.7%.
Agreement to half a point of damping says nothing else contributes.

**And putting it back closes it, which is the prediction run forward.** The model has
carried `CLadot`/`Cmadot` fields since the α̇ work — `dynamics.derivatives` resolves both
the gust's half and the aircraft's own implicit half — but `boeing747` leaves them at
zero. Restoring Table A2's own `C_mα̇` = −5.40 alongside its other derivatives:

| 747 short period, M 0.8 / 6,096 m | `ω_n` | vs ref | `ζ` | vs ref |
|---|---|---|---|---|
| Table A2 derivatives, no α̇ | 1.2823 | −0.59% | 0.5031 | **−11.73%** |
| **+ `C_mα̇` = −5.40 (Table A2)** | 1.2823 | −0.59% | **0.5708** | **+0.14%** |
| reference, Table A5 | 1.2900 | — | 0.5700 | — |

**Given CR-2144's own derivatives for the condition, including α̇, this model reproduces
CR-2144's own short period to 0.6% in frequency and 0.14% in damping.** `ω_n` does not
move, which is the check that the term went into the right place — `M_α̇` enters the
damping and not the frequency.

**`boeing747` is NOT given α̇ terms, because no source supplies them at flight condition 9.**
Table A2 is 20,000 ft and this entry is 40,000; applying it across 2.48× of dynamic
pressure is the exact error this section just measured. CR-2144 Table IX-4 tabulates
`Zwd` and `Mwd` for FC9 and `aircraft.py`'s comment says so, but the values were never
transcribed and the document is not held.

**`boeing747_approach` is not given them either, and that one is a judgement call.**
Table IX-2 *does* tabulate `CL_α̇` and `Cm_α̇` for that condition, and `aircraft.py`
records them. Measured against Caughey Eq. (5.54):

| | `ω_n` sp | `ζ` sp | `ω_n` ph | `ζ` ph |
|---|---|---|---|---|
| as shipped | +1.4% | −5.5% | −0.2% | −4.6% |
| with `C_mα̇` −3.2 and `C_Lα̇` +6.7 | **−0.3%** | **+0.1%** | −0.2% | **+8.4%** |

The short period — the mode α̇ physically governs — improves markedly. The phugoid
damping, which it should not touch, degrades **past the 5% tolerance
`test_validation.py` asserts**. Adopting the terms would mean re-pinning that tolerance
to let the change through, and `docs/ASSUMPTIONS.md` B5 is the precedent for not doing
that. The measurement is kept in
`test_cat_validation.py::test_the_approach_747s_tabulated_alpha_dot_terms_are_a_trade_not_a_win`
so it survives the decision not to act on it.

**The sign is resolved, and empirically.** `aircraft.py` transcribes `CL_α̇` as −6.7;
Caughey uses +6.7 for the same CR-2144 case, and +6.7 is what a conventional aft tail
must have. Flown both ways, +6.7 takes short-period `ω_n` to −0.3% and −6.7 takes it to
+3.2% — the data agrees with the physics, so the transcribed sign is the wrong one.

Verification that the reference is usable at all, before any of the above: Table A2's
dimensional column is recovered from its non-dimensional column using **AtiSim's own**
mass, inertia, wing area and MAC, worst error **0.47%** across five derivatives. Same
aeroplane, and both transcriptions sound.

#### The headline field flown by a second engine (session 23d)

`scripts/vortex_compare.py`, now carrying a fifth encounter. Session 21 compared
the two engines on **single Parks cores**; this is the first time both have flown
**Mehta's converged five-vortex array** — the field the headline result actually
uses — with the same aircraft (`boeing747_jsbsim`, recovered from JSBSim's own
B747), the same starting state and the same density.

**The gate first.** The generator writes the array field in numpy with an oblique
traverse; `wind.vortex_wind` writes it in JAX. They agree to **8.8e-10 m/s** at
every sample, so neither the five-way superposition nor the 31° obliquity is
carrying a shared bug.

| | `n_z` span | `θ` span | % of the recorded 2.70 g |
|---|---|---|---|
| **JSBSim** | **2.0152** | **10.1066°** | **74.6%** |
| AtiSim, translational (like-for-like) | 1.8124 | 10.1854° | 67.1% |
| AtiSim, + `ω_gust` | 2.0517 | 9.3905° | 76.0% |
| AtiSim, + both gradient terms | 2.0011 | 9.4123° | 74.1% |
| like-for-like error | **−10.1%** | **+0.8%** | |

**The result step 5 existed to get: neither engine reaches the record.** An
independent flight-dynamics code, given the same field and the same aeroplane,
also falls well short of the DC-10's recorded peak-to-peak. The solver is
excluded as the explanation **on the headline field**, rather than inferred from
adjacent cases.

##### But the two engines part on the array where they did not on one core

This qualifies session 22, which measured both engines under-predicting "by the
same amount" and read that as the shortfall being in the inputs.

| | window | in short periods | JSBSim `n_z` extremes | AtiSim `n_z` extremes |
|---|---|---|---|---|
| Hannibal, one core | 305 m, 1.3 s | 0.20 | max @ 2135 m, min @ 2434 m | max @ 2135 m, min @ 2436 m |
| Mehta, five cores | 8,125 m, 34.4 s | 5.2 | max @ 7794 m, min @ 7383 m | max @ 6511 m, min @ 6116 m |

On the single core the engines put their extremes **within two metres of each
other**. On the array they **select different cores** — JSBSim's pair sits on core
3 (7,209 m), AtiSim's on core 2 (5,958 m). And the disagreement is one-sided:
AtiSim's **peak is 2.2% higher** (1.7148 vs 1.6773) while its **trough is 3.5×
shallower** (−0.0976 vs −0.3379). The whole −10.1% span error is the trough,
which is the channel that has been short against the DFDR from the beginning.

**A hypothesis that was tested and failed.** The obvious cause is airspeed: the
array run is 45 s at fixed throttle and JSBSim's true airspeed swings −5.5% across
the window against −0.009% on the single core, and `n_z ∝ q̄ ∝ V²`. But shrinking
the window to ±2 core radii around the penetrated pair leaves the drift at −0.19%
and the disagreement at **−7.6%**, and AtiSim's span is **identical (1.8124) at
every window width from ±2 to ±32 radii**. The drift is not the cause; it is
recorded here so the next reader does not spend the same hour on it.

**What is left, stated as the open question it is.** The array is the first
cross-code comparison that runs for *several* short periods rather than a fifth
of one, and the engines' trajectories diverge enough over five cores to land on
different worst cores. Whether that is a damping difference, a thrust/drag
difference over 45 s, or an accumulation of both is not established here.
§8 carries it.

#### Fig. 8 with error bars — §7 step 6, at last (session 23d)

`scripts/cat_ensemble.py`, figure `07-ensemble.png`. Step 6 has waited on step 4
(Dryden) since session 3; step 4 landed in 23b. Its stated criterion is that the
**vortex < updraft < manoeuvre ordering holds across the ensemble.**

σ_w is the **sourced** range from 23c — `wind.mehta_unmodelled_wind` to
`wind.mehta_residual_ceiling` — rather than a picked number. 16 seeds each.

| σ_w | vortex pitch | updraft pitch | ordering | pitch gap | load gap |
|---|---|---|---|---|---|
| 2.108 m/s | 2.113° [1.619, 2.432] | 4.591° [3.318, 5.699] | **16/16** | +0.886° | +0.756 g |
| 4.459 m/s | 2.114° [0.988, 2.809] | 5.253° [2.869, 7.286] | **16/16** | **+0.060°** | +0.343 g |

**The criterion is met, and the interesting number is the margin.** Raising σ_w
by 2.1× collapses the **pitch** gap between the vortex and updraft clouds by 15×,
to 0.060° — they are all but touching. The **load** gap survives at 0.343 g. Fig. 8
is a two-dimensional discriminator, so the categories still separate; but at the
top of the sourced turbulence range they separate on **load alone**, and a pitch-only
reading of the chart would stop working.

**Two things this does not establish.** The manoeuvre limb is flown at zero wind —
that is the category's definition, not an oversight — so the three-way ordering is
helped by construction and the vortex-versus-updraft separation is the part tested
on equal terms. And a 16-seed gap is an estimate of the extremes of a distribution
whose tails grow with sample size, so "the clouds do not overlap" should be read as
*marginal*, not as established.

#### Sealed predictions — the register (session 23d)

`atisim/predictions.py`. Every number in this section is **retrodictive**: the
paper was open beside the model. That is the weaker kind of evidence and no amount
of it becomes the stronger kind. The register is where this project starts saying
what will happen **before** it can check.

Two entries, sealed against tree `0c72200`:

| | claim | settled by |
|---|---|---|
| `dc10_does_not_close_the_hannibal_gap` | a DC-10 on Mehta's field lands in **[1.563, 2.114] g** and does **not** reach 2.70 | any published DC-10 cruise derivative set |
| `mil_f_8785c_sigma_w_exceeds_the_mehta_ceiling` | the severe curve at 37,000 ft gives **σ_w > 4.46 m/s** | MIL-F-8785C Fig. 7, digitised |

**The first one bets against this document.** §5 now names aircraft type as the
surviving explanation for the 32% and a DC-10 set as the highest-value acquisition.
The prediction says that acquisition will *not* close the gap, resting on §4's
measured null that quadrupling mass moved the load 0.6% because wing loading
cancels in `Δn = ΔC_L/C_L,trim`. One of the two is wrong. That is the point.

**What the seal is.** The seal is the git history: each entry records the commit
its author could see and a hand-written digest of its own claim, and
`test_predictions.py` recomputes it. An accidental edit fails the build; a
deliberate one updates the digest too and shows up as a diff on a line whose only
job is to be stable. It makes tampering **visible, not impossible**, and the module
says so rather than overselling it. A first draft computed the digest at import
from the same fields it hashes, which can never fail — that mistake is recorded in
the module and has its own negative-control test.

#### The lateral dimension, and what it was worth (session 24)

`scripts/lateral.py`, figure `08-lateral.png`. Phase 1 of the plan, and the first
result in this project that is not longitudinal.

##### The two vortex forms, reconciled

`wind.line_vortex_wind` is a second implementation of a field the project already
had, so it earns its place only by reproducing the first where the first is
defined — on the flight path:

| Geometry | vertical | horizontal magnitude | full vector |
|---|---|---|---|
| perpendicular, Δψ = 0 | 7.1e-15 | 5.3e-15 | **7.1e-15** |
| oblique, Δψ = 31° | 1.2e-14 | 7.1e-15 | **7.08** |

**The oblique row is the finding.** The vertical component and the horizontal
*magnitude* agree to machine precision; only the horizontal *direction* differs.
At north = 600 m:

```
vortex_wind      NED = [ -2.1213  +0.0000  -0.4843]
line_vortex_wind NED = [ -1.8183  +1.0926  -0.4843]
north ratio 0.8572 = cos 31°;  east/(−north) 0.5150 = sin 31°
```

Parks' model is two-dimensional in the plane **perpendicular to the vortex
lines**. `vortex_wind` returns his horizontal magnitude along the **flight
path**, exact only at Δψ = 0. At 31° the true perturbation is that magnitude
rotated by Δψ, and **the 0.515 of it that lies across the path — the only
sideslip input this field has ever had — was being discarded.** §5 carries it.

##### Flown: what the missing dimension was worth

747 through Mehta's own five-vortex field at his own 37,000 ft, fixed controls,
dt 0.01.

| Run | \|φ\| | \|β\| | \|p\| | \|p_gust\| | `n_z` max |
|---|---|---|---|---|---|
| point (`vortex_wind`) | **0.000°** | **0.000°** | **0.0000** | **0.0000** | 1.6019 |
| line (`line_vortex_wind`) | 12.508° | 3.106° | 0.1635 | 0.0879 | 1.6375 |
| line + **strip loads** | **15.376°** | 3.533° | 0.2008 | 0.0880 | 1.6344 |

**Two firsts.**

1. **A wind field rolls the aircraft** — 12.5° of bank, 15.4° with strip loads.
   The point model gives **exactly 0.000**, not something small: the equations
   have no `y` in them.
2. **`loads.strip_increment` moves a reported number**, +22.9% on peak bank.
   Built in session 14, it had changed every result by *exactly* 0.000000
   because no field varied across the span.

**And the longitudinal answer barely moves** — the up-increment goes 0.6019 →
0.6375 g (+5.9%). So nothing this project has concluded was resting on the
missing dimension, which is the reassuring half of the result.

##### The two new Dryden components

`wind.dryden_field` adds `u` and `v` to the existing `w`. MIL-F-8785C is not in
the folder — the vertical form was already second-hand and these carry the same
standing — so the check that matters is **internal consistency**: for an
isotropic field the one-dimensional spectra satisfy
`Φ_transverse = ½(Φ_long − Ω dΦ_long/dΩ)`, and the implemented pair satisfies it
to **1.3e-16**, differentiated with `jax.grad` rather than a finite difference.
Both integrate to σ² over the half line; the three components realise σ to within
1.5% and cross-correlate below 0.05.

**These give sideslip, not a rolling gust.** All three components are functions
of along-track distance, so every strip still sees the same vertical gust —
asserted, so the docstring cannot quietly become false.

##### A bug this work made and caught

`cos_dpsi` predates `sin_dpsi` by a session, so `mehta_hannibal_array` carried a
cosine with the sine still at its 0.0 default — **a vortex axis of length
cos Δψ rather than 1**, which scaled the whole induced velocity and looked
exactly like physics. Found by `scripts/lateral.py` reporting a 7 m/s
disagreement in the one geometry where the two forms must be identical. Fixed
twice over: the constructor now sets both halves, and `vortex_axis` normalises,
so a mismatched pair becomes a wrong *angle* — which a test can see — instead of
a wrong *magnitude*, which does not look wrong at all.

**Nothing here is validated against a source.** No document held by this project
records a lateral CAT response. Every number above is a capability
demonstration; §5 says so and the validation claim in §1 excludes it explicitly.

### AtiSim against JSBSim through a Kelvin–Helmholtz vortex (session 21)

Same field, same starting state, same density, fixed controls. atisim run
**translation-only**, because JSBSim has no writable gust-rate input and carries no
gradient at all. Measured **peak to peak across the core window**, which is the encounter;
see the row below for why not from trim.

| Case | Airframe | JSBSim n_z span | AtiSim n_z span | Error | JSBSim θ span | AtiSim θ span | Error |
|---|---|---|---|---|---|---|---|
| Cimarron, 33,000 ft | `boeing737` | 1.1337 g | 1.1408 g | **+0.6%** | 1.1191° | 1.0940° | **−2.2%** |
| Hannibal, 37,000 ft | `boeing747_jsbsim` | 1.7077 g | 1.7443 g | **+2.1%** | 1.1289° | 1.1336° | **+0.4%** |
| Morton, 39,000 ft | `boeing747_jsbsim` | 1.3626 g | 1.4016 g | **+2.9%** | 0.9211° | 0.9703° | **+5.3%** |

**Measured from trim instead, the same runs disagree on pitch by 15–77%, and none of it is
the encounter.** The run-in is fifteen core radii — nine to eleven seconds through the
vortex's 1/r far field — and attitude accumulates over it, because atisim is started from
JSBSim's trim state and JSBSim's trim is not atisim's. At the window edge atisim is already
1.55° nose-down of JSBSim on Hannibal and 1.48° on Morton, before the core.

| Case | Run-in θ offset at the window edge | Δn error from trim | Δθ error from trim |
|---|---|---|---|
| Cimarron | 0.336° | −1.8% | −15.0% |
| Hannibal | 1.553° | −4.3% | −42.8% |
| Morton | 1.482° | −6.0% | −53.7% |

Load factor is algebraic in the state and does not accumulate; attitude is an integral and
does. That is the whole of the difference.

### What JSBSim cannot carry, measured (session 21)

`atmosphere/{p,q,r}-turb-rad_sec` are **read-only** in JSBSim 1.3.1's property catalog, and
a write to `q-turb-rad_sec` reads back `0.0` after one step. Every writable wind input is
translational and sampled at one point. The gradient's two terms pull in **opposite**
directions, so a single "with gradient" figure would hide both:

| Case | translational | + wind alphadot | + omega_gust | + both |
|---|---|---|---|---|
| Cimarron, n_z span | 1.1408 g | 1.0108 g | 1.3630 g | 1.2315 g |
| Hannibal, n_z span | 1.7443 g | 1.6923 g | 2.0197 g | 1.9663 g |
| Morton, n_z span | 1.4016 g | 1.3696 g | 1.5693 g | 1.5377 g |

Net, both terms on: **+7.9%** (Cimarron), **+12.7%** (Hannibal), **+9.7%** (Morton) of load span. Translational injection itself
is verified, not assumed: `wind-down-fps = -50` at M 0.78 / 30,000 ft moves α **3.687°**
against a predicted `atan(50/776) = 3.69°`.

### Why the two large errors are large (session 22)

Two numbers in the comparison look bad beside a core response the engines agree on to a few
percent. They are different kinds of error and `scripts/vortex_diagnose.py` separates them.

**The 15–54% pitch error is the datum, not the physics.** See §8. Measured, not argued:
still-air drift accounts for 101–103% of it, and starting in-trim removes it entirely.

> **Session 23d qualifies the paragraph below, and the qualification matters.** "Shared by
> both engines" was measured on the **single Parks cores**, whose window is 1.3 s — a fifth
> of a short period — and then carried over to the Mehta array, which is 34 s and five short
> periods. Flown, the array does not behave like the core: the two engines pick **different
> cores** as the worst one, JSBSim reaches 74.6% of the record and AtiSim 67.1%. The
> conclusion survives in the form that matters — *neither* engine reaches, so the shortfall
> is still not a solver defect — but "by the same amount" was a property of the short
> window, not a general result. §4 has the array comparison.

**The 30–43% DFDR shortfall is shared by both engines**, so by construction it cannot be a
solver difference — it is in the inputs they both received. Cimarron, against the recorded
−1.20 g:

| Lever changed | Δn⁻ | Fraction of the DFDR value |
|---|---|---|
| Nothing (as flown) | −0.841 g | 70% |
| C_Lα × 1.20 | −0.913 g | 76% |
| C_Lα × 1.60 (Prandtl–Glauert at M 0.78) | −1.022 g | 85% |
| Core strength 50 → 60 ft/s | −1.015 g | 85% |
| Core strength 50 → 70 ft/s | −1.192 g | **99%** |
| 737 → 747, four times the mass | −0.837 g | 70% — **no effect** |

The quasi-steady load, before the aircraft responds at all, is **+0.725 g** — essentially
the DFDR's +0.73. The simulated aircraft sheds 42% of that by pitching away and climbing
during the traverse.

Two findings there. **Gust strength dominates**: the fitted Rankine core is smoother than
the data Fig. 4 itself overlays on it, and a 40% stronger core closes the gap almost
exactly. **Airframe mass does not matter at all** — a clean null. The intuition that a
heavier aircraft holds more of the quasi-steady load is wrong, because Δn = ΔC_L/C_L,trim.

> ### ⚠ That row's *explanation* was wrong, and session 26 measured the right one
>
> It read "…and C_L,trim = W/qS, **so wing loading cancels**; four times the mass moves the
> answer by 0.6%". **Wing loading does not cancel — it is the only thing that matters.**
> `C_L,trim = (W/S)/q`, so `Δn = C_Lα·(w_g/V)·q/(W/S)`, and the load goes as **1/(W/S)**.
>
> **The measurement never tested it.** The row is *737 → 747*, and those two entries differ
> by **5.95× in mass but only 1.267× in wing loading** — mass and area moved together, which
> is what an aircraft family does. A controlled sweep, one airframe at one condition
> (`boeing747`, Mehta's field, first core):
>
> | change | W/S | peak-to-peak Δn | vs baseline | `1/(W/S)` would give |
> |---|---|---|---|---|
> | mass ×0.8 | 4434 | 2.2553 g | ×1.192 | ×1.250 |
> | **baseline** | **5542** | **1.8918 g** | — | — |
> | mass ×1.25 | 6928 | 1.5742 g | ×0.832 | ×0.800 |
> | area ×1.25 | 4434 | 2.2543 g | ×1.192 | ×1.250 |
> | **mass ×1.25 *and* area ×1.25** | **5542** | **1.8895 g** | **×0.999** | ×1.000 |
>
> The last row is the real null: **change the mass by a quarter and the area with it, and the
> load does not move — 0.1%.** And `mass ×0.8` and `area ×1.25` land on the same wing loading
> and the same load to four figures, from opposite directions. The scaling is slightly
> **sub**-inverse (×1.192 where 1/(W/S) says ×1.250) because an aircraft that responds more
> also pitches away more, which is §4's own shedding effect.
>
> **So the original sentence had the right null and the wrong reason**, and the reason is what
> §5 and a sealed prediction were both leaning on. See §5 for what that does to the
> aircraft-type explanation.

### Comparison preconditions (session 21)

| Check | Measured | Why it has to be checked |
|---|---|---|
| The two vortex fields, written independently in numpy and JAX | **< 1e-9 m/s** at every sample | a bug on both sides of a comparison is invisible to it |
| JSBSim `accelerations/Nz` vs atisim `dynamics.load_factor`, same still-air state | **6.6e-5 to 2.1e-4 g** | otherwise the Δn column compares two different quantities |
| JSBSim elevator over the whole encounter | **0.0000° of movement** | a moving surface would be commanding the pitch difference |
| Morton flown at both sources' radii, which agree | **identical to the last digit** | establishes Hannibal's spread as the radius, not the harness |
| Density match, atisim geometric vs JSBSim geopotential | residual **0.0**, shift −69.19 ft | qbar ∝ ρ, so a nominal match biases every force one way |


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
| Fig. 8 vortex point, point vs strip | **unchanged: d(θ) 2.160°, d(n) −1.261 g** | reported |
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
| `vortex.py` unmoved by the constants move: first-core Δθ | 2.160 deg | was 2.24 |
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
| 0.02 | 2.1261° | 2.1628° | **+1.72%** |
| **0.01 (published)** | **2.1602°** | **2.1434°** | **−0.78%** |
| 0.005 | 2.1506° | 2.1593° | +0.40% |

Re-measured session 22 at Hannibal's 500 ft core; at 600 ft the same three rows read
2.2596/2.2230/−1.62%, 2.2400/2.2216/−0.82% and 2.2271/2.2179/−0.41%. **The magnitudes
are what carry the conclusion and they barely moved** — 1.72%, 0.78%, 0.40%, still
halving with dt. The signs now alternate, which is a property of the measurement rather
than of the scheme: the quantity is a max-minus-min over a discretely sampled trace, so
which sample lands nearest the peak flips with the step.

So the cost at the production step is **~0.8%, not ~1e-4 — about 80× the figure this
paragraph carried** — and it halves with dt, as an O(h) error must. Pinned by
`test_the_wind_hold_costs_the_headline_figure_more_than_E4_bounds_it`.

**No conclusion changes**, because §5 caps the vortex claims at orderings and puts ±25%
bands on the identified parameters, and 0.8% sits far inside both. What does change is
the precision claim: **Δθ = 2.160° is not good to four significant figures** — its last
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
This is the measurement that was made instead of the change: `atisim.dynamics.G0`
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

> **Partly superseded by "Model fidelity from the JSBSim comparison (session 18)" below.**
> This entry was written at commit `1dcc5d4`, before four further commits changed the model.
> Its framing, its three findings and its layer-1/2 results still hold. **Superseded rows:**
> the derivative table (moments are now referred to the AERORP, so the entry carries
> 737.xml's own constants), `Cmq` (unfolded), layer 1's lateral figures, layer 3's short
> period and phugoid, and layer 4's divergences. Superseded, not deleted, per this
> document's own rule -- the numbers below were true of the code at the time.
>
> **Also superseded by "The Ixz sign, and what layer 4 was really measuring (session 19)":**
> layer 3's `3 lateral` row and every layer 4 figure, including the per-component table and
> its attributions. The 737 was flown with the wrong `Ixz` sign throughout this entry.

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
| 4 trajectory | elevator doublet 0.410 m/s over 20 s, rudder kick 1.46 m/s — but see the correction below |

**No defect was found in AtiSim.** Every disagreement traces to a documented model
difference with a measured magnitude.

**A correction to the first reporting of layer 4.** The doublet's 0.410 m/s was recorded as being
the Earth-rotation floor "exactly". It is not, and the error was comparing two different components.
The floor is 0.403 m/s in **u**; the doublet's 0.410 is in **w**. Per component:

| case | u | v | w | max \|β\| |
|---|---|---|---|---|
| elevator doublet | 0.296 | 0.012 | **0.410** | 0.003° |
| rudder kick | **1.462** | 1.234 | 0.522 | 2.909° |
| Coriolis floor (lat 0 vs 47) | 0.403 | 0.012 | 0.067 | — |

So the doublet's **u** divergence is *below* the floor, and its **w** divergence is six times the
floor in that component — a real residual needing its own explanation, which is the α̇ fold: `Cmq`
carries `Cmq + Cmα̇`, exact only when α̇ = q, and the recorded doublet reaches
\|α̇ − q\| = 0.0122 rad/s, worth \|ΔCm\| up to 0.0015 against 0.0186 for one degree of α.

The two cases differ by 3.6× because they excite different physics, not because one is worse: the
doublet is essentially sideslip-free so every lateral difference is inert, while the kick's **v**
divergence is a transient Dutch-roll phase difference and its **u** divergence is secular sideslip
drag.

**Three findings worth keeping.**

*JSBSim's `do_linearization` is closed-loop and nothing says so.* Its FCS is inside the
exported model. Against the bare airframe the lateral comparison reads as catastrophic —
Dutch roll ζ 0.101 against 0.344, spiral **127.6 s against 16.7 s, a 664% disagreement** —
and the natural conclusion is that this project's lateral dynamics are broken. The yaw
damper feeds yaw rate to the rudder with unit gain above M 0.11 geared by 0.35 rad, adding
`ΔCnr = Cndr × 0.35 × 2V/b = −1.147` against a bare Cnr of −0.350. Folding it in gives
ζ 0.338 and spiral 16.61 s. Pitch and roll have no feedback, which is why the short period
needed no correction at all.

*AtiSim's ISA uses geometric altitude where the standard uses geopotential.* Density runs
0.159% low at 30,000 ft and 0.368% at 40,000 — a same-signed bias on every force in every
layer, since q̄ ∝ ρ. Predicted temperature errors match measured ones to four decimal places.
Neutralised for the comparison by matching on **density rather than altitude** (43.22 ft
lower, agreeing to 1e-16); the underlying defect is pre-existing and filed rather than fixed
here. **Anything altitude-dependent in this ledger carries it.**

*Scripts run from a git worktree import the wrong tree.* `atisim` is installed editable
against the main checkout, so `python scripts/foo.py` from a worktree silently runs the other
tree's code — no error, wrong answers. pytest is immune because it puts its rootdir first,
which is precisely why a green suite did not catch it.

The thrust model gained a **Mach ram term** (`+12.1%` from M 0 to M 0.8 on the CFM56, an 11%
error at cruise that would read as a drag defect). The field defaults to neutral, so the 747,
747-approach, Cherokee, Cessna and the synthetic fixture are **bit-for-bit unchanged** —
asserted, not inspected.

Report: `docs/summary/jsbsim-737-report.pdf`.

### Model fidelity from the JSBSim comparison (session 18)

Five model changes arising from session 17, and one sequence worth reading as a whole.
Design: `docs/superpowers/specs/2026-08-20-model-fidelity-improvements-design.md`.

| Change | What it is |
|---|---|
| `aero_ref` | Body-axis CG→AERORP vector; `moment += r × F` in `aero_forces_moments`. Zero for every other aircraft, so `jnp.cross` adds an exact zero |
| `CD_beta` | Sideslip drag, **quadratic**. JSBSim's own table is linear-interpolated through zero, which makes CD ∝ \|β\| at the origin — a coarse-table artifact no symmetric airframe can produce, and the one place this work declines to follow JSBSim |
| `CD_alpha` | Profile-drag slope with incidence, linear because the reference sits away from drag's minimum in α — where β sits **at** it |
| `CLadot`/`Cmadot` | α̇ derivatives, Stengel Eq. (3.4-25)/(3.4-26). `dynamics.derivatives` now resolves the aircraft's own α̇ as well as the wind's, so the 737 carries a **bare** `Cmq` = −27.0 with `Cmadot` = −16.0 separate |
| second condition | `boeing737_approach`, recovered at 5,000 ft / M 0.40, α 3.63° against cruise's 1.97° |

**Referring moments to the AERORP lets the entry carry 737.xml's own constants.** Recovered
about the AERORP the finite difference lands on the file: `Cma` −0.599999 against −0.600,
`Clb` −0.0899998 against −0.090, `Cnb` +0.2599999 against +0.260, `Cmde` −0.849000 against
−0.849. `Cm0` comes out at −3.0e−08 against **no such term in the file** — what read as a
pitching-moment offset was entirely the AERORP arm. Referred to the CG these were −1.1309,
−0.1440, +0.2730 and depended on the fuel state.

The pitch axis needs a **least-squares fit over a crossed design**, not central differences:
setting α away from trim also sets α̇ (`dα̇/dα` = −0.529 /s), which contaminates a differenced
`Cma` by +0.067. The fit separates `Cm0`, `Cma`, `Cmq`, `Cmadot`, `Cmde` with max residual
2e-11. Their **sum** is exact and their **split** is conditioned at 1.9e8, so the split is the
softer number.

Measured after all five changes (this tree, both conditions):

| Layer | cruise | approach |
|---|---|---|
| 1 `CL` / `CY` | 5.5e-9 / 1.6e-14 | 9.6e-9 / 1.4e-14 |
| 1 `Cl` / `Cn` | **1.1e-8 / 1.0e-8** (was 7.0e-6 / 1.7e-6) | 1.4e-8 / 1.0e-8 |
| 1 `CD` / `Cm` | 8.0e-3 / 5.2e-3 raw — **predicted** from 737.xml's tables, not defects | 8.0e-3 / 9.5e-3 |
| 2 trim | α 1.9807°, δe −0.053362 rad, throttle 0.77950 | — |
| 3 short period | **ωn 0.04%, ζ 0.03%** | 0.08% / 0.08% |
| 3 phugoid | ωn 6.58%, ζ 3.44% | 3.4% |
| 4 doublet | 0.5584 m/s (u 0.507, v 0.012, w 0.558) | — |
| 4 rudder kick | 1.5663 m/s (u 1.566, v 1.245, w 0.558) | — |

**The short period went 0.04% → 3.95% → 1.30% → 0.04%, and only the last is honest.** The
first came from an α̇-contaminated `Cma` of −1.0637 standing in for a coupling the model did
not have — two errors cancelling. AERORP referencing fixed the coefficient and left the
missing term exposed (3.95%); resolving the aircraft's own α̇ supplied it (1.30%); `CD_alpha`
closed the rest (0.04%, now at **both** conditions). The first and last are the same number
and mean opposite things.

**This is the argument for AERORP referencing, made by measurement.** Referring moments to
the AERORP makes the pitching moment inherit the force error through `r × F` instead of
absorbing it into a fitted `Cma`. A drag slope of 0.1267 against JSBSim's 0.2113 could then
no longer hide, and fixing it moved the short period by 1.3%. The CG-referenced model would
have shown nothing — it had a coefficient free to absorb exactly that error.

**Layer 4 got worse, and that is not a contradiction.** The doublet moved 0.483 → 0.558 m/s.
Trajectory divergence is set by total drag along the path; the slope at one point and the
integral over the path are independently adjustable, so improving the slope does not have to
improve the integral. Both cases remain inside their derived tolerances (0.6 and 2.0 m/s).

**Still open, and named rather than absorbed:** the phugoid at 6.58% — attributed here to "a
slow drag-and-thrust energy exchange against a thrust model still linear in throttle", which
is **half right and is corrected in session 19**: thrust is the *damping* story and has
nothing to do with the *frequency* error, which is the larger of the two and is a missing
Mach dependence of the pitching moment; `CDde` (JSBSim's `0.059·|δe|`, frozen into `CD0` at
the trim elevator, so moving the elevator changes no drag here); Mach scheduling of `Cmde`
and `Clda`, which is why one aeroplane needs two registry entries; banked trim, where layer
2's turn case is recorded but not compared; and `wave_drag`, which the comparison does not
test at all because both engines give exactly zero at M 0.78.

### The Ixz sign, and what layer 4 was really measuring (session 19)

Two findings from an outside evaluation of the session 17–18 comparison. Both are cases of
the same thing: a number the comparison reported as a model difference that was not one.

**AtiSim's 737 was flown with the wrong `Ixz` sign, through the whole comparison.** JSBSim
reports `inertia/ixz-slugs_ft2` = +19109.13, and the generator negated it on the reasoning
that 737.xml carries `negated_crossproduct_inertia="true"`. That property is already the
**tensor element**; negating it again fed the two engines different airframes.

The design named this exact risk — "`Ixz` sign needs care, not assumption… establish which
convention the reported +19109.1 is in and **assert it**, rather than pick one" — and the
risk table recorded it as mitigated. It was not. The only guard,
`test_737_mass_and_inertia_match_the_engine`, compares AtiSim's tensor against the
reference XML's, and both descend from one line of `gen_jsbsim_reference.py`. Flipping the
sign fails **that test alone**; every physics layer stays green.

*Settled from the engine's own behaviour instead.* 737.xml defines neither `Cnp` nor `CYp`,
so roll rate makes no yaw moment and no side force — which leaves JSBSim's own
`∂ṙ/∂p` as pure inertia coupling, and makes `L_p` the same about the CG and the AERORP:

| | cruise |
|---|---|
| JSBSim `A[ṙ, p]` | **+1.180789e-02** |
| as shipped | **−1.180789e-02** |
| sign corrected | **+1.180789e-02** |
| `A[ṗ, p]`, sign-independent control | −1.227332e+00 both ways, matching JSBSim exactly |

Right in magnitude to seven digits and wrong in sign, at **both** recovery conditions. The
p column is also immune to the yaw damper, which feeds r. Now asserted in
`test_inertia_cross_product_sign_matches_the_engines_own_coupling`.

**Supersedes session 17's `3 lateral` row and session 18's lateral figures.** With the sign
right, the lateral comparison is as sharp as the longitudinal one:

| | JSBSim | as shipped | corrected |
|---|---|---|---|
| Dutch roll ωn | 2.11920 | 2.08487 (1.62%) | **2.11966 (0.02%)** |
| Dutch roll ζ | 0.34410 | 0.33803 (1.76%) | **0.34402 (0.03%)** |
| roll TC | 0.82845 | 0.80179 (3.22%) | **0.82847 (0.003%)** |
| spiral TC | 16.69112 | 16.64702 (0.26%) | 16.65300 (0.23%) |

Layer 3's lateral tolerances go from a blanket 5% to **1e-3**, with 5e-3 for the spiral —
the one mode where the two reductions are genuinely different problems. A 5% tolerance is
how a 1.6% defect survives.

**Layer 4 was partly measuring its own replay.** The reference is sampled at 0.05 s and
replayed zero-order-hold while JSBSim ran at 1/120 s. That this mattered was known — 0.25 s
sampling put the rudder kick at 5.43 m/s — but the rate was then raised until the number
looked acceptable rather than until the two parts were separated. Decimating the reference
separates them, because decimation coarsens the hold and changes nothing else. Measured
order on the most sensitive component: **1.04, 1.01** at cruise and **1.02, 1.02** at
approach — first order, as a hold must be. Asserting 1.8–2.2 instead fails all four.

Extrapolating the hold to Δt → 0 with `2·f(h) − f(2h)`:

| | u | v | w |
|---|---|---|---|
| cruise doublet | 0.507 → 0.507 | 0.012 | 0.558 → **0.172** |
| cruise kick | 1.556 → 1.561 | 0.933 → **−0.125** | 0.573 → 0.530 |
| approach doublet | 0.437 → 0.435 | 0.008 | 0.261 → **0.092** |
| approach kick | 0.818 → 0.820 | 0.345 → **−0.057** | 0.252 → 0.253 |

**Withdrawn:** the kick's v divergence read as "a transient Dutch-roll phase difference",
and the doublet's w read as the α̇ fold. The v channel and the angular rates extrapolate to
zero or past it — they are the replay, and underneath them the two engines agree on the
lateral channel to within the replay's own resolution. The kick's cruise v had also already
fallen from 1.245 to 0.933 on the `Ixz` correction alone, so what was being reported as
Dutch-roll physics was a wrong inertia term plus a sampling artifact.

**What is real.** `u`, which does not move with the interval and is still growing at
t = 20 s — the secular drag-and-thrust difference, part of it the 0.409 m/s Earth-rotation
floor, which sits in `u` too. And `w`, which peaks with the sideslip excursion (−0.573 m/s
at t = 2.66 s against a β peak of 2.9°) and decays to a third by t = 20. Divided by
airspeed, `w` is nearly the **same angle** at both conditions — doublet 0.0416° / 0.0395°,
kick 0.128° / 0.108° — across a 1.77× change in speed and 6× in altitude, which is the
signature of a coefficient-level difference rather than anything that accumulates. **Named,
not explained:** no term has been identified that predicts it.

**Layer 4 now runs at both recovery conditions**, which layers 1 and 2 already did. Its
tolerances are labelled **backstops** — allowances, not predictions — and the predictive
statements live in the two tests that do the extrapolation. Fed the cruise entry at the
approach condition the new cells read 9.7 and 10.0 m/s against 0.55 and 1.00, so they
constrain something.

**The validity guard the recovery design declined is now built.** `_boeing_737`'s docstring
says flying it at 5,000 ft and 200 kt "produces numbers that are wrong without anything
failing, warning or logging", and the design chose documentation over a runtime guard —
"Mitigation taken: documentation only, deliberately". After a documented-but-unasserted
`Ixz` convention turned out to be wrong for a whole comparison, that trade no longer holds.

`Aircraft` gains `valid_mach` and `valid_altitude`, both `[lo, hi]` and both defaulting to
`lo == hi`, meaning **no band declared** — so every CR-2144 and Nelson entry is untouched,
and asserted to be: `test_the_recovery_band_reaches_no_force` checks all six coefficients
are bit-for-bit unmoved by the fields. `checks.recovery_band` gates on them, measuring the
excursion in **band widths** so Mach and altitude are one number:

| run | verdict |
|---|---|
| `boeing737` at its recovery point | gate, **pass**, 0.000 |
| `boeing737` at 5,000 ft / 200 kt | gate, **FAIL, 2.00 band widths** |
| `boeing737_approach` at its point | gate, pass, and says the Mach axis was not checked |
| `boeing747` | **report**, not a pass — no band declared, so nothing was checked |

The bands are the entry's own fit ranges, not a judgement about where it probably still
works: `thrust_fit` samples altitude at the recovery point ±5,000 ft and Mach at 0.60–0.95.
An aircraft that declares no band gets `report`, never a green tick — the same reasoning
`Check.kind` already applies to tripwires. It is separate from `alpha_band`, which asks
about `aero.py`'s linear range and applies to every aircraft equally; a 737 at 5,000 ft and
200 kt sits comfortably inside the alpha band and is still nonsense.

**A gap the guard exposed while it was being written.** `gen_jsbsim_reference.thrust_fit`
brackets its altitude samples around the condition — the comment there records learning that
lesson — but its **Mach samples are hard-coded at 0.60–0.95 and are not rebound per
condition**. The approach entry flies at **M 0.40**, below its own ram fit, so its
`mach_ram` of 0.3346 is an extrapolation and the 0.28% residual recorded beside it describes
M 0.60–0.95 rather than the condition in use. Declaring `[0.60, 0.95]` for that entry would
condemn it at its own recovery point, and inventing a lower bound would be inventing; so its
Mach half is left undeclared and the check says so out loud. The fix is one line in the
generator plus a regeneration, which needs JSBSim installed.

**The phugoid, diagnosed. It is two different problems wearing one name.** Sessions 17 and 18
attributed the whole of it to drag and thrust. Localised by comparing the two longitudinal
plant matrices **entry by entry** rather than on eigenvalues — substituting JSBSim's value for
one entry at a time and re-solving — it splits cleanly:

*The frequency error is one entry, `M_u` = ∂q̇/∂vt.*

| substituting JSBSim's… | cruise ωn | approach ωn |
|---|---|---|
| nothing (as built) | 0.05581 — 6.58% | 0.09370 — 3.42% |
| `∂q̇/∂vt` alone | 0.05251 — **0.27%** | 0.09063 — **0.03%** |
| JSBSim | 0.05237 | 0.09060 |

Every other entry of the 4×4 agrees to within 2.7% and moves the phugoid not at all. So one
derivative accounts for 96% of the cruise error and 99% of the approach one.

*AtiSim's `M_u` is entirely an α̇ coupling, and that is measured, not argued.* At fixed α
the build-up has no Mach dependence at all except wave drag, since there is no
Prandtl–Glauert correction anywhere in `aero.py` — and wave drag is exactly zero at both
recovery points, so it contributes nothing to `M_u` either.

> **Correction to the first writing of this entry**, which said "`CL`, `CD` and `Cm` are
> bit-identical at V ± 10 m/s … `M_crit` sits at 0.898 against M 0.78". 0.8977 is
> **M_dd**, not `M_crit`: `wave_drag` subtracts `_MDD_OFFSET` = 0.10772, putting the onset at
> **M 0.78998** — 0.010 Mach above the cruise trim point, about **3 m/s**. `CL` and `Cm` are
> speed-independent as stated, but `CD` is only so *below that onset*, and V + 10 m/s at
> cruise is above it. The `M_u` conclusion is unaffected — the onset is above the
> linearisation point, so wave drag has value and slope exactly zero there — but the margin
> is thin enough to be worth a tripwire, and now has one in
> `test_the_wave_drag_onset_sits_above_the_recovery_mach`.

What produces `M_u` is dynamic: perturbing speed changes the force balance, which changes ẇ, which changes α̇, which
`Cmadot` = −16 turns into a pitching moment. Setting `Cmadot` = `CLadot` = 0 sends `M_u` to
**−4.9e-19**, machine zero. (That probe is clean because α̇ = 0 at the trim point, so zeroing
the term does not move the equilibrium it is linearised about.)

JSBSim carries the same coupling **plus** genuinely Mach-dependent aerodynamics. The two
therefore differ in sign at cruise — AtiSim +1.114e-04 against JSBSim −1.024e-04 — and by a
factor of two at approach, +4.509e-04 against +2.254e-04.

**This is structural, not a defect.** A constant-coefficient model cannot carry a Mach-tuck
term. The phugoid is the only mode slow enough for speed derivatives to dominate — the short
period is over before the speed has changed, which is exactly why it agrees to 0.04% while
the phugoid does not. The consistent-with explanation on JSBSim's side is the Mach-scheduled
`Cmde` this document already records from 737.xml; at δe_trim = −0.052 rad a modest table
slope gives this magnitude. **Confirmed in session 20 by reading the file and running the
engine** — the table is −1.20 at M 0 and −0.30 at M 2, a slope of **+0.45 per Mach**, and
δe × 0.45 accounts for **90.5%** of the `M_u` gap at cruise and **83.1%** at approach.

*The damping error is a different derivative, `X_u` = ∂v̇t/∂vt.*

| | cruise | approach |
|---|---|---|
| `X_u` error | 1.06% | **8.56%** |
| phugoid ζ error | 3.44% | **13.72%** |
| ζ with JSBSim's `X_u` substituted | — | 13.72% → **3.0%** |

At approach the culprit is the ram term, which contributes **+6.50e-04 1/s of de-damping** —
thrust rising with speed. This is the generator's hard-coded Mach fit band showing up as a
number: `mach_ram` = 0.3346 was fitted over M 0.60–0.95 and is being applied at M 0.40.

**NO MODEL CHANGE WAS MADE, AND THAT IS THE POINT.** Setting `mach_ram` = 0 moves approach ζ
from 13.72% to 7.55% and cruise ζ from 3.44% to **13.35%** — it buys one condition at the
other's expense, because at cruise the term sits inside its fit band and is doing correct
work. Deleting it would be fitting the solver to a test result. The same applies to the
frequency half: inventing a `Cm`-versus-Mach term sized to close 6.58% would be tuning, not
modelling. Both fixes have to come from the source:

- **`mach_ram`:** re-fit from JSBSim's own CFM56 tables over a band that brackets each
  condition's Mach, exactly as `thrust_fit` already brackets altitude. **Done in session
  20** — see below.
- **`Cmde` Mach schedule:** read from 737.xml's own table, not chosen. Until then the
  frequency error is *explained* rather than removed, which is the honest state.

**The diagnosis is now asserted, not just written down.** Three tests, all test-only — no
coefficient moved:

| test | what it pins |
|---|---|
| `..._phugoid_frequency_gap_is_the_pitching_moment_speed_derivative` | substituting JSBSim's `M_u` alone must cut the frequency error **tenfold**, at both conditions. Substituting `X_u` instead moves it by under 1e-4 of itself, which is how the test is known to discriminate between entries rather than restate arithmetic |
| `test_atisim_has_no_aerodynamic_speed_derivative_of_pitching_moment` | zeroing `Cmadot`/`CLadot` drives `M_u` below 1e-12, at both conditions, with a vacuity guard that the un-zeroed value is non-trivial |
| `test_the_wave_drag_onset_sits_above_the_recovery_mach` | the one Mach term AtiSim does carry stays off at both recovery points, so the α̇ attribution holds |

This moves the phugoid from a 10% allowance to a **localisation**: the frequency gap is one
named derivative, and if it ever stops being that, the first test fails rather than the
number quietly drifting inside a tolerance.

**Still open, and unchanged by this session:** layer 3 runs at cruise only, and the approach
phugoid ζ is out by **13.72%** (0.04850 against 0.05621) with no test asserting it — the
largest disagreement anywhere in the comparison, and it would fail the 10% the cruise
phugoid test applies. Session 18's approach column reads "3.4%", which is ωn alone. Also
unchanged: the reference XML still carries no `<tolerances>` block, so the design's promise
that a widened tolerance shows up as a mismatch with its recorded derivation is still only
half-built.

### The engine run live, and the thrust fit re-banded (session 20)

JSBSim was available after all — installed in the system interpreter rather than the project
venv. Everything below is from the **same build** the reference was frozen at, 1.3.1 build
1837 commit `3b25f25e`, so it is a check of the record rather than a new baseline. The
generator gained a root-directory fallback so it can run from an interpreter that has
`atisim` (and therefore JAX) while reaching `jsbsim` over `PYTHONPATH`;
`get_default_root_dir()` raises `OSError` in that configuration.

**Three claims checked. All held; one method of mine did not.**

*`Ixz`, confirmed by a second and independent route.* Session 19 took the sign from JSBSim's
exported linearisation. Finite-differencing the running engine's own
`accelerations/rdot-rad_sec2` instead gives **∂ṙ/∂p = +1.180808e-02** against the exported
matrix's +1.180789e-02 — five digits, same positive sign, so the tensor element is +19109.1
and the property must not be negated. The generator's new `assert_inertia` also fired for
real during the regeneration and passed at 1.3e-07 relative.

*`Cmde` is a Mach table, read from the file.* −1.20 at M 0, −0.30 at M 2, so **+0.45 per
Mach**; at M 0.78 it evaluates to −0.849, which is the recovered value exactly. Reading each
`aero/coefficient/*` property directly shows `Cmalpha` and `Cmq` have **zero** Mach
dependence and `d(Cmde)/dM` is δe × 0.45 to the digit at both conditions.

*A measurement of mine was contaminated, and it is the same trap the original work hit.* My
first `∂Cm/∂M` was a fixed-α Mach sweep, which gave −0.0099 at cruise and **+0.048** at
approach — disagreeing with the table in magnitude and, at approach, in sign. The
decomposition shows why: `d(Cmadot)/dM` contributes +0.0134 and +0.0824, because setting a
state off-equilibrium sets α̇ and α̇ moves with the perturbation. That is exactly why `Cma`
needed a least-squares crossed design rather than a central difference. The per-function read
is the clean measurement; the sweep is not.

**The thrust fit is re-banded, and this is a model change — the only one this session.**
`thrust_fit`'s Mach samples were hard-coded at 0.60–0.95 regardless of condition; they now
bracket `MACH` at ±0.10, exactly as the altitude samples have always bracketed `ALT_FT`. The
number comes from JSBSim's own engine table, not from anything tuned. Measured at the
approach trim throttle, thrust runs 46309 / 43591 / **40864** / 41383 / 41906 N at
M 0.20 / 0.30 / **0.40** / 0.50 / 0.60 — a bucket whose minimum sits essentially at the
condition, so the local slope is *downward* where the extrapolated fit supplied *upward*.

Regenerated against the same build, **only three lines changed in each reference file** —
every derivative, trim, A/B matrix, sweep point and trajectory sample is bit-identical:

| | cruise | approach |
|---|---|---|
| `mach_ram` | 0.2511 → 0.2456 | **+0.3346 → −0.2949** |
| `max_thrust` | 101375 → 101668 N | 82579 → 91307 N |
| `thrust_mach_residual` | 0.193% → **0.151%** | 0.283% → **2.54%** |

The approach residual got **worse**, and that is the honest number: `1 + ram·M²` is monotonic
in |M| and cannot represent a bucket at all. The fit now reports a poor fit *at the
condition* instead of a good fit *somewhere else*.

| | before | after |
|---|---|---|
| approach phugoid ζ | 13.72% | **1.53%** |
| cruise phugoid ζ | 3.44% | **3.12%** |
| approach phugoid ωn | 3.42% | 3.39% |
| cruise phugoid ωn | 6.58% | 6.58% |

**The frequencies did not move, and that is the cross-check.** The refit touched thrust only,
so if the frequency error were drag-and-thrust it would have moved too. It did not, at either
condition — which confirms the two halves are separate mechanisms rather than one error split
two ways, and leaves the frequency where the `Cmde` table says it belongs.

The trim **thrust level** did not move either (+1.12% cruise, +1.56% approach, unchanged),
and could not have: `max_thrust` is solved so the model reproduces JSBSim's thrust *at* the
condition, so the two move together and only the Mach **slope** changes. That is precisely
the quantity the phugoid damping reads.

**`boeing737_approach` now declares a Mach validity band** of 0.30–0.50, which session 19 had
to leave undeclared because there was no honest band to state. `test_a_recovery_band_is_...`
caught the change and now asserts both entries carry a band containing their own flight
condition.

**The 737 carries JSBSim's own CL(α) table now, and the ceiling in §7 is open.** Four
points, transcribed exactly:

| α (rad) | α (deg) | CL | segment slope |
|---|---|---|---|
| −0.20 | −11.46° | −0.68 | |
| 0.00 | 0° | 0.20 | **4.400** below zero incidence |
| 0.23 | **13.18°** | **1.20** | **4.3478** — this *is* `CL0 + CLa·α` |
| 0.46 | 26.36° | 0.20 | −4.3478, past the break |

Segment two is the linear model already verified: its slope matches `CLa` to **4.8e-13** and
its intercept matches `CL0` to 5.5e-09. So the table changes nothing at either recovery
point — and in fact **improves layer 1's CL by five orders**, from 5.5e-09 to **7.1e-14** at
cruise and 9.6e-09 to 1.3e-13 at approach, because `CL0` was an intercept *solved* to
reproduce lift at the reference and carried that solve's residual where the table is simply
737.xml's own 0.20. Layers 2, 3 and 4 are bit-identical. The layer-1 CL tolerance is
tightened 1e-7 → **1e-12** to hold the improvement.

Outside the segment it changes everything that matters: the linear form reported **2.5× the
source's lift at α 20°** and 6.9× at 25°. Agreement across the full table — both clamped
endpoints, the break, and the falling branch — is now better than **1e-9**, asserted in
`test_layer1_lift_matches_through_the_stall` against a new `<stall_sweep>` block in the
reference. That block is additive: the ordinary sweep is untouched, so no existing layer-1
result moved.

**Two consequences §5 and §7 record as permanent are now conditional.** The ±g asymmetry was
a property of the *linear form*, not of the airframe — the table's slope is 4.400 below zero
incidence against 4.3478 above, so an up-gust and an equal down-gust no longer give equal and
opposite increments for an aircraft carrying one. And the Cessna's stall tables are still
unused, but now for want of a caller rather than for want of a mechanism. Both remain exactly
as recorded for every entry without a table, which is all four other aircraft.

**What opening the seam actually cost**, neither of which §7 anticipated: a kink at every
breakpoint, which `jacfwd` turns into a one-sided slope if anything linearises *at* one — the
737's cruise margin is 1.98° and is now asserted — and trim ceasing to be single-valued above
the break, which `trim.trim` has no defence against since it is an unbracketed root-find.

### Looking for a 747 lift curve, and why the Mach tables are not applied (session 21)

**No source this project can reach publishes a post-stall 747 lift curve.** Checked, not
assumed:

- **CR-2144** returns zero hits for "stall", "lift curve", "CLmax" or "nonlinear" across all
  352 pages. §IX gives linear derivative tables (IX-1, IX-2) and derivative-vs-Mach figures.
  It is a handling-qualities compendium of linearised sets by construction.
- **NASA CR-114494** (Hanke & Nordwall, Boeing D6-30643, 1970 — the canonical nonlinear 747
  model, now in `refs/`) has `CL_BASIC` vs α on pp. 2.0-7 and 2.0-8 as **straight lines**,
  annotated *"extrapolate linearly to higher α_WDP if required"*. Stall is a **boundary** —
  buffet onset, stick shaker, certification stall speeds — not a falling curve. §19 revises
  ground effect and buffet-onset α, not the lift curve.
- **JSBSim's B747** was rejected: its table is the 737's first three points verbatim with a
  different tail, author "Unknown", `release="ALPHA"`, implying `CLα` = 4.3478 against
  CR-2144's 4.9441. Adopting it would replace a qualified NASA number with an unqualified one
  24% different.

**Two figures were digitised, and both are recorded here rather than applied.** The working
patch is kept out of the tree; what follows is the data and the reason.

*`CL_MAX(M)` — CR-114494 p. 2.0-38, flaps up, gear up, trimmed.* Reading uncertainty **±0.02**:
the fine grid is 0.1 CL over 63 px and the drawn line is 4–8 px. The curve had to be separated
from the major gridlines by stroke thickness — gridlines render 1–2 px, curves 3 or more —
which is what stopped the trace latching onto the CL 0.894 gridline.

| M | 0.10 | 0.30 | 0.50 | 0.70 | 0.78 | 0.82 | 0.86 | 0.90 | 0.94 |
|---|---|---|---|---|---|---|---|---|---|
| CL_max | 1.097 | 1.053 | 1.008 | 0.952 | 0.910 | 0.878 | 0.834 | 0.763 | 0.678 |

*`CLα(M)` — CR-2144 Figure IX-5, 40,000 ft curve.* **The cross-check you would want:** read on
its own the figure gives **4.892 at M 0.80** against Table IX-4's **4.9441** — agreeing to
**1.05%**, which is the measured reading uncertainty of the whole digitisation. Level from the
table, shape from the figure. Uncertainty ±0.05 in `CLα`, about ±1%.

**WHY NEITHER IS APPLIED.** Both were implemented and the full suite was run. **Eleven tests
failed**, and not cosmetically:

- `test_the_six_dof_rollout_is_fourth_order`, `test_extracting_rk4_step_did_not_move_a_single_bit`,
  `test_the_rollout_is_only_first_order_through_a_spatially_varying_wind`,
  `test_a_time_varying_uniform_wind_adds_no_body_force` — **a hard `min` ceiling is a kink, and
  a kink destroys RK4's formal order** wherever a trajectory crosses it. The integrator
  verification in §4 is not compatible with a non-smooth force model.
- `test_load_factor_is_a_straight_line_in_air_relative_incidence` and
  `test_logging_the_run_did_not_move_the_headline_numbers` — the ceiling retires the linearity
  invariant the vortex analysis is built on. That is the ceiling *working*: at FC9 the aircraft
  trims at CL 0.654 against a 0.894 ceiling, and a vortex adding 10° of α asks for CL 1.52. The
  existing vortex results were in the regime §7 already warns "is not evidence of anything" —
  but bounding it changes every published figure in that analysis.

And `CLα(M)` **does not improve the 747** — measured at FC9, phugoid ωn goes 17.8% → 19.4%
(worse) and ζ 14.4% → 12.6% (better), the short period unmoved. That is the shape of a
**partial** correction: CR-2144's own FC9 model carries `Xu`, `Zu` and `Mu`, this project omits
all three by form, and a Mach-dependent `CLα` is an indirect stand-in for `Zu` alone.

**What shipping it properly needs**, and it is a planned piece of work rather than a coefficient
addition: a **smooth** saturation in place of `min` so the integrator keeps its order, a
re-verification that it does, and a re-baselining of the vortex analysis against a model that
now has a ceiling. **And the higher-value fix first:** `Xu`, `Zu`, `Mu` are exact tabulated
numbers already in Table IX-4, which `_boeing_747` skips with the note "the speed and alpha-dot
derivatives are outside this model's form". No digitisation, no reading uncertainty.

### Response spectra and load exceedance — session 25 (phase 2)

**The statistic changed, not the run.** Every row above this one compares a **peak** from a
single encounter. That is one realisation of a random process and its error bar does not
shrink with effort. `atisim/response.py` adds the two statistics that do, and neither
existed anywhere in the tree before: there was no FFT, periodogram or PSD of any
**response** (`wind.dryden_spectrum` is an *input* spectrum), and no exceedance count of
anything. `scripts/cat_spectra.py --seeds 32`, figure `09-spectra.png`.

**1. The Mehta run follows the airframe, not the forcing.** Mehta's five-core array is not
uniform: its four core spacings pass the aircraft at **0.0909, 0.1224, 0.1354 and
0.1886 Hz** at once, mean 0.1343. The 747 at that condition has one short period,
**0.16404 Hz** (ω_n = 1.0307 rad/s, ζ = 0.36455).

| | Hz | where the response peak sits relative to it |
|---|---|---|
| response peak, `n_z` PSD | **0.16885** | — |
| the airframe's short period | 0.16404 | **+2.9% above it**, 0.23 of a bin |
| the array's mean core passage | 0.13433 | +25.7% above it, 1.64 bins |

The record is 47.4 s, so a bin is 0.0211 Hz and the two candidates are only 1.4 bins apart
— which is why this alone is suggestive rather than decisive, and why the next row uses a
stationary field and an ensemble. **This is the first statement in this project about
which frequency the load followed, rather than how large it got.**

**2. Yoshimura's protocol, run.** Yoshimura et al. 2023 (GRL 50, e2022GL101286) validate a
CAT simulation by comparing the frequency spectrum of vertical acceleration against flight
records and checking the peak lands near the aircraft's own natural frequency — 0.14 Hz for
their B787, from 151 virtual flights of 100 s. Here: **32 flights × 100 s** through
`wind.dryden_field`, first 20 s of each discarded (declared), σ at the two ends of the
sourced range.

| σ_w, m/s | ensemble peak, Hz | vs short period | half-power band, Hz | `n_z` rms, g | worst condition drift over the record |
|---|---|---|---|---|---|
| 2.108 (`mehta_unmodelled_wind`) | 0.1400 | −14.7% | 0.1200–0.2000 | 0.0823 | 300 m altitude, 12.9 m/s (5.4% of V), \|α\| range 3.98° |
| 4.459 (`mehta_residual_ceiling`) | 0.1700 | +3.6% | 0.1200–0.2400 | 0.1836 | **703 m** altitude, **30.9 m/s (13.1% of V)**, \|α\| range **8.38°** |

**And TM-102186 states this mechanism in words, which the spectrum now measures.** Its
p. 3-4 discussion of Fig. 8 says the pitch and load variations "are dependent upon the
relationship between the time span of the vortex traverse and the aircraft's short
oscillatory period", and explains its three-aircraft ordering by exactly that relationship.
That is the resonance argument, asserted by the source and never tested here because the
project had no frequency-domain statistic. It now has one, and the headline run's peak
lands on the short period.

**The robust form of this result is not the peak location.** An averaged periodogram at
N = 32 is still noisy and the peak moved a bin and a half between the two intensities.
*(**Session 27 re-ran this at N = 151**, Yoshimura's own count: the movement was
under-sampling and both intensities land on 0.1700 Hz. The paragraph below is still the
right way to read the result, but the peak location turns out to be steadier than this
sentence allows — see "The Dryden ensemble at Yoshimura's own sample count" above.)* The
statement that does not depend on N is a ratio of two numbers: the Dryden **input** has
strictly *more* energy at 0.05 Hz than at the short period (it is flat below Ω = 1/L_w and
falls as Ω⁻² above), and the **response** has more than **3×** *less*. That reversal is the
whole content of "the airframe organises the load", and `test_cat_spectra.py` asserts it in
that form.

**3. The two σ limbs are one test, and what they measure is the drift.**
`dryden_field(σ, seed)` takes its phases from the seed alone and its amplitudes scale
exactly as σ, so seed *k* at the two intensities is the same field scaled — a linear
aircraft would return the same normalised spectrum from both. Measured: σ ratio 2.1156,
`n_z` rms ratio **2.2312**, i.e. **+5.46% against exact linearity**, with the normalised
spectra differing by 0.129 in L1.

**That superlinearity is not aerodynamic and the run was nearly quoted as if it were.**
`aero.py` is linear in α and `boeing747` carries no `CL` table, so the aero *cannot*
produce it. The drift column above is the candidate, and it is large: with fixed controls
there is nobody flying the aeroplane, so it leaves the condition it was trimmed for —
703 m of altitude and **13.1% of airspeed** at the upper σ, which moves dynamic pressure
by about a quarter. **The upper-σ ensemble is therefore not a spectrum of one flight
condition**, and its |α| range of 8.38° is inside §1's 10° envelope with little to spare.
Read the lower-σ ensemble as the clean one and the upper as a bound. Registered as
`ASSUMPTIONS.md` **E11**.

**4. The first exceedance curve.** Upcrossings per second of each load level over the same
ensemble, both signs, N in the denominator — 32 flights, **3,199 s of record** at each σ.

| Δn level, g | σ = 2.108 m/s, up / down per s | σ = 4.459 m/s, up / down per s |
|---|---|---|
| 0.10 | 0.551 / 0.555 | 0.937 / 0.922 |
| 0.20 | 0.049 / 0.060 | 0.617 / 0.590 |
| 0.30 | 0.0025 / 0.0025 | 0.312 / 0.266 |
| peak \|Δn\| reached | 0.346 | 0.825 |

Up and down agree to **0.973–1.175** at the upper σ across every level both populate, which
is the sampling error on the exact odd symmetry §5 records as structural. **At the lower σ
the same ratio reads 0.500–1.012, and that 0.500 is one event against two** in a tail bin —
which is the whole argument for the statistic: a rate has an N, so its error bar is visible
and shrinks, where a peak's is neither.

**What this does NOT establish, and the script says so before the figure.** It is not yet a
*comparison*. No published exceedance curve is held — TM-102186 gives Hannibal's load as a
two-number band and Wingrove & Bach 1994 Table 2 gives twelve single incidents, neither of
which is a rate — and no digitised acceleration history exists here, so the observed half of
Yoshimura's protocol cannot be run. Both are named in §5 as acquisitions rather than quietly
dropped.

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

  **Session 23 measured what it costs on the headline run, and it is small.** Swapping the
  tangent gust rates for rates fitted across the airframe moves the load increment by
  **−1.67% up and +4.40% down**; strip-integrated loads move it by **exactly zero**,
  because `loads.strip_increment` is roll-only and `vortex_wind` has no `y` dependence for
  it to see. Step-size refinement from dt 0.02 to 0.0025 moves the peak by 0.14% total.
  **So the model's own approximations bound at about 4.4% of the 32% by which the run
  under-reaches the DC-10's recorded load** — the shortfall is elsewhere. §4 has the table.
  Note which channel moves: the **down** excursion, not the peak.

  **Session 23c closed the remaining wind-side candidates, and the shortfall is now
  cornered on the aircraft.** Propagating the sourced `V₀` uncertainty with a declared
  `r₀` band, the most favourable corner reaches **72.7%** of the recorded peak-to-peak;
  and the peak load turns out to be **saturated** — tripling `V₀` moves the up-increment
  from 0.441 to 0.526 g against a recorded 0.70, elasticity under 0.15 and changing sign,
  with peak |α| still near 8°. The peak first reaches +1.7 g between **×3.25 and ×3.50**
  of the identified `V₀`, and |α| leaves the 10° linear range in the **same interval** —
  so **there is no gust strength at which this model both reaches the record and may be
  believed.** That excludes amplitude rather than merely bounding it. §4 has both tables.

- **No wind field varied across the span, so the model was longitudinal by construction.**
  **Found session 24 while writing the validation claim above; it was in no document.**
  Every field was a function of along-track distance alone, and each for a different
  reason: `vortex_wind` has no `y` in its equations, the updraft and microburst are
  axisymmetric and penetrated on-axis, `LeeWave` is a function of north, and
  `dryden_vertical_field` is one component. Three things followed and they compounded —
  `wind.strip_roll_moment` integrated to **exactly zero** on every field, so the strip
  load path built in session 14 had never moved a reported number; the lateral modes were
  validated as eigenvalues and never once excited; and `vortex_viz.Encounter` carried no
  roll, sideslip or rate channel, so the pipeline could not have *reported* a rolling
  response if one had occurred.

  **Session 24 (phase 1) closed the capability and found a real simplification doing it.**
  Parks' model is two-dimensional in the plane **perpendicular to the vortex lines**;
  `vortex_wind` returns his horizontal magnitude along the **flight path**, which is exact
  only when the lines are square to it. At Mehta's ψ = 31° the true perturbation is that
  magnitude rotated by ψ, so **cos ψ = 0.857 of it lies along the path and sin ψ = 0.515
  across it** — and the across-path half, the only sideslip input this field has ever had,
  was being discarded. `wind.line_vortex_wind` writes the same equations as lines in space:
  it reproduces `vortex_wind` along the flight path to **1.2e-14 m/s** in both geometries
  and differs off it. §4 has the run.

  **What is now capability and what is still a gap.** The model can represent a rolling
  gust and a lateral one (`wind.dryden_field` adds the `u` and `v` components), and the
  channels exist to see them. **None of it is validated against anything** — no source
  held here supplies a recorded lateral CAT response. §3 already notes Wingrove & Bach
  1994 supplies *"no lateral data"*, which had been recorded as a limitation of the paper
  rather than as a hole in the model. Treat every lateral number as a capability
  demonstration, never as evidence.

- **Nothing in the project measured a response in the frequency domain, and the one paper
  the dossier calls a blueprint validates that way.** Until session 25 there was no FFT,
  periodogram, PSD or Welch estimate anywhere in the tree. Every comparison matched a
  **peak** from a single encounter — one realisation of a random process, whose error bar
  does not shrink however much more work is done. Yoshimura et al. 2023 compare **spectra**
  and check the peak against the aircraft's own natural frequency; that protocol could not
  be run here at all.

  **Session 25 (phase 2) closed the capability half.** `atisim/response.py` supplies the
  estimators, `scripts/cat_spectra.py` runs Yoshimura's protocol on this project's aircraft
  and its only stochastic field, and §4 has the numbers — including the result that the
  load follows the airframe's short period rather than the array's core-passage frequency,
  which is a statement about coupling that no peak could have made.

  **The comparison half is still open, and it is source-gated.** Two documents would close
  it, and neither is held:

  - a **digitised acceleration history** from a recorded CAT encounter, which is what turns
    limb B from "the model agrees with its own dynamics" into "the model agrees with a
    record". **This one is NOT an acquisition, and an earlier draft of this entry said it
    was.** `Reference_papers/19890016606.pdf` — TM-102186, held since session 23 — carries
    the Hannibal DFDR **g-load time history as Figure 6**, p. 3-5, plotted against GMT
    1:21–1:26 on a −1 to +2 g axis, beside the horizontal and vertical winds derived from
    it. Yoshimura's equivalent is *not* in the figshare set (withheld by confidentiality),
    so this held figure is the only recorded trace within reach.

    **Its difficulty is real and is a different difficulty from the project's earlier
    digitisations.** `CL_MAX(M)` and `CLα(M)` were smooth single-valued curves; this is a
    dense oscillatory trace whose severe passage occupies perhaps 15% of a five-minute axis
    on a 1989 scan, so the usable band is set by print resolution rather than by the DFDR.
    Attempt it as a bounded experiment with its reading uncertainty measured first — and if
    the band it supports does not reach the 0.164 Hz short period, say so and stop.
  - a **published load-exceedance curve** for transport-category cruise. The model's own
    curve now exists with N in its denominator; there is nothing held to overlay it with.
    TM-102186's two-number band and Wingrove & Bach's twelve incidents are not rates.

  Until then, treat the spectra as evidence about the **model's internal consistency** —
  which is real, and is what §4's rows claim — and not as agreement with the atmosphere.

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
  "Source qualification"). ~~Not quantified: doing so needs a rigid derivative set the
  project does not hold.~~

  **Session 23: bounded, and it did not need a rigid set.** Stop asking for one. For a
  rigid aircraft the non-dimensional derivatives are functions of Mach and geometry;
  altitude does not enter. So at *constant Mach* they must be identical at two altitudes,
  and any movement bounds everything altitude-dependent — including the aeroelastic term.
  Yoshimura's Table A2 supplies CR-2144's own 747 at M 0.8 / 6,096 m against
  `aircraft.boeing747`'s M 0.8 / 40,000 ft, a **2.48× change in dynamic pressure**
  (at constant Mach `q̄ = ½γM²p`, so it follows static pressure, not density — the
  density ratio is 2.16):
  `C_Zα` −16.6%, `C_mα` −60.9%, `C_mq` −16.7%, `C_Zq` −18.7%, **every one less stiff at
  the higher q̄**, which is the direction aeroelastic relief predicts and the opposite of
  a transcription error.

  **A ceiling, not a measurement of flexibility.** Reynolds number moves by the same
  factor, and CG is not excluded — Table A3 says 25% MAC and CR-2144's CG for FC9 is not
  in the material held here. A CG shift would land almost entirely on `C_mα`, the largest
  mover, so that row is the least trustworthy and should not be quoted alone. Weight *is*
  excluded: 2.888e5 kg against Table A3's 2.89e5, `I_yy` 4.488e7 against 4.49e7.
  `docs/ASSUMPTIONS.md` B1 and C3 carry the tables and the consequence.

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
  the identified r₀/V₀. ~~A 1° α error maps to 4.12 m/s of wind, 27% of a 50 ft/s peak.
  Treat the identified parameters as order-of-magnitude with roughly ±25% bands.~~

  **Session 23: the FIRST layer is now measured and the band is about a third of that.**
  Lester, Sen & Bach 1989 Table 1 propagates the uncertainties of exactly this
  reconstruction and populates them — RSS **2.449 m/s horizontal, 2.236 m/s vertical** at
  V = 250 m/s. Against Mehta's converged `V₀` = 26.46 m/s that is **8.45%**, not 25%. The
  same table shows the 1° assumption was itself twice too large: `V·δ(Θ−α) = 2.0` m/s at
  250 m/s is **0.458°**.

  **The SECOND layer is still attributed, and that is the one the wording above is really
  about.** Bach 1991 (NASA RP-1252) ch. 7 states the α reconstruction is only used where
  the recorder carried no vane, and that the L-1011 carried two at 2 Hz — it does not say
  what the Hannibal DC-10 carried. The definitive treatment is **Bach & Parks 1987,
  *J. Aircraft* 24(11) 789–792, which this project does not hold.** So: reconstruction
  error measured, identification error attributed, orderings-only rule unchanged.
  `docs/ASSUMPTIONS.md` E2 carries the table.

  **Session 23c: the band is now PROPAGATED rather than merely stated, and a third layer
  got measured.** Flying `V₀` across the 8.45% and `r₀` across a declared ±15%, the most
  favourable corner reaches 72.7% of the recorded load against a baseline 68.1% — so the
  parameter uncertainty is real but far too small to be the discrepancy. Separately,
  Mehta's published cost series bounds the **field-form** error, which is the layer
  neither Lester nor Bach speaks to: his Eq. (A3) is a *mean* square, so `J` = 214
  converts to a 4.46 m/s RMS residual without knowing `N`, and he states that n = 6, 7 do
  not improve on it. What the array cannot represent is therefore bounded at
  **2.11–4.46 m/s** of wind. §4 has both.

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

  **Session 23: still declared, but the declaration now has a measurement standing next to
  it.** Lester et al. 1989 derive a lee wave of **wavelength about 22 km** from DFDR data
  at 10 km (33,000 ft), which the same paper places about a kilometre above the tropopause
  — i.e. in the regime Doyle declined to quantify. That is **12% from the declared 25 km**
  and inside Doyle's band. Stored as `wind.LESTER_LEE_WAVE_WAVELENGTH`.

  **Not substituted, and the reason matters.** Swapping it would trade a declared number
  inside a measured band for a measured number from a different mountain range (Greenland,
  not the Sierra Nevada that `LEE_WAVE_AMPLITUDE` comes from) at a different altitude
  (10 km, not the modelled 12.192). That is not obviously an improvement. What changes is
  that the declaration is now bounded rather than unbounded.

  **The same paper also corroborates the omission below it.** It reports the flight-level
  horizontal windspeed dropping sharply at exactly the point of largest vertical motion,
  read there as a critical level from overturning waves — direct evidence that the
  horizontal perturbation `wind.LeeWave` omits is real, which turns "the reported F is a
  lower bound" from an argument into an observation.

  **Session 23 follow-up: flown, and the omission turns out to be much larger than the
  horizontal perturbation.** The 747 through a 22 km wave at Lester's own altitude
  reproduces his recorded **300 m altitude gain** at `w0` ≈ 4.0 m/s — inside Doyle's
  measured 3–6 m/s — and needs `w0` ≈ **113 m/s** to reach his recorded **+2.7/−1.0 g**.
  The two observations imply amplitudes **28× apart**, so a single smooth wave cannot
  produce both. §4 has the sweep. What `wind.LeeWave` is missing is not primarily the
  quadrature horizontal term — it is everything the wave breaks down INTO, which is what
  Lester's critical level is about and what carries the accelerations.

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

  **Session 23: the ALTITUDE failure now has an instrument that does not have it.**
  Misaka et al. 2008 §IV.B apply RMS normal load `σ_n` — a moving 5 s average — with
  moderate 0.2–0.3 g and severe ≥ 0.3 g, attributed there to Hamilton & Proctor. It is
  defined for cruise-altitude turbulence, which is where every CAT case this project holds
  actually happened. `checks.rms_normal_load` implements it as a **report**, never a gate:
  the bands say how rough the air was, not whether the run was good, and a gate would fail
  the simulator for correctly flying through severe turbulence.

  **What it cannot do, from the source's own results.** Misaka Figs. 26–27 find `σ_n`
  tracks the *trend* of measured vertical acceleration and misses the peaks, by
  construction of the 5 s window. So it grades an encounter and is the wrong instrument
  for a core penetration — the 1.5 s a 747 spends inside a Parks core is under a third of
  one window. `rms_normal_load` returns `nan` with a stated reason rather than a number
  for any run shorter than its own window.
- **The approach 747 flies below its minimum-drag speed**, by 12.2 m/s, because 1.4 Vs at
  max landing weight is on the back side of the drag curve — which is where an airliner on
  final is. The autopilot's loop pairing is therefore inverted for that entry. It holds
  trim, but no gain set repairs the pairing, so it is flown open loop for all analysis.
- **There is no ground.** No terrain, no landing gear, no ground effect, no stall. A
  microburst run therefore ends when the aircraft descends within one wingspan of the
  surface, because below that the integration is arithmetic rather than physics — left to
  itself the model bounces and climbs away, which reads as a survival and is not one.

- **Two errors in the sources themselves, found session 23. Neither is this project's, and
  both change what may be quoted from those documents.**

  **(a) TM-102186 publishes Schultz's pre-fit guesses as his DFW answer.** It reports the
  microburst rings as "outer ring 15,000 ft diameter with vortex core diameter of 3000 ft;
  inner ring 2500 ft diameter with core diameter 900 ft". Those are Schultz 1990's
  **Table 1 *initial estimates***, to the digit. His converged Table 2 reads: outer ring
  radius 7,574 ft (diameter 15,148 ft) with core **4,856.8 ft**; inner ring radius
  1,594.3 ft (diameter 3,188.6 ft) with core **985.0 ft**. The outer core is **62% larger**
  than the figure TM-102186 prints. **If microburst geometry is ever taken from
  TM-102186, take it from Schultz Table 2 instead.**

  **(b) Yoshimura 2022's conclusion contradicts its own Table A5.** The conclusion calls
  the B747 short-period frequency "1.29 Hz" and derives a 200 m resonant wavelength from
  258 ÷ 1.29, then argues 10–20 m grid spacing is needed. Table A5 labels the quantity
  `s⁻¹`, and its own Table A2 derivatives reproduce 1.29 **rad/s** = 0.205 Hz (checked:
  `ω_n² = Z_α M_q/U₀ − M_α` gives 1.286). At 258 m/s the resonant wavelength is
  2π·258/1.29 ≈ **1,257 m**, not 200 m — so the grid-resolution conclusion does not follow
  from that calculation. **Table A2/A3/A5's numbers are used here; that conclusion is
  not.**

  These are the same failure mode as the 600 ft Hannibal radius (§3): a downstream paper
  quoting a startup estimate, or a mislabelled unit, as a result. It is why §3's rule that
  every number carries the table it came from is worth its cost.

- **The Hannibal encounter is dated two ways.** Mehta 1987 says July 1981; TM-102186
  says April 1981 in three figure captions and Bach 1991's Table 7.1 lists case 1 as
  `4/81`. Everything else matches across the accounts — 37,000 ft, DC-10, ψ = 31°, a
  ~150 kt horizontal wind bias, the same ±25 kft plot range — so they are the same
  encounter. **Two NASA documents say April; cite April 1981.** Nothing physical depends
  on it.

**Structurally impossible — cannot be fixed from any source currently held:**

- **The ±g asymmetry.** Both papers attribute it to stall buffet. `aero.py` is
  `CL = CL0 + CLa·α`, exactly odd-symmetric in Δα, so an up-gust and an equal down-gust
  give equal and opposite load increments to machine precision. The only aircraft in the
  project with nonlinear data is the Cessna, which is out of scope; **CR-2144 provides no
  buffet-onset table for the 747**. Reproducing this needs a source the project does not
  have. Do not promise it.

  **Session 25 splits this in two, and only one half is still blocked.** The *nonlinear lift
  curve* is genuinely not published in anything reachable — §4's session-21 search found
  CR-114494 draws `CL_BASIC` as straight lines annotated *"extrapolate linearly to higher α
  if required"*. But the **buffet-onset BOUNDARY is held**, on a page this project has
  already read: `refs/NASA-CR-114494.pdf` p. 2.0-38 is titled *"LIFT COEFFICIENT — BUFFET
  BOUNDARY AND C_Lmax"* and carries **two** curves against Mach, flaps up, gear up, trimmed
  — *maximum demonstrated `C_L`/trimmed `C_Lmax`*, which session 21 digitised, and *initial
  buffet boundary*, which it did not. The sheet points to **§19 for revised data**, and
  p. 428 defines the boundary in α and Mach and notes that above M 0.85 buffet is
  shock growth rather than stall.

  **The boundary buys a bound, not the asymmetry.** It cannot produce a falling lift curve,
  so the ±g asymmetry stays structurally impossible. What it can do is say *where the linear
  model stops being defensible in `C_L`–Mach* — the service `panel.ALPHA_LINEAR_DEG`
  performs in α, and currently the only guard of its kind. §1's envelope would gain a
  sourced ceiling in place of a declared one.
- **Absolute agreement with the papers' g-loads.** Wingrove & Bach never identifies an
  aircraft type; Parks' two cases are DC-10s at 37–39 kft against this project's 747 at
  40 kft with roughly 0.8× the wing loading. Every load comparison is order-of-magnitude
  or clustering. Assert bands and orderings, never values.

  **Session 23c turned this from a caveat into the surviving explanation.** It used to be
  a reason not to over-claim; it is now what is left after everything else was excluded on
  the Hannibal run — numerics (0.14%), the point gust (≤4.4%), the strip path (0.00%), the
  identified parameters (≤72.7% at the favourable corner) and gust amplitude at *any*
  strength inside the linear range. **Session 23d added the last one: an independent
  engine.** JSBSim on the same five-vortex field, with the same aircraft and the same
  starting state, reaches 74.6% of the record against AtiSim's 67.1% — *neither* reaches,
  so the solver is excluded on the headline field rather than by inference from the Parks
  cores. The aircraft is the residual, and closing it needs a DC-10 derivative set, which
  is the highest-value acquisition on the list.

  **A sealed prediction now bets against this entry.** `predictions.py`'s
  `dc10_does_not_close_the_hannibal_gap` says that acquisition will *not* close the gap,
  on the strength of §4's measured null that quadrupling mass moved the load 0.6% because
  wing loading cancels in `Δn = ΔC_L/C_L,trim`. If it is right, what survives is not
  aircraft type but the missing buffet nonlinearity, and this entry is right for the wrong
  reason. Do not settle it by editing this paragraph — settle it by flying a DC-10.

  > ### ⚠ **THE SIGN, CHECKED — session 26 — AND IT POINTS AGAINST THIS ENTRY**
  >
  > The wing-loading figure was flagged as "unverified and may be inverted", with the note
  > that "nothing downstream depends on the number". **The number is still unverified. The
  > DIRECTION is now measured, and everything depends on it.**
  >
  > §4's controlled sweep gives the load as **1/(W/S)**, slightly sub-inverse. So **lower
  > wing loading means MORE gust response**. The project's own two statements of the figure
  > agree with each other — §5 said the 747 has "roughly 0.8×" and `test_wind.py` says "the
  > DC-10's wing loading is roughly 1.3× the 747's, so the 747 takes MORE g for the same
  > gust" — and 1/1.3 = 0.77. **They also both point the wrong way for this entry.**
  >
  > Flown: `boeing747` at 1.3× its wing loading on Mehta's field reaches **56.4%** of the
  > recorded 2.70 g against the baseline's **70.1%** — peak-to-peak 1.5231 g against 1.8918.
  >
  > **A DC-10 with 1.3× the 747's wing loading would therefore be worse, by 14 points, not
  > better.** If that figure survives verification, aircraft type is not the surviving
  > explanation for the shortfall — it is a term with the wrong sign, and this entry is
  > wrong in a way that no amount of DC-10 derivative data would repair. (Peak |α| reaches
  > 9.24° in that run, close to the 10° linear limit, so read it as a direction with a
  > magnitude attached rather than a prediction.)
  >
  > **The sealed prediction is the beneficiary and its stated reasoning is not.**
  > `dc10_does_not_close_the_hannibal_gap` bets the DC-10 will not close the gap, resting on
  > "wing loading cancels" — which §4 now records as false. The bet looks **more** likely to
  > land, for a reason its author did not give. The entry is SEALED and has not been touched;
  > that is what the register is for.
  >
  > **What would settle it, and how precisely.** A sourced DC-10 wing loading — not weight
  > and area separately, since Bach & Parks 1987 Eq. (2) shows `m` and `S` enter their
  > identification only as `m/S`. **And it does not need to be precise:** their Eq. (4) gives
  > about **0.05° of α per 1% of `C_L`**, so 1% of wing loading is 0.05° of α against
  > excursions of 7–9°. Anything better than a few per cent buys nothing.
  >
  > **What not to do.** Parks Fig. 7 is the vortex array and Fig. 6 is the g trace for the
  > same encounter. **Do not fit `W/S` to reproduce Fig. 6.** That converts the project's
  > only end-to-end check into a calibration, and there would then be nothing left to test.
  >
  > **SESSION 27 WENT AND GOT THE NUMBER, AND IT DOES NOT SETTLE ANYTHING — because the
  > uncertainty was never in the wing area.** `presentation_package/_evidence/dc10_wing_loading.log`.
  > The modelled 747 is 288,773.2 kg over 510.97 m² = **565.15 kg/m²** (SOURCED, CR-2144
  > Table IX-3). Against it:
  >
  > | DC-10 | at OEW | at mid weight | at MTOW |
  > |---|---|---|---|
  > | **−10** (430,000 lb / 3,550 ft²) | 0.584× | **0.815×** | 1.046× |
  > | **−30** (555,000 lb / 3,647 ft²) | 0.631× | **0.973×** | 1.315× |
  >
  > **The project's "roughly 1.3×" is the bottom-right cell — the heavier variant at maximum
  > takeoff weight.** It is the extreme corner of the range, not a central estimate. At any
  > realistic cruise weight the ratio **straddles 1.0**, and for a −10 it is **below** it,
  > which by §4's own `1/(W/S)` law means a DC-10 would respond **more** than the 747, not
  > less. **So the session-26 sign reversal is not robust, and the sentence "the surviving
  > explanation is the aircraft" is withdrawn rather than replaced.**
  >
  > **Three things block pinning it, and none is the wing area.** (1) **The variant is not
  > identified** — TM-102186 and Parks both say only "a DC-10", and −10 versus −30 moves the
  > ratio by 26%. (2) **The weight at the encounter is recorded nowhere in any source held**,
  > and OEW-to-MTOW spans a factor of 1.8. (3) Even the *published* area disagrees across
  > secondary sources for the same variant — 3,647 ft² (Wikipedia, citing *DC-10 Airplane
  > Characteristics*) against 3,958 ft² (flugzeuginfo), an 8.5% spread. The manufacturer
  > document itself was fetched this session — Douglas **DAC-67803A**, *DC/MD-10 Airplane
  > Characteristics for Airport Planning*, reprinted January 1991, revision A April 2004 —
  > and it gives **Series 10 maximum design taxi weight 458,000 lb (207,746 kg)** but
  > **carries no wing area at all**, being a pavement-and-gate document. So the best available
  > weight is now SOURCED and the area is still only tertiary.
  >
  > **What would actually settle it** is therefore not a wing loading but **the flight**: an
  > NTSB or operator record identifying the aircraft and its weight that day. Absent that,
  > quote the ratio as a **range, 0.58–1.32×**, and do not let any conclusion rest on where
  > in it the truth sits.
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

> **THE ORIGINAL TEN-STEP PLAN IS BELOW AND IS ESSENTIALLY COMPLETE.** The plan the
> project is now executing is the five-phase one written at the end of session 23d, in
> answer to "what would make this a predictive tool?". It is recorded here because it is
> §7's job to carry it; sessions 24 and 25 refer to its phase numbers throughout.

### The finishing plan — phases, gates and status

The decision it rests on: **"predictive tool" is three different projects.** **T1**
comparative and mechanistic, already true and — until session 24 — unclaimed. **T2**
bounded absolute load, months and source-gated. **T3** CAT hazard including roll, which
needs the lateral capability built from nothing. The chosen route is *bank T1 now, target
T2, build T3's foundation on the way*.

| Phase | What | Gate | Status |
|---|---|---|---|
| **0** | Bank what is already true: a formal validation claim in §1 with its envelope attached, and the lateral gap written into §5 | §1 carries the claim; §5 and `ASSUMPTIONS.md` E10 carry the gap | **DONE, session 24.** Writing the claim is what found the gap |
| **1** | Give the model a lateral dimension: Dryden `u`/`v`, lateral channels on `Encounter`, the vortex as lines in space | the strip path moves a reported number for the first time | **DONE, session 24.** +22.9% on peak bank, after nine sessions of exactly 0.000000 |
| **2** | Change what counts as agreement: response spectra instead of peaks, exceedance distributions over ensembles | a load-exceedance curve with N in its denominator becomes sayable | **DONE, session 25.** §4 has both. The comparison against a *published* curve is now the open half, and it is source-gated |
| **3** | Acquire four documents, in priority order | each arrival settles a sealed prediction or closes a §5 entry | **In progress, session 25 searched for all four** — see the table below |
| **4** | Keep the predictive discipline running: every new capability ships with a prediction made before it is tested | at least one sealed prediction settled, right or wrong | **Two of three settled, both right.** `the_dryden_response_peaks_at_the_short_period` (session 25) and `mil_f_8785c_sigma_w_exceeds_the_mehta_ceiling` (**session 27**, by 0.34 m/s with the merge band clearing it). The one still sealed is the DC-10 entry, which bets *against* the project's own story and is now the only open bet |

**Phase 3's four documents**, in the priority the plan gives them, with what each unblocks:

| # | Document | Unblocks |
|---|---|---|
| 1 | a **DC-10 cruise derivative set** | **STILL OPEN — the only genuine acquisition, and now worth twice as much.** Parks identifies *both* cases as DC-10s, so one set serves Hannibal **and** Morton. Weight Morton lower: Parks calls its fit "not as good as case 1" and blames mountain-wave contamination of the short-period pattern. **But check §5's sign first** — at 1.3× wing loading the load falls to 56.4% of the record, so if the DC-10 is the heavier-loaded aircraft this acquisition makes the shortfall worse, not better |
| 2 | **747 buffet onset / nonlinear C_L** | **SPLIT, session 25 — half of it was already held.** The buffet-onset BOUNDARY is on `refs/NASA-CR-114494.pdf` p. 2.0-38, the same sheet session 21 digitised `C_Lmax` from, with revised data in its §19: not an acquisition, a digitisation. The nonlinear lift curve is not published there or anywhere reachable, so the ±g asymmetry stays structurally impossible. **Acquire nothing; digitise the second curve** |
| 3 | **MIL-F-8785C Fig. 7**, digitised | **DOCUMENT HELD** (`refs/MIL-F-8785C.pdf`, session 25); Fig. 7 confirmed on printed p. 49. The digitisation is still to do. **Session 26 adds a caution about the check:** AFFDL-TR-70-101 (HICAT) is now held and measures the same quantity, but its band is 45,000–70,000 ft against this project's 33,000–41,000, and it compares itself against MIL-A-8861A and Steiner's NASA U-2 data — so Fig. 7's high-altitude end may descend from the same aircraft and would not be an independent check. Settling that needs ADA119421, not held |
| 3 | **Yoshimura 2023 figshare dataset (21152203)** | **HELD session 25, verified** — CC BY 4.0, `data.tar`, 17,942,056,960 B, md5 `d23cbb3c77b3940653a0b643147d71c3` matching figshare's stated checksum, in `UROP/yoshimura-figshare-21152203/` outside the repo with a `PROVENANCE.txt` beside it. Five nested bz2 archives: the four LES domains (2.3, 3.6, 4.0 and **7.5 GB** for dx = 500/250/70/**35** m) and **`flightsim-data.tar.bz2`, only 561 MB** — Yoshimura's own 2-D B787 simulation code and its outputs, i.e. the SIMULATED half of their Fig. 6. That last one is the cheap one and supports a cross-code response-spectrum comparison the way JSBSim serves the vortex work. A published LES CAT wind field: the first field in this project not identified from the aircraft's own accelerations. **It does NOT carry a recorded acceleration history** — the three onboard records and the PIREP are withheld under confidentiality — so it does not unblock the observed half of the spectral protocol, which is what this row used to claim |
| 4 | **Bach & Parks 1987**, J. Aircraft 24(11) | **HELD, session 26 — and it does NOT help #1.** Its two validation cases are an L-1011 and a B-747SP; no DC-10. What it does give is the error budget: Eq. (2) shows `m` and `S` enter only as `m/S`, and Eq. (4) gives ~0.05° of α per 1% of `C_L`, which is what sets the tolerance on #2 |
| 5 | **Parks et al. 1985**, J. Aircraft 22(2) | **HELD, session 26 — §8's open question CLOSED.** r₀ = 600 ft with V₀ 85 and spacing 3500, and an abstract giving core *diameters* 900–1200 ft that removes radius-versus-diameter entirely. Reversed the 500 ft this project flew from session 22 to 25 and deleted the hybrid; `ASSUMPTIONS.md` E12 carries the four lineages. Cost 4.26% of the headline load, downward |
| 6 | **AFFDL-TR-70-101** (HICAT), Ashburn, Waco & Melvin 1970 | **HELD, session 26.** Measured RMS gust exceedances from U-2 flights — the published exceedance data phase 2 lacked. **Band mismatch is the catch**: 45,000–70,000 ft against this project's 33,000–41,000, so any comparison is extrapolated, and it may share its U-2 lineage with Fig. 7 |

**~~Phase 3 is entirely acquisition and this project cannot do the acquiring.~~ Sessions 25
and 26 emptied most of it, and almost none of what was left was an acquisition.** Session 25
searched and found that three of the original four were digitisations of documents already on
disk. Session 26 received four more papers and closed items 3, 4 and 5 outright.

**What is left is one document and three digitisations:**

| Still needed | Kind |
|---|---|
| ~~a **DC-10 wing loading** (`m/S`, to a few per cent)~~ | **ATTEMPTED AND CLOSED AS UNPINNABLE, session 27.** Not for want of a wing area: the variant is unidentified and the encounter weight is recorded nowhere held, so the ratio is a **range 0.58–1.32×** and cannot be narrowed by another specification sheet. §5 carries the table. **What is left is not an acquisition but the flight record** — an NTSB or operator document naming the aircraft and its weight |
| a **DC-10 cruise derivative set** | acquisition, and **§5's sign no longer argues against it** — session 27 withdrew that sign rather than confirming it |
| ~~**MIL-F-8785C Fig. 7** at 33–41 kft~~ | **DONE session 27** — `scripts/digitise_mil_f_8785c_fig7.py`; σ_severe(37 kft) = 4.80 ± 0.12 m/s; settles a sealed prediction RIGHT |
| ~~**TM-102186 Fig. 6**, the recorded g trace~~ | **DONE session 27** — `scripts/digitise_tm102186_fig6.py`; and it moved two numbers, see §4 |
| ~~747 buffet onset boundary~~ | **DONE session 26** — `aircraft.buffet_cl` |
| **the Hannibal flight record** (operator, tail, weight) | **NEW, session 27.** The only thing that would pin the wing-loading ratio, and therefore the only thing that would let the aircraft-type explanation be tested rather than argued |
| ~~four B787 numbers — `m`, `S`, `c`, `I_yy`~~ | **OBTAINED session 27, and three of the four are CHECKED rather than looked up.** m = 215,910 kg, S = 325.3 m² (trapezoidal), c̄ = 6.437 m, all confirmed by inverting their own `Z_a` to a physical `C_Lα` = 4.847. `I_yy` is DECLARED and provably **unobservable** — it cancels against `C_mα`. §4 has the table. **This is now an implementation task, not an acquisition** |
| **build the `boeing787_yoshimura` registry entry, then re-fly the LES** | **NEW, session 27, and it is the top of the list.** All inputs are in hand. Give it a `valid_mach`/`valid_altitude` band around its own condition. Carry the known difference that AtiSim has no `M_α̇` where Yoshimura has −0.137 |
| **test the frozen-`C_Lα` explanation of the LES ratio** | **NEW, session 27, and it is CHEAP.** Rescale the 747's `C_Lα` by the Prandtl–Glauert ratio 1.521 and re-fly D03/D04. If the 1.42× ratio collapses toward 1, the LES discrepancy is this project's frozen derivative and **not** a code disagreement — and it becomes the **first quantified point on the Mach axis** ASSUMPTIONS C3 has left unbounded since session 12, with no chart read needed |
| ~~run the LES limb~~ | **ALREADY RUN, session 3–4 Sept, and found NOT LIKE-FOR-LIKE in session 27.** All four domains × two aircraft are on disk. §4's input audit says why no number from them is quoted: the aeroplane is 5.4% or 82.9% away in natural frequency, and the entry closest in frequency is 2.63 band widths outside its own envelope. The **field reader is sound** (+0.978/−0.968/−0.935 against their own sampled wind) and reusable; the **load comparison is not yet a comparison.** This remains the project's only route out of the circularity every load row carries |

### The original ten-step plan

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
6. [DONE] Fig. 8 with ensemble error bars                  -> verify: DONE, 16/16 members at
   (session 23d; sigma_w is the SOURCED range from                  both ends of the sourced
    Mehta's residual, not a picked number)                          sigma_w range. The margin is
                                                                    the finding: the PITCH gap
                                                                    collapses 0.886 -> 0.060 deg
                                                                    and only the LOAD axis still
                                                                    separates the categories
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

~~Step 5 is done, so **step 6 now waits only on step 4** (Dryden), which is where the
ensemble spread would come from — the three deterministic points have no spread by
construction, since every member of a batch meets the same field.~~ **Step 4 landed in
session 23b and step 6 in 23d.** The spread came from `wind.dryden_vertical_field` at the
σ_w range session 23c derived from Mehta's own fit residual, so the ensemble's intensity is
sourced rather than chosen. §4 has the result.

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

> **The seam this section said was not taken has now been opened — session 20.**
> `aero.coefficients` interpolates a `CL(α)` table when the aircraft carries one, and the
> two 737 entries carry 737.xml's own. The three consequences below still hold **for every
> entry without a table**, which is all four light and heavy aircraft; they no longer hold
> unconditionally, and the first of them was never a property of the airframe.
>
> Nothing else about this section changed: the exit named at the bottom is the one that was
> used, and the traps it did not mention are recorded with it.

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
protocol with a linear and a table implementation. ~~That was the option not taken.~~
**Taken in session 20**, and it cost less than this paragraph implies: two `Aircraft` fields
defaulting to empty, one `jnp.interp` behind a shape test that resolves at trace time, and
no change to any other aircraft. What it did cost was two things this paragraph did not
mention:

- **A kink at every breakpoint.** `validation.longitudinal_matrix` takes `jacfwd` of the real
  dynamics, so a linearisation *at* a knot returns a one-sided slope and the modes become an
  artifact of knot placement. The 737's knots sit at α = 0.00 and 0.23 rad and it linearises
  at 0.0346 and 0.0631 — inside a segment both times, asserted rather than assumed in
  `test_the_lift_table_is_not_linearised_at_a_breakpoint`. The cruise margin is only 1.98°.
- **Trim stops being single-valued** above the break, since two incidences give the same
  lift. Both entries trim far below it, but `trim.trim` is an unbracketed root-find and has
  no defence if one ever does not.

Linear interpolation is deliberate and is *faithful* rather than lazy: JSBSim's own `<table>`
blocks are linearly interpolated and clamp at their endpoints, so `jnp.interp` reproduces the
source exactly. A smoother interpolant would agree with the source less.

## 8. Open questions

- ~~**Hannibal's core radius: 500 ft or 600 ft?**~~ **CLOSED, session 26 — 600 ft, on the
  paper that identified it.** Parks et al. 1985 is now held: J. Aircraft **22**(2) 124–129,
  DOI 10.2514/3.45095, p. 127 — r₀ = 600 ft, V₀ = 85 ft/s, spacing 3500 ft, and an abstract
  giving core **diameters** of 900–1200 ft that removes the radius-versus-diameter question
  entirely.

  **Session 23 answered this the other way and the reasoning is worth keeping**, because it
  was sound on what it had. It read Mehta 1987's converged 500.5 ft and TM-102186's "1,000 ft
  diameter" as two independent confirmations of Fig. 4, and Parks as an unretrievable
  transcription. Two of those three premises survive: Mehta's refit is real and better
  converged, and TM-102186 does report it. The third does not — Parks is retrievable, says
  600 ft, and the "three sources now agree" was really **one later fit reported three times**.

  **What is left is not a resolved question but a carried disagreement**, and it now lives in
  `ASSUMPTIONS.md` **E12** with all four lineages and what the spread is worth. The project
  flies Parks' triple in `PARKS_CASES` and Mehta's five-core field in `MEHTA_HANNIBAL_*`, and
  never crosses them. The hybrid that existed from session 22 to 25 — Fig. 4's radius with
  Parks' strength and spacing — is gone.

  **The correction cost 4.26% of the headline peak-to-peak load, downward**, taking JSBSim's
  Hannibal from 70.3% to 67.3% of the recorded 2.70 g. `dw/dx = V₀/r₀` inside a solid-body
  core, so a larger core at fixed V₀ is a gentler gradient. It makes §5's shortfall worse,
  which is the direction that argues it was not chosen for convenience.


- **New, session 23d: why do the two engines choose different cores on the array?**
  On a single Parks core AtiSim and JSBSim put their load extremes within two metres of
  each other. On Mehta's five-core array they land on different cores — AtiSim's peak is
  2.2% *higher* than JSBSim's, its trough 3.5× shallower, and the whole −10.1% span error
  is the trough. The obvious cause was tested and **failed**: shrinking the window until
  the airspeed drift is −0.19% leaves the disagreement at −7.6%, and AtiSim's span does not
  move at any window width.

  What is different about the array is its **length** — 34 s and five short periods, against
  1.3 s and a fifth of one. Damping, thrust/drag bookkeeping over 45 s, and accumulated
  trajectory divergence are all candidates and none is established. The cheapest
  discriminator is probably a **still-air run of the same duration from the same state**:
  whatever the two engines do to each other over 45 s with no wind at all is the part that
  is not the encounter. `scripts/vortex_diagnose.py` did exactly this for session 22's pitch
  offset and settled it in one run.

- **New, session 23c: what are the units of Mehta's cost `J`, and what was `N`?** The
  decomposition in §4 rests on `e` being a difference of winds in **ft/s** for both
  components. That is the natural reading — `e` comes from his Eq. (4), which is
  homogeneous in `V₀`, and `V₀` is quoted in ft/s throughout — but he never labels `J`,
  and his own Fig. 5 plots the *horizontal* wind in knots while the vertical is in ft/s.

  **What survives either way, and what does not.** `mehta_residual_ceiling` is immune:
  `B` is the identity, so both terms are non-negative and either one alone is bounded by
  `J` whatever the other's scale. The 4.46 m/s ceiling therefore stands unconditionally.
  `mehta_unmodelled_wind`'s 2.11 m/s does **not** — it splits `J` evenly between the two
  components, which a mixed-unit `e` would break. Both functions say so in their
  docstrings.

  **`N` is separately unrecoverable and would sharpen nothing structural.** It is not
  needed for any number quoted — that is the whole point of the `1/N` in Eq. (A3) — but it
  would say how many degrees of freedom the fit had, which is the missing ingredient for
  turning the residual into a confidence interval on `V₀` and `r₀` rather than the
  conservative per-sample ceiling §4 uses. TM-102186 Table II gives DFDR sample rates by
  parameter and aircraft, but its text layer OCRs to unaligned columns and the wind
  reconstruction's own output rate is not stated anywhere held here.

- **New, session 23: what does `PARKS_CASES["hannibal"]` being a hybrid cost?** It pairs
  Fig. 4's radius (500 ft) with Parks' strength (85 ft/s); Mehta and TM-102186 both pair
  500 ft with 87. The field is exactly linear in V₀, so every wind-derived quantity moves
  by 2.1% — measured, not argued. It is left alone because §4 baselines sit downstream.
  **The question is whether a frozen baseline should be re-pinned to a coherent
  single-source pair**, which is a decision about what §4 is for and not one to take
  incidentally.
- ~~**Why does atisim accumulate more nose-down attitude than JSBSim through the run-in?**~~
  **ANSWERED, session 22 — it is the shared-start compromise, not the physics.**
  `scripts/vortex_diagnose.py` settles it two ways. Flown from JSBSim's state in
  **still air**, with the vortex switched off entirely, atisim drifts −0.340°, −1.583°
  and −1.502° over the same run-in — which is **101%, 103% and 101%** of the
  atisim-minus-JSBSim offset measured at the window edge with the vortex on. The vortex
  contributes nothing to it. And started from **atisim's own trim** instead, the from-trim
  pitch error collapses from −15.0% / −42.8% / −53.7% to **+0% / +1% / +1%**.
  So the whole of the apparent 15–54% pitch disagreement is atisim settling out of a trim
  that is not its own, at 0.95% of C_L, over eight to nine seconds. It is the price of the
  identical initial condition, it is now measured rather than assumed, and it does not
  touch the core response.

## 9. Session log

### Session 27 — the recorded trace is read at last, and it moves the blame back to the wind

**The session began as presentation preparation and turned into three measurements**, because
the honest answer to *"how do we know?"* kept running into things the project had listed as
done-or-blocked that were neither.

**First, a process failure worth recording because it is the reason this entry exists.** The
work of digitising TM-102186 Fig. 6 and MIL-F-8785C Fig. 7 **had already been done** — in the
`weekly-summary-analysis-7520db` worktree, by `scripts/digitise_tm102186_fig6.py` and
`scripts/digitise_mil_f_8785c_fig7.py`, complete with three independent checks apiece. Both
scripts were **untracked**, their outputs were gitignored PNGs, and that worktree's
`PROJECT.md` still listed both figures as *open acquisitions*. **The measurement was not the
hard part; keeping it was.** `CLAUDE.md` now carries the rule that closes this, and
§"How to update this document" points at it.

**Five scripts were rescued into this tree. Two were re-run and are in §4; three were not,
and this row says so rather than implying they were.**

| Rescued script | State in this session |
|---|---|
| `scripts/digitise_tm102186_fig6.py` | **RE-RUN HERE, verified against this tree.** §4 carries the result |
| `scripts/digitise_mil_f_8785c_fig7.py` | **RE-RUN HERE, verified.** §4 carries the result; it settles a sealed prediction |
| `scripts/les_flight.py` | **ALREADY RUN in the weekly worktree — all four domains × two aircraft, with logs, `.npy` arrays and summary CSVs (`runs/cat/les-D0*.log`, `les-summary.csv`).** Not re-run here. **Its results are NOT admitted to §4** — see the input-matching audit below, which is why |
| `scripts/les_compare.py` | **ALREADY RUN** — `les-comparison.csv`, `14-les-resolution.png`. Same status |
| `scripts/yoshimura_flightsim.py` | **ALREADY RUN** — `12-yoshimura-flightsim.png`. Reads the simulated half of their Fig. 6, 604 flights, ~17 hours of record |

**A correction to this entry's own first draft.** It said these three were "not run". They had
been run, extensively, on 3–4 September — the same failure mode as the two digitisations, one
layer deeper: the *outputs* were there to be found and the entry recorded their absence
instead of reading them. **Check the artefacts before describing the state of the work.**

**The LES limb is the single most valuable unrecorded thing in the project.** It is now tracked
rather than stranded, and §4 records what its inputs would have to satisfy before any number
from it can be quoted.

**Second, the recorded trace.** §4 has the table. The headline is that the project quoted
`−1.0 / +1.7 g` for four sessions **without looking at the curve between them**, and the curve
says three things the two numbers could not: the paper's `+1.7` is the **second** peak (to
0.002 g) and not the largest, which is `+1.855`; the gust spacing is **5.60 s**, against which
the flown model reads **−4.3%** where the prose alone gave **+7.2%**; and — the one that
matters — **the vertical wind the record carries reaches −98.3 ft/s where Mehta's fitted field
gives the model only −86.8**, i.e. **the model is flying air 12% too gentle.** §5 had stopped
naming the wind as a candidate for the 32% shortfall. It is a candidate again, and now a
measured one.

**Third, the wing loading, which was supposed to be the cheap one.** Session 26 made *aircraft
type* the surviving explanation and flagged its own `1.3×` as unverified. Fetched this session:
Douglas **DAC-67803A** gives a SOURCED Series-10 taxi weight and **no wing area at all**, and
the ratio turns out to depend far more on **which DC-10 and how much fuel** than on any area
figure — **0.58× to 1.32×, straddling 1.0**, with the project's `1.3×` sitting in the extreme
corner (heavier variant, at MTOW). **So session 26's sign is withdrawn rather than confirmed**,
and the shortfall has no surviving single explanation. That is a worse position to be in and a
more honest one. The thing that would settle it is not another spec sheet but **the flight
record**, which §7 now lists.

**Fourth, a second sealed prediction settled, and it was right.** σ_severe(37 kft) =
**4.80 ± 0.12 m/s** against a predicted `> 4.46`. The band exists because SEVERE and 10⁻⁵ are
**one stroke of ink** at that altitude, and the whole band clears the threshold. JSBSim's
independent transcription of the same figure lands 0.5% away. **Two of three settled, both
right** — and the register's rules were followed exactly: only `status` and `outcome` moved,
the digest is unchanged, and `test_predictions.py`'s classification was corrected (the entry
moved SOURCE_GATED → RUN_GATED) **because the document arrived**, not to make anything pass.

**Fifth, and it reverses one of this session's own recommendations.** The wing loading was
argued here as the highest-value cheap fix. **It was not** — it was the item whose answer was
least likely to exist, and fetching it produced a withdrawal rather than a result. The
higher-value work turned out to be **reading what was already on disk**: two digitisations and
a full LES run, all complete, none recorded. **Prefer auditing held artefacts over acquiring
new ones** until the held ones are known to be exhausted.

**Sixth, the LES limb was audited against its inputs and its numbers are refused.** §4 has the
table. The short version: their aeroplane's short period is **0.1436 Hz** (recovered from their
own `fs.f90`, confirming the paper's 0.14), against `boeing747` at +5.4% and
`boeing737_approach` at **+82.9%** — and the field tops out at 5,215 m, so the 747 flies it
**2.63 band widths outside its validated envelope** with `checks.recovery_band` failing on
every row. A 1.4× rms ratio means nothing against that. **The field reader is sound and stays;
the load comparison is withdrawn until a registry entry is built from Yoshimura's own
derivatives.**

**Seventh, and it is the session's most useful finding: the LES discrepancy is probably ours,
and it is the oldest known limitation in the model.** Asked whether "AtiSim is only accurate
near cruise" had been fixed or quietly ignored, the answer is **neither fixed nor accounted
for**. `ASSUMPTIONS.md` C3 is explicit that derivatives are frozen and that the **Mach axis is
unbounded**; session 23 bounded only the *altitude* axis. The LES runs fly the 747 at
**M 0.406** against its M 0.80 linearisation — **ΔM = −0.393, thirteen times the largest
excursion C3 had ever been tested against** — while the `q̄` ratio is a harmless 0.965, so the
one bound that exists does not apply. `aero.py` carries Mach only into `wave_drag`, so
`C_Lα` is used unchanged; the Prandtl–Glauert ratio between the conditions is **1.521**; load
goes linearly as `C_Lα`; and the measured ratio on the two resolved domains is **1.427 and
1.420**. **The discrepancy is the size the frozen derivative predicts.** §4 has the arithmetic
and a one-run falsification test, which would also be the **first quantified point on the Mach
axis** — and it needs no chart read, which is the reason §7 has been declining that bound since
session 12.

**Eighth, the B787 gate opened the same day it was set.** Yoshimura's own `fs.f90` carries the
complete derivative set; the four numbers needed to turn it into a registry entry are now held,
and **three of them are confirmed by inverting their own `Z_a`** to a physical `C_Lα` = 4.847
rather than being taken on a spec sheet's word. The fourth, `I_yy`, is DECLARED and **provably
cannot matter** — it and `C_mα` enter only as their product `M_α`, verified invariant by
round-trip across a 4× range. *The unsourceable number turned out to be the unobservable one.*

**What this session did NOT do**, so the next one does not look for it: it did not fit `W/S` to
Fig. 6, it did not "correct" the trace's 0.951 g cruise datum, it did not move §4's headline
denominator off 2.70 g, it did not re-run the LES (no matched registry entry exists **yet** —
but every input for one is now held), and **it did not run the `C_Lα` falsification test**,
which is the cheapest high-value thing left and should be first.

### Session 26 — four papers arrive, and one of them says the project was flying the wrong vortex

**Parks et al. 1985 is in hand** — J. Aircraft **22**(2) 124–129, the paper §8 had listed as
unobtainable since the audit. With it, Bach & Parks 1987, and AFFDL-TR-70-101 (HICAT).

**The headline is a correction, not an addition.** `PARKS_CASES["hannibal"]` carried a
**hybrid**: Wingrove Fig. 4's 500 ft radius with Parks' 85 ft/s and Parks' 3500 ft spacing,
under a comment declaring the hybrid deliberate because Parks "has never been retrieved". He
says **600 ft, 85 ft/s, 3500 ft** — one coherent triple — and TM-102186's rival triple is
equally coherent and equally whole: **1000 ft diameter, 87 ft/s, 3400 ft**. The project had
been flying one number from each. `ASSUMPTIONS.md` **E12** now holds all four lineages with
the rule never to cross them, and a test names the three crossings that must not reappear.

**His abstract settles the question session 22 had to infer.** *"the vortex cores had
diameters in the range of 900 to 1200 ft"* — 900 = 2×450, 1200 = 2×600. And he quotes the
Scorer ratio as *a range across both cases*, "about 2.9 to 3.5"; this document said "Parks
quotes 2.92", which he does not.

**The reversal cost 4.26% of the headline load, downward** — JSBSim's Hannibal peak-to-peak
1.8969 → 1.8162 g, 70.3% → 67.3% of the recorded 2.70. `dw/dx = V₀/r₀` in a solid-body core,
so a bigger core at fixed `V₀` is a *gentler* gradient. It makes §5's shortfall worse, which
is the direction that argues nobody chose it for convenience.

**And the pin came back bit-for-bit.** `test_vortex_viz`'s Fig-8 coordinates return to
`2.239956221700959 / -1.2352174348304876` — the exact doubles captured *before* the logging
refactor, which session 22 had recorded as permanently lost. Not to a tolerance. Through the
trim solve, the rollout, the logging path and the windowing, one altered bit anywhere would
have landed somewhere else. It says the two changes moved the radius and nothing else.

**Both cases are DC-10s**, p. 127 and p. 128 — so one derivative set serves both, and §7's
highest-value acquisition is worth twice what that row assumed. **And they are not equally
good targets:** Parks calls case 2 *"not as good as … case 1"* and blames *"strong mountain
wave activity which influences the short-period wind pattern"* — a contaminant in exactly the
band a vortex-passage comparison measures. `wind.PARKS_FIT_QUALITY` carries both assessments.

**THE SIGN CHECK IS THE LARGEST RESULT, AND IT GOES AGAINST §5.** That entry names aircraft
type as the surviving explanation for the load shortfall, with a flagged note that the
"0.8× wing loading" figure "is unverified and may be inverted" and that "nothing downstream
depends on the number". **Nothing depended on the number; everything depends on its sign.**

- §4's row *"737 → 747, four times the mass … no effect"* was explained as *"wing loading
  cancels"*. It does not. `C_L,trim = (W/S)/q`, so `Δn ∝ 1/(W/S)`. **The measurement never
  tested it**: those two entries differ by 5.95× in mass and only **1.267×** in wing loading.
- A controlled sweep on one airframe at one condition settles it. Change mass by ×1.25 **and**
  area by ×1.25 — wing loading unchanged — and the load moves **0.1%**. Change either alone
  and it moves as 1/(W/S), slightly sub-inverse because an aircraft that responds more also
  pitches away more.
- Flown at **1.3× wing loading**, the direction `test_wind.py` states for the DC-10, the load
  falls to **56.4%** of the record against the baseline's 70.1%.

**So if that figure survives verification, a DC-10 set makes the shortfall worse by 14 points
and aircraft type is a term with the wrong sign.** The sealed prediction
`dc10_does_not_close_the_hannibal_gap` becomes *more* likely to land — for a reason its author
did not give, since its stated reasoning is the "wing loading cancels" line §4 now records as
false. **The entry is SEALED and has not been touched.** That is what the register is for.

**What settling it needs is small, and the paper says how small.** Bach & Parks Eq. (2) shows
`m` and `S` enter their identification only as `m/S`, and Eq. (4) gives about **0.05° of α per
1% of `C_L`** — so 1% of wing loading is 0.05° against excursions of 7–9°. A DC-10 wing loading
to a few per cent is enough; precision beyond that buys nothing. **And Parks Fig. 7 is the
array while Fig. 6 is the g trace for the same encounter, so `W/S` must not be fitted to
reproduce Fig. 6** — that would convert the only end-to-end check into a calibration.

**Bach & Parks does not help acquisition #1**: its two validation cases are an L-1011 and a
B-747SP. **HICAT does supply measured RMS gust exceedances** — the published exceedance data
phase 2 lacked — but over **45,000–70,000 ft** against this project's 33,000–41,000, so any
comparison is extrapolated; and it compares itself against MIL-A-8861A and Steiner's NASA U-2
data, so **Fig. 7's high-altitude end may descend from the same aircraft** and would not be an
independent check. Settling that needs ADA119421, which is not held.

**The buffet boundary is digitised** from `refs/NASA-CR-114494.pdf` p. 2.0-38 — the sheet
session 21 read `CL_max` off, whose *second* curve nobody had traced. `aircraft.buffet_cl`.
The method is in that file; the check that makes it credible is that the same pipeline run on
the **upper** curve reproduces session 21's published table to **rms 0.0052, worst 0.0063**,
four times inside their stated ±0.02, from an independent trace.

**It buys a number where §5 had an argument.** At 37,000 ft and M 0.80 the 747 trims at
`C_L` 0.572 against a boundary of 0.721 — **1.26 g to initial buffet** — and the Hannibal
encounter drives `n_z` to **1.68 g**. The run is well past initial buffet and the linear aero
cannot know it. Nothing in the force model reads the table; a hard ceiling is a kink and §4
records eleven tests going red the last time one was tried.

**`d9d4442` is merged.** The thrust-moment bound the audit found abandoned on
`claude/jolly-bhaskara-def594` since 14 August — 0.38° of equivalent elevator at cruise trim,
1.5% of pitch authority, from CR-114494 p. 19.0-2. `ASSUMPTIONS.md` C5 stops saying
"unquantified".

### Session 25 — a statistic with an N in it, and where the four documents actually are

Phases 2 and 3 of the finishing plan recorded in §7.

**Phase 2 — change what counts as agreement.** Every comparison in §4 matched a *peak* from
a single encounter: one realisation of a random process, whose error bar does not shrink
however much more work is done. `atisim/response.py` adds the two statistics that do —
a one-sided PSD of a response history and an upcrossing exceedance rate. Neither existed
anywhere in this tree. `wind.dryden_spectrum` is an **input** spectrum, which is a different
object, and the collision of names is worth keeping straight.

**The bet was sealed before the machinery was pointed at an aircraft**, which is what
commit `2e6fd2b` is for: `scripts/cat_spectra.py` was written and committed unrun, beside a
prediction that the ensemble response peak would land within 20% of the 747's own short
period, 0.1640 Hz. **Settled RIGHT, at both intensities and not comfortably** — 0.1400 Hz
at σ = 2.108 m/s and 0.1700 at 4.459, against a predicted [0.131, 0.197]. The lower one
sits less than one bin inside the band's edge, pulled down by the falling Dryden input
exactly as the sealed reasoning said it would compete. A tighter band would have been wrong.

**The robust form of the result is not the peak location, and the entry says so.** An
averaged periodogram at N = 32 is still noisy. What does not depend on N is a ratio of two
numbers: the **input** has strictly more energy at 0.05 Hz than at the short period, and
the **response** has more than 3× less. That reversal is the whole content of "the airframe
organises the load", and it is what `test_cat_spectra.py` asserts.

**And the headline run answers a question it could not previously be asked.** Mehta's array
is not uniform — its four core spacings force the aircraft at 0.0909, 0.1224, 0.1354 and
0.1886 Hz at once. The response peaks at **0.16885 Hz**: 0.23 of a bin from the airframe's
short period and 1.64 bins from the mean forcing. **The load follows the aeroplane, not the
array.**

**A tripwire fired, and it was right to.** `test_predictions.py` asserted that every sealed
claim was blocked on a source §5 names as unobtainable, on the reasoning that a claim
checkable from what is held "is not a prediction — it is a run someone has not done yet".
The new entry is exactly that kind. The rule the tripwire was protecting turns out not to
be the unobtainable source but the **seal preceding the run**, which is a fact about the git
history; an unobtainable source merely makes that ordering free. Rewritten as two admissible
classes, so adding an entry means classifying it rather than incrementing a count.

**What phase 2 did NOT do**, and the script prints it before the figure: it is not yet a
*comparison*. No published exceedance curve is held and no digitised acceleration history
exists here, so the observed half of Yoshimura's protocol cannot be run. §5 now carries both
as named acquisitions.

**And it produced one result that was nearly written up as physics.** The two ensembles
differ in `n_z` rms by **+5.46%** more than their σ ratio, which reads as a superlinear
response to gust intensity. It is not: `aero.py` is linear in α and `boeing747` carries no
`CL` table, so the aerodynamics *cannot* produce it. Measuring what else changed found it —
**with fixed controls there is nobody flying the aeroplane**, and over the 100 s record at
the upper σ it drifts **703 m in altitude and 13.1% in airspeed**, which is about a quarter
of dynamic pressure, with |α| reaching 8.38° against §1's 10° limit. So the upper-σ
ensemble is an average over flight conditions rather than a spectrum of one. Registered as
`ASSUMPTIONS.md` E11, reported per ensemble by the script from now on, and the reason the
lower-σ ensemble is the one to quote. **The fixed controls are not a defect** — §7's
discriminator is *defined* by them — which is why this is a limit to declare rather than a
bug to fix.

**Phase 3 — the four documents, and the search changed their order.**

| # | Document | Found? | What the search established |
|---|---|---|---|
| 1 | DC-10 cruise derivative set | **no** | Not in the open literature. Heffley's own library — the source of CR-2144 — has no DC-10, and its other compilation, CR-96008, is 1969, before the type flew. Every hit was the winglet programme (NTRS 19850002628, 19870008261), which reports that winglets *did not change* the stability characteristics and therefore prints no baseline table. This is the highest-value item and it is the one nobody is giving away |
| 2 | 747 buffet onset / nonlinear `C_L` | **no — and it turned out half of it was on the shelf** | **The boundary is held**: `refs/NASA-CR-114494.pdf` p. 2.0-38 is *"LIFT COEFFICIENT — BUFFET BOUNDARY AND C_Lmax"* and carries an *initial buffet boundary* curve beside the `C_Lmax` curve session 21 digitised off the same sheet. The web search was looking for something the folder already had. What is genuinely absent is the NONLINEAR LIFT CURVE, and the open 747 buffet literature is the **Shuttle Carrier Aircraft**: 0.03- and 0.046-scale tail-buffet wind-tunnel tests (NTRS 19750025089, 19770003191). That is the orbiter's wake on the empennage, not wing buffet onset at cruise — **the wrong buffet**, and quoting it would be worse than having nothing. NASA TN D-7131, whose title promises "Maneuver and Buffet Characteristics", is fighters |
| 3 | **MIL-F-8785C Fig. 7** | **YES, free** | The specification itself is public at everyspec.com, and DTIC's Background Information and User Guide (ADA119421) is on archive.org. **And an independent transcription already exists**: JSBSim's `FGWinds.cpp` carries the table under the comment *"this is Figure 7 from p. 49 of MIL-F-8785C"*, values in **ft/s**, rows a probability-of-exceedance index 1–7 with *"3=light, 4=moderate, 6=severe"* |
| 4 | Yoshimura figshare **21152203** | **YES — but not for what it was named for** | CC BY 4.0, one file `data.tar`, **17,942,056,960 bytes (17.9 GB)**: LES outputs at dx = 500/250/70/35 m, the flight-simulation code and its outputs, GrADS control and script files. **The three onboard flight records and the JAL PIREP are excluded by confidentiality agreement** — the paper says so and the dataset page repeats it |

**Two of those findings change the plan rather than execute it.**

- **Item 4 was listed as the route to a recorded acceleration history. It is not** — that is
  precisely the part that could not be released. What it *is* is a published **LES CAT wind
  field**, which is the non-circular field §5 has wanted since session 23: every load
  comparison in this project so far is flown through a field identified *from* the
  aircraft's own accelerations. That is a different and still substantial prize, at 17.9 GB.
- **Item 3 is free and immediate**, which promotes it above items 1 and 2 on cost even
  though the plan ranks it third on value. Better still, it can be done the way this project
  does everything else: **digitise the figure from the document, then check it against
  JSBSim's independent transcription**, which is the same two-independent-readings pattern
  as the vortex fields agreeing to 8.8e-10 m/s.

**The sealed prediction it would settle is left SEALED, deliberately.**
`mil_f_8785c_sigma_w_exceeds_the_mehta_ceiling` bets that the severe curve at 37,000 ft
exceeds 4.46 m/s. Interpolating JSBSim's transcription between its 35,000 ft and 45,000 ft
columns on the severe row gives **15.82 ft/s = 4.82 m/s**, which would settle it RIGHT.
**That is not this project's evidence and the entry is not being settled on it.** Its own
`settled_by` says "MIL-F-8785C Figure 7, **digitised** at 37,000 ft"; a third party's
transcription of a figure is not a digitisation of it, and §3's rule that every number
carries the table it came from exists for exactly this case. The number is recorded here as
what to *expect*, which is the honest use of it — and it makes the digitisation a check
rather than a discovery.

**Then both obtainable documents were fetched, and one of them closed something on
arrival.** `refs/MIL-F-8785C.pdf` (95 pp., 5 Nov 1980) and the 17.9 GB figshare tarball.
Figure 7 is confirmed on printed p. 49 — *"Turbulence exceedance probability"*, a rotated
scan whose σ axis reads **RMS TURBULENCE AMPLITUDE, σ (FT/SEC — TAS)**, which independently
confirms the units JSBSim's transcription uses. The digitisation itself is a session's work
on a poor scan and is deliberately not attempted in passing; the prediction stays sealed
until it is done properly.

**What did close, unasked, is the pair of Dryden spectral forms.** §3.7.1.2, printed p. 47,
prints all three components, and `wind.dryden_spectrum` and
`wind.dryden_longitudinal_spectrum` match it **verbatim** — including that the spec gives
`v` and `w` *identical* right-hand sides, which session 24 had inferred from the isotropy
relation and can now cite. `test_lateral.py` gains a check against the printed formula at
three arguments chosen so the values are arithmetic rather than a re-typing of the code:
Ω = 0 pins the factor of 2 on the longitudinal form, LΩ = 1 is where the two forms cross
exactly, and LΩ = 2 pins the 3 in the numerator against the square in the denominator by
which form is larger. **Holding the document closed the forms, not the intensity** — and
the forms were the part nobody had asked about.

**Then a second search, of the folder rather than the web, moved one more item off the
acquisition list.** The plan named "a digitised acceleration history" as something to
acquire, and this document's own §5 entry repeated it. **It is already held.** TM-102186
is `Reference_papers/19890016606.pdf`, in the project since session 23, and its **Figure 6**
is the Hannibal DFDR **g-load time history** — the recorded trace whose +1.7/−1.0 g extremes
this project has been quoting as `wind.TM102186_HANNIBAL_NZ` for two sessions **without ever
looking at the curve between them**. Yoshimura's equivalent is the part figshare withholds,
so this is the only recorded trace within reach, and it was never out of reach.

**The same page also supplies the phase-2 result's source.** TM-102186 p. 3-4 explains its
three-aircraft ordering by saying the pitch and load variations "are dependent upon the
relationship between the time span of the vortex traverse and the aircraft's short
oscillatory period". **That is the resonance claim the spectrum measures**, asserted by the
source and untestable here until this session — the project had no frequency-domain
statistic at all. §4 now cites it beside the measurement.

**Two lessons, and they are the same lesson.** `ASSUMPTIONS.md` E10 already records it —
*ask what every case has in common, not just what each one assumes* — and the folder version
is: **ask what the held papers contain, not only what they were fetched for.** Session 23
took TM-102186's two-number band and moved on. Session 24 inferred the `v`/`w` spectral
identity that MIL-F-8785C prints. Both were right; both left something on the table that a
second reading found.

**A third reading found a third one, and it changes the shape of phase 3.** The plan's
item 2 is "747 buffet onset / nonlinear `C_L`". `refs/NASA-CR-114494.pdf` p. 2.0-38 is
titled *"LIFT COEFFICIENT — BUFFET BOUNDARY AND C_Lmax"* and draws **two** curves against
Mach. Session 21 digitised the upper one — §4's `CL_MAX(M)` table cites that exact page —
and left the *initial buffet boundary* beside it untouched, with the sheet's own pointer to
§19 for revised data. **The web search in this session was looking for a document the
folder already held.**

**So phase 3's premise was wrong about three of its four items, and the corrected list is
much shorter:**

| Plan item | What it actually is |
|---|---|
| DC-10 cruise derivative set | **the only genuine acquisition.** Not in the open literature |
| 747 buffet onset | **held** — CR-114494 p. 2.0-38's second curve, plus §19. The *nonlinear* curve is unreachable and stays a §5 impossibility |
| MIL-F-8785C Fig. 7 | **held** — free, fetched this session |
| Yoshimura figshare | **obtainable, but not for the stated reason.** The recorded trace it was wanted for is TM-102186 Fig. 6, held since session 23 |

**Three digitisations and one acquisition**, where the plan wrote four acquisitions. Every
one of the three is a figure in a PDF already on disk, and each has a stated difficulty
rather than an assumed one — Fig. 7 is a rotated scan, TM-102186 Fig. 6 is a dense
oscillatory trace at low print resolution, and the buffet boundary is the easiest of the
three because it is a smooth single-valued curve of exactly the kind session 21 has already
done twice on that same sheet.

**The figshare tarball landed and verified**, 17,942,056,960 bytes to the byte and md5
`d23cbb3c…` against figshare's own checksum. Its shape is more useful than the headline
size suggests: four LES domains at 2.3, 3.6, 4.0 and 7.5 GB, and **`flightsim-data.tar.bz2`
at only 561 MB** — Yoshimura's own 2-D B787 simulation code and outputs, the *simulated*
half of their Fig. 6. **The 3% of the download that is not LES field is the part that can be
used first**, and it supports a cross-code response-spectrum comparison of exactly the kind
`scripts/vortex_compare.py` already does against JSBSim. The LES field remains the larger
prize and the harder one: it is the first field available to this project that was **not**
identified from an aircraft's own accelerations, which is the circularity §4's session-23
entry opens with.

**Suite: 788 → 807 passed, 1 skipped, 15m14s.** Twelve unit checks on the estimators, all
against signals whose answer is closed-form, six on the flown result — including a still-air
negative control in which every statistic in the new module returns exactly zero — and one
on the Dryden spectral forms against the specification now that it is held.

### Session 24 — the model gets a lateral dimension, and the record gets a claim

Two phases of the plan written at the end of session 23d.

**Phase 0 — bank what is already true.** §1 now carries a formal **validation claim**
with its envelope attached: aircraft class, Mach and altitude band, |α| limit, gust length
scale in spans, and *longitudinal only*. The project had a great deal of evidence and no
single statement of what it added up to, which meant the honest narrow claim was going
unmade while a broader one was available to be inferred by accident.

**And writing it found the gap.** Every wind field in the project was a function of
along-track distance alone — each for a different reason, which is why nobody had noticed
the pattern. `strip_roll_moment` integrated to exactly zero on all of them, the lateral
modes were validated as eigenvalues and never excited, and `Encounter` had no channel that
could have recorded a roll. **The model was longitudinal by construction and no document
said so.** Now §5 does.

**Phase 1 — give it the dimension.** Three pieces:

- `wind.dryden_field` adds the `u` and `v` components. The spec is not in the folder, so
  the check that matters is that the two spectral forms belong to one isotropic field —
  `Φ_t = ½(Φ_u − Ω dΦ_u/dΩ)`, satisfied to **1.3e-16**.
- `vortex_viz.Encounter` gains `phi`, `beta`, `p`, `r` and `p_gust`, trailing and
  defaulted so nothing existing moves.
- `wind.line_vortex_wind` writes Parks' vortex as lines in space rather than as a formula
  on the flight path.

**The result, and it is not small.** Through Mehta's own field the 747 reaches **12.5° of
bank, 15.4° with strip loads** — against **exactly 0.000°** from the point model. And
`loads.strip_increment`, built in session 14, **moves a reported number for the first
time**: +22.9% on peak bank, where it had previously changed every result by exactly
0.000000. The longitudinal answer moves +5.9%, so nothing already concluded was resting
on the missing dimension.

**A real simplification, found on the way.** Parks' model is 2-D in the plane
perpendicular to the vortex *lines*; `vortex_wind` returns his horizontal magnitude along
the *flight path*. At Mehta's 31° the true perturbation is that magnitude rotated by ψ, so
**sin 31° = 0.515 of it lies across the path** — the only sideslip input the field has
ever had, discarded since session 3.

**A bug this work made, and caught.** `cos_dpsi` predates `sin_dpsi`, so
`mehta_hannibal_array` carried a cosine with the sine at its 0.0 default: a vortex axis of
length cos ψ rather than 1, scaling the whole induced velocity and looking exactly like
physics. Caught because the script compares the two forms in a geometry where they must be
identical and got 7 m/s. Fixed in the constructor *and* by normalising the axis, so a
mismatched pair is a wrong angle rather than a wrong magnitude.

**What this does NOT establish, and the documents say so three times.** No source held
here records a lateral CAT response. Every lateral number is a capability demonstration.
§1's claim excludes lateral behaviour explicitly, §5 repeats it, and `test_lateral.py`
carries a negative control on every non-zero assertion — because §6's lesson is that a
suite exercising only the easy case cannot see the bug, and a longitudinal-only suite
hides lateral bugs exactly the way a still-air one hid air-relative bugs for three
sessions.

### Session 23d — a second engine on the headline field, error bars on Fig. 8, and the first sealed bets

Three follow-ups, and one of them qualifies a conclusion this document has been
carrying since session 22.

**What was built.** A five-vortex oblique case in `scripts/gen_jsbsim_vortex_reference.py`
and its `<cores>` schema in `atisim/jsbsim_vortex_ref.py`; `scripts/cat_ensemble.py` with
figure `07-ensemble.png`; `atisim/predictions.py` and `test_predictions.py`; four new
tests in `test_cat_validation.py`; four tests in `test_jsbsim_vortex.py` rescoped.

**1. Neither engine reaches the record on the field the headline result flies.** JSBSim
given Mehta's five-vortex array, the same aircraft and the same starting state reaches
**74.6%** of the DC-10's recorded peak-to-peak; AtiSim reaches **67.1%**. That closes the
solver as an explanation *on the headline field* rather than by inference from the
adjacent Parks cases. The two field implementations — numpy in the generator, JAX in
`wind.vortex_wind` — agree to **8.8e-10 m/s**, including the five-way superposition and
the 31° obliquity, so nothing here rests on a bug shared by both sides.

**2. And the same run qualifies session 22.** "Both engines under-predict by the same
amount" was measured on single cores, whose window is 1.3 s — a fifth of a short period.
On the array, 34 s and five short periods, the engines **select different cores** as the
worst: AtiSim's peak is 2.2% *higher* than JSBSim's while its trough is 3.5× shallower,
and the entire −10.1% span disagreement is the trough. The conclusion survives in the form
that matters — neither engine reaches — but "by the same amount" was a property of the
short window.

**A hypothesis tested and failed, recorded so nobody repeats it.** The array run is 45 s at
fixed throttle and JSBSim's airspeed swings −5.5% across the window, against −0.009% on the
single core; `n_z ∝ V²`, so drift looked like the whole story. Shrinking the window until
the drift is −0.19% leaves the disagreement at −7.6%, and AtiSim's span is identical at
every window width from ±2 to ±32 core radii. It is not the drift. §8 carries what is left.

**3. §7 step 6 is finally done, and its margin is the interesting part.** The ordering
holds **16/16** at both ends of the σ_w range derived in 23c from Mehta's residual — the
intensity is sourced, not picked. But raising σ_w by 2.1× collapses the vortex/updraft
**pitch** gap by 15×, to **0.060°**. The **load** gap survives at 0.343 g. Fig. 8 is a
two-dimensional discriminator so the categories still separate, but at the top of the
sourced range they separate on load alone. Stated with its limits: the manoeuvre limb is
noiseless by category definition, which helps the three-way ordering; and a 16-seed gap
estimates the tails of a distribution whose extremes grow with sample size, so
"non-overlapping" means *marginal*.

**4. The first sealed predictions.** Every number in §4 is retrodictive — the paper was
open beside the model. `atisim/predictions.py` is where that stops being the only kind of
evidence here. Two entries, sealed against tree `0c72200`, both blocked on sources §5
already names as unobtainable. The first **bets against this document**: §5 now names
aircraft type as the surviving explanation for the 32% and a DC-10 derivative set as the
highest-value acquisition, and the prediction says that acquisition will *not* close the
gap, on the strength of §4's measured null that quadrupling mass moved the load 0.6%.
One of the two is wrong.

**A drafting mistake worth keeping.** The first version computed each prediction's digest
at import from the fields it hashes, which makes the seal vacuous — change a claim and the
digest follows it, and the check can never fail. The digests are literals now, and
`test_a_computed_digest_would_have_made_this_check_vacuous` is the negative control.

**Four existing tests were rescoped, not relaxed.** `test_jsbsim_vortex.py` characterised
single cores in four places — one sign change in the gust, a 1% run-in speed bound, a fixed
key set, a single-core field reconciliation. The array violates three of them legitimately.
Each was scoped to `cores is None` and given the array's own assertion beside it; the 1%
bound was **not** widened to admit a 2.34% array value, because 1% characterises a
15-radius run-in to one core and says nothing about five.

### Session 23c — error bars on both sides, and the shortfall stops being a wind problem

Three follow-ups, all aimed at the same weakness: **every load comparison in this project
is partly circular.** The vortex parameters were identified *from* the recorded
accelerations, through Parks' and Mehta's aircraft model, so predicting those
accelerations back tests the composition of two models rather than this one.

**What was built.** `wind.MEHTA_COST_STARTUP`, `wind.MEHTA_COST`,
`wind.MEHTA_COST_SATURATES_AT`, `wind.mehta_residual_ceiling`,
`wind.mehta_unmodelled_wind`; `scripts/cat_uncertainty.py` with figure
`06-uncertainty.png`; eight tests; four ledger entries.

**Three results.**

1. **A non-circular channel reproduces.** Gust *spacing* is set by the fitted core
   positions and the aircraft's speed, not by the fitted strengths — measured, not
   asserted: the field is exactly linear in `V₀`, so scaling it by 0.5, 1.5 or 3.0 moves
   every wind value and **not one turning point**. The flown run gives **5.29–5.43 s**
   depending on which reading of "apart" is taken, against TM-102186's prose "about
   5 sec". The three readings span 0.14 s, so the answer belongs to the field and not to
   the definition. The 6% residual is a speed proxy — 5.0 s needs M 0.848 and this 747
   flies M 0.800 — and the model is **slow**, which is the direction that mismatch
   requires.

2. **Mehta's own fit residual bounds the field form, and it needed re-reading the
   Appendix to get right.** Eq. (A3) carries a **1/N**: the cost is a *mean* square, so
   `J` = 214 converts to a 4.46 m/s RMS residual **without knowing `N`**, which the paper
   never states. Two readings of that formula differ by ~65× and give opposite
   conclusions, so it was rendered from the PDF rather than guessed. Subtracting Lester's
   reconstruction error leaves the unmodelled wind at **2.11–4.46 m/s**.

3. **The 32% shortfall is not in the wind at all.** The propagated input band tops out at
   **72.7%** of the recorded peak-to-peak, and the peak load is **saturated**: tripling
   `V₀` moves the up-increment from 0.441 to 0.526 g against a recorded 0.70, elasticity
   under 0.15 and changing sign, with peak |α| still near 8°. The aircraft pitches away
   and sheds the gust — Fig. 8's own mechanism, seen from the inside. Bracketing finer
   than the first sweep did turned this from a bound into an **exclusion**: the peak
   first reaches +1.7 g between ×3.25 and ×3.50, and |α| leaves the linear range in the
   same interval, so no gust strength both reaches the record and may be believed.

**This weakens a previous session's conclusion rather than extending it.** Session 23b
reported σ_w ≈ 4–5 m/s closing the upper extreme. That still happens, but 4.46 m/s is now
known to be the **ceiling** of what Mehta's residual permits — it requires the entire
residual to be vertical, unmodelled, and free of reconstruction error — and the ceiling
run sits *on* the 10° edge of the linear range. At the sourced lower bound the run does
not reach the band at all. The Dryden explanation is not excluded, but it now needs the
top of its own range rather than the middle.

**A mistake of this document's own, found and fixed.** §3 and §8 both described Mehta's
"documented cost history (482 → 214)". 482 is his *manual startup estimate*; the converged
series is 355 → 303 → 226 → 214. Pairing a startup value with a converged one as a single
history is precisely the error §5 flags TM-102186 for making with Schultz's Table 1 —
made here, by this project, about the same paper family. Now separated in code, with a
test asserting the separation.

**Guards that fired.** `test_the_provenance_ledger_does_not_cover_the_source_modules`
failed the build on two unledgered constants, and was fixed by writing entries. A first
draft of the invariance test used an absolute gust threshold against a scaled field, so it
compared thresholds rather than gusts; caught because the extremum count changed with
scale.

**What is now left.** The shortfall is cornered on the **aircraft** — a DC-10 record flown
by a 747, with no buffet-onset data. That is §5's structural entry, and it has stopped
being a caveat and become the surviving explanation. A DC-10 derivative set is the
highest-value acquisition on the list. Also flagged, not fixed: §5's "roughly 0.8× the
wing loading" is unverified and may be inverted — AtiSim's 747 is W/S ≈ 115.8 lb/ft² and
no sourced DC-10 figure is held.

### Session 23 — eleven papers arrive, and four "not quantified" entries stop being that

**What came in.** Twelve PDFs in `Reference_papers`, eleven distinct works. Seven are new
to the project: Mehta 1987, NASA TM-102186, Lester/Sen/Bach 1989, Misaka 2008, Yoshimura
2022 and 2023, Schultz 1990, plus Bach 1991's SMACK manual and Palacios & Cesnik ch. 4.
Loving 1965 is context with no reusable dataset; Stengel was already load-bearing; one
file is a duplicate.

**The theme: four §5 entries that said "not quantified" or "declared, not sourced" were
wrong to say so, and closing them needed no new measurement — only a source that had
not been read.**

| Entry | Was | Now |
|---|---|---|
| Hannibal core radius | conflict, unarbitrable | **500 ft** — Mehta's later five-vortex refit, reported again by TM-102186. 600 ft is Parks' own earlier fit, self-consistent with his Scorer ratio, and is superseded rather than corrected |
| vortex parameter band | ±25%, reasoned | **8.45%**, from Lester Table 1's propagated RMS errors |
| flexible-airframe mismatch | "cannot be produced from sources the project holds" | bounded at constant Mach: −17% to −61% across 2.48× q̄ |
| lee-wave wavelength | declared, unbounded | 12% from a measured 22 km at the right altitude regime |

**What was built.** `VortexArray.cos_dpsi` (the oblique traverse the module had refused,
now transcribed from a source that constrains it); `wind.mehta_hannibal_array`, the only
wind field in the project that declares nothing; `vortex_viz.fly_in_moving_air`;
`checks.rms_normal_load`; twelve `REFERENCES` rows for a third CR-2144 flight condition;
`scripts/cat_validation.py` and 23 tests in `test_cat_validation.py`.

**Three results worth remembering.**

1. **The 747 lands inside the DC-10's measured load band** on Mehta's own identified
   field, at 68% of its peak-to-peak. Absolute agreement was never available; being
   *inside* is what was reachable and it was reached.
2. **The paper's non-obvious claim reproduces.** TM-102186 Fig. 8's `n_z` minimum
   ordering runs *opposite* to its pitch ordering — the aircraft that pitches least eats
   the most load — and the model gets that reversal, with the cherokee/RPV pair agreeing
   inside the digitisation error. The mechanism behind it collapses monotonically across
   all six aircraft, six for six.
3. **A 23.5% short-period error was traced to its cause in one experiment.** Swapping the
   derivative set for the one CR-2144 tabulates at that condition took it to 0.6%, and the
   remaining damping shortfall was *predicted* to −12.2% from the missing `C_mα̇` against a
   measured −11.7%. Neither number is a fit.

**Two mistakes in the sources, and three of the project's own guards firing.** TM-102186
quotes Schultz's initial estimates as his results; Yoshimura 2022 misreads its own `s⁻¹`
as `Hz`. And every safety net this project has built caught something real:

| Guard | What it caught |
|---|---|
| §10's tree check | `scripts/cat_validation.py` printed `atisim.__file__` on its first run and found itself importing the **main checkout**. The failure mode that produces no error message, producing one |
| `test_the_provenance_ledger_does_not_cover_the_source_modules` | eleven new `wind.py` constants added without ledger entries. **Failed the build**, exactly as §2 promises. Fixed by adding the entries, never by widening the baseline |
| `test_checks_round_trip_with_their_kind` | the new severity check returning `NaN` on a short run, which is not valid strict JSON. Fixed by omitting the check rather than fabricating a value |

**Two of my own numbers were wrong and were caught by recomputing them.** The dynamic
pressure ratio between the two 747 conditions was written as 2.16× — that is the *density*
ratio; at constant Mach `q̄ = ½γM²p`, so it is **2.48×**. And the Mehta/Parks strength
spread was written as 2.4% when 86.8/85 is **2.1%** (2.4% is TM-102186's rounded 87).
Both were in six files before being corrected. The air-vs-inertial α divergence was
likewise read off a plot at 7.0° and is **6.85°** computed; the script now prints it.

**What did not change.** No frozen §4 baseline moved. `PARKS_CASES` is untouched and still
a hybrid, deliberately; the coherent single-source pair lives beside it and the 2.1%
spread is measured rather than argued. The orderings-only rule on vortex conclusions
stands — everything above is an ordering, a band, or an attribution.

### Session 23b — five follow-ups, and the shortfall is cornered rather than closed

The session-23 pass ended with the Mehta run reaching 68% of the DC-10's recorded load
and five named next steps. All five were run.

| | Asked | Answered |
|---|---|---|
| 1 | restore the α̇ terms | `ζ` at M 0.8 / 6,096 m goes **−11.7% → +0.14%**. Given CR-2144's own derivatives for a condition, this model now reproduces CR-2144's own short period to 0.6% and 0.14% |
| 2 | declare the 747's valid band | `[0.70, 0.90]` M, `[35,000, 45,000]` ft. The altitude edges are **derived**, not declared — a two-point interpolation in q̄ that self-checks to +24.2% against a measured +23.5% |

**Item 2 changed a documented rule, which is worth flagging loudly.** The rule was *only an
entry that IS a fit may carry a band*, and it rested on the reasoning that a linear
derivative set from CR-2144 is "valid across the ordinary linear range" — that its
limitation is on α and not on flight condition. For a rigid aircraft that is right:
non-dimensional derivatives depend on Mach and geometry, not altitude.

**The measurement says it is false for this data.** At constant M 0.80 CR-2144's own 747
derivatives move −17% to −61% between 40,000 and 20,000 ft, and the short period that
follows is 23.5% high in silence. The tabulated set does have a condition range; there
were simply never two conditions to compare before.

The replacement rule is what the old test was really enforcing: **a band may be declared
only where its edges were established by measurement**, and there are now two ways to
establish them — a JSBSim fit sweep, or an interpolation between two held conditions of
the same document. `aircraft.DECLARES_A_BAND` is a separate set from
`RECOVERED_FROM_JSBSIM` precisely because membership of the latter implies two further
things (the source CL(α) table, trimming inside a table segment) that `boeing747` does not
have and must not be given.

Three guards fired on this change and all three were right: the band test above, the
`recovery_band` report/gate test (whose specimen was `boeing747` and is now the approach
entry, which still declares none), and the provenance ledger over the two new Dryden
constants.
| 3 | fly Lester's Greenland 747 | the recorded climb and the recorded g-load imply lee-wave amplitudes **28× apart** |
| 4 | Dryden on top of Mehta | σ_w ≈ 4–5 m/s closes the **upper** extreme; nothing closes the lower one |
| 5 | bound the point gust and the step | **≤4.4%** of a 32% gap. Strip loads move it by exactly zero |

**The shortfall is now cornered.** It is not numerics (0.14% over an 8× dt range), not the
point-sampled gust (≤4.4%), and not the strip path (0.00%). Adding the random component
Mehta's fit explicitly excludes closes the positive extreme at a plausible intensity and
leaves the negative one untouched. What remains unexplained is a **downward** excursion,
on an aircraft that is not the one in the record.

**Item 3 is the result worth keeping.** It is the first load comparison in the project
flown by the aircraft type the record is actually of, and it produces a physical
conclusion rather than a percentage: the smooth wave reproduces the slow response at an
amplitude Doyle measured and misses the fast one by a factor of 28, so the accelerations
came from the wave breaking down — which is what Lester's own critical-level reading says.
`wind.LeeWave` is missing the breakdown, not a coefficient.

**Two things were deliberately NOT done.** `boeing747` gets no α̇ terms, because no source
supplies them at flight condition 9 and Table A2's are 2.48× of q̄ away. `boeing747_approach`
gets none either, although Table IX-2 tabulates them: they take the short period from
+1.4%/−5.5% to −0.3%/+0.1% and the phugoid damping from −4.6% to **+8.4%**, past the 5%
`test_validation.py` asserts. Adopting them means re-pinning a tolerance to admit a change,
and B5 is the precedent for refusing that. The measurement is kept as a test.

### Session 21 — the Wingrove paper arrives, and JSBSim gains a 747

**The source the audit could not obtain is now in hand.** Wingrove & Bach, *Severe
Turbulence and Maneuvering from Airline Flight Records*, J. Aircraft **31**(4), Jul-Aug
1994, 753-760. `AUDIT.md` rows 19, 168 and 393 marked `UPDRAFT_W0`, `UPDRAFT_SECONDS`, the
5.2 deg pitch figure, the +0.66/-1.58 g and the Fig. 8 discriminator as
`unverifiable -- source not available`. All five check out verbatim; closing those rows is
still outstanding.

**The paper names no aircraft.** All twelve incidents are "modern airliners" carrying a
DFDR; Table 1 gives date, location and altitude, Table 2 gives load increments, and no
table, figure or sentence identifies an airframe. Any comparison against it therefore
chooses its own aircraft, and this project's choice is the JSBSim-intersection below.

**Fig. 4 supplies a third vortex case and contradicts `PARKS_CASES` on a second.** The
figure quotes core DIAMETERS: Hannibal 1000 ft, Morton 900, Cimarron 900, at 85, 70 and
50 ft/s. Morton halves to 450 ft, which is `wind.py`'s Morton radius to the digit and
settles that the column is a diameter. Hannibal halves to **500 ft where `wind.py` says
600**, and Parks 1985 has still never been obtained, so nothing arbitrates. Both are kept
in separate dicts, `PARKS_CASES` and the new `WINGROVE_FIG4_CASES`, and both will be flown.
Cimarron exists only in this paper, and it is the only case with published time histories.

**JSBSim cannot be asked to carry a gust gradient.** Its property catalog reports
`atmosphere/{p,q,r}-turb-rad_sec` as READ-ONLY, and a write to `q-turb-rad_sec` reads back
0.0 after one step. Every writable wind input -- `wind-*-fps`, `gust-*-fps`, `turb-*-fps`,
`cosine-gust/*` -- is translational, sampled at one point. Translational injection does
work and was checked against arithmetic: `wind-down-fps = -50` at M 0.78 / 30,000 ft moved
alpha 3.687 deg against a predicted `atan(50/776) = 3.69`. So the gradient arm of the
vortex comparison measures what JSBSim omits, using atisim as the instrument, and cannot
be a JSBSim-to-JSBSim delta.

**`boeing747_jsbsim` joins the registry, and `boeing747` is untouched.** JSBSim's B747 is
not the CR-2144 aeroplane: 249,974 kg against 288,773 (15.5%), 64.46 m of span against
59.64 (8.1%), 524.7 m^2 against 511.0 (2.7%) -- while sharing an inertia tensor to under
0.1%. Since n = L/W, a load-factor difference between the two engines flying "a 747" would
have been dominated by that mass gap. The new entry carries JSBSim's own numbers, recovered
at 38,000 ft / M 0.80 by `scripts/gen_jsbsim_747.py`, which imports the 737's recovery
machinery rather than copying it.

**It is not a credible 747 and the entry says so.** B747.xml is `release="ALPHA"`, author
"Unknown", and its CLalpha table shares its first three points with 737.xml's -- both give
CLa = 4.3478 /rad. That is one Aeromatic template used twice. It costs the cross-code
comparison nothing and makes any comparison of that entry against flight data meaningless;
`boeing747` remains the aeroplane for that.

**Two things were caught by tests rather than by inspection**, and both are recorded because
neither was loud:

- **The Ixz sign was wrong.** `inertia_tensor` negates its argument, and JSBSim's
  `inertia/ixz-slugs_ft2` is already the tensor element, so the engine's -970000 had to be
  passed as +969999.99. Passing it as reported flipped the cross-product term -- the exact
  failure the 737 shipped with once, worth 1.6-3.2% on the lateral modes and inside every
  layer tolerance. `test_mass_and_inertia_match_the_engine` compares the ASSEMBLED tensor
  against the reference rather than the argument against a remembered convention, which is
  the only form of the check that catches it.
- **A unit constant differing in the 8th digit.** slug ft^2 values converted with
  1.35581796190452 instead of `units.SLUG_FT2_TO_KG_M2` = 1.3558179483314003 left a 1e-8
  relative gap. Recomputed with the project's own constant, Izz and Ixz land on B747.xml's
  stated 4.97e+07 and -970000 to ten digits.

**The recovery lands on B747.xml's own constants**, which is the check that it is a recovery
rather than a plausible fit: CYb, Clb, Clp, Cldr, Cnb, Cndr and CLde exactly, Cma -0.699965
against -0.7000, Clda 0.0732 against the Mach-scheduled table's M 0.80 value. Cmq and
Cmadot come back as -21.0055 and -3.9945 against -21 and -4 -- each 0.0055 out, equal and
opposite, at a design condition number of 2.7e9 -- while their SUM is -25.000000. The same
q/alphadot collinearity the 737 records, and the sum is the quantity that carries weight.

**The refactor that made the reuse possible changed nothing**, and that is measured: both
737 reference XMLs regenerate to the same blob hash they had before
(`c0d5522a...`, `44eb69f7...`).

**The two engines do not trim to the same point, and that is a design input.** At the
recovery condition atisim trims to alpha 4.3321 deg where JSBSim trims to 4.2786, a
+0.0535 deg gap. It is not an entry defect: fed JSBSim's own trim state, atisim returns
CL to 3e-6. JSBSim's `do_simple_trim` converges to **Nz = 0.99093, not 1.0**, and atisim
needs 0.95% more CL for a true 1 g -- worth +0.0627 deg of alpha against the +0.0535
observed. So the vortex comparison must be driven from a MATCHED STATE rather than from
each engine's own trim, the way the 737 sweep already is.


### Session 18 — the AtiSim rename, and correcting five stale claims

**The project is now AtiSim.** The import package is `atisim`, the distribution is `atisim`,
and the display name in the docs and generated reports is **AtiSim**. 1,510 occurrences
across 156 tracked files, plus `git mv flightsim atisim` and
`docs/summary/flightsim-summary.pdf` to `atisim-summary.pdf`.

**Two categories were deliberately left at the old name**, because renaming them would
falsify a record rather than update one:

- **`audit/` in full** — 67 files. The evidence `.md` files quote captured stdout, the logs
  *are* captured stdout, and `G-archaeology.md` names the real branch
  `claude/flightsim-sweep-ui-graphs-d4d036`, which still exists under that name. Decisively,
  `audit_evidence/E-falsification-scripts/common.py` hard-codes a `sys.path.insert` pointing
  at the **main checkout**, which still contains `flightsim/`. Renaming those imports would
  have broken scripts that currently work.
- **`FIX_PROMPT.md` and `analysis-ui-investigation-prompt.md`** — archived prompts, tracked
  as records by commit `78ca398`. What was asked is not editable after the fact.

**The rename turns this project's worst failure mode into a loud one.** §10's table records
that a wrong-tree import resolves silently to the main checkout with no error at all. The
editable install still maps `flightsim`, and nothing installs `atisim`, so on this branch
that same mistake now raises `ModuleNotFoundError`. Measured, not assumed. It reverts to the
old hazard the moment anyone runs `pip install -e .` under the new name — §10 carries both.

**Five stale claims corrected**, all of the same kind: a documented fact that a later commit
made false, with nothing failing to mark it.

| Where | Claimed | Actually |
|---|---|---|
| `aircraft.py` x2, `aero.py` x1 | α̇ is applied for the **wind only** and `Cmq` stays folded at −43.0 | `dynamics.derivatives` resolves the aircraft's own α̇ too (commit `cb23a9e`); the 737 carries a **bare** `Cmq` = −27.0 |
| fidelity spec, "Outcome" + "Explicitly not done" | item 4 (AERORP) and the α̇ solve are **not done** | both were done, in `fdbeff2` and `cb23a9e`; §4 and §5 of that same spec describe them |
| §4 ledger, session 17 | derivative table, layer 1 lateral, layer 3, layer 4 | written at `1dcc5d4`, before four commits changed all of them — superseded above, not deleted |
| layer-4 test docstring | doublet u 0.296 / w 0.410; Coriolis floor u 0.403, v 0.012 | measured u 0.507 / w 0.558; the reference's own diagnostics say u 0.409, v 0.028 |
| §10 | `pytest -q` runs **322 tests** | **626 passed, 1 skipped**, ~8 min |

`pyproject.toml` also gained the `ref = ["jsbsim"]` extra its own design spec's architecture
table has claimed since `f04bfac` and which was never added. `scripts/gen_jsbsim_reference.py`
is confirmed the only file importing `jsbsim`.

**Still missing, and not invented here:** session 17 has a §4 ledger entry but **no §9 session
log entry**, which this document's own update rules require. Writing one retroactively would
be fabricating a record of work this session did not do, so it is flagged instead.

**Deliberately not done:** none of §4's "still open" items — the phugoid's 6.58% against a
throttle-linear thrust model, `CDde`, Mach scheduling, banked trim, or a condition above
M 0.8 that would test `wave_drag` at all. This session changed no physics; the suite is
626 passed / 1 skipped before and after, which is what makes that claim checkable.

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
root**; the scripts import `atisim` from the editable install, not from `scripts/`.

### Before you trust a run, check which tree it imported

This is the one failure mode here that produces **no error message at all.** Worktrees do
not get their own `.venv`; they share the main checkout's, and that editable install's
finder maps `atisim` to the **main checkout's** `atisim/` permanently. The finder is
*appended* to `sys.meta_path`, so it is reached only once `sys.path` has already failed —
and whether `sys.path` succeeds depends on how the process was started. Measured from a
worktree, all four rows:

| how it is run | what `import atisim` resolves to |
|---|---|
| `.venv/Scripts/python.exe -m pytest`, cwd = **worktree root** | **the worktree.** `-m` puts cwd on `sys.path` first |
| `pytest` / `pytest.exe`, cwd = worktree root | **the main checkout.** The console script does not put cwd on `sys.path`, and `atisim/tests/conftest.py` imports `atisim` before pytest's own insertion helps. `sys.path[0]` *is* the worktree by the time a test body runs, which is why this one looks fine and is not |
| anything, cwd = **any other directory** — `notebooks/`, `scripts/` | **the main checkout** |
| any of the above with `PYTHONPATH` set to the **absolute** worktree root | **the worktree** |

So the documented `.venv/Scripts/python.exe -m pytest -q` is safe from the worktree root,
and **nothing else in that table is.**

> **The `atisim` rename changes this, and for the better — read this before trusting the
> table above.** The table describes the package when it was called `flightsim`. The editable
> install still maps **`flightsim`** to the main checkout, and nothing installs `atisim`, so on
> this branch every row that used to resolve silently to the main checkout now raises
> `ModuleNotFoundError: No module named 'atisim'` instead. Measured, from `C:/Users/mateusz`:
> `import atisim` raises `ModuleNotFoundError`; `import flightsim` still returns
> `.../Claude_Flight_Sim/flightsim/__init__.py`.
>
> **The failure mode that produced no error message now produces one.** That holds only until
> someone runs `pip install -e .` from a tree carrying the new name, which re-creates exactly
> the old hazard under the new spelling — at which point this table applies again verbatim.
> The one-line check below is still the thing to run, and is now spelled `atisim`. The check costs one line, run from the directory you
are about to run the suite from:

```
.venv/Scripts/python.exe -c "import atisim; print(atisim.__file__)"
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
| `.venv/Scripts/python.exe -m pytest -q` | **811 passed, 1 skipped, 12m05s** (measured session 26; 807 at session 25; it was 788 at session 24 and **758 measured session 23b**; the 626 this row claimed was stale by five sessions, and the 322 before that by several more — this row has now been wrong twice, so re-measure it rather than trusting it). The first thing to run and the only complete statement of what works. `testpaths` is set in `pyproject.toml`, so the bare command collects `atisim/tests`. |
| `.venv/Scripts/python.exe scripts/sanity.py` | **The ladder, for a reader who does not yet trust the model.** Twelve cases from degenerate inputs upward — zero the wind, zero a coefficient so a motion becomes impossible, then signs, then hand-computable numbers, then structural properties. Every expected value is derived by hand in the source and printed beside the model's answer, so it is read rather than trusted. Ends with the item 08 convention probe, which is a measurement rather than a pass/fail. |
| `.venv/Scripts/python.exe -m pytest --nbval-lax notebooks/ -q` | **The second gate.** Executes `notebooks/solver-validation.ipynb` so it cannot rot. Needs the `dev` extra (`jupyter`, `nbval`). Deliberately *not* in `testpaths` and `--nbval-lax` is deliberately *not* in `addopts`: that would make every `pytest` run fail with "unrecognized arguments" wherever nbval is absent. **Run it from a worktree with an ABSOLUTE `PYTHONPATH`** — nbval starts the kernel with its cwd in `notebooks/`, so a relative `PYTHONPATH=.` resolves to the wrong directory and `atisim` silently loads from the main checkout. |
| `.venv/Scripts/python.exe scripts/checkpoint.py` | 747 only, no flags. Trim residuals, 60 s fixed-control hold, longitudinal modes against CR-2144 Table IX-5. |
| `.venv/Scripts/python.exe scripts/tune.py --aircraft cherokee` | Autopilot step responses for one aircraft. Exits non-zero on failure, so it is usable as a gate. |
| `.venv/Scripts/python.exe scripts/fly.py --aircraft cherokee --save runs/a.npz` | Interactive flight, basic-T cockpit plus a flight-test overlay. |
| `.venv/Scripts/python.exe scripts/fly.py --wind hannibal` | The same, hand-flown into the Parks vortex array. The panel counts the range down. |
| `.venv/Scripts/python.exe scripts/vortex.py --case hannibal --png runs/v.png` | Flies the 747 through the Parks vortex array, the Wingrove updraft, and an elevator pushdown, and draws the analysis figure with all three Fig. 8 categories. This is the turbulence path. Prints each point's Δθ, Δn and peak \|α\| with its band, then whether the ordering holds. |
| `.venv/Scripts/python.exe scripts/microburst.py --png runs/mb.png` | Flies the Cherokee through an Oseguera & Bowles microburst at 300 m and reports the 1 km average F against its thrust authority. Cuts the run at one wingspan above the ground and says so. |
| `.venv/Scripts/python.exe scripts/leewave.py --png runs/lw.png` | Flies the 747 through a Doyle et al. lee wave and compares the Bowles F-factor against the aircraft's own `(T−D)/W`. Prints both of the source's flight legs and which of them the engines can cover. |
| `.venv/Scripts/python.exe scripts/analyse.py runs/a.npz` | Replays a saved `.npz`. Accepts several files; `--png DIR` writes instead of showing. |
| `.venv/Scripts/python.exe scripts/vortex.py --artifacts runs/analysis` | The same run, **also written as a run artifact per encounter** — Parquet series plus `meta.json` and `checks.json`. Prints `checks ok` or names the checks that failed. Needs the `ui` extra. Until this flag existed, every number in §4's encounter tables came from a run that did not survive the script that produced it. |
| `.venv/Scripts/python.exe -m atisim.apps.sweep runs/analysis` | **The analysis UI.** Sweep view per run — provenance header with caveats, check badges, the causal strip stack, the 3D field with the trajectory through it, Fig. 8, and n_z-vs-α — plus a shared time cursor: click any strip and every panel, the 3D marker and the readout move to that sample together. Needs the `ui` extra (`pip install -e .[ui]`). |
| `.venv/Scripts/python.exe scripts/summary.py docs/summary/atisim-summary.pdf docs/summary/panel.png` | Rebuilds the plain-English summary PDF (14 pages). The parts that are *computed* cannot drift from the code — the vortex figures call `wind.vortex_wind`, and the aircraft table reads `CRUISE`. **The prose and the summary statistics are literals and can**: the test and line counts were stale by session 7, and four page cross-references were wrong by session 8. The page numbers are now generated from `PAGE_ORDER` with a build-time count check; the statistics are still literals. Re-run it after anything that changes those. |
| `C:/Users/mateusz/AppData/Local/Programs/Python/Python310/python.exe` `scripts/gen_jsbsim_747.py` | **Recovers the `boeing747_jsbsim` entry from the running B747.** Needs jsbsim, so it runs under the reference interpreter above, NOT the project venv. Writes `atisim/tests/data/jsbsim_747_reference.xml`. Run only when the recovery condition changes; drift shows up in `git diff`. |
| `C:/Users/mateusz/AppData/Local/Programs/Python/Python310/python.exe` `scripts/gen_jsbsim_vortex_reference.py` | **Freezes JSBSim's answer to the three vortex cases.** Same interpreter, same reason. Writes `atisim/tests/data/jsbsim_vortex_reference.xml`. |
| `.venv/Scripts/python.exe scripts/vortex_compare.py --png runs/vc.png` | **The cross-code vortex comparison.** Flies atisim through the identical field the frozen reference was generated from and reports where the two engines part, against Wingrove & Bach's own g-loads. Imports no jsbsim. |
| `.venv/Scripts/python.exe scripts/vortex_diagnose.py` | **Why the comparison's two large errors are large.** Three experiments: the same start state flown in still air, atisim flown from its own trim, and a one-lever-at-a-time sweep against the DFDR. Imports no jsbsim. |
| `PYTHONPATH=<abs worktree root> .venv/Scripts/python.exe scripts/digitise_tm102186_fig6.py --outdir runs/cat` | **The recorded g trace (session 27).** Reads TM-102186 Fig. 6's G LOAD panel out of `Reference_papers/19890016606.pdf` at 600 dpi, column by column, as the top and bottom of the ink — nothing fitted, nothing smoothed. Prints the three checks (the paper's own band, a **negative control** on the vertical-wind panel, and the gust spacing) and writes `10-tm102186-fig6.png` plus `tm102186-fig6-gload.csv`. §4 has what it found. |
| `PYTHONPATH=<abs worktree root> .venv/Scripts/python.exe scripts/digitise_mil_f_8785c_fig7.py --outdir runs/cat --pdf refs/MIL-F-8785C.pdf` | **The severe-turbulence σ_w chart (session 27).** Digitises all nine curves of Fig. 7 from printed p. 49, flagging where two share **one stroke of ink** rather than reading a number out of a merge. Settles `mil_f_8785c_sigma_w_exceeds_the_mehta_ceiling`. Writes `11-mil-f-8785c-fig7.png` and `mil-f-8785c-fig7-lines.csv`. **`--pdf` is required from a worktree** — `refs/` is gitignored and lives only in the main checkout. |
| `PYTHONPATH=<abs worktree root> .venv/Scripts/python.exe scripts/cat_bounds.py --outdir runs/cat` | **The bounding experiments (session 23 follow-up).** What the point-sampled gust, the strip path and the step size cost on the Mehta run; what Dryden intensity would close the residual load gap; and Lester's Greenland 747 against a lee wave, inverted on both the g-load and the altitude gain. Same `PYTHONPATH` rule. |
| `PYTHONPATH=<abs worktree root> .venv/Scripts/python.exe scripts/lateral.py --outdir runs/cat` | **The lateral dimension (session 24, phase 1).** Reconciles `wind.line_vortex_wind` against `wind.vortex_wind` along the flight path, shows where the oblique difference is, then flies Mehta's field three ways -- point, line, and line with strip-integrated loads -- and reports the bank, sideslip and rolling gust rate the project could not previously see. Writes `08-lateral.png`. Same `PYTHONPATH` rule. |
| `PYTHONPATH=<abs worktree root> .venv/Scripts/python.exe scripts/cat_ensemble.py --outdir runs/cat` | **Fig. 8 with error bars (session 23d, section 7 step 6).** Superposes a Dryden layer at the SOURCED sigma_w range from Mehta's residual and reports whether the vortex/updraft/manoeuvre ordering survives, and by how much margin on each of Fig. 8's two axes. Writes `07-ensemble.png`. Same `PYTHONPATH` rule. |
| `PYTHONPATH=<abs worktree root> .venv/Scripts/python.exe scripts/cat_spectra.py --outdir runs/cat` | **Response spectra and load exceedance (session 25, phase 2).** Three limbs: Mehta's headline field as a `n_z` spectrum against the aircraft's short period and the array's four core-passage frequencies; Yoshimura et al. 2023's protocol -- N virtual flights through `wind.dryden_field`, spectra averaged, peak against the airframe's own frequency -- which settles a sealed prediction; and the first load-exceedance curve, both signs, with N in the denominator. `--seeds` defaults to 32; **session 27 ran `--seeds 151` to match Yoshimura and §4 records the result, which is cleaner — use 151 when the peak location matters.** Writes `09-spectra.png`. Same `PYTHONPATH` rule. |
| `PYTHONPATH=<abs worktree root> .venv/Scripts/python.exe scripts/cat_uncertainty.py --outdir runs/cat` | **The Hannibal comparison with error bars (session 23c).** Measures the gust SPACING against TM-102186's "about 5 sec apart" -- the one channel the identification did not set -- converts Mehta's own Eq. (A3) cost into an RMS wind residual and decomposes it against Lester's reconstruction error, then flies the propagated `V0` and `r0` band and a gust-strength sweep to show the peak load is saturated. Writes `06-uncertainty.png`. Same `PYTHONPATH` rule. |
| `PYTHONPATH=<abs worktree root> .venv/Scripts/python.exe scripts/cat_validation.py --outdir runs/cat` | **The CAT source pass (session 23).** Flies Mehta 1987's five-vortex Hannibal field, reproduces TM-102186 Fig. 8's three-aircraft ordering and tests its stated mechanism across the whole registry, compares the 747's short period at a third CR-2144 flight condition, and grades every run on Misaka's `σ_n`. Prints every number and writes four figures. **`PYTHONPATH` is mandatory** — `python scripts/…` resolves `atisim` to the main checkout, which this script detects and prints on its first line. |
| `docs/summary/jsbsim-atisim-vortex-report.html` | **The written comparison** — the numbers above with the reasoning, the figure, and what the result does and does not establish. Not generated; edit it when the numbers move. |

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
import atisim                                   # enables x64 — import first
from atisim import trim, integrate, autopilot as ap
from atisim.aircraft import REGISTRY, CRUISE

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
