# Cross-code verification against JSBSim, via the 737 — Design Spec

Date: 2026-08-20

## Purpose

Every check in `verification.py` and `validation.py` is either a closed-form identity or a
comparison against a *published table*. Nothing yet compares this project against another
**executing** 6-DOF implementation. That is a distinct class of evidence: a transcription error,
an axis-convention slip or a sign error in the aero build-up can satisfy every analytic identity
and still be wrong, because the identity is symmetric in the error.

JSBSim 1.3.1 is installed as a Python extension (GitHub build 1837, commit
`3b25f25e49b42d0489c04ac805674fc1450ca579`) and ships a 737 with reference scripts. It runs
headless and is fully drivable through its property tree. This spec uses it as an independent
reference **engine** — not as a reference **dataset**.

**What this claims:** flightsim's aerodynamic build-up, trim solver, linearisation and integrator
agree with an independent, mature 6-DOF engine when both are fed the same coefficients at the same
state.

**What this does not claim:** that the JSBSim 737 is a correct 737. See "Source qualification".

## Decisions taken during brainstorming

| Question | Decision |
|---|---|
| Framing | **Cross-code verification, not validation.** The 737's data quality never enters the claim. |
| Aero mapping | **Linearise at cruise.** Recover equivalent constant derivatives by finite-differencing JSBSim about its own trimmed cruise point; assert agreement in a stated neighbourhood. |
| Comparison depth | **Four layers**: coefficients, trim, modes, trajectory. |
| Where the data lives | **Hard-coded `_boeing_737()` in project style, plus a frozen reference XML** of JSBSim's own outputs. Tests never import `jsbsim`. |
| On disagreement | **Diagnose and report; fix only unambiguous flightsim defects.** Nothing changes on JSBSim's authority alone. |
| Thrust model | **Add a Mach ram term to flightsim** rather than prescribing thrust into the trajectory layer. |

## Source qualification

The project's existing rule is that a source is used for what it can support, and the support is
stated. Applying it here gives an unusually sharp answer, because the source disqualifies itself.

`737.xml`'s own `<fileheader><note>` says the model was built from public data, technical reports,
textbooks **and guesses**; that validation extends only to the extent that it "seems to fly right";
and that it is for "educational and entertainment purposes only."

So the 737 dataset sits in **no tier at all**. It cannot support any claim about a real 737, and
this spec makes none. What is being used is JSBSim's *engine* — its equations of motion, trim
algorithm and integrator — which is mature, widely used and independently developed. Feeding two
codes the same coefficients and comparing makes the dataset's quality irrelevant: a disagreement is
a defect in one of the two implementations regardless of whether the numbers describe any real
aeroplane.

This is the same logical move already made for CR-2144 ("if the derivatives were 10% out, the
solver must still reproduce the source's own outputs from the source's own inputs"), pushed one
step further: here even the source's outputs are generated on demand rather than published.

The `boeing737` registry entry carries this in its docstring, so it is never later mistaken for a
qualified source in the CR-2144 sense.

## Why derivatives are recovered from the engine, not read from the XML

This was measured during brainstorming and is the single most important implementation decision.

JSBSim applies aerodynamic forces at the **AERORP** (x = 625 in) but takes moments about the **CG**
(x = 610.8 in with the modelled fuel load). The 14.2 in offset is 1.183 ft, or 0.096 c̄. The
force-times-offset moment is therefore folded into every pitch derivative the *engine* exhibits,
while the XML's `<function>` constants describe only the coefficient before that offset is applied.

Measured, at 30,000 ft / M 0.78:

| Derivative | XML constant | Engine behaviour | Note |
|---|---|---|---|
| `Cma` | −0.600 /rad | **−1.0637 /rad** | **77% difference.** Dominated by `CLa × 0.096 c̄` |
| `Cmde` | −0.849 /rad (M 0.78 of the table) | −0.8710 /rad | `CLde × 0.096 c̄` = 0.019 accounts for most of the 0.022 gap |
| `Cm` at α = 0, δe = 0 | 0 (no `Cm0` term exists) | **−0.01238** | Lift and drag acting at the offset |
| `CLa` | +4.3478 /rad (table slope) | +4.3478 /rad | Exact — no offset effect on lift |
| `CLde` | +0.200 /rad | +0.2000 /rad | Exact |

flightsim applies aero forces at the CG. Transcribing `Cmalpha = −0.6` would therefore have
produced a 737 with 56% of the correct pitch stiffness, and the resulting mode mismatch would have
been indistinguishable from a defect in `dynamics.py`. Recovering from the running engine folds the
offset in automatically and is the only mapping that is correct by construction.

The arithmetic closes independently: at α = 0, offset lift contributes −0.2006 × 0.096 = −0.0193,
and drag acting 4.93 ft above the CG contributes roughly +0.009 nose-up, for ≈ −0.010 against the
−0.0124 measured. Nothing unexplained remains.

## Architecture

| Path | Contents |
|---|---|
| `flightsim/aircraft.py` | `_boeing_737()`, `REGISTRY["boeing737"]`, `CRUISE["boeing737"]`. Existing style: literal numbers, unit conversions, per-line provenance |
| `scripts/gen_jsbsim_reference.py` | Drives JSBSim and writes the reference XML. **The only file that imports `jsbsim`** |
| `flightsim/tests/data/jsbsim_737_reference.xml` | Frozen: trim point, recovered derivatives, A/B matrices, trajectory samples, tolerance derivations, JSBSim version and commit |
| `flightsim/tests/test_jsbsim_737.py` | The four layers. Reads the XML. Never imports `jsbsim` |
| `scripts/jsbsim_report.py` | Builds the results PDF |
| `docs/summary/jsbsim-737-report.pdf` | Plots and tables of every layer's results |
| `pyproject.toml` | New `ref = ["jsbsim"]` extra |

The report follows the rule already written into `scripts/summary.py` and
`scripts/turbulence_report.py`: **every number is either read from the reference XML or computed
at build time by calling the project's own code. Nothing is typed in from memory.** A regression
therefore changes a figure in the report rather than leaving a stale claim behind. Contents:
the recovered derivative set against JSBSim's XML constants; per-coefficient error across the
layer-1 sweep; trim comparison for both trim modes; mode eigenvalues on the complex plane for both
engines; trajectory overlays with divergence envelopes; the thrust ram fit against the CFM56 table;
and the input-condition matching evidence above.

The `ref` extra follows the reasoning already written for `ui`: the simulator and every script in
`scripts/` must keep working without it. The test suite gains no dependency on a compiled package,
and reference drift is visible in `git diff`.

## The 737 aircraft entry

Recovered by central-differencing JSBSim about the trimmed `737_test` condition: 30,000 ft, M 0.78,
γ = 0 → α = 1.9516°, δe = −0.0516268 rad, throttle = 0.7709, 4980.41 lbf/engine, W = 107,000 lb,
q̄ = 268.163 psf, V = 775.982 ft/s.

Geometry and mass come straight from the XML: S = 1171 ft², b = 94.7 ft, c̄ = 12.31 ft
(AR = 7.6585); Ixx = 591572.3, Iyy = 1539552.7, Izz = 1986235.4, Ixz = 19109.1 slug-ft² as reported
by the engine with fuel aboard.

**`Ixz` sign needs care, not assumption.** `<mass_balance>` carries
`negated_crossproduct_inertia="true"`, and `inertia_tensor` already takes `Ixz` in the
positive-forward-up sense and negates it internally. The implementation must establish which
convention the reported +19109.1 is in and assert it, rather than pick one.

**Six derivatives are absent from JSBSim's 737** — `CLq`, `CYp`, `CYr`, `CYdr`, `Cnp`, `Cnda` — so
flightsim gets 0.0 for each. This is agreement, not approximation: both engines then compute the
same thing. The generator **asserts the finite difference returns zero** for all six rather than
assuming it, converting six silent-mismatch risks into six checks. Note in particular that
JSBSim's 737 has `Cndr` and `Cldr` but **no `CYdr`**: rudder produces yaw and roll moments but no
side force. That is physically odd and is reproduced deliberately.

Induced drag: JSBSim uses `CDi = 0.043·CL²`. Setting `e` so that `CL²/(π·e·AR)` matches gives
**e = 0.9666**, making this term identical rather than merely close.

## Thrust model — adding a Mach ram term

Measured during brainstorming: the CFM56's `MilThrust` table at M = 0 tracks `(ρ/ρ₀)^1.0` to within
0.6% up to 30,000 ft (2.2% at 40–50k). flightsim's existing `throttle · Fmax · (ρ/ρ₀)^n` already
reproduces JSBSim's **altitude** behaviour with `thrust_lapse = 1.0` and no change whatever.

The genuine gap is **Mach**, which flightsim has none of. At 30,000 ft, normalised to M = 0:

| Mach | 0.0 | 0.2 | 0.4 | 0.6 | 0.8 | 1.0 |
|---|---|---|---|---|---|---|
| ram ratio | 1.000 | 0.954 | 0.960 | 1.016 | 1.121 | 1.277 |

At the M 0.78 cruise point that is an ~11% thrust error — which in a trajectory comparison appears
as a slow speed divergence indistinguishable from a drag defect.

**Change:** one new `Aircraft` field carrying the ram fit, applied in `aero.py` and `dynamics.py`,
defaulting to a **neutral value**. Every existing aircraft — 747, 747-approach, Cherokee, Cessna,
Navion, and the `conftest` fixture — is then **bit-for-bit unchanged**, which is asserted rather
than established by inspection. `test_conservation.py`'s explicit field list gains the new name.

The fit is over the cruise band and its residual is recorded in the reference XML. A single
low-order form cannot capture the M 0.2 dip, so it is accurate over roughly M 0.6–1.0 and wrong at
low speed — see Limitation 7.

With this in place the trajectory layer **verifies** thrust rather than prescribing it, and
`max_thrust` stops being a registry number no layer asserts.

## Matching the input conditions

Checked before implementation, because a systematic offset in the *inputs* biases every layer at
once and looks like a defect in the code.

### Sign conventions: all sixteen agree, no flips needed

Measured directly against JSBSim, not inferred from documentation:

| Input | JSBSim behaviour | flightsim convention | Agree? |
|---|---|---|---|
| β > 0 | `v-fps` = +40.6, `aero/beta-rad` = +0.0523 | `beta = arcsin(v/V)`, relative wind from the right | yes |
| β > 0 → side force | `fwy-aero` = −16424 (`CYb` = −1) | `CYb < 0` | yes |
| β > 0 → roll, yaw | `l-aero` < 0, `n-aero` > 0 | `Clb < 0`, `Cnb > 0` | yes |
| aileron > 0 | left = +0.175, right = −0.175, `l-aero` > 0 | δa positive → right-wing-down roll, `Clda > 0` | yes, mapping δa ≡ `left-aileron-pos-rad` |
| rudder > 0 | `n-aero` < 0, `fwy-aero` = **0.0** | `Cndr < 0`, `CYdr > 0` | sign yes; JSBSim has **no** `CYdr` at all |
| elevator > 0 | `m-aero` < 0, `fwz-aero` > 0 | δe positive TE-down, `Cmde < 0`, `CLde > 0` | yes |
| wind-axis forces | `fwx`, `fwz` both > 0 at α = 5° | `CD`, `CL` positive | yes |

The AERORP offset explains the lateral moments as well as the pitch ones: at β = 3°, side force
acting 4.925 ft above the CG accounts for the roll moment to **0.15%** (−220388 predicted vs
−220722 measured) and acting 1.183 ft aft of the CG accounts for the yaw moment to **0.17%**
(422530 vs 423259). Recovering derivatives from the running engine therefore handles all three
axes, not just pitch.

### Atmosphere: one systematic error found, diagnosed, and neutralised

flightsim's ISA and JSBSim's disagree, and the error grows with altitude:

| altitude | 0 | 10,000 ft | 20,000 ft | 30,000 ft | 40,000 ft |
|---|---|---|---|---|---|
| density error | −0.0008% | −0.0158% | −0.0656% | **−0.1592%** | **−0.3677%** |

**Cause, confirmed exactly:** the ISA is defined on *geopotential* altitude; `atmosphere.py` uses
*geometric*. Predicted temperature errors from that hypothesis are −0.0035 / −0.0153 / −0.0373% at
10/20/30k against measured −0.0035 / −0.0153 / −0.0373% — agreement to four decimal places. The
implied 43 ft offset at 30,000 ft is recovered independently below.

This matters because q̄ ∝ ρ, so a 0.16% density bias is a 0.16% bias on *every force in every
layer*, in the same direction — precisely the systematic error that would be misread as a
modelling defect. It consumes about 22% of the layer-2 trim tolerance on its own.

**Neutralised by matching on density rather than on altitude.** Altitude is not itself an input to
the physics; it enters only through ρ and a. The harness therefore solves for the geometric
altitude at which flightsim's density equals JSBSim's, and runs flightsim there:

| JSBSim altitude | flightsim altitude | shift | ρ error | a error |
|---|---|---|---|---|
| 30,000 ft | 29,956.78 ft | −43.22 ft | −0.159% → **2.4e-14** | −0.0185% → **+0.0002%** |
| 40,000 ft | 39,923.36 ft | −76.64 ft | −0.368% → **3.9e-13** | +0.0001% → **+0.0001%** |

Density matches to machine precision and the speed-of-sound residual improves by ~90×, to 2e-6
relative — small enough that it cannot be confused with anything. The −43.22 ft shift is the
geopotential correction, arrived at independently, which closes the diagnosis.

The underlying geopotential/geometric difference is a defect in **existing** code and is therefore
filed as a separate finding rather than fixed here, per Limitation 5.

## The four comparison layers

Common protocol, forced by a finding: JSBSim's FCS **injects control deflections that were never
commanded**. With zero rudder command at M 0.78, `fcs/rudder-pos-rad` reads 0.0035 — the
scheduled-gain yaw damper, active above M 0.11. Therefore every layer **reads back** JSBSim's
actual α, β, rates and surface positions and feeds *those* to flightsim. Commanded values are never
assumed to have taken effect.

### Layer 1 — coefficients

Sweep α, β, p, q, r, δe, δa, δr; compare all six of `CL, CD, CY, Cl, Cm, Cn` from
`aero.coefficients` against JSBSim's `forces/fw*-aero-lbs` and `moments/*-aero-lbsft`
non-dimensionalised by the engine's own q̄, S, b, c̄. Isolates the build-up from everything else.

### Layer 2 — trim

JSBSim `do_simple_trim` against `trim.trim`. **Longitudinal (mode 0) and steady turn (mode 5)**;
the turn case exercises lateral-directional coupling and the `Ixz` term that longitudinal trim
never touches. Tolerances: α to 0.05°, δe to 0.002 rad, thrust to 1%.

### Layer 3 — modes

JSBSim's `do_linearization` writes a 12-state Scilab model. The state ordering is
`[vt, α, θ, q, β, φ, p, ψ, r, lat, lon, h]` and inputs are `[throttle, aileron, elevator, rudder]`
— derived from the integrator rows of A (θ̇ = q; ψ̇ = r/cos θ, coefficient 1.00058 = 1/cos 1.95°;
ḣ = vt(θ − α)), **not** from memory of JSBSim's source.

The parser **re-derives this ordering at parse time from those same structural rows** and fails
loudly if it does not hold, so a future JSBSim reordering cannot silently compare the wrong states.
Longitudinal block is `[0:4]`, lateral `[4:9]`; compared against `validation.longitudinal_matrix`
after the `(vt, α) → (u, w)` transform, on eigenvalues.

### Layer 4 — trajectory

20 s from the trimmed state: elevator doublet and rudder kick.

- **Surface positions are prescribed to both engines.** JSBSim's rudder command is pre-compensated
  each step by the yaw-damper output so the *surface* follows the commanded history; the harness
  asserts `fcs/rudder-pos-rad` hits target each step. This neutralises the FCS without modifying
  the stock model.
- **Thrust is verified, not prescribed**, with throttle held at its trim value.
- **Earth-model differences are measured, not absorbed.** flightsim is flat-Earth with constant
  g = 9.80665; JSBSim is WGS-84 with inverse-square gravity — a 0.28% g difference at 30,000 ft
  (9.779 vs 9.807) plus Coriolis ≈ 0.025 m/s² at 47° latitude. JSBSim's gravity model is set to
  constant-g, and the residual Coriolis contribution is quantified by running lat 0° against
  lat 47° and reported as a number.

### Tolerances

Derived, not tuned. flightsim's 737 is a linearisation, so it matches exactly only at the reference
point; away from it the residual is dominated by known nonlinearities (the `CD0(α)` table
curvature, the `CL(α)` table rolling over). Each tolerance is set to that analytically-computed
residual plus margin.

To make this auditable rather than a promise: **every tolerance carries its derivation as a comment
and its computed value in the reference XML**, so a tolerance widened past its derivation shows up
as a mismatch between the two rather than as a still-green test.

## Limitations — read this before reusing anything here

### 1. The `boeing737` entry is valid only near cruise, and this is its only warning

**This is the most important limitation in this document.**

The 737 derivative set is a linearisation about 30,000 ft / M 0.78 / α = 1.95°. It is *not*
equivalent to the project's other registry entries. `boeing747`, `cherokee` and `cessna172` are
linear derivative sets from CR-2144 and Nelson, valid across the ordinary linear range. The 737 is
a local fit to a **nonlinear** model, and away from its recovery point it degrades in ways that are
specific and known:

- `CL(α)` in JSBSim is a 4-point table that **peaks at CL = 1.20 near α = 13° and then falls**. The
  linear `CL0 + CLa·α` keeps climbing. Above roughly 10° the 737 entry does not merely lose
  accuracy — it has no stall behaviour at all and will report lift the source model does not have.
- `CD0(α)` in JSBSim runs 0.021 at α = 0 to 0.042 at α = 15°. The constant `CD0` is the α = 1.95°
  value, so drag is progressively under-predicted as α departs from cruise.
- Wave drag uses this project's Korn/Lock law rather than JSBSim's `CDmach` table. These agree on
  onset near M 0.79 by construction and diverge above it.
- The Mach ram thrust fit holds over roughly M 0.6–1.0 and is wrong at low speed.

**Consequence:** flying `boeing737` at, say, 5,000 ft and 200 kt produces numbers that are wrong
without anything failing, warning or logging. Nothing in the code prevents this.

**Mitigation taken:** documentation only, deliberately — no runtime guard is built. The recovery
condition and the validity band are recorded in the `_boeing_737()` docstring, in the `CRUISE`
entry, in the reference XML, and here. Anyone extending this to other flight conditions must
re-recover the derivatives at that condition; the generator script is written so that this is a
parameter change rather than a rewrite.

### 2. Two of the six shipped 737 reference scripts are not used

`B737_Runway.xml` and `b737_runway_new.xml` are ground-roll cases. flightsim has no landing-gear or
ground-reaction model, so they are unmodellable rather than merely out of scope. The report states
this explicitly rather than quietly presenting four-of-six as complete coverage.

### 3. The frozen reference XML can go stale silently

Tests read the checked-in XML and never import JSBSim, which is what makes them fast and portable.
If a future JSBSim release changes a result, nothing notices until someone runs the generator by
hand. The byte-for-byte regeneration check is a manual gate, not a CI one. An opt-in live-compare
marker was considered and **not** built, to avoid a second code path.

### 4. Tolerance derivations are self-reported

The XML-versus-comment cross-check above makes a *widened* tolerance visible. It does not make a
*weak derivation* visible. A derivation that was hand-wavy from the start will pass its own check.

### 5. Defect fixing is capped

Unambiguous defects in the **new** code are fixed here. Anything found in existing code — for
instance a lateral sign-convention disagreement that would ripple into the 747 results PROJECT.md
records — is filed as a finding for a separate session rather than fixed on this branch, so the
diff stays reviewable.

### 6. Agreement with JSBSim is not evidence about reality

Both engines can be wrong in the same way, and both descend from the same textbook formulations of
rigid-body flight dynamics. This layer catches implementation defects. It says nothing about
whether either engine models a real 737.

### 7. The thrust ram fit is band-limited

Stated above; repeated here because it is a second instance of Limitation 1's pattern and will bite
the same way.

## Success criteria

1. `pytest flightsim/tests/test_jsbsim_737.py` passes **with JSBSim not installed**.
2. `python scripts/gen_jsbsim_reference.py` reproduces the checked-in XML byte-for-byte on a machine
   with JSBSim 1.3.1.
3. The existing suite stays green, and the four pre-existing aircraft plus the `conftest` fixture
   are **bit-for-bit unchanged** by the thrust-model field — asserted, not inspected.
4. All six absent-derivative assertions hold (finite difference returns zero).
5. The linearisation state-ordering derivation holds against the structural rows of A.
6. Every disagreement beyond tolerance is either fixed or documented with a measured magnitude.
7. Limitation 1 appears in the `_boeing_737()` docstring, the `CRUISE` entry and the reference XML.
8. Density agrees with JSBSim to better than 1e-10 relative at every compared condition, asserted
   by the harness rather than assumed, so no layer carries a q̄ bias.
9. `scripts/jsbsim_report.py` builds `docs/summary/jsbsim-737-report.pdf` from the reference XML and
   live project code, with no number typed in from memory.

## Risks

| Risk | Mitigation |
|---|---|
| Axis/sign convention mismatch between JSBSim's wind-axis forces and this project's | Layer 1 sweeps each axis independently, so a sign error localises to one coefficient instead of cancelling in trim |
| The FCS injects deflections not commanded | Read-back protocol; asserted each step in layer 4 |
| `Ixz` sign convention | Established and asserted rather than assumed |
| JSBSim linearisation state order changes | Re-derived from structural rows at parse time; fails loudly |
| Scope growth from defects found in existing code | Capped by Limitation 5 |
| The 737 entry is later reused outside its band | **Not mitigated in code.** Documentation only — see Limitation 1 |

## What changed during implementation

Recorded as deltas rather than by rewriting the sections above, so the design
record and what actually happened stay separable.

### Findings the design did not anticipate

**JSBSim's `do_linearization` is CLOSED-LOOP.** Its FCS is inside the exported model and nothing
in the output says so. Against the bare airframe the lateral comparison reads as catastrophic —
Dutch roll ζ 0.101 against 0.344, spiral 127.6 s against 16.7 s, a 664% disagreement — and the
obvious conclusion is that this project's lateral dynamics are broken. They are not. 737.xml's yaw
damper feeds yaw rate to the rudder with unit gain above M 0.11, geared by 0.35 rad, adding
`ΔCnr = Cndr × 0.35 × 2V/b = −1.147` against a bare `Cnr` of −0.350. Folding that in gives ζ 0.338
against 0.344 and spiral 16.61 s against 16.69 s. The pitch and roll channels have no feedback, so
the longitudinal comparison needs no correction and lands at 0.04%. A second test asserts the
correction is load-bearing, so it cannot quietly become wrong.

**`run()` integrates as well as running the FCS.** At JSBSim's default 1/120 s step, a pitch-rate
perturbation drifts α far enough to manufacture `CLq` = +4.60 where 737.xml defines none. The
absent-derivative assertion caught it; the predicted artifact for that step is +4.57, so the
assertion was right and the method was wrong. Settling on a 1e-6 s step leaves the FCS converged
(pure gain blocks, no actuator lags) and drops the residual to 5.5e-4, now checked against a
derived `CLa·dt/ci2vel` floor.

**Adding an aircraft to `REGISTRY` is not free.** `GAINS` and `MANUAL_GAINS` must cover every
entry or `fly.py` raises `KeyError`, and the tail-arm plausibility gate asserts an exact dict over
the registry. Both were handled: the gains are derived rather than hand-tuned, and the 737 fails
the plausibility gate for a cleaner reason than the light aircraft — `l_eff = −Cmq/CLq` and JSBSim
defines no `CLq`, so the arm is genuinely undefined rather than implausible, and the strip path is
structurally unavailable for it.

**The roll loop could not be scaled from the 747's.** The 737 has 26× the roll authority but also
far more of its own damping, so solving for the rate gain reproducing the 747's closed-loop
damping gives `p_d` = −0.18 — negative, i.e. deliberately de-damping a well-behaved roll mode. The
loop is sized from the 737's own dynamics instead.

**Scripts in a worktree import the wrong tree.** `flightsim` is pip-installed editable against the
main checkout, so `python scripts/foo.py` from a worktree silently runs the *other* tree's code —
no error, just wrong answers. pytest is immune because it puts its rootdir first, which is exactly
why a green suite did not catch it. Both scripts now insert their own tree ahead of the installed one.

### Corrections to claims made above

**`thrust_lapse` is 0.72, not 1.0.** The "Thrust model" section's claim that the CFM56 tracks
`(ρ/ρ₀)^1.0` is true of the **full-power** `MilThrust` table, but the aircraft cruises at part
throttle, where the idle/military blend lapses differently. Fitted at the trim throttle over
25,000–35,000 ft the exponent is 0.7208, with a 1.79% residual. Fitting over 10,000–40,000 ft
instead leaves 6.75%, so the fit is band-limited like everything else here. `max_thrust` likewise
fits to ~11,700 lbf per engine against a 20,000 lbf rating, because flightsim's throttle map is
linear and JSBSim's varies 4.3× across the range.

**Layer 4's residual is layer 1's missing drag, not Dutch-roll phase.** The divergence is secular,
not oscillatory. Integrating the drag terms flightsim has no home for over the recorded sideslip
history predicts 1.015 m/s of the rudder kick's 1.516 m/s, and 0.002 m/s for the elevator doublet
— which is why that case sits at the 0.409 m/s Earth-rotation floor. An earlier claim attributing
it to accumulated Dutch-roll phase was wrong and is corrected in the test and the report.

**The `(vt, α) → (u, w)` transform was unnecessary.** Eigenvalues are invariant under it, so the
mode comparison needs no basis conversion and none can be got wrong.

**Trajectory sampling had to go from 0.25 s to 0.05 s.** The consumer holds each control sample
until the next, and the yaw damper moves the rudder continuously, so a coarse sample made the
replay fly a stale rudder — worth 5.43 m/s on the rudder kick until the rate came up.

**Layer 2's turn case is recorded, not compared.** `trim.trim` solves the wings-level problem only:
its unknowns are `[alpha, elevator, throttle]`, with no bank. JSBSim's 30° banked solution is in the
reference, so the comparison is one banked-trim solver away. The test asserts the gap rather than
skipping, so adding one forces the comparison to be written.

**Tolerances for `CD` and `Cm` became predictions rather than allowances.** Both are computed from
737.xml's own table constants: the `CD` difference is asserted to *equal* the terms flightsim lacks
(3.6e-4 worst residual, 2.9e-10 at the sideslip points), and `Cm` is bounded by its two identified
mechanisms and separately asserted exact at the reference point (5.9e-7).

### Results

| Layer | Result |
|---|---|
| 1 coefficients | `CL` 5.5e-9, `CY` 1.6e-14, `Cl` 7.0e-6, `Cn` 1.7e-6; `CD`/`Cm` predicted to 3.6e-4 |
| 2 trim | α 1.980° vs 1.965°, δe −0.05311 vs −0.05192 rad, thrust +1.04% |
| 3 modes | short period wn 0.04% / ζ 0.02%; phugoid 3.3% / 1.1%; lateral all under 3.3% |
| 4 trajectory | elevator doublet 0.410 m/s (floor 0.409); rudder kick 1.52 m/s |

No defect was found in flightsim. Every disagreement traces to a documented model difference with a
measured magnitude.

## Explicitly not done

Ground reactions and the two runway scripts; flaps, spoilers and speedbrake; `CDbeta` and `CDde`;
an α̇ term (`Cmadot` = −16 is folded into an effective `Cmq` for the short period only, and
flagged); N1/N2 spool dynamics; table-driven aero or thrust in `Aircraft`; a runtime validity guard
on `boeing737`; cloning the JSBSim repository — its `tests/` exercise JSBSim's property tree and
script parser rather than aircraft physics, and add nothing the Python API does not give.
