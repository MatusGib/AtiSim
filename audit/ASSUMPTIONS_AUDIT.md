# ASSUMPTIONS_AUDIT — Phase 4

`docs/ASSUMPTIONS.md` audited against what this audit found. Every entry gets
exactly one status.

Judged against the **declared scope**: *turbulence-encounter analysis of a rigid
fixed-wing aircraft over seconds to minutes*.

**Statuses.** The prompt supplies four. This audit adds a fifth, because the
four have no slot for the most valuable thing an audit can do — see `AUDIT.md`
§5.6.

| Status | Meaning |
|---|---|
| **Valid** | argument and citation given, within a stated envelope |
| **Known flaw, bounded** | wrong or incomplete, magnitude and onset stated |
| **Known flaw, unbounded** | wrong, consequence not quantified; the work to bound it is named |
| **Previously unbounded, now bounded** | the register said no bound was obtainable; this audit obtained one |
| **Undocumented** | an assumption the code makes that the register never declared |

**Headline.** Of the 19 declared entries, **13 survive as written**. One (B4)
was declared unboundable and is **now bounded**. Five need correction: **E4**'s
bound is wrong by ~80× (it measures discretisation where the flaw is scheme),
**E2**'s correction profile covers only half a core traverse, **C5** calls a
bound unobtainable that the project itself already obtained and abandoned, and
**C2** and **C4** overstate their own certainty in one word each. **Nineteen
undocumented assumptions were found**, of which eight carry real consequence.

Every one of the five corrections is to a *bound*, not to the *judgement*. The
register reached the right verdict on all nineteen; on five of them it reached
it with a number that does not support it.

---

## Part 1 — the register's own entries

### A. Frames and Earth

| # | Assumption | Status | Finding |
|---|---|---|---|
| **A1** | Flat, non-rotating Earth; NED inertial | **Valid** | The register's own measurement stands: Coriolis 0.0035 g at 747 cruise against ~1 g excursions, 6.88 m of position error per 20 s window. Correctly scoped, and correctly says it is not sound beyond ~10 minutes. Nothing found to contradict it. |
| **A2** | Constant g = 9.80665 | **Valid**, within the stated envelope | Closed by session 12's own measurement, and the closure is sound: worst mode movement consumes 7.6% of its tolerance. The register's correction of session 11's over-strong warning is itself correct — the 1:1 Lanchester mapping really is the phugoid's alone. |
| **A3** | Geopotential altitude | **Valid** | 0.17% at 11 km, stated in the source module. Smaller sibling of A2 and correctly paired with it. |

### B. Mass and structure

| # | Assumption | Status | Finding |
|---|---|---|---|
| **B1** | The airframe is rigid, the data is flexible | **Known flaw, unbounded** — and correctly so | CR-2144's Section IX derivative plots are labelled "Flexible" (verified: printed pp. 218, 220–228 all carry the label). Bounding it needs a rigid derivative set for the same aircraft and condition, which CR-2144 does not tabulate. **The register is right that this is the most significant open assumption, and right that it cannot be closed from held sources.** |
| **B2** | Constant mass, no fuel burn | **Valid** | 0.02% over a 20 s updraft, 0.6% over ten minutes. Comfortable. |
| **B3** | `Ixy = Iyz = 0` | **Valid** | Verified against CR-2144 Table IX-3, which tabulates only `IXZ`. The register is also right that `Ixz` itself is *not* neglected — `aircraft.inertia_tensor` carries it, `_unprime` handles the source's priming, and Agent C recovered all eight of Table IX-8's primed values to ±0.000% through the plant, with the test failing by 57.3% when `Ixz` is negated. |
| **B4** | Accelerometer at the CG. Register says **"Bound: none"** | **PREVIOUSLY UNBOUNDED, NOW BOUNDED** | **CR-2144 Table IX-3 gives `LXP = 86.0 ft` and `LZP = −10.0 ft` for every 747 flight condition, and Table IX-5 publishes `N(AZP/DE)` — normal acceleration at the pilot station.** The source models exactly this offset. Measured: manoeuvre **8.3%**, updraft **11.3%**, **Parks vortex 26.4%** of their own load excursions. The register names the pitch manoeuvre as the worrying case; it is the *smallest* of the three. Detail in `AUDIT.md` §3. |

### C. Aerodynamics

| # | Assumption | Status | Finding |
|---|---|---|---|
| **C1** | Lift linear in α, no stall | **Known flaw, bounded** | Fully and honestly documented; the three consequences (±g asymmetry unreachable, Cessna tables unused, |α| > 10–12° invalid) are all real and all stated. Nothing to add. |
| **C2** | Quasi-steady, no α̇ | **Known flaw, bounded** — but the **wording overstates its own certainty** | The bound is right and this audit sharpens it: restoring `Zẇ, Mẇ` to the engine's own plant matrix closes short-period damping to **+0.6%** of CR-2144's published 0.387 and leaves the phugoid frequency exactly unmoved. With the speed derivatives restored too, **all four** published mode factors close to ≤1%. But the register calls the omission's cost *"exact rather than attributed"* on the strength of a reconstruction that uses `CLα̇ = 6.7` and calls it **"tabulated"**. CR-2144 Table IX-2 prints **`−6.7`**, and its own Table IX-4 prints `ZWD = +0.0338` where Appendix A applied to `−6.7` gives `−0.0341`. **The source contradicts itself on this sign** (`NOTATION.md` §7.3); the reconstruction silently picks Caughey's side. The bound survives; the word "exact" does not. |
| **C3** | Derivatives frozen across the envelope | **Known flaw, unbounded** — and correctly so | The register calls this "the largest unbounded assumption" and is right. Bounding it needs the derivative-vs-Mach plots digitised, which is a chart read off a poor scan — weaker evidence than the tabulated set. This audit independently hit the same wall: the cruise `C_LM`/`C_mM` are plots, which is why the Appendix-A correction below is quoted with a sensitivity rather than a value. |
| **C4** | Parabolic polar + Korn wave drag | **Known flaw, bounded**, with a **circularity caveat the register does not carry** | Residuals as stated (0.004 near the fit, 0.014 below M 0.75, 0.006 above M 0.88). But `CD0` and `e` are back-solved from one point on Figure IX-6, and the nine points that break the circularity are **the project's own digitisation of the same figure**. The fit-circularity is broken; the source-circularity is not. `test_drag_polar.py`'s own docstring is honest about the first and silent about the second. |
| **C5** | Thrust along body x, no moment, no spool | **Known flaw, unbounded → the "unbounded" is now WRONG** | The register says the thrust moment is "unquantified — CR-2144 does not tabulate a thrust-line offset". True of CR-2144. **But an abandoned commit in the repo's own worktree `jolly-bhaskara-def594` (sha `d9d4442`, never merged) identifies NASA CR-114494 / Boeing D6-30643 Vol. II, which DOES tabulate the 747 thrust-line offsets**, resolves a conflict between two printings of that table, and quantifies the moment. Found by Agent G. **So the bound is obtainable and the work to obtain it has already been done and lost.** Status should be "bounded, pending re-landing", not "unbounded". |
| **C6** | Control surfaces move instantly | **Valid** for mode work, **known flaw, bounded** for the manoeuvring point | 0.241 s of real elevator travel against a 6.609 s short period = 3.65%, inside the existing pulse-length sensitivity. Correctly reasoned. |

### D. Atmosphere

| # | Assumption | Status | Finding |
|---|---|---|---|
| **D1** | ISA exactly, two layers to 20 km, dry air | **Valid**, with one **undocumented** rider | The reasoning is right — the wind fields *are* the weather, and inventing a temperature offset would violate the project's own rule. The rider is U6 below: the 20 km ceiling is stated in a docstring and **enforced nowhere**. |

### E. Wind and turbulence

| # | Assumption | Status | Finding |
|---|---|---|---|
| **E1** | One-way coupling | **Valid** | Every field is orders of magnitude larger than the aircraft. The register correctly flags that it would fail for a wake-vortex encounter. Note FD2e (printed p. 218) makes the same point in the same terms. |
| **E2** | The aircraft is a point for translational gusts | **Known flaw, bounded** — the best-handled entry in the register | The measured correction profile (exactly 0 inside the Parks core, 2.0·V₀/r₀ at the boundary, 0.109 at 1.25 r₀) is right and the reasoning about why is right. The one-directional implication ("diagnostic = +1 ⇒ rigid rotation ⇒ linear field ⇒ point model exact") is correctly stated and correctly *not* reversed. **But see U2 below**: the strip path that was built to go beyond this bound does not currently reproduce its own calibration at the station count it ships with. |
| **E3** | Frozen field | **Valid** | Evolution timescales (minutes to tens of minutes) against traverses of seconds. Correctly flagged as failing for Dryden. |
| **E4** | Wind held across all four RK4 stages | **Known flaw, bounded — but the BOUND IS WRONG BY ~80×** | The structure of this entry is exemplary: both halves closed by measurement, both falsified by bug injection, the order half (O(h) not O(h⁴), restored to 4.05 by per-stage sampling) and the body-force half (3.98e-12 m against a closed form, 13.33 m with the bug injected). **The consequence bound is the flaw.** E4 measures an h-vs-h/2 *position* difference (0.0169 m → 0.0024 m/s of gust, "~1e-4 relative") and concludes "No result the project quotes is affected." That is the **discretisation** error with the hold still in place; the **scheme** error is hold-vs-per-stage at the same dt. Measured on the number `scripts/vortex.py` prints: in-core Δθ **2.2596° → 2.2230°, −1.62% at dt = 0.02**, and **−0.82% at the published dt = 0.01**. So the cost is ~0.8%, not ~1e-4. No *conclusion* moves — vortex claims are orderings — but **the quoted 2.240° is not good to four significant figures.** |

### F. Numerics

| # | Assumption | Status | Finding |
|---|---|---|---|
| **F1** | Fixed-step RK4 at 50 Hz | **Valid** | Observed order 3.99982 (closed form) and 3.98913 (6-DOF). |
| **F2** | Quaternion renormalised every step | **Valid** | `max|1 − ‖q‖| = 1.1e-16` after 3,000 steps. The register's reading is the right one: a normalisation doing real work would be the bad outcome. |
| **F3** | float64 throughout | **Valid** | Verified: `jax.config.update("jax_enable_x64", True)` in `flightsim/__init__.py`, before any array is created, so any import of the package gets it. |
| **F4** | Round-off floor on trajectory differences | **Valid** | Not an assumption but a ceiling, as the register says. ~7e-11 m at 40,000 ft; pairwise order goes to −0.685 past dt = 1/256. Correctly used to justify stopping the fitted window at 1/32. |

---

## Part 2 — undocumented assumptions

**These are the important ones**, per the prompt. Ordered by consequence.

> **Remediation status, from commit `4e925c4` on branch
> `claude/flight-sim-prompt-fix-7ce581`.**
> Sixteen of the nineteen were acted on: **thirteen are now declared in
> `docs/ASSUMPTIONS.md`**, which is where a reader looks; **two were repaired** (U7, U17);
> and **one was examined and deliberately left** (U2). The three not in the table below —
> U3, U5 and U12 — are covered elsewhere: U3 by the register's existing B4, U5 by
> `AUDIT.md` §2.3's account of why a clean-room rebuild is not an independent instrument,
> and U12 by the ledger-coverage repair recorded in `PROJECT.md` §2 and `provenance.py`.
> This file remains the audit's own working record and is not the register. The mapping:
>
> | here | now in the register as | and |
> |---|---|---|
> | U1 | **F5** | decision recorded: `N_SPAN` and `a₀` both unchanged, convergence order measured at **1.50** |
> | U2 | E4's neighbours / `AUDIT.md` §2.2 | left alone — the literal Appendix-A form cannot be completed from held sources, and it was tested as a cause of the mode gap and falsified |
> | U4 | **C7** | declared |
> | U6 | **D2** | declared |
> | U7 | — | **REPAIRED**: `is_physical(x, ac)` now checks all three unknowns |
> | U8 | **F6** | declared |
> | U9 | **C10** | declared |
> | U10 | **C8** | declared, with its size measured (−0.004% at trim, +4.5% at full elevator) |
> | U11 | **B5** | change made, measured, and **reverted** — it broke two bit-exact guards |
> | U13 | **E6** | declared |
> | U14 | **E5** | declared; and `superpose()` with no arguments **REPAIRED** |
> | U15 | **C9** | declared, deliberately **not** repaired |
> | U16 | **A4** | declared |
> | U17 | — | **REPAIRED**: `qbar` built from the true airspeed |
> | U18 | **F7** | declared |
> | U19 | **C11** | declared; neither derivative touched |
>
> *Also from this part:* the note under U13 that `wind.py` had the Oseguera & Bowles
> constants' dependency backwards is **REPAIRED** — checked against the paper's own
> wording, which states `z_m/z* = 0.22` as the TASS input and 12.5 as what was iterated
> from it.

### U1. The strip integral does not reproduce its own calibration at the station count it ships with
**Consequence: high. This is the one live defect found.**

`airframe.calibrated_lift_slope` pins `a₀ = −8·Clp` so that a rigid roll rate
through the strip integral returns the tabulated `Clp` **exactly**; two
docstrings say "exactly". It is exact only in the continuum limit.
`loads.strip_model` builds stations from `airframe.stations(ac)`, i.e. the
default `N_SPAN = 9`, where the integral returns **82.64% of `Clp` — a −17.4%
systematic understatement**, identical for every aircraft, which identifies it
as trapezoidal quadrature error on the elliptic chord's square-root tip
singularity rather than a data problem.

Every existing test of the identity overrides the count: `test_airframe.py` uses
2001, `test_loads.py` uses 201, `test_wind.py`'s helper defaults to 2001. **None
exercises 9.** `test_the_default_station_count_has_converged` does not cover it
either — its own docstring says it measures "the fitted **rates**", i.e.
`sampled_rates`, a linear slope fit that converges immediately, not the
strip integral, whose integrand is singular at the tips.

*Found by Agent C, confirmed independently here. Pinned by
`test_the_strip_integral_only_reproduces_Clp_in_the_continuum_limit`.*

**What it costs today: nothing quoted.** Both of the project's source fields
have no spanwise variation, so the strip path returns exactly zero for them and
the headline numbers are the point model's. It is a defect waiting for the first
field with genuine spanwise structure — which is precisely what the strip path
was built for. The cubic-gust positive control (0.187 m) is understated by ~17%.

### U2. The conversion chain uses small-angle forms, not CR-2144 Appendix A's definitions
**Consequence: moderate, bounded at ~1.1%.**

Appendix A (printed A-16/A-17) defines `X_w`, `Z_w`, `M_w` with a
`−2(W₀/U₀)(C + (M/2)C_M)` term and with `U₀ = V_To cos α₀`. `aircraft.py`
inverts them as if the bracket were the first term alone and as if `U₀ = V_To`,
and it works in `C_L`/`C_D` where Appendix A works in `C_N`/`C_X`. Measured at
cruise: **`CLa` +1.15%, `CDa` +1.13%, `Cma` +0.32%.** At the approach condition
the chain closes against the source's own Table IX-2 to within 2.2%, which is
what bounds it. Never declared anywhere. `NOTATION.md` §7.2; pinned by
`test_the_appendix_a_W0_omission_stays_within_its_measured_bound`.

### U3. The moment reference, the inertia reference and the accelerometer are one point
**Consequence: benign here, but undeclared.**

Nothing in the code records a CG. `CR-2144` Table IX-3 gives `C.G.(MGC) = .250`
for **both** flight conditions used, so the derivatives and inertias really are
referenced to the same station and the assumption holds — but it holds by luck
of the source, not by anything the code checks. A future aircraft whose source
tabulates derivatives at a different CG from its inertias would break silently.

### U4. Drag responds to pitch rate and elevator, producing an `X_q` the source does not model
**Consequence: small, and it is the engine being *more* complete than its reference.**

`aero.coefficients` builds `CD` from the **total** `CL`, which includes
`CLq·q̂` and `CLde·δe`. So a pitch rate changes induced drag, and the engine has
a stability-axis `A[0,2] = X_q = −0.881 s⁻¹` at the approach condition where
CR-2144 tabulates no `C_Dq` at all. Measured against the parabolic polar's own
prediction: **−0.8887 predicted, −0.8807 in the plant, agreeing to 0.9%**, which
identifies the mechanism rather than merely noting the element is non-zero.
*Predicted by Agent F as the discriminator for its unexplained phugoid-damping
residual, then confirmed.* Pinned by
`test_the_engine_carries_an_Xq_that_cr2144_does_not_model`.

### U5. `−ω×v` gives the engine a `∂u̇/∂q = −w₀` term the reference model lacks
**Consequence: none numerically, but it invalidates naive element-by-element comparison.**

−62.6 ft/s at cruise. It is correct body-axis physics; CR-2144's linear model
simply does not carry it. No comparison in the project accounts for it.

### U6. The atmosphere has no ceiling — and no floor
`atmosphere.py`'s docstring says "0 to 20 km"; above 20 km the isothermal branch
continues silently rather than raising or clamping, holding `T = 216.65 K` to
arbitrary altitude and returning finite numbers to 1000 km. Since the speed of
sound is then frozen at 295.07 m/s, the error feeds Mach and therefore the
wave-drag term. **Below** sea level the troposphere lapse extrapolates equally
freely — 320.65 K and 1.93 kg/m³ at −5 km, 613.15 K and 30.47 kg/m³ at −50 km.
Nothing in scope flies at either end, but nothing prevents it either. The
downward half matters because U16 means runs actually go there.

### U7. `trim.trim` has no convergence test, and `is_physical` is not one
A fixed 40-iteration `lax.scan`. It returns whatever it lands on and says
nothing. `is_physical` exists but is opt-in, is not called by `trim`, and checks
`|α| ≤ 15°` **and nothing else** — not throttle, not elevator, not even the
residual it was reached with.

**Now quantified.** On a plain (V, h) sweep, solutions that *pass* `is_physical`
while demanding throttle outside [0, 1] or elevator past its limit:
**319/640** (747), 22/640 (747-approach), 284/640 (Cherokee), **364/640**
(Cessna). The 747 at V = 471.8 m/s passes at α = −0.57° on a throttle of **567**.
So the check does not merely miss some cases — on two aircraft it endorses the
majority of a routine sweep. `PROJECT.md` §5 discusses the absurd-root problem
but frames it as an aero-model property rather than as a solver that cannot
report failure and a gate that reads one of three unknowns.

### U8. `airframe._active_shape` is process-global mutable state
A physics parameter set by a context manager. Correctness under `pytest-xdist`,
threads, or any concurrent use is assumed and untested.

### U9. `δa` is a compound control treated as a single angle
CR-2144 Table IX-2's footnote defines `δa` as *"total deflection of right
inboard aileron plus left inboard aileron with the effect of outboard ailerons
included"*. `aircraft.py` quotes the footnote and takes the number as given —
correct — but `Controls.aileron` is then used as though it were a single surface
angle, and `aileron_limit = 20°` is a declared limit on a compound quantity
whose physical meaning is not a deflection.

### U10. Wave drag acts on the total `CL` including control and rate contributions
The `20·(M − M_crit)⁴` law and the `CL/(10cos³Λ)` Korn term are applied to a
`CL` that includes `CLq·q̂` and `CLde·δe`. Defensible; undeclared. Related to
U4 but a separate channel (compressibility rather than induced drag).

### U11. `_B747_G = 32.174` is an inlined conversion factor
`units.py` states the rule: *"Never inline a conversion factor anywhere else."*
`aircraft._B747_G` does, and truncates `g₀ = 32.17404855643044`. Measured slip
against the `LB2KG` route: **1.5e-6 relative**. Harmless; a stated rule broken.

### U13. The microburst has no ceiling — the source's fourth parameter is unmodelled
**Consequence: bounded by the run geometry, and only just.**

Oseguera & Bowles specify **four** parameters; `wind.Microburst` carries three.
The missing one is `z_h`, the depth of the outflow. Without it the field has no
upper bound and grows monotonically to a **50.6 m/s (98.5 kt) on-axis
asymptote**. The project's default 300 m AGL penetration sits essentially
exactly at the paper's own `z_h` (294.4 m), so the runs on record are inside the
model's valid band — but by coincidence of the chosen altitude, not by anything
the code enforces. Verified against the primary source, now in `refs/`.

*Also found: the module docstring says `z_m/z* = 0.22` is derivable as
`ln(12.5)/11.5`, "a genuine cross-check rather than four restatements of one
number". The paper has it the other way round — 0.22 is an empirical TASS input
and 12.5 and 0.2357 are its consequences. The arithmetic in the docstring is
correct; the epistemic claim built on it is backwards.*

### U14. Superposition is exact for the fields but breaks the microburst's boundary condition
**Consequence: latent — no run currently superposes onto the microburst.**

`wind.superpose` sums fields exactly (`wind_ned` bit-identical, `omega_gust` to
6.5e-16). But the microburst's defining property — both components vanish at
z = 0, which its paper's introduction singles out as *the* thing earlier
analytic models got wrong — does not survive addition. Superposing anything with
a non-zero ground value puts **3 m/s through the ground everywhere**. The module
comment says summing "is exact within the model's own linearisation" and does
not name this exception.

*(Also: `superpose()` with no arguments returns the Python `int` 0, not a
callable field.)*

### U15. Aerodynamic lift is applied at the CG's relative wind, so control moments do unopposed work
**Consequence: bounded — a genuine energy-conservation violation, wholly outside
the declared envelope, and never globally winning.**

`aero.aero_forces_moments` builds one lift vector perpendicular to the relative
wind **at the CG**. A surface at arm `l` actually sees a local wind tilted by
`ε = q·l/V`, and the streamwise component of its tilted lift is what makes the
channel dissipative. The model omits that component, so the elevator's pitching
moment does work on the airframe with nothing opposing it.

**Measured:** the Cherokee delivers `P_aero = +56.9 kW` in motionless air with
the throttle shut, at |q| = 8.256 rad/s and full elevator. The omitted term
cancels the moment power **exactly** — algebraically, to a relative residual of
1.9e-16 — and restoring it takes the violating region from 1190 grid points to
**0**. See `AUDIT.md` §2.5.

**Why it is nevertheless bounded:** `max(E − E₀) = +0` exactly in every still-air
run tried, including feedback laws designed to pump it; `P_aero > 0` at **0 of
80,000** randomised states inside the declared |α| ≤ 12° envelope; and the
threshold pitch rate is 84–201 °/s. The register does not mention the assumption
at all — C1 and C3 cover linear aero and frozen derivatives, but neither says
that the force build-up places every lift increment at one point.

### U16. There is no ground
**Consequence: guarded downstream, undeclared upstream.**

`integrate.step` has no ground plane and `atmosphere.py` extrapolates the
troposphere lapse below sea level without limit — 30.47 kg/m³ at −50 km. A
747-approach released at 300 m crosses `h = 0` at t = 8.65 s at a 25.1 m/s sink
rate and integrates on to −698 m, every sample finite.

**The register has no entry for it**, in any of its six sections. What keeps this
from being a live defect is that `scripts/microburst.py` — the one shipped
analysis that flies at terrain — cuts its run at one wingspan of clearance, with
a well-reasoned comment saying why. **The guard is at the call site, not in the
engine**, so it protects exactly one caller.

### U17. The airspeed floor makes aerodynamic force nonzero at zero airspeed
**Consequence: bounded and tiny; the docstring's claim is narrower than it reads.**

`aero.V_MIN = 1.0` is described as a NaN guard that "never binds in normal
operation". True of normal operation — but `V` is floored *before* `qbar`, so at
|V| < 1 m/s the model reports dynamic pressure it does not have: `(V_MIN/V)²`,
4× at 0.5 m/s and 100× at 0.1 m/s. At exactly V = 0 the residual `CL0` term
yields 19.86 N (747), 170.73 N (747-approach), 4.30 N (Cherokee) and 1.44 N
(Cessna), each at its own cruise altitude, so **free fall does not read zero**:
`n_z` = +7.0e-6 to +4.0e-4 rather
than 0. Zeroing the aero coefficients gives exactly `−0.0`, which localises it to
the floor and not to `load_factor`.

### U18. A control channel with zero authority makes any Newton solve return NaN in silence
**Consequence: latent for the shipped solvers; a trap for any new one.**

The Cessna's `CYdr = Cldr = Cndr = 0` is a declared modelling choice, and its
consequence is not declared: the rudder column of any control Jacobian is
identically zero, so `det = 0` and `jnp.linalg.solve(J, b)` returns
`[nan, nan, inf]` **without raising**. `trim` does not carry rudder as an
unknown so it is unaffected, but nothing marks the aircraft as unusable by a
solver that does — and a steady-turn solve is the obvious next one.

The same shape appears with `max_thrust = 0`, where `trim` itself returns
`[nan, nan, nan]` with a `nan` residual and no flag. Related to U7: the solver
has no way to report failure.

**One real mitigation, worth stating precisely because it is narrower than it
looks.** `flightsim/tests/conftest.py` sets `jax_debug_nans`, so under pytest
this raises rather than propagating. That is genuine protection for anything the
suite exercises — and it is a **test-time setting only**. Nothing enables it for
`scripts/`, for the notebook, or for a library caller, which are exactly the
paths a new solver would be written on. The audit's own regression test has to
switch it off to observe what a non-test caller gets.

### U19. `effective_tail_arm` reads one of two available estimates, and never compares them
**Consequence: a free consistency check the project does not run — and it fails
on two aircraft.**

The tail arm is recovered from the rate pair, `l/c = −Cmq/CLq`. The *control*
pair gives the same geometry independently: `l/c = −Cmδe/CLδe`. Both are in the
registry for all four aircraft, and nothing compares them:

| aircraft | rate pair | control pair | ratio |
|---|---|---|---|
| boeing747 | 4.0241 | 3.9694 | **0.986** |
| boeing747_approach | 3.8519 | 3.9645 | **1.029** |
| cherokee | 1.2802 | 2.5621 | **2.001** |
| cessna172 | 0.8558 | 2.4681 | **2.884** |

**The split is exactly the sourcing split.** The two CR-2144 sets agree to 1.4%
and 2.9% — a real, independent corroboration of that transcription. The two sets
whose cited source file *does not exist in the repository* (`AUDIT.md` finding
20) disagree by factors of 2.0 and 2.9. This is the only internal evidence the
audit found bearing on the quality of the unverifiable light-aircraft data, and
it is not encouraging. `tail_arm_is_plausible` already rejects both on the rate
estimate, so the gate works — but it works by accident of which estimate it reads.

### U12. The provenance ledger enforces nothing
`PROJECT.md` §2 (twice), and `provenance.py`'s own module docstring, state that
*"a constant added without a ledger entry fails the build"*. **No test has ever
walked the source modules.** All eight tests in `test_provenance.py` iterate
`LEDGER` and check it against itself. Measured coverage: **~346 non-trivial
numeric literals across the nine physics modules against 13 ledger entries** —
about 2%. `wind.py`, `aero.py`, `atmosphere.py`, `units.py` and `autopilot.py`
(45 gains) have **zero** entries.

*Found independently by this auditor, Agent B and Agent G.* The missing
direction is now supplied by
`test_the_provenance_ledger_does_not_cover_the_source_modules`, which pins the
measured baseline so a **new** unledgered constant fails the build — which is
what the documentation has been promising.

---

## Part 3 — the register's own summary table, re-scored

`ASSUMPTIONS.md` ends with a seven-row "which assumptions need action" table.
Re-scored against this audit:

| Register's row | Register's verdict | This audit |
|---|---|---|
| 1. E4 wind held across RK4 stages | closed / measured | **Agreed.** Exemplary. |
| 2. B1 rigid vs flexible | unquantifiable | **Agreed.** |
| 3. C3 derivatives frozen | unbounded | **Agreed**, and the reason it stays unbounded is sound. |
| 4. E2 point-aircraft gusts | closed for the linear fit; strip path flyable and measured | **Qualified.** The bound is right, but the strip path built to exceed it does not reproduce its own calibration at the shipped station count (U1). |
| 5. A2 constant g | closed | **Agreed.** |
| 6. C5 no thrust moment, no spool | sound for now | **Downgraded.** The thrust moment is stated to be unboundable; a source that bounds it was found by this project and abandoned unmerged (see C5). |
| 7. B4 accelerometer vs DFDR | caveat, keep claims as orderings | **Upgraded to a measured bound**, and the register's guidance to keep Fig. 8 claims as orderings is now *quantitatively justified* rather than prudential — 8–26%, category-dependent, so it does not cancel between clusters. The orderings themselves survive. |

**Four entries the register does not have and should:**

- **The strip quadrature defect (U1)** — the only finding here that is a bug
  rather than a bound.
- **The conversion-chain omission (U2)** — the only systematic error in the
  cruise 747's aerodynamic data that the project has never recorded.
- **The energy seam (U15)** — the register has a section on aerodynamics and a
  section on numerics, and this belongs to neither: the model can *create*
  mechanical energy, and the reason is a placement assumption in the force
  build-up that C1 and C3 do not cover.
- **The absent ground (U16)** — an assumption so basic the register never
  thought to state it, and the only one on this list that a reader would
  reasonably assume the opposite of.

**And one row the register would be entitled to add in its own favour:** the two
independent tail-arm estimates agree to 1.4% and 2.9% on the CR-2144 aircraft
(U19). That is a corroboration of the transcription which nothing in the project
currently claims.

---

## Part 4 — how good is this register?

Stated plainly, because an audit that only lists faults misrepresents the thing
it audited.

`docs/ASSUMPTIONS.md` is **unusually good**. It states a rule ("an assumption
without a number is not documented, it is asserted") and largely keeps it. It
records where its own earlier reasoning was **wrong** and was corrected by
measurement — twice, explicitly, with the superseded text struck through rather
than deleted. It distinguishes bounded from unbounded honestly, and it does not
claim bounds it does not have.

The audit's negative findings are correspondingly narrow: four live defects (U1,
the mode swap, U15's energy seam and U18's silent NaN), one undeclared systematic
(U2), one bound declared unobtainable that was obtainable (B4), one abandoned
bound (C5), one word doing too much work ("exact" in C2), and a documentation
claim the code never implemented (U12). Everything else in the register survives
scrutiny as written.

The register has **two** blind spots, and the second only became visible once the
falsification phase reported.

**First: it audits the model against itself, and almost never against the source
document's other pages.** B4's "Bound: none", C5's "CR-2144 does not tabulate a
thrust-line offset", and the two-of-four mode comparison are all cases where the
answer was in a table the project had already cited for something else. Reading
the rest of the source is what most of this audit consisted of.

**Second: every entry answers "is this modelling choice accurate enough?" and
none answers "what does the model do when it leaves the envelope?"** The register
is organised around error magnitudes — 0.383% here, 2.6% there — which is the
right frame for A1 through F4 and the wrong frame for U15, U16, U17 and U18.
Those four are not inaccuracies; they are absences. There is no ground, there is
no stall, there is no energy check, and a singular solve returns NaN rather than
raising. Each is invisible to a question of the form "how big is the error?",
because outside the envelope the error is not small — it is undefined. **A
register built entirely on bounds cannot record a missing floor.** That is the
structural gap, and it is why the prompt's insistence on a falsification phase
earned its place: nothing in Phases 1 through 4 was shaped to find these.
