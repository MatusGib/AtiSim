# Model assumptions — what is assumed, why, and how big it is

Companion to `PROJECT.md`. That document records **what has been measured**; this one
records **what has been assumed**, with a bound on each. Written session 11, after external
review asked that the underlying assumptions be well founded and explained.

**The rule here is the same as §3's:** an assumption without a number is not documented, it
is asserted. Every row below either carries a measured bound or says plainly that it has
none and why.

---

## What this model is scoped to

**Turbulence-encounter analysis of a rigid fixed-wing aircraft over seconds to minutes.**
Concretely: the Wingrove & Bach Fig. 8 discriminator, the Bowles F-factor against thrust
authority, and a Dryden layer to come.

That scope is what makes most of the assumptions below acceptable. Several of them would
be indefensible for a different use — navigation over hundreds of kilometres, a full
flight-envelope simulator, or anything asserting agreement below about 0.5%.

**Three questions decide whether an assumption is safe here:**

1. Does it bias a quantity the project actually reports?
2. Is the bias smaller than the agreement being claimed?
3. Does it get worse over the encounter window, or stay bounded?

An assumption that fails (2) is the dangerous kind, because it is invisible: the model
agrees with a source to 0.4% while carrying a 0.4% systematic error, and the agreement is
partly luck. Session 11 flagged **two** entries as failing (2). Session 12 measured both,
and **A2 turned out to pass** — the 0.383% gravity error reaches the phugoid 1:1 but the
lateral modes only at 0.06–0.08%, so it does not threaten the agreements it appeared to.
**E2, the vortex span ratio, still fails (2)** and is the one to respect.

---

## A. Frames and Earth

### A1. Flat, non-rotating Earth; NED is an inertial frame

**Where:** `dynamics.derivatives` — the Newton-Euler equations carry no Earth-rate or
transport terms. `state.py` treats NED as inertial.

**Why:** an encounter lasts 1.5 s (a vortex core) to 95 s (a microburst to ground) and
covers a few kilometres.

**Bound, measured:**

| | 747 at 235.9 m/s, 12,192 m | Cherokee at 50 m/s, 1,500 m |
|---|---|---|
| Coriolis `2Ω×V` | 0.03441 m/s² = **0.00351 g** | 0.00729 m/s² = 0.00074 g |
| Transport `V²/(R+h)` | 0.00872 m/s² = 0.00089 g | 0.00039 m/s² = 0.00004 g |
| Position error over a 20 s window | **6.88 m** | 1.46 m |

**Verdict: sound for this use case.** Load-factor excursions the project reports are of
order 1 g (−1.23 g for the vortex, −1.90 g for the manoeuvre), so 0.0035 g is 0.35% of the
signal and does not accumulate within a window.

**Not sound for:** any claim about ground track over hundreds of km, or a run longer than
about ten minutes. 6.88 m per 20 s grows as t².

### A2. Constant gravity, g = 9.80665 m/s²

**Where:** `atmosphere.G0`, used by `dynamics.derivatives`, `trim`, `specific_force`.

**Why:** simplicity, and it is exact at sea level.

**Bound, measured** — using `g(h) = g₀·(R/(R+h))²`, R = 6,371,008.8 m:

| Aircraft | Altitude | True g | Model is |
|---|---|---|---|
| 747 cruise | 12,192 m | 9.76922 | **+0.383% high** |
| 747 approach | 0 m | 9.80665 | exactly right |
| Cherokee | 1,500 m | 9.80204 | +0.047% high |

Lanchester gives `ωn_phugoid = √2·g/u₀`, so a gravity error should map **1:1** into phugoid
frequency. Session 11 reasoned from that to a warning covering every mode. **Session 12
measured it instead, and the warning was too strong.**

**Bound, measured session 12** — `flightsim.dynamics.G0` replaced by `g(h)`, re-trimmed,
all five modes recomputed. §4's tolerance is the one each mode is actually asserted to:

| Mode at 747 cruise | g = 9.80665 | g(h) = 9.76922 | Movement | §4 tolerance | Consumed |
|---|---|---|---|---|---|
| phugoid ωn | 0.055319 | 0.055109 | **−0.3798%** | 5% | 7.6% |
| phugoid ζ | 0.055956 | 0.055654 | −0.5385% | 10% | 5.4% |
| short period ωn | 0.950773 | 0.950775 | **+0.0002%** | 3% | 0.01% |
| short period ζ | 0.342526 | 0.342510 | −0.0046% | 5% | 0.1% |
| Dutch roll ωn | 0.943202 | 0.942458 | −0.0788% | 2% | 3.9% |
| Dutch roll ζ | 0.036085 | 0.035931 | −0.4288% | 10% | 4.3% |
| roll τ | 1.795366 | 1.794285 | −0.0602% | 5% | 1.2% |
| spiral τ | 138.0424 | 138.1187 | +0.0552% | 2% | 2.8% |
| *trim α* | *4.6362°* | *4.6059°* | *−0.6535%* | — | — |

**Control: the same sweep on the sea-level approach 747 moves every quantity by exactly
0.0000%**, since `g(0) = g₀` identically. The effect is altitude and nothing else.

**Three things the measurement says that the reasoning did not:**

1. **Lanchester holds, to three figures.** Phugoid ωn moves −0.3798% against a gravity
   change of −0.3816%. The 1:1 claim is now measured rather than asserted.
2. **The 1:1 mapping is the phugoid's alone.** Short period is *immune* (+0.0002%), and the
   lateral modes move 0.055–0.079% — **five to fifteen times smaller** than the agreements
   session 11 feared for them (Dutch roll ωn 0.4%, spiral τ 0.8%, roll τ 0.9%). So the
   blanket "no sub-0.5% claim at altitude is safe" was wrong: it applied the phugoid's
   sensitivity to modes that do not have it.
3. **The lateral modes' sensitivity is indirect.** There is no gravity term dominating the
   lateral equations. g moves the **trim point** — α falls 0.65%, because less weight needs
   less lift — and the derivatives are then read at a different α. That is why the lateral
   movement is an order of magnitude below the phugoid's.

**Decision, session 12: `g(h)` is NOT modelled, and this bound closes the entry.** Every
movement is comfortably inside the tolerance of the check it would affect — the worst
consumes 7.6% of its band. Against that, `G0` is imported by `dynamics`, `trim`, `aircraft`
and `vortex_viz`, and changing it would move §4 baselines that are off-limits to feature
work. And every result the project quotes is at **one altitude per aircraft**, so a constant
g is *exactly* right per run; the bias only exists for comparisons across altitudes, which
the project does not make.

**What replaces the old warning:** a sub-0.5% claim at altitude is unsafe **for the
phugoid**, which carries the full 0.38%. It is safe for the lateral modes (0.06–0.08%) and
for the short period (~0). Revisit if the project ever compares one aircraft across two
altitudes, which is the case a constant g genuinely cannot serve.

### A3. Altitude is geopotential, not geometric

**Where:** `atmosphere.py`, and its docstring already says so.

**Bound:** 0.17% at 11 km, 0.31% at 20 km — stated in the source module.

**Verdict: sound**, and it is the smaller sibling of A2. Both are altitude-dependent
systematic errors of a few tenths of a percent, and both should be revisited together if
either is.

---

## B. Mass and structure

### B1. The airframe is rigid

**Where:** everywhere. `Aircraft.inertia` is a constant tensor; there are no structural
degrees of freedom.

**The problem:** CR-2144 §IX's derivative plots are labelled **"Flexible"**. The
derivatives carry aeroelastic corrections for a structure this model does not have. The
data and the model disagree about what kind of aircraft this is.

**Bound: NONE, and it cannot be produced from sources the project holds.** Quantifying it
needs a rigid-airframe derivative set for the same aircraft and condition, which CR-2144
does not tabulate.

**Verdict: the most significant open assumption in the project.** It is not a reason to
distrust the mode comparisons — those are closed-loop against the source's own arithmetic
from the source's own derivatives, so they test the solver regardless (see `PROJECT.md`
§3, "Source qualification"). It *is* a reason not to claim the model represents a real
747's structural response.

### B2. Constant mass and inertia; no fuel burn

**Where:** `Aircraft` is a constant pytree through every rollout.

**Bound, measured:** 747 trim thrust is 186.0 kN; at a typical high-bypass cruise TSFC of
0.55 lb/(lbf·h) that is 2.898 kg/s.

| Run length | Fuel burned | Fraction of mass |
|---|---|---|
| 20 s (an updraft column) | 58 kg | **0.020%** |
| 60 s (a fixed-control hold) | 174 kg | 0.060% |
| 600 s (ten minutes) | 1,739 kg | 0.602% |

**Verdict: sound**, comfortably. Even the ten-minute figure is below the ±25% band §5
places on the vortex parameters. Revisit only if a run ever exceeds an hour.

### B3. The aircraft is symmetric: Ixy = Iyz = 0

**Where:** `aircraft.inertia_tensor` accepts only `Ixz`.

**Verdict: sound.** Every source the project uses tabulates only Ixz, because a
port/starboard-symmetric airframe has no other product of inertia. Ixz itself is **not**
neglected and `PROJECT.md` §2 notes it "is not negligible for the 747" — session 11
measured its consequence: it is the +0.302 intercept in the roll-mode law (§4).

### B4. The accelerometer is at the CG, and the CG is at the inertia reference

**Where:** `dynamics.specific_force`.

**Why:** it is the quantity `load_factor` needs, and Fig. 8 is built from DFDR normal
acceleration.

**Bound: none, and this is a caveat on the comparison rather than on the model.** A real
DFDR accelerometer sits metres from the CG and therefore reads an additional
`ω̇ × r + ω × (ω × r)`. In a pitch manoeuvre with the sensor well forward that term is not
negligible, and **the Fig. 8 reference data carries it while this model does not.** The
project's Fig. 8 claims are orderings rather than values (§5), which is what keeps this
safe — but it is another reason those claims must stay orderings.

---

## C. Aerodynamics

### C1. Lift is linear in α, with no stall

**Where:** `aero.coefficients`, `CL = CL0 + CLa·α`.

Fully documented already in `PROJECT.md` §5 and §7, including the three consequences: the
±g asymmetry is unreachable, the Cessna's stall tables stay unused, and any encounter
driving |α| past ~10–12° reports lift the sources deny. Not repeated here.

### C2. Aerodynamics are quasi-steady: no α̇ or unsteady lag

**Where:** `aero.coefficients` takes instantaneous `vel_rel` and `omega_rel` only.

**Bound, measured session 11 — and this one is now exact rather than attributed.**
Restoring Caughey's own tabulated CLα̇ = 6.7 recovers his published A[1,1] to **0.03%** and
A[1,2] to 0.003%; the model's A[2,2] **is** his raw Eq. (5.51) Mq. So the entire effect of
the omission is a known multiplicative factor `1/(1 − Zẇ)` on the Z row plus one term in
the M row. `PROJECT.md` §4 carries the table.

**Verdict: sound and bounded.** This is the best-characterised assumption in the project.

### C3. Stability derivatives are constant across the whole flight envelope

**Where:** `Aircraft` is a constant pytree. `aero.coefficients` uses the same numbers at
any speed, altitude and angle of attack the sim reaches.

**Why:** CR-2144 tabulates derivatives at discrete flight conditions. The project
transcribed one per aircraft, two for the 747.

**This is the largest unbounded assumption in the model**, and it is the one that most
deserves attention next. The derivatives are a small-perturbation linearisation about a
single point, and the sim is nonlinear and flies away from that point:

| Run | Excursion from the linearisation point |
|---|---|
| Trim (747 cruise) | M 0.7995 — the reference |
| Lee wave, south leg | minimum airspeed 226.9 m/s → **M 0.7690, ΔM = −0.031** |
| Manoeuvring Fig. 8 point | peak \|α\| **10.31°** against a 5.7° linearisation |

CR-2144's own §IX plots show CLα, CDα, Cmα, Cmq, CLδe and Cmδe all varying with Mach
through the transonic region, so ΔM = 0.03 near M 0.80 is not a small excursion — it is
where those curves bend.

**Bound: none.** Producing one means digitising the derivative-vs-Mach plots, which §7
already declines for the same reason it declines further flight conditions: they are chart
reads off a poor scan, and a chart read is weaker evidence than the tabulated set already
in use. **The honest statement is that every result away from the trim point inherits an
unquantified derivative error, and that this is why §5 caps analysis windows at the linear
range.**

### C4. Parabolic drag polar plus a Korn wave-drag rise

Documented in `PROJECT.md` §5 with measured residuals: within 0.004 near the fit, up to
0.014 below M 0.75 and 0.006 above M 0.88. Not repeated here.

### C5. Thrust acts along body x, through the CG, with no engine dynamics

**Where:** `aero.thrust_force` — `throttle × max_thrust × (ρ/ρ₀)^lapse`.

**Three separate assumptions bundled together, and they are not equally safe:**

- **No spool dynamics.** A high-bypass turbofan takes several seconds from idle to full
  thrust. Every result in the project is either fixed-throttle or trimmed, so nothing
  currently depends on a throttle transient — but the F-factor work compares against
  *thrust authority*, which is a statement about what the engines could do. If a recovery
  manoeuvre is ever flown, spool time is the first thing that must be added.
- **No thrust moment.** The 747's engines hang below the CG, so real thrust produces a
  nose-up pitching moment that changes with throttle. Unmodelled, and unquantified —
  CR-2144 does not tabulate a thrust-line offset.
- **No Mach dependence of thrust.** Only a density lapse.

**Verdict: sound for the fixed-throttle encounters flown so far; the first thing to fix
before any powered-recovery result.**

### C6. Control surfaces move instantly, with no rate limit or actuator lag

**Where:** `Controls` is applied directly; `manual.py` ramps the *stick*, not the surface.

**Bound, measured:** a 747 elevator moves at roughly 37°/s, so the manoeuvring case's
8.926° deflection would take **0.241 s** in reality — **3.65% of the 6.609 s short
period** — and the model applies it in one 0.02 s sample.

**Verdict: sound for mode and frequency work; a real caveat on the manoeuvring Fig. 8
point.** §8 already records that Δθ moves from 25° to 30° across defensible pulse lengths,
so a 0.24 s ramp sits inside that existing sensitivity rather than beside it.

---

## D. Atmosphere

### D1. ISA exactly, with no weather deviation; two layers to 20 km; dry air

**Where:** `atmosphere.py`.

**Verdict: sound, and deliberately so.** The wind fields *are* the weather in this project,
and each is taken from a source that specifies its own conditions. Adding a temperature
offset would mean inventing one, which §3's rule forbids. Humidity changes density by well
under a percent at these altitudes.

---

## E. Wind and turbulence

### E1. One-way coupling — the aircraft does not disturb the air

**Where:** the wind model contract; `wind_ned` is a function of position and state, never
of the aircraft's own circulation.

**Verdict: sound.** Every field in use is an atmospheric phenomenon orders of magnitude
larger than the aircraft. It would **not** hold for a wake-vortex encounter behind a
preceding aircraft, which §7 lists as a possible future field.

### E2. The aircraft is a point for translational gusts

**Where:** the wind field is evaluated at `state.pos_ned` — one point — and `field_model`
derives `omega_gust` from the analytic gradient, which is a **first-order** correction for
variation across the airframe.

**Bound, measured session 11 — and this has never been stated:**

| Field | Characteristic scale | In 747 spans (59.64 m) |
|---|---|---|
| **Parks Hannibal vortex, core radius** | 182.9 m | **3.07** |
| **Parks Morton vortex, core radius** | 137.2 m | **2.30** |
| Wingrove updraft, column radius | 2,359 m | 39.6 |
| Doyle lee wave, quarter wavelength | 6,250 m | 104.8 |
| Oseguera microburst, radius | 1,000 m | 109.8 (Cherokee spans) |

**The vortex is the marginal case, and it is the headline result.** At 2.3–3.1 spans the
linear-gradient `omega_gust` correction is doing real work rather than being a small
tidy-up, and second-order variation across the span is not represented at all. Everything
else in the project is 40 spans or more and is comfortably a point.

**Verdict: sound for the updraft, lee wave and microburst; the weakest link in the vortex
result.** It compounds with §5's existing ±25% band on the identified vortex parameters
and with §2's note that the rotational gust from a Wingrove-scale vortex already exceeds
the 747's full aileron authority by ~1.5×. Vortex conclusions should stay orderings, which
is what §5 already requires for a different reason.

### E3. The field is frozen — wind depends on position, not time

**Where:** `field_model(field)` wraps `pos_ned -> wind_ned`.

**Why:** Taylor's frozen-turbulence hypothesis, and every source field is quoted as a
spatial structure.

**Verdict: sound for these fields**, whose evolution timescales (minutes for a microburst,
tens of minutes for a lee wave) are long against a traverse of seconds. **It stops being
true for Dryden**, which is a stochastic process in time — see §7's extensibility table.

### E4. Wind is sampled once per step and held across all four RK4 stages

**Where:** `integrate.step`, documented in its module docstring as the standard treatment
for Dryden and von Kármán.

**Why it was the one seam session 11's verification did NOT cover.** The order-of-accuracy
test flies at fixed controls in still air, so it cannot see a wind term evaluated at the
wrong stage. The Galilean test uses a *steady* wind, whose material derivative is zero, so
it could not see a spurious `−m·dW/dt` term either — the other error `PROJECT.md` §2
explicitly warns about.

**Bound, measured session 12: no spurious body force, to 1e-9 m against an exact solution.**
Two tests, in `test_verification.py`:

| Test | Instrument | Result |
|---|---|---|
| `..._time_varying_uniform_wind_adds_no_body_force` | every aerodynamic coefficient and the thrust zeroed, so free fall is the **closed form** and the wind has no legitimate route into the equations at all; flown through a uniform wind swinging at 3 rad/s with peak \|dW/dt\| = 91 m/s² | position matches `p₀ + v₀t + ½gt²` to **1e-9 m** over 300 steps |
| `..._step_ignores_the_wind_the_previous_step_applied` | full 747 aerodynamics; one step taken twice, varying **only** `SimState.wind_ned` — the cached previous wind, which is the ingredient such a term would be differenced from | **bit-identical** |

**An invariance assertion is the wrong instrument here, and that is worth recording.**
Writing `ṽ_b = v_b − Cᵀ W(t)` for the air-relative body velocity and differentiating gives
`ṽ̇_b = F(ṽ_b,ω)/m + g_b − ω×ṽ_b − Cᵀ Ẇ`: the air-relative state obeys the still-air
equation **plus** a `−Cᵀ Ẇ` term. So a time-varying wind is *not* a change of inertial
frame, and two runs offset by `W(0)` genuinely must diverge. The seam needs a closed form,
not an invariance.

**The falsification was run, since a test that can only pass demonstrates nothing.** With
the bug injected into `step`, both tests fail by many orders of magnitude — and the
Galilean test **passes with the bug still in** (2.7e-15 on quaternion, 6.9e-16 on `ω`,
against its own 1e-11 tolerances) once the wind cache is seeded consistently. Its blindness
is therefore measured rather than argued.

**Verdict: closed.** The seam Dryden will load is the one now covered.

---

## F. Numerics

### F1. Fixed-step RK4 at 50 Hz

**Bound, measured session 11:** observed order **3.99982** on a closed-form problem and
**3.98913** through the real 6-DOF. Asserted nowhere before that session.

### F2. The quaternion is re-normalised every step

**Where:** `integrate.step`, after `rk4_step`.

**Bound, measured:** `max |1 − ‖q‖| = 1.1e-16` after 3,000 steps — machine precision. The
projection is removing essentially nothing, which is the desired result: a normalisation
doing real work every step would mean the integrator was drifting off the unit sphere and
the projection was hiding it.

### F3. float64 throughout

Documented in `PROJECT.md` §1, with the reason: a Newton trim solve to 1e-10 and quaternion
norm stability over 1e5 steps are both marginal in float32.

### F4. There is a round-off floor on trajectory differences

**Bound, measured session 11:** at 40,000 ft `pos_ned` carries a 12,184 m altitude that
float64 resolves to 2.7e-12 m. The discretisation error reaches that floor near **7e-11 m**,
and refining past dt = 1/128 makes the answer *worse* — measured pairwise order **−0.685**
at dt = 1/256.

**Verdict: not an assumption so much as a ceiling, and it binds anything that measures a
difference of two trajectories at altitude.** It is why the order-of-accuracy window stops
at dt = 1/32. Any future convergence study must check it is above the floor before
believing its own slope.

---

## Summary: which assumptions need action

| | Assumption | Status | Action |
|---|---|---|---|
| 1 | **E4** wind held across RK4 stages | **CLOSED, session 12** | no spurious body force, to 1e-9 m against a closed form |
| 2 | **B1** rigid airframe vs flexible data | **unquantifiable** | cap claims; do not assert structural fidelity |
| 3 | **C3** derivatives frozen across the envelope | **unbounded** | state the excursion with every result away from trim |
| 4 | **E2** point-aircraft gusts, vortex at 2.3–3.1 spans | **newly bounded** | record in §5; keep vortex claims as orderings |
| 5 | **A2** constant g, +0.383% at cruise | **CLOSED, session 12** | not modelled: worst mode movement is 7.6% of its tolerance. Phugoid only carries the full 0.38% |
| 6 | **C5** no thrust moment, no spool | sound for now | required before any powered-recovery result |
| 7 | **B4** accelerometer at CG vs DFDR | caveat | keep Fig. 8 claims as orderings |

Items 1 and 5 — the two session 11 flagged as new and actionable — are both closed by
measurement, and in both cases the measurement changed the answer the reasoning had given.
Items 2 and 3 are honest limits rather than bugs, and the correct response to both is to
stop short of claims they cannot support.

---

## How the notebook demonstrates validity

`notebooks/solver-validation.ipynb` exists because a table of numbers in a document cannot
be interrogated, and review specifically asked for an interface where problems with known
answers could be run by hand.

**What makes it evidence rather than decoration is the division of labour.** The notebook
contains **no arithmetic**. Every number it displays is computed by `flightsim.verification`
or `flightsim.validation`, and every one of those numbers is also asserted by a test in
`flightsim/tests/`. So the notebook cannot drift from the code, and it cannot quietly
disagree with the suite. `pytest --nbval-lax notebooks/` executes it as a required gate, so
a notebook that stops working fails the build rather than rotting.

It demonstrates validity in three distinct registers, and the distinction is the point:

**1. Verification — run a problem whose exact answer is known.** Refine the timestep on a
harmonic oscillator and watch the error fall as dt⁴. Nothing about aircraft enters. If this
fails, the integrator is broken and no aerodynamic data can rescue it. The notebook plots
the refinement against a slope-4 reference line, so the claim is visual rather than a
single number to take on trust — and the cell that follows shows how close the finest step
is to the round-off floor (F4), which is the part a bare slope would hide.

**2. Sensitivity — change one coefficient, get a predicted change.** This is review's
request read literally. The notebook sweeps CD0 and Cmα and overlays the analytic law on
the model's output. What makes it convincing is that the sweeps are *falsifiable*: three of
the four relations originally expected were the wrong functional form, and the notebook
shows the affine fits that actually hold, with the intercept labelled as the term the
textbook approximation drops. A sweep that could only ever agree would demonstrate nothing.

**3. Reproduction — match an independent implementation element by element.** The notebook
prints the model's plant matrix beside Caughey's published one with a per-element
difference column, then shows the two omitted elements being reconstructed from his own
tabulated α̇ derivatives. This is the strongest form available: same input data, different
code, published intermediates. A reader can check any single element by hand.

**What the notebook deliberately does not do** is claim the model behaves like a real
Boeing 747. Every tier-2 comparison is closed-loop against a document's own arithmetic —
which is exactly why the age of that document is not a threat, and equally why the result
says nothing about the real aeroplane. The closing cell states that limit explicitly, and
this file is where the reasons live.

**To extend it:** add the computation to `verification.py` or `validation.py`, assert it in
the corresponding test, then add a notebook cell that calls it. Never the other way round.
