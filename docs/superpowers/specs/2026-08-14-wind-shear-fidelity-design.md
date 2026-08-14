# Wind-shear fidelity: from a point sample to a distributed airframe

**Date:** 2026-08-14
**Status:** design, approved for planning
**Theory backbone:** Robert F. Stengel, *Flight Dynamics*, 2nd ed., Princeton University Press,
2022. Chapter 2 §2.1, Chapter 3 §3.2 and §3.4. Page and equation numbers throughout are that
edition's.

**Goal, as set by the user:** enable flow fields whose length scale approaches a wingspan —
wake vortices, span-correlated turbulence, tighter atmospheric rotors. This puts the
higher-fidelity path in the production loop rather than in a one-off measurement.

**Evidence standard, as set by the user:** every constant must be classified, in a document,
as bulletproof-from-a-cited-table or as predicted/calibrated. A calibrated number earns no
credit for landing in a plausible range. §5 is that ledger and it is a first-class
deliverable, not documentation added afterwards.

---

## 1. What is wrong today

`wind.field_model` evaluates the wind field at exactly one point — `state.pos_ned`, the CG —
and derives the rotational gust from the analytic Jacobian at that same point
(`wind.gust_rates`). Three consequences, in increasing order of severity for the stated goal:

1. **Only three of the six available rate-equivalences are used.** The model forms
   `p_gust = +∂w_g/∂y`, `q_gust = −∂w_g/∂x`, `r_gust = +∂v_g/∂x`. Stengel's eqs. 3.4-48 to
   3.4-53 list six; the other three (`∂v/∂z`, `∂u/∂z`, `∂u/∂y`) are dropped entirely.
2. **The gradient is a tangent at the CG, not a fit across the airframe.** For a field with
   curvature over the span, the slope at the centreline is not the slope the wing responds to.
3. **Second-order variation across the span is not represented at all**, and a rigid-body
   coefficient build-up has no channel that could carry it.

`ASSUMPTIONS.md` §E2 already records (3) and bounds the scales: the Parks vortex core is
**2.30–3.07 wingspans** against 39.6 for the updraft and 104.8 for the lee wave. Stengel's
own validity condition for the point-plus-gradient method is that the disturbance be large
compared with the aircraft (p. 215), and he states plainly that the method has limited value
for wake vortices, where strip theory or CFD is required (p. 217). The project's target
fields sit on the wrong side of that line.

### 1a. The cleanest statement of the current approximation

Derived during this design pass, and useful because it is testable:

> The current model is exactly equivalent to assuming **the shear matrix has rigid-rotation
> structure** — that `∂v/∂z = −∂w/∂y` and `∂u/∂z = −∂w/∂x`.

When that holds, the model's aggregate-derivative treatment and Stengel's eq. 3.4-54 agree
**identically**. They diverge precisely to the extent the real field departs from it. This
gives a cheap diagnostic — compute both members of each pair and take the ratio — which §6
adopts as a reporting requirement.

Applied to the Parks vortex:

| Axis | Diagnostic | Consequence |
|---|---|---|
| Roll | Both terms zero — the field has no lateral wind and no lateral variation | **No error** |
| Pitch | `∂w/∂x = +V₀/r₀` and `∂u/∂z = −V₀/r₀` inside the core | **Two equal, additive contributions; the model captures one** |

So the pitch channel is where the current treatment actually loses something on the headline
result, and it loses a factor of order two, not a rounding term.

---

## 2. Verification of the source theory

The user's instruction was to introduce only verified theory and to double-check adjacent
formulations. This was done **before** any design work, by re-deriving Stengel's results
independently rather than transcribing them.

**Method.** Expand the relative flow at a body-fixed point:

```
v_rel(r) = v_cg + ω × r − w_g(r),      ω × r = (qz − ry,  rx − pz,  py − qx)
```

Taylor-expand `w_g` about the CG and match coefficients. Then apply a **rigid-rotation
self-consistency test** to each composite formula: substitute a shear matrix that exactly
matches a rigid rotation and confirm the moment returned equals the equivalent-rate moment.

### 2a. Results

| Formulation | Verdict |
|---|---|
| Eq. 2.1-11 / 3.4-46, spatial shear matrix | ✅ Confirmed. Already implemented as `jax.jacfwd(field)` |
| Eq. 3.4-47, similarity transform `W_B = H W_I Hᵀ` | ✅ Confirmed. Already implemented as `dcm.T @ jac_ned @ dcm` |
| Eq. 3.4-48 `∂w/∂y = p` | ✅ Confirmed |
| **Eq. 3.4-49 `∂v/∂z = p`** | ❌ **Sign error as printed. Should be `−p`.** |
| Eq. 3.4-50 `∂w/∂x = −q` | ✅ Confirmed |
| Eq. 3.4-51 `∂u/∂z = q` | ✅ Confirmed |
| Eq. 3.4-52 `∂v/∂x = r` | ✅ Confirmed |
| Eq. 3.4-53 `∂u/∂y = −r` | ✅ Confirmed |
| Eq. 3.4-54, roll moment from shear | ✅ Self-checks — **but only** once 3.4-49's sign is corrected |
| Eq. 3.4-55, pitch moment from shear | ❌ **Fails by a factor of −2. Not used.** |
| Eq. 3.4-56, yaw moment from shear | ✅ Self-checks with the corrected sign |
| Eq. 3.2-85 / 3.2-86, air-relative velocity and angles | ✅ Confirmed. Already implemented |
| Eq. 3.2-119, acceleration at an offset point | ✅ Confirmed (not yet used; see §7) |

**On eq. 3.4-49.** For `ω = (p,0,0)` at a point directly below the CG, `ω × r = (0, −pz, 0)`.
Physically: roll right-wing-down and the belly swings *left*. The printed `+p` is a sign slip.
That eqs. 3.4-54 and 3.4-56 self-check *only* with the corrected sign is good evidence the
composite equations were derived correctly and the error is confined to that one line.

**On eq. 3.4-55.** Substituting a rigid pitch rate gives `∂w/∂x = −q` and `∂u/∂z = +q`, so the
bracket `(∂w/∂x − ∂u/∂z)` evaluates to `−2q` and the formula returns `+2·M_q·q` where it should
return `−M_q·q`. A factor of ½ repairs the magnitude but cannot be justified: a 50/50 split
between the two mechanisms is not physical, because the tail-incidence mechanism (`∂w/∂x`,
eq. 3.4-4) dominates `M_q` for a conventional aircraft while `∂u/∂z` principally alters the
tail's *dynamic pressure* rather than its angle of attack.

**Design consequence, and the reason for choosing §3's approach:** the pitch limitation
identified in §1 is real, but the book's own formula for repairing it is not usable. That
removes the option of simply transcribing eqs. 3.4-54 to 3.4-56, and is the single strongest
argument for a method that never forms an equivalent rate at all.

### 2b. Confirmed-correct existing behaviour

The three gust-rate signs currently in `wind.gust_rates` were independently re-derived and are
**correct**. No change is required to them, and none is proposed. This is recorded so that a
future reader does not "fix" a working sign convention.

---

## 3. Approach

**Chosen: spanwise and longitudinal strip integration, delivered in two stages.**

### 3a. Why this and not the alternatives

| Option | Rejected because |
|---|---|
| Transcribe eqs. 3.4-54 to 3.4-56 (per-surface split) | Needs wing / tailplane / fin derivative splits and tail geometry. CR-2144 tabulates **whole-aircraft** dimensional derivatives only; `Aircraft` carries no tail arm, tail area, fin height or CG chordwise position. The split would have to be estimated, failing the evidence standard. And eq. 3.4-55 fails verification (§2a) |
| Use all six equivalences with aggregate derivatives | Cheap and needs no new data, but **does not fix curvature**, so it does not enable small-scale fields — it fails the stated goal. The pitch combination is exactly the unresolved weighting of §2a |
| Second-order Taylor correction | Dead end. A coefficient build-up has no derivative for "the spanwise gust profile is bent". Would require inventing derivatives no source tabulates |

Strip integration is the only route that addresses all three limitations, and it is what
Stengel names as the required method below rotor scale (p. 215 for the estimation route,
p. 217 for its limit). It also generalises to any field without further per-field work,
which is the stated goal.

### 3b. Stage A1 — replace the tangent with a fit

**What changes.** `gust_rates` currently returns the analytic Jacobian at the CG. A1 evaluates
the field at a set of body-fixed stations spanning the airframe and fits the best
uniform-plus-linear profile across those samples, returning the fitted slopes.

**Scope discipline — the same three components, computed better.** A1 deliberately does *not*
add the three missing equivalences, because combining each pair into one effective rate
reintroduces exactly the weighting question that §2a left unresolved. A1 is a pure accuracy
improvement to quantities already in use: a secant across the airframe in place of a tangent
at its centre.

**Why this is a real improvement.** Stengel eq. 3.4-39 gives the physical content: a roll rate
adds a spanwise incidence increment `Δα = p·y/V`, linear in `y`, and `Clp` is the integral of
that distribution (his Fig. 3.4-3). The aerodynamics respond to the profile across the span,
not to its derivative at the centreline. Where the field is curved, the fit is the better
estimator of what the wing integrates.

**New quantities required: exactly one.** The fit needs an extent in each direction.

- *Lateral extent:* `±b/2`. `b` is **sourced** (CR-2144 Table IX-3).
- *Longitudinal extent:* an effective tail arm `l_eff`, which is **not** in the model. It is
  recovered from sourced derivatives by an exact relation — see §3d.

**Reduction property (mandatory test).** For a uniform or exactly linear field, the fitted
slopes equal the analytic Jacobian, so A1 must reproduce today's trajectories to round-off.
Any difference is a defect.

### 3c. Stage A2 — integrate forces over the strips

**What changes.** Rather than collapsing the sampled field to three numbers, compute the local
incidence at each strip, weight by the local lift contribution, and integrate to forces and
moments directly.

**Why this is the stage that fixes curvature.** An equivalent rate is by construction a linear
profile; three numbers cannot carry a bent one. Only integration retains it.

**Calibration.** The integration needs a spanwise lift distribution. The proposal is:

- **Shape: elliptic.** A one-parameter shape needing no taper ratio (which the model does not
  carry). Classified **DECLARED** in §5.
- **Magnitude: pinned to sourced `Clp`.** The distribution is scaled so that integrating a
  rigid roll rate reproduces CR-2144's tabulated `Clp` exactly. Classified **CALIBRATED**.
- **Sensitivity: mandatory.** Re-run against a taper-based and a uniform distribution and
  report the spread on every result. A single shape reported without its spread is not an
  acceptable output.

**Independent cross-check available.** Stengel eq. 3.4-40 gives a closed-form strip-theory
`Clp` in terms of lift slope and taper ratio. Where taper ratio can be sourced, this checks
the integration machinery against the book independently of the calibration.

### 3d. The derived effective tail arm

This is the one genuinely new piece of theory in the design, so its derivation is given in
full and its provenance is explicit.

From Stengel eq. 3.4-7, the tail's lift response to pitch rate is
`CL_q,ht = CL_α,ht · (l_ht/V)`. Non-dimensionalising with `q̂ = qc/2V` (eq. 3.4-10):

```
CL_q̂,ht = 2 · CL_α,ht · (l_ht/c)
```

From eq. 3.4-12, the corresponding moment derivative is

```
Cm_q̂,ht = −2 · (l_ht/c)² · CL_α,ht
```

The ratio eliminates the unknown tail lift slope entirely:

```
l_eff / c  =  − Cm_q̂ / CL_q̂
```

Both inputs are sourced. Evaluated on CR-2144 Table IX-3/IX-4 for the 747 at FC9:

| Quantity | Value |
|---|---|
| `CL_q̂` | 5.9450 |
| `Cm_q̂` | −23.9232 |
| `l_eff/c` | **4.0241** |
| `l_eff` | **109.90 ft = 33.50 m** |

**Independent plausibility check:** the real 747-100 CG-to-tailplane distance is of order
100–110 ft. The derived arm lands inside that band without having been told anything about
747 geometry beyond `c`. The Cherokee set gives 3.85 chords, also plausible.

**A defect this check catches.** The Cessna 172 set gives `l_eff/c = 0.856`, which is not
physically credible for that airframe. This indicates its `CLq` (Roskam/DATCOM via PyFME) is
inconsistent in normalisation with its `Cmq` — consistent with `PROJECT.md` §3's existing note
that the same source's rudder derivatives are inconsistent and were zeroed. **Gate:** an
aircraft whose derived `l_eff/c` is not physically plausible is excluded from the strip path
and must fall back to the current point model, with the exclusion recorded.

**Stated assumption.** The relation attributes both derivatives to the tail. Stengel eq. 3.4-19
notes explicitly that `Cmq` also carries wing and fuselage contributions, and eqs. 3.4-13/3.4-14
give the wing's own share as a function of CG chordwise position `h_cm`.

**`h_cm` turns out to be sourced — but the correction it would enable does not work.** The
source audit (§3e) found `h_cm = 0.25 c̄`, tabulated for every flight condition. Substituting it
into eq. 3.4-13 gives a wing contribution of `+CL_α ≈ 4.94` against a total `CL_q̂` of 5.945 —
i.e. the wing would account for **83%** of `CL_q̂`, leaving a tail contribution that implies a
23-chord tail arm. That is not credible. The reason is that eqs. 3.4-13/3.4-14 are Etkin's
**two-dimensional, infinite-aspect-ratio** result, which Stengel is explicit about (p. 208), and
it does not transfer to a finite swept wing with a tail.

**Independent confirmation that the tail-dominated reading is the right one:** the standard tail
volume estimate `CL_q̂ ≈ 2·CL_α,ht·η·V_H` with representative 747 values (`V_H ≈ 0.95`,
`CL_α,ht ≈ 3.5`, `η ≈ 0.9`) gives ≈ 6.0 against the tabulated 5.945 — the tail alone accounts for
essentially all of it, with no room for an 83% wing share.

**Decision: the eq. 3.4-13/3.4-14 correction is examined and rejected**, and `l_eff` stands on
its own plausibility check. Classified **DERIVED (under a stated assumption)** in §5, never as
sourced. Recorded here so the correction is not re-attempted.

### 3e. Source audit of CR-2144 — §9 item 1, resolved

The design originally deferred the question of what CR-2144 actually contains. It has now been
answered by reading the document itself: NASA CR-2144 was retrieved from NTRS
(`ntrs.nasa.gov/citations/19730003312`, public domain), and Section IX examined page by page.
The scan carries no text layer, so pages were decoded from their CCITT Group 4 streams and read
as images.

**Findings, and their effect on this design:**

| Quantity | In CR-2144? | Where | Effect |
|---|---|---|---|
| `S`, `b`, `c̄` | ✅ **SOURCED** | Table IX-3 header; repeated on Figure IX-2 | Already used |
| **CG chordwise position `h_cm`** | ✅ **SOURCED = 0.25 c̄** | Table IX-3 row `C.G.(MGC)`, all 10 flight conditions; Figure IX-1; Figure IX-2 as F.S. 1339.9 | §3d — but the correction it enables fails; see above |
| **MGC spanwise station** | ✅ **SOURCED = B.L. 491** | Figure IX-2 | Available for strip-station geometry |
| **Pilot station offset** | ✅ **SOURCED = LXP 86.0 ft, LZP −10.0 ft** | Table IX-3 | **Resolves 7i** — see below |
| **Taper ratio** | ❌ **NOT TABULATED** | Planform drawn on Figure IX-2 only | 7c stands: loading shape stays DECLARED |
| **Tail area, tail arm, fin geometry** | ❌ **NOT TABULATED** | Drawn on Figure IX-2 only | §3a's rejection of the per-surface split stands |
| Sweep angle | ❌ NOT TABULATED | — | Existing model value keeps whatever provenance it already had |

**On the drawing.** Figure IX-2 (*B-747 General Arrangement*, printed p. 213) is a dimensioned
3-view: a 0–100 ft scale bar, fuselage stations, butt lines and water lines in inches. Tail area
and tail arm are therefore *measurable*, and a measurement off a dimensioned drawing is
meaningfully stronger evidence than the log-axis curve reads `PROJECT.md` §7 declines. **This is
recorded as an available option, not adopted.** If taken, it is CALIBRATED or DECLARED with a
stated reading uncertainty — never SOURCED.

**Cross-check that validates the pilot-station reading.** Figure IX-2 marks the CG at F.S. 1339.9
and the forward station at F.S. 307.9. The difference is 1032 in = **86.0 ft**, exactly Table
IX-3's `LXP`. Two independent places in the document agree, which is what makes this SOURCED
rather than inferred.

**Transcription verification, obtained for free.** Tables IX-3, IX-4 and IX-8 were compared
element by element against `aircraft.py`. **Every value matches**, including `Mq = −0.339` (the
value a previous session corrected from −0.330) and all ten lateral primed derivatives. The
existing transcription is confirmed against the source.

---

## 4. Architecture

Four small units, each independently testable, following the module boundaries the project
already uses.

| Unit | Responsibility | Depends on |
|---|---|---|
| `airframe.py` (new) | Sample-station geometry: where on the airframe the field is evaluated. Derives `l_eff`, exposes the elliptic distribution and its alternatives | `Aircraft` |
| `wind.sampled_rates` (new, beside `gust_rates`) | Stage A1. Evaluates the field at the stations and fits the linear profile | `airframe`, the field |
| `wind.strip_loads` (new) | Stage A2. Local incidence per strip, weighted integration to forces and moments | `airframe`, `aero` |
| `provenance.py` (new) | The ledger as data, with the enforcing test | — |

**The existing `gust_rates` is retained, not replaced.** It remains the reference
implementation that A1 must reduce to, and the fallback for aircraft failing the §3d gate.
This follows the project's existing pattern of keeping `verification.py` separate so a check
cannot drift from the thing it checks.

**Wind-model contract is unchanged.** `wind_model(wind_state, state, key, dt) -> (wind_ned,
omega_gust, wind_state, key)` still holds for A1, so `integrate.step`, `autopilot` and
`panel` need no changes. A2 needs a wider contract and that widening is called out in the
plan as its own reviewed step, not smuggled in.

---

## 5. The provenance ledger

The user's requirement, and a first-class deliverable. Four categories, mutually exclusive:

| Category | Definition | Example |
|---|---|---|
| **SOURCED** | Read directly from a cited table, with document, table and page | `b = 195.68 ft`, CR-2144 Table IX-3 |
| **DERIVED** | Computed from SOURCED values by a stated exact relation. The relation is cited; the number is not independently checkable | `AR = b²/S`; `l_eff/c = −Cm_q̂/CL_q̂` (Stengel 3.4-7, 3.4-12) |
| **CALIBRATED** | Fitted so the model reproduces a SOURCED number to a stated tolerance | Elliptic distribution magnitude, pinned to `Clp` |
| **DECLARED** | Chosen. Not derivable from any source the project holds. Requires a reported sensitivity | Loading shape; strip count; updraft edge sharpness (existing) |

**Scope: the whole model, not only the new work.** The user asked that all such data be
covered. The ledger retrofits every constant in `aircraft.py`, `atmosphere.py` and `wind.py`,
not just the constants this design introduces. Existing prose in `PROJECT.md` §3 and
`ASSUMPTIONS.md` covers much of this narratively; the ledger makes it per-constant and
machine-checkable.

**Enforcement, so it cannot rot.** The project already enforces that every reference number
carries a citation via a `Reference.source` field and a test. The ledger follows that pattern:
each constant carries its category and citation as data, and a test asserts that

1. every constant appears in the ledger,
2. every SOURCED entry carries document, table and page,
3. every DERIVED entry names its relation and its inputs, and every input is itself in the ledger,
4. every CALIBRATED entry names its target and tolerance, and a test demonstrates the target is met,
5. every DECLARED entry has a reported sensitivity range.

A constant added without a ledger entry fails the build. This is the mechanism that makes the
answer to "which numbers are bulletproof?" a query rather than a memory.

**Reporting rule.** Any result quoted from the strip path states the category mix it depends
on. A result resting on a DECLARED shape is reported with its sensitivity spread, never as a
bare value.

---

## 6. Validation

Every gate is against numbers the project already holds. Nothing here requires new source data.

| # | Check | Instrument | Gate |
|---|---|---|---|
| 1 | Uniform field | A1 and A2 vs current model | **Bit-identical** |
| 2 | Exactly linear field | A1 vs current model | Round-off |
| 3 | Rigid roll rate through the strip integration | vs CR-2144 `Clp` | Calibration target; must be exact by construction, asserted anyway |
| 4 | Rigid pitch rate | vs CR-2144 `Cmq`, `CLq` | Must reproduce both from one `l_eff` |
| 5 | Strip theory closed form | vs Stengel eq. 3.4-40 | Independent of the calibration |
| 6 | Derived tail arm | `l_eff/c` physically plausible | §3d gate; excludes the Cessna |
| 7 | Rigid-rotation-structure diagnostic | Ratio of each shear pair, per field | Reported for every field, not asserted |
| 8 | 747 modes | All five, before and after | Unmoved — `PROJECT.md` §4 baselines are off-limits |
| 9 | Parks vortex Fig. 8 point | Δθ, Δn, ordering | Movement **quantified and explained**, not required to be zero |
| 10 | Curvature sensitivity | Point vs A1 vs A2 on the Parks core | This is the E2 bound, finally measured |

**Check 9 is deliberately not a no-change gate.** The whole purpose is to change the vortex
answer. What is required is that the movement is explained by check 10 and reported, and that
the *ordering* claim `vortex < updraft < manoeuvre` survives — since `PROJECT.md` §5 already
restricts vortex conclusions to orderings.

**Check 10 closes `ASSUMPTIONS.md` §E2**, which currently carries a scale ratio but no measured
consequence. That is a stated project objective independent of this work.

---

## 7. Limitations

Included at the user's request. These are the things this design does **not** fix, stated so
no reader over-reads the result.

**7a. Strip theory is still not CFD.** Stengel (p. 217) gives two reasons the method degrades
for wake vortices, and this design addresses only the first. It samples the field across the
airframe, but it still treats each strip with two-dimensional section behaviour and no
spanwise induced-flow coupling between strips. A vortex-lattice or CFD treatment would carry
the induced interaction; this will not.

**7b. The aircraft remains transparent.** Stengel's second reason (p. 217) is that the
receiving aircraft's own flow field distorts the impinging vortex, coupling response back into
the disturbance. This design keeps one-way coupling (`ASSUMPTIONS.md` §E1) and therefore does
**not** make genuine wake-vortex encounters trustworthy — it makes them *less wrong*. Any
wake-vortex result must carry this caveat explicitly.

**7c. The calibrated distribution is a shape assumption.** Pinning the magnitude to `Clp`
guarantees the *integral* is right for a rigid roll rate. It does not guarantee the shape is
right, and a curved gust field weights the span differently from a linear one. This is why
§3c makes the sensitivity sweep mandatory rather than optional.

**7d. `l_eff` folds wing and fuselage pitch damping into a tail arm.** §3d states the assumption.
`h_cm` is sourced (§3e), but the textbook correction it would enable is a 2-D result that does
not transfer to this configuration and was rejected, so the assumption is **justified
empirically rather than bounded analytically**: by the derived arm landing inside the real
aircraft's 100–110 ft, and by an independent tail-volume estimate accounting for essentially all
of `CL_q̂`. That is weaker than a computed error bar. The number is DERIVED, never SOURCED, and
must not be quoted as 747 geometry.

**7e. Quasi-steady aerodynamics are unchanged.** `ASSUMPTIONS.md` §C2 stands. Stengel
eq. 3.4-58 gives the unsteady contribution of a time-varying vertical gust through the `α̇`
derivative; the model has no `α̇` term, so that channel remains absent. This matters more as
fields get smaller, because the traverse time falls — so this limitation *grows* exactly where
the design is aimed. It is the first thing to revisit afterwards.

**7f. Constant derivatives across the envelope.** `ASSUMPTIONS.md` §C3 stands and is unbounded.
Strip integration redistributes the same derivatives spatially; it does not make them
condition-dependent.

**7g. The three unused equivalences remain unused.** A1 and A2 both avoid the eq. 3.4-55
weighting question rather than solving it. A2 sidesteps it structurally — sampling the tail's
actual location captures `∂u/∂z` implicitly — but the model never forms the missing rates
explicitly, so anything expecting them by name will not find them.

**7h. Rigid airframe.** `ASSUMPTIONS.md` §B1 stands, and worsens here. Stengel notes (p. 38)
that wind shear has persistent effect on aeroelastic modes. Sampling a stiffer gradient across
the span makes the rigid assumption *more* strained, not less, because the loads this design
newly resolves are exactly the ones that would flex a real wing.

**7i. Accelerometer still at the CG — but this is now unblocked.** `ASSUMPTIONS.md` §B4 stands
*as an unimplemented correction rather than an impossible one*. This entry originally said the
lever arm needed a sensor offset the project does not hold. **That was wrong:** the source audit
(§3e) found `LXP = 86.0 ft`, `LZP = −10.0 ft` in Table IX-3, the pilot station relative to the
CG, cross-checked against Figure IX-2's fuselage stations. With Stengel eq. 3.2-119 already
verified (§2a), the correction is implementable entirely on sourced data.

The residual limitation is one of *interpretation*, not data: the pilot station is not the DFDR
accelerometer station, so this would give pilot-station normal acceleration — a standard
handling-qualities quantity, and the one Stengel notes (p. 195) governs what the crew actually
feels — rather than closing the Fig. 8 comparison exactly. Still out of scope for this design,
but it is now a scheduling decision rather than a blocked one.

---

## 8. Explicitly out of scope

- Adding an `α̇` derivative (7e) — separate work, separate sourcing question.
- Implementing eq. 3.2-119 (7i) — needs data the project does not have.
- Any change to the three existing gust-rate signs — verified correct (§2b).
- Any change to `PROJECT.md` §4 mode baselines — off-limits to feature work.
- Re-deriving eq. 3.4-55 into a usable form. Possible, but it is a research task with no
  source to check the answer against, and §3's approach makes it unnecessary.

---

## 9. Open items for the plan

1. ~~Confirm whether CR-2144 supplies taper ratio and tail geometry.~~ **RESOLVED — see §3e.**
   Neither is tabulated; both are only measurable off the Figure IX-2 3-view. §3c's loading
   shape therefore stays **DECLARED** and 7c stands as written. The audit did, however, return
   three quantities the design had assumed unavailable: `h_cm`, the MGC spanwise station, and
   the pilot-station offset.
2. Choose the strip count by convergence, not by taste. It is a DECLARED parameter and needs a
   refinement study showing the answer has stopped moving.
3. Decide whether A2's wider contract replaces the current one or runs beside it. The plan
   should treat this as a reviewed decision point once A1's numbers are in.
4. **New, from §3e:** decide whether to measure tail area and arm off Figure IX-2. It would make
   Stengel's per-surface split (eqs. 3.4-54/3.4-56, roll and yaw only — 3.4-55 remains unusable)
   reachable as a CALIBRATED entry with a stated reading uncertainty. Not required by A1 or A2;
   worth a decision rather than silent omission.
