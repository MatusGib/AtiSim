# Model fidelity improvements from the JSBSim comparison — Design Spec

Date: 2026-08-20

## Purpose

Four changes arising from the JSBSim cross-code verification
(`2026-08-20-jsbsim-737-verification-design.md`). The standing direction for this work is that
**JSBSim's methods are treated as good practice and changes should tend toward them** — with one
qualification recorded below, where JSBSim's *dataset* and JSBSim's *method* diverge and following
the dataset would import an artifact.

Every change carries a reference that has been **read and verified in the copy held in `refs/` or
the project root**, not cited from memory. Where no such reference could be found, that is stated
rather than papered over.

| # | Change | Reference | Status |
|---|---|---|---|
| 1 | Sideslip drag `CD_beta` | none found — see §1 | **flagged**, form derived from evenness |
| 2 | α̇ derivatives `CLα̇`, `Cmα̇` | Stengel Eq. (3.4-25), (3.4-26) | verified |
| 3 | Verification at a second condition | n/a — test coverage | — |
| 4 | Aerodynamic reference point (AERORP) | Stengel Eq. (2.4-68), §2.4 | verified, **implemented** |

Reference copies: `Flight_Dynamics_-_Second_Edition.pdf` (Stengel, *Flight Dynamics*, 2nd ed.,
Princeton University Press, 2022 — already this project's theory backbone, cited by
`airframe.py` and `scripts/turbulence_report.py`), and `refs/Caughey-MAE5070-...pdf`.

---

## 1. Sideslip drag — the reference could not be found, and that is informative

**Measured need.** Integrating the drag terms AtiSim lacks over the recorded sideslip history
accounts for **1.015 m/s of the rudder kick's 1.516 m/s** velocity divergence over 20 s. It is the
single largest contributor to that case.

**No formula for `CDβ` appears in Stengel.** Searching the full text for "drag due to sideslip",
"sideslip drag" and "crossflow" returns the crossflow discussion supporting `Clβ` (Eq. 2.4-92,
2.4-93) and nothing for an axial-force sensitivity to β.

That absence is not an oversight in the text. **Linear small-perturbation theory has no `CDβ`,
because drag is an even function of sideslip**: `∂CD/∂β = 0` at β = 0 identically, by symmetry of a
laterally symmetric airframe. The leading term is second order, `∝ β²`. So AtiSim's linear
model is *correct within its own framework* to omit it, and adding it is a deliberate step outside
that framework rather than the filling of a gap.

**Where JSBSim's dataset and JSBSim's method diverge.** 737.xml's `CDbeta` is a five-point table
interpolated linearly, with breakpoints at β = 0 and ±0.26 rad. Linear interpolation through zero
makes `CD ∝ |β|` near the origin — a kink, with `dCD/d|β|` discontinuous at β = 0. Measured
directly: the sweep gives ΔCD = 1.007e-2 at β = ±3° and 5.035e-3 at ±1.5°, **exactly half**, which
is linear in |β| and not quadratic.

An even, smooth drag rise is the physical requirement. So this is the one place where "tend toward
JSBSim" is declined on purpose: we adopt JSBSim's *method* (carry a sideslip drag term) and reject
its *table's form* (a linear-interpolation artifact of a coarse table). The implementation uses

    CD_beta_term = CD_beta * beta**2

fitted so it reproduces JSBSim's table at the working sideslip range, and the layer-1 prediction is
updated to account for the difference in form.

**Status: implemented, with the missing reference recorded here.** The physical basis (fuselage
crossflow drag plus the induced drag of the fin's side force) is standard and uncontroversial, but
no equation from a source held in `refs/` supports a specific coefficient form, so the β² form is
justified by symmetry rather than by citation. Anyone extending this should find a proper source.

---

## 2. Angle-of-attack-rate derivatives

**Reference, verified.** Stengel, *Flight Dynamics*, 2nd ed., §3.4, "Lag of the downwash",
pp. 227–229.

The physical mechanism: the flow angle over an aft tail is modified by the wing's downwash ε, and
pressure disturbances from a change in wing lift convect downstream, reaching the tail after
`Δt = l_ht/V`. The lagged adjustment produces an incremental downwash proportional to α̇:

- Eq. (3.4-23): `Δε = −(∂ε/∂α)(l_ht/V)·α̇`
- Eq. (3.4-24): `ΔCL_ht = CLα_ht·(∂ε/∂α)(l_ht/V)·α̇`
- **Eq. (3.4-25):** `CL_α̇_ht = 2·CLα_ht·(∂ε/∂α)·(l_ht/c̄)`
- **Eq. (3.4-26):** `Cm_α̇_ht = −2·CLα_ht·(∂ε/∂α)·(l_ht/c̄)²`

Both are referred to `α̇_hat = α̇·c̄/(2V)`, the same non-dimensionalisation `aero.py` already uses
for `q_hat`.

**Why this matters here, in Stengel's own words** (p. 227):

> "a plunging aircraft experiences non-zero α̇ with zero q; a pitching and heaving aircraft could
> experience non-zero q with zero α̇. Dynamic forces and moments measured for a wind tunnel model
> that rotates but does not heave are proportional to (α̇ + q)."

That is exactly the gust case. §5 of PROJECT.md already records Żw and Ṁw as "genuinely absent",
with the remedy being a second linear model built for mode extraction. That patches mode
extraction; it does not patch the nonlinear sim used for the F-factor and strip-loads work, where
α̇ and q decouple.

**Magnitude**, using the 737's measured split (`Cmq` = −27, `Cmα̇` = −16 — so α̇ carries 37% of
pitch damping when the two coincide), for a 5 m/s 1-cosine vertical gust:

| gradient | ramp | `Cm(α̇)` as % of `Cm(α)` |
|---|---|---|
| 12.5 m | 0.053 s | **226%** |
| 30 m | 0.127 s | 94% |
| 60 m | 0.254 s | 47% |
| 107 m | 0.452 s | 26% |

At short gradients the missing moment dominates the one that is modelled.

### Where α̇ comes from, and why only the wind part is used

α̇ has two sources: the aircraft's own motion, and the wind field it is flying into.

The **aircraft-motion** part is implicit — α̇ depends on ẇ, which depends on the forces, which
depend on α̇. The closed-form resolution is the `1/(1 − Zẇ)` factor Caughey uses and which §4
already reconstructs to 0.03%. It is not attempted here.

The **wind** part is explicit and exact. `wind.gust_rates` already computes the full Jacobian
`∂(wind_ned)_i/∂(pos_ned)_j` with `jax.jacfwd` in order to produce `omega_gust`. Contracting that
same Jacobian with the NED velocity gives the material derivative of the wind following the
aircraft, with no new differentiation and no finite differencing:

    d(wind_ned)/dt = J @ vel_ned          (frozen-field / Taylor hypothesis)
    alphadot_gust  = -(d(wind_body)/dt)_z / V

**This composes correctly with the existing folded `Cmq`, and does not double count.** For an
aircraft whose `Cmq` carries `Cmq + Cmα̇` (as the 737's does, because JSBSim's engine folds them),
the model produces

    Cm = (Cmq + Cmα̇)·q_hat + Cmα̇·α̇_gust_hat

against a true value of `Cmq·q_hat + Cmα̇·(α̇_aircraft + α̇_gust)_hat`. These agree exactly whenever
`α̇_aircraft = q`, which is the standard short-period condition and the assumption already implicit
in the fold. The gust term supplies precisely the part the aircraft's own motion cannot produce.

For the four pre-existing aircraft, whose sources give a pure `Cmq` with `Cmα̇` absent, the new
fields default to zero and nothing changes.

---

## 3. Verification at a second condition

Not a model change — test coverage. Everything currently rests on one point, 30,000 ft / M 0.78.
`gen_jsbsim_reference.py` is parameterised by altitude and Mach for exactly this reason.

A low, slow condition tests the solver where compressibility vanishes. The project's own §4 records
that its phugoid error falls from 17.8% at M 0.80 / 40,000 ft to **0.4%** at M 0.25 / sea level with
the same code and the same omissions, because compressibility drives the missing terms. A second
reference condition turns that from an inference into a measurement on an independent data path.

The second condition is recorded in the same reference file under its own `<condition>`,
`<trim>` and `<sweep>` elements, and the layer-1 and layer-2 tests run over both.

---

## 4. Aerodynamic reference point — IMPLEMENTED

**Reference, verified.** Stengel §2.4, Eq. (2.4-66) through (2.4-70), pp. 109–111:

> "To this point, we have taken the center of mass as the origin for defining the axial length x,
> **but the center of mass varies with aircraft loading.** To examine the effects of
> center-of-mass variation, we choose a point fixed in the airframe, such as the nose tip or a
> particular fuselage bulkhead, as the origin (or fiducial point), replacing x in the prior
> equations by (x − x_cm)."

- Eq. (2.4-68): `Cm = Cm_c + ((x_cp − x_cm)/c̄)·C_N`
- Eq. (2.4-85): `Cnβ_vt = CYβ_vt·[(x_cm − x_cp_vt)/b]` — the same transfer, laterally

Note Eq. (2.4-68) is written in **`C_N`, the body-axis normal force**, not in `CL`. That is the
whole point: the transfer acts on body-axis force components, which rotate with α relative to the
wind-axis lift and drag. A constant `Cma` cannot represent a term whose α-dependence comes from
that rotation.

**Measured, at 30,000 ft / M 0.78, sweeping α from 0° to 8°:**

| method | worst \|error\| in Cm |
|---|---|
| AERORP-referenced coefficients + explicit `r × F`, with `Cmα̇` | **2.8e-17** |
| same, omitting `Cmα̇` | 7.2e-3 |
| constant-`Cma` fold about the CG (what the 737 entry does today) | 1.6e-2 |

The reference-point method reproduces JSBSim's pitching moment **to machine precision**. It is not
a better approximation — it is exact, because it is the same decomposition JSBSim uses. The fold's
1.6e-2 is the quadratic residual already measured in layer 1 and attributed there to "lift rotating
into body x at the AERORP offset"; this is that attribution confirmed to 17 significant figures.

The measurement also shows the two changes are **entangled**: the reconstruction is only exact with
the α̇ term present, which is independent evidence for §2.

### What it would cost

- Two new `Aircraft` fields for the reference-point offset from the CG in body axes, defaulting to
  zero, so all existing aircraft stay bit-for-bit identical.
- `aero.py` computes the transfer `M += r × F` from the body-axis force it already has.
- The 737's entry becomes **737.xml's own constants** — `Cma` = −0.6, `Clb` = −0.09, `Cnb` = +0.26
  — with the offset generating the rest. That is a strictly better provenance story than the
  recovered values, because the numbers then trace to a file rather than to a finite difference.
- Layer 1's `Cm` bound stops being a two-mechanism estimate and becomes an equality.

### What it actually did

Implemented as `Aircraft.aero_ref`, a body-axis CG→reference vector defaulting to zero, with
`moment += jnp.cross(ac.aero_ref, force)` in `aero_forces_moments`. Every pre-existing aircraft is
bit-identical, because `jnp.cross` of a zero vector is an exact zero.

The generator now refers JSBSim's moments back with `M_arp = M_cg − r × F` before differencing, so
the derivatives are still *recovered from the running engine* rather than transcribed. They land on
737.xml's own constants at **both** recovery conditions, which is the check that the referencing is
right:

| | recovered, AERORP | 737.xml | recovered, about the CG |
|---|---|---|---|
| `Cma` | **−0.599999** | −0.6 | −1.0637 (cruise), −1.0567 (approach) |
| `Clb` | **−0.0899998** | −0.09 | −0.1440 |
| `Cnb` | **+0.2599999** | +0.26 | +0.2730 |
| `Cm0` | **−3.0e−08** | *no such term* | −0.0107 |
| `Cmde` | **−0.849000** | −0.849 | −0.8943 |

`Cma` is now the *same number at both conditions*, as a constant in the file should be; referred to
the CG it was condition-dependent. `Cm0` turning out to be zero is the sharpest result — what
looked like a pitching-moment offset was entirely the AERORP arm.

**The pitch axis needed a least-squares fit, not central differences.** Setting α away from trim
also sets α̇ (measured, `dα̇/dα` = −0.529 /s), which contaminates a differenced `Cma` by +0.067 —
exactly the gap between the −0.5328 a difference gives and 737.xml's −0.600. Fitting
`[1, α, q̂, α̇̂, δe]` over a crossed design separates them and recovers all five: `Cm0` ≈ 0,
`Cma` −0.599999, `Cmq` −27.000, `Cmadot` −16.000, `Cmde` −0.849000, max residual 2e-11.

α̇ and q are nearly collinear in any reachable state, so their **split** is ill-conditioned
(condition number 1.9e8) while their **sum** is exact. The entry therefore carries the sum, folded,
which is what AtiSim needs.

### What it exposed, and this is the important part

Layer 1's lateral moments went from 7.0e-6 / 1.7e-6 to **1.1e-8 / 1.0e-8** — round-off. `Cm` agrees
to **2.5e-11** once one bookkeeping term is accounted for.

But the **short period went from 0.04% to 3.95%**, and the reason is that two errors had been
cancelling:

- the old `Cma` of −1.0637 was the α̇-contaminated central difference, not the −1.1309 truth;
- AtiSim has no aircraft-motion α̇ coupling, so its linearisation is missing exactly the term
  that contamination stood in for.

A wrong coefficient was compensating a missing term, and the modes agreed almost exactly as a
result. With honest coefficients the gap is visible. **That is a better state to be in — the model
is wrong in one identified place instead of right by cancellation — but it is not the same as being
right.** Closing it means giving `derivatives` the implicit α̇ solve, which §2 lists as not done and
which is now the clear next step.

A smaller contribution, 0.6% of the 4%, is the drag error reaching the moment through `r × F`: with
JSBSim's own force in the transfer the effective `Cma` is −1.1331 against the −1.1309 truth; with
AtiSim's it is −1.1666. AERORP makes the moment inherit the force error rather than absorbing it
into a fitted constant, which is correct and is another reason the drag terms matter.

---

## Success criteria

1. Existing suite green; the four pre-existing aircraft and the `conftest` fixture bit-for-bit
   unchanged by items 1 and 2 — asserted, not inspected.
2. Item 1: layer 1's `CD` prediction updated for the β² form and still bounded.
3. Item 2: the α̇ term is zero in still air by construction, asserted; and non-zero in a gust.
4. Item 3: reference carries two conditions; layer 1 and layer 2 pass at both.
5. Every claim above that cites Stengel quotes an equation number verified against the PDF.

## Outcome

**This section was written after items 1-3 and is SUPERSEDED by §4 and §5 below, which
were appended later.** It is left standing rather than rewritten, per the convention the
verification spec uses: the record of what was true at each step stays separable from the
final state. Read §4, §5 and "Explicitly not done" for what actually shipped -- item 4 and
the aircraft-motion α̇ solve were both subsequently done, and `CD_alpha` was added on top.

Items 1, 2 and 3 are implemented; item 4 is not. Measured after the change:

| | result |
|---|---|
| Existing aircraft | bit-for-bit unchanged; both new fields default neutral, asserted |
| Sideslip drag | even in beta to 1e-12, quadratic to 1e-9, meets JSBSim's table at its 0.26 rad breakpoint |
| Angle-of-attack rate | the GUST half is exactly zero in still air and under a uniform wind; correct sign and magnitude in a gradient; reaches pitch rate through the real integrator. (The aircraft's own half, added later in §5's sequence, is not zero in still air -- see `Aircraft.Cmadot`) |
| Second condition | 5,000 ft / M 0.40, alpha 3.63 deg against cruise's 1.97 deg. Layers 1 and 2 pass at both |

**What moves between the two conditions is itself the check.** `Cmde` -0.894 to -1.067 and
`Clda` +0.0739 to +0.0866 because JSBSim schedules both on Mach; `CD0` +0.0271 to +0.0312 and
`Cm0` -0.0107 to -0.0141 because both are read at a larger alpha. `CLa`, `CLde`, `CYb`, `Clp`,
`Clr`, `Cnr`, `Cldr` and `Cndr` are identical to seven figures at both, because 737.xml defines
them as constants. A condition-dependent solver defect could not leave those unchanged while
moving the others by exactly the amounts the tables predict.

Two implementation traps worth recording, both of which produced silent wrong answers rather than
errors:

- **`alt_ft=ALT_FT` as a default argument** binds at definition time, so rebinding the module
  global per condition never reached it. The approach build ran at the cruise altitude with the
  approach Mach -- 30,000 ft at M 0.40 -- which is genuinely untrimmable, and the only symptom was
  a trim failure that looked like a bad choice of condition.
- **The thrust fit band was pinned to 25,000-35,000 ft.** Fitting a 5,000 ft condition against it
  left a 31.3% altitude residual. The band now brackets whatever condition is being built, which
  takes it to 0.83%.

## 5. Closing the loop: CD_alpha, and what the sequence showed

Added after item 4, because item 4 is what made it visible.

`CD_alpha`, a linear profile-drag rise with incidence, recovered from the engine as
`dCD/dα − 2·CL·CLa/(π·e·AR)` — the total drag slope minus the induced part AtiSim already had.
Measured 0.0847 (cruise) and 0.0864 (approach) against 737.xml's CD0 table slope of 0.0808, the
excess being the CDde and ground-effect residue.

**Linear, where `CD_beta` is quadratic, and the asymmetry is the point.** Drag is even about its
minimum in both variables. For sideslip the reference condition sits *at* that minimum, β = 0, so a
linear term would put a kink through the operating point and the quadratic is the only defensible
form. For incidence the reference sits well away from it — α = 1.97° cruise, 3.63° approach — so
the first-order Taylor term is exactly right for a model that is explicitly a linearisation about
that point. It is wrong at negative α, which is the same cruise-local caveat the entry already has.

### The short period, in four steps

| state | cruise | why |
|---|---|---|
| original | 0.04% | **two errors cancelling** |
| after AERORP | 3.95% | honest `Cma`, missing α̇ term now exposed |
| after the α̇ solve | 1.30% | α̇ supplied; drag slope still wrong |
| after `CD_alpha` | **0.04%** | honest throughout, and at **both** conditions |

The first and last are the same number and mean opposite things. The first came from an
α̇-contaminated `Cma` of −1.0637 standing in for a coupling the model did not have. The last comes
from `Cma` = −0.6 referred to the AERORP, a real α̇ coupling, and a real drag slope.

**This is the argument for AERORP referencing, made by measurement rather than by principle.**
Referring moments to the AERORP makes the pitching moment inherit the force error through `r × F`
instead of absorbing it into a fitted `Cma`. A drag slope of 0.1267 against JSBSim's 0.2113 could
then no longer hide, and fixing it moved the short period by 1.3%. The CG-referenced model would
have shown nothing at all — it had a coefficient free to absorb exactly that error.

### What is left

Phugoid 6.6% at cruise, 3.4% at approach. It is a slow drag-and-thrust energy exchange and the
thrust model is still linear in throttle where JSBSim's varies 4.3× across the range. That is the
next thing, if the phugoid matters.

Layer 4 moved 0.483 → 0.558 m/s on the doublet. Trajectory divergence is set by total drag along
the path, not by its slope at one point, and the two are independently adjustable — improving the
slope does not have to improve the integral.

## Explicitly not done

**Corrected after the fact.** This list named item 4 and the aircraft-motion α̇ solve, and both
were then done — item 4 in §4, the α̇ solve in the sequence §5 records, where it is the step
that took the short period from 3.95% to 1.30%. The α̇ solve is a one-pass opening of the loop
rather than the closed-form `1/(1 − Zẇ)` factor, which is exact for every aircraft in this
registry because all of them carry `CLadot = 0`; `dynamics.derivatives` records the residual for
the case where one does not.

Genuinely not done, and still open: spool dynamics, stall, and banked trim — all previously
identified, none in this round. Also `CDde` (JSBSim's `0.059·|δe|`, absorbed into `CD0` at the
trim elevator), Mach scheduling of `Cmde` and `Clda`, and a recovery condition above M 0.8,
which is the only thing that would test `wave_drag` at all.
