# Model sensitivity analysis — design

**Written session 29, before any number was measured.** Scope agreed with the conversation
that commissioned it: **three quantities of interest**, **two factor tiers**, **one tiered
method**, **one budget**. What was deliberately excluded is in §7 and is excluded because it
is already measured elsewhere, not because it is uninteresting.

This document is a design, not a result. **Nothing here may be quoted as evidence.** Results
go to `PROJECT.md` §4 as they are measured, one phase at a time, per `CLAUDE.md` rule 1.

---

## 1. What this is for

The project can already say *which encounter is worse* and *how the response scales*. It
cannot say **which of its own numbers the answer rests on**. Three consequences:

1. **`PROJECT.md` §1's headline is a bare point.** "68% of a recorded peak-to-peak" has never
   been separable into *the model is wrong by 32%* and *the inputs are not known to 32%*.
   `scripts/cat_uncertainty.py` did this for the **wind** inputs. Nothing has done it for the
   **aircraft**.
2. **`ASSUMPTIONS.md` C3 is the register's one openly UNBOUNDED row**, and the summary table's
   action for it is a sentence ("state the excursion with every result") rather than a number.
   Session 27 produced the first quantified point on its Mach axis. One point is not a bound.
3. **Phase 3's remaining acquisition — a DC-10 cruise derivative set — has no price.** Whether
   it is worth a session is a question about how much the answer moves per unit of derivative,
   which is exactly what an elasticity is.

The deliverable is a **ranked elasticity table**, not a prediction. `CLAUDE.md` rule 6 governs
every line of output: bands and orderings, never values.

---

## 2. The three quantities of interest

Each is defined **at the re-trimmed condition**. Session 11 established this and the reason is
not cosmetic: changing a derivative moves the trim point, and comparing a mode or a load across
two different trims confounds the coefficient with the condition. `validation.sweep` already
re-trims per sample and checks both the residual **and** `trim.is_physical`; the AD path must
reproduce that discipline through the implicit function theorem (§4.1), not skip it.

| | QoI | Definition | Base value lives in | Smooth in the inputs? |
|---|---|---|---|---|
| **Q1** | **headline CAT gust load** | peak-to-peak `Δn_z` over the declared window, Mehta five-core Hannibal array, `boeing747` at `CRUISE` | §4 — 67.3% of TM-102186's recorded 2.70 g | **No.** `max − min` is differentiable a.e.; the derivative is that of whichever sample is the argmax, and it jumps when the argmax moves core |
| **Q2** | **linear modes at cruise** | phugoid and short-period (ω_n, ζ); Dutch roll (ω_n, ζ); τ_roll; τ_spiral | §4 "747 modes vs CR-2144" | **Yes**, except where two eigenvalues coalesce. Assert separation before quoting |
| **Q3** | **Dryden ensemble statistics** | `n_z` rms; response spectral peak; exceedance rate at \|Δn\| = 0.2 g | §4 "Response spectra and load exceedance" | **rms yes. Peak location and exceedance rate NO** — both are step functions of the inputs, so their AD gradient is 0 or undefined and only a sweep can price them |

**Q2 is first in the build order and is the falsification step**, because it is the only QoI
with an existing independent answer: session 11's four sweep slopes at the power approach.
If the AD screen cannot reproduce those, the machinery is wrong and nothing downstream of it
may be believed. That check is a result and goes in §4 whether it passes or fails.

---

## 3. The two factor tiers

### Tier A — aerodynamic derivatives (`boeing747`)

Every `Aircraft` field that carries a CR-2144 table, screened as one vector:

- **longitudinal** `CL0 CLa CLq CLde Cm0 Cma Cmq Cmde`
- **lateral–directional** `CYb CYp CYr CYdr Clb Clp Clr Clda Cldr Cnb Cnp Cnr Cnda Cndr`
- **drag** `CD0 e` (and `kappa_airfoil`, `t_over_c`, `sweep` through the Korn rise)
- **mass and geometry**, as a separate block because they are not derivatives and their
  provenance is different: `mass`, `inertia` (including `Ixz`), `S`, `b`, `c`

### Tier B — modelling choices, one row per `ASSUMPTIONS.md` entry

Each is already declared and each has a defined "off" state, so no new modelling choice is
invented to run this. **The point of tier B is that a modelling choice and a derivative are
finally priced on the same axis**, against the same QoI, in the same table.

| Register | Factor | Varied by | What the register says today |
|---|---|---|---|
| **C3** | frozen derivatives across the envelope | Prandtl–Glauert rescale of the longitudinal lift family, as `scripts/les_mach_test.py` did at M 0.406 — here swept across `valid_mach` | **UNBOUNDED**; one point measured, session 27 |
| **E4** | wind held across the four RK4 stages | hold vs per-stage sampling | 0.82% of the headline in-core Δθ |
| **E2 / E10** | point-aircraft gusts vs strip integration | `strip=True` on the flown path | 0.000000 m on the vortex; +22.9% on peak bank |
| **F5** | strip quadrature station count | `N_SPAN` 9 → 57 | returns 82.6% of its own calibration, order 1.50 |
| **F1** | fixed-step RK4 at 50 Hz | dt ∈ {0.02, 0.01, 0.005} | order 1.05 in a spatially varying field |
| **A2** | `g(z)` vs constant `g₀` | the pre-merge constant | +0.383% at cruise; modelled since session 23 |
| **E2** | declared spanwise loading shape | the defensible bracket, and the uniform one separately | 2.6% across defensible shapes; 49.7% including uniform |

### Explicitly out of tier B: wind and scenario inputs

`V₀`, `r₀`, core spacing, `σ_w`, `L_w`, altitude, airspeed. **`scripts/cat_bounds.py` and
`scripts/cat_uncertainty.py` already price these**, and §8/§4 already carry the numbers
(−4.26% for the core-radius lineage; the propagated `V₀`/`r₀` band; the σ_w limbs). The budget
in phase S6 **cites** them; it does not re-measure them. Re-measuring would be the exact
duplication `CLAUDE.md` rule 1 exists to prevent.

---

## 4. Method — three tiers, and what each one cannot see

### 4.1 Tier 1 — the AD screen (a capability this project does not have)

`jax.jacfwd` over the whole `Aircraft` NamedTuple gives **every** local sensitivity in one
pass. The tree already differentiates fields (`wind.gust_rates`), the trim residual
(`trim`), and the plant matrix (`validation.longitudinal_matrix` **is** a `jacfwd` of the real
dynamics) — but it has never differentiated a *result* with respect to a *coefficient*.

Report **elasticity** `E = (∂Q/∂p)·(p/Q)`, dimensionless, so `Cmq` and `mass` are rankable
against each other. Raw gradients are recorded beside it but never ranked.

**Three obstacles stand between `jacfwd` and each QoI. All three have resolutions, and each
resolution carries its own check — this is the technical core of the plan.**

1. **Trim is a Newton solve.** Differentiating through the unrolled loop is both wasteful and
   wrong if the loop has not converged. Use the **implicit function theorem**:

   ```
   dx*/dp = − (∂r/∂x)⁻¹ (∂r/∂p)
   ```

   Both blocks come from `jacfwd(trim.residual)`, which already exists and is already used by
   the solver itself. **Check:** at one base point, compare against differentiating straight
   through the unrolled Newton iteration. They must agree to solver tolerance.

2. **Mode extraction goes through `np.linalg.eigvals` — a NumPy call, so the chain breaks
   there.** `A(p)` is differentiable; its eigenvalues are not, in JAX. Resolution is **exact
   first-order eigenvalue perturbation** rather than a new eig implementation:

   ```
   dλ/dp = (yᴴ (dA/dp) x) / (yᴴ x)
   ```

   with left/right eigenvectors `y`, `x` taken from NumPy at the base point, then
   `ω_n = |λ|` and `ζ = −Re λ / |λ|` by the chain rule. **Check:** central difference on the
   assembled `A`, and — the real gate — session 11's four published slopes.
   **This fails at repeated or coalescing eigenvalues**, so eigenvalue separation is asserted
   before any elasticity is quoted, and reported when it is marginal.

3. **`vortex_viz._measure` returns NumPy.** It ends in
   `np.asarray(jax.vmap(analyse)(...))`, so the entire `Encounter` path is an AD dead end.
   Resolution: a **jnp-only QoI** built on the same `fly_from_state` rollout and the same
   `dynamics.load_factor` call. **This creates exactly the hazard this repo names elsewhere —
   two paths that measure the same thing and can drift apart — so the guard is a test, not a
   comment:** `test_sensitivity.py` asserts the differentiable QoI equals `_measure`'s `n_z`
   channel sample-for-sample at the base point, to 1e-12. If that test is ever deleted or
   loosened, every elasticity in §4 becomes unprovenanced.

**What the AD screen cannot see, stated here so that no result implies otherwise:**

- **Q1's peak** — the gradient belongs to one sample. Valid a.e., discontinuous when the
  argmax moves. Priced by comparing against a ±10% sweep in tier 2, and the size of the
  disagreement is the finding.
- **Q3's spectral peak and exceedance rate** — step functions; AD returns 0 or nothing.
  Sweep only.
- **Interactions.** A one-at-a-time gradient is a diagonal. **Nothing in this plan measures
  interaction terms**, and the write-up must say so rather than let a ranked table imply a
  variance decomposition. Sobol/Morris was considered and declined (§7).

### 4.2 Tier 2 — the OAT confirm

The top-`k` factors from tier 1 go through `validation.sweep` + `validation.affine_fit` — the
session-11 idiom, so the output table has the shape §4 already carries — at **±1%, ±5%, ±10%,
±25%**.

Two things this buys that AD cannot:

- **A nonlinearity measure.** The ratio of the affine slope to the AD elasticity is 1.0 exactly
  when the local gradient is the whole answer. Where it is not, the factor's rank is
  excursion-dependent and the table must say at what excursion it was ranked.
- **The gate.** **The AD elasticity and the ±1% central difference must agree to better than
  1%** for every factor, or the screen is not believed and phases S3 onward do not run. That
  comparison is itself a §4 row.

### 4.3 Tier 3 — banded propagation

A factor gets a band **only** where a held source states one, or where the register states a
measured cost. Everything else is fixed and named DECLARED, per `CLAUDE.md` rule 2. Then, for
Q1, combine:

- **RSS** of the sourced bands — the statement if the errors are independent, which is not
  established and is therefore declared as an assumption at the point of use.
- **Linear sum** — the conservative bracket that needs no independence claim.

Both are reported. **The output is `67.3% [x, y]`, or — the stronger negative result — that
the shortfall survives every band the sources support**, which would move §5's attribution
rather than merely widening it.

---

## 5. Phases and gates

| # | Work | Gate — the phase is not done until this is measured | Lands in |
|---|---|---|---|
| **S0** | `atisim/sensitivity.py`: `elasticity`, `implicit_trim_jacobian`, `eig_perturbation`, `qoi_load`, `qoi_modes`, `qoi_rms` | `qoi_load` matches `_measure`'s `n_z` to **1e-12** sample-for-sample | module + `test_sensitivity.py` |
| **S1** | **AD screen on Q2 (modes)** — the smooth QoI, and the only one with an independent answer | **reproduce session 11's four slopes to <2%** at the power-approach condition | §4 row |
| **S2** | AD screen on Q1 (load), all of tier A | AD vs ±1% central difference **<1%**, or the failure is explained before anything is ranked | §4 row |
| **S3** | OAT confirm on top-`k`, all four excursions, nonlinearity ratios | a ranked table with the excursion it was ranked at | §4 table |
| **S4** | **Tier B on Q1** — each register row gets a number on the same axis | **C3 gets its first bound at cruise**, not just a point at M 0.406 | §4 + `ASSUMPTIONS.md` summary rows |
| **S5** | Q3 ensemble `n_z` rms elasticity; peak/rate by sweep only | the ensemble N is stated with the result, and the E11 drift is reported beside it | §4 row |
| **S6** | Banded budget for Q1; edit §1 and §5 in place | §1's headline carries a band, or states that the shortfall survives every band | §1, §5, §4 |

Phases are independently landable. **Each one that runs writes its §4 row before the next
starts** — the failure mode this project has paid for twice is a session that measures four
things and records none.

---

## 6. Risks, and what would sink each phase

| Risk | Where it bites | What is done about it |
|---|---|---|
| **Eigenvalue coalescence** | S1, S3 at large excursions | assert separation; report the margin; drop the factor from the ranking rather than quoting a wrong number |
| **The argmax moves core** | S2 — Q1's peak-to-peak | measured directly by the ±10% sweep against the AD value; a large gap is a finding about the array, not a bug |
| **Trim fails to converge or returns an unphysical solution** at wide excursions | S3 | `validation.sweep` already checks both residual and `trim.is_physical`; the AD path must apply the same two checks or refuse to report |
| **The two load paths drift apart** | everything downstream of S0 | the 1e-12 equality test; and it is named in `ASSUMPTIONS.md` if it ever has to be loosened |
| **This container is not the documented environment** | every phase | there is no `.venv` here; `jax` was installed fresh on Linux, where §10 documents Python 3.10.11 with a Windows `.venv`. §10's four-row import table does not apply, but the principle does: **every script prints `atisim.__file__` before its first number**, and any §4 row measured here says so |
| **A tolerance moves in the validated baseline** | S4, which changes dt, `N_SPAN` and the loading shape | those are `CLAUDE.md` rule 3 files. Tier B varies these **in the study's own harness**, never by editing shipped defaults |

---

## 7. What this will NOT do, and why

Recorded so a later session does not go looking, per `CLAUDE.md` rule 1 point 3.

- **Not the wind and scenario inputs.** Already priced by `cat_bounds.py` and
  `cat_uncertainty.py`; cited, not re-measured.
- **Not the LES residual ratio** (1.14–1.23 after the PG correction). It is a comparison
  against one external code at one condition, not a property of this model. Session 27's
  named candidate — the `boeing787_yoshimura` entry — is the work that closes it, and it is
  already top of §7's list.
- **Not global variance-based sensitivity (Sobol / Morris).** It would give interaction terms,
  which nothing here measures. It was declined on cost and on fit: §4's rule is to quote a band
  and an ordering, and a variance decomposition invites a reading — *this factor explains X% of
  the variance* — that needs an input distribution this project cannot source. **The
  consequence is a real limitation and is stated with every ranked table: the ranking is
  one-at-a-time and interactions are unmeasured.**
- **Not B1 (rigid vs flexible).** Still unquantifiable. A sensitivity study cannot manufacture
  a flexible airframe, and varying a rigid-body parameter does not bound a structural one.
- **Not a predictive claim.** No output of this study takes the form "the load will be X g".

---

## 8. What lands in the repository

| Path | What |
|---|---|
| `atisim/sensitivity.py` | the module: elasticities, the implicit trim Jacobian, eigenvalue perturbation, the three differentiable QoIs |
| `atisim/tests/test_sensitivity.py` | the 1e-12 agreement test, the implicit-vs-unrolled check, the eig-perturbation-vs-central-difference check, and negative controls |
| `scripts/sensitivity_screen.py` | tier 1, all of tier A, per QoI |
| `scripts/sensitivity_sweep.py` | tier 2, top-`k`, four excursions |
| `scripts/sensitivity_budget.py` | tier 3, the banded budget for Q1 |
| `docs/PROJECT.md` §4 | one row per phase, as it is measured |
| `docs/ASSUMPTIONS.md` | the summary table's C3 row, when S4 gives it a bound |
| `docs/PROJECT.md` §1, §5 | edited **in place** at S6 if the headline gains a band |
