# Solver validation harness — Design Spec

Date: 2026-08-11

## Purpose

External review asked for the solver's validity to be established **before** more layers go on top
of it — specifically, for an interface in which problems with known answers can be run by hand, and
in which a known change of coefficients produces a known change in result.

The request is reasonable and the project cannot currently meet it. PROJECT.md §4 is a large
evidence ledger, but almost every row in it is a *validation* row: a measured quantity compared
against a published one for a particular aircraft. There is almost no *verification* evidence —
nothing that says the integrator, the trim solve or the linearisation are arithmetically correct
independent of any aircraft's data. The conservation-drift rows are the only entries of that kind,
and drift is a weak proxy: an integrator can conserve angular momentum beautifully and still be
second-order accurate when it claims to be fourth.

The end state: a tested `flightsim/validation.py`, a notebook front end over it, and three families
of check — code verification, coefficient sensitivity against analytic laws, and reproduction of
published worked examples.

Out of scope, by decision: new aircraft, transcription of further CR-2144 flight conditions,
nonlinear aero, and any change to the validated-baseline tolerances of §4.

## Decisions taken during brainstorming

| Question | Decision |
|---|---|
| Depth or breadth | **Depth first.** Exhaust what the three existing aircraft can prove. Jetstar and C-5A are documented as the next step, not taken now. |
| Notebook's relationship to tests | **Thin front end, executed in CI.** All arithmetic lives in a tested module; the notebook imports, calls and plots. `nbval` runs it in the suite so it cannot rot. |
| Framing | **Verification and validation split**, per Roache / AIAA G-077. §4 currently mixes them. |
| Source qualification | **Tiered by source-dependence** (below). Load-bearing claims sit in tiers no document's age can affect. |
| What is *not* claimed | **Fidelity to a real 747.** This work validates the solver, not the airframe data. |

## Source qualification

Review raised whether a 1972 document is a source of error. It is worth answering precisely,
because the answer determines how the evidence is weighted rather than whether the source is used.

The distinction is between two different questions:

1. **"Does the code produce the modes these coefficients imply?"** This needs only a source that
   publishes both its inputs *and* its own computed outputs. If CR-2144's 747 derivatives were
   10% away from the real aeroplane, the solver must **still** reproduce CR-2144's own
   transfer-function factors from CR-2144's own derivatives. The document's age cannot affect this.
2. **"Does the simulation match a real 747?"** This needs modern flight-test data, which the project
   does not hold. It is not claimed anywhere and must not become an implied claim.

Review is asking (1). CR-2144's vintage is therefore not a source of error for anything this spec
asserts. The genuine risks are different, and none of them is the publication date:

| Risk | Status |
|---|---|
| **CR-2144's 747 data is labelled "Flexible"** — the derivatives carry aeroelastic corrections, and `dynamics.py` is a rigid body | **New finding. Not recorded in §5.** Larger in consequence than the vintage; see "Findings to land in PROJECT.md" |
| Secondary compilation read off a poor scan | Known. Session 1 caught a Table IX-4 slip by line-by-line re-read; transcription is the dominant error channel |
| Small-perturbation derivatives valid near one trim point only | Known; §5 already bounds the linear range at \|α\| ≲ 10–12° |
| Internal inconsistency within the source | Known; §3 records Figure IX-1 vs Table IX-3 disagreeing by 6% on approach inertias |

Every check below is therefore tagged with how much it depends on any source at all:

| Tier | Depends on | Examples | Ageing risk |
|---|---|---|---|
| **0** | nothing — pure mathematics | RK4 order of accuracy, Galilean invariance, quaternion norm, torque-free rigid body, Newton convergence | **none** |
| **1** | an analytic relation, not data | `ωn_ph ≈ √2·g/u₀` (contains no aerodynamic coefficient at all), `τ_roll = −1/L_p` | **none** |
| **2** | a source's own arithmetic, closed loop | CR-2144 derivatives → CR-2144 transfer-function factors; Caughey's coefficients → Caughey's A matrix and roots | **none** — immune to whether the data describes a real aircraft |
| **3** | a claim about the physical world | "this behaves like a 747" | high — **not claimed** |

The plan's weight sits in tiers 0–2. That is what makes it robust, not the recency of any document.

### On Caughey as a source

D. A. Caughey, *Introduction to Aircraft Stability and Control*, Cornell MAE 5070 course notes,
Chapter 5, is used for the 747 power-approach worked example. Its status needs stating exactly:

- It is **not an independent data source.** Its Eq. (5.48)–(5.50) cite Heffley & Jewell, NASA
  CR-2144 (its reference [2]) — the same document the project transcribed in session 10.
- It **is an independent implementation.** Caughey formed the dimensional derivatives, the 4×4
  plant matrix, the characteristic polynomial and the roots himself, and published all four.

Same inputs, different code, published intermediates. For a solver check that is precisely the
right kind of independence, and it is worth more here than a second dataset would be.

**Transcription cross-check, already performed.** Every longitudinal coefficient Caughey lists
matches `aircraft.boeing747_approach` exactly: CL 1.11 (his 1.108, recomputed), CD 0.102,
CLα 5.70, CDα 0.66, Cmα −1.26, CLq 5.4, Cmq −20.8, W 564,032 lb, Iy 32.3×10⁶ slug-ft²,
S 5,500 ft², c̄ 27.3 ft. Session 10's transcription is therefore confirmed by a second party.

One discrepancy, recorded rather than smoothed: Caughey states V = 279.1 ft/s (M 0.25 at sea level);
CR-2144's Table IX-2 header states 165 KTAS = 278.49 ft/s. A 0.2% difference. Both are carried in
the reference table and the checks are run at Caughey's speed when comparing against Caughey.

### Open source question, flagged not assumed

Whether Etkin & Reid, *Dynamics of Flight* (3rd ed., 1996) publishes an independent worked example
at the **cruise** condition (M 0.8 / 40,000 ft) has **not been verified**. If it does, it would give
tier-2 coverage at exactly the point where the model's phugoid error is largest (17.8%). To be
checked against the book, not assumed from secondary sources.

## Architecture

```
flightsim/validation.py          # all arithmetic; tested like any other module
flightsim/tests/test_validation.py
notebooks/solver-validation.ipynb  # narrative, plots, no arithmetic
```

`validation.py` exposes three groups:

```python
# (a) code verification
def observed_order(rollout_fn, reference, dt_values) -> tuple[Array, float]  # errors, fitted slope
def newton_residual_history(V, H, ac, iterations) -> Array

# (b) sensitivity
def sweep(ac, field: str, values, quantity) -> Array   # Aircraft._replace under the hood

# (c) references
REFERENCES: dict[str, Reference]   # published value + tolerance + citation string
```

`Reference` carries the citation as a field, so a number cannot enter the module without the table
it came from — the same rule §3 applies to `aircraft.py`.

`observed_order` is run against two different kinds of reference, because they fail differently:

- **Manufactured case.** A linear ODE with a closed-form solution, advanced by the same RK4 stage
  machinery. The reference is exact, so the measured slope is unpolluted; this isolates the stage
  weights themselves.
- **Real 6-DOF case.** The trimmed 747 under fixed controls, against a reference generated at the
  smallest step in the sequence. That reference carries its own error, so the fit uses only steps at
  least 8× the reference step. This catches defects the manufactured case cannot — notably a wind
  sample or control update applied at the wrong stage, which is precisely the seam turbulence work
  will lean on (§2: "wind sampled once per step, held across the four stages").

The notebook holds narrative and figures only. Every number it displays is also asserted by
`test_validation.py`, so the two cannot disagree; and `nbval` executes the notebook in the suite, so
a stale or broken notebook fails the build rather than rotting quietly. This is the failure mode
§10 already records for `summary.py`, whose prose statistics are literals and drifted twice.

New dev dependencies: `jupyter`, `nbval`.

## Family A — code verification (tier 0)

No aircraft data, no literature. A failure here is a bug in the core.

1. **RK4 observed order of accuracy.** Refine `dt` over a decade against a reference solution; fit
   log‖error‖ against log dt; expect slope 4.00 ± 0.05. **This is the largest single gap in the
   project's evidence.** A defect making the integrator second-order — a mis-weighted stage, a
   mis-timed wind sample — passes every existing test at dt = 0.02 and silently degrades every
   result at larger steps. Nothing currently asserts it.
2. **Trim Newton convergence is quadratic.** Residual per iteration from the documented start point.
3. **Galilean invariance.** A uniform wind **W** with the initial ground velocity offset by −**W**
   must reproduce the still-air trajectory. §2 names substituting `vel_rel` into the Coriolis term
   as a classic gust-modelling error; today this is asserted only in the degenerate zero-wind case,
   which cannot distinguish the two.
4. **Torque-free asymmetric rigid body.** No aero, no gravity: angular momentum and rotational
   kinetic energy conserved, and the polhode period checked against the analytic Jacobi-elliptic
   value. Extends §4's drift rows from "does not drift" to "is the right answer".

## Family B — coefficient sensitivity against analytic laws (tier 1)

This is review's request read literally: vary one coefficient, get a known change.

| Sweep | Expected law |
|---|---|
| Cmα → 0⁻ | `ωn_sp ∝ √(−Mα)`; → 0 at the neutral point, divergent real root beyond |
| Cmq | ζ_sp linear in −Mq, ωn_sp approximately unchanged |
| CD0 (hence L/D) | `ζ_ph ≈ 1/(√2·L/D)` |
| airspeed u₀ | `ωn_ph ≈ √2·g/u₀` — **contains no aerodynamic coefficient at all** |
| Clp | `τ_roll = −1/L_p` |
| Clβ, Cnr | spiral root changes sign at `L_β N_r − L_r N_β = 0` |
| Cnβ | `ωn_dr ≈ √N_β` |

The two Lanchester rows are the strongest available evidence in the whole plan, because the
approximation's *error* is itself published. At the 747 approach point:

| Quantity | Lanchester | Model | Ratio | Caughey states |
|---|---|---|---|---|
| ωn_ph | 0.1630 | 0.1334 | 1.22 | "over predicts … by about 20 per cent" |
| ζ_ph | 0.0651 | 0.01329 | 4.9 | "over predicts … by a factor of almost 5" |

Both reproduced from the measured values before this spec was written. So the check is not merely
"does the sweep trend the right way" but "does the disagreement between the exact and approximate
analyses match the published disagreement". That is a far tighter assertion, and it needs no
aircraft data to be true of the real world.

## Family C — published worked examples (tier 2)

1. **Reproduce Caughey Eq. (5.52) element by element** — the 4×4 longitudinal plant matrix for the
   747 power approach — then Eq. (5.53)'s characteristic polynomial and Eq. (5.54)'s roots. This
   exercises units, trim, `dynamics.derivatives` and `jax.jacfwd` in a single comparison against an
   outside implementation.

   **This requires an axis transform, discovered while planning and material enough to record.**
   Caughey states Θ₀ = 0, which is true only in **stability axes**; `tests/modes.py` linearises in
   **body axes**, where θ₀ = α₀ = 5.57° and w₀ = V·sin α₀ ≠ 0. Compared raw, most elements disagree
   — A[1,3] reads −0.952 against his 0.0, which is just −g·sin θ₀. A rotation by α₀ in the (u, w)
   plane is a similarity transform, so it must move every element and leave the eigenvalues alone.
   Measured, it does: eigenvalues identical to 8 decimals, and

   | Element | Body axes | Stability axes | Caughey | rel |
   |---|---|---|---|---|
   | A[0,0] Xu | −0.00883 | **−0.02094** | −0.02120 | 1.2% |
   | A[0,1] Xw | 0.10434 | **0.04632** | 0.04660 | 0.6% |
   | A[0,3] −g cos Θ₀ | −32.022 | **−32.174** | −32.174 | **0.000%** |
   | A[1,3] −g sin Θ₀ | −0.952 | **0.00000** | 0.0 | **exact** |
   | A[1,1] Zw | −0.61575 | −0.60364 | −0.58390 | 3.4% |
   | A[1,2] u₀+Zq | 270.06 | 271.43 | 262.472 | 3.4% |
   | A[2,2] Mq | −0.4381 | −0.4381 | −0.5015 | 12.6% |

   **The two remaining gaps are exactly the omitted α̇ derivatives, and are recoverable to four
   decimals rather than merely "attributable".** Caughey's Z row carries a factor `1/(1 − Zẇ)`,
   with Zẇ = −0.0341 from the CLα̇ = 6.7 this model excludes: 271.43/1.0341 = 262.48 against his
   262.472, and 0.60364/1.0341 = 0.58374 against his 0.58390 — **0.03%**. And our A[2,2] = −0.4381
   *is* his raw Mq from Eq. (5.51); his extra −0.063 is the `(u₀+Zq)·Mẇ/(1−Zẇ)` term from
   Cmα̇ = −3.2.

   So the assertion is not "the matrix roughly agrees" but "every element the model contains matches
   to ≤3.4%, and every element it does not is recovered to <1% by restoring the source's own
   tabulated α̇ derivatives". The eigenvalue invariance under the transform is itself a free tier-0
   check.
2. **Turn §5's attributed gap into a measurement — and correct its wording.** §5 currently says
   "CR-2144 Table IX-4's `Xu, Zu, Mu, Żw, Ṁw` are deliberately excluded". **That is imprecise, and
   the element table above disproves it.** The model *has* Xu and Zu: they fall out of
   dynamic-pressure variation, since lift and drag both go as V², and A[0,0] lands within 1.2% of
   Caughey's Xu without any Xu having been entered anywhere. What is excluded is their **Mach
   content** (CXu, CZu, from CL_M and CD_M) — which is precisely why the gap opens at M 0.8 and
   closes at M 0.25. Mu genuinely is ≈ 0, since Cm = 0 at trim and there is no Cm_M term, matching
   Caughey's A[2,0] = 0.0001 being his Mẇ·Zu rather than an Mu. Only **Zẇ and Mẇ** are truly absent.

   Measured before writing this spec, at **both** 747 points:

   | Mode | Model as shipped | Caughey Eq. (5.54) | Error |
   |---|---|---|---|
   | Phugoid ωn | 0.1334 | 0.1339 | **0.4%** |
   | Phugoid ζ | 0.01289 | 0.01329 | 3.0% |
   | Short-period ωn | 0.8961 | 0.8818 | 1.6% |
   | Short-period ζ | 0.5911 | 0.6255 | 5.5% |

   Trim residual 1.8e-15. Against §4's cruise figures of 17.8% and 11.5% for the same two modes and
   the same omissions, this is a 45× improvement from changing nothing but the flight condition. §5
   attributes the cruise gap to those omissions; this is the first evidence for that attribution
   from outside the project, and it sharpens the claim to a **condition-dependent** one — the
   missing Mach content bites at M 0.8 / 40,000 ft and barely bites at M 0.25 / sea level.
3. **Existing mode tests fold in unchanged** as the third row: `test_cr2144_modes.py`,
   `test_navion.py`.

## Findings to land in PROJECT.md

Three, independent of whether every check passes:

- **§5, new entry: CR-2144's 747 derivatives are the flexible airframe.** The section IX plots are
  labelled "Flexible"; `dynamics.py` integrates a rigid body. An attributed gap, in the same sense
  as the excluded speed derivatives.
- **§4/§5: the phugoid and short-period offsets are condition-dependent**, with the table above.
- **§3: session 10's Table IX-2 transcription is confirmed by a second source** (Caughey / Cornell
  MAE 5070), with the 279.1 vs 278.49 ft/s discrepancy recorded.

## Success criteria

1. RK4 observed order 4.00 ± 0.05.
2. Galilean invariance to 1e-12.
3. Caughey's A matrix, after the documented stability-axis transform, matched to ≤3.4% in every
   element the model contains, and to <1% in the elements carrying Zẇ and Mẇ once the source's own
   CLα̇ = 6.7 and Cmα̇ = −3.2 are restored. Eigenvalues invariant under the transform to 1e-8.
4. Every family-B sweep matches its law within the law's own stated validity, including the two
   Lanchester over-prediction ratios above.
5. Suite green. **No baseline tolerance moved** (§4's rule).
6. Notebook executes clean under `nbval`.
7. PROJECT.md §4 gains a verification subsection and the three findings above.

## Risks

- **A1 may fail.** If RK4 is not fourth-order, that is a real defect in the core the entire
  turbulence programme rests on. Discovering it is the purpose of doing this before adding layers,
  and is exactly what review was asking for.
- **The Lanchester ratios may not reproduce across aircraft.** They are approximations with a
  stated validity range; the Cherokee and Navion may sit outside it. If so, the finding is the
  boundary, recorded — not a loosened tolerance.
- **`nbval` adds suite runtime**, against §8's note that suite runtime is already not reliably
  measurable. The notebook must stay small enough that this does not dominate.

## Explicitly not done

- No new aircraft. **Lockheed Jetstar** (CR-2144 Table VII-1, power approach, non-dimensional,
  *body axis*) and **C-5A** (Table X-1, same form, *stability axis*, so `stability_to_body` applies)
  are the cheapest next additions and are recorded here so the research is not repeated. CR-2144
  documents ten aircraft in total — NT-33A, F-104A, F-4C, X-15, HL-10, Jetstar, CV-880M, B-747,
  C-5A, XB-70A — each with derivatives, transfer-function factors and handling-qualities parameters
  under the same Appendix A/B/C conventions the code already implements.
- **No transcription of the 747's other flight conditions.** CR-2144 report pp. 229–236 are scanned
  line-printer output whose text layer OCRs to noise; the cruise non-dimensional derivatives are
  published as **plots against Mach**, not tables, which is why session 1 had to recover them from
  the dimensional set. Reading further conditions is the same error-prone eye-work as adding a new
  aircraft. The two points already held span M 0.25 / sea level to M 0.8 / 40,000 ft, which is
  enough to establish the condition-dependence claim.
- No change to `aero.py`'s linear form, per §7's stated ceiling.
