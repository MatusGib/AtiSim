# Session 27 — what was measured, and what it does and does not validate

**`docs/PROJECT.md` remains the standing record.** This file is the merge summary: what
changed, and what it means for the question *"how valid is AtiSim at predicting real
aircraft response in clear-air turbulence?"* Every number here traces to a run in
`presentation_package/_evidence/`. Section references are to `PROJECT.md`.

**Suite at merge: 812 passed, 1 skipped, 0 failed** (`_evidence/pytest_final_session27.log`).
Sanity ladder 11/11, notebook gate 13 passed.

---

## 1. The headline: what got stronger

### A second sealed prediction settled, and it was right

The project's strongest evidence is its **sealed prediction register** — claims hashed and
committed to git *before* the answer was obtainable. Two of three are now settled, **both
right**:

| prediction | claim | outcome |
|---|---|---|
| `the_dryden_response_peaks_at_the_short_period` | load peaks at the airframe's own frequency, 0.131–0.197 Hz | **RIGHT** — 0.1700 Hz at both intensities (re-run at Yoshimura's own 151 seeds) |
| `mil_f_8785c_sigma_w_exceeds_the_mehta_ceiling` | σ_w > 4.46 m/s at 37,000 ft | **RIGHT** — 4.80 ± 0.12 m/s, settled session 27 |

The third still bets **against** the project's own story and stays open. Only `status` and
`outcome` were touched; digests are unchanged.

### The recorded trace was finally read, after four sessions of quoting two numbers from it

TM-102186 Fig. 6 digitised (§4). Three checks including a **negative control** — the same
extraction on the vertical-wind panel returns +1.17 ft/s where smooth cruise demands zero.

It moved two things and neither flatters the model:

- **The model's wind is ~12% weaker than the record.** Recorded vertical gust −98.3 ft/s
  against the fitted field's −86.8. Load follows gust, so this is a real, previously
  unattributed slice of the load shortfall — and it points at **the wind**, which §5 had
  stopped naming as a candidate.
- **The paper's "+1.7 g" is the *second* peak.** The trace reaches +1.855 g. Three
  denominators are now defensible; §4 keeps 2.70 g so the headline does not silently move.

### Yoshimura's own aeroplane, rebuilt from their source code

`aircraft.boeing787_yoshimura()` is constructed from the complete dimensional derivative set
in their `fs.f90`. It **reproduces their own `A_lon` eigenvalues to 0.01%** (short period) and
0.20% (phugoid), trimming at α = +0.0000°, elevator −0.0000°, residual 7.4e-17.

**This is a verification result independent of any turbulence.** AtiSim's nonlinear 6-DOF
linearised by `jax.jacfwd`, against their hand-built 4×4 matrix, by two entirely different
routes, agreeing to a hundredth of a percent. It says the derivative-conversion chain and the
linearisation are right.

---

## 2. What got weaker — and this is the more important half

### The load-shortfall explanation was withdrawn, not confirmed

Session 26 named **aircraft type** (DC-10 vs 747) as the surviving explanation for the
headline 32% load shortfall, flagging its own 1.3× wing-loading ratio as unverified.
Session 27 went and got it. **It cannot be pinned, and not for want of a wing area:**

| DC-10 | at OEW | at mid weight | at MTOW |
|---|---|---|---|
| −10 | 0.584× | 0.815× | 1.046× |
| −30 | 0.631× | 0.973× | **1.315×** |

The project's "roughly 1.3×" is the bottom-right cell — heavier variant, at maximum takeoff
weight, the extreme corner. **At realistic cruise weight the ratio straddles 1.0.** The
variant is unidentified and the encounter weight is recorded nowhere held, so the range is
**0.58–1.32×** and no specification sheet narrows it.

**So the shortfall now has no surviving single explanation.** That is a worse position than
session 26 believed it was in, and a more honest one.

### A claim of my own, cut in half by its own test

§4 argued that the frozen lift-curve slope at M 0.406 *probably was* the LES discrepancy —
Prandtl–Glauert predicts 1.52×, the measurement was 1.42×. Run properly, it closes **47–68%**
of the excess, not all of it. The heading was corrected in place.

The by-product is worth more than the claim: **the fuller compressibility correction performs
worse than the partial one.** Scaling `C_mα` alongside `C_Lα` drops the short period into more
energetic turbulence and adds load back. Correcting physics more completely made agreement
worse — a statement about how gust load is set in this regime that no peak comparison could
have produced.

### "Only accurate near cruise" — true, never fixed, and it was being ignored

`ASSUMPTIONS.md` C3 calls frozen derivatives *"the largest unbounded assumption in the
model"*. Session 23 bounded only the **altitude** axis. The Mach axis was, and largely
remains, unbounded — and the LES runs flew the 747 at **M 0.406 against its M 0.80
linearisation, ΔM = −0.393, thirteen times the largest excursion the assumption had ever been
tested against.** That is now measured rather than assumed: ΔM = −0.393 moves gust-load rms by
×0.80 to ×0.86. **The first quantified point on the Mach axis**, obtained with no chart read.

---

## 3. The LES limb: refused, then earned

Four LES runs existed on disk from an earlier session and had never been recorded. Audited,
their inputs were **not like-for-like** — wrong aeroplane, and out of envelope — so their
numbers were refused. With the aeroplane rebuilt:

| run | rms (h-p) | ratio | short period | response peak |
|---|---|---|---|---|
| baseline `boeing747` | 0.0905 g | **1.427** | 0.1647 Hz | 0.0600 Hz |
| matched `boeing787_yoshimura` | 0.0762 g | **1.202** | 0.1441 Hz | 0.1300 Hz |
| *Yoshimura's own* | *0.06342 g* | *1.000* | *0.1436 Hz* | *0.0800 Hz* |

**Matching the aeroplane closes 53%**, and honestly — the matched entry's derivatives are
tabulated *at* M 0.406, so it has no frozen-slope error to correct. The qualitative change is
larger than the ratio: with the 747 the response peaked at the bottom of the search band; with
the matched aeroplane it peaks at 0.1300 Hz, **9.8% below its own short period**. The airframe
has started selecting its own frequency out of the field.

**A 20% disagreement survives, attributable to neither aeroplane nor Mach** — both eliminated
by construction. Leading candidate, and the next experiment: AtiSim loses **39.8 m/s, 30% of
its airspeed**, over the record flying fixed-control with real drag, where their model holds
altitude to 26–43 m and cannot decelerate, having neither drag nor thrust. Load goes as q̄, so
the two are not yet measuring the same thing.

**What the LES limb IS good for now.** Both confounders are constant multipliers per code, so
they cancel in each code's own resolution ratio. On the two domains that resolve the
turbulence the codes agree on **resolution scaling to 0.5%** (×1.7381 against ×1.7469) while
absolute levels differ by 42%. Three usable statements: the load is **not grid-converged at
35 m** (still growing ×1.74 per halving, both codes agreeing); **below ~70 m the codes stop
agreeing**, so coarse LES cannot drive an aircraft-load calculation; and the **field reader is
validated infrastructure** (+0.978 / −0.968 / −0.935 against their own sampled wind).

---

## 4. So: does this validate AtiSim?

**It validates the engine. It does not validate the predictions.** That distinction is the
whole answer.

**On solid footing, and strengthened this session:**

- **The numerical core.** Conservation and accuracy checked against exact mathematics, not
  documents: RK4 order 3.9998, angular momentum drift 5.7e-13 over 600 s, quaternion norm at
  machine precision.
- **The derivative-to-behaviour chain.** Now verified against a *third* independent
  implementation — Yoshimura's, to **0.01%** — on top of CR-2144's own tables and JSBSim.
- **Comparative and mechanistic turbulence results.** Which encounter is worse, how response
  scales, and why — including TM-102186's counter-intuitive load ordering, six-for-six, which
  the model was never tuned to produce.
- **Predictive discipline.** Two sealed predictions settled, both right, one of them this
  session.

**Provisional:**

- The phugoid is 17.8% off (fully attributed, not fixed).
- The Navion check tests signs and ranges, not values, against sources not held.
- Frequency-domain results are tested against the model's own dynamics; no recorded spectrum
  is held.

**Not validated — say so plainly:**

- **Absolute load prediction.** Structurally blocked, and this session removed its leading
  explanation rather than confirming it.
- **Lateral gust response.** Capability exists; no source held contains a recorded lateral CAT
  response.
- **Behaviour away from the linearisation point.** C3 is unbounded on the Mach axis, and this
  session measured — for the first time — what that costs: up to 20% of gust-load rms at
  ΔM = 0.4.

**The honest one-line verdict:** *AtiSim is a validated comparative and mechanistic tool for
longitudinal gust response, whose numerical core and derivative chain are verified against
three independent implementations, and which cannot yet predict an absolute load — a limit
this session made sharper rather than softer.*

**The biggest remaining risk** is unchanged in shape and better understood in content: every
comparison against a real encounter is of **summary numbers**, never a time history against a
time history. The recorded trace is now digitised, which makes that comparison possible for
the first time. It has not been done.

---

## 5. The process finding, which is why `CLAUDE.md` now exists

Two digitisations and a full four-domain LES run had **already been done** in another
worktree — untracked scripts, gitignored outputs — while that worktree's own `PROJECT.md`
listed them as open work. **The measuring was never the hard part; keeping it was.**
`CLAUDE.md` rule 1 is now *"nothing is done until it is in `docs/PROJECT.md`"*, with the
checklist that enforces it. This session then made the same mistake one layer down — writing
"not run" about runs whose outputs were on disk — which is recorded in §9 rather than quietly
fixed.
