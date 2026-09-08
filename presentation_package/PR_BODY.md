Title: Session 27: read the recorded trace, rebuild Yoshimura's aeroplane, and correct two of our own claims

---

## What this is

A validation pass that was meant to prepare a presentation and turned into six measurements, two corrections to the project's own claims, and one process fix. `docs/SESSION_27_REPORT.md` is the merge summary; `docs/PROJECT.md` remains the standing record.

**Suite: 812 passed, 1 skipped, 0 failed.** Sanity ladder 11/11, notebook gate 13 passed.

## What got stronger

- **A second sealed prediction settled, and it was right.** MIL-F-8785C Fig. 7 digitised: sigma_w = **4.80 +/- 0.12 m/s** at 37,000 ft against a predicted `> 4.46`. The band exists because SEVERE and 10^-5 are one stroke of ink there; the whole band clears. JSBSim's independent transcription lands 0.5% away. Two of three now settled, both right. Only `status` and `outcome` moved -- digests unchanged.
- **The recorded trace was finally read.** TM-102186 Fig. 6 digitised with three checks including a negative control. The project had quoted its two extremes for four sessions without reading the curve between them.
- **Yoshimura's own aeroplane, rebuilt from their source code.** `aircraft.boeing787_yoshimura()` reproduces their own `A_lon` eigenvalues to **0.01%** (short period) and 0.20% (phugoid), trimming at alpha = +0.0000 deg, residual 7.4e-17. AtiSim's nonlinear model linearised by `jax.jacfwd` against their hand-built 4x4, by two entirely different routes.

## What got weaker -- the more important half

- **The load-shortfall explanation is withdrawn, not confirmed.** Session 26 named aircraft type (DC-10 vs 747) as the surviving explanation and flagged its own 1.3x as unverified. Fetched: the ratio spans **0.58-1.32x** and straddles 1.0; the 1.3x is the extreme corner (heavier variant, at MTOW). The variant is unidentified and the encounter weight is recorded nowhere held. **The shortfall now has no surviving single explanation.**
- **The digitised trace says our wind is ~12% light.** Recorded gust -98.3 ft/s against the fitted field's -86.8. Load follows gust, so this points back at *the wind*, which section 5 had stopped naming as a candidate.
- **A claim of mine, cut in half by its own test.** Section 4 argued the frozen lift slope *probably was* the LES discrepancy. Run properly it closes **47-68%**, not all. Heading corrected in place; the script that killed it quotes the old wording.
- **"Only accurate near cruise" -- true, never fixed, and it was being ignored.** The LES runs flew the 747 at M 0.406 against its M 0.80 linearisation, **dM = -0.393, thirteen times** the largest excursion `ASSUMPTIONS.md` C3 had ever been tested against. Now measured: dM = -0.393 moves gust-load rms by x0.80 to x0.86 -- the **first quantified point on the Mach axis**, with no chart read.

## The LES limb: refused, then earned

Four LES runs existed on disk from an earlier session, never recorded. Audited, their inputs were not like-for-like, so their numbers were **refused**. With the aeroplane rebuilt:

| run | rms (h-p) | ratio | short period | response peak |
|---|---|---|---|---|
| baseline `boeing747` | 0.0905 g | **1.427** | 0.1647 Hz | 0.0600 Hz |
| matched `boeing787_yoshimura` | 0.0762 g | **1.202** | 0.1441 Hz | 0.1300 Hz |
| *Yoshimura's own* | *0.06342 g* | *1.000* | *0.1436 Hz* | *0.0800 Hz* |

Matching the aeroplane closes 53%, and honestly -- its derivatives are tabulated *at* M 0.406, so there was no frozen-slope error to correct. A **20% disagreement survives, attributable to neither aeroplane nor Mach**. Leading candidate and next experiment: AtiSim loses **39.8 m/s, 30% of airspeed**, flying fixed-control with real drag, where their model cannot decelerate.

Separately, the confounders cancel in each code's own resolution ratio: on the two resolved domains the codes agree on **resolution scaling to 0.5%** while absolute levels differ by 42%.

## Reviewer notes

- **`boeing787_yoshimura` is deliberately NOT in `REGISTRY`.** It was for one commit and the suite said no in eight places. `REGISTRY`'s contract is "an aircraft this project flies" -- autopilot gains, closed-loop capture, aileron authority, cruise above V_md. Yoshimura's aeroplane has **no control surfaces at all**. Registering it would have meant inventing aileron authority, two sets of gains, and a validity band their source does not state. It is a constructor; callers inject it. What replaces registry membership is a better test -- that it reproduces their eigenvalues.
- **No tolerance was loosened to make anything pass.** One test classification changed (`SOURCE_GATED` -> `RUN_GATED` for the MIL-F prediction) *because the document arrived*, with the reason in a comment at the change.
- **`CLAUDE.md` is new**, and rule 1 is *"nothing is done until it is in `docs/PROJECT.md`"*. Two digitisations and a full LES run had already been done in another worktree -- untracked, outputs gitignored -- while that worktree's `PROJECT.md` listed them as open work. Five stranded scripts are now tracked.
- `checks.recovery_band: passed=False` on the matched run is an **artefact** -- the entry declares no band by design, so the check has nothing to compare against. |alpha| range is 5.74 deg, well inside linear.
- `presentation_package/` carries the evidence base: scorecard, figures (each with its generating script), known issues, and raw logs for every number.

## Verdict

**It validates the engine. It does not validate the predictions.** The numerical core and the derivative chain are now verified against a *third* independent implementation. Absolute load prediction remains structurally blocked -- and this session removed its leading explanation rather than confirming it.
