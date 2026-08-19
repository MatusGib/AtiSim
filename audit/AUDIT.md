# AUDIT — flight dynamics engine

Audit executed against `AUDIT_PROMPT.md`. Companion documents: `INVENTORY.md`
(Phase 1), `NOTATION.md` (Phase 1B), `ASSUMPTIONS_AUDIT.md` (Phase 4). Raw
evidence from the Phase 2 agents is in `audit_evidence/`.

**Status vocabulary:** `verified` · `failed` · `unverified` · `circular` ·
`unverifiable — source not available` · `known flaw, bounded` ·
`known flaw, unbounded`.

---

## 0. Bibliographic record

**Theoretical backbone.** Stengel, Robert F. *Flight Dynamics*, Second Edition.
Princeton, New Jersey: Princeton University Press, 2022. ISBN 9780691220253
(hardback); ISBN 9780691237046 (ebook). LCCN 2021052561 (print), 2021052562
(ebook). Copyright © 2004, 2022. Library of Congress classification TL570 .S73
2022; DDC 629.132/3. Cover image: Overture airliner, courtesy Boom Supersonic.
Composed in Sabon and Univers. 914 PDF pages. In repo at
`./Flight_Dynamics_-_Second_Edition.pdf`. **All of the above was read from the
book's own title page (PDF p. 5) and copyright page (PDF p. 6), not assumed.**
Cited below as `[FD2e §x.y, eq. n]`.

**Primary numerical dataset.** Heffley, Robert K., and Wayne F. Jewell.
*Aircraft Handling Qualities Data*. NASA CR-2144, December 1972. Systems
Technology, Inc., Hawthorne, California. Technical Report 1004-1; Contract
NAS 4-1729. 352 PDF pages; Section IX is the B-747.

> **This document was NOT in the repository when the audit began.** It is named
> by `AUDIT_PROMPT.md` as the primary numerical dataset, and by `PROJECT.md`,
> `provenance.py` and `aircraft.py` as the source of essentially every 747
> number — but no copy existed anywhere in the working tree. Under the prompt's
> own rule ("If a source is not in your context, you may not cite it") every
> 747 constant would have been `unverifiable — source not available`.
>
> It was fetched during this audit from NASA NTRS,
> `https://ntrs.nasa.gov/api/citations/19730003312/downloads/19730003312.pdf`,
> and saved to **`./refs/NASA-CR-2144.pdf`**, so every citation below is
> reproducible. Its scanned text layer is OCR noise; every value cited was read
> from a **rendered image** of the page at 200–600 dpi.
>
> **Page offset: PDF page = printed page + 6** in Section IX.

**Validity and provenance of the two sources.** FD2e is a textbook of record
(Princeton UP, second edition, peer-reviewed publisher, extensive reference
lists per chapter). CR-2144 is a **compilation**, not a primary measurement: its
own abstract says "Available information on weight and inertia, aerodynamic
derivatives, control characteristics… is documented for 10 representative
contemporary airplanes. Data sources are given for each airplane." Its 747
section cites Boeing simulation reports (printed p. 242, "B-747 DATA SOURCES",
including Rodney & Nordwall, *The Simulation of a Large Jet Transport Aircraft*,
Boeing Rept.). **So CR-2144 is one step downstream of Boeing's own simulation
data, and the derivative plots in Section IX are labelled "Flexible".** Nothing
in this repository is an independent measurement of a real 747.

**No other source was available.** Parks et al. 1985, Wingrove & Bach 1994,
Doyle et al. 2011, Proctor et al. 2000, Oseguera & Bowles 1988, Caughey's Cornell
notes, McCormick, Roskam/PyFME and the Navion references are all cited by the
code and none was in the repository. See §6 and `audit_evidence/B-*.md` for what
could and could not be retrieved.

---

## 1. Headline findings

> **Statuses below were updated by the remediation pass** (commit `4e925c4`, branch
> `claude/flight-sim-prompt-fix-7ce581`), per
> `FIX_PROMPT.md`. Nothing was deleted: a finding that was repaired reads **REPAIRED** and
> names what now pins it, and a finding that was deliberately *not* repaired says so and
> names the register entry that now declares it. The distinction matters more than the
> count. **Ten findings were repaired.** Eight more were real, measured, and left alone
> because the correct form is not established by any source this project holds — changing
> those would have traded a documented limitation for an invented certainty, which is
> strictly worse than the flaw. One repair was made, measured, and **reverted** (finding
> 11's sibling, `_B747_G`): it broke two bit-exact arithmetic-neutrality guards, and
> re-pinning those would have cost more than the 1.5e-6 rule violation it fixed. See
> `docs/ASSUMPTIONS.md` B5.

| # | Finding | Status |
|---|---|---|
| **1** | **Every one of the 29 values in CR-2144 Table IX-2 matches `aircraft._boeing_747_approach` exactly**, including the δa footnote. The approach set is a faithful transcription. | `verified` |
| **2** | **Tables IX-3, IX-4 and IX-8 likewise match** for both flight conditions used, including `Mq = −0.339` (the session-5 transcription fix is correct) and the primed lateral set. | `verified` |
| **3** | **`aircraft._unprime` reproduces CR-2144 Appendix A (printed A-18) exactly.** The Ixz de-priming is right. | `verified` |
| **4** | **The dimensional→non-dimensional conversion chain closes against the source's own two tables** — a check CR-2144 makes possible and the project never ran. Worst error 2.2%. | `verified` |
| **5** | **But the code's conversion relations are the small-angle forms, not CR-2144 Appendix A's actual definitions.** Appendix A's `X_w`/`Z_w`/`M_w` each carry a `−2(W₀/U₀)(C + (M/2)C_M)` term the code drops, and use `U₀ = V_To cos α₀` where the code uses `V_To`. Cost at cruise: `CLa` +1.15%, `CDa` +1.13%, `Cma` +0.32%. | **`known flaw, bounded`** |
| **6** | **ALL FOUR published longitudinal mode factors are ATTRIBUTED.** Restoring the two omitted derivative families to the engine's own plant matrix takes every one to within **1%** — from −17.8%, +14.4%, −1.4%, −11.5%. The whole cruise mode discrepancy is the two omissions and nothing else; the aerodynamic data, conversion chain, trim solve and eigen-extraction are exonerated. | `verified` (demonstrated) |
| **7** | **The two causes are orthogonal, and phugoid ζ needs both.** Speed derivatives `Xu, Zu, Mu` → phugoid frequency alone (−17.8% → +0.3%, short-period damping unmoved). α̇ derivatives `Zẇ, Mẇ` → short-period damping alone (−11.5% → +0.6%, phugoid frequency unmoved). Phugoid ζ: +14.4% shipped, +3.7% with speed alone, +13.0% with α̇ alone, **+0.5% with both**. | `verified` (demonstrated) |
| **8** | **`PROJECT.md` §4 compares only 2 of the 4 published longitudinal factors.** Phugoid ζ (+14.4%) and short-period ωn (−1.4%) are in Table IX-5 FC9 and appear nowhere in the project's record. | **REPAIRED** — all four rows now in `PROJECT.md` §4, with the p.231 values re-read at 600 dpi |
| **9** | **`ASSUMPTIONS.md` B4 declares the accelerometer-offset error unboundable. CR-2144 Table IX-3 supplies the arm** (`LXP = 86.0 ft`, `LZP = −10.0 ft`, every flight condition) and Table IX-5 publishes `N(AZP/DE)`, normal acceleration *at the pilot station*. **Now bounded.** | **previously unbounded → `known flaw, bounded`** |
| **10** | **The largest sensor-offset correction falls on the project's headline result, not on the case the register worries about.** Parks vortex first core: **26.4%** of its own load excursion. Manoeuvre: 8.3%. Updraft: 11.3%. | **`known flaw, bounded`** |
| **11** | **`PROJECT.md` §2's claim that "a constant with no ledger entry fails the build" is false.** No test walks the source modules; all eight tests in `test_provenance.py` iterate `LEDGER` itself. Independently found by this auditor and by Agent G. | **REPAIRED (documentation)** — both documents now describe what is actually enforced; ledger coverage extended by 7 entries |
| **12** | **CR-2144 contradicts itself on `C_Lα̇`.** Table IX-2 prints `−6.7`; Table IX-4 FC2 prints `ZWD = +0.0338`; Appendix A applied to the former gives `−0.0341`. The `C_mα̇` route is consistent to 0.4%; only the Z row conflicts. | **source conflict, recorded** |
| **13** | **`wind.py`'s citation of FD2e on strip theory overstates the source.** FD2e (printed p. 218) endorses rotary derivatives for *wind rotors* and demands strip theory only for *wake vortices*. A Parks vortex is a wind rotor. | **`failed` — citation does not support the claim** |
| **14** | **FD2e eq. 3.4-49 has a sign error and eq. 3.4-55 is wrong by a factor of −2.** The project's independent claim to this effect is **confirmed** — re-derived here from rigid-rotation kinematics rather than transcribed. The code does not use either equation. | `verified` |
| **15** | **The project's integrity rule held.** No tolerance was ever loosened and no reference value ever edited, across 591 deleted test lines in all 53 commits touching `flightsim/tests/` on all 9 branches. | `verified` |
| **16** | **LIVE DEFECT 1 of 4. The strip integral returns 82.6% of `Clp` at the station count production uses.** `loads.strip_model` builds stations at `N_SPAN = 9`; the calibration identity `a₀ = −8·Clp` is exact only in the continuum limit. Every test of it overrides the count (201, 2001, 21); none exercises 9. | **`failed`, DECISION RECORDED** — `N_SPAN` and `a₀` both left as they are, reasoning and convergence order in `ASSUMPTIONS.md` F5 |
| **17** | **`aircraft.py:217` misquotes its own cross-check, in the direction that flatters the source rather than the model.** It cites Table IX-1's `CLδe` as **0.396**; the table (printed p.216) reads **.356** — read at 500 dpi by Agent B and again independently here. The recovered 0.364 therefore agrees with the table to **2.2%**, not the 8.1% the comment implies. The other three cross-check numbers (`Cmδe` −1.40, `Clb` −0.281, `Cnb` +0.184) are correct. | **REPAIRED** — comment corrected against printed p.216, re-read at 500 dpi in this pass |
| **18** | **The engine carries an `X_q` CR-2144 does not model** — drag responds to `CLq·q̂`. Stability-axis `A[0,2] = −0.8807 s⁻¹` against the parabolic polar's own prediction of −0.8887, agreeing to 0.9%. | `verified`, undeclared |
| **19** | **Four more sources retrieved and verified**: Oseguera & Bowles TM-100632, Proctor et al. 2000, Doyle et al. 2011, Caughey's Cornell notes. All match the code's quotations. **Parks 1985 and Wingrove & Bach 1994 could not be obtained** — so the vortex parameters, the vortex field equations, and the updraft magnitudes remain content-unverified. | mixed |
| **20** | **`aircraft_data_validated.py`, cited as the source of the Cherokee and Cessna data, does not exist in the repository.** 168 of 346 constants terminate in a file that is not there. | `unverifiable — source not available`, **NOW STATED AS SUCH IN THE CODE** |
| **21** | **LIVE DEFECT 2 of 4. `validation.lateral_modes` returns roll and spiral swapped when the spiral is unstable.** `reals.sort()` orders *signed* time constants, so the Cherokee's unstable spiral (τ = −51.6 s) is returned as `roll_tau` and its true roll subsidence (0.360 s) as `spiral_tau`. No shipped test covers an aircraft with an unstable spiral. | **REPAIRED** — `reals.sort(key=abs)`; the three stable-spiral aircraft are bit-identical |
| **22** | **`rk4_step` is classical RK4 — proved, not read.** Its fitted stability polynomial matches `1+z+z²/2+z³/6+z⁴/24` to 4.34e-13, and its quadrature of `tᵏ` is exact through k=3 with exactly Simpson's `+1/120` error at k=4, which admits no other tableau. | `verified` |
| **23** | **Trim is not unique on unmodified registry data.** 5–13% of a 2,090-point guess grid lands on a converged-but-absurd root at the aircraft's own cruise condition (747: α = −118°, throttle −194, residual 3.2e-14). `PROJECT.md` §5 documents this only for *degenerate coefficients*. `is_physical` catches them via α; elevator and throttle are unbounded and unchecked. | widens a known flaw; **the `is_physical` half REPAIRED** — `trim`'s own silence is unchanged and still pinned |
| **25** | **The four wind fields' analytic gradients are correct to machine precision.** `gust_rates` checked against `jacrev` (max 2.5e-15), central differences at five step sizes, **hand-derived closed-form Jacobians for all four fields** (max 7.8e-15), and a rigid-rotation identity, over 400 randomised position+quaternion pairs each. All four fields divergence-free to ~1e-15; Rankine structure exact (`−2V₀/r₀` inside, <3e-17 outside); microburst continuity 6.6e-15. | `verified` |
| **26** | **The microburst is an exact transcription of its primary source** (`refs/NASA-TM-100632`, verified over 12,000 points: `w` to 4e-15, `u` to 1.6e-11). But **the paper's fourth parameter `z_h`, the depth of the outflow, is not modelled**, so the field has no ceiling and grows to a 50.6 m/s on-axis asymptote. The project's 300 m AGL penetration sits essentially at the paper's own `z_h` (294.4 m). | `verified` transcription; omission **NOW DECLARED** — `ASSUMPTIONS.md` E6 |
| **27** | **`ASSUMPTIONS.md` E4's bound on the wind hold measures the wrong quantity and understates it ~80×.** E4 bounds it by an h-vs-h/2 *discretisation* refinement (0.0024 m/s, "~1e-4 relative", "no result the project quotes is affected"). The *scheme* error is hold-vs-per-stage at the same dt: the headline in-core Δθ moves **1.62% at dt = 0.02 and 0.82% at the published dt = 0.01**. No conclusion changes — vortex claims are orderings — but **the quoted 2.240° is not good to four figures**. | **REPAIRED (documentation)** — E4 re-bounded with the right instrument; a third dt added |
| **28** | **`airframe.stations`' longitudinal set is entirely aft of the CG** (centroid −16.75 m), so `sampled_rates`' pitch channel is a *backward* secant with an `O(arm/2·f'')` bias. Consequence: E2's "the correction is exactly zero inside the core" holds on the **downstream** half only — at −0.99 r₀ it is **1.80 V₀/r₀**. The project's test samples positive fractions only. | **`known flaw`, NOW DECLARED** — `ASSUMPTIONS.md` E8 |
| **29** | **Superposing any field onto the microburst breaks its ground boundary condition** — 3 m/s through the ground everywhere. The fields sum exactly, but the microburst's defining property (both components vanish at z = 0, which its paper singles out as what earlier models got wrong) does not survive superposition. | **NOW DECLARED** — `ASSUMPTIONS.md` E5 |
| **30** | **`along_track_shear`'s constant-heading approximation omits `−u_⊥·ψ̇`, which reaches ΔF = 0.140 at a standard-rate turn** — the entire FAA hazard threshold. **It is exactly zero in all four of the project's runs**, because every field has zero east wind on the north axis so `ψ̇ ≡ 0`. A bounded trap, not a live error. | **REPAIRED** — term derived, reduction to Eq. (4) asserted; ΔF reproduced at 0.1423 |
| **31** | **At the vortex core boundary, which branch is taken is decided by round-off in the position.** Approaching vertically at 12 km altitude, 8e-13 m of cancellation flips the sample to the inside branch and reverses the sign of a 0.283 s⁻¹ gust rate. | **`known flaw`, NOW DECLARED** — `ASSUMPTIONS.md` E9 |
| **32** | **Things that could not be broken, and were tried hard.** No NaN or Inf at any admissible parameter — vortex core centre, updraft axis down to sharpness 0.1, microburst axis, the ground plane, and 1,000 km below it — checked with value, `jacfwd`, `jacrev`, Hessian, `gust_rates` and `along_track_shear`. The `jnp.where` reverse-mode transpose trap **exists only at the degenerate `r0 = 0`**. Superposition is exact (`wind_ned` bit-identical, `omega_gust` 6.5e-16). | `verified` |
| **24** | **`jit` is not bit-identical to eager** — last-ulp only (1.6e-17 on trim, 4.5e-13 m on a 1000-step rollout). The extracted modes come out bit-identical anyway. `rollout`, `logged_rollout`, a Python loop of `step`, and `vmap` member 0 are all bit-exact with each other. | `verified` |
| **33** | **LIVE DEFECT 3 of 4. Mechanical energy is created in still air with the throttle shut.** The Cherokee delivers `P_aero = +56.9 kW` to the airframe at V 75 m/s, q −8.256 rad/s, elevator at its 25° limit — in motionless air, where the only energy source is the aircraft itself. Reachable from the aircraft's own trim: a fixed-control rollout manufactures **520.6 J**. | **`failed`**, bounded, **NOW DECLARED** — `ASSUMPTIONS.md` C9; deliberately not repaired |
| **34** | **Finding 33's cause is demonstrated, and it is not the one Phase 2E proposed.** The elevator's moment power is cancelled **exactly** by the lift-tilt drag of its own lift increment placed at the arm its own two derivatives imply, `l_δe = −Cmδe/CLδe·c`. Identity holds to **1.9e-16**, and to ≤5.6e-16 over 500 random states on all four aircraft. Restoring that one term takes the Cherokee from **1190 violating grid points to 0**. The missing physics therefore needs **no new constant**. | `verified` (demonstrated); **NOT IMPLEMENTED, deliberately** — see `ASSUMPTIONS.md` C9 |
| **35** | **The rate pair and the control pair imply two different tail arms, and the disagreement tracks source availability.** `−Cmq/CLq` against `−Cmδe/CLδe`: 747 cruise **4.024 vs 3.969 (1.4%)**, 747 approach **3.852 vs 3.965 (2.9%)** — both from CR-2144. Cherokee **2.00× apart**, Cessna **2.88× apart** — both from the file that does not exist (finding 20). An internal consistency check on precisely the data that could not be sourced. | `verified`, new; **NOW DECLARED** — `ASSUMPTIONS.md` C11 |
| **36** | **The RK4 order collapse is caused by the spatial gradient, not by wind.** Still air **3.9873**; a constant uniform 13 m/s field **3.9875**; lee wave **0.9920**; updraft column **0.9994**. A field with no gradient keeps fourth order, which isolates the variable and sharpens `ASSUMPTIONS.md` E4 from "in a wind field" to "in a field with a gradient". | `verified` |
| **37** | **`is_physical` passes trims that no aircraft could fly**, on a plain (V, h) sweep rather than the guess-grid of finding 23: **319/640** (747), 22/640 (747-approach), 284/640 (Cherokee), **364/640** (Cessna) pass while demanding throttle outside [0,1] or elevator past its limit. The 747 at V 471.8 m/s passes at α −0.57° on **throttle 567**. | **REPAIRED** — `is_physical(x, ac)` checks throttle ∈ [0,1] and \|elevator\| ≤ limit |
| **38** | **LIVE DEFECT 4 of 4. A control channel with zero authority makes the Jacobian singular, and `jnp.linalg.solve` returns NaN silently.** The Cessna's `CYdr = Cldr = Cndr = 0` (a declared choice) zeroes the rudder column: `det = 0`, `solve(J, [1,1,1]) = [nan, nan, inf]`, no exception raised. Any solver carrying rudder as an unknown for this aircraft fails silently. **Mitigated under test only:** `conftest.py` sets `jax_debug_nans`, so the suite would catch it — but nothing sets it for `scripts/` or for a library caller, and the audit's own test has to switch it off to observe the shipped behaviour. | **`failed`**, caught at test time only; **NOW DECLARED** — `ASSUMPTIONS.md` F7 |
| **39** | **The engine has no ground plane, and `ASSUMPTIONS.md` never declares it.** A 747-approach released at 300 m crosses `h = 0` at **t = 8.65 s** at a 25.1 m/s sink rate and 106.6 m/s, and the run continues to **−698 m**, every sample finite. The atmosphere extrapolates below sea level without limit (30.47 kg/m³ at −50 km). **But `scripts/microburst.py` already truncates at one wingspan with a documented rationale** — so the gap is that the guard lives at the call site, not in the engine. | **NOW DECLARED** — `ASSUMPTIONS.md` A4 and D2 |
| **40** | **Aerodynamic force is nonzero at exactly zero airspeed**, via the `V_MIN = 1.0` floor: 19.86 N (747), 170.73 N (747-approach), 4.30 N (Cherokee), 1.44 N (Cessna), each at its own cruise altitude. So free fall does not read zero: `n_z` = +7.0e-6 to **+4.0e-4** instead of 0. | **REPAIRED** — `qbar` built from the true airspeed; force at V = 0 is exactly zero |
| **41** | **What could not be broken, second pass.** Total energy never once exceeded its initial value: `max(E − E₀) = +0` **exactly** in every still-air run, including five deliberate energy-pumping feedback laws and full-deflection reversals at up to 1000 Hz. Aerodynamic power is strictly negative at **0 of 80,000** randomised in-envelope states. 1,000,000 steps at trim moves the state by **0 m**. Quaternion norm holds to one ulp under deliberate denormalisation to scale 10. | `verified` |

---

## 2. Main table — claims, sources, status

### 2.1 Aircraft data and the conversion chain

| Claim | Code location | Source | Status | Notes |
|---|---|---|---|---|
| 747 reference geometry S 5500 ft², b 195.68 ft, c̄ 27.31 ft | `aircraft.py:227,385` | CR-2144 Table IX-3 header, printed p.229 | `verified` | read from the rendered page |
| Cruise mass/inertia W 636,636 lb; Ix/Iy/Iz/Ixz = 1.82e7/3.31e7/4.97e7/970,056 | `aircraft.py:229` | Table IX-3, FC9 | `verified` | exact |
| Approach mass/inertia W 564,032 lb; 1.42e7/3.23e7/4.54e7/870,050 | `aircraft.py:398-399` | Table IX-3, FC2 | `verified` | exact, incl. the 870,050 the project rounds to 0.870e6 in prose |
| Cruise `Xw, Zw, Zq, Mw, Mq, Zde, Mde` = 0.0389, −0.317, −5.16, −0.00105, −0.339, −17.9, −1.16 | `aircraft.py:243` | Table IX-4, FC9, printed p.230 | `verified` | all seven; **`Mq = −0.339` confirmed** |
| Table IX-4 states "(BODY AXIS SYSTEM)" | comment `aircraft.py:216` | Table IX-4 header | `verified` | printed in the header |
| Cruise lateral primed set (10 values) | `aircraft.py:248-255` | Table IX-8, FC9, printed p.234 | `verified` | all ten; header states "(BODY AXIS SYSTEM)" |
| `Yv = Yb/U0` and `Y*δr = Yδr/V_To` | comment `aircraft.py:294` | Table IX-8 `YV`/`YB` rows; Appendix A A-17 `Y_β = V_To Y_v` | `verified` | −43.2/774 = −0.05581 vs tabulated −0.0558 |
| `_unprime` inverts CR-2144's priming | `aircraft._unprime` | Appendix A, printed A-18 | `verified` | the code's docstring relation is A-18 verbatim |
| Approach set, all 29 non-dimensional values | `aircraft.py:406-420` | Table IX-2, printed p.217 | `verified` | exact, including the δa footnote |
| `CD_trim = 0.043` for cruise | `aircraft.py:238` | Figure IX-6, 40,000 ft curve, **chart read** | `unverified` (reading), ±0.003 declared | the project's own digitisation of the same figure reads 0.0439 at M 0.80 |
| `CD0` and `e` back-solved from one point | `aircraft.py:290-291` | — | **`circular`**, and declared so | `test_drag_polar.py` breaks it with 9 more points **off the same figure** — same source, not independent |
| Conversion relations for `Cmq`, `Cmδe`, `CLδe`, and every lateral derivative | `aircraft.py`, `from_dimensional_*` | Appendix A A-17/A-18 | `verified` **exact** | these carry no `W₀` term in Appendix A |
| Conversion relations for `CLa`, `Cma`, `CDa` | `aircraft.py:263,266,281` | Appendix A A-16/A-17 | **`known flaw, bounded`** | omits `−2(W₀/U₀)(C + (M/2)C_M)`; uses `V_To` for `U₀`; inverts on a `C_L/C_D` basis where Appendix A uses `C_N/C_X`. Cruise cost: `CLa` +1.15%, `CDa` +1.13%, `Cma` +0.32% |
| `Z_q`'s definition | `aircraft.py:264` | **Appendix A §5 defines no `Z_q`** | `verified` to 1.2% *by cross-check* | the assumed `M_q`-analogue reproduces Table IX-2's `CLq = 5.4` as 5.335 from Table IX-4 FC2's `ZQ = −7.58` |
| Whole chain closure, FC2 | — | IX-4 → IX-2, both printed | `verified` | `CLa` +1.57%, `CDa` +2.16%, `Cma` +0.68%, `CLq` −1.21%, `Cmq` +0.24%, `CLde` −0.48%, `Cmde` +0.09% |
| Sweep 37.5°, t/c 0.09, κ 0.87 | `aircraft.py:282-284` | none — declared | `unverified` | not in the provenance ledger |
| `max_thrust` 4 × 43,500 lbf, lapse 0.8 | `aircraft.py:344-348` | none — declared, and the code says so | `unverified` | not in the ledger |
| Deflection limits 25/20/25° | `aircraft.py:349-351` | none — declared | `unverified` | not in the ledger |
| `CYp = CYr = 0` for all four aircraft | four sites | CR-2144 tabulates neither | `verified` (as a faithful omission) | the code's comment is correct; understates Dutch-roll damping |
| Cherokee, Cessna, Navion data | `aircraft.py` | McCormick / Roskam-PyFME / Nelson-Etkin-McRuer | `unverifiable — source not available` | see `audit_evidence/B-*.md` |
| The rate pair and the control pair describe the same tail | `airframe.effective_tail_arm` vs `−Cmδe/CLδe` | CR-2144 IX-2/IX-4 for the 747s; nothing held for the light pair | `verified` for the 747s, **`failed`** for the light pair | 1.4% and 2.9% on the two CR-2144 sets; **2.00× and 2.88×** on the two whose source file does not exist. Finding 35 |
| `tail_arm_is_plausible` gates the strip model on the derived arm | `airframe.py:55` | declared band (2, 6) chords | `verified` — the gate works | it rejects the Cherokee (1.28) and Cessna (0.86). It reads only the **rate** estimate, though both are available |

### 2.2 Equations of motion, integrator, and the wind seam

| Claim | Code location | Source | Status | Notes |
|---|---|---|---|---|
| `v̇ = F/m + g_b − ω×v` with **inertial** v | `dynamics.py:72` | `[FD2e §3.2]` | `verified` | and it produces `∂u̇/∂q = −w₀ = −62.6 ft/s` at cruise, an element CR-2144's own linear model does not carry |
| `ω̇ = I⁻¹(M − ω×Iω)` | `dynamics.py:73` | `[FD2e §3.2]` | `verified` | full Euler, unlinearised |
| Gust rates `p_g=+∂w/∂y`, `q_g=−∂w/∂x`, `r_g=+∂v/∂x` | `wind.gust_rates` | `[FD2e eqs. 3.4-48, 3.4-50, 3.4-52]`, printed p.216 | `verified` | re-derived here from `v = ω×r`, not transcribed |
| FD2e eq. 3.4-49 (`∂v/∂z = +p`) is wrong | project claim | `[FD2e eq. 3.4-49]` | `verified` — **the book is wrong** | rigid rotation gives `∂v/∂z = −p`; inconsistent with the book's own 3.4-48 |
| FD2e eq. 3.4-55 is wrong by −2 | project claim | `[FD2e eq. 3.4-55]`, printed p.217 | `verified` — **the book is wrong** | book gives `+2M_q q_a`; correct is `−M_q q_a` |
| Effective tail arm `= −Cmq/CLq` | `airframe.effective_tail_arm` | `[FD2e eqs. 3.4-7, 3.4-10, 3.4-12]` | `verified` (algebra) | the tail lift slope does cancel; both derivatives attributed to the tail, as the docstring states |
| The 83%-of-CLq rejection note | `airframe.py:47` | `[FD2e eqs. 3.4-13, 3.4-14]` | `verified` | with `h_cm = 0.25`, eq. 3.4-13 gives `(CL_q̂)_wing = +CLα`, ≈83% of the total. The docstring's reasoning checks out |
| Strip roll integral vs `[FD2e eq. 3.4-40]` for a rectangular wing | `wind._strip_rolling_coefficient` | eq. 3.4-40, printed p.213 | `verified` (algebra) | λ=1 gives `−CLα/6`; the uniform-chord strip integral gives `−a₀/6` |
| `a₀ = −8·Clp` from elliptic loading | `airframe.calibrated_lift_slope` | derived | **`circular`** by construction | the code says so; `PROJECT.md` §4 lists the resulting check in a measurement table without the label |
| `Clp` from a rigid roll rate reproduces the tabulated `Clp` | test | — | **`circular`** | `a₀` was calibrated to make it so |
| Wind held across all four RK4 stages | `integrate.step:108` | — | **`known flaw, bounded`** — and documented | scheme is O(h), not O(h⁴), in a spatially varying field |
| "the standard treatment for Dryden and von Kármán" | `integrate.py` docstring | none cited | `unverified` | a plausible claim with no citation anywhere in the repo |
| Microburst constants 1.1212, 0.22, 12.5, 0.2357 | `wind.py:408-411` | Oseguera & Bowles 1988 | `verified` **by re-derivation** | 1.1212 solves `e^{−x²}(2x²+1)=1`; 0.22 = ln(12.5)/11.5 = 0.21963; 0.2357 recomputed as 0.23570 from the other three |
| Microburst satisfies continuity exactly | `wind.microburst_wind` | Oseguera & Bowles eqs. (5)–(6) | `verified` (analytically) | `(1/r)∂(ru)/∂r + ∂w/∂z = 0` identically |
| Parks vortex is divergence-free and irrotational outside the core, solid-body inside | `wind.vortex_wind` | Parks et al. 1985 | `verified` (analytically) | both branches checked by hand; continuous at `r₀` |
| Parks `r₀`, `V₀`, spacing values | `wind.PARKS_CASES` | Parks et al. 1985 pp.127-128 | `unverifiable — source not available` | paywalled, not retrieved |
| Lee wave / updraft amplitudes and durations | `wind.py` | Doyle 2011 / Wingrove & Bach 1994 | `unverifiable — source not available` | |
| F-factor `= U̇ₓ/g − w/Vₐ`, 1 km average, 0.1/0.13 thresholds | `dynamics.py:158-211` | Proctor et al. 2000 | `unverifiable — source not available` | |
| Lift is perpendicular to the relative wind at non-zero β | `aero.py:157-163` | — | `verified` (algebra) | `[sinα,0,−cosα]·[cosαcosβ, sinβ, sinαcosβ] = 0` exactly |
| Korn/Lock `M_crit = M_dd − (0.1/80)^⅓` | `aero._MDD_OFFSET` | — | `verified` (algebra) | `dCD/dM = 80(M−M_crit)³ = 0.1` ⇒ offset 0.10772 |
| ISA constants, two layers | `atmosphere.py` | ICAO Doc 7488 (cited, not held) | `unverified` | values are conventional but no held source |
| float64 set before first array | `flightsim/__init__.py` | — | `verified` | `jax.config.update` at import, before any `jnp` use |
| Aerodynamic power is dissipative in still air | `aero.aero_forces_moments` | energy conservation | **`failed`**, bounded | +56.9 kW on the Cherokee; attributed to the omitted lift-tilt term in §2.5; 0/80,000 in-envelope states affected |
| `V_MIN` "never binds in normal operation" | comment `aero.py:22-24` | none | **`known flaw, bounded`** | strictly true of *normal operation*, but the floor makes aerodynamic force nonzero at exactly V = 0, so free fall reads `n_z` up to +4.0e-4. Finding 40 |
| The integrator has a ground | `integrate.step` | — | **`undocumented`** | there is none; a run continues to −698 m, all finite. `scripts/microburst.py` guards its own case at one wingspan. Finding 39 |
| A converged trim is a flyable trim | `trim.is_physical` | — | **`failed`** | checks `\|α\| ≤ 15°` and nothing else — not throttle, not elevator, not the residual. Findings 23 and 37 |
| `jnp.linalg.solve` signals a singular system | `trim.trim`, any Newton solve | — | **`failed`** | returns `[nan, nan, inf]` silently when a control column is identically zero. Finding 38 |

### 2.3 Mode comparison — the full table the project does not print

747 cruise, CR-2144 flight condition 9. Reference = **Table IX-5 denominator,
printed p. 231**, `Z(DET)1 = .0489`, `W(DET)1 = .0673`, `Z(DET)2 = .387`,
`W(DET)2 = .964` (read at 600 dpi).

| Quantity | Engine | CR-2144 Table IX-5 | Error | In `PROJECT.md` §4? |
|---|---|---|---|---|
| Phugoid ωn | 0.05532 | 0.0673 | **−17.8%** | yes |
| **Phugoid ζ** | **0.05596** | **0.0489** | **+14.4%** | **NO** |
| **Short-period ωn** | **0.95077** | **0.964** | **−1.4%** | **NO** |
| Short-period ζ | 0.34253 | 0.387 | **−11.5%** | yes |

**Attribution**, by patching the **engine's own** cruise plant matrix with the
derivative families it omits, taken from CR-2144 Table IX-4 FC9, one family at a
time:

| | Phugoid ωn | Phugoid ζ | SP ωn | SP ζ |
|---|---|---|---|---|
| **engine as shipped** | 0.05532 (**−17.8%**) | 0.05596 (**+14.4%**) | 0.9508 (−1.4%) | 0.34253 (**−11.5%**) |
| **+ speed derivatives `Xu, Zu, Mu`** | **0.06750 (+0.3%)** | 0.05070 (+3.7%) | 0.9530 (−1.1%) | 0.34056 (−12.0%) |
| **+ α̇ derivatives `Zẇ, Mẇ`** | 0.05532 (−17.8%) | 0.05527 (+13.0%) | 0.9534 (−1.1%) | **0.38942 (+0.6%)** |
| **+ both** | **0.06750 (+0.3%)** | **0.04914 (+0.5%)** | **0.9556 (−0.9%)** | **0.38744 (+0.1%)** |
| CR-2144 Table IX-5 FC9 | 0.0673 | 0.0489 | 0.964 | 0.387 |

- **Every one of the four closes to ≤1% when both families are restored.** So
  the entire 747 cruise mode discrepancy is the two documented omissions and
  nothing else. The aerodynamic data, the conversion chain, the trim solve and
  the eigen-extraction are all exonerated by this — which is a stronger positive
  result than any of the audit's negative findings.
- **Phugoid ωn ← speed derivatives, alone.** They take it to +0.3% and leave
  short-period damping *worse* (−11.5% → −12.0%).
- **Short-period ζ ← α̇ derivatives, alone.** They take it to +0.6% and leave the
  phugoid frequency **exactly** unmoved.
- **Phugoid ζ needs both.** +14.4% shipped, +3.7% with speed alone, +13.0% with
  α̇ alone, +0.5% with both. That is why it is not attributable to either.
- `PROJECT.md` §5 already names these two families as the explanation. **This
  audit's contribution is that it is now demonstrated rather than asserted, on
  all four factors rather than two, with the two families separated.**

> **A detour worth recording, because it produced a wrong answer first.** The
> obvious instrument is a clean-room 4-state rebuild from Table IX-4's
> derivatives, with no engine code involved. It reproduces phugoid ωn (+0.3%),
> short-period ωn (−0.6%) and short-period ζ (+1.4%) — but returns phugoid
> **ζ = −0.017**, the wrong sign, and an earlier pass of this audit concluded
> from it that phugoid damping was unattributable and that CR-2144 contradicted
> itself by a factor of 4.4 in `Xu`. It does not. The clean-room rebuild
> discards two terms the engine has and CR-2144's tabulated linear model has no
> row for: the `∂u̇/∂q = −w₀` Coriolis term and the `X_q` the parabolic drag
> polar produces (finding 18). Phugoid damping is the one factor sensitive to
> both. **The lesson is that an "independent instrument" which is missing terms
> the subject has is not independent, it is different** — and on a
> near-cancelling quantity like phugoid damping that difference decides the
> sign. Agent F, working blind, flagged exactly this and named `A[0,2]` as the
> discriminator; reading it settled the matter.

**Does the conversion-chain flaw (finding 5) explain any of this? No.** Rebuilding
the cruise 747 with Appendix-A-literal `CLa`, `CDa`, `Cma`, `CD0`, `e` moves
phugoid ωn from −17.80% to −17.73% and short-period ζ from −11.49% to −12.04%.
**A candidate cause was tested and falsified**, which is why finding 5 is reported
as a bounded flaw and not as an attribution.

### 2.4 Numerics

| Claim | Status | Measured |
|---|---|---|
| `rk4_step` is classical RK4 | `verified` | stability polynomial to 4.34e-13; Simpson's `+1/120` at k=4 |
| Observed order, oscillator / 6-DOF | `verified` | 3.999819 / 3.989127 (reproduces `PROJECT.md`'s 3.98913 to six digits) |
| Richardson, still air | `verified` — an order exists | p → 4.00 |
| Richardson, smooth spatial field | `verified` — order is **1.00** | p_obs 1.0991 → 1.0019 monotonically |
| Richardson, Rankine core traverse | **no order exists** | half the triples return NaN from sign flips; the extrapolated "exact" moves with the assumed p, 1.2 m of spread between triples |
| Round-off floor at 40,000 ft | `verified` | 3.6e-11 – 9.4e-11 m, confirming the project's ~7e-11 |
| Quaternion renormalisation at dt = 0.02 | `verified` — removes 2.1e-13, bounded | but at dt = 0.1 with ‖ω‖ = 1.08 rad/s it is **secular**: 1.7e-10/step, and removing it moves the trajectory **233 m** |
| Torque-free conservation | `verified` | all three of `PROJECT.md`'s figures reproduce exactly |
| Galilean invariance, horizontal | `verified` | 2.8e-11 m |
| Galilean invariance, **vertical** wind | fails by 3.08 m | expected: altitude changes density, so a vertical boost is not a symmetry of this model. Within declared scope, but not stated anywhere |
| Rotational invariance about NED z, **including the vortex field** | `verified` | 1.1e-9 m |
| Trim convergence | `verified` quadratic | converges by iteration 3; the fixed 40 is 7–13× more than needed |
| Trim uniqueness | **`failed`** | see finding 23 |
| float64 reaches a plain user import | `verified` | `flightsim/__init__.py:10` |
| float32 cost | `verified` | 1e-8–2e-6 relative on modes; 499 m of position after 2000 s; the Cessna trim residual stalls at 9.7e-7, which would fail `TRIM_RESIDUAL_LIMIT = 1e-9` |
| `jit` vs eager | last-ulp, not bit-identical | 1.6e-17 (trim), 4.5e-13 m (1000 steps) |
| Timestep sensitivity of the headline vortex numbers | **`known flaw, bounded`, small** | Measured here over a 16× `dt` range (0.02 → 0.00125): the **in-core load excursion is −1.2393, −1.2352, −1.2403, −1.2392, −1.2387 g** — 0.4% scatter, **non-monotone**, exactly as the C⁰ core kink predicts. Whole-run peak `n_z` drifts 1.8698 → 1.8671, **0.14%**. Neither threatens the quoted −1.235 g. *Agent A reported a 0.9% effect with peak values of 2.0148/2.0237; those numbers do not reproduce from the shipped `vortex_viz.fly` path and are not relayed. Disagreement recorded, not averaged.* |
| `test_conservation.py:82` comment says "measured ~5.7e-14" | comment wrong by a decade | the value is 5.6958e-13; `PROJECT.md` has it right |

---

### 2.5 Falsification — the energy seam, and what it is

Phase 2E's brief was to produce nonphysical behaviour. It found one thing that
is unambiguously nonphysical, and this section attributes it, because a
symptom without a cause is what the prompt's Phase 3 exists to prevent.

**The violation.** In motionless, uniform air with the throttle shut, the only
energy source is the aircraft. A quasi-steady aerodynamic model of a body moving
through initially-still fluid must therefore satisfy

```
P_aero = F_aero · v_body + M_aero · ω  ≤  0
```

because the fluid starts at rest and can only gain energy. Measured on the
Cherokee at its own cruise altitude, `vel_body = [75, 0, 0]`, `ω = [0, −8.256, 0]`,
elevator at its 25° limit, throttle 0:

| quantity | value |
|---|---|
| `F_aero` | `[−2334.26, 0, −19534.06]` N |
| `M_aero` | `[0, −28100.43, 0]` N·m |
| **`P_aero`** | **+56,927.7 W** — must be ≤ 0 |

**Which channel.** Splitting the pitching-moment power `q̄Sc·Cm·q` by term, at
this state where `Cm0 = 0` and `α = 0`:

| term | coefficient | power |
|---|---|---|
| `Cmq·q̂` (damping) | +0.64939 | **−382.9 kW** — correctly dissipative |
| `Cmδe·δe` (control) | −1.04283 | **+614.9 kW** |
| force power `F·v` | — | −175.1 kW |

The damping term can never be the culprit: its power is `q̄Sc·Cmq·(c/2V)·q²`,
and `Cmq < 0`, so it is negative-definite in `q`. The control term is the one
that changes sign.

**The cause, demonstrated.** The elevator's lift increment is applied
perpendicular to the relative wind **at the CG**. A surface at arm `l` sees a
local relative wind tilted by `ε = q·l/V`, and a lift tilted by `ε` acquires a
streamwise component — a drag — of `ΔL·sin ε`. The elevator's own two
derivatives fix its arm, `l_δe = −(Cmδe/CLδe)·c`. Substituting into the linear
form of that tilt power:

```
P_tilt = q̄S·CLδe·δe · (q·l_δe/V) · V
       = q̄S·CLδe·δe · q · (−(Cmδe/CLδe)·c)
       = −q̄Sc·Cmδe·δe·q
       = −(elevator moment power)
```

**The cancellation is exact and algebraic, not fitted.** Measured:

| check | result |
|---|---|
| elevator moment power | +614.916 kW |
| tilt power `ΔL·ε·V` | **−614.916 kW** |
| sum, relative residual | **1.9e-16** |
| same identity, 500 random (V, q, δe) × 4 aircraft | max residual **5.6e-16** |

And restoring that single term removes the violation everywhere it occurred:

| aircraft | P > 0 as shipped | max P | P > 0 with the tilt restored | max P |
|---|---|---|---|---|
| boeing747 | 0 / 6400 | −515.4 kW | 0 / 6400 | −1141.0 kW |
| boeing747_approach | 0 / 6400 | −466.9 kW | 0 / 6400 | −690.8 kW |
| **cherokee** | **1190 / 6400** | **+69.2 kW** | **0 / 6400** | **−2.3 kW** |
| cessna172 | 0 / 6400 | −0.9 kW | 0 / 6400 | −3.2 kW |

**So the missing physics is a lift-tilt term whose magnitude is already
determined by derivatives the registry holds.** It requires no new source data —
which is the practical difference between this attribution and Phase 2E's, and
the reason the two are not interchangeable (see §3b).

**Bound on the consequence.** The violation is real but local, and three
independent measurements bound it:

1. **It never wins globally.** `max(E − E₀) = +0` **exactly** in every still-air
   run attempted, including five feedback laws designed from this very mechanism
   to pump energy, and full-deflection reversals at 25–1000 Hz. The creation is
   always superimposed on a larger drag loss.
2. **It is outside the declared envelope.** At 0/80,000 randomised states with
   |α| ≤ 12° (`PROJECT.md`'s own linear-aero ceiling), |q| ≤ 0.5 rad/s and
   controls to their limits, across all four aircraft, is `P_aero` positive. The
   threshold pitch rate is **±1.47 to ±3.51 rad/s** — 84 to 201 °/s — an order of
   magnitude outside anything the project's scope contemplates.
3. **Only the Cherokee reaches it.** Both 747s resist the entire corner scan; the
   Cessna needs the roll channel. A fixed-control rollout from the Cherokee's own
   trim reaches |q| = 3.677 rad/s and manufactures 520.6 J against a 1.9e7 J total.

**Verdict: `known flaw, bounded`.** It is a genuine violation of energy
conservation, it is reachable within the model, and it lies wholly outside the
envelope the project declares — but nothing in the repository stops a caller
going there, and no test asserts the bound. §7 now pins it.

---

## 3. Assumption B4 — from "no bound" to a measured bound

`ASSUMPTIONS.md` B4 says: *"A real DFDR accelerometer sits metres from the CG…
**Bound: none**"*. CR-2144 supplies what is needed:

- **Table IX-3** (printed p. 229) gives `LXP(FT) = 86.0` and `LZP(FT) = −10.0`
  for **every** 747 flight condition.
- **Table IX-5** (printed p. 231) publishes a transfer function `N(AZP/DE)` —
  normal acceleration **at the pilot station** — so the report models exactly
  this offset.

With `a_p = a_cg + ω̇×r + ω×(ω×r)` and `r = (26.21, 0, −3.05) m`:

| Encounter | CG excursion | correction at the load peak | peak \|corr\| in window |
|---|---|---|---|
| Elevator pushdown (Fig. 8 manoeuvre) | −1.9000 g | **+0.1577 g (8.3%)** | 0.4766 g |
| **Parks vortex, first core** | **−1.2352 g** | **+0.3259 g (26.4%)** | **0.4391 g** |
| Wingrove updraft column | +0.1902 g | −0.0215 g (11.3%) | 0.0246 g |

**Two things follow that the register does not say.**

1. **The register worries about the wrong case.** B4 says "in a pitch manoeuvre
   with the sensor well forward that term is not negligible". The manoeuvre is
   the *smallest* relative correction of the three, at 8.3%. The **vortex — the
   project's headline result — is 26.4%**, because it drives a large rapid `q̇`
   against a small load excursion.
2. **Re-targeting the Fig. 8 manoeuvring point at the sensor moves it into
   dangerous territory.** The project bisects the elevator until `Δn = −1.9 g`
   *at the CG*, getting 8.926° and `|α|max = 10.31°` — declared "marginal" in a
   10–12° amber band. Targeting −1.9 g at the pilot station instead needs
   **9.708° and drives `|α|max` to 11.62°**, with Δθ 30.37° → 33.05°. Still
   inside the amber band, but the margin to the 12° INVALID boundary falls from
   1.69° to 0.38°.

**The 26.4% is converged.** The correction differentiates a pitch rate through a
field whose gradient is discontinuous at the core edge, so it is exactly the
kind of number that could be an artefact of the step size. Measured across an
8× range: **26.3% at dt = 0.02, 26.4% at 0.01, 0.005 and 0.0025.** (The *peak*
correction in the window is not converged at dt = 0.02 — 0.326 g against 0.445 g
at finer steps — which is why the ratio is quoted at the load peak rather than
at the peak of the correction.)

**Caveats, stated because they bound the bound.** `LXP` is the *pilot station*,
not necessarily a DFDR accelerometer location, so 86 ft is an **upper bound** on
the arm. Wingrove & Bach never identify their aircraft type, so the true arm for
the reference data is unknown. The correct reading is therefore: *if the
reference sensor sits where CR-2144 puts the 747's pilot, the category-dependent
correction is 8–26%.* It is **category-dependent**, so it does not cancel when
clusters are compared — which is the reason it matters at all.

**The Fig. 8 orderings survive.** By |load|: updraft < vortex < manoeuvre both
before and after correction. By Δθ: 2.24 < 4.37 < 30.37, untouched (the pitch
coordinate has no sensor-offset term). The project's claims are orderings, and
the orderings hold. What does not survive is any reading of the *values*.

---

## 3b. Where the evidence agents disagreed

The prompt requires these to be resolved by evidence or recorded, never
averaged. There was one real disagreement and one apparent one.

**Resolved: the size of the conversion-chain error (finding 5).** Agent C
measured the code's `CLa` as **0.66% LOW** against Appendix A. This auditor
measured it **1.15% HIGH**. Opposite signs, so one of us was wrong about
something.

*Resolution by evidence, not by splitting.* The two omissions have **opposite
signs**: dropping Appendix A's `−2(W₀/U₀)C_N` term makes the recovered `CLa`
too high, and using `C_L/C_D` where Appendix A uses `C_N/C_X` makes it too low.
Agent C measured the second effect in isolation; this auditor measured the net
of both. Neither measurement is wrong; they are answers to different questions.

The tie-break is that CR-2144 supplies an **answer key**. Flight condition 2 has
both a dimensional table (IX-4) and a non-dimensional one (IX-2), so the chain
can be run end to end and compared with the source's own number: the code's
relations give `CLa = 5.7893` against Table IX-2's tabulated **5.70**, i.e.
**+1.57% high**, and the Appendix-A-literal inversion gives 5.6889, **−0.20%**.
So the *net* error is positive and the code is high, which is this auditor's
result. Both decompositions are recorded in `NOTATION.md` §7.2; the net is what
`AUDIT.md` reports and what the regression test pins.

**Resolved: what causes the energy creation (findings 33–34).** Agent E
attributed it to a missing drag term: *"there is **no `CD_de`** term, so a
deflected elevator generates a pitching moment that does work on the airframe
while incurring no drag penalty of its own."*

*Resolution by evidence.* That hypothesis is testable, so it was tested. The drag
coefficient increment needed to cancel the imbalance at the reproduction state is
**0.0170 — half the whole aircraft's `CD0`**, from an elevator, and it is a
number no source in this repository supplies. So E's fix would require inventing
a constant, which §3's rule forbids.

The lift-tilt term does the job instead, **exactly, and out of derivatives the
registry already holds** (§2.5). The two are not the same physics: E names a
profile-drag term whose size is free, this audit identifies an induced term whose
size is fixed by `Cmδe` and `CLδe`. The tie-break is that the lift-tilt
cancellation is an algebraic identity verified to 1.9e-16 and it takes the
violating region to **exactly zero**, whereas the `CD_de` route is a plausible
story with a free parameter. **Not averaged: E's mechanism is superseded, and its
symptom — which reproduces exactly — stands.**

> **This auditor got it wrong first, and it is worth recording.** The first pass
> computed the tilt power with the sign reversed, which made it *add* to the
> moment power instead of cancelling it, doubled the violation, and appeared to
> **falsify** the hypothesis — the scan went from 2846 violating points to 6045
> and the 747 acquired a +22 MW excursion it does not have. The error was a
> rotation direction in the x–z plane with body z pointing **down**. What caught
> it was that the "refuted" result was too strong: a correction that creates a
> 22 MW violation in an aircraft with none is not a failed hypothesis, it is a
> bug in the instrument. **The same lesson as §2.3's clean-room detour, from the
> other direction** — there, an instrument missing a term the subject had; here,
> an instrument with a term the subject would not have.

**Recorded, not resolved: how many aircraft cross the `V_MIN` floor.** Agent E
reports three of four (747 0.865 m/s, Cherokee 0.659, Cessna 0.518) over 21
fixed-control runs. Under this auditor's control settings — four runs per
aircraft at full three-axis deflection, throttle 0, 120 s, with and without an
initial rate — **only the Cherokee crosses it** (0.5375 m/s); the 747 bottoms out
at 70.3 m/s and the Cessna at 36.0 m/s. The disagreement is about which
fixed-control settings were flown, which E's report does not fully specify, so it
cannot be settled from what is recorded. **The finding does not depend on it:**
finding 40 rests on the force at *exactly* V = 0, which is a property of the
model and not of any run.

**Corrected: Agent E overstates the ground-penetration finding.** E-6 is headed
*"There is no ground, and runs reach it routinely"* and presents a microburst run
continuing 153 m below sea level. The engine half is right and is finding 39. But
`scripts/microburst.py` — the one shipped analysis that flies at terrain —
**already cuts its run at one wingspan of clearance**, with an explicit rationale:
*"there is no terrain, no landing gear and no ground effect in this model, so an
aeroplane whose wings are within their own span of the surface is not flying any
more and the integration past that point is arithmetic, not physics."* E's report
does not mention it. The finding is therefore much narrower than stated: the
guard exists, it is well reasoned, and it is **per-script rather than in the
engine**, so a new caller inherits nothing.

**Apparent only: phugoid damping.** Agent F (working blind) found an unexplained
+15.2% residual in phugoid ζ and named `A[0,2] = X_q` as the discriminator that
would settle it. This auditor's clean-room reconstruction independently returned
the *wrong sign* for the same quantity. Both were symptoms of the same missing
term, and reading `A[0,2]` — as Agent F proposed — resolved both at once. Not a
disagreement; two views of one gap, and the blind agent's proposed experiment
was the right one.

---

## 4. What I could not check, and why

Honest coverage is part of the deliverable.

| Area | Why not |
|---|---|
| **Every non-CR-2144 numerical source** | Parks 1985, Wingrove & Bach 1994, Doyle 2011, Proctor 2000, Oseguera & Bowles 1988, Caughey, McCormick, Roskam/PyFME, Nelson/Etkin/McRuer. None was in the repository. `audit_evidence/B-reference-tracing.md` records which could be retrieved. Anything not retrieved is `unverifiable — source not available`, **not** "probably fine". |
| **The 747's real aerodynamics** | Nothing here is an independent measurement of a 747. CR-2144 is a compilation of Boeing simulation data and its Section IX plots are labelled "Flexible". Every tier-2 agreement is closed-loop against one document's own arithmetic. |
| **The rigid-vs-flexible mismatch** | Needs a rigid derivative set for the same aircraft and condition. CR-2144 does not tabulate one. Unquantifiable from held sources — the register (B1) already says so and is right. |
| **Cruise `CL_M`, `CD_M`, `Cm_M`** | Published as *plots against Mach*, not tables (printed pp. 224–228). Reading them is a chart read off a poor scan. This is why the Appendix-A `W₀` correction in finding 5 is quoted with `C_NM = 0` and a sensitivity instead of a value. |
| **Table IX-2's axis system** | The table states none. See `NOTATION.md` §7.1. |
| **Phugoid damping attribution** | See §2.3. Localised to `Xu` but not explained. |
| **Whether `Xq` should be zero** | The engine carries `∂u̇/∂q = −w₀`; CR-2144's linear model does not. Both are defensible; no source in the repository settles which the transfer-function factors used. |
| **Interactive panel frame rate** | Requires a display; not attempted. |
| **Multi-realisation ensemble behaviour** | Dryden is not implemented, so the stochastic path has never run. Everything about it in `PROJECT.md` §7 is design intent, not measurement. |
| **The Parks and Wingrove & Bach field definitions** | Both papers are paywalled and were not retrieved, so the vortex equations, `PARKS_CASES`, `UPDRAFT_W0` and `UPDRAFT_SECONDS` were checked for *internal* consistency only. The Rankine structure, divergence-freedom and continuity are verified as mathematics; that they are the *identified* values is not. |
| **The claim that holding wind is "the standard treatment for Dryden and von Kármán"** | Asserted in `integrate.py`'s module docstring and repeated in `ASSUMPTIONS.md` E4. No citation exists anywhere in the repository, and none was found. `unverified`. |
| **float32 behaviour of the three wind-field guard thresholds** (`1e-12`, `1e-8`, `1e-9`) | Not attempted. They are tuned for float64 and the project is float64 throughout, but nothing enforces that. |
| **Falsification (Phase 2E)** | **Now covered.** The agent returned after the first assembly of this report; its evidence is in `audit_evidence/E-falsification.md` and every load-bearing claim was re-derived independently before entering §1 (findings 33–41), which is how three of them changed. What it did *not* reach is listed in the next four rows. |
| **The stochastic wind path** | `WindState`, the PRNG-key threading in `integrate.step`, and the Dryden/von Kármán path `integrate.py`'s docstring cites were never driven, by any agent. Every field exercised anywhere in this audit is deterministic. So finding 36's order result says nothing about the stochastic case — **which is the case the once-per-step hold was chosen for**, making this the most load-bearing untested claim in the repository. |
| **The strip path under falsification** | `wind.sampled_rates`, `loads.strip_increment`, `loads.strip_model` and `airframe.stations` were attacked by nobody. Findings 16 and 28 come from structured checks, not from an attempt to break them. |
| **The autopilot and `batched_rollout`** | `autopilot.closed_loop_rollout`, the loop-pairing boundary at `minimum_drag_speed`, and `vmap` consistency against the serial path were not attacked. The energy-pumping laws in §2.5 were hand-rolled, not the shipped controller. |
| **Energy accounting *with* a wind field** | §2.5's argument holds in still air only. No correct energy budget for a moving air mass was constructed, so **conservation during an actual turbulence encounter — the project's own subject — is untested**. This is now the largest hole. |
| **`sensors.py`** | Untouched by every agent and by this auditor. |

---

## 5. What I think is wrong with `AUDIT_PROMPT.md`

The prompt invites this, and there are six things.

1. **The source hierarchy asserts an availability it never checks.** CR-2144 is
   named "Primary numerical dataset" and exempted from the bulletproofing rules
   that govern "external sources" — but it was **not in the repository**. Followed
   literally, the prompt produces an audit in which every 747 number is
   `unverifiable`, which would have been a true but useless report. The fix was
   cheap (one HTTP GET from NTRS). **The availability check should apply to the
   named primary sources first, not only to sources found later.**

2. **"Timestep convergence by Richardson extrapolation on full simulation
   output" is not a valid instrument for this engine's headline case.**
   Richardson assumes a single known integer order. The scheme is fourth order
   in still air, **first** order in a smooth spatial field (the once-per-step
   wind hold), and has **no order at all** across the Rankine core kink, where
   the error depends on where the step grid lands relative to the crossing.
   Run anyway, as the prompt asks: **half the Richardson triples return NaN
   from sign flips, and the extrapolated "exact" solution moves with the order
   you assume — 1.2 m of spread between triples.** The project had already
   measured a least-squares slope of 0.62 there and correctly called it an
   artefact. So the prompt asks for a measurement that, applied to the vortex
   encounter, returns a number describing nothing. It is valid in still air and
   for `C∞` fields only, and the prompt should say so.

3. **"No agent attributes causes" in Phase 2 has a cost the prompt does not
   acknowledge.** The decisive attribution in §2.3 required *running an
   experiment on a causal hypothesis* — add the speed derivatives back and see
   what moves. A strict reading forbids Phase 2 from doing that, so the work
   lands in Phase 3 and Phase 2's evidence is thinner than it needs to be. The
   better rule is: **Phase 2 may falsify hypotheses; it may not conclude.**

4. **Phase 2F "blind reproduction" is weaker than it looks, and the prompt
   half-knows it.** For the diff to exist at all, the blind agent must be told
   the engine's outputs — so the check is "an independent derivation exists",
   not "an independent derivation was made in ignorance of the answer". The
   prompt's escape hatch ("mark this check compromised") covers the auditor
   reading the code, not the orchestrator leaking the answer. It should say what
   may be passed in.

5. **Phase 4's four categories have no slot for the most valuable outcome.**
   B4 was declared **unbounded** by the register and this audit **bounded** it
   from the source's own table. That is neither "valid", nor "known flaw,
   bounded" (which reads as *already* bounded), nor "unbounded", nor
   "undocumented". A fifth status — **"previously unbounded, now bounded"** — is
   what records that an audit made progress rather than just marking homework.

6. **"Every verified check converted into a permanent regression test" pins the
   wrong half.** The findings most likely to drift are the *bounds on known
   flaws*, not the verified transcriptions. This audit found the sharpest
   possible illustration: `ASSUMPTIONS.md` E4 carries a bound on the wind-hold
   seam that is **wrong by a factor of ~80**, because it measured a
   discretisation refinement where the flaw is a scheme error. It has sat there
   since session 15, asserted in three documents, with no test on it — precisely
   because it is a bound and not a verification. A regression test on that
   number would have caught it the first time someone re-measured. §7 pins both
   kinds, and labels which is which.

7. **"No agent attributes causes" is unenforceable, and the audit is better off
   assuming it will be broken.** Phase 2E was told not to attribute and
   attributed anyway — it named a missing `CD_de` as the cause of the energy
   violation, in bold, in the same paragraph as the measurement. That attribution
   is wrong (§3b), and it is wrong in the specific way the rule exists to
   prevent: it is the first plausible mechanism that fits the symptom, and it
   would have cost a fabricated constant to "fix". **But the rule did not stop
   it, and no rule will**, because an agent that has just measured a thing has a
   hypothesis about it whether or not it is allowed one. The workable version is
   the one from item 3 — *Phase 2 may propose and falsify, it may not conclude* —
   plus a requirement that any mechanism an evidence agent names be **labelled as
   a conjecture in the evidence file**, so Phase 3 knows to test it rather than
   inherit it. Had this audit relayed E-1's stated cause, the report would have
   carried a confident wrong answer with a real measurement attached to it, which
   is the most dangerous shape a finding can have.

**One thing the prompt gets exactly right, and it is worth saying:** "A known,
bounded, honestly-recorded flaw is a pass. An unexamined assumption presented as
fine is a failure." Applied to this repository that framing is what makes the
audit's negative findings small in number — `docs/ASSUMPTIONS.md` had already
done most of this work honestly, and the audit's job reduced to finding the
handful of places where a bound was claimed unobtainable and was not, or where a
document claimed something the code did not do.

---

## 6. External sources — retrieval record

Every source beyond the two named in the prompt's hierarchy, with the five
things the prompt requires of each.

### Retrieved during this audit, now in `refs/`

| Source | Identification | Availability | Why valid | Upstream provenance | Why it applies |
|---|---|---|---|---|---|
| **NASA CR-2144** | Heffley & Jewell, *Aircraft Handling Qualities Data*, NASA CR-2144, Dec 1972; STI Tech. Rep. 1004-1; Contract NAS 4-1729. Tables IX-1..IX-9, Figure IX-1, Figure IX-6, Appendix A §§4–5. | NTRS `19730003312`; **`refs/NASA-CR-2144.pdf`** | Government contractor report; a **compilation**, not a primary measurement | Its own p.242 "B-747 DATA SOURCES" cites Boeing simulation reports (Rodney & Nordwall, Boeing Rept.). **One step downstream of Boeing.** Section IX plots labelled "Flexible". | Same aircraft, same flight conditions (FC2, FC9), same CG (0.250 MGC), body axes stated in IX-4/IX-5/IX-6/IX-8 |
| **Oseguera & Bowles 1988** | NASA TM-100632, *A Simple, Analytic 3-Dimensional Downburst Model Based on Boundary Layer Stagnation Flow*, July 1988. Eqs. (5), (6); constants 1.1212, 0.22, 12.5, 0.2357. | NTRS; **`refs/NASA-TM-100632-Oseguera-Bowles-1988.pdf`** | NASA TM; velocity profiles from the TASS model, itself built on JAWS field data | TASS ← JAWS. Independent of CR-2144. | Same phenomenon; the code uses the paper's own three reported quantities and recovers λ, z*, ε through the paper's own relations |
| **Proctor, Hinton & Bowles 2000** | 9th Conf. on Aviation, Range & Aerospace Meteorology, Orlando, 11–15 Sept 2000, paper 7.7, pp. 482–487. Eqs. (3), (4), (7); 1 km; 0.1/0.13; the `(T_r − D)/W` passage; 0.15; the §4.1 500 m bound. | **`refs/Proctor-Hinton-Bowles-2000-Windshear-Hazard-Index.pdf`** | Conference paper by the NASA Langley / AeroTech group that originated the index | Index originally Bowles 1990a/1990b; this paper is the one the code reads. Independent of CR-2144. | Same index, and the code correctly restricts the thresholds to jet transports at low altitude, per the paper's own statement that they "are yet to be determined" for piston aircraft |
| **Doyle, Jiang, Smith & Grubišić 2011** | *Mon. Wea. Rev.* **139**, 3–23, DOI 10.1175/2010MWR3466.1. IOP 4, 14 Mar 2006, NSF/NCAR G-V at 13.1 km. | Open access; **`refs/Doyle-etal-2011-MWR139-T-REX-stratospheric-mountain-waves.pdf`** | Peer-reviewed journal; primary aircraft measurement | Independent of CR-2144 | Right altitude band (11.3–13.1 km against the 747's 12.19 km); the code's 3.0/6.0 m/s are half the paper's verbatim "12 m s⁻¹ to the south and 6 m s⁻¹ to the north" |
| **Caughey** | D. A. Caughey, *Introduction to Aircraft Stability and Control*, Cornell MAE 5070 course notes, Ch. 5. Eqs. (5.51), (5.52), (5.54). | **`refs/Caughey-MAE5070-Aircraft-Stability-and-Control-2011.pdf`** | Course notes — an independent **implementation**, not a dataset | **His Bibliography [2] IS CR-2144.** Same inputs, different code. The project's non-independence flag is correct and Agent B confirmed it directly. | Same aircraft, same power-approach condition |

**Verification outcome for the retrieved four:** every quotation and number the
code attributes to Oseguera & Bowles, Proctor et al., Doyle et al. and Caughey
checks out. Two details worth recording: the code's re-derivation of Oseguera's
peak-radius condition **silently corrects a sign typo in that paper's printed
r-derivative**; and the exact root of `e^{−x²}(2x²+1) = 1` is **1.120906**
against the paper's printed **1.1212** — the paper's own rounding, worth under
1e-7 in `u_max`.

### Could not be obtained

| Source | What depends on it | Status |
|---|---|---|
| **Parks, Wingrove, Bach & Mehta 1985**, J. Aircraft **22**(2), 124–129 | The entire vortex model: the Rankine field equations, `PARKS_CASES`' six numbers (r₀, V₀, spacing × 2 cases), the Scorer 2.7 spacing/diameter check, the ±25% α-inference caveat | **`unverifiable — source not available`.** AIAA paywalled; Unpaywall `is_oa:false`; NTRS holds metadata only. Bibliographic details confirmed via Crossref and match the code. **The headline result rests on this.** |
| **Wingrove & Bach 1994**, J. Aircraft **31**(4), 753–760 | `UPDRAFT_W0`, `UPDRAFT_SECONDS`, the Fig. 8 discriminator, the 5.2° pitch figure, the ±g asymmetry attribution | **`unverifiable — source not available`.** Same reasons. |
| McCormick (Cherokee), Roskam/DATCOM via PyFME (Cessna), Nelson/Etkin/McRuer (Navion) | 168 of 346 constants | **`unverifiable — source not available`.** All three are cited via `aircraft_data_validated.py`, **a file that does not exist in this repository.** |
| ICAO Doc 7488 | Every ISA constant | **`unverified`.** Values are conventional; no held source. |
| MIL-F-8785C | Nothing yet (Dryden is unimplemented) | not applicable |

**The honest summary:** the 747 work is now well sourced — the primary document
is in the repo and its tables have been read. The **turbulence** work, which is
the project's actual subject, rests on two paywalled J. Aircraft papers that
could not be retrieved. Every vortex and updraft number is `unverifiable`, and
that is the largest single gap in this audit's coverage.

## 7. Regression tests

`flightsim/tests/test_audit_regression.py` — **109 tests, all passing**, added
without modifying any existing test, tolerance or reference value. Measured
before and after: the pre-existing suite is **444 passed, 1 skipped**, re-measured
after this phase and unchanged; the suite with the audit fixtures added is
**553 passed, 1 skipped**. 444 + 109 = 553, so nothing pre-existing changed state.

**29 of the 109 are Phase 2E's**, added after the falsification agent returned.

Two kinds, and the distinction is deliberate (see §5.6):

**Verifications pinned** — Table IX-2 verbatim (21 parametrised); Table IX-3
geometry, mass and inertia for both flight conditions; the Table IX-4 round
trip including `Mq = −0.339`; the Table IX-8 lateral chain; `_unprime` against
Appendix A A-18; the Oseguera constants re-derived; the Parks vortex's
solid-body/irrotational split; the Korn/Lock offset; gust-rate signs on all
three axes against a rigid-rotating air mass; lift perpendicular and drag
anti-parallel at non-zero α **and** β; the DCM at a general attitude; FD2e eqs.
3.4-49 and 3.4-55 re-derived as source defects.

**Bounds on known flaws pinned** — the conversion chain's closure against
CR-2144's own two tables (7 quantities); the Appendix-A `W₀` omission at cruise
(1.15% / 1.13% / 0.32%); all **four** published longitudinal mode factors with
their measured errors; the two attributions, their orthogonality, and the fact
that phugoid damping needs both; the `X_q` the source does not model; the B4
sensor-offset bound for the manoeuvre (8.3%) and the vortex (26.4%); the
strip-quadrature ratio at five station counts; the wind-hold cost on the
headline Δθ (1.62% at dt = 0.02, correcting E4's ~1e-4); the one-sided
longitudinal station bias on both sides of the core; the two live defects
(strip quadrature, lateral mode swap); trim's absurd roots from plausible
guesses; and the provenance-ledger coverage gap, supplied as the source→ledger
direction `PROJECT.md` has been promising.

Phase 2E adds, under both headings: the energy violation at its reproduction
state and the **exact lift-tilt cancellation** that attributes it (parametrised
over all four aircraft); the demonstration that restoring that one term takes the
violating region from 1190 states to zero; the in-envelope dissipativity bound
that makes it a bounded flaw; the two tail-arm routes and the fact that they
agree only where the source is held; the force at exactly zero airspeed; the
silent NaN from a zero-authority control column; the absent ground plane and the
atmosphere's downward extrapolation; `is_physical` endorsing a 567-throttle root;
and the gradient-free wind field that keeps fourth order.

**Ten tests are written to fail when the defect they pin is fixed**, with a
message saying so and what to replace them with. That is deliberate: a test that
pins a flaw must not silently keep passing once the flaw is gone.
