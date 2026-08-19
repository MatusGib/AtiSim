# B — Reference tracing

Agent B. Scope: trace every numeric constant in `flightsim/` back to a source that was
actually opened and read. Evidence only; no causal attribution.

**Rule applied throughout:** if I could not point at a page/table/equation in a file I
opened in this session, the status is `unverified`. Nothing below is recalled from memory.

## Documents I actually read

| File | What it is | How obtained |
|---|---|---|
| `refs/NASA-CR-2144.pdf` | Heffley & Jewell, *Aircraft Handling Qualities Data*, NASA CR-2144, Dec 1972. §IX = B-747. PDF page = printed page + 6. Rendered to PNG at 200–600 dpi with PyMuPDF and read as images (the OCR text layer is noise). | already in repo (lead auditor) |
| `refs/NASA-TM-100632-Oseguera-Bowles-1988.pdf` | Oseguera & Bowles, *A Simple, Analytic 3-Dimensional Downburst Model Based on Boundary Layer Stagnation Flow*, NASA TM-100632, July 1988, 19 pp. | **fetched this pass** — NTRS 19880018674 |
| `refs/Proctor-Hinton-Bowles-2000-Windshear-Hazard-Index.pdf` | Proctor, Hinton & Bowles, *A Windshear Hazard Index*, 9th Conf. on Aviation, Range and Aerospace Meteorology, Orlando, 11–15 Sept 2000, paper 7.7, pp. 482–487. | **fetched this pass** — NTRS 20000116199 |
| `refs/Doyle-etal-2011-MWR139-T-REX-stratospheric-mountain-waves.pdf` | Doyle, Jiang, Smith & Grubišić, *Three-Dimensional Characteristics of Stratospheric Mountain Waves during T-REX*, Mon. Wea. Rev. **139**(1), Jan 2011, 3–23. | **fetched this pass** — AMS open access, DOI 10.1175/2010MWR3466.1 (Unpaywall `is_oa: true`) |
| `refs/Caughey-MAE5070-Aircraft-Stability-and-Control-2011.pdf` | D. A. Caughey, *Introduction to Aircraft Stability and Control*, Course Notes for M&AE 5070, Cornell, 2011, 153 pp. | **fetched this pass** — `courses.cit.cornell.edu/mae5070/Caughey_2011_04.pdf` |
| `Flight_Dynamics_-_Second_Edition.pdf` | Stengel, *Flight Dynamics*, 2nd ed. (in repo root) | already in repo |

Sources I could **not** obtain — see §5.

---

## 1. HEADLINE FINDING — the ledger enforces nothing about constants

`docs/PROJECT.md` line 61: *"a constant with no entry fails the build"*.
`docs/PROJECT.md` line 128: *"A constant added without a ledger entry fails the build."*
`flightsim/provenance.py` line 23: *"adding a constant without saying where it came from fails the build"*.

**This is false as implemented.** `flightsim/tests/test_provenance.py` contains eight tests.
All eight iterate over `LEDGER` itself and check internal consistency (category is one of
four; `detail` longer than 20 chars; `inputs` name existing keys; the graph is acyclic; three
named entries have expected categories). **No test reads any other module, parses any source
file, or compares the set of constants in the code to the set of keys in the ledger.**
Grepping the whole package (`grep -rn "LEDGER\|provenance" flightsim/`) finds only
`test_provenance.py`, plus three prose comments in `airframe.py`. Adding a bare literal
anywhere in `flightsim/` cannot fail any test.

**Coverage, measured.** AST walk over the nine in-scope modules, excluding trivial literals
(0, ±1, 2, 3, 4, 0.5, 1e-8/1e-9/1e-12 NaN guards, array indices):

| module | non-trivial numeric literals |
|---|---|
| `aircraft.py` | 261 |
| `autopilot.py` | 45 |
| `wind.py` | 14 |
| `atmosphere.py` | 8 |
| `units.py` | 8 |
| `airframe.py` | 5 |
| `aero.py` | 4 |
| `dynamics.py` | 1 |
| `sensors.py` | 0 |
| **total** | **346** |

`LEDGER` has **13 entries**, of which at most **8** correspond to a literal in the code
(`b747.S/b/c` = `aircraft.py:227`; `b747.Zq/Mq` = `:243`; `b747.Clp` = `:251`;
`strip.n_stations` = `airframe.py:68-69`; `airframe.tail_arm_band` = `airframe.py:30`). The
other five (`b747.CLq`, `b747.Cmq`, `b747.l_eff`, `strip.loading_shape`, `strip.lift_slope`)
describe computed quantities, not literals. **Ledger coverage of the code's constants is
therefore ~2%.** The other ~338 are governed by comments only, or by nothing.

`aircraft.py` is 261 of the 346, split by section (AST count, same exclusions):

| section | lines | literals |
|---|---|---|
| conversion helpers | 1–224 | 1 |
| `_boeing_747` (cruise) | 225–352 | 46 |
| `_boeing_747_approach` | 353–488 | 39 |
| `_cherokee_pa28_180` | 489–618 | 35 |
| `CESSNA172_TABLES` | 653–676 | 107 |
| `_cessna_172` | 677–780 | 26 |
| `CRUISE` dict | 783–799 | 7 |

**168 of the 261 (Cherokee 35 + Cessna 133) trace to sources I could not open**, via an
intermediate file that is not in the repository (§5).

---

## 2. Constant ledger — full classification

Legend for **status**: `verified` = I read the cited table/equation and it says this;
`unverified` = no ledger entry and no source I could open, OR a figure reading I cannot
reproduce to the stated precision; `circular` = checked against data it was fitted from;
`unverifiable` = named source not obtainable.
**L** column: ✔ = has a `provenance.LEDGER` entry; **C** = cited in a code comment only;
**–** = uncited.

### 2a. `flightsim/units.py` — 8 literals

| Constant | Location | Claimed source | L | Status | Note |
|---|---|---|---|---|---|
| `FT2M = 0.3048` | units.py:14 | "exact" (int'l foot 1959) | – | `unverified` | No standards document in repo. Internally self-consistent; not checked against a read source. |
| `LBF2N = 4.4482216152605` | units.py:18 | "exact" | – | `unverified` | same |
| `LB2KG = 0.45359237` | units.py:21 | "exact" | – | `unverified` | same |
| `KT2MS = 1852.0/3600.0` | units.py:36 | "int'l nautical mile" | – | `unverified` | same |
| `HP2W = 550*LBF2N*FT2M` | units.py:43 | "mechanical hp, 550 ft·lbf/s" | – | `unverified` | same |
| `DEG2RAD/RAD2DEG` | units.py:28-29 | `math.pi` | – | n/a | definitional |

These are conversion definitions, not physics. I flag them `unverified` only to be literal
about the rule; none is a risk. `units.py` has **no ledger entries at all**.

### 2b. `flightsim/atmosphere.py` — 8 literals, none in the ledger

| Constant | Location | Claimed source | L | Status |
|---|---|---|---|---|
| `T0 = 288.15` K | atmosphere.py:14 | comment "ISA / ICAO Doc 7488" (line 13) | C | `unverifiable — source not available` |
| `P0 = 101325.0` Pa | :15 | same | C | `unverifiable — source not available` |
| `RHO0 = 1.225` kg/m³ | :16 | same | C | `unverifiable — source not available` |
| `LAPSE = -0.0065` K/m | :17 | same | C | `unverifiable — source not available` |
| `H_TROPOPAUSE = 11000.0` m | :18 | same | C | `unverifiable — source not available` |
| `G0 = 9.80665` m/s² | :20 | same | C | `unverifiable — source not available` |
| `R_AIR = 287.05287` J/(kg·K) | :21 | same | C | `unverifiable — source not available` |
| `GAMMA = 1.4` | :22 | *nothing* | – | `unverified` |

ICAO Doc 7488 is a paid ICAO publication; I did not obtain it. The comment names a document
but no table or page within it. `GAMMA = 1.4` carries no citation at all, not even the
blanket ISA comment (which sits above `T0`).

### 2c. `flightsim/aero.py` — 4 literals

| Constant | Location | Claimed source | L | Status | Note |
|---|---|---|---|---|---|
| `V_MIN = 1.0` m/s | aero.py:25 | "NaN guard" | – | n/a | modelling guard, self-declaring |
| `20.0` in `CD_wave = 20(M−Mcrit)^4` | aero.py:57, also aircraft.py:288 | "Lock's fourth-power drag-rise law" (aero.py:28) | – | `unverified` | No Lock reference — no author, year, page. Not in any document in `refs/`. |
| `0.1` in `_MDD_OFFSET` | aero.py:30 | "definition of drag divergence, dCD/dM = 0.1 at M_dd" | – | `unverified` | Same: named convention, no citable document. |
| `80.0` in `_MDD_OFFSET` | aero.py:30 | derived (= 4×20) | – | n/a | algebra from the `20.0` above |
| Korn eq. form `κ/cosΛ − (t/c)/cos²Λ − CL/(10cos³Λ)` | aero.py:43-47 | "Korn equation" | – | `unverified` | The `10` in the CL term is a constant of the correlation with no citation. |

`aero.py` has **no ledger entries**. The wave-drag model is three uncited constants
(20, 0.1, 10) plus `kappa_airfoil`.

### 2d. `flightsim/dynamics.py` — 1 literal, plus quoted text

| Constant / claim | Location | Claimed source | L | Status | Note |
|---|---|---|---|---|---|
| F-factor `F = U̇x/g − w/Va` | dynamics.py:178 | "Proctor et al. Eq. (3)" | – | **`verified`** | Proctor/Hinton/Bowles 2000, §3, Eq. (3), printed p.482–483. |
| sign convention "positive for a descending air mass (w<0) and a wind field accelerating in the direction of the flight path" | dynamics.py:164-165 | quoted | – | **`verified`** | Verbatim, printed p.483. |
| "U_x … positive for a tailwind" | dynamics.py:358 | Proctor Eq. (3) requirement | – | **`verified`** | "U_x is the component of atmospheric wind directed horizontally along the flight path (positive for a tail wind)", printed p.483. |
| 3-term shear decomposition | wind.py:361-364 | "Proctor et al. Eq. (4)" | – | **`verified`** | Eq. (4) = ∂Ux/∂x·ẋ + ∂Ux/∂z·ż + ∂Ux/∂t; the paper's own prose names all three terms, printed p.483. |
| `length = 1000.0` m averaging window | dynamics.py:182 | "Proctor et al. Eq. (7)"; "FAA adopted the 1 km average" | – | **`verified`** | "The Federal Aviation Administration (FAA) has adopted the 1-km average F-factor as its hazard metric for windshear detection systems on jet transports", printed p.484. |
| thresholds 0.1 / 0.13 | dynamics.py:195 (prose only, not a literal) | Proctor et al. | – | **`verified`** | "…considers windshear hazardous for F > 0.1, and a must alert threshold at 0.13 (e.g., Hinton 1994)", printed p.484. |
| quote "The minimum averaging scale and hazard threshold are yet to be determined for these types of aircraft." | dynamics.py:196-197 | Proctor et al. | – | **`verified`** | Verbatim, printed p.484. |
| quote on F-factor oscillations / perceived turbulence | dynamics.py:187-191 | Proctor et al. | – | **`verified`** (light paraphrase) | Paper: "the positive values of F are over small length scales…"; code opens "Peaks of F are over small length scales…". Same sentence, first clause reworded. |
| quote "For a strong shear that exceeds the thrust capability… some compromise of the two." | dynamics.py:220-223 | "Proctor et al., discussing their Eq. (5)" | – | **`verified`** | Verbatim, printed p.483, in the paragraph immediately after Eq. (5). |
| 0.15 for a 4-engine jet | dynamics.py:231 (prose) | Proctor et al. | – | **`verified`** | "For a 4-engine jet aircraft, the ratio may be about 0.15 at full thrust and maximum takeoff weight", printed p.483. |
| 500 m altitude bound, §4.1 | dynamics.py:173 (prose) | Proctor et al. §4.1 | – | **`verified`** | §4.1 "Altitude Bound for Windshear Threat": "the hazard from windshear is limited to elevations below 500 m", printed p.484. |

**`dynamics.py`'s windshear section is the cleanest citation work in the package.** Every
quoted string and every number I checked is in the paper, at the stated equation number.

### 2e. `flightsim/airframe.py` — 5 literals

| Constant | Location | Claimed source | L | Status | Note |
|---|---|---|---|---|---|
| `TAIL_ARM_BAND = (2.0, 6.0)` | airframe.py:30 | `LEDGER["airframe.tail_arm_band"]`, DECLARED | ✔ | `verified` (as DECLARED) | Ledger carries the sensitivity as required. Not sourced, correctly so. |
| `N_SPAN = 9`, `N_LON = 9` | airframe.py:68-69 | `LEDGER["strip.n_stations"]`, DECLARED | ✔ | `verified` (as DECLARED) | |
| `_SENSITIVITY_TAPER = 0.3` | airframe.py:122 | "a representative transport value" | – | `unverified` | **No ledger entry.** The comment is explicit that it is not 747 geometry, but "representative transport value" names no source. |
| `8.0` in `a0 = -8*Clp` | airframe.py:212 | `LEDGER["strip.lift_slope"]`, CALIBRATED | ✔ | `verified` (algebra) | The −a0/8 elliptic closed form is re-derived in `test_airframe.py:214`. |
| `l_eff/c = -Cmq/CLq` | airframe.py:52 | "Stengel eqs. 3.4-7, 3.4-10, 3.4-12" | ✔ | `unverified` | Stengel PDF is in the repo but I did not open it this pass; equation numbers not checked. Agent scope split. |

### 2f. `flightsim/wind.py` — 14 literals

| Constant | Location | Claimed source | L | Status | Note |
|---|---|---|---|---|---|
| Hannibal `r0=600 ft, v0=85 ft/s, spacing=3500 ft` | wind.py:113 | Parks et al. 1985 pp.127–128 | – | `unverifiable — source not available` | AIAA paywalled, not OA (Unpaywall `is_oa:false` for 10.2514/3.45095), NTRS record 19850039692 is metadata-only with `downloads: []`. **Bibliographic citation itself verified via Crossref: J. Aircraft 22(2), Feb 1985, 124–129 — matches wind.py:66 exactly.** |
| Morton `r0=450 ft, v0=70 ft/s, spacing=3200 ft` | wind.py:114 | same | – | `unverifiable — source not available` | same |
| Rankine vortex Eqs. (3)–(6) form | wind.py:75-76 | Parks et al. §"Vortex Modeling" | – | `unverifiable — source not available` | Implementation is internally consistent (both branches agree at r=r0) but the source equations were not read. |
| `UPDRAFT_W0 = 80 ft/s` | wind.py:251 | "Wingrove & Bach 1994, p.756, Bermuda 12 Oct 1983" | – | `unverifiable — source not available` | AIAA paywalled. **Bibliographic citation verified via Crossref: the J. Aircraft 31(4), Jul 1994, 753–760 paper is Wingrove & Bach, "Severe turbulence and maneuvering from airline flight records", DOI 10.2514/3.46557 — volume/issue/pages match wind.py:230 exactly.** NTRS 19970012379 / 19920072706 are metadata-only. |
| `UPDRAFT_SECONDS = 20.0` | wind.py:252 | same | – | `unverifiable — source not available` | |
| `sharpness` (no default literal) | wind.py:270 | explicitly DECLARED in the comment | – | `verified` (as declared) | **Not in the ledger** despite being called "a DECLARED MODELLING PARAMETER" in prose at wind.py:243. |
| `LEE_WAVE_AMPLITUDE = {north: 3.0, south: 6.0}` | wind.py:313 | Doyle et al. 2011, "12 m/s south, 6 m/s north crest-to-trough", halved | – | **`verified`** | Doyle et al., printed p.10 (PDF p.8), §5c *Research aircraft measurements*: "The wave amplitude (crest to trough) of the primary wave is 12 m s⁻¹ to the south and 6 m s⁻¹ to the north." Verbatim. Half of each = 6.0 / 3.0 ✔ |
| — attribution "IOP 4 (14 March 2006), G-V at 13.1 km" | wind.py:296 | Doyle et al. | – | **`verified`** | Same paragraph opens "…on 14 March during IOP 4… flight level data from the 13.1-km level"; Fig. 6 caption "Flight-level data from the G-V at 13.1 km on 14 Mar (IOP 4)". |
| — quote "maximum vertical velocities in excess of 6 m s-1" | wind.py:300 | Doyle et al. | – | **`verified`** | Verbatim, same paragraph. |
| — "flew legs at 11.3 km and 13.1 km" | wind.py:293 | Doyle et al. | – | **`verified`** | Table 1 (PDF p.5) lists 37 kft = 11.3 km on every IOP; Fig. 10 is "G-V at 13.1 and 11.3 km on 16 Apr (IOP 13)". |
| `LEE_WAVE_WAVELENGTH = 25_000.0` m | wind.py:321 | DECLARED; "the paper's 20–35 km is tropospheric and it says stratospheric wavelengths are shorter without quantifying them" | – | **`verified` (as DECLARED, and the *justification* is exactly right)** | Doyle et al. printed p.16 (PDF p.14): "…a tropospheric lee wave characterized by relatively long wavelengths (~20–35 km)… **Shorter wavelengths are apparent in the stratosphere.**" Both clauses in the same paragraph, no stratospheric number given. **Not in the ledger** despite being DECLARED. |
| `MICROBURST_PEAK_RADIUS_RATIO = 1.1212` | wind.py:408 | Oseguera & Bowles TM-100632 | – | **`verified`** | O&B printed p.5 (PDF p.7): "the values **1.1212** and **12.5** were obtained from iteration for the ratios r/R and z*/ε, respectively." |
| `MICROBURST_ZM_OVER_ZSTAR = 0.22` | wind.py:409 | same | – | **`verified`** | O&B printed p.4 (PDF p.6), displayed equation "z_m/z* = 0.22"; repeated in the nomenclature block on PDF p.10 ("From TASS"). |
| `MICROBURST_ZSTAR_OVER_EPS = 12.5` | wind.py:410 | same | – | **`verified`** | O&B printed p.5 (PDF p.7), as above. |
| `MICROBURST_UMAX_COEFF = 0.2357` | wind.py:411 | same | – | **`verified`** | O&B printed p.5 (PDF p.7): "u_max = 0.2357 λ R"; repeated PDF p.10. |
| Microburst eqs. (5) and (6) as coded | wind.py:392-393, 464-484 | O&B eqs. 5 and 6 | – | **`verified`** | O&B printed p.4 (PDF p.6), Eq. (5) `u = (λR²/2r)[1−e^(−(r/R)²)](e^(−z/z*) − e^(−z/ε))` and Eq. (6) `w = −λe^(−(r/R)²)[ε(e^(−z/ε)−1) − z*(e^(−z/z*)−1)]`. The code's rearranged w is algebraically identical. |
| — code's re-derivation claim "1.1212 solves exp(−x²)(2x²+1)=1" | wind.py:402-403 | code's own | – | **`verified`, and it silently corrects a typo in the source** | O&B's printed r-derivative condition is `2(r/R)² = e^(−(r/R)²) − 1`, which has no non-zero real root (LHS ≥ 0, RHS ≤ 0); the sign in the exponent must be positive. The code's form is the correct one and reproduces 1.1212 to 1.3e-4. |
| — code's claim "0.22 is ln(12.5)/11.5" | wind.py:403 | code's own | – | **`verified`** | O&B's printed z-derivative condition is `z_m/z* = ln(z*/ε)/((z*/ε) − 1)`; ln(12.5)/11.5 = 0.21963. |

`wind.py` has **no ledger entries at all** — 14 literals including four DECLARED modelling
parameters that the prose itself labels DECLARED.

### 2g. `flightsim/autopilot.py` — 45 literals, none in the ledger, none cited

| Constant | Location | Claimed source | L | Status |
|---|---|---|---|---|
| `BOEING747_GAINS` (16 gains) | autopilot.py:267-284 | "Hand-tuned against the trimmed 747 at 40,000 ft, M 0.80" | – | `unverified` |
| `CHEROKEE_GAINS` (16) | :297-314 | "Hand-tuned… roll loop sized from wn ~2 rad/s against Lda=3.1, Lp=−2.78" | – | `unverified` |
| `CESSNA172_GAINS` (16) | :325-342 | "Hand-tuned against the trimmed Cessna" | – | `unverified` |
| `BOEING747_APPROACH_GAINS` (7 overrides) | :350-358 | "re-scaled for dynamic pressure" | – | `unverified` |

These are design choices, not physical data, so `unverified` is the expected and acceptable
outcome. Recording them anyway because PROJECT.md's stated rule is *every* constant, and 45
of the 346 are here with no ledger entry and no sensitivity statement. `beta_p = 0.0` for the
Cessna carries a genuine, checkable justification (the rudder set is zeroed in `aircraft.py`).

### 2h. `flightsim/aircraft.py` — 261 literals

**747 cruise, `_boeing_747()` (aircraft.py:225–352).** All CR-2144 numbers below were read
by me from rendered page images this pass unless marked "(lead)".

| Constant | Location | Claimed source | L | Status | Note |
|---|---|---|---|---|---|
| `S,b,c = 5500, 195.68, 27.31` | :227 | Table IX-3 header | ✔ | **`verified`** | Table IX-3 header, printed p.229 (PDF 235): "s = 5500 sq ft, b = 195.68 ft, c̄ = 27.31 ft". Ledger's stated page 229 is correct. |
| `W = 636636.0` lb | :228 | Table IX-3 FC9 | – | **`verified`** | IX-3 col 9, W(LBS) = 636636. |
| `Ix,Iy,Iz,Ixz = 1.82e7, 3.31e7, 4.97e7, 970056` | :229 | Table IX-3 FC9 | – | **`verified`** | IX-3 col 9: .182E+8, .331E+8, .497E+8, 970056. |
| `U0, qbar = 774.0, 177.0` | :230 | Table IX-3 FC9 | – | **`verified`** | IX-3 col 9: VTO(FPS)=774, Q(PSF)=177. |
| `alpha0 = 4.60 deg` | :231 | Table IX-3 FC9 | – | **`verified`** | IX-3 col 9: ALPHA(DEG)=4.60. |
| `mach0 = 0.800` | :232 | Table IX-3 FC9 | – | **`verified`** | IX-3 col 9: M = .800, H = 40 K. |
| `CD_trim = 0.043` | :237 | "Figure IX-6, CD vs Mach, 40,000 ft curve read at M=0.80… ±0.003" | – | **`verified` (figure reading)** | The CD-vs-Mach log plot with flight-condition markers is at printed p.219 (PDF 225); FC-9 marker sits just below the .05 gridline on the 40,000 ft dash-dot curve. My own pixel reading of the curve gives ≈0.042–0.044. **Caveat: the page carries no caption.** Figure number IX-6 is inferred from the section's sequence (Fig. IX-1 captioned at printed p.212; "Figure IX-4. B-747 SAS" captioned at printed p.215 ⇒ printed 218 = IX-5, printed 219 = IX-6). |
| `Xw,Zw,Zq,Mw,Mq,Zde,Mde = 0.0389, −0.317, −5.16, −0.00105, −0.339, −17.9, −1.16` | :243 | Table IX-4, body axis, FC9 | ✔ (Zq, Mq only) | **`verified` (lead)** | Ledger's stated page 230 is correct (PDF 236). |
| `Yb = −43.2`, `Ydr_star = 0.00729` | :248 | Table IX-8 FC9 | – | **`verified`** | Table IX-8, printed p.234 (PDF 240), col 9: YB = −43.2, Y*CR = .00729. Internal cross-check: YV = −.0558 and YB/U0 = −43.2/774 = −.05581 ✔ |
| `lateral_primed` 10 values | :250-254 | Table IX-8 FC9, primed | ✔ (`Clp` only) | **`verified`** | IX-8 col 9: LB′=−3.05, NB′=.598, LP′=−.465, NP′=−.0316, LR′=.388, NR′=−.115, L′CA=.143, N′CA=.00775, L′CR=.153, N′CR=−.475. All ten match. Ledger's stated page 234 is correct. |
| `_B747_G = 32.174` ft/s² | :222 | *nothing* | – | `unverified` | Uncited. Used to convert W→slugs. |
| `sweep = 37.5 deg` | :282, :454 | "quarter-chord" — **no source** | – | `unverified` | Not in CR-2144 §IX (Table IX-3 tabulates only S, b, c̄). Uncited. |
| `t_over_c = 0.09` | :283, :455 | "mean; 747 wing runs ~13% root to ~8% tip" — **no source** | – | `unverified` | Uncited. |
| `kappa = 0.87` | :284, :456 | "conventional (non-supercritical) 1960s section" — **no source** | – | `unverified` | Uncited; appears 4× across all four aircraft. |
| `max_thrust = 4 × 43500 lbf` | :347, :483 | "4 × Pratt & Whitney JT9D-3A at 43,500 lbf sea-level static" | C | `unverifiable — source not available` | No engine document held. |
| `thrust_lapse = 0.8` | :348, :484 | "the usual high-bypass turbofan value; CR-2144 models no engine, so this is a modelling choice, not source data" | – | `unverified` | Honestly self-labelled. |
| `elevator/aileron/rudder_limit = 25/20/25 deg` | :349-351 etc. | *nothing* (747); "MODELLING CHOICE — deflection limits are not in the source" for the light aircraft | – | `unverified` | The 747 pair (:349-351, :485-487) carries **no** comment at all; the Cherokee/Cessna copies do. |
| `CD0`, `e` (derived) | :290-291 | back-solved from `Xw` + `CD_trim` | – | **`circular`** | See §4. |
| `CYp = CYr = 0.0` | :331-332 | "CR-2144 does not tabulate CYp or CYr for the 747 in any configuration" | – | **`verified`** | Table IX-8 (printed p.234) has rows YV, YB, LB′, NB′, LP′, NP′, LR′, NR′, Y*CA, L′CA, N′CA, Y*CR, L′CR, N′CR — no Yp, no Yr. Table IX-1 (p.216) and IX-2 (p.217) likewise list Cyβ and Cyδr only. Claim is correct. |

**747 power approach, `_boeing_747_approach()` (aircraft.py:383–488).**

| Constant | Location | Claimed source | Status | Note |
|---|---|---|---|---|
| 29 non-dimensional derivatives, `V0`, `alpha0` | :402-420 | Table IX-2 | **`verified` (lead)** | |
| `W = 564032`, `Ix..Ixz = 0.142e8, 0.323e8, 0.454e8, 870050` | :398-399 | Table IX-3 FC2 | **`verified`** | IX-3 col 2: 564032., .142E+8, .323E+8, .454E+8, 870050. |
| Q = 92.2 psf / VTO = 165 KTAS / ALPHA = 5.70 consistency claim | :395-396 | Table IX-3 FC2 vs IX-2 header | **`verified`** | IX-3 col 2: Q(PSF)=92.2, VTO(KTAS)=165., ALPHA(DEG)=5.70. |
| Figure IX-1 side of the "SOURCE CONFLICT" | :390-391 | "Figure IX-1's Power Approach block gives W = 564,000 lb with Ix/Iy/Iz/Ixz = 13.7/30.5/43.1/0.825 × 10⁶" | **`verified`** | Figure IX-1, printed p.212 (PDF 218), "Power Approach Configuration / Max Landing Weight / 20° Flaps / Gear Up / 1.4 Vs / W = 564,000 lb / c.g. at 0.25 c̄ / Ix = 13.7 × 10⁶ / Iy = 30.5 × 10⁶ / Iz = 43.1 × 10⁶ / Ixz = 0.825 × 10⁶ slug-ft² / Body Axis". **Every digit as the code states.** |
| "inertias up to 6% larger" | :392-393 | derived | **`verified`** | 14.2/13.7 = +3.6%, 32.3/30.5 = **+5.9%**, 45.4/43.1 = +5.3%, 0.870/0.825 = +5.5%. "Up to 6%" is right. |
| "For CRUISE the two agree (both 18.2e6, 970056)" | :396-397 | Fig IX-1 nominal vs IX-3 FC9 | **`verified` with one wording slip** | Figure IX-1 "Nominal Configuration": W = 636,600 lb, Ix = 18.2 × 10⁶, Iy = 33.1 × 10⁶, Iz = 49.7 × 10⁶, **Ixz = 0.97 × 10⁶**. Table IX-3 FC9 gives Ixz = 970056 and W = 636636. So the two agree *to Figure IX-1's stated precision*, but Figure IX-1 does not say "970056" — it says 0.97 × 10⁶. Also the weights differ (636,600 vs 636,636). Minor. |
| CL/mass consistency check "Figure IX-1's 564,000 lb gives 1.1126" | :377-378 | Fig IX-1 + IX-2 | **`verified`** | Both weights are in the document; 564,000 lb at 165 KTAS SL gives W/qS = 1.1126 against IX-2's CL = 1.11. Two different pages, so a genuine cross-page check. |
| `CD0 = 0.0377`, `e = 0.8766` | :434-436 | two equations, two unknowns from IX-2's CL, CD, CLα, CDα | **`verified` (algebra), not circular** | Unlike cruise, both inputs are tabulated; no chart reading. |

**Cherokee PA-28-180 (aircraft.py:516–618) — 35 literals.**

| Constant | Claimed source | Status |
|---|---|---|
| mass 1090 kg, S/b/c 15/9.11/1.6, Ixx/Iyy/Izz 3100/1700/1400, U0 50, rho 1.06, CD_trim 0.0615, 21 dimensional derivatives (:517-530) | "McCormick, *Aerodynamics, Aeronautics, and Flight Mechanics*, worked example for the Cherokee 180, as collated in **`aircraft_data_validated.py`**" | `unverifiable — source not available` |
| self-check targets "CZa −4.68, CMa −0.741, CMq −7.42, CZde −0.934, CMde −2.4" (:505-506) | McCormick's own non-dimensional table | `unverifiable — source not available` |
| "CD0 = 0.0343, inside the 0.03–0.04 band the source predicts independently" (:551) | McCormick | `unverifiable — source not available` |
| `max_thrust = 0.8 × 180 hp / U0` (:612) | "MODELLING CHOICE… Lycoming O-360 (180 hp) at 80% propeller efficiency" | `unverified` |

**`aircraft_data_validated.py` does not exist anywhere in this repository.** `ls
flightsim/aircraft_data_validated.py` → no such file; `grep -rn aircraft_data_validated` finds
it only in these two comments (and their copies inside `.claude/worktrees/`). So the stated
provenance chain for both light aircraft terminates in a file the repo does not hold, and
McCormick itself is a textbook I could not obtain. Every Cherokee number is `unverifiable`.

**Cessna 172 (aircraft.py:655–780) — 133 literals.**

| Constant | Claimed source | Status |
|---|---|---|
| `CESSNA172_TABLES` — 107 counted literals across 10 series (:655–675) | "Roskam and USAF DATCOM as transcribed in PyFME (AeroPython/PyFME, MIT), collated in `aircraft_data_validated.py`" | `unverifiable — source not available` |
| mass 1043.3, S/b/c 16.2/10.91184/1.49352, Ixx/Iyy/Izz 1285.3/1824.7/2666.7 (:681-683) | same | `unverifiable — source not available` |
| `CLq = 7.282`, `Cmq = −6.232`, `CYb = −0.268`, `Cnb = 0.0126` (:746, 750, 752, 765) | same | `unverifiable — source not available` |
| `Zde=−17.19, Mde=−36.23, Lda=135.9, Nda=−3.108`, `airspeed=67.0` (:719-725) | "the source's own linearisation at 67 m/s" | `unverifiable — source not available` |
| `U0, rho = 60.0, 1.055` (:692) | the project's own choice, argued in the comment | `unverified` (declared choice, no ledger entry) |
| `_C172_LINEAR_MAX_ALPHA_DEG = 10.0` (:677) | the project's own fit window | `unverified` (declared choice, no ledger entry) |
| the sign-flip correction at α = 2.5° in `Clp` (:666-667) | "The source corrects a sign-flip typo (PyFME has +0.487…)" | `unverifiable — source not available` |
| `CRUISE` dict entries (:791–798) | Table IX-3 FC9 / IX-2 header / the two light-aircraft sources | 747 entries **`verified`**; light-aircraft entries `unverifiable` |

---

## 3. CR-2144 items checked this pass (item 2 of the brief)

### 3a. Figure IX-1 — **verified, both sides of the conflict**

Printed p.212 = PDF p.218, captioned "Figure IX-1. B-747 Flight Conditions". Read as a
600-dpi rotated crop.

| Block | Figure IX-1 | Table IX-3 (printed p.229) | Δ |
|---|---|---|---|
| Nominal / FC 9 W | 636,600 lb | 636,636 lb | 0.006% |
| Nominal / FC 9 Ix | 18.2 × 10⁶ | .182E+8 | 0 |
| Nominal / FC 9 Iy | 33.1 × 10⁶ | .331E+8 | 0 |
| Nominal / FC 9 Iz | 49.7 × 10⁶ | .497E+8 | 0 |
| Nominal / FC 9 Ixz | 0.97 × 10⁶ | 970056 | 0.006% |
| Power Approach / FC 2 W | 564,000 lb | 564,032 lb | 0.006% |
| Power Approach / FC 2 Ix | 13.7 × 10⁶ | .142E+8 | **+3.6%** |
| Power Approach / FC 2 Iy | 30.5 × 10⁶ | .323E+8 | **+5.9%** |
| Power Approach / FC 2 Iz | 43.1 × 10⁶ | .454E+8 | **+5.3%** |
| Power Approach / FC 2 Ixz | 0.825 × 10⁶ | 870050 | **+5.5%** |

The conflict is real, is confined to the approach configuration, and "up to 6%" is accurate.
Figure IX-1's Power Approach block also states the configuration the code quotes: "Max
Landing Weight / 20° Flaps / **Gear Up** / 1.4 Vs / c.g. at 0.25 c̄ / Body Axis".

### 3b. Table IX-1 (landing configuration) — **three of the four cross-check numbers verified, one is wrong**

Table IX-1, printed p.216 = PDF p.222. Header: h = sea level, VTo = 131 KTAS, α₀ = 8.5°,
δs = −6.3°.

| aircraft.py:216-219 claims | Table IX-1 actually says | Model value (recomputed) | Verdict |
|---|---|---|---|
| "Cmde −1.44 vs **−1.40**" | Cm_δe = **−1.40**/rad | −1.4442 | ✅ |
| "CLde 0.364 vs **0.396**" | CL_δe = **.356**/rad | 0.36383 | ❌ **the reference number is wrong** |
| "Clb −0.294 vs **−0.281**" | Cl_β = **−.281**/rad | −0.29444 | ✅ |
| "Cnb +0.172 vs **+0.184**" | Cn_β = **.184**/rad | +0.17155 | ✅ |

`.356` was re-read at 500 dpi on an isolated crop to rule out a 5/9 confusion; the
neighbouring `Cm_δe = −1.40/rad` and `Cm_M = .36` render at the same quality. **0.396 is not
in Table IX-1, and is not Table IX-2's value either (IX-2 gives CL_δe = 0.338).** The effect
of the error is to *understate* the model's agreement: the true comparison is 0.364 vs 0.356
(2.2%), not 0.364 vs 0.396 (8.1%).

Other Table IX-1 values, for the record: CL 1.76, CD .263, CLα 5.67, CDα 1.13, Cmα −1.45,
CLα̇ −6.7, Cmα̇ −3.3, CLq 5.65, Cmq −21.4, CLM −1.1, CmM .36; Cyβ −1.08, Clp −.502,
Cnp −.222, Clr .195, Cnr −.36, Clδa .0530, Cnδa .0083, Cyδr .179, Clδr 0, Cnδr −.112.

### 3c. Transfer-function factors — **all six 747 CRUISE reference modes verified**

The lateral factors are **not** in Tables IX-5/6/7. Actual layout:

| Table | Printed p. | PDF p. | Title |
|---|---|---|---|
| IX-5 | 231 | 237 | B-747 **ELEVATOR** TRANSFER FUNCTION FACTORS, Bare Airframe (BODY AXIS SYSTEM) |
| IX-6 | 232 | 238 | B-747 **THRUST** TRANSFER FUNCTION FACTORS, Bare Airframe (BODY AXIS SYSTEM) |
| IX-7 | 233 | 239 | B-747 **LONGITUDINAL HANDLING QUALITIES PARAMETERS**, Stick Fixed (Body Axis) |
| IX-8 | 234 | 240 | B-747 LATERAL-DIRECTIONAL DIMENSIONAL DERIVATIVES (BODY AXIS) |
| **IX-9** | **235** | **241** | B-747 **AILERON** TRANSFER FUNCTION FACTORS, SAS Off (BODY AXIS SYSTEM) |

| Project's reference value | Where it actually is | Read value | Verdict |
|---|---|---|---|
| Phugoid ωn 0.0673 rad/s | **Table IX-5**, DENOMINATOR row `W(DET)1`, column F/C 9 (H = 40 K, M = .800) | .0673 | ✅ |
| Short-period ζ 0.387 | **Table IX-5**, DENOMINATOR row `Z(DET)2`, F/C 9 | .387 | ✅ |
| Dutch roll ωn 0.947 rad/s | **Table IX-9**, DENOMINATOR row `W(DET)1`, F/C 9 | .947 | ✅ |
| Dutch roll ζ 0.0349 | **Table IX-9**, DENOMINATOR row `Z(DET)1`, F/C 9 | .0349 (positive) | ✅ |
| Roll τ 1.779 s | **Table IX-9**, `1/T(DET)2` = .562 ⇒ 1/.562 = **1.7794 s** | derived | ✅ |
| Spiral τ 137.0 s | **Table IX-9**, `1/T(DET)1` = .00730 ⇒ 1/.00730 = **136.99 s** | derived | ✅ |

Also read at F/C 9 in Table IX-5: `Z(DET)1` = .0489 (phugoid ζ) and `W(DET)2` = .964
(short-period ωn). Table IX-6 (thrust) carries the same four denominator values, as it must.

**Citation-form note.** `flightsim/validation.py:185` defines
`_IX5 = "NASA CR-2144 Table IX-5 via PROJECT.md section 4, '747 modes vs CR-2144'"` and uses
it as the `source` for `747cruise_phugoid_wn_ref = 0.0673` and
`747cruise_short_period_zeta_ref = 0.387`. Both numbers are correct, but the citation routes
through the project's own document rather than the report. The four **lateral** reference
values in `docs/PROJECT.md` §4 ("747 modes vs CR-2144") are attributed there to CR-2144
generally; they are in **Table IX-9**, not IX-5. Two of the four (roll τ, spiral τ) are
reciprocals of tabulated quantities, i.e. DERIVED, not read.

### 3d. Appendix A relations — mostly exact, one relation the report does not contain

Appendix A §5 "DIMENSIONAL STABILITY DERIVATIVE DEFINITIONS", pages A-16 to A-18 (PDF
338–340), and Appendix B page B-1 (PDF 341).

| Code relation | Appendix A / B | Verdict |
|---|---|---|
| `_unprime`: `L' = (L + (Ixz/Ix)N)/(1−Ixz²/(IxIz))`, `N' = (N + (Ixz/Iz)L)/(…)` (aircraft.py:191-192) | A-18: `L′β = (Lβ + IxzNβ/Ix)G`, `N′β = (Nβ + IxzLβ/Iz)G`, `G = 1/(1 − Ixz²/(IxIz))` | ✅ **exact** |
| `Cmde = Mde·Iy/(qS·c)` | A-17: `Mδe = (ρSc V_To²/2Iy) Cmδe` | ✅ exact |
| `Cmq = Mq·2·U0·Iy/(qS·c²)` | A-17: `Mq = (ρSc²V_To/4Iy) Cmq` | ✅ exact (with U0 ≡ V_To) |
| `CYb = Yv·V·m/qS`, `CYdr = Y*dr·U0·m/qS` | A-17: `Yv = (ρSV_To/2m)Cyβ`, `Y*δr = (ρSV_To/2m)Cyδr` (1/sec; A-10 confirms Y* is the U0-normalised form) | ✅ exact |
| `Clb = Lv·V·Ix/(qS·b)`, `Clp = Lp·Ix/(qS b²/2V)`, and the Cn analogues | A-17/A-18: `Lβ = (ρSV²b/2Ix)Clβ`, `Lp = (ρSVb²/4Ix)Clp`, `Nr = (ρSVb²/4Iz)Cnr`, `Lδa = (ρSV²b/2Ix)Clδa` | ✅ exact |
| `CDa` back-solve from `Xw` (aircraft.py:281) | Appendix B B-1: `CXα = CDα cos α₀ − CD sin α₀ − CLα sin α₀ − CL cos α₀` | ✅ exact inversion |
| `CLa = −Zw·m·U0/qS − CD` (aircraft.py:263) | B-1 exact form is `CNα = CLα cos α₀ − CL sin α₀ + CDα sin α₀ + CD cos α₀`. Code uses the small-angle `CLα ≈ CNα − CD`. | ⚠ **approximation, not the stated relation.** At α₀ = 4.6° the exact inversion gives CLα = 4.977 vs the code's 4.944 — **0.67%**. Note the code *does* use the exact form for CDα two lines later, so the two are inconsistent with each other. |
| `Cma = Mw·Iy·U0/(qS·c)` | A-17: `Mw = (ρScU0/2Iy)[Cmα + (2W0/U0)(Cm + (M/2)CmM)]` — the code inverts the leading term only | ⚠ **approximation.** The dropped term is ~2% of Cmα if CmM is of the order Table IX-1 reports for landing (.36). Same for `Zw`→`CNα`, where the dropped term is ~0.7%. |
| `CLq = −Zq·2·m·U0/(qS·c)` (aircraft.py:264) and `LEDGER["b747.CLq"]`'s claim "CR-2144 Appendix A" | **Appendix A §5a does not define Zq.** A-16 lists X*u, Xu, Xw, Xδe, Z*u, Zu, Zw, Zẇ, Zδe, M*u; A-17 lists Mu, Mw, Mẇ, Mα, Mα̇, Mq, Mδe, Tu. Zq appears only in the mnemonic table (A-9, "Zq, 1/sec") and inside Appendix C's equation matrix (C-1, term `(−Zq − U0)s`), which shows Zq is a *velocity* (ft/s per rad), contradicting A-9's "1/sec". | ❌ **the ledger cites a relation Appendix A does not state.** The relation is inferred by analogy with `Mq`. It is *corroborated* — CLq comes out 5.945 beside the report's own tabulated CLq = 5.65 (Table IX-1, landing) and 5.4 (Table IX-2, approach) — but that is corroboration, not a citation. |
| `CNq = CLq cos α₀` (Appendix B B-1) | The code does not apply the `/cos α₀`. | ⚠ 0.32% at α₀ = 4.6°. |

---

## 4. Circularity

| Item | Where | Assessment |
|---|---|---|
| **747 cruise `CD0` and `e`** | aircraft.py:290-291 | **`circular` — and the repo says so itself.** Both are back-solved from a single trim point (`CD_trim` = 0.043 read off Figure IX-6, plus `Xw` from Table IX-4). `flightsim/tests/test_drag_polar.py`'s own docstring, line 3-5: *"CD0 and e were both back-solved from ONE (M, CD) reading at trim (M 0.80), so the model reproducing that point is circular — it proves nothing about the polar's shape anywhere else."* The escape it builds — digitising nine more points off the same 40,000 ft curve — is a **partial** escape only: it is the same hand-drawn 1972 figure, the same log axis, the same digitiser. Independent of the *fit*, not of the *source*. Its digitised value at M 0.80 is 0.0439, against the 0.043 in `aircraft.py`; my own independent pixel read of the curve is 0.042–0.044. |
| **`airframe.calibrated_lift_slope` vs the strip test** | airframe.py:212 ↔ `test_wind.py:744` | **`circular` by construction.** `a0 ≡ −8·Clp`; `test_a_rigid_roll_rate_through_the_strip_integral_returns_the_sourced_Clp` then asserts the strip integral returns `Clp` to `rel=1e-3`. Since the elliptic closed form *is* `Clp_hat = −a0/8`, the only thing this can fail on is trapezoid quadrature error at 9 stations — it cannot falsify the calibration or the loading shape. The test's docstring is candid ("asserted end to end through the real integral rather than through the closed form it was derived from"), and `test_airframe.py:177` is more openly tautological (`a0 == pytest.approx(-8.0*Clp, rel=1e-12)`). **The two genuinely non-circular tests in this area are `test_the_strip_integral_reproduces_stengels_closed_form_for_a_rectangular_wing` (an independent closed form, Stengel eq. 3.4-40) and `test_the_elliptic_closed_form_used_for_calibration_is_correct`.** The `provenance.LEDGER` classification CALIBRATED is correct and the entry names its fit target, so this is disclosed, not hidden. |
| **Cherokee `CD0`, `e`** | aircraft.py:552-554 | Same back-solve structure as the 747 (from `Xw` and a total `CD_trim`). The comment offers a corroboration — "CD0 = 0.0343, inside the 0.03–0.04 band the source predicts independently" — which would be a genuine escape *if the source could be read*. It cannot (§5), so this is `unverifiable`, and the escape claim is untested. |
| **Cessna 172 `e = 0.97`** | aircraft.py:705-706 | Least-squares fit of `CD` against `CL²` over the source's own table, then not compared to anything. Not circular (nothing is claimed to be validated by it), but not verified either. The comment is honest that 0.97 is "optimistic for a strut-braced high-wing aeroplane". |
| **747 approach `CD0`, `e`** | aircraft.py:434-436 | **Not circular.** Two equations in two unknowns, both inputs (`CL`, `CD`, `CLα`, `CDα`) tabulated in Table IX-2. No chart reading. The code says as much. |
| **Caughey as a validation reference** | validation.py:176-186 | **Correctly flagged as non-independent, and I confirmed why.** Caughey's Eq. (5.48)–(5.50) are attributed to his reference **[2]**, and his Bibliography (PDF p.109) reads: *"[2] R. K. Heffley & W. F. Jewell, Aircraft Handling Qualities Data, NASA CR-2144, December 1972."* Same data, different code. |
| **`_IX5` reference routed via PROJECT.md** | validation.py:185 | Not circular in value (the numbers are right, §3c), but it is a self-citation in form: the `Reference.source` field points at the project's own document instead of the table. |

---

## 5. Non-CR-2144 sources — obtainability

| Source | Cited for | Obtained? | Verdict |
|---|---|---|---|
| **Oseguera & Bowles 1988, NASA TM-100632** | microburst Eqs. (5)–(6) and the four constants (wind.py:392-411) | ✅ `refs/NASA-TM-100632-Oseguera-Bowles-1988.pdf` (NTRS 19880018674) | **all six items `verified`** (§2f). *Why valid:* NASA Technical Memorandum, primary — the model is the authors' own, derived from TASS numerical simulations built on JAWS field data (stated in the paper's own §"Approach"). *Upstream:* independent of CR-2144 entirely. |
| **Proctor, Hinton & Bowles 2000, paper 7.7** | F-factor Eqs. (3), (4), (7); 1 km; 0.1/0.13; (Tr−D)/W; 0.15; 500 m | ✅ `refs/Proctor-Hinton-Bowles-2000-Windshear-Hazard-Index.pdf` (NTRS 20000116199) | **every claim `verified`** (§2d). *Why valid:* AMS conference preprint by the index's own originator (Bowles) and the NASA Langley team that flight-tested it; pp. 482–487 as cited. *Upstream:* the index itself is Bowles 1990a/1990b, which this paper cites and does not reproduce; the paper is explicit that it is a review of that work. Independent of CR-2144. |
| **Doyle, Jiang, Smith & Grubišić 2011, MWR 139, 3–23** | lee-wave amplitudes 6 and 12 m/s crest-to-trough; the 20–35 km tropospheric wavelength and the stratospheric caveat | ✅ `refs/Doyle-etal-2011-MWR139-T-REX-stratospheric-mountain-waves.pdf` | **all quoted material `verified` verbatim** (§2f). *Why valid:* peer-reviewed primary measurement — NSF/NCAR Gulfstream V flight-level data from T-REX, March–April 2006. *Upstream:* independent. Crossref confirms vol. 139, issue 1, pp. 3–23, Jan 2011, DOI 10.1175/2010MWR3466.1 — matching the citation in `wind.py:287-289` exactly. |
| **Caughey, Cornell M&AE 5070 notes, Ch. 5** | `validation.REFERENCES` and `validation.CAUGHEY_A` | ✅ `refs/Caughey-MAE5070-Aircraft-Stability-and-Control-2011.pdf` | **`verified`.** Eq. (5.51) gives `Mq = −0.4381`, `Zẇ = −0.0341`, `Mẇ = −0.0002` — the three the project quotes, exactly. Eq. (5.52)'s plant matrix matches `validation.CAUGHEY_A` element for element (−0.0212, 0.0466, 0.000, −32.174 / −0.2229, −0.5839, 262.472, 0.0 / 0.0001, −0.0018, −0.5015, 0.0 / 0, 0, 1, 0). Eq. (5.54)'s roots are λsp = −0.5515 ± 0.6880i, λph = −0.00178 ± 0.1339i; the project's 0.88178 / 0.62546 / 0.13391 / 0.01329 are |λ| and −Re(λ)/|λ| computed from those, reproducing to 5 s.f. Eq. (5.48) states V = 279.1 ft/s (vs Table IX-3 FC 2's VTO(FPS) = 278.) as PROJECT.md claims. *Upstream:* **NOT independent** — Bibliography [2] is CR-2144 itself. Also worth recording: Caughey's Ixz = **−2.23 × 10⁶** slug-ft² (Eq. 6.146) against Table IX-3 FC 2's **+870050**, and his Ix = 14.3e6 / Iz = 45.3e6 against 14.2e6 / 45.4e6. |
| **Parks, Wingrove, Bach & Mehta 1985, J. Aircraft 22(2) 124–129** | Rankine vortex Eqs. (3)–(6); Hannibal r0/V0/spacing; Morton r0/V0/spacing | ❌ | `unverifiable — source not available`. AIAA paywalled; Unpaywall `is_oa: false` for 10.2514/3.45095 and for the 1984 conference precursor 10.2514/6.1984-270. NTRS record 19850039692 exists with `distribution: PUBLIC` but `downloads: []`. **Bibliographic details verified via Crossref and NTRS metadata** (authors Parks E. K., Wingrove R. C., …; J. Aircraft vol. 22, Feb 1985, 124–129) — these match `wind.py:64-67` exactly. **Content unverified: 6 numbers in `PARKS_CASES` and the vortex field equations.** |
| **Wingrove & Bach 1994, J. Aircraft 31(4) 753–760** | `UPDRAFT_W0`, `UPDRAFT_SECONDS`, the 50/100 ft/s statement, Fig. 8, the +0.66/−1.58 g | ❌ | `unverifiable — source not available`. Same paywall; Unpaywall `is_oa: false` for 10.2514/3.46557; NTRS 19970012379 and 19920072706 are metadata-only. **Bibliographic details verified via Crossref**: J. Aircraft 31(4), Jul 1994, 753–760 is Wingrove & Bach, *Severe turbulence and maneuvering from airline flight records* — volume/issue/pages match `wind.py:230`. |
| **McCormick, *Aerodynamics, Aeronautics, and Flight Mechanics*** | the entire Cherokee PA-28-180 set (35 numbers) | ❌ | `unverifiable — source not available`. Copyrighted textbook, no legitimate free copy. Compounded: the stated intermediate, `aircraft_data_validated.py`, **is not in this repository**. |
| **Roskam / USAF DATCOM via PyFME** | the entire Cessna 172 set (133 numbers) | ❌ | `unverifiable — source not available`. Roskam is a copyrighted series. PyFME is MIT-licensed and on GitHub, so this one *is* in principle recoverable, but PyFME is itself a transcription, not the source, and the code's stated chain (`aircraft_data_validated.py`) is missing. |
| **Nelson / Etkin & Reid / McRuer (Navion)** | `flightsim/tests/test_navion.py` derivative set | ❌ | `unverifiable — source not available`. Copyrighted textbooks. **Note: the test file already discloses this itself** — its docstring states that no externally published mode table for this derivative set could be confirmed, and it asserts signs and ranges rather than values. That is the correct handling of an unobtainable source. |
| **ICAO Doc 7488 (ISA)** | all 7 cited `atmosphere.py` constants | ❌ | `unverifiable — source not available`. Paid ICAO publication. |
| **"Lock's fourth-power law", "Korn equation"** | `aero.py` `20.0`, `0.1`, `10` | ❌ | `unverified` — named conventions with **no author, year, or page** anywhere in the codebase. Nothing to try to fetch. |

---

## 6. Citations that do not say what the code claims

Ordered by consequence.

1. **`aircraft.py:217` — "CLde 0.364 vs 0.396".** Table IX-1 (printed p.216) gives
   `CL_δe = .356/rad`. **0.396 appears nowhere in Table IX-1 or IX-2.** This is one of four
   numbers presented as *"the check that the conversion chain is right"* (aircraft.py:219).
   The direction of the error is benign (the true agreement is 2.2%, better than the 8.1%
   claimed) but the reference number is not the source's.

2. **`docs/PROJECT.md` §2 lines 61 and 128, and `provenance.py:23` — "a constant with no
   entry fails the build".** No such enforcement exists. 346 non-trivial numeric literals,
   13 ledger entries, ~8 of which map to a literal. `test_provenance.py` validates the ledger
   against itself only. This is the single largest gap between what the project documents and
   what it does.

3. **`provenance.LEDGER["b747.CLq"]` — "CLq = −Zq·2·m·U0/(qS·c), CR-2144 Appendix A".**
   Appendix A §5a does not define `Zq` in terms of any non-dimensional coefficient. The
   relation is inferred by analogy with `Mq`; it is corroborated by the recovered value
   (5.945) sitting beside the report's own tabulated `CLq` for other configurations
   (5.65, 5.4), but "CR-2144 Appendix A" as a citation for the *relation* is not supportable.
   `LEDGER["b747.Cmq"]`'s corresponding claim **is** supportable (A-17 gives `Mq` explicitly).

4. **`aircraft.py:135-141` — "The relations are the ones in CR-2144 Appendix A".** Two of the
   six are the leading term of Appendix A's relation rather than the relation: `Cma` drops
   `Mw`'s `(2W0/U0)(Cm + (M/2)CmM)` term, and `CLa` uses `CNα − CD` where Appendix B B-1
   gives the exact `CNα = CLα cos α₀ − CL sin α₀ + CDα sin α₀ + CD cos α₀`. Effects 2% and
   0.67% respectively. Notably the *same function's* `CDa` back-solve **does** use B-1's
   exact form, so the file is internally inconsistent about which version it applies.

5. **`aircraft.py:396-397` — "For CRUISE the two agree (both 18.2e6, 970056)".** Figure IX-1
   prints `Ixz = 0.97 × 10⁶`, not 970056, and `W = 636,600 lb`, not 636,636. The *conclusion*
   (they agree, unlike the approach case) is correct; the quoted digits are Table IX-3's, not
   Figure IX-1's.

6. **`validation.py:185` — `_IX5 = "NASA CR-2144 Table IX-5 via PROJECT.md section 4"`.** The
   two values it labels (0.0673, 0.387) really are in Table IX-5 F/C 9 — but the `source`
   field, whose whole purpose per its docstring is to make an uncited number unconstructible,
   points at the project's own markdown. Separately, `docs/PROJECT.md` §4's "747 modes vs
   CR-2144" table attributes all six modes to CR-2144 without distinguishing that the four
   lateral ones are in **Table IX-9**, and that two of those four (roll τ, spiral τ) are
   reciprocals of the tabulated `1/T` values rather than tabulated quantities.

7. **`aircraft.py:495` and `:625` — "as collated in `aircraft_data_validated.py`".** That
   file does not exist in this repository. The provenance chain for 168 of the 346 constants
   (Cherokee 35 + Cessna 133) terminates in a missing artefact.

8. **`aero.py:28-30` — "Lock's fourth-power drag-rise law… anchored on the definition of drag
   divergence, dCD/dM = 0.1 at M_dd".** No document, no author, no year. Same for the Korn
   equation at `aero.py:35-47` and its `10` in the CL term. Three constants that shape the
   747's cruise drag rise, all uncited.

## 7. What I could not do

- Stengel *Flight Dynamics* 2nd ed. is in the repo but I did not open it; the equation numbers
  cited in `airframe.py` (3.4-7, 3.4-10, 3.4-12, 3.4-19), `wind.py` (3.4-39, 3.4-48 to 3.4-55)
  and `test_airframe.py` (3.4-40) are therefore `unverified` by me.
- Table IX-2's 29 values and Table IX-4's 7 values are marked verified **on the lead auditor's
  reading**, not mine. I independently re-read Table IX-3 (both columns), Table IX-8 (all 14
  rows at F/C 9), Table IX-1, Figure IX-1, Tables IX-5/6/7/9, and Appendix A pages A-9, A-10,
  A-16, A-17, A-18 and Appendix B page B-1.
- The 0.043 `CD_trim` is a graphical reading. My independent pixel reading of the same curve
  spans 0.042–0.044 depending on whether the curve endpoint or the marker centre is taken. I
  can confirm the value is defensible; I cannot confirm it to better than the code's own
  stated ±0.003.
