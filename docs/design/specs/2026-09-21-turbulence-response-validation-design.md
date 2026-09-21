# Turbulence response validation — design

**Written 21 September 2026, before any number was measured.** Scope agreed with the
conversation that commissioned it: **five verification phases against closed-form theory**,
**three acquisition phases against newly obtained sources**, **one statistic that ties the two
halves together**.

This document is a design, not a result. **Nothing here may be quoted as evidence.** Results go
to `PROJECT.md` §4 as they are measured, one phase at a time, per `docs/DEVELOPMENT.md` rule 1.

---

## 1. What this is for

`docs/validation.md` names the project's largest remaining risk in one sentence:

> every comparison against a real encounter is between summary numbers — peaks, spacings,
> orderings — never a time history against a time history.

That is true and it is not the only hole. Two others sit beside it and neither is recorded
anywhere:

1. **The gust path has never been checked against mathematics.** `test_verification.py` checks
   the integrator against a closed-form solution and measures RK4's observed order at 3.99982.
   Nothing does the equivalent for the response to a gust. Every turbulence result in §4 is a
   comparison against a *document*; none is a comparison against an *exact answer*.

2. **`Sears`, `Küssner`, `Wagner` and `Theodorsen` appear nowhere in the tree.** Verified by
   grep across `*.py` and `*.md`, 21 September 2026: zero hits. `ASSUMPTIONS.md` **C2** is
   titled *"Aerodynamics are quasi-steady: no α̇ **or unsteady lag**"* and calls itself "the
   best-characterised assumption in the project" — but every word of its bound is about **α̇**,
   which is the lag on the aircraft's *own* motion. The lag on the **gust's** arrival is a
   different physical effect and is bounded nowhere.

**The deliverable is two things: a set of pass/fail gates against exact answers, and one
statistic — the peak factor — measured on both sides of the same question.** Bands and
orderings throughout, per rule 6.

---

## 2. The one number that ties the halves together

The **peak factor**, `Δn_max / σ_Δn`.

Why this one and not another:

- `wind.dryden_vertical_field` realises the field as a sum of sinusoids with **fixed amplitudes
  and random phases** — the Shinozuka construction. That reproduces the target PSD exactly and
  it is why `test_cat_validation.py::test_a_dryden_realisation_has_the_variance_it_claims`
  passes. It does **not** reproduce Gaussian higher-order statistics: each realisation's
  amplitude spectrum has zero variance across the ensemble.
- Every peak and exceedance claim the project makes rests on statistics the construction is not
  guaranteed to get right. §4's "Response spectra and load exceedance" and `response.exceedance`
  are both downstream of it. **This is unmeasured.**
- The peak factor is exactly where a fixed-amplitude realisation and a true Gaussian process
  differ, and it is exactly what the newly obtained TPAWS table reports for 51 real encounters.

So one statistic falsifies the realisation method **and** compares the model to the atmosphere.
Phases V5 and S2 measure the two sides; they are sealed as a prediction before either runs.

---

## 3. Phase V — verification against closed-form theory

No source acquisition. Every gate is an exact answer or a signed, predicted discrepancy.

| Phase | Work | Gate |
|---|---|---|
| **V1** | Derive `H(ω)` from vertical gust to `n_z` in closed form from `boeing747`'s own derivative set. Drive the nonlinear sim with single-frequency gusts at ~12 frequencies spanning 0.01–1 Hz; extract amplitude and phase by fit | **both** amplitude and phase within **0.5%** of `\|H\|` and `∠H`. Phase is not optional — an amplitude-only match passes with a sign error in `Cmq` |
| **V2** | PSD identity: ensemble `σ_nz` from the Dryden Monte Carlo against `Ā·σ_w`, with `Ā² = ∫\|H\|²Φ_w dΩ` computed from V1's `H` and `wind.dryden_spectrum` | agreement inside the ensemble's own standard error, **with N stated**. Catches the one-sided/two-sided factor of 2, the log-grid Riemann truncation and ensemble convergence in one test |
| **V3** | Pratt–Walker: 1-cosine gust, sweep gradient distance, compare `Δn` against `K_g U_de V a ρ S / 2W` | the model must **undershoot** Pratt–Walker by approximately V4's Sears factor. A *signed, predicted* discrepancy, not agreement |
| **V4** | Bound the Küssner gust lag. Split C2 into its α̇ half (bounded, keep) and its gust half (new row, currently unbounded) | a number in `ASSUMPTIONS.md` at the two frequencies that matter, with the phase lag beside it |
| **V5** | Dryden realisation audit: (a) realised rotational gust spectra against MIL-HDBK-1797's analytic `Φ_p`, `Φ_q`; (b) the peak factor of the realisation, against a **Gaussian control realisation of the same PSD** — random amplitudes as well as random phases | peak factor quoted **with its sampling error and its window definition**. The control is a realisation, not the textbook `√(2 ln νT)`, which does not apply at the window lengths involved (§4, S2). (a) is expected to fail — see §6 |

### V1 and V3's arithmetic, pre-computed

Computed 21 September 2026 from `aircraft._boeing_747()` at M 0.80 / 37,000 ft, ISA. Recorded
here so the phase has something to reproduce rather than to discover:

| Quantity | Value |
|---|---|
| ρ, a, V | 0.3494 kg/m³, 295.07 m/s, 236.06 m/s |
| Pratt–Walker mass ratio μ = 2W/(ρ g c̄ a S) | **78.597** |
| Gust alleviation factor K_g = 0.88μ/(5.3+μ) | **0.8244** |
| Sears at the short period (0.164 Hz, k = 0.0182) | \|S\| = **0.9690**, −4.15° → **3.10%** attenuation |
| Sears at the Parks core passage (V/r₀, k = 0.0228) | \|S\| = **0.9610**, −4.87° → **3.90%** attenuation |

**The 3.90% is the number that matters.** §5.1 carries **4.4%** as the point-gust cost and calls
it the model's largest self-approximation. The unsteady gust lag is the same order and is
currently unbounded. It acts to **reduce** simulated load, so it **widens** the headline
shortfall rather than explaining it — which is the direction that says it is not being raised
for convenience.

---

## 4. Phase S — the newly obtained sources

### S0 — TPAWS ingest

**NASA/TM-2012-217337**, Hamilton, Proctor & Ahmad, *Flight Tests of the Turbulence Prediction
and Warning System*, NTRS 20120003172. NASA work, "Public Use Permitted".

Its **Table 1 extracts from the PDF text layer**; no digitisation, no reading uncertainty, no
calibration step. 51 rows parsed 21 September 2026:

| Column | Range across 51 rows |
|---|---|
| Altitude | 15 – 35 kft |
| **Aircraft weight** | 167.7 – 192.0 klb |
| TAS | 185 – 238 m/s |
| σ_Δn (5 s window) | 0.20 – 0.49 g |
| Δn_max / Δn_min | +0.37…+1.30 / −0.34…−1.40 g |
| Peak vertical wind | +2.8…+23.4 / −2.1…−21.4 m/s |

**Three rows are admissible and forty-eight are not**, and the distinction is §6's band rule
applied to this project's own aircraft rather than to HICAT:

| | Rows | Verdict |
|---|---|---|
| ≥ 35 kft — inside `boeing747.valid_altitude` | **3** — `232-06`, `232-08`, `232-10` at 35 kft, M 0.775–0.779 | admissible; Mach is inside `valid_mach` too |
| 30–34 kft | 5 | **outside the declared floor.** `checks.recovery_band` would refuse the run |
| < 30 kft | 43 | outside, and far outside |

So the table's value is **three in-band encounters, not fifty-one**. That is still three more
than the project has ever had — the headline is N = 1 — and the aircraft is a B-757 with **the
weight tabulated for every row**, which is the exact quantity whose absence makes 5.19
structurally impossible for Hannibal.

The other forty-eight are not waste: they set the **peak factor's population statistics** (S2),
which is a property of the response process rather than of a flight condition, and they are
quoted for that and for nothing else.

Deliverable: the table as a data file under `atisim/data/`, with `provenance.py` marking every
column **sourced**, and a test asserting the row count and the altitude range so a re-extraction
cannot drift.

### S1 — Stewart ingest, and the §5.3 verdict

**NASA/TM-2003-212666**, Stewart, *Description of a Normal-Force In-Situ Turbulence Algorithm
for Airplanes*, NTRS 20040021314. NASA work.

A severe encounter on the same B-757 at **33,000 ft, M 0.78**, with the input and the response
both plotted: Fig. 3 time histories, Fig. 4 estimated vertical wind in the time domain, Fig. 5 a
**frequency-domain comparison**. Accelerometer at 50 sps through a 20 Hz anti-alias filter.

`PROJECT.md` §5.3 currently reads:

> **IMPOSSIBLE WITH SOURCES HELD** (an in-band published curve) — HICAT is 45–70 kft against
> this project's 33–41 kft

**This does not overturn that verdict, and an earlier draft of this design said it did.**
33,000 ft is **2,000 ft below `boeing747.valid_altitude`'s floor** of 35,000 ft
(`aircraft.py:780`), so `checks.recovery_band` would refuse the run. The band objection that
disqualified HICAT applies here too, by 2,000 ft instead of by 4,000 — smaller, and the same
kind.

What Stewart is worth is therefore narrower and should be claimed narrowly: it is a **published
input-and-response pair at near-cruise on a transport**, with a frequency-domain comparison, at
50 sps through a 20 Hz filter. It bears on V1 and V2's *shape* — does a real transport's
response spectrum peak where its own short period is — without licensing a level comparison.

**§5.3's own numbers do not match the code, and that is a separate finding.** The row says
"this project's 33–41 kft". `boeing747` declares **35,000–45,000 ft** (`aircraft.py:780`) and
`_boeing_747_jsbsim` declares **35,000–41,000 ft** (`aircraft.py:1816`). The "41" appears to
come from the cross-code entry and the "33" from neither. The README's "35,000–45,000 ft"
agrees with the code. **Correcting §5.3's band is a documentation fix and is in scope for this
phase; deciding what the corrected band does to the verdict is not** — §8's precedent, *"a
decision, not a measurement"*, says that is not taken incidentally. The phase measures and
proposes; the record's owner decides.

### S2 — the peak factor, both sides

Computed from S0's 51 rows, 21 September 2026, **before any model run**:

```
all 51 encounters, 15-35 kft:
  peak factor  Δn_max/σ  : mean 2.386  sd 0.506  range 1.54-4.03
  peak factor |Δn_min|/σ : mean 2.441  sd 0.440  range 1.67-3.54
  ±g asymmetry Δn_max/|Δn_min| : mean 1.021, up-heavy in 23/51

the three in-band rows (≥35 kft): 2.296, 3.037, 2.756  -> mean 2.696
```

Gate: the model's Dryden-ensemble peak factor lands inside **2.386 ± 0.506**, or is reported as
not landing, with the ensemble N beside it.

**The band is the population's, and the population is 15–35 kft.** That is the one place in
this design where an out-of-band number is used deliberately, and the justification must be
stated every time it is quoted: the peak factor is a property of the **response process** —
how a narrowband oscillator's extreme relates to its rms — not of a flight condition, so it
does not carry the altitude dependence that a *load level* does. **Three points cannot set a
band**, which is why the 51-row population sets it and the three in-band rows are reported
beside it rather than instead of it. If that argument does not survive contact with the phase,
the comparison is abandoned and the reason recorded.

**A definitional hazard that must be settled before this comparison is made, per rule 2.** TPAWS
computes σ over a **5 s window**; at the 757's ~0.3 Hz short period that is 1.5 cycles, so this
is *not* the asymptotic Gaussian peak factor and the textbook `√(2 ln νT)` form does not apply.
Whether `Δn_max` is the peak within that same window or over the whole encounter **must be read
off the paper, not assumed.** If the definition cannot be established from the document, the
comparison is abandoned and that is the result.

**The ±g asymmetry is a second, unplanned finding in the same table.** Mean `Δn_max/|Δn_min|` =
1.021 across 51 encounters — essentially symmetric. 5.18 and 5.20 attribute the model's
unreachable ±g asymmetry to the absent stall. This table does not obviously support a large
asymmetry being there to reach. That is an observation, not a conclusion, and it is recorded
here so it cannot be quietly dropped if it proves inconvenient.

---

## 5. The sources, and which may be redistributed

`Reference_papers/**/*.pdf` is gitignored, so nothing below enters history by accident. The
licences were read off each document's own printed statement, 21 September 2026, per the
session-32 policy in `Reference_papers/SOURCES.md`.

| Document | Licence, as printed | Redistributable | Serves |
|---|---|---|---|
| NASA/TM-2012-217337 (TPAWS) | NASA, "Work of the US Gov. Public Use Permitted" | **yes** | S0, S2 |
| NASA/TM-2003-212666 (Stewart) | NASA work | **yes** | S1 |
| NASA CR-178736 / CR-178288 (B-57B gust gradient) | NASA contractor reports | **yes** | V5(a) only — see §6 |
| Wu, Cao & Ismail 2019, *Gust loads on aircraft*, Aeronautical J. 123(1266), doi:10.1017/aer.2019.48 | **CC BY 4.0, Open Access** | **yes** | V2, V3 grounding |
| Kopeć et al. 2015, AMTD 8, 11817, doi:10.5194/amtd-8-11817-2015 | **CC BY 3.0** | **yes** | CAT/EDR context |
| Sharman & Pearson 2017, JAMC 56(2), doi:10.1175/JAMC-D-16-0205.1 | © American Meteorological Society | **no** — cite only | EDR thresholds |
| Epps & Roesler 2018, AIAA J., doi:10.2514/1.J056399 | AIAA | **no** — cite only | V4 |
| Kayran **2006**, J. Aircraft **43**(5), Küssner correction | AIAA | **no** — cite only | V4 |

**Two corrections to the uploaded set, flagged rather than silently accepted:**

1. **Kayran is 2006, not 2012.** The file is named `kayran-2012-…` but the paper's own header
   reads *Journal of Aircraft, Vol. 43, No. 5, September–October 2006*. This project cites by
   year; the filename would have propagated a wrong one.
2. **Kopeć et al. is the AMTD *Discussions* preprint, not the final AMT paper.** Its own front
   matter says so: *"This discussion paper is/has been under review … Please refer to the
   corresponding final paper in AMT if available."* Rule 2 says flag, never invent: before
   anything is read from it for a number, the final AMT version must be checked for.

---

## 6. What is excluded, and why

- **The B-57B gust gradient flights are at 2,500–10,000 ft**, and CR-178288 says in terms that
  the turbulence there is neither isotropic nor homogeneous. They are the only direct
  measurement of **spanwise** gust structure found — wingtip and nose probes, two-point
  correlations and cross-spectra against von Kármán — so they serve V5(a)'s question about the
  *form* of `Φ_p`. **They may not be used for a cruise response level.** That is the band
  mismatch that disqualified HICAT, and repeating it knowingly would be worse than the original.

- **V5(a) is expected to fail, and the failure is the result.** `dryden_vertical_field` varies
  along north only, so the field has no spanwise decorrelation and the realised `Φ_p` cannot
  match MIL-HDBK-1797's. E10 already records that no field varies across the span. The phase
  measures *by how much*, which E10 does not.

- **No lateral turbulence validation.** §6(h) — the strip path double-counts the gust's rolling
  moment — is open and touches every lateral number. Nothing here depends on it and nothing here
  fixes it.

- **No new aircraft.** JSBSim ships 61 aircraft including an MD-11, the DC-10's closest
  relative. Its model header reads *"author: Unknown … textbooks, and guesses … educational and
  entertainment purposes only."* That makes it a **cross-code comparator, never a source**, on
  the precedent already set at `aircraft.py:1660` for `_boeing_747_jsbsim`. It would not settle
  `dc10_does_not_close_the_hannibal_gap`, which needs a published derivative set. Out of scope.

- **No EDR conversion.** Sharman & Pearson gives the thresholds and Kopeć gives the CAT context,
  but converting the model's response to EDR needs an aircraft response factor whose derivation
  is in a paper not held. Recorded as future work, not attempted.

- **No tolerance in `PROJECT.md` §4's "validated baseline — do not touch these tolerances" is
  touched**, and no existing result is re-pinned. Rule 3. The one documentation change in scope
  is §5.3's stated altitude band, which matches neither aircraft's declared `valid_altitude`
  (S1) — that is a transcription fix to a §5 status row, not a tolerance and not a result.

---

## 7. Sealed predictions

Two are sealed in `atisim/predictions.py` **before V5 and S2 run**, per rule 1 and
`predictions.py`'s own rule 3 — each names the observation that would falsify it:

1. **The Shinozuka realisation's peak factor is biased low against a true Gaussian process of
   the same PSD.** Falsified if the measured peak factor is at or above the control
   realisation's, over an ensemble large enough for the difference to clear its own sampling
   error — with that N stated.
2. **The model's peak factor lands below TPAWS' 2.386 ± 0.506.** Falsified if it lands inside
   or above.

They are separate because they can disagree: the realisation could be biased low and the model
still land inside the band, and what that would mean is worth being unable to fudge.

---

## 8. Build order

**V1 → V2 → V4 → V3 → V5 → S2**, with **S0 and S1's ingest pulled to the front** — they are data
and provenance work with no modelling in them, and having the real table in the tree early keeps
the comparison honest about what it is aiming at.

V4 precedes V3 because V3's gate *is* V4's number: without the Sears factor, V3 has only
"approximately agrees" to offer, which rule 6 does not accept.

**V1 is the falsification step.** It is the only phase with an exact answer and no source in it
at all. If the nonlinear sim's gust response does not match its own linearisation's transfer
function in amplitude *and* phase, nothing downstream may be believed, and that result goes to
§4 whether it passes or fails.
