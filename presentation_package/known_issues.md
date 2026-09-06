# Known issues — open data-quality problems and structural limits

Compiled from the repository as it stands, verified against the code and against fresh
runs made this session. Each entry says what the issue **actually** is, its **current
status**, and **what resolving it would take**.

> **Three of the four issues named in the task brief were mis-stated.** The brief's
> framing has been corrected against the repository in each case, and the correction is
> shown explicitly. This matters: two of them name the wrong aircraft.

---

## Contents

**Corrections to the brief** — 1, 3, 4
**Open, actionable** — 2, 5, 6, 8
**Closed by work done in this session** — 7 (Fig. 6 digitised; ensemble re-run at 151 seeds)
**Structural limits (cannot be fixed from any source held)** — 9, 10, 11
**Documentation defects found this session** — 12, 13
**Smaller open items** — 14
**New finding: the model's wind is ~12% light** — 15

---

## 1. The Hannibal core-radius conflict — **RESOLVED in session 26, not open**

**The brief says:** an unresolved "radius conflict between Parks 1985 and TM-102186".

**What is actually true.** The conflict was real from session 22 to session 25, and it was
worse than a disagreement between two papers — the project was flying a vortex **no source
states**. `PARKS_CASES["hannibal"]` paired Wingrove Fig. 4's 500 ft radius with Parks'
85 ft/s strength and Parks' 3500 ft spacing, under a comment declaring the hybrid
deliberate. Session 26 obtained Parks 1985 itself and replaced the hybrid with that
paper's own coherent triple.

Current state, from `atisim/wind.py:191`:

```python
PARKS_CASES = {
    "hannibal": {"r0": 600.0 * FT2M, "v0": 85.0 * FT2M, "spacing": 3500.0 * FT2M},
    "morton":   {"r0": 450.0 * FT2M, "v0": 70.0 * FT2M, "spacing": 3200.0 * FT2M},
}
```

with the rule stated in capitals directly above it: *"EACH CASE IS ONE PAPER'S COHERENT
TRIPLE. DO NOT CROSS THEM."*

**The two radii are not rivals — they are two different identifications of one encounter.**

| identification | r0 | V0 | held in |
|---|---|---|---|
| Parks 1985 case 1 | 600 ft | 85 ft/s | `wind.PARKS_CASES["hannibal"]` |
| Wingrove Fig. 4 reading | 500 ft | 85 ft/s | `wind.WINGROVE_FIG4_CASES["hannibal"]` |
| Mehta 1987 five-vortex refit | 500.5 ft | 86.8 ft/s | `wind.mehta_hannibal_array` — **the headline run** |

All three describe the same event (Hannibal MO, DC-10, 37,000 ft). **The project now holds
each in its own named dictionary**, so a case can be flown against any one of them without
crossing sources (`_evidence/sweepB_parks_radius.log`). That is the resolution: not picking
a winner, but making each identification separately quotable and refusing to mix them.

The same file records Parks' own assessment of his two fits — case 1 (Hannibal)
*"reasonably good agreement"*, case 2 (Morton) *"not as good… mountain-wave contamination of
the short-period wind pattern"*. **Quote Hannibal as the primary case and Morton as
support**, and say so when the two disagree.

**What it cost, measured.** Going 500 ft → 600 ft at the same V0 is a *gentler* gradient,
so it costs load: peak-to-peak falls 1.8969 → 1.8162 g, i.e. **70.3% → 67.3%** of the
DFDR's 2.70 g (`_evidence/vortex_compare_jsbsim.log`).

**Status: closed.** Nothing to do. Do not present this as an open conflict.

---

## 2. Yoshimura et al. 2022 contradicts its own Table A5 — **open in the source; handled here**

**The issue.** The paper's *conclusion* calls the B747 short-period frequency
**"1.29 Hz"** and derives a 200 m resonant wavelength from 258 ÷ 1.29, then argues that
10–20 m grid spacing is needed to resolve CAT. But its own **Table A5 labels that quantity
`s⁻¹`**, i.e. rad/s. 1.29 rad/s = **0.205 Hz**, a factor of 2π out.

**Cross-checked independently by this project**, not merely asserted: Table A2's own
derivatives reproduce 1.29 rad/s via ω_n² = Z_α·M_q/U₀ − M_α, giving 1.286. At 258 m/s the
resonant wavelength is 2π·258/1.29 ≈ **1,257 m**, not 200 m — so the paper's
grid-resolution conclusion does not follow from its own calculation.

**Does it affect any number used here?** **No.** The project uses Tables A2/A3/A5 and
explicitly does not use that conclusion. This session's run confirms the tables are used
correctly: with Table A2's derivatives the 747 short period comes out **1.2823 s⁻¹ against
Table A5's 1.2900, −0.60%** (`_evidence/cat_validation.log`).

**Why it still matters for a presentation.** Anyone who checks the paper will read
"1.29 Hz" in its conclusion and find a number 6.28× different from the one quoted here.
Say the unit was checked and which reading was adopted.

**What resolving it would take:** nothing on this side. It is an error in a published
paper; the project already sides with the table over the conclusion and documents why.

---

## 3. Cherokee inertia — **the brief calls this "Ixx"; it is Izz**

**The brief says:** a "Cherokee Ixx discrepancy".

**What is actually true.** The flagged quantity is **Izz**, and the problem is an
*ordering* one. Measured from the registry this session (`_evidence/inertia_table.log`):

| aircraft | Ixx | Iyy | Izz | ordering |
|---|---|---|---|---|
| **cherokee** | **3100.0** | **1700.0** | **1400.0** | **Izz < Iyy < Ixx — backwards** |
| cessna172 | 1285.3 | 1824.7 | 2666.7 | Izz > Iyy > Ixx — normal |
| boeing747 | 24,675,887 | 44,877,574 | 67,384,152 | normal |
| boeing737 | 802,064 | 2,087,353 | 2,692,974 | normal |

For a conventional fixed-wing aeroplane, mass is distributed along **both** the span and
the fuselage, so yaw inertia Izz is normally the **largest** of the three. The Cherokee's
set has it as the **smallest**.

`atisim/aircraft.py:721` carries the flag in the source itself:

> `Izz = 1400 < Iyy = 1700, atypical for a conventional aeroplane.`

and records that this is **flagged by the source's own author** (McCormick, via a worked
example) and carried as given rather than silently corrected — per the project rule
against inventing plausible data. The same block notes the lateral set has **no
independent second source**.

**What it affects.** Yaw response and the Dutch roll / spiral behaviour of the Cherokee
only. The Cherokee appears in the Fig. 8 ordering comparison, where it contributes the
slow-aircraft end. Its *pitch* behaviour — which is what the incidence-gain mechanism
turns on — depends on Iyy, which is not flagged.

**What resolving it would take:** an independent PA-28-180 inertia set from a second
source. None has been found. Note the derivative conversion **is** independently checked:
the source's own non-dimensional table (CZa −4.68, CMa −0.741, CMq −7.42, CZde −0.934,
CMde −2.4) reproduces to better than 0.7%, asserted in `test_aircraft.py`.

---

## 4. Cessna 172 — **the brief's numbers belong to the Cherokee, and the physics is inverted**

**The brief says:** *"Cessna 172 Izz = 1400 kg·m² is physically implausible — pitch inertia
should exceed yaw inertia for this airframe."*

**Three things are wrong with that.**

1. **1400 kg·m² is the Cherokee's Izz, not the Cessna's.** The Cessna's Izz is **2666.7**.
2. **The Cessna's ordering is normal**, not implausible: Izz 2666.7 > Iyy 1824.7 > Ixx 1285.3.
3. **"Pitch inertia should exceed yaw inertia" is backwards.** For a fixed-wing aircraft,
   yaw inertia normally exceeds pitch inertia. It is the *Cherokee*, where Iyy > Izz, that
   violates the expected ordering.

**The Cessna IS out of scope — but for an entirely different reason.** From
`docs/PROJECT.md:212`:

> Roskam / USAF DATCOM via PyFME | Cessna 172 non-dimensional tables | **rudder derivatives
> omitted and inconsistent — the whole rudder set is zeroed**

Because the source omits them, the rudder set is zeroed rather than guessed, so **the
Cessna's turns are uncoordinated**. §10 states it plainly: *"The Cessna is out of scope…
It trims, flies and passes its tests, but no result should be quoted from it."*

**Consequence for the scorecard.** The Cessna appears in the Fig. 8 six-aircraft ordering
table. That comparison is longitudinal (pitch and normal load), so the zeroed rudder does
not contaminate it — but the row should not be quoted on its own.

**What resolving it would take:** a Cessna 172 rudder derivative set from a source that
supplies a consistent one. Low value: the Cessna is not used for turbulence work.

---

## 5. The load shortfall, and the session-26 finding that inverts its explanation

**The issue.** On the headline Mehta field the 747 reaches **68.1%** of the DC-10's
recorded 2.70 g peak-to-peak. Where does the missing third go?

**Session 23 onward excluded nearly everything, by measurement:**

| candidate | measured contribution |
|---|---|
| step size (dt 0.02 → 0.0025) | 0.14% total |
| point-sampled gust vs rates fitted across the airframe | −1.67% up, +4.40% down |
| strip-integrated loads | **exactly 0.00%** |
| identified parameter band (V0, r0) at its most favourable corner | reaches 72.7% |
| gust amplitude at any strength inside the linear range | saturated — see below |
| **the solver itself** | **excluded — JSBSim also falls short** |

Gust amplitude is *excluded rather than bounded*: tripling V0 moves the up-increment only
0.441 → 0.526 g against a recorded 0.70, and the peak first reaches +1.7 g between ×3.25
and ×3.50 of the identified V0 — while |α| leaves the 10° linear range **in the same
interval**. So there is no gust strength at which the model both reaches the record and
may be believed.

**The residual was attributed to aircraft type (DC-10 vs 747). Session 26 measured the
sign and it points the other way.** A controlled sweep gives load as ≈1/(W/S), so *lower*
wing loading means *more* gust response. The DC-10's wing loading is roughly 1.3× the
747's — so a DC-10 should respond **less**, not more. Flown: the 747 at 1.3× its wing
loading reaches **56.4%** against the baseline's 70.1% — **14 points worse, not better.**

**Session 27 fetched the wing loading. It does not settle anything — because the uncertainty
was never the wing area.** The modelled 747 is 565.15 kg/m² (SOURCED, CR-2144 Table IX-3).
Against it:

| DC-10 | at empty weight | at mid weight | at max takeoff |
|---|---|---|---|
| **−10** | 0.584× | **0.815×** | 1.046× |
| **−30** | 0.631× | **0.973×** | 1.315× |

**The project's "roughly 1.3×" is the bottom-right cell — the heavier variant at maximum
takeoff weight**, the extreme corner rather than a central estimate. At any realistic cruise
weight the ratio **straddles 1.0**, and for a −10 it sits *below*, which by the project's own
1/(W/S) law means a DC-10 would respond **more** than the 747 — the direction that helps.

**Three things block pinning it, and none is the wing area.** (1) The **variant is
unidentified** — both papers say only "a DC-10", and −10 vs −30 moves the ratio 26%. (2) The
**weight at the encounter is recorded nowhere** in any source held; empty-to-max spans a
factor of 1.8. (3) Even published areas disagree across secondary sources for the same
variant — 3,647 ft² vs 3,958 ft², an 8.5% spread. The manufacturer document was fetched
(Douglas **DAC-67803A**, *DC/MD-10 Airplane Characteristics for Airport Planning*) and gives
a **SOURCED Series-10 max design taxi weight of 458,000 lb** but **no wing area at all** —
it is a pavement-and-gate document.

**Status: the aircraft-type attribution is WITHDRAWN, not confirmed and not refuted.** The
shortfall has no surviving single explanation. That is a worse position than session 26
thought it was in, and a more honest one.

**And session 27 found a piece that does have a measured sign** — see issue 15 below: the
digitised record shows the model's **wind** is ~12% weaker than the recorded wind.

**What resolving it would take:** not another specification sheet, but **the flight record** —
an NTSB or operator document naming the aircraft and its weight that day. Failing that, a
published DC-10 longitudinal derivative set flown through the existing harness; one set
serves **both** Parks cases, since both are DC-10s.

**What NOT to do**, stated in the source: Parks Fig. 7 is the vortex array and Fig. 6 is
the g trace for the same encounter. **Do not fit W/S to reproduce Fig. 6.** That would turn
the project's only end-to-end check into a calibration and leave nothing to test.

---

## 6. Lateral response is capability, not validation

Until session 24 **every** wind field was a function of along-track distance alone, so the
model was longitudinal by construction. Three consequences compounded:
`wind.strip_roll_moment` integrated to exactly zero on every field, the lateral modes were
validated as eigenvalues and never once excited, and `vortex_viz.Encounter` carried no
roll, sideslip or rate channel — so the pipeline **could not have reported** a rolling
response if one had occurred.

Session 24 closed the capability and found a real simplification doing it: Parks' model is
2-D in the plane perpendicular to the vortex lines, and `vortex_wind` returns his
horizontal magnitude along the flight path. At Mehta's ψ = 31°, cos ψ = 0.857 lies along
the path and **sin ψ = 0.515 across it — and the across-path half, the only sideslip input
this field has ever had, was being discarded.** `wind.line_vortex_wind` reproduces
`vortex_wind` along the path to 1.2e-14 m/s and differs off it.

**Status: capability exists; validation does not.** No source held supplies a recorded
lateral CAT response — Wingrove & Bach 1994 explicitly supplies *no lateral data*.

**What resolving it would take:** a recorded encounter with lateral channels (roll rate,
sideslip, or lateral acceleration). None is held. **Treat every lateral number as a
capability demonstration, never as evidence.**

---

## 7. The frequency-domain comparison is half-open, and it is source-gated

Session 25 added the estimators and ran Yoshimura's protocol against the model's own
dynamics. What is missing is the **observed** half.

- Yoshimura's own recorded traces are **withheld by confidentiality** — the figshare
  dataset (17.9 GB, verified by md5 this project holds) contains the LES fields and their
  simulation code, but the three onboard flight records and the JAL PIREP are excluded.
- The one recorded trace within reach is **TM-102186 Figure 6**, p. 3-5 — the Hannibal DFDR
  g-load history. It is held.

**DONE, session 27 — TM-102186 Fig. 6 is digitised.** `scripts/digitise_tm102186_fig6.py`.
The difficulty was real (a dense oscillatory trace on a 1989 scan) and the method was made
checkable rather than trusted: three independent checks, including a **negative control** —
the same extraction run on the vertical-wind panel returns +1.17 ft/s through smooth
pre-encounter cruise where it must return zero, so the method carries no vertical bias. See
issue 15 for what it found.

**Also closed, session 27: the under-sampling.** The Dryden ensemble was re-run at
**151 seeds**, matching Yoshimura's own count. Both intensities now peak at **0.1700 Hz**
(+3.6% of the short period); the 0.1400 Hz that appeared at 32 seeds was noise, and the
apparent intensity-dependence is gone. **This caveat no longer applies.**

**Still open:** there is no **published load-exceedance curve** for transport-category cruise
to overlay the model's own against, and Yoshimura's own *recorded* traces remain withheld
under confidentiality.

---

## 8. Provenance coverage is far narrower than it reads

`atisim/provenance.py`'s docstring used to claim that "adding a constant without saying
where it came from fails the build". **Nothing checked that.** The audit measured it:
roughly **346 non-trivial numeric literals across nine physics modules against 13 ledger
entries — about 2% coverage.**

The gap has been partly closed and the claim narrowed to what is actually enforced: a *new*
module-level constant in one of five physics modules (`aero`, `airframe`, `atmosphere`,
`trim`, `wind`), added without a ledger entry, now fails the build
(`test_audit_regression.py`). Constants **inside functions**, in the other four physics
modules, and **all of the aircraft data in `aircraft.py`** are still not covered.

**Related, and worth naming:** the audit records that the Cherokee, Cessna and Navion data
are all cited via `aircraft_data_validated.py` — **a file that does not exist in this
repository** — and grades 168 of 346 constants `unverifiable — source not available`.

**What resolving it would take:** extending the ledger to `aircraft.py`. Mechanical but
large. The enforcement direction is right; the coverage is the work.

---

## 9. ±g asymmetry — **structurally impossible from sources held**

Both source papers attribute the asymmetry between up-gust and down-gust load to **stall
buffet**. `aero.py` is `CL = CL0 + CLα·α`, exactly odd-symmetric in Δα, so an up-gust and
an equal down-gust give equal and opposite increments **to machine precision**. The only
aircraft in the project with nonlinear data is the Cessna, which is out of scope, and
CR-2144 provides no buffet-onset table for the 747.

Session 25 split this in two. The nonlinear lift curve is genuinely unreachable —
CR-114494 draws `CL_BASIC` as straight lines annotated *"extrapolate linearly to higher α
if required"*. But the **buffet-onset boundary IS held**, on a page already read
(CR-114494 p. 2.0-38). That boundary cannot produce a falling lift curve, so it **buys a
bound, not the asymmetry**: it would replace the *declared* 10°/12° α limit with a
*sourced* ceiling in C_L–Mach.

**Do not promise the asymmetry.**

---

## 10. CR-2144's 747 derivatives are the **flexible** airframe

Section IX's derivative plots are labelled "Flexible" — they carry aeroelastic corrections
— and `dynamics.py` integrates a **rigid** body. This is a genuine model/data mismatch.

Session 23 bounded it without needing a rigid set: for a rigid aircraft the non-dimensional
derivatives depend on Mach and geometry only, so at constant Mach they must be identical at
two altitudes. Against Yoshimura's Table A2 (same 747, M 0.8, 6,096 m vs 40,000 ft, a 2.48×
change in dynamic pressure): `C_Zα` −16.6%, `C_mα` −60.9%, `C_mq` −16.7%, `C_Zq` −18.7% —
**every one less stiff at the higher q̄**, the direction aeroelastic relief predicts.

**This is a ceiling, not a measurement of flexibility.** Reynolds number moves by the same
factor and CG is not excluded; a CG shift would land almost entirely on `C_mα`, the largest
mover, so that row is the least trustworthy and **should not be quoted alone**.

---

## 11. Absolute load prediction — **ruled structurally out of reach**

Wingrove & Bach never identifies an aircraft type. Parks' two cases are DC-10s at
37–39 kft against this project's 747 at 40 kft. Half the Fig. 8 load band is unreachable
inside the linear range at all: read as an absolute load factor, −1.9 g needs ~13.8° of
elevator and drives |α| to ~18.5° — well past the 12° where the model reports lift the
sources deny.

**Assert bands and orderings, never values.** This is the project's own standing rule and
it is the correct one.

---

## 12. `PROJECT.md` §1 quotes the wrong load percentage

`docs/PROJECT.md:101` says *"the headline load comparison reaches **67%** of a recorded
peak-to-peak"*. **67.1% is the like-for-like *translational* figure** — AtiSim with its
gradient terms deliberately disabled for parity with JSBSim. The **headline** figure is
**68.1%** (§4:327, and five other places).

Seven different percentages exist, all legitimately, because the runs differ. Full
disambiguation table: `_evidence/load_percentage_disambiguation.log`. **Quote 68% for the
headline.** A minor documentation defect; no modelling error and no run artifact affected.

---

## 13. Two errors in the sources themselves, and one date

Neither is this project's, and both are worth naming because a reader may check.

- **TM-102186 quotes Schultz 1990's *initial estimates* as results.** Its microburst
  geometry ("inner ring 2500 ft diameter, core 900 ft") is Schultz's Table 1 starting
  guess. His converged Table 2 gives an outer core **62% larger**. *If microburst geometry
  is ever taken from TM-102186, take it from Schultz Table 2 instead.*
- **Yoshimura 2022's unit error** — issue 2 above.
- **The Hannibal encounter is dated two ways.** Mehta 1987 says July 1981; TM-102186 says
  April 1981 in three figure captions and Bach 1991's Table 7.1 lists case 1 as `4/81`.
  Everything else matches across the accounts, so they are the same encounter. **Two NASA
  documents say April; cite April 1981.** Nothing physical depends on it.

---

## 14. Smaller open items

- **`trim.trim` converges to physically absurd roots for degenerate coefficients.** Because
  `CL = CL0 + CLα·α` is linear, a huge α compensates a tiny CLα and Newton reaches a root
  satisfying the residual to machine precision that is not a flight condition
  (`boeing747_approach`, CLα = 1e-4 → **−632.1°** at residual 5.7e-15). Convergence and
  sense are different questions. Guarded, but worth knowing.
- **Gravity is constant at 9.80665 m/s²**, +0.383% high at 747 cruise. Measured and
  deliberately not modelled: it maps 1:1 onto the phugoid (−0.3798% for −0.3816% in g) and
  moves the lateral modes only 0.055–0.079%. **A sub-0.5% claim at altitude is unsafe for
  the phugoid and safe for the other four modes.**
- **Derivatives are frozen across the envelope**, unbounded (ASSUMPTIONS C3). Quote the
  excursion with every result.
- **There is no ground, no stall, no landing gear and no ground effect.**
- **The approach 747 sits 12.2 m/s below its minimum-drag speed** and must be flown open
  loop.
- **The Parks vortex core is 2.3–3.1 wingspans** and the gust field is sampled at a point
  with a first-order gradient correction. At that scale the correction is doing real work
  rather than tidying up, and second-order variation across the span is not represented.
  Measured cost on the headline run: ≤4.4%.
- **Two 747 lateral derivatives are exactly zero because CR-2144 omits them.** `CYp` and
  `CYr` — side force due to roll rate and to yaw rate — are carried as `0.00000000` rather
  than estimated, per the project rule that omitted derivatives stay zero and are not
  guessed (`_evidence/target2-lateral-modes-chain.log`). They are small contributors to the
  Dutch roll, which agrees to 0.4% without them, but they are a genuine hole in the lateral
  set and matter more for a rolling gust than for a mode. **Resolving it needs a 747 lateral
  set from a source that tabulates them.**

---

## 15. The model's wind is ~12% weaker than the recorded wind — **NEW, session 27**

**Found by digitising TM-102186 Figure 6**, which the project had quoted as two numbers
(−1.0 / +1.7 g) for four sessions without reading the curve between them.

| | recorded (Fig. 6, digitised) | the model's field | gap |
|---|---|---|---|
| vertical gust, down | **−98.3 ft/s** | −86.8 ft/s | **11.7% weaker** |
| vertical gust, up | **+64.8 ft/s** | +59.1 ft/s | **8.8% weaker** |

Mehta's five-vortex field is a *smooth fit* to that record, so under-shooting its extremes
is expected. **What is new is that the size of the under-shoot is now measured** rather than
assumed negligible. In the linear range load follows gust, so this is a real and previously
unattributed slice of the 32% load shortfall — and it points at **the wind**, which issue 5
had stopped naming as a candidate.

**Two further findings from the same digitisation:**

- **The paper's "+1.7 g" is the *second* peak, not the largest.** The trace reaches
  **+1.855 g** in one narrow spike the prose never mentions. The prose quotes the *sustained*
  band. Three denominators are now defensible — 2.70 g (prose band), 2.674 g (digitised
  sustained), 2.827 g (digitised absolute) — putting the model at **68.1%, 68.8% or 65.0%**.
  The evidence ledger keeps 2.70 g so the headline does not silently move.
- **The gust-spacing check got tighter and the model got worse against it.** Digitised
  spacing is **5.60 s**; the flown model gives 5.36 s. That is **−4.3%** against the figure
  where it read **+7.2%** against the prose alone. The figure is a tighter reference than the
  sentence describing it, and the sign flips.

**One thing declared rather than resolved.** The trace sits at **0.9513 g** through level
cruise where the definition says 1.000. The negative control rules out the extraction method,
so it is either a recorder bias or a registration offset of the plotted curve within its own
axes, and **the figure alone cannot separate those**. Both envelopes are reported raw, in
plot coordinates. **Do not "correct" the trace to 1.0 g** — that would be fitting the record
to the model.

**What resolving it would take:** nothing further on the figure. The open question is whether
the 12% gap is Mehta's fit under-shooting a noisy trace (expected, and quantifiable by
re-fitting) or a real deficit in the identified field. Re-fitting Mehta's five vortices to
the digitised trace would answer it — **but note the standing prohibition in issue 5: do not
fit the field to reproduce the g trace**, or the only end-to-end check becomes a calibration.
