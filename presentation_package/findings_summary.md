# How valid is the engine at predicting real aircraft response in CAT encounters — and how do we know?

*Plain-language summary. Every number here traces to a run made this session or to a
literature value already cited in the repository. Nothing is estimated.*

---

## What "validity" means for this engine, and how it is tested

This is a flight simulator built to study **clear-air turbulence**: the invisible, violent
air that injures passengers without warning. It is not a stochastic turbulence model — it
flies aeroplanes through **specific, real, reconstructed turbulence events**, identified by
NASA investigators from the flight-data recorders of airliners that actually hit them.

"Validity" is tested in **four tiers, each answering a different question**, and it is
important not to confuse them:

1. **Is the arithmetic right?** Does the numerical engine conserve the things physics says
   must be conserved? This needs no aircraft data at all and is the strongest evidence in
   the project, because it can be checked against exact mathematics.
2. **Does the aeroplane match its own data sheet?** The 747's aerodynamic numbers come from
   a 1972 NASA report. Fed those numbers, does the simulator reproduce the wobbles and
   rolls that same report predicts? This checks the plumbing between data and behaviour.
3. **Does an independent code agree?** The same aeroplane and the same gust are flown
   through **JSBSim**, a separate, mature, open-source flight-dynamics engine. If two
   independent implementations disagree, one of them has a bug.
4. **Does it match what really happened?** The hardest tier. Fly the reconstructed
   turbulence and compare against what the recorder on the real aircraft measured.

**The crucial honesty point, and it should lead the talk:** the project's own standing
claim is deliberately narrow. It states that the engine is validated as a **comparative and
mechanistic** tool — it can tell you *which encounter is worse, how response scales with
airspeed, and why* — and that it is **not a validated absolute-load predictor**. It cannot
tell you "the load will be 1.7 g." That limit is not a shortfall discovered by this review;
it is the project's own published position, and the evidence supports it.

**Baseline check:** the full test suite was re-run from scratch this session —
**811 passed, 1 skipped, 12 minutes 20 seconds, zero failures**, matching the project's
recorded figure exactly. The single skip is benign (one aircraft deliberately excluded from
a shared test and covered by its own dedicated one). The evidence below is current, not
inherited.

---

## Target 1 — The numerical engine · **VALIDATED**

**What was tested.** Whether the mathematical machinery that steps the aeroplane forward in
time actually obeys the conservation laws it must, and whether it is as accurate as
advertised.

**What was found.** It is essentially exact. Set a rigid body tumbling in space with no
forces on it and run it for 600 seconds — 60,000 steps — and its spin energy drifts by
**0.00000000009%**, its angular momentum by about the same, and its orientation stays
mathematically valid to the last digit the computer can represent. The integration method
is advertised as fourth-order accurate; measured, it comes out **3.9998**.

**Why this tier is the strongest.** These checks need no aircraft data and no 1972 report.
They are compared against exact mathematics, so there is nothing to argue about. Everything
else in the project rests on this, and it holds.

*Figures: none needed — the numbers are the result. Evidence: `_evidence/invariants.log`.*

---

## Target 2 — Lateral (roll and yaw) behaviour vs NASA data · **PARTIALLY VALIDATED**

**What was tested.** Feed the simulator NASA CR-2144's published 747 numbers and check the
side-to-side motions it predicts — the Dutch roll wobble, how fast a roll settles, and the
slow spiral — against that same report's own answers.

**What was found.** Very close agreement: **Dutch roll frequency within 0.4%**, roll
settling time within 0.9%, spiral within 0.8%, damping within 3.4%. All four pass tolerances
the tests assert directly against NASA's tables.

**Why this is only "partially" validated, and this is the important part.** These are
*eigenvalues* — a description of how the aeroplane wobbles when nudged. They are checked
against **the same document the input numbers came from**, so what they really prove is
that the data was transcribed and converted correctly. That is worth having, but it is a
statement about the paperwork, not about the aeroplane in turbulence.

**Critically: the simulator's lateral response to a gust has never been validated against
anything.** Until recently every wind field in the project varied only along the flight
path, so nothing ever *rolled* the aircraft. That capability now exists — and building it
found a real simplification, that about **half the sideways gust component had been silently
discarded** — but no source held anywhere in the project contains a recorded sideways CAT
response to check it against. **Treat every lateral number as a demonstration, never as
evidence.**

*Figure: `fig-A-747-modes-vs-NASA-CR-2144.png` (lower blue group).*

---

## Target 3 — The pitch-oscillation gap, explained · **VALIDATED as an attribution**

**What was tested.** The simulator's two up-and-down oscillation modes — the fast
"short period" bob and the slow "phugoid" swoop — do **not** match NASA's published values.
The question is whether that is a bug or an understood consequence.

**What was found.** The gap is real and it is large in one place: the slow phugoid comes out
**17.8% low**. The fast short period — which is the one that matters for turbulence — is
only **1.4% off**.

**And the gap is fully explained.** The simulator deliberately models the aeroplane with a
simplified aerodynamic form that omits several of NASA's terms. Build a separate textbook
calculation using **all** of NASA's terms, and it reproduces NASA's published answers to
**better than 1.2% on all four quantities**. So the missing terms are exactly the gap. The
diagnosis is confirmed by the gap being condition-dependent: the same code is 17.8% off at
cruise and **0.4% off at low speed and sea level**, where those omitted compressibility
effects vanish.

**What must not be overstated.** The textbook model **is not the simulator** — it exists
only to explain the gap. The simulator itself is still 17.8% off on the phugoid. What has
been earned is the right to say *"we know exactly why, and it is not a defect in what
remains."*

*Figure: `fig-B-phugoid-gap-attribution.png`.*

---

## Target 4 — Drag, including the transonic rise · **PARTIALLY VALIDATED**

**What was tested.** How much drag the 747 produces across its speed range, against a curve
in the 1972 NASA report, including the sharp drag rise near the speed of sound.

**What was found.** Near the cruise point the model is excellent: across five speeds from
M 0.78 to M 0.88 the worst disagreement is **0.0011**, against a tolerance of 0.004. Away
from that band it degrades in an understood way — below M 0.75 it **under-predicts drag by
up to 0.0144** (about 19%), and above M 0.88 it over-predicts.

**Two honest caveats.** First, the agreement at exactly M 0.80 is **circular**: two of the
model's drag constants were back-solved from that single point, so matching there proves
nothing. The five *other* points in the band are the real test, and they pass well. Second,
both tolerances were **chosen by the project**, not set by any source.

The transonic drag term is confirmed **live** at the condition all the turbulence work is
done at (747 at M 0.800), and correctly **inert** for the light aircraft.

*Figure: `fig-C-747-drag-polar-vs-CR-2144-IX6.png`.*

---

## Target 5 — The Navion independent check · **PARTIALLY VALIDATED — weaker than it sounds**

**What was tested.** A completely different aeroplane — a small 1940s Navion — built from
different textbooks, to check the machinery works on something other than the 747.

**What was found.** All four tests pass. **But they check signs and plausible ranges, not
published numbers**, and the test file says so itself: no externally published table for
this derivative set could be confirmed. The project's own audit grades the Navion source
`unverifiable — source not available` — the textbooks are copyrighted and absent — and
notes it is cited via a file that **does not exist in this repository**.

**What it actually buys, which is real but narrow.** The Navion's data is in a form that
**bypasses the entire conversion chain** the 747 goes through. So if that chain had a
systematic bug, the Navion would expose it. It is a **plumbing check on a second aeroplane**,
not independent numerical confirmation. The scorecard leaves its predicted/reference cells
**blank**, because there is no published value to compare against.

---

## Target 6 — The real turbulence encounters · **PARTIALLY VALIDATED — and this is the headline**

**What was tested.** In 1981 a DC-10 flew into severe clear-air turbulence near Hannibal,
Missouri, and people were hurt. NASA investigators reconstructed the invisible wind from the
flight recorder as a row of five vortices. The simulator flies a 747 through that
reconstructed wind and is compared against what the recorder measured.

**What was found — three results, in increasing order of what they are worth.**

**1. The load lands inside the recorded band, at about two thirds of it.** The recorder
measured −1.0 to +1.7 g. The model gives **−0.398 to +1.441 g** — inside the band, reaching
**68%** of its peak-to-peak. Not a match; a bracket.

**2. An independent flight-dynamics code falls short too — and this is what makes the
shortfall interpretable.** JSBSim, given the identical wind, the identical aeroplane and the
identical starting state, reaches **74.6%** of the record where this engine reaches
**67.1%** on a deliberately handicapped like-for-like setting (its own full configuration
reaches 74.1%, within half a point of JSBSim). **Neither engine reaches the record.** That
*excludes the simulator's solver as the explanation*, by measurement rather than by
argument. On the three other vortex cases the two codes agree within 3%.

**3. The model reproduces the counter-intuitive result, and this is the strongest evidence
in the project.** NASA's TM-102186 flew six aircraft types through such an encounter and
found something surprising: **the aircraft that pitches *least* takes the *most* load.** A
light aircraft has time to pitch and ride the gust; an airliner is past it before it can
respond, and eats the whole thing. Run six aircraft through the same field and the model
reproduces **both** orderings — including the reversal — and the mechanism NASA states in
words comes out **monotonic, six times out of six, with no inversions**.

That third result is the one to lead with, because the model was never tuned to produce it.
It falls out of the physics.

**Also worth reporting:** the encounter grades **severe** on the standard severity index
(0.6394 g against a 0.3 g threshold), and the model's gust peak lands on the reconstructed
vortex strength **exactly**, which confirms the wind field is being evaluated correctly.

**What is blocked.** Predicting the *absolute* load is **structurally out of reach**. The
recorded aircraft were DC-10s; the model flies a 747. And the model has no stall, so it
physically cannot reproduce the up/down load asymmetry that both source papers attribute to
stall buffet.

*Figures: `fig-D` (the encounter), `fig-E` (the six-aircraft ordering and mechanism),
`fig-G` (severity), `fig-H` (the independent-engine comparison).*

---

## One result that is different in kind: a sealed prediction that came true

Everything above is **retrodiction** — the paper was open beside the model when the
comparison was made. That is the weaker kind of evidence, and no amount of it becomes the
stronger kind.

The project keeps a **sealed prediction register**: claims written down, hashed, and
committed to git *before* the answer was available, with the observation that would falsify
each one stated up front. Three exist. **Two are now settled and both were right.**

1. **The resonance bet.** The model predicted that an aircraft flown through random
   turbulence would show its load peaking at its **own natural frequency** rather than at the
   frequency of the turbulence driving it, inside a stated band of 0.131–0.197 Hz. Re-run
   this session at **151 flights** — matching the sample count of the paper whose protocol it
   copies — both intensities land on **0.170 Hz**. (At the original 32 flights one of them
   read 0.140; that difference was under-sampling, and the result is *cleaner* than when it
   was settled.)
2. **The turbulence-intensity bet, settled this session.** Before reading the chart, the
   model predicted that MIL-F-8785C's severe-turbulence curve at 37,000 ft would exceed
   4.46 m/s. Digitised: **4.80 ± 0.12 m/s**. Right, by 0.34. The uncertainty is a band rather
   than a point because two curves are drawn as **one stroke of ink** at that altitude — and
   an independent transcription of the same chart inside JSBSim lands **0.5% away**.

**The third is still sealed, and it bets against the project's own story** — it says that
acquiring DC-10 data will *not* close the load gap.

All three digests were re-verified this session. This is the only genuinely **predictive**
evidence the project has, and it is worth a slide — including the detail that the settled
outcome was **not** rewritten when better statistics arrived.

---

## The honest verdict

**On solid footing, quotable without qualification:**

- **The numerical engine.** Conservation and accuracy are essentially exact and checked
  against mathematics, not documents.
- **The 747's lateral modes and short period against NASA's own tables**, and the drag polar
  near cruise. The data-to-behaviour plumbing works.
- **The relative and mechanistic turbulence results.** Which encounter is worse, how response
  scales with airspeed, and *why* — including the counter-intuitive load ordering, six for
  six. This is the project's real contribution.
- **Cross-code agreement with JSBSim** within 3% on three of four vortex cases.

**Provisional — true but narrower than it sounds:**

- The **phugoid** is 17.8% off. Fully explained, not fixed.
- The **Navion** check tests signs and ranges, not values, against sources the project does
  not hold.
- The **drag polar** degrades away from cruise, and its headline point is circular.
- The **frequency-domain** results are tested against the model's own dynamics, not against
  a recorded spectrum, and used 32 samples where the source paper used 151.

**Not validated at all — say so plainly:**

- **Lateral gust response.** Capability exists; evidence does not. No source held contains a
  recorded sideways CAT response.
- **Absolute load prediction.** Structurally blocked and correctly disclaimed.

**The biggest single risk to trustworthiness — and it moved during this work.** The risk was
that **every comparison against a real encounter was a comparison of *summary* numbers** —
peaks, bands, orderings — and that the one recorded trace within reach, TM-102186's Figure 6,
had never been read. **It has now been digitised** (session 27), and it changed two things:

- **The model is flying air that is too gentle.** The recorded vertical wind reaches
  **−98.3 ft/s**; the reconstructed field the model flies gives it only **−86.8** — about
  **12% weaker**. Since load follows gust in the linear range, that is a real and previously
  unattributed slice of the 32% load shortfall, and it points at **the wind**, which the
  project had stopped treating as a candidate.
- **The paper's "+1.7 g" is not the largest peak.** The trace reaches **+1.855 g** in one
  narrow spike the prose does not mention. Three denominators are now defensible (2.70,
  2.674 and 2.827 g), putting the model at 68.1%, 68.8% or 65.0%. **Quote 68% against 2.70 g**
  and say which one you mean.

**What replaces it as the top risk: the shortfall now has no surviving single explanation.**
The 32% gap was attributed to aircraft type (DC-10 vs 747) after everything else had been
excluded by measurement. Session 26 checked the *direction* of the wing-loading effect and
found it pointed the wrong way. **Session 27 went and got the number, and it does not settle
anything** — because the uncertainty was never the wing area. The DC-10 variant is
unidentified and the aircraft's weight that day is recorded nowhere, so the ratio is a
**range of 0.58× to 1.32×** that straddles 1.0. The project's assumed 1.3× is the extreme
corner. **That attribution is now withdrawn rather than confirmed**, which is a worse position
and a more honest one.

**Say this plainly if asked**, because it is the strongest version of the answer: *we have
excluded our own solver by measurement, including against an independent engine that also
falls short; we have now measured that part of the gap is the reconstructed wind being 12%
light; and we have withdrawn the explanation we could not support rather than keeping it
because it was convenient.*

---

## Scoreboard

| # | Target | Status |
|---|---|---|
| 1 | Integrator / rigid-body invariants | **Validated** |
| 2 | Lateral derivative chain vs CR-2144 | Partially validated *(modes only; gust response not validated)* |
| 3 | Phugoid / short-period gap attribution | **Validated as an attribution** *(the gap itself remains)* |
| 4 | Drag polar incl. wave drag at M 0.8 | Partially validated *(good near cruise, circular at the fit point)* |
| 5 | Navion end-to-end acceptance | Partially validated *(signs and ranges, not values)* |
| 6 | Vortex encounter vs Wingrove & Bach / TM-102186 / Parks | Partially validated *(orderings and mechanism yes; absolute load blocked)* |

**Two of six are on solid footing. Four are partially validated. None is unstarted.** The
two genuinely blocked items — lateral gust response and absolute load — are both blocked by
**missing source data**, not by defects in the engine.
