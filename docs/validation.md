# What AtiSim may be used for

The claim below and the status table after it are **included verbatim from {doc}`the project record <PROJECT>`**
so they cannot drift from it. The summary between them is the
only prose written for this page, and every number in it names the section of {doc}`the project record <PROJECT>` it
comes from.

## The validation claim

```{include} PROJECT.md
:start-after: "deliberately narrower than what the evidence might be stretched to support."
:end-before: "## 2. Architecture"
```

## The verdict

**It validates the engine. It does not validate an absolute load.** That distinction is the
whole answer.

**On solid footing:**

- **The numerical core**, checked against exact mathematics rather than documents: RK4 observed
  order 3.99982 on a problem with a closed-form solution, and angular-momentum drift of 5.7e-13
  over 600 s (§4).
- **The chain from derivatives to behaviour**, verified against three independent
  implementations: CR-2144's own tables, JSBSim, and Yoshimura et al.'s source code, whose
  eigenvalues it reproduces to 0.01% on the short period and 0.20% on the phugoid (§4).
- **The Boeing 747's declared speed derivatives.** They come from a hand digitisation of
  CR-2144, and an independent automated trace of the same pages, scored against Table IX-4,
  supports them rather than weakening them (§4, "Two readings of CR-2144").
- **Comparative and mechanistic turbulence results** — which encounter is worse, how the
  response scales, and why — including TM-102186's counter-intuitive load ordering, reproduced
  six for six without being tuned to (§4).
- **Predictive discipline.** Claims are sealed before the run that decides them
  (`atisim/predictions.py`). Three have been settled: two right and one wrong, and the wrong one
  is recorded as wrong rather than reworded.
- **The gust path against exact answers**, added session 33 and the first comparison in the
  project that is against mathematics rather than a document. The flown response to a gust
  matches the aircraft's own linearised transfer function to **0.0043% in amplitude and 0.0017°
  in phase** over two decades of frequency, and the random-process identity
  `σ_nz² = ∫|H|²Φ dΩ` holds to **0.081%, 0.24 standard errors** at N = 24 (§4, phases V1–V2).
  It verifies the **path** — trim, the integrator, the three gust channels, the measurement —
  and not the aerodynamic data, which it shares.

**Measured, and bounded rather than fixed:**

- The phugoid is within **+1.69%** in frequency and **+2.83%** in damping of CR-2144 Table IX-5,
  once the report's speed derivatives and a thrust line are declared (§4, session 30).
- Freezing the derivatives at one Mach number costs **−5.04%** of the headline load across the
  Mach range that encounter actually flies (§4, session 29). The angle-of-attack axis of the
  same assumption is not yet bounded.
- The headline itself carries a band, stated in the claim above.

**Not validated — said plainly:**

- **An absolute load.** Structurally blocked: the aircraft that flew the recorded encounter was
  a DC-10-10 at an unrecorded weight, and no DC-10 derivative set is held (the status table
  below, 5.19).
- **The lateral response.** The capability exists, but no source held records a lateral
  turbulence response to compare against — and the strip load path counts the gust's rolling
  moment twice, an open bug that touches no longitudinal result (§6(h)).
- **Anything past the linear aerodynamic range.** The lift model has no stall, so it will fly to
  large angles of attack and report plausible nonsense (5.18 and 5.20 below).

**In one line:** *AtiSim is a validated comparative and mechanistic tool for longitudinal gust
response, whose numerical core and derivative chain are verified against three independent
implementations, and which cannot predict an absolute load.*

**The largest remaining risk:** every comparison against a real encounter is between summary
numbers — peaks, spacings, orderings — never a time history against a time history. The
recorded load trace is digitised, which makes that comparison possible; it has not been made.

**Two more, named session 33 and neither previously recorded.** The model has **no unsteady lag
on the gust's arrival** — no Sears attenuation, no Küssner build-up — which is worth **3–4%** of
the lift at the frequencies this project forces and **6.78%** of σ_nz across the Dryden band,
and which makes the simulated load *smaller* rather than larger ({doc}`ASSUMPTIONS` C12). And
the Dryden field is realised by a **fixed-amplitude sum of sinusoids**, which reproduces the
target spectrum exactly and is not a Gaussian process — every peak and exceedance figure the
project quotes sits downstream of that (§4, phase V5).

## Status of every known gap

```{include} PROJECT.md
:start-after: "### Status of every open item, at the release — session 32"
:end-before: "**Attributed — understood, documented, not bugs:**"
```
