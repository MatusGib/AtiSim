# Physics Audit — Flight Dynamics Engine

## Objective

Establish whether the physics in this engine is correct, and whether every
assumption it rests on is either **valid** or **a documented known flaw with a
stated bound on its consequences**.

A known, bounded, honestly-recorded flaw is a **pass**. An unexamined
assumption presented as fine is a **failure**. Do not hide weaknesses to make
the report look clean — the entire value of this audit is that I can trust its
negative findings.

The standard throughout is *justification*, not test-passing.

## Autonomy

The checklists below are a floor, not a ceiling. They reflect what I already
suspect, so an audit that only covers them tells me nothing I don't know.

You are expected to:

- Explore the codebase independently and find claims, couplings, and failure
  modes I have not anticipated.
- Research beyond the supplied material where it would strengthen a finding —
  published derivative datasets, validation cases, established results for
  these aircraft.
- Challenge the framing of this prompt itself. If a phase is misordered, a
  check is meaningless, or I have asked for something that cannot establish
  what I want it to establish, say so in the report.

Where my checklist and your findings diverge, report **both directions**:
what I named that you did not find, and what you found that I did not name.

## Source hierarchy

**Theoretical backbone.** `./Flight_Dynamics_-_Second_Edition.pdf` (repo root).
Every derivation, equation of motion, linearisation, and stated physical
principle must be traceable to a numbered equation or section in this text.
Cite as `[FD2e §x.y, eq. n]`. Record the book's full bibliographic details in
`AUDIT.md`, read from the PDF itself — do not assume them.

**Primary numerical dataset.** NASA CR-2144 for the 747 cases.

**Conflict rule.** Where the textbook and the code disagree, that is a finding.
Do not reconcile it by assuming the code implements a variant the book doesn't
cover, unless you can point to where the code says so.

## External sources — every one must be bulletproof

Any source beyond the two above must carry, in `AUDIT.md`:

1. **Full identification** — author, title, edition/version, year, and the
   specific table, figure, or equation number used.
2. **Availability** — where you obtained it, and whether it is now in the repo.
   If you fetch a source from the web, save it to `refs/` so the citation is
   reproducible. If a source is not in your context, you may not cite it;
   status is `unverifiable — source not available`.
3. **Why it is valid** — primary measurement, peer-reviewed derivation,
   textbook of record, or manufacturer data. Say which, explicitly.
4. **Upstream provenance** — what does it cite? A source tracing back to
   CR-2144 is not independent confirmation of CR-2144. Caughey (Cornell MAE
   5070) is the known example: an independent *implementation*, not an
   independent dataset. Assume there are others and check each one.
5. **Why it applies here** — same aircraft, same flight condition, same
   configuration, same conventions. A derivative from a different CG or Mach
   number is not a match.

**Banned as sources:** "standard result", "commonly known", "textbook value",
"widely accepted", or any number recalled without a citable location. If you
cannot point to a page, table, or equation in a file you can actually read, the
status is `unverified`.

This is not a formatting preference. An invented citation makes the whole audit
worthless, because I then cannot tell which of the real ones to trust either.
Being unable to verify something is an acceptable, expected outcome. Fabricating
verification is not.

---

## Phase 0 — Orientation

Read the codebase, `PROJECT.md`, and `MODEL_ASSUMPTIONS` before forming any
conclusions. Produce a one-page map of the engine: modules, data flow, where
physics decisions are actually made.

## Phase 1 — Discovery (single agent, **blocking**)

Build an inventory of every physical claim the engine makes, derived from the
code itself. A claim is anything asserting how the world behaves: a derivative
value, a conservation property, an equation of motion, a coordinate convention,
a fitted constant, a stated assumption, an implicit assumption visible only in
what the code omits.

Pay attention to the last category. Missing physics is the hardest thing to
find and the most likely to matter.

Output `INVENTORY.md`. Reconcile against this checklist and report both
directions:

- Rigid-body core: quaternion state, RK4 integration, frame/axis transforms
- Aerodynamic model: stability and control derivatives, drag polar, trim solver
- Wind fields: Parks Hannibal/Morton line vortices, Wingrove updraft column,
  Doyle lee wave, Oseguera microburst; analytic gradients; `omega_gust`
- The `field_model(field)` wind contract, and the seam where wind is sampled
  once per step and held across all four RK4 stages
- Mode predictions: phugoid, short-period, dutch roll, spiral, roll subsidence
- Atmosphere model and its consistency with everything above

**Stop here and present `INVENTORY.md` to me before continuing.** Everything
downstream inherits its blind spots.

## Phase 1B — Notation reconciliation (single agent, blocking)

Before any numerical comparison, build `NOTATION.md`: a mapping across the
textbook, CR-2144, and this codebase for every symbol used in the audit.

Columns: quantity | textbook symbol and sign convention | CR-2144 symbol and
convention | code variable | conversion required.

Cover at minimum: axis systems (body / stability / wind, NED sign conventions);
alpha and beta signs; moment sign conventions; nondimensionalisation of rate
derivatives (chord vs half-chord, `c/2V` vs `b/2V`); per-radian vs per-degree;
the inertia tensor sign convention for `Ixz`; the CG reference point for moment
derivatives; reference area, span and chord.

**Any comparison made without an entry in this table is invalid.** Where two
sources genuinely conflict on a convention, record the conflict rather than
silently picking one.

## Phase 2 — Parallel evidence (sub-agents, independent)

Each agent gathers evidence and records findings. **No agent attributes causes**
— that is Phase 3's job, and doing it here produces confident wrong answers.

**A. Numerics.** Integrator order; timestep convergence by Richardson
extrapolation on full simulation output, not the integrator in isolation;
quaternion norm drift and renormalisation effects; conservation and invariance
(energy, angular momentum in the torque-free case, Galilean, rotational);
trim convergence and uniqueness. Check floating-point precision explicitly:
confirm whether `jax_enable_x64` is set, and quantify what float32 costs the
lightly-damped modes. Check whether `jit` changes results.

**B. Reference tracing.** Every number back to its source table. Any constant
without provenance in the code is a finding.

**C. Units and conventions.** Dimensional consistency throughout; everything in
`NOTATION.md` verified in the code rather than assumed; sign conventions traced
end to end. Sign errors cancel in symmetric test cases and survive test suites,
so check asymmetric cases deliberately.

**D. Wind fields.** Cross-check every analytic gradient against
`jax.jacfwd`/`jacrev` of the field at several hundred randomised points, for
every field. Assess the RK4 sampling seam: is holding wind across all four
stages defensible, or does it reduce the integrator's order? Behaviour at
vortex cores, domain edges, and superposition of multiple fields.

**E. Falsification.** Actively try to produce nonphysical behaviour. Energy
gain in still air; divergence over long horizons; trim converging at
implausible states; singularities; ground penetration; discontinuities in the
wind field; extreme but physically reachable inputs. Report what you tried and
**failed** to break, not only what broke — the failures to break are the
evidence.

**F. Blind reproduction.** **Without reading the implementation**, derive the
747 power-approach linearisation from `Flight_Dynamics_-_Second_Edition.pdf`
and CR-2144 alone. Then diff against the code. Report your derivation, the
code's values, and the difference. If you have already read the implementation,
say so and mark this check compromised rather than performing it anyway.

**G. Archaeology.** Git history of every tolerance, fitted constant, and
reference value. When did it enter? Was it ever changed to make something pass?
Was a test ever weakened rather than a bug fixed?

## Phase 3 — Attribution (single agent, serial, after all of Phase 2)

Reconcile findings across agents. Physics findings are heavily coupled: a trim
solver bug surfaces as a mode discrepancy, a units bug surfaces as a derivative
mismatch. **Look explicitly for single causes behind multiple symptoms** rather
than reporting symptoms separately.

Specific open item: phugoid and short-period damping discrepancies are
currently unattributed. Attribute them or state plainly that you could not.

Where agents disagree, resolve by evidence or record the disagreement. Never
average.

Then a **devil's-advocate pass** over the synthesis: challenge every causal
claim and strike any that is not demonstrated.

## Phase 4 — Assumptions register

Audit `MODEL_ASSUMPTIONS` against what you found, and produce
`ASSUMPTIONS_AUDIT.md`. Every assumption gets exactly one of:

- **Valid** — with the argument and citation for why, within a stated envelope.
- **Known flaw, bounded** — the assumption is wrong or incomplete, and here is
  the magnitude of the error it introduces and when it starts to matter.
- **Known flaw, unbounded** — wrong, and the consequence has not been
  quantified. State what work would bound it.
- **Undocumented** — an assumption the code makes that the register never
  declared. These are the important ones.

Assumptions must be judged against the declared scope: turbulence-encounter
analysis of a rigid fixed-wing aircraft over seconds to minutes.

---

## Rules

- Docstrings, comments, and variable names are **claims**, not evidence.
- Never modify a tolerance, test, or reference value to make something pass.
  A failure is a finding.
- Do not resolve a discrepancy by asserting a plausible cause. Demonstrate it,
  or mark it unattributed.
- Circular validation is a failure, not a pass. Flag anything checked against
  the data it was fitted from — `CD0` and `e` fitted from a single trim point
  is the known instance; look for others.
- Report coverage honestly. What you could not check matters as much as what
  you could.

## Status vocabulary

`verified` · `failed` · `unverified` · `circular` · `unverifiable — source not
available` · `known flaw, bounded` · `known flaw, unbounded`

## Outputs

1. `INVENTORY.md` — Phase 1, presented before continuing.
2. `NOTATION.md` — Phase 1B symbol mapping.
3. `AUDIT.md` — the main table: claim | code location | source (table/eqn) |
   status | notes. Plus a coverage section on what could not be checked and
   why, and a section on anything in this prompt you think is wrong.
4. `ASSUMPTIONS_AUDIT.md` — Phase 4.
5. A PDF report **generated from** `AUDIT.md`. Do not write the PDF directly.
6. Every verified check converted into a permanent regression test, so this
   audit becomes a fixture rather than a one-off document.
