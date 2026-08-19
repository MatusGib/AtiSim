# Remediation — fixing what the audit found

Companion to `AUDIT_PROMPT.md` and `AUDIT.md`. That audit produced 41 findings.
This prompt says **which of them to fix, which to leave alone, and why the
difference matters.**

## Objective

Fix the errors that are **certain** and whose **correct form is determined by a
source in this repository or by arithmetic**. Leave everything else exactly as
it is, and improve its documentation instead.

The audit's own standard applies unchanged: *a known, bounded, honestly-recorded
flaw is a pass; an unexamined assumption presented as fine is a failure.* It
follows that **"fix" is not always the right response to a finding.** Several
findings below are real and are still not yours to change, because the evidence
that would tell you what to change them *to* is not in this repository. Changing
them anyway converts an honest documented limitation into a fabricated
certainty, which is strictly worse than the flaw.

---

## Rules

These are inherited from the project and from the audit, and they are not
negotiable.

1. **Never modify a tolerance, test, or reference value to make something pass.**
   The audit verified this rule had held across 591 deleted test lines in all 53
   commits touching `flightsim/tests/` on all 9 branches. Do not be the first to
   break it.
2. **Never invent a constant.** If a fix requires a number, that number must come
   from a document in `refs/`, from `Flight_Dynamics_-_Second_Edition.pdf`, or
   from arithmetic on numbers that do. "Typical", "standard", and "textbook
   value" are banned, exactly as in the audit prompt.
3. **A docstring is a claim.** If you fix code and leave a docstring asserting the
   old behaviour, you have not finished. If you fix a docstring's *number*, say
   where the new number came from.
4. **Every fix carries a verification.** Not "the tests pass" — a specific
   measurement showing the defect is gone and the things that were right are
   unmoved.
5. **If a fix moves a number the project quotes, stop and report it** rather than
   updating the quoted number. Several Tier 1 items are flagged for this.

## The baseline you must preserve

Measured immediately before this prompt was written:

```
pre-existing suite (excluding test_audit_regression.py)   444 passed, 1 skipped
test_audit_regression.py                                  109 tests
full suite                                                553 passed, 1 skipped
```

Any change to the first number is a regression and must be explained, not
absorbed.

## The fail-when-fixed tests

`flightsim/tests/test_audit_regression.py` contains **ten tests deliberately
written to go red when the defect they pin is repaired**, each carrying a message
saying what to replace it with. This is by design: a test that pins a flaw must
not silently keep passing once the flaw is gone.

Find them with:

```
grep -n "WRITTEN TO FAIL WHEN FIXED\|should be replaced\|replace this test\|replace it with" \
    flightsim/tests/test_audit_regression.py
```

**When one of your fixes turns one of these red, that is success, not failure.**
Replace it with the positive assertion its message names. Do not delete it, do
not weaken it, and do not mark it `xfail`.

---

# TIER 1 — Fix these. They are certainly wrong.

Each is certain because the evidence is in this repository and the corrected form
is forced by a held source or by arithmetic. Ordered by consequence.

## 1.1 `validation.lateral_modes` swaps roll and spiral when the spiral is unstable

**Where:** `flightsim/validation.py:159`.

**What is wrong.** `reals.sort()` orders **signed** time constants. The comment on
that very line states the intent correctly — *"roll subsidence is fast (small
tau), spiral is slow"* — but a negative τ sorts below every positive one. So for
an aircraft with an unstable spiral, the two are returned swapped. The Cherokee's
unstable spiral (τ = −51.6 s) is returned as `roll_tau`, and its true roll
subsidence (0.360 s) as `spiral_tau`.

**Why it is certain.** Pure logic; no source needed. Roll subsidence is the fast
mode by definition, and "fast" is a statement about |τ|.

**The fix.** Sort by `abs(tau)`. The audit test at
`test_audit_regression.py:841` already names this fix in its own docstring.

**Verify.** The Cherokee returns `roll_tau ≈ 0.360 s` and `spiral_tau ≈ −51.6 s`.
The three aircraft with stable spirals return exactly what they returned before —
bit-identical, since `sort()` and `sort(key=abs)` agree when all values are
positive. **That bit-identity is the check that this fix moved nothing else.**

**Replaces:** the fail-when-fixed test at line ~841.

## 1.2 `aircraft.py` misquotes its own cross-check, in the flattering direction

**Where:** `flightsim/aircraft.py:217`.

**What is wrong.** The comment cites Table IX-1's `CLδe` as **0.396**. The table
reads **.356**. The recovered 0.364 therefore agrees with the source to **2.2%**,
not the 8.1% the comment implies.

**Why it is certain.** The source is in the repository —
`refs/NASA-CR-2144.pdf`, printed p.216 (PDF p.222, offset +6). It was read at
500 dpi twice, independently, by audit Agent B and by the Phase 3 auditor.

**The fix.** Correct the number and the agreement it implies. Re-read the page
yourself rather than trusting this prompt. The other three numbers in that
comment (`Cmδe` −1.40, `Clb` −0.281, `Cnb` +0.184) are correct — leave them.

**Verify.** Read printed p.216 at ≥300 dpi and quote what you see.

## 1.3 The airspeed floor is applied to dynamic pressure, where it does not belong

**Where:** `flightsim/aero.py:63` (`air_data`), consumed at `aero.py:141`.

**What is wrong.** `V_MIN = 1.0` exists as a NaN guard, because α, β and the
non-dimensional rates all divide by V. But the floored V is then also used to
build `qbar = 0.5·ρ·V²`, which needs no guard at all — it is finite at V = 0. The
consequence is that a **stationary aircraft reports dynamic pressure it does not
have**, and the residual `CL0` term produces force out of still air: 19.86 N
(747), 170.73 N (747-approach), 4.30 N (Cherokee), 1.44 N (Cessna), each at its
own cruise altitude. Free fall therefore reads `n_z` = +7.0e-6 to +4.0e-4 instead
of 0.

**Why it is certain.** `qbar` has no division by V. The floor cannot be defended
there on the grounds it is defended everywhere else.

**The fix.** Build `qbar` from the true `‖vel_rel‖`; keep the floor only where
something divides by V (β, `p_hat`, `q_hat`, `r_hat`). Update the `V_MIN` comment,
which currently says only that the floor "never binds in normal operation" and
does not mention that it inflates `qbar`.

**Verify.**
- Force at exactly V = 0 becomes exactly zero for all four aircraft.
- `load_factor` in aero-zeroed free fall stays exactly `−0.0`, and in *aero-live*
  free fall becomes exactly `−0.0` too.
- **Above 1 m/s nothing changes bit-for-bit.** `jnp.maximum(norm, 1.0)` returns
  `norm` exactly for `norm ≥ 1`, so every quoted result must be bit-identical.
  Assert this — it is the whole safety argument for the change.

**Replaces:** `test_aerodynamic_force_is_nonzero_at_exactly_zero_airspeed`
(line ~1506) and `test_the_airspeed_floor_overstates_dynamic_pressure_below_itself`.

## 1.4 `trim.is_physical` endorses trims no aircraft could fly

**Where:** `flightsim/trim.py:123`.

**What is wrong.** The function is named `is_physical` and checks `|α| ≤ 15°` and
**nothing else** — not throttle, not elevator, not even the residual it was
reached with. On a plain (V, h) sweep at the shipped initial guess, solutions that
*pass* while demanding throttle outside [0,1] or elevator past its limit:
**319/640** (747), 22/640 (747-approach), 284/640 (Cherokee), **364/640**
(Cessna). The 747 at V = 471.8 m/s is endorsed at α = −0.57° on a throttle of
**567**.

**Why it is certain.** The deflection and throttle limits are already in the
`Aircraft` pytree. A trim requiring 567× full thrust is not a flight condition
under any reading of the word.

**The fix.** Extend the check to throttle ∈ [0, 1] and |elevator| ≤
`ac.elevator_limit`. This requires the aircraft, so **the signature must change**
— `is_physical(x, ac)`. Update every call site. Consider also taking the residual;
if you do, take it as an argument rather than recomputing, so the function stays
free of solver assumptions.

**Verify.** All four registry aircraft at their own cruise conditions still pass —
`test_trim.py` already asserts this as the positive control, and it must stay
green. The pinned 747 root at V = 471.8 m/s, h = 11579 m now fails.

**Replaces:** `test_is_physical_endorses_trims_that_no_aircraft_could_fly`
(line ~1654), and the absurd-root test at line ~1129 if it also goes red.

## 1.5 `along_track_shear` omits the heading-rotation term

**Where:** `flightsim/wind.py:355-373`.

**What is wrong.** `heading` is computed **outside** `u_x` and closed over, so
`jax.grad(u_x)` differentiates the wind field with the track direction held
fixed. The along-track wind of a *turning* aircraft also changes because the
along-track direction itself rotates. The missing term is `−u_⊥·ψ̇`, and it
reaches **ΔF = 0.140** at a standard-rate turn — the entire FAA hazard threshold.

**Why it is safe to fix now.** It is **exactly zero in all four of the project's
runs**, because every field has zero east wind on the north axis, so `ψ̇ ≡ 0`.
Nothing quoted moves.

**Why the fix needs care.** Proctor et al. Eq. (4) — the cited source, held at
`refs/Proctor-Hinton-Bowles-2000-Windshear-Hazard-Index.pdf` — is written for a
straight track. **Extending it to a turning aircraft is a derivation, not a
transcription.** So derive it, show the derivation in the docstring, and show
that it reduces to the paper's Eq. (4) exactly when `ψ̇ = 0`. That reduction is
the check that you have extended the source rather than replaced it.

**Verify.**
- All four project runs unchanged to ~1e-15.
- A constructed standard-rate turn through a field with cross-track structure
  reproduces the ΔF = 0.140 the audit measured.

## 1.6 Three documentation claims the code does not implement

Each is a statement in prose that the code contradicts. Fix the prose or the code,
and say which you chose.

**(a) The provenance ledger enforces nothing.** `PROJECT.md` §2 (twice) and
`provenance.py`'s module docstring state that *"a constant added without a ledger
entry fails the build"*. No test has ever walked the source modules; all eight
tests in `test_provenance.py` iterate `LEDGER` against itself. Measured coverage:
~346 non-trivial numeric literals across nine physics modules against 13 ledger
entries — about **2%**. `wind.py`, `aero.py`, `atmosphere.py`, `units.py` and
`autopilot.py` have **zero** entries.
*The audit already supplied the missing direction as
`test_the_provenance_ledger_does_not_cover_the_source_modules`, which pins the
baseline so a **new** unledgered constant fails the build.* Your job is to make
the documentation true — either by adding ledger entries and shrinking
`KNOWN_UNLEDGERED`, or by correcting both documents to describe what is actually
enforced. **Never widen `KNOWN_UNLEDGERED`.**

**(b) `ASSUMPTIONS.md` E4's bound is wrong by ~80×.** E4 bounds the wind-hold seam
by an h-vs-h/2 **discretisation** refinement (0.0024 m/s, "~1e-4 relative", "no
result the project quotes is affected"). The flaw is a **scheme** error, and the
right instrument is hold-vs-per-stage at the same dt: the headline in-core Δθ
moves **1.62% at dt = 0.02 and 0.82% at the published dt = 0.01**. No conclusion
changes — the vortex claims are orderings — but **the quoted 2.240° is not good
to four figures**, and E4 says it is. Correct the number and the instrument
description.

**(c) The Dryden/von Kármán justification has no citation.** `integrate.py`'s
module docstring calls holding wind across the four RK4 stages *"the standard
treatment for Dryden and von Kármán turbulence"*. **No citation exists anywhere in
the repository and none was found.** This matters more than a normal uncited
claim, because it is the sole stated justification for the design choice that
costs the scheme three orders of accuracy. Either cite a real source you have
read, or restate it as what it is: a deliberate design choice, correct for a
stochastic field, with the reasoning given and no appeal to authority.

## 1.7 `PROJECT.md` §4 compares 2 of the 4 published mode factors

**What is wrong.** CR-2144 Table IX-5's denominator publishes four longitudinal
factors for FC9. `PROJECT.md` §4 compares two of them. The two it omits are the
two that look worse: **phugoid ζ (+14.4%)** and **short-period ωn (−1.4%)**.

**Why it is certain.** All four are printed on p.231 of a document in `refs/`, read
at 600 dpi: `Z(DET)1 = .0489`, `W(DET)1 = .0673`, `Z(DET)2 = .387`,
`W(DET)2 = .964`.

**The fix.** Add both rows. `AUDIT.md` §2.3 carries the full table and the
attribution — every one of the four closes to ≤1% when the two omitted derivative
families are restored, so **adding the missing rows strengthens the project's
position rather than weakening it.**

## 1.8 `superpose()` with no arguments returns the integer 0

**Where:** `flightsim/wind.py:500`.

`sum(field(pos) for field in ())` is `0`, an `int`, not a callable field. Any
caller that superposes an empty list gets a value where it expected a function.
Return a zero field or raise. Trivial and certain.

## 1.9 `test_conservation.py:82`'s comment is wrong by a decade

The comment says "measured ~5.7e-14"; the value is **5.6958e-13**. `PROJECT.md`
has it right. Fix the comment.

## 1.10 Four assumptions the code makes that the register never declared

Documentation only — no code change. Add entries to `docs/ASSUMPTIONS.md` for
each, with the bound the audit measured. All four are written up in
`ASSUMPTIONS_AUDIT.md`; port them into the register itself, which is where a
reader looks.

- **There is no ground** (U16). A 747-approach released at 300 m crosses h = 0 at
  t = 8.65 s at a 25.1 m/s sink rate and integrates on to −698 m, all finite. Say
  so, and say that `scripts/microburst.py` guards its own case at one wingspan
  while nothing guards anyone else's.
- **Superposition breaks the microburst's ground boundary condition** (U14).
  Fields sum exactly, but the microburst's defining property — both components
  vanish at z = 0, which its paper singles out as what earlier models got wrong —
  does not survive addition. Superposing anything with a non-zero ground value
  puts **3 m/s through the ground everywhere**.
- **The engine carries an `X_q` that CR-2144 does not model** (finding 18). Drag
  responds to `CLq·q̂`; stability-axis `A[0,2] = −0.8807 s⁻¹` against the
  parabolic polar's own prediction of −0.8887, agreeing to 0.9%. Correct physics,
  undeclared, and it is one of the two terms that decides phugoid damping.
- **Wave drag acts on the total `CL`**, including the `CLq·q̂` and `CLδe·δe`
  contributions (U10). Defensible; undeclared.

## 1.11 `_B747_G = 32.174` inlines a conversion factor the project forbids

**Where:** `flightsim/aircraft.py:222`. `units.py` states the rule: *"Never inline
a conversion factor anywhere else."* This does, and truncates
`g₀ = 32.17404855643044`.

**⚠ This one moves a reference value.** The slip against the `units.py` route is
**1.5e-6 relative**, which propagates into the 747's derived mass and therefore
into every downstream number. **Do not fix it silently.** Make the change, measure
exactly which tests and which quoted numbers move and by how much, and **report
that before committing**. If anything in `PROJECT.md` §4 moves outside its stated
tolerance, revert and raise it instead — a 1.5e-6 rule violation is not worth
disturbing a baseline over, and the honest outcome may be to document the
exception rather than take it.

## 1.12 Provenance strings cite a file that does not exist

`aircraft_data_validated.py` is named as the source of the Cherokee and Cessna
data. **It is not in this repository.** 168 of 346 constants terminate in it.

**The fix is honesty, not substitution.** Correct the citations to state the
actual status — `unverifiable — source not available` — naming the original
upstream references (McCormick, Roskam/DATCOM via PyFME) as *claimed but
unverified*. **Do not go and find replacement numbers from elsewhere and quietly
swap them in.** That would convert a documented gap into an undocumented one. See
also Tier 2.7, which is independent evidence that this data has a real problem.

---

# TIER 2 — Real findings. **Do not "fix" these.** The correct form is not
# established by any source in this repository.

These are the ones the title of this prompt is asking about. Each is a genuine
finding. Not one of them has a repair whose correctness this repository can
demonstrate. For each: what is uncertain, and what would settle it.

## 2.1 The energy seam — the lift-tilt term ⚠ **highest risk of a wrong "fix"**

**The finding is solid.** The model creates mechanical energy in still air:
`P_aero = +56,927.7 W` on the Cherokee. The audit demonstrated the mechanism to
machine precision — the elevator's moment power is cancelled *exactly* by the
lift-tilt drag of its own lift increment placed at `l_δe = −(Cmδe/CLδe)·c`,
residual 1.9e-16, and restoring that one term takes the violating region from 1190
grid points to zero.

**Why you must not implement it anyway.** Three reasons, each sufficient:

1. **It would be internally inconsistent.** The tilt applies to *every* lift
   contribution at *its own* arm — `CL0`, `CLa`, `CLq` as well as `CLδe`. The
   model has no arms for the others. Correcting one channel and not the rest
   makes the force build-up less coherent than leaving all four alone.
2. **The arm is only valid under an attribution the project has already
   rejected.** `l_δe = −Cmδe/CLδe·c` holds if *all* of both derivatives come from
   the tail. `airframe.effective_tail_arm`'s docstring records that separating the
   wing's share was tried, gave the wing 83% of `CLq` and implied a 23-chord arm,
   and was **rejected with an explicit do-not-re-attempt note**. The same
   objection applies here and no source settles it.
3. **It moves everything.** Trim, all five modes, and every headline number.

**The violation is also entirely outside the declared envelope**: 0 of 80,000
randomised in-envelope states show it, the threshold pitch rate is 84–201 °/s, and
`max(E − E₀) = +0` **exactly** in every still-air run ever attempted, including
feedback laws designed to pump it.

**Correct response: document it** as `ASSUMPTIONS.md` U15 and leave the code
alone. **What would settle it:** a source giving the wing/tail split of `CLδe`,
or replacing the derivative build-up with a strip or panel force model that places
every load where it acts.

## 2.2 The conversion chain vs CR-2144 Appendix A's literal definitions

**The finding is solid.** The code uses small-angle forms. Appendix A's `X_w`,
`Z_w`, `M_w` each carry a `−2(W₀/U₀)(C + (M/2)C_M)` term the code drops, and use
`U₀ = V_To cos α₀` where the code uses `V_To`. Cost at cruise: `CLa` +1.15%,
`CDa` +1.13%, `Cma` +0.32%.

**Why you must not implement it.** The `W₀` correction needs `C_NM` and `C_M` —
and **CR-2144 publishes those only as plots against Mach, not tables** (printed
pp. 224–228). Reading them is a chart read off a poor scan, which the project
already declines elsewhere for the stated reason that a chart read is weaker
evidence than the tabulated set in use. So the literal form **cannot be completed
from held sources**; the audit quotes it with `C_NM = 0` and a sensitivity rather
than a value.

**And it explains nothing.** Rebuilding the cruise 747 with Appendix-A-literal
values moves phugoid ωn from −17.80% to −17.73% and short-period ζ from −11.49%
to −12.04%. It was tested as a candidate cause of the mode discrepancies and
**falsified**.

**Correct response:** it is already recorded as U2. Leave it.

## 2.3 The strip quadrature at the shipped station count ⚠ **needs a decision, not a patch**

**The finding is solid and it is a live defect.** `loads.strip_model` builds
stations at `N_SPAN = 9`, where the strip integral returns **82.6%** of `Clp` — a
−17.4% systematic understatement. The calibration identity `a₀ = −8·Clp` is exact
only in the continuum limit, and two docstrings say "exactly". Every existing test
overrides the count (201, 2001, 21); none exercises 9.

**Why the fix is not obvious.** Two defensible repairs, and they mean different
things:

- **Raise `N_SPAN` until converged.** Keeps `a₀` meaning what its docstring says.
  Costs field evaluations per step, and the integrand is sqrt-singular at the tips
  so convergence is slow — measure how slow before choosing.
- **Calibrate `a₀` against the discrete quadrature at the shipped `N`.** Makes the
  identity exact *in the code as run*, which is arguably what a calibration is
  for. But it makes `a₀` depend on `N_SPAN`, which changes what the constant
  means, and `a₀` is already declared an *effective* value absorbing sweep and the
  tail's share.

**Correct response: pick one deliberately, record the reasoning in
`provenance.py` and `ASSUMPTIONS.md`, and state the loading-shape sensitivity
beside any result** (2.6% across defensible shapes, 49.7% including a uniform
bracket). Do not silently switch calibration basis.

## 2.4 The microburst's unmodelled fourth parameter `z_h`

**The finding is solid.** Oseguera & Bowles specify four parameters;
`wind.Microburst` carries three. The field has no ceiling and grows to a 50.6 m/s
on-axis asymptote. The project's 300 m AGL penetration sits essentially at the
paper's own `z_h` (294.4 m), so it is not academic.

**Why it is uncertain.** The paper **is** held
(`refs/NASA-TM-100632-Oseguera-Bowles-1988.pdf`), so unlike 2.2 this *could* be
implemented from source. But doing so changes the field and moves every microburst
number the project reports. That makes it a **modelling decision, not an error
correction** — and the transcription of eqs. (5)–(6) is itself exact, verified
over 12,000 points to 4e-15 (`w`) and 1.6e-11 (`u`).

**Correct response:** the *undeclared* half is a Tier 1 documentation fix (already
covered as U13). Implementing `z_h` is a scoped change with its own
re-measurement, not part of this remediation.

## 2.5 The singular Jacobian's silent NaN

**The finding is solid.** The Cessna's `CYdr = Cldr = Cndr = 0` zeroes the rudder
column of any control Jacobian; `det = 0` and `jnp.linalg.solve` returns
`[nan, nan, inf]` **without raising**.

**Why it is uncertain.** `trim` does not carry rudder as an unknown, so no shipped
solver hits it. And **JAX cannot raise inside a jitted function**, so the obvious
guard is not available where it would be needed. A non-jitted precondition check
is possible but is a design decision about where solver preconditions live.
There is also a genuine partial mitigation: `conftest.py` sets `jax_debug_nans`,
so the suite catches it — but that is a **test-time setting only**, and nothing
enables it for `scripts/` or a library caller.

**Correct response:** documented as U18. If you do add a guard, add it as an
explicit non-jitted precondition, not by making `trim` non-jittable.

## 2.6 The one-sided longitudinal station set

`airframe.stations`' longitudinal set runs from `−arm` to `0` — **entirely aft of
the CG**, centroid −16.75 m — so `sampled_rates`' pitch channel is a *backward*
secant with an `O(arm/2·f'')` bias. At −0.99 r₀ the E2 correction is **1.80
V₀/r₀**, not the zero the register claims for the core interior.

**Why it is uncertain.** Centring the stations needs a forward extent, and **no
source tabulates one** for any aircraft here. The aft arm is itself derived, not
measured. Inventing a nose station would breach Rule 2.

## 2.7 The two tail arms disagree by 2–2.9× on the light aircraft

The rate pair `−Cmq/CLq` and the control pair `−Cmδe/CLδe` describe the same
geometry. 747 cruise **4.024 vs 3.969**; 747 approach **3.852 vs 3.965**;
Cherokee **1.280 vs 2.562**; Cessna **0.856 vs 2.468**.

**Why it is uncertain — and important.** The split is exactly the sourcing split.
The two CR-2144 sets agree to 1.4% and 2.9%, corroborating that transcription. The
two that disagree are precisely the two whose cited source file does not exist
(Tier 1.12). **Which of the two estimates is wrong, or whether both are, cannot be
determined** — the data has no retrievable source.

**Do not adjust either derivative to make them agree.** That is fabrication. A
defensible cheap improvement is to widen `tail_arm_is_plausible` to require the
two routes to *agree* as well as to fall in band — it currently reads only the
rate estimate and rejects both light aircraft by luck rather than by design.

## 2.8 Smaller uncertain items

- **Vortex core branch at exactly `r = r₀`** (finding 31). `vortex_wind`'s strict
  `<` resolves the tie to the outside branch, and round-off in the position
  decides which branch a sample takes. A one-character change picks the inside
  branch instead. Measure-zero, and both branches agree in *value* to 2.5e-13 —
  so it is not clear the change is an improvement rather than a different
  arbitrary choice.
- **The atmosphere has no ceiling or floor** (U6). Above 11 km it holds
  T = 216.65 K forever; below 0 it extrapolates the lapse to 30.47 kg/m³ at
  −50 km. Clamping changes behaviour for existing callers, and JAX cannot raise
  inside `jit`. Design decision.
- **`airframe._active_shape` is process-global mutable state** (U8). A physics
  parameter set by a context manager; correctness under `pytest-xdist` or threads
  is assumed and untested. A refactor, not a bug fix.
- **`δa` is a compound control treated as a single angle** (U9). CR-2144's
  footnote defines it as total deflection of two ailerons with outboard effects
  included; `aileron_limit = 20°` is a declared limit on a compound quantity.
  Document; do not reinterpret the source's number.

---

# TIER 3 — Do not touch. There is nothing to fix.

1. **CR-2144 contradicts itself on `C_Lα̇`** (finding 12). Table IX-2 prints −6.7;
   Table IX-4 FC2's `ZWD = +0.0338` gives −0.0341 through Appendix A. The `C_mα̇`
   route is consistent to 0.4%; only the Z row conflicts. **Record both, pick
   neither.** Choosing one silently would be inventing agreement.
2. **Parks 1985 and Wingrove & Bach 1994 could not be obtained.** Both are AIAA
   paywalled; Unpaywall reports `is_oa:false`; NTRS holds metadata only. The
   vortex field equations, `PARKS_CASES`, `UPDRAFT_W0`, `UPDRAFT_SECONDS`, the
   Fig. 8 discriminator and the ±25% band all remain `unverifiable`. **Do not
   substitute values from any other vortex model to make them verifiable.** The
   headline result rests on this and the honest status is the correct one.
3. **FD2e eqs. 3.4-49 and 3.4-55 are wrong** (finding 14) — the sign error and the
   factor of −2 were re-derived and confirmed. **The book is wrong and the code
   does not use either equation.** Nothing to do. Do not "correct" the code toward
   the book.
4. **The omitted speed (`Xu, Zu, Mu`) and α̇ (`Zẇ, Mẇ`) derivative families**
   (findings 6, 7). These are a documented, deliberate scope limitation, and the
   audit *demonstrated* that restoring them closes all four published mode factors
   to ≤1%. Restoring them is a **feature**, with its own design and
   re-measurement. It is not an error correction and does not belong in this pass.
5. **The wind hold's first-order behaviour** (finding 36, E4). The scheme is
   fourth order in still air and in a *gradient-free* field (3.9873 and 3.9875) and
   first order in any field with a spatial gradient (0.99, 1.00). This is a
   property of a deliberate design choice that is **correct for a stochastic
   field**, which is what it was chosen for. Not a bug. (The *uncited* claim about
   why is Tier 1.6c — that is a citation problem, not a physics problem.)
6. **Rigid airframe vs "Flexible" data** (B1) and **frozen derivatives across the
   envelope** (C3). Honest unbounded limits, correctly declared. The right
   response is to stop short of claims they cannot support, which the register
   already does.
7. **`jit` vs eager last-ulp differences, float32 costs, vertical Galilean
   non-invariance.** All measured, all expected, none an error.

---

## Success criteria

A fix is done when all of these hold. "Tests pass" is not one of them.

1. **Every Tier 1 item is either fixed or has a written reason it was not**, and
   the reason is not "it was hard".
2. **The pre-existing suite is still 444 passed, 1 skipped.** Any movement is
   explained, not absorbed.
3. **Every fail-when-fixed test your change turned red has been replaced by the
   positive assertion its message names** — not deleted, weakened, or `xfail`ed.
4. **No tolerance, reference value, or test was weakened.** State this explicitly,
   and be able to show it with a diff of deleted test lines.
5. **No constant was invented.** Every number added traces to a document in
   `refs/`, to FD2e, or to arithmetic on numbers that do.
6. **Each fix carries its own verification measurement**, including the
   bit-identity checks named in 1.1 and 1.3 — those are what prove a fix was
   surgical rather than merely green.
7. **`AUDIT.md`, `ASSUMPTIONS_AUDIT.md` and `docs/ASSUMPTIONS.md` are updated to
   reflect what is now fixed**, with the findings' status changed rather than the
   findings deleted. An audit finding that was repaired should read as repaired,
   with a pointer to the commit — not vanish.
8. **The PDF is regenerated** with `python scripts/audit_report.py`, never written
   directly.

## One thing to keep in mind throughout

The audit's most valuable outputs were not the transcriptions that were right —
they were the **measured sizes of the things that are wrong**. Tier 2 exists
because several of those measurements are more useful to this project than the
repairs would be. A bounded, cited, honestly-recorded flaw is a finished piece of
work. Resist the urge to close it just because it is open.
