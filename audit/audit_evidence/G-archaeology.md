# G — Archaeology: git-history evidence

Read-only audit of `C:\Users\mateusz\UROP\Claude_Flight_Sim`, branch `master`, 86 commits,
9 branches. No git state was altered. Evidence gathered with `git log -p`, `git log -S`,
`git log -L`, `git show`, `git reflog`, `git fsck`.

**Standing rule under test** (`docs/PROJECT.md`): *"Do not edit a tolerance to make a test
pass"* / *"A failure is a finding."*

---

## 0. Headline

**No tolerance was ever loosened to make a test pass. No reference value was ever edited.**

In the entire recorded history of `flightsim/tests/` exactly **one** numeric bound was ever
changed (`ff98d93`), and it was **net-tightening**, documented in the commit message, and
landed with **no change to the implementation under test**.

Two findings that are not rule violations but are reportable:

- **F1 (docs claim never implemented).** `docs/PROJECT.md` §2 asserts "a constant added
  without a ledger entry fails the build." Nothing in `test_provenance.py` has ever
  enforced this. See §7.
- **F2 (abandoned finding).** Commit `d9d4442` — a sourced, quantified bound on the 747
  thrust pitching moment, including a new primary source — never landed on `master`, and
  `master`'s `docs/ASSUMPTIONS.md` still carries the claim that commit proves false. See §6.

### How thoroughly item 1 was checked

The audit is exhaustive rather than sampled. A tolerance can only change by deleting a
line, so every deleted line was read:

| Scope | Count |
|---|---|
| Commits touching `flightsim/tests/` (all branches) | 53 |
| Lines inserted / deleted in `flightsim/tests/` over all history | 8,971 / **590** |
| Deleted lines individually examined | **591** (590 + 1 trailing) |
| Merge commits touching tests | 3 — **all with zero combined-diff content** (no conflict-resolution edits) |
| Test files ever deleted or renamed | **0** |

Tolerance sites present in the current suite, all of which therefore trace to a single
introducing commit unless listed in §1:

`atol=` 27 · `rtol=` 2 · `abs=` 92 · `rel=` 91 · `approx(` 263 ·
asserts with a bare numeric comparison 192.

---

## 1. Every numeric tolerance in `flightsim/tests/`

### The only bound ever changed

**`test_wind.py` — Parks Case 1 first-core pitch excursion**

| | |
|---|---|
| First commit | `079a34f` 2026-08-05 "Kelvin-Helmholtz vortex array from Parks et al. 1985" |
| Changed? | **Yes, once** |
| Commit | `ff98d93` 2026-08-06 "Add the project record, load factor, and the vortex-analysis figure" |
| Before | `assert 0.5 < dtheta < 3.0, dtheta  # measured 1.89 deg; Fig. 8 vortex ~1.4` |
| After | `assert 1.0 < dtheta < 3.5, dtheta  # measured 2.20 deg; Fig. 8 vortex ~1.4` |
| Implementation changed in same commit? | **No.** `flightsim/wind.py` was not touched. Files changed: `docs/PROJECT.md`, `flightsim/dynamics.py` (**adds** `load_factor`, purely additive), `flightsim/vortex_viz.py` (new), `scripts/vortex.py` (new), and three test files. |
| Verdict | **Not a violation. Net-tightening.** |

Why it is net-tightening, not loosening:

- Band **width is identical** (2.5 deg before and after); the band was *shifted*, not widened.
- The **lower bound tightened** 0.5 → 1.0.
- What changed was the **test's own set-up**, not the physics: the vortex lead-in went from
  `-6.0 * CASE1_R0` to `-40.0 * CASE1_R0`, because the 1/r far field has not decayed at 6·r0
  and the aircraft was launched 0.20 g out of equilibrium.
- The same commit **added a new assertion** that did not exist before, pinning the thing that
  had been silently wrong:
  `assert n_z0 == pytest.approx(0.9967, abs=0.05), n_z0`
- The commit message states the defect and the convergence study in the open, and the
  docstring records the whole sensitivity sweep (n_z at 6/20/40/100 · r0; dtheta converging
  to 2.18–2.22 beyond ~12·r0).

*Minor observation, not a violation:* the new `n_z0` assertion is centred on the theoretical
`cos(theta) = 0.9967` with `abs=0.05`, while the comment records the measured value as
1.0352 (a 0.0385 gap). The tolerance is wide enough to admit the measurement. It is the
assertion's **first** value — never subsequently edited — so no rule was broken, but the
tolerance was sized around a known offset rather than derived.

### Everything else

Across all 53 test-touching commits, **no `atol=`, `rtol=`, `abs=`, `rel=` or `approx()`
value was ever edited.** Every one of the 27/2/92/91/263 sites was introduced at its current
value and never modified.

The only other tolerance-bearing lines ever deleted, all benign:

| Commit | Date | What | Verdict |
|---|---|---|---|
| `01faba0` | 2026-08-06 | 18 tests moved `test_viz.py` → `test_panel.py` verbatim | move, no value change |
| `0872b87` | 2026-08-06 | `pilot_input()` → `key_demand()` API rename in assertions | rename, no value change |
| `099bd33` | 2026-08-07 | tautological test deleted (see §4) | strengthening |
| `2aa9faa` | 2026-08-11 | inlined RK4 order fit → `verification.oscillator_refinement`; `abs=0.05` **unchanged** | refactor |
| `4947dde` | 2026-08-12 | `np.testing.assert_allclose(..., atol=1e-9)` → `assert got.max_position_error < 1e-9`; **same bound**, plus two *new* assertions (`peak_wind`, `peak_dwdt`, both `abs=1e-4`) | refactor + strengthening |
| `db67f4a` | 2026-08-14 | cycle check corrected to permit diamonds | correctness fix |

**Stale-but-never-loosened threshold.** `test_wind.py:375`
`assert dtheta > 1.5 * 1.89  # separated from the vortex case, the Fig. 8 claim`
entered in `8ca5148` (2026-08-06) and has never been touched. The `1.89` is the *pre-fix*
vortex value, later superseded twice (→2.20 in `ff98d93`, →2.24 in `553e3e3`). Because the
true value rose while the constant did not, the separation threshold is *weaker* than it
should be (2.835 instead of 3.36). This is a **failure to tighten**, not a loosening — but
it is a stale reference embedded in a live assertion.

---

## 2. Fitted / calibrated / declared constants

Format: name | first commit | changed? | verdict.

| Constant | First commit (sha, date, subject) | Value ever changed? | Verdict |
|---|---|---|---|
| `aircraft` 747-cruise `CD0`/`e` back-solve | `dca63ad` 2026-08-04 "Mach-dependent drag polar (Korn + Lock) and Boeing 747 from CR-2144" | **No** — lines are `+`-only in all history | clean |
| `aircraft` Cessna 172 `CD0`/`e` back-solve | `d13d382` 2026-08-05 "Add the Piper PA-28-180 Cherokee and the Cessna 172" | **No** | clean |
| `aircraft` Navion `CD0`/`e` (polyfit of tabulated polar) | `d13d382` 2026-08-05 | **No** | clean |
| `aircraft` 747 power-approach `CD0`/`e` back-solve | `71e6127` 2026-08-08 "Add the 747 power-approach condition from CR-2144 Table IX-2" | **No** | clean |
| `airframe.calibrated_lift_slope` | `9323069` 2026-08-15 "Add elliptic loading, calibrated so a rigid roll rate returns the sourced Clp" | **No** | clean |
| `airframe.TAIL_ARM_BAND = (2.0, 6.0)` | `5b0eebd` 2026-08-14 "Derive the effective tail arm from Cmq and CLq, and gate it on plausibility" | **No** | clean |
| `airframe.N_SPAN = 9` | `85dcfa3` 2026-08-14 "Add airframe sample stations, spanning the span and the tail arm" | **No** | clean; convergence study discharged separately in `4da367b` |
| `airframe.N_LON = 9` | `85dcfa3` 2026-08-14 | **No** | clean |
| `airframe._SENSITIVITY_TAPER = 0.3` | `9323069` 2026-08-15 | **No** | clean |
| `wind.LEE_WAVE_WAVELENGTH = 25_000.0` | `bdd3236` 2026-08-07 "Fly the mountain lee wave against the 747's thrust authority" | **No** | clean |
| `wind.UPDRAFT_W0 = 80.0 * FT2M` | `8ca5148` 2026-08-06 "Add the thunderstorm updraft column and field composition" | **No** — appears in 3 commits, all identical value; `8003fcc` (same day) *moved* it into `wind.py`, `e3822ad` is an abandoned WIP branch | clean |
| `wind.UPDRAFT_SECONDS = 20.0` | `8ca5148` 2026-08-06 | **No** — same three commits, same value | clean |
| updraft `sharpness = 6.0` | `8ca5148` 2026-08-06 (declared parameter); default `6.0` in `panel.py` | **No** — every literal in every commit is `6.0`; no other value ever appears | clean; explicitly labelled DECLARED, and `test_wind.py` records the 2/4/6/10 sensitivity sweep |
| `aero.V_MIN = 1.0` | `0840576` 2026-08-04 "Step 2: aero build-up, 6-DOF dynamics, wind/gust hooks" | **No** | clean |
| `trim.ALPHA_LIMIT = radians(15.0)` | value from `a7b3501` 2026-08-11 (as `validation.ALPHA_LIMIT`); **relocated** to `trim.py` in `d69fed5` 2026-08-11 "Close the other gust seam, and move the trim bound to where the defect is" | **No** — 15.0 deg both sides of the move | clean; a *relocation* (`np.radians` → `math.radians`), with the rationale recorded: the defect is in `trim.trim`, so the bound belongs there |
| `trim.INITIAL_GUESS = [0.05, 0.0, 0.5]` | value from `332ba1e` 2026-08-04 "Step 5: Newton trim solver…" (inline literal); **named** in `7a9ee55` 2026-08-11 | **No** | clean; extracted to a constant precisely so `verification.py` cannot measure a hand-copied duplicate |
| `validation.TRIM_RESIDUAL_LIMIT = 1e-9` | `a7b3501` 2026-08-11 "Reproduce Caughey's 747, and find the laws the sweeps actually follow" | **No** | clean |

**None of the 17 fitted/calibrated/declared constants above ever had its value changed.**
Two moved file (`ALPHA_LIMIT`, `UPDRAFT_*`) and one was extracted from a literal
(`INITIAL_GUESS`); all three kept their value exactly. No constant change coincides with a
test starting to pass, because no constant value changed at all.

`flightsim/aircraft.py` was touched by only 5 commits in its life
(`0840576`, `dca63ad`, `d13d382`, `553e3e3`, `71e6127`); every `CD0 =` / `e =` line in the
history is an insertion with no matching deletion.

---

## 3. Reference values

**No reference value was ever edited. Not one.**

| Reference set | First commit | Ever edited? |
|---|---|---|
| `flightsim/tests/test_cr2144_modes.py` | `7e9086e` 2026-08-05 "Validate integrator, CR-2144 derivative chain, drag polar, and Navion" | **No — the file has never been touched again** |
| `flightsim/tests/test_navion.py` | `7e9086e` 2026-08-05 | **No — never touched again** |
| `flightsim/tests/test_drag_polar.py` | `7e9086e` 2026-08-05 | **No — never touched again** |
| `validation.REFERENCES` | `a7b3501` 2026-08-11 | **No** |
| `validation.CAUGHEY_A` | `a7b3501` 2026-08-11 | **No** |

`flightsim/validation.py` has been touched by exactly two commits in its life: `a7b3501`
(creation) and `d69fed5`. A `git log -L 185,235` over the `REFERENCES`/`CAUGHEY_A` block
shows the only deletion in that range is the `ALPHA_LIMIT` definition and its comment
(relocated, §2) — the reference data itself is untouched.

The strongest single piece of evidence in this audit is that the three files holding the
published comparison numbers were written once, on 2026-08-05, and have not been modified
in the 81 commits since.

---

## 4. Tests deleted, renamed, skipped or weakened

**Test files deleted or renamed: 0.**

**`def test_` removals: 20 total**, every one accounted for:

| Commit | Date | Removals | Disposition |
|---|---|---|---|
| `01faba0` | 2026-08-06 | 18 | **Move.** `test_viz.py` → `test_panel.py` when the live cockpit moved to `flightsim/panel.py`. All 18 verified present in `test_panel.py` today. |
| `0872b87` | 2026-08-06 | 1 | **Rename.** `test_held_keys_become_a_stick_position` → `test_held_keys_become_a_stick_demand`, tracking `pilot_input()` → `key_demand()`. Same commit **adds 17 new tests**. |
| `099bd33` | 2026-08-07 | 1 | **Deliberate deletion of a tautology — strengthening.** See below. |

The one genuine deletion, `test_derived_agrees_with_the_aero_module`, was removed because it
fed `viz.derived` and `aero.air_data` the same input and compared them, so it could never go
red. It was flagged as such in `PROJECT.md` §6(b) and in the session 6 log *before* removal.
A tombstone comment remains at `flightsim/tests/test_viz.py:105-112` naming the replacement,
and the replacement exists: `flightsim/tests/test_sensors.py:196`
`test_derived_is_air_relative_and_this_test_can_fail`, which computes the expectation from
the recorded wind independently. Commit `553e3e3`'s message is explicit: *"the plan asked
that test_viz's derived test 'can actually go red' and the honest answer was that it never
could… It is replaced, not repaired."*

**`@pytest.mark.skip` / `@pytest.mark.xfail`: never added, anywhere, in the entire history.**
An exhaustive grep of every `+`/`-` line mentioning `skip` or `xfail` across all test
commits returns only four entries, all additions:

- `71e6127` 2026-08-08 — the one `pytest.skip` (below)
- `d6079e6` 2026-08-16 — three `pytest.importorskip` guards for optional deps
  (`pyarrow`, `plotly.graph_objects`, `flightsim.analysis.figures`), i.e. optional-extra
  gating, not test suppression.

### The 1 skipped test

| | |
|---|---|
| Location | `flightsim/tests/test_aircraft.py:252` |
| Test | `test_cruise_is_above_the_minimum_drag_speed`, parameter `boeing747_approach` |
| Statement | `pytest.skip("approach condition; asserted below instead")` |
| Skipped in | `71e6127` 2026-08-08 "Add the 747 power-approach condition from CR-2144 Table IX-2" — the same commit that introduced the approach condition |
| Why | The 747 power-approach point is *deliberately* below V_md — an airliner on final is flown on the back side of the drag curve. It is not a cruise condition, so the cruise test does not apply. |
| Compensating assertion | **Yes** — the immediately following test, `test_the_approach_condition_is_below_the_minimum_drag_speed`, asserts the opposite positively rather than leaving a silent exemption. The docstring says so: *"see the next test, which asserts that rather than leaving it as a silent exemption."* |
| Verdict | **Not a weakening.** A parametrised case correctly routed to a more specific assertion. |

**Assertions changed from equality to approximate, or bounds widened: none**, beyond the
single shifted band in §1.

---

## 5. The known transcription fixes

### (a) `Mq` −0.330 → −0.339

| | |
|---|---|
| Commit | `553e3e3` 2026-08-06 "Fix latent bugs (a)-(d); add the air-relative sensing seam" (session 5) |
| File | `flightsim/aircraft.py` |
| Before | `Xw, Zw, Zq, Mw, Mq, Zde, Mde = 0.0389, -0.317, -5.16, -0.00105, -0.330, -17.9, -1.16` |
| After | `… -0.00105, -0.339, -17.9, -1.16`, with an in-code comment recording the slip and its cost |
| What moved | Short-period damping match **12.6% → 11.5%** error; first-core vortex `dtheta` **2.20 → 2.24 deg** |
| Tolerance moved in the same commit? | **No.** `553e3e3` did not touch `test_wind.py` at all. The band `1.0 < dtheta < 3.5` set in `ff98d93` absorbed the move with room to spare. |
| Other test edits in `553e3e3` | `test_aircraft.py` (+7/−3), `test_autopilot.py`, `test_integrate.py`, `test_manual.py`, `test_viz.py` — **all pure API plumbing** (`ap_mod.engage(state, …)` → `ap_mod.engage(sense(state), …)`). No numeric value in any assertion changed. `test_sensors.py` (+256) is entirely new. |
| Verdict | **Clean.** The commit message states both moved numbers explicitly and records that they were *"superseded in the ledger rather than overwritten."* |

Corroboration that the fix is honest: `provenance.py` records
`"b747.Mq": Entry("SOURCED", "-0.339, dimensional. …CR-2144 Table IX-4")` — the ledger
carries the corrected value with its table citation.

*Documentation drift, minor:* the `test_wind.py` docstring and inline comments still read
"measured 2.20 deg" / "8.79 deg" / "8.33 vs 2.20", i.e. the pre-`Mq`-fix numbers. The
commit chose to supersede in the ledger rather than overwrite in place, so the comments were
left behind. The **assertions** are correct; only the commentary is stale.

### (b) Cessna control-derivative dynamic-pressure bug

| | |
|---|---|
| Nature | Control derivatives must be recovered at the dynamic pressure of the *source's own linearisation* (67 m/s), not at the model's chosen cruise (60 m/s). Using the wrong one inflates all four by **25%**. |
| First commit | `d13d382` 2026-08-05 "Add the Piper PA-28-180 Cherokee and the Cessna 172" |
| What the history actually shows | The correct form — `qS_source = 0.5 * rho * 67.0**2 * S`, used for `CLde`, `Cmde`, `Clda`, `Cnda` — is present **from the first commit that introduced those derivatives**, together with the warning comment "Using the wrong dynamic pressure here inflates all four by 25%." `git log -S"67.0" -- flightsim/aircraft.py` returns exactly one commit: `d13d382`. |
| Conclusion | **The wrong-dynamic-pressure values never shipped on `master`.** The bug was caught during authoring; the git record contains no before/after numeric change for these derivatives. |
| Follow-up | `553e3e3` generalised the fix structurally, adding `FlightCondition` and `from_dimensional_longitudinal` / `_lateral` / `_controls`, with the Cherokee and Cessna retrofitted onto them. Rationale in the commit message: *"the same algebra was hand-transcribed three times and the third produced a real bug… Bundling the condition with the conversion leaves no argument to get wrong."* |
| Tolerance moved in the same commits? | **No** — neither `d13d382` nor `553e3e3` changed any tolerance. |
| Verdict | **Clean.** A class-of-bug elimination, not a number quietly adjusted. |

---

## 6. Reverts, force-pushes, amended history, stale worktrees

| Check | Result |
|---|---|
| Revert commits | **None.** No commit subject begins with `Revert`. |
| `git reflog --all` for `reset` / `rebase` / `amend` / `filter` | **No entries.** The reflog is 30 entries of `commit` and `merge` only. |
| Duplicate commit subjects (amend-and-recommit twins) | **None.** |
| `git fsck --lost-found` unreachable commits | **None.** One dangling *blob* only (a harmless orphaned object; no dangling commit or tree). |
| Merge commits | 3, all with **zero** combined-diff content — no conflict-resolution edits were smuggled into a merge. |

### Commits that never landed on `master` — exactly 3

| Commit | Date | Branch | Content | Assessment |
|---|---|---|---|---|
| **`d9d4442`** | 2026-08-14 | `claude/jolly-bhaskara-def594` | `docs/ASSUMPTIONS.md` only, **+70/−6** | **F2 — a real abandoned finding. See below.** |
| `22ab478` | 2026-08-11 | `claude/flight-dynamics-email-3601f2` | "WIP: summary PDF regenerated to 13 pages, automated line counts" | WIP, self-labelled; superseded |
| `e3822ad` | 2026-08-11 | `claude/flight-dynamics-validation-67fa59` | "WIP: manoeuvring case and Fig 8 analysis **(superseded on master by 099bd33)**" | WIP, self-labelled as superseded in its own subject; `UPDRAFT_W0`/`UPDRAFT_SECONDS` values identical to master |

### F2 — the abandoned finding in `d9d4442`

`d9d4442`, "Bound the thrust pitching moment, and find the table CR-2144 does not have",
is a docs-only commit that:

- identifies a **new primary source** for the 747 — NASA CR-114494 / Boeing D6-30643 Vol. II
  (Hanke & Nordwall, Sept 1970, NTRS 19730001300, public domain) — and cross-validates it
  against `aircraft.py`'s existing CR-2144 Table IX-3 figures digit for digit
  (5500 ft², 27.31 ft, 195.68 ft);
- resolves a **conflict between two printings** of that source's moment-arm table
  (p. 1.1-3 gives Z_EI 14.6 / Z_EO 5.4 ft, stamped "SEE SECTION 19 FOR REVISED DATA";
  p. 19.0-2 gives 8.3 / 3.1), establishing the revised pair as operative — taking the first
  at face value would have **overstated the bound by 1.75×**;
- notes that the commonly quoted 39.6 / 69.4 ft figures are the **yawing** arms Y_EI/Y_EO,
  the wrong ones for this purpose;
- quantifies the bound: 323 kN·m nose-up, dCm +0.00906, 0.379 deg of elevator once re-trimmed
  — 1.5% of authority; worst §4 tolerance consumption 10.7% (spiral tau).

**It never landed.** `master`'s `docs/ASSUMPTIONS.md` today still reads, at lines 267–269:

> **No thrust moment.** … Unmodelled, and unquantified — **CR-2144 does not tabulate a
> thrust-line offset.**

and line 572 still classes C5 as "sound for now". The unlanded commit's own diff shows those
are the exact lines it replaced. The statement on `master` is literally true of CR-2144 and
— per this abandoned work — false of the document CR-2144 draws its 747 data from. The
research was done, recorded, and left on a branch.

*Note: `d9d4442` touched no flight-model code, so nothing about the simulator's numbers
depends on it. The loss is the source research and the quantified bound.*

### Stale worktrees (`.claude/worktrees/`)

| Worktree | HEAD | Ahead/behind master | Uncommitted |
|---|---|---|---|
| `jolly-bhaskara-def594` | `d9d4442` | 1 / 32 | clean — **holds F2** |
| `flight-dynamics-validation-e9bc67` | `4947dde` | merged | clean |
| `flightsim-sweep-ui-graphs-d4d036` | `98dd6f0` | 0 / 0 | **4 files modified** (see below) |
| `sleepy-moore-7186bb` | `80c9d00` | 0 / 2 | clean |
| `strip-loads-6dof-df502b` | `a0d791d` | 0 / 5 | clean |
| `vortex-plane-interaction-b1b7d5` | `b984eb1` | 0 / 13 | clean |
| `free-air-flying-interface-98ef5f` | — | no `.git` | empty shell |

The one worktree with uncommitted work, `flightsim-sweep-ui-graphs-d4d036`
(`docs/PROJECT.md`, `flightsim/analysis/figures.py`, `flightsim/apps/sweep.py`,
`flightsim/tests/test_figures.py`, +353/−46), is **in-progress figure-layout work**. Its
test diff is **entirely additive** — new assertions on title bands, legend anchoring, axis
overlap and cursor placement. **No tolerance is loosened and no assertion is removed.** Not
a hidden finding.

Branches `claude/flightsim-sweep-ui-graphs-d4d036`, `claude/sleepy-moore-7186bb`,
`claude/strip-loads-6dof-df502b`, `claude/verification-plan-500fc9`,
`claude/vortex-plane-interaction-b1b7d5` are all **0 ahead** of master — fully merged, nothing
hidden.

---

## 7. F1 — `PROJECT.md` §2's build claim vs. what `test_provenance.py` enforces

**The claim.** `docs/PROJECT.md` states it twice:

- line 61: *"`provenance.py` | the **ledger**… | enforced by `test_provenance.py`; **a constant
  with no entry fails the build.**"*
- line 128: *"**A constant added without a ledger entry fails the build.** This is the
  `Reference.source` rule generalised from published reference values to every number in the
  model."*

`flightsim/tests/test_provenance.py`'s own module docstring (lines 6–7) repeats it:
*"a constant added without a ledger entry fails the build."*

**Provenance of the claim:**

| Where | First commit |
|---|---|
| `test_provenance.py` docstring | `084d045` 2026-08-14 "Add the provenance ledger and the rules it must satisfy" |
| `docs/PROJECT.md` §2 (both places) | `150091c` 2026-08-15 "Close ASSUMPTIONS E2 with its measured bound, and record the new checks" |

**What the code has ever actually enforced.** `test_provenance.py` has been touched by three
commits (`084d045`, `db67f4a`, `291dc01`, all 2026-08-14) and contains eight tests. Every one
of them either iterates `LEDGER.items()` or indexes a specific known key:

| Test | What it checks | Direction |
|---|---|---|
| `test_every_entry_uses_one_of_the_four_categories` | over `LEDGER.items()` | ledger → itself |
| `test_sourced_and_declared_entries_carry_a_usable_detail` | over `LEDGER.items()` | ledger → itself |
| `test_derived_and_calibrated_entries_name_inputs_that_exist` | over `LEDGER.items()` | ledger → ledger |
| `test_the_dependency_graph_has_no_cycles` | over `LEDGER` | ledger → ledger |
| `test_an_entry_with_an_unknown_category_is_rejected` | constructs a bad `Entry` | ledger → itself |
| `test_the_747_reference_geometry_is_sourced_from_cr2144` | `LEDGER["b747.S"/"b747.b"/"b747.c"]` | named keys |
| `test_the_effective_tail_arm_is_derived_and_never_sourced` | `LEDGER["b747.l_eff"]` | named key |
| `test_the_calibrated_lift_slope_names_the_number_it_is_pinned_to` | `LEDGER["strip.lift_slope"]` | named key |

**There has never been a test that walks the source modules and asserts that each constant
appears in `LEDGER`.** The required direction — *source → ledger* — is not implemented and
never has been. `flightsim/provenance.py` imports only `typing.NamedTuple`; it contains no
`inspect`, no `ast`, no `getmembers`, no module walking, and nothing in the package outside
`provenance.py` reads `LEDGER` except two explanatory comments in `airframe.py`.

**Consequence, measured.** The ledger holds **14 entries**. Of the constants this audit
tracked, only 2 are covered:

| Constant | In ledger? |
|---|---|
| `airframe.TAIL_ARM_BAND` | yes (`airframe.tail_arm_band`) |
| `airframe.calibrated_lift_slope` | yes (`strip.lift_slope`) |
| `airframe.N_SPAN` / `N_LON` | as `strip.n_stations` |
| `airframe._SENSITIVITY_TAPER` | **no** |
| `wind.LEE_WAVE_WAVELENGTH` | **no** |
| `wind.UPDRAFT_W0` / `UPDRAFT_SECONDS` | **no** |
| updraft `sharpness` | **no** (though labelled DECLARED in comments) |
| `aero.V_MIN` | **no** |
| `trim.ALPHA_LIMIT` | **no** |
| `trim.INITIAL_GUESS` | **no** |
| `validation.TRIM_RESIDUAL_LIMIT` | **no** |

Adding any of these today would not fail the build.

**Verdict.** The enforcement claim is **aspirational, not implemented** — the exact failure
mode `test_provenance.py`'s own docstring says the file exists to prevent
(*"enforceable rather than aspirational"*). This is a documentation defect, not a physics
defect: the 14 entries that *are* in the ledger are well-formed, acyclic, and correctly
categorised, and the ledger's coverage of the strip-loads work it was built for is complete.
The overreach is the words "every number in the model."

---

## Verdict summary

| # | Item | Verdict |
|---|---|---|
| 1 | Tolerances in `flightsim/tests/` | **Clean.** 1 of ~475 tolerance sites ever changed; that change was net-tightening, documented, and shipped with no change to the code under test. All 591 deleted lines across 53 commits examined. |
| 2 | Fitted / calibrated / declared constants | **Clean.** 17 constants traced; **zero** value changes. Two relocations and one literal→constant extraction, all value-preserving. |
| 3 | Reference values | **Clean.** Zero edits. The three reference test files have not been touched since the day they were written. |
| 4 | Deleted / renamed / skipped / weakened tests | **Clean.** No file deleted; 18 moved, 1 renamed, 1 tautology deliberately deleted with a documented and verified replacement. No `mark.skip`/`xfail` ever added. The 1 skipped test has a compensating positive assertion. |
| 5 | Known transcription fixes | **Clean.** `Mq` fix recorded with both moved numbers; no tolerance touched. Cessna dynamic-pressure bug never shipped on `master`. Minor stale commentary in `test_wind.py`. |
| 6 | Reverts / force-push / amended history | **Clean** — no reverts, no rewrites, no dangling commits, merges empty. **But F2:** one substantive sourced finding (`d9d4442`) never landed, and `master` still carries the claim it disproves. |
| 7 | Docs claim vs. implementation | **F1 — claim never implemented.** "A constant with no ledger entry fails the build" is unenforced and always has been. |

**On the standing rule: it was not broken.** Across 86 commits, 9 branches, 53 test-touching
commits and 591 deleted test lines, there is no instance of a tolerance being edited to make
a test pass, and no instance of a reference value being edited at all.
