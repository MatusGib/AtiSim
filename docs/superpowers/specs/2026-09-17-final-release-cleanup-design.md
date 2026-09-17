# Final release cleanup — design

**Written:** 17 September 2026, session 32. **Deadline:** 30 September 2026.
**Scope decided by the project owner**, in answer to questions asked before any work started.
The answers are recorded here because they are the reason the plan has the shape it has, and a
later session that disagrees with the shape should disagree with the answers.

| Question | Answer |
|---|---|
| What must "close all the unresolved gaps" mean, in 13 days? | **Everything, including the WGS-84 merge.** The alternative — close the *record* and merge only what is cheap — was recommended and declined |
| Which documentation toolchain? | **Sphinx + autodoc**, published to GitHub Pages. Doxygen was asked for, and declined on the evidence below |
| Who is the repository for? | **A public open-source release**, MIT licensed |
| What should the runnable notebook cover? | **The validation ladder, end to end**, nbval-gated in CI |
| **Amended, later on day 1:** repository visibility | **Private now, public when done.** Going private immediately closes the copyright exposure; the Phase 6 history rewrite **still happens**, because the copyrighted PDFs are in history and going public again would re-expose them |
| **Amended:** history strategy | **Clean up in place.** Keep `MatusGib/AtiSim`, its 183 commits and its PR record. A fresh repository was offered and declined |
| **Amended:** `CLAUDE.md` | **Moved to `docs/DEVELOPMENT.md`.** Off the front page entirely. The cost was stated before the choice and is recorded in the file itself: automated tooling no longer loads it, so rules 1 and 1b lose continuous enforcement |

### Amendment, day 1: the repository must read as *finished*

**Added requirement, and it changes Phase 4 substantially.** The repository must not look like a
work in progress conducted in numbered sessions. "Include the main changes in the updates but
not every single session."

**What this does NOT mean.** It does not mean deleting the record. `PROJECT.md` §4 is the
project's principal asset and §5's negative results are worth more than most of the code;
rule 1's whole point is that a result not written down was never obtained. **The session
*narrative* is what goes, not the evidence.**

**Measured before deciding, because the assumption was that the commit log was the problem:**

| | |
|---|---|
| Commits on `main` | **183** |
| Subject lines mentioning a session | **8** |
| Subject lines mentioning Claude | **0** |

**So the history is not the problem** — it reads as engineering work, with subjects like "Read
the DC-10's altitude from Parks Fig. 6: it did not climb before the pair". That is the evidence
against starting a fresh repository: 183 such commits are credibility, and one `initial commit`
of 47,000 lines is a dump. The 8 subjects are reworded in the Phase 6 `filter-repo` pass, which
was already scheduled for the PDFs.

**The real tells, and where each is handled:**

| Tell | Disposition |
|---|---|
| `FIX_PROMPT.md` (31 KB), `analysis-ui-investigation-prompt.md`, `audit/AUDIT_PROMPT.md` — prompts written to an AI, referenced by nothing | **DONE, day 1.** Deleted |
| `CLAUDE.md` at the root | **DONE, day 1.** → `docs/DEVELOPMENT.md`, 36 cross-references updated |
| No licence, `license: null` on a public repo | **DONE, day 1.** MIT |
| Repo description "Jax based vortex flight simulator ", no topics | **DONE, day 1.** Both set |
| `docs/superpowers/` — the directory name | **Phase 4c.** → `docs/design/`. **Deferred deliberately**: three unmerged branches modify files under it, so a rename now conflicts with all three |
| `docs/SESSION_27_REPORT.md` | **Phase 4c.** Its closing section states the validation claim more plainly than anything else in the repository. **Harvest that prose into the docs site, then delete the file** — not the other way round |
| `PROJECT.md` §9, 32 session entries | **Phase 4c.** Becomes `CHANGELOG.md` keyed to **capabilities, not sessions** — the frame, the CAT validation, the sensitivity study, the compressibility work. §9 itself is kept, moved behind the docs site as the provenance appendix |
| `PROJECT.md` §0, a work-in-progress inventory | **Phase 7.** Empty at the release, or every remaining row justified. §0 going empty *is* the finished signal |
| README's "I asked claude to add test", "23 scripts" (there are 42) | **Phase 4b** |
| No release, no tag | **Phase 7.** `v1.0.0` |

**One ordering consequence.** Structural renames conflict with unmerged branches, so the rule is:
**delete freely what nothing else touches, and defer every rename until after Phase 2.** Day 1's
moves were all checked against all seven substantial branches first; `docs/superpowers/` failed
that check and waits.

**Why not Doxygen, since it was asked for.** `atisim` is 31,632 lines of Python carrying
*narrative* docstrings — `dynamics.gravity` spends six lines naming the effects it omits and
how large each is. Doxygen's Python support parses neither docstring markup nor type hints, so
that prose renders as flat unformatted text; getting good output would mean adding
`\param`/`\return` blocks across modules that `docs/DEVELOPMENT.md` §3 puts off-limits to casual edits,
duplicating docstrings that already exist. Sphinx + autodoc renders what is already written.

---

## 0. The state this plan starts from

Measured 17 September 2026, on `main` at `c1b7d71`, which is identical to `origin/main`.

- **10 working days**: Thu 17, Fri 18, Mon 21 – Fri 25, Mon 28 – Wed 30 September. Weekends
  are buffer and are not scheduled.
- **23 local branches**, 16 with unique commits. **14 worktrees.**
- **Two worktrees hold uncommitted work.** `project-md-restructure-95b7b8` holds **49 files**:
  modifications to `atisim/aircraft.py` and `atisim/dynamics.py`, a new
  `atisim/tests/test_alphadot_derivatives.py`, a `.pptx` deck, and **14 untracked scripts**
  including `alphadot_conversion.py`, `alphadot_isolate.py`, `galilean_alphadot_probe.py`,
  `prandtl_glauert_check.py`, `e4_windhold_remeasure.py` and `recapture_fig8_pins.py`. The
  main checkout holds 5 untracked reference PDFs. **This is the failure `docs/DEVELOPMENT.md` rule 1b
  exists to prevent, live, for the third recorded time.**
- **`docs/PROJECT.md` is 604 KB / 8,096 lines.** An excellent standing record and not an
  entry point.
- **The repository is already public** (`private: false`) and **carries no licence**
  (`license: null`), so it is "all rights reserved" by default.
- **The suite** was 906 passed, 1 skipped, 1,534 s at the end of session 30 (§10).

### The two WGS-84 branches — §0's open question, answered

`PROJECT.md` §0 records that "which of the two is the trunk is not recorded anywhere and must
be established before either is merged". **Established here: neither is.** They diverge at
`01d7502` (28 Aug) after 61 shared commits, and each carries unique work.

| Branch | Unique commits | What only it has |
|---|---|---|
| `claude/wgs84-earth-rotation-tasks-5dbdc3` | 4, latest 31 Aug | Measures what FLAT costs against the pre-Earth model; **re-measures the §4 evidence ledger on the rotating Earth**; linearises about the equilibrium and **closes three red tests**; fixes seven docstrings that claimed a red suite |
| `claude/atisim-wgs84-earth-rotation-32fbdd` | 7, latest 28 Aug | **Retires `ASSUMPTIONS.md` A1 and A2** and revisits A3; re-measures F4 for the ECEF state; re-captures the Fig-8 vortex pin; linearises validation **about the transport rate**; the architecture and audit-inventory updates |

**The merge is therefore the union**, and `git merge-tree` reports **8 files changed in both**:
`atisim/validation.py`, `atisim/tests/test_validation.py`,
`atisim/tests/test_vortex_viz.py`, `atisim/tests/test_jsbsim_737_layers.py`,
`audit/INVENTORY.md`, `audit/NOTATION.md`, `docs/ASSUMPTIONS.md`, `docs/PROJECT.md`.
**`atisim/validation.py` is the hard one**: both branches linearise it independently, one
about the transport rate and one about the equilibrium, and whether those compose or contradict
is not known and is Phase 2a's first question.

### The copyright blocker

**14 third-party PDFs are tracked in git history**; `size-pack` is 90.83 MiB. NASA and USAF
documents (`19890016606.pdf`, `19910009769.pdf`, `AFFDL-TR-70-101-…`) are US Government works
and may be redistributed. These may not:

| File | Publisher |
|---|---|
| `parks-1985-identification-of-vortex-induced-clear-air-turbulence-JA22-2.pdf` | AIAA, *J. Aircraft* 22(2) |
| `bach-parks-1987-angle-of-attack-estimation-JA24-11.pdf` | AIAA, *J. Aircraft* 24(11) |
| `1520-0493_1989_117_1103_tuodii_2_0_co_2.pdf` | AMS, *Mon. Wea. Rev.* 117 |
| `Ger/loving-2012-…`, `Ger/schultz-2012-…`, `Ger/apme-JAMC-D-21-0071.1.pdf` (x2) | AMS |
| `Geophysical Research Letters - 2023 - Yoshimura - …pdf` | AGU/Wiley — **check, GRL may be CC BY** |
| `mehta-2012-modeling-clear-air-turbulence-with-vortices-…pdf` | to be established |
| `measurement-integrated-simulation-of-clear-air-turbulence-…pdf` | to be established |

`.gitignore` already reasons this out correctly for `refs/` — "deliberately kept out of
history: they are re-downloadable and mostly not ours to redistribute" — and
`Reference_papers/` then does the opposite. **The repository being public makes this live
rather than prospective.**

---

## 1. The three constraints that set the order

1. **Ten working days**, with no slack. Phase 2 is four of them.
2. **The WGS-84 merge re-baselines every frame-dependent number in §4.** So the documentation
   and the notebook must **compute or cross-reference** every figure they show, never quote a
   literal. Built that way they can be written in parallel with Phase 2 and the re-baseline
   flows through them; built the other way, Phase 2 invalidates them on the last day. This is
   what `docs/DEVELOPMENT.md` §2 and §6 already require, so the constraint costs nothing.
3. **A history rewrite invalidates every unmerged branch**, so it must come *after* all
   merging. It is also irreversible and force-pushes a public repository.

**An interim mitigation is available on day 1 and is taken:** deleting the copyrighted PDFs
from the working tree and gitignoring them stops GitHub serving them from `main` immediately,
while history waits for the Tue 29 rewrite.

---

## 2. Phases, with gates

Every phase ends on a measured gate, not on a judgement.

### Phase 0 — Stop the bleeding (Thu 17, day 1)

1. Commit `project-md-restructure-95b7b8`'s 49 files **on their own branch, unreviewed and
   labelled as such** in the commit message — the session-28 precedent, which exists so that
   rescuing work is never confused with endorsing it.
2. Adjudicate the main checkout's 5 untracked reference PDFs against the copyright table above
   before adding any of them.
3. **Delete the copyrighted PDFs from the tree and gitignore `Reference_papers/`**, with the
   per-file reasoning recorded.
4. Write a `PROJECT.md` §0 row for every branch carrying unique commits.

**Gate:** `git status --porcelain` is empty in all 14 worktrees; the rule-1b two-line check
prints nothing that §0 does not have a row for.

### Phase 1 — Branch triage (Fri 18, day 2)

1. Delete the 6 branches already merged into `main`, and the 2 `WIP:` branches §0 names as the
   safe deletions — **after** checking each attached worktree for uncommitted files.
2. Merge the five cheap branches (1–3 commits each).
3. **Harvest `claude/weekly-summary-analysis-7520db`'s `runs/cat/` outputs** before anything
   touches it: they are gitignored, exist only in that worktree, and are what §4's session-28
   POD row reads.
4. Review and either merge or abandon the four rescued-but-unendorsed branches, chief among
   them `claude/zen-maxwell-1ad0a4`'s tail-arm gate refactor.
5. Take the two decisions §0 says the project has never taken: whether `old-origin`
   (`MatusGib/Flight_sim`) still exists, and whether `docs/DEVELOPMENT.md` becomes `AGENTS.md`.

**Gate:** the two-line check prints only the WGS-84 pair and anything deliberately kept, each
with a §0 row; suite green on `main`.

### Phase 2 — WGS-84 (Mon 21 – Thu 24, days 3–6)

**2a, Mon 21.** Build the union on a fresh branch `wgs84-earth`. Reconcile the 8 overlapping
files. Answer `atisim/validation.py`'s question first: do the two linearisations compose, or
does one supersede the other? A red suite is expected here and is enumerated rather than
fixed.

**2b, Tue 22.** Rebase onto current `main`, which now carries sessions 29 and 30 — the CR-2144
speed derivatives, the declared thrust line and the replayed Hannibal headline, all of which
are frame-sensitive. Drive the suite green.

> **GATE, end of Tue 22 — go/no-go, agreed in advance so it is not a judgement call under
> deadline pressure.** If the suite is not green on the union by end of day:
> **do not merge.** Push `wgs84-earth` with the reconciliation notes as a tracked document,
> write the §0 row naming which of the 8 files resisted and why, and spend Wed 23 – Thu 24 on
> Phases 3–5 instead. **A half-merged inertial frame is worse than an unmerged one**, and the
> month still ships a clean, documented, runnable repository.

**2c, Wed 23 – Thu 24.** The re-baseline. Every frame-dependent §4 row re-measured and
**superseded in place, never deleted** (§4's own rule). `ASSUMPTIONS.md` A1 and A2 retired, A3
revisited. §1's validation claim reconciled with whatever §4 now says — `docs/DEVELOPMENT.md` rule 1.4
exists because §1 once carried a stale 67% for several sessions.

**Gate:** suite green; no §4 row deleted; §1 and §4 agree on the headline; A1/A2 retired in
`ASSUMPTIONS.md` and in its summary table.

### Phase 3 — Gaps to terminal status (Fri 25, day 7)

Every open item in §5, §7 and §8 gets one of exactly three statuses, with the evidence:

- **measured** — a §4 row,
- **impossible with sources held** — naming *which* source failed and *why*, per `docs/DEVELOPMENT.md` §2,
- **abandoned** — a §9 entry saying why, per rule 1b.

Includes: assessing the 5 new DC-10/MD-11 papers against the one genuinely-open acquisition
(a DC-10 cruise derivative set); and settling or explicitly re-sealing `predictions.py`'s one
remaining open bet, the DC-10 entry that bets against the project's own story.

**Gate:** zero items in §5, §7 or §8 without a terminal status; no SEALED prediction sitting
past its trigger un-adjudicated.

### Phase 4 — Documentation (parallel: Fri 18 pm, Wed 23 – Thu 24, Tue 29)

**4a, Fri 18 pm.** Sphinx + autodoc scaffold; GitHub Actions workflow; Pages; `LICENSE` (MIT);
root clutter out — `FIX_PROMPT.md`, `analysis-ui-investigation-prompt.md`,
`audit/AUDIT_PROMPT.md`, `docs/SESSION_27_REPORT.md`.

**4b, Wed 23 – Thu 24**, in the gaps while suite runs and re-measures execute. README rewritten
in a neutral third-person voice: the licence stated, the script count correct (42 in `scripts/`
against the README's "23"), the test count read from a real run rather than asserted, and the
"I asked claude to add test" line replaced with what the suite actually gates.
`CONTRIBUTING.md`. A docstring sweep for autodoc — additive only, no rewriting of existing
prose.

**4c, Tue 29.** Site content: the architecture page, a running-it page generated from §10's
table, and the validation-claim page carrying §1's envelope. `PROJECT.md` keeps all 8,096
lines and gains an entry point in front of it; it is not shortened.

**Gate:** docs build green in CI from a clean clone; every number on the site is computed or
carries a §4 cross-reference; no literal results.

### Phase 5 — The notebook (Mon 28, day 8)

`notebooks/validation-ladder.ipynb`, structured like §4 and reusing the existing scripts rather
than restating them: degenerate-input sanity, then published-source comparisons (CR-2144
modes, the drag polar, the JSBSim 737), then the Hannibal CAT headline with its envelope
attached, then Fig. 8's ordering. **Every number computed in-cell.**
`notebooks/solver-validation.ipynb` is left untouched as the solver-level gate.

**Gate:** `pytest --nbval-lax notebooks/ -q` green in CI on a clean clone. Per §10, run from
the worktree root with an **absolute** `PYTHONPATH` — nbval starts its kernel in `notebooks/`
and a relative path silently loads the main checkout.

### Phase 6 — History rewrite (Tue 29, day 9, after 4c)

`git filter-repo` to strip the copyrighted PDFs from all history. Force-push. Re-verify.
**Last among structural steps**, because it invalidates every branch not yet merged.

**Gate:** a clean clone is under 15 MiB; no copyrighted PDF appears in any commit; the suite is
still green on the rewritten history.

### Phase 7 — Ship (Wed 30, day 10)

Final §9 session entry. §0 empty, or every remaining row carrying its branch, worktree, state
and blocker. Tag `v1.0.0`.

**Gate — the only one that matters to a reader:** a stranger clones the repository, follows the
README, and gets a green suite and a rendered documentation site.

---

## 3. The execution loop

Every phase runs this, because `docs/DEVELOPMENT.md` rules 1, 1b and 4 require it:

```
for each phase:
  0. VERIFY THE TREE   .venv/Scripts/python.exe -c "import atisim; print(atisim.__file__)"
                       -> abort if that is not the tree you edited (docs/DEVELOPMENT.md rule 4)
  1. STATE THE GATE    write the success criterion down BEFORE starting
  2. DO THE WORK       surgical; every changed line traces to the gate
  3. MEASURE           run the gate command; capture the real output
  4. RECORD            PROJECT.md section 4 (numbers) / 5 (gaps) / 6 (bugs) / 9 (log)
                       -> supersede in place, never delete a row
  5. TRACK             git add every script; `git status` shows no '??' lines
  6. SECTION 0         one row per unfinished thing, with BRANCH NAME and WORKTREE
  7. COMMIT            then, and only then, advance
```

---

## 4. Out of scope, deliberately

Named here so a later session does not go looking, per `docs/DEVELOPMENT.md` rule 1.3.

- **No new physics beyond what closes a gap.** The `boeing787_yoshimura` registry entry, Parks
  1985 Fig. 6's pitch and true-airspeed channels, and Hannibal inventory items 2–5 get a
  terminal status in §5/§7 — not an implementation. **Unless Phase 2 aborts**, which frees
  Wed 23 – Thu 24 and makes them the first candidates.
- **No tolerance edited to make a test pass** (`docs/DEVELOPMENT.md` §3). If a §4 baseline file moves
  during the re-baseline, that is Phase 2c's finding, recorded with the cause, not smoothed.
- **`PROJECT.md` is not shortened.** All 8,096 lines stay.
- **Doxygen is not used**, for the reasons at the top.
- **The Cessna stays out of every conclusion** (`docs/DEVELOPMENT.md` §6).
