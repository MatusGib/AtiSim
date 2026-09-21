# AtiSim — project record

A 6-DOF fixed-wing flight-dynamics core in JAX, built as a foundation for turbulence
modelling. This document is the standing record: what exists, what is validated, what is
known-broken, and what happens next.

**Last updated:** session 32 — **the run to the 30 September deadline.** The plan is
`docs/design/specs/2026-09-17-final-release-cleanup-design.md`. Findings that change what
the rest of it has to do: **the repository is public and was redistributing seven copyrighted
papers** (now untracked, with `Reference_papers/SOURCES.md` as the index and the history rewrite
scheduled after the merges); **rule 1b failed three more times** — the α̇ work stranded in a
worktree, a full **automated digitisation of CR-2144 pp. 218–228** nobody had tracked, and
**a branch this file did not mention once**, which is session 31's below and is **now merged**;
**§0's claim about `main` was stale** and inflated every count in that section; and **§0's
question about the two WGS-84 branches is answered — neither is the trunk, the merge is the
union**, with `atisim/validation.py` deciding its cost. §9's session-32 entry has all of it.

**Then every open item was given one status** — the table at the head of §5 is where to start. Doing it turned up **one open latent bug, §6(h): the strip load path counts the gust's rolling moment twice**, so the strip path's published +22.9% on peak bank is −2.5% with one path owning roll; nothing longitudinal and nothing in §1 moves. And **the speed derivatives the 747 declares were checked from outside their own reading for the first time** — against an independent trace of the same pages, adjudicated on Table IX-4 — **and they hold**. The rotating-Earth union is banked, not merged, so A1 (flat, non-rotating Earth) stays open.

**Session 31** (an investigation, no model code changed). **Session 29's `mass` row is right as arithmetic and wrong as a reading.** `mass` and `CLa` are one lever, mirrored to 0.04. The row does not price an error in CR-2144's weight, which cancels to +0.008 because the builder derives `CLa` from it. Holding inertia fixed moves the load elasticity by −0.36 to +0.19, and the sign depends on the flight form (as flown, or replayed on the identified path). The cruise phugoid-damping "`mass` +1.728" is mostly the DECLARED Korn drag rise, +0.447 without it. §4's first entry has the tables. **It also identified the Hannibal aircraft** — United Airlines 12, **N1809U, a DC-10-10** — which narrows §5's wing-loading ratio to **0.584–1.046×** and removes the 1.3× corner. The weight on the day is still not found.

**Session 30.** CR-2144's speed derivatives were digitised, checked against
Table IX-4 at every circled condition, and **declared on `boeing747`** through a new
`Aircraft` seam. **Phugoid ω_n goes −18.1% → +4.05%**; the residual is the engine's missing
thrust line, not the reading — **and then a thrust line was declared too**, at CR-114494's
revised **5.70 ft** rather than CR-2144's 10 ft: **phugoid ω_n +1.69%, ζ +2.83%** against Table
IX-5 (10 ft would read −0.05% / +1.13%; an arm is not chosen by its answer). The price is recorded, not hidden: the CAT headline
moves **68.2% → 64.5%** of the record (64.4% with the thrust line), and three mechanism claims
weaken, all through Cm_M.
Then TM-102186 Fig. 7's **horizontal wind** was digitised. It **confirms which side of the path
each core sits on** (1.58 kt RMS as built, against 15.38 flipped). It also shows the headline
**double-counts the 747's own climb** past cores 3–4: replaying the fitted wind along the
nominal path reads **75.0%** (75.3% with the thrust line). Parks 1985 Fig. 6's altitude, digitised, then showed the DC-10
**did not climb** through the pair: its inertial estimate was 36,985–36,996 ft at cores 3 and
4, while the simulated 747 had climbed 500–600 ft. So the replayed form is the right reading,
and **the headline now flies it: 75.3% of the record.** Measuring that **reversed session 23c's
saturation result**: replayed, the load tracks the gust, and a vortex strength 15% above
Mehta's fit reaches the recorded +1.7 g inside the linear range (§4, first entry).
Session 29 was merged from `main` into this branch before its pull request; its results are §4's session-29 entries, and they were measured on the 747 as it stood before session 30 and flown at its own altitude.

Session 29 (**the sensitivity study, designed and then run end to end, S0–S6**, plus two follow-ups. **ASSUMPTIONS C3's Mach axis is bounded at cruise, −5.04%**; **§1's headline carries a band — 68.2%, 57.1–74.0% — and the shortfall survives all of it**; `CLa` +0.692 and `mass` −0.649 lead the load, with a DECLARED constant third — ~~two leads~~ **one lever counted twice, session 31**. **Two `sqrt(0)` bugs repaired** — the model was not differentiable in its own coefficients, and no quoted result was ever wrong. And **session 23d's Fig. 8 reading is CORRECTED**: the pitch axis does not stop discriminating, the extremes gap it used shrinks with N, and it is the load axis that degrades faster)

Session 28 (an audit, no code changed: the "the agreement got worse" hypothesis tested and **falsified** — every early number re-measured and unchanged, no tolerance ever loosened, and the growth traced to a change of *reference class* dated to commit `c6b5342`, 1 Sep 2026, with ASSUMPTIONS C3's frozen derivatives the largest identified physical cause; the strip-load path judged: off the published path, +22.9% on one unvalidated channel, and worth keeping for its negative result — **a verdict session 32 strengthened: the +22.9% was the gust's roll counted twice, §6(h)**).

Session 27 (the recorded trace digitised, and it says the *wind* is 12% light; a second sealed prediction settled RIGHT; the DC-10 wing loading found unpinnable, withdrawing session 26's sign; the LES comparison audited, refused, then re-run with Yoshimura's OWN aeroplane rebuilt from their source code -- 1.427 to 1.202, with the residual now attributable to neither aircraft nor Mach).

**To run any of it, see §10.**

---

## How to update this document

This file is split into **stable** sections (§1–§5) and **volatile** ones (§6–§9). A
session normally edits only the volatile ones.

| When you… | Edit |
|---|---|
| **leave anything unfinished** | **§0 work in progress — one row, with the BRANCH NAME and the worktree. `docs/DEVELOPMENT.md` rule 1b** |
| finish any session | §9 session log — add an entry at the top |
| land a new module or change a public API | §2 architecture **and** §10 running it |
| add or change a script, flag or key binding | §10 running it |
| measure a number against a source | §4 evidence ledger — never delete a row, supersede it |
| find a gap the source cannot fill | §5 gaps — say which source failed and why |
| find a bug that does not fail a test | §6 latent bugs |
| complete or re-order planned work | §7 plan |
| discover something that changes the approach | §8 open questions |

> **`docs/DEVELOPMENT.md` is the enforcing document, and rule 1 there is the
> one this file depends on: NOTHING IS DONE UNTIL IT IS IN THIS FILE.** Session 27 found
> two completed digitisations stranded in another worktree — untracked scripts, gitignored
> outputs — while that worktree's copy of this document still listed both as open work. **Session
> 32 found three more caches the same way**, one of them a full digitisation of the project's
> primary aerodynamic source and one a branch this file did not mention at all. A
> measurement that is not written down here will be paid for twice. Before ending a session,
> add the §9 entry, `git add` the scripts you wrote, and record what you deliberately did
> *not* do.
>
> **That document was `CLAUDE.md` at the repository root until the September 2026 release**, and
> was loaded automatically at the start of every working session. It is now ordinary
> documentation, which means **rule 1 is no longer enforced by anything but the reader**. Until
> session 27 it did not exist at all, though this line referenced it.

Rules carried from `docs/DEVELOPMENT.md` and enforced throughout the code:

- **Check that the tree you are testing is the tree you edited.** Every worktree shares the
  main checkout's `.venv`, whose editable install maps `atisim` to the **main checkout**
  for the life of the install. Nothing warns you when that mapping wins — the tests import,
  collect and pass, against code you did not change, and the result reads exactly like a
  real one. Print `atisim.__file__` before believing any run. §10 carries the one-line
  check and the cases it catches.
- **Flag, never invent.** Every number carries the table it came from. A parameter the
  source does not supply is named as a declared modelling choice, not given a plausible
  default. If you add a number here without a citation, you have broken the project.
- **Do not edit a tolerance to make a test pass.** §4's "validated baseline" files are
  off-limits to feature work; if one moves, something real broke.

---

## 0. Work in progress, and where it lives

> **Required by `docs/DEVELOPMENT.md` rule 1b.** Rule 1 covers work that is done. This section covers
> work that is **not**, because an unmerged branch is indistinguishable from a branch nobody
> wrote. **A row without a branch name is not a row.** Worktree directory names do NOT match
> branch names in this repo — `turbulence-research-sources-39f87e` holds
> `claude/zen-maxwell-1ad0a4` — so both are given. Delete a row only when the work is merged
> or abandoned, and abandoning is a §9 entry saying why.
>
> **Established session 28 by auditing all 39 branches and 15 worktrees**, after this
> conversation asked whether the session-28 recommendations had already been done elsewhere.
> **Three of the five had been, in whole or in part, and nothing in this document said so.**
> Suite status is `not run` on every row below unless stated: nothing here was checked out
> and tested this session.

**At the release, 20 September 2026, two rows are still live.** Everything else in this section
is struck through or marked merged and is kept as the record of how it was found.

| Still live | Where it is, after the history rewrite |
|---|---|
| **The WGS-84 union, `wgs84-earth`** | **In this repository**, rewritten with the rest of the history. `main` is a flat, non-rotating Earth — `ASSUMPTIONS.md` A1, and the README says so before the install instructions — and the union is one branch away, measured and banked rather than merged. §9, session 32, point 8 has the measurement and the decision; the row below has the address |
| **The α̇ derivatives** | **Not in this repository.** The branch was never pushed while the release was prepared, and the rewrite did not carry it — so when the local branches were cleaned up it turned out to be a **single copy, on one machine**, which this row had already claimed was also in the archive. It now is: pushed to `MatusGib/AtiSim-archive` as **`alphadot-derivatives-parked`** at `6148cd8`, renamed on the way out because a `claude/*` name is not pushed anywhere. Locally it is still `claude/engine-validity-presentation-1408e8`, checked out in `.claude/worktrees/project-md-restructure-95b7b8`. §5's status table records what it would close and what it costs; §9, point 9 records the review that parked it |

**Commit ids below predate the rewrite.** `docs/design/commit-map.txt` translates them, except
for branches deleted before it, which live only in the archive.

### ~~The one that matters~~ MERGED, session 28 — kept as the record of how it was found

> **`claude/new-session-943052` was rebased onto `main` and merged on 8 September 2026**, at
> `50b78a1` plus `993fa91`. The suite went red first (3 failed) and green after three pins
> were re-captured — §4's "The compressibility branch, rebased and merged" has what moved and
> why. **The row below is left as written**, because how a fully-built implementation sat
> unmerged for thirteen days while three sessions worked around it is the thing rule 1b
> exists to prevent, and deleting the row would delete the lesson.


| | |
|---|---|
| **Branch** | **`claude/new-session-943052`** |
| **Worktree** | `.claude/worktrees/new-session-943052` |
| **State** | **1 commit ahead of `main`, 27 behind.** Single commit `c6dba3c`, 26 August 2026. Suite not run |
| **What it is** | **A Prandtl–Glauert implementation, already built.** A `pg_mach_ref` field on `Aircraft`; the factor applied to the **whole longitudinal lift-slope family** rather than `C_Lα` alone, with `PG_MACH_MAX = 0.90` and a floor so the M → 1 singularity cannot reach the integrator; Mach-scheduled `Cmde` and `Clda` tables; **`g(z) = g₀(R/(R+z))²`**; and the **geopotential-altitude conversion** at the atmosphere boundary |
| **What it closes** | `ASSUMPTIONS.md` **C3's Mach axis** — the assumption §4 calls the dominant identified contributor to the LES discrepancy. Also **A2** (constant g, reversing session 12's decision but not its measurement) and **A3** (geometric-vs-geopotential, which that commit message calls "not an assumption, a defect": density at a nominal 30,000 ft goes from 0.159% out to **0.000479%** against JSBSim) |
| **Blocking** | **Nothing. It is unmerged.** It is 27 commits behind `main`, so it needs rebasing and a full re-measure of every altitude-dependent row in §4 — which its own commit message says it already did once, on the tree as it stood in August |
| **Read this beside it** | Its commit message reports an unforced dividend: phugoid ζ against Table IX-5 improved **14.4% → 13.2%** with nothing aimed at IX-5. And it records what it did *not* do: the **latitude** variation of gravity, 0.53%, larger than the altitude term, is still absent |

**Session 27 priced the Mach axis with a hand-rolled Prandtl–Glauert factor while this was
sitting in the repository.** §4's session-28 entry on CR-2144's `CL_α(M)` figures should be
read against it: the branch supplies the *theoretical* correction, printed p. 220 supplies
the *sourced* curve, and the two are a check on each other rather than alternatives.

### ~~Session 30's own work — unmerged~~ MERGED, session 30, at pull request #3

| | |
|---|---|
| **Branch** | **`session-30-speed-derivatives-hannibal`** — renamed from `claude/cr-2144-speed-derivative-data-5012ac` before it was pushed, per the no-`claude/*`-names rule |
| **Worktree** | `.claude/worktrees/engine-validity-error-check-d8ccdf` — **the directory name does not match the branch**, as everywhere else in this repo |
| **State** | **7 commits beyond `main`'s `4e5191e` once merged, 0 behind** — six session-30 commits (the inert seam, the declaration, the Hannibal horizontal wind, the DC-10's altitude from Parks Fig. 6, CR-2144's thrust line, then the 5.70 ft arm and the replayed headline) and a merge of `main`, which brought in session 29. Opened as a pull request. The suite on the merged tree is in §9's session-30 entry, point 14. Nothing uncommitted except the tool folder `.impeccable/` |
| **What it is** | CR-2144 printed pp. 220–222 digitised, checked against Table IX-4 at eight conditions, and flown (§4). `atisim/cr2144_mach.py` and the tracked points; the `mach_deriv_ref / CL_M / CD_M / Cm_M` seam in `aircraft.py` and `aero.py`; `scripts/cr2144_speed_derivatives.py`; 23 tests. **Then TM-102186 Fig. 7's horizontal wind** (§4): `scripts/digitise_hannibal_horizontal_wind.py`, `scripts/hannibal_along_track_wind.py`, `atisim/data/tm102186_fig7_winds.csv`, 8 tests, and `scripts/cr2144_report_figures.py` for the report figures. The horizontal-wind work changes no model code |
| **What it closes** | §5's "Mach content of `Xu`, `Zu`" stops being an attribution and becomes SOURCED, verified against Appendix A, and **declared on `boeing747`**: phugoid ω_n +4.05%, ζ +3.45% against Table IX-5 — and with a thrust line declared as well (CR-114494's revised 5.70 ft), **+1.69% and +2.83%**. It also weakens three mechanism claims, all through Cm_M, and moves the CAT headline 68.2% → 64.5% — §4, item 7 |
| **Blocking** | **Nothing — the decision was taken.** `boeing747` **declares** the FC9 set: phugoid ω_n −18.13% → **+4.05%**, ζ +13.16% → **+3.45%** against Table IX-5. What that moved elsewhere — the Fig. 8 pins, the CAT headline — is in §4, re-measured rather than predicted. ~~The horizontal-wind work leaves one question open, which path the headline should fly Mehta's cores on (§8).~~ Parks Fig. 6 answered it: on the path the fit was made along. **What remains is a decision**, whether to switch the shipped headline flight to that form. It does not block a merge |
| **Read this beside it** | The eight `.dig` originals are **tracked, in `atisim/data/cr2144_dig/`** (384 KB, each embedding the page crop it was traced on), beside the `atisim/data/cr2144_p220_222_digitised.csv` that `--dig-dir` regenerates from them |

### Reported done elsewhere, and NOT ON ORIGIN — added 16 September 2026, address found session 31

> **RESOLVED, end of session 30.** The branch this row points at was not on `origin` because it had not been pushed. It is now, as `session-30-speed-derivatives-hannibal`, merged with `main` and opened as a pull request — the row above is its record. Every result it reported is in §4, measured. This row is kept as written.

| | |
|---|---|
| **Branch** | ~~`claude/cr-2144-speed-derivative-data-5012ac`~~ **`session-30-speed-derivatives-hannibal`** — **FOUND, session 31 (17 Sep 2026).** The reported name exists nowhere. The commits do, on this LOCAL branch: `6ef5e4f`, `0d84eae`, `9d7a77f`, `4e97e8c`, `b551403`, `ea74849`. That branch's own §0 still gives the old name |
| **Worktree** | **`.claude/worktrees/engine-validity-error-check-d8ccdf`** |
| **State** | **SUPERSEDED THE SAME DAY: MERGED.** `origin/main` carries it as PR #3 (`c1b7d71`, merged 17 Sep 2026 09:59 UTC), after `993817a` merged `main` into it. The sentence that follows was true at 10:30 local time and is kept as the record of what was read. ~~**6 ahead of `origin/main`, 11 behind, and NOT PUSHED.**~~ It forks from `7b71816` and does **not** contain session 29. **A merge of `origin/main` (`4e5191e`) into it is IN PROGRESS in that worktree**: `MERGE_HEAD` was written 17 Sep 10:30, conflicts in `PROJECT.md` and `ASSUMPTIONS.md` are staged as resolved, and nothing is committed. Session 31 read it and touched nothing. Suite on the merge: not run. ~~Checked three ways on 16 Sep 2026 … Neither named commit exists~~ — true of `origin`, which is where that check looked |
| **What it is** | **The CR-2144 pp. 220–222 digitisation** — `CL_α(M)`, `Cm_α(M)`, and the `CL_M`/`CD_M`/`Cm_M` speed derivatives — reported complete, with results said to be in that branch's own §4. **Session 31 read the branch, and it holds more than that.** The speed set declared on `boeing747`; CR-2144's **thrust line** declared at CR-114494's revised 5.70 ft; TM-102186 Fig. 7's horizontal wind; Parks 1985 Fig. 6's altitude; and the Hannibal headline switched to the field **replayed on its identified path, 75.3%**. That §4 and §9 carry all of it |
| **What it closes, if it lands** | The item §4's "C3's Mach axis is not unbounded" calls **the highest-value item on the list**: it turns session 29's DECLARED Prandtl–Glauert bound into a **SOURCED** curve, gives the altitude axis a second independent check free, and supplies the Mach content §5 names as missing from the phugoid |
| **Blocking** | **The half-done merge, then a push.** Finishing the merge puts `sensitivity.py` and session 30's model in one tree for the first time. **After that, session 29's S2/S5/S6 must be re-run on it.** Every elasticity and §1's band were measured on the bare 747 and the as-flown field, and session 30 changed both. §4's first entry shows that a ±1% central difference on the replayed headline can land on a sample jump |

> **Read this row as a POINTER, not as a result.** The address is now verified; the numbers are
> not re-measured here, and none may be quoted from it except the handful §4's first entry
> re-ran on its tip `ea74849` (base loads 64.52% as flown and 75.34% replayed, which match it). §4's own status for the Mach axis is unchanged: bounded at cruise by
> session 29's *theoretical* Prandtl–Glauert factor, and still awaiting the *sourced* curve.
> **Whoever merges that branch should expect conflicts in §0, §4 and §9**, which both branches
> edit; the `aero.py` changes were reported not to overlap.

### ~~Large, stranded, and a decision rather than a merge~~ UNITED as `wgs84-earth`, session 32 — the rows below are the record

| Branch | Worktree | State | What it is |
|---|---|---|---|
| `claude/atisim-wgs84-earth-rotation-32fbdd` | `atisim-wgs84-earth-rotation-b7b3bc` | **68 ahead, 27 behind** | A **rotating WGS-84 Earth**: `earth.py` and `earth_ref.py`, an ECEF state, every test migrated, a latent autopilot sign bug found and fixed, and **`ASSUMPTIONS.md` A1 and A2 retired**. Carries its own §9 entry ("Record session 23 in the project log") |
| `claude/wgs84-earth-rotation-tasks-5dbdc3` | `sleepy-moore-7186bb` | **65 ahead, 27 behind** | **Sibling of the above, sharing commits** (`01d7502`, `84a8e53`, …). Adds "Measure what FLAT costs against the pre-Earth model, and find it is not round-off" and "Linearise about the equilibrium, and close the three red tests". ~~**Which of the two is the trunk is not recorded anywhere and must be established before either is merged**~~ **ESTABLISHED, session 32 — see below** |

**These two are not cleanup.** Merging 65–68 commits that replace the inertial frame changes
every number in §4 and retires two assumptions the validation claim rests on. That is a
session's work with a full re-baseline, and §7 has never listed it.

> **Which is the trunk? NEITHER — established session 32, and the answer is that the merge is
> the UNION.** They diverge at `01d7502` (28 August) after **61 shared commits**, and each
> carries unique work the other does not have.
>
> | | `…tasks-5dbdc3` | `…earth-rotation-32fbdd` |
> |---|---|---|
> | unique commits | **4**, latest 31 Aug | **7**, latest 28 Aug |
> | only it has | measures what FLAT costs against the pre-Earth model; **re-measures the §4 evidence ledger on the rotating Earth**; linearises **about the equilibrium** and **closes three red tests**; corrects seven docstrings that claimed a red suite | **retires `ASSUMPTIONS.md` A1 and A2** and revisits A3; re-measures F4 for the ECEF state; re-captures the Fig-8 vortex pin; linearises validation **about the transport rate**; the architecture and audit-inventory updates |
>
> **`git merge-tree` reports 8 files changed in both**: `atisim/validation.py`,
> `atisim/tests/test_validation.py`, `atisim/tests/test_vortex_viz.py`,
> `atisim/tests/test_jsbsim_737_layers.py`, `audit/INVENTORY.md`, `audit/NOTATION.md`,
> `docs/ASSUMPTIONS.md`, `docs/PROJECT.md`.
>
> **`atisim/validation.py` is the one that decides the cost.** Both branches linearise it
> independently — one about the **transport rate**, one about the **equilibrium** — and
> whether those compose or contradict is **not established**. Answer that before touching the
> other seven, because it decides whether this is a merge or a rewrite.

#### Measured, session 32 — what the merge actually costs

**Question 1, answered: the two linearisations neither compose nor contradict. `32fbdd`'s is an
incomplete version of `5dbdc3`'s.** Both linearise at `q = q0` and cite the *identical*
measurement — `−1.369798008797381e-07` leaking into `A[2,0]` — so it is one diagnosis, fixed
twice. But `5dbdc3` measured what `32fbdd` did not: at `q = q0` alone the residual is
`|f(x0)| = 1.336e-05`, **still not an equilibrium**. The kinematic row must also become the
local-NED pitch rate `θ̇ = q − q0`, because `omega` is a rate relative to ECEF while `θ` is an
angle in a frame itself turning at `q0`. Only with both does `|f(x0)|` fall to `2.064e-13`.
**`32fbdd`'s linearisation point is therefore off-equilibrium by 1.3e-05**, and keeping both
blocks would compute `q0` twice. **Resolution: take `5dbdc3`'s, drop `32fbdd`'s hunk.**

**The union itself is cheap** — trial-merged in a throwaway worktree: **7 files, 13 hunks**, one of
them `validation.py`, now answered.

**The union against today's `main` is not.** §0's "27 behind" was stale: both branches are
**78 behind**, having left `main` at `889714d` on 26 August. Since then `main` has changed
**272 files, +45,459 lines**. Trial-merged:

| | |
|---|---|
| Files changed by both the WGS-84 union and `main` | **32** — every file that touches the state vector |
| Textual conflicts, trial merge of the union into `main` | **20 files, 50 hunks** — `integrate.py` 6, `dynamics.py` 5, `test_audit_regression.py` 5, `checks.py` 4, `test_jsbsim_737_layers.py` 4 |

**The 50 hunks are the floor, not the cost, and this is why.** WGS-84 changed the `State`:

| field | `main` | WGS-84 | if `main`'s code reads it unchanged |
|---|---|---|---|
| `pos_ned` → `pos_ecef` | NED position | ECEF offset from the run anchor | **fails LOUDLY** — renamed, `AttributeError` |
| `quat` | body → **NED** | body → **ECEF** | **fails SILENTLY** — same name, new meaning |
| `omega` | body rate | body rate **relative to ECEF** | **fails SILENTLY** |
| `vel_body` | body velocity | **ECEF-relative**, body axes | **fails SILENTLY** |

Counted in code `main` added since the fork — code the WGS-84 branches never saw: **15
references to `pos_ned`**, which will announce themselves, and **30 to `quat`, `omega` and
`vel_body`** (10, 7, 13), **which will merge clean, run without error, and return a wrong
number.** `wind.py` shows the shape of it: **1 textual hunk, against 836 lines `main` changed** —
sessions 23–30's CAT work, written in new regions, so it merges clean and was written against
the flat-Earth state.

**Every one of those 30 needs a human to decide what it means in ECEF.** That is the part
that cannot be estimated from a hunk count, and it is also the part that fails in the way this
project most needs to avoid: a plausible number with no error attached.

**Where they are — the rebase map.** Counted by walking `git diff -U0` from the fork point to
`main` and recording the file and line of every added reference:

| field | n | files |
|---|---|---|
| `quat` | 10 | `integrate.py` 2, `sensitivity.py` 2, `sensitivity_assumptions.py` 2, `dynamics.py`, `vortex_viz.py`, `cat_bounds.py`, `les_flight.py` |
| `omega` | 7 | `sensitivity.py` 3, `sensitivity_assumptions.py` 2, `dynamics.py`, `test_cat_validation.py` |
| `vel_body` | 13 | `sensitivity.py` 4, `les_flight.py` 2, `sensitivity_assumptions.py` 2, `integrate.py`, `test_cat_validation.py`, `vortex_viz.py`, `cat_bounds.py`, `cat_spectra.py` |
| *`pos_ned`* | *15* | *loud — renamed, so these announce themselves: `integrate.py` 3, `sensitivity.py` 2, `verification.py` 2, and eight files with one each* |

**Half of the 30 silent references — 15 — are in session 29's sensitivity study**:
`sensitivity.py` has 9 and `sensitivity_assumptions.py` 6. That is the differentiable path, the
one that produced §1's banded headline and the `CLa`/`mass` ranking. **It is therefore where the
rebase is hardest, and where a frame error would reach a published number** — so it is the
first thing to review and the first thing to re-measure after, rather than the last.

#### The union — BANKED, session 32, and not merged

> **Decided after the measurements above: build the union, push it as one branch, and stop
> short of `main`.** Merging into `main` this month was judged not to fit alongside the
> release, on the numbers in this section: 50 textual hunks in the engine core, 30 silent
> references, and a §4 re-baseline behind them. **A green suite was also judged an
> insufficient gate** for this particular merge — the suite asserts bands and orderings, so a
> small frame error in one of the 30 can land inside a band and pass. Whoever lands it should
> add an exact before/after on the four CR-2144 modes and the Hannibal headline.

| | |
|---|---|
| **Branch** | **`wgs84-earth`** — the union, built from `claude/wgs84-earth-rotation-tasks-5dbdc3` with `claude/atisim-wgs84-earth-rotation-32fbdd` merged in |
| **Worktree** | `.claude/worktrees/wgs84-earth` |
| **What it resolves** | The two branches' **13 hunks across 7 files**, each on its merits rather than by taking a side. **It supersedes both source branches**, which carry nothing it lacks |
| **Suite on the union** | **750 passed, 1 skipped, 0 failed**, 993 s, measured on the union's own tree (`atisim.__file__` checked). The three files the resolution touched pass 69 of 69. **750 against `main`'s 918 is not a loss**: the union is 78 commits behind and predates the tests sessions 23-32 added. What it shows is that reconciling both branches left nothing broken |
| **What it does NOT do** | Touch `main`. It is still **78 behind**; the 50 hunks and 30 silent references above are the work that remains. **Since phase 4, `docs/superpowers/` is `docs/design/` on `main`**; the union adds two files under the old path, which git's rename detection should carry across on rebase — check they land in `docs/design/` |
| **Source branches** | **RETIRED, session 32**, after checking the union holds every commit of each — 0 missing from either. `claude/wgs84-earth-rotation-tasks-5dbdc3` was `d030701`; `claude/atisim-wgs84-earth-rotation-32fbdd` was `ec153fb`. Their worktrees, `sleepy-moore-7186bb` and `atisim-wgs84-earth-rotation-b7b3bc`, were clean and are removed |

### ~~UNRECORDED ENTIRELY, and it closes a §7 item~~ MERGED, session 32 — kept as the record of how it was found

> **This document did not mention this branch, in any section, at all.** Not a stale row, not a
> wrong address — **zero occurrences**. It is four commits ahead of `origin/main`, was written
> **on 17 September 2026 between 10:49 and 14:46**, and it reports closing the item §7 calls
> "the only thing that would let the aircraft-type explanation be tested rather than argued".
> Session 32 found it by running the rule-1b two-line check that §0 exists to make routine.
>
> **It also calls itself "session 31", and so did this session at first.** Two sessions ran the
> same day and took the same number. **That branch keeps 31; this one renumbered to 32**, on
> the basis of the clock. A later reader seeing "session 31" in `PROJECT.md` should expect *that*
> work, not this.

| | |
|---|---|
| **Branch** | **`claude/work-summary-derivatives-sensitivity-9a2c03`** |
| **Worktree** | `.claude/worktrees/jsbsim-atisim-vortex-rings-34659d` — **the directory name is unrelated to the branch**, as everywhere else here |
| **State** | **MERGED, session 32.** Was 4 ahead of `origin/main`, 0 behind. Suite **not run** — it changes no model code, only `docs/PROJECT.md` (+261) and one new script, `scripts/sensitivity_mass_diagnosis.py` |
| **What it reports** | (1) **The Hannibal flight identified** as **United Airlines Flight 12**, DC-10, LAX → EWR, 3 April 1981, cited to NTSB Safety Recommendation A-84-108 and NASA CR-203832's Ames incident table. (2) **The aircraft pinned** as **N1809U**, from NTSB case CHI81DA042 (CAROL Mkey 127770), **a DC-10-10** — the variant flagged in its own text as **SECONDARY**, from two JetPhotos records, because the FAA registry refused automated access with a 403. (3) **§5's wing-loading ratio narrowed from 0.58–1.32× to 0.584–1.046×**, 0.815× at mid weight, **because the heavier −30 corner no longer applies**. (4) Session 29's `mass` elasticity re-read: `mass` and `CLa` are **one lever**, `CLa·q̄S/W`, mirrored to 0.04, and an error in CR-2144's *printed* weight **cancels to +0.008** because the builder derives `CLa` from it. (5) The DECLARED Korn drag rise priced on the cruise phugoid |
| **What it closes, if it lands** | **§7's "the Hannibal flight record"**, in part — operator, flight number, tail and variant. **§5's "DC-10 wing loading, unpinnable"** narrows but does **not** close: **the weight on the day is still not found**, and its own text says so. It records that NASA held both the weight and a DC-10 aerodynamic model (TM-102186 §2 computes C_L "using the aircraft weight") and **published neither**, and names the routes left: the NTSB docket, United's archives, and the Sept 1982 SFTE paper by Parks, Bach & Wingrove, which is not online. It also records what it did **not** try: the NTSB pre-1982 database, a 39 MB Access file, with no reader installed |
| **Why it matters beyond itself** | §5 records that **at 1.3× wing loading the load falls to 56.4% of the record**, and §7 warns that if the DC-10 were the heavier-loaded aircraft the DC-10 acquisition would make the shortfall *worse*. **Removing the −30 corner removes that 1.3× case.** Whether that changes §5's conclusion is **not established here and may not be quoted until the branch is verified and merged** |
| **Blocking** | ~~**Nothing — it is unmerged and was never written down.**~~ **Merged, session 32.** It conflicted in §0, §4, §7 and §9 exactly as predicted, in five hunks. §4 keeps **both** entries, newest first; §9 reads 32, 31, 30; §7 takes its Hannibal row, which supersedes session 27's |

> ~~**Read this row as a POINTER, not as a result.**~~ **Merged, so it is now a result** — its
> numbers are in §4's first entry and §7's flight-record row, under its own name. What has
> **not** changed is the standard of evidence behind them, and the branch is careful about this
> where it matters: **the −10 variant is a SECONDARY source**, two JetPhotos records, because
> the FAA registry refused automated access with a 403, and **the weight on the day is still not
> found**. §5's ratio is therefore narrowed, not closed, and the aircraft-type explanation is
> still not testable.

### ~~Session 32's own work — the release cleanup~~ MERGED, phase by phase, through pull requests #4–#11

| | |
|---|---|
| **Branch** | **`release-cleanup-september`** — deliberately not a `claude/*` name |
| **Worktree** | `.claude/worktrees/new-session-943052` — **the directory name is the old branch's**, as everywhere else here |
| **State** | ~~4 commits beyond `main`'s `c1b7d71`, 0 behind.~~ **MERGED.** `release-cleanup-september` landed through PRs #4–#8 (phases 0–1), `wgs84-merge-analysis` through #9 (phase 2), and `phase-3` through #11. Each phase's suite result is in its §9 point |
| **What it is** | The plan for the final month, in `docs/design/specs/2026-09-17-final-release-cleanup-design.md`, and the first phase of it: the copyright remediation, the CR-2144 rescue, and this §0 pass |
| **What it closes** | §0 itself, as the month's work merges. The plan's phases 1–7 are the route |
| **Blocking** | Nothing — merged. The release's remaining phases (documentation, notebook, history rewrite) work from `main` |

### Found untracked in the MAIN CHECKOUT — session 32

> **A third instance of the rule-1b failure, and the largest by data volume.** Sessions 27 and
> 28 each found stranded work elsewhere; this session found two more caches, one of them a
> digitisation of the project's primary aerodynamic source. **Now tracked**, at `e521dbc`.

| | |
|---|---|
| **What it was** | `Reference_papers/CR-2144/` — an **automated digitisation of NASA CR-2144 printed pp. 218–228**: 25 panels, 78 curves, 77 emitted, at 300 dpi off the embedded scans, each panel calibrated on its own printed ticks by a RANSAC fit. With overlays, a verification sheet and an overrides log |
| **Why it matters** | §7 records session 30 digitising **pp. 220–222 by hand** and §4 calls that the highest-value item on the plan. This covers **pp. 218–228**: p.221 `Cm_adot` — the α̇ curves; p.222's `CL_M`/`CD_M`/`Cm_M`, which session 30 read by hand; and the **whole lateral set** (`Cl_p`, `Cn_p`, `Cl_r`, `Cn_r`, `Cl_da`, `Cn_da`, `Cy_dr`, `Cn_dr`, `Cl_dr`), which §5 names as a gap |
| **What is now free** | **Two independent digitisations of pp. 220–222 exist** — this one and session 30's Engauge reading in `atisim/data/cr2144_dig/`. Comparing them **prices the reading error on both by a route neither can price alone**, and costs one script. §7 carries it |
| **Its own stated limits** | The README's confidence column marks **four panels `poor`** — `Cm_adot`, `CL_M`, `Cy_beta`, `Cl_p` — where the three altitude curves print within a line width of each other and the altitude split is partly interpretive. Tick-fit residuals are 0.2–3.5 px; the dominant error is the **printed line width**, ~1–2% of full scale, not the calibration |
| **Status** | ~~**TRACKED, NOT REVIEWED, NOT USED.** No number in it has been checked against Table IX-4 and none reaches any aircraft entry~~ **TRACKED, REVIEWED, STILL NOT USED — session 32.** §4's "Two readings of CR-2144" scores it against Table IX-4 beside session 30's hand reading: on all three declared speed derivatives **the hand reading is the better anchored** — `Cm_M` by 2.3×, `CL_M` by 4.9× — and a sealed prediction that the two would agree on the good panels was settled **WRONG** on one of four. `SOURCES.md` carries the resulting rule: do not take `CL_M` or `Cm_adot` from it; it is the better reading of `CL_alpha` only. **No number in it reaches any aircraft entry**, which is the part that has not changed |

| | |
|---|---|
| **What it was** | Five NASA/NTRS reference PDFs, 31 MB, untracked beside them — CR-3677, CR-3748, TM-4745, TM-1998-206552 and Taylor 1978 |
| **Decision** | **Not tracked.** All five are permanently available from NTRS, and `Reference_papers/SOURCES.md` now carries each one's md5, NTRS ID and what it does and does not supply. CR-3677's assessment against the open DC-10 acquisition is already in §7 |

### ~~Small, unmerged, and cheap to resolve~~ RESOLVED, session 32 — every row settled

> **All six rows are closed.** Two were merged, two abandoned as subsumed, one applied across
> the rename, and one is moot. The abandonment table above carries the reason and the commit
> SHA for each; this table is kept because §0 supersedes rather than deletes.

| Branch | What happened |
|---|---|
| `claude/linearisation-verification-bounds-b73868` | **DEFERRED, not resolved** — it modifies `atisim/validation.py`, the file the WGS-84 union fight is over, so it is handled with that merge. See "Deferred to the WGS-84 merge" above |
| `claude/flight-dynamics-solver-oscillation-17139b` | **DEFERRED**, same reason, and it also touches two of rule 3's five off-limits files |
| `claude/priceless-cori-688ee5` | **ABANDONED** — subsumed exactly as this row predicted: the rename is on `main`, the geopotential ISA read landed at `50b78a1` |
| `claude/flightsim-sweep-ui-graphs-d4d036` | **APPLIED**, not abandoned. Its files had not drifted despite the branch being 97 behind, so the diff was rewritten across the rename and applied. It fixes subtitles drawn inside the plot area and a legend drawn through the caption — see "The panel-chrome fix" above |
| `claude/weekly-summary-analysis-7520db` | **ABANDONED**, but only after its `runs/cat/` outputs — 55 files, the ten LES arrays, existing nowhere else — were harvested to the main checkout and verified by md5. §10 says where they now live |
| `session-27-validation` | **MOOT.** It renamed `CLAUDE.md` to `AGENTS.md`. Session 32 moved that file to `docs/DEVELOPMENT.md` instead, so the naming decision §0 said "this document has not taken" is taken, differently. The branch is not on this machine |

### ~~Rescued from a worktree at the session-32 audit, and unreviewed~~ REVIEWED AND PARKED, later in session 32 — THE α̇ DERIVATIVES

> **Committed at `6148cd8` to stop it being lost. NOT reviewed, NOT endorsed, suite NOT run.**
> Rescuing work and approving it are different acts, per the session-28 precedent.

| | |
|---|---|
| **Branch** | **`claude/engine-validity-presentation-1408e8`** |
| **Worktree** | `.claude/worktrees/project-md-restructure-95b7b8` — **the directory name is a different branch's**, again |
| **State** | **1 commit ahead of `7b71816`**, which is `main` as it stood before session 29 merged, so it is ~10 behind. **48 files.** Suite not run |
| **What it is** | **`aircraft.py` restores Table IX-4's `Mwd` to the 747 as `Cmadot`** — the α̇ pitching derivative — with the conversion round-tripping to the tabulated −0.000116 and an argument that it *adds* rather than double-counts because the tabulated `Mq` is bare. **It leaves `CLadot` at zero and says why in twelve lines**: Table IX-4's `Zwd` = +0.00556 converts to a *negative* `CLadot`, which is unphysical, and rebuilding Table IX-5 with +0.00556, −0.00556 and 0.0 gives 1.11%, 0.66% and 0.68% — all inside a three-figure reference's reading precision, so **IX-5 cannot arbitrate the sign either**. That is rule 2 applied correctly. Also `dynamics.py`; four α̇ scripts (`alphadot_conversion`, `alphadot_isolate`, `galilean_alphadot_probe`, `zwdot_sign_probe`); `test_alphadot_derivatives.py`; and ten more scripts including `prandtl_glauert_check`, `e4_windhold_remeasure` and `recapture_fig8_pins`. Plus a presentation package: a 1.1 MB deck, `build_deck.py`, a QA defence brief and eleven figures |
| **What it closes** | **§5's α̇ entry**, if it survives review — the derivative the model has excluded by form since session 1 |
| **Blocking** | ~~**Review, and one thing specifically. `atisim/tests/test_cr2144_modes.py` IS MODIFIED**, and it is one of the five files `docs/DEVELOPMENT.md` rule 3 declares off-limits to feature work. The `aircraft.py` comment claims the augmented model there closes all four Table IX-5 factors to ≤1.2%, which would make it a **re-capture** rather than a loosening — but rule 3 puts the burden of showing which *on the change, in a comment, at the change*, and that has not been shown. **Settle this before anything else in the branch.** Its `PROJECT.md` (+479) and `ASSUMPTIONS.md` (+28) edits predate sessions 29–30 and will conflict~~ **REVIEWED, later in session 32 — a RE-CAPTURE, not a loosening, and the burden WAS discharged; the line above claiming it was not is wrong.** It was written having read `aircraft.py` and not the test diff. The branch changes exactly one assertion in `test_cr2144_modes.py`: short-period ζ is re-pinned **0.338 → 0.3895** with `rel = 0.05` **unchanged**, the other three assertions untouched, and a comment at the change arguing it — the value moves from **11.35% below** Table IX-5's 0.387 to **0.65% above** it, and switching `Cmadot` on and off leaves trim and all four lateral modes bit-identical. Both halves of rule 3's test hold: no tolerance widened, and the value moved toward the source. **What it does not settle: the branch predates session 30**, which restructured that same test to fly `boeing747_without_thrust_line()` with the Mach seam shut, pins unchanged. Merging `Cmadot` into the base 747 would reach it through that builder, so whoever merges must choose between session 30's pattern — switch `Cmadot` off in the bare form, so the test keeps documenting α/q/δe — and the branch's re-pin. **No run on `main` backs either.** **Parked as FUTURE WORK** (§5 status table, 5.5): merging moves the four CR-2144 modes and §1's headline, a week before the release. |

### ~~Rescued from worktrees at the session-28 audit, and unreviewed~~ ALL FIVE SETTLED, session 32

**Five worktrees held uncommitted work.** All five were committed on their own branches to
stop them being lost. **None was reviewed and none was endorsed** — the commit messages said
so, and for three weeks nobody reviewed them. **Session 32 reviewed all five and settled each.**

| Branch | What was rescued | Settled |
|---|---|---|
| `claude/zen-maxwell-1ad0a4` | **The most substantial: a tail-arm gate refactor with tests** | **MERGED.** Reviewed, and it closes a silent NaN now recorded as **§6(g)** — `stations` built sample points out to −∞ for any entry with no `CLq`, and every fitted gradient returned NaN with nothing raised |
| `claude/cat-flight-model-dossier-04adb9` | `docs/design/plans/2026-08-31-close-the-dossier-limitations.md` | **ABANDONED — plan superseded, ledger harvested.** Five sources in it appear nowhere else in this document and are now §7, **14 CFR 25.341** chief among them |
| `claude/project-md-restructure-95b7b8` | Local `PROJECT.md` edits | **ABANDONED** — 92 commits stale; reapplying would undo sessions 28–32 |
| `claude/flight-dynamics-cat-prompt-ec9839` | A CAT-sources search prompt | **ABANDONED** — a prompt written to an AI, the class of file deleted from the root this session |
| `claude/project-readme-mockup-0ef89a` | A README mockup | **ABANDONED, two headings harvested** into the README rewrite: "What it can't do (read this bit)" and "Why there's so much paperwork in here" |

**The lesson the session-28 audit drew was "rescue it so it is not lost". The lesson this one
adds is that a rescue is not a resolution.** Four of these five were worth nothing by the time
they were read — and the fifth was a bug fix that had been sitting unmerged for three weeks
while the bug it fixes stayed in `main`.

### ~~Designed, built and run~~ MERGED, session 29 — kept for the two bugs it found

> **Merged into `main` on 16 September 2026** at `105f689`, as a `--no-ff` merge of all eight
> commits. The merged tree is **byte-identical** to the one the suite last ran on, so the
> 819-passed result transfers exactly rather than needing a re-run. **The row is kept rather
> than deleted**, on the same grounds as the compressibility branch above: it carries the two
> `sqrt(0)` findings and the corrected Fig. 8 statistic, and deleting it would delete why they
> were found.

| | |
|---|---|
| **Branch** | **`claude/model-sensitivity-analysis-t18v3v`** |
| **Worktree** | none — a remote container on the main checkout, no `.venv` (`jax` installed fresh on Linux; §10's Windows table does not apply) |
| **State** | **MERGED at `105f689`.** All seven phases S0–S6 done, plus two follow-ups: `airframe.py`'s `sqrt(0)` closed, and Fig. 8's pitch axis investigated (§4, correcting session 23d). Suite: **776 passed** before any of this work, **808** after S0–S6, **819** after the follow-ups; the same 2 platform bit-pins fail throughout and predate the work |
| **What it is** | `docs/design/specs/2026-09-10-model-sensitivity-analysis-design.md` — a sensitivity study over three quantities of interest (headline CAT load, cruise modes, Dryden ensemble statistics) and two factor tiers (aerodynamic derivatives, `ASSUMPTIONS.md` modelling choices), by a tiered method: AD screen → OAT confirm → banded propagation |
| **What it closes** | **CLOSED, all three.** `ASSUMPTIONS.md` **C3's Mach axis is bounded at cruise: −5.04%** of the headline load over a measured Mach span of 0.7187–0.8257. **§1's headline carries a band — 68.2%, 57.1–74.0%** — and states that the shortfall survives it. And phase 3's DC-10 acquisition has a **price**: `CLa` is the top-ranked coefficient on the load at **+0.692**, `mass` second at **−0.649**. **Session 31 qualifies that price** (§4, first entry): the two are one lever, `CLa·q̄S/W`. `mass` was moved with the inertia held fixed, and scaling the inertia too gives −0.800. And an error in a *published* weight cancels (+0.008), because the entry derives `CLa` from it |
| **Blocking** | **Nothing — merged.** What remains is named rather than pending: C3's **α axis**, still unbounded; **interaction terms**, which this study measured none of and says so with every table; and the question §8 raises about whether Fig. 8's pitch axis survives past this aeroplane's linear range. ~~`airframe.py`'s second `sqrt(0)`~~ was closed in the same session |

**All three obstacles the design named are solved.** `trim`'s Newton solve is handled by the
implicit function theorem twice over — `implicit_trim_jacobian` as a table and `solved_trim` as a
differentiable primitive, checked against each other AND against differentiating through the
unrolled loop. The `np.linalg.eigvals` break is closed by first-order eigenvalue perturbation.
And `vortex_viz._measure`'s NumPy return is closed by `load_history`, which is **bit-identical**
to it across all 4,737 samples of the headline run.

**A fourth obstacle was not in the design and was found by running it:** `aero.py`'s
Prandtl–Glauert sentinel made the model non-differentiable in its own coefficients for every
aircraft in the registry. Fixed, proved value-identical, recorded in **§6(f)** — with a second,
open instance in `airframe.py` named there.

### ~~Superseded, kept only until someone confirms~~ ABANDONED, session 32 — the decision taken

`claude/flight-dynamics-email-3601f2` (1 ahead, **143 behind**) and
`claude/flight-dynamics-validation-67fa59` (1 ahead, **169 behind**) are both `WIP:` commits,
and the second says of itself "superseded on master by 099bd33". **These are the two safe
deletions on the list**, and they were recorded here rather than deleted so the deletion is a
decision someone takes rather than one that audit took for them. **Session 32 took it.**

### Abandoned, session 32 — with what was taken from each first

> **Rule 1b: abandoning is a §9 entry saying why, never a silent removal.** Eight branches were
> abandoned after checking each for anything not already on `main`. **Two were not empty**, and
> what they held was harvested into this document *before* the branch went — which is the whole
> point of doing it in this order.

| Branch | Why abandoned | What was taken first |
|---|---|---|
| `claude/weekly-summary-analysis-7520db` | **Fully subsumed.** All five of its scripts — `digitise_mil_f_8785c_fig7`, `digitise_tm102186_fig6`, `les_compare`, `les_flight`, `yoshimura_flightsim` — are on `main`, verified by `git cat-file` on each | **Its `runs/cat/` outputs, which existed nowhere else**: 55 files, 11 MB, including the **ten `les-nz-*.npy` ensemble arrays** §4's session-28 POD row reads and §10's `les_ensemble_svd.py --runs` needs. Copied to the main checkout's `runs/cat/` and **verified file-by-file by md5**. They are gitignored by design, so §10 now names where they live |
| `claude/cat-flight-model-dossier-04adb9` | **Plan superseded.** Its Phase 0 argues the 500 ft core radius session 26 reversed, and it asserts "Parks 1985 has never been obtained", which session 26 disproved by obtaining it | **Its retrieval ledger — five rows recorded nowhere else**, now §7's "retrieval ledger, harvested from an abandoned plan". Chief among them **14 CFR 25.341**, the only free source that would give the σ-against-altitude table *independently* of the U-2 lineage §7 warns MIL-F-8785C Fig. 7 may share |
| `claude/priceless-cori-688ee5` | **Subsumed**, as §0 predicted. The AtiSim rename is on `main`; the geopotential ISA read landed at `50b78a1` with the compressibility branch | nothing left |
| `claude/project-md-restructure-95b7b8` | A restructure of this document from **92 commits ago**, before sessions 28–32 rewrote §0, §4, §6, §7 and §9. Reapplying it would undo them | nothing left |
| `claude/flight-dynamics-cat-prompt-ec9839` | A 278-line **search prompt written to an AI**, the same class of file as the three deleted from the repository root this session | nothing left |
| `claude/project-readme-mockup-0ef89a` | A README mockup; the README is being rewritten from scratch in the release work | **Two headings better than the current README's**, carried into the rewrite plan: "What it can't do (read this bit)", which puts the envelope *before* the results, and "Why there's so much paperwork in here", which is the only place anything explains why this repository ships an 8,000-line evidence ledger |
| `claude/flight-dynamics-email-3601f2` | 143 behind; a `summary.py` rewrite and a `flightsim-summary.pdf` under the pre-rename name | nothing left |
| `claude/flight-dynamics-validation-67fa59` | 169 behind, and says of itself "superseded on master by 099bd33" | nothing left |

**The commit each pointed at, so abandoning is reversible.** A deleted branch whose SHA is
written down can be recovered — `git show`, `git cherry-pick`, or a new branch at that commit —
for as long as the object survives. A deleted branch whose SHA is *not* written down is the
thing rule 1b is about. These were read before they were deleted, and this table is why the
reading does not have to be repeated.

| Branch | Commit |
|---|---|
| `claude/weekly-summary-analysis-7520db` | `f9b9717` |
| `claude/cat-flight-model-dossier-04adb9` | `d703d37` |
| `claude/priceless-cori-688ee5` | `ce886f6` |
| `claude/project-md-restructure-95b7b8` | `39d1c56` |
| `claude/flight-dynamics-cat-prompt-ec9839` | `124ae05` |
| `claude/project-readme-mockup-0ef89a` | `39b3660` |
| `claude/flight-dynamics-email-3601f2` | `04e9ae0` |
| `claude/flight-dynamics-validation-67fa59` | `bd56979` |
| `claude/flightsim-sweep-ui-graphs-d4d036` | `f5ed73d` — **not abandoned; applied**, see below |

### The panel-chrome fix, applied across the rename — session 32

**`claude/flightsim-sweep-ui-graphs-d4d036` was nearly abandoned with the rest** for being 97
commits behind on the pre-rename `flightsim/` package. Measuring first changed the answer:
`analysis/figures.py`, `apps/sweep.py` and `test_figures.py` are **the same length on `main` as
at the merge base**, with 2–5 lines changed in each — the rename itself. So the branch was 97
commits behind and its *files* were not behind at all.

Applied by rewriting the diff's paths `flightsim/` → `atisim/` and `git apply --3way`: three
files clean, **352 insertions**. `test_figures.py` gains 154 lines and the file passes 38 tests.

**What it fixes is visible and was being shipped.** The panel chrome reserved its top margin by
a guessed formula, wrong three ways at once on every panel carrying a subtitle: a title block's
lines are spaced at 1.3× the *title* size, 17 px at 13, where the formula reserved 13 px, so
**the last line of every subtitle was drawn inside the plot area**; the block's own offset was
missing from the sum; and the legend sat at paper y = 1.02, the band the title was overflowing
into — measured on Fig. 8 as title x 0–293 and legend x 110–649 **both at y 39–68, the legend
drawn straight through the caption**. Each element now gets its own band from a running pixel
offset, with the constants measured at three panel heights and at three- and four-line blocks.

### ~~Deferred to the WGS-84 merge, deliberately~~ HARVESTED AND ABANDONED, later in session 32

> **Both branches were read commit by commit and each claim checked against `main` before either
> was deleted** — the order the rest of this section follows. They were **not merged**: 92–94
> commits behind, on the pre-rename `flightsim/` package, and touching two of rule 3's five
> off-limits files. **They held three findings `main` did not have, and one of them was live and
> contaminating a published number.**
>
> | Claim on the branch | On `main` | Now recorded as |
> |---|---|---|
> | **Roll counted twice** — `field_model` and `strip_model` both own the spanwise rolling moment (`cf538dd`) | **LIVE, and measured**: `strip=True` adds the strip path to the point path's roll rather than replacing it. The published strip effect of +17.4% on peak bank becomes **−2.5%** with one path owning roll | **§6(h)**, with §4's lateral result and §7's phase-1 gate superseded in place |
> | **A third `Ixz`** — Caughey's 747 approach −2.23e6 against CR-2144's 0.825e6 and 0.870e6 (`65b7ce3`, `8c36d92`) | **Not recorded anywhere on `main`** | **§5.16(c)**, an error in the source |
> | **The flown phugoid is not the reported one** — altitude couples through density; +20.5% ωn, −50.6% ζ (`e50ad31`) | **Live as a fact, not a bug**: `main` reports the 4-state, which is the correct CR-2144 comparator | **§5.21**, FUTURE WORK to re-measure on the shipped 747 |
> | `L1 = −(F11 + ω̃)`, not `−F11`, off trim | **N/A** — the identity belongs to the branch's own 12-state linearisation, which `main` does not have | — |
> | ψ left behind by a one-pass reduction | **N/A** — the branch's own reduction code | — |
>
> **Commits, so the abandonment is reversible:** `claude/linearisation-verification-bounds-b73868`
> was **`8c36d92`**; `claude/flight-dynamics-solver-oscillation-17139b` was **`69c8c00`**. The
> first carries ~3,750 lines of verification work — `verification.py` at 953 lines against
> `main`'s 359 — which is the instrument §5.21 needs and a starting point for §6(h)'s repair.

*The original deferral, as written:*


> **The reason below has lapsed, later in session 32.** It was that these two should be handled
> *with* the WGS-84 merge because all four touch `atisim/validation.py`. The WGS-84 union was
> then banked and **not** merged into `main`, so these two now face `main`'s flat-Earth
> `validation.py` on their own — a different and smaller question. **They move to the gap
> phase**, to be merged against `main` or abandoned there, on their own merits.

**Two branches were NOT abandoned and NOT merged, for one reason: they contest the same file
the WGS-84 union fight is over.** `atisim/validation.py` is modified by all four, and merging
these two first would make that fight harder rather than easier. They are also on the
**pre-rename `flightsim/` package** and 92–94 commits behind, so neither is a cheap merge.

| Branch | State | What it holds |
|---|---|---|
| `claude/linearisation-verification-bounds-b73868` | 3 ahead, 92 behind, `flightsim/` | **~3,750 insertions of verification work**: `verification.py` at **953 lines against `main`'s 359**, plus `test_linearisation.py` (547), `test_caughey_lateral.py` (273), `test_scale_bounds.py` (230), `test_effect_ownership.py` (230). "Bound the linearisation, and find roll counted twice"; "a third `Ixz`"; "Say what kind of model this is, and who owns which effect" |
| `claude/flight-dynamics-solver-oscillation-17139b` | 2 ahead, 94 behind, `flightsim/` | "Carry altitude in the longitudinal reduction, and keep every published comparison 4-state"; "Reduce FD2e's own state sets, and discover the zero columns instead of assuming them". **Touches `test_cr2144_modes.py` and `test_navion.py`**, two of rule 3's five off-limits files |

### What `main` itself is doing

~~`main` is **7 ahead of and 1 behind `origin/main`** — the two have diverged and neither is a
superset.~~ **SUPERSEDED, session 32, re-measured: `main` is 0 ahead and 19 BEHIND
`origin/main`** — local `main` sits at `7b71816` while `origin/main` is at `c1b7d71`, session
30's merge. **It is a pure fast-forward**, with nothing to reconcile. The divergence the row
above described has been resolved by pushing, not by merging.

> **Measure branch counts against `origin/main`, not against local `main`, until that
> fast-forward is taken.** Every count in this section is otherwise inflated by 19, which is
> what made the two WGS-84 branches read as "68 and 65 ahead" when against `origin/main` they
> are the same 68 and 65 but everything else shrinks — `session-30-speed-derivatives-hannibal`
> and `claude/project-cleanup-docs-plan-f0710b` go to **0**, being already merged, and
> `claude/work-summary-derivatives-sensitivity-9a2c03` is **4**, not 15.

**At the release.** `origin` is `MatusGib/AtiSim` — this repository, as the history rewrite
left it — and local `main` tracks it. The pre-rewrite repository, with the nineteen
pull-request pages that made an in-place rewrite insufficient, is `MatusGib/AtiSim-archive`,
private. `old-origin` predates both and is unchanged. The paragraph below was written before
that and is kept as written.

There are two remotes: `origin` (MatusGib/Atisim) and `old-origin`
(MatusGib/Flight_sim), the pre-rename repository. **Nothing in this document says which is
authoritative or whether `old-origin` still needs to exist.**

**`origin` carries only three branches** — `main`, `claude/model-sensitivity-analysis-t18v3v`
and `session-30-speed-derivatives-hannibal` — against **23 local**. Everything else in this
section exists on one machine only, which is the other half of why rule 1b keeps failing here:
an unpushed branch is invisible to every check that looks at the remote.

**Twenty-three of the thirty-nine branches carry no unique commits at all** and are merged
into `main`. They are deletable, but seven of them have worktrees attached and one of those
seven — `claude/zen-maxwell-1ad0a4` — held the tail-arm work above until this session
committed it. **Deleting a merged branch is safe; deleting its worktree is not, until the
worktree is checked for uncommitted files.**


## 1. What this is

Quaternion state, fixed-step RK4, `lax.scan` rollout, `jit` + `vmap` over PRNG keys.
Three aircraft, a cascaded PID autopilot, manual control, matplotlib visuals, and a
turbulence layer under construction. Float64 throughout (`jax_enable_x64`, set before any
array is created — a Newton trim solve to 1e-10 and quaternion norm stability over 1e5
steps are both marginal in float32).

The original design spec is `docs/design/specs/2026-08-04-jax-flight-sim-design.md`.
It remains accurate on architecture. **It is stale in one respect: it lists turbulence as
out of scope, which is no longer true** — that was always the intended destination, and
the two "non-negotiable interfaces" it names exist precisely so turbulence could be added
without a core rewrite. Both have now been exercised and both held.

### The validation claim — what this model may and may not be used for

**Added session 24 (phase 0).** Everything the project has measured lives in §4, which is
long and grows by session. This is the one paragraph a reader needs before using any of
it, and it is deliberately narrower than what the evidence might be stretched to support.

> **AtiSim is validated as a comparative and mechanistic tool for the LONGITUDINAL gust
> response of a rigid transport aircraft at cruise.** Within the envelope below it
> reproduces published response orderings, the mechanisms behind them, and the linear
> modes of its own source data. **It is not a validated absolute-load predictor**, and
> until session 24 it could not represent lateral turbulence at all.

**The envelope the claim is made inside.** Outside any one of these the claim does not
transfer, and §4 will not have measured it:

| | Validated range | Set by |
|---|---|---|
| Aircraft | 747-class transport; `boeing747`, `boeing747_jsbsim` | the only entries with a measured recovery band |
| Mach | **0.70 – 0.90** | `aircraft.valid_mach`, edges measured not assumed |
| Altitude | **35,000 – 45,000 ft** (`boeing747`); **35,000 – 41,000 ft** (`boeing747_jsbsim`) | `aircraft.valid_altitude`. The two differ and the narrower one governs any statement made about both — do not quote the union |
| Incidence | **\|α\| < 10°** green, 12° amber | `panel.ALPHA_LINEAR_DEG`; the aero has no stall |
| Gust length scale | **> 3 wingspans** | ASSUMPTIONS E2; the Parks core at 2.3–3.1 spans is the marginal case and is called out as such |
| Axis | **longitudinal only** | see §5 — no field varied across the span before session 24 |

**What "comparative and mechanistic" buys, and it is more than it sounds.** The
questions this model answers with evidence behind it are the ones most engineering
questions actually are: *which encounter is worse, how does the response scale with
airspeed, what happens to the load if the aircraft is slower, does the ordering hold.*
§4 records six-for-six monotone agreement on TM-102186's stated mechanism and both
halves of its three-aircraft ordering — including the counter-intuitive half.

**Session 25 adds a frequency-domain member to that list, and it is a different kind of
claim.** The `n_z` response spectrum of the headline run peaks 0.23 of a bin from the
aircraft's own short period and 1.64 bins from the mean core-passage frequency of the
array forcing it, and over a Dryden ensemble the response carries **3× more** energy at
the short period than at 0.05 Hz where the **input** carries more. That is a statement
about **coupling** — which frequencies the airframe selects out of what it is given —
and no comparison of peaks could have made it. It is still a comparative claim: it is
tested against the aircraft's own linearised dynamics, not against a recorded spectrum,
because none is held. §5 says what would change that.

**What it does not buy.** Any statement of the form *"the load will be X g"*. The
headline load comparison reaches ~~**68%**~~ ~~**64.5%**~~ ~~**64.4%**~~ ~~**75.3%**~~ **70.2%** of a recorded peak-to-peak (1.896 g of 2.70), flown with Mehta's field **replayed on the path it was identified along**. It read 75.3% from the end of session
30 until `boeing747` declared Table IX-4's alpha-dot pitching derivative as `Cmadot`: the term damps exactly this response, so the load falls. **That is the second time a SOURCED derivative has improved agreement with the model's own source and moved this figure away from the record** — the speed derivatives were the first — and it is recorded rather than resolved, because a derivative is not chosen by what it does to the headline. Flown at the 747's own altitude, as every figure before that was, it reads 64.5% with the thrust line at 5.70 ft — it was 68.2% until `boeing747` declared CR-2144's speed derivatives,
which **improved the model's agreement with its own source's modes and moved this figure
further from the record**, through a first-order tangent extrapolated across the encounter's
Mach excursion; §4 has both halves. **It also carries a ~10-point method choice found later in
session 30**: the 747 climbs past two cores that Mehta placed relative to the DC-10's own path.
Replaying the fitted wind along the nominal path reads **75.0%** (**75.3%** with the thrust line). ~~Which form is right is an
open question (§8).~~ **Parks 1985 Fig. 6 then showed the DC-10 did not climb through the
pair**, so the replayed 75.0% is the reading of Mehta's field (§4). ~~The shipped headline flight
has not been switched to it; that is a decision still to take (§8).~~ **It has been switched**, and
measuring it overturned the saturation argument §5 had used to exclude gust amplitude (§4, first entry). (67.1% is the like-for-like *translational* figure
quoted against JSBSim, and this line once carried it by mistake.) §5 records what is left to
**Session 29 banded the as-flown figure**: 68.2% on the 747 before session 30, 57.1–74.0% with every priced input at its bound, and concluded the shortfall survived all of it. That band was measured flown at the aircraft's own altitude, and **session 30's replayed headline found the peak load is not saturated in that form** — a vortex strength 15% above Mehta's fit reaches +1.7 g inside the linear range — so the band has not been re-measured on the headline as it now stands and must not be quoted against 75.3%. explain the shortfall. **Session 27 unsettled that
attribution twice over**, and §5 now says so: the wing-loading sign that made *the
aircraft* the surviving explanation rests on a ratio that is not pinned, and the digitised
record shows the *wind* the model flies is ~12% weaker than the wind the record carries.
Absolute agreement was ruled structurally out of reach in session 7 and nothing since has
changed that.

**Three claims a reader could reasonably infer and should not.**

1. **Structural fidelity.** The derivatives are a flexible airframe's; the solver
   integrates a rigid body (ASSUMPTIONS B1). Never assert it.
2. **Behaviour away from trim.** Derivatives are frozen across the envelope
   (ASSUMPTIONS C3, unbounded). Quote the excursion with every result.
3. **Lateral behaviour, from the lateral MODES.** Dutch roll agrees to 0.4% *as an
   eigenvalue*. That is a statement about the linearisation, not about what the aircraft
   does when a gust rolls it — which nothing measured until session 24, and which is
   still not validated against any source.


## 2. Architecture

| Module | Responsibility | Notes |
|---|---|---|
| `units.py` | conversion constants only | no logic; factors are never inlined elsewhere |
| `verification.py` | **tier 0** — `fitted_order`, `oscillator_refinement`, `fixed_control_refinement`, `newton_residual_history`, `torque_free_omega`, `without_aerodynamics`, `free_fall_through_a_swinging_wind` | takes **no aircraft data as a reference**; a failure here is a defect in the core. Every check lives here rather than inside its test, so the notebook runs the same code the suite asserts on |
| **`cr2144_mach.py`** | **CR-2144's 747 Mach sheets, digitised, and what checks them**: `curves`, `value`, `perturbed`, Tables IX-3/IX-4 at FC3–10, `backsolve` (Appendix A inverted), `mach_increment`, `modes`, `errors_vs_ix5` | added session 30. Reads `atisim/data/cr2144_p220_222_digitised.csv` — the hand-placed points only, never extrapolated. **No model code imports it.** It is the evidence behind `Aircraft.CL_M/CD_M/Cm_M` and the instrument that prices them |
| `validation.py` | **tiers 1–2** — `longitudinal_matrix`, `to_stability_axes`, `to_imperial_matrix`, `longitudinal_modes`, `lateral_modes`, `Reference`/`REFERENCES`, `CAUGHEY_A`, `sweep`, `affine_fit` | the linearisation lives here, not in `tests/modes.py`, which is now a re-export. Every reference number carries its citation as a `Reference.source` field, enforced by a test. **`longitudinal_matrix` is the 4-state, constant-density model** — right for comparing against CR-2144's own published matrices, which are constant-density too, and **not the system this simulator integrates**: §5.21 and `gust._linearise` |
| **`docs/ASSUMPTIONS.md`** | not code — the **assumption register**: what the model assumes, why, and a measured bound on each | this document records what has been *measured*; that one records what has been *assumed*. Read it before quoting any result to better than ~0.5%, before flying far from a trim point, and before adding a wind field whose scale approaches a wingspan |
| **`provenance.py`** | the **ledger**: a constant's category and citation, as data — SOURCED / DERIVED / CALIBRATED / DECLARED | `test_provenance.py` enforces the entries' internal consistency; coverage is enforced separately and only over five modules' module-level constants — see §2's point 4, which corrects what this row used to claim. Answers "which numbers are bulletproof?" as a query rather than a memory |
| **`airframe.py`** | where on the airframe the field is sampled: derived tail arm, sample stations, spanwise loading | the tail arm is DERIVED from `Cmq`/`CLq`, never sourced; the loading shape is DECLARED and carries a measured sensitivity |
| `state.py` | `State`/`Controls`, quaternion utilities | NED inertial, body x-fwd/y-right/z-down; quat is `[w,x,y,z]`, body→NED |
| `atmosphere.py` | ISA to 20 km | two layers — the 747 cruise sits above the tropopause |
| `aero.py` | coefficient build-up; `thrust_force` and `thrust_moment` along the aircraft's thrust line | **takes `vel_rel`/`omega_rel` only; never sees inertial velocity**. The thrust line (`Aircraft.thrust_arm`, `thrust_incidence`) defaults to body x through the CG; `boeing747` declares CR-2144's since session 30 |
| `dynamics.py` | 6-DOF Newton-Euler, `load_factor`, `f_factor`, `average_f_factor`, `thrust_authority` | wind enters here and nowhere else |
| `predictions.py` | sealed predictions | the register of claims made BEFORE their answer is available. Not imported by any model code and deliberately not a source of numbers: nothing here may be quoted as evidence FOR the model. `test_predictions.py` enforces the rules |
| `wind.py` | wind fields and composition | vortex array, updraft column, lee wave, microburst, `superpose`, `field_model`, `along_track_shear`. Session 33 adds three verification fields — `sinusoidal_vertical_field` (V1), `one_minus_cosine_gust` (V3) and `gaussian_vertical_field` (V5's Gaussian control) — and exposes `dryden_vertical_components`, the triple `dryden_vertical_field` sums, so a realisation's exact response variance can be computed rather than approximated |
| `integrate.py` | RK4 `step`, `rollout`, batched rollout | wind sampled once per step and held across the four stages **by default** — `stage_sampled=True` re-evaluates it per stage and buys back three orders, and until session 33 no `vortex_viz` caller could ask for it (§6(i)) |
| `aircraft.py` | three aircraft + `REGISTRY`/`CRUISE`; `boeing747_without_thrust_line()` rebuilds the pre-line 747 exactly, for before/after measurement | every derivative cites its source table; `FlightCondition` + `from_dimensional_*` do the conversions |
| `sensors.py` | `AirData`, `sense(state, wind_ned)` | **the only supported way to ask what the aircraft is doing**; air-relative where a real sensor is |
| `trim.py` | Newton solve for steady level flight | still-air by construction, and must stay so |
| `autopilot.py` | cascaded PID | per-aircraft gains; bumpless engage |
| `manual.py` | manual control, mode switching, pitch trim | trim moves the stick's centring point, never `controls` |
| `panel.py` | live cockpit, instruments, `Stick`, `LiveSim`, `run_live` | basic T + test overlay; takes a `wind_model` and a `field_range` |
| `viz.py` | `Trajectory`, `Recorder`, `derived`, `post_flight` | the log and the post-flight figure only; no simulator needed to read a run |
| **`checks.py`** | **tier 3 — RUN checks**: `quaternion_norm`, `field_divergence`, `energy_closure`, `energy_residual_profile`, `trimmed_start`, `alpha_band`, `lateral_symmetry`, `recorded_wind_matches_field`, `run_checks` | `verification`/`validation` ask whether the MODEL is right, once, in the suite. This asks whether ONE RUN is sensible, every time one is flown. Each check carries a `kind`: **gate** (can and does fail), **tripwire** (has never fired — renders as a number and the word, never a green tick), **report** (a number with no honest threshold). Every check has a **negative control** in `test_checks.py` |
| **`analysis/`** | `artifact.py` (run artifacts: Parquet + `meta.json` + `checks.json`, and `rebuild_field`), `series.py` (every plotted channel), `figures.py` (pure Plotly figures) | needs the **`ui` extra**. Imports `atisim`, never the reverse. Nothing in `atisim/` proper imports it, so the simulator and every script keep working without it |
| **`apps/`** | `sweep.py` — the Dash analysis UI | **the only package that imports Dash, and it computes nothing.** It never runs the simulator either: `n_steps` is a `static_argname`, so every distinct dt pays a fresh 0.6–0.9 s compile and a panel whose contents depend on machine warmth is not a check |
| `vortex_viz.py` | encounter analysis and the Fig. 8 figure | air-relative throughout; deliberately separate from `viz.py`. `fly` for a wind field with fixed controls, `manoeuvre` for an elevator schedule at zero wind; both go through `_measure`, so the three Fig. 8 points cannot drift apart. **TM-102186's** Fig. 8, a different figure: `MECHANISM_FLEET`, `fly_mehta`, `excursion` and `traverse_ratio`, moved from `scripts/cat_validation.py` in phase 5 so the suite can assert the fleet ordering |
| **`response.py`** | **tier 3 — RUN statistics**: `spectrum`, `peak_frequency`, `exceedance` | added session 25 (phase 2). A run as a SPECTRUM and as a RATE, rather than as a peak. Numpy, takes a sampled history, same standing as `checks.py` — nothing here is jitted or differentiated. Note the name collision worth keeping straight: `wind.dryden_spectrum` is an INPUT spectrum, this is the RESPONSE. Every unit check in `test_response.py` is against a signal whose answer is closed-form |
| **`gust.py`** | **tier 1 — CLOSED-FORM GUST RESPONSE**: `gust_transfer` (H(Ω) from vertical gust to n_z), `mean_square_ratio`, `realisation_mean_square_ratio`, `sears`/`theodorsen`/`kussner_attenuation`, `pratt_walker`/`mass_ratio`/`alleviation_factor`, `measure_gust_transfer`, `fit_at_frequency`, `steady_state_seed` | added session 33 (phases V1–V4). `wind.py` builds gust FIELDS and `response.py` analyses the series a run produces; **nothing between them said what the load SHOULD be.** It differentiates the same `dynamics.derivatives` the simulator integrates, so it is **not an independent aerodynamic model and must never be quoted as one** — what it is independent of is the whole time-domain path, which is what V1 falsifies. Its linearisation carries **five** states: height is a real state here because `dynamics` reads `density(altitude)` afresh (§5.21). Scipy for the Bessel and Hankel functions; no JAX in the closed-form half |
| **`sensitivity.py`** | **the derivative of a RESULT with respect to a COEFFICIENT**: `implicit_trim_jacobian`, `longitudinal_matrix_jnp`/`lateral_matrix_jnp`, `plant_matrix_sensitivity`, `eigenvalue_sensitivity`, `eigenvalue_separation`, `mode_sensitivity`, `elasticity` | added session 29 (phases S0/S1). Everything else here measures the model against a SOURCE; this measures it against ITSELF. Reports **elasticity** `(∂Q/∂p)(p/Q)`, never a raw gradient, because a per-radian derivative and a mass are not otherwise rankable. Carries `INDEPENDENT_FIELDS` and `COUPLED_FIELDS`: **`Aircraft` is NOT a set of independent parameters** — `inertia_inv` is the inverse of `inertia` and `AR` is b²/S — so a naive `jacfwd` over the whole tuple is wrong, and those five fields are refused rather than screened. `mode_sensitivity` returns roots UNSORTED, because sorting is what makes a swept mode discontinuous where two cross |

### The two interfaces turbulence depends on

1. **Aero is air-relative.** `dynamics.py` forms `vel_rel = vel_body - dcm.T @ wind_ned`
   and `omega_rel = omega - omega_gust`, then passes only those to `aero.py`. The
   Coriolis, gyroscopic and kinematic terms deliberately keep the **inertial** velocity
   and rate — a gust changes the flow the wings see, not the airframe's ground velocity.
   Substituting `vel_rel` into the Coriolis term breaks Galilean invariance; adding an
   explicit `-m·dW/dt` term double-counts. Both are classic gust-modelling errors.
2. **A PRNG key is threaded through `step`.** Deterministic components return it
   untouched, so a batch of keys varies only the stochastic part — every member of an
   ensemble meets the same vortex at the same place.

### Wind model contract

```python
wind_model(wind_state, state, key, dt) -> (wind_ned, omega_gust, wind_state, key)
```
`wind_ned` is NED; **`omega_gust` is body-axis** (it is subtracted from `state.omega`).
`state.pos_ned` is available, so a spatial field needs no signature change.
`field_model(field)` wraps any position-only field and derives `omega_gust` from its
analytic gradient, so a component cannot contribute a translational gust while silently
omitting its rotational one.

Gust-rate signs, derived from the repo's own conventions:
`p_gust = +∂w_g/∂y`, `q_gust = −∂w_g/∂x`, `r_gust = +∂v_g/∂x`.

**These three signs are independently verified** against Stengel, *Flight Dynamics* 2nd ed.,
eqs. 3.4-48, 3.4-50 and 3.4-52, by re-deriving them from `v_rel = v_cg + ω×r − w_g(r)` rather
than transcribing them. **Two equations in that source are wrong** — eq. 3.4-49's sign, and
eq. 3.4-55 by a factor of −2 — and
`docs/design/specs/2026-08-14-wind-shear-fidelity-design.md` §2 records which, with the
rigid-rotation self-consistency test that found them. **Read it before changing any sign here.**

3. **Gust gradients may be sampled rather than differentiated.** `wind.gust_rates` takes the
   analytic Jacobian at the CG; `wind.sampled_rates` fits the slope across the airframe using
   `airframe.stations`. The two agree *exactly* for any field that is linear across the
   aircraft, which is asserted — so this is a better estimator of the same quantity, not a new
   one. `wind.strip_roll_moment` goes further and integrates the field per strip, which is the
   only form that carries curvature.

   **`field_model` remains the default and still uses the tangent.** A test asserts the two
   disagree at the vortex core edge, because that is what proves the default has not been
   switched over — §4's frozen baselines sit downstream of it.

4. **Constants carry their provenance, and the ledger's reach is bounded.**
   `atisim/provenance.py` classifies each entry as SOURCED, DERIVED, CALIBRATED or
   DECLARED, and `test_provenance.py` enforces that DERIVED chains name inputs that exist,
   are acyclic, and bottom out in something sourced. **This document used to add "a
   constant added without a ledger entry fails the build", and that was not true**: every
   test in `test_provenance.py` iterates the ledger against itself, so nothing checked
   coverage. The audit measured it — roughly 346 non-trivial numeric literals across nine
   physics modules against 13 entries, about **2%**. The missing direction now exists as
   `test_audit_regression.py::test_the_provenance_ledger_does_not_cover_the_source_modules`,
   and the true statement is narrower: **a NEW module-level constant in `aero`, `airframe`,
   `atmosphere`, `trim` or `wind`, added without a ledger entry, fails the build.**
   Constants inside functions, the other four physics modules, and the aircraft data in
   `aircraft.py` are not covered. The recorded baseline may only ever shrink — a second
   test fails if a name is ledgered and left in it.

## 3. Sources

| Source | Supplied | Known gap |
|---|---|---|
| NASA CR-2144 (Heffley & Jewell 1972), §IX | 747 geometry, inertia, dimensional derivatives, transfer-function factors, drag figure; **and Table IX-2, a complete non-dimensional POWER-APPROACH set** | no non-dimensional *cruise* set; **no buffet-onset data at all**; Figure IX-1 and Table IX-3 disagree on the approach inertias by up to 6% (§4) |
| McCormick (via a worked example) | Cherokee PA-28-180 dimensional derivatives | no second source for the lateral set; `Izz < Iyy` flagged by its own author |
| Roskam / USAF DATCOM via PyFME | Cessna 172 non-dimensional tables | rudder derivatives omitted and inconsistent — the whole rudder set is zeroed |
| Nelson / Etkin / McRuer | Navion per-radian derivatives | no extractable published mode table was found; tests assert ranges, not values |
| **Parks, Wingrove, Bach & Mehta 1985**, J. Aircraft **22**(2) 124–129, DOI 10.2514/3.45095 — **HELD, session 26**, `Reference_papers/parks-1985-identification-of-vortex-induced-clear-air-turbulence-JA22-2.pdf` | **the vortex model and both identified cases, first-hand.** Rankine core, array by superposition, Eqs. (3)–(6). Case 1 Hannibal p. 127: r₀ 600 ft, V₀ 85 ft/s, spacing 3500 ft. Case 2 Morton p. 128: 450, 70, 3200. **Both DC-10s.** The abstract gives core *diameters* 900–1200 ft, which settles radius-versus-diameter. His own assessment of the two fits, p. 128, is in `wind.PARKS_FIT_QUALITY` | α is *inferred* from accelerometers through an assumed aero model — see §5. Gives **no** aircraft weight or wing area, deferring aerodynamic characteristics to a 1982 SFTE symposium paper. **Case 2 is the weaker fit and he says so**: "not as good as … case 1", attributed to "strong mountain wave activity which influences the short-period wind pattern" |
| **Wingrove & Bach 1994**, J. Aircraft 31(4) 753–760 | updraft magnitudes/duration, g-load statistics, the Fig. 8 discriminator | never identifies an aircraft type; no updraft edge gradient; no lateral data |
| **Doyle, Jiang, Smith & Grubišić 2011**, *Mon. Wea. Rev.* 139, 3–23, DOI 10.1175/2010MWR3466.1 | **the lee-wave amplitudes** — T-REX Gulfstream V over the Sierra Nevada, IOP 4 primary wave, 6 and 12 m/s crest-to-trough | gives a **tropospheric** wavelength band (20–35 km) and says stratospheric ones are shorter **without a number** — see §5 |
| **Proctor, Hinton & Bowles 2000**, 9th Conf. Aviation Range & Aerospace Meteorology, paper 7.7, 482–487 | **the F-factor** — Eq. (3) `F = U̇ₓ/g − w/Vₐ`, Eq. (4) for the shear term, Eq. (7) for the **1 km average**, the `F > (T−D)/W` thrust criterion, the 0.1/0.13 thresholds, and F = 0.2–0.36 in real accidents | its thresholds are **low-altitude** (§4.1 bounds the threat below 500 m) **and jet-transport only** — it states the scale and threshold "are yet to be determined" for piston aircraft |
| **Oseguera & Bowles 1988**, NASA TM-100632 | **the microburst** — Eqs. (5)–(6), an axisymmetric stagnation flow satisfying continuity, with four stated constants (r/R = 1.1212, z_m/z* = 0.22, z*/ε = 12.5, u_max = 0.2357λR) | the example's `R` is legible only in a scanned figure, so the downdraft radius is declared inside the 1–4 km band Wilson et al. use to define a microburst |
| **MIL-F-8785C** — **now held, `refs/MIL-F-8785C.pdf`, session 25** (5 Nov 1980, 95 pp., everyspec.com) | **the Dryden spectral forms, first-hand.** §3.7.1.2 "Turbulence model (Dryden form)", printed p. 47, prints all three components, and both implemented forms match it **verbatim** — including that the spec gives `v` and `w` identical right-hand sides, which this project had inferred from isotropy and now cites. Also `L_w` = 1750 ft above 2000 ft | **σ_w is still un-digitised.** Figure 7, *"Turbulence exceedance probability"*, printed p. 49, is confirmed present — a rotated scan, axes RMS turbulence amplitude σ (ft/sec TAS) against altitude, curves 10⁻¹ to 10⁻⁶ banded LIGHT/MODERATE/SEVERE. **Holding the document closed the FORMS, not the intensity**; every use still sweeps σ_w and reports what value the result implies. §4's σ_w ≈ 4–5 m/s is an implication, not a validated intensity |
| **Caughey, *Introduction to Aircraft Stability and Control*, Cornell MAE 5070 notes, Ch. 5** | an **independent implementation** of CR-2144's 747 power-approach case: dimensional derivatives Eq. (5.51), plant matrix Eq. (5.52), characteristic polynomial (5.53), roots (5.54) | **not an independent dataset** — its Eq. (5.48)–(5.50) cite CR-2144, the same document §IX comes from. Same inputs, different code. Also states V = 279.1 ft/s (M 0.25 at sea level) where Table IX-2's header says 165 KTAS = 278.49 ft/s, a 0.2% difference |
| **Mehta 1987**, *J. Guidance, Control & Dynamics* 10(1) 27–31 (AIAA 84-2083) | **the only wind field in the project that declares nothing**: the converged five-vortex Hannibal solution — five core positions, `r₀` = 500.5 ft, `V₀` = 86.8 ft/s, ψ = 31°, altitude, bias and trend terms. Also the identification method behind Parks, and the cost at each array size | the fit is to DFDR-derived winds, so it inherits their reconstruction error (bounded by Lester below). States the encounter as **July** 1981 where two NASA documents say April |
| **Wingrove, Bach & Schultz 1989**, NASA TM-102186 | the Hannibal encounter's **measured** normal acceleration (+1.7 to −1.0 g, gusts ~5 s apart); the vortex-array model in words (1,000 ft diameter, 87 ft/s, 3,400 ft spacing); **Fig. 8's three-aircraft simulation** at V = 150 / 700 / 800 ft/s and the mechanism it states | Fig. 8's exact wind field is not recoverable from the paper, so only orderings and excursion ratios can be compared. **Quotes Schultz 1990's Table 1 *initial estimates* as if they were his converged DFW results** — see §5 |
| **Lester, Sen & Bach 1989**, *Mon. Wea. Rev.* 117 1103–1107 | **Table 1: the RMS error of a DFDR-plus-radar wind reconstruction** (2.449 m/s horizontal, 2.236 m/s vertical at V = 250 m/s); a **B-747** mountain-wave encounter at 33,000 ft, +2.7/−1.0 g, 1,000 ft altitude gain; a **measured 22 km lee-wave wavelength** ~1 km above the tropopause | one case, over Greenland rather than the Sierra Nevada that `LEE_WAVE_AMPLITUDE` comes from; no ATC radar fixes, so the track was initialised from the pilot's log and a six-minute mean was removed from the derived vertical velocity |
| **Bach & Parks 1987**, J. Aircraft **24**(11) 789–792 — **HELD, session 26**, `Reference_papers/bach-parks-1987-angle-of-attack-estimation-JA24-11.pdf` | **the error budget on the identification this project's fields rest on.** Eq. (2) gives `C_L` from body-axis accelerations and thrust over `QS`, so **mass and wing area enter only as the ratio `m/S`**. Eq. (4)'s error analysis: the AOA estimate moves **about 0.05° for a 1% error in acceleration**, and "also about 0.05 deg for a 1% error in the lift coefficient" | **contains no DC-10.** Its two validation cases are an **L-1011** and a **B-747SP**, so it does not help §7's acquisition #1. It bounds the input, not the airframe |
| **Ashburn, Waco & Melvin 1970**, AFFDL-TR-70-101 (HICAT), AD878415 — **HELD, session 26**, `Reference_papers/AFFDL-TR-70-101-Ashburn-Waco-Melvin-1970-HICAT-AD878415.pdf` | **measured** high-altitude turbulence: probability densities and **exceedance curves of RMS gust velocity**, from U-2 flights | **its band is 45,000–70,000 ft and this project flies 35,000–45,000** (corrected session 33; it said 33,000–41,000, which matched neither aircraft's declared band)**.** Against MIL-F-8785C Fig. 7 in that band it is an *extrapolated* check, and possibly not an independent one — the report compares itself against **MIL-A-8861A** and against **Steiner's NASA U-2** data, so the high-altitude end of Fig. 7 may descend from the same aircraft. Settling that needs the spec's Background Information and User Guide (ADA119421), which is ~~**not held**~~ **HELD, session 31** (`refs/ADA119421-MIL-F-8785C-background-user-guide-Moorhouse-Woodcock-1982.pdf`, Internet Archive copy, md5 `e410411a…`). **It does not settle the question; it moves it back one report.** The guide never mentions HICAT or the U-2. Its intensities combine Av.P. 970's time-in-turbulence fraction P₁(h) (its Ref. 20, reproduced as its Fig. 34) with one Rayleigh σ distribution (Pritchard, in Chalk et al. AFFDL-TR-69-72, 1969: mode 2.3 ft/s, mean 2.8 ft/s), taken as constant with altitude for clear-air turbulence. Independence from HICAT now needs AFFDL-TR-69-72 or Av.P. 970, neither held |
| **Misaka, Obayashi & Endo 2008**, *J. Aircraft* 45(4) 1217–1229 | **the RMS normal load severity index** — `σ_n` over a moving 5 s average, moderate 0.2–0.3 g, severe ≥ 0.3 g (attributed there to Hamilton & Proctor). Defined at cruise altitude, which the F-factor thresholds are not | its own Figs. 26–27 show `σ_n` tracks the *trend* of measured acceleration and misses the peaks, by construction of the 5 s window |
| **Yoshimura et al. 2022**, *J. Appl. Meteor. Climatol.* 61 503–519 | Tables A2/A3/A5: a **third CR-2144 747 flight condition** — M 0.8 at 6,096 m — with a complete non-dimensional longitudinal set including `C_mα̇`, the flight condition, and the short-period pair (`ω_n` 1.29, `ζ` 0.57) | **not an independent dataset** — Table A2 is attributed to Heffley & Jewell, i.e. CR-2144 again. Same standing as Caughey. Its own conclusion misreads Table A5's `s⁻¹` as `Hz` — see §5 |

### The vortex model, as cited

Parks §"Vortex Modeling", Eqs. (3)–(6): *"a rotational (solid-body) core embedded in an
irrotational flow"*, axis horizontal and perpendicular to the wind vector, with
`r = (ℓ²cos²Δψ + d²)^½`:

| | horizontal `w_xy` | vertical `w_z` |
|---|---|---|
| outside (`r ≥ r₀`) | `V₀r₀d/r²` | `−V₀r₀ℓcosΔψ/r²` |
| inside (`r < r₀`) | `V₀d/r₀` | `−V₀ℓcosΔψ/r₀` |

Arrays are linear superposition. Identified cases, both DC-10s near the tropopause:

| Case | Altitude | r₀ | V₀ | Spacing | Spacing/diameter |
|---|---|---|---|---|---|
| 1 Hannibal MO | 37,000 ft | 600 ft † | 85 ft/s | 3500 ft | 2.92 † |
| 2 Morton WY | 39,000 ft | 450 ft | 70 ft/s | 3200 ft | 3.56 |

**† This table records what Parks reports, and Hannibal's row is no longer what the project
flies.** Session 22 replaced that radius with Wingrove & Bach Fig. 4's **500 ft** (ratio
3.50) — see the note below. `wind.PARKS_CASES['hannibal']['r0']` now carries 500 ft; the
superseded 600 ft survives as `wind.HANNIBAL_R0_SUPERSEDED` with its own ledger entry. The
strength and both spacings are still Parks' and have not moved.

Parks checks that ratio against Scorer's theoretical 2.7 — which is what turned the array
spacing from a free parameter into a cited one.

> **~~Note a source conflict:~~ ~~RESOLVED, session 23 in favour of 500 ft.~~
> **REVERSED, SESSION 26: PARKS IS IN HAND AND THE TABLE ABOVE IS HIS.**
>
> The 500 ft reading is real and is Wingrove & Bach 1994 Fig. 4's, which gives this vortex
> a **1000 ft diameter**. Session 23 adopted it on the reasoning that Fig. 4 was the source
> actually held while "Parks 1985 has never been obtained". Parks 1985 has now been
> obtained — J. Aircraft **22**(2) 124–129, DOI 10.2514/3.45095 — and says r₀ = 600 ft
> with V₀ = 85 ft/s and 3500 ft of spacing, p. 127.
>
> **His abstract settles radius-versus-diameter without any inference.** p. 124: *"the
> vortex cores had diameters in the range of 900 to 1200 ft with tangential velocities in
> the range of 70 to 85 ft/s"* — 900 = 2×450 and 1200 = 2×600. Session 22 had to deduce
> that Fig. 4's column was a diameter from Morton happening to agree; the primary source
> states it.
>
> **And the Scorer check reads differently from the paper than from memory.** p. 129 gives
> Scorer's ratio as *"of the order of 2.7"* and says that for these two cases *"the ratio
> of spacing to core diameter ranged from about 2.9 to 3.5"* — **a range across both
> cases**. This document previously said "Parks quotes 2.92"; he does not. 2.92 is our
> arithmetic on his Hannibal numbers, and that it lands inside his printed range is the
> check. It reproduces at 600 ft and does not at 500 ft, where Hannibal would give 3.50 and
> the lower end of his range would be unreachable.
>
> **~~What remains a hybrid, deliberately.~~ THE HYBRID IS GONE.** From session 22 to 25
> `PARKS_CASES["hannibal"]` paired Fig. 4's *radius* (500 ft) with Parks' *strength*
> (85 ft/s) and Parks' *spacing* (3500 ft) — a vortex no paper states, kept because §4
> baselines sat downstream of it. Session 26 restored Parks' triple and re-measured every
> baseline that moved rather than leaving the hybrid in place; `docs/ASSUMPTIONS.md` **E12**
> carries the four lineages and what the disagreement is worth.
>
> **What the correction cost, and it is in the uncomfortable direction.** JSBSim's Hannibal
> peak-to-peak `n_z` falls from 1.8969 g to **1.8162 g**, i.e. from 70.3% to **67.3%** of
> the recorded 2.70 g. A solid-body core has `dw/dx = V₀/r₀`, so a larger core at the same
> tangential velocity is a *gentler* gradient. **The load shortfall §5 records got worse.**
>
> **Both cases are DC-10s** — p. 127 and p. 128 — which was not previously recorded and
> doubles what a DC-10 derivative set buys: one set serves both validation cases.
>
> **The two fits are not equally good, and Parks says so.** Case 1 shows *"reasonably good
> agreement"*; case 2 *"some agreement, but not as good as that shown previously for case
> 1"*, which he attributes to *"strong mountain wave activity which influences the
> short-period wind pattern"*. Morton therefore carries a contaminant in exactly the band a
> vortex-passage comparison measures. **Quote Hannibal as the primary case and Morton as
> support**, and say which when they disagree. `wind.PARKS_FIT_QUALITY` carries both
> assessments.
>
> **The array spacing is confirmed, and its convention identified.** Mehta's two
> core-penetrating vortices are 4,104 ft apart *along the flight path*. The vortex lines
> run perpendicular to the wind, 31° off that path, so the separation measured
> perpendicular to the lines is 4104·cos31° = 3,518 ft, or 3,522 ft with the 160 ft
> vertical offset in quadrature — **0.6% from Parks' 3,500 ft**, from an entirely separate
> fit. So 3,500 ft is real and it is a **perpendicular** spacing. Read as an along-track
> spacing it would place the cores 15% too close, and nothing else in the project would
> catch that.

## 4. Evidence ledger

Every figure below is measured, with the tolerance the test asserts.

### Gust response against closed-form theory — phases V1–V4 — session 33

**`docs/validation.md` names the project's largest remaining risk as the absence of a
time-history comparison. It is not the only hole, and this is the other one:** every
turbulence figure above was a comparison against a **document**; none was a comparison
against an **exact answer**. `test_verification.py` checks the integrator against a
closed-form solution and measures RK4's order at 3.99982, and nothing did the equivalent
for the response to a gust. Design:
`docs/design/specs/2026-09-21-turbulence-response-validation-design.md`.

New code: `atisim/gust.py`, `wind.sinusoidal_vertical_field`,
`wind.one_minus_cosine_gust`, `wind.gaussian_vertical_field`,
`wind.dryden_vertical_components`. Tests: `atisim/tests/test_gust.py` (18).
Scripts: `gust_transfer_sweep.py`, `gust_psd_identity.py`, `gust_lag_bound.py`,
`discrete_gust.py`.

All four phases are at **`boeing747`, M 0.80, 37,000 ft** — inside `valid_mach` [0.70,
0.90] and `valid_altitude` [35, 45] kft, and the condition every CAT run above is flown
at. V = 236.056 m/s, ρ = 0.349430 kg/m³.

#### V1 — the flown gust response against its own linearisation. PASSES, by 116×

`H(Ω) = C(iωI − A)⁻¹B + D` from the linearised plant, against twelve single-frequency
gusts flown through the full nonlinear path and fitted in amplitude **and phase**. Phase
is not an extra: an amplitude-only comparison passes with the sign of `Cmq` reversed.

| f (Hz) | \|H\| measured | \|H\| theory | amplitude error | phase error |
|---|---|---|---|---|
| 0.0100 | 0.011002 | 0.011002 | −0.00004% | −0.00028° |
| 0.0231 | 0.000954 | 0.000954 | +0.00333% | +0.00171° |
| 0.0534 | 0.009202 | 0.009202 | +0.00061% | −0.00020° |
| 0.1233 | 0.037870 | 0.037870 | −0.00002% | +0.00016° |
| **0.1874** | **0.057662** | **0.057663** | **−0.00074%** | **−0.00071°** |
| 0.4329 | 0.042460 | 0.042462 | −0.00430% | −0.00053° |
| 1.0000 | 0.038562 | 0.038564 | −0.00411% | −0.00016° |

**Worst over all twelve: 0.00430% in amplitude against a 0.5% gate, 0.00171° in phase.**
The 0.0231 Hz row is a near-null of the transfer function — \|H\| is 60× smaller there
than at the peak — and it agrees to the same relative precision, which is the row that
says the match is not an artefact of a large signal.

**Amplitude independence, so the answer is the model's and not the gust's.** Swept over
2.0, 0.5, 0.125 and 0.03125 m/s at the short period the error reads −0.00089%, −0.00005%,
−0.00000%, +0.00000% — **falling as the square of the gust**, which is the aerodynamic
nonlinearity's signature and nothing else's. At the 0.5 m/s used throughout it is five
parts in ten million.

**V1 had to find two defects before it could pass, and both are recorded separately.** It
did not pass on its first run and the failures were not in the gust path:

| what was wrong | what it cost V1 | where it is recorded |
|---|---|---|
| `vortex_viz.fly` could not turn on per-stage wind sampling | 0.81% at the short period, first order in dt | **§6(i)**, new |
| the linearisation held altitude fixed, as `validation.longitudinal_matrix` does | 6–26% on the free response over 20 s | **§5.21**, which this closes |

#### V2 — the PSD identity. Three errors separated, not summed

`σ_nz² = ∫|H|²Φ_w dΩ`, against N = 24 flights of 1200 s each through
`wind.dryden_vertical_field` at σ_w = 4.4588 m/s (`wind.mehta_residual_ceiling`).

| | value | against | |
|---|---|---|---|
| Ā = √(∫\|H\|²Φ dΩ) | 0.034992 g per (m/s) | — | |
| predicted σ_nz = Ā σ_w | 0.156026 g | — | |
| **flown σ_nz, N = 24** | **0.155665 ± 0.000279 g** | −0.231% | **1.29 se** |
| flown σ_nz/σ_w,record | 0.035544 ± 0.000117 | vs **the realisation's exact ratio** 0.035573 | **−0.081%, 0.24 se** |
| the same, vs the continuous Ā | | +1.577% | 4.70 se |

**The factor of two would have been +41.4% and is absent.** Reading
`wind.dryden_spectrum` as two-sided gives 0.220654 g against a flown 0.155665.

**The 400-component log grid carries 1.634% less variance than σ_w names** — 4.385976
against 4.458840 m/s — so a realisation cannot deliver the intensity its argument asks
for, and charging that to the model would be wrong. `gust.realisation_mean_square_ratio`
computes what a given realisation exactly contains; against **that**, the identity holds
to **0.081%, 0.24 standard errors**. Against the continuous integral it holds to 0.231%
(1.29 se) — looser, and only because the grid's shortfall enters the input and the output
nearly proportionally and cancels in the product. That cancellation is pinned by a test so
a future change to the grid cannot break it silently.

#### V4 — the gust lag, bounded for the first time

`ASSUMPTIONS` **C2** was titled *"no α̇ **or unsteady lag**"* and every number in it is
about α̇ — the lag on the aircraft's **own** motion, which `Cmadot` carries. The lag on the
**gust's** arrival is Sears' problem, not Theodorsen's, and `Sears`, `Küssner`, `Wagner`
and `Theodorsen` appeared nowhere in the tree. **C2 is now split; the gust half is
`ASSUMPTIONS` C12**, which carries the table.

| forcing | k | \|S\| | ∠S | lift lost |
|---|---|---|---|---|
| short period | 0.01819 | 0.9689 | −4.15° | **3.11%** |
| Parks core passage, V/r₀ | 0.02276 | 0.9610 | −4.87° | **3.90%** |

**Integrated over the whole Dryden band rather than quoted at a frequency, applying \|S\|
inside V2's integral reduces σ_nz by 6.78%.**

**The sign is why it is worth raising.** The attenuation is a loss, so this term makes the
simulated load **smaller** — it **widens** §5's 32% Hannibal shortfall rather than
explaining it. A newly-found term that happened to close the project's headline gap would
deserve much more scepticism than one that does not. For scale, `ASSUMPTIONS` E2 prices
the point-gust approximation at 4.4% on the same encounter and calls it the model's
largest self-approximation; this one is the same order and was unbounded.

#### V3 — Pratt & Walker. **The design's predicted sign was WRONG, and the reason is V4**

NACA Report 1206 Eq. (5) with Eq. (7)'s alleviation factor. μ = **78.5974**, K_g =
**0.824408** — reproducing the design's own pre-computed values. Flown through
`wind.one_minus_cosine_gust` from level trim, fixed controls, peak load increment:

| H/c̄ | flown Δn (g) | Δn / Δn_PW | \|S\| at that gust's frequency |
|---|---|---|---|
| 5.0 | 0.037967 | **+25.18%** | 0.6262 |
| 8.0 | 0.036806 | +21.35% | 0.7231 |
| **12.5 — Pratt & Walker's own** | **0.035660** | **+17.57%** | **0.8029** |
| 20.0 | 0.033825 | +11.52% | 0.8688 |
| 30.0 | 0.031210 | +2.90% | 0.9103 |
| 50.0 | 0.026034 | −14.17% | 0.9459 |
| 80.0 | 0.019674 | −35.14% | 0.9664 |

**The design said the model must UNDERSHOOT the formula by about the Sears factor. It
overshoots it, by 17.57% at the gradient distance K_g was fitted at.** That is recorded as
a wrong prediction rather than smoothed over.

**The physical reasoning underneath the prediction survives, and the miss is what makes it
interesting.** The model's own alleviation factor is **0.9693** against K_g's 0.8244: it
does shed load, by pitching into the gust, but it recovers only 3.1% of the 17.6% the
empirical fit contains. The rest of the fit is the unsteady lag, which V4 prices at 19.7%
at that gust's frequency — and **0.8029 × 1.1757 = 0.9439**, turning a +17.6% overshoot
into a −5.6% undershoot. One missing term of the right size and sign accounts for most of
a discrepancy the design got backwards.

Linear in U_de (0.035649–0.035709 over a 16× range) and converged in dt at 0.01
(0.035660 at dt = 0.01 and 0.005), so neither the amplitude nor the step size is what is
being read.

### Two readings of CR-2144 pp. 220–222, compared — and the hand reading is the better one — session 32

**Session 30 read CR-2144 printed pp. 220–222 by hand** (295 Engauge points,
`atisim/data/cr2144_p220_222_digitised.csv`), and every error estimate it made — interpolation,
leave-one-out, a pixel Monte Carlo, the Table IX-4 residuals — was **internal to that one
reading.** An **automated trace of pp. 218–228** (`Reference_papers/CR-2144/csv/`, RANSAC tick
calibration at 300 dpi) was found untracked in session 32. A second trace of the same ink prices
the first from outside. `scripts/cr2144_digitisation_crosscheck.py`; `atisim.cr2144_mach.crosscheck`.

**Sealed first.** `cr2144_two_readings_agree_on_the_good_panels` was committed at `c41888a`,
together with the script, while `crosscheck` did not exist — so no run could precede it. It
predicted agreement within **2% of full scale** on the four panels the automated trace's own
README rates *good*, 2% being that README's stated accuracy.

**Measured: the prediction is WRONG, on one panel of four.** Median |hand − automated| as % of the
panel's full scale, over the hand-placed points:

| panel | confidence | SL | 20K | 40K |
|---|---|---|---|---|
| `CL_α` | good | 0.23 | 0.29 | 0.74 |
| `CD_α` | good | 0.55 | 0.52 | 0.30 |
| `Cm_α` | good | 0.27 | 0.22 | 0.56 |
| **`Cm_M`** | good | 1.32 | **2.40** | **2.02** |
| `CD_M` | fair | — | 1.80 | 9.08 |
| `CL_M` | poor | 12.12 | 12.75 | 21.82 |
| `Cm_α̇` | poor | 7.32 | 11.20 † | 0.91 † |

† the hand 20K and 40K points sit closest to the automated **SL** curve: one of the two readings
has the altitudes mislabelled on that panel, which the automated README warns is "partly
interpretive". All twelve *good* curves sit on their own altitude. **`Cm_M` is biased the same way
at all three altitudes** — a systematic offset, not scatter.

**A disagreement does not say which reading is wrong, so both were scored against the same
reference**: the value Table IX-4 implies at each circled flight condition through Appendix A,
`cr2144_mach.backsolve` — the check session 30 used. RMS of reading − table:

| | n | hand | automated | closer |
|---|---|---|---|---|
| **`Cm_M`** — declared on `boeing747` | 7 | **0.0063** | 0.0146 | **hand, 2.3×** |
| **`CL_M`** — declared | 6 | **0.0358** | 0.1764 | **hand, 4.9×** |
| **`CD_M`** — declared | 3 | **0.0054** | 0.0073 | **hand** |
| `Cm_α̇` | 8 | 0.0674 | 0.6376 | hand, 9.5× |
| `CD_α` | 8 | 0.0101 | 0.0368 | hand |
| `Cm_α` | 6 | 0.0061 | 0.0076 | hand |
| `CL_α` | 8 | 0.0354 | **0.0136** | **automated** |

**What this establishes.** The `Cm_M` failure is the **automated trace's** bias. On all three
speed derivatives the shipped 747 declares, the session-30 hand reading is the better-anchored of
the two independent readings — so **this check supports the declared values rather than weakening
them**, which is the first time they have been checked from outside their own reading. The
automated `CL_M` stays within ±0.03 where the table implies values from −0.23 to +0.32 — it has
traced the wrong line — and its `Cm_α̇` misses FC4 by 1.41, a wrong curve, exactly the two panels its
README rates *poor*.

**What it does not establish, and one thing it corrects.** On `CL_α` the automated trace is the
closer one, so the hand reading carries an RMS error there of 0.035, about 0.8% of the value; no
entry declares `CL_α` from it. **§7 had pointed at the automated `p221_Cm_adot` curve as the route
to a Mach schedule for `Cmα̇`. It is 9.5× further from Table IX-4 than the hand reading, and must
not be used for that** — §7's row is corrected. The pinned test marks `Cm_M`'s band as a strict
`xfail` rather than widening it (`atisim/tests/test_cr2144_crosscheck.py`); a corrected automated
trace would surface as an XPASS.

### What session 29's `mass` row measures — and what it does not — session 31

**Asked directly: "the sensitivity analysis shows mass to have a huge impact, but the results
don't look correct".** The numbers are **right as arithmetic and wrong as a reading.** They
reproduce exactly, there is no bug, and the row does not measure "a heavier aeroplane" or "an
error in CR-2144's weight". `scripts/sensitivity_mass_diagnosis.py` does the separation.
It uses central differences through the shipped `vortex_viz.fly_in_moving_air` and
`validation.longitudinal_modes`, **not** `sensitivity.py`, so it is independent of the AD
machinery. Each hypothesis gets one controlled variant.

**What session 29 did to `mass`.** It moved `ac.mass` alone with the trim re-solved, as every
field in the screen was. `inertia` is a `COUPLED_FIELDS` entry (`inertia_inv` is its inverse),
so it **could not** move, although the design spec listed it in the same block. And
`_boeing_747` **derives** `CLa`, `CL0`, `e` and `CD0` from CR-2144's `W`, because Table IX-4's
derivatives are dimensional. A screen that moves `ac.mass` after construction holds all four at
the old weight's values.

#### On the tree session 29 measured (bare `boeing747`), ±1% unless stated

| variant | load, as flown | load, replayed at the nominal altitude | phugoid ζ | short-period ω_n | short-period ζ |
|---|---|---|---|---|---|
| **`mass`, as session 29 moved it** | **−0.649** | −0.624 | **+1.728** | −0.056 | −0.421 |
| `CLa` | **+0.692** | +0.668 | −0.166 | +0.058 | **+0.422** |
| `mass` + inertia scaled with it | **−0.800** | **−0.445** | +1.554 | **−0.556** | −0.434 |
| `mass`, wave drag off (`kappa_airfoil` ×1.25) | −0.678 | −0.674 | **+0.447** | −0.056 | −0.422 |
| **CR-2144's printed `W`, builder re-run** | **+0.008** | +0.019 | +0.566 | +0.000 | +0.000 |

Every load figure agrees at h = 0.25% and 1% to three decimals. The base reproduces session 29
to the digit: 1.841396 g, 68.20% as flown, and 2.093763 g, 77.55% replayed. At ±10% and ±25% `mass` reads
−0.693 and −0.725, and `CLa` +0.707 and +0.706. **S3's asymmetry is a power law of about
m^−0.7, not an anomaly**: −25% gives +22.03% and +25% gives −14.20%.

**Four findings.**

1. **`mass` and `CLa` are ONE lever, counted twice.** The gust increment goes as
   `CLα·q̄S/W`. So the two elasticities mirror each other to **0.04–0.05** on the load and to
   **0.001** on short-period ζ. "`CLa` +0.692 and `mass` −0.649 lead the load" is one finding:
   **lift slope per unit wing loading leads the load.** It is session 26's "wing loading is the
   only thing that matters", re-measured.
2. **The row does not price an error in the source's weight.** Re-running the builder with `W`
   scaled moves the derived `CLa` with it, and the load elasticity is **+0.008**: the two
   cancel. §0's "the DC-10 acquisition has a price … `mass` −0.649" prices a *heavier aeroplane
   with the same aerodynamics and the same pitch inertia*. That is not what an error in a
   published weight is.
3. **Holding inertia fixed is not a detail.** It means ballast at the CG. Scaling the inertia
   with the mass (constant radius of gyration, a **DECLARED** choice made for this diagnostic)
   moves the load elasticity −0.649 → **−0.800** as flown and −0.624 → **−0.445** replayed.
   **The inertia term changes sign with the flight form**, so no single `mass` number
   describes a heavier aeroplane. It also moves short-period ω_n −0.056 →
   **−0.556**, which would tie `c` (+0.557) at the top of S1's short-period row. The screen
   could not show that row.
4. **The cruise phugoid-damping "`mass` +1.728" is mostly the Korn drag rise, a DECLARED model.**
   - At trim, wave drag is **2% of drag** (0.00100 of 0.04280, M 0.7995, M_crit 0.7153). But its
     Mach slope, 80(M − M_crit)³ = **0.0477 per Mach** — 1.9× CR-2144's sourced total of 0.0251
     (§4, "CR-2144's speed derivatives") — supplies **40% of the phugoid damping**: ζ 0.05534
     with it, 0.03306 without.
   - Mass raises the trim C_L, which lowers M_crit, which steepens that slope by about 5% per 1%
     of mass.
   - Switch it off and the elasticity is **+0.447**, beside the static CD/CL estimate of +0.366.
   - **§4's "the phugoid damping is a `mass` result at cruise" is a Korn result at cruise.**
   - **How fragile that term is.** This is arithmetic of `aero.drag_divergence_mach` and
     `wave_drag` at the cruise trim (M 0.7995, C_L 0.652), not a flight.
     - At the shipped κ = 0.87: M_dd 0.8230, M_crit 0.7153, slope **0.0477** per Mach.
     - κ ×0.95 gives 0.2149, and κ ×1.05 gives 0.0020.
     - At κ ×1.10 wave drag is identically zero. That is S3's saturation.
     - **κ = 0.8829 (×1.015) reproduces CR-2144's sourced 0.0251.** A 1.5% change in a
       DECLARED constant halves the slope. Adopting it would make κ CALIBRATED; it has not
       been adopted.
     - C_L ±1% moves the slope 0.0455 … 0.0500.
     - At fixed C_L the slope runs 0 at M 0.70, 0.0033 at M 0.75, 0.1206 at M 0.83 and
       0.1954 at M 0.85, across the encounter's measured Mach span of 0.72–0.83.

**Two hypotheses ruled out.**

- **Path.** The fixed-control climb at cores 3–4 is **151.5 / 155.0 m** at base and
  150.7–151.6 m across mass ×0.75–×1.25. Replaying the field on the nominal altitude moves the
  elasticity by 0.025. The climb session 30 found is real, but it is not mass-dependent.
- **Korn on the load.** Switching wave drag off moves it by 0.029.

#### On session 30's shipped `boeing747` (speed derivatives + 5.70 ft thrust line)

Measured on `ea74849`, exported with `git archive` to a scratch directory, with `atisim.__file__`
printed there. **That branch does not contain `sensitivity.py`**, so only the central-difference
path was run. Load at h = 0.25% / 1% / 5%; modes at 1%.

| variant | load, **as flown** (1.742058 g, 64.52%) | load, **replayed** (2.034180 g, 75.34%, the session-30 headline) | phugoid ζ |
|---|---|---|---|
| `mass` | −0.602 / −0.597 / −0.574 | −0.699 / −0.699 / −0.754 | **+2.230** |
| `CLa` | +0.664 / +0.659 / +0.639 | **+0.740 / +0.497 / +0.695** | −0.106 |
| `mass` + inertia | **−0.958** / −0.954 / −0.929 | **−0.505** / −0.505 / −0.512 | +1.992 |
| printed `W`, builder re-run | −0.038 / −0.038 / −0.038 | **−0.051 / +0.196 / +0.001** | −0.937 |
| `mass`, `CD_M` re-netted at the perturbed trim | — | — | **+0.535** |

- **The mirror holds** (−0.699 against +0.740), and a wrong printed weight still cancels to
  within 0.05.
- **The inertia term changes sign with the flight form here too, and is larger as flown.** As
  flown it steepens the load elasticity by 0.36 (bare entry: 0.15). Replayed it flattens it by
  0.19 (bare entry: 0.18), because the maximum's share turns positive (+0.137): a 747 with more
  pitch inertia pitches away less from the +1.62 g core.
- **The Korn inflation persists, and it is worse.** `CD_M` is stored **net of the Korn slope at
  the construction C_L**, so moving `mass` shifts the Korn slope while the netting stays put. It
  silently changes the total drag Mach slope the entry declares as sourced. Holding that total
  at its base value (0.0257 here) takes phugoid ζ's mass elasticity from +2.230 to **+0.535**.
- **A hazard for anyone re-running S2 on the replayed headline.** At ±1% two of these rows are
  **sampling artefacts**: `CLa` +0.497 and printed `W` +0.196. The trough moves one sample,
  2962 → 2961, the same sample-placement mechanism S4 found in dt. The 0.25% and 5% columns
  agree with each other and not with 1%. **Quote a peak-to-peak elasticity only with two step
  sizes that agree**, which is what the script now prints.

**What this entry does NOT do.**

- No table above it has been re-computed. Session 29's rows stand as measurements and are
  annotated in place.
- `sensitivity.py` still screens `mass` with inertia fixed. No reparameterisation (`W/S`, or
  mass with a radius of gyration) was added, because which question the screen should ask is a
  decision, not a correction.
- S2, S5 and S6 were not re-run on session 30's model, and the Dryden rms (S5) was not re-run
  with inertia scaled. The approach condition was not measured.
- §1's band (57.1–74.0%) does not contain `mass`, so it does not move. It is built on the bare
  747 and the as-flown field, and session 30 moved both.

### The headline flies the replayed field, the arm is the revised 5.70 ft, and the load is no longer saturated — session 30

**Two decisions taken on request, and what they moved.**

1. **The thrust arm is NASA CR-114494's revised 5.70 ft, not CR-2144's printed 10 ft.**
   - CR-2144's tables match 10 ft better, and 10 ft gives the better phugoid.
   - Neither is a reason to choose it. CR-114494 marks its own 10.0 ft superseded by the
     revised table, and **an arm is not chosen by the answer it gives**.
2. **The Hannibal headline flies Mehta's field replayed on the path it was identified along**
   (`wind.on_identified_path`), because Parks 1985 Fig. 6 shows the DC-10 held its altitude
   through cores 3 and 4 (§4, below).

**Measuring the second decision reversed a result the project has leaned on since session 23c:
the peak load is not saturated.** Most of that "saturation" was the simulated 747 climbing out
of the vortex cores.

**1. The arm.** Against Table IX-5, from `scripts/cr2144_speed_derivatives.py`:

| thrust line on the shipped 747 | phugoid ω_n | phugoid ζ | short-period ω_n | short-period ζ |
|---|---|---|---|---|
| none: speed set only (`0d84eae`) | +4.05% | +3.45% | −1.34% | −11.50% |
| **SHIPPED: 5.70 ft, CR-114494 revised** | **+1.69%** | **+2.83%** | **−1.32%** | **−11.48%** |
| 10 ft, CR-2144 Table IX-3 — not declared | −0.05% | +1.13% | −1.30% | −11.46% |
| 5.70 ft line alone, speed seam shut | −21.13% | +11.83% | −1.15% | −11.33% |

- **Reading uncertainty around the shipped value:**
  - 1 px, 5–95%: frequency −1.36 … +4.56%, damping −1.12 … +8.26%
  - envelope up to 3 px: frequency −8.33 … +7.29%, damping −10.16 … +19.63%
- **CR-2144's own 10 ft sits inside that band**, so the two arms are not separable by this
  retest.
- `test_the_declared_thrust_line_closes_most_of_the_overshoot` pins the arm at 5.70 ft *and*
  asserts that 10 ft would read closer. The cost is recorded, and a later edit cannot quietly
  swap the arm for the better number.

**2. The headline, replayed** (`scripts/cat_validation.py`, dt 0.01):

| | as flown (the headline until now) | **replayed (the headline)** |
|---|---|---|
| shipped 747, `n_z` | −0.372 to +1.370 g | **−0.415 to +1.619 g** |
| peak-to-peak, % of 2.70 g | 64.5% | **75.3%** |
| pitch peak-to-peak | 7.01° | 8.21° |
| bare 747 (before session 30) | 68.2% | 77.5% |

- **The replay is for the 747 headline only.** `fly_mehta` still flies the TM-102186 Fig. 8
  fleet at each aircraft's own altitude. Replaying a slow aircraft pins it inside a core it
  would fly out of, and it was measured:
  - Cherokee |α| **102°**, pitch peak-to-peak 154°
  - Cessna |α| 14°
  - 747 approach |α| 11.5°
  
  Mehta's field was identified from the DC-10, and only the DC-10's record justifies the
  replay.

**3. The saturation result reverses.** Session 23c found that tripling V₀ barely moved the
peak, that the recorded +1.7 g was reached only as |α| left the linear range, and concluded
**"amplitude is excluded"** (§4's "Why no amount of wind helps"; §5). Flown at its own altitude,
each stronger updraft lifts the 747 further above the cores, so the gust it meets grows more
slowly than V₀. Replayed, it cannot climb out. Shipped 747, dt 0.02:

| V₀ × | replayed `n_z` | peak-to-peak | \|α\| | | as flown `n_z` max | \|α\| |
|---|---|---|---|---|---|---|
| 1.00 | −0.423 … **+1.604** | 75.1% | 8.05° | short | +1.370 | — |
| 1.10 | −0.570 … +1.693 | 83.8% | 8.52° | short | | |
| **1.15** | −0.641 … **+1.714** | 87.2% | **8.78°** | **reaches +1.7 g, inside** | | |
| 1.20 | −0.705 … +1.740 | 90.6% | 9.01° | inside | | |
| 1.30 | −0.842 … +1.783 | 97.2% | 9.47° | inside | | |
| 1.50 | **−1.118** … +1.926 | 112.8% | 10.40° | reaches −1.0 g, outside | | |
| 3.00 | … +2.893 | — | 22.2° | elasticity **+1.07** | +1.584 | 8.91° (elasticity +0.29) |

- **A vortex strength 15% above Mehta's fit reaches the recorded +1.7 g with the aerodynamics
  still linear.** The full swing needs about ×1.35, and the −1.0 g trough needs about ×1.5,
  just outside the range. The bare 747 reaches +1.7 g at ×1.10.
- **This points the same way as two independent findings.** Session 27's digitised record says
  the wind the model flies is ~12% light. Lester's reconstruction error is 8.45% of V₀.
- **So the shortfall is not "cornered on the aircraft".** A wind 15–35% stronger than the fit,
  on the path the fit was made along, closes most of it. §5's attribution is superseded, not
  deleted.
- **The favourable corner of the identified parameters** (V₀ × 1.0845, r₀ × 0.85) reaches
  **80.4%** replayed, against 68.0% as flown. Session 23c's "under three quarters" held only
  as flown.
- **What still stands, in the as-flown form it was established in, with every threshold
  unchanged:**
  - the saturation bracket, ×3.0–×3.25 on the shipped entry
  - the favourable corner under 75%
  - the gust spacing on the bare entry
  
  The replayed gust spacing is 5.48 s peak-to-peak and 5.40 s centre-to-centre (+8.0%), inside
  10% as before.

**4. The suite.**
- **Final: 863 passed, 1 skipped.** That is 861 plus two new tests:
  `test_the_replayed_headline_is_not_saturated_and_reaches_the_record_inside_the_linear_range`
  and `test_the_identified_path_replays_mehtas_field_at_its_own_altitude`.
- **The 5.70 ft arm moved five pins, each re-captured with its reason:**
  - phugoid ω_n +0.0169, ζ +0.0283
  - wind hold 2.3583 / 2.3228
  - the RK4 hash
  - Fig. 8 at (2.33327, −1.24627)
- **The headline switch moved two claims.** Both are now asserted as established (as flown),
  plus their measured replayed values: the favourable corner, and saturation.
- No tolerance was widened, and no validated-baseline file moved.

**What this entry does NOT do.**
- `cat_bounds.py`, `cat_uncertainty.py`, `cat_spectra.py`, `cat_ensemble.py` and `lateral.py`
  still fly Mehta's field as flown, and were not re-run on either change. Every figure they
  own is the as-flown form, including the 72.7% favourable corner, the exceedance curve and
  the lateral bank.
- The strip/line forms were not replayed.
- The sealed `dc10_does_not_close_the_hannibal_gap` is untouched. Its text names the field as
  flown on `mehta_hannibal_array`, which is still what that prediction means.
- The replayed V₀ sweep is the shipped model at dt 0.02 under fixed controls, not a fit. No
  V₀ scale is declared anywhere.

### CR-2144's thrust line, declared: the phugoid closes — session 30

> **SUPERSEDED IN PART, later in session 30.** The arm below is **10 ft**, and the shipped
> entry now declares CR-114494's revised **5.70 ft** instead: phugoid **+1.69% / +2.83%**, not
> −0.05% / +1.13%. Everything else here — the seam, the re-referenced trim, the arm sweep, the
> mechanism, the neither-half-alone result — stands. §4's first entry has the numbers.

**`boeing747` now carries CR-2144 Table IX-3's thrust line**: LTH = 10.0 ft below the CG and
XI = 2.50° above body x. With the speed derivatives already declared, **the phugoid is −0.05%
in frequency and +1.13% in damping against Table IX-5**. That residual was the overshoot the
speed-derivative entry below attributed to "the thrust line the engine does not have". The
short period is unmoved. The Hannibal load barely notices: **64.5% → 64.4%** as flown.

**1. The seam, verified inert before anything was declared.**

- `Aircraft.thrust_arm` (m, below the CG) and `thrust_incidence` (rad, nose-up from body x)
  both default to zero.
- `aero.thrust_force` tilts the vector by the incidence.
- `aero.thrust_moment` adds magnitude × arm. It recovers the magnitude by projecting onto the
  line, not with a vector norm, which has no derivative at zero thrust (session 29's `sqrt(0)`
  lesson).
- `dynamics.derivatives` *selects* the moment only where an arm is declared, so no −0.0
  component flips sign.
- With the seam in and nothing declared, 73 tests passed, including the single-bit RK4 hash,
  `test_conservation` and `test_trim`.

**2. The declaration, and what it re-derives.**

- **SOURCED:** LTH and XI (Table IX-3, identical at all eight conditions).
- **DERIVED:** the trim thrust, from the Figure IX-6 drag along the line, T = C_D q̄S /
  cos(α₀ + XI) ≈ 42,200 lb. From that:
  - the aerodynamic trim lift, (W − T sin(α₀ + XI))/q̄S = **0.649**, was W/q̄S = 0.654
  - Cm0 now carries −T·LTH/(q̄S c̄) = **−0.0159**, so trim stays at CR-2144's α₀ with zero
    elevator
- The trim C_L feeds the entry's drag derivation: e, CD0, M_crit, and the net CD_M. So the
  construction is CR-2144's own throughout.
- **Trim:** α 4.619° (source 4.60°), elevator −0.015°.
- **`aircraft.boeing747_without_thrust_line()`** rebuilds the pre-line entry. Checked **bit
  for bit, every field**, against commit `4e97e8c`. Every "bare 747" in the tests and scripts
  is now built from it, so each earlier claim still runs on the aeroplane it was established
  on.

**3. Which arm: CR-2144's tables pick 10 ft.** `ASSUMPTIONS.md` C5 had said CR-2144 tabulates
no arm and quoted NASA CR-114494: 10.00 ft as issued, 5.70 ft revised. Table IX-3 does
tabulate one. The Table IX-4 back-solve of Cm_M against the hand-read curve, at seven
conditions, discriminates between them:

| LTH | 0 | 5.70 | 8 | 9 | **10** | 11 | 12 | 15 ft |
|---|---|---|---|---|---|---|---|---|
| Cm_M residual RMS | 0.0382 | 0.0167 | 0.0090 | 0.0067 | **0.0063** | 0.0080 | 0.0109 | 0.0215 |
| mean | +0.035 | +0.014 | +0.006 | +0.002 | **−0.001** | −0.005 | −0.008 | −0.019 |

**CR-2144's derivatives were built on 10 ft**, so that is what the entry declares, and C5 is
corrected in place. The revised 5.70 ft may describe the real aeroplane better; it does not
describe this data set.

**4. The retest, against Table IX-5** (`scripts/cr2144_speed_derivatives.py`, section 4):

| configuration | phugoid ω_n | phugoid ζ | short-period ω_n | short-period ζ |
|---|---|---|---|---|
| bare — the 747 before session 30 | −18.13% | +13.16% | −1.17% | −11.35% |
| speed set only, thrust through the CG (commit `0d84eae`) | +4.05% | +3.45% | −1.34% | −11.50% |
| **SHIPPED: speed set + thrust line** | **−0.05%** | **+1.13%** | **−1.30%** | **−11.46%** |
| thrust line alone, speed seam shut | **−23.37%** | +10.58% | −1.13% | −11.32% |
| shipped, CR-114494's revised 5.70 ft arm | +1.69% | +2.83% | −1.32% | −11.48% |
| shipped, 8 ft / 12 ft arm | +0.76% / −0.87% | +1.92% / +0.34% | −1.31% / −1.29% | −11.47% / −11.45% |
| *unsourced diagnostic from the entry below: Cm_M + C_m,trim/(M/2)* | *+0.07%* | *+0.64%* | *−1.30%* | *−11.46%* |
| a copy trimmed by elevator, trim not re-referenced | −0.00% | −1.08% | −1.30% | −11.46% |

- **Why it works, and why `ASSUMPTIONS.md` C5 missed it.** A thrust moment does not scale
  with dynamic pressure, so the aerodynamic C_m left balancing it at trim is −0.0159, not
  zero. Since q̄ ∝ u², ∂(q̄Sc̄ C_m)/∂u = ρuSc̄·C_m is an M_u term — exactly the one CR-2144
  Appendix A carries. C5 bounded the moment by folding it into the *aerodynamic* C_m as a
  constant offset. That offset trims to zero with the rest, so it could not show the term.
  C5 measured "≤0.93%"; the phugoid actually moves by points.
- **Neither half works alone.** The line with the speed seam shut takes the frequency from
  −18.1% to **−23.4%**. The M_u term is the partner CR-2144 pairs with Cm_M.
- **The sourced line lands where the unsourced diagnostic did**, within 0.1 and 0.5 points.
- **The reference choice is worth 2.2 points of damping.** Re-referencing the trim, or
  letting the elevator absorb the moment (+0.69°), gives +1.13% against −1.08%. That is
  inside the reading band below. The entry keeps its documented convention: trim at α₀ with
  zero elevator.
- **Phugoid frequency moves about 0.43 points per foot of arm.**

**5. What the hand reading now costs**, around the shipped line (section 5, re-based on it):

| source | phugoid ω_n | phugoid ζ |
|---|---|---|
| interpolation: linear / PCHIP / natural cubic / Akima | −0.05 … +0.95% | +0.63 … +1.43% |
| leave one point out, M 0.70–0.90 | −2.06 … +1.22% | −0.67 … +6.25% |
| Monte Carlo, 1 px, N 4000, 5–95% | −3.15 … +2.87% | −2.87 … +6.66% |
| Monte Carlo, 3 px | −10.26 … +5.63% | −12.09 … +18.21% |
| check residuals, ±2 RMS | −1.65 … +1.53% | −8.52 … +11.08% |

**The frequency result now brackets zero at every realistic reading level, and damping is
still known to about ±5 points.** One CD_M point decides damping, as before.

**6. What it moves outside the modes.**

- **Hannibal, Mehta's field at 37,000 ft**, as-flown (A) and replayed (F) (§4, "The Hannibal
  horizontal wind"):
  - A: 64.5% → **64.4%** (1.7407 → 1.7396 g); pitch peak-to-peak 7.23° → 6.82°
  - F: 75.0% → **75.3%**
  - The line is a phugoid effect; the encounter is a few seconds of short period.
- **The Fig. 8 vortex pin:** Δθ 2.3630 → 2.3115° (−2.18%), Δn −1.2488 → −1.2442 g (−0.37%).
- **The wind-hold pitch values:** 2.3878 → 2.3366 held, 2.3523 → 2.3009 per stage. The
  scheme cost is 1.53%, inside its unchanged band around 1.62%.
- **The suite on the declared tree read 9 failed, 850 passed, 1 skipped**, and every failure
  was accounted for before a test changed:
  - **3 were claims about the speed set** that ran on "the shipped entry". They now run on the
    speed-only entry they were established on, with every band kept, and each says so.
  - **1 recomputed the drag slope at the old trim C_L**, W/q̄S: 0.0284 against the sourced
    0.0251. It now uses CR-2144's aerodynamic trim lift, as the entry does, and the sourced
    band holds.
  - **2 new tests pin the thrust line.** `test_the_declared_thrust_line_closes_the_phugoid`,
    and `test_the_thrust_line_needs_the_speed_derivatives`.
  - **5 value pins were re-captured**, each with its reason at the change: the two phugoid
    mode errors, the wind-hold figure, the RK4 hash and the Fig. 8 pin. **No tolerance was
    widened, and no validated-baseline file moved**: conservation, CR-2144 modes, drag polar,
    Navion and trim all passed unchanged.
  - **The suite on the final tree: 861 passed, 1 skipped**, 1,000 s from the worktree root
    with the tree printed. That is the previous 859 plus the two new tests.

**What this entry does NOT do.**

- `boeing747_approach` and `boeing747_jsbsim` declare no line. The approach set comes from
  Table IX-2 at sea level, and Table IX-3's line was not checked against it. JSBSim's 747
  carries its own engine geometry.
- The 2° inward engine cant and spool dynamics are still absent (C5).
- The thrust moment now scales with throttle. That matters for a powered recovery and
  nothing flown so far, because every encounter holds its throttle fixed.
- `cat_bounds.py`, `cat_uncertainty.py`, `cat_spectra.py`, `cat_ensemble.py`,
  `cat_validation.py` and `lateral.py` were not re-run on the declared line. Their tests pass
  on it.

### The DC-10's altitude through the vortex pair: it did not climb — session 30

**Parks et al. 1985 Fig. 6's altitude panel was digitised, and it answers §8's question.** The
DC-10 passed Mehta's cores 3 and 4 **within ~15 ft of 37,000 ft and 17–61 ft below where the
747's run starts**. By that point the simulated 747 has climbed +512 ft (bare) and +596 ft
(shipped). The DC-10's climb, +436 ft on the barometric trace, came **21.6 s after the pair,
past core 5**. So the as-flown 64.5% headline counts a climb the DC-10 did not make, and
§4's replayed form (F, 75.0% shipped) is the one that meets Mehta's field the way the
identification placed it. Mehta p. 29 had said the path was "nearly straight and level"
(`provenance.py`); this measures the "nearly".

- `scripts/digitise_parks_fig6_altitude.py` reads Parks et al. 1985, J. Aircraft 22(2), Fig. 6,
  printed p. 127, PDF page index 3, at the scan's native 300 dpi. `--pdf` is required.
- It writes `atisim/data/parks1985_fig6_altitude.csv`: 3,370 rows covering both altitude
  traces, plus the load, vertical-wind and true-airspeed panels, which are needed for timing
  and speed.
- `atisim/tests/test_parks_fig6_altitude.py` pins the result as bands and orderings (7 tests).

**1. What the altitude panel holds.** Two traces, and they are not the same quantity.

- **Solid: the measured (barometric) altitude.** Parks p. 127 says its fluctuations "indicate
  localized variations in the flowfield" — pressure, not height.
- **Dashed, "ESTIMATE": the altitude estimated from inertial measurements.** Per p. 126, the
  accelerations are integrated with biases matched to radar (x, y) and to the measured
  altitude (z), and **the winds are computed along that path**. So the dashed trace is the path
  Mehta's cores are placed relative to.
- The two are drawn as one line except over **203.9–226.2 s** after 1:21:00 GMT.

**2. The reading.**

- **Calibration.** The scan is skewed ~0.2°, so each panel is calibrated on its own ticks.
  - x: the minute ticks. Worst residual 1.5–2.2 px (≤1.4 s); the altitude panel reads
    0.62 s per pixel.
  - y: the left-axis ticks at the printed labels. Worst residual ≤0.85 px; the altitude panel
    reads 3.10 ft per pixel.
  - A trace that meets the axis hides its tick, as at true airspeed 480 kt, so matching
    tolerates missing ticks.
- **Separation.** The "ESTIMATE" glyphs are removed as components, and its leader line by a
  straight-line fit at 49.0° (including a one-pixel anti-aliasing fringe that first read as
  dashes). The solid trace is every remaining component of real size; the dashes are the short
  components where the traces part.
- **Where the trace hides under the 37,000 ft axis line**, it reads 37,000 only if it meets the
  axis on both sides. Otherwise the gap is left out: the leader cuts the solid trace twice, and
  those columns first read as 37,000.
- `--overlay` writes `parks-fig6-classified.png`, and it was looked at.

**3. The checks, none used to set a parameter.**

| check | digitised | reference |
|---|---|---|
| load band, Parks p. 127 | −0.96 / **+1.90** g | "+1.7 to −1.0 gs" |
| cruise altitude 1:21–1:24, Parks p. 126 | mean **37,018 ft** (36,994–37,042) | "at 37,000 ft" |
| one clock across panels | load min **212.9 s**, vertical-wind min **213.9 s** | separately calibrated panels |
| an independent figure | — | TM-102186 Fig. 6 trough **213.1 s** (session 27's reader, re-run) |

The +1.90 g peak above the quoted +1.7 is the same finding session 27 made on TM-102186's
redraw (+1.855): the prose quotes the sustained band.

**4. Time to distance.** Mehta's cores are placed in feet and the figure is in seconds.

- **The anchor.** The record's deepest downdraft (213.9 s here) is placed where TM-102186
  Fig. 7 draws the measured vertical-wind minimum: −94 ft/s at **x = +4.55 kft**.
- **The scale is DECLARED as a band**, because no source held says whether Mehta's distance
  is air-relative or ground distance:
  - air-relative: the true airspeed read here, 484 kt → **817 ft/s**
  - ground: that plus Mehta's b_xy × cos 31° → **1,034 ft/s**

| on Mehta's axis | inertial estimate, air-relative | inertial estimate, ground | simulated 747, as flown (bare / shipped) |
|---|---|---|---|
| run start, x −18.4 kft | 37,013 ft | 37,045 ft | 0 ft |
| core 3, x −0.34 kft | **36,996** (−17 since start) | **36,985** (−61) | **+512 / +596 ft** climb by here |
| core 4, x +3.76 kft | **36,993** (−20) | **36,994** (−51) | passes +255 / +310 ft *above* the core |
| core 5, x +12.27 kft | 37,216 (+203) | 37,129 (+84) | — |
| barometric maximum, **37,436 ft** | x +22.2 kft | x +26.9 kft | — |

- **The DC-10 passed 98–109 ft below core 3 and 260–261 ft below core 4**, against Mehta's
  94 and 254 ft. It met the pair from the side the fit says, and the simulated 747 meets it
  from the other side.
- **From the run start to core 4, the estimate stays within 36,968–37,078 ft.** Its lowest
  point, 36,968 ft, is 2.6 s before the deepest downdraft. The measured altitude through the
  spikes swings 37,010–37,224 ft; that swing is pressure, as p. 127 says.
- **Insensitive to the reading that matters.** Both speed readings put cores 3 and 4 inside
  a ~5 s window where the estimate is within ±40 ft of 37,000.

**5. What it changes.**

- **The §8 question is answered: the replayed form is the reading of Mehta's field.** The
  simulated 747 climbs ~500–600 ft before the pair because its controls are held fixed in a
  sustained updraft. The DC-10, flown and recorded, did not. §4's "The Hannibal horizontal
  wind" item 5 and §1 are edited in place.
- **Not changed.** The headline flight in `cat_validation.py`, its pins, and every other CAT
  script still fly the as-flown form. Switching the shipped headline to the replayed form, or
  flying the 747 with an altitude hold instead, is a decision about §1's number and is not
  taken here.
- **Not done.**
  - Pitch angle and true airspeed from the same figure are read only as far as this needed.
    True airspeed is used for its pre-encounter median alone, and pitch is not extracted.
    §7 item 6 stays open for them.
  - The DC-10's own altitude excursion was not flown in AtiSim.

### The Hannibal horizontal wind: which side of the path, and what the headline double-counts — session 30

**TM-102186 Fig. 7's horizontal-wind panel was digitised. It settles the one geometric fact
about Mehta's field that nothing the project compared could see, and it exposes a method
choice inside the headline load worth about ten points of the record.** A core the same
distance above or below the flight path gives the *same* vertical wind; only the horizontal
perturbation changes sign. So which side of the path each core sits on — `wind.py`'s sign of
z — was argued from Mehta's prose until now. §7's inventory item 1.

- `scripts/digitise_hannibal_horizontal_wind.py` reads NASA TM-102186 Fig. 7 (printed p. 3-5,
  PDF page index 6, at the scan's 300 dpi; `--pdf` required) into
  `atisim/data/tm102186_fig7_winds.csv`: **1,460 points**, both panels, the solid MODEL line
  column by column and the dotted ACTUAL curve dot by dot.
- `scripts/hannibal_along_track_wind.py` flies seven fields built from it and draws h1–h7.
- `atisim/tests/test_hannibal_horizontal_wind.py` pins the reading and the sign test as bands
  (8 tests).

**1. The reading, and its control.**

- **Calibration**, least squares, never through the frame. y goes through each panel's printed
  labels (worst residual 0.077 kt and 0.54 ft/s). x goes through the tick marks the DISTANCE
  labels name: 15.19 px per 1,000 ft, worst residual **0.241 kft**. Glyph centres alone gave
  0.550, because a minus sign widens a label.
- **Classification by shape**: the solid line is a few long ink components, the dotted curve
  many small dots, and the legend is boxed out. `--overlay` writes the result over the scan
  (`tm102186-fig7-classified.png`), and it was looked at.
- **THE CONTROL. The digitised vertical MODEL against AtiSim's own vertical wind**, which other
  tests already pin: **RMS 1.35 ft/s** on a ±100 ft/s axis, 95% of columns within 2.80. The
  vertical wind is identical with every core flipped (asserted to 1e-9), so the control
  cannot favour either sign.

**2. The sign test: `wind.py`'s convention is confirmed.** The digitised horizontal MODEL
against AtiSim's horizontal perturbation plus Mehta's b_xy = 149.8 kt:

| AtiSim field | RMS | mean | 95% \|r\| |
|---|---|---|---|
| **z as transcribed in `wind.py`** | **1.58 kt** | +1.40 | 2.54 |
| every core's z flipped | **15.38 kt** | −3.95 | 29.75 |

- **The three horizontal extremes fall where Mehta drew them.** Digitised against AtiSim: minima
  at −6.69 and −6.76 kft, and at +3.44 and +3.26 kft; maxima at +12.13 and +12.28 kft. The
  values agree within 1.6 kt.
- The residual is +1.06 kt far upstream and +2.35 kt far downstream: a drawn offset of 1–2 kt,
  not structure.

**3. The field against the record: the fit misses a sustained tailwind rise.** ACTUAL (the
DC-10's reconstructed wind) minus AtiSim, over 82 horizontal dots, has **RMS 10.14 kt**.

| stretch | dots | horizontal, mean | vertical, mean (64 dots in total) |
|---|---|---|---|
| −25 … −5 kft, before the pair | 28 | −3.52 kt | +0.82 ft/s |
| −5 … +6 kft, through cores 3 and 4 | 16 | **−9.70 kt** | +8.03 ft/s |
| +6 … +26 kft, after the pair | 38 | **+12.08 kt** | **+19.27 ft/s** |

- **The recorded dip between cores 3 and 4 is deeper than the fit** — 116.5 kt at +3.08 kft,
  against 123.1 — **and the peak after them higher**: 174.0 kt at +12.80 kft, against 164.1.
- **The real along-track wind rose through the encounter by roughly 15–20 kt more than the
  five-vortex fit carries.** A constant bias cannot represent that rise.
- **Flagged, not resolved.** Both residuals exceed the √J = 4.46 m/s this ledger derives from
  Mehta's printed cost ("The CAT source pass"): 10.14 kt is 5.22 m/s, and the vertical RMS is
  23.17 ft/s. The dots are TM-102186's redraw, not the samples Mehta fitted, and §8 already
  records that J's units and N are unstated. Which of those explains the gap is not measured.

**4. What the along-track wind does to the 747.** Seven fields were flown, each by the bare
entry (seam shut) and the shipped one. All runs use Mehta's array at 37,000 ft, fixed
controls, dt 0.01 and `cat_validation.py`'s window. Peak-to-peak `n_z` is given as % of the
recorded 2.70 g:

| field | bare | speed set only (`0d84eae`) | **shipped, + thrust line 5.70 ft** (at 10 ft) | pitch ptp, bare / speed set only | airspeed, speed set only |
|---|---|---|---|---|---|
| **A** as flown — `wind.vortex_wind` unchanged, the headline | 68.2% | 64.5% | **64.5%** (64.4%) | 6.70° / 7.23° | 400.8–473.1 kt |
| **B** horizontal × cos 31° (the point form puts all of it along the path) | 70.1% | 66.5% | 66.6% (66.7%) | 6.92° / 6.70° | 412.1–471.0 kt |
| **C** horizontal perturbation removed | **81.8%** | **81.9%** | **81.9%** (81.9%) | 8.03° / 8.08° | 454.4–462.4 kt |
| **D** B plus the recorded miss (item 3) × cos 31° | 78.3% | 79.9% | 79.2% (79.3%) | 7.55° / 7.34° | 413.9–463.8 kt |
| **E** every core flipped — for scale only | 65.5% | 73.8% | 72.7% (72.2%) | 6.51° / 7.53° | 437.4–481.7 kt |
| **F** A, evaluated at the NOMINAL altitude | **77.5%** | **75.0%** | **75.3%** (75.3%) | 8.35° / 8.22° | 428.7–478.0 kt |
| **G** D, evaluated at the nominal altitude | 78.3% | 77.2% | 76.7% (76.4%) | 8.74° / 9.13° | 427.0–483.1 kt |

*The "shipped, + thrust line" column was added later in session 30, when `boeing747` declared CR-2144's thrust line (§4, first entry). It moves no row by more than 1.6 points and no conclusion below; the numbers quoted in items 4–5 are the speed-set-only ones they were measured on.* With the line at 10 ft, A is **64.4%** and F **75.3%**; at the shipped 5.70 ft, **64.5%** and **75.3%**, the 747 climbing +589 ft and passing +481 / +307 ft above cores 3 and 4. **F is the headline since the end of session 30.**

**5. The finding: the headline double-counts the 747's own climb.**

- **Mehta placed cores 3 and 4 94 ft and 254 ft above the path the DC-10 actually flew.**
  On that path the fit's horizontal perturbation there is −12.0 and −26.5 kt.
- **The updrafts before them lift the simulated 747 by +512 ft (bare) and +596 ft (shipped).**
  It passes **+403 and +255 ft above** cores 3 and 4 (shipped: +488 and +310), on the other
  side from the DC-10, and meets tailwind gusts of **+40.6 and +27.2 kt** there. That is the
  opposite sign from the fit's value on its own path, and larger.
- **A tailwind gust cuts airspeed, and with it q̄ and the load.** The shipped 747 drops to
  401 kt in A. Replaying the fitted wind at the nominal altitude (F) gives the 747 the wind
  the fit says the DC-10 met, whatever the 747 itself does. F reads **77.5% bare and 75.0%
  shipped**: +9.3 and +10.5 points.
- **So the 64.5% headline carries a ~10-point method choice no earlier session saw, and the
  other choice is closer to the record.** Neither form is a correction by itself.
  - A is what the project has always flown, and it is consistent physics for a 747 placed in
    that field.
  - F is what the identification means, if the DC-10 did not make the same climb.
  - ~~Which one applies depends on the DC-10's own altitude through the pair. Parks 1985 Fig. 6
    plots that altitude (§7 item 6), and nobody has read it yet. §8.~~ **ANSWERED, later in
    session 30: the DC-10 did not make the climb.** Parks Fig. 6's inertial altitude estimate
    is 36,985–36,996 ft at cores 3 and 4, so F is the reading of Mehta's field. See "The
    DC-10's altitude through the vortex pair" above.
- **The along-track wind was never a Cm_M-only channel.** On the bare entry, with no speed
  derivative at all, removing it moves the load 68.2% → **81.8%** (C). It acts through q̄ and
  always has.
  - What Cm_M adds is sensitivity to *which side*. Flipping the cores moves the bare entry
    2.7 points and the shipped entry 9.3 (E).
  - §7's inventory and item 7 of the CR-2144 entry below said otherwise and are edited in
    place.
- **Once the double count is removed, the fit's missing wind is worth little to the load.**
  Projection plus the recorded miss adds +10.1 and +15.4 points on the as-flown path (A → D),
  but only +0.8 and +2.2 points on the replayed one (F → G).

**6. Caveats, and what this entry does NOT do.**

- TM-102186's ACTUAL is a **reconstruction** from the flight recorder, not a direct
  measurement. The dots are sparse — 82 horizontal and 64 vertical — and x is read to ±0.24 kft.
- Every flight holds **fixed controls**. The airframe is a 747, not the DC-10 that met the wind.
- **The shipped field and the headline are unchanged.** `wind.vortex_wind` still puts the whole
  horizontal perturbation along the path, at the aircraft's own altitude. B, D, F and G exist
  only inside the script. Which one the headline should fly is §8's question, not a decision
  taken here. (The measurement half of that question is answered by the entry above; the
  decision is still open.)
- Mehta's own Figs. 5 and 9 (the same fit, drawn less cleanly) were not digitised.
  ~~Neither was Parks Fig. 6.~~ Its altitude panel was, in the entry above.
- `cat_validation.py`, `cat_uncertainty.py` and the other CAT scripts were not re-run on any
  of B–G.

### CR-2144's speed derivatives, digitised and flown — session 30

**The improvement §4 attributed to the speed derivatives in analysis arrives through the
engine's own linearisation, most of the way. The part that does not arrive is one named term
the engine has no form for.** CR-2144 printed pp. 220–222 were digitised by hand (Engauge
Digitizer, eight `.dig` files, **295 placed points**). Every curve was checked against Table
IX-4 at every circled flight condition. The FC9 set is **declared on the shipped `boeing747`
entry** through a new `Aircraft` seam, and was retested by re-trimming and linearising with
the engine's own `jacfwd`, then priced for reading error. **The decision to declare was taken
after that price was measured**, and the commit before it holds the undeclared state.
`scripts/cr2144_speed_derivatives.py` prints every number below;
`atisim/tests/test_cr2144_speed_derivatives.py` pins them as bands.

**1. The files, and what was wrong with the CSVs that came with them.**

- Engauge stores graph coordinates only for its three axis points. Curve points are screen
  pixels, mapped here through each file's own affine calibration.
- **Every CSV column is the curve its header names.** The exports reproduce the placed points
  to ≤6e-6 at every shared Mach (CL_α 40 kft to 2.2e-3). The order is `SL, 40,000, 20,000`,
  and `40,000, 20,000` for CD_M.
- **But the CSVs are not usable as they stand, and are not used.** The headers cannot be
  parsed as CSV: the thousands separators split `x,SL,40,000,20,000` into six fields over four
  columns. Worse, **every column is extrapolated past its drawn curve** onto a shared Mach
  grid: Cm_M 40 kft reaches **105**, CL_M SL **−47.4**, CL_α 20 kft **13.4**, CD_α 40 kft
  **−4.65**. CD_α is resampled on a regular grid and shares no Mach with its own points.
- **Two artefacts in the `.dig` files were dropped, and the script reports both.**
  - **Nine SL points copied verbatim into CL_M's 40,000 ft curve**, at M 0.35–0.62 where no
    40,000 ft condition flies. That curve now starts at M 0.676.
  - **Eight exact repeats** in CL_α's SL curve.
- **Coverage gaps, recorded rather than filled.** Five readings are NaN rather than
  extrapolated:
  - CL_M's SL curve ends at M 0.618, short of FC4's 0.650.
  - Cm_M's and Cm_α's 40,000 ft curves start at M 0.717 and 0.701, just past FC8's 0.700.
  - Cm_α's 40,000 ft curve ends at 0.898, short of FC10.

**2. The check: the curves against the source's own tables.** CR-2144 Appendix A (printed
pp. A-16, A-17, body axis) relates the plotted non-dimensional derivatives to Table IX-4's
dimensional ones, **including** the `W₀/U₀` and `(M/2)·C_M` terms. Each (u, w) pair then
inverts in closed form. Digitised (linear) minus table, over every circled condition a curve
spans:

| curve | conditions | worst | RMS |
|---|---|---|---|
| Cm_q | 8 | **0.9%** of value | 0.122 |
| Cm_α | 6 | **0.7%** | 0.0061 |
| CL_α | 8 | **1.5%** | 0.035 |
| Cm_α̇ | 8 | **4.3%** | 0.067 |
| **Cm_M** | 7 | abs **0.0105** | **0.0060** |
| CL_M | 6 | abs 0.067 (FC10, on the steep limb) | 0.036 |
| CD_M | 3 | abs 0.0062 | 0.0049 |
| CD_α | 8 | abs 0.019 | 0.010 |

**The finding the check produced: CR-2144's aerodynamic `C_m` at trim is not zero.**

- **Why it is non-zero.** Table IX-3 gives a thrust moment arm LTH = 10.0 ft and a thrust
  inclination XI = 2.50° at every condition. The aerodynamic moment at trim therefore balances
  thrust — C_m = −0.0159 at FC9 — and Appendix A's M_u and M_w both carry it.
- **What it does to the check.** With the term, the seven Cm_M residuals have RMS **0.0060**.
  Without it they have RMS **0.0382**, and the FC9 back-solve reads 0.130 against a curve at
  0.175. **The sign of LTH is fixed by the tables agreeing, not assumed.**
- **Two inputs the check needs that no table prints.**
  - **Trim C_D.** SOURCED at 40,000 ft from Figure IX-6 (0.0747 / 0.043 / 0.0427). DECLARED
    from the FC9 polar at SL and 20,000 ft.
  - **T_u = 0.** DECLARED: XU, ZU and MU are the table's starred forms, which would carry it.
- **How much the C_D assumption costs.** ±0.01 of C_D moves the CD_M back-solve by
  0.022–0.044 and CL_M by 0.001–0.007. That is why CD_M is anchored only where C_D is
  sourced.

**3. The seam, verified.** `Aircraft.mach_deriv_ref`, `CL_M`, `CD_M` and `Cm_M`: CL, CD and Cm
each gain `C_M·(M − M_ref)`, applied after the drag build-up. A negative reference adds an
exact zero. **Verified, not argued.** Declared at the trim Mach, the difference between two
`jacfwd` plant matrices equals Appendix A's six Mach terms, built independently in
`cr2144_mach.mach_increment`, to rtol 1e-9, and nothing else moves. `CD_M` adds to
`aero.wave_drag` rather than replacing it, deliberately — and that is the hazard item 4 finds.

**4. The retest.** The FC9 values read off the 40,000 ft curves at M 0.800 (linear) are
**CL_M +0.1304, CD_M +0.0251, Cm_M +0.1753**. Table IX-4 implies 0.1474, 0.0276 and 0.1700.
They are declared on `boeing747` at M_ref 0.800; every row below is that entry or a copy of
it, re-trimmed, linearised by `jacfwd`, and compared against Table IX-5, printed p. 231:

| configuration | phugoid ω_n | phugoid ζ | short-period ω_n | short-period ζ |
|---|---|---|---|---|
| bare entry, seam shut — the 747 before this session | −18.13% | +13.16% | −1.17% | −11.35% |
| **SHIPPED `boeing747`, FC9 set declared** | **+4.05%** | **+3.45%** | **−1.34%** | **−11.50%** |
| a copy, with CD_M net of Korn/Lock at the *flown trim* | +4.05% | +4.55% | −1.34% | −11.50% |
| … Korn/Lock slope kept as the drag Mach slope | +4.05% | **+21.32%** | −1.34% | −11.50% |
| … digitised CD_M added on top of Korn/Lock | +4.05% | **+39.87%** | −1.34% | −11.50% |
| Table IX-4's own implied set, same seam | +3.86% | +5.26% | −1.34% | −11.49% |
| DIAGNOSTIC: Cm_M + C_m,trim/(M/2) = 0.1357 | **+0.07%** | **+0.64%** | −1.30% | −11.46% |
| digitised speed set + α̇ (digitised Cm_α̇ −6.44, IX-4 CL_α̇ −4.97) | +4.05% | +2.54% | −1.06% | **+0.72%** |
| *`AUDIT.md` §2.3's instrument: IX-4 Xu\*, Zu\*, Mu\* written into the matrix* | *+0.08%* | *+3.11%* | *−0.95%* | *−11.86%* |

- **The improvement arrives.** Phugoid ω_n goes −18.13% → **+4.05%** and ζ +13.16% →
  **+3.45%**. The short period is unmoved, to ≤0.17 points. **It overshoots.**
- **Two rows differ only in where Korn/Lock's slope was subtracted, and it is worth 1.1 points
  of damping.** The entry subtracts it at its own construction point (M 0.800, C_L = W/q̄S),
  which leaves the TOTAL drag Mach slope at the *flown* trim (M 0.7995, C_L from CL0 + CLα·α)
  at **0.0236 against the sourced 0.0251** — 6% low. A copy that subtracts at the flown trim
  reads ζ +4.55%. Both sit well inside the ±5 points the reading itself carries, and the
  entry's form is the one that needs no trim solve to construct.
- **The overshoot is not the reading.** Table IX-4's own set overshoots the same way, to within
  0.2 and 0.7 points of the digitised set.
- **It is the thrust moment.** The engine puts thrust through the CG, so its trim C_m is zero
  where CR-2144's is −0.0159, and its M_u is short the `(ρScU₀/I_y)·C_m` term. Folding that
  term into Cm_M takes both phugoid errors under 1%. **The compensated value is a diagnostic,
  not a declaration**: the sourced fix is a thrust line, ~~which the model does not have~~
  **which `boeing747` declares later in session 30: phugoid −0.05% / +1.13%** (§4, "CR-2144's
  thrust line, declared").
- **The drag term is where a partial correction goes wrong.**
  - The engine already has a drag Mach slope, Korn/Lock's. At FC9 it is **0.0477 per Mach**,
    against a sourced total of 0.0251 (curve) or 0.0276 (table) — **1.7–1.9× too steep**.
  - Adding CL_M and Cm_M while leaving that slope in place makes phugoid damping **worse than
    shipping nothing**: +21.3%.
  - So CD_M must be declared as the source value minus Korn/Lock's slope, −0.0226 here. It is
    the same signature §7 recorded for CLa(M) applied alone.
- **Both families together, all from sources this project holds.** Short-period ζ goes −11.4%
  → +0.7% and phugoid ζ to +2.5%. Phugoid ω_n stays at +4.1%: that residual is the thrust
  moment, and the α̇ terms do not touch it.
- **Why the italic row differs from `test_audit_regression`'s docstrings** (+0.3% / +3.7%):
  those are pre-merge values. This is the same instrument, re-measured on the post-merge tree.

**5. What the hand reading costs.** Mode errors for the sourced-total configuration, via the
verified analytic increment:

| source of uncertainty | phugoid ω_n | phugoid ζ |
|---|---|---|
| interpolation: linear / PCHIP / natural cubic / Akima | +4.05 … +5.01% | +3.88 … +4.66% |
| leave one point out, M 0.70–0.90 (PCHIP) | +2.12 … +5.27% | +2.61 … +9.28% |
| Monte Carlo, 1 px scatter + axis calibration, N 4000, 5–95% | +1.07 … +6.86% | +0.48 … +9.71% |
| Monte Carlo, 2 px | −2.09 … +8.25% | −3.63 … +15.00% |
| Monte Carlo, 3 px | −5.73 … +9.53% | −8.38 … +20.82% |
| the check's residuals at the *other* conditions, ±2 RMS | +2.52 … +5.57% | −4.04 … +13.04% |
| *shipped, for scale* | *−18.13%* | *+13.16%* |

Per +0.01 of each derivative, one at a time:

- Cm_M moves phugoid ω_n **+0.98 points**.
- CD_M moves phugoid ζ **+7.41 points**.
- CL_M moves them +0.20 and −0.35 points.
- The short period moves under 0.2 points in every case.

What that says:

- **The frequency result is robust.** No reading case brings phugoid ω_n within half of its
  shipped error — not 3 px of scatter on every point, and not dropping the most influential
  point (Cm_M at M 0.796, which gives +2.1%).
- **The damping result is not.** CD_M at FC9 sits at the foot of the drag rise, on a sheet
  whose value axis is 0.3 over 327 px, and a single point (M 0.8019) decides it: dropping it
  reads 0.0315 and ζ +9.3%. **Phugoid ζ is known to about ±5 points from this digitisation.**
  Quoting +4.6% without that band would be the error rule 6 exists to prevent.
- **Which pixel level is real: 1 px.** Cm_M's check residuals at six other conditions have RMS
  0.0060, well inside the 1 px Monte Carlo's own ±0.029 (5–95%) at FC9, which sits on a slope
  of −6.6 per Mach. 2 and 3 px are stress cases.

**6. The headline moved, and it moved the uncomfortable way.** Mehta's Hannibal array at
37,000 ft, dt 0.01, the same window `cat_validation.py` uses — the bare entry against the
shipped one:

- Peak-to-peak load **1.8414 g (68.2% of the recorded 2.70 g) → 1.7407 g (64.5%)**, **−5.47%**.
- Pitch peak-to-peak **6.704° → 7.229°**, **+7.82%**.
- **This is the shipped model now, not an analysis-only figure.** Every percentage of the
  record quoted anywhere in this document for the 747 is superseded by it.
- **And it is the least trustworthy number in this entry.** It is a first-order tangent
  extrapolated across the encounter's Mach excursion — M 0.72–0.83, per the unmerged
  session-29 branch (§0), not re-measured here — and across that span the 40,000 ft Cm_M curve
  runs from +0.28 to below zero, so the tangent is wrong at both ends in the same direction.
  Quoted as a direction and a size, never as a load.
- **The mode agreement improved and the load agreement got worse, in the same commit.** Those
  are two different reference classes — tier 2 against the model's own source document, tier 4
  against a recorder trace — and it is §4's own "the errors did not grow, the questions did"
  entry arriving from the other side. **A sourced derivative is not a licence to expect the
  load comparison to close**, and this is the cleanest example the project has of that.

**7. What else declaring the set moved — three mechanism claims, and one latent defect.** The
full suite on the declared tree read **17 failed, 827 passed, 1 skipped**. Every failure was
accounted for before a line of test changed:

- **8 were value pins** on the shipped 747 — the Table IX-5 mode errors, the Fig. 8 vortex
  pin, the single-bit RK4 hash, the wind-hold pitch values, the short period at the Mehta
  altitude, the exceedance rate. Each was re-captured with the reason at the change. **None
  was a widened tolerance.**
- **5 were claims about the pre-declaration engine**, and now run on it (`BARE`, the seam
  shut), with every tolerance kept. Three are the audit's attribution tests, which write
  Table IX-4's Xu, Zu, Mu into the engine's matrix; on the declared entry the w-column already
  carries the Mach terms, so the speed family counted twice. The other two are the
  airspeed-floor test and `test_cr2144_modes`' unaugmented-gap pin. The airspeed-floor test's
  premise — "coefficients constant along this line" — is false once coefficients depend on
  Mach.
- **1 was a real defect, caught by its own test.** `verification.without_aerodynamics` zeroes
  a hand-written list of coefficient fields for the free-fall closed form. It did not know
  about `CL_M/CD_M/Cm_M`, so a "de-aerodynamicised" 747 still made lift in free fall: a
  **0.205 m** position error against a 1e-9 bound. The list now carries them. **A list of
  fields to zero goes stale silently every time a coefficient is added**; the closed form is
  what noticed.
- **3 are mechanism claims that are weaker on the shipped entry.** Each is now asserted twice:
  on the bare entry exactly as established, and on the shipped entry as a measured ordering
  with a tripwire band. The attribution was measured by declaring one derivative at a time,
  and **it is the same term every time — Cm_M**:

| claim | bare | shipped | Cm_M only | CL_M only | CD_M only |
|---|---|---|---|---|---|
| line vs point vortex: `n_z` max moved / increment (claim < 0.10) | 0.064 | **0.160** | 0.129 | 0.057 | 0.060 |
| saturation elasticity at ×3 `V₀` (claim < 0.20) | 0.097 | **0.305** | 0.266 | 0.129 | 0.102 |
| peak line-vortex bank, dt 0.05 | 12.61° | **18.31°** | 17.60° | 13.58° | 12.38° |

- **Why Cm_M.** An oblique line vortex carries an **along-track** gust component, and so does
  a core met off-centre. Without a speed derivative that component only changes q̄ — **and
  "only" understates it**: the horizontal-wind entry above measures q̄ alone as worth 13.6
  points of the record on the bare entry (68.2% → 81.8% with it removed). With Cm_M it also
  changes the pitching moment, so the aircraft pitches into or away from the next core.
  **That is the physics this entry added, doing what it should — and it is the channel the
  tangent is least trustworthy in**, because the along-track gust is what drives the Mach
  excursion.
- **What survived intact.**
  - The saturation bracket's sharp form, which still coincides, now at ×3.0–×3.25.
  - The strip path still moves peak bank by more than a tenth: +17% on the shipped entry.
  - TM-102186's pitch and `n_z`-minimum orderings, with less margin ("The CAT source pass").
  - The six-for-six incidence-gain mechanism.
  - The sealed prediction's short-period band: 0.16407 Hz, inside [0.131, 0.197].
- **The gust spacing moved with the path, not the sampling.** It now reads 5.50 s
  peak-to-peak, +10.0% at dt 0.02 and at dt 0.01 alike; centre-to-centre is 5.40 s, +8.0%.

**What this entry does NOT do.**

- ~~It does not declare the set on `boeing747`.~~ **It does** — asked for directly, after items
  4–6 were measured. The pre-declaration tree is the commit before the declaration.
- ~~It does not model a thrust line,~~ It does not replace or re-fit Korn/Lock. (The thrust line was declared later in session 30 — §4's first entry.)
- It does not schedule anything on Mach from pp. 220–221's α-family curves; those are used
  only to check the reading.
- It does not re-read p. 222 from the PDF independently of the hand digitisation.
- It does not anchor CD_M or CL_M at SL or 20,000 ft beyond the DECLARED-C_D caveat above.
- **The `.dig` originals are tracked**, in `atisim/data/cr2144_dig/` (384 KB, each embedding
  the CR-2144 page crop it was traced on), beside the CSV extracted from them. A curve can
  therefore be re-read or corrected rather than only its result trusted, and a bare
  `--dig-dir` reads them.

### The headline load, differentiated and then swept — session 29 (phases S2–S6)

**The study's own deliverable: `PROJECT.md` §1's headline is no longer a bare point.**
`scripts/sensitivity_load.py`, `scripts/sensitivity_assumptions.py`,
`scripts/sensitivity_ensemble.py`. The run is unchanged — `boeing747` through Mehta's
five-core Hannibal array at 37,000 ft, moving-air start at a 12 r₀ lead, measured over the
array ±2 r₀, dt 0.01, 4,737 steps — the configuration `cat_validation.py:fly_mehta` flies.

**Base: 1.841396 g peak-to-peak = 68.20% of TM-102186's recorded 2.70 g.**

#### The gate: the differentiable path is BIT-IDENTICAL to `vortex_viz._measure`

`_measure` ends in `np.asarray`, so the `Encounter` path cannot be differentiated and
`sensitivity.load_history` is a **second path** to the same channel. Requirement was 1e-12;
measured **0.000e+00** — `np.array_equal` on all 4,737 samples, and the window mask identical.
A tolerance would have admitted a second path that had become a different model.

#### S2 — which of the 747's own numbers the headline load rests on

31 tangents through one 4,737-step scan, 42 s. Elasticity `(∂Q/∂p)(p/Q)`.

| rank | field | elasticity | what it is |
|---|---|---|---|
| 1 | **`CLa`** | **+0.69242** | SOURCED, CR-2144 |
| 2 | **`mass`** | **−0.64869** | SOURCED. **Session 31: this is rank 1 again, mirrored** — the lever is `CLa·q̄S/W`. Inertia was held fixed (−0.800 with it scaled), and a wrong *printed* weight cancels to +0.008 (§4, first entry) |
| 3 | **`kappa_airfoil`** | **−0.33850** | **DECLARED — the Korn technology factor, "~0.87 conventional"** |
| 4 | `Cma` | +0.19233 | SOURCED |
| 5 | `c` | +0.12242 | SOURCED |
| 6 | `sweep` | −0.07100 | SOURCED |
| 7 | `t_over_c` | +0.04414 | SOURCED |
| 8 | `Cmq` | −0.03845 | SOURCED |
| | `CL0`, `e`, `CD0`, `CLq`, `thrust_lapse`, `Cm0`, `CLde`, `Cmde` | \|E\| < 0.02 | |

**`CLa` at the top is session 27's LES result arriving from the other direction** — that run
found the frozen lift slope the dominant identified contributor to the load discrepancy, and
this says the same thing about the load itself, at a different condition, by a different
method.

**The third entry is the one that should not be third.** `kappa_airfoil` is **DECLARED**, not
sourced — a Korn technology factor taken as "~0.87 conventional" — and it outranks `Cmα`, `c`
and every drag coefficient. With `sweep` and `t_over_c` it makes the **wave-drag trio worth
0.454 of summed \|elasticity\| against `CD0`'s 0.0087**: on this run **compressible drag
matters ~52× more than parasite drag**, and the largest single contributor to it is a
declared constant.

**Thirteen fields are EXACTLY 0.000000**, and that is a measurement, not a rounding: `CYb`,
`CYdr`, `Clb`, `Clda`, `Cldr`, `Clp`, `Clr`, `Cnb`, `Cnda`, `Cndr`, `Cnp`, `Cnr`, `max_thrust`.
**Every lateral derivative is there.** Mehta's array is a function of along-track distance
alone, so no lateral coefficient can reach `n_z` — `ASSUMPTIONS.md` E10, stated as a
capability gap since session 24 and now measured as an exact zero.

#### S3 — how far the tangent survives, and where it breaks

A peak-to-peak is `max − min`: differentiable **almost** everywhere, and its gradient belongs
to whichever samples ARE the extremes. `actual/tangent` is 1.000 where the local gradient is
the whole answer.

| field | ±1% | ±5% | ±10% | ±25% | verdict |
|---|---|---|---|---|---|
| `CLa` | 0.997 / 1.002 | 0.986 / 1.014 | 1.013 / 1.029 | 0.952 / 1.086 | **the tangent holds to ±25%** |
| `mass` | 0.992 / 1.007 | 0.964 / 1.121 | 0.930 / 1.205 | 0.876 / 1.358 | usable to ±5%, asymmetric beyond |
| `kappa_airfoil` | 0.861 / 1.151 | **0.060** / 0.886 | **0.035** / 1.240 | **0.014** / 2.119 | **SATURATES upward** |
| `Cma` | 0.988 / 0.995 | 1.247 / 0.623 | 1.563 / 0.158 | 1.502 / **−0.136** | **loses its SIGN by −25%** |
| `c` | 0.971 / 1.014 | 0.882 / 1.111 | 1.281 / 1.236 | 1.007 / 1.376 | usable to ±5% |
| `sweep` | 0.973 / 1.027 | 0.866 / 1.133 | **−0.182** / 0.602 | 0.054 / 0.649 | **saturates upward** |

**Three of the six top fields lose the tangent's magnitude or its sign by ±10%.** Two
mechanisms, both physical:

1. **Wave-drag saturation.** Raising `kappa_airfoil` raises M_crit until the wave-drag term is
   identically zero at this Mach, after which more does nothing: Q is **1.83923 at both +10%
   and +25%, to six figures**. Its rank-3 elasticity is real and **one-sided**. `sweep` does
   the same thing for the same reason.
2. **The extreme changes core.** At −25% of `Cmα` the maximum jumps from sample 2699 to 2819,
   and at +25% to 3188 — a **different vortex** in the five-core array. That is where the sign
   inverts. The extremes moved at *almost every* excursion including ±1%, but by one or two
   samples; only the jumps of hundreds break the tangent.

**So the S2 ranking is a ranking at ±1–5%, and the table says so.** Nothing here supports a
statement about a coefficient being wrong by 25%.

#### S4 — every modelling choice on the SAME axis as the coefficients

`ASSUMPTIONS.md`'s rows each carry a cost in their own units — m/s of gust, per cent of
density, per cent of an in-core Δθ, a fraction of a quadrature calibration — and none is
comparable with any other or with a derivative. Here they are all per cent of the headline load.

| register | variant | load, g | vs base |
|---|---|---|---|
| **C3** | **Prandtl–Glauert on, ref M = 0.7995** | **1.748571** | **−5.041%** |
| C3 | ref M = 0.75 *(sensitivity, not a band)* | 1.897357 | +3.039% |
| C3 | ref M = 0.85 *(sensitivity, not a band)* | 1.549600 | **−15.846%** |
| E4 | wind sampled per RK4 stage, not held | 1.840416 | −0.053% |
| F1 | dt = 0.02 | 1.846017 | +0.251% |
| F1 | dt = 0.005 | 1.845656 | +0.231% |
| F1 | dt = 0.0025 | 1.845401 | +0.217% |
| E2/E10 | strip loads, 9 stations | 1.841396 | **+0.000%** |
| F5 | strip loads, 17 / 57 stations | 1.841396 | **+0.000%** |
| E2 | strip loads, uniform / tapered shape | 1.841396 | **+0.000%** |

**C3's Mach axis is bounded at cruise for the first time: −5.04%.** The excursion is measured,
not assumed — Mach **0.7187 to 0.8257, span 0.1070**, because `aero.py` builds Mach from
`vel_rel` and the gust moves it. That is 3.5× the lee wave's ΔM and 0.27 of session 27's LES
point. `ASSUMPTIONS.md` C3 now carries it.

**The dt rows are NOT an integration-order result and must not be read as one.** dt = 0.02,
0.005 and 0.0025 agree with each other to 0.034% and all sit **+0.22% above the shipped
dt = 0.01**. A convergent integration error would be monotone in dt; this is not. It is
**sample placement**: a peak-to-peak of a sampled signal depends on where the samples fall
relative to the extremum, and the shipped dt happens to straddle it. **The headline load is
therefore ~0.22% low for a reason that has nothing to do with RK4.**

**The five 0.000% rows were PROVEN to be physics rather than plumbing**, because a
0.000% from "roll only" is indistinguishable from a 0.000% from "the load model never ran".
The discriminator is the roll rate: peak \|p\| is **exactly 0** on the point path and
**2.12e-17 rad/s** on the strip path. The path ran; `n_z` genuinely cannot see a rolling
moment; and the round-off-level magnitude is itself E10's statement that the field has no
spanwise variation.

#### S5 — the same screen against an rms, to see whether the ranking is the AEROPLANE or the PEAK

Session 25's protocol unchanged: `wind.dryden_field` at the two ends of the sourced σ range,
100 s flights, first 20 s discarded, moving-air start. **8 seeds**, dt 0.02, and the standard
deviation across seeds is reported beside every mean because most of the table is **not
resolved at that N**.

| field | peak elasticity (S2) | rms, σ = 2.108 | rms, σ = 4.459 | |
|---|---|---|---|---|
| `mass` | −0.649 | **−0.726** ± 0.124 | **−0.839** ± 0.191 | rank 1–2 either way |
| `CLa` | +0.692 | **+0.704** ± 0.115 | **+0.811** ± 0.154 | rank 1–2 either way |
| `Clb` | **0.000000** | +0.084 ± 0.063 | **+0.264** ± 0.125 | **zero on the vortex, third here** |
| `c` | **+0.122** | **−0.080** ± 0.057 | **−0.151** ± 0.072 | **SIGN REVERSES** |
| `Cma` | **+0.192** | −0.031 ± 0.051 | **−0.075** ± 0.071 | **SIGN REVERSES** |
| `kappa_airfoil` | −0.339 | +0.129 ± 0.200 | +0.160 ± 0.261 | **not resolved at N = 8** |
| `Cnb` | 0.000000 | −0.009 ± 0.116 | −0.020 ± 0.251 | **not resolved at N = 8** |

`n_z` rms: **0.07676 g** [0.07313, 0.08116] at σ = 2.108; **0.17141 g** [0.16352, 0.18058] at
σ = 4.459.

**The answer is: partly the aeroplane, partly the peak.** `CLa` and `mass` are the top two
under *both* statistics, at both intensities, with the same signs and magnitudes within 20% —
that pair is a property of the airframe. **Session 31: and it is one property, not two.** The
pair mirrors to within 0.022–0.028 here and 0.043 on the peak, because both enter as
`CLa·q̄S/W`. The inertia-scaled rms was not measured. **`c` and `Cmα` reverse sign**, which no ranking taken
from one statistic could have revealed, and **`Clb` goes from an exact zero to third place**,
because `dryden_field`'s v component is a real lateral input (sideslip, not a rolling gust) where
the vortex array has none. **A sensitivity ranking is a property of the statistic as well as of
the model**, and this project now has the measurement that says so.

#### S6 — the budget, and §1's headline gains a band

A row enters only where a source states a band or the register states a measured cost.
Everything else is DECLARED fixed and named so.

| source of error | low % | high % | kind | provenance |
|---|---|---|---|---|
| C3 Mach, derivatives frozen | −5.041 | 0 | measured | session 29 |
| E12 Hannibal core radius | −4.260 | 0 | measured | session 26 |
| `V₀` ±8.45% | −4.280 | +2.560 | **SOURCED** | session 23c |
| `r₀` ±15% | −2.700 | +5.240 | **DECLARED** | session 23c |
| F1 step size | 0 | +0.250 | measured | session 29 |
| E4 wind held across RK4 stages | −0.053 | 0 | measured | session 29 |
| A2 constant g vs g(z) | 0 | +0.383 | measured | sessions 12, 23 |

| | band | headline |
|---|---|---|
| **linear sum** (needs no independence claim) | **−16.33% / +8.43%** | **57.06% … 73.95%** |
| RSS (needs independence, which is NOT established) | −8.32% / +5.85% | 62.53% … 72.19% |

**`68.20%` becomes `68.2%, and 57.1–74.0% once every priced input is moved to its bound`.**

**The shortfall survives the entire band.** The top of the linear sum is **73.95%** against
100%. Session 23c reached the same conclusion from the **wind** inputs alone and topped out at
**72.7%**; adding the **aircraft's** own priced errors moves that to 74.0% and changes nothing.
**The 32% is not input ignorance.** That was the question §1 could not answer and now can.

**The tabulation-Mach rows are deliberately NOT in the budget.** §4 records that this 747 flies
M 0.800, *"which is the Mach its derivative set is tabulated at"* — the number is sourced, so
±0.05 on it is a sensitivity and not an uncertainty, and summing it would inflate the bracket
with an error the source does not have. It is reported because **a 0.05 error there would be
worth three times the whole rest of this budget**, and nothing before now depended on it.

### Fig. 8's pitch axis does NOT stop discriminating — session 29, and it corrects session 23d

**`scripts/fig8_discriminator.py`, 6 intensities × 32 seeds × 2 categories = 384 flights.**
Session 23d found the vortex–updraft **pitch gap** collapsing from 0.886° to 0.060° across the
sourced σ_w range while the **load gap** survived at 0.343 g, and read that as *"at the top of
the sourced turbulence range they separate on load alone, and a pitch-only reading of the chart
would stop working."* **That inference does not survive a statistic that does not move with N.**

#### Why the old statistic could not answer the question

23d reported the **gap between the extremes** of two clouds. The extremes of a distribution
spread as more samples are drawn, so that gap falls towards zero with effort whatever the truth
is. Measured directly, at fixed σ_w = 2.108 m/s, by subsampling **one** grid so nothing but N
changes:

| N | extremes gap | separability (AUC) | Cohen's d |
|---|---|---|---|
| 8 | **1.5690°** | 1.0000 | 6.435 |
| 16 | **0.7211°** | 1.0000 | 4.672 |
| 32 | **0.6567°** | 1.0000 | 5.222 |

**The gap more than halves between N = 8 and N = 32 while the separability does not move at
all.** Gathering more evidence made 23d's statistic say the clouds were *closer*. That is the
wrong way round, and it is why `atisim/response.py` now carries `separability` (Mann–Whitney
AUC, unpaired) and `standardised_difference` beside `exceedance`.

#### The sweep

`separability` = P(updraft > vortex) for one independent draw from each; 1.0 is perfect
separation, 0.5 is none. **1,024 independent pairs per intensity.**

| σ_w | AUC pitch | AUC load | d pitch | d load | overlapping pairs, pitch | load | peak \|α\| | |
|---|---|---|---|---|---|---|---|---|
| 1.000 | **1.0000** | **1.0000** | 10.264 | 23.500 | **0 / 1024** | **0 / 1024** | 8.75° | *below the sourced range* |
| **2.108** | **1.0000** | **1.0000** | 5.222 | 10.830 | **0 / 1024** | **0 / 1024** | 9.35° | sourced lower bound |
| **3.000** | **1.0000** | **1.0000** | 3.925 | 7.399 | **0 / 1024** | **0 / 1024** | 9.82° | |
| 4.000 | 0.9961 | 1.0000 | 3.405 | 5.012 | 4 / 1024 | 0 / 1024 | **10.40°** | **outside the envelope** |
| 4.459 | 0.9971 | 1.0000 | 3.315 | 4.323 | 3 / 1024 | 0 / 1024 | **10.65°** | sourced ceiling, **outside** |
| 5.500 | 1.0000 | 0.9893 | 3.205 | 3.229 | 0 / 1024 | 11 / 1024 | **11.21°** | **outside**, extrapolated |

**Three results, and the first is the one that corrects the record.**

1. **Inside §1's envelope the discriminator is perfect on BOTH axes.** At σ_w = 1.0, 2.108 and
   3.0: **zero overlapping pairs out of 1,024, on pitch and on load.** The pitch axis does not
   stop working, does not come close to stopping working, and "all but touching" was a property
   of the extremes gap rather than of the clouds.

2. **The extremes gap goes NEGATIVE while the separability stays above 0.996.** At σ_w = 4.0 the
   pitch gap is **−0.2995°** — 23d's statistic reads that as *overlapping* — and yet only
   **4 pairs in 1,024** are misordered. That single row is the whole argument for the change of
   statistic, and it is a measurement rather than an argument.

3. **In standardised terms it is the LOAD axis that degrades faster, not the pitch axis** —
   the reverse of 23d's reading. Across the sweep `d` falls **×7.28 on load** (23.500 → 3.229)
   against **×3.20 on pitch** (10.264 → 3.205), and by σ_w = 5.5 the two axes are equally
   informative (3.205 against 3.229) rather than one carrying the other.

#### What actually binds, and it is not the statistic

**The envelope closes before the discriminator does.** Peak \|α\| reaches **10.40° at
σ_w = 4.0**, past §1's 10° linear-aero ceiling, so the in-envelope window is only
**σ_w ≤ ~3.0** — and the **sourced ceiling of 4.459 m/s sits outside it at 10.65°.**
So: within the range this model may be asked about, the pitch axis never fails; beyond it, the
model cannot be believed, and the question of where the axis *would* fail **cannot be answered
by this aeroplane**. `AUC pitch falls through 0.95` is *not crossed* anywhere in the sweep.

**Session 23d's upper limb was therefore already outside the envelope**, at |α| 10.65° in this
configuration, and nothing in its entry said so — `ASSUMPTIONS.md` E11 records the drift for the
*100 s Dryden* ensemble, which is a different run. That is recorded here rather than in a
correction to the number, because the 16/16 ordering 23d reported still holds.

**One number does not reproduce and the cause is named, not resolved.** 23d reports the pitch
gap at σ_w = 2.108, N = 16 as **+0.886°**; the same configuration here gives **+0.7211°**, 19%
lower. The most likely cause is the session-28 compressibility merge, which moved the Fig. 8
vortex point (Δn −1.261 → −1.265) and every altitude-dependent number with it. **Not chased**,
because the conclusion drawn from that number is the one being corrected anyway.

**What this does NOT establish.** The manoeuvre limb is still flown at zero wind by definition,
so only the vortex-versus-updraft separation is tested on equal terms. N = 32 bounds the
overlap at roughly 1 pair in 1,024, not at zero. And this is one aeroplane at one condition
against one pair of field models — it says what Fig. 8 does for *this* 747 through *these*
fields, not what it does for the DC-10-class records the chart was drawn from.

### The model, differentiated in its own coefficients — session 29 (phases S0, S1)

**The first sensitivity machinery this project has had, and the first time any quantity here
has been differentiated with respect to a COEFFICIENT rather than with respect to a state.**
`atisim/sensitivity.py`, `scripts/sensitivity_screen.py`, 24 tests in `test_sensitivity.py`.
Design: `docs/design/specs/2026-09-10-model-sensitivity-analysis-design.md`.

**Measured on Linux with JAX 0.10.2 / NumPy 2.4.6, not on §10's Windows `.venv`.** The
mode path reproduces exactly there (row B below); two bit-exact rollout pins do not, and
that is measured rather than assumed — see "What does not reproduce on another platform".

#### A. The machinery, against a central difference at the same point — THE GATE

| relation | AD tangent | central difference | relative |
|---|---|---|---|
| `CD0` → ζ_phugoid | 0.76556 | 0.76556 | **5.21e-10** |
| `Cmα` → ω_n,sp² | −0.41049 | −0.41049 | **2.02e-10** |
| \|`Clp`\| → 1/τ_roll | 1.98196 | 1.98196 | **1.03e-10** |
| `Cnβ` → ω_n,dr² | 2.07539 | 2.07539 | **6.51e-11** |

**Gate was 1e-6; worst is 5.21e-10, four orders inside it.** The difference is taken through
`validation`'s own shipped functions, not through the jnp twins the AD path uses — a
difference on the same code would check the arithmetic and not the model.

#### B. Session 11's four slopes, re-run on this platform

| relation | slope here | published | intercept here | published | worst residual |
|---|---|---|---|---|---|
| `CD0` → ζ_phugoid | **0.76994** | 0.76994 | −0.01621 | −0.0162 | 0.31% / 0.31% |
| `Cmα` → ω_n,sp² | **−0.42251** | −0.42251 | +0.27366 | +0.27366 | 1.68% / 1.68% |
| \|`Clp`\| → 1/τ_roll | **2.06872** | 2.06872 | +0.30240 | +0.30240 | 1.73% / 1.73% |
| `Cnβ` → ω_n,dr² | **2.10545** | 2.10545 | +0.26626 | +0.26626 | 0.56% / 0.56% |

**All four reproduce to five decimal places**, seventeen sessions and a platform change later.

#### C. The AD tangent and the fitted slope are DIFFERENT OBJECTS, and the gap is curvature

The tangent at session 11's base point sits **0.6–4.2% off** the fitted slope. That is not an
error and the distinction matters, because it is the whole content of the study's tier 2:

| relation | tangent at base | tangent at range centroid | fitted slope | base/fit | **centroid/fit** |
|---|---|---|---|---|---|
| `CD0` → ζ_phugoid | 0.76556 | 0.76927 | 0.76994 | 0.9943 | **0.9991** |
| `Cmα` → ω_n,sp² | −0.41049 | −0.41552 | −0.42251 | 0.9715 | **0.9835** |
| \|`Clp`\| → 1/τ_roll | 1.98196 | 2.09866 | 2.06872 | 0.9581 | **1.0145** |
| `Cnβ` → ω_n,dr² | 2.07539 | 2.10051 | 2.10545 | 0.9857 | **0.9977** |

**Moving the tangent point to the centre of each swept range closes the gap to about the
fit's own worst residual, in all four cases.** So session 11's ranges are affine to the
residual it reported, the base point sits at one END of each of them, and **an elasticity
must be quoted with the excursion it was taken at.** A ranked table taken at the base point
is a ranking *at the base point*.

#### D. The screen: which coefficients the 747's modes actually rest on

Elasticity `(∂Q/∂p)(p/Q)` — per cent of the answer per per cent of the input. Top rows only;
`scripts/sensitivity_screen.py --json` writes all of them. **One-at-a-time: no interaction
term is measured, and none may be inferred.**

| mode | cruise (M 0.80, 40,000 ft) | power approach (Caughey) |
|---|---|---|
| phugoid ω_n | `Cmq` −0.059, `Cmα` +0.059, `c` −0.057 | `Cmα` +0.178, `Cmq` −0.167, `CLα` −0.163 |
| phugoid ζ | **`mass` +1.728**, `e` −0.410, `CD0` +0.327 | **`c` +6.891**, `Cmq` +3.034, `CD0` +2.241 |
| short period ω_n | `c` +0.557, `Cmα` +0.441 | `c` +0.646, `Cmα` +0.322 |
| short period ζ | `c` +0.473, `Cmq` +0.457, `Cmα` −0.443, `CLα` +0.422 | `CLα` +0.400, `mass` −0.396 |
| dutch roll ω_n | `Cnβ` +0.350, `Clβ` +0.126 | `Cnβ` +0.268, `Clβ` +0.187 |
| dutch roll ζ | `Clβ` −1.743, `Clp` +1.730, `Cnr` +1.240 | `Clp` +1.311, `Clβ` −0.892, `Cnr` +0.794 |
| roll τ | `Clp` −0.626, `Clβ` −0.146 | `Clp` −0.734, `Clβ` −0.083 |
| spiral τ | **`Cnr` −4.253, `Clβ` −3.800, `Cnβ` +3.794, `Clr` +2.922** | `Cnr` −1.568, `Clβ` −1.062, `Cnβ` +1.037 |

**Three things this says that no comparison against a source could have.**

1. **The mean aerodynamic chord `c` outranks `Cmα` for the short-period frequency at both
   conditions** (+0.557 against +0.441 at cruise). `c` is geometry, read once off CR-2144;
   `Cmα` is the derivative every discussion of pitch stiffness is about.
2. **The spiral mode is the fragile one, and it is fragile at cruise specifically** — four
   coefficients above 2.9 in elasticity, against a worst of 1.57 at the approach. A 1% error
   in `Cnr` moves the cruise spiral time constant by 4.3%.
3. **The phugoid damping is a `mass` result at cruise (+1.728) and a `c` result at the
   approach (+6.891).** The dominant coefficient is not a property of the mode; it is a
   property of the mode *at a condition*. **Session 31: at cruise it is mostly a Korn result.**
   Wave drag is 2% of trim drag, but its Mach slope (0.0477 per Mach, 1.9× the sourced total)
   carries 40% of the phugoid damping and steepens with trim C_L. With it off, `mass` reads
   **+0.447** (§4, first entry).

**Nine fields are STRUCTURALLY INERT at both conditions** — elasticity exactly 0.000000,
which means *not used*, not *unimportant*: `CYp`, `CYr`, `CYdr`, `Clda`, `Cldr`, `Cnda`,
`Cndr`, `max_thrust`, `thrust_lapse`. The control derivatives are inert because the linear
modes are taken about a fixed-control trim; the two thrust fields because thrust acts along
body x through the CG with no moment (`ASSUMPTIONS.md` C5). **A zero in an elasticity table
means one of these two things and never "small", so the screen prints them separately.**

**And a fourth finding, from the trim path rather than the modes.** `CLα` has **exactly
zero** effect on `boeing747_jsbsim`, `boeing737` and `boeing737_approach`: those entries
carry a `CL_table_alpha`, and `aero.py` takes the table *instead of* `CL0 + CLα·α`. **Their
`CLa` field is dead data.** No test asserts this and nothing else in the record says it.

#### What does not reproduce on another platform, measured

The suite runs **2 failed, 776 passed, 3 skipped in 811 s** here, **2 failed, 808 passed,
3 skipped in 898 s** once phases S0–S6's 32 tests are in it, and **2 failed, 819 passed,
3 skipped in 1,359 s** after the two follow-ups added 11 more. Both failures are the two
tests that assert **exact bit equality**, and both differ in the 13th significant digit:

| pin | this platform | recorded | relative | ulps |
|---|---|---|---|---|
| `FIG8_VORTEX[0]` (Δθ) | 2.24167400998675 | 2.241674009986879 | 5.77e-14 | ~260 |
| `FIG8_VORTEX[1]` (Δn) | −1.2396439681557032 | −1.2396439681557148 | 9.31e-15 | ~42 |
| `PRE_REFACTOR_VEL_HASH` | (sha256 differs) | — | — | — |

**Neither tolerance was touched and neither should be** — `docs/DEVELOPMENT.md` rule 3, and these are
doing precisely their job: they detect that the arithmetic environment changed. The reading
is that ~~**a rollout of 10⁴–10⁵ steps is bit-reproducible only within one platform**~~ ~~**a
rollout of 10⁴–10⁵ steps is bit-reproducible only within one set of library versions**~~ **a
rollout of 10⁴–10⁵ steps is bit-reproducible only within one set of library versions and one
family of OpenBLAS kernels** (session 32, below), while the linearisation path is stable across platforms to five decimals (row B). The test count
is 781 rather than session 28's 813 because `pyarrow`/`plotly`/`dash` are absent here, so
`test_artifact.py` and `test_figures.py` do not collect.

**Session 32: it is the library versions, not the platform.** CI's Linux runner resolves Python
3.10 to JAX 0.6.2, NumPy 2.2.6 and SciPy 1.15.3 — the versions of §10's Windows `.venv` — and
there **every bit-exact pin passes**, both of these included (run 35370419455: 926 passed, 1
skipped, 1 xfailed, the Windows count exactly). The failures above were at JAX 0.10.2 and NumPy
2.4.6. `pyproject.toml` pins no versions, so an interpreter that resolves newer ones can still
fail the two pins — which is them doing their job, as above.

**Later in session 32: the versions are not the whole of it.** After two merges, `main`'s CI
failed `test_extracting_rk4_step_did_not_move_a_single_bit` with **the same library versions**,
while the pull-request run of the same tree had passed. The failing hash was identical in both
failing runs, so the difference was deterministic per machine. **GitHub's runners are a mix,
and some expose AVX-512.** Measured with two throwaway diagnostic workflows, 18 runner jobs in
all; the last two columns come from the second, on ten:

| Hosts | Default | NumPy's AVX-512 paths off | OpenBLAS held to its Haswell kernels |
|---|---|---|---|
| no AVX-512 — AMD EPYC 7763, EPYC 9V74 as some VMs expose it; locally, AMD Ryzen 5 7535HS | the pin | the pin | the pin |
| AVX-512 — AMD EPYC 9V45, EPYC 9V74 as other VMs expose it, Intel Xeon Platinum 8573C | `031b8db0…` | `031b8db0…` | **the pin** |

**So the cause is OpenBLAS**, the linear-algebra library under NumPy and SciPy, which chooses
its kernels by CPU at run time; NumPy's own SIMD dispatch changes nothing, and neither does
capping XLA's code generation at AVX2 (tried first, on two AVX-512 hosts). The model is the
same: the two kernel families round differently in the last bits, and a 500-step rollout
carries that into the hash. The same CPU model can land in either row, because a VM may hide
AVX-512 — the host's feature flags decide, not the model name. The other bit-exact pin,
`FIG8_VORTEX`, passed on the AVX-512 hosts.

**Fixed in CI's environment, not in the pin:** `tests.yml` sets `OPENBLAS_CORETYPE=Haswell`,
which makes every runner the environment the pin was captured in, and records each runner's CPU
in the log. The pin keeps its single value, and its failure message now names the three things
that move it — the JAX version, the NumPy version and, on an AVX-512 host, the OpenBLAS kernels.

### Why the agreement "got worse": the reference class changed, not the model — session 28

**The question this answers, asked while preparing the presentation:** *the project used to
agree with the papers to a couple of per cent and now reports 20–40% errors — what broke?*
**Nothing broke.** Every early number was re-measured this session and is unchanged to the
digit. What changed is **what the model is being marked against**, and the change is dated.

**Re-measured, this tree, this session.** `.venv/Scripts/python.exe -m pytest -q` from the
worktree root — the one invocation §10's table calls safe — gives **812 passed, 1 skipped,
761.79 s**. §10 recorded 811 at session 26; the extra test is session 27's
`boeing787_yoshimura` assertion. Then, one script per tier:

| Re-run | Result | Against §4's recorded value |
|---|---|---|
| `scripts/checkpoint.py` | phugoid ω_n **0.0553** / ζ **0.0560**; short period ω_n **0.9508** / ζ **0.3425**; trim residual **1.93e-20**; 60 s hold drift **0.0000 m / 0.0000 m/s** | **identical to every digit** |
| `scripts/cat_validation.py` | `n_z` **−0.398 to +1.441 g**; gust −86.8 to +59.1 ft/s; σ_n 0.6394 g; six-aircraft ordering and mechanism unchanged | **identical** |
| `scripts/lateral.py` | point 0.000° bank / line 12.508° / line+strip **15.376°**; `n_z` max 1.6019 / 1.6375 / 1.6344 | **identical** |

> **Superseded, session 30 — kept as what the tree measured on 8 September.** `boeing747`
> now declares CR-2144's speed derivatives, so these rows have moved deliberately. Re-run on
> the declared tree:
> - `checkpoint.py`: phugoid ω_n **0.0700** / ζ **0.0506**, short period **0.9511** /
>   **0.3425**, trim residual **1.11e-16**, 60 s hold drift **0.0000 m / 0.0000 m/s**.
> - `cat_validation.py`: `n_z` **−0.374 to +1.367 g**, σ_n **0.6273 g**; the six-aircraft
>   incidence-gain mechanism is still monotone, six for six.
> - `lateral.py` was not re-run as a script. What its test measures moved, and §4's session-30
>   entry says how far and why.

**And the tolerances were not moved to get there**, which is the other half of the claim and
is checked in git rather than asserted. The five "validated baseline" files carry **three
commits between them** in the project's whole history. Two are the package rename (imports
only — `git show e06914a` on `test_cr2144_modes.py` is 6 changed lines) and the audit repair,
whose entire diff on those files is a **comment** corrected from `~5.7e-14` to the measured
`5.6958e-13` and `is_physical(x)` gaining its `ac` argument. **No threshold in
`test_conservation.py`, `test_cr2144_modes.py`, `test_drag_polar.py`, `test_navion.py` or
`test_trim.py` has ever been loosened.**

> **Provenance of the three re-runs.** `checkpoint.py` and `lateral.py` were run from the
> worktree root; `cat_validation.py` printed `atisim imported from: …\Claude_Flight_Sim\atisim`
> — the **main checkout**, because `python scripts/…` puts `scripts/` on `sys.path` and not the
> cwd, exactly as §10's table warns. It is admitted here only because the two trees were then
> hashed and are **byte-identical**: sha256 over every `.py` under `atisim/` is
> `e9d7e7826e2fc183` in both. Had they differed, the row would have been withdrawn.

#### The four tiers, and the date the project changed tiers

Every comparison this project has ever made falls into one of four classes, and **the class
sets the error, not the code**:

| Tier | What the model is marked against | Typical disagreement | Sessions | Dates |
|---|---|---|---|---|
| **0 · verification** | closed-form mathematics — RK4 order, torque-free rigid body, Newton residual | **1e-13 … 5e-3 %** | 1–12 | 6–11 Aug |
| **1–2 · self-consistency** | **the same document the derivatives were read from** (CR-2144's own mode table) | **0.4 – 3.4 %** | 1–12 | 6–11 Aug |
| **3 · cross-code** | another executing engine **fed identical coefficients** (JSBSim) | **0.02 – 0.08 %** on modes; 3–7% on the phugoid | 17–20 | 20–25 Aug |
| **4 · reality** | a **flight-data recorder** or a **published LES** — measured atmosphere | **20 – 42 %** | 23–27 | 1–6 Sep |

**The inflection is a single commit: `c6b5342`, "Fly Mehta's identified field", 1 September
2026.** Before it, the project had never once compared itself to a measured atmosphere. Every
error above 10% in this ledger post-dates it, and every "couple of per cent" figure predates
it and still stands.

**Why tier 1–2 could never have been anything but small, and why that matters for the talk.**
Tier 2 asks *"was CR-2144 transcribed and unprimed correctly?"* — the model is being marked
against its own input. `ASSUMPTIONS.md` closes on this in its own words: every tier-2
comparison is closed-loop against a document's own arithmetic, *"which is exactly why the age
of that document is not a threat, and equally why the result says nothing about the real
aeroplane."* **A 0.4% Dutch roll and a 32% load shortfall are not the same measurement getting
worse. They are two different measurements, and the project only started making the second
one six days before this session.**

#### Four mechanisms move the *published* numbers the wrong way, and none is a regression

**(1) The model was flown further from where its data was linearised. This is the largest
identified physical cause, and it is `ASSUMPTIONS.md` C3 — "derivatives frozen, Mach axis
UNBOUNDED" — collecting on a debt it has carried since session 12.** Confirmed in the code
this session: `aero.py` carries Mach into **`wave_drag` and nothing else** (plus the thrust
ram term); there is no Prandtl–Glauert factor on `C_Lα`, `C_mα` or any other coefficient.
The cost is measurable and it grows with the excursion:

| Excursion from the M 0.80 / 40 kft linearisation | Frozen-derivative error | Same run with derivatives tabulated at the condition |
|---|---|---|
| **none** (FC9 itself) | short period ω_n **1.4%** | — |
| **altitude only**, M 0.8 / 6,096 m (Yoshimura Table A2/A5) | ω_n **+23.46%** | **−0.60%** |
| **Mach**, ΔM = −0.393 (the LES at M 0.406) | load rms ratio **1.427 / 1.420** on the two resolved domains | matched `boeing787_yoshimura`: **1.202** |

**The middle row is the cleanest statement the project owns about its own biggest limitation:
same code, same solver, same aeroplane, one condition apart — 23.46% against 0.60%, a factor
of 39, entirely from freezing.** Nothing about turbulence enters it.

**(2) The record got more honest, three times, and each time a number got worse on purpose.**

- **Session 16.** The 747 mode table grew from **two** of CR-2144 Table IX-5's four published
  longitudinal factors to **all four**, adding rows at **+14.4%** and **−1.4%**. The pass added
  two *rows*, not two derivatives: 42 of 44 scalars were bit-identical across it, compared as
  hex representations against the tree extracted at the preceding commit.
- **Session 18.** The JSBSim short period read **0.04% → 3.95% → 1.30% → 0.04%**, and only the
  last is honest — the first was two errors cancelling, an α̇-contaminated `C_mα` of −1.0637
  standing in for a coupling the model did not have. Same number, opposite meaning.
- **Session 18, again.** Five fidelity improvements made the JSBSim phugoid **worse**, 3.33% →
  6.58%, and layer 4's doublet divergence 0.483 → 0.558 m/s. Session 19 then localised **96%
  of the phugoid frequency error to one entry of the 4×4**, `M_u`, which a constant-coefficient
  model structurally cannot carry.

**(3) A source correction moved the headline load the uncomfortable way.** Session 26 obtained
Parks et al. 1985 and found `PARKS_CASES["hannibal"]` had been a **hybrid** — Wingrove Fig. 4's
500 ft radius with Parks' 85 ft/s and 3500 ft. Parks' own triple is 600 ft / 85 / 3500, and
`dw/dx = V₀/r₀`, so a larger core at fixed strength is a **gentler** gradient. Cost:
**−4.26% of the headline peak-to-peak, 70.3% → 67.3% of the record.** A correction that makes
your own result worse is the one kind that cannot have been chosen for convenience.

**(4) Seven different percentages exist for one comparison and they are not interchangeable.**
`presentation_package/_evidence/load_percentage_disambiguation.log` lists them: 68.1% (headline),
67.1% (translational-only, deliberately handicapped for JSBSim parity), 76.0%, 74.1%, 74.6%
(JSBSim), 72.7%, 70.1%. **A reader comparing 74.1% from one session against 67.3% from another
sees a regression that is really two different runs.** §1 still carries 67% where the headline
is 68.1% — known, recorded at `known_issues.md` §12, and it is the like-for-like figure attached
to the wrong run.

**And one thing genuinely improved in the direction nobody claims credit for:** the LES ratio
went **1.427 → 1.202** in session 27 when Yoshimura's own aeroplane was rebuilt from their
`fs.f90` and flown in place of the 747.

#### The one-sentence version

**The errors did not grow; the questions did.** Every couple-of-per-cent figure is still a
couple of per cent — re-measured above — and it was small because the model was being marked
against its own input data. The 20–40% figures are the project's **first** comparisons against
measured atmosphere, they are six days old, and the largest single identified contributor to
them is one named assumption: **the derivatives are frozen at M 0.80 and the model is being
flown at M 0.41.**

### The compressibility branch, rebased and merged — what it moved — session 28

**`claude/new-session-943052` was one commit ahead of `main` and 27 behind, dated 26 August
2026, and carried more of §7's plan than the plan knew about.** §0 records how it was found.
It is now rebased onto `main` and merged. **The suite is the gate and it was run twice: red
first, green after three pins were re-captured.**

**What landed, and what it does NOT do yet.**

| | |
|---|---|
| `dynamics.gravity(z) = g₀(R/(R+z))²` | **LIVE.** g(12,192 m) = **9.76922** against 9.80665. Reverses session 12's decision, not its measurement |
| geometric → geopotential ISA conversion | **LIVE.** `atmosphere.geopotential(12,192)` = **12,168.66 m**. §5 called this an assumption; the branch calls it a defect, and it is right — the module used geopotential formulas while every caller passed geometric altitude |
| Prandtl–Glauert on the lift-slope family | **PRESENT AND INERT.** `Aircraft.pg_mach_ref` defaults to **−1.0**, and the code reads negative as "undeclared → factor exactly 1". **No registry entry declares one**, so PG changes nothing today. The infrastructure is in; the opt-in is not taken |
| Mach-scheduled `Cmde` and `Clda` | **PRESENT AND INERT** by the same mechanism — empty tables |
| `PG_MACH_MAX = 0.90` with a floor | the M → 1 singularity cannot reach the integrator. A guard, not a modelling claim |

**So the merge is a gravity-and-atmosphere change, not yet a compressibility change.** Anyone
reading "Prandtl–Glauert merged" and expecting §4's LES ratio to move will find it has not.
**What it buys is that the Mach axis is now one field-assignment away instead of one
implementation away** — and §4's CR-2144 row says printed p. 220 carries the sourced curve to
assign from.

**What moved, measured on the merged tree.** 747 at CR-2144 FC9, against Table IX-5:

| mode | before | after | error before | error after | |
|---|---|---|---|---|---|
| phugoid ω_n | 0.055319 | **0.055099** | −17.80% | −18.13% | **worse** |
| phugoid ζ | 0.055956 | **0.055336** | +14.43% | **+13.16%** | better |
| short period ω_n | 0.950773 | **0.952723** | −1.37% | **−1.17%** | better |
| short period ζ | 0.342526 | **0.343058** | −11.49% | **−11.35%** | better |

**Three of four improve and nothing was aimed at Table IX-5**, which is the only reason the
fourth is quotable: the phugoid ω_n moves **−0.398%** against g's **−0.3817%**, which is
Lanchester's `ω_n = √2·g/u₀` at 1:1 to three figures. It is the *right* answer moving
*away* from the reference, and §5 already says why — the phugoid gap is `M_u`, a term this
model has no form for, so improving gravity cannot help it and does not.

**Three pins were re-captured, and each records why at the change.** None is a widened
tolerance; all three are numbers taken at a physics combination the tree no longer has.

| pin | before | after | why it moved |
|---|---|---|---|
| `test_vortex_viz.FIG8_VORTEX` | (2.239956221700959, −1.2352174348304876) | **(2.241674009986879, −1.2396439681557148)** | +0.0767% on Δθ, **+0.358% on Δn** — the Δn limb is the gravity change essentially alone, since `n_z` is divided by **standard** gravity by construction, so a lighter local g reports a larger excursion |
| `test_cat_spectra` short period | 0.16404 Hz | **0.16433 Hz** | +0.18%, g(z) and the ISA conversion both reaching the Mehta altitude |
| `test_wind` Lamb–Oseen gradient bound | 0.260 `V₀/r₀` | **0.217 `V₀/r₀`** | **not gravity — the CORE RADIUS.** 0.260 was taken at the session-22 hybrid's 500 ft; session 26 restored Parks' own 600 ft. The bound is normalised by `V₀/r₀` but the airframe is not, so a wider core puts the span across less of it and the curvature the secant misses falls |

> **The short-period pin is the one to read carefully, because it sits next to the sealed
> register.** `predictions.py`'s settled `the_dryden_response_peaks_at_the_short_period`
> carries a band of **[0.131, 0.197] Hz** and an outcome of 0.1400/0.1700 Hz. **0.16433 is
> comfortably inside it and no verdict changes.** The line that failed is the regression pin
> whose own docstring says it exists "so it cannot drift underneath a settled prediction" —
> it did exactly that job. **The sealed entry was not touched, and `rel=1e-3` was not
> widened.**

**One process note, recorded because it cost real time and is the project's own named
hazard.** The first mode measurement taken after the rebase read *unchanged*, and it was
wrong: `python scripts/checkpoint.py` from the worktree puts `scripts/` on `sys.path` and not
the cwd, so it imported `atisim` from the **main checkout** — §10's table says so in the row
that has been there since session 17. The numbers above were re-taken with the cwd on the
path and the tree printed. **The hazard is not theoretical and it caught this session mid-merge.**

### C3's Mach axis is not unbounded — the curves are in CR-2144 and have been all along — session 28

**`ASSUMPTIONS.md` C3 has called the Mach axis UNBOUNDED since session 12, §7 declined the
bound repeatedly because it "needed chart reads off a poor scan", and session 27 priced it
with Prandtl–Glauert — a DECLARED theoretical form — while a SOURCED Mach dependence sat in
a document this project has held since session 1.**

Found by reading `refs/NASA-CR-2144.pdf`'s own text layer, not by digitising anything. Three
consecutive figure pages, printed pp. 220–222 (PDF indices 225–227), all captioned
**`B-747 / 636600 lb / .25 c̄ / Flexible`** and all drawn against a **Mach axis running 0 to
1.0**, with three altitude curves apiece — **SL, 20,000 ft and 40,000 ft**:

| printed page | curves the text layer names |
|---|---|
| **220** | **`CL_α(M)`** and `CD_α(M)`, rad⁻¹ |
| **221** | **`Cm_α(M)`**, and `Cm_α̇(M)` with `Cm_q(M)` |
| **222** | **`CL_M`, `CD_M`, `Cm_M`** — the Mach derivatives themselves |

**What this settles.**

1. **The axis spans the excursion.** The LES condition is **M 0.406**; the figures start at
   M 0. The 40,000 ft curve is the one `_boeing_747` was linearised on, so the *same* curve
   gives both the value in use and the value at the LES condition. Session 27's
   Prandtl–Glauert ratio of **1.521** becomes a *checkable prediction* against a source
   rather than the only estimate available.
2. **The altitude axis comes free.** Three curves, so the altitude dependence C3 also carries
   is on the same sheets — and §4's one altitude point (ω_n **+23.46%** frozen against
   **−0.60%** local, at M 0.8 / 6,096 m) gets a second, independent check.
3. **`CL_M`, `CD_M` and `Cm_M` are the missing speed-derivative content by name.** §5 says
   what is excluded from the phugoid is "the **Mach content** of those derivatives (CXu, CZu,
   from CL_M and CD_M)". Printed p. 222 is that content, plotted.
4. **The reading pipeline exists and is validated twice over.** Session 21 digitised
   `CL_MAX(M)` off CR-114494 at a stated ±0.02; session 26 re-read the same curve with an
   independent tracer and reproduced it to **RMS 0.0052**, four times inside that
   uncertainty. Session 21 also read `CL_α(M)` off **Figure IX-5 itself** and cross-checked
   it at M 0.80: **4.892 from the figure against Table IX-4's 4.9441, agreeing to 1.05%.**

**And that last reading is gone, which is why this row exists.** Session 21's entry says in
its own words that "the working patch is kept out of the tree". The single M 0.80 cross-check
survives in §4's prose; **the curve does not.** This is precisely the failure `docs/DEVELOPMENT.md`
rule 1 was written for, occurring six sessions before that rule existed, and it cost the
project the bound it then spent sessions 22–27 saying it could not have.

**Superseded in part, session 30: all eight curves on pp. 220–222 have now been read by hand
and checked against Table IX-4 at every circled condition.** The scan was good enough: the
smooth curves agree with the tables to 0.7–1.5%, and Cm_M to RMS 0.006. See "CR-2144's speed
derivatives, digitised and flown". The paragraph below is left as written.

**What is NOT claimed here.** No curve has been read this session. The evidence is the PDF's
own text layer naming the axes and the altitude legends; the scan quality, the gridline
separation and the achievable uncertainty are unmeasured, and session 21's "poor scan"
warning stands until someone re-reads it. **What has changed is the status, not the number:
C3's Mach axis moves from UNBOUNDED-and-source-gated to UNBOUNDED-and-one-digitisation-away,
on a sheet whose neighbour this project has already digitised successfully.**

**Why it is the highest-value item on the list.** §4 records the frozen slope as the
**dominant identified contributor** to the LES discrepancy — 47–68% of it — and §1's envelope
is a Mach band asserted from the fit range rather than from any measurement of what happens
outside it. One digitisation of printed p. 220 turns both into sourced numbers. It needs no
acquisition, no correspondence, and no new method.

### The LES ensemble POD, run — and it does not separate the drift — session 28

`scripts/les_ensemble_svd.py`, on the 16 × 5000 `n_z` arrays `les_flight.py` wrote for all
four domains. §4 lists **condition drift** as candidate 1 for the ~20% that survives the
matched-aeroplane LES comparison, and an SVD of the (flights × time) matrix is the obvious
instrument: the drift is common to every flight, so it should fall out as one dominant mode.

**It does not, on the two domains that matter, and the reason is that there is no dominant
mode to remove.** The `(iv)` row below reproduces `les-comparison.csv` **exactly** —
0.09049 and 0.15728 — which is what licenses the other three rows.

| domain | mode 1 share | **mode 1 below 0.05 Hz** | loading uniformity | (ii) ens-mean | (iii) rank-1 | **(iv) high-pass** |
|---|---|---|---|---|---|---|
| D01, 500 m | 66.0% | 98.6% | 0.011 | 0.02359 | 0.02231 | **0.00747** |
| D02, 250 m | 55.2% | 97.7% | 0.025 | 0.07861 | 0.05961 | **0.01820** |
| **D03, 70 m** | **25.1%** | **53.3%** | 0.900 | 0.10086 | 0.09623 | **0.09049** |
| **D04, 35 m** | **18.3%** | **71.9%** | 0.300 | 0.17585 | 0.16585 | **0.15728** |

**Three readings, and all three say no.**

1. **On D03/D04 the ensemble is not low-rank.** Mode 1 carries 25.1% and 18.3%, and the
   spectrum is flat behind it — 25/16/13/10/8 on D03. That is broadband turbulence. There is
   no drift mode standing above the rest to subtract.
2. **Mode 1 is not the drift.** Only **53.3%** and **71.9%** of it sits below 0.05 Hz, so it
   is a *mixture*; removing it discards genuine in-band gust response. Its correlation with
   time is −0.241 and +0.180, i.e. not a monotone trend either.
3. **Rank-1 removal is strictly worse than the shipped high-pass**, leaving **1.064×** and
   **1.055×** its rms — it removes *less* low-frequency content while additionally destroying
   in-band signal. It beats subtracting the ensemble mean (0.954×, 0.943×), which is the only
   comparison it wins, and that one is a single line of numpy.

**D01 and D02 flip, and it changes nothing.** There mode 1 does carry 55–66% and is 98%
sub-0.05 Hz — those ensembles genuinely are drift-dominated and low-rank. But §4 already
records that 500 m and 250 m LES **cannot drive an aircraft-load calculation**, so the one
regime where the decomposition works is the one regime whose numbers are not used.

**The question was then asked with the right instrument instead, and that is inconclusive
too — which is the more useful result.** A 0.05 Hz high-pass removes a secular drift outright,
so the drift cannot be an *additive* term in the published rms; it is already gone from both
sides. What a high-pass cannot remove is the drift's real effect — load goes as `q̄`, so an
aeroplane that has lost 30% of its airspeed responds to the same gust with a smaller
excursion, which is an amplitude **modulation** and shows as early-half against late-half rms:

| domain | early | late | late/early | flights weaker late |
|---|---|---|---|---|
| D01 | 0.00535 g | 0.00898 g | **1.680** | 1/16 |
| D02 | 0.01891 g | 0.01624 g | 0.859 | 10/16 |
| D03 | 0.07970 g | 0.09755 g | **1.224** | 4/16 |
| D04 | 0.17847 g | 0.11919 g | **0.668** | 11/16 |

**The ratios go both ways.** A monotone `q̄` decay cannot make the late half 68% *stronger*
on one domain and 33% weaker on another. What dominates is that the aeroplane flies through
**different turbulence** in the second half: the field is not statistically homogeneous along
the track, and that swamps the `q̄` term.

> **THE CONCLUSION, AND IT APPLIES BEYOND THE SVD.** Drift and along-track field
> inhomogeneity are confounded in **every statistic computable from these arrays**, because
> they are confounded in the **run**. No post-processing separates them — not a POD, not a
> harder high-pass, not a split. §4's own wording is the only route left and it is an
> experiment, not a filter: *"hold the condition ... and re-read"* — fly it again with `q̄`
> maintained and difference the two runs. **The "or high-pass harder" half of that sentence
> is now measured and should be struck.**

**Caveat carried, per the input audit.** These are the `boeing747` runs §4 refuses for load
level — 5.4% away in short period, 2.63 band widths outside its envelope. Nothing here is
evidence about load. It does not affect this question: the drift is a property of flying
fixed-control with real drag, which that entry does, and the confounders §4 names are
constant multipliers that can neither create nor remove a secular trend.

### What an SVD can and cannot see in this model — session 28

`scripts/svd_probe.py`. Asked whether the singular value decomposition is worth applying
here, three hypotheses were formed **before** running anything. **One is falsified, one
holds cleanly, and the third holds only for the mode the CAT work depends on.** Nothing in
the model changed: the script imports `trim` and `validation` and measures matrices they
already build.

**(a) FALSIFIED — the §5 absurd-trim defect is NOT a rank problem, and no decomposition of
the Jacobian will find it.** The hypothesis was that `cond(J)` would separate a healthy trim
from the converged-but-absurd roots §5 records, where the residual cannot. It does not, and
it fails in the *opposite* direction to the one predicted. `boeing747_approach`, 85 m/s,
sea level:

| case | α | residual norm | **cond(J)** |
|---|---|---|---|
| healthy, as shipped | 5.590° | 1.11e-16 | **9.51e+01** |
| degenerate `CLa` = 0.1 | −272.695° | 2.33e-15 | **1.97e+01** |
| degenerate `CLa` = 1e-4 | −632.136° | 5.72e-15 | **2.26e+01** |

**The two absurd roots are BETTER conditioned than the real one**, and the whole spread is a
factor of 4.8 against the residual's 51.6. The reason is clear once measured and should have
been clear before: at the absurd root the Jacobian is a perfectly good matrix — Newton found
a genuine, well-conditioned root of a function that **has several roots**. Multiple roots of
a nonlinear system is not rank deficiency. **`trim.is_physical` stays exactly as it is, and
this row exists so nobody proposes replacing it with a conditioning gate.**

**(b) HOLDS — `ASSUMPTIONS.md` F7 is a rank problem, and σ_min sees it before the solve
does.** Zero `CLde` and `Cmde`, so the elevator has no authority, and `trim.trim` returns
`[nan nan nan]` with a `nan` residual — F7's "in silence". The Jacobian at the **initial
guess**, before the first Newton step, has singular values `5.149e+01, 3.015e+00,
0.000e+00`. **σ_min is exactly zero at a point where every other instrument in the project
returns NaN and therefore says nothing.** This is the one place an SVD earns its keep in the
solver, and it matters exactly when F7 predicts: the first solve that carries a control the
airframe cannot exercise — F7 names a steady-turn solve with rudder as the candidate.

**(c) HOLDS FOR THE SHORT PERIOD — identifiability, derived instead of discovered.** Session
27 found `I_yy` unobservable on the B787 entry by sweeping it 1.0e7 → 4.0e7 and noticing
`M_α` came back identical every time. The same fact is available from a decomposition
without knowing to look for it. Sensitivity of the longitudinal modes to
(`Cma`, `Cmq`, `Iyy`), central differences in **log** parameter space so the singular
vectors read as power laws, `boeing747` at its own cruise:

| observables | σ | unseen direction (log space) | vs equal scaling |
|---|---|---|---|
| all four mode scalars | 0.797 / 0.528 / **0.192** | `Cma` +0.5397, `Cmq` +0.6080, `Iyy` +0.5822 | ratio only 4.15 — **soft, not null** |
| **short period only** | 0.769 / 0.513 | `Cma` +0.5776, `Cmq` +0.5748, `Iyy` +0.5796 | **0.1936° from (1,1,1)/√3** |

**Read the second row correctly.** Two observables against three parameters is a 2×3 matrix,
so a null direction exists **by shape** and is not itself evidence. What is evidence is where
it points: within a fifth of a degree of the equal-scaling combination that
`M_α = C_mα·q̄Sc̄/I_yy` and `M_q = C_mq·q̄Sc̄²/(2V·I_yy)` predict analytically. **The SVD was
not told that and recovers it.**

**And the first row is a finding in its own right.** Scaling all three by 1.5 at once moves
the short period by **+6.6e-4 and +7.5e-4** — nothing — and the phugoid by **−10.9%**. The
invariance is exact for the short period and broken for the phugoid, because scaling `Cma`
and `Cmq` without `Cm0` and `Cmde` destroys the moment balance at the frozen trim point, so
`C_m` is no longer zero there and an `M_u` appears. **That is §4's own localisation arriving
from a different direction** — the phugoid is the mode speed derivatives dominate, which is
why it is the mode that notices.

**Where this generalises, and it is the reason to keep the script.**
`gen_jsbsim_reference.recover_pitch_axis` already calls `np.linalg.lstsq` — which *is* an SVD
— and already records `np.linalg.cond(design)` = **1.9e8** (737) and **2.7e9** (747). §4
resolves both by hand: *"their SUM is exact and their SPLIT is conditioned"*, and
`aircraft.py`'s 747 comment says *"their SUM is −25.000000 and is the well-determined
quantity"*. **Those are statements about a singular vector, arrived at by inspection.** The
project has now hit the same class of problem three times — the 737 `Cmq`/`Cmadot` split, the
747 one, and the B787 `I_yy` — and solved it three different ad-hoc ways. The decomposition
answers all three in one call and does not need to be told which combination to try.

**What an SVD does NOT do here, stated so it is not proposed again.** It does not finish the
LES limb: those runs are **not compute-bound** — all four domains × two aircraft are already
on disk, `les_flight.py` loads only a subvolume around the flight path, and what blocked the
comparison was the *aeroplane*, which session 27 fixed by building
`boeing787_yoshimura`. A POD of the field would also have nothing to decompose in time —
the domain files carry **one time step**. And it is not an instrument for the scale content
of a gust along a track, which is what `response.py` and `wind.dryden_spectrum` already do
properly with a Fourier estimate.

### The strip-load path, judged — session 28

**Asked directly: is it helpful, and is it even being used?** Measured rather than recalled.

**It is not used on any path that produces a published number.** `integrate.step` defaults to
`load_model=None`; the shipped gust treatment is `field_model` — point sample plus analytic
tangent gradient. The strip path is opt-in at three call sites only (`scripts/vortex.py
--strip`, `scripts/cat_bounds.py`, `scripts/lateral.py`), and a test deliberately asserts that
the strip and tangent estimators **disagree** at the vortex core edge, which is what proves the
default has not been quietly switched — §4's frozen baselines sit downstream of the tangent.

**It has moved a reported number exactly once in fourteen sessions**, and re-measured this
session it still does:

| Mehta's five-vortex field, 747 at 37,000 ft | \|φ\| | \|β\| | `n_z` max | up-increment |
|---|---|---|---|---|
| point (`vortex_wind`) — the published path | **0.000°** | **0.000°** | 1.6019 | 0.6019 g |
| line (`line_vortex_wind`) | 12.508° | 3.106° | 1.6375 | 0.6375 g (**+5.92%**) |
| line + **strip loads** | **15.376°** (**+22.9%**) | 3.533° | 1.6344 | 0.6344 g (**+5.40%**) |
| **session 30, speed derivatives declared:** point | 0.000° | 0.000° | 1.4038 | 0.4038 g |
| **session 30:** line | 18.418° | 4.551° | 1.4678 | 0.4678 g (**+15.86%**) |
| **session 30:** line + strip | **21.579°** (**+17.2%**) | 4.941° | 1.4696 | 0.4696 g (**+16.31%**) |

~~**So the strip increment's own contribution, isolated, is +22.9% of peak bank and −0.5% of the
longitudinal load increment.**~~ **SUPERSEDED, session 32 — the +22.9% (and session 30's +17.2%)
is the gust's rolling moment counted twice, not the strip path's own contribution.** `strip=True`
adds the strip rolling moment on top of the point path's equivalent roll rate rather than
replacing it; §6(h) has the mechanism. With roll owned by one path, the strip path's effect on
peak bank is **−2.5%**, and that is its known quadrature deficit (`ASSUMPTIONS.md` F5), not new
physics. **The absolute rows above are also stale**: on today's 747 the line run reads 18.067°
and line + strip 21.209° — `scripts/lateral.py` itself, same tree — against the 18.418° and
21.579° recorded, which predate session 30's thrust line. Everything the project claims is
longitudinal, and none of it moves.

**Three things have to be said together about that +22.9%, and only the first is favourable.** *A fourth, found session 32 and stated in the superseding note above: it is a double count.*

1. **The kernel is well verified.** `wind.strip_roll_moment` agrees with the tabulated `Clp`
   for a rigid roll rate (rel 1e-3), with Stengel eq. 3.4-40's closed form for a rectangular
   wing (1e-6), with the elliptic identity `∫y²c dy = c₀b³π/64` (1e-5), and with the
   equivalent-rate treatment for a linear gradient (1e-6). It is **roll only**, deliberately:
   there is no validated pitch or yaw integral and those channels are held at exact zero.
2. **The quadrature that computes it is 17% off its own calibration.** `ASSUMPTIONS.md` **F5**:
   at the shipped `N_SPAN = 9` the strip integral returns **82.6%** of the `Clp` it is
   calibrated against, converging at order **1.50**, needing ~56 stations for 1%. The one
   number the strip path moves is computed by that quadrature, on a **DECLARED** loading shape
   carrying a 2.6% spread across defensible shapes.
3. **The channel it moves is not validated against anything.** §5 and `ASSUMPTIONS.md` E10:
   **no source held by this project records a lateral CAT response.** Wingrove & Bach 1994 says
   so in as many words. Every bank angle in this table is a capability demonstration.

**The verdict, and it is not "wasted work".** The strip path's value to date is a **negative
result, and a load-bearing one**: it is the measurement that says the point-gust approximation
is adequate for everything §1 claims. `ASSUMPTIONS.md` E2 rests on it — point-vs-strip on the
headline field is **0.000000 m**, so the 68% shortfall cannot be blamed on gust resolution
across the span. Without the strip path that would be an argument; with it, it is a number.
**What it must not be used for is a lateral load claim**, and quoting the 15.376° without F5's
17% and E10's "unvalidated" beside it would be exactly that.

### The LES comparison is NOT like-for-like — input audit, session 27

**The runs exist and their numbers are not admitted here.** `scripts/les_flight.py` and
`les_compare.py` were run in the weekly worktree over all four LES domains and two aircraft,
and report AtiSim/Yoshimura rms ratios of **1.068, 1.827, 1.427, 1.420** (D01…D04). **No row
of §4 quotes them**, because an audit of the two codes' inputs finds differences large enough
to account for a ratio of that size without any code disagreement at all.

**Yoshimura's own aircraft, recovered from their own source** — `flightsim-data/src/fs.f90`
lines 252–265 in the figshare extract, which carries their **complete dimensional derivative
set**. Built into their own `A_lon` exactly as their code builds it:

| | from their `fs.f90` |
|---|---|
| short period | **ω_n 0.9023 rad/s = 0.1436 Hz**, ζ = 0.621 |
| phugoid | ω_n 0.0880 rad/s = 0.0140 Hz, ζ = 0.012 |
| airspeed `U_0` | 133.0 m/s |
| `θ_0` | **0** — they do not trim |
| `Z_δe`, `M_δe` | **0** — no control surfaces at all |
| `I_x`, `I_z`, `I_xz` | 1.5752e6, 3.8109e6, 7.5789e4 (× 9.81) |

**0.1436 Hz against the paper's stated 0.14 Hz** — so their prose is confirmed by their code,
and this project now holds the frequency as SOURCED rather than quoted.

**The mismatches, in the order they matter.**

| # | Input | Yoshimura | AtiSim as flown | Consequence |
|---|---|---|---|---|
| 1 | **Aircraft** | short period **0.1436 Hz** | `boeing747` 0.1513 Hz (**+5.4%**); `boeing737_approach` 0.2627 Hz (**+82.9%**) | Neither is their aeroplane. The resonance the whole comparison turns on sits in a different place |
| 2 | **Envelope** | n/a — frozen linear model, no envelope | `boeing747` at **M 0.31–0.50, 2,661–3,209 m** against a validated M 0.70–0.90 / 10,668–13,716 m. `checks.recovery_band` **fails at 2.63 band widths** | The 747 rows are **extrapolation**, not measurement. Its derivatives were linearised at M 0.80 / 40 kft and are frozen (ASSUMPTIONS C3, unbounded) |
| 3 | **Trim** | `θ_0 = 0`, untrimmed | trims to **α 4.92°** | Different starting attitude, and AtiSim's α excursion reaches **11.81°** — past the 10° linear band |
| 4 | **Model class** | 9-state **linear**, frozen derivatives | nonlinear 6-DOF | The irreducible difference, and the *point* of a cross-code check — but only once 1–3 are removed |
| 5 | Time step | 1/128 s | 0.02 s | Minor; bounded elsewhere at ~0.14% |
| 6 | `g` | 9.81 | 9.80665 | 0.034%, negligible |

**And one thing wrong inside their own model, which matters for anyone reading it.** Their
lateral block is commented **`!Lateral/Directional derivatives >>> B747`** while the
longitudinal set gives a 0.1436 Hz short period and inertias about 0.63× the 747's — i.e.
**their aeroplane appears to be a B787 longitudinally and a B747 laterally.** This project's
own rule against crossing sources says what to do with that: quote their **longitudinal**
comparison, which is what the load is, and do not use their lateral response as a reference.

**Why the 747/737 choice is a genuine dilemma rather than an oversight.** The LES tops out at
**5,215 m** and cannot reach 747 cruise at 11,278 m, so the comparison must happen at ~3,000 m.
At that condition the only registry entry *inside* its own validated envelope is
`boeing737_approach` — whose short period is **83% away from theirs**. The entry closest in
frequency, `boeing747`, is the one 2.63 band widths outside its envelope. **You cannot
currently match both the aeroplane and the envelope, and the runs on disk chose to match
neither cleanly.**

**The gate is now open: the four B787 numbers are obtained, and three of them are checked
against Yoshimura's own derivatives rather than merely looked up.**

| | value | provenance |
|---|---|---|
| `m` | **215,910 kg** (476,000 lb, MTOW) | SOURCED — Piano 787-8 analysis; **and confirmed by the check below** |
| `S` | **325.3 m²** (3,501.39 ft², **trapezoidal reference**) | SOURCED, same. *The definition matters*: Wimpress 3,870 ft² and Piano gross 4,028 ft² also exist and differ by 15% |
| `c̄` | **6.437 m** (21.12 ft, trapezoidal MAC) | SOURCED, same |
| `I_yy` | **≈ 2.34e7 kg·m² ± 12.7%** | **DECLARED — and it does not matter, see below** |

**The mass and area are not assumed, they are pinned by their own `Z_a`.** Inverting
`Z_α = −(C_Lα + C_D)·q̄·S/m` at their condition:

| mass | implied `C_Lα` (trapezoidal S) | verdict |
|---|---|---|
| **MTOW 476,000 lb** | **4.847 /rad** | physical, and within 2% of this project's 747 (4.944) |
| mid (OEW+MTOW)/2 | 3.634 /rad | too low for a swept transport |
| OEW 239,200 lb | 2.421 /rad | unphysical |

**So their aeroplane is a 787-8 at or near maximum takeoff weight**, and that is a conclusion
drawn from their derivative rather than an assumption fed into it. Every other coefficient
recovers physical too: `C_mα` −0.76…−0.86, `C_mq` −29…−33, `C_lβ` −0.2515, `C_nβ` +0.1596 —
all the same sign and order as the 747's.

**`I_yy` is unobservable and therefore free.** `C_mα` and `I_yy` enter the dynamics only as
their product `M_α`, which is SOURCED. Verified numerically by round-trip: `I_yy` from 1.0e7
to 4.0e7 moves `C_mα` from −0.346 to −1.383 and recovers **`M_α = −0.582000` and
`M_q = −0.537000` exactly, every time** (`_evidence/b787_consistency_check.log`). So the one
number that could not be sourced is the one number that cannot affect the answer. **Declare it,
state the range, and move on** — do not go looking for a published 787 pitch inertia.

**What remains is implementation, not acquisition:** build the registry entry, give it a
`valid_mach` / `valid_altitude` band around *its own* condition, and re-run the LES limb with
both codes flying the same aeroplane. **Note the one term AtiSim still will not carry:
Yoshimura's `M_α̇ = −0.137`** — the same `M_ẇ` class the 747's phugoid gap is attributed to
(§4, "Where the longitudinal gap comes from"). Quote that as a known, attributed difference
rather than discovering it again afterwards.

### The LES comparison with the aeroplane MATCHED — session 27

`aircraft.boeing787_yoshimura()` is built from their own `fs.f90` derivative set and
**reproduces their own `A_lon` eigenvalues to 0.01%** on the short period and 0.20% on the
phugoid, trimming at α = +0.0000° and elevator −0.0000° — i.e. at their `θ_0 = 0`
linearisation point, with residual 7.4e-17. That agreement is a result in its own right:
**this project's nonlinear model linearised by `jax.jacfwd`, against their hand-built 4×4
matrix, by two entirely different routes.** `test_aircraft.py` asserts it.

Flown through D03, 16 flights, against Yoshimura's own ensemble, both sides high-passed
identically:

| run | rms (h-p) | ratio | excess closed | short period | response peak |
|---|---|---|---|---|---|
| baseline `boeing747` (wrong aeroplane **and** frozen slope) | 0.0905 g | **1.427** | — | 0.1647 Hz | 0.0600 Hz |
| full PG-corrected 747 | 0.0777 g | 1.225 | 47% | 0.1272 Hz | 0.0600 Hz |
| lift-only corrected 747 | 0.0720 g | 1.135 | 68% | 0.1568 Hz | 0.0600 Hz |
| **matched `boeing787_yoshimura`** | **0.0762 g** | **1.202** | **53%** | **0.1441 Hz** | **0.1300 Hz** |
| *Yoshimura's own* | *0.06342 g* | *1.000* | — | *0.1436 Hz* | *0.0800 Hz* |

**Matching the aeroplane closes 53% of the discrepancy, and it does it the honest way:** the
matched entry carries derivatives tabulated AT M 0.406, so unlike every 747 row above it has
**no frozen-slope error at all** — nothing was corrected, there was nothing to correct.

**The qualitative change is larger than the ratio suggests.** With the 747 the response peaked
at 0.0600 Hz — the bottom of the search band, nowhere near the airframe. With the matched
aeroplane it peaks at **0.1300 Hz**, 9.8% below its own 0.1441 Hz short period. **The airframe
has started selecting its own frequency out of the field**, which is the mechanism §4's spectral
work is about, and the 747 at M 0.41 could not show it.

**A 20% disagreement survives, and it is now attributable to neither the aeroplane nor Mach.**
Both are eliminated by construction. What is left, in order of suspicion:

1. **Condition drift.** AtiSim's aeroplane loses **39.8 m/s over the record — 30% of its
   airspeed** — and 552 m of altitude, because it flies fixed-control with real drag. Yoshimura's
   holds altitude to 26–43 m and cannot decelerate, having neither drag nor thrust. Load goes as
   q̄, so a decelerating aeroplane and a non-decelerating one are not measuring the same thing.
   **This is the next experiment**: hold the condition, or high-pass harder, and re-read.
2. **Model class.** Nonlinear 6-DOF against a frozen 9-state linear model — the irreducible
   difference, and the one a cross-code comparison exists to expose.
3. Their response peaks at 0.0800 Hz where the matched entry peaks at 0.1300, so the two codes
   still disagree about *which* frequency the load follows, not only how large it is.

**`checks.recovery_band` reports `passed=False` and that is an artefact, not a violation:** the
entry declares no band (§4's rule — it is a transcribed set, not a fit), so the check has
nothing to compare against. Do not read it as an envelope excursion. The |α| range is **5.74°**,
comfortably inside the linear band, where the 747 at this condition ran to 11.81°.

### What the LES comparison IS good for: resolution, not level — session 27

The audit above refuses the LES *load* comparison. **One thing survives it intact, and it is
worth more than the number that was refused.**

Both confounders — the wrong aeroplane and the frozen `C_Lα` — are **constant multipliers on
AtiSim's side across all four domains**: same aircraft, same condition, same Mach, only the
field changes. So they **cancel exactly** in each code's own resolution ratio. Normalising each
code to its own finest domain:

| domain | dx | AtiSim / D04 | Yoshimura / D04 | disagreement |
|---|---|---|---|---|
| D01 | 500 m | 0.0475 | 0.0632 | −24.8% |
| D02 | 250 m | 0.1157 | 0.0899 | +28.7% |
| **D03** | **70 m** | **0.5753** | **0.5724** | **+0.5%** |
| D04 | 35 m | 1.000 | 1.000 | — |

**On the two domains that resolve the turbulence the two codes agree on the resolution
scaling to 0.5% — AtiSim grows ×1.7381 from 70 m to 35 m, Yoshimura ×1.7469 — while their
absolute levels differ by 42%.** That is the project's own "comparative, not absolute" claim,
demonstrated for the first time **against an independent code on a field that was not
identified from the accelerations it is asked to predict.** Everything §1 says this model is
for, and nothing it says it is not.

**Three usable statements, and they are the ones to quote.**

1. **The load is NOT grid-converged at 35 m.** It is still growing **×1.74 per halving of the
   grid**, and *both codes agree it is*. Any CAT load computed from an LES coarser than this is
   an underestimate of unknown size — a result about **the meteorology's requirements**, not
   about either aeroplane.
2. **Below ~70 m the comparison degrades and the codes stop agreeing** (−24.8%, +28.7% at 250
   and 500 m). So 500 m and 250 m LES **cannot** drive an aircraft-load calculation, and there
   are now two independent codes saying where that floor is.
3. **The field reader is validated infrastructure**: +0.978 / −0.968 / −0.935 against
   Yoshimura's own sampled wind on u/v/w, registration residual measured at 1–2 cells (under
   150 m) and deliberately not fed back. It is reusable for any published LES.

**And it reaches Yoshimura's own conclusion by a route their arithmetic cannot.** Their paper
argues for fine grid spacing from a resonant wavelength of 200 m — computed from the 1.29 Hz
that §5 shows is a misread `s⁻¹`, where the correct figure is ~1,257 m and would argue the
opposite way. **The measurement above supports fine grids anyway, empirically, and without
touching that calculation.** Being right for a reason the source did not have is the most this
comparison can currently claim, and it is a real claim.

**What it may NOT be used for**, until a `boeing787_yoshimura` entry exists: any statement
about absolute load, any statement that the two codes agree or disagree *in level*, anything
about the 747 at that condition, and anything lateral.

### The frozen lift-curve slope: 47-68% of the LES discrepancy, measured — session 27

**`ASSUMPTIONS.md` C3 says derivatives are frozen across the envelope and calls the Mach axis
UNBOUNDED. The LES runs are the largest Mach excursion in the project's history, and nothing
in them accounts for it.**

| | cruise, where the 747 was linearised | the LES condition |
|---|---|---|
| altitude | 12,192 m | 3,000 m |
| airspeed | 235.9 m/s | 133.5 m/s |
| **Mach** | **0.7995** | **0.4063** |
| dynamic pressure | 8,392 Pa | 8,101 Pa — **ratio 0.965** |

**The `q̄` axis is essentially matched.** That matters, because `q̄` is the axis session 23
*bounded* (23.5% of `ω_n` per 2.48× of `q̄`). Here it is 1.04×, so that bound is not the
issue. **What moves is Mach, by ΔM = −0.393 — and the largest excursion C3 previously
recorded is the lee wave's ΔM = −0.031. This is thirteen times larger than anything the
assumption had been stress-tested against.**

`aero.py` carries Mach **only** into `wave_drag`. There is no Prandtl–Glauert factor and no
compressibility correction on `C_Lα`, `C_mα` or anything else — they are literally constant.
So the 747 flies the LES field with its **M 0.80** lift-curve slope:

- Prandtl–Glauert `1/√(1−M²)`: **1.6649** at M 0.80, **1.0944** at M 0.406
- ⇒ a slope tabulated at M 0.80 is **1.521× too large** at M 0.406
- `C_Lα` carried: **4.944 /rad**; PG-consistent value there: **3.250 /rad**

**Gust load goes linearly as `C_Lα`, so this alone predicts AtiSim/Yoshimura ≈ 1.52×.**
Measured on the runs on disk: **D03 1.427, D04 1.420** — the two domains that actually resolve
the turbulence. **The discrepancy the LES comparison reports is the size the frozen derivative
predicts, and Yoshimura's model does not share the error because theirs is linearised at
their own condition.**

**This is a mechanism, not yet a measurement**, and it is falsifiable in one run: rescale the
747's `C_Lα` by the PG ratio, re-fly D03/D04, and the ratio should collapse toward 1. **Do that
before attributing anything in the LES comparison to either code.** It is also the cheapest
Mach-axis bound the project has ever had within reach — C3 has wanted one since session 12 and
declined it because it needed chart reads off a poor scan; this needs no chart at all.

> ### THE TEST WAS RUN, AND IT CUT THE CLAIM ABOVE IN HALF — session 27
>
> `scripts/les_mach_test.py`, D03, 16 flights, identical field, path and condition.
> **The heading of this section is too strong and is corrected here rather than rewritten:
> the frozen slope is about HALF of the discrepancy, not all of it.**
>
> | run | `C_Lα` | rms (high-passed) | vs Yoshimura 0.06342 | short period | excess closed |
> |---|---|---|---|---|---|
> | **baseline `boeing747`** | 4.9441 | **0.0905 g** | **1.427** | 0.1647 Hz | — |
> | **full PG correction** | 3.2465 | **0.0777 g** | **1.225** | **0.1272 Hz** | **47%** |
> | **lift-only** (`C_Lα`,`C_Lq`,`C_Lδe`) | 3.2465 | **0.0720 g** | **1.135** | 0.1568 Hz | **68%** |
> | *pure `C_Lα` linearity would give* | *3.2465* | *0.0594 g* | *0.937* | *—* | *100%* |
>
> **The baseline reproduces the weekly worktree's 0.09049 exactly**, so this tree and that one
> agree and the comparison is sound.
>
> **The correction is large and real but not sufficient, and the two variants bracket it.**
> Holding the resonance fixed (lift-only) closes **68%** of the excess over 1.0; the physically
> consistent full PG correction closes only **47%**. Neither reaches the ×0.6566 that pure
> linearity in `C_Lα` predicts — the lift-only run falls ×0.7956 and the full one ×0.8586.
>
> **The difference between the two variants is itself the finding, and it was predicted before
> the run.** The full PG set scales `C_mα` too, dropping the short period **0.1647 → 0.1272 Hz**
> — *toward* the energetic low-frequency end of the LES spectrum — which pushes load back up and
> cancels a third of the lift reduction. **Correcting compressibility more completely makes the
> agreement worse, because the airframe's resonance moves into more energetic turbulence.** That
> is a statement about how gust load is set in this regime, and no peak comparison could have
> made it.
>
> **What may now be said, and it is narrower than the heading promised:** the frozen lift-curve
> slope is the **dominant identified contributor** to the LES load discrepancy — **47–68% of it,
> depending on whether the resonance is held fixed** — and it is **this project's error, not
> Yoshimura's.** What may **not** be said is that it explains the discrepancy. A residual ratio
> of **1.14–1.23** survives the best correction available, and the leading remaining candidate is
> the aircraft mismatch itself: their short period is **0.1436 Hz** against 0.1568–0.1647 here.
>
> **And the residual is now small enough that the `boeing787_yoshimura` entry could close it.**
> Before this run the discrepancy was 43% and unattributed; it is now 14% with a named candidate.
>
> **This is still the first quantified point on the Mach axis** — the thing C3 has wanted since
> session 12 — and it cost no chart read: **a ΔM of −0.393 on a frozen derivative set moves a
> gust-load rms by ×0.859, of which the lift slope alone predicts ×0.657.**

**Until then:** the LES runs stand as a capability demonstration and a reader for their field —
the field reader itself is independently validated, correlating **+0.978 / −0.968 / −0.935**
against Yoshimura's own sampled wind on all three components, with the registration residual
measured (1–2 cells, under 150 m) and **deliberately not fed back**. That part is sound and
reusable. The **load comparison** is not yet a comparison.

### The recorded trace, digitised at last — session 27

**The project quoted `wind.TM102186_HANNIBAL_NZ` = −1.0 / +1.7 g for four sessions without
ever looking at the curve between those two numbers.** `scripts/digitise_tm102186_fig6.py`
reads TM-102186 Figure 6's G LOAD panel out of `Reference_papers/19890016606.pdf` at 600 dpi,
column by column, as the top and bottom of the ink. Nothing is fitted and nothing smoothed.
1,420 columns over t = 6.9 … 301.2 s.

| Quantity | Digitised | The paper's own prose | Agreement |
|---|---|---|---|
| downward extreme | **−0.972 g** | −1.0 g | 0.028 g |
| **second** upward peak | **+1.702 g** | +1.7 g | **0.002 g** |
| 99.5th pctile, upper envelope | +1.701 g | +1.7 g | 0.001 g |
| **absolute** upward peak | **+1.855 g** | *not mentioned* | — |
| absolute peak-to-peak | **2.827 g** | 2.70 g implied by the band | — |
| pre-encounter cruise datum | **0.9513 g** (sd 0.0100) | 1.000 g by definition | **unresolved, declared** |

**Three checks, none of which set a parameter.** (1) The paper's band, above — the prose
quotes the *sustained* band, and the absolute peak is a single narrow spike 0.153 g above it
that the paper does not mention. (2) **A negative control on the method**: the same
column-envelope extraction run on the *vertical wind* panel must return zero through the
pre-encounter cruise, because the aircraft was straight and level in smooth air. It returns
**+1.17 ft/s**. So the method carries no vertical bias and the 0.951 g datum belongs to the
G LOAD panel, not to this code. (3) **Gust spacing**, the one channel Mehta's identification
did not set.

**Two results that change what §5 may claim.**

**(a) The model's wind is weaker than the recorded wind.** The digitised vertical-wind panel
reaches **+64.8 / −98.3 ft/s**. The 747 flown through Mehta's five-vortex field sees
**+59.1 / −86.8 ft/s** — **11.7% weaker on the down gust, 8.8% on the up.** Mehta's field is
a smooth five-vortex *fit* to that record, so under-shooting its extremes is expected; what
is new is that the size of the under-shoot is now **measured** rather than assumed away. In
the linear range load goes with gust, so this is a previously-unattributed piece of the 32%
shortfall, and it points at **the wind**, which §5 had stopped naming as a candidate.

**(b) The spacing comparison gets tighter, and the model gets worse against it.** The two
principal up-gusts land at t = 209.3 and 214.9 s, i.e. **5.60 s apart**. `cat_uncertainty.py`'s
flown model gives 5.36 s centre-to-centre (**5.40 s since session 30**, when `boeing747`
declared CR-2144's speed derivatives and began meeting the cores off-centre; its
peak-to-peak reading moved further, 5.28 → 5.50 s, and `test_cat_validation` now asserts the
original claim on the undeclared entry and bounds the shift on the shipped one). Against the digitised figure that is **−4.3%**;
against the prose *"about 5 sec apart"* alone it read **+7.2%**. **The figure is a tighter
reference than the sentence describing it**, and the sign flips.

**One thing declared rather than resolved.** The trace sits at 0.9513 g through level cruise
where the definition says 1.000. Check (2) rules out the method. It is either a recorder bias
or a registration offset of the plotted curve inside its own axes, and **the figure alone
cannot separate those**. Both envelopes are therefore reported RAW, in plot coordinates,
because that is what the paper's own numbers are. **Do not "correct" the trace to 1.0 g** —
that would be fitting the record to the model.

**Which denominator to quote.** Three are now defensible and they are not interchangeable:
2.70 g (the prose band, and what §4's headline row has always used), 2.674 g (the digitised
*sustained* band) and 2.827 g (the digitised *absolute* peak-to-peak). The 747's 1.839 g is
**68.1%**, **68.8%** and **65.0%** of them respectively. **§4's headline row keeps 2.70 g**
so the number does not silently move, and this row records the other two beside it.

### MIL-F-8785C Figure 7, and a second sealed prediction settled — session 27

`scripts/digitise_mil_f_8785c_fig7.py` reads Figure 7 out of `refs/MIL-F-8785C.pdf` p. 49.
At 37,000 ft the severe curve gives **σ_w = 15.7 ft/s = 4.80 m/s**.

**It is a band, not a point, and the reason is in the figure.** SEVERE and the 10⁻⁵
exceedance curve are drawn as **one stroke of ink** at that altitude, so the value is
**4.80 ± ~0.12 m/s** depending which member the stroke belongs to. This project does not read
a number out of a merge and call it one line.

**Independently cross-checked.** JSBSim's `FGWinds.cpp` transcribes the same figure; its
severe row interpolated to 37 kft gives **4.822 m/s** — 0.5% away, inside the merge band, and
neither reading set the other. *An earlier pass here quoted 0.4% by reading the merged stroke
as though it were SEVERE alone; that was too good and is superseded.*

**This settles `mil_f_8785c_sigma_w_exceeds_the_mehta_ceiling`: RIGHT**, by 0.34 m/s against
`wind.mehta_residual_ceiling` = 4.459 m/s, with the whole merge band clearing it. **Two of
three sealed predictions are now settled and both were right.** What it buys is what the
sealed reasoning reserved: the ~4–5 m/s closure is **physically available** at the
specification's severe level. It is **not** evidence that Hannibal contained it.

### The Dryden ensemble at Yoshimura's own sample count — session 27

Session 25 ran 32 flights where Yoshimura et al. 2023 used **151**, and §5 carried the
under-sampling as a caveat. `scripts/cat_spectra.py --seeds 151` closes it. 151 × 100 s at
each intensity, 15,097 s of record apiece.

| σ_w, m/s | peak at **N = 32** | peak at **N = 151** | vs short period (0.1640 Hz) |
|---|---|---|---|
| 2.108 (`mehta_unmodelled_wind`) | 0.1400 Hz | **0.1700 Hz** | **+3.6%** |
| 4.459 (`mehta_residual_ceiling`) | 0.1700 Hz | **0.1700 Hz** | **+3.6%** |

**The intensity-dependence was under-sampling, and it is gone.** At N = 32 the low-σ peak sat
0.1400 Hz — 14.7% *below* the short period and less than one bin inside the sealed band's
lower edge — and session 25 read that as the falling Dryden input pulling the resonance down.
At N = 151 **both intensities land on the same 0.1700 Hz**, per-flight median 0.1700 in both
cases. So the honest reading is that the N = 32 low-σ figure was noise, not a competing
mechanism, and the resonance is **cleaner** than the settled prediction's outcome suggests.

**The settled prediction's `outcome` is deliberately NOT edited.** It records 0.1400 / 0.1700
at N = 32, which is what was measured when it was settled, and `predictions.py` rule 1 exists
precisely so a settled entry is not improved after the fact. **It stands as written; this row
supersedes it as a measurement, per §4's own rule.** The verdict is unchanged either way —
RIGHT, at both intensities, and now with room to spare.

Half-power band 0.1200–0.2300 Hz at σ = 2.108 and 0.1200–0.2400 at 4.459. Peak |Δn| 0.3698 g
and 0.8378 g. The σ ratio is 2.1156 against an `n_z` rms ratio that tracks it, so the response
is linear in intensity across the sourced range — which is the other thing the two σ values
were there to measure.

### The CAT source pass (session 23)

Everything here comes from `scripts/cat_validation.py`, whose printed report and four
figures are the artefact. Run it as:

```
PYTHONPATH=<worktree root> .venv/Scripts/python.exe scripts/cat_validation.py --outdir runs/cat
```

**`PYTHONPATH` is not optional** — `python scripts/…` resolves `atisim` to the **main
checkout**, which §10 documents and which this script caught on its first run by printing
`atisim.__file__` before doing anything else.

#### Mehta 1987's five-vortex field, flown by the 747 at Mehta's own 37,000 ft

| Quantity | Model | Source | Note |
|---|---|---|---|
| vertical gust peak | −86.8 / +59.1 ft/s | `V₀` = 86.8 ft/s | the negative peak lands on `−V₀` exactly, which is the Rankine core signature |
| load factor `n_z` | **−0.398 to +1.441 g** | **−1.0 to +1.7 g** (DC-10, TM-102186 Fig. 6) | inside the measured band, at **68%** of its peak-to-peak |
| air-relative α | −5.98 to +7.86° | — | inside the 10° linear band |
| air- vs inertial α | diverge by up to **6.85°** | — | the reason `vortex_viz` exists separately from `viz`. Computed by the script, not read off the figure |
| start out of trim | **−0.0001 g** | — | `fly_in_moving_air`; the still-air start was **+0.198 g** out and inflated peak load by 12% |
| `σ_n` | **0.6394 g** | severe ≥ 0.3 g (Misaka) | severe, on a field identified from an encounter that injured people |
| vertical gust peak — **session 30, speed derivatives declared** | **−81.5 / +51.9 ft/s** | `V₀` = 86.8 ft/s | **no longer lands on −V₀.** The declared 747 pitches further and flies through the cores off-centre, so it meets weaker gusts — which also widens the gap to the recorded wind §5 attributes part of the shortfall to |
| load factor `n_z` — **session 30** | **−0.374 to +1.367 g** | −1.0 to +1.7 g | inside the band, at **64.5%** of its peak-to-peak: further from the record than the row above |
| load factor `n_z` — session 30, with CR-2144's 10 ft thrust line | −0.367 to +1.372 g | −1.0 to +1.7 g | inside the band, at 64.4%: the line moves the headline 0.1 point |
| load factor `n_z` — **end of session 30: 5.70 ft line, field REPLAYED on its identified path (the headline)** | **−0.415 to +1.619 g** | −1.0 to +1.7 g | inside the band, at **75.3%**; as flown at the 747's own altitude it is 64.5% |
| air-relative α — **session 30** | −5.83 to +7.16° | — | inside the 10° linear band |
| air- vs inertial α — **session 30** | up to **6.47°** | — | — |
| `σ_n` — **session 30** | **0.6273 g** | severe ≥ 0.3 g | still severe |

**Absolute agreement is not claimed and is not available** — DC-10 against 747, roughly
0.8× the wing loading, §5 rules it structurally out. What is claimed is that the model
lands **inside** the measured band and reaches about two thirds of it.

#### TM-102186 Fig. 8 — the ordering, and the mechanism

Six aircraft through the same field, at their own cruise conditions. `trav/T_sp` is core
traverse time over the aircraft's own short period, which is the paper's stated mechanism.

| aircraft | V ft/s | `trav/T_sp` | pitch p-p | `n_z` min | incidence gain | \|α\| peak |
|---|---|---|---|---|---|---|
| cherokee | 164 | 4.024 | 46.98° | +0.403 | 0.21 | 5.44° |
| cessna172 | 197 | 4.046 | 74.92° | +0.413 | 0.18 | 5.51° |
| boeing737_approach | 439 | 0.599 | 17.17° | −0.136 | 0.54 | 7.06° |
| boeing747_approach | 278 | 0.512 | 40.08° | +0.070 | 0.56 | 9.78° |
| boeing737 | 776 | 0.363 | 7.35° | −0.374 | 1.00 | 5.17° |
| boeing747 | 774 | 0.196 | 6.70° | −0.398 | 1.49 | 7.86° |
| **boeing747, session 30 (speed derivatives declared)** | 774 | 0.196 | **7.23°** | **−0.374** | **1.55** | **7.16°** |

TM-102186 Fig. 8, **digitised** (±3°, ±0.15 g — extrema only, off a 1989 photocopy;
nothing asserts against these):

| | V ft/s | pitch p-p | `n_z` min | `n_z` max |
|---|---|---|---|---|
| RPV | 150 | 66.0° | +0.35 | 2.05 |
| EXECUTIVE | 700 | 16.0° | −0.35 | 1.95 |
| AIRLINER | 800 | 5.0° | −0.55 | 1.55 |

**Three results, in increasing order of how much they are worth.**

1. **The pitch ordering reproduces.** Slow aircraft pitch hugely, fast ones barely.
   Cherokee 46.98° against the RPV's 66.0°; 747 6.70° against the AIRLINER's 5.0°, inside
   the ±3° read error. The middle slot is empty — the registry has nothing at 700 ft/s and
   the nearest entry is a different flight regime, so it was left empty rather than filled.
2. **The `n_z` minimum ordering REVERSES, in both.** Reference: RPV +0.35, EXECUTIVE
   −0.35, AIRLINER −0.55. Model: cherokee +0.403, 737 −0.374, 747 −0.398. The aircraft
   that pitches least eats the most load. This is the non-obvious half of the paper's
   claim and the model gets it, with the cherokee/RPV pair agreeing inside the read error.

   **Session 30: both orderings still hold, with less margin.** With CR-2144's speed
   derivatives declared, the 747 pitches **7.23°** — still inside ±3° of the AIRLINER's 5.0°,
   and still under the 737's 7.49° in the same run. Its `n_z` minimum is **−0.374** against
   the 737's **−0.364** in that run: still the lowest, but by **0.010**, where the bare entry
   led by 0.039 on the same tree. The declaration narrows the reversal and does not undo it.
   (The other five rows above were not touched by the declaration; where today's run differs
   from them in the third figure, that predates session 30.)

   **Session 32, on the current tree** (the thrust line at 5.70 ft), as
   `notebooks/validation-ladder.ipynb` computes it: the 747 pitches **7.01°** with an `n_z`
   minimum of **−0.372**, the 737 **7.49°** and **−0.364**. Both orderings hold; the reversal's
   margin is **0.008 g**. The notebook asserts the pitch and `n_z`-minimum orderings between
   the Cherokee and the 747 — the paper's two filled slots — not the 747–737 margin. Later in
   phase 5, `test_cat_validation.py` asserts the same two orderings.
3. **The mechanism itself is monotonic, six for six.** *Incidence gain* is the α that
   actually reached the wing divided by the α a rigidly-held attitude would have seen
   (`atan(max|w_up|/V)`). Against `trav/T_sp` it falls **1.49 → 1.00 → 0.56 → 0.54 → 0.21
   → 0.18** with no inversions. Above 1 the aircraft's own motion *adds* incidence; below
   it, the aircraft pitches away and sheds the gust. Peak-to-peak pitch does **not**
   collapse as cleanly, because it also scales with how large the gust is in incidence
   terms, and that varies fourfold across the fleet through airspeed alone.
   **Session 32:** 1.55 → 0.98 → 0.57 → 0.55 → 0.21 → 0.18 on the current tree, still
   monotone — and now asserted, by the validation notebook; until then nothing did (§9,
   session 32, point 11). Later in phase 5 the suite asserts it too, in
   `test_cat_validation.py`.

**Every run stays inside the 10° linear band — and the mechanism is why.** A Cherokee at
50 m/s meets a gust worth 25.6° of incidence at a frozen attitude and sees 5.44°, because
it has time to pitch into the flow. Without that relief the low-speed runs would be
outside the model's validity and the test could not be run at all.

**Confound, stated:** each aircraft flies the field at its own cruise altitude, so density
varies with the aircraft. The vortex field itself does not — `vortex_wind` is a velocity
field with no density in it — and the 747 flies Mehta's own 37,000 ft, so the headline
case carries no such confound.

#### Where the 32% load shortfall is NOT (session 23 follow-up)

`scripts/cat_bounds.py` §A. The Mehta run reaches 68% of the DC-10's recorded
peak-to-peak. Three candidate causes inside the model, each turned one at a time:

| Variant | `n_z` min | `n_z` max | Δ(up increment) | Δ(down increment) |
|---|---|---|---|---|
| baseline — tangent gust rates, point loads | −0.3976 | 1.4410 | — | — |
| gust rates fitted across the airframe | −0.3361 | 1.4336 | **−1.67%** | **+4.40%** |
| strip-integrated loads | −0.3976 | 1.4410 | **0.00%** | **0.00%** |

Step size, peak load: dt 0.02 → 0.01 moves it **−0.085%**, → 0.005 **−0.032%**,
→ 0.0025 **−0.021%**. Converged.

**Together these bound the model's own approximations at about 4.4% of a 32%
gap.** The shortfall is not numerics and not the point-sampled gust.

**The strip result is exactly zero, and that is informative rather than a
null.** `loads.strip_increment` is roll-only and `vortex_wind` has no `y`
dependence at all, so integrating across the span *must* return the point value.
The point-gust assumption on this field can only be probed through the
longitudinal gust RATES, which is what the sampled-rate row does — and it is the
**down** excursion that moves, not the peak.

#### What the random component Mehta excludes is worth (session 23 follow-up)

`scripts/cat_bounds.py` §B. Mehta states his fit represents "the sudden,
violent, and periodic disturbances … and not the small, random fluctuations that
are part of the overall turbulence", so the field flown is a *smoothed* version
of the air the DC-10 met. A Dryden layer (`wind.dryden_vertical_field`, MIL-F-8785C
form, `L_w` = 1750 ft sourced, **σ_w not sourced and therefore swept**) was
superposed, six seeds per point:

| σ_w m/s | `n_z` min, mean [min, max] | `n_z` max, mean [min, max] | reaches the band? |
|---|---|---|---|
| 0 | −0.398 | 1.441 | no |
| 2 | −0.421 [−0.500, −0.343] | 1.527 [1.440, 1.561] | no |
| 4 | −0.423 [−0.613, −0.234] | 1.642 [1.456, 1.737] | **yes** |
| 6 | −0.415 [−0.730, −0.148] | 1.797 [1.643, 1.952] | **yes** |

**σ_w ≈ 4–5 m/s closes the upper extreme. Nothing in this sweep closes the lower
one** — at σ_w = 6 the mean minimum is −0.415 against a recorded −1.0, and even
the most extreme of six seeds reaches only −0.730. The spread widens; the mean
barely moves, because the down-peak is set by the vortex core and the Dryden
layer only adds variance around it.

**Do not read σ_w ≈ 4 m/s as a validated intensity.** It is the value the gap
implies, and MIL-F-8785C gives σ_w as a chart against altitude and exceedance
probability that §3 records as un-digitised. Comparing the two is the next step,
not a step already taken.

> **Session 23c bounded it from the other direction, and the news is not good for
> this explanation.** Mehta's own fit residual puts the unmodelled random
> component at **2.11–4.46 m/s**, and 4.46 is the ceiling — it requires the
> *entire* residual to be vertical, unmodelled, and free of reconstruction
> error. So σ_w ≈ 4–5 m/s is not merely un-validated, it sits **at or above the
> top of what the source data permit.** Flown at the sourced lower bound the run
> does not reach the band at all. See the next subsection.

#### The Hannibal comparison with error bars on both sides (session 23c)

Everything here comes from `scripts/cat_uncertainty.py`, figure `06-uncertainty.png`.
Until this pass every load number quoted against TM-102186 was a point compared
with a point, so "68% of the recorded peak-to-peak" could not be split into *the
model is 32% wrong* and *the inputs are not known to 32%*.

**The standing problem this addresses.** The vortex parameters were identified
**from** the recorded accelerations, through Parks' and Mehta's aircraft model.
Predicting those accelerations back therefore tests the composition of two
aircraft models, not this one alone. Two ways out are taken below: a channel
outside that loop, and a bound on what the loop's inputs are worth.

##### A. Gust spacing — the channel the identification did not set

TM-102186 p. 3–4, in **prose**, not a figure: *"sharp up-and-down gusts about
5 sec apart"*. Spacing is set by the fitted core **positions** and the aircraft's
speed; it is independent of the fitted **strengths**, and that independence is
measured rather than asserted — the field is exactly linear in `V₀`, so scaling
`V₀` by 0.5, 1.5 or 3.0 moves every wind value and **not one turning point**.

| Reading of "apart" | Model | vs record |
|---|---|---|
| peak-to-peak | 5.290 s | **+5.8%** |
| centre-to-centre | 5.360 s | **+7.2%** |
| trough-to-trough | 5.430 s | **+8.6%** |

The three span 0.14 s, so the answer is a property of the field and not of the
definition. Geometric prediction from the 4,104 ft core separation at this 747's
774 ft/s is 5.302 s, and the flown run adds 1.1% to it.

**The residual is a speed proxy, not a field error.** Exactly 5.0 s needs
**250.2 m/s (M 0.848)**; this 747 flies **235.9 m/s (M 0.800)**, which is the
Mach its derivative set is tabulated at. The DC-10's own true airspeed is in no
source held here, and 6% between two transports at 37,000 ft is unremarkable.
The model is **slow**, which is the direction that mismatch requires — a fast
model would have needed explaining.

##### B. What Mehta's own fit leaves over

Mehta's Appendix Eq. (A3) is **`J = (1/N) Σ eᵀ B e`**, with `B` the identity for
every cost he quotes. **The `1/N` is the load-bearing part**: `J` is a *mean*
square, so it converts to an RMS wind residual without `N` — which the paper
never states and no source here supplies. Read as a *sum* it is uninterpretable:
over any plausible `N` the implied residual lands well below the error of the
winds being fitted — a factor of **4.1 even at `N` = 30**, and 6.0 at `N` = 65 —
which no honest fit can do.

| n | cost `J` (ft/s)² | RMS residual | Δ`J` | Δ RMS |
|---|---|---|---|---|
| 2 | 482 | 6.692 m/s | — | — |
| 2 | 355 | 5.743 m/s | −26.3% | −14.18% |
| 3 | 303 | 5.306 m/s | −14.6% | −7.61% |
| 4 | 226 | 4.582 m/s | −25.4% | −13.64% |
| 5 | **214** | **4.459 m/s** | −5.3% | −2.69% |

The first row is the **manual startup estimate**, not a fit — see §3, where this
document previously paired it with a converged cost as one "history".

**J = 214 is a floor, not a stopping point.** Mehta p. 30: *"Further increases in
the number of vortices (n = 6,7, etc.) do not result in decreases in the cost. In
fact, the algorithm 'pushes' the extra vortices away from the flight path"*. The
fifth vortex already buys only 2.7% of RMS. **So the residual bounds the field
FORM** — what a Rankine array cannot represent about this record — rather than
one author's patience.

Decomposing it against Lester's reconstruction error, since Mehta fitted
*reconstructed* winds and his residual therefore contains theirs:

| | RMS wind |
|---|---|
| total residual, `√J` | 4.459 m/s |
| reconstruction floor, Lester Table 1 RSS | 3.316 m/s |
| **unmodelled, per component** (quadrature) | **2.108 m/s** — a *lower* bound |
| **ceiling**, all of `J` vertical and unmodelled | **4.459 m/s** — unit-robust |

Both caveats on the lower bound push the same way, **up**: independence is
assumed where Mehta fits bias and trend terms explicitly, and Lester's case had
no ATC radar fixes where Hannibal did. The ceiling survives any unit convention
for the horizontal term because `B = I` makes both squares non-negative.

**Flown, 8 seeds each:**

| σ_w | `n_z` min, mean [min, max] | `n_z` max, mean [min, max] | reaches band? | peak \|α\| |
|---|---|---|---|---|
| 2.108 (sourced lower) | −0.394 [−0.506, −0.246] | 1.529 [1.441, 1.568] | **no** | 8.85° |
| 4.459 (ceiling) | −0.367 [−0.641, −0.031] | 1.663 [1.459, 1.790] | yes | 9.97° |

**This weakens session 23b's Dryden reading rather than confirming it.** The
intensity that closes the gap is the intensity at the absolute ceiling of what
the sources permit, and the ceiling run sits **on** the 10° edge of the linear
range — the last intensity this model may be asked about at all. What the pass
*does* buy is that σ_w went from unbounded to **2.11–4.46 m/s** without
digitising anything.

##### C. The propagated input band

`V₀ ± 8.45%` is sourced (Lester's vertical RMS over Mehta's `V₀`) and is a
**ceiling, not a 1σ** — it is the error of one reconstructed sample, and `V₀` was
fitted to `N` of them. `r₀ ± 15%` is **declared**: Mehta's sensitivity study
reports convergence from initial guesses of 100–1300 ft, which is a statement
about his algorithm, not about how well `r₀` is known.

| Variant | `n_z` min | `n_z` max | p-p | % of recorded | \|α\| |
|---|---|---|---|---|---|
| baseline (identified) | −0.3976 | 1.4410 | 1.8385 | 68.1% | 7.86° |
| `V₀ × 0.9155` (sourced) | −0.3312 | 1.4287 | 1.7599 | 65.2% | 7.72° |
| `V₀ × 1.0845` (sourced) | −0.4374 | 1.4482 | 1.8855 | 69.8% | 7.89° |
| `r₀ × 0.85` (declared) | −0.4582 | 1.4765 | 1.9348 | 71.7% | 8.01° |
| `r₀ × 1.15` (declared) | −0.3418 | 1.4470 | 1.7888 | 66.3% | 7.74° |
| **corner: `V₀` high, `r₀` low** | −0.5065 | 1.4565 | 1.9631 | **72.7%** | 8.15° |

**The whole input band tops out at 72.7%. The 32% shortfall is not inside the
uncertainty of the inputs.**

##### Why no amount of wind helps: the peak load is saturated

> ### ⚠ SUPERSEDED, end of session 30 — the saturation is the as-flown geometry
>
> Everything in this subsection was flown with Mehta's field at the 747's OWN altitude. Each
> stronger updraft lifts the fixed-control 747 further above cores 3 and 4, so the gust it
> meets grows more slowly than V₀ — that is most of the "saturation". **Replayed on the path
> the field was identified along** (the headline since the end of session 30, justified by
> Parks 1985 Fig. 6), the shipped 747's peak **tracks the gust** (elasticity **+1.07** at ×3),
> reaches the recorded +1.7 g at **×1.15 V₀ with |α| 8.8°** — inside the linear range — and
> the full 2.7 g swing at about ×1.35. **"Amplitude is excluded" does not survive.** The
> tables below are kept as the as-flown record; the bracket they establish still holds in that
> form (×3.0–×3.25 on the shipped entry). §4's first entry has the replayed sweep.

> **Session 30 — half of this weakened, and the sharp half did not.** Every figure in this
> subsection is the 747 **without** Mach derivatives. With CR-2144's speed derivatives
> declared on `boeing747`:
> - **The saturation elasticity is +0.305, against +0.097 for the same entry with the seam
>   shut** (`test_cat_validation`, dt 0.02). The shipped model is *less* saturated, and the
>   pitching-moment term does most of that alone: Cm_M only +0.266, CL_M only +0.129, CD_M
>   only +0.102.
> - **The bracket survives.** The shipped entry reaches the recorded +1.7 g and leaves the
>   10° linear range in one interval — now ×3.0–×3.25 (1.591 g at 8.94°; 1.745 g at 10.32°),
>   where it was ×3.25–×3.5. There is still no gust strength at which the model both reaches
>   the record and may be believed.
> - **What weakened is the supporting argument.** "The load cannot be an amplitude error
>   because tripling the gust barely moves it" is weaker than the text below makes it, and at
>   3× the gust the Mach excursion is largest, which is where the declared tangent is least
>   trustworthy.

| `V₀ ×` | `n_z` min | `n_z` max | up increment | elasticity | \|α\| | |
|---|---|---|---|---|---|---|
| 1.00 | −0.3976 | 1.4410 | 0.4410 | — | 7.86° | |
| 1.25 | −0.4422 | 1.4567 | 0.4567 | 0.143 | 7.76° | |
| 1.50 | −0.3372 | 1.4465 | 0.4465 | 0.025 | 7.97° | |
| 2.00 | −0.1429 | 1.3820 | 0.3820 | −0.134 | 7.98° | |
| 3.00 | +0.0820 | 1.5257 | 0.5257 | 0.096 | 8.13° | |
| **3.25** | −0.2002 | **1.6416** | 0.6416 | 0.202 | **9.11°** | short, **inside** |
| **3.50** | −0.5839 | **1.7862** | 0.7862 | 0.313 | **10.34°** | reaches, **outside** |
| 4.00 | −1.9049 | 2.2065 | 1.2065 | 0.579 | 13.93° | outside |

**Tripling the identified gust leaves the up-increment at 0.526 g against a
recorded 0.70, with peak |α| still near 8°.** A peak that tracked the gust would
have elasticity ≈ 1; this one is under 0.15 and changes sign. The aircraft
pitches away and sheds the gust — TM-102186 Fig. 8's incidence-gain mechanism
seen from the inside.

**And the two boundaries coincide, which is the sharp form of the result.** The
peak first reaches +1.7 g between **×3.25 and ×3.50**, and |α| leaves the 10°
linear range in the **same interval**. So there is no gust strength at which this
model both reaches the record and may be believed. That is a stronger statement
than "the shortfall is large": it **excludes amplitude** as the explanation
rather than merely bounding it. Anything past the crossing is past the 12° where
§7 says this aero reports lift the sources deny — not a harder test, a different
model.

**What all three sections together leave.** The shortfall is not numerics (0.14%
over 8× `dt`, session 23b), not the point-sampled gust (≤4.4%), not the strip path
(0.00%), **not the identified parameters (≤72.7% at the favourable corner), and
not gust amplitude at any strength inside the linear range.** The random
component is bounded to 2.11–4.46 m/s and only its ceiling reaches. What is left
is the aircraft: a DC-10 record flown by a 747, and no buffet-onset data — which
§5 already lists as structurally out of reach from the sources held.

#### Lester's Greenland 747 — one wave cannot produce both observations

`scripts/cat_bounds.py` §C. The first load comparison in the project flown by
the aircraft type the record is *of*: a B-747, at Lester's 33,000 ft, through a
22 km lee wave.

| `w0` m/s | `n_z` min | `n_z` max | peak \|Δn\| | climb m | band widths out |
|---|---|---|---|---|---|
| 3 | 0.955 | 1.052 | 0.052 | 223 | 0.25 |
| **6** (Doyle's largest) | 0.911 | 1.106 | **0.106** | **451** | 0.29 |
| 12 | 0.826 | 1.222 | 0.222 | 915 | 0.39 |
| 30 | 0.602 | 1.610 | 0.610 | 2367 | 0.69 |

Lester records **two** things about the same event, and the model inverts each:

- the **300 m altitude gain** needs `w0` ≈ **4.0 m/s** — squarely inside Doyle's
  measured 3–6 m/s;
- the **+2.7/−1.0 g** needs `w0` ≈ **113 m/s**, which is 18.8× Doyle's largest
  and not a physical lee-wave amplitude.

**The two imply amplitudes 28× apart, so one smooth wave cannot produce both.**
The slow, large-scale response is reproduced at a plausible amplitude; the fast
one is not reproduced at any. That is a statement about the WIND MODEL, not the
flight dynamics — and it is exactly what the source says happened, since Lester
reads the flight-level windspeed collapse at the point of largest vertical
motion as a critical level from *overturning* waves. The loads came from the
wave breaking down, which `wind.LeeWave` does not contain.

**Caveat, reported by the run itself.** The band declared on `boeing747` this
session puts 33,000 ft outside it: at `w0` = 6 m/s the run is 0.29 band widths
out, on the altitude axis only (Mach 0.764–0.835 is inside). `aircraft.py`'s
interpolation puts the short-period frequency error there at about 5%. The guard
fired on its first real use.

#### The 747 at a third CR-2144 flight condition, M 0.8 / 6,096 m

Condition reproduced before any derivative is used: `U₀` 252.8 m/s against Table A3's 253,
`ρ` 0.6527 against 0.653, and **`C_L` 0.2657 against 0.266** — the last from mass, wing
area and the ISA atmosphere alone, using no derivative from either document.

| | `ω_n` | vs ref | `ζ` | vs ref |
|---|---|---|---|---|
| AtiSim, FC9 (40,000 ft) derivatives | 1.5926 | **+23.5%** | 0.4723 | −17.1% |
| AtiSim, Table A2 (6,096 m) derivatives | 1.2823 | **−0.6%** | 0.5031 | −11.7% |
| reference, Table A5 | 1.2900 | — | 0.5700 | — |

**The frequency error is entirely the derivative set.** Swap the coefficients for the ones
CR-2144 tabulates at *that* condition, change nothing else, and 23.5% becomes 0.6%.

**The residual damping error is entirely the missing `C_mα̇`.** Deleting `M_α̇ = −0.176`
from the reference's own damping formula predicts −12.2%; the model measures −11.7%.
Agreement to half a point of damping says nothing else contributes.

**And putting it back closes it, which is the prediction run forward.** The model has
carried `CLadot`/`Cmadot` fields since the α̇ work — `dynamics.derivatives` resolves both
the gust's half and the aircraft's own implicit half — but `boeing747` leaves them at
zero. Restoring Table A2's own `C_mα̇` = −5.40 alongside its other derivatives:

| 747 short period, M 0.8 / 6,096 m | `ω_n` | vs ref | `ζ` | vs ref |
|---|---|---|---|---|
| Table A2 derivatives, no α̇ | 1.2823 | −0.59% | 0.5031 | **−11.73%** |
| **+ `C_mα̇` = −5.40 (Table A2)** | 1.2823 | −0.59% | **0.5708** | **+0.14%** |
| reference, Table A5 | 1.2900 | — | 0.5700 | — |

**Given CR-2144's own derivatives for the condition, including α̇, this model reproduces
CR-2144's own short period to 0.6% in frequency and 0.14% in damping.** `ω_n` does not
move, which is the check that the term went into the right place — `M_α̇` enters the
damping and not the frequency.

**`boeing747` is NOT given α̇ terms, because no source supplies them at flight condition 9.**
Table A2 is 20,000 ft and this entry is 40,000; applying it across 2.48× of dynamic
pressure is the exact error this section just measured. CR-2144 Table IX-4 tabulates
`Zwd` and `Mwd` for FC9 and `aircraft.py`'s comment says so, but the values were never
transcribed and the document is not held.

**`boeing747_approach` is not given them either, and that one is a judgement call.**
Table IX-2 *does* tabulate `CL_α̇` and `Cm_α̇` for that condition, and `aircraft.py`
records them. Measured against Caughey Eq. (5.54):

| | `ω_n` sp | `ζ` sp | `ω_n` ph | `ζ` ph |
|---|---|---|---|---|
| as shipped | +1.4% | −5.5% | −0.2% | −4.6% |
| with `C_mα̇` −3.2 and `C_Lα̇` +6.7 | **−0.3%** | **+0.1%** | −0.2% | **+8.4%** |

The short period — the mode α̇ physically governs — improves markedly. The phugoid
damping, which it should not touch, degrades **past the 5% tolerance
`test_validation.py` asserts**. Adopting the terms would mean re-pinning that tolerance
to let the change through, and `docs/ASSUMPTIONS.md` B5 is the precedent for not doing
that. The measurement is kept in
`test_cat_validation.py::test_the_approach_747s_tabulated_alpha_dot_terms_are_a_trade_not_a_win`
so it survives the decision not to act on it.

**The sign is resolved, and empirically.** `aircraft.py` transcribes `CL_α̇` as −6.7;
Caughey uses +6.7 for the same CR-2144 case, and +6.7 is what a conventional aft tail
must have. Flown both ways, +6.7 takes short-period `ω_n` to −0.3% and −6.7 takes it to
+3.2% — the data agrees with the physics, so the transcribed sign is the wrong one.

Verification that the reference is usable at all, before any of the above: Table A2's
dimensional column is recovered from its non-dimensional column using **AtiSim's own**
mass, inertia, wing area and MAC, worst error **0.47%** across five derivatives. Same
aeroplane, and both transcriptions sound.

#### The headline field flown by a second engine (session 23d)

`scripts/vortex_compare.py`, now carrying a fifth encounter. Session 21 compared
the two engines on **single Parks cores**; this is the first time both have flown
**Mehta's converged five-vortex array** — the field the headline result actually
uses — with the same aircraft (`boeing747_jsbsim`, recovered from JSBSim's own
B747), the same starting state and the same density.

**The gate first.** The generator writes the array field in numpy with an oblique
traverse; `wind.vortex_wind` writes it in JAX. They agree to **8.8e-10 m/s** at
every sample, so neither the five-way superposition nor the 31° obliquity is
carrying a shared bug.

| | `n_z` span | `θ` span | % of the recorded 2.70 g |
|---|---|---|---|
| **JSBSim** | **2.0152** | **10.1066°** | **74.6%** |
| AtiSim, translational (like-for-like) | 1.8124 | 10.1854° | 67.1% |
| AtiSim, + `ω_gust` | 2.0517 | 9.3905° | 76.0% |
| AtiSim, + both gradient terms | 2.0011 | 9.4123° | 74.1% |
| like-for-like error | **−10.1%** | **+0.8%** | |

**The result step 5 existed to get: neither engine reaches the record.** An
independent flight-dynamics code, given the same field and the same aeroplane,
also falls well short of the DC-10's recorded peak-to-peak. The solver is
excluded as the explanation **on the headline field**, rather than inferred from
adjacent cases.

##### But the two engines part on the array where they did not on one core

This qualifies session 22, which measured both engines under-predicting "by the
same amount" and read that as the shortfall being in the inputs.

| | window | in short periods | JSBSim `n_z` extremes | AtiSim `n_z` extremes |
|---|---|---|---|---|
| Hannibal, one core | 305 m, 1.3 s | 0.20 | max @ 2135 m, min @ 2434 m | max @ 2135 m, min @ 2436 m |
| Mehta, five cores | 8,125 m, 34.4 s | 5.2 | max @ 7794 m, min @ 7383 m | max @ 6511 m, min @ 6116 m |

On the single core the engines put their extremes **within two metres of each
other**. On the array they **select different cores** — JSBSim's pair sits on core
3 (7,209 m), AtiSim's on core 2 (5,958 m). And the disagreement is one-sided:
AtiSim's **peak is 2.2% higher** (1.7148 vs 1.6773) while its **trough is 3.5×
shallower** (−0.0976 vs −0.3379). The whole −10.1% span error is the trough,
which is the channel that has been short against the DFDR from the beginning.

**A hypothesis that was tested and failed.** The obvious cause is airspeed: the
array run is 45 s at fixed throttle and JSBSim's true airspeed swings −5.5% across
the window against −0.009% on the single core, and `n_z ∝ q̄ ∝ V²`. But shrinking
the window to ±2 core radii around the penetrated pair leaves the drift at −0.19%
and the disagreement at **−7.6%**, and AtiSim's span is **identical (1.8124) at
every window width from ±2 to ±32 radii**. The drift is not the cause; it is
recorded here so the next reader does not spend the same hour on it.

**What is left, stated as the open question it is.** The array is the first
cross-code comparison that runs for *several* short periods rather than a fifth
of one, and the engines' trajectories diverge enough over five cores to land on
different worst cores. Whether that is a damping difference, a thrust/drag
difference over 45 s, or an accumulation of both is not established here.
§8 carries it.

#### Fig. 8 with error bars — §7 step 6, at last (session 23d)

`scripts/cat_ensemble.py`, figure `07-ensemble.png`. Step 6 has waited on step 4
(Dryden) since session 3; step 4 landed in 23b. Its stated criterion is that the
**vortex < updraft < manoeuvre ordering holds across the ensemble.**

σ_w is the **sourced** range from 23c — `wind.mehta_unmodelled_wind` to
`wind.mehta_residual_ceiling` — rather than a picked number. 16 seeds each.

| σ_w | vortex pitch | updraft pitch | ordering | pitch gap | load gap |
|---|---|---|---|---|---|
| 2.108 m/s | 2.113° [1.619, 2.432] | 4.591° [3.318, 5.699] | **16/16** | +0.886° | +0.756 g |
| 4.459 m/s | 2.114° [0.988, 2.809] | 5.253° [2.869, 7.286] | **16/16** | **+0.060°** | +0.343 g |

**The criterion is met, and the interesting number is the margin.** Raising σ_w
by 2.1× collapses the **pitch** gap between the vortex and updraft clouds by 15×,
to 0.060° — they are all but touching. The **load** gap survives at 0.343 g. Fig. 8
is a two-dimensional discriminator, so the categories still separate; but at the
top of the sourced turbulence range they separate on **load alone**, and a pitch-only
reading of the chart would stop working.

> ### ~~they are all but touching~~ ~~a pitch-only reading would stop working~~ — **BOTH WITHDRAWN, session 29**
>
> **The gap between two clouds' extremes is not a measure of whether they overlap**, and this
> paragraph's conclusion rests entirely on reading it as one. §4's "Fig. 8's pitch axis does NOT
> stop discriminating" has the measurement; the three findings that bear on these two sentences:
>
> - **The gap shrinks with N at FIXED σ_w.** Subsampling one grid at σ_w = 2.108: 1.5690° at
>   N = 8, 0.7211° at N = 16, 0.6567° at N = 32, while the separability stays at exactly 1.0000
>   throughout. Collecting more evidence made this statistic say the clouds were *closer*.
> - **Inside §1's envelope the pitch axis separates PERFECTLY** — 0 overlapping pairs out of
>   1,024 at σ_w = 1.0, 2.108 and 3.0. At σ_w = 4.0 the gap is **−0.2995°**, which reads as
>   overlapping, while only **4 pairs in 1,024** are actually misordered.
> - **It is the LOAD axis that degrades faster**, not the pitch axis: across the sweep Cohen's
>   `d` falls ×7.28 on load against ×3.20 on pitch, the reverse of what this paragraph says.
>
> **What survives unchanged is the criterion**: 16/16 on the ordering, at both intensities. And
> one caveat this entry should have carried — **the upper limb is outside §1's envelope**, at
> peak |α| 10.65° in this configuration, so the σ_w = 4.459 row was never a run the model may be
> asked about. §4 also records that the +0.886° above does not reproduce (0.7211° now), most
> likely through the session-28 compressibility merge, and that it was not chased because the
> conclusion drawn from it is withdrawn regardless.

**Two things this does not establish.** The manoeuvre limb is flown at zero wind —
that is the category's definition, not an oversight — so the three-way ordering is
helped by construction and the vortex-versus-updraft separation is the part tested
on equal terms. And a 16-seed gap is an estimate of the extremes of a distribution
whose tails grow with sample size, so "the clouds do not overlap" should be read as
*marginal*, not as established.

#### Sealed predictions — the register (session 23d)

`atisim/predictions.py`. Every number in this section is **retrodictive**: the
paper was open beside the model. That is the weaker kind of evidence and no amount
of it becomes the stronger kind. The register is where this project starts saying
what will happen **before** it can check.

Two entries, sealed against tree `0c72200`:

| | claim | settled by |
|---|---|---|
| `dc10_does_not_close_the_hannibal_gap` | a DC-10 on Mehta's field lands in **[1.563, 2.114] g** and does **not** reach 2.70 | any published DC-10 cruise derivative set |
| `mil_f_8785c_sigma_w_exceeds_the_mehta_ceiling` | the severe curve at 37,000 ft gives **σ_w > 4.46 m/s** | MIL-F-8785C Fig. 7, digitised |

**The first one bets against this document.** §5 now names aircraft type as the
surviving explanation for the 32% and a DC-10 set as the highest-value acquisition.
The prediction says that acquisition will *not* close the gap, resting on §4's
measured null that quadrupling mass moved the load 0.6% because wing loading
cancels in `Δn = ΔC_L/C_L,trim`. One of the two is wrong. That is the point.

**What the seal is.** The seal is the git history: each entry records the commit
its author could see and a hand-written digest of its own claim, and
`test_predictions.py` recomputes it. An accidental edit fails the build; a
deliberate one updates the digest too and shows up as a diff on a line whose only
job is to be stable. It makes tampering **visible, not impossible**, and the module
says so rather than overselling it. A first draft computed the digest at import
from the same fields it hashes, which can never fail — that mistake is recorded in
the module and has its own negative-control test.

#### The lateral dimension, and what it was worth (session 24)

`scripts/lateral.py`, figure `08-lateral.png`. Phase 1 of the plan, and the first
result in this project that is not longitudinal.

##### The two vortex forms, reconciled

`wind.line_vortex_wind` is a second implementation of a field the project already
had, so it earns its place only by reproducing the first where the first is
defined — on the flight path:

| Geometry | vertical | horizontal magnitude | full vector |
|---|---|---|---|
| perpendicular, Δψ = 0 | 7.1e-15 | 5.3e-15 | **7.1e-15** |
| oblique, Δψ = 31° | 1.2e-14 | 7.1e-15 | **7.08** |

**The oblique row is the finding.** The vertical component and the horizontal
*magnitude* agree to machine precision; only the horizontal *direction* differs.
At north = 600 m:

```
vortex_wind      NED = [ -2.1213  +0.0000  -0.4843]
line_vortex_wind NED = [ -1.8183  +1.0926  -0.4843]
north ratio 0.8572 = cos 31°;  east/(−north) 0.5150 = sin 31°
```

Parks' model is two-dimensional in the plane **perpendicular to the vortex
lines**. `vortex_wind` returns his horizontal magnitude along the **flight
path**, exact only at Δψ = 0. At 31° the true perturbation is that magnitude
rotated by Δψ, and **the 0.515 of it that lies across the path — the only
sideslip input this field has ever had — was being discarded.** §5 carries it.

##### Flown: what the missing dimension was worth

747 through Mehta's own five-vortex field at his own 37,000 ft, fixed controls,
dt 0.01.

| Run | \|φ\| | \|β\| | \|p\| | \|p_gust\| | `n_z` max |
|---|---|---|---|---|---|
| point (`vortex_wind`) | **0.000°** | **0.000°** | **0.0000** | **0.0000** | 1.6019 |
| line (`line_vortex_wind`) | 12.508° | 3.106° | 0.1635 | 0.0879 | 1.6375 |
| line + **strip loads** | **15.376°** | 3.533° | 0.2008 | 0.0880 | 1.6344 |
| **session 30, speed derivatives declared:** point | 0.000° | 0.000° | 0.0000 | 0.0000 | 1.4038 |
| **session 30:** line | 18.418° | 4.551° | 0.2301 | 0.0887 | 1.4678 |
| **session 30:** line + strip | **21.579°** | 4.941° | 0.2608 | 0.0888 | 1.4696 |

**Two firsts.**

1. **A wind field rolls the aircraft** — 12.5° of bank, 15.4° with strip loads.
   The point model gives **exactly 0.000**, not something small: the equations
   have no `y` in them.
2. ~~**`loads.strip_increment` moves a reported number**, +22.9% on peak bank.~~ **[Superseded session 32: a double count — `PROJECT.md` §6(h). With one path owning roll the strip effect is −2.5%, its F5 quadrature deficit.]**
   Built in session 14, it had changed every result by *exactly* 0.000000
   because no field varied across the span.

**And the longitudinal answer barely moves** — the up-increment goes 0.6019 →
0.6375 g (+5.9%). So nothing this project has concluded was resting on the
missing dimension, which is the reassuring half of the result.

> **Superseded session 30 — true of the 747 it was measured on, not of the shipped one.**
> With CR-2144's speed derivatives declared on `boeing747`, the same runs give 0.4038 →
> 0.4678 g (line) and 0.4696 g (strip): **+15.9% and +16.3%**. The missing dimension now
> reaches the longitudinal answer, and the attribution is measured rather than argued. A
> pitching-moment speed derivative turns the oblique line vortex's along-track gust into
> pitch. Cm_M alone reproduces most of it (0.129 of the increment, against 0.160 for the
> full set and 0.064 bare). §4's session-30 entry, item 7. `scripts/lateral.py` used to print
> this paragraph's conclusion unconditionally; it now reads the verdict off the number.

##### The two new Dryden components

`wind.dryden_field` adds `u` and `v` to the existing `w`. MIL-F-8785C is not in
the folder — the vertical form was already second-hand and these carry the same
standing — so the check that matters is **internal consistency**: for an
isotropic field the one-dimensional spectra satisfy
`Φ_transverse = ½(Φ_long − Ω dΦ_long/dΩ)`, and the implemented pair satisfies it
to **1.3e-16**, differentiated with `jax.grad` rather than a finite difference.
Both integrate to σ² over the half line; the three components realise σ to within
1.5% and cross-correlate below 0.05.

**These give sideslip, not a rolling gust.** All three components are functions
of along-track distance, so every strip still sees the same vertical gust —
asserted, so the docstring cannot quietly become false.

##### A bug this work made and caught

`cos_dpsi` predates `sin_dpsi` by a session, so `mehta_hannibal_array` carried a
cosine with the sine still at its 0.0 default — **a vortex axis of length
cos Δψ rather than 1**, which scaled the whole induced velocity and looked
exactly like physics. Found by `scripts/lateral.py` reporting a 7 m/s
disagreement in the one geometry where the two forms must be identical. Fixed
twice over: the constructor now sets both halves, and `vortex_axis` normalises,
so a mismatched pair becomes a wrong *angle* — which a test can see — instead of
a wrong *magnitude*, which does not look wrong at all.

**Nothing here is validated against a source.** No document held by this project
records a lateral CAT response. Every number above is a capability
demonstration; §5 says so and the validation claim in §1 excludes it explicitly.

### AtiSim against JSBSim through a Kelvin–Helmholtz vortex (session 21)

Same field, same starting state, same density, fixed controls. atisim run
**translation-only**, because JSBSim has no writable gust-rate input and carries no
gradient at all. Measured **peak to peak across the core window**, which is the encounter;
see the row below for why not from trim.

| Case | Airframe | JSBSim n_z span | AtiSim n_z span | Error | JSBSim θ span | AtiSim θ span | Error |
|---|---|---|---|---|---|---|---|
| Cimarron, 33,000 ft | `boeing737` | 1.1337 g | 1.1408 g | **+0.6%** | 1.1191° | 1.0940° | **−2.2%** |
| Hannibal, 37,000 ft | `boeing747_jsbsim` | 1.7077 g | 1.7443 g | **+2.1%** | 1.1289° | 1.1336° | **+0.4%** |
| Morton, 39,000 ft | `boeing747_jsbsim` | 1.3626 g | 1.4016 g | **+2.9%** | 0.9211° | 0.9703° | **+5.3%** |

**Measured from trim instead, the same runs disagree on pitch by 15–54% on Δθ⁻ and by
31–77% on Δθ⁺, and none of it is the encounter.** *(Quote the column with the number: this
paragraph used to say "15–77%", which silently spanned both and disagreed with the two
places that say 15–54% — §8 and the report — because those quote Δθ⁻ alone.)* The run-in is
fifteen core radii — eight to nine seconds through the vortex's 1/r far field — and attitude
accumulates over it, because atisim is started from JSBSim's trim state and JSBSim's trim is
not atisim's. At the window edge atisim is already 1.55° nose-down of JSBSim on Hannibal and
1.48° on Morton, before the core.

| Case | Run-in | Run-in θ offset at the window edge | Δn⁻ error from trim | Δθ⁺ error from trim | Δθ⁻ error from trim |
|---|---|---|---|---|---|
| Cimarron | 8.24 s | 0.336° | −1.8% | −31.2% | −15.0% |
| Hannibal | 9.04 s | 1.553° | −4.3% | −61.9% | −42.8% |
| Morton | 8.14 s | 1.482° | −6.0% | −76.7% | −53.7% |

Load factor is algebraic in the state and does not accumulate; attitude is an integral and
does. That is the whole of the difference.

### What JSBSim cannot carry, measured (session 21)

`atmosphere/{p,q,r}-turb-rad_sec` are **read-only** in JSBSim 1.3.1's property catalog, and
a write to `q-turb-rad_sec` reads back `0.0` after one step. Every writable wind input is
translational and sampled at one point. The gradient's two terms pull in **opposite**
directions, so a single "with gradient" figure would hide both:

| Case | translational | + wind alphadot | + omega_gust | + both |
|---|---|---|---|---|
| Cimarron, n_z span | 1.1408 g | 1.0108 g | 1.3630 g | 1.2315 g |
| Hannibal, n_z span | 1.7443 g | 1.6923 g | 2.0197 g | 1.9663 g |
| Morton, n_z span | 1.4016 g | 1.3696 g | 1.5693 g | 1.5377 g |

Net, both terms on: **+7.9%** (Cimarron), **+12.7%** (Hannibal), **+9.7%** (Morton) of load span. Translational injection itself
is verified, not assumed: `wind-down-fps = -50` at M 0.78 / 30,000 ft moves α **3.687°**
against a predicted `atan(50/776) = 3.69°`.

### Why the two large errors are large (session 22)

Two numbers in the comparison look bad beside a core response the engines agree on to a few
percent. They are different kinds of error and `scripts/vortex_diagnose.py` separates them.

**The 15–54% pitch error is the datum, not the physics.** See §8. Measured, not argued:
still-air drift accounts for 101–103% of it, and starting in-trim removes it entirely.

> **Session 23d qualifies the paragraph below, and the qualification matters.** "Shared by
> both engines" was measured on the **single Parks cores**, whose window is 1.3 s — a fifth
> of a short period — and then carried over to the Mehta array, which is 34 s and five short
> periods. Flown, the array does not behave like the core: the two engines pick **different
> cores** as the worst one, JSBSim reaches 74.6% of the record and AtiSim 67.1%. The
> conclusion survives in the form that matters — *neither* engine reaches, so the shortfall
> is still not a solver defect — but "by the same amount" was a property of the short
> window, not a general result. §4 has the array comparison.

**The 30–43% DFDR shortfall is shared by both engines**, so by construction it cannot be a
solver difference — it is in the inputs they both received. Cimarron, against the recorded
−1.20 g:

| Lever changed | Δn⁻ | Fraction of the DFDR value |
|---|---|---|
| Nothing (as flown) | −0.841 g | 70% |
| C_Lα × 1.20 | −0.913 g | 76% |
| C_Lα × 1.60 (Prandtl–Glauert at M 0.78) | −1.022 g | 85% |
| Core strength 50 → 60 ft/s | −1.015 g | 85% |
| Core strength 50 → 70 ft/s | −1.192 g | **99%** |
| 737 → 747, four times the mass | −0.837 g | 70% — **no effect** |

The quasi-steady load, before the aircraft responds at all, is **+0.725 g** — essentially
the DFDR's +0.73. The simulated aircraft sheds 42% of that by pitching away and climbing
during the traverse.

Two findings there. **Gust strength dominates**: the fitted Rankine core is smoother than
the data Fig. 4 itself overlays on it, and a 40% stronger core closes the gap almost
exactly. **Airframe mass does not matter at all** — a clean null. The intuition that a
heavier aircraft holds more of the quasi-steady load is wrong, because Δn = ΔC_L/C_L,trim.

> ### ⚠ That row's *explanation* was wrong, and session 26 measured the right one
>
> It read "…and C_L,trim = W/qS, **so wing loading cancels**; four times the mass moves the
> answer by 0.6%". **Wing loading does not cancel — it is the only thing that matters.**
> `C_L,trim = (W/S)/q`, so `Δn = C_Lα·(w_g/V)·q/(W/S)`, and the load goes as **1/(W/S)**.
>
> **The measurement never tested it.** The row is *737 → 747*, and those two entries differ
> by **5.95× in mass but only 1.267× in wing loading** — mass and area moved together, which
> is what an aircraft family does. A controlled sweep, one airframe at one condition
> (`boeing747`, Mehta's field, first core):
>
> | change | W/S | peak-to-peak Δn | vs baseline | `1/(W/S)` would give |
> |---|---|---|---|---|
> | mass ×0.8 | 4434 | 2.2553 g | ×1.192 | ×1.250 |
> | **baseline** | **5542** | **1.8918 g** | — | — |
> | mass ×1.25 | 6928 | 1.5742 g | ×0.832 | ×0.800 |
> | area ×1.25 | 4434 | 2.2543 g | ×1.192 | ×1.250 |
> | **mass ×1.25 *and* area ×1.25** | **5542** | **1.8895 g** | **×0.999** | ×1.000 |
>
> The last row is the real null: **change the mass by a quarter and the area with it, and the
> load does not move — 0.1%.** And `mass ×0.8` and `area ×1.25` land on the same wing loading
> and the same load to four figures, from opposite directions. The scaling is slightly
> **sub**-inverse (×1.192 where 1/(W/S) says ×1.250) because an aircraft that responds more
> also pitches away more, which is §4's own shedding effect.
>
> **So the original sentence had the right null and the wrong reason**, and the reason is what
> §5 and a sealed prediction were both leaning on. See §5 for what that does to the
> aircraft-type explanation.

### Compressibility, a smooth core, and two things that had no cross-check (session 24)

**1. The phugoid frequency gap is closed, by supplying the mechanism rather than the
number.** Session 19 localised 96% of the cruise error to one derivative,
`M_u = ∂q̇/∂v_t`, and named the fix: *"`Cmde` Mach schedule: read from 737.xml's own table,
not chosen."* Session 24 read it — the file is on this machine and the two-point table was
taken from it directly, not from this document's transcription of it.

| | Before | After |
|---|---|---|
| Cruise phugoid ωn error | 6.58% | **+0.45%** |
| Approach phugoid ωn error | 3.39% | **+0.49%** |
| `M_u`, cruise | +1.114e-04 — **wrong sign** vs JSBSim's −1.024e-04 | **−8.66e-05**, right sign |

`737.xml` schedules exactly three coefficients on Mach and no others, checked by reading
the file: `Cmde` (−1.20 at M 0 to −0.30 at M 2), `Clda` (0.100 to 0.033) and `CDmach`. The
third is wave drag, which `aero.wave_drag` already models with an onset at M 0.78998
against JSBSim's table breaking at M 0.79 — so it needed nothing. **The tables change
nothing at the recovery points** (at M 0.78 they give −0.849 and 0.07387, the scalars the
entry already carried), which is why every layer-1 through layer-4 result is unmoved. What
they add is the *slope*, which is exactly what `M_u` reads.

> **`CLalpha` is NOT Mach-scheduled in 737.xml.** JSBSim applies no compressibility
> correction to lift at any Mach. So Prandtl–Glauert is a **departure** from JSBSim, not a
> match to it — the opposite in character to the two tables above, which make AtiSim agree
> with JSBSim *more*. The two must not be conflated.

**2. Prandtl–Glauert: the mechanism ships, and NO ENTRY DECLARES A REFERENCE MACH. That is
a measured decision, and it is the most useful thing this session found.**

`pg_mach_ref` is the Mach at which an entry's lift-slope derivatives were measured; the
correction is `√(1−M_ref²)/√(1−M²)`, **exactly 1 at M_ref**, so an entry flown at its own
recovery condition is bit-for-bit unmoved and the correction only acts away from it.
Writing it as an absolute `1/√(1−M²)` would assume every tabulated set is incompressible —
false for CR-2144's 747, whose derivatives are published *at* M 0.80.

Declaring M_ref = 0.80 on `boeing747` is the obvious move and it was tried. **The
measurement rejected it.**

> **Prandtl–Glauert's `1/β` is a two-dimensional SECTION result, and `CLa` here is a
> FINITE-WING coefficient.** For a finite wing the compressibility and downwash corrections
> interact and the true variation is much gentler; the standard 3D form is
> `CLα(M) = 2πAR / (2 + √(AR²β²(1 + tan²Λ/β²) + 4))`. Applying the 2D form to a 3D
> coefficient over-corrects away from M_ref, and for this airframe that is not subtle: it
> implies an **incompressible slope of 4.9441·√(1−0.8²) = 2.97 /rad** for an AR 7.0 wing
> whose real low-speed value is about 4.5–5.0.

Measured cost of enabling it anyway: it broke the **exact V² scaling** of the aerodynamic
force, moved the short-period damping attribution by 1.1e-4, and moved **every Fig. 8
vortex number** — for a correction whose own premise fails at the conditions it was
reaching, on an entry that declares no Mach band to confine it. §7 had already recorded
`CLα(M)` applied to this airframe making its phugoid *worse*, 17.8% → 19.4%; this explains
why.

**So the seam is built, exercised and shut.** Two tests hold it that way: one asserts no
entry declares a reference Mach and names the 3D form needed before one should, the other
asserts the mechanism is real (exactly neutral undeclared, active when declared) so a shut
seam cannot rot. Turning it on properly needs a quarter-chord sweep angle no source this
project holds supplies — which is the same shape of answer as session 20's `mach_ram`:
**the honest move was not to fit the solver to a test result.**

Applied to the whole longitudinal lift-slope family rather than `CLa` alone, since a partial
correction is what made the phugoid worse. Lateral derivatives are not scaled: they mix
section lift slope with dihedral, fin geometry and sidewash in proportions no source
supplies.

**3. `wind.lamb_oseen_wind` — a smooth core, matched to Parks' own two numbers.** Both
matching constants are *solved*, not transcribed, and `test_wind` re-derives them with
`brentq`: `rc/r0 = 0.892135` puts the peak at r₀, and `Γ/2π = 1.397953·V₀r₀` makes the peak
value V₀. So Lamb–Oseen and Rankine agree on core radius and peak tangential velocity and
differ **only in shape**, which is what makes substituting one for the other a controlled
experiment.

| Refinement through one core, 747 at cruise | Fitted behaviour |
|---|---|
| Rankine (C⁰ but not C¹ at r₀) | **non-monotone** — no order exists to fit |
| Lamb–Oseen (smooth) | **monotone convergence** |

**It is not the default, deliberately.** The frozen JSBSim vortex reference was generated
with the Rankine form and `test_jsbsim_vortex` reconciles the two implementations to 1e−9,
so switching the default would invalidate that reference rather than improve it.

**4. The gradient arm now has the independent cross-check §4 said it lacked** — run where
the data supports it. `boeing747` carries a real `CLq`, so its tail arm (4.0241 c̄ =
109.9 ft, already validated) lets `sampled_rates` (a secant across the airframe) be
compared against `gust_rates` (the analytic tangent the comparison used).

| Profile | Worst tangent-vs-secant disagreement over ±2.5 r₀ |
|---|---|
| Rankine | **2.000 V₀/r₀** |
| Lamb–Oseen | **0.260 V₀/r₀** — 7.7× smaller |

**The instrument validates itself:** Rankine's worst case comes out at 2.000 V₀/r₀, which
is *exactly* the one-sided derivative jump `ASSUMPTIONS` E2 records at the core edge,
reached by a completely different route. Inside the core the two agree to 1e−9, because the
field is solid-body and therefore linear — the negative control.

**What it bounds:** the 8–13% gradient contribution is confirmed real, and its precision is
now bounded rather than unknown. Even on the smooth profile the two estimators differ by 26%
of the characteristic gradient, peaking near the velocity maximum where the tangent passes
through zero while the airframe still spans a varying field. **Read the gradient result as
an ordering, not to two figures.**

**5. The RK4 wind hold can now be switched off per model, and it buys back three orders.**
`step`/`rollout` take `stage_sampled`, and `wind.field_model` marks its output as safe to
re-evaluate mid-step. Measured on the same field and steps, changing nothing else:

| | Fitted order |
|---|---|
| Wind held across all four stages (default) | **1.0534** |
| Re-evaluated per stage | **4.0552** |

which is the 4.05 `ASSUMPTIONS` E4 recorded from the falsification probe that first
identified the cost. **The hold stays the default**: re-sampling is valid only for a model
that is a pure function of position — re-drawing a stochastic field per stage would make the
realisation depend on step size, and a convergence study would then measure the noise rather
than the integrator. Only `field_model` output carries the mark.

> **Two items from the session's request were declined, with reasons.** *Copying JSBSim's
> integrator* would replace 4th-order RK4 with 2nd-order Adams–Bashforth, delete the tier-0
> order suite, break the `lax.scan` single-step contract the `vmap` ensemble rests on, and
> not touch layer 4's dominant error — which is the 0.05 s replay **hold**, measured
> first-order at 1.01–1.04, in the harness rather than the integrator. Item 5 above attacks
> the same accuracy question from the correct end. *WGS-84 + Earth rotation* is deferred to
> its own session: NED stops being inertial, every Coriolis and centrifugal term changes,
> and it would not improve the JSBSim comparison, which deliberately sets JSBSim to
> constant-g and quantifies the Coriolis floor by running lat 0° against 47°.
>
> **There is a general point here worth keeping.** The comparison's premise is that *"a
> disagreement is a defect in one of the two implementations"* — and that holds only while
> the implementations are independent. Sharing the aero *data* is correct, because it
> isolates the solver. Sharing the gravity model, the integrator and the Mach scheduling
> would remove the remaining independent axes, at which point 0.04% agreement on the short
> period stops meaning "two independently-written solvers agree" and starts meaning "the
> transcription was faithful."

### Three model changes, and what each moved (session 23)

Two of these repair defects this document had recorded and worked around for twenty-two
sessions; the third adds a control-system term the real aeroplane never flies without.
**Every altitude-dependent number in this ledger moved.** The rows below are what to read
instead of chasing individual supersessions.

**1. The ISA now converts geometric altitude to geopotential.** `atmosphere.py` documented
"altitude is treated as geopotential" and used the geopotential formulas, while every
caller passed geometric altitude — `dynamics.derivatives` passes `-state.pos_ned[2]` — and
nothing converted it. Not an approximation; two different quantities.

| Check | Before | After |
|---|---|---|
| Density error vs JSBSim at a **nominal** 30,000 ft | 0.159% low | **+0.000479%** |
| `matched_altitude` shift needed to close it | 43.22 ft | **0.13 ft** |
| Density at 40,000 ft | 0.368% low | matched |
| ISA table check (`test_atmosphere`) | passed a geopotential argument | now enters each row at the **geometric** height whose geopotential is the tabulated one |

The workaround this retires is instructive: `matched_altitude` used to be **frozen in the
reference XML**, which froze a dependency on a model this project owns. When the atmosphere
was corrected the stored shift became an *error of the same size and sign* as the bias it
was introduced to remove. It is now recomputed at load time from the frozen JSBSim density —
which **is** a measurement — by `jsbsim_ref._match_density`. The mechanism is kept rather
than deleted because the residual 4.8e-6 between the two codes' ISA constants is real.

**2. Gravity varies with height:** `dynamics.gravity(z) = g₀(R/(R+z))²`, R = 6,371,000 m.
§5 recorded session 12 measuring this and deciding **not** to model it. That decision is
reversed, not the measurement.

| Quantity | Movement | Against |
|---|---|---|
| 747 phugoid ωn | **−0.3984%** | g's −0.3817% — Lanchester's `√2g/u₀` predicts **1:1** |
| 747 phugoid ζ vs CR-2144 Table IX-5 | +14.4% → **+13.2%** | an **unforced improvement**; the change was not aimed at IX-5 |
| 747-approach (sea level) α, δe, throttle | **bit-identical** | the control — g(0) = g₀ exactly |
| Cherokee trim α | −0.0035° absolute | a −28% *relative* move on a near-zero trim; the percentage is an artefact |
| Fig. 8 vortex point Δn | −1.261 → **−1.265 g** (+0.380%) | g(40,000 ft)/G₀ to three figures |

`atmosphere.G0` **stays constant inside the barometric integration and must** — geopotential
altitude is defined as the coordinate that absorbs g's variation, so ISA already accounts
for it and substituting g(h) there would double-count.

> **This changes what `load_factor` means, and the change is deliberate.** Specific force is
> divided by **G0**, not g(h), because "g units" are standard gravity — an accelerometer is
> calibrated in them, so is JSBSim's `accelerations/Nz`, so is a DFDR trace. So trimmed
> level flight at altitude now reads `cos(θ₀)·g(h)/G0` = **0.9930** at 40,000 ft rather than
> cos(θ₀) = 0.9968. That is what a real accelerometer reads there. Dividing by g(h) would
> restore the tidier invariant and silently redefine the unit, making n_z at altitude
> incomparable with n_z at sea level or with either external dataset. `checks.trimmed_start`
> and `test_load_factor_in_trimmed_level_flight_is_cos_theta_not_one` carry the corrected
> form; the old one failed by 3.8e-3 g, nearly four times its own tolerance.

**3. The 737 carries JSBSim's yaw damper, in the plant.** `yaw_damper_gain = 0.35` s of
rudder per rad/s of yaw rate, above M 0.11 — the gain measured off JSBSim's FCS in session
17. Zero for every other aircraft, so the term adds an exact zero and they are bit-for-bit
unmoved.

| Layer 3, cruise | JSBSim | Analytic fold (sessions 17–22) | **Damper in the plant** |
|---|---|---|---|
| Dutch roll ωn | 2.11920 | 2.11966 (0.02%) | **2.11929 (0.004%)** |
| Dutch roll ζ | 0.34410 | 0.34402 (0.03%) | **0.34410 (exact to 5 dp)** |

**The plant form is closer than the fold, and subsumes all three of its corrections.**
`δr = 0.35·r` reaches CY, Cl and Cn through the real `CYdr`, `Cldr` and `Cndr`, which is
algebraically what adding `C*dr·0.35·2V/b` to `C*r` did by hand — except the fold linearised
about one airspeed and the plant term uses the aircraft's actual yaw rate every step.

> **The cost, stated plainly: a damped entry is no longer a bare airframe.** Layer 1's
> yaw-rate sweep now moves the rudder, so it no longer isolates `Cnr`. **Layer 4 switches
> the damper off** (`_replay` zeroes the gain) and must: that layer's whole design is that
> surface positions are *prescribed to both engines*, with JSBSim's damper pre-compensated
> out, so leaving atisim's on would fly a rudder JSBSim did not — worth **1.83 m/s** of
> spurious `v` divergence on the approach rudder kick against a 0.013 m/s Earth-rotation
> floor.

**Superseded by the above:** the Fig. 8 point in §4's strip-loads table (Δθ 2.160°, Δn
−1.261 g → **2.163° / −1.265 g**), the lee-wave thrust envelope (+0.023 → **+0.0241**, since
weight is now m·g(h)), and every §4 mode figure for an aircraft above sea level.

### Comparison preconditions (session 21)

| Check | Measured | Why it has to be checked |
|---|---|---|
| The two vortex fields, written independently in numpy and JAX | **< 1e-9 m/s** at every sample | a bug on both sides of a comparison is invisible to it |
| JSBSim `accelerations/Nz` vs atisim `dynamics.load_factor`, same still-air state | **6.6e-5 to 2.1e-4 g** | otherwise the Δn column compares two different quantities |
| JSBSim elevator over the whole encounter | **0.0000° of movement** | a moving surface would be commanding the pitch difference |
| Morton flown at both sources' radii, which agree | **identical to the last digit** | establishes Hannibal's spread as the radius, not the harness |
| Density match, atisim geometric vs JSBSim geopotential | residual **0.0**, shift −69.19 ft | qbar ∝ ρ, so a nominal match biases every force one way |


### Integrator and rigid body

| Check | Measured | Tolerance |
|---|---|---|
| Angular-momentum magnitude drift, 600 s / 60,000 steps | 5.7e-13 | 1e-11 |
| Angular-momentum direction drift | 1.5e-6 deg | 3e-5 deg |
| Rotational KE drift | 9.3e-13 | 2e-11 |
| Quaternion norm, 1e5 steps | holds | atol 1e-12 |
| Coordinated turn vs `g·tanφ/V`, 25.4° bank | 1.12% | 3% |
| Zero-strength wind vs still air, 2000 steps | **bit-identical**, max diff 0.0 | `np.array_equal` |

### Distributed-airframe sampling (session 13)

| Check | Measured | Tolerance |
|---|---|---|
| Uniform field, sampled rates | **exactly 0** | `np.array_equal` |
| Linear field, sampled vs analytic gradient | agrees | rtol 1e-9 |
| Inside the Parks core, secant vs tangent | agrees | rel 1e-9 |
| At the core edge, one-sided derivatives | differ by **2·`V₀/r₀`**, opposite signs | rel 1e-6 |
| Curvature correction, 1.25 r₀ / 2.0 r₀ | **0.109 / 0.025** `V₀/r₀` | reported |
| Station-count convergence, 9 → 18 | below 0.1% | 1e-3 |
| Derived tail arm, 747 | **4.0241 c̄ = 109.90 ft** | inside the real 100–110 ft |
| Plausibility gate | passes both 747 sets, rejects both light-aircraft sets | exact |
| Rigid roll rate through the strip integral vs CR-2144 `Clp` | agrees | rel 1e-3 |
| Rectangular-wing strip integral vs Stengel eq. 3.4-40 | agrees | rel 1e-6 |
| Elliptic closed form `∫y²c dy = c₀b³π/64` | agrees | rel 1e-5 |
| Linear gust gradient, strip vs equivalent rate | agrees | rel 1e-6 |
| Loading-shape spread, elliptic vs tapered | **2.6%** | reported |
| …including uniform as a bracket | **49.7%** | reported |
| Existing wind path, before and after | **reproducible, and still the tangent** | `np.array_equal` |

Suite: **358 passed, 1 skipped**, up from 342 with nothing broken.

### Strip loads in the 6-DOF (session 14)

| Check | Measured | Tolerance |
|---|---|---|
| Zero increment vs omitting it, `derivatives` | **bit-identical** | `np.array_equal` |
| Omitting `load_model`, 100-step rollout | **bit-identical** | `np.array_equal` |
| Strip increment vs `wind.strip_roll_moment` | agrees | rel 1e-9 |
| Strip increment under a 50 m/s tailwind | uses air-relative speed, not ground speed | strict inequality |
| Tail-arm gate at construction | raises for both light aircraft | `pytest.raises` |
| Parks core traverse, point vs strip position | **0.000000 m** (`Cl` = 1.8e-19) | reported |
| Cubic spanwise gust, point vs strip position | **0.187463 m** (`Cl` = 1.335e-05) | must be > 0 |
| Fig. 8 vortex point, point vs strip | **unchanged: d(θ) 2.160°, d(n) −1.261 g** | reported |
| Fig. 8 updraft point, point vs strip | **unchanged: d(θ) 4.366°, d(n) −0.114 g** | reported |
| Ordering vortex < updraft < manoeuvre | **HOLDS on both paths** | exact |
| Rigid-rotation structure, inside the Parks core | **q-pair +1.0000**, p-pair n/a | reported |
| …outside the core | **q-pair −1.0000**, p-pair n/a | reported |
| `CL` increment through `load_factor` | raises it | strict inequality |
| `Cl` increment through `load_factor` | **exactly no effect** — it enters the moment, not the force | exact |
| Zero increment vs omitting it, `specific_force` | **bit-identical** | `np.array_equal` |
| Load model reaching the measured `n_z` in `_measure` | changes it | `not allclose` |
| Omitting the load model in `fly` | **bit-identical** `n_z` | `np.array_equal` |

Suite: **377 passed, 1 skipped**, up from 358 with nothing broken.

> **Precision note added by the remediation pass.** The Δθ figures above are quoted to
> four significant figures and are only good to three. They are unchanged as
> measurements — nothing moved them — but the once-per-step wind hold costs the in-core
> Δθ **0.82%** at the published dt (§4, "The ORDER half of the wind seam", and
> `ASSUMPTIONS.md` E4, whose bound this pass corrected by ~80×). Read them as 2.24° and
> 4.37°. The comparison this table is making — point path against strip path — is
> unaffected, because both paths carry the same hold.

**Read the two zeros together, because they have one cause.** The strip path
changes neither turbulence encounter, and that is a property of the two fields
rather than a defect in the seam. The Parks vortex has no east variation and
the updraft column is axisymmetric about an axis the aircraft flies straight
through, so in both cases every strip on the span sees the same vertical gust
and the antisymmetric roll integral cancels. The diagnostic reports the same
fact from the other side: `p-pair n/a` means `∂w/∂y` is identically zero.

**What the `q-pair` measures, stated carefully.** `gust_rates` reads three
entries out of the 3×3 body-frame velocity-gradient tensor: `p_g = +∂w/∂y`,
`q_g = −∂w/∂x`, `r_g = +∂v/∂x`. A rigid rotation of the air mass has a
*skew-symmetric* gradient tensor, which has exactly three free parameters — so
when the tensor is skew, those three numbers capture the field's entire
first-order structure with nothing left over. The q-pair reports
`−(∂u/∂z)/(∂w/∂x)`, which is `+1` iff the (x,z) block is skew.

Inside the core it is **+1.0000**: Rankine solid-body rotation, tensor skew,
three numbers sufficient. Outside it is **−1.0000**: irrotational, in Parks'
own terms — the tensor is *symmetric*, pure **strain**, zero vorticity, and the
model has no channel for strain at all.

The implication that holds is one-directional: q-pair `= +1` ⇒ rigid rotation
⇒ the field is linear in position ⇒ the point treatment is exact. That is
independently why the curvature correction is exactly `0.0000` at 0.50 r₀ and
0.99 r₀ in the session-13 table. **The converse does not hold**, and a strip
*pitch* integral would not address the `−1` region: a pitch integral gives each
longitudinal station the gust at its own `x` instead of fitting one slope, so
it fixes curvature in `w(x)` and never reads `∂u/∂z` at all. Two different
failures. That irrotational and curved coincide outside the core is a property
of the Rankine profile, not a theorem.

The 0.187 m cubic case is the positive control. It is the only evidence here
that the seam reaches the equations of motion at all; without it every number
in this table would be satisfied by a `load_model` that was computed and
discarded.

### The remediation repairs (session 16)

Seven code changes, and the measurement that shows each was surgical. Every one is a
repair to a defect the audit found; none is a change to the model's physics or its data.
**No aerodynamic derivative was added, removed or altered.**

| Check | Measured | Tolerance |
|---|---|---|
| **Whole-model regression: every mode and trim of all four aircraft, pre- vs post-remediation** | **42 of 44 scalars bit-identical**; the 2 that moved are the Cherokee's `roll_tau`/`spiral_tau`, the same two values swapped into the correct slots | `==` on the hex repr |
| 747 phugoid ωn / ζ, short-period ωn / ζ across the whole pass | **bit-identical** (0.055319, 0.055956, 0.950773, 0.342526) | `==` |
| `lateral_modes`, three stable-spiral aircraft, signed sort vs `key=abs` | **bit-identical** | `==` |
| `lateral_modes`, Cherokee (unstable spiral) | roll **0.3595 s**, spiral **−51.59 s** — was returning them swapped | rel 0.02 |
| Aerodynamic force **and** moment at exactly V = 0, all four aircraft | **exactly 0.0** — was 1.44 N (Cessna) to 170.73 N (747-approach) | `== 0.0` |
| `load_factor` in free fall, aerodynamics **live** | **−0.0** exactly, matching the aero-zeroed control — was +7.0e-6 to +4.0e-4 | `== 0.0` |
| Force/moment/coefficients at ‖v‖ ≥ 1 m/s, floor confined to the divisions | **81 of 81 sampled states bit-identical** | `==` on the hex repr |
| `is_physical`, four registry aircraft at their own cruise conditions | all **pass** (the positive control) | `bool` |
| `is_physical`, pinned 747 root at V = 471.8 m/s (throttle 567) | now **rejected** | `bool` |
| `along_track_shear` vs `dU_x/dt` differentiated along a prescribed circular track | **0 to 5.6e-17** | abs 1e-15 |
| `along_track_shear` with `accel_ned = 0` vs the straight-track expression it replaced | **bit-identical** — the reduction to Proctor Eq. (4) | `==` |
| ψ̇ along both lee-wave legs and the microburst penetration, 77,036 samples | **identically 0.0**, so the new term contributes nothing to any published run | `== 0.0` |
| `scripts/leewave.py` and `scripts/microburst.py` printed output across the change | **byte-identical** | `diff` |
| Heading-rotation term, standard-rate turn one core radius above a Parks core | **ΔF = 0.1423** (single-core closed form `v₀ψ̇/g` = 0.1383) | abs 5e-4 |
| `superpose()` with no fields | returns the zero field; superposing it is **bit-identical** to not superposing | `np.array_equal` |
| Angular-momentum drift, re-measured for the comment that was wrong by a decade | **5.695769e-13** | 1e-11 |

**The whole-model regression row is the one that matters** and it is the reason the mode
table above is untouched. It was taken by extracting the tree at the commit before the
remediation, running the same probe against both, and comparing hex representations —
not by re-reading the numbers and finding them similar.

**What was deliberately NOT repaired** is recorded in `docs/ASSUMPTIONS.md`, not here,
because a bounded flaw left in place with its size stated is an assumption rather than a
measurement: C9 (the lift-tilt energy seam), F5 (the strip quadrature at 9 stations), E6,
E8, E9, F7, C10 and C11. `AUDIT.md` §1 carries the per-finding status.

**One repair was made, measured and reverted**: `_B747_G`. See `ASSUMPTIONS.md` B5 — it
moved 19 quantities on the 747 by ≤ 4.1e-6 and broke two bit-exact guards, and the flaw
is smaller than the fix.

### 747 modes vs CR-2144

| Mode | Model | Reference | Error |
|---|---|---|---|
| Dutch roll ωn | 0.943 | 0.947 rad/s | 0.4% |
| Dutch roll ζ | 0.0361 | 0.0349 | 3.4% |
| Roll τ | 1.795 s | 1.779 s | 0.9% |
| Spiral τ | 138.0 s | 137.0 s | 0.8% |
| Phugoid ωn (as shipped until session 30) | 0.0553 | 0.0673 rad/s | 17.8% — attributed, §5 |
| Phugoid ζ (as shipped until session 30) | 0.0560 | 0.0489 | 14.4% — attributed, §5 |
| Short-period ωn (as shipped until session 30) | 0.9508 | 0.964 rad/s | 1.4% — attributed, §5 |
| Short-period ζ (as shipped until session 30) | 0.3425 | 0.387 | 11.5% — attributed, §5 |
| Phugoid ωn (speed derivatives declared, session 30) | 0.0700 | 0.0673 rad/s | +4.0% — the residual was the thrust line, §4 |
| Phugoid ζ (speed derivatives declared, session 30) | 0.0506 | 0.0489 | +3.5% |
| Phugoid ωn (speed derivatives + CR-2144's 10 ft thrust line, session 30) | 0.06727 | 0.0673 rad/s | −0.05% — replaced, the next rows |
| Phugoid ζ (the same) | 0.04945 | 0.0489 | +1.1% |
| **Phugoid ωn (as shipped: speed derivatives + 5.70 ft thrust line, session 30)** | **0.06844** | 0.0673 rad/s | **+1.7%** — CR-114494's revised arm, §4 |
| **Phugoid ζ (as shipped, session 30)** | **0.05028** | 0.0489 | **+2.8%** — ±5 points from the hand reading, §4 |
| **Short-period ωn (as shipped, session 30)** | **0.9513** | 0.964 rad/s | −1.3% — α̇ family still omitted, §5 |
| **Short-period ζ (as shipped, session 30)** | **0.3426** | 0.387 | −11.5% — α̇ family still omitted, §5 |
| Phugoid / short period (augmented model) | — | — | ~1% |

**All four published longitudinal factors are now listed.** CR-2144 Table IX-5's
denominator publishes four for FC9 and this table compared two of them until the
remediation pass — and the two it omitted were the two that look worse, phugoid ζ at
+14.4% and short-period ωn at −1.4%. Reference values are the denominator block on
printed **p.231**, `Z(DET)1 = .0489`, `W(DET)1 = .0673`, `Z(DET)2 = .387`,
`W(DET)2 = .964`, read at 600 dpi and re-read independently when the rows were added.

**THE MODEL DID NOT CHANGE — the comparison did.** Every "as shipped" figure above is
the engine's own value, unchanged: the remediation pass added two ROWS, not two
derivatives. `Aircraft` still carries no speed derivative (`Xu, Zu, Mu`) and no α̇
derivative (`Zẇ, Mẇ`), and §5 still declares both families out of scope. Read the row
labels literally.

**Session 30 changes the first half of that sentence, and not the rows.** `Aircraft` now carries
the SEAM for the speed derivatives' Mach content: `mach_deriv_ref`, `CL_M`, `CD_M` and `Cm_M`.
**No registry entry declares it**, so every "as shipped" row above is still the engine's own
value. The retest on a copy — phugoid ω_n +4.05%, ζ +4.55% with the digitised set — is in "CR-2144's
speed derivatives, digitised and flown".

That is measured, not asserted. All four are **bit-identical** across the remediation —
see the whole-model regression row in "The remediation repairs" above, which extracted
the tree at the preceding commit and compared hex representations. The pass did change
code, in seven places; none of it is on the path that produces these four numbers.

The ≤1% figures below are an **attribution, computed in analysis**, and are not a state
this code can be run in. `AUDIT.md` §2.3 patches the engine's own cruise plant matrix
with those two families, taken from CR-2144 Table IX-4 FC9, one family at a time, and
**every one of the four closes to ≤1% when both are restored**: phugoid ωn +0.3%,
phugoid ζ +0.5%, short-period ωn −0.9%, short-period ζ +0.1%. The speed derivatives alone
fix the phugoid frequency and make short-period damping slightly worse; the α̇ derivatives
alone fix short-period damping and leave the phugoid frequency exactly unmoved; phugoid
damping needs both, which is why it is attributable to neither.

**Why that strengthens the position rather than weakening it.** The two rows added are
the two that look worse, so the table now shows a larger worst-case error than it did —
and it simultaneously shows that the whole cruise mode discrepancy is the two documented
omissions and nothing else. The aerodynamic data, the conversion chain, the trim solve
and the eigen-extraction are all exonerated by it. Restoring the two families in the
engine is a **feature with its own design and re-measurement**, not an error correction,
and deliberately did not happen here.

Superseded by the session-5 `Mq` fix (§6d), kept per §4's rule: short-period ζ read
**0.338 / 12.6%** and phugoid ωn **0.0554** while `Mq` was −0.330. Short-period ωn moved
0.9493 → 0.9508.

The same fix moved the encounter table below, because `Mq` is pitch damping and those runs
are open-loop pitch responses: first-core Δθ **2.20 → 2.24 deg**, updraft Δθ
**4.39 → 4.37 deg**. Superseded values recorded here; no conclusion in §5 or the Fig. 8
mechanism changes, since both are orderings rather than values.

### Air-relative sensing (session 5)

| Check | Measured | Tolerance |
|---|---|---|
| Still-air sensing unchanged by the fix | exact | atol 1e-12 |
| Airspeed in a 25 m/s headwind vs groundspeed + 25 | exact | rel 1e-6 |
| Δα from a 10 m/s updraft at 236 m/s | 2.43 deg = atan(10/236) | rel 0.02 |
| Autopilot into a 25 m/s headwind: airspeed on target | yes | ±1.5 m/s |
| …and groundspeed 25 m/s below it | yes | ±1.5 m/s |
| Attitude/rates unmoved by the wind correction | exact | atol 1e-12 |
| Bugs (a)+(b) re-introduced → their tests go red | 2 failed, 219 passed | — |

### The live flying interface (session 6)

| Check | Measured | Tolerance |
|---|---|---|
| Live loop in still air, with and without an explicit `zero_wind` | **bit-identical** | `np.array_equal` |
| Live loop through an updraft vs still air, 40 frames | 1.0 m of altitude | > 1.0 m |
| `vortex.py` unmoved by the constants move: first-core Δθ | 2.160 deg | was 2.24 |
| …updraft Δθ / peak load | 4.366 deg / −1.235 g | were 4.37 / −1.23 |
| Slip ball vs β under held rudder | **opposite signs** | product < 0 |
| Stick ramp: 10 steps in 1 frame vs in 10 frames | identical | `approx` |
| One second of held trim, all three aircraft | 0.25 × full stick | rel 0.05 |
| Suite | 256 tests, 207 s | — |

The bit-identical row is the one that matters: it is the same statement §4 already
makes about a zero-strength wind model, applied to the live loop, and it is what says
the wind hook did not perturb the default path.

### Panel frame rate (session 7)

Medians of 120 runs, Agg backend, 12-core machine, 7 Aug 2026. The "before" column is
the panel as session 6 shipped it; the "after" column is the same panel with `sense`
and `accelerometers` jitted.

| Item | Before | After |
|---|---|---|
| jitted RK4 step | 0.114 ms | — |
| `sense` | 6.56 ms eager | 0.028 ms jitted |
| `accelerometers` | 10.70 ms eager | ~0.03 ms jitted |
| blit, 14 axes / 72 artists | 28.6 ms | 30.1 ms |
| whole frame | 73.0 ms | 37.8 ms |
| achieved rate | 13.7 fps | 26.4 fps |

Session 6 shipped the re-laid-out panel **below** its own 20 fps target and did not
know it, because the target is asserted nowhere and the only measurement on record
(§10's 19.9 fps) predated the re-layout. The two sensing calls were the cost: both ran
eagerly once per frame, and together they were 17.3 ms of a 73.0 ms frame. Blitting is
now the floor — 30.1 ms of 37.8 — and it did not improve, which is the expected result
of jitting something that was never the bottleneck's neighbour.

### Vortex, updraft and manoeuvre encounters (747 at CR-2144 FC9)

| Quantity | Measured | Reference |
|---|---|---|
| Gust spacing, Parks Case 1 | 4.52 s | "about 5 s apart" |
| In-core Δθ, first core | 2.24 deg | Fig. 8 vortex ≈1.4 deg |
| In-core Δθ, second core | 4.17 deg | response builds through the array |
| Whole-run Δθ | 8.33 deg | the phugoid, **not** the encounter |
| Peak load excursion, vortex | −1.23 g | — |
| In-column Δθ, updraft (sharpness 6) | 4.37 deg | paper states 5.2 deg; Fig. 8 cluster 6.2 |
| Updraft Δθ across sharpness 2→10 | 3.63 → 5.34 deg | the declared parameter's influence |
| Air-relative vs inertial α, peak difference | 7.0 deg | — |
| corr(n_z, α) air-relative / inertial | 0.9990 / 0.5572 | — |

### The manoeuvring case (session 7)

Zero wind. Elevator pulse of one short period, **declared**; the deflection is
**bisected**, not chosen, so the sourced quantity is the load and the angle is an output.

| Quantity | Measured | Reference |
|---|---|---|
| Elevator to reach the band | 8.926 deg from trim | derived by bisection, tol 1e-5 rad |
| Load excursion, in-pulse | −1.900 g | Fig. 8 band −2.01…−1.69, **increment** reading |
| Δθ, in-pulse | 30.37 deg | Fig. 8 manoeuvring 12.0 |
| Peak \|α\| in-pulse | 10.31 deg | **marginal** — §7's 10–12 deg amber band |
| n_z at the first sample | 0.9967 | the trimmed value, i.e. the lead-in worked |
| Δθ, whole run vs in-pulse | 30.74 vs 30.37 deg | 1.2% — the window barely matters here |
| Fig. 8 pitch ordering, model | 2.24 < 4.37 < 30.37 | paper 1.4 < 6.2 < 12.0 — **ordering holds** |
| Suite | 260 tests, 104 s | was 256, 126 s on the same machine this session |
| Suite (session 8) | 270 tests, 167 s | the lee wave added 10 |
| Suite (session 9) | 284 tests, 272 s | the microburst added 11, the averaged index 3 |
| Suite (session 10) | 296 tests + 1 skipped, 193 s | the approach 747 added 12 |

Two of those rows are the result and the rest are the guard. **The ordering holds**, which
is the only claim §5 permits. **The absolute values do not agree** and are not meant to:
30.37 against 12.0 is 2.5×, in a comparison whose reference aircraft the paper never
identifies.

The whole-run/in-pulse row is worth keeping for contrast: the vortex moves 2.24 → 8.33 deg
between the two windows and the manoeuvre moves 30.37 → 30.74. A manoeuvre is bounded —
the elevator comes back — so the window rule barely bites. For a vortex the aircraft is
left ringing and it bites hard. That asymmetry is why §8's window rule had to be stated
before the third point could be computed rather than after.

### The Fig. 8 mechanism

747 short period is 6.609 s undamped. A **Hannibal** core traverse is 1.550 s, 0.235 of
that (impulsive); a 20 s updraft is 3.026 (quasi-steady). **That 12.9× separation in
non-dimensional encounter duration is what separates the two weather categories**, and it
is a rigid-body timescale effect requiring no nonlinear aerodynamics — which is why this
model reproduces the clustering while it can never reproduce the ±g asymmetry.

> **Superseded, session 7: this said 17×, and 17× is the wrong case.** It is Morton's
> ratio — r₀ = 137.16 m gives a 1.163 s traverse and 20/1.163 = 17.2 — while every run,
> figure and ledger row in this project uses Hannibal, whose r₀ = 182.88 m gives 1.550 s
> and 12.9. Nothing downstream moves: the discriminator is an ordering claim and both
> ratios are an order of magnitude. Corrected because §8's window table now states the
> durations to four figures and a reader would otherwise find them contradicting this
> paragraph.

**The manoeuvre is not a third point on this scale, and that is the point.** Its pulse is
6.609 s — 1.000 short periods, *between* the vortex and the updraft — yet it lands at
30.37° of pitch, far right of both. So duration does not order the three categories, and
timescale is not "the whole discriminator" once the third one exists. It separates the two
**weather** categories. The manoeuvre separates for a different reason: the elevator is
moving, so pitch follows the stick rather than the air. That is precisely the distinction
Wingrove & Bach's chart was drawn to make, and it is why `vortex_viz.fly` holds its
controls fixed.

### Mountain lee wave and the F-factor (session 8)

747 at CR-2144 FC9, fixed controls, three wavelengths flown. The thrust envelope is
**recomputed**, not taken on trust from session 2 — it agrees.

| Quantity | Measured | Reference |
|---|---|---|
| (T−D)/W, full throttle | **+0.0234** | session 2 recorded +0.023 — confirmed |
| (T−D)/W, idle | **−0.0657** | session 2 recorded −0.066 — confirmed |
| Peak F, north leg (w₀ 3.0 m/s) | **+0.01291** | **within** thrust authority |
| Peak F, south leg (w₀ 6.0 m/s) | **+0.02621** | **exceeds** +0.0234 — unrecoverable by thrust |
| Critical amplitude, F = full throttle | **w₀ = 5.51 m/s** | Doyle's two legs are 3.0 and 6.0 |
| Shear term, `U̇ₓ/g` | **0.0 exactly** | zero by construction — see §5 |
| Minimum airspeed, south leg | 226.9 m/s | from 235.9 — this is why F beats `w₀/V` |

Superseded, same session: the south leg first read **+0.02623**, from a run that opened on
a wave **crest** rather than a zero crossing — 6 m/s of updraft, about 1.5° of α out of
equilibrium before the first sample. Corrected to a zero-crossing start. The defect moved
the answer by 8e-5 and changed no conclusion, but it is the same shape as the too-short
vortex lead-in in §9 session 3 and is recorded rather than quietly repaired. Peak F also
drifts about 3% across six wavelengths, because with fixed controls the aircraft never
reaches a periodic steady state; the quoted figure is the run maximum.

**The result, and it is sharper than step 8 asked for.** §7 step 8 wanted "F exceeds the
measured envelope". It does — but not for both of the *same paper's two flight legs*. The
critical amplitude, 5.51 m/s, falls **between** Doyle et al.'s northern (3.0) and southern
(6.0) primary-wave amplitudes, measured on one aircraft on one day 50 km apart. So the
honest statement is not "a lee wave defeats a 747" but **"the threshold sits inside the
observed range"**, which is a much more useful thing to know and was not knowable before
the envelope and the field were in the same place.

Note the peak F exceeds the naive `w₀/V` = 0.02543: flown, it reaches 0.02623, because the
aircraft *slows* in the downdraft and F is inversely proportional to airspeed. The
encounter makes itself slightly worse, and only flying it shows that.

### Microburst penetration (session 9)

Cherokee, 50 m/s, **300 m AGL**, fixed controls, straight through the axis. Oseguera &
Bowles field at the paper's own 37 kt peak outflow. **The hazard metric is the 1 km
average F (Eq. 7), not the instantaneous value** — that is the FAA's metric and the one
this project should have been using all along.

| Quantity | Measured | Reference |
|---|---|---|
| Peak instantaneous F | +0.2326 | not the metric — see below |
| **Peak 1 km average F** | **+0.1929** | the metric that counts |
| …its shear term `U̇ₓ/g` | +0.1342 | **the lee wave's was exactly zero** |
| …its vertical term `−w/Vₐ` | +0.1456 | the two are comparable here |
| Cherokee thrust authority at 300 m | +0.0784 | **exceeded 2.5×** |
| FAA jet-transport hazard / must-alert | 0.10 / 0.13 | 1.9× hazardous — *for scale only* |
| F in real accidents | 0.2 – 0.36 | this run sits just below that band |
| Outflow strength that first beats the Cherokee | u_max = 7.73 m/s (15 kt) | far below anything called a microburst |
| Ground contact | t = 95.5 s, +383 m past the axis | entered 300 m up, never reached the far side |

**Why this one matters more than the lee wave.** The lee-wave field is purely vertical, so
`U̇ₓ` was identically zero and only half of Eq. (3) was ever exercised. A microburst has a
horizontal outflow, and here the shear term (+0.134) is the same size as the vertical one
(+0.146). The index is now tested on both its legs rather than one.

**The instantaneous/average distinction is not cosmetic.** Peak instantaneous F is +0.2326
against an averaged +0.1929 — 21% higher. The paper is blunt about why the average is the
right quantity: peaks "over small length scales... are quickly followed by negative values",
which an aircraft experiences as turbulence rather than as a trajectory loss. A 100 m spike
of F = 0.5 averages to 0.05 over a kilometre, and a test asserts exactly that.

**The Cherokee cannot survive any microburst worth the name.** F scales linearly with the
field, so its +0.0784 of authority is first exceeded at 7.73 m/s of peak outflow — well
below the 10 m/s of divergence Wilson et al. require before an outflow is even *called* a
microburst. Unlike the lee wave, where the threshold fell inside the observed range, here
it falls below the bottom of it.

### The 747 power-approach set (session 10)

CR-2144 Table IX-2, sea level, 165 KTAS, α₀ = 5.7°, 20° flaps, gear up, 1.4 Vs; mass and
inertia from Table IX-3 flight condition 2. **The table is already non-dimensional**, so
this set involves no conversion chain at all — unlike the cruise set, which is recovered
from dimensional derivatives.

| Check | Measured | Source |
|---|---|---|
| Trim residual at the tabulated condition | 6.5e-20 | — |
| Trim α | 5.62° | Table IX-3 F/C 2 states 5.70° |
| W/qS at 165 KTAS sea level | 1.1123 | Table IX-2 states **CL = 1.11** |
| CD rebuilt at α₀ | 0.102 | Table IX-2, exact by construction |
| dCD/dα rebuilt at α₀ | 0.66 | Table IX-2, exact by construction |
| CD0 / e (solved, not read) | 0.0377 / 0.877 | cruise CD0 carries ±0.003 from a chart read; this does not |
| Minimum-drag speed | 97.06 m/s | **12.2 m/s above the 84.88 m/s approach speed** |

**Two source conflicts, both recorded rather than smoothed over.**

1. **Figure IX-1 and Table IX-3 disagree on the approach inertias.** The figure's Power
   Approach block gives 13.7/30.5/43.1/0.825 ×10⁶ slug-ft²; Table IX-3 column 2 gives
   14.2/32.3/45.4/0.870 — up to **6%** larger. **Table IX-3 is used**, because it is the
   table the derivatives were computed at: its Q = 92.2 psf, VTO = 165 KTAS and
   ALPHA = 5.70° all match Table IX-2's header exactly, and the cruise set already reads
   flight condition 9 from it. For *cruise* the two sources agree exactly (18.2e6,
   970056), so the disagreement is specific to the approach configuration.
2. **The weights differ in the last two digits** the same way: Figure IX-1 rounds to
   564,000 and 636,600 lb, Table IX-3 gives 564,032 and 636,636. The existing cruise set's
   636,636 was checked against this and is **correct**, not a transcription slip.

**The approach point sits below minimum-drag speed, and that is not an error.** 1.4 Vs at
max landing weight comes out 12.2 m/s below V_md — the back side of the drag curve, which
is where an airliner on final actually is. It is also the reason a windshear encounter is
lethal on approach and merely uncomfortable at cruise. The consequence is recorded: the
autopilot's throttle-to-speed / elevator-to-altitude pairing is inverted for this entry,
so it holds trim but is not to be trusted through a large speed excursion. All microburst
work flies it **open loop**, which sidesteps the question entirely.

### Microburst on the aircraft class the thresholds were written for (session 10)

The same Oseguera & Bowles field, now flown by the 747 in power-approach configuration.

| Quantity | 747 approach | Cherokee | Note |
|---|---|---|---|
| Thrust authority (T−D)/W | **+0.2094** | +0.0784 | Proctor et al. quote ~0.15 for a 4-engine jet at max **takeoff** weight; this is max **landing** weight, hence more |
| Peak 1 km average F | **+0.2835** | +0.1929 | |
| …shear term | +0.2430 | +0.1342 | scales with airspeed, as Eq. (4) says it must |
| …vertical term | +0.1453 | +0.1456 | |
| Exceeded by | **1.4×** | 2.5× | |
| FAA hazard / must-alert | **2.8× / 2.2×** | *not applicable* | **the thresholds apply to the jet and not to the piston aircraft** |
| Real-accident band 0.2–0.36 | **inside it** | below it | |
| Ground contact | 51.5 s, +441 m | 95.5 s, +383 m | neither reaches the far side |

**This is what adding the approach set bought.** The Cherokee result could only ever be
compared against the aircraft's own `(T−D)/W`, because Proctor et al. state the FAA scale
and threshold "are yet to be determined" for piston aircraft. The 747 in power-approach
configuration *is* the class Lewis et al. studied, so the same run now carries a
certification-grade verdict as well as a physical one — and it lands **inside** the band
the paper reports for real microburst accidents.

Note the jet has **2.7× the thrust authority** of the light aircraft and is still beaten,
by 1.4×. More engine does not buy immunity; it buys a smaller multiple.

### Verification — is the arithmetic right? (session 11)

Every row here takes **no aircraft data as a reference**. §4 was almost entirely
validation before this: a measured quantity against a published one for one aircraft.
Conservation drift was the only entry of the other kind, and drift measures a symmetry
rather than an order — a scheme can conserve angular momentum to 5.7e-13 and still be
second order when it claims to be fourth.

| Check | Measured | Tolerance |
|---|---|---|
| RK4 observed order, harmonic oscillator (exact solution known) | **3.99982** | 4.00 ± 0.05 |
| RK4 observed order, real 6-DOF vs fine-step reference | **3.98913** | 4.00 ± 0.05 |
| Galilean invariance, uniform horizontal wind: quaternion and rates | exact | atol 1e-11 |
| …and position differs by exactly W·t | exact | atol 1e-6 |
| **No `−m·dW/dt` body force** (session 12): zero-aero free fall through a wind swinging to 29.46 m/s, peak \|dW/dt\| = **88.39 m/s² (9.01 g)**, vs `p₀ + v₀t + ½gt²` | **3.98e-12 m**, 300 steps | atol 1e-9 |
| …and the same experiment with a `−m·dW/dt` term injected into `step` (session 13) | **13.33 m** — falsified | must fail |
| …and one step is independent of the cached previous wind, full 747 aero | **bit-identical** | equality |
| Trim Newton convergence ratio (log-residual exponent) | > 1.6, i.e. quadratic | > 1.6 |
| Torque-free asymmetric body vs Jacobi elliptic closed form, 1500 steps | agrees | atol 1e-8 |
| …the closed form itself vs Euler's equations | 8.3e-8 | atol 1e-6 |
| `rk4_step` extraction from `step` | **bit-identical**, sha256 pinned | equality |

### The ORDER half of the wind seam (session 15)

Session 12 closed E4's **body-force** half — no spurious `−m·dW/dt` term. The
**order** half was left open, and `ASSUMPTIONS.md` §E4 said plainly why: the
order-of-accuracy test flies in still air, so it cannot see the once-per-step
wind hold. `verification.fixed_control_refinement` now takes a `wind_model`, and
the answer is **not 4**.

| Check | Measured | Tolerance |
|---|---|---|
| Observed order, still air (the control, same window) | **3.9891** | 4.00 ± 0.05 |
| Observed order, C∞ field (lee wave, 1.2 km, 25 m/s) | **1.0537** | 1.00 ± 0.10 |
| …its pairwise orders | 1.073 / 1.048 / 1.042 | reported |
| **…with the hold removed** (wind re-sampled per RK4 stage) | **4.0542** | must fail |
| Observed order across a Rankine core traverse | **non-monotone** | asserted non-monotone |
| …its errors, dt 1/16 → 1/128 | 0.435 / 0.0396 / 0.0827 / 0.0806 m | > 1e-4, i.e. off the floor |

**This is a property of the scheme, not a defect.** `integrate.step` samples the
wind once per step and holds it across the four stages. Session 2 justified that
as *"the standard treatment for Dryden and von Karman"* — **an appeal to
authority with no citation behind it**, which the remediation pass replaced with
the actual argument in `integrate.py`'s module docstring: a Dryden field is a
stochastic process drawn from a key, so re-sampling it per stage makes the
realisation depend on the step size and a convergence study would then be
measuring the noise rather than the integrator. Correct for a stochastic field,
on that reasoning rather than on a source. For a field varying in **space** it is an O(h)
perturbation of the right-hand side inside the step, so the scheme is **first
order** however good the stage weights are. The falsification is what makes that
attribution rather than assertion: re-sampling the wind at each stage restores
**4.0542**.

**What it costs the project's own results, re-measured with the right instrument.**
The figure this paragraph used to give was **0.0169 m** of h-vs-h/2 position difference
inside the first core, worth **0.0024 m/s of gust out of a ~26 m/s peak**, about 1e-4
relative. **That measures the wrong quantity.** h-vs-h/2 is the DISCRETISATION error with
the hold still in place; the SCHEME error is hold-vs-per-stage at the SAME dt. On the
number `scripts/vortex.py` actually prints — the in-core Fig-8 Δθ:

| dt | wind held | wind re-sampled per stage | cost |
|---|---|---|---|
| 0.02 | 2.1261° | 2.1628° | **+1.72%** |
| **0.01 (published)** | **2.1602°** | **2.1434°** | **−0.78%** |
| 0.005 | 2.1506° | 2.1593° | +0.40% |

Re-measured session 22 at Hannibal's 500 ft core; at 600 ft the same three rows read
2.2596/2.2230/−1.62%, 2.2400/2.2216/−0.82% and 2.2271/2.2179/−0.41%. **The magnitudes
are what carry the conclusion and they barely moved** — 1.72%, 0.78%, 0.40%, still
halving with dt. The signs now alternate, which is a property of the measurement rather
than of the scheme: the quantity is a max-minus-min over a discretely sampled trace, so
which sample lands nearest the peak flips with the step.

So the cost at the production step is **~0.8%, not ~1e-4 — about 80× the figure this
paragraph carried** — and it halves with dt, as an O(h) error must. Pinned by
`test_the_wind_hold_costs_the_headline_figure_more_than_E4_bounds_it`.

**No conclusion changes**, because §5 caps the vortex claims at orderings and puts ±25%
bands on the identified parameters, and 0.8% sits far inside both. What does change is
the precision claim: **Δθ = 2.160° is not good to four significant figures** — its last
two digits are scheme-dependent. Quote it as 2.24°.

**The Rankine row is a second, separate mechanism.** `vortex_wind` switches
branches at `r = r₀`, where §E2 records the one-sided derivatives differ by
`2·V₀/r₀` with opposite signs, so the right-hand side is C⁰ but not C¹ there.
RK4 across a kink has an error depending on where the step grid lands relative
to the crossing, so refining dt does not monotonically improve the answer. The
core test **passes with the falsification probe still in**, which is what says
the two mechanisms are independent. **Anything reporting a fitted order across a
core traverse is reporting an artefact** — a least-squares slope through that
sequence returns 0.62 and describes nothing.

**The 6-DOF error floor is round-off, and it bites earlier than expected.** The first
fitted window read **3.82** and the cause was the measurement, not the integrator. The
747 cruises at 40,000 ft, so `pos_ned` is about [944, 0, −12184] and float64 resolves it
to 2.7e-12 m. Measured pairwise orders across a wide sweep:

| dt | 1/4→1/8 | 1/8→1/16 | 1/16→1/32 | 1/32→1/64 | 1/64→1/128 | 1/128→1/256 |
|---|---|---|---|---|---|---|
| order | 3.973 | 3.993 | 4.008 | 4.167 | 3.420 | **−0.685** |

so the error bottoms out near **7e-11 m at dt = 1/128** and refining past it makes the
answer *worse*. The fitted window stops at 1/32, 203× above the floor. Anything
measuring a difference of trajectories at 40,000 ft has this ceiling.

### What constant gravity costs, and why it stays (session 12)

`ASSUMPTIONS.md` §A2 records that `G0 = 9.80665` is **+0.383% high** at the 747's cruise
altitude. Session 11 reasoned from Lanchester that this threatened every sub-0.5% claim.
This is the measurement that was made instead of the change: `atisim.dynamics.G0`
replaced by `g(h) = g₀(R/(R+h))²`, the aircraft **re-trimmed**, and all five modes
recomputed. Tolerances are the ones each mode is actually asserted to in
`test_cr2144_modes.py`.

| Mode, 747 at cruise | g = 9.80665 | g(h) = 9.76922 | Movement | Tolerance | Consumed |
|---|---|---|---|---|---|
| phugoid ωn | 0.055319 | 0.055109 | **−0.3798%** | 5% | 7.6% |
| phugoid ζ | 0.055956 | 0.055654 | −0.5385% | 10% | 5.4% |
| short period ωn | 0.950773 | 0.950775 | **+0.0002%** | 3% | 0.01% |
| short period ζ | 0.342526 | 0.342510 | −0.0046% | 5% | 0.1% |
| Dutch roll ωn | 0.943202 | 0.942458 | −0.0788% | 2% | 3.9% |
| Dutch roll ζ | 0.036085 | 0.035931 | −0.4288% | 10% | 4.3% |
| roll τ | 1.795366 | 1.794285 | −0.0602% | 5% | 1.2% |
| spiral τ | 138.0424 | 138.1187 | +0.0552% | 2% | 2.8% |
| trim α | 4.6362° | 4.6059° | −0.6535% | — | — |

**The control is the approach 747 at sea level, where every quantity moves by exactly
0.0000%** — `g(0) = g₀` identically, so the experiment is measuring altitude and nothing
else. `aircraft.py`'s two `G0` uses are deliberately not patched: they are in the Navion and
Cessna transcription paths, where the conversion must use the g the *source* used, and
neither 747 is built through them.

**Decision: `g(h)` is not modelled.** The worst movement consumes 7.6% of its tolerance,
`G0` is imported by four modules, and every result the project quotes is at one altitude
per aircraft — so a constant g is *exactly* right per run and the bias exists only for
comparisons across altitudes, which the project does not make. Revisit if that changes.
§5 carries what the measurement corrected in the reasoning.

### Coefficient sensitivity — known change, known result (session 11)

747 power approach, re-trimmed at every sample. **Every one of these is affine with a
non-zero intercept, and the intercept is the term the textbook approximation drops.**
Three of the four laws originally planned were the wrong functional form.

| Sweep | Law that holds | Slope | Intercept | Worst residual |
|---|---|---|---|---|
| CD0 → ζ_phugoid | linear in CD0 | 0.76994 | −0.0162 | **0.31%** |
| Cmα → ωn_sp² | affine in Cmα | −0.42251 | +0.27366 | **1.68%** |
| \|Clp\| → 1/τ_roll | affine in \|Clp\| | 2.06872 | +0.30240 | **1.73%** |
| Cnβ → ωn_dr² | affine in Cnβ | 2.10545 | +0.26626 | **0.56%** |

- ζ_phugoid's slope against the textbook `1/(√2·CL)` = 0.637 is **1.21×**: the *form*
  holds tightly, the *coefficient* is 21% high.
- ωn_sp² does **not** vanish at the neutral point. The intercept is `Zα·Mq/u₀`, which
  survives there. A test asserting ωn → 0 at Cmα = 0 was written first and was wrong
  physics; the model was right.
- 1/τ_roll's intercept is Ixz roll–yaw coupling — §2 already says Ixz "is not negligible
  for the 747", so the roll root is not the pure −L_p of the two-term approximation.
- ωn_dr's intercept is the Yβ/u₀ term. ωn/√Cnβ is **not** flat: 2.385 → 1.597 over 8×.

**The neutral point is exact.** Largest real root **0.00000** at Cmα = 0, −ve inside,
**+0.0475** at Cmα = +0.1. Nothing was tuned to put it there.

### The 747 approach against an independent implementation (session 11)

Caughey (§3) works CR-2144's own power-approach data and publishes every intermediate.
Comparison requires a **stability-axis rotation**: he states Θ₀ = 0, true only there,
while the model linearises in body axes where θ₀ = α₀ = 5.57° and w₀ ≠ 0. A rotation by
α₀ is a similarity transform — asserted to move every element and no eigenvalue (1e-8).

| Element | Body axes | Stability axes | Caughey Eq. (5.52) | rel |
|---|---|---|---|---|
| A[0,0] Xu | −0.00883 | **−0.02094** | −0.02120 | 1.2% |
| A[0,1] Xw | 0.10434 | **0.04632** | 0.04660 | 0.6% |
| A[1,0] Zu | −0.17049 | **−0.22851** | −0.22290 | 2.5% |
| A[0,3] −g cos Θ₀ | −32.022 | **−32.174** | −32.174 | **0.000%** |
| A[1,3] −g sin Θ₀ | −0.952 | **0.00000** | 0.0 | **exact** |

**The elements that disagree are reconstructed, not merely attributed.** Caughey's Z row
carries a factor `1/(1 − Zẇ)` with Zẇ = −0.0341 from the CLα̇ = 6.7 this model excludes:

| Reconstruction | Result | Caughey | rel |
|---|---|---|---|
| A[1,1] / (1 − Zẇ) | 0.58374 | 0.58390 | **0.03%** |
| A[1,2] / (1 − Zẇ) | 262.48 | 262.472 | **0.003%** |
| A[2,2] (raw) vs his Eq. (5.51) Mq | −0.4381 | −0.4381 | **exact** |
| A[2,2] + (u₀+Zq)·Mẇ | −0.4906 | −0.5015 | 2.2% ‡ |

‡ Mẇ is published to one significant figure (−0.0002), which bounds this independently
of anything the model does.

| Mode | Model | Caughey Eq. (5.54) | Error |
|---|---|---|---|
| Phugoid ωn | 0.1334 | 0.13391 | **0.4%** |
| Phugoid ζ | 0.01289 | 0.01329 | 3.0% |
| Short-period ωn | 0.8961 | 0.88178 | 1.6% |
| Short-period ζ | 0.5911 | 0.62546 | 5.5% |

Trim residual 1.8e-15. **Against the cruise column's 17.8% and 11.5% for the same two
modes with the same omissions, this is 45× better from changing nothing but the flight
condition** — see §5, which this corrects.

Both Lanchester approximations reproduce the *size* of their own published error:
ωn 0.163028 vs 0.13391 is 1.217× against Caughey's stated "about 20 per cent", and
ζ 0.0651 vs 0.01329 is 4.9× against his "a factor of almost 5".

### Cross-code verification against JSBSim (session 17)

> **Partly superseded by "Model fidelity from the JSBSim comparison (session 18)" below.**
> This entry was written at commit `1dcc5d4`, before four further commits changed the model.
> Its framing, its three findings and its layer-1/2 results still hold. **Superseded rows:**
> the derivative table (moments are now referred to the AERORP, so the entry carries
> 737.xml's own constants), `Cmq` (unfolded), layer 1's lateral figures, layer 3's short
> period and phugoid, and layer 4's divergences. Superseded, not deleted, per this
> document's own rule -- the numbers below were true of the code at the time.
>
> **Also superseded by "The Ixz sign, and what layer 4 was really measuring (session 19)":**
> layer 3's `3 lateral` row and every layer 4 figure, including the per-component table and
> its attributions. The 737 was flown with the wrong `Ixz` sign throughout this entry.

The first comparison against another **executing** 6-DOF implementation rather than a
published table. JSBSim 1.3.1 (build 1837, commit `3b25f25e`) is driven headless and its
737 is used as an **engine**, never as a dataset: `737.xml` says of itself that it was
built from public data "and guesses", validated only to the extent that it "seems to fly
right", and is for "educational and entertainment purposes only". Feeding two codes the
same coefficients makes the dataset's quality irrelevant — a disagreement is a defect in
one of the two implementations. Design:
`docs/design/specs/2026-08-20-jsbsim-737-verification-design.md`.

**Derivatives are recovered by finite-differencing the running engine, not read from the
XML, and the difference is not cosmetic.** JSBSim applies aero forces at the AERORP
(x = 625 in) and takes moments about the CG (x = 610.8 in):

| | 737.xml | engine | why |
|---|---|---|---|
| Cmα | −0.600 | **−1.0637** | 77%; `CLa × 0.096 c̄`. Transcribing gives 56% of the right pitch stiffness |
| Clβ | −0.090 | −0.1440 | side force 4.925 ft above the CG — closes to 0.15% |
| Cnβ | +0.260 | +0.2730 | side force 1.183 ft aft of the CG — closes to 0.17% |
| CLα | +4.3478 | +4.3478 | **exact** — no offset effect on lift |
| Cmq | −27.0 | **−43.000** | = Cmq + Cmα̇ (−16); α̇ = q here, so they fold with coefficient one |

| Layer | Result |
|---|---|
| 1 build-up | CL **5.5e-9**, CY **1.6e-14**, Cl 7.0e-6, Cn 1.7e-6 |
| 1 CD, Cm | differ; **predicted** from 737.xml's own tables to 3.6e-4, and to **2.9e-10** at the sideslip points |
| 2 trim | α 1.980° vs 1.965°, δe −0.05311 vs −0.05192 rad, thrust **+1.04%** |
| 3 short period | ωn **0.040%**, ζ **0.019%** |
| 3 phugoid | ωn 3.33%, ζ 1.13% |
| 3 lateral | Dutch roll ζ 1.79%, roll TC 3.22%, spiral TC **0.50%** — after the yaw-damper correction below |
| 4 trajectory | elevator doublet 0.410 m/s over 20 s, rudder kick 1.46 m/s — but see the correction below |

**No defect was found in AtiSim.** Every disagreement traces to a documented model
difference with a measured magnitude.

**A correction to the first reporting of layer 4.** The doublet's 0.410 m/s was recorded as being
the Earth-rotation floor "exactly". It is not, and the error was comparing two different components.
The floor is 0.403 m/s in **u**; the doublet's 0.410 is in **w**. Per component:

| case | u | v | w | max \|β\| |
|---|---|---|---|---|
| elevator doublet | 0.296 | 0.012 | **0.410** | 0.003° |
| rudder kick | **1.462** | 1.234 | 0.522 | 2.909° |
| Coriolis floor (lat 0 vs 47) | 0.403 | 0.012 | 0.067 | — |

So the doublet's **u** divergence is *below* the floor, and its **w** divergence is six times the
floor in that component — a real residual needing its own explanation, which is the α̇ fold: `Cmq`
carries `Cmq + Cmα̇`, exact only when α̇ = q, and the recorded doublet reaches
\|α̇ − q\| = 0.0122 rad/s, worth \|ΔCm\| up to 0.0015 against 0.0186 for one degree of α.

The two cases differ by 3.6× because they excite different physics, not because one is worse: the
doublet is essentially sideslip-free so every lateral difference is inert, while the kick's **v**
divergence is a transient Dutch-roll phase difference and its **u** divergence is secular sideslip
drag.

**Three findings worth keeping.**

*JSBSim's `do_linearization` is closed-loop and nothing says so.* Its FCS is inside the
exported model. Against the bare airframe the lateral comparison reads as catastrophic —
Dutch roll ζ 0.101 against 0.344, spiral **127.6 s against 16.7 s, a 664% disagreement** —
and the natural conclusion is that this project's lateral dynamics are broken. The yaw
damper feeds yaw rate to the rudder with unit gain above M 0.11 geared by 0.35 rad, adding
`ΔCnr = Cndr × 0.35 × 2V/b = −1.147` against a bare Cnr of −0.350. Folding it in gives
ζ 0.338 and spiral 16.61 s. Pitch and roll have no feedback, which is why the short period
needed no correction at all.

*AtiSim's ISA uses geometric altitude where the standard uses geopotential.* Density runs
0.159% low at 30,000 ft and 0.368% at 40,000 — a same-signed bias on every force in every
layer, since q̄ ∝ ρ. Predicted temperature errors match measured ones to four decimal places.
Neutralised for the comparison by matching on **density rather than altitude** (43.22 ft
lower, agreeing to 1e-16); the underlying defect is pre-existing and filed rather than fixed
here. ~~**Anything altitude-dependent in this ledger carries it.**~~

> **FIXED IN SESSION 23.** `atmosphere.geopotential` converts at the boundary, and atisim's
> density at a **nominal** 30,000 ft now agrees with JSBSim's to +0.000479%. The 43.22 ft
> match is down to 0.13 ft. Session 17 filing this rather than fixing it was the right call
> for a comparison — but it stood for five more sessions, and the workaround it installed
> later became an error of its own size when the defect was finally repaired.

*Scripts run from a git worktree import the wrong tree.* `atisim` is installed editable
against the main checkout, so `python scripts/foo.py` from a worktree silently runs the other
tree's code — no error, wrong answers. pytest is immune because it puts its rootdir first,
which is precisely why a green suite did not catch it.

The thrust model gained a **Mach ram term** (`+12.1%` from M 0 to M 0.8 on the CFM56, an 11%
error at cruise that would read as a drag defect). The field defaults to neutral, so the 747,
747-approach, Cherokee, Cessna and the synthetic fixture are **bit-for-bit unchanged** —
asserted, not inspected.

Report: `docs/summary/jsbsim-737-report.pdf`.

### Model fidelity from the JSBSim comparison (session 18)

Five model changes arising from session 17, and one sequence worth reading as a whole.
Design: `docs/design/specs/2026-08-20-model-fidelity-improvements-design.md`.

| Change | What it is |
|---|---|
| `aero_ref` | Body-axis CG→AERORP vector; `moment += r × F` in `aero_forces_moments`. Zero for every other aircraft, so `jnp.cross` adds an exact zero |
| `CD_beta` | Sideslip drag, **quadratic**. JSBSim's own table is linear-interpolated through zero, which makes CD ∝ \|β\| at the origin — a coarse-table artifact no symmetric airframe can produce, and the one place this work declines to follow JSBSim |
| `CD_alpha` | Profile-drag slope with incidence, linear because the reference sits away from drag's minimum in α — where β sits **at** it |
| `CLadot`/`Cmadot` | α̇ derivatives, Stengel Eq. (3.4-25)/(3.4-26). `dynamics.derivatives` now resolves the aircraft's own α̇ as well as the wind's, so the 737 carries a **bare** `Cmq` = −27.0 with `Cmadot` = −16.0 separate |
| second condition | `boeing737_approach`, recovered at 5,000 ft / M 0.40, α 3.63° against cruise's 1.97° |

**Referring moments to the AERORP lets the entry carry 737.xml's own constants.** Recovered
about the AERORP the finite difference lands on the file: `Cma` −0.599999 against −0.600,
`Clb` −0.0899998 against −0.090, `Cnb` +0.2599999 against +0.260, `Cmde` −0.849000 against
−0.849. `Cm0` comes out at −3.0e−08 against **no such term in the file** — what read as a
pitching-moment offset was entirely the AERORP arm. Referred to the CG these were −1.1309,
−0.1440, +0.2730 and depended on the fuel state.

The pitch axis needs a **least-squares fit over a crossed design**, not central differences:
setting α away from trim also sets α̇ (`dα̇/dα` = −0.529 /s), which contaminates a differenced
`Cma` by +0.067. The fit separates `Cm0`, `Cma`, `Cmq`, `Cmadot`, `Cmde` with max residual
2e-11. Their **sum** is exact and their **split** is conditioned at 1.9e8, so the split is the
softer number.

Measured after all five changes (this tree, both conditions):

| Layer | cruise | approach |
|---|---|---|
| 1 `CL` / `CY` | 5.5e-9 / 1.6e-14 | 9.6e-9 / 1.4e-14 |
| 1 `Cl` / `Cn` | **1.1e-8 / 1.0e-8** (was 7.0e-6 / 1.7e-6) | 1.4e-8 / 1.0e-8 |
| 1 `CD` / `Cm` | 8.0e-3 / 5.2e-3 raw — **predicted** from 737.xml's tables, not defects | 8.0e-3 / 9.5e-3 |
| 2 trim | α 1.9807°, δe −0.053362 rad, throttle 0.77950 | — |
| 3 short period | **ωn 0.04%, ζ 0.03%** | 0.08% / 0.08% |
| 3 phugoid | ωn 6.58%, ζ 3.44% | 3.4% |
| 4 doublet | 0.5584 m/s (u 0.507, v 0.012, w 0.558) | — |
| 4 rudder kick | 1.5663 m/s (u 1.566, v 1.245, w 0.558) | — |

**The short period went 0.04% → 3.95% → 1.30% → 0.04%, and only the last is honest.** The
first came from an α̇-contaminated `Cma` of −1.0637 standing in for a coupling the model did
not have — two errors cancelling. AERORP referencing fixed the coefficient and left the
missing term exposed (3.95%); resolving the aircraft's own α̇ supplied it (1.30%); `CD_alpha`
closed the rest (0.04%, now at **both** conditions). The first and last are the same number
and mean opposite things.

**This is the argument for AERORP referencing, made by measurement.** Referring moments to
the AERORP makes the pitching moment inherit the force error through `r × F` instead of
absorbing it into a fitted `Cma`. A drag slope of 0.1267 against JSBSim's 0.2113 could then
no longer hide, and fixing it moved the short period by 1.3%. The CG-referenced model would
have shown nothing — it had a coefficient free to absorb exactly that error.

**Layer 4 got worse, and that is not a contradiction.** The doublet moved 0.483 → 0.558 m/s.
Trajectory divergence is set by total drag along the path; the slope at one point and the
integral over the path are independently adjustable, so improving the slope does not have to
improve the integral. Both cases remain inside their derived tolerances (0.6 and 2.0 m/s).

**Still open, and named rather than absorbed:** the phugoid at 6.58% — attributed here to "a
slow drag-and-thrust energy exchange against a thrust model still linear in throttle", which
is **half right and is corrected in session 19**: thrust is the *damping* story and has
nothing to do with the *frequency* error, which is the larger of the two and is a missing
Mach dependence of the pitching moment; `CDde` (JSBSim's `0.059·|δe|`, frozen into `CD0` at
the trim elevator, so moving the elevator changes no drag here); Mach scheduling of `Cmde`
and `Clda`, which is why one aeroplane needs two registry entries; banked trim, where layer
2's turn case is recorded but not compared; and `wave_drag`, which the comparison does not
test at all because both engines give exactly zero at M 0.78.

### The Ixz sign, and what layer 4 was really measuring (session 19)

Two findings from an outside evaluation of the session 17–18 comparison. Both are cases of
the same thing: a number the comparison reported as a model difference that was not one.

**AtiSim's 737 was flown with the wrong `Ixz` sign, through the whole comparison.** JSBSim
reports `inertia/ixz-slugs_ft2` = +19109.13, and the generator negated it on the reasoning
that 737.xml carries `negated_crossproduct_inertia="true"`. That property is already the
**tensor element**; negating it again fed the two engines different airframes.

The design named this exact risk — "`Ixz` sign needs care, not assumption… establish which
convention the reported +19109.1 is in and **assert it**, rather than pick one" — and the
risk table recorded it as mitigated. It was not. The only guard,
`test_737_mass_and_inertia_match_the_engine`, compares AtiSim's tensor against the
reference XML's, and both descend from one line of `gen_jsbsim_reference.py`. Flipping the
sign fails **that test alone**; every physics layer stays green.

*Settled from the engine's own behaviour instead.* 737.xml defines neither `Cnp` nor `CYp`,
so roll rate makes no yaw moment and no side force — which leaves JSBSim's own
`∂ṙ/∂p` as pure inertia coupling, and makes `L_p` the same about the CG and the AERORP:

| | cruise |
|---|---|
| JSBSim `A[ṙ, p]` | **+1.180789e-02** |
| as shipped | **−1.180789e-02** |
| sign corrected | **+1.180789e-02** |
| `A[ṗ, p]`, sign-independent control | −1.227332e+00 both ways, matching JSBSim exactly |

Right in magnitude to seven digits and wrong in sign, at **both** recovery conditions. The
p column is also immune to the yaw damper, which feeds r. Now asserted in
`test_inertia_cross_product_sign_matches_the_engines_own_coupling`.

**Supersedes session 17's `3 lateral` row and session 18's lateral figures.** With the sign
right, the lateral comparison is as sharp as the longitudinal one:

| | JSBSim | as shipped | corrected |
|---|---|---|---|
| Dutch roll ωn | 2.11920 | 2.08487 (1.62%) | **2.11966 (0.02%)** |
| Dutch roll ζ | 0.34410 | 0.33803 (1.76%) | **0.34402 (0.03%)** |
| roll TC | 0.82845 | 0.80179 (3.22%) | **0.82847 (0.003%)** |
| spiral TC | 16.69112 | 16.64702 (0.26%) | 16.65300 (0.23%) |

Layer 3's lateral tolerances go from a blanket 5% to **1e-3**, with 5e-3 for the spiral —
the one mode where the two reductions are genuinely different problems. A 5% tolerance is
how a 1.6% defect survives.

**Layer 4 was partly measuring its own replay.** The reference is sampled at 0.05 s and
replayed zero-order-hold while JSBSim ran at 1/120 s. That this mattered was known — 0.25 s
sampling put the rudder kick at 5.43 m/s — but the rate was then raised until the number
looked acceptable rather than until the two parts were separated. Decimating the reference
separates them, because decimation coarsens the hold and changes nothing else. Measured
order on the most sensitive component: **1.04, 1.01** at cruise and **1.02, 1.02** at
approach — first order, as a hold must be. Asserting 1.8–2.2 instead fails all four.

Extrapolating the hold to Δt → 0 with `2·f(h) − f(2h)`:

| | u | v | w |
|---|---|---|---|
| cruise doublet | 0.507 → 0.507 | 0.012 | 0.558 → **0.172** |
| cruise kick | 1.556 → 1.561 | 0.933 → **−0.125** | 0.573 → 0.530 |
| approach doublet | 0.437 → 0.435 | 0.008 | 0.261 → **0.092** |
| approach kick | 0.818 → 0.820 | 0.345 → **−0.057** | 0.252 → 0.253 |

**Withdrawn:** the kick's v divergence read as "a transient Dutch-roll phase difference",
and the doublet's w read as the α̇ fold. The v channel and the angular rates extrapolate to
zero or past it — they are the replay, and underneath them the two engines agree on the
lateral channel to within the replay's own resolution. The kick's cruise v had also already
fallen from 1.245 to 0.933 on the `Ixz` correction alone, so what was being reported as
Dutch-roll physics was a wrong inertia term plus a sampling artifact.

**What is real.** `u`, which does not move with the interval and is still growing at
t = 20 s — the secular drag-and-thrust difference, part of it the 0.409 m/s Earth-rotation
floor, which sits in `u` too. And `w`, which peaks with the sideslip excursion (−0.573 m/s
at t = 2.66 s against a β peak of 2.9°) and decays to a third by t = 20. Divided by
airspeed, `w` is nearly the **same angle** at both conditions — doublet 0.0416° / 0.0395°,
kick 0.128° / 0.108° — across a 1.77× change in speed and 6× in altitude, which is the
signature of a coefficient-level difference rather than anything that accumulates. **Named,
not explained:** no term has been identified that predicts it.

**Layer 4 now runs at both recovery conditions**, which layers 1 and 2 already did. Its
tolerances are labelled **backstops** — allowances, not predictions — and the predictive
statements live in the two tests that do the extrapolation. Fed the cruise entry at the
approach condition the new cells read 9.7 and 10.0 m/s against 0.55 and 1.00, so they
constrain something.

**The validity guard the recovery design declined is now built.** `_boeing_737`'s docstring
says flying it at 5,000 ft and 200 kt "produces numbers that are wrong without anything
failing, warning or logging", and the design chose documentation over a runtime guard —
"Mitigation taken: documentation only, deliberately". After a documented-but-unasserted
`Ixz` convention turned out to be wrong for a whole comparison, that trade no longer holds.

`Aircraft` gains `valid_mach` and `valid_altitude`, both `[lo, hi]` and both defaulting to
`lo == hi`, meaning **no band declared** — so every CR-2144 and Nelson entry is untouched,
and asserted to be: `test_the_recovery_band_reaches_no_force` checks all six coefficients
are bit-for-bit unmoved by the fields. `checks.recovery_band` gates on them, measuring the
excursion in **band widths** so Mach and altitude are one number:

| run | verdict |
|---|---|
| `boeing737` at its recovery point | gate, **pass**, 0.000 |
| `boeing737` at 5,000 ft / 200 kt | gate, **FAIL, 2.00 band widths** |
| `boeing737_approach` at its point | gate, pass, and says the Mach axis was not checked |
| `boeing747` | **report**, not a pass — no band declared, so nothing was checked |

The bands are the entry's own fit ranges, not a judgement about where it probably still
works: `thrust_fit` samples altitude at the recovery point ±5,000 ft and Mach at 0.60–0.95.
An aircraft that declares no band gets `report`, never a green tick — the same reasoning
`Check.kind` already applies to tripwires. It is separate from `alpha_band`, which asks
about `aero.py`'s linear range and applies to every aircraft equally; a 737 at 5,000 ft and
200 kt sits comfortably inside the alpha band and is still nonsense.

**A gap the guard exposed while it was being written.** `gen_jsbsim_reference.thrust_fit`
brackets its altitude samples around the condition — the comment there records learning that
lesson — but its **Mach samples are hard-coded at 0.60–0.95 and are not rebound per
condition**. The approach entry flies at **M 0.40**, below its own ram fit, so its
`mach_ram` of 0.3346 is an extrapolation and the 0.28% residual recorded beside it describes
M 0.60–0.95 rather than the condition in use. Declaring `[0.60, 0.95]` for that entry would
condemn it at its own recovery point, and inventing a lower bound would be inventing; so its
Mach half is left undeclared and the check says so out loud. The fix is one line in the
generator plus a regeneration, which needs JSBSim installed.

**The phugoid, diagnosed. It is two different problems wearing one name.** Sessions 17 and 18
attributed the whole of it to drag and thrust. Localised by comparing the two longitudinal
plant matrices **entry by entry** rather than on eigenvalues — substituting JSBSim's value for
one entry at a time and re-solving — it splits cleanly:

*The frequency error is one entry, `M_u` = ∂q̇/∂vt.*

| substituting JSBSim's… | cruise ωn | approach ωn |
|---|---|---|
| nothing (as built) | 0.05581 — 6.58% | 0.09370 — 3.42% |
| `∂q̇/∂vt` alone | 0.05251 — **0.27%** | 0.09063 — **0.03%** |
| JSBSim | 0.05237 | 0.09060 |

Every other entry of the 4×4 agrees to within 2.7% and moves the phugoid not at all. So one
derivative accounts for 96% of the cruise error and 99% of the approach one.

*AtiSim's `M_u` is entirely an α̇ coupling, and that is measured, not argued.* At fixed α
the build-up has no Mach dependence at all except wave drag, since there is no
Prandtl–Glauert correction anywhere in `aero.py` — and wave drag is exactly zero at both
recovery points, so it contributes nothing to `M_u` either.

> **Correction to the first writing of this entry**, which said "`CL`, `CD` and `Cm` are
> bit-identical at V ± 10 m/s … `M_crit` sits at 0.898 against M 0.78". 0.8977 is
> **M_dd**, not `M_crit`: `wave_drag` subtracts `_MDD_OFFSET` = 0.10772, putting the onset at
> **M 0.78998** — 0.010 Mach above the cruise trim point, about **3 m/s**. `CL` and `Cm` are
> speed-independent as stated, but `CD` is only so *below that onset*, and V + 10 m/s at
> cruise is above it. The `M_u` conclusion is unaffected — the onset is above the
> linearisation point, so wave drag has value and slope exactly zero there — but the margin
> is thin enough to be worth a tripwire, and now has one in
> `test_the_wave_drag_onset_sits_above_the_recovery_mach`.

What produces `M_u` is dynamic: perturbing speed changes the force balance, which changes ẇ, which changes α̇, which
`Cmadot` = −16 turns into a pitching moment. Setting `Cmadot` = `CLadot` = 0 sends `M_u` to
**−4.9e-19**, machine zero. (That probe is clean because α̇ = 0 at the trim point, so zeroing
the term does not move the equilibrium it is linearised about.)

JSBSim carries the same coupling **plus** genuinely Mach-dependent aerodynamics. The two
therefore differ in sign at cruise — AtiSim +1.114e-04 against JSBSim −1.024e-04 — and by a
factor of two at approach, +4.509e-04 against +2.254e-04.

**This is structural, not a defect.** A constant-coefficient model cannot carry a Mach-tuck
term. The phugoid is the only mode slow enough for speed derivatives to dominate — the short
period is over before the speed has changed, which is exactly why it agrees to 0.04% while
the phugoid does not. The consistent-with explanation on JSBSim's side is the Mach-scheduled
`Cmde` this document already records from 737.xml; at δe_trim = −0.052 rad a modest table
slope gives this magnitude. **Confirmed in session 20 by reading the file and running the
engine** — the table is −1.20 at M 0 and −0.30 at M 2, a slope of **+0.45 per Mach**, and
δe × 0.45 accounts for **90.5%** of the `M_u` gap at cruise and **83.1%** at approach.

*The damping error is a different derivative, `X_u` = ∂v̇t/∂vt.*

| | cruise | approach |
|---|---|---|
| `X_u` error | 1.06% | **8.56%** |
| phugoid ζ error | 3.44% | **13.72%** |
| ζ with JSBSim's `X_u` substituted | — | 13.72% → **3.0%** |

At approach the culprit is the ram term, which contributes **+6.50e-04 1/s of de-damping** —
thrust rising with speed. This is the generator's hard-coded Mach fit band showing up as a
number: `mach_ram` = 0.3346 was fitted over M 0.60–0.95 and is being applied at M 0.40.

**NO MODEL CHANGE WAS MADE, AND THAT IS THE POINT.** Setting `mach_ram` = 0 moves approach ζ
from 13.72% to 7.55% and cruise ζ from 3.44% to **13.35%** — it buys one condition at the
other's expense, because at cruise the term sits inside its fit band and is doing correct
work. Deleting it would be fitting the solver to a test result. The same applies to the
frequency half: inventing a `Cm`-versus-Mach term sized to close 6.58% would be tuning, not
modelling. Both fixes have to come from the source:

- **`mach_ram`:** re-fit from JSBSim's own CFM56 tables over a band that brackets each
  condition's Mach, exactly as `thrust_fit` already brackets altitude. **Done in session
  20** — see below.
- **`Cmde` Mach schedule:** read from 737.xml's own table, not chosen. Until then the
  frequency error is *explained* rather than removed, which is the honest state.

**The diagnosis is now asserted, not just written down.** Three tests, all test-only — no
coefficient moved:

| test | what it pins |
|---|---|
| `..._phugoid_frequency_gap_is_the_pitching_moment_speed_derivative` | substituting JSBSim's `M_u` alone must cut the frequency error **tenfold**, at both conditions. Substituting `X_u` instead moves it by under 1e-4 of itself, which is how the test is known to discriminate between entries rather than restate arithmetic |
| `test_atisim_has_no_aerodynamic_speed_derivative_of_pitching_moment` | zeroing `Cmadot`/`CLadot` drives `M_u` below 1e-12, at both conditions, with a vacuity guard that the un-zeroed value is non-trivial |
| `test_the_wave_drag_onset_sits_above_the_recovery_mach` | the one Mach term AtiSim does carry stays off at both recovery points, so the α̇ attribution holds |

This moves the phugoid from a 10% allowance to a **localisation**: the frequency gap is one
named derivative, and if it ever stops being that, the first test fails rather than the
number quietly drifting inside a tolerance.

**Still open, and unchanged by this session:** layer 3 runs at cruise only, and the approach
phugoid ζ is out by **13.72%** (0.04850 against 0.05621) with no test asserting it — the
largest disagreement anywhere in the comparison, and it would fail the 10% the cruise
phugoid test applies. Session 18's approach column reads "3.4%", which is ωn alone. Also
unchanged: the reference XML still carries no `<tolerances>` block, so the design's promise
that a widened tolerance shows up as a mismatch with its recorded derivation is still only
half-built.

### The engine run live, and the thrust fit re-banded (session 20)

JSBSim was available after all — installed in the system interpreter rather than the project
venv. Everything below is from the **same build** the reference was frozen at, 1.3.1 build
1837 commit `3b25f25e`, so it is a check of the record rather than a new baseline. The
generator gained a root-directory fallback so it can run from an interpreter that has
`atisim` (and therefore JAX) while reaching `jsbsim` over `PYTHONPATH`;
`get_default_root_dir()` raises `OSError` in that configuration.

**Three claims checked. All held; one method of mine did not.**

*`Ixz`, confirmed by a second and independent route.* Session 19 took the sign from JSBSim's
exported linearisation. Finite-differencing the running engine's own
`accelerations/rdot-rad_sec2` instead gives **∂ṙ/∂p = +1.180808e-02** against the exported
matrix's +1.180789e-02 — five digits, same positive sign, so the tensor element is +19109.1
and the property must not be negated. The generator's new `assert_inertia` also fired for
real during the regeneration and passed at 1.3e-07 relative.

*`Cmde` is a Mach table, read from the file.* −1.20 at M 0, −0.30 at M 2, so **+0.45 per
Mach**; at M 0.78 it evaluates to −0.849, which is the recovered value exactly. Reading each
`aero/coefficient/*` property directly shows `Cmalpha` and `Cmq` have **zero** Mach
dependence and `d(Cmde)/dM` is δe × 0.45 to the digit at both conditions.

*A measurement of mine was contaminated, and it is the same trap the original work hit.* My
first `∂Cm/∂M` was a fixed-α Mach sweep, which gave −0.0099 at cruise and **+0.048** at
approach — disagreeing with the table in magnitude and, at approach, in sign. The
decomposition shows why: `d(Cmadot)/dM` contributes +0.0134 and +0.0824, because setting a
state off-equilibrium sets α̇ and α̇ moves with the perturbation. That is exactly why `Cma`
needed a least-squares crossed design rather than a central difference. The per-function read
is the clean measurement; the sweep is not.

**The thrust fit is re-banded, and this is a model change — the only one this session.**
`thrust_fit`'s Mach samples were hard-coded at 0.60–0.95 regardless of condition; they now
bracket `MACH` at ±0.10, exactly as the altitude samples have always bracketed `ALT_FT`. The
number comes from JSBSim's own engine table, not from anything tuned. Measured at the
approach trim throttle, thrust runs 46309 / 43591 / **40864** / 41383 / 41906 N at
M 0.20 / 0.30 / **0.40** / 0.50 / 0.60 — a bucket whose minimum sits essentially at the
condition, so the local slope is *downward* where the extrapolated fit supplied *upward*.

Regenerated against the same build, **only three lines changed in each reference file** —
every derivative, trim, A/B matrix, sweep point and trajectory sample is bit-identical:

| | cruise | approach |
|---|---|---|
| `mach_ram` | 0.2511 → 0.2456 | **+0.3346 → −0.2949** |
| `max_thrust` | 101375 → 101668 N | 82579 → 91307 N |
| `thrust_mach_residual` | 0.193% → **0.151%** | 0.283% → **2.54%** |

The approach residual got **worse**, and that is the honest number: `1 + ram·M²` is monotonic
in |M| and cannot represent a bucket at all. The fit now reports a poor fit *at the
condition* instead of a good fit *somewhere else*.

| | before | after |
|---|---|---|
| approach phugoid ζ | 13.72% | **1.53%** |
| cruise phugoid ζ | 3.44% | **3.12%** |
| approach phugoid ωn | 3.42% | 3.39% |
| cruise phugoid ωn | 6.58% | 6.58% |

**The frequencies did not move, and that is the cross-check.** The refit touched thrust only,
so if the frequency error were drag-and-thrust it would have moved too. It did not, at either
condition — which confirms the two halves are separate mechanisms rather than one error split
two ways, and leaves the frequency where the `Cmde` table says it belongs.

The trim **thrust level** did not move either (+1.12% cruise, +1.56% approach, unchanged),
and could not have: `max_thrust` is solved so the model reproduces JSBSim's thrust *at* the
condition, so the two move together and only the Mach **slope** changes. That is precisely
the quantity the phugoid damping reads.

**`boeing737_approach` now declares a Mach validity band** of 0.30–0.50, which session 19 had
to leave undeclared because there was no honest band to state. `test_a_recovery_band_is_...`
caught the change and now asserts both entries carry a band containing their own flight
condition.

**The 737 carries JSBSim's own CL(α) table now, and the ceiling in §7 is open.** Four
points, transcribed exactly:

| α (rad) | α (deg) | CL | segment slope |
|---|---|---|---|
| −0.20 | −11.46° | −0.68 | |
| 0.00 | 0° | 0.20 | **4.400** below zero incidence |
| 0.23 | **13.18°** | **1.20** | **4.3478** — this *is* `CL0 + CLa·α` |
| 0.46 | 26.36° | 0.20 | −4.3478, past the break |

Segment two is the linear model already verified: its slope matches `CLa` to **4.8e-13** and
its intercept matches `CL0` to 5.5e-09. So the table changes nothing at either recovery
point — and in fact **improves layer 1's CL by five orders**, from 5.5e-09 to **7.1e-14** at
cruise and 9.6e-09 to 1.3e-13 at approach, because `CL0` was an intercept *solved* to
reproduce lift at the reference and carried that solve's residual where the table is simply
737.xml's own 0.20. Layers 2, 3 and 4 are bit-identical. The layer-1 CL tolerance is
tightened 1e-7 → **1e-12** to hold the improvement.

Outside the segment it changes everything that matters: the linear form reported **2.5× the
source's lift at α 20°** and 6.9× at 25°. Agreement across the full table — both clamped
endpoints, the break, and the falling branch — is now better than **1e-9**, asserted in
`test_layer1_lift_matches_through_the_stall` against a new `<stall_sweep>` block in the
reference. That block is additive: the ordinary sweep is untouched, so no existing layer-1
result moved.

**Two consequences §5 and §7 record as permanent are now conditional.** The ±g asymmetry was
a property of the *linear form*, not of the airframe — the table's slope is 4.400 below zero
incidence against 4.3478 above, so an up-gust and an equal down-gust no longer give equal and
opposite increments for an aircraft carrying one. And the Cessna's stall tables are still
unused, but now for want of a caller rather than for want of a mechanism. Both remain exactly
as recorded for every entry without a table, which is all four other aircraft.

**What opening the seam actually cost**, neither of which §7 anticipated: a kink at every
breakpoint, which `jacfwd` turns into a one-sided slope if anything linearises *at* one — the
737's cruise margin is 1.98° and is now asserted — and trim ceasing to be single-valued above
the break, which `trim.trim` has no defence against since it is an unbracketed root-find.

### Looking for a 747 lift curve, and why the Mach tables are not applied (session 21)

**No source this project can reach publishes a post-stall 747 lift curve.** Checked, not
assumed:

- **CR-2144** returns zero hits for "stall", "lift curve", "CLmax" or "nonlinear" across all
  352 pages. §IX gives linear derivative tables (IX-1, IX-2) and derivative-vs-Mach figures.
  It is a handling-qualities compendium of linearised sets by construction.
- **NASA CR-114494** (Hanke & Nordwall, Boeing D6-30643, 1970 — the canonical nonlinear 747
  model, now in `refs/`) has `CL_BASIC` vs α on pp. 2.0-7 and 2.0-8 as **straight lines**,
  annotated *"extrapolate linearly to higher α_WDP if required"*. Stall is a **boundary** —
  buffet onset, stick shaker, certification stall speeds — not a falling curve. §19 revises
  ground effect and buffet-onset α, not the lift curve.
- **JSBSim's B747** was rejected: its table is the 737's first three points verbatim with a
  different tail, author "Unknown", `release="ALPHA"`, implying `CLα` = 4.3478 against
  CR-2144's 4.9441. Adopting it would replace a qualified NASA number with an unqualified one
  24% different.

**Two figures were digitised, and both are recorded here rather than applied.** The working
patch is kept out of the tree; what follows is the data and the reason.

*`CL_MAX(M)` — CR-114494 p. 2.0-38, flaps up, gear up, trimmed.* Reading uncertainty **±0.02**:
the fine grid is 0.1 CL over 63 px and the drawn line is 4–8 px. The curve had to be separated
from the major gridlines by stroke thickness — gridlines render 1–2 px, curves 3 or more —
which is what stopped the trace latching onto the CL 0.894 gridline.

| M | 0.10 | 0.30 | 0.50 | 0.70 | 0.78 | 0.82 | 0.86 | 0.90 | 0.94 |
|---|---|---|---|---|---|---|---|---|---|
| CL_max | 1.097 | 1.053 | 1.008 | 0.952 | 0.910 | 0.878 | 0.834 | 0.763 | 0.678 |

*`CLα(M)` — CR-2144 Figure IX-5, 40,000 ft curve.* **The cross-check you would want:** read on
its own the figure gives **4.892 at M 0.80** against Table IX-4's **4.9441** — agreeing to
**1.05%**, which is the measured reading uncertainty of the whole digitisation. Level from the
table, shape from the figure. Uncertainty ±0.05 in `CLα`, about ±1%.

**WHY NEITHER IS APPLIED.** Both were implemented and the full suite was run. **Eleven tests
failed**, and not cosmetically:

- `test_the_six_dof_rollout_is_fourth_order`, `test_extracting_rk4_step_did_not_move_a_single_bit`,
  `test_the_rollout_is_only_first_order_through_a_spatially_varying_wind`,
  `test_a_time_varying_uniform_wind_adds_no_body_force` — **a hard `min` ceiling is a kink, and
  a kink destroys RK4's formal order** wherever a trajectory crosses it. The integrator
  verification in §4 is not compatible with a non-smooth force model.
- `test_load_factor_is_a_straight_line_in_air_relative_incidence` and
  `test_logging_the_run_did_not_move_the_headline_numbers` — the ceiling retires the linearity
  invariant the vortex analysis is built on. That is the ceiling *working*: at FC9 the aircraft
  trims at CL 0.654 against a 0.894 ceiling, and a vortex adding 10° of α asks for CL 1.52. The
  existing vortex results were in the regime §7 already warns "is not evidence of anything" —
  but bounding it changes every published figure in that analysis.

And `CLα(M)` **does not improve the 747** — measured at FC9, phugoid ωn goes 17.8% → 19.4%
(worse) and ζ 14.4% → 12.6% (better), the short period unmoved. That is the shape of a
**partial** correction: CR-2144's own FC9 model carries `Xu`, `Zu` and `Mu`, this project omits
all three by form, and a Mach-dependent `CLα` is an indirect stand-in for `Zu` alone.

**What shipping it properly needs**, and it is a planned piece of work rather than a coefficient
addition: a **smooth** saturation in place of `min` so the integrator keeps its order, a
re-verification that it does, and a re-baselining of the vortex analysis against a model that
now has a ceiling. **And the higher-value fix first:** `Xu`, `Zu`, `Mu` are exact tabulated
numbers already in Table IX-4, which `_boeing_747` skips with the note "the speed and alpha-dot
derivatives are outside this model's form". No digitisation, no reading uncertainty.

### Response spectra and load exceedance — session 25 (phase 2)

**The statistic changed, not the run.** Every row above this one compares a **peak** from a
single encounter. That is one realisation of a random process and its error bar does not
shrink with effort. `atisim/response.py` adds the two statistics that do, and neither
existed anywhere in the tree before: there was no FFT, periodogram or PSD of any
**response** (`wind.dryden_spectrum` is an *input* spectrum), and no exceedance count of
anything. `scripts/cat_spectra.py --seeds 32`, figure `09-spectra.png`.

**1. The Mehta run follows the airframe, not the forcing.** Mehta's five-core array is not
uniform: its four core spacings pass the aircraft at **0.0909, 0.1224, 0.1354 and
0.1886 Hz** at once, mean 0.1343. The 747 at that condition has one short period,
**0.16404 Hz** (ω_n = 1.0307 rad/s, ζ = 0.36455).

| | Hz | where the response peak sits relative to it |
|---|---|---|
| response peak, `n_z` PSD | **0.16885** | — |
| the airframe's short period | 0.16404 | **+2.9% above it**, 0.23 of a bin |
| the array's mean core passage | 0.13433 | +25.7% above it, 1.64 bins |

The record is 47.4 s, so a bin is 0.0211 Hz and the two candidates are only 1.4 bins apart
— which is why this alone is suggestive rather than decisive, and why the next row uses a
stationary field and an ensemble. **This is the first statement in this project about
which frequency the load followed, rather than how large it got.**

**2. Yoshimura's protocol, run.** Yoshimura et al. 2023 (GRL 50, e2022GL101286) validate a
CAT simulation by comparing the frequency spectrum of vertical acceleration against flight
records and checking the peak lands near the aircraft's own natural frequency — 0.14 Hz for
their B787, from 151 virtual flights of 100 s. Here: **32 flights × 100 s** through
`wind.dryden_field`, first 20 s of each discarded (declared), σ at the two ends of the
sourced range.

| σ_w, m/s | ensemble peak, Hz | vs short period | half-power band, Hz | `n_z` rms, g | worst condition drift over the record |
|---|---|---|---|---|---|
| 2.108 (`mehta_unmodelled_wind`) | 0.1400 | −14.7% | 0.1200–0.2000 | 0.0823 | 300 m altitude, 12.9 m/s (5.4% of V), \|α\| range 3.98° |
| 4.459 (`mehta_residual_ceiling`) | 0.1700 | +3.6% | 0.1200–0.2400 | 0.1836 | **703 m** altitude, **30.9 m/s (13.1% of V)**, \|α\| range **8.38°** |

**And TM-102186 states this mechanism in words, which the spectrum now measures.** Its
p. 3-4 discussion of Fig. 8 says the pitch and load variations "are dependent upon the
relationship between the time span of the vortex traverse and the aircraft's short
oscillatory period", and explains its three-aircraft ordering by exactly that relationship.
That is the resonance argument, asserted by the source and never tested here because the
project had no frequency-domain statistic. It now has one, and the headline run's peak
lands on the short period.

**The robust form of this result is not the peak location.** An averaged periodogram at
N = 32 is still noisy and the peak moved a bin and a half between the two intensities.
*(**Session 27 re-ran this at N = 151**, Yoshimura's own count: the movement was
under-sampling and both intensities land on 0.1700 Hz. The paragraph below is still the
right way to read the result, but the peak location turns out to be steadier than this
sentence allows — see "The Dryden ensemble at Yoshimura's own sample count" above.)* The
statement that does not depend on N is a ratio of two numbers: the Dryden **input** has
strictly *more* energy at 0.05 Hz than at the short period (it is flat below Ω = 1/L_w and
falls as Ω⁻² above), and the **response** has more than **3×** *less*. That reversal is the
whole content of "the airframe organises the load", and `test_cat_spectra.py` asserts it in
that form.

**3. The two σ limbs are one test, and what they measure is the drift.**
`dryden_field(σ, seed)` takes its phases from the seed alone and its amplitudes scale
exactly as σ, so seed *k* at the two intensities is the same field scaled — a linear
aircraft would return the same normalised spectrum from both. Measured: σ ratio 2.1156,
`n_z` rms ratio **2.2312**, i.e. **+5.46% against exact linearity**, with the normalised
spectra differing by 0.129 in L1.

**That superlinearity is not aerodynamic and the run was nearly quoted as if it were.**
`aero.py` is linear in α and `boeing747` carries no `CL` table, so the aero *cannot*
produce it. The drift column above is the candidate, and it is large: with fixed controls
there is nobody flying the aeroplane, so it leaves the condition it was trimmed for —
703 m of altitude and **13.1% of airspeed** at the upper σ, which moves dynamic pressure
by about a quarter. **The upper-σ ensemble is therefore not a spectrum of one flight
condition**, and its |α| range of 8.38° is inside §1's 10° envelope with little to spare.
Read the lower-σ ensemble as the clean one and the upper as a bound. Registered as
`ASSUMPTIONS.md` **E11**.

**4. The first exceedance curve.** Upcrossings per second of each load level over the same
ensemble, both signs, N in the denominator — 32 flights, **3,199 s of record** at each σ.

| Δn level, g | σ = 2.108 m/s, up / down per s | σ = 4.459 m/s, up / down per s |
|---|---|---|
| 0.10 | 0.551 / 0.555 | 0.937 / 0.922 |
| 0.20 | 0.049 / 0.060 | 0.617 / 0.590 |
| 0.30 | 0.0025 / 0.0025 | 0.312 / 0.266 |
| peak \|Δn\| reached | 0.346 | 0.825 |

Up and down agree to **0.973–1.175** at the upper σ across every level both populate, which
is the sampling error on the exact odd symmetry §5 records as structural. **At the lower σ
the same ratio reads 0.500–1.012, and that 0.500 is one event against two** in a tail bin —
which is the whole argument for the statistic: a rate has an N, so its error bar is visible
and shrinks, where a peak's is neither.

**What this does NOT establish, and the script says so before the figure.** It is not yet a
*comparison*. No published exceedance curve is held — TM-102186 gives Hannibal's load as a
two-number band and Wingrove & Bach 1994 Table 2 gives twelve single incidents, neither of
which is a rate — and no digitised acceleration history exists here, so the observed half of
Yoshimura's protocol cannot be run. Both are named in §5 as acquisitions rather than quietly
dropped.

### The validated baseline — do not touch these tolerances

`test_conservation.py`, `test_cr2144_modes.py`, `test_drag_polar.py`, `test_navion.py`,
`test_trim.py`. All are still-air statements; no reading of "turbulence landed" makes any
of them stale. If one moves, the derivative chain or the integrator changed.

## 5. Attributed gaps and structural impossibilities

> **`docs/ASSUMPTIONS.md` is the companion to this section.** §5 lists gaps found by
> comparing against sources; that file lists what the model *assumes* before any comparison
> happens, with a measured bound on each. Two of its entries are load-bearing enough to be
> repeated here.

### Status of every open item, at the release — session 32

**Every open item in §5, §7 and §8 carries exactly one of four statuses.** This table is the
index; the entry below it, or the section it names, is the evidence. Read this first if you
want to know what the model can and cannot be used for, and what is left to do.

| Status | Means |
|---|---|
| **CLOSED / MEASURED / BOUNDED** | answered — by a measurement in §4, a bound with its size, or a guard in code |
| **IMPOSSIBLE WITH SOURCES HELD** | cannot be answered from any document the project holds, and the row names which source failed and why (`docs/DEVELOPMENT.md` rule 2) |
| **FUTURE WORK** | a named, credible route exists and this release does not take it. **Not** a defect in the release; the list a successor starts from |
| **OUT OF SCOPE** | deliberately outside what the model is for, and declared |

**Established by reading each item against the code and against §4, not against its own
wording** — which is how five of them were found already closed (§9, session 32).

#### §5 — attributed gaps and structural impossibilities

| # | Item | Status | Evidence / route |
|---|---|---|---|
| 5.1 | Vortex core 2.3–3.1 spans, gust sampled at a point | **BOUNDED** | point-gust cost ≤ 4.4% on the headline (5.19); the strip path exists since session 24 |
| 5.2 | No wind varied across the span | **CLOSED**, session 24 | the lateral phase |
| 5.3 | No frequency-domain comparison | **CLOSED** (capability), session 25 · **IMPOSSIBLE WITH SOURCES HELD** (an in-band published curve) | HICAT is 45–70 kft against this project's ~~33–41~~ **35–45 kft** — band corrected session 33, verdict **NOT** revisited; see the note below the table |
| 5.4 | Gravity constant | **CLOSED** by session 28's merge — `g(z)` modelled | latitude (0.53%) and centrifugal stay DECLARED; `wgs84-earth` retires them (§0) |
| 5.5 | Phugoid / short-period offsets | Mach content **CLOSED**, session 30 · `Ṁw` ~~**FUTURE WORK**~~ **CLOSED, release 1.1** — declared as `Cmadot`, short-period damping −11.5% → **+0.6%** · `Żw` sign **IMPOSSIBLE WITH SOURCES HELD** | `Ṁw`: the α̇ branch, reviewed and parked (§0). `Żw`: IX-4 and IX-5 cannot arbitrate its sign |
| 5.6 | CR-2144 derivatives are the flexible airframe | **BOUNDED**, session 23 | the constant-Mach, two-altitude argument |
| 5.7 | `trim` converges to absurd roots for degenerate coefficients | **BOUNDED** — guarded | the 15° bound, non-binding on the registry's 0.01°–5.62° |
| 5.8 | Drag polar away from its fitted point | **BOUNDED** | residuals 0.004 / 0.014 / 0.006, quantified |
| 5.9 | Vortex parameter uncertainty | first layer **MEASURED** (8.45%) · second layer **IMPOSSIBLE WITH SOURCES HELD** | NASA's DC-10 aerodynamic model is unpublished (TM-102186 §2) |
| 5.10 | Lee wave has no horizontal perturbation | **IMPOSSIBLE WITH SOURCES HELD** | Doyle et al. give neither N nor cross-mountain U at 12 km |
| 5.11 | Lee-wave wavelength declared | **BOUNDED** — declared, cross-checked | Lester et al. 1989 measure ~22 km, 12% from the declared 25 |
| 5.12 | 747 cannot fly a microburst | **CLOSED**, session 10 | |
| 5.13 | FAA windshear thresholds are jet-only | **CLOSED** | physical verdict `F > (T−D)/W`; `σ_n` for altitude |
| 5.14 | Approach 747 below minimum-drag speed | **OUT OF SCOPE** | flown open loop, stated |
| 5.15 | No ground | **OUT OF SCOPE** | runs cut at one wingspan |
| 5.16 | Two errors in the sources | **CLOSED** — documented | take microburst geometry from Schultz Table 2 |
| 5.17 | Hannibal dated two ways | **CLOSED**, session 31 | NTSB CHI81DA042: 3 April 1981 |
| 5.18 | ±g asymmetry | buffet boundary **CLOSED**, session 26 · nonlinear lift curve **IMPOSSIBLE WITH SOURCES HELD** | CR-114494 draws `CL_BASIC` as straight lines |
| 5.19 | Absolute agreement with recorded g-loads | **IMPOSSIBLE WITH SOURCES HELD** · route **FUTURE WORK** | aircraft identified (N1809U, DC-10-10); the weight on the day is not found. Routes: the NTSB docket, the NTSB pre-1982 database (39 MB Access file, not tried), and the 1982 SFTE paper by Parks, Bach & Wingrove |
| 5.20 | Half the Fig. 8 load band unreachable | **IMPOSSIBLE WITH SOURCES HELD** — structural | linear aero; the same ceiling as 5.18 |
| 5.21 | The reported phugoid (4-state) is not the flown one — altitude couples through density | ~~**FUTURE WORK**~~ **MEASURED ON THE SHIPPED 747, session 33** (phase V1, `gust._linearise`): adding height as a fifth state moves the phugoid **ωn +12.61%, ζ −42.57%** at M 0.80 / 37,000 ft and **+13.30% / −41.56%** at `CRUISE`. The short period moves **+0.024% / +0.018%** — untouched, which is what says the effect is the slow height–density exchange and not a change of plant. The 4-state stays the correct CR-2144 comparator | the unmerged branch's +20.5% / −50.6% on the pre-session-30 747 is corroborated in sign and order and superseded in place. Free-response check: the 4-state matrix mispredicts the flown Δn by **6–26% over 20 s**, the 5-state by **0.1–1.4%**. Pinned by `test_gust.py::test_the_altitude_state_moves_the_phugoid_by_more_than_a_tenth` |

**§6 carries the one open latent bug, (h): the strip load path counts the gust's rolling moment
twice.** Measured this session; it supersedes §4's lateral strip result and the finishing plan's
phase-1 gate, and touches no longitudinal claim and nothing in §1.

> **§5.3's stated altitude band was wrong and is corrected — session 33. THE VERDICT IS
> DELIBERATELY NOT REVISITED.** The row read *"HICAT is 45–70 kft against this project's
> 33–41 kft"*, and **33–41 matches neither aircraft**: `boeing747` declares
> **35,000–45,000 ft** (`aircraft.py:780`) and `_boeing_747_jsbsim` declares
> **35,000–41,000** (`aircraft.py:1816`). The README's "35,000–45,000 ft" agrees with the
> code. The 41 appears to come from the cross-code entry and the 33 from neither — most
> likely from the Wingrove Cimarron case at 33,000 ft, which §4 flies with the **737**, not
> the 747.
>
> **What the correction does to the argument, stated rather than acted on.** The two bands
> no longer sit 4,000 ft apart: 45,000 ft is `boeing747`'s declared ceiling and HICAT's
> declared floor, so **they abut at a shared endpoint**. That is a materially weaker version
> of the band objection than the row has been carrying, and the same objection was used to
> disqualify NASA/TM-2003-212666 (design phase S1) at 33,000 ft — 2,000 ft below the floor,
> where it is still sound. **Whether an abutting band changes 5.3's `IMPOSSIBLE` verdict is
> a decision for the record's owner and not an incidental one** — §8's precedent, *"a
> decision, not a measurement"*. This session corrected the number, states the consequence,
> and changed no verdict. The stale figure survives in two places on purpose: a struck-through
> completed task row in §7 and a session-9 entry, both of which record what was believed at
> the time.

#### §7 — the plan's open rows

| Row | Status |
|---|---|
| DC-10 cruise derivative set | **IMPOSSIBLE WITH SOURCES HELD** — final, session 32: all five held DC-10/MD-11 documents assessed. CR-3677 partial; CR-3748, TM-4745, TM-1998-206552 and Taylor 1978 carry no stability derivatives |
| Hannibal flight record | flight and aircraft **CLOSED**, session 31 · weight as 5.19 |
| Build `boeing787_yoshimura` | **CLOSED**, session 27 — the row was stale; superseded in place |
| Frozen-`C_Lα` LES test | **CLOSED**, session 27 — the row was stale; superseded in place |
| Run the LES limb | **FUTURE WORK** — a 20% residual attributable to neither aircraft nor Mach, with the field reader shown sound |
| Compare the two CR-2144 digitisations | **MEASURED**, session 32 — the sealed prediction was WRONG on `Cm_M`, and the adjudication against Table IX-4 shows the automated trace is the biased one; the hand reading the 747 declares from is better-anchored on all three speed derivatives (§4) |
| The α̇ derivatives | ~~**FUTURE WORK**~~ **DONE, release 1.1** — `Ṁw` declared on `boeing747` from Table IX-4, re-derived here rather than merged from the parked branch; `Żw` stays out on its sign. §4, "The α̇ pitching derivative, declared". *Was:* **FUTURE WORK** — the branch is reviewed: its change to a rule-3 file is a re-capture, not a loosening (§0). Parked rather than merged because it moves the four CR-2144 modes and §1's headline. Schedule `Cmadot` from the **hand** reading, not the automated `p221` curve (§4) |
| Hannibal inventory items 2–6 | **FUTURE WORK** — each has its script already named in the inventory table below |
| 14 CFR 25.341 | **IMPOSSIBLE WITH SOURCES HELD** for the lineage check — read session 32: its continuous-turbulence `Uσref` is a design limit, constant at 79 fps TAS above 24,000 ft, with no exceedance probability, so it is not commensurable with Fig. 7. ADA119421 is still what would settle the lineage |
| NASA TP-2469, Sharman et al. 2014, MIL-STD-1797A | **FUTURE WORK** |
| Sensitivity study: C3's α axis, interaction terms | **FUTURE WORK** — both are declared limits of the study |

#### §8 — open questions

| Question | Status |
|---|---|
| Where would Fig. 8's pitch axis fail? | **IMPOSSIBLE WITH SOURCES HELD** — needs a nonlinear lift curve, which no held source publishes |
| Why do the two engines choose different cores on the array? | **FUTURE WORK** — §8 names the discriminator (a still-air run of the same length from the same state); JSBSim 1.3.1 is installed on the reference interpreter to run it |
| Units of Mehta's cost `J`, and `N` | **IMPOSSIBLE WITH SOURCES HELD** — Mehta 1987 never labels `J`. The 4.46 m/s ceiling stands unconditionally; the 2.11 m/s split does not |
| Cost of the `PARKS_CASES` hybrid | **CLOSED** — moot since session 26, verified in code session 32 |

#### The one sealed prediction

`dc10_does_not_close_the_hannibal_gap` stays **SEALED and unedited** (`atisim/predictions.py`,
rule 1). It is **IMPOSSIBLE WITH SOURCES HELD** to settle: it needs a DC-10 built from a
published derivative set. The 747's peak-to-peak has moved since it was sealed — 1.8385 g then,
about 2.03 g now (75.3% of 2.70 g) — but its band is absolute, **[1.563, 2.114] g**, so it
remains decidable exactly as written.


**Attributed — understood, documented, not bugs:**

- **The Parks vortex core is 2.3–3.1 wingspans, and the gust field is sampled at a point.**
  The wind is evaluated at `pos_ned` and `field_model` derives `omega_gust` from the
  analytic gradient — a **first-order** correction for variation across the airframe. That
  is comfortable for every field in the project except the one the headline result uses:

  | Field | Scale | In 747 spans (59.64 m) |
  |---|---|---|
  | Parks Hannibal core radius | 182.9 m | **3.07** |
  | Parks Morton core radius | 137.2 m | **2.30** |
  | Wingrove updraft radius | 2359 m | 39.6 |
  | Doyle lee wave, quarter wavelength | 6250 m | 104.8 |
  | Oseguera microburst radius | 1000 m | 109.8 (Cherokee spans) |

  At 2–3 spans the linear-gradient correction is doing real work rather than tidying up,
  and second-order variation across the span is not represented at all. This compounds with
  the ±25% parameter band below and with §2's note that the rotational gust already exceeds
  the 747's full aileron authority by ~1.5×. **Vortex conclusions stay orderings** — which
  this section already required for a different reason.

  **Session 23 measured what it costs on the headline run, and it is small.** Swapping the
  tangent gust rates for rates fitted across the airframe moves the load increment by
  **−1.67% up and +4.40% down**; strip-integrated loads move it by **exactly zero**,
  because `loads.strip_increment` is roll-only and `vortex_wind` has no `y` dependence for
  it to see. Step-size refinement from dt 0.02 to 0.0025 moves the peak by 0.14% total.
  **So the model's own approximations bound at about 4.4% of the 32% by which the run
  under-reaches the DC-10's recorded load** — the shortfall is elsewhere. §4 has the table.
  Note which channel moves: the **down** excursion, not the peak.

  > **SUPERSEDED, end of session 30.** The two results this paragraph rests on were flown with
  > Mehta's field at the 747's own altitude. Replayed on the path the field was identified along
  > — the headline since the end of session 30 — the favourable corner reaches **80.4%**, not
  > 72.7%, and the peak is **not saturated**: +1.7 g is reached at ×1.15 V₀ inside the linear
  > range. A wind 15–35% stronger than Mehta's fit, which is the direction session 27's digitised
  > record already pointed (~12% light), closes most of the shortfall. **The shortfall is not
  > cornered on the aircraft.** §4's first entry.

  **Session 23c closed the remaining wind-side candidates, and the shortfall is now
  cornered on the aircraft.** Propagating the sourced `V₀` uncertainty with a declared
  `r₀` band, the most favourable corner reaches **72.7%** of the recorded peak-to-peak;
  and the peak load turns out to be **saturated** — tripling `V₀` moves the up-increment
  from 0.441 to 0.526 g against a recorded 0.70, elasticity under 0.15 and changing sign,
  with peak |α| still near 8°. The peak first reaches +1.7 g between **×3.25 and ×3.50**
  of the identified `V₀`, and |α| leaves the 10° linear range in the **same interval** —
  so **there is no gust strength at which this model both reaches the record and may be
  believed.** That excludes amplitude rather than merely bounding it. §4 has both tables.

  **Session 30: the sharp claim survives and its support weakened.** With CR-2144's speed
  derivatives declared on `boeing747`, the bracket moves to ×3.0–×3.25 and still coincides.
  But the saturation elasticity rises to **+0.305** (+0.097 with the seam shut), and the
  pitching-moment term does most of that on its own. Every other figure in this bullet —
  including the 72.7% favourable corner, from `cat_uncertainty.py`, which was not re-run —
  is pre-declaration. §4's session-30 entry, item 7.

- **No wind field varied across the span, so the model was longitudinal by construction.**
  **Found session 24 while writing the validation claim above; it was in no document.**
  Every field was a function of along-track distance alone, and each for a different
  reason: `vortex_wind` has no `y` in its equations, the updraft and microburst are
  axisymmetric and penetrated on-axis, `LeeWave` is a function of north, and
  `dryden_vertical_field` is one component. Three things followed and they compounded —
  `wind.strip_roll_moment` integrated to **exactly zero** on every field, so the strip
  load path built in session 14 had never moved a reported number; the lateral modes were
  validated as eigenvalues and never once excited; and `vortex_viz.Encounter` carried no
  roll, sideslip or rate channel, so the pipeline could not have *reported* a rolling
  response if one had occurred.

  **Session 24 (phase 1) closed the capability and found a real simplification doing it.**
  Parks' model is two-dimensional in the plane **perpendicular to the vortex lines**;
  `vortex_wind` returns his horizontal magnitude along the **flight path**, which is exact
  only when the lines are square to it. At Mehta's ψ = 31° the true perturbation is that
  magnitude rotated by ψ, so **cos ψ = 0.857 of it lies along the path and sin ψ = 0.515
  across it** — and the across-path half, the only sideslip input this field has ever had,
  was being discarded. `wind.line_vortex_wind` writes the same equations as lines in space:
  it reproduces `vortex_wind` along the flight path to **1.2e-14 m/s** in both geometries
  and differs off it. §4 has the run.

  **What is now capability and what is still a gap.** The model can represent a rolling
  gust and a lateral one (`wind.dryden_field` adds the `u` and `v` components), and the
  channels exist to see them. **None of it is validated against anything** — no source
  held here supplies a recorded lateral CAT response. §3 already notes Wingrove & Bach
  1994 supplies *"no lateral data"*, which had been recorded as a limitation of the paper
  rather than as a hole in the model. Treat every lateral number as a capability
  demonstration, never as evidence.

- **Nothing in the project measured a response in the frequency domain, and the one paper
  the dossier calls a blueprint validates that way.** Until session 25 there was no FFT,
  periodogram, PSD or Welch estimate anywhere in the tree. Every comparison matched a
  **peak** from a single encounter — one realisation of a random process, whose error bar
  does not shrink however much more work is done. Yoshimura et al. 2023 compare **spectra**
  and check the peak against the aircraft's own natural frequency; that protocol could not
  be run here at all.

  **Session 25 (phase 2) closed the capability half.** `atisim/response.py` supplies the
  estimators, `scripts/cat_spectra.py` runs Yoshimura's protocol on this project's aircraft
  and its only stochastic field, and §4 has the numbers — including the result that the
  load follows the airframe's short period rather than the array's core-passage frequency,
  which is a statement about coupling that no peak could have made.

  **The comparison half is still open, and it is source-gated.** Two documents would close
  it, and neither is held:

  - a **digitised acceleration history** from a recorded CAT encounter, which is what turns
    limb B from "the model agrees with its own dynamics" into "the model agrees with a
    record". **This one is NOT an acquisition, and an earlier draft of this entry said it
    was.** `Reference_papers/19890016606.pdf` — TM-102186, held since session 23 — carries
    the Hannibal DFDR **g-load time history as Figure 6**, p. 3-5, plotted against GMT
    1:21–1:26 on a −1 to +2 g axis, beside the horizontal and vertical winds derived from
    it. Yoshimura's equivalent is *not* in the figshare set (withheld by confidentiality),
    so this held figure is the only recorded trace within reach.

    **Its difficulty is real and is a different difficulty from the project's earlier
    digitisations.** `CL_MAX(M)` and `CLα(M)` were smooth single-valued curves; this is a
    dense oscillatory trace whose severe passage occupies perhaps 15% of a five-minute axis
    on a 1989 scan, so the usable band is set by print resolution rather than by the DFDR.
    Attempt it as a bounded experiment with its reading uncertainty measured first — and if
    the band it supports does not reach the 0.164 Hz short period, say so and stop.
  - a **published load-exceedance curve** for transport-category cruise. The model's own
    curve now exists with N in its denominator; there is nothing held to overlay it with.
    TM-102186's two-number band and Wingrove & Bach's twelve incidents are not rates.

  Until then, treat the spectra as evidence about the **model's internal consistency** —
  which is real, and is what §4's rows claim — and not as agreement with the atmosphere.

- ~~**Gravity is constant at 9.80665 m/s², which is +0.383% high at the 747's cruise
  altitude.**~~ **SUPERSEDED by session 28's compressibility merge — found stale in session 32.**
  Gravity is no longer constant on `main`: `dynamics.gravity(z) = g₀(R/(R+z))²`
  (`atisim/dynamics.py:31`), and `ASSUMPTIONS.md` A2 is retired. **Still assumed:** the
  latitude variation (0.53%) and the centrifugal term; `wgs84-earth` carries both. The
  session-12 measurement below is kept, because it is what said the change was safe.
  *The session-12 record, as written:* True `g(h) = g₀(R/(R+h))²` is 9.76922 at 12,192 m. **Session 12 measured what
  that costs and decided not to model it**; §4 carries the table and `ASSUMPTIONS.md` §A2
  the reasoning. Lanchester's `ωn_phugoid = √2·g/u₀` predicts a **1:1** mapping and the
  measurement confirms it to three figures — phugoid ωn moves −0.3798% against a −0.3816%
  change in g.

  ~~It is the same order as the tightest agreements in §4, so no claim below ~0.5% at
  altitude is safe until `g(h)` is modelled.~~ **That was session 11's reasoning and the
  measurement contradicts it.** The 1:1 mapping is the phugoid's alone: the short period is
  immune (+0.0002%) and the lateral modes — which are where §4's tightest agreements are —
  move only **0.055–0.079%**, five to fifteen times smaller than the agreements they were
  feared to threaten. Their sensitivity is indirect, through a trim α that falls 0.65%, not
  through a gravity term in the lateral equations. The corrected rule: **a sub-0.5% claim
  at altitude is unsafe for the phugoid and safe for the other four modes.** At sea level —
  the approach 747 — the error is exactly zero, and the measured movement there is exactly
  0.0000%, which is the control on the whole experiment.

- **Phugoid and short-period offsets.** ~~The sim's aero form is α/q/δe only; CR-2144
  Table IX-4's `Xu, Zu, Mu, Żw, Ṁw` are deliberately excluded.~~ **That wording was wrong
  and session 11 has the measurement to fix it.** The model **has** Xu and Zu: they fall
  out of dynamic-pressure variation, since lift and drag both go as V², and the
  stability-axis A[0,0] lands within **1.2%** of Caughey's Xu with no Xu entered anywhere
  in `aircraft.py`. `Mu` genuinely is ≈ 0, since Cm = 0 at trim and there is no Cm_M term
  — consistent with Caughey's A[2,0] = 0.0001 being his `Ṁw·Zu`, not an Mu.

  What is excluded is the **Mach content** of those derivatives (CXu, CZu, from CL_M and
  CD_M), plus **Żw and Ṁw** which are genuinely absent. That distinction is the
  explanation, because it predicts what §4 measures: **the offsets are
  condition-dependent.** Phugoid ωn is off by 17.8% at M 0.80 / 40,000 ft and **0.4%** at
  M 0.25 / sea level, same code, same omissions. Compressibility is what drives the
  missing terms, and there is none at M 0.25.

  **Session 30: the Mach content is no longer only attributed.** It has been sourced (CR-2144
  p. 222, digitised and checked against Table IX-4) and made available through
  `Aircraft.CL_M/CD_M/Cm_M`. **No entry declares it yet.** Declared on a copy, it takes phugoid
  ω_n from −18.1% to +4.1% and ζ from +13.2% to +4.6% (±5 points from the reading).

  It leaves two named residuals, neither a digitisation error:
  - ~~**No thrust line.** CR-2144's trim C_m is −0.0159 and the engine's is 0. Compensating for
    that term closes both phugoid errors to under 1%.~~ **CLOSED later in session 30**: Table
    IX-3's thrust line was declared (−0.05% / +1.13% at 10 ft), and then CR-114494's revised 5.70 ft replaced the arm: **+1.69% / +2.83%** (§4).
  - **Korn/Lock's drag Mach slope is 1.7–1.9× CR-2144's at FC9.** Left in place, it makes
    phugoid damping worse than not declaring at all.

  §4 has the entry.

  Żw and Ṁw are now **reconstructed rather than attributed** — restoring Caughey's own
  CLα̇ = 6.7 recovers his published A[1,1] to 0.03% (§4). A second linear model built for
  mode extraction only, restoring all of them, closes both modes to ~1% of the reference.

- **CR-2144's 747 derivatives are the FLEXIBLE airframe.** Section IX's derivative plots
  are labelled "Flexible" — they carry aeroelastic corrections — and `dynamics.py`
  integrates a rigid body. This is a genuine model/data mismatch and it is larger in
  consequence than the document's 1972 date, which threatens nothing that is only ever
  compared against the document's own arithmetic (§3, and the design spec's
  "Source qualification"). ~~Not quantified: doing so needs a rigid derivative set the
  project does not hold.~~

  **Session 23: bounded, and it did not need a rigid set.** Stop asking for one. For a
  rigid aircraft the non-dimensional derivatives are functions of Mach and geometry;
  altitude does not enter. So at *constant Mach* they must be identical at two altitudes,
  and any movement bounds everything altitude-dependent — including the aeroelastic term.
  Yoshimura's Table A2 supplies CR-2144's own 747 at M 0.8 / 6,096 m against
  `aircraft.boeing747`'s M 0.8 / 40,000 ft, a **2.48× change in dynamic pressure**
  (at constant Mach `q̄ = ½γM²p`, so it follows static pressure, not density — the
  density ratio is 2.16):
  `C_Zα` −16.6%, `C_mα` −60.9%, `C_mq` −16.7%, `C_Zq` −18.7%, **every one less stiff at
  the higher q̄**, which is the direction aeroelastic relief predicts and the opposite of
  a transcription error.

  **A ceiling, not a measurement of flexibility.** Reynolds number moves by the same
  factor, and CG is not excluded — Table A3 says 25% MAC and CR-2144's CG for FC9 is not
  in the material held here. A CG shift would land almost entirely on `C_mα`, the largest
  mover, so that row is the least trustworthy and should not be quoted alone. Weight *is*
  excluded: 2.888e5 kg against Table A3's 2.89e5, `I_yy` 4.488e7 against 4.49e7.
  `docs/ASSUMPTIONS.md` B1 and C3 carry the tables and the consequence.

- **`trim.trim` converges to physically absurd roots for degenerate coefficients.**
  `CL = CL0 + CLa·α` is linear, so a huge α compensates a small CLa and Newton reaches a
  root that satisfies the residual to machine precision and is not a flight condition.
  Convergence and sense are different questions. Found by a sweep guard failing to fire.

  **Two numbers in the session-11 wording were wrong, corrected session 12.** The −633°
  was attributed to CLa = 0.1; it is **CLa = 1e-4**. Measured, `boeing747_approach` at
  85 m/s and sea level: CLa = 0.1 gives **−272.7°** at residual 2.3e-15, CLa = 1e-4 gives
  **−632.1°** at 5.7e-15. And "every real aircraft trims at 5–6°" was wrong in the other
  direction — the registry spans **0.01° (Cherokee) to 5.62°** at its own cruise
  conditions, which is what makes the 15° bound non-binding on legitimate data.

  **The angle itself is not reproducible, and only the phenomenon is.** The far root is
  chaotically sensitive to the start conditions: same aircraft, same CLa = 1e-4, sea level,
  **85.0 m/s gives −632.1° and 84.9 m/s gives −4232.1°**. So no specific angle is asserted
  anywhere — the test asserts converged-and-absurd, which is the stable fact. This is why
  quoting one in §5 produced two wrong numbers in the first place.

  The bound now lives in **`trim.is_physical`** rather than in `validation.sweep`, which is
  where session 11 put it. The defect is in `trim.trim` — it returns the absurd root and
  says nothing — so every other caller was equally exposed. It is not folded into `trim`
  itself because `trim` is jitted and vmapped (`minimum_drag_speed`) and therefore cannot
  raise. Nothing in the project's own results is affected; any future parameter study must
  check the angle, not just the residual.
- **Drag polar away from its fitted point.** `CD0` and `e` were back-solved from a single
  reading. Residuals are within 0.004 near the fit, up to 0.014 below M 0.75 (parabolic
  polar misses the induced rise) and 0.006 above M 0.88 (Korn law extrapolating past its
  single anchor).
- **Vortex parameter uncertainty inherited from the source.** Parks derives α from
  accelerometers *"together with a knowledge of the aircraft's aerodynamic
  characteristics"* — so there are two layers of modelling between the raw DFDR data and
  the identified r₀/V₀. ~~A 1° α error maps to 4.12 m/s of wind, 27% of a 50 ft/s peak.
  Treat the identified parameters as order-of-magnitude with roughly ±25% bands.~~

  **Session 23: the FIRST layer is now measured and the band is about a third of that.**
  Lester, Sen & Bach 1989 Table 1 propagates the uncertainties of exactly this
  reconstruction and populates them — RSS **2.449 m/s horizontal, 2.236 m/s vertical** at
  V = 250 m/s. Against Mehta's converged `V₀` = 26.46 m/s that is **8.45%**, not 25%. The
  same table shows the 1° assumption was itself twice too large: `V·δ(Θ−α) = 2.0` m/s at
  250 m/s is **0.458°**.

  **The SECOND layer is still attributed, and that is the one the wording above is really
  about.** Bach 1991 (NASA RP-1252) ch. 7 states the α reconstruction is only used where
  the recorder carried no vane, and that the L-1011 carried two at 2 Hz — it does not say
  what the Hannibal DC-10 carried. The definitive treatment is **Bach & Parks 1987,
  *J. Aircraft* 24(11) 789–792, which this project does not hold.** So: reconstruction
  error measured, identification error attributed, orderings-only rule unchanged.
  `docs/ASSUMPTIONS.md` E2 carries the table.

  **Session 23c: the band is now PROPAGATED rather than merely stated, and a third layer
  got measured.** Flying `V₀` across the 8.45% and `r₀` across a declared ±15%, the most
  favourable corner reaches 72.7% of the recorded load against a baseline 68.1% — so the
  parameter uncertainty is real but far too small to be the discrepancy. Separately,
  Mehta's published cost series bounds the **field-form** error, which is the layer
  neither Lester nor Bach speaks to: his Eq. (A3) is a *mean* square, so `J` = 214
  converts to a 4.46 m/s RMS residual without knowing `N`, and he states that n = 6, 7 do
  not improve on it. What the array cannot represent is therefore bounded at
  **2.11–4.46 m/s** of wind. §4 has both.

- **The lee wave carries no horizontal perturbation, so half the F-factor is missing.**
  `wind.LeeWave` is purely vertical and constant in altitude. That is divergence-free, so
  it is an admissible flow rather than a convenience — but a real lee wave also has a
  horizontal velocity perturbation, in quadrature with the vertical one, with amplitude
  ratio `m/k` (vertical to horizontal wavenumber). Building it needs a Brunt–Väisälä
  frequency and an ambient cross-mountain wind speed at 12 km, and **Doyle et al. supplies
  neither** — the paper gives wave amplitudes and a tropospheric wavelength band, not a
  stratification profile. So `U̇ₓ/g` is exactly zero here and the reported F is the
  vertical term alone.
  Two things follow, and they point opposite ways. The omitted term is **in quadrature**,
  so it peaks where the vertical term vanishes and vice versa — the *location* of peak F
  would move but the peak *magnitude* would not simply double. Against that, an
  order-of-magnitude estimate with a plausible `N` and ambient wind puts the shear term
  **larger** than the vertical one, so the true hazard is probably understated. The
  measured result is therefore a **lower bound**, and is quoted as one.
- **The wavelength is declared, not sourced.** Doyle et al.'s 20–35 km is tropospheric and
  the same paragraph warns "shorter wavelengths are apparent in the stratosphere" without
  quantifying them. 25 km is the middle of the band the paper *does* give. It does not
  move the F-factor peak at all — with no horizontal perturbation F is `−w/Vₐ`,
  independent of wavelength — but it sets the encounter duration and the pitching gust
  rate, so anything depending on those must say which value was used.

  **Session 23: still declared, but the declaration now has a measurement standing next to
  it.** Lester et al. 1989 derive a lee wave of **wavelength about 22 km** from DFDR data
  at 10 km (33,000 ft), which the same paper places about a kilometre above the tropopause
  — i.e. in the regime Doyle declined to quantify. That is **12% from the declared 25 km**
  and inside Doyle's band. Stored as `wind.LESTER_LEE_WAVE_WAVELENGTH`.

  **Not substituted, and the reason matters.** Swapping it would trade a declared number
  inside a measured band for a measured number from a different mountain range (Greenland,
  not the Sierra Nevada that `LEE_WAVE_AMPLITUDE` comes from) at a different altitude
  (10 km, not the modelled 12.192). That is not obviously an improvement. What changes is
  that the declaration is now bounded rather than unbounded.

  **The same paper also corroborates the omission below it.** It reports the flight-level
  horizontal windspeed dropping sharply at exactly the point of largest vertical motion,
  read there as a critical level from overturning waves — direct evidence that the
  horizontal perturbation `wind.LeeWave` omits is real, which turns "the reported F is a
  lower bound" from an argument into an observation.

  **Session 23 follow-up: flown, and the omission turns out to be much larger than the
  horizontal perturbation.** The 747 through a 22 km wave at Lester's own altitude
  reproduces his recorded **300 m altitude gain** at `w0` ≈ 4.0 m/s — inside Doyle's
  measured 3–6 m/s — and needs `w0` ≈ **113 m/s** to reach his recorded **+2.7/−1.0 g**.
  The two observations imply amplitudes **28× apart**, so a single smooth wave cannot
  produce both. §4 has the sweep. What `wind.LeeWave` is missing is not primarily the
  quadrature horizontal term — it is everything the wave breaks down INTO, which is what
  Lester's critical level is about and what carries the accelerations.

- ~~**The 747 cannot be flown into a microburst.**~~ **CLOSED, session 10.** It was true
  while the only derivative set was flight condition 9, Mach 0.8 at 40,000 ft. CR-2144
  Table IX-2 turned out to hold a complete **non-dimensional power-approach set** —
  the module's own header had said so since session 1 without anyone acting on it — so
  `boeing747_approach` now exists and the microburst runs on the aircraft class the
  thresholds were written for. §4 has the numbers.
- **The FAA windshear thresholds apply to the jet and not to the light aircraft.**
  Proctor et al. state plainly that the 0.1 hazard and 0.13 must-alert figures, and the
  1 km averaging scale itself, were established for jet transports and "are yet to be
  determined" for piston aircraft. So `scripts/microburst.py` decides per aircraft:
  the 747 gets a verdict against them, the Cherokee gets them printed for scale only. In
  both cases the **physical verdict is `F > (T−D)/W`**, which is that paper's own
  criterion and needs nobody's certification basis. Note this is a *different* reason from
  the lee wave's, where the thresholds failed on altitude rather than aircraft class.

  **Session 23: the ALTITUDE failure now has an instrument that does not have it.**
  Misaka et al. 2008 §IV.B apply RMS normal load `σ_n` — a moving 5 s average — with
  moderate 0.2–0.3 g and severe ≥ 0.3 g, attributed there to Hamilton & Proctor. It is
  defined for cruise-altitude turbulence, which is where every CAT case this project holds
  actually happened. `checks.rms_normal_load` implements it as a **report**, never a gate:
  the bands say how rough the air was, not whether the run was good, and a gate would fail
  the simulator for correctly flying through severe turbulence.

  **What it cannot do, from the source's own results.** Misaka Figs. 26–27 find `σ_n`
  tracks the *trend* of measured vertical acceleration and misses the peaks, by
  construction of the 5 s window. So it grades an encounter and is the wrong instrument
  for a core penetration — the 1.5 s a 747 spends inside a Parks core is under a third of
  one window. `rms_normal_load` returns `nan` with a stated reason rather than a number
  for any run shorter than its own window.
- **The approach 747 flies below its minimum-drag speed**, by 12.2 m/s, because 1.4 Vs at
  max landing weight is on the back side of the drag curve — which is where an airliner on
  final is. The autopilot's loop pairing is therefore inverted for that entry. It holds
  trim, but no gain set repairs the pairing, so it is flown open loop for all analysis.
- **There is no ground.** No terrain, no landing gear, no ground effect, no stall. A
  microburst run therefore ends when the aircraft descends within one wingspan of the
  surface, because below that the integration is arithmetic rather than physics — left to
  itself the model bounces and climbs away, which reads as a survival and is not one.

- **Two errors in the sources themselves, found session 23. Neither is this project's, and
  both change what may be quoted from those documents.**

  **(a) TM-102186 publishes Schultz's pre-fit guesses as his DFW answer.** It reports the
  microburst rings as "outer ring 15,000 ft diameter with vortex core diameter of 3000 ft;
  inner ring 2500 ft diameter with core diameter 900 ft". Those are Schultz 1990's
  **Table 1 *initial estimates***, to the digit. His converged Table 2 reads: outer ring
  radius 7,574 ft (diameter 15,148 ft) with core **4,856.8 ft**; inner ring radius
  1,594.3 ft (diameter 3,188.6 ft) with core **985.0 ft**. The outer core is **62% larger**
  than the figure TM-102186 prints. **If microburst geometry is ever taken from
  TM-102186, take it from Schultz Table 2 instead.**

  **(b) Yoshimura 2022's conclusion contradicts its own Table A5.** The conclusion calls
  the B747 short-period frequency "1.29 Hz" and derives a 200 m resonant wavelength from
  258 ÷ 1.29, then argues 10–20 m grid spacing is needed. Table A5 labels the quantity
  `s⁻¹`, and its own Table A2 derivatives reproduce 1.29 **rad/s** = 0.205 Hz (checked:
  `ω_n² = Z_α M_q/U₀ − M_α` gives 1.286). At 258 m/s the resonant wavelength is
  2π·258/1.29 ≈ **1,257 m**, not 200 m — so the grid-resolution conclusion does not follow
  from that calculation. **Table A2/A3/A5's numbers are used here; that conclusion is
  not.**

  These are the same failure mode as the 600 ft Hannibal radius (§3): a downstream paper
  quoting a startup estimate, or a mislabelled unit, as a result. It is why §3's rule that
  every number carries the table it came from is worth its cost.

  **(c) A third, added session 32: Caughey's 747 approach `Ixz` is a third value, and not a
  sign convention.** Caughey's MAE 5070 lateral worked example uses **Ixz = −2.23e6** for the 747
  power approach, against CR-2144's **0.825e6** (Figure IX-1) and **0.870e6** (Table IX-3) —
  opposite sign and 2.56× the magnitude. It is not a convention difference: his own appendix
  carries **+0.97e6** for the cruise condition, matching CR-2144. Substituting his Eq. (5.89)
  inertias closes every element of that worked example to under 3%, against disagreements of up
  to 189% otherwise, so **the whole lateral discrepancy with Caughey is his `Ixz`**. This project
  takes Figure IX-1's value and has not moved it. **Found by
  `claude/linearisation-verification-bounds-b73868` (commits `65b7ce3`, `8c36d92`), never merged,
  harvested session 32** — measured on that branch's tree, not re-run here. If the 747 approach's
  lateral modes are ever compared against Caughey's worked example, use CR-2144's `Ixz` for both
  sides.

- **The Hannibal encounter is dated two ways.** Mehta 1987 says July 1981; TM-102186
  says April 1981 in three figure captions and Bach 1991's Table 7.1 lists case 1 as
  `4/81`. Everything else matches across the accounts — 37,000 ft, DC-10, ψ = 31°, a
  ~150 kt horizontal wind bias, the same ±25 kft plot range — so they are the same
  encounter. **Two NASA documents say April; cite April 1981.** Nothing physical depends
  on it. **CLOSED, session 31:** NTSB case CHI81DA042 dates the encounter **3 April 1981**,
  which settles it for the two NASA documents against Mehta.

**Structurally impossible — cannot be fixed from any source currently held:**

- **The ±g asymmetry.** Both papers attribute it to stall buffet. `aero.py` is
  `CL = CL0 + CLa·α`, exactly odd-symmetric in Δα, so an up-gust and an equal down-gust
  give equal and opposite load increments to machine precision. The only aircraft in the
  project with nonlinear data is the Cessna, which is out of scope; **CR-2144 provides no
  buffet-onset table for the 747**. Reproducing this needs a source the project does not
  have. Do not promise it.

  **Session 25 splits this in two, and only one half is still blocked.** The *nonlinear lift
  curve* is genuinely not published in anything reachable — §4's session-21 search found
  CR-114494 draws `CL_BASIC` as straight lines annotated *"extrapolate linearly to higher α
  if required"*. But the **buffet-onset BOUNDARY is held**, on a page this project has
  already read: `refs/NASA-CR-114494.pdf` p. 2.0-38 is titled *"LIFT COEFFICIENT — BUFFET
  BOUNDARY AND C_Lmax"* and carries **two** curves against Mach, flaps up, gear up, trimmed
  — *maximum demonstrated `C_L`/trimmed `C_Lmax`*, which session 21 digitised, and *initial
  buffet boundary*, which it did not. The sheet points to **§19 for revised data**, and
  p. 428 defines the boundary in α and Mach and notes that above M 0.85 buffet is
  shock growth rather than stall.

  **The boundary buys a bound, not the asymmetry.** It cannot produce a falling lift curve,
  so the ±g asymmetry stays structurally impossible. What it can do is say *where the linear
  model stops being defensible in `C_L`–Mach* — the service `panel.ALPHA_LINEAR_DEG`
  performs in α, and currently the only guard of its kind. §1's envelope would gain a
  sourced ceiling in place of a declared one.
- **Absolute agreement with the papers' g-loads.** Wingrove & Bach never identifies an
  aircraft type; Parks' two cases are DC-10s at 37–39 kft against this project's 747 at
  40 kft with roughly 0.8× the wing loading. Every load comparison is order-of-magnitude
  or clustering. Assert bands and orderings, never values.

  **Session 23c turned this from a caveat into the surviving explanation.** It used to be
  a reason not to over-claim; it is now what is left after everything else was excluded on
  the Hannibal run — numerics (0.14%), the point gust (≤4.4%), the strip path (0.00%), the
  identified parameters (≤72.7% at the favourable corner) and gust amplitude at *any*
  strength inside the linear range. **Session 23d added the last one: an independent
  engine.** JSBSim on the same five-vortex field, with the same aircraft and the same
  starting state, reaches 74.6% of the record against AtiSim's 67.1% — *neither* reaches,
  so the solver is excluded on the headline field rather than by inference from the Parks
  cores. The aircraft is the residual, and closing it needs a DC-10 derivative set, which
  is the highest-value acquisition on the list.

  **A sealed prediction now bets against this entry.** `predictions.py`'s
  `dc10_does_not_close_the_hannibal_gap` says that acquisition will *not* close the gap,
  on the strength of §4's measured null that quadrupling mass moved the load 0.6% because
  wing loading cancels in `Δn = ΔC_L/C_L,trim`. If it is right, what survives is not
  aircraft type but the missing buffet nonlinearity, and this entry is right for the wrong
  reason. Do not settle it by editing this paragraph — settle it by flying a DC-10.

  > ### ⚠ **THE SIGN, CHECKED — session 26 — AND IT POINTS AGAINST THIS ENTRY**
  >
  > The wing-loading figure was flagged as "unverified and may be inverted", with the note
  > that "nothing downstream depends on the number". **The number is still unverified. The
  > DIRECTION is now measured, and everything depends on it.**
  >
  > §4's controlled sweep gives the load as **1/(W/S)**, slightly sub-inverse. So **lower
  > wing loading means MORE gust response**. The project's own two statements of the figure
  > agree with each other — §5 said the 747 has "roughly 0.8×" and `test_wind.py` says "the
  > DC-10's wing loading is roughly 1.3× the 747's, so the 747 takes MORE g for the same
  > gust" — and 1/1.3 = 0.77. **They also both point the wrong way for this entry.**
  >
  > Flown: `boeing747` at 1.3× its wing loading on Mehta's field reaches **56.4%** of the
  > recorded 2.70 g against the baseline's **70.1%** — peak-to-peak 1.5231 g against 1.8918.
  >
  > **A DC-10 with 1.3× the 747's wing loading would therefore be worse, by 14 points, not
  > better.** If that figure survives verification, aircraft type is not the surviving
  > explanation for the shortfall — it is a term with the wrong sign, and this entry is
  > wrong in a way that no amount of DC-10 derivative data would repair. (Peak |α| reaches
  > 9.24° in that run, close to the 10° linear limit, so read it as a direction with a
  > magnitude attached rather than a prediction.)
  >
  > **The sealed prediction is the beneficiary and its stated reasoning is not.**
  > `dc10_does_not_close_the_hannibal_gap` bets the DC-10 will not close the gap, resting on
  > "wing loading cancels" — which §4 now records as false. The bet looks **more** likely to
  > land, for a reason its author did not give. The entry is SEALED and has not been touched;
  > that is what the register is for.
  >
  > **What would settle it, and how precisely.** A sourced DC-10 wing loading — not weight
  > and area separately, since Bach & Parks 1987 Eq. (2) shows `m` and `S` enter their
  > identification only as `m/S`. **And it does not need to be precise:** their Eq. (4) gives
  > about **0.05° of α per 1% of `C_L`**, so 1% of wing loading is 0.05° of α against
  > excursions of 7–9°. Anything better than a few per cent buys nothing.
  >
  > **What not to do.** Parks Fig. 7 is the vortex array and Fig. 6 is the g trace for the
  > same encounter. **Do not fit `W/S` to reproduce Fig. 6.** That converts the project's
  > only end-to-end check into a calibration, and there would then be nothing left to test.
  >
  > **SESSION 27 WENT AND GOT THE NUMBER, AND IT DOES NOT SETTLE ANYTHING — because the
  > uncertainty was never in the wing area.** `presentation_package/_evidence/dc10_wing_loading.log`.
  > The modelled 747 is 288,773.2 kg over 510.97 m² = **565.15 kg/m²** (SOURCED, CR-2144
  > Table IX-3). Against it:
  >
  > | DC-10 | at OEW | at mid weight | at MTOW |
  > |---|---|---|---|
  > | **−10** (430,000 lb / 3,550 ft²) | 0.584× | **0.815×** | 1.046× |
  > | **−30** (555,000 lb / 3,647 ft²) | 0.631× | **0.973×** | 1.315× |
  >
  > **The project's "roughly 1.3×" is the bottom-right cell — the heavier variant at maximum
  > takeoff weight.** It is the extreme corner of the range, not a central estimate. At any
  > realistic cruise weight the ratio **straddles 1.0**, and for a −10 it is **below** it,
  > which by §4's own `1/(W/S)` law means a DC-10 would respond **more** than the 747, not
  > less. **So the session-26 sign reversal is not robust, and the sentence "the surviving
  > explanation is the aircraft" is withdrawn rather than replaced.**
  >
  > **Three things block pinning it, and none is the wing area.** (1) **The variant is not
  > identified** — TM-102186 and Parks both say only "a DC-10", and −10 versus −30 moves the
  > ratio by 26%. (2) **The weight at the encounter is recorded nowhere in any source held**,
  > and OEW-to-MTOW spans a factor of 1.8. (3) Even the *published* area disagrees across
  > secondary sources for the same variant — 3,647 ft² (Wikipedia, citing *DC-10 Airplane
  > Characteristics*) against 3,958 ft² (flugzeuginfo), an 8.5% spread. The manufacturer
  > document itself was fetched this session — Douglas **DAC-67803A**, *DC/MD-10 Airplane
  > Characteristics for Airport Planning*, reprinted January 1991, revision A April 2004 —
  > and it gives **Series 10 maximum design taxi weight 458,000 lb (207,746 kg)** but
  > **carries no wing area at all**, being a pavement-and-gate document. So the best available
  > weight is now SOURCED and the area is still only tertiary.
  >
  > **What would actually settle it** is therefore not a wing loading but **the flight**: an
  > NTSB or operator record identifying the aircraft and its weight that day. Absent that,
  > quote the ratio as a **range, 0.58–1.32×**, and do not let any conclusion rest on where
  > in it the truth sits.
  >
  > **Session 31 narrowed it by identifying the aircraft**: United 12, **N1809U, a DC-10-10**
  > (NTSB CHI81DA042; variant per JetPhotos, secondary). **The range is now the −10 row,
  > 0.584–1.046×**, and 0.815× at mid weight. The heavier −30 corner that produced "1.3×" does
  > not apply. The weight on the day is still unknown, so the ratio remains a range.
  > §7's flight-record row has the sources.
- **Half of the Fig. 8 load band is unreachable inside the linear range.** Read as an
  *absolute* load factor, the band's −1.9 g needs about 13.8° of elevator from trim and
  drives |α| to roughly 18.5° — half again past the 12° where §7 says this model reports
  lift the sources deny. Read as an *increment* it needs 8.926° and |α| 10.31°, which is
  marginal but flyable. §8 records the decision to fly the increment; what belongs *here*
  is that the choice was not free. This is the same ceiling as the ±g asymmetry seen from
  the other side: `CL = CL0 + CLa·α` has no stall, so the only way to reach a large
  negative load is a large negative α, and there is no aerodynamic mechanism to get there
  sooner. A run flown to the absolute reading would not be a harder test of the model, it
  would be outside it, and would prove nothing.

**Added session 32 — attributed, from the branch harvest:**

- **The phugoid this project REPORTS is not the phugoid its simulation FLIES, and both are
  right for their purpose.** `validation.longitudinal_modes` linearises in the 4-state
  `[u, w, q, θ]` — which is the correct comparator for CR-2144: its Appendix C gives the
  longitudinal system as 3×3 in `[u, w, θ]` with `q = sθ`, a quartic denominator, and altitude
  only as an *output*. But the simulation integrates altitude, and density varies with it, so the
  aircraft that is actually flown carries a fifth state and a height–density coupling the
  reported modes omit. **Measured on `claude/flight-dynamics-solver-oscillation-17139b`**
  (commit `e50ad31`), at 747 cruise:

  | | 4-state (reported) | 5-state `[u, w, q, θ, z]` | nonlinear rollout |
  |---|---|---|---|
  | phugoid ωn | 0.055319 | 0.066502 | 0.066657 |
  | phugoid ζ | 0.055956 | 0.027408 | 0.027628 |

  The rollout agrees with the 5-state to 0.23% in ωn and 0.80% in ζ, and the 4-state is
  **+20.5% / −50.6%** away from what is flown. **Those numbers predate session 30**, which
  declared CR-2144's speed derivatives and a thrust line on `boeing747` and moved its phugoid, so
  they describe an earlier aircraft and **may not be quoted for the shipped one**. The mechanism
  does not depend on the derivatives. **What it affects:** any claim about how the simulated 747
  behaves over phugoid timescales, which none of the CAT results are — Mehta's array is crossed
  in about 34 s, inside a 45 s run, against a phugoid period of 94–114 s on either reading. **What it does not affect:** every
  comparison against CR-2144, Caughey or the Navion, which are correctly 4-state. **FUTURE
  WORK:** re-measure on the shipped 747; the branch's `full_matrix` and `constant_altitude` are
  the instrument, recoverable from its SHA (§0).

## 6. Latent bugs — (a)–(d) fixed in session 5, (e) in session 7, (f) in session 29, (g) in session 32; (h) and (i) OPEN

Seven are closed; **(h) and (i) are open** — (h) measured and deliberately not fixed before the release, (i) found by phase V1 in session 33 and fixed as an API while its default is deliberately left alone. Kept here rather than deleted because the *shape* of (a) and (b) is
the thing worth remembering: both survived three sessions and a 209-test suite because
every test in the project was still air, and still air cannot distinguish airspeed from
groundspeed.

**(a) `autopilot.py` sensed inertial airspeed.** FIXED. `autopilot` and `engage` now take
`sensors.AirData` rather than `State`, so they cannot be handed inertial velocity — there
is no `vel_body` in scope to misuse. Verified by
`test_the_autopilot_holds_airspeed_not_groundspeed`: into a 25 m/s headwind the loop now
settles airspeed on target with groundspeed 25 m/s below, where it previously did the
reverse.

**(b) `viz.derived` computed incidence from inertial velocity.** FIXED. It now uses the
recorded wind. The old `test_derived_agrees_with_the_aero_module` fed both sides the same
input and structurally could not fail; the replacement,
`test_derived_is_air_relative_and_this_test_can_fail`, computes the expectation
independently. **Verified by re-introducing the bug**: exactly that test and (a)'s went
red, and nothing else moved.

**(c) `viz.Trajectory` recorded no wind.** FIXED, by recording rather than replay.
`SimState` now carries the wind the previous step applied, and `Recorder.append` takes the
whole `SimState`, so a run is self-describing. Replay was the cheaper option but needs the
caller to reconstruct the exact model and key; recording cannot be got wrong later.
`load` defaults the two new columns to zero, so `.npz` written before they existed still
open — honestly, since those runs were all still air.

**(d) 747 `Mq` transcribed as −0.330.** FIXED to −0.339. Short-period damping error
12.6% → 11.5%; everything else moved in the fourth decimal or not at all.

**(e) `panel.AlphaGauge` was one-sided.** FIXED in session 7. Introduced by session 6's
own re-layout: `set_xlim(0, 15)`, a needle clipped to `[0, 15]`, and a `state()` comparing
**signed** degrees. At α = −16° it pegged the needle at zero and reported `linear`.

`aero.py` is `CL = CL0 + CLa·α`, exactly odd-symmetric in Δα — the same property §5 blames
for the ±g asymmetry being unreachable — so **|α| is what decides validity, not α**. The
gauge could not see half of its own invalid range.

It is (a) and (b)'s shape one more time: right in the easy case. Every test drove the gauge
positive, because level flight and a pull-up both do; nothing pushed. Found by asking what
the manoeuvring case (§7 step 5) would display, not by a test failing — a pushdown drives
α negative, so the instrument would have said `linear` throughout precisely the run whose
entire job is to report whether the model stayed in range. Two tests now pin it: the band
by magnitude, and the needle position, because fixing `state()` alone would have left the
needle still lying.

### (f) The model was not differentiable in its own coefficients — FIXED session 29

**`aero.py`'s Prandtl–Glauert factor made every `jvp` and `jacfwd` through
`aero.coefficients` return NaN, for every aircraft that declares no reference Mach — which
is all of them but the compressibility entries.** No result this project has ever quoted was
wrong, and that is exactly why it survived: the NaN was in a *tangent*, never in a value.

```python
pg = jnp.where(ac.pg_mach_ref < 0.0, 1.0,
               jnp.sqrt(jnp.maximum(1.0 - jnp.minimum(ac.pg_mach_ref, PG_MACH_MAX)**2, 0.0)) / ...)
```

`jnp.where` evaluates **both** branches. `pg_mach_ref = -1.0` is the "undeclared" sentinel,
so the unselected branch computes `sqrt(1 - min(-1, 0.90)²) = sqrt(0)`. The forward-mode
tangent of `sqrt` is `du / (2·sqrt(u))`, which at `u = 0` is **0/0 — NaN for ANY tangent,
including a zero one.** The select discards the value; nothing discards the NaN.

**How it was found, and it is the argument for the NaN guard rather than for a review.**
`atisim/sensitivity.py`'s first directional derivative through `trim.residual` tripped
`conftest.py`'s `jax_debug_nans`, which has been on for the whole suite since session 5. The
probe scripts written the same hour did **not** trip it — they ran outside pytest, produced
finite elasticities, and agreed with central differences to 1e-10. **A NaN that is always
discarded is invisible to every check except the one that looks for NaNs specifically.**

**The repair is the standard double-`where`**: clamp the *unselected* branch's input to 0.0
before the `sqrt`. It is value-identical by construction — a declared reference Mach is ≥ 0
and passes through untouched, and the sentinel branch's value was thrown away either way —
and it was **proved** so rather than argued, two ways: the SHA-256 of `[trim, v̇_body, ω̇]`
is unchanged for all seven registry aircraft with a cruise condition, and the two frozen
bit-exact pins that fail on this platform fail with **byte-identical values before and after
the patch**. `test_aero.py::test_the_undeclared_prandtl_glauert_sentinel_has_a_FINITE_derivative`
is the negative control.

**A second instance existed and is now CLOSED — same session, after it was flagged.**
`airframe.elliptic_chord` — `c0 * sqrt(max(1 − (2y/b)², 0))` — is the same construction, and
`airframe.stations` uses `jnp.linspace(-b/2, b/2, n)`, so a tip station sits **exactly** at
±b/2 and the argument is exactly 0.0 there. **Confirmed by running it, not by reading it:**
a `jvp` seeded in `CLa` — a coefficient the chord does not depend on at all — raised
`invalid value (nan) encountered in mul` under the NaN guard, at every station count and
every loading shape.

Repaired with the same double-`where`, and value-identity proved the same way: the SHA-256
of every chord, at every station count (9/17/57) and every shape (elliptic/uniform/tapered),
for every registry aircraft, is **unchanged**. Two tests guard it —
`test_the_elliptic_chord_has_a_FINITE_derivative_at_the_wingtip` and
`test_the_tip_chord_is_still_exactly_zero`, the second being the value-side control that
would catch a repair returning `sqrt(1.0)` at the tip.

**The tip tangent must be exactly ZERO and not merely finite, and that is physics.** The tip
chord is zero for every aircraft — `2y/b` stays exactly 1 when `b` moves, and `c0` multiplies
an exact zero when `S` moves — so the derivative of the tip chord with respect to any
aircraft parameter is genuinely 0. The test asserts `== 0.0` for that reason: `isfinite`
alone would pass a repair that made the singularity finite-but-wrong. **What the repair does
not give correctly is `d(chord)/dy` at the tip**, which is genuinely infinite — that is the
same sqrt singularity `calibrated_lift_slope` blames for the 82.6% quadrature shortfall.
Nothing in this project differentiates the chord with respect to spanwise position, and the
docstring says so.

### What made (a) and (b) invisible

A still-air test suite cannot catch an air-relative/inertial confusion, because in still
air the two are the same number. Any future quantity with an air-relative and an inertial
form needs at least one test that flies through a non-zero wind field —
`test_sensors.py` exists for exactly that and nothing else.

### (g) `airframe.stations` built sample stations out to −∞, silently — FIXED session 32

**`loads.strip_model` gated on the tail arm and `airframe.stations` did not**, though
`stations` is the only place a `Stations` is built and therefore the one point every
sampling consumer passes through. For the registry entries whose source defines no `CLq`,
`effective_tail_arm = −Cmq/CLq` is a **division by zero**, so `arm` was `inf`, the
`linspace` ran to `−inf`, and **every fitted gradient came back NaN with nothing raised
anywhere.** A rollout would run to completion and report a NaN rms.

**Measured on the merged tree, rather than taken from the branch's own docstring: 5 of the
7 registry entries are refused, and the two causes are different problems.**

| entry | arm, chords | `CLq` | why |
|---|---|---|---|
| `boeing737`, `boeing737_approach`, `boeing747_jsbsim` | `inf` | **0** | the arm **does not exist** — division by zero, and no better estimator recovers it |
| `cessna172` | 0.8558 | 7.282 | arm **finite and outside the band**: `CLq` and `Cmq` disagree about what airframe they describe |
| `cherokee` | 1.2802 | 5.760 | as above |
| `boeing747`, `boeing747_approach` | 4.0241, 3.8519 | 5.945, 5.400 | pass |

**Fixed by `airframe.require_plausible_tail_arm`**, called from *both* `stations` and
`strip_model`, so a caller cannot get two different accounts of the same refusal depending
on which door they came in by. `_refusal` separates the two causes in the message, because
blurring them would send someone looking for a fix where there is nothing to fix.

**Why it survived**: the same shape as (a) and (b). The suite had no test that asked a
refused aircraft for stations, and a NaN rms **looks like a result**.

### (i) The autopilot's engage seed had the wrong sign on its pitch-rate term — FIXED, release 1.1

**Found by reading the parked WGS-84 branch, which had fixed it, and confirmed against `main`.**
`autopilot.engage` promises bumpless transfer: the first output equals the controls the pilot
was holding. The running loop is

    elevator_raw = -(theta_p*theta_err + theta_i*state - q_d*q)

so at theta_err = 0 the seed must be `(-elevator + q_d*q)/theta_i`. It read `- q_d*q`, leaving
an engage transient of exactly `2*q_d*q`.

**Why nothing caught it for the life of the project.** Every engagement test starts from trim,
and a flat-Earth trim has `q` identically zero, so the error is multiplied by zero. The roll
axis beside it has always carried the correct `+ p_d*p`, which is what the fix is measured
against. Reachable in the shipped code: `scripts/fly.py` hands over on a key press, at whatever
rate the aircraft is pitching at the time.

**Measured:** at q = 0.02 rad/s the first elevator output moved by the full surface rate limit,
6.0e-3 rad in one step, where the promise is zero.
`test_autopilot.py::test_engagement_with_a_body_rate_is_still_bumpless` asserts it at three
rates and fails on all three with the old sign; its roll twin passes with the old sign, which
is what isolates the axis.

### (j) The gust's α̇ dropped the transport term of d(wind_body)/dt — FIXED, release 1.1

**A wind constant in NED is not constant in body axes while the aircraft rotates.** By the
transport theorem `d(wind_body)/dt = C^T d(wind_ned)/dt - omega x wind_body`, and the function
returned only the first term, the field's own gradient. So a uniform steady wind — gradient
identically zero — produced no α̇ at all.

**It was invisible for the same reason (i) was: a zero multiplied it.** Every entry flown in
wind had `Cmadot` zero until `boeing747` declared one. The moment it did,
`test_verification.py`'s uniform-wind invariance test failed on 48% of its samples: a uniform
wind is a change of inertial frame and must not touch the attitude, which is exactly the error
§2 names, in the one place it could still hide.

**Where the fix went, and why not the obvious place.** Putting the term in
`wind.gust_alphadot` beside the gradient half works and was tried first. It is wrong for two
reasons, both measured. The wind is HELD across an RK4 step, and this term varies with attitude
and rate within the step, so holding it cost the scheme three orders of accuracy: a
gradient-free field went from **3.99 to 1.03**. And a caller can forget it — `alphadot_gust`
defaults to zero and a wind model may return four values instead of five — which is precisely
how it stayed hidden.

So `dynamics.derivatives` forms it, from the wind it is handed and the state it is evaluating
at. The field's gradient still comes from `wind.gust_alphadot`, because that is the only half
that needs a field. Per stage even when the wind is held, and impossible to omit.

**Measured after the fix:** the uniform-wind invariance holds to 1e-11 on the ordinary held
path, with the test's wind model returning four values, which is what a uniform wind should
return; fourth order is restored on a gradient-free field; and the wind-hold scheme error on
the Fig. 8 pitch excursion is unchanged in magnitude at ~1% at dt 0.02, halving with dt.

**One more thing it made visible.** The α̇ loop couples to the Mach derivatives: with `Cmadot`
declared, the plant-matrix difference that isolates CR-2144 Appendix A's Mach terms carries a
cross term of 5.2e-3 relative, against 5.7e-14 without it. Neither family is wrong; they are
not exactly separable, and `test_cr2144_speed_derivatives.py` now says so where it asserts the
separation.

### (h) The strip load path counts the gust's rolling moment twice — OPEN, measured session 32

**`vortex_viz.fly(strip=True)` adds the strip rolling moment on top of the point path's,
instead of replacing it.** Its docstring says strip "**swaps** the point-plus-gradient load path
for strip-integrated loads". The code does not: it sets `load_model = loads.strip_model(field, ac)`
and passes the same `field` on to `fly_from_state`, which still builds the default
`wind.field_model(field)`. That model's `omega_gust[0]` is the point path's equivalent roll rate
from `dw/dy`, and `loads.strip_increment` adds its integrated rolling moment without removing it.
`integrate.step` applies both. **The strip kernel's own verification is the evidence**: §4 records
it agreeing with "the equivalent-rate treatment for a linear gradient" to 1e-6 — which is the
physics the point path already carries.

**Found by `claude/linearisation-verification-bounds-b73868` (commit `cf538dd`) before session
24, and never merged.** It measured "both = point + strip to every digit, 1.82× at the shipped
station count and 1.996× converged", and noted it could not show because every field then had
`dw/dy` identically zero along its track. **Session 24 then added line vortices, which do not** —
and published the strip path's effect.

**Measured on `main`, session 32**, `scripts/strip_roll_double_count.py`: Mehta's five-core field
as line vortices, shipped `boeing747`, flown exactly as `scripts/lateral.py` flies it — whose own
run on the same tree prints the first two rows to three decimals, so the harness is the canonical
one:

| run | peak \|φ\| | against line |
|---|---|---|
| line — point path only | 18.067° | — |
| **line + strip — what `strip=True` does** | **21.209°** | **+17.4%** |
| **strip owns roll** — point roll rate removed, pitch and yaw gust rates kept | **17.614°** | **−2.5%** |
| no gust roll at all | 14.674° | 81% of line |

`lateral.py` shows the mechanism in its own columns: `|p_gust|` is **0.0887 rad/s with the strip
path off and 0.0888 with it on** — the point path's roll rate is fully live in the strip run.

**With roll owned by one path, the strip path's effect on peak bank is −2.5%, and that residual is
its known quadrature error, not new physics.** At the shipped 9 stations the strip integral
returns 82.6% of `Clp` (`ASSUMPTIONS.md` F5). The gust-roll share of the line run's bank is
18.067 − 14.674 = 3.393°; scaled by that deficit it predicts 17.48° for the corrected run, against
17.614° measured. **So, properly attributed, the strip path moves no reported number beyond its
own calibration error** — which strengthens session 28's verdict on it rather than overturning it.

**What it contaminates:** §4's "the strip increment's own contribution, isolated, is +22.9% of
peak bank" (and +17.2% on session 30's 747), and §7's finishing-plan phase 1, whose gate — "the
strip path moves a reported number for the first time" — was met by this double count. Both are
superseded in place. **It does NOT touch §1's headline or any longitudinal claim**: the strip path
is roll only, the headline flies the point path, and `n_z` max moves by 0.0020 g or less across the line,
double-counted and corrected runs (1.4806–1.4826).

**Not fixed, deliberately, a week before release.** The repair is a modelling decision about which
path owns roll, and `point_roll_removed` in the measuring script is one candidate for it, not a
reviewed change. **FUTURE WORK**, with the measurement above as its acceptance test. The work was rescued
from an uncommitted worktree at the session-28 audit and sat unreviewed on
`claude/zen-maxwell-1ad0a4` for three weeks — so the bug was found, fixed and then nearly
lost, which is rule 1b's case in one line.

**(i) `wind.field_model`'s `stage_sampled` mark was never read by anything, and
`vortex_viz.fly` had no way to pass the flag.** OPEN as a default; the API gap is FIXED.
Found by design phase V1, session 33 — the first thing in this project to compare a flown
run against an exact answer.

`integrate.step` takes a `stage_sampled` argument and re-evaluates the wind at each RK4
stage when it is true, which `ASSUMPTIONS` E4 measures as the difference between first and
fourth order through a spatially varying field. `wind.field_model` sets
`model.stage_sampled = True` on its output to mark it safe, with a comment saying *"this
attribute is read at trace time"*. **It is not read anywhere.** `step` gates on its own
argument; a `getattr` on the model would have closed the loop and there is none.
`test_verification.py` asserts the mark **exists**, which is why nothing went red.

**So every run ever flown through `vortex_viz.fly`, `fly_in_moving_air` or
`fly_from_state` took the first-order path, and the remedy E4's own closing paragraph
points at was unreachable from the harness that flies every §4 turbulence result.** Only
`scripts/sensitivity_assumptions.py` and `verification.fixed_control_refinement` ever
passed the flag, and both take it as an argument of their own.

**What it costs, measured two ways that agree.** V1 measured it on the gust transfer
function at the short period; E4 had measured it on the in-core Fig-8 Δθ. Different
quantities, different step sizes, the same ladder:

| | halving dt → | | | |
|---|---|---|---|---|
| **V1, \|H\| error at 0.1686 Hz** (dt 0.119 → 0.0148) | **+3.20%** | **+1.62%** | **+0.81%** | **+0.41%** |
| **E4, in-core Δθ** (dt 0.02 → 0.005) | — | −1.62% | −0.82% | −0.41% |
| V1, the same run stage-sampled | +0.00022% | −0.00013% | −0.00015% | −0.00015% |

Halving with dt is first order and is the hold's signature; the stage-sampled column does
not move at all. **V1 could not pass its 0.5% gate on the held path and passes it by 116×
on the sampled one.**

**Fixed as an API, NOT as a default.** `fly`, `fly_in_moving_air` and `fly_from_state` now
take `stage_sampled`, defaulting to **False**, so every §4 baseline flown through them is
bit-identical to what it was. Flipping the default would move every deterministic-field
result in the document at once, which is a decision for the record's owner and not an
incidental repair — §5's precedent, *"a decision, not a measurement"*. What changes today
is that the cost is a **declared** choice with a reachable alternative rather than an
invisible one. `atisim/gust.py` passes True for every run compared against a closed-form
answer, and only there.

**No published number moves.** E4 already priced the hold at −0.82% on the headline Δθ at
the published dt and already says the last two digits of 2.240° are scheme-dependent. This
entry does not change that price; it records that the escape hatch E4 offers was welded
shut. Pinned by
`test_gust.py::test_v1_fails_on_the_held_wind_and_passes_on_the_stage_sampled_one`.

## 7. Plan

> **THE ORIGINAL TEN-STEP PLAN IS BELOW AND IS ESSENTIALLY COMPLETE.** The plan the
> project is now executing is the five-phase one written at the end of session 23d, in
> answer to "what would make this a predictive tool?". It is recorded here because it is
> §7's job to carry it; sessions 24 and 25 refer to its phase numbers throughout.

### The finishing plan — phases, gates and status

The decision it rests on: **"predictive tool" is three different projects.** **T1**
comparative and mechanistic, already true and — until session 24 — unclaimed. **T2**
bounded absolute load, months and source-gated. **T3** CAT hazard including roll, which
needs the lateral capability built from nothing. The chosen route is *bank T1 now, target
T2, build T3's foundation on the way*.

| Phase | What | Gate | Status |
|---|---|---|---|
| **0** | Bank what is already true: a formal validation claim in §1 with its envelope attached, and the lateral gap written into §5 | §1 carries the claim; §5 and `ASSUMPTIONS.md` E10 carry the gap | **DONE, session 24.** Writing the claim is what found the gap |
| **1** | Give the model a lateral dimension: Dryden `u`/`v`, lateral channels on `Encounter`, the vortex as lines in space | the strip path moves a reported number for the first time | **DONE, session 24** — the lateral dimension is real: line vortices take peak bank from 0.000° to 18.07°. ~~+22.9% on peak bank, after nine sessions of exactly 0.000000~~ **The gate itself was met by a double count — session 32, §6(h).** With roll owned by one path the strip path moves peak bank by −2.5%, its known quadrature error. **The strip path still moves no reported number beyond its own calibration** |
| **2** | Change what counts as agreement: response spectra instead of peaks, exceedance distributions over ensembles | a load-exceedance curve with N in its denominator becomes sayable | **DONE, session 25.** §4 has both. The comparison against a *published* curve is now the open half, and it is source-gated |
| **3** | Acquire four documents, in priority order | each arrival settles a sealed prediction or closes a §5 entry | **In progress, session 25 searched for all four** — see the table below |
| **4** | Keep the predictive discipline running: every new capability ships with a prediction made before it is tested | at least one sealed prediction settled, right or wrong | **Two of three settled, both right.** `the_dryden_response_peaks_at_the_short_period` (session 25) and `mil_f_8785c_sigma_w_exceeds_the_mehta_ceiling` (**session 27**, by 0.34 m/s with the merge band clearing it). The one still sealed is the DC-10 entry, which bets *against* the project's own story and is now the only open bet |

**Phase 3's four documents**, in the priority the plan gives them, with what each unblocks:

| # | Document | Unblocks |
|---|---|---|
| 1 | a **DC-10 cruise derivative set** | **STILL OPEN — the only genuine acquisition, and now worth twice as much.** Parks identifies *both* cases as DC-10s, so one set serves Hannibal **and** Morton. Weight Morton lower: Parks calls its fit "not as good as case 1" and blames mountain-wave contamination of the short-period pattern. **But check §5's sign first** — at 1.3× wing loading the load falls to 56.4% of the record, so if the DC-10 is the heavier-loaded aircraft this acquisition makes the shortfall worse, not better |
| 2 | **747 buffet onset / nonlinear C_L** | **SPLIT, session 25 — half of it was already held.** The buffet-onset BOUNDARY is on `refs/NASA-CR-114494.pdf` p. 2.0-38, the same sheet session 21 digitised `C_Lmax` from, with revised data in its §19: not an acquisition, a digitisation. The nonlinear lift curve is not published there or anywhere reachable, so the ±g asymmetry stays structurally impossible. **Acquire nothing; digitise the second curve** |
| 3 | **MIL-F-8785C Fig. 7**, digitised | **DOCUMENT HELD** (`refs/MIL-F-8785C.pdf`, session 25); Fig. 7 confirmed on printed p. 49. The digitisation is still to do. **Session 26 adds a caution about the check:** AFFDL-TR-70-101 (HICAT) is now held and measures the same quantity, but its band is 45,000–70,000 ft against this project's 35,000–45,000 (**corrected session 33**), and it compares itself against MIL-A-8861A and Steiner's NASA U-2 data — so Fig. 7's high-altitude end may descend from the same aircraft and would not be an independent check. Settling that needs ADA119421, ~~not held~~ **held session 31, and it points back to AFFDL-TR-69-72 and Av.P. 970 without naming HICAT (§3's HICAT row)** |
| 3 | **Yoshimura 2023 figshare dataset (21152203)** | **HELD session 25, verified** — CC BY 4.0, `data.tar`, 17,942,056,960 B, md5 `d23cbb3c77b3940653a0b643147d71c3` matching figshare's stated checksum, in `UROP/yoshimura-figshare-21152203/` outside the repo with a `PROVENANCE.txt` beside it. Five nested bz2 archives: the four LES domains (2.3, 3.6, 4.0 and **7.5 GB** for dx = 500/250/70/**35** m) and **`flightsim-data.tar.bz2`, only 561 MB** — Yoshimura's own 2-D B787 simulation code and its outputs, i.e. the SIMULATED half of their Fig. 6. That last one is the cheap one and supports a cross-code response-spectrum comparison the way JSBSim serves the vortex work. A published LES CAT wind field: the first field in this project not identified from the aircraft's own accelerations. **It does NOT carry a recorded acceleration history** — the three onboard records and the PIREP are withheld under confidentiality — so it does not unblock the observed half of the spectral protocol, which is what this row used to claim |
| 4 | **Bach & Parks 1987**, J. Aircraft 24(11) | **HELD, session 26 — and it does NOT help #1.** Its two validation cases are an L-1011 and a B-747SP; no DC-10. What it does give is the error budget: Eq. (2) shows `m` and `S` enter only as `m/S`, and Eq. (4) gives ~0.05° of α per 1% of `C_L`, which is what sets the tolerance on #2 |
| 5 | **Parks et al. 1985**, J. Aircraft 22(2) | **HELD, session 26 — §8's open question CLOSED.** r₀ = 600 ft with V₀ 85 and spacing 3500, and an abstract giving core *diameters* 900–1200 ft that removes radius-versus-diameter entirely. Reversed the 500 ft this project flew from session 22 to 25 and deleted the hybrid; `ASSUMPTIONS.md` E12 carries the four lineages. Cost 4.26% of the headline load, downward |
| 6 | **AFFDL-TR-70-101** (HICAT), Ashburn, Waco & Melvin 1970 | **HELD, session 26.** Measured RMS gust exceedances from U-2 flights — the published exceedance data phase 2 lacked. **Band mismatch is the catch**: 45,000–70,000 ft against this project's 35,000–45,000 (**corrected session 33 — the two now ABUT at 45,000 ft rather than being separated by 4,000**), so any comparison is extrapolated, and it may share its U-2 lineage with Fig. 7 |

**~~Phase 3 is entirely acquisition and this project cannot do the acquiring.~~ Sessions 25
and 26 emptied most of it, and almost none of what was left was an acquisition.** Session 25
searched and found that three of the original four were digitisations of documents already on
disk. Session 26 received four more papers and closed items 3, 4 and 5 outright.

**What is left is one document and three digitisations:**

| Still needed | Kind |
|---|---|
| ~~a **DC-10 wing loading** (`m/S`, to a few per cent)~~ | **ATTEMPTED AND CLOSED AS UNPINNABLE, session 27.** Not for want of a wing area: the variant is unidentified and the encounter weight is recorded nowhere held, so the ratio is a **range 0.58–1.32×** and cannot be narrowed by another specification sheet. §5 carries the table. **What is left is not an acquisition but the flight record** — an NTSB or operator document naming the aircraft and its weight |
| a **DC-10 cruise derivative set** | acquisition, and **§5's sign no longer argues against it** — session 27 withdrew that sign rather than confirming it. **PARTLY HELD, session 30: NASA CR-3677** (Shollenberger et al., McDonnell Douglas, March 1983, "Results of Winglet Development Studies for DC-10 Derivatives" — *derivative aircraft*, not stability derivatives; NTRS 19850002628), `Reference_papers/NASA-CR-3677-Shollenberger-1983-DC-10-winglet-development-19850002628.pdf`, md5 `a984a78c…c0`. **What it has**, baseline and with winglets, from a rigid 3.25% DC-10-30 model in the Ames 11-ft tunnel (M 0.60–0.95, Re 6.0e6/ft, stability axes, moments about 25% MAC): **Fig. 23**, tail-off C_L vs α at M 0.82, with buffet C_L 0.74; **Fig. 5**, C_mCL vs Mach tail-on and tail-off; **Figs. 7–9**, tail-on C_m vs α at M 0.60/0.80/0.90; **Fig. 6**, tail-off C_m0; **Figs. 10, 14, 17**, C_lβ, C_nβ, C_Yβ vs Mach. **What it lacks**: C_mq, C_mα̇, cruise control power, the cruise drag polar, mass, inertias, CG, stabiliser setting, and the reference S and c̄ (not printed). **Its limits**: a rigid model at model Reynolds number, with no aeroelastic correction — the size of that correction is not held, and it is the same order as the effect a DC-10 set is wanted for. The Hannibal variant is unidentified; this is a -30. **Not digitised; nothing flown** **FINAL, session 32 — IMPOSSIBLE WITH SOURCES HELD, every held DC-10 or MD-11 document now assessed.** CR-3677 is partial (above); CR-3748 has weights, a buffet boundary and column-force stability but no derivatives (session 31); and the three not assessed until now hold none: **NASA TM-4745** and **TM-1998-206552** (Burcham, MD-11 and 747-400 propulsion-controlled flight) describe phugoid and thrust-line effects qualitatively and plot flight time histories, with no derivative table, mass, inertia or reference geometry; **Taylor 1978**'s "DC-10 derivatives" are derivative *aircraft* — winglets and nacelles — the same trap as CR-3677's title. So the sealed prediction `dc10_does_not_close_the_hannibal_gap` stays **SEALED and unsettleable with sources held**. It is not edited. One note sits here rather than in it: the 747's peak-to-peak has moved since sealing, 1.8385 g then and about 2.03 g now (75.3% of 2.70 g), but the claim's band is absolute, **[1.563, 2.114] g**, so it stays decidable exactly as written. |
| ~~**MIL-F-8785C Fig. 7** at 33–41 kft~~ | **DONE session 27** — `scripts/digitise_mil_f_8785c_fig7.py`; σ_severe(37 kft) = 4.80 ± 0.12 m/s; settles a sealed prediction RIGHT |
| ~~**CR-2144 printed pp. 220–222**, the 747's derivative-vs-Mach curves~~ | **DONE session 30** — `scripts/cr2144_speed_derivatives.py`, checked against Table IX-4 at eight conditions; §4 has the entry. Session 28 called this the highest-value item on the list. **What is left from it is a DECISION rather than work**: whether `boeing747` declares the FC9 speed set, which moves the shipped phugoid, the Fig. 8 pins and the CAT headline |
| ~~**TM-102186 Fig. 6**, the recorded g trace~~ | **DONE session 27** — `scripts/digitise_tm102186_fig6.py`; and it moved two numbers, see §4 |
| ~~747 buffet onset boundary~~ | **DONE session 26** — `aircraft.buffet_cl` |
| ~~**compare the two independent digitisations of CR-2144 pp. 220–222**~~ | **MEASURED, session 32 — §4, "Two readings of CR-2144".** The sealed prediction was WRONG on `Cm_M` (2.40% of full scale at 20K), but scored against Table IX-4 the automated trace is the biased one: **on all three declared speed derivatives the hand reading is the better-anchored**, `Cm_M` by 2.3×, `CL_M` by 4.9×. The first check of the declared values from outside their own reading, and it supports them. *Was:* **NEW, session 32, and it is the cheapest item on this list.** Session 30 read pp. 220–222 by hand through Engauge (`atisim/data/cr2144_dig/`, 295 placed points). `Reference_papers/CR-2144/`, found untracked this session, read pp. 218–228 automatically at 300 dpi with per-panel RANSAC tick calibration. **Neither has been compared with the other.** §4's session-30 entry prices the hand reading by interpolation, leave-one-out, a pixel Monte Carlo and the Table IX-4 residuals — all of which are *internal* to one reading. A second independent trace of the same ink prices it from **outside**, which nothing else on this list can do, and it needs no new source and no new method. Read the automated set's own confidence column first: it marks `CL_M` **poor** |
| **the α̇ derivatives — `Cm_adot` from p.221, and the `Zwd` sign** | **NEW, session 32.** Two routes now exist and they should be run together. `claude/engine-validity-presentation-1408e8` (§0) restores Table IX-4's `Mwd` as `Cmadot` and argues `CLadot`'s sign is unsettleable from IX-4 or IX-5; ~~`Reference_papers/CR-2144/csv/p221_Cm_adot_*.csv` is the printed **curve** for the same derivative, across Mach.~~ **CORRECTED, session 32: the automated `Cm_adot` curve must not be used for this.** Scored against Table IX-4 it is 9.5× further off than the hand reading (RMS 0.638 against 0.067) and misses FC4 by 1.41 — a wrong curve. **The hand reading in `atisim/data/cr2144_p220_222_digitised.csv` carries `cm_alpha_dot` and is the one to schedule from.** The curve cannot settle the `Zwd` sign either — it is `Cm_adot`, not `CL_adot` — but it is what would turn a single tabulated value into a Mach schedule, and **its own README marks that panel `poor`**: the three altitude curves print within a line width of each other |
| **the Hannibal flight record** (operator, tail, weight) | **FLIGHT IDENTIFIED, session 31 — weight still not found.**
- **The flight is United Airlines Flight 12**, DC-10, Los Angeles → Newark, 3 April 1981, near Hannibal, MO. 20 passengers and 9 crew were injured, and it diverted to O'Hare. Source: NTSB Safety Recommendation letter A-84-108. NASA-CR-203832 (Lester & Chan 1996, the Ames incident table) lists it as "4/81 UA012 … DC-10 … 37,000".
- ~~**NOT found: registration, variant (−10 or −30) and gross weight.**~~ **Registration and variant FOUND, later in session 31; weight still not found.**
  - **NTSB case CHI81DA042**, a Director's Brief (CAROL Mkey 127770): Douglas DC-10, **N1809U**, Hannibal, MO, 3 Apr 1981, highest injury "Serious". Read from the CAROL query API.
  - CAROL's report generator does not serve pre-1982 briefs: it returns "The case with MKey 127770 does not exist".
  - **N1809U is a DC-10-10**, per two JetPhotos records (1972 and 1991). This is SECONDARY: the FAA registry refused automated access (403).
  - **This pins the variant, and §5's wing-loading ratio narrows from 0.58–1.32× to the −10 row: 0.584× at OEW, 0.815× mid, 1.046× at MTOW.**
  - DC-10-10 MTOW of 430,000 lb is now SOURCED from NASA CR-3748 p. 57, a Douglas flight-test report on a Series 10.
  - **Not tried:** the NTSB pre-1982 database (39 MB Access file). Whether it carries gross weight was not checked, and no Access reader is installed here.
- **NASA held both the weight and a DC-10 aerodynamic model and published neither.** TM-102186 §2 computes C_L "using the aircraft weight", with "C_L(α, δ_f), C_Lα and C_Lδe … based on the aircraft aerodynamic characteristics". The method paper (Parks, Bach & Wingrove, SFTE Symposium, New York, Sept 1982) is not online.
- **Routes:** the NTSB brief or docket for that date (registration, load manifest); the 1982 SFTE paper; and NASA CR-3748, *DC-10 Winglet Flight Evaluation* (McDonnell Douglas, 18.6 MB, not yet read), for flight-measured DC-10 aerodynamics.
- *Was:* **NEW, session 27.** The only thing that would pin the wing-loading ratio, and therefore the only thing that would let the aircraft-type explanation be tested rather than argued |
| ~~four B787 numbers — `m`, `S`, `c`, `I_yy`~~ | **OBTAINED session 27, and three of the four are CHECKED rather than looked up.** m = 215,910 kg, S = 325.3 m² (trapezoidal), c̄ = 6.437 m, all confirmed by inverting their own `Z_a` to a physical `C_Lα` = 4.847. `I_yy` is DECLARED and provably **unobservable** — it cancels against `C_mα`. §4 has the table. **This is now an implementation task, not an acquisition** |
| ~~**build the `boeing787_yoshimura` registry entry, then re-fly the LES**~~ | **DONE, session 27 — this row was stale for five sessions.** `atisim/aircraft.py:1798`, flown through D03 against Yoshimura's own ensemble: the ratio goes **1.427 → 1.202**, closing **53%** of the discrepancy, with no frozen-slope error at all because its derivatives are tabulated at M 0.406. §4, "The LES comparison", has the table. Found stale in session 32 by reading §4 before planning from this row. *Was:* ~~**NEW, session 27, and it is the top of the list.**~~ All inputs are in hand. Give it a `valid_mach`/`valid_altitude` band around its own condition. Carry the known difference that AtiSim has no `M_α̇` where Yoshimura has −0.137 |
| ~~**test the frozen-`C_Lα` explanation of the LES ratio**~~ | **DONE, session 27**, in the same §4 table: the full Prandtl–Glauert-corrected 747 reads **1.225 (47% closed)** and lift-only **1.135 (68%)**. The Mach axis is a partial explanation, not the whole one — which is why the residual is recorded as attributable to neither aircraft nor Mach. Found stale in session 32. *Was:* ~~**NEW, session 27, and it is CHEAP.**~~ Rescale the 747's `C_Lα` by the Prandtl–Glauert ratio 1.521 and re-fly D03/D04. If the 1.42× ratio collapses toward 1, the LES discrepancy is this project's frozen derivative and **not** a code disagreement — and it becomes the **first quantified point on the Mach axis** ASSUMPTIONS C3 has left unbounded since session 12, with no chart read needed |
| ~~run the LES limb~~ | **ALREADY RUN, session 3–4 Sept, and found NOT LIKE-FOR-LIKE in session 27.** All four domains × two aircraft are on disk. §4's input audit says why no number from them is quoted: the aeroplane is 5.4% or 82.9% away in natural frequency, and the entry closest in frequency is 2.63 band widths outside its own envelope. The **field reader is sound** (+0.978/−0.968/−0.935 against their own sampled wind) and reusable; the **load comparison is not yet a comparison.** This remains the project's only route out of the circularity every load row carries |

### The retrieval ledger, harvested from an abandoned plan — session 32

> **Recovered from `claude/cat-flight-model-dossier-04adb9`**, an unmerged plan dated
> **31 August 2026** that §0 listed only as "a plan on this conversation's own subject". The
> plan itself is **superseded** — its Phase 0 argues the 500 ft core radius that session 26
> reversed, and it states "Parks 1985 has never been obtained", which session 26 disproved by
> obtaining it. **Its retrieval ledger is not superseded, and five of its seven rows appear
> nowhere else in this document.** The branch is abandoned (§9); this table is what was taken
> from it.

| Document | What it would close | Status here |
|---|---|---|
| ~~**14 CFR 25.341**, at the amendment in force~~ | **READ, session 32 — and it cannot do what this row hoped.** Current text, **Amdt. 25-141, 80 FR 4762, 29 Jan 2015**. **(a) Discrete gust:** reference velocity **56.0 ft/s EAS** at sea level, linear to **44.0** at 15,000 ft, linear to **20.86** at 60,000 ft — a **peak**, in **EAS**. **(b) Continuous turbulence:** `Uσref` **90 fps TAS** at sea level, linear to **79 fps TAS** at 24,000 ft, then **constant at 79 fps to 60,000 ft** — an RMS in TAS, which is Fig. 7's statistic, but a **regulatory design limit** that enters a load through a transfer factor, **with no exceedance probability and no altitude variation anywhere in 35–45 kft (band corrected session 33; the conclusion is unchanged, since 79 fps is constant from 24,000 ft to 60,000)**. So it cannot be placed against Fig. 7's severe curve (σ = 4.80 m/s at 37 kft; `Uσref` is 24.08 m/s, a different quantity), and **it cannot test Fig. 7's U-2 lineage. IMPOSSIBLE WITH SOURCES HELD for that purpose**; settling the lineage still needs ADA119421. **The discrepancy this row warned of dissolves on arithmetic**: the current law evaluates to **26.0 ft/s EAS at 50,000 ft** — the "older" figure quoted — so the two texts are one linear law extended in 2015, not two values. The older text itself was not re-read. Read from Cornell LII's rendering of the eCFR; ecfr.gov redirected to a bot check, which was not bypassed. *Was:* **Turbulence intensity against altitude, tabulated to 60,000 ft.** The plan calls it "the one fully-open document" for t… |
| **NASA TP-2469**, Campbell | von Kármán rational-filter validity — i.e. whether the Dryden form this project uses is defensible where a von Kármán spectrum is the better model | **Not recorded anywhere.** NTRS, so free |
| **Sharman et al. 2014** | The **EDR-to-σ bridge**. EDR is the operational turbulence metric; this project reports σ_w and `σ_n`, and has no route between them. That is why no result here can be stated in the units an airline or a forecaster uses | **Not recorded anywhere.** This is a *capability* gap, not a data gap, and it is the one that would make the model's output comparable to operational practice |
| **MIL-STD-1797A** | The military alternative to 14 CFR 25.341 for the same table | **Not recorded anywhere.** Limited distribution — which is exactly why 25.341 is worth trying first |
| Etkin 1981 | Four-point gust model coefficients | Already cited once in `ASSUMPTIONS.md`; the *four-point* use is not |
| Doyle et al. 2011 / T-REX | N and cross-mountain U at 12 km | Held; `scripts/leewave.py` already flies it |
| Parks et al. 1985 | Per-case vortex table, Scorer ratio | **HELD since session 26.** The plan's claim that it "has never been obtained" is what dates the plan |

### Hannibal comparisons beyond the peak load — inventory, session 30

**Asked directly: what else do Mehta 1987 and TM-102186 hold that the model could be marked
against?** Until session 30 the project compared the peak `n_z`, the vertical-wind extremes,
the 5 s gust spacing and Fig. 8's orderings. Both documents were re-read page by page; this
is what else they carry, in the order it is worth doing.

| # | Data | Where | What it tests | Status |
|---|---|---|---|---|
| 1 | **horizontal (along-track) wind**, data and model | Mehta Figs. 5, 9; TM-102186 Figs. 6–7 top panels | **which side of the path each core sits on** — the vertical wind is identical for a core the same distance above or below, only the horizontal flips, and `wind.py`'s sign of z is argued from Mehta's prose, never checked against his plotted curve; and how much along-track gust the fitted field misses. ~~Newly consequential: `Cm_M` turns along-track gust into pitch~~ **Always consequential, through q̄: 13.6 points of the record on the bare entry. Cm_M adds sensitivity to the side** | **DONE, session 30 — §4 "The Hannibal horizontal wind".** Sign confirmed (1.58 kt against 15.38 flipped). The fit misses a ~15–20 kt tailwind rise. Found: the headline double-counts the 747's climb past cores 3–4 (64.5% → 75.0% replayed) |
| 2 | vertical wind, full trace | TM-102186 Fig. 6 | an unmodelled-wind residual read straight off the trace, independent of Mehta's unstated `N` (§8) | open; the digitisation pipeline already reads the panel |
| 3 | `n_z`, full time history | TM-102186 Fig. 6 | peak timing against the cores; the post-encounter ringing, as a band against a Dryden ensemble | open; already digitised, only extremes used |
| 4 | Mehta's **2-vortex** fit | Mehta p. 29, all parameters printed | how much the headline load depends on the identified field's form | open; one flight |
| 5 | Fig. 8 full pitch and `n_z` curves | TM-102186 Fig. 8 | where along the vortex each extreme falls, at ordering level | open; six curves to digitise |

**Not in either of those two documents:** pitch angle, pressure altitude, airspeed and
elevator. TM-102186 Table II says the DC-10's recorder logged all four, but neither paper plots
them. ~~Parks 1985 has not been checked for them.~~ **Checked, session 30, and it has three of
them:** Parks et al. 1985 Fig. 6 plots the Hannibal encounter's normal acceleration, **pitch
angle, true airspeed**, air temperature and **altitude** against time (p. 126). Elevator is
still unplotted anywhere held. **This is the largest unused source the project has** — it is
the aircraft's own response, not the wind — and it is item 6 below.

| 6 | **pitch angle, true airspeed, altitude** time histories | Parks et al. 1985 Fig. 6 | the aircraft's RESPONSE channels, beyond load. True airspeed is what an along-track gust acts on. **Since item 1, also the DC-10's own ALTITUDE through cores 3–4**, which decides whether the headline should fly Mehta's field on the 747's path or the nominal one (§8) | **ALTITUDE DONE, session 30** — §4 "The DC-10's altitude through the vortex pair": the DC-10 did not climb before cores 3–4, so the replayed form is right. **Pitch and true airspeed still open**: true airspeed is read only for its pre-encounter median, and pitch not at all. `scripts/digitise_parks_fig6_altitude.py` already calibrates both panels |

Parks p. 125 also fixes a sign the project had only argued: the DC-10 was **"cruising in an
easterly direction"** with the jet stream at **240°**, so the ambient wind blew toward about
060°, **~30° off the path from behind** — consistent with Mehta's ψ = 31°. A positive
horizontal perturbation is therefore a **tailwind** gust, which is the direction
`vortex_wind` adds it in.

### The sensitivity study — designed session 29, not yet run

> **ALL SEVEN PHASES RAN IN SESSION 29.** The table below is kept as written, with each
> row's outcome recorded in place. §4 has two measured entries covering all of it;
> `ASSUMPTIONS.md` C3 is no longer unbounded; §1's headline carries a band.

**The question it answers is one §4 has never asked: which of this model's own numbers does
the answer rest on?** §4 measures the model against sources. It does not measure the model
against *itself*, so no row in it says whether the headline load is set by `CLa` or by `Cmq`,
nor whether a modelling choice from `ASSUMPTIONS.md` costs more than a derivative does.

Full design in `docs/design/specs/2026-09-10-model-sensitivity-analysis-design.md`.
**Scope was agreed before it was written**, and what it excludes is as deliberate as what it
covers — the wind and scenario inputs are excluded **because `cat_bounds.py` and
`cat_uncertainty.py` already price them**, and the budget cites those numbers rather than
re-measuring them.

| Phase | Work | Gate | Status |
|---|---|---|---|
| **S0** | `atisim/sensitivity.py` + `test_sensitivity.py` | the differentiable load QoI matches `vortex_viz._measure`'s `n_z` to **1e-12**, sample-for-sample | **DONE for the MODE path, session 29** — 24 tests. The load QoI and its 1e-12 gate move to S2, because §6(f) had to be fixed first before anything could be differentiated at all |
| **S1** | AD screen on the cruise modes | ~~reproduces session 11's four sweep slopes to <2%~~ **the gate as written was the wrong test and is corrected here**: an AD tangent and a least-squares slope over a 3–8× range are different objects, so the machinery gate is AD against a **central difference at the same point** | **DONE, session 29. PASSED at 5.2e-10 against a 1e-6 gate.** §4 has all four sections, including the proof that the 0.6–4.2% tangent-vs-fit gap is curvature |
| **S2** | AD screen on the headline CAT load, all of tier A | AD elasticity vs ±1% central difference agree to **<1%** | **DONE, session 29.** Gate passed; the load path is **bit-identical** to `_measure`, better than the 1e-12 asked. `CLa` +0.692, `mass` −0.649, **`kappa_airfoil` −0.339 — a DECLARED constant in third place**. *Session 31: the first two are one lever, and `mass` was moved with inertia fixed (§4, first entry)* |
| **S3** | OAT confirm on the top factors at ±1/5/10/25% | a ranked table carrying the excursion it was ranked at | **DONE, session 29.** **Three of the six top fields lose the tangent's sign or magnitude by ±10%** — wave-drag saturation, and the extreme jumping to a different core. The ranking is valid at ±1–5% and the table says so |
| **S4** | Tier B — the `ASSUMPTIONS.md` modelling choices, on the same axis | **C3 gets its first BOUND at cruise** | **DONE, session 29. C3's Mach axis: −5.04%** over a measured Mach span of 0.7187–0.8257. `ASSUMPTIONS.md`'s C3 row and summary table both updated. dt found worth +0.22% by **sample placement, not integration order** |
| **S5** | Dryden ensemble `n_z` rms elasticity | N stated with the result | **DONE, session 29, N = 8 with the per-seed spread beside every mean.** `CLa`/`mass` survive the change of statistic; **`c` and `Cmα` REVERSE SIGN** and `Clb` goes from an exact zero to third. A ranking is a property of the statistic too |
| **S6** | The banded budget: RSS and linear-sum brackets on the headline | §1 carries a band, **or** states that the shortfall survives every band | **DONE, session 29 — and it is BOTH.** §1 now carries **68.2%, band 57.1–74.0%**, and states that the shortfall survives all of it. The 32% is not input ignorance |

**Two limits the write-up must carry every time it quotes a ranking.** The ranking is
**one-at-a-time and measures no interaction terms** — global variance-based methods (Sobol,
Morris) were considered and declined, because the input distributions they need are not
sourced and a variance decomposition invites a reading this project cannot support. And
**B1 stays unquantifiable**: varying a rigid-body parameter does not bound a structural one.

### The original ten-step plan

```
1. [DONE] Rankine vortex array, cited to Parks 1985       -> verify: source's own three
                                                                    stated properties
2. [DONE] Updraft column, declared edge sharpness         -> verify: Δθ(updraft) > 1.5×Δθ(vortex)
3. [DONE] load_factor + air-relative analysis figure      -> verify: corr(n_z, α_air) > 0.999
7. [DONE] Fix latent bugs (a)-(d)                         -> verify: DONE, by re-introducing
   (was step 7; done early because 4 and 6 both                     (a) and (b) and watching
    depend on the sensing being right)                              exactly their tests go red
4. Dryden background layer                                -> verify: sample σ to rel 0.10;
   (needs MIL-F-8785C Fig. 7 σ at 40 kft DIGITISED;                AR(1) pole = exp(−V·dt/L)
    forces init_sim/batch_sim to be parameterised)
5. [DONE] Manoeuvring case: elevator pushdown to        -> verify: DONE, 30.37 deg against
   Δn = −1.9 g at zero wind, elevator bisected                     the updraft's 4.37 and the
                                                                   vortex's 2.24; ordering holds
6. [DONE] Fig. 8 with ensemble error bars                  -> verify: DONE, 16/16 members at
   (session 23d; sigma_w is the SOURCED range from                  both ends of the sourced
    Mehta's residual, not a picked number)                          sigma_w range. The margin is
                                                                    the finding: the PITCH gap
                                                                    collapses 0.886 -> 0.060 deg
                                                                    and only the LOAD axis still
                                                                    separates the categories
9. [DONE] Microburst + averaged F-factor                   -> verify: DONE, 1 km average F
   (not in the original plan; the lee wave                          +0.193 against the Cherokee's
    left half of the F-factor untested)                             +0.078 of thrust, and the
                                                                    shear term is no longer zero
8. [DONE] Mountain lee wave + F-factor                     -> verify: DONE, and sharper than
                                                                    asked: peak F +0.0262 on
                                                                    Doyle's south leg exceeds
                                                                    +0.0234, +0.0129 on the
                                                                    north leg does not
```

Step 10 (session 11, not in the original plan): **verify the solver before adding layers.**
DONE — §4's three new subsections. Prompted by external review, which asked for validity to
be established first and for an interface where a known coefficient change gives a known
result. What it did **not** cover, deliberately:

- **No new aircraft.** CR-2144 documents **ten** — NT-33A, F-104A, F-4C, X-15, HL-10,
  **Jetstar**, **CV-880M**, B-747, **C-5A**, XB-70A — each with derivatives,
  transfer-function factors and handling-qualities parameters under the same Appendix
  A/B/C conventions this code already implements. The cheapest next two are the
  **Jetstar** (Table VII-1, power approach, non-dimensional, **body axis**) and the
  **C-5A** (Table X-1, same form, **stability axis**, so `stability_to_body` applies) —
  both in the identical form to Table IX-2, which session 10 noted "involves no conversion
  chain at all".
- **No further 747 flight conditions.** CR-2144 pp. 229–236 are scanned line-printer
  output whose text layer OCRs to noise, and the cruise non-dimensional derivatives are
  published as **plots against Mach**, not tables — which is why session 1 had to recover
  them from the dimensional set. Reading more is the same error-prone eye-work as adding a
  new aircraft.

~~Step 5 is done, so **step 6 now waits only on step 4** (Dryden), which is where the
ensemble spread would come from — the three deterministic points have no spread by
construction, since every member of a batch meets the same field.~~ **Step 4 landed in
session 23b and step 6 in 23d.** The spread came from `wind.dryden_vertical_field` at the
σ_w range session 23c derived from Mehta's own fit residual, so the ensemble's intensity is
sourced rather than chosen. §4 has the result.

### Extensibility: what the next wind model will cost

Everything so far is a **deterministic, position-only field**: `field_model(field)` wraps
`pos_ned -> wind_ned` and derives `omega_gust` from the analytic gradient. Dryden is not
that, and the gap is where the work is.

| | Deterministic field (vortex, updraft, lee wave) | Stochastic filter (Dryden, von Kármán) |
|---|---|---|
| Depends on | position only | its own previous output |
| Needs `WindState` | no — it is an empty tuple today | **yes**, one filter state per axis |
| Needs the key | no — returns it untouched | **yes**, splits it every step |
| `omega_gust` from | analytic gradient of the field | its own separate shaping filter |
| Ensemble meaning | every member meets the same field | every member is a different realisation |

Three things must change before Dryden lands, none of them large but all of them structural:

1. **`init_sim`/`batch_sim` must be parameterised to seed a filter state.** Today they
   hard-code `zero_wind_state()`, so a stateful model cannot be initialised **through them**.
   This is the actual blocker, and session 12 narrowed it: it is the *seeding* that is
   missing, not the *carrying*.

   `step` threads `wind_state` opaquely and never interprets it, so a model already brings
   its own state type — `test_integrate.py`'s `FilterState` has done so since session 2, and
   session 12's `_Clock` carries a time field the same way. **So `wind.WindState` does not
   need to grow fields, and the wind-model signature does not change.** Constructing the
   `SimState` directly is the workaround until `init_sim` takes a seed; that is a two-line
   change to one function rather than a structural one. Growing `WindState` a `t` field
   *now* would also break `FilterState`, since `step` would have to `_replace` a field that
   a bring-your-own state does not have.
2. **`omega_gust` needs its own filter.** `field_model`'s analytic-gradient trick has no
   equivalent for a stochastic field; MIL-F-8785C gives separate rate spectra, and reusing
   the translational filter would be wrong.
3. **The ensemble contract needs stating in a test.** "Deterministic components return the
   key untouched, so every member meets the same vortex" is currently true by construction
   and asserted nowhere. It stops being true for free the moment a stochastic layer is
   superposed with a deterministic one.

Adding another *deterministic* field — mountain lee wave, microburst, wake vortex from a
preceding aircraft — needs none of this. Write the field function, wrap it in
`field_model`, done. That path is genuinely extensible today.

### Extensibility: what the next encounter category will cost

Session 7 added the first encounter that is **not** a wind field, and the shape of that
change is the reusable part. There are now two ways to disturb the aircraft, and they are
siblings rather than one general mechanism:

| | `fly` | `manoeuvre` |
|---|---|---|
| Excitation | a wind field | an elevator schedule |
| Controls | fixed, by design | time-varying, necessarily |
| Wind | the field under test | zero |
| Rollout | `integrate.rollout` | its own `lax.scan` |
| Window from | north position | time |
| Analysis | **`_measure`, shared** | **`_measure`, shared** |

The last row is the load-bearing one. A category is a *mask plus a history*; everything
downstream — air-relative α, load factor, the Fig. 8 coordinate — happens in one place.
So a fourth category costs: excite the aircraft however it must be excited, build a
boolean window from the rule below, hand both to `_measure`. Nothing in `fig8_point` or
the figure needs to know which kind it is.

**The window rule, which is what makes the categories comparable at all:** the window is
the *disturbance's own extent*. The core for a vortex, the column for an updraft, the
pulse for a manoeuvre. Stated in §8 with the measurements that forced it. Without it,
categories are not commensurable and the discriminator means nothing — Δθ over a
badly-chosen window moves by a factor of three.

**Resist generalising `fly`.** A schedule that happens to be constant is a strictly larger
surface than a constant, and `fly`'s fixed controls are a *physical* statement — the
discriminator separates turbulence from manoeuvring by whether pitch correlates with
elevator, so an autopilot or a moving stick in the turbulence cases would blur exactly the
distinction being measured.

**`elevator_for_load` is more general than its name.** It is an inverse solve — "what
input produces this response?" — done as a bisection over a vmapped rollout, and it is why
the manoeuvre's deflection is *derived* rather than chosen. Any future "fly to a stated
condition" (a target roll rate, a target rate of descent) is the same three lines with a
different scalar extracted from the history. That is the difference between a model that
reaches a source's number and one that was handed it.

### Extensibility: what the next aircraft will cost

`FlightCondition` + `from_dimensional_longitudinal/lateral/controls` (session 5) are the
shared conversion path. They exist because the same algebra was hand-transcribed three
times and produced a real bug the third time — the Cessna's control derivatives recovered
at the wrong dynamic pressure, 25% high across all four.

Adding an aircraft whose source gives **dimensional** derivatives is now: state the source's
own `FlightCondition`, call the three helpers, fill in propulsion and limits, add gains.
Whose source gives **non-dimensional** derivatives (Navion, Cessna): skip the helpers
entirely.

What is still per-aircraft and unavoidable: the drag polar. `CD0` and `e` are back-solved
differently for every aircraft — from `Xw` for the 747 and Cherokee, by least squares on a
table for the Cessna — because no source states them. That is source variety, not missing
abstraction, and pushing it into a data file would hide the derivation rather than share it.

**Data-file definitions were considered and deferred.** The blocker is that the per-number
provenance comments and the back-solve logic are the most valuable part of `aircraft.py`,
and TOML expresses neither. The sequence that would work: keep factoring derivation into
tested helpers until a definition is *only* citations plus literal numbers, then the data
file is a mechanical translation. Not before.

### Extensibility: the ceiling nobody should walk into

> **The seam this section said was not taken has now been opened — session 20.**
> `aero.coefficients` interpolates a `CL(α)` table when the aircraft carries one, and the
> two 737 entries carry 737.xml's own. The three consequences below still hold **for every
> entry without a table**, which is all four light and heavy aircraft; they no longer hold
> unconditionally, and the first of them was never a property of the airframe.
>
> Nothing else about this section changed: the exit named at the bottom is the one that was
> used, and the traps it did not mention are recorded with it.

`aero.py` is `CL = CL0 + CLa·α`, linear, with no stall — and by decision it stays that way.
Three consequences, stated here so they are not rediscovered:

- **The ±g asymmetry cannot be reproduced.** It is exactly odd-symmetric in Δα, so an
  up-gust and an equal down-gust give equal and opposite load increments to machine
  precision. §5 already calls this structurally impossible; the decision to stay linear
  makes it permanent, not merely pending.
- **The Cessna's stall tables stay unused.** `CESSNA172_TABLES` runs to CLmax 1.889 at
  19.5 deg and nothing reads it.
- **Any encounter driving α past ~10-12 deg reports lift the sources say is not there.**
  Parks-scale vortices do exactly this: the source's own note estimates α excursions of
  order 20 deg. Analysis windows must therefore stay in the linear range, and a run that
  leaves it is not evidence of anything.

If that ceiling ever needs lifting, the seam is `aero.coefficients` — swap it for a
protocol with a linear and a table implementation. ~~That was the option not taken.~~
**Taken in session 20**, and it cost less than this paragraph implies: two `Aircraft` fields
defaulting to empty, one `jnp.interp` behind a shape test that resolves at trace time, and
no change to any other aircraft. What it did cost was two things this paragraph did not
mention:

- **A kink at every breakpoint.** `validation.longitudinal_matrix` takes `jacfwd` of the real
  dynamics, so a linearisation *at* a knot returns a one-sided slope and the modes become an
  artifact of knot placement. The 737's knots sit at α = 0.00 and 0.23 rad and it linearises
  at 0.0346 and 0.0631 — inside a segment both times, asserted rather than assumed in
  `test_the_lift_table_is_not_linearised_at_a_breakpoint`. The cruise margin is only 1.98°.
- **Trim stops being single-valued** above the break, since two incidences give the same
  lift. Both entries trim far below it, but `trim.trim` is an unbracketed root-find and has
  no defence if one ever does not.

Linear interpolation is deliberate and is *faithful* rather than lazy: JSBSim's own `<table>`
blocks are linearly interpolated and clamp at their endpoints, so `jnp.interp` reproduces the
source exactly. A smoother interpolant would agree with the source less.

## 8. Open questions

- ~~**New, session 30: should the Hannibal headline fly Mehta's cores on the 747's own path,
  or on the path the fit was made along?**~~ **ANSWERED, later in session 30 — on the path the
  fit was made along.** Parks et al. 1985 Fig. 6, digitised, puts the DC-10's inertially
  estimated altitude at 36,985–36,996 ft at cores 3 and 4, 17–61 ft below where the 747's run
  starts. The climb came 21.6 s later, past core 5. §4, "The DC-10's altitude through the
  vortex pair".
  - ~~**What stays open is a decision, not a measurement.** Whether §1's headline moves to the
    replayed 75.0%, or the 747 is flown with an altitude hold instead, would re-pin
    `cat_validation.py` and its dependants.~~ **DECIDED, end of session 30: the headline flies the
    replayed field** (75.3%). An altitude-hold 747 was not flown. The other CAT scripts still fly
    the field as flown and were not re-run (§4, first entry).
  - The question as it was written is kept below.

  **Should the Hannibal headline fly Mehta's cores on the 747's own path, or on the path the
  fit was made along?** §4's "The Hannibal horizontal wind" measured the
  difference at **~10 points of the record**: 64.5% as flown, and 75.0% with the fitted wind
  replayed at the nominal altitude (bare entry 68.2% and 77.5%).
  - **The mechanism.** Mehta's cores 3 and 4 sit 94 ft and 254 ft above the DC-10's path.
    The simulated 747 climbs ~500–600 ft before reaching them, so it passes above both. The
    horizontal wind flips sign across a core, so it meets +40.6 and +27.2 kt tailwind gusts
    that cut airspeed and load.
  - **What decides it.** Whether the DC-10 itself climbed that far through the pair:
    - If it did, Mehta's z already carries the climb, and the as-flown form double-counts
      nothing the DC-10 did not also do.
    - If it held altitude, the replayed form is the faithful one.
  - **What would answer it.** Parks et al. 1985 Fig. 6 plots the DC-10's altitude against
    time (§7 inventory item 6). **Nothing has been read from it.** Until then, the headline
    is quoted as flown, and this sensitivity goes with it.

- ~~**Hannibal's core radius: 500 ft or 600 ft?**~~ **CLOSED, session 26 — 600 ft, on the
  paper that identified it.** Parks et al. 1985 is now held: J. Aircraft **22**(2) 124–129,
  DOI 10.2514/3.45095, p. 127 — r₀ = 600 ft, V₀ = 85 ft/s, spacing 3500 ft, and an abstract
  giving core **diameters** of 900–1200 ft that removes the radius-versus-diameter question
  entirely.

  **Session 23 answered this the other way and the reasoning is worth keeping**, because it
  was sound on what it had. It read Mehta 1987's converged 500.5 ft and TM-102186's "1,000 ft
  diameter" as two independent confirmations of Fig. 4, and Parks as an unretrievable
  transcription. Two of those three premises survive: Mehta's refit is real and better
  converged, and TM-102186 does report it. The third does not — Parks is retrievable, says
  600 ft, and the "three sources now agree" was really **one later fit reported three times**.

  **What is left is not a resolved question but a carried disagreement**, and it now lives in
  `ASSUMPTIONS.md` **E12** with all four lineages and what the spread is worth. The project
  flies Parks' triple in `PARKS_CASES` and Mehta's five-core field in `MEHTA_HANNIBAL_*`, and
  never crosses them. The hybrid that existed from session 22 to 25 — Fig. 4's radius with
  Parks' strength and spacing — is gone.

  **The correction cost 4.26% of the headline peak-to-peak load, downward**, taking JSBSim's
  Hannibal from 70.3% to 67.3% of the recorded 2.70 g. `dw/dx = V₀/r₀` inside a solid-body
  core, so a larger core at fixed V₀ is a gentler gradient. It makes §5's shortfall worse,
  which is the direction that argues it was not chosen for convenience.


- **New, session 29: where WOULD Fig. 8's pitch axis fail, and can any aeroplane in this
  project answer?** §4 records that inside §1's envelope both axes separate perfectly — 0
  overlapping pairs in 1,024 — and that peak |α| passes the 10° linear-aero ceiling at
  σ_w = 4.0, *below* the sourced ceiling of 4.459 m/s. So the axis never fails anywhere the
  model may be believed, and `AUC pitch falls through 0.95` is not crossed in the sweep.

  **The question is whether that is a fact about Fig. 8 or a fact about this aeroplane's
  linear aero.** The envelope closes because `CL = CL0 + CLa·α` has no stall, which §5 already
  blames for the ±g asymmetry being unreachable — the same ceiling from a third side. An
  aircraft with a nonlinear lift curve would reach higher σ_w before leaving its own linear
  range, and `refs/NASA-CR-114494.pdf` holds the 747's buffet boundary but not the nonlinear
  curve. **Nothing held here settles it**, and the honest form of §4's result is therefore
  "the pitch axis does not fail inside the envelope", never "the pitch axis does not fail".

- **New, session 23d: why do the two engines choose different cores on the array?**
  On a single Parks core AtiSim and JSBSim put their load extremes within two metres of
  each other. On Mehta's five-core array they land on different cores — AtiSim's peak is
  2.2% *higher* than JSBSim's, its trough 3.5× shallower, and the whole −10.1% span error
  is the trough. The obvious cause was tested and **failed**: shrinking the window until
  the airspeed drift is −0.19% leaves the disagreement at −7.6%, and AtiSim's span does not
  move at any window width.

  What is different about the array is its **length** — 34 s and five short periods, against
  1.3 s and a fifth of one. Damping, thrust/drag bookkeeping over 45 s, and accumulated
  trajectory divergence are all candidates and none is established. The cheapest
  discriminator is probably a **still-air run of the same duration from the same state**:
  whatever the two engines do to each other over 45 s with no wind at all is the part that
  is not the encounter. `scripts/vortex_diagnose.py` did exactly this for session 22's pitch
  offset and settled it in one run.

- **New, session 23c: what are the units of Mehta's cost `J`, and what was `N`?** The
  decomposition in §4 rests on `e` being a difference of winds in **ft/s** for both
  components. That is the natural reading — `e` comes from his Eq. (4), which is
  homogeneous in `V₀`, and `V₀` is quoted in ft/s throughout — but he never labels `J`,
  and his own Fig. 5 plots the *horizontal* wind in knots while the vertical is in ft/s.

  **What survives either way, and what does not.** `mehta_residual_ceiling` is immune:
  `B` is the identity, so both terms are non-negative and either one alone is bounded by
  `J` whatever the other's scale. The 4.46 m/s ceiling therefore stands unconditionally.
  `mehta_unmodelled_wind`'s 2.11 m/s does **not** — it splits `J` evenly between the two
  components, which a mixed-unit `e` would break. Both functions say so in their
  docstrings.

  **`N` is separately unrecoverable and would sharpen nothing structural.** It is not
  needed for any number quoted — that is the whole point of the `1/N` in Eq. (A3) — but it
  would say how many degrees of freedom the fit had, which is the missing ingredient for
  turning the residual into a confidence interval on `V₀` and `r₀` rather than the
  conservative per-sample ceiling §4 uses. TM-102186 Table II gives DFDR sample rates by
  parameter and aircraft, but its text layer OCRs to unaligned columns and the wind
  reconstruction's own output rate is not stated anywhere held here.

- ~~**New, session 23: what does `PARKS_CASES["hannibal"]` being a hybrid cost?**~~ **MOOT since
  session 26**, which obtained Parks 1985, adopted 600 ft, and deleted the hybrid; §8's
  core-radius entry records it. **Verified in code, session 32:** `PARKS_CASES["hannibal"]`
  is r0 = 182.88 m (600 ft), v0 = 25.908 m/s (85 ft/s), spacing 1066.8 m (3,500 ft) — Parks'
  own triple, no longer a hybrid. Found still listed as open in session 32. *Was:* It pairs
  Fig. 4's radius (500 ft) with Parks' strength (85 ft/s); Mehta and TM-102186 both pair
  500 ft with 87. The field is exactly linear in V₀, so every wind-derived quantity moves
  by 2.1% — measured, not argued. It is left alone because §4 baselines sit downstream.
  **The question is whether a frozen baseline should be re-pinned to a coherent
  single-source pair**, which is a decision about what §4 is for and not one to take
  incidentally.
- ~~**Why does atisim accumulate more nose-down attitude than JSBSim through the run-in?**~~
  **ANSWERED, session 22 — it is the shared-start compromise, not the physics.**
  `scripts/vortex_diagnose.py` settles it two ways. Flown from JSBSim's state in
  **still air**, with the vortex switched off entirely, atisim drifts −0.340°, −1.583°
  and −1.502° over the same run-in — which is **101%, 103% and 101%** of the
  atisim-minus-JSBSim offset measured at the window edge with the vortex on. The vortex
  contributes nothing to it. And started from **atisim's own trim** instead, the from-trim
  pitch error collapses from −15.0% / −42.8% / −53.7% to **+0% / +1% / +1%**.
  So the whole of the apparent 15–54% pitch disagreement is atisim settling out of a trim
  that is not its own, at 0.95% of C_L, over eight to nine seconds. It is the price of the
  identical initial condition, it is now measured rather than assumed, and it does not
  touch the core response.

## 9. Session log

### Release 1.1 — the α̇ derivative declared, and two defects it uncovered

**The question asked was "fix the phugoid gap".** There is no open phugoid gap: the 747 reads
+1.69% in frequency and +2.83% in damping against Table IX-5, and the 737 +0.45% against
JSBSim's own linearisation. The 6.58% figure that made it look open was stale prose in
`test_jsbsim_737_layers.py`, left behind when session 24 closed that gap; it is marked
superseded in place. **The one materially open longitudinal error was short-period damping, at
−11.5%**, and that is what this work closes.

**1. `boeing747` declares Table IX-4's `Mwd` as `Cmadot` = −6.3360.** The conversion is the
`Cmq` relation with one more U0, because `Mwd` is per unit `wdot` and α̇ = `wdot`/U0; it
round-trips to the table's −0.000116 exactly. The model needed no new structure — `Aircraft`
has carried `Cmadot` since session 18 and `aero.coefficients` applies it — so the only thing
missing was the number, from the same table the entry's other seven longitudinal derivatives
come from.

| against CR-2144 Table IX-5 | before | after |
|---|---|---|
| short-period ζ | −11.48% | **+0.57%** |
| phugoid ζ | +2.83% | **+1.38%** |
| phugoid ω_n | +1.69% | +1.68% |
| short-period ω_n | −1.32% | −1.32% |

**`Zwd` = +0.00556 is tabulated and is deliberately NOT taken.** Converted the same way it
gives a negative `CL_α̇`, and downwash lag makes that derivative positive. Nothing the project
holds settles whether the sign is the table's convention or a misprint, so rule 2 applies:
flag, never invent.

**Corroborated from two directions that did not know about each other.** Yoshimura's Table A2,
at a different altitude and from a different author, gives `C_m_α̇` = −5.40 against this
−6.336, and supplying the declared value closes the FL200 damping shortfall from 11.7% to under
3%. §4's own digitised reading of the same CR-2144 panels gives −6.44, 1.6% away.

**2. What it costs, stated plainly.** The headline falls from **75.3% to 70.2%** of the
recorded peak-to-peak: the α̇ moment damps exactly the response the encounter excites. This is
the second time a SOURCED derivative has improved agreement with the model's own source and
moved the headline away from the record — the speed derivatives were the first — and it is
recorded rather than resolved. The gust strength that reaches the recorded peak moves with it,
from ×1.2 to ×1.25 replayed, and the saturation bracket returns to ×3.25–×3.5, where it sat
before session 30. TM-102186 Fig. 8's mechanism stays monotone six-for-six; the paper's own
asserted pair (Cherokee against 747) holds. **The 747 is no longer the fleet's lowest `n_z`
minimum** — −0.306 against the 737's −0.364 — a margin §4 recorded at 0.008 before this and
which has now changed sign. The paper does not assert that pair; the record does, and it says
so.

**3. Two defects surfaced, both live before this work, both invisible because a zero multiplied
them.** §6(i), the autopilot's engage seed, found by reading the parked WGS-84 branch and
confirmed on `main`; and §6(j), the gust α̇'s missing transport term, found by this declaration
and by nothing else. The second is a §2-class error — a change of inertial frame moving the
attitude — and it had survived every test in the project. Its fix is in
`dynamics.derivatives` rather than in `wind.gust_alphadot`, because the term varies within an
RK4 step and holding it costs three orders of accuracy (3.99 → 1.03, measured); putting it
where the equations are evaluated also makes it impossible for a caller to omit, which is how
it hid.

**What was NOT done.** The WGS-84 branch stays banked: at this flight condition Coriolis is
3.5 milli-g and the centrifugal term 2.7 milli-g against a headline shortfall of 24.7%, so it
cannot move the claim, and merging it means rebasing across 347 changed files. Its `earth.py`
was read and checked — the WGS-84 constants, Bowring's method, the non-singular altitude
formula and the J2 acceleration are all correct — with one documentation fault worth fixing if
it is ever merged: the constants are cited to the JSBSim binary they were recovered from rather
than to the standard they match.

### Session 32 — the final month is planned, and the repository is found publishing seven copyrighted papers

**This session measures nothing about the aeroplane.** It is the first of the run to the
30 September deadline, and its subject is the repository rather than the model. The plan is
`docs/design/specs/2026-09-17-final-release-cleanup-design.md`; what follows is phase 0
of it, plus three findings that were not expected and that change what the remaining phases
have to do.

**1. The repository is public, and has been redistributing seven copyrighted papers.**
`private: false`, `license: null`. Fourteen third-party PDFs were tracked. All 24 documents on
disk were adjudicated **against their own printed copyright pages**, not against the
publisher's name, and the split is not what the name predicts: `parks-1985` prints "U.S.
Government and therefore is in the public domain"; `mehta-2012` and `bach-parks-1987` both
print the Title 17 "no copyright is asserted in the United States" statement **despite being
AIAA *J. Aircraft* papers**; `schultz-2012` says it is "not subject to copyright protection in
the United States". Seven are kept on that basis and seven removed — Yoshimura 2023 (CC BY-**NC**,
and the copy carries an institutional-access watermark), an AIAA 2007 paper printing "All
rights reserved", two AMS papers **one of which is a byte-for-byte 4.1 MB duplicate of the
other**, `loving-2012`, a 1989 *Mon. Wea. Rev.* paper, and `dynamics_of_rigid_aircraft`, which
is excluded **because its licence could not be established** — rule 2 applies to a licence
exactly as it applies to a coefficient. `Reference_papers/SOURCES.md` is the new index, with
md5 and a permanent locator for every document. `.gitignore` now denies by default there, as
it always has for `refs/`. **The files remain in history** until the rewrite, which cannot run
until the outstanding branches are merged without orphaning them.

**2. Rule 1b failed twice more, and one of them is large.** `project-md-restructure-95b7b8`
held **49 uncommitted files** — the α̇ work, §0 has the row — and the main checkout held
`Reference_papers/CR-2144/`, **an automated digitisation of CR-2144 printed pp. 218–228**: 25
panels, 77 curves emitted, with overlays and a verification sheet. §7 records session 30
digitising **pp. 220–222 by hand** and §4 calls that the highest-value item on the plan. Both
are now tracked, neither is reviewed. **Two independent digitisations of pp. 220–222 now
exist**, which prices the reading error on both for the cost of one script; §7 carries it.

**3. A branch that closes a §7 item was not mentioned in this document at all.**
`claude/work-summary-derivatives-sensitivity-9a2c03`, four commits ahead of `origin/main`,
written the same day, **zero occurrences anywhere in this file**. It reports identifying the
Hannibal flight as **United 12** and the aircraft as **N1809U, a DC-10-10**, and narrowing §5's
wing-loading ratio from 0.58–1.32× to 0.584–1.046× by removing the −30 corner. §7 calls that
item "the only thing that would let the aircraft-type explanation be tested rather than
argued". **It is a POINTER until verified** — §0 has the row and the caveats, including that the
weight on the day is still not found and that the variant is a secondary source. **It also took
the number "session 31", which is why this session is 32.**

**4. §0's own claim about `main` was stale, and it inflated every count in the section.**
`main` is **0 ahead and 19 behind `origin/main`**, a pure fast-forward — not "7 ahead and 1
behind, diverged". Measured against `origin/main` the branch list shrinks sharply: two branches
go to zero and the unrecorded one above is 4, not 15. Also measured: **`origin` carries three
branches against 23 local**, so most of this section exists on one machine, which is the other
half of why rule 1b keeps failing.

**5. §0's open question about the two WGS-84 branches is answered, and the answer is
"neither".** They diverge at `01d7502` after 61 shared commits and each carries unique work —
one retires `ASSUMPTIONS.md` A1 and A2, the other closes three red tests and re-measures the
§4 ledger. The merge is the **union**, with 8 files changed in both, and `atisim/validation.py`
is the one that decides the cost: both linearise it, one about the transport rate and one about
the equilibrium. §0 has the table.

**6. Phase 1, the branch triage.** Sixteen branches carried unique commits at the start of it.

**Merged, two:**

- **Session 31's**, above — five conflicts in this file, every one resolved by keeping *both*
  sides, because that branch predates the session-30 merge and taking "theirs" anywhere would
  have deleted session 30. §4 went **55 entries → 56**: nothing superseded, one gained.
- **`claude/zen-maxwell-1ad0a4`**, the tail-arm gate, rescued unreviewed at the session-28 audit
  and left for three weeks. **Reviewed and merged, and it closes a silent bug now recorded as
  §6(g)**: `airframe.stations` did not gate on the tail arm though it is the only place a
  `Stations` is built, so for an entry with no `CLq` the arm was `−Cmq/0 = inf`, the `linspace`
  ran to `−inf`, and **every fitted gradient returned NaN with nothing raised** — a rollout
  would finish and report a NaN rms. Its docstring's claim of "five of the seven aircraft" was
  **measured rather than trusted**, and holds: 5 refused of 7. 68 tests pass across the three
  files it touches.

**Abandoned, eight — and two of them were not empty.** §0 has the table of what was taken from
each before it went, which is the only order that is safe:

- `claude/weekly-summary-analysis-7520db`'s scripts were all already on `main`, but **its
  `runs/cat/` outputs existed nowhere else** — 55 files, 11 MB, including the ten LES ensemble
  arrays §4's POD row reads. Copied to the main checkout and **verified by md5 file-by-file**
  before the branch was touched. §10 now says where they live.
- `claude/cat-flight-model-dossier-04adb9`'s *plan* is superseded — it argues the 500 ft radius
  session 26 reversed and says Parks 1985 "has never been obtained", which session 26 disproved
  — but **its retrieval ledger had five rows recorded nowhere in this document**, now §7. The
  best of them is **14 CFR 25.341**: free from eCFR, and the only identified source that would
  give σ against altitude *independently* of the U-2 lineage §7 warns MIL-F-8785C Fig. 7 may
  share with HICAT. **Cheapest open acquisition the project has.**

**Deferred, two, for a stated reason rather than for lack of time.**
`claude/linearisation-verification-bounds-b73868` (953 lines of `verification.py` against
`main`'s 359) and `claude/flight-dynamics-solver-oscillation-17139b` both modify
`atisim/validation.py` — **the same file the two WGS-84 branches fight over.** Merging them
first would make that fight harder, not easier, so all four are handled together.

**7. The suite, on the merged triage tree: 918 passed, 1 skipped, 0 failed, 1,092 s.** That is
session 30's 906 plus the 12 `test_figures.py` tests the panel-chrome fix brings, and **zero
failures** — the two platform bit-pins that failed through session 29 are green. §10's count row
is re-measured rather than incremented, per its own instruction; it has been wrong twice before.

**8. Phase 2, the WGS-84 merge — measured first, then descoped by the project owner, then
banked.** The plan gave it four days and a go/no-go gate on Tue 22. The first day measured what
it would cost before spending the rest, and the plan's figures were wrong by enough to change the
decision: **78 behind, not 27; 20 files and 50 hunks against `main`, not 8 files; and 30 silent
conflicts the plan had not considered**, because `quat`, `omega` and `vel_body` keep their names
across the frame change and alter their meaning. Those merge clean, run without error, and return
wrong numbers. §0 has the table and the file-by-file map.

**The gate itself was judged inadequate.** "Suite green" catches the 15 loud `pos_ned` breaks and
gross frame errors, but the suite asserts bands and orderings, so a small frame error in one of the
30 can land inside a band. A green suite could have certified a broken merge.

**Decision: build the union, push it, and stop short of `main`.** `wgs84-earth` resolves the two
branches' 13 hunks one at a time, and three of those resolutions are findings in their own right:

- **The two linearisations are one fix done twice, and one copy is incomplete.** `32fbdd`'s
  linearises off-equilibrium by `|f(x0)| = 1.3e-05`; `5dbdc3`'s also corrects the kinematic row and
  reaches `2.064e-13`.
- **A tolerance had been loosened 25× to absorb an error.** `32fbdd` held a bare-`M_u` check at
  `rel = 0.05` because its helper took the transport rate at `θ = 0`. `5dbdc3` took θ₀ from the trim
  and holds it at `2e-3`. Rule 3's case exactly, and the tight version passes.
- **An architecture inventory misdescribed its own code.** `5dbdc3`'s R6 row omits the Earth-rotation
  term that `dynamics.py:107–109` computes.

**Two `ASSUMPTIONS.md` entries contradicted each other**, and the superseded side is kept struck
through beside its correction rather than dropped: A3's "sound, 0.17%/0.31%" was an altitude
difference standing in for a density error 2–3× larger, and F4's "the ceiling did NOT drop" was
overtaken by a measurement at 1.2e-11 m.

**Union suite: 750 passed, 1 skipped, 0 failed.** The two source branches are retired with their
SHAs in §0, after confirming the union holds every commit of both.

~~**What it costs the release, stated plainly: `ASSUMPTIONS.md` A1 and A2 stay open on `main`.**~~
**CORRECTED, later in session 32: only A1 stays open.** A2 and A3 were already retired on `main` by
the compressibility merge in session 28 — `dynamics.gravity(z) = g₀(R/(R+z))²` and the geopotential
conversion. What the union would add beyond `main` is gravity's latitude variation (0.53%) and the
centrifugal term, both of which `main`'s A2 lists as still assumed. The error was made writing this
entry, repeated in PR #9 and in the plan document, and found while inventorying §5 for phase 3. It
did not reach code. The
release ships a flat, non-rotating Earth with that assumption declared rather than retired. The
rotating Earth is complete, reconciled, pushed, and one rebase from landing — and §0 says how big
that rebase is, and where in it to look first: **15 of the 30 silent references are in session 29's
sensitivity study**, which is the path that produced §1's banded headline.

**9. Phase 3 — every open item given one status.** The plan is
`docs/design/plans/2026-09-18-phase-3-close-every-open-item.md`, reviewed and approved with
five decisions before it ran. **The head of §5 now carries a status table**: every open item in §5,
§7 and §8, and the one sealed prediction, each with exactly one of four statuses — CLOSED /
MEASURED / BOUNDED, IMPOSSIBLE WITH SOURCES HELD, FUTURE WORK, OUT OF SCOPE — and the evidence or
route beside it. FUTURE WORK is new and was the owner's decision: it is the list a successor
starts from, not a list of defects.

**The inventory was taken against the code and against §4, not against each item's own wording,
and that changed it.** Five rows were already closed — two of them the ones §7 called "the top of
the list" and "CHEAP", both done in session 27 — and one claim of this session's own, that A2 stays
open on `main`, was wrong and is struck through in point 8.

**Four measurements, and the most consequential was not planned:**

- **The strip load path counts the gust's rolling moment twice — §6(h), OPEN.** Found in a branch
  about to be deleted, which had seen it before session 24 could make it show. `strip=True` adds
  the strip rolling moment to the point path's equivalent roll rate instead of replacing it.
  Measured on `main` with a harness that reproduces `lateral.py` to three decimals: the published
  strip effect of **+17.4%** on peak bank is **−2.5%** with one path owning roll, and that residual
  is F5's quadrature deficit. **§4's "+22.9%" and §7's phase-1 gate were both met by the double
  count** and are superseded in place. Nothing longitudinal moves, and nothing in §1.
- **CR-2144 read twice.** Sealed first, while the comparing function did not exist; **WRONG on
  `Cm_M`** at 2.40% of full scale. Scored against Table IX-4, the automated trace is the biased one:
  **on all three speed derivatives the shipped 747 declares, the session-30 hand reading is the
  better-anchored** — the first check of those values from outside their own reading, and it
  supports them. The automated `CL_M` and `Cm_α̇` are wrong curves, which corrects §7's route for
  a `Cmα̇` Mach schedule.
- **14 CFR 25.341 is not the independent check §7 hoped for**: its continuous-turbulence `Uσref` is
  a design limit, constant above 24,000 ft, with no exceedance probability. The high-altitude
  discrete-gust "discrepancy" the ledger warned of is one linear law, extended in 2015.
- **The last three DC-10/MD-11 papers hold no derivative set**, so §7's DC-10 row is final and the
  sealed DC-10 prediction stays unsettleable.

**The α̇ branch was reviewed**: its change to a rule-3 file is a re-capture, not a loosening, and
the burden was discharged at the change — §0's claim that it was not is corrected. **Parked as
FUTURE WORK** by decision, because merging moves the four CR-2144 modes and §1's headline.

**Also harvested before two branches were deleted:** a third, sign-flipped `Ixz` in Caughey's
worked example (§5.16c), and the finding that the phugoid the simulation flies is not the 4-state
one it reports (§5.21).

**What phase 3 did NOT do:** fix §6(h) — the repair is a modelling decision, a week before release;
merge the α̇ branch or `wgs84-earth`; run the engines' still-air discriminator for §8 (owner's
decision; FUTURE WORK, with the code already written into the plan); re-measure §5.21 on the
shipped 747; or fetch the NTSB pre-1982 database. **Re-measured no §4 row except incidentally** —
the lateral rows, which are recorded stale beside §6(h).

**Suite at the end of phase 3: 926 passed, 1 skipped, 1 xfailed, 0 failed, 1,561 s** — 918 plus the
crosscheck's eight, and the one xfail is `Cm_M`'s band, marked strict where the sealed prediction
failed. §10's row is re-measured, not incremented.

**10. Phase 4 — the documentation.** A Sphinx site in `docs/`, themed with Furo: **74 pages, zero
warnings, built green by CI on a clean Linux runner** (`.github/workflows/docs.yml`, run
35364436940, with warnings as errors). **Its narrative pages include sections of this file
verbatim** through MyST's `{include}` — the *Running it* page is §10, the *Validation* page is §1's
claim plus the §5 status table — so the site cannot carry a number the record does not. The API
reference is generated from the docstrings, and a hook in `docs/conf.py` renders their plain-prose
maths and aligned tables as written: the first build showed 26 of them as red error text.

**What it changed besides adding pages.** `docs/superpowers/` became `docs/design/`, and the
agent-tooling header was stripped from every plan in it. `SESSION_27_REPORT.md`'s verdict was
harvested into the *Validation* page, brought up to date, and the file deleted. `CHANGELOG.md`
groups the release by capability; checking it against `main`'s code corrected three of its own
statements before they shipped. The README was rewritten — its quoted results predated session 30
— and `CONTRIBUTING.md` added. **Checking the changelog's "44 scripts, each documented in §10" found
nine that were not**, now all listed; two of them, `les_flight.py` and `les_compare.py`, hard-coded
the maintainer's path to the LES dataset and could not run for anyone else. Both now take
`--dataset` or `ATISIM_LES_ROOT`, with the old path as the default.

**What phase 4 did NOT do:** enable GitHub Pages — it cannot publish from a private repository on a
free plan, and the workflow's deploy job switches on by itself when the repository goes public; or
add a test-suite workflow, which is phase 5's, because session 29 recorded two platform bit-pins
failing on Linux and a red badge for a known platform difference would mislead. *(Point 11: at
the `.venv`'s library versions they pass on Linux.)*

**11. Phase 5 — the validation notebook, and the suite in CI.** `notebooks/validation-ladder.ipynb`
walks §1's claim in four rungs, computing every number as it runs and asserting bands and
orderings only:
- `scripts/sanity.py`;
- CR-2144's modes (Tables IX-5, IX-9 and IX-10) and drag polar (Fig. IX-6), and the JSBSim
  737's trim and modes;
- the Hannibal headline, with its envelope checked on the run (`checks.recovery_band`,
  `checks.alpha_band`);
- TM-102186 Fig. 8's fleet ordering and mechanism, and Wingrove & Bach Fig. 8's
  vortex < updraft < manoeuvre ordering.

Published values and tolerances are imported from the tests that assert them, not retyped. **Its
first cell asserts that `atisim` is imported from the checkout the notebook sits in**, and was
shown to refuse a run from this worktree with no `PYTHONPATH` — rule 4 and §10's launch table,
enforced rather than described. `.github/workflows/tests.yml` runs the suite, then
`pytest --nbval-lax notebooks/`, on every pull request and on `main`. Its first run, 35370419455:
**926 passed, 1 skipped, 1 xfailed in 1,566 s**, and the solver notebook's 13 cells green. On
the final commit, run 35390817171: **926 passed, 1 skipped, 1 xfailed in 1,672 s**, then **24
passed in 113 s** — both notebooks, every cell.

**Four things it found.**

1. **`scripts/sanity.py` had been reporting 8/11, and nothing ran it.** All three failures were
   the model changing under a hand-derived expectation, not a regression. Each expectation is
   derived by hand again, with no tolerance touched:
   - **[2]**, free fall, expected `G0`. Gravity has varied with height since session 23; at
     cruise it is 9.76922. The script now derives `g(h)` from R = 6,371 km.
   - **[5]**, "Cm = 0 gives no pitch acceleration", read 0.0073 rad/s². 99% of that is the
     thrust line the 747 has declared since session 30, 5.70 ft below the CG. The other 1% is
     `Cm_M`, because the check's 12 m/s sideslip moves the Mach off 0.80. Zeroing both, and
     `Cmadot`, gives exactly 0.
   - **[10]**, trimmed `n_z`, expected cos α. The lift holds up the *local* g, but `n_z` is
     counted in standard g, so the answer is cos α · g(h)/G0. `load_factor`'s docstring made
     the same claim and is corrected.

   The script now reads 11/11. The README and the site called it "twelve cases"; the script has
   eleven, and both now say so. The notebook runs it in CI, so it cannot go stale unseen again.

   **Checked later in phase 5 against the suite, check by check. The notebook is the only gate
   on four of the eleven.**
   - **[2], [3], [4], [5]:** nothing in `atisim/tests/` asserts them. `test_dynamics.py`'s
     free-fall test runs at 1,000 m against `G0` with atol 5e-3, and g(1,000 m) is only 3.1e-3
     below `G0`, so it cannot separate the two. No test zeroes the rolling or pitching terms on
     a registry aircraft, which is why the thrust line moved [5] in session 30 without failing
     anything.
   - **[11]:** asserted for longitudinal wind fields (`test_lateral.py`, `test_checks.py`),
     never for an elevator input.
   - **[10]:** its test takes g(h)/G0 from `dynamics.gravity` itself. Its only independent check
     is a pin of 0.9930 ± 1e-4, where the script asks for 1e-6.
   - **[1], [6], [7], [8], [9]:** equivalents exist, at looser tolerances than the script's.

   ~~**Not closed here.**~~ **Closed afterwards, in the same phase.** Closing it means asserting [2]–[5] and an elevator-input [11] at the
   script's own tolerances, against hand-derived expressions rather than pinned numbers.
   `ASSUMPTIONS.md`'s protocol section records it as the second exception, and as closed.

   `atisim/tests/test_sanity.py` does that: four tests on the trimmed 747, at the script's
   tolerances. Free fall is checked against g(h) worked out from R = 6,371 km, and [4] also
   asserts that the real, coupled inertia *does* give a roll acceleration, so it cannot pass by
   the inputs doing nothing. **Each test was shown to fail when its check is broken:**
   expecting `G0` instead of g(h); leaving the aileron term in [4]; leaving the thrust arm, or
   `Cm_M`, in [5]; adding an aileron input to [11]. One detail, noticed and left: `sanity.py`'s
   [4] swaps in a hard-coded diagonal inertia that is not the 747's current one. Any diagonal
   gives the same zero, so the check is unaffected; the test uses the aircraft's own diagonal. **The Wingrove & Bach
   cell had the same shape and is not a gap:** `test_vortex_viz.py` asserts the full
   vortex < updraft < manoeuvre ordering. Its only difference from `scripts/vortex.py` is a
   pinned 8.926° elevator where the script bisects to 8.9227°, and that pinned step is held
   within 0.01 g of the band.
2. **The platform bit-pins pass on Linux.** Session 29 measured its two failures at JAX 0.10.2 and
   NumPy 2.4.6. CI resolves Python 3.10 to the `.venv`'s own JAX 0.6.2 and NumPy 2.2.6, and every
   pin passes, ~~so the dependence is on library versions rather than the OS~~. §4's "What does
   not reproduce on another platform" is corrected. **Incomplete, found in phase 6:** those runs happened to land on hosts
   without AVX-512. On a host with it, OpenBLAS's kernels move the rollout pin at the same
   library versions; CI now holds them to Haswell. §4 has the measurement.
3. **TM-102186 Fig. 8's fleet ordering and its six-for-six mechanism, which §1 cites, were
   asserted by no test.** They lived in `scripts/cat_validation.py`'s printout and §4's table.
   The notebook's rung 4 now asserts them and CI runs it, so they have a gate. But the notebook
   is that gate on its own, which departs from `ASSUMPTIONS.md`'s notebook protocol: the protocol
   puts the computation in the package and the assertion in the suite. That means moving
   `fly_mehta`, `excursion` and `traverse_ratio` out of a script. ~~**Not done here**, and the
   protocol's section says so.~~ §4 now carries the current tree's numbers beside the table.

   **Done later in phase 5.** The three functions and `MECHANISM_FLEET` moved into
   `atisim/vortex_viz.py`, and `scripts/cat_validation.py` imports them. Its printout at
   `--dt 0.01` is identical before and after the move, and so are its four figures, byte for
   byte. `test_cat_validation.py` flies the fleet at dt 0.01, the step §4's table and the
   notebook use, and asserts rung 4's four things as orderings (rule 6):
   - the Cherokee's pitch peak-to-peak exceeds the 747's;
   - its `n_z` minimum is above the 747's;
   - incidence gain falls strictly when the fleet is sorted by traverse ratio;
   - every |α| peak is under `panel.ALPHA_LINEAR_DEG`.

   Each assertion was shown to fail when its ordering is broken, including a tie in the gains.
   Rungs 3 and 4 now call `vortex_viz`, and the notebook no longer imports the script.
   `ASSUMPTIONS.md`'s protocol section records the exception as closed. The suite reads **931
   passed, 1 skipped, 1 xfailed, 0 failed** on Windows, with pull request #14 below (§10), and
   the notebooks **24 passed**.
4. **`test_jsbsim_737_layers.py`'s layer-3 docstring still describes the 6.58% phugoid gap**
   that session 24 closed to +0.45%. The notebook reads 0.0526 against JSBSim's 0.0524 rad/s. The
   test passes. Its docstring is noticed and left alone.

**Two changes landed on `phase-5` after pull request #13 had merged it.**
- **Pull request #14, `vortex-figure-layout`, merged into `phase-5` at 14:52 UTC on 19
  September, 40 minutes after #13 took `phase-5` into `main`.** So it reached neither `main` nor
  the `phase-6` branch, and this file did not mention it. It changes no number:
  - `scripts/vortex.py`'s figure was titled "vortex analysis progress". It is the finished
    analysis, so the title and the script's docstring now read "Vortex encounter analysis".
  - The seven-line provenance footer is drawn in figure coordinates, and nothing reserved room
    for it: its first line ran through the Wingrove & Bach Fig. 8 panel's x-axis label.
    `vortex_viz.figure` now ends the panels above the footer, sized by the footer's own line
    count, and `test_vortex_viz.py::test_the_provenance_footer_overprints_no_panel` asserts that
    it overlaps no axes.
  - In `validation-ladder.ipynb` only the Wingrove & Bach cell's figure changed. Every other
    output is identical, checked cell by cell.
- **Item 3's follow-up**, rebased onto it.

Both reach `main` through pull request #17, a second one from `phase-5`.

**12. Phase 6 — the history rewrite, prepared and dry-run.** `git filter-repo` runs in a fresh
bare clone, never in a working checkout. It does two things:
- it strips from every commit the seven publisher-held PDFs that `SOURCES.md` lists as removed;
- it rewords eighteen commit subjects that named a working session, a WIP state or the tooling
  ("prompts", "agent-tooling", `docs/superpowers`). Commit bodies are left as written.

**The dry run, verified:**
- none of the seven PDFs' blobs survives under any path, and no commit touches their paths;
- the tip trees of `main` and `wgs84-earth` are byte-identical to before, so the suite's inputs
  have not changed;
- no commit was emptied or lost.

**Three things the dry run found, and what was decided.**

1. **A force-push cannot remove the PDFs from GitHub.** Every pull request keeps a read-only
   `refs/pull/N/head`, and all of them reach the initial commit, which carries the PDFs. GitHub's
   documentation says only its Support can dereference them, and that Support "won't remove
   non-sensitive data". The spec planned the rewrite in place without knowing this. **Decided:
   the rewritten history goes to a new `MatusGib/AtiSim`.** The current repository is renamed
   `AtiSim-archive` and kept private. Every commit survives; the pull-request pages stay with the
   archive.
2. **Ten commits on `main` carried a tool's author and committer identity rather than the
   maintainer's**, with co-author trailers to match. They are the sensitivity study, 10–16
   September, made in a remote container. The rewrite corrects the identity with a mailmap and
   drops the trailers.
3. **The size gate fails as the spec set it: 20.8 MiB against 15 MiB.** Most of what is left is
   PDFs ruled redistributable. **Decided: the two largest, which nothing reads, leave the
   repository too.** They are HICAT (DTIC AD878415) and NTRS 19910009769, 9.1 MB between them.
   Both move to `SOURCES.md`'s fetch-it-yourself table with their locators.

**The order is decided too: the rewrite runs after the two open branches merge** — the
vortex-figure fix and the Fig. 8 suite test. A branch made on the old history and merged after
the rewrite would bring every stripped PDF back.

**Commit IDs change throughout**, including every `sealed_at` in `atisim/predictions.py`. The
rewrite's old-to-new map is committed beside the result, so every ID quoted in this file stays
resolvable, and no sealed entry is edited.

**One more change before the rewrite: `main`'s CI went red, and the cause is the runner's
CPU.** `test_extracting_rk4_step_did_not_move_a_single_bit` failed on two runs at the same
library versions that had passed before. OpenBLAS picks its kernels by CPU, some of GitHub's
runners expose AVX-512, and on those the rollout's last bits move. CI now sets
`OPENBLAS_CORETYPE=Haswell`, measured to reproduce the pin on every runner type seen. No
tolerance and no pinned value changed. §4, "What does not reproduce on another platform", has
the measurement, and point 11, item 2 is corrected.

**The rewrite ran, and this repository is its result.** `main` was `ff6562f` before it and is
`39ca6f6` after. Verified on the pushed result, re-cloned:
- **the nine PDFs are gone from every commit**, checked by object id rather than by path, so a
  copy under another name could not survive;
- **317 commits**, none emptied or lost; 244 of them on `main`;
- **the tip trees of `main` and `wgs84-earth` are byte-identical to the pre-rewrite ones**, so
  the suite's inputs did not move;
- **every author and committer is the maintainer's**, or GitHub's own for merges made on the
  website. The ten commits made in a remote container carried a tool's identity and two
  attribution trailers; a mailmap corrects the identity and the trailers are gone;
- **18 subjects reworded**, none naming a session, a WIP state or the tooling. Bodies are
  untouched, so the `claude/*` branch names they record as history remain, as §0 does;
- **a clean clone is 13.16 MiB**, against the spec's 15 MiB gate and 49.3 MiB before.

**Where the old repository went.** `MatusGib/AtiSim` is this one, new. The pre-rewrite
repository is `MatusGib/AtiSim-archive`, private, and keeps the nineteen pull-request pages —
including the read-only `refs/pull/*` that made an in-place force-push insufficient (point 12's
first finding).

**Commit IDs before the rewrite** — every one quoted in this file, and every `sealed_at` in
`atisim/predictions.py` — address the pre-rewrite history. `docs/design/commit-map.txt` is
`filter-repo`'s own old-to-new map, committed here so each one stays resolvable. Two limits
worth stating: commits on branches deleted before the rewrite are not in it, because they were
not in the repository it read, and they live only in the archive; and no sealed entry was
edited to follow the rewrite, which is rule 1 of `predictions.py` and the reason the map exists.

**13. Phase 7 — the release.** The plan's last phase, and its gate is the only one written for
a reader rather than for the record: **a stranger clones the repository, follows the README, and
gets a green suite and a rendered documentation site.** Measured that way — a fresh clone of
`MatusGib/AtiSim`, a new virtual environment, `pip install -e .[dev,docs]`, then the two
commands the README gives:
- the suite: **885 passed, 3 skipped, 1 xfailed in 1,302 s, from 887 collected**;
- the documentation site: **74 pages, zero warnings, with `-W`**.

**The gate found one thing, and it is a documentation defect rather than a code one.** A reader
following the README installed `.[dev,docs]` and got a green suite of **887 collected**, where
this record says 937. `test_artifact.py` and `test_figures.py` call `importorskip` at module
level, so without the `ui` extra's `pyarrow` and `plotly` they do not collect **at all** — 50
tests absent from a run that says nothing about them. The README and the *Getting started* page
now install `.[dev,ui]` for the suite and say what `.[dev]` alone leaves out; §10's row carries
both counts. Nothing about the tests changed.

**Tagged `v1.0.0`.** What the tag contains is `CHANGELOG.md`, by capability; what it does not is
§5's status table, unchanged by the release: the WGS-84 union banked on its branch, the α̇
derivatives parked, the strip path's double count open, and no absolute-load claim.

**What the release does not change.** Every number in §4, every band in §5 and every sealed
entry in `atisim/predictions.py` are as they were before phase 6; the rewrite moved commit ids
and nothing else. §0 above says which two rows remain live and where they now live.

*The paragraph below was written at the end of phase 2 and is kept as written; point 9 above
supersedes it where they differ — the α̇ work is now reviewed.*

**What this session did NOT do**, so the next one does not go looking: it made
**no re-measurement** of anything in §4 — every number there is still session 30's or earlier,
and the merges this session took changed no model code that §4 reads. It did not review
`claude/engine-validity-presentation-1408e8`, the α̇ work, which still modifies a rule-3
validated-baseline file with the burden unmet. It did not resolve `old-origin`, and the
`AGENTS.md` question is now moot: `CLAUDE.md` became `docs/DEVELOPMENT.md` instead. **It did not merge the WGS-84 union into
`main`** — by decision, on the measurements in point 8 — so `ASSUMPTIONS.md` A1 remains open there (A2 was already retired on `main` — see point 8's
correction). It did not rebase the union, review the 30 silent references, or re-run anything in
§4 on a rotating Earth.


### Session 31 — a summary is asked for, and the `mass` row turns out to be one lever read three ways

**Asked for a summary of the past fortnight** (speed derivatives, the sensitivity analysis, the
thrust line) **and an explanation of a `mass` result that "does not look correct".** An
investigation session: no model code, no test, and no tolerance changed. New:
`scripts/sensitivity_mass_diagnosis.py`. §4's first entry carries every number.

**1. Where the fortnight's work actually is.**
- Session 29 is merged: on `main` and `origin/main` at `105f689`.
- Session 30 is **not**. It is on the local branch `session-30-speed-derivatives-hannibal`, not
  the name either §0 gives, and it is not pushed.
- A merge of `origin/main` into it is half-done in its worktree. §0 now carries the address.
- The two-line check in `docs/DEVELOPMENT.md` would print that branch. It would not print the half-done
  merge, which is a worktree state, not a ref.

**2. The `mass` row, diagnosed rather than defended.** Four hypotheses, one controlled variant
each, by central difference on the shipped paths.

Two survive:
- **`mass` and `CLa` are one lever.** They mirror to 0.04.
- **Inertia held fixed matters**: −0.649 → −0.800 as flown and −0.624 → −0.445 replayed on the
  bare 747. The same sign change between flight forms appears on session 30's.

Two are refuted:
- **Path.** The fixed-control climb does not depend on mass.
- **Korn, on the load.**

**And one that was not asked.** An error in CR-2144's *printed* weight cancels (+0.008),
because `_boeing_747` derives `CLa` from it. **Separately, the cruise phugoid-damping
"`mass` +1.728" is mostly the DECLARED Korn drag rise** (+0.447 without it). On session 30's
entry it is worse, because `CD_M` is netted against the Korn slope at the construction C_L
(+2.230, and +0.535 re-netted).

**3. Session 29's claims were annotated in place, not rewritten**: the header, §0's "price",
S2's rank-2 row, S5's pair, S1's phugoid finding, the §7 S2 row, and session 29's §9 bullet. Its
tables are measurements and stand.

**4. A method hazard, found by accident.** On session 30's replayed headline, ±1% central
differences land on a one-sample trough jump. `CLa` reads +0.497 at 1% against +0.740 at 0.25%
and +0.695 at 5%. The diagnosis script prints two step sizes and flags disagreement. **Any
re-run of S2 there must do the same.**

**5. Then, asked to "find the numbers", a search for the DC-10 and its data.**
- **The Hannibal aircraft is United 12, N1809U, a DC-10-10** (NTSB CHI81DA042; variant secondary). §5's wing-loading ratio narrows to 0.584–1.046×.
- **The weight on the day was not found.** NASA used it (TM-102186 §2) and never printed it. The routes left are the NTSB docket, United's archives, and the Sept 1982 SFTE symposium paper (Parks, Bach & Wingrove), which is not online.
- **Fetched into `Reference_papers/`:**
  - NASA CR-3748, DC-10-10 winglet flight evaluation (md5 `f8fb0080…`): weights, buffet boundary (Fig. 32), and column-force static stability (Figs. 33–34). **No lift slope or derivatives.**
  - NASA TM-4745 and TM-1998-206552, the MD-11 thrust-only control reports: **no aerodynamic derivatives or lift data.**
  - Taylor 1978, DC-10 derivatives winglet and nacelle development: wind tunnel, not reviewed in detail.
- **The MD-11 is not a pitch proxy for the DC-10.** Secondary sources report a horizontal tail about 30% smaller, relaxed static stability and LSAS. At most its wing data might transfer, and none was found.
- **The best find is local: CR-114494 §3.0 holds 747 drag polars against Mach.**
  - PDF p. 94, printed 3.0-8, covers M ≤ 0.7 (0.3, 0.5, 0.7), in a clean scan.
  - PDF p. 95, printed 3.0-9, is a fold-out, "Drag coefficient, effect of Mach number, M > .7", with about nine polars. Its halftone scan is dark, and **the per-curve Mach labels were not readable** at the resolutions tried.
  - It is the sourced alternative to the Korn/Lock drag rise, if the labels can be recovered.

**What this session did NOT do.**
- It did not touch the in-progress merge in `engine-validity-error-check-d8ccdf`, which belongs
  to whoever started it.
- It did not push anything.
- It did not re-run S2/S5/S6 on session 30's model. That tree lacks `sensitivity.py` until the
  merge lands; only central differences were run there, from a `git archive` of `ea74849` in a
  scratch directory.
- It did not reparameterise the screen's `mass` factor (`W/S`, or mass with a radius of
  gyration). Which question the screen asks is a decision.
- It did not re-run the Dryden rms with inertia scaled, or measure the approach condition.
- It did not refit Korn/Lock.
- **The suite was not run**, because nothing it imports changed. `scripts/sensitivity_mass_diagnosis.py` was run
  from the worktree root with `atisim.__file__` printed and reproduces the bare-747 figures.


### Session 30 — the speed derivatives are read, and the overshoot is the thrust line

**Session 28 called digitising CR-2144 printed pp. 220–222 the highest-value item on the
plan, needing "no acquisition, no correspondence, and no new method". It is done.** Eight
Engauge `.dig` files arrived with this conversation. They were extracted through each sheet's
own three-point axis calibration (**295 placed points**), checked against Table IX-4 at every
circled flight condition, declared on a **copy** of the 747 through a new `Aircraft` seam,
retested against Table IX-5 by the engine's own `jacfwd`, and priced for reading error. §4
carries the entry. New: `atisim/cr2144_mach.py`, `atisim/data/cr2144_p220_222_digitised.csv`,
`scripts/cr2144_speed_derivatives.py`, `atisim/tests/test_cr2144_speed_derivatives.py` (22
tests). Changed: four fields on `Aircraft`, four lines in `aero.coefficients`.

**The suite is 843 passed, 1 skipped**, 897 s from the worktree root with the tree printed —
against **821 passed, 1 skipped** at session 28. **The arithmetic is exact: 821 + 22 = 843, so
nothing else moved.** That is the claim the seam has to support, and it was checked twice: the
197-test slice carrying the Fig. 8 vortex pin, the single-bit RK4 test, `test_cr2144_modes`,
`test_drag_polar`, `test_trim` and `test_conservation` passed with the seam present and
declared nowhere, before the full run. **No tolerance was touched.** (The wall clock is not
comparable with session 28's 34m37s — different machine load, same suite.)

**1. The Engauge CSV exports name their columns correctly and are unusable anyway.** Every
column reproduces its `.dig` curve at the shared Mach values to ≤6e-6, so the header order —
`SL, 40,000, 20,000` — is confirmed rather than assumed. But the thousands separators make the
header unparseable as CSV, and **every column is extrapolated past the end of its own drawn
curve** onto a shared grid: Cm_M's 40,000 ft column reaches **105**, CL_M's SL column −47.4.
The placed points are used instead, and two artefacts were dropped and reported: **nine SL
points copied verbatim into CL_M's 40,000 ft curve**, and eight exact repeats in CL_α's SL
curve. Five circled conditions fall outside a curve's drawn span and read NaN rather than being
extrapolated to reach them.

**2. The check found something about the source, not only about the reading.** Appendix A's
body-axis relations invert in closed form, so Table IX-4 implies a value at every circled
point. Reading them requires CR-2144's own **thrust trim**: Table IX-3 puts the thrust line
10 ft from the CG at 2.5° of incidence, so the aerodynamic `C_m` at trim is **not zero** and
M_u carries it. With that term seven Cm_M residuals have **RMS 0.0060**; without it **0.0382**.
The digitisation then agrees with the tables to **0.7–1.5%** on the smooth curves. **The
"poor scan" objection that `ASSUMPTIONS.md` C3 carried since session 12 does not survive
this**, and C3 is edited in place to say so.

**3. The retest: the improvement arrives, and it overshoots.** Phugoid ω_n **−18.13% →
+4.05%** and ζ **+13.16% → +4.55%**, with the short period unmoved to ≤0.17 points. **The
overshoot is not the reading** — Table IX-4's own implied set gives +3.86% / +5.26% — **it is
the thrust line the engine does not have.** Folding CR-2144's trim `C_m` into Cm_M closes both
to under 1%, and that compensated value is recorded as a diagnostic, not declared as data.

**4. A partial correction is worse than none, again.** The engine already has a drag Mach
slope — Korn/Lock's — and at FC9 it is **0.0477 per Mach against a sourced 0.0251–0.0276**.
Declaring CL_M and Cm_M while leaving it in place takes phugoid damping to **+21.3%**, worse
than shipping nothing; adding the sourced CD_M on top of it gives **+39.9%**. So `CD_M` is
declared net of the model's own slope, and the field's own comment says why. §7 recorded the
same signature for `CLα(M)` applied alone.

**5. The reading was priced, not asserted.** Interpolation scheme, leave-one-out, a pixel-level
Monte Carlo on the sheets' own scales, and the check residuals propagated as an empirical
error. **The frequency result is robust** — no case, including 3 px of scatter on every point,
brings phugoid ω_n within half of its shipped error. **The damping result is not**: one CD_M
point at M 0.8019 decides it, and phugoid ζ is known to about **±5 points** from this
digitisation. Quoted with that band everywhere it appears.

**6. What this session did NOT do, so the next one does not go looking.**

- **It DID declare the set on `boeing747`** — asked for directly, after the price below was
  measured rather than before. Phugoid ω_n **−18.13% → +4.05%**, ζ **+13.16% → +3.45%**; the
  CAT headline **68.2% → 64.5%** of the record, *further from it*, through a tangent
  extrapolated across the encounter's Mach excursion. The commit before the declaration holds
  the undeclared tree, and `BARE` in the script and tests is that entry with the seam shut, so
  both states stay measurable.
- It did not model a thrust line, refit Korn/Lock, or schedule anything on the α-family curves
  of pp. 220–221, which are read and used only to check the digitisation.
- It did not re-read p. 222 independently of the hand digitisation, and did not anchor CD_M or
  CL_M where trim C_D is only DECLARED (SL and 20,000 ft).
- The eight `.dig` originals are **tracked**, in `atisim/data/cr2144_dig/`, so the placement
  of every hand-set point survives the folder they arrived in, and `--dig-dir` re-derives the
  CSV from them.

**7. Session 29 is not in this document, and §0 now says where it is.** The sensitivity study
— design, module, four scripts, the two `sqrt(0)` repairs, C3 bounded at cruise, §1's headline
banded — is **8 commits ahead of `main` and 0 behind, on `origin` only**. `docs/DEVELOPMENT.md` rule 1b's
two-line check reads `refs/heads` and cannot see it. **Adding `refs/remotes/` to that check is
the one-line fix**, and this session did not edit `docs/DEVELOPMENT.md` to make it.

**8. Then the set was declared — on request, after points 3–5 were measured — and the price
was paid in the open.**

- **What moved on the shipped model.** Phugoid ω_n **−18.13% → +4.05%** and ζ **+13.16% →
  +3.45%**. The Hannibal headline went **68.2% → 64.5%** of the record: further from it.
- **The suite on the declared tree read 17 failed, 827 passed, 1 skipped**, and every failure
  was explained before a test changed. §4's session-30 entry, item 7, has the ledger:
  - 8 value pins were re-captured, each with its reason, and **no tolerance was widened**.
  - 5 claims about the pre-declaration engine now run on it (`BARE`, the seam shut), with
    their tolerances kept.
  - **1 was a real defect.** `verification.without_aerodynamics` did not zero the new
    fields, so the free-fall closed form broke by **0.205 m**. It is fixed at the function,
    which `scripts/sanity.py` also uses.
  - **3 mechanism claims weakened, all through Cm_M**, and each is now asserted on both
    entries:
    - line-vs-point longitudinal coupling 0.064 → **0.160** (claim < 0.10)
    - saturation elasticity 0.097 → **0.305** (claim < 0.20)
    - gust spacing, peak-to-peak 5.28 → **5.50 s**

    **The saturation bracket's sharp form survived**, moving to ×3.0–×3.25.
- **One script was printing a conclusion instead of a measurement.** `scripts/lateral.py`
  said "the longitudinal answer barely moves … nothing this project has concluded was resting
  on the missing dimension" whatever it measured. At +16% that is false; it now reads the
  verdict off the number.
- **Re-run on the declared tree:** `checkpoint.py`, `cat_validation.py`, `lateral.py` and this
  session's script.
- **Not re-run as scripts, so the §4 figures they own are pre-declaration:** `cat_bounds.py`,
  `cat_uncertainty.py` (the 72.7% corner), `cat_spectra.py`, `cat_ensemble.py`, `vortex.py`,
  `vortex_compare.py` and the LES limb. Their tests pass on the declared tree. The audit
  page's figures are session 28's, except the two rows annotated for session 30.
- **The suite on the final declared tree: 844 passed, 1 skipped**, 887 s from the worktree
  root with the tree printed. That is the pre-declaration 843 plus the one test this
  session's file gained when the declaration split its retest. Nothing new is skipped.

**9. Then the horizontal wind, asked for directly: "what else does the Mehta paper have that
can be compared?"** The inventory is §7's "Hannibal comparisons beyond the peak load". Item 1
is done, and §4's "The Hannibal horizontal wind" has the numbers.

- **The sign of z in `wind.py` is confirmed, not argued.** TM-102186 Fig. 7's horizontal MODEL
  line matches AtiSim to **1.58 kt RMS**, against **15.38 kt** with every core flipped. The
  vertical panel is the calibration control: **1.35 ft/s**, and blind to the sign by
  construction.
- **The fit misses a sustained tailwind rise.** The recorded along-track wind sits 9.7 kt below
  the fit through cores 3–4 and 12.1 kt above it after them.
- **The finding nobody went looking for: the headline double-counts the 747's own climb.**
  Mehta put cores 3 and 4 just above the DC-10's path. The simulated 747 climbs ~500–600 ft
  first and passes *above* them, so it meets tailwind gusts of the opposite sign. Replaying the
  fitted wind at the nominal altitude moves the load **64.5% → 75.0%** (bare entry 68.2% →
  77.5%).
  - **Not adopted.** ~~Which form is right depends on the DC-10's own altitude, which Parks Fig. 6
    plots and nobody has read (§8; §7 item 6, now the highest-value item).~~ Point 10 below read
    it: the DC-10 did not climb, so the replayed form is the right reading. The shipped
    headline is still not switched.
  - §1 carries the sensitivity beside the headline.
- **A claim this session made earlier was corrected in place.** §7's inventory had called the
  along-track wind "newly consequential" through Cm_M. **On the bare entry, removing it moves
  the load 13.6 points**, all through q̄. Cm_M only adds sensitivity to which side a core is on.
- **New:** `scripts/digitise_hannibal_horizontal_wind.py` (`--pdf` required),
  `scripts/hannibal_along_track_wind.py`, `atisim/data/tm102186_fig7_winds.csv`,
  `atisim/tests/test_hannibal_horizontal_wind.py` (8 tests), and
  `scripts/cr2144_report_figures.py`. The last one draws the four figures of the session's
  plain-language report, reusing the analysis script's own functions. `cat_validation.py`'s
  figure 03 had its two-line title printed over the panel titles; only the layout was fixed,
  and figure 03 was redrawn alone from `fl200_comparison()`. **No model code changed.**
- **Not done.**
  - B, D, F and G are not options in `wind.py`.
  - The CAT scripts were not re-run on them.
  - Mehta's own Figs. 5 and 9 were not digitised. ~~Nor was Parks Fig. 6.~~ Its altitude
    panel was, in point 10.
  - The horizontal flights hold fixed controls.
- **The suite with the horizontal-wind tests is 852 passed, 1 skipped**: 844 + 8, with nothing
  else moved. It ran for 1,295 s from the worktree root with the tree printed, on a machine
  also flying the figure scripts.

**10. Then the DC-10's altitude, asked for directly: "digitise the Parks Fig. 6 altitude".**
It answers point 9's open question. §4 has the entry, "The DC-10's altitude through the
vortex pair".

- **The DC-10 did not climb before cores 3 and 4.** Its inertially estimated altitude — the
  path the winds were computed along — was **36,985–36,996 ft** at the two passages, 17–61 ft
  below where the 747's run starts. The simulated 747 has climbed +512 and +596 ft by then.
- **The DC-10's climb came 21.6 s after the pair, past core 5**: +436 ft on the barometric
  trace. Through the spikes that trace swings 37,010–37,224 ft, which Parks attributes to
  pressure, not height.
- **So the replayed form (75.0% shipped, 77.5% bare) is the reading of Mehta's field.** The
  as-flown 64.5% counts a climb the DC-10 did not make; the 747's comes from holding its
  controls fixed in a sustained updraft. §1, §8 and the horizontal-wind entry are edited in
  place.
- **Decision not taken.** The shipped headline flight in `cat_validation.py` and its pins still
  fly the as-flown form. Switching it, or flying the 747 with an altitude hold, would move §1's
  number and re-pin dependants.
- **The reading was checked three ways, none used to set a parameter:**
  - the paper's load band: −0.96 / +1.90 g against "+1.7 to −1.0"
  - the paper's cruise altitude: 37,018 ft mean against 37,000
  - one clock across separately calibrated panels, and against TM-102186 Fig. 6: load minimum
    212.9 s, vertical-wind minimum 213.9 s, TM-102186's trough 213.1 s
- **Time to distance is a DECLARED band**, 817–1,034 ft/s (air-relative to ground), anchored on
  the deepest downdraft. The answer does not change across it.
- **Three extraction mistakes were caught by looking at the overlay, not by a number:**
  - the leader line cut the solid trace, which dropped the whole climb as specks
  - the peak ran above the panel's top tick and was clipped
  - a one-pixel fringe of the leader read as dashes, and the leader's cuts read as 37,000 ft
  
  Each is handled in the script, and its docstring says how.
- **New:** `scripts/digitise_parks_fig6_altitude.py` (`--pdf` required),
  `atisim/data/parks1985_fig6_altitude.csv`, and `atisim/tests/test_parks_fig6_altitude.py`
  (7 tests). `hannibal_along_track_wind.py`'s h7 now draws the DC-10's path beside the
  747's. **No model code changed.**
- **Not done.** Pitch and true airspeed from the same figure (§7 item 6 stays open for them).
  The 747 flown with an altitude hold. Any change to the shipped headline.
- **Suite: 859 passed, 1 skipped**: 852 plus the 7 new tests, with nothing else moved. It ran
  for 1,673 s from the worktree root with the tree printed.

**11. Then a document offered as "a base for the DC-10 derivatives": NTRS 19850002628.** It is
**NASA CR-3677**, and "DC-10 Derivatives" in its title means derivative *aircraft*. Session 25
had dismissed it for printing no baseline table. **That was half right**: there is no table,
but the no-winglet baseline is plotted in every high-speed stability figure. §7's phase-3 row
1 records what it has, what it lacks and where the PDF now lives.

- **What it could anchor:** the two quantities session 29 priced highest for a DC-10, after
  wing loading — the lift-curve slope (Fig. 23, tail-off, M 0.82) and pitch stiffness (Fig. 5
  and Figs. 7–9, tail-on). Plus the static lateral derivatives and a cruise buffet C_L.
- **Read by eye only, not digitised, and not to be quoted as data:** Fig. 23's tail-off slope
  looks like ~0.10/deg (~6/rad). Figs. 5 and 8 at M 0.80 look like C_mCL ≈ −0.26 and
  C_mα ≈ −0.033/deg, which would imply a tail-on slope near 7/rad. `boeing747` flies 4.94/rad
  from CR-2144 flight-condition data. That gap is exactly where the rigid-model caveat lives.
- **It cannot make a DC-10 entry on its own.** Damping, control power, drag, mass, inertia, CG
  and reference geometry would all be DECLARED. Wing loading stays unpinnable (§5). A hybrid
  entry would be weak ground for settling the sealed
  `dc10_does_not_close_the_hannibal_gap`, whose settlement names "a published DC-10 cruise
  derivative set".

**12. Then the thrust line, asked for directly: "add the engine thrust line and redo the SPPO".**
§4's first entry, "CR-2144's thrust line, declared", has the detail.

- **Declared on `boeing747`: LTH 10.0 ft, XI 2.50°** (Table IX-3). The trim is referenced to
  it, so it stays at α₀ with zero elevator and aerodynamic C_m = −0.0159.
- **Phugoid, against Table IX-5: +4.05% / +3.45% → −0.05% / +1.13%.** Short period
  −1.30% / −11.46%, unmoved. Reading uncertainty around the new value: frequency −3.2 … +2.9%
  and damping −2.9 … +6.7% at 1 px.
- **Two earlier statements were wrong and are corrected in place:**
  - `ASSUMPTIONS.md` C5 said CR-2144 tabulates no thrust line. Table IX-3 does, and CR-2144's
    tables agree with its 10 ft (Cm_M RMS 0.0063), not CR-114494's revised 5.70 ft (0.0167).
  - C5's verdict that modelling it buys ≤0.93% came from folding the moment into the
    aerodynamic C_m, which removes the M_u term that matters.
- **The Hannibal headline moves 0.1 point**: A 64.5% → 64.4%, F 75.0% → 75.3%. The line is a
  phugoid effect, and the encounter lasts a few seconds.
- **`aircraft.boeing747_without_thrust_line()`** rebuilds the old entry bit for bit, so every
  bare-747 claim still runs on the aeroplane it was measured on.
- **Suite on the declared tree: 9 failed, then fixed**, every one accounted for in §4: 3 claims
  moved onto the speed-only entry, 1 stale trim-C_L recomputation, 5 re-captured pins, and 2
  new tests. No tolerance widened, and no validated-baseline file moved. **Final: 861 passed,
  1 skipped.**
- **Re-run:** `cr2144_speed_derivatives.py`, `hannibal_along_track_wind.py` and
  `cr2144_report_figures.py`. **Not re-run:** the other CAT scripts, whose tests pass on the
  line.
- **Not done.** The approach and JSBSim 747 entries declare no line. Engine cant and spool
  dynamics are still absent.

**13. Then two decisions, asked for directly: "switch the headline to the replayed wind, use
5.7 ft even if the result is worse".** §4's first entry has the detail.

- **The arm is CR-114494's revised 5.70 ft.** Phugoid +1.69% / +2.83% against Table IX-5,
  where CR-2144's own 10 ft gave −0.05% / +1.13%. CR-2144's tables matching 10 ft better is
  recorded, not used to choose. A test pins the arm at 5.70 ft *and* asserts that 10 ft would
  read closer.
- **The headline flies Mehta's field replayed on its identified path**:
  `wind.on_identified_path`, used by `cat_validation.fly_mehta(..., replayed=True)` for the
  747 headline only. **75.3%** of the record, n_z −0.415 to +1.619 g, σ_n 0.684 g. The Fig. 8
  fleet still flies each aircraft at its own altitude: replaying a Cherokee pinned it inside a
  core at |α| 102°.
- **Measuring the switch reversed session 23c's saturation result, the finding of this step.**
  - Replayed, tripling V₀ gives n_z 2.89 g and elasticity +1.07.
  - +1.7 g is reached at ×1.15 V₀ with |α| 8.8°.
  - The favourable corner reaches 80.4%.
  - So "amplitude is excluded" and "cornered on the aircraft" were properties of flying the
    field at the 747's own altitude. Both are superseded in §4 and §5, and asserted where they
    were established, as flown.
- **Suite: 863 passed, 1 skipped.** Five pins were re-captured for the arm, and two claims
  split into as-flown and replayed halves. Two new tests. No tolerance widened, and no
  validated-baseline file moved.
- **Not re-run on either change:** `cat_bounds.py`, `cat_uncertainty.py`, `cat_spectra.py`,
  `cat_ensemble.py` and `lateral.py`. They still fly the field as flown, so every figure they
  own is that form. The sealed DC-10 prediction is untouched.

**14. Then `main` was merged in for the pull request, bringing session 29.**
- **Conflicts only in this document and `ASSUMPTIONS.md`.** Both sessions' records are kept:
  session 30's §4, §7 and §9 entries first, and the C3 row combined.
- **§1 now says session 29's 57.1–74.0% band was measured as flown on the pre-session-30 747.**
  It must not be quoted against the replayed 75.3%.
- **The branch was renamed to `session-30-speed-derivatives-hannibal` before pushing.**
- **Suite on the merged tree: 906 passed, 1 skipped**, 1,534 s from the worktree root with the
  tree printed. That is 863 plus session 29's 43, with nothing failing where the two meet:
  its `sqrt(0)` repairs sit in the Prandtl–Glauert block, and this branch's terms come after
  the drag build-up.

### Session 29 — the sensitivity study, designed and then run end to end

**The scope was settled by question, the design written before any number was measured, and
then all seven phases S0–S6 ran.** New: `atisim/sensitivity.py`, `test_sensitivity.py` (31
tests), `scripts/sensitivity_screen.py`, `scripts/sensitivity_load.py`,
`scripts/sensitivity_assumptions.py`, `scripts/sensitivity_ensemble.py`. Changed: one repair
to `aero.py` with a test beside it. §4 has two measured entries covering every phase;
`ASSUMPTIONS.md` C3 and §1's headline both moved.

**Three things this session establishes that the project did not have.**

**1. `ASSUMPTIONS.md` C3's Mach axis is bounded at cruise: −5.04% of the headline load.**
Wanted since session 12, declined every time as "needs a chart read off a poor scan". It
needed no chart read at all — the Prandtl–Glauert factor merged in session 28 supplies the
theory, and declaring `pg_mach_ref` at the derivative set's own tabulation Mach turns frozen
derivatives into Mach-scheduled ones. The excursion is **measured**: Mach **0.7187 to 0.8257,
span 0.1070**, because `aero.py` builds Mach from `vel_rel` and Mehta's array carries 26.5 m/s
of vertical gust. That is 3.5× the lee wave's ΔM and 0.27 of session 27's LES point, and it is
the first bound on this axis taken **at a condition the project actually claims**.

**2. §1's headline is no longer a bare point: 68.2%, band 57.1–74.0%, and the shortfall
survives all of it.** Every priced input moved to its bound — C3's Mach, E12's core radius,
`V₀` ±8.45% sourced, `r₀` ±15% declared, dt, E4, A2 — sums linearly to −16.33%/+8.43%. The
**top** of that band is 73.95% against a record of 100%. Session 23c reached the same
conclusion from the wind inputs alone and topped out at 72.7%; adding the **aircraft's** own
priced errors moves the ceiling by 1.3 points and changes nothing. **The 32% is not input
ignorance.** That is the question §1 has been unable to answer since session 7.

**3. The model could not be differentiated in its own coefficients at all, and §6(f) is why.**
`aero.py`'s Prandtl–Glauert factor is a `jnp.where` whose unselected branch, at the undeclared
sentinel `pg_mach_ref = -1.0`, computes `sqrt(0)` — whose forward-mode tangent is 0/0, **NaN
for any tangent, a zero one included**. The select discards the value, so no result this
project has ever quoted was wrong. It was found by `conftest.py`'s `jax_debug_nans`, on since
session 5; the probe scripts written the same hour ran outside pytest and produced finite
elasticities agreeing with central differences to 1e-10. **A NaN that is always discarded is
invisible to everything except a check that looks for NaNs.** Repaired with the standard
double-`where` and **proved** value-identical: the sha256 of `[trim, v̇_body, ω̇]` unchanged for
all seven registry aircraft, and the two frozen bit-pins failing with byte-identical values
before and after.

**What the screens found, and none of it is a comparison against a source.**

- **`CLa` +0.692 and `mass` −0.649 lead the headline load**, and `CLa` at the top is session
  27's LES result arriving from the other direction, at a different condition by a different
  method. *(Session 31: one lever, `CLa·q̄S/W`, counted twice. See §4's first entry.)*
- **A DECLARED constant is third.** `kappa_airfoil` — the Korn technology factor, "~0.87
  conventional" — scores −0.339 and outranks `Cmα`, `c` and every drag coefficient. With
  `sweep` and `t_over_c` the wave-drag trio sums to 0.454 against `CD0`'s 0.0087: on this run
  **compressible drag matters ~52× more than parasite drag**.
- **The mean aerodynamic chord `c` outranks `Cmα` for short-period frequency** at both
  conditions, and the **spiral mode is fragile only at cruise** — four coefficients above 2.9
  in elasticity against a worst of 1.57 at the approach.
- **Three of the six top load fields lose the tangent's sign or magnitude by ±10%**, for two
  physical reasons: wave drag **saturates** (Q is 1.83923 at both +10% and +25% of
  `kappa_airfoil`, to six figures) and the load extreme **jumps to a different core** (sample
  2699 → 2819 at −25% of `Cmα`, where the sign inverts). The ranking is valid at ±1–5%.
- **A ranking is a property of the STATISTIC as well as of the model.** Re-run against the
  Dryden ensemble rms, `CLa` and `mass` stay the top two — but **`c` and `Cmα` reverse sign**
  and `Clb` goes from an exact 0.000000 on the vortex to third place, because
  `dryden_field`'s v component is a real sideslip input where the vortex array has none.
- **Thirteen fields are exactly 0.000000 on the vortex load, every lateral derivative among
  them** — E10 measured rather than asserted. And the five 0.000% strip rows were **proved to
  be physics rather than plumbing**: peak |p| is exactly 0 on the point path and 2.12e-17 on
  the strip path, so the path ran and `n_z` genuinely cannot see a rolling moment.
- **The shipped dt = 0.01 under-reads the headline load by ~0.22%, and it is NOT integration
  error.** dt 0.02, 0.005 and 0.0025 agree with each other to 0.034% and all sit above dt 0.01.
  A convergent integration error is monotone in dt; this is not. It is **sample placement** —
  a peak-to-peak of a sampled signal depends on where the samples fall relative to the extremum.

**Two follow-ups after the study closed, both asked for directly.**

**`airframe.py`'s `sqrt(0)` is closed** — the second instance §6(f) flagged from reading the
code. It was checked by running it first and the flag was right: a `jvp` seeded in `CLa`, a
coefficient the chord does not depend on, raised `invalid value (nan) encountered in mul` at
every station count and every loading shape. Repaired the same way, value-identity proved the
same way, and the tip tangent asserted as an **exact zero** rather than merely finite, because
that is what the physics says and `isfinite` alone would pass a wrong repair.

**Fig. 8's pitch axis was investigated, and it corrects session 23d.** §4 has the entry. The
short form: 23d's "all but touching" and "a pitch-only reading of the chart would stop working"
are **both withdrawn**, because the gap between two clouds' extremes is not a measure of whether
they overlap — it shrinks with N at fixed σ_w (1.5690° → 0.6567° from N = 8 to 32 while the
separability never moves off 1.0000), and at σ_w = 4.0 it reads **−0.2995°** while only 4 pairs
in 1,024 are misordered. **Inside §1's envelope both axes separate perfectly**, and in
standardised terms it is the **load** axis that degrades faster, ×7.28 against pitch's ×3.20.
What binds is not the statistic but the envelope: peak |α| passes 10° at σ_w = 4.0, so the
sourced ceiling of 4.459 m/s is already outside it and **23d's upper limb always was**.
`atisim/response.py` gained `separability` and `standardised_difference` with **nine**
closed-form tests, and the 23d entry is edited in place rather than left standing beside
this. *(Their commit message says ten and is wrong by one. The count here is measured —
12 `def test_` before, 21 after — because a project whose discipline is that numbers carry
their provenance does not get to round its own test count.)*

**The suite after both follow-ups: 2 failed, 819 passed, 3 skipped in 1,359 s.** The same
two failures as every run this session — the exact-bit pins, failing with byte-identical
values, so nothing moved — and the **+11** over the 808 after S0–S6 is those 9 `response`
tests plus 2 in `test_airframe.py` (14 `def test_` before, 16 after), counted rather than
assumed. The wall time is longer only because a 384-flight ensemble was competing for the
same cores.

**And two failures of the RUN rather than of the physics, recorded because the next ensemble
will hit them.** `integrate.rollout` takes `wind_model` as a **static** argument and
`wind.dryden_vertical_field(σ, seed)` returns a fresh closure per seed, so every seed is a cache
miss and JAX **retains** an executable for each: the first grid died at roughly 144 of them with
`LLVM compilation error: Cannot allocate memory`. Clearing on every seed bounds it but costs
3× — 6.6 s per flight pair becomes 19 s, because the trim solve and the analysis vmap recompile
too — so it now clears every tenth seed. And a second grid lost **45 minutes with nothing on
disk** to a SIGTERM, because its output was piped through `tail` (which buffers until exit) and
its JSON was written once at the end. It now checkpoints after every intensity. **The real
repair is to give the Dryden field its phases as a traced argument** so one compilation serves
the ensemble; that is a change to `wind.py`'s field contract and is not a script's to make.

**Two design gates were wrong as written and are corrected at the change, per rule 3.** S1's
gate asked for session 11's fitted slopes to be reproduced to 2%; two came in at 2.9% and 4.2%,
and the reason is that an AD tangent and a least-squares slope over a 3–8× range are different
objects. The machinery gate is AD against a **central difference at the same point** — passed
at **5.2e-10** against 1e-6 — and moving the tangent to each swept range's centroid closes the
gap to the fit's own residual, so it is **curvature**, measured. S0's load gate asked for 1e-12
and got **bit-identity**.

**What was NOT done, so nobody goes looking.** **No interaction term was measured anywhere** —
every ranking here is one-at-a-time, Sobol and Morris were declined because their input
distributions are not sourced, and every table says so. C3's **α axis** is still unbounded.
~~`airframe.py:162` still carries the same `sqrt(0)` construction~~ — **closed later in the
same session; §6(f) has it, with the confirmation that it really did NaN.** The S5 ensemble is **N = 8**
and most of its table is not resolved at that N — the per-seed standard deviation is printed
beside every mean rather than averaged away. And the tabulation-Mach sensitivity (±0.05 worth
+3.0%/−15.8%) is deliberately **kept out of the budget**: §4 records that this 747 flies the
Mach its derivatives are tabulated at, so that is a sensitivity and not an uncertainty.

### Session 28 — the errors did not grow, the questions did

**An audit session, prompted by two questions from outside the record**: *is the engine valid,
and what is left unresolved?* — and *the project used to agree with the papers to a couple of
per cent and now reports 20–40%; what broke?* **No code was changed and no data was added.**
Everything below is either a re-measurement of something already in §4 or a reading of the
tree. §4 carries two new subsections.

**First, the regression hypothesis was tested and is false.** The suite runs **812 passed,
1 skipped, 761.79 s** from the worktree root (811 at session 26; the extra is session 27's
`boeing787_yoshimura` assertion). `checkpoint.py`, `cat_validation.py` and `lateral.py` were
re-run and every headline number reproduces to the digit — the 747's four modes, the
−0.398/+1.441 g Hannibal response, the 0.000/12.508/15.376° strip comparison. **And the
tolerances were checked in git rather than trusted:** the five validated-baseline files carry
three commits between them in the whole project history, and the only edits are the package
rename's imports and one *comment* corrected to a measured value. Nothing was ever loosened.

**Second, the real answer is a change of reference class, and it has a date.** Every
comparison the project has made is tier 0 (closed-form maths), tier 1–2 (the same document the
derivatives came from), tier 3 (another engine fed identical coefficients) or tier 4 (a
flight-data recorder or a published LES). Tiers 0–3 ran from 6–25 August and produce 1e-13% to
3.4%. Tier 4 begins at commit `c6b5342`, **1 September 2026**, and produces 20–42%. **Every
error above 10% in the ledger post-dates that commit; every couple-of-per-cent figure predates
it and still stands.** A 0.4% Dutch roll is the answer to "was CR-2144 transcribed correctly";
a 32% load shortfall is the answer to "does a rigid 747 reproduce a DC-10's recorder trace".
They were never the same measurement.

**Third, the largest identified physical cause of the tier-4 errors is one assumption, and it
is the oldest one in the register.** `ASSUMPTIONS.md` C3 — derivatives frozen, **Mach axis
UNBOUNDED**. Verified in the code this session: `aero.py` carries Mach into `wave_drag` and the
thrust ram term and **nowhere else**; there is no Prandtl–Glauert correction on any
coefficient. The project already holds two points that price it, and §4 now puts them in one
table: at M 0.8 / 6,096 m the short period is **+23.46%** with FC9's frozen derivatives and
**−0.60%** with derivatives tabulated at that condition — *a factor of 39, same code, same
solver, no turbulence involved* — and at the LES's M 0.406 the frozen `C_Lα` is 1.521× too
large by Prandtl–Glauert against a measured load ratio of 1.427.

**Fourth, three cases where the record got worse on purpose, which no reader can distinguish
from a regression without being told.** The session-16 mode table grew from two of CR-2144's
four published factors to all four (adding +14.4% and −1.4%) with 42 of 44 scalars
bit-identical across the pass; session 18's JSBSim short period read 0.04% → 3.95% → 1.30% →
0.04% where only the last is honest; and session 18's five fidelity improvements made the
phugoid **worse**, 3.33% → 6.58%, before session 19 localised 96% of it to a single derivative
the model structurally cannot carry. Add session 26's source correction, which cost 4.26% of
the headline load **downward**, and the trend in the published numbers is fully accounted for
without a single defect.

**Fifth, the strip-load path was judged rather than described.** It is **not on any published
path** — `integrate.step` defaults to `load_model=None` and a test asserts the strip and
tangent estimators disagree at the core edge precisely so the default cannot be switched
silently. Re-measured, it moves peak bank **+22.9%** **[Superseded session 32: a double count — `PROJECT.md` §6(h). With one path owning roll the strip effect is −2.5%, its F5 quadrature deficit.]** and the longitudinal load increment
**−0.5%**, and that single non-zero effect sits on a quadrature that F5 records as returning
**82.6%** of its own calibration at the shipped nine stations, in a channel E10 records as
validated against **nothing**. **Its real value is a negative result**: point-vs-strip on the
headline field is 0.000000 m, which is what lets §5 exclude gust resolution across the span
from the 68% shortfall. Kept, used sparingly, never quoted as a lateral load.

**Sixth, asked whether an SVD would help, three hypotheses were formed before running and one died.** `scripts/svd_probe.py`. The §5 absurd-trim defect is **not** a rank problem — the absurd roots are *better* conditioned than the healthy one (1.97e+01 and 2.26e+01 against 9.51e+01), because Newton found a genuine well-conditioned root of a function with several roots. `is_physical` stays. What does hold: `ASSUMPTIONS.md` F7's zero-authority channel gives **sigma_min = 0 exactly at the initial guess**, where the solver's own output is NaN and says nothing; and the identifiability of (`Cma`, `Cmq`, `Iyy`) against the short period has its unseen direction **0.1936 deg** from the (1,1,1)/sqrt(3) combination that `M_alpha` and `M_q` predict — i.e. session 27's B787 `I_yy` finding, derived rather than discovered by sweeping. The project has now met that same class of problem three times (the 737 and 747 `Cmq`/`Cmadot` splits at fit conditions 1.9e8 and 2.7e9, and the B787 `I_yy`) and solved it three ad-hoc ways. **An SVD does not finish the LES limb**, which is not compute-bound: all four domains are on disk, only a subvolume is ever loaded, the domain files carry one time step so there is nothing to decompose in time, and what blocked the comparison was the aeroplane.

**Seventh, and it changed what the session was.** Asked to check whether the recommendations
had already been done on another branch, the answer was **yes, and the biggest one had.**
`claude/new-session-943052` — one commit, 26 August, never merged — carried a working
Prandtl–Glauert implementation, `g(z)`, the geopotential fix and Mach-scheduled control
derivatives. **It is now rebased and merged**, at 821 passed / 1 skipped, after three pins
were re-captured against the new gravity and atmosphere. §4 has what moved; §0 exists because
of it, and `docs/DEVELOPMENT.md` gained rule 1b. **The repo audit also cut 39 branches to 19** — 20 were
fully merged and carried nothing — and rescued uncommitted work from five worktrees, the
largest a tail-arm gate refactor with tests.

**A correction to this entry's own opening.** It said "no code was changed". **That was true
when it was written and is no longer true**: the merge changes `aero.py`, `aircraft.py`,
`atmosphere.py`, `dynamics.py`, `trim.py`, `wind.py` and three pinned tests. The audit half of
the session changed nothing; the merge half changed the model. Read the two halves separately.

**What this session did NOT do**, so the next one does not go looking: it did not change any
code, coefficient, tolerance or datum; it did not re-run the LES limb or the JSBSim
generators; it did not fix §1's 67%-versus-68.1% mismatch (`known_issues.md` §12 owns it); and
it did not extend the Mach-axis bound beyond the two points already held. **The cheapest
remaining high-value item is unchanged from session 27** — a Prandtl–Glauert sweep across Mach
would turn C3's two points into a curve, and it needs no acquisition and no chart read.

### Session 27 — the recorded trace is read at last, and it moves the blame back to the wind

**The session began as presentation preparation and turned into three measurements**, because
the honest answer to *"how do we know?"* kept running into things the project had listed as
done-or-blocked that were neither.

**First, a process failure worth recording because it is the reason this entry exists.** The
work of digitising TM-102186 Fig. 6 and MIL-F-8785C Fig. 7 **had already been done** — in the
`weekly-summary-analysis-7520db` worktree, by `scripts/digitise_tm102186_fig6.py` and
`scripts/digitise_mil_f_8785c_fig7.py`, complete with three independent checks apiece. Both
scripts were **untracked**, their outputs were gitignored PNGs, and that worktree's
`PROJECT.md` still listed both figures as *open acquisitions*. **The measurement was not the
hard part; keeping it was.** `docs/DEVELOPMENT.md` now carries the rule that closes this, and
§"How to update this document" points at it.

**Five scripts were rescued into this tree. Two were re-run and are in §4; three were not,
and this row says so rather than implying they were.**

| Rescued script | State in this session |
|---|---|
| `scripts/digitise_tm102186_fig6.py` | **RE-RUN HERE, verified against this tree.** §4 carries the result |
| `scripts/digitise_mil_f_8785c_fig7.py` | **RE-RUN HERE, verified.** §4 carries the result; it settles a sealed prediction |
| `scripts/les_flight.py` | **ALREADY RUN in the weekly worktree — all four domains × two aircraft, with logs, `.npy` arrays and summary CSVs (`runs/cat/les-D0*.log`, `les-summary.csv`).** Not re-run here. **Its results are NOT admitted to §4** — see the input-matching audit below, which is why |
| `scripts/les_compare.py` | **ALREADY RUN** — `les-comparison.csv`, `14-les-resolution.png`. Same status |
| `scripts/yoshimura_flightsim.py` | **ALREADY RUN** — `12-yoshimura-flightsim.png`. Reads the simulated half of their Fig. 6, 604 flights, ~17 hours of record |

**A correction to this entry's own first draft.** It said these three were "not run". They had
been run, extensively, on 3–4 September — the same failure mode as the two digitisations, one
layer deeper: the *outputs* were there to be found and the entry recorded their absence
instead of reading them. **Check the artefacts before describing the state of the work.**

**The LES limb is the single most valuable unrecorded thing in the project.** It is now tracked
rather than stranded, and §4 records what its inputs would have to satisfy before any number
from it can be quoted.

**Second, the recorded trace.** §4 has the table. The headline is that the project quoted
`−1.0 / +1.7 g` for four sessions **without looking at the curve between them**, and the curve
says three things the two numbers could not: the paper's `+1.7` is the **second** peak (to
0.002 g) and not the largest, which is `+1.855`; the gust spacing is **5.60 s**, against which
the flown model reads **−4.3%** where the prose alone gave **+7.2%**; and — the one that
matters — **the vertical wind the record carries reaches −98.3 ft/s where Mehta's fitted field
gives the model only −86.8**, i.e. **the model is flying air 12% too gentle.** §5 had stopped
naming the wind as a candidate for the 32% shortfall. It is a candidate again, and now a
measured one.

**Third, the wing loading, which was supposed to be the cheap one.** Session 26 made *aircraft
type* the surviving explanation and flagged its own `1.3×` as unverified. Fetched this session:
Douglas **DAC-67803A** gives a SOURCED Series-10 taxi weight and **no wing area at all**, and
the ratio turns out to depend far more on **which DC-10 and how much fuel** than on any area
figure — **0.58× to 1.32×, straddling 1.0**, with the project's `1.3×` sitting in the extreme
corner (heavier variant, at MTOW). **So session 26's sign is withdrawn rather than confirmed**,
and the shortfall has no surviving single explanation. That is a worse position to be in and a
more honest one. The thing that would settle it is not another spec sheet but **the flight
record**, which §7 now lists.

**Fourth, a second sealed prediction settled, and it was right.** σ_severe(37 kft) =
**4.80 ± 0.12 m/s** against a predicted `> 4.46`. The band exists because SEVERE and 10⁻⁵ are
**one stroke of ink** at that altitude, and the whole band clears the threshold. JSBSim's
independent transcription of the same figure lands 0.5% away. **Two of three settled, both
right** — and the register's rules were followed exactly: only `status` and `outcome` moved,
the digest is unchanged, and `test_predictions.py`'s classification was corrected (the entry
moved SOURCE_GATED → RUN_GATED) **because the document arrived**, not to make anything pass.

**Fifth, and it reverses one of this session's own recommendations.** The wing loading was
argued here as the highest-value cheap fix. **It was not** — it was the item whose answer was
least likely to exist, and fetching it produced a withdrawal rather than a result. The
higher-value work turned out to be **reading what was already on disk**: two digitisations and
a full LES run, all complete, none recorded. **Prefer auditing held artefacts over acquiring
new ones** until the held ones are known to be exhausted.

**Sixth, the LES limb was audited against its inputs and its numbers are refused.** §4 has the
table. The short version: their aeroplane's short period is **0.1436 Hz** (recovered from their
own `fs.f90`, confirming the paper's 0.14), against `boeing747` at +5.4% and
`boeing737_approach` at **+82.9%** — and the field tops out at 5,215 m, so the 747 flies it
**2.63 band widths outside its validated envelope** with `checks.recovery_band` failing on
every row. A 1.4× rms ratio means nothing against that. **The field reader is sound and stays;
the load comparison is withdrawn until a registry entry is built from Yoshimura's own
derivatives.**

**Seventh, and it is the session's most useful finding: the LES discrepancy is probably ours,
and it is the oldest known limitation in the model.** Asked whether "AtiSim is only accurate
near cruise" had been fixed or quietly ignored, the answer is **neither fixed nor accounted
for**. `ASSUMPTIONS.md` C3 is explicit that derivatives are frozen and that the **Mach axis is
unbounded**; session 23 bounded only the *altitude* axis. The LES runs fly the 747 at
**M 0.406** against its M 0.80 linearisation — **ΔM = −0.393, thirteen times the largest
excursion C3 had ever been tested against** — while the `q̄` ratio is a harmless 0.965, so the
one bound that exists does not apply. `aero.py` carries Mach only into `wave_drag`, so
`C_Lα` is used unchanged; the Prandtl–Glauert ratio between the conditions is **1.521**; load
goes linearly as `C_Lα`; and the measured ratio on the two resolved domains is **1.427 and
1.420**. **The discrepancy is the size the frozen derivative predicts.** §4 has the arithmetic
and a one-run falsification test, which would also be the **first quantified point on the Mach
axis** — and it needs no chart read, which is the reason §7 has been declining that bound since
session 12.

**Eighth, the B787 gate opened the same day it was set.** Yoshimura's own `fs.f90` carries the
complete derivative set; the four numbers needed to turn it into a registry entry are now held,
and **three of them are confirmed by inverting their own `Z_a`** to a physical `C_Lα` = 4.847
rather than being taken on a spec sheet's word. The fourth, `I_yy`, is DECLARED and **provably
cannot matter** — it and `C_mα` enter only as their product `M_α`, verified invariant by
round-trip across a 4× range. *The unsourceable number turned out to be the unobservable one.*

**What this session did NOT do**, so the next one does not look for it: it did not fit `W/S` to
Fig. 6, it did not "correct" the trace's 0.951 g cruise datum, it did not move §4's headline
denominator off 2.70 g, it did not re-run the LES (no matched registry entry exists **yet** —
but every input for one is now held), and **it did not run the `C_Lα` falsification test**,
which is the cheapest high-value thing left and should be first.

### Session 26 — four papers arrive, and one of them says the project was flying the wrong vortex

**Parks et al. 1985 is in hand** — J. Aircraft **22**(2) 124–129, the paper §8 had listed as
unobtainable since the audit. With it, Bach & Parks 1987, and AFFDL-TR-70-101 (HICAT).

**The headline is a correction, not an addition.** `PARKS_CASES["hannibal"]` carried a
**hybrid**: Wingrove Fig. 4's 500 ft radius with Parks' 85 ft/s and Parks' 3500 ft spacing,
under a comment declaring the hybrid deliberate because Parks "has never been retrieved". He
says **600 ft, 85 ft/s, 3500 ft** — one coherent triple — and TM-102186's rival triple is
equally coherent and equally whole: **1000 ft diameter, 87 ft/s, 3400 ft**. The project had
been flying one number from each. `ASSUMPTIONS.md` **E12** now holds all four lineages with
the rule never to cross them, and a test names the three crossings that must not reappear.

**His abstract settles the question session 22 had to infer.** *"the vortex cores had
diameters in the range of 900 to 1200 ft"* — 900 = 2×450, 1200 = 2×600. And he quotes the
Scorer ratio as *a range across both cases*, "about 2.9 to 3.5"; this document said "Parks
quotes 2.92", which he does not.

**The reversal cost 4.26% of the headline load, downward** — JSBSim's Hannibal peak-to-peak
1.8969 → 1.8162 g, 70.3% → 67.3% of the recorded 2.70. `dw/dx = V₀/r₀` in a solid-body core,
so a bigger core at fixed `V₀` is a *gentler* gradient. It makes §5's shortfall worse, which
is the direction that argues nobody chose it for convenience.

**And the pin came back bit-for-bit.** `test_vortex_viz`'s Fig-8 coordinates return to
`2.239956221700959 / -1.2352174348304876` — the exact doubles captured *before* the logging
refactor, which session 22 had recorded as permanently lost. Not to a tolerance. Through the
trim solve, the rollout, the logging path and the windowing, one altered bit anywhere would
have landed somewhere else. It says the two changes moved the radius and nothing else.

**Both cases are DC-10s**, p. 127 and p. 128 — so one derivative set serves both, and §7's
highest-value acquisition is worth twice what that row assumed. **And they are not equally
good targets:** Parks calls case 2 *"not as good as … case 1"* and blames *"strong mountain
wave activity which influences the short-period wind pattern"* — a contaminant in exactly the
band a vortex-passage comparison measures. `wind.PARKS_FIT_QUALITY` carries both assessments.

**THE SIGN CHECK IS THE LARGEST RESULT, AND IT GOES AGAINST §5.** That entry names aircraft
type as the surviving explanation for the load shortfall, with a flagged note that the
"0.8× wing loading" figure "is unverified and may be inverted" and that "nothing downstream
depends on the number". **Nothing depended on the number; everything depends on its sign.**

- §4's row *"737 → 747, four times the mass … no effect"* was explained as *"wing loading
  cancels"*. It does not. `C_L,trim = (W/S)/q`, so `Δn ∝ 1/(W/S)`. **The measurement never
  tested it**: those two entries differ by 5.95× in mass and only **1.267×** in wing loading.
- A controlled sweep on one airframe at one condition settles it. Change mass by ×1.25 **and**
  area by ×1.25 — wing loading unchanged — and the load moves **0.1%**. Change either alone
  and it moves as 1/(W/S), slightly sub-inverse because an aircraft that responds more also
  pitches away more.
- Flown at **1.3× wing loading**, the direction `test_wind.py` states for the DC-10, the load
  falls to **56.4%** of the record against the baseline's 70.1%.

**So if that figure survives verification, a DC-10 set makes the shortfall worse by 14 points
and aircraft type is a term with the wrong sign.** The sealed prediction
`dc10_does_not_close_the_hannibal_gap` becomes *more* likely to land — for a reason its author
did not give, since its stated reasoning is the "wing loading cancels" line §4 now records as
false. **The entry is SEALED and has not been touched.** That is what the register is for.

**What settling it needs is small, and the paper says how small.** Bach & Parks Eq. (2) shows
`m` and `S` enter their identification only as `m/S`, and Eq. (4) gives about **0.05° of α per
1% of `C_L`** — so 1% of wing loading is 0.05° against excursions of 7–9°. A DC-10 wing loading
to a few per cent is enough; precision beyond that buys nothing. **And Parks Fig. 7 is the
array while Fig. 6 is the g trace for the same encounter, so `W/S` must not be fitted to
reproduce Fig. 6** — that would convert the only end-to-end check into a calibration.

**Bach & Parks does not help acquisition #1**: its two validation cases are an L-1011 and a
B-747SP. **HICAT does supply measured RMS gust exceedances** — the published exceedance data
phase 2 lacked — but over **45,000–70,000 ft** against this project's 33,000–41,000, so any
comparison is extrapolated; and it compares itself against MIL-A-8861A and Steiner's NASA U-2
data, so **Fig. 7's high-altitude end may descend from the same aircraft** and would not be an
independent check. Settling that needs ADA119421, which is not held.

**The buffet boundary is digitised** from `refs/NASA-CR-114494.pdf` p. 2.0-38 — the sheet
session 21 read `CL_max` off, whose *second* curve nobody had traced. `aircraft.buffet_cl`.
The method is in that file; the check that makes it credible is that the same pipeline run on
the **upper** curve reproduces session 21's published table to **rms 0.0052, worst 0.0063**,
four times inside their stated ±0.02, from an independent trace.

**It buys a number where §5 had an argument.** At 37,000 ft and M 0.80 the 747 trims at
`C_L` 0.572 against a boundary of 0.721 — **1.26 g to initial buffet** — and the Hannibal
encounter drives `n_z` to **1.68 g**. The run is well past initial buffet and the linear aero
cannot know it. Nothing in the force model reads the table; a hard ceiling is a kink and §4
records eleven tests going red the last time one was tried.

**`d9d4442` is merged.** The thrust-moment bound the audit found abandoned on
`claude/jolly-bhaskara-def594` since 14 August — 0.38° of equivalent elevator at cruise trim,
1.5% of pitch authority, from CR-114494 p. 19.0-2. `ASSUMPTIONS.md` C5 stops saying
"unquantified".

### Session 25 — a statistic with an N in it, and where the four documents actually are

Phases 2 and 3 of the finishing plan recorded in §7.

**Phase 2 — change what counts as agreement.** Every comparison in §4 matched a *peak* from
a single encounter: one realisation of a random process, whose error bar does not shrink
however much more work is done. `atisim/response.py` adds the two statistics that do —
a one-sided PSD of a response history and an upcrossing exceedance rate. Neither existed
anywhere in this tree. `wind.dryden_spectrum` is an **input** spectrum, which is a different
object, and the collision of names is worth keeping straight.

**The bet was sealed before the machinery was pointed at an aircraft**, which is what
commit `2e6fd2b` is for: `scripts/cat_spectra.py` was written and committed unrun, beside a
prediction that the ensemble response peak would land within 20% of the 747's own short
period, 0.1640 Hz. **Settled RIGHT, at both intensities and not comfortably** — 0.1400 Hz
at σ = 2.108 m/s and 0.1700 at 4.459, against a predicted [0.131, 0.197]. The lower one
sits less than one bin inside the band's edge, pulled down by the falling Dryden input
exactly as the sealed reasoning said it would compete. A tighter band would have been wrong.

**The robust form of the result is not the peak location, and the entry says so.** An
averaged periodogram at N = 32 is still noisy. What does not depend on N is a ratio of two
numbers: the **input** has strictly more energy at 0.05 Hz than at the short period, and
the **response** has more than 3× less. That reversal is the whole content of "the airframe
organises the load", and it is what `test_cat_spectra.py` asserts.

**And the headline run answers a question it could not previously be asked.** Mehta's array
is not uniform — its four core spacings force the aircraft at 0.0909, 0.1224, 0.1354 and
0.1886 Hz at once. The response peaks at **0.16885 Hz**: 0.23 of a bin from the airframe's
short period and 1.64 bins from the mean forcing. **The load follows the aeroplane, not the
array.**

**A tripwire fired, and it was right to.** `test_predictions.py` asserted that every sealed
claim was blocked on a source §5 names as unobtainable, on the reasoning that a claim
checkable from what is held "is not a prediction — it is a run someone has not done yet".
The new entry is exactly that kind. The rule the tripwire was protecting turns out not to
be the unobtainable source but the **seal preceding the run**, which is a fact about the git
history; an unobtainable source merely makes that ordering free. Rewritten as two admissible
classes, so adding an entry means classifying it rather than incrementing a count.

**What phase 2 did NOT do**, and the script prints it before the figure: it is not yet a
*comparison*. No published exceedance curve is held and no digitised acceleration history
exists here, so the observed half of Yoshimura's protocol cannot be run. §5 now carries both
as named acquisitions.

**And it produced one result that was nearly written up as physics.** The two ensembles
differ in `n_z` rms by **+5.46%** more than their σ ratio, which reads as a superlinear
response to gust intensity. It is not: `aero.py` is linear in α and `boeing747` carries no
`CL` table, so the aerodynamics *cannot* produce it. Measuring what else changed found it —
**with fixed controls there is nobody flying the aeroplane**, and over the 100 s record at
the upper σ it drifts **703 m in altitude and 13.1% in airspeed**, which is about a quarter
of dynamic pressure, with |α| reaching 8.38° against §1's 10° limit. So the upper-σ
ensemble is an average over flight conditions rather than a spectrum of one. Registered as
`ASSUMPTIONS.md` E11, reported per ensemble by the script from now on, and the reason the
lower-σ ensemble is the one to quote. **The fixed controls are not a defect** — §7's
discriminator is *defined* by them — which is why this is a limit to declare rather than a
bug to fix.

**Phase 3 — the four documents, and the search changed their order.**

| # | Document | Found? | What the search established |
|---|---|---|---|
| 1 | DC-10 cruise derivative set | **no** | Not in the open literature. Heffley's own library — the source of CR-2144 — has no DC-10, and its other compilation, CR-96008, is 1969, before the type flew. Every hit was the winglet programme (NTRS 19850002628, 19870008261), which reports that winglets *did not change* the stability characteristics and therefore prints no baseline table. ~~This is the highest-value item and it is the one nobody is giving away~~ **CORRECTED, session 30: no baseline *table*, but baseline *curves*.** NTRS 19850002628 is NASA CR-3677, and it plots the no-winglet DC-10-30 wind-tunnel model beside the winglet one: the tail-off lift curve at M 0.82, pitch stiffness vs Mach tail-on and tail-off, C_m vs α at M 0.60/0.80/0.90, and the three sideslip derivatives. A partial set, not a full one — §7, phase 3, row 1 |
| 2 | 747 buffet onset / nonlinear `C_L` | **no — and it turned out half of it was on the shelf** | **The boundary is held**: `refs/NASA-CR-114494.pdf` p. 2.0-38 is *"LIFT COEFFICIENT — BUFFET BOUNDARY AND C_Lmax"* and carries an *initial buffet boundary* curve beside the `C_Lmax` curve session 21 digitised off the same sheet. The web search was looking for something the folder already had. What is genuinely absent is the NONLINEAR LIFT CURVE, and the open 747 buffet literature is the **Shuttle Carrier Aircraft**: 0.03- and 0.046-scale tail-buffet wind-tunnel tests (NTRS 19750025089, 19770003191). That is the orbiter's wake on the empennage, not wing buffet onset at cruise — **the wrong buffet**, and quoting it would be worse than having nothing. NASA TN D-7131, whose title promises "Maneuver and Buffet Characteristics", is fighters |
| 3 | **MIL-F-8785C Fig. 7** | **YES, free** | The specification itself is public at everyspec.com, and DTIC's Background Information and User Guide (ADA119421) is on archive.org. **And an independent transcription already exists**: JSBSim's `FGWinds.cpp` carries the table under the comment *"this is Figure 7 from p. 49 of MIL-F-8785C"*, values in **ft/s**, rows a probability-of-exceedance index 1–7 with *"3=light, 4=moderate, 6=severe"* |
| 4 | Yoshimura figshare **21152203** | **YES — but not for what it was named for** | CC BY 4.0, one file `data.tar`, **17,942,056,960 bytes (17.9 GB)**: LES outputs at dx = 500/250/70/35 m, the flight-simulation code and its outputs, GrADS control and script files. **The three onboard flight records and the JAL PIREP are excluded by confidentiality agreement** — the paper says so and the dataset page repeats it |

**Two of those findings change the plan rather than execute it.**

- **Item 4 was listed as the route to a recorded acceleration history. It is not** — that is
  precisely the part that could not be released. What it *is* is a published **LES CAT wind
  field**, which is the non-circular field §5 has wanted since session 23: every load
  comparison in this project so far is flown through a field identified *from* the
  aircraft's own accelerations. That is a different and still substantial prize, at 17.9 GB.
- **Item 3 is free and immediate**, which promotes it above items 1 and 2 on cost even
  though the plan ranks it third on value. Better still, it can be done the way this project
  does everything else: **digitise the figure from the document, then check it against
  JSBSim's independent transcription**, which is the same two-independent-readings pattern
  as the vortex fields agreeing to 8.8e-10 m/s.

**The sealed prediction it would settle is left SEALED, deliberately.**
`mil_f_8785c_sigma_w_exceeds_the_mehta_ceiling` bets that the severe curve at 37,000 ft
exceeds 4.46 m/s. Interpolating JSBSim's transcription between its 35,000 ft and 45,000 ft
columns on the severe row gives **15.82 ft/s = 4.82 m/s**, which would settle it RIGHT.
**That is not this project's evidence and the entry is not being settled on it.** Its own
`settled_by` says "MIL-F-8785C Figure 7, **digitised** at 37,000 ft"; a third party's
transcription of a figure is not a digitisation of it, and §3's rule that every number
carries the table it came from exists for exactly this case. The number is recorded here as
what to *expect*, which is the honest use of it — and it makes the digitisation a check
rather than a discovery.

**Then both obtainable documents were fetched, and one of them closed something on
arrival.** `refs/MIL-F-8785C.pdf` (95 pp., 5 Nov 1980) and the 17.9 GB figshare tarball.
Figure 7 is confirmed on printed p. 49 — *"Turbulence exceedance probability"*, a rotated
scan whose σ axis reads **RMS TURBULENCE AMPLITUDE, σ (FT/SEC — TAS)**, which independently
confirms the units JSBSim's transcription uses. The digitisation itself is a session's work
on a poor scan and is deliberately not attempted in passing; the prediction stays sealed
until it is done properly.

**What did close, unasked, is the pair of Dryden spectral forms.** §3.7.1.2, printed p. 47,
prints all three components, and `wind.dryden_spectrum` and
`wind.dryden_longitudinal_spectrum` match it **verbatim** — including that the spec gives
`v` and `w` *identical* right-hand sides, which session 24 had inferred from the isotropy
relation and can now cite. `test_lateral.py` gains a check against the printed formula at
three arguments chosen so the values are arithmetic rather than a re-typing of the code:
Ω = 0 pins the factor of 2 on the longitudinal form, LΩ = 1 is where the two forms cross
exactly, and LΩ = 2 pins the 3 in the numerator against the square in the denominator by
which form is larger. **Holding the document closed the forms, not the intensity** — and
the forms were the part nobody had asked about.

**Then a second search, of the folder rather than the web, moved one more item off the
acquisition list.** The plan named "a digitised acceleration history" as something to
acquire, and this document's own §5 entry repeated it. **It is already held.** TM-102186
is `Reference_papers/19890016606.pdf`, in the project since session 23, and its **Figure 6**
is the Hannibal DFDR **g-load time history** — the recorded trace whose +1.7/−1.0 g extremes
this project has been quoting as `wind.TM102186_HANNIBAL_NZ` for two sessions **without ever
looking at the curve between them**. Yoshimura's equivalent is the part figshare withholds,
so this is the only recorded trace within reach, and it was never out of reach.

**The same page also supplies the phase-2 result's source.** TM-102186 p. 3-4 explains its
three-aircraft ordering by saying the pitch and load variations "are dependent upon the
relationship between the time span of the vortex traverse and the aircraft's short
oscillatory period". **That is the resonance claim the spectrum measures**, asserted by the
source and untestable here until this session — the project had no frequency-domain
statistic at all. §4 now cites it beside the measurement.

**Two lessons, and they are the same lesson.** `ASSUMPTIONS.md` E10 already records it —
*ask what every case has in common, not just what each one assumes* — and the folder version
is: **ask what the held papers contain, not only what they were fetched for.** Session 23
took TM-102186's two-number band and moved on. Session 24 inferred the `v`/`w` spectral
identity that MIL-F-8785C prints. Both were right; both left something on the table that a
second reading found.

**A third reading found a third one, and it changes the shape of phase 3.** The plan's
item 2 is "747 buffet onset / nonlinear `C_L`". `refs/NASA-CR-114494.pdf` p. 2.0-38 is
titled *"LIFT COEFFICIENT — BUFFET BOUNDARY AND C_Lmax"* and draws **two** curves against
Mach. Session 21 digitised the upper one — §4's `CL_MAX(M)` table cites that exact page —
and left the *initial buffet boundary* beside it untouched, with the sheet's own pointer to
§19 for revised data. **The web search in this session was looking for a document the
folder already held.**

**So phase 3's premise was wrong about three of its four items, and the corrected list is
much shorter:**

| Plan item | What it actually is |
|---|---|
| DC-10 cruise derivative set | **the only genuine acquisition.** Not in the open literature |
| 747 buffet onset | **held** — CR-114494 p. 2.0-38's second curve, plus §19. The *nonlinear* curve is unreachable and stays a §5 impossibility |
| MIL-F-8785C Fig. 7 | **held** — free, fetched this session |
| Yoshimura figshare | **obtainable, but not for the stated reason.** The recorded trace it was wanted for is TM-102186 Fig. 6, held since session 23 |

**Three digitisations and one acquisition**, where the plan wrote four acquisitions. Every
one of the three is a figure in a PDF already on disk, and each has a stated difficulty
rather than an assumed one — Fig. 7 is a rotated scan, TM-102186 Fig. 6 is a dense
oscillatory trace at low print resolution, and the buffet boundary is the easiest of the
three because it is a smooth single-valued curve of exactly the kind session 21 has already
done twice on that same sheet.

**The figshare tarball landed and verified**, 17,942,056,960 bytes to the byte and md5
`d23cbb3c…` against figshare's own checksum. Its shape is more useful than the headline
size suggests: four LES domains at 2.3, 3.6, 4.0 and 7.5 GB, and **`flightsim-data.tar.bz2`
at only 561 MB** — Yoshimura's own 2-D B787 simulation code and outputs, the *simulated*
half of their Fig. 6. **The 3% of the download that is not LES field is the part that can be
used first**, and it supports a cross-code response-spectrum comparison of exactly the kind
`scripts/vortex_compare.py` already does against JSBSim. The LES field remains the larger
prize and the harder one: it is the first field available to this project that was **not**
identified from an aircraft's own accelerations, which is the circularity §4's session-23
entry opens with.

**Suite: 788 → 807 passed, 1 skipped, 15m14s.** Twelve unit checks on the estimators, all
against signals whose answer is closed-form, six on the flown result — including a still-air
negative control in which every statistic in the new module returns exactly zero — and one
on the Dryden spectral forms against the specification now that it is held.

### Session 24 — the model gets a lateral dimension, and the record gets a claim

Two phases of the plan written at the end of session 23d.

**Phase 0 — bank what is already true.** §1 now carries a formal **validation claim**
with its envelope attached: aircraft class, Mach and altitude band, |α| limit, gust length
scale in spans, and *longitudinal only*. The project had a great deal of evidence and no
single statement of what it added up to, which meant the honest narrow claim was going
unmade while a broader one was available to be inferred by accident.

**And writing it found the gap.** Every wind field in the project was a function of
along-track distance alone — each for a different reason, which is why nobody had noticed
the pattern. `strip_roll_moment` integrated to exactly zero on all of them, the lateral
modes were validated as eigenvalues and never excited, and `Encounter` had no channel that
could have recorded a roll. **The model was longitudinal by construction and no document
said so.** Now §5 does.

**Phase 1 — give it the dimension.** Three pieces:

- `wind.dryden_field` adds the `u` and `v` components. The spec is not in the folder, so
  the check that matters is that the two spectral forms belong to one isotropic field —
  `Φ_t = ½(Φ_u − Ω dΦ_u/dΩ)`, satisfied to **1.3e-16**.
- `vortex_viz.Encounter` gains `phi`, `beta`, `p`, `r` and `p_gust`, trailing and
  defaulted so nothing existing moves.
- `wind.line_vortex_wind` writes Parks' vortex as lines in space rather than as a formula
  on the flight path.

**The result, and it is not small.** Through Mehta's own field the 747 reaches **12.5° of
bank, 15.4° with strip loads** — against **exactly 0.000°** from the point model. And
`loads.strip_increment`, built in session 14, **moves a reported number for the first
time**: +22.9% on peak bank **[Superseded session 32: a double count — `PROJECT.md` §6(h). With one path owning roll the strip effect is −2.5%, its F5 quadrature deficit.]**, where it had previously changed every result by exactly
0.000000. The longitudinal answer moves +5.9%, so nothing already concluded was resting
on the missing dimension.

**A real simplification, found on the way.** Parks' model is 2-D in the plane
perpendicular to the vortex *lines*; `vortex_wind` returns his horizontal magnitude along
the *flight path*. At Mehta's 31° the true perturbation is that magnitude rotated by ψ, so
**sin 31° = 0.515 of it lies across the path** — the only sideslip input the field has
ever had, discarded since session 3.

**A bug this work made, and caught.** `cos_dpsi` predates `sin_dpsi`, so
`mehta_hannibal_array` carried a cosine with the sine at its 0.0 default: a vortex axis of
length cos ψ rather than 1, scaling the whole induced velocity and looking exactly like
physics. Caught because the script compares the two forms in a geometry where they must be
identical and got 7 m/s. Fixed in the constructor *and* by normalising the axis, so a
mismatched pair is a wrong angle rather than a wrong magnitude.

**What this does NOT establish, and the documents say so three times.** No source held
here records a lateral CAT response. Every lateral number is a capability demonstration.
§1's claim excludes lateral behaviour explicitly, §5 repeats it, and `test_lateral.py`
carries a negative control on every non-zero assertion — because §6's lesson is that a
suite exercising only the easy case cannot see the bug, and a longitudinal-only suite
hides lateral bugs exactly the way a still-air one hid air-relative bugs for three
sessions.

### Session 23d — a second engine on the headline field, error bars on Fig. 8, and the first sealed bets

Three follow-ups, and one of them qualifies a conclusion this document has been
carrying since session 22.

**What was built.** A five-vortex oblique case in `scripts/gen_jsbsim_vortex_reference.py`
and its `<cores>` schema in `atisim/jsbsim_vortex_ref.py`; `scripts/cat_ensemble.py` with
figure `07-ensemble.png`; `atisim/predictions.py` and `test_predictions.py`; four new
tests in `test_cat_validation.py`; four tests in `test_jsbsim_vortex.py` rescoped.

**1. Neither engine reaches the record on the field the headline result flies.** JSBSim
given Mehta's five-vortex array, the same aircraft and the same starting state reaches
**74.6%** of the DC-10's recorded peak-to-peak; AtiSim reaches **67.1%**. That closes the
solver as an explanation *on the headline field* rather than by inference from the
adjacent Parks cases. The two field implementations — numpy in the generator, JAX in
`wind.vortex_wind` — agree to **8.8e-10 m/s**, including the five-way superposition and
the 31° obliquity, so nothing here rests on a bug shared by both sides.

**2. And the same run qualifies session 22.** "Both engines under-predict by the same
amount" was measured on single cores, whose window is 1.3 s — a fifth of a short period.
On the array, 34 s and five short periods, the engines **select different cores** as the
worst: AtiSim's peak is 2.2% *higher* than JSBSim's while its trough is 3.5× shallower,
and the entire −10.1% span disagreement is the trough. The conclusion survives in the form
that matters — neither engine reaches — but "by the same amount" was a property of the
short window.

**A hypothesis tested and failed, recorded so nobody repeats it.** The array run is 45 s at
fixed throttle and JSBSim's airspeed swings −5.5% across the window, against −0.009% on the
single core; `n_z ∝ V²`, so drift looked like the whole story. Shrinking the window until
the drift is −0.19% leaves the disagreement at −7.6%, and AtiSim's span is identical at
every window width from ±2 to ±32 core radii. It is not the drift. §8 carries what is left.

**3. §7 step 6 is finally done, and its margin is the interesting part.** The ordering
holds **16/16** at both ends of the σ_w range derived in 23c from Mehta's residual — the
intensity is sourced, not picked. But raising σ_w by 2.1× collapses the vortex/updraft
**pitch** gap by 15×, to **0.060°**. The **load** gap survives at 0.343 g. Fig. 8 is a
two-dimensional discriminator so the categories still separate, but at the top of the
sourced range they separate on load alone. Stated with its limits: the manoeuvre limb is
noiseless by category definition, which helps the three-way ordering; and a 16-seed gap
estimates the tails of a distribution whose extremes grow with sample size, so
"non-overlapping" means *marginal*.

**4. The first sealed predictions.** Every number in §4 is retrodictive — the paper was
open beside the model. `atisim/predictions.py` is where that stops being the only kind of
evidence here. Two entries, sealed against tree `0c72200`, both blocked on sources §5
already names as unobtainable. The first **bets against this document**: §5 now names
aircraft type as the surviving explanation for the 32% and a DC-10 derivative set as the
highest-value acquisition, and the prediction says that acquisition will *not* close the
gap, on the strength of §4's measured null that quadrupling mass moved the load 0.6%.
One of the two is wrong.

**A drafting mistake worth keeping.** The first version computed each prediction's digest
at import from the fields it hashes, which makes the seal vacuous — change a claim and the
digest follows it, and the check can never fail. The digests are literals now, and
`test_a_computed_digest_would_have_made_this_check_vacuous` is the negative control.

**Four existing tests were rescoped, not relaxed.** `test_jsbsim_vortex.py` characterised
single cores in four places — one sign change in the gust, a 1% run-in speed bound, a fixed
key set, a single-core field reconciliation. The array violates three of them legitimately.
Each was scoped to `cores is None` and given the array's own assertion beside it; the 1%
bound was **not** widened to admit a 2.34% array value, because 1% characterises a
15-radius run-in to one core and says nothing about five.

### Session 23c — error bars on both sides, and the shortfall stops being a wind problem

Three follow-ups, all aimed at the same weakness: **every load comparison in this project
is partly circular.** The vortex parameters were identified *from* the recorded
accelerations, through Parks' and Mehta's aircraft model, so predicting those
accelerations back tests the composition of two models rather than this one.

**What was built.** `wind.MEHTA_COST_STARTUP`, `wind.MEHTA_COST`,
`wind.MEHTA_COST_SATURATES_AT`, `wind.mehta_residual_ceiling`,
`wind.mehta_unmodelled_wind`; `scripts/cat_uncertainty.py` with figure
`06-uncertainty.png`; eight tests; four ledger entries.

**Three results.**

1. **A non-circular channel reproduces.** Gust *spacing* is set by the fitted core
   positions and the aircraft's speed, not by the fitted strengths — measured, not
   asserted: the field is exactly linear in `V₀`, so scaling it by 0.5, 1.5 or 3.0 moves
   every wind value and **not one turning point**. The flown run gives **5.29–5.43 s**
   depending on which reading of "apart" is taken, against TM-102186's prose "about
   5 sec". The three readings span 0.14 s, so the answer belongs to the field and not to
   the definition. The 6% residual is a speed proxy — 5.0 s needs M 0.848 and this 747
   flies M 0.800 — and the model is **slow**, which is the direction that mismatch
   requires.

2. **Mehta's own fit residual bounds the field form, and it needed re-reading the
   Appendix to get right.** Eq. (A3) carries a **1/N**: the cost is a *mean* square, so
   `J` = 214 converts to a 4.46 m/s RMS residual **without knowing `N`**, which the paper
   never states. Two readings of that formula differ by ~65× and give opposite
   conclusions, so it was rendered from the PDF rather than guessed. Subtracting Lester's
   reconstruction error leaves the unmodelled wind at **2.11–4.46 m/s**.

3. **The 32% shortfall is not in the wind at all.** The propagated input band tops out at
   **72.7%** of the recorded peak-to-peak, and the peak load is **saturated**: tripling
   `V₀` moves the up-increment from 0.441 to 0.526 g against a recorded 0.70, elasticity
   under 0.15 and changing sign, with peak |α| still near 8°. The aircraft pitches away
   and sheds the gust — Fig. 8's own mechanism, seen from the inside. Bracketing finer
   than the first sweep did turned this from a bound into an **exclusion**: the peak
   first reaches +1.7 g between ×3.25 and ×3.50, and |α| leaves the linear range in the
   same interval, so no gust strength both reaches the record and may be believed.

**This weakens a previous session's conclusion rather than extending it.** Session 23b
reported σ_w ≈ 4–5 m/s closing the upper extreme. That still happens, but 4.46 m/s is now
known to be the **ceiling** of what Mehta's residual permits — it requires the entire
residual to be vertical, unmodelled, and free of reconstruction error — and the ceiling
run sits *on* the 10° edge of the linear range. At the sourced lower bound the run does
not reach the band at all. The Dryden explanation is not excluded, but it now needs the
top of its own range rather than the middle.

**A mistake of this document's own, found and fixed.** §3 and §8 both described Mehta's
"documented cost history (482 → 214)". 482 is his *manual startup estimate*; the converged
series is 355 → 303 → 226 → 214. Pairing a startup value with a converged one as a single
history is precisely the error §5 flags TM-102186 for making with Schultz's Table 1 —
made here, by this project, about the same paper family. Now separated in code, with a
test asserting the separation.

**Guards that fired.** `test_the_provenance_ledger_does_not_cover_the_source_modules`
failed the build on two unledgered constants, and was fixed by writing entries. A first
draft of the invariance test used an absolute gust threshold against a scaled field, so it
compared thresholds rather than gusts; caught because the extremum count changed with
scale.

**What is now left.** The shortfall is cornered on the **aircraft** — a DC-10 record flown
by a 747, with no buffet-onset data. That is §5's structural entry, and it has stopped
being a caveat and become the surviving explanation. A DC-10 derivative set is the
highest-value acquisition on the list. Also flagged, not fixed: §5's "roughly 0.8× the
wing loading" is unverified and may be inverted — AtiSim's 747 is W/S ≈ 115.8 lb/ft² and
no sourced DC-10 figure is held.

### Session 23 — eleven papers arrive, and four "not quantified" entries stop being that

**What came in.** Twelve PDFs in `Reference_papers`, eleven distinct works. Seven are new
to the project: Mehta 1987, NASA TM-102186, Lester/Sen/Bach 1989, Misaka 2008, Yoshimura
2022 and 2023, Schultz 1990, plus Bach 1991's SMACK manual and Palacios & Cesnik ch. 4.
Loving 1965 is context with no reusable dataset; Stengel was already load-bearing; one
file is a duplicate.

**The theme: four §5 entries that said "not quantified" or "declared, not sourced" were
wrong to say so, and closing them needed no new measurement — only a source that had
not been read.**

| Entry | Was | Now |
|---|---|---|
| Hannibal core radius | conflict, unarbitrable | **500 ft** — Mehta's later five-vortex refit, reported again by TM-102186. 600 ft is Parks' own earlier fit, self-consistent with his Scorer ratio, and is superseded rather than corrected |
| vortex parameter band | ±25%, reasoned | **8.45%**, from Lester Table 1's propagated RMS errors |
| flexible-airframe mismatch | "cannot be produced from sources the project holds" | bounded at constant Mach: −17% to −61% across 2.48× q̄ |
| lee-wave wavelength | declared, unbounded | 12% from a measured 22 km at the right altitude regime |

**What was built.** `VortexArray.cos_dpsi` (the oblique traverse the module had refused,
now transcribed from a source that constrains it); `wind.mehta_hannibal_array`, the only
wind field in the project that declares nothing; `vortex_viz.fly_in_moving_air`;
`checks.rms_normal_load`; twelve `REFERENCES` rows for a third CR-2144 flight condition;
`scripts/cat_validation.py` and 23 tests in `test_cat_validation.py`.

**Three results worth remembering.**

1. **The 747 lands inside the DC-10's measured load band** on Mehta's own identified
   field, at 68% of its peak-to-peak. Absolute agreement was never available; being
   *inside* is what was reachable and it was reached.
2. **The paper's non-obvious claim reproduces.** TM-102186 Fig. 8's `n_z` minimum
   ordering runs *opposite* to its pitch ordering — the aircraft that pitches least eats
   the most load — and the model gets that reversal, with the cherokee/RPV pair agreeing
   inside the digitisation error. The mechanism behind it collapses monotonically across
   all six aircraft, six for six.
3. **A 23.5% short-period error was traced to its cause in one experiment.** Swapping the
   derivative set for the one CR-2144 tabulates at that condition took it to 0.6%, and the
   remaining damping shortfall was *predicted* to −12.2% from the missing `C_mα̇` against a
   measured −11.7%. Neither number is a fit.

**Two mistakes in the sources, and three of the project's own guards firing.** TM-102186
quotes Schultz's initial estimates as his results; Yoshimura 2022 misreads its own `s⁻¹`
as `Hz`. And every safety net this project has built caught something real:

| Guard | What it caught |
|---|---|
| §10's tree check | `scripts/cat_validation.py` printed `atisim.__file__` on its first run and found itself importing the **main checkout**. The failure mode that produces no error message, producing one |
| `test_the_provenance_ledger_does_not_cover_the_source_modules` | eleven new `wind.py` constants added without ledger entries. **Failed the build**, exactly as §2 promises. Fixed by adding the entries, never by widening the baseline |
| `test_checks_round_trip_with_their_kind` | the new severity check returning `NaN` on a short run, which is not valid strict JSON. Fixed by omitting the check rather than fabricating a value |

**Two of my own numbers were wrong and were caught by recomputing them.** The dynamic
pressure ratio between the two 747 conditions was written as 2.16× — that is the *density*
ratio; at constant Mach `q̄ = ½γM²p`, so it is **2.48×**. And the Mehta/Parks strength
spread was written as 2.4% when 86.8/85 is **2.1%** (2.4% is TM-102186's rounded 87).
Both were in six files before being corrected. The air-vs-inertial α divergence was
likewise read off a plot at 7.0° and is **6.85°** computed; the script now prints it.

**What did not change.** No frozen §4 baseline moved. `PARKS_CASES` is untouched and still
a hybrid, deliberately; the coherent single-source pair lives beside it and the 2.1%
spread is measured rather than argued. The orderings-only rule on vortex conclusions
stands — everything above is an ordering, a band, or an attribution.

### Session 23b — five follow-ups, and the shortfall is cornered rather than closed

The session-23 pass ended with the Mehta run reaching 68% of the DC-10's recorded load
and five named next steps. All five were run.

| | Asked | Answered |
|---|---|---|
| 1 | restore the α̇ terms | `ζ` at M 0.8 / 6,096 m goes **−11.7% → +0.14%**. Given CR-2144's own derivatives for a condition, this model now reproduces CR-2144's own short period to 0.6% and 0.14% |
| 2 | declare the 747's valid band | `[0.70, 0.90]` M, `[35,000, 45,000]` ft. The altitude edges are **derived**, not declared — a two-point interpolation in q̄ that self-checks to +24.2% against a measured +23.5% |

**Item 2 changed a documented rule, which is worth flagging loudly.** The rule was *only an
entry that IS a fit may carry a band*, and it rested on the reasoning that a linear
derivative set from CR-2144 is "valid across the ordinary linear range" — that its
limitation is on α and not on flight condition. For a rigid aircraft that is right:
non-dimensional derivatives depend on Mach and geometry, not altitude.

**The measurement says it is false for this data.** At constant M 0.80 CR-2144's own 747
derivatives move −17% to −61% between 40,000 and 20,000 ft, and the short period that
follows is 23.5% high in silence. The tabulated set does have a condition range; there
were simply never two conditions to compare before.

The replacement rule is what the old test was really enforcing: **a band may be declared
only where its edges were established by measurement**, and there are now two ways to
establish them — a JSBSim fit sweep, or an interpolation between two held conditions of
the same document. `aircraft.DECLARES_A_BAND` is a separate set from
`RECOVERED_FROM_JSBSIM` precisely because membership of the latter implies two further
things (the source CL(α) table, trimming inside a table segment) that `boeing747` does not
have and must not be given.

Three guards fired on this change and all three were right: the band test above, the
`recovery_band` report/gate test (whose specimen was `boeing747` and is now the approach
entry, which still declares none), and the provenance ledger over the two new Dryden
constants.
| 3 | fly Lester's Greenland 747 | the recorded climb and the recorded g-load imply lee-wave amplitudes **28× apart** |
| 4 | Dryden on top of Mehta | σ_w ≈ 4–5 m/s closes the **upper** extreme; nothing closes the lower one |
| 5 | bound the point gust and the step | **≤4.4%** of a 32% gap. Strip loads move it by exactly zero |

**The shortfall is now cornered.** It is not numerics (0.14% over an 8× dt range), not the
point-sampled gust (≤4.4%), and not the strip path (0.00%). Adding the random component
Mehta's fit explicitly excludes closes the positive extreme at a plausible intensity and
leaves the negative one untouched. What remains unexplained is a **downward** excursion,
on an aircraft that is not the one in the record.

**Item 3 is the result worth keeping.** It is the first load comparison in the project
flown by the aircraft type the record is actually of, and it produces a physical
conclusion rather than a percentage: the smooth wave reproduces the slow response at an
amplitude Doyle measured and misses the fast one by a factor of 28, so the accelerations
came from the wave breaking down — which is what Lester's own critical-level reading says.
`wind.LeeWave` is missing the breakdown, not a coefficient.

**Two things were deliberately NOT done.** `boeing747` gets no α̇ terms, because no source
supplies them at flight condition 9 and Table A2's are 2.48× of q̄ away. `boeing747_approach`
gets none either, although Table IX-2 tabulates them: they take the short period from
+1.4%/−5.5% to −0.3%/+0.1% and the phugoid damping from −4.6% to **+8.4%**, past the 5%
`test_validation.py` asserts. Adopting them means re-pinning a tolerance to admit a change,
and B5 is the precedent for refusing that. The measurement is kept as a test.

### Session 21 — the Wingrove paper arrives, and JSBSim gains a 747

**The source the audit could not obtain is now in hand.** Wingrove & Bach, *Severe
Turbulence and Maneuvering from Airline Flight Records*, J. Aircraft **31**(4), Jul-Aug
1994, 753-760. `AUDIT.md` rows 19, 168 and 393 marked `UPDRAFT_W0`, `UPDRAFT_SECONDS`, the
5.2 deg pitch figure, the +0.66/-1.58 g and the Fig. 8 discriminator as
`unverifiable -- source not available`. All five check out verbatim; closing those rows is
still outstanding.

**The paper names no aircraft.** All twelve incidents are "modern airliners" carrying a
DFDR; Table 1 gives date, location and altitude, Table 2 gives load increments, and no
table, figure or sentence identifies an airframe. Any comparison against it therefore
chooses its own aircraft, and this project's choice is the JSBSim-intersection below.

**Fig. 4 supplies a third vortex case and contradicts `PARKS_CASES` on a second.** The
figure quotes core DIAMETERS: Hannibal 1000 ft, Morton 900, Cimarron 900, at 85, 70 and
50 ft/s. Morton halves to 450 ft, which is `wind.py`'s Morton radius to the digit and
settles that the column is a diameter. Hannibal halves to **500 ft where `wind.py` says
600**, and Parks 1985 has still never been obtained, so nothing arbitrates. Both are kept
in separate dicts, `PARKS_CASES` and the new `WINGROVE_FIG4_CASES`, and both will be flown.
Cimarron exists only in this paper, and it is the only case with published time histories.

**JSBSim cannot be asked to carry a gust gradient.** Its property catalog reports
`atmosphere/{p,q,r}-turb-rad_sec` as READ-ONLY, and a write to `q-turb-rad_sec` reads back
0.0 after one step. Every writable wind input -- `wind-*-fps`, `gust-*-fps`, `turb-*-fps`,
`cosine-gust/*` -- is translational, sampled at one point. Translational injection does
work and was checked against arithmetic: `wind-down-fps = -50` at M 0.78 / 30,000 ft moved
alpha 3.687 deg against a predicted `atan(50/776) = 3.69`. So the gradient arm of the
vortex comparison measures what JSBSim omits, using atisim as the instrument, and cannot
be a JSBSim-to-JSBSim delta.

**`boeing747_jsbsim` joins the registry, and `boeing747` is untouched.** JSBSim's B747 is
not the CR-2144 aeroplane: 249,974 kg against 288,773 (15.5%), 64.46 m of span against
59.64 (8.1%), 524.7 m^2 against 511.0 (2.7%) -- while sharing an inertia tensor to under
0.1%. Since n = L/W, a load-factor difference between the two engines flying "a 747" would
have been dominated by that mass gap. The new entry carries JSBSim's own numbers, recovered
at 38,000 ft / M 0.80 by `scripts/gen_jsbsim_747.py`, which imports the 737's recovery
machinery rather than copying it.

**It is not a credible 747 and the entry says so.** B747.xml is `release="ALPHA"`, author
"Unknown", and its CLalpha table shares its first three points with 737.xml's -- both give
CLa = 4.3478 /rad. That is one Aeromatic template used twice. It costs the cross-code
comparison nothing and makes any comparison of that entry against flight data meaningless;
`boeing747` remains the aeroplane for that.

**Two things were caught by tests rather than by inspection**, and both are recorded because
neither was loud:

- **The Ixz sign was wrong.** `inertia_tensor` negates its argument, and JSBSim's
  `inertia/ixz-slugs_ft2` is already the tensor element, so the engine's -970000 had to be
  passed as +969999.99. Passing it as reported flipped the cross-product term -- the exact
  failure the 737 shipped with once, worth 1.6-3.2% on the lateral modes and inside every
  layer tolerance. `test_mass_and_inertia_match_the_engine` compares the ASSEMBLED tensor
  against the reference rather than the argument against a remembered convention, which is
  the only form of the check that catches it.
- **A unit constant differing in the 8th digit.** slug ft^2 values converted with
  1.35581796190452 instead of `units.SLUG_FT2_TO_KG_M2` = 1.3558179483314003 left a 1e-8
  relative gap. Recomputed with the project's own constant, Izz and Ixz land on B747.xml's
  stated 4.97e+07 and -970000 to ten digits.

**The recovery lands on B747.xml's own constants**, which is the check that it is a recovery
rather than a plausible fit: CYb, Clb, Clp, Cldr, Cnb, Cndr and CLde exactly, Cma -0.699965
against -0.7000, Clda 0.0732 against the Mach-scheduled table's M 0.80 value. Cmq and
Cmadot come back as -21.0055 and -3.9945 against -21 and -4 -- each 0.0055 out, equal and
opposite, at a design condition number of 2.7e9 -- while their SUM is -25.000000. The same
q/alphadot collinearity the 737 records, and the sum is the quantity that carries weight.

**The refactor that made the reuse possible changed nothing**, and that is measured: both
737 reference XMLs regenerate to the same blob hash they had before
(`c0d5522a...`, `44eb69f7...`).

**The two engines do not trim to the same point, and that is a design input.** At the
recovery condition atisim trims to alpha 4.3321 deg where JSBSim trims to 4.2786, a
+0.0535 deg gap. It is not an entry defect: fed JSBSim's own trim state, atisim returns
CL to 3e-6. JSBSim's `do_simple_trim` converges to **Nz = 0.99093, not 1.0**, and atisim
needs 0.95% more CL for a true 1 g -- worth +0.0627 deg of alpha against the +0.0535
observed. So the vortex comparison must be driven from a MATCHED STATE rather than from
each engine's own trim, the way the 737 sweep already is.


### Session 18 — the AtiSim rename, and correcting five stale claims

**The project is now AtiSim.** The import package is `atisim`, the distribution is `atisim`,
and the display name in the docs and generated reports is **AtiSim**. 1,510 occurrences
across 156 tracked files, plus `git mv flightsim atisim` and
`docs/summary/flightsim-summary.pdf` to `atisim-summary.pdf`.

**Two categories were deliberately left at the old name**, because renaming them would
falsify a record rather than update one:

- **`audit/` in full** — 67 files. The evidence `.md` files quote captured stdout, the logs
  *are* captured stdout, and `G-archaeology.md` names the real branch
  `claude/flightsim-sweep-ui-graphs-d4d036`, which still exists under that name. Decisively,
  `audit_evidence/E-falsification-scripts/common.py` hard-codes a `sys.path.insert` pointing
  at the **main checkout**, which still contains `flightsim/`. Renaming those imports would
  have broken scripts that currently work.
- **`FIX_PROMPT.md` and `analysis-ui-investigation-prompt.md`** — archived prompts, tracked
  as records by commit `78ca398`. What was asked is not editable after the fact.

**The rename turns this project's worst failure mode into a loud one.** §10's table records
that a wrong-tree import resolves silently to the main checkout with no error at all. The
editable install still maps `flightsim`, and nothing installs `atisim`, so on this branch
that same mistake now raises `ModuleNotFoundError`. Measured, not assumed. It reverts to the
old hazard the moment anyone runs `pip install -e .` under the new name — §10 carries both.

**Five stale claims corrected**, all of the same kind: a documented fact that a later commit
made false, with nothing failing to mark it.

| Where | Claimed | Actually |
|---|---|---|
| `aircraft.py` x2, `aero.py` x1 | α̇ is applied for the **wind only** and `Cmq` stays folded at −43.0 | `dynamics.derivatives` resolves the aircraft's own α̇ too (commit `cb23a9e`); the 737 carries a **bare** `Cmq` = −27.0 |
| fidelity spec, "Outcome" + "Explicitly not done" | item 4 (AERORP) and the α̇ solve are **not done** | both were done, in `fdbeff2` and `cb23a9e`; §4 and §5 of that same spec describe them |
| §4 ledger, session 17 | derivative table, layer 1 lateral, layer 3, layer 4 | written at `1dcc5d4`, before four commits changed all of them — superseded above, not deleted |
| layer-4 test docstring | doublet u 0.296 / w 0.410; Coriolis floor u 0.403, v 0.012 | measured u 0.507 / w 0.558; the reference's own diagnostics say u 0.409, v 0.028 |
| §10 | `pytest -q` runs **322 tests** | **626 passed, 1 skipped**, ~8 min |

`pyproject.toml` also gained the `ref = ["jsbsim"]` extra its own design spec's architecture
table has claimed since `f04bfac` and which was never added. `scripts/gen_jsbsim_reference.py`
is confirmed the only file importing `jsbsim`.

**Still missing, and not invented here:** session 17 has a §4 ledger entry but **no §9 session
log entry**, which this document's own update rules require. Writing one retroactively would
be fabricating a record of work this session did not do, so it is flagged instead.

**Deliberately not done:** none of §4's "still open" items — the phugoid's 6.58% against a
throttle-linear thrust model, `CDde`, Mach scheduling, banked trim, or a condition above
M 0.8 that would test `wave_drag` at all. This session changed no physics; the suite is
626 passed / 1 skipped before and after, which is what makes that claim checkable.

### Session 16 — remediation: fixing what the audit found, and not fixing the rest

Code and documents in commit `4e925c4`; the audit documents, the regression tests and
the regenerated PDF in the commit after it.

The audit produced 41 findings. This session acted on them under `FIX_PROMPT.md`, whose
standing rule is the audit's own: *a known, bounded, honestly-recorded flaw is a pass; an
unexamined assumption presented as fine is a failure.* It follows that **"fix" is not
always the right response to a finding**, and the most important thing this session did was
decide which were which.

**Ten repairs, each with its own measurement rather than "the tests pass".**

| # | Repair | Verification that it was surgical |
|---|---|---|
| 21 | `validation.lateral_modes` sorted **signed** time constants, so an unstable spiral came back as `roll_tau`. Now `sort(key=abs)`. | Cherokee returns roll 0.3595 s and spiral −51.59 s, the same two numbers in the right slots. The three stable-spiral aircraft are **bit-identical**, which is what `sort()` and `sort(key=abs)` must be when every value is positive |
| 40 | `V_MIN` floored `V` before `qbar`, so a stationary airframe made 1.4–171 N out of still air. `qbar` and Mach now use the true airspeed; the floor stays on β and the three rates. | Force and moment at V = 0 are **exactly** zero on all four aircraft; free fall reads `n_z = −0.0` with the aerodynamics live; and **81/81 sampled states at ‖v‖ ≥ 1 m/s are bit-identical**, which is the whole safety argument — `jnp.maximum(x, 1)` returns `x` exactly for `x ≥ 1` |
| 37 | `trim.is_physical` checked α and nothing else, endorsing up to 364/640 of a (V, h) sweep on throttle outside [0,1] or elevator past the stops. Now `is_physical(x, ac)`. | The pinned 747 root at 471.8 m/s — α = −0.57° on throttle 567 — is rejected; all four aircraft still pass at their own cruise conditions, which is the positive control `test_trim.py` already asserted |
| 30 | `along_track_shear` held the track direction fixed, dropping the heading-rotation term. Derived, with the reduction to Proctor Eq. (4) at ψ̇ = 0 shown in the docstring. | Reproduces the differentiated truth along a prescribed circular track to **0–5.6e-17**; ΔF = **0.1423** at a standard-rate turn, the FAA threshold entire. **ψ̇ was identically 0.0 at all 77,036 samples** of the runs on record and both scripts' printed output was byte-identical |
| 17 | `aircraft.py` cited Table IX-1's `CLδe` as .396; the table reads **.356**. | Printed p.216 re-read at 500 dpi in this session. The recovered 0.3638 agrees to **2.2%**, not the 8.1% the comment implied |
| 8 | `PROJECT.md` §4 compared 2 of the 4 published longitudinal factors. | All four now listed, from p.231 re-read at 600 dpi. The two that were missing are the two that look worse — and adding them **strengthens** the position, because §2.3's attribution closes all four to ≤1% |
| 11 | "A constant with no ledger entry fails the build" was false — nothing checked coverage. | Both documents now state what is enforced: a **new** module-level constant in one of five modules. Seven ledger entries added, `KNOWN_UNLEDGERED` shrunk to match, and a second test now fails if a ledgered name is left in it |
| 27 | `ASSUMPTIONS.md` E4 bounded the wind hold with the wrong instrument, by ~80×. | Re-measured hold-vs-per-stage at the same dt: **−1.62%** at dt 0.02, **−0.82%** at the published 0.01, **−0.41%** at 0.005. Halving with dt is what identifies it as O(h) |
| — | `integrate.py` justified the wind hold as "the standard treatment for Dryden and von Kármán". **No such source exists in this repository and none was found.** | Replaced with the actual argument — a stochastic field is drawn from a key, so per-stage sampling makes the realisation depend on dt — with no appeal to authority |
| 29 | `superpose()` with no arguments returned the integer `0`. | Returns the zero field, and superposing it is bit-identical to not superposing |

Also repaired: `test_conservation.py`'s comment was wrong by a decade (5.6958e-13, not
5.7e-14); `wind.py` had Oseguera & Bowles' four constants' dependency backwards, which the
paper's own wording settles — 0.22 is the TASS input and 12.5 was iterated *from* it; and
the Cherokee and Cessna source notes now say **`unverifiable — source not available`**
rather than citing a file that is not in the repository.

**One repair was made, measured, and reverted.** `_B747_G = 32.174` breaks `units.py`'s
no-inlined-factor rule and truncates g₀ by 1.5e-6. Going through `LB2KG / SLUG2KG` moved 19
quantities on the 747 and nothing on any other aircraft, all ≤ 4.1e-6, with **every number
§4 quotes unchanged at its quoted precision** — and broke two bit-exact
arithmetic-neutrality guards on earlier refactors. Re-pinning those would have spent the
guarantee they exist to provide on a violation that moves no result. **The flaw is smaller
than the fix.** Recorded as `ASSUMPTIONS.md` B5 and at the constant.

**Eight findings were real, measured, and deliberately left alone**, because the correct
form is not established by any source this project holds. Changing them would convert an
honest documented limitation into a fabricated certainty, which is strictly worse than the
flaw. They are now declared in the register: the **lift-tilt energy seam** (C9 — the fix is
exact to 1.9e-16 and would still be wrong to apply, because the tilt belongs to every lift
channel and the only available arm rests on an attribution this project already rejected);
the **strip quadrature at 9 stations** (F5 — 82.6%, convergence order measured at **1.50**,
and the decision to change neither `N_SPAN` nor `a₀` recorded in `provenance.py` so it is
not re-taken silently); the **microburst's unmodelled `z_h`** (E6); the **silent NaN on a
singular control Jacobian** (F7); the **one-sided longitudinal station set** (E8); the **two
tail arms that disagree by 2–2.9×** (C11 — and which of them is wrong cannot be determined,
because the split is exactly the sourcing split); the **vortex core branch tie** (E9); and
the **compound `δa`** (C10). Four more assumptions the code made and the register never
declared are now A4, C7, C8 and D2.

**Verification.** Pre-existing suite **444 passed, 1 skipped** — the baseline, unmoved.
`test_audit_regression.py` grew from 109 to 116: three fail-when-fixed tests were replaced
by the positive assertions their own messages named, and new tests pin the heading-rotation
derivation, its reduction to Eq. (4), the straightness of the shipped runs, the empty
superposition, and the widened trim gate. **No tolerance, reference value or assertion was
weakened anywhere** — the only assertions removed from the whole repository are the seven
that constituted those three tests.


### Session 15 — the order half of the wind seam, and the analysis UI

Two things, and the first came out of designing the second.

**The integrator is first order, not fourth, in a spatially varying wind field.**
Found while measuring for the UI design: the energy-closure residual converged at
order 1.00 and the explanation first written for it — trapezoidal quadrature
across the core kink — was wrong. The trajectory itself is first order. §4 has
the table; the cause is the once-per-step wind hold, which `integrate.py` has
documented as deliberate since session 2 and which `ASSUMPTIONS.md` §E4 said in
as many words the still-air order test could not see. Session 12 closed E4's
body-force half; this closes the order half.

`verification.fixed_control_refinement` now takes a `wind_model`, which is what
its own docstring always claimed it was for — it promised to catch "a wind sample
applied at the wrong RK4 stage" while having no way to pass a wind model.
**Falsified by injecting the fix**: re-sampling the wind at each stage restores
4.0542, so the 1.05 is attributable to the hold and to nothing else. Bounded and
harmless at the step sizes in use — 0.0024 m/s of gust error in the Parks core,
~1e-4 relative — so nothing §4 quotes moves.

**The analysis UI.** `checks.py`, `analysis/` and `apps/` (§2), fed by run
artifacts from `scripts/vortex.py --artifacts`. `integrate.logged_rollout` is the
one core addition: `rollout` emits `carry.state` and therefore discards the wind
a run actually flew, so no analysis script could write a self-describing run.
`vortex_viz.fly` now uses it and carries the log on the `Encounter`; a test pins
the headline pair to exact equality with the pre-change values, and it did not
move a bit.

**Three things the work found in itself, all recorded rather than repaired
quietly:**

- **A 14 r₀ lead-in starts the aircraft 0.100 g out of equilibrium**, and
  `trimmed_start` caught it in its own test fixture on first use — the same
  defect §9 session 3 records at −6 r₀. But the check as first written was also
  wrong: at `scripts/vortex.py`'s own **40 r₀ it is still 0.0384 g**, because the
  Parks far field is 1/r and never dies away. There is no trimmed start in a
  vortex, so demanding one would fail the project's canonical run and loosening
  the tolerance until it passed would be choosing a number to make a check
  succeed. It is now a **report** carrying the offset as a fraction of the run's
  own peak excursion (3.1% at 40 r₀ against 16% at −6 r₀), and a **gate** only in
  still air, where cos θ₀ is genuinely the right answer.
- **The α gate condemned the project's own published result.** Collapsing three
  declared bands into a boolean as `band == "linear"` failed the manoeuvring
  Fig. 8 point at |α| 10.31°, which is amber and which §4 reports. The gate now
  answers "does this run prove anything?" and condemns **INVALID** alone, with
  the band word carried separately for the UI to colour.
- **Two browser measurements of `uirevision` disagreed with each other**, because
  the synthetic drag fired a `plotly_relayout` and populated `_preGUI` without
  actually moving the camera — so the mechanism looked broken when the probe was.
  The app therefore does not rely on it: `figures.apply_camera` restores the
  camera explicitly from `relayoutData`, which is deterministic, unit-tested
  without a browser, and verified end-to-end.
- **The 3D panel drew a flight path through an empty field, and looked fine
  doing it.** The field grid was centred on the trajectory's midpoint, which for
  a 40 r₀ lead-in is ~2.5 km upstream of both cores — where the Parks field is
  **exactly irrotational** — so the vorticity isosurface was computed over a
  region of zero vorticity. `vortex_viz._field_panel` had already written the
  rule: *"Zoom to the structure, not the run."* The grid is now centred from the
  artifact's field spec; `isomin` reads 0.255 against 0.9·2V₀/r₀ = 0.255 and the
  isosurface peaks at the in-core 0.28333 s⁻¹. **Found by reading a number off
  the rendered page, not by a test** — which is the same way §9 session 7 found
  the one-sided α gauge.

Suite **434 passed, 1 skipped** in 339 s, up from 377. Nothing in §4's validated
baseline moved.

**Deliberately not done:** no comparison driver (Galilean, strip-vs-point and the
h/2 Richardson panel all need a second run and belong to a driver that writes its
own artifact); no multi-run sweep table; `leewave.py` and `microburst.py` do not
write artifacts yet — the schema and `rebuild_field` already cover their fields,
so it is the same wiring as `vortex.py` and nothing yet needs it.

### Session 13 — putting session 12's work where the notebook can see it

Session 12 closed E4 but left its two checks **inside the test file**, unlike every other
tier-0 check, which is a `verification.py` function driven by a thin test. The consequence
was not cosmetic. `ASSUMPTIONS.md` states the protocol — *add the computation to
`verification.py`, assert it in a test, then add a notebook cell* — and session 12 did only
the middle step, so **the notebook still told a reader the `−m·dW/dt` term could not be
detected.** The review deliverable was misstating the project's status.

Extracted `without_aerodynamics` and `free_fall_through_a_swinging_wind` into
`verification.py`, returning a `FreeFallResult` the test asserts on and the notebook prints.
The notebook is now 13 cells; its summary moves the seam from "known gaps" to "established"
and picks up the g(h) decision.

**Having the experiment report its own numbers immediately found a wrong one.** The comment
claimed `|W0|` = 30.5 m/s and peak `|dW/dt|` = 91 m/s² (9.3 g). It is
`√(18²+20²+12²)` = **29.4618 m/s** and **88.3855 m/s² (9.01 g)**. The slip had reached
§4, §9 and `ASSUMPTIONS.md` E4. All corrected, and the test now asserts both figures rather
than trusting a comment — which is the entire argument for computing a number where a test
can see it.

Two other numbers are now sharper. The free-fall agreement is **3.98e-12 m**, not merely
"under 1e-9". And the falsification is quantified: with the spurious term injected,
the same figure is **13.33 m**, ten orders of magnitude above the bound.

The experiment now runs on the **747** rather than `conftest.make_test_aircraft`, whose own
docstring says it is synthetic and "must never be used for results" — which a notebook is.
Free fall is independent of mass and airframe, so the choice cannot flatter the result.

322 tests + 1 skipped and the 13-cell notebook, both green. No test count change: the
existing test was rewritten as a driver, not duplicated.

### Session 12 — closing the two actionable assumptions, and what measuring changed

Session 11's register ended with two entries marked new and actionable: **E4**, the
untested `−m·dW/dt` gust seam, and **A2**, constant gravity. Both are now closed, and in
**both cases the measurement contradicted the reasoning that raised them**. That is the
theme worth carrying forward.

**E4 — the gust seam.** The plan proposed catching a spurious `−m·dW/dt` term by offsetting
the start state by `W(0)` and demanding the rates match still air. That asserts false
physics: the air-relative velocity obeys the still-air equation *plus* `−CᵀẆ`, so the two
runs must diverge, and no tolerance could have been chosen honestly. The instrument that
works is a **closed form** — zero the aerodynamics and the thrust and free fall is the exact
answer, while the wind has no legitimate route into the equations at all, so any dependence
on it is the spurious term. Matches `p₀ + v₀t + ½gt²` to 3.98e-12 m through a wind swinging
to 29.46 m/s at peak `|dW/dt|` = 88.39 m/s². A second test keeps the full 747 aero and
varies only the cached previous wind: bit-identical. Both were falsified by injecting the
bug; the Galilean test **passes with the bug still in**, so its documented blindness is now
measured. (Session 12 recorded 91 m/s² here from an arithmetic slip; session 13 corrected
it by having the experiment report the figure rather than a comment assert it.)

**A2 — constant gravity.** Session 11 reasoned from Lanchester that a 0.383% gravity error
threatened every agreement below 0.5%. Measured, the 1:1 mapping is the **phugoid's alone**
(−0.3798%, confirming Lanchester to three figures); the short period is immune (+0.0002%)
and the lateral modes — where §4's tightest agreements actually are — move 0.055–0.079%,
five to fifteen times smaller than feared. Their sensitivity is *indirect*, through a trim α
that falls 0.65%. Worst tolerance consumption is 7.6%, so **`g(h)` is not modelled** and the
entry closes on the bound. Sea level moves exactly 0.0000%, which is the control.

**Two wrong numbers found in §5 while writing a test.** The absurd-trim example attributed
−633° to `CLa = 0.1`; it is `CLa = 1e-4`, and 0.1 gives −272.7°. "Every real aircraft trims
at 5–6°" is wrong the other way — the registry spans 0.01° to 5.62°. Worse, the angle is
**not reproducible at all**: same aircraft and CLa, 85.0 m/s gives −632.1° and 84.9 m/s
gives −4232.1°. Nothing asserts an angle now, only converged-and-absurd.

`trim.is_physical` moved the |α| bound out of `validation.sweep`, where session 11 put it,
into the module whose function actually has the defect. **`WindState` did not need a time
field** — `step` threads the wind state opaquely, so a model brings its own type, and adding
`t` would have broken `test_integrate.py`'s `FilterState`. §7 records that Dryden therefore
costs no signature change; the blocker is `init_sim` seeding, not `WindState` carrying.

**322 tests + 1 skipped and the 12-cell notebook, both green.** Four new tests: two on the
gust seam, two on `is_physical` (one of which is the positive control, without which
`is_physical` could simply return `False` always). Nothing in the validated baseline moved.
Note the preceding commit's message says 320 — that count was taken from a background run
that predated the two trim tests in the same commit; the correct figure there is 322.

### Session 11 — verifying the solver, and correcting what §5 claimed

External review asked for the solver's validity to be established **before** more layers,
and for an interface where a known coefficient change gives a known result. The project
could not meet that: §4 was almost entirely validation rows, with next to nothing saying
the arithmetic is right independent of any aircraft's data.

Review also asked whether a 1972 source is itself a source of error. The answer is that it
is not, for what is actually being checked — if CR-2144's derivatives were 10% from the
real aeroplane, this model must **still** reproduce CR-2144's own transfer-function factors
from CR-2144's own derivatives. The real risks are different and are now recorded: the data
is the **flexible** airframe against a rigid-body model (§5, new), the scan, and the
small-perturbation range. The design spec ranks every check by how much it depends on any
source at all.

**Three things were found, and none of them was the failure being looked for.**

1. **`aero.py`'s exclusions were misdescribed, and had been since session 1.** §5 said
   `Xu, Zu, Mu, Żw, Ṁw` are excluded. The model **has** Xu and Zu — they fall out of
   dynamic-pressure variation, and A[0,0] lands within 1.2% of Caughey's Xu with no Xu
   entered anywhere. What is missing is their **Mach content**, which is why the offsets
   are **condition-dependent**: 17.8% at M 0.80 and **0.4%** at M 0.25, same code.
2. **The 6-DOF order-of-accuracy fit read 3.82, and the measurement was at fault.** At
   40,000 ft `pos_ned` carries a 12,184 m altitude that float64 resolves to 2.7e-12 m, so
   the error floors near 7e-11 m and past dt = 1/128 refining makes it *worse* — pairwise
   order **−0.685** at 1/256. Fitted in the asymptotic range it is **3.98913**.
3. **`trim.trim` converges to absurd roots.** CLa = 0.1 gives α = −633° at a residual of
   1.6e-15. A residual check detects non-convergence and cannot detect nonsense.

**Three of the four coefficient sweeps were the wrong functional form, and the model was
right each time.** Every relation turned out affine with a non-zero intercept, and the
intercept is the term the textbook approximation drops — `Zα·Mq/u₀` for the short period,
`Yβ/u₀` for Dutch roll, Ixz coupling for roll. A test asserting ωn_sp → 0 at the neutral
point was written first; that is wrong physics. What *is* exact is the neutral point
itself: largest real root **0.00000** at Cmα = 0, and nothing was tuned to put it there.

**What the tier-2 comparison bought.** Caughey's Cornell notes work CR-2144's own approach
data and publish every intermediate, so this is an independent *implementation* rather than
an independent dataset — same inputs, different code. It needed a stability-axis rotation
(he states Θ₀ = 0, true only there) which is asserted to be a similarity transform first.
After it, every element the model contains matches, and the two that do not are
**reconstructed** from his own α̇ derivatives to 0.03%. §5's attribution stops being an
attribution.

**Deliberately not done:** no new aircraft, and no further 747 flight conditions — §7 says
why, and records that CR-2144 holds nine more airframes including the Jetstar and C-5A.
The notebook is a thin front end over tested code; it holds no arithmetic, so nothing it
displays can drift from the suite.

318 tests, was 296. Nothing in the validated baseline moved.

### Session 10 — the 747 power-approach set, and what it was hiding

§5 said the 747 could not be flown into a microburst because its only derivative set was
Mach 0.8 at 40,000 ft, and that adding an approach set would be "data entry rather than
modelling". Both halves turned out to be true, and the data was closer to hand than that
implies: **`aircraft.py`'s own header had said since session 1 that CR-2144 tabulates
non-dimensional derivatives for the landing and power-approach configurations.** It was
written down as a *reason the cruise set needed converting* and never read as an
opportunity. Table IX-2 is a complete, already-non-dimensional set — no conversion chain,
no primed-to-unprimed lateral algebra, no chart read.

**Two source conflicts came out of cross-checking it, and one nearly shipped.**

- **Figure IX-1 and Table IX-3 disagree on the approach inertias by up to 6%.** The figure
  says 13.7/30.5/43.1/0.825 ×10⁶ slug-ft²; Table IX-3's flight condition 2 says
  14.2/32.3/45.4/0.870. Figure IX-1 was transcribed first and would have gone in unnoticed.
  Table IX-3 is used, because it is the table the derivatives were *computed* at — its
  Q = 92.2 psf, VTO = 165 KTAS and ALPHA = 5.70° all match Table IX-2's header exactly.
  For **cruise** the two agree exactly, so this is specific to the approach configuration.
- The weights differ in the last two digits the same way (564,000 vs 564,032; 636,600 vs
  636,636). This let the **existing cruise weight be checked**: 636,636 is Table IX-3's
  value and is correct, not the transcription slip it briefly looked like.

**A finding that is physics rather than bookkeeping:** the approach point sits **12.2 m/s
below minimum-drag speed**. `test_cruise_is_above_the_minimum_drag_speed` had asserted the
opposite for every registry entry — correctly, while every entry was a cruise point. Rather
than weaken it, it now skips the approach entry and a second test asserts the *inverse*
with the reason: 1.4 Vs at max landing weight is the back side of the drag curve, which is
where an airliner on final actually is, and is why windshear is lethal on approach and
merely uncomfortable at cruise. The autopilot's loop pairing is inverted there; all
analysis flies it open loop.

**What it bought.** The microburst re-flown as a jet transport: 1 km average F = **+0.2835**
against **+0.2094** of thrust authority — beaten by 1.4×, **2.8× the FAA hazard threshold
that now legitimately applies**, and **inside** the 0.2–0.36 band the paper reports for
real accidents. The Cherokee's answer could only ever be measured against its own thrust,
because Proctor et al. say the FAA scale was never established for piston aircraft;
`scripts/microburst.py` now decides that per aircraft instead of disclaiming it globally.

The jet has **2.7× the light aircraft's thrust authority and is still beaten**. More engine
does not buy immunity, it buys a smaller multiple.

**Deliberately not done:** the landing configuration (Table IX-1, 131 KTAS, 30° flaps,
gear down) is not added — it is the same job again and nothing yet needs it. The approach
gains are re-scaled from cruise by dynamic pressure and pass the engage-and-hold test, but
they are **not hand-tuned** and §9 session 6's point stands: that needs a human flying it.
The summary PDF is unchanged.

296 tests.

### Session 9 — the microburst, and the metric session 8 got wrong

Chosen as the next field because of what it would *test*, not because it was next on a
list. The lee wave is purely vertical, so `U̇ₓ` was identically zero and **half of the
F-factor had never been exercised**. A microburst has a horizontal outflow, and in the
flown result the shear term (+0.134) comes out the same size as the vertical one (+0.146).

The field is Oseguera & Bowles 1988 (NASA TM-100632), the standard analytic microburst —
and Bowles also wrote the F-factor, so the field and the index it is measured with come
from the same group. **The 1988 scan OCRs badly**, so the equations were reconstructed and
then checked against four constants the paper states independently: peak outflow at
r/R = 1.1212 solves `exp(−x²)(2x²+1) = 1`; z_m/z* = 0.22 is `ln(12.5)/11.5 = 0.2196`;
u_max = 0.2357λR is the product of those two; and the paper's `w_max = λz*(e^(−z_h/z*) −
0.92)` is the vertical equation with `ε = z*/12.5` substituted, which is where 0.92 comes
from — `1 − 1/12.5`. Four different consequences of the same two shaping functions, so a
mis-transcription could not have satisfied all of them. That is what makes the
reconstruction trustworthy rather than merely plausible.

**Session 8 measured the wrong quantity, and this session's source says so explicitly.**
The F-factor's hazard metric is the **1 km average** (Proctor et al. Eq. 7), not the
instantaneous value: peaks "over small length scales... are quickly followed by negative
values", felt as turbulence rather than as a loss of flight path. `average_f_factor` now
implements it, and a test asserts that a 100 m spike of F = 0.5 averages to 0.05. The lee
wave was re-reported with it and **barely moved** — +0.02621 to +0.02614, because a 25 km
wave and a 1 km window is `sin(x)/x` at 99.7% — so session 8's conclusion stands. It was
still the wrong quantity, and on this session's field the gap is 21%.

**Two defects found in this session's own work, both by auditing rather than by tests:**

- The microburst's outflow was written as the paper writes it, `(λR²/2r)[1 − e^(−(r/R)²)]`,
  which is 0/0 on the axis. Guarding the radius made the **value** right and the
  **gradient** wrong — and `field_model` differentiates the field to get `omega_gust`, so
  anything flying through the core would have been handed a silently wrong rotational
  gust. Fixed by factoring the direction cosine back in, leaving a function of r² with a
  removable singularity. Caught only because the continuity test evaluates *on the axis*.
- With fixed controls the aeroplane descends, reaches 6 m, and **climbs away again**.
  There is no terrain, no gear and no ground effect in this model, so that is arithmetic,
  not a survival. Runs now stop at one wingspan.

**The result.** Cherokee at 300 m, 1 km average F = **+0.1929** against **+0.0784** of
thrust authority — exceeded 2.5×, 1.9× the FAA jet-transport hazard threshold, and just
below the 0.2–0.36 band the paper reports for real accidents. Ground contact 95.5 s in,
383 m past the axis; it never reaches the far side. Scaling linearly, the Cherokee's
authority is first beaten at 7.73 m/s of peak outflow — **below the 10 m/s of divergence
Wilson et al. require before an outflow is called a microburst at all.** Where the lee
wave's threshold fell *inside* the observed range, this one falls below the bottom of it.

**The aircraft choice is a finding, not a convenience.** The 747 could not be flown here:
its only derivative set is Mach 0.8 at 40,000 ft and a microburst is met below 500 m at
approach speed. §5 records that, and that the FAA thresholds are jet-transport-only so the
verdict is always the aircraft's own `(T−D)/W`.

**Deliberately not done:** no approach-configuration 747 (CR-2144 has other flight
conditions; adding one is data entry, and it would let this be re-flown on the aircraft
class the thresholds were written for). No ground model. The summary PDF is unchanged.

284 tests.

### Session 8 — the mountain lee wave, and what a 747 can do about it

§7 step 8, and it went the way session 3 went: **retrieve the source, then write the
model**. Two were needed and they do different jobs — Doyle et al. 2011 for the *field*,
Proctor/Hinton/Bowles 2000 for the *index*. Both are now in §3.

Doyle et al. was chosen over any textbook lee-wave treatment for one reason: **altitude**.
Its Gulfstream V flew legs at 11.3 and 13.1 km over the Sierra Nevada during T-REX, and
this project's 747 cruises at 12.192 km, between them. Everything else in `wind.py` is
DC-10-class data near the tropopause, so the whole module stays altitude-comparable rather
than mixing a low-level wave model into high-altitude work.

**The result is sharper than the step asked for.** §7 wanted "F exceeds the measured
+0.023/−0.066 envelope". It does — but not for both of the *same paper's two flight legs*,
flown by one aircraft on one day 50 km apart. The critical amplitude is **w₀ = 5.51 m/s**,
and Doyle's legs are 3.0 and 6.0. So the honest statement is not "a lee wave defeats a
747" but **the hazard threshold sits inside the observed range** — which is the more
useful claim and was not knowable until the envelope and the field were in one place.

**A defect found in this session's own work, by auditing it rather than by a test.** The
first version of `leewave.py` started the run on a wave **crest** while its own comment
claimed it opened "in undisturbed-mean air": 6 m/s of updraft, about 1.5° of α out of trim
before the first sample. That is exactly the §9 session-3 vortex lead-in defect wearing a
different hat, and a periodic field makes it easy to walk into because there *is* no
undisturbed region to lead in through — the nearest equivalent is a zero crossing, which
is what it now uses. It moved peak F by 8e-5 and changed nothing, which is precisely why
it would have survived: the number it produced was not wrong enough to look wrong.

The envelope itself was **recomputed rather than trusted**: session 2 recorded
+0.023/−0.066 with no derivation on the record, and `dynamics.thrust_authority` now
reproduces +0.0234/−0.0657 from the trim solution, with a test. In trimmed level flight
T = D, so the drag *is* the trim thrust and the envelope is just how far the throttle can
travel either way over the weight — which is why it needs a trim solve and not a drag
model.

Two things worth keeping:

- **Flying it matters.** Peak F comes out at 0.02623 against a naive `w₀/V` of 0.02543,
  because the aircraft *slows* in the downdraft and F goes as 1/Vₐ. The encounter makes
  itself slightly worse, and only integrating it shows that.
- **The FAA's 0.1 threshold is deliberately not used.** Proctor et al. §4.1 bounds the
  windshear threat to below 500 m, since higher up an aircraft has potential energy to
  trade. What transfers to 12 km is the *index* and the paper's own `F > (T−D)/W`
  criterion, not a number calibrated for approach. Quoting 0.1 here would have been the
  easy mistake and it is flagged in the code.

**The honest gap, in §5:** the field is purely vertical, so `U̇ₓ/g` is exactly zero and
only half of Eq. (3) is exercised. A real lee wave has a horizontal perturbation in
quadrature with the vertical one, needing a stratification and an ambient wind speed
Doyle et al. does not give. The omitted term peaks where the vertical one vanishes, so the
peak *location* would move rather than the peak simply doubling — but an order-of-magnitude
estimate puts it **larger** than the vertical term, so **the measured F is a lower bound
and is quoted as one**. The wavelength is declared for the same reason: the paper's
20–35 km is tropospheric and it says stratospheric wavelengths are shorter without saying
how much.

**The summary PDF gained a page**, "A second result: when the engines are not enough",
drawing the thrust-authority band to scale with both legs' F on it. Writing it turned up
a second class of defect, and this one had been there for sessions: **four of the five
page cross-references in the PDF were wrong.** "The reason is on page 7" pointed at the
panel page, "the cluster diagram discussed on page 6" pointed at the vortex page, and so
on. They were typed-in prose numbers in a *generated* document, and inserting the lee-wave
page shifted one of them from off-by-one to off-by-two.

Fixed structurally rather than by retyping: `PAGE_ORDER` names the pages, `pageno()` looks
them up, every reference is now an f-string, and `check_pagination()` aborts the build if
the emitted count and the declared order disagree. Verified by deleting a name and
watching it fail. This is the same lesson as session 7's stale test counts, one level up:
**a generated document is only as undriftable as the fraction of it that is generated**,
and prose numbers referring to the document's own structure are the most fragile kind
because nothing renders an error.

**Deliberately not done:** the lee wave is not on the Fig. 8 discriminator. It would cost
almost nothing — it is a deterministic field, so `vortex_viz.fly` takes it directly — but
Fig. 8 is a *pitch-and-load* clustering chart and the lee-wave result is an *energy* one,
so putting it there would imply a comparison the paper does not make.

270 tests.

### Session 7 — frame rate, the summary PDF, and the third Fig. 8 cluster

Two pieces of work landed *after* session 6's §9 entry was written, so the record was
self-contradictory when this session opened: §8 asked whether the panel still held 20 fps,
and the answer had already been measured. It had not. **13.7 fps on Agg**, against a 20 fps
target the code sets and no test asserts. `sense` and `accelerometers` were running eagerly
once per frame — 6.56 ms and 10.70 ms of a 73.0 ms frame. Jitting both took the frame to
37.8 ms, **26.4 fps**. Full table in §4; §10's 19.9 fps is superseded, not deleted.

The remaining cost is blitting, 30.1 ms of 37.8, and it did **not** improve — which is
what should happen when you jit something that was never the bottleneck's neighbour. It is
recorded rather than fixed: the panel is now comfortably above its target.

`scripts/summary.py` generates the plain-English summary PDF rather than it being written
by hand, so it cannot drift from the code — page 6's vortex figures call `wind.vortex_wind`
and the aircraft table reads `CRUISE`.

**The α gauge was one-sided**, and it was found by asking what the manoeuvring case would
*display* rather than by any test failing. §6(e) has the detail. It is (a) and (b)'s shape
a third time — correct in the easy case — and it would have lied during precisely the run
built next.

**Two open questions were closed before any code was written**, which is the part of this
session worth copying. Both had sat in §8 as "not resolvable from the paper's text", and
both still are; what changed is that they became decidable on *the model's own evidence*:

- The Fig. 8 load band's two readings turn out to land on **opposite sides of this model's
  validity boundary** — 10.31° of |α| as an increment, ≈18.5° as an absolute load. That
  makes the choice forced rather than arbitrary. §5 now carries the unreachability of the
  absolute reading next to the ±g asymmetry, which it shares a cause with: no stall means
  no way to reach a big negative load except a big negative α.
- The analysis window is **the disturbance's own extent** — which is what the vortex and
  updraft points were already doing, unstated. Writing it down was the whole difficulty:
  the third point had no rule to follow. The measurement that forced it is that Δn and
  |α| both **saturate 4 s into a held elevator** and never move again, while Δθ grows at
  3.6°/s for as long as it is held. A "manoeuvre" measured over 12 s reads 43° because by
  then it is a descent.

**The third cluster is in**, and `vortex_viz.fly` could not do it: it flies with fixed
controls by design, and a manoeuvre needs a time-varying elevator. `manoeuvre` is a
sibling rather than a generalisation of `fly` — a schedule that happens to be constant is
a strictly larger surface than a constant — and both now go through `_measure`, so the
three Fig. 8 coordinates are computed by one piece of code rather than two that could
drift. The elevator angle is **bisected** to reach the band, not chosen, so the sourced
quantity is the load. Result: 2.24 < 4.37 < 30.37 deg against the paper's 1.4 < 6.2 <
12.0. **The ordering holds; the values do not agree and §5 forbids claiming they should.**

Found while building it, and it would have been an invisible 0.08 g error: `fig8_point`
measures the load excursion from `enc.n_z[0]`, the run's own first sample. For a vortex
that is the trimmed value because the run has a long lead-in. Step the elevator at t=0 and
it is not — the first sample is already loaded. `manoeuvre` therefore has a lead-in at
trim for the same reason the vortex run has 40 core radii, and §4 records `n_z[0] = 0.9967`
as the check that it worked.

The figure's "manoeuvring: NOT MODELLED" annotation is gone because the slot is filled,
and the discriminator panel's x-axis is now scaled to the data rather than to the paper's
range — cropping to the reference would have hidden the 2.5× disagreement instead of
showing it.

**The summary PDF's claim about itself was too strong.** §10 said it "cannot drift from the
code". Its *computed* parts cannot; its prose and summary statistics are literals, and the
test and line counts had already drifted (256/4,300 against an actual 260/4,600). Corrected
by hand and the claim in §10 narrowed to what is actually true. A generated document is
only as undriftable as the fraction of it that is generated.

**The summary PDF now carries all three categories**, and building that page turned up a
number that had been wrong since session 6. Its timing chart said the two weather events
differ **seventeen-fold** in duration, and so did §4. **17× is Morton's ratio**, not
Hannibal's: r₀ = 137.16 m gives a 1.163 s traverse and 20/1.163 = 17.2, while every run,
figure and ledger row in this project uses Hannibal, whose r₀ = 182.88 m gives 1.550 s and
**12.9**. Corrected in both, superseded rather than deleted in §4. Nothing downstream
moves — the discriminator is an ordering claim and both ratios are the same order — but
§8's new window table states these durations to four figures, and a reader would have
found them contradicting §4's own sentence.

Writing that page also forced a sharper statement of the mechanism than §4 had. **Timing is
not "the whole discriminator" once the third category exists.** The pushdown's pulse is
6.609 s, *between* the vortex's 1.550 and the updraft's 20.0, yet it lands furthest right
of the three. So duration does not order the categories. It separates the two **weather**
ones; the manoeuvre separates because the elevator is moving and pitch follows the stick
rather than the air — which is the distinction Wingrove & Bach's chart was drawn to make,
and the reason `fly` holds its controls fixed.

**Deliberately not done:** `_trace_stack` still hard-codes the elevator
trace to ±5°, which would clip if the manoeuvre were ever made the figure's *primary*
encounter; it is not — `scripts/vortex.py` keeps the vortex primary — so this is flagged,
not fixed. Dryden (§7 step 4) is untouched and still blocked on digitising MIL-F-8785C
Fig. 7. `ManualGains` still not re-tuned. The interactive TkAgg frame rate still not
re-taken.

`test_viz.py::test_derived_agrees_with_the_aero_module` is **deleted**, after being flagged
in two session logs. Every assertion in it re-derived the implementation, so none could go
red; the replacement in `test_sensors.py` computes its expectation from the recorded wind
independently. What went with it: nothing else asserts `viz.derived`'s field wiring or its
altitude sign convention. That is a real if small loss, recorded here rather than left to
be discovered.

260 tests.

### Session 6 — the free-air flying interface
`run_live` took no wind model at all. `LiveSim.advance` called `step(sim, controls, dt,
ac)` and got the `zero_wind` default, so the Parks array and the Wingrove updraft — the
only things this project is building toward — could not be hand-flown. §10's "still air
only" note read as the session-5 sensing bug; this was a separate gap and it was still
open. It is now threaded through, with `--wind {none,hannibal,morton,updraft}` on
`fly.py`.

**The verification that mattered** was the same shape as session 5's: the guard test was
re-run with the wind model accepted but not applied, and the blown and still-air runs came
out at *identical* altitude while exactly that one test went red. An accepted-and-ignored
parameter is precisely how (a) and (b) survived three sessions.

The panel was re-laid-out as a basic T with a flight-test overlay. The 3D trace is gone;
it was the largest cell and told a pilot the least. New: VSI, slip ball, α against the
declared §7 band, load factor with a peak hold, wind, gust rate. Two decisions worth
keeping:

- **The slip ball reads lateral specific force, not β.** A ball is a pendulum. The two
  agree in steady coordinated flight and part company everywhere interesting — under held
  rudder they come out with *opposite signs*, which is what the test asserts. Wiring β into
  a ball would have been the same shape of mistake as §6(a): right in the easy case.
- **`omega_gust` is labelled SIM TRUTH.** It is a gradient across the span and chord and no
  instrument can sense it. The translational wind is not labelled, because with no sensor
  noise ground velocity minus air velocity *is* the wind and a real aircraft could compute
  it.

`dynamics.load_factor` was factored into `specific_force` returning all three components;
its two existing tests pass unchanged, which is the regression guard. `AirData` gained
`vertical_speed`; the accelerometer package is a *separate* function because a specific
force needs a mass and a set of deflections and an air-data computer has neither.

The stick now ramps, and there is pitch trim plus a trim-here key. The three `trim_rate`
values are derived from one stated rule — one second of trim is a quarter of full stick —
and a test asserts the rule, so the next person cannot quietly pick a fourth number.

`viz.py` was split: `panel.py` takes the live cockpit, `viz.py` keeps the log and the
post-flight figure. The split was landed as a **pure move** in its own commit, verified by
the collected test count not changing, so the re-layout's diff is only the re-layout.

Two things found in this session's own work, both by rendering the panel rather than by a
test: four layout defects (help text off the edge, strip labels over the overlay gauges, a
VSI drawn as a diagonal, a third of the figure empty), and a first trim test that asserted
the wrong thing — it trimmed *after* the stick had centred, where trim-here is correctly a
no-op. The test was rewritten, not the code.

**Deliberately not done:** `ManualGains` not re-tuned. The ramp makes higher authorities
available for the first time — they were geared down because a keyboard snapped to full
travel — but re-tuning is hand work verifiable only by flying, and `test_manual.py`'s
response bounds (±2°/±45°) are far too loose to pin it. No Mach, no control-position
display, no re-arm key, no uniform-wind option: considered and not chosen. `test_viz.py`'s
`test_derived_agrees_with_the_aero_module` is left alone — it is the
structurally-cannot-fail test §6(b) says was replaced, and the replacement did land in
`test_sensors.py`, but the original was never deleted. Flagged, not this work's mess.

**Unchanged, and still the Dryden blocker:** `init_sim`/`batch_sim` still hard-code
`zero_wind_state()`. Threading a wind *model* through the live loop does not touch that,
so §7's three structural changes stand exactly as written.

256 tests.

### Session 5 — the four latent bugs, and the extensibility seams
Fixed (a)-(d). (a) and (b) were one root cause, so they got one fix: `sensors.AirData`
and `sense(state, wind_ned)`, with `autopilot`/`engage`/`manual.update`/`viz.derived` now
taking the sensor set instead of `State`. That is deliberately the invasive option — with
no `vel_body` in scope there is nothing left to misuse. Which quantities are air-relative
is physics and is documented in `sensors.py`: pitot and vanes yes, IMU and rate gyros no.
Feeding `omega - omega_gust` to a rate-damping loop is the overcorrection, and there is a
test pinning against it.

(c) was fixed by recording rather than the replay §6 originally suggested: `SimState`
carries the wind the previous step applied, `Recorder` takes the whole `SimState`, and
`load` defaults the new columns so old `.npz` still open. Replay was cheaper but needs the
caller to reconstruct the model and key correctly every time.

(d) was one character. Short-period ζ error 12.6% → 11.5%; the affected ledger rows are
superseded in §4, not deleted.

**The verification that mattered:** re-introduced (a) and (b) and confirmed exactly their
two tests went red while the other 219 stayed green. §7 asked for "test_viz's derived test
can actually go red" and the honest answer was that it never could — it fed both sides the
same input. It is replaced rather than repaired.

Also added `FlightCondition` and the `from_dimensional_*` helpers, and retrofitted the
Cherokee and Cessna onto them. They exist because the Cessna bug was a wrong dynamic
pressure, and bundling the condition with the conversion leaves no argument to get wrong.

Deliberately not done: nonlinear aero. Asked and declined, so the ±g asymmetry is now
permanently out of reach rather than pending — recorded in §7 so it is not rediscovered.
Data-file aircraft definitions also deferred, with the condition for revisiting written
down.

221 tests.

### Session 4 — usage record
No code changed. Added §10 because nothing in this document said how to *run* any of it:
the architecture table names modules, not entry points, and the four scripts had their
usage only in their own docstrings. Recorded every script, flag, key binding and library
entry point, and which of them are trustworthy under wind (most are not — §6(a) and (b)
mean the live panel and the autopilot both mis-sense under a wind field, so `scripts/
vortex.py` and `vortex_viz.py` are the only air-relative analysis path).

Corrected a stale count in §8: 187 → 209 collected tests.

Deliberately not done: the Cessna was left exactly as it is (§5 keeps it out of scope) and
no attempt was made to reconcile the design spec, which stays a historical document per §1.

### Session 3 — vortex model, updraft, analysis figure
Retrieved Parks 1985 and used it to replace every assumed part of the vortex model with a
cited one: Rankine profile, superposed arrays, and a spacing that Parks itself checks
against Scorer. Added the updraft column with edge sharpness as a declared parameter.
Added `dynamics.load_factor`, `vortex_viz.py` and `scripts/vortex.py`.

Two defects found in the previous session's own work:
- The committed vortex test used a −6·r₀ lead-in, which starts the aircraft 0.20 g out of
  equilibrium because the 1/r far field has not died away. First-core Δθ was understated
  by 15% (1.89 vs a converged 2.20 deg). The initial load factor is now asserted.
- `n_z` in trimmed level flight is `cos θ = 0.9967`, not 1.0 — the body-normal
  accelerometer reads `g·cos θ`. The first version of the test asserted the wrong physics.

### Session 2 — turbulence design
Design pass over turbulence options and analysis methods. Established that a zero-strength
wind model is bit-identical to still air, that the rotational gust from a Wingrove-scale
vortex exceeds the 747's full aileron authority by ~1.5×, and that the mountain-wave
thrust-authority result (+0.023/−0.066 against F-factor ±0.2) is already true from
existing data with no wind model at all. Surfaced latent bugs (a)–(c).

### Session 1 — validation pass
Converted ad hoc checkpoint prints into 18 asserted tolerance-bound tests. Found no
lateral bug — the suspected Dutch-roll damping gap did not exist. Established that a
previously reported 103 s spiral was a linearisation artifact (a dropped
`r·cosφ·tanθ₀` term), confirmed by a nonlinear decay fit giving 138.05 s. Attributed the
phugoid/short-period gap to the excluded speed and α̇ derivatives.

### Template for a new entry
```
### Session N — one-line theme
What changed and why. Numbers measured, with what they were compared against.
Anything found to be wrong in earlier work, stated plainly.
What was deliberately not done.
```

---

## 10. Running it

Python 3.10.11, `.venv` in the project root. All commands are run **from the project
root**; the scripts import `atisim` from the editable install, not from `scripts/`.

### Before you trust a run, check which tree it imported

This is the one failure mode here that produces **no error message at all.** Worktrees do
not get their own `.venv`; they share the main checkout's, and that editable install's
finder maps `atisim` to the **main checkout's** `atisim/` permanently. The finder is
*appended* to `sys.meta_path`, so it is reached only once `sys.path` has already failed —
and whether `sys.path` succeeds depends on how the process was started. Measured from a
worktree, all four rows:

| how it is run | what `import atisim` resolves to |
|---|---|
| `.venv/Scripts/python.exe -m pytest`, cwd = **worktree root** | **the worktree.** `-m` puts cwd on `sys.path` first |
| `pytest` / `pytest.exe`, cwd = worktree root | **the main checkout.** The console script does not put cwd on `sys.path`, and `atisim/tests/conftest.py` imports `atisim` before pytest's own insertion helps. `sys.path[0]` *is* the worktree by the time a test body runs, which is why this one looks fine and is not |
| anything, cwd = **any other directory** — `notebooks/`, `scripts/` | **the main checkout** |
| any of the above with `PYTHONPATH` set to the **absolute** worktree root | **the worktree** |

So the documented `.venv/Scripts/python.exe -m pytest -q` is safe from the worktree root,
and **nothing else in that table is.**

> **The `atisim` rename changes this, and for the better — read this before trusting the
> table above.** The table describes the package when it was called `flightsim`. The editable
> install still maps **`flightsim`** to the main checkout, and nothing installs `atisim`, so on
> this branch every row that used to resolve silently to the main checkout now raises
> `ModuleNotFoundError: No module named 'atisim'` instead. Measured, from `C:/Users/mateusz`:
> `import atisim` raises `ModuleNotFoundError`; `import flightsim` still returns
> `.../Claude_Flight_Sim/flightsim/__init__.py`.
>
> **The failure mode that produced no error message now produces one.** That holds only until
> someone runs `pip install -e .` from a tree carrying the new name, which re-creates exactly
> the old hazard under the new spelling — at which point this table applies again verbatim.
> The one-line check below is still the thing to run, and is now spelled `atisim`. The check costs one line, run from the directory you
are about to run the suite from:

```
.venv/Scripts/python.exe -c "import atisim; print(atisim.__file__)"
```

If that path is not the tree you edited, everything downstream is about someone else's
code: a passing suite, a green notebook gate, a sanity ladder that agrees with itself, and
a measurement that lands in §4 under false provenance. A change that is *absent* from the
tree under test fails in the safest possible way — the old behaviour is asserted and
passes — which is precisely why it survives review.

### The entry points

The count is deliberately out of this heading: it read "five" over an eleven-row table for
several sessions, which is the drift §4's rules exist to prevent.

| Command | What it does |
|---|---|
| `.venv/Scripts/python.exe -m pytest -q` | **942 passed, 1 skipped, 1 xfailed, 0 failed, 1,100 s** on Windows (release 1.1, on the tree `atisim.__file__` confirmed: the α̇ declaration re-pinned nine figures across six files, each with the argument at the change, and added three tests). Before that **935 passed, 1 skipped, 1 xfailed, 0 failed, 1,118 s** on Windows (later in phase 5, on the tree `atisim.__file__` confirmed: the 931 below plus the 4 in `test_sanity.py`. Not yet run in CI). Before that **931 passed, 1 skipped, 1 xfailed, 0 failed, 1,327 s** on Windows (later in phase 5, on the tree `atisim.__file__` confirmed: the 926 below, plus pull request #14's footer test in `test_vortex_viz.py`, plus the 4 TM-102186 Fig. 8 fleet tests in `test_cat_validation.py`, whose shared fixture takes 50–64 s. 933 collected. Not yet run in CI). Before that **926 passed, 1 skipped, 1 xfailed, 0 failed, 1,672 s on Linux in CI** (`.github/workflows/tests.yml`, run 35390817171 on phase 5's final commit, and 1,566 s on its first, run 35370419455: the same count as Windows, at the same library versions — §4, "What does not reproduce on another platform"). On Windows, **926 passed, 1 skipped, 1 xfailed, 0 failed, 1,561 s** (end of session 32's phase 3, on the tree `atisim.__file__` confirmed: the 918 below plus 8 of the 9 tests in `test_cr2144_crosscheck.py`. **The one xfail is deliberate and strict** — `Cm_M`'s band in that file, where a sealed prediction was measured WRONG and the band was marked rather than widened, §4. 928 collected). Before that **918 passed, 1 skipped, 0 failed, 1,092 s** (end of session 32, measured on the merged triage tree: the 906 below, plus 12 in `test_figures.py` from the panel-chrome fix applied across the rename. 919 collected. **Zero failures** — the 2 platform bit-pins that failed through session 29 are green here). Before that **906 passed, 1 skipped, 1,534 s** (end of session 30, after merging `main`'s session 29: 863 plus its 43). Before that **863 passed, 1 skipped, 1,716 s** (end of session 30: the arm moved to 5.70 ft and the Hannibal headline switched to the replayed field, with 2 new tests). Before that **861 passed, 1 skipped, 1,000 s** (session 30, after `boeing747` declared CR-2144's thrust line: 859 plus 2 in `test_cr2144_speed_derivatives.py`, with 9 existing tests re-pointed or re-captured — §4's thrust-line entry says which). Before that **859 passed, 1 skipped, 1,673 s** (session 30, after the Parks Fig. 6 altitude: 852 plus 7 in `test_parks_fig6_altitude.py`). Before that **852 passed, 1 skipped, 1,295 s** (the 844 below plus the 8 tests in `test_hannibal_horizontal_wind.py`). Before that, **844 passed, 1 skipped, 887 s** (measured session 30 after `boeing747` declared CR-2144's speed derivatives: 23 tests in `test_cr2144_speed_derivatives.py`, and 17 existing tests re-captured, moved onto the undeclared entry, split, or fixed — §4's session-30 entry, item 7, says which. Earlier the same session measured **843** before the declaration, which was 821 + 22 with nothing else moved. The wall clock is machine load, not the suite). Previously **821 passed, 1 skipped, 34m37s** at session 28 after the compressibility merge; 812 before it, same session; 811 at session 26; 807 at session 25; it was 788 at session 24 and **758 measured session 23b**; the 626 this row claimed was stale by five sessions, and the 322 before that by several more — this row has now been wrong twice, so re-measure it rather than trusting it). The first thing to run and the only complete statement of what works. `testpaths` is set in `pyproject.toml`, so the bare command collects `atisim/tests`. **Install the `ui` extra as well as `dev`:** `test_artifact.py` and `test_figures.py` `importorskip` at module level, so without `pyarrow` and `plotly` they do not collect at all — **887 collected instead of 937**, green either way and silent about the 50 it left out. Measured at the release on a stranger's clone (§9, point 13). |
| `.venv/Scripts/python.exe scripts/sanity.py` | **The ladder, for a reader who does not yet trust the model.** ~~Twelve cases~~ Eleven checks, **11/11** since session 32 (it read 8/11 before — §9, session 32, point 11), from degenerate inputs upward — zero the wind, zero a coefficient so a motion becomes impossible, then signs, then hand-computable numbers, then structural properties. Every expected value is derived by hand in the source and printed beside the model's answer, so it is read rather than trusted. Ends with the item 08 convention probe, which is a measurement rather than a pass/fail. |
| `.venv/Scripts/python.exe -m sphinx -b html -W --keep-going docs docs/_build/html` | **Builds the documentation site (phase 4).** Needs the `docs` extra. The site's narrative pages `{include}` sections of this file verbatim — the *Running it* page **is** this section, the *Validation* page is §1's claim plus the §5 status table — so editing the record updates the site and nothing can drift. `-W` makes a warning an error, which is how CI runs it (`.github/workflows/docs.yml`); it builds clean with zero warnings. `docs/conf.py` carries a hook that renders the package's plain-prose docstrings as written. |
| `.venv/Scripts/python.exe -m pytest --nbval-lax notebooks/ -q` | **The second gate.** Executes ~~`notebooks/solver-validation.ipynb`~~ both notebooks — `solver-validation.ipynb` and, since session 32, `validation-ladder.ipynb`, which walks §1's claim — so they cannot rot: **24 passed, 157 s** locally and **113 s** in CI (run 35390817171). CI runs it after the suite (`.github/workflows/tests.yml`). Needs the `dev` extra (`jupyter`, `nbval`). Deliberately *not* in `testpaths` and `--nbval-lax` is deliberately *not* in `addopts`: that would make every `pytest` run fail with "unrecognized arguments" wherever nbval is absent. **Run it from a worktree with an ABSOLUTE `PYTHONPATH`** — nbval starts the kernel with its cwd in `notebooks/`, so a relative `PYTHONPATH=.` resolves to the wrong directory and `atisim` silently loads from the main checkout. |
| `.venv/Scripts/python.exe scripts/checkpoint.py` | 747 only, no flags. Trim residuals, 60 s fixed-control hold, longitudinal modes against CR-2144 Table IX-5. |
| `.venv/Scripts/python.exe scripts/tune.py --aircraft cherokee` | Autopilot step responses for one aircraft. Exits non-zero on failure, so it is usable as a gate. |
| `.venv/Scripts/python.exe scripts/fly.py --aircraft cherokee --save runs/a.npz` | Interactive flight, basic-T cockpit plus a flight-test overlay. |
| `.venv/Scripts/python.exe scripts/fly.py --wind hannibal` | The same, hand-flown into the Parks vortex array. The panel counts the range down. |
| `.venv/Scripts/python.exe scripts/vortex.py --case hannibal --png runs/v.png` | Flies the 747 through the Parks vortex array, the Wingrove updraft, and an elevator pushdown, and draws the analysis figure with all three Fig. 8 categories. This is the turbulence path. Prints each point's Δθ, Δn and peak \|α\| with its band, then whether the ordering holds. |
| `.venv/Scripts/python.exe scripts/microburst.py --png runs/mb.png` | Flies the Cherokee through an Oseguera & Bowles microburst at 300 m and reports the 1 km average F against its thrust authority. Cuts the run at one wingspan above the ground and says so. |
| `.venv/Scripts/python.exe scripts/leewave.py --png runs/lw.png` | Flies the 747 through a Doyle et al. lee wave and compares the Bowles F-factor against the aircraft's own `(T−D)/W`. Prints both of the source's flight legs and which of them the engines can cover. |
| `.venv/Scripts/python.exe scripts/analyse.py runs/a.npz` | Replays a saved `.npz`. Accepts several files; `--png DIR` writes instead of showing. |
| `.venv/Scripts/python.exe scripts/vortex.py --artifacts runs/analysis` | The same run, **also written as a run artifact per encounter** — Parquet series plus `meta.json` and `checks.json`. Prints `checks ok` or names the checks that failed. Needs the `ui` extra. Until this flag existed, every number in §4's encounter tables came from a run that did not survive the script that produced it. |
| `.venv/Scripts/python.exe -m atisim.apps.sweep runs/analysis` | **The analysis UI.** Sweep view per run — provenance header with caveats, check badges, the causal strip stack, the 3D field with the trajectory through it, Fig. 8, and n_z-vs-α — plus a shared time cursor: click any strip and every panel, the 3D marker and the readout move to that sample together. Needs the `ui` extra (`pip install -e .[ui]`). |
| `.venv/Scripts/python.exe scripts/summary.py docs/summary/atisim-summary.pdf docs/summary/panel.png` | Rebuilds the plain-English summary PDF (14 pages). The parts that are *computed* cannot drift from the code — the vortex figures call `wind.vortex_wind`, and the aircraft table reads `CRUISE`. **The prose and the summary statistics are literals and can**: the test and line counts were stale by session 7, and four page cross-references were wrong by session 8. The page numbers are now generated from `PAGE_ORDER` with a build-time count check; the statistics are still literals. Re-run it after anything that changes those. |
| `C:/Users/mateusz/AppData/Local/Programs/Python/Python310/python.exe` `scripts/gen_jsbsim_747.py` | **Recovers the `boeing747_jsbsim` entry from the running B747.** Needs jsbsim, so it runs under the reference interpreter above, NOT the project venv. Writes `atisim/tests/data/jsbsim_747_reference.xml`. Run only when the recovery condition changes; drift shows up in `git diff`. |
| `C:/Users/mateusz/AppData/Local/Programs/Python/Python310/python.exe` `scripts/gen_jsbsim_vortex_reference.py` | **Freezes JSBSim's answer to the three vortex cases.** Same interpreter, same reason. Writes `atisim/tests/data/jsbsim_vortex_reference.xml`. |
| `.venv/Scripts/python.exe scripts/vortex_compare.py --png runs/vc.png` | **The cross-code vortex comparison.** Flies atisim through the identical field the frozen reference was generated from and reports where the two engines part, against Wingrove & Bach's own g-loads. Imports no jsbsim. |
| `PYTHONPATH=<abs worktree root> .venv/Scripts/python.exe scripts/les_ensemble_svd.py --runs <dir>` | **The LES ensemble POD (session 28).** Asks whether an SVD of the 16 x 5000 `n_z` arrays separates condition drift from gust response better than the shipped 0.05 Hz high-pass. **It does not** — on D03/D04 mode 1 carries 25% and 18% and is only half sub-0.05 Hz. Then asks the drift question directly, early half against late, and finds it confounded with along-track field inhomogeneity. `--runs` is required: the arrays are gitignored, so they are not in the repository. **Since session 32 they live in the MAIN CHECKOUT at `runs/cat/`** — 55 files, 11 MB, including the ten `les-nz-*.npy` arrays. They were harvested from `claude/weekly-summary-analysis-7520db`'s worktree, which was the only copy, and verified file-by-file by md5 before that branch was abandoned. Regenerating them needs the 17.9 GB figshare dataset, which is held outside the repository. |
| `PYTHONPATH=<abs worktree root> .venv/Scripts/python.exe scripts/svd_probe.py` | **What a singular value decomposition can and cannot see (session 28).** Four sections: cond(J) at the healthy trim against §5's two absurd roots (it does NOT separate them, and that is the point); F7's zero-authority channel, where sigma_min is exactly 0 at the initial guess; and the identifiability of (`Cma`, `Cmq`, `Iyy`) against the longitudinal modes, which recovers session 27's B787 `I_yy` result to 0.19 deg without being told it. Changes nothing — it imports `trim` and `validation` and measures matrices they already build. §4 has what it found. |
| `PYTHONPATH=<abs worktree root> .venv/Scripts/python.exe scripts/fig8_discriminator.py` | **Does Fig. 8's pitch axis stop discriminating? (session 29).** Replaces session 23d's extremes-gap statistic with `response.separability` and `response.standardised_difference`, whose expectation does not move with N. Section A shows the old statistic shrinking with N at fixed σ_w while the new one does not; section B sweeps intensity on both axes, marking every row that is outside the SOURCED σ range or past §1's 10° envelope; section C looks for the crossing and reports that there isn't one. **Checkpoints after every intensity** and prints with flush — redirect it, never pipe it through `tail`, and see `fly_grid` for why. `--seeds`, `--sigmas`, `--clear-every`. |
| `PYTHONPATH=<abs worktree root> .venv/Scripts/python.exe scripts/sensitivity_load.py` | **The headline load, differentiated and swept (session 29, S2/S3).** Section A is the gate — the differentiable path against `vortex_viz._measure`, and the claim is bit-identity. Section B ranks every independent coefficient by elasticity in ONE 4,737-step forward pass. Section C sweeps the top six at ±1/5/10/25% and prints, for each, whether the tangent survived and whether an **extreme changed sample**, which is what a peak-to-peak derivative actually turns on. `--top` widens section C. |
| `PYTHONPATH=<abs worktree root> .venv/Scripts/python.exe scripts/sensitivity_assumptions.py` | **What each MODELLING CHOICE costs the headline load (session 29, S4).** Puts `ASSUMPTIONS.md`'s rows on the same axis as the coefficients — C3's Mach axis, E4's RK4 wind hold, F1's step size, E2/E10's strip path, F5's station count, the declared loading shape. Prints C3's Mach excursion first, per §1's rule that an excursion accompanies every frozen-derivative claim, and ends with a **liveness check on the strip path**, because a 0.000% from "roll only" and a 0.000% from "it never ran" look identical. |
| `PYTHONPATH=<abs worktree root> .venv/Scripts/python.exe scripts/sensitivity_ensemble.py` | **The same screen against an ensemble rms, and the budget (session 29, S5/S6).** Re-runs the ranking on session 25's Dryden protocol to ask whether it is a property of the aeroplane or of the peak — the per-seed standard deviation is printed beside every mean, because most of the table is not resolved at N = 8. Then combines every priced input into linear-sum and RSS brackets on the headline. `--seeds` and `--dt` control the cost. |
| `PYTHONPATH=<abs worktree root> .venv/Scripts/python.exe scripts/sensitivity_screen.py` | **The sensitivity screen (session 29, phases S0/S1).** Four sections: the AD machinery against a central difference at the same point (the S1 gate, worst 5.2e-10); session 11's four fitted slopes re-run; the tangent moved to each swept range's centroid, which is what shows the AD-vs-fit gap is **curvature**; then the screen proper — every independent coefficient against every mode, at the Caughey approach and at cruise, with the **structurally inert** fields listed separately because a zero there means *not used*, not *unimportant*. `--json` writes every number. Imports nothing new. |
| `.venv/Scripts/python.exe scripts/vortex_diagnose.py` | **Why the comparison's two large errors are large.** Three experiments: the same start state flown in still air, atisim flown from its own trim, and a one-lever-at-a-time sweep against the DFDR. Imports no jsbsim. |
| `PYTHONPATH=<abs worktree root> .venv/Scripts/python.exe scripts/digitise_tm102186_fig6.py --outdir runs/cat` | **The recorded g trace (session 27).** Reads TM-102186 Fig. 6's G LOAD panel out of `Reference_papers/19890016606.pdf` at 600 dpi, column by column, as the top and bottom of the ink — nothing fitted, nothing smoothed. Prints the three checks (the paper's own band, a **negative control** on the vertical-wind panel, and the gust spacing) and writes `10-tm102186-fig6.png` plus `tm102186-fig6-gload.csv`. §4 has what it found. |
| `PYTHONPATH=<abs worktree root> .venv/Scripts/python.exe scripts/digitise_mil_f_8785c_fig7.py --outdir runs/cat --pdf refs/MIL-F-8785C.pdf` | **The severe-turbulence σ_w chart (session 27).** Digitises all nine curves of Fig. 7 from printed p. 49, flagging where two share **one stroke of ink** rather than reading a number out of a merge. Settles `mil_f_8785c_sigma_w_exceeds_the_mehta_ceiling`. Writes `11-mil-f-8785c-fig7.png` and `mil-f-8785c-fig7-lines.csv`. **`--pdf` is required from a worktree** — `refs/` is gitignored and lives only in the main checkout. |
| `PYTHONPATH=<abs worktree root> .venv/Scripts/python.exe scripts/cr2144_speed_derivatives.py --dig-dir <folder> --csv-dir <folder> [--headline]` | **CR-2144's speed derivatives, digitised (session 30).** What it does, in order: (1) re-extracts the hand-placed points from the eight Engauge `.dig` files and compares them with the tracked CSV (`--write` regenerates it); (2) audits Engauge's own CSV exports; (3) checks every curve against Table IX-4 through Appendix A; (4) declares the FC9 set on a copy of the 747 and retests all four modes against Table IX-5; (5) prices the hand reading — interpolation, leave-one-out, a pixel Monte Carlo and the check residuals. `--headline` also flies Mehta's field with the set declared. Sections 3–5 run without either folder. **The eight `.dig` originals are tracked in `atisim/data/cr2144_dig/`**, which is where a bare `--dig-dir` looks. |
| `PYTHONPATH=<abs worktree root> .venv/Scripts/python.exe scripts/cr2144_digitisation_crosscheck.py` | **CR-2144 pp. 220–222 read twice, compared (session 32).** Session 30's hand reading against the automated trace in `Reference_papers/CR-2144/csv/`: median disagreement per curve in % of full scale and in the hand sheet's pixels, plus which altitude each hand curve actually sits on. Then scores **both** against Table IX-4 at every circled condition through `cr2144_mach.backsolve`, which is what says which reading is off. Settles `cr2144_two_readings_agree_on_the_good_panels`. Seconds; reads no PDF. **The `PYTHONPATH` is not optional** — without it the script imports the main checkout. |
| `PYTHONPATH=<abs worktree root> .venv/Scripts/python.exe scripts/digitise_hannibal_horizontal_wind.py --pdf <Reference_papers/19890016606.pdf> --outdir runs/cat [--overlay] [--write]` | **The Hannibal horizontal wind, digitised (session 30).** Reads TM-102186 Fig. 7 (printed p. 3-5) at 300 dpi. It separates the solid MODEL line from the dotted ACTUAL curve by shape and calibrates through the printed labels and tick marks. Then it prints three checks: the vertical-panel calibration control, the sign test (as built against every core flipped), and ACTUAL minus AtiSim by stretch. `--overlay` writes `tm102186-fig7-classified.png`; `--write` regenerates the tracked `atisim/data/tm102186_fig7_winds.csv`. **`--pdf` is required from a worktree.** |
| `PYTHONPATH=<abs worktree root> .venv/Scripts/python.exe scripts/digitise_parks_fig6_altitude.py --pdf <Reference_papers/parks-1985-identification-of-vortex-induced-clear-air-turbulence-JA22-2.pdf> --outdir runs/cat [--overlay] [--write]` | **The DC-10's altitude through the Hannibal encounter (session 30).** Reads Parks et al. 1985 Fig. 6 (printed p. 127) at 300 dpi and calibrates each panel on its own ticks, because the scan is skewed. It separates the measured (barometric) altitude from the dashed inertial estimate and reads the load, vertical-wind and airspeed panels for timing. It prints three checks (the paper's load band, its cruise altitude, one clock across panels), then the DC-10's altitude at each of Mehta's cores under both readings of the distance scale. Writes `parks-fig6-altitude.png`; `--overlay` writes `parks-fig6-classified.png`; `--write` regenerates the tracked `atisim/data/parks1985_fig6_altitude.csv`. **`--pdf` is required from a worktree.** |
| `PYTHONPATH=<abs worktree root> .venv/Scripts/python.exe scripts/hannibal_along_track_wind.py --outdir runs/cat` | **What the along-track wind does to the 747 (session 30).** Reads the tracked CSV, so it needs no PDF. It flies seven fields (A as flown … G replayed plus the recorded miss) on the bare and shipped 747, and prints load, pitch, airspeed and each core-passage height. Writes `hannibal-along-track.json` and figures h1–h7. §4 has the table, and the double-counting finding it produced. Several minutes: 14 flights. |
| `PYTHONPATH=<abs worktree root> .venv/Scripts/python.exe scripts/cr2144_report_figures.py --outdir runs/cat` | **The session-30 report figures, s1–s4.** Hand-read curves against Table IX-4, modes before and after, the reading uncertainty, and the headline load before and after. It imports `cr2144_speed_derivatives` and `hannibal_along_track_wind` rather than restating them. |
| `PYTHONPATH=<abs worktree root> .venv/Scripts/python.exe scripts/cat_bounds.py --outdir runs/cat` | **The bounding experiments (session 23 follow-up).** What the point-sampled gust, the strip path and the step size cost on the Mehta run; what Dryden intensity would close the residual load gap; and Lester's Greenland 747 against a lee wave, inverted on both the g-load and the altitude gain. Same `PYTHONPATH` rule. |
| `PYTHONPATH=<abs worktree root> .venv/Scripts/python.exe scripts/lateral.py --outdir runs/cat` | **The lateral dimension (session 24, phase 1).** Reconciles `wind.line_vortex_wind` against `wind.vortex_wind` along the flight path, shows where the oblique difference is, then flies Mehta's field three ways -- point, line, and line with strip-integrated loads -- and reports the bank, sideslip and rolling gust rate the project could not previously see. Writes `08-lateral.png`. Same `PYTHONPATH` rule. |
| `PYTHONPATH=<abs worktree root> .venv/Scripts/python.exe scripts/strip_roll_double_count.py` | **Does the strip path count the gust's roll twice? (session 32, §6(h)).** Flies Mehta's field as line vortices four ways on the shipped 747 -- point path only, point + strip as `strip=True` ships it, strip owning roll with the point roll rate removed, and no gust roll at all -- and prints peak bank for each. Its first two rows reproduce `lateral.py`'s own run to three decimals, which is the harness check. It found the published strip effect of +17.4% is **−2.5%** with one path owning roll. A few minutes: four flights. |
| `PYTHONPATH=<abs worktree root> .venv/Scripts/python.exe scripts/cat_ensemble.py --outdir runs/cat` | **Fig. 8 with error bars (session 23d, section 7 step 6).** Superposes a Dryden layer at the SOURCED sigma_w range from Mehta's residual and reports whether the vortex/updraft/manoeuvre ordering survives, and by how much margin on each of Fig. 8's two axes. Writes `07-ensemble.png`. Same `PYTHONPATH` rule. |
| `PYTHONPATH=<abs worktree root> .venv/Scripts/python.exe scripts/cat_spectra.py --outdir runs/cat` | **Response spectra and load exceedance (session 25, phase 2).** Three limbs: Mehta's headline field as a `n_z` spectrum against the aircraft's short period and the array's four core-passage frequencies; Yoshimura et al. 2023's protocol -- N virtual flights through `wind.dryden_field`, spectra averaged, peak against the airframe's own frequency -- which settles a sealed prediction; and the first load-exceedance curve, both signs, with N in the denominator. `--seeds` defaults to 32; **session 27 ran `--seeds 151` to match Yoshimura and §4 records the result, which is cleaner — use 151 when the peak location matters.** Writes `09-spectra.png`. Same `PYTHONPATH` rule. |
| `PYTHONPATH=<abs worktree root> .venv/Scripts/python.exe scripts/cat_uncertainty.py --outdir runs/cat` | **The Hannibal comparison with error bars (session 23c).** Measures the gust SPACING against TM-102186's "about 5 sec apart" -- the one channel the identification did not set -- converts Mehta's own Eq. (A3) cost into an RMS wind residual and decomposes it against Lester's reconstruction error, then flies the propagated `V0` and `r0` band and a gust-strength sweep to show the peak load is saturated. Writes `06-uncertainty.png`. Same `PYTHONPATH` rule. |
| `PYTHONPATH=<abs worktree root> .venv/Scripts/python.exe scripts/cat_validation.py --outdir runs/cat` | **The CAT source pass (session 23).** Flies Mehta 1987's five-vortex Hannibal field, reproduces TM-102186 Fig. 8's three-aircraft ordering and tests its stated mechanism across the whole registry, compares the 747's short period at a third CR-2144 flight condition, and grades every run on Misaka's `σ_n`. **Since the end of session 30 the 747 headline (section 1, figure 01) flies Mehta's field replayed on its identified path (`wind.on_identified_path`); the fleet ordering and severity table still fly each aircraft at its own altitude.** Prints every number and writes four figures. **`PYTHONPATH` is mandatory** — `python scripts/…` resolves `atisim` to the main checkout, which this script detects and prints on its first line. |
| `docs/summary/jsbsim-atisim-vortex-report.html` | **The written comparison** — the numbers above with the reasoning, the figure, and what the result does and does not establish. Not generated; edit it when the numbers move. |
| `presentation_package/engine_validity_audit.html` | **The session-28 audit, as a page to present from.** The four reference classes on one log axis, the four mechanisms behind the apparent error growth, the strip-load verdict, and the unresolved-pathway inventory with a status on each. Every figure traces to §4 or to this session's re-runs. Not generated; edit it when §4 moves. |
| `PYTHONPATH=<abs worktree root> .venv/Scripts/python.exe scripts/les_flight.py --dataset <figshare root> [--domain D03] [--flights 24] [--aircraft boeing747]` | **AtiSim through Yoshimura et al.'s LES field** — the first wind field this project flies that was *not* identified from the accelerations it is then asked to predict. Reads one LES domain and flies N virtual flights through it beside Yoshimura's own. **Needs the Yoshimura figshare dataset** (21152203, CC BY 4.0, 17.9 GB, held outside the repository); `--dataset` or `ATISIM_LES_ROOT` names its root, the directory holding `les/` and `unpacked/`. §4, "The LES comparison". |
| `PYTHONPATH=<abs worktree root> .venv/Scripts/python.exe scripts/les_compare.py --dataset <figshare root>` | **AtiSim against Yoshimura across all four LES resolutions**, processed identically on both sides — 20 s of settling discarded, high-passed at `response.PHUGOID_FLOOR_HZ`, then rms, peak load, spectral peak and upcrossing rate. The docstring says why the shared filter is not cosmetic. Same dataset as `les_flight.py`. |
| `PYTHONPATH=<abs worktree root> .venv/Scripts/python.exe scripts/les_mach_test.py --dataset <figshare root> [--lift-only] [--baseline]` | **Is the LES discrepancy just the frozen lift slope?** A falsification test: rescales the 747's lift-slope family by the Prandtl–Glauert ratio and hands off to `les_flight.py`, so the reader, path and filtering are byte-identical to the run it tests. Its docstring carries the result — 47–68% of the discrepancy, not all of it — so nobody re-runs it to find out. |
| `PYTHONPATH=<abs worktree root> .venv/Scripts/python.exe scripts/yoshimura_flightsim.py --root <flightsim-data/work>` | **Yoshimura et al.'s own flight simulation, read and measured**: the 561 MB `flightsim-data` part of the figshare download, 604 flights of their 2-D aircraft through their LES field. The simulated half of their Fig. 6, and what the `boeing787_yoshimura` entry was rebuilt from. |
| `PYTHONPATH=<abs worktree root> .venv/Scripts/python.exe scripts/sensitivity_mass_diagnosis.py` | **What session 29's `mass` elasticities actually measure (session 31).** Separates the four things a one-at-a-time move of `ac.mass` could be carrying, each by one controlled variant, by central difference on the shipped paths — so it is independent of the AD machinery. §4, "What session 29's `mass` row measures". |
| `C:/Users/mateusz/AppData/Local/Programs/Python/Python310/python.exe` `scripts/gen_jsbsim_reference.py` | **Freezes JSBSim's answer for the 737 cross-code comparison.** Needs `jsbsim`, so it runs under the reference interpreter, not the project venv; writes `atisim/tests/data/jsbsim_737_reference.xml`, which the suite reads. Run only when the recovery condition changes; drift shows in `git diff`. |
| `.venv/Scripts/python.exe scripts/jsbsim_report.py docs/summary/jsbsim-737-report.pdf` | Rebuilds the **JSBSim 737 cross-code verification report**. Every number is read from the frozen reference or computed at build time. |
| `.venv/Scripts/python.exe scripts/turbulence_report.py docs/summary/turbulence-report.pdf` | Rebuilds the **turbulence-fidelity technical report** — the distributed-airframe wind-shear work: the physics, the source verification, what was built and measured. |
| `.venv/Scripts/python.exe scripts/audit_report.py [--out docs/summary/audit-report.pdf]` | Rebuilds the **audit report PDF** from `audit/AUDIT.md`, which is the only file it reads, so the PDF cannot drift from the document the findings live in. |

### The documents, and which question each answers

| Document | Answers |
|---|---|
| **`docs/PROJECT.md`** (this file) | what exists, what is **measured**, what is known-broken, what happens next |
| **`docs/ASSUMPTIONS.md`** | what is **assumed** before any measurement, with a bound on each. Read before quoting a result to better than ~0.5%, before flying far from a trim point, or before adding a wind field whose scale approaches a wingspan. Its closing section explains what the notebook does and does not demonstrate |
| `docs/design/plans/2026-08-11-close-the-verification-gaps.md` | the current plan, plus a full handover of session 11 for a session that was not there |
| `docs/design/specs/2026-08-11-solver-validation-design.md` | why the verification/validation split, and the source-qualification tiers that answer "is a 1972 document a source of error" |

Flags: `tune.py` takes `--aircraft` only. `fly.py` takes `--aircraft --autopilot --save
--dt --fps --window --seed --wind --lead-in --sharpness`. `vortex.py` takes `--case
{hannibal,morton} --aircraft --dt --lead-in --sharpness --pushdown-seconds --png`.
`--lead-in` below ~12 core radii contaminates the first core (§9 session 3);
`leewave.py` takes `--aircraft --dt --wavelength --waves --png`. `microburst.py` takes
`--aircraft --altitude --dt --u-max --radius --z-m --png`.
`--sharpness`, `--pushdown-seconds` and `--wavelength` are declared modelling parameters,
not source data
— the papers fix the updraft's magnitude and duration but not its edge, and fix the load
the pilot reached but not how long they held it. Both scripts read the case constants from
`wind.PARKS_CASES`, so they cannot disagree about a sourced number. The manoeuvre's
elevator angle is **not** a flag: it is bisected to land on the Fig. 8 load band, so the
sourced quantity is the load and the deflection is an output.

### Flying it

    arrows   centre stick: up is stick forward, so up pitches the nose DOWN
    ,  .     rudder left/right
    -  =     throttle down/up
    [  ]     pitch trim, nose down/up
    t        trim here — hold the deflections the stick is holding now
    a        toggle manual/autopilot

The stick **ramps** rather than snapping to full travel: a held key reaches the stop in
0.4 s and a released one springs back at the same rate, so a tap is a small input. That
rate is a declared figure in `panel.py`, not a measured one, and it is stepped on the
physics clock — stepping it per frame would make the feel depend on the render rate.

Releasing a surface axis returns it to `ManualState.reference`, and **trim is what moves
that reference**. Without trimming, the reference is whatever the surfaces were doing at
the last mode handover, so after a manoeuvre it is stale and the aircraft drifts. `t`
trims to what the stick is holding *right now*, so it must be pressed while the stick is
still held — once the stick has centred, the surfaces are already at the reference and
trim-here correctly does nothing. The throttle stays where it is left, because a lever
does, and trim is not sprung either.

Close the window to end the flight — the post-flight figure opens afterwards, and
`--save` writes the `.npz` first.

### What is on the panel

Basic T: airspeed and altitude tapes flanking the attitude ball, VSI beside the altitude,
heading tape below. The slip ball is at the top of the ball and reads **lateral specific
force, not β** — a ball is a pendulum, and the two quantities agree only in steady
coordinated flight. The teal marker on the ball is the body-axis **incidence** pair
(−α, +β); it is deliberately not called a flight path vector, which is earth-referenced
and would rotate with bank.

The overlay is the flight-test half: load factor with a peak hold, air-relative α against
the **declared** linear-aero band (green to ±10°, amber to ±12°, red beyond — PROJECT.md
§7, not a stall table; the band is **symmetric** because `aero.py` is odd-symmetric in α,
so a pushdown leaves the model exactly as far as an equal pull-up — see §6(e)), the
applied wind, and the gust rate labelled **SIM TRUTH** because
`omega_gust` is a span-wise gradient and no instrument can sense it. Under `--wind` the
status line carries the range to the field: a north distance and a closure rate for a
vortex array, whose cores are infinite east–west lines and therefore have no bearing, and
a range and bearing for an updraft column, which is a point.

Physics runs at a fixed 50 Hz regardless of frame rate; rendering targets 20 fps and
measures itself to hold that (matplotlib's `interval` is the gap between frames, not the
period). Current measurement is **26.4 fps headless on Agg** (§4, session 7).

Superseded, kept per §4's rule: "Measured on TkAgg: 19.9 fps, real-time ratio 0.9994, no
drift over 15 s" was taken **before** the session-6 re-layout replaced a 3D axes with a
dozen 2D ones, and is no longer a statement about this panel. The interactive rate has
not been re-taken since (§8).

### Which paths are trustworthy under wind

This is the part that is easy to get wrong, because most of the tooling predates the wind
model and silently assumes still air.

| Path | Under wind |
|---|---|
| `scripts/vortex.py`, `vortex_viz.py` | **Correct.** Air-relative throughout, by construction. |
| `integrate.step`, `dynamics`, `aero` | **Correct.** The core has always been air-relative. |
| `autopilot.py`, `manual.py` | **Correct since session 5.** Take `AirData`; the speed loop holds true airspeed. |
| `scripts/fly.py`, `panel.py` | **Correct since session 6.** `run_live` takes a `wind_model` and the panel senses from the wind the last step applied. Before that the live path could not fly through a field at all. |
| `viz.py` `Derived` and `post_flight` | **Correct since session 5.** Sensed from the recorded wind. |
| `viz.Trajectory` / saved `.npz` | Records the applied wind. Files written before session 5 load as still air. |
| `trim.py` | Still-air by construction and must stay so. Not a defect. |

Ask for state through `sensors.sense(state, wind_ned)`. Reaching into `state.vel_body` for
"airspeed" is the bug that took three sessions to find — see §6.

### Aircraft

`--aircraft` accepts `boeing747`, `boeing747_approach`, `cherokee`, `cessna172`. The 747 is the only one with
modes validated against a source (§4) and the only one used for turbulence work.
`boeing747_approach` is the same airframe at CR-2144's power-approach point (sea level,
165 KTAS, 20° flaps, gear up) and exists for low-altitude windshear work; it sits below
V_md, so fly it **open loop** (§4, §5). The
Cherokee is the validated light aircraft. **The Cessna is out of scope** (§5): its rudder
set is zeroed because the source omits it, so its turns are uncoordinated. It trims, flies
and passes its tests, but no result should be quoted from it.

### Library use, without any script

```python
import atisim                                   # enables x64 — import first
from atisim import trim, integrate, autopilot as ap
from atisim.aircraft import REGISTRY, CRUISE

ac = REGISTRY["boeing747"]
V, H = CRUISE["boeing747"]["airspeed"], CRUISE["boeing747"]["altitude"]
x, residual = trim.trim(jnp.array(V), jnp.array(H), ac)         # [alpha, elevator, throttle]
state    = trim.trimmed_state(x[0], jnp.array(V), jnp.array(H))
controls = trim.trimmed_controls(x[1], x[2])
sim      = integrate.init_sim(state, jax.random.PRNGKey(0))
```

- Open loop: `integrate.rollout(sim, controls, dt, ac, n_steps, wind_model=...)`.
- Closed loop: `ap.engage(...)` then `ap.closed_loop_rollout(...)` — autopilot inside `lax.scan`.
- Ensembles: `integrate.batch_sim(state, keys)` then `jax.vmap` the rollout. Deterministic
  wind components return the key untouched, so every member meets the same field.
- Wind: build with `wind.VortexArray` / `wind.UpdraftColumn`, wrap with
  `wind.field_model(...)`, combine with `wind.superpose(...)`, pass as `wind_model=`.
- `trim.minimum_drag_speed(ac, altitude)` bounds the autopilot: below V_md the
  throttle-to-speed, elevator-to-altitude pairing inverts and no gain set repairs it. The
  Cherokee cruises 2.7 m/s above it, the 747 2.5 m/s.

### Environment notes

The `.venv` lives in the project root and only there — running from a git worktree needs
`PYTHONPATH` set to the worktree, or the editable install resolves to the main checkout.
Live flight needs an interactive matplotlib backend (TkAgg is the default here); the test
suite forces Agg and drives the animation, blitting and key events for real.
