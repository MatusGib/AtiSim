# Phase 3 — Close Every Open Item: Implementation Plan

**Goal:** At the end of Wed 23 September, every open item in `docs/PROJECT.md` §5, §7 and §8 carries one explicit terminal status with its evidence, and §0 holds nothing but the work that is deliberately parked.

**Architecture:** Mostly a record-keeping phase with three cheap measurements. Day 1 corrects the record — stale rows, one error of mine from Phase 2, and a status table over every open item. Days 2–3 run the measurements that are genuinely cheap and genuinely new, each with its expected result sealed *before* it runs. Anything not reached by the hard stop becomes a FUTURE WORK row naming its route.

**Tech Stack:** Python 3.10, JAX, NumPy, pytest; `atisim/predictions.py` for sealing; JSBSim 1.3.1 on the reference interpreter (Task 11 only); eCFR over the web (Task 8 only).

**Runs on:** a fresh branch `phase-3` off `main` (`0458ef6`), in `.claude/worktrees/new-session-943052`.

---

## Read this first — what the inventory found

The plan in `docs/design/specs/2026-09-17-final-release-cleanup-design.md` gave Phase 3 one day, sized on what §7 *says* is open. **Reading each item against the code and against §4 changed that list materially.**

**1. An error of mine from Phase 2, corrected first.** I wrote that "`ASSUMPTIONS.md` A1 and A2 stay open on `main`" — in §9 twice, in the spec, and in PR #9's description. **A2 is not open on `main`.** The compressibility branch merged in session 28 already models `g(z) = g₀(R/(R+z))²` (`atisim/dynamics.py:31`), and `ASSUMPTIONS.md` marks **A2 and A3 both retired**. Only **A1** — flat, non-rotating Earth — stays open. What the WGS-84 union would add beyond `main` is gravity's *latitude* variation (0.53%) and the centrifugal term, which `main`'s A2 already lists as still assumed. Task 1.

**2. Five rows are stale — and two of them are the ones §7 calls highest priority.**

| Row | What it says | What is true |
|---|---|---|
| §7 "build the `boeing787_yoshimura` registry entry, then re-fly the LES" | "NEW, session 27, and it is the top of the list" | **Done in session 27.** `atisim/aircraft.py:1798`, flown through D03: ratio **1.427 → 1.202, 53%** of the discrepancy closed. §4 has the table |
| §7 "test the frozen-`C_Lα` explanation of the LES ratio" | "NEW, session 27, and it is CHEAP" | **Done in session 27.** PG-corrected 747 **1.225 (47%)**, lift-only **1.135 (68%)**. Same §4 table |
| §5 "Gravity is constant at 9.80665 m/s²" | still true | **False since session 28's merge** — `g(z)` is modelled |
| §5 "The Hannibal encounter is dated two ways" | cite April 1981 | **Closed by session 31**: NTSB CHI81DA042 dates it **3 April 1981** |
| §8 "what does `PARKS_CASES["hannibal"]` being a hybrid cost?" | open | **Moot since session 26**, which deleted the hybrid |

Planning from §7's wording would have spent a day redoing finished work. Task 2 supersedes all five in place.

**3. Most of §5 is already terminal.** 13 of its 20 items are understood, documented and bounded; they need a status *label*, not work. The genuinely open list is short, and it is in the table below.

---

## Decisions for you, before execution

These change what gets done. My recommendation is first in each.

| # | Decision | Recommended | Alternative |
|---|---|---|---|
| **D1** | Add a fourth status, **FUTURE WORK** — an item with a named, credible route that this release does not pursue | **Yes.** The spec allowed three: *measured*, *impossible with sources held*, *abandoned*. "Abandoned" undersells real leads like 14 CFR 25.341 and the rotating-Earth union, and a finished research repository normally ends with exactly this list | Force every such item into "abandoned" |
| **D2** | Which measurements to run in Days 2–3 | **Tasks 7, 8, 9** (all cheap, all new, none moves §1's headline), then **Task 10's review** | Also Task 11 (engines, ~half a day) |
| **D3** | The α̇ branch — `claude/engine-validity-presentation-1408e8` | **Review the rule-3 question (Task 10, ~2 h), then park it as FUTURE WORK.** Merging adds `Cmadot` to the 747 on top of session 30's speed derivatives and thrust line, which moves the four CR-2144 modes and §1's headline — a week before release, just as the docs and notebook are written from it | Review and merge, spending most of Wed on re-pinning tests and re-measuring §1 |
| **D4** | Download the NTSB pre-1982 database (39 MB Access file) to look for the DC-10's weight | **No — FUTURE WORK.** Session 31 recorded it as "not tried"; no Access reader is installed, and whether it even carries gross weight is unknown | Yes, and it needs your explicit download approval |
| **D5** | The two deferred `flightsim/`-era branches | **Harvest their findings, then abandon with SHAs** (Task 5). Their commit titles claim bugs — "find roll counted twice", "a third `Ixz`" — which must be checked against `main` before the branches go | Merge them: 92–94 commits behind, on the old package name, touching two rule-3 files |

---

## Schedule, and the hard stop

| Day | Date | Work |
|---|---|---|
| 1 | **Mon 21** | Tasks 1–6: correct the record, status table, harvest the two branches |
| 2 | **Tue 22** | Tasks 7–9: the CR-2144 crosscheck, 14 CFR 25.341, the three unassessed NASA papers |
| 3 | **Wed 23** | Task 10 (α̇ review); Task 11 if D2 includes it; Task 12 close-out |
| — | **Wed 23, end of day** | **HARD STOP.** Whatever is unfinished becomes a FUTURE WORK row. Phase 4 (documentation) starts Thu 24 regardless |

Phase 2's descope freed Mon 21 – Thu 24; this gives three of those days to Phase 3 and keeps two for documentation, which is the release's primary deliverable.

---

## Every open item, and the status this plan gives it

This is the table Task 3 inserts at the head of §5. **It is the phase's main deliverable, and the main thing to review here.**

### §5 — attributed gaps and structural impossibilities

| # | Item | Proposed status | Evidence / route |
|---|---|---|---|
| 5.1 | Vortex core 2.3–3.1 spans, gust sampled at a point | **BOUNDED** | point-gust cost ≤ 4.4% on the headline (§5.19); strip path exists since session 24 |
| 5.2 | No wind varied across the span | **CLOSED**, session 24 | lateral phase |
| 5.3 | No frequency-domain comparison | **CLOSED** (capability), session 25 · **IMPOSSIBLE WITH SOURCES HELD** (published in-band curve) | HICAT is 45–70 kft against this project's 33–41 kft |
| 5.4 | Gravity constant | **SUPERSEDED** — `g(z)` modelled since session 28 | latitude (0.53%) and centrifugal remain DECLARED; `wgs84-earth` retires them |
| 5.5 | Phugoid / short-period offsets | Mach content **CLOSED**, session 30 · `Ṁw` → **FUTURE WORK** (D3) · `Żw` sign **IMPOSSIBLE WITH SOURCES HELD** | IX-4 and IX-5 cannot arbitrate the sign (the α̇ branch's own argument) |
| 5.6 | CR-2144 derivatives are the flexible airframe | **BOUNDED**, session 23 | constant-Mach, two-altitude argument |
| 5.7 | `trim` converges to absurd roots for degenerate coefficients | **GUARDED** | the 15° bound, non-binding on the registry's 0.01°–5.62° |
| 5.8 | Drag polar away from its fitted point | **BOUNDED** | residuals 0.004 / 0.014 / 0.006 quantified |
| 5.9 | Vortex parameter uncertainty | first layer **MEASURED** (8.45%) · second layer **IMPOSSIBLE WITH SOURCES HELD** | NASA's DC-10 aerodynamic model is unpublished (TM-102186 §2) |
| 5.10 | Lee wave has no horizontal perturbation | **IMPOSSIBLE WITH SOURCES HELD** | Doyle et al. give neither N nor cross-mountain U at 12 km |
| 5.11 | Lee-wave wavelength declared | **DECLARED, cross-checked** | Lester et al. 1989 measure ~22 km, 12% from the declared 25 |
| 5.12 | 747 cannot fly a microburst | **CLOSED**, session 10 | |
| 5.13 | FAA windshear thresholds are jet-only | **RESOLVED** | physical verdict `F > (T−D)/W`; `σ_n` for altitude |
| 5.14 | Approach 747 below minimum-drag speed | **ATTRIBUTED — out of scope** | flown open loop, stated |
| 5.15 | No ground | **OUT OF SCOPE, declared** | runs cut at one span |
| 5.16 | Two errors in the sources | **DOCUMENTED** | |
| 5.17 | Hannibal dated two ways | **CLOSED**, session 31 | NTSB CHI81DA042: 3 April 1981 |
| 5.18 | ±g asymmetry | buffet boundary **DONE**, session 26 · nonlinear lift curve **IMPOSSIBLE WITH SOURCES HELD** | CR-114494 draws `CL_BASIC` as straight lines |
| 5.19 | Absolute agreement with recorded g-loads | **IMPOSSIBLE WITH SOURCES HELD** · route **FUTURE WORK** | aircraft identified (N1809U, DC-10-10); weight on the day not found. Routes: NTSB docket, pre-1982 database (D4), 1982 SFTE paper |
| 5.20 | Half the Fig. 8 load band unreachable | **STRUCTURAL** — linear aero | same ceiling as 5.18 |

### §7 — the plan's open rows

| Row | Proposed status |
|---|---|
| DC-10 cruise derivative set | **IMPOSSIBLE WITH SOURCES HELD** — CR-3677 lacks `Cmq`, `Cmα̇`, mass, inertias, S, c̄; CR-3748 has no derivatives (session 31). **Final after Task 9** reads the three unassessed papers |
| Hannibal flight record | flight and aircraft **CLOSED**, session 31 · weight as §5.19 |
| Build `boeing787_yoshimura` | **DONE**, session 27 — stale, superseded by Task 2 |
| Frozen-`C_Lα` LES test | **DONE**, session 27 — stale, superseded by Task 2 |
| Run the LES limb | **ATTRIBUTED** — a 20% residual attributable to neither aircraft nor Mach · **FUTURE WORK** |
| Compare the two CR-2144 digitisations | **Task 7** |
| The α̇ derivatives | **Task 10**, then per D3 |
| Hannibal inventory items 2–6 | **FUTURE WORK**, each with its script already named in §7 |
| 14 CFR 25.341 | **Task 8** |
| NASA TP-2469, Sharman et al. 2014, MIL-STD-1797A | **FUTURE WORK** |
| Sensitivity: C3's α axis, interaction terms | **FUTURE WORK**, declared limits of the study |

### §8 — open questions

| Question | Proposed status |
|---|---|
| Where would Fig. 8's pitch axis fail? | **IMPOSSIBLE WITH SOURCES HELD** — needs a nonlinear lift curve, not published |
| Why do the two engines choose different cores on the array? | **Task 11** if D2 includes it, else **FUTURE WORK** with the discriminator §8 already names |
| Units of Mehta's cost `J`, and `N` | **IMPOSSIBLE WITH SOURCES HELD** — Mehta 1987 never labels `J`; the 4.46 m/s ceiling stands unconditionally, the 2.11 m/s split does not |
| Cost of the `PARKS_CASES` hybrid | **MOOT** — deleted session 26; superseded by Task 2 |

### The sealed prediction

`dc10_does_not_close_the_hannibal_gap` stays **SEALED and is not edited** (`atisim/predictions.py`, rule 1). It is **unsettleable with sources held**: it needs a DC-10 built from a published derivative set. One note belongs beside it in §7, not in the entry: the 747's peak-to-peak has moved since sealing — 1.8385 g then, about **2.03 g** now (75.3% of 2.70 g) — but the claim's band is absolute, **[1.563, 2.114] g**, so it remains decidable exactly as written.

---

## Files

| File | Change | Task |
|---|---|---|
| `docs/PROJECT.md` | §0, §5, §6, §7, §8, §9 — supersede in place, never delete | 1–6, 12 |
| `docs/design/specs/2026-09-17-final-release-cleanup-design.md` | correct the A2 claim | 1 |
| `atisim/predictions.py` | seal one new prediction; settle it | 7 |
| `atisim/cr2144_mach.py` | add `automated_curves()`, `CrossCheck`, `crosscheck()` beside the existing `curves()` | 7 |
| `atisim/tests/test_cr2144_crosscheck.py` | new | 7 |
| `scripts/cr2144_digitisation_crosscheck.py` | new | 7 |
| `Reference_papers/SOURCES.md` | three papers moved from "not assessed" to assessed | 9 |
| `scripts/gen_jsbsim_vortex_reference.py` | `--still-air --out` flag, writing a **separate** file | 11 |

---

## Day 1 — Mon 21: correct the record

### Task 0: Branch and verify the tree

- [ ] **Step 1: Check out `phase-3`** — it already exists, carrying this plan as its first commit.

```bash
cd "C:/Users/mateusz/UROP/Claude_Flight_Sim/.claude/worktrees/new-session-943052"
git fetch -q origin && git checkout -q phase-3 && git rebase -q origin/main
git log --oneline -1
```

Expected: the plan's commit, on top of the current `main`.

- [ ] **Step 2: Rule 4 — confirm which tree Python imports**

```bash
../../../.venv/Scripts/python.exe -c "import atisim; print(atisim.__file__)"
```

Expected: a path ending `worktrees\new-session-943052\atisim\__init__.py`. **If it points anywhere else, stop.** Every run after this would measure someone else's code.

### Task 1: Correct my A2 error, everywhere it was written

**Files:** Modify `docs/PROJECT.md` (the two §9 sentences), `docs/design/specs/2026-09-17-final-release-cleanup-design.md` (Phase 2 amendment). PR #9's description on GitHub.

- [ ] **Step 1: Confirm the three places**

```bash
git grep -n "A1 and A2 stay\|A1 and A2 remain" -- docs/
```

Expected: `docs/PROJECT.md` twice (§9 point 8, and the "did NOT do" paragraph), the spec once.

- [ ] **Step 2: In §9 point 8, replace the sentence**

Old: `**What it costs the release, stated plainly: `ASSUMPTIONS.md` A1 and A2 stay open on `main`.**`

New:

```markdown
~~**What it costs the release, stated plainly: `ASSUMPTIONS.md` A1 and A2 stay open on `main`.**~~
**CORRECTED, session 32: only A1 stays open.** A2 and A3 were already retired on `main` by the
compressibility merge in session 28 — `dynamics.gravity(z) = g₀(R/(R+z))²` and the
geopotential conversion. What the union would add beyond `main` is gravity's latitude variation
(0.53%) and the centrifugal term, both of which `main`'s A2 lists as still assumed. The error was
made writing this entry and repeated in PR #9; it did not reach code.
```

- [ ] **Step 3: In the "did NOT do" paragraph, change** `so `ASSUMPTIONS.md` A1 and A2 remain` **to** `so `ASSUMPTIONS.md` A1 remains` **and add** `(A2 was already retired on `main` — see point 8's correction)`.

- [ ] **Step 4: In the spec's Phase 2 amendment**, wrap `` `ASSUMPTIONS.md` A1 and A2 stay **open on `main`** `` in `~~ ~~` and append: `**Corrected, session 32: only A1 stays open — A2 and A3 were retired on `main` in session 28.**`

- [ ] **Step 5: Correct PR #9's description** (it is merged; the description is still editable)

```bash
gh pr edit 9 --body "$(gh pr view 9 --json body --jq .body | sed 's/\*\*`ASSUMPTIONS.md` A1 and A2 stay open on `main`.\*\*/**`ASSUMPTIONS.md` A1 stays open on `main`.** (Corrected: A2 and A3 were already retired there in session 28.)/')"
gh pr view 9 --json body --jq .body | grep -n "A1"
```

Expected: the line now reads "A1 stays open". **Commit message `a9f40d1` keeps the error** — history is not rewritten for this; the record is.

- [ ] **Step 6: Commit**

```bash
git add docs/ && git commit -q -m "Correct my own claim that A2 stays open on main: it was retired in session 28"
```

### Task 2: Supersede the five stale rows in place

**Files:** Modify `docs/PROJECT.md` §5 (items 5.4, 5.17), §7 (the `boeing787_yoshimura` and frozen-`C_Lα` rows), §8 (the `PARKS_CASES` hybrid question).

The house rule is to strike through and add the correction beside it — never delete a row.

- [ ] **Step 1: §7, `boeing787_yoshimura` row.** Strike `**NEW, session 27, and it is the top of the list.**` and append:

```markdown
**DONE, session 27 — this row was stale for five sessions.** `atisim/aircraft.py:1798`, flown
through D03 against Yoshimura's own ensemble: the ratio goes **1.427 → 1.202**, closing **53%** of
the discrepancy, with no frozen-slope error at all because its derivatives are tabulated at
M 0.406. §4, "The LES comparison", has the table. Found stale in session 32 by reading §4 before
planning from §7.
```

- [ ] **Step 2: §7, frozen-`C_Lα` row.** Same treatment:

```markdown
**DONE, session 27.** In the same §4 table: the full Prandtl–Glauert-corrected 747 reads **1.225
(47% closed)** and lift-only **1.135 (68%)**. The Mach axis is a partial explanation, not the
whole one — which is why the residual is recorded as attributable to neither aircraft nor Mach.
```

- [ ] **Step 3: §5.4, gravity.** Strike the heading claim and prepend:

```markdown
**SUPERSEDED, session 28's merge — found stale session 32.** Gravity is no longer constant on
`main`: `dynamics.gravity(z) = g₀(R/(R+z))²`, and `ASSUMPTIONS.md` A2 is retired. Still assumed:
the latitude variation (0.53%) and the centrifugal term. `wgs84-earth` carries both. The
session-12 measurement below is kept because it is what said the change was safe.
```

- [ ] **Step 4: §5.17, the date.** Append: `**CLOSED, session 31.** NTSB case CHI81DA042 dates the encounter **3 April 1981**, which settles it for the NASA documents against Mehta.`

- [ ] **Step 5: §8, the hybrid.** Strike the question and append: `**MOOT since session 26**, which obtained Parks 1985, adopted 600 ft, and deleted the hybrid; §8's core-radius entry records it. Found still listed as open in session 32.`

- [ ] **Step 6: Verify nothing was deleted**

```bash
git diff --stat docs/PROJECT.md
git diff docs/PROJECT.md | grep -c "^-[^-]"
```

Expected: deletions only on the struck lines (each removed line re-appears with `~~` in a `+` line). Any other `-` line is a mistake.

- [ ] **Step 7: Commit** — `git commit -am "Supersede five stale rows, two of them the ones section 7 called highest priority"`

### Task 3: Insert the status table at the head of §5

**Files:** Modify `docs/PROJECT.md`, directly under `## 5. Attributed gaps and structural impossibilities`.

- [ ] **Step 1:** Insert a subsection `### Status of every open item, at the release — session 32` containing the **§5, §7 and §8 tables and the sealed-prediction note from this plan's "Every open item" section, verbatim**, adjusted only for D1–D5 as decided.

- [ ] **Step 2: Check every §5 item appears exactly once**

```bash
S5=$(grep -n '^## 5\.' docs/PROJECT.md | cut -d: -f1); S6=$(grep -n '^## 6\.' docs/PROJECT.md | cut -d: -f1)
sed -n "${S5},${S6}p" docs/PROJECT.md | grep -c "^| 5\.[0-9]"
```

Expected: `20`.

- [ ] **Step 3: Commit** — `git commit -am "Give every open item in sections 5, 7 and 8 one terminal status, in one table"`

### Task 4: Record the sealed prediction as unsettleable — without touching it

- [ ] **Step 1: Confirm the entry is untouched and its digest still verifies**

```bash
../../../.venv/Scripts/python.exe -m pytest atisim/tests/test_predictions.py -q
```

Expected: all pass.

- [ ] **Step 2:** Add the note from this plan's sealed-prediction paragraph to §7's DC-10 row. **Do not edit `atisim/predictions.py`.** Commit.

### Task 5: Harvest the two `flightsim/`-era branches, then abandon them (D5)

Their commit titles claim bugs. Those must be checked against `main` before the branches go — the same order Phase 1 used.

- [ ] **Step 1: Read every claim**

```bash
git log --format='%h %s%n%b%n----' main..claude/linearisation-verification-bounds-b73868 main..claude/flight-dynamics-solver-oscillation-17139b
```

- [ ] **Step 2:** For each claimed bug — **"roll counted twice"**, **"a third `Ixz`"**, **"the zero columns"** — find the code it names on `main` (`atisim/validation.py`, `atisim/verification.py`, `atisim/aircraft.py`) and classify it **LIVE** (present on `main`), **FIXED** (already corrected on `main`), or **N/A** (the code no longer exists). Record the file:line on `main` for each.

- [ ] **Step 3:** A **LIVE** bug becomes a §6 entry with the branch SHA as its source, **not a fix**. Fixing it is out of scope a week before release unless it is one line and its test already exists on the branch.

- [ ] **Step 4: Record the SHAs, then delete**

```bash
for b in claude/linearisation-verification-bounds-b73868 claude/flight-dynamics-solver-oscillation-17139b; do echo "$b $(git rev-parse --short $b)"; done
git -C ../flight-dynamics-solver-oscillation-17139b status --porcelain
```

Expected: two SHAs (copy them into §0), and **no output** from the status line. If the status line prints anything, stop and commit it to the branch first. Then:

```bash
git worktree remove ../flight-dynamics-solver-oscillation-17139b
git branch -D claude/linearisation-verification-bounds-b73868 claude/flight-dynamics-solver-oscillation-17139b
```

- [ ] **Step 5:** Add both to §0's abandonment table with SHAs and findings. Commit.

### Task 6: Day 1 gate

- [ ] **Step 1:** The §5 check from Task 3 returns 20; `git grep -n "A1 and A2 stay"` returns nothing in `docs/`; `test_predictions.py` passes.
- [ ] **Step 2:** Push `phase-3` and open a draft PR so Day 1 is not a local-only result:

```bash
git push -q -u origin phase-3 && gh pr create --draft --base main --head phase-3 --title "Phase 3: close every open item" --body "Day 1: record corrected. Days 2-3 in progress."
```

---

## Day 2 — Tue 22: three cheap measurements

### Task 7: Two independent digitisations of CR-2144 pp. 220–222, compared

**Why:** session 30 read these pages by hand (295 points); an automated trace of pp. 218–228 was found untracked in session 32. Every error estimate session 30 made is internal to its own reading. A second trace of the same ink prices it from outside. **It needs no new source and moves no model number.**

**Files:**
- Modify: `atisim/predictions.py` (seal, then settle)
- Modify: `atisim/cr2144_mach.py` (add after `value()`, around line 112)
- Create: `atisim/tests/test_cr2144_crosscheck.py`
- Create: `scripts/cr2144_digitisation_crosscheck.py`

- [ ] **Step 1: Seal the expected result BEFORE writing any comparison code.** Append to `PREDICTIONS` in `atisim/predictions.py`:

```python
    Prediction(
        name="cr2144_two_readings_agree_on_the_good_panels",
        claim=("On the four CR-2144 pp. 220-222 panels the automated trace rates "
               "'good' (CL_alpha, CD_alpha, Cm_alpha, Cm_M), session 30's hand "
               "reading and the automated trace agree to within 2.0% of the "
               "panel's full scale -- the median over the hand-placed points -- "
               "at every altitude, and each hand curve sits closest to the "
               "automated curve of its OWN altitude."),
        falsified_if=("Any of those twelve curves shows a median disagreement "
                      "above 2.0% of full scale, or a hand curve sits closer to a "
                      "different altitude's automated curve."),
        reasoning=("2% of full scale is the automated trace's own stated accuracy "
                   "(Reference_papers/CR-2144/README.md: 'good to about 1-2% of full "
                   "scale', the printed line width dominating). The hand reading was "
                   "checked against Table IX-4 at eight conditions in session 30. "
                   "Two readings each good to that level of the same ink should "
                   "agree to it."),
        settled_by="atisim.cr2144_mach.crosscheck()",
        sealed_at="",   # filled in Step 1b -- computed, never typed from memory
        digest="",      # filled in Step 1c
    ),
```

**Step 1b** — `sealed_at` is the commit the author can see. Paste the output as the string:

```bash
git rev-parse --short HEAD
```

**Step 1c** — `digest` is computed from the entry as it now stands, so the claim is hashed exactly as written. Paste the output as the string:

```bash
../../../.venv/Scripts/python.exe -c "from atisim.predictions import PREDICTIONS, digest_of; print(digest_of(next(p for p in PREDICTIONS if p.name == 'cr2144_two_readings_agree_on_the_good_panels')))"
```

Then:

```bash
../../../.venv/Scripts/python.exe -m pytest atisim/tests/test_predictions.py -q
git add atisim/predictions.py && git commit -q -m "Seal what the two CR-2144 readings should agree to, before comparing them"
```

Expected: `test_predictions.py` passes (the digest verifies). **The commit is the seal; nothing below may run before it.**

- [ ] **Step 2: Write the failing test** — `atisim/tests/test_cr2144_crosscheck.py`:

```python
"""Two independent digitisations of CR-2144 pp. 220-222, against each other.

The band is the automated trace's OWN stated accuracy, 2% of full scale
(Reference_papers/CR-2144/README.md), fixed before the comparison was run and
sealed as `cr2144_two_readings_agree_on_the_good_panels` in atisim/predictions.py.
It is asserted only on the four panels that README rates 'good'.

If it fails, that is a finding for docs/PROJECT.md section 4. The band is NOT to
be widened (docs/DEVELOPMENT.md rule 3); mark the failing case
pytest.mark.xfail(strict=True) citing the section-4 entry instead.
"""
import pytest

from atisim import cr2144_mach

GOOD = ("cl_alpha", "cd_alpha", "cm_alpha", "cm_m")   # README confidence: good
BAND_PCT_FS = 2.0

@pytest.fixture(scope="module")
def rows():
    return cr2144_mach.crosscheck()

def test_every_hand_curve_finds_its_automated_counterpart(rows):
    compared = {(r.quantity, r.altitude) for r in rows}
    assert compared == set(cr2144_mach.curves()) - {("cm_q", a) for a in ("SL", "20K", "40K")}

@pytest.mark.parametrize("quantity", GOOD)
def test_the_two_readings_agree_on_the_good_panels(rows, quantity):
    for r in (r for r in rows if r.quantity == quantity):
        assert r.median_pct_fs <= BAND_PCT_FS, (
            f"{quantity} {r.altitude}: median disagreement {r.median_pct_fs:.2f}% "
            f"of full scale over {r.n} points")

@pytest.mark.parametrize("quantity", GOOD)
def test_each_good_curve_sits_on_its_own_altitude(rows, quantity):
    for r in (r for r in rows if r.quantity == quantity):
        assert r.best_altitude == r.altitude, (
            f"{quantity}: the hand {r.altitude} points sit closest to the "
            f"automated {r.best_altitude} curve")
```

- [ ] **Step 3: Run it; it must fail for the right reason**

```bash
../../../.venv/Scripts/python.exe -m pytest atisim/tests/test_cr2144_crosscheck.py -q
```

Expected: FAIL — `AttributeError: module 'atisim.cr2144_mach' has no attribute 'crosscheck'`.

- [ ] **Step 4: Implement** — add to `atisim/cr2144_mach.py` after `value()`:

```python
AUTO_DIR = Path(__file__).parent.parent / "Reference_papers" / "CR-2144" / "csv"

# The automated trace names files by printed symbol and altitude in feet; the
# hand reading uses snake_case quantities and SL / 20K / 40K. cm_q is absent
# from the hand reading, so it is not mapped.
AUTO_NAME = {"cl_alpha": "p220_CL_alpha", "cd_alpha": "p220_CD_alpha",
             "cm_alpha": "p221_Cm_alpha", "cm_alpha_dot": "p221_Cm_adot",
             "cl_m": "p222_CL_M", "cd_m": "p222_CD_M", "cm_m": "p222_Cm_M"}
AUTO_ALT = {"SL": "SL", "20K": "20000ft", "40K": "40000ft"}

def automated_curves(csv_dir: Path = AUTO_DIR) -> dict:
    """{(quantity, altitude): (mach, value)} from the automated pp. 218-228
    trace, for the curves the hand reading also has. Lines starting '#' are the
    file's provenance header and are skipped."""
    out = {}
    for q, stem in AUTO_NAME.items():
        for alt, suffix in AUTO_ALT.items():
            p = Path(csv_dir) / f"{stem}_{suffix}.csv"
            if not p.exists():
                continue
            lines = [ln for ln in p.read_text(encoding="utf-8").splitlines()
                     if ln and not ln.startswith("#")]
            data = np.array([[float(x) for x in ln.split(",")] for ln in lines[1:]])
            order = np.argsort(data[:, 0], kind="stable")
            out[(q, alt)] = (data[order, 0], data[order, 1])
    return out

class CrossCheck(NamedTuple):
    quantity: str
    altitude: str
    n: int                 # hand points compared
    median_pct_fs: float   # median |hand - automated|, % of the panel's full scale
    max_pct_fs: float
    median_px: float       # the same median, in the hand sheet's own pixels
    bias_pct_fs: float     # signed median (hand - automated), % of full scale
    best_altitude: str     # the automated altitude these hand points sit closest to

def crosscheck(max_gap: float = 0.01, csv_dir: Path = AUTO_DIR) -> list:
    """Every hand-placed point, against the automated trace at the same Mach.

    A point is compared only where the automated trace has a sample within
    `max_gap` Mach on BOTH sides. Its README records short gaps where curves
    cross; interpolating across one would compare the hand reading against a
    straight line the automated trace never drew.
    """
    hand, auto = curves(), automated_curves(csv_dir)

    def residual(c: Curve, am: np.ndarray, av: np.ndarray) -> np.ndarray:
        i = np.clip(np.searchsorted(am, c.mach), 1, len(am) - 1)
        inside = (c.mach >= am[0]) & (c.mach <= am[-1])
        near = (c.mach - am[i - 1] <= max_gap) & (am[i] - c.mach <= max_gap)
        ok = inside & near
        return c.value[ok] - np.interp(c.mach[ok], am, av)

    rows = []
    for (q, alt), c in sorted(hand.items()):
        if (q, alt) not in auto:
            continue
        r = residual(c, *auto[(q, alt)])
        if r.size == 0:
            continue
        nearest = {}
        for a in AUTO_ALT:
            if (q, a) in auto:
                ra = residual(c, *auto[(q, a)])
                if ra.size:
                    nearest[a] = float(np.median(np.abs(ra)))
        fs = c.value_span
        rows.append(CrossCheck(
            q, alt, int(r.size),
            100.0 * float(np.median(np.abs(r))) / fs,
            100.0 * float(np.max(np.abs(r))) / fs,
            float(np.median(np.abs(r))) / c.value_per_px,
            100.0 * float(np.median(r)) / fs,
            min(nearest, key=nearest.get)))
    return rows
```

- [ ] **Step 5: Write the script** — `scripts/cr2144_digitisation_crosscheck.py`:

```python
"""CR-2144 pp. 220-222 read twice, compared: session 30's hand reading against
the automated trace found untracked in session 32.

Run from the repository root:
    .venv/Scripts/python.exe scripts/cr2144_digitisation_crosscheck.py

Source: NASA CR-2144, printed pp. 220-222 (PDF pp. 225-227). Reads neither the
PDF nor the scans; both inputs are tracked derived data --
atisim/data/cr2144_p220_222_digitised.csv and Reference_papers/CR-2144/csv/.
"""
from atisim import cr2144_mach

# Reference_papers/CR-2144/README.md, "Per-panel confidence"
CONFIDENCE = {"cl_alpha": "good", "cd_alpha": "good", "cm_alpha": "good",
              "cm_m": "good", "cd_m": "fair", "cl_m": "poor",
              "cm_alpha_dot": "poor"}

def main():
    print(f"{'quantity':14}{'alt':>5}{'conf':>6}{'n':>5}{'med %FS':>9}"
          f"{'max %FS':>9}{'med px':>8}{'bias %FS':>10}  sits on")
    for r in cr2144_mach.crosscheck():
        swap = "" if r.best_altitude == r.altitude else "   <-- altitude?"
        print(f"{r.quantity:14}{r.altitude:>5}{CONFIDENCE[r.quantity]:>6}{r.n:>5}"
              f"{r.median_pct_fs:>9.2f}{r.max_pct_fs:>9.2f}{r.median_px:>8.1f}"
              f"{r.bias_pct_fs:>+10.2f}  {r.best_altitude}{swap}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the measurement**

```bash
../../../.venv/Scripts/python.exe scripts/cr2144_digitisation_crosscheck.py
../../../.venv/Scripts/python.exe -m pytest atisim/tests/test_cr2144_crosscheck.py -q
```

Expected: a 20-row table, then the tests. **Either outcome is a result.** If a `GOOD` band fails, do **not** change `BAND_PCT_FS`; mark that case `xfail(strict=True)` citing the §4 entry you write in Step 7.

- [ ] **Step 7: Record and settle.** Write a §4 entry with the full table, the good/fair/poor split, any altitude swap on a poor panel, and what it means for session 30's quoted reading error. Settle the prediction — **`status` and `outcome` only**:

Set `status="SETTLED"`. Set `outcome` to a sentence that begins `RIGHT:` or `WRONG:` and quotes the twelve measured medians for the good panels from the Step 6 table — for example `"WRONG: cl_alpha 40K reads 2.6% FS; the other eleven lie between 0.4% and 1.7%"`. **Never edit `claim`, `falsified_if` or `reasoning`** — `test_predictions.py` will fail if you do, which is the point.

```bash
../../../.venv/Scripts/python.exe -m pytest atisim/tests/test_predictions.py atisim/tests/test_cr2144_crosscheck.py -q
git add atisim/ scripts/cr2144_digitisation_crosscheck.py docs/PROJECT.md
git commit -q -m "Price the CR-2144 reading from outside it: two digitisations, compared"
```

### Task 8: 14 CFR 25.341 — is it the independent check §7 hopes for?

**Why:** §7 warns that MIL-F-8785C Fig. 7's high-altitude end may descend from the same U-2 data as HICAT, so it may not be an independent check. The retrieval ledger recovered in session 32 names 14 CFR 25.341 as free and fully open. **Whether it is commensurable with Fig. 7 is the first question, and "no" is a legitimate answer.**

**Timebox: 2 hours.**

- [ ] **Step 1:** Read the current text at **https://www.ecfr.gov/current/title-14/section-25.341**, and the section's history from the "Timeline" link on the same page. Record the **amendment number in force**, and the text of (a) discrete gust and (b) continuous turbulence.

- [ ] **Step 2: Decide commensurability before comparing anything.** Fig. 7 plots **σ_w, an RMS vertical gust in m/s TAS, against altitude at stated exceedance probabilities**. For each quantity 25.341 tabulates, write down: RMS or discrete peak? EAS or TAS? A measured statistic or a design reference value?

- [ ] **Step 3, if NOT commensurable:** Close §7's row as **IMPOSSIBLE WITH SOURCES HELD for this purpose** — "25.341 tabulates a design reference velocity, not a measured σ at a stated exceedance, so it cannot test Fig. 7's lineage" — naming the amendment. Record the older-versus-current discrete-gust discrepancy the ledger flagged, if confirmed. Stop.

- [ ] **Step 3, if commensurable:** convert to σ_w in m/s TAS at 33–41 kft, stating every conversion, and place it against Fig. 7's severe curve at 37 kft (σ_severe = **4.80 ± 0.12 m/s**, §4, session 27). Record as a §4 row.

- [ ] **Step 4:** Update §7's retrieval-ledger row and `Reference_papers/SOURCES.md` (a regulation, public domain — record the URL and amendment; do not store a PDF). Commit.

### Task 9: The three unassessed NASA papers

`SOURCES.md` lists NASA TM-4745 (Burcham 1996), TM-1998-206552 (Burcham) and Taylor 1978 as "not assessed". The two Burcham papers concern propulsion-controlled flight of the **MD-11**, a DC-10 derivative, and propulsion-control work usually carries a linear model — so they are the most likely place in the held set for a longitudinal derivative set close to a DC-10.

**Timebox: 2 hours.** The files are in the main checkout's `Reference_papers/`.

- [ ] **Step 1:** For each paper, search for a tabulated longitudinal model: `CLα`, `Cmα`, `Cmq`, `Cmα̇`, `CLδe`, `Cmδe`, dimensional `Zw`, `Mw`, `Mq`, a state-space A matrix, and mass, inertias, S and c̄.

- [ ] **Step 2:** Classify each: **HAS A SET** (which coefficients, at which condition), **PARTIAL**, or **NONE**.

- [ ] **Step 3:** If any has a set, record it as the best candidate for settling the sealed DC-10 prediction, **with the caveat stated**: the MD-11 has a longer fuselage and a smaller horizontal tail than a DC-10-10, so it is a proxy, not the aircraft. Do not build a registry entry this phase.

- [ ] **Step 4:** Update `SOURCES.md` and §7's DC-10 row, which now becomes final. Commit.

---

## Day 3 — Wed 23

### Task 10: The α̇ branch — settle the rule-3 question (D3)

`claude/engine-validity-presentation-1408e8` modifies `atisim/tests/test_cr2144_modes.py`, one of the five files rule 3 puts off-limits. The rule puts the burden on the change: is it a **re-capture** (the aircraft changed, so the pinned values moved) or a **loosening**? That question has been open since the branch was rescued.

**Timebox: 2 hours for the review.**

- [ ] **Step 1: Look at exactly what changed in the protected file**

```bash
git diff main...claude/engine-validity-presentation-1408e8 -- atisim/tests/test_cr2144_modes.py
```

- [ ] **Step 2: Apply the decision rule.** It is a **re-capture** if and only if (a) **every tolerance is equal or tighter**, and (b) **every re-pinned value moves toward Table IX-5**. It is a **loosening** if any tolerance widens.

- [ ] **Step 3:** Record the verdict in §0's row for the branch, with the diff's tolerance lines quoted.

- [ ] **Step 4, per D3 as recommended:** park it as **FUTURE WORK** with the verdict attached — the next person then inherits a *reviewed* change rather than a suspect one. **Do not merge this phase.** Merging moves the four modes and §1's headline, and would need its own re-baseline.

### Task 11 (only if D2 includes it): why the two engines pick different cores

§8 names the cheapest discriminator: **a still-air run of the same duration from the same state.** Whatever the engines do to each other over ~45 s with no wind is the part that is not the encounter. `jsbsim 1.3.1` is installed on the reference interpreter.

**Timebox: half a day.**

**Files:** Modify `scripts/gen_jsbsim_vortex_reference.py` (`run_case` at line 232, `main` at line 331). Create `scripts/engines_still_air.py`. Create `atisim/tests/data/jsbsim_still_air_reference.xml` (generated).

- [ ] **Step 1: Give `run_case` a still-air switch.** Change its signature and the wind line inside its loop (around line 304):

```python
def run_case(case, model, mach, source, pad_radii=6.0, mehta=False, still_air=False):
```

```python
        w = (np.zeros(3) if still_air else
             rankine_array(north, down, core_north, core_down, r0, v0, cos_dpsi)
             if mehta else
             rankine(north, down, core_north, core_down, r0, v0))
```

Everything else in the loop — the fixed trim elevator, the sampling, the duration — is unchanged, which is the point: same start state, same length, no wind.

- [ ] **Step 2: Give `main` the flags, with a guard that makes clobbering the frozen reference impossible.** Replace `def main():` and its first line, the `runs` construction, the `run_case` call, and the final write:

```python
def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--still-air", action="store_true",
                    help="fly ONLY the Mehta array case, with zero injected wind")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    if args.still_air and args.out.resolve() == OUT.resolve():
        ap.error("--still-air writes its own file; the frozen vortex reference is "
                 "never regenerated by this flag")
    f, vec = ref.f, ref.vec
```

```python
    L.append(f"    <still_air>{str(args.still_air).lower()}</still_air>")
    L.append("  </provenance>")

    if args.still_air:
        runs = [MEHTA_CASE + ("mehta", True)]
    else:
        runs = [(c, m, e, ma, CASE_SOURCE[c], False) for (c, m, e, ma) in CASES]
        runs.append(MEHTA_CASE + ("mehta", True))

    for case, model, entry, mach, source, mehta in runs:
        values, initial, samples, cores = run_case(
            case, model, mach, source, mehta=mehta, still_air=args.still_air)
```

```python
    args.out.write_text("
".join(L) + "
", encoding="utf-8", newline="
")
    print(f"
wrote {args.out.resolve().relative_to(ROOT)}  ({len(L)} lines)")
```

(The existing `<provenance>` block gains one `<still_air>` line; the default run writes `false` there. **Verified while writing this plan:** `jsbsim_vortex_ref.load()` reads elements by name with `find`/`findall`, so an extra provenance element is ignored — and it keys encounters by `(case, radius_source)`, which is why Step 4 asks for `("mehta", "mehta")`.)

- [ ] **Step 3: Generate, and prove the frozen reference did not move**

```bash
"C:/Users/mateusz/AppData/Local/Programs/Python/Python310/python.exe" scripts/gen_jsbsim_vortex_reference.py --still-air --out atisim/tests/data/jsbsim_still_air_reference.xml
git status --porcelain atisim/tests/data/
```

Expected: exactly one line, `?? atisim/tests/data/jsbsim_still_air_reference.xml`. **If `jsbsim_vortex_reference.xml` appears as modified, stop and `git checkout` it.**

- [ ] **Step 4: Fly AtiSim from the same state for the same duration** — `scripts/engines_still_air.py`:

```python
"""The two engines over the Mehta array's full duration with NO wind.

Answers PROJECT.md section 8's open question -- why AtiSim and JSBSim put their
load extremes on different cores of Mehta's five-core array -- by the
discriminator that section names: whatever the engines do to each other in
still air, from the same state, over the same time, is the part of the
disagreement that is not the encounter.

Reads atisim/tests/data/jsbsim_still_air_reference.xml, written by
    scripts/gen_jsbsim_vortex_reference.py --still-air --out atisim/tests/data/jsbsim_still_air_reference.xml
Run from the repository root, with an ABSOLUTE PYTHONPATH from a worktree.
"""
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import atisim  # noqa: F401  -- enables x64
from atisim import jsbsim_vortex_ref, vortex_viz
from atisim.aircraft import REGISTRY
from atisim.units import RAD2DEG

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vortex_compare import partial_field_model  # noqa: E402
from vortex_diagnose import _state_and_controls  # noqa: E402

STILL = Path(atisim.__file__).parent / "tests" / "data" / "jsbsim_still_air_reference.xml"

def main():
    enc = jsbsim_vortex_ref.load(STILL).encounters[("mehta", "mehta")]  # keyed (case, radius_source)
    state, controls = _state_and_controls(enc, enc.values["matched_altitude"])
    zero = lambda p: jnp.zeros(3)  # noqa: E731
    at = vortex_viz.fly_from_state(
        REGISTRY[enc.aircraft], zero, state, controls, label="still",
        seconds=enc.values["duration"], dt=0.01, window=(-1e12, 1e12),
        window_name="all",
        wind_model=partial_field_model(zero, omega_gust=False, alphadot=False))

    js_t = np.array([s.t for s in enc.samples])
    js_nz = np.array([s.Nz for s in enc.samples])
    js_th = np.array([s.theta for s in enc.samples])
    at_nz = np.interp(js_t, np.asarray(at.t), np.asarray(at.n_z))
    at_th = np.interp(js_t, np.asarray(at.t), np.asarray(at.theta))

    print(f"still air, {enc.values['duration']:.1f} s from JSBSim's trimmed state")
    print(f"{'t s':>6}{'JSBSim n_z':>12}{'AtiSim n_z':>12}{'d n_z':>9}{'d theta deg':>13}")
    for k in np.linspace(0, len(js_t) - 1, 12).astype(int):
        print(f"{js_t[k]:6.1f}{js_nz[k]:12.4f}{at_nz[k]:12.4f}"
              f"{at_nz[k] - js_nz[k]:+9.4f}{(at_th[k] - js_th[k]) * RAD2DEG:+13.4f}")
    print(f"
peak-to-peak n_z, still air:  JSBSim {np.ptp(js_nz):.4f} g   "
          f"AtiSim {np.ptp(at_nz):.4f} g")
    print(f"max |d theta|: {np.max(np.abs(at_th - js_th)) * RAD2DEG:.4f} deg")

if __name__ == "__main__":
    main()
```

```bash
PYTHONPATH="C:/Users/mateusz/UROP/Claude_Flight_Sim/.claude/worktrees/new-session-943052" ../../../.venv/Scripts/python.exe scripts/engines_still_air.py
../../../.venv/Scripts/python.exe -m pytest atisim/tests/test_jsbsim_vortex.py -q
```

Expected: a 12-row table and two summary lines; `test_jsbsim_vortex.py` still passes (the frozen reference is untouched, and the loader tolerates the new provenance line).

- [ ] **Step 5: Decision rule.** Compare `n_z` and θ. If still-air divergence reproduces **at least half** of the −10.1% span disagreement, the answer is the run length, not the encounter — record it and close §8. Otherwise record which candidates are eliminated and park the rest as FUTURE WORK.

### Task 12: Close-out and the phase gate

- [ ] **Step 1: Everything unfinished at the hard stop becomes a FUTURE WORK row**, with its route and the script that would run it. Nothing is left in the state "open".

- [ ] **Step 2: §0 should now hold only** `wgs84-earth` and, per D3, the α̇ branch — each with its address and why it is parked. That near-empty §0 is the release's "finished" signal.

- [ ] **Step 3: Full suite, against the right tree**

```bash
../../../.venv/Scripts/python.exe -c "import atisim; print(atisim.__file__)"
../../../.venv/Scripts/python.exe -m pytest -q
```

Expected: **918 + the Task 7 tests passed, 0 failed.** Record the number in §10, re-measured rather than incremented.

- [ ] **Step 4:** §9 entry — what was done, what was NOT done and why. Mark the PR ready, merge.

**Phase gate:** the §5 status table has 20 rows; every §7 and §8 item has a status from the four; no SEALED prediction sits past its trigger un-adjudicated; the suite is green; `git grep "A1 and A2 stay"` finds nothing in `docs/`.

---

## Deliberately NOT in this phase

- **No change to any aircraft or to §1's headline.** Tasks 7–9 measure data and sources, not the model. The α̇ merge is parked for exactly this reason.
- **No merge of `wgs84-earth`.** Phase 2 decided that.
- **No tolerance widened.** A failing band is a finding (Task 7, Step 6).
- **No edit to a sealed prediction** beyond `status` and `outcome` on the one this phase seals.
- **The documentation site and notebook** — Phases 4 and 5, from Thu 24.
