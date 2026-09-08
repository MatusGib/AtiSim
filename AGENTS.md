# AtiSim — working rules

`docs/PROJECT.md` is the standing record. This file is the short list of rules that
protect it. Read both before doing anything.

---

## 1. NOTHING IS DONE UNTIL IT IS IN `docs/PROJECT.md`

**A measurement that exists only as a script, a log, a PNG or a chat message has not been
made.** It will be re-done by a later session that has no way to know it was already
answered, and the second attempt will not know what the first one concluded.

This is not hypothetical. **Session 27 found the two highest-value digitisations in the
plan — TM-102186 Fig. 6 and MIL-F-8785C Fig. 7 — already complete in another worktree,
untracked, with their outputs gitignored, while that same worktree's `PROJECT.md` still
listed both as open acquisitions.** Three further scripts and a run of LES comparisons were
stranded the same way. The measuring was never the hard part.

**So, before a session ends — every session, including ones that "only investigated":**

1. **Write the result into `docs/PROJECT.md`.** §4 if it is a number measured against
   something; §5 if a source failed to supply something; §6 if it is a bug that no test
   catches; §9 always — add a session-log entry at the top.
2. **`git add` every script you wrote.** An untracked script is a deleted script. Check
   `git status` for `??` lines before you finish, and either commit them or say in §9 why
   not.
3. **Say what you did NOT do**, and why, so the next session does not go looking. Negative
   results and abandoned approaches are findings and belong in the record.
4. **A result that changes an earlier claim must edit that claim**, not merely sit beside
   it. Supersede in place — never delete a row (see §4's own rule) — and fix any *other*
   section that repeats the number. §1 carried a stale "67%" for several sessions because
   §4 was updated and §1 was not.

**If a session produces no durable record, it produced nothing.**

## 2. Flag, never invent

Every number carries the table it came from. A parameter a source does not supply is named
as a **declared modelling choice**, not given a plausible default. Use the four provenance
categories in `atisim/provenance.py` — SOURCED, DERIVED, CALIBRATED, DECLARED — and mean
them. **A number without a citation added to `PROJECT.md` has broken the project.**

Where no run backs a claim, say so in those words. Do not estimate, extrapolate, or write a
placeholder that reads like a measurement.

## 3. Do not edit a tolerance to make a test pass

§4's "validated baseline" files are off-limits to feature work: `test_conservation.py`,
`test_cr2144_modes.py`, `test_drag_polar.py`, `test_navion.py`, `test_trim.py`. **If one of
them moves, something real broke** — find out what before touching the number.

The same applies to `atisim/predictions.py`: a SEALED entry is never edited. Settling one
touches `status` and `outcome` **and nothing else** — those are the only two fields outside
the digest.

Correcting a test because *the world changed* is different from loosening one because it
failed, but the burden is on you to show which it is, **in a comment, at the change**.

## 4. Check that the tree you are testing is the tree you edited

Worktrees share the main checkout's `.venv`. Before believing any run:

```
.venv/Scripts/python.exe -c "import atisim; print(atisim.__file__)"
```

If that path is not the tree you edited, everything downstream is a measurement of someone
else's code — and it will look exactly like a real one. Run scripts with `PYTHONPATH` set
to the **absolute** worktree root. `PROJECT.md` §10 has the four-row table of what resolves
where.

## 5. Reference documents live outside git

`refs/` and `Reference_papers/` hold the source PDFs; `refs/` and `runs/` are gitignored.
A script that reads one must take a `--pdf`-style path argument so it can run from a
worktree, and must name the document, edition and page in its own docstring.

## 6. Assert bands and orderings, not values

The validation claim in `PROJECT.md` §1 is deliberately narrow: **comparative and
mechanistic**, not absolute. Do not write a test or a claim of the form *"the load will be
X g"*. Quote the excursion and the envelope with every result, and keep results from
out-of-scope aircraft (the Cessna) out of conclusions entirely.
