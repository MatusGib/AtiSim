# Contributing to AtiSim

AtiSim's value is less its code than its record: every number carries its source, every result is
written down with the tolerance it was measured to, and nothing inconvenient is deleted. A change
that improves the code but breaks the record makes the project worse. This page is the short
version of the rules; [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) has the full text and, for each
rule, the failure that made it necessary.

## Setting up

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e .[dev,docs]
.venv/Scripts/python.exe -m pytest -q
```

**Check which tree you are testing.** An editable install maps `atisim` to the directory it was
installed from, so in a second checkout or worktree the tests can pass against code you did not
edit — with no warning. Before believing any run:

```bash
.venv/Scripts/python.exe -c "import atisim; print(atisim.__file__)"
```

If the path is not the tree you edited, set `PYTHONPATH` to its absolute path.

## The rules

1. **A result is not finished until it is in [`docs/PROJECT.md`](docs/PROJECT.md).** A number that
   lives only in a script's output, a log or a pull request will be re-measured by someone who
   cannot know it was done. Record measurements in §4, gaps a source cannot fill in §5, bugs no
   test catches in §6, and your work in §9. Commit every script you wrote.
2. **Unfinished work goes in §0, with its address** — the branch name and what blocks it. An
   unmerged branch nobody wrote down is indistinguishable from one nobody wrote.
3. **Flag, never invent.** Every constant carries one of four provenance categories in
   `atisim/provenance.py` — sourced, derived, calibrated, declared — and means it. A value a source
   does not supply is a *declared modelling choice*, named as such, never a plausible default.
4. **Never edit a tolerance to make a test pass.** The validated baselines —
   `test_conservation.py`, `test_cr2144_modes.py`, `test_drag_polar.py`, `test_navion.py`,
   `test_trim.py` — are off-limits to feature work. If one moves, something real broke. Re-pinning
   one because the model legitimately changed is allowed, but the argument goes in a comment at
   the change.
5. **Supersede, never delete.** A result that changes an earlier one strikes it through and
   records the new one beside it — and fixes every other section that repeats the old number.
6. **Assert bands and orderings, not values.** The validation claim is comparative, so the tests
   are too: "the vortex produces a larger load than the updraft", not "the load is 1.63 g".
7. **Seal a prediction before the run that decides it.** A new capability that makes a claim about
   a case not yet run adds an entry to `atisim/predictions.py` first. A sealed entry is never
   edited; settling it changes only `status` and `outcome`, and a wrong prediction stays wrong.

## Reference documents

Publisher-held papers are not in the repository. Scripts that read a document take its path as an
argument (`--pdf`, `--dataset`), and name the document, edition and page in their own docstring.
[`Reference_papers/SOURCES.md`](Reference_papers/SOURCES.md) lists every source with its licence
basis and where to obtain it.

## Documentation

```bash
.venv/Scripts/python.exe -m sphinx -b html docs docs/_build/html
```

The site includes sections of `docs/PROJECT.md` verbatim rather than restating them, so update the
record and the site follows. The build should finish with no warnings.

## Pull requests

Describe what was measured and what changed, and say what you did **not** do and why — negative
results and abandoned approaches are findings too. Run the full suite and state its result.
