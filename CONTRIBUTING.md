# Contributing to AtiSim

Thank you for your interest in AtiSim. Bug reports, questions and pull requests are all welcome
through [GitHub issues](https://github.com/MatusGib/AtiSim/issues).

## Setting up

```bash
git clone https://github.com/MatusGib/AtiSim.git
cd AtiSim
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev,ui,docs]"
python -m pytest -q
```

An editable install points `atisim` at the directory it was installed from. If you work in more
than one checkout, confirm you are testing the code you edited:

```bash
python -c "import atisim; print(atisim.__file__)"
```

## Conventions

AtiSim is only useful if its numbers can be traced, so a few conventions matter more than they
would in most projects.

1. **Every constant has a source.** Aircraft data cites the document, table and page it came
   from. `atisim/provenance.py` classes each constant as sourced, derived, calibrated or
   declared. If a source does not supply a value, declare it as a modelling choice, state its
   sensitivity, and do not invent a plausible default.
2. **Tests check bands and orderings, not exact values.** Write "the vortex produces a larger
   load than the updraft", not "the load is 1.63 g".
3. **Do not loosen a validated tolerance to make a test pass.** The baseline tests
   (`test_conservation.py`, `test_cr2144_modes.py`, `test_drag_polar.py`, `test_navion.py`,
   `test_trim.py`) guard the core. If one moves, something real has changed. If the model changed
   legitimately, re-pin the value and explain why in a comment beside the change.
4. **Wind enters only through the air-relative velocity.** Aerodynamic code takes `vel_rel` and
   `omega_rel` and never sees inertial velocity. Read the wind-model contract in the user guide
   before you add a field.
5. **Reference documents stay out of the repository.** A script that reads a third-party
   document takes its path as an argument (`--pdf`), and names the document, edition and page in
   its docstring. `docs/references.md` lists the sources.

## Checks before a pull request

```bash
python -m pytest -q                                  # the test suite
python -m pytest --nbval-lax notebooks/              # the executed notebooks
python -m sphinx -b html -W docs docs/_build/html    # the docs, with warnings as errors
```

CI runs all three on every pull request. In the pull request, describe what changed and what you
measured, and say what you deliberately left alone.

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md).
