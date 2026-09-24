# Development Manual

This manual is for persons who change or extend AtiSim. It gives the structure of the code, the
rules for changes, and the procedures for tests, documentation and releases. For the use of
AtiSim, refer to {doc}`user-manual`. For the equations, refer to {doc}`physics-and-assumptions`.

## 1 Principles

AtiSim is only as good as the traceability of its numbers. These principles apply to each change:

1. Give a source for each number. If no source gives a value, declare the value as a modeling
   choice, and give its sensitivity.
2. Let the wind change only the air-relative velocity and rates. The aerodynamic code must not
   use the inertial velocity.
3. Write tests that check ranges and orders, not exact values.
4. Do not make a validated tolerance larger to make a test pass.
5. Do not put third-party documents in the repository.

Section 5 gives these rules in full.

## 2 Prepare the development environment

1. Get a copy of the source code:

   ```bash
   git clone https://github.com/MatusGib/AtiSim.git
   cd AtiSim
   ```

2. Make a virtual environment, and start it:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Install AtiSim with all the development parts:

   ```bash
   python -m pip install -e ".[dev,ui,docs]"
   ```

4. Make sure that Python uses this copy of the code:

   ```bash
   python -c "import atisim; print(atisim.__file__)"
   ```

5. Run the tests (section 7).

An editable installation points to the directory that you installed from. If you have two
copies of the repository, the tests can use the other copy. Step 4 shows the copy that Python
uses.

## 3 Repository layout

| Path | Content |
|---|---|
| `atisim/` | the Python package |
| `atisim/data/` | digitized source data that the package reads |
| `atisim/tests/` | the test suite |
| `atisim/tests/data/` | the frozen JSBSim reference data |
| `atisim/analysis/` | run files, time series and figures. Needs the `ui` part. |
| `atisim/apps/` | the Dash analysis application. Needs the `ui` part. |
| `scripts/` | the command-line scripts |
| `notebooks/` | the validation notebooks |
| `docs/` | this documentation |
| `.github/workflows/` | the continuous integration |

## 4 Architecture

### 4.1 Data flow

This table shows the sequence of the modules in one step of a run:

| Sequence | Function | What it does |
|---|---|---|
| 1 | `integrate.step` | It calls the wind model. The wind model gives the wind and the gust rates at the aircraft. |
| 2 | `dynamics.derivatives` | It calculates the air-relative velocity and rates. |
| 3 | `aero.coefficients`, `aero.aero_forces_moments` | They calculate the aerodynamic forces and moments. |
| 4 | `aero.thrust_force`, `dynamics.gravity` | They add the thrust and the gravity. |
| 5 | `dynamics.derivatives` | It gives the time derivative of the state. |
| 6 | `integrate.step` | It does one RK4 step, and sets the quaternion norm to 1. |

`integrate.rollout` repeats the step in `jax.lax.scan`. `vortex_viz` and `checks` measure the
result.

### 4.2 Modules

| Module | Function |
|---|---|
| `state` | `State` and `Controls`, and the quaternion functions |
| `units` | conversion factors. Do not write a conversion factor in a different module. |
| `atmosphere` | the International Standard Atmosphere |
| `aircraft` | the aircraft data, `REGISTRY` and `CRUISE` |
| `airframe` | the tail arm, the sample stations and the span loading |
| `aero` | the aerodynamic coefficients, forces and moments, and the thrust |
| `dynamics` | the equations of motion, gravity, load factor and F-factor |
| `wind` | the wind fields, the gust rates, `field_model` and `superpose` |
| `loads` | the strip-integrated loads |
| `integrate` | the RK4 step and the rollouts |
| `trim` | the Newton trim for steady level flight |
| `sensors` | the air data that a real sensor measures |
| `autopilot`, `manual` | the cascaded PID autopilot, and the manual controls |
| `panel`, `viz` | the cockpit display, and the post-flight figure |
| `vortex_viz` | the encounters and their measurements |
| `response` | the spectra and the exceedance rates of a run |
| `verification` | checks against exact mathematics |
| `validation` | the linearization, the modes and the reference values |
| `checks` | the checks of one run |
| `sensitivity` | the derivatives of results with respect to coefficients |
| `cr2144_mach`, `jsbsim_ref`, `jsbsim_vortex_ref` | the source data and the reference data for validation |
| `provenance` | the category and source of each constant |
| `predictions` | the predictions before their result |

### 4.3 Interfaces

- `Aircraft` is a `NamedTuple` of JAX arrays. A function can take it as an argument inside `jit`.
- `State` holds the position, the velocity, the quaternion and the rates (section 2.3 of {doc}`physics-and-assumptions`).
- The wind model has the interface `wind_model(wind_state, state, key, dt) -> (wind_ned, omega_gust, wind_state, key)`. A deterministic model returns the key with no change.
- The load model takes a `State` and gives a `loads.CoeffIncrement`. The integrator samples it one time for each step.

### 4.4 Optional dependencies

The runtime needs only `jax`, `numpy`, `scipy` and `matplotlib`. Keep these limits:

- Only `atisim.analysis` and `atisim.apps` can import `pyarrow`, `plotly` or `dash`.
- Only `atisim.apps` can import `dash`.
- No module in `atisim` can import `jsbsim`. Only the `gen_jsbsim_*` scripts import it.

The tests read the frozen reference data in `atisim/tests/data/`. Thus the tests do not need
JSBSim.

## 5 Rules for changes

### 5.1 Sources of numbers

1. Give each coefficient a comment with the document, the table and the page.
2. Put the category of each module-level constant in `atisim/provenance.py`:

   | Category | Meaning |
   |---|---|
   | SOURCED | read directly from a table in a cited document |
   | DERIVED | calculated from sourced values with an exact relation |
   | CALIBRATED | adjusted so that the model agrees with a sourced value |
   | DECLARED | a modeling choice. Give the sensitivity range. |

3. If a source does not give a value, do not use a typical value. Declare the value.

If a new module-level constant in `aero`, `airframe`, `atmosphere`, `trim` or `wind` has no
entry in the ledger, a test fails.

### 5.2 Wind and aerodynamics

1. Give only `vel_rel` and `omega_rel` to the aerodynamic functions.
2. Keep the inertial velocity and rates in the Coriolis, gyroscopic and kinematic terms.
3. Do not add a `-m dW/dt` term.

{doc}`physics-and-assumptions`, section 7.1, gives the reasons.

### 5.3 Tests and tolerances

1. Write assertions about ranges and orders. For example, write "the vortex gives a larger load
   than the updraft". Do not write "the load is 1.63 g".
2. Do not make a tolerance larger in these baseline files: `test_conservation.py`,
   `test_cr2144_modes.py`, `test_drag_polar.py`, `test_navion.py` and `test_trim.py`.
3. If the model changes correctly and a baseline value moves, set the new value. Put the reason
   in a comment beside the value.
4. Give each new run check a negative control: an input that makes the check fail.

### 5.4 Reference documents

1. Do not add a third-party document to the repository.
2. Give a script that reads a document a `--pdf` argument for the path.
3. Put the document, the edition and the page in the docstring of the script.
4. Add each new source to {doc}`references`.

## 6 Procedures

### 6.1 Add an aircraft

1. Write a function in `atisim/aircraft.py` that returns an `Aircraft`.
2. Give each coefficient a comment with its source.
3. If the source gives dimensional derivatives, use `FlightCondition` and the
   `from_dimensional_*` functions.
4. Set `valid_mach` and `valid_altitude` if you have a measured validation range.
5. Add the aircraft to `REGISTRY`, and add a reference condition to `CRUISE`.
6. Add tests that compare the trim and the modes with the source.
7. Add the aircraft to the table in {doc}`user-manual`, section 3.2.

### 6.2 Add a wind field

1. Write a function in `atisim/wind.py` that takes a position in NED and gives the wind in NED.
2. Use only `jax.numpy`, so that `jax.jacfwd` can calculate the gradient.
3. Put the parameters in a `NamedTuple`, and give each parameter its source.
4. Add a test that compares the field with a value from the source.
5. Add a test that `field_model` gives the correct gust rates.
6. Add the field to {doc}`user-manual`, section 5.3, and to {doc}`physics-and-assumptions`,
   section 7.

### 6.3 Add a run check

1. Write a function in `atisim/checks.py` that returns a `Check`.
2. Set its `kind`: `gate`, `tripwire` or `report`.
3. Add the check to `run_checks`.
4. In `atisim/tests/test_checks.py`, add a test with a negative control.

A `tripwire` is a check that has no known failure. The user interface shows its value and its kind,
and not a pass mark.

### 6.4 Add a script

1. Put the script in `scripts/`.
2. Write a docstring that says what the script measures, and the source that it compares with.
3. Use `argparse`, with `description=__doc__`.
4. Give the script a `--png` or `--outdir` option to write its figures.
5. Add the script to {doc}`user-manual`, section 5.1.

## 7 Tests

### 7.1 Run the tests

1. Run the test suite:

   ```bash
   python -m pytest -q
   ```

2. Run the notebooks:

   ```bash
   python -m pytest --nbval-lax notebooks/
   ```

The full suite takes about 30 minutes. Without the `ui` part, the tests for run files and
figures show as skipped.

### 7.2 Types of test

| Type | Examples | What a failure means |
|---|---|---|
| Verification | `test_verification.py`, `test_conservation.py` | The code does not solve the equations correctly. |
| Validation | `test_cr2144_modes.py`, `test_jsbsim_737.py`, `test_cat_validation.py` | The model does not agree with its source data. |
| Run checks | `test_checks.py` | A check does not find the defect that it must find. |
| Ledger | `test_provenance.py`, `test_audit_regression.py` | A constant has no source, or the ledger is not consistent. |
| Predictions | `test_predictions.py` | A sealed prediction changed. |

### 7.3 Settings of the test suite

- `conftest.py` sets `jax_debug_nans`. Thus a NaN stops the test at the operation that made it.
- Some tests compare results bit by bit. These results can change with the CPU and with the
  version of JAX. The continuous integration sets `OPENBLAS_CORETYPE=Haswell` to make the
  results the same on all runners.

## 8 Documentation

### 8.1 Build the documentation

1. Build the documentation:

   ```bash
   python -m sphinx -b html -W docs docs/_build/html
   ```

2. Open `docs/_build/html/index.html`.

The option `-W` stops the build at the first warning. The continuous integration uses the same
option.

### 8.2 Structure

The documentation has the structure of the JSBSim Reference Manual:

| Document | Content |
|---|---|
| `README.md` | the installation, the first run, and the location of the manuals |
| User Manual | the concepts, the procedures, the reference data and the troubleshooting (ISO/IEC/IEEE 26514) |
| Physics and Assumptions | the model, the assumptions, the limits and the validation (NASA-STD-7009) |
| Development Manual | this manual |
| API Reference | made from the docstrings |

### 8.3 Writing rules

Write the documentation in ASD-STE100 Simplified Technical English. These are the main rules:

1. Write one instruction in each sentence. Start the instruction with a verb.
2. Write not more than 20 words in an instruction, and not more than 25 words in a description.
3. Write not more than 6 sentences in a paragraph.
4. Use the active voice.
5. Use the simple tenses. Do not use the "-ing" form of a verb.
6. Do not use more than 3 nouns together.
7. Use one word for one meaning. Use the same name for the same item.
8. Do not use semicolons, contractions or phrasal verbs.
9. Use American spelling.
10. Use a numbered list for a procedure and a bullet list for other lists.

A tool such as `ste-cli` can find most breaks of these rules. Code, commands and equations are
not text, and the rules do not apply to them.

### 8.4 Docstrings

The API Reference shows the docstrings of the modules, classes and functions. In a docstring,
say what the code does, what it assumes, and where its numbers come from. Do not describe the
history of the code in a docstring. Use a comment for that.

## 9 Continuous integration

| Workflow | When | What it does |
|---|---|---|
| `tests.yml` | each pull request, and each push to `main` | runs the test suite and the notebooks on Linux |
| `docs.yml` | each push and each pull request | builds the documentation with `-W`, and publishes it from `main` |

## 10 Release procedure

1. Make sure that the tests and the documentation build pass on `main`.
2. In `CHANGELOG.md`, move the items under `[Unreleased]` to a new version heading, with the date.
3. Change `version` in `pyproject.toml`.
4. Merge the change into `main`.
5. Make a tag `vX.Y.Z` on the merge commit, and push the tag.

AtiSim uses semantic versioning. Increase the major number for a change that breaks the
interface. Increase the minor number for a new function. Increase the patch number for a
correction.
