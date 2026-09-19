# Getting started

## Install

AtiSim needs Python 3.10 or later. The runtime has four dependencies — `jax`, `numpy`, `scipy`
and `matplotlib`.

```bash
git clone https://github.com/MatusGib/AtiSim.git
cd AtiSim
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e .
```

On Linux and macOS the interpreter is `.venv/bin/python`. Optional extras:

| Extra | For |
|---|---|
| `.[dev]` | the test suite and the executed notebooks |
| `.[ui]` | the Dash analysis app |
| `.[ref]` | regenerating the JSBSim comparison data — the suite itself never needs JSBSim |
| `.[docs]` | building this site |

## Check which code you are running

An editable install maps `atisim` to the directory it was installed from. If you work in more
than one checkout, a run can import a *different* tree from the one you edited — and nothing
warns you: the tests pass, against code you did not change. Before trusting any result:

```bash
.venv/Scripts/python.exe -c "import atisim; print(atisim.__file__)"
```

If that path is not the tree you edited, set `PYTHONPATH` to the absolute path of the tree you
mean. {doc}`running` has the full table of which launch method resolves where.

## Run the tests

```bash
.venv/Scripts/python.exe -m pytest -q
```

The suite asserts **bands and orderings, not exact values**, for the reason {doc}`validation`
gives. The count and runtime of the last full run are recorded in {doc}`running`.

## Before you believe the model: the validation ladder

```bash
.venv/Scripts/python.exe -m pip install -e .[dev]
.venv/Scripts/python.exe -m jupyter lab notebooks/validation-ladder.ipynb
```

One notebook re-runs the evidence behind the validation claim, in four rungs, each resting on
the one below: answers you can work out by hand; the model's own source data and an independent
engine fed the same coefficients; a recorded encounter; and the published orderings with the
mechanism behind them. Every number is computed as it runs, and each rung asserts what the
claim says and no more. `notebooks/solver-validation.ipynb` does the same for the solver.

The first rung is also a script:

```bash
.venv/Scripts/python.exe scripts/sanity.py
```

Eleven checks from degenerate inputs upward. Zero the wind: does it fly straight? Zero a
coefficient so a motion becomes physically impossible: does the motion stop? Then signs, then
numbers derived by hand in the script and printed beside the model's answer, so they can be
read rather than trusted.

## Fly a real encounter

```bash
.venv/Scripts/python.exe scripts/vortex.py --case hannibal --png runs/v.png
```

This flies a Boeing 747 through the vortex array Mehta (1987) identified from a DC-10's flight
recorder near Hannibal, Missouri, on 3 April 1981, alongside an updraft and an elevator
manoeuvre, and draws the analysis figure. How the result compares with the recorded load is in
{doc}`validation`.

To fly it by hand, with a cockpit display and the autopilot a key press away:

```bash
.venv/Scripts/python.exe scripts/fly.py --wind hannibal
```

The arrow keys are the stick — up is stick *forward*, so it pitches the nose down — `,` and `.`
the rudder, `-` and `=` the throttle, `[` and `]` the trim, and `a` hands control to the
autopilot.
