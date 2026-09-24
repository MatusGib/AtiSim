# Getting started

## Install

AtiSim needs Python 3.10 or later.

```bash
git clone https://github.com/MatusGib/AtiSim.git
cd AtiSim
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -e .
```

The runtime has four dependencies: `jax`, `numpy`, `scipy` and `matplotlib`. The optional
extras are:

| Extra | Installs | For |
|---|---|---|
| `.[dev]` | pytest, Jupyter, nbval | the test suite and the executed notebooks |
| `.[ui]` | Dash, Plotly, PyArrow | run artifacts, the analysis app, and the tests that cover them |
| `.[docs]` | Sphinx, MyST, Furo | building this site |
| `.[ref]` | JSBSim | regenerating the cross-code reference data only |

Importing `atisim` turns on float64 in JAX. Because that setting only takes effect before the
first array is created, import `atisim` before you create any JAX array.

## Check the installation

```bash
python scripts/sanity.py
```

This runs eleven checks, starting from degenerate inputs. With zero wind, does the aircraft
fly straight? With a coefficient zeroed so that a motion is physically impossible, does that
motion stop? Next it checks signs, and then numbers derived by hand in the script, which it
prints beside the model's answer.

If you work in more than one checkout, confirm that Python imports the tree you expect:

```bash
python -c "import atisim; print(atisim.__file__)"
```

## Run the tests

```bash
python -m pip install -e ".[dev,ui]"
python -m pytest -q
```

The suite checks bands and orderings rather than exact values, for the reasons given in
{doc}`validation`. A full run takes about half an hour. Install `ui` as well as `dev`: without
PyArrow and Plotly, the artifact and figure tests skip themselves.

The two notebooks are executed as a second gate:

```bash
python -m pytest --nbval-lax notebooks/
```

## Walk the validation ladder

```bash
python -m jupyter lab notebooks/validation-ladder.ipynb
```

This notebook re-runs the evidence behind the validation claim in four rungs, each resting on
the one below:

1. answers you can work out by hand
2. the model's own source data, and an independent engine fed the same coefficients
3. a recorded encounter
4. the published orderings, and the mechanism behind them

Every number is computed as the notebook runs. `notebooks/solver-validation.ipynb` does the same
for the solver.

## Fly a real encounter

```bash
python scripts/vortex.py --case hannibal --png runs/hannibal.png
```

This flies a Boeing 747 through the vortex array that Mehta (1987) identified from a DC-10's
flight recorder near Hannibal, Missouri, on 3 April 1981. It also flies an updraft and an
elevator manoeuvre, and then draws the analysis figure.

To fly the encounter by hand, with a cockpit display:

```bash
python scripts/fly.py --wind hannibal
```

| Key | Action |
|---|---|
| arrow keys | stick (up is stick *forward*, which pitches the nose down) |
| `,` `.` | rudder |
| `-` `=` | throttle |
| `[` `]` | pitch trim |
| `a` | hand control to the autopilot |

Next, {doc}`user-guide` shows how to do the same things from Python.
