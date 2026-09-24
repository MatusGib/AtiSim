# AtiSim

[![tests](https://github.com/MatusGib/AtiSim/actions/workflows/tests.yml/badge.svg)](https://github.com/MatusGib/AtiSim/actions/workflows/tests.yml)
[![docs](https://github.com/MatusGib/AtiSim/actions/workflows/docs.yml/badge.svg)](https://matusgib.github.io/AtiSim/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**AtiSim is a six-degree-of-freedom fixed-wing flight dynamics model written in JAX, built to
study how aircraft respond to clear-air turbulence.** It has been validated by rebuilding real
turbulence encounters from NASA flight records and flying the model through them.

![The AtiSim cockpit panel](docs/images/panel.png)

## Features

- **6-DOF rigid-body dynamics.** Quaternion attitude, fixed-step RK4, `lax.scan` rollouts,
  `jit` and `vmap` over ensembles, and float64 throughout.
- **Air-relative aerodynamics.** Wind enters only through the air-relative velocity and the
  gust rates, so you can fly any wind field without changing the integrator.
- **Wind and turbulence models.** Rankine vortex arrays, including the Hannibal field
  identified from a DC-10's flight recorder, as well as line vortices, updrafts,
  Oseguera–Bowles microbursts, mountain lee waves and Dryden turbulence.
- **Aircraft.** A Boeing 747 (cruise and approach, from NASA CR-2144), a Boeing 737
  linearised from JSBSim for cross-code checks, and two light aircraft. Every coefficient
  cites its source table.
- **Trim, linear modes and sensitivities.** A Newton trim solve, modal analysis about any trim
  point, and derivatives of results with respect to the model's own coefficients.
- **Tools.** A cockpit display you can fly by hand, a cascaded PID autopilot, run artifacts, and
  a Dash app for exploring a run.

## Installation

AtiSim needs Python 3.10 or later.

```bash
git clone https://github.com/MatusGib/AtiSim.git
cd AtiSim
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -e .
```

The runtime depends only on `jax`, `numpy`, `scipy` and `matplotlib`. Optional extras:

| Extra | Installs |
|---|---|
| `.[dev]` | pytest and Jupyter, for the test suite and notebooks |
| `.[ui]` | Dash, Plotly and PyArrow, for run artifacts and the analysis app |
| `.[docs]` | Sphinx, to build the documentation |
| `.[ref]` | JSBSim, only to regenerate the cross-code reference data |

## Quick start

To fly a Boeing 747 through the vortex array NASA identified at Hannibal, Missouri, and read
the load it pulls:

```python
import atisim  # enables float64; import before creating any array
from atisim import vortex_viz

enc, info = vortex_viz.fly_mehta("boeing747", dt=0.01, replayed=True)
n_z = enc.n_z[enc.window]
print(f"peak-to-peak load: {n_z.max() - n_z.min():.2f} g")   # 1.90 g
```

You can also run the same encounter from the command line and draw the analysis figure:

```bash
python scripts/vortex.py --case hannibal --png runs/hannibal.png
```

Or fly it by hand. The arrow keys work the stick, and `a` hands control to the autopilot:

```bash
python scripts/fly.py --wind hannibal
```

The [user guide](https://matusgib.github.io/AtiSim/user-guide.html) covers trim, rollouts,
wind fields and linear modes, and the [scripts page](https://matusgib.github.io/AtiSim/scripts.html)
lists every command-line tool.

## What it is validated for

AtiSim is a **comparative and mechanistic** tool for the **longitudinal** gust response of a
transport aircraft at cruise. It can tell you which of two encounters is worse, how the response
scales and why. It is **not** a load calculator.

- Flown through the recorded Hannibal encounter, the 747 reaches **70.2%** of the recorded
  peak-to-peak load. That encounter was flown by a DC-10, for which no derivative set is
  published, so the model is not expected to match the recorded load exactly.
- It reproduces the three-aircraft load ordering and mechanism in NASA TM-102186, and the linear
  modes of its source data to within 2%.
- The validated envelope is 747-class aircraft at Mach 0.70–0.90 and 35,000–45,000 ft, with
  |α| < 10° (**the lift model has no stall**) and gusts larger than about three wingspans.

For the full statement, the evidence behind it and the known limitations, see
[Validation](https://matusgib.github.io/AtiSim/validation.html).

## Testing

```bash
python -m pip install -e ".[dev,ui]"
python -m pytest -q                        # the test suite
python -m pytest --nbval-lax notebooks/    # the executed validation notebooks
```

The tests check bands and orderings rather than exact values. Without the `ui` extra, the
artifact and figure tests skip themselves.

## Documentation

The full documentation, including the API reference, is at
**[matusgib.github.io/AtiSim](https://matusgib.github.io/AtiSim/)**. To build it locally:

```bash
python -m pip install -e ".[docs]"
python -m sphinx -b html docs docs/_build/html
```

## Contributing

Bug reports and pull requests are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md)
before you start. Changes are listed in [CHANGELOG.md](CHANGELOG.md).

## License

AtiSim is released under the MIT license; see [LICENSE](LICENSE). [NOTICE](NOTICE) explains what
the license covers and what it does not, including the third-party reference data.
