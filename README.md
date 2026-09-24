# AtiSim

[![tests](https://github.com/MatusGib/AtiSim/actions/workflows/tests.yml/badge.svg)](https://github.com/MatusGib/AtiSim/actions/workflows/tests.yml)
[![docs](https://github.com/MatusGib/AtiSim/actions/workflows/docs.yml/badge.svg)](https://matusgib.github.io/AtiSim/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

AtiSim is a flight dynamics model with six degrees of freedom, written in JAX. It calculates
the response of a fixed-wing aircraft to clear-air turbulence.

![The AtiSim cockpit display](docs/images/panel.png)

## Install

You must have Python 3.10 or later.

1. Get a copy of the source code:

   ```bash
   git clone https://github.com/MatusGib/AtiSim.git
   cd AtiSim
   ```

2. Make a virtual environment, and start it:

   ```bash
   python -m venv .venv
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   ```

3. Install AtiSim:

   ```bash
   python -m pip install -e .
   ```

## Run

1. Test the installation. Make sure that the last line is `11/11 checks passed`:

   ```bash
   python scripts/sanity.py
   ```

2. Fly a Boeing 747 through a recorded turbulence encounter, and save the figure:

   ```bash
   python scripts/vortex.py --case hannibal --png runs/hannibal.png
   ```

3. Fly the aircraft manually. Use the arrow keys to fly, and push `a` for the autopilot:

   ```bash
   python scripts/fly.py --wind hannibal
   ```

To use AtiSim from Python:

```python
import atisim
from atisim import vortex_viz

enc, info = vortex_viz.fly_mehta("boeing747", dt=0.01, replayed=True)
print(enc.n_z[enc.window].max())   # the peak load factor, in g
```

## Documentation

The documentation is at **[matusgib.github.io/AtiSim](https://matusgib.github.io/AtiSim/)**.
Its source is in [`docs/`](docs/).

| Document | Read it to |
|---|---|
| [User Manual](https://matusgib.github.io/AtiSim/user-manual.html) | install and use AtiSim: concepts, procedures, scripts and troubleshooting |
| [Physics and Assumptions](https://matusgib.github.io/AtiSim/physics-and-assumptions.html) | know the equations, the assumptions, the limits of use and the validation |
| [Development Manual](https://matusgib.github.io/AtiSim/development-manual.html) | change or extend the code, and run the tests |
| [API Reference](https://matusgib.github.io/AtiSim/api/index.html) | find a module, a class or a function |

Use AtiSim to compare turbulence encounters. Do not use it to calculate design loads. The
Physics and Assumptions manual gives the limits.

## License

AtiSim has the MIT license. Refer to [LICENSE](LICENSE), and to [NOTICE](NOTICE) for the
third-party data. The changes in each version are in [CHANGELOG.md](CHANGELOG.md).
