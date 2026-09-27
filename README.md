# AtiSim

[![tests](https://github.com/MatusGib/AtiSim/actions/workflows/tests.yml/badge.svg)](https://github.com/MatusGib/AtiSim/actions/workflows/tests.yml)
[![docs](https://github.com/MatusGib/AtiSim/actions/workflows/docs.yml/badge.svg)](https://matusgib.github.io/AtiSim/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

AtiSim is a flight dynamics model with six degrees of freedom, written in JAX. It calculates
the response of a fixed-wing aircraft to clear-air turbulence.

![A test flight through the Hannibal vortex pair. The flight instruments are on the left and the test card is on the right.](docs/images/cockpit.png)

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

3. Install AtiSim and its application:

   ```bash
   python -m pip install -e ".[ui]"
   ```

4. Test the installation. Make sure that the last line is `11/11 checks passed`:

   ```bash
   python scripts/sanity.py
   ```

5. Start the application. The argument is the directory for the run files:

   ```bash
   atisim ui runs
   ```

   The application opens in your web browser at <http://127.0.0.1:8050>. To stop it, push
   `Ctrl+C` in the terminal.

## Modes

The application has three modes. The switch at the top of each page selects the mode.

| Mode | Use it to |
|---|---|
| **Test card** | fly four test points in a cockpit in the web browser. The debrief gives the peak load factor. |
| **Lab** | fly a preset case with a few changed values, run four analyses and compare two results. |
| **Engineering** | set all the values of a run, run all the analyses and scripts, and examine each result in detail. |

The User Manual, section 4, gives the procedures.

## Documentation

The documentation is at **[matusgib.github.io/AtiSim](https://matusgib.github.io/AtiSim/)**.

| Document | Read it to |
|---|---|
| [User Manual](https://matusgib.github.io/AtiSim/user-manual.html) | install and use AtiSim |
| [Physics and Assumptions](https://matusgib.github.io/AtiSim/physics-and-assumptions.html) | know the equations, the assumptions and the limits of use |
| [Development Manual](https://matusgib.github.io/AtiSim/development-manual.html) | change the code and run the tests |
| [API Reference](https://matusgib.github.io/AtiSim/api/index.html) | find a module, a class or a function |

Use AtiSim to compare turbulence encounters. Do not use it to calculate design loads.

## License

AtiSim has the MIT license. Refer to [LICENSE](LICENSE), and to [NOTICE](NOTICE) for the
third-party data. The changes in each version are in [CHANGELOG.md](CHANGELOG.md).
