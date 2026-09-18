# AtiSim

**A six-degree-of-freedom fixed-wing flight dynamics core in JAX, built to study how aircraft
respond to clear-air turbulence — and validated by rebuilding real encounters from NASA flight
records and flying the model through them.**

![The AtiSim cockpit panel](docs/summary/panel.png)

Quaternion state, fixed-step RK4, `lax.scan` rollouts, `jit` and `vmap` over ensembles, float64
throughout. Wind enters only through the air-relative velocity, so any wind field — a vortex
array identified from a flight recorder, a Dryden ensemble, a microburst, a mountain lee wave —
drops into the same integrator.

## What it does

The headline test is the **Hannibal, Missouri encounter of 3 April 1981**: a DC-10 at 37,000 ft
flew through a row of clear-air-turbulence vortices, and NASA identified the wind field from its
flight recorder (Parks et al. 1985; Mehta 1987). AtiSim rebuilds that field, flies a Boeing 747
through it, and compares the response with the recorded load.

The simulated load reaches **75.3% of the recorded peak-to-peak** — and the documentation
explains the shortfall rather than tuning it away. The model also reproduces the ordering and
mechanism of NASA TM-102186's three-aircraft comparison, and the linear modes of its source data.
The documentation's *Validation* page states the claim, the envelope it holds in, and the status
of every known gap.

## What it cannot do — read this first

**AtiSim is a comparative and mechanistic tool, not a load calculator.** It will tell you which
encounter is worse and why, and get the ordering right. It will not tell you "the load will be
2.3 g": the recorded encounter was flown by a DC-10 at an unrecorded weight, and no DC-10
derivative set is published.

It is validated only inside this envelope:

- 747-class transports, Mach 0.70–0.90, 35,000–45,000 ft
- longitudinal response — lateral fields exist, lateral validation does not
- angle of attack below about 10° — **the lift model has no stall**
- gusts larger than about three wingspans
- a flat, non-rotating Earth — a rotating WGS-84 model is complete on the `wgs84-earth` branch

## Install

Python 3.10 or later.

```bash
git clone https://github.com/MatusGib/AtiSim.git
cd AtiSim
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e .
```

On Linux and macOS use `.venv/bin/python`. The runtime needs four packages — `jax`, `numpy`,
`scipy`, `matplotlib`. Extras: `.[dev]` for the tests and notebooks, `.[ui]` for the analysis
app, `.[docs]` for the documentation site, `.[ref]` for regenerating the JSBSim comparison data.

## Quick start

**Walk the validation ladder** — one notebook that re-runs the evidence behind the validation
claim, from answers you can work out by hand up to the recorded encounter and the published
orderings, asserting each as it goes. It needs the `dev` extra and takes a few minutes:

```bash
.venv/Scripts/python.exe -m pip install -e .[dev]
.venv/Scripts/python.exe -m jupyter lab notebooks/validation-ladder.ipynb
```

**The sanity checks** on their own — eleven cases from degenerate inputs upward, each expected
value derived by hand and printed beside the model's answer:

```bash
.venv/Scripts/python.exe scripts/sanity.py
```

**Fly the Hannibal encounter** and draw the analysis figure:

```bash
.venv/Scripts/python.exe scripts/vortex.py --case hannibal --png runs/v.png
```

**Fly it by hand**, with a cockpit display — arrow keys are the stick, `a` hands over to the
autopilot:

```bash
.venv/Scripts/python.exe scripts/fly.py --wind hannibal
```

**Explore a run** in the analysis app, where clicking any time series moves every panel —
including the 3-D wind field with the trajectory through it — to that instant:

```bash
.venv/Scripts/python.exe scripts/vortex.py --artifacts runs/analysis
.venv/Scripts/python.exe -m atisim.apps.sweep runs/analysis
```

Every one of the 44 scripts is described in the documentation's *Running it* page.

## Tests

```bash
.venv/Scripts/python.exe -m pytest -q
```

Over 900 tests, asserting bands and orderings rather than exact values. The count and runtime of
the last full run are recorded in [`docs/PROJECT.md`](docs/PROJECT.md) §10. The notebooks are a
second gate, executed cell by cell:

```bash
.venv/Scripts/python.exe -m pytest --nbval-lax notebooks/
```

CI runs both on every pull request. If you work in more than one checkout, first confirm which
one Python imports:

```bash
.venv/Scripts/python.exe -c "import atisim; print(atisim.__file__)"
```

## Documentation

```bash
.venv/Scripts/python.exe -m pip install -e .[docs]
.venv/Scripts/python.exe -m sphinx -b html docs docs/_build/html
```

The site covers getting started, what the model may be used for, every script, and the API
reference generated from the source.

- [`docs/PROJECT.md`](docs/PROJECT.md) — the standing record: what exists, what is measured and
  to what tolerance, what is known to be wrong, and what is left
- [`docs/ASSUMPTIONS.md`](docs/ASSUMPTIONS.md) — every modelling assumption, each with a measured
  bound
- [`CHANGELOG.md`](CHANGELOG.md) — what this release contains, by capability
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — how to extend it without breaking the record

## Why there is so much documentation

Every number in AtiSim carries the table it came from, and `atisim/provenance.py` records whether
each constant was sourced, derived, calibrated or declared — a test fails if a derived value
points at something that does not exist. Every result is a row in `docs/PROJECT.md` beside the
tolerance it was measured to. Superseded results are struck through rather than deleted, so an
inconvenient finding cannot quietly disappear, and claims about untested cases are sealed in
`atisim/predictions.py` *before* the run that decides them. It is more paperwork than a flight
simulator usually carries; it is what lets the validation claim be checked rather than taken on
trust.

## Sources

The reference documents are listed, with checksums and permanent locators, in
[`Reference_papers/SOURCES.md`](Reference_papers/SOURCES.md). US Government documents are
included; publisher-held papers are cited, not redistributed.

## License

MIT — see [`LICENSE`](LICENSE), with [`NOTICE`](NOTICE) for what it does and does not cover.
