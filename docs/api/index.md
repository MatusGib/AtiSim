# API Reference

This reference gives the modules of the `atisim` package. Sphinx makes it from the docstrings of
the code. Each docstring tells what the code models, what it assumes, and where its numbers come
from.

Import `atisim` before you make a JAX array. The import sets 64-bit precision.

## Dynamics core

```{eval-rst}
.. autosummary::
   :toctree: generated

   atisim.state
   atisim.dynamics
   atisim.integrate
   atisim.trim
   atisim.aero
   atisim.atmosphere
   atisim.units
```

## Aircraft

```{eval-rst}
.. autosummary::
   :toctree: generated

   atisim.aircraft
   atisim.airframe
   atisim.loads
```

## Wind and turbulence

```{eval-rst}
.. autosummary::
   :toctree: generated

   atisim.wind
   atisim.gust
   atisim.insitu
   atisim.response
```

## Control

```{eval-rst}
.. autosummary::
   :toctree: generated

   atisim.autopilot
   atisim.manual
   atisim.sensors
```

## Validation and verification

```{eval-rst}
.. autosummary::
   :toctree: generated

   atisim.validation
   atisim.verification
   atisim.checks
   atisim.records
   atisim.cr2144_mach
   atisim.jsbsim_ref
   atisim.jsbsim_vortex_ref
   atisim.predictions
   atisim.provenance
   atisim.sensitivity
```

## Visualisation and analysis

```{eval-rst}
.. autosummary::
   :toctree: generated

   atisim.viz
   atisim.vortex_viz
   atisim.panel
   atisim.analysis.artifact
   atisim.analysis.figures
   atisim.analysis.series
   atisim.analysis.runs
   atisim.analysis.report
```

## Analyses, studies and engine development

```{eval-rst}
.. autosummary::
   :toctree: generated

   atisim.analyses
   atisim.studies
   atisim.coverage
   atisim.fieldkinds
   atisim.analysis.diagnostics
   atisim.analysis.profiles
   atisim.analysis.step_inspector
   atisim.analysis.devfigures
   atisim.analysis.commits
```

## Runs and the application

```{eval-rst}
.. autosummary::
   :toctree: generated

   atisim.run
   atisim.cli
   atisim.apps.shell
   atisim.apps.start
   atisim.apps.setup
   atisim.apps.results
   atisim.apps.analyses
   atisim.apps.diagnostics
   atisim.apps.compare
   atisim.apps.jobs
   atisim.apps.worker
   atisim.apps.theme
   atisim.apps.components
   atisim.apps.sweep
```
