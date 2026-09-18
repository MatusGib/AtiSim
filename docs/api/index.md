# API reference

The `atisim` package, module by module. Each module's docstring states what it models, what it
assumes, and which source its numbers come from — the reference is generated from those
docstrings, so it describes the code as it is.

Importing `atisim` enables float64 in JAX, and it must be imported before any array is created.

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
   atisim.apps.sweep
```
