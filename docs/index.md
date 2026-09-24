# AtiSim

**AtiSim is a six-degree-of-freedom fixed-wing flight dynamics model written in JAX, built to
study how aircraft respond to clear-air turbulence.** It has been validated by rebuilding real
turbulence encounters from NASA flight records and flying the model through them.

![The AtiSim cockpit panel](images/panel.png)

AtiSim uses a quaternion state, fixed-step RK4 integration and `lax.scan` rollouts, with `jit`
and `vmap` over ensembles and float64 throughout. Wind enters only through the air-relative
velocity, so you can fly any wind field through the same integrator. That includes a vortex
array identified from a flight recorder, a Dryden ensemble, a microburst and a mountain lee
wave.

## What it is for

AtiSim is a **comparative and mechanistic** tool for the **longitudinal** gust response of a
transport aircraft. It can tell you which of two encounters is worse, how the response scales
and why. It will get the ordering of encounters right and explain it, but it will not tell you
that "the load will be 2.3 g". {doc}`validation` states exactly what the model can be used for,
and within which envelope.

## Where to start

- {doc}`getting-started` shows how to install AtiSim, run the tests and fly your first
  encounter.
- {doc}`user-guide` covers trim, rollouts, wind fields, linear modes and run analysis from
  Python.
- {doc}`scripts` lists every command-line tool.
- {doc}`validation` describes the evidence behind the model and its known limitations.
- {doc}`model` sets out the equations, conventions and modelling assumptions.
- {doc}`api/index` is the reference for the `atisim` package, module by module.

```{toctree}
:maxdepth: 2
:caption: Using AtiSim
:hidden:

getting-started
user-guide
scripts
```

```{toctree}
:maxdepth: 2
:caption: The model
:hidden:

validation
model
references
```

```{toctree}
:maxdepth: 2
:caption: Reference
:hidden:

api/index
changelog
```
