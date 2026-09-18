# AtiSim

**A six-degree-of-freedom fixed-wing flight dynamics core in JAX, built to study how aircraft
respond to clear-air turbulence — and validated by rebuilding real encounters from NASA flight
records and flying the model through them.**

Quaternion state, fixed-step RK4, `lax.scan` rollouts, `jit` and `vmap` over ensembles, float64
throughout. Wind enters only through the air-relative velocity, so any wind field — a Rankine
vortex array identified from a DC-10's flight recorder, a Dryden ensemble, a microburst, a lee
wave — drops into the same integrator.

## What it is for

AtiSim is a **comparative and mechanistic** tool for **longitudinal** gust response: which
encounter is worse, how the response scales, and why. It is **not** a load calculator. It will
order encounters correctly and explain the ordering; it will not tell you "the load will be
2.3 g". {doc}`validation` states the claim exactly, with the envelope it holds in, and lists what
is and is not validated.

## Where to start

- {doc}`getting-started` — install, check you are running the tree you think you are, and fly
  the first encounter.
- {doc}`validation` — what the model may be used for, and the status of every known gap.
- {doc}`running` — every script and command, with what each one measures.
- {doc}`api/index` — the `atisim` package, module by module.

## Why there is so much documentation

Every number in this project carries the table it came from, and every result is recorded in
{doc}`the project record <PROJECT>`, beside the tolerance it was measured to. Superseded
results are struck through rather than deleted, so an inconvenient finding cannot quietly
disappear. {doc}`the assumptions register <ASSUMPTIONS>` lists what the model assumes before any comparison, each with a
measured bound. {doc}`the working rules <DEVELOPMENT>` holds the six rules the record was kept by. It reads as
paperwork for a flight simulator; it is the reason the validation claim can be checked rather
than taken on trust.

```{toctree}
:maxdepth: 2
:caption: Using AtiSim
:hidden:

getting-started
validation
running
```

```{toctree}
:maxdepth: 2
:caption: Reference
:hidden:

api/index
```

```{toctree}
:maxdepth: 1
:caption: The project record
:hidden:

Changelog <changelog>
Project record <PROJECT>
Assumptions <ASSUMPTIONS>
Working rules <DEVELOPMENT>
```
