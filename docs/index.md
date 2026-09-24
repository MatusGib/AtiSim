# AtiSim

AtiSim is a flight dynamics model with six degrees of freedom, written in JAX. It calculates the
response of a fixed-wing aircraft to clear-air turbulence.

![The AtiSim cockpit display](images/panel.png)

## The manuals

| Manual | Read it to |
|---|---|
| {doc}`user-manual` | install and use AtiSim: concepts, procedures, scripts and troubleshooting |
| {doc}`physics-and-assumptions` | know the equations, the assumptions, the limits of use and the validation |
| {doc}`development-manual` | change or extend the code, and run the tests |
| {doc}`api/index` | find a module, a class or a function |

The manuals have the structure of the JSBSim Reference Manual. They use ASD-STE100 Simplified
Technical English.

## Quick start

1. Install AtiSim:

   ```bash
   git clone https://github.com/MatusGib/AtiSim.git
   cd AtiSim
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install -e .
   ```

2. Test the installation:

   ```bash
   python scripts/sanity.py
   ```

3. Fly a recorded turbulence encounter:

   ```bash
   python scripts/vortex.py --case hannibal --png runs/hannibal.png
   ```

{doc}`user-manual`, section 4, gives more procedures.

## Limits of use

Use AtiSim to compare turbulence encounters and to find the mechanism of a response. Do not use
it to calculate design loads. {doc}`physics-and-assumptions`, section 1, gives the limits.

```{toctree}
:maxdepth: 2
:caption: Manuals
:hidden:

user-manual
physics-and-assumptions
development-manual
```

```{toctree}
:maxdepth: 2
:caption: Reference
:hidden:

api/index
references
changelog
```
