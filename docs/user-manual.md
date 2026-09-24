# User Manual

This manual tells you how to install AtiSim, how to run it, and how to use its results. For the
equations and the assumptions of the model, refer to {doc}`physics-and-assumptions`. To change
the code, refer to {doc}`development-manual`.

## 1 Introduction

### 1.1 What AtiSim does

AtiSim is a flight dynamics model with six degrees of freedom. It calculates the response of a
fixed-wing aircraft to turbulence. It has these parts:

- A rigid-body model of the aircraft, with quaternion attitude and a fixed-step RK4 integrator.
- Aircraft data for a Boeing 747, a Boeing 737 and two light aircraft.
- Wind fields: vortex arrays, updrafts, microbursts, mountain lee waves and Dryden turbulence.
- A trim solution, linear modes and sensitivities.
- A cockpit display, an autopilot, and an application to examine a run.

AtiSim uses JAX. You can compile a run with `jit`, and you can fly many runs at the same time
with `vmap`.

### 1.2 Limits of use

Use AtiSim to compare turbulence encounters and to find the mechanism of a response. Use it
only for the longitudinal response of a 747-class aircraft at Mach 0.70 to 0.90, between
35,000 ft and 45,000 ft. {doc}`physics-and-assumptions`, section 1, gives the full limits.

:::{caution}
Do not use AtiSim to calculate a design load or a certification load. AtiSim does not predict
absolute loads. An incorrect load can cause an unsafe design.
:::

### 1.3 How to use this manual

- Section 2 tells you how to install AtiSim.
- Section 3 gives the concepts that you must know before you use AtiSim.
- Section 4 gives the procedures for the usual tasks.
- Section 5 gives reference data: scripts, keys and wind fields.
- Section 6 tells you what to do when a problem occurs.
- Section 7 is a glossary.

The procedures show commands for Linux and macOS. On Windows, use `.venv\Scripts\python.exe`
where the procedure shows `python` in an active environment.

## 2 Installation

### 2.1 Requirements

- Python 3.10 or later.
- The Python packages `jax`, `numpy`, `scipy` and `matplotlib`. The installation adds them.

### 2.2 Install AtiSim

1. Get a copy of the source code:

   ```bash
   git clone https://github.com/MatusGib/AtiSim.git
   cd AtiSim
   ```

2. Make a virtual environment:

   ```bash
   python -m venv .venv
   ```

3. Start the virtual environment:

   ```bash
   source .venv/bin/activate
   ```

4. Install AtiSim:

   ```bash
   python -m pip install -e .
   ```

### 2.3 Optional parts

Each optional part adds packages for a special task. To install a part, give its name in
brackets, for example `python -m pip install -e ".[ui]"`.

| Part | Packages | Use |
|---|---|---|
| `dev` | pytest, Jupyter, nbval | the tests and the notebooks |
| `ui` | Dash, Plotly, PyArrow | run files and the analysis application |
| `docs` | Sphinx, MyST, Furo | the documentation |
| `ref` | JSBSim | the scripts that make the JSBSim reference data |

### 2.4 Test the installation

1. Run the sanity checks:

   ```bash
   python scripts/sanity.py
   ```

2. Make sure that the last line is `11/11 checks passed`.

The sanity checks compare the model with values that you can calculate by hand. For example, an
aircraft with no aerodynamic forces must fall with the local gravity.

## 3 Concepts

### 3.1 Units and frames

| Item | Convention |
|---|---|
| Units | SI units: meters, seconds, kilograms and radians. `atisim.units` holds the conversion factors. |
| Inertial frame | NED: x north, y east, z down. The altitude is `-pos_ned[2]`. |
| Body frame | x forward, y right, z down. |
| Attitude | a quaternion `[w, x, y, z]` from the body frame to NED. |
| Wind | `wind_ned` is in NED. An updraft has a negative z component. `omega_gust` is in the body frame. |
| Precision | 64-bit floating point. The import of `atisim` sets this. |

:::{important}
Import `atisim` before you make a JAX array. The 64-bit setting has no effect on arrays that
exist before the import.
:::

### 3.2 Aircraft

The dictionary `atisim.aircraft.REGISTRY` holds the aircraft. The dictionary
`atisim.aircraft.CRUISE` holds a reference condition for each aircraft.

| Key | Aircraft | Reference condition | Use |
|---|---|---|---|
| `boeing747` | Boeing 747, from NASA CR-2144 | 12,192 m, 235.9 m/s | validated. Use this aircraft for turbulence studies. |
| `boeing747_approach` | Boeing 747, power approach | sea level, 84.9 m/s | modes and approach cases |
| `boeing747_jsbsim` | Boeing 747, from the JSBSim B747 | 11,582 m, 236.1 m/s | validated. Use it for comparison with JSBSim. |
| `boeing737` | Boeing 737, from JSBSim | 9,144 m, 236.5 m/s | comparison with JSBSim only |
| `boeing737_approach` | Boeing 737, approach | 1,524 m, 133.8 m/s | comparison with JSBSim only |
| `cherokee` | Piper Cherokee PA-28-180 | 1,500 m, 50 m/s | manual flight only |
| `cessna172` | Cessna 172 | 1,524 m, 60 m/s | manual flight only |

Each aircraft is a named tuple of JAX arrays: mass, inertia, geometry and stability
derivatives. Each derivative has a reference to the table that it comes from. The ledger
`atisim.provenance.LEDGER` gives the category of each constant: sourced, derived, calibrated or
declared.

### 3.3 Wind fields and wind models

A wind field is a function. It takes a position in NED, and it gives the wind in NED at that
position.

A wind model is the function that the integrator calls at each step. It gives the wind and
the gust rates. `wind.field_model(field)` makes a wind model from a wind field. It calculates
the gust rates from the gradient of the field.

### 3.4 A simulation run

A run has three steps:

1. The trim finds the angle of attack, the elevator and the throttle for steady level flight.
2. The rollout integrates the equations of motion with fixed controls, or with the autopilot.
3. The analysis measures the response: load factor, pitch angle, angle of attack and rates.

JAX compiles the rollout at the first run. Thus the first run is slow, and the subsequent runs
are fast.

### 3.5 Encounters

An encounter is one flight through a wind field. The module `atisim.vortex_viz` flies
encounters and measures them. It gives an `Encounter`, which holds NumPy arrays of the response.
The analysis window is the part of the flight that the measurements use.

| Field of `Encounter` | Content |
|---|---|
| `t` | time, s |
| `north`, `altitude` | position, m |
| `w_up` | vertical gust at the aircraft, m/s, positive up |
| `alpha_air` | angle of attack relative to the air, rad |
| `theta`, `q` | pitch angle, rad, and pitch rate, rad/s |
| `n_z` | normal load factor, g |
| `elevator` | elevator deflection, rad |
| `window` | a mask that is true inside the analysis window |

## 4 Procedures

### 4.1 Fly an encounter from the command line

1. Fly the 747 through the Hannibal vortex array, and save the figure:

   ```bash
   python scripts/vortex.py --case hannibal --png runs/hannibal.png
   ```

2. Open `runs/hannibal.png`.

The script flies a vortex, an updraft and an elevator maneuver. It prints a table of the results,
and it draws the analysis figure.

### 4.2 Fly the aircraft manually

1. Start the cockpit display with the Hannibal wind field:

   ```bash
   python scripts/fly.py --wind hannibal
   ```

2. Use the keys in section 5.2 to fly the aircraft.
3. To engage the autopilot, push `a`.
4. To keep the flight for later analysis, start the script with `--save runs/flight.npz`.

### 4.3 Trim an aircraft and run a simulation

This example trims the 747 and flies it through the Hannibal vortex array for 60 s:

```python
import jax
import jax.numpy as jnp

import atisim  # sets 64-bit precision. Import it before you make an array.
from atisim import integrate, trim, wind
from atisim.aircraft import CRUISE, REGISTRY

ac = REGISTRY["boeing747"]
V, H = CRUISE["boeing747"]["airspeed"], CRUISE["boeing747"]["altitude"]

# Trim for steady level flight. x holds [alpha, elevator, throttle].
x, residual = trim.trim(jnp.array(V), jnp.array(H), ac)
state = trim.trimmed_state(x[0], jnp.array(V), jnp.array(H))
controls = trim.trimmed_controls(x[1], x[2])

# Start 5 km before the first vortex.
array = wind.mehta_hannibal_array(H)
state = state._replace(pos_ned=jnp.array([float(array.north.min()) - 5000.0, 0.0, -H]))
wind_model = wind.field_model(lambda p: wind.vortex_wind(p, array))

sim = integrate.init_sim(state, jax.random.PRNGKey(0))
final, history = integrate.rollout(sim, controls, jnp.array(0.01), ac,
                                   n_steps=6000, wind_model=wind_model)
```

`rollout` gives the last `SimState` and the `State` at each step. To fly with the autopilot, use
`autopilot.closed_loop_rollout`.

### 4.4 Fly an encounter and measure the load

1. Fly the 747 through the Hannibal field on its identified path:

   ```python
   import atisim
   from atisim import vortex_viz

   enc, info = vortex_viz.fly_mehta("boeing747", dt=0.01, replayed=True)
   ```

2. Read the load factor in the analysis window:

   ```python
   n_z = enc.n_z[enc.window]
   print(f"peak-to-peak load: {n_z.max() - n_z.min():.2f} g")
   ```

3. Make sure that the result is 1.90 g.

The argument `replayed=True` samples the field along the path of the recorded aircraft.
{doc}`physics-and-assumptions`, section 10.3, gives the reason.

### 4.5 Combine wind fields

Use `wind.superpose` to add two or more fields. Use `vortex_viz.fly` to fly any field:

```python
import atisim
from atisim import vortex_viz, wind
from atisim.aircraft import CRUISE, REGISTRY

ac = REGISTRY["boeing747"]
V, H = CRUISE["boeing747"]["airspeed"], CRUISE["boeing747"]["altitude"]

array = wind.mehta_hannibal_array(H)
column = wind.UpdraftColumn(north=8000.0, east=0.0, w0=10.0, radius=2000.0, sharpness=4.0)
field = wind.superpose(
    lambda p: wind.vortex_wind(p, array),
    lambda p: wind.updraft_wind(p, column),
)

enc = vortex_viz.fly(ac, field, V, H, label="vortices and updraft",
                     start_north=-8000.0, seconds=90.0,
                     window=(-3000.0, 12000.0), window_name="the disturbed part")
```

### 4.6 Add a wind field

1. Write a function that takes a position in NED and gives the wind in NED, in m/s.
2. Use only `jax.numpy` in the function, so that JAX can calculate its gradient.
3. Give the function to `vortex_viz.fly`, or to `wind.field_model` for a rollout.

The upward wind is negative in NED. For example, this field is a uniform updraft of 2 m/s:

```python
import jax.numpy as jnp

def uniform_updraft(pos_ned):
    return jnp.array([0.0, 0.0, -2.0])
```

The full wind-model interface is in {doc}`physics-and-assumptions`, section 7.1.

### 4.7 Calculate the linear modes

This example gives the phugoid and the short-period modes of the 747 at cruise:

```python
import jax.numpy as jnp

import atisim
from atisim import trim, validation
from atisim.aircraft import CRUISE, REGISTRY

ac = REGISTRY["boeing747"]
V, H = CRUISE["boeing747"]["airspeed"], CRUISE["boeing747"]["altitude"]
x, _ = trim.trim(jnp.array(V), jnp.array(H), ac)
alpha, elevator, throttle = (float(v) for v in x)

(ph_wn, ph_zeta), (sp_wn, sp_zeta) = validation.longitudinal_modes(
    ac, alpha, elevator, throttle, V, H)
```

Each mode is a pair of the natural frequency, in rad/s, and the damping ratio.
`validation.lateral_modes` gives the lateral modes. `atisim.sensitivity` gives the change of a
result when a coefficient changes.

### 4.8 Fly an ensemble

1. Make one Dryden field for each seed with `wind.dryden_field(sigma, seed)`.
2. Fly each field.
3. Use `atisim.response` to calculate the spectrum and the exceedance rate of the response.

The scripts `cat_ensemble.py` and `cat_spectra.py` are complete examples. For a batch of runs
with one deterministic field, use `integrate.batch_sim` and `integrate.batched_rollout`.

### 4.9 Save a run and examine it

The analysis application needs the `ui` part (section 2.3).

1. Fly the encounters, and save each run:

   ```bash
   python scripts/vortex.py --artifacts runs/analysis
   ```

2. Start the analysis application:

   ```bash
   python -m atisim.apps.sweep runs/analysis
   ```

3. Open the address that the application prints in a web browser.
4. Click a point in a time series. All panels move to that time, and the 3-D wind field shows
   the position of the aircraft.

To read a saved run in Python, use `atisim.analysis.artifact.read_run`.

### 4.10 Do the validation again

The validation needs the `dev` part (section 2.3).

1. Open the validation notebook:

   ```bash
   python -m jupyter lab notebooks/validation-ladder.ipynb
   ```

2. Run all cells.

The notebook does the validation in four steps: checks by hand, the source data, a recorded
encounter, and the published orders of loads. Each step makes assertions. The notebook
`notebooks/solver-validation.ipynb` does a check of the integrator.

## 5 Reference

### 5.1 Command-line scripts

Run the scripts from the root of the repository. Each script shows its options with `--help`.
If you give `--png` or `--outdir`, the script writes its figures to that location. If not, the
script opens a window.

**Flight and encounters**

| Script | Function |
|---|---|
| `fly.py` | Fly the aircraft manually with a cockpit display. Options: `--aircraft`, `--wind`, `--autopilot`, `--save`. |
| `vortex.py` | Fly a vortex, an updraft and an elevator maneuver, and draw the analysis figure. Options: `--case`, `--strip`, `--artifacts`. |
| `microburst.py` | Fly through a microburst, and compare the F-factor with the thrust of the aircraft. |
| `leewave.py` | Fly the 747 through a mountain lee wave. |
| `lateral.py` | Fly vortex lines that change across the span, and show the roll response. |
| `analyse.py` | Analyze runs that `fly.py` saved. |
| `tune.py` | Show the step responses of the autopilot. |

**Validation**

| Script | Function |
|---|---|
| `sanity.py` | Do 11 checks against values that you can calculate by hand. |
| `checkpoint.py` | Trim the 747, and compare its modes with CR-2144. |
| `cat_validation.py` | Fly the published turbulence cases, and compare them with the papers. |
| `cat_ensemble.py` | Fly the three categories of TM-102186 Figure 8 through a Dryden ensemble. |
| `cat_spectra.py` | Calculate the response spectra and the exceedance rates of an ensemble. |
| `vortex_compare.py` | Compare AtiSim with JSBSim through the same vortex. |
| `cr2144_digitisation_crosscheck.py` | Compare the two readings of CR-2144 pages 220 to 222. |

**Reference data**

The tests use the stored data. Use these scripts only to make the data again.

| Script | Function | Necessary part |
|---|---|---|
| `gen_jsbsim_reference.py` | Make the 737 reference data from JSBSim. | `ref` |
| `gen_jsbsim_vortex_reference.py` | Make the vortex reference data from JSBSim. | `ref` |
| `gen_jsbsim_747.py` | Make the `boeing747_jsbsim` data from the JSBSim B747. | `ref` |
| `digitise_mil_f_8785c_fig7.py` | Read MIL-F-8785C Figure 7 from the PDF. | a copy of the document, with `--pdf` |

### 5.2 Keyboard controls

| Key | Function |
|---|---|
| Up arrow, down arrow | pitch. Up moves the stick forward, and the nose goes down. |
| Left arrow, right arrow | roll |
| `,` and `.` | rudder |
| `-` and `=` | throttle |
| `[` and `]` | pitch trim, nose down and nose up |
| `t` | set the trim to the current stick position |
| `a` | engage or disengage the autopilot |

### 5.3 Wind fields

| Field | Function | Source |
|---|---|---|
| Vortex array | `vortex_wind(p, array)`, with `mehta_hannibal_array()` or `PARKS_CASES` | Parks et al. (1985), Mehta (1987) |
| Vortex lines | `line_vortex_wind(p, array)` | the same equations as lines in space |
| Smooth-core vortex | `lamb_oseen_wind(p, array)` | Lamb–Oseen core |
| Updraft | `updraft_wind(p, UpdraftColumn(...))` | Wingrove and Bach (1994) |
| Microburst | `microburst_wind(p, microburst(u_max=..., radius=..., z_m=...))` | Oseguera and Bowles (1988) |
| Lee wave | `lee_wave_wind(p, LeeWave(...))` | Doyle et al. (2011) |
| Dryden turbulence | `dryden_field(sigma, seed)` | MIL-F-8785C |
| Sum of fields | `superpose(*fields)` | |

All functions are in `atisim.wind`. {doc}`api/index` gives the full Python interface.

## 6 Troubleshooting

| Problem | Possible cause | Action |
|---|---|---|
| The results are not accurate, or the trim does not converge. | A JAX array exists before the import of `atisim`, so the precision is 32-bit. | Import `atisim` first. |
| A change in the code has no effect. | Python imports AtiSim from a different copy. | Run `python -c "import atisim; print(atisim.__file__)"`. If the path is incorrect, install AtiSim again from the correct copy. |
| The tests for run files and figures show as skipped. | The `ui` part is not installed. | Install the `ui` part (section 2.3). |
| A script shows `ModuleNotFoundError: No module named 'jsbsim'`. | The script makes JSBSim reference data. | Install the `ref` part. |
| The first run is slow. | JAX compiles the run. | Wait. JAX compiles again only for a new number of steps or a new wind model. |
| The load factor is not 1 at the start of a vortex run. | The first vortex is too near. | Start the run 12 core radii or more before the first core. Use `--lead-in`. |
| The angle of attack goes above 10°. | The run is outside the limits of the model. | Do not use the results. The `alpha_band` check shows this condition. |
| The trim gives an angle of attack of more than 15°. | The Newton iteration found a root that is not a flight condition. | Use `trim.is_physical` to examine the result. Change the airspeed or the altitude. |
| The altitude goes below zero. | The model has no ground. | Stop the run at a clearance that you select. |
| `digitise_mil_f_8785c_fig7.py` cannot find the PDF. | The repository does not include the document. | Give the path with `--pdf`. |

## 7 Glossary

| Term | Meaning |
|---|---|
| Air-relative | Relative to the moving air, not to the ground. |
| Analysis window | The part of a run that the measurements use. |
| CR-2144 | NASA CR-2144, the source of the 747 data. |
| Dryden turbulence | A random turbulence model with the spectra of MIL-F-8785C. |
| Encounter | One flight through a wind field. |
| F-factor | A hazard index for wind shear. A positive value decreases the energy of the aircraft. |
| Gust rate | The rotation rate that the gradient of a wind field gives to the aircraft. |
| Load factor | The normal acceleration, in units of g. In level flight, it is about 1. |
| NED | The frame with axes north, east and down. |
| Phugoid | The slow longitudinal mode, with changes of speed and altitude. |
| Rollout | An integration of the equations of motion for many steps. |
| Short period | The fast longitudinal mode, with changes of pitch and angle of attack. |
| Trim | The controls and attitude for steady flight. |
| Vortex core | The center part of a vortex, which turns as a solid body. |
