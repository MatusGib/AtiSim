# User guide

This page shows how to use AtiSim from Python. Every example runs as written from the
repository root, once the package is installed.

## Conventions

| | |
|---|---|
| Units | SI throughout: metres, seconds, radians, kilograms. `atisim.units` holds the conversion factors. |
| Inertial frame | NED (north, east, down), flat and non-rotating. Altitude is `-pos_ned[2]`. |
| Body axes | x forward, y right, z down. |
| Attitude | a quaternion `[w, x, y, z]` that rotates body to NED. |
| Wind | `wind_ned` is in NED, where an updraft is **negative**. `omega_gust` is in body axes. |
| Precision | float64. Import `atisim` before you create any JAX array. |

## Aircraft

Aircraft live in `atisim.aircraft.REGISTRY`, and each one has a reference trim condition in
`CRUISE`:

| Key | Aircraft | Reference condition | Use |
|---|---|---|---|
| `boeing747` | Boeing 747, NASA CR-2144 | 12,192 m, 235.9 m/s | **validated**; the default for turbulence work |
| `boeing747_approach` | Boeing 747, CR-2144 power approach | sea level, 84.9 m/s | modes and approach cases |
| `boeing747_jsbsim` | Boeing 747, recovered from JSBSim's B747 | 11,582 m, 236.1 m/s | **validated**; cross-code vortex comparison |
| `boeing737` | Boeing 737, linearised from JSBSim | 9,144 m, 236.5 m/s | cross-code verification only |
| `boeing737_approach` | Boeing 737, approach | 1,524 m, 133.8 m/s | cross-code verification only |
| `cherokee` | Piper Cherokee PA-28-180 | 1,500 m, 50 m/s | manual flying; outside the validated scope |
| `cessna172` | Cessna 172 | 1,524 m, 60 m/s | manual flying; outside the validated scope |

An `Aircraft` is a NamedTuple of JAX arrays: mass, inertia, geometry and stability
derivatives. Every derivative cites the table it came from, and `atisim.provenance.LEDGER`
records whether each constant was sourced, derived, calibrated or declared.

## Trim and fly

This example trims the 747 for level flight, then integrates it through a wind field:

```python
import jax
import jax.numpy as jnp

import atisim  # enables float64; import before creating any array
from atisim import integrate, trim, wind
from atisim.aircraft import CRUISE, REGISTRY

ac = REGISTRY["boeing747"]
V, H = CRUISE["boeing747"]["airspeed"], CRUISE["boeing747"]["altitude"]

# Newton trim for steady level flight: solves for [alpha, elevator, throttle].
x, residual = trim.trim(jnp.array(V), jnp.array(H), ac)
state = trim.trimmed_state(x[0], jnp.array(V), jnp.array(H))
controls = trim.trimmed_controls(x[1], x[2])

# Start 5 km short of the Hannibal vortex array.
array = wind.mehta_hannibal_array(H)
state = state._replace(pos_ned=jnp.array([float(array.north.min()) - 5000.0, 0.0, -H]))

# field_model turns a position -> wind function into a wind model, adding the gust rates.
wind_model = wind.field_model(lambda p: wind.vortex_wind(p, array))

sim = integrate.init_sim(state, jax.random.PRNGKey(0))
final, history = integrate.rollout(sim, controls, jnp.array(0.01), ac,
                                   n_steps=6000, wind_model=wind_model)
```

`rollout` runs under `lax.scan` with fixed controls, and returns the final `SimState` and the
`State` history stacked along a leading time axis. To fly with the autopilot engaged, use
`autopilot.closed_loop_rollout`.

## Fly an encounter and measure it

`atisim.vortex_viz` wraps trim, rollout and measurement. It returns an `Encounter`: NumPy
arrays of time, position, gust, air-relative angle of attack, pitch, pitch rate, load factor
and elevator, together with the declared analysis window.

```python
import atisim
from atisim import vortex_viz

enc, info = vortex_viz.fly_mehta("boeing747", dt=0.01, replayed=True)
n_z = enc.n_z[enc.window]
print(f"peak-to-peak load: {n_z.max() - n_z.min():.2f} g")   # 1.90 g
```

`replayed=True` evaluates Mehta's field along the path on which it was identified, rather than
along the 747's own path. This is the headline form used in {doc}`validation`.

For any other field, use `vortex_viz.fly`:

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

enc = vortex_viz.fly(ac, field, V, H, label="vortices plus updraft",
                     start_north=-8000.0, seconds=90.0,
                     window=(-3000.0, 12000.0), window_name="the disturbed stretch")
```

## Wind fields

A **field** is any function from an NED position to an NED wind vector. A **wind model** is what
the integrator calls. `wind.field_model(field)` converts the first into the second, and derives
the gust rates from the field's analytic gradient. This means a component cannot add a
translational gust while silently leaving out its rotational one.

| Field | Function | Source |
|---|---|---|
| Rankine vortex array | `vortex_wind(p, array)`; `mehta_hannibal_array()`, `PARKS_CASES` | Parks et al. 1985; Mehta 1987 |
| Line vortices, varying across the span | `line_vortex_wind(p, array)` | the same equations, written as lines in space |
| Smooth-core vortex | `lamb_oseen_wind(p, array)` | Lamb–Oseen core |
| Updraft column | `updraft_wind(p, UpdraftColumn(...))` | Wingrove & Bach 1994 magnitudes |
| Microburst | `microburst_wind(p, microburst(u_max=, radius=, z_m=))` | Oseguera & Bowles 1988 |
| Mountain lee wave | `lee_wave_wind(p, LeeWave(...))` | Doyle et al. 2011 amplitudes |
| Dryden turbulence | `dryden_field(sigma, seed)` | MIL-F-8785C spectral forms |
| Sum of fields | `superpose(*fields)` | |

To add a field of your own, write a function of position and pass it through `field_model`. The
full wind-model contract is:

```python
wind_model(wind_state, state, key, dt) -> (wind_ned, omega_gust, wind_state, key)
```

`wind_ned` is in NED. `omega_gust` is in body axes, because it is subtracted from
`state.omega`. The PRNG key is threaded through every step, so a stochastic model can draw from
it, while a deterministic one returns it untouched. The gust-rate signs are
`p_gust = +∂w_g/∂y`, `q_gust = −∂w_g/∂x` and `r_gust = +∂v_g/∂x`.

## Linear modes

This example linearises about a trim point and reads off the phugoid and short-period modes:

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

`validation.lateral_modes` does the same for the Dutch roll, roll and spiral modes. To
differentiate these results with respect to the aircraft's own coefficients, use
`atisim.sensitivity`, which reports elasticities rather than raw gradients.

## Ensembles

`integrate.batch_sim` replicates one initial state across a batch of PRNG keys, and
`integrate.batched_rollout` runs `rollout` under `vmap` over that batch. Every member of the
batch meets the same deterministic field at the same place, so only the stochastic part
varies. A Dryden field is frozen per seed (`dryden_field(sigma, seed)`), so to build an
ensemble of Dryden realisations, fly one field per seed. `scripts/cat_ensemble.py` and
`scripts/cat_spectra.py` are complete examples, and `atisim.response` turns the results into
spectra and exceedance rates.

## Outputs

- `dynamics.load_factor(state, controls, ac, wind_ned, omega_gust)` returns the body-normal
  load factor, as an accelerometer reads it: about +1 g in level flight.
- `sensors.sense(state, wind_ned)` returns air data (airspeed, angle of attack, sideslip)
  measured relative to the air, as a real sensor measures it.
- `checks.run_checks` asks whether one run holds together, with checks such as quaternion norm,
  energy closure, field divergence, a trimmed start and the angle-of-attack band.

## Run artifacts and the analysis app

With the `ui` extra installed, the scripts can save a run to disk as Parquet, together with its
metadata and check results. You can then explore it in a Dash app, where clicking any time
series moves every panel to that instant, including the 3-D wind field with the trajectory
through it:

```bash
python scripts/vortex.py --artifacts runs/analysis
python -m atisim.apps.sweep runs/analysis
```

`atisim.analysis.artifact.read_run` loads a saved run back into Python.
