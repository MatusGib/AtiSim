"""Wind and turbulence.

Today this returns zero wind. It exists now so that the interfaces around it are
already the right shape when Dryden or von Karman turbulence arrives.

Three things a turbulence model needs, all present here already:

  wind_ned    translational gust velocity, subtracted from inertial velocity
              before any angle of attack is computed
  omega_gust  angular-rate perturbation (p_g, q_g, r_g); a gust gradient across
              the span and chord is a rate, not just a velocity
  wind_state  the shaping filters that turn white noise into Dryden/von Karman
              spectra are dynamic systems and carry state between steps

`wind_state` is not in the original plan's signature. It is here because a
shaped-noise turbulence model cannot work without somewhere to keep its filter
states, and adding the slot later would mean changing the signature of `step`
-- exactly the retrofit these hooks exist to avoid. It costs an empty tuple
today.

`state` is passed in because Dryden scale lengths and intensities are functions
of altitude, and the filter time constants are functions of true airspeed.
"""

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from flightsim.state import State


class WindState(NamedTuple):
    """Empty until a turbulence model needs filter states."""


def zero_wind_state() -> WindState:
    return WindState()


def zero_wind(
    wind_state: WindState, state: State, key: Array, dt: float
) -> tuple[Array, Array, WindState, Array]:
    """Still air. Returns (wind_ned, omega_gust, wind_state, key)."""
    del state, dt
    return jnp.zeros(3), jnp.zeros(3), wind_state, key
