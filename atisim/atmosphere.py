"""ISA standard atmosphere, 0 to 20 km.

Two layers are needed: the 747 cruise condition sits above the tropopause.

Altitude is treated as geopotential. The geometric/geopotential difference is
0.17% at 11 km and 0.31% at 20 km, which is far below the uncertainty in the
aero data and not worth the complication here.
"""

import jax.numpy as jnp
from jax import Array

# ISA / ICAO Doc 7488
T0 = 288.15  # K, sea-level temperature
P0 = 101325.0  # Pa, sea-level pressure
RHO0 = 1.225  # kg/m^3, sea-level density
LAPSE = -0.0065  # K/m, troposphere temperature gradient
H_TROPOPAUSE = 11000.0  # m
T_TROPOPAUSE = T0 + LAPSE * H_TROPOPAUSE  # 216.65 K
G0 = 9.80665  # m/s^2
R_AIR = 287.05287  # J/(kg.K)
GAMMA = 1.4

P_TROPOPAUSE = P0 * (T_TROPOPAUSE / T0) ** (-G0 / (LAPSE * R_AIR))


def temperature(h: Array) -> Array:
    return jnp.where(
        h < H_TROPOPAUSE,
        T0 + LAPSE * jnp.minimum(h, H_TROPOPAUSE),
        T_TROPOPAUSE,
    )


def pressure(h: Array) -> Array:
    h_trop = jnp.minimum(h, H_TROPOPAUSE)
    p_lower = P0 * ((T0 + LAPSE * h_trop) / T0) ** (-G0 / (LAPSE * R_AIR))
    p_upper = P_TROPOPAUSE * jnp.exp(
        -G0 * jnp.maximum(h - H_TROPOPAUSE, 0.0) / (R_AIR * T_TROPOPAUSE)
    )
    return jnp.where(h < H_TROPOPAUSE, p_lower, p_upper)


def density(h: Array) -> Array:
    return pressure(h) / (R_AIR * temperature(h))


def speed_of_sound(h: Array) -> Array:
    return jnp.sqrt(GAMMA * R_AIR * temperature(h))
