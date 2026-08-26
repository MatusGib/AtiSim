"""ISA standard atmosphere, 0 to 20 km.

Two layers are needed: the 747 cruise condition sits above the tropopause.

*** THIS MODULE TAKES GEOMETRIC ALTITUDE AND CONVERTS IT INTERNALLY. ***

It used to take its argument as GEOPOTENTIAL while every caller passed
GEOMETRIC -- `dynamics.derivatives` passes `-state.pos_ned[2]`, which is a
geometric height straight out of the NED position, and nothing converted it.
That put density 0.159% low at 30,000 ft and 0.368% at 40,000 ft, a same-signed
bias on every force at altitude since qbar is proportional to rho.

The ISA formulas below are the geopotential ones and are unchanged. What is new
is `geopotential`, applied at the boundary, so callers keep passing the
geometric altitude they actually have.

*** G0 STAYS CONSTANT IN THE BAROMETRIC INTEGRATION, AND THAT IS NOT AN
OVERSIGHT. *** Geopotential altitude is DEFINED as the height coordinate that
absorbs the variation of g into itself, so that the hydrostatic integration can
use a constant g0. Substituting g(h) here as well would double-count it. The
real gravity variation belongs in `dynamics`, acting on the airframe, and that
is where it lives.
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

# ICAO Doc 7488's nominal earth radius, the value the standard atmosphere's own
# geopotential conversion is defined with. It is NOT a physical earth radius --
# 6,356,766 m is neither the equatorial (6,378,137) nor the polar (6,356,752)
# figure but the radius that makes the ISA tables self-consistent at 45 deg
# latitude. Using a geodetic radius here would disagree with the tables this
# module reproduces.
R_EARTH_ISA = 6356766.0  # m


def geopotential(z: Array) -> Array:
    """Geopotential altitude from geometric altitude, both in metres.

    `H = R z / (R + z)`. Monotonic, exactly zero at zero, and smooth -- so it
    adds no kink for `jacfwd` to trip over and costs the integrator nothing.

    It always REDUCES the altitude: 30,000 ft geometric is 29,957.6 ft
    geopotential, so the air is warmer and denser than the unconverted call
    reported. That is the sign of the 0.159% bias it removes.
    """
    return R_EARTH_ISA * z / (R_EARTH_ISA + z)


def temperature(z: Array) -> Array:
    h = geopotential(z)
    return jnp.where(
        h < H_TROPOPAUSE,
        T0 + LAPSE * jnp.minimum(h, H_TROPOPAUSE),
        T_TROPOPAUSE,
    )


def pressure(z: Array) -> Array:
    h = geopotential(z)
    h_trop = jnp.minimum(h, H_TROPOPAUSE)
    p_lower = P0 * ((T0 + LAPSE * h_trop) / T0) ** (-G0 / (LAPSE * R_AIR))
    p_upper = P_TROPOPAUSE * jnp.exp(
        -G0 * jnp.maximum(h - H_TROPOPAUSE, 0.0) / (R_AIR * T_TROPOPAUSE)
    )
    return jnp.where(h < H_TROPOPAUSE, p_lower, p_upper)


def density(z: Array) -> Array:
    return pressure(z) / (R_AIR * temperature(z))


def speed_of_sound(z: Array) -> Array:
    return jnp.sqrt(GAMMA * R_AIR * temperature(z))
