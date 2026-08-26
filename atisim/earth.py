"""WGS-84 geodesy, gravity, and the frames of a rotating Earth.

Conventions, fixed everywhere in this module:

  ECEF     Earth-centred Earth-fixed. x through (0 lat, 0 lon), z through the
           north pole. Rotates with the Earth at OMEGA_WGS84.
  NED      Local North-East-Down at a stated geodetic latitude/longitude.
  Latitude is GEODETIC unless a name says `geocentric`. This is not a
           preference: JSBSim's local frame uses geodetic latitude, verified to
           machine precision, and using geocentric instead costs 65 m/s in
           v_north at 47 degrees.

EVERY CONSTANT HERE WAS RECOVERED FROM THE RUNNING JSBSim BINARY (v1.3.1,
GitHub build 1837, commit 3b25f25e) rather than transcribed. JSBSim ships as a
compiled wheel with no C++ source, so a transcription could not have been
checked. The recovery and its residuals are in
docs/superpowers/specs/2026-08-26-wgs84-earth-rotation-design.md section 2.
"""

import jax
import jax.numpy as jnp
from jax import Array

# -- Defining constants. WGS-84 defines `a` and `f`; everything else follows. --
A_WGS84 = 6378137.0             # m, semi-major axis
F_WGS84 = 1.0 / 298.257223563   # flattening
GM_WGS84 = 3.986004418e14       # m^3/s^2, geocentric gravitational constant
J2_WGS84 = 1.08262982e-3        # second dynamic form factor
OMEGA_WGS84 = 7.292115e-5       # rad/s, Earth rotation rate

# -- Derived. NEVER quote these from a table; they follow from the pair above. --
B_WGS84 = A_WGS84 * (1.0 - F_WGS84)          # m, semi-minor axis
E2_WGS84 = F_WGS84 * (2.0 - F_WGS84)         # first eccentricity squared
EP2_WGS84 = E2_WGS84 / (1.0 - E2_WGS84)      # second eccentricity squared

# Bowring's method, at a FIXED iteration count so it stays jittable and
# differentiable. Three reaches the float64 floor -- measured 3.5e-6 m at one
# iteration, 1.4e-8 at two, 3.7e-9 at three, against a 9.3e-10 m ECEF ulp.
BOWRING_ITERATIONS = 3


def geodetic_to_ecef(lat: Array, lon: Array, h: Array) -> Array:
    """Geodetic (rad, rad, m) -> ECEF (3,) m."""
    s, c = jnp.sin(lat), jnp.cos(lat)
    n = A_WGS84 / jnp.sqrt(1.0 - E2_WGS84 * s * s)
    return jnp.array([
        (n + h) * c * jnp.cos(lon),
        (n + h) * c * jnp.sin(lon),
        (n * (1.0 - E2_WGS84) + h) * s,
    ])


def ecef_to_geodetic(r: Array) -> tuple[Array, Array, Array]:
    """ECEF (3,) m -> (lat, lon, h) in rad, rad, m. Latitude is GEODETIC.

    Altitude uses h = p cos(lat) + z sin(lat) - a sqrt(1 - e2 sin^2 lat), which
    is exact and NON-SINGULAR. The textbook h = p/cos(lat) - N divides by zero
    at the poles; under jit that is a silent inf rather than an exception, so
    the singular form is not used at all rather than guarded.
    """
    x, y, z = r[0], r[1], r[2]
    lon = jnp.arctan2(y, x)
    p = jnp.sqrt(x * x + y * y)

    theta = jnp.arctan2(z * A_WGS84, p * B_WGS84)
    lat = jnp.arctan2(
        z + EP2_WGS84 * B_WGS84 * jnp.sin(theta) ** 3,
        p - E2_WGS84 * A_WGS84 * jnp.cos(theta) ** 3,
    )

    def refine(lat, _):
        n = A_WGS84 / jnp.sqrt(1.0 - E2_WGS84 * jnp.sin(lat) ** 2)
        return jnp.arctan2(z + E2_WGS84 * n * jnp.sin(lat), p), None

    lat, _ = jax.lax.scan(refine, lat, None, length=BOWRING_ITERATIONS - 1)

    s, c = jnp.sin(lat), jnp.cos(lat)
    h = p * c + z * s - A_WGS84 * jnp.sqrt(1.0 - E2_WGS84 * s * s)
    return lat, lon, h
