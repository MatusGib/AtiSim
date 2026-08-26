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

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from atisim.atmosphere import G0

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


def ecef_to_ned_matrix(lat: Array, lon: Array) -> Array:
    """ECEF -> local NED rotation. v_ned = ecef_to_ned_matrix(lat, lon) @ v_ecef.

    `lat` is GEODETIC. Verified against JSBSim: rotating its ECEF velocity by
    this matrix at the geodetic latitude reproduces its own
    velocities/v-north|east|down to machine precision, and at the geocentric
    latitude it does not.
    """
    sf, cf = jnp.sin(lat), jnp.cos(lat)
    sl, cl = jnp.sin(lon), jnp.cos(lon)
    return jnp.array([
        [-sf * cl, -sf * sl, cf],
        [-sl, cl, jnp.zeros_like(lat)],
        [-cf * cl, -cf * sl, -sf],
    ])


class EarthModel(NamedTuple):
    """Which Earth the plant is flying over.

    Carried as a STATIC argument through `integrate.step`, so every branch below
    resolves at trace time and costs nothing under jit. That is also why the
    fields are plain str/float rather than arrays -- a NamedTuple of those is
    hashable, and a static argument must be.

    `gravity` is one of "j2", "inverse_square", "constant".
    """

    gravity: str
    rotation_rate: float
    ellipsoidal: bool


WGS84_J2 = EarthModel("j2", OMEGA_WGS84, True)
WGS84_INVERSE_SQUARE = EarthModel("inverse_square", OMEGA_WGS84, True)
# FLAT: non-rotating, spherical, constant g along the local vertical. It is a
# CONFIGURATION of this one plant, not a second implementation retained
# alongside. It does NOT reproduce the pre-Earth-model trajectories
# bit-for-bit -- design doc section 8 explains why: an ECEF-accumulated state
# cannot be bit-identical even where the physics agrees. The round-off floor
# this costs is NOT YET measured. ASSUMPTIONS.md F4 covers only
# dt/discretisation round-off today; Task 14 re-measures it and folds the FLAT
# floor in. Until then this claim rests on the design doc's reasoning rather
# than on a recorded number.
FLAT = EarthModel("constant", 0.0, False)


def gravitation(r_ecef: Array, model: EarthModel) -> Array:
    """Gravitational acceleration in the ECEF frame, (3,) m/s^2.

    GRAVITATION, NOT APPARENT GRAVITY. The centrifugal term belongs to the
    equation of motion in `dynamics.derivatives` and is not folded in here.
    JSBSim makes the same split, which is why it reports 9.8142 m/s^2 at the
    equator rather than 9.7803.
    """
    radius = jnp.linalg.norm(r_ecef)
    direction = r_ecef / radius

    if model.gravity == "constant":
        return -G0 * direction

    magnitude = GM_WGS84 / (radius * radius)
    if model.gravity == "inverse_square":
        return -magnitude * direction

    if model.gravity != "j2":
        raise ValueError(f"unknown gravity model {model.gravity!r}")

    # J2 zonal harmonic. `sin_gc` is the sine of the GEOCENTRIC latitude, which
    # is what a spherical-harmonic expansion is written in -- unlike the local
    # frame, which is geodetic.
    sin_gc = r_ecef[2] / radius
    common = 1.5 * J2_WGS84 * (A_WGS84 / radius) ** 2
    horizontal = 1.0 + common * (1.0 - 5.0 * sin_gc * sin_gc)
    vertical = 1.0 + common * (3.0 - 5.0 * sin_gc * sin_gc)
    return -magnitude * jnp.array([
        direction[0] * horizontal,
        direction[1] * horizontal,
        direction[2] * vertical,
    ])


class Anchor(NamedTuple):
    """A run's geodetic origin: where local NED is pinned.

    A traced pytree, not a static argument, because it carries arrays. It is
    constant for the life of a run.

    THERE IS DELIBERATELY NO DEFAULT ANCHOR. A default would silently pick a
    latitude, and latitude now changes the answer -- the Coriolis term, the
    trimmed bank angle and the gravity magnitude all depend on it. Every caller
    states where it is flying.
    """

    lat: Array
    lon: Array
    h: Array
    r_ecef: Array   # (3,)
    T_e2l: Array    # (3,3), ECEF -> NED at (lat, lon)


def anchor_at(lat: Array, lon: Array, h: Array) -> Anchor:
    lat, lon, h = jnp.asarray(lat), jnp.asarray(lon), jnp.asarray(h)
    return Anchor(
        lat=lat, lon=lon, h=h,
        r_ecef=geodetic_to_ecef(lat, lon, h),
        T_e2l=ecef_to_ned_matrix(lat, lon),
    )
