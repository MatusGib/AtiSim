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
