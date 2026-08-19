"""Shared field constructors for the Agent-D wind audit. Read-only w.r.t. flightsim."""

import numpy as np
import jax
import jax.numpy as jnp

import flightsim  # noqa: F401  enables x64
from flightsim import wind
from flightsim.units import FT2M

H_CRUISE = 12192.0  # m, the 747 CR-2144 FC9 cruise altitude used by scripts/vortex.py
V_CRUISE = 236.0    # m/s, approximate; exact value pulled from aircraft.CRUISE when needed

# --- Parks Hannibal, exactly as scripts/vortex.py builds it -------------------
_case = wind.PARKS_CASES["hannibal"]
R0 = _case["r0"]
V0 = _case["v0"]
SPACING = _case["spacing"]
VORTEX = wind.VortexArray(
    north=jnp.array([0.0, SPACING]),
    down=jnp.array([-H_CRUISE, -H_CRUISE]),
    r0=jnp.array(R0),
    v0=jnp.array(V0),
)
vortex_field = lambda p: wind.vortex_wind(p, VORTEX)  # noqa: E731

# single-core version, for the hand-derived Jacobian algebra
VORTEX1 = wind.VortexArray(
    north=jnp.array([0.0]), down=jnp.array([-H_CRUISE]),
    r0=jnp.array(R0), v0=jnp.array(V0),
)
vortex1_field = lambda p: wind.vortex_wind(p, VORTEX1)  # noqa: E731

# --- Wingrove & Bach updraft, as scripts/vortex.py builds it ------------------
UPDRAFT_RADIUS = 0.5 * wind.UPDRAFT_SECONDS * V_CRUISE
SHARPNESS = 6.0
COLUMN = wind.UpdraftColumn(
    north=jnp.array(0.0), east=jnp.array(0.0),
    w0=jnp.array(wind.UPDRAFT_W0), radius=jnp.array(UPDRAFT_RADIUS),
    sharpness=jnp.array(SHARPNESS),
)
updraft_field = lambda p: wind.updraft_wind(p, COLUMN)  # noqa: E731

# --- Doyle lee wave, as scripts/leewave.py builds it --------------------------
WAVE = wind.LeeWave(
    w0=jnp.array(wind.LEE_WAVE_AMPLITUDE["south"]),
    wavelength=jnp.array(wind.LEE_WAVE_WAVELENGTH),
    north=jnp.array(0.0),
)
lee_field = lambda p: wind.lee_wave_wind(p, WAVE)  # noqa: E731

# --- Oseguera & Bowles microburst, as scripts/microburst.py builds it ---------
MB_UMAX, MB_R, MB_ZM = 19.03, 1000.0, 150.0
BURST = wind.microburst(u_max=MB_UMAX, radius=MB_R, z_m=MB_ZM)
mb_field = lambda p: wind.microburst_wind(p, BURST)  # noqa: E731


def random_quats(key, n):
    """Uniform random unit quaternions (Shoemake), shape (n, 4)."""
    u = jax.random.uniform(key, (n, 3))
    q = jnp.stack([
        jnp.sqrt(1 - u[:, 0]) * jnp.sin(2 * jnp.pi * u[:, 1]),
        jnp.sqrt(1 - u[:, 0]) * jnp.cos(2 * jnp.pi * u[:, 1]),
        jnp.sqrt(u[:, 0]) * jnp.sin(2 * jnp.pi * u[:, 2]),
        jnp.sqrt(u[:, 0]) * jnp.cos(2 * jnp.pi * u[:, 2]),
    ], axis=1)
    # Shoemake gives (x,y,z,w); flightsim wants (w,x,y,z)
    return jnp.concatenate([q[:, 3:4], q[:, 0:3]], axis=1)


def sample_positions(key, n, kind):
    """Positions covering the regime each field is actually flown in."""
    k1, k2, k3 = jax.random.split(key, 3)
    if kind == "vortex":
        north = jax.random.uniform(k1, (n,), minval=-4 * R0, maxval=SPACING + 4 * R0)
        east = jax.random.uniform(k2, (n,), minval=-500.0, maxval=500.0)
        down = -H_CRUISE + jax.random.uniform(k3, (n,), minval=-3 * R0, maxval=3 * R0)
    elif kind == "updraft":
        # annulus 0.4R - 1.8R: sharpness 6 makes the field numerically flat
        # outside that, so a uniform box would sample mostly exact zeros.
        rho = jax.random.uniform(k1, (n,), minval=0.4 * UPDRAFT_RADIUS,
                                 maxval=1.8 * UPDRAFT_RADIUS)
        th = jax.random.uniform(k2, (n,), minval=0.0, maxval=2 * jnp.pi)
        north, east = rho * jnp.cos(th), rho * jnp.sin(th)
        down = -H_CRUISE + jax.random.uniform(k3, (n,), minval=-1000.0, maxval=1000.0)
    elif kind == "lee":
        north = jax.random.uniform(k1, (n,), minval=-2 * wind.LEE_WAVE_WAVELENGTH,
                                   maxval=2 * wind.LEE_WAVE_WAVELENGTH)
        east = jax.random.uniform(k2, (n,), minval=-5000.0, maxval=5000.0)
        down = -H_CRUISE + jax.random.uniform(k3, (n,), minval=-1000.0, maxval=1000.0)
    elif kind == "microburst":
        north = jax.random.uniform(k1, (n,), minval=-3 * MB_R, maxval=3 * MB_R)
        east = jax.random.uniform(k2, (n,), minval=-3 * MB_R, maxval=3 * MB_R)
        down = -jax.random.uniform(k3, (n,), minval=20.0, maxval=900.0)
    else:
        raise ValueError(kind)
    return jnp.stack([north, east, down], axis=1)


FIELDS = {
    "vortex(hannibal, 2 cores)": ("vortex", vortex_field),
    "updraft(w0=24.4, R=2360, p=6)": ("updraft", updraft_field),
    "lee_wave(w0=6, L=25km)": ("lee", lee_field),
    "microburst(19.03, R=1000, zm=150)": ("microburst", mb_field),
}


def relerr(a, b, floor):
    """Relative error, floored so a near-zero reference does not blow up."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    num = np.linalg.norm(a - b, axis=-1)
    den = np.maximum(np.linalg.norm(b, axis=-1), floor)
    return num / den
