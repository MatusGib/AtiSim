"""Shared helpers for the falsification attacks."""
import sys
sys.path.insert(0, r"C:\Users\mateusz\UROP\Claude_Flight_Sim")

import flightsim  # noqa: F401  enables x64
import jax
import jax.numpy as jnp
import numpy as np

from flightsim import integrate, trim
from flightsim.aircraft import CRUISE, REGISTRY
from flightsim.atmosphere import G0
from flightsim.state import Controls, State, euler_to_quat, quat_to_dcm

NAMES = ["boeing747", "boeing747_approach", "cherokee", "cessna172"]


def ctrl(de=0.0, da=0.0, dr=0.0, thr=0.0):
    return Controls(
        elevator=jnp.array(float(de)),
        aileron=jnp.array(float(da)),
        rudder=jnp.array(float(dr)),
        throttle=jnp.array(float(thr)),
    )


def energy(hist, ac):
    """Total mechanical energy per unit mass pieces, in J."""
    m = float(ac.mass)
    I = np.asarray(ac.inertia)
    v = np.asarray(hist.vel_body)
    om = np.asarray(hist.omega)
    h = -np.asarray(hist.pos_ned)[:, 2]
    ke = 0.5 * m * (v * v).sum(axis=1)
    rke = 0.5 * np.einsum("ni,ij,nj->n", om, I, om)
    pe = m * G0 * h
    return ke + rke + pe, ke, rke, pe


def run(state0, controls, ac, dt, n, wind_model=None):
    sim = integrate.init_sim(state0, jax.random.PRNGKey(0))
    kw = {} if wind_model is None else {"wind_model": wind_model}
    return integrate.rollout(sim, controls, jnp.array(dt), ac, n, **kw)


def trim_state(name, V=None, h=None):
    ac = REGISTRY[name]
    V = CRUISE[name]["airspeed"] if V is None else V
    h = CRUISE[name]["altitude"] if h is None else h
    x, r = trim.trim(jnp.array(float(V)), jnp.array(float(h)), ac)
    st = trim.trimmed_state(x[0], jnp.array(float(V)), jnp.array(float(h)))
    return ac, st, x, r
