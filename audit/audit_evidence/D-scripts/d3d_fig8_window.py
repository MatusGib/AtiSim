"""D3d: the hold's cost in the WINDOWED Fig.8 coordinate that scripts/vortex.py prints."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import jax
import jax.numpy as jnp

import flightsim  # noqa: F401
from flightsim import dynamics, integrate, trim, wind
from flightsim.state import State, quat_to_euler
from flightsim.units import RAD2DEG
import wcommon as W
from d3_rk4_seam import AC, V, H, step_perstage
from d3c_headline_cost import traj_j, measure


def fig8(field, start, seconds, dt, window, altitude=H, per_stage=False):
    model = wind.field_model(field)
    x, _ = trim.trim(jnp.array(V), jnp.array(altitude), AC)
    ctrl = trim.trimmed_controls(x[1], x[2])
    st = trim.trimmed_state(jnp.array(float(x[0])), jnp.array(V), jnp.array(altitude))
    st = st._replace(pos_ned=jnp.array([start, 0.0, -altitude]))
    sim = integrate.init_sim(st, jax.random.PRNGKey(0))
    n = int(round(seconds / dt))
    hist = traj_j(sim, ctrl, jnp.array(dt), n, AC, model, per_stage)
    m = measure(hist, ctrl, AC, model, dt)     # columns: theta, n_z, altitude
    north = np.asarray(hist.pos_ned)[:, 0]
    win = (north >= window[0]) & (north <= window[1])
    th = m[win, 0]
    return (float(th.max() - th.min()) * RAD2DEG, float(m[win, 1].min() - m[0, 1]),
            int(win.sum()))


LEAD = 40.0 * W.R0
SEC = (W.SPACING + LEAD + 6 * W.R0) / V
print("Parks hannibal, window = first core (north in [-r0, r0]), "
      "as scripts/vortex.py declares")
print(f"{'dt':>6s} {'n in window':>12s} | {'d(theta) held':>14s} {'per-stage':>11s} "
      f"{'rel':>10s} | {'d(n) held':>11s} {'per-stage':>11s} {'rel':>10s}")
for dt in (0.02, 0.01):
    a = fig8(W.vortex_field, -LEAD, SEC, dt, (-W.R0, W.R0), per_stage=False)
    b = fig8(W.vortex_field, -LEAD, SEC, dt, (-W.R0, W.R0), per_stage=True)
    print(f"{dt:6.3f} {a[2]:12d} | {a[0]:14.6f} {b[0]:11.6f} "
          f"{abs(a[0]-b[0])/abs(a[0]):10.3e} | {a[1]:11.6f} {b[1]:11.6f} "
          f"{abs(a[1]-b[1])/abs(a[1]):10.3e}")

RAD = W.UPDRAFT_RADIUS
print("\nWingrove updraft, window = column (north in [-radius, radius])")
for dt in (0.02, 0.01):
    a = fig8(W.updraft_field, -2 * RAD, 4 * RAD / V, dt, (-RAD, RAD), per_stage=False)
    b = fig8(W.updraft_field, -2 * RAD, 4 * RAD / V, dt, (-RAD, RAD), per_stage=True)
    print(f"{dt:6.3f} {a[2]:12d} | {a[0]:14.6f} {b[0]:11.6f} "
          f"{abs(a[0]-b[0])/abs(a[0]):10.3e} | {a[1]:11.6f} {b[1]:11.6f} "
          f"{abs(a[1]-b[1])/abs(a[1]):10.3e}")
