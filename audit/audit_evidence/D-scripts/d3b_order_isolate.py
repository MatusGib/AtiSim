"""D3b: isolate the sources of order loss in the RK4 seam study.

Four right-hand sides of increasing non-smoothness, so the wind hold's order
penalty can be separated from kinks that are already in the plant.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

import flightsim  # noqa: F401
from flightsim import aero, integrate, trim, wind
from flightsim.aircraft import CRUISE, REGISTRY
from flightsim.atmosphere import speed_of_sound
import wcommon as W
from d3_rk4_seam import AC, V, H, MODEL, run, setup, flatten, step_perstage  # noqa

# --- is the wave-drag kink active at cruise? ---------------------------------
x, _ = trim.trim(jnp.array(V), jnp.array(H), AC)
alpha = float(x[0])
st = trim.trimmed_state(jnp.array(alpha), jnp.array(V), jnp.array(H))
CL, *_ = aero.coefficients(st.vel_body, jnp.zeros(3),
                           trim.trimmed_controls(x[1], x[2]), AC, speed_of_sound(jnp.array(H)))
m = V / float(speed_of_sound(jnp.array(H)))
m_crit = float(aero.drag_divergence_mach(CL, AC)) - aero._MDD_OFFSET
print(f"cruise Mach {m:.5f}   m_crit {m_crit:.5f}   "
      f"wave drag {'ACTIVE (max() kink is in the C3 branch)' if m > m_crit else 'inactive'}")
print(f"cruise altitude {H} m vs tropopause 11000 m -> "
      f"{'above, atmosphere is the smooth upper branch' if H > 11000 else 'BELOW'}")
print()

# --- does the actual traverse cross the Rankine core boundary r = r0? --------
LEAD = 40.0 * W.R0
SECONDS = (W.SPACING + LEAD + 6.0 * W.R0) / V
sim1, c1 = setup(-LEAD)


@partial(jax.jit, static_argnames=("n_steps",))
def traj(sim, controls, dt, n_steps):
    def body(carry, _):
        c = integrate.step(carry, controls, dt, AC, wind_model=MODEL)
        return c, c.state
    return jax.lax.scan(body, sim, None, length=n_steps)[1]


hist = traj(sim1, c1, jnp.array(0.02), int(round(SECONDS / 0.02)))
p = np.asarray(hist.pos_ned)
for i, nc in enumerate([0.0, W.SPACING]):
    r = np.hypot(p[:, 0] - nc, (-W.H_CRUISE) - p[:, 2])
    inside = r < W.R0
    print(f"core {i}: min r/r0 = {r.min()/W.R0:.4f}, samples inside core = {inside.sum()}"
          f"  -> {'CROSSES the C1 kink twice' if inside.any() else 'stays outside'}")
print()

FIELDS = {
    "zero wind": None,
    "lee wave (C-inf)": wind.field_model(W.lee_field),
    "updraft p=6 (C-inf)": wind.field_model(W.updraft_field),
    "vortex, path THROUGH the core (C1 kink)": MODEL,
    "vortex, path 2 r0 ABOVE the core (analytic)": MODEL,
}


def order_table(label, model, start_north, start_alt, T, ns):
    sim0, controls = setup(start_north, altitude=start_alt)
    mdl = wind.zero_wind if model is None else model
    nref = 1 << 18
    xref = flatten(run(sim0, controls, jnp.array(T / nref), AC, nref, mdl, True))[:3]
    xref2 = flatten(run(sim0, controls, jnp.array(T / (nref // 2)), AC, nref // 2, mdl, True))[:3]
    print(f"--- {label}")
    print(f"    reference dt = {T/nref:.3e} s, self-consistency "
          f"{np.linalg.norm(xref-xref2):.3e} m")
    print(f"    {'n':>7s} {'dt':>10s} | {'HELD err (m)':>14s} {'p':>6s} | "
          f"{'PER-STAGE err':>14s} {'p':>6s}")
    ph = pp = None
    for n in ns:
        dt = T / n
        eh = np.linalg.norm(flatten(run(sim0, controls, jnp.array(dt), AC, n, mdl, False))[:3] - xref)
        ep = np.linalg.norm(flatten(run(sim0, controls, jnp.array(dt), AC, n, mdl, True))[:3] - xref)
        oh = np.log2(ph / eh) if ph else float("nan")
        op = np.log2(pp / ep) if pp else float("nan")
        print(f"    {n:7d} {dt:10.5f} | {eh:14.4e} {oh:6.2f} | {ep:14.4e} {op:6.2f}")
        ph, pp = eh, ep
    print()


T = 6.0 * W.R0 / V
NS = [32, 64, 128, 256, 512, 1024]
order_table("zero wind (integrator + plant baseline)", None, -2 * W.R0, H, T, NS)
order_table("lee wave, C-infinity field", FIELDS["lee wave (C-inf)"],
            -0.25 * 25000.0, H, 3.0, NS)
order_table("updraft sharpness 6, C-infinity field", FIELDS["updraft p=6 (C-inf)"],
            -1.2 * W.UPDRAFT_RADIUS, H, 8.0, NS)
order_table("vortex, straight THROUGH the core (C1 kink on the path)",
            MODEL, -2 * W.R0, H, T, NS)
order_table("vortex, offset 2 r0 above the cores (field analytic on the path)",
            MODEL, -2 * W.R0, H + 2 * W.R0, T, NS)
