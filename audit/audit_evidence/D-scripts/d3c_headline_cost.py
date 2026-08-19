"""D3c: what the wind hold costs the project's four headline encounters."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

import flightsim  # noqa: F401
from flightsim import dynamics, integrate, trim, wind
from flightsim.aircraft import CRUISE, REGISTRY
from flightsim.state import State, quat_to_euler
from flightsim.units import RAD2DEG
import wcommon as W
from d3_rk4_seam import AC, V, H, step_perstage


def traj(sim, controls, dt, n, ac, model, per_stage):
    stepper = step_perstage if per_stage else (
        lambda s, c, d, a, wm: integrate.step(s, c, d, a, wind_model=wm))

    def body(carry, _):
        c = stepper(carry, controls, dt, ac, model)
        return c, c.state

    return jax.lax.scan(body, sim, None, length=n)[1]


traj_j = jax.jit(traj, static_argnames=("n", "model", "per_stage"))


def measure(hist, controls, ac, model, dt):
    def one(pos, vel, quat, om):
        s = State(pos_ned=pos, vel_body=vel, quat=quat, omega=om)
        w, og, _, _ = model(wind.zero_wind_state(), s, jax.random.PRNGKey(0), jnp.array(dt))
        _, th, _ = quat_to_euler(quat)
        return jnp.array([th, dynamics.load_factor(s, controls, ac, w, og), -pos[2]])
    return np.asarray(jax.vmap(one)(hist.pos_ned, hist.vel_body, hist.quat, hist.omega))


def scenario(label, field, start, seconds, dt, ac=AC, airspeed=V, altitude=H):
    model = wind.field_model(field)
    x, _ = trim.trim(jnp.array(airspeed), jnp.array(altitude), ac)
    controls = trim.trimmed_controls(x[1], x[2])
    st = trim.trimmed_state(jnp.array(float(x[0])), jnp.array(airspeed), jnp.array(altitude))
    st = st._replace(pos_ned=jnp.array([start, 0.0, -altitude]))
    sim = integrate.init_sim(st, jax.random.PRNGKey(0))
    n = int(round(seconds / dt))
    h = traj_j(sim, controls, jnp.array(dt), n, ac, model, False)
    p = traj_j(sim, controls, jnp.array(dt), n, ac, model, True)
    mh = measure(h, controls, ac, model, dt)
    mp = measure(p, controls, ac, model, dt)
    exc = mh.max(axis=0) - mh.min(axis=0)     # the run's own excursion in each channel
    dif = np.abs(mh - mp).max(axis=0)
    dpos = np.linalg.norm(np.asarray(h.pos_ned) - np.asarray(p.pos_ned), axis=1)
    print(f"\n{label}   dt = {dt} s, {n} steps, {seconds:.1f} s")
    print(f"    max |d pos|            {dpos.max():.4e} m")
    print(f"    theta   excursion {exc[0]*RAD2DEG:9.4f} deg   hold error "
          f"{dif[0]*RAD2DEG:.4e} deg   ({dif[0]/max(exc[0],1e-300):.3e} of it)")
    print(f"    n_z     excursion {exc[1]:9.4f} g     hold error "
          f"{dif[1]:.4e} g     ({dif[1]/max(exc[1],1e-300):.3e} of it)")
    print(f"    alt     excursion {exc[2]:9.4f} m     hold error "
          f"{dif[2]:.4e} m     ({dif[2]/max(exc[2],1e-300):.3e} of it)")
    # the Fig. 8 coordinate itself
    d_theta_h = (mh[:, 0].max() - mh[:, 0].min()) * RAD2DEG
    d_theta_p = (mp[:, 0].max() - mp[:, 0].min()) * RAD2DEG
    dn_h = (mh[:, 1] - mh[0, 1]).min()
    dn_p = (mp[:, 1] - mp[0, 1]).min()
    print(f"    Fig.8   d(theta)  held {d_theta_h:8.5f}  per-stage {d_theta_p:8.5f} deg"
          f"   -> {abs(d_theta_h-d_theta_p)/max(abs(d_theta_h),1e-300):.3e} rel")
    print(f"    Fig.8   d(n)      held {dn_h:+8.5f}  per-stage {dn_p:+8.5f} g"
          f"   -> {abs(dn_h-dn_p)/max(abs(dn_h),1e-300):.3e} rel")


print("=" * 78)
print("D3c  HOLD vs PER-STAGE ON THE PROJECT'S OWN ENCOUNTERS")
print("=" * 78)

LEAD = 40.0 * W.R0
SEC = (W.SPACING + LEAD + 6.0 * W.R0) / V
for dt in (0.02, 0.01):
    scenario(f"Parks hannibal vortex (scripts/vortex.py, lead 40 r0)",
             W.vortex_field, -LEAD, SEC, dt)

RAD = W.UPDRAFT_RADIUS
for dt in (0.02, 0.01):
    scenario("Wingrove updraft, sharpness 6 (scripts/vortex.py)",
             W.updraft_field, -2.0 * RAD, 4.0 * RAD / V, dt)

LAM = 25000.0
for dt in (0.02, 0.01):
    scenario("Doyle lee wave, 3 wavelengths (scripts/leewave.py)",
             W.lee_field, -0.25 * LAM, 3.0 * LAM / V, dt)

# microburst runs at 300 m AGL and much lower speed geometry
Vm = V
peak_r = 1.1212 * W.MB_R
for dt in (0.02, 0.01):
    scenario("Oseguera-Bowles microburst, 300 m AGL (scripts/microburst.py)",
             W.mb_field, -3.0 * peak_r, 6.0 * peak_r / Vm, dt, altitude=300.0)
