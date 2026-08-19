"""A9: how big is the O(dt) wind-hold error at the timesteps the project actually uses?

scripts/: fly.py dt=0.02, leewave.py/microburst.py/vortex.py dt=0.01,
checkpoint.py dt=0.02.  Measure the wind-hold error at those dt on the real
fields, over the run lengths those scripts use.
"""
import numpy as np
import jax, jax.numpy as jnp
import flightsim
from flightsim import integrate, trim, wind
from flightsim.aircraft import CRUISE, REGISTRY
from flightsim.units import FT2M

AC = REGISTRY["boeing747"]
V, H = CRUISE["boeing747"]["airspeed"], CRUISE["boeing747"]["altitude"]
xt, _ = trim.trim(jnp.array(V), jnp.array(H), AC)


def fly(dt, t_end, model, start_north=0.0):
    st = trim.trimmed_state(xt[0], jnp.array(V), jnp.array(H))
    st = st._replace(pos_ned=st.pos_ned.at[0].set(jnp.asarray(start_north, float)))
    ctl = trim.trimmed_controls(xt[1], xt[2])
    n = int(round(t_end / dt))
    fin, _ = integrate.rollout(integrate.init_sim(st, jax.random.PRNGKey(0)),
                               ctl, jnp.array(dt), AC, n, wind_model=model)
    s = fin.state
    return dict(pos=np.asarray(s.pos_ned), vel=np.asarray(s.vel_body),
                quat=np.asarray(s.quat), om=np.asarray(s.omega))


CASES = {}
# lee wave, as scripts/leewave.py builds it
CASES["lee wave (Doyle, w0=6 m/s, 25 km)"] = (
    wind.lee_wave_model(wind.LeeWave(w0=jnp.array(6.0),
                                     wavelength=jnp.array(wind.LEE_WAVE_WAVELENGTH),
                                     north=jnp.array(20000.0))), 0.0, 60.0)
# Parks vortex array, as scripts/vortex.py builds it
c = wind.PARKS_CASES["hannibal"]
sp = c["spacing"]
CASES["Parks vortex array (hannibal, 5 cores)"] = (
    wind.vortex_model(wind.VortexArray(
        north=jnp.array([0.0, sp, 2*sp, 3*sp, 4*sp]),
        down=jnp.array([-H]*5), r0=jnp.array(c["r0"]), v0=jnp.array(c["v0"]))),
    -3.0*c["r0"], 60.0)
# updraft column
CASES["updraft column (Wingrove & Bach)"] = (
    wind.updraft_model(wind.UpdraftColumn(
        north=jnp.array(6000.0), east=jnp.array(0.0),
        w0=jnp.array(wind.UPDRAFT_W0), radius=jnp.array(2400.0),
        sharpness=jnp.array(4.0))), 0.0, 60.0)

print("Reference = the same field at dt = 1/4096 s. Errors are the FULL-SIM error")
print("of the state at the end of the run, at the dt the project's scripts use.\n")
for label, (model, north0, t_end) in CASES.items():
    ref = fly(1.0/4096, t_end, model, north0)
    print(f"--- {label}, {t_end:g} s ---")
    print(f"{'dt':>10} {'|dpos| [m]':>14} {'|dvel| [m/s]':>14} {'|dquat|':>12} "
          f"{'|domega|':>12}")
    for dt in (0.02, 0.01, 0.005, 0.0025):
        g = fly(dt, t_end, model, north0)
        print(f"{dt:>10.5g} {np.linalg.norm(g['pos']-ref['pos']):>14.5e} "
              f"{np.linalg.norm(g['vel']-ref['vel']):>14.5e} "
              f"{np.linalg.norm(g['quat']-ref['quat']):>12.5e} "
              f"{np.linalg.norm(g['om']-ref['om']):>12.5e}")
    print(f"   reference final pos = {ref['pos']}")
    # still air comparison over the same horizon, same dts
    print()
