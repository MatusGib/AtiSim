"""A11: does the dt error reach the project's HEADLINE quantity, peak load factor?"""
import numpy as np
import jax, jax.numpy as jnp
import flightsim
from flightsim import integrate, trim, wind
from flightsim.aircraft import CRUISE, REGISTRY
from flightsim.dynamics import load_factor

AC = REGISTRY["boeing747"]
V, H = CRUISE["boeing747"]["airspeed"], CRUISE["boeing747"]["altitude"]
xt, _ = trim.trim(jnp.array(V), jnp.array(H), AC)
ctl = trim.trimmed_controls(xt[1], xt[2])

c = wind.PARKS_CASES["hannibal"]; sp = c["spacing"]
arr = wind.VortexArray(north=jnp.array([0.0, sp, 2*sp, 3*sp, 4*sp]),
                       down=jnp.array([-H]*5),
                       r0=jnp.array(c["r0"]), v0=jnp.array(c["v0"]))
model = wind.vortex_model(arr)
start_north = -3.0 * c["r0"]
T_END = 60.0


def peak_nz(dt):
    st = trim.trimmed_state(xt[0], jnp.array(V), jnp.array(H))
    st = st._replace(pos_ned=st.pos_ned.at[0].set(jnp.array(start_north)))
    n = int(round(T_END / dt))
    _, log = integrate.logged_rollout(integrate.init_sim(st, jax.random.PRNGKey(0)),
                                      ctl, jnp.array(dt), AC, n, wind_model=model)
    nz = jax.vmap(lambda s, w, og: load_factor(s, ctl, AC, w, og))(
        log.state, log.wind_ned, log.omega_gust)
    nz = np.asarray(nz)
    return nz.max(), nz.min(), nz


print("Parks 5-vortex array, 747 cruise, fixed controls, 60 s.")
print("`logged_rollout` + `dynamics.load_factor`, exactly as vortex_viz._measure does.")
print(f"{'dt':>10} {'n_z max':>14} {'n_z min':>14} {'peak-to-peak':>14} "
      f"{'err vs ref max':>16} {'err vs ref min':>16}")
ref_max, ref_min, _ = peak_nz(1.0 / 4096)
rows = []
for dt in (0.05, 0.02, 0.01, 0.005, 0.0025, 0.00125):
    mx, mn, _ = peak_nz(dt)
    print(f"{dt:>10.6g} {mx:>14.9f} {mn:>14.9f} {mx-mn:>14.9f} "
          f"{mx-ref_max:>+16.3e} {mn-ref_min:>+16.3e}")
print(f"{'1/4096':>10} {ref_max:>14.9f} {ref_min:>14.9f} {ref_max-ref_min:>14.9f}")

print()
print("Same, but for the SMOOTH lee-wave field (no kink) so the O(dt) hold is isolated:")
lw = wind.lee_wave_model(wind.LeeWave(w0=jnp.array(6.0),
                                      wavelength=jnp.array(wind.LEE_WAVE_WAVELENGTH),
                                      north=jnp.array(20000.0)))


def peak_nz_lw(dt):
    st = trim.trimmed_state(xt[0], jnp.array(V), jnp.array(H))
    n = int(round(T_END / dt))
    _, log = integrate.logged_rollout(integrate.init_sim(st, jax.random.PRNGKey(0)),
                                      ctl, jnp.array(dt), AC, n, wind_model=lw)
    nz = np.asarray(jax.vmap(lambda s, w, og: load_factor(s, ctl, AC, w, og))(
        log.state, log.wind_ned, log.omega_gust))
    return nz.max(), nz.min()


rmx, rmn = peak_nz_lw(1.0 / 4096)
print(f"{'dt':>10} {'n_z max':>14} {'n_z min':>14} {'err max':>14} {'err min':>14}")
for dt in (0.05, 0.02, 0.01, 0.005):
    mx, mn = peak_nz_lw(dt)
    print(f"{dt:>10.6g} {mx:>14.9f} {mn:>14.9f} {mx-rmx:>+14.3e} {mn-rmn:>+14.3e}")
print(f"{'1/4096':>10} {rmx:>14.9f} {rmn:>14.9f}")
