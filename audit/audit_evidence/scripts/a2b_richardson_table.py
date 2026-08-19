"""A2b: error at each dt against the RICHARDSON-EXTRAPOLATED exact solution."""
import numpy as np
import jax, jax.numpy as jnp
import flightsim
from flightsim import integrate, trim, wind
from flightsim.aircraft import CRUISE, REGISTRY

AC = REGISTRY["boeing747"]
V, H = CRUISE["boeing747"]["airspeed"], CRUISE["boeing747"]["altitude"]
T_END = 4.0
xt, _ = trim.trim(jnp.array(V), jnp.array(H), AC)


def run(dt, model=None, north0=0.0, de=0.02):
    st = trim.trimmed_state(xt[0], jnp.array(V), jnp.array(H))
    st = st._replace(pos_ned=st.pos_ned.at[0].set(jnp.asarray(north0, float)))
    ct = trim.trimmed_controls(xt[1] + de, xt[2])
    m = wind.zero_wind if model is None else model
    f, _ = integrate.rollout(integrate.init_sim(st, jax.random.PRNGKey(0)),
                             ct, jnp.array(dt), AC, int(round(T_END/dt)), wind_model=m)
    return np.asarray(f.state.pos_ned)


DTS = [1/4, 1/8, 1/16, 1/32, 1/64, 1/128, 1/256, 1/512, 1/1024]
FLOOR = 7e-11  # the project's stated round-off floor at 40,000 ft


def report(name, p_assumed, **kw):
    F = {d: run(d, **kw) for d in DTS}
    # Richardson from the three finest grids
    f1, f2, f3 = F[DTS[-3]], F[DTS[-2]], F[DTS[-1]]
    d12, d23 = f1 - f2, f2 - f3
    with np.errstate(all="ignore"):
        p_comp = np.log(np.abs(d12 / d23)) / np.log(2.0)
    p = p_assumed
    exact = f3 + (f3 - f2) / (2.0 ** p - 1.0)
    print(f"\n{'='*76}\n{name}")
    print(f"  three finest grids 1/{1/DTS[-3]:.0f}, 1/{1/DTS[-2]:.0f}, 1/{1/DTS[-1]:.0f}")
    print(f"  componentwise observed p from that triple: {np.round(p_comp, 4)}")
    print(f"  extrapolation uses p = {p}")
    print(f"  RICHARDSON-EXTRAPOLATED EXACT pos_ned = "
          f"[{exact[0]:.10f}, {exact[1]:.10f}, {exact[2]:.10f}]")
    print(f"  {'dt':>10} {'|pos - exact| [m]':>20} {'order':>9} {'x above 7e-11 floor':>22}")
    prev = None
    for d in DTS:
        e = float(np.linalg.norm(F[d] - exact))
        o = "" if prev is None else f"{np.log(prev[0]/e)/np.log(prev[1]/d):.4f}"
        print(f"  {d:>10.6g} {e:>20.6e} {o:>9} {e/FLOOR:>22.4g}")
        prev = (e, d)
    return F


report("(a) STILL AIR   [RK4 should be p=4]", 4.0)

wave = wind.LeeWave(w0=jnp.array(25.0), wavelength=jnp.array(1200.0), north=jnp.array(0.0))
report("(b) SMOOTH LeeWave field, w0=25 m/s, lambda=1200 m   [wind held -> p=1]", 1.0,
       model=wind.field_model(lambda p: wind.lee_wave_wind(p, wave)))

c = wind.PARKS_CASES["hannibal"]; r0 = c["r0"]
arr = wind.VortexArray(north=jnp.array([0.0]), down=jnp.array([-H]),
                       r0=jnp.array(r0), v0=jnp.array(c["v0"]))
print("\n\n(c) RANKINE CORE: p is assumed 1 ONLY to show the extrapolation is unusable.")
report("(c) RANKINE VORTEX CORE CROSSING   [no order exists]", 1.0,
       model=wind.field_model(lambda p: wind.vortex_wind(p, arr)),
       north0=-2.0*r0, de=0.0)
print("\nAlso extrapolate (c) with p=2 and p=4 to show the answer moves with the")
print("assumed order -- the mark of an extrapolation that means nothing:")
F = {d: run(d, wind.field_model(lambda p: wind.vortex_wind(p, arr)), -2.0*r0, 0.0)
     for d in DTS[-3:]}
f2, f3 = F[DTS[-2]], F[DTS[-1]]
for p in (0.5, 1.0, 2.0, 4.0):
    ex = f3 + (f3 - f2) / (2.0 ** p - 1.0)
    print(f"   p={p}: exact_pD = {ex[2]:.6f} m   exact_pN = {ex[0]:.6f} m")
