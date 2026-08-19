"""A6b: float32 vs float64 on the SAME rollout, and float32 quaternion norm."""
import numpy as np
import jax, jax.numpy as jnp
import flightsim
from flightsim import integrate, trim
from flightsim.aircraft import CRUISE, REGISTRY

AC = REGISTRY["boeing747"]
V, H = CRUISE["boeing747"]["airspeed"], CRUISE["boeing747"]["altitude"]
x64, _ = trim.trim(jnp.array(V), jnp.array(H), AC)
st64 = trim.trimmed_state(x64[0], jnp.array(V), jnp.array(H))
ct64 = trim.trimmed_controls(x64[1] + 0.02, x64[2])
MARKS = [1000, 10000, 50000, 100000]
out64 = {}
sim = integrate.init_sim(st64, jax.random.PRNGKey(0))
prev = 0
for n in MARKS:
    fin, _ = integrate.rollout(sim, ct64, jnp.array(0.02), AC, n - prev)
    sim = fin; prev = n
    out64[n] = (np.asarray(fin.state.pos_ned).copy(),
                np.asarray(fin.state.vel_body).copy(),
                np.asarray(fin.state.quat).copy(),
                np.asarray(fin.state.omega).copy())

jax.config.update("jax_enable_x64", False)
ac32 = jax.tree.map(lambda v: jnp.asarray(np.asarray(v), jnp.float32), AC)
# use the SAME initial condition, cast down, so the only difference is precision
st32 = jax.tree.map(lambda v: jnp.asarray(np.asarray(v), jnp.float32), st64)
ct32 = jax.tree.map(lambda v: jnp.asarray(np.asarray(v), jnp.float32), ct64)
sim = integrate.init_sim(st32, jax.random.PRNGKey(0))
prev = 0
print("float32 vs float64, SAME initial condition and controls, dt=0.02, 747 cruise")
print(f"{'steps':>8} {'t [s]':>8} {'|dpos| [m]':>14} {'|dvel| [m/s]':>14} "
      f"{'|dquat|':>12} {'|domega|':>12} {'f32 ||q||-1':>14}")
for n in MARKS:
    fin, _ = integrate.rollout(sim, ct32, jnp.float32(0.02), ac32, n - prev)
    sim = fin; prev = n
    p, v, q, o = (np.asarray(fin.state.pos_ned, dtype=float),
                  np.asarray(fin.state.vel_body, dtype=float),
                  np.asarray(fin.state.quat, dtype=float),
                  np.asarray(fin.state.omega, dtype=float))
    P, Vv, Q, O = out64[n]
    print(f"{n:>8d} {n*0.02:>8.0f} {np.linalg.norm(p-P):>14.6e} "
          f"{np.linalg.norm(v-Vv):>14.6e} {np.linalg.norm(q-Q):>12.6e} "
          f"{np.linalg.norm(o-O):>12.6e} {float(np.linalg.norm(q))-1:>+14.3e}")
    print(f"         f64 pos = {P}")
    print(f"         f32 pos = {p}")
