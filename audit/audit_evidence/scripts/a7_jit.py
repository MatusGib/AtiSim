"""A7: does jit change results? jitted vs jax.disable_jit(), bit for bit."""
import hashlib
import numpy as np
import jax, jax.numpy as jnp
import flightsim
from flightsim import integrate, trim, validation, wind
from flightsim.aircraft import CRUISE, REGISTRY

AC = REGISTRY["boeing747"]
V, H = CRUISE["boeing747"]["airspeed"], CRUISE["boeing747"]["altitude"]


def h(x):
    return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()[:24]


def leafhash(tree):
    return {k: h(v) for k, v in zip(tree._fields, tree)}


print("=" * 78)
print("### 7. JIT vs EAGER (jax.disable_jit())")
print("=" * 78)

# ---------------------------------------------------------------- trim
print("\n-- trim --")
xj, rj = trim.trim(jnp.array(V), jnp.array(H), AC)
with jax.disable_jit():
    xe, re = trim.trim(jnp.array(V), jnp.array(H), AC)
print(f"  jit   x = {np.asarray(xj)!r}")
print(f"  eager x = {np.asarray(xe)!r}")
print(f"  bit-identical x: {h(xj) == h(xe)}   |dx| = {float(jnp.linalg.norm(xj-xe)):.3e}")
print(f"  bit-identical residual: {h(rj) == h(re)}  |dres| = {float(jnp.linalg.norm(rj-re)):.3e}")

x = xj
state = trim.trimmed_state(x[0], jnp.array(V), jnp.array(H))
controls = trim.trimmed_controls(x[1] + 0.02, x[2])
sim0 = integrate.init_sim(state, jax.random.PRNGKey(0))

# ---------------------------------------------------------------- one step
print("\n-- one integrate.step, dt=0.02 --")
sj = integrate.step(sim0, controls, jnp.array(0.02), AC)
with jax.disable_jit():
    se = integrate.step(sim0, controls, jnp.array(0.02), AC)
hj, he = leafhash(sj.state), leafhash(se.state)
for k in hj:
    d = float(jnp.max(jnp.abs(getattr(sj.state, k) - getattr(se.state, k))))
    print(f"  {k:<9} identical={hj[k]==he[k]}  max|diff|={d:.3e}")

# ------------------------------------------------------------ 1000-step rollout
print("\n-- 1000-step rollout, dt=0.02 --")
fj, tj = integrate.rollout(sim0, controls, jnp.array(0.02), AC, 1000)
with jax.disable_jit():
    fe, te = integrate.rollout(sim0, controls, jnp.array(0.02), AC, 1000)
for k in tj._fields:
    a, b = getattr(tj, k), getattr(te, k)
    d = float(jnp.max(jnp.abs(a - b)))
    print(f"  traj.{k:<9} identical={h(a)==h(b)}  max|diff|={d:.3e}")
print(f"  final pos jit   = {np.asarray(fj.state.pos_ned)!r}")
print(f"  final pos eager = {np.asarray(fe.state.pos_ned)!r}")

# ------------------------------------------------------- rollout with a wind field
print("\n-- 1000-step rollout through the smooth lee-wave field --")
wave = wind.LeeWave(w0=jnp.array(25.0), wavelength=jnp.array(1200.0), north=jnp.array(0.0))
model = wind.field_model(lambda p: wind.lee_wave_wind(p, wave))
fj, tj = integrate.rollout(sim0, controls, jnp.array(0.02), AC, 1000, wind_model=model)
with jax.disable_jit():
    fe, te = integrate.rollout(sim0, controls, jnp.array(0.02), AC, 1000, wind_model=model)
for k in tj._fields:
    a, b = getattr(tj, k), getattr(te, k)
    print(f"  traj.{k:<9} identical={h(a)==h(b)}  max|diff|={float(jnp.max(jnp.abs(a-b))):.3e}")

# ------------------------------------------------------------- mode extraction
print("\n-- mode extraction (jacfwd through validation) --")
a, e, t = float(x[0]), float(x[1]), float(x[2])
Aj = validation.longitudinal_matrix(AC, a, e, t, V, H)
with jax.disable_jit():
    Ae = validation.longitudinal_matrix(AC, a, e, t, V, H)
print(f"  A_lon identical: {h(Aj)==h(Ae)}  max|diff| = {np.abs(Aj-Ae).max():.3e}")
lj = validation.longitudinal_modes(AC, a, e, t, V, H)
with jax.disable_jit():
    le = validation.longitudinal_modes(AC, a, e, t, V, H)
for (w1, z1), (w2, z2) in zip(lj, le):
    print(f"  wn {w1!r} vs {w2!r}  equal={w1==w2}")
    print(f"  zeta {z1!r} vs {z2!r}  equal={z1==z2}")
dj = validation.lateral_modes(AC, a, e, t, V, H)
with jax.disable_jit():
    de = validation.lateral_modes(AC, a, e, t, V, H)
print(f"  lateral jit  : {dj}")
print(f"  lateral eager: {de}")
print(f"  lateral equal: {dj == de}")

# --------------------------------------- rollout vs logged_rollout vs manual loop
print("\n-- rollout vs logged_rollout vs a hand Python loop of `step` --")
f1, t1 = integrate.rollout(sim0, controls, jnp.array(0.02), AC, 500)
f2, l2 = integrate.logged_rollout(sim0, controls, jnp.array(0.02), AC, 500)
s = sim0
for _ in range(500):
    s = integrate.step(s, controls, jnp.array(0.02), AC)
print(f"  rollout vs logged_rollout final state identical: "
      f"{leafhash(f1.state) == leafhash(f2.state)}")
print(f"  rollout vs python loop      final state identical: "
      f"{leafhash(f1.state) == leafhash(s.state)}")
print(f"    max |dpos| = {float(jnp.max(jnp.abs(f1.state.pos_ned - s.state.pos_ned))):.3e}")

# ------------------------------------------------- vmap: does batching change bits?
print("\n-- batched_rollout (vmap) vs single rollout --")
keys = jax.random.split(jax.random.PRNGKey(0), 4)
bsim = integrate.batch_sim(state, keys)
bf, bt = integrate.batched_rollout(bsim, controls, jnp.array(0.02), AC, 500)
print(f"  vmap member 0 vs single rollout identical: "
      f"{h(bf.state.pos_ned[0]) == h(f1.state.pos_ned)}")
print(f"    max |dpos| = {float(jnp.max(jnp.abs(bf.state.pos_ned[0] - f1.state.pos_ned))):.3e}")
