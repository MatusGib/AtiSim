"""ATTACK 2: divergence over long horizons (1e5 - 1e6 steps).
ATTACK 9: lax.scan rollout vs a manual Python loop of step -- bit-identical?
"""
from common import *  # noqa

from flightsim.integrate import step, init_sim, logged_rollout
from flightsim.state import quat_to_euler

print("=== 2a. long-horizon trimmed hold: does the trim point survive? ===")
print(f"{'aircraft':20s} {'steps':>9s} {'sim s':>8s} {'max|q|-1|':>11s} "
      f"{'dV (m/s)':>11s} {'dh (m)':>12s} {'max|omega|':>11s} {'finite':>7s}")
for name in NAMES:
    ac, st, x, r = trim_state(name)
    c = ctrl(de=float(x[1]), thr=float(x[2]))
    for n, dt in [(100_000, 0.01), (1_000_000, 0.01)]:
        _, hist = run(st, c, ac, dt, n)
        q = np.asarray(hist.quat)
        v = np.asarray(hist.vel_body)
        p = np.asarray(hist.pos_ned)
        om = np.asarray(hist.omega)
        fin = bool(np.all(np.isfinite(v)) and np.all(np.isfinite(q)))
        V0 = float(np.linalg.norm(np.asarray(st.vel_body)))
        print(f"{name:20s} {n:9d} {n*dt:8.0f} "
              f"{np.abs(np.linalg.norm(q, axis=1)-1).max():11.3e} "
              f"{np.linalg.norm(v,axis=1).max()-V0:11.4f} "
              f"{-p[:,2].max()+p[0,2]*-1:12.3f} {np.abs(om).max():11.3e} "
              f"{str(fin):>7s}")

print("\n=== 2b. long-horizon with a large initial perturbation ===")
for name in NAMES:
    ac, st, x, r = trim_state(name)
    st2 = st._replace(omega=jnp.array([0.3, 0.2, -0.15]))
    c = ctrl(de=float(x[1]), thr=float(x[2]))
    n, dt = 500_000, 0.01
    _, hist = run(st2, c, ac, dt, n)
    q = np.asarray(hist.quat); v = np.asarray(hist.vel_body)
    p = np.asarray(hist.pos_ned)
    fin = bool(np.all(np.isfinite(v)))
    print(f"  {name:20s} {n} steps ({n*dt:.0f} s) finite={fin} "
          f"quat-norm dev {np.abs(np.linalg.norm(q,axis=1)-1).max():.2e} "
          f"Vmax={np.linalg.norm(v,axis=1).max():.1f} m/s "
          f"h range [{-p[:,2].max():.0f},{-p[:,2].min():.0f}] m")

print("\n=== 9. lax.scan rollout vs manual Python loop of step ===")
for name in NAMES:
    ac, st, x, r = trim_state(name)
    c = ctrl(de=float(x[1]), thr=float(x[2]))
    dt = jnp.array(0.01)
    n = 500
    _, hist = integrate.rollout(init_sim(st, jax.random.PRNGKey(0)), c, dt, ac, n)
    sim = init_sim(st, jax.random.PRNGKey(0))
    manual = []
    for _ in range(n):
        sim = step(sim, c, dt, ac)
        manual.append(sim.state)
    man = jax.tree.map(lambda *xs: jnp.stack(xs), *manual)
    diffs = {}
    for f in ("pos_ned", "vel_body", "quat", "omega"):
        a = np.asarray(getattr(hist, f)); b = np.asarray(getattr(man, f))
        diffs[f] = float(np.abs(a - b).max())
    ident = all(v == 0.0 for v in diffs.values())
    print(f"  {name:20s} bit-identical={ident}  max abs diff: "
          + "  ".join(f"{k}={v:.3e}" for k, v in diffs.items()))

print("\n=== 9b. rollout vs logged_rollout ===")
for name in NAMES:
    ac, st, x, r = trim_state(name)
    c = ctrl(de=float(x[1]), thr=float(x[2]))
    dt = jnp.array(0.01)
    n = 500
    _, h1 = integrate.rollout(init_sim(st, jax.random.PRNGKey(0)), c, dt, ac, n)
    _, h2 = logged_rollout(init_sim(st, jax.random.PRNGKey(0)), c, dt, ac, n)
    d = max(float(np.abs(np.asarray(getattr(h1, f))
                         - np.asarray(getattr(h2.state, f))).max())
            for f in ("pos_ned", "vel_body", "quat", "omega"))
    print(f"  {name:20s} max abs diff = {d:.3e}  bit-identical={d == 0.0}")

print("\n=== 9c. jit vs no-jit on `step` ===")
for name in NAMES:
    ac, st, x, r = trim_state(name)
    c = ctrl(de=float(x[1]), thr=float(x[2]))
    dt = jnp.array(0.01)
    s1 = step(init_sim(st, jax.random.PRNGKey(0)), c, dt, ac)
    with jax.disable_jit():
        s2 = step(init_sim(st, jax.random.PRNGKey(0)), c, dt, ac)
    d = max(float(np.abs(np.asarray(getattr(s1.state, f))
                         - np.asarray(getattr(s2.state, f))).max())
            for f in ("pos_ned", "vel_body", "quat", "omega"))
    print(f"  {name:20s} 1 step, jit vs no-jit max diff = {d:.3e}")
