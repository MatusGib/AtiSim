"""ATTACK 7d (corrected): full-deflection reversals at the sample rate,
THROTTLE = 0, so any rise in total mechanical energy is energy created.

The earlier version of this test left the throttle at its trim value, so a rise
in E was legitimate engine work. This one cuts the throttle.
"""
from common import *  # noqa

from functools import partial
from flightsim.integrate import rk4_step
from flightsim.state import quat_normalize
from flightsim.dynamics import derivatives

Z3 = jnp.zeros(3)


@partial(jax.jit, static_argnames=("n",))
def alternate(st0, ac, dt, n, lim, thr):
    def body(st, i):
        de = lim * jnp.where(i % 2 == 0, 1.0, -1.0)
        c = Controls(elevator=de, aileron=jnp.array(0.0), rudder=jnp.array(0.0),
                     throttle=thr)
        st = rk4_step(lambda s: derivatives(s, c, ac, Z3, Z3), st, dt)
        return st._replace(quat=quat_normalize(st.quat)), st
    return jax.lax.scan(body, st0, jnp.arange(n))


print("=== full-deflection elevator reversals every step, THROTTLE = 0 ===")
print("  any max(E - E0) > 0 is mechanical energy created in still air")
for name in NAMES:
    ac, st, x, r = trim_state(name)
    lim = jnp.array(float(ac.elevator_limit))
    print(f"  {name}:")
    for dt in (0.04, 0.02, 0.01, 0.005, 0.002, 0.001):
        n = int(30.0 / dt)
        _, hist = alternate(st, ac, jnp.array(dt), n, lim, jnp.array(0.0))
        E, *_ = energy(hist, ac)
        v = np.linalg.norm(np.asarray(hist.vel_body), axis=1)
        q = np.asarray(hist.omega)[:, 1]
        excess = float((E - E[0]).max())
        run_min = np.minimum.accumulate(E)
        rise = float((E - run_min).max())
        print(f"    dt={dt:6.3f} ({1/dt:5.0f} Hz) finite="
              f"{bool(np.all(np.isfinite(E)))} Vfin={v[-1]:8.2f} "
              f"max|q|={np.abs(q).max():7.4f}  max(E-E0)={excess:+11.4g} J  "
              f"max rise from a local min = {rise:11.4g} J")

print("\n=== same, but ALSO aileron+rudder reversing (worst case) ===")


@partial(jax.jit, static_argnames=("n",))
def alt3(st0, ac, dt, n, el, al, rl):
    def body(st, i):
        s = jnp.where(i % 2 == 0, 1.0, -1.0)
        c = Controls(elevator=el * s, aileron=al * s, rudder=rl * s,
                     throttle=jnp.array(0.0))
        st = rk4_step(lambda ss: derivatives(ss, c, ac, Z3, Z3), st, dt)
        return st._replace(quat=quat_normalize(st.quat)), st
    return jax.lax.scan(body, st0, jnp.arange(n))


for name in NAMES:
    ac, st, x, r = trim_state(name)
    for dt in (0.02, 0.005):
        n = int(30.0 / dt)
        _, hist = alt3(st, ac, jnp.array(dt), n,
                       jnp.array(float(ac.elevator_limit)),
                       jnp.array(float(ac.aileron_limit)),
                       jnp.array(float(ac.rudder_limit)))
        E, *_ = energy(hist, ac)
        run_min = np.minimum.accumulate(E)
        print(f"  {name:20s} dt={dt:6.3f}  max(E-E0)={float((E-E[0]).max()):+11.4g} J  "
              f"max rise={float((E-run_min).max()):11.4g} J  "
              f"finite={bool(np.all(np.isfinite(E)))}")
