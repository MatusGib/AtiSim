"""ATTACK 7/1e (jitted): pump mechanical energy out of STILL AIR, throttle = 0.

P_moment = qbar*S*c*Cmde*de*q. With Cmde < 0, choosing de = -K*q makes it
POSITIVE. That is an ordinary pitch-rate feedback with the sign reversed -- a
realisable control input, not a state injection. If total mechanical energy
rises, the model is extracting energy from motionless, uniform air.

Also tested: full-deflection reversals at the 50 Hz sample rate.
"""
from common import *  # noqa

from functools import partial
from flightsim.integrate import rk4_step
from flightsim.state import quat_normalize
from flightsim.dynamics import derivatives

Z3 = jnp.zeros(3)


@partial(jax.jit, static_argnames=("n", "mode"))
def fly(st0, ac, dt, n, K, lim, mode, thr=0.0):
    def body(carry, i):
        st = carry
        q = st.omega[1]
        if mode == "prop":                 # de = -K*q, saturated
            de = jnp.clip(-K * q, -lim, lim)
        elif mode == "bang_anti":          # full deflection opposing pitch rate
            de = -lim * jnp.sign(q)
        elif mode == "bang_with":
            de = lim * jnp.sign(q)
        elif mode == "alternate":          # +-lim every single step
            de = lim * jnp.where(i % 2 == 0, 1.0, -1.0)
        elif mode == "roll_anti":          # same trick on the roll axis
            de = jnp.array(0.0)
        c = Controls(elevator=de, aileron=jnp.array(0.0), rudder=jnp.array(0.0),
                     throttle=jnp.array(thr))
        if mode == "roll_anti":
            c = c._replace(aileron=jnp.clip(-K * st.omega[0], -lim, lim))
        f = lambda s: derivatives(s, c, ac, Z3, Z3)
        st = rk4_step(f, st, dt)
        st = st._replace(quat=quat_normalize(st.quat))
        return st, (st, de)

    return jax.lax.scan(body, st0, jnp.arange(n))


DT = 0.02
SEC = 120.0
N = int(SEC / DT)

print(f"=== energy pumping in still air, throttle = 0, dt={DT}s, {SEC:.0f}s ===")
print(f"{'aircraft':20s} {'law':22s} {'K':>7s} {'E0 (J)':>12s} {'Emax/E0':>9s} "
      f"{'Efin/E0':>9s} {'maxV':>7s} {'max|q|':>7s} {'finite':>7s}")
worst = []
for name in NAMES:
    ac, st, x, r = trim_state(name)
    lim = float(ac.elevator_limit)
    for mode, Ks in [("prop", [1.0, 5.0, 20.0, 100.0, 1000.0]),
                     ("bang_anti", [0.0]), ("bang_with", [0.0]),
                     ("alternate", [0.0]),
                     ("roll_anti", [5.0, 100.0])]:
        for K in Ks:
            _, (hist, de) = fly(st, ac, jnp.array(DT), N, jnp.array(K),
                                jnp.array(lim), mode)
            E, ke, rke, pe = energy(hist, ac)
            fin = bool(np.all(np.isfinite(E)))
            if not fin:
                print(f"{name:20s} {mode:22s} {K:7.1f}  NON-FINITE at step "
                      f"{int(np.argmax(~np.isfinite(E)))}")
                continue
            V = np.linalg.norm(np.asarray(hist.vel_body), axis=1)
            q = np.asarray(hist.omega)[:, 1]
            ratio = E.max() / E[0]
            print(f"{name:20s} {mode:22s} {K:7.1f} {E[0]:12.5g} {ratio:9.5f} "
                  f"{E[-1]/E[0]:9.5f} {V.max():7.1f} {np.abs(q).max():7.2f} "
                  f"{str(fin):>7s}")
            worst.append((ratio, name, mode, K, float(E[0]), float(E.max())))

print("\n--- biggest energy GAIN found (Emax/E0 > 1 means energy created) ---")
for ratio, name, mode, K, E0, Emx in sorted(worst, reverse=True)[:8]:
    print(f"  {name:20s} {mode:12s} K={K:7.1f}  Emax/E0 = {ratio:.6f}  "
          f"gain = {Emx-E0:+.5g} J")

# --- longer horizon on the best pump, to see if it is a growing mode ---------
print("\n=== 600 s on the strongest pump found ===")
ratio, name, mode, K, _, _ = max(worst)
ac, st, x, r = trim_state(name)
lim = float(ac.elevator_limit)
N2 = int(600.0 / DT)
_, (hist, de) = fly(st, ac, jnp.array(DT), N2, jnp.array(K), jnp.array(lim), mode)
E, ke, rke, pe = energy(hist, ac)
t = np.arange(N2) * DT
print(f"  {name} / {mode} / K={K}: E0={E[0]:.6g} J")
for frac in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]:
    i = min(int(frac * (N2 - 1)), N2 - 1)
    print(f"    t={t[i]:7.1f} s  E={E[i]:14.7g} J  E/E0={E[i]/E[0]:9.6f}  "
          f"KE={ke[i]:12.6g} PE={pe[i]:12.6g} RKE={rke[i]:10.5g}")
print(f"  monotone increasing? {bool(np.all(np.diff(E) >= 0))}; "
      f"steps with dE>0: {int((np.diff(E)>0).sum())}/{N2-1}")
