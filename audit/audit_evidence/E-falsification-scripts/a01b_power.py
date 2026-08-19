"""ATTACK 1b: direct aerodynamic power probe.

For E = 0.5 m|v|^2 + 0.5 w.I.w + m g h and the EOM in dynamics.py,
    dE/dt = F_aero . v_body + M_aero . omega     (exactly; gravity cancels)
so in still air with throttle 0 a quasi-steady aero model must give
P_aero <= 0 at every state. Positive P_aero is energy extracted from still air.
"""
from common import *  # noqa

from flightsim.aero import aero_forces_moments, air_data
from flightsim.atmosphere import density, speed_of_sound


def P_aero(vel, om, c, ac, h):
    rho = density(h)
    a = speed_of_sound(h)
    F, M = aero_forces_moments(vel, om, c, ac, rho, a)
    return float(jnp.dot(F, vel) + jnp.dot(M, om)), F, M


# --- 1. verify the identity dE/dt = F.v + M.w against a finite difference ------
print("=== identity check: dE/dt vs F.v + M.w (throttle 0) ===")
ac, st, x, r = trim_state("cessna172")
st = st._replace(omega=jnp.array([0.2, 0.15, -0.1]))
c = ctrl(de=float(x[1]), thr=0.0)
DT = 1e-5
_, hist = run(st, c, ac, DT, 2)
E, *_ = energy(hist, ac)
E0, _, _, _ = energy(jax.tree.map(lambda a: a[None], st), ac)
num = (E[0] - E0[0]) / DT
ana, F, M = P_aero(st.vel_body, st.omega, c, ac, -float(st.pos_ned[2]))
print(f"  finite difference dE/dt = {num:.6f} W")
print(f"  F.v + M.w              = {ana:.6f} W   rel err {abs(num-ana)/abs(ana):.2e}")

# --- 2. does P_aero ever go positive at physically reachable states? ----------
print("\n=== P_aero sign scan, still air, throttle 0 ===")
key = jax.random.PRNGKey(1)
for name in NAMES:
    ac = REGISTRY[name]
    V0 = CRUISE[name]["airspeed"]
    h0 = CRUISE[name]["altitude"]
    worst = None
    npos = 0
    N = 20000
    key, k = jax.random.split(key)
    ks = jax.random.split(k, 6)
    Vs = V0 * jax.random.uniform(ks[0], (N,), minval=0.5, maxval=1.6)
    al = jax.random.uniform(ks[1], (N,), minval=-0.25, maxval=0.25)
    be = jax.random.uniform(ks[2], (N,), minval=-0.20, maxval=0.20)
    oms = jax.random.uniform(ks[3], (N, 3), minval=-0.35, maxval=0.35)
    des = jax.random.uniform(ks[4], (N,), minval=-float(ac.elevator_limit),
                             maxval=float(ac.elevator_limit))
    das = jax.random.uniform(ks[5], (N, 2), minval=-float(ac.aileron_limit),
                             maxval=float(ac.aileron_limit))
    vel = jnp.stack([Vs * jnp.cos(al) * jnp.cos(be), Vs * jnp.sin(be),
                     Vs * jnp.sin(al) * jnp.cos(be)], axis=1)
    rho, a = density(h0), speed_of_sound(h0)

    def p_of(v, o, de, da, dr):
        cc = Controls(elevator=de, aileron=da, rudder=dr, throttle=jnp.array(0.0))
        F, M = aero_forces_moments(v, o, cc, ac, rho, a)
        return jnp.dot(F, v) + jnp.dot(M, o)

    P = jax.vmap(p_of)(vel, oms, des, das[:, 0], das[:, 1])
    P = np.asarray(P)
    W = float(ac.mass) * G0
    print(f"  {name:20s} N={N}  P>0 in {int((P>0).sum()):6d} "
          f"({100*(P>0).mean():5.2f}%)  max P = {P.max():+.4g} W "
          f"= {P.max()/(W*V0):+.4g} * (W*V)  min P = {P.min():.3g} W")
    if (P > 0).any():
        i = int(np.argmax(P))
        Vv, alv, bev = float(Vs[i]), float(al[i]), float(be[i])
        print(f"      worst: V={Vv:.1f} m/s alpha={np.degrees(alv):+.2f} deg "
              f"beta={np.degrees(bev):+.2f} deg omega={np.asarray(oms[i])} "
              f"de={float(des[i]):+.3f} da={float(das[i,0]):+.3f} dr={float(das[i,1]):+.3f}")

# --- 3. which channel supplies the positive power? ---------------------------
print("\n=== channel decomposition at the worst cessna172 point ===")
ac = REGISTRY["cessna172"]
h0 = CRUISE["cessna172"]["altitude"]
rho, a = density(h0), speed_of_sound(h0)
V0 = CRUISE["cessna172"]["airspeed"]
# a deliberately constructed pitch-rate case
vel = jnp.array([V0, 0.0, 0.0])
om = jnp.array([0.0, 0.30, 0.0])
cc = ctrl(de=-0.3)
F, M = aero_forces_moments(vel, om, cc, ac, rho, a)
Pf, Pm = float(jnp.dot(F, vel)), float(jnp.dot(M, om))
print(f"  V={V0} q=0.30 rad/s de=-0.30: F.v = {Pf:.1f} W   M.w = {Pm:.1f} W   "
      f"total = {Pf+Pm:.1f} W")
Vsp = float(jnp.linalg.norm(vel))
print(f"  for scale, weight*V = {float(ac.mass)*G0*Vsp:.0f} W")
