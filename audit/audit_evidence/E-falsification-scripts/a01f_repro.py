"""ATTACK 1f (DECISIVE REPRO): total mechanical energy RISES in still air with
throttle = 0, by starting the integration on the positive-aero-power manifold.

Cherokee (registry values, untouched), sea-level-referenced cruise altitude:
    V = 75 m/s along body x, q = -8.256 rad/s, elevator = +25 deg (its limit),
    throttle = 0, wind = 0, omega_gust = 0.
P_aero = F.v + M.omega = +5.7e4 W > 0, so E must rise, and it does.

The energy source is the control-moment power term qbar*S*c*Cmde*de*q. The model
charges no drag for elevator deflection (CD has no CD_de term), so a deflected
elevator can do net positive work on the airframe in motionless, uniform air.
"""
from common import *  # noqa

from flightsim.aero import aero_forces_moments
from flightsim.atmosphere import density, speed_of_sound

ac = REGISTRY["cherokee"]
H = CRUISE["cherokee"]["altitude"]
lim = float(ac.elevator_limit)

st0 = State(
    pos_ned=jnp.array([0.0, 0.0, -H]),
    vel_body=jnp.array([75.0, 0.0, 0.0]),
    quat=euler_to_quat(jnp.array(0.0), jnp.array(0.0), jnp.array(0.0)),
    omega=jnp.array([0.0, -8.256, 0.0]),
)
c = ctrl(de=lim, thr=0.0)

rho, a = density(jnp.array(H)), speed_of_sound(jnp.array(H))
F, M = aero_forces_moments(st0.vel_body, st0.omega, c, ac, rho, a)
P = float(jnp.dot(F, st0.vel_body) + jnp.dot(M, st0.omega))
print("=== initial state ===")
print(f"  cherokee (registry, unmodified), h = {H:.1f} m, rho = {float(rho):.4f}")
print(f"  V = 75.0 m/s, q = -8.256 rad/s, elevator = +{np.degrees(lim):.1f} deg, "
      f"throttle = 0, still air")
print(f"  F_aero = {np.asarray(F)} N")
print(f"  M_aero = {np.asarray(M)} N.m")
print(f"  P_aero = F.v + M.omega = {P:+.6g} W   (MUST be <= 0)")

for DT in (1e-4, 1e-3, 1e-2):
    n = int(0.05 / DT)
    _, hist = run(st0, c, ac, DT, n)
    E, ke, rke, pe = energy(hist, ac)
    E0, ke0, rke0, pe0 = energy(jax.tree.map(lambda x: x[None], st0), ac)
    E = np.concatenate([E0, E]); ke = np.concatenate([ke0, ke])
    rke = np.concatenate([rke0, rke]); pe = np.concatenate([pe0, pe])
    t = np.arange(len(E)) * DT
    i = int(np.argmax(E))
    print(f"\n  dt = {DT:g} s, {n} steps:")
    print(f"    E(0)    = {E[0]:.10g} J")
    print(f"    E(peak) = {E[i]:.10g} J at t = {t[i]:.5f} s")
    print(f"    GAIN    = {E[i]-E[0]:+.6g} J   ({100*(E[i]-E[0])/E[0]:+.4e} %)")
    print(f"    mean rate over the rise = {(E[i]-E[0])/max(t[i],DT):+.6g} W "
          f"(P_aero at t=0 was {P:+.6g} W)")
    print(f"    breakdown at peak: dKE={ke[i]-ke[0]:+.5g}  dPE={pe[i]-pe[0]:+.5g}  "
          f"dRKE={rke[i]-rke[0]:+.5g} J")
    print(f"    steps with dE>0: {int((np.diff(E)>0).sum())}/{len(E)-1}")

print("\n=== how deep into the reachable envelope does this go? ===")
print("  minimum |q| at which P_aero > 0, cherokee, elevator at its limit:")
for V in [30.0, 40.0, 50.0, 60.0, 75.0, 90.0]:
    lo = None
    for q in -np.linspace(0.01, 15.0, 3000):
        vel = jnp.array([V, 0.0, 0.0])
        om = jnp.array([0.0, float(q), 0.0])
        F, M = aero_forces_moments(vel, om, c, ac, rho, a)
        if float(jnp.dot(F, vel) + jnp.dot(M, om)) > 0:
            lo = q
            break
    print(f"    V={V:6.1f} m/s -> P_aero>0 first at q = "
          f"{lo if lo is not None else float('nan'):+8.3f} rad/s "
          f"({np.degrees(lo) if lo is not None else float('nan'):+8.1f} deg/s)")

print("\n  for reference, the largest |q| a fixed-control rollout from trim")
print("  reaches with full elevator and throttle 0:")
ac2, st, x, r = trim_state("cherokee")
for de_s, lbl in [(+1, "+lim"), (-1, "-lim")]:
    _, hist = run(st, ctrl(de=de_s * lim, thr=0.0), ac2, 0.002, 30000)
    q = np.asarray(hist.omega)[:, 1]
    print(f"    de={lbl}: max|q| = {np.abs(q).max():.3f} rad/s "
          f"({np.degrees(np.abs(q).max()):.0f} deg/s)")
