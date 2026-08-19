"""ATTACK 1d: decompose the Cherokee's positive aerodynamic power."""
from common import *  # noqa

from flightsim.aero import aero_forces_moments, coefficients, air_data
from flightsim.atmosphere import density, speed_of_sound

for name in NAMES:
    ac = REGISTRY[name]
    print(f"{name:20s} CL0={float(ac.CL0):+.4f} CLa={float(ac.CLa):+.4f} "
          f"CLq={float(ac.CLq):+8.3f} CLde={float(ac.CLde):+.4f} "
          f"Cm0={float(ac.Cm0):+.4f} Cma={float(ac.Cma):+.4f} "
          f"Cmq={float(ac.Cmq):+9.4f} Cmde={float(ac.Cmde):+.4f} "
          f"CD0={float(ac.CD0):.4f} e={float(ac.e):.4f} AR={float(ac.AR):.3f} "
          f"c={float(ac.c):.3f} b={float(ac.b):.3f}")

print("\n=== Cherokee: P_aero along the worst ray (de = +limit, q sweep) ===")
ac = REGISTRY["cherokee"]
h = CRUISE["cherokee"]["altitude"]
rho, a = density(h), speed_of_sound(h)
lim = float(ac.elevator_limit)
V = 75.0
print(f"  V={V} m/s, h={h:.0f} m, de=+{lim:.4f} rad")
print(f"  {'q':>8s} {'P_force':>13s} {'P_moment':>13s} {'P_tot':>13s} "
      f"{'CL':>9s} {'CD':>9s} {'Cm':>9s} {'q_hat':>8s}")
for q in [-12, -10, -8.256, -6, -4, -2, -1, 0, 1, 2, 4, 6]:
    vel = jnp.array([V, 0.0, 0.0])
    om = jnp.array([0.0, float(q), 0.0])
    c = ctrl(de=lim)
    F, M = aero_forces_moments(vel, om, c, ac, rho, a)
    CL, CD, CY, Cl, Cm, Cn = coefficients(vel, om, c, ac, a)
    Pf, Pm = float(jnp.dot(F, vel)), float(jnp.dot(M, om))
    print(f"  {q:8.3f} {Pf:13.4g} {Pm:13.4g} {Pf+Pm:13.4g} "
          f"{float(CL):9.4f} {float(CD):9.4f} {float(Cm):9.4f} "
          f"{q*float(ac.c)/(2*V):8.4f}")

print("\n=== which coefficient supplies it? Cm decomposition at q=-8.256 ===")
q = -8.256
qh = q * float(ac.c) / (2 * V)
print(f"  q_hat = {qh:.4f}")
print(f"  Cm0        = {float(ac.Cm0):+.5f}")
print(f"  Cma*alpha  = {float(ac.Cma)*0.0:+.5f}   (alpha=0)")
print(f"  Cmq*q_hat  = {float(ac.Cmq)*qh:+.5f}")
print(f"  Cmde*de    = {float(ac.Cmde)*lim:+.5f}")
print(f"  -> Cm      = {float(ac.Cm0)+float(ac.Cmq)*qh+float(ac.Cmde)*lim:+.5f}")
print(f"  M.w = qbar*S*c*Cm*q, sign of Cm*q = "
      f"{(float(ac.Cm0)+float(ac.Cmq)*qh+float(ac.Cmde)*lim)*q:+.5f}")
print("  Cmq damping power alone (qbar*S*c*Cmq*q_hat*q):")
qbar = 0.5 * float(rho) * V * V
print(f"    {qbar*float(ac.S)*float(ac.c)*float(ac.Cmq)*qh*q:+.4g} W  "
      f"(must be <=0 for Cmq<0)")
print("  Cmde control power alone (qbar*S*c*Cmde*de*q):")
print(f"    {qbar*float(ac.S)*float(ac.c)*float(ac.Cmde)*lim*q:+.4g} W")

print("\n=== is the positive-power state REACHABLE in a rollout? ===")
# Fly the Cherokee from trim with full nose-down elevator, throttle 0, and watch
# for a step where total energy rises.
ac2, st, x, r = trim_state("cherokee")
for de_sign, lbl in [(+1, "full +elevator (TE down)"), (-1, "full -elevator")]:
    DT = 0.002
    n = int(60.0 / DT)
    _, hist = run(st, ctrl(de=de_sign * lim, thr=0.0), ac2, DT, n)
    E, ke, rke, pe = energy(hist, ac2)
    qh = np.asarray(hist.omega)[:, 1]
    Vh = np.linalg.norm(np.asarray(hist.vel_body), axis=1)
    dE = np.diff(E)
    print(f"  {lbl:26s} max q={qh.max():+7.3f} min q={qh.min():+7.3f} rad/s  "
          f"max|V|={Vh.max():7.2f}  E-E0 max={float((E-E[0]).max()):+.4g} J  "
          f"steps with dE>0: {int((dE>0).sum())}/{len(dE)}  "
          f"max dE/dt={float(dE.max()/DT):+.4g} W")
