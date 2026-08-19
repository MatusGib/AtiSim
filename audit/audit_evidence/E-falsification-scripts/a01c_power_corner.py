"""ATTACK 1c: hunt the corner where the aero MOMENT channel outpowers drag.

P_drag  ~ 0.5 rho V^3 S CD          (always negative)
P_moment ~ 0.5 rho V^2 S c (Cm0 + Cma a + Cmde de) q   (either sign)
so the ratio scales like q/V: low speed + high rate should flip the sign.
"""
from common import *  # noqa

from flightsim.aero import aero_forces_moments
from flightsim.atmosphere import density, speed_of_sound


def sweep(name, h=None):
    ac = REGISTRY[name]
    h = CRUISE[name]["altitude"] if h is None else h
    rho, a = density(h), speed_of_sound(h)
    Vc = CRUISE[name]["airspeed"]

    Vs = np.geomspace(2.0, 2.0 * Vc, 90)
    qs = np.concatenate([-np.geomspace(0.02, 6.0, 60)[::-1], np.geomspace(0.02, 6.0, 60)])
    des = np.array([-1.0, -0.5, 0.0, 0.5, 1.0]) * float(ac.elevator_limit)
    best = (-np.inf, None)
    for de in des:
        for V in Vs:
            for q in qs:
                vel = jnp.array([V, 0.0, 0.0])
                om = jnp.array([0.0, q, 0.0])
                cc = ctrl(de=de)
                F, M = aero_forces_moments(vel, om, cc, ac, rho, a)
                P = float(jnp.dot(F, vel) + jnp.dot(M, om))
                if P > best[0]:
                    best = (P, (V, q, de))
    return ac, best


print("=== pitch-plane corner scan: max P_aero over (V, q, de), still air, thr 0 ===")
for name in NAMES:
    ac, (P, (V, q, de)) = sweep(name)
    Vc = CRUISE[name]["airspeed"]
    print(f"  {name:20s} max P = {P:+12.4g} W   at V={V:6.2f} m/s ({V/Vc:.2f}*Vcruise) "
          f"q={q:+6.3f} rad/s de={de:+.3f} rad")

# Refine on the best aircraft: is there a whole positive-power region, and is it
# reachable? Map P>0 in (V, q) at full nose-up elevator.
print("\n=== positive-power region, full elevator, at cruise altitude ===")
for name in NAMES:
    ac = REGISTRY[name]
    h = CRUISE[name]["altitude"]
    rho, a = density(h), speed_of_sound(h)
    Vc = CRUISE[name]["airspeed"]
    lim = float(ac.elevator_limit)
    found = []
    for de in (-lim, lim):
        for V in np.geomspace(1.0, 1.5 * Vc, 160):
            for q in np.concatenate([-np.geomspace(0.005, 10, 120)[::-1],
                                     np.geomspace(0.005, 10, 120)]):
                vel = jnp.array([V, 0.0, 0.0])
                om = jnp.array([0.0, q, 0.0])
                F, M = aero_forces_moments(vel, om, ctrl(de=de), ac, rho, a)
                P = float(jnp.dot(F, vel) + jnp.dot(M, om))
                if P > 0:
                    found.append((V, q, de, P))
    if found:
        Vmin = min(f[0] for f in found)
        Vmax = max(f[0] for f in found)
        Pmax = max(f[3] for f in found)
        arg = [f for f in found if f[3] == Pmax][0]
        print(f"  {name:20s} P>0 at {len(found)} grid points; V in "
              f"[{Vmin:.2f},{Vmax:.2f}] m/s (Vcruise={Vc:.1f}); "
              f"max P={Pmax:.4g} W at V={arg[0]:.2f} q={arg[1]:+.3f} de={arg[2]:+.3f}")
    else:
        print(f"  {name:20s} no positive-power point found")
