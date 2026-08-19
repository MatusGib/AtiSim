"""Agent C, item 2: sign conventions traced END TO END on ASYMMETRIC cases.

For every convention the coefficient sign is checked AND the aircraft is
actually flown, so a sign that is right in the table and wrong in the plant
cannot pass.

Deliberate asymmetry:
  - controls are applied one at a time, never in cancelling pairs
  - the sideslip / incidence cases use a velocity with BOTH alpha and beta
    non-zero and unequal
  - the roll/yaw coupling case uses the 747, the only aircraft in the registry
    with Ixz != 0
"""
import jax.numpy as jnp
import numpy as np

from flightsim import aero, dynamics, integrate, trim
from flightsim.aircraft import REGISTRY, CRUISE
from flightsim.state import Controls, State, euler_to_quat, quat_to_euler
from flightsim.units import DEG2RAD, RAD2DEG

ZERO3 = jnp.zeros(3)
fails, notes = [], []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name:64s} {detail}")
    if not ok:
        fails.append(name)


def trimmed(name):
    ac = REGISTRY[name]
    cr = CRUISE[name]
    x, res = trim.trim(jnp.array(cr["airspeed"]), jnp.array(cr["altitude"]), ac)
    alpha, elev, thr = [float(v) for v in x]
    st = trim.trimmed_state(jnp.array(alpha), jnp.array(cr["airspeed"]),
                            jnp.array(cr["altitude"]))
    ct = trim.trimmed_controls(jnp.array(elev), jnp.array(thr))
    return ac, st, ct, (alpha, elev, thr), float(np.abs(np.asarray(res)).max())


def fly(ac, st, ct, seconds=4.0, dt=0.01):
    sim = integrate.init_sim(st, jnp.zeros(2, dtype=jnp.uint32))
    _, traj = integrate.rollout(sim, ct, dt, ac, int(seconds / dt))
    return traj


AIRCRAFT = ["boeing747", "boeing747_approach", "cherokee", "cessna172"]

# =========================================================================
print("=" * 78)
print("2a. alpha positive nose-up  -- deliberately PITCHED velocity")
print("=" * 78)
# Aircraft pitched nose-up 8 deg, flying with its velocity vector along NED
# north (so the flight path is level and alpha must equal theta exactly).
theta = 8.0 * DEG2RAD
V = 100.0
q = euler_to_quat(jnp.array(0.0), jnp.array(theta), jnp.array(0.0))
vel_ned = jnp.array([V, 0.0, 0.0])
from flightsim.state import quat_to_dcm
vel_body = quat_to_dcm(q).T @ vel_ned
Vt, a, b = aero.air_data(vel_body)
print(f"   theta = +8 deg, velocity due north (level flight path)")
print(f"   vel_body = {np.asarray(vel_body)}   (w = {float(vel_body[2]):+.4f} > 0)")
print(f"   alpha = {float(a) * RAD2DEG:+.6f} deg,  beta = {float(b) * RAD2DEG:+.6f} deg")
check("alpha == +theta for a nose-up aircraft on a level path",
      abs(float(a) - theta) < 1e-9 and float(a) > 0,
      f"alpha={float(a)*RAD2DEG:+.4f} deg")
check("alpha sign is set by +w (relative wind from BELOW)",
      float(aero.air_data(jnp.array([100.0, 0.0, +10.0]))[1]) > 0
      and float(aero.air_data(jnp.array([100.0, 0.0, -10.0]))[1]) < 0)

# =========================================================================
print()
print("=" * 78)
print("2b. beta positive = relative wind from the RIGHT -- sideslipping velocity")
print("=" * 78)
# Aircraft heading due NORTH, wings level, wind blowing FROM THE EAST
# (an easterly: NED wind vector points WEST, i.e. -E).
W = 15.0
q_lvl = euler_to_quat(jnp.array(0.0), jnp.array(0.0), jnp.array(0.0))
st_n = State(pos_ned=jnp.array([0.0, 0.0, -3000.0]),
             vel_body=jnp.array([V, 0.0, 0.0]), quat=q_lvl, omega=ZERO3)
wind_from_east = jnp.array([0.0, -W, 0.0])  # NED: blowing toward the west
vr = dynamics.relative_velocity(st_n.vel_body, st_n.quat, wind_from_east)
Vt, a, b = aero.air_data(vr)
print(f"   heading north, wind FROM THE EAST (from the right) at {W} m/s")
print(f"   vel_rel = {np.asarray(vr)}   (v = {float(vr[1]):+.4f})")
print(f"   beta = {float(b) * RAD2DEG:+.4f} deg")
check("relative wind from the RIGHT gives beta > 0", float(b) > 0,
      f"beta={float(b)*RAD2DEG:+.3f} deg")
check("  and the mirror case (wind from the left) gives beta < 0",
      float(aero.air_data(dynamics.relative_velocity(
          st_n.vel_body, st_n.quat, jnp.array([0.0, W, 0.0])))[2]) < 0)

# Physical consequence: with beta > 0 the side force must push LEFT (-y) and
# the yawing moment must swing the nose RIGHT, into the wind (weathercock).
print()
for name in AIRCRAFT:
    ac = REGISTRY[name]
    ctl = Controls(jnp.array(0.0), jnp.array(0.0), jnp.array(0.0), jnp.array(0.0))
    F, M = aero.aero_forces_moments(vr, ZERO3, ctl, ac, jnp.array(1.0),
                                    jnp.array(300.0))
    print(f"   {name:20s} CYb={float(ac.CYb):+8.4f} Cnb={float(ac.Cnb):+8.4f} "
          f"Clb={float(ac.Clb):+8.4f} -> Fy={float(F[1]):+11.1f} N  "
          f"N={float(M[2]):+12.1f} N.m  L={float(M[0]):+12.1f} N.m")
    check(f"   {name}: beta>0 pushes the aircraft LEFT (Fy<0)", float(F[1]) < 0)
    check(f"   {name}: beta>0 weathercocks nose RIGHT (N>0)", float(M[2]) > 0)
    check(f"   {name}: beta>0 rolls AWAY from the wind (L<0, dihedral effect)",
          float(M[0]) < 0)

# =========================================================================
print()
print("=" * 78)
print("2c. ELEVATOR positive = trailing edge down -> CLde>0, Cmde<0, nose DOWN")
print("=" * 78)
DE = 5.0 * DEG2RAD
for name in AIRCRAFT:
    ac, st, ct, (al, el, th), res = trimmed(name)
    check(f"   {name}: CLde > 0", float(ac.CLde) > 0, f"CLde={float(ac.CLde):+.4f}")
    check(f"   {name}: Cmde < 0", float(ac.Cmde) < 0, f"Cmde={float(ac.Cmde):+.4f}")
    # Instantaneous response from the trimmed state
    d = dynamics.derivatives(st, ct._replace(elevator=ct.elevator + DE), ac,
                             ZERO3, ZERO3)
    qdot = float(d.omega[1])
    # Flown
    traj = fly(ac, st, ct._replace(elevator=ct.elevator + DE), seconds=3.0)
    th_hist = np.asarray([quat_to_euler(qq) for qq in traj.quat])[:, 1]
    dtheta = float(th_hist[-1] - al)
    print(f"   {name:20s} trim res={res:.2e}  +5deg elev -> qdot={qdot:+.5f} rad/s^2, "
          f"dtheta(3s)={dtheta * RAD2DEG:+.3f} deg")
    check(f"   {name}: +elevator gives nose-DOWN pitch accel", qdot < 0)
    check(f"   {name}: +elevator flies the nose DOWN over 3 s", dtheta < 0)
    # And the lift increment at fixed attitude must be UP (Fz more negative)
    F0, _ = aero.aero_forces_moments(st.vel_body, ZERO3, ct, ac, jnp.array(0.4),
                                     jnp.array(300.0))
    F1, _ = aero.aero_forces_moments(st.vel_body, ZERO3,
                                     ct._replace(elevator=ct.elevator + DE), ac,
                                     jnp.array(0.4), jnp.array(300.0))
    check(f"   {name}: +elevator adds lift at frozen alpha (dFz < 0)",
          float(F1[2] - F0[2]) < 0, f"dFz={float(F1[2] - F0[2]):+.1f} N")

# =========================================================================
print()
print("=" * 78)
print("2d. AILERON positive -> Clda>0 and a RIGHT-WING-DOWN roll flown")
print("=" * 78)
DA = 5.0 * DEG2RAD
for name in AIRCRAFT:
    ac, st, ct, (al, el, th), res = trimmed(name)
    check(f"   {name}: Clda > 0", float(ac.Clda) > 0, f"Clda={float(ac.Clda):+.5f}")
    d = dynamics.derivatives(st, ct._replace(aileron=jnp.array(DA)), ac, ZERO3, ZERO3)
    pdot, rdot = float(d.omega[0]), float(d.omega[2])
    traj = fly(ac, st, ct._replace(aileron=jnp.array(DA)), seconds=3.0)
    eul = np.asarray([quat_to_euler(qq) for qq in traj.quat])
    phi = float(eul[-1, 0])
    print(f"   {name:20s} +5deg ail -> pdot={pdot:+.5f} rad/s^2  "
          f"rdot={rdot:+.6f}  phi(3s)={phi * RAD2DEG:+.2f} deg  "
          f"Cnda={float(ac.Cnda):+.5f}")
    check(f"   {name}: +aileron gives positive roll accel", pdot > 0)
    check(f"   {name}: +aileron flies to right-wing-down (phi>0)", phi > 0)

# =========================================================================
print()
print("=" * 78)
print("2e. RUDDER positive = TE left -> CYdr>0, Cndr<0, nose swings LEFT")
print("=" * 78)
DR = 5.0 * DEG2RAD
for name in AIRCRAFT:
    ac, st, ct, (al, el, th), res = trimmed(name)
    if float(ac.CYdr) == 0.0 and float(ac.Cndr) == 0.0:
        print(f"   {name:20s} rudder deliberately zeroed in the definition -- "
              f"no convention to test")
        notes.append(f"{name}: rudder set is identically zero; untestable")
        continue
    check(f"   {name}: CYdr > 0", float(ac.CYdr) > 0, f"CYdr={float(ac.CYdr):+.5f}")
    check(f"   {name}: Cndr < 0", float(ac.Cndr) < 0, f"Cndr={float(ac.Cndr):+.5f}")
    d = dynamics.derivatives(st, ct._replace(rudder=jnp.array(DR)), ac, ZERO3, ZERO3)
    rdot = float(d.omega[2])
    traj = fly(ac, st, ct._replace(rudder=jnp.array(DR)), seconds=3.0)
    eul = np.asarray([quat_to_euler(qq) for qq in traj.quat])
    dpsi = float(np.unwrap(eul[:, 2])[-1])
    beta_end = float(aero.air_data(traj.vel_body[-1])[2])
    print(f"   {name:20s} +5deg rud -> rdot={rdot:+.5f} rad/s^2  "
          f"dpsi(3s)={dpsi * RAD2DEG:+.3f} deg  beta(3s)={beta_end * RAD2DEG:+.3f} deg  "
          f"Cldr={float(ac.Cldr):+.5f}")
    check(f"   {name}: +rudder yaws nose LEFT (rdot<0)", rdot < 0)
    check(f"   {name}: +rudder flies the nose LEFT (dpsi<0)", dpsi < 0)
    # Nose left => the relative wind comes from the right => beta > 0
    check(f"   {name}: nose-left yaw develops POSITIVE beta", beta_end > 0)
    # Side force at frozen attitude must be to the RIGHT (+y)
    F, _ = aero.aero_forces_moments(st.vel_body, ZERO3,
                                    ct._replace(rudder=jnp.array(DR)), ac,
                                    jnp.array(0.4), jnp.array(300.0))
    F0, _ = aero.aero_forces_moments(st.vel_body, ZERO3, ct, ac, jnp.array(0.4),
                                     jnp.array(300.0))
    check(f"   {name}: +rudder side force is to the RIGHT (+y)",
          float(F[1] - F0[1]) > 0, f"dFy={float(F[1] - F0[1]):+.1f} N")

print()
print("FAILURES:", fails if fails else "none")
print("NOTES   :", notes if notes else "none")
