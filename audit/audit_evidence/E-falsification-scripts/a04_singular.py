"""ATTACK 4: singularities, V_MIN, atmosphere out of scope, degenerate Jacobians."""
from common import *  # noqa

from flightsim import atmosphere as atm
from flightsim.aero import V_MIN, air_data, aero_forces_moments
from flightsim.dynamics import derivatives
from flightsim.state import quat_to_euler, quat_to_dcm

Z3 = jnp.zeros(3)

print("=== 4a. atmosphere outside its declared 0-20 km scope ===")
print(f"{'h (m)':>10s} {'T (K)':>10s} {'p (Pa)':>14s} {'rho':>14s} {'a (m/s)':>10s}  note")
for h in [-1000.0, -500.0, -100.0, 0.0, 5000.0, 11000.0, 20000.0, 25000.0,
          30000.0, 47000.0, 50000.0, 86000.0, 1e5, 1e6]:
    T = float(atm.temperature(jnp.array(h)))
    p = float(atm.pressure(jnp.array(h)))
    rho = float(atm.density(jnp.array(h)))
    a = float(atm.speed_of_sound(jnp.array(h)))
    print(f"{h:10.0f} {T:10.3f} {p:14.6g} {rho:14.6g} {a:10.3f}")

print("\n  ISA reference values (US Std Atmos 1976 / ICAO), for comparison:")
print("    h=20 km: T=216.65 K, rho=0.088035 kg/m^3")
print("    h=30 km: T=226.65 K (stratosphere warms from 20 km, +0.001 K/m),")
print("             rho=0.018410 kg/m^3")
print("    h=47 km: T=270.65 K, rho=0.0014275 kg/m^3")
rho30 = float(atm.density(jnp.array(30000.0)))
print(f"  model at 30 km: rho={rho30:.6g}; ISA 0.018410 -> error "
      f"{100*(rho30-0.018410)/0.018410:+.2f}%")
rho47 = float(atm.density(jnp.array(47000.0)))
print(f"  model at 47 km: rho={rho47:.6g}; ISA 0.0014275 -> error "
      f"{100*(rho47-0.0014275)/0.0014275:+.2f}%")

print("\n  below sea level (there is no ground; a run CAN go here):")
for h in [-100.0, -1000.0, -5000.0, -11000.0, -20000.0, -44330.0, -44331.0, -50000.0]:
    T = float(atm.temperature(jnp.array(h)))
    rho = float(atm.density(jnp.array(h)))
    print(f"    h={h:9.0f} m  T={T:9.3f} K  rho={rho:12.6g}  "
          f"{'*** T<=0 / NON-PHYSICAL' if T <= 0 else ''}"
          f"{'  *** NaN' if not np.isfinite(rho) else ''}")

print("\n=== 4b. V_MIN = 1.0 m/s: construct a case where it BINDS ===")
ac = REGISTRY["cessna172"]
h = 1000.0
rho, a = atm.density(jnp.array(h)), atm.speed_of_sound(jnp.array(h))
print(f"  {'V (m/s)':>10s} {'|F_aero| (N)':>14s} {'Fz (N)':>13s} {'alpha':>9s} "
      f"{'qbar':>10s}  note")
for V in [0.0, 1e-9, 0.01, 0.1, 0.5, 0.9, 1.0, 1.1, 2.0, 5.0]:
    vel = jnp.array([V, 0.0, 0.0])
    Vr, al, be = air_data(vel)
    F, M = aero_forces_moments(vel, Z3, ctrl(), ac, rho, a)
    qbar = 0.5 * float(rho) * float(Vr) ** 2
    note = "V_MIN BINDS" if V < V_MIN else ""
    print(f"  {V:10.4g} {float(jnp.linalg.norm(F)):14.6g} {float(F[2]):13.6g} "
          f"{float(al):9.4f} {qbar:10.5g}  {note}")
print(f"  -> at V=0 the model reports |F| = "
      f"{float(jnp.linalg.norm(aero_forces_moments(jnp.zeros(3), Z3, ctrl(), ac, rho, a)[0])):.4g} N "
      f"= {float(jnp.linalg.norm(aero_forces_moments(jnp.zeros(3), Z3, ctrl(), ac, rho, a)[0]))/(float(ac.mass)*G0)*1000:.4g} milli-g")

print("\n  the OVERSTATEMENT factor: reported qbar / true qbar below V_MIN")
for V in [0.1, 0.25, 0.5, 0.75, 0.99]:
    print(f"    V={V:5.2f} m/s -> factor {(V_MIN/V)**2:8.1f}x too much dynamic pressure")

print("\n=== 4c. pure vertical velocity / alpha at +-90 deg ===")
for lbl, vel in [("pure +w (straight down rel. wind)", jnp.array([0.0, 0.0, 60.0])),
                 ("pure -w", jnp.array([0.0, 0.0, -60.0])),
                 ("pure +v (pure sideslip)", jnp.array([0.0, 60.0, 0.0])),
                 ("backwards flight u<0", jnp.array([-60.0, 0.0, 0.0]))]:
    V, al, be = air_data(vel)
    F, M = aero_forces_moments(vel, Z3, ctrl(), ac, rho, a)
    print(f"  {lbl:34s} V={float(V):6.2f} alpha={np.degrees(float(al)):+8.2f} deg "
          f"beta={np.degrees(float(be)):+7.2f} deg  F=[{float(F[0]):+.4g} "
          f"{float(F[1]):+.4g} {float(F[2]):+.4g}] N")

print("\n=== 4d. quat_to_euler at theta = +-90 deg (gimbal lock) ===")
for th_deg in [89.0, 89.9, 89.999, 90.0, 90.001, 90.1, -90.0]:
    q = euler_to_quat(jnp.array(0.3), jnp.array(np.radians(th_deg)), jnp.array(0.7))
    e = np.degrees(np.asarray(quat_to_euler(q)))
    dcm = np.asarray(quat_to_dcm(q))
    print(f"  theta_in={th_deg:9.4f}  -> phi={e[0]:+9.4f} theta={e[1]:+9.4f} "
          f"psi={e[2]:+9.4f}   phi+psi={e[0]+e[2]:+9.4f} (in: 0.3rad+0.7rad="
          f"{np.degrees(1.0):.4f})  |DCM orth err|="
          f"{np.abs(dcm@dcm.T-np.eye(3)).max():.2e}")

print("\n=== 4e. exactly-vertical flight integrated (does it survive?) ===")
for name in NAMES:
    ac2 = REGISTRY[name]
    V = CRUISE[name]["airspeed"]
    st = State(pos_ned=jnp.array([0.0, 0.0, -8000.0]),
               vel_body=jnp.array([V, 0.0, 0.0]),
               quat=euler_to_quat(jnp.array(0.0), jnp.array(np.pi / 2),
                                  jnp.array(0.0)),
               omega=Z3)
    _, hist = run(st, ctrl(), ac2, 0.01, 3000)
    fin = np.asarray(hist.quat)[-1]
    nrm = np.linalg.norm(np.asarray(hist.quat), axis=1)
    ok = np.all(np.isfinite(np.asarray(hist.vel_body)))
    print(f"  {name:20s} finite={ok}  quat norm dev max="
          f"{np.abs(nrm-1).max():.3e}  h_final="
          f"{-np.asarray(hist.pos_ned)[-1,2]:10.1f} m")

print("\n=== 4f. structurally singular Jacobian: cessna172 rudder ===")
ac2 = REGISTRY["cessna172"]
print(f"  cessna172 CYdr={float(ac2.CYdr)} Cldr={float(ac2.Cldr)} "
      f"Cndr={float(ac2.Cndr)}  -> rudder is inert")
st = trim.trimmed_state(jnp.array(0.05), jnp.array(60.0), jnp.array(1500.0))


def res3(u):
    c = Controls(elevator=u[0], aileron=u[1], rudder=u[2], throttle=jnp.array(0.5))
    return derivatives(st, c, ac2, Z3, Z3).omega


J = np.asarray(jax.jacfwd(res3)(jnp.array([0.0, 0.0, 0.0])))
print(f"  d(pdot,qdot,rdot)/d(de,da,dr) =\n{J}")
print(f"  det = {np.linalg.det(J):.3e}  cond = {np.linalg.cond(J):.3e}")
print(f"  jnp.linalg.solve(J, [1,1,1]) = "
      f"{np.asarray(jnp.linalg.solve(jnp.asarray(J), jnp.ones(3)))}")
print("  (any Newton solve that carries rudder as an unknown for this aircraft")
print("   inverts a singular matrix and returns NaN with no error raised)")
