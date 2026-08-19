"""A8: other numerically load-bearing things, plus energy conservation done cleanly."""
import numpy as np
import jax, jax.numpy as jnp
import flightsim
from flightsim import integrate, trim, validation, verification, wind, aero
from flightsim.aircraft import CRUISE, REGISTRY, inertia_tensor
from flightsim.atmosphere import G0, density, temperature, pressure
from flightsim.state import State, euler_to_quat, quat_to_dcm
from flightsim.tests.conftest import make_test_aircraft

print("=" * 78)
print("### 8a. lateral_modes: roll/spiral assignment sorts on tau, not on |tau|")
print("=" * 78)
for name in REGISTRY:
    V, H = CRUISE[name]["airspeed"], CRUISE[name]["altitude"]
    ac = REGISTRY[name]
    x, _ = trim.trim(jnp.array(V), jnp.array(H), ac)
    a, e, t = float(x[0]), float(x[1]), float(x[2])
    theta0 = a
    u0, w0 = V * np.cos(a), V * np.sin(a)
    from flightsim.state import Controls
    from flightsim.dynamics import derivatives
    ctl = Controls(elevator=jnp.array(e), aileron=jnp.array(0.0),
                   rudder=jnp.array(0.0), throttle=jnp.array(t))

    def f(xx):
        v, p, r, phi = xx
        st = State(pos_ned=jnp.array([0.0, 0.0, -H]),
                   vel_body=jnp.array([u0, v, w0]),
                   quat=euler_to_quat(phi, jnp.array(theta0), jnp.array(0.0)),
                   omega=jnp.array([p, 0.0, r]))
        d = derivatives(st, ctl, ac, jnp.zeros(3), jnp.zeros(3))
        return jnp.array([d.vel_body[1], d.omega[0], d.omega[2],
                          p + r * jnp.cos(phi) * jnp.tan(theta0)])

    A = np.asarray(jax.jacfwd(f)(jnp.array([0.0, 0.0, 0.0, 0.0])))
    eig = np.linalg.eigvals(A)
    reals = sorted([lam.real for lam in eig if abs(lam.imag) <= 1e-9])
    dr, rt, st_ = validation.lateral_modes(ac, a, e, t, V, H)
    print(f"\n{name}: real lateral eigenvalues = {np.round(reals, 8)}")
    print(f"   taus = -1/lambda            = {np.round([-1/r for r in reals], 6)}")
    print(f"   lateral_modes returns  roll_tau={rt:.6f}  spiral_tau={st_:.6f}")
    fast = min([abs(-1/r) for r in reals])
    print(f"   FASTEST |tau| (physically the roll subsidence) = {fast:.6f} s")
    if abs(abs(rt) - fast) > 1e-9:
        print("   *** MISMATCH: `reals.sort()` sorts on the SIGNED tau, so an")
        print("       UNSTABLE spiral (tau<0) sorts first and is labelled 'roll'. ***")

print()
print("=" * 78)
print("### 8b. ENERGY: a drag-free, thrust-free, moment-free aircraft")
print("=" * 78)
print("Lift in body axes is [L sin a, 0, -L cos a] and vel_rel is")
print("V[cos a cos b, sin b, sin a cos b]; the dot product is exactly zero, so")
print("lift does NO work and KE+PE must be conserved.")
AC0 = make_test_aircraft()._replace(
    CD0=jnp.array(0.0), e=jnp.array(1e12),   # kills induced drag
    Cm0=jnp.array(0.0), Cma=jnp.array(0.0), Cmq=jnp.array(0.0), Cmde=jnp.array(0.0),
    CYb=jnp.array(0.0), CYp=jnp.array(0.0), CYr=jnp.array(0.0), CYdr=jnp.array(0.0),
    Clb=jnp.array(0.0), Clp=jnp.array(0.0), Clr=jnp.array(0.0), Clda=jnp.array(0.0),
    Cldr=jnp.array(0.0), Cnb=jnp.array(0.0), Cnp=jnp.array(0.0), Cnr=jnp.array(0.0),
    Cnda=jnp.array(0.0), Cndr=jnp.array(0.0), max_thrust=jnp.array(0.0),
    CLq=jnp.array(0.0),
)
s0 = State(pos_ned=jnp.array([0.0, 0.0, -3000.0]),
           vel_body=jnp.array([70.0, 0.0, 3.0]),
           quat=euler_to_quat(jnp.array(0.0), jnp.array(0.05), jnp.array(0.0)),
           omega=jnp.zeros(3))
ctl = trim.trimmed_controls(jnp.array(0.0), jnp.array(0.0))
m = float(AC0.mass)
for dt, n in ((0.02, 50000), (0.02, 500000), (0.002, 500000), (0.1, 100000)):
    _, tr = integrate.rollout(integrate.init_sim(s0, jax.random.PRNGKey(0)),
                              ctl, jnp.array(dt), AC0, n)
    pos = np.asarray(tr.pos_ned); vel = np.asarray(tr.vel_body)
    E = 0.5 * m * (vel**2).sum(axis=1) + m * float(G0) * (-pos[:, 2])
    E0 = 0.5 * m * float((s0.vel_body**2).sum()) + m * float(G0) * 3000.0
    print(f"  dt={dt:<7g} n={n:<8d} T={dt*n:>8.0f} s   "
          f"max|E-E0|/E0={np.abs(E-E0).max()/E0:.4e}  final dE/E0={(E[-1]-E0)/E0:+.4e}  "
          f"alt span {(-pos[:,2]).min():.1f}..{(-pos[:,2]).max():.1f} m")

print()
print("=" * 78)
print("### 8c. ROTATIONAL INVARIANCE with the WIND FIELD rotated too")
print("=" * 78)
AC = REGISTRY["boeing747"]
V, H = CRUISE["boeing747"]["airspeed"], CRUISE["boeing747"]["altitude"]
xt, _ = trim.trim(jnp.array(V), jnp.array(H), AC)
state = trim.trimmed_state(xt[0], jnp.array(V), jnp.array(H))


def quat_mul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return jnp.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                      w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])


case = wind.PARKS_CASES["hannibal"]
arr = wind.VortexArray(north=jnp.array([300.0, 1000.0]), down=jnp.array([-H, -H + 200.0]),
                       r0=jnp.array(case["r0"]), v0=jnp.array(case["v0"]))


def base_field(p):
    return wind.vortex_wind(p, arr)


for psi in (0.7, -2.1):
    c, s = np.cos(psi), np.sin(psi)
    Rz = jnp.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    RzT = Rz.T

    def rot_field(p, Rz=Rz, RzT=RzT):
        return Rz @ base_field(RzT @ p)

    qz = euler_to_quat(jnp.array(0.0), jnp.array(0.0), jnp.array(psi))
    m0 = wind.field_model(base_field)
    m1 = wind.field_model(rot_field)
    b = state
    rstate = b._replace(pos_ned=Rz @ b.pos_ned, quat=quat_mul(qz, b.quat))
    ctl = trim.trimmed_controls(xt[1], xt[2])
    dt, n = jnp.array(0.02), 1500
    _, A = integrate.rollout(integrate.init_sim(b, jax.random.PRNGKey(0)), ctl, dt, AC, n,
                             wind_model=m0)
    _, B = integrate.rollout(integrate.init_sim(rstate, jax.random.PRNGKey(0)), ctl, dt, AC, n,
                             wind_model=m1)
    Rnp = np.asarray(Rz)
    pos_exp = np.asarray(A.pos_ned) @ Rnp.T
    q_exp = np.stack([np.asarray(quat_mul(qz, jnp.asarray(q))) for q in np.asarray(A.quat)])
    print(f"\npsi={psi} rad, 30 s, TWO Rankine vortices rotated with the problem")
    print(f"  max |vel_body diff| = {np.abs(np.asarray(B.vel_body)-np.asarray(A.vel_body)).max():.4e} m/s")
    print(f"  max |omega diff|    = {np.abs(np.asarray(B.omega)-np.asarray(A.omega)).max():.4e} rad/s")
    print(f"  max |pos - R pos|   = {np.abs(np.asarray(B.pos_ned)-pos_exp).max():.4e} m")
    print(f"  max |quat - qz q|   = {np.abs(np.asarray(B.quat)-q_exp).max():.4e}")
    print(f"  wind actually seen: max |wind_ned| along run A = "
          f"{np.abs(np.asarray(A.pos_ned)).max():.1f} m position scale")

print()
print("=" * 78)
print("### 8d. odds and ends")
print("=" * 78)
print(f"aero.V_MIN airspeed floor = {aero.V_MIN} m/s  (alpha/beta/rate denominators)")
print(f"trim.minimum_drag_speed grid: linspace(20, 400, 4000) -> resolution "
      f"{(400-20)/3999:.6f} m/s")
for name in REGISTRY:
    vmd = float(trim.minimum_drag_speed(REGISTRY[name], jnp.array(CRUISE[name]['altitude'])))
    print(f"   {name:<20s} V_md = {vmd:.6f} m/s   (cruise {CRUISE[name]['airspeed']:.3f} m/s)")

print(f"\ntrim.ALPHA_LIMIT = {np.degrees(trim.ALPHA_LIMIT):.1f} deg; `is_physical` checks "
      "alpha ONLY --")
print("   elevator and throttle are unbounded in `trim.trim` and unchecked in `is_physical`.")

print("\ndensity/temperature extrapolated BELOW sea level (a power-off run reached -3994 m):")
for h in (-5000.0, -3994.0, -1000.0, 0.0):
    print(f"   h={h:>9.1f} m  T={float(temperature(jnp.array(h))):.3f} K  "
          f"p={float(pressure(jnp.array(h))):.1f} Pa  rho={float(density(jnp.array(h))):.6f} kg/m^3")
print(f"   the troposphere formula has T -> 0 at h = {-288.15/0.0065:.1f} m")

print("\nverification.fitted_order raises on a zero/negative error:")
try:
    verification.fitted_order([0.1, 0.05], [1e-3, 0.0])
except Exception as exc:
    print(f"   {type(exc).__name__}: {exc}")

print("\nvalidation.modes_from_matrix drops any root with lam.imag <= 1e-9 (ABSOLUTE):")
print("   a mode of frequency below 1e-9 rad/s would be silently reported as two reals.")
print("   lateral_modes then does reals[0], reals[1] and would IndexError / mislabel.")

print("\nRankine core: one-sided d(w)/dr at r=r0 (ASSUMPTIONS E2 claims 2*V0/r0 jump)")
r0, v0 = case["r0"], case["v0"]
eps = 1e-6
p_in = jnp.array([0.0, 0.0, -H + (r0 - eps)])
p_out = jnp.array([0.0, 0.0, -H + (r0 + eps)])
arr1 = wind.VortexArray(north=jnp.array([0.0]), down=jnp.array([-H]),
                        r0=jnp.array(r0), v0=jnp.array(v0))
g_in = np.asarray(jax.jacfwd(lambda p: wind.vortex_wind(p, arr1))(p_in))
g_out = np.asarray(jax.jacfwd(lambda p: wind.vortex_wind(p, arr1))(p_out))
print(f"   d(w_horiz)/d(z) inside  = {g_in[0,2]:+.8f} 1/s")
print(f"   d(w_horiz)/d(z) outside = {g_out[0,2]:+.8f} 1/s")
print(f"   jump = {g_out[0,2]-g_in[0,2]:+.8f} 1/s ;  2*v0/r0 = {2*v0/r0:.8f} 1/s")

print("\nstate.quat_to_dcm does NOT normalise: det(DCM) for ||q||=1+d")
for d in (0.0, 1e-8, 1e-5, 1e-3):
    q = jnp.array([1.0, 0.0, 0.0, 0.0]) * (1.0 + d)
    print(f"   ||q||-1={d:<8g}  det = {float(jnp.linalg.det(quat_to_dcm(q))):.12f}  "
          f"(should be 1)")
