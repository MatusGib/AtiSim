"""A4: conservation and invariance."""
import numpy as np
import jax, jax.numpy as jnp
import flightsim
from flightsim import integrate, trim, wind, verification
from flightsim.aircraft import CRUISE, REGISTRY, inertia_tensor
from flightsim.atmosphere import G0, density
from flightsim.state import State, euler_to_quat, quat_to_dcm, quat_to_euler
from flightsim.tests.conftest import make_test_aircraft

np.set_printoptions(precision=12, suppress=False)

# --------------------------------------------------------------- torque-free
print("=" * 78)
print("### 4a. TORQUE-FREE: angular momentum and rotational KE drift")
print("=" * 78)
_I1, _I2, _I3 = 1420.0, 4070.0, 4780.0
inertia = inertia_tensor(_I1, _I2, _I3, 0.0)
zeroed = dict(CL0=0.0, CLa=0.0, CLq=0.0, CLde=0.0, Cm0=0.0, Cma=0.0, Cmq=0.0,
              Cmde=0.0, CD0=0.0, CYb=0.0, CYp=0.0, CYr=0.0, CYdr=0.0, Clb=0.0,
              Clp=0.0, Clr=0.0, Clda=0.0, Cldr=0.0, Cnb=0.0, Cnp=0.0, Cnr=0.0,
              Cnda=0.0, Cndr=0.0, max_thrust=0.0)
AC_TF = make_test_aircraft()._replace(inertia=inertia, inertia_inv=jnp.linalg.inv(inertia),
                                      **{k: jnp.array(v) for k, v in zeroed.items()})
OM0 = np.array([0.6, 0.0, 0.9])
S0 = State(pos_ned=jnp.array([0.0, 0.0, -3000.0]), vel_body=jnp.array([60.0, 0.0, 0.0]),
           quat=euler_to_quat(jnp.array(0.0), jnp.array(0.0), jnp.array(0.0)),
           omega=jnp.array(OM0))
CTRL = trim.trimmed_controls(jnp.array(0.0), jnp.array(0.0))
I = np.diag([_I1, _I2, _I3])

for dt, n in ((0.002, 30000), (0.02, 60000), (0.02, 300000)):
    sim = integrate.init_sim(S0, jax.random.PRNGKey(0))
    _, traj = integrate.rollout(sim, CTRL, jnp.array(dt), AC_TF, n)
    om = np.asarray(traj.omega)          # (n, 3) body
    q = np.asarray(traj.quat)            # (n, 4)
    # body-frame |H| is invariant for a torque-free body; also check the INERTIAL H
    Hb = om @ I
    Hb_norm = np.linalg.norm(Hb, axis=1)
    T = 0.5 * np.einsum("ij,jk,ik->i", om, I, om)
    H0 = np.linalg.norm(I @ OM0); T0 = 0.5 * OM0 @ I @ OM0
    # inertial angular momentum vector
    def dcm(qq):
        w, x, y, z = qq
        return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                         [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                         [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])
    Hi = np.stack([dcm(q[i]) @ Hb[i] for i in range(0, n, max(1, n // 2000))])
    Hi0 = dcm(np.asarray(S0.quat)) @ (I @ OM0)
    print(f"\ndt={dt}, n={n}, t_end={dt*n:g} s   H0={H0:.9f} kg m^2/s  T0={T0:.9f} J")
    print(f"  max |H|/H0 - 1                     = {np.abs(Hb_norm/H0 - 1).max():.6e}")
    print(f"  final |H|/H0 - 1                   = {Hb_norm[-1]/H0 - 1:+.6e}")
    print(f"  max |T/T0 - 1|                     = {np.abs(T/T0 - 1).max():.6e}")
    print(f"  final T/T0 - 1                     = {T[-1]/T0 - 1:+.6e}")
    print(f"  max |H_inertial - H_inertial(0)|   = "
          f"{np.abs(Hi - Hi0).max():.6e}  (direction drift, kg m^2/s)")
    print(f"  ||q|| - 1 final                    = {np.linalg.norm(q[-1]) - 1:+.6e}")

# ------------------------------------------------------- Galilean invariance
print()
print("=" * 78)
print("### 4b. GALILEAN INVARIANCE under a uniform STEADY wind")
print("=" * 78)
AC = REGISTRY["boeing747"]
V = CRUISE["boeing747"]["airspeed"]; H = CRUISE["boeing747"]["altitude"]
xt, _ = trim.trim(jnp.array(V), jnp.array(H), AC)
state = trim.trimmed_state(xt[0], jnp.array(V), jnp.array(H))
controls = trim.trimmed_controls(xt[1] + 0.01, xt[2])

for Wv in ([7.0, -3.0, 0.0], [70.0, -30.0, 0.0], [0.0, 0.0, 2.0]):
    W = jnp.array(Wv)

    def uniform(ws, s, key, dt, W=W):
        return W, jnp.zeros(3), ws, key

    dcm = quat_to_dcm(state.quat)
    shifted = state._replace(vel_body=state.vel_body + dcm.T @ W)
    dt, n = jnp.array(0.02), 1000
    _, still = integrate.rollout(integrate.init_sim(state, jax.random.PRNGKey(0)),
                                 controls, dt, AC, n)
    _, blown = integrate.rollout(integrate.init_sim(shifted, jax.random.PRNGKey(0)),
                                 controls, dt, AC, n, wind_model=uniform)
    t = np.arange(1, n + 1) * float(dt)
    exp_pos = np.asarray(still.pos_ned) + t[:, None] * np.asarray(W)
    print(f"\nW = {Wv} m/s NED, 20 s at dt=0.02")
    print(f"  max |dquat|          = {np.abs(np.asarray(blown.quat)-np.asarray(still.quat)).max():.6e}")
    print(f"  max |domega| [rad/s] = {np.abs(np.asarray(blown.omega)-np.asarray(still.omega)).max():.6e}")
    print(f"  max |dvel_body|      = {np.abs(np.asarray(blown.vel_body)-np.asarray(still.vel_body)-(np.asarray(dcm.T@W))).max():.6e}")
    print(f"  max |pos - (pos_still + W t)| [m] = {np.abs(np.asarray(blown.pos_ned)-exp_pos).max():.6e}")

# ------------------------------------------------------ Rotational invariance
print()
print("=" * 78)
print("### 4c. ROTATIONAL INVARIANCE about the NED z (down) axis")
print("=" * 78)
print("Gravity is along NED +z, so ONLY rotations about z are a symmetry of the")
print("problem. Rotating the whole configuration by psi about z must rotate the")
print("whole trajectory by psi and leave body-frame quantities untouched.")


def rotz(psi):
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def quat_mul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return jnp.array([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                      w1*x2 + x1*w2 + y1*z2 - z1*y2,
                      w1*y2 - x1*z2 + y1*w2 + z1*x2,
                      w1*z2 + x1*y2 - y1*x2 + z1*w2])


for psi in (0.7, 2.5, -1.9):
    qz = euler_to_quat(jnp.array(0.0), jnp.array(0.0), jnp.array(psi))
    R = rotz(psi)
    ctrl = trim.Controls(elevator=xt[1] + 0.01, aileron=jnp.array(0.015),
                         rudder=jnp.array(-0.01), throttle=xt[2])
    base = state._replace(omega=jnp.array([0.02, -0.01, 0.03]))
    rotated = base._replace(pos_ned=jnp.asarray(R) @ base.pos_ned,
                            quat=quat_mul(qz, base.quat))  # same body vel & omega
    dt, n = jnp.array(0.02), 2000
    _, a = integrate.rollout(integrate.init_sim(base, jax.random.PRNGKey(0)), ctrl, dt, AC, n)
    _, b = integrate.rollout(integrate.init_sim(rotated, jax.random.PRNGKey(0)), ctrl, dt, AC, n)
    pos_exp = np.asarray(a.pos_ned) @ R.T
    q_exp = np.stack([np.asarray(quat_mul(qz, jnp.asarray(qq))) for qq in np.asarray(a.quat)])
    print(f"\npsi = {psi} rad, 40 s at dt=0.02, aileron+rudder excited")
    print(f"  max |vel_body diff|   = {np.abs(np.asarray(b.vel_body)-np.asarray(a.vel_body)).max():.6e} m/s")
    print(f"  max |omega diff|      = {np.abs(np.asarray(b.omega)-np.asarray(a.omega)).max():.6e} rad/s")
    print(f"  max |pos - R pos|     = {np.abs(np.asarray(b.pos_ned)-pos_exp).max():.6e} m")
    print(f"  max |quat - qz*quat|  = {np.abs(np.asarray(b.quat)-q_exp).max():.6e}")
    print(f"  |pos| scale           = {np.abs(pos_exp).max():.4g} m")

print()
print("-- control: a rotation about NED x (NOT a symmetry, gravity is along z) --")
psi = 0.3
qx = euler_to_quat(jnp.array(psi), jnp.array(0.0), jnp.array(0.0))
Rx = np.array([[1, 0, 0], [0, np.cos(psi), -np.sin(psi)], [0, np.sin(psi), np.cos(psi)]])
base = state._replace(omega=jnp.array([0.02, -0.01, 0.03]))
rotated = base._replace(pos_ned=jnp.asarray(Rx) @ base.pos_ned, quat=quat_mul(qx, base.quat))
ctrl = trim.Controls(elevator=xt[1] + 0.01, aileron=jnp.array(0.015),
                     rudder=jnp.array(-0.01), throttle=xt[2])
dt, n = jnp.array(0.02), 2000
_, a = integrate.rollout(integrate.init_sim(base, jax.random.PRNGKey(0)), ctrl, dt, AC, n)
_, b = integrate.rollout(integrate.init_sim(rotated, jax.random.PRNGKey(0)), ctrl, dt, AC, n)
print(f"  max |vel_body diff| = {np.abs(np.asarray(b.vel_body)-np.asarray(a.vel_body)).max():.6e} m/s"
      "   <- expected to be LARGE; this is the control that the test above is not vacuous")

# ------------------------------------------------------------------- energy
print()
print("=" * 78)
print("### 4d. ENERGY in still air over a long horizon")
print("=" * 78)
for label, thr_delta, n in (("trimmed, powered", 0.0, 500000),
                            ("power off (throttle=0)", None, 100000)):
    thr = jnp.array(0.0) if thr_delta is None else xt[2]
    ctrl = trim.trimmed_controls(xt[1], thr)
    dt = jnp.array(0.02)
    sim = integrate.init_sim(trim.trimmed_state(xt[0], jnp.array(V), jnp.array(H)),
                             jax.random.PRNGKey(0))
    _, traj = integrate.rollout(sim, ctrl, dt, AC, n)
    pos = np.asarray(traj.pos_ned); vel = np.asarray(traj.vel_body)
    m = float(AC.mass)
    KE = 0.5 * m * (vel ** 2).sum(axis=1)
    PE = m * float(G0) * (-pos[:, 2])
    E = KE + PE
    t = np.arange(1, n + 1) * float(dt)
    print(f"\n{label}: {n} steps = {n*float(dt):g} s")
    print(f"  E(0)   = {E[0]:.6e} J    E(end) = {E[-1]:.6e} J   dE/E0 = {(E[-1]-E[0])/E[0]:+.6e}")
    print(f"  altitude 0 -> end : {-pos[0,2]:.4f} -> {-pos[-1,2]:.4f} m")
    print(f"  |V| 0 -> end      : {np.linalg.norm(vel[0]):.6f} -> {np.linalg.norm(vel[-1]):.6f} m/s")
    print(f"  max |E-E0|/E0 over the run = {np.abs(E-E[0]).max()/E[0]:.6e}")
