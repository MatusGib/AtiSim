"""ATTACK 10c: STEADY LEVEL turn, 7 unknowns, 7 equations.

unknowns  [alpha, theta, de, da, dr, thr, psidot],  beta fixed 0
equations [udot, vdot, wdot, pdot, qdot, rdot, hdot] = 0

Kinematic identities that must hold regardless of aerodynamics:
  |a_spec|/g == sqrt(1 + (V*psidot/g)^2)          (level steady turn)
  n_z        == -(a_spec . z_body)/g
and the coordinated-turn relation tan(phi) == V*psidot/g holds only when the
net side specific force is zero.
"""
from common import *  # noqa

from flightsim.dynamics import derivatives, load_factor, specific_force
from flightsim.state import quat_to_dcm

Z3 = jnp.zeros(3)


def level_turn(ac, V, H, phi, x0):
    pos = jnp.array([0.0, 0.0, -H])

    def build(x):
        alpha, theta, de, da, dr, thr, psidot = x
        p = -psidot * jnp.sin(theta)
        q = psidot * jnp.sin(phi) * jnp.cos(theta)
        r = psidot * jnp.cos(phi) * jnp.cos(theta)
        vel = V * jnp.array([jnp.cos(alpha), 0.0, jnp.sin(alpha)])  # beta = 0
        st = State(pos_ned=pos, vel_body=vel,
                   quat=euler_to_quat(phi, theta, jnp.array(0.0)),
                   omega=jnp.array([p, q, r]))
        return st, Controls(elevator=de, aileron=da, rudder=dr, throttle=thr)

    def res(x):
        st, c = build(x)
        d = derivatives(st, c, ac, Z3, Z3)
        hdot = -(quat_to_dcm(st.quat) @ st.vel_body)[2]
        return jnp.concatenate([d.vel_body, d.omega, jnp.array([hdot])])

    x = x0
    for _ in range(200):
        J = jax.jacfwd(res)(x)
        dx = jnp.linalg.solve(J, res(x))
        nrm = jnp.linalg.norm(dx)
        x = x - jnp.where(nrm > 0.2, 0.2 / nrm, 1.0) * dx
    st, c = build(x)
    return x, st, c, float(jnp.linalg.norm(res(x)))


print("=== STEADY LEVEL coordinated-ish turn ===")
print(f"{'aircraft':20s} {'phi':>5s} {'n_z':>8s} {'|a_sp|/g':>9s} "
      f"{'sqrt(1+(Vpd/g)^2)':>18s} {'1/cos(phi)':>11s} {'Vpd/g':>8s} "
      f"{'resid':>9s} {'alpha_deg':>9s}")
for name in NAMES:
    ac = REGISTRY[name]
    V, H = CRUISE[name]["airspeed"], CRUISE[name]["altitude"]
    xt, _ = trim.trim(jnp.array(V), jnp.array(H), ac)
    x0 = jnp.array([float(xt[0]), float(xt[0]), float(xt[1]), 0.0, 0.0,
                    float(xt[2]), 0.0])
    for phi_deg in (0.0, 15.0, 30.0, 45.0, 60.0):
        phi = jnp.array(np.radians(phi_deg))
        # continuation from the previous bank angle
        x, st, c, rn = level_turn(ac, V, H, phi, x0)
        n = float(load_factor(st, c, ac, Z3, Z3))
        asp = np.asarray(specific_force(st, c, ac, Z3, Z3))
        mag = float(np.linalg.norm(asp))
        psidot = float(x[6])
        kin = np.sqrt(1.0 + (V * psidot / G0) ** 2)
        ref = 1.0 / np.cos(np.radians(phi_deg))
        print(f"{name:20s} {phi_deg:5.1f} {n:8.5f} {mag:9.5f} {kin:18.5f} "
              f"{ref:11.5f} {V*psidot/G0:8.5f} {rn:9.1e} "
              f"{np.degrees(float(x[0])):9.3f}")
        if rn < 1e-8:
            x0 = x
