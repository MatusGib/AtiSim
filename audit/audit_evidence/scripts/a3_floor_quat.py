"""A3: round-off floor evidence, and quaternion norm drift with/without renormalisation."""
import numpy as np
import jax, jax.numpy as jnp
import flightsim
from flightsim import integrate, trim, wind
from flightsim.aircraft import CRUISE, REGISTRY
from flightsim.dynamics import derivatives
from flightsim.integrate import SimState, rk4_step, _axpy
from flightsim.loads import zero_increment
from flightsim.state import quat_normalize

AC = REGISTRY["boeing747"]
V = CRUISE["boeing747"]["airspeed"]
H = CRUISE["boeing747"]["altitude"]
x_trim, _ = trim.trim(jnp.array(V), jnp.array(H), AC)

print("=" * 78)
print("### ROUND-OFF FLOOR")
print("=" * 78)
print(f"pos_ned at cruise = [944.48, 0, {-H:.3f}]")
print(f"  np.spacing(12184.567)      = {np.spacing(12184.567):.6e} m   (1 ulp, float64)")
print(f"  eps*|pD| = 2.22e-16*12184.6= {np.finfo(float).eps*12184.567:.6e} m")
print(f"  np.spacing(944.4786)       = {np.spacing(944.4786):.6e} m")
for dt in (1/4, 1/32, 1/128, 1/1024):
    n = int(round(4.0 / dt))
    print(f"  dt={dt:<10.6g} n={n:<6d} sqrt(n)*ulp/2 random walk = "
          f"{np.sqrt(n)*np.spacing(12184.567)/2:.3e} m   n*ulp/2 worst case = "
          f"{n*np.spacing(12184.567)/2:.3e} m")

print()
print("-- Direct probe: in STILL AIR pN and pE do not enter `derivatives` at all")
print("   (only pD does, through density). So offsetting the start north by D must")
print("   translate the answer by exactly D. Any residue is pure round-off.")


def run_still(dt, north0, t_end=4.0, d_elevator=0.02):
    state = trim.trimmed_state(x_trim[0], jnp.array(V), jnp.array(H))
    state = state._replace(pos_ned=state.pos_ned.at[0].set(jnp.asarray(north0, float)))
    controls = trim.trimmed_controls(x_trim[1] + d_elevator, x_trim[2])
    sim = integrate.init_sim(state, jax.random.PRNGKey(0))
    final, _ = integrate.rollout(sim, controls, jnp.array(dt), AC, int(round(t_end/dt)))
    s = final.state
    return np.concatenate([np.asarray(s.pos_ned), np.asarray(s.vel_body),
                           np.asarray(s.quat), np.asarray(s.omega)])


print(f"{'dt':>10} {'n':>6} {'|residue pN| [m]':>18} {'|residue pD| [m]':>18} "
      f"{'|residue vel|':>14} {'|residue quat|':>15}")
for dt in (1/4, 1/16, 1/64, 1/128, 1/256, 1/1024):
    a = run_still(dt, 0.0)
    D = 1.0e6
    b = run_still(dt, D)
    print(f"{dt:>10.6g} {int(round(4.0/dt)):>6d} {abs(b[0]-D-a[0]):>18.6e} "
          f"{abs(b[2]-a[2]):>18.6e} {np.linalg.norm(b[3:6]-a[3:6]):>14.6e} "
          f"{np.linalg.norm(b[6:10]-a[6:10]):>15.6e}")

print()
print("-- Determinism: is the same call bit-identical across invocations? --")
a1 = run_still(1/128, 0.0); a2 = run_still(1/128, 0.0)
print("   identical bits:", (a1.tobytes() == a2.tobytes()))

print()
print("-- 1-ulp perturbation amplification through the rollout (still air, dt=1/128) --")
base = run_still(1/128, 0.0)
pert = run_still(1/128, np.spacing(944.4786))   # perturb start north by 1 ulp
print(f"   input perturbation  = {np.spacing(944.4786):.6e} m")
print(f"   output |dpos|       = {np.linalg.norm(pert[0:3]-base[0:3]):.6e} m")

print()
print("=" * 78)
print("### QUATERNION NORM DRIFT")
print("=" * 78)


def step_no_renorm(sim, controls, dt, ac, wind_model=wind.zero_wind):
    """`integrate.step` with the quat_normalize line REMOVED. Nothing in
    flightsim is modified; this is a local copy of the same body."""
    wind_ned, omega_gust, wind_state, key = wind_model(sim.wind, sim.state, sim.key, dt)

    def f(s):
        return derivatives(s, controls, ac, wind_ned, omega_gust, increment=None)

    new_state = rk4_step(f, sim.state, dt)
    return SimState(state=new_state, wind=wind_state, key=key, wind_ned=wind_ned,
                    omega_gust=omega_gust, increment=zero_increment())


step_no_renorm_j = jax.jit(step_no_renorm, static_argnames=("wind_model",))


def rollout_variant(renorm, n_steps, dt, d_elevator=0.02, d_aileron=0.0, record_every=None):
    state = trim.trimmed_state(x_trim[0], jnp.array(V), jnp.array(H))
    controls = trim.Controls(elevator=x_trim[1] + d_elevator, aileron=jnp.array(d_aileron),
                             rudder=jnp.array(0.0), throttle=x_trim[2])
    sim = integrate.init_sim(state, jax.random.PRNGKey(0))
    fn = integrate.step if renorm else step_no_renorm_j
    dtj = jnp.array(dt)
    marks = set(record_every or [])
    hist = {}
    for i in range(1, n_steps + 1):
        sim = fn(sim, controls, dtj, AC)
        if i in marks:
            hist[i] = (float(jnp.linalg.norm(sim.state.quat)),
                       np.asarray(sim.state.pos_ned), np.asarray(sim.state.quat),
                       np.asarray(sim.state.vel_body), np.asarray(sim.state.omega))
    return hist


MARKS = [1, 10, 100, 1000, 10000, 100000]
print("\n-- Excited longitudinally + laterally (de +0.02, da +0.02), dt=0.02 --")
print("   ||q|| - 1 at each step count:")
h_on = rollout_variant(True, 100000, 0.02, d_aileron=0.02, record_every=MARKS)
h_off = rollout_variant(False, 100000, 0.02, d_aileron=0.02, record_every=MARKS)
print(f"{'steps':>8} {'renorm ON: ||q||-1':>22} {'renorm OFF: ||q||-1':>22} "
      f"{'|dpos| [m]':>14} {'|dquat|':>12} {'|dvel|':>12} {'|domega|':>12}")
for k in MARKS:
    n_on, p_on, q_on, v_on, w_on = h_on[k]
    n_off, p_off, q_off, v_off, w_off = h_off[k]
    print(f"{k:>8d} {n_on-1.0:>22.6e} {n_off-1.0:>22.6e} "
          f"{np.linalg.norm(p_on-p_off):>14.6e} {np.linalg.norm(q_on-q_off):>12.6e} "
          f"{np.linalg.norm(v_on-v_off):>12.6e} {np.linalg.norm(w_on-w_off):>12.6e}")
