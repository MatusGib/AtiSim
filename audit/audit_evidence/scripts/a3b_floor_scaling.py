"""A3b: floor scaling with coordinate magnitude + high-rate quaternion drift."""
import numpy as np
import jax, jax.numpy as jnp
import flightsim
from flightsim import integrate, trim, wind
from flightsim.aircraft import CRUISE, REGISTRY, inertia_tensor
from flightsim.dynamics import derivatives
from flightsim.integrate import SimState, rk4_step
from flightsim.loads import zero_increment
from flightsim.state import State, euler_to_quat

AC = REGISTRY["boeing747"]
V = CRUISE["boeing747"]["airspeed"]
H = CRUISE["boeing747"]["altitude"]
x_trim, _ = trim.trim(jnp.array(V), jnp.array(H), AC)


def run_still(dt, north0, t_end=4.0):
    state = trim.trimmed_state(x_trim[0], jnp.array(V), jnp.array(H))
    state = state._replace(pos_ned=state.pos_ned.at[0].set(jnp.asarray(north0, float)))
    controls = trim.trimmed_controls(x_trim[1] + 0.02, x_trim[2])
    sim = integrate.init_sim(state, jax.random.PRNGKey(0))
    final, _ = integrate.rollout(sim, controls, jnp.array(dt), AC, int(round(t_end/dt)))
    return np.asarray(final.state.pos_ned)


print("### floor scales with the MAGNITUDE of the accumulator, not with dt alone")
print("north-offset residue  |pN(D) - D - pN(0)|  for several D, dt=1/1024 (n=4096)")
print(f"{'D [m]':>14} {'ulp(D)':>14} {'residue [m]':>14} {'residue/ulp(D)':>16}")
base = run_still(1/1024, 0.0)
for D in (1.0e3, 1.2192e4, 1.0e5, 1.0e6, 1.0e7):
    b = run_still(1/1024, D)
    res = abs(b[0] - D - base[0])
    print(f"{D:>14.4g} {np.spacing(D):>14.4e} {res:>14.4e} {res/np.spacing(D):>16.2f}")

print()
print("At the cruise altitude magnitude |pD| = 12192 m, ulp = "
      f"{np.spacing(12192.0):.4e} m, and over n=4096 steps a random walk of half-ulps")
print(f"is {np.sqrt(4096)*np.spacing(12192.0)/2:.3e} m. "
      "The project's stated floor is ~7e-11 m.")

print()
print("### observed still-air floor (repeat of the refinement, focused)")
dts = [1/64, 1/128, 1/256, 1/512, 1/1024, 1/2048]
res = {d: run_still(d, 0.0) for d in dts}
ref = res[1/2048]
print(f"{'dt':>12} {'|pos-ref| [m]':>16}")
for d in dts[:-1]:
    print(f"{d:>12.6g} {np.linalg.norm(res[d]-ref):>16.6e}")

# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("### QUATERNION NORM DRIFT at HIGH angular rate (no aero, torque-free)")
print("=" * 78)
_I1, _I2, _I3 = 1420.0, 4070.0, 4780.0
inertia = inertia_tensor(_I1, _I2, _I3, 0.0)
zeroed = dict(CL0=0.0, CLa=0.0, CLq=0.0, CLde=0.0, Cm0=0.0, Cma=0.0, Cmq=0.0,
              Cmde=0.0, CD0=0.0, CYb=0.0, CYp=0.0, CYr=0.0, CYdr=0.0, Clb=0.0,
              Clp=0.0, Clr=0.0, Clda=0.0, Cldr=0.0, Cnb=0.0, Cnp=0.0, Cnr=0.0,
              Cnda=0.0, Cndr=0.0, max_thrust=0.0)
from flightsim.tests.conftest import make_test_aircraft
AC_TF = make_test_aircraft()._replace(inertia=inertia,
                                      inertia_inv=jnp.linalg.inv(inertia),
                                      **{k: jnp.array(v) for k, v in zeroed.items()})
OM0 = jnp.array([0.6, 0.0, 0.9])
S0 = State(pos_ned=jnp.array([0.0, 0.0, -3000.0]), vel_body=jnp.array([60.0, 0.0, 0.0]),
           quat=euler_to_quat(jnp.array(0.0), jnp.array(0.0), jnp.array(0.0)), omega=OM0)
CTRL = trim.trimmed_controls(jnp.array(0.0), jnp.array(0.0))


def step_no_renorm(sim, controls, dt, ac):
    wind_ned, omega_gust, ws, key = wind.zero_wind(sim.wind, sim.state, sim.key, dt)

    def f(s):
        return derivatives(s, controls, ac, wind_ned, omega_gust, increment=None)
    return SimState(state=rk4_step(f, sim.state, dt), wind=ws, key=key,
                    wind_ned=wind_ned, omega_gust=omega_gust, increment=zero_increment())


snr = jax.jit(step_no_renorm)
MARKS = [1, 10, 100, 1000, 10000, 100000]


def drift(renorm, dt, n_max, marks):
    sim = integrate.init_sim(S0, jax.random.PRNGKey(0))
    fn = integrate.step if renorm else snr
    dtj = jnp.array(dt)
    out = {}
    ms = set(marks)
    for i in range(1, n_max + 1):
        sim = fn(sim, CTRL, dtj, AC_TF) if renorm else fn(sim, CTRL, dtj, AC_TF)
        if i in ms:
            out[i] = (float(jnp.linalg.norm(sim.state.quat)),
                      np.asarray(sim.state.quat), np.asarray(sim.state.omega),
                      np.asarray(sim.state.pos_ned))
    return out


for dt in (0.02, 0.1):
    print(f"\n-- torque-free, |omega|={float(jnp.linalg.norm(OM0)):.3f} rad/s, dt={dt} --")
    on = drift(True, dt, 100000, MARKS)
    off = drift(False, dt, 100000, MARKS)
    print(f"{'steps':>8} {'ON ||q||-1':>16} {'OFF ||q||-1':>16} {'|dquat|':>13} "
          f"{'|domega|':>13} {'|dpos| [m]':>13}")
    for k in MARKS:
        print(f"{k:>8d} {on[k][0]-1:>16.6e} {off[k][0]-1:>16.6e} "
              f"{np.linalg.norm(on[k][1]-off[k][1]):>13.6e} "
              f"{np.linalg.norm(on[k][2]-off[k][2]):>13.6e} "
              f"{np.linalg.norm(on[k][3]-off[k][3]):>13.6e}")
