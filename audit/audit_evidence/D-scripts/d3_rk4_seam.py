"""D3: the RK4 wind-sampling seam.

integrate.step samples the wind ONCE per step and holds it across all four RK4
stages. This file re-implements `step` here (NOT in flightsim/) with the wind
re-sampled at every stage, and measures the observed order of accuracy and the
trajectory cost of the hold.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

import flightsim  # noqa: F401
from flightsim import integrate, trim, wind
from flightsim.aircraft import CRUISE, REGISTRY
from flightsim.dynamics import derivatives
from flightsim.integrate import SimState, _axpy
from flightsim.state import quat_normalize
import wcommon as W

AC = REGISTRY["boeing747"]
V = float(CRUISE["boeing747"]["airspeed"])
H = float(CRUISE["boeing747"]["altitude"])


# --- SCRATCH COPY of integrate.step, wind re-sampled per RK4 stage ------------
@partial(jax.jit, static_argnames=("wind_model",))
def step_perstage(sim, controls, dt, ac, wind_model):
    """Identical to integrate.step except the wind model is called inside f().

    Deterministic fields only: the key is threaded unchanged, which is exactly
    what field_model does anyway, so no key is consumed four times.
    """
    def f(s):
        w, og, _, _ = wind_model(sim.wind, s, sim.key, dt)
        return derivatives(s, controls, ac, w, og, increment=None)

    new_state = integrate.rk4_step(f, sim.state, dt)
    new_state = new_state._replace(quat=quat_normalize(new_state.quat))
    w0, og0, ws, key = wind_model(sim.wind, sim.state, sim.key, dt)
    return SimState(state=new_state, wind=ws, key=key,
                    wind_ned=w0, omega_gust=og0, increment=sim.increment)


@partial(jax.jit, static_argnames=("n_steps", "wind_model", "per_stage"))
def run(sim, controls, dt, ac, n_steps, wind_model, per_stage):
    stepper = step_perstage if per_stage else (
        lambda s, c, d, a, wm: integrate.step(s, c, d, a, wind_model=wm))

    def body(carry, _):
        return stepper(carry, controls, dt, ac, wind_model), None

    out, _ = jax.lax.scan(body, sim, None, length=n_steps)
    return out


def setup(start_north, altitude=H, airspeed=V):
    x, _ = trim.trim(jnp.array(airspeed), jnp.array(altitude), AC)
    controls = trim.trimmed_controls(x[1], x[2])
    state = trim.trimmed_state(jnp.array(float(x[0])), jnp.array(airspeed),
                               jnp.array(altitude))
    state = state._replace(pos_ned=jnp.array([start_north, 0.0, -altitude]))
    return integrate.init_sim(state, jax.random.PRNGKey(0)), controls


def flatten(s):
    return np.concatenate([np.asarray(s.state.pos_ned), np.asarray(s.state.vel_body),
                           np.asarray(s.state.quat), np.asarray(s.state.omega)])


MODEL = wind.field_model(W.vortex_field)

print("=" * 78)
print("D3a  OBSERVED ORDER OF ACCURACY, Parks hannibal core traverse")
print("=" * 78)
# Short traverse: enter 2 core radii upstream, exit 4 downstream, so the whole
# integration interval is inside the region where the field actually varies.
START = -2.0 * W.R0
T_END = 6.0 * W.R0 / V
print(f"747 CR-2144 FC9: V = {V:.3f} m/s, H = {H:.1f} m")
print(f"start north = {START:.1f} m, T = {T_END:.5f} s "
      f"(6 core radii, r0 = {W.R0:.2f} m)\n")

sim0, controls = setup(START)

# reference: per-stage RK4, tiny step -- the true ODE solution
dt_ref = T_END / 262144
ref = run(sim0, controls, jnp.array(dt_ref), AC, 262144, MODEL, True)
xref = flatten(ref)
print(f"reference: per-stage RK4, dt = {dt_ref:.3e} s, {262144} steps")

# convergence check on the reference itself
ref2 = run(sim0, controls, jnp.array(T_END / 131072), AC, 131072, MODEL, True)
print(f"reference self-consistency (dt vs 2dt): "
      f"|dpos| = {np.linalg.norm(flatten(ref2)[:3] - xref[:3]):.3e} m\n")

ns = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
print(f"{'n_steps':>8s} {'dt (s)':>10s} | {'HELD |dpos| (m)':>17s} {'ord':>6s} | "
      f"{'PER-STAGE |dpos|':>17s} {'ord':>6s}")
prev_h = prev_p = None
rows = []
for n in ns:
    dt = T_END / n
    eh = np.linalg.norm(flatten(run(sim0, controls, jnp.array(dt), AC, n, MODEL, False))[:3] - xref[:3])
    ep = np.linalg.norm(flatten(run(sim0, controls, jnp.array(dt), AC, n, MODEL, True))[:3] - xref[:3])
    oh = np.log2(prev_h / eh) if prev_h else float("nan")
    op = np.log2(prev_p / ep) if prev_p else float("nan")
    print(f"{n:8d} {dt:10.5f} | {eh:17.4e} {oh:6.2f} | {ep:17.4e} {op:6.2f}")
    rows.append((n, dt, eh, ep))
    prev_h, prev_p = eh, ep

print("\nfitted slope of log|err| vs log(dt), over the three finest usable steps:")
for label, idx in (("held", 2), ("per-stage", 3)):
    a = np.array([r for r in rows if r[idx] > 1e-12])
    if len(a) >= 3:
        use = a[-4:-1] if label == "per-stage" else a[-3:]
        sl = np.polyfit(np.log(use[:, 1]), np.log(use[:, idx]), 1)[0]
        print(f"  {label:10s} order = {sl:.3f}")

print()
print("=" * 78)
print("D3b  WHAT THE HOLD COSTS IN THE PROJECT'S ACTUAL RUN")
print("=" * 78)
LEAD = 40.0 * W.R0
SECONDS = (W.SPACING + LEAD + 6.0 * W.R0) / V
print(f"scripts/vortex.py geometry: lead-in 40 r0 = {LEAD:.1f} m, "
      f"{SECONDS:.3f} s of flight")
sim1, controls1 = setup(-LEAD)

for dt in (0.02, 0.01):
    n = int(round(SECONDS / dt))
    held = run(sim1, controls1, jnp.array(dt), AC, n, MODEL, False)
    per = run(sim1, controls1, jnp.array(dt), AC, n, MODEL, True)
    dh, dp = flatten(held), flatten(per)
    print(f"\n  dt = {dt} s, {n} steps")
    print(f"    final position   held {np.asarray(held.state.pos_ned)}")
    print(f"                per-stage {np.asarray(per.state.pos_ned)}")
    print(f"    |d pos| = {np.linalg.norm(dh[:3]-dp[:3]):.6e} m"
          f"   (d altitude = {(dh[2]-dp[2]):+.6e} m)")
    print(f"    |d vel_body| = {np.linalg.norm(dh[3:6]-dp[3:6]):.6e} m/s")
    print(f"    |d omega|    = {np.linalg.norm(dh[10:13]-dp[10:13]):.6e} rad/s")
    print(f"    |d quat|     = {np.linalg.norm(dh[6:10]-dp[6:10]):.6e}")

# whole-trajectory divergence, dt = 0.02, and the analysed quantities
print("\n  per-sample trajectory divergence, dt = 0.02:")
dt = 0.02
n = int(round(SECONDS / dt))


@partial(jax.jit, static_argnames=("n_steps", "per_stage"))
def traj(sim, controls, dt, n_steps, per_stage):
    stepper = step_perstage if per_stage else (
        lambda s, c, d, a, wm: integrate.step(s, c, d, a, wind_model=wm))

    def body(carry, _):
        c = stepper(carry, controls, dt, AC, MODEL)
        return c, c.state

    return jax.lax.scan(body, sim, None, length=n_steps)[1]


th = traj(sim1, controls1, jnp.array(dt), n, False)
tp = traj(sim1, controls1, jnp.array(dt), n, True)
pos_h, pos_p = np.asarray(th.pos_ned), np.asarray(tp.pos_ned)
alt_h, alt_p = -pos_h[:, 2], -pos_p[:, 2]
print(f"    max |d north|    = {np.abs(pos_h[:,0]-pos_p[:,0]).max():.4e} m")
print(f"    max |d altitude| = {np.abs(alt_h-alt_p).max():.4e} m")
print(f"    altitude excursion of the run itself = "
      f"{alt_h.max()-alt_h.min():.4f} m  -> relative {np.abs(alt_h-alt_p).max()/(alt_h.max()-alt_h.min()):.3e}")
q_h, q_p = np.asarray(th.omega)[:, 1], np.asarray(tp.omega)[:, 1]
print(f"    max |d q|        = {np.abs(q_h-q_p).max():.4e} rad/s"
      f"   (peak |q| in the run = {np.abs(q_h).max():.4e})")
print(f"    -> relative {np.abs(q_h-q_p).max()/np.abs(q_h).max():.3e}")

# and against the dt->0 answer, so the hold's error is placed next to the
# discretisation error the run already carries
ref_run = run(sim1, controls1, jnp.array(SECONDS / 200000), AC, 200000, MODEL, True)
xr = flatten(ref_run)
for dt in (0.02, 0.01):
    n = int(round(SECONDS / dt))
    eh = np.linalg.norm(flatten(run(sim1, controls1, jnp.array(dt), AC, n, MODEL, False))[:3] - xr[:3])
    ep = np.linalg.norm(flatten(run(sim1, controls1, jnp.array(dt), AC, n, MODEL, True))[:3] - xr[:3])
    print(f"\n  dt={dt}: |pos err vs dt->0|  held {eh:.4e} m   per-stage {ep:.4e} m"
          f"   ratio {eh/max(ep,1e-300):.3g}")
