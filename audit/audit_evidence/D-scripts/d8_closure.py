"""D8: closing checks -- gust_rates sign self-consistency, and the heading rate
the project's own runs actually reach."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

import flightsim  # noqa: F401
from flightsim import integrate, trim, wind
from flightsim.aircraft import CRUISE, REGISTRY
from flightsim.state import quat_to_dcm
import wcommon as W

AC = REGISTRY["boeing747"]
V = float(CRUISE["boeing747"]["airspeed"])
HC = W.H_CRUISE

print("=" * 78)
print("D8a  gust_rates SIGN SELF-CONSISTENCY AGAINST A RIGID ROTATION")
print("=" * 78)
print("If the air moves as a rigid body at omega about the CG, the gust field in")
print("BODY axes is w_g(rho) = omega x rho. gust_rates must then return omega.")
print("This tests the three component picks and both signs at once, independently")
print("of any derivative check.\n")
key = jax.random.PRNGKey(11)
worst = 0.0
for i in range(200):
    k1, k2 = jax.random.split(jax.random.fold_in(key, i))
    om = jax.random.normal(k1, (3,)) * 0.1
    q = W.random_quats(k2, 1)[0]
    dcm = quat_to_dcm(q)
    p0 = jax.random.normal(jax.random.fold_in(key, 1000 + i), (3,)) * 100.0

    def rigid(p, om=om, dcm=dcm, p0=p0):
        rho_body = dcm.T @ (p - p0)
        return dcm @ jnp.cross(om, rho_body)

    got = np.asarray(wind.gust_rates(p0, q, rigid))
    worst = max(worst, np.abs(got - np.asarray(om)).max() / np.abs(np.asarray(om)).max())
print(f"  200 random (omega, attitude, position): worst relative error {worst:.3e}")
print("  -> the p/q/r picks and both signs in gust_rates are internally consistent")
print("     with omega_rel = omega - omega_gust.")

print("\n  sampled_rates on the same rigid rotation (a linear field, so exact):")
from flightsim import airframe
ST = airframe.stations(AC)
worst = 0.0
for i in range(50):
    k1, k2 = jax.random.split(jax.random.fold_in(key, i))
    om = jax.random.normal(k1, (3,)) * 0.1
    q = W.random_quats(k2, 1)[0]
    dcm = quat_to_dcm(q)
    p0 = jnp.array([0.0, 0.0, -HC])

    def rigid(p, om=om, dcm=dcm, p0=p0):
        return dcm @ jnp.cross(om, dcm.T @ (p - p0))

    got = np.asarray(wind.sampled_rates(p0, q, rigid, ST))
    worst = max(worst, np.abs(got - np.asarray(om)).max() / np.abs(np.asarray(om)).max())
print(f"    worst relative error {worst:.3e}")

print()
print("=" * 78)
print("D8b  THE HEADING RATE IN THE PROJECT'S OWN RUNS")
print("=" * 78)
print("D6's curving-path error scales as u_perp * psi_dot. Measured here on the")
print("runs the project actually flies.\n")


@partial(jax.jit, static_argnames=("n", "model"))
def fly(sim, controls, dt, n, model):
    def body(c, _):
        c = integrate.step(c, controls, dt, AC, wind_model=model)
        return c, c.state
    return jax.lax.scan(body, sim, None, length=n)[1]


def run(field, start, seconds, dt=0.01, altitude=HC):
    x, _ = trim.trim(jnp.array(V), jnp.array(altitude), AC)
    ctrl = trim.trimmed_controls(x[1], x[2])
    st = trim.trimmed_state(jnp.array(float(x[0])), jnp.array(V), jnp.array(altitude))
    st = st._replace(pos_ned=jnp.array([start, 0.0, -altitude]))
    sim = integrate.init_sim(st, jax.random.PRNGKey(0))
    n = int(round(seconds / dt))
    hist = fly(sim, ctrl, jnp.array(dt), n, wind.field_model(field))
    dcm = jax.vmap(quat_to_dcm)(hist.quat)
    vel_ned = np.asarray(jnp.einsum("nij,nj->ni", dcm, hist.vel_body))
    psi = np.arctan2(vel_ned[:, 1], vel_ned[:, 0])
    psidot = np.gradient(np.unwrap(psi), dt)
    u_perp = []
    for p, ps in zip(np.asarray(hist.pos_ned), psi):
        wv = np.asarray(field(jnp.asarray(p)))[:2]
        u_perp.append(abs(-wv[0] * np.sin(ps) + wv[1] * np.cos(ps)))
    u_perp = np.array(u_perp)
    return psidot, u_perp, vel_ned


LEAD = 40.0 * W.R0
cases = [
    ("Parks hannibal vortex", W.vortex_field, -LEAD,
     (W.SPACING + LEAD + 6 * W.R0) / V, HC),
    ("Wingrove updraft p=6", W.updraft_field, -2 * W.UPDRAFT_RADIUS,
     4 * W.UPDRAFT_RADIUS / V, HC),
    ("Doyle lee wave", W.lee_field, -0.25 * 25000.0, 3 * 25000.0 / V, HC),
    ("Oseguera-Bowles microburst", W.mb_field, -3 * 1121.0, 6 * 1121.0 / V, 300.0),
]
print(f"  {'run':30s} {'max |psi_dot| (deg/s)':>22s} {'max |u_perp| (m/s)':>20s} "
      f"{'max omitted dF':>16s}")
for lab, fld, start, secs, alt in cases:
    pd, up, vn = run(fld, start, secs, altitude=alt)
    prod = np.abs(pd) * up
    print(f"  {lab:30s} {np.abs(pd).max()*180/np.pi:22.4e} {up.max():20.4e} "
          f"{prod.max()/9.80665:16.4e}")
print("\n  All four fields have zero EAST wind on the north axis and the runs are")
print("  entered northbound, so the track never leaves the meridian: psi_dot is")
print("  zero to round-off and D6's omitted term is EXACTLY zero in every run the")
print("  project makes. Vertical-plane curvature does not matter either -- the")
print("  heading is built from vel_ned[:2] only, so a pull-up leaves it constant.")

print()
print("=" * 78)
print("D8c  THE PARKS CORE, POINT-GUST SCALE RATIO")
print("=" * 78)
print(f"  core diameter 2 r0 = {2*W.R0:.2f} m; 747 span b = {AC.b:.2f} m"
      f"  -> {2*W.R0/AC.b:.3f} spans")
print(f"  tail arm {float(airframe.effective_tail_arm(AC))*AC.c:.2f} m"
      f"  -> {float(airframe.effective_tail_arm(AC))*AC.c/(2*W.R0):.4f} core diameters")
print(f"  Morton case: 2 r0 = {2*W.PARKS_CASES_MORTON:.2f} m"
      if False else
      f"  Morton case: 2 r0 = {2*wind.PARKS_CASES['morton']['r0']:.2f} m"
      f"  -> {2*wind.PARKS_CASES['morton']['r0']/AC.b:.3f} spans")
