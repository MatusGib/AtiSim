"""A10: reproduce the project's own published verification numbers, exactly."""
import itertools
import numpy as np
import jax, jax.numpy as jnp
import flightsim
from flightsim import integrate, trim, verification, validation
from flightsim.aircraft import CRUISE, REGISTRY
from flightsim.state import Controls, State, euler_to_quat, quat_to_dcm
from flightsim.tests.conftest import make_test_aircraft

print("=" * 78)
print("### repro of test_conservation.test_free_rigid_body_... (claims 5.7e-14 / 9.3e-13)")
print("=" * 78)
Z = ["CD0", "CL0", "CLa", "CLq", "CLde", "Cm0", "Cma", "Cmq", "Cmde", "CYb", "CYp",
     "CYr", "CYdr", "Clb", "Clp", "Clr", "Clda", "Cldr", "Cnb", "Cnp", "Cnr",
     "Cnda", "Cndr", "max_thrust"]
ac = make_test_aircraft(Ixz=200.0)._replace(**{f: jnp.array(0.0) for f in Z})
omega0 = jnp.array([0.5, 0.3, -0.2])
s0 = State(pos_ned=jnp.array([0.0, 0.0, -1000.0]), vel_body=jnp.zeros(3),
           quat=euler_to_quat(jnp.array(0.3), jnp.array(-0.2), jnp.array(0.5)),
           omega=omega0)
ctl = Controls(elevator=jnp.array(0.0), aileron=jnp.array(0.0),
               rudder=jnp.array(0.0), throttle=jnp.array(0.0))
for dt, seconds in ((0.01, 600.0), (0.02, 600.0), (0.05, 600.0), (0.01, 6000.0)):
    n = int(round(seconds / dt))
    _, hist = integrate.rollout(integrate.init_sim(s0, jax.random.PRNGKey(0)),
                                ctl, jnp.array(dt), ac, n)
    om = np.asarray(hist.omega); qh = np.asarray(hist.quat)
    I = np.asarray(ac.inertia)
    Lb = om @ I.T
    dcms = np.asarray(jax.vmap(quat_to_dcm)(jnp.asarray(qh)))
    Ln = np.einsum("nij,nj->ni", dcms, Lb)
    Lm = np.linalg.norm(Ln, axis=1)
    mag = np.abs(Lm - Lm[0]).max() / Lm[0]
    d0 = Ln[0] / Lm[0]
    ca = np.clip((Ln / Lm[:, None] * d0).sum(axis=1), -1.0, 1.0)
    ddeg = np.degrees(np.arccos(ca)).max()
    KE = 0.5 * np.einsum("ni,ij,nj->n", om, I, om)
    ke = np.abs(KE - KE[0]).max() / KE[0]
    print(f"  dt={dt:<6g} T={seconds:<7g} n={n:<7d}  |L| drift={mag:.4e} (tol 1e-11)  "
          f"dir drift={ddeg:.4e} deg (tol 3e-5)  KE drift={ke:.4e} (tol 2e-11)")

print()
print("=" * 78)
print("### repro of test_verification order numbers")
print("=" * 78)
AC = REGISTRY["boeing747"]
V, H = CRUISE["boeing747"]["airspeed"], CRUISE["boeing747"]["altitude"]
dts = np.array([1.0/4, 1.0/8, 1.0/16, 1.0/32])
e, s = verification.oscillator_refinement(np.array([0.2, 0.1, 0.05, 0.025]))
print(f"  oscillator                 order = {s:.6f}   (test asserts 4.0 +- 0.05)")
e, s = verification.fixed_control_refinement(AC, V, H, dts, dt_ref=1.0/1024.0)
print(f"  6-DOF still air            order = {s:.6f}   (PROJECT.md records 3.98913)")
print(f"    errors = {e}")

from flightsim import wind
wave = wind.LeeWave(w0=jnp.array(25.0), wavelength=jnp.array(1200.0), north=jnp.array(0.0))
model = wind.field_model(lambda p: wind.lee_wave_wind(p, wave))
e, s = verification.fixed_control_refinement(AC, V, H, dts, dt_ref=1.0/1024.0,
                                             wind_model=model)
print(f"  6-DOF smooth field         order = {s:.6f}   (test docstring records 1.0537)")
print(f"    errors = {e}")

case = wind.PARKS_CASES["hannibal"]; r0 = case["r0"]
arr = wind.VortexArray(north=jnp.array([0.0]), down=jnp.array([-H]),
                       r0=jnp.array(r0), v0=jnp.array(case["v0"]))
vm = wind.field_model(lambda p: wind.vortex_wind(p, arr))
e, s = verification.fixed_control_refinement(AC, V, H,
                                             np.array([1/16, 1/32, 1/64, 1/128]),
                                             dt_ref=1.0/2048.0, wind_model=vm,
                                             start_north=-2.0*r0, d_elevator=0.0)
print(f"  6-DOF Rankine core   'order' = {s:.6f}  (MEANINGLESS: sequence non-monotone)")
print(f"    errors = {e}")
print(f"    monotone? {all(b < a for a, b in zip(e, e[1:]))}")

print()
print("### the WIDE sweep the docstring of test_the_six_dof_rollout_is_fourth_order quotes")
wide = np.array([1/4, 1/8, 1/16, 1/32, 1/64, 1/128, 1/256])
ew, sw = verification.fixed_control_refinement(AC, V, H, wide, dt_ref=1.0/1024.0)
print(f"  errors = {ew}")
pair = [np.log(ew[i]/ew[i+1])/np.log(wide[i]/wide[i+1]) for i in range(len(wide)-1)]
print("  pairwise orders (docstring: 3.973, 3.993, 4.008, 4.167, 3.420, -0.685):")
for (a, b), p in zip(zip(wide, wide[1:]), pair):
    print(f"    1/{1/a:.0f} -> 1/{1/b:.0f}   {p:+.4f}")
print(f"  fitted slope over the WIDE window = {verification.fitted_order(wide, ew):.4f} "
      "(docstring says such a window reads 3.82)")

print()
print("=" * 78)
print("### free-fall / swinging-wind experiment (test claims < 1e-9 m)")
print("=" * 78)
r = verification.free_fall_through_a_swinging_wind(
    verification.without_aerodynamics(AC), dt=0.02, n=300)
print(f"  {r}")

print()
print("=" * 78)
print("### the pre-refactor hash (test_extracting_rk4_step_did_not_move_a_single_bit)")
print("=" * 78)
import hashlib
x, _ = trim.trim(jnp.array(V), jnp.array(H), AC)
st = trim.trimmed_state(x[0], jnp.array(V), jnp.array(H))
ct = trim.trimmed_controls(x[1] + 0.02, x[2])
_, tr = integrate.rollout(integrate.init_sim(st, jax.random.PRNGKey(0)),
                          ct, jnp.array(0.02), AC, 500)
got = hashlib.sha256(np.asarray(tr.vel_body).tobytes()).hexdigest()
want = "bbc0323e77183d73bd03817a98a530d0d563b56b4962520705aaee589f276aa4"
print(f"  got  {got}")
print(f"  want {want}")
print(f"  MATCH: {got == want}")
