"""ATTACK 6 (focused): the Parks Rankine vortex gradient AT the core edge.

Rankine core: inside r<r0 solid body, outside potential. The VALUE matches at
r = r0 but the GRADIENT does not. If so, `field_model` -- which differentiates
the field to build omega_gust -- hands the aircraft a step change in rotational
gust as it crosses the core boundary, and the analytic jump is 2*v0/r0.
"""
from common import *  # noqa

from flightsim import wind as W
from flightsim.state import euler_to_quat

Q0 = euler_to_quat(jnp.array(0.0), jnp.array(0.0), jnp.array(0.0))

for cname in ("hannibal", "morton"):
    case = W.PARKS_CASES[cname]
    r0, v0 = float(case["r0"]), float(case["v0"])
    arr = W.VortexArray(north=jnp.array([0.0]), down=jnp.array([0.0]),
                        r0=jnp.array(r0), v0=jnp.array(v0))
    f = lambda p: W.vortex_wind(p, arr)
    print(f"\n=== {cname}: r0 = {r0:.3f} m, v0 = {v0:.3f} m/s, "
          f"2*v0/r0 = {2*v0/r0:.6f} 1/s ===")

    print("  RADIAL cut (aircraft directly above the core, offset = r):")
    print(f"  {'r/r0':>11s} {'W_north':>11s} {'dW_n/dz':>12s} {'dW_n/dx':>12s} "
          f"{'dW_d/dx':>12s}")
    for frac in [0.99, 0.999, 0.99999, 1.0, 1.00001, 1.001, 1.01]:
        p = jnp.array([0.0, 0.0, -r0 * frac])
        v = np.asarray(f(p)); J = np.asarray(jax.jacfwd(f)(p))
        print(f"  {frac:11.5f} {v[0]:11.6f} {J[0,2]:12.7f} {J[0,0]:12.7f} "
              f"{J[2,0]:12.7f}")

    print("  TANGENTIAL cut (aircraft level with the core, offset = l):")
    print(f"  {'r/r0':>11s} {'W_down':>11s} {'dW_d/dx':>12s} {'dW_n/dz':>12s}")
    for frac in [0.99, 0.999, 0.99999, 1.0, 1.00001, 1.001, 1.01]:
        p = jnp.array([r0 * frac, 0.0, 0.0])
        v = np.asarray(f(p)); J = np.asarray(jax.jacfwd(f)(p))
        print(f"  {frac:11.5f} {v[2]:11.6f} {J[2,0]:12.7f} {J[0,2]:12.7f}")

    # the jump in each Jacobian entry, measured either side of r0
    eps = 1e-7
    for lbl, mk in [("radial", lambda s: jnp.array([0.0, 0.0, -r0 * s])),
                    ("tangential", lambda s: jnp.array([r0 * s, 0.0, 0.0])),
                    ("45 deg", lambda s: jnp.array([r0*s/np.sqrt(2), 0.0,
                                                    -r0*s/np.sqrt(2)]))]:
        Ji = np.asarray(jax.jacfwd(f)(mk(1.0 - eps)))
        Jo = np.asarray(jax.jacfwd(f)(mk(1.0 + eps)))
        vi = np.asarray(f(mk(1.0 - eps))); vo = np.asarray(f(mk(1.0 + eps)))
        print(f"  {lbl:12s} |W jump| = {np.abs(vo-vi).max():.3e} m/s   "
              f"|dW/dx jump| = {np.abs(Jo-Ji).max():.7f} 1/s   "
              f"(= {np.abs(Jo-Ji).max()/(2*v0/r0):.4f} x 2*v0/r0)")
        og_i = np.asarray(W.gust_rates(mk(1.0 - eps), Q0, f))
        og_o = np.asarray(W.gust_rates(mk(1.0 + eps), Q0, f))
        print(f"  {'':12s} omega_gust inside  = [{og_i[0]:+.7f} {og_i[1]:+.7f} "
              f"{og_i[2]:+.7f}] rad/s")
        print(f"  {'':12s} omega_gust outside = [{og_o[0]:+.7f} {og_o[1]:+.7f} "
              f"{og_o[2]:+.7f}] rad/s")
        print(f"  {'':12s} JUMP in omega_gust = "
              f"{np.abs(og_o-og_i).max():.7f} rad/s "
              f"({np.degrees(np.abs(og_o-og_i).max()):.4f} deg/s)")

print("\n=== what that jump does to the aircraft: p_gust step across the edge ===")
case = W.PARKS_CASES["hannibal"]
r0, v0 = float(case["r0"]), float(case["v0"])
arr = W.VortexArray(north=jnp.array([0.0]), down=jnp.array([0.0]),
                    r0=jnp.array(r0), v0=jnp.array(v0))
f = lambda p: W.vortex_wind(p, arr)
print("  aircraft flying north at the core altitude, sweeping through l = r0:")
print(f"  {'l/r0':>9s} {'p_gust':>12s} {'q_gust':>12s} {'r_gust':>12s}  (rad/s)")
for frac in [0.98, 0.999, 0.99999, 1.0, 1.00001, 1.001, 1.02]:
    p = jnp.array([r0 * frac, 0.0, 0.0])
    og = np.asarray(W.gust_rates(p, Q0, f))
    print(f"  {frac:9.5f} {og[0]:12.7f} {og[1]:12.7f} {og[2]:12.7f}")

print("\n=== resulting step in the aerodynamic moment (a 747 crossing the edge) ===")
from flightsim.aero import aero_forces_moments
from flightsim.atmosphere import density, speed_of_sound
ac = REGISTRY["boeing747"]
Vc, hc = CRUISE["boeing747"]["airspeed"], CRUISE["boeing747"]["altitude"]
rho, a = density(jnp.array(hc)), speed_of_sound(jnp.array(hc))
x, _ = trim.trim(jnp.array(Vc), jnp.array(hc), ac)
vel = Vc * jnp.array([np.cos(float(x[0])), 0.0, np.sin(float(x[0]))])
for frac in [0.99999, 1.00001]:
    p = jnp.array([r0 * frac, 0.0, 0.0])
    og = W.gust_rates(p, Q0, f)
    F, M = aero_forces_moments(vel, -og, ctrl(de=float(x[1])), ac, rho, a)
    print(f"  l/r0={frac:9.5f}  omega_gust={np.asarray(og)}  "
          f"M=[{float(M[0]):+.5g} {float(M[1]):+.5g} {float(M[2]):+.5g}] N.m")
Ma = []
for frac in [1 - 1e-7, 1 + 1e-7]:
    p = jnp.array([r0 * frac, 0.0, 0.0])
    og = W.gust_rates(p, Q0, f)
    Ma.append(np.asarray(aero_forces_moments(vel, -og, ctrl(de=float(x[1])),
                                             ac, rho, a)[1]))
dM = np.abs(Ma[1] - Ma[0])
Iyy = float(ac.inertia[1, 1]); Ixx = float(ac.inertia[0, 0])
print(f"  STEP in moment across the edge: [{Ma[1][0]-Ma[0][0]:+.5g} "
      f"{Ma[1][1]-Ma[0][1]:+.5g} {Ma[1][2]-Ma[0][2]:+.5g}] N.m")
print(f"  -> instantaneous step in angular acceleration: "
      f"pdot {(Ma[1][0]-Ma[0][0])/Ixx:+.6f} rad/s^2, "
      f"qdot {(Ma[1][1]-Ma[0][1])/Iyy:+.6f} rad/s^2")
print(f"  the aircraft crosses this edge in "
      f"{2*r0/Vc*1000:.1f} ms of core traverse at {Vc:.1f} m/s; "
      f"at dt=0.01 s the step lands inside a single RK4 stage set")
