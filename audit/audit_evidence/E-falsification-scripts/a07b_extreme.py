"""ATTACK 7 (rest): extreme but physically reachable inputs.

  - alpha driven past stall: quantify the lift the model reports at 30 deg
  - 747 at Cherokee speed and Cherokee at 747 speed
  - max throttle in a dive
  - full deflection reversals at the sample rate (dt-resolution stress)
"""
from common import *  # noqa

from flightsim.aero import coefficients, aero_forces_moments
from flightsim.aircraft import CESSNA172_TABLES
from flightsim.atmosphere import density, speed_of_sound

Z3 = jnp.zeros(3)

print("=== 7a. NO STALL: model CL vs the source's own measured table ===")
ac = REGISTRY["cessna172"]
tab_a = np.array(CESSNA172_TABLES["alpha_deg"])
tab_CL = np.array(CESSNA172_TABLES["CL"])
tab_CD = np.array(CESSNA172_TABLES["CD"])
V, h = 60.0, 1500.0
rho, a = density(jnp.array(h)), speed_of_sound(jnp.array(h))
print(f"  cessna172, V={V} m/s, h={h} m, zero rates, zero elevator")
print(f"  {'alpha':>7s} {'model CL':>10s} {'table CL':>10s} {'ratio':>8s} "
      f"{'model CD':>10s} {'table CD':>10s} {'model L (N)':>12s} {'n_z equiv':>10s}")
W = float(ac.mass) * G0
for ad in [0.0, 5.0, 10.0, 15.0, 19.5, 25.0, 30.0, 45.0, 60.0, 90.0]:
    al = np.radians(ad)
    vel = V * jnp.array([np.cos(al), 0.0, np.sin(al)])
    CL, CD, CY, Cl, Cm, Cn = coefficients(vel, Z3, ctrl(), ac, a)
    qbar = 0.5 * float(rho) * V * V
    L = qbar * float(ac.S) * float(CL)
    tCL = np.interp(ad, tab_a, tab_CL) if ad <= tab_a[-1] else np.nan
    tCD = np.interp(ad, tab_a, tab_CD) if ad <= tab_a[-1] else np.nan
    rat = float(CL) / tCL if np.isfinite(tCL) else np.nan
    print(f"  {ad:7.1f} {float(CL):10.4f} {tCL:10.4f} {rat:8.3f} "
          f"{float(CD):10.4f} {tCD:10.4f} {L:12.5g} {L/W:10.3f}")
print(f"  source CLmax = {tab_CL.max():.3f} at {tab_a[np.argmax(tab_CL)]:.1f} deg;")
CL30 = float(coefficients(V*jnp.array([np.cos(np.radians(30)),0.,np.sin(np.radians(30))]),
                          Z3, ctrl(), ac, a)[0])
print(f"  model at 30 deg reports CL = {CL30:.3f} = {CL30/tab_CL.max():.2f}x CLmax")
print(f"  a real wing past CLmax LOSES lift; the model's CL keeps rising linearly.")

print("\n  all four aircraft at alpha = 30 deg:")
for name in NAMES:
    ac2 = REGISTRY[name]
    Vc, hc = CRUISE[name]["airspeed"], CRUISE[name]["altitude"]
    r2, a2 = density(jnp.array(hc)), speed_of_sound(jnp.array(hc))
    al = np.radians(30.0)
    vel = Vc * jnp.array([np.cos(al), 0.0, np.sin(al)])
    CL, CD, *_ = coefficients(vel, Z3, ctrl(), ac2, a2)
    qbar = 0.5 * float(r2) * Vc * Vc
    L = qbar * float(ac2.S) * float(CL)
    print(f"    {name:20s} CL={float(CL):7.4f} CD={float(CD):7.4f} "
          f"L/W = {L/(float(ac2.mass)*G0):7.3f} g  (typical CLmax ~1.4-2.1)")

print("\n=== 7b. cross-envelope: each aircraft at every other's cruise speed ===")
print(f"  {'aircraft':20s} {'V (m/s)':>9s} {'Mach':>7s} {'alpha_trim':>11s} "
      f"{'thr_trim':>9s} {'CL':>8s} {'resid':>9s} {'is_phys':>8s}")
from flightsim.trim import is_physical
for name in NAMES:
    ac2 = REGISTRY[name]
    hc = CRUISE[name]["altitude"]
    a2 = float(speed_of_sound(jnp.array(hc)))
    for other in NAMES:
        Vo = CRUISE[other]["airspeed"]
        x, r = trim.trim(jnp.array(Vo), jnp.array(hc), ac2)
        xn = np.asarray(x)
        vel = Vo * jnp.array([np.cos(xn[0]), 0.0, np.sin(xn[0])])
        CL = float(coefficients(vel, Z3, ctrl(de=xn[1]), ac2,
                                jnp.array(a2))[0])
        tag = "" if other == name else f"  <- {other}'s speed"
        print(f"  {name:20s} {Vo:9.1f} {Vo/a2:7.3f} "
              f"{np.degrees(xn[0]):11.2f} {xn[2]:9.3f} {CL:8.3f} "
              f"{float(np.linalg.norm(np.asarray(r))):9.1e} "
              f"{str(is_physical(x)):>8s}{tag}")

print("\n=== 7c. max throttle in a vertical dive: terminal state ===")
for name in NAMES:
    ac2 = REGISTRY[name]
    Vc, hc = CRUISE[name]["airspeed"], CRUISE[name]["altitude"]
    st = State(pos_ned=jnp.array([0.0, 0.0, -max(hc, 12000.0)]),
               vel_body=jnp.array([Vc, 0.0, 0.0]),
               quat=euler_to_quat(jnp.array(0.0), jnp.array(-np.pi / 2),
                                  jnp.array(0.0)),
               omega=Z3)
    _, hist = run(st, ctrl(thr=1.0), ac2, 0.005, 40000)
    v = np.asarray(hist.vel_body); p = np.asarray(hist.pos_ned)
    V = np.linalg.norm(v, axis=1)
    a2 = np.asarray(speed_of_sound(jnp.asarray(-p[:, 2])))
    M = V / a2
    print(f"  {name:20s} Vmax={V.max():8.1f} m/s  Mmax={M.max():6.3f}  "
          f"h final={-p[-1,2]:10.1f} m  finite={bool(np.all(np.isfinite(v)))}")
    print(f"    {'':20s} (Mach > 1 means the wave-drag law is far outside "
          f"Lock's 4th-power range)" if M.max() > 1.0 else "")

print("\n=== 7d. full deflection REVERSALS at the sample rate, dt sensitivity ===")
from functools import partial
from flightsim.integrate import rk4_step
from flightsim.state import quat_normalize
from flightsim.dynamics import derivatives


@partial(jax.jit, static_argnames=("n",))
def alternate(st0, ac, dt, n, lim, thr):
    def body(st, i):
        de = lim * jnp.where(i % 2 == 0, 1.0, -1.0)
        c = Controls(elevator=de, aileron=jnp.array(0.0), rudder=jnp.array(0.0),
                     throttle=thr)
        st = rk4_step(lambda s: derivatives(s, c, ac, Z3, Z3), st, dt)
        return st._replace(quat=quat_normalize(st.quat)), st
    return jax.lax.scan(body, st0, jnp.arange(n))


for name in NAMES:
    ac2, st, x, r = trim_state(name)
    lim = jnp.array(float(ac2.elevator_limit))
    print(f"  {name}:")
    for dt in (0.04, 0.02, 0.01, 0.005, 0.002):
        n = int(30.0 / dt)
        _, hist = alternate(st, ac2, jnp.array(dt), n, lim,
                            jnp.array(float(x[2])))
        E, *_ = energy(hist, ac2)
        v = np.linalg.norm(np.asarray(hist.vel_body), axis=1)
        q = np.asarray(hist.omega)[:, 1]
        fin = bool(np.all(np.isfinite(E)))
        print(f"    dt={dt:6.3f} ({1/dt:5.0f} Hz reversal) finite={fin} "
              f"Vfinal={v[-1]:8.2f} max|q|={np.abs(q).max():7.4f} "
              f"E_final/E0={E[-1]/E[0]:9.6f}  max(E-E0)={float((E-E[0]).max()):+.4g}")
