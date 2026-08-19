"""Agent C, item 1: dimensional consistency of the formulas, checked by running.

Three kinds of evidence:

  (a) ATMOSPHERE against published ISA table values -- an absolute check that
      catches any misplaced R, g or lapse rate.
  (b) SCALING LAWS.  A dimensionally consistent force law must scale exactly as
      rho^1, V^2, S^1; a moment as rho^1 V^2 S^1 L^1 with L the SPAN for l and n
      and the CHORD for m.  These are measured, not read off the source.
  (c) The remaining formulas (f_factor, thrust, gravity, Coriolis) checked for
      the units they must produce.
"""
import jax.numpy as jnp
import numpy as np

from flightsim import aero, atmosphere, dynamics
from flightsim.aircraft import REGISTRY
from flightsim.atmosphere import G0, RHO0
from flightsim.state import Controls, State, euler_to_quat
from flightsim.units import FT2M

fails = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name:58s} {detail}")
    if not ok:
        fails.append(name)


print("=" * 78)
print("(a) ISA standard atmosphere against published table values")
print("=" * 78)
# ICAO Doc 7488 / ISO 2533 standard atmosphere.
ISA = {          # h[m]: (T[K], p[Pa], rho[kg/m^3], a[m/s])
    0.0:     (288.150, 101325.0, 1.22500, 340.294),
    5000.0:  (255.650,  54019.9, 0.736116, 320.529),
    11000.0: (216.650,  22632.1, 0.363918, 295.070),
    12192.0: (216.650,  18753.9, 0.301559, 295.070),   # 40,000 ft
    20000.0: (216.650,   5474.89, 0.0880349, 295.070),
}
print(f"   {'h[m]':>8s} {'T':>18s} {'p':>22s} {'rho':>22s} {'a':>18s}")
for h, (T, p, rho, a) in ISA.items():
    gT = float(atmosphere.temperature(jnp.array(h)))
    gp = float(atmosphere.pressure(jnp.array(h)))
    gr = float(atmosphere.density(jnp.array(h)))
    ga = float(atmosphere.speed_of_sound(jnp.array(h)))
    print(f"   {h:8.0f} {gT:9.3f}/{T:<8.3f} {gp:11.2f}/{p:<10.2f} "
          f"{gr:10.6f}/{gr and rho:<10.6f} {ga:8.3f}/{a:<8.3f}")
    check(f"   ISA T at {h:.0f} m", abs(gT - T) / T < 1e-5, f"{100*(gT/T-1):+.4f}%")
    check(f"   ISA p at {h:.0f} m", abs(gp - p) / p < 3e-5, f"{100*(gp/p-1):+.4f}%")
    check(f"   ISA rho at {h:.0f} m", abs(gr - rho) / rho < 5e-5,
          f"{100*(gr/rho-1):+.4f}%")
    check(f"   ISA a at {h:.0f} m", abs(ga - a) / a < 3e-5, f"{100*(ga/a-1):+.4f}%")
# Continuity across the tropopause seam
lo = float(atmosphere.pressure(jnp.array(11000.0 - 1e-6)))
hi = float(atmosphere.pressure(jnp.array(11000.0 + 1e-6)))
check("   pressure is continuous at the tropopause", abs(lo - hi) / lo < 1e-9,
      f"{lo:.6f} vs {hi:.6f}")

print()
print("=" * 78)
print("(b) SCALING LAWS of aero_forces_moments")
print("=" * 78)
ac = REGISTRY["boeing747"]
ZERO3 = jnp.zeros(3)
CTL = Controls(jnp.array(0.05), jnp.array(0.03), jnp.array(-0.02), jnp.array(0.0))
A_HI = jnp.array(1.0e6)   # no wave drag: it is Mach-dependent by design
V0, RHO_0 = 200.0, 0.35
vel = jnp.array([V0 * 0.99, V0 * 0.09, V0 * 0.11])


def FM(rho=RHO_0, scale_v=1.0, a=None):
    return aero.aero_forces_moments(vel * scale_v, ZERO3, CTL, a or ac,
                                    jnp.array(rho), A_HI)


F1, M1 = FM()
F2, M2 = FM(rho=2.0 * RHO_0)
r = np.asarray(F2) / np.asarray(F1)
check("force is exactly linear in rho", np.abs(r - 2.0).max() < 1e-12,
      f"ratios={r}")
r = np.asarray(M2) / np.asarray(M1)
check("moment is exactly linear in rho", np.abs(r - 2.0).max() < 1e-12,
      f"ratios={r}")
F3, M3 = FM(scale_v=3.0)
r = np.asarray(F3) / np.asarray(F1)
check("force is exactly quadratic in V (alpha,beta held)",
      np.abs(r - 9.0).max() < 1e-11, f"ratios={r}")
r = np.asarray(M3) / np.asarray(M1)
check("moment is exactly quadratic in V", np.abs(r - 9.0).max() < 1e-11,
      f"ratios={r}")
F4, M4 = FM(a=ac._replace(S=ac.S * 5.0))
r = np.asarray(F4) / np.asarray(F1)
check("force is exactly linear in S", np.abs(r - 5.0).max() < 1e-11, f"ratios={r}")

# THE REFERENCE-LENGTH TEST: which length multiplies which moment channel.
Fb, Mb = FM(a=ac._replace(b=ac.b * 2.0))
rb = np.asarray(Mb) / np.asarray(M1)
Fc, Mc = FM(a=ac._replace(c=ac.c * 2.0))
rc = np.asarray(Mc) / np.asarray(M1)
print(f"   doubling the SPAN  scales (l, m, n) by {rb}")
print(f"   doubling the CHORD scales (l, m, n) by {rc}")
check("rolling moment l is referenced to the SPAN only",
      abs(rb[0] - 2.0) < 1e-12 and abs(rc[0] - 1.0) < 1e-12)
check("pitching moment m is referenced to the CHORD only",
      abs(rc[1] - 2.0) < 1e-12 and abs(rb[1] - 1.0) < 1e-12)
check("yawing moment n is referenced to the SPAN only",
      abs(rb[2] - 2.0) < 1e-12 and abs(rc[2] - 1.0) < 1e-12)

print()
print("=" * 78)
print("(c) the remaining formulas")
print("=" * 78)
# thrust: N, and the density lapse
T_sl = float(aero.thrust_force(Controls(jnp.array(0.0), jnp.array(0.0),
                                        jnp.array(0.0), jnp.array(1.0)),
                               ac, jnp.array(RHO0))[0])
print(f"   full-throttle sea-level thrust = {T_sl:.1f} N = "
      f"{T_sl / 4.4482216152605 / 1000:.2f} klbf (4 x 43.5 klbf = 174 klbf)")
check("thrust_force returns newtons and equals max_thrust at rho0",
      abs(T_sl - float(ac.max_thrust)) < 1e-6, f"{T_sl:.2f} vs {float(ac.max_thrust):.2f}")
lapsed = float(aero.thrust_force(Controls(jnp.array(0.0), jnp.array(0.0),
                                          jnp.array(0.0), jnp.array(1.0)),
                                 ac, jnp.array(0.5 * RHO0))[0])
check("thrust lapses as (rho/rho0)^0.8",
      abs(lapsed / T_sl - 0.5 ** 0.8) < 1e-9, f"ratio={lapsed / T_sl:.8f}")
check("thrust acts along body +x only",
      float(aero.thrust_force(Controls(jnp.array(0.0), jnp.array(0.0),
                                       jnp.array(0.0), jnp.array(1.0)),
                              ac, jnp.array(RHO0))[1]) == 0.0)

# gravity: in free fall with no air the body-frame acceleration must be g in
# the NED-down direction, whatever the attitude.
vac = ac._replace(CL0=jnp.array(0.0), CLa=jnp.array(0.0), CLq=jnp.array(0.0),
                  CLde=jnp.array(0.0), CD0=jnp.array(0.0), e=jnp.array(1e12),
                  CYb=jnp.array(0.0), CYdr=jnp.array(0.0), CYp=jnp.array(0.0),
                  CYr=jnp.array(0.0), max_thrust=jnp.array(0.0))
qn = euler_to_quat(jnp.array(0.4), jnp.array(-0.3), jnp.array(1.1))
st = State(pos_ned=jnp.array([0.0, 0.0, -5000.0]),
           vel_body=jnp.array([100.0, 0.0, 0.0]), quat=qn, omega=jnp.zeros(3))
d = dynamics.derivatives(st, Controls(*[jnp.array(0.0)] * 4), vac,
                         jnp.zeros(3), jnp.zeros(3))
from flightsim.state import quat_to_dcm
a_ned = np.asarray(quat_to_dcm(qn)) @ np.asarray(d.vel_body)
print(f"   drag-free, thrust-free, omega=0: accel in NED = {a_ned} m/s^2")
check("gravity is exactly (0,0,G0) in NED at a non-trivial attitude",
      np.abs(a_ned - np.array([0.0, 0.0, G0])).max() < 1e-9,
      f"err={np.abs(a_ned - np.array([0.0, 0.0, G0])).max():.2e}")

# f_factor: dimensionless, and the paper's stated signs
f_desc = float(dynamics.f_factor(jnp.array(0.0), jnp.array(-5.0), jnp.array(200.0)))
f_tail = float(dynamics.f_factor(jnp.array(1.0), jnp.array(0.0), jnp.array(200.0)))
print(f"   F for a descending air mass (w=-5 up)      = {f_desc:+.6f} (must be > 0)")
print(f"   F for a growing tailwind (Uxdot=+1 m/s^2)  = {f_tail:+.6f} (must be > 0)")
check("f_factor is positive for a descending air mass", f_desc > 0)
check("f_factor is positive for an accelerating tailwind", f_tail > 0)
check("f_factor is dimensionless (shear/g and w/V both non-dimensional)",
      abs(f_tail - 1.0 / G0) < 1e-12)

# Coriolis term: a pure body rate with zero force must not change the SPEED.
st2 = State(pos_ned=jnp.array([0.0, 0.0, -5000.0]),
            vel_body=jnp.array([100.0, 7.0, -3.0]), quat=qn,
            omega=jnp.array([0.2, -0.1, 0.15]))
vac2 = vac._replace(mass=jnp.array(1e12))  # make aero and gravity negligible
d2 = dynamics.derivatives(st2, Controls(*[jnp.array(0.0)] * 4), vac2,
                          jnp.zeros(3), jnp.zeros(3))
vb = np.asarray(st2.vel_body)
rate_of_speed = float(np.dot(vb, np.asarray(d2.vel_body) - np.asarray(
    quat_to_dcm(qn)).T @ np.array([0.0, 0.0, G0]))) / np.linalg.norm(vb)
check("-omega x v does no work (transport term is orthogonal to v)",
      abs(rate_of_speed) < 1e-9, f"d|v|/dt from Coriolis = {rate_of_speed:.2e}")

print()
print("FAILURES:", fails if fails else "none")
