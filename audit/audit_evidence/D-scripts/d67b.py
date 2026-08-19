"""D6b/D7b: the curving-path term where it is actually non-zero, and the
one-sided longitudinal station set."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import jax
import jax.numpy as jnp

import flightsim  # noqa: F401
from flightsim import airframe, wind
from flightsim.aircraft import CRUISE, REGISTRY
from flightsim.atmosphere import G0
import wcommon as W

AC = REGISTRY["boeing747"]
V = float(CRUISE["boeing747"]["airspeed"])
HC = W.H_CRUISE
Q0 = jnp.array([1.0, 0.0, 0.0, 0.0])
ST = airframe.stations(AC)

print("=" * 78)
print("D6b  CURVING PATH -- AT HEADINGS WHERE THE OMITTED TERM IS NOT ZERO")
print("=" * 78)
print("Every field in the module has zero east wind on the north axis, so a")
print("northbound turn there has no cross-track wind and the omitted term")
print("vanishes identically. That is a property of the SAMPLE POINT, not of the")
print("approximation. Repeated below at headings that do expose it.\n")


def curved(field, p0, psi0, turn_rate, speed=V, climb=0.0, dt=1e-4):
    def path(t):
        psi = psi0 + turn_rate * t
        rad = speed / turn_rate
        n = p0[0] + rad * (jnp.sin(psi) - jnp.sin(psi0))
        e = p0[1] - rad * (jnp.cos(psi) - jnp.cos(psi0))
        return (jnp.array([n, e, p0[2] + climb * t]),
                jnp.array([speed * jnp.cos(psi), speed * jnp.sin(psi), climb]))

    def U_x(t):
        p, v = path(t)
        return jnp.dot(field(p)[:2], v[:2] / jnp.linalg.norm(v[:2]))

    exact = float((U_x(dt) - U_x(-dt)) / (2 * dt))
    p, v = path(0.0)
    return exact, float(wind.along_track_shear(p, v, field))


rows = [
    ("Parks vortex, core edge, heading EAST", W.vortex_field, [W.R0, 0.0, -HC], 90.0),
    ("Parks vortex, core edge, heading 45 deg", W.vortex_field, [W.R0, 0.0, -HC], 45.0),
    ("Parks vortex, 0.5 r0 above core, heading EAST", W.vortex_field,
     [0.0, 0.0, -HC + 0.5 * W.R0], 90.0),
    ("Parks vortex, 2 r0 out, heading EAST", W.vortex_field, [2 * W.R0, 0.0, -HC], 90.0),
    ("microburst peak outflow, 45 deg off the radius", W.mb_field,
     [1121.0 / np.sqrt(2), 1121.0 / np.sqrt(2), -150.0], 0.0),
    ("microburst peak outflow, heading EAST on the north axis", W.mb_field,
     [1121.0, 0.0, -150.0], 90.0),
    ("microburst r=2000 m, heading EAST", W.mb_field, [2000.0, 0.0, -150.0], 90.0),
]
print(f"    {'case':50s} {'psi_dot':>8s} {'exact':>10s} {'code':>10s} {'omitted':>10s} {'dF':>8s}")
for lab, fld, p0, psi0 in rows:
    for tr_deg in (1.0, 3.0):
        ex, co = curved(fld, jnp.array(p0), np.deg2rad(psi0), np.deg2rad(tr_deg))
        print(f"    {lab:50s} {tr_deg:6.1f}/s {ex:10.5f} {co:10.5f} "
              f"{ex-co:10.5f} {abs(ex-co)/G0:8.4f}")
print("\n    (dF = the F-factor error. FAA 1-km-average alert threshold is 0.1;")
print("     dynamics.f_factor's docstring says that threshold is NOT used at 12 km,")
print("     but thrust_authority for the 747 at 40 kft is the comparison and is small.)")

print("\n    the omitted term in closed form: u_h . d(h_hat)/dt = -u_perp * psi_dot.")
print("    Check that identity directly:")
for lab, fld, p0, psi0 in rows[:4]:
    p = jnp.array(p0)
    tr = np.deg2rad(3.0)
    h = np.array([np.cos(np.deg2rad(psi0)), np.sin(np.deg2rad(psi0))])
    perp = np.array([-h[1], h[0]])
    u_perp = float(np.dot(np.asarray(fld(p))[:2], perp))
    ex, co = curved(fld, p, np.deg2rad(psi0), tr)
    print(f"      {lab:50s} u_perp {u_perp:+8.4f}  -u_perp*psi_dot "
          f"{-u_perp*tr:+9.5f}  measured {ex-co:+9.5f}")

print("\n    all four fields' worst case over a dense sweep of position x heading,")
print("    at a 3 deg/s turn (a standard rate turn):")
for name, (kind, fld) in W.FIELDS.items():
    pos = W.sample_positions(jax.random.PRNGKey(3), 300, kind)
    worst = 0.0
    arg = None
    for p in np.asarray(pos):
        wv = np.asarray(fld(jnp.asarray(p)))[:2]
        # the worst heading is the one putting all the horizontal wind cross-track
        u = np.linalg.norm(wv)
        if u * np.deg2rad(3.0) > worst:
            worst = u * np.deg2rad(3.0)
            arg = p
    print(f"      {name:34s} max |omitted| = {worst:.5f} m/s^2 -> dF = {worst/G0:.5f}"
          f"  at {np.array2string(arg, precision=1) if arg is not None else '-'}")
print("      (the Parks vortex and the lee wave have NO horizontal component at")
print("       all in this module, so the omitted term is identically zero for the")
print("       lee wave and non-zero for the vortex only through its north component.)")

print()
print("=" * 78)
print("D7b  THE LONGITUDINAL STATION SET IS ONE-SIDED")
print("=" * 78)
lon = np.asarray(ST.longitudinal)
print(f"    airframe.stations gives longitudinal = linspace(-{-lon.min():.3f}, 0, 9)")
print(f"    -> all stations are AFT of the CG; their centroid is x = {lon.mean():.3f} m")
print("    A least-squares slope over a one-sided set estimates the derivative at")
print("    the set's CENTROID, so for a curved field it is a BACKWARD secant and")
print("    carries an O(arm/2 * f'') bias, not a symmetric O(arm^2 f''') one.\n")

print("    quadratic test field  w_D(x) = c x^2  (exact tangent at CG = 0):")
for c in [1e-4, 1e-3]:
    quad = lambda p, c=c: jnp.array([0.0, 0.0, c * p[0] ** 2])  # noqa: E731
    p = jnp.array([0.0, 0.0, -HC])
    g = float(wind.gust_rates(p, Q0, quad)[1])
    s = float(wind.sampled_rates(p, Q0, quad, ST)[1])
    print(f"      c = {c:g}:  tangent q {g:+.6e}   secant q {s:+.6e}"
          f"   predicted -2 c x_centroid = {-2*c*lon.mean():+.6e}")

print("\n    Parks core traverse, both sides of the edge, in units of V0/r0:")
char = W.V0 / W.R0
print(f"    {'north/r0':>10s} {'tangent/(V0/r0)':>17s} {'secant/(V0/r0)':>16s} "
      f"{'|diff|/(V0/r0)':>16s}")
for frac in [-3.0, -2.0, -1.5, -1.25, -1.1, -1.0, -0.99, -0.5, 0.0,
             0.5, 0.99, 1.0, 1.1, 1.25, 1.5, 2.0, 3.0]:
    p = jnp.array([frac * W.R0, 0.0, -HC])
    g = float(wind.gust_rates(p, Q0, W.vortex1_field)[1])
    s = float(wind.sampled_rates(p, Q0, W.vortex1_field, ST)[1])
    print(f"    {frac:10.2f} {g/char:17.5f} {s/char:16.5f} {abs(g-s)/char:16.5f}")
print("    test_wind.py's E2 profile samples POSITIVE fracs only. The correction")
print("    at -1.0 r0 is much smaller than at +1.0 r0 because the station set is")
print("    entirely aft, so on the upstream edge the whole airframe is outside the")
print("    core and the secant sees no kink.")

print("\n    peak of |secant - tangent| over the whole traverse (dense):")
fr = np.linspace(-4, 4, 4001)
d = []
for f in fr:
    p = jnp.array([f * W.R0, 0.0, -HC])
    d.append(abs(float(wind.gust_rates(p, Q0, W.vortex1_field)[1])
                 - float(wind.sampled_rates(p, Q0, W.vortex1_field, ST)[1])) / char)
d = np.array(d)
i = int(np.argmax(d))
print(f"      max {d.max():.5f} V0/r0 at north/r0 = {fr[i]:.4f}")
print(f"      (V0/r0 = {char:.6f} 1/s, so that is {d.max()*char:.5f} rad/s of q_gust)")
