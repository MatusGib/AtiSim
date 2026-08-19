"""D4b: follow-ups -- do the guards actually earn their keep, and what do they cost?"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import jax
import jax.numpy as jnp

import flightsim  # noqa: F401
from flightsim import wind
import wcommon as W

HC = W.H_CRUISE
Q0 = jnp.array([1.0, 0.0, 0.0, 0.0])

print("=" * 78)
print("D4b-1  DOES THE UPDRAFT'S 1e-12 CLAMP ACTUALLY PREVENT A NaN?")
print("=" * 78)
print("Unclamped copy of updraft_wind (scratch only, flightsim untouched):")


def updraft_unclamped(p, col):
    offset = jnp.hypot(p[0] - col.north, p[1] - col.east)
    scaled = offset / col.radius              # <-- the clamp removed
    return jnp.array([0.0, 0.0, -col.w0 * jnp.exp(-(scaled ** col.sharpness))])


for sh in [0.5, 1.0, 2.0, 6.0]:
    col = wind.UpdraftColumn(north=jnp.array(0.0), east=jnp.array(0.0),
                             w0=jnp.array(24.384), radius=jnp.array(2360.0),
                             sharpness=jnp.array(sh))
    p = jnp.array([0.0, 0.0, -HC])
    ju = np.asarray(jax.jacfwd(lambda q: updraft_unclamped(q, col))(p))
    jc = np.asarray(jax.jacfwd(lambda q: wind.updraft_wind(q, col))(p))
    print(f"  sharpness {sh:4.1f}  unclamped jacfwd row2 = {ju[2]}"
          f"   clamped = {jc[2]}")

print()
print("=" * 78)
print("D4b-2  WHAT THE CLAMP COSTS, AND THE GRADIENT JUST OUTSIDE IT")
print("=" * 78)
print(f"{'sharp':>6s} {'clamp cost on w(0)':>20s} {'r where s=1e-12 (m)':>21s} "
      f"{'|p_gust| at r=1 m':>18s} {'|p_gust| at r=10 m':>19s} {'max |p_gust| (1/s)':>19s}")
for sh in [0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 6.0]:
    col = wind.UpdraftColumn(north=jnp.array(0.0), east=jnp.array(0.0),
                             w0=jnp.array(24.384), radius=jnp.array(2360.0),
                             sharpness=jnp.array(sh))
    fu = lambda q, c=col: wind.updraft_wind(q, c)  # noqa: E731
    w_axis = -float(fu(jnp.array([0.0, 0.0, -HC]))[2])
    cost = 1 - w_axis / 24.384
    def prate(r):
        return abs(float(wind.gust_rates(jnp.array([0.0, r, -HC]), Q0, fu)[0]))
    rs = np.logspace(-9, 4.2, 4000)
    mx = max(prate(r) for r in rs[::40])
    print(f"{sh:6.1f} {cost:20.3e} {1e-12*2360.0:21.3e} {prate(1.0):18.4e} "
          f"{prate(10.0):19.4e} {mx:19.4e}")
print("  (p_gust = d w_g/d y at a lateral offset; the 747 half-span is 29.8 m)")

print()
print("=" * 78)
print("D4b-3  MICROBURST 'scaled > 1e-8' GUARD, CONTINUITY DONE PROPERLY")
print("=" * 78)
print("The 'exact' column below uses expm1, not (1-exp(-s)), so the comparison is")
print("not swamped by numpy cancellation the way a naive check is.")
print(f"{'scaled':>12s} {'series 1-s/2':>20s} {'-expm1(-s)/s':>20s} {'|diff|':>12s}")
for s in [1e-6, 1e-7, 2e-8, 1.0000001e-8, 1e-8, 0.9999999e-8, 1e-9, 1e-12]:
    ser = 1 - 0.5 * s
    ex = -np.expm1(-s) / s
    print(f"{s:12.4e} {ser:20.16f} {ex:20.16f} {abs(ser-ex):12.3e}")
print("  -> the two branches agree to <1e-16 at the switch, so the guard is C0"
      " to machine precision.")
print("  series error vs exact at the switch point s=1e-8: "
      f"{abs((1-0.5e-8) - (-np.expm1(-1e-8)/1e-8)):.3e}")

print()
print("=" * 78)
print("D4b-4  DIFFERENTIATING W.R.T. THE FIELD PARAMETERS, NOT POSITION")
print("=" * 78)
print("gust_rates only differentiates position, but a sensitivity study would")
print("differentiate r0/v0/w0. Checked for NaN because those sit inside the guards.")


def vw(r0, v0, p):
    arr = wind.VortexArray(north=jnp.array([0.0]), down=jnp.array([-HC]),
                           r0=r0, v0=v0)
    return wind.vortex_wind(p, arr)


for label, p in [("centre r=0", [0.0, 0.0, -HC]),
                 ("r = r0 exactly", [W.R0, 0.0, -HC]),
                 ("inside r=0.5 r0", [0.5 * W.R0, 0.0, -HC]),
                 ("outside r=2 r0", [2 * W.R0, 0.0, -HC])]:
    p = jnp.array(p)
    g = np.asarray(jax.grad(lambda r: jnp.sum(vw(r, jnp.array(W.V0), p) ** 2))(jnp.array(W.R0)))
    gv = np.asarray(jax.grad(lambda v: jnp.sum(vw(jnp.array(W.R0), v, p) ** 2))(jnp.array(W.V0)))
    print(f"  {label:18s} d/dr0 = {g:+.6e}  d/dv0 = {gv:+.6e}"
          f"  {'<<< NaN' if np.isnan(g) or np.isnan(gv) else ''}")


def uw(w0, rad, sh, p):
    col = wind.UpdraftColumn(north=jnp.array(0.0), east=jnp.array(0.0),
                             w0=w0, radius=rad, sharpness=sh)
    return wind.updraft_wind(p, col)


for sh in [0.5, 2.0, 6.0]:
    for label, p in [("axis", [0.0, 0.0, -HC]), ("r=1000 m", [1000.0, 0.0, -HC])]:
        p = jnp.array(p)
        g = float(jax.grad(lambda s: jnp.sum(uw(jnp.array(24.384), jnp.array(2360.0),
                                                s, p) ** 2))(jnp.array(sh)))
        print(f"  updraft sharpness={sh:4.1f} {label:10s} d/d(sharpness) = {g:+.6e}"
              f"  {'<<< NaN' if np.isnan(g) else ''}")

print()
print("=" * 78)
print("D4b-5  VORTEX FAR FIELD AT THE RUN'S OWN LEAD-IN")
print("=" * 78)
for k in [4, 8, 12, 20, 40, 80]:
    p = jnp.array([-k * W.R0, 0.0, -HC])
    wv = np.asarray(W.vortex_field(p))
    gr = np.asarray(wind.gust_rates(p, Q0, W.vortex_field))
    print(f"  lead-in {k:3d} r0 = {k*W.R0:8.1f} m:  |w| = {np.linalg.norm(wv):.6f} m/s"
          f"   w_D = {wv[2]:+.6f}   |omega_gust| = {np.linalg.norm(gr):.3e} rad/s")
print("  (a 2-core array decays like 1/r, so the 'still air' the run is trimmed")
print("   for is never actually still -- this is the lead-in argument in")
print("   scripts/vortex.py made numeric.)")

print()
print("=" * 78)
print("D4b-6  MICROBURST BELOW GROUND: WHAT AN AIRCRAFT WOULD FLY INTO")
print("=" * 78)
for z in [50.0, 10.0, 1.0, 0.1, 0.0, -0.1, -1.0, -10.0]:
    p = jnp.array([1121.0, 0.0, -z])
    wv = np.asarray(W.mb_field(p))
    gr = np.asarray(wind.gust_rates(p, Q0, W.mb_field))
    print(f"  z = {z:+7.2f} m   w_N = {wv[0]:+9.5f}  w_D = {wv[2]:+9.5f}   "
          f"|omega_gust| = {np.linalg.norm(gr):.4e} rad/s")
print("  -> the field is C0 across z = 0 (both components vanish there) but the")
print("     GRADIENT steps to exactly zero, so the gust rate is discontinuous.")
