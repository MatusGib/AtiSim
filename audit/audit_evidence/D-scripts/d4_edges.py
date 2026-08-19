"""D4: edges, singularities, guard branches and the jnp.where NaN trap."""

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


def probe(label, field, p, extra=()):
    """Value, jacfwd, jacrev, hessian, gust_rates, along_track_shear at one point."""
    p = jnp.asarray(p, dtype=float)
    out = {}
    def safe(f):
        try:
            v = np.asarray(f())
            return v, bool(np.isnan(v).any()), bool(np.isinf(v).any())
        except Exception as e:  # noqa: BLE001
            return f"EXC {type(e).__name__}: {e}", False, False
    out["w"] = safe(lambda: field(p))
    out["jacfwd"] = safe(lambda: jax.jacfwd(field)(p))
    out["jacrev"] = safe(lambda: jax.jacrev(field)(p))
    out["hess"] = safe(lambda: jax.jacfwd(jax.jacfwd(field))(p))
    out["gust_rates"] = safe(lambda: wind.gust_rates(p, Q0, field))
    out["ats"] = safe(lambda: wind.along_track_shear(
        p, jnp.array([236.0, 0.0, 0.0]), field))
    flags = []
    for k, (v, nan, inf) in out.items():
        if isinstance(v, str):
            flags.append(f"{k}:{v}")
        elif nan:
            flags.append(f"{k}:NaN")
        elif inf:
            flags.append(f"{k}:Inf")
    tag = "  <<< " + ", ".join(flags) if flags else "  ok"
    w = out["w"][0]
    wstr = np.array2string(w, precision=6) if not isinstance(w, str) else w
    print(f"  {label:44s} w = {wstr:40s}{tag}")
    return out


print("=" * 78)
print("D4a  PARKS VORTEX -- CORE BOUNDARY r = r0")
print("=" * 78)
r0, v0 = W.R0, W.V0
print(f"r0 = {r0:.6f} m, V0 = {v0:.6f} m/s, single core at north=0, down=-{HC}")
f = W.vortex1_field
print("\nvalue continuity across r = r0 (approach along +north, d = 0):")
for e in [1e-6, 1e-9, 1e-12, 0.0, -1e-12, -1e-9, -1e-6]:
    p = jnp.array([r0 + e, 0.0, -HC])
    wv = np.asarray(f(p))
    J = np.asarray(jax.jacfwd(f)(p))
    print(f"  r - r0 = {e:+.1e}  w_D = {wv[2]:+.12f}  dW_D/dN = {J[2,0]:+.12e}"
          f"  omega_y = {J[0,2]-J[2,0]:+.6e}")
print("\n  one-sided closed forms at r = r0 exactly:")
print(f"    inside  dW_D/dN = +V0/r0        = {v0/r0:+.12e}")
print(f"    outside dW_D/dN = V0 r0 (r2-2l^2)/r2^2 = {v0*r0*(r0**2-2*r0**2)/r0**4:+.12e}")
print(f"    jump    = {v0/r0 - (-v0/r0):+.12e} 1/s  (a factor of -1 flip)")
print("\n  approach along the vertical (l = 0, d = r):")
for e in [1e-6, 0.0, -1e-6]:
    p = jnp.array([0.0, 0.0, -HC + (r0 + e)])   # aircraft ABOVE the core by r0+e
    wv = np.asarray(f(p))
    J = np.asarray(jax.jacfwd(f)(p))
    print(f"    d - r0 = {e:+.1e}  w_N = {wv[0]:+.12f}   dW_N/dD = {J[0,2]:+.12e}")
print("\n  AT r = r0 EXACTLY jax takes the branch selected by (r2 < r0**2) = False,")
print("  i.e. the OUTSIDE form. omega_y there is 0, not -2V0/r0.")

print("\n  probes:")
probe("core centre r = 0", f, [0.0, 0.0, -HC])
probe("core boundary r = r0 exactly", f, [r0, 0.0, -HC])
probe("just inside r = r0(1-1e-15)", f, [r0 * (1 - 1e-15), 0.0, -HC])
probe("far field r = 1e6 r0", f, [1e6 * r0, 0.0, -HC])
probe("far field r = 1e12 r0", f, [1e12 * r0, 0.0, -HC])
probe("degenerate r0 = 0 core", lambda p: wind.vortex_wind(p, wind.VortexArray(
    north=jnp.array([0.0]), down=jnp.array([-HC]),
    r0=jnp.array(0.0), v0=jnp.array(v0))), [0.0, 0.0, -HC])
probe("degenerate r0 = 0, off-axis", lambda p: wind.vortex_wind(p, wind.VortexArray(
    north=jnp.array([0.0]), down=jnp.array([-HC]),
    r0=jnp.array(0.0), v0=jnp.array(v0))), [100.0, 0.0, -HC])

print("\n  C0 check of the two branches at r = r0 (both forms evaluated by hand):")
d_, l_ = 0.0, r0
print(f"    inside  w_h = V0 d/r0        = {v0*d_/r0:.12e}")
print(f"    outside w_h = V0 r0 d / r^2  = {v0*r0*d_/r0**2:.12e}")
for frac in (0.3, 0.7):
    d_ = frac * r0
    l_ = np.sqrt(r0**2 - d_**2)
    print(f"    at (l,d) = ({l_:.3f},{d_:.3f}) on the circle:"
          f" inside {v0*d_/r0:.12e}  outside {v0*r0*d_/r0**2:.12e}")

print()
print("=" * 78)
print("D4b  UPDRAFT COLUMN -- AXIS r = 0, AND sharpness < 2")
print("=" * 78)
for sh in [0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 6.0]:
    col = wind.UpdraftColumn(north=jnp.array(0.0), east=jnp.array(0.0),
                             w0=jnp.array(wind.UPDRAFT_W0),
                             radius=jnp.array(W.UPDRAFT_RADIUS),
                             sharpness=jnp.array(sh))
    fu = lambda p, c=col: wind.updraft_wind(p, c)  # noqa: E731
    print(f"\n sharpness = {sh}")
    probe("on axis r = 0 exactly", fu, [0.0, 0.0, -HC])
    probe("r = 1e-12 m", fu, [1e-12, 0.0, -HC])
    probe("r = 1e-6 m", fu, [1e-6, 0.0, -HC])
    probe("r = 1 m", fu, [1.0, 0.0, -HC])
    probe("r = 1e9 m (far field)", fu, [1e9, 0.0, -HC])
    # what the 1e-12 clamp does to the PEAK VALUE
    w_axis = -float(fu(jnp.array([0.0, 0.0, -HC]))[2])
    print(f"   peak w_up on axis {w_axis:.12f} vs w0 {float(wind.UPDRAFT_W0):.12f}"
          f"  -> clamp costs {1 - w_axis/float(wind.UPDRAFT_W0):.3e} relative")
    # gradient magnitude just off the clamp
    g = np.asarray(jax.jacfwd(fu)(jnp.array([1e-9, 0.0, -HC])))
    print(f"   |dW_D/dN| at r = 1e-9 m: {abs(g[2,0]):.6e} 1/s")

print()
print("=" * 78)
print("D4c  MICROBURST -- AXIS, GROUND PLANE, AND BELOW IT")
print("=" * 78)
fm = W.mb_field
b = W.BURST
print(f"lam {float(b.lam):.6f} 1/s, R {float(b.radius):.1f} m, "
      f"z* {float(b.z_star):.4f} m, eps {float(b.epsilon):.4f} m")
probe("axis r = 0, z = 150 m", fm, [0.0, 0.0, -150.0])
probe("axis r = 0, z = 0 (ground)", fm, [0.0, 0.0, 0.0])
probe("r = 1e-9 m, z = 150", fm, [1e-9, 0.0, -150.0])
probe("scaled just above the 1e-8 guard", fm, [np.sqrt(1.01e-8) * 1000.0, 0.0, -150.0])
probe("scaled just below the 1e-8 guard", fm, [np.sqrt(0.99e-8) * 1000.0, 0.0, -150.0])
probe("ground plane z = 0, r = 1121 m", fm, [1121.0, 0.0, 0.0])
probe("BELOW ground z = -1 m", fm, [1121.0, 0.0, 1.0])
probe("BELOW ground z = -1000 m", fm, [1121.0, 0.0, 1000.0])
probe("BELOW ground z = -1e6 m", fm, [1121.0, 0.0, 1e6])
probe("far field r = 1e6 m", fm, [1e6, 0.0, -150.0])
probe("far field r = 1e9 m", fm, [1e9, 0.0, -150.0])
probe("very high z = 1e5 m", fm, [1121.0, 0.0, -1e5])

print("\n  the jnp.maximum(-z, 0) tie at EXACTLY z = 0:")
for dz in [1e-6, 1e-12, 0.0, -1e-12, -1e-6]:
    p = jnp.array([1121.0, 0.0, -dz])
    J = np.asarray(jax.jacfwd(fm)(p))
    print(f"    altitude = {dz:+.1e} m   dW_N/dD = {J[0,2]:+.9e}   "
          f"dW_D/dD = {J[2,2]:+.9e}")
print("    one-sided from above (analytic): dW_N/dD = -0.5 lam g(s) n dshp/da")
n_ = 1121.0
s_ = n_**2 / float(b.radius) ** 2
g_ = (1 - np.exp(-s_)) / s_
dshp = -np.exp(0) / float(b.z_star) + np.exp(0) / float(b.epsilon)
print(f"      = {-0.5*float(b.lam)*g_*n_*dshp:+.9e}   "
      f"(the value at z=0 exactly is HALF this)")

print("\n  the r^2 continuity guard 'scaled > 1e-8':")
for s in [1e-6, 1e-7, 1.01e-8, 1e-8, 0.99e-8, 1e-9, 1e-12, 0.0]:
    r = np.sqrt(s) * float(b.radius)
    p = jnp.array([r, 0.0, -150.0])
    wv = np.asarray(fm(p))
    ratio_series = 1 - 0.5 * s
    ratio_exact = (1 - np.exp(-s)) / s if s > 0 else 1.0
    print(f"    scaled = {s:9.3e}  W_N = {wv[0]:+.15e}  "
          f"ratio series {ratio_series:.15f} exact {ratio_exact:.15f}"
          f"  diff {abs(ratio_series-ratio_exact):.2e}")

print()
print("=" * 78)
print("D4d  LEE WAVE -- far field and degenerate wavelength")
print("=" * 78)
probe("north = 0", W.lee_field, [0.0, 0.0, -HC])
probe("north = 1e9 m", W.lee_field, [1e9, 0.0, -HC])
probe("north = 1e15 m", W.lee_field, [1e15, 0.0, -HC])
probe("wavelength = 0 (degenerate)", lambda p: wind.lee_wave_wind(
    p, wind.LeeWave(w0=jnp.array(6.0), wavelength=jnp.array(0.0),
                    north=jnp.array(0.0))), [1.0, 0.0, -HC])

print()
print("=" * 78)
print("D4e  REVERSE-MODE (jnp.where transpose) NaN TRAP, EXPLICIT")
print("=" * 78)
print("jnp.where blocks a NaN VALUE but its transpose sends a zero cotangent into")
print("the unused branch, so 0 * inf can still appear in reverse mode. Checked at")
print("every guard point above via jacrev/along_track_shear; also checked here on")
print("a scalar reduction, which is the shape jax.grad actually takes.")
for name, fld, pts in [
    ("vortex", W.vortex1_field, [[0.0, 0.0, -HC], [r0, 0.0, -HC]]),
    ("updraft p=0.5", lambda p: wind.updraft_wind(p, wind.UpdraftColumn(
        north=jnp.array(0.0), east=jnp.array(0.0), w0=jnp.array(24.4),
        radius=jnp.array(2360.0), sharpness=jnp.array(0.5))),
     [[0.0, 0.0, -HC], [1e-13, 0.0, -HC]]),
    ("updraft p=6", W.updraft_field, [[0.0, 0.0, -HC]]),
    ("microburst", W.mb_field, [[0.0, 0.0, -150.0], [0.0, 0.0, 0.0],
                                [1121.0, 0.0, 0.0], [1121.0, 0.0, 5.0]]),
]:
    for pt in pts:
        p = jnp.array(pt, dtype=float)
        g = np.asarray(jax.grad(lambda q: jnp.sum(fld(q) ** 2))(p))
        gr = np.asarray(jax.grad(lambda q: fld(q)[2])(p))
        bad = np.isnan(g).any() or np.isinf(g).any() or np.isnan(gr).any()
        print(f"  {name:16s} p = {str(pt):26s} grad(sum w^2) = "
              f"{np.array2string(g, precision=4):40s} {'<<< NaN/Inf' if bad else 'ok'}")

print("\n  hypot-at-zero VJP, the classic trap, in isolation:")
print("   jax.grad(hypot)(0,0)            =",
      np.asarray(jax.grad(lambda a: jnp.hypot(a[0], a[1]))(jnp.zeros(2))))
print("   jax.grad(max(hypot/R,1e-12))(0) =",
      np.asarray(jax.grad(lambda a: jnp.maximum(jnp.hypot(a[0], a[1]) / 2360.0,
                                                1e-12))(jnp.zeros(2))))
print("   jax.grad(jnp.maximum(x,0))(0)   =",
      float(jax.grad(lambda x: jnp.maximum(x, 0.0))(0.0)), "  <- the tie rule")
