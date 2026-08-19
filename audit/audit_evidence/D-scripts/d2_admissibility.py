"""D2: divergence, vorticity, and the microburst's four stated constants."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import brentq

import flightsim  # noqa: F401
from flightsim import wind
import wcommon as W


def div_curl(field, pos):
    J = jax.vmap(jax.jacfwd(field))(pos)
    div = J[:, 0, 0] + J[:, 1, 1] + J[:, 2, 2]
    curl = jnp.stack([J[:, 2, 1] - J[:, 1, 2],
                      J[:, 0, 2] - J[:, 2, 0],
                      J[:, 1, 0] - J[:, 0, 1]], axis=1)
    # scale: the largest single Jacobian entry, so div is judged against the
    # gradients it is a cancellation of, not against zero.
    scale = jnp.abs(J.reshape(J.shape[0], 9)).max(axis=1)
    return np.asarray(div), np.asarray(curl), np.asarray(scale)


def div_fd(field, pos, h):
    out = []
    for p in np.asarray(pos):
        p = jnp.asarray(p)
        d = 0.0
        for j in range(3):
            e = jnp.zeros(3).at[j].set(h)
            d += float((field(p + e)[j] - field(p - e)[j]) / (2 * h))
        out.append(d)
    return np.array(out)


def grid(kind, n_per_axis):
    """Deterministic grid over the same regime D1 sampled."""
    if kind == "vortex":
        a = np.linspace(-4 * W.R0, W.SPACING + 4 * W.R0, n_per_axis)
        b = np.linspace(-500, 500, 3)
        c = -W.H_CRUISE + np.linspace(-3 * W.R0, 3 * W.R0, n_per_axis)
    elif kind == "updraft":
        a = np.linspace(-2 * W.UPDRAFT_RADIUS, 2 * W.UPDRAFT_RADIUS, n_per_axis)
        b = np.linspace(-2 * W.UPDRAFT_RADIUS, 2 * W.UPDRAFT_RADIUS, n_per_axis)
        c = -W.H_CRUISE + np.linspace(-500, 500, 3)
    elif kind == "lee":
        a = np.linspace(-2 * 25000, 2 * 25000, n_per_axis)
        b = np.linspace(-5000, 5000, 3)
        c = -W.H_CRUISE + np.linspace(-500, 500, 3)
    elif kind == "microburst":
        a = np.linspace(-3 * W.MB_R, 3 * W.MB_R, n_per_axis)
        b = np.linspace(-3 * W.MB_R, 3 * W.MB_R, n_per_axis)
        c = -np.linspace(5.0, 900.0, max(3, n_per_axis // 4))
    A, B, C = np.meshgrid(a, b, c, indexing="ij")
    return jnp.asarray(np.stack([A.ravel(), B.ravel(), C.ravel()], axis=1))


print("=" * 78)
print("D2a  DIVERGENCE AND VORTICITY ON A GRID")
print("=" * 78)
print(f"{'field':34s} {'npts':>6s} {'max|div|':>11s} {'max|div|/|J|max':>16s} "
      f"{'max|curl|':>11s} {'median|curl|':>13s}")
for name, (kind, field) in W.FIELDS.items():
    pos = grid(kind, 21)
    div, curl, scale = div_curl(field, pos)
    cn = np.linalg.norm(curl, axis=1)
    rel = np.abs(div) / np.maximum(scale, 1e-300)
    print(f"{name:34s} {len(pos):6d} {np.abs(div).max():11.3e} {rel.max():16.3e} "
          f"{cn.max():11.3e} {np.median(cn):13.3e}")

print()
print("cross-check: divergence by central finite difference (h = 0.5 m), 200 pts")
for name, (kind, field) in W.FIELDS.items():
    pos = grid(kind, 7)[:200]
    d = div_fd(field, pos, 0.5)
    print(f"  {name:34s} max|div_fd| = {np.abs(d).max():.3e} 1/s")

print()
print("Rankine vorticity structure (single core, Parks hannibal):")
for rr in [0.0, 0.25, 0.5, 0.75, 0.99, 1.0, 1.001, 1.5, 3.0, 10.0]:
    p = jnp.array([rr * W.R0, 0.0, -W.H_CRUISE])
    J = jax.jacfwd(W.vortex1_field)(p)
    wy = float(J[0, 2] - J[2, 0])
    print(f"  r/r0 = {rr:6.3f}   omega_y = {wy:+.6e} 1/s"
          f"   (solid-body value -2 V0/r0 = {-2*W.V0/W.R0:+.6e})")

print()
print("=" * 78)
print("D2b  THE FOUR OSEGUERA-BOWLES CONSTANTS")
print("=" * 78)
b = W.BURST
lam, R = float(b.lam), float(b.radius)
zs, eps = float(b.z_star), float(b.epsilon)

# 1. peak outflow radius: solves exp(-x^2)(2x^2+1) = 1
f1 = lambda x: np.exp(-x * x) * (2 * x * x + 1) - 1.0
x_root = brentq(f1, 0.5, 3.0, xtol=1e-14, rtol=1e-15)
# and directly from the shipped field, by dense scan on the peak-outflow plane
z_m_used = wind.MICROBURST_ZM_OVER_ZSTAR * zs
rr = np.linspace(1.0, 4 * R, 400001)
u = np.array(jax.vmap(lambda r: W.mb_field(jnp.array([r, 0.0, -z_m_used]))[0])(jnp.asarray(rr)))
r_peak = rr[int(np.argmax(u))]
print(f"1) r/R at peak outflow")
print(f"   stated                       {wind.MICROBURST_PEAK_RADIUS_RATIO:.10f}")
print(f"   root of exp(-x^2)(2x^2+1)=1  {x_root:.10f}")
print(f"   measured on the shipped field {r_peak / R:.10f}")
print(f"   stated vs exact root: rel err {abs(wind.MICROBURST_PEAK_RADIUS_RATIO - x_root)/x_root:.3e}")

# 2. z_m / z*
zm_exact = np.log(wind.MICROBURST_ZSTAR_OVER_EPS) / (wind.MICROBURST_ZSTAR_OVER_EPS - 1.0)
zz = np.linspace(0.1, 5 * zs, 400001)
uu = np.array(jax.vmap(lambda z: W.mb_field(jnp.array([x_root * R, 0.0, -z]))[0])(jnp.asarray(zz)))
z_peak = zz[int(np.argmax(uu))]
print(f"\n2) z_m / z*")
print(f"   stated                        {wind.MICROBURST_ZM_OVER_ZSTAR:.10f}")
print(f"   exact ln(k)/(k-1), k=12.5     {zm_exact:.10f}")
print(f"   measured on the shipped field {z_peak / zs:.10f}")
print(f"   stated vs exact: rel err      {abs(wind.MICROBURST_ZM_OVER_ZSTAR - zm_exact)/zm_exact:.3e}")

# 3. z*/eps -- a declared model input, not derivable; check the constructor honours it
print(f"\n3) z* / epsilon")
print(f"   stated   {wind.MICROBURST_ZSTAR_OVER_EPS}")
print(f"   built    {zs / eps:.12f}   (constructor identity, not a physical check)")
print(f"   NOTE: this is an INPUT to the model, not a consequence of eqs. 5-6.")

# 4. u_max = 0.2357 lam R
gx = (1 - np.exp(-x_root**2)) / x_root
shape_at = lambda t: np.exp(-t) - np.exp(-t * wind.MICROBURST_ZSTAR_OVER_EPS)
coeff_exact = 0.5 * gx * shape_at(zm_exact)
coeff_at_022 = 0.5 * gx * shape_at(wind.MICROBURST_ZM_OVER_ZSTAR)
u_meas = float(u.max())
print(f"\n4) u_max / (lambda R)")
print(f"   stated                          {wind.MICROBURST_UMAX_COEFF:.10f}")
print(f"   closed form at exact z_m/z*     {coeff_exact:.10f}")
print(f"   closed form at the shipped 0.22 {coeff_at_022:.10f}")
print(f"   measured on the shipped field   {u_meas / (lam * R):.10f}")
print(f"   stated vs exact: rel err        {abs(wind.MICROBURST_UMAX_COEFF - coeff_exact)/coeff_exact:.3e}")

print("\n--- consequence of the 0.22 rounding for wind.microburst() ---")
print(f"   requested u_max {W.MB_UMAX} m/s, requested z_m {W.MB_ZM} m")
print(f"   delivered u_max {u_meas:.6f} m/s        (rel err {abs(u_meas-W.MB_UMAX)/W.MB_UMAX:.3e})")
print(f"   delivered z_m   {z_peak:.6f} m          (rel err {abs(z_peak-W.MB_ZM)/W.MB_ZM:.3e})")
print(f"   delivered r_peak {r_peak:.3f} m vs 1.1212R = {1.1212*R:.3f} m")

print("\n--- the paper's fifth statement: w_max = lam z* (exp(-z_h/z*) - 0.92) ---")
print("   (not in the module's constant list; checked because test_microburst.py cites it)")
for zh in [100.0, 300.0, 600.0, 1000.0]:
    w_model = -float(W.mb_field(jnp.array([0.0, 0.0, -zh]))[2])
    w_paper = lam * zs * (np.exp(-zh / zs) - 0.92)
    print(f"   z = {zh:6.0f} m   model w_up {w_model:+9.4f}   paper form {w_paper:+9.4f}"
          f"   diff {abs(w_model-w_paper):.3e}")

print("\n--- continuity of the microburst in cylindrical form, symbolically checked ---")
print("   (1/r) d(r u)/dr = lam exp(-(r/R)^2) [e^{-z/z*} - e^{-z/eps}]")
print("   dw/dz          = -lam exp(-(r/R)^2) [e^{-z/z*} - e^{-z/eps}]     -> sum 0")
rs = np.linspace(1.0, 4000.0, 60)
zsamp = np.linspace(1.0, 1500.0, 40)
worst = 0.0
for r in rs:
    for z in zsamp:
        p = jnp.array([r, 0.0, -z])
        J = jax.jacfwd(W.mb_field)(p)
        dd = float(J[0, 0] + J[1, 1] + J[2, 2])
        sc = float(jnp.abs(J).max())
        worst = max(worst, abs(dd) / sc)
print(f"   worst |div| / max|dJ| over a 60x40 (r,z) grid: {worst:.3e}")
