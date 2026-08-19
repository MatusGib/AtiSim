"""D1: cross-check wind.gust_rates against jacfwd, jacrev, central FD, and hand algebra."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import jax
import jax.numpy as jnp

import flightsim  # noqa: F401
from flightsim import wind
from flightsim.state import quat_to_dcm
import wcommon as W

N = 400
KEY = jax.random.PRNGKey(20260817)


def rates_from_jac(jac_ned, quat):
    """The gust_rates combination, applied to an externally supplied Jacobian."""
    dcm = quat_to_dcm(quat)
    g = dcm.T @ jac_ned @ dcm
    return jnp.array([g[2, 1], -g[2, 0], g[1, 0]])


def jac_rev(field, p):
    return jax.jacrev(field)(p)


def jac_fd(field, p, h):
    """Central difference, column j = (f(p+h e_j) - f(p-h e_j)) / 2h."""
    cols = []
    for j in range(3):
        e = jnp.zeros(3).at[j].set(h)
        cols.append((field(p + e) - field(p - e)) / (2 * h))
    return jnp.stack(cols, axis=1)


# --- hand-derived closed-form Jacobians --------------------------------------
def jac_vortex_hand(p, arr):
    """d(wind_ned)_i/d(pos_ned)_j for the Parks array, differentiated by hand.

    One core, l = x - n_c, d = z_c - z, r2 = l^2 + d^2.
      outside: W_N = v0 r0 d / r2 ,  W_D = v0 r0 l / r2
      inside : W_N = v0 d / r0     ,  W_D = v0 l / r0
    dl/dx=1, dd/dz=-1, dr2/dx=2l, dr2/dz=-2d.
      outside  dW_N/dx = -2 v0 r0 l d / r2^2
               dW_N/dz =  v0 r0 (2 d^2 - r2) / r2^2
               dW_D/dx =  v0 r0 (r2 - 2 l^2) / r2^2
               dW_D/dz =  2 v0 r0 l d / r2^2
      inside   dW_N/dz = -v0/r0 ,  dW_D/dx = v0/r0 , others 0
    Every d/dy is identically zero (the field has no east dependence and no
    east component).
    """
    v0, r0 = float(arr.v0), float(arr.r0)
    J = np.zeros((3, 3))
    for nc, dc in zip(np.asarray(arr.north), np.asarray(arr.down)):
        l = float(p[0]) - nc
        d = dc - float(p[2])
        r2 = l * l + d * d
        if r2 < r0 * r0:
            J[0, 2] += -v0 / r0
            J[2, 0] += v0 / r0
        else:
            J[0, 0] += -2 * v0 * r0 * l * d / r2**2
            J[0, 2] += v0 * r0 * (2 * d * d - r2) / r2**2
            J[2, 0] += v0 * r0 * (r2 - 2 * l * l) / r2**2
            J[2, 2] += 2 * v0 * r0 * l * d / r2**2
    return jnp.array(J)


def jac_lee_hand(p, wave):
    """W = [0, 0, w0 cos(phase)], phase = 2 pi (x - n0)/L.
    dW_D/dx = -w0 (2 pi / L) sin(phase); every other entry is exactly zero."""
    w0, L, n0 = float(wave.w0), float(wave.wavelength), float(wave.north)
    phase = 2 * np.pi * (float(p[0]) - n0) / L
    J = np.zeros((3, 3))
    J[2, 0] = -w0 * (2 * np.pi / L) * np.sin(phase)
    return jnp.array(J)


def jac_updraft_hand(p, col):
    """W = [0,0,-w0 exp(-s^k)], s = max(rho/R, 1e-12), rho = hypot(dx,dy).
    dW_D/dx = w0 k s^(k-1) exp(-s^k) dx/(rho R), same for y. dW_D/dz = 0."""
    w0, R, k = float(col.w0), float(col.radius), float(col.sharpness)
    dx = float(p[0]) - float(col.north)
    dy = float(p[1]) - float(col.east)
    rho = np.hypot(dx, dy)
    s = max(rho / R, 1e-12)
    J = np.zeros((3, 3))
    if rho > 0:
        pref = w0 * k * s ** (k - 1) * np.exp(-(s**k)) / (rho * R)
        J[2, 0] = pref * dx
        J[2, 1] = pref * dy
    return jnp.array(J)


def jac_mb_hand(p, b):
    """Oseguera-Bowles, altitude a = max(-z, 0), s = (n^2+e^2)/R^2.
      W_N = 0.5 lam g(s) shp(a) n,  g(s) = (1-e^-s)/s,  shp = e^{-a/z*} - e^{-a/eps}
      W_D = lam e^{-s} [ z*(1-e^{-a/z*}) - eps(1-e^{-a/eps}) ]
    ds/dn = 2n/R^2 ; g'(s) = (s e^{-s} - (1-e^{-s}))/s^2 ; da/dz = -1 (a>0).
    """
    lam, R = float(b.lam), float(b.radius)
    zs, eps = float(b.z_star), float(b.epsilon)
    n = float(p[0]) - float(b.north)
    e = float(p[1]) - float(b.east)
    a = max(-float(p[2]), 0.0)
    s = (n * n + e * e) / R**2
    if s > 1e-8:
        g = (1 - np.exp(-s)) / s
        gp = (s * np.exp(-s) - (1 - np.exp(-s))) / s**2
    else:
        g = 1 - 0.5 * s
        gp = -0.5
    shp = np.exp(-a / zs) - np.exp(-a / eps)
    dshp_da = -np.exp(-a / zs) / zs + np.exp(-a / eps) / eps
    dsdn, dsde = 2 * n / R**2, 2 * e / R**2
    J = np.zeros((3, 3))
    # W_N = 0.5 lam shp (g n)
    J[0, 0] = 0.5 * lam * shp * (gp * dsdn * n + g)
    J[0, 1] = 0.5 * lam * shp * (gp * dsde * n)
    J[1, 0] = 0.5 * lam * shp * (gp * dsdn * e)
    J[1, 1] = 0.5 * lam * shp * (gp * dsde * e + g)
    dadz = -1.0 if a > 0 else 0.0
    J[0, 2] = 0.5 * lam * g * n * dshp_da * dadz
    J[1, 2] = 0.5 * lam * g * e * dshp_da * dadz
    # W_D = lam e^{-s} * P(a),  P = z*(1-e^{-a/z*}) - eps(1-e^{-a/eps})
    P = zs * (1 - np.exp(-a / zs)) - eps * (1 - np.exp(-a / eps))
    dP = np.exp(-a / zs) - np.exp(-a / eps)
    J[2, 0] = lam * (-np.exp(-s)) * dsdn * P
    J[2, 1] = lam * (-np.exp(-s)) * dsde * P
    J[2, 2] = lam * np.exp(-s) * dP * dadz
    return jnp.array(J)


HAND = {
    "vortex(hannibal, 2 cores)": lambda p: jac_vortex_hand(p, W.VORTEX),
    "lee_wave(w0=6, L=25km)": lambda p: jac_lee_hand(p, W.WAVE),
    "updraft(w0=24.4, R=2360, p=6)": lambda p: jac_updraft_hand(p, W.COLUMN),
    "microburst(19.03, R=1000, zm=150)": lambda p: jac_mb_hand(p, W.BURST),
}

FD_STEPS = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]


def main():
    print(f"x64 = {jax.config.jax_enable_x64}")
    print(f"N = {N} random (position, quaternion) pairs per field\n")
    rows = []
    for name, (kind, field) in W.FIELDS.items():
        k1, k2 = jax.random.split(jax.random.fold_in(KEY, abs(hash(name)) % 10000))
        pos = W.sample_positions(k1, N, kind)
        quat = W.random_quats(k2, N)

        gr = np.asarray(jax.vmap(lambda p, q: wind.gust_rates(p, q, field))(pos, quat))
        scale = np.linalg.norm(gr, axis=-1)
        # floor relative to the LARGEST rate in the sample, so a point where the
        # field happens to be flat cannot manufacture a huge relative error.
        floor = float(scale.max()) * 1e-3

        out = {"field": name, "median|gust_rates|": float(np.median(scale)),
               "max|gust_rates|": float(scale.max())}

        rv = np.asarray(jax.vmap(
            lambda p, q: rates_from_jac(jac_rev(field, p), q))(pos, quat))
        e = W.relerr(gr, rv, floor)
        out["jacrev"] = (float(np.median(e)), float(e.max()))

        # explicit jacfwd through the same combination (not a no-op: it checks
        # the dcm.T J dcm contraction and the component picks, not the AD)
        fw = np.asarray(jax.vmap(
            lambda p, q: rates_from_jac(jax.jacfwd(field)(p), q))(pos, quat))
        e = W.relerr(gr, fw, floor)
        out["jacfwd(explicit)"] = (float(np.median(e)), float(e.max()))

        for h in FD_STEPS:
            fd = np.asarray(jax.vmap(
                lambda p, q: rates_from_jac(jac_fd(field, p, h), q))(pos, quat))
            e = W.relerr(gr, fd, floor)
            out[f"cfd h={h:g}"] = (float(np.median(e)), float(e.max()))

        hj = np.stack([np.asarray(HAND[name](p)) for p in np.asarray(pos)])
        hr = np.asarray(jax.vmap(rates_from_jac)(jnp.asarray(hj), quat))
        e = W.relerr(gr, hr, floor)
        out["hand-derived"] = (float(np.median(e)), float(e.max()))

        # also compare the raw 3x3 Jacobians, hand vs jacfwd, in Frobenius norm
        aj = np.asarray(jax.vmap(jax.jacfwd(field))(pos))
        jn = np.linalg.norm(aj.reshape(N, 9), axis=1)
        jerr = np.linalg.norm((aj - hj).reshape(N, 9), axis=1) / np.maximum(jn, 1e-30)
        out["hand-vs-AD full Jacobian"] = (float(np.median(jerr)), float(jerr.max()))
        rows.append(out)

    keys = ["jacrev", "jacfwd(explicit)"] + [f"cfd h={h:g}" for h in FD_STEPS] + \
           ["hand-derived", "hand-vs-AD full Jacobian"]
    hdr = f"{'comparison':26s}" + "".join(f"{r['field'][:22]:>26s}" for r in rows)
    print(hdr)
    print("-" * len(hdr))
    for k in keys:
        line = f"{k:26s}"
        for r in rows:
            med, mx = r[k]
            line += f"{med:11.2e}/{mx:<13.2e}"
        print(line)
    print("\n(each cell is median / max relative error, floored at 1e-3 x median|rates|)")
    print()
    for r in rows:
        print(f"{r['field']:34s} median|gust_rates| = {r['median|gust_rates|']:.4e} rad/s"
              f"   max = {r['max|gust_rates|']:.4e} rad/s")


if __name__ == "__main__":
    main()
