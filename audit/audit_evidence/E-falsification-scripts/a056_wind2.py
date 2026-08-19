"""ATTACK 5/6/8 (vectorised): wind-field discontinuities, ground, superposition."""
from common import *  # noqa

from flightsim import wind as W
from flightsim import atmosphere as atm
from flightsim.state import euler_to_quat

Q0 = euler_to_quat(jnp.array(0.0), jnp.array(0.0), jnp.array(0.0))


def vals_and_jac(f, pts):
    v = jax.jit(jax.vmap(f))(pts)
    J = jax.jit(jax.vmap(jax.jacfwd(f)))(pts)
    return np.asarray(v), np.asarray(J)


# ---------------------------------------------------------------------------
print("=== 6a. Parks vortex: across the core edge r = r0 ===")
case = W.PARKS_CASES["hannibal"]
arr = W.VortexArray(north=jnp.array([0.0]), down=jnp.array([0.0]),
                    r0=jnp.array(case["r0"]), v0=jnp.array(case["v0"]))
r0 = float(case["r0"])
f = lambda p: W.vortex_wind(p, arr)
print(f"  r0 = {r0:.3f} m, v0 = {float(case['v0']):.3f} m/s")

for label, mk in [("radial (aircraft above core)",
                   lambda r: jnp.stack([jnp.zeros_like(r), jnp.zeros_like(r), -r], -1)),
                  ("tangential (level with core)",
                   lambda r: jnp.stack([r, jnp.zeros_like(r), jnp.zeros_like(r)], -1)),
                  ("45 deg",
                   lambda r: jnp.stack([r/np.sqrt(2), jnp.zeros_like(r),
                                        -r/np.sqrt(2)], -1))]:
    rs = jnp.asarray(r0 * np.linspace(0.98, 1.02, 4001))
    pts = mk(rs)
    v, J = vals_and_jac(f, pts)
    ds = float(rs[1] - rs[0])
    H = np.asarray(jax.jit(jax.vmap(jax.jacfwd(jax.jacfwd(f))))(pts))
    print(f"  {label:30s}")
    print(f"    max |dW| between adjacent samples  = {np.abs(np.diff(v,axis=0)).max():.4e} m/s"
          f"   (sample spacing {ds:.4f} m)")
    print(f"    max |d(dW/dx)| between adjacent    = {np.abs(np.diff(J,axis=0)).max():.4e} 1/s")
    print(f"    max |d2W/dx2| just inside / at / just outside r0 = "
          f"{np.abs(H[1900]).max():.4e} / {np.abs(H[2000]).max():.4e} / "
          f"{np.abs(H[2100]).max():.4e} 1/(s.m)")

print("\n  values straddling the edge (radial cut):")
print(f"  {'r/r0':>10s} {'w_h':>11s} {'w_up':>11s} {'|dW/dz|':>12s} {'|d2W/dz2|':>13s}")
for frac in [0.98, 0.999, 0.99999, 1.0, 1.00001, 1.001, 1.02]:
    p = jnp.array([0.0, 0.0, -r0 * frac])
    v = np.asarray(f(p)); J = np.asarray(jax.jacfwd(f)(p))
    H = np.asarray(jax.jacfwd(jax.jacfwd(f))(p))
    print(f"  {frac:10.5f} {v[0]:11.6f} {-v[2]:11.6f} {np.abs(J[:,2]).max():12.8f} "
          f"{np.abs(H[:,2,2]).max():13.8f}")

# ---------------------------------------------------------------------------
print("\n=== 6b/5a. microburst ground plane z = 0 ===")
mb = W.microburst(u_max=25.0, radius=1000.0, z_m=100.0)
g = lambda p: W.microburst_wind(p, mb)
print(f"  {'h (m)':>11s} {'u_N':>11s} {'w_down':>11s} {'max|dW/dz|':>13s}")
for h in [200.0, 50.0, 10.0, 1.0, 0.01, 0.0, -0.01, -1.0, -100.0, -10000.0]:
    p = jnp.array([700.0, 0.0, -h])
    v = np.asarray(g(p)); J = np.asarray(jax.jacfwd(g)(p))
    print(f"  {h:11.2f} {v[0]:11.6f} {v[2]:11.6f} {np.abs(J[:,2]).max():13.8f}")

hs = jnp.asarray(np.linspace(-0.5, 0.5, 4001))
pts = jnp.stack([jnp.full_like(hs, 700.0), jnp.zeros_like(hs), -hs], -1)
v, J = vals_and_jac(g, pts)
i0 = int(np.argmin(np.abs(np.asarray(hs))))
print(f"\n  fine sweep across h=0 (spacing {float(hs[1]-hs[0]):.5f} m):")
print(f"    max |dW| between adjacent samples   = {np.abs(np.diff(v,axis=0)).max():.4e} m/s"
      f"   -> field is C0 (continuous in value)")
print(f"    max |d(dW/dz)| between adjacent     = {np.abs(np.diff(J,axis=0)).max():.4e} 1/s")
print(f"    dW/dz five samples ABOVE h=0: {J[i0+5][:,2]}")
print(f"    dW/dz five samples BELOW h=0: {J[i0-5][:,2]}")
print("    -> vertical gradient drops to EXACTLY zero below ground: a C1 KINK")

print("\n  omega_gust across the ground plane (what actually reaches the aircraft):")
for h in [5.0, 1.0, 0.1, 0.001, 0.0, -0.001, -0.1, -1.0, -5.0]:
    p = jnp.array([700.0, 0.0, -float(h)])
    og = np.asarray(W.gust_rates(p, Q0, g))
    print(f"    h={h:8.3f} m  omega_gust = [{og[0]:+.6e} {og[1]:+.6e} {og[2]:+.6e}]")

# ---------------------------------------------------------------------------
print("\n=== 5b. FLY INTO THE GROUND ===")
ac = REGISTRY["boeing747_approach"]
V = CRUISE["boeing747_approach"]["airspeed"]
x, _ = trim.trim(jnp.array(V), jnp.array(300.0), ac)
st = trim.trimmed_state(x[0], jnp.array(V), jnp.array(300.0))
c = ctrl(de=float(x[1]), thr=float(x[2]))
dt, n = 0.01, 20000
_, hist = run(st, c, ac, dt, n, wind_model=W.field_model(g))
h = -np.asarray(hist.pos_ned)[:, 2]
vb = np.asarray(hist.vel_body)
below = np.where(h < 0)[0]
print(f"  747-approach released at 300 m into a 25 m/s microburst, fixed controls")
print(f"  min altitude = {h.min():.1f} m at t = {np.argmin(h)*dt:.1f} s")
if len(below):
    i = below[0]
    vz = float((np.asarray(hist.pos_ned)[i, 2]
                - np.asarray(hist.pos_ned)[i-1, 2]) / dt)
    print(f"  crosses h = 0 at t = {i*dt:.2f} s with sink rate {vz:+.2f} m/s "
          f"({vz*196.85:+.0f} ft/min) and V = {np.linalg.norm(vb[i]):.1f} m/s")
    print(f"  THE RUN CONTINUES. At t = {n*dt:.0f} s: h = {h[-1]:.1f} m, "
          f"V = {np.linalg.norm(vb[-1]):.1f} m/s, finite = "
          f"{bool(np.all(np.isfinite(vb)))}")
    print(f"  deepest penetration = {h.min():.1f} m below sea level")
    print(f"  air density there = {float(atm.density(jnp.array(h.min()))):.5f} kg/m^3 "
          f"({100*(float(atm.density(jnp.array(h.min())))/1.225-1):+.2f}% vs sea level); "
          f"T = {float(atm.temperature(jnp.array(h.min()))):.2f} K")
else:
    print("  never reached the ground")

print("\n  microburst wind BELOW ground (the clamp holds the h=0 value):")
for hh in [0.0, -10.0, -1000.0, -1e5]:
    v = np.asarray(g(jnp.array([700.0, 0.0, -hh])))
    print(f"    h={hh:10.0f} m -> W = [{v[0]:+.6f} {v[1]:+.6f} {v[2]:+.6f}] m/s")

# ---------------------------------------------------------------------------
print("\n=== 8. superposition ===")
hc, mc = W.PARKS_CASES["hannibal"], W.PARKS_CASES["morton"]
nv = 5
a1 = W.VortexArray(north=jnp.arange(nv) * hc["spacing"], down=jnp.zeros(nv),
                   r0=jnp.array(hc["r0"]), v0=jnp.array(hc["v0"]))
a2 = W.VortexArray(north=jnp.arange(nv) * mc["spacing"] + 500.0,
                   down=jnp.full(nv, 300.0),
                   r0=jnp.array(mc["r0"]), v0=jnp.array(mc["v0"]))
col = W.UpdraftColumn(north=jnp.array(4000.0), east=jnp.array(0.0),
                      w0=jnp.array(W.UPDRAFT_W0), radius=jnp.array(2400.0),
                      sharpness=jnp.array(2.0))
wave = W.LeeWave(w0=jnp.array(6.0), wavelength=jnp.array(W.LEE_WAVE_WAVELENGTH),
                 north=jnp.array(0.0))
f1 = lambda p: W.vortex_wind(p, a1)
f2 = lambda p: W.vortex_wind(p, a2)
f3 = lambda p: W.updraft_wind(p, col)
f4 = lambda p: W.lee_wave_wind(p, wave)
comb = W.superpose(f1, f2, f3, f4)

pts = jax.random.uniform(jax.random.PRNGKey(7), (2000, 3),
                         minval=-6000.0, maxval=6000.0)
vc, Jc = vals_and_jac(comb, pts)
vs = sum(np.asarray(jax.jit(jax.vmap(fi))(pts)) for fi in (f1, f2, f3, f4))
Js = sum(np.asarray(jax.jit(jax.vmap(jax.jacfwd(fi)))(pts)) for fi in (f1, f2, f3, f4))
print(f"  linearity: max |W_sum - sum W| = {np.abs(vc-vs).max():.3e} m/s")
print(f"             max |J_sum - sum J| = {np.abs(Jc-Js).max():.3e} 1/s")

print("\n  divergence (incompressibility) of each field over 2000 random points:")
for lbl, fi in [("vortex array (Hannibal)", f1), ("vortex array (Morton)", f2),
                ("updraft column", f3), ("lee wave", f4),
                ("microburst", g), ("all four superposed", comb)]:
    J = np.asarray(jax.jit(jax.vmap(jax.jacfwd(fi)))(pts))
    d = np.abs(np.trace(J, axis1=1, axis2=2))
    W_mag = np.linalg.norm(np.asarray(jax.jit(jax.vmap(fi))(pts)), axis=1)
    print(f"    {lbl:24s} max |div W| = {d.max():.4e} 1/s   "
          f"median {np.median(d):.3e}   (peak |W| here {W_mag.max():.2f} m/s)")

og = np.asarray(jax.jit(jax.vmap(lambda p: W.gust_rates(p, Q0, comb)))(pts))
print(f"\n  non-finite W or omega_gust at "
      f"{int((~np.isfinite(vc).all(1)).sum() + (~np.isfinite(og).all(1)).sum())}"
      f"/{len(pts)} random points")
print(f"  peak |omega_gust| in the superposition = {np.linalg.norm(og,axis=1).max():.5f} rad/s")

gn = jnp.asarray(np.linspace(-2000, 16000, 300))
gd = jnp.asarray(np.linspace(-2000, 2000, 150))
GN, GD = jnp.meshgrid(gn, gd, indexing="ij")
grid = jnp.stack([GN.ravel(), jnp.zeros(GN.size), GD.ravel()], -1)
m1 = float(jnp.linalg.norm(jax.jit(jax.vmap(f1))(grid), axis=1).max())
mc_ = float(jnp.linalg.norm(jax.jit(jax.vmap(comb))(grid), axis=1).max())
print(f"\n  peak |W|: single Hannibal array = {m1:.2f} m/s;  "
      f"four-field superposition = {mc_:.2f} m/s")

print("\n  TWO OVERLAPPING vortex arrays only (co-located, same case):")
a3 = W.VortexArray(north=jnp.arange(nv) * hc["spacing"] + 1.0,
                   down=jnp.zeros(nv),
                   r0=jnp.array(hc["r0"]), v0=jnp.array(hc["v0"]))
f5 = lambda p: W.vortex_wind(p, a3)
ov = W.superpose(f1, f5)
vo = np.asarray(jax.jit(jax.vmap(ov))(grid))
print(f"    peak |W| = {np.linalg.norm(vo,axis=1).max():.2f} m/s "
      f"(single array {m1:.2f}); finite everywhere = "
      f"{bool(np.all(np.isfinite(vo)))}")
Jo = np.asarray(jax.jit(jax.vmap(jax.jacfwd(ov)))(pts))
print(f"    max |div W| = {np.abs(np.trace(Jo,axis1=1,axis2=2)).max():.4e} 1/s")
