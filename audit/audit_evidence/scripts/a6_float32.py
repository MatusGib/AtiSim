"""A6: what float32 costs. Run LAST-ish: it flips jax_enable_x64 off mid-process."""
import numpy as np
import jax, jax.numpy as jnp
import flightsim                       # turns x64 ON
from flightsim import integrate, trim, validation, wind
from flightsim.aircraft import CRUISE, REGISTRY

print("### where is x64 set?")
import inspect, flightsim as fs
print("flightsim/__init__.py:")
print(inspect.getsource(fs))
print("x64 now:", jax.config.read("jax_enable_x64"))

# ---- reference: float64 modes ---------------------------------------------
REF = {}
for name in REGISTRY:
    V, H = CRUISE[name]["airspeed"], CRUISE[name]["altitude"]
    ac = REGISTRY[name]
    x, r = trim.trim(jnp.array(V), jnp.array(H), ac)
    a, e, t = float(x[0]), float(x[1]), float(x[2])
    lon = validation.longitudinal_modes(ac, a, e, t, V, H)
    lat = validation.lateral_modes(ac, a, e, t, V, H)
    A = validation.longitudinal_matrix(ac, a, e, t, V, H)
    REF[name] = dict(trim=(a, e, t), res=float(jnp.linalg.norm(r)), lon=lon, lat=lat, A=A)

print("\n=== float64 REFERENCE ===")
for name, d in REF.items():
    print(f"\n{name}: alpha={d['trim'][0]:.12f} de={d['trim'][1]:.12f} thr={d['trim'][2]:.12f} "
          f"|res|={d['res']:.3e}")
    for (wn, z) in d["lon"]:
        print(f"   longitudinal  wn={wn:.10f} rad/s  zeta={z:.10f}")
    dr, rt, st = d["lat"]
    print(f"   dutch roll    wn={dr[0]:.10f} rad/s  zeta={dr[1]:.10f}")
    print(f"   roll tau={rt:.8f} s   spiral tau={st:.8f} s")
    print(f"   cond(A_lon) = {np.linalg.cond(d['A']):.6e}")

# ---- now float32 -----------------------------------------------------------
jax.config.update("jax_enable_x64", False)
print("\n\n### x64 turned OFF; every aircraft leaf cast to float32")
print("x64 now:", jax.config.read("jax_enable_x64"), " jnp.zeros(3).dtype =", jnp.zeros(3).dtype)


def to32(pytree):
    return jax.tree.map(lambda x: jnp.asarray(np.asarray(x), dtype=jnp.float32), pytree)


print("\n=== float32 ===")
for name in REGISTRY:
    V, H = CRUISE[name]["airspeed"], CRUISE[name]["altitude"]
    ac32 = to32(REGISTRY[name])
    print(f"   aircraft leaf dtype check: mass {ac32.mass.dtype}, inertia {ac32.inertia.dtype}")
    # trim.INITIAL_GUESS is a module-level array frozen as float64 at import
    # time, so it must be re-made in float32 or lax.scan rejects the carry.
    g32 = jnp.asarray(np.asarray(trim.INITIAL_GUESS), dtype=jnp.float32)
    x32, r32 = trim.trim(jnp.float32(V), jnp.float32(H), ac32, guess=g32)
    a, e, t = float(x32[0]), float(x32[1]), float(x32[2])
    d64 = REF[name]
    print(f"\n{name}")
    print(f"   trim  f32: alpha={a:.12f} de={e:.12f} thr={t:.12f}  |res|={float(jnp.linalg.norm(r32)):.4e}")
    print(f"   trim  f64: alpha={d64['trim'][0]:.12f} de={d64['trim'][1]:.12f} thr={d64['trim'][2]:.12f}  |res|={d64['res']:.4e}")
    print(f"   trim  ERR: dalpha={a-d64['trim'][0]:+.3e} rad ({np.degrees(a-d64['trim'][0]):+.3e} deg) "
          f"dde={e-d64['trim'][1]:+.3e}  dthr={t-d64['trim'][2]:+.3e}")
    try:
        lon32 = validation.longitudinal_modes(ac32, a, e, t, V, H)
        lat32 = validation.lateral_modes(ac32, a, e, t, V, H)
        A32 = validation.longitudinal_matrix(ac32, a, e, t, V, H)
        print(f"   A_lon dtype {A32.dtype}, cond = {np.linalg.cond(A32):.6e}")
        for i, (wn, z) in enumerate(lon32):
            wr, zr = d64["lon"][i]
            print(f"   lon mode {i}: wn={wn:.10f} (f64 {wr:.10f})  rel err {abs(wn-wr)/wr:.3e}")
            print(f"                zeta={z:.10f} (f64 {zr:.10f})  rel err {abs(z-zr)/abs(zr):.3e}  "
                  f"abs err {z-zr:+.3e}")
        dr32, rt32, st32 = lat32
        dr, rt, st = d64["lat"]
        print(f"   dutch roll : wn={dr32[0]:.10f} (f64 {dr[0]:.10f}) rel err {abs(dr32[0]-dr[0])/dr[0]:.3e}")
        print(f"                zeta={dr32[1]:.10f} (f64 {dr[1]:.10f}) rel err {abs(dr32[1]-dr[1])/abs(dr[1]):.3e} "
              f" abs err {dr32[1]-dr[1]:+.3e}")
        print(f"   roll tau   : {rt32:.8f} (f64 {rt:.8f})  rel err {abs(rt32-rt)/abs(rt):.3e}")
        print(f"   spiral tau : {st32:.8f} (f64 {st:.8f})  rel err {abs(st32-st)/abs(st):.3e}")
    except Exception as exc:
        print(f"   MODE EXTRACTION FAILED in float32: {type(exc).__name__}: {exc}")

# ---- float32 rollout + quaternion norm -------------------------------------
print("\n\n=== float32 6-DOF rollout / quaternion norm ===")
name = "boeing747"
V, H = CRUISE[name]["airspeed"], CRUISE[name]["altitude"]
ac32 = to32(REGISTRY[name])
g32 = jnp.asarray(np.asarray(trim.INITIAL_GUESS), dtype=jnp.float32)
x32, _ = trim.trim(jnp.float32(V), jnp.float32(H), ac32, guess=g32)
st32 = trim.trimmed_state(x32[0], jnp.float32(V), jnp.float32(H))
ct32 = trim.trimmed_controls(x32[1] + jnp.float32(0.02), x32[2])
print("state dtypes:", {k: v.dtype for k, v in st32._asdict().items()})
for n in (1000, 10000, 50000):
    sim = integrate.init_sim(st32, jax.random.PRNGKey(0))
    fin, _ = integrate.rollout(sim, ct32, jnp.float32(0.02), ac32, n)
    print(f"  n={n:>6d}  pos={np.asarray(fin.state.pos_ned)}  "
          f"||q||-1={float(jnp.linalg.norm(fin.state.quat))-1:+.3e}")
print("  NOTE: pos_ned carries ~12192 m; float32 ulp there is "
      f"{np.spacing(np.float32(12192.0)):.4e} m")
