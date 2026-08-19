"""Loose ends: V_MIN binding in dynamic runs, updraft axis guard,
along_track_shear guard, quaternion renormalisation, negative-altitude reach."""
from common import *  # noqa

from flightsim import wind as W
from flightsim.aero import V_MIN, air_data
from flightsim.state import quat_normalize

print("=== does V_MIN ever bind in a dynamic run? ===")
minV = {}
for name in NAMES:
    ac, st, x, r = trim_state(name)
    el, al, rl = (float(ac.elevator_limit), float(ac.aileron_limit),
                  float(ac.rudder_limit))
    lo = np.inf
    for c in [ctrl(), ctrl(de=el), ctrl(de=-el), ctrl(de=el, da=al, dr=rl),
              ctrl(de=-el, da=al, dr=rl), ctrl(de=el, thr=1.0),
              ctrl(de=-el, thr=1.0)]:
        for st0 in [st, st._replace(omega=jnp.array([1.0, 1.0, 1.0])),
                    st._replace(quat=euler_to_quat(jnp.array(0.0),
                                                   jnp.array(1.4),
                                                   jnp.array(0.0)))]:
            _, hist = run(st0, c, ac, 0.005, 24000)
            V = np.linalg.norm(np.asarray(hist.vel_body), axis=1)
            lo = min(lo, float(V.min()))
    minV[name] = lo
    print(f"  {name:20s} lowest |V| over 21 x 120 s runs = {lo:8.3f} m/s "
          f"(V_MIN = {V_MIN})  binds: {lo < V_MIN}")

print("\n=== how far below sea level do fixed-control runs get? ===")
for name in NAMES:
    ac, st, x, r = trim_state(name)
    el = float(ac.elevator_limit)
    lo = np.inf
    for c in [ctrl(de=el), ctrl(de=-el), ctrl(de=-el, thr=1.0), ctrl(thr=0.0)]:
        _, hist = run(st, c, ac, 0.01, 30000)
        lo = min(lo, float((-np.asarray(hist.pos_ned)[:, 2]).min()))
    print(f"  {name:20s} lowest altitude reached = {lo:12.1f} m "
          f"({'BELOW SEA LEVEL' if lo < 0 else 'stayed above'})")

print("\n=== updraft column on the axis (the 1e-12 guard) ===")
col = W.UpdraftColumn(north=jnp.array(0.0), east=jnp.array(0.0),
                      w0=jnp.array(W.UPDRAFT_W0), radius=jnp.array(2400.0),
                      sharpness=jnp.array(2.0))
f = lambda p: W.updraft_wind(p, col)
for rr in [0.0, 1e-15, 1e-9, 1e-3, 1.0, 100.0]:
    p = jnp.array([float(rr), 0.0, -12000.0])
    v = np.asarray(f(p)); J = np.asarray(jax.jacfwd(f)(p))
    print(f"  r={rr:10.3g} m  w_up={-v[2]:10.6f} m/s  max|dW/dx|={np.abs(J).max():.6e} "
          f"finite={bool(np.all(np.isfinite(v)) and np.all(np.isfinite(J)))}")
print("  sharpness < 2 (fractional power) on the axis:")
for sh in [1.0, 1.5, 0.5]:
    c2 = col._replace(sharpness=jnp.array(sh))
    f2 = lambda p: W.updraft_wind(p, c2)
    p = jnp.array([0.0, 0.0, -12000.0])
    v = np.asarray(f2(p)); J = np.asarray(jax.jacfwd(f2)(p))
    print(f"    sharpness={sh}: w_up={-v[2]:.6f} max|dW/dx|={np.abs(J).max():.6e} "
          f"finite={bool(np.all(np.isfinite(J)))}")

print("\n=== along_track_shear with zero ground track (the 1e-9 guard) ===")
wave = W.LeeWave(w0=jnp.array(6.0), wavelength=jnp.array(25000.0),
                 north=jnp.array(0.0))
lee = lambda p: W.lee_wave_wind(p, wave)
for vel in [jnp.array([0.0, 0.0, 0.0]), jnp.array([0.0, 0.0, -50.0]),
            jnp.array([1e-12, 0.0, 0.0]), jnp.array([100.0, 0.0, 0.0])]:
    s = W.along_track_shear(jnp.array([1000.0, 0.0, -12000.0]), vel, lee)
    print(f"  vel_ned={np.asarray(vel)} -> shear = {float(s):+.6e} 1/s^2 "
          f"finite={bool(np.isfinite(float(s)))}")

print("\n=== quaternion: does renormalisation ever have real work to do? ===")
for name in NAMES:
    ac, st, x, r = trim_state(name)
    st0 = st._replace(omega=jnp.array([2.0, 1.5, -1.0]))
    _, hist = run(st0, ctrl(de=float(x[1])), ac, 0.01, 100000)
    q = np.asarray(hist.quat)
    print(f"  {name:20s} post-normalisation |q|-1: max "
          f"{np.abs(np.linalg.norm(q,axis=1)-1).max():.3e}")

print("\n=== a deliberately DENORMALISED initial quaternion ===")
ac, st, x, r = trim_state("cessna172")
for scale in [1.0, 1.5, 0.5, 10.0, 0.01]:
    st0 = st._replace(quat=st.quat * scale)
    _, hist = run(st0, ctrl(de=float(x[1]), thr=float(x[2])), ac, 0.01, 1000)
    q = np.asarray(hist.quat)
    v = np.asarray(hist.vel_body)
    print(f"  |q0| = {scale:6.2f}: after 1 step |q|={np.linalg.norm(q[0]):.10f}; "
          f"final V={np.linalg.norm(v[-1]):8.3f} m/s finite="
          f"{bool(np.all(np.isfinite(v)))}")

print("\n=== zero-length quaternion (pathological) ===")
st0 = st._replace(quat=jnp.zeros(4))
_, hist = run(st0, ctrl(), ac, 0.01, 10)
print(f"  q0 = [0,0,0,0] -> first quat {np.asarray(hist.quat)[0]}, "
      f"finite={bool(np.all(np.isfinite(np.asarray(hist.quat))))}")
