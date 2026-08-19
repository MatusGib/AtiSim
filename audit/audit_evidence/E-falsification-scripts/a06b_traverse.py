"""ATTACK 6b: what the core-edge gradient jump costs a rollout.

A smooth field should give RK4's 4th order under dt refinement. A field whose
gradient steps inside a step has no valid Taylor expansion there.
Compare a vortex traverse (crosses r0) with a lee wave (analytic, C-infinity).
"""
from common import *  # noqa

from flightsim import wind as W

DTS = [0.04, 0.02, 0.01, 0.005, 0.0025, 0.00125]
SEC = 20.0


def traverse(field, ac, st0, c, dt, sec):
    n = int(round(sec / dt))
    _, hist = run(st0, c, ac, dt, n, wind_model=W.field_model(field))
    return jax.tree.map(lambda a: np.asarray(a)[-1], hist)


def order_table(label, field, name="boeing747", offset_down=0.0):
    ac, st, x, r = trim_state(name)
    Vc = CRUISE[name]["airspeed"]
    hc = CRUISE[name]["altitude"]
    st0 = st._replace(pos_ned=jnp.array([-1500.0, 0.0, -hc + offset_down]))
    c = ctrl(de=float(x[1]), thr=float(x[2]))
    finals = [traverse(field, ac, st0, c, dt, SEC) for dt in DTS]
    ref = finals[-1]

    def err(a, b):
        return max(np.abs(np.asarray(getattr(a, f)) - np.asarray(getattr(b, f))).max()
                   for f in ("pos_ned", "vel_body", "quat", "omega"))
    print(f"\n  {label}")
    print(f"    {'dt':>9s} {'err vs finest':>15s} {'ratio':>8s} {'implied order':>14s}")
    prev = None
    for dt, fin in zip(DTS[:-1], finals[:-1]):
        e = err(fin, ref)
        if prev is None:
            print(f"    {dt:9.5f} {e:15.6e} {'':>8s} {'':>14s}")
        else:
            ratio = prev / e if e > 0 else float("inf")
            print(f"    {dt:9.5f} {e:15.6e} {ratio:8.2f} "
                  f"{np.log2(ratio) if np.isfinite(ratio) and ratio>0 else float('nan'):14.2f}")
        prev = e


hc = CRUISE["boeing747"]["altitude"]
case = W.PARKS_CASES["hannibal"]
r0, v0 = float(case["r0"]), float(case["v0"])
# single vortex, core AT the aircraft's altitude, so a level traverse cuts
# straight through the core and crosses r = r0 twice
arr = W.VortexArray(north=jnp.array([0.0]), down=jnp.array([-hc]),
                    r0=jnp.array(r0), v0=jnp.array(v0))
vortex = lambda p: W.vortex_wind(p, arr)
# same vortex but the aircraft passes 3*r0 BELOW the core: never enters the core
arr_far = W.VortexArray(north=jnp.array([0.0]), down=jnp.array([-hc - 3 * r0]),
                        r0=jnp.array(r0), v0=jnp.array(v0))
vortex_far = lambda p: W.vortex_wind(p, arr_far)
wave = W.LeeWave(w0=jnp.array(6.0), wavelength=jnp.array(W.LEE_WAVE_WAVELENGTH),
                 north=jnp.array(0.0))
lee = lambda p: W.lee_wave_wind(p, wave)
col = W.UpdraftColumn(north=jnp.array(0.0), east=jnp.array(0.0),
                      w0=jnp.array(W.UPDRAFT_W0), radius=jnp.array(2400.0),
                      sharpness=jnp.array(2.0))
updraft = lambda p: W.updraft_wind(p, col)

print("=== dt refinement through different wind fields, 747 @ cruise, 20 s ===")
print("  (error is max abs state difference against the dt=0.00125 run)")
order_table("lee wave (C-infinity, smooth)", lee)
order_table("updraft column (C-infinity, Gaussian)", updraft)
order_table("Parks vortex, traverse OUTSIDE the core (never crosses r0)",
            vortex_far)
order_table("Parks vortex, traverse THROUGH the core (crosses r0 twice)",
            vortex)

print("\n=== confirm the core is actually entered ===")
ac, st, x, r = trim_state("boeing747")
st0 = st._replace(pos_ned=jnp.array([-1500.0, 0.0, -hc]))
c = ctrl(de=float(x[1]), thr=float(x[2]))
_, hist = run(st0, c, ac, 0.005, int(SEC / 0.005),
              wind_model=W.field_model(vortex))
p = np.asarray(hist.pos_ned)
rr = np.hypot(p[:, 0] - 0.0, (-hc) - p[:, 2])
print(f"  min r/r0 along the through-core run  = {rr.min()/r0:.4f} "
      f"(inside for {int((rr<r0).sum())} of {len(rr)} steps)")
_, hist2 = run(st0, c, ac, 0.005, int(SEC / 0.005),
               wind_model=W.field_model(vortex_far))
p2 = np.asarray(hist2.pos_ned)
rr2 = np.hypot(p2[:, 0], (-hc - 3 * r0) - p2[:, 2])
print(f"  min r/r0 along the outside-core run  = {rr2.min()/r0:.4f} "
      f"(inside for {int((rr2<r0).sum())} of {len(rr2)} steps)")
og = np.asarray(jax.vmap(lambda pp: W.gust_rates(pp, st0.quat, vortex))(hist.pos_ned))
print(f"  q_gust along the through-core run: min {og[:,1].min():+.5f} "
      f"max {og[:,1].max():+.5f} rad/s; largest step between consecutive "
      f"samples at dt=0.005 s = {np.abs(np.diff(og[:,1])).max():.5f} rad/s")
