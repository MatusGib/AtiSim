"""ATTACK 6c: isolate the order loss. Same rollout with (a) no wind at all,
(b) a CONSTANT uniform wind (zero gradient, so the hold is exact),
(c) a smooth spatially varying field.

Proper Richardson: error(dt) estimated as |x(dt) - x(dt/2)|, so the reference
does not contaminate the finest entry.
"""
from common import *  # noqa

from flightsim import wind as W

DTS = [0.04, 0.02, 0.01, 0.005, 0.0025, 0.00125, 0.000625]
SEC = 20.0


def final(ac, st0, c, dt, model):
    n = int(round(SEC / dt))
    kw = {} if model is None else {"wind_model": model}
    _, hist = run(st0, c, ac, dt, n, **kw)
    return jax.tree.map(lambda a: np.asarray(a)[-1], hist)


def diff(a, b):
    return max(np.abs(np.asarray(getattr(a, f)) - np.asarray(getattr(b, f))).max()
               for f in ("pos_ned", "vel_body", "quat", "omega"))


def table(label, model, st0=None, perturb=True):
    ac, st, x, r = trim_state("boeing747")
    hc = CRUISE["boeing747"]["altitude"]
    s0 = st0 if st0 is not None else st._replace(
        pos_ned=jnp.array([-1500.0, 0.0, -hc]),
        omega=jnp.array([0.02, 0.03, -0.01]) if perturb else st.omega)
    c = ctrl(de=float(x[1]), thr=float(x[2]))
    fins = [final(ac, s0, c, dt, model) for dt in DTS]
    print(f"\n  {label}")
    print(f"    {'dt':>10s} {'|x(dt)-x(dt/2)|':>17s} {'ratio':>8s} {'order':>7s}")
    prev = None
    for i in range(len(DTS) - 1):
        e = diff(fins[i], fins[i + 1])
        if prev is None or e == 0 or prev == 0:
            print(f"    {DTS[i]:10.6f} {e:17.6e}")
        else:
            print(f"    {DTS[i]:10.6f} {e:17.6e} {prev/e:8.2f} "
                  f"{np.log2(prev/e):7.2f}")
        prev = e


hc = CRUISE["boeing747"]["altitude"]
print("=== isolating the RK4 order loss (747, 20 s, perturbed rates) ===")
table("(a) NO wind model at all (zero_wind)", None)


def const_wind(p):
    return jnp.array([12.0, 5.0, -2.0])


table("(b) CONSTANT uniform wind (12,5,-2) m/s -- zero gradient",
      W.field_model(const_wind))

wave = W.LeeWave(w0=jnp.array(6.0), wavelength=jnp.array(W.LEE_WAVE_WAVELENGTH),
                 north=jnp.array(0.0))
table("(c) lee wave -- smooth, spatially varying",
      W.field_model(lambda p: W.lee_wave_wind(p, wave)))

col = W.UpdraftColumn(north=jnp.array(0.0), east=jnp.array(0.0),
                      w0=jnp.array(W.UPDRAFT_W0), radius=jnp.array(2400.0),
                      sharpness=jnp.array(2.0))
table("(d) updraft column -- smooth, spatially varying",
      W.field_model(lambda p: W.updraft_wind(p, col)))

case = W.PARKS_CASES["hannibal"]
r0 = float(case["r0"])
arr = W.VortexArray(north=jnp.array([0.0]), down=jnp.array([-hc]),
                    r0=jnp.array(r0), v0=jnp.array(case["v0"]))
table("(e) Parks vortex THROUGH the core -- gradient discontinuity",
      W.field_model(lambda p: W.vortex_wind(p, arr)))
