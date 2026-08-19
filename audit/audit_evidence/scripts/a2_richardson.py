"""A2: Richardson extrapolation on FULL SIMULATION OUTPUT (747 cruise, 4 s).

Three fields:  (a) still air  (b) smooth lee-wave field  (c) Rankine vortex core.
Refinement ratio 2 throughout, so p_obs = log2((f1-f2)/(f2-f3)).
"""
import numpy as np
import jax, jax.numpy as jnp
import flightsim
from flightsim import integrate, trim, wind
from flightsim.aircraft import CRUISE, REGISTRY

np.set_printoptions(precision=17)

AC = REGISTRY["boeing747"]
V = CRUISE["boeing747"]["airspeed"]
H = CRUISE["boeing747"]["altitude"]
T_END = 4.0

x_trim, r_trim = trim.trim(jnp.array(V), jnp.array(H), AC)
print(f"747 cruise trim: alpha={float(x_trim[0]):.9f} rad  de={float(x_trim[1]):.9f} "
      f"thr={float(x_trim[2]):.9f}  |res|={float(jnp.linalg.norm(r_trim)):.3e}")
print(f"V={V:.6f} m/s  H={H:.6f} m")


def run(dt, wind_model=None, start_north=0.0, d_elevator=0.02):
    state = trim.trimmed_state(x_trim[0], jnp.array(V), jnp.array(H))
    state = state._replace(pos_ned=state.pos_ned.at[0].set(jnp.asarray(start_north, float)))
    controls = trim.trimmed_controls(x_trim[1] + d_elevator, x_trim[2])
    model = wind.zero_wind if wind_model is None else wind_model
    sim = integrate.init_sim(state, jax.random.PRNGKey(0))
    n = int(round(T_END / dt))
    assert abs(n * dt - T_END) < 1e-12, (dt, n)
    final, _ = integrate.rollout(sim, controls, jnp.array(dt), AC, n, wind_model=model)
    s = final.state
    return np.concatenate([np.asarray(s.pos_ned), np.asarray(s.vel_body),
                           np.asarray(s.quat), np.asarray(s.omega)])


LABELS = ["pN", "pE", "pD", "u", "v", "w", "q0", "q1", "q2", "q3", "p", "q", "r"]


def richardson(name, wind_model=None, start_north=0.0, d_elevator=0.02,
               dts=(1/4, 1/8, 1/16, 1/32, 1/64, 1/128, 1/256, 1/512, 1/1024)):
    print("\n" + "=" * 78)
    print(f"### {name}")
    print("=" * 78)
    dts = [float(d) for d in dts]
    F = {d: run(d, wind_model, start_north, d_elevator) for d in dts}

    # --- (i) Richardson triples on the position norm and per-component -------
    print("\n-- Richardson triples (h, h/2, h/4), ratio r=2 --")
    print(f"{'h':>10} {'comp':>5} {'f_h':>22} {'f_h/2':>22} {'f_h/4':>22} "
          f"{'p_obs':>9} {'f_exact(Rich)':>22}")
    for i in range(len(dts) - 2):
        h1, h2, h3 = dts[i], dts[i + 1], dts[i + 2]
        for c in (0, 1, 2):  # pN pE pD
            f1, f2, f3 = F[h1][c], F[h2][c], F[h3][c]
            d12, d23 = f1 - f2, f2 - f3
            if d23 == 0.0:
                p = float("nan"); fex = f3
            else:
                ratio = d12 / d23
                p = np.log(abs(ratio)) / np.log(2.0) if ratio > 0 else float("nan")
                fex = f3 + (f3 - f2) / (2.0 ** p - 1.0) if np.isfinite(p) and p > 0 else float("nan")
            print(f"{h1:>10.6g} {LABELS[c]:>5} {f1:>22.14f} {f2:>22.14f} {f3:>22.14f} "
                  f"{p:>9.4f} {fex:>22.14f}")
        print()

    # --- (ii) error against the finest run, and fitted order -----------------
    ref = F[dts[-1]]
    print(f"-- |pos(dt) - pos(dt_ref={dts[-1]:g})| , 3-vector norm --")
    print(f"{'dt':>10} {'err_pos [m]':>16} {'pairwise order':>16} {'err_vel [m/s]':>16} "
          f"{'err_quat':>14} {'err_omega':>14}")
    prev = None
    for d in dts[:-1]:
        ep = float(np.linalg.norm(F[d][0:3] - ref[0:3]))
        ev = float(np.linalg.norm(F[d][3:6] - ref[3:6]))
        eq = float(np.linalg.norm(F[d][6:10] - ref[6:10]))
        eo = float(np.linalg.norm(F[d][10:13] - ref[10:13]))
        po = "" if prev is None else f"{np.log(prev[0]/ep)/np.log(prev[1]/d):.4f}"
        print(f"{d:>10.6g} {ep:>16.6e} {po:>16} {ev:>16.6e} {eq:>14.6e} {eo:>14.6e}")
        prev = (ep, d)
    return F


# ---------------------------------------------------------------- (a) still air
F_still = richardson("(a) STILL AIR")

# ------------------------------------------------------- (b) smooth spatial wind
wave = wind.LeeWave(w0=jnp.array(25.0), wavelength=jnp.array(1200.0), north=jnp.array(0.0))
smooth = wind.field_model(lambda p: wind.lee_wave_wind(p, wave))
F_smooth = richardson("(b) SMOOTH SPATIAL FIELD (LeeWave w0=25 m/s, lambda=1200 m)",
                      wind_model=smooth)

# ------------------------------------------------------ (c) Rankine vortex core
case = wind.PARKS_CASES["hannibal"]
r0 = case["r0"]
array = wind.VortexArray(north=jnp.array([0.0]), down=jnp.array([-H]),
                         r0=jnp.array(r0), v0=jnp.array(case["v0"]))
vortex = wind.field_model(lambda p: wind.vortex_wind(p, array))
print(f"\n[vortex] r0 = {r0:.4f} m, v0 = {case['v0']:.4f} m/s, "
      f"start_north = {-2*r0:.4f} m, ground speed ~{V:.1f} m/s "
      f"-> core edge crossed at t ~ {(2*r0 - r0)/V:.4f} s and {(2*r0 + r0)/V:.4f} s")
F_vortex = richardson("(c) RANKINE VORTEX CORE CROSSING",
                      wind_model=vortex, start_north=-2.0 * r0, d_elevator=0.0)
