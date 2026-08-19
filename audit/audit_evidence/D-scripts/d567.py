"""D5 superposition, D6 along_track_shear, D7 sampled_rates vs gust_rates."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import jax
import jax.numpy as jnp

import flightsim  # noqa: F401
from flightsim import aero, airframe, dynamics, wind
from flightsim.aircraft import CRUISE, REGISTRY
from flightsim.atmosphere import G0, density, speed_of_sound
from flightsim.state import State, quat_to_dcm, euler_to_quat
import wcommon as W

AC = REGISTRY["boeing747"]
V = float(CRUISE["boeing747"]["airspeed"])
HC = W.H_CRUISE
Q0 = jnp.array([1.0, 0.0, 0.0, 0.0])
KEY = jax.random.PRNGKey(7)

print("=" * 78)
print("D5  SUPERPOSITION")
print("=" * 78)
combo = wind.superpose(W.vortex_field, W.updraft_field, W.lee_field)
manual = lambda p: W.vortex_field(p) + W.updraft_field(p) + W.lee_field(p)  # noqa: E731

pos = W.sample_positions(KEY, 500, "vortex")
a = np.asarray(jax.vmap(combo)(pos))
b = np.asarray(jax.vmap(manual)(pos))
print(f"superpose vs a hand-written sum, 500 pts:  max |diff| = {np.abs(a-b).max():.3e}"
      f"   bit-identical = {np.array_equal(a, b)}")

# field_model omega_gust: sum of parts vs the composed field
quats = W.random_quats(jax.random.split(KEY)[1], 500)


def og(field, p, q):
    m = wind.field_model(field)
    return m(wind.zero_wind_state(), State(pos_ned=p, vel_body=jnp.zeros(3), quat=q,
                                           omega=jnp.zeros(3)), KEY, jnp.array(0.01))[1]


og_c = np.asarray(jax.vmap(lambda p, q: og(combo, p, q))(pos, quats))
og_s = sum(np.asarray(jax.vmap(lambda p, q, f=f: og(f, p, q))(pos, quats))
           for f in (W.vortex_field, W.updraft_field, W.lee_field))
den = np.maximum(np.linalg.norm(og_c, axis=1), 1e-12)
print(f"omega_gust(superposed field) vs sum of omega_gust:  "
      f"max rel = {(np.linalg.norm(og_c-og_s, axis=1)/den).max():.3e}"
      f"   bit-identical = {np.array_equal(og_c, og_s)}")

# wind_ned channel
wc = np.asarray(jax.vmap(combo)(pos))
ws = sum(np.asarray(jax.vmap(f)(pos)) for f in (W.vortex_field, W.updraft_field, W.lee_field))
print(f"wind_ned(superposed)     vs sum of wind_ned:      "
      f"max |diff| = {np.abs(wc-ws).max():.3e}   bit-identical = {np.array_equal(wc, ws)}")

# strip_roll_moment linearity
st = airframe.stations(AC)
srm = lambda f, p, q: wind.strip_roll_moment(p, q, f, AC, st, jnp.array(V))  # noqa: E731
sc = np.asarray(jax.vmap(lambda p, q: srm(combo, p, q))(pos[:100], quats[:100]))
ss = sum(np.asarray(jax.vmap(lambda p, q, f=f: srm(f, p, q))(pos[:100], quats[:100]))
         for f in (W.vortex_field, W.updraft_field, W.lee_field))
print(f"strip_roll_moment(superposed) vs sum:              "
      f"max |diff| = {np.abs(sc-ss).max():.3e}  (Cl scale {np.abs(sc).max():.3e})")

print("\ndegenerate calls:")
try:
    print("  superpose() with no fields ->", wind.superpose()(jnp.zeros(3)))
except Exception as e:  # noqa: BLE001
    print(f"  superpose() with no fields -> {type(e).__name__}: {e}")
print("  superpose(one field) exact:",
      np.array_equal(np.asarray(wind.superpose(W.lee_field)(pos[0])),
                     np.asarray(W.lee_field(pos[0]))))

print("\nWHERE SUPERPOSITION IS NOT LEGITIMATE")
print("(a) the microburst's ground boundary condition")
mb_plus_vortex = wind.superpose(W.mb_field, lambda p: jnp.array([0.0, 0.0, -3.0]))
for r in [0.0, 500.0, 1121.0, 3000.0]:
    p = jnp.array([r, 0.0, 0.0])
    print(f"    r={r:7.1f} m, z=0:  microburst w_D = {float(W.mb_field(p)[2]):+.4f}"
          f"   with a 3 m/s updraft added = {float(mb_plus_vortex(p)[2]):+.4f} m/s"
          f"  <- through the ground")

print("(b) the AERODYNAMIC response is not linear in the wind, even though the")
print("    fields are. force(w1+w2) - force(0) vs [force(w1)-force(0)] + [force(w2)-force(0)]:")
rho = density(jnp.array(HC))
a_s = speed_of_sound(jnp.array(HC))
from flightsim import trim
x, _ = trim.trim(jnp.array(V), jnp.array(HC), AC)
ctrl = trim.trimmed_controls(x[1], x[2])
vb = trim.trimmed_state(jnp.array(float(x[0])), jnp.array(V), jnp.array(HC)).vel_body


def fm(w):
    vr = dynamics.relative_velocity(vb, Q0, jnp.asarray(w))
    return aero.aero_forces_moments(vr, jnp.zeros(3), ctrl, AC, rho, a_s)[0]


f0 = np.asarray(fm(jnp.zeros(3)))
for lab, w1, w2 in [
    ("vortex core peak + lee wave crest", [0.0, 0.0, -25.908], [0.0, 0.0, 6.0]),
    ("updraft peak + lee wave crest", [0.0, 0.0, -24.384], [0.0, 0.0, 6.0]),
    ("two 10 m/s vertical gusts", [0.0, 0.0, -10.0], [0.0, 0.0, -10.0]),
]:
    fa = np.asarray(fm(jnp.array(w1))) - f0
    fb = np.asarray(fm(jnp.array(w2))) - f0
    fab = np.asarray(fm(jnp.array(w1) + jnp.array(w2))) - f0
    err = np.linalg.norm(fab - (fa + fb)) / max(np.linalg.norm(fab), 1e-30)
    print(f"    {lab:36s} rel non-additivity = {err:.4e}"
          f"   ({np.linalg.norm(fab-(fa+fb)):.1f} N of {np.linalg.norm(fab):.1f} N)")

print()
print("=" * 78)
print("D6  along_track_shear")
print("=" * 78)
print("(a) SIGN CONVENTION, end to end")
shear = 0.01  # 1/s: tailwind growing along +north


def lin_tail(p):
    return jnp.array([shear * p[0], 0.0, 0.0])


for lab, vel in [("flying north into a growing tailwind", [V, 0.0, 0.0]),
                 ("flying south (same field, now a headwind that weakens)", [-V, 0.0, 0.0]),
                 ("flying north, descending", [V, 0.0, 5.0]),
                 ("flying east across the gradient", [0.0, V, 0.0])]:
    p = jnp.array([1000.0, 0.0, -HC])
    v = jnp.array(vel)
    s = float(wind.along_track_shear(p, v, lin_tail))
    ux = float(jnp.dot(lin_tail(p)[:2], v[:2] / jnp.linalg.norm(v[:2])))
    print(f"    {lab:52s} U_x = {ux:+8.3f} m/s   dU_x/dt = {s:+9.5f} m/s^2"
          f"   F = {float(dynamics.f_factor(jnp.array(s), jnp.array(0.0), jnp.array(V))):+.5f}")
print("    Proctor Eq.(3): F = U_x_dot/g - w/V, positive = hazardous.")
print("    A tailwind that GROWS ahead of you is an energy loss -> F > 0. Confirmed.")
print("    Pure vertical: w_up = +6 m/s (updraft) ->",
      f"F = {float(dynamics.f_factor(jnp.array(0.0), jnp.array(6.0), jnp.array(V))):+.5f}"
      "  (an updraft is favourable, F < 0). Confirmed.")

print("\n(b) CURVING FLIGHT PATH: the constant-heading approximation")
print("    True  dU_x/dt = (grad U . v) + u_h . dh_hat/dt")
print("    Code  dU_x/dt = (grad U . v)  only.")
print("    Measured by finite-differencing U_x along a prescribed circular ground")
print("    track (a coordinated turn) through each field.\n")


def curved_error(field, p0, turn_rate, speed=V, climb=0.0, dt=1e-4):
    """Exact dU_x/dt on a circular track vs what along_track_shear returns."""
    def path(t):
        psi = turn_rate * t
        # constant-speed circular ground track starting northbound
        rad = speed / turn_rate
        n = p0[0] + rad * jnp.sin(psi)
        e = p0[1] + rad * (1 - jnp.cos(psi))
        d = p0[2] + climb * t
        return jnp.array([n, e, d]), jnp.array([speed * jnp.cos(psi),
                                                speed * jnp.sin(psi), climb])

    def U_x(t):
        p, v = path(t)
        h = v[:2] / jnp.linalg.norm(v[:2])
        return jnp.dot(field(p)[:2], h)

    exact = float((U_x(dt) - U_x(-dt)) / (2 * dt))
    p, v = path(0.0)
    code = float(wind.along_track_shear(p, v, field))
    return exact, code


print(f"    {'field / point':44s} {'turn':>8s} {'exact':>11s} {'code':>11s} "
      f"{'abs err':>11s} {'dF':>10s}")
cases = [
    ("Parks vortex, 1 r0 from a core", W.vortex_field,
     [W.R0, 0.0, -HC + 0.5 * W.R0]),
    ("Parks vortex, core edge", W.vortex_field, [W.R0, 0.0, -HC]),
    ("microburst, peak outflow r=1121 z=150", W.mb_field, [1121.0, 0.0, -150.0]),
    ("microburst, r=2000 z=150", W.mb_field, [2000.0, 0.0, -150.0]),
    ("lee wave (no horizontal component at all)", W.lee_field, [6000.0, 0.0, -HC]),
    ("updraft (no horizontal component at all)", W.updraft_field, [2000.0, 0.0, -HC]),
]
for lab, fld, p0 in cases:
    for tr_deg, tr_name in [(3.0, "3 deg/s"), (1.0, "1 deg/s")]:
        tr = np.deg2rad(tr_deg)
        ex, co = curved_error(fld, jnp.array(p0), tr)
        dF = abs(ex - co) / G0
        print(f"    {lab:44s} {tr_name:>8s} {ex:11.5f} {co:11.5f} "
              f"{abs(ex-co):11.3e} {dF:10.3e}")
print("    dF is the F-factor error the omission causes (err/g). FAA alert = 0.1.")

print("\n    scaling of the omitted term: it is  -u_perp * psi_dot  where u_perp is")
print("    the CROSS-track horizontal wind. Direct check:")
for u_perp in [5.0, 10.0, 20.0]:
    for tr_deg in [1.0, 3.0, 10.0]:
        tr = np.deg2rad(tr_deg)
        cross = lambda p, u=u_perp: jnp.array([0.0, u, 0.0])  # noqa: E731
        ex, co = curved_error(cross, jnp.array([0.0, 0.0, -HC]), tr)
        print(f"      u_perp={u_perp:5.1f} m/s, psi_dot={tr_deg:4.1f} deg/s:"
              f"  exact {ex:+9.5f}  code {co:+9.5f}  omitted {ex-co:+9.5f} m/s^2"
              f"  -> dF = {abs(ex-co)/G0:.4f}")

print("\n(c) degenerate ground track (the 1e-9 floor):")
for v in [[1e-3, 0.0, 0.0], [1e-12, 0.0, 0.0], [0.0, 0.0, -5.0], [0.0, 0.0, 0.0]]:
    s = wind.along_track_shear(jnp.array([1121.0, 0.0, -150.0]), jnp.array(v), W.mb_field)
    print(f"    vel_ned = {str(v):22s} -> {float(s):+.6e}"
          f"  {'<<< NaN' if np.isnan(float(s)) else ''}")

print()
print("=" * 78)
print("D7  sampled_rates vs gust_rates")
print("=" * 78)
st = airframe.stations(AC)
print(f"stations: span {np.asarray(st.span).min():.2f}..{np.asarray(st.span).max():.2f} m"
      f" ({len(st.span)} pts), longitudinal {np.asarray(st.longitudinal).min():.2f}"
      f"..{np.asarray(st.longitudinal).max():.2f} m ({len(st.longitudinal)} pts)")

print("\n(a) a field LINEAR across the airframe -- the claimed exact agreement")
kA, kb, kq = jax.random.split(KEY, 3)
worst = 0.0
for i in range(50):
    A = jax.random.normal(jax.random.fold_in(kA, i), (3, 3)) * 1e-3
    bb = jax.random.normal(jax.random.fold_in(kb, i), (3,))
    lin = lambda p, A=A, bb=bb: A @ p + bb  # noqa: E731
    q = W.random_quats(jax.random.fold_in(kq, i), 1)[0]
    p = jnp.array([100.0, 50.0, -HC])
    g = np.asarray(wind.gust_rates(p, q, lin))
    s = np.asarray(wind.sampled_rates(p, q, lin, st))
    worst = max(worst, np.abs(g - s).max() / max(np.abs(g).max(), 1e-30))
print(f"    50 random linear fields x random attitudes:"
      f" worst relative |sampled - tangent| = {worst:.3e}")
print(f"    (also checked bit-identity: {'yes' if worst == 0.0 else 'no, but at round-off'})")

print("\n(b) across the Parks core -- the disagreement the secant is FOR")
print(f"    core radius {W.R0:.2f} m = {2*W.R0/AC.b:.3f} spans"
      f" (747 span {AC.b:.2f} m); tail arm {float(airframe.effective_tail_arm(AC))*AC.c:.2f} m")
print(f"    {'north (m)':>12s} {'r/r0':>8s} {'q tangent':>13s} {'q secant':>13s} "
      f"{'rel diff':>11s} | {'p tangent':>11s} {'p secant':>11s}")
for nn in [-4 * W.R0, -2 * W.R0, -1.2 * W.R0, -W.R0, -0.75 * W.R0, -0.5 * W.R0,
           -0.25 * W.R0, -30.0, -1.0, 0.0, 30.0, 0.5 * W.R0, W.R0, 2 * W.R0]:
    p = jnp.array([nn, 0.0, -HC])
    g = np.asarray(wind.gust_rates(p, Q0, W.vortex_field))
    s = np.asarray(wind.sampled_rates(p, Q0, W.vortex_field, st))
    rel = abs(g[1] - s[1]) / max(abs(g[1]), 1e-30)
    print(f"    {nn:12.2f} {abs(nn)/W.R0:8.3f} {g[1]:13.6e} {s[1]:13.6e} "
          f"{rel:11.3e} | {g[0]:11.3e} {s[0]:11.3e}")

print("\n    worst-case over a dense traverse of the first core:")
ns = np.linspace(-3 * W.R0, 3 * W.R0, 2001)
gg = np.array([np.asarray(wind.gust_rates(jnp.array([n, 0.0, -HC]), Q0, W.vortex_field))
               for n in ns])
sss = np.array([np.asarray(wind.sampled_rates(jnp.array([n, 0.0, -HC]), Q0, W.vortex_field, st))
                for n in ns])
for i, nm in enumerate("pqr"):
    d = np.abs(gg[:, i] - sss[:, i])
    sc = np.abs(gg[:, i]).max()
    if sc > 0:
        j = int(np.argmax(d))
        print(f"      {nm}: peak tangent {sc:.5e}, peak |secant-tangent| {d.max():.5e}"
              f" ({d.max()/sc:.2%} of peak) at north = {ns[j]:.2f} m (r/r0 = {abs(ns[j])/W.R0:.3f})")

print("\n    the same across the OTHER fields, for scale:")
for name, (kind, fld) in W.FIELDS.items():
    pos = W.sample_positions(jax.random.fold_in(KEY, hash(name) % 999), 400, kind)
    qs = W.random_quats(jax.random.fold_in(KEY, 1 + hash(name) % 999), 400)
    g = np.asarray(jax.vmap(lambda p, q: wind.gust_rates(p, q, fld))(pos, qs))
    s = np.asarray(jax.vmap(lambda p, q: wind.sampled_rates(p, q, fld, st))(pos, qs))
    sc = np.linalg.norm(g, axis=1).max()
    d = np.linalg.norm(g - s, axis=1)
    print(f"      {name:34s} peak |tangent| {sc:.4e}  max |secant-tangent| "
          f"{d.max():.4e}  ({d.max()/sc:.2%})  median {np.median(d)/sc:.2%}")
