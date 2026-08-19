"""Agent C, incidental finding: the strip calibration identity does not hold at
the SHIPPED station count.

This is a MAGNITUDE finding, not a sign finding.  The sign of
`wind.strip_roll_moment` is confirmed correct in c6_gust_rates.py.  What is
wrong is the documented claim of exactness:

  airframe.calibrated_lift_slope: "What it guarantees is that a rigid roll rate
  through the strip integral reproduces the tabulated Clp EXACTLY."
  wind.strip_clp_from_rate:       "this MUST return `Clp * p_hat`."

Both are true in the continuum limit and false at `airframe.N_SPAN = 9`, which
is what the production path (`loads.strip_model` -> `airframe.stations(ac)`)
actually uses.  The repo's own calibration test passes only because its helper
`_b747_and_stations` overrides the count to n_span=2001.
"""
import jax
import jax.numpy as jnp
import numpy as np

from flightsim import airframe, integrate, loads, trim, wind
from flightsim.aircraft import CRUISE, REGISTRY
from flightsim.state import euler_to_quat

print("=" * 78)
print("A. the shipped default vs the count the calibration test uses")
print("=" * 78)
print(f"   airframe.N_SPAN (production)             = {airframe.N_SPAN}")
print(f"   test_wind._b747_and_stations default     = 2001")
print()

ac = REGISTRY["boeing747"]
p_hat = 0.01
print(f"   {'n_span':>8s} {'strip Cl/p_hat':>16s} {'sourced Clp':>13s} {'error':>10s}")
for n in (9, 11, 21, 51, 101, 501, 2001, 20001):
    st = airframe.stations(ac, n_span=n)
    got = float(wind.strip_clp_from_rate(ac, st, jnp.array(p_hat))) / p_hat
    err = 100.0 * (got / float(ac.Clp) - 1.0)
    mark = "  <-- SHIPPED" if n == airframe.N_SPAN else (
        "  <-- test uses this" if n == 2001 else "")
    print(f"   {n:8d} {got:16.6f} {float(ac.Clp):13.6f} {err:9.3f}%{mark}")

print()
print("   The deficit is a pure quadrature error and is IDENTICAL for all four")
print("   aircraft (ratio 0.826435 at n=9), because it depends only on the")
print("   elliptic chord shape and the station count, not on the airframe:")
for name in REGISTRY:
    a = REGISTRY[name]
    st = airframe.stations(a, n_span=9)
    got = float(wind.strip_clp_from_rate(a, st, jnp.array(p_hat))) / p_hat
    print(f"     {name:20s} ratio = {got / float(a.Clp):.6f}")

print()
print("   Other loading shapes at n_span = 9:")
for shape in airframe.LOADING_SHAPES:
    with airframe.loading_shape(shape):
        st = airframe.stations(ac, n_span=9)
        got = float(wind.strip_clp_from_rate(ac, st, jnp.array(p_hat))) / p_hat
    print(f"     {shape:10s} ratio = {got / float(ac.Clp):.6f}  "
          f"({100 * (got / float(ac.Clp) - 1):+.3f}%)")

print()
print("=" * 78)
print("B. does it reach the PRODUCTION path?  (loads.strip_model, no override)")
print("=" * 78)
# A spanwise-linear vertical gust: the strip path and the equivalent-rate path
# should agree, because a linear gradient is exactly what an equivalent rate
# represents.  Any gap is the quadrature deficit reaching flight.
V = float(CRUISE["boeing747"]["airspeed"])
h = float(CRUISE["boeing747"]["altitude"])
x, _ = trim.trim(jnp.array(V), jnp.array(h), ac)
state = trim.trimmed_state(x[0], jnp.array(V), jnp.array(h))
controls = trim.trimmed_controls(x[1], x[2])

GRAD = 0.05  # 1/s : w_g = GRAD * y (NED down component varying with East)


def field(pos_ned):
    return jnp.array([0.0, 0.0, GRAD * (pos_ned[1] - state.pos_ned[1])])


model = loads.strip_model(field, ac)          # uses airframe.stations(ac) -> N_SPAN
inc = model(state)
strip_Cl = float(inc.Cl)

# The equivalent-rate answer the point model would apply for the same field.
p_gust = float(wind.gust_rates(state.pos_ned, state.quat, field)[0])
airspeed = float(jnp.linalg.norm(state.vel_body))
p_hat_eq = -p_gust * float(ac.b) / (2.0 * airspeed)
rate_Cl = float(ac.Clp) * p_hat_eq

print(f"   linear gust dw/dy = {GRAD} 1/s")
print(f"   gust_rates p_gust           = {p_gust:+.6f} rad/s")
print(f"   equivalent-rate Cl          = {rate_Cl:+.8f}   (Clp * p_hat)")
print(f"   strip_model Cl (N_SPAN={airframe.N_SPAN})    = {strip_Cl:+.8f}")
print(f"   ratio strip/rate            = {strip_Cl / rate_Cl:.6f}  "
       f"({100 * (strip_Cl / rate_Cl - 1):+.3f}%)")
print(f"   SIGN AGREES (both {'negative' if rate_Cl < 0 else 'positive'}); "
      f"only the MAGNITUDE is short.")

st2001 = airframe.stations(ac, n_span=2001)
strip_2001 = float(loads.strip_increment(state, field, ac, st2001).Cl)
print(f"   strip with n_span=2001      = {strip_2001:+.8f}  "
      f"ratio {strip_2001 / rate_Cl:.6f}")

print()
print("=" * 78)
print("C. how much does it move a flown trajectory?")
print("=" * 78)
sim = integrate.init_sim(state, jax.random.PRNGKey(0))
wm = wind.field_model(field)
dt, steps = jnp.array(0.02), 500
_, tr_point = integrate.rollout(sim, controls, dt, ac, steps, wind_model=wm)
_, tr_9 = integrate.rollout(sim, controls, dt, ac, steps, wind_model=wm,
                            load_model=loads.strip_model(field, ac))
_, tr_2001 = integrate.rollout(sim, controls, dt, ac, steps, wind_model=wm,
                               load_model=lambda s: loads.strip_increment(
                                   s, field, ac, st2001))
from flightsim.state import quat_to_euler
phi9 = float(quat_to_euler(tr_9.quat[-1])[0])
phi2001 = float(quat_to_euler(tr_2001.quat[-1])[0])
phip = float(quat_to_euler(tr_point.quat[-1])[0])
print(f"   after {steps * float(dt):.0f} s in a dw/dy = {GRAD} 1/s gradient:")
print(f"     point model (equivalent rate only) phi = {np.degrees(phip):+.4f} deg")
print(f"     strip, N_SPAN=9                    phi = {np.degrees(phi9):+.4f} deg")
print(f"     strip, n_span=2001                 phi = {np.degrees(phi2001):+.4f} deg")
print(f"   the 9-station strip understates the 2001-station answer's DEVIATION")
print(f"   from the point model by "
      f"{100 * (1 - (phi9 - phip) / (phi2001 - phip)):.1f}%"
      if abs(phi2001 - phip) > 1e-12 else "   (no separation to measure)")
