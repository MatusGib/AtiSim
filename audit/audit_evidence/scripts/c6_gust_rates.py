"""Agent C, item 3: gust-rate signs, and wind.strip_roll_moment's sign.

THE TEST.  Put the air mass into RIGID ROTATION at a known rate Omega about one
body axis at a time.  An aircraft rotating WITH the air at omega = Omega is, by
definition, motionless relative to it, so the aerodynamics must see an effective
rate of exactly zero.  dynamics forms omega_rel = omega - omega_gust, so this
requires omega_gust == Omega, exactly, for each axis SEPARATELY.

Doing all three axes separately is what makes this asymmetric: a convention that
mixed p and r, or flipped one sign of three, passes a test that rotates about
all axes at once with equal rates and fails here.

The aircraft attitude is deliberately non-trivial (all three Euler angles
non-zero) so the body<->NED rotation inside gust_rates is genuinely exercised
rather than being the identity.
"""
import jax
import jax.numpy as jnp
import numpy as np

from flightsim import aero, airframe, dynamics, wind
from flightsim.aircraft import REGISTRY
from flightsim.state import Controls, State, euler_to_quat, quat_to_dcm

fails = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name:64s} {detail}")
    if not ok:
        fails.append(name)


ac = REGISTRY["boeing747"]
PHI, THETA, PSI = 0.23, -0.17, 0.9      # non-trivial attitude
QUAT = euler_to_quat(jnp.array(PHI), jnp.array(THETA), jnp.array(PSI))
DCM = np.asarray(quat_to_dcm(QUAT))     # body -> NED
POS = jnp.array([1234.0, -567.0, -11000.0])
ZERO3 = jnp.zeros(3)
NOCTL = Controls(jnp.array(0.0), jnp.array(0.0), jnp.array(0.0), jnp.array(0.0))


def rigid_rotation_field(omega_body, centre_ned):
    """A wind field in rigid rotation at `omega_body` (BODY axes) about `centre_ned`.

    Built in NED: the angular velocity vector is rotated into NED once, then the
    field is the plain rigid-body formula v = Omega_ned x (r - r0).  Nothing in
    here uses any flightsim function, so it is an independent construction.
    """
    omega_ned = DCM @ np.asarray(omega_body)

    def field(pos_ned):
        return jnp.cross(jnp.array(omega_ned), pos_ned - centre_ned)

    return field


print("=" * 78)
print("3.1 RIGID ROTATION about each BODY axis in turn")
print("=" * 78)
OM = 0.031  # rad/s, a deliberately un-round rate
AXES = {"p (body x, roll)": [OM, 0.0, 0.0],
        "q (body y, pitch)": [0.0, OM, 0.0],
        "r (body z, yaw)": [0.0, 0.0, OM]}

for label, omega_body in AXES.items():
    field = rigid_rotation_field(omega_body, POS)
    got = np.asarray(wind.gust_rates(POS, QUAT, field))
    want = np.array(omega_body)
    print(f"   air in rigid rotation about {label:20s}")
    print(f"      gust_rates -> {got}   want {want}")
    check(f"   gust_rates recovers the rigid rate, {label}",
          np.abs(got - want).max() < 1e-9, f"err={np.abs(got - want).max():.2e}")
    # And the OTHER two components must be exactly zero -- no cross-talk.
    idx = int(np.argmax(np.abs(want)))
    others = np.delete(got, idx)
    check(f"   no cross-axis leakage, {label}", np.abs(others).max() < 1e-9,
          f"max other={np.abs(others).max():.2e}")

print()
print("=" * 78)
print("3.2 THE PHYSICAL STATEMENT: rotating WITH the air must be felt as still air")
print("=" * 78)
V = 235.0
vel_body = jnp.array([V, 0.0, 0.0])
for label, omega_body in AXES.items():
    field = rigid_rotation_field(omega_body, POS)
    st = State(pos_ned=POS, vel_body=vel_body, quat=QUAT,
               omega=jnp.array(omega_body))
    wind_ned = field(POS)          # zero at the centre by construction
    omega_gust = wind.gust_rates(POS, QUAT, field)
    omega_rel = np.asarray(st.omega) - np.asarray(omega_gust)
    # Compare the moments with those in perfectly still air at zero rate.
    M_air = np.asarray(aero.aero_forces_moments(
        dynamics.relative_velocity(st.vel_body, st.quat, wind_ned),
        jnp.array(omega_rel), NOCTL, ac, jnp.array(0.3), jnp.array(299.0))[1])
    M_still = np.asarray(aero.aero_forces_moments(
        vel_body, ZERO3, NOCTL, ac, jnp.array(0.3), jnp.array(299.0))[1])
    print(f"   {label:20s} omega_rel = {omega_rel}   |M - M_still| = "
          f"{np.abs(M_air - M_still).max():.3e} N.m")
    check(f"   rotating with the air feels no rate, {label}",
          np.abs(omega_rel).max() < 1e-9)
    check(f"   ...and produces the still-air moment, {label}",
          np.abs(M_air - M_still).max() < 1e-6)

print()
print("=" * 78)
print("3.3 The three claimed gradient relations, one gradient at a time")
print("=" * 78)
# Body-axis linear gust fields with exactly ONE non-zero gradient each.
G = 0.017  # 1/s


def body_linear_field(mat):
    """Gust whose BODY components are mat @ offset_body, offset from POS."""
    mat = np.asarray(mat, dtype=float)

    def field(pos_ned):
        off_body = jnp.array(DCM.T) @ (pos_ned - POS)
        return jnp.array(DCM) @ (jnp.array(mat) @ off_body)

    return field


CASES = {
    "dw_g/dy = +G  ->  p_gust = +G": (np.array([[0, 0, 0], [0, 0, 0], [0, G, 0]]),
                                      [G, 0.0, 0.0]),
    "dw_g/dx = +G  ->  q_gust = -G": (np.array([[0, 0, 0], [0, 0, 0], [G, 0, 0]]),
                                      [0.0, -G, 0.0]),
    "dv_g/dx = +G  ->  r_gust = +G": (np.array([[0, 0, 0], [G, 0, 0], [0, 0, 0]]),
                                      [0.0, 0.0, G]),
}
for label, (mat, want) in CASES.items():
    field = body_linear_field(mat)
    got = np.asarray(wind.gust_rates(POS, QUAT, field))
    print(f"   {label:34s} got {got}")
    check(f"   {label}", np.abs(got - np.array(want)).max() < 1e-9,
          f"err={np.abs(got - np.array(want)).max():.2e}")

print()
print("=" * 78)
print("3.4 sampled_rates agrees with gust_rates on a LINEAR field (same sign)")
print("=" * 78)
st_ac = airframe.stations(ac)
for label, omega_body in AXES.items():
    field = rigid_rotation_field(omega_body, POS)
    g = np.asarray(wind.gust_rates(POS, QUAT, field))
    s = np.asarray(wind.sampled_rates(POS, QUAT, field, st_ac))
    print(f"   {label:20s} gust_rates={g}  sampled_rates={s}")
    check(f"   sampled_rates == gust_rates for {label}",
          np.abs(g - s).max() < 1e-9, f"err={np.abs(g - s).max():.2e}")

print()
print("=" * 78)
print("3.5 strip_roll_moment sign: -w_g/V, not +w_g/V")
print("=" * 78)
# (a) The calibration identity: a rigid roll rate through the strip integral
#     must return Clp * p_hat.
#
#     *** THESE TWO CHECKS FAIL, DELIBERATELY LEFT FAILING. ***
#     They fail at the SHIPPED airframe.N_SPAN = 9 by -17.4%, which is a
#     quadrature error, not a sign error. Fully characterised in
#     c9_strip_stations.py. The tolerance here is NOT relaxed to hide it.
for p_hat in (0.05, -0.03):
    got = float(wind.strip_clp_from_rate(ac, st_ac, jnp.array(p_hat)))
    want = float(ac.Clp) * p_hat
    print(f"   strip_clp_from_rate(p_hat={p_hat:+.3f}) = {got:+.8f}  "
          f"Clp*p_hat = {want:+.8f}")
    check(f"   rigid roll rate reproduces Clp at p_hat={p_hat:+.2f}",
          abs(got - want) < 1e-6 * max(1e-9, abs(want)), f"err={got - want:.2e}")

# (b) THE SIGN TEST.  Build the gust field that a rigid roll rate p is
#     equivalent to, and check the strip integral returns the SAME moment.
#     A wing rolling right (p>0) has its right wing moving DOWN, which is
#     equivalent to the air at y>0 moving UP, i.e. w_g = -p*y.
P = 0.04
V_AS = 235.0
p_hat = P * float(ac.b) / (2.0 * V_AS)


def equivalent_gust_field(pos_ned):
    """w_g(y) = -P*y in BODY axes; no other component."""
    off_body = jnp.array(DCM.T) @ (pos_ned - POS)
    gust_body = jnp.array([0.0, 0.0, -P * off_body[1]])
    return jnp.array(DCM) @ gust_body


strip = float(wind.strip_roll_moment(POS, QUAT, equivalent_gust_field, ac,
                                     st_ac, jnp.array(V_AS)))
rate = float(wind.strip_clp_from_rate(ac, st_ac, jnp.array(p_hat)))
print(f"   equivalent gust field w_g = -{P}*y  ->  Cl = {strip:+.8f}")
print(f"   rigid roll rate p = {P} (p_hat={p_hat:.5f})  ->  Cl = {rate:+.8f}")
check("   a gust field equivalent to a roll rate gives the SAME Cl (sign and "
      "magnitude)", abs(strip - rate) < 1e-9, f"diff={strip - rate:.2e}")
# With the OPPOSITE sign convention the two would differ by 2x the value:
print(f"   (had the convention been +w_g/V the two would differ by "
      f"{2 * abs(rate):.6f}; observed difference {abs(strip - rate):.2e})")

# (c) Physical direction: air going UP on the right wing must roll the aircraft
#     to the LEFT (extra lift to starboard => port roll).
def up_on_right(pos_ned):
    off_body = jnp.array(DCM.T) @ (pos_ned - POS)
    return jnp.array(DCM) @ jnp.array([0.0, 0.0, -1.0 * off_body[1]])


cl_up_right = float(wind.strip_roll_moment(POS, QUAT, up_on_right, ac, st_ac,
                                           jnp.array(V_AS)))
print(f"   air rising on the RIGHT wing (w_g = -y) -> Cl = {cl_up_right:+.8f}")
check("   updraught on the right wing rolls the aircraft LEFT (Cl < 0)",
      cl_up_right < 0)

# (d) And a uniform vertical gust (no gradient) must produce EXACTLY zero roll.
def uniform_down(pos_ned):
    return jnp.array(DCM) @ jnp.array([0.0, 0.0, 5.0])


cl_uniform = float(wind.strip_roll_moment(POS, QUAT, uniform_down, ac, st_ac,
                                          jnp.array(V_AS)))
print(f"   uniform 5 m/s body-z gust -> Cl = {cl_uniform:+.3e}")
check("   a uniform vertical gust produces no rolling moment",
      abs(cl_uniform) < 1e-12, f"Cl={cl_uniform:.2e}")

print()
print("FAILURES:", fails if fails else "none")
