# quick checks to make sure the simulator isn't lying to us.
# basic idea: work out what the answer SHOULD be by hand first, then
# compare that to what the model spits out. if the "expected" number
# ever comes from running the model itself, the test is worthless -
# don't do that.
#
# run it with:
#   PYTHONPATH=. python scripts/sanity.py

import jax.numpy as jnp
import numpy as np

import atisim  # noqa: F401 -- has to be imported first, turns on float64
from atisim import trim, verification
from atisim.aero import aero_forces_moments, air_data, coefficients
from atisim.aircraft import CRUISE, REGISTRY, inertia_tensor
from atisim.atmosphere import G0, density, speed_of_sound
from atisim.dynamics import derivatives, load_factor, relative_velocity
from atisim.state import State, euler_to_quat, quat_to_dcm

results = []
n = 0


def check(name, expected, got, tol=0.0):
    global n
    n += 1
    ok = abs(got - expected) <= tol if tol else got == expected
    results.append(ok)
    print(f"[{n}] {name}")
    print(f"    expected {expected}")
    print(f"    got      {got}")
    print("    PASS" if ok else "    FAIL <-----")
    print()


# ----- boeing 747 at cruise, trimmed -----

ac = REGISTRY["boeing747"]
v = CRUISE["boeing747"]["airspeed"]
h = CRUISE["boeing747"]["altitude"]
rho = density(jnp.array(h))
a_sound = speed_of_sound(jnp.array(h))

x_trim, res = trim.trim(jnp.array(v), jnp.array(h), ac)
alpha = float(x_trim[0])
trim_controls = trim.trimmed_controls(x_trim[1], x_trim[2])
trim_state = trim.trimmed_state(x_trim[0], jnp.array(v), jnp.array(h))

print(f"747 at {v:.1f} m/s, {h:.0f} m")
print(f"trim alpha = {np.degrees(alpha):.3f} deg (residual {float(jnp.linalg.norm(res)):.2e})")
print()

# =====================================================================
# easy stuff first - answer should just be 0, don't even need to think
# about the aero model for these
# =====================================================================

zw = relative_velocity(trim_state.vel_body, trim_state.quat, jnp.zeros(3))
check("zero wind doesn't touch the relative velocity",
      0.0, float(jnp.abs(zw - trim_state.vel_body).max()))

# free fall - kill the aero model completely, gravity's the only thing
# left, so it should fall straight down at g and nothing else
naked = verification.without_aerodynamics(ac)
d_free = derivatives(
    trim_state._replace(omega=jnp.zeros(3)),
    trim.trimmed_controls(jnp.array(0.0), jnp.array(1.0)),  # no thrust either way this is called
    naked, jnp.zeros(3), jnp.zeros(3),
)
accel_ned = quat_to_dcm(trim_state.quat) @ d_free.vel_body
check("no aero + no thrust = straight down at g",
      G0, float(accel_ned[2]), tol=1e-12)
check("...and nothing sideways or forwards",
      0.0, float(jnp.abs(accel_ned[:2]).max()), tol=1e-12)

# roll test - zero the roll coefficients, full aileron should do
# nothing... except the 747's inertia tensor has an Ixz term, so a
# yawing moment can sneak into roll anyway. only actually zero once
# the inertia is also made diagonal, which is the first case below.
no_roll = dict(Clb=0.0, Clp=0.0, Clr=0.0, Clda=0.0, Cldr=0.0)
slip_state = trim_state._replace(
    vel_body=trim_state.vel_body + jnp.array([0.0, 12.0, 0.0]),  # roughly 3 deg sideslip
    omega=jnp.zeros(3),
)
big_aileron = trim.trimmed_controls(x_trim[1], x_trim[2])._replace(
    aileron=jnp.array(0.3), rudder=jnp.array(0.2)
)

diag_ac = ac._replace(
    **{k: jnp.array(vv) for k, vv in no_roll.items()},
    inertia=inertia_tensor(4.678e7, 3.312e7, 6.722e7, 0.0),
    inertia_inv=jnp.linalg.inv(inertia_tensor(4.678e7, 3.312e7, 6.722e7, 0.0)),
)
d_diag = derivatives(slip_state, big_aileron, diag_ac, jnp.zeros(3), jnp.zeros(3))
check("Cl=0 AND diagonal inertia -> aileron does nothing to roll rate",
      0.0, float(d_diag.omega[0]), tol=1e-18)

# same thing but with the REAL (coupled) inertia this time. not a
# pass/fail, just showing it's not zero anymore and that this is
# actually correct, not a bug
real_ac = ac._replace(**{k: jnp.array(vv) for k, vv in no_roll.items()})
d_real = derivatives(slip_state, big_aileron, real_ac, jnp.zeros(3), jnp.zeros(3))
print("same case but with the real inertia (Ixz != 0):")
print(f"  p_dot = {float(d_real.omega[0]):.6e}")
print("  not zero, and that's fine - the yaw moment leaks into roll through Ixz")
print()

# pitch is a cleaner version of this test because Iyy doesn't couple
# to anything else, no Ixy or Iyz on this aircraft
no_pitch_ac = ac._replace(
    Cm0=jnp.array(0.0), Cma=jnp.array(0.0), Cmq=jnp.array(0.0), Cmde=jnp.array(0.0),
)
d_nopitch = derivatives(
    slip_state,
    trim.trimmed_controls(jnp.array(0.4), x_trim[2]),
    no_pitch_ac, jnp.zeros(3), jnp.zeros(3),
)
check("Cm=0 -> even a big elevator input gives zero pitch accel",
      0.0, float(d_nopitch.omega[1]), tol=1e-18)

# =====================================================================
# one variable at a time, just checking the sign / direction is right
# =====================================================================

level_q = euler_to_quat(jnp.array(0.0), jnp.array(0.0), jnp.array(0.0))
headwind = relative_velocity(jnp.array([v, 0.0, 0.0]), level_q, jnp.array([-30.0, 0.0, 0.0]))
check("30 m/s headwind should add 30 m/s to the airspeed",
      v + 30.0, float(air_data(headwind)[0]), tol=1e-9)

check("w equal to u should give alpha of exactly 45 deg",
      45.0, float(np.degrees(air_data(jnp.array([100.0, 0.0, 100.0]))[1])), tol=1e-9)


def cm_at(alpha_rad):
    vel = v * jnp.array([jnp.cos(alpha_rad), 0.0, jnp.sin(alpha_rad)])
    return float(coefficients(vel, jnp.zeros(3), trim_controls, ac, a_sound)[4])


dcm = cm_at(jnp.array(alpha + 0.05)) - cm_at(jnp.array(alpha))
print(f"static stability: bumping alpha up by 2.9 deg changes Cm by {dcm:+.5f}")
print(f"  should be negative (nose-down, restoring). Cma = {float(ac.Cma):+.3f}")
print("  PASS" if dcm < 0 else "  FAIL")
print()

cm_pitching = float(coefficients(trim_state.vel_body, jnp.array([0.0, 0.1, 0.0]), trim_controls, ac, a_sound)[4])
cm_still = float(coefficients(trim_state.vel_body, jnp.zeros(3), trim_controls, ac, a_sound)[4])
print(f"pitch damping: +0.1 rad/s of q changes Cm by {cm_pitching - cm_still:+.5f}")
print(f"  should be negative, opposing the rotation. Cmq = {float(ac.Cmq):+.3f}")
print("  PASS" if cm_pitching < cm_still else "  FAIL")
print()

# =====================================================================
# two things happening at once, still doable by hand
# =====================================================================

pitched_q = euler_to_quat(jnp.array(0.0), jnp.radians(jnp.array(10.0)), jnp.array(0.0))
g_body = quat_to_dcm(pitched_q).T @ jnp.array([0.0, 0.0, G0])
check("pitched 10 deg nose-up: gravity along x_b should be -g*sin(10)",
      -G0 * np.sin(np.radians(10.0)), float(g_body[0]), tol=1e-12)
check("...and gravity along z_b should be +g*cos(10)",
      G0 * np.cos(np.radians(10.0)), float(g_body[2]), tol=1e-12)

nz = float(load_factor(trim_state, trim_controls, ac, jnp.zeros(3), jnp.zeros(3)))
check("load factor in trimmed level flight is cos(alpha), NOT 1",
      float(np.cos(alpha)), nz, tol=1e-6)

# =====================================================================
# does the whole assembled model behave right structurally
# =====================================================================

elev_only = trim.trimmed_controls(x_trim[1] + 0.15, x_trim[2])
d_long = derivatives(
    trim_state._replace(omega=jnp.array([0.0, 0.05, 0.0])),
    elev_only, ac, jnp.zeros(3), jnp.zeros(3),
)
lateral_stuff = jnp.array([d_long.vel_body[1], d_long.omega[0], d_long.omega[2]])
check("wings level, beta=0: elevator alone shouldn't wake up roll/yaw at all",
      0.0, float(jnp.abs(lateral_stuff).max()), tol=1e-18)

# last one isn't pass/fail - just checking which convention the force
# code is actually using for CY (body-axis or wind-axis). both are
# "correct" in the sense that people do it both ways, the question is
# just which one this code happens to implement
print("checking body-axis vs wind-axis convention for CY (not pass/fail):")
print()
print(f"{'beta':>6}  {'vs body-axis':>14}  {'vs wind-axis':>14}")
for beta_deg in (0.0, 5.0, 10.0):
    beta = np.radians(beta_deg)
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    vel = jnp.array([v * ca * cb, v * sb, v * sa * cb])

    force, _ = aero_forces_moments(vel, jnp.zeros(3), trim_controls, ac, rho, a_sound)
    cl, cd, cy, _, _, _ = coefficients(vel, jnp.zeros(3), trim_controls, ac, a_sound)
    qs = float(0.5 * rho * v**2 * ac.S)
    lift, drag, side = qs * float(cl), qs * float(cd), qs * float(cy)

    body_guess = np.array([-drag * ca * cb + lift * sa, -drag * sb + side, -drag * sa * cb - lift * ca])
    wind_guess = np.array([
        -drag * ca * cb - side * ca * sb + lift * sa,
        -drag * sb + side * cb,
        -drag * sa * cb - side * sa * sb - lift * ca,
    ])
    f = np.asarray(force)
    print(f"{beta_deg:>4.0f}    {np.abs(f - body_guess).max():>12.3e}  {np.abs(f - wind_guess).max():>12.3e}")

print()
print("looks like CY is being treated as body-axis already - both columns")
print("agree at beta=0, which is exactly why nothing else catches this")
print()

# =====================================================================
print(f"{sum(results)}/{len(results)} checks passed")