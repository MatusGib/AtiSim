"""Sanity ladder: dummy cases first, then build up.

Every case here states what the PHYSICS requires before it asks the model, and
the expected value is derived by hand rather than read out of the simulator. A
case whose expectation comes from the model would agree with it by construction
and demonstrate nothing.

The ladder runs in four rungs, easiest to reason about first:

  0  degenerate inputs   -- zero the wind, or zero a coefficient so a motion
                            becomes impossible. The answer is 0 or "unchanged"
                            and needs no aerodynamics to predict.
  1  single axis, signs  -- one input, one response, direction only.
  2  two things at once  -- hand-computable numbers, still checkable on paper.
  3  the whole model     -- structural properties of the assembled 6-DOF.

Run it:

    PYTHONPATH=. python scripts/sanity.py

Everything printed is also asserted somewhere in flightsim/tests/, except the
last case, which is a convention probe rather than a pass/fail (see below).
"""

import jax.numpy as jnp
import numpy as np

import flightsim  # noqa: F401  -- enables float64 before any array is made
from flightsim import trim, verification
from flightsim.aero import aero_forces_moments, air_data, coefficients
from flightsim.aircraft import CRUISE, REGISTRY, inertia_tensor
from flightsim.atmosphere import G0, density, speed_of_sound
from flightsim.dynamics import derivatives, load_factor, relative_velocity
from flightsim.state import State, euler_to_quat, quat_to_dcm

# ---------------------------------------------------------------------------

_n = 0


def case(title, physics, expected, got, tol=0.0):
    """Print one rung of the ladder. `expected` is hand-derived, never sampled."""
    global _n
    _n += 1
    ok = abs(got - expected) <= tol if tol else got == expected
    mark = "PASS" if ok else "FAIL"
    print(f"[{_n:>2}] {title}")
    print(f"     physics : {physics}")
    print(f"     expected: {expected:.10g}" + (f"   (tol {tol:g})" if tol else "  exactly"))
    print(f"     model   : {got:.10g}")
    print(f"     {mark}\n")
    return ok


def rung(n, title):
    print(f"\n{'=' * 72}\nRUNG {n} - {title}\n{'=' * 72}\n")


# ---------------------------------------------------------------------------
# Shared setup: the 747 at its cruise condition, trimmed.
# ---------------------------------------------------------------------------

AC = REGISTRY["boeing747"]
V = CRUISE["boeing747"]["airspeed"]
H = CRUISE["boeing747"]["altitude"]
RHO = density(jnp.array(H))
A_SOUND = speed_of_sound(jnp.array(H))

x_trim, res = trim.trim(jnp.array(V), jnp.array(H), AC)
ALPHA = float(x_trim[0])
TRIM_CONTROLS = trim.trimmed_controls(x_trim[1], x_trim[2])
TRIM_STATE = trim.trimmed_state(x_trim[0], jnp.array(V), jnp.array(H))

print(f"Boeing 747, {V:.1f} m/s at {H:.0f} m.")
print(f"Trim: alpha {np.degrees(ALPHA):.3f} deg, residual "
      f"{float(jnp.linalg.norm(res)):.2e}\n")

results = []

# ---------------------------------------------------------------------------
rung(0, "degenerate inputs: the answer is zero, and you can see why")

# [1] Zero wind. The reviewer's example, and the cheapest check in the project.
zero_wind = relative_velocity(TRIM_STATE.vel_body, TRIM_STATE.quat, jnp.zeros(3))
results.append(case(
    "Zero wind vector leaves the air-relative velocity alone",
    "vel_rel = vel_body - DCM' * 0, so the difference must be identically zero",
    0.0,
    float(jnp.abs(zero_wind - TRIM_STATE.vel_body).max()),
))

# [2] Free fall. Nothing but gravity can act, so the answer is g, downward.
naked = verification.without_aerodynamics(AC)
d_free = derivatives(
    TRIM_STATE._replace(omega=jnp.zeros(3)),
    trim.trimmed_controls(jnp.array(0.0), jnp.array(1.0)),  # full throttle, zero thrust
    naked, jnp.zeros(3), jnp.zeros(3),
)
accel_ned = quat_to_dcm(TRIM_STATE.quat) @ d_free.vel_body
results.append(case(
    "No aerodynamics and no thrust gives exactly g, straight down",
    "every force but weight removed, so a_ned = (0, 0, g) with g = 9.80665",
    G0, float(accel_ned[2]), tol=1e-12,
))
results.append(case(
    "...and nothing horizontal",
    "a_ned north and east components must vanish",
    0.0, float(jnp.abs(accel_ned[:2]).max()), tol=1e-12,
))

# [3] Roll made impossible -- the reviewer's other example.
#
# CAREFUL: zeroing the rolling-moment coefficients is NOT enough on a real
# aircraft. omega_dot = I^-1 (M - omega x I omega), and the 747's inertia
# tensor carries Ixz, so I^-1 is NOT diagonal: a YAWING moment still produces
# roll acceleration through inertial coupling. Both halves are shown, because
# the naive expectation is wrong and the reason is real flight dynamics.
no_roll_coeffs = dict(Clb=0.0, Clp=0.0, Clr=0.0, Clda=0.0, Cldr=0.0)
sideslipping = TRIM_STATE._replace(
    vel_body=TRIM_STATE.vel_body + jnp.array([0.0, 12.0, 0.0]),  # beta ~ 3 deg
    omega=jnp.zeros(3),
)
full_aileron = trim.trimmed_controls(x_trim[1], x_trim[2])._replace(
    aileron=jnp.array(0.3), rudder=jnp.array(0.2)
)

diagonal = AC._replace(
    **{k: jnp.array(v) for k, v in no_roll_coeffs.items()},
    inertia=inertia_tensor(4.678e7, 3.312e7, 6.722e7, 0.0),
    inertia_inv=jnp.linalg.inv(inertia_tensor(4.678e7, 3.312e7, 6.722e7, 0.0)),
)
d_diag = derivatives(sideslipping, full_aileron, diagonal, jnp.zeros(3), jnp.zeros(3))
results.append(case(
    "Roll coefficients zeroed, DIAGONAL inertia: full aileron gives no roll",
    "Cl == 0 so L == 0; with Ixz = 0 the p row of I^-1 touches nothing else",
    0.0, float(d_diag.omega[0]), tol=1e-18,
))

coupled = AC._replace(**{k: jnp.array(v) for k, v in no_roll_coeffs.items()})
d_coup = derivatives(sideslipping, full_aileron, coupled, jnp.zeros(3), jnp.zeros(3))
print(f"[--] Same case on the REAL 747 inertia (Ixz != 0): p_dot = "
      f"{float(d_coup.omega[0]):.6e} rad/s^2")
print("     NOT zero, and correctly so -- the yawing moment reaches the roll axis")
print("     through Ixz. This is the first case where a naive expectation fails.\n")

# [4] Pitch made impossible. Cleaner, because the q row of I^-1 IS isolated.
no_pitch = AC._replace(
    Cm0=jnp.array(0.0), Cma=jnp.array(0.0),
    Cmq=jnp.array(0.0), Cmde=jnp.array(0.0),
)
d_nopitch = derivatives(
    sideslipping,
    trim.trimmed_controls(jnp.array(0.4), x_trim[2]),  # large elevator
    no_pitch, jnp.zeros(3), jnp.zeros(3),
)
results.append(case(
    "Pitch coefficients zeroed: large elevator gives no pitch acceleration",
    "Cm == 0 so M == 0; no Ixy or Iyz, so the q row of I^-1 is isolated",
    0.0, float(d_nopitch.omega[1]), tol=1e-18,
))

# ---------------------------------------------------------------------------
rung(1, "one input, one response: signs only")

# [5] Headwind. Level, heading north, wind blowing from the north.
level = euler_to_quat(jnp.array(0.0), jnp.array(0.0), jnp.array(0.0))
head = relative_velocity(jnp.array([V, 0.0, 0.0]), level, jnp.array([-30.0, 0.0, 0.0]))
results.append(case(
    "A 30 m/s headwind raises airspeed 30 m/s above groundspeed",
    "wind_ned = (-30,0,0) blows south; vel_rel_x = V - (-30) = V + 30",
    V + 30.0, float(air_data(head)[0]), tol=1e-9,
))

# [6] Angle of attack sign.
results.append(case(
    "Upward relative flow (w > 0) gives positive alpha",
    "alpha = arctan2(w, u); w = u gives exactly 45 degrees",
    45.0, float(np.degrees(air_data(jnp.array([100.0, 0.0, 100.0]))[1])), tol=1e-9,
))

# [7] Static pitch stability: the restoring sign.
def Cm_at(alpha_rad):
    vel = V * jnp.array([jnp.cos(alpha_rad), 0.0, jnp.sin(alpha_rad)])
    return float(coefficients(vel, jnp.zeros(3), TRIM_CONTROLS, AC, A_SOUND)[4])


dCm = Cm_at(jnp.array(ALPHA + 0.05)) - Cm_at(jnp.array(ALPHA))
print(f"[--] Static stability: raising alpha by 2.9 deg changes Cm by {dCm:+.5f}")
print(f"     Must be NEGATIVE (nose-down restoring). Cma = {float(AC.Cma):+.3f}")
print(f"     {'PASS' if dCm < 0 else 'FAIL'}\n")

# [8] Pitch damping opposes the rotation.
Cm_pitching = float(coefficients(
    TRIM_STATE.vel_body, jnp.array([0.0, 0.1, 0.0]), TRIM_CONTROLS, AC, A_SOUND
)[4])
Cm_still = float(coefficients(
    TRIM_STATE.vel_body, jnp.zeros(3), TRIM_CONTROLS, AC, A_SOUND
)[4])
print(f"[--] Pitch damping: a +0.1 rad/s pitch rate changes Cm by "
      f"{Cm_pitching - Cm_still:+.5f}")
print(f"     Must be NEGATIVE (opposing). Cmq = {float(AC.Cmq):+.3f}")
print(f"     {'PASS' if Cm_pitching < Cm_still else 'FAIL'}\n")

# ---------------------------------------------------------------------------
rung(2, "two things at once, still checkable on paper")

# [9] Gravity resolved through a known attitude.
pitched = euler_to_quat(jnp.array(0.0), jnp.radians(jnp.array(10.0)), jnp.array(0.0))
g_body = quat_to_dcm(pitched).T @ jnp.array([0.0, 0.0, G0])
results.append(case(
    "Pitched 10 deg nose-up: body-x gravity is -g sin(10 deg)",
    "g_b = g(-sin th, sin ph cos th, cos ph cos th) = 9.80665 * -0.173648",
    -G0 * np.sin(np.radians(10.0)), float(g_body[0]), tol=1e-12,
))
results.append(case(
    "...and body-z gravity is +g cos(10 deg)",
    "9.80665 * 0.984808",
    G0 * np.cos(np.radians(10.0)), float(g_body[2]), tol=1e-12,
))

# [10] Load factor in level trim is cos(theta), NOT 1. A real subtlety.
nz = float(load_factor(TRIM_STATE, TRIM_CONTROLS, AC, jnp.zeros(3), jnp.zeros(3)))
results.append(case(
    "Load factor in trimmed level flight is cos(theta), not 1",
    f"body-normal accelerometer reads g cos(theta); theta = alpha = "
    f"{np.degrees(ALPHA):.3f} deg",
    float(np.cos(ALPHA)), nz, tol=1e-6,
))

# ---------------------------------------------------------------------------
rung(3, "structural properties of the assembled model")

# [11] Longitudinal and lateral motion decouple at zero sideslip.
elevator_only = trim.trimmed_controls(x_trim[1] + 0.15, x_trim[2])
d_long = derivatives(
    TRIM_STATE._replace(omega=jnp.array([0.0, 0.05, 0.0])),
    elevator_only, AC, jnp.zeros(3), jnp.zeros(3),
)
lateral = jnp.array([d_long.vel_body[1], d_long.omega[0], d_long.omega[2]])
results.append(case(
    "Wings level at beta = 0, a pure elevator input excites NO lateral motion",
    "CY, Cl, Cn all vanish at beta = 0 with no aileron, rudder, p or r; and "
    "g_y = g sin(phi) cos(theta) = 0 wings-level",
    0.0, float(jnp.abs(lateral).max()), tol=1e-18,
))

# [12] Item 08: which wind-to-body convention does the force assembly use?
#
# NOT a pass/fail. Both readings are defensible; the question is which one the
# code implements and whether that matches the source document's axis system
# for CY. See the equation-mapping document, item 08.
print("[--] Item 08 probe: wind-axis to body-axis force convention\n")
print(f"     {'beta':>6} {'vs body-axis CY':>18} {'vs wind-axis CY':>18}")
for beta_deg in (0.0, 5.0, 10.0):
    beta = np.radians(beta_deg)
    ca, sa = np.cos(ALPHA), np.sin(ALPHA)
    cb, sb = np.cos(beta), np.sin(beta)
    vel = jnp.array([V * ca * cb, V * sb, V * sa * cb])

    force, _ = aero_forces_moments(vel, jnp.zeros(3), TRIM_CONTROLS, AC, RHO, A_SOUND)
    CL, CD, CY, _, _, _ = coefficients(vel, jnp.zeros(3), TRIM_CONTROLS, AC, A_SOUND)
    qS = float(0.5 * RHO * V**2 * AC.S)
    L, D, Yf = qS * float(CL), qS * float(CD), qS * float(CY)

    body_axis = np.array([-D * ca * cb + L * sa, -D * sb + Yf, -D * sa * cb - L * ca])
    wind_axis = np.array([
        -D * ca * cb - Yf * ca * sb + L * sa,
        -D * sb + Yf * cb,
        -D * sa * cb - Yf * sa * sb - L * ca,
    ])
    f = np.asarray(force)
    print(f"     {beta_deg:>4.0f} deg {np.abs(f - body_axis).max():>16.3e} N "
          f"{np.abs(f - wind_axis).max():>16.3e} N")

print("\n     The code treats CY as ALREADY BODY-AXIS. Both columns agree at")
print("     beta = 0, which is why no existing test can tell them apart.\n")

# ---------------------------------------------------------------------------
print("=" * 72)
print(f"{sum(results)} of {len(results)} pass/fail cases passed.")
print("Sign checks and the item 08 probe are reported inline above.")
print("=" * 72)
