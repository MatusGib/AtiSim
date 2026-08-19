"""Agent C, item 2 (quaternion half): quat_to_dcm and quat_derivative.

Everything here is checked against a matrix built from scratch in this file --
no flightsim function is used to validate another flightsim function.

Non-trivial attitude throughout: all three Euler angles distinct, non-zero, and
not multiples of 45 deg, so a transposed or axis-swapped DCM cannot pass.
"""
import numpy as np

from flightsim.state import (
    euler_to_quat, quat_derivative, quat_to_dcm, quat_to_euler, quat_normalize,
)

np.set_printoptions(precision=12, suppress=False, linewidth=150)

# ---------------------------------------------------------------- independent
def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def R_body_to_ned_321(phi, theta, psi):
    """3-2-1 (yaw-pitch-roll) body->NED, built from elementary rotations.

    A vector fixed in the body is expressed in NED by undoing roll, then pitch,
    then yaw: v_ned = Rz(psi) Ry(theta) Rx(phi) v_body.
    """
    return Rz(psi) @ Ry(theta) @ Rx(phi)


def rodrigues(axis_times_angle):
    """Exact exp(skew(v)) -- rotation by |v| about v."""
    th = np.linalg.norm(axis_times_angle)
    if th < 1e-300:
        return np.eye(3)
    k = axis_times_angle / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


fails = []


def report(name, err, tol):
    ok = err < tol
    print(f"{'PASS' if ok else 'FAIL'}  {name:58s} err={err:.3e}  tol={tol:.0e}")
    if not ok:
        fails.append(name)


# ================================================================== test 2g
print("=== quat_to_dcm claims BODY -> NED (v_ned = C @ v_body) ===")
PHI, THETA, PSI = 0.37, -0.21, 1.13  # rad; all three non-zero, unequal
q = np.asarray(euler_to_quat(PHI, THETA, PSI))
C = np.asarray(quat_to_dcm(q))
C_ref = R_body_to_ned_321(PHI, THETA, PSI)
print("quat_to_dcm(q) =\n", C)
print("Rz(psi)Ry(theta)Rx(phi) =\n", C_ref)
report("quat_to_dcm == Rz(psi)Ry(theta)Rx(phi)", np.abs(C - C_ref).max(), 1e-12)
report("NOT the transpose (must be LARGE)  [inverted sense check]",
       1.0 / max(np.abs(C - C_ref.T).max(), 1e-30), 1e12)
print(f"   |C - C_ref^T|max = {np.abs(C - C_ref.T).max():.4f}  "
      f"(must be non-zero, else the sense is ambiguous)")

# Concrete physical spot-check that does not depend on any matrix algebra:
# nose-up pitch alone -> the body x axis must point UP in NED, i.e. have
# negative D (third) component.
q_pitch = np.asarray(euler_to_quat(0.0, 0.30, 0.0))
xb_in_ned = np.asarray(quat_to_dcm(q_pitch)) @ np.array([1.0, 0.0, 0.0])
print(f"\nnose-up theta=+0.30 rad: body-x in NED = {xb_in_ned}"
      f"   (D component {xb_in_ned[2]:+.4f} must be NEGATIVE = up)")
report("theta>0 puts the nose above the horizon", 0.0 if xb_in_ned[2] < 0 else 1.0, 0.5)

# Right roll alone -> body y (right wing) points DOWN in NED.
q_roll = np.asarray(euler_to_quat(0.30, 0.0, 0.0))
yb_in_ned = np.asarray(quat_to_dcm(q_roll)) @ np.array([0.0, 1.0, 0.0])
print(f"right roll phi=+0.30 rad: body-y in NED = {yb_in_ned}"
      f"   (D component {yb_in_ned[2]:+.4f} must be POSITIVE = right wing down)")
report("phi>0 is right-wing-down", 0.0 if yb_in_ned[2] > 0 else 1.0, 0.5)

# Yaw alone -> nose swings to the EAST.
q_yaw = np.asarray(euler_to_quat(0.0, 0.0, 0.30))
xb_yaw = np.asarray(quat_to_dcm(q_yaw)) @ np.array([1.0, 0.0, 0.0])
print(f"yaw psi=+0.30 rad:      body-x in NED = {xb_yaw}"
      f"   (E component {xb_yaw[1]:+.4f} must be POSITIVE = nose right/east)")
report("psi>0 turns the nose to the east", 0.0 if xb_yaw[1] > 0 else 1.0, 0.5)

# Euler round trip
e = np.asarray(quat_to_euler(q))
report("quat_to_euler round trip", np.abs(e - np.array([PHI, THETA, PSI])).max(), 1e-12)
report("|q| == 1", abs(np.linalg.norm(q) - 1.0), 1e-14)

# ================================================================== test 2h
print("\n=== quat_derivative vs an INDEPENDENTLY integrated attitude ===")
# Body rates chosen asymmetric: all three different, none zero, none equal.
OM = np.array([0.17, -0.29, 0.11])  # rad/s, p q r

# Reference: dR/dt = R skew(omega)  (R = body->NED, omega body-axis).
# Over a step dt with constant body rates the EXACT solution is
#   R(t+dt) = R(t) @ expm(skew(omega) dt) = R(t) @ rodrigues(omega*dt)
# Build the reference with rodrigues only -- no quaternion code involved.
R0 = R_body_to_ned_321(PHI, THETA, PSI)

for dt in (1e-3, 1e-4, 1e-5):
    R_ref = R0 @ rodrigues(OM * dt)
    # Euler step on the quaternion using the function under test.
    qd = np.asarray(quat_derivative(q, OM))
    q_next = quat_normalize(q + qd * dt)
    C_next = np.asarray(quat_to_dcm(q_next))
    err = np.abs(C_next - R_ref).max()
    # Euler is O(dt^2); the coefficient is ~|omega|^2/2 ~ 0.06
    print(f"   dt={dt:.0e}  |C_euler - C_exact|max = {err:.3e}   "
          f"(O(dt^2) budget {0.2 * dt * dt:.1e})")
    report(f"quat_derivative Euler step matches exact rotation, dt={dt:.0e}",
           err, 0.2 * dt * dt)

# The decisive test: how does the error scale with dt?  A wrong sign or a
# swapped axis leaves an O(dt) error (ratio 2 on halving); a correct derivative
# integrated by Euler leaves O(dt^2) -- EXCEPT that the whole O(dt^2) term of
# q*exp(w dt/2) lies along the real part (exp(x) = 1 + x + x^2/2 with x
# imaginary and |x|^2 real), which quat_normalize removes exactly. So the
# observed order for a CORRECT derivative followed by normalisation is 3,
# ratio 8 on halving.  Anything less than 3 is a defect in quat_derivative.
errs = []
for dt in (4e-4, 2e-4, 1e-4):
    R_ref = R0 @ rodrigues(OM * dt)
    q_next = quat_normalize(q + np.asarray(quat_derivative(q, OM)) * dt)
    errs.append(np.abs(np.asarray(quat_to_dcm(q_next)) - R_ref).max())
ratios = [errs[i] / errs[i + 1] for i in range(2)]
orders = [np.log2(r) for r in ratios]
print(f"   error ratios on halving dt: {ratios}")
print(f"   implied convergence order:  {orders}  (>=3 => derivative is exact)")
report("convergence order >= 3", max(0.0, 3.0 - min(orders)), 0.15)

# Exact quaternion reference, built here from the axis-angle formula only.
def q_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


dt = 1e-3
th = np.linalg.norm(OM) * dt
axis = OM / np.linalg.norm(OM)
q_exact = q_mul(q, np.concatenate([[np.cos(th / 2)], np.sin(th / 2) * axis]))
q_euler = quat_normalize(q + np.asarray(quat_derivative(q, OM)) * dt)
print(f"   q after dt=1e-3: euler={np.asarray(q_euler)}\n"
      f"                    exact={q_exact}")
report("quaternion right-multiplication convention (body rates)",
       np.abs(np.asarray(q_euler) - q_exact).max(), 1e-9)

# A negated-omega control: this MUST fail the same test, proving the test bites.
q_bad = quat_normalize(q + np.asarray(quat_derivative(q, -OM)) * 1e-3)
bad_err = np.abs(np.asarray(quat_to_dcm(q_bad)) - (R0 @ rodrigues(OM * 1e-3))).max()
print(f"   negated-rate control: err={bad_err:.3e} "
      f"(must be ~2*|omega|*dt = {2 * np.linalg.norm(OM) * 1e-3:.1e}, i.e. the test bites)")
report("negated-rate control is rejected", 1.0 / bad_err, 1.0 / 1e-4)

# ---- also check the body->NED velocity kinematics used by dynamics.derivatives
print("\n=== pos_ned dot = DCM @ vel_body : straight/level north at theta>0 ===")
V = 100.0
q_lvl = np.asarray(euler_to_quat(0.0, 0.10, 0.0))
vb = V * np.array([np.cos(0.10), 0.0, np.sin(0.10)])  # zero flight-path angle
vned = np.asarray(quat_to_dcm(q_lvl)) @ vb
print(f"   theta=+0.10, alpha=+0.10 -> vel_ned = {vned}  (D must be ~0 => level)")
report("level flight has zero NED-down velocity", abs(vned[2]), 1e-10)

print("\nFAILURES:", fails if fails else "none")
