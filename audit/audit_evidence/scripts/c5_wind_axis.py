"""Agent C, item 4: the wind-axis -> body-axis force rotation in aero.py.

The whole point is the ASYMMETRIC case: alpha != 0 AND beta != 0 at the same
time, and unequal, so a term that is only right at beta = 0 shows up.

Method: rather than re-deriving the rotation and comparing formulas, the force
is decomposed geometrically.  Drag must be exactly anti-parallel to the relative
velocity; lift must be exactly perpendicular to it.  Both are properties of the
vector the code returns, so they cannot be satisfied by a matching typo.

The three coefficient channels are isolated by evaluating a stripped Aircraft
with all but one of CL/CD/CY forced to zero, so each direction is tested alone.
"""
import itertools

import jax.numpy as jnp
import numpy as np

from flightsim import aero
from flightsim.aircraft import REGISTRY
from flightsim.state import Controls

fails, findings = [], []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name:60s} {detail}")
    if not ok:
        fails.append(name)


ac = REGISTRY["boeing747"]
RHO, A = jnp.array(0.3), jnp.array(299.0)
ZERO3 = jnp.zeros(3)
NOCTL = Controls(jnp.array(0.0), jnp.array(0.0), jnp.array(0.0), jnp.array(0.0))

# Aircraft with only ONE of the three force channels alive.
ZEROED = dict(CL0=0.0, CLa=0.0, CLq=0.0, CLde=0.0, CD0=0.0, CYb=0.0, CYp=0.0,
              CYr=0.0, CYdr=0.0, Clb=0.0, Clp=0.0, Clr=0.0, Clda=0.0, Cldr=0.0,
              Cnb=0.0, Cnp=0.0, Cnr=0.0, Cnda=0.0, Cndr=0.0, Cm0=0.0, Cma=0.0,
              Cmq=0.0, Cmde=0.0)


def stripped(**alive):
    kw = {k: jnp.array(v) for k, v in {**ZEROED, **alive}.items()}
    return ac._replace(**kw)


# NOTE: CD is NOT just CD0 -- it carries CL^2/(pi e AR) and the Korn wave term.
# To isolate lift the induced term is killed with a huge Oswald factor and the
# wave term by evaluating at a negligible Mach (a_sound below).  Getting this
# wrong is how the first run of this script produced a spurious failure.
LIFT_ONLY = stripped(CL0=1.0, e=1.0e12)             # CL == 1, CD == 0, CY == 0
DRAG_ONLY = stripped(CD0=0.5)                       # CD == 0.5 (CL == 0 so no induced)
SIDE_ONLY = stripped(CYb=1.0, e=1.0e12)             # CY == beta
A_INCOMP = jnp.array(1.0e6)                         # M ~ 0 => wave drag identically 0
print(f"   sanity: CD of LIFT_ONLY at M~0 = "
      f"{float(aero.coefficients(jnp.array([240.0, 0.0, 0.0]), jnp.zeros(3), Controls(jnp.array(0.0), jnp.array(0.0), jnp.array(0.0), jnp.array(0.0)), LIFT_ONLY, A_INCOMP)[1]):.3e}")
print(f"   sanity: CD of DRAG_ONLY at M~0 = "
      f"{float(aero.coefficients(jnp.array([240.0, 0.0, 0.0]), jnp.zeros(3), Controls(jnp.array(0.0), jnp.array(0.0), jnp.array(0.0), jnp.array(0.0)), DRAG_ONLY, A_INCOMP)[1]):.3e}")


def vel_from(alpha, beta, V=240.0):
    """Body velocity with EXACTLY this alpha and beta, by construction."""
    return jnp.array([V * np.cos(alpha) * np.cos(beta),
                      V * np.sin(beta),
                      V * np.sin(alpha) * np.cos(beta)])


ALPHAS = [-0.25, -0.08, 0.0, 0.06, 0.19]
BETAS = [-0.30, -0.11, 0.0, 0.07, 0.22]

print("=" * 78)
print("4.0 air_data inverts the (alpha, beta) construction exactly")
print("=" * 78)
worst = 0.0
for al, be in itertools.product(ALPHAS, BETAS):
    v = vel_from(al, be)
    V, a, b = [float(x) for x in aero.air_data(v)]
    worst = max(worst, abs(a - al), abs(b - be), abs(V - 240.0) / 240.0)
check("air_data recovers alpha and beta over the whole grid", worst < 1e-12,
      f"worst={worst:.2e}")

print()
print("=" * 78)
print("4.1 DRAG is exactly anti-parallel to the relative velocity")
print("=" * 78)
worst_ang, worst_mag = 0.0, 0.0
for al, be in itertools.product(ALPHAS, BETAS):
    v = vel_from(al, be)
    F, _ = aero.aero_forces_moments(v, ZERO3, NOCTL, DRAG_ONLY, RHO, A_INCOMP)
    F = np.asarray(F)
    vhat = np.asarray(v) / np.linalg.norm(np.asarray(v))
    Fhat = F / np.linalg.norm(F)
    cosang = float(np.dot(Fhat, vhat))
    worst_ang = max(worst_ang, abs(cosang + 1.0))
    # magnitude must be qbar*S*CD
    want = 0.5 * float(RHO) * 240.0**2 * float(ac.S) * 0.5
    worst_mag = max(worst_mag, abs(np.linalg.norm(F) - want) / want)
    if abs(al) > 0.1 and abs(be) > 0.1:
        print(f"   alpha={al:+.2f} beta={be:+.2f}: cos(F,v) = {cosang:.15f}  "
              f"|F| = {np.linalg.norm(F):.4f} N")
check("drag is exactly anti-parallel to v_rel at all (alpha,beta)",
      worst_ang < 1e-12, f"max|cos+1|={worst_ang:.2e}")
check("drag magnitude is qbar*S*CD at all (alpha,beta)", worst_mag < 1e-12,
      f"max rel err={worst_mag:.2e}")

print()
print("=" * 78)
print("4.2 LIFT is exactly perpendicular to the relative velocity")
print("=" * 78)
worst_perp, worst_mag = 0.0, 0.0
for al, be in itertools.product(ALPHAS, BETAS):
    v = vel_from(al, be)
    F, _ = aero.aero_forces_moments(v, ZERO3, NOCTL, LIFT_ONLY, RHO, A_INCOMP)
    F = np.asarray(F)
    vhat = np.asarray(v) / np.linalg.norm(np.asarray(v))
    Fhat = F / np.linalg.norm(F)
    worst_perp = max(worst_perp, abs(float(np.dot(Fhat, vhat))))
    want = 0.5 * float(RHO) * 240.0**2 * float(ac.S) * 1.0
    worst_mag = max(worst_mag, abs(np.linalg.norm(F) - want) / want)
    if abs(al) > 0.1 and abs(be) > 0.1:
        print(f"   alpha={al:+.2f} beta={be:+.2f}: (Fhat . vhat) = "
              f"{float(np.dot(Fhat, vhat)):+.3e}  |F| = {np.linalg.norm(F):.4f} N  "
              f"F = {F}")
check("lift is exactly perpendicular to v_rel at all (alpha,beta), "
      "INCLUDING beta != 0", worst_perp < 1e-12, f"max|cos|={worst_perp:.2e}")
check("lift magnitude is qbar*S*CL at all (alpha,beta)", worst_mag < 1e-12,
      f"max rel err={worst_mag:.2e}")
# lift must point "up" (negative body z) for a positive CL at small alpha
F, _ = aero.aero_forces_moments(vel_from(0.06, 0.22), ZERO3, NOCTL, LIFT_ONLY, RHO, A_INCOMP)
check("positive CL acts upward (Fz < 0)", float(F[2]) < 0, f"Fz={float(F[2]):+.1f}")
# lift must have NO body-y component under this convention.  Relative
# tolerance: the e=1e12 trick leaves CD ~ 5e-14, whose -D*sin(beta) term is a
# genuine (and correct) body-y contribution of that same relative size.
check("lift has no body-y component (wind-axis z lies in the body xz plane)",
      abs(float(F[1])) / float(jnp.linalg.norm(F)) < 1e-12,
      f"Fy/|F|={abs(float(F[1])) / float(jnp.linalg.norm(F)):.2e}")

print()
print("=" * 78)
print("4.3 SIDE FORCE: what convention is it actually in?")
print("=" * 78)
for al, be in [(0.19, 0.22), (-0.25, -0.30), (0.0, 0.22), (0.19, 0.0)]:
    v = vel_from(al, be)
    F, _ = aero.aero_forces_moments(v, ZERO3, NOCTL, SIDE_ONLY, RHO, A_INCOMP)
    F = np.asarray(F)
    vhat = np.asarray(v) / np.linalg.norm(np.asarray(v))
    qbarS = 0.5 * float(RHO) * 240.0**2 * float(ac.S)
    print(f"   alpha={al:+.2f} beta={be:+.2f}: F = {F}   "
          f"F/(qbarS*CY) = {F / (qbarS * be) if be else 'n/a'}")
    if be:
        print(f"        (Fhat . vhat) = {float(np.dot(F / np.linalg.norm(F), vhat)):+.6f}"
              f"   <- non-zero => side force is BODY-y, not wind-axis-y")
# The claim in the docstring is that CY is a BODY-axis coefficient.
v = vel_from(0.19, 0.22)
F, _ = aero.aero_forces_moments(v, ZERO3, NOCTL, SIDE_ONLY, RHO, A_INCOMP)
qbarS = 0.5 * float(RHO) * 240.0**2 * float(ac.S)
check("side force is exactly (0, qbar*S*CY, 0) in body axes",
      abs(float(F[0])) < 1e-9 and abs(float(F[2])) < 1e-9
      and abs(float(F[1]) - qbarS * 0.22) < 1e-6 * qbarS,
      f"F={np.asarray(F)}")
findings.append(
    "CY is applied as a pure body-y force, NOT rotated through beta. The "
    "docstring says so explicitly ('Side force is already body-axis, as the "
    "lateral derivatives are defined'), and CR-2144's CY_beta is indeed a "
    "body-axis derivative, so this is a stated convention rather than a defect. "
    "Consequence: the side force is not perpendicular to the relative wind at "
    "beta != 0, so it does a small amount of work along the flight path -- "
    "second order in beta (cos(F,v) ~ sin(beta))."
)

print()
print("=" * 78)
print("4.4 the full force reassembles from the three isolated channels")
print("=" * 78)
# Superposition must hold exactly (the build-up is linear in the coefficients),
# and the total must equal the independently-constructed rotation.
FULL = ac
worst = 0.0
for al, be in itertools.product(ALPHAS, BETAS):
    v = vel_from(al, be)
    F, _ = aero.aero_forces_moments(v, ZERO3, NOCTL, FULL, RHO, A)
    CL, CD, CY, _, _, _ = aero.coefficients(v, ZERO3, NOCTL, FULL, A)
    qbarS = 0.5 * float(RHO) * 240.0**2 * float(FULL.S)
    ca, sa = np.cos(al), np.sin(al)
    cb, sb = np.cos(be), np.sin(be)
    # Independently written: F_body = R_bw @ (-D, 0, -L) + (0, Y_body, 0), with
    # R_bw the standard wind->body matrix.
    R_bw = np.array([[ca * cb, -ca * sb, -sa],
                     [sb,       cb,       0.0],
                     [sa * cb, -sa * sb,  ca]])
    F_ref = R_bw @ np.array([-qbarS * float(CD), 0.0, -qbarS * float(CL)]) \
        + np.array([0.0, qbarS * float(CY), 0.0])
    worst = max(worst, float(np.abs(np.asarray(F) - F_ref).max()) / qbarS)
check("aero.py force == R_bw(-D,0,-L) + (0,Y_body,0), independently built",
      worst < 1e-12, f"max err/qbarS={worst:.2e}")

# And for contrast: how far is it from the FULLY wind-axis convention
# F = R_bw @ (-D, Y, -L)?
worst_wind = 0.0
for al, be in itertools.product(ALPHAS, BETAS):
    v = vel_from(al, be)
    F, _ = aero.aero_forces_moments(v, ZERO3, NOCTL, FULL, RHO, A)
    CL, CD, CY, _, _, _ = aero.coefficients(v, ZERO3, NOCTL, FULL, A)
    qbarS = 0.5 * float(RHO) * 240.0**2 * float(FULL.S)
    ca, sa, cb, sb = np.cos(al), np.sin(al), np.cos(be), np.sin(be)
    R_bw = np.array([[ca * cb, -ca * sb, -sa], [sb, cb, 0.0], [sa * cb, -sa * sb, ca]])
    F_wind = R_bw @ np.array([-qbarS * float(CD), qbarS * float(CY), -qbarS * float(CL)])
    worst_wind = max(worst_wind, float(np.abs(np.asarray(F) - F_wind).max()) / qbarS)
print(f"   difference from the fully-wind-axis convention: "
      f"max |dF|/qbarS = {worst_wind:.4f} (i.e. {worst_wind:.4f} in CY units)")
print(f"   -- at beta = 0 the two conventions are identical: ", end="")
v0 = vel_from(0.19, 0.0)
F0, _ = aero.aero_forces_moments(v0, ZERO3, NOCTL, FULL, RHO, A)
CL, CD, CY, _, _, _ = aero.coefficients(v0, ZERO3, NOCTL, FULL, A)
qbarS = 0.5 * float(RHO) * 240.0**2 * float(FULL.S)
ca, sa = np.cos(0.19), np.sin(0.19)
R_bw = np.array([[ca, 0.0, -sa], [0.0, 1.0, 0.0], [sa, 0.0, ca]])
F_wind0 = R_bw @ np.array([-qbarS * float(CD), qbarS * float(CY), -qbarS * float(CL)])
print(f"{float(np.abs(np.asarray(F0) - F_wind0).max()):.2e} N")

print()
print("FAILURES:", fails if fails else "none")
for f in findings:
    print("\nFINDING:", f)
