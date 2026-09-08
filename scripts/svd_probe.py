"""Does an SVD of the trim Jacobian see what the residual cannot?

PROJECT.md section 5 records a defect that has no test able to catch it in
general: `trim.trim` converges to physically absurd roots for degenerate
coefficients. `CL = CL0 + CLa*alpha` is linear, so a huge alpha compensates a
small CLa, and Newton reaches a root that satisfies the residual to machine
precision at hundreds of degrees of incidence. The record's own words:
"Convergence and sense are different questions."

`trim.is_physical` catches it AFTER the fact, by looking at the answer and
asking whether the three unknowns are flyable. That works and stays. What it
cannot do is say WHY, or warn BEFORE the solve, or generalise to a degeneracy
that happens to land inside the limits.

ASSUMPTIONS.md F7 records the same failure mode from the other side: a control
channel with zero authority makes the Newton solve return NaN in silence, and
"no shipped solver carries rudder as an unknown; a steady-turn solve would be
the first".

BOTH ARE RANK PROBLEMS IN ONE 3x3 MATRIX, and the singular value decomposition
is the instrument that measures rank. This script asks whether it does, on the
project's own recorded cases, and it is a MEASUREMENT rather than a change:
nothing here is imported by the model, and `trim.py` is untouched.

Three questions:

  1. Does the residual separate a healthy trim from an absurd one?  (Recorded
     answer: no. Section 5 quotes 2.3e-15 and 5.7e-15 for the two absurd roots.)
  2. Does sigma_min / the condition number of the trim Jacobian separate them?
  3. In the rank-deficient case, does the smallest right singular vector NAME
     the direction the solve cannot see -- i.e. is it interpretable as a
     combination of [alpha, elevator, throttle]?

Question 3 is the one that matters for the wider claim, because the same
question is already open elsewhere in this project under a different name:
`gen_jsbsim_reference.recover_pitch_axis` reports a fit condition of 1.9e8 (737)
and 2.7e9 (747), and section 4 resolves it by hand -- "their SUM is exact and
their SPLIT is conditioned". An SVD does not need to be told which combination
is the sum.

*** WHAT IT FOUND, WRITTEN HERE SO NOBODY RE-RUNS IT HOPEFULLY ***

  Q1/Q2: NO. THE HYPOTHESIS IS FALSIFIED AND THE SECTION-5 DEFECT IS NOT A RANK
  PROBLEM. cond(J) at the healthy root is 9.51e+01 and at the two absurd roots
  it is 1.97e+01 and 2.26e+01 -- the degenerate cases are BETTER conditioned
  than the healthy one, and the whole spread is a factor of 4.8. An SVD cannot
  see this defect, and in hindsight the reason is obvious: at the absurd root
  the Jacobian is a perfectly good matrix. Newton found a genuine, well-
  conditioned root of a function that HAS several roots. Multiple roots of a
  nonlinear system is not rank deficiency, and no decomposition of the local
  Jacobian is going to report it. `trim.is_physical` stays exactly as it is.

  Q3 / F7: YES, AND CLEANLY. Zero the elevator's two coefficients and
  sigma_min is 0.000e+00 EXACTLY -- at the initial guess, before the first
  Newton step, where the solver's own output is NaN and carries no information
  at all. That is a real detection of a real recorded defect, available at a
  point where nothing else in the project can say anything.

  Q4 (added after the above): identifiability. The SVD of the sensitivity of
  the longitudinal modes to (C_mALPHA, C_mq, I_yy) has a null direction, and it
  is the (+1,+1,+1)/sqrt(3) log-space combination -- i.e. scale all three
  together and the aeroplane does not move. That is session 27's B787 finding
  (`I_yy` is unobservable because it and `C_mALPHA` enter only as their product
  `M_ALPHA`), DERIVED rather than discovered by sweeping a parameter and
  noticing nothing happened.

  SO: one hypothesis dead, one live, and the live one is not the one this
  script was written for.

Run: PYTHONPATH=<abs worktree root> .venv/Scripts/python.exe scripts/svd_probe.py
"""

import math

import jax
import jax.numpy as jnp
import numpy as np

from atisim import aircraft, trim
from atisim.units import RAD2DEG

# Section 5's own two cases, on the aircraft it names, at the condition it
# names: boeing747_approach at 85 m/s and sea level. The CLa values are the
# corrected ones -- session 12 found session 11 had attributed the -633 deg to
# CLa = 0.1 when it belongs to CLa = 1e-4.
AIRSPEED = 85.0
ALTITUDE = 0.0
CLA_CASES = (
    ("healthy (as shipped)", None),
    ("degenerate CLa = 0.1", 0.1),
    ("degenerate CLa = 1e-4", 1e-4),
)


def jacobian_at(x, airspeed, altitude, ac):
    """The matrix `trim.trim` inverts every Newton step, at the point x."""
    return np.asarray(
        jax.jacfwd(trim.residual)(x, jnp.array(airspeed), jnp.array(altitude), ac)
    )


def report(name, ac, airspeed=AIRSPEED, altitude=ALTITUDE):
    x, residual = trim.trim(jnp.array(airspeed), jnp.array(altitude), ac)
    r_norm = float(jnp.linalg.norm(residual))
    alpha_deg = float(x[0]) * RAD2DEG

    J = jacobian_at(x, airspeed, altitude, ac)
    sv = np.linalg.svd(J, compute_uv=False)
    cond = sv[0] / sv[-1] if sv[-1] > 0 else math.inf

    print(f"  {name:26s}")
    print(f"    alpha            {alpha_deg:14.3f} deg")
    print(f"    residual norm    {r_norm:14.2e}   <- what the solver reports")
    print(f"    is_physical      {str(trim.is_physical(x, ac)):>14s}")
    print(f"    singular values  {sv[0]:9.3e} {sv[1]:9.3e} {sv[2]:9.3e}")
    print(f"    sigma_min        {sv[-1]:14.2e}")
    print(f"    condition        {cond:14.2e}   <- what an SVD reports")
    return alpha_deg, r_norm, cond


def main():
    print("atisim imported from:", aircraft.__file__)
    print()
    print("=" * 78)
    print("1. THE SECTION-5 CASE: does the residual see an absurd root? Does an SVD?")
    print("=" * 78)
    print(f"  boeing747_approach, V = {AIRSPEED} m/s, altitude = {ALTITUDE} m")
    print()

    rows = []
    for name, cla in CLA_CASES:
        ac = aircraft.REGISTRY["boeing747_approach"]
        if cla is not None:
            ac = ac._replace(CLa=jnp.array(cla))
        rows.append((name, *report(name, ac)))
        print()

    print("  SUMMARY")
    print(f"  {'case':26s} {'alpha, deg':>14s} {'residual':>12s} {'cond(J)':>12s}")
    for name, alpha_deg, r_norm, cond in rows:
        print(f"  {name:26s} {alpha_deg:14.3f} {r_norm:12.2e} {cond:12.2e}")
    spread_r = max(r[2] for r in rows) / min(r[2] for r in rows)
    spread_c = max(r[3] for r in rows) / min(r[3] for r in rows)
    print()
    print(f"  residual spans a factor of {spread_r:.1f} across the three cases")
    print(f"  cond(J)  spans a factor of {spread_c:.3e} across the three cases")

    print()
    print("=" * 78)
    print("2. F7: A CONTROL CHANNEL WITH NO AUTHORITY")
    print("=" * 78)
    print("  ASSUMPTIONS F7 says a zero-authority channel returns NaN in silence.")
    print("  Zeroing CLde and Cmde removes the elevator from the residual entirely.")
    print()
    ac = aircraft.REGISTRY["boeing747_approach"]._replace(
        CLde=jnp.array(0.0), Cmde=jnp.array(0.0)
    )
    x, residual = trim.trim(jnp.array(AIRSPEED), jnp.array(ALTITUDE), ac)
    print(f"    solution         {np.asarray(x)}")
    print(f"    residual norm    {float(jnp.linalg.norm(residual)):14.2e}")
    J = jacobian_at(jnp.array(trim.INITIAL_GUESS), AIRSPEED, ALTITUDE, ac)
    sv = np.linalg.svd(J, compute_uv=False)
    print(f"    singular values at the START point: {sv[0]:.3e} {sv[1]:.3e} {sv[2]:.3e}")
    print(f"    sigma_min        {sv[-1]:14.2e}")
    print("    ^ measured at the INITIAL GUESS, i.e. BEFORE the first Newton step.")
    print("      A NaN solution cannot be diagnosed after the fact; this can.")

    print()
    print("=" * 78)
    print("3. WHAT THE SMALLEST SINGULAR VECTOR NAMES")
    print("=" * 78)
    print("  The right singular vector of sigma_min is the combination of")
    print("  [alpha, elevator, throttle] the Jacobian is blindest to.")
    print()
    for name, cla in CLA_CASES:
        ac = aircraft.REGISTRY["boeing747_approach"]
        if cla is not None:
            ac = ac._replace(CLa=jnp.array(cla))
        x, _ = trim.trim(jnp.array(AIRSPEED), jnp.array(ALTITUDE), ac)
        _, sv, vt = np.linalg.svd(jacobian_at(x, AIRSPEED, ALTITUDE, ac))
        v = vt[-1]
        v = v * np.sign(v[np.argmax(np.abs(v))])
        print(
            f"  {name:26s} sigma_min {sv[-1]:.2e}  "
            f"direction [a {v[0]:+.4f}, de {v[1]:+.4f}, dt {v[2]:+.4f}]"
        )

    identifiability()

    print()
    print("NOTHING ABOVE CHANGES THE MODEL. trim.py is untouched; this script")
    print("imports it and measures the matrix it already builds every step.")


# The three coefficients that enter the pitch equation only through I_yy.
# M_alpha = C_malpha * qbar*S*cbar / I_yy and M_q = C_mq * qbar*S*cbar^2 /
# (2*V*I_yy), so scaling all three by one factor leaves both untouched.
# Session 27 found the C_malpha/I_yy half of this by sweeping I_yy from 1.0e7
# to 4.0e7 on the B787 entry and watching M_alpha come back identical every
# time. This asks the SVD instead, on the 747, and does not tell it the answer.
IDENT_PARAMS = ("Cma", "Cmq", "Iyy")
REL_STEP = 1e-4


def _modes_with(ac, scale, alpha, elevator, throttle, V, H):
    """The four longitudinal mode scalars with the three parameters scaled."""
    from atisim import validation

    inertia = np.asarray(ac.inertia).copy()
    inertia[1, 1] = inertia[1, 1] * scale[2]
    perturbed = ac._replace(
        Cma=ac.Cma * scale[0],
        Cmq=ac.Cmq * scale[1],
        inertia=jnp.array(inertia),
        inertia_inv=jnp.linalg.inv(jnp.array(inertia)),
    )
    (p_wn, p_z), (s_wn, s_z) = validation.longitudinal_modes(
        perturbed, alpha, elevator, throttle, V, H
    )
    return np.array([p_wn, p_z, s_wn, s_z])


def identifiability():
    print()
    print("=" * 78)
    print("4. IDENTIFIABILITY: WHICH PARAMETER COMBINATIONS THE MODES CANNOT SEE")
    print("=" * 78)
    ac = aircraft.REGISTRY["boeing747"]
    cr = aircraft.CRUISE["boeing747"]
    V, H = cr["airspeed"], cr["altitude"]
    x, _ = trim.trim(jnp.array(V), jnp.array(H), ac)
    alpha, elevator, throttle = (float(v) for v in x)

    base = _modes_with(ac, np.ones(3), alpha, elevator, throttle, V, H)
    print(f"  boeing747 at its own cruise, trimmed at alpha = {alpha * RAD2DEG:.3f} deg")
    print(f"  observables: phugoid (wn, zeta), short period (wn, zeta)")
    print(f"    {base[0]:.6f} {base[1]:.6f} {base[2]:.6f} {base[3]:.6f}")
    print()

    # Central differences in LOG parameter space, so the singular vectors read
    # as power-law combinations and the matrix is dimensionless.
    S = np.zeros((4, 3))
    for j in range(3):
        hi, lo = np.ones(3), np.ones(3)
        hi[j], lo[j] = 1.0 + REL_STEP, 1.0 - REL_STEP
        up = _modes_with(ac, hi, alpha, elevator, throttle, V, H)
        dn = _modes_with(ac, lo, alpha, elevator, throttle, V, H)
        S[:, j] = (up - dn) / (2.0 * REL_STEP) / base

    sv = np.linalg.svd(S, compute_uv=False)
    _, _, vt = np.linalg.svd(S)
    print(f"  sensitivity singular values  {sv[0]:.4e}  {sv[1]:.4e}  {sv[2]:.4e}")
    print(f"  ratio sigma_max / sigma_min  {sv[0] / max(sv[-1], 1e-300):.3e}")
    print()
    v = vt[-1]
    v = v * np.sign(v[np.argmax(np.abs(v))])
    print("  THE UNSEEN DIRECTION (right singular vector of sigma_min), in log space:")
    for name, comp in zip(IDENT_PARAMS, v):
        print(f"    {name:6s} {comp:+.6f}")
    print(f"  against an equal-scaling prediction of {1 / math.sqrt(3):+.6f} each")
    print()
    print("  TEST OF IT, by construction rather than by inspection: scale all")
    print("  three by 1.5 at once and see which modes move.")
    moved = _modes_with(ac, np.full(3, 1.5), alpha, elevator, throttle, V, H)
    print(f"    {moved[0]:.6f} {moved[1]:.6f} {moved[2]:.6f} {moved[3]:.6f}")
    for label, i in (("phugoid wn", 0), ("phugoid zeta", 1),
                     ("short period wn", 2), ("short period zeta", 3)):
        print(f"    {label:20s} moves {moved[i] / base[i] - 1:+.3e}")
    print()
    print("  ^ THE SHORT PERIOD IS INVARIANT AND THE PHUGOID IS NOT, and that is")
    print("    the finding rather than a defect in the probe. Scaling Cma and Cmq")
    print("    without Cm0 and Cmde breaks the moment balance at the frozen trim")
    print("    point, so C_m is no longer zero there and an M_u appears -- and the")
    print("    phugoid is the mode that speed derivatives dominate (section 4's own")
    print("    localisation). The unobservable direction is exact for the mode the")
    print("    CAT work depends on and only approximate for the whole 4-state model.")
    print()
    print("  THE SAME SVD ON THE SHORT PERIOD ALONE:")
    S_sp = np.zeros((2, 3))
    for j in range(3):
        hi, lo = np.ones(3), np.ones(3)
        hi[j], lo[j] = 1.0 + REL_STEP, 1.0 - REL_STEP
        up = _modes_with(ac, hi, alpha, elevator, throttle, V, H)[2:]
        dn = _modes_with(ac, lo, alpha, elevator, throttle, V, H)[2:]
        S_sp[:, j] = (up - dn) / (2.0 * REL_STEP) / base[2:]
    sv_sp = np.linalg.svd(S_sp, compute_uv=False)
    _, _, vt_sp = np.linalg.svd(S_sp)
    v_sp = vt_sp[-1]
    v_sp = v_sp * np.sign(v_sp[np.argmax(np.abs(v_sp))])
    print(f"    singular values  {sv_sp[0]:.4e}  {sv_sp[1]:.4e}")
    print(f"    unseen direction  " + "  ".join(
        f"{n} {c:+.6f}" for n, c in zip(IDENT_PARAMS, v_sp)))
    print(f"    against equal scaling {1 / math.sqrt(3):+.6f} each")
    print(f"    angle from equal scaling: "
          f"{math.degrees(math.acos(min(1.0, abs(v_sp @ np.full(3, 1 / math.sqrt(3)))))):.4f} deg")
    print()
    print("    READ THIS ROW CORRECTLY. Two observables against three parameters")
    print("    is a 2x3 matrix, so a null direction EXISTS by shape and is not")
    print("    itself evidence of anything. What is evidence is WHERE it points:")
    print("    within a fifth of a degree of the equal-scaling combination that")
    print("    M_alpha = Cma*qbar*S*cbar/Iyy and M_q = Cmq*qbar*S*cbar^2/(2*V*Iyy)")
    print("    predict analytically. The SVD was not told that and recovers it.")


if __name__ == "__main__":
    main()
