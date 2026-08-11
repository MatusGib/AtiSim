"""Validation against analytic laws and published worked examples.

Tier 1 checks a coefficient sweep against a closed-form relation; tier 2 checks
the model against a source that published both its inputs and its own computed
outputs. Neither can be invalidated by a source's age: if CR-2144's derivatives
were 10% from the real aeroplane, this model must still reproduce CR-2144's own
transfer-function factors from CR-2144's own derivatives. See the design spec,
"Source qualification".

Tier 0 -- the checks needing no aircraft data at all -- is flightsim/verification.py.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from flightsim.aircraft import Aircraft
from flightsim.dynamics import derivatives
from flightsim.state import Controls, State, euler_to_quat
from flightsim.units import FT2M


def longitudinal_matrix(ac: Aircraft, alpha: float, elevator: float,
                        throttle: float, V: float, H: float):
    """Body-axis plant matrix in [u, w, q, theta], by jacfwd of the real dynamics.

    Body axes, so theta0 = alpha0 and w0 = V sin(alpha0) are both non-zero. Most
    textbook longitudinal matrices are quoted in STABILITY axes, where Theta0 = 0
    -- see `to_stability_axes`, which is what makes them comparable at all.
    """
    u0, w0 = V * np.cos(alpha), V * np.sin(alpha)
    controls = Controls(
        elevator=jnp.array(elevator), aileron=jnp.array(0.0),
        rudder=jnp.array(0.0), throttle=jnp.array(throttle),
    )

    def f(x):
        u, w, q, theta = x
        state = State(
            pos_ned=jnp.array([0.0, 0.0, -H]),
            vel_body=jnp.array([u, 0.0, w]),
            quat=euler_to_quat(jnp.array(0.0), theta, jnp.array(0.0)),
            omega=jnp.array([0.0, q, 0.0]),
        )
        d = derivatives(state, controls, ac, jnp.zeros(3), jnp.zeros(3))
        return jnp.array([d.vel_body[0], d.vel_body[2], d.omega[1], q])

    return np.asarray(jax.jacfwd(f)(jnp.array([u0, w0, 0.0, alpha])))


def to_stability_axes(A, alpha):
    """Rotate a [u, w, q, theta] plant matrix from body to stability axes.

    Stability axes are body axes turned by the trim angle of attack, so the
    perturbation velocities mix and q and theta do not. That is a similarity
    transform: every element moves, the eigenvalues do not. Both halves are
    asserted in test_validation.py.

    Needed because published matrices state Theta0 = 0, which is a stability-axis
    statement -- compared raw against a body-axis matrix the (2, 4) element reads
    -g sin(alpha0) against a published 0 and looks like a defect.
    """
    ca, sa = np.cos(alpha), np.sin(alpha)
    T = np.array([[ca, sa, 0.0, 0.0],
                  [-sa, ca, 0.0, 0.0],
                  [0.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0, 1.0]])
    return T @ A @ np.linalg.inv(T)


def to_imperial_matrix(A):
    """A [u, w, q, theta] plant matrix from SI into ft/s-rad units.

    The state mixes dimensions, so each element converts differently. The five
    rate elements (0,0) (0,1) (1,0) (1,1) (2,2) are 1/s and unchanged; (3,2) is
    dimensionless. The rest:

        A[0,2]  m/s     -> ft/s      (du_dot/dq)
        A[0,3]  m/s^2   -> ft/s^2    (du_dot/dtheta)
        A[1,2]  m/s     -> ft/s      (dw_dot/dq)
        A[1,3]  m/s^2   -> ft/s^2    (dw_dot/dtheta)
        A[2,0]  1/(m.s) -> 1/(ft.s)  (dq_dot/du)
        A[2,1]  1/(m.s) -> 1/(ft.s)  (dq_dot/dw)

    All six are converted, not only the four that a published comparison happens
    to exercise: A[0,2] and A[1,3] compare against zeros in the tests and would
    fail nothing, but the notebook prints a per-element difference column and an
    incomplete conversion would mislead there.
    """
    out = np.array(A, dtype=float, copy=True)
    for i, j in ((0, 2), (0, 3), (1, 2), (1, 3)):
        out[i, j] /= FT2M
    for i, j in ((2, 0), (2, 1)):
        out[i, j] *= FT2M
    return out


def modes_from_matrix(A):
    """(wn, zeta) for every oscillatory root, sorted low-to-high wn."""
    eig = np.linalg.eigvals(A)
    return sorted((abs(lam), -lam.real / abs(lam)) for lam in eig if lam.imag > 1e-9)


def longitudinal_modes(ac: Aircraft, alpha: float, elevator: float,
                       throttle: float, V: float, H: float):
    """Phugoid and short-period (wn, zeta), sorted low-to-high wn.

    Linearises in body-axis [u, w, q, theta] about the given trim, exactly as
    scripts/checkpoint.py's longitudinal_modes does for the 747.
    """
    return modes_from_matrix(longitudinal_matrix(ac, alpha, elevator, throttle, V, H))


def lateral_modes(ac: Aircraft, alpha: float, elevator: float, throttle: float,
                  V: float, H: float):
    """Dutch roll (wn, zeta), roll-subsidence time constant, spiral time constant.

    4-state reduction [v, p, r, phi] about the wings-level trim, holding
    u = u0, w = w0, theta = theta0 fixed (standard small-perturbation lateral
    split, valid because the trim is wings-level and symmetric).

    thetadot's exact kinematic term keeps r*cos(phi)*tan(theta0), not just p:
    at this aircraft's trim pitch attitude that term is NOT negligible for the
    spiral root specifically (dropping it changes the spiral time constant by
    order 30%, confirmed by directly perturbing the trimmed nonlinear sim in
    bank and fitting the late-time decay of phi -- see
    test_cr2144_lateral_modes.py). Dutch roll and roll are insensitive to it.
    """
    theta0 = alpha
    u0, w0 = V * np.cos(alpha), V * np.sin(alpha)
    controls = Controls(
        elevator=jnp.array(elevator), aileron=jnp.array(0.0),
        rudder=jnp.array(0.0), throttle=jnp.array(throttle),
    )

    def f(x):
        v, p, r, phi = x
        state = State(
            pos_ned=jnp.array([0.0, 0.0, -H]),
            vel_body=jnp.array([u0, v, w0]),
            quat=euler_to_quat(phi, jnp.array(theta0), jnp.array(0.0)),
            omega=jnp.array([p, 0.0, r]),
        )
        d = derivatives(state, controls, ac, jnp.zeros(3), jnp.zeros(3))
        phidot = p + r * jnp.cos(phi) * jnp.tan(theta0)
        return jnp.array([d.vel_body[1], d.omega[0], d.omega[2], phidot])

    A = np.asarray(jax.jacfwd(f)(jnp.array([0.0, 0.0, 0.0, 0.0])))
    eig = np.linalg.eigvals(A)
    dutch_roll = None
    reals = []
    for lam in eig:
        if lam.imag > 1e-9:
            dutch_roll = (abs(lam), -lam.real / abs(lam))
        elif abs(lam.imag) <= 1e-9:
            reals.append(-1.0 / lam.real)
    reals.sort()  # roll subsidence is fast (small tau), spiral is slow
    roll_tau, spiral_tau = reals[0], reals[1]
    return dutch_roll, roll_tau, spiral_tau


class Reference(NamedTuple):
    """A published number and the document it came from.

    `source` is not optional and not a comment. PROJECT.md's standing rule is
    that a figure without its table has broken the project, and a NamedTuple
    field is the cheapest way to make an uncited number unconstructible.
    """

    value: float
    source: str


# D. A. Caughey, "Introduction to Aircraft Stability and Control", Cornell
# MAE 5070 course notes, Chapter 5. Its Eq. (5.48)-(5.50) cite Heffley & Jewell,
# NASA CR-2144 -- the same document this project transcribed for
# `boeing747_approach` -- so Caughey is NOT an independent data source. He is an
# independent IMPLEMENTATION: he formed the dimensional derivatives, the plant
# matrix, the characteristic polynomial and the roots himself and published all
# four. For checking a solver, that is the useful kind of independence.
_C = "Caughey, Cornell MAE 5070 notes, Ch. 5, {}"
_IX2 = "NASA CR-2144, Heffley & Jewell 1972, Table IX-2 (747 power approach)"
_IX5 = "NASA CR-2144 Table IX-5 via PROJECT.md section 4, '747 modes vs CR-2144'"

REFERENCES = {
    # Eq. (5.54): the roots of the power-approach longitudinal system.
    "747pa_short_period_wn": Reference(0.88178, _C.format("Eq. (5.54)")),
    "747pa_short_period_zeta": Reference(0.62546, _C.format("Eq. (5.54)")),
    "747pa_phugoid_wn": Reference(0.13391, _C.format("Eq. (5.54)")),
    "747pa_phugoid_zeta": Reference(0.01329, _C.format("Eq. (5.54)")),
    # Eq. (5.51): dimensional derivatives, ft-s-rad. Mq is the one this model can
    # be expected to match exactly -- it has no alpha-dot content. Zwdot and Mwdot
    # are the two this model excludes by design, and are what the residuals in
    # test_the_omitted_alpha_dot_terms_are_recoverable are reconstructed from.
    # Mwdot is quoted to ONE significant figure, which bounds how well any
    # reconstruction using it can do -- see that test's tolerances.
    "747pa_Mq": Reference(-0.4381, _C.format("Eq. (5.51)")),
    "747pa_Zwdot": Reference(-0.0341, _C.format("Eq. (5.51)")),
    "747pa_Mwdot": Reference(-0.0002, _C.format("Eq. (5.51)")),
    # Table IX-2 as transcribed into aircraft.boeing747_approach.
    "747pa_CL": Reference(1.11, _IX2),
    "747pa_CD": Reference(0.102, _IX2),
    "747pa_Cma": Reference(-1.26, _IX2),
    # The CRUISE comparison, for the condition-dependence test. Model and
    # reference both, so the error can be recomputed rather than hard-coded.
    "747cruise_phugoid_wn_ref": Reference(0.0673, _IX5),
    "747cruise_phugoid_wn_model": Reference(0.0553, _IX5),
    "747cruise_short_period_zeta_ref": Reference(0.387, _IX5),
    "747cruise_short_period_zeta_model": Reference(0.3425, _IX5),
}

# Caughey Eq. (5.52), the [u, w, q, theta] plant matrix in ft/s and radians.
# Stability axes: he states Theta0 = 0, which is only true there.
CAUGHEY_A = np.array([
    [-0.0212,  0.0466,   0.000,   -32.174],
    [-0.2229, -0.5839, 262.472,     0.0],
    [ 0.0001, -0.0018,  -0.5015,    0.0],
    [ 0.0,     0.0,      1.0,       0.0],
])

TRIM_RESIDUAL_LIMIT = 1e-9

# The |alpha| bound is `trim.is_physical`, and it lives there rather than here.
# Session 11 put it in this module because this is where it was needed; the
# defect is in `trim.trim`, which returns an absurd root and says nothing, so
# every other caller was equally exposed. One bound, one place.


def sweep(ac: Aircraft, field: str, values, quantity, V: float, H: float):
    """Vary one coefficient and report a scalar per value.

    `Aircraft` is a NamedTuple, so `_replace` gives an independent airframe per
    sample with no mutation and no copy discipline to get wrong.

    The aircraft is RE-TRIMMED at every sample, because changing a derivative
    moves the trim point and comparing modes across different trims would
    confound the two. Both the residual AND the resulting angle of attack are
    checked, the latter through `trim.is_physical` -- convergence and sense are
    different questions and only the first is what a residual measures.

    `quantity` takes (aircraft, alpha, elevator, throttle) and returns a float.
    """
    from flightsim.trim import is_physical
    from flightsim.trim import trim as solve_trim

    out = []
    for v in values:
        swept = ac._replace(**{field: jnp.array(float(v))})
        x, res = solve_trim(jnp.array(V), jnp.array(H), swept)
        residual_norm = float(jnp.linalg.norm(res))
        if residual_norm > TRIM_RESIDUAL_LIMIT:
            raise RuntimeError(f"{field}={v} did not trim: residual {residual_norm:.3e}")
        alpha = float(x[0])
        if not is_physical(x):
            raise RuntimeError(
                f"{field}={v} did not trim: alpha {np.degrees(alpha):.1f} deg is "
                f"outside the linear-aero range (PROJECT.md section 7)"
            )
        out.append(quantity(swept, alpha, float(x[1]), float(x[2])))
    return np.array(out)


def affine_fit(xs, ys):
    """Least-squares (slope, intercept, worst relative residual) of ys against xs.

    Every mode/derivative relation this project sweeps turns out to be AFFINE
    rather than proportional, and the intercept is the physically interesting
    part: it is the term the textbook approximation drops. Short-period wn^2 has
    a non-zero intercept at Cm_alpha = 0 because Z_alpha*M_q/u0 survives there;
    Dutch-roll wn^2 has one because of Y_beta/u0; 1/tau_roll has one because of
    Ixz roll-yaw coupling. Reporting the residual alongside is what makes "this
    law holds" a measurement rather than an assertion.
    """
    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    slope, intercept = np.polyfit(xs, ys, 1)
    predicted = intercept + slope * xs
    worst = float(np.max(np.abs(predicted - ys) / np.abs(ys)))
    return float(slope), float(intercept), worst
