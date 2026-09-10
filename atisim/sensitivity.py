"""How much does a result depend on the model's own numbers?

WHY THIS MODULE EXISTS. Section 4 of PROJECT.md measures the model against
SOURCES. Nothing in it measures the model against ITSELF, so no row there says
whether the headline gust load is set by `CLa` or by `Cmq`, nor whether a
modelling choice from `ASSUMPTIONS.md` costs more than a derivative does. This
module supplies the derivative of a result with respect to a coefficient, which
is the object all three of those questions are asking for.

Design: docs/superpowers/specs/2026-09-10-model-sensitivity-analysis-design.md.
Nothing here is evidence on its own -- a number produced by this module belongs
in section 4 with the run that produced it.

WHAT IS REPORTED, AND WHY IT IS NOT THE GRADIENT. `Cmq` is per radian and `mass`
is in kilograms, so their gradients are not comparable and a table sorted by
|dQ/dp| would rank by units. The ELASTICITY (dQ/dp)(p/Q) is dimensionless: it is
the per-cent change in the answer per per-cent change in the input, and it is
the only form in which a derivative and a mass belong in the same ranking. Raw
gradients are returned beside it, never ranked.

THE THREE OBSTACLES BETWEEN `jax.jacfwd` AND AN ANSWER. Each is a property of
this codebase rather than of sensitivity analysis, and each has its own check in
test_sensitivity.py:

  1. TRIM IS A NEWTON SOLVE. Every quantity worth differentiating is defined at
     the re-trimmed condition -- `validation.sweep` re-trims at every sample
     because changing a derivative moves the trim point, and comparing across
     two different trims confounds the coefficient with the condition.
     Differentiating through the unrolled loop would work but is wasteful and is
     silently wrong if the loop has not converged. `implicit_trim_jacobian` uses
     the implicit function theorem on the residual instead, which needs no
     iteration count and is exact at the solution.

  2. THE MODE CHAIN BREAKS AT NUMPY. `validation.longitudinal_matrix` is itself
     a `jacfwd` of the real dynamics, so A(p) differentiates cleanly -- and then
     `validation.modes_from_matrix` calls `np.linalg.eigvals`, which JAX cannot
     see through. `eigenvalue_sensitivity` closes the gap with the first-order
     perturbation dlambda = diag(W dA V), W = inv(V), rather than by writing a
     differentiable eig. It is exact to first order and it FAILS AT REPEATED OR
     NEARLY REPEATED EIGENVALUES, so `eigenvalue_separation` is reported beside
     every result and the caller is expected to look at it.

  3. THE `Aircraft` TUPLE IS NOT A SET OF INDEPENDENT PARAMETERS, and a naive
     `jacfwd` over the whole of it is therefore WRONG. `inertia_inv` is the
     inverse of `inertia`; `AR` is b^2/S. Perturbing either half of such a pair
     alone produces an airframe that does not exist. `INDEPENDENT_FIELDS` is the
     subset that may be varied one at a time, and `COUPLED_FIELDS` records what
     was excluded and what it is tied to, so the exclusion is a declared
     modelling statement rather than an oversight.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from atisim import trim
from atisim.aircraft import Aircraft
from atisim.dynamics import derivatives
from atisim.state import Controls, State, euler_to_quat

# --------------------------------------------------------------------------
# Which fields may be varied one at a time
# --------------------------------------------------------------------------

# Algebraically dependent groups. Varying one member alone builds an airframe
# whose stored quantities contradict each other -- an `inertia_inv` that is not
# the inverse of its `inertia`, or an `AR` that is not b^2/S -- and the
# resulting derivative is of nothing physical. Recorded rather than silently
# skipped, because "why is `mass` in the screen and `inertia` not" is exactly
# the question a reader of the table will ask.
COUPLED_FIELDS: dict[str, str] = {
    "inertia": "inertia_inv is its inverse; vary the pair through a reparameterisation, not alone",
    "inertia_inv": "precomputed from inertia",
    "AR": "b**2 / S, so it moves with either",
    "S": "AR = b**2 / S, so S cannot move alone",
    "b": "AR = b**2 / S, so b cannot move alone",
}

# Scalar fields that carry a source table and are independent of each other.
# `mass` is here and the inertia is not, for the reason above.
INDEPENDENT_FIELDS: tuple[str, ...] = (
    "mass", "c",
    "CD0", "e", "sweep", "t_over_c", "kappa_airfoil",
    "CL0", "CLa", "CLq", "CLde",
    "Cm0", "Cma", "Cmq", "Cmde",
    "CYb", "CYp", "CYr", "CYdr",
    "Clb", "Clp", "Clr", "Clda", "Cldr",
    "Cnb", "Cnp", "Cnr", "Cnda", "Cndr",
    "max_thrust", "thrust_lapse",
)

LONGITUDINAL_FIELDS: tuple[str, ...] = (
    "mass", "c", "CD0", "e", "CL0", "CLa", "CLq", "CLde",
    "Cm0", "Cma", "Cmq", "Cmde", "max_thrust", "thrust_lapse",
)

LATERAL_FIELDS: tuple[str, ...] = (
    "CYb", "CYp", "CYr", "CYdr",
    "Clb", "Clp", "Clr", "Clda", "Cldr",
    "Cnb", "Cnp", "Cnr", "Cnda", "Cndr",
)


def _check_fields(fields):
    bad = [f for f in fields if f in COUPLED_FIELDS]
    if bad:
        raise ValueError(
            f"{bad} cannot be varied alone: "
            + "; ".join(f"{f} -- {COUPLED_FIELDS[f]}" for f in bad)
        )
    unknown = [f for f in fields if f not in Aircraft._fields]
    if unknown:
        raise ValueError(f"not fields of Aircraft: {unknown}")


# --------------------------------------------------------------------------
# 1. The trim solve, differentiated implicitly
# --------------------------------------------------------------------------


def implicit_trim_jacobian(ac: Aircraft, V: float, H: float, fields=None,
                           x: jnp.ndarray = None) -> dict[str, np.ndarray]:
    """d[alpha, elevator, throttle]/dp at the trim solution, per field.

    `trim.trim` solves r(x, V, H, p) = 0 by Newton. At the solution the implicit
    function theorem gives

        dx*/dp = - (dr/dx)^-1 (dr/dp)

    with both blocks available from `jax.jacfwd` of `trim.residual`, which is the
    same residual the solver itself differentiates. No iteration count enters,
    and the result is exact AT the solution rather than at wherever the loop
    happened to stop -- which is the failure differentiating through the unrolled
    scan would hide.

    A LATERAL DERIVATIVE MUST RETURN EXACTLY ZERO HERE. The trim is wings-level
    with beta = 0 and no body rates, so no lateral coefficient enters any of
    [udot, wdot, qdot]. That is the cheapest available check that this function
    is differentiating the thing it claims to, and test_sensitivity.py asserts it
    as an exact zero rather than as a small number.
    """
    fields = INDEPENDENT_FIELDS if fields is None else tuple(fields)
    _check_fields(fields)
    V, H = jnp.array(float(V)), jnp.array(float(H))
    if x is None:
        x, _ = trim.trim(V, H, ac)

    dr_dx = jax.jacfwd(trim.residual, argnums=0)(x, V, H, ac)
    dr_dp = jax.jacfwd(trim.residual, argnums=3)(x, V, H, ac)

    out = {}
    for f in fields:
        block = np.asarray(getattr(dr_dp, f))  # (3,) + field shape
        flat = block.reshape(3, -1)
        out[f] = np.asarray(-np.linalg.solve(np.asarray(dr_dx), flat)).reshape(block.shape)
    return out


# --------------------------------------------------------------------------
# 2. The plant matrices, in jnp so they can be differentiated
# --------------------------------------------------------------------------
#
# These are jnp TWINS of `validation.longitudinal_matrix` and of the matrix
# built inside `validation.lateral_modes`. Both of those end in `np.asarray`,
# which is where the AD chain stops, and neither takes a traced `alpha` because
# both use `np.cos`.
#
# TWO PATHS THAT COMPUTE THE SAME THING CAN DRIFT APART, and this repository says
# so elsewhere -- `vortex_viz._measure` exists precisely so the three Fig. 8
# points cannot. The guard here is not a comment: test_sensitivity.py asserts
# these return the same matrices as `validation`'s, element for element, and
# reports the largest difference when it is not exact.


def longitudinal_matrix_jnp(ac: Aircraft, alpha, elevator, throttle, V, H):
    """Body-axis [u, w, q, theta] plant matrix, differentiable in `ac` and the trim."""
    alpha = jnp.asarray(alpha)
    u0, w0 = V * jnp.cos(alpha), V * jnp.sin(alpha)
    controls = Controls(
        elevator=jnp.asarray(elevator), aileron=jnp.array(0.0),
        rudder=jnp.array(0.0), throttle=jnp.asarray(throttle),
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

    return jax.jacfwd(f)(jnp.array([u0, w0, 0.0, alpha]))


def lateral_matrix_jnp(ac: Aircraft, alpha, elevator, throttle, V, H):
    """Body-axis [v, p, r, phi] plant matrix, differentiable in `ac` and the trim.

    The exact kinematic phidot term `p + r cos(phi) tan(theta0)` is kept, for the
    reason `validation.lateral_modes` gives: at this aircraft's trim attitude
    dropping it moves the SPIRAL root by order 30%.
    """
    alpha = jnp.asarray(alpha)
    theta0 = alpha
    u0, w0 = V * jnp.cos(alpha), V * jnp.sin(alpha)
    controls = Controls(
        elevator=jnp.asarray(elevator), aileron=jnp.array(0.0),
        rudder=jnp.array(0.0), throttle=jnp.asarray(throttle),
    )

    def f(x):
        v, p, r, phi = x
        state = State(
            pos_ned=jnp.array([0.0, 0.0, -H]),
            vel_body=jnp.array([u0, v, w0]),
            quat=euler_to_quat(phi, theta0, jnp.array(0.0)),
            omega=jnp.array([p, 0.0, r]),
        )
        d = derivatives(state, controls, ac, jnp.zeros(3), jnp.zeros(3))
        phidot = p + r * jnp.cos(phi) * jnp.tan(theta0)
        return jnp.array([d.vel_body[1], d.omega[0], d.omega[2], phidot])

    return jax.jacfwd(f)(jnp.zeros(4))


# --------------------------------------------------------------------------
# 3. The total derivative of the plant matrix, trim path included
# --------------------------------------------------------------------------


def plant_matrix_sensitivity(ac: Aircraft, V: float, H: float, *, axis: str,
                             fields=None):
    """(A, {field: dA/dp}, trim vector, trim residual), re-trim included.

    A depends on a coefficient TWICE: directly, and through the trim point the
    matrix is built about. The chain rule over both is

        dA/dp = (dA/dp)|_x + (dA/dx) (dx*/dp)

    and omitting the second term is the error `validation.sweep` avoids by
    re-trimming. Measured on the 747 approach it is not a rounding correction:
    section 4 will carry what it is worth per field.
    """
    fields = INDEPENDENT_FIELDS if fields is None else tuple(fields)
    _check_fields(fields)
    builder = {"longitudinal": longitudinal_matrix_jnp,
               "lateral": lateral_matrix_jnp}[axis]

    Vf, Hf = float(V), float(H)
    x, res = trim.trim(jnp.array(Vf), jnp.array(Hf), ac)
    if not bool(trim.is_physical(x, ac)):
        raise ValueError("the base trim is not a flight condition")
    alpha, elevator, throttle = x

    A = np.asarray(builder(ac, alpha, elevator, throttle, Vf, Hf))

    # Partial in the coefficients, holding the trim fixed.
    dA_dp = jax.jacfwd(lambda a: builder(a, alpha, elevator, throttle, Vf, Hf))(ac)
    # Partial in the trim vector, holding the coefficients fixed: (4, 4, 3).
    dA_dx = np.asarray(
        jax.jacfwd(lambda xx: builder(ac, xx[0], xx[1], xx[2], Vf, Hf))(x)
    )
    dx_dp = implicit_trim_jacobian(ac, Vf, Hf, fields=fields, x=x)

    out = {}
    for f in fields:
        direct = np.asarray(getattr(dA_dp, f))          # (4, 4) + field shape
        through = np.tensordot(dA_dx, dx_dp[f], axes=([2], [0]))
        out[f] = direct + through
    return A, out, np.asarray(x), np.asarray(res)


# --------------------------------------------------------------------------
# 4. Eigenvalues, perturbed rather than re-solved
# --------------------------------------------------------------------------


class ModeSensitivity(NamedTuple):
    """One eigenvalue, its (wn, zeta) or tau, and their derivatives w.r.t. one field."""

    lam: complex
    dlam: complex
    wn: float
    dwn: float
    zeta: float
    dzeta: float
    tau: float          # -1/Re(lam); meaningful for the real roots
    dtau: float


def eigenvalue_separation(A) -> float:
    """Smallest gap between any two eigenvalues of A, in the same units as A.

    First-order eigenvalue perturbation divides by the gap in effect: as two
    eigenvalues approach each other the individual dlambda blow up while their
    SUM stays finite. This number is what says whether the result may be quoted,
    and it is returned with every sensitivity rather than checked internally --
    an automatic threshold here would be a modelling choice made silently.
    """
    lam = np.linalg.eigvals(np.asarray(A))
    gaps = [abs(lam[i] - lam[j]) for i in range(len(lam)) for j in range(i + 1, len(lam))]
    return float(min(gaps))


def eigenvalue_sensitivity(A, dA):
    """(eigenvalues, dlambda/dp) by first-order perturbation.

    For a simple eigenvalue with right eigenvector v_i and left eigenvector w_i
    normalised so that W = V^-1,

        dlambda_i = (W dA V)_ii

    which is the standard result and needs no left-eigenvector solve of its own.
    Exact to first order in dA; see `eigenvalue_separation` for when that is not
    good enough.
    """
    A, dA = np.asarray(A), np.asarray(dA)
    lam, Vr = np.linalg.eig(A)
    W = np.linalg.inv(Vr)
    return lam, np.einsum("ij,jk,ki->i", W, dA, Vr)


def _mode_from(lam, dlam) -> ModeSensitivity:
    """(wn, zeta, tau) and their derivatives, from one eigenvalue and its dlambda."""
    wn = abs(lam)
    dwn = float((lam.real * dlam.real + lam.imag * dlam.imag) / wn)
    zeta = -lam.real / wn
    dzeta = float(-dlam.real / wn + lam.real * dwn / wn**2)
    tau = -1.0 / lam.real
    dtau = float(dlam.real / lam.real**2)
    return ModeSensitivity(lam=lam, dlam=dlam, wn=float(wn), dwn=dwn,
                           zeta=float(zeta), dzeta=dzeta, tau=float(tau), dtau=dtau)


def mode_sensitivity(ac: Aircraft, V: float, H: float, *, axis: str, fields=None):
    """({field: [ModeSensitivity per eigenvalue]}, A, eigenvalues, separation, trim, residual).

    The eigenvalues come back in `np.linalg.eig` order, which is not sorted --
    `longitudinal_modes` sorts by wn and `lateral_modes` sorts the real roots by
    |tau|, and neither ordering survives differentiation cleanly, because a sort
    is what makes a swept quantity discontinuous where two modes cross. Callers
    pick the root they mean by a predicate (see `pick`), which fails loudly when
    the predicate stops matching exactly one root.
    """
    A, dA, x, res = plant_matrix_sensitivity(ac, V, H, axis=axis, fields=fields)
    lam, _ = eigenvalue_sensitivity(A, np.zeros_like(A))
    out = {}
    for f, dAf in dA.items():
        _, dlam = eigenvalue_sensitivity(A, dAf)
        out[f] = [_mode_from(l, d) for l, d in zip(lam, dlam)]
    return out, A, lam, float(eigenvalue_separation(A)), x, res


def pick(mode_list, predicate, what: str):
    """The one entry matching `predicate`, or an error naming how many matched.

    Selecting a mode by "the oscillatory one" or "the fast real root" is the step
    at which a sensitivity study quietly starts reporting a different mode than
    the one it names. Anything other than exactly one match is an error here
    rather than a silent `[0]`.
    """
    hits = [m for m in mode_list if predicate(m)]
    if len(hits) != 1:
        raise ValueError(f"{what}: {len(hits)} roots matched, expected exactly 1")
    return hits[0]


def oscillatory(m: ModeSensitivity) -> bool:
    return m.lam.imag > 1e-9


def real_root(m: ModeSensitivity) -> bool:
    return abs(m.lam.imag) <= 1e-9


def oscillatory_modes(mode_list):
    """The imag > 0 roots, sorted low-to-high wn -- `longitudinal_modes`' order.

    Sorted the same way as `validation.modes_from_matrix` so "the phugoid" means
    the same root in both places. The sort is also the reason a swept quantity
    goes discontinuous where two modes cross, which is why `mode_sensitivity`
    returns roots unsorted and this is applied by the caller, in the open.
    """
    return sorted([m for m in mode_list if oscillatory(m)], key=lambda m: m.wn)


def real_modes(mode_list):
    """The real roots, sorted by |tau| -- `lateral_modes`' order, and for its reason.

    `lateral_modes` sorts by ABSOLUTE time constant because an unstable spiral has
    a negative tau, and a signed sort would put it in front of every positive one
    and return it as the roll mode.
    """
    return sorted([m for m in mode_list if real_root(m)], key=lambda m: abs(m.tau))


# --------------------------------------------------------------------------
# 5. Elasticity
# --------------------------------------------------------------------------


def elasticity(dQ_dp: float, p: float, Q: float) -> float:
    """(dQ/dp)(p/Q): per-cent of the answer per per-cent of the input.

    Dimensionless, so a per-radian derivative and a mass in kilograms are
    rankable against each other -- which a table of raw gradients is not.
    Undefined where Q or p is zero, and returns NaN there rather than a large
    number that would sort to the top of a ranking.
    """
    if Q == 0.0 or p == 0.0:
        return float("nan")
    return float(dQ_dp * p / Q)
