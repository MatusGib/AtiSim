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

from atisim import dynamics, integrate, trim, wind
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


# --------------------------------------------------------------------------
# 6. The trim solve as a differentiable primitive
# --------------------------------------------------------------------------


@jax.custom_jvp
def solved_trim(V, H, ac: Aircraft):
    """`trim.trim`'s solution, carrying the implicit derivative as its own JVP.

    `implicit_trim_jacobian` gives dx*/dp as a table, which is what a screen
    wants. This gives the same thing as a DIFFERENTIABLE FUNCTION, which is what
    a quantity defined downstream of the trim -- a flown load, say -- needs in
    order to be differentiated at all.

    Differentiating straight through the 40 unrolled Newton steps also works and
    is asserted here to agree, but it is the wrong object twice over: it costs 40
    linear solves per tangent, and it silently returns the derivative of
    "wherever the loop stopped" rather than of the solution if the loop has not
    converged.
    """
    x, _ = trim.trim(V, H, ac)
    return x


@solved_trim.defjvp
def _solved_trim_jvp(primals, tangents):
    """r(x*, V, H, p) = 0 differentiated: (dr/dx) dx + dr|_x = 0."""
    V, H, ac = primals
    dV, dH, dac = tangents
    x = solved_trim(V, H, ac)
    dr_dx = jax.jacfwd(trim.residual, argnums=0)(x, V, H, ac)
    _, dr = jax.jvp(trim.residual, (x, V, H, ac),
                    (jnp.zeros_like(x), dV, dH, dac))
    return x, -jnp.linalg.solve(dr_dx, dr)


# --------------------------------------------------------------------------
# 7. The flown load, differentiably
# --------------------------------------------------------------------------
#
# `vortex_viz._measure` ends in `np.asarray(jax.vmap(analyse)(...))`, so the
# whole `Encounter` path is an AD dead end. This is the second path, and the
# comment on the plant matrices applies here with more force: it must be
# asserted equal to `_measure`'s `n_z` channel, sample for sample, or every
# elasticity taken through it is about a load this project does not report.
#
# TWO DECLARED CHOICES, both of which change what the derivative MEANS:
#
#   1. THE ANALYSIS WINDOW IS HELD FIXED at the base point's trajectory. The
#      window is defined by north position, so a boolean mask over it is a step
#      function of every coefficient and has no derivative. Fixing the mask asks
#      "how does the load inside THIS window move", which is the question the
#      headline number is an answer to. The window edges sit 2 r0 beyond the
#      outermost core, so the peak is nowhere near them.
#
#   2. PEAK-TO-PEAK IS max - min, WHICH IS DIFFERENTIABLE ALMOST EVERYWHERE AND
#      NOT EVERYWHERE. Its gradient is the gradient of the two samples that
#      happen to be the extremes, and it JUMPS when an extreme moves from one
#      core to another. That is not a flaw to be worked around -- it is a
#      property of the statistic, and `scripts/sensitivity_load.py` measures how
#      far the tangent survives by comparing it against a finite excursion.


def load_history(ac: Aircraft, wind_model, V, H, *, start_north, n_steps, dt,
                 moving_air: bool = True, load_model=None, key_seed: int = 0,
                 stage_sampled: bool = False):
    """n_z per sample -- the differentiable twin of `vortex_viz._measure`'s n_z.

    Mirrors `fly_in_moving_air` -> `fly_from_state` -> `_measure` exactly,
    including re-evaluating the wind model at each recorded state rather than
    reading the logged value: that is what `_measure` does, and it is exact for
    a deterministic field. `wind_model` is passed already built, because
    `integrate.rollout` takes it as a STATIC argument and a fresh closure per
    call would recompile the scan every time.
    """
    V, H, dt = jnp.asarray(V), jnp.asarray(H), jnp.asarray(dt)
    x = solved_trim(V, H, ac)
    controls = trim.trimmed_controls(x[1], x[2])

    state = trim.trimmed_state(x[0], V, H)
    state = state._replace(
        pos_ned=jnp.array([start_north, 0.0, 0.0]) + jnp.array([0.0, 0.0, -H])
    )
    if moving_air:
        # `relative_velocity` is `vel_body - dcm.T @ wind_ned`, so adding the
        # same term makes the air-relative velocity EXACTLY the still-air trim
        # value. `fly_in_moving_air`'s docstring has why this is not optional on
        # Mehta's array: at a 12 r0 lead the superposed far field is 5.28 m/s.
        from atisim.state import quat_to_dcm
        wind_ned0, *_ = wind_model(
            wind.zero_wind_state(), state, jax.random.PRNGKey(key_seed), dt
        )
        state = state._replace(
            vel_body=state.vel_body + quat_to_dcm(state.quat).T @ wind_ned0
        )

    _, hist = integrate.rollout(
        integrate.init_sim(state, jax.random.PRNGKey(key_seed)),
        controls, dt, ac, n_steps, wind_model=wind_model, load_model=load_model,
        stage_sampled=stage_sampled,
    )

    def n_z_of(pos_ned, vel_body, quat, omega):
        s = State(pos_ned=pos_ned, vel_body=vel_body, quat=quat, omega=omega)
        wind_ned, omega_gust, _, _, _ = wind_model(
            wind.zero_wind_state(), s, jax.random.PRNGKey(0), dt
        )
        increment = None if load_model is None else load_model(s)
        return dynamics.load_factor(s, controls, ac, wind_ned, omega_gust, increment)

    n_z = jax.vmap(n_z_of)(hist.pos_ned, hist.vel_body, hist.quat, hist.omega)
    return n_z, hist.pos_ned[:, 0]


def peak_to_peak(n_z, mask) -> jnp.ndarray:
    """max - min of n_z inside a FIXED boolean mask.

    The mask is an argument rather than a window pair on purpose: it is computed
    once at the base point and reused across every perturbed run, so that what
    moves is the load and not the definition of where it was measured.
    """
    big = jnp.where(mask, n_z, -jnp.inf)
    small = jnp.where(mask, n_z, jnp.inf)
    return jnp.max(big) - jnp.min(small)


def field_vector(ac: Aircraft, fields) -> jnp.ndarray:
    """The named scalar fields as one vector, so a whole screen is one jacfwd."""
    _check_fields(fields)
    return jnp.array([jnp.asarray(getattr(ac, f)).reshape(()) for f in fields])


def with_field_vector(ac: Aircraft, fields, vec) -> Aircraft:
    """`ac` with the named fields taken from `vec`. The inverse of `field_vector`."""
    return ac._replace(**{f: vec[i] for i, f in enumerate(fields)})


def load_elasticities(ac: Aircraft, wind_model, V, H, *, start_north, n_steps, dt,
                      window, fields=None, moving_air: bool = True,
                      load_model=None):
    """Elasticity of the peak-to-peak load w.r.t. each field, in ONE forward pass.

    Every field is pushed through the same 4,700-step scan as a separate tangent
    rather than as a separate run, so the whole screen costs one compile and one
    rollout's worth of forward-mode work.

    Returns (Q, {field: gradient}, {field: elasticity}, extras) where `extras`
    carries the base mask and the indices of the two extremes -- the second of
    which is what a caller needs to see, because the gradient of a peak belongs
    to whichever sample IS the peak.
    """
    fields = INDEPENDENT_FIELDS if fields is None else tuple(fields)
    _check_fields(fields)

    base_n_z, base_north = load_history(
        ac, wind_model, V, H, start_north=start_north, n_steps=n_steps, dt=dt,
        moving_air=moving_air, load_model=load_model)
    mask = (base_north >= window[0]) & (base_north <= window[1])

    def quantity(vec):
        n_z, _ = load_history(
            with_field_vector(ac, fields, vec), wind_model, V, H,
            start_north=start_north, n_steps=n_steps, dt=dt,
            moving_air=moving_air, load_model=load_model)
        return peak_to_peak(n_z, mask)

    base = field_vector(ac, fields)
    Q = float(quantity(base))
    grad = np.asarray(jax.jacfwd(quantity)(base))

    values = {f: float(getattr(ac, f)) for f in fields}
    gradients = {f: float(grad[i]) for i, f in enumerate(fields)}
    elasticities = {f: elasticity(gradients[f], values[f], Q) for f in fields}
    inside = jnp.where(mask, base_n_z, -jnp.inf)
    outside = jnp.where(mask, base_n_z, jnp.inf)
    extras = dict(mask=np.asarray(mask), n_z=np.asarray(base_n_z),
                  north=np.asarray(base_north),
                  argmax=int(jnp.argmax(inside)), argmin=int(jnp.argmin(outside)),
                  values=values)
    return Q, gradients, elasticities, extras
