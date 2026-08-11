"""Tier-0 verification: is the arithmetic right?

Nothing here takes an aircraft's published data as a reference. These checks ask
whether the integrator, the trim solve and the rigid-body equations are correct
as mathematics -- a different question from whether the aerodynamic coefficients
describe a real aeroplane, and the one that has to be settled first. A failure
here is a defect in the core.

The split is the standard verification/validation one (Roache; AIAA G-077).
PROJECT.md section 4 mixes them; the design spec says why separating them is most
of the value. Tiers 1 and 2 -- analytic laws and published worked examples -- are
flightsim/validation.py.
"""

import jax
import jax.numpy as jnp
import numpy as np


def fitted_order(dts, errors):
    """Observed order of accuracy: the slope of log(error) against log(dt).

    Least squares over the whole sequence rather than a two-point ratio, so one
    noisy refinement cannot carry the answer.
    """
    dts = np.asarray(dts, dtype=float)
    errors = np.asarray(errors, dtype=float)
    if np.any(errors <= 0.0):
        raise ValueError("an error is zero or negative; the sequence is saturated")
    slope, _ = np.polyfit(np.log(dts), np.log(errors), 1)
    return float(slope)


def oscillator_refinement(dts, t_end=2.0):
    """Refine a harmonic oscillator through the real RK4 and return (errors, order).

    xdot = [[0, 1], [-1, 0]] x, whose exact solution is a rotation. Drives
    `integrate.rk4_step` DIRECTLY rather than a copy of it, which is the point of
    that function being split out of `step`.

    Lives here rather than inline in the test so the notebook runs the same
    experiment instead of a second version of it that could drift.
    """
    from flightsim.integrate import rk4_step

    def f(x):
        return jnp.array([x[1], -x[0]])

    x0 = jnp.array([1.0, 0.0])
    exact = np.array([np.cos(t_end), -np.sin(t_end)])

    errors = []
    for dt in np.asarray(dts, dtype=float):
        x = x0
        for _ in range(int(round(t_end / dt))):
            x = rk4_step(f, x, dt)
        errors.append(float(np.linalg.norm(np.asarray(x) - exact)))
    errors = np.array(errors)
    return errors, fitted_order(dts, errors)


def fixed_control_refinement(ac, airspeed, altitude, dts, dt_ref, t_end=4.0,
                             d_elevator=0.02):
    """Refine the real 6-DOF rollout against a fine-step reference.

    The manufactured case isolates the stage weights; this one can also see a
    wind sample or control update applied at the wrong RK4 stage.

    `dts` must stay in the asymptotic range. Round-off puts a floor under the
    error -- for the 747 at 40,000 ft that floor is about 7e-11 m, because
    pos_ned carries a 12,184 m altitude that float64 resolves to 2.7e-12 m -- and
    a sequence crossing it fits partly to round-off. See the test for the
    measured pairwise orders either side of the floor.
    """
    import jax

    from flightsim.integrate import init_sim, rollout
    from flightsim.trim import trim, trimmed_controls, trimmed_state

    x, _ = trim(jnp.array(airspeed), jnp.array(altitude), ac)
    state = trimmed_state(x[0], jnp.array(airspeed), jnp.array(altitude))
    controls = trimmed_controls(x[1] + d_elevator, x[2])

    def final_pos(dt):
        sim = init_sim(state, jax.random.PRNGKey(0))
        _, traj = rollout(sim, controls, jnp.array(dt), ac, int(round(t_end / dt)))
        return np.asarray(traj.pos_ned[-1])

    reference = final_pos(dt_ref)
    errors = np.array([
        float(np.linalg.norm(final_pos(float(dt)) - reference))
        for dt in np.asarray(dts, dtype=float)
    ])
    return errors, fitted_order(dts, errors)


def newton_residual_history(airspeed, altitude, ac, iterations=6):
    """Residual norm after each Newton iteration, from trim.py's own start point.

    `trim.trim` runs a fixed iteration count inside `lax.scan` and returns only
    the final answer, so the convergence rate is not observable through it. This
    repeats the same update -- verified against trim.py's scan body as the
    identical undamped Newton step, no damping and no least squares -- and keeps
    every iterate.
    """
    from flightsim.trim import INITIAL_GUESS, residual

    x = INITIAL_GUESS
    history = [float(jnp.linalg.norm(residual(x, airspeed, altitude, ac)))]
    for _ in range(iterations):
        r = residual(x, airspeed, altitude, ac)
        jacobian = jax.jacfwd(residual)(x, airspeed, altitude, ac)
        x = x - jnp.linalg.solve(jacobian, r)
        history.append(float(jnp.linalg.norm(residual(x, airspeed, altitude, ac))))
    return np.array(history)


def torque_free_omega(I1, I2, I3, omega0, t):
    """Exact torque-free rotation of an asymmetric rigid body. Returns (3, len(t)).

    Landau & Lifshitz, *Mechanics*, section 37, for I1 < I2 < I3 with the rotation
    nearer the I3 axis. That branch requires L^2 > 2*T*I2; on the other side of
    the separatrix the elliptic modulus exceeds 1 and `scipy.special.ellipj`
    returns NaN. conftest.py's jax_debug_nans does NOT see that -- this path is
    NumPy -- so the domain is asserted here rather than left to be discovered.

    Note omega_2(0) = a2*sn(0) = 0 identically, so this branch can only represent
    an initial rate whose MIDDLE component is zero.
    """
    from scipy.special import ellipj

    I = np.array([I1, I2, I3], dtype=float)
    if not (I1 < I2 < I3):
        raise ValueError("this form assumes I1 < I2 < I3")
    w0 = np.asarray(omega0, dtype=float)
    if w0[1] != 0.0:
        raise ValueError("omega_2(0) is identically zero on this branch")
    L2 = float(np.sum((I * w0) ** 2))
    twoT = float(np.sum(I * w0**2))
    if L2 <= twoT * I2:
        raise ValueError(
            f"L^2 = {L2:.4e} <= 2T*I2 = {twoT * I2:.4e}: wrong side of the "
            "separatrix, the elliptic modulus would exceed 1"
        )

    a1 = np.sqrt((twoT * I3 - L2) / (I1 * (I3 - I1)))
    a2 = np.sqrt((twoT * I3 - L2) / (I2 * (I3 - I2)))
    a3 = np.sqrt((L2 - twoT * I1) / (I3 * (I3 - I1)))
    rate = np.sqrt((I3 - I2) * (L2 - twoT * I1) / (I1 * I2 * I3))
    m = ((I2 - I1) * (twoT * I3 - L2)) / ((I3 - I2) * (L2 - twoT * I1))

    sn, cn, dn, _ = ellipj(rate * np.asarray(t, dtype=float), m)
    return np.vstack([a1 * cn, a2 * sn, a3 * dn])
