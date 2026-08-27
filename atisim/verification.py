"""Tier-0 verification: is the arithmetic right?

Nothing here takes an aircraft's published data as a reference. These checks ask
whether the integrator, the trim solve and the rigid-body equations are correct
as mathematics -- a different question from whether the aerodynamic coefficients
describe a real aeroplane, and the one that has to be settled first. A failure
here is a defect in the core.

The split is the standard verification/validation one (Roache; AIAA G-077).
PROJECT.md section 4 mixes them; the design spec says why separating them is most
of the value. Tiers 1 and 2 -- analytic laws and published worked examples -- are
atisim/validation.py.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array


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
    from atisim.integrate import rk4_step

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


def fixed_control_refinement(ac, airspeed, altitude, dts, dt_ref, anchor,
                             earth_model, t_end=4.0,
                             d_elevator=0.02, wind_model=None, start_north=0.0):
    """Refine the real 6-DOF rollout against a fine-step reference.

    The manufactured case isolates the stage weights; this one can also see a
    wind sample or control update applied at the wrong RK4 stage.

    `dts` must stay in the asymptotic range. Round-off puts a floor under the
    error -- for the 747 at 40,000 ft that floor was about 7e-11 m, because
    pos_ned carried a 12,184 m altitude that float64 resolves to 2.7e-12 m -- and
    a sequence crossing it fits partly to round-off. See the test for the
    measured pairwise orders either side of the floor.

    THAT FLOOR IS NOT RE-MEASURED HERE and the figure above is the pre-ECEF one.
    The state now accumulates an ECEF OFFSET from the anchor rather than an NED
    position, and the two carry different magnitudes at the same altitude, so the
    round-off floor moves. Storing an offset rather than an absolute ECEF
    coordinate is what keeps the move small -- an absolute coordinate's ulp is
    512x coarser -- but small is not zero and it has not been measured. Task 14
    re-measures F4 and folds this in; until then, treat the number above as
    superseded rather than current.

    `wind_model` and `start_north` exist because the paragraph above promises
    this function can see a wind sample applied at the wrong stage and, with no
    way to pass a wind model, it never could. Both defaults reproduce the
    still-air call exactly -- `trimmed_state` already puts the aircraft at
    north = 0 -- so adding them moved no existing result.

    THE ANSWER IS NOT 4 ONCE A SPATIAL FIELD IS PASSED, and that is a property
    of the scheme rather than a defect. `integrate.step` samples the wind once
    per step and holds it across the four stages, which is an O(h) perturbation
    of the right-hand side within the step, so the observed order falls to 1.
    Measured 1.01-1.03 across four independent smooth fields against 3.989 in
    still air; see `test_verification.py` and PROJECT.md section 4. A field with
    a KINK -- the Rankine core edge -- has no order at all, because the error
    depends on where the step grid lands relative to the crossing.
    """
    import jax

    from atisim.integrate import init_sim, rollout
    from atisim.state import dcm_body_to_ned, dcm_to_quat, pos_ned, state_from_ned
    from atisim.trim import trim, trimmed_controls, trimmed_state
    from atisim.wind import zero_wind

    x, _ = trim(jnp.array(airspeed), jnp.array(altitude), ac, anchor, earth_model)
    # Due north, matching `trim`'s own default heading: the refinement measures a
    # trajectory difference and the two must be trimmed for the same track.
    state = trimmed_state(
        x[0], x[3], jnp.array(airspeed), jnp.array(altitude), anchor,
        jnp.array(0.0),
    )
    # `start_north` used to be a one-element write into `pos_ned`. It is rebuilt
    # through `state_from_ned` instead of by writing into `pos_ecef`, and the
    # difference is not cosmetic: the trimmed attitude is LEVEL, and level is a
    # statement about the local frame at the aircraft's own position. Writing the
    # offset in directly would carry the anchor's attitude to the new place and
    # tilt it by start_north/R. `trimmed_state` puts the aircraft AT the anchor,
    # so the body -> NED matrix recovered here is exactly the one it built.
    state = state_from_ned(
        jnp.array([jnp.asarray(start_north, dtype=float), 0.0, 0.0]),
        state.vel_body,
        dcm_to_quat(dcm_body_to_ned(state, anchor)),
        state.omega,
        anchor,
    )
    controls = trimmed_controls(x[1] + d_elevator, x[2])
    model = zero_wind if wind_model is None else wind_model

    def final_pos(dt):
        sim = init_sim(state, jax.random.PRNGKey(0))
        _, traj = rollout(
            sim, controls, jnp.array(dt), ac, int(round(t_end / dt)),
            anchor, earth_model, wind_model=model,
        )
        # The refinement is measured in the LOCAL frame, as it always was: the
        # errors are metres of trajectory difference and reading them as ECEF
        # components would make them depend on where on the Earth the run was.
        final = jax.tree.map(lambda column: column[-1], traj)
        return np.asarray(pos_ned(final, anchor))

    reference = final_pos(dt_ref)
    errors = np.array([
        float(np.linalg.norm(final_pos(float(dt)) - reference))
        for dt in np.asarray(dts, dtype=float)
    ])
    return errors, fitted_order(dts, errors)


def newton_residual_history(airspeed, altitude, ac, anchor, earth_model,
                            heading=0.0, iterations=6):
    """Residual norm after each Newton iteration, from trim.py's own start point.

    `trim.trim` runs a fixed iteration count inside `lax.scan` and returns only
    the final answer, so the convergence rate is not observable through it. This
    repeats the same update -- verified against trim.py's scan body as the
    identical undamped step, no damping -- and keeps every iterate.

    **THE STEP IS `lstsq`, NOT `solve`, AND THAT IS NOT A LOOSENING.** It is what
    `trim.trim`'s scan body actually does, and this function's whole claim is
    that it repeats that body. `trim` moved to `lstsq` when it gained `rudder` as
    an unknown: an aircraft with no rudder authority -- the Cessna 172, whose
    CYdr = Cldr = Cndr are all zero deliberately -- has an identically zero sixth
    Jacobian column, and `solve` on a rank-5 system returns six NaNs in silence.
    Keeping `solve` here would have made this measure the convergence of a solver
    the package does not ship, and returned NaN for a registry aircraft while
    `trim` converged for it. `trim.trim`'s own comment records the measurement
    that settled it: every aircraft converges to 1e-13, and an unsolvable request
    still shows up as a large residual rather than as a NaN.
    """
    from atisim.trim import INITIAL_GUESS, residual

    args = (airspeed, altitude, ac, anchor, earth_model, heading)
    x = INITIAL_GUESS
    history = [float(jnp.linalg.norm(residual(x, *args)))]
    for _ in range(iterations):
        r = residual(x, *args)
        jacobian = jax.jacfwd(residual)(x, *args)
        x = x - jnp.linalg.lstsq(jacobian, r)[0]
        history.append(float(jnp.linalg.norm(residual(x, *args))))
    return np.array(history)


class WindClock(NamedTuple):
    """A wind state that carries time, for a field that varies with it.

    `integrate.step` threads `wind_state` opaquely and never interprets it, so a
    model brings whatever state it needs -- see wind.py's module docstring, and
    `FilterState` in test_integrate.py, which stands in for a Dryden shaping
    filter the same way. A time-varying uniform field needs a clock and nothing
    else, so it brings one here rather than `wind.WindState` growing a field for
    a model that does not exist yet. PROJECT.md section 7 records the
    consequence: Dryden's time dependence costs no signature change.
    """

    t: Array


# Big and fast on purpose. |W0| is sqrt(18^2 + 20^2 + 12^2) = 29.4618 m/s and
# OMEGA is 3 rad/s, so the peak |dW/dt| is 88.3855 m/s^2 -- 9.01 g of spurious
# specific force if such a term exists.
#
# Session 12 wrote 30.5 m/s and 91 m/s^2 (9.3 g) here from an arithmetic slip,
# and it reached ASSUMPTIONS.md E4 and PROJECT.md section 9 before session 13
# extracted this experiment and had `FreeFallResult` report the figure instead of
# a comment asserting it. That is the whole argument for computing a number where
# a test can see it.
SWING_W0 = jnp.array([18.0, -20.0, 12.0])
SWING_OMEGA = 3.0


def _swinging_wind(wind_state, state, key, dt):
    """Uniform in SPACE, violently varying in TIME. Sampled before the clock ticks."""
    del state
    gust = SWING_W0 * jnp.sin(SWING_OMEGA * wind_state.t)
    return gust, jnp.zeros(3), WindClock(t=wind_state.t + dt), key


def without_aerodynamics(ac):
    """Every aerodynamic coefficient and the thrust zeroed.

    This is the ISOLATION the free-fall experiment needs, not a weakness of it:
    with no aerodynamic force the wind has no legitimate route into the equations
    at all, so any dependence of the trajectory on it is a spurious term and
    nothing else. Inertia is left alone -- the experiment holds omega at zero for
    its whole run, so it never enters.
    """
    zeroed = dict(
        CL0=0.0, CLa=0.0, CLq=0.0, CLde=0.0, Cm0=0.0, Cma=0.0, Cmq=0.0, Cmde=0.0,
        CD0=0.0, CYb=0.0, CYp=0.0, CYr=0.0, CYdr=0.0, Clb=0.0, Clp=0.0, Clr=0.0,
        Clda=0.0, Cldr=0.0, Cnb=0.0, Cnp=0.0, Cnr=0.0, Cnda=0.0, Cndr=0.0,
        max_thrust=0.0,
    )
    return ac._replace(**{k: jnp.array(v) for k, v in zeroed.items()})


class FreeFallResult(NamedTuple):
    """What `free_fall_through_a_swinging_wind` measured."""

    max_position_error: float  # m, against the closed form
    peak_wind: float  # m/s, largest |W| the aircraft actually flew through
    peak_dwdt: float  # m/s^2, largest |dW/dt| the field carried
    elapsed: float  # s, the clock the wind model advanced itself


def free_fall_through_a_swinging_wind(ac, anchor, dt=0.02, n=300):
    """Fly a de-aerodynamicised body through a violently time-varying uniform wind.

    The second of the two gust-modelling errors PROJECT.md section 2 names. An
    air mass that accelerates does not push on the aeroplane: it only changes the
    flow the wings see, so the wind may enter through `vel_rel` and nowhere else,
    and an explicit -m*dW/dt term double-counts.

    The reference is a CLOSED FORM rather than another simulation. With every
    coefficient zeroed there are no moments, so omega stays zero, the quaternion
    is constant, and vdot_body = g_body is constant. RK4 on a constant derivative
    is exact, so free fall is reproduced to round-off:

        pos_ned(t) = pos0 + v_ned(0)*t + [0, 0, g]*t^2/2

    **THE RUN IS FLOWN UNDER `earth.FLAT` AND THAT IS NOT NEGOTIABLE HERE.** The
    line above is the flat, non-rotating, constant-g solution; under `WGS84_J2`
    the body is also subject to Coriolis, centrifugal and an inverse-square-plus-
    J2 gravity, and it would not follow it. `FLAT` is the configuration of this
    one plant in which the closed form IS the answer, which is what lets the
    residual be read as "does the wind reach the trajectory" and nothing else.
    `earth_model` is therefore NOT an argument -- offering one would let a caller
    ask this function a question its reference cannot answer.

    `FLAT` is not literally flat, and the residual it leaves is GEOMETRIC rather
    than a defect. Its gravity is `G0` along the LOCAL geodetic vertical, which
    rotates as the body travels, while the closed form uses the anchor's. Over
    the default 6 s the body covers about 480 m of ground, which is 7.5e-5 rad of
    vertical rotation, and the departure it leaves is **3.32e-3 m**, measured. It
    grows as t^3 -- 3.32e-6 m at 0.6 s, 1.23e-4 at 2 s, 3.32e-3 at 6 s -- which
    is the signature of an acceleration error rising linearly with distance
    travelled, integrated twice, and not of anything stochastic or accumulating.

    **THAT RESIDUAL IS BIT-IDENTICAL WITH THE WIND SWITCHED OFF, AND THAT IS THE
    MEASUREMENT THAT MATTERS.** Re-running with the gust replaced by zeros gives
    0.0033222180416032643 m against 0.0033222180416032643 m -- the same float,
    not the same to a tolerance. So the wind-dependent part of this error is
    EXACTLY zero, which is the claim the experiment exists to make, and the
    3.3 mm is the coordinate system rather than a spurious -m dW/dt term. The
    instrument therefore has its full discriminating power; what it lost is only
    the ability to be asserted against round-off directly. A caller wanting that
    should difference two wind settings rather than tighten this number.

    An invariance assertion is the WRONG instrument here and that is worth
    recording. Writing v~_b = v_b - C^T W(t) for the air-relative body velocity
    and differentiating gives

        v~_b_dot = F(v~_b, omega)/m + g_b - omega x v~_b - C^T Wdot

    so the air-relative state obeys the still-air equation PLUS a -C^T Wdot term.
    A time-varying wind is therefore not a change of inertial frame, and two runs
    offset by W(0) genuinely must diverge. The seam needs a closed form.

    The attitude is deliberately NOT level: a spurious term rotated through the
    wrong DCM would survive a level test and fail this one. Speeds stay low
    enough that `wave_drag`'s max() is on its flat side, so CD is exactly zero
    rather than nearly so.

    `ac` should be a real registry airframe put through `without_aerodynamics`.
    Free fall is independent of mass and airframe, so the choice cannot flatter
    the result -- which is what lets the notebook and the test run the same code
    on the same aircraft.
    """
    from atisim import earth
    from atisim.atmosphere import G0
    from atisim.integrate import SimState, rollout
    from atisim.loads import zero_increment
    from atisim.state import dcm_body_to_ned, euler_to_quat, pos_ned, state_from_ned
    from atisim.trim import trimmed_controls

    quat_ned = euler_to_quat(jnp.array(0.3), jnp.array(-0.2), jnp.array(0.7))
    start_ned = jnp.array([0.0, 0.0, -3000.0])
    state = state_from_ned(
        start_ned,
        jnp.array([80.0, 0.0, 0.0]),
        quat_ned,
        jnp.zeros(3),
        anchor,
    )
    # Throttle 0.5 against max_thrust = 0, so "no thrust" is a property of the
    # airframe rather than of the control input.
    controls = trimmed_controls(jnp.array(0.0), jnp.array(0.5))

    sim = SimState(
        state=state,
        wind=WindClock(t=jnp.array(0.0)),
        key=jax.random.PRNGKey(0),
        wind_ned=jnp.zeros(3),
        omega_gust=jnp.zeros(3),
        increment=zero_increment(),
    )
    final, traj = rollout(
        sim, controls, jnp.array(dt), ac, n, anchor, earth.FLAT,
        wind_model=_swinging_wind,
    )

    t = np.arange(1, n + 1) * dt
    # The initial NED velocity, taken at the aircraft's own position. Straight up
    # from the anchor is the one place the local frame and the anchor's coincide
    # exactly, and `start_ned` is on that line, so this is the anchor frame the
    # closed form below is written in.
    v_ned0 = np.asarray(dcm_body_to_ned(state, anchor) @ state.vel_body)
    gravity = np.array([0.0, 0.0, float(G0)])
    exact = (
        np.asarray(start_ned)
        + t[:, None] * v_ned0
        + 0.5 * (t**2)[:, None] * gravity
    )
    flown = np.asarray(jax.vmap(pos_ned, in_axes=(0, None))(traj, anchor))
    error = float(np.abs(flown - exact).max())

    # Reported so a caller can assert the wind really swung, rather than the
    # model quietly returning zeros and the experiment passing for that reason.
    sampled = np.asarray(SWING_W0) * np.sin(SWING_OMEGA * np.arange(n) * dt)[:, None]
    return FreeFallResult(
        max_position_error=error,
        peak_wind=float(np.linalg.norm(sampled, axis=1).max()),
        peak_dwdt=float(np.linalg.norm(np.asarray(SWING_W0)) * SWING_OMEGA),
        elapsed=float(final.wind.t),
    )


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
