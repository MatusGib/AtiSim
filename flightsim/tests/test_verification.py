"""Tier-0 verification: checks that depend on no aircraft data at all.

PROJECT.md section 4 is almost entirely validation -- a measured quantity against
a published one for one aircraft. These are the other kind: if one of them fails,
the arithmetic is wrong and no source can say otherwise.

Note that conftest.py's jax_debug_nans guards JAX only. The torque-free reference
solution goes through SciPy and asserts its own domain instead.
"""

import hashlib
import itertools
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import flightsim  # noqa: F401  -- enables x64 before any array is made
from flightsim import integrate, trim, verification
from flightsim.aircraft import CRUISE, REGISTRY

# Captured from the integrator BEFORE rk4_step was extracted from `step`. This is
# the whole guard on that refactor: a test that re-derives the stage weights
# inline would compare the new code against itself and pass on an extraction that
# changed the arithmetic.
PRE_REFACTOR_VEL_HASH = "bbc0323e77183d73bd03817a98a530d0d563b56b4962520705aaee589f276aa4"


def _fixed_control_rollout(dt, n_steps, d_elevator=0.02):
    """747 at cruise trim with the elevator off trim, so something happens."""
    ac = REGISTRY["boeing747"]
    V, H = CRUISE["boeing747"]["airspeed"], CRUISE["boeing747"]["altitude"]
    x, _ = trim.trim(jnp.array(V), jnp.array(H), ac)
    state = trim.trimmed_state(x[0], jnp.array(V), jnp.array(H))
    controls = trim.trimmed_controls(x[1] + d_elevator, x[2])
    sim = integrate.init_sim(state, jax.random.PRNGKey(0))
    return integrate.rollout(sim, controls, jnp.array(dt), ac, n_steps)


def test_extracting_rk4_step_did_not_move_a_single_bit():
    """Bit-identity against a hash taken before the refactor.

    The same instrument PROJECT.md section 4 uses for the zero-wind path. A
    tolerance would not do: the claim is that the extraction was arithmetic
    neutral, and any tolerance admits an extraction that was not.
    """
    _, traj = _fixed_control_rollout(0.02, 500)
    got = hashlib.sha256(np.asarray(traj.vel_body).tobytes()).hexdigest()
    assert got == PRE_REFACTOR_VEL_HASH


def test_rk4_is_fourth_order_on_a_problem_with_a_closed_form():
    """The single biggest gap in the project's evidence.

    PROJECT.md section 4 asserts angular momentum barely drifts. An integrator
    can conserve beautifully and still be second-order: drift measures a
    symmetry, not an order. Nothing asserted the order, so a mis-weighted stage
    would pass every existing test at dt = 0.02 and quietly degrade every result
    taken at a larger step.

    Harmonic oscillator, xdot = [[0, 1], [-1, 0]] x, exact solution a rotation.
    Drives integrate.rk4_step DIRECTLY -- not a copy of it -- which is the entire
    point of extracting it.
    """
    dts = np.array([0.2, 0.1, 0.05, 0.025])
    errors, slope = verification.oscillator_refinement(dts)
    assert slope == pytest.approx(4.0, abs=0.05), f"observed order {slope}, {errors}"


def test_the_six_dof_rollout_is_fourth_order():
    """Same claim, through the real dynamics.

    The manufactured case isolates the stage weights. It cannot see a wind sample
    or control update applied at the wrong stage -- the seam turbulence will lean
    on (PROJECT.md section 2: "wind sampled once per step, held across the four
    stages").

    The reference is generated at the smallest step rather than analytically, so
    it carries its own error. At dt = 1/1024 that is around 1e-14, four orders
    below the smallest error being fitted.

    THE FITTED WINDOW MUST STOP AT 1/32, and the reason is a measured property of
    this problem rather than a convenience. The aircraft cruises at 40,000 ft, so
    pos_ned is about [944, 0, -12184] and float64 resolves it to roughly 2.7e-12
    m. Round-off accumulates above that, and the discretisation error reaches it:
    measured pairwise orders over a wider sweep were

        1/4 -> 1/8    3.973
        1/8 -> 1/16   3.993
        1/16 -> 1/32  4.008
        1/32 -> 1/64  4.167
        1/64 -> 1/128 3.420
        1/128 -> 1/256  -0.685   <- refining now makes the answer WORSE

    so the error floor is about 7e-11 m at dt = 1/128. A window running through
    that floor fits partly to round-off and reads 3.82, which is an artefact of
    the measurement and not a defect in the integrator. The window here keeps the
    smallest fitted error 203x above the floor, and the asymptotic range is what
    an order-of-accuracy check is defined on.
    """
    ac = REGISTRY["boeing747"]
    V, H = CRUISE["boeing747"]["airspeed"], CRUISE["boeing747"]["altitude"]
    dts = np.array([1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32])
    errors, slope = verification.fixed_control_refinement(
        ac, V, H, dts, dt_ref=1.0 / 1024.0
    )
    assert slope == pytest.approx(4.0, abs=0.05), f"observed order {slope}, errors {errors}"


def test_a_uniform_horizontal_wind_only_translates_the_trajectory():
    """Galilean invariance -- the assertion the zero-wind test cannot make.

    A uniform wind W is a change of inertial frame. Fly the same aircraft with
    its ground velocity offset by W and the aerodynamics see an identical
    relative flow, so attitude and body rates must be untouched and position must
    differ by exactly W*t.

    PROJECT.md section 2 names putting vel_rel into the Coriolis term as a
    classic gust-modelling error. This test catches exactly that, and it is
    invisible in still air, which is why every test in the project was blind to
    it until now.

    SCOPE, stated because it is narrower than it looks: the wind is STEADY, so
    its material derivative is zero and this test CANNOT detect the other error
    section 2 names -- a spurious -m dW/dt term. That needs a time-varying field
    and a different assertion than invariance; it is section 8's open question
    rather than something claimed here.

    The wind is HORIZONTAL. A vertical component would leave the blown aircraft
    at a different altitude (2 m/s over 20 s is 40 m), and density is a function
    of altitude, so the two aircraft would not see the same dynamic pressure and
    the invariance would not hold to any tolerance worth asserting.
    """
    W = jnp.array([7.0, -3.0, 0.0])  # m/s NED, horizontal by necessity

    def uniform_wind(wind_state, state, key, dt):
        return W, jnp.zeros(3), wind_state, key

    ac = REGISTRY["boeing747"]
    V, H = CRUISE["boeing747"]["airspeed"], CRUISE["boeing747"]["altitude"]
    x, _ = trim.trim(jnp.array(V), jnp.array(H), ac)
    state = trim.trimmed_state(x[0], jnp.array(V), jnp.array(H))
    controls = trim.trimmed_controls(x[1] + 0.01, x[2])

    from flightsim.state import quat_to_dcm

    dcm = quat_to_dcm(state.quat)  # body -> NED
    shifted = state._replace(vel_body=state.vel_body + dcm.T @ W)

    dt, n = jnp.array(0.02), 1000
    _, still = integrate.rollout(
        integrate.init_sim(state, jax.random.PRNGKey(0)), controls, dt, ac, n
    )
    _, blown = integrate.rollout(
        integrate.init_sim(shifted, jax.random.PRNGKey(0)),
        controls, dt, ac, n, wind_model=uniform_wind,
    )

    np.testing.assert_allclose(np.asarray(blown.quat), np.asarray(still.quat), atol=1e-11)
    np.testing.assert_allclose(np.asarray(blown.omega), np.asarray(still.omega), atol=1e-11)

    t = np.arange(1, n + 1) * float(dt)
    expected = np.asarray(still.pos_ned) + t[:, None] * np.asarray(W)
    np.testing.assert_allclose(np.asarray(blown.pos_ned), expected, atol=1e-6)


# --- the other gust error PROJECT.md section 2 names ---
#
# The Galilean test above catches the first (vel_rel in the Coriolis term) and
# says in its own docstring that it CANNOT catch the second, because a steady
# wind's material derivative is zero. These two close that seam.
#
# Note what the correct claim is NOT. It is tempting to fly a time-varying wind
# from a start state offset by W(0) and demand the rates match still air, as the
# steady test does. That is FALSE PHYSICS. Writing v~_b = v_b - C^T W(t) for the
# air-relative body velocity and differentiating,
#
#     v~_b_dot = F(v~_b, omega)/m + g_b - omega x v~_b - C^T Wdot
#
# so the air-relative state obeys the still-air equation PLUS a -C^T Wdot term.
# The two trajectories must diverge, and by a lot -- a 30 m/s wind swing on a
# 236 m/s cruise is a 13% airspeed excursion. That term is precisely what makes
# a time-varying wind something other than a change of inertial frame, and an
# invariance assertion is therefore the wrong instrument for it.
#
# The right instrument is a case with a closed-form answer, which is the first
# test, plus a structural check on the one place the bug could actually be
# written, which is the second.
#
# BOTH WERE CHECKED BY INJECTING THE BUG, because a test that can only pass
# demonstrates nothing. Adding
#
#     d.vel_body -= C^T (wind_ned - sim.wind_ned) / dt
#
# to `step`'s right-hand side fails both tests by many orders of magnitude. It
# also fails the Galilean test above -- but ONLY through a first-step transient,
# because `init_sim` seeds the wind cache to zeros while the model immediately
# returns W. Seed that cache to W, which is the one line anyone would write on
# seeing a spurious impulse at t = 0, and the Galilean test passes with the bug
# still in: 2.7e-15 on quat and 6.9e-16 on omega, against its own 1e-11
# tolerances. So the blindness its docstring claims is real, and measured.


class _Clock(NamedTuple):
    """A wind state that carries time.

    The `wind_state` slot exists so a model can keep whatever it needs between
    steps -- see wind.py's module docstring, and `FilterState` in
    test_integrate.py, which stands in for a Dryden shaping filter the same way.
    A time-varying field needs a clock and nothing else, so it brings one here
    rather than `wind.WindState` growing a field for a model that does not exist
    yet. Recorded in PROJECT.md section 7: Dryden's time dependence costs no
    signature change.
    """

    t: Array


# Big and fast on purpose. |W0| is 30.5 m/s and OMEGA is 3 rad/s, so the peak
# |dW/dt| is 91 m/s^2 -- 9.3 g of spurious specific force if such a term exists.
_SWING_W0 = jnp.array([18.0, -20.0, 12.0])
_SWING_OMEGA = 3.0


def _swinging_wind(wind_state, state, key, dt):
    """Uniform in SPACE, violently varying in TIME. Sampled before the clock ticks."""
    del state
    gust = _SWING_W0 * jnp.sin(_SWING_OMEGA * wind_state.t)
    return gust, jnp.zeros(3), _Clock(t=wind_state.t + dt), key


def test_a_time_varying_uniform_wind_adds_no_body_force():
    """The second error section 2 names, against an exact solution.

    An air mass that accelerates does not push on the aeroplane. It only changes
    the flow the wings see, so the wind may enter through `vel_rel` and nowhere
    else; an explicit -m*dW/dt term double-counts.

    The experiment zeroes every aerodynamic coefficient and the thrust. That is
    the ISOLATION, not a weakness: with no aerodynamic force the wind has no
    legitimate route into the equations at all, so any dependence of the
    trajectory on it is the spurious term and nothing else. The signal is
    unmissable -- a -m*dW/dt term integrates to -(W(t) - W(0)), up to 30 m/s of
    velocity error against the tolerance below.

    The reference is a closed form rather than another simulation. The body
    starts with zero angular rate and there are no moments, so omega stays zero,
    the quaternion is constant, and vdot_body = g_body is constant. RK4 on a
    constant derivative is exact, so free fall is reproduced to round-off:

        pos_ned(t) = pos0 + v_ned(0)*t + [0, 0, g]*t^2/2

    The attitude is deliberately NOT level. A spurious term rotated through the
    wrong DCM would survive a level test and fail this one.

    Speeds stay below M 0.64, where `wave_drag`'s max() is on its flat side, so
    CD is exactly zero rather than nearly so -- see aero.drag_divergence_mach.
    """
    from flightsim.atmosphere import G0
    from flightsim.state import State, euler_to_quat, quat_to_dcm
    from flightsim.tests.conftest import make_test_aircraft

    # The inertia is left alone -- omega is zero for the whole run, so it never
    # enters. Throttle is 0.5 against max_thrust = 0, which is what makes "no
    # thrust" a property of the airframe rather than of the control input.
    zeroed = dict(
        CL0=0.0, CLa=0.0, CLq=0.0, CLde=0.0, Cm0=0.0, Cma=0.0, Cmq=0.0, Cmde=0.0,
        CD0=0.0, CYb=0.0, CYp=0.0, CYr=0.0, CYdr=0.0, Clb=0.0, Clp=0.0, Clr=0.0,
        Clda=0.0, Cldr=0.0, Cnb=0.0, Cnp=0.0, Cnr=0.0, Cnda=0.0, Cndr=0.0,
        max_thrust=0.0,
    )
    ac = make_test_aircraft()._replace(**{k: jnp.array(v) for k, v in zeroed.items()})
    quat = euler_to_quat(jnp.array(0.3), jnp.array(-0.2), jnp.array(0.7))
    state = State(
        pos_ned=jnp.array([0.0, 0.0, -3000.0]),
        vel_body=jnp.array([80.0, 0.0, 0.0]),
        quat=quat,
        omega=jnp.zeros(3),
    )
    controls = trim.trimmed_controls(jnp.array(0.0), jnp.array(0.5))

    dt, n = 0.02, 300
    sim = integrate.SimState(
        state=state,
        wind=_Clock(t=jnp.array(0.0)),
        key=jax.random.PRNGKey(0),
        wind_ned=jnp.zeros(3),
        omega_gust=jnp.zeros(3),
    )
    final, traj = integrate.rollout(
        sim, controls, jnp.array(dt), ac, n, wind_model=_swinging_wind
    )

    # The wind really did swing, rather than the model quietly returning zeros.
    assert float(final.wind.t) == pytest.approx(n * dt)
    assert np.abs(np.asarray(final.wind_ned)).max() > 5.0

    t = np.arange(1, n + 1) * dt
    v_ned0 = np.asarray(quat_to_dcm(quat) @ state.vel_body)
    gravity = np.array([0.0, 0.0, float(G0)])
    exact = (
        np.asarray(state.pos_ned)
        + t[:, None] * v_ned0
        + 0.5 * (t**2)[:, None] * gravity
    )
    np.testing.assert_allclose(np.asarray(traj.pos_ned), exact, atol=1e-9)


def test_a_step_ignores_the_wind_the_previous_step_applied():
    """The same claim structurally, with the real aerodynamics left in.

    `SimState` caches `wind_ned` from the previous step so a controller can sense
    the air without re-evaluating the model. That cache is exactly the ingredient
    a -m*dW/dt term would be built from: differencing it against the current
    sample is the obvious way to get a dW/dt, and it is one line.

    So: take one step twice from the same rigid-body state with the same wind
    model, varying ONLY the cached previous wind. `derivatives` takes an
    instantaneous wind value and no rate, so the two must agree bit for bit.
    Anything that differences the cache makes them differ by dt*(difference),
    which at these values is metres per second.

    Bit-identity rather than a tolerance, for the reason given at
    PRE_REFACTOR_VEL_HASH: the claim is that the cache is not read at all, and
    any tolerance admits a small amount of reading it.
    """
    held = jnp.array([11.0, -6.0, 2.0])

    def steady(wind_state, state, key, dt):
        del state, dt
        return held, jnp.zeros(3), wind_state, key

    ac = REGISTRY["boeing747"]
    V, H = CRUISE["boeing747"]["airspeed"], CRUISE["boeing747"]["altitude"]
    x, _ = trim.trim(jnp.array(V), jnp.array(H), ac)
    state = trim.trimmed_state(x[0], jnp.array(V), jnp.array(H))
    controls = trim.trimmed_controls(x[1] + 0.01, x[2])
    base = integrate.init_sim(state, jax.random.PRNGKey(0))

    def one_step(cached):
        after = integrate.step(
            base._replace(wind_ned=cached), controls, jnp.array(0.02), ac,
            wind_model=steady,
        )
        return jax.tree.map(lambda leaf: np.asarray(leaf).tobytes(), after.state)

    # Second cache is the negative of the wind actually applied, so a difference
    # quotient would see 2*held/dt -- around 1100 m/s^2 -- rather than zero.
    assert one_step(held) == one_step(-held)
    assert one_step(held) == one_step(jnp.zeros(3))


def test_the_trim_solve_converges_quadratically():
    """A Newton solve that merely converges may have a wrong Jacobian.

    Quadratic convergence -- the residual exponent roughly doubling each step --
    is the signature that jacfwd differentiates the same function the residual
    evaluates. A finite-difference or stale Jacobian still converges, linearly,
    and the final residual alone cannot tell them apart.
    """
    ac = REGISTRY["boeing747"]
    V, H = CRUISE["boeing747"]["airspeed"], CRUISE["boeing747"]["altitude"]
    history = verification.newton_residual_history(jnp.array(V), jnp.array(H), ac, 6)
    assert history[0] > 1.0, "the start point is already converged; pick a worse one"

    # takewhile, not a filter: the residual can dip below float64 resolution and
    # come back up, and a filter would splice non-contiguous iterations into a
    # sequence that was never actually walked.
    usable = np.array(list(itertools.takewhile(lambda r: r > 1e-13, history)))
    assert len(usable) >= 3, f"converged too fast to measure: {history}"

    e = np.log10(usable)
    ratios = (e[2:] - e[1:-1]) / (e[1:-1] - e[:-2])
    assert np.max(ratios) > 1.6, f"convergence looks linear, ratios {ratios}"


# Principal-axis inertias, well separated so the elliptic modulus is not near 0
# or 1. Not an aircraft -- these tests have no aircraft in them.
_I1, _I2, _I3 = 1420.0, 4070.0, 4780.0
# Middle component MUST be zero (see torque_free_omega). This pair gives
# m = 0.4928, rate = 0.5782 rad/s, and a1, a3 equal to 0.6 and 0.9 exactly.
_OMEGA0 = np.array([0.6, 0.0, 0.9])


def test_the_analytic_torque_free_solution_solves_eulers_equations():
    """Check the reference before using it as one.

    A mis-stated closed form would otherwise surface as a phantom integrator
    defect in the next test. 2001 points, not 601: the central-difference
    residual scales as h^2 and 601 gives 9.27e-7 against a 1e-6 tolerance, an 8%
    margin that would fail on any tightening.
    """
    t = np.linspace(0.0, 3.0, 2001)
    w = verification.torque_free_omega(_I1, _I2, _I3, _OMEGA0, t)
    I = np.array([_I1, _I2, _I3])[:, None]
    dwdt = np.gradient(w, t, axis=1)
    euler = -np.cross(w.T, (I * w).T).T / I
    np.testing.assert_allclose(dwdt[:, 5:-5], euler[:, 5:-5], atol=1e-6)


def test_the_analytic_solution_rejects_the_wrong_branch():
    """The domain guard, exercised. Without it this returns silent NaN.

    scipy.special.ellipj gives NaN for a modulus above 1, and conftest.py's
    jax_debug_nans cannot see it because this path is NumPy.
    """
    with pytest.raises(ValueError, match="separatrix"):
        verification.torque_free_omega(
            _I1, _I2, _I3, np.array([0.9, 0.0, 0.05]), np.array([0.0, 1.0])
        )


def test_the_integrator_reproduces_torque_free_rotation():
    """Conservation is not correctness.

    PROJECT.md section 4 records angular-momentum drift of 5.7e-13 over 60,000
    steps. A scheme can conserve H exactly and traverse the polhode at the wrong
    rate. This checks the trajectory instead of the invariant.
    """
    from flightsim.aircraft import inertia_tensor
    from flightsim.state import State, euler_to_quat
    from flightsim.tests.conftest import make_test_aircraft

    inertia = inertia_tensor(_I1, _I2, _I3, 0.0)
    zeroed = dict(
        CL0=0.0, CLa=0.0, CLq=0.0, CLde=0.0, Cm0=0.0, Cma=0.0, Cmq=0.0, Cmde=0.0,
        CD0=0.0, CYb=0.0, CYp=0.0, CYr=0.0, CYdr=0.0, Clb=0.0, Clp=0.0, Clr=0.0,
        Clda=0.0, Cldr=0.0, Cnb=0.0, Cnp=0.0, Cnr=0.0, Cnda=0.0, Cndr=0.0,
        max_thrust=0.0,
    )
    ac = make_test_aircraft()._replace(
        inertia=inertia,
        inertia_inv=jnp.linalg.inv(inertia),
        **{k: jnp.array(v) for k, v in zeroed.items()},
    )
    state = State(
        pos_ned=jnp.array([0.0, 0.0, -3000.0]),
        vel_body=jnp.array([60.0, 0.0, 0.0]),
        quat=euler_to_quat(jnp.array(0.0), jnp.array(0.0), jnp.array(0.0)),
        omega=jnp.array(_OMEGA0),
    )
    controls = trim.trimmed_controls(jnp.array(0.0), jnp.array(0.0))
    dt, n = 0.002, 1500
    _, traj = integrate.rollout(
        integrate.init_sim(state, jax.random.PRNGKey(0)), controls, jnp.array(dt), ac, n
    )

    t = np.arange(1, n + 1) * dt
    exact = verification.torque_free_omega(_I1, _I2, _I3, _OMEGA0, t)
    np.testing.assert_allclose(np.asarray(traj.omega).T, exact, atol=1e-8)
