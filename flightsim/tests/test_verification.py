"""Tier-0 verification: checks that depend on no aircraft data at all.

PROJECT.md section 4 is almost entirely validation -- a measured quantity against
a published one for one aircraft. These are the other kind: if one of them fails,
the arithmetic is wrong and no source can say otherwise.

Note that conftest.py's jax_debug_nans guards JAX only. The torque-free reference
solution goes through SciPy and asserts its own domain instead.
"""

import hashlib
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

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


# --- the ORDER half of the wind seam ---
#
# test_the_six_dof_rollout_is_fourth_order says in its own docstring that it
# "cannot see a wind sample or control update applied at the wrong RK4 stage",
# because it flies in STILL AIR. ASSUMPTIONS.md E4 says the same thing. Session
# 12 closed the BODY-FORCE half of that seam -- no spurious -m dW/dt term -- and
# left the ORDER half open. These two close it, and the answer is not 4.
#
# The two test fields are declared here rather than sourced. Their job is to be
# smooth and to produce a response large enough to sit above the round-off floor,
# not to represent any measured atmosphere.

_SMOOTH_WAVELENGTH = 1200.0  # m, DECLARED: short enough that 4 s covers 0.8 of it
_SMOOTH_AMPLITUDE = 25.0  # m/s, DECLARED: large enough to clear the 7e-11 m floor


def _smooth_wind_model():
    """A C-infinity wind field, through the real `field_model` path.

    Deliberately goes through `wind.field_model` rather than a hand-written
    closure, so `omega_gust` is derived by the same analytic gradient the real
    fields use. The per-step hold applies to both components and a closure that
    returned a zero gust rate would not exercise the seam being measured.
    """
    from flightsim import wind

    wave = wind.LeeWave(
        w0=jnp.array(_SMOOTH_AMPLITUDE),
        wavelength=jnp.array(_SMOOTH_WAVELENGTH),
        north=jnp.array(0.0),
    )
    return wind.field_model(lambda p: wind.lee_wave_wind(p, wave))


def test_the_rollout_is_only_first_order_through_a_spatially_varying_wind():
    """The wind is held across the four RK4 stages, and that costs three orders.

    `integrate.py` samples the wind once per step and holds it. That is correct
    for a stochastic field, for the reason its module docstring now gives --
    re-sampling a key-drawn process per stage makes the realisation a function of
    the step size -- and NOT, as this docstring used to say, because it is "the
    standard treatment for Dryden and von Karman". No source for that claim
    exists in this repository. For a field that varies in SPACE it is an O(h) perturbation
    of the right-hand side within the step, so the scheme is FIRST order however
    good the stage weights are.

    The control is the same function, same window, same aircraft, same elevator
    perturbation, with the wind model removed: that must still read 4. Without
    it this test would only show a number, not attribute it to the wind.

    FALSIFIED, per PROJECT.md section 3's rule that a check which can only pass
    shows nothing. With `integrate.step`'s `f` changed to re-sample the wind at
    each RK4 stage instead of holding it, this test goes RED at an observed order
    of 4.054 -- fourth order restored. So the 1.05 measured here is caused by the
    hold and by nothing else. Measured values, dts 1/4 .. 1/32, dt_ref 1/1024:

        still air        3.9891   (PROJECT.md section 4 records 3.98913)
        smooth field     1.0537   pairwise 1.073, 1.048, 1.042
        ...with the hold removed   4.0542

    The field is C-infinity, so nothing here is about the Rankine core edge --
    that is the next test, and it is a different mechanism. The core test passes
    with the probe still in, which is what says the two are separate.
    """
    ac = REGISTRY["boeing747"]
    V, H = CRUISE["boeing747"]["airspeed"], CRUISE["boeing747"]["altitude"]
    dts = np.array([1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32])

    _, still_air = verification.fixed_control_refinement(ac, V, H, dts, dt_ref=1.0 / 1024.0)
    errors, windy = verification.fixed_control_refinement(
        ac, V, H, dts, dt_ref=1.0 / 1024.0, wind_model=_smooth_wind_model()
    )

    assert still_air == pytest.approx(4.0, abs=0.05), f"control moved: {still_air}"
    assert windy == pytest.approx(1.0, abs=0.1), f"observed order {windy}, errors {errors}"


def test_a_rankine_core_crossing_destroys_even_first_order_convergence():
    """Crossing the core edge makes the error non-monotonic in dt.

    `vortex_wind` switches branches at r = r0, where ASSUMPTIONS.md E2 records
    that the one-sided derivatives differ by 2*V0/r0 and have OPPOSITE SIGNS. The
    right-hand side is therefore C0 but not C1 there, and RK4 across a kink has
    an error that depends on where the step grid happens to land relative to the
    crossing. So refining dt does not monotonically improve the answer.

    Asserted as non-monotonicity rather than as an order, because there is no
    order to assert -- which is the whole point. A fitted slope through this
    sequence returns a number that describes nothing, and anything reporting one
    is reporting an artefact.
    """
    from flightsim import wind

    ac = REGISTRY["boeing747"]
    V, H = CRUISE["boeing747"]["airspeed"], CRUISE["boeing747"]["altitude"]
    case = wind.PARKS_CASES["hannibal"]
    r0 = case["r0"]
    # ONE core, so "through the core" means through exactly one kink pair.
    array = wind.VortexArray(
        north=jnp.array([0.0]), down=jnp.array([-H]),
        r0=jnp.array(r0), v0=jnp.array(case["v0"]),
    )
    model = wind.field_model(lambda p: wind.vortex_wind(p, array))

    dts = np.array([1.0 / 16, 1.0 / 32, 1.0 / 64, 1.0 / 128])
    errors, _ = verification.fixed_control_refinement(
        ac, V, H, dts, dt_ref=1.0 / 2048.0,
        wind_model=model, start_north=-2.0 * r0, d_elevator=0.0,
    )

    # Monotone refinement would mean every error smaller than the last.
    monotone = all(b < a for a, b in zip(errors, errors[1:]))
    assert not monotone, f"expected non-monotone refinement across the core, got {errors}"
    # And it is not the round-off floor doing it: the floor at 40,000 ft is about
    # 7e-11 m (ASSUMPTIONS.md F4) and these errors are centimetres.
    assert errors.min() > 1e-4, f"errors are at the round-off floor: {errors}"


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


def test_a_time_varying_uniform_wind_adds_no_body_force():
    """The second error section 2 names, against an exact solution.

    An air mass that accelerates does not push on the aeroplane. It only changes
    the flow the wings see, so the wind may enter through `vel_rel` and nowhere
    else; an explicit -m*dW/dt term double-counts. The signal such a term leaves
    is unmissable -- it integrates to -(W(t) - W(0)), up to 30 m/s of velocity
    error against the tolerance below.

    The experiment is `verification.free_fall_through_a_swinging_wind`, which
    carries the derivation of why a closed form and not an invariance is the
    right instrument. It lives there rather than here so the notebook runs THIS
    code rather than a second copy of it that could drift.
    """
    n, dt = 300, 0.02
    ac = verification.without_aerodynamics(REGISTRY["boeing747"])
    got = verification.free_fall_through_a_swinging_wind(ac, dt=dt, n=n)

    # The wind really did swing, rather than the model quietly returning zeros
    # and the experiment passing for that reason.
    assert got.elapsed == pytest.approx(n * dt)
    assert got.peak_wind == pytest.approx(29.4618, abs=1e-4)
    assert got.peak_dwdt == pytest.approx(88.3855, abs=1e-4)

    assert got.max_position_error < 1e-9, f"free fall broke by {got.max_position_error} m"


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
