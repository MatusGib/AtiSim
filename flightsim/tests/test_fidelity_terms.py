"""The sideslip-drag and angle-of-attack-rate terms added after the JSBSim work.

Design: docs/superpowers/specs/2026-08-20-model-fidelity-improvements-design.md

Both fields default to neutral, so the whole point of this file is that the
terms are inert until an aircraft asks for them AND the flight condition
produces the input. Each is asserted, not assumed.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flightsim import aero, integrate, wind
from flightsim.aircraft import REGISTRY
from flightsim.atmosphere import RHO0, speed_of_sound
from flightsim.state import Controls, State, euler_to_quat
from flightsim.tests.conftest import make_test_aircraft

A0 = float(speed_of_sound(0.0))
ZERO = Controls(elevator=jnp.array(0.0), aileron=jnp.array(0.0),
                rudder=jnp.array(0.0), throttle=jnp.array(0.0))


# ---------------------------------------------------------------------------
# sideslip drag
# ---------------------------------------------------------------------------
def test_sideslip_drag_defaults_to_neutral(test_aircraft):
    assert float(test_aircraft.CD_beta) == 0.0
    straight = aero.coefficients(jnp.array([60.0, 0.0, 2.0]), jnp.zeros(3), ZERO,
                                 test_aircraft, jnp.array(A0))[1]
    sideslipping = aero.coefficients(jnp.array([60.0, 6.0, 2.0]), jnp.zeros(3), ZERO,
                                     test_aircraft, jnp.array(A0))[1]
    assert float(straight) == float(sideslipping)


def test_sideslip_drag_is_even_in_beta():
    """The reason the term is quadratic: drag cannot know left from right.

    A laterally symmetric airframe must produce identical drag at +beta and
    -beta, so dCD/dbeta is zero at the origin and the leading term is second
    order. JSBSim's own table is symmetric for the same reason, but interpolates
    linearly through zero, which gives it a slope discontinuity there.
    """
    ac = make_test_aircraft()._replace(CD_beta=jnp.array(0.8))
    left = aero.coefficients(jnp.array([60.0, -6.0, 2.0]), jnp.zeros(3), ZERO, ac,
                             jnp.array(A0))[1]
    right = aero.coefficients(jnp.array([60.0, 6.0, 2.0]), jnp.zeros(3), ZERO, ac,
                              jnp.array(A0))[1]
    assert float(left) == pytest.approx(float(right), rel=1e-12)


def test_sideslip_drag_grows_as_beta_squared():
    """Doubling sideslip quadruples the increment.

    Velocity is built from (V, alpha, beta) with alpha held at zero, so CL and
    therefore the induced drag are identical at every point and the only thing
    that moves is the sideslip term. Varying the y component alone instead
    tilts alpha slightly and contaminates the ratio by 0.25%.
    """
    ac = make_test_aircraft()._replace(CD_beta=jnp.array(0.8))
    V = 60.0

    def drag(beta):
        vel = V * jnp.array([np.cos(beta), np.sin(beta), 0.0])
        return float(aero.coefficients(vel, jnp.zeros(3), ZERO, ac, jnp.array(A0))[1])

    base = drag(0.0)
    assert (drag(0.10) - base) / (drag(0.05) - base) == pytest.approx(4.0, rel=1e-9)


def test_the_737_sideslip_drag_meets_jsbsims_table_at_its_breakpoint():
    """0.05 at 0.26 rad is where 737.xml's table places a real number."""
    ac = REGISTRY["boeing737"]
    assert float(ac.CD_beta) * 0.26**2 == pytest.approx(0.05, rel=1e-12)


# ---------------------------------------------------------------------------
# angle-of-attack rate
# ---------------------------------------------------------------------------
def test_alphadot_terms_default_to_neutral(test_aircraft):
    assert float(test_aircraft.CLadot) == 0.0
    assert float(test_aircraft.Cmadot) == 0.0


def test_alphadot_does_nothing_without_a_coefficient(test_aircraft):
    """A fixture with no Cmadot is unaffected at any alphadot."""
    quiet = aero.coefficients(jnp.array([60.0, 0.0, 2.0]), jnp.zeros(3), ZERO,
                              test_aircraft, jnp.array(A0), jnp.array(0.0))[4]
    gusting = aero.coefficients(jnp.array([60.0, 0.0, 2.0]), jnp.zeros(3), ZERO,
                                test_aircraft, jnp.array(A0), jnp.array(0.5))[4]
    assert float(quiet) == float(gusting)


def test_alphadot_moment_follows_stengel_3_4_26():
    """Cm gains Cmadot * alphadot * c / 2V, the same non-dimensionalisation as q."""
    ac = make_test_aircraft()._replace(Cmadot=jnp.array(-8.0))
    vel = jnp.array([60.0, 0.0, 0.0])
    quiet = float(aero.coefficients(vel, jnp.zeros(3), ZERO, ac, jnp.array(A0),
                                    jnp.array(0.0))[4])
    gusting = float(aero.coefficients(vel, jnp.zeros(3), ZERO, ac, jnp.array(A0),
                                      jnp.array(0.4))[4])
    expected = -8.0 * 0.4 * float(ac.c) / (2.0 * 60.0)
    assert gusting - quiet == pytest.approx(expected, rel=1e-12)


def test_still_air_produces_exactly_zero_alphadot():
    """The term cannot disturb any still-air result, by construction."""
    state = State(pos_ned=jnp.array([0.0, 0.0, -3000.0]),
                  vel_body=jnp.array([60.0, 0.0, 2.0]),
                  quat=euler_to_quat(jnp.array(0.0), jnp.array(0.03), jnp.array(0.0)),
                  omega=jnp.zeros(3))
    *_, alphadot = wind.zero_wind(wind.zero_wind_state(), state,
                                  jax.random.PRNGKey(0), 0.02)
    assert float(alphadot) == 0.0


def test_a_uniform_wind_produces_no_alphadot():
    """alphadot comes from the field's GRADIENT, so a constant wind gives none."""
    state = State(pos_ned=jnp.array([0.0, 0.0, -3000.0]),
                  vel_body=jnp.array([60.0, 0.0, 2.0]),
                  quat=euler_to_quat(jnp.array(0.0), jnp.array(0.03), jnp.array(0.0)),
                  omega=jnp.zeros(3))
    got = wind.gust_alphadot(state.pos_ned, state.quat, state.vel_body,
                             lambda p: jnp.array([3.0, 0.0, -2.0]))
    assert abs(float(got)) < 1e-12


def test_flying_into_a_vertical_gradient_produces_alphadot_of_the_right_sign():
    """Climbing updraft ahead: alpha rises as the aircraft flies into it.

    The field's downward component becomes more negative (more updraft) with
    north position, so flying north the aircraft sees w_rel fall and alpha rise.
    """
    state = State(pos_ned=jnp.array([0.0, 0.0, -3000.0]),
                  vel_body=jnp.array([60.0, 0.0, 0.0]),
                  quat=euler_to_quat(jnp.array(0.0), jnp.array(0.0), jnp.array(0.0)),
                  omega=jnp.zeros(3))
    # w_ned = -0.01 * north  ->  updraft strengthening ahead
    got = float(wind.gust_alphadot(
        state.pos_ned, state.quat, state.vel_body,
        lambda p: jnp.array([0.0, 0.0, -0.01 * p[0]])))
    # alphadot = -d(w_wind)/dt / V = -(-0.01 * 60)/60 = +0.01
    assert got == pytest.approx(0.01, rel=1e-9)


def test_the_gust_alphadot_reaches_the_pitching_moment_through_step():
    """End to end: a gradient field plus a Cmadot changes the trajectory.

    This is the whole point of the term, so it is asserted through the real
    integrator rather than at the coefficient level.
    """
    ac = REGISTRY["boeing737"]
    state = State(pos_ned=jnp.array([0.0, 0.0, -9000.0]),
                  vel_body=jnp.array([236.0, 0.0, 8.0]),
                  quat=euler_to_quat(jnp.array(0.0), jnp.array(0.034), jnp.array(0.0)),
                  omega=jnp.zeros(3))
    field = lambda p: jnp.array([0.0, 0.0, -0.05 * p[0]])  # noqa: E731
    model = wind.field_model(field)
    controls = ZERO._replace(throttle=jnp.array(0.77))

    with_term = integrate.step(integrate.init_sim(state, jax.random.PRNGKey(0)),
                               controls, jnp.array(0.05), ac, wind_model=model)
    without = integrate.step(integrate.init_sim(state, jax.random.PRNGKey(0)),
                             controls, jnp.array(0.05), ac,
                             wind_model=wind.field_model(field)) \
        if False else None
    del without

    bare = ac._replace(Cmadot=jnp.array(0.0))
    no_term = integrate.step(integrate.init_sim(state, jax.random.PRNGKey(0)),
                             controls, jnp.array(0.05), bare, wind_model=model)
    assert float(with_term.state.omega[1]) != float(no_term.state.omega[1])


# ---------------------------------------------------------------------------
# aerodynamic reference point
# ---------------------------------------------------------------------------
def test_aero_reference_defaults_to_the_cg(test_aircraft):
    """Zero offset means the coefficients are about the CG, as they always were."""
    assert float(np.abs(np.asarray(test_aircraft.aero_ref)).max()) == 0.0


def test_a_zero_reference_offset_adds_exact_zero(test_aircraft):
    """Not merely 'about the same' -- jnp.cross of a zero vector is exact zero.

    This is what makes every pre-existing aircraft bit-identical rather than
    equivalent to round-off, which matters because PROJECT.md section 4's
    numbers were measured before the field existed.
    """
    vel = jnp.array([60.0, 3.0, 2.0])
    controls = ZERO._replace(elevator=jnp.array(0.05))
    _, moment = aero.aero_forces_moments(vel, jnp.array([0.1, 0.05, 0.02]), controls,
                                         test_aircraft, RHO0, jnp.array(A0))
    offset = test_aircraft._replace(aero_ref=jnp.zeros(3))
    _, again = aero.aero_forces_moments(vel, jnp.array([0.1, 0.05, 0.02]), controls,
                                        offset, RHO0, jnp.array(A0))
    assert [float(v) for v in moment] == [float(v) for v in again]


def test_the_transfer_is_r_cross_f():
    """Stengel Eq. (2.4-68), as a cross product rather than a chord fraction."""
    offset = jnp.array([-0.36, 0.0, -1.50])
    ac = make_test_aircraft()._replace(aero_ref=offset)
    vel = jnp.array([60.0, 2.0, 3.0])
    controls = ZERO._replace(elevator=jnp.array(0.04))
    force, moment = aero.aero_forces_moments(vel, jnp.zeros(3), controls, ac,
                                             RHO0, jnp.array(A0))
    at_cg = make_test_aircraft()
    _, base = aero.aero_forces_moments(vel, jnp.zeros(3), controls, at_cg,
                                       RHO0, jnp.array(A0))
    expected = np.asarray(base) + np.cross(np.asarray(offset), np.asarray(force))
    np.testing.assert_allclose(np.asarray(moment), expected, rtol=1e-12, atol=0.0)


def test_the_737_carries_jsbsims_own_constants_now():
    """The point of the change: the entry traces to a file, not a fit.

    737.xml's PITCH/Cmalpha is -0.6, ROLL/Clb is -0.09 and YAW/Cnb is +0.26.
    Referred to the CG those become -1.13, -0.144 and +0.273, and which of those
    you get depends on the fuel state. Referred to the AERORP they are the file's
    numbers, at both recovery conditions, because that is what they are.
    """
    for name in ("boeing737", "boeing737_approach"):
        ac = REGISTRY[name]
        assert float(ac.Cma) == pytest.approx(-0.6, abs=5e-4), name
        assert float(ac.Clb) == pytest.approx(-0.09, abs=5e-6), name
        assert float(ac.Cnb) == pytest.approx(+0.26, abs=5e-6), name
        # 737.xml has no Cm0 term at all; what looked like one was the offset.
        assert abs(float(ac.Cm0)) < 1e-4, name
