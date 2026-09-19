"""The hand-derived checks in scripts/sanity.py that nothing else in the suite asserted.

Rung 1 of notebooks/validation-ladder.ipynb runs scripts/sanity.py and asserts
its count. For checks [2] to [5], and for [11] with an elevator input, that was
the only gate: nothing in atisim/tests/ checked them, which is how [2] and [5]
went stale unseen when gravity began to vary with height and the 747 declared a
thrust line (docs/PROJECT.md section 9, session 32, point 11).

These are the same checks, on the same trimmed 747, at the script's own
tolerances. Each expected value is derived by hand here rather than taken from
the model or imported from the script.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from atisim import trim, verification
from atisim.aircraft import CRUISE, REGISTRY, inertia_tensor
from atisim.atmosphere import G0
from atisim.dynamics import derivatives
from atisim.state import quat_to_dcm

AC = REGISTRY["boeing747"]
V = CRUISE["boeing747"]["airspeed"]
H = CRUISE["boeing747"]["altitude"]
ZERO = jnp.zeros(3)


@pytest.fixture(scope="module")
def cruise():
    """The 747 trimmed at cruise: its state, and (alpha, elevator, throttle)."""
    x, _ = trim.trim(jnp.array(V), jnp.array(H), AC)
    return trim.trimmed_state(x[0], jnp.array(V), jnp.array(H)), x


def _sideslipped(state):
    """About 3 deg of sideslip and no rotation: the state checks [4] and [5] load."""
    return state._replace(vel_body=state.vel_body + jnp.array([0.0, 12.0, 0.0]), omega=ZERO)


def test_free_fall_is_straight_down_at_the_local_gravity(cruise):
    """[2] and [3]. With no aerodynamics and no thrust, gravity is the only force,
    so the NED acceleration is (0, 0, g(h)) at any attitude; the trimmed state is
    pitched nose-up by alpha. g(h) by hand: inverse-square, R = 6,371 km. At cruise
    it is 0.38% below G0, which a check against G0 cannot see.
    """
    state, _ = cruise
    g_here = G0 * (6371000.0 / (6371000.0 + H)) ** 2
    d = derivatives(
        state._replace(omega=ZERO), trim.trimmed_controls(jnp.array(0.0), jnp.array(1.0)),
        verification.without_aerodynamics(AC), ZERO, ZERO,
    )
    accel_ned = np.asarray(quat_to_dcm(state.quat) @ d.vel_body)
    assert abs(accel_ned[2] - g_here) <= 1e-12
    assert np.abs(accel_ned[:2]).max() <= 1e-12


def test_no_rolling_terms_and_a_diagonal_inertia_give_no_roll_acceleration(cruise):
    """[4]. Every rolling-moment coefficient zeroed and the inertia made diagonal:
    full aileron and rudder in sideslip then produce no roll acceleration at all.

    Both halves are needed. With the 747's real inertia the yawing moment reaches
    roll through Ixz, which is correct, so p-dot is not zero -- asserted too, so
    this test cannot pass by the inputs having no effect.
    """
    state, x = cruise
    no_roll = {k: jnp.array(0.0) for k in ("Clb", "Clp", "Clr", "Clda", "Cldr")}
    inertia = np.asarray(AC.inertia)
    diagonal = inertia_tensor(inertia[0, 0], inertia[1, 1], inertia[2, 2], 0.0)
    controls = trim.trimmed_controls(x[1], x[2])._replace(
        aileron=jnp.array(0.3), rudder=jnp.array(0.2))

    coupled = derivatives(_sideslipped(state), controls, AC._replace(**no_roll), ZERO, ZERO)
    uncoupled = derivatives(
        _sideslipped(state), controls,
        AC._replace(**no_roll, inertia=diagonal, inertia_inv=jnp.linalg.inv(diagonal)),
        ZERO, ZERO,
    )
    assert abs(float(uncoupled.omega[0])) <= 1e-18
    assert abs(float(coupled.omega[0])) > 1e-6


def test_no_pitching_terms_give_no_pitch_acceleration(cruise):
    """[5]. Every pitching-moment term zeroed, then no elevator input can pitch
    the aircraft. "Every" is more than the four obvious coefficients: alpha-dot,
    the Mach derivative (the sideslip moves the Mach off its reference), and the
    thrust line's arm -- the 747's thrust line sits 5.70 ft below the CG, so
    thrust on its own pitches the nose up. Iyy couples to nothing on this aircraft.
    """
    state, x = cruise
    zeroed = ("Cm0", "Cma", "Cmq", "Cmde", "Cmadot", "Cm_M", "thrust_arm")
    d = derivatives(
        _sideslipped(state), trim.trimmed_controls(jnp.array(0.4), x[2]),
        AC._replace(**{k: jnp.array(0.0) for k in zeroed}), ZERO, ZERO,
    )
    assert abs(float(d.omega[1])) <= 1e-18


def test_an_elevator_input_excites_nothing_lateral(cruise):
    """[11]. Wings level with no sideslip, a pitch rate and an elevator step: the
    longitudinal and lateral equations must not couple, so the sideslip, roll and
    yaw accelerations are exactly zero. The suite already asserts this for
    longitudinal wind fields; this is the control-input case.
    """
    state, x = cruise
    d = derivatives(
        state._replace(omega=jnp.array([0.0, 0.05, 0.0])),
        trim.trimmed_controls(x[1] + 0.15, x[2]), AC, ZERO, ZERO,
    )
    lateral = np.abs([float(d.vel_body[1]), float(d.omega[0]), float(d.omega[2])])
    assert lateral.max() <= 1e-18
