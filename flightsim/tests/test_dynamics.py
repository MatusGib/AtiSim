import jax.numpy as jnp
import numpy as np
import pytest

from flightsim import dynamics
from flightsim.atmosphere import G0
from flightsim.state import Controls, State, euler_to_quat
from flightsim.tests.conftest import make_test_aircraft

ZERO_CONTROLS = Controls(
    elevator=jnp.array(0.0),
    aileron=jnp.array(0.0),
    rudder=jnp.array(0.0),
    throttle=jnp.array(0.0),
)


def level_state(u=50.0, altitude=1000.0, phi=0.0, theta=0.0, psi=0.0) -> State:
    return State(
        pos_ned=jnp.array([0.0, 0.0, -altitude]),
        vel_body=jnp.array([u, 0.0, 0.0]),
        quat=euler_to_quat(jnp.array(phi), jnp.array(theta), jnp.array(psi)),
        omega=jnp.zeros(3),
    )


# --- the wind hook: the whole reason these signatures look the way they do ---


def test_zero_wind_gives_inertial_velocity():
    s = level_state(phi=0.4, theta=-0.2, psi=1.3)
    rel = dynamics.relative_velocity(s.vel_body, s.quat, jnp.zeros(3))
    np.testing.assert_allclose(np.asarray(rel), np.asarray(s.vel_body), atol=1e-15)


def test_headwind_raises_airspeed_above_groundspeed():
    """Flying north at 50 m/s into air moving south at 10 m/s -> 60 m/s TAS."""
    s = level_state(u=50.0)
    wind_ned = jnp.array([-10.0, 0.0, 0.0])  # air mass moving south
    rel = dynamics.relative_velocity(s.vel_body, s.quat, wind_ned)
    assert float(jnp.linalg.norm(rel)) == pytest.approx(60.0)


def test_tailwind_lowers_airspeed_below_groundspeed():
    s = level_state(u=50.0)
    rel = dynamics.relative_velocity(s.vel_body, s.quat, jnp.array([10.0, 0.0, 0.0]))
    assert float(jnp.linalg.norm(rel)) == pytest.approx(40.0)


def test_updraft_increases_angle_of_attack():
    """An updraft is air moving up, i.e. negative NED z. It must raise alpha.

    This is the single physical behaviour the whole air-relative-velocity design
    exists to get right.
    """
    from flightsim.aero import air_data

    s = level_state(u=50.0)
    still = air_data(dynamics.relative_velocity(s.vel_body, s.quat, jnp.zeros(3)))[1]
    updraft = air_data(
        dynamics.relative_velocity(s.vel_body, s.quat, jnp.array([0.0, 0.0, -5.0]))
    )[1]
    assert float(updraft) > float(still)
    assert float(updraft) == pytest.approx(np.arctan2(5.0, 50.0))


def test_wind_is_resolved_through_attitude():
    """The same NED wind must give different body components when banked."""
    s_level = level_state(u=50.0)
    s_banked = level_state(u=50.0, phi=np.pi / 2)
    wind = jnp.array([0.0, 0.0, -5.0])  # updraft
    rel_level = dynamics.relative_velocity(s_level.vel_body, s_level.quat, wind)
    rel_banked = dynamics.relative_velocity(s_banked.vel_body, s_banked.quat, wind)
    # Knife-edge to the right, so body y points down. The updraft now appears
    # as sideslip rather than angle of attack: air moving up is body -y, so the
    # relative flow is +y.
    assert float(rel_level[2]) == pytest.approx(5.0)
    assert float(rel_banked[1]) == pytest.approx(5.0, abs=1e-9)
    assert float(rel_banked[2]) == pytest.approx(0.0, abs=1e-9)


def test_omega_gust_subtracts_from_body_rates(test_aircraft):
    """A rolling gust must change the aerodynamics exactly like a roll rate."""
    from flightsim.aero import coefficients
    from flightsim.atmosphere import speed_of_sound

    A0 = speed_of_sound(0.0)

    vel = jnp.array([50.0, 0.0, 0.0])
    gust = jnp.array([0.2, 0.0, 0.0])
    rolling = coefficients(vel, -gust, ZERO_CONTROLS, test_aircraft, A0)
    still = coefficients(vel, jnp.zeros(3), ZERO_CONTROLS, test_aircraft, A0)
    assert float(rolling[3]) > float(still[3])  # Clp < 0, negative rate -> +Cl


# --- gravity, kinematics, inertia ---


def test_gravity_in_body_axes_when_level(test_aircraft):
    s = level_state()
    d = dynamics.derivatives(s, ZERO_CONTROLS, test_aircraft, jnp.zeros(3), jnp.zeros(3))
    # Level, no thrust: x accel is drag only, z accel is g minus lift over mass.
    lift_over_m = G0 - float(d.vel_body[2])
    assert lift_over_m > 0
    assert float(d.vel_body[1]) == pytest.approx(0.0)


def test_gravity_resolves_into_body_y_when_banked(test_aircraft):
    """At 30 degrees of bank, body-axis gravity is (0, g sin30, g cos30)."""
    from flightsim.state import quat_to_dcm

    phi = np.pi / 6
    q = euler_to_quat(jnp.array(phi), jnp.array(0.0), jnp.array(0.0))
    g_body = quat_to_dcm(q).T @ jnp.array([0.0, 0.0, G0])
    np.testing.assert_allclose(
        np.asarray(g_body), [0.0, G0 * np.sin(phi), G0 * np.cos(phi)], atol=1e-12
    )


def test_position_derivative_is_velocity_in_ned(test_aircraft):
    """Heading east, wings level: the position rate is due east."""
    s = level_state(u=50.0, psi=np.pi / 2)
    d = dynamics.derivatives(s, ZERO_CONTROLS, test_aircraft, jnp.zeros(3), jnp.zeros(3))
    np.testing.assert_allclose(np.asarray(d.pos_ned), [0.0, 50.0, 0.0], atol=1e-12)


def test_free_fall_gives_exactly_g(test_aircraft):
    """No airspeed, no thrust: the only acceleration is gravity."""
    s = State(
        pos_ned=jnp.array([0.0, 0.0, -1000.0]),
        vel_body=jnp.zeros(3),
        quat=euler_to_quat(jnp.array(0.0), jnp.array(0.0), jnp.array(0.0)),
        omega=jnp.zeros(3),
    )
    d = dynamics.derivatives(s, ZERO_CONTROLS, test_aircraft, jnp.zeros(3), jnp.zeros(3))
    # The V_MIN airspeed floor leaves a residue of CL0 lift and CD0 drag at
    # 1 m/s. It is ~0.03% of g here and only exists to keep alpha/beta finite at
    # zero velocity, a condition that never arises in flight.
    np.testing.assert_allclose(np.asarray(d.vel_body), [0.0, 0.0, G0], atol=5e-3)
    assert np.isfinite(np.asarray(d.vel_body)).all()


def test_gyroscopic_coupling_requires_ixz():
    """With Ixz non-zero, combined roll and yaw rate changes pitch acceleration.

    Note p must differ from r: when p == r the two Ixz contributions to
    (omega x I omega)_y cancel exactly and the test would pass with Ixz ignored.
    """
    s = level_state()._replace(omega=jnp.array([0.3, 0.0, 0.1]))

    without = dynamics.derivatives(
        s, ZERO_CONTROLS, make_test_aircraft(Ixz=0.0), jnp.zeros(3), jnp.zeros(3)
    )
    with_ixz = dynamics.derivatives(
        s, ZERO_CONTROLS, make_test_aircraft(Ixz=200.0), jnp.zeros(3), jnp.zeros(3)
    )
    assert abs(float(without.omega[1]) - float(with_ixz.omega[1])) > 1e-3


def test_ixz_couples_roll_and_yaw_acceleration():
    """Ixz makes I^-1 non-diagonal, so a pure rolling moment also yaws."""
    ac = make_test_aircraft(Ixz=200.0)
    roll_only = jnp.array([1000.0, 0.0, 0.0])
    accel = ac.inertia_inv @ roll_only
    assert abs(float(accel[2])) > 1e-6  # yaw acceleration from a roll moment

    diagonal = make_test_aircraft(Ixz=0.0)
    assert float((diagonal.inertia_inv @ roll_only)[2]) == pytest.approx(0.0, abs=1e-15)


def test_inertia_tensor_layout():
    from flightsim.aircraft import inertia_tensor

    inertia = np.asarray(inertia_tensor(1.0, 2.0, 3.0, 0.5))
    np.testing.assert_allclose(
        inertia, [[1.0, 0.0, -0.5], [0.0, 2.0, 0.0], [-0.5, 0.0, 3.0]]
    )
    np.testing.assert_allclose(inertia, inertia.T)


def test_stability_to_body_is_identity_at_zero_alpha():
    from flightsim.aircraft import stability_to_body

    cl, cn = stability_to_body(jnp.array(-0.07), jnp.array(0.09), jnp.array(0.0))
    assert float(cl) == pytest.approx(-0.07)
    assert float(cn) == pytest.approx(0.09)


def test_stability_to_body_round_trips():
    from flightsim.aircraft import stability_to_body

    alpha = jnp.array(0.15)
    cl_b, cn_b = stability_to_body(jnp.array(-0.07), jnp.array(0.09), alpha)
    cl_s, cn_s = stability_to_body(cl_b, cn_b, -alpha)
    assert float(cl_s) == pytest.approx(-0.07, abs=1e-15)
    assert float(cn_s) == pytest.approx(0.09, abs=1e-15)


def test_load_factor_in_trimmed_level_flight_is_cos_theta_not_one():
    """n_z is the BODY-NORMAL load factor -- what an accelerometer reads.

    In trimmed level flight the body z axis is tilted from the vertical by the
    pitch attitude, so the reading is g*cos(theta), not g. At the 747's CR-2144
    trim of theta = 4.636 deg that is 0.99673, and asserting 1.0 here would be
    asserting the wrong physics.

    The distinction matters for the turbulence work: Wingrove & Bach's Fig. 8
    is built from DFDR "normal acceleration", which is exactly this body-normal
    quantity, so this is the right convention for that comparison rather than a
    flight-path-normal one.
    """
    import jax.numpy as jnp

    from flightsim import trim
    from flightsim.aircraft import CRUISE, REGISTRY
    from flightsim.state import quat_to_euler

    ac = REGISTRY["boeing747"]
    v, h = CRUISE["boeing747"]["airspeed"], CRUISE["boeing747"]["altitude"]
    x, _ = trim.trim(jnp.array(v), jnp.array(h), ac)
    state = trim.trimmed_state(x[0], jnp.array(v), jnp.array(h))
    n_z = dynamics.load_factor(
        state, trim.trimmed_controls(x[1], x[2]), ac, jnp.zeros(3), jnp.zeros(3)
    )
    theta = float(quat_to_euler(state.quat)[1])
    assert float(n_z) == pytest.approx(np.cos(theta), abs=1e-9)  # measured 0.996728
    assert float(n_z) == pytest.approx(0.9967, abs=1e-4)


def test_load_factor_matches_the_aerodynamic_and_thrust_force_directly(test_aircraft):
    """n_z is recovered by INVERTING derivatives, so check it against the forces.

    The inversion is what makes it robust to a future force term being added to
    `derivatives`; this test is what proves the inversion is right today.
    """
    from flightsim.aero import aero_forces_moments, thrust_force
    from flightsim.atmosphere import density, speed_of_sound

    s = level_state(u=60.0, altitude=2000.0)._replace(omega=jnp.array([0.1, 0.2, -0.05]))
    controls = Controls(
        elevator=jnp.array(0.1), aileron=jnp.array(-0.05),
        rudder=jnp.array(0.02), throttle=jnp.array(0.6),
    )
    altitude = -s.pos_ned[2]
    force, _ = aero_forces_moments(
        s.vel_body, s.omega, controls, test_aircraft,
        density(altitude), speed_of_sound(altitude),
    )
    force = force + thrust_force(controls, test_aircraft, density(altitude))
    expected = -float((force / test_aircraft.mass)[2]) / G0

    got = float(dynamics.load_factor(s, controls, test_aircraft, jnp.zeros(3), jnp.zeros(3)))
    assert got == pytest.approx(expected, abs=1e-12)  # measured agreement ~4e-16


def test_derivatives_are_finite_across_a_wide_envelope(test_aircraft):
    """NaN guard. NaNs inside jit are silent, so catch them at the source."""
    for u in [1.0, 30.0, 250.0]:
        for w in [-40.0, 0.0, 40.0]:
            for h in [0.0, 12000.0]:
                s = State(
                    pos_ned=jnp.array([0.0, 0.0, -h]),
                    vel_body=jnp.array([u, 5.0, w]),
                    quat=euler_to_quat(jnp.array(0.5), jnp.array(-0.3), jnp.array(2.0)),
                    omega=jnp.array([0.2, -0.1, 0.05]),
                )
                d = dynamics.derivatives(
                    s,
                    Controls(
                        elevator=jnp.array(0.2),
                        aileron=jnp.array(-0.1),
                        rudder=jnp.array(0.1),
                        throttle=jnp.array(0.8),
                    ),
                    test_aircraft,
                    jnp.zeros(3),
                    jnp.zeros(3),
                )
                for field in d:
                    assert np.isfinite(np.asarray(field)).all(), (u, w, h)


def test_specific_force_matches_the_forces_in_all_three_axes(test_aircraft):
    """All three components, recomputed from aero + thrust rather than from the same call.

    load_factor only ever pinned the z component. n_x and n_y are new, and n_y is
    about to drive the slip indicator, so a wrong sign there would be a display
    that is confidently backwards.
    """
    from flightsim.aero import aero_forces_moments, thrust_force
    from flightsim.atmosphere import density, speed_of_sound

    s = level_state(u=60.0, altitude=2000.0)._replace(omega=jnp.array([0.1, 0.2, -0.05]))
    controls = Controls(
        elevator=jnp.array(0.1), aileron=jnp.array(-0.05),
        rudder=jnp.array(0.02), throttle=jnp.array(0.6),
    )
    altitude = -s.pos_ned[2]
    force, _ = aero_forces_moments(
        s.vel_body, s.omega, controls, test_aircraft,
        density(altitude), speed_of_sound(altitude),
    )
    force = force + thrust_force(controls, test_aircraft, density(altitude))
    expected = np.asarray(force / test_aircraft.mass) / G0

    got = np.asarray(
        dynamics.specific_force(s, controls, test_aircraft, jnp.zeros(3), jnp.zeros(3))
    )
    assert got == pytest.approx(expected, abs=1e-12)


def test_load_factor_is_the_negated_z_component_of_specific_force(test_aircraft):
    """The wrapper must not quietly change sign or scale."""
    s = level_state(u=60.0, altitude=2000.0)._replace(omega=jnp.array([0.1, 0.2, -0.05]))
    controls = Controls(
        elevator=jnp.array(0.1), aileron=jnp.array(-0.05),
        rudder=jnp.array(0.02), throttle=jnp.array(0.6),
    )
    n = dynamics.specific_force(s, controls, test_aircraft, jnp.zeros(3), jnp.zeros(3))
    n_z = dynamics.load_factor(s, controls, test_aircraft, jnp.zeros(3), jnp.zeros(3))
    assert float(n_z) == pytest.approx(-float(n[2]), abs=1e-15)
