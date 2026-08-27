"""The equations of motion, term by term.

WHICH EARTH EACH TEST FLIES OVER IS PART OF WHAT IT ASSERTS, so it is chosen
per test rather than once for the file. `EARTH` (the shipped `WGS84_J2`) is the
default and is what anything making a claim about flight behaviour uses.
`earth.FLAT` appears at exactly one site -- `test_free_fall_gives_exactly_g`,
which says why in its own docstring -- because that test measures the AERO
build-up, and a rotating, J2 Earth moves the number it is watching without
being any part of the claim.

TWO TESTS BELOW ARE EXPECTED TO FAIL, both marked as such in their own
docstrings with the old and new numbers: `test_gravity_in_body_axes_when_level`
asserts a lateral acceleration of exactly zero, which Coriolis has ended, and
`test_load_factor_in_trimmed_level_flight_is_cos_theta_not_one` asserts
`n_z = cos(theta)`, which apparent gravity no longer being G0 has ended. Both
are migrated mechanically with their tolerances untouched, because a tolerance
widened to absorb a stated physical change stops measuring anything.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from atisim import dynamics, earth
from atisim import state as st
from atisim.atmosphere import G0
from atisim.state import Controls, State, euler_to_quat, state_from_ned
from atisim.tests.conftest import make_test_aircraft

ZERO_CONTROLS = Controls(
    elevator=jnp.array(0.0),
    aileron=jnp.array(0.0),
    rudder=jnp.array(0.0),
    throttle=jnp.array(0.0),
)

# 47N is the latitude the rest of this project's Earth-rotation work uses. The
# anchor sits at SEA LEVEL here rather than at a flight altitude, unlike the run
# files, because this file sweeps altitude as a parameter (0 to 12,000 m).
# `level_state` displaces the aircraft straight up the anchor's OWN normal, so
# its `altitude` argument still means geodetic altitude exactly -- which is what
# `derivatives` reads for density and speed of sound -- and the aircraft shares
# the anchor's latitude and longitude, so the two local frames coincide.
ANCHOR = earth.anchor_at(np.radians(47.0), 0.0, 0.0)
EARTH = earth.WGS84_J2


def level_state(u=50.0, altitude=1000.0, phi=0.0, theta=0.0, psi=0.0) -> State:
    """The local-NED description these tests are written in, converted once."""
    return state_from_ned(
        jnp.array([0.0, 0.0, -altitude]),
        jnp.array([u, 0.0, 0.0]),
        level_ned_quat(phi=phi, theta=theta, psi=psi),
        jnp.zeros(3),
        ANCHOR,
    )


def level_ned_quat(phi=0.0, theta=0.0, psi=0.0):
    """The BODY -> NED quaternion behind the same Euler set.

    `State.quat` is body -> ECEF now, and `relative_velocity_ned` wants
    body -> NED. Both are unit quaternions and neither carries a label, which is
    exactly why `relative_velocity` was renamed: passing `state.quat` still runs
    and still returns a plausible airspeed, in the wrong frame. The tests below
    take the local-frame rotation from here instead.
    """
    return euler_to_quat(jnp.array(phi), jnp.array(theta), jnp.array(psi))


# --- the wind hook: the whole reason these signatures look the way they do ---


def test_zero_wind_gives_inertial_velocity():
    s = level_state(phi=0.4, theta=-0.2, psi=1.3)
    rel = dynamics.relative_velocity_ned(
        s.vel_body, level_ned_quat(phi=0.4, theta=-0.2, psi=1.3), jnp.zeros(3)
    )
    np.testing.assert_allclose(np.asarray(rel), np.asarray(s.vel_body), atol=1e-15)


def test_headwind_raises_airspeed_above_groundspeed():
    """Flying north at 50 m/s into air moving south at 10 m/s -> 60 m/s TAS."""
    s = level_state(u=50.0)
    wind_ned = jnp.array([-10.0, 0.0, 0.0])  # air mass moving south
    rel = dynamics.relative_velocity_ned(s.vel_body, level_ned_quat(), wind_ned)
    assert float(jnp.linalg.norm(rel)) == pytest.approx(60.0)


def test_tailwind_lowers_airspeed_below_groundspeed():
    s = level_state(u=50.0)
    rel = dynamics.relative_velocity_ned(
        s.vel_body, level_ned_quat(), jnp.array([10.0, 0.0, 0.0])
    )
    assert float(jnp.linalg.norm(rel)) == pytest.approx(40.0)


def test_updraft_increases_angle_of_attack():
    """An updraft is air moving up, i.e. negative NED z. It must raise alpha.

    This is the single physical behaviour the whole air-relative-velocity design
    exists to get right.
    """
    from atisim.aero import air_data

    s = level_state(u=50.0)
    still = air_data(
        dynamics.relative_velocity_ned(s.vel_body, level_ned_quat(), jnp.zeros(3))
    )[1]
    updraft = air_data(
        dynamics.relative_velocity_ned(
            s.vel_body, level_ned_quat(), jnp.array([0.0, 0.0, -5.0])
        )
    )[1]
    assert float(updraft) > float(still)
    assert float(updraft) == pytest.approx(np.arctan2(5.0, 50.0))


def test_wind_is_resolved_through_attitude():
    """The same NED wind must give different body components when banked."""
    s_level = level_state(u=50.0)
    s_banked = level_state(u=50.0, phi=np.pi / 2)
    wind = jnp.array([0.0, 0.0, -5.0])  # updraft
    rel_level = dynamics.relative_velocity_ned(
        s_level.vel_body, level_ned_quat(), wind
    )
    rel_banked = dynamics.relative_velocity_ned(
        s_banked.vel_body, level_ned_quat(phi=np.pi / 2), wind
    )
    # Knife-edge to the right, so body y points down. The updraft now appears
    # as sideslip rather than angle of attack: air moving up is body -y, so the
    # relative flow is +y.
    assert float(rel_level[2]) == pytest.approx(5.0)
    assert float(rel_banked[1]) == pytest.approx(5.0, abs=1e-9)
    assert float(rel_banked[2]) == pytest.approx(0.0, abs=1e-9)


def test_omega_gust_subtracts_from_body_rates(test_aircraft):
    """A rolling gust must change the aerodynamics exactly like a roll rate."""
    from atisim.aero import coefficients
    from atisim.atmosphere import speed_of_sound

    A0 = speed_of_sound(0.0)

    vel = jnp.array([50.0, 0.0, 0.0])
    gust = jnp.array([0.2, 0.0, 0.0])
    rolling = coefficients(vel, -gust, ZERO_CONTROLS, test_aircraft, A0)
    still = coefficients(vel, jnp.zeros(3), ZERO_CONTROLS, test_aircraft, A0)
    assert float(rolling[3]) > float(still[3])  # Clp < 0, negative rate -> +Cl


# --- gravity, kinematics, inertia ---


def test_gravity_in_body_axes_when_level(test_aircraft):
    """**THIS TEST IS EXPECTED TO FAIL AND THE TOLERANCE IS LEFT ALONE.**

    `pytest.approx(0.0)` is `abs=1e-12`, i.e. an EXACT zero, and a wings-level
    aeroplane on a rotating Earth does not have one. Measured here:

        quantity                  earth.FLAT    earth.WGS84_J2
        vdot_y (m/s^2)            0.0           5.33312e-03
        vdot_x (m/s^2)           -1.1594248    -1.1594540
        vdot_z (m/s^2)            2.0129244     2.0112413

    The vdot_y number is Coriolis and it is not approximately Coriolis: the
    horizontal Coriolis acceleration is `2 Omega V sin(lat)`, which at 50 m/s
    and 47N is 5.33189e-3 m/s^2, and the 2.3e-4 relative remainder is the
    vertical channel `2 Omega_x w`. So the value is right and the assertion is
    what is wrong -- the same discovery `trim.py` records, where it is why the
    solver grew a bank angle and three more unknowns.

    Migrated mechanically and left red rather than absorbed, because a
    tolerance wide enough to admit 5.3e-3 would also admit a genuine gravity
    leak into body y, which is the failure this test exists to catch. The
    re-measurement has to decide what the claim becomes: either "vdot_y is
    2 Omega V sin(lat)" over `WGS84_J2`, or the wings-level gravity-resolution
    claim moved onto `earth.FLAT`. Note `test_earth.py` already pins the
    Coriolis row against JSBSim's own probes, so this test is not the file
    carrying that coverage.
    """
    s = level_state()
    d = dynamics.derivatives(
        s, ZERO_CONTROLS, test_aircraft, jnp.zeros(3), jnp.zeros(3), ANCHOR, EARTH
    )
    # Level, no thrust: x accel is drag only, z accel is g minus lift over mass.
    lift_over_m = G0 - float(d.vel_body[2])
    assert lift_over_m > 0
    assert float(d.vel_body[1]) == pytest.approx(0.0)


def test_gravity_resolves_into_body_y_when_banked(test_aircraft):
    """At 30 degrees of bank, body-axis gravity is (0, g sin30, g cos30)."""
    from atisim.state import quat_to_matrix

    phi = np.pi / 6
    # `quat_to_matrix` is frame-agnostic; the frame is fixed by what is fed to
    # it. Here that is a body -> NED rotation, so the NED gravity vector
    # transposes into body axes. Nothing about this is Earth-model dependent:
    # it is the rotation, not the plant.
    q_ned = euler_to_quat(jnp.array(phi), jnp.array(0.0), jnp.array(0.0))
    g_body = quat_to_matrix(q_ned).T @ jnp.array([0.0, 0.0, G0])
    np.testing.assert_allclose(
        np.asarray(g_body), [0.0, G0 * np.sin(phi), G0 * np.cos(phi)], atol=1e-12
    )


def test_position_derivative_is_velocity_in_ned(test_aircraft):
    """Heading east, wings level: the position rate is due east."""
    s = level_state(u=50.0, psi=np.pi / 2)
    d = dynamics.derivatives(
        s, ZERO_CONTROLS, test_aircraft, jnp.zeros(3), jnp.zeros(3), ANCHOR, EARTH
    )
    # The derivative is ECEF now, so it is rotated back into the local frame to
    # make the same claim. `ANCHOR.T_e2l` is the AIRCRAFT's frame here and not
    # merely the anchor's: `level_state` displaces straight up the anchor's own
    # normal, so the two share a latitude and longitude. The claim is
    # Earth-model independent -- `pos_ecef_dot` is `T_b2e v` and nothing else --
    # and it is measured identical to the last bit over FLAT and WGS84_J2.
    np.testing.assert_allclose(
        np.asarray(ANCHOR.T_e2l @ d.pos_ecef), [0.0, 50.0, 0.0], atol=1e-12
    )


def test_free_fall_gives_exactly_g(test_aircraft):
    """No airspeed, no thrust: the only acceleration is gravity.

    OVER `earth.FLAT`, and that is what keeps this a test of the AERO build-up
    rather than of the gravity model. What it measures is a residue in
    `aero.py`, described below, whose size was ~0.03% of g -- 2.9e-3 m/s^2.
    Over `WGS84_J2` the free-fall acceleration is apparent gravity at 47N and
    1,000 m, which is 9.804967 m/s^2 against G0's 9.806650, plus a -2.92e-5
    m/s^2 north component where the J2-truncated field is not quite normal to
    the ellipsoid. That is 1.68e-3 m/s^2 of offset -- it fits inside the 5e-3
    tolerance, so the test would still pass, but it would be spending most of
    its remaining margin on gravity and would no longer reliably catch the
    aero residue it was written for.
    """
    s = state_from_ned(
        jnp.array([0.0, 0.0, -1000.0]),
        jnp.zeros(3),
        level_ned_quat(),
        jnp.zeros(3),
        ANCHOR,
    )
    d = dynamics.derivatives(
        s, ZERO_CONTROLS, test_aircraft, jnp.zeros(3), jnp.zeros(3),
        ANCHOR, earth.FLAT,
    )
    # This used to carry a residue: the V_MIN airspeed floor was applied to qbar
    # as well as to the divisions it guards, so CL0 and CD0 produced force at
    # 1 m/s of dynamic pressure the aircraft did not have -- ~0.03% of g here.
    # The floor is now confined to beta and the non-dimensional rates and free
    # fall is exact, which test_audit_regression.py asserts to the bit. The
    # tolerance below is left as it was rather than tightened on the back of a
    # change to the code it is measuring.
    np.testing.assert_allclose(np.asarray(d.vel_body), [0.0, 0.0, G0], atol=5e-3)
    assert np.isfinite(np.asarray(d.vel_body)).all()


def test_gyroscopic_coupling_requires_ixz():
    """With Ixz non-zero, combined roll and yaw rate changes pitch acceleration.

    Note p must differ from r: when p == r the two Ixz contributions to
    (omega x I omega)_y cancel exactly and the test would pass with Ixz ignored.
    """
    s = level_state()._replace(omega=jnp.array([0.3, 0.0, 0.1]))

    without = dynamics.derivatives(
        s, ZERO_CONTROLS, make_test_aircraft(Ixz=0.0), jnp.zeros(3), jnp.zeros(3),
        ANCHOR, EARTH,
    )
    with_ixz = dynamics.derivatives(
        s, ZERO_CONTROLS, make_test_aircraft(Ixz=200.0), jnp.zeros(3), jnp.zeros(3),
        ANCHOR, EARTH,
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
    from atisim.aircraft import inertia_tensor

    inertia = np.asarray(inertia_tensor(1.0, 2.0, 3.0, 0.5))
    np.testing.assert_allclose(
        inertia, [[1.0, 0.0, -0.5], [0.0, 2.0, 0.0], [-0.5, 0.0, 3.0]]
    )
    np.testing.assert_allclose(inertia, inertia.T)


def test_stability_to_body_is_identity_at_zero_alpha():
    from atisim.aircraft import stability_to_body

    cl, cn = stability_to_body(jnp.array(-0.07), jnp.array(0.09), jnp.array(0.0))
    assert float(cl) == pytest.approx(-0.07)
    assert float(cn) == pytest.approx(0.09)


def test_stability_to_body_round_trips():
    from atisim.aircraft import stability_to_body

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

    **THIS TEST IS EXPECTED TO FAIL AND BOTH TOLERANCES ARE LEFT ALONE.** The
    reading is a genuine accelerometer quantity in trimmed cruise, so it belongs
    over the shipped `WGS84_J2` and not over `earth.FLAT`, and over that Earth
    `n_z = cos(theta)` is no longer true. Measured on the 747 at its CR-2144
    cruise, 47N, heading 000:

        quantity              earth.FLAT     earth.WGS84_J2
        n_z                   0.996727965    0.993100670
        cos(theta)            0.996727965    0.996769074
        n_z - cos(theta)     -1.11e-13      -3.668e-03
        theta (deg)           4.636231       4.606999
        phi (deg)             0.000000      -0.148039

    The relation did not break, it acquired two factors that were both exactly
    1 on a flat Earth:

        n_z = (g_apparent / G0) * cos(theta) * cos(phi)

    `phi` is the Coriolis-balancing bank `trim.py` documents, worth 3.3e-6 and
    negligible; the whole discrepancy is `g_apparent / G0`. Apparent gravity at
    47N and 12,192 m is 9.77061 m/s^2 -- gravitation 9.786389 less 0.015779 of
    centrifugal -- against the CONSTANT G0 = 9.806650 that `specific_force`
    divides by. That ratio is 0.996323, and 0.996769 * 0.996323 * 0.9999967 =
    0.993101, which is the measured n_z to every digit printed.

    Dividing by the constant rather than by local gravity is deliberate and
    documented in `specific_force`: it is what makes a load factor comparable
    between latitudes. The consequence is that "1 g" and "n_z = cos(theta)" are
    now latitude- and altitude-dependent statements, and the re-measurement has
    to say which of the two the Wingrove & Bach comparison wants.
    """
    import jax.numpy as jnp

    from atisim import trim
    from atisim.aircraft import CRUISE, REGISTRY
    from atisim.state import quat_to_euler_ned

    ac = REGISTRY["boeing747"]
    v, h = CRUISE["boeing747"]["airspeed"], CRUISE["boeing747"]["altitude"]
    # Its own anchor, at the 747's cruise altitude rather than the module's sea
    # level: `trimmed_state` places the aircraft AT the anchor, so the anchor
    # altitude is the trim altitude.
    anchor = earth.anchor_at(np.radians(47.0), 0.0, h)
    x, _ = trim.trim(jnp.array(v), jnp.array(h), ac, anchor, EARTH)
    # `x[3]` is the trimmed BANK. Carried, not dropped: it is what balances the
    # lateral Coriolis acceleration, so a state built without it is not the
    # equilibrium the solver found.
    state = trim.trimmed_state(x[0], x[3], jnp.array(v), jnp.array(h), anchor, 0.0)
    n_z = dynamics.load_factor(
        state, trim.trimmed_controls(x), ac, jnp.zeros(3), jnp.zeros(3),
        anchor, EARTH,
    )
    theta = float(quat_to_euler_ned(state, anchor)[1])
    assert float(n_z) == pytest.approx(np.cos(theta), abs=1e-9)  # measured 0.996728
    assert float(n_z) == pytest.approx(0.9967, abs=1e-4)


def test_load_factor_matches_the_aerodynamic_and_thrust_force_directly(test_aircraft):
    """n_z is recovered by INVERTING derivatives, so check it against the forces.

    The inversion is what makes it robust to a future force term being added to
    `derivatives`; this test is what proves the inversion is right today.
    """
    from atisim.aero import aero_forces_moments, thrust_force
    from atisim.atmosphere import density, speed_of_sound

    s = level_state(u=60.0, altitude=2000.0)._replace(omega=jnp.array([0.1, 0.2, -0.05]))
    controls = Controls(
        elevator=jnp.array(0.1), aileron=jnp.array(-0.05),
        rudder=jnp.array(0.02), throttle=jnp.array(0.6),
    )
    # GEODETIC, and read through the same path `derivatives` uses, so the
    # density and speed of sound below are the ones the plant actually saw.
    altitude = st.altitude(s, ANCHOR)
    force, _ = aero_forces_moments(
        s.vel_body, s.omega, controls, test_aircraft,
        density(altitude), speed_of_sound(altitude),
    )
    force = force + thrust_force(controls, test_aircraft, density(altitude))
    expected = -float((force / test_aircraft.mass)[2]) / G0

    got = float(dynamics.load_factor(
        s, controls, test_aircraft, jnp.zeros(3), jnp.zeros(3), ANCHOR, EARTH
    ))
    assert got == pytest.approx(expected, abs=1e-12)  # measured agreement ~4e-16


def test_derivatives_are_finite_across_a_wide_envelope(test_aircraft):
    """NaN guard. NaNs inside jit are silent, so catch them at the source."""
    for u in [1.0, 30.0, 250.0]:
        for w in [-40.0, 0.0, 40.0]:
            for h in [0.0, 12000.0]:
                s = state_from_ned(
                    jnp.array([0.0, 0.0, -h]),
                    jnp.array([u, 5.0, w]),
                    level_ned_quat(phi=0.5, theta=-0.3, psi=2.0),
                    jnp.array([0.2, -0.1, 0.05]),
                    ANCHOR,
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
                    ANCHOR,
                    EARTH,
                )
                for field in d:
                    assert np.isfinite(np.asarray(field)).all(), (u, w, h)


def test_specific_force_matches_the_forces_in_all_three_axes(test_aircraft):
    """All three components, recomputed from aero + thrust rather than from the same call.

    load_factor only ever pinned the z component. n_x and n_y are new, and n_y is
    about to drive the slip indicator, so a wrong sign there would be a display
    that is confidently backwards.
    """
    from atisim.aero import aero_forces_moments, thrust_force
    from atisim.atmosphere import density, speed_of_sound

    s = level_state(u=60.0, altitude=2000.0)._replace(omega=jnp.array([0.1, 0.2, -0.05]))
    controls = Controls(
        elevator=jnp.array(0.1), aileron=jnp.array(-0.05),
        rudder=jnp.array(0.02), throttle=jnp.array(0.6),
    )
    altitude = st.altitude(s, ANCHOR)  # geodetic; see the previous test
    force, _ = aero_forces_moments(
        s.vel_body, s.omega, controls, test_aircraft,
        density(altitude), speed_of_sound(altitude),
    )
    force = force + thrust_force(controls, test_aircraft, density(altitude))
    expected = np.asarray(force / test_aircraft.mass) / G0

    got = np.asarray(
        dynamics.specific_force(
            s, controls, test_aircraft, jnp.zeros(3), jnp.zeros(3), ANCHOR, EARTH
        )
    )
    assert got == pytest.approx(expected, abs=1e-12)


def test_load_factor_is_the_negated_z_component_of_specific_force(test_aircraft):
    """The wrapper must not quietly change sign or scale."""
    s = level_state(u=60.0, altitude=2000.0)._replace(omega=jnp.array([0.1, 0.2, -0.05]))
    controls = Controls(
        elevator=jnp.array(0.1), aileron=jnp.array(-0.05),
        rudder=jnp.array(0.02), throttle=jnp.array(0.6),
    )
    n = dynamics.specific_force(
        s, controls, test_aircraft, jnp.zeros(3), jnp.zeros(3), ANCHOR, EARTH
    )
    n_z = dynamics.load_factor(
        s, controls, test_aircraft, jnp.zeros(3), jnp.zeros(3), ANCHOR, EARTH
    )
    assert float(n_z) == pytest.approx(-float(n[2]), abs=1e-15)


# --- the load seam: how strip-integrated coefficients reach the equations ---


def test_a_zero_increment_is_bit_identical_to_not_passing_one(test_aircraft):
    """The property the whole design rests on. Every existing caller omits the
    increment, so if the zero case were merely close rather than exact, every
    frozen baseline in PROJECT.md section 4 would drift.

    Bit-identical, not approximately equal: adding exact 0.0 to a float is the
    identity for every finite value, so there is no reason to accept less.
    """
    from atisim import loads

    state = state_from_ned(
        jnp.array([0.0, 0.0, -2000.0]),
        jnp.array([60.0, 2.0, 3.0]),
        level_ned_quat(phi=0.1, theta=0.05, psi=0.2),
        jnp.array([0.1, 0.2, -0.05]),
        ANCHOR,
    )
    controls = Controls(
        elevator=jnp.array(0.1), aileron=jnp.array(-0.05),
        rudder=jnp.array(0.02), throttle=jnp.array(0.6),
    )
    wind_ned, omega_gust = jnp.zeros(3), jnp.zeros(3)

    without = dynamics.derivatives(
        state, controls, test_aircraft, wind_ned, omega_gust, ANCHOR, EARTH
    )
    with_zero = dynamics.derivatives(
        state, controls, test_aircraft, wind_ned, omega_gust, ANCHOR, EARTH,
        increment=loads.zero_increment(),
    )
    for field in ("pos_ecef", "vel_body", "quat", "omega"):
        assert np.array_equal(
            np.asarray(getattr(without, field)), np.asarray(getattr(with_zero, field))
        ), f"{field} differs between omitting the increment and passing zero"


def test_a_rolling_increment_produces_a_rolling_acceleration(test_aircraft):
    """The increment must actually reach the equations of motion, and reach the
    right axis. A test that only checked the zero case would pass just as
    happily if the increment were ignored entirely."""
    from atisim import loads

    state = state_from_ned(
        jnp.array([0.0, 0.0, -2000.0]),
        jnp.array([60.0, 0.0, 0.0]),
        level_ned_quat(),
        jnp.zeros(3),
        ANCHOR,
    )
    controls = Controls(
        elevator=jnp.array(0.0), aileron=jnp.array(0.0),
        rudder=jnp.array(0.0), throttle=jnp.array(0.5),
    )
    wind_ned, omega_gust = jnp.zeros(3), jnp.zeros(3)

    base = dynamics.derivatives(
        state, controls, test_aircraft, wind_ned, omega_gust, ANCHOR, EARTH
    )
    rolled = dynamics.derivatives(
        state, controls, test_aircraft, wind_ned, omega_gust, ANCHOR, EARTH,
        increment=loads.zero_increment()._replace(Cl=jnp.array(0.01)),
    )
    # A positive rolling-moment coefficient must raise p-dot and leave q-dot alone.
    assert float(rolled.omega[0]) > float(base.omega[0])
    assert float(rolled.omega[1]) == pytest.approx(float(base.omega[1]), abs=1e-12)


def test_a_lift_increment_reaches_the_load_factor(test_aircraft):
    """The gap this closes. `specific_force` inverts `derivatives`' own sum, so
    it sees whatever `derivatives` was given -- but it was calling `derivatives`
    WITHOUT the increment, so a strip run's accelerometer read as if the strip
    loads were not there.

    CL is the channel that matters and the only one: force is built from CL, CD
    and CY, while Cl, Cm and Cn go into the moment. A rolling increment
    therefore cannot move specific force at all, which is why the gap was exact
    rather than small while `strip_increment` populated only Cl. It stops being
    exact the moment CL is filled, and `vortex_viz._measure` builds every Fig. 8
    n_z through `load_factor`.
    """
    from atisim import loads

    s = level_state(u=60.0, altitude=2000.0)
    controls = Controls(
        elevator=jnp.array(0.02), aileron=jnp.array(0.0),
        rudder=jnp.array(0.0), throttle=jnp.array(0.5),
    )
    z3 = jnp.zeros(3)

    base = float(dynamics.load_factor(
        s, controls, test_aircraft, z3, z3, ANCHOR, EARTH
    ))
    lifted = float(dynamics.load_factor(
        s, controls, test_aircraft, z3, z3, ANCHOR, EARTH,
        increment=loads.zero_increment()._replace(CL=jnp.array(0.05)),
    ))
    rolled = float(dynamics.load_factor(
        s, controls, test_aircraft, z3, z3, ANCHOR, EARTH,
        increment=loads.zero_increment()._replace(Cl=jnp.array(0.05)),
    ))
    assert lifted > base, "a positive lift increment must raise the load factor"
    assert rolled == base, "a rolling increment must not touch specific force at all"


def test_omitting_the_increment_in_specific_force_is_bit_identical(test_aircraft):
    """Same guarantee as `derivatives`, and it must hold for the same reason:
    every existing caller omits the increment."""
    from atisim import loads

    s = level_state(u=60.0, altitude=2000.0)._replace(omega=jnp.array([0.1, 0.2, -0.05]))
    controls = Controls(
        elevator=jnp.array(0.1), aileron=jnp.array(-0.05),
        rudder=jnp.array(0.02), throttle=jnp.array(0.6),
    )
    z3 = jnp.zeros(3)

    without = np.asarray(dynamics.specific_force(
        s, controls, test_aircraft, z3, z3, ANCHOR, EARTH
    ))
    with_zero = np.asarray(dynamics.specific_force(
        s, controls, test_aircraft, z3, z3, ANCHOR, EARTH,
        increment=loads.zero_increment(),
    ))
    assert np.array_equal(without, with_zero)
