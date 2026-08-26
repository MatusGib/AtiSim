import jax
import jax.numpy as jnp
import numpy as np
import pytest

from atisim import earth
from atisim import state as st


def euler_dcm_reference(phi, theta, psi):
    """Independent body -> NED 3-2-1 DCM, built from first principles.

    Exists to catch a convention error in quat_to_matrix/euler_to_quat that a
    round-trip test would happily agree with.
    """
    rz = np.array(
        [[np.cos(psi), -np.sin(psi), 0], [np.sin(psi), np.cos(psi), 0], [0, 0, 1]]
    )
    ry = np.array(
        [[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]]
    )
    rx = np.array(
        [[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]]
    )
    return rz @ ry @ rx


ANGLE_CASES = [
    (0.0, 0.0, 0.0),
    (0.3, -0.2, 1.1),
    (-1.2, 0.7, -2.5),
    (np.pi / 4, np.pi / 6, -np.pi / 3),
    (2.9, -1.4, 0.05),
]


@pytest.mark.parametrize("phi,theta,psi", ANGLE_CASES)
def test_quat_dcm_matches_independent_euler_dcm(phi, theta, psi):
    q = st.euler_to_quat(jnp.array(phi), jnp.array(theta), jnp.array(psi))
    np.testing.assert_allclose(
        np.asarray(st.quat_to_matrix(q)), euler_dcm_reference(phi, theta, psi), atol=1e-12
    )


@pytest.mark.parametrize("phi,theta,psi", ANGLE_CASES)
def test_euler_quat_round_trip(phi, theta, psi):
    q = st.euler_to_quat(jnp.array(phi), jnp.array(theta), jnp.array(psi))
    back = np.asarray(st.matrix_to_euler(st.quat_to_matrix(q)))
    np.testing.assert_allclose(back, [phi, theta, psi], atol=1e-12)


def test_dcm_orthonormal_for_random_quaternions():
    key = jax.random.PRNGKey(0)
    qs = jax.random.normal(key, (200, 4))
    qs = qs / jnp.linalg.norm(qs, axis=1, keepdims=True)
    dcms = jax.vmap(st.quat_to_matrix)(qs)
    eye = jnp.eye(3)
    for dcm in dcms:
        np.testing.assert_allclose(np.asarray(dcm @ dcm.T), eye, atol=1e-12)
        assert float(jnp.linalg.det(dcm)) == pytest.approx(1.0, abs=1e-12)


def test_quaternion_norm_stable_over_1e5_steps():
    """Euler-integrate a tumbling rotation for 1e5 steps and check the norm."""
    dt = 1e-2
    omega = jnp.array([0.7, -0.4, 0.3])

    def body(q, _):
        q = q + dt * st.quat_derivative(q, omega)
        return st.quat_normalize(q), None

    q0 = st.euler_to_quat(jnp.array(0.1), jnp.array(0.2), jnp.array(0.3))
    qf, _ = jax.lax.scan(body, q0, None, length=100_000)
    assert float(jnp.linalg.norm(qf)) == pytest.approx(1.0, abs=1e-12)
    assert bool(jnp.all(jnp.isfinite(qf)))


@pytest.mark.parametrize(
    "omega,index,name",
    [
        (jnp.array([0.1, 0.0, 0.0]), 0, "roll rate p increases phi"),
        (jnp.array([0.0, 0.1, 0.0]), 1, "pitch rate q increases theta"),
        (jnp.array([0.0, 0.0, 0.1]), 2, "yaw rate r increases psi"),
    ],
)
def test_body_rate_signs_drive_the_right_euler_angle(omega, index, name):
    """From level flight, a positive body rate must increase its Euler angle.

    This is the sign check that separates a sim that flies from a sim that is
    correct.
    """
    dt = 1e-3
    q = st.euler_to_quat(jnp.array(0.0), jnp.array(0.0), jnp.array(0.0))
    for _ in range(100):
        q = st.quat_normalize(q + dt * st.quat_derivative(q, omega))
    euler = np.asarray(st.matrix_to_euler(st.quat_to_matrix(q)))
    expected = 0.1 * 0.1  # rate * elapsed time
    assert euler[index] == pytest.approx(expected, rel=1e-3), name
    others = [e for i, e in enumerate(euler) if i != index]
    assert np.allclose(others, 0.0, atol=1e-9)


def test_altitude_is_negative_down():
    """Down is still negative, but altitude now comes back through the geodesy.

    Straight up from the anchor is the one place the tangent plane and the
    ellipsoid still agree exactly -- NED's D axis IS the geodetic normal -- so
    this stays a clean statement of the sign convention.
    """
    anchor = earth.anchor_at(np.radians(47.0), np.radians(11.0), 0.0)
    s = st.state_from_ned(
        jnp.array([0.0, 0.0, -3000.0]),
        jnp.zeros(3),
        jnp.array([1.0, 0.0, 0.0, 0.0]),
        jnp.zeros(3),
        anchor,
    )
    assert float(st.altitude(s, anchor)) == pytest.approx(3000.0, abs=1e-6)


def test_x64_is_enabled():
    assert jnp.zeros(3).dtype == jnp.float64


def test_quat_to_dcm_no_longer_exists_under_its_old_name():
    """The rename is the safety mechanism, not cosmetics.

    `quat_to_dcm(state.quat)` meant body->NED and now means body->ECEF. Both are
    valid rotation matrices, so every one of the 40-odd call sites would have
    kept running and returned a wrong answer silently. Deleting the name makes
    each one an ImportError instead. If this test ever fails because someone
    re-added the alias, the aliasing is the bug.
    """
    import atisim.state as st

    assert not hasattr(st, "quat_to_dcm"), (
        "quat_to_dcm was re-added; it is ambiguous about frames now and each "
        "call site must say body_to_ecef or body_to_ned"
    )
    assert hasattr(st, "quat_to_matrix")


def test_dcm_to_quat_round_trips_including_the_trace_minus_one_case():
    """Shepperd, branch-free. The naive trace form fails near trace = -1."""
    import atisim.state as st

    rng = np.random.default_rng(0)
    for _ in range(200):
        q = rng.normal(size=4)
        q = q / np.linalg.norm(q)
        m = st.quat_to_matrix(jnp.asarray(q))
        back = np.asarray(st.dcm_to_quat(m))
        # q and -q are the same rotation.
        if np.dot(back, q) < 0:
            back = -back
        assert np.allclose(back, q, atol=1e-12)

    # The adversarial case: 180 degrees about y, where trace = -1.
    m = jnp.asarray(np.diag([-1.0, 1.0, -1.0]))
    back = st.dcm_to_quat(m)
    assert np.allclose(np.asarray(st.quat_to_matrix(back)), np.diag([-1.0, 1.0, -1.0]), atol=1e-12)


def test_pos_ned_is_a_rotation_of_the_offset_not_a_difference_of_large_numbers():
    """This is what preserves F4's round-off floor.

    An absolute ECEF coordinate has a 9.3e-10 m ulp against 1.8e-12 m for a
    12 km NED altitude -- 512x. Forming pos_ned by DIFFERENCING two absolute
    positions would inherit the coarse one. Rotating the stored offset does not.
    """
    import atisim.state as st

    anchor = earth.anchor_at(np.radians(47.0), np.radians(11.0), 0.0)
    offset = jnp.asarray([1234.5, -678.25, -9144.0])
    state = st.State(
        pos_ecef=anchor.T_e2l.T @ offset,
        vel_body=jnp.zeros(3),
        quat=jnp.asarray([1.0, 0.0, 0.0, 0.0]),
        omega=jnp.zeros(3),
    )
    assert np.allclose(np.asarray(st.pos_ned(state, anchor)), np.asarray(offset), atol=1e-9)


def test_altitude_is_geodetic_and_diverges_from_the_tangent_plane_with_range():
    """-pos_ned[2] is no longer altitude, and the difference is not small.

    The ellipsoid falls away beneath the anchor's tangent plane as d^2/2R, so a
    point that is level in that plane is ABOVE the ellipsoid by about 7.8 m at
    10 km of ground track and 785 m at 100 km -- a positive geodetic altitude
    even though pos_ned[2] is exactly zero. Anything still reading -pos_ned[2]
    as altitude is a bug this asserts the existence of.
    """
    import atisim.state as st

    anchor = earth.anchor_at(0.0, 0.0, 0.0)
    for distance, expected in ((10.0e3, 7.8), (100.0e3, 785.0)):
        offset = jnp.asarray([distance, 0.0, 0.0])   # level in the tangent plane
        state = st.State(
            pos_ecef=anchor.T_e2l.T @ offset,
            vel_body=jnp.zeros(3), quat=jnp.asarray([1.0, 0.0, 0.0, 0.0]), omega=jnp.zeros(3),
        )
        assert float(st.pos_ned(state, anchor)[2]) == pytest.approx(0.0, abs=1e-6)
        divergence = float(st.altitude(state, anchor))
        assert divergence == pytest.approx(expected, rel=0.05), (
            f"tangent-plane divergence at {distance/1e3:.0f} km is {divergence:.1f} m, "
            f"expected ~{expected}"
        )
