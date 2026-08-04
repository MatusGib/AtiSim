import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flightsim import state as st


def euler_dcm_reference(phi, theta, psi):
    """Independent body -> NED 3-2-1 DCM, built from first principles.

    Exists to catch a convention error in quat_to_dcm/euler_to_quat that a
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
        np.asarray(st.quat_to_dcm(q)), euler_dcm_reference(phi, theta, psi), atol=1e-12
    )


@pytest.mark.parametrize("phi,theta,psi", ANGLE_CASES)
def test_euler_quat_round_trip(phi, theta, psi):
    q = st.euler_to_quat(jnp.array(phi), jnp.array(theta), jnp.array(psi))
    back = np.asarray(st.quat_to_euler(q))
    np.testing.assert_allclose(back, [phi, theta, psi], atol=1e-12)


def test_dcm_orthonormal_for_random_quaternions():
    key = jax.random.PRNGKey(0)
    qs = jax.random.normal(key, (200, 4))
    qs = qs / jnp.linalg.norm(qs, axis=1, keepdims=True)
    dcms = jax.vmap(st.quat_to_dcm)(qs)
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
    euler = np.asarray(st.quat_to_euler(q))
    expected = 0.1 * 0.1  # rate * elapsed time
    assert euler[index] == pytest.approx(expected, rel=1e-3), name
    others = [e for i, e in enumerate(euler) if i != index]
    assert np.allclose(others, 0.0, atol=1e-9)


def test_altitude_is_negative_down():
    s = st.State(
        pos_ned=jnp.array([0.0, 0.0, -3000.0]),
        vel_body=jnp.zeros(3),
        quat=jnp.array([1.0, 0.0, 0.0, 0.0]),
        omega=jnp.zeros(3),
    )
    assert float(st.altitude(s)) == pytest.approx(3000.0)


def test_x64_is_enabled():
    assert jnp.zeros(3).dtype == jnp.float64
