"""Simulation state and quaternion utilities.

Conventions, fixed everywhere in this package:

  Frames   ECEF is the propagation frame. LOCAL NED is a derived VIEW, taken at
           the aircraft's own geodetic position -- not at the anchor's, because
           the two diverge with range and the whole point of carrying an
           ellipsoid is to be right at range.
  Position `pos_ecef` is an OFFSET from the run anchor, not an absolute ECEF
           coordinate. This is the one deliberate deviation from JSBSim's
           FGPropagate, and it is numerical, not physical: an absolute
           coordinate's ulp is 512x coarser and would bind the convergence
           studies. See ASSUMPTIONS.md F4.
  Quat     [w, x, y, z], unit norm, rotates BODY vectors into ECEF.
  Omega    body rate relative to ECEF (JSBSim's vPQR), body axes.
  Euler    3-2-1 (yaw psi, pitch theta, roll phi) in the LOCAL NED frame.
           Display only -- the state carries the quaternion.
  Altitude GEODETIC, via `altitude(state, anchor)`. It is NOT -pos_ned[2] any
           more, and the two differ by 785 m at 100 km of ground track.

`quat_to_dcm` HAS BEEN DELETED, deliberately. It meant body->NED and would now
mean body->ECEF; both are valid rotations, so every call site would have kept
running and returned a wrong answer. The generic `quat_to_matrix` is
frame-agnostic, and the two named helpers say which frame they mean.
"""

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from atisim import earth


class State(NamedTuple):
    """13-element rigid-body state, on a rotating Earth."""

    pos_ecef: Array  # (3,) m, OFFSET from the run anchor
    vel_body: Array  # (3,) m/s, ECEF-relative, body axes (JSBSim vUVW)
    quat: Array      # (4,) w x y z, body -> ECEF
    omega: Array     # (3,) rad/s, body rate relative to ECEF (JSBSim vPQR)


class Controls(NamedTuple):
    elevator: Array  # rad, positive trailing-edge down
    aileron: Array   # rad, positive right-roll command
    rudder: Array    # rad, positive trailing-edge left
    throttle: Array  # 0-1


def quat_normalize(q: Array) -> Array:
    return q / jnp.linalg.norm(q)


def quat_to_matrix(q: Array) -> Array:
    """Rotation matrix from a quaternion. FRAME-AGNOSTIC on purpose.

    For a state quaternion this is body -> ECEF. Use `dcm_body_to_ned` when you
    want the local frame, and never assume which one you have.
    """
    w, x, y, z = q
    return jnp.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def dcm_to_quat(m: Array) -> Array:
    """Quaternion [w,x,y,z] from a rotation matrix. Branch-free Shepperd.

    All four candidates are computed and the one with the largest denominator is
    selected, which is the numerically stable choice: the naive trace form loses
    all its precision near trace = -1. Branch-free because it runs under jit.
    """
    c = jnp.array([
        1.0 + m[0, 0] + m[1, 1] + m[2, 2],
        1.0 + m[0, 0] - m[1, 1] - m[2, 2],
        1.0 - m[0, 0] + m[1, 1] - m[2, 2],
        1.0 - m[0, 0] - m[1, 1] + m[2, 2],
    ])
    i = jnp.argmax(c)
    s = jnp.sqrt(jnp.maximum(c[i], 1e-300)) * 2.0
    candidates = jnp.stack([
        jnp.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]),
        jnp.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]),
        jnp.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]),
        jnp.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]),
    ])
    return quat_normalize(candidates[i])


def absolute_ecef(state: State, anchor: earth.Anchor) -> Array:
    """The aircraft's ABSOLUTE ECEF position. Formed only where geodesy needs it."""
    return anchor.r_ecef + state.pos_ecef


def geodetic(state: State, anchor: earth.Anchor) -> tuple[Array, Array, Array]:
    """(lat, lon, h) of the aircraft. Latitude is geodetic."""
    return earth.ecef_to_geodetic(absolute_ecef(state, anchor))


def altitude(state: State, anchor: earth.Anchor) -> Array:
    """GEODETIC altitude, m. Not -pos_ned[2] -- see the module docstring."""
    return geodetic(state, anchor)[2]


def pos_ned(state: State, anchor: earth.Anchor) -> Array:
    """Local NED offset from the anchor, m.

    A pure ROTATION of the stored offset. Never a difference of two absolute
    ECEF coordinates -- that would inherit the 512x coarser ulp and undo the
    reason position is stored as an offset at all.
    """
    return anchor.T_e2l @ state.pos_ecef


def dcm_body_to_ned(state: State, anchor: earth.Anchor) -> Array:
    """Body -> local NED, at the AIRCRAFT's position, not the anchor's.

    The distinction is the point of the ellipsoid: the local vertical at the
    aircraft is not the local vertical at the anchor once they are any distance
    apart, and using the anchor's frame here would put that error straight into
    the Euler angles and every attitude loop.
    """
    lat, lon, _ = geodetic(state, anchor)
    return earth.ecef_to_ned_matrix(lat, lon) @ quat_to_matrix(state.quat)


def matrix_to_euler(dcm: Array) -> Array:
    """[phi, theta, psi] from a body -> NED matrix. Display only."""
    phi = jnp.arctan2(dcm[2, 1], dcm[2, 2])
    # Clip guards asin against a norm error of a few ulp at +-90 deg pitch.
    theta = -jnp.arcsin(jnp.clip(dcm[2, 0], -1.0, 1.0))
    psi = jnp.arctan2(dcm[1, 0], dcm[0, 0])
    return jnp.array([phi, theta, psi])


def quat_to_euler_ned(state: State, anchor: earth.Anchor) -> Array:
    """[phi, theta, psi] in rad, in the local NED frame. Display only."""
    return matrix_to_euler(dcm_body_to_ned(state, anchor))


def state_from_ned(pos_ned_, vel_body, quat_ned, omega, anchor: earth.Anchor) -> State:
    """Build a State from the local-NED description most call sites use.

    Exists so migrating the existing call sites is mechanical rather than a
    rewrite: they keep describing the aircraft in the terms they always did, and
    this converts once.

    `quat_ned` is body -> NED and `omega` is the body rate relative to ECEF.
    """
    pos_ecef = anchor.T_e2l.T @ jnp.asarray(pos_ned_)
    lat, lon, _ = earth.ecef_to_geodetic(anchor.r_ecef + pos_ecef)
    t_e2l = earth.ecef_to_ned_matrix(lat, lon)
    dcm_b2e = t_e2l.T @ quat_to_matrix(jnp.asarray(quat_ned))
    return State(
        pos_ecef=pos_ecef,
        vel_body=jnp.asarray(vel_body),
        quat=dcm_to_quat(dcm_b2e),
        omega=jnp.asarray(omega),
    )


def euler_to_quat(phi: Array, theta: Array, psi: Array) -> Array:
    cr, sr = jnp.cos(phi / 2), jnp.sin(phi / 2)
    cp, sp = jnp.cos(theta / 2), jnp.sin(theta / 2)
    cy, sy = jnp.cos(psi / 2), jnp.sin(psi / 2)
    return jnp.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )


def quat_derivative(q: Array, omega: Array) -> Array:
    """qdot = 0.5 * q (x) [0, omega], with omega the body-axis rates."""
    w, x, y, z = q
    p, qq, r = omega
    return 0.5 * jnp.array(
        [
            -x * p - y * qq - z * r,
            w * p + y * r - z * qq,
            w * qq + z * p - x * r,
            w * r + x * qq - y * p,
        ]
    )
