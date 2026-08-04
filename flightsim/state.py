"""Simulation state and quaternion utilities.

Conventions, fixed everywhere in this package:

  Frames   NED (North-East-Down) inertial; body axes x-forward, y-right, z-down.
  Quat     [w, x, y, z], unit norm, rotates BODY vectors into NED.
  Euler    3-2-1 (yaw psi, pitch theta, roll phi). Display only -- the state
           carries the quaternion, never Euler angles.
  Altitude -pos_ned[2].
"""

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array


class State(NamedTuple):
    """13-element rigid-body state."""

    pos_ned: Array  # (3,) m
    vel_body: Array  # (3,) m/s, u v w
    quat: Array  # (4,) w x y z, body -> NED
    omega: Array  # (3,) rad/s, p q r


class Controls(NamedTuple):
    elevator: Array  # rad, positive trailing-edge down
    aileron: Array  # rad, positive right-roll command
    rudder: Array  # rad, positive trailing-edge left
    throttle: Array  # 0-1


def altitude(state: State) -> Array:
    return -state.pos_ned[2]


def quat_normalize(q: Array) -> Array:
    return q / jnp.linalg.norm(q)


def quat_to_dcm(q: Array) -> Array:
    """Body -> NED rotation matrix. v_ned = quat_to_dcm(q) @ v_body."""
    w, x, y, z = q
    return jnp.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def quat_to_euler(q: Array) -> Array:
    """[phi, theta, psi] in rad. Display only."""
    dcm = quat_to_dcm(q)
    phi = jnp.arctan2(dcm[2, 1], dcm[2, 2])
    # Clip guards asin against a norm error of a few ulp at +-90 deg pitch.
    theta = -jnp.arcsin(jnp.clip(dcm[2, 0], -1.0, 1.0))
    psi = jnp.arctan2(dcm[1, 0], dcm[0, 0])
    return jnp.array([phi, theta, psi])


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
