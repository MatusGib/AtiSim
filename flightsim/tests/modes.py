"""Linear-mode extraction, shared by the CR-2144 (747) and Navion validation tests.

Not part of the simulator -- these functions exist only to linearise a trimmed
flight condition (via jax.jacfwd of the real nonlinear `dynamics.derivatives`,
the same approach scripts/checkpoint.py uses) and extract phugoid,
short-period, Dutch roll, roll subsidence and spiral for comparison against
source-document reference values.
"""

import jax
import jax.numpy as jnp
import numpy as np

from flightsim.aircraft import Aircraft
from flightsim.dynamics import derivatives
from flightsim.state import Controls, State, euler_to_quat


def longitudinal_modes(ac: Aircraft, alpha: float, elevator: float, throttle: float,
                        V: float, H: float):
    """Phugoid and short-period (wn, zeta), sorted low-to-high wn.

    Linearises in body-axis [u, w, q, theta] about the given trim, exactly as
    scripts/checkpoint.py's longitudinal_modes does for the 747.
    """
    u0, w0 = V * np.cos(alpha), V * np.sin(alpha)
    controls = Controls(
        elevator=jnp.array(elevator), aileron=jnp.array(0.0),
        rudder=jnp.array(0.0), throttle=jnp.array(throttle),
    )

    def f(x):
        u, w, q, theta = x
        state = State(
            pos_ned=jnp.array([0.0, 0.0, -H]),
            vel_body=jnp.array([u, 0.0, w]),
            quat=euler_to_quat(jnp.array(0.0), theta, jnp.array(0.0)),
            omega=jnp.array([0.0, q, 0.0]),
        )
        d = derivatives(state, controls, ac, jnp.zeros(3), jnp.zeros(3))
        return jnp.array([d.vel_body[0], d.vel_body[2], d.omega[1], q])

    A = np.asarray(jax.jacfwd(f)(jnp.array([u0, w0, 0.0, alpha])))
    eig = np.linalg.eigvals(A)
    modes = [(abs(lam), -lam.real / abs(lam)) for lam in eig if lam.imag > 1e-9]
    return sorted(modes)  # phugoid (low wn) first, then short period


def lateral_modes(ac: Aircraft, alpha: float, elevator: float, throttle: float,
                   V: float, H: float):
    """Dutch roll (wn, zeta), roll-subsidence time constant, spiral time constant.

    4-state reduction [v, p, r, phi] about the wings-level trim, holding
    u = u0, w = w0, theta = theta0 fixed (standard small-perturbation lateral
    split, valid because the trim is wings-level and symmetric).

    thetadot's exact kinematic term keeps r*cos(phi)*tan(theta0), not just p:
    at this aircraft's trim pitch attitude that term is NOT negligible for the
    spiral root specifically (dropping it changes the spiral time constant by
    order 30%, confirmed by directly perturbing the trimmed nonlinear sim in
    bank and fitting the late-time decay of phi -- see
    test_cr2144_lateral_modes.py). Dutch roll and roll are insensitive to it.
    """
    theta0 = alpha
    u0, w0 = V * np.cos(alpha), V * np.sin(alpha)
    controls = Controls(
        elevator=jnp.array(elevator), aileron=jnp.array(0.0),
        rudder=jnp.array(0.0), throttle=jnp.array(throttle),
    )

    def f(x):
        v, p, r, phi = x
        state = State(
            pos_ned=jnp.array([0.0, 0.0, -H]),
            vel_body=jnp.array([u0, v, w0]),
            quat=euler_to_quat(phi, jnp.array(theta0), jnp.array(0.0)),
            omega=jnp.array([p, 0.0, r]),
        )
        d = derivatives(state, controls, ac, jnp.zeros(3), jnp.zeros(3))
        phidot = p + r * jnp.cos(phi) * jnp.tan(theta0)
        return jnp.array([d.vel_body[1], d.omega[0], d.omega[2], phidot])

    A = np.asarray(jax.jacfwd(f)(jnp.array([0.0, 0.0, 0.0, 0.0])))
    eig = np.linalg.eigvals(A)
    dutch_roll = None
    reals = []
    for lam in eig:
        if lam.imag > 1e-9:
            dutch_roll = (abs(lam), -lam.real / abs(lam))
        elif abs(lam.imag) <= 1e-9:
            reals.append(-1.0 / lam.real)
    reals.sort()  # roll subsidence is fast (small tau), spiral is slow
    roll_tau, spiral_tau = reals[0], reals[1]
    return dutch_roll, roll_tau, spiral_tau
