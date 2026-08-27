"""Integrator/rigid-body invariants that need no external source data.

Everything after this in the validation pass assumes the integrator and the
derivative chain are trustworthy. These checks establish that independently of
any aircraft data: a torque-free rigid body must conserve angular momentum and
rotational kinetic energy, and a wings-level coordinated turn must obey
psi_dot = g*tan(phi)/V regardless of which aircraft flies it.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from atisim import earth, integrate, trim
from atisim.aircraft import CRUISE, REGISTRY
from atisim.atmosphere import G0
from atisim.dynamics import derivatives
from atisim.state import (
    Controls,
    euler_to_quat,
    quat_to_euler_ned,
    quat_to_matrix,
    state_from_ned,
)
from atisim.tests.conftest import make_test_aircraft
from atisim.units import DEG2RAD

# 47N, the latitude this project's Earth-rotation work uses throughout, at each
# test's own altitude. There is no default anchor: latitude changes the answer.
ANCHOR_1KM = earth.anchor_at(np.radians(47.0), 0.0, 1000.0)

_ZERO_AERO_FIELDS = [
    "CD0", "CL0", "CLa", "CLq", "CLde", "Cm0", "Cma", "Cmq", "Cmde",
    "CYb", "CYp", "CYr", "CYdr", "Clb", "Clp", "Clr", "Clda", "Cldr",
    "Cnb", "Cnp", "Cnr", "Cnda", "Cndr", "max_thrust",
]


def _zero_aero_aircraft():
    """The synthetic test aircraft with every aero/thrust term zeroed.

    Ixz = 200 (nonzero, asymmetric) so the free-rotation check exercises the
    same inertia cross-coupling path test_gyroscopic_coupling_requires_ixz
    does. Gravity still acts (it produces no torque about the CG), so only the
    rotational states are conserved invariants here -- not the translational
    ones.
    """
    ac = make_test_aircraft(Ixz=200.0)
    return ac._replace(**{f: jnp.array(0.0) for f in _ZERO_AERO_FIELDS})


def test_free_rigid_body_conserves_angular_momentum_and_energy():
    """600 s / 60,000 steps of torque-free tumbling (tilted spin axis).

    Angular momentum (expressed in the inertial frame) and rotational kinetic
    energy are the two classical invariants of torque-free rigid-body motion.
    Tolerances are set an order of magnitude looser than the drift actually
    measured on this run (float64 RK4, ~2.4e5 derivative evaluations), which
    leaves headroom for platform variance while still catching a real
    integrator or cross-product-term regression.

    FLOWN UNDER `earth.FLAT`, AND THAT IS THE WHOLE POINT OF THIS TEST. The
    invariant is a statement about the INERTIAL frame; `state.omega` is the body
    rate relative to ECEF, and ECEF is only inertial when the Earth is not
    turning. Under `WGS84_J2` the conserved quantity is `I omega_bi` with
    `omega_bi = omega_be + Omega_b`, so rotating `I omega_be` into ECEF and
    calling it conserved would be measuring the Earth's rotation, not the
    integrator's drift -- which is what this file exists to bound. `FLAT` sets
    `rotation_rate = 0`, `omega_bi = omega_be`, and ECEF IS inertial, so the
    assertion below is once again exactly the classical one.
    """
    ac = _zero_aero_aircraft()
    omega0 = jnp.array([0.5, 0.3, -0.2])  # not aligned with any principal axis
    state0 = state_from_ned(
        jnp.zeros(3),
        jnp.zeros(3),
        euler_to_quat(jnp.array(0.3), jnp.array(-0.2), jnp.array(0.5)),
        omega0,
        ANCHOR_1KM,
    )
    controls = Controls(
        elevator=jnp.array(0.0), aileron=jnp.array(0.0),
        rudder=jnp.array(0.0), throttle=jnp.array(0.0),
    )

    dt, seconds = 0.01, 600.0
    n = int(round(seconds / dt))
    _, hist = integrate.rollout(
        integrate.init_sim(state0, jax.random.PRNGKey(0)), controls, jnp.array(dt), ac,
        n, ANCHOR_1KM, earth.FLAT,
    )

    omega_hist = np.asarray(hist.omega)
    quat_hist = np.asarray(hist.quat)
    I = np.asarray(ac.inertia)

    L_body = omega_hist @ I.T
    # body -> ECEF, and under FLAT that frame is inertial, which is the frame
    # the invariant is a statement about.
    dcms = np.asarray(jax.vmap(quat_to_matrix)(jnp.asarray(quat_hist)))
    L_ned = np.einsum("nij,nj->ni", dcms, L_body)

    L_mag = np.linalg.norm(L_ned, axis=1)
    mag_drift = np.abs(L_mag - L_mag[0]).max() / L_mag[0]
    assert mag_drift < 1e-11  # measured 5.6958e-13, as PROJECT.md section 4 records

    L_dir0 = L_ned[0] / L_mag[0]
    cos_angle = np.clip((L_ned / L_mag[:, None] * L_dir0).sum(axis=1), -1.0, 1.0)
    direction_drift_deg = np.degrees(np.arccos(cos_angle)).max()
    assert direction_drift_deg < 3e-5  # measured ~1.5e-6 deg

    KE = 0.5 * np.einsum("ni,ij,nj->n", omega_hist, I, omega_hist)
    ke_drift = np.abs(KE - KE[0]).max() / KE[0]
    assert ke_drift < 2e-11  # measured ~9.3e-13


def test_coordinated_turn_matches_g_tan_phi_over_v():
    """25.4 deg bank must give psi_dot close to the textbook g*tan(phi)/V.

    The bank/elevator/aileron/rudder are found by a 3x3 Newton solve (mirrors
    trim.trim's pattern) that zeros the three rotational accelerations at the
    kinematically-consistent steady-turn body rates, with beta fixed at 0 and
    theta fixed at the level-flight trim alpha -- i.e. NOT a full 7-unknown
    turning trim. Those two fixed approximations are exactly why a small
    residual is expected and bounded here rather than driven to zero.
    """
    ac = REGISTRY["boeing747"]
    V = CRUISE["boeing747"]["airspeed"]
    H = CRUISE["boeing747"]["altitude"]

    anchor = earth.anchor_at(np.radians(47.0), 0.0, H)

    # WGS84_J2, not FLAT: this one IS a statement about how the aircraft flies,
    # and g*tan(phi)/V is the textbook flat-Earth answer it is measured against.
    x, _ = trim.trim(jnp.array(V), jnp.array(H), ac, anchor, earth.WGS84_J2)
    alpha0, elevator0, throttle0 = float(x[0]), float(x[1]), float(x[2])

    phi = 25.4 * DEG2RAD
    theta = alpha0
    omega_analytic = G0 * np.tan(phi) / V
    p = -omega_analytic * np.sin(theta)
    q = omega_analytic * np.sin(phi) * np.cos(theta)
    r = omega_analytic * np.cos(phi) * np.cos(theta)

    vel_body = jnp.array([V * np.cos(alpha0), 0.0, V * np.sin(alpha0)])
    omega0 = jnp.array([p, q, r])
    quat0 = euler_to_quat(jnp.array(phi), jnp.array(theta), jnp.array(0.0))
    # AT the anchor, whose h is already H, so the aircraft starts at the trim
    # altitude with a zero offset rather than 12 km down the tangent plane.
    state0 = state_from_ned(jnp.zeros(3), vel_body, quat0, omega0, anchor)

    def rotational_residual(u):
        de, da, dr = u
        controls = Controls(elevator=de, aileron=da, rudder=dr, throttle=jnp.array(throttle0))
        return derivatives(
            state0, controls, ac, jnp.zeros(3), jnp.zeros(3), anchor, earth.WGS84_J2
        ).omega

    @jax.jit
    def solve(u0):
        def step(u, _):
            residual = rotational_residual(u)
            jacobian = jax.jacfwd(rotational_residual)(u)
            return u - jnp.linalg.solve(jacobian, residual), None

        u, _ = jax.lax.scan(step, u0, None, length=20)
        return u

    u = solve(jnp.array([elevator0, 0.0, 0.0]))
    assert float(jnp.linalg.norm(rotational_residual(u))) < 1e-10

    controls = Controls(elevator=u[0], aileron=u[1], rudder=u[2], throttle=jnp.array(throttle0))

    dt, seconds = 0.02, 1.0
    n = int(round(seconds / dt))
    _, hist = integrate.rollout(
        integrate.init_sim(state0, jax.random.PRNGKey(0)), controls, jnp.array(dt), ac,
        n, anchor, earth.WGS84_J2,
    )
    euler = np.asarray(jax.vmap(quat_to_euler_ned, in_axes=(0, None))(hist, anchor))
    psi_rate = np.unwrap(euler[:, 2])[-1] / seconds

    error = abs(psi_rate - omega_analytic) / omega_analytic
    # measured ~1.1%; source's own ad hoc figure was 1.55%. THE ROTATING EARTH
    # DID NOT MOVE THIS: 1.0998% under WGS84_J2 against 1.1206% under FLAT, so
    # the 1.1% is the beta = 0 / theta = alpha approximation above and not the
    # Earth model. Worth the two numbers, because a §4 baseline that survives a
    # stated physical change is evidence, and a reader would otherwise assume
    # this one had simply not been re-measured.
    assert error < 0.03
    assert abs(euler[-1, 0] - phi) < 1.0 * DEG2RAD  # bank actually held
