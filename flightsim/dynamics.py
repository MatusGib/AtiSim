"""6-DOF Newton-Euler equations of motion.

    vdot     = F/m + g_body - omega x v
    omegadot = I^-1 (M - omega x (I omega))
    posdot   = DCM v
    quatdot  = 0.5 q (x) [0, omega]

The Coriolis terms use the *inertial* velocity and angular rate. Only the
aerodynamics see air-relative quantities.
"""

import jax.numpy as jnp
from jax import Array

from flightsim.aero import aero_forces_moments, thrust_force
from flightsim.aircraft import Aircraft
from flightsim.atmosphere import G0, RHO0, density, speed_of_sound
from flightsim.loads import CoeffIncrement
from flightsim.state import Controls, State, quat_derivative, quat_to_dcm


def relative_velocity(vel_body: Array, quat: Array, wind_ned: Array) -> Array:
    """Body-axis velocity relative to the surrounding air mass."""
    dcm = quat_to_dcm(quat)  # body -> NED
    return vel_body - dcm.T @ wind_ned


def derivatives(
    state: State,
    controls: Controls,
    ac: Aircraft,
    wind_ned: Array,
    omega_gust: Array,
    increment: CoeffIncrement | None = None,
) -> State:
    """State derivative. Returns a State whose fields are time derivatives.

    `increment` carries aerodynamic coefficients computed from the wind field
    ACROSS the airframe -- strip-integrated loads -- which `aero.py` cannot
    produce because its standing rule is that it never sees the field. It is
    summed into the coefficients here, at the same seam where wind already
    enters and nowhere else.

    None is passed straight through rather than being replaced by an explicit
    zero, so an omitted increment performs NO arithmetic at all and is therefore
    bit-identical to the pre-increment model by construction, not merely by the
    identity property of adding 0.0. That also keeps the two paths genuinely
    distinct, which is what gives
    `test_a_zero_increment_is_bit_identical_to_not_passing_one` something to
    test -- defaulting to `zero_increment()` here would make both branches run
    the same code and assert nothing.

    `specific_force` and `load_factor` forward it too. They invert this
    function's own sum rather than recomputing it, so only channels that entered
    `force` can reach them -- `CL`, and not `Cl`, `Cm` or `Cn`.
    """
    dcm = quat_to_dcm(state.quat)

    vel_rel = relative_velocity(state.vel_body, state.quat, wind_ned)
    omega_rel = state.omega - omega_gust

    altitude = -state.pos_ned[2]
    rho = density(altitude)
    force, moment = aero_forces_moments(
        vel_rel, omega_rel, controls, ac, rho, speed_of_sound(altitude),
        increment=increment,
    )
    force = force + thrust_force(controls, ac, rho)

    gravity_body = dcm.T @ jnp.array([0.0, 0.0, G0])

    accel = force / ac.mass + gravity_body - jnp.cross(state.omega, state.vel_body)
    omega_dot = ac.inertia_inv @ (
        moment - jnp.cross(state.omega, ac.inertia @ state.omega)
    )

    return State(
        pos_ned=dcm @ state.vel_body,
        vel_body=accel,
        quat=quat_derivative(state.quat, state.omega),
        omega=omega_dot,
    )


def specific_force(
    state: State,
    controls: Controls,
    ac: Aircraft,
    wind_ned: Array,
    omega_gust: Array,
    increment: CoeffIncrement | None = None,
) -> Array:
    """Body-axis specific force in g: what a three-axis accelerometer at the CG reads.

    Specific force is the aerodynamic plus propulsive force over mass and
    excludes gravity. `derivatives` computes exactly that quantity as
    `force / ac.mass` and then discards it inside the sum at the top of this
    module, so it is recovered here by inverting that sum:

        a_spec = vdot_body - g_body + omega x vel_body

    Inverting rather than recomputing `force / mass` is deliberate. It cannot
    silently disagree with the plant if a force term is ever added to
    `derivatives`, because it inverts whatever `derivatives` actually did.

    Signs are body axes throughout: +x forward, +y right, +z down. Note that
    `load_factor` NEGATES the z component, because the load-factor convention is
    +1 in level flight while a_spec[2] is negative there.

    `increment` is forwarded to `derivatives`, so an accelerometer on a strip run
    reads the loads the aircraft actually flew. Because this INVERTS the sum
    rather than recomputing it, only the channels that entered `force` can show
    up here -- CL, and not Cl, Cm or Cn. That is asserted rather than assumed by
    `test_a_lift_increment_reaches_the_load_factor`, and it is why omitting the
    increment was exactly, not approximately, right while `strip_increment`
    populated Cl alone.
    """
    d = derivatives(state, controls, ac, wind_ned, omega_gust, increment=increment)
    gravity_body = quat_to_dcm(state.quat).T @ jnp.array([0.0, 0.0, G0])
    return (d.vel_body - gravity_body + jnp.cross(state.omega, state.vel_body)) / G0


def load_factor(
    state: State,
    controls: Controls,
    ac: Aircraft,
    wind_ned: Array,
    omega_gust: Array,
    increment: CoeffIncrement | None = None,
) -> Array:
    """Normal load factor n_z. +1 in level flight, 0 in free fall.

    Body-normal, not flight-path-normal: this is the quantity Wingrove & Bach's
    Fig. 8 is built from, since DFDR "normal acceleration" is what an
    accelerometer reads. In trimmed level flight it is cos(theta), not 1.

    Needed because the turbulence work's headline comparison is stated in load
    factor, and nothing in the package produced it before.

    `increment` is forwarded. Every Fig. 8 load coordinate is built through here
    by `vortex_viz._measure`, so a strip run whose increment stopped short of
    this function would understate its own headline number.
    """
    return -specific_force(state, controls, ac, wind_ned, omega_gust, increment)[2]


# ---------------------------------------------------------------------------
# Windshear hazard
#
# Source: F. H. Proctor, D. A. Hinton (NASA Langley) and R. L. Bowles (AeroTech),
# "A Windshear Hazard Index", 9th Conf. on Aviation, Range and Aerospace
# Meteorology, Orlando, 11-15 Sept 2000, paper 7.7, pp. 482-487. The index is
# originally Bowles (1990a, 1990b); this paper is the one actually read here and
# is where the equation numbers below come from.
# ---------------------------------------------------------------------------


def f_factor(shear_rate: Array, w_up: Array, airspeed: Array) -> Array:
    """The Bowles F-factor. Proctor et al. Eq. (3): F = U_x_dot/g - w/V_a.

    Non-dimensional rate at which the wind field is draining the aircraft's
    total energy. POSITIVE IS HAZARDOUS, and the paper is explicit about both
    signs: F is positive "for a descending air mass (w < 0) and a wind field
    accelerating in the direction of the flight path".

    `shear_rate` is `U_x_dot`, the rate of change of the along-track horizontal
    wind (positive for a tailwind) experienced by the aircraft -- see
    `wind.along_track_shear`. `w_up` is the airmass vertical velocity, positive
    UP. Both in SI.

    The paper develops this for low-level windshear and the FAA's 0.1 alerting
    threshold is calibrated for takeoff and landing; §4.1 explicitly bounds the
    threat to below 500 m, because higher up an aircraft has potential energy to
    trade. None of that transfers to 12 km, so the threshold is NOT used here.
    What transfers is the index itself and the paper's own thrust criterion --
    see `thrust_authority`.
    """
    return shear_rate / G0 - w_up / airspeed


def average_f_factor(
    f: Array, distance: Array, length: float = 1000.0
) -> tuple[Array, Array]:
    """Running forward mean of F over `length` m of track. Proctor et al. Eq. (7).

    **This, not the instantaneous value, is the hazard metric.** The paper is
    blunt about why: "Peaks of F are over small length scales and are quickly
    followed by negative values. Such oscillations of F-factor result in
    perceived turbulence, with airspeed oscillations and little net trajectory
    change." A spike the aircraft flies through before it can respond is not the
    same threat as a sustained loss, and only the average distinguishes them.

    The FAA adopted the **1 km** average as its metric for windshear detection on
    jet transports, hazardous above 0.1 with a must-alert threshold at 0.13. The
    same paper records that this was never established for piston aircraft: "The
    minimum averaging scale and hazard threshold are yet to be determined for
    these types of aircraft."

    Returns `(average, valid)`. `valid` is False where the window would run off
    the end of the run, because `jnp.interp` clamps there rather than extending
    and those entries would understate a rising hazard. Never take a peak
    without masking.
    """
    f = jnp.asarray(f)
    distance = jnp.asarray(distance)
    integral = jnp.concatenate([
        jnp.zeros(1),
        jnp.cumsum(jnp.diff(distance) * 0.5 * (f[1:] + f[:-1])),
    ])
    ahead = jnp.interp(distance + length, distance, integral)
    return (ahead - integral) / length, (distance + length) <= distance[-1]


def thrust_authority(
    ac: Aircraft, trim_throttle: Array, altitude: Array
) -> tuple[Array, Array]:
    """The (T - D)/W envelope in level flight: (full throttle, idle).

    The comparison the F-factor is FOR. Proctor et al., discussing their Eq. (5):
    "For a strong shear that exceeds the thrust capability of the aircraft, i.e.
    F > (T_r - D)/W, a pilot may either manage his flight so as to maintain
    altitude while decelerating, maintain airspeed while descending, or some
    compromise of the two."

    In trimmed level flight T = D, so the drag is the trim thrust and the
    envelope is just how far the throttle can travel either way, over the
    weight. That is why this needs a trim solution and not a drag model.

    For scale, the same paper gives 0.15 for a 4-engine jet at full thrust and
    maximum takeoff weight. A 747 at 40,000 ft has far less, and that is the
    physical point rather than a discrepancy: thrust available falls with
    density while weight does not.
    """
    available = ac.max_thrust * (density(altitude) / RHO0) ** ac.thrust_lapse
    drag = trim_throttle * available
    weight = ac.mass * G0
    return (available - drag) / weight, -drag / weight
