"""6-DOF equations of motion on a rotating, ellipsoidal Earth.

    rdot_e   = T_b2e v
    vdot     = F/m + g_b - (w_be + 2 Om_b) x v - [Om x (Om x r_e)]_b
    qdot     = 0.5 q (x) [0, w_be]                     (q: body -> ECEF)
    wdot_be  = I^-1 (M - w_bi x I w_bi) + w_be x Om_b

`v` is velocity relative to ECEF in body axes, `w_be` the body rate relative to
ECEF, `w_bi = w_be + Om_b` the body rate relative to ECI, `Om_b = T_e2b Om` with
`Om = [0,0,Omega]` in ECEF, and `F` excludes gravity. THESE WERE ESTABLISHED
EMPIRICALLY FROM THE JSBSim BINARY, not transcribed -- JSBSim ships compiled
with no source, so a transcription could not have been checked. Residuals are in
the design doc section 2.

The Coriolis terms use the *inertial* velocity and angular rate. Only the
aerodynamics see air-relative quantities.
"""

import jax.numpy as jnp
from jax import Array

from atisim import earth
from atisim.aero import V_MIN, aero_forces_moments, thrust_force
from atisim.aircraft import Aircraft
from atisim.atmosphere import G0, RHO0, density, speed_of_sound
from atisim.loads import CoeffIncrement
from atisim.state import Controls, State, quat_derivative, quat_to_matrix


def relative_velocity(vel_body: Array, quat: Array, wind_ned: Array) -> Array:
    """Body-axis velocity relative to the surrounding air mass.

    `quat` is body -> NED, which is the frame `wind_ned` is stated in. The state
    quaternion is body -> ECEF and is NOT that; `derivatives` forms the local
    frame at the aircraft's own position instead of calling this.
    """
    dcm = quat_to_matrix(quat)  # body -> NED
    return vel_body - dcm.T @ wind_ned


def earth_acceleration_terms(
    force: Array,
    moment: Array,
    mass: Array,
    inertia: Array,
    inertia_inv: Array,
    vel_body: Array,
    omega: Array,
    quat_body_to_ecef: Array,
    r_ecef: Array,
    model: earth.EarthModel,
) -> tuple[Array, Array]:
    """The rotating-Earth equations of motion, isolated so they can be tested alone.

    Established empirically from JSBSim v1.3.1, not transcribed -- the residuals
    are in the design doc section 2. `force` EXCLUDES gravity; gravity is added
    here so the caller cannot double-count it with the centrifugal term.

        vdot    = F/m + g - (omega_be + 2 Omega_b) x v - [Omega x (Omega x r)]_b
        wdot_be = I^-1 (M - omega_bi x I omega_bi) + omega_be x Omega_b

    THE PLUS IN THE SECOND LINE IS MEASURED, NOT DERIVED-AND-HOPED. The first
    probe of it was inconclusive: near wings-level the term is ~1e-5 and so is
    the wrong sign's residual, so both signs "fit", and the wrong one would have
    shipped. Re-probed at |omega_be| ~ 0.15 rad/s they separate cleanly --
    6.9e-18 for plus against 1.9e-5 for minus.
    `test_the_frame_transfer_sign_is_plus_and_a_wings_level_probe_cannot_tell`
    pins that, and carries the per-probe measurement on the frozen grid.

    `r_ecef` is the ABSOLUTE ECEF position, not the state's anchor-relative
    offset: both gravity and the centrifugal term are functions of where the
    aircraft actually is on the Earth.
    """
    t_b2e = quat_to_matrix(quat_body_to_ecef)
    omega_ecef = jnp.array([0.0, 0.0, model.rotation_rate])
    omega_earth_body = t_b2e.T @ omega_ecef

    gravity_body = t_b2e.T @ earth.gravitation(r_ecef, model)
    centrifugal_body = t_b2e.T @ jnp.cross(omega_ecef, jnp.cross(omega_ecef, r_ecef))

    accel = (
        force / mass
        + gravity_body
        - jnp.cross(omega + 2.0 * omega_earth_body, vel_body)
        - centrifugal_body
    )

    omega_bi = omega + omega_earth_body
    omega_dot_bi = inertia_inv @ (moment - jnp.cross(omega_bi, inertia @ omega_bi))
    omega_dot_be = omega_dot_bi + jnp.cross(omega, omega_earth_body)
    return accel, omega_dot_be


def derivatives(
    state: State,
    controls: Controls,
    ac: Aircraft,
    wind_ned: Array,
    omega_gust: Array,
    anchor: earth.Anchor,
    earth_model: earth.EarthModel,
    increment: CoeffIncrement | None = None,
    alphadot_gust: Array = 0.0,
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

    `anchor` fixes where on the Earth the run is flying and `earth_model` which
    Earth it is. Neither has a default: latitude now changes the answer, through
    Coriolis, gravity magnitude and the trimmed bank angle alike.
    """
    t_b2e = quat_to_matrix(state.quat)
    r_ecef = anchor.r_ecef + state.pos_ecef
    # The local frame is taken at the AIRCRAFT's position, not the anchor's:
    # the two verticals diverge with range, and being right at range is the
    # whole reason the ellipsoid is here. Altitude is geodetic for the same
    # reason -- the tangent plane falls away as d^2/2R, 785 m at 100 km.
    lat, lon, h = earth.ecef_to_geodetic(r_ecef)
    dcm_b2n = earth.ecef_to_ned_matrix(lat, lon) @ t_b2e

    vel_rel = state.vel_body - dcm_b2n.T @ wind_ned
    omega_rel = state.omega - omega_gust

    rho = density(h)
    a_sound = speed_of_sound(h)
    # Air-relative Mach, and unfloored for the same reason aero.py does not
    # floor it: Mach is finite at V = 0, so a floor would report thrust the
    # aircraft does not have.
    mach = jnp.linalg.norm(vel_rel) / a_sound
    thrust = thrust_force(controls, ac, rho, mach)

    def accelerate(alphadot):
        f, m = aero_forces_moments(
            vel_rel, omega_rel, controls, ac, rho, a_sound,
            increment=increment, alphadot_gust=alphadot,
        )
        f = f + thrust
        a, wdot = earth_acceleration_terms(
            f, m, ac.mass, ac.inertia, ac.inertia_inv,
            state.vel_body, state.omega, state.quat, r_ecef, earth_model,
        )
        return f, m, a, wdot

    # --- angle-of-attack rate, both halves -------------------------------
    # The WIND half arrives as alphadot_gust and is explicit. The AIRCRAFT half
    # is implicit -- alphadot depends on the acceleration, which depends on the
    # forces, which depend on alphadot -- so it takes a pass to open the loop:
    # evaluate the aero without it, read the acceleration that produces, and
    # feed the resulting alphadot back in.
    #
    # d(vel_rel)/dt = d(vel_body)/dt - d(wind_body)/dt, and the second term is
    # exactly what alphadot_gust already carries, so the two halves add without
    # overlapping.
    #
    # ONE pass is EXACT whenever CLadot is zero, which it is for every aircraft
    # in this registry: with no lift-due-to-alphadot there is no path from
    # alphadot back to the vertical acceleration, and the "implicit" loop is not
    # actually closed. With a non-zero CLadot the residual is O(B^2) where
    # B = -qbar*S*CLadot*c / (2 m V^2) is the loop gain -- about 1e-3 for a
    # transport, so O(1e-6) of a term that is itself a correction. If an
    # aircraft ever carries a large CLadot, replace this with the closed-form
    # 1/(1 - B) factor rather than more passes.
    _, _, accel_open, _ = accelerate(alphadot_gust)
    u_rel, w_rel = vel_rel[0], vel_rel[2]
    denominator = jnp.maximum(u_rel**2 + w_rel**2, V_MIN**2)
    alphadot_aircraft = (u_rel * accel_open[2] - w_rel * accel_open[0]) / denominator

    _, _, accel, omega_dot = accelerate(alphadot_gust + alphadot_aircraft)

    return State(
        pos_ecef=t_b2e @ state.vel_body,
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
    anchor: earth.Anchor,
    earth_model: earth.EarthModel,
    increment: CoeffIncrement | None = None,
) -> Array:
    """Body-axis specific force in g: what a three-axis accelerometer at the CG reads.

    Specific force is the aerodynamic plus propulsive force over mass and
    excludes gravity. `derivatives` computes exactly that quantity as
    `force / ac.mass` and then discards it inside the sum at the top of this
    module, so it is recovered here by inverting that sum. On a rotating Earth
    that sum has grown: it carries THREE non-force terms where it carried two,
    and one of the two it kept has changed shape.

        vdot = F/m + g_b - (omega_be + 2 Omega_b) x v - [Om x (Om x r)]_b
        =>  F/m = vdot - g_b + (omega_be + 2 Omega_b) x v + [Om x (Om x r)]_b

    Gravity is no longer the constant `G0` along the body z axis but
    `earth.gravitation` in body axes, and the centrifugal term has to be added
    back too -- it is part of what `derivatives` subtracted, and leaving it out
    would put 0.034 m/s^2 of un-inverted acceleration into the accelerometer at
    the equator. The Coriolis term now carries `2 Omega_b` alongside the body
    rate, for the same reason.

    Rather than restate any of that arithmetic, this evaluates
    `earth_acceleration_terms` WITH A ZERO FORCE. That gives exactly
    `g_b - (omega_be + 2 Omega_b) x v - [Om x (Om x r)]_b`, the whole non-force
    remainder, so the subtraction below is the algebra above with nothing
    transcribed twice. A sign or a term added to the plant's non-force side
    reaches this function automatically; a second hand-written copy of the
    inversion is precisely what could silently drift out of step with it. The
    moment argument is zero and the returned angular acceleration is discarded:
    only the translational row is being inverted. Measured on a 737 at 9,144 m
    and 47N with wind, gust and all three controls deflected, this recovers the
    aero-plus-thrust force to 2.2e-11 N out of 4.02e5 N -- 5.4e-17 relative,
    which is the float64 cancellation floor and nothing else -- and gives the
    same answer under `WGS84_J2` and `FLAT`.

    Inverting rather than recomputing `force / mass` is deliberate. It cannot
    silently disagree with the plant if a force term is ever added to
    `derivatives`, because it inverts whatever `derivatives` actually did.

    The result is still divided by the CONSTANT `G0`, not by local gravity.
    "In g" here means the conventional unit, so that a load factor is comparable
    between latitudes rather than being rescaled by the 0.5% the WGS-84 gravity
    magnitude varies over.

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
    d = derivatives(
        state, controls, ac, wind_ned, omega_gust, anchor, earth_model,
        increment=increment,
    )
    non_force, _ = earth_acceleration_terms(
        jnp.zeros(3), jnp.zeros(3), ac.mass, ac.inertia, ac.inertia_inv,
        state.vel_body, state.omega, state.quat,
        anchor.r_ecef + state.pos_ecef, earth_model,
    )
    return (d.vel_body - non_force) / G0


def load_factor(
    state: State,
    controls: Controls,
    ac: Aircraft,
    wind_ned: Array,
    omega_gust: Array,
    anchor: earth.Anchor,
    earth_model: earth.EarthModel,
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
    return -specific_force(
        state, controls, ac, wind_ned, omega_gust, anchor, earth_model, increment
    )[2]


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
    ac: Aircraft, trim_throttle: Array, altitude: Array, mach: Array = 0.0
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
    # `mach` defaults to zero, which switches the ram term off. Every existing
    # caller is a lee-wave or microburst case flown by an aircraft with
    # mach_ram = 0, so the term is identically 1 for them either way; the
    # argument exists so a ram-carrying aircraft is not silently under-thrusted.
    available = (
        ac.max_thrust
        * (density(altitude) / RHO0) ** ac.thrust_lapse
        * (1.0 + ac.mach_ram * mach**2)
    )
    drag = trim_throttle * available
    weight = ac.mass * G0
    return (available - drag) / weight, -drag / weight
