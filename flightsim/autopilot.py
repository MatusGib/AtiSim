"""Cascaded PID autopilot.

Stateless and functional: all memory lives in `APState`, so the whole thing goes
inside `lax.scan` and vmaps over a batch like everything else.

Two cascade levels, no more:

    altitude  -> pitch command -> elevator
    heading   -> bank command  -> aileron
    airspeed  -> throttle
    sideslip  -> rudder

Deflection limits come from the aircraft; rate limits are applied to the output
before it reaches the plant, not inside the loops. Integrators use conditional
integration for anti-windup -- they stop accumulating when the surface they
drive is already on its stop and the error would push it further.
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from flightsim.aircraft import Aircraft
from flightsim.integrate import SimState, step
from flightsim.sensors import AirData, sense
from flightsim.state import Controls
from flightsim.wind import zero_wind


class Targets(NamedTuple):
    altitude: Array  # m
    heading: Array  # rad
    airspeed: Array  # m/s


class Gains(NamedTuple):
    alt_p: Array  # altitude error -> pitch command
    alt_i: Array
    theta_limit: Array
    theta_p: Array  # pitch error -> elevator
    theta_i: Array
    q_d: Array
    hdg_p: Array  # heading error -> bank command
    phi_limit: Array
    phi_p: Array  # bank error -> aileron
    phi_i: Array
    p_d: Array
    beta_p: Array  # sideslip -> rudder
    spd_p: Array  # airspeed error -> throttle
    spd_i: Array
    surface_rate: Array  # rad/s
    throttle_rate: Array  # per s


class APState(NamedTuple):
    alt_i: Array
    theta_i: Array
    phi_i: Array
    spd_i: Array
    controls: Controls  # last output, for rate limiting and for handback


def wrap_pi(angle: Array) -> Array:
    return (angle + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


def _accumulate(
    acc: Array, err: Array, dt: Array, command: Array, limit: Array, bound: Array
) -> Array:
    """Conditional integration plus a hard bound on the accumulated state.

    Conditional integration alone is not enough. It stops the *command* running
    away, but the integrator state can still sit at a huge value that was only
    meaningful against the old error -- and integral gains here are small, so
    "huge" is the normal operating value, not an anomaly. Change the target and
    that stale state commands full deflection until it unwinds, which takes as
    long as it took to build.

    `bound` is chosen per loop so the integral term alone can never ask for more
    than the actuator or command limit, which caps the recovery time.
    """
    blocked = ((command >= limit) & (err > 0)) | ((command <= -limit) & (err < 0))
    return jnp.clip(jnp.where(blocked, acc, acc + err * dt), -bound, bound)


def _rate_limit(new: Array, old: Array, rate: Array, dt: Array) -> Array:
    return old + jnp.clip(new - old, -rate * dt, rate * dt)


def autopilot(
    air: AirData,
    ap: APState,
    targets: Targets,
    gains: Gains,
    ac: Aircraft,
    dt: Array,
) -> tuple[Controls, APState]:
    """One control step from sensed air data.

    This takes `AirData` rather than `State` on purpose. Given a `State` the
    natural thing to write is `air_data(state.vel_body)`, which is inertial --
    so under wind the speed loop regulates GROUNDSPEED and the sideslip-to-rudder
    term receives a flow angle no vane would produce. That was a real defect
    here for three sessions. Taking the sensor set as an argument means the
    caller has to have decided what the air is doing.
    """
    phi, theta, psi = air.phi, air.theta, air.psi
    p, q, r = air.p, air.q, air.r
    altitude = air.altitude
    airspeed, beta = air.airspeed, air.beta

    # --- outer: altitude -> pitch command ---
    alt_err = targets.altitude - altitude
    theta_cmd_raw = gains.alt_p * alt_err + gains.alt_i * ap.alt_i
    theta_cmd = jnp.clip(theta_cmd_raw, -gains.theta_limit, gains.theta_limit)
    alt_i = _accumulate(
        ap.alt_i, alt_err, dt, theta_cmd_raw, gains.theta_limit,
        gains.theta_limit / gains.alt_i,
    )

    # --- inner: pitch -> elevator. Positive elevator is nose-down, hence the sign.
    theta_err = theta_cmd - theta
    elevator_raw = -(gains.theta_p * theta_err + gains.theta_i * ap.theta_i - gains.q_d * q)
    elevator = jnp.clip(elevator_raw, -ac.elevator_limit, ac.elevator_limit)
    theta_i = _accumulate(
        ap.theta_i, theta_err, dt, -elevator_raw, ac.elevator_limit,
        ac.elevator_limit / gains.theta_i,
    )

    # --- outer: heading -> bank command ---
    hdg_err = wrap_pi(targets.heading - psi)
    phi_cmd_raw = gains.hdg_p * hdg_err
    phi_cmd = jnp.clip(phi_cmd_raw, -gains.phi_limit, gains.phi_limit)

    # --- inner: bank -> aileron ---
    phi_err = phi_cmd - phi
    aileron_raw = gains.phi_p * phi_err + gains.phi_i * ap.phi_i - gains.p_d * p
    aileron = jnp.clip(aileron_raw, -ac.aileron_limit, ac.aileron_limit)
    phi_i = _accumulate(
        ap.phi_i, phi_err, dt, aileron_raw, ac.aileron_limit,
        ac.aileron_limit / gains.phi_i,
    )

    # --- turn coordination: drive sideslip to zero ---
    rudder = jnp.clip(-gains.beta_p * beta, -ac.rudder_limit, ac.rudder_limit)

    # --- airspeed -> throttle. The integrator carries the trim setting. ---
    spd_err = targets.airspeed - airspeed
    throttle_raw = gains.spd_p * spd_err + gains.spd_i * ap.spd_i
    throttle = jnp.clip(throttle_raw, 0.0, 1.0)
    # Throttle is one-sided, so this integrator is bounded to [0, 1/gain] rather
    # than symmetrically: its job is to carry the trim setting.
    spd_i = jnp.clip(
        jnp.where(
            ((throttle_raw >= 1.0) & (spd_err > 0))
            | ((throttle_raw <= 0.0) & (spd_err < 0)),
            ap.spd_i,
            ap.spd_i + spd_err * dt,
        ),
        0.0,
        1.0 / gains.spd_i,
    )

    # --- rate limits, applied last, just before the plant ---
    out = Controls(
        elevator=_rate_limit(elevator, ap.controls.elevator, gains.surface_rate, dt),
        aileron=_rate_limit(aileron, ap.controls.aileron, gains.surface_rate, dt),
        rudder=_rate_limit(rudder, ap.controls.rudder, gains.surface_rate, dt),
        throttle=_rate_limit(throttle, ap.controls.throttle, gains.throttle_rate, dt),
    )
    return out, APState(alt_i=alt_i, theta_i=theta_i, phi_i=phi_i, spd_i=spd_i, controls=out)


def engage(
    air: AirData, controls: Controls, targets: Targets, gains: Gains, ac: Aircraft
) -> APState:
    """Seed the integrators so the first output equals the current controls.

    Without this the aircraft lurches the instant the autopilot is switched on,
    by exactly the difference between the trimmed deflections and whatever the
    proportional terms happen to ask for. Ten lines, and it is the most common
    bug in this kind of system.
    """
    phi, theta, psi = air.phi, air.theta, air.psi
    p, q, r = air.p, air.q, air.r
    altitude = air.altitude
    airspeed = air.airspeed

    # Choose alt_i so the pitch command equals the current pitch attitude: the
    # aircraft is then already tracking its own state and nothing moves.
    #
    # The seeds are clamped to the same bounds the running loops enforce. That
    # matters: engaging with a large target error would otherwise seed a state
    # the first update immediately clips, which is worse than not seeding at
    # all. Transfer is therefore exactly bumpless when engaging near the
    # targets, and a normal bounded capture when engaging far from them --
    # which is the correct behaviour in both cases.
    alt_err = targets.altitude - altitude
    alt_i = jnp.clip(
        (theta - gains.alt_p * alt_err) / gains.alt_i,
        -gains.theta_limit / gains.alt_i,
        gains.theta_limit / gains.alt_i,
    )

    # theta_err is zero by that construction, so the integrator carries the
    # whole current elevator deflection.
    theta_i = jnp.clip(
        (-controls.elevator - gains.q_d * q) / gains.theta_i,
        -ac.elevator_limit / gains.theta_i,
        ac.elevator_limit / gains.theta_i,
    )

    hdg_err = wrap_pi(targets.heading - psi)
    phi_cmd = jnp.clip(gains.hdg_p * hdg_err, -gains.phi_limit, gains.phi_limit)
    phi_i = jnp.clip(
        (controls.aileron - gains.phi_p * (phi_cmd - phi) + gains.p_d * p) / gains.phi_i,
        -ac.aileron_limit / gains.phi_i,
        ac.aileron_limit / gains.phi_i,
    )

    spd_err = targets.airspeed - airspeed
    spd_i = jnp.clip(
        (controls.throttle - gains.spd_p * spd_err) / gains.spd_i, 0.0, 1.0 / gains.spd_i
    )

    return APState(
        alt_i=alt_i, theta_i=theta_i, phi_i=phi_i, spd_i=spd_i, controls=controls
    )


@partial(jax.jit, static_argnames=("n_steps", "wind_model"))
def closed_loop_rollout(
    sim: SimState,
    ap: APState,
    targets: Targets,
    gains: Gains,
    dt: Array,
    ac: Aircraft,
    n_steps: int,
    wind_model=zero_wind,
):
    """Fly the autopilot inside lax.scan.

    This is the Monte Carlo path: vmap it over a batch of PRNG keys and it flies
    the same profile through many turbulence realisations.
    """

    def body(carry, _):
        sim, ap = carry
        # Sense the wind the previous step applied. Re-evaluating the model here
        # would split the key a second time and hand the controller a different
        # realisation from the one the aircraft is flying through.
        controls, ap = autopilot(
            sense(sim.state, sim.wind_ned), ap, targets, gains, ac, dt
        )
        sim = step(sim, controls, dt, ac, wind_model=wind_model)
        return (sim, ap), (sim.state, controls)

    return jax.lax.scan(body, (sim, ap), None, length=n_steps)


# Hand-tuned against the trimmed 747 at 40,000 ft, M 0.80. A light aircraft
# needs its own set -- these will not fly one.
BOEING747_GAINS = Gains(
    alt_p=jnp.array(0.0012),
    alt_i=jnp.array(2.0e-5),
    theta_limit=jnp.array(0.15),
    theta_p=jnp.array(2.2),
    theta_i=jnp.array(0.45),
    q_d=jnp.array(3.0),
    hdg_p=jnp.array(1.1),
    phi_limit=jnp.array(0.44),
    phi_p=jnp.array(1.0),
    phi_i=jnp.array(0.08),
    p_d=jnp.array(0.7),
    beta_p=jnp.array(1.5),
    spd_p=jnp.array(0.05),
    spd_i=jnp.array(0.01),
    surface_rate=jnp.array(0.6),
    throttle_rate=jnp.array(0.2),
)

# Hand-tuned against the trimmed Cherokee at 4,920 ft, 50 m/s. The pitch and
# speed loops are scaled from the 747's by control authority -- this aircraft
# has 39x the pitch acceleration per radian of elevator -- so those gains come
# down by roughly that factor.
#
# The roll loop is NOT scaled that way. It is sized from the roll dynamics
# themselves (wn ~ 2 rad/s against Lda = 3.1, Lp = -2.78), because the Cherokee's
# spiral is mildly unstable (T2 = 52 s) and an authority-scaled loop is too weak
# to hold it: during tuning it diverged to 48 deg of bank on a 30 deg heading
# change. Authority scaling sets how hard a surface pushes; it says nothing about
# what the airframe does when the loop lets go.
CHEROKEE_GAINS = Gains(
    alt_p=jnp.array(0.0025),
    alt_i=jnp.array(2.5e-5),
    theta_limit=jnp.array(0.20),
    theta_p=jnp.array(0.057),
    theta_i=jnp.array(0.012),
    q_d=jnp.array(0.077),
    hdg_p=jnp.array(0.8),
    phi_limit=jnp.array(0.44),
    phi_p=jnp.array(1.0),
    phi_i=jnp.array(0.05),
    p_d=jnp.array(0.15),
    beta_p=jnp.array(1.0),
    spd_p=jnp.array(0.026),
    spd_i=jnp.array(0.005),
    surface_rate=jnp.array(1.0),
    throttle_rate=jnp.array(0.5),
)

# Hand-tuned against the trimmed Cessna at 5,000 ft, 60 m/s.
#
# beta_p is ZERO because this aircraft has no rudder: the source gives neither
# Cldr nor Cndr, and the one rudder derivative it does give is inconsistent, so
# aircraft.py zeroes the whole set. A non-zero sideslip gain would deflect a
# surface that produces no moment, which reads as a control system doing
# something when it is doing nothing. Turns are therefore uncoordinated:
# sideslip peaks near 1.9 deg in a 30 deg heading change, against 0.6 deg for the
# Cherokee, which has a working rudder. Restore this gain if Cndr is sourced.
CESSNA172_GAINS = Gains(
    alt_p=jnp.array(0.0015),
    alt_i=jnp.array(2.5e-5),
    theta_limit=jnp.array(0.20),
    theta_p=jnp.array(0.087),
    theta_i=jnp.array(0.018),
    q_d=jnp.array(0.119),
    hdg_p=jnp.array(0.8),
    phi_limit=jnp.array(0.44),
    phi_p=jnp.array(0.15),
    phi_i=jnp.array(0.01),
    p_d=jnp.array(0.02),
    beta_p=jnp.array(0.0),
    spd_p=jnp.array(0.035),
    spd_i=jnp.array(0.007),
    surface_rate=jnp.array(1.0),
    throttle_rate=jnp.array(0.5),
)

# Power-approach 747. Started from the cruise set and re-scaled for dynamic
# pressure: 4.4 kPa at 165 KTAS sea level against 8.4 kPa at the cruise point, so
# every surface is about half as effective per degree. See the note in
# tests/test_aircraft.py about V_md -- this condition sits BELOW minimum-drag
# speed, which is what an approach is, so the speed loop is working against the
# back side of the drag curve and no gain set fixes that.
BOEING747_APPROACH_GAINS = BOEING747_GAINS._replace(
    theta_p=jnp.array(4.0),
    theta_i=jnp.array(0.80),
    q_d=jnp.array(5.0),
    phi_p=jnp.array(1.8),
    p_d=jnp.array(1.2),
    spd_p=jnp.array(0.10),
    spd_i=jnp.array(0.02),
)

GAINS: dict[str, Gains] = {
    "boeing747": BOEING747_GAINS,
    "boeing747_approach": BOEING747_APPROACH_GAINS,
    "cherokee": CHEROKEE_GAINS,
    "cessna172": CESSNA172_GAINS,
}
