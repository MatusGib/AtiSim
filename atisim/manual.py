"""Hand flying, and the manual/autopilot mode switch.

The pilot model is deliberately the same shape as the autopilot:

    manual(manual_state, pilot, gains, ac, dt) -> (Controls, ManualState)

so the mode switch is a choice between two functions with identical signatures
rather than two different control paths through the simulator.

**Stick convention.** The three surface axes behave like a spring-centred centre
stick: an axis held at full travel commands a fixed fraction of the deflection
limit, and releasing it returns the surface to `ManualState.reference` -- the
deflection the stick centres to. That reference is whatever the surfaces were
doing at the moment manual control was taken, so letting go leaves the aircraft
where the previous controller left it rather than snapping the surfaces to zero.

**Trim is what moves that reference.** Without it the reference goes stale after
any manoeuvre and the aircraft drifts, exactly as an untrimmed real stick would;
`pilot.trim` walks the elevator reference at `gains.trim_rate`, and `trim_here`
snaps it to the deflections currently reaching the plant. Trim writes to the
reference and never to `controls`, which is the whole distinction between a trim
system and a second elevator: let go, and the aircraft stays trimmed.

The throttle is *not* spring-centred, because a throttle lever is not. The key
integrates the setting at a fixed rate and it stays where it is left. Trim is
not sprung either, for the same reason -- a trim wheel that recentred itself
would be worse than no trim at all.

**Sign conventions** follow atisim.aero, so pushing the stick forward pitches
the nose down:

    pitch  +1 = stick forward = elevator trailing-edge down = nose down
    roll   +1 = stick right   = positive aileron            = right roll
    yaw    +1 = right pedal   = negative rudder             = nose right

**Bumpless transfer** is the reason the mode machine lives here rather than in
the animation loop. Both directions are handled in `toggle`:

    manual -> autopilot   autopilot.engage seeds the integrators from the
                          deflections the pilot was holding
    autopilot -> manual   the stick reference and the current deflections are
                          both seeded from the autopilot's last output

Nothing in this module goes inside `lax.scan`: `Mode` is a Python enum and the
dispatch in `update` is a Python branch. The live loop is plain Python at 50 Hz
(see atisim.panel), so that is where this runs. The jitted, scannable path is
`autopilot.closed_loop_rollout`, which is unaffected.
"""

from enum import IntEnum
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from atisim.aircraft import Aircraft
from atisim.autopilot import APState, Gains, Targets, autopilot, engage
# Shared with the autopilot on purpose: the rate limit is a property of the
# actuator, so both controllers must apply the same one.
from atisim.autopilot import _rate_limit
from atisim.sensors import AirData
from atisim.state import Controls
from atisim.units import DEG2RAD


class PilotInput(NamedTuple):
    """Stick and throttle demand, each in [-1, 1]. Zero is centred/no change."""

    pitch: Array = 0.0
    roll: Array = 0.0
    yaw: Array = 0.0
    throttle: Array = 0.0
    trim: Array = 0.0  # +1 is nose-up: moves the CENTRING POINT, not the surface


NEUTRAL = PilotInput()


class ManualGains(NamedTuple):
    # Fraction of the deflection limit that a fully deflected stick commands.
    # Hand-chosen per aircraft for keyboard flyability, like the autopilot gains:
    # full travel on a keyboard is instantaneous, so full authority is unflyable.
    elevator_authority: Array
    aileron_authority: Array
    rudder_authority: Array
    surface_rate: Array  # rad/s
    throttle_rate: Array  # per s
    trim_rate: Array  # rad/s of elevator REFERENCE per unit trim input


class ManualState(NamedTuple):
    controls: Controls  # current deflections, for rate limiting and handover
    reference: Controls  # what a released stick returns to


@jax.jit
def manual(
    ms: ManualState,
    pilot: PilotInput,
    gains: ManualGains,
    ac: Aircraft,
    dt: Array,
) -> tuple[Controls, ManualState]:
    """One step of hand flying. Deflection limits then rate limits, as the plant sees them."""

    def surface(reference, demand, authority, limit):
        return jnp.clip(reference + demand * authority * limit, -limit, limit)

    # Trim moves the point a released stick returns to, which is what makes it a
    # trim system rather than a second elevator: let go, and the aircraft stays
    # where it was trimmed. Nose-up is trailing-edge up, i.e. a NEGATIVE
    # deflection here -- the same convention as the stick, where +pitch is
    # forward and pitches the nose down.
    reference = ms.reference._replace(
        elevator=jnp.clip(
            ms.reference.elevator - pilot.trim * gains.trim_rate * dt,
            -ac.elevator_limit,
            ac.elevator_limit,
        )
    )

    elevator = surface(
        reference.elevator, pilot.pitch, gains.elevator_authority, ac.elevator_limit
    )
    aileron = surface(
        reference.aileron, pilot.roll, gains.aileron_authority, ac.aileron_limit
    )
    rudder = surface(
        reference.rudder, -pilot.yaw, gains.rudder_authority, ac.rudder_limit
    )

    out = Controls(
        elevator=_rate_limit(elevator, ms.controls.elevator, gains.surface_rate, dt),
        aileron=_rate_limit(aileron, ms.controls.aileron, gains.surface_rate, dt),
        rudder=_rate_limit(rudder, ms.controls.rudder, gains.surface_rate, dt),
        # The throttle key commands a rate directly, so integrating it at
        # throttle_rate *is* the rate limit; a second one would halve it.
        throttle=jnp.clip(
            ms.controls.throttle + pilot.throttle * gains.throttle_rate * dt, 0.0, 1.0
        ),
    )
    return out, ManualState(controls=out, reference=reference)


def take_control(controls: Controls) -> ManualState:
    """Hand the given deflections to the pilot: hold them, and centre to them."""
    return ManualState(controls=controls, reference=controls)


# ---------------------------------------------------------------------------
# Mode switching
# ---------------------------------------------------------------------------


class Mode(IntEnum):
    MANUAL = 0
    AUTOPILOT = 1


class Controller(NamedTuple):
    """Which controller is flying, plus the state of both of them.

    The inactive controller's state is stale while the other one flies. That is
    fine and is the whole point: `toggle` reseeds it from the live deflections
    at the moment of the switch, which is what makes the transfer bumpless.
    """

    mode: Mode
    manual: ManualState
    ap: APState


def start(
    air: AirData,
    controls: Controls,
    targets: Targets,
    gains: Gains,
    ac: Aircraft,
    mode: Mode = Mode.MANUAL,
) -> Controller:
    """Both controllers seeded from the same starting deflections, usually trim."""
    return Controller(
        mode=mode,
        manual=take_control(controls),
        ap=engage(air, controls, targets, gains, ac),
    )


def current_controls(ctl: Controller) -> Controls:
    """The deflections currently reaching the plant, whoever is flying."""
    return ctl.manual.controls if ctl.mode is Mode.MANUAL else ctl.ap.controls


def toggle(
    ctl: Controller, air: AirData, targets: Targets, gains: Gains, ac: Aircraft
) -> Controller:
    """Switch mode, seeding the incoming controller from the outgoing deflections."""
    controls = current_controls(ctl)
    if ctl.mode is Mode.MANUAL:
        return Controller(
            mode=Mode.AUTOPILOT,
            manual=ctl.manual,
            ap=engage(air, controls, targets, gains, ac),
        )
    return Controller(mode=Mode.MANUAL, manual=take_control(controls), ap=ctl.ap)


def trim_here(ctl: Controller) -> Controller:
    """Snap the stick's centring point to the deflections now reaching the plant.

    Not a control any real aircraft has, but it is the reference concept made
    explicit: "the aircraft is doing what I want -- hold this". Without it,
    trimming out a manoeuvre on a keyboard means holding two keys and watching a
    number, which is a worse experience than the trim wheel it stands in for.

    A no-op in AUTOPILOT, because `toggle` already reseeds the reference from the
    live deflections on the way out, so there is nothing left for it to do.
    """
    if ctl.mode is not Mode.MANUAL:
        return ctl
    return ctl._replace(manual=take_control(ctl.manual.controls))


def update(
    ctl: Controller,
    air: AirData,
    pilot: PilotInput,
    targets: Targets,
    gains: Gains,
    mgains: ManualGains,
    ac: Aircraft,
    dt: Array,
) -> tuple[Controls, Controller]:
    """One control step from whichever controller currently has the aircraft.

    Takes `AirData` rather than `State` for the reason given in
    `autopilot.autopilot`: the manual branch ignores it, but the autopilot branch
    must not be handed inertial velocity.
    """
    if ctl.mode is Mode.MANUAL:
        controls, ms = manual(ctl.manual, pilot, mgains, ac, dt)
        return controls, ctl._replace(manual=ms)

    controls, ap = autopilot(air, ctl.ap, targets, gains, ac, dt)
    return controls, ctl._replace(ap=ap)


# Hand-tuned against the trimmed 747 at 40,000 ft, M 0.80, on a keyboard. The
# rates match BOEING747_GAINS: same actuators, same limits.
#
# The three authorities are far apart because the aircraft is: at this condition
# 6.25 deg of elevator pitches it 19 deg in three seconds, while 10 deg of
# aileron rolls it at only 3 deg/s (Clda 0.0136 against Clp -0.350). So the
# elevator is geared well down and the aileron is given its full travel.
#
# The rudder is geared lowest of all. Full pedal is 3.7 deg, worth about 3 deg
# of sideslip -- a coordination input. Holding it there for eight seconds still
# rolls the aircraft past 60 deg of bank through dihedral effect; that is the
# aeroplane, not a missing limiter, and a real 747 behaves the same way.
# `trim_rate` is DECLARED, and derived from one rule so that the three numbers
# are not three separate guesses: one second of held trim moves the reference by
# about a quarter of what full stick commands. Finer than that and trim is
# useless; coarser and it is unflyable. Every deflection limit here is 25 deg.
#
#   747       full stick 0.25 * 25 = 6.25 deg  ->  1.50 deg/s
#   cherokee  full stick 0.06 * 25 = 1.50 deg  ->  0.37 deg/s
#   cessna    full stick 0.08 * 25 = 2.00 deg  ->  0.50 deg/s
BOEING747_MANUAL = ManualGains(
    elevator_authority=jnp.array(0.25),
    aileron_authority=jnp.array(1.00),
    rudder_authority=jnp.array(0.15),
    surface_rate=jnp.array(0.6),
    throttle_rate=jnp.array(0.2),
    trim_rate=jnp.array(1.5 * DEG2RAD),
)

# Hand-tuned against the trimmed Cherokee on a keyboard. The elevator is geared
# right down: this aircraft has 39x the 747's pitch acceleration per radian, so
# full stick is 1.5 deg of elevator and still pitches it 14 deg in two seconds.
CHEROKEE_MANUAL = ManualGains(
    elevator_authority=jnp.array(0.06),
    aileron_authority=jnp.array(0.60),
    rudder_authority=jnp.array(0.30),
    surface_rate=jnp.array(1.0),
    throttle_rate=jnp.array(0.5),
    trim_rate=jnp.array(0.37 * DEG2RAD),
)

# Hand-tuned against the trimmed Cessna on a keyboard.
#
# rudder_authority is ZERO: aircraft.py zeroes this aircraft's rudder because
# the source does not provide it, so the pedals would move a surface that
# produces no force and no moment. The aileron gearing is high to compensate --
# with no rudder a roll input builds sideslip, and the strong dihedral effect
# (Clb -0.21) against weak directional stability (Cnb 0.0126) turns that
# sideslip straight back into an opposing roll moment, so bank builds a few
# degrees and stalls there instead of continuing to roll.
CESSNA172_MANUAL = ManualGains(
    elevator_authority=jnp.array(0.08),
    aileron_authority=jnp.array(0.40),
    rudder_authority=jnp.array(0.0),
    surface_rate=jnp.array(1.0),
    throttle_rate=jnp.array(0.5),
    trim_rate=jnp.array(0.5 * DEG2RAD),
)

# Power-approach 747: the cruise gearing at half the dynamic pressure, so the
# authorities are roughly doubled to give the same response per key press.
# UNTUNED -- nobody has flown it, and section 9 records that hand-tuning needs a
# human at the keyboard.
BOEING747_APPROACH_MANUAL = BOEING747_MANUAL._replace(
    elevator_authority=jnp.array(0.50),
    aileron_authority=jnp.array(1.60),
    rudder_authority=jnp.array(0.30),
)

# JSBSim's 737. UNTUNED -- nobody has flown it, and section 9 records that hand
# tuning needs a human at the keyboard. These are derived by the same authority
# rule the Cherokee's comment sets out, so that they are one decision rather
# than three guesses:
#
#   elevator  1.955x the 747's pitch acceleration per radian, so 0.25 / 1.955
#   aileron   26.3x the roll acceleration per radian, so 1.00 / 26.3
#   rudder    kept at the 747's 0.15; JSBSim's 737 has Cndr -0.20 against the
#             747's set, and no CYdr at all, so pedal yaws without any direct
#             side force. Worth a human's judgement rather than a scale factor.
#
# trim_rate follows the same one-second-moves-a-quarter-of-full-stick rule as
# the others. Full stick here is 0.13 * 17.2 deg (this aircraft's elevator limit
# is 0.3 rad, not the 25 deg the other three share) = 2.23 deg -> 0.56 deg/s.
BOEING737_MANUAL = BOEING747_MANUAL._replace(
    elevator_authority=jnp.array(0.13),
    aileron_authority=jnp.array(0.04),
    rudder_authority=jnp.array(0.15),
    trim_rate=jnp.array(0.56 * DEG2RAD),
)

MANUAL_GAINS: dict[str, ManualGains] = {
    "boeing747": BOEING747_MANUAL,
    "boeing737": BOEING737_MANUAL,
    "boeing737_approach": BOEING737_MANUAL,
    "boeing747_approach": BOEING747_APPROACH_MANUAL,
    # Same reasoning as GAINS: carried over from the CR-2144 747, untuned, and
    # unused by the fixed-control comparison this entry exists for.
    "boeing747_jsbsim": BOEING747_MANUAL,
    "cherokee": CHEROKEE_MANUAL,
    "cessna172": CESSNA172_MANUAL,
}
