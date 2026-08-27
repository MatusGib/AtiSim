import jax
import jax.numpy as jnp
import numpy as np
import pytest

from atisim import autopilot as ap_mod
from atisim import earth, integrate, manual as man, trim
from atisim.aircraft import CRUISE, REGISTRY
from atisim.manual import Mode, PilotInput
from atisim.sensors import sense
from atisim.state import Controls
from atisim.state import altitude as geodetic_altitude
from atisim.state import quat_to_euler_ned

AC = REGISTRY["boeing747"]
GAINS = ap_mod.GAINS["boeing747"]
MGAINS = man.MANUAL_GAINS["boeing747"]
V = CRUISE["boeing747"]["airspeed"]
H = CRUISE["boeing747"]["altitude"]
DT = 0.02
# 47N at the cruise altitude these tests hand-fly. WGS84_J2: hand flying is the
# aircraft's actual flight behaviour, so it flies the shipped Earth.
ANCHOR = earth.anchor_at(np.radians(47.0), 0.0, H)
EARTH = earth.WGS84_J2


def trimmed():
    x, _ = trim.trim(jnp.array(V), jnp.array(H), AC, ANCHOR, EARTH)
    return (
        trim.trimmed_state(x[0], x[3], jnp.array(V), jnp.array(H), ANCHOR, 0.0),
        trim.trimmed_controls(x[1], x[2]),
    )


def hold_targets():
    return ap_mod.Targets(
        altitude=jnp.array(H), heading=jnp.array(0.0), airspeed=jnp.array(V)
    )


def hold(ms, pilot, seconds):
    """Run the manual controller with a fixed stick position, controls only."""
    for _ in range(int(seconds / DT)):
        controls, ms = man.manual(ms, pilot, MGAINS, AC, jnp.array(DT))
    return controls, ms


def hand_fly(state, controls, pilot, seconds):
    """Fly the plant under a fixed stick position. Returns the state history."""
    ms = man.take_control(controls)
    sim = integrate.init_sim(state, jax.random.PRNGKey(0))
    history = []
    for _ in range(int(seconds / DT)):
        controls, ms = man.manual(ms, pilot, MGAINS, AC, jnp.array(DT))
        sim = integrate.step(sim, controls, jnp.array(DT), AC, ANCHOR, EARTH)
        history.append(sim.state)
    return jax.tree.map(lambda *xs: jnp.stack(xs), *history)


# --- the stick ---


def test_full_stick_commands_the_configured_authority():
    _, controls = trimmed()
    ms = man.take_control(controls)
    out, _ = hold(ms, PilotInput(pitch=1.0), 5.0)
    expected = float(controls.elevator) + float(
        MGAINS.elevator_authority * AC.elevator_limit
    )
    assert float(out.elevator) == pytest.approx(expected, abs=1e-9)


def test_releasing_the_stick_returns_the_surfaces_to_the_reference():
    _, controls = trimmed()
    ms = man.take_control(controls)
    _, ms = hold(ms, PilotInput(pitch=1.0, roll=-1.0, yaw=1.0), 5.0)
    out, _ = hold(ms, man.NEUTRAL, 5.0)
    for surface in ("elevator", "aileron", "rudder"):
        assert float(getattr(out, surface)) == pytest.approx(
            float(getattr(controls, surface)), abs=1e-9
        )


def test_the_stick_centres_to_the_reference_not_to_zero():
    """Releasing must not snap a trimmed surface back to neutral."""
    reference = Controls(
        elevator=jnp.array(-0.05),
        aileron=jnp.array(0.02),
        rudder=jnp.array(0.0),
        throttle=jnp.array(0.4),
    )
    out, _ = hold(man.take_control(reference), man.NEUTRAL, 5.0)
    assert float(out.elevator) == pytest.approx(-0.05, abs=1e-12)
    assert float(out.aileron) == pytest.approx(0.02, abs=1e-12)


def test_stick_axes_carry_the_documented_signs():
    _, controls = trimmed()
    out, _ = hold(man.take_control(controls), PilotInput(pitch=1.0), 5.0)
    assert float(out.elevator) > float(controls.elevator)  # forward = nose down
    out, _ = hold(man.take_control(controls), PilotInput(roll=1.0), 5.0)
    assert float(out.aileron) > float(controls.aileron)  # right = right roll
    out, _ = hold(man.take_control(controls), PilotInput(yaw=1.0), 5.0)
    assert float(out.rudder) < float(controls.rudder)  # right pedal = negative rudder


# --- limits ---


def test_surfaces_respect_deflection_limits():
    """A reference already near the stop plus full stick must still clip."""
    reference = Controls(
        elevator=jnp.array(0.9) * AC.elevator_limit,
        aileron=jnp.array(0.9) * AC.aileron_limit,
        rudder=jnp.array(0.9) * AC.rudder_limit,
        throttle=jnp.array(0.5),
    )
    out, _ = hold(man.take_control(reference), PilotInput(pitch=1.0, roll=1.0), 10.0)
    assert float(out.elevator) <= float(AC.elevator_limit) + 1e-12
    assert float(out.aileron) <= float(AC.aileron_limit) + 1e-12


def test_surface_rate_limits_are_respected():
    _, controls = trimmed()
    ms = man.take_control(controls)
    # Slam the stick from one stop to the other every 10 steps.
    previous, rates = controls, []
    for i in range(200):
        pilot = PilotInput(pitch=1.0 if (i // 10) % 2 else -1.0, roll=-1.0, yaw=1.0)
        out, ms = man.manual(ms, pilot, MGAINS, AC, jnp.array(DT))
        for surface in ("elevator", "aileron", "rudder"):
            rates.append(
                abs(float(getattr(out, surface)) - float(getattr(previous, surface)))
                / DT
            )
        previous = out
    assert max(rates) <= float(MGAINS.surface_rate) + 1e-9


def test_throttle_is_incremental_rate_limited_and_clamped():
    ms = man.take_control(
        Controls(
            elevator=jnp.array(0.0),
            aileron=jnp.array(0.0),
            rudder=jnp.array(0.0),
            throttle=jnp.array(0.5),
        )
    )
    out, ms = man.manual(ms, PilotInput(throttle=1.0), MGAINS, AC, jnp.array(DT))
    assert float(out.throttle) == pytest.approx(
        0.5 + float(MGAINS.throttle_rate) * DT, abs=1e-12
    )
    # A lever stays where it is left.
    held, ms = hold(ms, man.NEUTRAL, 5.0)
    assert float(held.throttle) == pytest.approx(float(out.throttle), abs=1e-12)
    # And it saturates rather than running past the stops.
    up, ms = hold(ms, PilotInput(throttle=1.0), 60.0)
    assert float(up.throttle) == pytest.approx(1.0, abs=1e-12)
    down, _ = hold(ms, PilotInput(throttle=-1.0), 60.0)
    assert float(down.throttle) == pytest.approx(0.0, abs=1e-12)


# --- hand flying the actual plant ---


def test_neutral_stick_from_trim_holds_the_trimmed_condition():
    """Neutral stick must not move the surfaces. The aircraft still drifts 2.8 m.

    THE 1.0 m BOUND WAS A FLAT-EARTH NUMBER AND IT IS GONE. `trimmed_state` sets
    `omega = 0`, which is an instantaneous equilibrium -- the trim residual
    really does vanish at t = 0 -- but it is NOT a steady level-flight condition
    over a curved Earth, because level flight round an ellipsoid needs a
    continuous nose-down transport rate of V/R and this state has none. The
    aircraft therefore flies a straighter line than the surface curves and gains
    geodetic altitude: measured 2.8412 m over these 30 s (7.08 km of track).

    IT IS CURVATURE, NOT ROTATION, and the control says so: `earth.FLAT` gives
    2.8412 m against WGS84_J2's 2.8412 m -- identical to five decimals, because
    FLAT is non-rotating and constant-g but still an ellipsoid. So neither the
    Coriolis term nor J2 is involved, and this is not something the manual
    controller could fix.

    Bounded at 4.0 m, which leaves 1.16 m of headroom over the known drift --
    comparable to the 1.0 m the flat-Earth version had over zero -- so a
    neutral stick that actually moved a surface is still caught.
    """
    state, controls = trimmed()
    hist = hand_fly(state, controls, man.NEUTRAL, 30.0)
    # GEODETIC, not -pos_ned[:, 2]: the tangent plane the anchor defines falls
    # away from the ellipsoid as d^2/2R, which is 3.9 m over the 7 km this run
    # covers -- and is a different quantity from the altitude being held.
    altitude = np.asarray(jax.vmap(geodetic_altitude, in_axes=(0, None))(hist, ANCHOR))
    assert np.abs(altitude - H).max() < 4.0


def test_forward_stick_pitches_the_nose_down_without_departing():
    """Sanity check on the chosen authority: a response, not a departure."""
    state, controls = trimmed()
    hist = hand_fly(state, controls, PilotInput(pitch=1.0), 10.0)
    theta = np.asarray(jax.vmap(quat_to_euler_ned, in_axes=(0, None))(hist, ANCHOR))[:, 1]
    assert theta[-1] < theta[0] - np.deg2rad(2.0)
    assert np.abs(theta).max() < np.deg2rad(45.0)
    assert np.isfinite(np.asarray(hist.vel_body)).all()


def test_right_stick_rolls_right_without_departing():
    state, controls = trimmed()
    hist = hand_fly(state, controls, PilotInput(roll=1.0), 6.0)
    phi = np.asarray(jax.vmap(quat_to_euler_ned, in_axes=(0, None))(hist, ANCHOR))[:, 0]
    assert phi[-1] > np.deg2rad(5.0)
    assert np.abs(phi).max() < np.deg2rad(80.0)


# --- switching ---


def test_start_seeds_both_controllers_from_the_same_deflections():
    state, controls = trimmed()
    ctl = man.start(sense(state, ANCHOR), controls, hold_targets(), GAINS, AC)
    assert ctl.mode is Mode.MANUAL
    for held, current in zip(man.current_controls(ctl), controls):
        assert float(held) == float(current)


def test_toggle_into_autopilot_is_bumpless_from_a_hand_flown_deflection():
    """The classic lurch: engage while holding the stick off-centre."""
    state, controls = trimmed()
    ctl = man.start(sense(state, ANCHOR), controls, hold_targets(), GAINS, AC)
    targets = hold_targets()

    flown, ctl = man.update(
        ctl, sense(state, ANCHOR), PilotInput(pitch=1.0), targets, GAINS, MGAINS, AC, jnp.array(DT)
    )
    ctl = man.toggle(ctl, sense(state, ANCHOR), targets, GAINS, AC)
    assert ctl.mode is Mode.AUTOPILOT

    first, _ = man.update(
        ctl, sense(state, ANCHOR), man.NEUTRAL, targets, GAINS, MGAINS, AC, jnp.array(DT)
    )
    for engaged, hand_flown in zip(first, flown):
        assert float(engaged) == pytest.approx(float(hand_flown), abs=1e-12)


def test_toggle_out_of_autopilot_hands_back_the_live_deflections():
    state, controls = trimmed()
    targets = ap_mod.Targets(
        altitude=jnp.array(H + 300.0), heading=jnp.array(0.0), airspeed=jnp.array(V)
    )
    ctl = man.start(sense(state, ANCHOR), controls, targets, GAINS, AC, mode=Mode.AUTOPILOT)

    # Let the autopilot pull the surfaces away from trim chasing the step.
    sim = integrate.init_sim(state, jax.random.PRNGKey(0))
    for _ in range(500):
        flown, ctl = man.update(
            ctl, sense(sim.state, ANCHOR), man.NEUTRAL, targets, GAINS, MGAINS, AC, jnp.array(DT)
        )
        sim = integrate.step(sim, flown, jnp.array(DT), AC, ANCHOR, EARTH)
    assert abs(float(flown.elevator) - float(controls.elevator)) > 1e-4

    ctl = man.toggle(ctl, sense(sim.state, ANCHOR), targets, GAINS, AC)
    assert ctl.mode is Mode.MANUAL
    for handback, live in zip(man.current_controls(ctl), flown):
        assert float(handback) == float(live)

    # And with the stick centred the first manual step must not move anything.
    first, _ = man.update(
        ctl, sense(sim.state, ANCHOR), man.NEUTRAL, targets, GAINS, MGAINS, AC, jnp.array(DT)
    )
    for after, before in zip(first, flown):
        assert float(after) == pytest.approx(float(before), abs=1e-12)


def test_round_trip_through_both_modes_leaves_no_discontinuity():
    """Toggle repeatedly while hand flying; no switch may step the surfaces."""
    state, controls = trimmed()
    targets = hold_targets()
    ctl = man.start(sense(state, ANCHOR), controls, targets, GAINS, AC)
    sim = integrate.init_sim(state, jax.random.PRNGKey(0))

    previous = controls
    worst = 0.0
    for i in range(1500):
        if i % 300 == 299:
            ctl = man.toggle(ctl, sense(sim.state, ANCHOR), targets, GAINS, AC)
        pilot = PilotInput(pitch=0.3) if ctl.mode is Mode.MANUAL else man.NEUTRAL
        flown, ctl = man.update(
            ctl, sense(sim.state, ANCHOR), pilot, targets, GAINS, MGAINS, AC, jnp.array(DT)
        )
        worst = max(worst, abs(float(flown.elevator) - float(previous.elevator)) / DT)
        previous = flown
        sim = integrate.step(sim, flown, jnp.array(DT), AC, ANCHOR, EARTH)

    assert worst <= float(GAINS.surface_rate) + 1e-9
    assert np.isfinite(np.asarray(sim.state.vel_body)).all()


# --- trim -------------------------------------------------------------------


def test_trim_moves_the_reference_and_not_the_current_deflection():
    """Trim writes to the CENTRING POINT. That is the whole distinction between
    a trim system and a second elevator: let go, and the aircraft stays put."""
    _, controls = trimmed()
    ms = man.take_control(controls)
    before = float(ms.reference.elevator)

    _, ms = hold(ms, PilotInput(trim=1.0), 1.0)  # one second of nose-up trim
    assert float(ms.reference.elevator) < before  # nose-up is trailing edge UP

    # And a released stick now settles on the NEW reference, not the old one.
    released, ms = hold(ms, man.NEUTRAL, 5.0)
    assert float(released.elevator) == pytest.approx(
        float(ms.reference.elevator), abs=1e-9
    )
    assert float(released.elevator) != pytest.approx(before, abs=1e-6)


def test_one_second_of_trim_is_about_a_quarter_of_full_stick():
    """The rule the three trim_rate values were derived from, asserted rather
    than left in a comment -- otherwise the next person picks a fourth number."""
    _, controls = trimmed()
    for name in ("boeing747", "cherokee", "cessna172"):
        ac = REGISTRY[name]
        gains = man.MANUAL_GAINS[name]
        ms = man.take_control(controls)
        for _ in range(int(1.0 / DT)):
            _, ms = man.manual(ms, PilotInput(trim=1.0), gains, ac, jnp.array(DT))
        moved = abs(float(ms.reference.elevator) - float(controls.elevator))
        full_stick = float(gains.elevator_authority) * float(ac.elevator_limit)
        assert moved == pytest.approx(0.25 * full_stick, rel=0.05)


def test_trim_respects_the_elevator_limit():
    _, controls = trimmed()
    ms = man.take_control(controls)
    _, ms = hold(ms, PilotInput(trim=1.0), 600.0)  # far longer than full travel
    assert abs(float(ms.reference.elevator)) <= float(AC.elevator_limit) + 1e-12


def test_neutral_trim_leaves_the_reference_untouched():
    """Every existing manual test passes PilotInput without a trim field, so a
    sign or scale error here would move the reference under all of them."""
    _, controls = trimmed()
    ms = man.take_control(controls)
    _, ms = hold(ms, man.NEUTRAL, 10.0)
    assert float(ms.reference.elevator) == pytest.approx(
        float(controls.elevator), abs=1e-15
    )


def test_trim_here_snaps_the_reference_to_the_live_deflections():
    state, controls = trimmed()
    ctl = man.start(sense(state, ANCHOR), controls, hold_targets(), GAINS, AC)
    held, ctl = man.update(
        ctl, sense(state, ANCHOR), PilotInput(pitch=1.0), hold_targets(),
        GAINS, MGAINS, AC, jnp.array(DT),
    )
    assert float(ctl.manual.reference.elevator) != pytest.approx(float(held.elevator))

    ctl = man.trim_here(ctl)
    assert float(ctl.manual.reference.elevator) == pytest.approx(float(held.elevator))


def test_trim_here_does_nothing_under_the_autopilot():
    """`toggle` already reseeds the reference on the way out, so there is nothing
    left to do -- and trim-here must not reach into the autopilot's state."""
    state, controls = trimmed()
    ctl = man.start(
        sense(state, ANCHOR), controls, hold_targets(), GAINS, AC, mode=Mode.AUTOPILOT
    )
    assert man.trim_here(ctl) is ctl
