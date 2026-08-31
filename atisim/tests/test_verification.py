"""Tier-0 verification: checks that depend on no aircraft data at all.

PROJECT.md section 4 is almost entirely validation -- a measured quantity against
a published one for one aircraft. These are the other kind: if one of them fails,
the arithmetic is wrong and no source can say otherwise.

Note that conftest.py's jax_debug_nans guards JAX only. The torque-free reference
solution goes through SciPy and asserts its own domain instead.
"""

import hashlib
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import atisim  # noqa: F401  -- enables x64 before any array is made
from atisim import dynamics, earth, integrate, trim, verification
from atisim.aircraft import CRUISE, REGISTRY

# 47N is the latitude the rest of this project's Earth-rotation work uses, and
# the anchor sits at the 747's cruise altitude -- the condition every rollout in
# this file that carries an aircraft is trimmed for. WGS84_J2 is the Earth the
# project flies, and it is what the checks below use UNLESS their reference is a
# non-rotating closed form; the two that are say so at their own site.
ANCHOR = earth.anchor_at(np.radians(47.0), 0.0, CRUISE["boeing747"]["altitude"])
EARTH = earth.WGS84_J2

# Captured from the integrator BEFORE rk4_step was extracted from `step`. This is
# the whole guard on that refactor: a test that re-derives the stage weights
# inline would compare the new code against itself and pass on an extraction that
# changed the arithmetic.
#
# **IT CANNOT BE REPRODUCED ANY MORE, AND IT IS KEPT RATHER THAN REPLACED.** See
# `test_extracting_rk4_step_did_not_move_a_single_bit` for the measurement and
# for why re-capturing it here would destroy the guard rather than repair it.
PRE_REFACTOR_VEL_HASH = "bbc0323e77183d73bd03817a98a530d0d563b56b4962520705aaee589f276aa4"


def _fixed_control_rollout(dt, n_steps, d_elevator=0.02):
    """747 at cruise trim with the elevator off trim, so something happens."""
    ac = REGISTRY["boeing747"]
    V, H = CRUISE["boeing747"]["airspeed"], CRUISE["boeing747"]["altitude"]
    x, _ = trim.trim(jnp.array(V), jnp.array(H), ac, ANCHOR, EARTH)
    # `x[3]` is the trimmed BANK, non-zero on a rotating Earth. Dropping it --
    # which `x[0], x[1], x[2]` unpacking does in silence -- would start the run
    # out of equilibrium in exactly the channel Coriolis acts in.
    state = trim.trimmed_state(
        x[0], x[3], jnp.array(V), jnp.array(H), ANCHOR, jnp.array(0.0)
    )
    controls = trim.trimmed_controls(x)._replace(elevator=x[1] + d_elevator)
    sim = integrate.init_sim(state, jax.random.PRNGKey(0))
    return integrate.rollout(sim, controls, jnp.array(dt), ac, n_steps, ANCHOR, EARTH)


def test_extracting_rk4_step_did_not_move_a_single_bit():
    """Bit-identity against a hash taken before the refactor.

    The same instrument PROJECT.md section 4 uses for the zero-wind path. A
    tolerance would not do: the claim is that the extraction was arithmetic
    neutral, and any tolerance admits an extraction that was not.

    **THE PREMISE IS GONE, THE HASH IS LEFT EXACTLY AS IT WAS, AND THIS TEST NO
    LONGER ASSERTS AGAINST IT.** It was red for one session while that was
    worked out. `PRE_REFACTOR_VEL_HASH` was captured from a plant that
    no longer exists: a flat, non-rotating Earth with a fixed `down`, a
    wings-level trim and `omega = 0`. The rotating Earth changes the plant by
    construction -- the trim is banked, `trimmed_state` carries a transport
    rate, gravity is J2 and Coriolis acts -- so no byte of `vel_body` can match.
    Measured on this rollout, 747 cruise, dt 0.02, 500 steps:

        pre-refactor (flat, non-rotating, unbanked)
            bbc0323e77183d73bd03817a98a530d0d563b56b4962520705aaee589f276aa4
        WGS84_J2
            74807d2b158234ea0b3b8f0975e9dd82ae9fd56593438c8b9bcc2df2daf40b36
        earth.FLAT
            7a6ff050e486bcc0919dcee9d49d1b0b3643959f332a11e02749659ac8254230

    `earth.FLAT` does not recover it either, and that is not a defect: earth.py
    says so in FLAT's own docstring -- an ECEF-accumulated state cannot be
    bit-identical to an NED-accumulated one even where the physics agrees.

    **RE-CAPTURING THE HASH WOULD NOT BE A MIGRATION, IT WOULD BE THE FAILURE
    THIS TEST EXISTS TO PREVENT.** The constant's own comment says the guard's
    whole value is that it predates the extraction; a hash taken now compares
    the current code against itself and would pass on an extraction that changed
    the arithmetic -- the same defect as re-deriving the stage weights inline.
    So the hash is NOT re-captured, and `PRE_REFACTOR_VEL_HASH` is kept above as
    the record of a plant that no longer exists.

    **WHAT REPLACES IT IS STRONGER, NOT WEAKER.** The claim was only ever "the
    extraction is arithmetically neutral". A frozen recording is one way to
    check that; an INDEPENDENT IMPLEMENTATION is another, and it does not rot
    when the plant legitimately changes. Below, `integrate.rk4_step` is compared
    BIT-FOR-BIT against a classical RK4 written out longhand in this test, on
    the real rotating-Earth plant. A hash could only ever say "the same as last
    time"; this says "the same as the method it claims to be", which is the
    property the extraction was supposed to preserve in the first place.
    """
    # THE CLAIM IS RE-ESTABLISHED WITHOUT A FROZEN HASH, which is the only
    # honest way left. `rk4_step` is compared against a classical RK4 written
    # out longhand HERE, on the real plant -- so it tests the extraction the
    # same way the hash did, but against an independent implementation rather
    # than against a recording of a plant that no longer exists.
    ac = REGISTRY["boeing747"]
    speed, height = CRUISE["boeing747"]["airspeed"], CRUISE["boeing747"]["altitude"]
    x, _ = trim.trim(jnp.array(speed), jnp.array(height), ac, ANCHOR, EARTH)
    state = trim.trimmed_state(
        x[0], x[3], jnp.array(speed), jnp.array(height), ANCHOR, jnp.array(0.0)
    )
    controls = trim.trimmed_controls(x)._replace(elevator=x[1] + 0.02)
    dt = 0.02

    def f(st):
        return dynamics.derivatives(
            st, controls, ac, jnp.zeros(3), jnp.zeros(3), ANCHOR, EARTH
        )

    def axpy(a, b, h):
        return jax.tree.map(lambda u, v: u + h * v, a, b)

    # Classical RK4, written out rather than reused.
    k1 = f(state)
    k2 = f(axpy(state, k1, dt / 2))
    k3 = f(axpy(state, k2, dt / 2))
    k4 = f(axpy(state, k3, dt))
    increment = jax.tree.map(
        lambda a, b, c, d: (a + 2.0 * b + 2.0 * c + d) / 6.0, k1, k2, k3, k4
    )
    longhand = axpy(state, increment, dt)
    extracted = integrate.rk4_step(f, state, dt)

    for field in ("pos_ecef", "vel_body", "quat", "omega"):
        mine = np.asarray(getattr(longhand, field))
        theirs = np.asarray(getattr(extracted, field))
        assert mine.tobytes() == theirs.tobytes(), (
            f"rk4_step differs from a longhand RK4 in {field}: {mine} vs {theirs}"
        )


def test_rk4_is_fourth_order_on_a_problem_with_a_closed_form():
    """The single biggest gap in the project's evidence.

    PROJECT.md section 4 asserts angular momentum barely drifts. An integrator
    can conserve beautifully and still be second-order: drift measures a
    symmetry, not an order. Nothing asserted the order, so a mis-weighted stage
    would pass every existing test at dt = 0.02 and quietly degrade every result
    taken at a larger step.

    Harmonic oscillator, xdot = [[0, 1], [-1, 0]] x, exact solution a rotation.
    Drives integrate.rk4_step DIRECTLY -- not a copy of it -- which is the entire
    point of extracting it.
    """
    dts = np.array([0.2, 0.1, 0.05, 0.025])
    errors, slope = verification.oscillator_refinement(dts)
    assert slope == pytest.approx(4.0, abs=0.05), f"observed order {slope}, {errors}"


def test_the_six_dof_rollout_is_fourth_order():
    """Same claim, through the real dynamics.

    The manufactured case isolates the stage weights. It cannot see a wind sample
    or control update applied at the wrong stage -- the seam turbulence will lean
    on (PROJECT.md section 2: "wind sampled once per step, held across the four
    stages").

    The reference is generated at the smallest step rather than analytically, so
    it carries its own error. At dt = 1/1024 that is around 1e-14, four orders
    below the smallest error being fitted.

    Flown on WGS84_J2, the Earth the project ships. An order of accuracy is a
    property of the SCHEME against whatever right-hand side it is given, and the
    right-hand side the project integrates is the rotating one; measured, the two
    Earths are indistinguishable here anyway -- 3.9872 on WGS84_J2 against
    3.9872 on earth.FLAT, with the four errors agreeing to three digits.

    THE FITTED WINDOW MUST STOP AT 1/32, and the reason is a measured property of
    this problem rather than a convenience. The aircraft cruises at 40,000 ft, so
    pos_ned is about [944, 0, -12184] and float64 resolves it to roughly 2.7e-12
    m. Round-off accumulates above that, and the discretisation error reaches it:
    measured pairwise orders over a wider sweep were

        1/4 -> 1/8    3.973
        1/8 -> 1/16   3.993
        1/16 -> 1/32  4.008
        1/32 -> 1/64  4.167
        1/64 -> 1/128 3.420
        1/128 -> 1/256  -0.685   <- refining now makes the answer WORSE

    so the error floor is about 7e-11 m at dt = 1/128. A window running through
    that floor fits partly to round-off and reads 3.82, which is an artefact of
    the measurement and not a defect in the integrator. The window here keeps the
    smallest fitted error 203x above the floor, and the asymptotic range is what
    an order-of-accuracy check is defined on.

    THAT FLOOR IS SUPERSEDED, NOT RE-MEASURED, and `fixed_control_refinement`'s
    own docstring says so: the state accumulates an ECEF offset now, not an NED
    position, so the ulp under it moves. The window is unchanged because the
    fitted result is unchanged -- 3.9872 measured here against the 3.98913
    PROJECT.md section 4 records -- but the 7e-11 m above is a pre-ECEF number
    and Task 14's re-measurement of ASSUMPTIONS.md F4 owns it.
    """
    ac = REGISTRY["boeing747"]
    V, H = CRUISE["boeing747"]["airspeed"], CRUISE["boeing747"]["altitude"]
    dts = np.array([1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32])
    errors, slope = verification.fixed_control_refinement(
        ac, V, H, dts, dt_ref=1.0 / 1024.0, anchor=ANCHOR, earth_model=EARTH
    )
    assert slope == pytest.approx(4.0, abs=0.05), f"observed order {slope}, errors {errors}"


# --- the ORDER half of the wind seam ---
#
# test_the_six_dof_rollout_is_fourth_order says in its own docstring that it
# "cannot see a wind sample or control update applied at the wrong RK4 stage",
# because it flies in STILL AIR. ASSUMPTIONS.md E4 says the same thing. Session
# 12 closed the BODY-FORCE half of that seam -- no spurious -m dW/dt term -- and
# left the ORDER half open. These two close it, and the answer is not 4.
#
# The two test fields are declared here rather than sourced. Their job is to be
# smooth and to produce a response large enough to sit above the round-off floor,
# not to represent any measured atmosphere.

_SMOOTH_WAVELENGTH = 1200.0  # m, DECLARED: short enough that 4 s covers 0.8 of it
_SMOOTH_AMPLITUDE = 25.0  # m/s, DECLARED: large enough to clear the 7e-11 m floor


def _smooth_wind_model():
    """A C-infinity wind field, through the real `field_model` path.

    Deliberately goes through `wind.field_model` rather than a hand-written
    closure, so `omega_gust` is derived by the same analytic gradient the real
    fields use. The per-step hold applies to both components and a closure that
    returned a zero gust rate would not exercise the seam being measured.
    """
    from atisim import wind

    wave = wind.LeeWave(
        w0=jnp.array(_SMOOTH_AMPLITUDE),
        wavelength=jnp.array(_SMOOTH_WAVELENGTH),
        north=jnp.array(0.0),
    )
    return wind.field_model(lambda p: wind.lee_wave_wind(p, wave), ANCHOR)


def test_the_rollout_is_only_first_order_through_a_spatially_varying_wind():
    """The wind is held across the four RK4 stages, and that costs three orders.

    `integrate.py` samples the wind once per step and holds it. That is correct
    for a stochastic field, for the reason its module docstring now gives --
    re-sampling a key-drawn process per stage makes the realisation a function of
    the step size -- and NOT, as this docstring used to say, because it is "the
    standard treatment for Dryden and von Karman". No source for that claim
    exists in this repository. For a field that varies in SPACE it is an O(h) perturbation
    of the right-hand side within the step, so the scheme is FIRST order however
    good the stage weights are.

    The control is the same function, same window, same aircraft, same elevator
    perturbation, with the wind model removed: that must still read 4. Without
    it this test would only show a number, not attribute it to the wind.

    FALSIFIED, per PROJECT.md section 3's rule that a check which can only pass
    shows nothing. With `integrate.step`'s `f` changed to re-sample the wind at
    each RK4 stage instead of holding it, this test goes RED at an observed order
    of 4.054 -- fourth order restored. So the 1.05 measured here is caused by the
    hold and by nothing else. Measured values, dts 1/4 .. 1/32, dt_ref 1/1024:

        still air        3.9891   (PROJECT.md section 4 records 3.98913)
        smooth field     1.0537   pairwise 1.073, 1.048, 1.042
        ...with the hold removed   4.0542

    ON THE ROTATING EARTH THE WINDY NUMBER DID NOT MOVE AT ALL: 1.0537,
    re-measured on WGS84_J2, the same four digits. The control moved in its last
    digit only, 3.9891 to 3.9872. That is what says the order is a property of
    the per-step HOLD and not of anything the Earth does -- the Earth changes the
    right-hand side, and the hold is an O(h) perturbation of whatever
    right-hand side it is given.

    The field is C-infinity, so nothing here is about the Rankine core edge --
    that is the next test, and it is a different mechanism. The core test passes
    with the probe still in, which is what says the two are separate.
    """
    ac = REGISTRY["boeing747"]
    V, H = CRUISE["boeing747"]["airspeed"], CRUISE["boeing747"]["altitude"]
    dts = np.array([1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32])

    _, still_air = verification.fixed_control_refinement(
        ac, V, H, dts, dt_ref=1.0 / 1024.0, anchor=ANCHOR, earth_model=EARTH
    )
    errors, windy = verification.fixed_control_refinement(
        ac, V, H, dts, dt_ref=1.0 / 1024.0, anchor=ANCHOR, earth_model=EARTH,
        wind_model=_smooth_wind_model(),
    )

    assert still_air == pytest.approx(4.0, abs=0.05), f"control moved: {still_air}"
    assert windy == pytest.approx(1.0, abs=0.1), f"observed order {windy}, errors {errors}"


def test_a_rankine_core_crossing_destroys_even_first_order_convergence():
    """Crossing the core edge makes the error non-monotonic in dt.

    `vortex_wind` switches branches at r = r0, where ASSUMPTIONS.md E2 records
    that the one-sided derivatives differ by 2*V0/r0 and have OPPOSITE SIGNS. The
    right-hand side is therefore C0 but not C1 there, and RK4 across a kink has
    an error that depends on where the step grid happens to land relative to the
    crossing. So refining dt does not monotonically improve the answer.

    Asserted as non-monotonicity rather than as an order, because there is no
    order to assert -- which is the whole point. A fitted slope through this
    sequence returns a number that describes nothing, and anything reporting one
    is reporting an artefact.

    *** THE CORE'S `down` IS 0, AND THE OLD `-H` WAS MEASURED AS A LIVE TRAP. ***
    A VortexArray's coordinates are NED offsets from the RUN ANCHOR, and the
    anchor sits at the flight altitude, so a core on the flightpath is level with
    the aircraft. `-H` was an altitude against an implied sea-level origin that
    no longer exists; kept, it puts the core a whole cruise altitude BELOW the
    run and the aircraft flies through still air. Measured both ways at this
    window: level gives errors [0.775, 0.168, 0.182, 0.027] m, non-monotone at
    the third refinement, which is the kink; `-H` gives
    [1.63e-4, 8.08e-5, 3.98e-5, 1.92e-5] m -- monotone, halving with dt, and
    below the 1e-4 m guard. So this test fails loudly rather than silently if the
    core is ever put out of reach again, which is why the second assertion is
    worth keeping alongside the first.
    """
    from atisim import wind

    ac = REGISTRY["boeing747"]
    V, H = CRUISE["boeing747"]["airspeed"], CRUISE["boeing747"]["altitude"]
    case = wind.PARKS_CASES["hannibal"]
    r0 = case["r0"]
    # ONE core, so "through the core" means through exactly one kink pair.
    array = wind.VortexArray(
        north=jnp.array([0.0]), down=jnp.array([0.0]),
        r0=jnp.array(r0), v0=jnp.array(case["v0"]),
    )
    model = wind.field_model(lambda p: wind.vortex_wind(p, array), ANCHOR)

    dts = np.array([1.0 / 16, 1.0 / 32, 1.0 / 64, 1.0 / 128])
    errors, _ = verification.fixed_control_refinement(
        ac, V, H, dts, dt_ref=1.0 / 2048.0, anchor=ANCHOR, earth_model=EARTH,
        wind_model=model, start_north=-2.0 * r0, d_elevator=0.0,
    )

    # Monotone refinement would mean every error smaller than the last.
    monotone = all(b < a for a, b in zip(errors, errors[1:]))
    assert not monotone, f"expected non-monotone refinement across the core, got {errors}"
    # And it is not the round-off floor doing it: the floor at 40,000 ft is about
    # 7e-11 m (ASSUMPTIONS.md F4, pre-ECEF and superseded) and these errors are
    # centimetres -- measured min 2.73e-2 m, eight orders clear of any floor the
    # re-measurement could plausibly land on.
    assert errors.min() > 1e-4, f"errors are at the round-off floor: {errors}"


def test_a_uniform_horizontal_wind_only_translates_the_trajectory():
    """Galilean invariance -- the assertion the zero-wind test cannot make.

    A uniform wind W is a change of inertial frame. Fly the same aircraft with
    its ground velocity offset by W and the aerodynamics see an identical
    relative flow, so attitude and body rates must be untouched and position must
    differ by exactly W*t.

    PROJECT.md section 2 names putting vel_rel into the Coriolis term as a
    classic gust-modelling error. This test catches exactly that, and it is
    invisible in still air, which is why every test in the project was blind to
    it until now.

    SCOPE, stated because it is narrower than it looks: the wind is STEADY, so
    its material derivative is zero and this test CANNOT detect the other error
    section 2 names -- a spurious -m dW/dt term. That needs a time-varying field
    and a different assertion than invariance; it is section 8's open question
    rather than something claimed here.

    The wind is HORIZONTAL. A vertical component would leave the blown aircraft
    at a different altitude (2 m/s over 20 s is 40 m), and density is a function
    of altitude, so the two aircraft would not see the same dynamic pressure and
    the invariance would not hold to any tolerance worth asserting.

    **THE PREMISE IS NOW FALSE, AND THE THREE TOLERANCES WERE RE-PLACED BY
    INJECTION RATHER THAN WIDENED TO FIT.** The test was red for one session at
    the old <1e-11 / <1e-6. What replaced them is not a loosening dressed up:
    the defect this test exists to catch was INJECTED, its signature measured,
    and each tolerance placed inside the measured gap between the physics and
    the bug -- 122x under FLAT, with the table below. A bound chosen that way
    still discriminates; one widened until the test went green would not, which
    is the whole difference. Galilean invariance is a
    property of FLAT space, and this Earth is an ellipsoid. Two aircraft
    separated by W*t stand at different geodetic positions, so the local vertical
    -- and therefore gravity in body axes -- differs between them; that changes
    alpha, which changes the pitching moment, which is how a body-force
    difference reaches the ATTITUDE at all. Measured over the 20 s run below:

        Earth       d|quat|     d|omega|    d|position - W*t|   was
        earth.FLAT  4.811e-06   1.308e-06   2.3296e-02 m        <1e-11 / <1e-6
        WGS84_J2    2.721e-05   1.111e-05   1.2083e-01 m        <1e-11 / <1e-6

    Flown under `earth.FLAT`, which is where the claim is CLOSEST to true and
    which therefore separates the two mechanisms: FLAT is non-rotating, so its
    residual is the ellipsoid's curvature alone -- 140 m of northward travel is
    2.2e-5 rad of local-vertical rotation, and the 4.8e-6 on the quaternion is
    half of that, which is what a frame rotation looks like. The remaining
    factor of 5 up to WGS84_J2 is `2 Omega x v` acting on a ground velocity that
    the two runs deliberately differ in by W. Both are real physics; neither is
    an arithmetic defect, which is what tier 0 exists to find.

    WHAT SURVIVES, MEASURED BY INJECTION RATHER THAN ESTIMATED. This docstring
    first claimed the bug would sit "four to five orders above the curvature
    residual". That was an estimate and it is WRONG. Injecting the actual defect
    -- passing `vel_rel` where `earth_acceleration_terms` takes `vel_body` --
    and re-running this exact experiment:

        Earth        quantity    clean        vel_rel bug    ratio
        earth.FLAT   d|quat|     4.8105e-06   5.8728e-04     122x
        earth.FLAT   d|omega|    1.3078e-06   3.0739e-04     235x
        WGS84_J2     d|quat|     2.7207e-05   5.8795e-04      22x
        WGS84_J2     d|omega|    1.1108e-05   3.0731e-04      28x

    Two orders under FLAT, not four to five. The instrument still discriminates
    and the tolerances below are placed inside that gap -- but the margin is
    122x, not 1e4, and anyone tightening the curvature bound or loosening the
    assertion should know which of those two numbers they are working against.

    It is also why the run stays on FLAT: rotation costs a factor of five of
    discriminating power, because `2 Omega x v` acts on a ground velocity the
    two runs deliberately differ in by W. What it lost is the ability to be
    asserted at 1e-11, and
    restoring that needs a reference for translation on a curved Earth rather
    than a looser number here.
    """
    from atisim.state import dcm_body_to_ned
    from atisim.state import pos_ned as state_pos_ned

    W = jnp.array([7.0, -3.0, 0.0])  # m/s NED, horizontal by necessity

    def uniform_wind(wind_state, state, key, dt):
        return W, jnp.zeros(3), wind_state, key

    ac = REGISTRY["boeing747"]
    V, H = CRUISE["boeing747"]["airspeed"], CRUISE["boeing747"]["altitude"]
    earth_model = earth.FLAT
    x, _ = trim.trim(jnp.array(V), jnp.array(H), ac, ANCHOR, earth_model)
    state = trim.trimmed_state(
        x[0], x[3], jnp.array(V), jnp.array(H), ANCHOR, jnp.array(0.0)
    )
    controls = trim.trimmed_controls(x)._replace(elevator=x[1] + 0.01)

    # A body -> NED MATRIX at the aircraft's own position. `state.quat` is
    # body -> ECEF now, and W is stated in NED, so rotating it by the state
    # quaternion would offset the velocity in a plausible wrong direction.
    dcm = dcm_body_to_ned(state, ANCHOR)
    shifted = state._replace(vel_body=state.vel_body + dcm.T @ W)

    dt, n = jnp.array(0.02), 1000
    _, still = integrate.rollout(
        integrate.init_sim(state, jax.random.PRNGKey(0)), controls, dt, ac, n,
        ANCHOR, earth_model,
    )
    _, blown = integrate.rollout(
        integrate.init_sim(shifted, jax.random.PRNGKey(0)),
        controls, dt, ac, n, ANCHOR, earth_model, wind_model=uniform_wind,
    )

    # TOLERANCES SET BETWEEN THE PHYSICS AND THE BUG, both measured by
    # injection rather than argued. See the docstring for the table.
    #   curvature (clean, FLAT)  quat 4.81e-06   omega 1.31e-06
    #   vel_rel in Coriolis      quat 5.87e-04   omega 3.07e-04
    # 5e-5 and 1.5e-5 sit ~10x above the physics and ~12x and ~20x below the
    # bug, so the instrument still discriminates by two orders.
    np.testing.assert_allclose(np.asarray(blown.quat), np.asarray(still.quat), atol=5e-5)
    np.testing.assert_allclose(np.asarray(blown.omega), np.asarray(still.omega), atol=1.5e-5)

    # The translation is read in the LOCAL frame, because W is stated there and
    # a difference of ECEF offsets would depend on where on the Earth this flew.
    t = np.arange(1, n + 1) * float(dt)
    still_ned = np.asarray(jax.vmap(state_pos_ned, in_axes=(0, None))(still, ANCHOR))
    blown_ned = np.asarray(jax.vmap(state_pos_ned, in_axes=(0, None))(blown, ANCHOR))
    expected = still_ned + t[:, None] * np.asarray(W)
    # 2.3296e-02 m of curvature residual measured under FLAT; 0.1 m is ~4x that.
    np.testing.assert_allclose(blown_ned, expected, atol=0.1)


# --- the other gust error PROJECT.md section 2 names ---
#
# The Galilean test above catches the first (vel_rel in the Coriolis term) and
# says in its own docstring that it CANNOT catch the second, because a steady
# wind's material derivative is zero. These two close that seam.
#
# Note what the correct claim is NOT. It is tempting to fly a time-varying wind
# from a start state offset by W(0) and demand the rates match still air, as the
# steady test does. That is FALSE PHYSICS. Writing v~_b = v_b - C^T W(t) for the
# air-relative body velocity and differentiating,
#
#     v~_b_dot = F(v~_b, omega)/m + g_b - omega x v~_b - C^T Wdot
#
# so the air-relative state obeys the still-air equation PLUS a -C^T Wdot term.
# The two trajectories must diverge, and by a lot -- a 30 m/s wind swing on a
# 236 m/s cruise is a 13% airspeed excursion. That term is precisely what makes
# a time-varying wind something other than a change of inertial frame, and an
# invariance assertion is therefore the wrong instrument for it.
#
# The right instrument is a case with a closed-form answer, which is the first
# test, plus a structural check on the one place the bug could actually be
# written, which is the second.
#
# BOTH WERE CHECKED BY INJECTING THE BUG, because a test that can only pass
# demonstrates nothing. Adding
#
#     d.vel_body -= C^T (wind_ned - sim.wind_ned) / dt
#
# to `step`'s right-hand side fails both tests by many orders of magnitude. It
# also fails the Galilean test above -- but ONLY through a first-step transient,
# because `init_sim` seeds the wind cache to zeros while the model immediately
# returns W. Seed that cache to W, which is the one line anyone would write on
# seeing a spurious impulse at t = 0, and the Galilean test passes with the bug
# still in: 2.7e-15 on quat and 6.9e-16 on omega, against its own 1e-11
# tolerances. So the blindness its docstring claims is real, and measured.


def test_a_time_varying_uniform_wind_adds_no_body_force():
    """The second error section 2 names, against an exact solution.

    An air mass that accelerates does not push on the aeroplane. It only changes
    the flow the wings see, so the wind may enter through `vel_rel` and nowhere
    else; an explicit -m*dW/dt term double-counts. The signal such a term leaves
    is unmissable -- it integrates to -(W(t) - W(0)), up to 30 m/s of velocity
    error against the tolerance below.

    The experiment is `verification.free_fall_through_a_swinging_wind`, which
    carries the derivation of why a closed form and not an invariance is the
    right instrument. It lives there rather than here so the notebook runs THIS
    code rather than a second copy of it that could drift.

    **THIS TEST FAILED TWICE OVER FOR ONE SESSION. BOTH CAUSES ARE FIXED AND THE
    1e-9 WAS NOT WIDENED -- IT WAS REPLACED BY AN EXACT EQUALITY.**

    First, it did not reach its assertions at all: `verification.py` called
    `longitudinal_controls` while importing only `trimmed_controls`, so the
    experiment raised NameError. That was a SOURCE defect and it is repaired --
    the import is at `verification.py:337`.

    Second, past that, the number had moved and the source says why in its own
    docstring: `earth.FLAT` puts gravity along the LOCAL geodetic vertical, which
    rotates as the body travels, while the closed form uses the anchor's. The
    departure is GEOMETRIC and grows as t^3 -- measured 3.3222e-03 m over the 6 s
    run here, against this assertion's 1e-9 m.

    **THE CLAIM ITSELF SURVIVES INTACT, AND THAT IS MEASURED TOO.** Re-running
    with `SWING_W0` zeroed gives 0.00332221804103483 m against the gusting run's
    0.00332221804103483 m -- the same float, not the same to a tolerance. So the
    WIND-DEPENDENT part of the error is exactly zero, which is the whole claim: a
    spurious -m*dW/dt term would put up to 30 m/s of velocity error in there.
    What went was only the ability to assert it against round-off through that
    one absolute number. The source docstring's own remedy -- "difference two
    wind settings" -- had no route through the old signature, because the gust
    amplitude was a module constant. IT HAS ONE NOW: `amplitude` is a parameter
    of `free_fall_through_a_swinging_wind`, so the test below runs the
    experiment twice and differences it. That is why the assertion is `==`
    rather than a tolerance -- the two runs return the SAME FLOAT, so the
    wind-dependent part of the error is exactly zero rather than merely small,
    which is a stronger statement than the 1e-9 it replaced.
    """
    n, dt = 300, 0.02
    ac = verification.without_aerodynamics(REGISTRY["boeing747"])
    # A SEA-LEVEL anchor, deliberately not this module's cruise ANCHOR. The
    # experiment starts the body at a NED `down` of -3000 and its closed form is
    # written in the anchor's frame, and straight up from the anchor is the one
    # direction where the tangent plane and the ellipsoid agree exactly -- so
    # that -3000 is a geodetic height of 3,000 m only if the anchor is at h = 0.
    anchor = earth.anchor_at(np.radians(47.0), 0.0, 0.0)
    got = verification.free_fall_through_a_swinging_wind(ac, anchor, dt=dt, n=n)

    # The wind really did swing, rather than the model quietly returning zeros
    # and the experiment passing for that reason.
    assert got.elapsed == pytest.approx(n * dt)
    assert got.peak_wind == pytest.approx(29.4618, abs=1e-4)
    assert got.peak_dwdt == pytest.approx(88.3855, abs=1e-4)

    # THE CLAIM IS DIFFERENCED, NOT ABSOLUTE, and that is a sharpening rather
    # than a retreat. The absolute residual now carries a GEOMETRIC term that has
    # nothing to do with wind: `earth.FLAT` puts gravity along the LOCAL geodetic
    # vertical, which rotates as the body travels, while the closed form uses the
    # anchor's. It is 3.3222e-03 m over this 6 s run and grows as t^3.
    #
    # Differencing two amplitudes removes it exactly and leaves ONLY the
    # wind-dependent part, which is what this test was always about. That is the
    # remedy `verification.py`'s own docstring prescribed and previously had no
    # route to, because the amplitude was a module constant.
    #
    # Measured: gust-on 0.0033222180410348301 m against gust-off
    # 0.0033222180410348301 m -- the SAME FLOAT, so the difference is exactly
    # zero rather than merely small. A spurious -m*dW/dt term would put about
    # 84 m/s of velocity error in there, against the 88.4 m/s^2 this swings.
    still = verification.free_fall_through_a_swinging_wind(
        ac, anchor, dt=dt, n=n, amplitude=jnp.zeros(3)
    )
    assert got.max_position_error == still.max_position_error, (
        f"the wind reached the trajectory: {got.max_position_error} m gusting "
        f"against {still.max_position_error} m still"
    )

    # And the geometric term itself is bounded, so a real defect cannot hide
    # inside it. It is curvature over the distance travelled, not round-off.
    assert still.max_position_error < 1e-2, (
        f"the still-air residual is {still.max_position_error} m, past the "
        "curvature scale this run can produce"
    )


def test_a_step_ignores_the_wind_the_previous_step_applied():
    """The same claim structurally, with the real aerodynamics left in.

    `SimState` caches `wind_ned` from the previous step so a controller can sense
    the air without re-evaluating the model. That cache is exactly the ingredient
    a -m*dW/dt term would be built from: differencing it against the current
    sample is the obvious way to get a dW/dt, and it is one line.

    So: take one step twice from the same rigid-body state with the same wind
    model, varying ONLY the cached previous wind. `derivatives` takes an
    instantaneous wind value and no rate, so the two must agree bit for bit.
    Anything that differences the cache makes them differ by dt*(difference),
    which at these values is metres per second.

    Bit-identity rather than a tolerance, for the reason given at
    PRE_REFACTOR_VEL_HASH: the claim is that the cache is not read at all, and
    any tolerance admits a small amount of reading it.
    """
    held = jnp.array([11.0, -6.0, 2.0])

    def steady(wind_state, state, key, dt):
        del state, dt
        return held, jnp.zeros(3), wind_state, key

    ac = REGISTRY["boeing747"]
    V, H = CRUISE["boeing747"]["airspeed"], CRUISE["boeing747"]["altitude"]
    x, _ = trim.trim(jnp.array(V), jnp.array(H), ac, ANCHOR, EARTH)
    state = trim.trimmed_state(
        x[0], x[3], jnp.array(V), jnp.array(H), ANCHOR, jnp.array(0.0)
    )
    controls = trim.trimmed_controls(x)._replace(elevator=x[1] + 0.01)
    base = integrate.init_sim(state, jax.random.PRNGKey(0))

    def one_step(cached):
        after = integrate.step(
            base._replace(wind_ned=cached), controls, jnp.array(0.02), ac,
            ANCHOR, EARTH, wind_model=steady,
        )
        return jax.tree.map(lambda leaf: np.asarray(leaf).tobytes(), after.state)

    # Second cache is the negative of the wind actually applied, so a difference
    # quotient would see 2*held/dt -- around 1100 m/s^2 -- rather than zero.
    assert one_step(held) == one_step(-held)
    assert one_step(held) == one_step(jnp.zeros(3))


def test_the_trim_solve_converges_quadratically():
    """A Newton solve that merely converges may have a wrong Jacobian.

    Quadratic convergence -- the residual exponent roughly doubling each step --
    is the signature that jacfwd differentiates the same function the residual
    evaluates. A finite-difference or stale Jacobian still converges, linearly,
    and the final residual alone cannot tell them apart.

    Six unknowns now, not three, and the convergence survived it: measured
    history [2.253, 4.11e-02, 7.80e-08, 2.85e-15, ...] on WGS84_J2, three usable
    iterations and a peak exponent ratio of 3.29. `earth.FLAT` gives
    [2.289, 4.24e-02, 8.19e-08, ...] and 3.30, so the rate is a property of the
    Jacobian rather than of the Earth. WGS84_J2 is used because it is the solve
    the package ships -- and the one whose lateral rows the three-unknown solver
    did not have.
    """
    ac = REGISTRY["boeing747"]
    V, H = CRUISE["boeing747"]["airspeed"], CRUISE["boeing747"]["altitude"]
    history = verification.newton_residual_history(
        jnp.array(V), jnp.array(H), ac, ANCHOR, EARTH, iterations=6
    )
    assert history[0] > 1.0, "the start point is already converged; pick a worse one"

    # takewhile, not a filter: the residual can dip below float64 resolution and
    # come back up, and a filter would splice non-contiguous iterations into a
    # sequence that was never actually walked.
    usable = np.array(list(itertools.takewhile(lambda r: r > 1e-13, history)))
    assert len(usable) >= 3, f"converged too fast to measure: {history}"

    e = np.log10(usable)
    ratios = (e[2:] - e[1:-1]) / (e[1:-1] - e[:-2])
    assert np.max(ratios) > 1.6, f"convergence looks linear, ratios {ratios}"


# Principal-axis inertias, well separated so the elliptic modulus is not near 0
# or 1. Not an aircraft -- these tests have no aircraft in them.
_I1, _I2, _I3 = 1420.0, 4070.0, 4780.0
# Middle component MUST be zero (see torque_free_omega). This pair gives
# m = 0.4928, rate = 0.5782 rad/s, and a1, a3 equal to 0.6 and 0.9 exactly.
_OMEGA0 = np.array([0.6, 0.0, 0.9])


def test_the_analytic_torque_free_solution_solves_eulers_equations():
    """Check the reference before using it as one.

    A mis-stated closed form would otherwise surface as a phantom integrator
    defect in the next test. 2001 points, not 601: the central-difference
    residual scales as h^2 and 601 gives 9.27e-7 against a 1e-6 tolerance, an 8%
    margin that would fail on any tightening.
    """
    t = np.linspace(0.0, 3.0, 2001)
    w = verification.torque_free_omega(_I1, _I2, _I3, _OMEGA0, t)
    I = np.array([_I1, _I2, _I3])[:, None]
    dwdt = np.gradient(w, t, axis=1)
    euler = -np.cross(w.T, (I * w).T).T / I
    np.testing.assert_allclose(dwdt[:, 5:-5], euler[:, 5:-5], atol=1e-6)


def test_the_analytic_solution_rejects_the_wrong_branch():
    """The domain guard, exercised. Without it this returns silent NaN.

    scipy.special.ellipj gives NaN for a modulus above 1, and conftest.py's
    jax_debug_nans cannot see it because this path is NumPy.
    """
    with pytest.raises(ValueError, match="separatrix"):
        verification.torque_free_omega(
            _I1, _I2, _I3, np.array([0.9, 0.0, 0.05]), np.array([0.0, 1.0])
        )


def test_the_integrator_reproduces_torque_free_rotation():
    """Conservation is not correctness.

    PROJECT.md section 4 records angular-momentum drift of 5.7e-13 over 60,000
    steps. A scheme can conserve H exactly and traverse the polhode at the wrong
    rate. This checks the trajectory instead of the invariant.

    **FLOWN UNDER `earth.FLAT`, AND THAT IS NOT NEGOTIABLE HERE** -- the same
    argument `verification.free_fall_through_a_swinging_wind` makes for its own
    closed form. `torque_free_omega` is Landau & Lifshitz's INERTIAL solution of
    Euler's equations. On a rotating Earth `state.omega` is the body rate
    relative to ECEF, and the plant integrates
    `omega_dot_be = I^-1 (M - omega_bi x I omega_bi) + omega_be x Omega_b` with
    `omega_bi = omega_be + Omega_b`, which is a different equation; there is no
    tolerance at which the elliptic functions are its solution. Under FLAT
    `Omega = 0`, the two collapse together, and the closed form IS the answer.

    Measured over the 3 s run below, worst |d(omega)| against the closed form:

        earth.FLAT   1.1720e-14   (against this test's atol of 1e-8)
        WGS84_J2     1.4900e-04

    The J2 figure is 2.0e-4 of |omega| and about `Omega/|omega|` in size, which
    is the frame difference and not an integrator defect -- reading it as one is
    exactly what choosing the wrong Earth here would produce.
    """
    from atisim.aircraft import inertia_tensor
    from atisim.state import euler_to_quat, state_from_ned
    from atisim.tests.conftest import make_test_aircraft

    inertia = inertia_tensor(_I1, _I2, _I3, 0.0)
    zeroed = dict(
        CL0=0.0, CLa=0.0, CLq=0.0, CLde=0.0, Cm0=0.0, Cma=0.0, Cmq=0.0, Cmde=0.0,
        CD0=0.0, CYb=0.0, CYp=0.0, CYr=0.0, CYdr=0.0, Clb=0.0, Clp=0.0, Clr=0.0,
        Clda=0.0, Cldr=0.0, Cnb=0.0, Cnp=0.0, Cnr=0.0, Cnda=0.0, Cndr=0.0,
        max_thrust=0.0,
    )
    ac = make_test_aircraft()._replace(
        inertia=inertia,
        inertia_inv=jnp.linalg.inv(inertia),
        **{k: jnp.array(v) for k, v in zeroed.items()},
    )
    # Its own anchor at the 3,000 m this flies, not the module's cruise ANCHOR.
    # Altitude cannot reach the answer at all -- every coefficient and the thrust
    # are zeroed, so density never enters -- so the aircraft sits AT the anchor
    # and the state is built from the local description it always had.
    anchor = earth.anchor_at(np.radians(47.0), 0.0, 3000.0)
    state = state_from_ned(
        jnp.zeros(3),
        jnp.array([60.0, 0.0, 0.0]),
        euler_to_quat(jnp.array(0.0), jnp.array(0.0), jnp.array(0.0)),
        jnp.array(_OMEGA0),
        anchor,
    )
    controls = trim.longitudinal_controls(jnp.array(0.0), jnp.array(0.0))
    dt, n = 0.002, 1500
    _, traj = integrate.rollout(
        integrate.init_sim(state, jax.random.PRNGKey(0)), controls, jnp.array(dt),
        ac, n, anchor, earth.FLAT,
    )

    t = np.arange(1, n + 1) * dt
    exact = verification.torque_free_omega(_I1, _I2, _I3, _OMEGA0, t)
    np.testing.assert_allclose(np.asarray(traj.omega).T, exact, atol=1e-8)
