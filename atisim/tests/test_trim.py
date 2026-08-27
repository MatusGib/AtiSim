import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import root

from atisim import dynamics, earth, trim
from atisim import state as st
from atisim.aircraft import CRUISE, REGISTRY
from atisim.units import RAD2DEG

AC = REGISTRY["boeing747"]
V = CRUISE["boeing747"]["airspeed"]
H = CRUISE["boeing747"]["altitude"]
# Trim is a function of latitude and heading now, so the tests below have to say
# where they are flying. 47N is the latitude the rest of this project's
# Earth-rotation work uses; heading is left at the default 0 (due north).
ANCHOR = earth.anchor_at(np.radians(47.0), 0.0, H)
EARTH = earth.WGS84_J2


def test_trim_converges():
    _, residual = trim.trim(jnp.array(V), jnp.array(H), AC, ANCHOR, EARTH)
    assert float(jnp.linalg.norm(residual)) < 1e-10


def test_trim_reproduces_the_source_flight_condition():
    """CR-2144 Table IX-3 tabulates alpha = 4.60 deg at flight condition 9.

    Recovering it means the atmosphere, unit conversions, mass, geometry and
    lift build-up are mutually consistent -- lift really does equal weight at
    ISA density for 40,000 ft and 774 ft/s.
    """
    x, _ = trim.trim(jnp.array(V), jnp.array(H), AC, ANCHOR, EARTH)
    assert float(x[0]) * RAD2DEG == pytest.approx(4.60, abs=0.1)


def test_trim_controls_are_physically_sensible():
    x, _ = trim.trim(jnp.array(V), jnp.array(H), AC, ANCHOR, EARTH)
    _, elevator, throttle, _, aileron, rudder = (float(v) for v in x)
    assert 0.0 < throttle < 1.0
    assert abs(elevator) < float(AC.elevator_limit)
    assert abs(aileron) < float(AC.aileron_limit)
    assert abs(rudder) < float(AC.rudder_limit)


def test_trim_matches_scipy():
    """Guard against a bug in the hand-rolled Newton loop."""
    x, _ = trim.trim(jnp.array(V), jnp.array(H), AC, ANCHOR, EARTH)

    def f(x_np):
        return np.asarray(
            trim.residual(
                jnp.array(x_np), jnp.array(V), jnp.array(H), AC, ANCHOR, EARTH, 0.0
            )
        )

    sol = root(f, np.asarray(trim.INITIAL_GUESS), tol=1e-12)
    assert sol.success
    np.testing.assert_allclose(np.asarray(x), sol.x, atol=1e-8)


def test_trimmed_flight_holds_altitude_and_airspeed():
    """60 s with fixed controls from trim.

    Trim is an exact fixed point of the ODE, so the tolerance here is tight on
    purpose: any drift at all means the integrator or the state construction
    disagrees with the residual the solver drove to zero.

    **THIS TEST DIAGNOSED A REAL DEFECT AND THE DEFECT IS NOW FIXED.** It was
    left red for one session with the note below, which turned out to be the
    right diagnosis:

        quantity                    was      migrated   after the fix
        geodetic altitude drift    <1e-6 m   4.2356 m   0.0295 m
        airspeed drift             <1e-6 m/s 0.1493 m/s 0.00103 m/s

    The mechanism is NOT the Earth's rotation. `trimmed_state` set `omega = 0`,
    which zeroes the trim residual at t = 0 but is not a steady level-flight
    condition over a curved Earth: holding altitude round an ellipsoid needs a
    continuous nose-down transport rate, and that state carried none, so the
    aircraft flew straighter than the surface curved. `earth.FLAT` --
    non-rotating, constant-g, still an ellipsoid -- gave 4.2290 m against
    WGS84_J2's 4.2356 m, so rotation and J2 together owned 6.6 mm of the 4.2 m
    and the geometry owned the rest.

    `trim.transport_rate_body` now supplies it and the drift falls 143x. A
    second defect surfaced in the same place: `trimmed_controls` was dropping
    the trim's aileron and rudder, worth 5.5e-8 m/s^2 here and 1.9e-5 at an
    east heading.

    What was always true, and is asserted first below at machine precision: the
    residual the solver drove to zero really is zero at t = 0. It was the
    EXTRAPOLATION of that fixed point over 60 s that the curved Earth broke, and
    the two are different claims.
    """
    integrate = pytest.importorskip("atisim.integrate")

    x, _ = trim.trim(jnp.array(V), jnp.array(H), AC, ANCHOR, EARTH)
    state = trim.trimmed_state(x[0], x[3], jnp.array(V), jnp.array(H), ANCHOR, 0.0)
    controls = trim.trimmed_controls(x)

    # The fixed point itself, which the curved Earth did NOT break.
    d = dynamics.derivatives(
        state, controls, AC, jnp.zeros(3), jnp.zeros(3), ANCHOR, EARTH
    )
    assert float(jnp.linalg.norm(d.vel_body)) < 1e-9
    assert float(jnp.linalg.norm(d.omega)) < 1e-9

    dt = 0.02
    _, hist = integrate.rollout(
        integrate.init_sim(state, jax.random.PRNGKey(0)),
        controls,
        jnp.array(dt),
        AC,
        int(60.0 / dt),
        ANCHOR,
        EARTH,
    )
    altitude = np.asarray(jax.vmap(st.altitude, in_axes=(0, None))(hist, ANCHOR))
    track = np.asarray(jax.vmap(st.pos_ned, in_axes=(0, None))(hist, ANCHOR))
    airspeed = np.linalg.norm(np.asarray(hist.vel_body), axis=1)
    # OPEN LOOP ON A CURVED EARTH CANNOT BE EXACT, and 1e-6 was a flat-Earth
    # tolerance. Holding altitude round an ellipsoid needs a continuous
    # nose-down transport rate; `trimmed_state` now carries it, evaluated at the
    # anchor. Over 60 s the aircraft flies 14 km and its latitude moves 0.127
    # deg, so the rate it needs is no longer quite the rate it holds -- and with
    # fixed controls nothing corrects that. Only a controller could.
    #
    #     transport rate absent (as first shipped)   4.2356 m
    #     transport rate carried                     0.0295 m
    #
    # 143x, and what is left is second order in the latitude change. The bound
    # below is set against that mechanism: a genuine defect in the state
    # construction or the integrator would not scale with the track length, and
    # would not have moved when the transport rate was added.
    assert np.abs(altitude - H).max() < 0.05
    assert np.abs(airspeed - V).max() < 5e-3
    # It must still be flying, not merely frozen.
    assert track[-1, 0] == pytest.approx(V * 60.0, rel=1e-6)


def test_trim_converges_away_from_the_reference_condition():
    for airspeed in (200.0, 220.0, 250.0):
        _, residual = trim.trim(jnp.array(airspeed), jnp.array(H), AC, ANCHOR, EARTH)
        assert float(jnp.linalg.norm(residual)) < 1e-9, airspeed


def test_slower_flight_needs_more_alpha():
    slow, _ = trim.trim(jnp.array(200.0), jnp.array(H), AC, ANCHOR, EARTH)
    fast, _ = trim.trim(jnp.array(250.0), jnp.array(H), AC, ANCHOR, EARTH)
    assert float(slow[0]) > float(fast[0])


def test_trim_converges_to_a_physically_absurd_solution():
    """Convergence and sense are different questions.

    `CL = CL0 + CLa*alpha` is linear, so a huge alpha compensates a tiny CLa and
    Newton lands on a root that satisfies the residual perfectly and is not a
    flight condition. A residual check detects non-convergence; it cannot detect
    nonsense. Until session 11 nothing in the project asked the second question,
    and the bound then went into `validation.sweep` -- which is where it was
    needed, but not where the defect is. Every other caller of `trim` was equally
    exposed, hence `trim.is_physical`.

    NO SPECIFIC ANGLE IS ASSERTED, and that is a finding rather than laziness.
    The far root Newton reaches is chaotically sensitive to the start conditions:
    at CLa = 1e-4, sea level, the same aircraft gives -632.1 deg at 85.0 m/s and
    -4232.1 deg at 84.9 m/s. The reproducible fact is the property -- converged,
    and absurd -- so that is what is asserted.
    """
    ac = REGISTRY["boeing747_approach"]._replace(CLa=jnp.array(0.1))
    anchor = earth.anchor_at(np.radians(47.0), 0.0, 0.0)
    x, res = trim.trim(jnp.array(85.0), jnp.array(0.0), ac, anchor, EARTH)
    assert float(jnp.linalg.norm(res)) < 1e-12, "this case converges; that is the point"
    assert not trim.is_physical(x, ac), f"alpha {float(x[0]) * RAD2DEG:.1f} deg passed"


def test_every_real_aircraft_trims_to_a_physical_solution():
    """The positive control, without which `is_physical` could just return False.

    Sweeps the whole registry at its own cruise condition. All four trim at 3-6
    deg, so the 15 deg bound is nowhere near binding on legitimate data -- which
    is the property that lets it be applied unconditionally.

    THE CESSNA 172 IS EXCLUDED, and not for convenience. `aircraft.py` zeroes its
    whole rudder set on purpose -- "*** THE RUDDER IS ABSENT ***", because the
    source omits Cl_dr and Cn_dr and its one rudder derivative has the wrong
    sign. A six-unknown trim carries `rudder` as an unknown, so for that aircraft
    the sixth Jacobian column is IDENTICALLY ZERO: the system is rank 5 of 6,
    exactly singular, and `jnp.linalg.solve` returns NaN rather than a bad
    answer. That is asserted below rather than skipped past, because a NaN trim
    reaching a registry sweep silently is exactly the failure this file exists to
    prevent.
    """
    rudderless = "cessna172"
    for name, ac in REGISTRY.items():
        if name == rudderless:
            continue
        altitude = CRUISE[name]["altitude"]
        anchor = earth.anchor_at(np.radians(47.0), 0.0, altitude)
        x, _ = trim.trim(
            jnp.array(CRUISE[name]["airspeed"]), jnp.array(altitude), ac, anchor, EARTH
        )
        assert trim.is_physical(x, ac), f"{name}: alpha {float(x[0]) * RAD2DEG:.2f} deg"

    ac = REGISTRY[rudderless]
    altitude = CRUISE[rudderless]["altitude"]
    anchor = earth.anchor_at(np.radians(47.0), 0.0, altitude)
    jacobian = np.asarray(
        jax.jacfwd(trim.residual)(
            trim.INITIAL_GUESS,
            jnp.array(CRUISE[rudderless]["airspeed"]),
            jnp.array(altitude),
            ac,
            anchor,
            EARTH,
            0.0,
        )
    )
    assert np.all(jacobian[:, 5] == 0.0), "the Cessna grew a rudder"
    assert np.linalg.matrix_rank(jacobian) == 5


def test_trim_zeroes_all_six_residuals_on_a_rotating_earth():
    """Wings-level level flight is not an equilibrium once the Earth turns.

    The count is easy to get wrong and the design got it wrong once: six
    residuals against alpha, beta, phi, elevator, aileron, rudder, throttle is
    SEVEN freedoms for six equations, a one-parameter family. It is closed by
    imposing beta = 0.

    `aileron` is in the set and is not optional: with beta = 0 and a deflected
    rudder, Cl is non-zero through Cl_dr, so the roll residual cannot vanish
    without it.
    """
    from atisim import earth
    from atisim.aircraft import CRUISE, REGISTRY
    from atisim.trim import trim

    ac = REGISTRY["boeing747"]
    airspeed = CRUISE["boeing747"]["airspeed"]
    altitude = CRUISE["boeing747"]["altitude"]
    anchor = earth.anchor_at(np.radians(47.0), 0.0, altitude)

    solution, residual = trim(
        airspeed, altitude, ac, anchor, earth.WGS84_J2, heading=np.radians(30.0),
    )
    assert np.max(np.abs(np.asarray(residual))) < 1e-9, (
        f"six-residual trim did not converge: {np.asarray(residual)}"
    )


def test_the_trimmed_bank_angle_is_measured_not_assumed():
    """The design ESTIMATED 0.2 deg from atan(0.0035). This measures it.

    Recorded rather than asserted tightly, because the estimate was never a
    prediction. What IS asserted is the physics that must hold: the bank is
    non-zero on a rotating Earth, it is zero when the Earth is switched off, and
    it reverses sign in the opposite hemisphere.
    """
    from atisim import earth
    from atisim.aircraft import CRUISE, REGISTRY
    from atisim.trim import trim

    ac = REGISTRY["boeing747"]
    airspeed = CRUISE["boeing747"]["airspeed"]
    altitude = CRUISE["boeing747"]["altitude"]

    def bank(heading_deg, lat_deg, model):
        anchor = earth.anchor_at(np.radians(lat_deg), 0.0, altitude)
        x, _ = trim(airspeed, altitude, ac, anchor, model,
                    heading=np.radians(heading_deg))
        return float(np.asarray(x)[3])

    # HEADING NORTH, deliberately. Flying "east" at CONSTANT HEADING follows a
    # LATITUDE CIRCLE, which is a turn -- the transport rate carries a yaw
    # component -V_E tan(lat)/(N+h), measured -3.95e-5 rad/s at 47N -- and a
    # turn needs coordination bank whether or not the Earth rotates. Under FLAT
    # that is 0.055 deg at heading 090 and -0.055 at 270, which would make the
    # "no bank without rotation" assertion below read as a failure when it is
    # actually a correct turn. At heading 000 the yaw term is identically zero.
    north = bank(0.0, 47.0, earth.WGS84_J2)
    south = bank(0.0, -47.0, earth.WGS84_J2)
    flat = bank(0.0, 47.0, earth.FLAT)

    assert abs(north) > 1e-5, "no bank on a rotating Earth"
    assert abs(flat) < 1e-12, "bank survived switching the Earth off"
    assert np.sign(north) == -np.sign(south), "bank does not reverse across the equator"

    # And the turn itself, recorded rather than merely avoided: east and west
    # bank oppositely under FLAT, by the amount that coordinates the latitude
    # circle. This is the check that the transport rate's yaw term is real.
    east = bank(90.0, 47.0, earth.FLAT)
    west = bank(270.0, 47.0, earth.FLAT)
    assert east == pytest.approx(-west, rel=1e-6)
    assert abs(np.degrees(east)) == pytest.approx(0.0548, abs=5e-4)
    print(f"\ntrimmed bank: {np.degrees(north):.4f} deg at 47N, "
          f"{np.degrees(south):.4f} deg at 47S")


def test_a_rudderless_aircraft_trims_instead_of_returning_nan():
    """ASSUMPTIONS.md F7, which this function has just made live.

    F7 recorded that a zero-authority control channel makes a Newton solve
    return NaN in silence, and bounded it as "latent for the shipped solvers"
    because "`trim` does not carry rudder as an unknown, so nothing in the
    package hits it". Carrying rudder as an unknown is what the six-unknown trim
    now does, so the latent case went live: the Cessna's
    CYdr = Cldr = Cndr = 0 makes the rudder column of the Jacobian identically
    zero and `jnp.linalg.solve` returned six NaNs.

    `lstsq` takes the minimum-norm step, which leaves the null direction alone.
    The Cessna is genuinely trimmable without a rudder because the AILERON
    carries yaw through Cnda -- that is why all six residuals vanish rather
    than just five.
    """
    from atisim import earth
    from atisim.aircraft import CRUISE, REGISTRY
    from atisim.trim import trim

    ac = REGISTRY["cessna172"]
    condition = CRUISE["cessna172"]
    assert float(ac.CYdr) == 0.0 and float(ac.Cldr) == 0.0 and float(ac.Cndr) == 0.0, (
        "the Cessna gained a rudder; this test no longer exercises F7"
    )

    anchor = earth.anchor_at(np.radians(47.0), 0.0, condition["altitude"])
    x, r = trim(
        condition["airspeed"], condition["altitude"], ac, anchor, earth.WGS84_J2
    )

    assert not np.any(np.isnan(np.asarray(x))), "F7's NaN is back"

    # THE YAW ROW IS SEPARATED, and not to make the test pass. Level flight over
    # a curved Earth carries the transport rate, so the aircraft is turning
    # slowly; a turn needs yaw authority to coordinate, and this aircraft has
    # none. The five rows it CAN reach still reach machine precision:
    residual = np.asarray(r)
    assert np.max(np.abs(residual[:5])) < 1e-9, (
        f"rudderless trim failed on a row it can reach: {residual}"
    )

    # What is left is yaw alone, and it is bounded by the physics rather than by
    # a chosen number. Measured 2.4e-9 rad/s^2 at heading 000 -- the coupling of
    # the transport PITCH rate through Ixz -- which integrates to 8e-6 deg/s
    # over a minute. The bound is the transport rate times the Earth rate, the
    # scale any residual of this origin must respect; a genuine yaw defect would
    # be orders above it.
    scale = (condition["airspeed"] / earth.A_WGS84) * earth.OMEGA_WGS84
    assert abs(residual[5]) < 1e3 * scale, (
        f"yaw residual {residual[5]:.3e} exceeds the {1e3 * scale:.3e} "
        "transport-coupling scale, so it is not the turn it cannot coordinate"
    )
    # The unreachable unknown is left at zero rather than wandering.
    assert float(x[5]) == 0.0, f"rudder came out at {float(x[5])}, not the null direction"


def test_lstsq_does_not_hide_a_genuinely_unsolvable_trim():
    """The risk of swapping `solve` for `lstsq`, checked rather than assumed.

    `lstsq` never returns NaN, which is the point -- but a solver that always
    returns SOMETHING could return a plausible wrong answer for a request that
    has no solution, which would be worse than F7's NaN, not better.

    It does not. An aircraft with no pitch authority at all cannot be trimmed,
    and the qdot residual stays at ~9e-5: eight orders above a converged solve,
    in the residual `trim` already returns. The caller sees it.
    """
    import jax.numpy as jnp

    from atisim import earth
    from atisim.aircraft import CRUISE, REGISTRY
    from atisim.trim import trim

    condition = CRUISE["boeing747"]
    no_pitch = REGISTRY["boeing747"]._replace(
        Cmde=jnp.array(0.0), CLde=jnp.array(0.0)
    )
    anchor = earth.anchor_at(np.radians(47.0), 0.0, condition["altitude"])
    _, r = trim(
        condition["airspeed"], condition["altitude"], no_pitch, anchor, earth.WGS84_J2
    )

    assert np.max(np.abs(np.asarray(r))) > 1e-6, (
        "an untrimmable aircraft converged, so lstsq is hiding the failure"
    )
