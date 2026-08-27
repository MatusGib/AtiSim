import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import root

from atisim import earth, trim
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
    """
    integrate = pytest.importorskip(
        "atisim.integrate",
        exc_type=ImportError,
        reason="integrate.py still holds the pre-Earth frame assumptions and does "
        "not import; the consumer migration re-establishes it. Its assertions "
        "belong to that migration too -- an ECEF-propagated trim does not hold "
        "local-NED altitude to 1e-6 over 60 s merely because vdot vanished at "
        "t = 0, so the body below is left on the old API deliberately, to fail "
        "loudly rather than quietly once the import works again.",
    )
    x, _ = trim.trim(jnp.array(V), jnp.array(H), AC)
    state = trim.trimmed_state(x[0], jnp.array(V), jnp.array(H))
    controls = trim.trimmed_controls(x[1], x[2])
    dt = 0.02
    _, hist = integrate.rollout(
        integrate.init_sim(state, jax.random.PRNGKey(0)),
        controls,
        jnp.array(dt),
        AC,
        int(60.0 / dt),
    )
    altitude = -np.asarray(hist.pos_ned)[:, 2]
    airspeed = np.linalg.norm(np.asarray(hist.vel_body), axis=1)
    assert np.abs(altitude - H).max() < 1e-6
    assert np.abs(airspeed - V).max() < 1e-6
    # It must still be flying, not merely frozen.
    assert np.asarray(hist.pos_ned)[-1, 0] == pytest.approx(V * 60.0, rel=1e-9)


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

    def bank(lat_deg, model):
        anchor = earth.anchor_at(np.radians(lat_deg), 0.0, altitude)
        x, _ = trim(airspeed, altitude, ac, anchor, model, heading=np.radians(90.0))
        return float(np.asarray(x)[3])

    north = bank(47.0, earth.WGS84_J2)
    south = bank(-47.0, earth.WGS84_J2)
    flat = bank(47.0, earth.FLAT)

    assert abs(north) > 1e-5, "no bank on a rotating Earth"
    assert abs(flat) < 1e-12, "bank survived switching the Earth off"
    assert np.sign(north) == -np.sign(south), "bank does not reverse across the equator"
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
    assert np.max(np.abs(np.asarray(r))) < 1e-9, (
        f"rudderless trim did not converge: {np.asarray(r)}"
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
