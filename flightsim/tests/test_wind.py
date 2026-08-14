"""Kelvin-Helmholtz vortex array, against its own source's stated behaviour.

Source: E. K. Parks, R. C. Wingrove, R. E. Bach, R. S. Mehta, "Identification of
Vortex-Induced Clear Air Turbulence Using Airline Flight Records", J. Aircraft
22(2), Feb 1985, pp. 124-129 (AIAA Paper 84-0270). This is the PRIMARY source
for the vortex model; Wingrove & Bach 1994 (which the rest of this project's
turbulence work is aimed at) cites it as ref 3 and reuses its identified
parameters.

Parks gives the model in closed form -- a solid-body (Rankine) core embedded in
an irrotational flow, arrays built by linear superposition -- and then states
three specific qualitative properties of it in prose. Those three statements are
what this module asserts, because they are the source's own words about its own
model and are therefore the sharpest available check that the transcription is
right rather than merely plausible.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flightsim import dynamics, wind
from flightsim.units import FT2M

# Parks Table/prose values, Case 1 (Hannibal, MO, 3 April 1981, DC-10, 37,000 ft)
CASE1_R0 = 600.0 * FT2M
CASE1_V0 = 85.0 * FT2M
CASE1_SPACING = 3500.0 * FT2M

# Parks Case 2 (Morton, WY, 16 July 1982, DC-10, 39,000 ft)
CASE2_R0 = 450.0 * FT2M
CASE2_V0 = 70.0 * FT2M
CASE2_SPACING = 3200.0 * FT2M


def single(r0=CASE1_R0, v0=CASE1_V0, altitude=11278.0):
    """One vortex, core centred at the origin at the given altitude."""
    return wind.VortexArray(
        north=jnp.array([0.0]),
        down=jnp.array([-altitude]),
        r0=jnp.array(r0),
        v0=jnp.array(v0),
    )


def _sample(array, north, altitude):
    return np.asarray(wind.vortex_wind(jnp.array([north, 0.0, -altitude]), array))


def _sweep(array, norths, altitude):
    """Vectorised sweep along track. A Python loop here re-traces per point and
    costs tens of seconds; vmap keeps the whole module inside its budget.
    """
    points = jnp.stack(
        [jnp.asarray(norths), jnp.zeros_like(norths), jnp.full_like(norths, -altitude)],
        axis=1,
    )
    return np.asarray(jax.vmap(lambda p: wind.vortex_wind(p, array))(points))


# --- the three properties Parks states in prose about its own model ----------


def test_through_the_core_centre_the_vertical_wind_peaks_at_v0():
    """Parks p.125: "For passage directly through the core center (d/r0 = 0) the
    vertical wind w_z has a maximum value of V0."

    Inside the core Eq. (6) is w_z = -V0 * l*cos(dpsi) / r0, linear in the
    along-track offset, so the peak sits exactly at the core edge l = r0.
    """
    array = single()
    alt = 11278.0
    offsets = np.linspace(-4 * CASE1_R0, 4 * CASE1_R0, 2001)
    w_up = -_sweep(array, offsets, alt)[:, 2]

    assert np.abs(w_up).max() == pytest.approx(CASE1_V0, rel=1e-3)
    peak_at = offsets[np.argmax(np.abs(w_up))]
    assert abs(abs(peak_at) - CASE1_R0) < 2.0  # peak is at the core edge


def test_through_the_core_centre_the_gust_is_an_antisymmetric_up_then_down_doublet():
    """Parks Fig. 6/9 describe the encounter as "sharp up and down gusts".

    Through the centre the horizontal component vanishes identically (Eq. 5
    with d = 0) and the vertical component is odd in the along-track offset.
    That antisymmetry is what separates a vortex from a 1-cosine gust, which is
    single-signed -- and it is why a vortex cannot produce a signed g-load
    asymmetry of its own.
    """
    array = single()
    alt = 11278.0
    for offset in (0.3 * CASE1_R0, CASE1_R0, 2.5 * CASE1_R0):
        before = _sample(array, -offset, alt)
        after = _sample(array, +offset, alt)
        assert before[2] == pytest.approx(-after[2], rel=1e-9)  # odd in z
        assert abs(before[0]) < 1e-9 and abs(after[0]) < 1e-9  # no horizontal
    # Up first, then down: approaching from the south of a positive-sense core.
    assert -_sample(array, -CASE1_R0, alt)[2] > 0.0
    assert -_sample(array, +CASE1_R0, alt)[2] < 0.0


def test_tangent_to_the_core_the_horizontal_wind_peaks_at_v0():
    """Parks p.125: "The largest peak for the horizontal wind is also V0,
    obtained when the flightpath is along the tangent to the cylindrical core
    (d/r0 = 1)."

    This is the check that the horizontal component and the d/r0 geometry are
    transcribed correctly -- the vertical-only tests above would pass with the
    horizontal term missing entirely.
    """
    alt = 11278.0
    array = single(altitude=alt - CASE1_R0)  # core one radius BELOW the aircraft
    offsets = np.linspace(-4 * CASE1_R0, 4 * CASE1_R0, 2001)
    w_horizontal = _sweep(array, offsets, alt)[:, 0]

    assert np.abs(w_horizontal).max() == pytest.approx(CASE1_V0, rel=1e-3)
    # and the peak is directly abeam the core, not offset along track
    assert abs(offsets[np.argmax(np.abs(w_horizontal))]) < 2.0


# --- the array ---------------------------------------------------------------


def test_an_array_is_the_linear_superposition_of_its_vortices():
    """Parks p.125: "The velocity perturbation induced by an array of vortices
    is modeled by superposition of the individual effects."

    Superposition is exact for a velocity field, and it is what makes the whole
    component framework (vortex + wave + updraft + Dryden) legitimate.
    """
    alt = 11278.0
    centres = np.array([-CASE1_SPACING, 0.0, CASE1_SPACING])
    array = wind.VortexArray(
        north=jnp.array(centres),
        down=jnp.array([-alt] * 3),
        r0=jnp.array(CASE1_R0),
        v0=jnp.array(CASE1_V0),
    )
    for x in (-2000.0, -300.0, 0.0, 450.0, 1800.0):
        total = _sample(array, x, alt)
        parts = sum(
            _sample(
                wind.VortexArray(
                    north=jnp.array([c]), down=jnp.array([-alt]),
                    r0=jnp.array(CASE1_R0), v0=jnp.array(CASE1_V0),
                ),
                x, alt,
            )
            for c in centres
        )
        np.testing.assert_allclose(total, parts, atol=1e-12)


def test_the_core_is_solid_body_and_the_outside_is_irrotational():
    """Parks p.125: "a rotational (solid-body) core embedded in an irrotational
    flow", magnitude Gamma*r/(2*pi*r0^2) inside and Gamma/(2*pi*r) outside.

    Inside, speed is LINEAR in radius; outside it falls as 1/r. The linear
    interior is why a point sample plus an analytic gradient is not an
    approximation at all while the whole span sits inside the core.
    """
    array = single()
    alt = 11278.0

    def speed(x):
        return np.linalg.norm(_sample(array, x, alt))

    for frac in (0.25, 0.5, 0.75):
        assert speed(frac * CASE1_R0) == pytest.approx(frac * CASE1_V0, rel=1e-9)
    for mult in (2.0, 4.0, 8.0):
        assert speed(mult * CASE1_R0) == pytest.approx(CASE1_V0 / mult, rel=1e-9)


def test_the_spacing_to_core_diameter_ratio_matches_the_sources_own_cases():
    """Parks p.128 checks its identified arrays against Scorer's theory: "Scorer
    calculated that the ratio of spacing to core diameter would be of the order
    of 2.7. For the two cases discussed in this paper, the ratio of spacing to
    core diameter ranged from about 2.9 to 3.5."

    This is a data check, not a physics check, and it is the reason the array
    spacing in this module is a cited number rather than a free parameter. It
    is asserted so that a later edit to the case constants cannot silently
    drift away from the source's own consistency check.
    """
    for spacing, r0 in ((CASE1_SPACING, CASE1_R0), (CASE2_SPACING, CASE2_R0)):
        assert 2.8 < spacing / (2.0 * r0) < 3.6
    assert CASE1_SPACING / (2.0 * CASE1_R0) == pytest.approx(2.92, abs=0.01)
    assert CASE2_SPACING / (2.0 * CASE2_R0) == pytest.approx(3.56, abs=0.01)


# --- the wind-model contract -------------------------------------------------


def test_the_vortex_model_satisfies_the_wind_model_contract(test_aircraft):
    """Shape, and that a deterministic model leaves the PRNG key untouched.

    Leaving the key alone is what makes a batch of keys vary ONLY the stochastic
    components, so a Monte Carlo ensemble hits the same vortex every time. That
    is the experiment design the Fig. 8 error bars depend on.
    """
    from flightsim.state import State, euler_to_quat

    model = wind.vortex_model(single())
    state = State(
        pos_ned=jnp.array([0.0, 0.0, -11278.0]),
        vel_body=jnp.array([236.0, 0.0, 0.0]),
        quat=euler_to_quat(jnp.array(0.0), jnp.array(0.05), jnp.array(0.0)),
        omega=jnp.zeros(3),
    )
    key = jax.random.PRNGKey(0)
    wind_ned, omega_gust, wind_state, out_key = model(
        wind.zero_wind_state(), state, key, jnp.array(0.02)
    )
    assert wind_ned.shape == (3,)
    assert omega_gust.shape == (3,)
    assert np.array_equal(np.asarray(out_key), np.asarray(key))


def test_the_vortex_produces_a_pitching_gust_at_the_core_edge():
    """omega_gust is not optional for a vortex.

    Inside the core the vertical gust is linear along track, so the gradient
    dw/dx is constant and equal to V0/r0 -- a genuine pitch-rate input of
    V0/r0 rad/s sustained for the whole core traverse, not a rounding term.

    Sign, which is the easy one to get backwards: q_gust = -dw_g/dx, so with
    the core's downwash increasing ahead of the aircraft q_gust is NEGATIVE.
    dynamics.py forms omega_rel = omega - omega_gust, so the aero model sees a
    POSITIVE effective pitch rate -- the tail sits in the updraft half of the
    doublet and the nose in the downdraft half, which is a nose-up rate. With
    Cmq < 0 that is a nose-down moment, matching the source's own report of the
    aircraft pitching down on entering the positive vertical gust.
    """
    from flightsim.state import State, euler_to_quat

    array = single()
    model = wind.vortex_model(array)
    state = State(
        pos_ned=jnp.array([0.5 * CASE1_R0, 0.0, -11278.0]),
        vel_body=jnp.array([236.0, 0.0, 0.0]),
        quat=euler_to_quat(jnp.array(0.0), jnp.array(0.0), jnp.array(0.0)),
        omega=jnp.zeros(3),
    )
    _, omega_gust, _, _ = model(
        wind.zero_wind_state(), state, jax.random.PRNGKey(0), jnp.array(0.02)
    )
    magnitude = CASE1_V0 / CASE1_R0  # rad/s, constant inside the core
    assert float(omega_gust[1]) == pytest.approx(-magnitude, rel=1e-6)
    # what the aero model actually sees, at zero body rate
    assert float(-omega_gust[1]) > 0.0
    assert abs(float(omega_gust[0])) < 1e-12  # no roll: axis is across track
    assert abs(float(omega_gust[2])) < 1e-12


# --- the updraft column ------------------------------------------------------

# Wingrove & Bach 1994 p.756, Bermuda 12 Oct 1983: over 80 ft/s, 20 s traverse.
UPDRAFT_W0 = 80.0 * FT2M
UPDRAFT_SECONDS = 20.0


def _column(sharpness, w0=UPDRAFT_W0, radius=2360.0):
    return wind.UpdraftColumn(
        north=jnp.array(0.0), east=jnp.array(0.0), w0=jnp.array(w0),
        radius=jnp.array(radius), sharpness=jnp.array(float(sharpness)),
    )


def test_the_updraft_peaks_at_its_stated_magnitude_and_points_up():
    """Sign first: an updraft is air moving UP, which is NEGATIVE in NED z.

    Getting this backwards is the single most likely error in the whole
    module and it would look plausible for a long time -- the aircraft would
    simply pitch the wrong way.
    """
    column = _column(2.0)
    centre = np.asarray(wind.updraft_wind(jnp.array([0.0, 0.0, -11278.0]), column))
    assert centre[2] == pytest.approx(-UPDRAFT_W0, rel=1e-9)  # negative == up
    assert abs(centre[0]) < 1e-12 and abs(centre[1]) < 1e-12
    # and it decays away from the axis
    far = np.asarray(wind.updraft_wind(jnp.array([8000.0, 0.0, -11278.0]), column))
    assert abs(far[2]) < 0.01 * UPDRAFT_W0


def test_the_declared_sharpness_controls_the_edge_gradient_not_the_magnitude():
    """`sharpness` is a MODELLING PARAMETER, not source data.

    The paper fixes the magnitude and the duration and says nothing about the
    edge. This asserts what the knob actually does -- it steepens the edge while
    leaving the peak alone -- so that a result which depends on it is visibly a
    function of a declared choice rather than of the cited 80 ft/s.
    """
    radius = 2360.0
    gradients = {}
    for sharpness in (2.0, 4.0, 8.0):
        column = _column(sharpness, radius=radius)
        offsets = np.linspace(0.0, 2.5 * radius, 1200)
        w_up = -np.array(
            [
                float(wind.updraft_wind(jnp.array([x, 0.0, -11278.0]), column)[2])
                for x in offsets[::40]
            ]
        )
        gradients[sharpness] = np.abs(np.gradient(w_up, offsets[::40])).max()
        assert w_up.max() == pytest.approx(UPDRAFT_W0, rel=1e-6)  # peak unchanged
    assert gradients[4.0] > gradients[2.0]
    assert gradients[8.0] > gradients[4.0]


def test_fields_superpose():
    """Parks builds arrays by superposition and aero sees only the summed field,
    so a vortex sitting inside an updraft costs nothing beyond the two parts.
    """
    alt = 11278.0
    array = single()
    column = _column(4.0)
    combined = wind.superpose(
        lambda p: wind.vortex_wind(p, array), lambda p: wind.updraft_wind(p, column)
    )
    for x in (-500.0, 0.0, 137.0, 3000.0):
        point = jnp.array([x, 0.0, -alt])
        np.testing.assert_allclose(
            np.asarray(combined(point)),
            np.asarray(wind.vortex_wind(point, array))
            + np.asarray(wind.updraft_wind(point, column)),
            atol=1e-12,
        )


def test_the_updraft_weathercocks_where_the_vortex_does_not():
    """The mechanism behind Wingrove & Bach's Fig. 8 discriminator.

    A 20 s updraft traverse is about three 747 short periods (6.6 s undamped),
    so the aircraft has time to reach a new trim attitude and weathercocks by
    several degrees. A 1.5 s vortex traverse is a fifth of one, so the same
    aircraft absorbs the incidence change as load with almost no attitude
    response. Same airframe, same linear aero -- the separation is a pure
    rigid-body timescale effect, which is exactly why this model can reproduce
    the discriminator while it cannot reproduce the papers' +/-g asymmetry.
    """
    from flightsim import integrate, trim
    from flightsim.aircraft import CRUISE, REGISTRY
    from flightsim.state import quat_to_euler
    from flightsim.units import RAD2DEG

    ac = REGISTRY["boeing747"]
    v, h = CRUISE["boeing747"]["airspeed"], CRUISE["boeing747"]["altitude"]
    x, _ = trim.trim(jnp.array(v), jnp.array(h), ac)
    controls = trim.trimmed_controls(x[1], x[2])

    radius = 0.5 * UPDRAFT_SECONDS * v  # the paper's 20 s traverse, as a distance
    column = _column(6.0, radius=radius)
    state = trim.trimmed_state(x[0], jnp.array(v), jnp.array(h))
    state = state._replace(pos_ned=jnp.array([-2.0 * radius, 0.0, -h]))

    dt = 0.02
    n = int(round(4.0 * radius / v / dt))
    _, hist = integrate.rollout(
        integrate.init_sim(state, jax.random.PRNGKey(0)), controls,
        jnp.array(dt), ac, n, wind_model=wind.updraft_model(column),
    )
    theta = np.asarray(jax.vmap(quat_to_euler)(hist.quat))[:, 1]
    north = np.asarray(hist.pos_ned)[:, 0]

    inside = np.abs(north) <= radius
    dtheta = (theta[inside].max() - theta[inside].min()) * RAD2DEG
    # Measured in-column pitch excursion, by declared sharpness:
    #   2 -> 3.63 deg,  4 -> 3.63,  6 -> 4.39,  10 -> 5.34
    # against the vortex's 1.89 deg in-core measured above, the paper's stated
    # 5.2 deg for this case, and Fig. 8's 6.2 deg updraft cluster. The spread
    # across sharpness is the whole reason that parameter is declared rather
    # than defaulted: the ANSWER depends on it, so a result must state it.
    assert dtheta > 3.0, dtheta  # measured 4.39 at sharpness 6
    assert dtheta > 1.5 * 1.89  # separated from the vortex case, the Fig. 8 claim


# --- the regression guarantee ------------------------------------------------


def test_a_zero_strength_vortex_is_bit_identical_to_still_air():
    """The validated baseline must not move by a single bit.

    np.array_equal, NOT assert_allclose: an atol=1e-12 comparison would pass
    even if the model perturbed the trajectory at the 1e-14 level per step,
    which is exactly the regime where an operation-reordering bug hides.

    This holds because dynamics.py:37-38 applies wind unconditionally --
    `vel_body - dcm.T @ wind_ned` and `omega - omega_gust` -- so a model
    returning exact zeros runs the identical sequence of floating-point
    operations zero_wind runs: `dcm.T @ zeros` is exactly zeros and
    `x - 0.0 == x` for every finite x. It would STOP holding the moment the
    gust were folded into aero.coefficients instead of applied at the dynamics
    boundary, which is precisely why it is asserted rather than assumed.

    The real 747 at CR-2144 FC9 rather than the synthetic fixture, so wave drag
    and the true magnitudes are in the loop.
    """
    from flightsim import integrate, trim
    from flightsim.aircraft import CRUISE, REGISTRY

    ac = REGISTRY["boeing747"]
    v, h = CRUISE["boeing747"]["airspeed"], CRUISE["boeing747"]["altitude"]
    x, _ = trim.trim(jnp.array(v), jnp.array(h), ac)
    state = trim.trimmed_state(x[0], jnp.array(v), jnp.array(h))
    controls = trim.trimmed_controls(x[1], x[2])

    inert = wind.vortex_model(
        wind.VortexArray(
            north=jnp.array([0.0]), down=jnp.array([-h]),
            r0=jnp.array(CASE1_R0), v0=jnp.array(0.0),  # zero strength
        )
    )
    sim = integrate.init_sim(state, jax.random.PRNGKey(0))
    base, _ = integrate.rollout(sim, controls, jnp.array(0.02), ac, 2000)
    off, _ = integrate.rollout(sim, controls, jnp.array(0.02), ac, 2000, wind_model=inert)

    for field in ("pos_ned", "vel_body", "quat", "omega"):
        assert np.array_equal(
            np.asarray(getattr(base.state, field)), np.asarray(getattr(off.state, field))
        ), field  # measured: exact equality, max |diff| 0.0, over 2,000 steps


# --- the source's own encounter ----------------------------------------------


def test_parks_case_1_reproduces_the_gust_spacing_and_pitch_signature():
    """Fly the 747 through the Case 1 array and check what is checkable.

    *** WHAT THIS CAN AND CANNOT CLAIM. *** Parks Case 1 was a DC-10 at
    37,000 ft; this project's 747 is validated at CR-2144 FC9, 40,000 ft
    M 0.80. Different type, altitude and speed, and the DC-10's wing loading is
    roughly 1.3x the 747's, so the 747 takes MORE g for the same gust. Every
    load comparison here is therefore order-of-magnitude, and the assertions
    below are bands and orderings, never values.

    What IS sharp is the gust spacing: it is pure kinematics, set by the
    identified 3500 ft array spacing and the flight speed, and independent of
    the airframe. Parks reports "sharp up and down gusts about 5 s apart";
    3500 ft at this aircraft's 235.9 m/s gives 4.52 s.

    The in-core pitch excursion is the Wingrove & Bach Fig. 8 discriminator.
    Measured here: 2.20 deg over the first core, against Fig. 8's 1.4 deg
    extreme for the vortex category and 6.2 deg / 12 deg for the updraft and
    manoeuvring categories -- the right cluster by a wide margin.

    *** THE LEAD-IN IS PART OF THE MEASUREMENT. *** The vortex far field falls
    off only as 1/r, so starting too close launches the aircraft out of
    equilibrium and contaminates the first core. Measured n_z at t=0 against
    lead-in distance: 1.2033 at 6*r0, 1.0696 at 20, 1.0352 at 40, 1.0127 at
    100; first-core pitch excursion correspondingly 1.888 deg at 6*r0 rising to
    a converged 2.18-2.22 beyond about 12. The 6*r0 lead-in used originally
    understated the answer by 15% and did so invisibly, which is why the
    initial load factor is now asserted rather than assumed.

    NOTE the windowing trap, which is why the assertion is on the FIRST core
    and not the run: whole-run pitch excursion is 8.79 deg, because the
    post-encounter phugoid dwarfs the encounter itself. Measuring that instead
    would land in the manoeuvring cluster and "confirm" the wrong physics.
    """
    from flightsim import integrate, trim
    from flightsim.aircraft import CRUISE, REGISTRY
    from flightsim.state import quat_to_euler
    from flightsim.units import RAD2DEG

    ac = REGISTRY["boeing747"]
    v, h = CRUISE["boeing747"]["airspeed"], CRUISE["boeing747"]["altitude"]
    x, _ = trim.trim(jnp.array(v), jnp.array(h), ac)
    alpha = float(x[0])
    controls = trim.trimmed_controls(x[1], x[2])

    array = wind.VortexArray(
        north=jnp.array([0.0, CASE1_SPACING]),
        down=jnp.array([-h, -h]),
        r0=jnp.array(CASE1_R0),
        v0=jnp.array(CASE1_V0),
    )
    lead_in = 40.0 * CASE1_R0
    state = trim.trimmed_state(jnp.array(alpha), jnp.array(v), jnp.array(h))
    state = state._replace(pos_ned=jnp.array([-lead_in, 0.0, -h]))

    model = wind.vortex_model(array)
    dt = 0.01
    n = int(round((CASE1_SPACING + lead_in + 6.0 * CASE1_R0) / v / dt))
    _, hist = integrate.rollout(
        integrate.init_sim(state, jax.random.PRNGKey(0)), controls,
        jnp.array(dt), ac, n, wind_model=model,
    )
    theta = np.asarray(jax.vmap(quat_to_euler)(hist.quat))[:, 1]
    north = np.asarray(hist.pos_ned)[:, 0]

    # The aircraft must START in equilibrium, or the first core is measuring
    # the launch transient as much as the vortex.
    n_z0 = float(
        dynamics.load_factor(state, controls, ac, *model(
            wind.zero_wind_state(), state, jax.random.PRNGKey(0), jnp.array(dt)
        )[:2])
    )
    assert n_z0 == pytest.approx(0.9967, abs=0.05), n_z0  # measured 1.0352

    # kinematics: gust spacing is airframe-independent
    assert CASE1_SPACING / v == pytest.approx(4.52, abs=0.05)

    in_first_core = np.abs(north - 0.0) <= CASE1_R0
    dtheta = (theta[in_first_core].max() - theta[in_first_core].min()) * RAD2DEG
    assert 1.0 < dtheta < 3.5, dtheta  # measured 2.20 deg; Fig. 8 vortex ~1.4

    # and the encounter must be far smaller in pitch than the post-encounter
    # phugoid, which is the windowing trap this test exists to pin down
    whole = (theta.max() - theta.min()) * RAD2DEG
    assert whole > 3.0 * dtheta  # measured 8.33 vs 2.20


# --- A1: sampled gradients ---------------------------------------------------


def _level_state(north=0.0, altitude=11278.0, u=236.0):
    """Wings-level, heading north, at altitude."""
    from flightsim.state import State, euler_to_quat

    return State(
        pos_ned=jnp.array([north, 0.0, -altitude]),
        vel_body=jnp.array([u, 0.0, 0.0]),
        quat=euler_to_quat(jnp.array(0.0), jnp.array(0.0), jnp.array(0.0)),
        omega=jnp.zeros(3),
    )


def test_a_uniform_field_produces_exactly_zero_sampled_rates():
    """Reduction property 1. A uniform field has no gradient, and a symmetric
    station set must return exactly zero rather than a small residual -- a
    residual here would be a spurious rolling input in still-ish air."""
    from flightsim import airframe
    from flightsim.aircraft import REGISTRY

    ac = REGISTRY["boeing747"]
    st = airframe.stations(ac)
    field = lambda p: jnp.array([3.0, -2.0, 1.5])  # noqa: E731
    s = _level_state()

    rates = wind.sampled_rates(s.pos_ned, s.quat, field, st)
    assert np.array_equal(np.asarray(rates), np.zeros(3))


def test_a_linear_field_reproduces_the_analytic_gradient_exactly():
    """Reduction property 2, and the one that makes A1 safe to adopt: a
    least-squares slope through samples of a linear function IS its exact
    slope, so for any field the current model handles correctly, A1 returns
    the identical answer. Every existing result is therefore unmoved."""
    from flightsim import airframe
    from flightsim.aircraft import REGISTRY

    ac = REGISTRY["boeing747"]
    st = airframe.stations(ac)
    # Linear in every component and every direction, with no curvature at all.
    field = lambda p: jnp.array(  # noqa: E731
        [0.01 * p[0] + 0.02 * p[1], 0.03 * p[0] - 0.01 * p[2], -0.02 * p[0] + 0.04 * p[1]]
    )
    s = _level_state()

    sampled = wind.sampled_rates(s.pos_ned, s.quat, field, st)
    analytic = wind.gust_rates(s.pos_ned, s.quat, field)
    assert np.allclose(np.asarray(sampled), np.asarray(analytic), rtol=1e-9, atol=1e-12)


def test_the_vortex_core_gives_the_same_pitch_rate_as_the_tangent():
    """Inside a Rankine core the vertical gust is LINEAR along track, so the
    secant and the tangent must agree exactly. This is the strength of this
    field/model pairing that ASSUMPTIONS.md section E2 records: while the whole
    airframe is inside the core, a point sample plus a gradient is not an
    approximation at all."""
    from flightsim import airframe
    from flightsim.aircraft import REGISTRY

    ac = REGISTRY["boeing747"]
    st = airframe.stations(ac)
    array = single(r0=8000.0)  # core far larger than the airframe, so it stays inside
    field = lambda p: wind.vortex_wind(p, array)  # noqa: E731
    s = _level_state(north=0.25 * 8000.0)

    sampled = wind.sampled_rates(s.pos_ned, s.quat, field, st)
    analytic = wind.gust_rates(s.pos_ned, s.quat, field)
    assert float(sampled[1]) == pytest.approx(float(analytic[1]), rel=1e-9)


def test_a_curved_field_makes_the_secant_differ_from_the_tangent():
    """The test that gives A1 a reason to exist. A quadratic gust profile has a
    centreline slope that is not the slope the wing integrates, and the two
    must therefore disagree. If this passes trivially, the fit is not being
    taken across the airframe at all."""
    from flightsim import airframe
    from flightsim.aircraft import REGISTRY

    ac = REGISTRY["boeing747"]
    st = airframe.stations(ac)
    # Vertical gust quadratic across the span: zero slope at the centreline,
    # non-zero average slope across it.
    field = lambda p: jnp.array([0.0, 0.0, 1e-4 * p[1] ** 2])  # noqa: E731
    s = _level_state()

    sampled = wind.sampled_rates(s.pos_ned, s.quat, field, st)
    analytic = wind.gust_rates(s.pos_ned, s.quat, field)
    assert float(analytic[0]) == pytest.approx(0.0, abs=1e-12)
    assert abs(float(sampled[0])) < 1e-12, "a symmetric quadratic still has zero net slope"

    # Now break the symmetry: a cubic has a genuinely different secant.
    field3 = lambda p: jnp.array([0.0, 0.0, 1e-7 * p[1] ** 3])  # noqa: E731
    sampled3 = wind.sampled_rates(s.pos_ned, s.quat, field3, st)
    analytic3 = wind.gust_rates(s.pos_ned, s.quat, field3)
    assert float(analytic3[0]) == pytest.approx(0.0, abs=1e-12)
    assert abs(float(sampled3[0])) > 1e-9, "the cubic's secant must differ from its tangent"
