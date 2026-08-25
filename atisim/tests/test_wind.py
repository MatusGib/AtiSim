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

from atisim import dynamics, wind
from atisim.units import FT2M

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
    from atisim.state import State, euler_to_quat

    model = wind.vortex_model(single())
    state = State(
        pos_ned=jnp.array([0.0, 0.0, -11278.0]),
        vel_body=jnp.array([236.0, 0.0, 0.0]),
        quat=euler_to_quat(jnp.array(0.0), jnp.array(0.05), jnp.array(0.0)),
        omega=jnp.zeros(3),
    )
    key = jax.random.PRNGKey(0)
    wind_ned, omega_gust, wind_state, out_key, _ = model(
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
    from atisim.state import State, euler_to_quat

    array = single()
    model = wind.vortex_model(array)
    state = State(
        pos_ned=jnp.array([0.5 * CASE1_R0, 0.0, -11278.0]),
        vel_body=jnp.array([236.0, 0.0, 0.0]),
        quat=euler_to_quat(jnp.array(0.0), jnp.array(0.0), jnp.array(0.0)),
        omega=jnp.zeros(3),
    )
    _, omega_gust, _, _, _ = model(
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
    from atisim import integrate, trim
    from atisim.aircraft import CRUISE, REGISTRY
    from atisim.state import quat_to_euler
    from atisim.units import RAD2DEG

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
    from atisim import integrate, trim
    from atisim.aircraft import CRUISE, REGISTRY

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
    from atisim import integrate, trim
    from atisim.aircraft import CRUISE, REGISTRY
    from atisim.state import quat_to_euler
    from atisim.units import RAD2DEG

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
    from atisim.state import State, euler_to_quat

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
    from atisim import airframe
    from atisim.aircraft import REGISTRY

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
    from atisim import airframe
    from atisim.aircraft import REGISTRY

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
    from atisim import airframe
    from atisim.aircraft import REGISTRY

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
    from atisim import airframe
    from atisim.aircraft import REGISTRY

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


def test_the_sampled_wind_model_matches_the_contract():
    """Same signature as zero_wind and field_model, so it drops into
    integrate.step, autopilot and panel with no change to any of them."""
    from atisim import airframe
    from atisim.aircraft import REGISTRY

    ac = REGISTRY["boeing747"]
    array = single()
    model = wind.sampled_field_model(
        lambda p: wind.vortex_wind(p, array), airframe.stations(ac)
    )
    s = _level_state()
    key = jax.random.PRNGKey(0)
    wind_ned, omega_gust, wind_state, out_key, _ = model(
        wind.zero_wind_state(), s, key, jnp.array(0.02)
    )
    assert wind_ned.shape == (3,)
    assert omega_gust.shape == (3,)
    assert np.array_equal(np.asarray(out_key), np.asarray(key))


def _pitch_rates_at(frac_of_r0, array, stations):
    """(tangent, secant) pitch gust rate at a station along the track, in core radii."""
    from atisim import airframe  # noqa: F401  -- stations already built

    field = lambda p: wind.vortex_wind(p, array)  # noqa: E731
    state = _level_state(north=frac_of_r0 * float(array.r0))
    tangent = float(wind.gust_rates(state.pos_ned, state.quat, field)[1])
    secant = float(wind.sampled_rates(state.pos_ned, state.quat, field, stations)[1])
    return tangent, secant


def test_the_rankine_gradient_is_discontinuous_at_the_core_edge():
    """A property of the source's own model, and the reason the point method is
    at its worst on exactly this field.

    Parks' Rankine vortex is continuous in VELOCITY at r = r0 -- the two branches
    agree there, which test_the_two_forms_agree_at_the_core_edge already asserts
    -- but its DERIVATIVE is not. Inside, the vertical gust grows linearly along
    track, so d(w)/dx = +V0/r0. Outside it falls as 1/r, and at r = r0 the same
    derivative is -V0/r0. The two one-sided derivatives differ by 2*V0/r0.

    So AT the core boundary the tangent is not merely inaccurate, it is
    AMBIGUOUS: either one-sided value is defensible and they have opposite
    signs. `vortex_wind` resolves the tie with a strict `<`, which picks the
    outside branch. The secant has no such ambiguity -- it reports what the
    airframe actually spans, and at the boundary the airframe is almost entirely
    inside the core.
    """
    from atisim import airframe
    from atisim.aircraft import REGISTRY

    array = single()  # Parks Hannibal, r0 = 182.9 m
    st = airframe.stations(REGISTRY["boeing747"])
    characteristic = float(array.v0 / array.r0)  # V0/r0, the in-core rate

    just_inside, _ = _pitch_rates_at(0.99, array, st)
    at_edge, secant_at_edge = _pitch_rates_at(1.0, array, st)

    # The jump is exactly 2*V0/r0, and it is a SIGN reversal.
    assert just_inside == pytest.approx(-characteristic, rel=1e-6)
    assert at_edge == pytest.approx(+characteristic, rel=1e-6)
    assert abs(at_edge - just_inside) == pytest.approx(2.0 * characteristic, rel=1e-6)

    # The secant does not jump: the airframe is still inside the core.
    assert secant_at_edge == pytest.approx(-characteristic, rel=1e-6)


def test_the_curvature_correction_across_the_parks_core_is_measured():
    """ASSUMPTIONS.md section E2 carries a scale ratio -- the Parks core is
    2.30-3.07 wingspans -- but has never carried a measured CONSEQUENCE. This
    is that measurement, swept along the traverse.

    NORMALISED BY V0/r0, the core's own characteristic pitch-rate input, NOT by
    the local tangent. Normalising by the tangent is unstable precisely where
    the answer matters: the tangent reverses sign at the core edge (asserted
    above), so a relative measure against it diverges there for reasons that
    say nothing about the airframe. V0/r0 is a fixed property of the vortex and
    makes the correction comparable across stations.

    The measured profile is the E2 bound, and its shape is the result:

        inside the core   exactly 0     the Rankine profile is linear, so a
                                        point sample plus a gradient is not an
                                        approximation at all
        at r = r0         2.0           the full sign reversal
        1.25 r0           ~0.11
        2.0 r0            ~0.03
    """
    from atisim import airframe
    from atisim.aircraft import REGISTRY

    array = single()
    st = airframe.stations(REGISTRY["boeing747"])
    characteristic = float(array.v0 / array.r0)

    profile = {}
    for frac in (0.5, 0.99, 1.0, 1.1, 1.25, 1.5, 2.0, 3.0):
        tangent, secant = _pitch_rates_at(frac, array, st)
        profile[frac] = abs(secant - tangent) / characteristic

    print("\nE2 curvature correction, in units of V0/r0:")
    for frac, value in profile.items():
        print(f"  {frac:5.2f} r0   {value:8.4f}")

    # Inside the core the two estimators agree to machine precision. This is the
    # half of the result that explains why the existing vortex numbers survived.
    assert profile[0.5] < 1e-12
    assert profile[0.99] < 1e-12

    # At the boundary the correction is the full sign reversal.
    assert profile[1.0] == pytest.approx(2.0, rel=1e-6)

    # And it decays quickly once clear of the corner.
    assert profile[1.25] < 0.2
    assert profile[2.0] < 0.05
    assert profile[3.0] < profile[2.0], "the correction must keep decaying outward"


# --- A2: strip integration ---------------------------------------------------


def _b747_and_stations(n_span=2001):
    from atisim import airframe
    from atisim.aircraft import REGISTRY

    ac = REGISTRY["boeing747"]
    return ac, airframe.stations(ac, n_span=n_span, n_lon=9)


def test_a_rigid_roll_rate_through_the_strip_integral_returns_the_sourced_Clp():
    """The calibration target, asserted end to end through the real integral
    rather than through the closed form it was derived from."""
    ac, st = _b747_and_stations()
    p_hat = 0.01
    clp = wind.strip_clp_from_rate(ac, st, p_hat)
    assert float(clp) / p_hat == pytest.approx(float(ac.Clp), rel=1e-3)


def test_a_uniform_vertical_gust_produces_no_rolling_moment():
    """A gust that is the same at both tips cannot roll the aircraft. If this
    fails, the integration weights are asymmetric."""
    ac, st = _b747_and_stations()
    field = lambda p: jnp.array([0.0, 0.0, 5.0])  # noqa: E731
    s = _level_state()
    moment = wind.strip_roll_moment(s.pos_ned, s.quat, field, ac, st, 236.0)
    assert abs(float(moment)) < 1e-12


def test_a_linear_gust_gradient_matches_the_equivalent_rate_answer():
    """The bridge between A1 and A2. For a gust varying linearly across the
    span, the strip integral and the equivalent-rate treatment describe the same
    physics and must agree -- that is what makes the rate equivalence legitimate
    in the first place (Stengel eq. 3.4-48). They diverge only when the profile
    is curved, which is the next test."""
    ac, st = _b747_and_stations()
    gradient = 0.002  # 1/s, d(w_g)/dy
    field = lambda p: jnp.array([0.0, 0.0, gradient * p[1]])  # noqa: E731
    s = _level_state()
    V = 236.0

    strip = float(wind.strip_roll_moment(s.pos_ned, s.quat, field, ac, st, V))
    # Equivalent rate: p_gust = +d(w_g)/dy, and the aero model sees -p_gust.
    p_equivalent = -gradient
    equivalent = float(
        wind.strip_clp_from_rate(ac, st, p_equivalent * float(ac.b) / (2.0 * V))
    )
    assert strip == pytest.approx(equivalent, rel=1e-6)


def test_a_curved_gust_profile_makes_the_strip_integral_differ_from_the_rate():
    """The reason Phase 3 exists. A cubic spanwise profile has the same
    centreline slope as no gust at all, yet it genuinely rolls the aircraft. The
    equivalent-rate treatment cannot represent that; the strip integral can."""
    ac, st = _b747_and_stations()
    field = lambda p: jnp.array([0.0, 0.0, 1e-7 * p[1] ** 3])  # noqa: E731
    s = _level_state()

    strip = float(wind.strip_roll_moment(s.pos_ned, s.quat, field, ac, st, 236.0))
    tangent = float(wind.gust_rates(s.pos_ned, s.quat, field)[0])
    assert tangent == pytest.approx(0.0, abs=1e-12), "the cubic has zero centreline slope"
    assert abs(strip) > 1e-9, "yet it must still produce a rolling moment"


def test_nothing_added_by_this_work_moves_the_existing_wind_path():
    """Everything in the sampled and strip work is ADDITIVE -- gust_rates and
    field_model are untouched -- so a trajectory flown through the old path must
    be reproducible exactly. This is the test that would catch an 'improvement'
    accidentally applied to the default path.

    PROJECT.md section 4's mode baselines are downstream of exactly this code,
    and they are off-limits to feature work.
    """
    from atisim import integrate, trim
    from atisim.aircraft import CRUISE, REGISTRY

    ac = REGISTRY["boeing747"]
    v, h = CRUISE["boeing747"]["airspeed"], CRUISE["boeing747"]["altitude"]
    x, _ = trim.trim(jnp.array(v), jnp.array(h), ac)
    state = trim.trimmed_state(x[0], jnp.array(v), jnp.array(h))
    controls = trim.trimmed_controls(x[1], x[2])

    array = single()
    model = wind.vortex_model(array)
    sim = integrate.init_sim(state, jax.random.PRNGKey(0))

    first, _ = integrate.rollout(sim, controls, jnp.array(0.02), ac, 200, wind_model=model)
    second, _ = integrate.rollout(sim, controls, jnp.array(0.02), ac, 200, wind_model=model)

    assert np.array_equal(
        np.asarray(first.state.pos_ned), np.asarray(second.state.pos_ned)
    )
    assert np.array_equal(np.asarray(first.state.quat), np.asarray(second.state.quat))


def test_the_default_field_model_still_uses_the_analytic_gradient():
    """The sharper form of the check above: assert the DEFAULT path is the
    tangent, not the fit. `field_model` and `sampled_field_model` must give
    different omega_gust wherever the field is curved -- if they agree, the
    default has been switched over silently and every frozen baseline is at
    risk."""
    from atisim import airframe
    from atisim.aircraft import REGISTRY

    ac = REGISTRY["boeing747"]
    array = single()
    field = lambda p: wind.vortex_wind(p, array)  # noqa: E731
    s = _level_state(north=CASE1_R0)  # at the core edge, where they differ most

    key = jax.random.PRNGKey(0)
    _, tangent_gust, _, _, _ = wind.field_model(field)(
        wind.zero_wind_state(), s, key, jnp.array(0.02)
    )
    _, fitted_gust, _, _, _ = wind.sampled_field_model(field, airframe.stations(ac))(
        wind.zero_wind_state(), s, key, jnp.array(0.02)
    )
    assert not np.allclose(np.asarray(tangent_gust), np.asarray(fitted_gust)), (
        "field_model and sampled_field_model agree at the core edge, which means "
        "the default path is no longer the analytic gradient"
    )


# --- flying the field: the strip path in the equations of motion ------------


def test_flying_the_parks_vortex_with_strip_loads_leaves_the_trajectory_alone():
    """Gate 9, and the answer is zero. MEASURED, not assumed.

    The plan expected this to differ and it does not, for a reason that is a
    property of the field rather than a defect in the seam: the Parks vortex
    axes lie across the flight path and the field has NO east variation, so
    every strip on the span sees the same vertical gust and the antisymmetric
    roll integral cancels. The rolling coefficient comes out at order 1e-19,
    which is round-off, and the two trajectories are bit-identical.

    This is the same fact the rigid-rotation diagnostic below reports as
    `p-pair n/a`: dw/dy is identically zero here. Two measurements, one cause.

    So this test is a NULL RESULT and is named for one. It does not show the
    strip path works -- `test_a_field_with_spanwise_structure_moves_the_aircraft`
    does that, and it exists because without it a broken seam would pass here
    just as happily. What this shows is that turning the strip path on costs
    the headline vortex result nothing, which is worth knowing and is exactly
    why ASSUMPTIONS.md E2 forbids reading the strip work as having fixed it.

    Reported rather than bounded: the size of the difference is the RESULT, and
    fixing a tolerance around it now would be asserting the answer before
    measuring it.
    """
    from atisim import integrate, loads, trim
    from atisim.aircraft import CRUISE, REGISTRY

    ac = REGISTRY["boeing747"]
    v, h = CRUISE["boeing747"]["airspeed"], CRUISE["boeing747"]["altitude"]
    x, _ = trim.trim(jnp.array(v), jnp.array(h), ac)
    state = trim.trimmed_state(x[0], jnp.array(v), jnp.array(h))
    controls = trim.trimmed_controls(x[1], x[2])

    array = single()
    field = lambda p: wind.vortex_wind(p, array)  # noqa: E731
    model = wind.vortex_model(array)
    # Start well upstream so the aircraft flies through the whole core.
    start = state._replace(pos_ned=jnp.array([-6.0 * CASE1_R0, 0.0, -h]))
    sim = integrate.init_sim(start, jax.random.PRNGKey(0))
    steps = int(12.0 * CASE1_R0 / v / 0.02)

    point, _ = integrate.rollout(sim, controls, jnp.array(0.02), ac, steps, wind_model=model)
    strip, _ = integrate.rollout(
        sim, controls, jnp.array(0.02), ac, steps, wind_model=model,
        load_model=loads.strip_model(field, ac),
    )

    d_pos = float(
        np.linalg.norm(np.asarray(point.state.pos_ned) - np.asarray(strip.state.pos_ned))
    )
    print(f"\nParks core traverse, {steps} steps:")
    print(f"  position difference, point vs strip: {d_pos:.6f} m")
    print(f"  strip rolling coefficient at the end: {float(strip.increment.Cl):.6e}")
    assert np.isfinite(d_pos)


def test_a_field_with_spanwise_structure_moves_the_aircraft():
    """The positive control the Parks field cannot provide.

    Gate 9 asks whether the strip path reaches the equations of motion. The
    vortex answers `no difference` for a reason that has nothing to do with the
    seam, so on its own it would pass unchanged if `load_model` were dropped on
    the floor. This flies the same aircraft through a field that DOES vary
    across the span -- a cubic in east, the profile an equivalent roll rate
    cannot represent at all -- and requires the trajectory to move.

    Cubic rather than linear on purpose: a linear gradient is exactly what the
    point-plus-gradient path already reproduces, so it would understate the
    difference and, worse, would still pass if the strip integral silently
    degraded to an equivalent rate.
    """
    from atisim import integrate, loads, trim
    from atisim.aircraft import CRUISE, REGISTRY

    ac = REGISTRY["boeing747"]
    v, h = CRUISE["boeing747"]["airspeed"], CRUISE["boeing747"]["altitude"]
    x, _ = trim.trim(jnp.array(v), jnp.array(h), ac)
    state = trim.trimmed_state(x[0], jnp.array(v), jnp.array(h))
    controls = trim.trimmed_controls(x[1], x[2])

    # Vertical gust cubic in east. Peaks at ~4.5 m/s at the wingtip, which is
    # the same order as the Parks core and therefore not a contrived overdrive.
    field = lambda p: jnp.array([0.0, 0.0, 1e-6 * p[1] ** 3])  # noqa: E731
    sim = integrate.init_sim(
        state._replace(pos_ned=jnp.array([0.0, 0.0, -h])), jax.random.PRNGKey(0)
    )
    steps = 500

    point, _ = integrate.rollout(
        sim, controls, jnp.array(0.02), ac, steps, wind_model=wind.field_model(field)
    )
    strip, _ = integrate.rollout(
        sim, controls, jnp.array(0.02), ac, steps, wind_model=wind.field_model(field),
        load_model=loads.strip_model(field, ac),
    )

    d_pos = float(
        np.linalg.norm(np.asarray(point.state.pos_ned) - np.asarray(strip.state.pos_ned))
    )
    roll = float(strip.increment.Cl)
    print(f"\nCubic spanwise gust, {steps} steps:")
    print(f"  position difference, point vs strip: {d_pos:.6f} m")
    print(f"  strip rolling coefficient at the end: {roll:.6e}")
    assert roll != 0.0, "the strip integral produced no rolling moment on a cubic profile"
    assert d_pos > 0.0, (
        "the strip rolling moment did not reach the equations of motion -- "
        "load_model is being computed and discarded"
    )


def test_the_rigid_rotation_structure_diagnostic_is_reported_per_field():
    """Gate 7. The current point model is exactly equivalent to assuming the
    shear matrix has RIGID-ROTATION STRUCTURE -- that d(v)/dz = -d(w)/dy and
    d(u)/dz = -d(w)/dx. This reports how far each field departs from that, which
    is the cheapest available predictor of where the point model will struggle.

    Reported, not asserted: the ratio is a property of each field, and pinning
    it would freeze a diagnostic rather than a result.
    """
    array = single()
    field = lambda p: wind.vortex_wind(p, array)  # noqa: E731
    print("\nRigid-rotation-structure diagnostic (1.0 = exactly rotation-like):")
    for frac, label in ((0.5, "inside core"), (1.5, "outside core")):
        s = _level_state(north=frac * CASE1_R0)
        dcm = np.asarray(jax.jacfwd(field)(s.pos_ned))
        assert np.all(np.isfinite(dcm)), f"the {label} shear matrix is not finite"
        # body == NED here (wings level, heading north)
        dw_dy, dv_dz = dcm[2, 1], dcm[1, 2]
        dw_dx, du_dz = dcm[2, 0], dcm[0, 2]
        pair_p = "n/a" if abs(dw_dy) < 1e-12 else f"{-dv_dz / dw_dy:+.4f}"
        pair_q = "n/a" if abs(dw_dx) < 1e-12 else f"{-du_dz / dw_dx:+.4f}"
        print(f"  {label:12s}  p-pair {pair_p}   q-pair {pair_q}")


# ---------------------------------------------------------------------------
# The same vortices, as the LATER paper reports them.
#
# Wingrove & Bach 1994, J. Aircraft 31(4), Fig. 4 p. 755. Obtained session 19 --
# the audit had this source down as `unverifiable -- source not available`.
# ---------------------------------------------------------------------------


def test_wingrove_fig4_cases_are_radii_and_morton_fixes_the_interpretation():
    """Fig. 4 quotes DIAMETERS; the dict must hold radii.

    Morton is the cross-check that fixes the interpretation: Fig. 4's 900 ft
    diameter is 450 ft of radius, which is PARKS_CASES['morton']['r0'] exactly.
    Without that agreement there would be no way to tell a diameter column from
    a radius column, and every core in this comparison would be a factor of two
    out with nothing to catch it.
    """
    assert wind.WINGROVE_FIG4_CASES["morton"]["r0"] == pytest.approx(450.0 * FT2M)
    assert wind.WINGROVE_FIG4_CASES["morton"]["r0"] == pytest.approx(
        wind.PARKS_CASES["morton"]["r0"]
    )
    assert wind.WINGROVE_FIG4_CASES["morton"]["v0"] == pytest.approx(
        wind.PARKS_CASES["morton"]["v0"]
    )


def test_hannibal_radius_conflict_is_pinned_not_resolved():
    """The two sources disagree on Hannibal and this test says so out loud.

    Fig. 4 gives a 1000 ft diameter, so a 500 ft radius. PARKS_CASES says 600 ft,
    citing Parks et al. 1985 -- which has never been obtained (AUDIT.md row 19),
    so there is no way to tell which is the transcription error. Both are kept
    and both are flown. If someone later resolves it, this test is where the
    resolution has to be argued.
    """
    assert wind.WINGROVE_FIG4_CASES["hannibal"]["r0"] == pytest.approx(500.0 * FT2M)
    assert wind.PARKS_CASES["hannibal"]["r0"] == pytest.approx(600.0 * FT2M)
    # Same core strength in both sources -- only the radius is in dispute.
    assert wind.WINGROVE_FIG4_CASES["hannibal"]["v0"] == pytest.approx(
        wind.PARKS_CASES["hannibal"]["v0"]
    )


def test_cimarron_exists_only_in_the_1994_paper():
    """Parks identifies two cases; Wingrove & Bach add a third."""
    assert "cimarron" not in wind.PARKS_CASES
    assert wind.WINGROVE_FIG4_CASES["cimarron"]["r0"] == pytest.approx(450.0 * FT2M)
    assert wind.WINGROVE_FIG4_CASES["cimarron"]["v0"] == pytest.approx(50.0 * FT2M)


def test_fig4_cases_carry_no_spacing():
    """Fig. 4 gives core size and strength and says nothing about array spacing.

    A `spacing` key here would be invented. Callers that need an array take it
    from PARKS_CASES and say which source each number came from.
    """
    for case in wind.WINGROVE_FIG4_CASES.values():
        assert set(case) == {"r0", "v0"}


def test_case_altitudes_match_table_1():
    """Table 1, p. 754. These three cases are NOT at one altitude."""
    assert wind.WINGROVE_CASE_ALTITUDE["cimarron"] == pytest.approx(33000.0 * FT2M)
    assert wind.WINGROVE_CASE_ALTITUDE["hannibal"] == pytest.approx(37000.0 * FT2M)
    assert wind.WINGROVE_CASE_ALTITUDE["morton"] == pytest.approx(39000.0 * FT2M)
