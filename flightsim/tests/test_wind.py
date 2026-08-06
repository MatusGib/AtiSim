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

from flightsim import wind
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
    Measured here: 1.89 deg over the first core, against Fig. 8's 1.4 deg
    extreme for the vortex category and 6.2 deg / 12 deg for the updraft and
    manoeuvring categories -- the right cluster by a wide margin.

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
    state = trim.trimmed_state(jnp.array(alpha), jnp.array(v), jnp.array(h))
    state = state._replace(pos_ned=jnp.array([-6.0 * CASE1_R0, 0.0, -h]))

    dt = 0.01
    n = int(round((CASE1_SPACING + 12.0 * CASE1_R0) / v / dt))
    _, hist = integrate.rollout(
        integrate.init_sim(state, jax.random.PRNGKey(0)), controls,
        jnp.array(dt), ac, n, wind_model=wind.vortex_model(array),
    )
    theta = np.asarray(jax.vmap(quat_to_euler)(hist.quat))[:, 1]
    north = np.asarray(hist.pos_ned)[:, 0]

    # kinematics: gust spacing is airframe-independent
    assert CASE1_SPACING / v == pytest.approx(4.52, abs=0.05)

    in_first_core = np.abs(north - 0.0) <= CASE1_R0
    dtheta = (theta[in_first_core].max() - theta[in_first_core].min()) * RAD2DEG
    assert 0.5 < dtheta < 3.0, dtheta  # measured 1.89 deg; Fig. 8 vortex ~1.4

    # and the encounter must be far smaller in pitch than the post-encounter
    # phugoid, which is the windowing trap this test exists to pin down
    whole = (theta.max() - theta.min()) * RAD2DEG
    assert whole > 3.0 * dtheta  # measured 8.79 vs 1.89
