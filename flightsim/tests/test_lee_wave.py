"""Mountain lee wave, and the windshear hazard index it is measured with.

Two sources, and they do different jobs.

The FIELD is cited to J. D. Doyle, Q. Jiang, R. B. Smith, V. Grubisic,
"Three-Dimensional Characteristics of Stratospheric Mountain Waves during
T-REX", Mon. Wea. Rev. 139 (Jan 2011), 3-23, DOI 10.1175/2010MWR3466.1. That
paper is used rather than a textbook because its measurements are at the right
ALTITUDE: the NSF/NCAR Gulfstream V flew legs at 11.3 and 13.1 km over the
Sierra Nevada, and this project's 747 cruises at 12.192 km, between them.

The INDEX is cited to F. H. Proctor, D. A. Hinton (NASA Langley) and R. L.
Bowles (AeroTech), "A Windshear Hazard Index", 9th Conf. on Aviation, Range and
Aerospace Meteorology, Orlando, Sept 2000, paper 7.7, pp. 482-487. Its Eq. (3)
defines the F-factor; the hazard criterion used here is that paper's own
statement that a shear exceeding (T_r - D)/W cannot be countered by thrust.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import flightsim  # noqa: F401  -- enables x64
from flightsim import dynamics, trim, wind
from flightsim.aircraft import CRUISE, REGISTRY

AC = REGISTRY["boeing747"]
V = CRUISE["boeing747"]["airspeed"]
H = CRUISE["boeing747"]["altitude"]


def a_wave(w0=6.0, wavelength=25_000.0, north=0.0):
    return wind.LeeWave(
        w0=jnp.array(w0), wavelength=jnp.array(wavelength), north=jnp.array(north)
    )


# --- the field, against the source's own description -------------------------


def test_the_trough_is_a_downdraft_of_the_stated_amplitude():
    """`north` marks the TROUGH, because the downdraft is the hazardous half."""
    w = a_wave(w0=6.0, north=0.0)
    at_trough = wind.lee_wave_wind(jnp.array([0.0, 0.0, -H]), w)
    # NED z is DOWN, so a downdraft is POSITIVE z.
    assert float(at_trough[2]) == pytest.approx(6.0)
    assert float(at_trough[0]) == 0.0
    assert float(at_trough[1]) == 0.0


def test_the_wave_is_periodic_at_its_stated_wavelength():
    """A lee wave is a downstream TRAIN; one wavelength on must repeat."""
    w = a_wave(wavelength=25_000.0)
    here = wind.lee_wave_wind(jnp.array([1234.0, 0.0, -H]), w)
    one_on = wind.lee_wave_wind(jnp.array([1234.0 + 25_000.0, 0.0, -H]), w)
    half_on = wind.lee_wave_wind(jnp.array([1234.0 + 12_500.0, 0.0, -H]), w)
    assert float(one_on[2]) == pytest.approx(float(here[2]))
    assert float(half_on[2]) == pytest.approx(-float(here[2]))  # crest opposes trough


def test_the_vertical_velocity_never_exceeds_the_stated_amplitude():
    north = np.linspace(-60_000.0, 60_000.0, 2001)
    w_up = np.array([
        -float(wind.lee_wave_wind(jnp.array([n, 0.0, -H]), a_wave(w0=6.0))[2])
        for n in north
    ])
    assert np.abs(w_up).max() == pytest.approx(6.0, abs=1e-6)


def test_the_field_is_divergence_free():
    """Continuity, and the reason this model carries no horizontal component.

    A purely vertical field with no vertical variation has zero divergence
    identically, which is what makes it an admissible incompressible flow. It is
    also what the DECLARED simplification costs: a real lee wave has a
    horizontal perturbation in quadrature with the vertical one, and reproducing
    it needs a stratification and an ambient wind speed no source here supplies.
    """
    w = a_wave()

    def divergence(p):
        return jnp.trace(jax.jacobian(lambda q: wind.lee_wave_wind(q, w))(p))

    for n in (-8000.0, 0.0, 3210.0, 12_500.0):
        assert float(divergence(jnp.array([n, 0.0, -H]))) == pytest.approx(0.0, abs=1e-12)


def test_the_lee_wave_satisfies_the_wind_model_contract():
    """Same contract every field obeys, so it composes with the others."""
    model = wind.lee_wave_model(a_wave())
    state = trim.trimmed_state(jnp.array(0.05), jnp.array(V), jnp.array(H))
    key = jax.random.PRNGKey(0)
    wind_ned, omega_gust, wind_state, out_key = model(
        wind.zero_wind_state(), state, key, jnp.array(0.01)
    )
    assert wind_ned.shape == (3,)
    assert omega_gust.shape == (3,)
    assert np.array_equal(np.asarray(out_key), np.asarray(key))  # deterministic


# --- the F-factor, against Proctor et al. Eq. (3) ----------------------------


def test_the_f_factor_is_zero_in_still_air():
    assert float(dynamics.f_factor(jnp.array(0.0), jnp.array(0.0), jnp.array(V))) == 0.0


def test_a_downdraft_gives_a_positive_f_factor_and_an_updraft_a_negative_one():
    """Eq. (3)'s sign: "the F-factor is positive for a descending air mass"."""
    down = dynamics.f_factor(jnp.array(0.0), jnp.array(-6.0), jnp.array(V))
    up = dynamics.f_factor(jnp.array(0.0), jnp.array(+6.0), jnp.array(V))
    assert float(down) == pytest.approx(6.0 / V)
    assert float(up) == pytest.approx(-6.0 / V)


def test_an_accelerating_tailwind_gives_a_positive_f_factor():
    """Eq. (3)'s other half: performance-decreasing shear is U_x_dot > 0."""
    f = dynamics.f_factor(jnp.array(0.5), jnp.array(0.0), jnp.array(V))
    assert float(f) == pytest.approx(0.5 / 9.80665)


# --- the averaged index, Proctor et al. Eq. (7) ------------------------------


def test_a_constant_f_averages_to_itself():
    distance = jnp.linspace(0.0, 5000.0, 501)
    f = jnp.full(501, 0.08)
    average, valid = dynamics.average_f_factor(f, distance)
    assert float(jnp.max(jnp.abs(average[valid] - 0.08))) < 1e-9


def test_averaging_flattens_a_spike_the_aircraft_flies_straight_through():
    """The paper's own reason for averaging, asserted.

    A 100 m spike of F = 0.5 is a tenth of the 1 km window, so it must average
    to about 0.05 -- below the jet-transport hazard threshold -- while the
    instantaneous peak is five times over it. Reporting the instantaneous value
    would call a bump an accident.
    """
    distance = jnp.linspace(0.0, 6000.0, 6001)
    f = jnp.where((distance >= 2000.0) & (distance < 2100.0), 0.5, 0.0)
    average, valid = dynamics.average_f_factor(f, distance)
    assert float(jnp.max(f)) == 0.5
    assert float(jnp.max(average[valid])) == pytest.approx(0.05, abs=0.002)


def test_a_long_wavelength_survives_the_average_almost_intact():
    """Why the lee-wave result is unchanged by using the correct metric.

    The window is 1 km and the wave is 25 km, so the average over the trough is
    sin(x)/x with x = pi/25 -- 99.7% of the peak. The averaging that guts a
    spike barely touches a wave the aircraft is inside for half a minute.
    """
    distance = jnp.linspace(0.0, 100_000.0, 20_001)
    f = 0.026 * jnp.cos(2.0 * jnp.pi * distance / wind.LEE_WAVE_WAVELENGTH)
    average, valid = dynamics.average_f_factor(f, distance)
    assert float(jnp.max(average[valid])) == pytest.approx(0.026 * 0.9974, rel=0.002)


# --- the result section 7 step 8 asks for ------------------------------------


def test_the_thrust_envelope_is_what_session_2_recorded():
    """Recomputed, not taken on trust: section 4 carries +0.023/-0.066."""
    x, _ = trim.trim(jnp.array(V), jnp.array(H), AC)
    full, idle = dynamics.thrust_authority(AC, x[2], jnp.array(H))
    assert float(full) == pytest.approx(0.023, abs=0.001)
    assert float(idle) == pytest.approx(-0.066, abs=0.001)


def test_the_stronger_doyle_lee_wave_exceeds_the_747s_thrust_authority():
    """Section 7 step 8's verification, and the whole point of the exercise.

    Doyle et al.'s IOP 4 primary wave is 12 m/s crest-to-trough on the southern
    leg and 6 m/s on the northern, i.e. amplitudes of 6.0 and 3.0 m/s. The
    stronger one produces an F the 747 cannot counter at cruise; the weaker one
    it can. The hazard threshold sits INSIDE the observed range, which is a
    sharper statement than "always" or "never" and is what the numbers say.
    """
    x, _ = trim.trim(jnp.array(V), jnp.array(H), AC)
    full, _ = dynamics.thrust_authority(AC, x[2], jnp.array(H))

    strong = wind.LEE_WAVE_AMPLITUDE["south"] / V  # F at the trough, no shear term
    weak = wind.LEE_WAVE_AMPLITUDE["north"] / V

    assert strong > float(full)
    assert weak < float(full)
