"""What the flown response spectra and exceedance curves actually say.

`test_response.py` checks the estimators against signals whose answer is known
in closed form. This file checks the AIRCRAFT, and every number here was
measured by `scripts/cat_spectra.py` before it was written down.

Sources referred to by tag:

  Y23      R. Yoshimura, J. Ito, P. A. Schittenhelm, K. Suzuki, A. Yakeno,
           S. Obayashi, "Clear Air Turbulence Resolved by Numerical Weather
           Prediction Model Validated by Onboard and Virtual Flight Data",
           Geophys. Res. Lett. 50, e2022GL101286 (2023). Fig. 6 and section 3.2
           are the protocol: N virtual flights, spectra averaged, peak compared
           against the aircraft's own natural frequency.

  MEHTA    R. S. Mehta, "Modeling Clear Air Turbulence with Vortices Using
           Parameter Identification Techniques", J. Guidance 10(1) 27-31 (1987).

THE ENSEMBLES HERE ARE SMALL ON PURPOSE -- four flights, not the script's
thirty-two -- so the suite stays runnable. Every assertion is therefore chosen
to be one that four flights can carry; the statistics that need thirty-two are
in the script's own output and in PROJECT.md section 4, with their N attached.
"""

import numpy as np
import pytest

import jax.numpy as jnp

from atisim import predictions, response, trim, validation, vortex_viz, wind
from atisim.aircraft import CRUISE, REGISTRY

DT = 0.02
SETTLE = 20.0     # s discarded from the head of every Dryden record
RECORD = 100.0    # s of record kept -- Y23's own length
SEEDS = 4
MEHTA_LEAD_R0 = 12.0


@pytest.fixture(scope="module")
def condition():
    ac = REGISTRY["boeing747"]
    V = CRUISE["boeing747"]["airspeed"]
    H = wind.MEHTA_HANNIBAL_ALTITUDE
    x, _ = trim.trim(jnp.array(V), jnp.array(H), ac)
    wn, zeta = validation.longitudinal_modes(
        ac, float(x[0]), float(x[1]), float(x[2]), V, H)[-1]
    return dict(ac=ac, V=V, H=H, f_sp=float(wn) / (2.0 * np.pi),
                zeta=float(zeta))


def _dryden_record(cond, sigma, seed):
    enc = vortex_viz.fly_in_moving_air(
        cond["ac"], wind.dryden_field(sigma, seed), cond["V"], cond["H"],
        label=f"dryden-{seed}", start_north=0.0, seconds=SETTLE + RECORD,
        dt=DT, window=(-1e12, 1e12), window_name="whole run",
    )
    return np.asarray(enc.n_z)[int(round(SETTLE / DT)):]


@pytest.fixture(scope="module")
def ensemble(condition):
    sigma = wind.mehta_unmodelled_wind()
    records = [_dryden_record(condition, sigma, s) for s in range(SEEDS)]
    spectra = [response.spectrum(r, DT) for r in records]
    return dict(sigma=sigma, records=records, f=spectra[0][0],
                psd=np.mean([s[1] for s in spectra], axis=0))


def test_the_short_period_is_the_frequency_the_prediction_was_sealed_against():
    """The one number the sealed claim rests on, pinned so it cannot drift
    underneath a settled prediction. 0.1640 Hz is 1.0307 rad/s -- and note it
    is NOT Yoshimura 2022's 1.29 rad/s, which is a different flight condition;
    PROJECT.md section 5 records that paper's own unit error separately."""
    ac = REGISTRY["boeing747"]
    V, H = CRUISE["boeing747"]["airspeed"], wind.MEHTA_HANNIBAL_ALTITUDE
    x, _ = trim.trim(jnp.array(V), jnp.array(H), ac)
    wn, zeta = validation.longitudinal_modes(
        ac, float(x[0]), float(x[1]), float(x[2]), V, H)[-1]
    assert float(wn) / (2.0 * np.pi) == pytest.approx(0.16404, rel=1e-3)
    assert float(zeta) == pytest.approx(0.36455, rel=1e-3)


def test_the_mehta_response_follows_the_airframe_and_not_the_forcing(condition):
    """MEHTA's array is not uniform: its four core spacings force the aircraft
    at 0.091, 0.122, 0.135 and 0.189 Hz at once, averaging 0.134. The aircraft
    has one short period, 0.164 Hz. The response peak sits within a quarter of a
    bin of the airframe and more than a bin and a half from the mean forcing.

    This is the first statement in this project about WHICH frequency the load
    followed, as opposed to how large it got.
    """
    ac, V, H = condition["ac"], condition["V"], condition["H"]
    array = wind.mehta_hannibal_array(H)
    field = lambda p: wind.vortex_wind(p, array)  # noqa: E731
    r0 = float(array.r0)
    x0, x1 = float(array.north.min()), float(array.north.max())
    start = x0 - MEHTA_LEAD_R0 * r0
    enc = vortex_viz.fly_in_moving_air(
        ac, field, V, H, label="mehta", start_north=start,
        seconds=(x1 + MEHTA_LEAD_R0 * r0 - start) / V, dt=DT,
        window=(x0 - 2.0 * r0, x1 + 2.0 * r0), window_name="array",
    )
    f, psd = response.spectrum(np.asarray(enc.n_z), DT)
    peak = response.peak_frequency(f, psd)
    bin_hz = f[1] - f[0]
    passage = (V / np.diff(np.asarray(array.north))).mean()

    assert peak == pytest.approx(0.16885, rel=1e-3)
    assert abs(peak - condition["f_sp"]) < bin_hz          # the airframe
    assert abs(peak - passage) > 1.5 * bin_hz              # not the forcing
    # The companion that must be false: the forcing is not merely far, it is
    # far in a direction. If this ever passes, the reading above is backwards.
    assert not abs(peak - passage) < abs(peak - condition["f_sp"])


def test_the_dryden_ensemble_peak_lands_inside_the_sealed_band(ensemble,
                                                               condition):
    """Y23's protocol at four flights instead of 151, and the band is the one
    `predictions.the_dryden_response_peaks_at_the_short_period` was sealed
    against at commit 2837ddd -- before this file existed.

    The band is read off the prediction's own text rather than retyped, so a
    re-seal that moved it would fail here rather than pass quietly.
    """
    peak = response.peak_frequency(ensemble["f"], ensemble["psd"])
    claim = predictions.BY_NAME[
        "the_dryden_response_peaks_at_the_short_period"].claim
    assert "between 0.131 and 0.197 Hz" in claim
    assert 0.131 <= peak <= 0.197
    assert peak == pytest.approx(0.1400, abs=0.011)  # one bin of a 100 s record


def test_the_response_has_a_resonant_hump_the_input_does_not(ensemble,
                                                             condition):
    """The coupling, stated as a ratio rather than as a peak location.

    `wind.dryden_spectrum` is flat below Omega = 1/L_w and falls as Omega^-2
    above, so the INPUT has strictly more energy at 0.05 Hz than at the short
    period. The RESPONSE has more at the short period. That reversal is the
    whole content of "the airframe organises the load", and it is a comparison
    of two numbers rather than an argmax, so it does not depend on N.
    """
    f, psd = ensemble["f"], ensemble["psd"]
    V, L = condition["V"], wind.DRYDEN_LW
    at = lambda hz: float(psd[np.argmin(np.abs(f - hz))])  # noqa: E731
    phi = lambda hz: float(  # noqa: E731
        wind.dryden_spectrum(jnp.array(2.0 * np.pi * hz / V), 1.0, L))

    assert phi(0.05) > phi(condition["f_sp"])       # input falls
    assert at(condition["f_sp"]) > at(0.05)         # response rises
    assert at(condition["f_sp"]) / at(0.05) > 3.0


def test_the_exceedance_curve_falls_and_the_two_signs_agree(ensemble):
    """A rate, with N in the denominator, and its own negative control.

    Section 5 records that `CL = CL0 + CLa*alpha` is exactly odd-symmetric in
    the gust, so up and down exceedances must agree to within sampling error --
    asserted only at the levels four flights actually populate, since the tail
    is where a small ensemble stops meaning anything.
    """
    levels = np.array([0.05, 0.10, 0.15])
    up = np.mean([response.exceedance(r, DT, 1.0 + levels)
                  for r in ensemble["records"]], axis=0)
    down = np.mean([response.exceedance(-r, DT, -(1.0 - levels))
                    for r in ensemble["records"]], axis=0)

    assert np.all(np.diff(up) < 0) and np.all(np.diff(down) < 0)
    assert up == pytest.approx(down, rel=0.25)
    assert up[0] == pytest.approx(0.948, rel=0.2)
    # And nothing crosses a level the ensemble never reaches.
    ceiling = max(np.abs(r - 1.0).max() for r in ensemble["records"])
    assert np.all(np.mean([response.exceedance(r, DT, [1.0 + ceiling + 0.01])
                           for r in ensemble["records"]], axis=0) == 0.0)


def test_still_air_produces_no_spectrum_and_no_exceedance(condition):
    """The negative control for the whole file.

    Fixed controls in zero wind is a trimmed aircraft doing nothing, so its
    load history is constant to machine precision and every statistic above
    must return exactly zero. Anything that reports structure here is reporting
    its own arithmetic.
    """
    enc = vortex_viz.fly_in_moving_air(
        condition["ac"], lambda p: jnp.zeros(3), condition["V"],
        condition["H"], label="still", start_north=0.0,
        seconds=SETTLE + RECORD, dt=DT, window=(-1e12, 1e12),
        window_name="whole run",
    )
    n_z = np.asarray(enc.n_z)[int(round(SETTLE / DT)):]
    assert np.ptp(n_z) == 0.0
    assert response.exceedance(n_z, DT, [1.0 + 1e-9, 1.01, 1.1]) \
        == pytest.approx(0.0)
    _, psd = response.spectrum(n_z, DT)
    # Not exactly zero, and the reason is worth stating rather than tolerating:
    # a PSD is a squared amplitude, so the floor is the SQUARE of the round-off
    # left by subtracting a 1 g mean from itself. 3.3e-30 is (1.8e-15)^2, which
    # is that floor and not a response.
    assert psd.max() < 1e-25
