"""`atisim.response` against signals whose answer is known in closed form.

WHY EVERY CHECK HERE IS ANALYTIC. A spectral estimator is exactly the kind of
code that looks right and is off by a factor of two, or by `fs`, or by the
window's mean square -- and a factor of two in a PSD is invisible to the eye on
a log plot. So nothing here compares one estimate against another; every
expected value is a number derived on the line above it.

EVERY POSITIVE ASSERTION HAS A COMPANION THAT MUST BE ZERO. That pattern is
what caught the session-24 vortex-axis bug (PROJECT.md section 9), and it is the
cheapest defence against an estimator that reports structure in anything.
"""

import numpy as np
import pytest

from atisim.response import (
    PHUGOID_FLOOR_HZ, exceedance, peak_frequency, spectrum,
)

DT = 0.02  # s, the step scripts/cat_ensemble.py flies
N = 20_000  # samples -> 400 s, long enough that a 0.16 Hz line is well resolved


def _time():
    return np.arange(N) * DT


# ---------------------------------------------------------------------------
# spectrum -- the normalisation, which is the only thing that can be wrong
# quietly
# ---------------------------------------------------------------------------


def test_the_spectrum_integrates_to_the_variance_of_a_sinusoid():
    """A sine of amplitude A has variance A^2/2, exactly.

    This is the assertion that pins the convention: one-sided, density, and
    corrected for the Hann window's mean square. Get any of those wrong and this
    is out by 2, by 8/3, or by the sample rate.
    """
    a, f0 = 0.3, 0.164
    x = 1.0 + a * np.sin(2.0 * np.pi * f0 * _time())
    f, psd = spectrum(x, DT)
    assert np.trapezoid(psd, f) == pytest.approx(a ** 2 / 2.0, rel=1e-3)


def test_the_spectrum_integrates_to_the_variance_of_white_noise():
    """The same convention on a broadband signal, where the window's leakage
    behaves differently. Looser tolerance because the variance of a periodogram
    estimate is 100% per bin -- the INTEGRAL is what converges, not the shape."""
    x = np.random.default_rng(0).normal(0.0, 2.0, N)
    f, psd = spectrum(x, DT)
    assert np.trapezoid(psd, f) == pytest.approx(x.var(), rel=0.01)


def test_the_mean_is_removed_so_a_trim_state_is_not_a_spectral_line():
    """n_z sits at 1 g; if the mean survived, its DC line would dwarf the gust
    response. The NEGATIVE control is the pair: adding any offset must leave the
    spectrum unchanged to machine precision."""
    x = 0.3 * np.sin(2.0 * np.pi * 0.164 * _time())
    _, plain = spectrum(x, DT)
    _, offset = spectrum(x + 17.0, DT)
    assert np.allclose(plain, offset, atol=1e-12, rtol=0)


def test_the_peak_lands_on_the_line_that_is_there_and_not_elsewhere():
    a, f0 = 0.3, 0.164
    f, psd = spectrum(a * np.sin(2.0 * np.pi * f0 * _time()), DT)
    bin_hz = f[1] - f[0]
    assert abs(peak_frequency(f, psd) - f0) <= bin_hz
    # The companion: away from the line there is nothing. TWO thresholds,
    # because there are two floors and quoting one would hide the other. The
    # first two bins carry the detrending residual through the Hann window's DC
    # main lobe -- 5e-5 of the peak, real and irreducible. Everywhere else it is
    # Hann's own sidelobe, 5e-6 of the peak at five bins out and falling as the
    # cube of the offset.
    far = np.abs(f - f0) > 5.0 * bin_hz
    assert psd[far][:2].max() < 1e-4 * psd.max()
    assert psd[far][2:].max() < 1e-5 * psd.max()


def test_the_default_floor_steps_over_a_phugoid_sized_line():
    """Two lines: a large one at 0.009 Hz where the 747's phugoid sits, and a
    small one at 0.164 Hz where its short period does. The peak of the whole
    spectrum is the phugoid; the peak this project asks for is not."""
    t = _time()
    x = 1.0 * np.sin(2.0 * np.pi * 0.009 * t) + 0.05 * np.sin(2.0 * np.pi * 0.164 * t)
    f, psd = spectrum(x, DT)
    assert peak_frequency(f, psd, floor=0.0) == pytest.approx(0.009, abs=0.003)
    assert peak_frequency(f, psd) == pytest.approx(0.164, abs=0.003)
    assert PHUGOID_FLOOR_HZ == 0.05  # the declared floor, pinned so it is a choice


def test_an_empty_band_raises_rather_than_returning_a_nan():
    f, psd = spectrum(np.sin(_time()), DT)
    with pytest.raises(ValueError, match="no spectral samples"):
        peak_frequency(f, psd, floor=100.0, ceiling=200.0)


def test_a_history_too_short_to_be_a_spectrum_is_refused():
    with pytest.raises(ValueError, match="not a spectrum"):
        spectrum(np.zeros(3), DT)
    with pytest.raises(ValueError, match="1-D"):
        spectrum(np.zeros((4, 4)), DT)


# ---------------------------------------------------------------------------
# exceedance -- a rate whose closed form is the signal's own frequency
# ---------------------------------------------------------------------------


def test_a_sinusoid_upcrosses_every_level_below_its_amplitude_once_per_cycle():
    """The exact control. A sine at f0 crosses each level |y| < A once per
    cycle, so the rate is f0 for every such level and nothing else."""
    a, f0 = 0.3, 0.164
    x = 1.0 + a * np.sin(2.0 * np.pi * f0 * _time())
    levels = np.array([1.0, 1.1, 1.2, 1.29])
    rates = exceedance(x, DT, levels)
    duration = (x.size - 1) * DT
    # The record holds floor(f0 * duration) whole cycles; the rate is that
    # integer count over the duration, which is f0 to within one crossing.
    assert rates == pytest.approx(np.full(4, f0), abs=1.0 / duration)
    assert len(set(rates)) == 1  # and it is the SAME rate at every level


def test_nothing_upcrosses_a_level_the_signal_never_reaches():
    """The negative control for the above, and the one that fails if a
    comparison operator is the wrong way round."""
    a = 0.3
    x = 1.0 + a * np.sin(2.0 * np.pi * 0.164 * _time())
    assert exceedance(x, DT, [1.0 + a + 1e-9, 2.0, 10.0]) == pytest.approx(0.0)
    assert exceedance(np.full(N, 1.0), DT, [0.5, 1.0, 1.5]) == pytest.approx(0.0)


def test_the_rate_is_per_second_of_record_not_per_sample():
    """Halving dt doubles the sample count and must not touch the rate."""
    f0 = 0.164
    coarse = np.sin(2.0 * np.pi * f0 * np.arange(N) * DT)
    fine = np.sin(2.0 * np.pi * f0 * np.arange(2 * N) * (DT / 2))
    assert exceedance(coarse, DT, [0.0]) == pytest.approx(
        exceedance(fine, DT / 2, [0.0]), rel=1e-3)


def test_a_scalar_level_is_accepted_and_returns_one_rate():
    x = 1.0 + 0.3 * np.sin(2.0 * np.pi * 0.164 * _time())
    assert exceedance(x, DT, 1.0).shape == (1,)


def test_an_exceedance_of_one_sample_is_refused():
    with pytest.raises(ValueError, match="at least two samples"):
        exceedance(np.zeros(1), DT, [0.0])
    with pytest.raises(ValueError, match="1-D"):
        exceedance(np.zeros((4, 4)), DT, [0.0])
