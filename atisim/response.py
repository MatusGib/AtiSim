"""Response statistics: a run as a SPECTRUM and as a RATE, not as a peak.

WHY THIS MODULE EXISTS. Every comparison in section 4 matches a PEAK from a
single encounter. That is one realisation of a random process, and it carries
the sampling error of a sample of size one -- the weakest statistic the data can
support. Two statistics answer better questions, and neither existed anywhere in
this tree before this module: there was no FFT, no periodogram and no PSD of any
RESPONSE (`wind.dryden_spectrum` is an INPUT spectrum, which is a different
object), and no exceedance count of anything.

WHAT EACH ONE BUYS, WHICH IS NOT THE SAME THING:

  `spectrum` tests the COUPLING. A peak says how hard the aircraft was shaken;
  a spectrum says at which frequencies, and therefore whether the airframe's own
  dynamics organised the response or merely passed the input through. Yoshimura
  et al. 2023 (GRL 50, e2022GL101286) validate a CAT simulation exactly this way
  -- their Fig. 6 compares the frequency spectra of simulated and recorded
  vertical acceleration and checks that the peak lands near the aircraft's own
  natural frequency (0.14 Hz for their B787). That protocol is second-hand here
  and is named as such; what is first-hand is that this project can now run it.

  `exceedance` puts an N in the denominator. A rate of upcrossings per second,
  measured over an ensemble, is a statistic whose error bar shrinks as more
  realisations are flown -- unlike a peak, whose error bar does not shrink at
  all.

WHY PERIODOGRAM-PER-RUN AND NOT WELCH. Welch averages SEGMENTS of one record,
which trades frequency resolution for variance reduction. These runs are short
(the Mehta array is ~47 s at cruise) and the frequencies of interest -- a short
period near 0.16 Hz, a core passage near 0.19 Hz -- are about two bins apart at
full record length. Splitting the record would merge them. Averaging over
REALISATIONS instead reduces the same variance and costs no resolution, and it
is what Yoshimura did: "average of 151 spectra", one per virtual flight.

EVERYTHING HERE IS NUMPY AND TAKES A SAMPLED HISTORY, not a JAX rollout --
same standing as `checks.py`. Nothing in this module is differentiated or
jitted, and none of it belongs inside the integrator.
"""

import numpy as np
from scipy import signal

__all__ = ["spectrum", "peak_frequency", "exceedance", "separability",
           "standardised_difference", "PHUGOID_FLOOR_HZ"]


# The default low-frequency floor for a peak search, in Hz. NOT a physical
# constant and not sourced -- a DECLARED analysis choice, stated here rather
# than buried in a call site. The 747's phugoid at the cruise conditions this
# project flies sits near 0.009 Hz with zeta ~ 0.05, so it is both far below
# every forcing frequency of interest and lightly enough damped to dominate the
# very low end of a short record. A peak search that includes it answers a
# question about the start transient instead of about the gust response. Any
# caller may override it; every caller must report what it used.
PHUGOID_FLOOR_HZ = 0.05


def spectrum(x, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """One-sided power spectral density of `x`, sampled uniformly at `dt`.

    Returns `(f_hz, psd)` with `psd` in units of `x`-squared per hertz, so that
    `np.trapezoid(psd, f_hz)` recovers the variance of the windowed signal.
    That normalisation is the reason this wrapper exists at all: "spectrum" is
    ambiguous over at least four conventions (one- or two-sided, density or
    amplitude, windowed or not), and a comparison between two spectra computed
    under different ones is meaningless. `test_response.py` asserts the
    convention against a signal whose variance is known in closed form.

    The mean is removed and a Hann window applied. Removing the mean is what
    makes this a spectrum of the DISTURBANCE rather than of the trim state --
    for `n_z` the mean is 1 g, whose DC line would otherwise be larger than
    everything else in the record by orders of magnitude.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"expected a 1-D history, got shape {x.shape}")
    if x.size < 4:
        raise ValueError(f"a spectrum of {x.size} samples is not a spectrum")
    return signal.periodogram(
        x, fs=1.0 / dt, window="hann", detrend="constant", scaling="density",
    )


def peak_frequency(f_hz, psd, *, floor: float = PHUGOID_FLOOR_HZ,
                   ceiling: float | None = None) -> float:
    """The frequency of the largest spectral density in `[floor, ceiling]`.

    `floor` defaults to `PHUGOID_FLOOR_HZ` for the reason given there. A band
    is REQUIRED rather than optional-and-ignored because "the peak of the
    spectrum" is not well defined without one: a band-limited response and a
    lightly damped low-frequency mode both produce maxima, and which one is
    reported is otherwise decided by record length.

    Raises if the band contains no samples, rather than returning a nan that
    would propagate silently into a comparison.
    """
    f_hz, psd = np.asarray(f_hz, float), np.asarray(psd, float)
    hi = f_hz.max() if ceiling is None else ceiling
    band = (f_hz >= floor) & (f_hz <= hi)
    if not band.any():
        raise ValueError(f"no spectral samples in [{floor}, {hi}] Hz")
    return float(f_hz[band][np.argmax(psd[band])])


def exceedance(x, dt: float, levels) -> np.ndarray:
    """Upcrossings of each level in `levels`, per second of record.

    An UPCROSSING is a sample below-or-at the level followed by one above it.
    Counting crossings rather than peaks is the classical form of a gust-load
    exceedance statistic, and it has the property that makes it checkable: a
    sinusoid of amplitude A crosses every level |y| < A exactly once per cycle,
    so its exceedance rate is its own frequency for all such levels and zero
    above them. `test_response.py` asserts exactly that.

    The rate is per second of the record supplied. It is NOT per flight hour and
    NOT per nautical mile; converting is the caller's business and depends on a
    ground speed this function is not given.

    Both signs are the caller's business too. For a load history, negative
    exceedances are the upcrossings of `-x` against `-levels`, and the two
    channels are worth reporting separately -- section 5 records that this
    model's linear lift makes them exactly symmetric, which is a claim only a
    two-sided count can display.
    """
    x = np.asarray(x, dtype=float)
    levels = np.atleast_1d(np.asarray(levels, dtype=float))
    if x.ndim != 1:
        raise ValueError(f"expected a 1-D history, got shape {x.shape}")
    if x.size < 2:
        raise ValueError("an exceedance rate needs at least two samples")
    below, above = x[:-1, None] <= levels[None, :], x[1:, None] > levels[None, :]
    return (below & above).sum(axis=0) / ((x.size - 1) * dt)


# ---------------------------------------------------------------------------
# Separability of two ensembles
#
# WHY THESE ARE HERE AND NOT IN A SCRIPT. Session 23d asked whether two clouds
# of Fig. 8 coordinates overlap and answered with the GAP BETWEEN THEIR
# EXTREMES. That statistic's expectation MOVES WITH N: the extremes of a
# distribution spread as more samples are drawn, so the gap shrinks towards zero
# with effort whatever the truth is, and collecting more evidence makes "the
# clouds are separate" harder to say. The record calls that result marginal, and
# it was right to.
#
# Both functions below have an expectation that does NOT depend on N, so more
# seeds shrink the error bar instead of moving the answer. They belong beside
# `exceedance` for the same reason it exists: a statistic with an N in its
# denominator is a different kind of claim from a peak.
# ---------------------------------------------------------------------------


def separability(a, b) -> float:
    """P(b > a) for one independent draw from each, ties counted as a half.

    The Mann-Whitney U statistic divided by `len(a) * len(b)`, also called the
    AUC. 1.0 is perfect separation with `b` above, 0.0 perfect separation with
    `b` below, and 0.5 no separation at all.

    UNPAIRED ON PURPOSE, even where the two ensembles share their random seeds.
    The question this answers is whether two INDEPENDENT records can be told
    apart, because a real diagnosis has one record and no matched control.
    Pairing would answer a different and easier question.

    It is a RANK statistic, so it is invariant under any strictly increasing
    transform applied to both inputs -- `test_response.py` asserts exactly that,
    because it is the property that makes the result about the distributions
    rather than about the units they were measured in.
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size == 0 or b.size == 0:
        raise ValueError("separability needs a non-empty sample on both sides")
    difference = b[None, :] - a[:, None]
    return float((np.count_nonzero(difference > 0)
                  + 0.5 * np.count_nonzero(difference == 0)) / difference.size)


def standardised_difference(a, b) -> float:
    """Cohen's d: (mean(b) - mean(a)) / pooled standard deviation.

    Reported beside `separability` because the two fail differently and a reader
    should see both. `separability` saturates at 1.0 the moment no pair
    overlaps, and cannot then say whether the clouds are a hair apart or a mile;
    `d` keeps counting. Against that, `d` assumes the spread is meaningful and a
    rank statistic does not.

    Returns NaN where both samples are constant, rather than raising or
    returning a large number that would sort to the top of a ranking.
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size < 2 or b.size < 2:
        raise ValueError("a pooled standard deviation needs 2 samples a side")
    pooled = np.sqrt(((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1))
                     / (a.size + b.size - 2))
    if pooled == 0.0:
        return float("nan")
    return float((b.mean() - a.mean()) / pooled)
