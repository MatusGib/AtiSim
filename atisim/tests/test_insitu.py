"""atisim.insitu -- NASA's in situ reductions, moved out of a script and pinned.

`running_std` and `reductions` were scripts/tpaws_peak_factor.py's, reached by
other scripts through sys.path and tested by nothing. They are pinned
here two ways: against a frozen copy of that script's code, bit for bit, and
against literals computed from it on 23 September 2026, so the module and the
copy cannot drift together.

Design: docs/design/specs/2026-09-23-turbulence-validation-2-design.md, 0.1.
"""

import numpy as np
import pytest
from numpy.lib.stride_tricks import sliding_window_view

from atisim import insitu


# --- the code as it stood in scripts/tpaws_peak_factor.py at 6297bcf --------
def _reference_running_std(d, win):
    if win > d.size:
        raise ValueError(f"window {win} longer than record {d.size}")
    c1 = np.concatenate(([0.0], np.cumsum(d)))
    c2 = np.concatenate(([0.0], np.cumsum(d * d)))
    m = (c1[win:] - c1[:-win]) / win
    m2 = (c2[win:] - c2[:-win]) / win
    return np.sqrt(np.clip(m2 - m * m, 0.0, None))


def _reference_reductions(d, dt):
    win = int(round(5.0 / dt))
    r1 = float(np.max(d) / np.std(d))
    r2 = float(np.max(d) / np.max(_reference_running_std(d, win)))
    r3 = {}
    for seconds in (15.0, 30.0, 60.0):
        n = int(round(seconds / dt))
        segs = d[:d.size - d.size % n].reshape(-1, n)
        r3[seconds] = [float(np.max(s) / np.max(_reference_running_std(s, win)))
                       for s in segs]
    return r1, r2, r3
# ---------------------------------------------------------------------------


def _peak_sigmas(d, n, win=250):
    """Each n-sample encounter's peak 5 s std, computed window by window (two-pass)."""
    return [sliding_window_view(d[k:k + n], win).std(axis=1).max()
            for k in range(0, d.size - n + 1, n)]


@pytest.fixture(scope="module")
def record():
    """120 s of smoothed noise at dt = 0.02 -- a 50-sample running mean."""
    rng = np.random.default_rng(20260923)
    d = np.convolve(rng.standard_normal(6000 + 49), np.ones(50) / 50.0, mode="valid")
    return d - d.mean()


def test_running_std_is_bit_identical_to_the_script_it_came_from(record):
    assert np.array_equal(insitu.running_std(record, 250),
                          _reference_running_std(record, 250))


def test_reductions_are_bit_identical_and_pinned(record):
    got = insitu.reductions(record, 0.02)
    assert got == _reference_reductions(record, 0.02)
    r1, r2, r3 = got
    assert r1 == pytest.approx(3.3845284656624357, rel=1e-12)
    assert r2 == pytest.approx(2.5545665173103074, rel=1e-12)
    assert np.mean(r3[15.0]) == pytest.approx(2.2340964814788156, rel=1e-12)
    assert np.mean(r3[30.0]) == pytest.approx(2.4514401949690208, rel=1e-12)
    assert np.mean(r3[60.0]) == pytest.approx(2.6137351960035855, rel=1e-12)


def test_running_std_is_the_standard_deviation_of_each_window(record):
    win = 250
    rs = insitu.running_std(record, win)
    for i in (0, 123, 5750):
        assert rs[i] == pytest.approx(np.std(record[i:i + win]), rel=1e-9)


def test_running_std_refuses_a_window_longer_than_the_record():
    with pytest.raises(ValueError):
        insitu.running_std(np.zeros(10), 11)


def test_segments_drop_the_incomplete_tail():
    assert insitu.segments(np.arange(10.0), 3).shape == (3, 3)


def test_segments_keep_the_head_and_drop_the_tail():
    assert np.array_equal(insitu.segments(np.arange(10.0), 3),
                          np.arange(9.0).reshape(3, 3))


def test_encounter_peak_factors_are_r3(record):
    """The upward per-encounter peak factor IS reductions' R3, one function."""
    _, _, r3 = insitu.reductions(record, 0.02)
    for L in insitu.ENCOUNTER_SECONDS:
        assert np.array_equal(insitu.encounter_peak_factors(record, 0.02, L),
                              np.array(r3[L]))


def test_the_moved_code_is_bit_identical_on_a_record_with_a_tail(record):
    """Every real record leaves a tail (55,000 samples: 250/1000/1000); 6000 leaves none."""
    d = record[:5900]
    got = insitu.reductions(d, 0.02)
    assert got == _reference_reductions(d, 0.02)
    for L in insitu.ENCOUNTER_SECONDS:
        assert np.array_equal(insitu.encounter_peak_factors(d, 0.02, L), np.array(got[2][L]))


def test_the_absolute_peak_factor_is_never_below_the_upward_one(record):
    win = insitu.window_samples(0.02)
    assert (insitu.peak_factor(record, win, absolute=True)
            >= insitu.peak_factor(record, win))
    # and they coincide when the largest excursion is upward
    up = np.abs(record)
    assert (insitu.peak_factor(up, win, absolute=True)
            == insitu.peak_factor(up, win))


def test_the_absolute_peak_factor_counts_a_downward_peak(record):
    win = insitu.window_samples(0.02)
    down = -record  # the fixture's largest excursion is upward, so this one's is downward
    rs = np.max(insitu.running_std(down, win))
    assert insitu.peak_factor(down, win, absolute=True) == pytest.approx(
        np.max(np.abs(down)) / rs, rel=1e-12)
    assert insitu.peak_factor(down, win, absolute=True) > insitu.peak_factor(down, win)


def test_encounter_peak_sigmas_are_peak_running_stds(record):
    """Whole-encounter std -- session 34's reduction -- fails this."""
    d = record[:5900]
    assert insitu.encounter_peak_sigmas(d, 0.02, 30.0) == pytest.approx(
        _peak_sigmas(d, 1500), rel=1e-9)


def test_matched_gain_of_an_exact_multiple_is_the_multiple(record):
    """A load that is exactly 0.04 g per m/s of gust reads 0.04 on any window."""
    assert insitu.matched_gain([(0.04 * record, record)], 0.02, 30.0) == pytest.approx(
        0.04, rel=1e-12)


def test_matched_gain_is_a_ratio_of_means_not_a_mean_of_ratios():
    """The measured side cannot pair sigma_dn to sigma_w, so it is a ratio of
    population means -- and the model must be reduced the same way."""
    rng = np.random.default_rng(1)
    g1 = rng.standard_normal(3000)
    g2 = 3.0 * rng.standard_normal(3000)
    got = insitu.matched_gain([(1.0 * g1, g1), (2.0 * g2, g2)], 0.02, seconds=30.0)
    s1 = insitu.encounter_peak_sigmas(g1, 0.02, 30.0)
    s2 = insitu.encounter_peak_sigmas(g2, 0.02, 30.0)
    expected = (np.sum(s1) + 2.0 * np.sum(s2)) / (np.sum(s1) + np.sum(s2))
    assert got == pytest.approx(expected, rel=1e-12)
    assert abs(got - 1.5) > 0.05   # the mean of the two ratios would be 1.5


def test_matched_gain_on_a_pair_that_is_not_a_multiple(record):
    load = np.convolve(record, np.ones(25) / 25.0, mode="same")
    want = np.mean(_peak_sigmas(load, 1500)) / np.mean(_peak_sigmas(record, 1500))
    assert insitu.matched_gain([(load, record)], 0.02, 30.0) == pytest.approx(want, rel=1e-9)


def test_peak_regression_recovers_a_constant_ratio():
    """Encounters that are exact multiples of one another lie on a line
    through the origin whose slope is their common absolute peak factor."""
    t = np.arange(1500) * 0.02
    base = np.sin(2 * np.pi * 0.23 * t) + 0.4 * np.sin(2 * np.pi * 1.7 * t + 0.3)
    record = np.concatenate([k * base for k in (1.0, 2.0, 3.0)])
    slope, intercept = insitu.peak_regression([record], 0.02, seconds=30.0)
    win = insitu.window_samples(0.02)
    assert slope == pytest.approx(insitu.peak_factor(base, win, absolute=True),
                                  rel=1e-9)
    assert abs(intercept) < 1e-9


def test_peak_regression_uses_the_absolute_peak():
    t = np.arange(1500) * 0.02
    base = np.sin(2 * np.pi * 0.23 * t) + 0.4 * np.sin(2 * np.pi * 1.7 * t + 0.3)
    base[700] -= 3.0  # the largest excursion is now downward
    slope, intercept = insitu.peak_regression(
        [np.concatenate([k * base for k in (1.0, 2.0, 3.0)])], 0.02, seconds=30.0)
    rs = np.max(insitu.running_std(base, insitu.window_samples(0.02)))
    assert slope == pytest.approx(np.max(np.abs(base)) / rs, rel=1e-9)
    assert slope > 1.5 * np.max(base) / rs
    assert abs(intercept) < 1e-9


def test_the_slope_through_the_origin_cannot_be_moved_by_the_intensities():
    """Two encounter shapes with different peak factors, flown at two sets of
    scales. The slope through the origin is sum(xy)/sum(x^2) by hand for both
    sets; the fitted slope is not the same for both, which is why T2 of
    scripts/peak_factor_distribution.py does not headline it."""
    t = np.arange(1500) * 0.02
    a = np.sin(2 * np.pi * 0.23 * t) + 0.4 * np.sin(2 * np.pi * 1.7 * t + 0.3)
    b = np.sin(2 * np.pi * 0.11 * t)
    b[900] += 2.5  # a sharper peak: a larger peak factor than a's
    win = insitu.window_samples(0.02)
    xa, xb = (np.max(insitu.running_std(s, win)) for s in (a, b))
    ya, yb = np.max(np.abs(a)), np.max(np.abs(b))
    by_hand = (xa * ya + xb * yb) / (xa**2 + xb**2)

    def flights(scales):
        return [np.concatenate([k * a, k * b]) for k in scales]

    for scales in ((1.0, 2.0, 3.0), (1.0, 5.0, 10.0)):
        assert insitu.peak_slope_through_origin(
            flights(scales), 0.02, 30.0) == pytest.approx(by_hand, rel=1e-12)
    narrow, _ = insitu.peak_regression(flights((1.0, 2.0, 3.0)), 0.02, 30.0)
    wide, _ = insitu.peak_regression(flights((1.0, 5.0, 10.0)), 0.02, 30.0)
    assert abs(narrow - wide) > 0.05


def test_peak_pairs_are_figure_2s_axes_per_encounter():
    t = np.arange(3000) * 0.02
    d = np.sin(2 * np.pi * 0.23 * t) * np.where(t < 30.0, 1.0, 2.0)
    x, y = insitu.peak_pairs([d], 0.02, 30.0)
    win = insitu.window_samples(0.02)
    first, second = insitu.segments(d, 1500)
    assert np.array_equal(x, [np.max(insitu.running_std(first, win)),
                              np.max(insitu.running_std(second, win))])
    assert np.array_equal(y, [np.max(np.abs(first)), np.max(np.abs(second))])


def test_peak_slope_se_is_the_sandwich_over_clusters():
    """Distinct labels make every encounter its own cluster, and the SE is then
    sqrt(sum (x e)^2) / sum(x^2), with e = y - b x, by hand."""
    rng = np.random.default_rng(11)
    recs = [s * rng.standard_normal(3000) for s in (1.0, 1.5, 2.5)]
    x, y = insitu.peak_pairs(recs, 0.02, 30.0)
    b = float(np.dot(x, y) / np.dot(x, x))
    by_hand = np.sqrt(np.sum((x * (y - b * x)) ** 2)) / np.dot(x, x)
    assert insitu.peak_slope_se(recs, [0, 1, 2], 0.02, 30.0) == pytest.approx(
        by_hand, rel=1e-12)


def test_a_realisation_flown_again_adds_no_information():
    """One record at two intensities under one label is one set of encounters,
    and the SE is the single record's. Labelled as independent, it shrinks."""
    d = np.random.default_rng(12).standard_normal(6000)
    once = insitu.peak_slope_se([d], [0], 0.02, 30.0)
    assert insitu.peak_slope_se([d, 2.0 * d], [0, 0], 0.02, 30.0) == pytest.approx(
        once, rel=1e-12)
    assert insitu.peak_slope_se([d, 2.0 * d], [0, 1], 0.02, 30.0) < 0.9 * once


def _bursty(dt=0.02):
    """Quiet noise with two 0.3 Hz bursts, 100-130 s and 300-330 s; only the
    first reaches the 0.2 g significance threshold (sigma 0.42 against 0.11)."""
    rng = np.random.default_rng(3)
    t = np.arange(0.0, 400.0, dt)
    d = 0.02 * rng.standard_normal(t.size)
    for start, amp in ((100.0, 0.6), (300.0, 0.15)):
        k = (t >= start) & (t < start + 30.0)
        d[k] += amp * np.sin(2 * np.pi * 0.3 * t[k])
    return t, d


@pytest.mark.parametrize("rule", ["fraction", "level", "fixed"])
def test_events_find_the_significant_burst_and_skip_the_weak_one(rule):
    t, d = _bursty()
    spans = insitu.events(d, 0.02, rule=rule)
    assert len(spans) == 1, spans
    a, b = spans[0]
    overlap = min(t[b - 1], 130.0) - max(t[a], 100.0)
    assert overlap >= 20.0, (rule, t[a], t[b - 1])


@pytest.mark.parametrize("rule", ["fraction", "level", "fixed"])
def test_events_never_overlap_and_are_at_least_a_window_long(rule):
    rng = np.random.default_rng(5)
    d = 3.0 * np.convolve(rng.standard_normal(30000 + 49), np.ones(50) / 50.0,
                          mode="valid")
    spans = insitu.events(d, 0.02, rule=rule, threshold=0.2)
    assert spans
    for (a1, b1), (a2, b2) in zip(spans, spans[1:]):
        assert b1 <= a2
    assert all(b - a >= insitu.window_samples(0.02) for a, b in spans)


def test_an_unknown_event_rule_is_refused():
    with pytest.raises(ValueError):
        insitu.events(np.zeros(1000), 0.02, rule="cloud")


def test_event_statistics_are_the_encounter_ones_on_each_span():
    t, d = _bursty()
    spans = insitu.events(d, 0.02, rule="fraction")
    win = insitu.window_samples(0.02)
    a, b = spans[0]
    assert insitu.event_peak_factors(d, spans, 0.02)[0] == insitu.peak_factor(d[a:b], win)
    x, y = insitu.event_points(d, spans, 0.02)
    assert y[0] == float(np.max(np.abs(d[a:b])))
    assert x[0] == float(np.max(insitu.running_std(d[a:b], win)))
