"""NASA's in situ turbulence reductions, in one tested module.

WHY THIS EXISTS. NASA/TM-2012-217337 (TPAWS) defines every load statistic
it tabulates on a 5 s sliding window, and a comparison with it is decided by
whether BOTH sides use that window. Reduced over a whole 20-minute record, the
model's peak factor reads 3.87; reduced TPAWS' way, the same flights read
1.94. The gust response gain carries the same mismatch, worth ~14%. Reduce a
simulated record with these functions before setting it against TPAWS.

Every function takes 1-D numpy records at a fixed `dt`. Callers pass
deviations from the record mean: the running standard deviation removes each
window's own mean, but a peak is measured from zero.
"""

import numpy as np

TAU_SECONDS = 5.0  # TPAWS' own window, printed p. 5
ENCOUNTER_SECONDS = (15.0, 30.0, 60.0)  # DECLARED; printed p. 7's duration band


def window_samples(dt, tau_seconds=TAU_SECONDS):
    """Samples in TPAWS' sliding window at step `dt`."""
    return int(round(tau_seconds / dt))


def running_std(d, win):
    """The printed p. 5 equation: std over a sliding window of `win` samples.

    Mean of the square minus the square of the mean, both over the SAME window,
    which is what the document's inner integral makes it. Clipped at zero before
    the root because cancellation can put the variance a few ulp below it.
    """
    if win > d.size:
        raise ValueError(f"window {win} longer than record {d.size}")
    c1 = np.concatenate(([0.0], np.cumsum(d)))
    c2 = np.concatenate(([0.0], np.cumsum(d * d)))
    m = (c1[win:] - c1[:-win]) / win
    m2 = (c2[win:] - c2[:-win]) / win
    return np.sqrt(np.clip(m2 - m * m, 0.0, None))


def reductions(d, dt):
    """R1, R2 and R3(L) for one record. Returns (r1, r2, {L: [per-segment...]})."""
    win = int(round(TAU_SECONDS / dt))
    r1 = float(np.max(d) / np.std(d))
    r2 = float(np.max(d) / np.max(running_std(d, win)))
    r3 = {}
    for seconds in ENCOUNTER_SECONDS:
        n = int(round(seconds / dt))
        segs = d[:d.size - d.size % n].reshape(-1, n)
        r3[seconds] = [float(np.max(s) / np.max(running_std(s, win))) for s in segs]
    return r1, r2, r3


def segments(d, n):
    """Non-overlapping `n`-sample encounters; a tail that cannot fill one is dropped.

    The same cut `reductions` makes for R3, so the gain and the peak factor are
    always taken over the SAME encounters.
    """
    return d[:d.size - d.size % n].reshape(-1, n)


def peak_factor(d, win, *, absolute=False):
    """One encounter's peak over the maximum of its 5 s running std.

    `absolute=False` is TPAWS Table 1's Delta_n_max / sigma_dn. `absolute=True`
    is max|Delta_n| / sigma_dn, which is what TPAWS Figure 2 plots for 606
    encounters (printed p. 6) -- a different statistic, never to be compared
    with the other.
    """
    peak = np.max(np.abs(d)) if absolute else np.max(d)
    return float(peak / np.max(running_std(d, win)))


def encounter_peak_factors(d, dt, seconds, *, absolute=False,
                           tau_seconds=TAU_SECONDS):
    """`peak_factor` of every `seconds`-long encounter in the record."""
    win = window_samples(dt, tau_seconds)
    n = int(round(seconds / dt))
    return np.array([peak_factor(s, win, absolute=absolute)
                     for s in segments(d, n)])


def encounter_peak_sigmas(d, dt, seconds, tau_seconds=TAU_SECONDS):
    """Each encounter's peak 5 s running std: TPAWS' sigma_dn, or Fig. 133's sigma_w."""
    win = window_samples(dt, tau_seconds)
    n = int(round(seconds / dt))
    return np.array([float(np.max(running_std(s, win))) for s in segments(d, n)])


def matched_gain(records, dt, seconds, tau_seconds=TAU_SECONDS):
    """sigma_nz / sigma_w reduced the way the measured B-757 side is.

    `records` is an iterable of (load, gust) pairs. Every encounter's peak
    running std is taken for both, pooled over all records, and the gain is
    the RATIO OF THE TWO MEANS -- not the mean of per-encounter ratios. The
    measured side cannot pair TPAWS Table 1's sigma_dn to Figure 133's sigma_w,
    so it is a ratio of population means, and the model is reduced to match.
    """
    loads, gusts = [], []
    for load, gust in records:
        loads.extend(encounter_peak_sigmas(load, dt, seconds, tau_seconds))
        gusts.extend(encounter_peak_sigmas(gust, dt, seconds, tau_seconds))
    return float(np.mean(loads) / np.mean(gusts))


def peak_pairs(records, dt, seconds, tau_seconds=TAU_SECONDS):
    """Every encounter's (peak running std, max|d|), pooled over `records`.

    TPAWS Figure 2's two axes: one point per encounter, peak |Delta_n| against
    peak sigma_dn on a 5 s window (printed p. 6). Returned as two arrays, in
    record order.
    """
    x, y = [], []
    win = window_samples(dt, tau_seconds)
    n = int(round(seconds / dt))
    for d in records:
        for s in segments(d, n):
            x.append(float(np.max(running_std(s, win))))
            y.append(float(np.max(np.abs(s))))
    return np.asarray(x), np.asarray(y)


def peak_regression(records, dt, seconds, tau_seconds=TAU_SECONDS):
    """(slope, intercept) of max|d| on peak running std, across every encounter.

    The fit TPAWS Figure 2 prints, y = 2.59462x + 0.00399 over 606 measured
    encounters. ON THE MODEL IT IS NOT COMPARABLE WITH THAT LINE. Within one
    intensity max|d| rises more slowly than peak sigma, so the fitted slope
    blends that within-intensity slope with the ratio of means, weighted by
    how far apart the intensities flown are: a few discrete intensities give a
    lower slope and a positive intercept. `peak_slope_through_origin` is the
    comparable statistic.
    """
    x, y = peak_pairs(records, dt, seconds, tau_seconds)
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def peak_slope_through_origin(records, dt, seconds, tau_seconds=TAU_SECONDS):
    """sum(xy) / sum(x^2) over every encounter's (peak running std, max|d|).

    The least-squares slope of a line through the origin. One realisation
    flown at several scales gives the same slope at every scale, so unlike
    `peak_regression` it cannot be moved by the choice of intensities. Figure
    2's 606 encounters spread over sigma continuously and its printed line
    passes 0.004 g from the origin, so this is the model-side statistic to set
    against its 2.59462.
    """
    x, y = peak_pairs(records, dt, seconds, tau_seconds)
    return float(np.dot(x, y) / np.dot(x, x))


def peak_slope_se(records, groups, dt, seconds, tau_seconds=TAU_SECONDS):
    """Standard error of `peak_slope_through_origin`, clustered on encounters.

    Encounters are not independent points: a realisation flown again at
    another intensity, or by another aircraft, repeats them. `groups` labels
    each record in the sequence `records`; encounters sharing a label and a
    position in their record form one cluster, and
    SE = sqrt(sum_g (sum_{i in g} x_i e_i)^2) / sum(x^2), with e = y - b x --
    the sandwich estimator of a slope through the origin.
    """
    b = peak_slope_through_origin(records, dt, seconds, tau_seconds)
    score, sxx = {}, 0.0
    for d, g in zip(records, groups, strict=True):
        x, y = peak_pairs([d], dt, seconds, tau_seconds)
        for k, (xi, yi) in enumerate(zip(x, y)):
            score[(g, k)] = score.get((g, k), 0.0) + xi * (yi - b * xi)
        sxx += float(np.dot(x, x))
    return float(np.sqrt(sum(s * s for s in score.values())) / sxx)


def events(d, dt, *, rule, threshold=0.2, fraction=0.5, level=0.1,
           span_seconds=30.0, tau_seconds=TAU_SECONDS):
    """Encounters as TPAWS selects them, under one DECLARED boundary rule.

    TPAWS calls an event significant if its peak 5 s running sigma_dn reaches
    0.2 g (printed p. 7) and says events "typically" last under 30 s (printed
    p. 133). It does not say where an event starts or stops, so the boundary is
    a choice, and this function offers a family of them:

      "fraction"  contiguous samples around the peak where the running sigma
                  stays at or above `fraction` of that peak
      "level"     ... at or above `level` g -- 0.1 g is Bowles & Buck's
                  light/moderate boundary (TPAWS printed p. 6)
      "fixed"     `span_seconds` centred on the peak

    Peaks are taken largest first, local maxima only; an event overlapping one
    already taken is skipped. Returns (start, stop) sample indices, stop
    exclusive, sorted by start. Every span is at least one window long.
    """
    if rule not in ("fraction", "level", "fixed"):
        raise ValueError(f"unknown event rule {rule!r}")
    win = window_samples(dt, tau_seconds)
    rs = running_std(d, win)          # rs[i] is the window d[i:i + win]
    taken = np.zeros(d.size, dtype=bool)
    spans = []
    for i in np.argsort(rs)[::-1]:
        if rs[i] < threshold:
            break
        if (i > 0 and rs[i - 1] > rs[i]) or (i + 1 < rs.size and rs[i + 1] > rs[i]):
            continue
        if rule == "fixed":
            half = int(round(span_seconds / dt)) // 2
            centre = i + win // 2
            start, stop = max(0, centre - half), min(d.size, centre + half)
        else:
            floor = fraction * rs[i] if rule == "fraction" else level
            a = i
            while a > 0 and rs[a - 1] >= floor:
                a -= 1
            b = i
            while b + 1 < rs.size and rs[b + 1] >= floor:
                b += 1
            start, stop = a, min(d.size, b + win)
        if stop - start < win or taken[start:stop].any():
            continue
        taken[start:stop] = True
        spans.append((int(start), int(stop)))
    return sorted(spans)


def event_peak_factors(d, spans, dt, *, absolute=False, tau_seconds=TAU_SECONDS):
    """`peak_factor` over each (start, stop) span from `events`."""
    win = window_samples(dt, tau_seconds)
    return np.array([peak_factor(d[a:b], win, absolute=absolute) for a, b in spans])


def event_points(d, spans, dt, tau_seconds=TAU_SECONDS):
    """Figure 2's axes for each event: (peak running std, max|d|) lists."""
    win = window_samples(dt, tau_seconds)
    return ([float(np.max(running_std(d[a:b], win))) for a, b in spans],
            [float(np.max(np.abs(d[a:b]))) for a, b in spans])
