"""The DC-10's altitude through the Hannibal vortex pair: did it climb?

Data: `atisim/data/parks1985_fig6_altitude.csv`, digitised by
`scripts/digitise_parks_fig6_altitude.py` from Parks, Wingrove, Bach & Mehta 1985
(J. Aircraft 22(2)) Fig. 6, printed p. 127 -- the recorded time histories of the
Hannibal encounter. The altitude panel carries the MEASURED (barometric) altitude
and the altitude ESTIMATED from inertial measurements, the path the winds were
computed along.

Why it matters: the simulated 747 climbs ~500-600 ft before Mehta's cores 3 and 4
and passes above them (PROJECT.md section 4, "The Hannibal horizontal wind").
Mehta placed those cores relative to the DC-10's path, so whether the DC-10 made
the same climb decides whether the 747 should meet the fitted wind on its own
path or on the nominal one. Bands and orderings, per docs/DEVELOPMENT.md rule 6. The suite
does not import from `scripts/`, so the time-to-distance reading is rebuilt here
from the two tracked tables.
"""

import csv
import math
from pathlib import Path

import numpy as np
import pytest

import atisim
from atisim import wind

DATA = Path(atisim.__file__).parent / "data" / "parks1985_fig6_altitude.csv"
FIG7 = Path(atisim.__file__).parent / "data" / "tm102186_fig7_winds.csv"
KT2FTS = 1.68781
BIAS_KT = 149.8  # Mehta 1987 p. 30, b_xy
CRUISE_FT = 37000.0  # Parks p. 126, "cruising ... at 37,000 ft"


@pytest.fixture(scope="module")
def d():
    out = {}
    with open(DATA, newline="") as f:
        for r in csv.DictReader(f):
            out.setdefault((r["panel"], r["curve"]), []).append((float(r["t_s_after_0121"]), float(r["value"])))
    return {k: np.array(sorted(v)) for k, v in out.items()}


def _baro(d):
    a = np.concatenate([d[("altitude", "barometric")], d[("altitude", "barometric_on_axis")]])
    return a[np.argsort(a[:, 0])]


def _estimate_at(d, t):
    est = d[("altitude", "inertial_estimate")]
    if est[0, 0] - 1.0 <= t <= est[-1, 0] + 1.0:
        return float(np.interp(t, est[:, 0], est[:, 1]))
    b = _baro(d)
    return float(np.interp(t, b[:, 0], b[:, 1]))


def _t_wz_min(d):
    w = d[("vertical_wind", "ink_bottom")]
    return float(w[np.argmin(w[:, 1]), 0])


def _clocks(d):
    """t(x_kft) for both readings of Mehta's distance axis: air-relative and ground."""
    lo, hi = d[("true_airspeed", "ink_bottom")], d[("true_airspeed", "ink_top")]
    quiet = lo[:, 0] < 180.0
    tas = float(np.median((lo[quiet, 1] + hi[quiet, 1]) / 2.0))
    with open(FIG7, newline="") as f:
        dots = [r for r in csv.DictReader(f) if r["panel"] == "vertical" and r["curve"] == "actual"]
    x0 = float(min(dots, key=lambda r: float(r["value"]))["x_kft"])
    cos_psi = math.cos(math.radians(wind.MEHTA_HANNIBAL_PSI_DEG))
    t0 = _t_wz_min(d)
    return [lambda x, v=v: t0 + (x - x0) * 1000.0 / v
            for v in (tas * KT2FTS, (tas + BIAS_KT * cos_psi) * KT2FTS)]


def test_the_table_carries_both_altitude_traces_and_the_timing_panels(d):
    assert _baro(d).shape[0] > 400, "the solid trace, column by column over five minutes"
    est = d[("altitude", "inertial_estimate")]
    assert est.shape[0] > 20 and 195.0 < est[0, 0] and est[-1, 0] < 240.0, "the dashes, only where the traces part"
    for panel in ("normal_acceleration", "vertical_wind", "true_airspeed"):
        assert d[(panel, "ink_top")].shape[0] > 400


def test_the_load_panel_reproduces_the_band_the_paper_quotes(d):
    """CONTROL on the y calibration. p. 127: "+1.7 to -1.0 gs". Measured -0.96 and
    +1.90: the plotted envelope's absolute peak exceeds the quoted +1.7, as it does
    in TM-102186 Fig. 6 (+1.855, session 27)."""
    assert -1.1 < d[("normal_acceleration", "ink_bottom")][:, 1].min() < -0.9
    assert 1.7 <= d[("normal_acceleration", "ink_top")][:, 1].max() < 2.0


def test_the_cruise_before_the_encounter_reads_37000_ft(d):
    """CONTROL. Parks p. 126, "cruising ... at 37,000 ft". Measured 37,018 ft mean,
    1:21-1:24, on a panel that resolves ~3 ft per pixel."""
    b = _baro(d)
    assert abs(b[b[:, 0] < 180.0, 1].mean() - CRUISE_FT) < 50.0


def test_the_panels_share_one_clock(d):
    """CONTROL on the per-panel x calibration of a skewed scan. The load minimum and
    the vertical-wind minimum come from separately calibrated panels; measured 1.0 s
    apart. TM-102186 Fig. 6 puts its trough at 213.1 s (session 27's reader)."""
    nz = d[("normal_acceleration", "ink_bottom")]
    t_nz = float(nz[np.argmin(nz[:, 1]), 0])
    assert abs(t_nz - _t_wz_min(d)) < 2.0
    assert 208.0 < t_nz < 218.0


@pytest.mark.parametrize("reading", [0, 1], ids=["air-relative", "ground"])
def test_the_dc10_did_not_climb_before_cores_3_and_4(d, reading):
    """THE ANSWER. Mapped onto Mehta's axis, the inertially estimated altitude at the
    core-3 and core-4 passages is 36,985-36,996 ft: 17-61 ft BELOW where the 747's
    run starts, where the simulated 747 has climbed +512 ft (bare) and +596 ft
    (shipped). The band is generous; the ordering against 500 ft is the claim."""
    t_at = _clocks(d)[reading]
    start = _estimate_at(d, t_at((wind.MEHTA_HANNIBAL_X_FT[0] - 12.0 * 500.5) / 1000.0))
    for core in (2, 3):
        e = _estimate_at(d, t_at(wind.MEHTA_HANNIBAL_X_FT[core] / 1000.0))
        assert abs(e - CRUISE_FT) < 60.0
        assert e - start < 50.0 < 500.0


def test_the_climb_came_after_the_vortex_pair(d):
    """THE ORDERING. The estimate bottoms out at the pair (measured 36,968 ft, 2.6 s
    before the vertical-wind minimum); the measured altitude peaks at 37,436 ft
    21.6 s after it -- past core 5 on either reading of the distance axis."""
    est = d[("altitude", "inertial_estimate")]
    t0 = _t_wz_min(d)
    assert abs(est[np.argmin(est[:, 1]), 0] - t0) < 5.0
    b = _baro(d)
    i = np.argmax(b[:, 1])
    assert b[i, 1] > CRUISE_FT + 350.0
    assert b[i, 0] - t0 > 15.0
    for t_at in _clocks(d):
        assert b[i, 0] > t_at(wind.MEHTA_HANNIBAL_X_FT[4] / 1000.0)
