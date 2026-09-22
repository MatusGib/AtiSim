"""TPAWS Table 1, pinned so a re-extraction cannot drift.

Data: `atisim/data/tpaws_tm2012_217337_table1.csv`, written by
`scripts/tpaws_table1_ingest.py` from NASA/TM-2012-217337 (Hamilton, Proctor &
Ahmad, February 2012, NTRS 20120003172) Table 1, printed p. 8. The table
EXTRACTS FROM THE TEXT LAYER, so unlike this project's digitised figures there
is no reading uncertainty to allow for and the pins can be exact.

Why the row count is pinned and not merely bounded: the design document's own
S2 statistics were computed on a 51-row parse, because the parser silently
dropped the two rows whose altitude prints as a RANGE -- 232-05 "31 to 35" and
235-05* "22 to 19". Neither dropped row moves any column's range, which is
exactly why the error survived review. `test_the_two_ranged_altitudes_survive`
pins the defect itself rather than only its symptom.

Rule 6 asks for bands and orderings rather than values. The bands here are the
DOCUMENT's own statements about its own table, not physical claims about the
model, so they are pinned as printed.
"""

import csv
from pathlib import Path

import pytest

import atisim
from atisim.aircraft import REGISTRY

DATA = Path(atisim.__file__).parent / "data" / "tpaws_tm2012_217337_table1.csv"

FT2M = 0.3048
N_EVENTS = 53  # printed p. 7 and printed p. 5, the document's own count
SIGMA_SIGNIFICANT_G = 0.2  # printed p. 7, the document's own criterion
RANGED = {"232-05": (31.0, 35.0), "235-05": (22.0, 19.0)}


@pytest.fixture(scope="module")
def rows():
    with open(DATA, newline="") as f:
        return list(csv.DictReader(f))


def _f(r, k):
    return float(r[k])


def test_the_row_count_is_the_documents_own(rows):
    assert len(rows) == N_EVENTS
    assert len({r["event"] for r in rows}) == N_EVENTS


def test_the_altitude_range_spans_15_to_35_kft(rows):
    alts = [v for r in rows
            for v in (_f(r, "altitude_kft_first"), _f(r, "altitude_kft_last"))]
    assert min(alts) == 15.0
    assert max(alts) == 35.0


def test_the_two_ranged_altitudes_survive(rows):
    """The exact defect that made a previous parse report 51 rows."""
    got = {r["event"]: (_f(r, "altitude_kft_first"), _f(r, "altitude_kft_last"))
           for r in rows if _f(r, "altitude_kft_first") != _f(r, "altitude_kft_last")}
    assert got == RANGED
    # 235-05 is a DESCENT and is printed high-to-low. A parser that sorted the
    # endpoints would lose that and nothing else would notice.
    assert got["235-05"][0] > got["235-05"][1]


def test_every_event_meets_the_documents_significance_criterion(rows):
    """Printed p. 7: significant turbulence is sigma_dn >= 0.2 g."""
    assert all(_f(r, "sigma_dn_g") >= SIGMA_SIGNIFICANT_G for r in rows)


def test_the_load_and_wind_extremes_straddle_zero(rows):
    for r in rows:
        assert _f(r, "dn_max_g") > 0 > _f(r, "dn_min_g"), r["event"]
        assert _f(r, "wind_up_ms") > 0 > _f(r, "wind_down_ms"), r["event"]


def test_the_weight_is_tabulated_for_every_row(rows):
    """The quantity whose absence makes PROJECT.md 5.19 impossible for Hannibal."""
    w = [_f(r, "weight_klb") for r in rows]
    assert len(w) == N_EVENTS
    assert 167.0 < min(w) and max(w) < 193.0


def test_reflectivity_sits_in_the_printed_band(rows):
    """Printed p. 7: peak values per event range from 0 to 40 dBz."""
    assert all(0.0 <= _f(r, "reflectivity_dbz") <= 40.0 for r in rows)


def test_only_three_rows_are_in_band_for_the_747(rows):
    """The admissibility split the design's S0 turns on.

    `boeing747.valid_altitude` is 35,000-45,000 ft, so `checks.recovery_band`
    would refuse every row below the floor. Three rows sit at 35 kft
    throughout; 232-05 STRADDLES the floor and is admissible on neither
    reading, which is a fourth category the 51-row parse could not have seen.
    """
    floor_kft = float(REGISTRY["boeing747"].valid_altitude[0]) / FT2M / 1000.0
    assert floor_kft == pytest.approx(35.0)

    lo = [min(_f(r, "altitude_kft_first"), _f(r, "altitude_kft_last")) for r in rows]
    hi = [max(_f(r, "altitude_kft_first"), _f(r, "altitude_kft_last")) for r in rows]
    events = [r["event"] for r in rows]

    in_band = {e for e, a in zip(events, lo) if a >= floor_kft}
    straddling = {e for e, a, b in zip(events, lo, hi) if a < floor_kft <= b}
    assert in_band == {"232-06", "232-08", "232-10"}
    assert straddling == {"232-05"}
    assert len(in_band) + len(straddling) < len(rows) // 2
