"""TPAWS Figure 133's measured gust rms, pinned.

Data: `atisim/data/tpaws_fig133_sigma_uw.csv`, written by
`scripts/digitise_tpaws_fig133.py` from NASA/TM-2012-217337 Figure 133, printed
p. 128. The figure is VECTOR art, so the markers' coordinates are stated by the
PDF rather than read off a raster; the axes are calibrated on the printed tick
labels' own boxes, residual 1.3e-3 data units.

The load-bearing test here is `test_the_anisotropy_claim_holds`. A range check
cannot tell sigma_u from sigma_w -- both span a similar band -- so a swapped
axis assignment would pass every bound in this file. The paper states the
asymmetry itself, and that statement is what pins the orientation.
"""

import csv
from pathlib import Path

import pytest

import atisim

DATA = Path(atisim.__file__).parent / "data" / "tpaws_fig133_sigma_uw.csv"
N_EVENTS = 78


@pytest.fixture(scope="module")
def rows():
    with open(DATA, newline="") as f:
        return [(float(r["sigma_u_ms"]), float(r["sigma_w_ms"]))
                for r in csv.DictReader(f)]


def test_the_event_count_is_the_figures_own(rows):
    assert len(rows) == N_EVENTS


def test_every_point_is_inside_the_printed_box(rows):
    assert all(0.0 <= u <= 10.0 and 0.0 <= w <= 10.0 for u, w in rows)


def test_the_anisotropy_claim_holds(rows):
    """Printed pp. 127-128: isotropy would give a 1:1 relationship, and the
    figure instead "shows a bias towards higher values" of sigma_w.

    Asserted as an ordering on the population, per rule 6 -- not as a value.
    This is the only test here that would catch the two axes being swapped.
    """
    above = sum(1 for u, w in rows if w > u)
    assert above > len(rows) * 0.6, f"only {above}/{len(rows)} above 1:1"
    assert sum(w for _, w in rows) > sum(u for u, _ in rows)


def test_the_bands_are_the_digitised_ones(rows):
    us = [u for u, _ in rows]
    ws = [w for _, w in rows]
    assert min(ws) == pytest.approx(2.14, abs=0.01)
    assert max(ws) == pytest.approx(8.62, abs=0.01)
    assert min(us) == pytest.approx(1.67, abs=0.01)
    assert max(us) == pytest.approx(7.60, abs=0.01)


def test_the_population_is_not_table_1s(rows):
    """78 here against Table 1's 53, and the mismatch governs every use.

    Figure 133 is ALL 2002 events; Table 1 is the SIGNIFICANT events across both
    campaigns, 49 of which are 2002. Any ratio of the two is bracketed rather
    than computed, so this asserts the two files really do disagree in size --
    a future re-extraction that silently made them match would be wrong.
    """
    table1 = Path(atisim.__file__).parent / "data" / "tpaws_tm2012_217337_table1.csv"
    with open(table1, newline="") as f:
        t1 = list(csv.DictReader(f))
    assert len(rows) != len(t1)
    y2002 = [r for r in t1 if not r["event"].startswith(("190", "191"))]
    assert len(y2002) == 49  # the paper's own count, printed p. 5
    assert len(rows) > len(y2002)
