"""Stewart's B-757 normal-force table, pinned so a re-extraction cannot drift.

Data: `atisim/data/stewart_tm2003_212666_table1.csv`, written by
`scripts/stewart_table1_ingest.py` from NASA/TM-2003-212666 (Stewart, December
2003, NTRS 20040021314) Table 1, printed pp. 12-13, text layer.

Why the shared dynamic-pressure grid is pinned: the table prints that grid
THREE times, once per coefficient, and a misparse of any one of them breaks the
agreement. The first parse of this table read a printed PAGE NUMBER as a cell,
because the 40 kft block straddles the page break -- `test_the_grid_is_shared`
is what catches that class of error, not the ranges.

This is a comparison target, never a model input: the aircraft is a B-757 and
nothing in `atisim` is one. Rule 6 asks for bands and orderings; the bands here
are the document's own printed numbers, so they are pinned as printed.
"""

import csv
import math
from pathlib import Path

import pytest

import atisim

DATA = Path(atisim.__file__).parent / "data" / "stewart_tm2003_212666_table1.csv"

COEFFS = ("CNalpha", "CNq", "CNdelta")
ALTITUDES_KFT = (0.0, 10.0, 20.0, 30.0, 40.0)
N_ROWS = 72


@pytest.fixture(scope="module")
def rows():
    with open(DATA, newline="") as f:
        return [{**r, "altitude_kft": float(r["altitude_kft"]),
                 "q_psf": float(r["q_psf"]), "value": float(r["value"]),
                 "mach": float(r["mach"])} for r in csv.DictReader(f)]


def test_the_shape_is_three_coefficients_over_five_altitudes(rows):
    assert len(rows) == N_ROWS
    assert {r["coefficient"] for r in rows} == set(COEFFS)
    assert {r["altitude_kft"] for r in rows} == set(ALTITUDES_KFT)


def test_the_grid_is_shared(rows):
    """The table prints one dynamic-pressure grid per coefficient, three times."""
    for alt in ALTITUDES_KFT:
        grids = [tuple(sorted(r["q_psf"] for r in rows
                              if r["coefficient"] == c and r["altitude_kft"] == alt))
                 for c in COEFFS]
        assert grids[0] == grids[1] == grids[2], alt
        assert len(grids[0]) == (4 if alt == 40.0 else 5), alt


def test_every_column_lands_at_a_plausible_mach(rows):
    """Mach is reconstructed from q and altitude through the paper's Eq. (8)."""
    assert all(0.2 < r["mach"] < 0.95 for r in rows)


def test_mach_rises_with_dynamic_pressure_at_fixed_altitude(rows):
    """An ordering, per rule 6 -- and the confound the comparison must carry."""
    for alt in ALTITUDES_KFT:
        got = sorted((r["q_psf"], r["mach"]) for r in rows
                     if r["coefficient"] == "CNalpha" and r["altitude_kft"] == alt)
        machs = [m for _, m in got]
        assert machs == sorted(machs), alt


def test_the_coefficient_extremes_are_the_printed_ones(rows):
    """Pinned EXACTLY, because a text-layer extract has no reading uncertainty.

    An earlier draft of this test asserted loose bands guessed from the first
    three altitude blocks, and CNdelta's 30 and 40 kft rows fall outside them
    (0.354756 low, 0.584220 high). The bands were wrong, not the data -- so
    they are replaced by the document's own extremes rather than widened,
    which is the distinction docs/DEVELOPMENT.md rule 3 asks to be shown at
    the change.
    """
    printed = {"CNalpha": (4.847166, 7.437156),
               "CNq": (3.629151, 8.536046),
               "CNdelta": (0.354756, 0.584220)}
    for coeff, (lo, hi) in printed.items():
        vals = [r["value"] for r in rows if r["coefficient"] == coeff]
        assert min(vals) == pytest.approx(lo), coeff
        assert max(vals) == pytest.approx(hi), coeff


def test_the_lift_curve_slope_rises_with_mach_overall(rows):
    """The ordering that makes this table a compressibility check at all.

    Stated as an ordering between the extreme Mach columns rather than as a fit,
    because a column mixes Mach with trim angle of attack -- see the provenance
    entry. The 40 kft M 0.642 column sits BELOW every other altitude's low-Mach
    value, so the claim is made on the overall extremes, not per altitude.
    """
    a = [r for r in rows if r["coefficient"] == "CNalpha"]
    lo = min(a, key=lambda r: r["mach"])
    hi = max(a, key=lambda r: r["mach"])
    assert hi["value"] > lo["value"]
    # Prandtl-Glauert over the same Mach span, for the record rather than as a gate.
    pg = math.sqrt(1 - lo["mach"] ** 2) / math.sqrt(1 - hi["mach"] ** 2)
    assert pg > 1.0
