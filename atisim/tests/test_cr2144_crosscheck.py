"""Two independent digitisations of CR-2144 pp. 220-222, against each other.

The band is the automated trace's OWN stated accuracy, 2% of full scale
(Reference_papers/CR-2144/README.md), fixed before the comparison was run and
sealed as `cr2144_two_readings_agree_on_the_good_panels` in atisim/predictions.py.
It is asserted only on the four panels that README rates 'good'.

If it fails, that is a finding for docs/PROJECT.md section 4. The band is NOT to
be widened (docs/DEVELOPMENT.md rule 3); mark the failing case
pytest.mark.xfail(strict=True) citing the section-4 entry instead.
"""
import pytest

from atisim import cr2144_mach

GOOD = ("cl_alpha", "cd_alpha", "cm_alpha", "cm_m")   # README confidence: good
BAND_PCT_FS = 2.0


@pytest.fixture(scope="module")
def rows():
    return cr2144_mach.crosscheck()


def test_every_hand_curve_finds_its_automated_counterpart(rows):
    compared = {(r.quantity, r.altitude) for r in rows}
    assert compared == set(cr2144_mach.curves()) - {("cm_q", a) for a in ("SL", "20K", "40K")}


# cm_m FAILED the sealed band, session 32: 2.40% of full scale at 20K and 2.02% at
# 40K, all three altitudes biased the same way. Scored against Table IX-4 at the
# circled conditions it is the AUTOMATED trace that is off -- RMS 0.0146 against
# the hand reading's 0.0063 -- so the hand reading the shipped 747 declares Cm_M
# from is the better-anchored one. PROJECT.md section 4, "Two readings of
# CR-2144". Marked strict so that a corrected automated trace shows up as XPASS
# rather than going unnoticed; the band is not widened (docs/DEVELOPMENT.md rule 3).
CM_M_FAILS = pytest.mark.xfail(strict=True, reason=(
    "measured session 32: the automated Cm_M trace is biased, 2.40% FS at 20K; "
    "PROJECT.md section 4"))


@pytest.mark.parametrize("quantity", [
    q if q != "cm_m" else pytest.param(q, marks=CM_M_FAILS) for q in GOOD])
def test_the_two_readings_agree_on_the_good_panels(rows, quantity):
    for r in (r for r in rows if r.quantity == quantity):
        assert r.median_pct_fs <= BAND_PCT_FS, (
            f"{quantity} {r.altitude}: median disagreement {r.median_pct_fs:.2f}% "
            f"of full scale over {r.n} points")


@pytest.mark.parametrize("quantity", GOOD)
def test_each_good_curve_sits_on_its_own_altitude(rows, quantity):
    for r in (r for r in rows if r.quantity == quantity):
        assert r.best_altitude == r.altitude, (
            f"{quantity}: the hand {r.altitude} points sit closest to the "
            f"automated {r.best_altitude} curve")
