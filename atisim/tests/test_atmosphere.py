import pytest

from atisim import atmosphere as atm

# ISA table values. Pressure/density to 5 significant figures.
#
# *** THE TABLE'S ALTITUDE COLUMN IS GEOPOTENTIAL. *** That is how the standard
# atmosphere is published and it is not a choice this file gets to make. Since
# session 23 `atmosphere` takes GEOMETRIC altitude and converts internally, so
# each row must be entered at the geometric height whose geopotential height is
# the tabulated one -- the inverse conversion `z = R H / (R - H)`.
#
# Before session 23 these rows were passed straight in, which silently asserted
# that the module's argument was geopotential. It was, inside this file; every
# caller in the package passed geometric. The conversion is what closed that,
# and rewriting the rows through `_geometric` is what makes this test check it:
# a regression that dropped the conversion would put 20 km out by 0.31% and fail
# the density row at rel=1e-4.
#   altitude m (GEOPOTENTIAL), T K,     p Pa,      rho kg/m^3
ISA_TABLE = [
    (0.0, 288.15, 101325.0, 1.2250),
    (5000.0, 255.65, 54019.9, 0.73612),
    (11000.0, 216.65, 22632.0, 0.36392),
    (15000.0, 216.65, 12044.6, 0.19367),
    (20000.0, 216.65, 5474.9, 0.08803),
]


def _geometric(h_geopotential):
    """Invert `atmosphere.geopotential`, so a table row can be flown at."""
    R = atm.R_EARTH_ISA
    return R * h_geopotential / (R - h_geopotential)


@pytest.mark.parametrize("h,t,p,rho", ISA_TABLE)
def test_isa_against_table(h, t, p, rho):
    z = _geometric(h)
    assert atm.geopotential(z) == pytest.approx(h, rel=1e-12), (
        "the inverse conversion must round-trip, or this test is measuring "
        "its own arithmetic rather than the atmosphere")
    assert atm.temperature(z) == pytest.approx(t, rel=1e-5)
    assert atm.pressure(z) == pytest.approx(p, rel=1e-4)
    assert atm.density(z) == pytest.approx(rho, rel=1e-4)


def test_the_geopotential_conversion_is_applied_and_is_worth_what_project_md_says():
    """Negative control on the whole session-23 change.

    If `geopotential` were ever reduced to the identity, this fails -- rather
    than the module quietly reverting to the behaviour nothing detected for 22
    sessions, which was to take a geometric argument through geopotential
    formulas.

    `density(_geometric(z))` reconstructs the OLD behaviour exactly: it forces
    the internal conversion to land back on `z`, so the formulas are evaluated
    at `z` itself, which is what the pre-session-23 code did with a geometric
    argument.
    """
    assert atm.geopotential(0.0) == 0.0                 # exact at the datum
    for z in (5000.0, 11000.0, 20000.0):
        assert atm.geopotential(z) < z                  # always REDUCES the height

    # The two figures PROJECT.md §4 quotes for the bias this removes.
    for z_ft, expected_pct in ((30000.0, 0.159), (40000.0, 0.368)):
        z = z_ft * 0.3048
        fixed, old = float(atm.density(z)), float(atm.density(_geometric(z)))
        bias_pct = (fixed / old - 1.0) * 100.0
        assert bias_pct == pytest.approx(expected_pct, abs=0.005), (
            f"at {z_ft:.0f} ft the conversion should raise density by "
            f"{expected_pct}%; measured {bias_pct:+.3f}%")


def test_sea_level_speed_of_sound():
    assert atm.speed_of_sound(0.0) == pytest.approx(340.294, rel=1e-5)


def test_continuous_across_tropopause():
    below = atm.pressure(11000.0 - 1e-6)
    above = atm.pressure(11000.0 + 1e-6)
    assert below == pytest.approx(above, rel=1e-9)
    assert atm.density(10999.0) == pytest.approx(atm.density(11001.0), rel=1e-3)


def test_density_decreases_monotonically():
    hs = [0.0, 1000.0, 5000.0, 11000.0, 15000.0, 20000.0]
    rhos = [float(atm.density(h)) for h in hs]
    assert all(a > b for a, b in zip(rhos, rhos[1:]))
