import pytest

from atisim import atmosphere as atm

# ISA table values. Pressure/density to 5 significant figures.
#   altitude m, T K,     p Pa,      rho kg/m^3
ISA_TABLE = [
    (0.0, 288.15, 101325.0, 1.2250),
    (5000.0, 255.65, 54019.9, 0.73612),
    (11000.0, 216.65, 22632.0, 0.36392),
    (15000.0, 216.65, 12044.6, 0.19367),
    (20000.0, 216.65, 5474.9, 0.08803),
]


@pytest.mark.parametrize("h,t,p,rho", ISA_TABLE)
def test_isa_against_table(h, t, p, rho):
    assert atm.temperature(h) == pytest.approx(t, rel=1e-5)
    assert atm.pressure(h) == pytest.approx(p, rel=1e-4)
    assert atm.density(h) == pytest.approx(rho, rel=1e-4)


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
