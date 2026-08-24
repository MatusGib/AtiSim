import math

from atisim import units


def test_length_and_force_are_exact_definitions():
    assert units.FT2M == 0.3048
    assert units.LB2KG == 0.45359237
    assert units.LBF2N == 4.4482216152605
    assert math.isclose(units.M2FT * units.FT2M, 1.0, rel_tol=1e-15)


def test_slug_is_lbf_s2_per_ft():
    # 1 slug = 14.5939 kg -- the value quoted in every standard table.
    assert math.isclose(units.SLUG2KG, 14.593902937206364, rel_tol=1e-12)


def test_slug_ft2_inertia():
    # 1 slug.ft^2 = 1.35581795 kg.m^2
    assert math.isclose(units.SLUG_FT2_TO_KG_M2, 1.3558179483314004, rel_tol=1e-12)


def test_angle_conversions():
    assert math.isclose(units.DEG2RAD * 180.0, math.pi, rel_tol=1e-15)
    assert math.isclose(units.RAD2DEG * math.pi, 180.0, rel_tol=1e-15)
    # A derivative of 0.1 per degree is 5.7296 per radian.
    assert math.isclose(0.1 * units.PER_DEG_TO_PER_RAD, 5.729577951308232, rel_tol=1e-12)


def test_knots():
    # 1 kt = 0.514444 m/s
    assert math.isclose(units.KT2MS, 0.5144444444444445, rel_tol=1e-12)


def test_density_conversion():
    # ISA sea-level density: 0.00237717 slug/ft^3 = 1.225 kg/m^3
    assert math.isclose(0.0023771 * units.SLUG_FT3_TO_KG_M3, 1.225, rel_tol=1e-4)
