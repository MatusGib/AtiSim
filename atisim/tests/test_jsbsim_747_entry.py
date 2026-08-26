"""`boeing747_jsbsim` against the frozen JSBSim B747 reference.

The entry is a transcription of scripts/gen_jsbsim_747.py's output. Nothing here
imports jsbsim; the reference XML is read directly, so a transcription slip or a
regenerated reference shows up as a test failure rather than as a quiet
disagreement inside a comparison.

This is NOT a validation of the aeroplane. B747.xml is release="ALPHA", author
"Unknown", and its lift curve is the 737's template -- see the entry's docstring.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from atisim.aircraft import CRUISE, REGISTRY
from atisim.units import FT2M, SLUG_FT2_TO_KG_M2

REFERENCE = Path(__file__).parent / "data" / "jsbsim_747_reference.xml"


@pytest.fixture(scope="module")
def reference():
    root = ET.parse(REFERENCE).getroot()
    return {
        "entry": {v.get("name"): float(v.text)
                  for v in root.findall("aircraft_entry/value")},
        "derivatives": {d.get("name"): float(d.get("value"))
                        for d in root.findall("derivatives/derivative")},
        "condition": root.find("condition"),
        "trim": {t.tag: float(t.text) for t in root.find("trim")},
    }


@pytest.fixture(scope="module")
def ac():
    return REGISTRY["boeing747_jsbsim"]


ENTRY_FIELDS = [
    "S", "b", "c", "CD0", "CD_alpha", "e", "CL0", "CLa", "CLde", "Cm0", "Cma",
    "Cmq", "Cmadot", "Cmde", "CYb", "Clb", "Clp", "Clr", "Clda", "Cldr", "Cnb",
    "Cnr", "Cnda", "Cndr", "max_thrust", "thrust_lapse", "mach_ram", "sweep",
    "t_over_c", "kappa_airfoil",
]


@pytest.mark.parametrize("field", ENTRY_FIELDS)
def test_entry_matches_the_frozen_reference(ac, reference, field):
    """Every transcribed number is the one the generator produced."""
    assert float(getattr(ac, field)) == pytest.approx(
        reference["entry"][field], rel=1e-12, abs=1e-15
    )


def test_mass_and_inertia_match_the_engine(ac, reference):
    condition = reference["condition"]
    assert float(ac.mass) == pytest.approx(
        float(condition.findtext("mass")), rel=1e-9)
    expected = np.array(
        [float(v) for v in condition.findtext("inertia").split()]
    ).reshape(3, 3)
    assert np.allclose(np.asarray(ac.inertia), expected, rtol=1e-9)


def test_cruise_altitude_is_density_matched_not_nominal(ac, reference):
    """A nominal altitude would bias every force in the comparison.

    The comparison must be flown where atisim's density equals the density
    JSBSim actually flew at, and CRUISE has to carry that.

    *** THE MATCH IS DERIVED HERE, NOT READ FROM THE FILE -- session 23. ***
    `matched_altitude_m` in the reference XML was solved against the atmosphere
    of the day it was generated, which took a geometric argument through
    geopotential formulas. Correcting that moved the answer 69 ft, so the frozen
    figure became an error of exactly the size it was introduced to remove. It
    is a property of ATISIM's atmosphere, not a JSBSim measurement, so it is
    recomputed from the frozen `density` -- which IS one -- exactly as
    `jsbsim_ref.load` now does.
    """
    from atisim.atmosphere import density
    from atisim.jsbsim_ref import _match_density

    condition = reference["condition"]
    nominal = float(condition.findtext("altitude_m"))
    rho = float(condition.findtext("density"))
    matched, residual = _match_density(rho, nominal)

    assert CRUISE["boeing747_jsbsim"]["altitude"] == pytest.approx(matched, rel=1e-12)
    assert residual < 1e-10
    # The point of the shift: at the matched altitude the densities agree.
    assert float(density(matched)) == pytest.approx(rho, rel=1e-12)

    # The shift is now SMALL, and that is the session-23 result rather than an
    # accident. Before the ISA fix it was 69.19 ft; what remains is the two
    # codes' differing ISA constants, well under a foot. Asserted as a band
    # rather than a point because it is a residual, not a target -- but bounded
    # in BOTH directions, so a regression that reinstated the old 69 ft error
    # fails here instead of passing a one-sided "small enough" check.
    shift_ft = (nominal - matched) / FT2M
    assert 0.0 < shift_ft < 1.0, (
        f"density match now needs {shift_ft:.2f} ft; before session 23 it "
        f"needed 69.19 ft, and anything approaching that means the "
        f"geometric/geopotential conversion has been lost")


def test_absent_derivatives_are_zero_because_b747_xml_defines_none(ac):
    """B747.xml defines no CLq, CYp, CYr, CYdr or Cnp.

    Carried as exact zeros. Cnda is NOT in this list -- B747.xml does define it,
    as exactly zero, which is a different fact and is asserted separately below.
    """
    for name in ("CLq", "CYp", "CYr", "CYdr", "Cnp"):
        assert float(getattr(ac, name)) == 0.0, name


def test_cnda_is_defined_and_zero_unlike_the_737(ac, reference):
    """The one derivative where the B747's coverage differs from the 737's."""
    assert "Cnda" in reference["derivatives"]
    assert float(ac.Cnda) == 0.0


def test_recovery_lands_on_b747_xmls_own_constants(reference):
    """The check that the recovery is a recovery and not a plausible fit.

    B747.xml states these as constants, and differencing the running engine has
    to return them. Cma is referred to the AERORP, which is why it comes back as
    the XML's -0.70 rather than the CG-referred value the engine applies.
    """
    d = reference["derivatives"]
    for name, stated in [
        ("CYb", -1.0), ("Clb", -0.10), ("Clp", -0.40), ("Clr", 0.15),
        ("Cldr", 0.01), ("Cnb", 0.12), ("Cnr", -0.15), ("Cndr", -0.10),
        ("CLde", 0.20), ("Cma", -0.70),
    ]:
        assert d[name] == pytest.approx(stated, abs=1e-4), name


def test_pitch_damping_sum_is_exact_even_though_the_split_is_not(reference):
    """Cmq and Cmadot are nearly collinear, so only their SUM is determined.

    The fit puts them at -21.0055 and -3.9945 against B747.xml's -21 and -4 --
    each 0.0055 out, equal and opposite, at a design condition number of 2.7e9.
    atisim needs the SUM for still-air damping and the split only for the wind
    term, so the well-determined quantity is the one carrying the weight. This
    test asserts the sum tightly and the split loosely, which is the honest way
    round.
    """
    d = reference["derivatives"]
    assert d["Cmq"] + d["Cmadot"] == pytest.approx(-25.0, abs=1e-6)
    assert d["Cmq"] == pytest.approx(-21.0, abs=0.02)
    assert d["Cmadot"] == pytest.approx(-4.0, abs=0.02)
    assert d["_pitch_fit_condition"] > 1e8  # the collinearity, recorded


def test_lift_curve_shares_the_737s_first_three_points(ac):
    """*** The reason this entry must never be compared to flight data. ***

    B747.xml's CLalpha table and 737.xml's agree on their first three points, so
    both give CLa = 4.3478 /rad. A 747 and a 737 do not share a lift-curve
    slope. This test exists so that the fact is asserted rather than buried in a
    docstring, and so it fails loudly if a future JSBSim fixes it.
    """
    b747 = REGISTRY["boeing747_jsbsim"]
    b737 = REGISTRY["boeing737"]
    assert np.allclose(np.asarray(b747.CL_table_alpha)[:3],
                       np.asarray(b737.CL_table_alpha)[:3])
    assert np.allclose(np.asarray(b747.CL_table_CL)[:3],
                       np.asarray(b737.CL_table_CL)[:3])
    assert float(b747.CLa) == pytest.approx(float(b737.CLa), rel=1e-12)
    # The fourth point is where they finally differ.
    assert float(b747.CL_table_alpha[3]) != float(b737.CL_table_alpha[3])


def test_it_is_not_the_cr2144_747(ac):
    """`boeing747_jsbsim` and `boeing747` are different aeroplanes, on purpose.

    If these ever converge, the reason this entry exists has gone away and the
    comparison should be re-thought rather than quietly kept.
    """
    cr2144 = REGISTRY["boeing747"]
    assert float(ac.mass) / float(cr2144.mass) == pytest.approx(0.866, abs=0.005)
    assert float(ac.b) / float(cr2144.b) == pytest.approx(1.081, abs=0.005)
    assert float(ac.S) / float(cr2144.S) == pytest.approx(1.027, abs=0.005)
    # The inertia tensors, however, ARE the same -- which is what makes the mass
    # difference incoherent as a physical aeroplane and irrelevant as a
    # cross-code subject.
    assert np.allclose(np.asarray(ac.inertia), np.asarray(cr2144.inertia),
                       rtol=2e-3)


def test_it_trims_well_inside_a_table_segment(ac):
    """jacfwd at a knot returns a one-sided slope.

    aircraft.RECOVERED_FROM_JSBSIM's comment states this as a property of
    membership, so it is asserted here rather than left as a claim.
    test_aero.test_the_lift_table_is_not_linearised_at_a_breakpoint makes the
    same check for the two 737 entries; it cannot cover this one, because it
    trims through test_jsbsim_737_layers._atisim_trim, which is 737-specific.

    The margin here is large: the entry trims near alpha 4.33 deg and the
    nearest knot is at 0.00 rad, so there is 4.33 deg of room -- against the
    737 cruise entry's 1.98 deg, which is the tighter of the two.
    """
    import jax.numpy as jnp

    from atisim import trim as trim_mod

    c = CRUISE["boeing747_jsbsim"]
    x, _ = trim_mod.trim(jnp.array(c["airspeed"]), jnp.array(c["altitude"]), ac)
    alpha = float(x[0])
    knots = np.asarray(ac.CL_table_alpha)
    gap = float(np.min(np.abs(knots - alpha)))
    assert np.degrees(gap) > 1.0, (
        f"trims at alpha {np.degrees(alpha):.3f} deg, only "
        f"{np.degrees(gap):.3f} deg from a table breakpoint"
    )


def test_it_carries_a_band_because_it_is_a_fit(ac):
    """Only an entry that IS a local fit may declare a recovery band.

    test_aircraft.test_a_recovery_band_is_declared_only_where_one_was_measured
    enforces the converse across the whole registry. This entry qualifies for
    the same reason the 737 entries do: it is a linearisation of a nonlinear
    model about one point, not a linear derivative set from a document.
    """
    assert float(ac.valid_altitude[1]) > float(ac.valid_altitude[0])
    assert float(ac.valid_mach[1]) > float(ac.valid_mach[0]) > 0.0


def test_validity_band_brackets_both_vortex_cases(ac):
    """The band exists so a run outside it reports as unchecked, not as passing.

    Hannibal is at 37,000 ft and Morton at 39,000; the entry was recovered at
    38,000 between them, so neither is extrapolated to.
    """
    from atisim.wind import WINGROVE_CASE_ALTITUDE

    low, high = (float(v) for v in ac.valid_altitude)
    assert low < WINGROVE_CASE_ALTITUDE["hannibal"] < high
    assert low < WINGROVE_CASE_ALTITUDE["morton"] < high
    # Cimarron is at 33,000 ft and is NOT in this band -- it is the 737's case.
    assert WINGROVE_CASE_ALTITUDE["cimarron"] < low


def test_entry_reproduces_jsbsims_own_trim_lift(ac, reference):
    """Fed JSBSim's trim state, atisim must return JSBSim's lift coefficient.

    This is the end-to-end check that the entry is faithful: not that the
    numbers were transcribed, but that running them through atisim's own aero
    path at the point they were recovered from returns what JSBSim returned.

    Cm is deliberately NOT checked here. JSBSim's trim state carries a small
    non-zero alphadot, and Cm depends on it through Cmadot; supplying zero moves
    Cm by 1.5e-4, which is larger than anything this test could resolve. The
    alphadot-carrying comparison belongs in the layered vortex comparison, where
    the rate is read from the reference rather than assumed.
    """
    import jax.numpy as jnp

    from atisim import aero, trim as trim_mod
    from atisim.atmosphere import speed_of_sound

    c = CRUISE["boeing747_jsbsim"]
    alpha, de = reference["trim"]["alpha"], reference["trim"]["elevator"]
    V = c["airspeed"]
    vel = jnp.array([V * np.cos(alpha), 0.0, V * np.sin(alpha)])
    CL, *_ = aero.coefficients(
        vel, jnp.zeros(3),
        trim_mod.trimmed_controls(jnp.array(de), jnp.array(0.63)),
        ac, speed_of_sound(jnp.array(c["altitude"])),
    )
    # atisim evaluates the TABLE, not CL0 + CLa*alpha, so the comparison has to
    # be against the table's own segment: intercept exactly 0.20 and slope
    # (1.20 - 0.20) / 0.23. The recovered CL0 is 0.1999999908, which differs
    # from the table's intercept by 9.2e-9 -- the same kind of gap the 737 entry
    # records as 5.5e-9, and the reason a 1e-9 bound against CL0 fails a correct
    # entry.
    e = reference["entry"]
    table = 0.20 + ((1.20 - 0.20) / 0.23) * alpha + e["CLde"] * de
    assert float(CL) == pytest.approx(table, rel=1e-12)
    # And the linear form agrees with the table to that documented 9.2e-9.
    linear = e["CL0"] + e["CLa"] * alpha + e["CLde"] * de
    assert abs(table - linear) < 1e-8
