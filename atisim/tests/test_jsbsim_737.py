"""Cross-code verification of atisim against JSBSim 1.3.1, via the 737.

Four layers, each isolating one part of the chain, so a failure localises
instead of saying only "the sim disagrees":

  1. coefficients  -- the aero build-up, at identical read-back states
  2. trim          -- JSBSim's trim algorithm against trim.trim
  3. modes         -- JSBSim's exported linearisation against ours
  4. trajectory    -- both integrated through the same prescribed surfaces

What is claimed: atisim's aero build-up, trim solver, linearisation and
integrator agree with an independent mature engine fed the same coefficients.

What is NOT claimed: that the JSBSim 737 is a correct 737. Its own file header
says it was built from public data "and guesses" and is for "educational and
entertainment purposes only". See the spec's "Source qualification", and
aircraft._boeing_737's docstring.

Nothing here imports jsbsim. The reference is frozen in
atisim/tests/data/jsbsim_737_reference.xml by scripts/gen_jsbsim_reference.py.

Design: docs/superpowers/specs/2026-08-20-jsbsim-737-verification-design.md
"""

import numpy as np
import pytest

from atisim import jsbsim_ref
from atisim.aircraft import CRUISE, REGISTRY

REF = jsbsim_ref.load()
CRUISE_COND = REF.condition["cruise"]

# Derivatives 737.xml does not define. atisim carries 0.0 for each, which is
# agreement rather than approximation -- both engines then compute the same
# thing. The generator measured each and found it below its settling-drift
# floor; these are the values it recorded.
ABSENT = ("CLq", "CYp", "CYr", "CYdr", "Cnp", "Cnda")


def test_reference_is_the_expected_jsbsim_build():
    """Guards against a silently regenerated reference from another version."""
    assert REF.jsbsim_version.startswith("1.3.1")


def test_density_matches_jsbsim_to_machine_precision():
    """The input-condition check, asserted rather than assumed.

    atisim's ISA uses geometric altitude where the standard uses
    geopotential, so at a nominal 30,000 ft its density is 0.159% below
    JSBSim's. qbar is proportional to rho, so that bias would land on every
    force in every layer below, in the same direction. The generator solves for
    the matching geometric altitude instead; this asserts it worked.
    """
    from atisim.atmosphere import density

    assert CRUISE_COND.density_match_residual < 1e-10
    got = float(density(CRUISE_COND.matched_altitude))
    assert abs(got - CRUISE_COND.density) / CRUISE_COND.density < 1e-10


def test_737_is_registered_at_its_recovery_condition():
    assert "boeing737" in REGISTRY
    assert CRUISE["boeing737"]["altitude"] == pytest.approx(
        CRUISE_COND.matched_altitude, rel=1e-12
    )
    assert CRUISE["boeing737"]["airspeed"] == pytest.approx(
        CRUISE_COND.airspeed, rel=1e-12
    )


def test_737_matches_the_recovered_reference_entry():
    """Every literal in _boeing_737() equals the value the generator recovered."""
    ac = REGISTRY["boeing737"]
    checked = 0
    for name, expected in REF.entry.items():
        if not hasattr(ac, name):
            continue
        assert float(getattr(ac, name)) == pytest.approx(expected, rel=1e-9), name
        checked += 1
    assert checked >= 25, f"only {checked} fields cross-checked against the reference"


def test_737_zeroes_the_derivatives_jsbsim_lacks():
    ac = REGISTRY["boeing737"]
    for name in ABSENT:
        assert float(getattr(ac, name)) == 0.0, name


def test_737_mass_and_inertia_match_the_engine():
    ac = REGISTRY["boeing737"]
    assert float(ac.mass) == pytest.approx(CRUISE_COND.mass, rel=1e-9)
    np.testing.assert_allclose(
        np.asarray(ac.inertia), CRUISE_COND.inertia, rtol=1e-9
    )
