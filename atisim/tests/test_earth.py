"""WGS-84 geodesy, gravity, and the rotating-Earth frames.

Every constant here was recovered from the running JSBSim binary, not quoted
from a table -- see the design doc section 2. The tests below assert the
DEFINING values; the term-by-term agreement with JSBSim is asserted separately
against the frozen reference.
"""

import numpy as np
import pytest

import atisim  # noqa: F401  -- enables x64 before any array is made
from atisim import earth


def test_the_defining_wgs84_constants_are_exact():
    """a and f are DEFINING; b and e2 are derived from them and never quoted.

    JSBSim reports a semi-minor axis of 6356752.314186 m, which differs from
    a(1-f) by 5.87e-5 m. That is JSBSim's internal storage in feet
    round-tripping, not a different ellipsoid, so this model carries the
    defining pair and derives the rest. Asserted here so nobody "fixes" b to
    JSBSim's reported value later.
    """
    assert earth.A_WGS84 == 6378137.0
    assert earth.F_WGS84 == 1.0 / 298.257223563
    assert earth.GM_WGS84 == 3.986004418e14
    assert earth.J2_WGS84 == 1.08262982e-3
    assert earth.OMEGA_WGS84 == 7.292115e-5

    assert earth.B_WGS84 == pytest.approx(6356752.314245179, abs=1e-6)
    assert earth.E2_WGS84 == pytest.approx(earth.F_WGS84 * (2.0 - earth.F_WGS84), rel=1e-15)
    # The recorded difference against JSBSim's reported value, so a future
    # reader meets it as a number rather than as a surprise.
    assert abs(earth.B_WGS84 - 6356752.314186481) == pytest.approx(5.87e-5, rel=0.02)
