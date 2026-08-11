"""Kept as an import shim.

The linearisation moved to `flightsim.validation` so the notebook and the
validation module can reach the plant matrix itself, not only the modes derived
from it. `test_cr2144_modes.py` and `test_navion.py` import from here and are
deliberately untouched.
"""

from flightsim.validation import lateral_modes, longitudinal_modes  # noqa: F401
