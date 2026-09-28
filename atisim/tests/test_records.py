"""The published record a run is read against: which record, and where it sits in time.

Data: `atisim/data/parks1985_fig6_altitude.csv`, the Hannibal DC-10 record of
Parks et al. 1985 Fig. 6, and the published bands in `wind` and `vortex_viz`.
"""

import json

import numpy as np
import pytest

from atisim import records
from atisim.vortex_viz import FIG8_LOAD_BAND
from atisim.wind import TM102186_HANNIBAL_NZ

T = np.linspace(0.0, 60.0, 3001)
W = -20.0 * np.exp(-((T - 30.0) / 2.0) ** 2)  # one downdraft, deepest at 30 s


def test_each_field_finds_its_record_and_other_fields_find_none():
    assert records.category("VortexArray", "hannibal") == "hannibal"
    assert records.category("VortexArray", "mehta") == "hannibal"
    assert records.category("VortexArray", "morton") == "vortex"
    assert records.category("UpdraftColumn") == "updraft"
    assert records.category("manoeuvre") == "manoeuvre"
    # Negative controls: no source measured these, so nothing is drawn.
    assert records.category("Dryden") is None
    assert records.category("none (zero wind)") is None
    assert records.for_run("Dryden", None, T, W) is None


def test_the_trace_is_moved_so_the_two_downdrafts_coincide():
    trace = records.hannibal_trace(T, W)
    assert trace.shift == pytest.approx(30.0 - records._deepest_downdraft_s(), abs=0.05)
    assert T[0] <= trace.t.min() and trace.t.max() <= T[-1]
    # The band keeps the extremes the paper quotes, "+1.7 to -1.0 gs" (p. 127).
    assert trace.lo.min() < -0.9 and trace.hi.max() > 1.7


def test_a_run_that_met_no_downdraft_gets_no_trace():
    assert records.hannibal_trace(T, np.zeros_like(T)) is None


def test_hannibal_carries_the_measured_band_and_the_others_the_fig8_band():
    rec = records.for_run("VortexArray", "hannibal", T, W)
    assert rec.trace is not None
    assert (rec.bands[0].lo, rec.bands[0].hi) == TM102186_HANNIBAL_NZ
    other = records.for_run("UpdraftColumn", None, T, W)
    assert other.trace is None
    # Read as an increment from level flight, as the manoeuvre preset reads it.
    assert other.bands[0].lo == pytest.approx(1.0 + FIG8_LOAD_BAND[0])
    assert other.bands[0].hi == pytest.approx(1.0 + FIG8_LOAD_BAND[1])


def test_the_record_is_json_ready_for_the_browser():
    out = records.as_dict(records.for_run("VortexArray", "hannibal", T, W))
    json.dumps(out)
    assert len(out["trace"]["t"]) == len(out["trace"]["lo"]) == len(out["trace"]["hi"])
