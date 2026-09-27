"""The Lab's content: which values a case offers, how an edit keeps provenance
honest, which analyses it runs, and what a flown result says. No Dash here;
the pages are checked in test_app.py."""

import math

import numpy as np
import pytest

from atisim import analyses, lab, run


def test_every_preset_offers_its_key_values_and_they_are_real_parameters():
    for name, spec in run.PRESETS.items():
        names = {p.name for p in run.PARAMETERS.get(spec.wind.kind, ())}
        for p in lab.key_params(spec):
            assert p.name in names, (name, p.name)
    assert [p.name for p in lab.key_params(run.PRESETS["vortex-hannibal"])] == ["v0", "r0"]
    assert [p.name for p in lab.key_params(run.PRESETS["dryden"])] == ["sigma_w", "seed"]


def test_an_untouched_form_flies_the_preset_exactly():
    for name, spec in run.PRESETS.items():
        values = {p.name: run.param_value(spec.wind, p) for p in lab.key_params(spec)}
        assert lab.edit(spec, spec.aircraft, spec.airspeed_mps, spec.altitude_m,
                        values) == spec, name


def test_an_edited_value_is_declared_and_says_what_the_source_had():
    base = run.PRESETS["vortex-hannibal"]
    spec = lab.edit(base, wind_values={"v0": 30.0}, airspeed=230.0)
    prov = run.provenance(spec)
    assert prov["v0"].status == "declared" and "25.908" in prov["v0"].note
    assert prov["airspeed_mps"].status == "declared"
    assert prov["r0"].status == "sourced"
    back = lab.edit(spec, wind_values={"v0": base.wind.params["v0"]})
    assert run.provenance(back)["v0"].status == "sourced"


def test_the_lab_analyses_are_real_and_each_says_what_it_teaches():
    assert [a.key for a in lab.ANALYSES] == ["ensemble", "modes", "convergence", "cross-code"]
    for a in lab.ANALYSES:
        assert a.key in analyses.ANALYSES and a.learn.endswith(".")


def test_a_summary_reads_the_peak_the_low_and_the_band():
    t = np.linspace(0.0, 10.0, 101)
    n_z = 1.0 + 0.8 * np.sin(t)
    checks = [{"name": "rms normal load", "kind": "report", "passed": None, "value": 0.42,
               "detail": ""},
              {"name": "energy closure", "kind": "gate", "passed": True, "value": 1e-4,
               "detail": ""}]
    s = lab.summarise(t, n_z, checks)
    assert s.peak == pytest.approx(1.8, abs=1e-3) and s.t_peak == pytest.approx(1.6, abs=0.05)
    assert s.low == pytest.approx(0.2, abs=1e-3)
    assert (s.severity, s.verdict) == ("severe", "pass")
    quiet = lab.summarise(t, n_z, [])
    assert math.isnan(quiet.sigma) and quiet.severity == "too short to rate"
