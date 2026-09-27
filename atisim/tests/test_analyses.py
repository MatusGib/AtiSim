"""The analyses: every simulation that is not one open-loop flight.

`atisim.analyses` runs the engine's own functions and writes a report; these
tests pin that each analysis runs end to end at small size, that the spec
round-trips and validates the way `RunSpec` does, that an ensemble's members are
ordinary runs, and that the CLI, the job runner and the pages reach them all.
"""

import json
import time

import numpy as np
import pytest

import atisim  # noqa: F401
from atisim import analyses, run

pytest.importorskip("plotly", reason="an analysis draws figures: the `ui` extra")
pytest.importorskip("pyarrow", reason="an analysis writes runs: the `ui` extra")

from atisim.analysis import artifact, report  # noqa: E402
from atisim.analysis import runs as runs_mod  # noqa: E402

# Small enough for the suite, large enough that every code path runs: two
# members, two wavelengths, two refinements.
SHORT = run.PRESETS["updraft"]._replace(dt=0.05, seconds=12.0)
SMALL = {
    "ensemble": ({"members": 2}, run.PRESETS["dryden"]._replace(seconds=6.0)),
    "step-response": ({"seconds": 10.0}, None),
    "closed-loop": ({}, SHORT),
    "trim": ({}, None),
    "modes": ({}, None),
    "mode-sensitivity": ({}, None),
    "coefficient-sweep": ({"points": 3}, None),
    "load-sensitivity": ({}, SHORT),
    "gust-transfer": ({"points": 2, "wavelength_min": 2000.0, "wavelength_max": 8000.0},
                      None),
    "pratt-walker": ({"gradients": "12.5"}, None),
    "convergence": ({"refinements": 2}, SHORT._replace(dt=0.02, seconds=6.0)),
    "cross-code": ({"arm": "translational"}, None),
    "verification": ({}, None),
    "study": ({}, None),  # scripts/sanity.py
}


def _small(key):
    params, base = SMALL[key]
    aspec = analyses.default(key, base)
    return aspec._replace(params={**aspec.params, **params})


def test_every_analysis_has_a_small_case_here():
    """A new analysis must be added to SMALL, so the test below runs it."""
    assert set(SMALL) == set(analyses.ANALYSES)


@pytest.mark.parametrize("key", sorted(SMALL))
def test_every_analysis_runs_and_writes_a_readable_report(key, tmp_path):
    aspec = _small(key)
    assert not [i for i in analyses.validate(aspec) if i.level == "error"], key
    stages = []
    path = analyses.perform(aspec, tmp_path, on_stage=stages.append)
    loaded = report.read_report(path)
    assert loaded.meta["result_kind"] == "report"
    assert loaded.meta["analysis"] == key
    assert loaded.meta["summary"]
    assert loaded.sections, key
    assert stages[-1] == "writing"
    # Every figure is Plotly JSON and every table is rectangular.
    for section in loaded.sections:
        if section["figure"] is not None:
            assert "data" in section["figure"] and "layout" in section["figure"]
        if section["table"] is not None:
            width = len(section["table"]["headers"])
            assert all(len(row) == width for row in section["table"]["rows"])
    # The spec in the report runs the same analysis again.
    again = analyses.AnalysisSpec.from_json(loaded.meta["spec"])
    assert again == aspec
    row = runs_mod.read_row(path)
    assert row.verdict == "report" and row.error is None


# --- the spec -------------------------------------------------------------


def test_the_spec_round_trips_with_and_without_a_base():
    for aspec in (analyses.default("modes"),
                  analyses.default("ensemble", run.PRESETS["dryden"])):
        assert analyses.AnalysisSpec.from_json(aspec.to_json()) == aspec
    with pytest.raises(ValueError, match="unknown AnalysisSpec fields"):
        analyses.AnalysisSpec.from_json({"name": "x", "analysis": "modes", "params": {},
                                         "colour": 1})


def test_an_analysis_of_a_condition_carries_no_base():
    """The trim of an aircraft does not depend on any run on screen."""
    assert analyses.default("modes", run.PRESETS["dryden"]).base is None


def test_validation_names_the_field():
    bad = analyses.default("modes")._replace(params={"aircraft": "concorde"})
    assert [i.field for i in analyses.validate(bad)] == ["aircraft"]
    bad = analyses.default("modes")._replace(params={"airspeed_mps": -3.0})
    assert "airspeed_mps" in [i.field for i in analyses.validate(bad)]
    bad = _small("pratt-walker")._replace(params={"gradients": "six"})
    assert "gradients" in [i.field for i in analyses.validate(bad)]
    bad = _small("ensemble")._replace(params={"members": 2.5})
    assert "members" in [i.field for i in analyses.validate(bad)]
    unknown = analyses.AnalysisSpec("x", "nothing", {})
    assert analyses.validate(unknown)[0].field == "analysis"


def test_an_analysis_on_a_run_needs_a_run():
    aspec = analyses.default("convergence")
    assert aspec.base is None
    assert [i.field for i in analyses.validate(aspec)] == ["base"]


def test_an_ensemble_needs_something_random():
    """A deterministic field has no seed to vary; turbulence on top gives one."""
    calm = analyses.default("ensemble", run.PRESETS["vortex-hannibal"])
    assert "no" in " ".join(i.message for i in analyses.validate(calm)
                            if i.level == "error")
    rough = run.with_overlay_kind(run.PRESETS["vortex-hannibal"], "Dryden")
    ok = analyses.default("ensemble", rough)
    assert not [i for i in analyses.validate(ok) if i.level == "error"]


def test_the_autopilot_does_not_fly_the_manoeuvre():
    aspec = analyses.default("closed-loop", run.PRESETS["manoeuvre"])
    assert any(i.field == "base" and "autopilot" in i.message
               for i in analyses.validate(aspec))


def test_a_run_error_is_the_analysis_error():
    aspec = analyses.default("convergence", SHORT._replace(dt=-1.0))
    assert any(i.field == "base" and i.level == "error" for i in analyses.validate(aspec))


def test_a_seed_offset_moves_only_the_seed():
    base = run.PRESETS["dryden"]
    moved = analyses.with_seed_offset(base, 3)
    assert moved.wind.params["seed"] == run.param_value(base.wind, run._param("Dryden",
                                                                              "seed")) + 3
    assert moved._replace(wind=base.wind) == base
    over = run.with_overlay_kind(run.PRESETS["vortex-hannibal"], "Dryden")
    moved = analyses.with_seed_offset(over, 2)
    assert moved.wind == over.wind
    assert moved.overlay.params["seed"] == 2
    with pytest.raises(ValueError, match="no stochastic"):
        analyses.with_seed_offset(run.PRESETS["updraft"], 1)


def test_set_reaches_the_parameters_the_base_and_the_name():
    aspec = analyses.default("ensemble", run.PRESETS["dryden"])
    aspec = analyses.apply_set(aspec, "members=16")
    aspec = analyses.apply_set(aspec, "base.dt=0.01")
    aspec = analyses.apply_set(aspec, "base.wind.sigma_w=2")
    aspec = analyses.apply_set(aspec, "name=rough")
    assert aspec.params["members"] == 16 and aspec.base.dt == 0.01
    assert aspec.base.wind.params["sigma_w"] == 2 and aspec.name == "rough"
    with pytest.raises(ValueError, match="settable"):
        analyses.apply_set(aspec, "colour=blue")
    with pytest.raises(ValueError, match="no base run"):
        analyses.apply_set(analyses.default("modes"), "base.dt=0.1")


# --- what an analysis writes -----------------------------------------------


def test_an_ensembles_members_are_ordinary_runs(tmp_path):
    """Each member opens in the deep dive like any run, and only the seed differs."""
    path = analyses.perform(_small("ensemble"), tmp_path)
    meta = report.read_report(path).meta
    assert meta["members"] == ["members/m000", "members/m001"]
    seeds = []
    for member in meta["members"]:
        flown = artifact.read_run(path / member)
        seeds.append(flown.meta["wind_field"]["params"]["seed"])
        assert flown.checks, "every member carries its checks"
    assert seeds == [0, 1]
    # The runs list shows the report, not its members.
    assert [r.name for r in runs_mod.scan(tmp_path)] == [path.name]


def test_the_modes_report_carries_the_published_reference(tmp_path):
    """At the 747's CRUISE point the model is set against CR-2144's FC 9."""
    loaded = report.read_report(analyses.perform(analyses.default("modes"), tmp_path))
    modes = next(s for s in loaded.sections if s["title"] == "Modes")["table"]["rows"]
    short = next(r for r in modes if r[0] == "short period")
    phugoid = next(r for r in modes if r[0] == "phugoid")
    assert short[3] == 0.964 and short[4] == 0.387
    # scripts/checkpoint.py prints these, to its four decimals.
    assert (round(short[1], 4), round(short[2], 4)) == (0.9513, 0.3892)
    assert (round(phugoid[1], 4), round(phugoid[2], 4)) == (0.0684, 0.0496)


def test_the_step_response_is_tune_py_to_the_digit(tmp_path):
    """The numbers scripts/tune.py prints for the 747, as it prints them."""
    loaded = report.read_report(analyses.perform(analyses.default("step-response"),
                                                 tmp_path))
    rows = {(r[0], r[1]): r for r in loaded.sections[0]["table"]["rows"]}
    alt = rows[("Altitude step +300 m", "altitude")]
    assert (round(alt[3], 3), round(alt[4], 1), round(alt[5], 3)) == (-1.287, 158.2, -1.287)
    hdg = rows[("Heading step +30 deg", "heading")]
    assert (round(hdg[3], 3), round(hdg[4], 1)) == (-0.001, 76.9)
    slow = rows[("Airspeed step -15 m/s", "airspeed")]
    assert slow[6] == "OK, not graded", "below V_md: reported, not graded"
    fast = rows[("Airspeed step +15 m/s", "airspeed")]
    assert (round(fast[3], 3), round(fast[4], 1), fast[6]) == (-0.001, 61.3, "OK")
    assert loaded.meta["summary"].startswith("all graded steps OK")


def test_convergence_is_fourth_order_through_a_smooth_core(tmp_path):
    """The plan's P7 check. A Lamb-Oseen core is smooth, so stage-sampled RK4
    keeps its order; a Rankine core's kink does not (ASSUMPTIONS E9)."""
    base = run.with_param(run.PRESETS["vortex-hannibal"], "profile", "lamb-oseen")
    aspec = analyses.default("convergence", base._replace(dt=0.02))
    aspec = aspec._replace(params={"refinements": 2})
    summary = report.read_report(analyses.perform(aspec, tmp_path)).meta["summary"]
    assert float(summary.split()[-1]) == pytest.approx(4.0, abs=0.15)


def test_the_cross_code_run_on_the_737_agrees_with_jsbsim(tmp_path):
    aspec = analyses.default("cross-code")._replace(
        params={"case": "cimarron", "arm": "translational"})
    loaded = report.read_report(analyses.perform(aspec, tmp_path))
    assert loaded.meta["aircraft"]["key"] == "boeing737"
    rows = loaded.sections[0]["table"]["rows"]
    assert abs(rows[1][3]) < 2.0, "like-for-like n_z span within 2% of JSBSim"


def test_the_gust_transfer_flown_matches_the_exact(tmp_path):
    loaded = report.read_report(analyses.perform(_small("gust-transfer"), tmp_path))
    rows = next(s for s in loaded.sections if s["title"] == "Points")["table"]["rows"]
    assert all(abs(r[4]) < 1.0 for r in rows), "magnitude within 1% of gust_transfer"


def test_a_failed_analysis_leaves_nothing_behind(tmp_path, monkeypatch):
    def boom(aspec, directory, stage):
        (directory / "partial.txt").write_text("half")
        raise RuntimeError("engine fault")

    key = "trim"
    monkeypatch.setitem(analyses.ANALYSES, key,
                        analyses.ANALYSES[key]._replace(execute=boom))
    with pytest.raises(RuntimeError, match="engine fault"):
        analyses.perform(analyses.default(key), tmp_path)
    assert not list(tmp_path.iterdir())


def test_an_invalid_analysis_is_refused_before_anything_is_written(tmp_path):
    with pytest.raises(analyses.AnalysisError, match="runs on a run"):
        analyses.perform(analyses.default("convergence"), tmp_path)
    assert not list(tmp_path.iterdir())


def test_a_second_analysis_never_overwrites_the_first(tmp_path):
    aspec = analyses.default("trim")
    first = analyses.perform(aspec, tmp_path)
    second = analyses.perform(aspec, tmp_path)
    assert first != second and second.name.endswith("-2")


def test_a_report_is_json_a_reader_can_open_without_this_code(tmp_path):
    path = analyses.perform(analyses.default("trim"), tmp_path)
    assert "\n" in (path / "meta.json").read_text()
    json.loads((path / "report.json").read_text())
    assert json.loads((path / "checks.json").read_text()) == []


# --- the CLI and the job runner ----------------------------------------------


def test_the_cli_lists_and_runs_analyses(tmp_path, capsys):
    from atisim import cli

    assert cli.main(["analyses"]) == 0
    listed = capsys.readouterr().out
    for key in analyses.ANALYSES:
        assert key in listed
    assert cli.main(["analyse", "trim", "--set", "aircraft=cessna172",
                     "--out", str(tmp_path)]) == 0
    assert "wrote" in capsys.readouterr().out
    assert cli.main(["analyse", "trim", "--set", "colour=blue", "--out",
                     str(tmp_path)]) == 2
    assert cli.main(["analyse", "ensemble", "--base", "updraft", "--out",
                     str(tmp_path)]) == 2
    spec = tmp_path / "spec.json"
    spec.write_text(analyses.default("trim").to_json())
    assert cli.main(["analyse", "--spec", str(spec), "--out", str(tmp_path)]) == 0


def test_the_job_runner_runs_an_analysis_between_runs(tmp_path):
    from atisim.apps.jobs import JobRunner

    runner = JobRunner(tmp_path)
    job_id = runner.submit_analysis(analyses.default("trim"))
    end = time.time() + 300
    while runner.get(job_id)["state"] not in ("done", "failed") and time.time() < end:
        time.sleep(0.1)
    job = runner.get(job_id)
    assert job["state"] == "done", job["error"]
    assert job["kind"] == "analysis" and job["summary"].startswith("alpha")
    assert [s["message"] for s in job["stages"]][:2] == ["validating", "trimming"]
    bad = runner.submit_analysis(analyses.default("convergence"))
    while runner.get(bad)["state"] not in ("done", "failed"):
        time.sleep(0.05)
    assert runner.get(bad)["error_field"] == "base"


# --- the pages ----------------------------------------------------------------


@pytest.fixture(scope="module")
def reports(tmp_path_factory):
    pytest.importorskip("dash")
    root = tmp_path_factory.mktemp("reports")
    ens = analyses.perform(_small("ensemble"), root)
    trim = analyses.perform(analyses.default("trim"), root)
    flown = run.save(run.fly(SHORT), root)
    return root, ens, trim, flown


def test_results_draws_a_report_and_opens_its_members(reports):
    from atisim.apps import results, shell
    from atisim.tests.test_app import _find, _text

    root, ens, _, _ = reports
    ws = shell.Workspace(root)
    page = results.layout(ws, ens.name)
    text = _text(page)
    assert "Ensemble of 2" in text and "Run again" in text
    assert _find(page, "report-run") is not None and _find(page, "run") is None
    member = f"{ens.name}/members/m001"
    page = results.layout(ws, member)
    assert _find(page, "run").value == member
    assert f"Back to {ens.name}" in _text(page)


def test_a_member_path_never_leaves_the_runs_directory(reports):
    from atisim.apps import results

    root, ens, _, _ = reports
    assert results.nested_run(root, f"{ens.name}/members/m000")
    assert not results.nested_run(root, f"{ens.name}/../{ens.name}/members/../..")
    assert not results.nested_run(root, "../elsewhere/members/m000")
    assert not results.nested_run(root, ens.name)


def test_the_fig8_panel_skips_reports(reports):
    from atisim.apps import shell

    root, _, _, flown = reports
    points = shell.Workspace(root).fig8_points()
    assert len(points) == 1


def test_every_analysis_page_builds_and_validates_clean(reports):
    from atisim.apps import analyses as page
    from atisim.apps import shell

    root, _, trim, _ = reports
    ws = shell.Workspace(root)
    for key, a in analyses.ANALYSES.items():
        layout = page.layout(ws, key)
        assert a.label in page_text(layout)
        base = page.DEFAULT_BASE.get(key, "preset:updraft")
        aspec = page.build_spec(key, [], [], base, None, None)
        errors = [i for i in analyses.validate(aspec) if i.level == "error"]
        assert not errors, (key, errors)
    # Run again from a report fills the form from its spec.
    layout = page.layout(ws, source=trim.name)
    assert "Trim" in page_text(layout)


def test_the_form_hands_the_spec_the_types_it_checks():
    from atisim.apps import analyses as page

    members = analyses.MEMBERS
    assert page.coerce(members, 8.0) == 8 and isinstance(page.coerce(members, 8.0), int)
    assert page.coerce(analyses.AIRSPEED, "") is None
    aspec = page.build_spec("ensemble", [{"type": "aparam", "name": "members"}], [3.0],
                            "setup", run.PRESETS["dryden"].to_dict(), None)
    assert aspec.params["members"] == 3 and aspec.base == run.PRESETS["dryden"]


def page_text(component) -> str:
    from atisim.tests.test_app import _text

    return _text(component)


def test_the_analyse_menu_offers_every_analysis_on_a_run():
    pytest.importorskip("dash")
    from atisim.apps import shell
    from atisim.tests.test_app import _text

    menu = _text(shell._analyse_menu())
    for a in analyses.ANALYSES.values():
        assert (a.label in menu) == a.needs_base, a.key


def test_no_analysis_moves_a_bit_of_a_flight():
    """An analysis calls `run.fly`; it must not change what `run.fly` returns.

    The ensemble's members fly `with_seed_offset(base, i)`: member 0 IS the
    base run, bit for bit.
    """
    base = run.PRESETS["dryden"]._replace(seconds=4.0)
    direct = run.fly(base).trajectory
    member0 = run.fly(analyses.with_seed_offset(base, 0)).trajectory
    for name in ("pos_ned", "vel_body", "quat", "omega", "wind_ned"):
        assert np.array_equal(np.asarray(getattr(direct, name)),
                              np.asarray(getattr(member0, name))), name
