"""The app builds, every route resolves, and the runs directory states render.

No browser: the router and the page layouts are called directly. What a browser
adds (the callbacks firing in order) is checked by hand in the session record.
"""

import json
import re
from pathlib import Path

import pytest

import atisim  # noqa: F401

pytest.importorskip("dash")
pytest.importorskip("dash_mantine_components")
pytest.importorskip("dash_iconify")
pytest.importorskip("pyarrow")

from atisim import run  # noqa: E402
from atisim.analysis import runs as runs_mod  # noqa: E402
from atisim.apps import results, shell, start, theme  # noqa: E402


def _route(app):
    """The router callback's own function, called outside a request."""
    for key, entry in app.callback_map.items():
        if key.startswith("..page.children"):
            fn = entry["callback"]
            return getattr(fn, "__wrapped__", fn)
    raise AssertionError("no router callback")


def _text(component) -> str:
    """Every string in a component tree, for asserting on what a page says."""
    out = []

    def walk(node):
        if node is None:
            return
        if isinstance(node, (str, int, float)):
            out.append(str(node))
            return
        if isinstance(node, (list, tuple)):
            for child in node:
                walk(child)
            return
        for attr in ("children", "label", "title"):
            walk(getattr(node, attr, None))
        for option in getattr(node, "options", None) or []:
            walk(option.get("label") if isinstance(option, dict) else option)

    walk(component)
    return " ".join(out)


@pytest.fixture(scope="module")
def one_run(tmp_path_factory):
    """A real artifact: a short updraft run."""
    root = tmp_path_factory.mktemp("runs")
    spec = run.PRESETS["updraft"]._replace(dt=0.05, seconds=12.0)  # reaches the column
    path = run.save(run.fly(spec), root)
    return root, path


def test_the_app_builds_and_every_route_resolves(one_run):
    root, path = one_run
    app = shell.build_app(root)
    route = _route(app)
    for pathname, search in [("/", ""), ("/start", ""), ("/setup", ""),
                             ("/setup", "?preset=vortex-hannibal"),
                             ("/results", ""), ("/results", f"?run={path.name}&t=5"),
                             ("/analyses", ""), ("/analyses", "?analysis=ensemble&base=setup"),
                             ("/analyses", "?analysis=nothing"), ("/analyses", "?from=missing"),
                             ("/research", ""), ("/research", "?script=sanity"),
                             ("/research", "?script=nowhere"),
                             ("/results", f"?run={path.name}&view=diagnostics"),
                             ("/results", f"?run={path.name}&view=compare"),
                             ("/nowhere", "")]:
        page, *_rest = route(pathname, search, 0)
        assert page is not None, pathname


def test_setup_with_a_preset_loads_it_into_the_store(one_run):
    root, _ = one_run
    route = _route(shell.build_app(root))
    out = route("/setup", "?preset=microburst", 3)
    spec_data, structure = out[-2], out[-1]
    assert run.RunSpec.from_json(spec_data) == run.PRESETS["microburst"]
    assert structure == 4


def test_the_run_button_shows_on_setup_only(one_run):
    """With it the Analyse menu, which analyses the spec being edited."""
    root, _ = one_run
    route = _route(shell.build_app(root))
    assert route("/setup", "", 0)[3]["display"] == "flex"
    for page in ("/results", "/analyses", "/start", "/research"):
        assert route(page, "", 0)[3] == {"display": "none"}


def test_the_sweep_entry_point_lands_on_results(one_run):
    from atisim.apps import sweep

    root, path = one_run
    route = _route(sweep.build_app(root))
    page, _tree, crumbs = route("/", "", 0)[:3]
    assert path.name in _text(page)
    assert _text(crumbs).startswith("Results")


def test_results_puts_the_cursor_on_the_sample_in_the_url(one_run):
    root, path = one_run
    ws = shell.Workspace(root)
    page = results.layout(ws, path.name, 7)
    store = _find(page, "cursor")
    assert store.data == pytest.approx(float(ws.loaded(path.name).series.t[7]))


def _find(node, ident):
    if getattr(node, "id", None) == ident:
        return node
    children = getattr(node, "children", None)
    for child in children if isinstance(children, (list, tuple)) else [children]:
        if child is not None and not isinstance(child, (str, int, float)):
            hit = _find(child, ident)
            if hit is not None:
                return hit
    return None


# --- runs directory states ---------------------------------------------------


def test_a_missing_runs_directory_renders_the_empty_state(tmp_path):
    ws = shell.Workspace(tmp_path / "missing")
    text = _text(start.layout(ws))
    assert "does not exist yet" in text and "atisim run --preset vortex-hannibal" in text
    assert "No runs yet" in _text(results.layout(ws))
    assert "does not exist yet" in _text(results.layout(ws))


def test_an_empty_runs_directory_renders_the_empty_state(tmp_path):
    ws = shell.Workspace(tmp_path)
    assert "holds no runs yet" in _text(start.layout(ws))


def test_one_run_lists_with_its_verdict(one_run):
    root, path = one_run
    text = _text(start.layout(shell.Workspace(root)))
    assert path.name in text
    row = runs_mod.scan(root)[0]
    assert row.verdict in text


def test_a_stale_run_is_a_badge_not_an_error(tmp_path, one_run):
    import shutil

    _, path = one_run
    copy = tmp_path / path.name
    shutil.copytree(path, copy)
    meta = json.loads((copy / "meta.json").read_text())
    meta["git_sha"] = "0" * 40
    (copy / "meta.json").write_text(json.dumps(meta))
    row = runs_mod.scan(tmp_path)[0]
    if not row.stale:
        pytest.skip("no git here, so nothing can be stale")
    assert row.error is None
    assert "stale" in _text(start.layout(shell.Workspace(tmp_path)))


def test_a_broken_meta_json_is_an_error_row_and_nothing_crashes(tmp_path, one_run):
    import shutil

    _, path = one_run
    shutil.copytree(path, tmp_path / path.name)
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "meta.json").write_text("{not json")
    rows = runs_mod.scan(tmp_path)
    assert [r.name for r in rows][-1] == "broken" and rows[-1].verdict == "error"
    ws = shell.Workspace(tmp_path)
    assert "meta.json cannot be read" in _text(start.layout(ws))
    from atisim.apps import explorer

    assert "meta.json cannot be read" in _text(explorer.tree(ws, "/results", {}))
    assert results.layout(ws) is not None
    assert len(ws.fig8_points()) == 1


# --- the job runner ------------------------------------------------------------


def _finished(runner, job_id, timeout=300.0):
    import time

    end = time.time() + timeout
    while time.time() < end:
        job = runner.get(job_id)
        if job["state"] in ("done", "failed"):
            return job
        time.sleep(0.1)
    raise AssertionError(f"the job is still {job['state']} after {timeout} s")


def test_a_failed_run_names_the_field_to_fix_and_the_runner_carries_on(tmp_path):
    """Setup lists a failure in Messages on this field's node, so it must be the
    field where the fix goes: the airspeed for a trim failure."""
    from atisim.apps.jobs import JobRunner

    runner = JobRunner(tmp_path)
    slow = run.PRESETS["vortex-hannibal"]._replace(airspeed_mps=60.0)
    job = _finished(runner, runner.submit(slow))
    assert job["state"] == "failed" and job["error_field"] == "airspeed_mps"
    assert "minimum-drag speed" in job["error"]
    assert [s["name"] for s in job["stages"]] == ["validating", "trimming"]
    job = _finished(runner, runner.submit(slow._replace(dt=-1.0)))
    assert job["state"] == "failed" and job["error_field"] == "dt"
    assert not list(tmp_path.iterdir()), "a failed run writes nothing"


# --- the visual system -------------------------------------------------------


def test_every_text_colour_clears_4_5_to_1_on_every_surface():
    surfaces = [theme.CHROME, theme.PANEL, theme.SURFACE, "#ffffff"]
    for fg in (theme.INK, theme.INK_MUTED, theme.ACCENT, theme.OK, theme.WARN, theme.FAIL):
        for bg in surfaces:
            assert theme.contrast(fg, bg) >= 4.5, (fg, bg)
    for fg, wash in ((theme.OK, theme.OK_WASH), (theme.WARN, theme.WARN_WASH),
                     (theme.FAIL, theme.FAIL_WASH), (theme.INK, theme.ACCENT_WASH)):
        assert theme.contrast(fg, wash) >= 4.5, (fg, wash)


def _icons(node, out=None) -> set:
    """Every icon name drawn in a component tree, in any property."""
    out = set() if out is None else out
    if isinstance(node, (list, tuple)):
        for child in node:
            _icons(child, out)
    elif hasattr(node, "to_plotly_json"):
        if type(node).__name__ == "DashIconify":
            out.add(str(node.icon).removeprefix("tabler:"))
        for value in node.to_plotly_json().get("props", {}).values():
            _icons(value, out)
    return out


def test_every_icon_the_app_draws_is_held_so_it_draws_with_no_internet(one_run):
    """`assets/icons.js` preloads the icons; one that it lacks would be fetched from
    the Iconify web API, and be blank on a computer with no internet."""
    from atisim.apps import analyses, icons, setup, start

    held = icons.held()
    assert icons.used() <= held, sorted(icons.used() - held)
    root, path = one_run
    ws = shell.Workspace(root)
    from atisim.apps import card, fly

    pages = [shell.build_app(root).layout, start.layout(ws), setup.layout(ws),
             results.layout(ws, path.name), card.layout(ws), fly.layout(ws, "hannibal")]
    pages += [setup.settings_panel(node, run.PRESETS["vortex-hannibal"], ws)
              for node, *_ in setup.NODES]
    pages += [analyses.layout(ws, key) for key in ("ensemble", "modes", "study")]
    drawn = _icons(pages)
    assert len(drawn) > 20 and drawn <= held, sorted(drawn - held)


def test_the_stylesheet_carries_the_theme_tokens():
    css = (Path(theme.__file__).parent / "assets" / "atisim.css").read_text()
    for token, value in {"--ati-chrome": theme.CHROME, "--ati-panel": theme.PANEL,
                         "--ati-surface": theme.SURFACE, "--ati-divider": theme.DIVIDER,
                         "--ati-ink": theme.INK, "--ati-ink-muted": theme.INK_MUTED,
                         "--ati-accent": theme.ACCENT, "--ati-ok": theme.OK,
                         "--ati-warn": theme.WARN, "--ati-fail": theme.FAIL}.items():
        assert re.search(rf"{token}:\s*{value};", css, re.I), token


def test_the_figures_pick_up_the_template_once_registered():
    from atisim.analysis import figures

    theme.register()
    fig = figures.no_field_preview("still air", 30.0)
    assert fig.layout.template.layout.font.family == theme.FONT


def test_a_report_check_is_never_coloured():
    from atisim.apps import components

    word, tone, _ = components.check_state(
        {"name": "x", "kind": "report", "passed": None, "value": 1.0, "detail": ""})
    assert (word, tone) == ("report", "neutral")
    word, tone, _ = components.check_state(
        {"name": "q", "kind": "tripwire", "passed": True, "value": 0.0, "detail": ""})
    assert (word, tone) == ("quiet", "neutral")


# --- the simple mode: the test card and the live cockpit -----------------------


def test_the_landing_is_the_test_card_with_every_test_point_and_no_workbench(tmp_path):
    from atisim import cockpit

    route = _route(shell.build_app(tmp_path))
    out = route("/", "", 0)
    text = _text(out[0])
    for tp in cockpit.TEST_POINTS.values():
        assert tp.title in text
    assert "Lab" in text and "Engineering" in text  # the mode switch
    assert text.count("not flown") == len(cockpit.TEST_POINTS)
    assert out[4] == "ati-app is-simple"  # no workbench header, explorer or status bar
    assert out[1] == [] and route("/start", "", 0)[4] == "ati-app"


def test_the_cockpit_resolves_and_an_unknown_test_point_says_so(tmp_path):
    route = _route(shell.build_app(tmp_path))
    page = route("/fly", "?tp=hannibal", 0)[0]
    assert _find(page, "cockpit") is not None and _find(page, "fly-pfd") is not None
    assert "There is no test point" in _text(route("/fly", "?tp=nowhere", 0)[0])


def test_the_flight_api_flies_pauses_and_debriefs_and_the_card_keeps_the_result(tmp_path):
    import time

    app = shell.build_app(tmp_path)
    client = app.server.test_client()
    started = client.post("/api/flight", json={"tp": "calm"}).get_json()
    assert started["frame"]["t"] == 0.0 and started["info"]["test_point"]["key"] == "calm"
    poll = f"/api/flight/{started['id']}/poll"
    client.post(poll, json={})  # starts the clock; a press acts on the next physics step
    time.sleep(0.1)
    frame = client.post(poll, json={"held": [], "presses": {"a": 1, "t": 0}}).get_json()
    assert frame["mode"] == "autopilot" and frame["t"] > 0.0
    assert client.post(f"/api/flight/{started['id']}/pause").status_code == 200
    summary = client.post(f"/api/flight/{started['id']}/summary").get_json()
    assert summary["severity_band"] == "too short to rate"
    text = _text(_route(app)("/", "", 0)[0])
    assert text.count("not flown") == 3 and f"{summary['nz_max']['value']:+.2f} g" in text


def test_the_flight_api_refuses_what_it_does_not_have(tmp_path):
    client = shell.build_app(tmp_path).server.test_client()
    assert client.post("/api/flight", json={"tp": "nowhere"}).status_code == 400
    gone = client.post("/api/flight/f999/poll", json={})
    assert gone.status_code == 404 and "has ended" in gone.get_json()["error"]


def test_only_the_newest_flights_are_kept():
    from atisim.apps import fly

    flights = fly.Flights(kept=1)
    first, _ = flights.start("calm")
    second, _ = flights.start("calm")
    assert flights.get(first) is None and flights.get(second) is not None


# --- engineering mode: the explorer and the research pages -------------------------


def test_a_result_row_knows_its_group(one_run, tmp_path):
    from atisim.analysis import report as report_mod

    root, path = one_run
    row = runs_mod.scan(root)[0]
    assert row.group == "flight" and row.analysis is None and row.script is None
    study = runs_mod.RunRow(tmp_path, "s", "study of x.py", "", None, "", "report", False,
                            None, [], analysis="study", script="x")
    assert study.group == "study"
    assert runs_mod.RunRow(tmp_path, "m", "modes report", "", None, "", "report", False,
                           None, [], analysis="modes").group == "report"
    assert report_mod is not None


def test_the_explorer_lists_everything_and_marks_the_page_on_screen(one_run):
    from atisim import analyses as registry
    from atisim import studies
    from atisim.apps import explorer

    root, path = one_run
    ws = shell.Workspace(root)
    text = _text(explorer.tree(ws, "/research", {"script": "cat_validation"}))
    for family, _ in registry.FAMILIES:
        if family != "Studies":
            assert family in text, family
    for topic, _ in studies.by_topic():
        assert topic.title in text, topic.title
    assert path.name in text and "vortex-hannibal" in text
    tree = explorer.tree(ws, "/results", {"run": path.name})
    active = [n for n in _walk(tree) if "is-active" in (getattr(n, "className", "") or "")]
    assert len(active) == 1 and path.name in _text(active[0])
    crumbs = explorer.crumbs(ws, "/research", {"script": "cat_validation"})
    assert [c[0] for c in crumbs] == ["Research scripts", "Clear-air turbulence cases",
                                      "cat_validation"]


def _walk(node):
    if isinstance(node, (list, tuple)):
        for child in node:
            yield from _walk(child)
        return
    if node is None or isinstance(node, (str, int, float)):
        return
    yield node
    yield from _walk(getattr(node, "children", None))


def test_the_research_pages_find_run_and_list_a_script(one_run):
    from atisim.apps import research

    root, _ = one_run
    ws = shell.Workspace(root)
    index = _text(research.layout(ws))
    assert "Clear-air turbulence cases" in index and "cat_validation" in index
    page = research.layout(ws, "sanity")
    assert "Quick checks" in _text(page) and "Run script" in _text(page)
    assert _find(page, "analysis-run") is not None  # the Analyses page's own form
    assert "There is no research script" in _text(research.layout(ws, "nowhere"))


# --- the Lab: the middle mode ---------------------------------------------------------


def test_every_lab_page_resolves_in_simple_chrome(one_run):
    root, path = one_run
    route = _route(shell.build_app(root))
    for pathname, search in [("/lab", ""), ("/lab/case", "?case=dryden"),
                             ("/lab/case", "?case=nowhere"), ("/lab/case", f"?from={path.name}"),
                             ("/lab/result", f"?run={path.name}"), ("/lab/result", "?run=x"),
                             ("/lab/analysis", "?key=modes"), ("/lab/analysis", "?key=x"),
                             ("/lab/compare", ""), ("/lab/compare", f"?a={path.name}")]:
        out = route(pathname, search, 0)
        assert out[0] is not None and out[4] == "ati-app is-simple", pathname + search


def test_the_lab_result_card_reads_the_run_in_plain_words(one_run):
    from atisim.apps import lab as lab_page

    root, path = one_run
    ws = shell.Workspace(root)
    text = _text(lab_page.result_page(ws, path.name))
    assert "Peak load factor" in text and " g" in text and "Misaka 2008" in text
    assert "Fly again with changes" in text and "Open in engineering mode" in text


def test_the_lab_form_builds_its_spec_from_the_preset_and_the_values():
    from atisim.apps import lab as lab_page

    base = run.PRESETS["vortex-hannibal"]
    ids = [{"type": "lab-val", "name": n} for n in
           ("aircraft", "airspeed_mps", "altitude_m", "wind.v0", "wind.r0")]
    same = [base.aircraft, base.airspeed_mps, base.altitude_m, base.wind.params["v0"],
            base.wind.params["r0"]]
    assert lab_page.spec_from(base.to_dict(), ids, same) == base
    edited = lab_page.spec_from(base.to_dict(), ids, same[:3] + [30, same[4]])
    assert edited.wind.params["v0"] == 30 and run.provenance(edited)["v0"].status == "declared"


def test_the_mode_switch_is_on_every_mode_and_marks_the_one_on_screen(one_run):
    from atisim.apps import card, lab as lab_page

    root, _ = one_run
    ws = shell.Workspace(root)
    app = shell.build_app(root)
    for page, mode in ((card.layout(ws), "Test card"), (lab_page.home(ws), "Lab"),
                       (app.layout, "Engineering")):
        active = [n for n in _walk(page) if "mode-item is-active" in
                  (getattr(n, "className", "") or "")]
        assert len(active) == 1 and _text(active[0]).startswith(mode), mode


# --- the stress-test fixes ----------------------------------------------------------


def test_a_debrief_saves_the_flight_once_and_the_card_sends_engineering_to_it(tmp_path):
    import time

    app = shell.build_app(tmp_path)
    client = app.server.test_client()
    started = client.post("/api/flight", json={"tp": "calm"}).get_json()
    poll = f"/api/flight/{started['id']}/poll"
    client.post(poll, json={})
    time.sleep(0.1)
    client.post(poll, json={})
    summary = client.post(f"/api/flight/{started['id']}/summary").get_json()
    assert summary["run"] and (tmp_path / summary["run"] / "run.parquet").exists()
    # A debrief read twice is not a second flight.
    again = client.post(f"/api/flight/{started['id']}/summary").get_json()
    assert again["run"] == summary["run"]
    page = _route(app)("/", "", 0)[0]
    hrefs = {getattr(n, "href", None) for n in _walk(page)}
    assert f"/results?run={summary['run']}" in hrefs
    # Still air has no turbulence to rate: the card does not call it severe.
    assert "still air" in _text(page)


def test_a_number_is_read_as_a_person_types_it():
    from atisim.apps import components as ui

    assert ui.parse_number("1e-3") == 0.001  # the number input made this -13
    assert ui.parse_number("236,5") == 236.5  # and this 2365
    assert ui.parse_number(" 12 ") == 12 and isinstance(ui.parse_number("12"), int)
    assert ui.parse_number("") is None
    # Negative control: text that is not a number stays text, for the validator.
    assert ui.parse_number("abc") == "abc" and ui.parse_number("1,000.5") == "1,000.5"
    assert ui.shown_number(21.336000000000002) == "21.336"


def test_the_lab_says_an_unknown_case_and_a_same_flight_comparison(one_run):
    from atisim.apps import lab as lab_page

    root, path = one_run
    ws = shell.Workspace(root)
    assert "There is no case" in _text(lab_page.case_page(ws, "nowhere"))
    assert "same flight" in _text(lab_page.compare_page(ws, path.name, path.name))


def test_an_analysis_page_shows_only_its_own_job():
    from atisim.apps import analyses as page

    job = {"kind": "analysis", "state": "done", "stages": [], "log": [], "path": "modes-x",
           "summary": "", "spec": {"name": "modes", "analysis": "modes", "params": {}}}
    assert "Open in Results" in _text(page.progress_view(job, "modes"))
    # Negative control: the Modes job on the Seed ensemble page.
    assert "Open in Results" not in _text(page.progress_view(job, "ensemble"))


def test_the_poll_runs_on_for_a_moment_after_the_last_job(tmp_path):
    import time

    from atisim.apps.jobs import JobRunner

    runner = JobRunner(tmp_path)
    runner._jobs["j"] = {"state": "done", "finished": time.time()}
    assert runner.live(grace=2.0) and not runner.live()
    runner._jobs["j"]["finished"] -= 5.0
    assert not runner.live(grace=2.0)


def test_compare_offers_only_runs_that_are_not_already_in_it(one_run):
    root, path = one_run
    rows = runs_mod.scan(root)
    offered = [item["value"] for group in results.picker(
        rows, None, "compare-add", runs_only=True, exclude=[path.name]).data
        for item in group["items"]]
    assert path.name not in offered
