"""The app shell: header, workspace switch, status bar, and the router.

    atisim ui runs/
    python -m atisim.apps.sweep runs/analysis     (opens on Results)

One Dash app, three modes (`components.mode_switch`). `/` is the test card
(`card.py`) and `/fly` its live cockpit (`fly.py`), for a reader who does not
know the code. `/lab` is the Lab (`lab.py`), the middle: any case with a few
values changed, four analyses, two results compared. Both are drawn in the test
card's world, with no workbench chrome. Engineering mode is the explorer down the left
(`explorer.py`: Fly a run, Analyses, Research scripts, Results) and the page
of the item chosen in it -- Overview, Setup, an analysis, a research script, a
result -- swapped by one router callback on `dcc.Location`. The header shows
where the page sits (the breadcrumb) and the page's own actions. State that must survive a page swap (the spec being edited, the selected
tree node, the running job) lives in stores here in the shell, so moving to
Results and back keeps the form.

Local only: the app binds 127.0.0.1. There is no Exit button (it is a browser
tab) and no login.
"""

import threading
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs

import dash_mantine_components as dmc
from dash import Dash, Input, Output, State, dcc, html, no_update
from dash.exceptions import PreventUpdate

from atisim.analysis import artifact
from atisim.analysis import runs as runs_mod
from atisim.apps import components as ui
from atisim.apps import (analyses, card, compare, diagnostics, explorer, fly, lab, research,
                         results, setup, start, theme)
from atisim.apps.jobs import JobRunner

DOCS_URL = "https://matusgib.github.io/AtiSim/"
ASSETS = Path(__file__).resolve().parent / "assets"
SIMPLE = ("/", "/fly")  # the pages with no workbench header or status bar, and /lab/*

# The simple mode's direction contract (impeccable, seed 9ad3c239). It rides in
# the served page, first in <body>, so a review of the page can read it.
CONTRACT = """<!--
THESIS: The landing is a flight-test card clipped to a kneeboard: numbered test
points you fly, not a dashboard of tiles or a hero with a Start button.
OWN-WORLD: White card stock on a graphite kneeboard with a flat clip; B612, the
cockpit face; ruled test-point table with ink rules; red only for limits;
blue-ink FLY stamps; black grease-pencil marks for what you flew.
STORY: A visitor reads in one line what AtiSim is, picks a test point from a
sketch of the air it holds, flies it, and finds the result written on the card.
Engineering mode is one labelled button away.
FIRST VIEWPORT: Graphite ground, the card centred with its clip; header box
(title, aircraft, start condition, card number), the objective, then the
four-row test-point table, each row ending in its FLY stamp.
FORM: flight-test card, number 1 of 7 on the ordered list, seed 9ad3c239.
FINISH: unreviewed and undocumented is unfinished; this build ends with the
finish review, the verdict, DESIGN.md, and every shipping raster carrying its
provenance
-->"""


class Workspace:
    """The runs directory, the job runner, and a cache of loaded runs."""

    def __init__(self, root: Path, landing: str = "/", dev: bool = False):
        self.root = Path(root)
        self.landing = landing
        self.dev = dev
        self.jobs = JobRunner(self.root, worker=dev)
        self._cache: dict[str, tuple[float, results.Loaded]] = {}
        self.flights = fly.Flights()
        # The test card's results, per test point: kept while the app runs.
        self.marks: dict[str, dict] = {}

    def loaded(self, name: str) -> "results.Loaded":
        path = self.root / name
        stamp = (path / "meta.json").stat().st_mtime
        hit = self._cache.get(name)
        if hit is None or hit[0] != stamp:
            self._cache[name] = (stamp, results.Loaded(path))
        return self._cache[name][1]

    def fig8_points(self) -> list[dict]:
        """Fig. 8 coordinates for every readable run; an unreadable one is skipped."""
        points = []
        for row in runs_mod.scan(self.root):
            if row.error or row.verdict == "report":
                continue
            try:
                points.append(self.loaded(row.name).fig8())
            except Exception:  # a broken artifact must not take the page down
                continue
        return points


def _header(ws: Workspace):
    return html.Header([
        html.Button(ui.icon("layout-sidebar-left-collapse", 18), id="explorer-toggle",
                    className="ati-iconbtn", type="button",
                    title="Hide or show the explorer",
                    **{"aria-label": "Hide or show the explorer",
                       "aria-controls": "explorer", "aria-expanded": "true"}),
        dcc.Link("AtiSim", href="/start", className="ati-wordmark"),
        html.Nav(id="crumbs", className="ati-crumbs", **{"aria-label": "Where you are"}),
        html.Span(className="ati-header-spacer"),
        ui.mode_switch("engineering", on="light"),
        html.A([ui.icon("book", 15), "Documentation", ui.icon("external-link", 12)],
               href=DOCS_URL, target="_blank", rel="noopener noreferrer",
               className="ati-doclink"),
        html.Div([
            _analyse_menu(),
            dmc.Tooltip(
                dmc.Button("Run", id="run-button", leftSection=ui.icon("player-play", 14),
                           size="xs", n_clicks=0, disabled=True),
                id="run-tooltip", label="Fly this spec.",
            ),
        ], id="run-slot", style={"display": "none"}),
    ], className="ati-header")


def _analyse_menu():
    """Setup's second action: run an analysis ON the spec being edited.

    Each item opens Analyses with `base=setup`, so the analysis starts from
    exactly the run on screen, unsaved edits included (they live in the
    session's spec store).
    """
    items = [dmc.MenuLabel("Analyse this run")]
    for key, a in analyses.registry.ANALYSES.items():
        if a.needs_base:
            items.append(dmc.MenuItem(a.label, href=f"/analyses?analysis={key}&base=setup",
                                      refresh=False, id=f"analyse-{key}"))
    return dmc.Menu([
        dmc.MenuTarget(dmc.Button("Analyse", id="analyse-button", size="xs",
                                  variant="default",
                                  leftSection=ui.icon("list-details", 14),
                                  rightSection=ui.icon("chevron-down", 12))),
        dmc.MenuDropdown(items),
    ], position="bottom-end", withinPortal=True)


def _jax_backend() -> str:
    try:
        import jax

        return jax.default_backend()
    except Exception:
        return "unknown"


def build_app(root, landing: str = "/", dev: bool = False, warm: bool = False) -> Dash:
    """The whole app. `landing` is the page `/` shows ("/" the test card,
    "/results").

    `dev` is engine-development mode: jobs run in the engine worker subprocess,
    and the status bar offers **Reload engine** when the engine's source changes
    (`jobs.JobRunner.reload`). The cockpit always flies in this process.

    `warm` draws the card's sketches and compiles each test point's flight in a
    background thread, so the first FLY in a demo does not wait on JAX."""
    theme.register()
    ws = Workspace(Path(root), landing, dev)
    if warm:
        threading.Thread(target=_warm, daemon=True, name="atisim-warm").start()
    backend = _jax_backend()
    sha = artifact.git_sha()[:7] or "nogit"

    app = Dash(__name__, title="AtiSim", assets_folder=str(ASSETS),
               suppress_callback_exceptions=True, update_title=None)
    app.index_string = (app.index_string.replace("<html>", '<html lang="en">')
                        .replace("<body>", "<body>\n" + CONTRACT, 1))
    app.layout = dmc.MantineProvider(
        theme=theme.MANTINE_THEME, forceColorScheme="light",
        children=html.Div([
            dcc.Location(id="url", refresh="callback-nav"),
            dcc.Store(id="spec", storage_type="session"),
            dcc.Store(id="node", storage_type="session", data="wind"),
            dcc.Store(id="structure", data=0),
            dcc.Store(id="job", storage_type="session"),
            dcc.Store(id="analysis-job", storage_type="session"),
            dcc.Store(id="engine-reloaded", data=0),
            dcc.Interval(id="poll", interval=500, disabled=True),
            dcc.Interval(id="status-tick", interval=5000),
            _header(ws),
            html.Div([
                explorer.panel(),
                html.Div(className="ati-split ati-split-explorer", role="separator",
                         tabIndex=0, **{"aria-orientation": "vertical",
                                        "aria-label": "Resize the explorer",
                                        "data-var": "--explorer-w", "data-min": "220",
                                        "data-max": "480"}),
                html.Main(id="page", className="ati-page"),
            ], className="ati-body", **{"data-split-host": ""}),
            html.Footer(id="status", className="ati-status"),
        ], id="app-root", className="ati-app"),
    )

    @app.callback(
        Output("page", "children"),
        Output("explorer", "children"), Output("crumbs", "children"),
        Output("run-slot", "style"),
        Output("app-root", "className"),
        Output("spec", "data"), Output("structure", "data"),
        Input("url", "pathname"), Input("url", "search"),
        State("structure", "data"),
    )
    def route(pathname, search, structure):
        query = {k: v[-1] for k, v in parse_qs((search or "").lstrip("?")).items()}
        path = (pathname or "/").rstrip("/") or "/"
        if path == "/" and ws.landing != "/":
            path = ws.landing
        spec_data, bump = no_update, no_update
        if path == "/":
            page = card.layout(ws)
        elif path == "/fly":
            page = fly.layout(ws, query.get("tp"))
        elif path == "/lab" or path.startswith("/lab/"):
            page = lab.layout(ws, path, query)
        elif path == "/setup":
            page = setup.layout(ws)
            preset = setup.spec_store_value(query.get("preset"))
            if preset is not None:
                spec_data, bump = preset, (structure or 0) + 1
        elif path == "/analyses":
            page = analyses.layout(ws, query.get("analysis"), query.get("base"),
                                   query.get("from"))
        elif path == "/research":
            page = research.layout(ws, query.get("script"))
        elif path == "/results":
            page = results.layout(ws, query.get("run"), query.get("t"), query.get("view"),
                                  query.get("runs"))
        elif path == "/start":
            page = start.layout(ws)
        else:
            page = html.Div([
                html.P([html.Strong("There is no page at "), html.Code(pathname), "."]),
                dcc.Link("Go to the overview", href="/start"),
            ], className="ati-empty")
        run_slot = ({"display": "flex", "gap": "8px", "alignItems": "center"}
                    if path == "/setup" else {"display": "none"})
        if path in SIMPLE or path == "/lab" or path.startswith("/lab/"):
            return (page, [], [], run_slot, "ati-app is-simple", spec_data, bump)
        tree = explorer.tree(ws, path, query,
                             setup.tree_nodes() if path == "/setup" else None)
        crumbs = explorer.crumb_bar(explorer.crumbs(ws, path, query))
        return (page, tree, crumbs, run_slot, "ati-app", spec_data, bump)

    @app.callback(
        Output("status", "children"),
        Output("poll", "disabled", allow_duplicate=True),
        Input("poll", "n_intervals"), Input("status-tick", "n_intervals"),
        Input("job", "data"), Input("analysis-job", "data"), Input("url", "pathname"),
        Input("engine-reloaded", "data"),
        prevent_initial_call="initial_duplicate",
    )
    def status(_poll, _tick, job_id, analysis_job_id, _path, _reloaded):
        # The newer of the two: one worker runs both kinds in order.
        jobs = [j for j in (ws.jobs.get(job_id), ws.jobs.get(analysis_job_id)) if j]
        job = max(jobs, key=lambda j: j["submitted"]) if jobs else None
        if job is None:
            state = ui.state("idle", "neutral", ui.ICONS["idle"], 13)
        elif job["state"] == "done":
            state = ui.state(f"done: {Path(job['path']).name}", "ok", size=13)
        elif job["state"] == "failed":
            state = ui.state("failed", "fail", size=13, title=job["error"])
        else:
            doing = (job["stages"][-1].get("message", job["state"])
                     if job["state"] == "running" and job["stages"] else job["state"])
            state = ui.state(f"{doing}, {job['elapsed']:.0f} s", "neutral",
                             ui.ICONS["running"], 13)
        count = sum(1 for r in runs_mod.scan(ws.root) if r.error is None)
        root_path = str(ws.root.resolve())
        # The job first, then the engine and its Reload button, which must not
        # be cut off: the job's text gives way when the bar is short.
        items = [
            html.Span(["Job ", state], className="ati-status-item ati-status-job"),
            *_engine_items(ws),
            html.Span([ui.icon("cpu", 13), f"JAX backend {backend}"],
                      className="ati-status-item"),
            html.Span([ui.icon("git-commit", 13), html.Span(sha, className="ati-mono")],
                      className="ati-status-item", title="git commit of this code"),
            html.Span([ui.icon("database", 13), f"{count} result{'s' if count != 1 else ''}"],
                      className="ati-status-item"),
            html.Span([ui.icon("folder", 13),
                       html.Span(ui.middle(root_path, 60), className="ati-mono")],
                      className="ati-status-item ati-status-path",
                      title=f"Runs directory: {root_path}"),
        ]
        if job is not None and job.get("kind") == "run":
            fidelity = (job.get("spec") or {}).get("fidelity", "standard")
            timing = job.get("timing") or {}
            last = ", ".join(f"{k} {v:.1f} s" for k, v in timing.items()
                             if k in ("trimming", "flying", "diagnostics"))
            items.insert(2, html.Span(
                [ui.icon("adjustments", 13), f"{fidelity} fidelity"],
                className="ati-status-item",
                title="The last run's fidelity" + (f" and stage times: {last}. The first "
                                                   "run's flying includes the JAX compile."
                                                   if last else ".")))
        live = ws.jobs.live()
        return items, not live

    @app.callback(
        Output("engine-reloaded", "data"),
        Input("engine-reload", "n_clicks"), State("engine-reloaded", "data"),
        prevent_initial_call=True,
    )
    def reload_engine(clicks, count):
        if not clicks:
            raise PreventUpdate
        ws.jobs.reload()
        return (count or 0) + 1

    @app.server.route("/files/<path:rel>")
    def files(rel):
        """A file a study kept, or its log: `results.file_url`. Only the kinds
        a study keeps, and only from inside the runs directory."""
        from flask import abort, send_from_directory

        from atisim import studies

        if not rel.lower().endswith(studies.KEPT):
            abort(404)
        return send_from_directory(ws.root.resolve(), rel)

    fly.register(app, ws)
    lab.register(app, ws)
    results.register(app, ws)
    diagnostics.register(app, ws)
    compare.register(app, ws)
    setup.register(app, ws)
    analyses.register(app, ws)
    return app


def _warm() -> None:
    """Draw the card's sketches, then compile every test point's flight."""
    from atisim import cockpit

    for key in cockpit.TEST_POINTS:
        card.profile(key)
    for key in cockpit.TEST_POINTS:
        cockpit.Flight(key)


def _engine_items(ws) -> list:
    """In development mode: where the engine runs, its commit, and a reload."""
    engine = ws.jobs.engine()
    if engine["mode"] != "subprocess":
        return [html.Span([ui.icon("engine", 13), "engine in-process"],
                          className="ati-status-item",
                          title="The engine runs inside the application. Start it with "
                                "atisim ui --dev to run the engine in a worker that you "
                                "can reload.")]
    if not engine.get("alive"):
        text, tone = "engine worker: starts with the next job", "neutral"
    elif engine.get("changed"):
        text, tone = "engine changed on disk", "warn"
    else:
        started = datetime.fromtimestamp(engine["started"]).strftime("%H:%M")
        text, tone = f"engine worker at {engine['sha'][:7]}, loaded {started}", "ok"
    button = [html.Button("Reload engine", id="engine-reload", n_clicks=0,
                          className="ati-status-button", type="button",
                          title="Stop the worker; the next job imports the engine "
                                "from disk again")] if engine.get("alive") else []
    return [html.Span([ui.state(text, tone, "tabler:engine", 13), *button],
                      className="ati-status-item ati-status-engine")]


def main(argv=None) -> None:
    """`python -m atisim.apps.shell RUNS` opens the whole app on Start."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="?", default=Path("runs"))
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args(argv)
    build_app(args.runs).run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
