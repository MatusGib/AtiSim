"""The explorer: everything in engineering mode, in one tree down the left.

    Overview
    Fly a run          the presets by family; on Setup, the run being edited
    Analyses           by family
    Research scripts   by topic (`studies.TOPICS`)
    Results            flights, analysis reports, script outputs

COMSOL's Model Builder and an IDE's explorer, played straight: one place that
shows what exists and where it is. Every item is a link to its own page, so
the tree is rebuilt from the URL by the router and the item on screen is
marked. Which groups are open, and the filter, live in the browser
(`assets/explorer.js`): typing in the filter shows every match in every group.

On Setup the run being edited sits at the top of Fly a run, with its model
tree (Aircraft, Flight condition, Wind field, Solver, Output) under it. Those
nodes are Setup's own (`setup.tree_nodes`), with their ids and callbacks.
"""

from dash import dcc, html

from atisim import analyses as registry
from atisim import run, studies
from atisim.analysis import runs as runs_mod
from atisim.apps import components as ui

AIRCRAFT_SHORT = {"boeing747": "747", "boeing747_approach": "747 app", "boeing737": "737",
                  "cherokee": "Cherokee", "cessna172": "C172"}
FAMILIES = (
    ("encounters", "Encounters"),
    ("test-inputs", "Test inputs"),
    ("turbulence", "Turbulence"),
)
RESULT_GROUPS = (("flight", "Flights"), ("report", "Analysis reports"),
                 ("study", "Script outputs"))
VERDICT_ICON = {"pass": "ok", "warning": "warn", "fail": "fail", "error": "fail"}


def preset_family(spec: run.RunSpec) -> str:
    """Which Fly a run group a preset sits in; the Start page's grouping."""
    if spec.wind.kind in ("Sinusoid", "OneMinusCosine"):
        return "test-inputs"
    if spec.wind.kind in run.STOCHASTIC_KINDS:
        return "turbulence"
    return "encounters"


def _item(label, href, *, active, meta=None, icon=None, title=None, depth=0):
    return dcc.Link([
        ui.icon(icon, 15) if icon else None,
        html.Span(label, className="ati-xlabel"),
        html.Span(meta, className="ati-xmeta") if meta is not None else None,
    ], href=href, className="ati-xitem" + (" is-active" if active else ""),
        title=title, style={"--depth": depth})  # explorer.js sets aria-current


def _group(key, title, count, items, *, open_=False, depth=1):
    return html.Details([
        html.Summary([html.Span(ui.icon("chevron-right", 13), className="ati-xchev"),
                      html.Span(title, className="ati-xlabel"),
                      html.Span(str(count), className="ati-xcount")],
                     className="ati-xgroup-head", style={"--depth": depth}),
        html.Div(items, className="ati-xgroup-body"),
    ], open=open_ or any("is-active" in (getattr(i, "className", "") or "") for i in items),
        className="ati-xgroup", **{"data-key": key})


def _section(key, title, icon, count, children, *, open_=True):
    return html.Details([
        html.Summary([html.Span(ui.icon("chevron-right", 13), className="ati-xchev"),
                      ui.icon(icon, 16),
                      html.Span(title, className="ati-xlabel"),
                      html.Span(str(count), className="ati-xcount") if count is not None
                      else None],
                     className="ati-xsec-head"),
        html.Div(children, className="ati-xsec-body"),
    ], open=open_, className="ati-xsec", **{"data-key": key})


def active_key(path: str, query: dict) -> str:
    """The item on screen, as the explorer names it."""
    if path == "/start":
        return "overview"
    if path == "/setup":
        return "setup"
    if path == "/analyses":
        key = query.get("analysis")
        return "research" if key == "study" else f"analysis:{key or ''}"
    if path == "/research":
        script = query.get("script")
        return f"script:{script}" if script else "research"
    if path == "/results":
        return f"result:{query.get('run') or ''}"
    return ""


def tree(ws, path: str, query: dict, setup_nodes=None) -> list:
    """The explorer's children for the page at `path`."""
    act = active_key(path, query)

    # -- Fly a run --------------------------------------------------------
    by_family: dict = {}
    for name, spec in run.PRESETS.items():
        by_family.setdefault(preset_family(spec), []).append(_item(
            name, f"/setup?preset={name}", active=query.get("preset") == name and path == "/setup",
            meta=AIRCRAFT_SHORT.get(spec.aircraft, spec.aircraft),
            title=spec.wind.source, depth=2))
    fly = []
    if setup_nodes is not None:
        fly.append(html.Div(setup_nodes, className="ati-xmodel", role="tree",
                            **{"aria-label": "The run being edited"}))
    fly += [_group(f"family:{k}", title, len(by_family.get(k, [])), by_family.get(k, []))
            for k, title in FAMILIES if by_family.get(k)]
    fly.append(_item("Blank run", "/setup?preset=blank", active=query.get("preset") == "blank" and path == "/setup",
                     icon="file", title="Still air, the 747 at cruise; set every value",
                     depth=1))

    # -- Analyses -----------------------------------------------------------
    groups = []
    for family, _blurb in registry.FAMILIES:
        items = [_item(a.label, f"/analyses?analysis={key}", active=act == f"analysis:{key}",
                       meta="on a run" if a.needs_base else None,
                       title=a.description, depth=2)
                 for key, a in registry.ANALYSES.items()
                 if a.family == family and key != "study"]
        if items:
            groups.append(_group(f"afamily:{family}", family, len(items), items))
    n_analyses = sum(1 for k in registry.ANALYSES if k != "study")

    # -- Research scripts ---------------------------------------------------
    topics = studies.by_topic()
    research = [_item("All research scripts", "/research", active=act == "research", icon="list-search", depth=1)]
    for topic, scripts in topics:
        items = [_item(s.name, f"/research?script={s.name}", active=act == f"script:{s.name}", title=s.description, depth=2)
                 for s in scripts]
        research.append(_group(f"topic:{topic.key}", topic.title, len(items), items))
    n_scripts = sum(len(s) for _, s in topics)

    # -- Results --------------------------------------------------------------
    rows = runs_mod.scan(ws.root)
    results = []
    for group, title in RESULT_GROUPS:
        items = []
        for r in rows:
            if r.error is None and r.group == group:
                items.append(_item(
                    ui.middle(r.name, 34), f"/results?run={r.name}", active=act == f"result:{r.name}", meta=_verdict(r),
                    title=f"{r.name}\n{r.kind} · {r.created_local}", depth=2))
        if items:
            results.append(_group(f"results:{group}", title, len(items), items))
    broken = [r for r in rows if r.error is not None]
    if broken:
        results.append(_group("results:error", "Unreadable", len(broken), [
            html.Div([ui.state("error", "fail", size=13), html.Span(r.name)],
                     className="ati-xitem is-static", title=r.error,
                     style={"--depth": 2}) for r in broken]))
    if not rows:
        results.append(html.Div("No results yet. Fly a run, or run an analysis or a script.",
                                className="ati-xempty"))

    return [
        _item("Overview", "/start", active=act == "overview", icon="home"),
        _section("fly", "Fly a run", "plane", len(run.PRESETS), fly),
        _section("analyses", "Analyses", "list-details", n_analyses, groups),
        _section("research", "Research scripts", "script", n_scripts, research),
        _section("results", "Results", "chart-line", sum(1 for r in rows if r.error is None),
                 results),
        html.Div("Nothing matches the filter.", className="ati-xempty ati-xnomatch",
                 hidden=True),
    ]


def _verdict(r: runs_mod.RunRow):
    tone = VERDICT_ICON.get(r.verdict)
    if tone is None:
        return None  # a report has no verdict to show
    word = {"warning": "warn"}.get(r.verdict, r.verdict)
    return ui.state(word, tone, size=12)


def panel(children=None):
    """The explorer's frame: the filter above, the tree below."""
    return html.Aside([
        html.Label([
            html.Span("Filter the explorer", className="ati-visually-hidden"),
            html.Span(ui.icon("search", 14), className="ati-xsearch-icon"),
            dcc.Input(id="explorer-filter", type="search", placeholder="Filter everything",
                      className="ati-xfilter", autoComplete="off"),
            html.Kbd("/", className="ati-xkbd", title="Press / to filter"),
        ], className="ati-xsearch"),
        html.Nav(children, id="explorer", className="ati-xtree",
                 **{"aria-label": "Explorer"}),
    ], className="ati-explorer")


def crumbs(ws, path: str, query: dict) -> list[tuple[str, str | None]]:
    """Where the page on screen sits: (label, link) from the root down."""
    if path == "/start":
        return [("Overview", None)]
    if path == "/setup":
        preset = query.get("preset")
        return [("Fly a run", None), (preset or "Setup", None)]
    if path == "/analyses":
        key = query.get("analysis")
        if key == "study":
            return [("Research scripts", "/research"), ("Run a script", None)]
        a = registry.ANALYSES.get(key) or next(iter(registry.ANALYSES.values()))
        return [("Analyses", None), (a.family, None), (a.label, None)]
    if path == "/research":
        script = query.get("script")
        if not script:
            return [("Research scripts", None)]
        topic = studies.topic_of(script)
        return [("Research scripts", "/research"),
                (topic.title, f"/research#topic-{topic.key}"), (script, None)]
    if path == "/results":
        name = query.get("run")
        rows = {r.name: r for r in runs_mod.scan(ws.root) if r.error is None}
        row = rows.get(name) or next(iter(rows.values()), None)
        out = [("Results", None)]
        if row is not None:
            out += [(dict(RESULT_GROUPS)[row.group], None), (ui.middle(row.name, 48), None)]
        view = query.get("view")
        if view in ("diagnostics", "compare"):
            out.append((view.capitalize(), None))
        return out
    return []


def crumb_bar(items) -> list:
    out = []
    for i, (label, href) in enumerate(items):
        if i:
            out.append(html.Span(ui.icon("chevron-right", 12), className="ati-crumb-sep",
                                 **{"aria-hidden": "true"}))
        last = i == len(items) - 1
        cls = "ati-crumb" + (" is-current" if last else "")
        out.append(dcc.Link(label, href=href, className=cls) if href and not last
                   else html.Span(label, className=cls))
    return out
