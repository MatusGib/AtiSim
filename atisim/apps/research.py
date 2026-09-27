"""Research scripts: the scripts in `scripts/`, found by topic and run from here.

    /research                  every script, under its topic (`studies.TOPICS`)
    /research?script=NAME      one script: what it does, a Run form, its results

A script runs unchanged as a study (`atisim.studies`, through the `study`
analysis), so the form is the Analyses page's own (`analyses.form`) with the
script pinned, and a finished run is a report under Results. This page adds
what the dropdown could not: where a script sits in the record, what its
docstring says before you run it, and what it wrote the last times it ran.
"""

from dash import dcc, html

from atisim import studies
from atisim.analysis import runs as runs_mod
from atisim.apps import analyses
from atisim.apps import components as ui


def _study_rows(ws) -> list:
    return [r for r in runs_mod.scan(ws.root) if r.error is None and r.group == "study"]


def _last_runs(rows) -> dict:
    """The newest result of each script (rows are newest first)."""
    last = {}
    for r in rows:
        last.setdefault(r.script, r)
    return last


def _last_cell(row):
    if row is None:
        return html.Span("never run", className="ati-muted")
    return dcc.Link(row.created_local, href=f"/results?run={row.name}", title=row.name,
                    className="ati-num")


def index(ws):
    groups = studies.by_topic()
    last = _last_runs(_study_rows(ws))
    count = sum(len(s) for _, s in groups)
    sections = []
    for topic, scripts in groups:
        sections.append(html.Section([
            html.Div([
                html.H2(topic.title, className="ati-topic-title"),
                html.Span(f"{len(scripts)} script{'s' if len(scripts) != 1 else ''}",
                          className="ati-muted"),
            ], className="ati-topic-head"),
            html.P(topic.blurb, className="ati-topic-blurb"),
            html.Table([
                html.Thead(html.Tr([html.Th("Script"), html.Th("What it does"),
                                    html.Th("Last run", className="num")])),
                html.Tbody([html.Tr([
                    html.Td(dcc.Link(s.name, href=f"/research?script={s.name}",
                                     className="ati-script-link")),
                    html.Td(s.description),
                    html.Td(_last_cell(last.get(s.name)), className="num ati-nowrap"),
                ]) for s in scripts]),
            ], className="ati-table ati-script-table"),
        ], id=f"topic-{topic.key}", className="ati-topic"))
    return html.Div([
        ui.page_head(
            "Research scripts",
            f"The {count} scripts in scripts/, by topic. Running one runs it unchanged: what it prints, and every "
            "figure, table and text file it writes, is kept as a result.",
            [("Scripts", str(count)), ("Topics", str(len(groups))),
             ("Kept", "images, CSV, JSON, Markdown and text files")]),
        html.Nav([dcc.Link([t.title, html.Span(str(len(s)), className="ati-xcount")],
                           href=f"#topic-{t.key}", className="ati-topic-jump")
                  for t, s in groups], className="ati-topic-nav",
                 **{"aria-label": "Topics"}),
        html.Div(sections, className="ati-topics"),
    ], className="ati-doc")


def script_page(ws, name: str):
    try:
        script = studies.find(name)
    except KeyError:
        return html.Div([
            html.P([html.Strong("There is no research script "), html.Code(name), "."]),
            dcc.Link("See every research script", href="/research"),
        ], className="ati-empty")
    topic = studies.topic_of(name)
    rows = [r for r in _study_rows(ws) if r.script == name]
    about = studies.about(script)
    siblings = [s for t, group in studies.by_topic() if t.key == topic.key
                for s in group if s.name != name]
    facts = [
        ("Topic", dcc.Link(topic.title, href=f"/research#topic-{topic.key}")),
        ("File", html.Span(f"scripts/{script.path.name}", className="ati-mono")),
        ("Last run", _last_cell(rows[0] if rows else None)),
    ]
    return html.Div([
        ui.page_head(name, script.description, facts),
        html.Div([
            html.Section([
                html.Div("What it does", className="ati-group-title"),
                html.Div([html.P(p) for p in about] if about else
                         html.P("The script's docstring says no more than its first line.",
                                className="ati-muted"),
                         className="ati-about"),
                html.Div("Run it", className="ati-group-title"),
                html.P("It runs as it is, from the repository root, in its own process. "
                       "What it prints and what it writes are kept, and Results shows them.",
                       className="ati-form-note"),
                analyses.form(ws, "study", fixed={"script": name}),
            ], className="ati-doc-main"),
            html.Aside([
                html.H2("Earlier results", className="ati-side-title"),
                ui.result_links(rows, empty="Never run from here yet."),
                html.H2(f"Also in {topic.title}", className="ati-side-title")
                if siblings else None,
                html.Ul([html.Li(dcc.Link(s.name, href=f"/research?script={s.name}",
                                          title=s.description)) for s in siblings],
                        className="ati-side-list") if siblings else None,
            ], className="ati-doc-side"),
        ], className="ati-doc-cols"),
    ], className="ati-doc")


def layout(ws, script: str | None = None):
    return script_page(ws, script) if script else index(ws)
