"""The Overview: engineering mode's start page.

What the workbench holds and where it is. Three ruled columns name the three
things to do -- fly a run, run an analysis, run a research script -- each with
its first links and the way to all of them; the explorer on the left lists
everything. Recent results is a table a reader scans: name, wind field or
analysis, aircraft, created, the checks in one word, and whether it is stale.
"""

import dash_mantine_components as dmc
from dash import dcc, html

from atisim import run
from atisim.analysis import runs as runs_mod
from atisim.apps import components as ui

def _runs_table(rows):
    body = []
    for r in rows:
        if r.error:
            body.append(html.Tr([
                html.Td(r.name, className="ati-mono"),
                html.Td(r.error, colSpan=3),
                html.Td(ui.state("error", "fail")),
            ], className="is-error"))
            continue
        body.append(html.Tr([
            html.Td(dcc.Link(r.name, href=f"/results?run={r.name}"),
                    className="ati-mono ati-wrap-name"),
            html.Td(r.kind),
            html.Td(r.aircraft),
            html.Td(r.created_local, className="ati-num", style={"whiteSpace": "nowrap"}),
            # Stale goes under the verdict, in the same cell: a column of its own
            # does not fit the pane at 1366 px.
            html.Td([ui.verdict_state(r.verdict)] + ([html.Div(ui.state(
                "stale", "neutral", "tabler:history", 13,
                title="Written by another commit; figures may not match today's code"))]
                if r.stale else []), style={"whiteSpace": "nowrap"}),
        ]))
    return html.Table([
        html.Thead(html.Tr([html.Th("Name"), html.Th("Wind field or analysis"),
                            html.Th("Aircraft"),
                            html.Th("Created"), html.Th("Checks")])),
        html.Tbody(body),
    ], className="ati-table")


def _empty(ws, missing: bool):
    why = (f"The runs directory {ws.root} does not exist yet. The first run creates it."
           if missing else f"The runs directory {ws.root} holds no runs yet.")
    return html.Div([
        html.P([html.Strong("No runs yet. "), why]),
        html.P("Pick a preset on the left, or fly one from a terminal:"),
        html.Pre(f"atisim run --preset vortex-hannibal --out {ws.root}",
                 className="ati-code"),
        html.Div(dcc.Link(dmc.Button("New run", leftSection=ui.icon("plus", 14)),
                          href="/setup?preset=vortex-hannibal"),
                 style={"marginTop": "10px"}),
    ], className="ati-empty", style={"maxWidth": "620px"})


FEATURED = ("vortex-hannibal", "updraft", "dryden", "sinusoid")
RECENT = 12


def _flow_column(title, text, links, more_href, more):
    return html.Section([
        html.H2(title, className="ati-flow-title"),
        html.P(text, className="ati-flow-text"),
        html.Ul([html.Li(link) for link in links], className="ati-flow-links"),
        dcc.Link([more, ui.icon("arrow-right", 14)], href=more_href, className="ati-flow-more"),
    ], className="ati-flow-col")


def layout(ws):
    from atisim import analyses as registry
    from atisim import studies

    rows = runs_mod.scan(ws.root)
    good = [r for r in rows if r.error is None]
    topics = studies.by_topic()
    n_scripts = sum(len(g) for _, g in topics)
    analyses = {k: a for k, a in registry.ANALYSES.items() if k != "study"}
    first_of = {}
    for key, a in analyses.items():
        first_of.setdefault(a.family, key)

    fly = [dcc.Link([html.Span(name, className="ati-flow-name"),
                     html.Span(run.PRESETS[name].wind.source, className="ati-flow-sub")],
                    href=f"/setup?preset={name}") for name in FEATURED if name in run.PRESETS]
    analyse = [dcc.Link([html.Span(family, className="ati-flow-name"),
                         html.Span(f"{sum(1 for a in analyses.values() if a.family == family)}"
                                   f" · {blurb}", className="ati-flow-sub")],
                        href=f"/analyses?analysis={first_of[family]}")
               for family, blurb in registry.FAMILIES if family in first_of]
    research = [dcc.Link([html.Span(t.title, className="ati-flow-name"),
                          html.Span(f"{len(g)} · {t.blurb}", className="ati-flow-sub")],
                         href=f"/research#topic-{t.key}") for t, g in topics[:5]]

    recent = (_runs_table(rows[:RECENT] + [r for r in rows[RECENT:] if r.error])
              if rows else _empty(ws, missing=not ws.root.exists()))
    return html.Div([
        ui.page_head(
            "Engineering workbench",
            "The presets, the analyses and the research scripts, in the explorer on the "
            "left: fly a run, run an analysis or a script, and read what each one wrote.",
            [("Presets", str(len(run.PRESETS))), ("Analyses", str(len(analyses))),
             ("Research scripts", str(n_scripts)), ("Results", str(len(good)))],
            actions=[dcc.Link(dmc.Button("New blank run", variant="default",
                                         leftSection=ui.icon("plus", 14)),
                              href="/setup?preset=blank")]),
        html.Div([
            _flow_column("Fly a run",
                         "Pick a preset. Its sourced values load into Setup, where you "
                         "change any of them, see the field ahead, and press Run.",
                         fly, "/setup?preset=vortex-hannibal",
                         "Start from the Hannibal vortex pair"),
            _flow_column("Run an analysis",
                         "Ensembles, the autopilot, modes, gust response and "
                         "verification. Some start from a run you choose.",
                         analyse, f"/analyses?analysis={next(iter(analyses))}",
                         f"Start with the {next(iter(analyses.values())).label.lower()}"),
            _flow_column("Run a research script",
                         "The scripts in scripts/, by topic. "
                         "Each runs unchanged, and what it writes is kept.",
                         research, "/research", f"All {n_scripts} research scripts"),
        ], className="ati-flow"),
        html.Section([
            html.Div([html.H2("Recent results", className="ati-section-title"),
                      html.Span(f"{len(good)} in the runs directory" if good else "",
                                className="ati-muted")], className="ati-section-head"),
            recent,
        ], className="ati-doc-section"),
    ], className="ati-doc")
