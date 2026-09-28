"""The Lab: fly any case with a few values changed, run a chosen analysis,
compare two results -- in the test card's world.

    /lab                       home: what the Lab does, and the recent results
    /lab/case?case=NAME        a preset, its few values, Run    (&from=RUN: that run's spec)
    /lab/result?run=NAME       a flown result in plain words, and its recorder
    /lab/analysis?key=KEY      one of the four analyses, with its defaults
    /lab/compare?a=RUN&b=RUN   two results side by side

The middle of three modes (`components.mode_switch`): more than the test card's
four live test points, far less than engineering mode's every value. What the
Lab offers is `atisim.lab`; runs go through the app's job runner like any
other, so a Lab result is a run engineering mode opens in full. The look is the
test card's (`assets/card.css`, its Lab section): an index card beside a main
card on the graphite kneeboard, labels left of values in ruled cells, red only
for limits, blue-ink stamps, and grease pencil for what the pilot changed.
"""

import math
from urllib.parse import quote

import dash_mantine_components as dmc
import numpy as np
import plotly.graph_objects as go
from dash import ALL, Input, Output, State, dcc, html, no_update
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots

from atisim import analyses as registry
from atisim import lab, run
from atisim.analysis import figures
from atisim.analysis import runs as runs_mod
from atisim.apps import analyses as analyses_page
from atisim.apps import components as ui
from atisim.apps.card import INK, PENCIL
from atisim.apps.explorer import AIRCRAFT_SHORT, FAMILIES, preset_family
from atisim.apps.jobs import LIVE

STAMP, RULE, STOCK = "#1b4d9b", "#d9dce0", "#fbfbf9"
RECENT = 8
DEFAULT_CASE = "vortex-hannibal"
_GRAPH = {"displaylogo": False, "responsive": True}


# ---------------------------------------------------------------------------
# The frame: a bar, the index card, the main card
# ---------------------------------------------------------------------------


def _flights(ws) -> list:
    return [r for r in runs_mod.scan(ws.root) if r.error is None and r.group == "flight"]


def _row(label, href, active, meta=None, title=None, depth=0):
    return dcc.Link([html.Span(label, className="lab-ilabel"),
                     html.Span(meta, className="lab-imeta") if meta is not None else None],
                    href=href, title=title, style={"--depth": depth},
                    className="lab-irow" + (" is-active" if active else ""))


def _section(title, count, children):
    return html.Section([
        html.Div([html.Span(title), html.Span(str(count), className="lab-count")
                  if count is not None else None], className="lab-isec"),
        *children,
    ])


def index_card(ws, here: str):
    """The Lab's index: the cases, the analyses and the recent results."""
    by_family: dict = {}
    for name, spec in run.PRESETS.items():
        by_family.setdefault(preset_family(spec), []).append(_row(
            name, f"/lab/case?case={name}", here == f"case:{name}",
            AIRCRAFT_SHORT.get(spec.aircraft, spec.aircraft), spec.wind.source, 1))
    cases = []
    for key, title in FAMILIES:
        rows = by_family.get(key, [])
        if rows:
            cases.append(html.Div(title, className="lab-igroup"))
            cases.extend(rows)
    flights = _flights(ws)
    results = [_row(ui.middle(r.name, 30), f"/lab/result?run={r.name}",
                    here == f"result:{r.name}", _verdict_word(r.verdict),
                    f"{r.name}\n{r.kind} · {r.created_local}")
               for r in flights[:RECENT]]
    if not flights:
        results = [html.Div("No flights yet. Fly a case.", className="lab-iempty")]
    return html.Nav([
        html.Div(className="tc-clip", **{"aria-hidden": "true"}),
        html.Span("Index", className="tc-field"),
        html.Div([
            _row("Lab home", "/lab", here == "home"),
            _section("Cases", len(run.PRESETS), cases),
            _section("Analyses", len(lab.ANALYSES), [
                _row(registry.ANALYSES[a.key].label, f"/lab/analysis?key={a.key}",
                     here == f"analysis:{a.key}",
                     "on a run" if registry.ANALYSES[a.key].needs_base else None, a.learn)
                for a in lab.ANALYSES]),
            _section("Results", len(flights), results + [
                _row("Compare two results", "/lab/compare", here == "compare")]),
        ], className="lab-iscroll"),
    ], className="tc-card lab-index", **{"aria-label": "Lab index"})


def _g(value: float) -> str:
    """A load factor, signed, with a true minus."""
    return f"{value:+.2f} g".replace("-", "−")


def _verdict_word(verdict: str):
    return {"warning": "marginal", "no checks": "no checks"}.get(verdict, verdict)


def frame(ws, crumbs: list, here: str, main: list):
    trail = []
    for i, label in enumerate(crumbs):
        if i:
            trail.append(html.Span(ui.icon("chevron-right", 12), className="lab-crumb-sep",
                                   **{"aria-hidden": "true"}))
        trail.append(html.Span(label, className="lab-crumb"
                               + (" is-current" if i == len(crumbs) - 1 else "")))
    return html.Div([
        html.Header([
            dcc.Link("AtiSim", href="/", className="tc-wordmark"),
            html.Nav(trail, className="lab-crumbs", **{"aria-label": "Where you are"}),
            html.Span(className="tc-bar-spacer"),
            ui.mode_switch("lab"),
        ], className="tc-bar lab-bar"),
        html.Div([
            index_card(ws, here),
            html.Main([html.Div(className="tc-clip", **{"aria-hidden": "true"}),
                       html.Div(main, className="lab-mscroll")],
                      className="tc-card lab-main"),
        ], className="lab-body"),
    ], className="tc-board lab")


def _head(title, cells):
    """The main card's ruled header box: the title, then label | value cells."""
    return html.Div([
        html.Div(html.H1(title, className="tc-title"), className="tc-head-cell tc-head-main"),
        *[html.Div([html.Span(label, className="tc-field"),
                    html.Span(value, className="tc-value")], className="tc-head-cell")
          for label, value in cells],
    ], className="tc-head lab-head")


def _lr(label, body, cls=""):
    """One label-left ruled row, the card's grammar."""
    return html.Div([html.Span(label, className="tc-field"),
                     html.Div(body, className="lab-lr-body")], className="lab-lr " + cls)


def _stamp(label, href, icon="arrow-right"):
    return dcc.Link([html.Span(label), ui.icon(icon, 16)], href=href, className="tc-stamp")


def _plain(label, href):
    return dcc.Link(label, href=href, className="lab-plain")


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def home(ws):
    flights = _flights(ws)
    rows = []
    for r in flights[:6]:
        try:
            s = _summary(ws, r.name)
            peak = _g(s.peak)
        except Exception:  # an unreadable run must not take the page down
            peak = "-"
        rows.append(html.Tr([
            html.Td(dcc.Link(r.name, href=f"/lab/result?run={r.name}")),
            html.Td(r.kind), html.Td(peak, className="lab-num"),
            html.Td(_check_word(r.verdict)),
        ]))
    recent = html.Table([
        html.Thead(html.Tr([html.Th("Result"), html.Th("Case"), html.Th("Peak", className="lab-num"),
                            html.Th("Checks")])),
        html.Tbody(rows),
    ], className="lab-table") if rows else html.P(
        "No flights yet. Fly a case and its result card appears here.", className="lab-muted")
    start = html.Table([html.Tbody([
        html.Tr([html.Td(html.Span("1", className="tc-tp-num"), className="tc-col-tp"),
                 html.Td([html.Div("Fly a case", className="tc-tp-title"),
                          html.P("Pick one under Cases. Its aircraft, speed, height and "
                                 "the field's strength and size are yours to change; the "
                                 "rest stays as published.", className="tc-tp-summary")]),
                 html.Td(_stamp("Fly a case", f"/lab/case?case={DEFAULT_CASE}", "plane"),
                         className="tc-col-fly")]),
        html.Tr([html.Td(html.Span("2", className="tc-tp-num"), className="tc-col-tp"),
                 html.Td([html.Div("Run an analysis", className="tc-tp-title"),
                          html.P("The seed ensemble, the modes, the step-size convergence "
                                 "and the JSBSim cross-code, each with its defaults and a "
                                 "line on what it shows.", className="tc-tp-summary")]),
                 html.Td(_stamp("Analyse", "/lab/analysis?key=modes", "list-details"),
                         className="tc-col-fly")]),
        html.Tr([html.Td(html.Span("3", className="tc-tp-num"), className="tc-col-tp"),
                 html.Td([html.Div("Compare two results", className="tc-tp-title"),
                          html.P("Two flights side by side: their values, their peaks, and "
                                 "their load factor on one chart.",
                                 className="tc-tp-summary")]),
                 html.Td(_stamp("Compare", "/lab/compare", "git-compare"),
                         className="tc-col-fly")]),
    ])], className="tc-points lab-start")
    main = [
        _head("Lab", [("Cases", str(len(run.PRESETS))), ("Analyses", str(len(lab.ANALYSES))),
                      ("Results", str(len(flights)))]),
        _lr("Objective", html.P(
            f"Between the test card and engineering mode. Fly any of the {len(run.PRESETS)} "
            "preset cases with a few of its values changed, and read the result in plain "
            "words; run "
            "four analyses with their defaults; compare two results. Every result is "
            "saved like any other, and engineering mode opens it in full.",
            className="lab-lead")),
        start,
        html.H2("Recent results", className="lab-h2"),
        recent,
    ]
    return frame(ws, ["Lab"], "home", main)


# -- a case ------------------------------------------------------------------------


def _base_spec(ws, case: str | None, source: str | None):
    """The spec the form starts from: a result's own (`from`) or a preset's."""
    if source:
        spec = run.spec_of(ws.root / source)
        if spec is not None:
            return spec, source
    name = case if case in run.PRESETS else DEFAULT_CASE
    return run.PRESETS[name], None


def _value_input(name, value, unit=None, integer=False, label=None):
    field = ui.number_input({"type": "lab-val", "name": name}, value, size="sm",
                            **{"aria-label": f"{label}, {unit}" if label and unit
                               else label or name})
    if unit and not integer:
        field.rightSection = html.Span(unit, className="lab-unit")
        field.rightSectionWidth = 48
        field.rightSectionPointerEvents = "none"
    return field


def _field_row(label, control, name):
    return html.Div([
        html.Span(label, className="tc-field"),
        html.Div(control, className="lab-form-control"),
        html.Div(id={"type": "lab-mark", "name": name}, className="lab-form-mark"),
    ], className="lab-form-row")


def case_page(ws, case: str | None, source: str | None = None):
    if case is not None and case not in run.PRESETS and not source:
        # Said, not replaced: an unknown name once opened vortex-hannibal as if
        # that were the case asked for.
        return frame(ws, ["Lab", "Cases"], "", [
            _head("No such case", []),
            _lr("Case", html.P(["There is no case ", html.Code(case),
                                ". Pick one under Cases in the index."],
                               className="lab-lead")),
        ])
    base, from_run = _base_spec(ws, case, source)
    kind = run.KIND_LABELS.get(base.wind.kind, base.wind.kind)
    rows = [
        _field_row("Aircraft", dmc.Select(
            id={"type": "lab-val", "name": "aircraft"}, value=base.aircraft, size="sm",
            data=[{"value": a, "label": a} for a in lab.AIRCRAFT],
            comboboxProps={"withinPortal": True}, **{"aria-label": "Aircraft"}), "aircraft"),
        _field_row("Airspeed", _value_input("airspeed_mps", base.airspeed_mps, "m/s",
                                            label="Airspeed"), "airspeed_mps"),
        _field_row("Altitude", _value_input("altitude_m", base.altitude_m, "m",
                                            label="Altitude"), "altitude_m"),
    ]
    for p in lab.key_params(base):
        rows.append(_field_row(p.label, _value_input(
            f"wind.{p.name}", run.param_value(base.wind, p), p.unit if p.unit != "-" else None,
            integer=p.check == "integer", label=p.label), f"wind.{p.name}"))
    title = base.wind.preset or base.name
    main = [
        _head(title, [("Wind field", kind), ("Aircraft", base.aircraft),
                      ("Starts from", from_run or "the preset")]),
        _lr("What it is", html.P([kind, ". Source: ", base.wind.source or "none", "."],
                                 className="lab-lead")),
        html.Div([
            html.Div([html.H2("Your values", className="lab-h2"),
                      html.P("Change any of these; the rest of the case stays as "
                             "published. A changed number is yours, and is marked so.",
                             className="lab-muted")], className="lab-form-head"),
            html.Div(rows, className="lab-form"),
        ]),
        html.Div(id="lab-limits"),
        html.Div([
            html.Button([html.Span("Run the case"), ui.icon("player-play", 16)],
                        id="lab-run", n_clicks=0, type="button", className="tc-stamp",
                        disabled=True),
            html.Span(id="lab-run-note", className="lab-muted"),
        ], className="lab-actions"),
        html.Div(id="lab-progress"),
        dcc.Store(id="lab-base", data=base.to_dict()),
        dcc.Store(id="lab-spec"),
        dcc.Store(id="lab-job"),
        dcc.Interval(id="lab-poll", interval=600, disabled=True),
    ]
    return frame(ws, ["Lab", "Cases", title], f"case:{title}", main)


# -- a result -------------------------------------------------------------------------


def _summary(ws, name: str) -> lab.Summary:
    loaded = ws.loaded(name)
    return lab.summarise(loaded.series.t, loaded.series.n_z, loaded.run.checks)


def _check_word(verdict: str):
    tone = {"pass": "ok", "warning": "warn", "fail": "fail", "error": "fail"}.get(verdict,
                                                                                  "neutral")
    return html.Span(ui.state(_verdict_word(verdict), tone, size=14),
                     className="lab-check lab-check-" + tone)


def recorder(series_list, names=None, record=None) -> go.Figure:
    """Load factor, height change and pitch on one time axis, drawn on card stock.

    `record` (an `atisim.records.Record`) draws what the source measured behind
    the load factor, as the engineering strips draw it."""
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=("Load factor, g", "Height change, m", "Pitch, deg"))
    colours = (INK, STAMP)
    for i, s in enumerate(series_list):
        name = names[i] if names else None
        colour = colours[i % 2]
        t = np.asarray(s.t)
        for row, y in enumerate((s.n_z, np.asarray(s.altitude) - s.altitude[0], s.theta_deg),
                                start=1):
            fig.add_trace(go.Scatter(x=t, y=y, mode="lines", name=name, legendgroup=name,
                                     showlegend=bool(names) and row == 1,
                                     line={"color": colour, "width": 1.6}), row=row, col=1)
        if not names:
            # The larger excursion from level flight, as the card's headline.
            hi = int(np.argmax(np.abs(np.asarray(s.n_z) - 1.0)))
            fig.add_annotation(x=float(t[hi]), y=float(s.n_z[hi]), row=1, col=1,
                               text=f"{float(s.n_z[hi]):+.2f} g", showarrow=True,
                               arrowcolor=PENCIL, ax=34, ay=-18,
                               font={"family": "Permanent Marker, cursive", "size": 17,
                                     "color": PENCIL})
    fig.update_layout(
        template=None, height=460, margin={"l": 56, "r": 16, "t": 34, "b": 36},
        paper_bgcolor=STOCK, plot_bgcolor=STOCK, hovermode="x unified",
        font={"family": "B612, system-ui, sans-serif", "size": 12, "color": INK},
        legend={"orientation": "h", "x": 1, "xanchor": "right", "y": 1.08},
    )
    fig.update_xaxes(gridcolor=RULE, linecolor=INK, zeroline=False, showline=True)
    fig.update_yaxes(gridcolor=RULE, linecolor=INK, zeroline=False, showline=True)
    fig.update_xaxes(title_text="time, s", row=3, col=1)
    if record is not None:
        n_z = np.concatenate([np.asarray(s.n_z) for s in series_list])
        low, high = figures.record_on_row(fig, record, 1, float(n_z.min()), float(n_z.max()))
        pad = 0.06 * (high - low)
        fig.update_yaxes(range=[low - pad, high + pad], row=1, col=1)
    for a in fig.layout.annotations:
        if a.text in ("Load factor, g", "Height change, m", "Pitch, deg"):
            a.update(x=0, xanchor="left", font={"family": "B612", "size": 12, "color": INK})
    return fig


def _values_rows(spec: run.RunSpec):
    prov = run.provenance(spec)
    out = [html.Tr([html.Td("Aircraft"), html.Td(spec.aircraft), html.Td("")])]
    for field, label, unit in (("airspeed_mps", "Airspeed", "m/s"),
                               ("altitude_m", "Altitude", "m")):
        out.append(html.Tr([html.Td(label), html.Td(f"{getattr(spec, field):g} {unit}"),
                            html.Td(_mark(prov.get(field)))]))
    for p in lab.key_params(spec):
        out.append(html.Tr([html.Td(p.label),
                            html.Td(f"{run.param_value(spec.wind, p):g} "
                                    f"{p.unit if p.unit != '-' else ''}".strip()),
                            html.Td(_mark(prov.get(p.name)))]))
    return out


def _mark(prov):
    """As published, or the pilot's own value in grease pencil."""
    if prov is None:
        return ""
    if prov.status == "sourced":
        return html.Span("as published", className="lab-muted", title=prov.note)
    return html.Span("your value", className="lab-pencil", title=prov.note)


def result_page(ws, name: str | None):
    flights = {r.name: r for r in _flights(ws)}
    if name not in flights:
        return frame(ws, ["Lab", "Results"], "", [
            _head("No such result", []),
            _lr("Result", html.P(["There is no flight ", html.Code(str(name)),
                                  " in the runs directory. Pick one in the index."],
                                 className="lab-lead")),
        ])
    row = flights[name]
    loaded = ws.loaded(name)
    s = _summary(ws, name)
    spec = run.spec_of(ws.root / name)
    by_hand = loaded.run.meta.get("flown_by_hand")
    case = ((spec.wind.preset or spec.name) if spec is not None
            else f"{by_hand['title']}, flown by hand" if by_hand else row.kind)
    lines = []
    for c in loaded.run.checks:
        word, tone, icon = ui.check_state(c)
        lines.append(html.Tr([
            html.Td(ui.state(word, tone, icon, 14), className="lab-check-" + tone),
            html.Td(c["name"]),
            html.Td(f"{c['value']:.4g}", className="lab-num"),
        ], title=c.get("detail", "")))
    sigma = "" if math.isnan(s.sigma) else f"RMS normal load {s.sigma:.2f} g over 5 s. "
    # The headline is the larger excursion from level flight: a vortex can take
    # the load to -1 g, and a headline of the highest value alone hides that.
    low_first = (1.0 - s.low) > (s.peak - 1.0)
    head, t_head = (s.low, s.t_low) if low_first else (s.peak, s.t_peak)
    other, t_other = (s.peak, s.t_peak) if low_first else (s.low, s.t_low)
    # Still air has no turbulence to rate: the index then measures the controls.
    still = loaded.kind.startswith("none")
    if spec is not None:
        values = html.Table(html.Tbody(_values_rows(spec)), className="lab-kv")
    elif by_hand:
        values = html.P(f"Flown by hand on the test card: {by_hand['title']}, the 747 at "
                        "its cruise condition. The controls are the pilot's inputs, so no "
                        "spec flies this run again.", className="lab-muted")
    else:
        values = html.P("This run was written before runs kept their spec, so its "
                        "values are only in engineering mode.", className="lab-muted")
    again = (_stamp("Fly this test point again", f"/fly?tp={quote(by_hand['test_point'])}",
                    "plane") if by_hand
             else _stamp("Fly again with changes", f"/lab/case?from={quote(name)}", "plane"))
    main = [
        _head(case, [("Aircraft", row.aircraft), ("Flown", row.created_local),
                     ("Checks", _check_word(row.verdict))]),
        _lr("Lowest load factor" if low_first else "Peak load factor", [
            html.Div(_g(head), className="lab-peak"),
            html.P(f"at {t_head:.1f} s. {'Highest' if low_first else 'Lowest'} {_g(other)} "
                   f"at {t_other:.1f} s. Level flight is +1 g.", className="lab-muted"),
        ]),
        _lr("Turbulence", html.P(
            [html.Strong("None: still air"), ". The load came from the controls: ", sigma,
             "In turbulence, moderate is from 0.2 g and severe from 0.3 g (Misaka 2008)."]
            if still else
            [html.Strong(s.severity.capitalize()), ". ", sigma,
             "Moderate from 0.2 g, severe from 0.3 g (Misaka 2008)."],
            className="lab-lead")),
        _lr("Values flown", values),
        _lr("Checks", html.Table(html.Tbody(lines), className="lab-kv lab-checks")),
        html.H2("Recorder", className="lab-h2"),
        html.Div(dcc.Graph(figure=recorder([loaded.series], record=loaded.record),
                           config=_GRAPH), className="lab-recorder"),
        html.P(f"Grey: what the source measured. {loaded.record.case}. {loaded.record.note}",
               className="lab-muted") if loaded.record is not None else None,
        html.Div([
            again,
            _plain("Compare with another result", f"/lab/compare?a={quote(name)}"),
            _plain("Open in engineering mode", f"/results?run={quote(name)}"),
        ], className="lab-actions"),
    ]
    return frame(ws, ["Lab", "Results", ui.middle(name, 40)], f"result:{name}", main)


# -- an analysis ------------------------------------------------------------------------


def analysis_page(ws, key: str | None):
    key = key if key in lab.LEARN else lab.ANALYSES[0].key
    a = registry.ANALYSES[key]
    earlier = [r for r in runs_mod.scan(ws.root) if r.error is None and r.analysis == key]
    main = [
        _head(a.label, [("Family", a.family),
                        ("Needs", "a run to start from" if a.needs_base else "nothing")]),
        _lr("What you learn", html.P(lab.LEARN[key], className="lab-lead")),
        _lr("What it does", html.P(a.description, className="lab-lead")),
        html.H2("Run it", className="lab-h2"),
        html.Div(analyses_page.form(ws, key), className="lab-analysis-form"),
        html.H2("Earlier results", className="lab-h2"),
        html.P("An analysis writes a report; engineering mode opens it.", className="lab-muted"),
        ui.result_links(earlier, empty="None yet."),
    ]
    return frame(ws, ["Lab", "Analyses", a.label], f"analysis:{key}", main)


# -- compare ----------------------------------------------------------------------------------


def compare_page(ws, a: str | None, b: str | None):
    flights = _flights(ws)
    names = {r.name for r in flights}
    a = a if a in names else None
    same = b is not None and b == a
    b = b if b in names and b != a else None
    data = [{"value": r.name, "label": r.name} for r in flights]

    def pick(ident, value, label):
        return html.Div([html.Span(label, className="tc-field"),
                         dmc.Select(id=ident, value=value, data=data, searchable=True,
                                    size="sm", placeholder="Choose a flight",
                                    nothingFoundMessage="No flight by that name",
                                    comboboxProps={"withinPortal": True},
                                    **{"aria-label": f"Result {label}"})],
                        className="lab-pick")

    body = [html.Div([pick("lab-compare-a", a, "A"), pick("lab-compare-b", b, "B")],
                     className="lab-picks")]
    chosen = [n for n in (a, b) if n]
    if len(chosen) < 2:
        body.append(html.P("A and B were the same flight, so B is cleared. Choose a "
                           "different flight for B." if same else
                           "Choose two flights to put them side by side.",
                           className="lab-lead"))
    else:
        rows = []
        facts = []
        for n in chosen:
            spec = run.spec_of(ws.root / n)
            s = _summary(ws, n)
            loaded = ws.loaded(n)
            # The field values the Lab lets a reader change, so two flights of
            # one case with a different V0 show what differs, not only the result.
            wind_rows = {} if spec is None else {
                p.label: f"{run.param_value(spec.wind, p):g} "
                         f"{p.unit if p.unit != '-' else ''}".strip()
                for p in lab.key_params(spec)}
            facts.append({
                "Case": (spec.wind.preset or spec.name) if spec else loaded.kind,
                "Aircraft": loaded.run.meta["aircraft"]["key"],
                "Airspeed": f"{loaded.run.meta['flight_condition']['airspeed_mps']:.1f} m/s",
                "Altitude": f"{loaded.run.meta['flight_condition']['altitude_m']:.0f} m",
                **wind_rows,
                "Peak load factor": _g(s.peak),
                "Lowest load factor": _g(s.low),
                "Turbulence": s.severity,
                "Checks": _verdict_word(s.verdict),
            })
        labels = list(facts[0]) + [k for k in facts[1] if k not in facts[0]]
        for label in labels:
            va, vb = facts[0].get(label, "—"), facts[1].get(label, "—")
            differs = va != vb
            rows.append(html.Tr([html.Td(label), html.Td(va), html.Td(vb),
                                 html.Td(html.Span("differs", className="lab-pencil")
                                         if differs else "")],
                                className="is-different" if differs else None))
        body += [
            html.Table([html.Thead(html.Tr([html.Th(""), html.Th("A"), html.Th("B"),
                                            html.Th("")])),
                        html.Tbody(rows)], className="lab-table lab-compare"),
            html.H2("Recorder", className="lab-h2"),
            html.Div(dcc.Graph(figure=recorder([ws.loaded(n).series for n in chosen],
                                               names=[f"A  {chosen[0]}", f"B  {chosen[1]}"]),
                               config=_GRAPH), className="lab-recorder"),
        ]
    main = [_head("Compare two results", [("A", a or "not chosen"), ("B", b or "not chosen")]),
            *body]
    return frame(ws, ["Lab", "Compare"], "compare", main)


def layout(ws, path: str, query: dict):
    if path == "/lab/case":
        return case_page(ws, query.get("case"), query.get("from"))
    if path == "/lab/result":
        return result_page(ws, query.get("run"))
    if path == "/lab/analysis":
        return analysis_page(ws, query.get("key"))
    if path == "/lab/compare":
        return compare_page(ws, query.get("a"), query.get("b"))
    return home(ws)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


def spec_from(base: dict, ids: list, values: list) -> run.RunSpec:
    """The form's spec: the base with every value the form holds applied."""
    got = {i["name"]: v for i, v in zip(ids, values)}
    def number(name):
        value = ui.parse_number(got.get(name))
        if isinstance(value, str):
            raise ValueError(f"{value!r} is not a number.")
        return value

    wind = {k[5:]: number(k) for k in got
            if k.startswith("wind.") and number(k) is not None}
    return lab.edit(run.RunSpec.from_json(base), got.get("aircraft"),
                    number("airspeed_mps"), number("altitude_m"), wind)


def register(app, ws) -> None:
    @app.callback(
        Output("lab-spec", "data"),
        Output({"type": "lab-mark", "name": ALL}, "children"),
        Output("lab-limits", "children"), Output("lab-run", "disabled"),
        Output("lab-run-note", "children"),
        Input({"type": "lab-val", "name": ALL}, "value"),
        Input("lab-job", "data"), Input("lab-poll", "n_intervals"),
        State({"type": "lab-val", "name": ALL}, "id"),
        State({"type": "lab-mark", "name": ALL}, "id"), State("lab-base", "data"),
    )
    def check(values, job_id, _n, ids, mark_ids, base):
        if not base:
            raise PreventUpdate
        try:
            spec = spec_from(base, ids, values)
        except (TypeError, ValueError) as exc:
            # In the Limits box too: left as it was, the box went on showing the
            # issues of the value before, beside a field it no longer describes.
            why = f"Fix this first: {str(exc) or 'a value is not a number.'}"
            box = html.Div([html.Div("Limits", className="tc-limits-title"),
                            html.Ul(html.Li(why))], className="tc-limits lab-limits",
                           role="note")
            return no_update, [no_update] * len(mark_ids), box, True, why
        prov = run.provenance(spec)
        marks = []
        for m in mark_ids:
            name = m["name"]
            if name == "aircraft":
                marks.append("" if spec.aircraft == run.RunSpec.from_json(base).aircraft
                             else html.Span("your value", className="lab-pencil"))
            else:
                marks.append(_mark(prov.get(name.removeprefix("wind."))))
        issues = run.validate(spec)
        errors = [i for i in issues if i.level == "error"]
        notes = [html.Li(["Fix this first: " if i.level == "error" else "", i.message])
                 for i in issues] + [html.Li(c) for c in spec.caveats]
        limits = html.Div([html.Div("Limits", className="tc-limits-title"),
                           html.Ul(notes)], className="tc-limits lab-limits",
                          role="note") if notes else None
        job = ws.jobs.get(job_id)
        busy = bool(job and job["state"] in LIVE)
        note = ("Flying." if busy else f"Fix this first: {errors[0].message}" if errors
                else "Saved under the runs directory, like any run.")
        return spec.to_dict(), marks, limits, busy or bool(errors), note

    @app.callback(
        Output({"type": "lab-val", "name": "airspeed_mps"}, "value"),
        Output({"type": "lab-val", "name": "altitude_m"}, "value"),
        Input({"type": "lab-val", "name": "aircraft"}, "value"),
        State("lab-base", "data"),
        prevent_initial_call=True,
    )
    def cruise(aircraft, base):
        """A new aircraft starts at its own cruise condition, where it is valid."""
        from atisim.aircraft import CRUISE

        if not aircraft or aircraft not in CRUISE:
            raise PreventUpdate
        if base and aircraft == base.get("aircraft"):
            return ui.shown_number(base["airspeed_mps"]), ui.shown_number(base["altitude_m"])
        return (ui.shown_number(CRUISE[aircraft]["airspeed"]),
                ui.shown_number(CRUISE[aircraft]["altitude"]))

    @app.callback(
        Output("lab-job", "data"), Output("lab-poll", "disabled"),
        Input("lab-run", "n_clicks"), State("lab-spec", "data"),
        prevent_initial_call=True,
    )
    def fly(n, data):
        if not n or not data:
            raise PreventUpdate
        spec = run.RunSpec.from_json(data)
        if run.errors(spec):
            raise PreventUpdate
        return ws.jobs.submit(spec), False

    @app.callback(
        Output("lab-progress", "children"),
        Output("lab-poll", "disabled", allow_duplicate=True),
        Output("url", "pathname", allow_duplicate=True),
        Output("url", "search", allow_duplicate=True),
        Input("lab-poll", "n_intervals"), State("lab-job", "data"),
        prevent_initial_call=True,
    )
    def follow(_n, job_id):
        job = ws.jobs.get(job_id)
        if job is None:
            raise PreventUpdate
        if job["state"] == "done":
            name = job["path"].replace("\\", "/").rstrip("/").split("/")[-1]
            return no_update, True, "/lab/result", f"?run={name}"
        rows = [html.Tr([html.Td(st.get("message", st["name"])),
                         html.Td(f"{st['elapsed']:.1f} s", className="lab-num")])
                for st in job["stages"]]
        if job["state"] == "failed":
            return html.Div([html.Div("The run failed", className="tc-limits-title"),
                             html.P(job["error"])], className="tc-limits lab-limits"), \
                True, no_update, no_update
        return html.Table([html.Tbody(rows)], className="lab-kv lab-stages"), False, \
            no_update, no_update

    @app.callback(
        Output("url", "search", allow_duplicate=True),
        Input("lab-compare-a", "value"), Input("lab-compare-b", "value"),
        prevent_initial_call=True,
    )
    def compare(a, b):
        parts = [f"a={quote(a)}" if a else "", f"b={quote(b)}" if b else ""]
        return "?" + "&".join(p for p in parts if p)
