"""Analyses: every simulation that is not one open-loop flight.

    /analyses                          the first analysis
    /analyses?analysis=ensemble        one analysis
    /analyses?analysis=ensemble&base=setup
                                       on the run being edited in Setup
    /analyses?from=<report directory>  a report's own spec, to run again

The analyses are listed in the explorer, by family. The page is the chosen
analysis: its head (what it does, what it needs, its engine code), a property
grid with its validation, a Run button and the job's stages as they happen
(`form`, which the research scripts' pages use too), and its earlier
results. The analyses themselves are `atisim.analyses`; this page holds no
engine code. A finished analysis is a report directory, and Results opens it.

Component ids here are `analysis-*`, and the parameter inputs use the pattern
type `aparam`, so none of Setup's pattern-matching callbacks ever sees them.
"""

from pathlib import Path

import dash_mantine_components as dmc
from dash import ALL, Input, Output, State, dcc, html, no_update

from atisim import analyses as registry
from atisim import run
from atisim.analysis import report
from atisim.analysis import runs as runs_mod
from atisim.apps import components as ui
from atisim.apps import setup
from atisim.apps.jobs import LIVE

# The run an analysis on a run starts from, when the URL does not say.
DEFAULT_BASE = {"ensemble": "preset:dryden"}


def _input(param: run.Param, value):
    pid = {"type": "aparam", "name": param.name}
    label = {"aria-label": param.label}
    if param.check == "choice":
        many = len(param.choices) > 12  # the studies: type to find a script
        return dmc.Select(id=pid, value=value, data=list(param.choices),
                          searchable=many, maxDropdownHeight=320,
                          nothingFoundMessage="No script by that name",
                          comboboxProps={"withinPortal": True}, **label)
    if param.check in ("text", "arguments"):
        return dmc.TextInput(id=pid, value=value or "", debounce=400, **label)
    number = dict(value=value if value is not None else "", hideControls=True,
                  debounce=400, **label)
    if param.check == "integer":
        return dmc.NumberInput(id=pid, allowDecimal=False, allowNegative=False, **number)
    unit = dict(rightSection=setup._unit(param.unit), rightSectionWidth=44,
                rightSectionPointerEvents="none") if param.unit else {}
    if param.check == "optional":
        return dmc.NumberInput(id=pid, placeholder="CRUISE", **unit, **number)
    return dmc.NumberInput(id=pid, **unit, **number)


def _hint(param: run.Param) -> str | None:
    if param.check == "optional":
        return "Empty: the aircraft's CRUISE value"
    if param.check == "text":
        return "Numbers separated by commas"
    if param.check == "arguments":
        return "As on a command line; empty for none"
    return None


def coerce(param: run.Param, value):
    """A form value as the analysis spec holds it.

    Empty is None (the CRUISE value, for the optional rows). A whole number from
    an integer input is an int even if the browser sent 8.0, so validation says
    what is wrong with a value rather than with its type.
    """
    if value == "" or value is None:
        return "" if param.check in ("text", "arguments") else None
    if param.check == "integer" and isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def base_options(from_report: str | None = None) -> list[dict]:
    options = [{"value": "setup", "label": "The run in Setup"}]
    if from_report:
        options.append({"value": "report", "label": f"The run in {from_report}"})
    options += [{"value": f"preset:{name}", "label": f"Preset: {name}"}
                for name in run.PRESETS]
    return options


def resolve_base(choice: str | None, setup_data, from_base) -> run.RunSpec | None:
    """The base `RunSpec` a choice names, or None if it cannot be read."""
    try:
        if choice == "setup":
            return setup.spec_of(setup_data)
        if choice == "report":
            return run.RunSpec.from_json(from_base) if from_base else None
        if choice and choice.startswith("preset:"):
            return run.preset(choice.split(":", 1)[1])
    except (KeyError, TypeError, ValueError):
        return None
    return None


def build_spec(key: str, names, values, base_choice, setup_data, from_base) -> \
        registry.AnalysisSpec:
    a = registry.ANALYSES[key]
    params = {p.name: p.default for p in a.params}
    by_name = {p.name: p for p in a.params}
    for pid, value in zip(names or [], values or []):
        name = pid["name"] if isinstance(pid, dict) else pid
        if name in by_name:
            params[name] = coerce(by_name[name], value)
    base = resolve_base(base_choice, setup_data, from_base) if a.needs_base else None
    aspec = registry.default(key, base)
    return aspec._replace(params=params)


def _resolve(ws, key, base, source):
    """The analysis, its starting values and its base, from the URL's parts."""
    from_spec, from_name = None, None
    if source:
        try:
            meta = report.read_report(ws.root / source).meta
            from_spec = registry.AnalysisSpec.from_json(meta["spec"])
            from_name = source
            key = from_spec.analysis
        except (OSError, KeyError, TypeError, ValueError):
            from_spec = None
    key = key if key in registry.ANALYSES else next(iter(registry.ANALYSES))
    values = dict(from_spec.params) if from_spec else {}
    if from_spec and from_spec.base is not None:
        base = "report"
    base = base or DEFAULT_BASE.get(key, "setup")
    return key, values, base, from_spec, from_name


def form(ws, key: str | None = None, base: str | None = None, source: str | None = None,
         fixed: dict | None = None):
    """The run form: the base, the parameters, validation, Run, and progress.

    `fixed` pins parameters to values (a research script's page pins `script`):
    their inputs stay in the page, hidden, so the callbacks read them as usual.
    """
    key, values, base, from_spec, from_name = _resolve(ws, key, base, source)
    a = registry.ANALYSES[key]
    values.update(fixed or {})
    rows, hidden = [], []
    for p in a.params:
        control = _input(p, values.get(p.name, p.default))
        if fixed and p.name in fixed:
            hidden.append(control)
            continue
        # The unit sits inside a number input; a text row carries it beside.
        aside = html.Span(p.unit, className="ati-muted") if p.check == "text" else None
        rows.append(setup._row(p.label, control, aside, hint=_hint(p)))
    base_block = html.Div([
        html.Div("Runs on", className="ati-group-title"),
        setup._row("Run", dmc.Select(
            id="analysis-base", value=base, data=base_options(from_name),
            searchable=True, comboboxProps={"withinPortal": True},
            **{"aria-label": "The run this analysis runs on"}), None),
        html.Div(id="analysis-base-note", className="ati-row-note"),
    ], style={} if a.needs_base else {"display": "none"})
    return html.Div([
        base_block,
        html.Div("Parameters", className="ati-group-title") if rows else None,
        *rows,
        html.Div(hidden, hidden=True) if hidden else None,
        html.Div(script_note(values.get("script", "sanity")), id="analysis-script-note",
                 className="ati-script-note")
        if key == "study" and not (fixed and "script" in fixed) else None,
        html.Div(id="analysis-issues", className="ati-form-issues"),
        html.Div([
            dmc.Button("Run analysis" if key != "study" else "Run script",
                       id="analysis-run", leftSection=ui.icon("player-play", 14),
                       size="sm", n_clicks=0, disabled=True),
            html.Span(id="analysis-run-note", className="ati-muted"),
        ], className="ati-form-actions"),
        html.Div("Progress", className="ati-group-title"),
        html.Div(id="analysis-progress"),
        dcc.Store(id="analysis-key", data=key),
        dcc.Store(id="analysis-from", data=None if from_spec is None or from_spec.base is None
                  else from_spec.base.to_dict()),
    ], className="ati-settings ati-form")


def layout(ws, key: str | None = None, base: str | None = None, source: str | None = None):
    """The page. `source` is a report directory whose spec fills the form."""
    key = _resolve(ws, key, base, source)[0]
    a = registry.ANALYSES[key]
    earlier = [r for r in runs_mod.scan(ws.root) if r.error is None and r.analysis == key]
    facts = [("Family", a.family),
             ("Needs", "a run to start from" if a.needs_base else "nothing: it stands alone"),
             ("Engine code", html.Span(a.source, className="ati-mono"))]
    return html.Div([
        ui.page_head(a.label, a.description, facts),
        html.Div([
            html.Section(form(ws, key, base, source), className="ati-doc-main"),
            html.Aside([
                html.H2("Earlier results", className="ati-side-title"),
                ui.result_links(earlier, empty="None yet. A finished run of this analysis "
                                                "is listed here and under Results."),
            ], className="ati-doc-side"),
        ], className="ati-doc-cols"),
    ], className="ati-doc")


def script_note(name) -> list:
    """What the chosen script says it does, and where it is."""
    from atisim import studies

    try:
        script = studies.find(name)
    except KeyError:
        return []
    return [html.Span(f"scripts/{script.path.name}", className="ati-mono"),
            html.Span(script.description)]


def issues_view(issues, noun: str = "analysis") -> list:
    if not issues:
        return [ui.state("No issues", "ok"),
                html.Span(f" The {noun} is ready to run.", className="ati-muted")]
    out = []
    for issue in issues:
        tone = "fail" if issue.level == "error" else "warn"
        out.append(html.Div([ui.state("error" if issue.level == "error" else "warning", tone),
                             html.Span(issue.message)],
                            className="ati-issue", style={"cursor": "default"}))
    return out


def progress_view(job: dict | None, key: str | None = None) -> list:
    if not job or job.get("kind") != "analysis":
        what = ("Not run yet. Press Run script" if key == "study"
                else "No analysis yet. Press Run analysis")
        return [html.Div(f"{what}; its stages and their times show here.",
                         className="ati-empty")]
    rows = []
    for i, stage in enumerate(job["stages"]):
        last = i == len(job["stages"]) - 1
        if job["state"] == "failed" and last:
            status = ui.state("failed", "fail")
        elif stage["ended"] is None and job["state"] not in ("failed", "done"):
            status = ui.state("running", "neutral", ui.ICONS["running"])
        else:
            status = ui.state("done", "ok")
        rows.append(html.Tr([html.Td(stage.get("message", stage["name"])), html.Td(status),
                             html.Td(f"{stage['elapsed']:.1f} s", className="num")]))
    out = [html.Div([html.Span("Job ", className="ati-muted"),
                     html.Span(job["spec"]["name"], className="ati-mono")],
                    style={"padding": "4px 12px"}),
           html.Table([html.Thead(html.Tr([html.Th("Stage"), html.Th("State"),
                                           html.Th("Elapsed", className="num")])),
                       html.Tbody(rows)], className="ati-table")]
    if job["state"] == "failed":
        out.append(dmc.Alert(job["error"], title="The analysis failed", color="red",
                             icon=ui.icon("circle-x", 16), m="8px 12px"))
    elif job["state"] == "done":
        name = Path(job["path"]).name
        out.append(html.Div([
            dcc.Link(dmc.Button("Open in Results", leftSection=ui.icon("chart-line", 14),
                                size="xs"),
                     href=f"/results?run={name}", id="analysis-open"),
            html.Span(job.get("summary", ""), className="ati-muted"),
        ], className="ati-controls", style={"padding": "8px 12px"}))
    out.append(html.Details([html.Summary("Log", style={"padding": "4px 12px",
                                                        "cursor": "pointer"}),
                             html.Pre("\n".join(job["log"]), className="ati-log")]))
    return out


def register(app, ws) -> None:
    @app.callback(
        Output("analysis-issues", "children"), Output("analysis-run", "disabled"),
        Output("analysis-run-note", "children"), Output("analysis-base-note", "children"),
        Input({"type": "aparam", "name": ALL}, "value"), Input("analysis-base", "value"),
        Input("poll", "n_intervals"), Input("analysis-job", "data"),
        State({"type": "aparam", "name": ALL}, "id"), State("analysis-key", "data"),
        State("spec", "data"), State("analysis-from", "data"),
    )
    def check(values, base_choice, _n, job_id, ids, key, setup_data, from_base):
        a = registry.ANALYSES[key]
        aspec = build_spec(key, ids, values, base_choice, setup_data, from_base)
        issues = registry.validate(aspec)
        note = ""
        if a.needs_base and aspec.base is not None:
            b = aspec.base
            note = (f"{b.name}: {run.KIND_LABELS.get(b.wind.kind, b.wind.kind)}, "
                    f"{b.aircraft} at {float(b.airspeed_mps):.1f} m/s, "
                    f"{float(b.altitude_m):.0f} m, dt {b.dt:g} s"
                    + (", with turbulence on top" if b.overlay is not None else ""))
        job = ws.jobs.get(job_id)
        busy = bool(job and job["state"] in LIVE)
        errors = [i for i in issues if i.level == "error"]
        why = ("An analysis is running." if busy
               else f"Fix this first: {errors[0].message}" if errors
               else f"Writes {registry.output_name(aspec)}-<commit> under the runs "
                    "directory.")
        noun = "script" if key == "study" else "analysis"
        return issues_view(issues, noun), busy or bool(errors), why, note

    @app.callback(
        Output("analysis-job", "data"),
        Output("poll", "disabled", allow_duplicate=True),
        Input("analysis-run", "n_clicks"),
        State({"type": "aparam", "name": ALL}, "value"),
        State({"type": "aparam", "name": ALL}, "id"),
        State("analysis-base", "value"), State("analysis-key", "data"),
        State("spec", "data"), State("analysis-from", "data"),
        prevent_initial_call=True,
    )
    def start(n, values, ids, base_choice, key, setup_data, from_base):
        if not n:
            return no_update, no_update
        aspec = build_spec(key, ids, values, base_choice, setup_data, from_base)
        if any(i.level == "error" for i in registry.validate(aspec)):
            return no_update, no_update
        return ws.jobs.submit_analysis(aspec), False

    @app.callback(
        Output("analysis-script-note", "children"),
        Input({"type": "aparam", "name": "script"}, "value"),
        prevent_initial_call=True,
    )
    def describe(name):
        return script_note(name)

    @app.callback(
        Output("analysis-progress", "children"),
        Input("poll", "n_intervals"), Input("analysis-job", "data"),
        Input("url", "pathname"), State("analysis-key", "data"),
    )
    def follow(_n, job_id, _path, key):
        return progress_view(ws.jobs.get(job_id), key)
