"""Setup: the Model Builder. Tree, Settings, Graphics preview, and a dock.

COMSOL's layout played straight. The tree is the run's definition (Run ›
Aircraft › Flight condition › Wind field › Solver › Output), each node carrying
the status `run.validate` gives it; it sits in the shell's explorer, at the top
of Fly a run (`tree_nodes`), so the page itself is Settings, Graphics and the
dock. Settings shows the selected node
as a property grid, every row *label | input | unit* with a provenance marker.
Graphics is a live geometry preview drawn from the spec without flying. The
dock holds Messages, Progress, Log, Checks and Script.

**The signature detail is the marker.** Editing a sourced value flips it to
Declared, because it is no longer the source's number; Reset to preset puts the
source's number back. `run.with_param` decides that, not this module.

The UI is thin: validation, provenance, geometry, the preview and the command
that reproduces a run are all computed in `atisim.run` and tested there. This
module lays them out and wires the callbacks.
"""

from pathlib import Path

import dash
import dash_mantine_components as dmc
from dash import ALL, Input, Output, State, dcc, html, no_update

from atisim import run
from atisim.aircraft import CRUISE, REGISTRY
from atisim.analysis import figures
from atisim.apps import components as ui
from atisim.apps.jobs import LIVE

NODES = (
    ("run", "Run", "player-play", 0),
    ("aircraft", "Aircraft", "plane", 1),
    ("condition", "Flight condition", "gauge", 1),
    ("wind", "Wind field", "wind", 1),
    ("solver", "Solver", "adjustments-horizontal", 1),
    ("output", "Output", "folder", 1),
)
NODE_LABELS = {k: label for k, label, _, _ in NODES}

# Which tree node owns each validated field.
FIELD_NODE = {"name": "run", "aircraft": "aircraft", "airspeed_mps": "condition",
              "altitude_m": "condition", "dt": "solver", "strip": "solver",
              "seconds": "solver", "lead_in": "wind", "wind": "wind",
              "overlay": "wind", "stage_sampled": "solver", "gust_lag": "solver",
              "wing_tail": "solver", "fidelity": "solver"}

STAGE_ORDER = ("validating", "trimming", "flying", "checks", "diagnostics", "writing")
STAGE_LABELS = {"validating": "Validate the spec", "trimming": "Trim",
                "flying": "Fly", "checks": "Run the checks",
                "diagnostics": "Probe the diagnostics", "writing": "Write the artifact"}


def node_of(field: str) -> str:
    return FIELD_NODE.get(field, "wind")


def spec_of(data) -> run.RunSpec:
    """The spec in the store, or the blank run when the store is empty."""
    if not data:
        return run.BLANK
    try:
        return run.RunSpec.from_json(data)
    except (TypeError, ValueError, KeyError):
        return run.BLANK


# ---------------------------------------------------------------------------
# Property grid
# ---------------------------------------------------------------------------


def _unit(text: str):
    return html.Span(text, className="ati-muted", style={"fontSize": "11.5px"})


def _number(name: str, value, unit: str, placeholder: str | None = None,
            label: str | None = None):
    return dmc.NumberInput(
        id={"type": "num", "name": name}, value=value if value is not None else "",
        rightSection=_unit(unit), rightSectionWidth=44,
        rightSectionPointerEvents="none", hideControls=True, debounce=400,
        placeholder=placeholder, **{"aria-label": f"{label or name}, {unit}"},
    )


def _param_input(name: str, param: run.Param, value):
    """The input a parameter's `check` calls for: a select for a choice, a
    whole-number input for a seed, a number with its unit otherwise."""
    if param.check == "choice":
        return dmc.Select(id={"type": "select", "name": name}, value=value,
                          data=list(param.choices), comboboxProps={"withinPortal": True},
                          **{"aria-label": param.label})
    if param.check == "integer":
        return dmc.NumberInput(
            id={"type": "num", "name": name}, value=value if value is not None else "",
            allowDecimal=False, allowNegative=False, hideControls=True, debounce=400,
            **{"aria-label": param.label})
    return _number(name, value, param.unit, label=param.label)


def _aside(name: str, prov: run.Provenance | None, resettable: bool):
    children = []
    if prov is not None:
        children.append(html.Span(ui.marker(prov.status, prov.note),
                                  id={"type": "marker-slot", "name": name}))
    if resettable:
        children.append(dmc.Tooltip(
            html.Button(ui.icon("arrow-back-up", 14), id={"type": "reset", "name": name},
                        className="ati-reset", n_clicks=0, type="button",
                        disabled=prov is None or prov.status == "sourced",
                        **{"aria-label": f"Reset {name} to the source value"}),
            label="Reset to preset: put the source's value back",
        ))
    return html.Div(children, className="ati-row-aside")


def _row(label, control, aside=None, hint: str | None = None):
    return html.Div([
        html.Label([label] + ([html.Small(hint)] if hint else []), className="ati-row-label"),
        control,
        aside if aside is not None else html.Div(),
    ], className="ati-row")


def _head(title: str, text: str):
    return html.Div([html.H2(title), html.P(text)], className="ati-settings-head")


def _group(title: str):
    return html.Div(title, className="ati-group-title")


def _value_row(label: str, text: str, mono: bool = False):
    """A read-only value: plain text that wraps. A box would read as an input
    and cut the value off."""
    return html.Div([
        html.Label(label, className="ati-row-label"),
        html.Div(text, className="ati-value" + (" ati-mono" if mono else "")),
    ], className="ati-row ati-row-wide")


def wind_derived(spec: run.RunSpec) -> list:
    """The read-only rows under the wind parameters. A callback redraws them on
    every edit, because the rows above them keep their inputs while you type."""
    if run.errors(spec):
        return []
    geo = run.geometry(spec)
    return [
        _group("Derived from the spec"),
        _value_row("Start", f"{geo.start_north:+.0f} m north"),
        _value_row("Duration", f"{geo.seconds:.2f} s"),
        _value_row("Window", geo.window_name
                   + (f", north {geo.window[0]:+.0f} to {geo.window[1]:+.0f} m"
                      if geo.window else "")),
    ]


def settings_panel(node: str, spec: run.RunSpec, ws) -> list:
    prov = run.provenance(spec)
    geo = None if run.errors(spec) else run.geometry(spec)

    if node == "run":
        return [
            _head("Run", "The run's name. It names the artifact directory, "
                         "{name}-{aircraft}-{git sha}."),
            _group("Identity"),
            _row("Name", dmc.TextInput(id={"type": "text", "name": "name"},
                                       value=spec.name, debounce=400,
                                       **{"aria-label": "Run name"})),
        ]

    if node == "aircraft":
        cruise = CRUISE.get(spec.aircraft)
        note = (f"Valid only near its CRUISE condition: {cruise['airspeed']:.1f} m/s at "
                f"{cruise['altitude']:.0f} m." if cruise else "")
        return [
            _head("Aircraft", "The derivative set that flies the run. " + note),
            _group("Model"),
            _row("Aircraft", dmc.Select(id={"type": "select", "name": "aircraft"},
                                        value=spec.aircraft, searchable=True,
                                        data=sorted(REGISTRY),
                                        comboboxProps={"withinPortal": True},
                                        **{"aria-label": "Aircraft"})),
        ]

    if node == "condition":
        return [
            _head("Flight condition", "Where the aircraft is trimmed. Each aircraft "
                                      "is valid only near its CRUISE entry."),
            _group("Trim point"),
            _row("Airspeed", _number("airspeed_mps", spec.airspeed_mps, "m/s", label="Airspeed"),
                 _aside("airspeed_mps", prov["airspeed_mps"], True)),
            _row("Altitude", _number("altitude_m", spec.altitude_m, "m", label="Altitude"),
                 _aside("altitude_m", prov["altitude_m"], True),
                 hint="above ground for a microburst"),
        ]

    if node == "wind":
        kind = spec.wind.kind
        options = [{"value": name, "label": name} for name in run.PRESETS]
        options.append({"value": "none", "label": "still air (no field)"})
        rows = [
            _head("Wind field", f"{run.KIND_LABELS.get(kind, kind)}. Source: "
                                f"{spec.wind.source}."),
            _group("Field"),
            _row("Preset", dmc.Select(id={"type": "select", "name": "wind.preset"},
                                      value=spec.wind.preset or "none", data=options,
                                      comboboxProps={"withinPortal": True},
                                      **{"aria-label": "Wind field preset"}),
                 hint="loads the field's sourced values"),
        ]
        params = run.PARAMETERS.get(kind, ())
        if params:
            rows.append(_group("Parameters"))
        base = run.PRESETS.get(spec.wind.preset or "")
        for p in params:
            rows.append(_row(
                p.label, _param_input(f"wind.{p.name}", p, run.param_value(spec.wind, p)),
                _aside(p.name, prov.get(p.name),
                       base is not None and base.wind.kind == kind),
            ))
        if kind in ("VortexArray", "SingleVortex", "MehtaHannibal"):
            rows.append(_row("Lead-in", _number("lead_in", spec.lead_in, "r0", label="Lead-in"),
                             _aside("lead_in", prov["lead_in"], False),
                             hint="start distance upstream, in core radii"))
        rows.append(_group("Turbulence on top"))
        overlay_kind = spec.overlay.kind if spec.overlay is not None else "none"
        rows.append(_row(
            "Overlay", dmc.Select(
                id={"type": "select", "name": "overlay.kind"}, value=overlay_kind,
                data=[{"value": "none", "label": "none"}]
                + [{"value": k, "label": run.KIND_LABELS[k]} for k in run.STOCHASTIC_KINDS],
                comboboxProps={"withinPortal": True},
                **{"aria-label": "Turbulence overlay"}),
            hint="a frozen random field added with wind.superpose"))
        if spec.overlay is not None:
            for p in run.PARAMETERS[spec.overlay.kind]:
                rows.append(_row(
                    p.label, _param_input(f"overlay.{p.name}", p,
                                          run.param_value(spec.overlay, p)),
                    _aside(f"overlay.{p.name}", prov.get(f"overlay.{p.name}"), False),
                ))
        rows.append(html.Div(wind_derived(spec), id="wind-derived"))
        return rows

    if node == "solver":
        derived = run.geometry(spec).seconds if geo is not None else None
        return [
            _head("Solver", "RK4 at a fixed step, float64. A position-only field is "
                            "sampled at every RK4 stage unless stage sampling is "
                            "off."),
            _group("Integration"),
            _row("Time step", _number("dt", spec.dt, "s", label="Time step"),
                 _aside("dt", prov["dt"], False)),
            _row("Duration", _number(
                "seconds", spec.seconds, "s", label="Duration",
                placeholder=f"derived: {derived:.2f}" if derived else "derived"),
                hint="empty: derived from the field"),
            _group("Wind sampling"),
            _row("Stage-sampled wind",
                 dmc.Switch(id={"type": "switch", "name": "stage_sampled"},
                            checked=spec.stage_sampled,
                            **{"aria-label": "Stage-sampled wind"}),
                 hint="on: fourth order in a field; off: a v1.1 number"),
            _row("Kussner gust lag",
                 dmc.Switch(id={"type": "switch", "name": "gust_lag"}, checked=spec.gust_lag,
                            **{"aria-label": "Kussner gust lag"}),
                 hint="opt-in v1.2 correction; needs a short step"),
            _row("Wing-tail gust delay",
                 dmc.Switch(id={"type": "switch", "name": "wing_tail"},
                            checked=spec.wing_tail,
                            **{"aria-label": "Wing-tail gust delay"}),
                 hint="opt-in v1.2 correction; gust rates fitted CG to tail"),
            _group("Loads"),
            _row("Strip-integrated loads",
                 dmc.Switch(id={"type": "switch", "name": "strip"}, checked=spec.strip,
                            **{"aria-label": "Strip-integrated loads"}),
                 hint="roll only; quote the loading-shape sensitivity"),
            _group("Output detail"),
            _row("Fidelity",
                 dmc.SegmentedControl(id={"type": "select", "name": "fidelity"},
                                      value=spec.fidelity,
                                      data=[{"value": "standard", "label": "Standard"},
                                            {"value": "high", "label": "High"}],
                                      size="xs", **{"aria-label": "Fidelity"}),
                 hint="high: every term, force and derivative per sample; same flight"),
        ]

    # output
    directory = run.run_directory(ws.root, spec) if spec.name else None
    return [
        _head("Output", "Each run writes one artifact directory: run.parquet, "
                        "meta.json and checks.json. A run is never overwritten."),
        _group("Artifact"),
        _value_row("Runs directory", str(ws.root), mono=True),
        _value_row("This run writes", directory.name if directory else "(name the run)",
                   mono=True),
    ]


# ---------------------------------------------------------------------------
# Dock contents
# ---------------------------------------------------------------------------


def messages(issues: list[run.Issue]):
    if not issues:
        return html.Div([ui.state("No issues", "ok"),
                         html.Span(" The spec is valid and carries no warnings.",
                                   className="ati-muted")], className="ati-empty")
    rows = []
    for i, issue in enumerate(issues):
        tone = "fail" if issue.level == "error" else "warn"
        rows.append(html.Button([
            ui.state("error" if issue.level == "error" else "warning", tone),
            html.Span(NODE_LABELS[node_of(issue.field)], className="ati-issue-where"),
            html.Span(issue.message),
        ], id={"type": "issue", "node": node_of(issue.field), "i": i},
            className="ati-issue", n_clicks=0, type="button"))
    return rows


def progress(job: dict | None):
    if not job:
        return html.Div("No run yet. Press Run to fly the spec; the stages and "
                        "their times show here.", className="ati-empty")
    done = {s["name"]: s for s in job["stages"]}
    high = run.RunSpec.from_json(job["spec"]).fidelity == "high" if job.get("spec") else False
    rows = []
    for name in STAGE_ORDER:
        if name == "diagnostics" and not high:
            continue
        stage = done.get(name)
        if stage is None:
            status = ui.state("waiting", "neutral", "tabler:circle")
            elapsed = ""
        elif stage["ended"] is None and job["state"] not in ("failed", "done"):
            status = ui.state("running", "neutral", "tabler:loader-2")
            elapsed = f"{stage['elapsed']:.1f} s"
        elif job["state"] == "failed" and name == job["stages"][-1]["name"]:
            status = ui.state("failed", "fail")
            elapsed = f"{stage['elapsed']:.1f} s"
        else:
            status = ui.state("done", "ok")
            elapsed = f"{stage['elapsed']:.1f} s"
        rows.append(html.Tr([html.Td(STAGE_LABELS[name]), html.Td(status),
                             html.Td(elapsed, className="num")]))
    table = html.Table([html.Thead(html.Tr([html.Th("Stage"), html.Th("State"),
                                            html.Th("Elapsed", className="num")])),
                        html.Tbody(rows)], className="ati-table")
    children = [table]
    if job["state"] == "failed":
        children.append(dmc.Alert(job["error"], title="The run failed", color="red",
                                  icon=ui.icon("circle-x", 16), m="8px 12px"))
    else:
        children.append(html.Div(f"Total {job['elapsed']:.1f} s", className="ati-empty"))
    return children


def log(job: dict | None):
    if not job:
        return html.Div("The log fills while a run flies.", className="ati-empty")
    return html.Pre("\n".join(job["log"]), className="ati-log")


def checks_table(job: dict | None):
    if not job or job["state"] != "done":
        return html.Div("The checks show here when a run finishes. They run on "
                        "every run whether or not anyone looks.", className="ati-empty")
    rows = []
    for c in job["checks"]:
        word, tone, icon_name = ui.check_state(c)
        tol = "" if c["tolerance"] is None else f"{c['tolerance']:.4g}"
        rows.append(html.Tr([
            html.Td(c["name"]), html.Td(c["kind"]),
            html.Td(f"{c['value']:.4g}", className="num"),
            html.Td(tol, className="num"),
            html.Td(ui.state(word, tone, icon_name), title=c["detail"]),
        ]))
    name = Path(job["path"]).name
    t = job["worst_index"] if job["worst_index"] is not None else 0
    where = ("the worst sample of the first gate that failed"
             if job["worst_index"] is not None else "t = 0 (no gate failed)")
    head = html.Div([
        dcc.Link(dmc.Button("Open in Results", leftSection=ui.icon("chart-line", 14)),
                 href=f"/results?run={name}&t={t}", id="open-results"),
        html.Span(f"Opens {name} with the cursor at {where}.", className="ati-muted"),
    ], style={"display": "flex", "gap": "10px", "alignItems": "center",
              "padding": "8px 12px"})
    return [head, html.Table(
        [html.Thead(html.Tr([html.Th("Check"), html.Th("Kind"),
                             html.Th("Value", className="num"),
                             html.Th("Tolerance", className="num"), html.Th("Verdict")])),
         html.Tbody(rows)], className="ati-table")]


def script(spec: run.RunSpec, ws):
    command = run.command(spec, out=str(ws.root))
    blocks = [
        ("Command", "script-command", command),
        (f"Spec ({spec.name}.json)", "script-json", spec.to_json()),
        ("Python", "script-python", run.snippet(spec, out=str(ws.root))),
    ]
    children = []
    for title, ident, text in blocks:
        children.append(html.Div([
            html.Div([html.Span(title, style={"fontWeight": 600}),
                      dcc.Clipboard(target_id=ident, title=f"Copy the {title.lower()}",
                                    style={"display": "inline-flex", "cursor": "pointer",
                                           "color": "var(--ati-ink-muted)"})],
                     style={"display": "flex", "justifyContent": "space-between",
                            "alignItems": "center", "margin": "8px 0 4px"}),
            html.Pre(text, id=ident, className="ati-code"),
        ]))
    return html.Div(children, style={"padding": "0 12px 12px"})


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def tree_nodes():
    """The model tree's nodes, for the explorer. Ids and classes are Setup's own."""
    nodes = []
    for key, label, icon_name, depth in NODES:
        nodes.append(html.Button([
            ui.icon(icon_name, 15),
            html.Span(label, className="ati-node-label",
                      **({"id": "tree-run-label"} if key == "run" else {})),
            html.Span(id={"type": "node-status", "id": key}, className="ati-node-status"),
        ], id={"type": "node", "id": key}, className="ati-node", n_clicks=0,
            type="button", style={"--depth": depth}))
    return nodes


def layout(ws):
    dock_tabs = dmc.Tabs(
        id="dock-tabs", value="messages", variant="default",
        children=[
            html.Div([
                dmc.TabsList([
                    dmc.TabsTab("Messages", value="messages", id="tab-messages",
                                leftSection=ui.icon("message-circle", 14)),
                    dmc.TabsTab("Progress", value="progress",
                                leftSection=ui.icon("list-check", 14)),
                    dmc.TabsTab("Log", value="log", leftSection=ui.icon("terminal-2", 14)),
                    dmc.TabsTab("Checks", value="checks",
                                leftSection=ui.icon("checklist", 14)),
                    dmc.TabsTab("Script", value="script", leftSection=ui.icon("code", 14)),
                ], style={"flex": "1 1 auto"}),
                dmc.Tooltip(dmc.ActionIcon(ui.icon("layout-bottombar-collapse", 16),
                                           id="dock-toggle", variant="subtle", color="gray",
                                           size="md", n_clicks=0,
                                           **{"aria-label": "Collapse or expand the dock"}),
                            label="Collapse or expand the dock"),
            ], className="ati-dock-bar", style={"paddingRight": "6px",
                                                "alignItems": "center"}),
            html.Div([
                dmc.TabsPanel(html.Div(id="dock-messages"), value="messages"),
                dmc.TabsPanel(html.Div(id="dock-progress"), value="progress"),
                dmc.TabsPanel(html.Div(id="dock-log"), value="log"),
                dmc.TabsPanel(html.Div(id="dock-checks"), value="checks"),
                dmc.TabsPanel(html.Div(id="dock-script"), value="script"),
            ], className="ati-dock-body"),
        ],
        style={"display": "flex", "flexDirection": "column", "height": "100%"},
    )
    return html.Div([
        # Which finished job the dock already switched tabs for, and the last
        # failed run. They live on this page so the dock callback has no output
        # outside it; session storage keeps them across visits to other pages.
        dcc.Store(id="job-shown", storage_type="session"),
        dcc.Store(id="job-failure", storage_type="session"),
        html.Div([
            html.Div([ui.icon("adjustments", 14), "Settings"], className="ati-panel-title"),
            html.Div(id="settings", className="ati-settings"),
        ], className="ati-panel"),
        html.Div(className="ati-split", role="separator", tabIndex=0,
                 **{"aria-orientation": "vertical", "aria-label": "Resize the settings",
                    "data-var": "--settings-w", "data-min": "400", "data-max": "640"}),
        html.Div([
            html.Div([ui.icon("chart-area", 14), "Graphics"], className="ati-panel-title"),
            dcc.Loading(dcc.Graph(id="preview", config={"displaylogo": False,
                                                        "responsive": True},
                                  className="ati-graph"),
                        type="dot", color="#1f5fbf", delay_show=400,
                        parent_style={"flex": "1 1 auto", "minHeight": 0,
                                      "padding": "6px 10px 0"}),
        ], className="ati-graphics"),
        html.Div(className="ati-split ati-split-h", role="separator", tabIndex=0,
                 **{"aria-orientation": "horizontal", "aria-label": "Resize the dock",
                    "data-var": "--dock-h", "data-min": "120", "data-max": "520",
                    "data-sign": "-1"}),
        html.Div(dock_tabs, className="ati-dock"),
    ], id="setup-root", className="ati-setup", **{"data-split-host": ""})


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


def register(app, ws) -> None:
    @app.callback(
        Output("settings", "children"),
        Input("node", "data"), Input("structure", "data"),
        State("spec", "data"),
    )
    def render_settings(node, _structure, data):
        """Rebuilt only when the node or the structure changes, never on an edit,
        so typing in a field never loses focus."""
        return settings_panel(node or "wind", spec_of(data), ws)

    @app.callback(
        Output("spec", "data", allow_duplicate=True),
        Output("structure", "data", allow_duplicate=True),
        Input({"type": "num", "name": ALL}, "value"),
        Input({"type": "text", "name": ALL}, "value"),
        Input({"type": "select", "name": ALL}, "value"),
        Input({"type": "switch", "name": ALL}, "checked"),
        State("spec", "data"), State("structure", "data"),
        prevent_initial_call=True,
    )
    def edit(_nums, _texts, _selects, _switches, data, structure):
        trigger = dash.ctx.triggered_id
        if not isinstance(trigger, dict):
            return no_update, no_update
        value = dash.ctx.triggered[0]["value"]
        name = trigger["name"]
        spec = spec_of(data)
        if trigger["type"] == "num" and value == "":
            value = None
        restructure = False
        if name == "wind.preset":
            wind = run.BLANK.wind if value == "none" else run.wind_preset(value)
            if wind == spec.wind:
                return no_update, no_update
            new = spec._replace(wind=wind)
            if value != "none" and spec.seconds is not None and spec.wind.kind == "none":
                new = new._replace(seconds=None)
            restructure = True
        elif name == "overlay.kind":
            current = spec.overlay.kind if spec.overlay is not None else "none"
            if value == current:
                return no_update, no_update
            new = run.with_overlay_kind(spec, value)
            restructure = True
        elif name.startswith("overlay."):
            param = name[8:]
            if spec.overlay is None or spec.overlay.params.get(param) == value:
                return no_update, no_update
            new = run.with_overlay_param(spec, param, value)
        elif name.startswith("wind."):
            param = name[5:]
            if spec.wind.params.get(param) == value:
                return no_update, no_update
            new = run.with_param(spec, param, value)
        else:
            if getattr(spec, name) == value:
                return no_update, no_update
            new = run.with_field(spec, name, value)
            restructure = name == "aircraft"
        return new.to_dict(), ((structure or 0) + 1) if restructure else no_update

    @app.callback(
        Output("spec", "data", allow_duplicate=True),
        Output("structure", "data", allow_duplicate=True),
        Input({"type": "reset", "name": ALL}, "n_clicks"),
        State("spec", "data"), State("structure", "data"),
        prevent_initial_call=True,
    )
    def reset(clicks, data, structure):
        trigger = dash.ctx.triggered_id
        if not isinstance(trigger, dict) or not any(clicks or []):
            return no_update, no_update
        spec = spec_of(data)
        name = trigger["name"]
        if name in ("airspeed_mps", "altitude_m"):
            new = run.reset_field(spec, name)
        else:
            new = run.reset_param(spec, name)
        return new.to_dict(), (structure or 0) + 1

    @app.callback(
        Output({"type": "marker-slot", "name": ALL}, "children"),
        Output({"type": "reset", "name": ALL}, "disabled"),
        Input("spec", "data"),
        State({"type": "marker-slot", "name": ALL}, "id"),
        State({"type": "reset", "name": ALL}, "id"),
    )
    def markers(data, marker_ids, reset_ids):
        """The provenance flip, in place: the input keeps its focus."""
        prov = run.provenance(spec_of(data))
        slots = [ui.marker(prov[i["name"]].status, prov[i["name"]].note)
                 if i["name"] in prov else None for i in marker_ids]
        disabled = [i["name"] not in prov or prov[i["name"]].status == "sourced"
                    for i in reset_ids]
        return slots, disabled

    @app.callback(
        Output({"type": "node-status", "id": ALL}, "children"),
        Output({"type": "node", "id": ALL}, "className"),
        Output("tree-run-label", "children"),
        Output("dock-messages", "children"),
        Output("tab-messages", "children"),
        Output("dock-script", "children"),
        Input("spec", "data"), Input("node", "data"), Input("job-failure", "data"),
        State({"type": "node", "id": ALL}, "id"),
    )
    def derived(data, node, failure, node_ids):
        spec = spec_of(data)
        issues = run.validate(spec)
        # A failed run of this very spec leads Messages until the spec changes,
        # on the node where the fix goes.
        if failure and spec_of(failure["spec"]).to_json() == spec.to_json():
            issues = [run.Issue("error", failure.get("field") or "name",
                                f"The run failed. {failure['error']}")] + issues
        worst: dict[str, str] = {}
        for issue in issues:
            key = node_of(issue.field)
            if issue.level == "error" or worst.get(key) != "error":
                worst[key] = issue.level
        statuses, classes = [], []
        for i in node_ids:
            level = worst.get(i["id"])
            if level == "error":
                statuses.append(ui.icon(ui.ICONS["fail"], 15, color="var(--ati-fail)"))
            elif level == "warning":
                statuses.append(ui.icon(ui.ICONS["warn"], 15, color="var(--ati-warn)"))
            else:
                statuses.append(ui.icon(ui.ICONS["ok"], 15, color="var(--ati-ok)"))
            classes.append("ati-node is-selected" if i["id"] == (node or "wind")
                           else "ati-node")
        count = len(issues)
        tab = f"Messages ({count})" if count else "Messages"
        return (statuses, classes, f'Run "{spec.name}"', messages(issues), tab,
                script(spec, ws))

    @app.callback(
        Output("node", "data"),
        Input({"type": "node", "id": ALL}, "n_clicks"),
        Input({"type": "issue", "node": ALL, "i": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def select(node_clicks, issue_clicks):
        trigger = dash.ctx.triggered_id
        if not isinstance(trigger, dict) or not dash.ctx.triggered[0]["value"]:
            return no_update
        return trigger.get("id") or trigger.get("node")

    @app.callback(Output("wind-derived", "children"), Input("spec", "data"))
    def redraw_derived(data):
        return wind_derived(spec_of(data))

    @app.callback(Output("preview", "figure"), Input("spec", "data"))
    def preview(data):
        spec = spec_of(data)
        problems = run.errors(spec)
        label = run.KIND_LABELS.get(spec.wind.kind, spec.wind.kind)
        if problems:
            fig = figures.no_field_preview(label, 0.0)
            fig.update_layout(annotations=[dict(
                x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                text="The preview needs a valid spec.<br>" + problems[0].message,
                font=dict(size=12, color=figures.INK_MUTED))])
            return fig
        args = run.preview_args(spec)
        if args is None:
            return figures.no_field_preview(label, run.geometry(spec).seconds)
        return figures.field_preview(run.build_field(spec), **args)

    @app.callback(
        Output("setup-root", "className"),
        Input("dock-toggle", "n_clicks"),
        State("setup-root", "className"),
        prevent_initial_call=True,
    )
    def toggle_dock(_n, current):
        if "is-dock-collapsed" in (current or ""):
            return "ati-setup"
        return "ati-setup is-dock-collapsed"

    @app.callback(
        Output("run-button", "disabled"), Output("run-tooltip", "label"),
        Input("spec", "data"), Input("poll", "n_intervals"), Input("job", "data"),
    )
    def run_button(data, _n, job_id):
        problems = run.errors(spec_of(data))
        job = ws.jobs.get(job_id)
        if job and job["state"] in LIVE:
            return True, f"A run is in progress ({job['state']})."
        if problems:
            return True, f"Fix this first: {problems[0].message}"
        return False, "Fly this spec, run the checks and write the artifact."

    @app.callback(
        Output("job", "data"),
        Output("poll", "disabled", allow_duplicate=True),
        Output("dock-tabs", "value", allow_duplicate=True),
        Input("run-button", "n_clicks"),
        State("spec", "data"),
        prevent_initial_call=True,
    )
    def start(n, data):
        if not n:
            return no_update, no_update, no_update
        spec = spec_of(data)
        if run.errors(spec):
            return no_update, no_update, "messages"
        return ws.jobs.submit(spec), False, "progress"

    @app.callback(
        Output("dock-progress", "children"), Output("dock-log", "children"),
        Output("dock-checks", "children"),
        Output("dock-tabs", "value", allow_duplicate=True),
        Output("job-shown", "data"), Output("job-failure", "data"),
        Input("poll", "n_intervals"), Input("job", "data"), Input("url", "pathname"),
        State("job-shown", "data"),
        prevent_initial_call="initial_duplicate",
    )
    def follow(_n, job_id, _path, shown):
        """Progress, Log and Checks from the job registry. When a run finishes,
        the dock switches once: to Checks, or to Messages when the run failed."""
        job = ws.jobs.get(job_id)
        switch, failure = no_update, no_update
        if job and job["state"] == "done" and shown != job["id"]:
            switch, shown, failure = "checks", job["id"], None
        elif job and job["state"] == "failed" and shown != job["id"]:
            switch, shown = "messages", job["id"]
            failure = {"spec": job["spec"], "error": job["error"],
                       "field": job["error_field"]}
        return progress(job), log(job), checks_table(job), switch, shown, failure


def spec_store_value(preset_name: str | None) -> dict | None:
    """What loading `/setup?preset=NAME` puts in the spec store."""
    if not preset_name:
        return None
    try:
        return run.preset(preset_name).to_dict()
    except KeyError:
        return None
