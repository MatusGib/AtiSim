"""Results, Compare workspace: what moved between runs (plan section 3.4).

`/results?view=compare&run=A&runs=B,C` compares A with B and C: a header that
lists every recorded setting that differs, and for a chosen channel the runs
overlaid and each one's difference from A. Runs on different time bases are
interpolated onto A's, and the plot says so.

The engine-development loop starts from A:

- **Fly A again with one change** (A/B by option): the same spec with one
  field changed, flown as a new run; it joins the comparison when written.
- **Refine A**: the convergence analysis with A as its base (dt, dt/2, dt/4),
  which reports the observed order.
- **Fly A at another commit** (A/B by commit): `jobs.JobRunner.submit_commit`.
"""

import json
from pathlib import Path

import dash_mantine_components as dmc
import numpy as np
from dash import Input, Output, State, ctx, dcc, html, no_update
from dash.exceptions import PreventUpdate

from atisim import analyses as registry
from atisim import run as run_mod
from atisim.analysis import devfigures, report
from atisim.analysis import runs as runs_mod
from atisim.apps import components as ui

_GRAPH = {"displaylogo": False, "responsive": True}

SERIES_CHANNELS = {
    "n_z": ("n_z", "g"), "alpha_deg": ("alpha, air-relative", "deg"),
    "theta_deg": ("theta", "deg"), "q_deg": ("q", "deg/s"), "w_up": ("gust up", "m/s"),
    "altitude": ("altitude", "m"), "airspeed": ("airspeed", "m/s"),
    "beta_deg": ("beta", "deg"), "elevator_deg": ("elevator", "deg"),
}

# The spec fields "Fly again with one change" offers, and how to read a value.
CHANGEABLE = {
    "stage_sampled": "true or false", "gust_lag": "true or false",
    "wing_tail": "true or false", "strip": "true or false", "dt": "seconds",
    "fidelity": "standard or high", "airspeed_mps": "m/s", "altitude_m": "m",
    "lead_in": "core radii or m", "seconds": "s",
}

_SKIP = {"created"}


def _flatten(d, prefix=""):
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        elif k not in _SKIP:
            out[key] = json.dumps(v, default=str) if isinstance(v, (list, tuple)) else v
    return out


def differences(metas: list[dict]) -> list[tuple[str, list]]:
    """Every flattened `meta.json` key whose value is not the same in all runs."""
    flat = [_flatten(m) for m in metas]
    keys = sorted(set().union(*flat))
    return [(k, [f.get(k, "(absent)") for f in flat]) for k in keys
            if len({json.dumps(f.get(k, "(absent)"), default=str) for f in flat}) > 1]


def _as_list(value):
    try:
        got = json.loads(value) if isinstance(value, str) else None
    except ValueError:
        return None
    return got if isinstance(got, list) else None


def shown(value, others) -> list[str]:
    """How one run's value reads in the differences table: its lines.

    A list (the caveats) shows its length and the items the other runs lack, so
    two lists that begin alike do not read the same. Other values show whole,
    up to 160 characters, and the cell wraps them."""
    items = _as_list(value)
    if items is not None:
        lists = [o for o in map(_as_list, others) if o is not None]
        extra = [x for x in items if any(x not in o for o in lists)]
        head = f"{len(items)} item{'s' if len(items) != 1 else ''}"
        if extra and len(extra) < len(items):
            head += f", {len(extra)} not in every other run"
        return [head] + [f"+ {str(x)[:160]}" for x in extra]
    text = str(value)
    return [text if len(text) <= 160 else text[:159] + "\u2026"]


def parse_runs(run: str | None, extra: str | None) -> list[str]:
    names = [run] if run else []
    for name in (extra or "").split(","):
        if name and name not in names:
            names.append(name)
    return names


def href(names: list[str]) -> str:
    if not names:
        return "/results?view=compare"
    tail = f"&runs={','.join(names[1:])}" if len(names) > 1 else ""
    return f"/results?view=compare&run={names[0]}{tail}"


def _usable(ws, name):
    path = ws.root / name
    return (path / "run.parquet").exists() and not report.is_report(path)


def _channels(ws, names):
    from atisim.apps import diagnostics

    options = [{"label": f"{label} ({unit})", "value": key}
               for key, (label, unit) in SERIES_CHANNELS.items()]
    diags = [diagnostics.diag(ws, n) for n in names]
    if names and all(d is not None for d in diags):
        common = set.intersection(*(set(d.columns) for d in diags)) - {"t"}
        options += [{"label": f"{c.name} ({c.unit}, High)", "value": f"diag:{c.name}"}
                    for c in diags[0].channels if c.name in common]
    return options


def layout(ws, run: str | None, runs_panel, switch, extra: str | None = None):
    names = [n for n in parse_runs(run, extra) if _usable(ws, n)]
    chips = []
    for i, name in enumerate(names):
        rest = [n for n in names if n != name]
        chips.append(html.Span([
            html.Strong("A" if i == 0 else chr(ord("A") + i)), " ",
            html.Span(name, className="ati-mono"), " ",
            ui.verdict_state(runs_mod.verdict(ws.loaded(name).run.checks)), " ",
            dcc.Link(ui.icon("x", 12), href=href(rest), title=f"Remove {name}",
                     className="ati-chip-remove"),
        ], className="ati-chip"))
    if not names:
        hint = "Choose a run in Add a run to compare, above, to start a comparison."
    elif len(names) == 1:
        hint = "Add a second run above, or fly A again with one change."
    else:
        hint = ""
    spec = run_mod.spec_of(ws.root / names[0]) if names else None
    actions = html.Div([
        html.Div("Fly A again with one change", className="ati-panel-title"),
        html.Div([
            dmc.Select(id="compare-field", data=list(CHANGEABLE), value="gust_lag", w=170,
                       label="Field", size="xs", comboboxProps={"withinPortal": True}),
            dmc.TextInput(id="compare-value", label="New value", size="xs", w=140),
            html.Span(CHANGEABLE["gust_lag"], id="compare-form", className="ati-muted"),
            dmc.Button("Fly the change", id="compare-fly", size="xs",
                       leftSection=ui.icon("player-play", 14), disabled=spec is None),
            dmc.Tooltip(dmc.Button("Refine A", id="compare-refine", size="xs",
                                   variant="default", disabled=spec is None),
                        label="The convergence analysis with A as its base: dt, dt/2, dt/4"),
        ], className="ati-controls ati-controls-labelled"),
        html.Div([
            dmc.TextInput(id="compare-ref", label="Git ref (a branch, tag or commit)",
                          size="xs", w=220),
            dmc.Button("Fly A at this commit", id="compare-commit", size="xs",
                       variant="default", disabled=spec is None),
        ], className="ati-controls ati-controls-labelled"),
        html.Div("" if spec is not None or not names else
                 "A was written before runs kept their spec (spec.json), so it cannot be "
                 "flown again from here.", className="ati-muted"),
    ], className="ati-deep-block") if names else None

    diff_rows = differences([ws.loaded(n).run.meta for n in names]) if len(names) > 1 else []
    header = None
    if len(names) > 1:
        head = [html.Th("Setting")] + [html.Th(chr(ord("A") + i)) for i in range(len(names))]
        body = [html.Tr([html.Td(k, className="ati-mono")]
                        + [html.Td([html.Div(line) for line in
                                    shown(v, values[:i] + values[i + 1:])], title=str(v))
                           for i, v in enumerate(values)])
                for k, values in diff_rows[:40]]
        header = html.Div([
            html.Div(f"What differs ({len(diff_rows)} settings)", className="ati-panel-title"),
            html.Table([html.Thead(html.Tr(head)), html.Tbody(body)],
                       className="ati-table ati-diff-table")
            if diff_rows else html.Div("Nothing recorded differs: the runs are the same "
                                       "experiment.", className="ati-empty"),
        ], className="ati-deep-block")

    deep = html.Div([x for x in [
        switch,
        html.Div([*chips, html.Span(hint, className="ati-muted")],
                 className="ati-controls ati-deep-block"),
        header,
        html.Div([
            dmc.Select(id="compare-channel", value="n_z", w=260,
                       data=_channels(ws, names), comboboxProps={"withinPortal": True},
                       **{"aria-label": "Channel"}),
        ], className="ati-controls") if len(names) > 1 else None,
        html.Div(dcc.Graph(id="compare-strips", config=_GRAPH), className="ati-deep-block",
                 style=None if len(names) > 1 else {"display": "none"}),
        actions,
        # Always in the page: the callbacks that write it run on every Compare page.
        html.Div(id="compare-note", className="ati-compare-note"),
        dcc.Store(id="compare-names", data=names),
        dcc.Store(id="compare-job"),
        dcc.Interval(id="compare-poll", interval=1000, disabled=True),
    ] if x is not None], className="ati-deep")
    return html.Div([runs_panel, deep], className="ati-results")


def _series(ws, name, channel):
    from atisim.apps import diagnostics

    if channel.startswith("diag:"):
        d = diagnostics.diag(ws, name)
        return np.asarray(d.columns["t"]), np.asarray(d.columns[channel[5:]])
    s = ws.loaded(name).series
    return np.asarray(s.t), np.asarray(getattr(s, channel))


def _changed(spec: run_mod.RunSpec, field: str, raw: str) -> run_mod.RunSpec:
    if field not in CHANGEABLE:
        raise ValueError(f"{field} cannot be changed here")
    new = run_mod.apply_set(spec, f"{field}={raw}")
    return new._replace(name=f"{spec.name}-{field}-{raw}".replace(".", "p")[:60])


def _warn(text: str) -> list:
    """A prompt: something is missing."""
    return [ui.state("warning", "warn", size=13), " ", text]


def _error(text: str) -> list:
    """A value the engine refuses."""
    return [ui.state("error", "fail", size=13), " ", text]


def register(app, ws) -> None:
    @app.callback(Output("compare-form", "children"), Input("compare-field", "value"),
                  prevent_initial_call=True)
    def form(field):
        return CHANGEABLE.get(field, "")

    @app.callback(
        Output("url", "search", allow_duplicate=True),
        Input("compare-add", "value"), State("compare-names", "data"),
        prevent_initial_call=True,
    )
    def add(name, names):
        if not name or name in (names or []):
            raise PreventUpdate
        return href((names or []) + [name]).split("/results", 1)[1]

    @app.callback(
        Output("compare-strips", "figure"), Output("compare-strips", "style"),
        Input("compare-channel", "value"), State("compare-names", "data"),
    )
    def strips(channel, names):
        if not names or len(names) < 2 or not channel:
            raise PreventUpdate
        label, unit = SERIES_CHANNELS.get(channel, (channel.split(":", 1)[-1], ""))
        fig = devfigures.compare_strips([_series(ws, n, channel) for n in names], label,
                                        [chr(ord("A") + i) for i in range(len(names))],
                                        unit)
        return fig, {"height": f"{fig.layout.height}px"}

    @app.callback(
        Output("compare-job", "data"), Output("compare-poll", "disabled"),
        Output("compare-note", "children"), Output("job", "data", allow_duplicate=True),
        Output("analysis-job", "data", allow_duplicate=True),
        Input("compare-fly", "n_clicks"), Input("compare-refine", "n_clicks"),
        Input("compare-commit", "n_clicks"),
        State("compare-field", "value"), State("compare-value", "value"),
        State("compare-ref", "value"), State("compare-names", "data"),
        prevent_initial_call=True,
    )
    def launch(_fly, _refine, _commit, field, raw, ref, names):
        trigger = ctx.triggered_id
        if not names or not ctx.triggered[0]["value"]:
            raise PreventUpdate
        spec = run_mod.spec_of(ws.root / names[0])
        if spec is None:
            raise PreventUpdate
        if trigger == "compare-refine":
            aspec = registry.AnalysisSpec(name=f"refine-{spec.name}", analysis="convergence",
                                          params={}, base=spec)
            job = ws.jobs.submit_analysis(aspec)
            return ({"id": job, "kind": "analysis"}, False,
                    "Refining A at dt, dt/2 and dt/4. The report opens when it is written.",
                    no_update, job)
        if trigger == "compare-commit":
            if not (ref or "").strip():
                return (no_update, True, _warn("Give a git ref, for example main or a "
                                               "commit SHA."), no_update, no_update)
            job = ws.jobs.submit_commit(spec, ref.strip())
            return ({"id": job, "kind": "commit"}, False,
                    f"Flying A at {ref.strip()} in a worktree. It joins the comparison "
                    "when written.", job, no_update)
        if raw is None or str(raw).strip() == "":
            return (no_update, True, _warn(f"Give a new value for {field} "
                                           f"({CHANGEABLE.get(field)})."), no_update, no_update)
        try:
            new = _changed(spec, field, str(raw).strip())
        except ValueError as exc:
            return no_update, True, _error(str(exc)), no_update, no_update
        problems = run_mod.errors(new)
        if problems:
            return no_update, True, _error(problems[0].message), no_update, no_update
        job = ws.jobs.submit(new)
        return ({"id": job, "kind": "run"}, False,
                f"Flying A with {field} = {raw}. It joins the comparison when written.",
                job, no_update)

    @app.callback(
        Output("url", "search", allow_duplicate=True),
        Output("compare-poll", "disabled", allow_duplicate=True),
        Output("compare-note", "children", allow_duplicate=True),
        Input("compare-poll", "n_intervals"), State("compare-job", "data"),
        State("compare-names", "data"),
        prevent_initial_call=True,
    )
    def follow(_n, pending, names):
        job = ws.jobs.get((pending or {}).get("id"))
        if job is None:
            return no_update, True, no_update
        if job["state"] == "failed":
            return no_update, True, [ui.state("failed", "fail", size=13), " ", job["error"]]
        if job["state"] != "done":
            return no_update, False, no_update
        name = Path(job["path"]).name
        if pending.get("kind") == "analysis":
            return f"?run={name}", True, no_update
        return href((names or []) + [name]).split("/results", 1)[1], True, no_update
