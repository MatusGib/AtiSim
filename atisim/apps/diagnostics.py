"""Results, Diagnostics workspace: one run, opened up for engine work (plan 3.3).

A High fidelity run (`diagnostics.parquet`) gets the channel browser, the
coefficient, force and energy budgets, the step inspector and the high-detail
scene. Every run gets the check profiles, which read the trajectory only. A
Standard run says what High would add and offers to fly it again at High.

The page is `/results?run=NAME&view=diagnostics`. Its components have ids of
their own (`diag-...`), so the Overview's callbacks never see this page and
this page's never see the Overview.
"""

import threading
from pathlib import Path

import dash_mantine_components as dmc
import numpy as np
from dash import ALL, ClientsideFunction, Input, Output, State, ctx, dcc, html, no_update
from dash.exceptions import PreventUpdate

from atisim import run as run_mod
from atisim.aircraft import REGISTRY
from atisim.analysis import devfigures, diagnostics, profiles, step_inspector
from atisim.apps import components as ui
from atisim.apps.jobs import LIVE

_GRAPH = {"displaylogo": False, "responsive": True}
DEFAULT_CHANNELS = ("n_z", "alpha", "Cm", "alphadot_gust")
COEFFICIENTS = ("CL", "CD", "CY", "Cl", "Cm", "Cn")
SPEEDS = {"0.25x": 0.25, "1x": 1.0, "4x": 4.0, "16x": 16.0}
TICK_MS = 200
# Playback (`assets/playback.js`) sends the cursor to the server at most this
# often: the 3D scene evaluates the wind field, and a queue of those makes every
# panel late. Between two, the other figures' cursor lines move in the browser.
COMMIT_S = 1.0
# The figures whose cursor moves in the browser: a new cursor time moves their
# named cursor lines (`devfigures._cursor`) and draws nothing else again.
CURSOR_GRAPHS = ("diag-strips", "diag-budget", "diag-forces", "diag-energy", "diag-profiles")
PLAYBACK = {"speeds": SPEEDS, "tick": TICK_MS / 1000.0, "commit": COMMIT_S,
            "graphs": list(CURSOR_GRAPHS)}
_scene_lock = threading.Lock()

_cache: dict[str, tuple[float, object]] = {}


def _cached(key: str, stamp_path: Path, build):
    stamp = stamp_path.stat().st_mtime if stamp_path.exists() else 0.0
    hit = _cache.get(key)
    if hit is None or hit[0] != stamp:
        _cache[key] = (stamp, build())
    return _cache[key][1]


def diag(ws, name: str):
    """The run's diagnostics, or None for a Standard run."""
    path = ws.root / name
    return _cached(f"diag:{name}", path / diagnostics.FILENAME,
                   lambda: diagnostics.read(path))


def check_profiles(ws, name: str) -> dict:
    loaded = ws.loaded(name)
    ac = REGISTRY[loaded.run.meta["aircraft"]["key"]]
    return _cached(f"profiles:{name}", ws.root / name / "meta.json",
                   lambda: profiles.build(loaded.run.trajectory, ac, loaded.field,
                                          loaded.run.checks))


def lag_states(d) -> np.ndarray | None:
    if d is None or "gust_lag_1" not in d.columns:
        return None
    return np.stack([d.columns["gust_lag_1"], d.columns["gust_lag_2"]], axis=1)


def _section(title: str, *children, hidden: bool = False):
    return html.Div([html.Div(title, className="ati-panel-title"), *children],
                    className="ati-deep-block", style={"display": "none"} if hidden else None)


def channel_picker(d, chosen) -> list:
    groups: dict[str, list] = {}
    for ch in d.channels if d is not None else []:
        groups.setdefault(ch.group, []).append(ch)
    items = []
    for group, chans in groups.items():
        items.append(dmc.AccordionItem([
            dmc.AccordionControl(f"{group} ({len(chans)})"),
            dmc.AccordionPanel(dmc.CheckboxGroup(
                id={"type": "diag-group", "group": group},
                value=[c.name for c in chans if c.name in chosen],
                children=dmc.Stack([dmc.Checkbox(
                    label=html.Span(c.name, title=f"{c.description} [{c.unit}]"),
                    value=c.name, size="xs") for c in chans], gap=4),
            )),
        ], value=group))
    return [dmc.Accordion(items, multiple=True, value=["load", "air data"],
                          chevronPosition="left", variant="default")]


def step_view(detail) -> list:
    def table(rows, headers, compact=False):
        return html.Table([html.Thead(html.Tr([html.Th(h, className="num" if j else None)
                                               for j, h in enumerate(headers)])),
                           html.Tbody([html.Tr([html.Td(c, className="num" if j else None)
                                                for j, c in enumerate(r)]) for r in rows])],
                          className="ati-table ati-table-compact" if compact else "ati-table")

    state = table([(k, f"{v:.6g}") for k, v in detail.state.items()], ("State", "Value"),
                  compact=True)
    controls = table([(k, f"{v:.6g}") for k, v in detail.controls.items()],
                     ("Control", "Value"), compact=True)
    stages = table([(f"{s.fraction:g} dt", f"{s.pos_ned[0]:.3f}",
                     f"{-s.pos_ned[2]:.3f}", f"{-s.wind_field[2]:+.5f}",
                     f"{-s.wind_used[2]:+.5f}", f"{np.degrees(s.alphadot_gust):+.5f}",
                     f"{s.vel_dot[2]:+.5f}", f"{np.degrees(s.omega_dot[1]):+.5f}")
                    for s in detail.stages],
                   ("Stage", "north m", "alt m", "field w_up", "used w_up",
                    "alphadot_gust deg/s", "wdot m/s²", "qdot deg/s²"))
    errors = table([(k, f"{detail.restep_error[k]:.3e}",
                     f"{detail.refine_difference[k]:.3e}") for k in detail.restep_error],
                   ("State group", "re-step minus log",
                    f"one step minus {detail.refine} of dt/{detail.refine}"))
    return [
        html.P(detail.note, className="ati-note"),
        html.Div([state, controls], className="ati-inspect-tables"),
        html.P("The four RK4 stages of this step. On a held run the used wind is the "
               "step's first sample at every stage; the field column shows what it "
               "would have been.", className="ati-muted"),
        stages,
        html.P("Re-step: the step flown again from the logged state with the run's own "
               "options; zero means the rebuild is the run. The refined column "
               "estimates the step's local truncation error.", className="ati-muted"),
        errors,
    ]


def _header(ws, name, d):
    meta = ws.loaded(name).run.meta
    facts = [ui.state("High fidelity", "ok") if d is not None
             else ui.state("Standard fidelity", "neutral"),
             html.Span(name, className="ati-mono"),
             html.Span(f"{meta['aircraft']['key']}, dt {meta['integrator']['dt_s']:g} s, "
                       f"{meta['integrator']['n_steps']} steps", className="ati-muted")]
    if d is not None and d.timing:
        facts.append(html.Span("timing: " + ", ".join(
            f"{k} {v:.2f} s" for k, v in d.timing.items()), className="ati-muted"))
    facts.append(html.Span("The cursor starts at the largest load excursion; click any "
                           "strip or check to move it.", className="ati-muted"))
    return html.Div(facts, className="ati-controls")


def layout(ws, name: str | None, t=None, runs_panel=None, switch=None):
    """The Diagnostics workspace for run `name` (already resolved by Results)."""
    d = diag(ws, name)
    high = d is not None
    loaded = ws.loaded(name)
    if t is not None:
        cursor = loaded.time_at(t)
    else:
        # Start where the run is most eventful: the largest load excursion.
        n_z = d.columns["n_z"] if high else loaded.series.n_z
        cursor = float(loaded.series.t[int(np.argmax(np.abs(n_z - n_z[0])))])
    spec = run_mod.spec_of(ws.root / name)
    standard_note = None
    if not high:
        standard_note = dmc.Alert([
            html.Span("This run was flown at Standard fidelity. High adds the channel "
                      "browser, the coefficient, force and energy budgets, the step "
                      "inspector and the high-detail scene. The check profiles below "
                      "need only the trajectory."),
            html.Div(dmc.Button("Re-fly at High", id="diag-refly", size="xs", mt=8,
                                disabled=spec is None,
                                leftSection=ui.icon("player-play", 14))),
            html.Div("" if spec is not None else
                     "This flight was flown by hand on the test card, so no spec flies "
                     "it again." if loaded.run.meta.get("flown_by_hand") else
                     "This run was written before runs kept their spec (spec.json), so "
                     "the app cannot fly it again. Fly it from Setup at High.",
                     className="ati-muted"),
            html.Div(id="diag-refly-note", className="ati-muted"),
            dcc.Store(id="diag-refly-job"),
        ], color="gray", variant="light")
    colour_options = ([{"label": f"Colour by {c.name}", "value": c.name}
                       for c in d.channels if c.name != "t"] if high else [])
    body = [
        switch,
        # The run and its checks in one block, inset like every other.
        html.Div([_header(ws, name, d),
                  html.Div([_badge(r) for r in loaded.run.checks], className="ati-badges")],
                 className="ati-deep-block"),
        standard_note,
        _section("Channels", html.Div([
            html.Div(channel_picker(d, DEFAULT_CHANNELS), className="ati-diag-picker"),
            html.Div(dcc.Graph(id="diag-strips", config=_GRAPH), className="ati-deep-cell"),
        ], className="ati-diag-channels"), hidden=not high),
        _section("Budgets", html.Div([
            dmc.SegmentedControl(id="diag-coeff", value="Cm", size="xs",
                                 data=list(COEFFICIENTS)),
            dmc.SegmentedControl(id="diag-relative", value="relative", size="xs",
                                 data=[{"label": "change since trim", "value": "relative"},
                                       {"label": "absolute", "value": "absolute"}]),
            html.Span(id="diag-dominant", className="ati-note"),
        ], className="ati-controls"),
            dcc.Graph(id="diag-budget", config=_GRAPH),
            html.Div([html.Div(dcc.Graph(id="diag-forces", config=_GRAPH),
                               className="ati-deep-cell"),
                      html.Div(className="ati-vrule"),
                      html.Div(dcc.Graph(id="diag-energy", config=_GRAPH),
                               className="ati-deep-cell")], className="ati-deep-row"),
            hidden=not high),
        _section("Step inspector", html.Div([
            dmc.Button("Inspect the step at the cursor", id="diag-inspect", size="xs",
                       variant="default", leftSection=ui.icon("zoom-in", 14)),
            html.Span("Flies the step again from the logged state, and again as ten "
                      "steps of dt/10.", className="ati-muted"),
        ], className="ati-controls"), dcc.Loading(html.Div(id="diag-step"), type="dot"),
            hidden=not high),
        _section("Scene, high detail", html.Div([
            dmc.Select(id="diag-colour", value="n_z", w=220, data=colour_options,
                       comboboxProps={"withinPortal": True}, **{"aria-label": "Colour"}),
            dmc.Button("Play", id="diag-play", size="xs", variant="default",
                       leftSection=ui.icon("player-play", 14)),
            dmc.SegmentedControl(id="diag-speed", value="1x", size="xs",
                                 data=list(SPEEDS)),
            html.Span("Playback moves the cursor; every panel follows it.",
                      className="ati-muted"),
        ], className="ati-controls"), dcc.Graph(id="diag-scene", config=_GRAPH),
            hidden=not high),
        _section("Check profiles", dcc.Graph(id="diag-profiles", config=_GRAPH)),
        dcc.Interval(id="diag-tick", interval=TICK_MS, disabled=True),
        dcc.Store(id="diag-cursor", data=cursor),
        dcc.Store(id="diag-clock"),
        dcc.Store(id="diag-played"),
        dcc.Store(id="diag-playback", data=PLAYBACK),
        dcc.Store(id="diag-bounds", data=[float(loaded.series.t[0]), float(loaded.series.t[-1])]),
        dcc.Store(id="diag-cursor-drawn"),
        dcc.Store(id="diag-shown", data=name),
    ]
    deep = html.Div([b for b in body if b is not None], className="ati-deep")
    return html.Div([runs_panel, deep], className="ati-results")


def _badge(check: dict):
    """As the Overview's badge, under this page's own id."""
    word, tone, icon_name = ui.check_state(check)
    return html.Button(
        [ui.state(word, tone, icon_name), html.Span(check["name"]),
         html.Span(f"{check['value']:.4g}", className="ati-num ati-muted")],
        title=check["detail"] + "  Click to move the cursor to the worst sample.",
        id={"type": "diag-badge", "index": check["name"]}, n_clicks=0,
        className="ati-badge", type="button")


def _chosen(groups) -> list[str]:
    out = []
    for values in groups or []:
        out.extend(values or [])
    return out


def register(app, ws) -> None:
    @app.callback(
        Output("url", "search", allow_duplicate=True),
        Input("diag-run", "value"), State("diag-shown", "data"),
        prevent_initial_call=True,
    )
    def open_run(name, shown):
        if not name or name == shown:
            return no_update
        return f"?run={name}&view=diagnostics"

    @app.callback(
        Output("diag-cursor", "data"),
        Input("diag-strips", "clickData"), Input("diag-budget", "clickData"),
        Input("diag-forces", "clickData"), Input("diag-energy", "clickData"),
        Input("diag-profiles", "clickData"),
        Input({"type": "diag-badge", "index": ALL}, "n_clicks"),
        State("diag-shown", "data"),
        prevent_initial_call=True,
    )
    def move_cursor(*args):
        *_, name = args
        trigger = ctx.triggered_id
        if not name:
            raise PreventUpdate
        t = ws.loaded(name).series.t
        if isinstance(trigger, dict):
            if not ctx.triggered[0]["value"]:
                raise PreventUpdate
            for check in ws.loaded(name).run.checks:
                if check["name"] == trigger["index"] and check.get("worst_index") is not None:
                    return float(t[min(int(check["worst_index"]), len(t) - 1)])
            raise PreventUpdate
        click = ctx.triggered[0]["value"]
        if not click:
            raise PreventUpdate
        return float(click["points"][0]["x"])

    app.clientside_callback(
        ClientsideFunction(namespace="atisim", function_name="tick"),
        Output("diag-played", "data"), Output("diag-cursor", "data", allow_duplicate=True),
        Output("diag-clock", "data", allow_duplicate=True),
        Input("diag-tick", "n_intervals"),
        State("diag-played", "data"), State("diag-cursor", "data"),
        State("diag-speed", "value"), State("diag-clock", "data"),
        State("diag-bounds", "data"), State("diag-playback", "data"),
        prevent_initial_call=True,
    )

    app.clientside_callback(ClientsideFunction(namespace="atisim", function_name="moveCursor"),
                            Output("diag-cursor-drawn", "data"),
                            Input("diag-cursor", "data"), State("diag-playback", "data"),
                            State("diag-tick", "disabled"))

    @app.callback(
        Output("diag-tick", "disabled"), Output("diag-play", "children"),
        Output("diag-play", "leftSection"), Output("diag-clock", "data"),
        Output("diag-cursor", "data", allow_duplicate=True),
        Input("diag-play", "n_clicks"), State("diag-tick", "disabled"),
        State("diag-played", "data"),
        prevent_initial_call=True,
    )
    def play(_clicks, disabled, played):
        # The clock starts at the first tick: it is the browser's, not this one's.
        if disabled:
            return False, "Pause", ui.icon("player-pause", 14), None, no_update
        # Paused: every panel takes the time that playback reached.
        return (True, "Play", ui.icon("player-play", 14), None,
                no_update if played is None else played)

    @app.callback(
        Output("diag-strips", "figure"), Output("diag-strips", "style"),
        Input({"type": "diag-group", "group": ALL}, "value"),
        Input("diag-strips", "relayoutData"), State("diag-cursor", "data"),
        State("diag-shown", "data"),
    )
    def strips(groups, relayout, cursor_t, name):
        d = diag(ws, name) if name else None
        if d is None:
            raise PreventUpdate
        if ctx.triggered_id == "diag-strips" and not (
                relayout and ("xaxis.range[0]" in relayout or "xaxis.autorange" in relayout)):
            raise PreventUpdate  # the cursor moved, or the view changed only in y
        xrange = None
        if relayout and "xaxis.range[0]" in relayout:
            xrange = (relayout["xaxis.range[0]"], relayout["xaxis.range[1]"])
        elif relayout and relayout.get("xaxis.autorange"):
            xrange = None
        fig = devfigures.channel_strips(d.columns, d.channels, _chosen(groups), cursor_t,
                                        xrange)
        return fig, {"height": f"{fig.layout.height}px"}

    @app.callback(
        Output("diag-budget", "figure"), Output("diag-forces", "figure"),
        Output("diag-energy", "figure"),
        Input("diag-coeff", "value"), Input("diag-relative", "value"),
        State("diag-cursor", "data"), State("diag-shown", "data"),
    )
    def budgets(coeff, relative, cursor_t, name):
        d = diag(ws, name) if name else None
        if d is None:
            raise PreventUpdate
        rel = relative != "absolute"
        return (devfigures.budget(d.columns, coeff, cursor_t, rel),
                devfigures.force_budget(d.columns, cursor_t),
                devfigures.energy_budget(d.columns, cursor_t))

    @app.callback(
        Output("diag-dominant", "children"),
        Input("diag-cursor", "data"), Input("diag-coeff", "value"),
        Input("diag-relative", "value"), State("diag-shown", "data"),
    )
    def dominant_term(cursor_t, coeff, relative, name):
        d = diag(ws, name) if name else None
        if d is None:
            raise PreventUpdate
        rel = relative != "absolute"
        t = np.asarray(d.columns["t"])
        note = ""
        if cursor_t is not None:
            i = int(np.argmin(np.abs(t - cursor_t)))
            top = devfigures.dominant(d.columns, coeff, i, rel)
            if top:
                term, value, share = top
                note = (f"At t = {t[i]:.2f} s the {term} term moves {coeff} most: "
                        f"{value:+.4g} ({share * 100:.0f}% of the terms' total movement).")
        else:
            note = "Put the cursor on a moment (click a strip or a check) to name the term."
        return note

    @app.callback(
        Output("diag-profiles", "figure"), Output("diag-profiles", "style"),
        Input("diag-shown", "data"), State("diag-cursor", "data"),
    )
    def profile_figure(name, cursor_t):
        if not name:
            raise PreventUpdate
        fig = devfigures.check_profiles(check_profiles(ws, name),
                                        ws.loaded(name).run.checks, cursor_t)
        return fig, {"height": f"{fig.layout.height}px"}

    @app.callback(
        Output("diag-scene", "figure"),
        Input("diag-cursor", "data"), Input("diag-colour", "value"),
        Input("diag-tick", "disabled"),
        State("diag-scene", "relayoutData"), State("diag-shown", "data"),
    )
    def scene(cursor_t, colour, paused, relayout, name):
        from atisim.analysis import figures

        d = diag(ws, name) if name else None
        if d is None:
            raise PreventUpdate
        # While playing, a scene still being drawn makes the next one wait
        # for the next time playback sends: no queue. Paused, it is drawn.
        if not _scene_lock.acquire(blocking=paused is not False):
            raise PreventUpdate
        try:
            loaded = ws.loaded(name)
            t = loaded.series.t
            index = None if cursor_t is None else int(np.argmin(np.abs(t - cursor_t)))
            fig = devfigures.scene_high(loaded.run.trajectory, d.columns, colour or "n_z",
                                        loaded.field, cursor_index=index, cores=loaded.cores,
                                        core_radius=loaded.core_radius)
        finally:
            _scene_lock.release()
        return figures.apply_camera(fig, relayout)

    @app.callback(
        Output("diag-step", "children"),
        Input("diag-inspect", "n_clicks"),
        State("diag-cursor", "data"), State("diag-shown", "data"),
        prevent_initial_call=True,
    )
    def inspect(_clicks, cursor_t, name):
        loaded = ws.loaded(name)
        t = loaded.series.t
        index = 0 if cursor_t is None else int(np.argmin(np.abs(t - cursor_t)))
        try:
            detail = step_inspector.detail(loaded.run.trajectory, loaded.run.meta, index,
                                           lag=lag_states(diag(ws, name)))
        except (ValueError, KeyError) as exc:
            return dmc.Alert(str(exc), color="red", title="Cannot inspect this step")
        return step_view(detail)

    @app.callback(
        Output("job", "data", allow_duplicate=True),
        Output("diag-refly-note", "children"),
        Output("diag-refly-job", "data"),
        Input("diag-refly", "n_clicks"), State("diag-shown", "data"),
        prevent_initial_call=True,
    )
    def refly(clicks, name):
        if not clicks or not name:
            raise PreventUpdate
        spec = run_mod.spec_of(ws.root / name)
        if spec is None:
            raise PreventUpdate
        job = ws.jobs.submit(spec._replace(fidelity="high"))
        return job, ("Flying at High fidelity. The status bar shows its progress, and "
                     "a link to the new run shows here when it is written."), job

    @app.callback(
        Output("diag-refly-note", "children", allow_duplicate=True),
        Input("poll", "n_intervals"), State("diag-refly-job", "data"),
        prevent_initial_call=True,
    )
    def refly_done(_n, job_id):
        """Say where the High run went. The note once kept saying "Flying" after
        the run was written, and the reader had to find it in the explorer."""
        job = ws.jobs.get(job_id)
        if job is None or job["state"] in LIVE:
            raise PreventUpdate
        if job["state"] == "failed":
            return f"The High run failed: {job['error']}"
        new = Path(job["path"]).name
        return ["The High run is written: ",
                dcc.Link(new, href=f"/results?run={new}&view=diagnostics")]

