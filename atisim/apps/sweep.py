"""The analysis UI: sweep view and deep dive over a directory of run artifacts.

    .venv/Scripts/python.exe -m atisim.apps.sweep runs/analysis

Two jobs, both first-class. The SWEEP is static on load -- no click is needed to
judge whether a run is physically sensible, which is the whole point of a
bug-detection instrument. The DEEP DIVE adds one piece of shared state, the time
cursor, and everything updates from it in a single callback so nothing tears.

**One click from "that peak looks wrong" to "here is the field geometry".** Click
the peak on any strip: a rule appears on every strip, the 3D scene drops a marker
at that position without moving the camera, and the readout gives the sample's
numbers including its along-track position IN CORE RADII -- which is what turns
"that peak" into "0.98 core radii, just inside the boundary" without the reader
doing arithmetic.

Borrowed, and worth naming. The shared cursor is IADS's Scrollback, where "all
data and displays are updated at the same time to display data from the exact
same moment". The readout beside the render is ParaView's Spreadsheet View. The
clickable check badge is the FDM exceedance workflow: the checks run whether or
not anyone looks, and the analyst is TAKEN to the event rather than asked to
scroll for it.
"""

import argparse
from pathlib import Path

import numpy as np
from dash import ALL, Dash, Input, Output, State, dcc, html

from atisim.aircraft import REGISTRY
from atisim.analysis import artifact, figures, series

# Which 3D treatment is legible for which field. One treatment does not fit four
# fields: an isosurface of a smooth column is a featureless blob, and rendering a
# one-dimensional lee wave as a volume manufactures structure it does not have.
REPRESENTATION = {
    "VortexArray": "isosurface",
    "UpdraftColumn": "streamtube",
    "Microburst": "streamtube",
    "LeeWave": "slice",
}

_CARD = {"background": "#fff", "border": "1px solid #e3e3e3", "borderRadius": "6px",
         "padding": "8px 10px", "marginBottom": "8px"}
_MONO = {"fontFamily": "ui-monospace, Consolas, monospace"}


class Loaded:
    """One artifact, with its derived series and rebuilt field. Read once."""

    def __init__(self, path: Path):
        self.run = artifact.read_run(path)
        self.name = path.name
        meta = self.run.meta
        ac = REGISTRY[meta["aircraft"]["key"]]
        self.series = series.build(self.run.trajectory, ac)
        self.field = artifact.rebuild_field(meta)
        self.kind = meta["wind_field"]["kind"]
        self.representation = REPRESENTATION.get(self.kind, "none")
        params = meta["wind_field"].get("params", {})
        self.core_radius = params.get("r0")  # vortex only; the readout's north/r0
        self.peak_tangential = params.get("v0")
        # A CHARACTERISTIC SCALE AND A PEAK, per field kind, so the 2D
        # cross-section is not a vortex-only panel. Three of the four fields
        # have both; a run with no field at all (the manoeuvre) has neither and
        # falls back to the 3D panel.
        #   scale sets the view extent, peak pins the diverging colour scale.
        # A quarter wavelength is the lee wave's scale because that is the
        # distance from a zero crossing to a trough -- the structure a reader
        # needs to see, where a whole wavelength would show two of everything.
        self.field_scale, self.field_peak = {
            "VortexArray": (params.get("r0"), params.get("v0")),
            "UpdraftColumn": (params.get("radius"), params.get("w0")),
            "LeeWave": ((params.get("wavelength") or 0) / 4 or None,
                        params.get("w0")),
            "Microburst": (params.get("radius"), params.get("u_max")),
        }.get(self.kind, (None, None))
        # Where the STRUCTURE is, from the field spec. Not the trajectory
        # midpoint: a 40 core-radii lead-in puts that ~2.5 km upstream of the
        # cores, in irrotational flow, and the isosurface comes out empty while
        # the panel still looks normal. `figures.field_3d` has the detail.
        centre = params.get("north")
        self.field_centre = (
            float(np.mean(centre)) if isinstance(centre, list) and centre
            else float(centre) if isinstance(centre, (int, float)) else None
        )
        # Core centres as (north, altitude), for the cross-section to draw
        # circles at. `down` is NED so altitude is its negation -- getting that
        # backwards would put the circles 24 km below the aircraft.
        down = params.get("down")
        self.cores = (
            [(float(n), -float(d)) for n, d in zip(centre, down)]
            if isinstance(centre, list) and isinstance(down, list) else []
        )
        window = meta["declared_parameters"].get("window", {})
        north = window.get("north_m")
        if north:
            self.window = ((self.series.north >= north[0])
                           & (self.series.north <= north[1]))
        else:
            # A TIME window -- the manoeuvre's elevator pulse. Both bounds come
            # from the artifact: the pulse start was briefly hard-coded here as
            # 2.0 s, which happened to be right for the only run that existed and
            # is exactly the kind of number that goes wrong silently the first
            # time somebody passes --pushdown-seconds with a different lead-in.
            seconds, start = window.get("seconds"), window.get("starts_at_s")
            t = self.series.t
            if seconds is None or start is None:
                self.window = np.ones(len(t), dtype=bool)
            else:
                self.window = (t > start) & (t <= start + seconds)

    def fig8(self) -> dict:
        """Windowed and whole-run coordinates. The connector between them IS the
        panel's content -- it draws the windowing trap rather than describing it."""
        s, w = self.series, self.window
        theta, n_z = s.theta_deg, s.n_z
        return dict(
            label=self.name.split("-")[0],
            dtheta=float(theta[w].max() - theta[w].min()),
            dn=float(n_z[w].min() - n_z[0]),
            dtheta_whole=float(theta.max() - theta.min()),
            dn_whole=float(n_z.min() - n_z[0]),
        )


def _badge(check: dict):
    """Check name, measured value, and a word that is not a bare tick.

    A `tripwire` that has never fired must NOT render as a green tick: that would
    claim evidence it does not provide. It renders its number and the word
    'tripwire', which is honest about what it is.
    """
    passed, kind = check["passed"], check["kind"]
    colour, word = {
        (True, "gate"): ("#1a7f37", "PASS"),
        (False, "gate"): ("#b42318", "FAIL"),
        (True, "tripwire"): ("#5b6b7a", "quiet"),
        (False, "tripwire"): ("#b42318", "FIRED"),
    }.get((passed, kind), ("#8a6d00", "report"))
    return html.Div(
        [
            html.Span(word, style={"color": colour, "fontWeight": 700,
                                   "marginRight": "6px", **_MONO}),
            html.Span(check["name"], style={"marginRight": "6px"}),
            html.Span(f"{check['value']:.4g}", style={"color": "#444", **_MONO}),
        ],
        title=check["detail"] + "  ·  click to jump to the worst sample",
        id={"type": "badge", "index": check["name"]},
        n_clicks=0,
        style={"display": "inline-block", "border": f"1px solid {colour}33",
               "background": f"{colour}0d", "borderRadius": "4px",
               "padding": "3px 8px", "margin": "0 6px 6px 0", "fontSize": "12px",
               "cursor": "pointer"},
    )


def _header(loaded: Loaded):
    m = loaded.run.meta
    fc, tr = m["flight_condition"], m["trim"]
    line = (
        f"{m['aircraft']['key']}  ·  {loaded.kind}  ·  "
        f"{fc['altitude_m']:.0f} m  ·  {fc['airspeed_mps']:.2f} m/s  ·  "
        f"dt {m['integrator']['dt_s']} s  ·  {m['integrator']['n_steps']} steps  ·  "
        f"{m['git_sha'][:7] or 'nogit'}"
    )
    second = (
        f"trim alpha {np.degrees(tr['alpha_rad']):.3f}°  ·  "
        f"load path {m['load_model'] or 'point sample plus analytic gradient'}  ·  "
        + "  ·  ".join(f"{k} {v}" for k, v in m["declared_parameters"].items()
                       if isinstance(v, (int, float, str)))
    )
    children = [
        html.Div(line, style={"fontSize": "12px", **_MONO}),
        html.Div(second, style={"fontSize": "11px", "color": "#555", **_MONO}),
    ]
    for caveat in m.get("caveats", []):
        children.append(html.Div("⚠ " + caveat, style={
            "fontSize": "11px", "color": "#8a6d00", "marginTop": "2px"}))
    if loaded.run.stale:
        children.append(html.Div(
            f"⚠ flown at {m['git_sha'][:7]}, which is not the commit you are "
            "reading it with — figures may not match today's code.",
            style={"fontSize": "11px", "color": "#b42318", "marginTop": "2px"}))
    return html.Div(children, style=_CARD)


def build_app(root: Path) -> Dash:
    paths = artifact.list_runs(root)
    if not paths:
        raise SystemExit(
            f"no run artifacts under {root}. Fly one first:\n"
            f"  .venv/Scripts/python.exe scripts/vortex.py --artifacts {root}"
        )
    runs = {p.name: Loaded(p) for p in paths}
    fig8_points = [r.fig8() for r in runs.values()]

    app = Dash(__name__, title="atisim analysis")
    app.layout = html.Div(
        [
            html.Div([
                html.Span("atisim  ", style={"fontWeight": 700}),
                html.Span("turbulence-encounter analysis",
                          style={"color": "#666", "marginRight": "16px"}),
                dcc.Dropdown(
                    id="run", value=paths[0].name, clearable=False,
                    options=[{"label": n, "value": n} for n in runs],
                    style={"width": "340px", "display": "inline-block",
                           "verticalAlign": "middle"},
                ),
                dcc.Dropdown(
                    id="scalar", value="n_z", clearable=False,
                    options=[{"label": f"colour 3D by {k}", "value": k}
                             for k in figures.SCALARS],
                    style={"width": "220px", "display": "inline-block",
                           "verticalAlign": "middle", "marginLeft": "8px"},
                ),
                # 2D IS THE DEFAULT, and for the Parks vortex it is not a
                # simplification: `wind.vortex_wind` fixes dpsi = 0, so the field
                # has NO east variation and one north-altitude plane contains all
                # of it. The 3D scene is the option, not the baseline.
                dcc.RadioItems(
                    id="fieldview", value="2d",
                    options=[{"label": " 2D cross-section", "value": "2d"},
                             {"label": " 3D scene", "value": "3d"}],
                    inline=True,
                    style={"display": "inline-block", "marginLeft": "12px",
                           "fontSize": "12px", "verticalAlign": "middle"},
                    inputStyle={"marginRight": "4px", "marginLeft": "8px"},
                ),
            ], style={"marginBottom": "8px"}),
            html.Div(id="header"),
            html.Div(id="badges", style=_CARD),
            html.Div([
                html.Div([
                    dcc.Graph(id="strips", config={"displaylogo": False, "responsive": True}),
                ], style={"width": "56%", "display": "inline-block",
                          "verticalAlign": "top", **_CARD}),
                html.Div([
                    dcc.Graph(id="scene", config={"displaylogo": False, "responsive": True}),
                    html.Div(id="readout", style={"fontSize": "11.5px", **_MONO}),
                ], style={"width": "41%", "display": "inline-block",
                          "verticalAlign": "top", "marginLeft": "1.5%", **_CARD}),
            ]),
            # The verdict FIRST, then the domain-standard figure it comes from.
            # Fig. 8's claim is an ordering and Fig. 8 does not state it; this
            # panel does, and it is the one a reader should hit first.
            html.Div([dcc.Graph(id="ordering",
                                config={"displaylogo": False, "responsive": True})],
                     style=_CARD),
            html.Div([
                html.Div([dcc.Graph(id="fig8", config={"displaylogo": False, "responsive": True})],
                         style={"width": "48%", "display": "inline-block", **_CARD}),
                html.Div([dcc.Graph(id="nzalpha", config={"displaylogo": False, "responsive": True})],
                         style={"width": "48%", "display": "inline-block",
                                "marginLeft": "1.5%", **_CARD}),
            ]),
            dcc.Store(id="cursor", data=None),
        ],
        style={"fontFamily": "system-ui, -apple-system, sans-serif",
               "background": "#f6f7f9", "padding": "12px", "minHeight": "100vh"},
    )

    @app.callback(
        Output("cursor", "data"),
        Input("strips", "clickData"), Input("run", "value"),
        Input({"type": "badge", "index": ALL}, "n_clicks"),
        State("cursor", "data"),
    )
    def set_cursor(click, run_name, badge_clicks, current):
        """The ONE piece of shared state, and the two ways to move it.

        Clicking a strip is the obvious one. Clicking a CHECK BADGE is the other,
        and it is the borrowed half: FDM exceedance analysis computes the events
        and takes the analyst TO them rather than asking them to scroll and look.
        Every check carries the sample where it is worst, so the badge knows
        where to send the cursor.

        Changing run clears it -- a time from another run points at a different
        moment entirely, and silently keeping it would put the cursor somewhere
        meaningless while looking deliberate.
        """
        import dash

        trigger = dash.ctx.triggered_id
        if trigger == "run":
            return None
        if isinstance(trigger, dict) and trigger.get("type") == "badge":
            if not any(badge_clicks or []):
                return current  # fired on layout build, not on a real click
            loaded = runs[run_name]
            for check in loaded.run.checks:
                if check["name"] == trigger["index"]:
                    index = check.get("worst_index")
                    if index is None:
                        return current
                    return float(loaded.series.t[int(index)])
            return current
        if not click:
            return current
        return float(click["points"][0]["x"])

    @app.callback(
        Output("header", "children"), Output("badges", "children"),
        Output("strips", "figure"), Output("scene", "figure"),
        Output("ordering", "figure"),
        Output("fig8", "figure"), Output("nzalpha", "figure"),
        Output("readout", "children"),
        Input("run", "value"), Input("scalar", "value"), Input("cursor", "data"),
        Input("fieldview", "value"),
        State("scene", "relayoutData"),
    )
    def render(run_name, scalar, cursor_t, fieldview, scene_relayout):
        """Everything updates from the cursor in ONE callback, so nothing tears.

        Six outputs, one input set. Splitting this into six callbacks would let
        the strips show one moment while the 3D marker showed another, which is
        exactly the failure the shared time base exists to prevent.

        `scene_relayout` is a `State`, not an `Input`: rotating the scene must not
        redraw anything. It exists so the camera can be put back EXPLICITLY --
        `uirevision` is set as well, but this project does not rely on a mechanism
        it cannot assert on. See `figures.apply_camera`.
        """
        loaded = runs[run_name]
        s = loaded.series
        index = None if cursor_t is None else int(np.argmin(np.abs(s.t - cursor_t)))

        # The cross-section needs a core radius and a peak to pin its scale to.
        # A field that supplies neither (the manoeuvre run has no field at all)
        # falls back to the 3D panel rather than drawing an unpinned map.
        can_slice = bool(loaded.field_scale and loaded.field_peak)
        if fieldview == "2d" and can_slice:
            scene = figures.field_cross_section(
                s, loaded.field,
                scale=loaded.field_scale, peak=loaded.field_peak,
                cores=loaded.cores, window=loaded.window,
                cursor_index=index, field_centre=loaded.field_centre,
                label=loaded.kind,
            )
        else:
            scene = figures.apply_camera(
                figures.field_3d(
                    s, loaded.field, scalar=scalar, cursor_index=index,
                    representation=loaded.representation,
                    scale=loaded.field_scale, peak=loaded.field_peak,
                    field_centre=loaded.field_centre,
                ),
                scene_relayout,
            )
        return (
            _header(loaded),
            [_badge(c) for c in loaded.run.checks],
            figures.strip_stack(s, loaded.window, cursor_t=cursor_t),
            scene,
            figures.ordering(fig8_points),
            figures.discriminator(fig8_points),
            figures.load_vs_alpha(s, cursor_index=index),
            _readout(loaded, index),
        )

    return app


def _readout(loaded: Loaded, index: int | None):
    """ParaView's Spreadsheet View: the numbers beside the render.

    `north / r0` is the row that earns this panel -- it turns "that peak" into
    "0.98 core radii, just inside the boundary" with no arithmetic by the reader.
    """
    if index is None:
        return html.Div("click a strip to place the time cursor",
                        style={"color": "#888", "padding": "6px 2px"})
    s = loaded.series
    rows = [
        ("t", f"{s.t[index]:.3f} s"),
        ("north", f"{s.north[index]:+.1f} m"),
    ]
    if loaded.core_radius:
        rows.append(("north / r0", f"{s.north[index] / loaded.core_radius:+.3f}"))
    rows += [
        ("altitude", f"{s.altitude[index]:.1f} m"),
        ("gust up", f"{s.w_up[index]:+.3f} m/s"),
        ("q gust", f"{s.q_gust_deg[index]:+.4f} deg/s  (SIM TRUTH)"),
        ("alpha air-rel", f"{s.alpha_deg[index]:+.3f} deg"),
        ("alpha inertial", f"{s.alpha_inertial_deg[index]:+.3f} deg"),
        ("n_z", f"{s.n_z[index]:+.4f} g"),
        ("theta", f"{s.theta_deg[index]:+.3f} deg"),
        ("elevator", f"{s.elevator_deg[index]:+.3f} deg"),
    ]
    return html.Table(
        [html.Tr([html.Td(k, style={"color": "#666", "paddingRight": "10px"}),
                  html.Td(v)]) for k, v in rows],
        style={"width": "100%", "borderCollapse": "collapse"},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="?", default=Path("runs/analysis"),
                        help="directory of run artifacts")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    build_app(args.runs).run(host="127.0.0.1", port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
