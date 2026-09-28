"""Results: the analysis deep dive over one run, under a result bar.

This is `sweep.py`'s app, moved into the shell with its behaviour unchanged.
Every component id is the one the analysis UI already had (`run`, `cursor`,
`badges`, `strips`, `scene`, `fieldview`, `scalar`, `ordering`, `fig8`,
`nzalpha`, `nzx`, `readout`). The results are listed in the explorer; the bar
over the page names the result, carries a searchable picker (`run`, which opens
the result it names), and switches between Overview, Diagnostics and Compare.

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
scroll for it. `/results?run=<dir>&t=<index>` is the same move from outside the
page: Setup's Open in Results lands on the worst sample of the first gate that
failed.
"""

import re
from pathlib import Path

import dash_mantine_components as dmc
import numpy as np
from dash import ALL, Input, Output, State, dcc, html, no_update
from dash.exceptions import PreventUpdate

from atisim import records
from atisim.aircraft import REGISTRY
from atisim.analysis import artifact, figures, report, series
from atisim.analysis import runs as runs_mod
from atisim.apps import components as ui
from atisim.run import KIND_LABELS

# Which 3D treatment is legible for which field. One treatment does not fit four
# fields: an isosurface of a smooth column is a featureless blob, and rendering a
# one-dimensional lee wave as a volume manufactures structure it does not have.
REPRESENTATION = {
    "VortexArray": "isosurface",
    "UpdraftColumn": "streamtube",
    "Microburst": "streamtube",
    "LeeWave": "slice",
    # One-dimensional along-track fields: a slice, for the lee wave's reason.
    "Sinusoid": "slice",
    "OneMinusCosine": "slice",
    "Dryden": "slice",
    "VonKarman": "slice",
    "GaussianDryden": "slice",
    "ModulatedDryden": "slice",
}

_GRAPH = {"displaylogo": False, "responsive": True}


def run_spec_kind(path: Path) -> str | None:
    """The wind kind of the spec a run was flown from, or None without one."""
    from atisim import run as run_mod

    spec = run_mod.spec_of(path)
    return spec.wind.kind if spec is not None else None



class Loaded:
    """One artifact, with its derived series and rebuilt field. Read once."""

    def __init__(self, path: Path):
        self.run = artifact.read_run(path)
        self.path = Path(path)
        self.name = path.name
        meta = self.run.meta
        ac = REGISTRY[meta["aircraft"]["key"]]
        self.series = series.build(self.run.trajectory, ac)
        self.field = artifact.rebuild_field(meta)
        self.kind = meta["wind_field"]["kind"]
        # What the source measured in this encounter, on this run's clock, or
        # None. The spec's kind tells the manoeuvre from still air.
        self.record = records.for_run(
            run_spec_kind(path) or self.kind, meta["wind_field"].get("case"),
            self.run.trajectory.t, records.w_up_from_ned(self.run.trajectory.wind_ned))
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
            "Sinusoid": ((params.get("wavelength") or 0) / 4 or None,
                         params.get("amplitude")),
            "OneMinusCosine": (params.get("gradient_distance"), params.get("peak")),
            **{k: (params.get("L_w"), 3.0 * params["sigma_w"] if params.get("sigma_w")
                   else None)
               for k in ("Dryden", "VonKarman", "GaussianDryden", "ModulatedDryden")},
        }.get(self.kind, (None, None))
        self.can_slice = bool(self.field_scale and self.field_peak)
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
        panel's content -- it draws the windowing trap rather than describing it.

        The label is the case, the directory name without `-{aircraft}-{sha}`:
        the first word alone named all three Wingrove & Bach cases "wingrove".
        The category comes from the spec's wind kind, because the manoeuvre and
        a still-air run both have no wind field in `meta.json`."""
        s, w = self.series, self.window
        theta, n_z = s.theta_deg, s.n_z
        spec = run_spec_kind(self.path)
        return dict(
            name=self.name,
            label=self.name.split(f"-{self.run.meta['aircraft']['key']}-")[0],
            category=figures.FIG8_CATEGORY.get(spec or self.kind),
            dtheta=float(theta[w].max() - theta[w].min()),
            dn=float(n_z[w].min() - n_z[0]),
            dtheta_whole=float(theta.max() - theta.min()),
            dn_whole=float(n_z.min() - n_z[0]),
        )

    def time_at(self, index) -> float | None:
        """The cursor time for a sample index from a URL, or None if out of range."""
        try:
            i = int(index)
        except (TypeError, ValueError):
            return None
        if 0 <= i < len(self.series.t):
            return float(self.series.t[i])
        return None


def _record_note(record, shown: bool) -> str:
    """One line under the Paper record switch: what the band is, or why none."""
    if record is None:
        return "No published record of this encounter is held."
    if not shown:
        return f"Show what the source measured: {record.case}."
    return f"{record.case}. {record.note}"


def _badge(check: dict):
    """Check name, measured value, and a word that is not a bare tick.

    A `tripwire` that has never fired must NOT render as a green tick: that would
    claim evidence it does not provide. It renders its number and the word
    'quiet', which is honest about what it is.
    """
    word, tone, icon_name = ui.check_state(check)
    return html.Button(
        [
            ui.state(word, tone, icon_name),
            html.Span(check["name"]),
            html.Span(f"{check['value']:.4g}", className="ati-num ati-muted"),
        ],
        title=check["detail"] + "  Click to move the cursor to the worst sample.",
        id={"type": "badge", "index": check["name"]},
        n_clicks=0,
        className="ati-badge",
        type="button",
    )


# A declared parameter's key carries its unit as a suffix; the label is what is
# left. `<prefix>_provenance` is the note for the value its prefix names.
_SUFFIXES = (("_deg_from_trim", "", "deg from trim"), ("_core_radii", "", "core radii"),
             ("_seconds", "", "s"), ("_mps", "", "m/s"), ("_deg", "", "deg"), ("_m", "", "m"))
_LABELS = {"lead_in": "Lead-in"}


def _window_text(window: dict) -> str:
    kind = window.get("kind", "window")
    if window.get("north_m"):
        lo, hi = window["north_m"]
        return f"{kind}, north {lo:+.0f} to {hi:+.0f} m"
    if window.get("seconds") is not None:
        start = window.get("starts_at_s")
        return (f"{kind}, {window['seconds']:.4g} s"
                + (f" from t = {start:.4g} s" if start is not None else ""))
    return kind


def declared(params: dict) -> list[tuple[str, str, str]]:
    """(label, value, note) for each entry of meta.json's declared_parameters.

    Rounded for reading: the artifact keeps every digit, and the Script tab and
    meta.json are where a reader goes for them.
    """
    notes = {k[: -len("_provenance")]: v for k, v in params.items()
             if k.endswith("_provenance") and isinstance(v, str)}
    rows = []
    for key, value in params.items():
        if key.endswith("_provenance") or key == "window_rule":
            continue
        note = next((n for prefix, n in notes.items()
                     if key == prefix or key.startswith(prefix + "_")), None)
        if key == "window" and isinstance(value, dict):
            rule = params.get("window_rule")
            rows.append(("Window", _window_text(value),
                         f"Window rule: {rule}." if rule else ""))
            continue
        label, unit = key, ""
        for suffix, tail, u in _SUFFIXES:
            if key.endswith(suffix):
                label, unit = key[: -len(suffix)] + tail, u
                break
        label = _LABELS.get(label, label.replace("_", " ").capitalize())
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            continue
        text = value if isinstance(value, str) else f"{value:.4g}" + (f" {unit}" if unit else "")
        if note:
            # The artifact's own wording starts "DECLARED, not sourced --", which
            # the marker already says.
            note = re.sub(r"^DECLARED,?\s*not sourced\s*-+\s*", "", note, flags=re.I)
            note = note[:1].upper() + note[1:]
        rows.append((label, text, note or ""))
    return rows


def _fact(label: str, value, marker=None):
    return html.Div(
        [html.Dt(label), html.Dd(value)] + ([marker] if marker is not None else []),
        className="ati-fact",
    )


def _header(loaded: Loaded):
    """The run's facts as label | value pairs, then its declared parameters,
    each with the Declared marker the Setup page gives it.

    The pairs flow and wrap rather than sit in fixed columns: at 1366 px a
    column grid took three rows where the flow takes two, and pushed the traces
    and the scene below the fold on the landing view.
    """
    m = loaded.run.meta
    fc, tr = m["flight_condition"], m["trim"]
    facts = [
        _fact("Aircraft", m["aircraft"]["key"]),
        _fact("Wind field", KIND_LABELS.get(loaded.kind, loaded.kind)),
        _fact("Altitude", f"{fc['altitude_m']:.0f} m"),
        _fact("Airspeed", f"{fc['airspeed_mps']:.2f} m/s"),
        _fact("Trim alpha", f"{np.degrees(tr['alpha_rad']):.3f} deg"),
        _fact("Time step", f"{m['integrator']['dt_s']:g} s"),
        _fact("Steps", f"{m['integrator']['n_steps']}"),
        _fact("Commit", html.Span(m["git_sha"][:7] or "nogit", className="ati-mono")),
        _fact("Load path", m["load_model"] or "point sample plus analytic gradient"),
    ]
    children = [html.Dl(facts, className="ati-facts")]
    bad = ~np.isfinite(np.asarray(loaded.series.n_z, dtype=float))
    if bad.any():
        # Said at the top: the plots of a diverged run draw axes of 1e146 and a
        # readout of "nan", which a reader can take for a display fault.
        first = float(np.asarray(loaded.series.t)[int(np.argmax(bad))])
        children.insert(0, dmc.Alert(
            f"This run diverged: its values are not numbers from t = {first:.2f} s. "
            "The time step is too long for the aircraft or the field. Fly it again "
            "with a shorter time step.", color="red", title="Diverged",
            icon=ui.icon("circle-x", 16)))
    params = declared(m["declared_parameters"])
    if params:
        children.append(html.Div([
            html.Span("Declared parameters", className="ati-facts-title"),
            html.Dl([_fact(label, value, ui.marker("declared", note))
                     for label, value, note in params], className="ati-facts"),
        ], className="ati-facts-group"))
    for caveat in m.get("caveats", []):
        children.append(html.Div([ui.icon("alert-triangle", 14), html.Span(caveat)],
                                 className="ati-caveat"))
    if loaded.run.stale:
        # A badge, not an error: the run is still readable, only possibly old.
        children.append(html.Div(
            [ui.state("stale", "neutral", "tabler:history", 12),
             html.Span(f"Flown at {m['git_sha'][:7]}, which is not the commit you are "
                       "reading it with. Figures may not match today's code.")],
            className="ati-stale"))
    return children


def _readout(loaded: Loaded, index: int | None):
    """ParaView's Spreadsheet View: the numbers beside the render.

    `north / r0` is the row that earns this panel -- it turns "that peak" into
    "0.98 core radii, just inside the boundary" with no arithmetic by the reader.
    """
    if index is None:
        return html.Div("Click a strip or a check to place the time cursor.",
                        className="ati-muted", style={"padding": "6px 2px"})
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
    return html.Table([html.Tr([html.Td(k), html.Td(v)]) for k, v in rows])


def _fieldview(loaded: Loaded) -> tuple[list, str, list]:
    """(segments, value, note) for the field-view control on this run.

    The cross-section needs a field scale and a peak to pin its colours to. A run
    with neither (the manoeuvre flies in still air) gets the 3D scene, and the 2D
    segment is disabled WITH ITS REASON beside it rather than left looking live.
    """
    segments = [{"label": "2D cross-section", "value": "2d", "disabled": not loaded.can_slice},
                {"label": "3D scene", "value": "3d"}]
    if loaded.can_slice:
        return segments, "2d", []
    return segments, "3d", [ui.icon("info-circle", 14),
                            "No wind field on this run, so no cross-section."]


def _empty(root: Path):
    why = (f"The runs directory {root} holds no readable run artifacts." if root.exists()
           else f"The runs directory {root} does not exist yet. The first run creates it.")
    return html.Div([
        html.P([html.Strong("No runs yet. "), why]),
        html.P("Set up and fly a run, or fly one from a terminal:"),
        html.Pre(f"atisim run --preset vortex-hannibal --out {root}", className="ati-code"),
        dcc.Link(dmc.Button("New run", leftSection=ui.icon("plus", 14)),
                 href="/setup?preset=vortex-hannibal"),
    ], className="ati-empty", style={"maxWidth": "560px"})


def nested_run(root: Path, name: str | None) -> bool:
    """True if `name` is a run inside a report (an ensemble member), under root.

    `members/m000` of an ensemble is an ordinary run directory one level down.
    The Runs list shows only the top level, so a member is reached by its link
    and must resolve INSIDE the runs directory: `..` never leaves it.
    """
    if not name or "/" not in name:
        return False
    try:
        path = (root / name).resolve()
        path.relative_to(root.resolve())
    except (OSError, ValueError):
        return False
    return (path / "run.parquet").exists() and (path / "meta.json").exists()


def _cell(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return "nan" if value != value else f"{value:.6g}"
    return str(value)


def _report_table(tab: dict):
    rows = tab.get("rows", [])
    numeric = [bool(rows) and all(isinstance(r[i], (int, float)) or r[i] is None
                                  for r in rows if i < len(r))
               for i in range(len(tab.get("headers", [])))]
    return html.Table([
        html.Thead(html.Tr([html.Th(h, className="num" if numeric[i] else None)
                            for i, h in enumerate(tab["headers"])])),
        html.Tbody([html.Tr([html.Td(_cell(v), className="num" if i < len(numeric)
                                     and numeric[i] else None)
                             for i, v in enumerate(r)]) for r in rows]),
    ], className="ati-table")


def file_url(run: str, rel: str) -> str:
    """Where the app serves a file of a run or report directory (`shell.FILES`)."""
    return f"/files/{run}/{rel}"


def report_view(root: Path, name: str) -> list:
    """A report directory: its facts, then each section's text, figure and table.

    The figures are the analysis's own Plotly JSON, drawn as stored; nothing
    here re-derives a number. An ensemble's members are ordinary runs and each
    opens in the deep dive.
    """
    loaded = report.read_report(root / name)
    m = loaded.meta
    spec = m.get("spec", {})
    study = m.get("analysis") == "study"
    facts = [
        _fact("Analysis", m.get("analysis", "")),
        _fact("Script", html.Span(f"scripts/{spec.get('params', {}).get('script', '')}.py",
                                  className="ati-mono")) if study else
        _fact("Aircraft", (m.get("aircraft") or {}).get("key") or "none"),
        _fact("Created", runs_mod._local(m.get("created"))),
        _fact("Commit", html.Span(m.get("git_sha", "")[:7] or "nogit", className="ati-mono")),
    ]
    head = [
        html.H2(m.get("title", name), className="ati-report-title"),
        html.P(m.get("summary", ""), className="ati-report-summary"),
        html.Dl(facts, className="ati-facts"),
    ]
    for caveat in m.get("caveats", []):
        head.append(html.Div([ui.icon("alert-triangle", 14), html.Span(caveat)],
                             className="ati-caveat"))
    sha, here = m.get("git_sha", ""), artifact.git_sha()
    if sha and here and sha != here:
        head.append(html.Div(
            [ui.state("stale", "neutral", "tabler:history", 12),
             html.Span(f"Written at {sha[:7]}, which is not the commit you are reading it "
                       "with. Run it again to see today's code.")], className="ati-stale"))
    head.append(html.Div([
        dcc.Link(dmc.Button("Run again", leftSection=ui.icon("refresh", 14), size="xs",
                            variant="default"),
                 href=f"/analyses?from={name}"),
        html.Span("Opens this analysis in Analyses with the same parameters.",
                  className="ati-muted"),
    ], className="ati-controls", style={"marginTop": "8px"}))
    blocks = [html.Div(head, className="ati-deep-block")]
    members = m.get("members", [])
    if members:
        blocks.append(html.Div([
            html.Div(f"Members ({len(members)})", className="ati-facts-title"),
            html.Div([dcc.Link(Path(member).name, href=f"/results?run={name}/{member}",
                               className="ati-mono") for member in members],
                     className="ati-member-links"),
        ], className="ati-deep-block"))
    for section in loaded.sections:
        children = []
        if section.get("title"):
            # A study names a file's section by its path: set in mono like a path.
            path = "/" in section["title"] and " " not in section["title"]
            children.append(html.H3(section["title"], className="ati-report-section"
                                    + (" ati-mono" if path else "")))
        if section.get("text"):
            children.append(html.P(section["text"], className="ati-report-text"))
        if section.get("figure"):
            fig = section["figure"]
            height = (fig.get("layout") or {}).get("height") or 360
            children.append(dcc.Graph(figure=fig, config=_GRAPH,
                                      style={"height": f"{height}px"}))
        if section.get("table"):
            children.append(html.Div(_report_table(section["table"]),
                                     className="ati-report-table"))
        if section.get("image"):
            children.append(html.A(html.Img(src=file_url(name, section["image"]),
                                            alt=section.get("title") or section["image"],
                                            className="ati-report-image"),
                                   href=file_url(name, section["image"]), target="_blank"))
        if section.get("log"):
            children.append(html.Pre(section["log"], className="ati-log ati-report-log"))
        if section.get("files"):
            children.append(html.Div([
                html.A([ui.icon("file", 14), f], href=file_url(name, f), target="_blank",
                       className="ati-file-link")
                for f in section["files"]], className="ati-file-links"))
        blocks.append(html.Div(children, className="ati-deep-block"))
    blocks.append(html.Details([
        html.Summary("Analysis spec", style={"cursor": "pointer"}),
        html.Pre(__import__("json").dumps(spec, indent=2), className="ati-log"),
    ], className="ati-deep-block"))
    return blocks


VIEWS = (("overview", "Overview"), ("diagnostics", "Diagnostics"), ("compare", "Compare"))


def workspace_switch(run: str | None, view: str) -> html.Div:
    """Overview, Diagnostics, Compare: one run three ways (plan section 4).

    Links, not a control with state: each workspace is its own page, so a
    reader can bookmark or share it, and no callback of one sees another."""
    items = []
    for key, label in VIEWS:
        query = [] if run is None else [f"run={run}"]
        if key != "overview":
            query.append(f"view={key}")
        items.append(dcc.Link(label, href="/results?" + "&".join(query),
                              className="ati-switch-item"
                              + (" is-active" if key == view else ""),
                              id=f"view-{key}"))
    return html.Div(items, className="ati-switch", role="navigation",
                    **{"aria-label": "Workspace"})


def picker(rows, value, picker_id, placeholder="Open another result",
           runs_only=False, exclude=()):
    """Every readable result, grouped as the explorer groups them, searchable.

    `runs_only` leaves out the reports, and `exclude` the names already chosen:
    Compare offered a Modes report, which it then dropped without a word, and
    run A itself."""
    from atisim.apps.explorer import RESULT_GROUPS

    data = []
    for group, title in RESULT_GROUPS:
        items = [{"value": r.name, "label": r.name} for r in rows
                 if r.error is None and r.group == group and r.name not in exclude
                 and not (runs_only and r.verdict == "report")]
        if items:
            data.append({"group": title, "items": items})
    return dmc.Select(id=picker_id, value=value, data=data, searchable=True, w=340,
                      placeholder=placeholder, nothingFoundMessage="No result by that name",
                      leftSection=ui.icon("search", 14), maxDropdownHeight=420,
                      comboboxProps={"withinPortal": True}, **{"aria-label": placeholder})


def result_bar(row, pick, switch=None, title=None):
    """The head of every Results page: the result, what it is, a picker, the views."""
    facts = []
    if row is not None:
        facts = [("Kind", row.kind)]
        if row.aircraft:
            facts.append(("Aircraft", row.aircraft))
        facts.append(("Created", html.Span(row.created_local, className="ati-num")))
        if row.verdict != "report":
            facts.append(("Checks", ui.verdict_state(row.verdict)))
        if row.stale:
            facts.append(("Code", ui.state("stale", "neutral", "tabler:history", 13,
                                           title="Written by another commit; figures may "
                                                 "not match today's code")))
    head = ui.page_head(title or (row.name if row is not None else "Results"), None, facts,
                        actions=[pick])
    return html.Div([head, switch], className="ati-result-bar")


def layout(ws, run: str | None = None, t=None, view: str | None = None,
           extra: str | None = None):
    view = view if view in dict(VIEWS) else "overview"
    rows = runs_mod.scan(ws.root)
    good = [r for r in rows if r.error is None]
    names = {r.name for r in good}
    member = nested_run(ws.root, run)
    selected = run if run in names or member else (good[0].name if good else None)
    is_report = selected is not None and not member and report.is_report(ws.root / selected)

    row = next((r for r in good if r.name == selected), None)
    if view == "compare":
        from atisim.apps import compare
        from atisim.apps.compare import parse_runs

        bar = result_bar(None, picker(rows, None, "compare-add", "Add a run to compare",
                                      runs_only=True, exclude=parse_runs(run, extra)),
                         workspace_switch(run, "compare"), title="Compare runs")
        return compare.layout(ws, run, bar, None, extra)
    if view == "diagnostics" and selected is not None and not is_report:
        from atisim.apps import diagnostics
        bar = result_bar(row, picker(rows, selected, "diag-run"),
                         workspace_switch(selected, "diagnostics"))
        return diagnostics.layout(ws, selected, t, bar, None)

    # A report is not a time series, so the deep dive's callbacks must never see
    # one: its page carries the picker under another id.
    if selected is None:
        return html.Div([result_bar(None, None),
                         html.Div(_empty(ws.root), className="ati-deep")],
                        className="ati-results")

    if is_report:
        return html.Div([result_bar(row, picker(rows, selected, "report-run")),
                         html.Div(report_view(ws.root, selected), className="ati-deep")],
                        className="ati-results")
    bar = result_bar(row, picker(rows, selected, "run"),
                     workspace_switch(selected, "overview"))

    loaded = ws.loaded(selected)
    cursor = loaded.time_at(t) if t is not None else None
    segments, view, note = _fieldview(loaded)
    parent = selected.split("/members/")[0] if member else None
    deep = html.Div([
        html.Div([ui.icon("arrow-left", 14), dcc.Link(f"Back to {parent}",
                                                      href=f"/results?run={parent}"),
                  html.Span(f"member {Path(selected).name} of the ensemble",
                            className="ati-muted")],
                 className="ati-deep-block ati-controls", style={"marginBottom": 0})
        if member else None,
        html.Div(id="header", className="ati-deep-block"),
        html.Div([
            html.Div(id="badges", className="ati-badges"),
        ], className="ati-deep-block"),
        html.Div([
            html.Div([
                html.Div([
                    dmc.Switch(id="record-show", checked=loaded.record is not None,
                               disabled=loaded.record is None, size="xs",
                               label="Paper record"),
                    html.Span(id="record-note", className="ati-note"),
                ], className="ati-controls"),
                dcc.Graph(id="strips", config=_GRAPH),
            ], className="ati-deep-cell"),
            html.Div(className="ati-vrule"),
            html.Div([
                html.Div([
                    # 2D IS THE DEFAULT, and for the Parks vortex it is not
                    # a simplification: `wind.vortex_wind` fixes dpsi = 0, so
                    # the field has NO east variation and one north-altitude
                    # plane contains all of it. The 3D scene is the option,
                    # not the baseline.
                    dmc.SegmentedControl(id="fieldview", value=view, size="xs",
                                         data=segments),
                    dmc.Select(
                        id="scalar", value="n_z", w=200,
                        data=[{"label": f"Colour 3D by {k}", "value": k}
                              for k in figures.SCALARS],
                        comboboxProps={"withinPortal": True},
                    ),
                    html.Span(note, id="fieldview-note", className="ati-note"),
                ], className="ati-controls"),
                # Tall from the start, as `nzalpha` below: the cross-section has
                # a colourbar, and a first draw in a 37 px box left it a 2 px plot.
                dcc.Graph(id="scene", config=_GRAPH, style={"height": "360px"}),
                html.Div(id="readout", className="ati-readout ati-num"),
            ], className="ati-deep-cell"),
        ], className="ati-deep-row"),
        # The verdict FIRST, then the domain-standard figure it comes from.
        # Fig. 8's claim is an ordering and Fig. 8 does not state it; this
        # panel does, and it is the one a reader should hit first.
        html.Div([dcc.Graph(id="ordering", config=_GRAPH)], className="ati-deep-block"),
        html.Div([
            html.Div([dcc.Graph(id="fig8", config=_GRAPH)], className="ati-deep-cell"),
            html.Div(className="ati-vrule"),
            html.Div([
                # Incidence and time EXCHANGE places; the load factor stays
                # on y in both. See `figures.load_vs_alpha`: the incidence
                # view is timeless on purpose, which is what makes the
                # straight line legible and what makes it unable to say when.
                html.Div([
                    html.Span("Load factor against"),
                    dmc.SegmentedControl(
                        id="nzx", value="alpha", size="xs",
                        data=[{"label": "angle of attack", "value": "alpha"},
                              {"label": "time", "value": "time"}],
                    ),
                ], className="ati-controls"),
                # Tall from the start. The render callback sets its height with
                # the figure, and the first draw came before it, in a 37 px box:
                # the colourbar then threw "axis scaling" and the panel stayed
                # a 3 px plot with no title.
                dcc.Graph(id="nzalpha", config=_GRAPH, style={"height": "340px"}),
            ], className="ati-deep-cell"),
        ], className="ati-deep-row is-even"),
        dcc.Store(id="cursor", data=cursor),
        dcc.Store(id="run-shown", data=selected),
    ], className="ati-deep")
    return html.Div([bar, deep], className="ati-results")


def register(app, ws) -> None:
    def _guard(run_name):
        """The deep dive draws runs only; a report has its own page."""
        if not run_name or report.is_report(ws.root / run_name):
            raise PreventUpdate

    @app.callback(
        Output("url", "search", allow_duplicate=True),
        Input("run", "value"), State("run-shown", "data"),
        prevent_initial_call=True,
    )
    def open_result(run_name, shown):
        """The picker in the result bar opens the result it names, as a page,
        so the title, the breadcrumb and the explorer follow it."""
        if run_name and run_name != shown:
            return f"?run={run_name}"
        return no_update

    @app.callback(
        Output("url", "search", allow_duplicate=True),
        Input("report-run", "value"),
        prevent_initial_call=True,
    )
    def leave_report(run_name):
        return f"?run={run_name}" if run_name else no_update

    @app.callback(
        Output("cursor", "data"),
        Input("strips", "clickData"), Input("run-shown", "data"),
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

        _guard(run_name)
        trigger = dash.ctx.triggered_id
        if trigger == "run-shown":
            return None
        if isinstance(trigger, dict) and trigger.get("type") == "badge":
            if not any(badge_clicks or []):
                return current  # fired on layout build, not on a real click
            loaded = ws.loaded(run_name)
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
        Output("fieldview", "data"), Output("fieldview", "value"),
        Output("fieldview-note", "children"),
        Input("run-shown", "data"), State("fieldview", "data"), State("fieldview", "value"),
        prevent_initial_call=True,
    )
    def fieldview_for_run(run_name, segments, view):
        """Changing run re-decides the field view. A reader's own 3D choice is
        kept; a 3D view the previous run FORCED goes back to the 2D default."""
        _guard(run_name)
        loaded = ws.loaded(run_name)
        new_segments, default, note = _fieldview(loaded)
        forced = any(item.get("disabled") for item in segments or [])
        keep = loaded.can_slice and not forced
        return new_segments, view if keep else default, note

    @app.callback(
        Output("header", "children"), Output("badges", "children"),
        Output("strips", "figure"), Output("scene", "figure"),
        Output("ordering", "figure"),
        Output("fig8", "figure"), Output("nzalpha", "figure"),
        Output("readout", "children"),
        Output("strips", "style"), Output("scene", "style"), Output("ordering", "style"),
        Output("fig8", "style"), Output("nzalpha", "style"),
        Output("record-note", "children"),
        Input("run-shown", "data"), Input("scalar", "value"), Input("cursor", "data"),
        Input("fieldview", "value"), Input("nzx", "value"), Input("record-show", "checked"),
        State("scene", "relayoutData"),
    )
    def render(run_name, scalar, cursor_t, fieldview, nzx, show_record, scene_relayout):
        """Everything updates from the cursor in ONE callback, so nothing tears.

        Six outputs, one input set. Splitting this into six callbacks would let
        the strips show one moment while the 3D marker showed another, which is
        exactly the failure the shared time base exists to prevent.

        `scene_relayout` is a `State`, not an `Input`: rotating the scene must not
        redraw anything. It exists so the camera can be put back EXPLICITLY --
        `uirevision` is set as well, but this project does not rely on a mechanism
        it cannot assert on. See `figures.apply_camera`.
        """
        _guard(run_name)
        loaded = ws.loaded(run_name)
        s = loaded.series
        index = None if cursor_t is None else int(np.argmin(np.abs(s.t - cursor_t)))

        # The cross-section needs a core radius and a peak to pin its scale to.
        # A field that supplies neither (the manoeuvre run has no field at all)
        # falls back to the 3D panel rather than drawing an unpinned map, and
        # `fieldview_for_run` has already said so on the control.
        if fieldview == "2d" and loaded.can_slice:
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
        points = ws.fig8_points()
        figs = (
            figures.strip_stack(s, loaded.window, cursor_t=cursor_t,
                                record=loaded.record if show_record else None),
            scene,
            figures.ordering(points, current=loaded.name),
            figures.discriminator(points, current=loaded.name),
            figures.load_vs_alpha(s, cursor_index=index, against=nzx),
        )
        return (
            _header(loaded),
            [_badge(c) for c in loaded.run.checks],
            *figs,
            _readout(loaded, index),
            # Each graph is as tall as its figure says. `responsive` then fills
            # only the width; left to fill a grid cell's height, a graph takes
            # the height of its neighbour and the figure's own layout breaks.
            *[{"height": f"{fig.layout.height}px"} for fig in figs],
            _record_note(loaded.record, show_record),
        )
