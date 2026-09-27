"""The landing page: a flight-test card clipped to a kneeboard.

For a reader who does not know the code. Four numbered test points, each with
a sketch of the vertical wind it holds, and FLY opens the live cockpit
(`fly.py`). The result of a flown point comes back onto the card in grease
pencil. Everything else -- every parameter, the checks, the analyses, the runs
-- is the engineering workbench, one button away (`/start`).

The card's grammar is the test card's: labels sit to the LEFT of their value
in a ruled grid, red is only for the limits, and the one action per row is a
blue-ink stamp.
"""

from urllib.parse import quote

from dash import dcc, html

from atisim import cockpit
from atisim.aircraft import CRUISE
from atisim.apps import components as ui

PROFILE_SECONDS = 70.0
SKETCH_W, SKETCH_H = 280, 64
SKETCH_SCALE = 30.0  # m/s at the sketch's edge: one scale for every row
# The card's inks, for the SVGs drawn as image elements, which cannot read the
# stylesheet's variables. They repeat card.css's --tc-ink, --tc-rule-soft and
# --tc-pencil.
INK, RULE_SOFT, PENCIL = "#16181b", "#b9bec5", "#2b2724"

_profiles: dict[str, dict] = {}


def profile(key: str) -> dict:
    """The test point's vertical wind against time, computed once per process."""
    if key not in _profiles:
        _profiles[key] = cockpit.gust_profile(key, PROFILE_SECONDS)
    return _profiles[key]


def _sketch(key: str):
    """The air ahead, drawn from the placed field. Up is up."""
    p = profile(key)
    mid = SKETCH_H / 2
    points = " ".join(
        f"{SKETCH_W * t / PROFILE_SECONDS:.1f},"
        f"{mid - (SKETCH_H / 2 - 4) * max(-1.0, min(1.0, w / SKETCH_SCALE)):.1f}"
        for t, w in zip(p["t"], p["w_up"]))
    peak = max(p["w_up"], key=abs)
    label = ("still air" if abs(peak) < 0.05
             else f"{'up' if peak > 0 else 'down'} to {abs(peak):.0f} m/s")
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SKETCH_W} {SKETCH_H}" '
           f'preserveAspectRatio="none">'
           f'<line x1="0" y1="{mid}" x2="{SKETCH_W}" y2="{mid}" stroke="{RULE_SOFT}" '
           f'stroke-width="1" vector-effect="non-scaling-stroke"/>'
           f'<polyline points="{points}" fill="none" stroke="{INK}" stroke-width="1.6" '
           f'stroke-linejoin="round" vector-effect="non-scaling-stroke"/></svg>')
    return html.Div([
        html.Img(src=_data_uri(svg), alt="", className="tc-sketch-plot"),
        html.Span(label, className="tc-sketch-label"),
    ], className="tc-sketch",
        title=f"Vertical wind along a straight, level path at cruise, first "
              f"{PROFILE_SECONDS:.0f} s. Drawn from the field the flight meets; "
              "nothing is flown for it.")


def _result(mark: dict | None):
    """What the pilot wrote on the card after flying the point."""
    if not mark:
        return html.Span("not flown", className="tc-unflown")
    return html.Div([
        html.Span([_tick(), f"{mark['nz_max']:+.2f} g"], className="tc-pencil"),
        html.Span(f"low {mark['nz_min']:+.2f} g · {mark['severity']}",
                  className="tc-pencil-note"),
    ], className="tc-result", title="Your last flight of this point: the peak and "
                                    "lowest load factor, and the turbulence band")


def _tick():
    """A grease-pencil tick: one stroke, drawn, not a glyph."""
    return html.Img(src=_data_uri(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
        f'<path d="M3 13.5c2.2 1.6 3.6 3.4 5 6.2C11 12.6 15.4 7 21 3.6" fill="none" '
        f'stroke="{PENCIL}" stroke-width="3.2" stroke-linecap="round" '
        'stroke-linejoin="round"/></svg>'), alt="", className="tc-tick")


def _data_uri(svg: str) -> str:
    return "data:image/svg+xml;charset=utf-8," + quote(svg)


def _row(n: int, tp: cockpit.TestPoint, mark):
    return html.Tr([
        html.Td(html.Span(f"{n}", className="tc-tp-num"), className="tc-col-tp"),
        html.Td([
            html.Div(tp.title, className="tc-tp-title"),
            html.Div(tp.where, className="tc-tp-where") if tp.where else None,
            html.P(tp.summary, className="tc-tp-summary"),
        ], className="tc-col-what"),
        html.Td(_sketch(tp.key), className="tc-col-air"),
        html.Td(_result(mark), className="tc-col-result"),
        html.Td(dcc.Link([html.Span("Fly"), html.Span(f"TP-{n}", className="tc-stamp-tp"),
                          ui.icon("plane", 18)],
                         href=f"/fly?tp={tp.key}", className="tc-stamp",
                         title=f"Fly test point {n}, {tp.title}, in the live cockpit"),
                className="tc-col-fly"),
    ], className="tc-point")


KEYS = (
    (("arrow-up", "arrow-down"), "Stick: forward is nose down, back is nose up"),
    (("arrow-left", "arrow-right"), "Roll left and right"),
    (("-", "="), "Throttle down and up"),
    ((",", "."), "Rudder left and right"),
    (("A",), "Autopilot on or off: it holds the start condition"),
    (("T",), "Trim here: hold the stick you have now"),
    (("Space",), "Pause and resume"),
)


def keycap(name: str):
    if name.startswith("arrow-"):
        return html.Kbd(ui.icon(name, 15), className="tc-key",
                        title=name.replace("-", " ") + " key")
    return html.Kbd(name, className="tc-key tc-key-wide" if len(name) > 1 else "tc-key")


def controls(compact: bool = False):
    return html.Dl([
        item for keys, what in KEYS
        for item in (html.Dt([keycap(k) for k in keys]),
                     html.Dd(what, className="tc-key-what"))
    ], className="tc-keys" + (" is-compact" if compact else ""))


def limits():
    return html.Div([
        html.Div("Limits", className="tc-limits-title"),
        html.Ul([
            html.Li(f"Angle of attack past {cockpit.LIMITS['alpha_linear_deg']:.0f}°: "
                    "the model's linear aerodynamics stop holding. Past "
                    f"{cockpit.LIMITS['alpha_invalid_deg']:.0f}° its lift is not "
                    "real, and the flight proves nothing."),
            html.Li("The 747 model is valid only near its cruise condition. Every "
                    "flight starts there; far from it, the numbers are not sourced."),
            html.Li("Parks identified both vortex pairs from DC-10 flight records. "
                    "The aircraft flying them here is the 747 model."),
        ]),
    ], className="tc-limits", role="note")


def top_bar(dark: bool = False, back: bool = False):
    return html.Header([
        dcc.Link("AtiSim", href="/", className="tc-wordmark"),
        dcc.Link([ui.icon("arrow-left", 15), "Test card"], href="/",
                 className="tc-bar-link") if back else None,
        html.Span(className="tc-bar-spacer"),
        ui.mode_switch("card"),
    ], className="tc-bar" + (" is-dark" if dark else ""))


def layout(ws):
    airspeed, altitude = (CRUISE[cockpit.AIRCRAFT][k] for k in ("airspeed", "altitude"))
    marks = getattr(ws, "marks", {})
    rows = [_row(n, tp, marks.get(tp.key))
            for n, tp in enumerate(cockpit.TEST_POINTS.values(), start=1)]
    head = html.Div([
        html.Div(html.H1("Test card: clear-air turbulence", className="tc-title"),
                 className="tc-head-cell tc-head-main"),
        html.Div([html.Span("Aircraft", className="tc-field"),
                  html.Span("Boeing 747 model", className="tc-value")],
                 className="tc-head-cell"),
        html.Div([html.Span("Start", className="tc-field"),
                  html.Span(f"{airspeed:.0f} m/s · {altitude:,.0f} m",
                            className="tc-value")],
                 className="tc-head-cell"),
        html.Div([html.Span("Card", className="tc-field"),
                  html.Span("1 of 1", className="tc-value")],
                 className="tc-head-cell"),
    ], className="tc-head")
    objective = html.Div([
        html.Span("Objective", className="tc-field"),
        html.P(["Fly an airliner through turbulence that real airliners met, "
                "and read what it did to the aircraft. AtiSim is a flight "
                "dynamics model: it computes the aircraft's motion 50 times a second "
                "from its equations of motion, as you fly."],
               className="tc-objective-text"),
    ], className="tc-objective")
    table = html.Table([
        html.Thead(html.Tr([
            html.Th("TP", className="tc-col-tp"),
            html.Th("Encounter", className="tc-col-what"),
            html.Th([html.Span("The air ahead"),
                     html.Span(f"vertical wind, first {PROFILE_SECONDS:.0f} s",
                               className="tc-th-sub")], className="tc-col-air"),
            html.Th("Result", className="tc-col-result"),
            html.Th(html.Span("Fly", className="tc-visually-hidden"),
                    className="tc-col-fly"),
        ])),
        html.Tbody(rows),
    ], className="tc-points")
    lower = html.Div([
        html.Section([html.Div("Controls", className="tc-section-title"), controls()],
                     className="tc-controls", **{"aria-label": "Controls"}),
        limits(),
    ], className="tc-lower")
    foot = html.Footer([
        html.P([html.Strong("Sources. "),
                "Vortex pairs: Parks, Wingrove, Bach & Mehta 1985, J. Aircraft 22(2) "
                "124-129. Updraft: Wingrove & Bach 1994, J. Aircraft 31(4) 753-760. "
                "Every run the workbench saves carries its own provenance."],
               className="tc-sources"),
        dcc.Link([ui.icon("adjustments-horizontal", 16),
                  "Open the engineering workbench"], href="/start",
                 className="tc-foot-eng"),
    ], className="tc-foot")
    return html.Div([
        top_bar(),
        html.Article([
            html.Div(className="tc-clip", **{"aria-hidden": "true"}),
            head, objective, table, lower, foot,
        ], className="tc-card", **{"aria-label": "Test card"}),
    ], className="tc-board")
