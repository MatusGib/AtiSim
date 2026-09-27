"""The live cockpit: fly a test point in the browser.

The page is a glareshield with the primary flight display drawn on a canvas,
and the test point's card clipped to a kneeboard beside it. `assets/cockpit.js`
does everything that moves: it sends the held keys about 25 times a second,
draws the instruments from what comes back, and shows the debrief. The flight
itself is `atisim.cockpit.Flight`, the loop `scripts/fly.py` flies.

The API, under /api/flight, is JSON over POST, bound to 127.0.0.1 with the app:

    POST /api/flight                {"tp": key}             -> {id, info, frame}
    POST /api/flight/<id>/poll      {"held": [...], "presses": {"a": n, "t": n}}
    POST /api/flight/<id>/pause
    POST /api/flight/<id>/summary   the debrief; also writes the card's result

A flight lives in memory until three newer ones exist. A poll for one that is
gone is a 404 that says so, and the page offers to fly the point again.
"""

import itertools
import threading
from collections import OrderedDict

from dash import dcc, html

from atisim import cockpit
from atisim.apps import card
from atisim.apps import components as ui

KEPT = 3

WATCH = {
    "calm": "Hold the start height and speed by hand for a minute. Then press A "
            "and let the autopilot hold them.",
    "hannibal": "The load factor, in g, as you cross each core. Try it by hand, "
                "then again on the autopilot.",
    "morton": "The load factor as you cross each core. The cores are smaller, so "
              "the bumps come faster.",
    "updraft": "Your altitude and pitch as the column lifts you, and the drop at "
               "its far edge.",
}


class Flights:
    """The flights this app is flying, newest last."""

    def __init__(self, kept: int = KEPT):
        self._flights: OrderedDict[str, cockpit.Flight] = OrderedDict()
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self.kept = kept

    def start(self, key: str) -> tuple[str, cockpit.Flight]:
        flight = cockpit.Flight(key)  # the slow part, outside the lock
        with self._lock:
            ident = f"f{next(self._ids)}"
            self._flights[ident] = flight
            while len(self._flights) > self.kept:
                self._flights.popitem(last=False)
        return ident, flight

    def get(self, ident: str) -> cockpit.Flight | None:
        with self._lock:
            return self._flights.get(ident)


def layout(ws, key: str | None):
    if key not in cockpit.TEST_POINTS:
        return html.Div([
            card.top_bar(dark=True, back=True),
            html.Div([
                html.P([html.Strong("There is no test point "), html.Code(str(key)), "."]),
                dcc.Link("Choose one on the test card", href="/"),
            ], className="fly-missing"),
        ], className="fly")
    n = list(cockpit.TEST_POINTS).index(key) + 1
    tp = cockpit.TEST_POINTS[key]
    knee = html.Article([
        html.Div(className="tc-clip", **{"aria-hidden": "true"}),
        html.Div([
            html.Span(f"TP-{n}", className="tc-tp-num"),
            html.Div([html.H1(tp.title, className="fly-title"),
                      html.Div(tp.where, className="tc-tp-where") if tp.where else None]),
        ], className="fly-knee-head"),
        html.Dl([
            html.Dt("Watch", className="tc-field"), html.Dd(WATCH[key]),
            html.Dt("Ahead", className="tc-field"),
            html.Dd("…", id="fly-ahead", className="fly-live"),
            html.Dt("Peak", className="tc-field"),
            html.Dd(html.Span("not yet", className="tc-unflown"), id="fly-peak",
                    className="fly-live"),
        ], className="fly-facts"),
        html.Div([
            html.Button([html.Span(ui.icon("player-play", 16), className="fly-ico-play"),
                         html.Span(ui.icon("player-pause", 16), className="fly-ico-pause"),
                         html.Span("Start", id="fly-pause-word")],
                        id="fly-pause", className="fly-btn fly-btn-primary", type="button",
                        disabled=True, title="Start or pause the flight (Space)"),
            html.Button([ui.icon("steering-wheel", 16),
                         html.Span("Autopilot off", id="fly-ap-word")],
                        id="fly-ap", className="fly-btn", type="button", disabled=True,
                        title="Engage or release the autopilot (A)",
                        **{"aria-pressed": "false"}),
            html.Button([ui.icon("report", 16), "End flight and read the recorder"],
                        id="fly-end", className="fly-btn", type="button", disabled=True),
        ], className="fly-actions"),
        html.Div("Controls", className="tc-section-title"),
        card.controls(compact=True),
        card.limits(),
        html.P(["Source: ", tp.source, "."], className="tc-sources"),
    ], className="tc-card fly-knee", **{"aria-label": f"Test point {n}"})
    deck = html.Section([
        html.Canvas(id="fly-pfd", className="fly-pfd", role="img",
                    **{"aria-label": "Primary flight display: airspeed, attitude, "
                                     "altitude, vertical speed, heading, load factor, "
                                     "angle of attack and wind"}),
        html.Div(id="fly-overlay", className="fly-overlay", **{"aria-live": "polite"}),
    ], className="fly-deck")
    return html.Div([
        card.top_bar(dark=True, back=True),
        html.Main([deck, html.Aside(knee, className="fly-side")], className="fly-main"),
    ], id="cockpit", className="fly", **{"data-tp": key, "data-tp-n": str(n)})


def register(app, ws) -> None:
    from flask import jsonify, request

    flights = ws.flights

    def missing(ident):
        return jsonify(error=f"flight {ident} has ended: the application restarted "
                             "or newer flights replaced it"), 404

    @app.server.post("/api/flight")
    def flight_start():
        key = (request.get_json(silent=True) or {}).get("tp")
        if key not in cockpit.TEST_POINTS:
            return jsonify(error=f"there is no test point {key!r}"), 400
        ident, flight = flights.start(key)
        return jsonify(id=ident, info=flight.info(), frame=flight.frame())

    @app.server.post("/api/flight/<ident>/poll")
    def flight_poll(ident):
        flight = flights.get(ident)
        if flight is None:
            return missing(ident)
        body = request.get_json(silent=True) or {}
        return jsonify(flight.poll(body.get("held", ()), body.get("presses")))

    @app.server.post("/api/flight/<ident>/pause")
    def flight_pause(ident):
        flight = flights.get(ident)
        if flight is None:
            return missing(ident)
        flight.pause()
        return jsonify(ok=True)

    @app.server.post("/api/flight/<ident>/summary")
    def flight_summary(ident):
        flight = flights.get(ident)
        if flight is None:
            return missing(ident)
        flight.pause()
        summary = flight.summary()
        ws.marks[flight.test_point.key] = {
            "nz_max": summary["nz_max"]["value"], "nz_min": summary["nz_min"]["value"],
            "severity": summary["severity_band"],
        }
        return jsonify(summary)
