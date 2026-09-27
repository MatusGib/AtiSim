"""Small shared pieces: icons, status words, provenance markers.

Status is never colour alone: every state here is an icon, a word and a
colour together, and the words are the ones `checks.py` defines. A `report`
check renders its number and the word "report" in ink, never green: it has no
threshold, and colouring it would invent a verdict.
"""

from dash import html
from dash_iconify import DashIconify

ICONS = {
    "ok": "tabler:circle-check",
    "warn": "tabler:alert-triangle",
    "fail": "tabler:circle-x",
    "neutral": "tabler:circle-dot",
    "report": "tabler:ruler-measure",
    "running": "tabler:loader-2",
    "idle": "tabler:player-pause",
}


def icon(name: str, size: int = 16, **style) -> DashIconify:
    """A Tabler icon by its short name (without the `tabler:` prefix) or full name."""
    full = name if ":" in name else f"tabler:{name}"
    return DashIconify(icon=full, width=size, height=size, style=style or None)


def state(word: str, tone: str, icon_name: str | None = None, size: int = 14,
          title: str | None = None) -> html.Span:
    """Icon, word and colour together. `tone` is ok, warn, fail or neutral.

    The running glyph turns (`is-busy` in the stylesheet), unless the reader has
    asked for reduced motion.
    """
    name = icon_name or ICONS.get(tone, ICONS["neutral"])
    busy = " is-busy" if name == ICONS["running"] else ""
    return html.Span(
        [icon(name, size), word],
        className=f"ati-state ati-state-{tone}{busy}", title=title,
    )


def check_state(check: dict) -> tuple[str, str, str]:
    """(word, tone, icon) for one check, following `checks.Check.kind`.

    - gate: PASS or FAIL, and MARGINAL (amber) for an alpha band that passes in
      the amber band, which `checks.AlphaBand` leaves for the UI to say.
    - tripwire: quiet (neutral, never a green tick: it has had nothing to catch)
      or FIRED.
    - report: report, in ink. No verdict exists to colour.
    """
    passed, kind = check.get("passed"), check.get("kind")
    if kind == "gate":
        if passed is False:
            return "FAIL", "fail", ICONS["fail"]
        if check.get("name") == "alpha band" and "(MARGINAL)" in check.get("detail", ""):
            return "MARGINAL", "warn", ICONS["warn"]
        return "PASS", "ok", ICONS["ok"]
    if kind == "tripwire":
        if passed is False:
            return "FIRED", "fail", ICONS["fail"]
        return "quiet", "neutral", ICONS["neutral"]
    return "report", "neutral", ICONS["report"]


def verdict_state(verdict: str) -> html.Span:
    """The runs list's one word per run."""
    tone, icon_name = {
        "pass": ("ok", ICONS["ok"]),
        "warning": ("warn", ICONS["warn"]),
        "fail": ("fail", ICONS["fail"]),
        "error": ("fail", ICONS["fail"]),
        "report": ("neutral", ICONS["report"]),
    }.get(verdict, ("neutral", ICONS["neutral"]))
    return state(verdict, tone, icon_name)


def marker(status: str, note: str, marker_id=None) -> html.Span:
    """The provenance marker: Sourced, or Declared with an amber outline.

    The tooltip is the citation or the reason, so the marker is never a bare
    word. This is the product's truth rule made visible. Declared has its own
    glyph, a pencil: the alert triangle means a validation warning, and a
    declared value is a choice someone made, not a problem.
    """
    declared = status == "declared"
    kwargs = {"id": marker_id} if marker_id is not None else {}
    return html.Span(
        [icon("pencil" if declared else "book-2", 12),
         "Declared" if declared else "Sourced"],
        className=f"ati-marker ati-marker-{'declared' if declared else 'sourced'}",
        title=(f"Declared: modelling parameter, not source data. {note}" if declared
               else f"Sourced: {note}"),
        tabIndex=0,
        **kwargs,
    )


def middle(text: str, limit: int = 48) -> str:
    """Truncate in the middle, keeping both ends of a path readable."""
    if len(text) <= limit:
        return text
    head = (limit - 1) // 2
    tail = limit - 1 - head
    return f"{text[:head]}…{text[-tail:]}"


def page_head(title, lead=None, facts=None, actions=None) -> html.Header:
    """A page's head: its title, one line on what it is, its facts, its actions.

    `facts` are (label, value) pairs, set as a row of label | value; a value may
    be a component (a marker, a status word, a link).
    """
    return html.Header([
        html.Div([
            html.H1(title, className="ati-page-title"),
            html.P(lead, className="ati-page-lead") if lead else None,
            html.Dl([item for label, value in facts
                     for item in (html.Dt(label), html.Dd(value))],
                    className="ati-page-facts") if facts else None,
        ], className="ati-page-head-text"),
        html.Div(actions, className="ati-page-actions") if actions else None,
    ], className="ati-page-head")


def result_links(rows, limit: int = 8, empty: str = "None yet.") -> html.Div:
    """Earlier results as a list of links, newest first."""
    rows = list(rows)[:limit]
    if not rows:
        return html.Div(empty, className="ati-side-empty")
    from dash import dcc

    return html.Ul([
        html.Li([
            dcc.Link(middle(r.name, 40), href=f"/results?run={r.name}", title=r.name),
            html.Span(r.created_local, className="ati-muted ati-num"),
        ]) for r in rows
    ], className="ati-side-list")


MODES = (("card", "Test card", "/"), ("lab", "Lab", "/lab"),
         ("engineering", "Engineering", "/start"))


def mode_switch(active: str, on: str = "dark") -> html.Nav:
    """The three modes, on every page: Test card, Lab, Engineering.

    `on` is the ground it sits on: "dark" (the kneeboard and the glareshield)
    or "light" (engineering mode's header). The mode on screen is marked by
    its ground and says so to a screen reader.
    """
    from dash import dcc

    items = []
    for key, label, href in MODES:
        here = key == active
        items.append(dcc.Link(
            [label, html.Span(" (this mode)", className="ati-visually-hidden")] if here
            else label,
            href=href, className="mode-item" + (" is-active" if here else "")))
    return html.Nav(items, className=f"mode-switch is-on-{on}",
                    **{"aria-label": "Mode"})
