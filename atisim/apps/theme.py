"""The app's visual system, in one place: palette, DMC theme, Plotly template.

The direction is an engineering simulation suite played straight (COMSOL Model
Builder, Ansys Workbench, MATLAB App Designer): light only, neutral chrome one
step off white, 1px dividers, 4px radius, one engineering blue for selection,
the primary action and focus. Green, amber and red are reserved for run and
check status and always come with an icon and a word.

The scene decides light: a researcher at a desk in daylight, then the same
screen projected in a lit meeting room. So contrast is high and no label is
thin grey text -- every text colour here clears 4.5:1 on every surface it sits
on (checked in `test_app.py`).

The Plotly template is registered here and nowhere else. `figures._base` picks
it up by name when it is registered, so the analysis figures match the shell
without being restyled one by one, and stay importable from a notebook with no
app in sight.
"""

import plotly.graph_objects as go
import plotly.io as pio

from atisim.analysis import figures

# --- palette ---------------------------------------------------------------
# Chrome: neutral, very slightly cool, so the warm-neutral figure surfaces
# (figures.SURFACE) read as content rather than as more chrome.
CHROME = "#eceef1"  # header, status bar, dock tabs
PANEL = "#f5f6f8"  # tree and settings panels
SURFACE = figures.SURFACE  # graphics and tables: the figures' own ground
DIVIDER = "#d3d8de"  # every 1px rule
INK = "#14181d"
INK_MUTED = "#4a525c"  # 7.2:1 on PANEL, 6.8:1 on CHROME
ACCENT = "#1f5fbf"  # 6.1:1 on white: selection, primary, focus
ACCENT_HOVER = "#184d9c"
ACCENT_WASH = "#e1ebf8"  # selected row

# Status, reserved. Text colours clear 4.5:1 on white and on their wash.
OK = "#1a7334"
OK_WASH = "#e3f1e6"
WARN = "#8a5a00"
WARN_WASH = "#fbf0d9"
FAIL = "#b42318"
FAIL_WASH = "#fbe7e5"

FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
MONO = 'ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace'


def _ramp(base: str) -> list[str]:
    """Ten shades for Mantine, lightest first, with `base` at index 6."""
    return ["#eef3fb", "#dce7f7", "#b7cdee", "#8fb1e4", "#6a96da", "#4479cd",
            base, ACCENT_HOVER, "#143f80", "#0e2d5c"]


MANTINE_THEME = {
    "primaryColor": "engblue",
    "primaryShade": 6,
    "colors": {"engblue": _ramp(ACCENT)},
    "fontFamily": FONT,
    "fontFamilyMonospace": MONO,
    "headings": {"fontFamily": FONT, "fontWeight": "600"},
    "defaultRadius": "sm",
    "radius": {"xs": "2px", "sm": "4px", "md": "4px", "lg": "4px", "xl": "4px"},
    "fontSizes": {"xs": "12px", "sm": "13px", "md": "14px", "lg": "16px", "xl": "18px"},
    "black": INK,
    "focusRing": "auto",
    "cursorType": "pointer",
    "components": {
        "Button": {"defaultProps": {"size": "xs", "radius": "sm"}},
        "NumberInput": {"defaultProps": {"size": "xs"}},
        "TextInput": {"defaultProps": {"size": "xs"}},
        "Select": {"defaultProps": {"size": "xs", "allowDeselect": False}},
        "Switch": {"defaultProps": {"size": "xs"}},
        "Tabs": {"defaultProps": {"radius": "sm"}},
        "Tooltip": {"defaultProps": {"withArrow": True, "openDelay": 250,
                                      "multiline": True, "maw": 360}},
    },
}

TEMPLATE_NAME = "atisim"


def plotly_template() -> go.layout.Template:
    """Fonts, colours and gridlines matched to the shell, over plotly_white."""
    base = pio.templates["plotly_white"]
    template = go.layout.Template(base)
    template.layout.font = dict(family=FONT, size=11, color=figures.INK)
    template.layout.colorway = list(figures.SERIES)
    template.layout.paper_bgcolor = SURFACE
    template.layout.plot_bgcolor = SURFACE
    template.layout.xaxis = dict(gridcolor=figures.GRID, linecolor=figures.AXIS_RULE,
                                 zerolinecolor=figures.AXIS_RULE)
    template.layout.yaxis = dict(gridcolor=figures.GRID, linecolor=figures.AXIS_RULE,
                                 zerolinecolor=figures.AXIS_RULE)
    # The 3D scene's walls are the figure surface, not plotly_white's white box.
    wall = dict(backgroundcolor=SURFACE, showbackground=True, gridcolor=figures.GRID,
                gridwidth=1, linecolor=figures.AXIS_RULE, zerolinecolor=figures.AXIS_RULE)
    template.layout.scene = dict(bgcolor=SURFACE, xaxis=wall, yaxis=wall, zaxis=wall)
    template.layout.hoverlabel = dict(font=dict(family=FONT, size=11),
                                      bgcolor="#ffffff", bordercolor=DIVIDER)
    template.layout.separators = ".,"
    return template


def register() -> None:
    """Register the template. Idempotent; the app calls it once at build."""
    pio.templates[TEMPLATE_NAME] = plotly_template()


def contrast(fg: str, bg: str) -> float:
    """WCAG contrast ratio between two hex colours."""

    def lum(hex_colour):
        rgb = [int(hex_colour.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4)]
        lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
        return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]

    hi, lo = sorted((lum(fg), lum(bg)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)
