"""Plotly figures for the analysis UI. Pure functions: run in, Figure out.

No Dash import anywhere in this module, which is what lets the same figures be
called from a notebook cell -- the reuse requirement is met by the ARCHITECTURE,
not by the framework. It is also the same standing property `vortex_viz.figure`
already has: it *"returns a Figure; never shows and never saves"*.

**The countermeasures in here are not styling.** Each one answers a specific way
a plot can look right while the data is wrong:

  autoscaling         axis ranges are pinned per panel, never left to the data.
                      Borrowed from ParaView, which makes "rescale over all
                      timesteps" an explicit button because the automatic mode
                      changes the picture with no indication that it did.
  decimation          `series.envelope`, never a stride. A stride costs 5.7% of
                      the headline load factor; the envelope costs 0.0000%.
  interpolation       `line_shape="linear"` everywhere. A spline drawn through
                      the Rankine core edge -- where the one-sided derivatives
                      differ by 2*V0/r0 with OPPOSITE SIGNS -- would draw a smooth
                      curve across a real discontinuity.
  colour              diverging maps ONLY for genuinely signed quantities, pinned
                      symmetric about zero. `vortex_viz._field_panel` already
                      makes this argument for pinning to +-V0, because the claim
                      under test is a signed up-then-down doublet and a sign error
                      must reverse the colour order visibly.
  shared axes         comparison panels plot the DIFFERENCE, never two overlaid
                      absolutes: h and h/2 differ by 4e-5 relative and are
                      indistinguishable when overlaid.

`UIREVISION` is held constant on every figure. Measured against Dash 4.4.1 /
Plotly 6.9.0: the 3D camera survives a callback-driven figure replacement, and
survives a change in trace count, only while it is unchanged.
"""

import numpy as np
import plotly.graph_objects as go

from flightsim.analysis.series import Series, envelope

# Held constant so a callback never resets the user's camera or zoom. Verified.
UIREVISION = "flightsim-analysis"

# ---------------------------------------------------------------------------
# Palette
#
# VALIDATED, not chosen. The previous set was matplotlib's default cycle, and
# `scripts/validate_palette.js` fails it: #2ca02c vs #ff7f0e is dE 0.7 under
# protanopia. Those two were the UPDRAFT and MANOEUVRE markers on Fig. 8 -- two
# of the three points whose ORDERING is the entire claim of the panel -- so a
# red-green colourblind reader could not read the project's headline result.
#
# These three clear every gate on the ALL-PAIRS list (Fig. 8 is a scatter, so
# adjacent-pairs is not the right test): worst CVD dE 9.2 light / 9.4 dark,
# worst normal-vision dE 24.0 light / 20.9 dark. Three is the documented cap for
# all-pairs, and three is exactly what the discriminator needs.
#
# AQUA IS BELOW 3:1 ON THE LIGHT SURFACE, so the relief rule applies: it never
# carries meaning alone. Every Fig. 8 point is direct-labelled and every run is
# in the table view.
# ---------------------------------------------------------------------------
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")  # blue, orange, aqua
REFERENCE = "#898781"  # muted ink -- the "wrong" trace, and the paper's markers
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e1e0d9"
AXIS_RULE = "#c3c2b7"
SURFACE = "#fcfcfb"
CRITICAL = "#d03b3b"  # status, reserved -- cursor and out-of-range only

# Sequential = ONE hue, light to dark. Viridis is perceptually uniform but it is
# multi-hue, and the rule here is stricter for a reason: a multi-hue ramp on the
# trajectory tube competes with the categorical hues elsewhere on the same
# screen. Blue steps 100 -> 700 from the reference ramp.
SEQUENTIAL = [
    [0.0, "#cde2fb"], [0.25, "#86b6ef"], [0.5, "#3987e5"],
    [0.75, "#256abf"], [1.0, "#0d366b"],
]
# Diverging: warm/cool poles with a NEUTRAL GRAY midpoint, never a hue at zero.
DIVERGING = [
    [0.0, "#184f95"], [0.25, "#6da7ec"], [0.5, "#f0efec"],
    [0.75, "#e88b8b"], [1.0, "#a52121"],
]

# Points per 2D panel. Measured: 12 Scattergl traces at 2,000 points serialise to
# 0.542 MB, at 4,018 to 1.081 MB and at 31,791 to 8.499 MB. 2,000 keeps a
# ten-panel sweep inside the 2 MB budget with the 3D scene alongside it.
STRIP_POINTS = 2000

# Plotly's Isosurface runs marching cubes CLIENT-SIDE, so the whole scalar volume
# ships: measured 1.465 MB at 32^3 and 5.002 MB at 48^3. (PyVista meshes
# server-side and would ship only triangles -- that is the swap to make if a
# finer core boundary is ever needed.)
#
# THE GRID IS NOT CUBIC, AND THAT IS PHYSICS RATHER THAN A BUDGET TRICK. The
# Parks vortex has NO east variation at all -- `wind.vortex_wind` fixes dpsi = 0,
# so the cores are infinite east-west lines, which is also why `panel.py` reports
# a range to them with no bearing. A cubic grid therefore spends 3/4 of its
# points resolving a direction the field is constant in. Eight planes is enough
# to define the extruded shape, and it buys the north-vertical resolution that
# actually draws the core boundary: a full cubic 32^3 scene came to 2.29 MB
# against the 2 MB budget, and this is 4x cheaper at the SAME resolution in the
# plane that matters.
ISO_GRID = 32  # north and vertical -- the plane the Rankine profile lives in
ISO_GRID_LATERAL = 8  # east -- the direction this field does not vary in

# Hairline, SOLID, one step off the surface. Never dashed: dashing reads as
# "threshold" or "projection" when it is only a grid.
_AXIS = dict(showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False,
             linecolor=AXIS_RULE, tickfont=dict(color=INK_MUTED, size=10),
             title_font=dict(color=INK_MUTED, size=11))


def _wrap(text: str, width: int) -> list[str]:
    """Greedy wrap that never splits a word.

    `textwrap` is not used because the strings carry HTML entities (`&lt;`,
    `&amp;`) that count as one glyph and several characters, and a wrapper that
    counted characters would break a line early or split an entity in half.
    """
    words, lines, current = text.split(), [], ""
    for word in words:
        glyphs = len(word)
        for entity, real in (("&lt;", 1), ("&gt;", 1), ("&amp;", 1)):
            glyphs -= word.count(entity) * (len(entity) - real)
        if current and len(current) + 1 + glyphs > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


def _base(fig: go.Figure, height: int, title: str | None = None,
          subtitle: str | None = None, legend: bool = False) -> go.Figure:
    """Chrome shared by every panel.

    `subtitle` carries WHAT THE PANEL IS FOR in one line. A reader who has to
    infer a panel's job from its axes will infer wrong -- that is what happened
    with the first version of this module, where the two scatter panels read as
    decoration because nothing on them said what they assert.
    """
    # WRAPPED, because plotly does not wrap a title and an unwrapped subtitle
    # runs straight off the panel. Measured in the browser: the cross-section's
    # subtitle overhung its card by 41 px and the incidence panel's by 96 px, on
    # a 597 px column. 70 characters fits the narrowest panel this layout makes.
    lines = _wrap(subtitle, 70) if subtitle else []
    text = title
    if title and lines:
        body = "<br>".join(lines)
        text = (f"{title}<br><span style='font-size:10px;color:{INK_MUTED}'>"
                f"{body}</span>")
    top = 30 if title else 8
    if lines:
        top = 34 + 13 * len(lines)
    fig.update_layout(
        uirevision=UIREVISION,
        height=height,
        margin=dict(l=68, r=16, t=top, b=38),
        title=dict(text=text, font=dict(size=13, color=INK), x=0.0, xanchor="left")
        if text else None,
        template="plotly_white",
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        showlegend=legend,
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=1.0,
                    xanchor="right", font=dict(size=10, color=INK_MUTED),
                    bgcolor="rgba(0,0,0,0)"),
        font=dict(size=11, color=INK,
                  family='system-ui, -apple-system, "Segoe UI", sans-serif'),
        hoverlabel=dict(font_size=11),
    )
    return fig


def apply_camera(fig: go.Figure, relayout_data: dict | None) -> go.Figure:
    """Put the user's 3D camera back into a freshly built figure.

    BELT AND BRACES OVER `uirevision`, and deliberately so. `uirevision` works by
    plotly's internal GUI-edit bookkeeping (`_fullLayout._preGUI`), which is
    genuine but is not something this project can assert on: two attempts to
    verify it from the browser produced two different `_preGUI` shapes, one of
    which restored the camera and one of which did not, and the difference was in
    the SIMULATION rather than in the mechanism.

    So the camera is also carried explicitly. `dcc.Graph`'s `relayoutData` holds
    whatever the user's last interaction relayed -- for a 3D drag that is
    `scene.camera` -- and writing it back into the new figure is deterministic,
    independent of plotly's internals, and testable without a browser.

    Handles both shapes plotly emits: a nested `{"scene.camera": {...}}` and the
    flattened `{"scene.camera.eye.x": 0.4, ...}` it sends for a partial update.
    """
    if not relayout_data:
        return fig
    camera = relayout_data.get("scene.camera")
    if camera is None:
        flat = {k: v for k, v in relayout_data.items()
                if k.startswith("scene.camera.")}
        if not flat:
            return fig
        camera = {}
        for key, value in flat.items():
            parts = key.split(".")[2:]  # eye, x
            node = camera
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = value
    fig.update_layout(scene_camera=camera)
    return fig


def _pad(lo: float, hi: float, frac: float = 0.08) -> tuple[float, float]:
    """A pinned range with headroom, so a flat trace does not fill the panel.

    A channel sitting at machine epsilon autoscales into a dramatic-looking
    oscillation. Anything whose correct value is ~0 must be given a real range.
    """
    span = max(hi - lo, 1e-12)
    return lo - frac * span, hi + frac * span


# ---------------------------------------------------------------------------
# P1 -- the causal strip stack
# ---------------------------------------------------------------------------

# The row spec, module level so a test can assert the unit rule structurally.
#
# ONE UNIT PER ROW. Two measures of different scale sharing an axis is the most
# common charting mistake there is: the alignment between the two scales is
# arbitrary, so the panel invents a correlation the data does not contain. The
# first version of this module had `w_up` twinned with `q_gust` and `theta`
# twinned with `q`, which is exactly that.
#
# q_gust and q now share a row and that is NOT the same thing -- both are deg/s
# on ONE scale, and putting them together is the point: the rotational gust going
# in and the body rate coming out are directly comparable, which is the whole
# causal claim of the panel.
STRIP_ROWS = [
    ("vertical gust", "m/s", [("gust, up positive", "w_up", SERIES[0])], None),
    ("pitch rates", "deg/s", [("q gust (sim truth)", "q_gust_deg", SERIES[1]),
                              ("q, body rate", "q_deg", SERIES[0])], None),
    ("incidence", "deg", [("air-relative", "alpha_deg", SERIES[0]),
                          ("inertial (wrong)", "alpha_inertial_deg", REFERENCE)],
     "alpha"),
    ("load factor", "g", [("n_z", "n_z", SERIES[0])], "nz"),
    ("pitch attitude", "deg", [("theta", "theta_deg", SERIES[0])], None),
    ("elevator", "deg", [("elevator", "elevator_deg", SERIES[0])], "elevator"),
]


def strip_stack(s: Series, window: np.ndarray, cursor_t: float | None = None,
                alpha_bands: bool = True) -> go.Figure:
    """Gust in at the top, response propagating down, control at the bottom.

    The order is the causal chain and it is the point of the panel: a response
    that leads its input is visible immediately. `elevator` at the bottom must be
    FLAT for any turbulence encounter -- fixed controls are a physical statement,
    since the discriminator separates turbulence from manoeuvring by whether pitch
    correlates with the stick.
    """
    from plotly.subplots import make_subplots

    rows = STRIP_ROWS
    fig = make_subplots(rows=len(rows), cols=1, shared_xaxes=True,
                        vertical_spacing=0.022)

    for i, (label, unit, traces, special) in enumerate(rows, start=1):
        multi = len(traces) > 1
        for name, attr, colour in traces:
            xs, ys = envelope(s.t, np.asarray(getattr(s, attr)), STRIP_POINTS)
            fig.add_trace(
                go.Scattergl(
                    x=xs, y=ys, mode="lines", name=name,
                    line=dict(color=colour, width=2, shape="linear"),
                    # A legend whenever a row carries two series: identity is
                    # never colour-alone. A single-series row needs none -- its
                    # own axis title names it.
                    showlegend=multi, legendgroup=name,
                    hovertemplate=f"{name}: %{{y:.4f}} {unit}"
                                  "<br>t %{x:.2f} s<extra></extra>",
                ), row=i, col=1,
            )
        fig.update_yaxes(title_text=f"{label}<br>{unit}", row=i, col=1, **_AXIS)

        if special == "alpha" and alpha_bands:
            # The DECLARED linear-aero band, PROJECT.md section 7. Symmetric,
            # because aero.py is odd-symmetric in alpha -- a pushdown is exactly
            # as far out of range as an equal pull-up, which is what latent bug
            # (e)'s one-sided gauge could not see.
            for lo, hi, colour in ((-10.0, 10.0, "rgba(12,163,12,0.08)"),
                                   (10.0, 12.0, "rgba(250,178,25,0.16)"),
                                   (-12.0, -10.0, "rgba(250,178,25,0.16)")):
                fig.add_hrect(y0=lo, y1=hi, line_width=0, fillcolor=colour,
                              row=i, col=1, layer="below")
            reach = max(13.0, float(np.abs(s.alpha_deg).max()) * 1.1)
            fig.update_yaxes(range=[-reach, reach], row=i, col=1)
        elif special == "nz":
            # Pinned to include 1 g. Autoscaled to the excursion, a 0.03 g trim
            # offset fills the panel and reads as a defect.
            fig.add_hline(y=float(s.n_z[0]), line=dict(color=AXIS_RULE, width=1,
                                                       dash="dot"), row=i, col=1)
            fig.update_yaxes(
                range=list(_pad(min(float(s.n_z.min()), 0.9),
                                max(float(s.n_z.max()), 1.05))), row=i, col=1)
        elif special == "elevator":
            # Pinned wide, so a genuinely fixed control is visibly flat rather
            # than autoscaled into noise.
            fig.update_yaxes(range=[-5.0, 5.0], row=i, col=1)

    if window is not None and np.asarray(window).any():
        wt = s.t[np.asarray(window, dtype=bool)]
        fig.add_vrect(x0=float(wt.min()), x1=float(wt.max()), line_width=0,
                      fillcolor="rgba(235,104,52,0.10)", layer="below",
                      annotation_text="analysis window", annotation_position="top left",
                      annotation_font=dict(size=9, color=INK_MUTED))
    if cursor_t is not None:
        fig.add_vline(x=float(cursor_t), line=dict(color=CRITICAL, width=1.5))

    fig.update_xaxes(**_AXIS)
    fig.update_xaxes(title_text="time  s", row=len(rows), col=1, **_AXIS)
    return _base(
        fig, height=620,
        title="Encounter traces — the causal chain, top to bottom",
        subtitle="gust in → incidence → load → attitude → control. "
                 "Elevator must be FLAT on a turbulence run; a response that "
                 "leads its input is a defect.",
        legend=True,
    )


# ---------------------------------------------------------------------------
# P3 -- load factor against incidence
# ---------------------------------------------------------------------------

def load_vs_alpha(s: Series, cursor_index: int | None = None) -> go.Figure:
    """This panel is an assertion, not a display.

    The air-relative points must fall on a STRAIGHT LINE and the inertial ones
    must scatter -- measured correlations 0.9990 against 0.5572. A scattered
    air-relative cloud means the wind used for sensing is not the wind that was
    flown, which is the shape of latent bugs (a) and (b).

    The straightness is also the honest picture of a permanent gap: a real
    aircraft's curve bends at buffet onset and `CL = CL0 + CLa*alpha` cannot,
    which is why no +-g asymmetry can come out of this model.
    """
    r_air = float(np.corrcoef(s.n_z, s.alpha_deg)[0, 1])
    r_inertial = float(np.corrcoef(s.n_z, s.alpha_inertial_deg)[0, 1])

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=s.alpha_inertial_deg, y=s.n_z, mode="markers",
        name=f"inertial — scatters (r={r_inertial:.3f})",
        marker=dict(size=4, color=REFERENCE, opacity=0.55),
        hovertemplate="inertial alpha %{x:.2f} deg<extra></extra>",
    ))
    fig.add_trace(go.Scattergl(
        x=s.alpha_deg, y=s.n_z, mode="markers",
        name=f"air-relative — a line (r={r_air:.4f})",
        marker=dict(size=4.5, color=s.t, colorscale=SEQUENTIAL,
                    colorbar=dict(title=dict(text="time  s", side="right",
                                             font=dict(size=10)),
                                  thickness=11, len=0.85, outlinewidth=0,
                                  tickfont=dict(size=9))),
        hovertemplate="air-relative alpha %{x:.2f} deg<br>n_z %{y:.3f} g<extra></extra>",
    ))
    if cursor_index is not None:
        fig.add_trace(go.Scatter(
            x=[s.alpha_deg[cursor_index]], y=[s.n_z[cursor_index]], mode="markers",
            marker=dict(size=13, color="rgba(0,0,0,0)",
                        line=dict(color=CRITICAL, width=2.5)),
            name="cursor", showlegend=False, hoverinfo="skip",
        ))
    fig.update_xaxes(title_text="angle of attack  deg", **_AXIS)
    fig.update_yaxes(title_text="load factor n_z  g", **_AXIS)
    return _base(
        fig, 340,
        title="Load factor against incidence — an assertion, not a display",
        subtitle="Air-relative points MUST fall on a straight line; the "
                 "ground-relative ones must scatter. A scattered blue cloud "
                 "means the wind is not reaching the sensing path.",
        legend=True,
    )


# ---------------------------------------------------------------------------
# P4 -- the Wingrove & Bach discriminator
# ---------------------------------------------------------------------------

FIG8_REFERENCE = {"vortex": 1.4, "updraft": 6.2, "manoeuvring": 12.0}
FIG8_LOAD_BAND = (-2.01, -1.69)


def discriminator(points: list[dict]) -> go.Figure:
    """Fig. 8, with the whole-run marker and the connector that shames it.

    Each entry: {label, dtheta, dn, dtheta_whole, dn_whole}. The hollow marker is
    the same run measured over its whole length, and the dotted line between them
    is the windowing trap drawn rather than described -- the vortex migrates from
    2.24 to 8.33 deg between the two windows while the manoeuvre moves 30.37 to
    30.74, and that asymmetry is why the window rule had to exist.

    THE X-AXIS SCALES TO THE DATA, NOT TO THE PAPER. PROJECT.md section 5 forbids
    reading this as agreement -- Wingrove & Bach never identifies an aircraft type
    -- and a chart cropped to the reference range would hide the model's
    manoeuvring point sitting 2.5x to the right of the paper's.

    AND IT SAYS SO WHEN A CATEGORY IS MISSING. The claim here is an ORDERING
    across three categories, so a chart with one or two points on it demonstrates
    nothing while looking perfectly finished. `vortex_viz._discriminator_panel`
    records the same guard in its matplotlib form -- it drew the empty
    manoeuvring slot as a labelled marker "so the figure could not read as
    complete while it was not" -- and that annotation was only removed once the
    third point existed.
    """
    lo, hi = FIG8_LOAD_BAND
    fig = go.Figure()
    fig.add_hrect(y0=lo, y1=hi, line_width=0, fillcolor="rgba(137,135,129,0.14)",
                  layer="below", annotation_text="paper's load band",
                  annotation_position="bottom right",
                  annotation_font=dict(size=9, color=INK_MUTED))

    for name, dtheta in FIG8_REFERENCE.items():
        fig.add_trace(go.Scatter(
            x=[dtheta], y=[0.5 * (lo + hi)], mode="markers+text", text=[name],
            textposition="bottom center",
            textfont=dict(size=9, color=INK_MUTED),
            marker=dict(size=11, symbol="square-open",
                        line=dict(color=REFERENCE, width=1.5)),
            name="paper (DC-10 class)", showlegend=name == "vortex",
            legendgroup="paper",
            hovertemplate=f"paper: {name}, {dtheta} deg<extra></extra>",
        ))

    spread = list(FIG8_REFERENCE.values())
    for i, p in enumerate(points):
        colour = SERIES[i % len(SERIES)]
        fig.add_trace(go.Scatter(
            x=[p["dtheta"], p["dtheta_whole"]], y=[p["dn"], p["dn_whole"]],
            mode="lines", line=dict(color=colour, width=1.2, dash="dot"),
            showlegend=False, hoverinfo="skip",
        ))
        # Text wears an INK token, never the series colour: the mark beside it
        # carries identity. Aqua at 2.74:1 on this surface is illegible as text,
        # and it is also why every point is direct-labelled -- the relief rule
        # for a sub-3:1 categorical hue.
        fig.add_trace(go.Scatter(
            x=[p["dtheta"]], y=[p["dn"]], mode="markers+text",
            text=[f"  {p['label']}  {p['dtheta']:.2f}°, {p['dn']:+.2f} g"],
            textposition="middle right", textfont=dict(size=10, color=INK),
            marker=dict(size=11, color=colour,
                        line=dict(color=SURFACE, width=2)),  # surface ring
            name=f"{p['label']} (windowed)", legendgroup=p["label"],
            hovertemplate=f"{p['label']} (windowed)<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=[p["dtheta_whole"]], y=[p["dn_whole"]], mode="markers",
            marker=dict(size=11, symbol="circle-open",
                        line=dict(color=colour, width=2)),
            name=f"{p['label']} (whole run)", legendgroup=p["label"],
            showlegend=False,
            hovertemplate=f"{p['label']} WHOLE RUN — wrong window<extra></extra>",
        ))
        spread += [p["dtheta"], p["dtheta_whole"]]

    shown = {p["label"].split()[0].lower() for p in points}
    missing = [c for c in ("vortex", "updraft", "manoeuvr")
               if not any(c in s for s in shown)]
    if missing:
        fig.add_annotation(
            xref="paper", yref="paper", x=0.5, y=1.0, showarrow=False,
            text=("INCOMPLETE — the claim is an <b>ordering</b> across three "
                  f"categories; missing: {', '.join(missing)}"),
            font=dict(size=10, color="#b42318"),
            bgcolor="rgba(180,35,24,0.07)", borderpad=3,
        )

    fig.update_xaxes(title_text="pitch attitude excursion in the window  deg",
                     range=[0, max(spread) * 1.32], **_AXIS)
    fig.update_yaxes(title_text="load excursion from trim  g", **_AXIS)
    return _base(
        fig, 360,
        title="Wingrove &amp; Bach Fig. 8 — does the ordering hold?",
        subtitle="The claim is the ORDERING vortex &lt; updraft &lt; manoeuvre, "
                 "never the values (§5: the paper never states an aircraft "
                 "type). Hollow = same run over its whole length.",
        legend=True,
    )


# ---------------------------------------------------------------------------
# P7a -- the field cross-section. THE PRIMARY FIELD VIEW.
# ---------------------------------------------------------------------------

def field_cross_section(s: Series, field, *, scale: float, peak: float,
                        cores: list[tuple[float, float]] | None = None,
                        window: np.ndarray | None = None,
                        cursor_index: int | None = None,
                        field_centre: float | None = None,
                        label: str = "field") -> go.Figure:
    """The vertical gust field with the flight path through it. 2D, on purpose.

    THIS REPLACES THE 3D SCENE AS THE PRIMARY VIEW FOR A PARKS VORTEX, and the
    reason is physics rather than taste. `wind.vortex_wind` fixes dpsi = 0, so
    the cores are infinite east-west lines and the field has NO east variation
    at all -- `panel.py` reports a range to them with no bearing for the same
    reason. A single north-altitude plane therefore contains the entire field,
    and rendering it as a volume adds a dimension the data does not have. The
    design document's own rule for the lee wave says it outright: rendering a
    field as a volume it has no structure in MANUFACTURES structure.

    It is also 15x cheaper -- about 0.1 MB against the isosurface's 1.5 MB --
    and it is legible on first sight, which the 3D scene measurably was not.

    This is `vortex_viz._field_panel` in Plotly, and deliberately so: that panel
    is the project's established rendering of this field and it was already
    right. Four properties carried across unchanged:

      * colour is DIVERGING and pinned to +-V0, never autoscaled, because the
        claim under test is a signed up-then-down doublet and a sign error must
        reverse the colour order visibly;
      * the zero contour is drawn, because superposition DISPLACES it between
        cores -- it is the visible check that the array is summed rather than
        repeated;
      * the core circles are drawn from r0 INDEPENDENTLY of the field, so a
        circle that does not sit on the colour transition is a transcription
        error;
      * both axes are metres with a 1:1 aspect, so a wrong core radius shows up
        as an ellipse.
    """
    import jax
    import jax.numpy as jnp

    cores = cores or []
    centre = (float(field_centre) if field_centre is not None
              else (cores[0][0] + cores[-1][0]) / 2.0 if cores
              else 0.5 * float(s.north.max() + s.north.min()))
    mid_alt = float(np.median(s.altitude))

    first = cores[0][0] if cores else centre - 3.0 * scale
    last = cores[-1][0] if cores else centre + 3.0 * scale
    margin = 3.0 * scale
    north = np.linspace(first - margin, last + margin, 300)
    alt = np.linspace(mid_alt - 2.2 * scale, mid_alt + 2.2 * scale, 150)
    mesh_n, mesh_a = np.meshgrid(north, alt, indexing="xy")
    pts = jnp.stack([jnp.asarray(mesh_n.ravel()), jnp.zeros(mesh_n.size),
                     jnp.asarray(-mesh_a.ravel())], axis=1)
    w_up = -np.asarray(jax.jit(jax.vmap(field))(pts))[:, 2].reshape(mesh_n.shape)

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=north, y=alt, z=w_up, colorscale=DIVERGING,
        zmin=-peak, zmax=peak, zmid=0.0,
        colorbar=dict(title=dict(text="vertical gust<br>up positive  m/s",
                                 side="right", font=dict(size=10)),
                      thickness=11, len=0.9, outlinewidth=0,
                      tickfont=dict(size=9)),
        hovertemplate="north %{x:.0f} m<br>alt %{y:.0f} m"
                      "<br>gust %{z:+.2f} m/s<extra></extra>",
    ))
    fig.add_trace(go.Contour(
        x=north, y=alt, z=w_up, contours=dict(start=0.0, end=0.0, size=1.0,
                                              coloring="none"),
        line=dict(color=INK, width=1), showscale=False, hoverinfo="skip",
        name="zero gust",
    ))

    for cn, ca in cores:
        fig.add_shape(type="circle", x0=cn - scale, x1=cn + scale,
                      y0=ca - scale, y1=ca + scale,
                      line=dict(color=INK, width=1.2, dash="dash"))
        fig.add_trace(go.Scatter(
            x=[cn], y=[ca], mode="markers",
            marker=dict(symbol="cross-thin", size=9,
                        line=dict(color=INK, width=1.4)),
            hovertemplate=f"core: north {cn:.0f} m, r0 {scale:.1f} m"
                          "<extra></extra>", showlegend=False,
        ))

    fig.add_trace(go.Scattergl(
        x=s.north, y=s.altitude, mode="lines",
        line=dict(color=INK, width=2.5, shape="linear"),
        name="flight path",
        hovertemplate="north %{x:.0f} m<br>alt %{y:.0f} m<extra></extra>",
    ))
    if window is not None and np.asarray(window).any():
        m = np.asarray(window, dtype=bool)
        fig.add_trace(go.Scattergl(
            x=s.north[m], y=s.altitude[m], mode="lines",
            line=dict(color=SERIES[1], width=4, shape="linear"),
            name="analysis window", hoverinfo="skip",
        ))
    if cursor_index is not None:
        fig.add_trace(go.Scatter(
            x=[s.north[cursor_index]], y=[s.altitude[cursor_index]],
            mode="markers",
            marker=dict(size=11, color=CRITICAL,
                        line=dict(color=SURFACE, width=2)),  # surface ring
            name="cursor", hoverinfo="skip",
        ))

    # 1:1 IN DATA SPACE, and the axis shrinks its BOX to get there rather than
    # widening its RANGE. `constrain="domain"` is the whole trick: with an
    # explicit `range` on both axes and no constraint, plotly satisfies the
    # aspect lock by collapsing the plot area -- measured 369 x 29 px, a
    # twenty-nine pixel tall field map that still looked like a figure in every
    # automated check. Caught by rendering it and reading the geometry back.
    fig.update_xaxes(title_text="along track (north)  m",
                     range=[north[0], north[-1]], constrain="domain", **_AXIS)
    fig.update_yaxes(title_text="altitude  m", range=[alt[0], alt[-1]],
                     scaleanchor="x", scaleratio=1.0, constrain="domain",
                     **_AXIS)
    return _base(
        fig, 360,
        title=f"{label}: vertical gust field and flight path",
        subtitle=(
            "Where the aircraft actually was. Colour is pinned to ±peak, never "
            "autoscaled, so a sign error reverses it visibly. Black line is "
            "zero gust." + (
                " Dashed circles are r₀ drawn from the source, not from the "
                "field, so a circle off the colour transition is a "
                "transcription error." if cores else "")),
        legend=True,
    )


# ---------------------------------------------------------------------------
# P7b -- the 3D scene, for fields that genuinely have three-dimensional structure
# ---------------------------------------------------------------------------

SCALARS = {
    "n_z": ("n_z  g", SEQUENTIAL, False),
    "alpha_deg": ("alpha  deg", SEQUENTIAL, False),
    "w_up": ("gust up  m/s", DIVERGING, True),  # SIGNED -> diverging, pinned
    "altitude": ("altitude  m", SEQUENTIAL, False),
}


def field_3d(s: Series, field, *, scalar: str = "n_z", cursor_index: int | None = None,
             representation: str = "isosurface", scale: float | None = None,
             peak: float | None = None,
             field_centre: float | None = None) -> go.Figure:
    """Trajectory through the field, coloured by a selectable scalar.

    ONE TREATMENT DOES NOT FIT FOUR FIELDS, so `representation` is explicit:

      isosurface  a Rankine vortex, where vorticity is exactly 0 outside the core
                  and exactly 2*V0/r0 inside. The core boundary IS the physical
                  claim and an isosurface draws exactly it.
      streamtube  an updraft column or a microburst, which have no sharp
                  structure but do have a topology -- only streamlines show a
                  stagnation flow.
      slice       a lee wave, which is one-dimensional and periodic. Rendering it
                  as a volume would MANUFACTURE structure it does not have.

    `field_centre` is the along-track position the field grid is drawn AROUND,
    and it must come from the FIELD, not from the trajectory. The canonical
    vortex run leads in for 40 core radii, so its trajectory midpoint sits about
    2.5 km upstream of both cores -- a grid centred there samples only the
    irrotational far field, where a Rankine vortex has exactly zero vorticity,
    and the isosurface comes out empty while the panel still looks normal.
    `vortex_viz._field_panel` states the rule: zoom to the structure, not the run.

    Axes are all metres with `aspectmode="data"`: a wrong core radius is only
    visible if the cores are drawn round.
    """
    label, colorscale, signed = SCALARS.get(scalar, SCALARS["n_z"])
    values = np.asarray(getattr(s, scalar))
    kwargs = {}
    if signed:
        # Diverging maps ONLY for genuinely signed quantities, pinned symmetric
        # about zero so a sign error reverses the colour order visibly.
        reach = float(np.abs(values).max()) or 1.0
        kwargs = dict(cmin=-reach, cmax=reach, cmid=0.0)

    # A 3D line is one trace, so a stride is safe here in a way it is not on the
    # 2D strips: the panel shows WHERE the aircraft was, and the scalar's peak is
    # read off `strip_stack` and the readout, not off the tube's colour.
    keep = np.linspace(0, len(s.t) - 1, min(len(s.t), 4000)).astype(int)

    fig = go.Figure()
    _add_field(fig, field, s, representation, scale, peak,
               field_centre)
    fig.add_trace(go.Scatter3d(
        x=s.east[keep], y=s.north[keep], z=s.altitude[keep], mode="lines",
        line=dict(width=6, color=values[keep], colorscale=colorscale,
                  colorbar=dict(title=dict(text=label, side="right"),
                                thickness=10, len=0.7, x=1.02),
                  **kwargs),
        name="flight path",
        hovertemplate="north %{y:.0f} m<br>alt %{z:.0f} m<extra></extra>",
    ))
    if cursor_index is not None:
        fig.add_trace(go.Scatter3d(
            x=[s.east[cursor_index]], y=[s.north[cursor_index]],
            z=[s.altitude[cursor_index]], mode="markers",
            marker=dict(size=7, color=CRITICAL), name="cursor",
            hovertemplate="cursor<extra></extra>",
        ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="east  m"), yaxis=dict(title="north  m"),
            zaxis=dict(title="altitude  m"),
            aspectmode="data",
        ),
    )
    return _base(fig, 460)


def _add_field(fig, field, s: Series, representation, scale, peak,
               field_centre=None):
    """Draw the wind field itself, in whichever form is legible for it."""
    import jax
    import jax.numpy as jnp

    span_n = float(s.north.max() - s.north.min())
    # The STRUCTURE's position, falling back to the run's midpoint only when the
    # field spec did not say. See `field_3d`'s docstring: for a long lead-in the
    # midpoint is nowhere near the structure.
    mid_n = float(field_centre) if field_centre is not None \
        else 0.5 * float(s.north.max() + s.north.min())
    mid_alt = float(np.median(s.altitude))
    reach = scale * 3.0 if scale else 0.35 * span_n

    if representation == "isosurface" and scale:
        g = ISO_GRID
        a = np.linspace(mid_n - 4.0 * scale, mid_n + 4.0 * scale, g)
        e = np.linspace(-reach, reach, ISO_GRID_LATERAL)
        d = np.linspace(-(mid_alt + 2.5 * scale), -(mid_alt - 2.5 * scale), g)
        ga, ge, gd = np.meshgrid(a, e, d, indexing="ij")
        pts = jnp.stack([ga.ravel(), ge.ravel(), gd.ravel()], axis=1)

        def vort(p):
            j = jax.jacfwd(field)(p)
            return jnp.linalg.norm(jnp.array(
                [j[2, 1] - j[1, 2], j[0, 2] - j[2, 0], j[1, 0] - j[0, 1]]))

        mag = np.asarray(jax.jit(jax.vmap(vort))(pts))
        peak = float(mag.max())
        if peak > 0:
            fig.add_trace(go.Isosurface(
                x=ge.ravel(), y=ga.ravel(), z=-gd.ravel(), value=mag,
                isomin=0.9 * peak, isomax=peak, surface_count=1,
                opacity=0.22, colorscale=SEQUENTIAL, showscale=False,
                caps=dict(x_show=False, y_show=False, z_show=False),
                name="core (|curl| = 2 V0/r0)", hoverinfo="skip",
            ))
    elif representation == "streamtube":
        g = 22
        a = np.linspace(mid_n - reach, mid_n + reach, g)
        e = np.linspace(-reach, reach, 5)
        d = np.linspace(-(mid_alt + reach * 0.4), -(mid_alt - reach * 0.4), g)
        ga, ge, gd = np.meshgrid(a, e, d, indexing="ij")
        pts = jnp.stack([ga.ravel(), ge.ravel(), gd.ravel()], axis=1)
        w = np.asarray(jax.jit(jax.vmap(field))(pts))
        seeds = np.linspace(-reach * 0.7, reach * 0.7, 24)
        fig.add_trace(go.Streamtube(
            x=ge.ravel(), y=ga.ravel(), z=-gd.ravel(),
            u=w[:, 1], v=w[:, 0], w=-w[:, 2],
            starts=dict(x=seeds, y=np.full(24, mid_n - reach),
                        z=np.full(24, mid_alt)),
            sizeref=0.4, opacity=0.35, colorscale=SEQUENTIAL, showscale=False,
            hoverinfo="skip",
        ))
    elif representation == "slice":
        g = 60
        a = np.linspace(mid_n - reach, mid_n + reach, g)
        d = np.linspace(-(mid_alt + reach * 0.3), -(mid_alt - reach * 0.3), g)
        ga, gd = np.meshgrid(a, d, indexing="ij")
        pts = jnp.stack([ga.ravel(), np.zeros(ga.size), gd.ravel()], axis=1)
        w_up = -np.asarray(jax.jit(jax.vmap(field))(pts))[:, 2].reshape(ga.shape)
        cap = float(np.abs(w_up).max()) or 1.0
        fig.add_trace(go.Surface(
            x=np.zeros_like(ga), y=ga, z=-gd, surfacecolor=w_up,
            colorscale=DIVERGING, cmin=-cap, cmax=cap, opacity=0.55,
            showscale=False, hoverinfo="skip",
        ))


# ---------------------------------------------------------------------------
# P9 -- energy closure residual against position
# ---------------------------------------------------------------------------

def energy_residual(profile, core_edges: list[float] | None = None) -> go.Figure:
    """Per-step closure residual against along-track position, log y.

    The sharpest positional diagnostic the project has. On a Rankine field the
    core boundaries -- where the one-sided derivatives differ by 2*V0/r0 with
    opposite signs -- read 20 to 350 times the run median. Log y because the
    dynamic range is three orders and a linear axis shows one spike and a flat
    line.
    """
    xs, ys = envelope(profile.north, profile.per_step, STRIP_POINTS)
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=xs / 1000.0, y=np.maximum(ys, 1e-30), mode="lines",
        line=dict(color=SERIES[0], width=2, shape="linear"),
        hovertemplate="north %{x:.3f} km<br>%{y:.3e} J<extra></extra>",
    ))
    fig.add_hline(y=max(profile.median, 1e-30),
                  line=dict(color=AXIS_RULE, width=1, dash="dot"),
                  annotation_text="run median", annotation_font_size=9)
    for edge in core_edges or []:
        fig.add_vline(x=edge / 1000.0,
                      line=dict(color=CRITICAL, width=1, dash="dash"))
    fig.update_xaxes(title_text="along track  km", **_AXIS)
    fig.update_yaxes(title_text="per-step |Δresidual|  J", type="log", **_AXIS)
    return _base(
        fig, 320,
        title="Energy closure residual — where the field is not smooth",
        subtitle="Per-step |Δresidual| against position, log axis. Spikes at "
                 "the core boundary are the Rankine gradient discontinuity, "
                 "20–350× the run median.")


# ---------------------------------------------------------------------------
# The comparison panel
# ---------------------------------------------------------------------------

def difference(a: Series, b: Series, channel: str, labels: tuple[str, str]) -> go.Figure:
    """Two runs compared as a DIFFERENCE, never as two overlaid absolutes.

    Measured: h and h/2 differ by 0.402 m over a 9.5 km track, 4e-5 relative.
    Overlaid on a shared axis they are one line. The difference is the only
    presentation in which the comparison is visible at all.

    Interpolated onto the coarser time base, linearly, and the panel says so --
    for a fourth-order trajectory that is a first-order comparison and could
    dominate what is being measured.
    """
    ya, yb = np.asarray(getattr(a, channel)), np.asarray(getattr(b, channel))
    coarse, fine = (a, b) if len(a.t) <= len(b.t) else (b, a)
    coarse_y = ya if len(a.t) <= len(b.t) else yb
    fine_y = yb if len(a.t) <= len(b.t) else ya
    on_coarse = np.interp(coarse.t, fine.t, fine_y)
    delta = coarse_y - on_coarse

    xs, ys = envelope(coarse.t, delta, STRIP_POINTS)
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=xs, y=ys, mode="lines",
        line=dict(color=SERIES[2], width=2, shape="linear"),
        hovertemplate="t %{x:.2f} s<br>Δ %{y:.4e}<extra></extra>",
    ))
    fig.add_hline(y=0.0, line=dict(color=AXIS_RULE, width=1))
    fig.update_xaxes(title_text="time  s", **_AXIS)
    fig.update_yaxes(title_text=f"Δ {channel}", **_AXIS)
    return _base(
        fig, 320,
        title=f"Difference: {labels[0]} − {labels[1]}",
        subtitle="Plotted as a difference, never as two overlaid absolutes — "
                 "h and h/2 differ by 4e-5 relative and are one line when "
                 "overlaid. Linearly interpolated onto the coarser grid.")
