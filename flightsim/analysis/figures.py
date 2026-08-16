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

_AXIS = dict(showgrid=True, gridcolor="rgba(128,128,128,0.25)", zeroline=False)


def _base(fig: go.Figure, height: int, title: str | None = None) -> go.Figure:
    fig.update_layout(
        uirevision=UIREVISION,
        height=height,
        margin=dict(l=64, r=16, t=30 if title else 8, b=36),
        title=dict(text=title, font=dict(size=13)) if title else None,
        template="plotly_white",
        showlegend=False,
        font=dict(size=11),
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

    rows = [
        ("gust up<br>m/s", [("w_up", s.w_up, "#1f77b4")], None),
        ("q gust<br>deg/s", [("q_gust", s.q_gust_deg, "#d62728")], None),
        ("alpha<br>deg", [("air-relative", s.alpha_deg, "#1f77b4"),
                          ("inertial", s.alpha_inertial_deg, "#bbbbbb")], "alpha"),
        ("n_z<br>g", [("n_z", s.n_z, "#1f77b4")], "nz"),
        ("theta<br>deg", [("theta", s.theta_deg, "#1f77b4"),
                          ("q", s.q_deg, "#d62728")], None),
        ("elevator<br>deg", [("elevator", s.elevator_deg, "#1f77b4")], "elevator"),
    ]
    fig = make_subplots(rows=len(rows), cols=1, shared_xaxes=True,
                        vertical_spacing=0.018)

    for i, (label, traces, special) in enumerate(rows, start=1):
        for name, y, colour in traces:
            xs, ys = envelope(s.t, np.asarray(y), STRIP_POINTS)
            fig.add_trace(
                go.Scattergl(
                    x=xs, y=ys, mode="lines", name=name,
                    line=dict(color=colour, width=1.4, shape="linear"),
                    hovertemplate=f"{name}: %{{y:.4f}}<br>t %{{x:.2f}} s<extra></extra>",
                ), row=i, col=1,
            )
        fig.update_yaxes(title_text=label, title_font=dict(size=10), row=i, col=1,
                         **_AXIS)

        if special == "alpha" and alpha_bands:
            # The DECLARED linear-aero band, PROJECT.md section 7. Symmetric,
            # because aero.py is odd-symmetric in alpha -- a pushdown is exactly
            # as far out of range as an equal pull-up, which is what latent bug
            # (e)'s one-sided gauge could not see.
            for lo, hi, colour in ((-10.0, 10.0, "rgba(44,160,44,0.10)"),
                                   (10.0, 12.0, "rgba(255,187,0,0.16)"),
                                   (-12.0, -10.0, "rgba(255,187,0,0.16)")):
                fig.add_hrect(y0=lo, y1=hi, line_width=0, fillcolor=colour,
                              row=i, col=1, layer="below")
            reach = max(13.0, float(np.abs(s.alpha_deg).max()) * 1.1)
            fig.update_yaxes(range=[-reach, reach], row=i, col=1)
        elif special == "nz":
            # Pinned to include 1 g. Autoscaled to the excursion, a 0.03 g trim
            # offset fills the panel and reads as a defect.
            fig.add_hline(y=float(s.n_z[0]), line=dict(color="#888", width=1,
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
                      fillcolor="rgba(255,127,14,0.12)", layer="below")
    if cursor_t is not None:
        fig.add_vline(x=float(cursor_t), line=dict(color="#d62728", width=1.2))

    fig.update_xaxes(title_text="time  s", row=len(rows), col=1, **_AXIS)
    fig.update_xaxes(**_AXIS)
    return _base(fig, height=560)


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
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=s.alpha_inertial_deg, y=s.n_z, mode="markers", name="inertial",
        marker=dict(size=3, color="#cccccc"),
        hovertemplate="inertial alpha %{x:.2f} deg<extra></extra>",
    ))
    fig.add_trace(go.Scattergl(
        x=s.alpha_deg, y=s.n_z, mode="markers", name="air-relative",
        marker=dict(size=3.5, color=s.t, colorscale="Viridis",
                    colorbar=dict(title=dict(text="t  s", side="right"),
                                  thickness=10, len=0.85)),
        hovertemplate="air-relative alpha %{x:.2f} deg<br>n_z %{y:.3f} g<extra></extra>",
    ))
    if cursor_index is not None:
        fig.add_trace(go.Scattergl(
            x=[s.alpha_deg[cursor_index]], y=[s.n_z[cursor_index]], mode="markers",
            marker=dict(size=12, color="rgba(0,0,0,0)",
                        line=dict(color="#d62728", width=2)),
            hoverinfo="skip",
        ))
    fig.update_xaxes(title_text="angle of attack  deg", **_AXIS)
    fig.update_yaxes(title_text="load factor  g", **_AXIS)
    return _base(fig, 300, "n_z vs incidence — air-relative must be a line")


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
    """
    lo, hi = FIG8_LOAD_BAND
    fig = go.Figure()
    fig.add_hrect(y0=lo, y1=hi, line_width=0, fillcolor="rgba(0,0,0,0.07)",
                  layer="below")

    for name, dtheta in FIG8_REFERENCE.items():
        fig.add_trace(go.Scatter(
            x=[dtheta], y=[0.5 * (lo + hi)], mode="markers+text", text=[name],
            textposition="top center", textfont=dict(size=9, color="#777"),
            marker=dict(size=11, symbol="square-open", color="#777"),
            hovertemplate=f"paper: {name}, {dtheta} deg<extra></extra>",
        ))

    spread = list(FIG8_REFERENCE.values())
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
    for i, p in enumerate(points):
        colour = palette[i % len(palette)]
        fig.add_trace(go.Scatter(
            x=[p["dtheta"], p["dtheta_whole"]], y=[p["dn"], p["dn_whole"]],
            mode="lines", line=dict(color=colour, width=1, dash="dot"),
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=[p["dtheta"]], y=[p["dn"]], mode="markers+text",
            text=[f"  {p['dtheta']:.2f}°, {p['dn']:+.2f} g"],
            textposition="middle right", textfont=dict(size=9, color=colour),
            marker=dict(size=10, color=colour), name=p["label"],
            hovertemplate=f"{p['label']} (windowed)<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=[p["dtheta_whole"]], y=[p["dn_whole"]], mode="markers",
            marker=dict(size=10, symbol="circle-open",
                        line=dict(color=colour, width=2)),
            hovertemplate=f"{p['label']} WHOLE RUN — wrong window<extra></extra>",
        ))
        spread += [p["dtheta"], p["dtheta_whole"]]

    fig.update_xaxes(title_text="pitch attitude excursion  deg",
                     range=[0, max(spread) * 1.22], **_AXIS)
    fig.update_yaxes(title_text="load excursion from trim  g", **_AXIS)
    return _base(fig, 320,
                 "Wingrove & Bach Fig. 8 — grey = paper (DC-10 class), "
                 "hollow = whole run (wrong window)")


# ---------------------------------------------------------------------------
# P7 -- the 3D scene
# ---------------------------------------------------------------------------

SCALARS = {
    "n_z": ("n_z  g", "Viridis", False),
    "alpha_deg": ("alpha  deg", "Viridis", False),
    "w_up": ("gust up  m/s", "RdBu_r", True),  # SIGNED -> diverging, pinned
    "altitude": ("altitude  m", "Viridis", False),
}


def field_3d(s: Series, field, *, scalar: str = "n_z", cursor_index: int | None = None,
             representation: str = "isosurface", core_radius: float | None = None,
             peak_tangential: float | None = None,
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
    _add_field(fig, field, s, representation, core_radius, peak_tangential,
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
            marker=dict(size=6, color="#d62728"), name="cursor",
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


def _add_field(fig, field, s: Series, representation, core_radius, peak_tangential,
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
    reach = core_radius * 3.0 if core_radius else 0.35 * span_n

    if representation == "isosurface" and core_radius:
        g = ISO_GRID
        a = np.linspace(mid_n - 4.0 * core_radius, mid_n + 4.0 * core_radius, g)
        e = np.linspace(-reach, reach, ISO_GRID_LATERAL)
        d = np.linspace(-(mid_alt + 2.5 * core_radius), -(mid_alt - 2.5 * core_radius), g)
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
                opacity=0.22, colorscale="Blues", showscale=False,
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
            sizeref=0.4, opacity=0.35, colorscale="Blues", showscale=False,
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
            colorscale="RdBu_r", cmin=-cap, cmax=cap, opacity=0.55,
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
        line=dict(color="#1f77b4", width=1.2, shape="linear"),
        hovertemplate="north %{x:.3f} km<br>%{y:.3e} J<extra></extra>",
    ))
    fig.add_hline(y=max(profile.median, 1e-30),
                  line=dict(color="#888", width=1, dash="dot"),
                  annotation_text="run median", annotation_font_size=9)
    for edge in core_edges or []:
        fig.add_vline(x=edge / 1000.0,
                      line=dict(color="#d62728", width=1, dash="dash"))
    fig.update_xaxes(title_text="along track  km", **_AXIS)
    fig.update_yaxes(title_text="per-step |Δresidual|  J", type="log", **_AXIS)
    return _base(fig, 300,
                 "energy closure residual — spikes locate the core boundary")


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
        line=dict(color="#9467bd", width=1.4, shape="linear"),
        hovertemplate="t %{x:.2f} s<br>Δ %{y:.4e}<extra></extra>",
    ))
    fig.add_hline(y=0.0, line=dict(color="#888", width=1))
    fig.update_xaxes(title_text="time  s", **_AXIS)
    fig.update_yaxes(title_text=f"Δ {channel}", **_AXIS)
    return _base(fig, 300,
                 f"{labels[0]} − {labels[1]}, linearly interpolated onto the "
                 "coarser grid")
