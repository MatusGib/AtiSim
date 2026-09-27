"""Figures for the development workspaces: Diagnostics and Compare.

The encounter figures (`figures.py`) answer "what happened to the aircraft".
These answer "what inside the model made it happen": which coefficient term
carries a peak, where the energy closure breaks, how a check's quantity moved
through the run, and what changed between two runs.

Same chrome as `figures.py` (`_base`, the palette, `series.envelope` for every
long line), so the workspaces read as the same application.
"""

import numpy as np
import plotly.graph_objects as go

from atisim.analysis.figures import (_AXIS, AXIS_RULE, CRITICAL, INK, INK_MUTED, REFERENCE,
                                     SEQUENTIAL, SERIES, STRIP_POINTS, _base, _frame)
from atisim.analysis.series import envelope

# Ten distinguishable hues for coefficient terms: the three series colours
# first, then muted companions. A term keeps its colour between the budget and
# the legend, never colour alone -- every trace is named.
TERM_COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#8e5bd0", "#c9a227", "#d0457b",
                "#4aa3b5", "#7b7a72", "#5d8f3a", "#a55a3a")


def _window(t, xrange):
    if not xrange:
        return np.ones(len(t), dtype=bool)
    lo, hi = sorted(float(v) for v in xrange)
    keep = (t >= lo) & (t <= hi)
    return keep if keep.sum() >= 2 else np.ones(len(t), dtype=bool)


def extreme_indices(y: np.ndarray, target: int) -> np.ndarray:
    """Indices of each bucket's minimum and maximum of `y`, in order.

    `series.envelope` decimates one line; a STACK needs every layer sampled at
    the same instants, or the sums are taken at different times and the areas
    tear. So the instants come from the total, and each layer is read there.
    """
    n = len(y)
    if n <= 2 * target:
        return np.arange(n)
    edges = np.linspace(0, n, target + 1).astype(int)
    out = [0, n - 1]
    for a, b in zip(edges[:-1], edges[1:]):
        if b > a:
            seg = y[a:b]
            out += [a + int(np.argmin(seg)), a + int(np.argmax(seg))]
    return np.unique(out)


def _cursor(fig, cursor_t):
    """The cursor, one line per strip. Named, so the application can move it in
    the browser without drawing the figure again."""
    if cursor_t is not None:
        fig.add_vline(x=float(cursor_t), line=dict(color=CRITICAL, width=1.5), name="cursor")


def channel_strips(columns: dict, channels: list, names: list[str], cursor_t=None,
                   xrange=None) -> go.Figure:
    """One strip per chosen channel on a shared time axis.

    Full-rate data, decimated to a min/max envelope over the visible range
    only (`series.envelope`), so zooming in shows every sample again.
    """
    from plotly.subplots import make_subplots

    by_name = {c.name: c for c in channels}
    names = [n for n in names if n in columns and n != "t"] or ["n_z"]
    t = np.asarray(columns["t"])
    keep = _window(t, xrange)
    fig = make_subplots(rows=len(names), cols=1, shared_xaxes=True,
                        vertical_spacing=min(0.04, 0.3 / max(len(names), 1)))
    for i, name in enumerate(names, start=1):
        ch = by_name.get(name)
        unit = ch.unit if ch else ""
        xs, ys = envelope(t[keep], np.asarray(columns[name])[keep], STRIP_POINTS)
        fig.add_trace(go.Scattergl(
            x=xs, y=ys, mode="lines", name=name,
            line=dict(color=SERIES[(i - 1) % len(SERIES)], width=1.8),
            showlegend=False,
            hovertemplate=f"{name}: %{{y:.6g}} {unit}<br>t %{{x:.3f}} s<extra></extra>",
        ), row=i, col=1)
        fig.update_yaxes(title_text=f"{name}<br>{unit}", row=i, col=1, **_AXIS)
    _cursor(fig, cursor_t)
    fig.update_xaxes(**_AXIS)
    fig.update_xaxes(title_text="time  s", row=len(names), col=1)
    if xrange:
        fig.update_xaxes(range=sorted(float(v) for v in xrange))
    height = 90 + 120 * len(names)
    fig = _base(fig, height, title="Channels",
                subtitle="Every sample; decimated to a min/max envelope of the "
                         "visible range, so zoom in to see each step.")
    fig.update_layout(uirevision=None)
    return fig


def terms_of(columns: dict, coeff: str) -> list[str]:
    return [k for k in columns if k.startswith(coeff + ".")]


def dominant(columns: dict, coeff: str, index: int, relative: bool = True):
    """The term that moves `coeff` most at sample `index`: (term, value, share).

    Relative to the first sample by default: at trim the terms of Cm cancel by
    construction, so the question a peak asks is which term CHANGED.
    """
    terms = terms_of(columns, coeff)
    if not terms:
        return None
    values = {k: float(columns[k][index] - (columns[k][0] if relative else 0.0))
              for k in terms}
    total = sum(abs(v) for v in values.values()) or 1.0
    term = max(values, key=lambda k: abs(values[k]))
    return term.split(".", 1)[1], values[term], abs(values[term]) / total


def budget(columns: dict, coeff: str, cursor_t=None, relative: bool = True,
           xrange=None) -> go.Figure:
    """The terms of one coefficient, stacked, with the total drawn over them.

    Positive and negative parts stack separately (two stack groups), so a
    term that pulls the other way reads as area below zero rather than
    cancelling a neighbour's area. `relative` subtracts each term's first
    sample: the change since trim.
    """
    t = np.asarray(columns["t"])
    keep = _window(t, xrange)
    fig = go.Figure()
    layers = []
    for i, key in enumerate(terms_of(columns, coeff)):
        y = np.asarray(columns[key])
        layers.append((i, key, (y - y[0] if relative else y)[keep]))
    total = sum(y for _, _, y in layers) if layers else np.zeros(int(keep.sum()))
    at = extreme_indices(total, STRIP_POINTS)
    xs = t[keep][at]
    for i, key, y in layers:
        if not np.any(y):
            continue
        colour = TERM_COLOURS[i % len(TERM_COLOURS)]
        term = key.split(".", 1)[1]
        for part, group, show in ((np.maximum(y, 0.0)[at], "positive", True),
                                  (np.minimum(y, 0.0)[at], "negative", False)):
            fig.add_trace(go.Scatter(
                x=xs, y=part, mode="lines", stackgroup=group, name=term,
                line=dict(width=0.5, color=colour), fillcolor=colour,
                legendgroup=term, showlegend=show,
                hovertemplate=f"{term}: %{{y:.4g}}<extra></extra>"))
    ys = total[at]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=f"{coeff} total",
                             line=dict(color="#0b0b0b", width=1.6),
                             hovertemplate=f"{coeff}: %{{y:.4g}}<extra></extra>"))
    fig.add_hline(y=0.0, line=dict(color=AXIS_RULE, width=1))
    _cursor(fig, cursor_t)
    fig.update_xaxes(title_text="time  s", **_AXIS)
    fig.update_yaxes(title_text=f"Δ{coeff} since trim" if relative else coeff, **_AXIS)
    fig = _base(fig, 360, title=f"{coeff} budget: which term makes the peak",
                subtitle="Each term of aero.coefficient_terms, stacked; the black "
                         "line is the total. " + ("Change since the first sample."
                                                  if relative else "Absolute values."),
                legend=True)
    return fig


def force_budget(columns: dict, cursor_t=None, xrange=None) -> go.Figure:
    """Body-axis forces by source, and the moments about the CG."""
    from plotly.subplots import make_subplots

    t = np.asarray(columns["t"])
    keep = _window(t, xrange)
    rows = [
        ("X  N", [("aero", "Fx_aero"), ("thrust", "Fx_thrust"), ("gravity", "Fx_gravity")]),
        ("Z  N", [("aero", "Fz_aero"), ("thrust", "Fz_thrust"), ("gravity", "Fz_gravity")]),
        ("moment  N·m", [("L roll", "L_moment"), ("M pitch", "M_moment"),
                         ("N yaw", "N_moment")]),
    ]
    fig = make_subplots(rows=len(rows), cols=1, shared_xaxes=True, vertical_spacing=0.05)
    for r, (label, traces) in enumerate(rows, start=1):
        for i, (name, key) in enumerate(traces):
            xs, ys = envelope(t[keep], np.asarray(columns[key])[keep], STRIP_POINTS)
            colour = (SERIES[i % 3] if r < 3 else TERM_COLOURS[3 + i])
            fig.add_trace(go.Scattergl(
                x=xs, y=ys, mode="lines", name=name, legendgroup=name,
                showlegend=r in (1, 3), line=dict(color=colour, width=1.6),
                hovertemplate=f"{name}: %{{y:.4g}}<extra></extra>"), row=r, col=1)
        fig.update_yaxes(title_text=label, row=r, col=1, **_AXIS)
    _cursor(fig, cursor_t)
    fig.update_xaxes(**_AXIS)
    fig.update_xaxes(title_text="time  s", row=len(rows), col=1)
    return _base(fig, 460, title="Forces and moments, body axes",
                 subtitle="Force by source on x and z; the moments about the CG, "
                          "thrust moment included.", legend=True)


def energy_budget(columns: dict, cursor_t=None, xrange=None) -> go.Figure:
    """Kinetic, potential, the work of the non-conservative forces, and what is left."""
    from plotly.subplots import make_subplots

    t = np.asarray(columns["t"])
    keep = _window(t, xrange)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        row_heights=[0.62, 0.38])
    for i, (name, key) in enumerate((("ΔKE", "kinetic"), ("ΔPE", "potential"),
                                     ("work", "work"))):
        y = np.asarray(columns[key])
        y = (y - y[0] if key != "work" else y)[keep]
        xs, ys = envelope(t[keep], y, STRIP_POINTS)
        fig.add_trace(go.Scattergl(x=xs, y=ys, mode="lines", name=name,
                                   line=dict(color=SERIES[i], width=1.6)), row=1, col=1)
    xs, ys = envelope(t[keep], np.asarray(columns["energy_residual"])[keep], STRIP_POINTS)
    fig.add_trace(go.Scattergl(x=xs, y=ys, mode="lines", name="residual",
                               line=dict(color=INK, width=1.6)), row=2, col=1)
    fig.add_hline(y=0.0, line=dict(color=AXIS_RULE, width=1), row=2, col=1)
    fig.update_yaxes(title_text="J", row=1, col=1, **_AXIS)
    fig.update_yaxes(title_text="(E - E0) - W  J", row=2, col=1, **_AXIS)
    _cursor(fig, cursor_t)
    fig.update_xaxes(**_AXIS)
    fig.update_xaxes(title_text="time  s", row=2, col=1)
    return _base(fig, 420, title="Energy budget",
                 subtitle="ΔKE + ΔPE should equal the work of aero and thrust; the "
                          "residual is what the energy-closure gate measures.",
                 legend=True)


def check_profiles(profiles: dict, rows: list[dict], cursor_t=None) -> go.Figure:
    """Every check with a time profile, one strip each, with its limits."""
    from plotly.subplots import make_subplots

    shown = [(r, profiles[r["name"]]) for r in rows
             if not isinstance(profiles.get(r["name"]), (str, type(None)))]
    if not shown:
        return _base(go.Figure(), 120, title="No check has a time profile")
    fig = make_subplots(rows=len(shown), cols=1, shared_xaxes=True,
                        vertical_spacing=min(0.035, 0.3 / len(shown)),
                        subplot_titles=[
                            f"{r['name']}  ({r['kind']}"
                            + ("" if r.get("passed") is None
                               else ", passed" if r["passed"] else ", FAILED") + ")"
                            for r, _ in shown])
    for i, (row, p) in enumerate(shown, start=1):
        zero = all(not np.any(np.nan_to_num(np.asarray(y, dtype=float)))
                   for _, y in p.lines)
        log = p.log and not zero  # a log axis cannot draw an exact zero
        for j, (label, y) in enumerate(p.lines):
            y = np.asarray(y, dtype=float)
            if log:
                y = np.where(y > 0, y, np.nan)
            xs, ys = envelope(np.asarray(p.t), y, STRIP_POINTS)
            fig.add_trace(go.Scattergl(
                x=xs, y=ys, mode="lines", name=label, showlegend=False,
                line=dict(color=SERIES[j % 3], width=1.6),
                hovertemplate=f"{label}: %{{y:.4g}} {p.unit}<extra></extra>"),
                row=i, col=1)
        for label, value in p.limits:
            if log and value <= 0:
                continue
            fig.add_hline(y=value, line=dict(color=REFERENCE, width=1, dash="dot"),
                          row=i, col=1)
            # An annotation on a log axis takes the exponent, not the value.
            fig.add_annotation(text=f"{label} {value:.3g}", showarrow=False,
                               xref="x domain", x=1.0, xanchor="right",
                               y=float(np.log10(value)) if log else value,
                               yanchor="bottom", font=dict(size=9, color=INK_MUTED),
                               row=i, col=1)
        if zero:
            fig.add_annotation(text="exactly zero at every sample", showarrow=False,
                               xref="x domain", x=0.5, yref="y domain", y=0.5, yshift=12,
                               bgcolor="#ffffff", font=dict(size=11, color=INK_MUTED),
                               row=i, col=1)
        fig.update_yaxes(title_text=p.unit, type="log" if log else "linear",
                         row=i, col=1, **_AXIS)
    for a in fig.layout.annotations[:len(shown)]:  # the subplot titles come first
        a.update(font=dict(size=11, color=INK_MUTED), x=0.0, xanchor="left")
    _cursor(fig, cursor_t)
    fig.update_xaxes(**_AXIS)
    fig.update_xaxes(title_text="time  s", row=len(shown), col=1)
    fig = _base(fig, 60 + 150 * len(shown), title="Check profiles",
                subtitle="Each check as its time series, with its tolerance. The "
                         "badge's value is this line's peak.")
    # The first strip's title sits above its plot: room for it below the subtitle.
    return fig.update_layout(height=fig.layout.height + 18,
                             margin=dict(t=fig.layout.margin.t + 18))


def _enu(v_ned):
    """NED to the scene's (east, north, altitude) axes."""
    v = np.asarray(v_ned, dtype=float)
    return np.array([v[..., 1], v[..., 0], -v[..., 2]])


def scene_high(traj, columns: dict, channel: str, field, *, cursor_index=None,
               cores=None, core_radius=None) -> go.Figure:
    """The flight path coloured by any channel; at the cursor the aircraft's body
    axes, its velocity, the relative wind, and the wind on a grid around it;
    vortex cores as tubes of radius r0.

    FRAMED ON THE CURSOR. The whole path is kilometres long and the aircraft
    is metres wide, so a frame that holds both shows neither: with a cursor the
    box is a window of the path around it, and the path outside is clipped.
    Without one it is the whole path. Arrows are drawn to one length per kind,
    and the legend says what that length means.
    """
    import jax
    import jax.numpy as jnp

    from atisim.state import quat_to_dcm

    pos = np.asarray(traj.pos_ned)
    n = len(pos)
    values = np.asarray(columns.get(channel, columns["n_z"]))
    east, north, alt = pos[:, 1], pos[:, 0], -pos[:, 2]
    span = float(np.ptp(north)) or 1.0
    half = max(3.0 * (core_radius or 0.0), 0.05 * span, 300.0)
    arrow = half / 5.0
    if cursor_index is None:
        local = np.ones(n, dtype=bool)
        arrow = 0.04 * span
    else:
        local = np.abs(north - north[int(cursor_index)]) <= half

    fig = go.Figure()
    idx = np.flatnonzero(local)
    idx = idx[np.linspace(0, len(idx) - 1, min(len(idx), 4000)).astype(int)]
    fig.add_trace(go.Scatter3d(
        x=east[idx], y=north[idx], z=alt[idx], mode="lines", name=channel,
        line=dict(width=6, color=values[idx], colorscale=SEQUENTIAL,
                  cmin=float(values.min()), cmax=float(values.max()),
                  colorbar=dict(title=dict(text=channel, side="top"), orientation="h",
                                thickness=10, len=0.5, x=0.5, xanchor="center", y=0.0,
                                yanchor="bottom")),
        hovertemplate="north %{y:.0f} m<br>alt %{z:.0f} m<extra></extra>"))

    lo_n, hi_n = float(north[idx].min()) - arrow, float(north[idx].max()) + arrow
    first = True
    for (n0, a0) in cores or []:
        if not core_radius or not (lo_n - core_radius <= n0 <= hi_n + core_radius):
            continue
        theta = np.linspace(0, 2 * np.pi, 24)
        e = np.array([float(east[idx].min()) - 2 * arrow, float(east[idx].max()) + 2 * arrow])
        th, ee = np.meshgrid(theta, e)
        fig.add_trace(go.Surface(
            x=ee, y=n0 + core_radius * np.cos(th), z=a0 + core_radius * np.sin(th),
            opacity=0.25, showscale=False, colorscale=[[0, REFERENCE], [1, REFERENCE]],
            name=f"cores, radius r0 = {core_radius:g} m", hoverinfo="skip",
            showlegend=first, legendgroup="cores"))
        first = False

    if cursor_index is not None:
        i = int(cursor_index)
        p = np.array([east[i], north[i], alt[i]])
        dcm = np.asarray(quat_to_dcm(jnp.asarray(traj.quat[i])))  # body -> NED
        for k, (label, colour) in enumerate((("body x", SERIES[0]), ("body y", SERIES[1]),
                                             ("body z", SERIES[2]))):
            d = _enu(dcm[:, k]) * arrow
            fig.add_trace(go.Scatter3d(
                x=[p[0], p[0] + d[0]], y=[p[1], p[1] + d[1]], z=[p[2], p[2] + d[2]],
                mode="lines", line=dict(color=colour, width=6),
                name=f"{label} ({arrow:.0f} m)"))
        v_ned = dcm @ np.asarray(traj.vel_body[i])
        wind_ned = np.asarray(traj.wind_ned[i])
        speed = float(np.linalg.norm(v_ned)) or 1.0
        for label, vec, colour in (("velocity", v_ned, "#0b0b0b"),
                                   ("relative wind", -(v_ned - wind_ned), INK_MUTED)):
            d = _enu(vec) / speed * 2 * arrow
            fig.add_trace(go.Scatter3d(
                x=[p[0], p[0] + d[0]], y=[p[1], p[1] + d[1]], z=[p[2], p[2] + d[2]],
                mode="lines", line=dict(color=colour, width=4),
                name=f"{label} ({2 * arrow:.0f} m per {speed:.0f} m/s)"))
        g = np.linspace(-2 * arrow, 2 * arrow, 5)
        gn, ga = np.meshgrid(g, g)
        pts = np.stack([north[i] + gn.ravel(), np.full(gn.size, east[i]),
                        -(alt[i] + ga.ravel())], axis=1)
        w = np.asarray(jax.jit(jax.vmap(field))(jnp.asarray(pts)))
        wmax = float(np.abs(w).max())
        if wmax > 0:
            fig.add_trace(go.Cone(
                x=pts[:, 1], y=pts[:, 0], z=-pts[:, 2], u=w[:, 1], v=w[:, 0], w=-w[:, 2],
                # Scaled to the grid spacing: the largest cone spans about one cell.
                sizemode="scaled", sizeref=0.9, anchor="tail",
                colorscale=[[0, "#9bbbe0"], [1, "#2a78d6"]], showscale=False,
                name=f"wind on a grid (peak {wmax:.1f} m/s)", showlegend=True, opacity=0.7,
                hovertemplate="wind %{norm:.2f} m/s<extra></extra>"))
        fig.add_trace(go.Scatter3d(x=[p[0]], y=[p[1]], z=[p[2]], mode="markers",
                                   marker=dict(size=5, color=CRITICAL),
                                   name=f"cursor, t = {float(traj.t[i]):.2f} s"))
    scene = dict(xaxis=dict(title="east  m"), yaxis=dict(title="north  m"),
                 zaxis=dict(title="altitude  m"),
                 domain=dict(x=[0.0, 1.0], y=[0.16, 1.0]))
    _frame(fig, scene)
    # Larger than the encounter scene's box: this one is a local window, and the
    # reader is here for the arrows at its centre. (The projection is
    # orthographic, so the eye's distance does not zoom; the box's size does.)
    scene["aspectratio"] = {k: 1.8 * v for k, v in scene["aspectratio"].items()}
    fig.update_layout(scene=scene)
    fig = _base(fig, 560, title="Scene, high detail",
                subtitle="Framed on the cursor. Body axes, velocity and relative wind at "
                         "the aircraft; the wind on a grid around it; cores as tubes of r0.",
                legend=True)
    fig.update_layout(margin_l=8)
    return fig


def compare_strips(runs: list, channel: str, labels: list[str], unit: str = "",
                   cursor_t=None) -> go.Figure:
    """Two or more runs overlaid, and the difference of each from the first.

    `runs` is a list of (t, y). A run on a different time base is interpolated
    onto the first run's, linearly, and the difference strip says so.
    """
    from plotly.subplots import make_subplots

    t0, y0 = (np.asarray(v) for v in runs[0])
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        row_heights=[0.55, 0.45])
    resampled = False
    for i, ((t, y), label) in enumerate(zip(runs, labels)):
        t, y = np.asarray(t), np.asarray(y)
        xs, ys = envelope(t, y, STRIP_POINTS)
        fig.add_trace(go.Scattergl(x=xs, y=ys, mode="lines", name=label,
                                   line=dict(color=TERM_COLOURS[i % 10], width=1.6)),
                      row=1, col=1)
        if i == 0:
            continue
        same = len(t) == len(t0) and np.array_equal(t, t0)
        resampled |= not same
        on_a = y if same else np.interp(t0, t, y, left=np.nan, right=np.nan)
        xs, ys = envelope(t0, on_a - y0, STRIP_POINTS)
        fig.add_trace(go.Scattergl(x=xs, y=ys, mode="lines", name=f"{label} − {labels[0]}",
                                   line=dict(color=TERM_COLOURS[i % 10], width=1.6)),
                      row=2, col=1)
    fig.add_hline(y=0.0, line=dict(color=AXIS_RULE, width=1), row=2, col=1)
    fig.update_yaxes(title_text=f"{channel}  {unit}", row=1, col=1, **_AXIS)
    fig.update_yaxes(title_text=f"Δ {channel}", row=2, col=1, **_AXIS)
    _cursor(fig, cursor_t)
    fig.update_xaxes(**_AXIS)
    fig.update_xaxes(title_text="time  s", row=2, col=1)
    note = (" B was interpolated onto A's time base (linear)." if resampled else "")
    return _base(fig, 440, title=f"{channel}: overlay and difference",
                 subtitle="Top: each run. Bottom: each run minus the first, on the "
                          "first run's time base." + note, legend=True)
