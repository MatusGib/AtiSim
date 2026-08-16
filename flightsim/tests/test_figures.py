"""The plot layer: the countermeasures, not the pixels.

`test_vortex_viz.py` sets the precedent -- *"plotting code is tested for the
things a reader would be misled by if they were wrong"*. Here that means the
decimation, the pinned axes, the payload budget and `uirevision`, because each
one is a specific way a plot can look right while the data is wrong.
"""

import numpy as np
import pytest

import flightsim  # noqa: F401
from flightsim.analysis import series

go = pytest.importorskip("plotly.graph_objects",
                         reason="the figure layer needs the `ui` extra")
figures = pytest.importorskip("flightsim.analysis.figures")


# --- decimation: the measured centrepiece of "how plots lie" -----------------


def test_envelope_decimation_keeps_the_peak_that_a_stride_destroys():
    """The measured case, reproduced on a signal with a narrow peak.

    On the canonical vortex run a stride to 250 points reports -1.164 g against
    the true -1.235 g -- 5.7% off the number PROJECT.md section 4 headlines --
    while the envelope loses 0.0000%. This is the same statement on a signal
    where the peak's position is known.
    """
    t = np.linspace(0.0, 40.0, 4000)
    y = np.sin(t)
    y[1717] = -5.0  # a one-sample spike, the shape of a core traverse

    xs, ys = series.envelope(t, y, 250)
    assert ys.min() == pytest.approx(-5.0), "the envelope must keep the peak"
    assert len(xs) <= 400

    stride = y[:: len(y) // 125]
    assert stride.min() > -5.0, "the stride was supposed to miss it"


def test_envelope_decimation_keeps_x_monotone():
    """Otherwise a line plot zigzags backwards between buckets."""
    t = np.linspace(0.0, 10.0, 5000)
    xs, _ = series.envelope(t, np.sin(7.0 * t), 300)
    assert np.all(np.diff(xs) > 0)


def test_envelope_decimation_is_a_no_op_below_the_target():
    t = np.arange(50.0)
    xs, ys = series.envelope(t, t**2, 2000)
    assert np.array_equal(xs, t)


def test_envelope_keeps_both_ends():
    """A run that is decimated away at the ends has lost its start and finish."""
    t = np.linspace(0.0, 1.0, 3000)
    xs, _ = series.envelope(t, np.cos(20 * t), 200)
    assert xs[0] == t[0] and xs[-1] == t[-1]


# --- the figures -------------------------------------------------------------


@pytest.fixture(scope="module")
def sample():
    """A short synthetic run. The figures must not need a simulator."""
    n = 3000
    t = np.linspace(0.0, 30.0, n)
    return series.Series(
        t=t,
        north=np.linspace(-4000.0, 4000.0, n),
        east=np.zeros(n),
        altitude=12192.0 + 40.0 * np.sin(t / 3),
        w_up=18.0 * np.sin(t / 2),
        q_gust_deg=3.0 * np.cos(t / 2),
        alpha_deg=4.6 + 3.0 * np.sin(t / 2),
        alpha_inertial_deg=4.6 + 5.0 * np.sin(t / 2 + 0.4),
        beta_deg=np.zeros(n),
        theta_deg=4.6 + 1.2 * np.sin(t / 4),
        q_deg=0.4 * np.cos(t / 4),
        n_z=1.0 - 1.2 * np.exp(-((t - 15.0) ** 2) / 0.5),
        elevator_deg=np.full(n, -0.6),
        airspeed=np.full(n, 235.9),
    )


def _payload_mb(fig) -> float:
    return len(fig.to_json().encode()) / 1e6


def test_the_strip_stack_holds_the_payload_budget(sample):
    """2 MB per callback is the measured budget; the strips are most of a sweep."""
    window = (sample.north >= -183.0) & (sample.north <= 183.0)
    fig = figures.strip_stack(sample, window, cursor_t=15.0)
    assert _payload_mb(fig) < 0.6, f"{_payload_mb(fig):.3f} MB"


def test_a_whole_sweep_view_holds_the_two_megabyte_callback_budget(sample):
    """The budget is enforced, not aspirational.

    Measured: 12 Scattergl traces at full rate serialise to 8.5 MB and a cubic
    48^3 isosurface to 5.0 MB, so a sweep view that does not police itself is
    seconds per interaction rather than milliseconds. A cubic 32^3 vortex scene
    came to 2.29 MB and this is what caught it.
    """
    window = np.ones(len(sample.t), dtype=bool)
    panels = [
        figures.strip_stack(sample, window, cursor_t=12.0),
        figures.field_3d(sample, lambda p: p * 0.0, cursor_index=3,
                         representation="isosurface", core_radius=182.88),
        figures.discriminator([dict(label="v", dtheta=2.24, dn=-1.235,
                                    dtheta_whole=8.33, dn_whole=-1.3)]),
        figures.load_vs_alpha(sample, cursor_index=3),
    ]
    total = sum(_payload_mb(f) for f in panels)
    assert total < 2.0, f"sweep view is {total:.3f} MB"


def test_the_field_is_drawn_around_the_STRUCTURE_not_the_trajectory_midpoint():
    """Zoom to the structure, not the run -- `vortex_viz._field_panel`'s own rule.

    The canonical vortex run has a 40 core-radii lead-in, so its trajectory
    spans -7315 m to +2364 m and its MIDPOINT is about -2476 m: roughly 2.5 km
    upstream of both cores. A field grid centred there samples only the
    irrotational far field, where a Rankine vortex has exactly zero vorticity --
    so the isosurface came out empty with `isomin` = 0, and the panel drew a
    trajectory through nothing at all while looking perfectly normal.

    Found by reading `isomin` off the live page, not by a failing test.
    """
    n = 600
    north = np.linspace(-7315.0, 2364.0, n)  # the real run's extent
    s = series.Series(
        t=np.linspace(0, 40, n), north=north, east=np.zeros(n),
        altitude=np.full(n, 12192.0), w_up=np.zeros(n), q_gust_deg=np.zeros(n),
        alpha_deg=np.full(n, 4.6), alpha_inertial_deg=np.full(n, 4.6),
        beta_deg=np.zeros(n), theta_deg=np.full(n, 4.6), q_deg=np.zeros(n),
        n_z=np.ones(n), elevator_deg=np.zeros(n), airspeed=np.full(n, 235.9),
    )
    import jax.numpy as jnp

    from flightsim import wind

    r0, v0 = 182.88, 25.908
    array = wind.VortexArray(north=jnp.array([0.0, 1066.8]),
                             down=jnp.array([-12192.0, -12192.0]),
                             r0=jnp.array(r0), v0=jnp.array(v0))
    field = lambda p: wind.vortex_wind(p, array)  # noqa: E731

    fig = figures.field_3d(s, field, representation="isosurface",
                           core_radius=r0, field_centre=533.4)
    iso = [t for t in fig.data if t.type == "isosurface"][0]
    peak = float(np.max(iso.value))
    assert peak > 0.5 * (2 * v0 / r0), (
        f"the isosurface grid missed the cores: peak |curl| {peak:.4f} against "
        f"the in-core {2 * v0 / r0:.4f}"
    )
    assert iso.isomin > 0.0


def test_the_isosurface_grid_is_coarse_in_the_direction_the_field_is_flat():
    """The Parks vortex fixes dpsi = 0: the cores are infinite east-west lines.

    Spending cubic grid points on a direction the field does not vary in is pure
    payload for no information, and it is what pushed the vortex scene over
    budget.
    """
    assert figures.ISO_GRID_LATERAL < figures.ISO_GRID


def test_every_figure_holds_uirevision_constant(sample):
    """The one line that decides whether the linked cursor is usable."""
    window = np.ones(len(sample.t), dtype=bool)
    for fig in (
        figures.strip_stack(sample, window),
        figures.load_vs_alpha(sample),
        figures.discriminator([dict(label="v", dtheta=2.24, dn=-1.235,
                                    dtheta_whole=8.33, dn_whole=-1.3)]),
    ):
        assert fig.layout.uirevision == figures.UIREVISION


def test_no_panel_uses_a_spline(sample):
    """A spline through the Rankine core edge draws a curve across a real
    discontinuity: the one-sided derivatives there have OPPOSITE SIGNS."""
    window = np.ones(len(sample.t), dtype=bool)
    fig = figures.strip_stack(sample, window)
    for trace in fig.data:
        shape = getattr(getattr(trace, "line", None), "shape", None)
        assert shape in (None, "linear"), shape


def test_the_load_factor_axis_is_pinned_to_include_one_g(sample):
    """Autoscaled to the excursion, a 0.03 g trim offset fills the panel."""
    window = np.ones(len(sample.t), dtype=bool)
    fig = figures.strip_stack(sample, window)
    # n_z is the fourth subplot -> yaxis4
    rng = fig.layout.yaxis4.range
    assert rng is not None, "the n_z axis must not autoscale"
    assert rng[0] <= sample.n_z.min() and rng[1] >= 1.0


def test_the_elevator_axis_is_pinned_so_fixed_controls_look_fixed(sample):
    window = np.ones(len(sample.t), dtype=bool)
    fig = figures.strip_stack(sample, window)
    assert tuple(fig.layout.yaxis6.range) == (-5.0, 5.0)


def test_a_signed_scalar_gets_a_diverging_map_pinned_about_zero(sample):
    """Diverging maps ONLY for signed quantities, and symmetric, so a sign error
    reverses the colour order visibly."""
    fig = figures.field_3d(sample, lambda p: p * 0.0, scalar="w_up",
                           representation="none")
    line = [tr for tr in fig.data if tr.type == "scatter3d"][0].line
    assert line.colorscale is not None
    assert line.cmid == 0.0
    assert line.cmin == pytest.approx(-line.cmax)


def test_an_unsigned_scalar_does_not_get_a_diverging_map(sample):
    fig = figures.field_3d(sample, lambda p: p * 0.0, scalar="n_z",
                           representation="none")
    line = [tr for tr in fig.data if tr.type == "scatter3d"][0].line
    assert line.cmid is None


def test_the_discriminator_draws_the_whole_run_marker_and_the_connector():
    """The windowing trap is the panel's main content, not an annotation."""
    fig = figures.discriminator([
        dict(label="vortex", dtheta=2.24, dn=-1.235, dtheta_whole=8.33, dn_whole=-1.30),
    ])
    modes = [tr.mode for tr in fig.data]
    assert "lines" in modes, "no connector between windowed and whole-run"
    symbols = [getattr(tr.marker, "symbol", None) for tr in fig.data]
    assert "circle-open" in symbols, "no hollow whole-run marker"


def test_the_discriminator_x_axis_scales_to_the_data_not_the_paper():
    """A chart cropped to the paper's range hides the model sitting 2.5x right."""
    fig = figures.discriminator([
        dict(label="manoeuvre", dtheta=30.37, dn=-1.90,
             dtheta_whole=30.74, dn_whole=-1.90),
    ])
    assert fig.layout.xaxis.range[1] > 30.37


# --- camera persistence, carried explicitly rather than trusted --------------


def test_the_camera_is_restored_from_a_nested_relayout(sample):
    """The shape plotly emits when the whole camera comes back at once."""
    fig = figures.field_3d(sample, lambda p: p * 0.0, representation="none")
    figures.apply_camera(fig, {"scene.camera": {
        "eye": {"x": 0.4, "y": -1.9, "z": 0.8},
        "up": {"x": 0, "y": 0, "z": 1}}})
    assert fig.layout.scene.camera.eye.x == pytest.approx(0.4)
    assert fig.layout.scene.camera.eye.y == pytest.approx(-1.9)


def test_the_camera_is_restored_from_a_flattened_relayout(sample):
    """The shape plotly emits for a PARTIAL update -- dotted paths, not a dict.

    Handling only the nested form would work in a manual test and silently drop
    the camera on a real drag, which is the worst of both.
    """
    fig = figures.field_3d(sample, lambda p: p * 0.0, representation="none")
    figures.apply_camera(fig, {
        "scene.camera.eye.x": 0.4, "scene.camera.eye.y": -1.9,
        "scene.camera.eye.z": 0.8, "scene.camera.up.z": 1,
    })
    assert fig.layout.scene.camera.eye.x == pytest.approx(0.4)
    assert fig.layout.scene.camera.eye.z == pytest.approx(0.8)
    assert fig.layout.scene.camera.up.z == pytest.approx(1)


def test_an_unrelated_relayout_leaves_the_camera_alone(sample):
    """A 2D zoom must not invent a 3D camera."""
    fig = figures.field_3d(sample, lambda p: p * 0.0, representation="none")
    figures.apply_camera(fig, {"xaxis.range[0]": 3.0, "xaxis.range[1]": 9.0})
    assert fig.layout.scene.camera.eye.x is None


def test_no_relayout_data_is_harmless(sample):
    fig = figures.field_3d(sample, lambda p: p * 0.0, representation="none")
    figures.apply_camera(fig, None)
    figures.apply_camera(fig, {})
    assert fig.layout.scene.camera.eye.x is None


def test_a_comparison_is_a_difference_not_two_overlaid_traces(sample):
    """h and h/2 differ by 4e-5 relative and are one line when overlaid."""
    other = sample._replace(n_z=sample.n_z + 1e-4)
    fig = figures.difference(sample, other, "n_z", ("h", "h/2"))
    assert len(fig.data) == 1
    assert float(np.max(np.abs(fig.data[0].y))) == pytest.approx(1e-4, rel=1e-3)
