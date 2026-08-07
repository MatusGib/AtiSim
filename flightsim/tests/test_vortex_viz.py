"""The vortex-analysis figure and the quantities behind it.

Plotting code is tested for the things a reader would be misled by if they were
wrong -- the derived quantities and the window -- not for pixel output.
"""

import matplotlib

matplotlib.use("Agg")  # before flightsim.vortex_viz imports pyplot

import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from flightsim import vortex_viz, wind  # noqa: E402
from flightsim.aircraft import CRUISE, REGISTRY  # noqa: E402
from flightsim.units import FT2M, RAD2DEG  # noqa: E402

AC = REGISTRY["boeing747"]
V = CRUISE["boeing747"]["airspeed"]
H = CRUISE["boeing747"]["altitude"]
R0, V0, SPACING = 600.0 * FT2M, 85.0 * FT2M, 3500.0 * FT2M


def _encounter(lead_in=40.0):
    array = wind.VortexArray(
        north=jnp.array([0.0, SPACING]), down=jnp.array([-H, -H]),
        r0=jnp.array(R0), v0=jnp.array(V0),
    )
    lead = lead_in * R0
    return vortex_viz.fly(
        AC, lambda p: wind.vortex_wind(p, array), V, H,
        label="vortex", start_north=-lead,
        seconds=(SPACING + lead + 6.0 * R0) / V, dt=0.01,
        window=(-R0, R0), window_name="first core",
    )


@pytest.fixture(scope="module")
def encounter():
    return _encounter()


# The manoeuvring limb. `hold` is DECLARED (PROJECT.md section 8): one 747 short
# period, which puts the manoeuvre between the vortex's 0.235 and the updraft's
# 3.026, so the third cluster's separation cannot be a duration effect.
PUSHDOWN_HOLD = 6.609
PUSHDOWN_LEAD = 2.0


@pytest.fixture(scope="module")
def updraft():
    radius = 0.5 * wind.UPDRAFT_SECONDS * V
    column = wind.UpdraftColumn(
        north=jnp.array(0.0), east=jnp.array(0.0),
        w0=jnp.array(wind.UPDRAFT_W0), radius=jnp.array(radius),
        sharpness=jnp.array(6.0),
    )
    return vortex_viz.fly(
        AC, lambda p: wind.updraft_wind(p, column), V, H,
        label="updraft", start_north=-2.0 * radius,
        seconds=4.0 * radius / V, dt=0.01,
        window=(-radius, radius), window_name="column",
    )


@pytest.fixture(scope="module")
def pushdown():
    return vortex_viz.manoeuvre(
        AC, V, H, label="manoeuvre",
        elevator_step=jnp.deg2rad(8.926),
        hold=PUSHDOWN_HOLD, lead_in=PUSHDOWN_LEAD,
        seconds=PUSHDOWN_LEAD + 3.0 * PUSHDOWN_HOLD, dt=0.01,
    )


def test_the_manoeuvring_point_separates_from_both_turbulence_clusters(
    encounter, updraft, pushdown
):
    """PROJECT.md section 7 step 5: the discriminator needs the paper's three.

    Ordering only, never absolute agreement -- section 5 forbids the latter,
    since Wingrove & Bach never identifies an aircraft type. The paper's own
    ordering is vortex 1.4 < updraft 6.2 < manoeuvring 12.0 deg.
    """
    vortex_dtheta, vortex_dn = vortex_viz.fig8_point(encounter)
    updraft_dtheta, updraft_dn = vortex_viz.fig8_point(updraft)
    pushdown_dtheta, pushdown_dn = vortex_viz.fig8_point(pushdown)

    assert vortex_dtheta < updraft_dtheta < pushdown_dtheta
    # Separation, not merely ordering: the gap must be bigger than the gap
    # between the two turbulence points, or there is no third CLUSTER.
    assert pushdown_dtheta - updraft_dtheta > updraft_dtheta - vortex_dtheta
    # And it is the most negative load of the three, which is what puts it in
    # the Fig. 8 band at all.
    assert pushdown_dn < vortex_dn < updraft_dn


def test_the_pushdown_reaches_the_fig8_load_band_as_an_increment(pushdown):
    """Decision A (PROJECT.md section 8): the band is read as an increment.

    The elevator angle is not a chosen number -- it is bisected to land on the
    band -- so this asserts the bisection's target was actually met.
    """
    _, dn = vortex_viz.fig8_point(pushdown)
    assert dn == pytest.approx(vortex_viz.FIG8_LOAD_INCREMENT, abs=0.01)


def test_the_pushdown_stays_inside_the_declared_alpha_band(pushdown):
    """The ceiling in section 7, checked on the run rather than assumed.

    |alpha|, not alpha: a pushdown drives incidence NEGATIVE, and aero.py is
    odd-symmetric, so magnitude is what decides validity (section 6(e)).
    Measured 10.31 deg -- marginal, inside the amber band, and the figure says
    so. If this ever exceeds 12 the run proves nothing and must be reported as
    such rather than quoted.
    """
    alpha = np.abs(pushdown.alpha_air[pushdown.window]) * RAD2DEG
    assert alpha.max() < 12.0


def test_air_relative_and_inertial_incidence_disagree_by_degrees(encounter):
    """The reason this module exists rather than reusing viz.derived.

    viz.instantaneous computes alpha from state.vel_body, which is INERTIAL.
    Under wind that is ground-relative and is not what the wing sees. If these
    two ever agree closely in a vortex encounter, something has stopped
    applying the wind.
    """
    difference = np.abs(encounter.alpha_air - encounter.alpha_inertial) * RAD2DEG
    assert difference.max() > 3.0  # measured ~7.0 deg


def test_load_factor_is_a_straight_line_in_air_relative_incidence(encounter):
    """CL = CL0 + CLa*alpha, so n_z must be affine in air-relative alpha.

    This is simultaneously a correctness check on the air-relative recovery and
    the honest picture of the permanent structural gap: a real aircraft's curve
    bends at buffet onset and this one cannot, which is why no +/-g asymmetry
    can arise from the aerodynamics here.
    """
    air = np.corrcoef(encounter.alpha_air, encounter.n_z)[0, 1]
    inertial = np.corrcoef(encounter.alpha_inertial, encounter.n_z)[0, 1]
    assert air > 0.999  # measured 0.9990
    assert inertial < 0.9  # measured 0.5572 -- the same data, wrongly derived


def test_the_lead_in_matters_and_a_short_one_starts_out_of_equilibrium():
    """The artifact that made an earlier measurement wrong by 15%, pinned down.

    The vortex far field falls off only as 1/r, so a short lead-in launches the
    aircraft out of trim and the first core then measures the launch transient
    as much as the vortex.
    """
    close = _encounter(lead_in=6.0)
    far = _encounter(lead_in=40.0)
    assert close.n_z[0] > 1.15  # measured 1.2033 -- 0.2 g out of equilibrium
    assert abs(far.n_z[0] - 0.9967) < 0.05  # measured 1.0352
    assert vortex_viz.fig8_point(far)[0] > vortex_viz.fig8_point(close)[0]


def test_the_window_is_what_decides_the_discriminator_coordinate(encounter):
    """The windowing trap, asserted.

    Measured over the whole run the pitch excursion is nearly four times the
    in-core value, which on Fig. 8 would migrate the point out of the vortex
    cluster entirely. The figure draws both for exactly this reason.
    """
    in_core, _ = vortex_viz.fig8_point(encounter)
    whole = float(encounter.theta.max() - encounter.theta.min()) * RAD2DEG
    assert in_core == pytest.approx(2.20, abs=0.15)
    assert whole > 3.0 * in_core  # measured 8.33 vs 2.20


def test_the_figure_builds_with_every_panel_and_does_not_display(encounter):
    """Returns a Figure, never shows -- matching viz.post_flight's contract."""
    array_cores = [(0.0, H), (SPACING, H)]
    array = wind.VortexArray(
        north=jnp.array([0.0, SPACING]), down=jnp.array([-H, -H]),
        r0=jnp.array(R0), v0=jnp.array(V0),
    )
    figure = vortex_viz.figure(
        [encounter], field=lambda p: wind.vortex_wind(p, array),
        array_cores=array_cores, core_radius=R0, peak_tangential=V0,
        provenance="test", title="test",
    )
    # three left-hand panels, five stacked traces, two colourbars
    assert len(figure.axes) >= 10
    titles = [ax.get_title() for ax in figure.axes if ax.get_title()]
    assert any("gust field" in t for t in titles)
    assert any("Fig. 8" in t for t in titles)
    plt.close(figure)


def test_the_discriminator_panel_renders_with_a_single_category_present(encounter):
    """Partial progress must not crash the deliverable.

    Only the vortex cluster exists today; updraft and manoeuvring arrive later.
    A panel that errored on a one-entry list would block the tool from being
    useful exactly when it is most needed.
    """
    figure = plt.figure()
    vortex_viz._discriminator_panel(figure.add_subplot(1, 1, 1), [encounter])
    plt.close(figure)
