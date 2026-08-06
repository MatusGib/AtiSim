"""Vortex-encounter analysis and its figure.

This is the evaluation surface for the turbulence work: one figure that answers
"is the vortex analysis going right?" without reading any code.

It is deliberately separate from `viz.py`. That module plots a FLOWN run --
ground track, altitude, mode timeline -- and computes its incidence from
`state.vel_body`, i.e. INERTIAL velocity. That is fine in still air and wrong
under wind, where it reports ground-relative angle of attack. Everything here
is air-relative, computed from `dynamics.relative_velocity`, which is the whole
point: in a Parks Case 1 encounter the two differ by up to 7 deg.

Nothing here modifies `State`, `SimState` or `viz.Trajectory`. Load factor is
recovered post hoc through `dynamics.load_factor`, and the wind is recovered by
re-evaluating the model at each logged state -- exact for a deterministic
position-dependent field, which is all this module currently handles.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from flightsim import dynamics, integrate, trim, wind
from flightsim.aero import air_data
from flightsim.aircraft import Aircraft
from flightsim.atmosphere import density
from flightsim.state import Controls, State, quat_to_euler
from flightsim.units import RAD2DEG

# Wingrove & Bach 1994 Fig. 8, "maximum negative changes" per category. These
# are DC-10/L-1011-class records; the sim flies a 747 at a different altitude
# and speed, so they are drawn as reference context, never as a target.
FIG8_REFERENCE = {"vortex": 1.4, "updraft": 6.2, "manoeuvring": 12.0}
FIG8_LOAD_BAND = (-2.01, -1.69)  # g, the range the three categories span


class Encounter(NamedTuple):
    """One analysed run through a wind field. All numpy, all SI unless noted."""

    label: str
    t: np.ndarray  # (n,) s
    north: np.ndarray  # (n,) m
    altitude: np.ndarray  # (n,) m
    w_up: np.ndarray  # (n,) m/s, vertical gust at the aircraft, positive up
    q_gust: np.ndarray  # (n,) rad/s, pitching gust rate
    alpha_air: np.ndarray  # (n,) rad, AIR-RELATIVE
    alpha_inertial: np.ndarray  # (n,) rad, what viz.derived would report
    theta: np.ndarray  # (n,) rad
    q: np.ndarray  # (n,) rad/s
    n_z: np.ndarray  # (n,) g
    elevator: np.ndarray  # (n,) rad
    window: np.ndarray  # (n,) bool, the declared analysis window
    window_name: str


def fly(
    ac: Aircraft,
    field,
    airspeed: float,
    altitude: float,
    *,
    label: str,
    start_north: float,
    seconds: float,
    dt: float = 0.01,
    window: tuple[float, float],
    window_name: str,
) -> Encounter:
    """Fly the trimmed aircraft through `field` with fixed controls.

    Open loop on purpose. The Fig. 8 discriminator separates turbulence from
    MANOEUVRING by whether the pitch response correlates with elevator, so an
    autopilot in the loop would make the categories ambiguous before the third
    one is even defined.

    `window` is (north_min, north_max) in metres and is a DECLARED choice: the
    same run yields different Fig. 8 coordinates depending on it, so it is a
    named argument rather than a default buried here.
    """
    x, _ = trim.trim(jnp.array(airspeed), jnp.array(altitude), ac)
    alpha_trim = jnp.array(float(x[0]))
    controls = trim.trimmed_controls(x[1], x[2])
    model = wind.field_model(field)

    state = trim.trimmed_state(alpha_trim, jnp.array(airspeed), jnp.array(altitude))
    state = state._replace(pos_ned=jnp.array([start_north, 0.0, -altitude]))

    n = int(round(seconds / dt))
    _, hist = integrate.rollout(
        integrate.init_sim(state, jax.random.PRNGKey(0)),
        controls, jnp.array(dt), ac, n, wind_model=model,
    )

    def analyse(pos_ned, vel_body, quat, omega):
        s = State(pos_ned=pos_ned, vel_body=vel_body, quat=quat, omega=omega)
        wind_ned, omega_gust, _, _ = model(
            wind.zero_wind_state(), s, jax.random.PRNGKey(0), jnp.array(dt)
        )
        vel_rel = dynamics.relative_velocity(vel_body, quat, wind_ned)
        _, alpha_air, _ = air_data(vel_rel)
        _, alpha_inertial, _ = air_data(vel_body)
        _, theta, _ = quat_to_euler(quat)
        return jnp.array([
            -wind_ned[2], omega_gust[1], alpha_air, alpha_inertial, theta,
            omega[1], dynamics.load_factor(s, controls, ac, wind_ned, omega_gust),
        ])

    rows = np.asarray(
        jax.vmap(analyse)(hist.pos_ned, hist.vel_body, hist.quat, hist.omega)
    )
    north = np.asarray(hist.pos_ned)[:, 0]
    return Encounter(
        label=label,
        t=np.arange(1, n + 1) * dt,
        north=north,
        altitude=-np.asarray(hist.pos_ned)[:, 2],
        w_up=rows[:, 0], q_gust=rows[:, 1],
        alpha_air=rows[:, 2], alpha_inertial=rows[:, 3],
        theta=rows[:, 4], q=rows[:, 5], n_z=rows[:, 6],
        elevator=np.full(n, float(x[1])),
        window=(north >= window[0]) & (north <= window[1]),
        window_name=window_name,
    )


def fig8_point(enc: Encounter) -> tuple[float, float]:
    """(pitch excursion in deg, most-negative load excursion in g) in the window.

    The load excursion is measured from the run's own trimmed value rather than
    from 1 g, because the body-normal load factor at trim is cos(theta), not 1.
    """
    theta = enc.theta[enc.window]
    return (
        float(theta.max() - theta.min()) * RAD2DEG,
        float(enc.n_z[enc.window].min() - enc.n_z[0]),
    )


def figure(
    encounters: list[Encounter],
    *,
    field,
    array_cores: list[tuple[float, float]],
    core_radius: float,
    peak_tangential: float,
    provenance: str,
    title: str = "vortex encounter analysis",
):
    """The evaluation figure. Returns a Figure; never shows and never saves.

    `encounters[0]` drives the field cross-section and the trace stack; every
    encounter contributes a point to the Fig. 8 panel.
    """
    primary = encounters[0]
    fig = plt.figure(figsize=(15.0, 9.5))
    fig.suptitle(title)
    grid = fig.add_gridspec(3, 2, width_ratios=(1.5, 1.0), hspace=0.45, wspace=0.18)

    _field_panel(
        fig.add_subplot(grid[0, 0]), primary, field, array_cores,
        core_radius, peak_tangential,
    )
    _load_vs_alpha_panel(fig.add_subplot(grid[1, 0]), primary)
    _discriminator_panel(fig.add_subplot(grid[2, 0]), encounters)
    _trace_stack(fig, grid[:, 1], primary)

    fig.text(0.01, 0.004, provenance, fontsize=6.5, family="monospace", color="0.35")
    return fig


def _field_panel(ax, enc, field, cores, r0, v0):
    """Vertical-gust field with the flight path through it.

    Diverging colour pinned to +-V0, because the physics claim under test is a
    SIGNED up-then-down doublet: a sign error reverses the colour order and is
    visible instantly, where a quiver plot would hide it in an arrow scale.
    """
    # Zoom to the structure, not the run. The lead-in is deliberately long (the
    # 1/r far field needs tens of core radii to die away) and plotting all of it
    # would shrink the cores to invisibility. Axes are BOTH in metres so
    # set_aspect("equal") is meaningful: a wrong core radius is only visible if
    # the cores are drawn round.
    first, last = cores[0][0], cores[-1][0]
    margin = 3.0 * r0
    north = np.linspace(first - margin, last + margin, 320)
    alt = np.linspace(enc.altitude.mean() - 2.0 * r0, enc.altitude.mean() + 2.0 * r0, 160)
    mesh_n, mesh_a = np.meshgrid(north, alt)
    points = jnp.stack(
        [
            jnp.asarray(mesh_n.ravel()),
            jnp.zeros(mesh_n.size),
            jnp.asarray(-mesh_a.ravel()),
        ],
        axis=1,
    )
    w_up = -np.asarray(jax.vmap(field)(points))[:, 2].reshape(mesh_n.shape)

    mesh = ax.pcolormesh(
        mesh_n, mesh_a, w_up, cmap="RdBu_r", vmin=-v0, vmax=v0, shading="gouraud",
    )
    # The zero contour is displaced between cores by superposition, so it is
    # the visible check that the array is summed rather than merely repeated.
    ax.contour(mesh_n, mesh_a, w_up, levels=[0.0], colors="k", linewidths=0.6)
    for centre_north, centre_alt in cores:
        # Drawn from r0 independently of the field: circle-vs-colour
        # disagreement is a transcription error.
        ax.add_patch(Circle(
            (centre_north, centre_alt), r0, fill=False, ec="k", lw=1.0, ls="--",
        ))
        ax.plot(centre_north, centre_alt, "k+", ms=7)
    ax.plot(enc.north, enc.altitude, color="k", lw=1.8, label="flight path")
    ax.axvspan(
        enc.north[enc.window].min(), enc.north[enc.window].max(),
        color="C1", alpha=0.15, label=f"window: {enc.window_name}",
    )
    ax.set_xlim(north[0], north[-1])
    ax.set_ylim(alt[0], alt[-1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("along track  m")
    ax.set_ylabel("altitude  m")
    ax.set_title("vertical gust field and flight path")
    ax.legend(fontsize=7, loc="upper left")
    plt.colorbar(mesh, ax=ax, label="vertical wind, up positive  m/s", pad=0.01)


def _load_vs_alpha_panel(ax, enc):
    """n_z against incidence. This panel is an assertion, not a display.

    The air-relative points must fall on a straight line; the inertial ones
    scatter. The straightness is also the honest picture of the permanent
    structural gap -- a real aircraft's curve bends at buffet onset and this
    one cannot, which is why no +/-g asymmetry can come from aero here.
    """
    ax.scatter(
        enc.alpha_inertial * RAD2DEG, enc.n_z, s=4, color="0.75",
        label="inertial alpha (what viz.derived reports)", zorder=1,
    )
    points = ax.scatter(
        enc.alpha_air * RAD2DEG, enc.n_z, c=enc.t, cmap="viridis", s=5,
        label="air-relative alpha", zorder=2,
    )
    ax.set_xlabel("angle of attack  deg")
    ax.set_ylabel("load factor  g")
    ax.set_title("load factor against incidence (linear by construction: no stall model)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper left")
    plt.colorbar(points, ax=ax, label="time  s", pad=0.01)


def _discriminator_panel(ax, encounters):
    """Wingrove & Bach Fig. 8, with the model's points and the missing category.

    Progress is shown by what is ABSENT as much as by what is present: the
    manoeuvring cluster is drawn as a labelled empty slot rather than omitted,
    so the figure cannot read as complete while it is not.
    """
    low, high = FIG8_LOAD_BAND
    ax.axhspan(low, high, color="0.88", zorder=0)
    for name, dtheta in FIG8_REFERENCE.items():
        ax.plot(
            dtheta, 0.5 * (low + high), "s", mfc="none", mec="0.45", ms=9,
            zorder=2,
        )
        ax.annotate(
            name, (dtheta, high), textcoords="offset points", xytext=(0, 6),
            ha="center", fontsize=7, color="0.35",
        )

    for index, enc in enumerate(encounters):
        dtheta, dn = fig8_point(enc)
        ax.plot(dtheta, dn, "o", color=f"C{index}", ms=8, label=f"{enc.label} (model)")
        ax.annotate(
            f"{dtheta:.2f} deg\n{dn:+.2f} g", (dtheta, dn),
            textcoords="offset points", xytext=(7, -2), fontsize=7, color=f"C{index}",
        )
        # The windowing trap, drawn: the same run measured over its whole
        # length migrates across the chart into another category's cluster.
        whole = float(enc.theta.max() - enc.theta.min()) * RAD2DEG
        whole_dn = float(enc.n_z.min() - enc.n_z[0])
        ax.plot(whole, whole_dn, "o", mfc="none", mec=f"C{index}", ms=8)
        ax.plot([dtheta, whole], [dn, whole_dn], ls=":", color=f"C{index}", lw=0.9)

    ax.axvline(FIG8_REFERENCE["manoeuvring"], color="0.6", ls=":", lw=1.0)
    ax.annotate(
        "manoeuvring:\nNOT MODELLED", (FIG8_REFERENCE["manoeuvring"], 0.02),
        xycoords=("data", "axes fraction"), textcoords="offset points",
        xytext=(-4, 4), ha="right", va="bottom", fontsize=7, color="0.45",
    )
    ax.set_xlim(0.0, FIG8_REFERENCE["manoeuvring"] + 2.0)
    ax.set_xlabel("pitch attitude excursion  deg")
    ax.set_ylabel("load excursion from trim  g")
    ax.set_title("Wingrove & Bach 1994 Fig. 8 discriminator (grey = paper, DC-10 class)")
    ax.grid(alpha=0.3)
    ax.legend(
        fontsize=7, loc="center right",
        title="hollow marker = whole run (WRONG window)", title_fontsize=6,
    )


def _trace_stack(fig, cell, enc):
    """The causal chain, top to bottom: gust in, incidence, load, attitude, control."""
    axes = cell.subgridspec(5, 1, hspace=0.12).subplots(sharex=True)
    t = enc.t

    axes[0].plot(t, enc.w_up, color="C0")
    axes[0].set_ylabel("gust up\nm/s", fontsize=8)
    twin = axes[0].twinx()
    twin.plot(t, enc.q_gust * RAD2DEG, color="C3", lw=0.9, ls="--")
    twin.set_ylabel("q gust\ndeg/s", fontsize=8, color="C3")
    twin.tick_params(axis="y", labelcolor="C3", labelsize=7)

    axes[1].plot(t, enc.alpha_air * RAD2DEG, color="C0", label="air-relative")
    axes[1].plot(
        t, enc.alpha_inertial * RAD2DEG, color="0.7", lw=0.8, ls="--", label="inertial",
    )
    axes[1].set_ylabel("alpha\ndeg", fontsize=8)
    axes[1].legend(fontsize=6, loc="upper right")

    axes[2].plot(t, enc.n_z, color="C0")
    axes[2].axhline(enc.n_z[0], color="0.5", lw=0.8)
    axes[2].set_ylabel("n_z\ng", fontsize=8)

    axes[3].plot(t, enc.theta * RAD2DEG, color="C0")
    axes[3].set_ylabel("theta\ndeg", fontsize=8)
    twin = axes[3].twinx()
    twin.plot(t, enc.q * RAD2DEG, color="C3", lw=0.9, ls="--")
    twin.set_ylabel("q\ndeg/s", fontsize=8, color="C3")
    twin.tick_params(axis="y", labelcolor="C3", labelsize=7)

    axes[4].plot(t, enc.elevator * RAD2DEG, color="C0")
    axes[4].set_ylabel("elevator\ndeg", fontsize=8)
    axes[4].set_xlabel("time  s")
    axes[4].set_ylim(-5.0, 5.0)

    window_t = t[enc.window]
    for ax in axes:
        ax.axvspan(window_t.min(), window_t.max(), color="C1", alpha=0.15)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=7)
    axes[0].set_title(f"{enc.label}: encounter traces", fontsize=9)
    return axes
