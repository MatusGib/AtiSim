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

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from atisim import dynamics, integrate, loads, trim, viz, wind
from atisim.aero import air_data
from atisim.aircraft import Aircraft
from atisim.atmosphere import density
from atisim.state import Controls, State, quat_to_euler
from atisim.units import RAD2DEG

# Wingrove & Bach 1994 Fig. 8, "maximum negative changes" per category. These
# are DC-10/L-1011-class records; the sim flies a 747 at a different altitude
# and speed, so they are drawn as reference context, never as a target.
FIG8_REFERENCE = {"vortex": 1.4, "updraft": 6.2, "manoeuvring": 12.0}
FIG8_LOAD_BAND = (-2.01, -1.69)  # g, the range the three categories span

# The band read as an INCREMENT from trim rather than as an absolute load factor.
# The paper's text does not resolve which it is; PROJECT.md section 8 records the
# decision and section 5 the reason it was forced -- the absolute reading needs
# |alpha| ~18.5 deg, half again past the ceiling where this model's linear aero
# reports lift the sources deny. The increment reading is flyable at |alpha|
# 10.31 deg. This is the midpoint of the band, to one decimal as the paper gives it.
FIG8_LOAD_INCREMENT = -1.9  # g


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
    # The run as flown, for `analysis.artifact.write_run`. Last and defaulted so
    # every existing positional construction still works. It carries the wind and
    # gust ACTUALLY APPLIED, which `integrate.rollout` discards -- re-deriving
    # them by re-evaluating the model is exact only for a deterministic field.
    log: object = None  # viz.Trajectory | None


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
    strip: bool = False,
    load_model=None,
) -> Encounter:
    """Fly the trimmed aircraft through `field` with fixed controls.

    Open loop on purpose. The Fig. 8 discriminator separates turbulence from
    MANOEUVRING by whether the pitch response correlates with elevator, so an
    autopilot in the loop would make the categories ambiguous before the third
    one is even defined.

    `window` is (north_min, north_max) in metres and is a DECLARED choice: the
    same run yields different Fig. 8 coordinates depending on it, so it is a
    named argument rather than a default buried here.

    `strip` swaps the point-plus-gradient load path for strip-integrated loads.
    It builds the load model from the field this function already holds, so it
    is a flag rather than an argument. ROLL ONLY -- see `loads.strip_increment`
    -- so it changes nothing for a field without spanwise structure, which the
    Parks vortex is. `strip=False` is byte-for-byte the run this did before.

    `load_model` is the general form, for a load model this function cannot
    build from the field alone. `strip=True` is exactly sugar for passing
    `loads.strip_model(field, ac)`, so giving both is a contradiction rather
    than an override and is refused.
    """
    if strip and load_model is not None:
        raise ValueError(
            "pass strip=True or load_model, not both -- strip=True IS "
            "load_model=loads.strip_model(field, ac), and silently preferring "
            "one would hide which load path the run actually flew"
        )
    x, _ = trim.trim(jnp.array(airspeed), jnp.array(altitude), ac)
    alpha_trim = jnp.array(float(x[0]))
    controls = trim.trimmed_controls(x[1], x[2])
    if strip:
        load_model = loads.strip_model(field, ac)

    state = trim.trimmed_state(alpha_trim, jnp.array(airspeed), jnp.array(altitude))
    state = state._replace(pos_ned=jnp.array([start_north, 0.0, -altitude]))

    return fly_from_state(
        ac, field, state, controls,
        label=label, seconds=seconds, dt=dt, window=window,
        window_name=window_name, load_model=load_model,
    )


def fly_from_state(
    ac: Aircraft,
    field,
    state,
    controls,
    *,
    label: str,
    seconds: float,
    dt: float = 0.01,
    window: tuple[float, float],
    window_name: str,
    load_model=None,
    wind_model=None,
) -> Encounter:
    """`fly`, but from a state and controls the caller already has.

    Split out of `fly` rather than copied, so the two entry points cannot drift
    apart in what they measure. `fly` trims and then calls this.

    It exists for the cross-code comparison. The two engines do NOT trim to the
    same point -- JSBSim's do_simple_trim converges to Nz = 0.99093 rather than
    1.0, worth about 0.05 deg of alpha -- so scripts/vortex_compare.py starts
    atisim from JSBSim's recorded state instead of from atisim's own trim.
    Trimming independently would begin the two runs at different angles of
    attack and carry that offset into every difference downstream.

    `wind_model` overrides the default `wind.field_model(field)`. The comparison
    needs a TRANSLATION-ONLY model for its like-for-like arm, because JSBSim has
    no writable gust-rate input and therefore carries no gradient at all; the
    default model would give atisim a term the other engine cannot have.
    """
    model = wind.field_model(field) if wind_model is None else wind_model
    n = int(round(seconds / dt))
    # `logged_rollout`, not `rollout`: same `step`, wider scan output, so the run
    # can be written to an artifact carrying the wind it actually flew.
    # `test_vortex_viz.py` pins the headline pair against the pre-change values.
    _, log = integrate.logged_rollout(
        integrate.init_sim(state, jax.random.PRNGKey(0)),
        controls, jnp.array(dt), ac, n, wind_model=model, load_model=load_model,
    )
    hist = log.state
    controls_hist = jax.tree.map(lambda v: jnp.full(n, v), controls)
    north = np.asarray(hist.pos_ned)[:, 0]
    return _measure(
        label=label, hist=hist,
        controls_hist=controls_hist,
        model=model, load_model=load_model, ac=ac, dt=dt,
        window=(north >= window[0]) & (north <= window[1]),
        window_name=window_name,
        log=integrate.trajectory_from_log(
            np.arange(1, n + 1) * dt, log, controls_hist
        ),
    )


def _measure(
    *,
    label: str,
    hist: State,
    controls_hist: Controls,
    model,
    ac: Aircraft,
    dt: float,
    window: np.ndarray,
    window_name: str,
    load_model=None,
    log=None,
) -> Encounter:
    """Turn a flown history into an `Encounter`. Every category comes through here.

    Factored out when the manoeuvring case arrived so the three Fig. 8 points are
    computed by one piece of code rather than by two that could drift apart --
    the whole value of the discriminator is that its three coordinates mean the
    same thing.

    `window` arrives as a boolean mask because what DEFINES it differs by
    category -- a field's extent for the two turbulence cases, the elevator pulse
    for the manoeuvre -- while what is done with it must not.

    `controls_hist` is per-sample, not one `Controls`, because a manoeuvre's
    elevator moves and `load_factor` needs the deflection that was actually
    flown at each sample.

    `load_model` is re-invoked per sample for exactly the reason the wind model
    is: this function receives a `State` trajectory, not a `SimState` one, so
    the increment cached on `SimState` is not in what it is handed. Omitting it
    would make every Fig. 8 load coordinate the point model's even on a run
    flown with strip loads -- and only the `CL` channel could reveal that, since
    `load_factor` inverts a force sum that `Cl`, `Cm` and `Cn` never enter.
    """

    def analyse(pos_ned, vel_body, quat, omega, controls):
        s = State(pos_ned=pos_ned, vel_body=vel_body, quat=quat, omega=omega)
        wind_ned, omega_gust, _, _, _ = model(
            wind.zero_wind_state(), s, jax.random.PRNGKey(0), jnp.array(dt)
        )
        increment = None if load_model is None else load_model(s)
        vel_rel = dynamics.relative_velocity(vel_body, quat, wind_ned)
        _, alpha_air, _ = air_data(vel_rel)
        _, alpha_inertial, _ = air_data(vel_body)
        _, theta, _ = quat_to_euler(quat)
        return jnp.array([
            -wind_ned[2], omega_gust[1], alpha_air, alpha_inertial, theta,
            omega[1],
            dynamics.load_factor(s, controls, ac, wind_ned, omega_gust, increment),
            controls.elevator,
        ])

    rows = np.asarray(jax.vmap(analyse)(
        hist.pos_ned, hist.vel_body, hist.quat, hist.omega, controls_hist
    ))
    n = rows.shape[0]
    return Encounter(
        label=label,
        t=np.arange(1, n + 1) * dt,
        north=np.asarray(hist.pos_ned)[:, 0],
        altitude=-np.asarray(hist.pos_ned)[:, 2],
        w_up=rows[:, 0], q_gust=rows[:, 1],
        alpha_air=rows[:, 2], alpha_inertial=rows[:, 3],
        theta=rows[:, 4], q=rows[:, 5], n_z=rows[:, 6],
        elevator=rows[:, 7],
        window=window,
        window_name=window_name,
        log=log,
    )


@partial(jax.jit, static_argnames=("n_steps",))
def _pulse_rollout(sim, ac, elev_trim, throttle, step, dt, n_steps, lead_in, hold):
    """Scan an elevator pulse: trim, `step` from trim for `hold`, trim again.

    Zero wind throughout -- a manoeuvre is the category the paper defines by the
    ABSENCE of turbulence. Returns the state history and the controls flown at
    each sample, stacked on a leading time axis like `integrate.rollout`.
    """

    def body(carry, i):
        t = i * dt
        pulsing = (t >= lead_in) & (t < lead_in + hold)
        controls = trim.trimmed_controls(
            elev_trim + jnp.where(pulsing, step, 0.0), throttle
        )
        carry = integrate.step(carry, controls, dt, ac, wind_model=wind.zero_wind)
        return carry, (carry.state, controls)

    _, out = jax.lax.scan(body, sim, jnp.arange(n_steps))
    return out


def _pushdown_setup(ac: Aircraft, airspeed: float, altitude: float):
    """Trim, and the SimState a pulse starts from. Shared by both entry points."""
    x, _ = trim.trim(jnp.array(airspeed), jnp.array(altitude), ac)
    state = trim.trimmed_state(
        jnp.array(float(x[0])), jnp.array(airspeed), jnp.array(altitude)
    )
    state = state._replace(pos_ned=jnp.array([0.0, 0.0, -altitude]))
    sim = integrate.init_sim(state, jax.random.PRNGKey(0))
    return sim, jnp.array(float(x[1])), jnp.array(float(x[2]))


def manoeuvre(
    ac: Aircraft,
    airspeed: float,
    altitude: float,
    *,
    label: str,
    elevator_step: float,
    hold: float,
    seconds: float,
    lead_in: float = 2.0,
    dt: float = 0.01,
) -> Encounter:
    """Fly an elevator pushdown at zero wind. The third Fig. 8 category.

    The sibling of `fly`, not a generalisation of it. `fly` holds the controls
    fixed on purpose -- the discriminator separates turbulence from MANOEUVRING
    by whether pitch correlates with elevator -- and a manoeuvre is precisely the
    case needing a time-varying elevator, which fixed `Controls` cannot express.
    `fly` is left alone deliberately: a schedule that happens to be constant is a
    strictly larger surface than a constant.

    `hold` is the pulse length, a DECLARED modelling choice in the same sense as
    the updraft's edge sharpness -- the paper constrains the load the pilot
    reached, not how long they took to reach it. It is also the analysis window,
    which is the rule the other two encounters already follow: the window is the
    disturbance's own extent (PROJECT.md section 8). Everything outside the pulse
    is flown so the recovery is on the record, and so the figure's whole-run
    marker means the same thing it means for the other two.

    `lead_in` is why `n_z[0]` is the TRIMMED load factor, which is what
    `fig8_point` measures the excursion from. Stepping the elevator at t=0 leaves
    the first sample already loaded and overstates the excursion by about 0.08 g
    -- the same shape of error as a too-short vortex lead-in (section 9,
    session 3), and the reason that one is 40 core radii.
    """
    sim, elev_trim, throttle = _pushdown_setup(ac, airspeed, altitude)
    n = int(round(seconds / dt))
    hist, controls_hist = _pulse_rollout(
        sim, ac, elev_trim, throttle, jnp.array(elevator_step),
        jnp.array(dt), n, jnp.array(lead_in), jnp.array(hold),
    )
    # Sample i is the state AFTER the step driven by the controls at t = i*dt, so
    # the samples actually flown under the pulse are (lead_in, lead_in + hold].
    t = np.arange(1, n + 1) * dt
    # Zero wind is a FACT about this run, not an absence of one -- the category is
    # defined by the absence of turbulence -- so the log records explicit zeros
    # rather than omitting the columns. A reader must be able to tell a still-air
    # run from an unrecorded one.
    zeros = np.zeros((n, 3))
    log = viz.Trajectory(
        t=t,
        pos_ned=np.asarray(hist.pos_ned, dtype=float),
        vel_body=np.asarray(hist.vel_body, dtype=float),
        quat=np.asarray(hist.quat, dtype=float),
        omega=np.asarray(hist.omega, dtype=float),
        controls=np.stack([np.asarray(c, dtype=float) for c in controls_hist], axis=1),
        mode=np.zeros(n, dtype=int),
        wind_ned=zeros,
        omega_gust=zeros,
    )
    return _measure(
        label=label, hist=hist, controls_hist=controls_hist,
        model=wind.zero_wind, ac=ac, dt=dt,
        window=(t > lead_in) & (t <= lead_in + hold),
        window_name=f"elevator pulse, {hold:g} s",
        log=log,
    )


def elevator_for_load(
    ac: Aircraft,
    airspeed: float,
    altitude: float,
    *,
    target: float,
    hold: float,
    seconds: float,
    lead_in: float = 2.0,
    dt: float = 0.01,
    bracket: tuple[float, float] = (0.0, 20.0),
    tolerance: float = 1e-5,
) -> float:
    """Bisect for the elevator step, in RADIANS from trim, reaching `target` g.

    The angle is derived, not chosen. The sourced quantity is the Fig. 8 load
    band, so the deflection that reaches it is an output -- which is the
    difference between "the model reaches the paper's load" and "the model was
    given the paper's answer".

    Bisection over a vmapped batch rather than a scalar one: each round evaluates
    the whole bracket at once and narrows it by the batch width, so a 20 deg
    bracket closes to 1e-5 deg in eight rounds instead of thirty-one. A Python
    loop calling `load_factor` sample by sample was tried first and is far too
    slow to sit in the analysis path.

    `target` must be reachable within `bracket`; the caller states the bracket
    because whether it is reachable at all is exactly the question section 5
    answers for the band's absolute reading (it is not).
    """
    sim, elev_trim, throttle = _pushdown_setup(ac, airspeed, altitude)
    n = int(round(seconds / dt))
    t = jnp.arange(1, n + 1) * dt
    mask = (t > lead_in) & (t <= lead_in + hold)
    zero3 = jnp.zeros(3)

    @jax.jit
    def excursion(steps):
        """Windowed load excursion from trim, for a batch of elevator steps."""

        def one(step):
            hist, controls = _pulse_rollout(
                sim, ac, elev_trim, throttle, step,
                jnp.array(dt), n, jnp.array(lead_in), jnp.array(hold),
            )
            n_z = jax.vmap(
                lambda p, v, q, w, c: dynamics.load_factor(
                    State(pos_ned=p, vel_body=v, quat=q, omega=w), c, ac, zero3, zero3
                )
            )(hist.pos_ned, hist.vel_body, hist.quat, hist.omega, controls)
            return jnp.min(jnp.where(mask, n_z, jnp.inf)) - n_z[0]

        return jax.vmap(one)(steps)

    width = 9  # candidates per round; the bracket narrows by (width - 1) each time
    low, high = bracket
    while high - low > tolerance:
        grid = jnp.linspace(low, high, width)
        # Monotone decreasing in the step, so the first candidate at or below the
        # target brackets it with its predecessor.
        reached = np.asarray(excursion(grid))
        below = reached <= target
        if not below.any():
            raise ValueError(
                f"load excursion {target} g is not reachable within {bracket} rad "
                f"of elevator: deepest reached {float(reached.min()):.4f} g"
            )
        if below[0]:
            # Already past the target at the bracket's low end, so the root is
            # below it and narrowing would converge on the endpoint and return a
            # confidently wrong angle. The band this feeds is compared against a
            # paper; silently wrong is the one outcome worth code to prevent.
            raise ValueError(
                f"load excursion {target} g is already exceeded at the low end of "
                f"{bracket} rad ({float(reached[0]):.4f} g): widen the bracket"
            )
        index = int(np.argmax(below))
        low, high = float(grid[index - 1]), float(grid[index])
    return 0.5 * (low + high)


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
    """Wingrove & Bach Fig. 8, with the model's points against the paper's three.

    All three categories now have a model point. The panel used to draw the
    manoeuvring slot as a labelled empty marker so the figure could not read as
    complete while it was not; that annotation is gone because the slot is
    filled. What it must NOT become is a claim of agreement -- section 5 forbids
    that outright, since Wingrove & Bach never identifies an aircraft type. The
    x-axis is therefore scaled to the DATA, not to the paper's range: the
    model's manoeuvring point sits far to the right of the paper's, and a chart
    cropped to the reference would hide that rather than show it.
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

    spread = list(FIG8_REFERENCE.values())
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
        spread += [dtheta, whole]

    ax.set_xlim(0.0, max(spread) * 1.12)
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
