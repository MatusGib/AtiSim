"""The trajectory log and the post-flight figure.

Two things live here: a `.npz` trajectory log that lets a run be re-analysed
without re-flying it, and the post-flight figure drawn from one. The live
instrument panel and the loop that drives it are in `flightsim.panel`.

Keeping them apart matters for one specific reason: the log has to be readable
with no simulator in sight -- `load` then `post_flight` must work on a machine
that never had an aircraft definition -- and the panel drags in the integrator,
both controllers and an animation timer.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from typing import NamedTuple

from flightsim.integrate import SimState
from flightsim.manual import Mode
from flightsim.sensors import sense
from flightsim.state import Controls, State
from flightsim.units import RAD2DEG

# ---------------------------------------------------------------------------
# Trajectory log
# ---------------------------------------------------------------------------


class Trajectory(NamedTuple):
    """A flown run, as numpy. Controls are stored in `Controls._fields` order.

    `wind_ned` and `omega_gust` are recorded because without them a saved run
    cannot be corrected even in principle: every incidence angle in it would be
    ground-relative with no way to recover the air-relative one. They are last so
    that `.npz` files written before they existed still load -- see `load`.
    """

    t: np.ndarray  # (n,) s
    pos_ned: np.ndarray  # (n, 3) m
    vel_body: np.ndarray  # (n, 3) m/s
    quat: np.ndarray  # (n, 4)
    omega: np.ndarray  # (n, 3) rad/s
    controls: np.ndarray  # (n, 4)
    mode: np.ndarray  # (n,) Mode
    wind_ned: np.ndarray  # (n, 3) m/s NED, the wind actually applied
    omega_gust: np.ndarray  # (n, 3) rad/s body, the gust rate actually applied


def save(traj: Trajectory, path) -> None:
    """Write a run to .npz so it can be re-analysed without re-flying it.

    This matters more than it looks: comparing turbulence realisations means
    comparing runs against each other, and re-flying to change a plot loses the
    realisation unless the PRNG key is also pinned.
    """
    np.savez_compressed(path, **traj._asdict())


def load(path) -> Trajectory:
    """Read a run back.

    Files written before wind was recorded load as still air rather than being
    rejected. That is honest for those files -- every run predating the wind
    columns was flown in still air because nothing else was possible -- and it
    keeps them analysable instead of stranding them.
    """
    with np.load(path) as data:
        fields = {name: data[name] for name in Trajectory._fields if name in data}
        n = len(fields["t"])
        for name in ("wind_ned", "omega_gust"):
            fields.setdefault(name, np.zeros((n, 3)))
        return Trajectory(**fields)


class Recorder:
    """Full-rate log. One row per physics step, not per rendered frame."""

    def __init__(self) -> None:
        self._rows: list[tuple] = []

    def append(
        self,
        t: float,
        sim: SimState,
        controls: Controls,
        mode: Mode,
    ) -> None:
        """Log one physics step.

        Takes the whole `SimState` rather than just `State` so the wind that was
        actually applied is recorded alongside the motion it caused. Passing only
        the rigid-body state is what made saved runs uncorrectable.
        """
        state = sim.state
        self._rows.append(
            (
                t,
                np.asarray(state.pos_ned, dtype=float),
                np.asarray(state.vel_body, dtype=float),
                np.asarray(state.quat, dtype=float),
                np.asarray(state.omega, dtype=float),
                np.array([float(c) for c in controls]),
                int(mode),
                np.asarray(sim.wind_ned, dtype=float),
                np.asarray(sim.omega_gust, dtype=float),
            )
        )

    def trajectory(self) -> Trajectory:
        columns = list(zip(*self._rows))
        return Trajectory(
            t=np.array(columns[0]),
            pos_ned=np.stack(columns[1]),
            vel_body=np.stack(columns[2]),
            quat=np.stack(columns[3]),
            omega=np.stack(columns[4]),
            controls=np.stack(columns[5]),
            mode=np.array(columns[6], dtype=int),
            wind_ned=np.stack(columns[7]),
            omega_gust=np.stack(columns[8]),
        )


class Derived(NamedTuple):
    """Air-relative incidence and inertial attitude, over a whole trajectory."""

    airspeed: np.ndarray  # m/s, AIR-RELATIVE
    alpha: np.ndarray  # rad, AIR-RELATIVE
    beta: np.ndarray  # rad, AIR-RELATIVE
    phi: np.ndarray  # rad, inertial
    theta: np.ndarray  # rad, inertial
    psi: np.ndarray  # rad, inertial
    altitude: np.ndarray  # m


def derived(traj: Trajectory) -> Derived:
    """Sensed air data for a whole trajectory, in one vmap.

    Uses the recorded wind, so the incidence angles are air-relative. Before the
    wind was recorded these were ground-relative and wrong by up to 7 deg in a
    vortex encounter, and no still-air test could see it.
    """
    states = State(
        pos_ned=jnp.asarray(traj.pos_ned),
        vel_body=jnp.asarray(traj.vel_body),
        quat=jnp.asarray(traj.quat),
        omega=jnp.asarray(traj.omega),
    )
    air = jax.vmap(sense)(states, jnp.asarray(traj.wind_ned))
    return Derived(
        airspeed=np.asarray(air.airspeed),
        alpha=np.asarray(air.alpha),
        beta=np.asarray(air.beta),
        phi=np.asarray(air.phi),
        theta=np.asarray(air.theta),
        psi=np.asarray(air.psi),
        altitude=np.asarray(air.altitude),
    )


# ---------------------------------------------------------------------------
# Post-flight
# ---------------------------------------------------------------------------


def post_flight(traj: Trajectory, *, title: str = "post-flight"):
    """Ground track, altitude, airspeed/alpha/beta, deflections, mode timeline."""
    d = derived(traj)
    t = traj.t
    autopilot_engaged = traj.mode == int(Mode.AUTOPILOT)

    fig = plt.figure(figsize=(14.0, 9.0))
    fig.suptitle(title)
    grid = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.22)

    ax = fig.add_subplot(grid[0:2, 0])
    east, north = traj.pos_ned[:, 1] / 1000.0, traj.pos_ned[:, 0] / 1000.0
    ax.plot(east, north, color="0.6", lw=1.2, label="track")
    ax.plot(
        np.ma.masked_where(~autopilot_engaged, east),
        np.ma.masked_where(~autopilot_engaged, north),
        color="C0",
        lw=1.6,
        label="autopilot",
    )
    ax.plot(east[0], north[0], "o", color="C2", ms=6, label="start")
    ax.plot(east[-1], north[-1], "s", color="C3", ms=6, label="end")
    ax.set_xlabel("east  km")
    ax.set_ylabel("north  km")
    ax.set_title("ground track")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(grid[0, 1])
    ax.plot(t, d.altitude, color="C0")
    ax.set_ylabel("altitude  m")
    ax.set_title("altitude profile")
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(grid[1, 1])
    ax.plot(t, d.airspeed, color="C0", label="TAS")
    ax.set_ylabel("TAS  m/s", color="C0")
    ax.tick_params(axis="y", labelcolor="C0")
    ax.grid(alpha=0.3)
    twin = ax.twinx()
    twin.plot(t, d.alpha * RAD2DEG, color="C1", lw=1.0, label="alpha")
    twin.plot(t, d.beta * RAD2DEG, color="C3", lw=1.0, label="beta")
    twin.set_ylabel("alpha, beta  deg")
    twin.legend(fontsize=8, loc="upper right")
    ax.set_title("airspeed and incidence")

    ax = fig.add_subplot(grid[2, 0])
    for index, name in enumerate(Controls._fields[:3]):
        ax.plot(t, traj.controls[:, index] * RAD2DEG, lw=1.0, label=name)
    ax.set_ylabel("deflection  deg")
    ax.set_xlabel("time  s")
    ax.grid(alpha=0.3)
    twin = ax.twinx()
    twin.plot(t, traj.controls[:, 3], color="0.4", lw=1.0, label="throttle")
    twin.set_ylabel("throttle")
    twin.set_ylim(0.0, 1.0)
    handles = ax.get_legend_handles_labels()[0] + twin.get_legend_handles_labels()[0]
    ax.legend(handles, [h.get_label() for h in handles], fontsize=8, ncol=2)
    ax.set_title("control deflections")

    ax = fig.add_subplot(grid[2, 1])
    ax.step(t, traj.mode, where="post", color="C0", lw=1.4)
    ax.fill_between(t, 0, 1, where=autopilot_engaged, step="post", alpha=0.2, color="C0")
    ax.set_yticks([0, 1])
    ax.set_yticklabels([Mode.MANUAL.name, Mode.AUTOPILOT.name])
    ax.set_ylim(-0.2, 1.2)
    ax.set_xlabel("time  s")
    ax.set_title("mode timeline")
    ax.grid(alpha=0.3)

    return fig
