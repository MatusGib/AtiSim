"""The trajectory log and the post-flight figure.

Two things live here: a `.npz` trajectory log that lets a run be re-analysed
without re-flying it, and the post-flight figure drawn from one. The live
instrument panel and the loop that drives it are in `atisim.panel`.

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

from atisim import earth
from atisim.integrate import SimState
from atisim.manual import Mode
from atisim.sensors import sense
from atisim.state import Controls, State
from atisim.state import pos_ned as state_pos_ned
from atisim.units import RAD2DEG

# ---------------------------------------------------------------------------
# Trajectory log
# ---------------------------------------------------------------------------


class Trajectory(NamedTuple):
    """A flown run, as numpy. Controls are stored in `Controls._fields` order.

    `wind_ned` and `omega_gust` are recorded because without them a saved run
    cannot be corrected even in principle: every incidence angle in it would be
    ground-relative with no way to recover the air-relative one.

    **THE RIGID-BODY COLUMNS ARE THE STATE AS FLOWN, NOT AN NED VIEW OF IT.**
    `pos_ecef` is the anchor-relative ECEF offset and `quat` rotates BODY into
    ECEF -- both exactly what `State` carried, stored without conversion. Two
    reasons, and the second is the one that decided it:

      - Rebuilding a `State` for `dynamics.specific_force` or `load_factor` is
        then free and exact. Storing the NED view would mean a
        `state_from_ned` round trip at every sample of every analysis.
      - `state_from_ned` NORMALISES the quaternion it builds. Recording an NED
        quaternion and rebuilding through it would therefore hand
        `checks.quaternion_norm` a unit quaternion by construction -- turning a
        tripwire that measures integrator drift off the unit sphere into a
        function that cannot fail. A check that cannot fail is worse than an
        absent one.

    `anchor` is what makes those columns mean anything: an offset from nowhere in
    particular is not a position. It is last so the field order of the array
    columns is unchanged, and it is the one non-array field -- `save` and `load`
    handle it as three scalars, since `r_ecef` and `T_e2l` follow from them.

    Local NED and geodetic altitude are DERIVED, by `pos_ned` and `derived`
    below. They are no longer stored, because on an ellipsoid they are two
    different quantities and storing one invites a reader to use it as the other.
    """

    t: np.ndarray  # (n,) s
    pos_ecef: np.ndarray  # (n, 3) m, OFFSET from `anchor`
    vel_body: np.ndarray  # (n, 3) m/s, ECEF-relative, body axes
    quat: np.ndarray  # (n, 4) body -> ECEF
    omega: np.ndarray  # (n, 3) rad/s, body rate relative to ECEF
    controls: np.ndarray  # (n, 4)
    mode: np.ndarray  # (n,) Mode
    wind_ned: np.ndarray  # (n, 3) m/s NED, the wind actually applied
    omega_gust: np.ndarray  # (n, 3) rad/s body, the gust rate actually applied
    anchor: earth.Anchor  # where the local frame is pinned. NOT an array column.


def states(traj: Trajectory) -> State:
    """The run's rigid-body states, as one batched `State`.

    Exact -- the columns ARE the state -- which is what lets `checks` invert the
    plant at every sample without a frame conversion in the way.
    """
    return State(
        pos_ecef=jnp.asarray(traj.pos_ecef),
        vel_body=jnp.asarray(traj.vel_body),
        quat=jnp.asarray(traj.quat),
        omega=jnp.asarray(traj.omega),
    )


def pos_ned(traj: Trajectory) -> np.ndarray:
    """(n, 3) local NED offset from the run anchor, m.

    A pure rotation of the stored offset, per `state.pos_ned`. NOT a source of
    altitude: `-pos_ned[:, 2]` is a tangent-plane height and departs from the
    geodetic one by 785 m at 100 km of ground track. Use `derived(traj).altitude`.
    """
    return np.asarray(
        jax.vmap(state_pos_ned, in_axes=(0, None))(states(traj), traj.anchor)
    )


def save(traj: Trajectory, path) -> None:
    """Write a run to .npz so it can be re-analysed without re-flying it.

    This matters more than it looks: comparing turbulence realisations means
    comparing runs against each other, and re-flying to change a plot loses the
    realisation unless the PRNG key is also pinned.

    The anchor is written as the three geodetic scalars it is defined by, not as
    its five fields: `r_ecef` and `T_e2l` are derived from (lat, lon, h) by
    `earth.anchor_at`, and storing a derived quantity is how a file ends up
    disagreeing with itself.
    """
    columns = {k: v for k, v in traj._asdict().items() if k != "anchor"}
    np.savez_compressed(
        path,
        anchor_lat=np.asarray(traj.anchor.lat, dtype=float),
        anchor_lon=np.asarray(traj.anchor.lon, dtype=float),
        anchor_h=np.asarray(traj.anchor.h, dtype=float),
        **columns,
    )


def load(path) -> Trajectory:
    """Read a run back.

    A file with no anchor columns is REFUSED rather than defaulted. Every `.npz`
    predating the ECEF state stores `pos_ned` and a body -> NED quaternion under
    the names this reader now gives to ECEF quantities, so loading one would not
    be a partial read -- it would be a full one, of the wrong frame, silently.
    And there is no anchor to supply on its behalf: `earth.py` ships no default
    precisely because latitude changes the answer.
    """
    with np.load(path) as data:
        if "anchor_lat" not in data:
            raise ValueError(
                f"{path} predates the ECEF state: its `pos_ned` and body->NED "
                "quaternion would be read as ECEF quantities and every angle in "
                "the run would be wrong without anything failing. Re-fly it."
            )
        fields = {
            name: data[name]
            for name in Trajectory._fields
            if name != "anchor" and name in data
        }
        n = len(fields["t"])
        for name in ("wind_ned", "omega_gust"):
            fields.setdefault(name, np.zeros((n, 3)))
        return Trajectory(
            **fields,
            anchor=earth.anchor_at(
                data["anchor_lat"], data["anchor_lon"], data["anchor_h"]
            ),
        )


class Recorder:
    """Full-rate log. One row per physics step, not per rendered frame.

    Takes the run's `anchor` at construction: it is constant for the run, so
    asking for it once is enough, and a recorder that could not say where its
    rows were flown would produce a trajectory nothing could interpret.
    """

    def __init__(self, anchor: earth.Anchor) -> None:
        self._anchor = anchor
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
                np.asarray(state.pos_ecef, dtype=float),
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
            pos_ecef=np.stack(columns[1]),
            vel_body=np.stack(columns[2]),
            quat=np.stack(columns[3]),
            omega=np.stack(columns[4]),
            controls=np.stack(columns[5]),
            mode=np.array(columns[6], dtype=int),
            wind_ned=np.stack(columns[7]),
            omega_gust=np.stack(columns[8]),
            anchor=self._anchor,
        )


class Derived(NamedTuple):
    """Air-relative incidence and inertial attitude, over a whole trajectory."""

    airspeed: np.ndarray  # m/s, AIR-RELATIVE
    alpha: np.ndarray  # rad, AIR-RELATIVE
    beta: np.ndarray  # rad, AIR-RELATIVE
    phi: np.ndarray  # rad, inertial
    theta: np.ndarray  # rad, inertial
    psi: np.ndarray  # rad, inertial
    altitude: np.ndarray  # m, GEODETIC


def derived(traj: Trajectory) -> Derived:
    """Sensed air data for a whole trajectory, in one vmap.

    Uses the recorded wind, so the incidence angles are air-relative. Before the
    wind was recorded these were ground-relative and wrong by up to 7 deg in a
    vortex encounter, and no still-air test could see it.

    The anchor comes from the trajectory rather than from the caller, which is
    the point of it travelling with the run: the attitude angles and the altitude
    below are all referred to the local vertical, and being asked for it a second
    time is how an analysis ends up using a different one from the flight.
    """
    air = jax.vmap(sense, in_axes=(0, None, 0))(
        states(traj), traj.anchor, jnp.asarray(traj.wind_ned)
    )
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
    track = pos_ned(traj)
    east, north = track[:, 1] / 1000.0, track[:, 0] / 1000.0
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
