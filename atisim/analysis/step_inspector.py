"""The step inspector: one logged step, flown again and opened up (plan section 3.3).

At a sample the inspector gives the state, the controls, and the four RK4
stages of the step that leaves it: where each stage evaluates, the wind the
field has there, the wind the derivatives were given (the same on a
stage-sampled run, the step's held sample otherwise), and the derivatives.

Two numbers say whether to trust what it shows:

- **Re-step error.** The step is flown again with `integrate.step` from the
  logged state, with the run's own options rebuilt from `meta.json`, and
  compared with the next logged sample. Zero (to round-off) means the rebuild
  is the run.
- **Refinement difference.** The same step flown as `refine` steps of dt/refine.
  The difference estimates the step's local truncation error.

The stages are computed here by the same RK4 combination `integrate.rk4_step`
uses, from the same right-hand side `integrate.step` builds, so they can be
shown. `test_diagnostics.py` holds their sum to `integrate.step`.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from atisim import airframe, integrate, loads, wind
from atisim.aircraft import REGISTRY, Aircraft
from atisim.dynamics import derivatives, relative_velocity
from atisim.state import Controls, State, quat_normalize, quat_to_euler

STAGE_FRACTIONS = (0.0, 0.5, 0.5, 1.0)


class Flight(NamedTuple):
    """How a run was flown, rebuilt from its `meta.json`."""

    ac: Aircraft
    field: object
    model: object
    load_model: object
    stage_sampled: bool
    gust_lag: bool
    dt: float


def flight(meta: dict) -> Flight:
    from atisim.analysis import artifact

    ac = REGISTRY[meta["aircraft"]["key"]]
    field = artifact.rebuild_field(meta)
    integrator = meta["integrator"]
    tail = integrator.get("wing_tail")
    if meta["wind_field"]["kind"] == "none (zero wind)":
        model = wind.zero_wind  # the manoeuvre: still air by definition
    elif tail:
        arm = tail["tail_arm_m"] if tail.get("source") == "JSBSim <htailarm>" else None
        model = wind.sampled_field_model(field, airframe.stations(ac, n_lon=2, tail_arm=arm))
    else:
        model = wind.field_model(field)
    return Flight(
        ac=ac, field=field, model=model,
        load_model=(loads.strip_model(field, ac)
                    if meta.get("load_model") == "loads.strip_model" else None),
        stage_sampled=integrator.get("stage_sampled", True) is not False,
        gust_lag="gust_lag" in integrator,
        dt=float(integrator["dt_s"]),
    )


class Stage(NamedTuple):
    fraction: float  # of dt
    pos_ned: np.ndarray
    wind_field: np.ndarray  # the field at the stage's position
    wind_used: np.ndarray  # what the derivatives were given
    alphadot_gust: float  # rad/s
    vel_dot: np.ndarray  # m/s^2, body
    omega_dot: np.ndarray  # rad/s^2


class StepDetail(NamedTuple):
    index: int
    t: float
    state: dict  # name -> value, SI with angles in rad
    controls: dict
    stages: list
    restep_error: dict  # |re-stepped - logged next| per state group
    refine_difference: dict  # |one step - refine steps| per state group
    refine: int
    note: str


def _sim(traj, i, lag):
    s = State(pos_ned=jnp.asarray(traj.pos_ned[i]), vel_body=jnp.asarray(traj.vel_body[i]),
              quat=jnp.asarray(traj.quat[i]), omega=jnp.asarray(traj.omega[i]))
    sim = integrate.init_sim(s, jax.random.PRNGKey(0))
    return sim if lag is None else sim._replace(gust_lag=jnp.asarray(lag[i]))


def _groups(a: State, b: State) -> dict:
    return {
        "position (m)": float(jnp.max(jnp.abs(a.pos_ned - b.pos_ned))),
        "velocity (m/s)": float(jnp.max(jnp.abs(a.vel_body - b.vel_body))),
        "attitude quaternion": float(jnp.max(jnp.abs(a.quat - b.quat))),
        "body rates (rad/s)": float(jnp.max(jnp.abs(a.omega - b.omega))),
    }


def _step(fl: Flight, sim, controls, dt):
    return integrate.step(sim, controls, jnp.asarray(dt), fl.ac, wind_model=fl.model,
                          load_model=fl.load_model, stage_sampled=fl.stage_sampled,
                          gust_lag=fl.gust_lag)


def stages(fl: Flight, sim, controls: Controls):
    """The four RK4 stages of the step from `sim`, and the state they produce."""
    ac, dt = fl.ac, fl.dt
    produced = fl.model(sim.wind, sim.state, sim.key, jnp.asarray(dt))
    held_wind, held_og = produced[0], produced[1]
    held_ad = produced[4] if len(produced) == 5 else jnp.array(0.0)
    increment = None if fl.load_model is None else fl.load_model(sim.state)
    field = getattr(fl.model, "field", None) if fl.stage_sampled else None
    stations = getattr(fl.model, "stations", None)

    def rates(s):
        if stations is None:
            return wind.gust_rates(s.pos_ned, s.quat, field)
        return wind.sampled_rates(s.pos_ned, s.quat, field, stations)

    def f(y):
        """(derivative of y, the field's wind, the wind used, alphadot_gust)."""
        s, lag = y
        if fl.gust_lag:
            w = field(s.pos_ned)
            airspeed = jnp.linalg.norm(relative_velocity(s.vel_body, s.quat, w))
            lag_dot = wind.kussner_lag_rate(lag, w[2], airspeed, ac.c)
            ad = wind.lagged_gust_alphadot(s.pos_ned, s.quat, s.vel_body, field, lag, lag_dot)
            used = wind.lagged_wind(w, lag)
            ds = derivatives(s, controls, ac, used, rates(s), increment=increment,
                             alphadot_gust=ad)
            return (ds, lag_dot), w, used, ad
        if field is None:
            ds = derivatives(s, controls, ac, held_wind, held_og, increment=increment,
                             alphadot_gust=held_ad)
            w = fl.field(s.pos_ned)
            return (ds, lag), w, held_wind, held_ad
        w = field(s.pos_ned)
        ad = wind.gust_alphadot(s.pos_ned, s.quat, s.vel_body, field)
        ds = derivatives(s, controls, ac, w, rates(s), increment=increment,
                         alphadot_gust=ad)
        return (ds, lag), w, w, ad

    def axpy(x, y, a):
        return jax.tree.map(lambda xi, yi: xi + a * yi, x, y)

    lag0 = sim.gust_lag if fl.gust_lag else jnp.zeros(2)
    y = (sim.state, lag0)
    out, ks, x = [], [], y
    for fraction in STAGE_FRACTIONS:
        x = y if fraction == 0.0 else axpy(y, ks[-1], fraction * dt)
        k, w, used, ad = f(x)
        ks.append(k)
        out.append(Stage(fraction, np.asarray(x[0].pos_ned), np.asarray(w), np.asarray(used),
                         float(ad), np.asarray(k[0].vel_body), np.asarray(k[0].omega)))
    k1, k2, k3, k4 = ks
    increment_ = jax.tree.map(lambda a, b, c, d: (a + 2.0 * b + 2.0 * c + d) / 6.0,
                              k1, k2, k3, k4)
    new_state, _ = axpy(y, increment_, dt)
    return out, new_state._replace(quat=quat_normalize(new_state.quat))


def detail(traj, meta: dict, index: int, lag=None, refine: int = 10) -> StepDetail:
    """The step from sample `index` to the next. `lag` is the run's (n, 2) Kussner
    lag states (`diagnostics.parquet` has them on a High run)."""
    fl = flight(meta)
    n = len(traj.t)
    index = int(min(max(index, 0), n - 2))
    if fl.gust_lag and lag is None:
        raise ValueError("A gust-lag step needs the lag states, which a High "
                         "fidelity run keeps. Fly the run at High.")
    sim = _sim(traj, index, lag)
    c = np.asarray(traj.controls[index + 1])
    controls = Controls(*(jnp.asarray(v) for v in c))
    logged_next = State(pos_ned=jnp.asarray(traj.pos_ned[index + 1]),
                        vel_body=jnp.asarray(traj.vel_body[index + 1]),
                        quat=jnp.asarray(traj.quat[index + 1]),
                        omega=jnp.asarray(traj.omega[index + 1]))
    restepped = _step(fl, sim, controls, fl.dt)
    fine = sim
    for _ in range(refine):
        fine = _step(fl, fine, controls, fl.dt / refine)
    stage_list, _ = stages(fl, sim, controls)

    phi, theta, psi = (float(v) for v in quat_to_euler(sim.state.quat))
    s = sim.state
    state = {
        "north (m)": float(s.pos_ned[0]), "east (m)": float(s.pos_ned[1]),
        "altitude (m)": float(-s.pos_ned[2]),
        "u (m/s)": float(s.vel_body[0]), "v (m/s)": float(s.vel_body[1]),
        "w (m/s)": float(s.vel_body[2]),
        "phi (deg)": np.degrees(phi), "theta (deg)": np.degrees(theta),
        "psi (deg)": np.degrees(psi),
        "p (deg/s)": float(np.degrees(s.omega[0])), "q (deg/s)": float(np.degrees(s.omega[1])),
        "r (deg/s)": float(np.degrees(s.omega[2])),
    }
    names = ("elevator (deg)", "aileron (deg)", "rudder (deg)")
    controls_out = {name: float(np.degrees(v)) for name, v in zip(names, c[:3])}
    controls_out["throttle (-)"] = float(c[3])
    how = ("stage-sampled: each stage reads the field at its own position"
           if fl.stage_sampled else
           "held: every stage uses the wind sampled at the step's start")
    if fl.gust_lag:
        how += "; the Kussner lag filters the vertical gust"
    return StepDetail(
        index=index, t=float(traj.t[index]), state=state, controls=controls_out,
        stages=stage_list, restep_error=_groups(restepped.state, logged_next),
        refine_difference=_groups(restepped.state, fine.state), refine=refine,
        note=f"Step {index} to {index + 1}, dt {fl.dt:g} s. Wind {how}.",
    )
