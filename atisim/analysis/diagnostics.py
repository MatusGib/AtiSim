"""High fidelity: every quantity the plant computed, per sample, after the flight.

A Standard run keeps the state history and the wind the aircraft flew. A High
run adds `diagnostics.parquet` beside `run.parquet`: air data, every
aerodynamic coefficient split into its terms (`aero.coefficient_terms`), the
forces and moments, the state derivatives, the three halves of the
angle-of-attack rate, the specific force, and the energy budget -- one row per
logged sample.

**The flight is not touched (plan decision D3).** This is a PROBE PASS: after
the flight, the engine's own functions are vmapped over the logged states, the
way `vortex_viz._measure` already re-evaluates the wind model to get n_z. The
hot path is unchanged, so `run.parquet` is bit-identical at Standard and High;
`test_diagnostics.py` asserts it.

`probe` mirrors `dynamics.derivatives` step by step to expose its
intermediates (the open-loop acceleration, the aircraft's alphadot), which
`derivatives` computes and discards. A mirror can drift, so the same test holds
it to the plant: the probe's accelerations equal `dynamics.derivatives` and its
n_z equals the run's to round-off, and the coefficient terms sum to the forces.

What a sample means. The logged state is the state AFTER each step. The wind
here is the model re-evaluated at that state, as `_measure` does for n_z, so on
a stage-sampled run it is the air the next step's first stage meets. On a gust
lag run the lagged gust comes from the logged lag states when the flight kept
them (`Encounter.sim_log`).

`SCHEMA_VERSION` stays 1 (plan decision D5): this is a new, optional file.
"""

import json
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from atisim import aero, dynamics, wind
from atisim.aircraft import Aircraft
from atisim.atmosphere import G0, density, speed_of_sound
from atisim.state import Controls, State, quat_to_dcm, quat_to_euler

FILENAME = "diagnostics.parquet"

_COEFFICIENTS = ("CL", "CD", "CY", "Cl", "Cm", "Cn")


class Channel(NamedTuple):
    name: str
    group: str
    unit: str
    description: str


def _channels(term_names: dict[str, list[str]], strip: bool,
              lag: bool = False) -> list[Channel]:
    out = [
        Channel("t", "time", "s", "time of the logged sample (after the step)"),
        Channel("north", "state", "m", "position north"),
        Channel("east", "state", "m", "position east"),
        Channel("altitude", "state", "m", "altitude, -pos_d"),
        Channel("u", "state", "m/s", "body velocity, x"),
        Channel("v", "state", "m/s", "body velocity, y"),
        Channel("w", "state", "m/s", "body velocity, z"),
        Channel("phi", "state", "deg", "bank"),
        Channel("theta", "state", "deg", "pitch attitude"),
        Channel("psi", "state", "deg", "heading"),
        Channel("p", "state", "deg/s", "body roll rate"),
        Channel("q", "state", "deg/s", "body pitch rate"),
        Channel("r", "state", "deg/s", "body yaw rate"),
        Channel("quat_norm_error", "state", "-", "| |quaternion| - 1 |"),
        Channel("elevator", "controls", "deg", "elevator flown"),
        Channel("aileron", "controls", "deg", "aileron flown"),
        Channel("rudder", "controls", "deg", "rudder the aerodynamics saw, yaw damper included"),
        Channel("throttle", "controls", "-", "throttle, 0 to 1"),
        Channel("airspeed", "air data", "m/s", "true airspeed, air-relative"),
        Channel("alpha", "air data", "deg", "angle of attack, air-relative"),
        Channel("beta", "air data", "deg", "sideslip, air-relative"),
        Channel("mach", "air data", "-", "Mach, air-relative"),
        Channel("qbar", "air data", "Pa", "dynamic pressure"),
        Channel("rho", "air data", "kg/m^3", "density"),
        Channel("wind_n", "wind", "m/s", "wind north, at the sample"),
        Channel("wind_e", "wind", "m/s", "wind east"),
        Channel("wind_d", "wind", "m/s", "wind down, as the aerodynamics saw it"),
        Channel("gust_p", "wind", "deg/s", "rolling gust rate"),
        Channel("gust_q", "wind", "deg/s", "pitching gust rate"),
        Channel("gust_r", "wind", "deg/s", "yawing gust rate"),
        Channel("alphadot_gust", "wind", "deg/s",
                "angle-of-attack rate from the field's gradient"),
        Channel("alphadot_transport", "wind", "deg/s",
                "angle-of-attack rate from the body turning under the wind"),
        Channel("alphadot_aircraft", "wind", "deg/s",
                "the aircraft's own angle-of-attack rate (one pass, as derivatives)"),
    ]
    if lag:
        out += [Channel(f"gust_lag_{i}", "wind", "m/s",
                        f"Kussner lag state {i} (Jones), NED down; the lagged gust is "
                        "their weighted sum") for i in (1, 2)]
    for coeff in _COEFFICIENTS:
        for term in term_names[coeff]:
            out.append(Channel(f"{coeff}.{term}", f"coefficients: {coeff}", "-",
                               f"{coeff}, {term} term (aero.coefficient_terms)"))
        if strip and coeff in ("CL", "Cl", "Cm", "Cn"):
            out.append(Channel(f"{coeff}.strip", f"coefficients: {coeff}", "-",
                               f"{coeff}, strip-load increment (loads.strip_model)"))
        out.append(Channel(coeff, f"coefficients: {coeff}", "-", f"{coeff}, total"))
    out += [
        Channel("Fx_aero", "forces", "N", "aerodynamic force, body x"),
        Channel("Fy_aero", "forces", "N", "aerodynamic force, body y"),
        Channel("Fz_aero", "forces", "N", "aerodynamic force, body z"),
        Channel("Fx_thrust", "forces", "N", "thrust, body x"),
        Channel("Fz_thrust", "forces", "N", "thrust, body z"),
        Channel("Fx_gravity", "forces", "N", "weight, body x"),
        Channel("Fy_gravity", "forces", "N", "weight, body y"),
        Channel("Fz_gravity", "forces", "N", "weight, body z"),
        Channel("L_moment", "moments", "N.m", "rolling moment about the CG"),
        Channel("M_moment", "moments", "N.m", "pitching moment about the CG"),
        Channel("N_moment", "moments", "N.m", "yawing moment about the CG"),
        Channel("udot", "derivatives", "m/s^2", "d(u)/dt, dynamics.derivatives"),
        Channel("vdot", "derivatives", "m/s^2", "d(v)/dt"),
        Channel("wdot", "derivatives", "m/s^2", "d(w)/dt"),
        Channel("pdot", "derivatives", "deg/s^2", "d(p)/dt"),
        Channel("qdot", "derivatives", "deg/s^2", "d(q)/dt"),
        Channel("rdot", "derivatives", "deg/s^2", "d(r)/dt"),
        Channel("n_x", "load", "g", "specific force, body x"),
        Channel("n_y", "load", "g", "specific force, body y"),
        Channel("n_z", "load", "g", "normal load factor, as the run records it"),
        Channel("kinetic", "energy", "J", "kinetic energy, 0.5 m |v|^2 (inertial)"),
        Channel("potential", "energy", "J", "potential energy, m g0 H (geopotential)"),
        Channel("energy", "energy", "J", "kinetic plus potential energy"),
        Channel("work", "energy", "J", "work done by the non-conservative forces, "
                                        "trapezoid in time (checks._energy_and_power)"),
        Channel("energy_residual", "energy", "J", "(E - E0) - work: what the "
                                                  "energy-closure gate measures"),
    ]
    return out


def probe(traj, ac: Aircraft, field, model=None, load_model=None, lag=None):
    """Every channel for every logged sample, as {name: array}, and the channels.

    `traj` is the run's `viz.Trajectory`; `model` is the wind model it flew
    (default `wind.field_model(field)`); `load_model` is the strip model if the
    run used one; `lag` is the (n, 2) Kussner lag states if the run had the lag
    (`Encounter.gust_lag`).
    """
    from atisim import checks

    model = wind.field_model(field) if model is None else model
    key = jax.random.PRNGKey(0)

    def one(pos, vel, quat, omega, controls_vec, lag_state):
        s = State(pos_ned=pos, vel_body=vel, quat=quat, omega=omega)
        controls = Controls(elevator=controls_vec[0], aileron=controls_vec[1],
                            rudder=controls_vec[2], throttle=controls_vec[3])
        field_wind, omega_gust, _, _, *rest = model(wind.zero_wind_state(), s, key,
                                                    jnp.array(0.0))
        if lag is None:
            wind_ned = field_wind
            ad_gust = rest[0] if rest else jnp.array(0.0)
        else:
            # `integrate.step`'s gust-lag stage: the lag's rate is driven by the
            # field's own gust at the unlagged airspeed; the aerodynamics see
            # the lagged gust and its rate.
            stage_field = model.field
            airspeed = jnp.linalg.norm(dynamics.relative_velocity(vel, quat, field_wind))
            lag_dot = wind.kussner_lag_rate(lag_state, field_wind[2], airspeed, ac.c)
            wind_ned = wind.lagged_wind(field_wind, lag_state)
            ad_gust = wind.lagged_gust_alphadot(pos, quat, vel, stage_field, lag_state,
                                                lag_dot)
        increment = None if load_model is None else load_model(s)

        # --- dynamics.derivatives, with its intermediates kept -------------
        dcm = quat_to_dcm(quat)
        vel_rel = dynamics.relative_velocity(vel, quat, wind_ned)
        omega_rel = omega - omega_gust
        altitude = -pos[2]
        rho = density(altitude)
        a_sound = speed_of_sound(altitude)
        mach = jnp.linalg.norm(vel_rel) / a_sound
        thrust = aero.thrust_force(controls, ac, rho, mach)
        gravity_body = dcm.T @ jnp.array([0.0, 0.0, dynamics.gravity(altitude)])
        surfaces = controls._replace(
            rudder=dynamics.yaw_damper_rudder(controls, ac, omega[2], mach))
        u_rel, w_rel = vel_rel[0], vel_rel[2]
        denominator = jnp.maximum(u_rel**2 + w_rel**2, aero.V_MIN**2)
        transport = jnp.cross(omega, dcm.T @ wind_ned)
        ad_transport = (u_rel * transport[2] - w_rel * transport[0]) / denominator
        ad_wind = ad_gust + ad_transport
        f_open, _ = aero.aero_forces_moments(vel_rel, omega_rel, surfaces, ac, rho,
                                             a_sound, increment=increment,
                                             alphadot_gust=ad_wind)
        accel_open = (f_open + thrust) / ac.mass + gravity_body - jnp.cross(omega, vel)
        ad_aircraft = (u_rel * accel_open[2] - w_rel * accel_open[0]) / denominator
        ad_total = ad_wind + ad_aircraft
        force, moment = aero.aero_forces_moments(vel_rel, omega_rel, surfaces, ac, rho,
                                                 a_sound, increment=increment,
                                                 alphadot_gust=ad_total)
        moment = jnp.where(ac.thrust_arm != 0.0, moment + aero.thrust_moment(thrust, ac),
                           moment)
        terms = aero.coefficient_terms(vel_rel, omega_rel, surfaces, ac, a_sound, ad_total)
        totals = aero.coefficients(vel_rel, omega_rel, surfaces, ac, a_sound, ad_total)
        d = dynamics.derivatives(s, controls, ac, wind_ned, omega_gust,
                                 increment=increment, alphadot_gust=ad_gust)
        spec_force = dynamics.specific_force(s, controls, ac, wind_ned, omega_gust,
                                             increment)
        n_z = dynamics.load_factor(s, controls, ac, wind_ned, omega_gust, increment)
        V, alpha, beta = aero.air_data(vel_rel)
        phi, theta, psi = quat_to_euler(quat)
        coeff = {}
        for name, total in zip(_COEFFICIENTS, totals):
            coeff[name] = total + (getattr(increment, name) if increment is not None
                                   and name in ("CL", "Cl", "Cm", "Cn") else 0.0)
        out = {
            "north": pos[0], "east": pos[1], "altitude": altitude,
            "u": vel[0], "v": vel[1], "w": vel[2],
            "phi": jnp.degrees(phi), "theta": jnp.degrees(theta), "psi": jnp.degrees(psi),
            "p": jnp.degrees(omega[0]), "q": jnp.degrees(omega[1]),
            "r": jnp.degrees(omega[2]),
            "quat_norm_error": jnp.abs(jnp.linalg.norm(quat) - 1.0),
            "elevator": jnp.degrees(controls.elevator),
            "aileron": jnp.degrees(controls.aileron),
            "rudder": jnp.degrees(surfaces.rudder), "throttle": controls.throttle,
            "airspeed": jnp.linalg.norm(vel_rel), "alpha": jnp.degrees(alpha),
            "beta": jnp.degrees(beta), "mach": mach,
            "qbar": 0.5 * rho * jnp.linalg.norm(vel_rel) ** 2, "rho": rho,
            "wind_n": wind_ned[0], "wind_e": wind_ned[1], "wind_d": wind_ned[2],
            "gust_p": jnp.degrees(omega_gust[0]), "gust_q": jnp.degrees(omega_gust[1]),
            "gust_r": jnp.degrees(omega_gust[2]),
            "alphadot_gust": jnp.degrees(ad_gust),
            "alphadot_transport": jnp.degrees(ad_transport),
            "alphadot_aircraft": jnp.degrees(ad_aircraft),
            "Fx_aero": force[0], "Fy_aero": force[1], "Fz_aero": force[2],
            "Fx_thrust": thrust[0], "Fz_thrust": thrust[2],
            "Fx_gravity": ac.mass * gravity_body[0], "Fy_gravity": ac.mass * gravity_body[1],
            "Fz_gravity": ac.mass * gravity_body[2],
            "L_moment": moment[0], "M_moment": moment[1], "N_moment": moment[2],
            "udot": d.vel_body[0], "vdot": d.vel_body[1], "wdot": d.vel_body[2],
            "pdot": jnp.degrees(d.omega[0]), "qdot": jnp.degrees(d.omega[1]),
            "rdot": jnp.degrees(d.omega[2]),
            "n_x": spec_force[0], "n_y": spec_force[1], "n_z": n_z,
        }
        for name in _COEFFICIENTS:
            for term, value in terms[name].items():
                out[f"{name}.{term}"] = value
            if increment is not None and name in ("CL", "Cl", "Cm", "Cn"):
                out[f"{name}.strip"] = getattr(increment, name)
            out[name] = coeff[name]
        # The acceleration rebuilt from the forces, to hold the mirror to the plant.
        out["_accel_from_forces"] = ((force + thrust) / ac.mass + gravity_body
                                     - jnp.cross(omega, vel))
        return out

    n = len(traj.t)
    lags = jnp.zeros((n, 2)) if lag is None else jnp.asarray(lag)
    rows = jax.vmap(one)(jnp.asarray(traj.pos_ned), jnp.asarray(traj.vel_body),
                         jnp.asarray(traj.quat), jnp.asarray(traj.omega),
                         jnp.asarray(traj.controls), lags)
    columns = {k: np.asarray(v, dtype=float) for k, v in rows.items()}
    columns["t"] = np.asarray(traj.t, dtype=float)
    if lag is not None:
        columns["gust_lag_1"] = np.asarray(lag, dtype=float)[:, 0]
        columns["gust_lag_2"] = np.asarray(lag, dtype=float)[:, 1]

    energy, power = checks._energy_and_power(traj, ac, field)
    t = np.asarray(traj.t)
    work = np.concatenate([[0.0], np.cumsum(np.diff(t) * 0.5 * (power[1:] + power[:-1]))])
    columns["energy"] = np.asarray(energy, dtype=float)
    # |v_ned| = |v_body|: the rotation keeps length, so this is the gate's KE.
    columns["kinetic"] = 0.5 * float(ac.mass) * np.sum(np.asarray(traj.vel_body) ** 2, axis=1)
    columns["potential"] = columns["energy"] - columns["kinetic"]
    columns["work"] = work
    columns["energy_residual"] = (columns["energy"] - columns["energy"][0]) - work

    term_names = {c: [k.split(".", 1)[1] for k in columns
                      if k.startswith(c + ".") and not k.endswith(".strip")]
                  for c in _COEFFICIENTS}
    channels = _channels(term_names, load_model is not None, lag is not None)
    return columns, channels


class Diagnostics(NamedTuple):
    columns: dict
    channels: list
    timing: dict


def write(directory, columns: dict, channels: list, timing: dict | None = None) -> Path:
    """`diagnostics.parquet`, with each column's unit, group and description."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    fields, arrays = [], []
    for ch in channels:
        fields.append(pa.field(ch.name, pa.float64(), metadata={
            "units": ch.unit, "group": ch.group, "description": ch.description}))
        arrays.append(pa.array(columns[ch.name], type=pa.float64()))
    schema = pa.schema(fields, metadata={"timing": json.dumps(timing or {})})
    path = Path(directory) / FILENAME
    pq.write_table(pa.Table.from_arrays(arrays, schema=schema), path, compression="zstd")
    return path


def read(directory) -> Diagnostics | None:
    """The diagnostics of a High run, or None for a Standard one."""
    path = Path(directory) / FILENAME
    if not path.exists():
        return None
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    channels = []
    for f in table.schema:
        m = {k.decode(): v.decode() for k, v in (f.metadata or {}).items()}
        channels.append(Channel(f.name, m.get("group", ""), m.get("units", ""),
                                m.get("description", "")))
    columns = {name: table.column(name).to_numpy() for name in table.column_names}
    raw = (table.schema.metadata or {}).get(b"timing", b"{}")
    return Diagnostics(columns, channels, json.loads(raw.decode()))


def is_high(directory) -> bool:
    return (Path(directory) / FILENAME).exists()


def g_units(value):
    """Specific force is in standard g; kept here so readers find the divisor."""
    return value / G0
