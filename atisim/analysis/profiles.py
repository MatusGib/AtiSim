"""Every check as its time series, with its tolerance (plan section 3.3).

A badge says whether a check passed and where it was worst. A profile says how
the checked quantity got there: the energy residual growing through a core, the
angle of attack leaving the linear band and coming back. Each profile is
computed here the way `checks.py` computes the check, from the same trajectory,
so the profile's peak is the check's value.

A profile needs no High fidelity run: every check reads the trajectory only.
A check with no time axis (the field divergence is sampled at positions) gets
a sentence instead of a profile.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from atisim import checks, dynamics
from atisim.aero import air_data
from atisim.aircraft import Aircraft
from atisim.atmosphere import speed_of_sound
from atisim.units import RAD2DEG


class Profile(NamedTuple):
    check: str  # the check's name, as checks.json has it
    t: np.ndarray  # (n,) s
    lines: list  # [(label, (n,) array)]
    unit: str
    limits: list  # [(label, value)] horizontal lines
    log: bool  # plot on a log axis
    note: str


def _air(traj):
    vel_rel = jax.vmap(dynamics.relative_velocity)(
        jnp.asarray(traj.vel_body), jnp.asarray(traj.quat), jnp.asarray(traj.wind_ned))
    return jax.vmap(air_data)(vel_rel)


def _moving_rms(x, width):
    c = np.concatenate([[0.0], np.cumsum(x ** 2)])
    out = np.full(len(x), np.nan)
    out[width - 1:] = np.sqrt((c[width:] - c[:-width]) / width)
    return out


def build(traj, ac: Aircraft, field, check_rows: list[dict]) -> dict:
    """{check name: Profile or a sentence}, for each row of the run's checks."""
    t = np.asarray(traj.t)
    tolerance = {r["name"]: r.get("tolerance") for r in check_rows}
    out: dict = {}
    energy = power = None

    def energy_terms():
        nonlocal energy, power
        if energy is None:
            energy, power = checks._energy_and_power(traj, ac, field)
        work = np.concatenate([[0.0], np.cumsum(np.diff(t) * 0.5 * (power[1:] + power[:-1]))])
        return energy, work

    for row in check_rows:
        name = row["name"]
        tol = tolerance.get(name)
        limit = [("tolerance", float(tol))] if tol is not None else []
        if name == "energy closure":
            e, work = energy_terms()
            residual = np.abs((e - e[0]) - work)
            scale = checks.energy_scale(e)
            out[name] = Profile(name, t, [("|dE - W| / |dE|max", residual / scale)], "-",
                                limit, True, "The gate is the peak of this line.")
        elif name == "energy residual peak":
            e, work = energy_terms()
            per_step = np.abs(np.diff((e - e[0]) - work))
            out[name] = Profile(name, t[1:], [("per-step residual", per_step)], "J",
                                [("run median", float(np.median(per_step)))], True,
                                "A spike names where the closure breaks; on a Rankine "
                                "field it is the core boundary.")
        elif name == "alpha band":
            _, alpha, _ = _air(traj)
            out[name] = Profile(name, t, [("|alpha|, air-relative",
                                           np.abs(np.asarray(alpha)) * RAD2DEG)], "deg",
                                [("linear", checks.ALPHA_LINEAR_DEG),
                                 ("invalid", checks.ALPHA_INVALID_DEG)], False,
                                "The gate reads the window's peak; the band is declared, "
                                "not a stall table.")
        elif name == "recovery band":
            mach_lo, mach_hi = (float(v) for v in ac.valid_mach)
            alt_lo, alt_hi = (float(v) for v in ac.valid_altitude)
            altitude = -np.asarray(traj.pos_ned)[:, 2]
            V, _, _ = _air(traj)
            mach = np.asarray(V) / np.asarray(speed_of_sound(jnp.asarray(altitude)))
            lines = []
            if mach_hi > mach_lo:
                lines.append(("Mach, band widths outside", checks._excursion(mach, mach_lo,
                                                                             mach_hi)))
            if alt_hi > alt_lo:
                lines.append(("altitude, band widths outside",
                              checks._excursion(altitude, alt_lo, alt_hi)))
            out[name] = (Profile(name, t, lines, "band widths", [("inside", 0.0)], False,
                                 "Zero is inside the condition the derivatives were "
                                 "recovered at.")
                         if lines else "This aircraft declares no recovery band.")
        elif name == "recorded wind":
            analytic = np.asarray(jax.jit(jax.vmap(field))(jnp.asarray(traj.pos_ned)))
            recorded = np.asarray(traj.wind_ned)
            shifted = np.abs(recorded[1:] - analytic[:-1]).max(axis=1)
            out[name] = Profile(name, t[1:], [("|recorded - field|, one step back",
                                               shifted)], "m/s", limit, True,
                                "The recorded wind is the previous step's; shifted one "
                                "step it must equal the field.")
        elif name == "trimmed start":
            n_z = checks.load_factor_series(traj, ac)
            out[name] = Profile(name, t, [("n_z", n_z)], "g", [("start", float(n_z[0]))],
                                False, "The check compares the start with the run's own "
                                       "peak excursion.")
        elif name == "rms normal load":
            n_z = checks.load_factor_series(traj, ac)
            dt = float(np.median(np.diff(t))) if t.size > 1 else 0.0
            width = int(round(checks.RMS_NORMAL_LOAD_WINDOW / dt)) if dt > 0 else 0
            out[name] = (Profile(name, t, [(f"{checks.RMS_NORMAL_LOAD_WINDOW:g} s moving "
                                            "RMS of n_z - 1", _moving_rms(n_z - 1.0, width))],
                                 "g", list(checks.RMS_NORMAL_LOAD_BANDS.items()), False,
                                 "A report: how rough the air was, not a defect.")
                         if 2 <= width <= len(n_z) else "The run is shorter than the window.")
        elif name == "quaternion norm":
            drift = np.abs(1.0 - np.linalg.norm(np.asarray(traj.quat), axis=1))
            out[name] = Profile(name, t, [("|1 - |q||", drift)], "-", limit, True,
                                "Renormalised every step: this sits at round-off.")
        elif name == "lateral symmetry":
            v = np.asarray(traj.vel_body)[:, 1]
            omega = np.asarray(traj.omega)
            out[name] = Profile(name, t, [("|v|", np.abs(v)), ("|p|", np.abs(omega[:, 0])),
                                          ("|r|", np.abs(omega[:, 2]))], "SI", limit, True,
                                "Exactly zero on a field with no spanwise structure.")
        else:
            out[name] = ("Sampled at positions along the path, not in time: no time "
                         "profile." if name == "field divergence"
                         else "No time profile for this check.")
    return out
