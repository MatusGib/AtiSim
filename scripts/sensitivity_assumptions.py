"""Phase S4: what each MODELLING CHOICE costs the headline load, on one axis.

WHY THIS IS THE PHASE THAT MATTERS. `docs/ASSUMPTIONS.md` has seventeen rows and
each carries a cost measured in its own units -- a gust error in m/s, a per-cent
of density, a per-cent of an in-core pitch excursion, a fraction of a quadrature
calibration. None of them is comparable with any other, and none is comparable
with a coefficient. This screen puts every one of them on the SAME axis as
`scripts/sensitivity_load.py`'s tier A: per cent of the headline peak-to-peak
load.

C3 IS THE POINT OF IT. The register calls C3 -- derivatives frozen across the
envelope -- **UNBOUNDED**, and has since session 12. Session 27 gave its Mach
axis one point, at M 0.406 on the LES field, 13x further from the tabulation
condition than anything else this project flies. This measures it AT CRUISE, on
the headline run, where the excursion is the aircraft's own speed change through
the encounter rather than a different flight condition.

WHAT IS CITED RATHER THAN RE-MEASURED, per the study's own rule. A2 (constant g)
was measured in sessions 12 and 23 at +0.383% of g at cruise and is modelled now;
the wind and scenario inputs are priced by `cat_bounds.py` and
`cat_uncertainty.py`. Re-running them here would duplicate work the record
already holds.

Run: PYTHONPATH=<abs worktree root> python scripts/sensitivity_assumptions.py
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import atisim  # noqa: F401  -- enables x64
from atisim import airframe, loads, sensitivity, wind
from atisim.atmosphere import speed_of_sound
from scripts.sensitivity_load import RECORDED_PEAK_TO_PEAK, mehta_setup



def mach_history(cfg):
    """The AIR-RELATIVE Mach at every sample of the base run.

    Not |vel_body|/a: `aero.py` builds Mach from `vel_rel`, which is the body
    velocity minus the wind, so the gust moves it. And `a_sound` is evaluated at
    the aircraft's own altitude, which drifts over 47 s of fixed-control flight.
    """
    from atisim import dynamics, integrate, trim
    from atisim.state import State, quat_to_dcm

    x = sensitivity.solved_trim(jnp.array(cfg["V"]), jnp.array(cfg["H"]), cfg["ac"])
    controls = trim.trimmed_controls(x[1], x[2])
    state = trim.trimmed_state(x[0], jnp.array(cfg["V"]), jnp.array(cfg["H"]))
    state = state._replace(
        pos_ned=jnp.array([cfg["start_north"], 0.0, -cfg["H"]]))
    w0, *_ = cfg["model"](wind.zero_wind_state(), state,
                          jax.random.PRNGKey(0), jnp.array(cfg["dt"]))
    state = state._replace(vel_body=state.vel_body + quat_to_dcm(state.quat).T @ w0)
    _, hist = integrate.rollout(
        integrate.init_sim(state, jax.random.PRNGKey(0)),
        controls, jnp.array(cfg["dt"]), cfg["ac"], cfg["n_steps"],
        wind_model=cfg["model"])

    def mach_of(pos_ned, vel_body, quat, omega):
        s = State(pos_ned=pos_ned, vel_body=vel_body, quat=quat, omega=omega)
        w, _, _, _, _ = cfg["model"](wind.zero_wind_state(), s,
                                     jax.random.PRNGKey(0),
                                     jnp.array(cfg["dt"]))
        vel_rel = dynamics.relative_velocity(vel_body, quat, w)
        return jnp.linalg.norm(vel_rel) / speed_of_sound(-pos_ned[2])

    return np.asarray(jax.vmap(mach_of)(hist.pos_ned, hist.vel_body,
                                        hist.quat, hist.omega))


def strip_path_is_live(cfg):
    """Prove the strip path RAN before reporting that it moved nothing.

    A 0.000% from "roll only, and n_z cannot see a rolling moment" looks exactly
    like a 0.000% from "the load model was never called". `loads.strip_increment`
    returns a rolling moment and nothing else, so the discriminator is the ROLL
    RATE: if the strip run's peak |p| differs from the point run's, the path is
    live and its silence on n_z is physics rather than plumbing.
    """
    from atisim import integrate, trim

    def peak_roll(load_model):
        x = sensitivity.solved_trim(jnp.array(cfg["V"]), jnp.array(cfg["H"]),
                                    cfg["ac"])
        controls = trim.trimmed_controls(x[1], x[2])
        state = trim.trimmed_state(x[0], jnp.array(cfg["V"]), jnp.array(cfg["H"]))
        state = state._replace(
            pos_ned=jnp.array([cfg["start_north"], 0.0, -cfg["H"]]))
        _, hist = integrate.rollout(
            integrate.init_sim(state, jax.random.PRNGKey(0)), controls,
            jnp.array(cfg["dt"]), cfg["ac"], cfg["n_steps"],
            wind_model=cfg["model"], load_model=load_model)
        return float(np.abs(np.asarray(hist.omega)[:, 0]).max())

    point = peak_roll(None)
    strip = peak_roll(loads.strip_model(cfg["field"], cfg["ac"]))
    return point, strip


def run(cfg, *, ac=None, dt=None, load_model=None, stage_sampled=False):
    """One flight of the headline run, returning its peak-to-peak over the window.

    The window is re-derived per run rather than reused, because a change of dt
    changes the sample count and a fixed index mask would then mean a different
    stretch of sky. The window is defined in METRES of north position, so
    re-deriving it keeps the measured stretch identical.
    """
    ac = cfg["ac"] if ac is None else ac
    dt = cfg["dt"] if dt is None else dt
    n_steps = int(round(cfg["seconds"] / dt))
    n_z, north = sensitivity.load_history(
        ac, cfg["model"], cfg["V"], cfg["H"], start_north=cfg["start_north"],
        n_steps=n_steps, dt=dt, load_model=load_model, stage_sampled=stage_sampled)
    n_z, north = np.asarray(n_z), np.asarray(north)
    mask = (north >= cfg["window"][0]) & (north <= cfg["window"][1])
    return float(n_z[mask].max() - n_z[mask].min()), n_z, north, mask


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    print(f"atisim imported from: {atisim.__file__}")
    cfg = mehta_setup()
    base, n_z, north, mask = run(cfg)
    out = {"tree": atisim.__file__, "base": base, "rows": []}

    def report(register, what, value, note=""):
        d = 100.0 * (value - base) / base
        out["rows"].append(dict(register=register, what=what, load=value,
                                pct=d, note=note))
        print(f"  {register:10s} {what:38s} {value:9.6f} {d:+8.3f}%  {note}")

    print(f"\nbase: {base:.6f} g peak-to-peak, {100*base/RECORDED_PEAK_TO_PEAK:.2f}% "
          f"of TM-102186's {RECORDED_PEAK_TO_PEAK} g")
    print(f"      dt {cfg['dt']}, {cfg['n_steps']} steps, point-gust load path, "
          f"elliptic loading\n")
    print(f"  {'register':10s} {'variant':38s} {'load, g':>9s} {'vs base':>8s}")

    # ---------------- C3: the Mach axis, at cruise ----------------
    a_sound = float(speed_of_sound(jnp.array(cfg["H"])))
    mach_trim = cfg["V"] / a_sound
    # THE SIZE OF C3'S EXCURSION ON THIS RUN, measured rather than assumed. The
    # Mach `aero.py` sees is |vel_rel|/a_sound -- AIR-RELATIVE -- so it moves
    # with the gust as well as with the aircraft's own speed, and the vortex
    # carries 26.5 m/s of vertical wind. Quoted with the result, per section 1's
    # standing rule that an excursion accompanies every frozen-derivative claim.
    mach_hist = mach_history(cfg)
    print(f"  C3's excursion on this run: Mach {mach_hist.min():.4f} to "
          f"{mach_hist.max():.4f}, trim {mach_trim:.4f}, span "
          f"{mach_hist.max()-mach_hist.min():.4f}")
    print(f"  (session 27's LES point was a Delta M of -0.393; this is "
          f"{(mach_hist.max()-mach_hist.min())/0.393:.3f} of that)\n")
    out["mach"] = dict(trim=mach_trim, lo=float(mach_hist.min()),
                       hi=float(mach_hist.max()))
    declared = cfg["ac"]._replace(pg_mach_ref=jnp.array(mach_trim))
    val, *_ = run(cfg, ac=declared)
    report("C3", f"Prandtl-Glauert on, ref M = {mach_trim:.4f}", val,
           "the derivatives allowed to vary with Mach")

    for ref in (0.75, 0.85):
        val, *_ = run(cfg, ac=cfg["ac"]._replace(pg_mach_ref=jnp.array(ref)))
        report("C3", f"Prandtl-Glauert on, ref M = {ref:.2f}", val,
               "if the tabulation Mach were this instead")

    # ---------------- E4: the wind held across the RK4 stages ----------------
    val, *_ = run(cfg, stage_sampled=True)
    report("E4", "wind sampled PER STAGE, not held", val,
           "the hold is the shipped scheme")

    # ---------------- F1: the step size ----------------
    for dt in (0.02, 0.005, 0.0025):
        val, *_ = run(cfg, dt=dt)
        report("F1", f"dt = {dt} (shipped {cfg['dt']})", val, "")

    # ---------------- E2 / E10: the strip load path ----------------
    strip = loads.strip_model(cfg["field"], cfg["ac"])
    val, *_ = run(cfg, load_model=strip)
    report("E2/E10", "strip-integrated loads, 9 stations", val,
           "roll only -- see the note below")

    # ---------------- F5: the station count ----------------
    for n_span in (17, 57):
        st = airframe.stations(cfg["ac"], n_span=n_span)
        val, *_ = run(cfg, load_model=loads.strip_model(cfg["field"], cfg["ac"], st))
        report("F5", f"strip loads, {n_span} stations", val, "")

    # ---------------- the declared loading shape ----------------
    for shape in ("uniform", "tapered"):
        with airframe.loading_shape(shape):
            val, *_ = run(cfg,
                          load_model=loads.strip_model(cfg["field"], cfg["ac"]))
        report("E2", f"strip loads, {shape} loading shape", val,
               "declared, not sourced")

    point_p, strip_p = strip_path_is_live(cfg)
    print(f"\n  IS THE STRIP PATH LIVE? peak |p| point {point_p:.6e} rad/s,"
          f" strip {strip_p:.6e} rad/s")
    verdict = ("DIFFERENT -- the path ran, and n_z genuinely cannot see a "
               "rolling moment" if point_p != strip_p else
               "IDENTICAL -- the path did NOT run, so the 0.000% rows above "
               "are plumbing and not physics")
    print(f"  {verdict}")

    print("\n  CITED RATHER THAN RE-MEASURED, per the study's own rule:")
    print("    A2   constant g vs g(z)      +0.383% of g at cruise (sessions 12, 23)")
    print("    E12  Hannibal core radius    -4.26% of the headline load (session 26)")
    print("    V0, r0, spacing, sigma_w     scripts/cat_bounds.py, cat_uncertainty.py")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=1, default=float))
        print(f"\nwritten: {args.json}")


if __name__ == "__main__":
    main()
