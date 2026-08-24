"""Fly interactively. Run from the project root:

    .venv/Scripts/python.exe scripts/fly.py
    .venv/Scripts/python.exe scripts/fly.py --autopilot --save runs/climb.npz
    .venv/Scripts/python.exe scripts/fly.py --wind hannibal

Every run starts trimmed at the aircraft's cruise condition, and the autopilot
targets are that same condition, so engaging the autopilot at t = 0 holds what
you already have.

    arrows   centre stick: up is stick forward, so up pitches the nose DOWN
    , .      rudder left/right
    -  =     throttle down/up
    [  ]     pitch trim, nose down/up
    t        trim here: hold the deflections the stick is holding now
    a        toggle manual/autopilot

The stick ramps rather than snapping to full travel, so a tap is a small input.

With `--wind`, the field is placed AHEAD of the aircraft and the panel counts
the distance down. The two named cases are the vortex arrays Parks et al. 1985
identified; `updraft` is the Wingrove & Bach 1994 column. Both are the 747's
turbulence cases -- flying a light aircraft into them is not a sourced result.

Close the window to end the flight. The post-flight plots open afterwards.
"""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import atisim  # noqa: F401  -- enables x64 before any array is made
from atisim import autopilot as ap_mod
from atisim import integrate, manual as man, panel as panel_mod, trim, viz, wind
from atisim.aircraft import CRUISE, REGISTRY
from atisim.sensors import sense
from atisim.units import RAD2DEG

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--aircraft", default="boeing747", choices=sorted(REGISTRY))
parser.add_argument("--autopilot", action="store_true", help="start with it engaged")
parser.add_argument("--save", type=Path, help="write the trajectory to this .npz")
parser.add_argument("--dt", type=float, default=0.02, help="physics step, s")
parser.add_argument("--fps", type=float, default=20.0, help="render rate")
parser.add_argument("--window", type=float, default=60.0, help="strip window, s")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--wind",
    default="none",
    choices=["none", *sorted(wind.PARKS_CASES), "updraft"],
    help="wind field to fly through. The named cases are Parks et al. 1985.",
)
parser.add_argument(
    "--lead-in",
    type=float,
    default=40.0,
    help="distance to the first core, in core radii. Below ~12 the 1/r far "
    "field launches the aircraft out of equilibrium and contaminates it.",
)
parser.add_argument(
    "--sharpness",
    type=float,
    default=6.0,
    help="updraft edge sharpness. DECLARED MODELLING PARAMETER, not source data.",
)
args = parser.parse_args()

ac = REGISTRY[args.aircraft]
gains = ap_mod.GAINS[args.aircraft]
mgains = man.MANUAL_GAINS[args.aircraft]
V = CRUISE[args.aircraft]["airspeed"]
H = CRUISE[args.aircraft]["altitude"]

x, residual = trim.trim(jnp.array(V), jnp.array(H), ac)
alpha, elevator, throttle = (float(v) for v in x)
print(f"{args.aircraft}: trimmed at {V:.1f} m/s, {H:.0f} m")
print(
    f"  alpha {alpha * RAD2DEG:+.3f} deg   elevator {elevator * RAD2DEG:+.3f} deg"
    f"   throttle {throttle:.4f}   residual {float(jnp.linalg.norm(residual)):.2e}"
)

state = trim.trimmed_state(jnp.array(alpha), jnp.array(V), jnp.array(H))
controls = trim.trimmed_controls(jnp.array(elevator), jnp.array(throttle))
targets = ap_mod.Targets(
    altitude=jnp.array(H), heading=jnp.array(0.0), airspeed=jnp.array(V)
)
mode = man.Mode.AUTOPILOT if args.autopilot else man.Mode.MANUAL
ctl = man.start(sense(state), controls, targets, gains, ac, mode=mode)
sim = integrate.init_sim(state, jax.random.PRNGKey(args.seed))

wind_model, field_range, note = panel_mod.field_ahead(
    args.wind,
    airspeed=V,
    altitude=H,
    lead_in=args.lead_in,
    sharpness=args.sharpness,
)
print(f"  {note}")

print(f"  starting in {mode.name}. Close the window to finish.")
traj = panel_mod.run_live(
    sim,
    ctl,
    targets,
    gains,
    mgains,
    ac,
    dt=args.dt,
    fps=args.fps,
    window=args.window,
    wind_model=wind_model,
    field_range=field_range,
    aircraft_name=args.aircraft,
)

print(f"\nflew {traj.t[-1]:.1f} s, {len(traj.t)} steps")
if args.save is not None:
    args.save.parent.mkdir(parents=True, exist_ok=True)
    viz.save(traj, args.save)
    print(f"saved {args.save}")

viz.post_flight(traj, title=f"{args.aircraft}  --  {traj.t[-1]:.0f} s")
plt.show()
