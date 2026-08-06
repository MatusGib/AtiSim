"""Fly interactively. Run from the project root:

    .venv/Scripts/python.exe scripts/fly.py
    .venv/Scripts/python.exe scripts/fly.py --autopilot --save runs/climb.npz

Every run starts trimmed at the aircraft's cruise condition, and the autopilot
targets are that same condition, so engaging the autopilot at t = 0 holds what
you already have.

    arrows   centre stick: up is stick forward, so up pitches the nose DOWN
    , .      rudder left/right
    -  =     throttle down/up
    a        toggle manual/autopilot

Close the window to end the flight. The post-flight plots open afterwards.
"""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import flightsim  # noqa: F401  -- enables x64 before any array is made
from flightsim import autopilot as ap_mod
from flightsim import integrate, manual as man, panel as panel_mod, trim, viz
from flightsim.aircraft import CRUISE, REGISTRY
from flightsim.sensors import sense
from flightsim.units import RAD2DEG

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--aircraft", default="boeing747", choices=sorted(REGISTRY))
parser.add_argument("--autopilot", action="store_true", help="start with it engaged")
parser.add_argument("--save", type=Path, help="write the trajectory to this .npz")
parser.add_argument("--dt", type=float, default=0.02, help="physics step, s")
parser.add_argument("--fps", type=float, default=20.0, help="render rate")
parser.add_argument("--window", type=float, default=60.0, help="strip/trace window, s")
parser.add_argument("--seed", type=int, default=0)
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

print(f"  starting in {mode.name}. Close the window to finish.")
traj = panel_mod.run_live(
    sim, ctl, targets, gains, mgains, ac, dt=args.dt, fps=args.fps, window=args.window
)

print(f"\nflew {traj.t[-1]:.1f} s, {len(traj.t)} steps")
if args.save is not None:
    args.save.parent.mkdir(parents=True, exist_ok=True)
    viz.save(traj, args.save)
    print(f"saved {args.save}")

viz.post_flight(traj, title=f"{args.aircraft}  --  {traj.t[-1]:.0f} s")
plt.show()
