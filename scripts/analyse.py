"""Re-analyse a saved run without re-flying it. Run from the project root:

    .venv/Scripts/python.exe scripts/analyse.py runs/climb.npz
    .venv/Scripts/python.exe scripts/analyse.py runs/*.npz --png out/

This is the point of writing .npz at all: comparing turbulence realisations
means comparing runs against each other, and a run that has to be re-flown to
change a plot is not the same run.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import flightsim  # noqa: F401  -- enables x64 before any array is made
from flightsim import viz
from flightsim.manual import Mode
from flightsim.units import RAD2DEG

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("runs", type=Path, nargs="+", help=".npz files written by fly.py")
parser.add_argument("--png", type=Path, help="save figures here instead of showing them")
args = parser.parse_args()

for path in args.runs:
    traj = viz.load(path)
    d = viz.derived(traj)
    autopilot_fraction = float(np.mean(traj.mode == int(Mode.AUTOPILOT)))
    print(f"{path}")
    print(f"  {traj.t[-1]:8.1f} s, {len(traj.t)} steps, {autopilot_fraction:.0%} autopilot")
    print(f"  altitude  {d.altitude.min():8.0f} to {d.altitude.max():8.0f} m")
    print(f"  TAS       {d.airspeed.min():8.1f} to {d.airspeed.max():8.1f} m/s")
    print(f"  alpha     {d.alpha.min() * RAD2DEG:8.2f} to {d.alpha.max() * RAD2DEG:8.2f} deg")
    print(f"  beta      {d.beta.min() * RAD2DEG:8.2f} to {d.beta.max() * RAD2DEG:8.2f} deg")

    figure = viz.post_flight(traj, title=str(path))
    if args.png is not None:
        args.png.mkdir(parents=True, exist_ok=True)
        out = args.png / f"{path.stem}.png"
        figure.savefig(out, dpi=120)
        plt.close(figure)
        print(f"  wrote {out}")

if args.png is None:
    plt.show()
