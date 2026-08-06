"""Vortex-analysis progress figure.

Flies the 747 through the Parks 1985 vortex array and through the Wingrove &
Bach updraft column, then draws one figure showing where the turbulence work
currently stands against the papers' own numbers.

Run: .venv/Scripts/python.exe scripts/vortex.py
     .venv/Scripts/python.exe scripts/vortex.py --png runs/vortex.png
"""

import argparse
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt

import flightsim  # noqa: F401  -- enables x64
from flightsim import trim, vortex_viz, wind
from flightsim.aircraft import CRUISE, REGISTRY
from flightsim.units import RAD2DEG
from flightsim.wind import PARKS_CASES as CASES
from flightsim.wind import UPDRAFT_SECONDS, UPDRAFT_W0

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--case", default="hannibal", choices=sorted(CASES))
parser.add_argument("--aircraft", default="boeing747", choices=sorted(REGISTRY))
parser.add_argument("--dt", type=float, default=0.01, help="physics step, s")
parser.add_argument(
    "--lead-in", type=float, default=40.0,
    help="start distance upstream, in core radii. Below ~12 the 1/r far field "
         "launches the aircraft out of equilibrium and contaminates the first core.",
)
parser.add_argument(
    "--sharpness", type=float, default=6.0,
    help="updraft edge sharpness. DECLARED MODELLING PARAMETER, not source data: "
         "the paper fixes the magnitude and duration and says nothing about the edge.",
)
parser.add_argument("--png", type=Path, help="save here instead of showing")
args = parser.parse_args()

case = CASES[args.case]
ac = REGISTRY[args.aircraft]
V = CRUISE[args.aircraft]["airspeed"]
H = CRUISE[args.aircraft]["altitude"]
r0, v0, spacing = case["r0"], case["v0"], case["spacing"]

x, _ = trim.trim(jnp.array(V), jnp.array(H), ac)
alpha, elevator, throttle = (float(v) for v in x)

# --- the vortex array: two cores on the flightpath, as Parks describes -------
cores = [(0.0, H), (spacing, H)]
array = wind.VortexArray(
    north=jnp.array([c[0] for c in cores]),
    down=jnp.array([-c[1] for c in cores]),
    r0=jnp.array(r0),
    v0=jnp.array(v0),
)
vortex_field = lambda p: wind.vortex_wind(p, array)  # noqa: E731

lead = args.lead_in * r0
vortex = vortex_viz.fly(
    ac, vortex_field, V, H,
    label=f"vortex ({args.case})",
    start_north=-lead,
    seconds=(spacing + lead + 6.0 * r0) / V,
    dt=args.dt,
    window=(-r0, r0),
    window_name="first core",
)

# --- the updraft column: the long-encounter limb of the discriminator -------
radius = 0.5 * UPDRAFT_SECONDS * V
column = wind.UpdraftColumn(
    north=jnp.array(0.0), east=jnp.array(0.0), w0=jnp.array(UPDRAFT_W0),
    radius=jnp.array(radius), sharpness=jnp.array(args.sharpness),
)
updraft_field = lambda p: wind.updraft_wind(p, column)  # noqa: E731
updraft = vortex_viz.fly(
    ac, updraft_field, V, H,
    label="updraft",
    start_north=-2.0 * radius,
    seconds=4.0 * radius / V,
    dt=args.dt,
    window=(-radius, radius),
    window_name="column",
)

provenance = (
    f"{args.aircraft}  CR-2144 FC9  {H:.0f} m  {V:.2f} m/s  "
    f"trim alpha {alpha * RAD2DEG:.3f} deg  elev {elevator * RAD2DEG:.3f} deg  "
    f"thr {throttle:.4f}  dt {args.dt} s  lead-in {args.lead_in:.0f}*r0\n"
    f"vortex: Parks et al. 1985 case '{args.case}', r0 {r0:.1f} m, V0 {v0:.2f} m/s, "
    f"spacing {spacing:.1f} m, 2 cores, window = first core\n"
    f"updraft: Wingrove & Bach 1994 w0 {UPDRAFT_W0:.2f} m/s, {UPDRAFT_SECONDS:.0f} s "
    f"traverse -> radius {radius:.0f} m; edge sharpness {args.sharpness:g} DECLARED, "
    f"not sourced\n"
    "Both papers' records are DC-10 class at 33-39 kft; this is a 747 at 40 kft. "
    "Load comparisons are order-of-magnitude only."
)

for enc in (vortex, updraft):
    dtheta, dn = vortex_viz.fig8_point(enc)
    print(f"{enc.label:22s} window={enc.window_name:12s} "
          f"d(theta)={dtheta:6.3f} deg  d(n)={dn:+.3f} g  "
          f"n_z(0)={enc.n_z[0]:.4f}")

figure = vortex_viz.figure(
    [vortex, updraft],
    field=vortex_field,
    array_cores=cores,
    core_radius=r0,
    peak_tangential=v0,
    provenance=provenance,
    title=f"vortex analysis progress -- {args.aircraft}, Parks case '{args.case}'",
)
if args.png:
    args.png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.png, dpi=120, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {args.png}")
else:
    plt.show()
