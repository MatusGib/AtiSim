"""Vortex-analysis progress figure.

Flies the 747 through the Parks 1985 vortex array, through the Wingrove & Bach
updraft column, and through an elevator pushdown, then draws one figure showing
where the turbulence work currently stands against the papers' own numbers.
Those are Wingrove & Bach Fig. 8's three categories, and the point of having all
three is that the discriminator is a claim about ORDERING, which two clusters
cannot test.

Run: .venv/Scripts/python.exe scripts/vortex.py
     .venv/Scripts/python.exe scripts/vortex.py --png runs/vortex.png
"""

import argparse
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import flightsim  # noqa: F401  -- enables x64
from flightsim import trim, vortex_viz, wind
from flightsim.aircraft import CRUISE, REGISTRY
from flightsim.panel import ALPHA_INVALID_DEG, ALPHA_LINEAR_DEG
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
parser.add_argument(
    "--pushdown-seconds", type=float, default=6.609,
    help="manoeuvre pulse length, s. DECLARED MODELLING PARAMETER, not source data: "
         "the paper fixes the load the pilot reached, not how long they held it. "
         "The default is the 747's short period (PROJECT.md section 4), which puts "
         "the manoeuvre between the vortex's 0.235 and the updraft's 3.026 short "
         "periods -- so the third cluster's separation is not a duration effect.",
)
parser.add_argument(
    "--strip", action="store_true",
    help="fly the vortex with strip-integrated loads instead of a point sample "
         "plus gradient. Changes the answer; that change is the result and must "
         "be reported with the loading-shape sensitivity beside it.",
)
parser.add_argument("--png", type=Path, help="save here instead of showing")
parser.add_argument(
    "--artifacts", type=Path, metavar="DIR",
    help="also write a run artifact per encounter, for the analysis UI. Needs "
         "the `ui` extra (pyarrow). Every number this script prints came from a "
         "run that did not survive it until this flag existed.",
)
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
    strip=args.strip,
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
    strip=args.strip,
)

# --- the manoeuvre: the third category, and the only one at ZERO wind -------
#
# The elevator angle is DERIVED, not chosen: the sourced quantity is Fig. 8's
# load band, so the deflection that reaches it is an output. Read as an
# INCREMENT (PROJECT.md section 8) -- the absolute reading needs |alpha| ~18.5
# deg, well past where this model's linear aero means anything (section 5).
hold = args.pushdown_seconds
pushdown_lead = 2.0
pushdown_seconds = pushdown_lead + 3.0 * hold
elevator_step = vortex_viz.elevator_for_load(
    ac, V, H, target=vortex_viz.FIG8_LOAD_INCREMENT,
    hold=hold, seconds=pushdown_seconds, lead_in=pushdown_lead, dt=args.dt,
)
pushdown = vortex_viz.manoeuvre(
    ac, V, H, label="manoeuvre", elevator_step=elevator_step,
    hold=hold, seconds=pushdown_seconds, lead_in=pushdown_lead, dt=args.dt,
)

load_path = (
    "STRIP-INTEGRATED (roll only; pitch and yaw still point-plus-gradient)"
    if args.strip else "point sample plus analytic gradient"
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
    f"manoeuvre: zero wind, elevator pulse {hold:g} s DECLARED (not sourced), "
    f"{elevator_step * RAD2DEG:.3f} deg from trim BISECTED to reach "
    f"d(n) = {vortex_viz.FIG8_LOAD_INCREMENT:+.1f} g, the Fig. 8 band read as an "
    f"INCREMENT; the absolute reading is out of the linear range (PROJECT.md 5, 8)\n"
    f"loads: {load_path}. "
    "Strip results must be quoted with the loading-shape sensitivity beside them: "
    "2.6% across defensible shapes, 49.7% including the uniform bracket.\n"
    "Window rule: the disturbance's own extent -- first core, column, elevator pulse.\n"
    "Both papers' records are DC-10 class at 33-39 kft; this is a 747 at 40 kft. "
    "Load comparisons are order-of-magnitude and ordering only, never values."
)

for enc in (vortex, updraft, pushdown):
    dtheta, dn = vortex_viz.fig8_point(enc)
    peak_alpha = float(np.abs(enc.alpha_air[enc.window]).max()) * RAD2DEG
    band = (
        "linear" if peak_alpha < ALPHA_LINEAR_DEG
        else "MARGINAL" if peak_alpha < ALPHA_INVALID_DEG
        else "INVALID -- this run proves nothing"
    )
    print(f"{enc.label:22s} window={enc.window_name:22s} "
          f"d(theta)={dtheta:6.3f} deg  d(n)={dn:+.3f} g  "
          f"n_z(0)={enc.n_z[0]:.4f}  |alpha|max={peak_alpha:5.2f} deg {band}")

paper = [FIG8 for FIG8 in vortex_viz.FIG8_REFERENCE.values()]
model = [vortex_viz.fig8_point(e)[0] for e in (vortex, updraft, pushdown)]
ordered = all(a < b for a, b in zip(model, model[1:]))
print(f"\nFig. 8 pitch excursion, deg:  paper {paper}  model "
      f"{[round(m, 2) for m in model]}")
print(f"ordering vortex < updraft < manoeuvre: {'HOLDS' if ordered else 'FAILS'}"
      "   (ordering is the claim; absolute agreement is forbidden by section 5)")

if args.artifacts:
    # The artifact carries what the PNG footer carries, as data rather than as
    # text. Everything below is already assembled above for `provenance`; this is
    # a restructuring, not new information.
    from flightsim import checks
    from flightsim.analysis import artifact

    common = dict(
        aircraft_key=args.aircraft,
        aircraft=ac,
        flight_condition={"airspeed_mps": float(V), "altitude_m": float(H),
                          "source": "NASA CR-2144 flight condition 9"},
        trim_solution={"alpha_rad": alpha, "elevator_rad": elevator,
                       "throttle": throttle, "residual_norm": None,
                       "is_physical": True},
        load_model=("loads.strip_model" if args.strip else None),
        loading_shape=("elliptic" if args.strip else None),
        caveats=[
            "Load comparisons are ORDERING ONLY, never values (PROJECT.md 5): "
            "both papers' records are DC-10 class and neither identifies an "
            "aircraft type.",
            "The Parks core is 3.07 spans, so the point-gust assumption E2 is "
            "marginal for the vortex case specifically.",
        ] + ([
            "STRIP loads: quote the loading-shape sensitivity beside any result "
            "-- 2.6% across defensible shapes, 49.7% including the uniform bracket."
        ] if args.strip else []),
    )

    encounters = [
        (vortex, f"vortex-{args.case}",
         {"kind": "VortexArray", "case": args.case,
          "source": "Parks, Wingrove, Bach & Mehta 1985, J. Aircraft 22(2) 124-129",
          "params": {"north": [c[0] for c in cores],
                     "down": [-c[1] for c in cores],
                     "r0": r0, "v0": v0, "spacing": spacing},
          "model": "wind.field_model",
          "omega_gust_estimator": "analytic tangent at CG"},
         {"lead_in_core_radii": args.lead_in,
          "window": {"kind": "first core", "north_m": [-r0, r0]},
          "window_rule": "the disturbance's own extent (PROJECT.md 8)"}),
        (updraft, "updraft",
         {"kind": "UpdraftColumn",
          "source": "Wingrove & Bach 1994, J. Aircraft 31(4) 753-760",
          "params": {"north": 0.0, "east": 0.0, "w0": float(UPDRAFT_W0),
                     "radius": float(radius), "sharpness": args.sharpness},
          "model": "wind.field_model",
          "omega_gust_estimator": "analytic tangent at CG"},
         {"sharpness": args.sharpness,
          "sharpness_provenance": "DECLARED, not sourced -- the paper fixes the "
                                  "magnitude and duration and says nothing about the edge",
          "window": {"kind": "column", "north_m": [-radius, radius]},
          "window_rule": "the disturbance's own extent (PROJECT.md 8)"}),
        (pushdown, "manoeuvre",
         {"kind": "none (zero wind)",
          "source": "the category is DEFINED by the absence of turbulence",
          "params": {}},
         {"pushdown_seconds": hold,
          "pushdown_provenance": "DECLARED, not sourced -- the paper fixes the "
                                 "load the pilot reached, not how long they held it",
          "elevator_deg_from_trim": float(elevator_step * RAD2DEG),
          "elevator_provenance": "BISECTED to reach the Fig. 8 band read as an "
                                 "INCREMENT, so the load is sourced and the angle "
                                 "is an output",
          "window": {"kind": "elevator pulse", "seconds": hold,
                     "starts_at_s": pushdown_lead},
          "window_rule": "the disturbance's own extent (PROJECT.md 8)"}),
    ]

    sha = artifact.git_sha()[:7] or "nogit"
    for enc, name, field_spec, declared in encounters:
        meta = artifact.build_meta(
            integrator={"dt_s": args.dt, "n_steps": len(enc.t)},
            wind_field=field_spec,
            declared_parameters=declared,
            **common,
        )
        enc_field = (
            vortex_field if name.startswith("vortex")
            else updraft_field if name == "updraft"
            else (lambda p: jnp.zeros(3))
        )
        report = checks.run_checks(
            enc.log, ac, trim.trimmed_controls(x[1], x[2]), enc_field, enc.window
        )
        out = artifact.write_run(
            args.artifacts / f"{name}-{args.aircraft}-{sha}", enc.log, meta, report
        )
        failed = [c.name for c in report if c.passed is False]
        print(f"wrote {out}"
              + (f"   CHECKS FAILED: {', '.join(failed)}" if failed else "   checks ok"))

figure = vortex_viz.figure(
    [vortex, updraft, pushdown],
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
