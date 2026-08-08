"""Microburst penetration against the aircraft's thrust authority.

Flies an aircraft on FIXED CONTROLS straight through the centre of an Oseguera &
Bowles microburst and measures the Bowles F-factor against the thrust the
aircraft actually has. Unlike the lee wave, this field has a horizontal outflow,
so both terms of the F-factor are exercised rather than one.

The default aircraft is the CHEROKEE, not the 747, and that is a deliberate
constraint rather than a preference -- see PROJECT.md section 5. The 747's only
derivative set is CR-2144 flight condition 9: Mach 0.8 at 40,000 ft. A microburst
is a sub-500 m phenomenon met at approach speed, and there is no honest way to
fly a cruise-only derivative set there.

Run: .venv/Scripts/python.exe scripts/microburst.py
     .venv/Scripts/python.exe scripts/microburst.py --png runs/microburst.png
"""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import flightsim  # noqa: F401  -- enables x64
from flightsim import dynamics, integrate, sensors, trim, wind
from flightsim.aircraft import CRUISE, REGISTRY
from flightsim.state import State, quat_to_dcm

# Proctor, Hinton & Bowles 2000. The FAA metric is the 1 km AVERAGE F, hazardous
# above 0.1 with a must-alert threshold at 0.13 -- for JET TRANSPORTS. The same
# paper records that neither the averaging scale nor the threshold was ever
# established for piston aircraft, so they are printed for scale and never used
# as a verdict here. The verdict is the aircraft's own (T-D)/W, which is physics
# and needs nobody's certification basis.
FAA_HAZARD = 0.10
FAA_MUST_ALERT = 0.13
AVERAGING_LENGTH = 1000.0

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--aircraft", default="cherokee", choices=sorted(REGISTRY))
parser.add_argument(
    "--altitude", type=float, default=300.0,
    help="penetration altitude AGL, m. Default 300 sits inside the 225-335 m "
         "band NASA's B-737 used for its 1991-92 microburst flight tests.",
)
parser.add_argument("--dt", type=float, default=0.01)
parser.add_argument(
    "--u-max", type=float, default=19.03,
    help="peak outflow, m/s. Default is Oseguera & Bowles' own example, 37 kt.",
)
parser.add_argument(
    "--radius", type=float, default=1000.0,
    help="downdraft shaft radius, m. DECLARED: the paper parameterises by it but "
         "its example value is only in a scanned figure. 1000 m puts peak outflow "
         "at 1.12 km radius, so a 2.2 km outflow diameter -- inside the 1-4 km "
         "band Wilson et al. use to call an outflow a microburst at all.",
)
parser.add_argument(
    "--z-m", type=float, default=150.0,
    help="altitude of peak outflow, m. Midpoint of the paper's stated 100-200 m.",
)
parser.add_argument("--png", type=Path, help="save here instead of showing")
args = parser.parse_args()

ac = REGISTRY[args.aircraft]
V = CRUISE[args.aircraft]["airspeed"]
H = args.altitude

x, _ = trim.trim(jnp.array(V), jnp.array(H), ac)
alpha, elevator, throttle = (float(v) for v in x)
full, idle = (float(v) for v in dynamics.thrust_authority(ac, x[2], jnp.array(H)))
controls = trim.trimmed_controls(x[1], x[2])

burst = wind.microburst(u_max=args.u_max, radius=args.radius, z_m=args.z_m)
field = lambda p: wind.microburst_wind(p, burst)  # noqa: E731
peak_radius = wind.MICROBURST_PEAK_RADIUS_RATIO * args.radius

# Enter well outside the outflow and fly straight at the axis.
start = -3.0 * peak_radius
seconds = (6.0 * peak_radius) / V
n = int(round(seconds / args.dt))
state = trim.trimmed_state(jnp.array(alpha), jnp.array(V), jnp.array(H))
state = state._replace(pos_ned=jnp.array([start, 0.0, -H]))
_, hist = integrate.rollout(
    integrate.init_sim(state, jax.random.PRNGKey(0)),
    controls, jnp.array(args.dt), ac, n, wind_model=wind.field_model(field),
)


def analyse(pos_ned, vel_body, quat, omega):
    s = State(pos_ned=pos_ned, vel_body=vel_body, quat=quat, omega=omega)
    wind_ned = field(pos_ned)
    air = sensors.sense(s, wind_ned)
    shear = wind.along_track_shear(pos_ned, quat_to_dcm(quat) @ vel_body, field)
    return jnp.array([
        dynamics.f_factor(shear, -wind_ned[2], air.airspeed),
        shear / dynamics.G0, -wind_ned[2] / air.airspeed,
        -wind_ned[2], wind_ned[0], air.airspeed,
    ])


rows = np.asarray(jax.vmap(analyse)(hist.pos_ned, hist.vel_body, hist.quat, hist.omega))
north = np.asarray(hist.pos_ned)[:, 0]
altitude = -np.asarray(hist.pos_ned)[:, 2]

# Fixed controls and a microburst mean the ground arrives, and the run is cut
# there rather than averaged over. The cut is at one WINGSPAN, not at zero:
# there is no terrain, no landing gear and no ground effect in this model, so an
# aeroplane whose wings are within their own span of the surface is not flying
# any more and the integration past that point is arithmetic, not physics. It
# would otherwise bounce and climb away, which is exactly the kind of result
# that looks like a survival and is not one.
clearance = float(ac.b)
below = np.flatnonzero(altitude <= clearance)
impact = int(below[0]) if below.size else len(altitude)
t = np.arange(1, n + 1) * args.dt
sl = slice(0, impact)
f, shear_term, vertical_term = rows[sl, 0], rows[sl, 1], -rows[sl, 2]
averaged, valid = (np.asarray(v) for v in dynamics.average_f_factor(
    jnp.asarray(f), jnp.asarray(north[sl]), AVERAGING_LENGTH
))

print(f"{args.aircraft}  {V:.1f} m/s  {H:.0f} m AGL  fixed controls")
print(f"microburst: u_max {args.u_max:.2f} m/s, R {args.radius:.0f} m, "
      f"z_m {args.z_m:.0f} m  ->  peak outflow at r = {peak_radius:.0f} m")
print(f"thrust authority (T-D)/W: full {full:+.4f}   idle {idle:+.4f}\n")

peak_f = float(f.max())
peak_avg = float(averaged[valid].max()) if valid.any() else float("nan")
print(f"peak instantaneous F      {peak_f:+.4f}")
print(f"peak 1 km average F       {peak_avg:+.4f}   <- the metric that counts")
print(f"  its shear term          {float(shear_term.max()):+.4f}   "
      f"(the lee wave's was exactly zero)")
print(f"  its vertical term       {float(vertical_term.max()):+.4f}")
print(f"aircraft thrust authority {full:+.4f}")
print(f"  exceeded by             {peak_avg / full:.1f}x")
print(f"\nFor scale, from the same paper (NOT verdicts for a piston aircraft):")
print(f"  FAA jet-transport hazard {FAA_HAZARD}, must-alert {FAA_MUST_ALERT}"
      f"  -> this is {peak_avg / FAA_HAZARD:.1f}x hazardous")
print(f"  F in real microburst accidents: 0.2 to 0.36  -> this run sits just "
      f"below that band")

if impact < len(altitude):
    print(f"\nGROUND CONTACT at t = {t[impact]:.1f} s, {north[impact]:+.0f} m "
          f"from the axis, having entered {H:.0f} m up and {-start:.0f} m out.")
    print(f"'Contact' is one wingspan ({clearance:.1f} m): no terrain, no gear "
          f"and no ground effect are modelled,\nso flight below that is "
          f"arithmetic rather than physics. It never reaches the far side.")
else:
    print(f"\nNo ground contact; lowest altitude {altitude.min():.0f} m.")

# The strength at which this aircraft's thrust authority is first exceeded. F is
# linear in the field's scaling factor, so it scales directly with u_max.
print(f"\nF scales linearly with the field, so this aircraft's authority is "
      f"first exceeded at\n  u_max = {args.u_max * full / peak_avg:.2f} m/s "
      f"({args.u_max * full / peak_avg * 1.94384:.1f} kt) of peak outflow.")
print("Wilson et al.'s threshold for calling an outflow a microburst at all is "
      "10 m/s\nof divergence over 1-4 km, which this aircraft cannot hold "
      "against either.")

# --- the figure --------------------------------------------------------------
figure, axes = plt.subplots(3, 1, figsize=(11.0, 9.4), sharex=True)
figure.subplots_adjust(bottom=0.17, top=0.92, hspace=0.22)
figure.suptitle(
    f"microburst penetration -- {args.aircraft} at {H:.0f} m AGL, "
    f"u_max {args.u_max:.1f} m/s"
)
km = north[sl] / 1000.0

axes[0].plot(km, rows[sl, 4], color="C0", label="outflow (along track)")
axes[0].plot(km, rows[sl, 3], color="C3", label="vertical gust, up positive")
axes[0].axhline(0.0, color="0.6", lw=0.8)
axes[0].set_ylabel("wind  m/s")
axes[0].set_title("the field: headwind, then downdraft, then tailwind")
axes[0].grid(alpha=0.3)
axes[0].legend(fontsize=8, loc="upper left")

axes[1].axhspan(idle, full, color="0.88", zorder=0)
axes[1].axhline(full, color="C3", lw=1.2, ls="--")
axes[1].annotate(
    f"full thrust {full:+.4f}", (0.01, full), xycoords=("axes fraction", "data"),
    textcoords="offset points", xytext=(0, 4), fontsize=8, color="C3",
)
axes[1].axhline(FAA_HAZARD, color="0.45", lw=1.0, ls=":")
axes[1].annotate(
    "FAA jet-transport hazard 0.10 (not a piston criterion)",
    (0.01, FAA_HAZARD), xycoords=("axes fraction", "data"),
    textcoords="offset points", xytext=(0, 4), fontsize=7.5, color="0.4",
)
axes[1].plot(km, f, color="0.65", lw=0.9, label="instantaneous F")
axes[1].plot(km[valid], averaged[valid], color="C1", lw=2.0,
             label=f"1 km average F, peak {peak_avg:+.4f}")
axes[1].set_ylabel("F-factor")
axes[1].set_title("hazard against the thrust the aircraft has")
axes[1].grid(alpha=0.3)
axes[1].legend(fontsize=8, loc="upper left")

axes[2].plot(km, altitude[sl], color="C0", lw=1.8)
axes[2].axhspan(0.0, clearance, color="#f2d9d6", zorder=0)
axes[2].axhline(0.0, color="C3", lw=1.4)
axes[2].annotate(
    f"ground, and one wingspan ({clearance:.1f} m) above it", (0.01, 0.0),
    xycoords=("axes fraction", "data"),
    textcoords="offset points", xytext=(0, 4), fontsize=8, color="C3",
)
if impact < len(altitude):
    axes[2].plot(north[impact] / 1000.0, 0.0, "x", color="C3", ms=11, mew=2.5)
axes[2].set_ylabel("altitude AGL  m")
axes[2].set_xlabel("along track, 0 = microburst axis  km")
axes[2].set_title("what happens to the aeroplane")
axes[2].grid(alpha=0.3)

figure.text(
    0.01, 0.012,
    f"{args.aircraft}  {V:.2f} m/s  {H:.0f} m AGL  dt {args.dt} s  fixed controls, "
    f"no pilot and no autopilot\n"
    f"field: Oseguera & Bowles, NASA TM-100632 (1988), eqs. 5-6. u_max "
    f"{args.u_max:.2f} m/s from the paper's own 37 kt example; R "
    f"{args.radius:.0f} m and z_m {args.z_m:.0f} m DECLARED within its stated "
    f"bands\n"
    f"index: Proctor, Hinton & Bowles 2000 eq. (3), averaged over "
    f"{AVERAGING_LENGTH:.0f} m by its eq. (7). Verdict is F > (T-D)/W, that "
    f"paper's own criterion.\n"
    "The FAA 0.1/0.13 thresholds are drawn for scale only: the same paper states "
    "they were never established for piston aircraft.",
    fontsize=6.5, family="monospace", color="0.35",
)

if args.png:
    args.png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.png, dpi=120)
    plt.close(figure)
    print(f"\nwrote {args.png}")
else:
    plt.show()
