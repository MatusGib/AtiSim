"""Mountain lee wave against the 747's thrust authority.

PROJECT.md section 7 step 8. Flies the 747 at its CR-2144 cruise condition
through a Doyle et al. 2011 lee wave and asks the one question the F-factor was
invented to answer: can the engines counter it?

The answer is not the same for both of that paper's flight legs, which is the
result. Run:

    .venv/Scripts/python.exe scripts/leewave.py
    .venv/Scripts/python.exe scripts/leewave.py --png runs/leewave.png
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
from flightsim.units import RAD2DEG

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--aircraft", default="boeing747", choices=sorted(REGISTRY))
parser.add_argument("--dt", type=float, default=0.01, help="physics step, s")
parser.add_argument(
    "--wavelength", type=float, default=wind.LEE_WAVE_WAVELENGTH / 1000.0,
    help="lee-wave wavelength, km. DECLARED MODELLING PARAMETER, not source "
         "data: Doyle et al.'s 20-35 km band is TROPOSPHERIC and the same "
         "paragraph says stratospheric wavelengths are shorter without giving a "
         "number. It does not move the F-factor peak at all -- with no "
         "horizontal perturbation the index is -w/V, independent of wavelength "
         "-- but it sets the encounter duration and the pitching gust rate.",
)
parser.add_argument("--waves", type=float, default=3.0, help="wavelengths flown")
parser.add_argument("--png", type=Path, help="save here instead of showing")
args = parser.parse_args()

ac = REGISTRY[args.aircraft]
V = CRUISE[args.aircraft]["airspeed"]
H = CRUISE[args.aircraft]["altitude"]
wavelength = args.wavelength * 1000.0

x, _ = trim.trim(jnp.array(V), jnp.array(H), ac)
alpha, elevator, throttle = (float(v) for v in x)
full, idle = (float(v) for v in dynamics.thrust_authority(ac, x[2], jnp.array(H)))
controls = trim.trimmed_controls(x[1], x[2])


def fly(w0):
    """Fly one wave train with fixed controls, returning the flown F-factor."""
    wave = wind.LeeWave(
        w0=jnp.array(w0), wavelength=jnp.array(wavelength), north=jnp.array(0.0)
    )
    field = lambda p: wind.lee_wave_wind(p, wave)  # noqa: E731
    state = trim.trimmed_state(jnp.array(alpha), jnp.array(V), jnp.array(H))
    # Start half a wavelength upstream of the first trough, so the run opens in
    # undisturbed-mean air rather than already sinking.
    state = state._replace(
        pos_ned=jnp.array([-0.5 * args.waves * wavelength, 0.0, -H])
    )
    n = int(round((args.waves * wavelength / V) / args.dt))
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
            shear / dynamics.G0, -wind_ned[2], air.airspeed,
        ])

    rows = np.asarray(
        jax.vmap(analyse)(hist.pos_ned, hist.vel_body, hist.quat, hist.omega)
    )
    return dict(
        w0=w0,
        t=np.arange(1, n + 1) * args.dt,
        north=np.asarray(hist.pos_ned)[:, 0],
        altitude=-np.asarray(hist.pos_ned)[:, 2],
        f=rows[:, 0], shear_term=rows[:, 1], w_up=rows[:, 2], airspeed=rows[:, 3],
    )


legs = {name: fly(w0) for name, w0 in sorted(wind.LEE_WAVE_AMPLITUDE.items())}

print(f"{args.aircraft}  CR-2144 FC9  {H:.0f} m  {V:.2f} m/s")
print(f"thrust authority (T-D)/W: full throttle {full:+.4f}   idle {idle:+.4f}\n")
print(f"{'leg':>7} {'w0':>6} {'peak F':>9} {'shear':>9} {'sink':>8}  verdict")
for name, r in legs.items():
    sink = r["altitude"][-1] - r["altitude"][0]
    verdict = (
        "EXCEEDS full thrust -- unrecoverable"
        if r["f"].max() > full else "within thrust authority"
    )
    print(f"{name:>7} {r['w0']:5.1f}  {r['f'].max():+9.5f} "
          f"{np.abs(r['shear_term']).max():9.2e} {sink:7.0f} m  {verdict}")

critical = full * V
print(f"\ncritical amplitude, F = {full:+.4f}: w0 = {critical:.2f} m/s")
print(f"Doyle et al. 2011 IOP 4 primary wave: 6 m/s crest-to-trough north, 12 south")
print("-> the hazard threshold sits INSIDE the observed range, not outside it.")
print("The shear term is zero BY CONSTRUCTION: this field carries no horizontal")
print("perturbation (PROJECT.md section 5). F here is the vertical term alone.")

# --- the figure --------------------------------------------------------------
figure, axes = plt.subplots(2, 1, figsize=(11.0, 7.6), sharex=True)
figure.subplots_adjust(bottom=0.20, top=0.90, hspace=0.22)
figure.suptitle(
    f"mountain lee wave vs thrust authority -- {args.aircraft} at "
    f"{H:.0f} m, {wavelength/1000:.0f} km wavelength"
)

for index, (name, r) in enumerate(legs.items()):
    colour = f"C{index}"
    axes[0].plot(r["north"] / 1000.0, r["w_up"], color=colour,
                 label=f"{name} leg, w0 = {r['w0']:.1f} m/s")
    axes[1].plot(r["north"] / 1000.0, r["f"], color=colour,
                 label=f"{name} leg, peak F = {r['f'].max():+.4f}")

axes[0].axhline(0.0, color="0.6", lw=0.8)
axes[0].set_ylabel("vertical gust, up positive  m/s")
axes[0].set_title("the wave (Doyle et al. 2011, T-REX, G-V at 11.3 and 13.1 km)")
axes[0].grid(alpha=0.3)
axes[0].legend(fontsize=8, loc="upper right")

# The band the engines can cover. Above it the wind drains energy faster than
# full thrust can restore it -- Proctor et al.'s own criterion, not a threshold
# invented here.
axes[1].axhspan(idle, full, color="0.88", zorder=0)
axes[1].axhline(full, color="C3", lw=1.0, ls="--")
axes[1].annotate(
    f"full thrust  {full:+.4f}", (0.01, full), xycoords=("axes fraction", "data"),
    textcoords="offset points", xytext=(0, 4), fontsize=8, color="C3",
)
axes[1].axhline(idle, color="0.5", lw=1.0, ls="--")
axes[1].annotate(
    f"idle  {idle:+.4f}", (0.01, idle), xycoords=("axes fraction", "data"),
    textcoords="offset points", xytext=(0, 4), fontsize=8, color="0.4",
)
axes[1].set_ylabel("F-factor,  positive = losing energy")
axes[1].set_xlabel("along track  km")
axes[1].set_title(
    "Bowles F-factor (Proctor et al. 2000 Eq. 3) against the thrust the aircraft has"
)
axes[1].grid(alpha=0.3)
axes[1].legend(fontsize=8, loc="upper right")

figure.text(
    0.01, 0.012,
    f"{args.aircraft}  CR-2144 FC9  {H:.0f} m  {V:.2f} m/s  trim alpha "
    f"{alpha * RAD2DEG:.3f} deg  elev {elevator * RAD2DEG:.3f} deg  "
    f"thr {throttle:.4f}  dt {args.dt} s  fixed controls\n"
    f"field: Doyle et al. 2011 Mon. Wea. Rev. 139 3-23, IOP 4 primary wave, "
    f"amplitudes {wind.LEE_WAVE_AMPLITUDE['north']:.1f}/"
    f"{wind.LEE_WAVE_AMPLITUDE['south']:.1f} m/s zero-to-peak; wavelength "
    f"{wavelength/1000:.0f} km DECLARED, not sourced\n"
    f"index: Proctor, Hinton & Bowles 2000 paper 7.7 Eq. (3); hazard criterion "
    f"F > (T_r-D)/W is that paper's own. The FAA 0.1 threshold is NOT used -- it "
    f"is calibrated for below 500 m.\n"
    "The field carries no horizontal perturbation, so the shear term is zero by "
    "construction and F is the vertical term alone (PROJECT.md section 5).",
    fontsize=6.5, family="monospace", color="0.35",
)

if args.png:
    args.png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.png, dpi=120)
    plt.close(figure)
    print(f"\nwrote {args.png}")
else:
    plt.show()
