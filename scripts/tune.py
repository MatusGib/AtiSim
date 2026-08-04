"""Step responses for autopilot gain tuning. Run from the project root."""

import sys

import jax
import jax.numpy as jnp
import numpy as np

import flightsim  # noqa: F401
from flightsim import autopilot as ap_mod
from flightsim import integrate, trim
from flightsim.aero import air_data
from flightsim.aircraft import CRUISE, REGISTRY
from flightsim.state import quat_to_euler
from flightsim.units import RAD2DEG

NAME = "boeing747"
ac = REGISTRY[NAME]
GAINS = ap_mod.GAINS[NAME]
V, H = CRUISE[NAME]["airspeed"], CRUISE[NAME]["altitude"]
DT = 0.02


def fly(targets, seconds, gains=GAINS):
    x, _ = trim.trim(jnp.array(V), jnp.array(H), ac)
    state = trim.trimmed_state(x[0], jnp.array(V), jnp.array(H))
    controls = trim.trimmed_controls(x[1], x[2])
    ap = ap_mod.engage(state, controls, targets, gains, ac)
    sim = integrate.init_sim(state, jax.random.PRNGKey(0))
    (_, _), (hist, ctrl) = ap_mod.closed_loop_rollout(
        sim, ap, targets, gains, jnp.array(DT), ac, int(seconds / DT)
    )
    return hist, ctrl


def summarise(label, signal, target, unit, tol, settle_band):
    t = np.arange(len(signal)) * DT
    err = np.abs(signal - target)
    settled = np.where(err > settle_band)[0]
    t_settle = t[settled[-1]] if len(settled) else 0.0
    overshoot = (
        (signal.max() - target) if signal[0] < target else (target - signal.min())
    )
    ok = err[-1] < tol
    print(
        f"  {label:22s} final err {signal[-1] - target:+9.3f} {unit:5s}"
        f" settle {t_settle:6.1f}s  overshoot {overshoot:+8.3f}  {'OK' if ok else 'FAIL'}"
    )
    return ok


print(f"=== {NAME} autopilot step responses (dt={DT}s) ===\n")
ok = True

print("Altitude step +300 m")
tg = ap_mod.Targets(
    altitude=jnp.array(H + 300.0), heading=jnp.array(0.0), airspeed=jnp.array(V)
)
hist, ctrl = fly(tg, 300.0)
alt = -np.asarray(hist.pos_ned)[:, 2]
spd = np.linalg.norm(np.asarray(hist.vel_body), axis=1)
ok &= summarise("altitude", alt, H + 300.0, "m", 10.0, 15.0)
ok &= summarise("airspeed (hold)", spd, V, "m/s", 3.0, 3.0)

print("\nHeading step +30 deg")
tg = ap_mod.Targets(
    altitude=jnp.array(H), heading=jnp.array(np.deg2rad(30.0)), airspeed=jnp.array(V)
)
hist, ctrl = fly(tg, 300.0)
eul = np.asarray(jax.vmap(quat_to_euler)(hist.quat))
alt = -np.asarray(hist.pos_ned)[:, 2]
beta = np.asarray(jax.vmap(lambda v: air_data(v)[2])(hist.vel_body))
ok &= summarise("heading", eul[:, 2] * RAD2DEG, 30.0, "deg", 1.0, 2.0)
ok &= summarise("altitude (hold)", alt, H, "m", 30.0, 40.0)
print(f"  peak bank {np.abs(eul[:, 0]).max() * RAD2DEG:.1f} deg,"
      f" peak sideslip {np.abs(beta).max() * RAD2DEG:.2f} deg")

print("\nAirspeed step -15 m/s")
tg = ap_mod.Targets(
    altitude=jnp.array(H), heading=jnp.array(0.0), airspeed=jnp.array(V - 15.0)
)
hist, ctrl = fly(tg, 400.0)
spd = np.linalg.norm(np.asarray(hist.vel_body), axis=1)
alt = -np.asarray(hist.pos_ned)[:, 2]
ok &= summarise("airspeed", spd, V - 15.0, "m/s", 1.0, 1.5)
ok &= summarise("altitude (hold)", alt, H, "m", 30.0, 40.0)

print("\nBumpless engagement (targets = current state, 20 s)")
tg = ap_mod.Targets(altitude=jnp.array(H), heading=jnp.array(0.0), airspeed=jnp.array(V))
hist, ctrl = fly(tg, 20.0)
alt = -np.asarray(hist.pos_ned)[:, 2]
de = np.asarray(ctrl.elevator)
x, _ = trim.trim(jnp.array(V), jnp.array(H), ac)
print(f"  elevator at trim {float(x[1]) * RAD2DEG:+.4f} deg,"
      f" first AP output {de[0] * RAD2DEG:+.4f} deg,"
      f" max excursion {np.abs(de - float(x[1])).max() * RAD2DEG:.4f} deg")
print(f"  altitude excursion {np.abs(alt - H).max():.4f} m")
ok &= np.abs(alt - H).max() < 1.0

print("\n" + ("ALL OK" if ok else "SOME CHECKS FAILED"))
sys.exit(0 if ok else 1)
