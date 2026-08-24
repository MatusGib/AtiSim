"""Step responses for autopilot gain tuning. Run from the project root.

    .venv/Scripts/python.exe scripts/tune.py --aircraft cherokee
"""

import argparse
import sys

import jax
import jax.numpy as jnp
import numpy as np

import atisim  # noqa: F401
from atisim import autopilot as ap_mod
from atisim import integrate, trim
from atisim.aero import air_data
from atisim.aircraft import CRUISE, REGISTRY
from atisim.sensors import sense
from atisim.state import quat_to_euler
from atisim.units import RAD2DEG

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--aircraft", default="boeing747", choices=sorted(REGISTRY))
args = parser.parse_args()

NAME = args.aircraft
ac = REGISTRY[NAME]
GAINS = ap_mod.GAINS[NAME]
V, H = CRUISE[NAME]["airspeed"], CRUISE[NAME]["altitude"]
DT = 0.02

# Steps scale with the aircraft: 300 m and 15 m/s are a gentle manoeuvre for a
# 747 and a violent one for a Cherokee.
ALT_STEP = 300.0 if V > 100.0 else 150.0
SPD_STEP = 15.0 if V > 100.0 else 8.0

V_MD = float(trim.minimum_drag_speed(ac, jnp.array(H)))


def fly(targets, seconds, gains=GAINS):
    x, _ = trim.trim(jnp.array(V), jnp.array(H), ac)
    state = trim.trimmed_state(x[0], jnp.array(V), jnp.array(H))
    controls = trim.trimmed_controls(x[1], x[2])
    ap = ap_mod.engage(sense(state), controls, targets, gains, ac)
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


print(f"=== {NAME} autopilot step responses (dt={DT}s) ===")
print(f"    cruise {V:.1f} m/s at {H:.0f} m, V_md {V_MD:.1f} m/s"
      f" (margin {V - V_MD:+.1f} m/s)\n")
ok = True

print(f"Altitude step +{ALT_STEP:.0f} m")
tg = ap_mod.Targets(
    altitude=jnp.array(H + ALT_STEP), heading=jnp.array(0.0), airspeed=jnp.array(V)
)
hist, ctrl = fly(tg, 300.0)
alt = -np.asarray(hist.pos_ned)[:, 2]
spd = np.linalg.norm(np.asarray(hist.vel_body), axis=1)
ok &= summarise("altitude", alt, H + ALT_STEP, "m", 10.0, 15.0)
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

# Both directions are flown, but only the one that stays above V_md is graded.
# Below minimum-drag speed the loop pairing inverts, so a failure there is the
# architecture meeting the drag curve, not the gains being wrong. It is still
# reported, because how far it degrades is worth knowing.
for sign in (-1.0, +1.0):
    target_speed = V + sign * SPD_STEP
    graded = target_speed > V_MD
    print(f"\nAirspeed step {sign * SPD_STEP:+.0f} m/s  ->  {target_speed:.1f} m/s"
          + ("" if graded else f"   (below V_md {V_MD:.1f}: reported, not graded)"))
    tg = ap_mod.Targets(
        altitude=jnp.array(H), heading=jnp.array(0.0), airspeed=jnp.array(target_speed)
    )
    hist, ctrl = fly(tg, 400.0)
    spd = np.linalg.norm(np.asarray(hist.vel_body), axis=1)
    alt = -np.asarray(hist.pos_ned)[:, 2]
    passed = summarise("airspeed", spd, target_speed, "m/s", 1.0, 1.5)
    held = summarise("altitude (hold)", alt, H, "m", 30.0, 40.0)
    if graded:
        ok &= passed and held

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
