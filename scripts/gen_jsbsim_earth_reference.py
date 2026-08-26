"""Freeze the JSBSim Earth-model reference: geodesy, gravity, and every EOM term.

Run with an interpreter that has JSBSim installed. The project venv deliberately
does NOT, so the suite runs anywhere:

    C:/Users/mateusz/AppData/Local/Programs/Python/Python310/python.exe \
        scripts/gen_jsbsim_earth_reference.py

Writes atisim/tests/data/jsbsim_earth_reference.xml.

THE RULE FROM gen_jsbsim_reference.py APPLIES HERE AND IS NOT OPTIONAL:
READ BACK, NEVER ASSUME. Feeding a NOMINAL altitude into the gravity formula
instead of the position JSBSim actually reached puts the agreement at 2.4e-6
relative; reading the ECEF position back off the engine puts it at 3.6e-13.
That factor of ten million is the whole difference between a verification and a
coincidence, and it was measured while writing the design.

This is the fourth and last file under scripts/ that imports jsbsim. Nothing
under atisim/ may import it.
"""

import sys
from pathlib import Path

import jsbsim
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from atisim.units import FT2M, LBF2N, SLUG2KG, SLUG_FT2_TO_KG_M2  # noqa: E402

OUT = ROOT / "atisim" / "tests" / "data" / "jsbsim_earth_reference.xml"

# The probe grid. Latitude spans both hemispheres because a sign error in the
# NED matrix or in J2 is antisymmetric and would cancel on one hemisphere alone.
PROBES = [
    # (lat_deg, lon_deg, alt_ft, psi_deg, phi_deg, aileron, rudder)
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (47.0, 11.0, 30000.0, 30.0, 12.0, 0.0, 0.0),
    (-33.0, 151.0, 20000.0, 240.0, -8.0, 0.0, 0.0),
    (60.0, -120.0, 5000.0, 90.0, 0.0, 0.6, 0.3),
    (-70.0, 45.0, 40000.0, 180.0, 25.0, 0.6, 0.3),
    (10.0, -60.0, 10000.0, 315.0, -20.0, 0.3, -0.4),
]

SETTLE_STEPS = 240   # 2 s at 120 Hz, so the FCS and engines are not in a transient


def f(x):
    return f"{float(x):.12e}"


def vec(a):
    return " ".join(f(v) for v in np.asarray(a).ravel())


def euler_to_matrix(phi, theta, psi):
    """Local NED -> body, 3-2-1. Rebuilt here rather than imported from atisim,
    so the reference cannot inherit a convention error from the code under test.
    """
    cp, sp = np.cos(phi), np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    cy, sy = np.cos(psi), np.sin(psi)
    return np.array([
        [ct * cy, ct * sy, -st],
        [sp * st * cy - cp * sy, sp * st * sy + cp * cy, sp * ct],
        [cp * st * cy + sp * sy, cp * st * sy - sp * cy, cp * ct],
    ])


def ecef_to_ned(lat, lon):
    sf, cf, sl, cl = np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon)
    return np.array([[-sf * cl, -sf * sl, cf], [-sl, cl, 0.0], [-cf * cl, -cf * sl, -sf]])


def probe(fdm_root, lat_deg, lon_deg, alt_ft, psi_deg, phi_deg, aileron, rudder):
    fdm = jsbsim.FGFDMExec(fdm_root)
    fdm.set_debug_level(0)
    fdm.load_model("737")
    fdm["ic/lat-geod-deg"] = lat_deg
    fdm["ic/long-gc-deg"] = lon_deg
    fdm["ic/h-sl-ft"] = alt_ft
    fdm["ic/vc-kts"] = 280.0
    fdm["ic/psi-true-deg"] = psi_deg
    fdm["ic/phi-deg"] = phi_deg
    fdm.run_ic()
    fdm["fcs/aileron-cmd-norm"] = aileron
    fdm["fcs/rudder-cmd-norm"] = rudder
    for _ in range(SETTLE_STEPS):
        fdm.run()

    g = fdm.__getitem__

    # -- everything READ BACK from the engine, never assumed ---------------
    lat = g("position/lat-geod-rad")
    lon = g("position/long-gc-rad")
    r_ecef = np.array([g(f"position/ecef-{c}-ft") for c in "xyz"]) * FT2M
    phi, theta, psi = (g("attitude/phi-rad"), g("attitude/theta-rad"), g("attitude/psi-rad"))

    mass = g("inertia/mass-slugs") * SLUG2KG
    inertia = np.array([
        [g("inertia/ixx-slugs_ft2"), -g("inertia/ixy-slugs_ft2"), g("inertia/ixz-slugs_ft2")],
        [-g("inertia/ixy-slugs_ft2"), g("inertia/iyy-slugs_ft2"), -g("inertia/iyz-slugs_ft2")],
        [g("inertia/ixz-slugs_ft2"), -g("inertia/iyz-slugs_ft2"), g("inertia/izz-slugs_ft2")],
    ]) * SLUG_FT2_TO_KG_M2
    # NOTE THE +Ixz IN THE OFF-DIAGONAL. Measured: the rotational equation closes
    # to 2.4e-14 with +Ixz and is out by 2.3e-3 with -Ixz.

    force = np.array([
        g(f"forces/fb{c}-aero-lbs") + g(f"forces/fb{c}-prop-lbs") + g(f"forces/fb{c}-gear-lbs")
        for c in "xyz"
    ]) * LBF2N
    weight = np.array([g(f"forces/fb{c}-weight-lbs") for c in "xyz"]) * LBF2N
    moment = np.array([
        g("moments/l-total-lbsft"), g("moments/m-total-lbsft"), g("moments/n-total-lbsft")
    ]) * LBF2N * FT2M

    vel_body = np.array([g(f"velocities/{c}-fps") for c in "uvw"]) * FT2M
    vel_ecef = np.array([g(f"velocities/ecef-{c}-fps") for c in "xyz"]) * FT2M
    vel_ned = np.array([
        g("velocities/v-north-fps"), g("velocities/v-east-fps"), g("velocities/v-down-fps")
    ]) * FT2M
    pqr = np.array([g(f"velocities/{c}-rad_sec") for c in "pqr"])
    pqri = np.array([g(f"velocities/{c}i-rad_sec") for c in ("p", "q", "r")])
    uvwdot = np.array([g(f"accelerations/{c}dot-ft_sec2") for c in "uvw"]) * FT2M
    pqrdot = np.array([g(f"accelerations/{c}dot-rad_sec2") for c in "pqr"])
    pqridot = np.array([g(f"accelerations/{c}idot-rad_sec2") for c in ("p", "q", "r")])

    return dict(
        lat=lat, lon=lon, r_ecef=r_ecef, euler=[phi, theta, psi],
        t_l2b=euler_to_matrix(phi, theta, psi), t_e2l=ecef_to_ned(lat, lon),
        mass=mass, inertia=inertia, force=force, weight=weight, moment=moment,
        gravity_magnitude=g("accelerations/gravity-ft_sec2") * FT2M,
        vel_body=vel_body, vel_ecef=vel_ecef, vel_ned=vel_ned,
        pqr=pqr, pqri=pqri, uvwdot=uvwdot, pqrdot=pqrdot, pqridot=pqridot,
    )


def main():
    root = str(Path(jsbsim.__file__).parent)
    lines = ['<?xml version="1.0" encoding="utf-8"?>', "<jsbsim_earth_reference>"]
    lines.append(f"  <version>{jsbsim.FGFDMExec(root).get_version()}</version>")
    for spec in PROBES:
        p = probe(root, *spec)
        lines.append(f'  <probe lat_deg="{f(spec[0])}" lon_deg="{f(spec[1])}" alt_ft="{f(spec[2])}">')
        for key in ("r_ecef", "euler", "force", "weight", "moment", "vel_body",
                    "vel_ecef", "vel_ned", "pqr", "pqri", "uvwdot", "pqrdot", "pqridot"):
            lines.append(f"    <{key}>{vec(p[key])}</{key}>")
        lines.append(f"    <t_l2b>{vec(p['t_l2b'])}</t_l2b>")
        lines.append(f"    <t_e2l>{vec(p['t_e2l'])}</t_e2l>")
        lines.append(f"    <inertia>{vec(p['inertia'])}</inertia>")
        lines.append(f"    <lat>{f(p['lat'])}</lat>")
        lines.append(f"    <lon>{f(p['lon'])}</lon>")
        lines.append(f"    <mass>{f(p['mass'])}</mass>")
        lines.append(f"    <gravity_magnitude>{f(p['gravity_magnitude'])}</gravity_magnitude>")
        lines.append("  </probe>")
    lines.append("</jsbsim_earth_reference>")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT.relative_to(ROOT)}  ({len(lines)} lines, {len(PROBES)} probes)")


if __name__ == "__main__":
    main()
