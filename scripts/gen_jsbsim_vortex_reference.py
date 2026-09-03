"""Freeze JSBSim's response to the Wingrove & Bach vortex cases.

Run with the interpreter that has JSBSim installed -- NOT the project venv:

    C:/Users/mateusz/AppData/Local/Programs/Python/Python310/python.exe \
        scripts/gen_jsbsim_vortex_reference.py

Writes atisim/tests/data/jsbsim_vortex_reference.xml. The suite and the report
read that file and never import jsbsim, so reference drift shows up in git diff
rather than as a mysterious change in test results. This is the third and last
file in the project that imports jsbsim.

WHAT IS INJECTED, AND WHAT CANNOT BE
------------------------------------
JSBSim's wind is sampled at ONE point and enters only as a translational
velocity. Its property catalog reports atmosphere/{p,q,r}-turb-rad_sec as
READ-ONLY, and a write to q-turb-rad_sec reads back 0.0 after a step (measured,
session 21). So the gust GRADIENT across the airframe -- which atisim carries
via wind.gust_rates and wind.sampled_rates -- has no injection point here and is
structurally absent from the JSBSim side.

That asymmetry is a RESULT of this comparison, not a defect in it. Both engines
are run with the TRANSLATIONAL FIELD ONLY for the headline numbers, which is
strictly like-for-like; atisim's gradient terms are then switched on separately
and the difference is reported as the term JSBSim cannot carry.

Translational injection itself is verified rather than assumed: writing
wind-down-fps = -50 at M 0.78 / 30,000 ft moves alpha 3.687 deg against a
predicted atan(50/776) = 3.69 deg.

WHY THE INITIAL STATE IS WRITTEN OUT IN FULL
--------------------------------------------
*** The two engines do not trim to the same point. *** At the 747 recovery
condition atisim trims to alpha 4.3321 deg and JSBSim to 4.2786. That is not an
entry error -- fed JSBSim's own trim state, atisim returns CL to 3e-6. The cause
is that JSBSim's do_simple_trim converges to Nz = 0.99093, not 1.0, so atisim
needs 0.95% more CL for a true 1 g, worth +0.063 deg of alpha against the
+0.054 observed.

If each engine flew from its OWN trim, the two runs would start at different
angles of attack and every difference downstream would be contaminated by that
offset. So the full initial state -- body velocity VECTOR, rates, Euler angles,
and the achieved control positions -- is recorded here, and atisim is driven
from it. Same reasoning as jsbsim_ref.SweepPoint, which drives atisim with the
velocity vector rather than with (V, alpha, beta).
"""

import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import gen_jsbsim_747 as g747  # noqa: E402
import gen_jsbsim_reference as ref  # noqa: E402
from atisim.atmosphere import density  # noqa: E402
from atisim.units import FT2M, LBF2N, SLUG_FT3_TO_KG_M3  # noqa: E402
from atisim.wind import (  # noqa: E402
    MEHTA_HANNIBAL_ALTITUDE, MEHTA_HANNIBAL_PSI_DEG, MEHTA_HANNIBAL_R0,
    MEHTA_HANNIBAL_V0, MEHTA_HANNIBAL_X_FT, MEHTA_HANNIBAL_Z_FT,
    PARKS_CASES, WINGROVE_CASE_ALTITUDE, WINGROVE_FIG4_CASES,
)

OUT = ROOT / "atisim" / "tests" / "data" / "jsbsim_vortex_reference.xml"

MS2FPS = 1.0 / FT2M
DT = 1.0 / 120.0

# The core traverse is about a second -- Fig. 4's own scale bar -- so the 0.05 s
# the 737 generator samples its doublets at would put roughly twenty samples
# across the entire event. 0.02 s puts fifty across it and still writes a file
# small enough to read.
SAMPLE_EVERY = 0.02

# Core radii of lead-in. scripts/vortex.py defaults to 40 because below about 12
# the 1/r far field launches the aircraft out of equilibrium and contaminates
# the first core. 40 radii is ~5.5 km, which at 236 m/s is 23 s of fixed-control
# flight BEFORE the core -- and JSBSim sheds airspeed at fixed throttle over
# that, so both engines would reach the core off-trim by different amounts and
# the drift would read as a solver difference. 15 is above the far-field
# threshold and a third of the drift, and the drift that remains is recorded per
# encounter and gated by the test suite rather than hoped about.
LEAD_IN_RADII = 15.0

# (case, jsbsim model, atisim registry entry, mach). Altitude comes from
# WINGROVE_CASE_ALTITUDE -- Table 1's own value for that incident.
#
# Cimarron flies the 737 because 33,000 ft is inside that entry's declared
# valid_altitude of [25,000, 35,000] ft and the other two cases are not. It is
# also the only case with published time histories to compare against.
# Hannibal and Morton fly boeing747_jsbsim, whose band is [35,000, 41,000].
CASES = [
    ("cimarron", "737", "boeing737", 0.78),
    ("hannibal", "B747", "boeing747_jsbsim", 0.80),
    ("morton", "B747", "boeing747_jsbsim", 0.80),
]

# THE MEHTA CASE, added session 23d, and it is a different KIND of encounter.
#
# The three above are single Rankine cores with parameters from a two-vortex
# fit. This one is Mehta 1987's converged FIVE-vortex solution -- the field the
# headline atisim result actually flies -- with an oblique traverse at 31 deg.
# It is flown by boeing747_jsbsim rather than boeing747 ON PURPOSE: that entry
# was recovered from JSBSim's own B747, so the two engines are given the same
# aircraft as well as the same field, and any difference left is the SOLVER.
# 37,000 ft is inside that entry's declared band of [35,000, 41,000] ft.
MEHTA_CASE = ("mehta", "B747", "boeing747_jsbsim", 0.80)

# WHICH PAPER EACH CASE'S NUMBERS COME FROM. One source per case, not two.
#
# This used to be `RADIUS_SOURCES = ("wingrove", "parks")` -- every case flown
# at BOTH radii so the Hannibal conflict was "flown rather than argued". Two
# things were wrong with that and session 26 removed it:
#
#   1. The "wingrove" arm was itself a CROSSING. It took Fig. 4's 500 ft radius
#      and flew it in a harness whose spacing and geometry are Parks'. No paper
#      states that vortex, so a run of it measures nothing about either paper.
#   2. After session 22 moved PARKS_CASES to 500 ft the two arms became
#      DEGENERATE -- identical runs under two labels -- and the test that noticed
#      kept them for determinism checking rather than for the radius question
#      they were built for.
#
# `radius_source` survives in the XML, with a better meaning: it names the paper
# each case's parameters were taken from, so the citation travels with the run.
CASE_SOURCE = {
    "hannibal": "parks",     # Parks 1985 p.127: 600 ft, 85 ft/s
    "morton": "parks",       # Parks 1985 p.128: 450 ft, 70 ft/s
    "cimarron": "wingrove",  # only in Wingrove & Bach 1994 Fig. 4
}


def radius_and_strength(case):
    """(r0, v0, source) in metres and m/s, from the ONE paper that states them."""
    source = CASE_SOURCE[case]
    entry = PARKS_CASES[case] if source == "parks" else WINGROVE_FIG4_CASES[case]
    return entry["r0"], entry["v0"], source


def rankine(north, down, core_north, core_down, r0, v0):
    """Parks Eqs. (3)-(6) for one core, in plain numpy.

    Deliberately NOT atisim.wind.vortex_wind. A bug shared by both sides of a
    cross-code comparison is invisible to it, so the field is written twice --
    once here and once in JAX -- and test_jsbsim_vortex reconciles them to 1e-9.
    dpsi is zero, matching wind.py: vortex axes perpendicular to the track.
    """
    along = north - core_north          # l, aircraft beyond the core
    above = core_down - down           # d, aircraft above the core
    r2 = along**2 + above**2
    if r2 < r0**2:
        w_horizontal = v0 * above / r0
        w_up = -v0 * along / r0
    else:
        w_horizontal = v0 * r0 * above / r2
        w_up = -v0 * r0 * along / r2
    return np.array([w_horizontal, 0.0, -w_up])  # NED, z DOWN


def mehta_cores(altitude):
    """(north, down) of Mehta's five cores, in the run's own frame.

    North is shifted so the FIRST core sits LEAD_IN_RADII radii from the start,
    matching where the single-core cases put theirs. Down places core i at
    `altitude - z_i`, since Mehta's z is the aircraft's height ABOVE the core.
    """
    x = np.array(MEHTA_HANNIBAL_X_FT) * FT2M
    z = np.array(MEHTA_HANNIBAL_Z_FT) * FT2M
    north = x - x.min() + LEAD_IN_RADII * MEHTA_HANNIBAL_R0
    return north, -(altitude - z)


def rankine_array(north, down, core_north, core_down, r0, v0, cos_dpsi):
    """Superposition of `rankine` over an array, with an oblique traverse.

    Parks r = (l^2 cos^2 dpsi + d^2)^(1/2): dpsi enters through the ALONG-track
    coordinate alone, which is exactly what `wind.VortexArray.cos_dpsi` does on
    the JAX side. Written here in numpy for the same reason `rankine` is -- a
    bug shared by both sides of a cross-code comparison is invisible to it.
    """
    total = np.zeros(3)
    for cn, cd in zip(core_north, core_down):
        along = (north - cn) * cos_dpsi
        above = cd - down
        r2 = along**2 + above**2
        if r2 < r0**2:
            w_horizontal, w_up = v0 * above / r0, -v0 * along / r0
        else:
            w_horizontal, w_up = v0 * r0 * above / r2, -v0 * r0 * along / r2
        total += np.array([w_horizontal, 0.0, -w_up])
    return total


def configure(model):
    """Point gen_jsbsim_reference at one airframe.

    Explicit for both, rather than relying on the module's 737 defaults, because
    the 747 cases mutate the module and a later 737 case would otherwise inherit
    a B747 elevator mapping and fly a deflection it never commanded.
    """
    if model == "B747":
        g747.install()
    else:
        ref.MODEL = "737"
        ref.ENGINES = (0, 1)
        ref.CDI_COEFF = 0.043
        ref.ELEVATOR_RANGE, ref.AILERON_RANGE, ref.RUDDER_RANGE = 0.3, 0.35, 0.35
        ref.ABSENT = ("CLq", "CYp", "CYr", "CYdr", "Cnp", "Cnda")
        ref.elevator_cmd = lambda de: de / ref.ELEVATOR_RANGE
        ref.rudder_cmd = lambda dr, r_aero: dr / ref.RUDDER_RANGE - r_aero


def matched_altitude(rho, nominal_m):
    """The geometric altitude at which atisim's density equals JSBSim's.

    atisim's ISA uses geometric altitude where the standard uses geopotential.
    qbar is proportional to rho, so flying atisim at the nominal altitude would
    put a same-signed bias on every force in every encounter.
    """
    h = brentq(lambda x: float(density(x)) - rho,
               nominal_m - 400.0, nominal_m + 400.0, xtol=1e-12)
    residual = abs(float(density(h)) - rho) / rho
    if residual > 1e-10:
        raise SystemExit(f"density match failed: residual {residual:.3e}")
    return h, residual


def run_case(case, model, mach, source, pad_radii=6.0, mehta=False):
    if mehta:
        altitude = MEHTA_HANNIBAL_ALTITUDE
        r0, v0 = MEHTA_HANNIBAL_R0, MEHTA_HANNIBAL_V0
        cos_dpsi = math.cos(math.radians(MEHTA_HANNIBAL_PSI_DEG))
    else:
        altitude = WINGROVE_CASE_ALTITUDE[case]
        r0, v0, _ = radius_and_strength(case)
        cos_dpsi = 1.0

    configure(model)
    ref.ALT_FT, ref.MACH = altitude / FT2M, mach
    fdm = ref.trimmed(0)
    fdm["atmosphere/turb-type"] = 0        # no internal turbulence
    fdm["simulation/gravity-model"] = 0    # constant g, matching atisim
    fdm.set_dt(DT)

    # Read the trimmed state back BEFORE any wind. This is what atisim is driven
    # from, so the two engines start at the same point rather than at each
    # engine's own trim -- see the module docstring.
    de_trim = fdm["fcs/elevator-pos-rad"]
    initial = {
        "vel_body": np.array([fdm["velocities/u-aero-fps"],
                              fdm["velocities/v-aero-fps"],
                              fdm["velocities/w-aero-fps"]]) * FT2M,
        "omega": np.array([fdm["velocities/p-rad_sec"],
                           fdm["velocities/q-rad_sec"],
                           fdm["velocities/r-rad_sec"]]),
        "euler": np.array([fdm["attitude/phi-rad"], fdm["attitude/theta-rad"],
                           fdm["attitude/psi-rad"]]),
        "controls": np.array([de_trim, fdm["fcs/left-aileron-pos-rad"]
                              if model == "B747" else fdm["fcs/left-aileron-pos-rad"],
                              fdm["fcs/rudder-pos-rad"]]),
        "throttle": fdm["fcs/throttle-cmd-norm[0]"],
        "thrust": ref.total_thrust_lbs(fdm) * LBF2N,
        "alpha": fdm["aero/alpha-rad"],
        "Nz": fdm["accelerations/Nz"],
    }

    rho = fdm["atmosphere/rho-slugs_ft3"] * SLUG_FT3_TO_KG_M3
    h_match, residual = matched_altitude(rho, altitude)
    V0 = fdm["velocities/vt-fps"] * FT2M

    if mehta:
        core_north, core_down = mehta_cores(altitude)
        duration = (core_north.max() + pad_radii * r0) / V0
    else:
        core_north = LEAD_IN_RADII * r0
        core_down = -altitude
        duration = (core_north + pad_radii * r0) / V0

    values = {
        "r0": r0, "v0": v0, "altitude": altitude, "cos_dpsi": cos_dpsi,
        # Scalar for a single core; the array case carries its geometry in the
        # <cores> element instead and reports the first core here so every
        # encounter still has a well-defined entry point.
        "core_north": float(np.min(core_north)) if mehta else core_north,
        "matched_altitude": h_match, "density_match_residual": residual,
        "density": rho, "airspeed": V0, "duration": duration,
        "mass": fdm["inertia/weight-lbs"] * LBF2N / 9.80665,
        "S": fdm["metrics/Sw-sqft"] * FT2M**2,
        "b": fdm["metrics/bw-ft"] * FT2M,
        "c": fdm["metrics/cbarw-ft"] * FT2M,
        "trim_alpha": initial["alpha"], "trim_elevator": de_trim,
        "trim_Nz": initial["Nz"], "trim_throttle": initial["throttle"],
        "trim_thrust": initial["thrust"],
    }

    samples, next_sample = [], 0.0
    while fdm["simulation/sim-time-sec"] <= duration:
        north = fdm["position/distance-from-start-lat-mt"]
        down = -fdm["position/h-sl-meters"]
        w = (rankine_array(north, down, core_north, core_down, r0, v0, cos_dpsi)
             if mehta else
             rankine(north, down, core_north, core_down, r0, v0))
        fdm["atmosphere/wind-north-fps"] = w[0] * MS2FPS
        fdm["atmosphere/wind-east-fps"] = w[1] * MS2FPS
        fdm["atmosphere/wind-down-fps"] = w[2] * MS2FPS
        # FIXED CONTROLS, held at the trim value every step. The Fig. 8
        # discriminator separates turbulence from MANOEUVRING by whether the
        # pitch response correlates with elevator, so any closed loop here would
        # make the categories ambiguous before the comparison starts.
        fdm["fcs/pitch-trim-cmd-norm"] = ref.elevator_cmd(de_trim)
        t = fdm["simulation/sim-time-sec"]
        if t >= next_sample:
            samples.append({
                "t": t, "north": north, "altitude": -down, "wind": w.copy(),
                "alpha": fdm["aero/alpha-rad"], "theta": fdm["attitude/theta-rad"],
                "q": fdm["velocities/q-rad_sec"], "Nz": fdm["accelerations/Nz"],
                "vtrue": fdm["velocities/vt-fps"] * FT2M,
                "elevator": fdm["fcs/elevator-pos-rad"],
            })
            next_sample += SAMPLE_EVERY
        fdm.run()

    cores = ((core_north, core_down) if mehta else None)
    return values, initial, samples, cores


def main():
    f, vec = ref.f, ref.vec
    L = ['<?xml version="1.0" encoding="utf-8"?>', "<jsbsim_vortex_reference>"]
    L.append("  <provenance>")
    L.append(f"    <jsbsim_version>{ref.jsbsim.FGJSBBase().get_version()}</jsbsim_version>")
    L.append(f"    <dt>{f(DT)}</dt>")
    L.append(f"    <sample_every>{f(SAMPLE_EVERY)}</sample_every>")
    L.append(f"    <lead_in_radii>{f(LEAD_IN_RADII)}</lead_in_radii>")
    L.append("    <gradient_injected>false</gradient_injected>")
    L.append("  </provenance>")

    runs = [(c, m, e, ma, CASE_SOURCE[c], False) for (c, m, e, ma) in CASES]
    runs.append(MEHTA_CASE + ("mehta", True))

    for case, model, entry, mach, source, mehta in runs:
        values, initial, samples, cores = run_case(
            case, model, mach, source, mehta=mehta)
        nz = [s["Nz"] for s in samples]
        wz = [s["wind"][2] for s in samples]
        drift = samples[-1]["vtrue"] - samples[0]["vtrue"]
        print(f"{case:9} {model:5} r0={source:8} "
              f"{values['r0'] / FT2M:6.1f} ft  {len(samples):5d} samples  "
              f"Nz [{min(nz):+.4f}, {max(nz):+.4f}]  "
              f"peak |w_down| {max(abs(v) for v in wz):6.2f} m/s  "
              f"dV {drift:+6.2f} m/s")
        L.append(f'  <encounter case="{case}" model="{model}" '
                 f'aircraft="{entry}" radius_source="{source}" '
                 f'mach="{f(mach)}">')
        for k in sorted(values):
            L.append(f'    <value name="{k}">{f(values[k])}</value>')
        if cores is not None:
            L.append(f'    <cores north="{vec(cores[0])}" '
                     f'down="{vec(cores[1])}"/>')
        L.append("    <initial "
                 f'vel_body="{vec(initial["vel_body"])}" '
                 f'omega="{vec(initial["omega"])}" '
                 f'euler="{vec(initial["euler"])}" '
                 f'controls="{vec(initial["controls"])}" '
                 f'throttle="{f(initial["throttle"])}"/>')
        for s in samples:
            L.append(
                f'    <sample t="{f(s["t"])}" north="{f(s["north"])}" '
                f'altitude="{f(s["altitude"])}" wind="{vec(s["wind"])}" '
                f'alpha="{f(s["alpha"])}" theta="{f(s["theta"])}" '
                f'q="{f(s["q"])}" Nz="{f(s["Nz"])}" '
                f'vtrue="{f(s["vtrue"])}" elevator="{f(s["elevator"])}"/>'
            )
        L.append("  </encounter>")

    L.append("</jsbsim_vortex_reference>")
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print(f"\nwrote {OUT.relative_to(ROOT)}  ({len(L)} lines)")


if __name__ == "__main__":
    main()
