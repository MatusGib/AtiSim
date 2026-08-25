"""Recover a 747 entry from JSBSim's own B747, for the vortex comparison.

Run with the interpreter that has JSBSim installed -- NOT the project venv:

    C:/Users/mateusz/AppData/Local/Programs/Python/Python310/python.exe \
        scripts/gen_jsbsim_747.py

Writes atisim/tests/data/jsbsim_747_reference.xml. Nothing under atisim/ imports
jsbsim; the suite reads the frozen file and drift shows up in git diff.

WHY THIS EXISTS
---------------
atisim already has a 747, from NASA CR-2144, and it is the validated one -- its
modes are checked against Table IX-5 and scripts/checkpoint.py, leewave.py and
vortex.py all fly it. It is NOT this entry and is not replaced by it.

The problem it does not solve is that JSBSim's B747 is a DIFFERENT AEROPLANE:
same inertia tensor, but 249,974 kg against CR-2144's 288,773 kg (15.5%), a
64.46 m span against 59.64 m (8.1%), and 524.7 m^2 of wing against 511.0 (2.7%).
Since n = L/W, a load-factor difference between the two engines flying "a 747"
would be dominated by that mass gap and not by anything about the solvers.

So this recovers a SECOND entry, `boeing747_jsbsim`, from JSBSim's own numbers,
exactly as the 737 entry was recovered -- after which the two engines carry
identical mass, geometry and inertia and a difference between them is a
difference between the engines.

WHAT THIS ENTRY IS NOT
----------------------
*** IT IS NOT A CREDIBLE 747, AND MUST NEVER BE COMPARED TO FLIGHT DATA. ***
B747.xml declares release="ALPHA", author "Unknown", and the same disclaimer the
737 carries: "publicly available data, publicly available technical reports,
textbooks, and guesses... educational and entertainment purposes only".

Worse, and specifically: its CLalpha table is
    (-0.20, -0.68), (0.00, 0.20), (0.23, 1.20), (0.60, 0.60)
and 737.xml's is
    (-0.20, -0.68), (0.00, 0.20), (0.23, 1.20), (0.46, 0.20).
The first THREE POINTS ARE IDENTICAL. Both give CLalpha = 4.35 /rad. A 747 and a
737 do not share a lift-curve slope; this is one Aeromatic template used twice.

That costs the cross-code comparison nothing -- both engines eat the same
numbers, so a difference between them is still honestly a difference between
them. It costs any comparison against Wingrove & Bach's measured g-loads
everything. Use `boeing747` for that.

HOW IT WORKS
------------
It imports scripts/gen_jsbsim_reference.py and rebinds the six names that file
marks as 737-specific. Everything else -- the trim, the differencing, the
pitch-axis least squares, the thrust fit, the AERORP referral -- is the same
code that recovered the 737, so the two entries cannot drift apart in method.

Both of that file's rules still govern here:

1. READ BACK, NEVER ASSUME. The B747 has its own yaw damper with its own
   scheduled gain, and its elevator range is ASYMMETRIC (-0.35 to +0.175). Both
   are measured off the running engine below rather than read off the XML.

2. RECOVER FROM THE ENGINE, NOT THE XML.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import gen_jsbsim_reference as ref  # noqa: E402
from atisim.atmosphere import density  # noqa: E402
from atisim.units import FT2M, LBF2N, SLUG_FT2_TO_KG_M2, SLUG_FT3_TO_KG_M3  # noqa: E402

OUT = ROOT / "atisim" / "tests" / "data" / "jsbsim_747_reference.xml"

# The recovery condition. 38,000 ft sits BETWEEN the two cases this entry is for
# -- Hannibal at 37,000 ft and Morton at 39,000 -- so neither is extrapolated to.
# M 0.80 is where B747.xml's CDmach table still reads exactly zero, so the wave
# drag term is untested here for the same reason it is untested for the 737.
ALT_FT, MACH = 38000.0, 0.80

# The band the entry's own fits were made over. Declared for the same reason the
# 737's is: nothing STOPS use outside it, but checks.recovery_band reports a run
# outside as unchecked rather than as passing. It brackets both cases.
VALID_ALTITUDE_FT = (35000.0, 41000.0)
VALID_MACH = (0.70, 0.90)

# B747.xml's elevator aerosurface_scale is zero-centred and ASYMMETRIC:
# range [-0.35, +0.175]. Measured on the running engine, cmd -> achieved rad:
#   -1.00 -> -0.350   -0.50 -> -0.175   0.00 -> 0.000   +0.50 -> +0.0875
# so the two sides have different gearing and one division cannot express it.
ELEVATOR_DOWN, ELEVATOR_UP = 0.35, 0.175
AILERON_RANGE = RUDDER_RANGE = 0.35

# Derivatives B747.xml does not define. Differs from the 737's list by Cnda,
# which the B747 DOES define. Each is measured and asserted to be zero below, so
# a future JSBSim that gains one fails loudly instead of disagreeing silently.
ABSENT = ("CLq", "CYp", "CYr", "CYdr", "Cnp")


def elevator_cmd(de):
    """Normalised pitch command achieving `de` rad on the B747's split range."""
    return de / ELEVATOR_UP if de > 0.0 else de / ELEVATOR_DOWN


def calibrate_yaw_damper():
    """Measure how much rudder the damper adds per rad/s of yaw rate.

    B747.xml schedules the damper gain on qbar and clips the sum, so the value
    at this condition is not the table entry and is not the 737's. Measured:
    r = +0.02 rad/s with zero rudder command gives dr = +0.014 rad, so the
    damper contributes 0.70 * r and a command must subtract 0.70/RUDDER_RANGE
    = 2.0 times the yaw rate to cancel it. The 737's factor is 1.0.

    Returned rather than hard-coded so a JSBSim that changes the schedule
    changes this number instead of silently biasing every lateral derivative.
    """
    # The compensation must be NULLED while measuring, or what comes back is the
    # residue left by whatever compensation is already installed rather than the
    # damper's own gain. Measured with the 737's default still in place this
    # reads 0.35 -- exactly half the true 0.70 -- because that default has
    # already removed RUDDER_RANGE * r, and installing 0.35 would then leave
    # half the damper uncancelled in every lateral derivative.
    saved = ref.rudder_cmd
    ref.rudder_cmd = lambda dr, r_aero: dr / RUDDER_RANGE
    try:
        fdm = ref.at_state(r=0.02)
        gain = fdm["fcs/rudder-pos-rad"] / fdm["velocities/r-aero-rad_sec"]
    finally:
        ref.rudder_cmd = saved
    # The damper must move the rudder WITH the yaw rate here, and a gain outside
    # (0, 2) would mean the qbar schedule or the clip has changed.
    if not 0.0 < gain < 2.0:
        raise SystemExit(f"yaw damper gain {gain:.4f} is not physical")
    return gain


def install():
    """Rebind gen_jsbsim_reference's 737-specific names onto the B747."""
    ref.MODEL = "B747"
    ref.ENGINES = (0, 1, 2, 3)
    # B747.xml: CDi = 0.0420 CL^2, against the 737's 0.043.
    ref.CDI_COEFF = 0.0420
    ref.ALT_FT, ref.MACH = ALT_FT, MACH
    ref.AILERON_RANGE, ref.RUDDER_RANGE = AILERON_RANGE, RUDDER_RANGE
    # The SYMMETRIC inner bound of the asymmetric range. atisim's Aircraft
    # carries one elevator_limit, and 0.175 is the half that both sides can
    # reach -- so a trim bound built from it can never command a deflection
    # JSBSim would clip. Trim needs about -0.13 rad, comfortably inside.
    ref.ELEVATOR_RANGE = ELEVATOR_UP
    ref.ABSENT = ABSENT
    ref.elevator_cmd = elevator_cmd

    # Installed AFTER the rebinds above, because calibrating it runs the engine.
    gain = calibrate_yaw_damper()
    factor = gain / RUDDER_RANGE
    ref.rudder_cmd = lambda dr, r_aero: dr / RUDDER_RANGE - factor * r_aero
    print(f"yaw damper: {gain:.6f} rad of rudder per rad/s of yaw rate "
          f"(pre-compensation factor {factor:.6f}; the 737's is 1.0)")
    return gain


def check_absent(d, at_trim):
    """Every ABSENT derivative must MEASURE zero, not merely be assumed zero.

    "Zero" cannot mean exact zero. at_state has to step the engine to gear the
    FCS commands onto the surfaces, and a step INTEGRATES: at a pitch-rate
    probe it drifts alpha slightly, which reads back as a small lift-due-to-q
    the model does not have. The floor is therefore DERIVED from that drift --
    the same expression gen_jsbsim_reference.build uses -- rather than picked.
    """
    drift = abs(d["CLa"]) * ref.SETTLE_DT / at_trim["ci2vel"]
    floor = 5.0 * drift
    measured = {k: d[k] for k in ABSENT}
    bad = {k: v for k, v in measured.items() if abs(v) > floor}
    if bad:
        raise SystemExit(
            f"derivatives B747.xml does not define measured above the "
            f"{floor:.2e} settling-drift floor: {bad}\n"
            "The atisim entry carries 0.0 for these, so the comparison would be "
            "wrong. Either JSBSim's model changed or the recovery is picking up "
            "cross-coupling."
        )
    print(f"all {len(ABSENT)} absent derivatives below the derived "
          f"settling-drift floor {floor:.2e} (predicted artifact {drift:.2e}, "
          f"worst measured {max(abs(v) for v in measured.values()):.2e})")
    return measured


def verify_entry_reproduces_trim(entry, d, at_trim):
    """The recovered set must rebuild JSBSim's own trim lift and moment.

    This is what catches a sign error or a missed intercept: the entry is not a
    list of plausible numbers, it is a model that has to reproduce the point it
    came from.

    The two axes are held to DIFFERENT standards, and deliberately.

    CL0 is solved algebraically as the intercept that reproduces this exact
    point, so lift must come back to floating-point. Anything else is a bug.

    Cm cannot. Its five constants come from a least-squares fit over 27 crossed
    states (recover_pitch_axis), so the most it can reproduce any single point
    to is that fit's own worst residual -- which the fit reports. Holding it to
    1e-9 would fail a correct recovery, which is exactly what it did on the
    first run here. The comparison is against the residual, and the ALPHADOT and
    q columns have to be carried: at_state reads a trim whose alphadot is not
    identically zero, and dropping that column moves Cm by more than the
    residual being tested for.
    """
    a, de = at_trim["alpha"], at_trim["de"]
    CL = entry["CL0"] + d["CLa"] * a + d["CLde"] * de

    # CL0 was solved against JSBSim's TRIM alpha and elevator; `at_trim` is the
    # state read back after at_state re-established that point, and the two
    # differ in the last few digits by the round trip through degrees() and the
    # FCS gearing. That makes the reconstruction miss by exactly
    #     CLa * (alpha_read - alpha_trim) + CLde * (de_read - de_trim)
    # which is an identity, not a tolerance -- so it is PREDICTED and then
    # required to hold to floating point, rather than absorbed into a loose
    # bound. A bound of 1e-9 relative rejected a correct recovery here; a bound
    # loose enough to pass it would no longer catch a real intercept error.
    d_alpha = a - entry["_trim_alpha"]
    d_de = de - entry["_trim_elevator"]
    predicted = d["CLa"] * d_alpha + d["CLde"] * d_de
    miss = abs((CL - at_trim["CL"]) - predicted)
    print(f"entry reproduces trim: CL {CL:.12f} vs {at_trim['CL']:.12f}")
    print(f"                       gap {CL - at_trim['CL']:+.3e}, predicted "
          f"from the state round trip {predicted:+.3e}, unexplained {miss:.2e} "
          f"(alpha {d_alpha:+.2e} rad, elevator {d_de:+.2e} rad)")
    if miss > 1e-14 * max(1.0, abs(at_trim["CL"])):
        raise SystemExit(
            f"CL0 misses its own trim by {miss:.3e} more than the state round "
            "trip accounts for. It is solved as that intercept, so this is a "
            "bug, not a fit residual."
        )

    q_hat = at_trim["q"] * at_trim["ci2vel"]
    adot_hat = at_trim["alphadot"] * at_trim["ci2vel"]
    Cm = (d["Cm0"] + d["Cma"] * a + d["Cmq"] * q_hat
          + d["Cmadot"] * adot_hat + d["Cmde"] * de)
    dCm = abs(Cm - at_trim["Cm"])
    bound = 2.0 * d["_pitch_fit_residual"]
    print(f"                       Cm {Cm:+.6e} vs {at_trim['Cm']:+.6e} "
          f"({dCm:.2e} absolute, fit residual {d['_pitch_fit_residual']:.2e}, "
          f"condition {d['_pitch_fit_condition']:.1f})")
    if dCm > bound:
        raise SystemExit(
            f"Cm misses its own trim by {dCm:.3e}, more than twice the pitch "
            f"fit's own worst residual ({d['_pitch_fit_residual']:.3e}). That "
            "is not fit error -- something in the pitch axis is wrong."
        )


def main():
    gain = install()
    print(f"===== B747: {ALT_FT:.0f} ft, M {MACH} =====")

    lon = ref.trimmed(0)
    trim_alpha = lon["aero/alpha-rad"]
    trim_de = lon["fcs/elevator-pos-rad"]
    throttle = lon["fcs/throttle-cmd-norm[0]"]
    thrust = ref.total_thrust_lbs(lon) * LBF2N
    print(f"trim: alpha {np.degrees(trim_alpha):.4f} deg, elevator "
          f"{trim_de:+.6f} rad, throttle {throttle:.4f}, thrust {thrust:.0f} N")

    d = ref.recover(np.degrees(trim_alpha), trim_de)
    at_trim = ref.read_state(ref.at_state(
        alpha_deg=np.degrees(trim_alpha), de=trim_de))
    check_absent(d, at_trim)

    # Density match. atisim's ISA uses geometric altitude where the standard
    # uses geopotential, so a nominal 38,000 ft in atisim is not JSBSim's
    # 38,000 ft. qbar is proportional to rho, so an unmatched altitude puts a
    # same-signed bias on every force in every layer of the comparison.
    rho = lon["atmosphere/rho-slugs_ft3"] * SLUG_FT3_TO_KG_M3
    h_js = ALT_FT * FT2M
    h_match = brentq(lambda h: float(density(h)) - rho,
                     h_js - 300.0, h_js + 300.0, xtol=1e-12)
    residual = abs(float(density(h_match)) - rho) / rho
    if residual > 1e-10:
        raise SystemExit(f"density match failed: residual {residual:.3e}")
    print(f"density match residual: {residual:.3e}  (altitude shift "
          f"{(h_match - h_js) / FT2M:+.2f} ft)")

    entry = ref.aircraft_entry(lon, d, at_trim, dict(
        alpha=trim_alpha, elevator=trim_de, throttle=throttle,
        thrust=thrust, bank=0.0), rho, h_match)
    # The alpha and elevator CL0 was solved against, kept so the verification
    # below can predict the state round-trip exactly rather than tolerate it.
    entry["_trim_alpha"], entry["_trim_elevator"] = trim_alpha, trim_de
    verify_entry_reproduces_trim(entry, d, at_trim)

    inertia = np.array([
        [lon["inertia/ixx-slugs_ft2"], 0.0, lon["inertia/ixz-slugs_ft2"]],
        [0.0, lon["inertia/iyy-slugs_ft2"], 0.0],
        [lon["inertia/ixz-slugs_ft2"], 0.0, lon["inertia/izz-slugs_ft2"]],
    ]) * SLUG_FT2_TO_KG_M2

    f, vec = ref.f, ref.vec
    L = ['<?xml version="1.0" encoding="utf-8"?>', "<jsbsim_747_reference>"]
    L.append("  <provenance>")
    L.append(f"    <jsbsim_version>{ref.jsbsim.FGJSBBase().get_version()}</jsbsim_version>")
    L.append(f"    <model>{ref.MODEL}</model>")
    L.append(f"    <yaw_damper_gain>{f(gain)}</yaw_damper_gain>")
    L.append("  </provenance>")
    L.append("  <condition>")
    L.append(f"    <altitude_m>{f(h_js)}</altitude_m>")
    L.append(f"    <matched_altitude_m>{f(h_match)}</matched_altitude_m>")
    L.append(f"    <density>{f(rho)}</density>")
    L.append(f"    <density_match_residual>{f(residual)}</density_match_residual>")
    L.append(f"    <mach>{f(MACH)}</mach>")
    L.append(f"    <airspeed>{f(lon['velocities/vt-fps'] * FT2M)}</airspeed>")
    L.append(f"    <mass>{f(lon['inertia/weight-lbs'] * LBF2N / 9.80665)}</mass>")
    L.append(f"    <inertia>{vec(inertia)}</inertia>")
    L.append("  </condition>")
    L.append("  <trim>")
    for k, v in (("alpha", trim_alpha), ("elevator", trim_de),
                 ("throttle", throttle), ("thrust", thrust)):
        L.append(f"    <{k}>{f(v)}</{k}>")
    L.append("  </trim>")
    L.append("  <derivatives>")
    for k in sorted(d):
        L.append(f'    <derivative name="{k}" value="{f(d[k])}"/>')
    L.append("  </derivatives>")
    L.append("  <aircraft_entry>")
    for k in sorted(k for k in entry if not k.startswith("_")):
        L.append(f'    <value name="{k}">{f(entry[k])}</value>')
    L.append("  </aircraft_entry>")
    L.append("  <validity>")
    L.append(f"    <mach_low>{f(VALID_MACH[0])}</mach_low>")
    L.append(f"    <mach_high>{f(VALID_MACH[1])}</mach_high>")
    L.append(f"    <altitude_low>{f(VALID_ALTITUDE_FT[0] * FT2M)}</altitude_low>")
    L.append(f"    <altitude_high>{f(VALID_ALTITUDE_FT[1] * FT2M)}</altitude_high>")
    L.append("  </validity>")
    L.append("</jsbsim_747_reference>")

    OUT.write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print(f"\nwrote {OUT.relative_to(ROOT)}  ({len(L)} lines)")


if __name__ == "__main__":
    main()
