"""Generate the frozen JSBSim reference data for the 737 cross-code comparison.

Run with an interpreter that has JSBSim installed (NOT the project venv, which
deliberately does not):

    python scripts/gen_jsbsim_reference.py

Writes flightsim/tests/data/jsbsim_737_reference.xml. The test suite reads that
file and never imports jsbsim, so the suite runs anywhere and reference drift
shows up in git diff. This is the ONLY file in the project that imports jsbsim.

Design: docs/superpowers/specs/2026-08-20-jsbsim-737-verification-design.md

Two rules govern everything here, and both come from measurements made before
the design was written:

1. READ BACK, NEVER ASSUME. JSBSim's FCS injects deflections that were never
   commanded -- the yaw damper moves the rudder 0.0035 rad at M 0.78 with zero
   rudder command. Every state and surface position is read back from the engine
   after the step and used as the actual independent variable.

2. RECOVER FROM THE ENGINE, NOT THE XML. JSBSim applies aero forces at the
   AERORP (x = 625 in) and takes moments about the CG (x = 610.8 in). That
   0.096 cbar offset makes the engine's effective Cma -1.064 where 737.xml's
   constant says -0.6. Differencing the running engine folds the offset in by
   construction.
"""

import math
import sys
from pathlib import Path

import jsbsim
import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from flightsim.atmosphere import density, speed_of_sound  # noqa: E402
from flightsim.units import (  # noqa: E402
    FT2M, LBF2N, SLUG_FT2_TO_KG_M2, SLUG_FT3_TO_KG_M3,
)

OUT = ROOT / "flightsim" / "tests" / "data" / "jsbsim_737_reference.xml"

# The condition everything is linearised about: JSBSim's own 737_test script.
ALT_FT, MACH, GAMMA_DEG = 30000.0, 0.78, 0.0
TURN_ALT_FT, TURN_VT_FPS, TURN_BANK_DEG = 25000.0, 750.0, 30.0

# FCS gearings, from 737.xml's aerosurface_scale blocks. Used only to COMMAND a
# deflection; the achieved deflection is always read back.
ELEVATOR_RANGE, AILERON_RANGE, RUDDER_RANGE = 0.3, 0.35, 0.35

# Step used only to let the FCS gear commands onto the surfaces. See at_state.
SETTLE_DT = 1e-6

# Derivatives 737.xml does not define at all. Asserted to measure zero rather
# than assumed -- if JSBSim ever gains one, this fails loudly instead of
# silently disagreeing with a flightsim entry that still carries 0.0.
ABSENT = ("CLq", "CYp", "CYr", "CYdr", "Cnp", "Cnda")

FT2 = FT2M**2
PSF2PA = LBF2N / FT2
LBFT2NM = LBF2N * FT2M


# --------------------------------------------------------------------------
# driving JSBSim
# --------------------------------------------------------------------------
def new_fdm():
    fdm = jsbsim.FGFDMExec(jsbsim.get_default_root_dir())
    fdm.set_debug_level(0)
    fdm.load_model("737")
    return fdm


def _clean(fdm):
    """Gear up, flaps up, engines lit: the configuration every case shares."""
    for i in (0, 1):
        fdm[f"propulsion/engine[{i}]/set-running"] = 1
    fdm["gear/gear-cmd-norm"] = 0
    fdm["gear/gear-pos-norm"] = 0
    fdm["fcs/flap-cmd-norm"] = 0
    fdm["fcs/flap-pos-norm"] = 0
    fdm["fcs/speedbrake-cmd-norm"] = 0
    fdm["fcs/spoiler-cmd-norm"] = 0


def at_state(alpha_deg=0.0, beta_deg=0.0, p=0.0, q=0.0, r=0.0,
             de=0.0, da=0.0, dr=0.0, alt_ft=ALT_FT, vt_fps=None):
    """Put JSBSim at a state and step it once so the aero outputs are populated.

    Commanded values are approximate by design: the FCS gears them and the yaw
    damper adds to them. `read_state` reports what actually happened.
    """
    fdm = new_fdm()
    fdm["ic/h-sl-ft"] = alt_ft
    if vt_fps is None:
        fdm["ic/mach"] = MACH
    else:
        fdm["ic/vt-fps"] = vt_fps
    fdm["ic/alpha-deg"] = alpha_deg
    fdm["ic/beta-deg"] = beta_deg
    fdm["ic/p-rad_sec"] = p
    fdm["ic/q-rad_sec"] = q
    fdm["ic/r-rad_sec"] = r
    fdm.run_ic()
    _clean(fdm)
    fdm["fcs/pitch-trim-cmd-norm"] = de / ELEVATOR_RANGE
    fdm["fcs/aileron-cmd-norm"] = da / AILERON_RANGE
    # Pre-compensate the yaw damper so the commanded rudder is what is achieved.
    fdm["fcs/rudder-cmd-norm"] = dr / RUDDER_RANGE - fdm["velocities/r-aero-rad_sec"]
    # A step is needed for the FCS to gear the commands onto the surfaces, but
    # run() also INTEGRATES, and at the default 1/120 s a pitch rate drifts
    # alpha enough to fake a lift-due-to-pitch-rate derivative the model does
    # not have: q = +-0.02 rad/s drifts alpha 3.3e-4 rad, which reads as
    # CLq = +4.57 against a measured +4.60. Stepping 1e-6 s instead leaves the
    # FCS fully settled -- it is pure gain blocks, no actuator lags -- while
    # cutting the drift to 4e-8 rad, an apparent CLq of +0.0005.
    fdm.set_dt(SETTLE_DT)
    fdm.run()
    return fdm


def read_state(fdm):
    """Everything the comparison needs, in SI, read back from the engine."""
    qbar = fdm["aero/qbar-psf"]
    S, b, c = fdm["metrics/Sw-sqft"], fdm["metrics/bw-ft"], fdm["metrics/cbarw-ft"]
    qS = qbar * S
    return dict(
        # The AIR-RELATIVE body velocity, recorded directly rather than as
        # (V, alpha, beta) so the comparison never depends on the two engines
        # defining alpha and beta the same way. (They do -- atan2(w,u) and
        # asin(v/V), checked to nine decimals -- but a recorded vector cannot
        # drift out of agreement later.)
        vel_body=np.array([fdm["velocities/u-aero-fps"],
                           fdm["velocities/v-aero-fps"],
                           fdm["velocities/w-aero-fps"]]) * FT2M,
        alpha=fdm["aero/alpha-rad"],
        alphadot=fdm["aero/alphadot-rad_sec"],
        beta=fdm["aero/beta-rad"],
        p=fdm["velocities/p-aero-rad_sec"],
        q=fdm["velocities/q-aero-rad_sec"],
        r=fdm["velocities/r-aero-rad_sec"],
        de=fdm["fcs/elevator-pos-rad"],
        da=fdm["fcs/left-aileron-pos-rad"],
        dr=fdm["fcs/rudder-pos-rad"],
        mach=fdm["velocities/mach"],
        a_sound=fdm["atmosphere/a-fps"] * FT2M,
        vt=fdm["velocities/vt-fps"] * FT2M,
        bi2vel=fdm["aero/bi2vel"],
        ci2vel=fdm["aero/ci2vel"],
        # CL, CD, CY, Cl, Cm, Cn -- the same six flightsim's aero.coefficients
        # returns, in the same order and the same senses (verified in the spec's
        # sign-convention table).
        CL=fdm["forces/fwz-aero-lbs"] / qS,
        CD=fdm["forces/fwx-aero-lbs"] / qS,
        CY=fdm["forces/fwy-aero-lbs"] / qS,
        Cl=fdm["moments/l-aero-lbsft"] / (qS * b),
        Cm=fdm["moments/m-aero-lbsft"] / (qS * c),
        Cn=fdm["moments/n-aero-lbsft"] / (qS * b),
    )


def trimmed(mode, alt_ft=ALT_FT, vt_fps=None, bank_deg=0.0, latitude_deg=47.0):
    """JSBSim's own trim. mode 0 = longitudinal, 5 = steady turn.

    latitude_deg must be set BEFORE run_ic(): an ic/ property written after
    it has no effect, which silently made the Coriolis probe read exactly
    zero the first time this ran.
    """
    fdm = new_fdm()
    fdm["ic/lat-gc-deg"] = latitude_deg
    fdm["ic/h-sl-ft"] = alt_ft
    if vt_fps is None:
        fdm["ic/mach"] = MACH
    else:
        fdm["ic/vt-fps"] = vt_fps
    fdm["ic/gamma-deg"] = GAMMA_DEG
    fdm["ic/phi-deg"] = bank_deg
    fdm.run_ic()
    _clean(fdm)
    fdm["simulation/do_simple_trim"] = mode
    return fdm


# --------------------------------------------------------------------------
# derivative recovery
# --------------------------------------------------------------------------
def central(coefficient, key, step, base, **fixed):
    """d(coefficient)/d(key) by central difference about `base`.

    The denominator is the READ-BACK difference in the independent variable, not
    the commanded step, so FCS gearing and yaw-damper injection cancel exactly.
    """
    lo = read_state(at_state(**{**fixed, key: base - step}))
    hi = read_state(at_state(**{**fixed, key: base + step}))
    return lo, hi


def _slope(lo, hi, coefficient, var):
    return (hi[coefficient] - lo[coefficient]) / (hi[var] - lo[var])


def recover(trim_alpha_deg, trim_de):
    """The full derivative set, differenced about the trimmed cruise point."""
    base = dict(alpha_deg=trim_alpha_deg, de=trim_de)
    d = {}

    # --- longitudinal: alpha ---
    lo, hi = central(None, "alpha_deg", 1.0, trim_alpha_deg, de=trim_de)
    d["CLa"] = _slope(lo, hi, "CL", "alpha")
    d["Cma"] = _slope(lo, hi, "Cm", "alpha")
    d["CDa_engine"] = _slope(lo, hi, "CD", "alpha")  # recorded, no flightsim home

    # --- longitudinal: elevator ---
    lo, hi = central(None, "de", 0.02, trim_de, alpha_deg=trim_alpha_deg)
    d["CLde"] = _slope(lo, hi, "CL", "de")
    d["Cmde"] = _slope(lo, hi, "Cm", "de")

    # --- pitch rate. Non-dimensional q_hat = q * c / (2V), JSBSim's ci2vel. ---
    lo, hi = central(None, "q", 0.02, 0.0, **base)
    dq = (hi["q"] - lo["q"]) * hi["ci2vel"]
    d["Cmq"] = (hi["Cm"] - lo["Cm"]) / dq
    d["CLq"] = (hi["CL"] - lo["CL"]) / dq

    # --- sideslip ---
    lo, hi = central(None, "beta_deg", 2.0, 0.0, **base)
    d["CYb"] = _slope(lo, hi, "CY", "beta")
    d["Clb"] = _slope(lo, hi, "Cl", "beta")
    d["Cnb"] = _slope(lo, hi, "Cn", "beta")

    # --- roll rate: p_hat = p * b / (2V) ---
    lo, hi = central(None, "p", 0.05, 0.0, **base)
    dp = (hi["p"] - lo["p"]) * hi["bi2vel"]
    d["Clp"] = (hi["Cl"] - lo["Cl"]) / dp
    d["Cnp"] = (hi["Cn"] - lo["Cn"]) / dp
    d["CYp"] = (hi["CY"] - lo["CY"]) / dp

    # --- yaw rate. The yaw damper is pre-compensated in at_state, so dr stays
    #     at its base value and this is a clean r derivative. ---
    lo, hi = central(None, "r", 0.02, 0.0, **base)
    dr_hat = (hi["r"] - lo["r"]) * hi["bi2vel"]
    d["Cnr"] = (hi["Cn"] - lo["Cn"]) / dr_hat
    d["Clr"] = (hi["Cl"] - lo["Cl"]) / dr_hat
    d["CYr"] = (hi["CY"] - lo["CY"]) / dr_hat

    # --- aileron. flightsim's da is JSBSim's LEFT aileron position. ---
    lo, hi = central(None, "da", 0.05, 0.0, **base)
    d["Clda"] = _slope(lo, hi, "Cl", "da")
    d["Cnda"] = _slope(lo, hi, "Cn", "da")

    # --- rudder ---
    lo, hi = central(None, "dr", 0.05, 0.0, **base)
    d["Cldr"] = _slope(lo, hi, "Cl", "dr")
    d["Cndr"] = _slope(lo, hi, "Cn", "dr")
    d["CYdr"] = _slope(lo, hi, "CY", "dr")
    return d


# --------------------------------------------------------------------------
# linearisation
# --------------------------------------------------------------------------
def _sce_blocks(text):
    """Every [...] block in a Scilab .sce file, in order, as float lists."""
    blocks, depth, buf = [], 0, []
    for ch in text:
        if ch == "[":
            depth += 1
            if depth == 1:
                buf = []
                continue
        if ch == "]":
            depth -= 1
            if depth == 0:
                blocks.append([float(v) for v in
                               buf_join(buf).replace(";", ",").split(",") if v.strip()])
                continue
        if depth >= 1:
            buf.append(ch)
    return blocks


def buf_join(buf):
    return "".join(buf)


def linearization(work_dir):
    """Trim, linearise, and re-derive the state ordering from A's structure.

    JSBSim writes 737_lin.sce with no state labels. Rather than trusting a
    remembered ordering, the ordering is DERIVED from the rows that must be pure
    integrators for any ordering to be right:

        theta_dot = q            -> a row with a single 1.0
        psi_dot   = r / cos(th)  -> coefficient 1/cos(theta0)
        h_dot     = vt(th - al)  -> +-vt in two columns

    and then checked against [vt, alpha, theta, q, beta, phi, p, psi, r, lat,
    lon, h]. A future JSBSim reordering fails here instead of silently comparing
    the wrong states.
    """
    import os
    cwd = os.getcwd()
    os.chdir(work_dir)
    try:
        fdm = trimmed(0)
        theta0 = fdm["attitude/theta-rad"]
        vt0 = fdm["velocities/vt-fps"]
        fdm["simulation/do_linearization"] = 0
        text = (Path(work_dir) / "737_lin.sce").read_text()
    finally:
        os.chdir(cwd)

    blocks = _sce_blocks(text)
    x0 = np.array(blocks[0])
    u0 = np.array(blocks[1])
    A = np.array(blocks[2]).reshape(12, 12)
    B = np.array(blocks[3]).reshape(12, 4)

    # --- re-derive the ordering ---
    i_theta = next(i for i in range(12)
                   if np.count_nonzero(np.abs(A[i]) > 1e-9) == 1
                   and abs(A[i][np.argmax(np.abs(A[i]))] - 1.0) < 1e-9
                   and i not in (5,))
    i_q = int(np.argmax(np.abs(A[i_theta])))
    inv_cos = 1.0 / math.cos(theta0)
    i_psi, i_r = None, None
    for i in range(12):
        hits = np.where(np.abs(A[i] - inv_cos) < 1e-6)[0]
        if len(hits) == 1 and np.count_nonzero(np.abs(A[i]) > 1e-9) == 1:
            i_psi, i_r = i, int(hits[0])
            break
    i_h = int(np.argmax([np.count_nonzero(np.abs(row) > 0.5 * vt0) for row in A]))
    hits = np.where(np.abs(A[i_h]) > 0.5 * vt0)[0]
    i_alpha, i_theta2 = int(hits[0]), int(hits[1])

    derived = dict(theta=i_theta, q=i_q, psi=i_psi, r=i_r, h=i_h, alpha=i_alpha)
    expected = dict(theta=2, q=3, psi=7, r=8, h=11, alpha=1)
    if derived != expected or i_theta2 != 2:
        raise SystemExit(
            "JSBSim's linearisation state ordering is not the expected\n"
            "[vt, alpha, theta, q, beta, phi, p, psi, r, lat, lon, h].\n"
            f"derived {derived} (theta col in h_dot row: {i_theta2}), expected {expected}.\n"
            "The layer-3 comparison would silently compare the wrong states; fix the\n"
            "block indices in this script and in test_jsbsim_737.py before continuing."
        )
    return A, B, x0, u0


# --------------------------------------------------------------------------
# trajectory
# --------------------------------------------------------------------------
def doublet(t, amplitude, start=1.0, width=2.0):
    if start <= t < start + width:
        return amplitude
    if start + width <= t < start + 2 * width:
        return -amplitude
    return 0.0


def kick(t, amplitude, start=1.0, width=1.0):
    return amplitude if start <= t < start + width else 0.0


def fly(case, duration=20.0, dt=1.0 / 120.0, sample_every=0.05, latitude_deg=47.0):
    """Fly one prescribed-surface case and sample the state.

    Sampled at 0.05 s, not the 0.25 s first tried. The consumer holds each
    control sample until the next one, and the yaw damper moves the rudder
    CONTINUOUSLY in response to yaw rate, so a coarse sample makes the replay
    fly a stale rudder. That is a sampling artifact and it dominated the
    rudder-kick divergence (5.43 m/s at 0.25 s) until the rate came up.

    The surface HISTORY is prescribed, not the stick: the rudder command is
    pre-compensated for the yaw damper each step so the achieved surface follows
    the schedule, and the achieved value is what gets recorded. flightsim is then
    driven with the achieved deflections, so the FCS cannot contribute to any
    difference between the two engines.
    """
    fdm = trimmed(0, latitude_deg=latitude_deg)
    try:
        fdm["simulation/gravity-model"] = 0  # constant g, matching flightsim
    except Exception:
        pass
    fdm.set_dt(dt)

    schedule = {
        "elevator_doublet": lambda t: (doublet(t, 0.02), 0.0, 0.0),
        "rudder_kick": lambda t: (0.0, 0.0, kick(t, 0.05)),
    }[case]
    de_trim = fdm["fcs/elevator-pos-rad"]

    samples, next_sample, worst_miss = [], 0.0, 0.0
    while fdm["simulation/sim-time-sec"] <= duration:
        t = fdm["simulation/sim-time-sec"]
        d_de, d_da, d_dr = schedule(t)
        fdm["fcs/pitch-trim-cmd-norm"] = (de_trim + d_de) / ELEVATOR_RANGE
        fdm["fcs/aileron-cmd-norm"] = d_da / AILERON_RANGE
        fdm["fcs/rudder-cmd-norm"] = (
            d_dr / RUDDER_RANGE - fdm["velocities/r-aero-rad_sec"]
        )
        fdm.run()
        worst_miss = max(worst_miss, abs(fdm["fcs/rudder-pos-rad"] - d_dr))
        t = fdm["simulation/sim-time-sec"]
        if t >= next_sample:
            samples.append(dict(
                t=t,
                vel_body=np.array([fdm["velocities/u-fps"], fdm["velocities/v-fps"],
                                   fdm["velocities/w-fps"]]) * FT2M,
                omega=np.array([fdm["velocities/p-rad_sec"], fdm["velocities/q-rad_sec"],
                                fdm["velocities/r-rad_sec"]]),
                euler=np.array([fdm["attitude/phi-rad"], fdm["attitude/theta-rad"],
                                fdm["attitude/psi-rad"]]),
                altitude=fdm["position/h-sl-ft"] * FT2M,
                controls=np.array([fdm["fcs/elevator-pos-rad"],
                                   fdm["fcs/left-aileron-pos-rad"],
                                   fdm["fcs/rudder-pos-rad"]]),
                thrust=(fdm["propulsion/engine[0]/thrust-lbs"]
                        + fdm["propulsion/engine[1]/thrust-lbs"]) * LBF2N,
            ))
            next_sample += sample_every
    return samples, worst_miss


def coriolis_contribution(case):
    """How much of any trajectory difference is Earth rotation, as a number.

    Same case at latitude 0 and 47 degrees. flightsim is flat-Earth and
    non-rotating, so this bounds what it cannot reproduce even in principle.
    """
    a, _ = fly(case, latitude_deg=0.0)
    b, _ = fly(case, latitude_deg=47.0)
    n = min(len(a), len(b))
    return max(float(np.max(np.abs(a[i]["vel_body"] - b[i]["vel_body"])))
               for i in range(n))


# --------------------------------------------------------------------------
# the flightsim entry: everything _boeing_737() needs, derived here so that
# every literal in aircraft.py traces to a line of this file's output
# --------------------------------------------------------------------------
def thrust_at(alt_ft, mach, throttle):
    """Total thrust with the turbine initialised already spooled to `throttle`.

    Setting the throttle BEFORE set-running is what makes this work: the turbine
    then initialises at that power rather than at idle, and the reading is
    stable from the first step. Spooling up instead needs thousands of steps,
    during which the aircraft accelerates and the reading is of a different
    flight condition than the one asked for.
    """
    fdm = new_fdm()
    fdm["ic/h-sl-ft"] = alt_ft
    fdm["ic/mach"] = mach
    fdm["ic/gamma-deg"] = GAMMA_DEG
    fdm.run_ic()
    for i in (0, 1):
        fdm[f"fcs/throttle-cmd-norm[{i}]"] = throttle
        fdm[f"propulsion/engine[{i}]/set-running"] = 1
    fdm["gear/gear-pos-norm"] = 0
    fdm.set_dt(SETTLE_DT)
    fdm.run()
    return (fdm["propulsion/engine[0]/thrust-lbs"]
            + fdm["propulsion/engine[1]/thrust-lbs"]) * LBF2N


def thrust_fit(throttle):
    """(max_thrust, thrust_lapse, mach_ram) fitted at the trim throttle.

    Fitted AT THE TRIM THROTTLE, not at the engine rating, because JSBSim blends
    idle and military thrust nonlinearly with throttle -- thrust/throttle runs
    from 22 kN at throttle 0.2 to 95 kN at 1.0, a factor of 4.3 -- while
    flightsim's model is linear in throttle. A fit that reproduced the rating
    would be wrong by 70% at the condition being compared.

    thrust_lapse comes out near 0.75 rather than the 1.0 the CFM56's MilThrust
    table shows at M = 0: that table is FULL power, and the idle/mil blend at a
    part-throttle cruise setting lapses differently.
    """
    # Both fits are over the CRUISE BAND only, matching the rest of the design.
    # A single power law over 10,000-40,000 ft leaves a 6.75% worst residual,
    # because JSBSim's idle/mil blend is not a power law over that span; over
    # the band the comparison actually flies it is far tighter. Values outside
    # the band are recorded in the limitations, not fitted to.
    machs = np.array([0.60, 0.70, 0.78, 0.85, 0.95])
    T_m = np.array([thrust_at(ALT_FT, m, throttle) for m in machs])
    alts = np.array([25000.0, 27500.0, 30000.0, 32500.0, 35000.0])
    T_h = np.array([thrust_at(h, MACH, throttle) for h in alts])
    ratio = np.array([float(density(h * FT2M)) / 1.225 for h in alts])

    lapse = float(np.polyfit(np.log(ratio), np.log(T_h), 1)[0])
    c0, c1 = np.linalg.lstsq(
        np.vstack([np.ones_like(machs), machs**2]).T, T_m, rcond=None
    )[0]
    ram = float(c1 / c0)

    rho_trim = float(density(ALT_FT * FT2M)) / 1.225
    fmax = thrust_at(ALT_FT, MACH, throttle) / (
        throttle * rho_trim**lapse * (1.0 + ram * MACH**2)
    )
    mach_res = float(np.max(np.abs(c0 * (1 + ram * machs**2) - T_m) / T_m))
    alt_res = float(np.max(np.abs(
        fmax * throttle * ratio**lapse * (1 + ram * MACH**2) - T_h) / T_h))
    return fmax, lapse, ram, mach_res, alt_res


def wave_drag_parameters(CL_trim, onset=0.79):
    """(sweep, t/c, kappa) placing this project's Korn/Lock rise at JSBSim's.

    737.xml gives NO sweep and NO thickness, so these cannot be read off. They
    are chosen so that flightsim's M_crit lands on the Mach at which JSBSim's
    CDmach table leaves zero (0.79), because otherwise the two drag models
    disagree grossly at the comparison point: a conventional kappa of 0.87 with
    737-class geometry puts M_crit at 0.636 and adds 0.0086 of wave drag where
    JSBSim has exactly zero -- a 32% error on a CD0 of 0.027.

    NOTE the resulting kappa exceeds 1.0, which is not a physical airfoil
    technology factor (0.87 conventional, ~0.95 supercritical). It is a fitting
    parameter forcing one drag model onto another's onset, and is reported as
    such. Wave drag is UNTESTED by this comparison: at M 0.78 both engines
    produce exactly zero.
    """
    sweep = 25.0 * math.pi / 180.0  # 737-class quarter-chord sweep
    t_over_c = 0.12
    cos_s = math.cos(sweep)
    mdd = onset + (0.1 / 80.0) ** (1.0 / 3.0)
    kappa = (mdd + t_over_c / cos_s**2 + CL_trim / (10.0 * cos_s**3)) * cos_s
    return sweep, t_over_c, kappa


def aircraft_entry(fdm, d, at_trim, trim, rho, h_match):
    """Every literal _boeing_737() needs, with the intercepts solved."""
    S = fdm["metrics/Sw-sqft"] * FT2
    b = fdm["metrics/bw-ft"] * FT2M
    c = fdm["metrics/cbarw-ft"] * FT2M
    AR = b * b / S

    # JSBSim's induced drag is CDi = 0.043 CL^2. Choosing e so that
    # CL^2/(pi e AR) equals that makes the term identical, not merely close.
    e = 1.0 / (0.043 * math.pi * AR)

    a_t, de_t = trim["alpha"], trim["elevator"]
    # Intercepts are solved so the model reproduces JSBSim's coefficients AT the
    # reference point exactly; that is what "linearised about cruise" means.
    CL0 = at_trim["CL"] - d["CLa"] * a_t - d["CLde"] * de_t
    Cm0 = at_trim["Cm"] - d["Cma"] * a_t - d["Cmde"] * de_t
    # Wave drag is zero at M 0.78 by construction, so CD0 absorbs everything
    # that is not induced -- including JSBSim's CDde term (0.059 |de|), which
    # has no home in flightsim and is therefore frozen at its trim value.
    CD0 = at_trim["CD"] - at_trim["CL"] ** 2 / (math.pi * e * AR)

    sweep, t_over_c, kappa = wave_drag_parameters(at_trim["CL"])
    fmax, lapse, ram, mach_res, alt_res = thrust_fit(trim["throttle"])
    return dict(
        mass=fdm["inertia/weight-lbs"] * LBF2N / 9.80665,
        S=S, b=b, c=c, AR=AR, e=e, CD0=CD0, CL0=CL0, Cm0=Cm0,
        sweep=sweep, t_over_c=t_over_c, kappa_airfoil=kappa,
        max_thrust=fmax, thrust_lapse=lapse, mach_ram=ram,
        thrust_mach_residual=mach_res, thrust_altitude_residual=alt_res,
        elevator_limit=ELEVATOR_RANGE, aileron_limit=AILERON_RANGE,
        rudder_limit=RUDDER_RANGE, matched_altitude=h_match,
        airspeed=fdm["velocities/vt-fps"] * FT2M,
        **{k: v for k, v in d.items() if k not in ABSENT and k != "CDa_engine"},
    )


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------
def f(x):
    return f"{float(x):.12e}"


def vec(a):
    return " ".join(f(v) for v in np.asarray(a).ravel())


def main():
    work = OUT.parent
    work.mkdir(parents=True, exist_ok=True)

    base = jsbsim.FGJSBBase()
    version = base.get_version()

    # --- trim, both modes ---
    lon = trimmed(0)
    trim_alpha = lon["aero/alpha-rad"]
    trim_de = lon["fcs/elevator-pos-rad"]
    trims = {
        "longitudinal": dict(
            alpha=trim_alpha, elevator=trim_de,
            throttle=lon["fcs/throttle-cmd-norm[0]"],
            thrust=(lon["propulsion/engine[0]/thrust-lbs"]
                    + lon["propulsion/engine[1]/thrust-lbs"]) * LBF2N,
            bank=0.0,
        ),
    }
    turn = trimmed(5, alt_ft=TURN_ALT_FT, vt_fps=TURN_VT_FPS, bank_deg=TURN_BANK_DEG)
    trims["turn"] = dict(
        alpha=turn["aero/alpha-rad"], elevator=turn["fcs/elevator-pos-rad"],
        throttle=turn["fcs/throttle-cmd-norm[0]"],
        thrust=(turn["propulsion/engine[0]/thrust-lbs"]
                + turn["propulsion/engine[1]/thrust-lbs"]) * LBF2N,
        bank=turn["attitude/phi-rad"],
    )

    # --- condition, with the density-matched altitude ---
    rho = lon["atmosphere/rho-slugs_ft3"] * SLUG_FT3_TO_KG_M3
    a_snd = lon["atmosphere/a-fps"] * FT2M
    h_js = ALT_FT * FT2M
    h_match = brentq(lambda h: float(density(h)) - rho, h_js - 300.0, h_js + 300.0,
                     xtol=1e-12)
    residual = abs(float(density(h_match)) - rho) / rho
    if residual > 1e-10:
        raise SystemExit(f"density match failed: residual {residual:.3e}")
    print(f"density match residual: {residual:.3e}  (altitude shift "
          f"{(h_match - h_js) / FT2M:+.2f} ft)")

    inertia = np.array([
        [lon["inertia/ixx-slugs_ft2"], 0.0, -lon["inertia/ixz-slugs_ft2"]],
        [0.0, lon["inertia/iyy-slugs_ft2"], 0.0],
        [-lon["inertia/ixz-slugs_ft2"], 0.0, lon["inertia/izz-slugs_ft2"]],
    ]) * SLUG_FT2_TO_KG_M2

    # --- derivatives ---
    d = recover(math.degrees(trim_alpha), trim_de)
    print("\nrecovered derivatives:")
    for k in sorted(d):
        print(f"  {k:12s} {d[k]:+.6f}")

    # The floor is not zero, and the reason is derivable rather than empirical.
    # run() has to integrate SETTLE_DT for the FCS to gear the commands onto the
    # surfaces, and that drifts alpha. For a rate derivative the artifact is
    #
    #   CLa * d(alpha) / d(rate_hat) = CLa * (2 s dt) / (2 s i2vel)
    #                                = CLa * dt / i2vel
    #
    # independent of the perturbation size s. At SETTLE_DT = 1e-6 that predicts
    # 5.48e-4 for CLq, against 5.48e-4 measured -- the entire residual is the
    # artifact. Anything five times larger is a real derivative, not drift.
    at_trim = read_state(at_state(alpha_deg=math.degrees(trim_alpha), de=trim_de))
    drift = abs(d["CLa"]) * SETTLE_DT / at_trim["ci2vel"]
    floor = 5.0 * drift
    absent = {k: d[k] for k in ABSENT}
    bad = {k: v for k, v in absent.items() if abs(v) > floor}
    if bad:
        raise SystemExit(
            f"derivatives 737.xml does not define measured above the {floor:.2e} "
            f"settling-drift floor: {bad}\n"
            "flightsim's entry carries 0.0 for these; the comparison would be "
            "wrong. Either JSBSim's model changed or the recovery is picking up "
            "cross-coupling."
        )
    print(f"\nall {len(ABSENT)} absent derivatives below the derived "
          f"settling-drift floor {floor:.2e} "
          f"(predicted artifact {drift:.2e}, worst measured "
          f"{max(abs(v) for v in absent.values()):.2e})")


    # --- the flightsim entry ---
    entry = aircraft_entry(lon, d, at_trim, trims["longitudinal"], rho, h_match)
    print("\nflightsim aircraft entry:")
    for k in sorted(entry):
        print(f"  {k:26s} {entry[k]:+.9g}")
    print(f"\nthrust fit residuals: {entry['thrust_mach_residual']*100:.2f}% over "
          f"Mach, {entry['thrust_altitude_residual']*100:.2f}% over altitude")

    # --- sweep ---
    sweep = []
    for a_deg in (0.0, 1.0, 2.0, 3.0, 4.0):
        sweep.append(read_state(at_state(alpha_deg=a_deg, de=trim_de)))
    for b_deg in (-3.0, -1.5, 1.5, 3.0):
        sweep.append(read_state(at_state(alpha_deg=math.degrees(trim_alpha),
                                         beta_deg=b_deg, de=trim_de)))
    for p in (-0.10, -0.05, 0.05, 0.10):
        sweep.append(read_state(at_state(alpha_deg=math.degrees(trim_alpha),
                                         p=p, de=trim_de)))
    for q in (-0.04, -0.02, 0.02, 0.04):
        sweep.append(read_state(at_state(alpha_deg=math.degrees(trim_alpha),
                                         q=q, de=trim_de)))
    for r in (-0.04, -0.02, 0.02, 0.04):
        sweep.append(read_state(at_state(alpha_deg=math.degrees(trim_alpha),
                                         r=r, de=trim_de)))
    for de in (trim_de - 0.06, trim_de - 0.03, trim_de + 0.03, trim_de + 0.06):
        sweep.append(read_state(at_state(alpha_deg=math.degrees(trim_alpha), de=de)))
    for da in (-0.10, -0.05, 0.05, 0.10):
        sweep.append(read_state(at_state(alpha_deg=math.degrees(trim_alpha),
                                         da=da, de=trim_de)))
    for dr in (-0.10, -0.05, 0.05, 0.10):
        sweep.append(read_state(at_state(alpha_deg=math.degrees(trim_alpha),
                                         dr=dr, de=trim_de)))
    print(f"sweep: {len(sweep)} points")

    # --- linearisation ---
    A, B, x0, u0 = linearization(work)
    print("linearisation state ordering re-derived and confirmed")

    # --- trajectories ---
    trajectories = {}
    for case in ("elevator_doublet", "rudder_kick"):
        samples, miss = fly(case)
        trajectories[case] = samples
        print(f"{case}: {len(samples)} samples, worst rudder miss {miss:.2e} rad")
    coriolis = coriolis_contribution("rudder_kick")
    print(f"coriolis contribution (lat 0 vs 47): {coriolis:.4f} m/s")

    # --- write ---
    L = []
    L.append('<?xml version="1.0" encoding="utf-8"?>')
    L.append("<!-- GENERATED by scripts/gen_jsbsim_reference.py. Do not edit. -->")
    L.append("<jsbsim_reference>")
    L.append("  <provenance>")
    L.append(f"    <jsbsim_version>{version}</jsbsim_version>")
    L.append(f"    <jsbsim_commit>{version}</jsbsim_commit>")
    L.append("    <aircraft>737</aircraft>")
    L.append("    <generator>scripts/gen_jsbsim_reference.py</generator>")
    L.append("  </provenance>")

    L.append('  <condition name="cruise">')
    L.append(f"    <altitude_m>{f(h_js)}</altitude_m>")
    L.append(f"    <matched_altitude_m>{f(h_match)}</matched_altitude_m>")
    L.append(f"    <density>{f(rho)}</density>")
    L.append(f"    <sound_speed>{f(a_snd)}</sound_speed>")
    L.append(f"    <airspeed>{f(lon['velocities/vt-fps'] * FT2M)}</airspeed>")
    L.append(f"    <mass>{f(lon['inertia/weight-lbs'] * LBF2N / 9.80665)}</mass>")
    L.append(f"    <inertia>{vec(inertia)}</inertia>")
    L.append(f"    <density_match_residual>{f(residual)}</density_match_residual>")
    L.append("  </condition>")

    for name in sorted(trims):
        t = trims[name]
        L.append(f'  <trim mode="{name}">')
        for k in ("alpha", "elevator", "throttle", "thrust", "bank"):
            L.append(f"    <{k}>{f(t[k])}</{k}>")
        L.append("  </trim>")

    L.append("  <derivatives>")
    for k in sorted(d):
        if k not in ABSENT:
            L.append(f'    <derivative name="{k}" value="{f(d[k])}"/>')
    for k in ("CL_trim", "CD_trim", "Cm_trim"):
        L.append(f'    <derivative name="{k}" value="{f(at_trim[k.split("_")[0]])}"/>')
    L.append("  </derivatives>")

    L.append("  <absent_derivatives>")
    for k in ABSENT:
        L.append(f'    <derivative name="{k}" measured="{f(absent[k])}"/>')
    L.append("  </absent_derivatives>")

    L.append("  <aircraft_entry>")
    for k in sorted(entry):
        L.append(f'    <value name="{k}">{f(entry[k])}</value>')
    L.append("  </aircraft_entry>")

    L.append("  <sweep>")
    for s in sweep:
        L.append(
            f'    <point alpha="{f(s["alpha"])}" beta="{f(s["beta"])}" '
            f'vel_body="{vec(s["vel_body"])}" sound_speed="{f(s["a_sound"])}" '
            f'rates="{vec([s["p"], s["q"], s["r"]])}" '
            f'controls="{vec([s["de"], s["da"], s["dr"]])}" '
            f'coefficients="{vec([s["CL"], s["CD"], s["CY"], s["Cl"], s["Cm"], s["Cn"]])}"/>'
        )
    L.append("  </sweep>")

    L.append("  <linearization>")
    L.append(f"    <A>{vec(A)}</A>")
    L.append(f"    <B>{vec(B)}</B>")
    L.append(f"    <x0>{vec(x0)}</x0>")
    L.append(f"    <u0>{vec(u0)}</u0>")
    L.append("  </linearization>")

    for name in sorted(trajectories):
        L.append(f'  <trajectory name="{name}">')
        for s in trajectories[name]:
            L.append(
                f'    <sample t="{f(s["t"])}" vel_body="{vec(s["vel_body"])}" '
                f'omega="{vec(s["omega"])}" euler="{vec(s["euler"])}" '
                f'altitude="{f(s["altitude"])}" controls="{vec(s["controls"])}" '
                f'thrust="{f(s["thrust"])}"/>'
            )
        L.append("  </trajectory>")

    L.append("  <diagnostics>")
    L.append(f'    <coriolis_velocity_m_s>{f(coriolis)}</coriolis_velocity_m_s>')
    L.append("  </diagnostics>")
    L.append("</jsbsim_reference>")

    OUT.write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    # JSBSim writes its linearisation next to the data file; it is an
    # intermediate, already parsed into <linearization>, and must not be checked in.
    (OUT.parent / "737_lin.sce").unlink(missing_ok=True)
    print(f"\nwrote {OUT.relative_to(ROOT)}  ({len(L)} lines)")


if __name__ == "__main__":
    main()
