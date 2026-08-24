"""Aircraft definitions.

Everything stored here is SI, per-radian, and body-axis. Conversions from the
source documents happen at the point of definition using flightsim.units, each
with a comment naming the table it came from.

`Aircraft` is a NamedTuple so it is a JAX pytree: it can be passed straight
through jit and vmap without static_argnums. The name is deliberately NOT a
field -- a string leaf would break tracing -- so aircraft are looked up in the
REGISTRY dict at the bottom of this module.
"""

import math
from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from flightsim.atmosphere import G0
from flightsim.units import (
    DEG2RAD, FT2M, HP2W, KT2MS, LB2KG, LBF2N, SLUG_FT2_TO_KG_M2,
)


class Aircraft(NamedTuple):
    # Mass and geometry
    mass: Array  # kg
    inertia: Array  # (3,3) kg.m^2, body axes, including Ixz
    inertia_inv: Array  # (3,3), precomputed
    S: Array  # m^2, reference wing area
    b: Array  # m, span
    c: Array  # m, mean aerodynamic chord

    # Drag: CD = CD0 + CL^2/(pi e AR) + 20 (M - M_crit)^4
    # M_crit comes from the Korn equation, so the wave-drag rise is a function
    # of CL as well as Mach.
    CD0: Array
    e: Array  # Oswald efficiency
    AR: Array  # b^2 / S
    sweep: Array  # rad, quarter-chord sweep
    t_over_c: Array  # thickness ratio
    kappa_airfoil: Array  # Korn technology factor: ~0.87 conventional

    # Longitudinal
    CL0: Array
    CLa: Array
    CLq: Array
    CLde: Array
    Cm0: Array
    Cma: Array
    Cmq: Array
    Cmde: Array

    # Lateral-directional
    CYb: Array
    CYp: Array
    CYr: Array
    CYdr: Array
    Clb: Array
    Clp: Array
    Clr: Array
    Clda: Array
    Cldr: Array
    Cnb: Array
    Cnp: Array
    Cnr: Array
    Cnda: Array
    Cndr: Array

    # Propulsion:
    #   thrust = throttle * max_thrust * (rho/rho0)^thrust_lapse * (1 + mach_ram M^2)
    max_thrust: Array  # N, sea-level static
    thrust_lapse: Array  # density-ratio exponent

    # Control deflection limits (rad), used by trim bounds and the autopilot
    elevator_limit: Array
    aileron_limit: Array
    rudder_limit: Array

    # Compressor ram recovery: the thrust a turbofan gains from forward speed.
    #
    # Defaulted, and the default is neutral, so every aircraft defined before
    # this field existed is bit-for-bit unchanged by it -- asserted in
    # test_aero.test_mach_ram_defaults_to_neutral rather than trusted. Only the
    # 737 sets it, because only the 737 has a reference engine to fit against:
    # JSBSim's CFM56 gains +12% between M 0 and M 0.8 at 30,000 ft, which is
    # large enough to read as a drag error in a trajectory comparison.
    #
    # It is a fit over roughly M 0.6-1.0 and does NOT capture the shallow dip
    # the real table has around M 0.2. See the 737's docstring.
    mach_ram: Array = jnp.array(0.0)

    # Sideslip drag, CD += CD_beta * beta^2. Quadratic because drag is an even
    # function of sideslip, so the linear term is identically zero at beta = 0.
    # Linear small-perturbation theory therefore has no such derivative, which
    # is why the sourced aircraft here carry none and the default is neutral.
    # See docs/superpowers/specs/2026-08-20-model-fidelity-improvements-design.md
    # section 1 -- including the fact that no reference in refs/ supplies a
    # coefficient form, so this one rests on symmetry rather than on a citation.
    CD_beta: Array = jnp.array(0.0)

    # Profile-drag rise with incidence: CD += CD_alpha * alpha. LINEAR, and the
    # asymmetry with CD_beta above is deliberate rather than an oversight.
    #
    # Drag is even about its minimum in BOTH variables. For sideslip the
    # reference condition sits AT that minimum -- beta = 0 -- so a linear term
    # would be a kink through the operating point, and the quadratic is the only
    # defensible form. For incidence the reference sits well away from it, at
    # alpha = 1.97 deg cruise and 3.63 deg approach, so the first-order Taylor
    # term is exactly the right object for a model that is explicitly a
    # linearisation about that point.
    #
    # It is therefore WRONG at negative alpha, where the true rise turns back
    # up and this term keeps falling. That is the same cruise-local caveat the
    # rest of the entry carries, not a new one.
    #
    # Without it the drag slope is the induced term alone -- measured, 0.1267
    # against JSBSim's 0.2113 at cruise -- and with moments referred to the
    # AERORP that error reaches the pitching moment through r x F.
    CD_alpha: Array = jnp.array(0.0)

    # Angle-of-attack-rate derivatives, referred to alphadot_hat = alphadot*c/2V
    # exactly as CLq and Cmq are referred to q_hat.
    #
    # Stengel, Flight Dynamics 2nd ed., Eq. (3.4-25) and (3.4-26): the flow over
    # an aft tail is modified by the wing's downwash, and a change in wing lift
    # convects downstream reaching the tail after l_ht/V, so the tail sees an
    # increment proportional to alphadot.
    #
    # ONLY THE WIND-INDUCED PART OF alphadot IS APPLIED. The aircraft's own
    # contribution is implicit -- alphadot depends on wdot depends on the forces
    # depend on alphadot -- and remains folded into Cmq, which is exact whenever
    # alphadot = q. The wind part is explicit, and it is the part the aircraft's
    # own motion cannot produce: Stengel p.227, "a plunging aircraft experiences
    # non-zero alphadot with zero q". That is the gust case, and it is where
    # this project's turbulence and microburst work lives.
    CLadot: Array = jnp.array(0.0)
    Cmadot: Array = jnp.array(0.0)

    # Body-axis vector from the CG to the point the moment coefficients are
    # referred to, in metres. Zero means "referred to the CG", which is what
    # every derivative set from CR-2144 and Nelson already is, so the default
    # leaves them untouched.
    #
    # Stengel, Flight Dynamics 2nd ed. Eq. (2.4-68): Cm = Cm_c + ((x_cp -
    # x_cm)/cbar) C_N, with the rationale at pp.109-111 -- "the center of mass
    # varies with aircraft loading... we choose a point fixed in the airframe".
    # The coefficients then describe the AIRFRAME rather than one loading, and
    # the CG-dependence falls out of r x F instead of being baked into Cma.
    #
    # Note Eq. (2.4-68) is written in C_N, the BODY-axis normal force. That is
    # exactly why folding the offset into a constant Cma cannot work: the
    # wind-axis lift and drag rotate into body axes as alpha changes, so the
    # transferred moment is nonlinear in alpha even when every coefficient is
    # constant. Measured on the 737, sweeping alpha 0-8 deg, worst |error| in Cm:
    # this formulation 2.8e-17, the constant-Cma fold 1.6e-2.
    aero_ref: Array = jnp.zeros(3)


def inertia_tensor(Ixx, Iyy, Izz, Ixz) -> Array:
    """Body-axis inertia tensor.

    Ixz is the product of inertia in its usual positive-forward-up sense, so it
    enters the tensor negated. It is not negligible for the 747.
    """
    return jnp.array(
        [
            [Ixx, 0.0, -Ixz],
            [0.0, Iyy, 0.0],
            [-Ixz, 0.0, Izz],
        ]
    )


def stability_to_body(Cl_s, Cn_s, alpha_ref):
    """Rotate rolling/yawing moment derivatives from stability to body axes.

    Stability axes are body axes rotated about y by the reference angle of
    attack, so the x and z components mix.
    """
    ca, sa = jnp.cos(alpha_ref), jnp.sin(alpha_ref)
    return Cl_s * ca - Cn_s * sa, Cn_s * ca + Cl_s * sa


class FlightCondition(NamedTuple):
    """The condition a set of DIMENSIONAL derivatives was linearised at.

    This type exists because of a bug that actually happened. The Cessna's
    dimensional control derivatives belong to its source's 67 m/s linearisation,
    but they were first recovered using the dynamic pressure at the 60 m/s cruise
    this package chose -- inflating all four by 25%. A non-dimensional
    coefficient is a property of the airframe, so it must be recovered at the
    condition its dimensional form was quoted at, and nothing in the arithmetic
    complains if it is not.

    Bundling the condition with the conversion makes the mismatch impossible to
    express: there is no argument left to get wrong.
    """

    airspeed: float  # m/s, the U0 the source linearised at
    density: float  # kg/m^3, the rho that goes with it
    mass: float  # kg
    Ixx: float  # kg.m^2
    Iyy: float
    Izz: float
    S: float  # m^2
    b: float  # m
    c: float  # m

    @property
    def qS(self) -> float:
        return 0.5 * self.density * self.airspeed**2 * self.S


def from_dimensional_longitudinal(fc: FlightCondition, *, Zw, Zq, Mw, Mq, Zde, Mde, CD):
    """(CLa, CLq, CLde, Cma, Cmq, Cmde) from body-axis dimensional derivatives.

    The relations are the ones in CR-2144 Appendix A, and they are shared rather
    than rewritten per aircraft: they were transcribed by hand three times before
    this helper existed. `CD` is the trim drag coefficient, which enters CLa only.
    """
    return (
        -Zw * fc.mass * fc.airspeed / fc.qS - CD,  # CLa
        -Zq * 2.0 * fc.airspeed * fc.mass / (fc.qS * fc.c),  # CLq
        -Zde * fc.mass / fc.qS,  # CLde
        Mw * fc.airspeed * fc.Iyy / (fc.qS * fc.c),  # Cma
        Mq * 2.0 * fc.airspeed * fc.Iyy / (fc.qS * fc.c * fc.c),  # Cmq
        Mde * fc.Iyy / (fc.qS * fc.c),  # Cmde
    )


def from_dimensional_lateral(fc: FlightCondition, *, Yv, Lv, Nv, Lp, Np, Lr, Nr):
    """(CYb, Clb, Cnb, Clp, Cnp, Clr, Cnr) from body-axis dimensional derivatives.

    Assumes the derivatives are UNPRIMED. CR-2144 tabulates primed values, which
    fold the Ixz cross-coupling in; run those through `_unprime` first or this
    double-counts it.
    """
    span = fc.qS * fc.b
    rate = fc.qS * fc.b * fc.b / (2.0 * fc.airspeed)
    return (
        Yv * fc.airspeed * fc.mass / fc.qS,  # CYb
        Lv * fc.airspeed * fc.Ixx / span,  # Clb
        Nv * fc.airspeed * fc.Izz / span,  # Cnb
        Lp * fc.Ixx / rate,  # Clp
        Np * fc.Izz / rate,  # Cnp
        Lr * fc.Ixx / rate,  # Clr
        Nr * fc.Izz / rate,  # Cnr
    )


def from_dimensional_controls(fc: FlightCondition, *, Ydr=0.0, Lda=0.0, Nda=0.0,
                              Ldr=0.0, Ndr=0.0):
    """(CYdr, Clda, Cnda, Cldr, Cndr). Omitted derivatives stay zero, not guessed."""
    span = fc.qS * fc.b
    return (
        Ydr * fc.mass / fc.qS,  # CYdr
        Lda * fc.Ixx / span,  # Clda
        Nda * fc.Izz / span,  # Cnda
        Ldr * fc.Ixx / span,  # Cldr
        Ndr * fc.Izz / span,  # Cndr
    )


def _unprime(Lp, Np, Ix, Iz, Ixz):
    """Recover raw rolling/yawing derivatives from Ixz-corrected primed ones.

    CR-2144 tabulates primed lateral derivatives, which already fold the inertia
    cross-coupling in:

        L' = (L + (Ixz/Ix) N) / (1 - Ixz^2/(Ix Iz))
        N' = (N + (Ixz/Iz) L) / (1 - Ixz^2/(Ix Iz))

    This module's dynamics carries the full inertia tensor and applies that
    coupling itself, so the raw values are what is wanted. Using the primed
    numbers directly would double-count Ixz.
    """
    det = 1.0 - Ixz**2 / (Ix * Iz)
    a, b_ = Ixz / Ix, Ixz / Iz
    return jnp.linalg.solve(
        jnp.array([[1.0, a], [b_, 1.0]]), jnp.array([Lp * det, Np * det])
    )


# ---------------------------------------------------------------------------
# Boeing 747-100
#
# Source: NASA CR-2144, Heffley & Jewell, "Aircraft Handling Qualities Data",
# December 1972, Section IX. Flight condition 9 of Table IX-3: 40,000 ft,
# M = 0.80, the nominal cruise point.
#
# CR-2144 does NOT tabulate non-dimensional derivatives for the 747 in cruise --
# only for the landing and power-approach configurations (Tables IX-1, IX-2, both
# sea level). Cruise appears as dimensional derivatives, so the set below is
# recovered from those. Tables IX-4 and IX-8 both state BODY AXIS SYSTEM, so no
# stability-to-body rotation is applied. The recovered values agree with the
# report's own landing-configuration table, Table IX-1 on printed p.216 -- read
# again at 500 dpi during the remediation pass, because this comment previously
# cited CLde as .396 and the table reads .356:
#
#   CLde  +0.3638 recovered vs  .356 tabulated   +2.2%
#   Cmde  -1.4442             vs -1.40           -3.2%
#   Clb   -0.2944             vs -.281           -4.8%
#   Cnb   +0.1715             vs  .184           -6.8%
#
# That is the check that the conversion chain is right. It is a cross-CONDITION
# check, not a repeat measurement -- landing configuration at sea level against
# cruise at 40,000 ft -- so a few per cent is what agreement looks like here,
# and the comparison bounds the chain rather than validating the numbers.
# ---------------------------------------------------------------------------

# ft/s^2, used only to turn the tabulated weight into slugs.
#
# A DECLARED EXCEPTION to units.py's "never inline a conversion factor anywhere
# else", kept deliberately after being measured. It truncates
# g0 = 32.17404855643044 ft/s^2, so this slug mass sits 1.5e-6 relative away
# from the kilogram mass the same function ships as `W * LB2KG`, and that slip
# reaches CLa, CLq, CLde, CD0 and e for this one aircraft.
#
# The remediation pass made the correct change -- `m = W * LB2KG / SLUG2KG`,
# pure units.py constants, no inlined number -- measured it, and REVERTED IT.
# What it moved: 19 quantities on the 747 and nothing on any other aircraft, all
# at <= 4.1e-6 relative; every number PROJECT.md section 4 quotes was unchanged
# at its quoted precision, as were the ledger's 5.9450, -23.9232 and 4.0241.
# What it broke: `test_extracting_rk4_step_did_not_move_a_single_bit` and
# `test_logging_the_run_did_not_move_the_headline_numbers`, two BIT-EXACT
# arithmetic-neutrality guards on earlier refactors, whose entire value is that
# they admit no tolerance at all.
#
# Re-pinning those two to new values would have been changing reference data to
# let a change through, which this project does not do -- and it would have
# spent the guarantee those tests exist to provide on a violation that moves no
# result. The flaw is smaller than the fix. See docs/ASSUMPTIONS.md B5.
_B747_G = 32.174


def _boeing_747() -> Aircraft:
    # -- Table IX-3: geometry and flight condition (imperial, verbatim) --
    S, b, c = 5500.0, 195.68, 27.31  # ft^2, ft, ft
    W = 636636.0  # lb
    Ix, Iy, Iz, Ixz = 1.82e7, 3.31e7, 4.97e7, 970056.0  # slug-ft^2, body axis
    U0, qbar = 774.0, 177.0  # ft/s, lb/ft^2
    alpha0 = 4.60 * DEG2RAD
    mach0 = 0.800

    # -- Figure IX-6, CD vs Mach, 40,000 ft curve read at M = 0.80 --
    # Log axis, point 9 sits just below the 0.05 gridline. Reading uncertainty
    # is about +-0.003, which propagates directly into CD0.
    CD_trim = 0.043

    # -- Table IX-4: longitudinal dimensional derivatives, body axis --
    # Mq is -0.339, not the -0.330 this line carried until it was caught by a
    # line-by-line re-read against Table IX-4. The transcription slip cost 1.1
    # points of short-period damping match (12.6% -> 11.5% against the reference).
    Xw, Zw, Zq, Mw, Mq, Zde, Mde = 0.0389, -0.317, -5.16, -0.00105, -0.339, -17.9, -1.16
    # (Xu, Zu, Mu, Zwd, Mwd, Xde also tabulated; the speed and alpha-dot
    #  derivatives are outside this model's form, which is alpha/q/de only.)

    # -- Table IX-8: lateral dimensional derivatives, body axis, primed --
    Yb, Ydr_star = -43.2, 0.00729
    lateral_primed = {
        "beta": (-3.05, 0.598),
        "p": (-0.465, -0.0316),
        "r": (0.388, -0.115),
        "da": (0.143, 0.00775),
        "dr": (0.153, -0.475),
    }

    m = W / _B747_G  # slugs
    qS = qbar * S
    AR = b * b / S
    CL_trim = W / qS  # 0.654

    # Longitudinal, from the dimensional definitions in CR-2144 Appendix A.
    CLa = -Zw * m * U0 / qS - CD_trim
    CLq = -Zq * 2.0 * m * U0 / (qS * c)
    CLde = -Zde * m / qS
    Cma = Mw * Iy * U0 / (qS * c)
    Cmq = Mq * 2.0 * Iy * U0 / (qS * c * c)
    Cmde = Mde * Iy / (qS * c)

    # The linear model is referenced to the trimmed condition: CL = CL_trim and
    # Cm = 0 at alpha0 with zero elevator (the stabiliser carries the trim).
    CL0 = CL_trim - CLa * alpha0
    Cm0 = -Cma * alpha0

    # Drag. Oswald efficiency is not tabulated, so it is recovered from Xw:
    #   Xw = qS(-CDa cos a + CD sin a + CLa sin a + CL cos a) / (m U0)
    # which gives dCD/dalpha = 0.441 at this condition. Subtracting the Korn
    # wave-drag contribution leaves the induced part, hence e. CD0 is then what
    # is left of CD_trim after induced and wave drag.
    ca, sa = math.cos(alpha0), math.sin(alpha0)
    CDa = (-Xw * m * U0 / qS + CD_trim * sa + CLa * sa + CL_trim * ca) / ca
    sweep = 37.5 * DEG2RAD  # quarter-chord
    t_over_c = 0.09  # mean; 747 wing runs ~13% root to ~8% tip
    kappa = 0.87  # conventional (non-supercritical) 1960s section
    cos_s = math.cos(sweep)
    m_dd = kappa / cos_s - t_over_c / cos_s**2 - CL_trim / (10.0 * cos_s**3)
    m_crit = m_dd - (0.1 / 80.0) ** (1.0 / 3.0)
    cd_wave = 20.0 * max(mach0 - m_crit, 0.0) ** 4
    dcd_wave_dalpha = 80.0 * max(mach0 - m_crit, 0.0) ** 3 / (10.0 * cos_s**3) * CLa
    e = 2.0 * CL_trim * CLa / (math.pi * AR * (CDa - dcd_wave_dalpha))
    CD0 = CD_trim - CL_trim**2 / (math.pi * e * AR) - cd_wave

    # Lateral. Y* derivatives are normalised by U0 (Yv = Yb/U0 in the table).
    CYb = Yb * m / qS
    CYdr = Ydr_star * U0 * m / qS
    nd = {}
    for key, (Lp, Np) in lateral_primed.items():
        L, N = _unprime(Lp, Np, Ix, Iz, Ixz)
        Cl, Cn = float(L) * Ix / (qS * b), float(N) * Iz / (qS * b)
        if key in ("p", "r"):  # rate derivatives carry the b/2U0 non-dimensionalisation
            Cl, Cn = Cl * 2.0 * U0 / b, Cn * 2.0 * U0 / b
        nd[key] = (Cl, Cn)

    inertia = inertia_tensor(
        *(v * SLUG_FT2_TO_KG_M2 for v in (Ix, Iy, Iz, Ixz))
    )
    return Aircraft(
        mass=jnp.array(W * LB2KG),
        inertia=inertia,
        inertia_inv=jnp.linalg.inv(inertia),
        S=jnp.array(S * FT2M**2),
        b=jnp.array(b * FT2M),
        c=jnp.array(c * FT2M),
        CD0=jnp.array(CD0),
        e=jnp.array(e),
        AR=jnp.array(AR),
        sweep=jnp.array(sweep),
        t_over_c=jnp.array(t_over_c),
        kappa_airfoil=jnp.array(kappa),
        CL0=jnp.array(CL0),
        CLa=jnp.array(CLa),
        CLq=jnp.array(CLq),
        CLde=jnp.array(CLde),
        Cm0=jnp.array(Cm0),
        Cma=jnp.array(Cma),
        Cmq=jnp.array(Cmq),
        Cmde=jnp.array(Cmde),
        CYb=jnp.array(CYb),
        # CR-2144 does not tabulate CYp or CYr for the 747 in any configuration.
        # Zero matches the source rather than inventing a value.
        CYp=jnp.array(0.0),
        CYr=jnp.array(0.0),
        CYdr=jnp.array(CYdr),
        Clb=jnp.array(nd["beta"][0]),
        Clp=jnp.array(nd["p"][0]),
        Clr=jnp.array(nd["r"][0]),
        Clda=jnp.array(nd["da"][0]),
        Cldr=jnp.array(nd["dr"][0]),
        Cnb=jnp.array(nd["beta"][1]),
        Cnp=jnp.array(nd["p"][1]),
        Cnr=jnp.array(nd["r"][1]),
        Cnda=jnp.array(nd["da"][1]),
        Cndr=jnp.array(nd["dr"][1]),
        # 4 x Pratt & Whitney JT9D-3A at 43,500 lbf sea-level static.
        # Lapse exponent 0.8 is the usual high-bypass turbofan value; CR-2144
        # models no engine, so this is a modelling choice, not source data.
        max_thrust=jnp.array(4 * 43500.0 * LBF2N),
        thrust_lapse=jnp.array(0.8),
        elevator_limit=jnp.array(25.0 * DEG2RAD),
        aileron_limit=jnp.array(20.0 * DEG2RAD),
        rudder_limit=jnp.array(25.0 * DEG2RAD),
    )


# ---------------------------------------------------------------------------
# Boeing 747-100, POWER APPROACH configuration
#
# Source: the same NASA CR-2144 Section IX. Table IX-2 "Power Approach
# Configuration Non-Dimensional Derivatives", h = sea level, VTo = 165 KTAS,
# alpha0 = 5.7 deg, stabiliser -2.1 deg; mass and inertia from Figure IX-1's
# Power Approach block: max landing weight, 20 deg flaps, GEAR UP, 1.4 Vs.
#
# This exists because the cruise set could not be used below 500 ft. The
# microburst work (section 4) needs an airliner at approach speed near the
# ground, and flying Mach 0.8 / 40,000 ft derivatives there would have been the
# largest extrapolation in the project -- and an invisible one, because the
# numbers would still have looked reasonable. Section 5 recorded that as a gap;
# this closes it.
#
# It is a far simpler transcription than the cruise set. Table IX-2 is ALREADY
# NON-DIMENSIONAL, so there is no dimensional conversion chain, no primed-to-
# unprimed lateral algebra, and no re-derivation of anything. The numbers below
# are the table, verbatim.
#
# The check that mass and aerodynamics belong to the same condition: the table
# states CL = 1.11, and W/qS at 165 KTAS sea level with Figure IX-1's 564,000 lb
# gives 1.1126. Those are independent entries on different pages, so agreeing to
# 0.2% says they were transcribed from the same flight condition. Asserted in
# tests/test_aircraft.py.
# ---------------------------------------------------------------------------


def _boeing_747_approach() -> Aircraft:
    # -- Table IX-3 geometry: the same airframe as the cruise set --
    S, b, c = 5500.0, 195.68, 27.31  # ft^2, ft, ft

    # -- Table IX-3, flight condition 2 (the power-approach column) --
    #
    # SOURCE CONFLICT, and IX-3 wins. Figure IX-1's Power Approach block gives
    # W = 564,000 lb with Ix/Iy/Iz/Ixz = 13.7/30.5/43.1/0.825 x 10^6 slug-ft^2.
    # Table IX-3 column 2 gives 564,032 lb with 14.2/32.3/45.4/0.870 x 10^6 --
    # the same weight to rounding, but inertias up to 6% larger. Table IX-3 is
    # taken because it is the table the derivatives were COMPUTED at: its
    # Q = 92.2 psf, VTO = 165 KTAS and ALPHA = 5.70 deg all match Table IX-2's
    # header exactly, and the project's cruise set already reads flight
    # condition 9 from this same table. For CRUISE the two agree (both 18.2e6,
    # 970056), so the disagreement is specific to the approach configuration.
    W = 564032.0  # lb
    Ix, Iy, Iz, Ixz = 0.142e8, 0.323e8, 0.454e8, 870050.0  # slug-ft^2, body axis

    # -- Table IX-2 header --
    V0 = 165.0 * KT2MS / FT2M  # ft/s, from 165 KTAS
    alpha0 = 5.7 * DEG2RAD

    # -- Table IX-2, longitudinal. Verbatim, per radian. --
    CL_trim, CD_trim = 1.11, 0.102
    CLa, CDa, Cma = 5.70, 0.66, -1.26
    CLq, Cmq = 5.4, -20.8
    CLde, Cmde = 0.338, -1.34
    # (CL_alphadot -6.7, Cm_alphadot -3.2, CL_M -0.81, Cm_M 0.27 are also
    #  tabulated. All four are outside this model's alpha/q/de form, exactly as
    #  the cruise set excludes its own speed and alpha-dot derivatives. Their
    #  omission is why section 5's phugoid and short-period offsets exist.)

    # -- Table IX-2, lateral-directional. Verbatim, per radian. --
    CYb, Clb, Cnb = -0.96, -0.221, 0.150
    Clp, Cnp = -0.45, -0.121
    Clr, Cnr = 0.101, -0.30
    Clda, Cnda = 0.0461, 0.0064
    CYdr, Cldr, Cndr = 0.175, 0.007, -0.109

    AR = b * b / S

    # Referenced to the trimmed condition, as for the cruise set: the stabiliser
    # at -2.1 deg carries the trim, so elevator is zero at alpha0.
    CL0 = CL_trim - CLa * alpha0
    Cm0 = -Cma * alpha0

    # Drag split. At 165 KTAS sea level the Mach number is 0.25, so the Korn wave
    # term is identically zero and the polar is the plain parabolic one:
    #   CD = CD0 + CL^2/(pi e AR),  dCD/dalpha = 2 CL CLa/(pi e AR) = CDa
    # Two equations, two unknowns, no reading off a chart -- which is why this
    # set carries none of the cruise set's +-0.003 CD0 uncertainty.
    pi_e_AR = 2.0 * CL_trim * CLa / CDa
    e = pi_e_AR / (math.pi * AR)
    CD0 = CD_trim - CL_trim**2 / pi_e_AR

    inertia = inertia_tensor(
        *(v * SLUG_FT2_TO_KG_M2 for v in (Ix, Iy, Iz, Ixz))
    )
    return Aircraft(
        mass=jnp.array(W * LB2KG),
        inertia=inertia,
        inertia_inv=jnp.linalg.inv(inertia),
        S=jnp.array(S * FT2M**2),
        b=jnp.array(b * FT2M),
        c=jnp.array(c * FT2M),
        CD0=jnp.array(CD0),
        e=jnp.array(e),
        AR=jnp.array(AR),
        # Same wing as the cruise set. Inert here -- the wave-drag term is zero
        # below the critical Mach -- but carried so the aircraft is still right
        # if it is ever flown faster.
        sweep=jnp.array(37.5 * DEG2RAD),
        t_over_c=jnp.array(0.09),
        kappa_airfoil=jnp.array(0.87),
        CL0=jnp.array(CL0),
        CLa=jnp.array(CLa),
        CLq=jnp.array(CLq),
        CLde=jnp.array(CLde),
        Cm0=jnp.array(Cm0),
        Cma=jnp.array(Cma),
        Cmq=jnp.array(Cmq),
        Cmde=jnp.array(Cmde),
        CYb=jnp.array(CYb),
        # Not tabulated for the 747 in ANY configuration, cruise included.
        CYp=jnp.array(0.0),
        CYr=jnp.array(0.0),
        CYdr=jnp.array(CYdr),
        Clb=jnp.array(Clb),
        Clp=jnp.array(Clp),
        Clr=jnp.array(Clr),
        # Table IX-2's footnote: "delta_a = total deflection of right inboard
        # aileron plus left inboard aileron with the effect of outboard ailerons
        # included". Taken as given, like every other number here.
        Clda=jnp.array(Clda),
        Cldr=jnp.array(Cldr),
        Cnb=jnp.array(Cnb),
        Cnp=jnp.array(Cnp),
        Cnr=jnp.array(Cnr),
        Cnda=jnp.array(Cnda),
        Cndr=jnp.array(Cndr),
        max_thrust=jnp.array(4 * 43500.0 * LBF2N),
        thrust_lapse=jnp.array(0.8),
        elevator_limit=jnp.array(25.0 * DEG2RAD),
        aileron_limit=jnp.array(20.0 * DEG2RAD),
        rudder_limit=jnp.array(25.0 * DEG2RAD),
    )


# ---------------------------------------------------------------------------
# Piper PA-28-180 Cherokee
#
# Source status: UNVERIFIABLE -- SOURCE NOT AVAILABLE.
#
# CLAIMED source: McCormick, "Aerodynamics, Aeronautics, and Flight Mechanics",
# worked example for the Cherokee 180. That claim reached this file through
# `aircraft_data_validated.py`, WHICH IS NOT IN THIS REPOSITORY, so neither the
# textbook nor the collation can be checked from anything held here. The earlier
# note that "two independent transcriptions of the same textbook table agree on
# every digit" describes work done in that missing file; it is reported here as
# a claim about an absent document, not as evidence.
#
# Left as it stands rather than replaced. Substituting numbers from some other
# source to make the citation resolvable would turn a documented gap into an
# undocumented one, and no source held by this project supplies this aircraft.
# What CAN be said from inside the repository is said below -- the source's own
# non-dimensional table is reproduced by the conversion chain -- and that is an
# internal consistency check on the transcription, not on the data.
#
# Independent evidence that the data has a real problem: the two routes to the
# tail arm, -Cmq/CLq and -Cmde/CLde, describe the same geometry and disagree by
# a factor of 2.0 here (1.280 against 2.562) while the two CR-2144 aircraft
# agree to 1.4% and 2.9%. The split is exactly the sourcing split. WHICH of the
# two estimates is wrong, or whether both are, cannot be determined.
# See AUDIT.md findings 20 and 35, and ASSUMPTIONS_AUDIT.md U19.
#
# Flight condition: level flight, 4,920 ft, V0 = 50 m/s, rho = 1.06. ISA density
# at that altitude is 1.0581, so the source condition is ISA to 0.18%.
#
# The source tabulates DIMENSIONAL derivatives. They are converted here with the
# same relations used for the 747 above. That conversion is checked against the
# source's own non-dimensional table (CZa -4.68, CMa -0.741, CMq -7.42,
# CZde -0.934, CMde -2.4): all five reproduce to better than 0.7%, which is what
# makes this set trustworthy rather than merely transcribed. The check is
# asserted in tests/test_aircraft.py, not just claimed here.
#
# Two numbers are flagged by the source's own author and carried as given rather
# than corrected, per the project rule against inventing plausible data:
#   Izz = 1400 < Iyy = 1700, atypical for a conventional aeroplane.
#   The lateral set has no independent second source.
# ---------------------------------------------------------------------------


def _cherokee_pa28_180() -> Aircraft:
    m = 1090.0  # kg
    S, b, c = 15.0, 9.11, 1.6  # m^2, m, m
    U0, rho = 50.0, 1.06  # m/s, kg/m^3
    Ixx, Iyy, Izz = 3100.0, 1700.0, 1400.0  # kg.m^2
    CD_trim = 0.0615  # TOTAL drag at trim, not parasite drag -- see below

    # -- longitudinal dimensional derivatives --
    Xw, Zw, Zq, Mw, Mq = 0.02323, -1.729, -1.6804, -0.2772, -2.207
    Zde, Mde = -17.01, -44.71
    # -- lateral dimensional derivatives (Ixz = 0, so these are unprimed) --
    Yv, Lv, Nv = -0.1444, -0.1166, 0.174
    Lp, Np, Lr, Nr = -2.283, -1.732, 1.053, -1.029
    Ydr, Ldr, Ndr = 2.113, 0.6133, -6.583
    Lda, Nda = 3.101, 0.0

    fc = FlightCondition(
        airspeed=U0, density=rho, mass=m, Ixx=Ixx, Iyy=Iyy, Izz=Izz, S=S, b=b, c=c
    )
    qS = fc.qS
    AR = b * b / S
    # theta0 = 0 in level flight, so alpha0 = 0 and the trim CL is the weight
    # coefficient. It agrees with the source's stated CL0 = 0.543 to 0.95%.
    CL_trim = m * G0 / qS

    CLa, CLq, CLde, Cma, Cmq, Cmde = from_dimensional_longitudinal(
        fc, Zw=Zw, Zq=Zq, Mw=Mw, Mq=Mq, Zde=Zde, Mde=Mde, CD=CD_trim
    )

    # Drag. The source's CD = 0.0615 is total drag at trim, and its own notes
    # warn against using it as a polar CD0. Oswald efficiency is recovered from
    # Xw exactly as for the 747 -- alpha0 = 0 collapses that relation to
    # dCD/dalpha = CL - Xw m U0 / qS -- and CD0 is what is left of CD_trim after
    # induced drag. That gives CD0 = 0.0343, inside the 0.03-0.04 band the
    # source predicts independently. The agreement is unforced, and it is the
    # reason this back-solve is trusted.
    CDa = CL_trim - Xw * m * U0 / qS
    e = 2.0 * CL_trim * CLa / (math.pi * AR * CDa)
    CD0 = CD_trim - CL_trim**2 / (math.pi * e * AR)

    CYb, Clb, Cnb, Clp, Cnp, Clr, Cnr = from_dimensional_lateral(
        fc, Yv=Yv, Lv=Lv, Nv=Nv, Lp=Lp, Np=Np, Lr=Lr, Nr=Nr
    )
    CYdr, Clda, Cnda, Cldr, Cndr = from_dimensional_controls(
        fc, Ydr=Ydr, Lda=Lda, Nda=Nda, Ldr=Ldr, Ndr=Ndr
    )

    inertia = inertia_tensor(Ixx, Iyy, Izz, 0.0)
    return Aircraft(
        mass=jnp.array(m),
        inertia=inertia,
        inertia_inv=jnp.linalg.inv(inertia),
        S=jnp.array(S),
        b=jnp.array(b),
        c=jnp.array(c),
        CD0=jnp.array(CD0),
        e=jnp.array(e),
        AR=jnp.array(AR),
        # Unswept, NACA 65(2)-415. At M 0.15 the Korn/Lock term is identically
        # zero, so these exist only to satisfy the shared drag build-up.
        sweep=jnp.array(0.0),
        t_over_c=jnp.array(0.15),
        kappa_airfoil=jnp.array(0.87),
        CL0=jnp.array(CL_trim),  # alpha0 = 0, so CL0 is the trim CL
        CLa=jnp.array(CLa),
        CLq=jnp.array(CLq),
        CLde=jnp.array(CLde),
        # alpha0 = 0 and the aircraft is trimmed there with zero elevator, so
        # Cm0 = -Cma * alpha0 = 0, the same convention used for the 747.
        Cm0=jnp.array(0.0),
        Cma=jnp.array(Cma),
        Cmq=jnp.array(Cmq),
        Cmde=jnp.array(Cmde),
        CYb=jnp.array(CYb),
        # Not tabulated by the source, as for the 747. Zero matches the source
        # rather than inventing a value; it understates Dutch roll damping.
        CYp=jnp.array(0.0),
        CYr=jnp.array(0.0),
        CYdr=jnp.array(CYdr),
        Clb=jnp.array(Clb),
        Clp=jnp.array(Clp),
        Clr=jnp.array(Clr),
        Clda=jnp.array(Clda),
        Cldr=jnp.array(Cldr),
        Cnb=jnp.array(Cnb),
        Cnp=jnp.array(Cnp),
        Cnr=jnp.array(Cnr),
        Cnda=jnp.array(Cnda),
        Cndr=jnp.array(Cndr),
        # MODELLING CHOICE, not source data -- the source has no propulsion at
        # all. Sea-level rated power for the Lycoming O-360 (180 hp) at 80%
        # propeller efficiency, evaluated at the cruise speed. The plant applies
        # thrust independent of airspeed, so this is a fixed-thrust stand-in for
        # a fixed-power propeller, calibrated only at cruise. Altitude is not
        # double-counted: the rating is sea-level and thrust_lapse handles the
        # falloff, which is why cruise trims at 66% throttle rather than 57%.
        max_thrust=jnp.array(0.8 * 180.0 * HP2W / U0),
        thrust_lapse=jnp.array(1.0),  # normally aspirated piston
        # MODELLING CHOICE -- deflection limits are not in the source.
        elevator_limit=jnp.array(25.0 * DEG2RAD),
        aileron_limit=jnp.array(20.0 * DEG2RAD),
        rudder_limit=jnp.array(25.0 * DEG2RAD),
    )


# ---------------------------------------------------------------------------
# Cessna 172
#
# Source status: UNVERIFIABLE -- SOURCE NOT AVAILABLE.
#
# CLAIMED source: Roskam and USAF DATCOM as transcribed in PyFME
# (AeroPython/PyFME, MIT). That claim reached this file through
# `aircraft_data_validated.py`, WHICH IS NOT IN THIS REPOSITORY, so the chain
# stops one link short of anything checkable. The corroboration previously
# claimed here -- "a second citation of the same Roskam table" -- also lives in
# that missing file and is likewise unverified. The inertia ordering
# Izz > Iyy > Ixx being the physically expected one is a plausibility remark and
# is worth exactly that.
#
# Not replaced, for the reason given on the Cherokee above: finding substitute
# numbers elsewhere would convert a documented gap into an undocumented one.
# The same tail-arm disagreement applies and is worse -- 0.856 by the rate pair
# against 2.468 by the control pair, a factor of 2.9.
# See AUDIT.md findings 20 and 35, and ASSUMPTIONS_AUDIT.md U19.
#
# Unlike the Cherokee this data is already NON-DIMENSIONAL, tabulated against
# angle of attack, so no dimensional conversion is needed. The project's aero
# model is linear in alpha with a parabolic polar, so the tables are fitted over
# alpha <= 10 deg -- below the stall knee, where they are straight to within
# 0.03 in CL and 0.0004 in CD.
#
# *** THE TABLE CARRIES STALL DATA THIS MODEL CANNOT USE. *** CL_data runs to
# CLmax 1.889 at 19.5 deg. flightsim.aero is linear in alpha and has no stall,
# so above roughly 10 deg this aircraft reports lift the source says it does not
# have. That matters for the intended vortex work, where the source's own note
# estimates alpha excursions of order 20 deg. The tables are therefore kept
# whole in CESSNA172_TABLES so a nonlinear aero path can use them later without
# going back to source.
#
# *** THE RUDDER IS ABSENT. *** Cl_delta_rud and CN_delta_rud are deliberately
# omitted by the source (PyFME applies an undocumented x0.075 tuning factor to
# them). The one rudder derivative given, Ydr = -20.63, is negative and about
# five times the Cherokee's CYdr in magnitude, inconsistent with the sign
# convention used throughout this package. A side force with no matching yawing
# moment is worse than no rudder, so the whole rudder set is zeroed here.
# Consequences: no rudder authority, and the autopilot's sideslip term does
# nothing, so turns in this aircraft are uncoordinated.
# ---------------------------------------------------------------------------

# Non-dimensional aero against angle of attack, degrees. Kept whole so that a
# nonlinear table-lookup aero path can use it without re-sourcing.
CESSNA172_TABLES: dict[str, tuple[float, ...]] = {
    "alpha_deg": (-7.5, -5, -2.5, 0, 2.5, 5, 7.5, 10, 15, 17, 18, 19.5),
    "CL": (-0.571, -0.321, -0.083, 0.148, 0.392, 0.65, 0.918, 1.195,
           1.659, 1.789, 1.84, 1.889),
    "CD": (0.044, 0.034, 0.03, 0.03, 0.036, 0.048, 0.067, 0.093,
           0.15, 0.169, 0.177, 0.184),
    "Cm": (0.0597, 0.0498, 0.0314, 0.0075, -0.0248, -0.068, -0.1227, -0.1927,
           -0.3779, -0.4605, -0.5043, -0.5496),
    "Clb": (-0.178, -0.186, -0.1943, -0.202, -0.2103, -0.219, -0.2283,
            -0.2376, -0.2516, -0.255, -0.256, -0.257),
    # The source corrects a sign-flip typo at alpha = 2.5 (PyFME has +0.487
    # against neighbours of -0.44 to -0.51); the corrected value is used.
    "Clp": (-0.4968, -0.4678, -0.4489, -0.4595, -0.487, -0.5085, -0.5231,
            -0.4916, -0.301, -0.203, -0.1498, -0.0671),
    "Clr": (-0.09675, -0.05245, -0.01087, 0.02986, 0.07342, 0.1193, 0.1667,
            0.2152, 0.2909, 0.3086, 0.3146, 0.3197),
    "Cnp": (0.03, 0.016, 0.00262, -0.0108, -0.0245, -0.0385, -0.0528, -0.0708,
            -0.113, -0.1284, -0.1356, -0.1422),
    "Cnr": (-0.028, -0.027, -0.027, -0.0275, -0.0293, -0.0325, -0.037, -0.043,
            -0.05484, -0.058, -0.0592, -0.06015),
}

_C172_LINEAR_MAX_ALPHA_DEG = 10.0


def _cessna_172() -> Aircraft:
    m = 1043.3  # kg, 2,300 lb
    S, b, c = 16.2, 10.91184, 1.49352  # m^2, m, m
    Ixx, Iyy, Izz = 1285.3, 1824.7, 2666.7  # kg.m^2
    AR = b * b / S

    # Cruise for the fit and the trim reference. The source linearises at
    # 67 m/s, but that sits at about 93% of full-throttle thrust for a 150 hp
    # 172, and its own stated trim alpha of 2.5 deg is inconsistent with its CL
    # table by roughly 13 m/s. 60 m/s (117 KTAS) at 5,000 ft is a
    # self-consistent ~70% power cruise. The tables are non-dimensional, so no
    # aero number here depends on that choice.
    U0, rho = 60.0, 1.055

    tables = {k: jnp.array(v) for k, v in CESSNA172_TABLES.items()}
    alpha = tables["alpha_deg"] * DEG2RAD
    linear = tables["alpha_deg"] <= _C172_LINEAR_MAX_ALPHA_DEG
    fit_a = alpha[linear]

    CLa, CL0 = jnp.polyfit(fit_a, tables["CL"][linear], 1)
    Cma, Cm0 = jnp.polyfit(fit_a, tables["Cm"][linear], 1)
    # Parabolic polar by least squares on CD against CL^2 over the same range.
    # e comes out 0.97 -- close to the elliptical limit and optimistic for a
    # strut-braced high-wing aeroplane, but it is what the DATCOM table implies
    # and it is not adjusted here.
    k, CD0 = jnp.polyfit(tables["CL"][linear] ** 2, tables["CD"][linear], 1)
    e = 1.0 / (jnp.pi * AR * k)

    # Reference alpha: level-flight incidence at the cruise condition. The
    # alpha-dependent lateral derivatives are read there.
    qS = 0.5 * rho * U0**2 * S
    alpha_ref = (m * G0 / qS - CL0) / CLa

    # The control derivatives are the only DIMENSIONAL numbers in this set, and
    # they belong to the source's own linearisation at 67 m/s -- NOT the 60 m/s
    # cruise chosen above. Recovering them at the wrong dynamic pressure inflates
    # all four by 25%, which is exactly the bug FlightCondition exists to prevent,
    # so the condition is stated once here and the helper does the rest.
    source_fc = FlightCondition(
        airspeed=67.0, density=rho, mass=m, Ixx=Ixx, Iyy=Iyy, Izz=Izz, S=S, b=b, c=c
    )
    _, _, CLde, _, _, Cmde = from_dimensional_longitudinal(
        source_fc, Zw=0.0, Zq=0.0, Mw=0.0, Mq=0.0, Zde=-17.19, Mde=-36.23, CD=0.0
    )
    # Rudder deliberately omitted -- see the module comment above.
    _, Clda, Cnda, _, _ = from_dimensional_controls(source_fc, Lda=135.9, Nda=-3.108)

    def at_reference(name):
        return jnp.interp(alpha_ref, alpha, tables[name])

    inertia = inertia_tensor(Ixx, Iyy, Izz, 0.0)
    return Aircraft(
        mass=jnp.array(m),
        inertia=inertia,
        inertia_inv=jnp.linalg.inv(inertia),
        S=jnp.array(S),
        b=jnp.array(b),
        c=jnp.array(c),
        CD0=CD0,
        e=e,
        AR=jnp.array(AR),
        sweep=jnp.array(0.0),  # unswept NACA 2412; wave drag is zero at M 0.18
        t_over_c=jnp.array(0.12),
        kappa_airfoil=jnp.array(0.87),
        CL0=CL0,
        CLa=CLa,
        CLq=jnp.array(7.282),  # constant with alpha in the source table
        CLde=jnp.array(CLde),
        Cm0=Cm0,
        Cma=Cma,
        Cmq=jnp.array(-6.232),  # constant with alpha in the source table
        Cmde=jnp.array(Cmde),
        CYb=jnp.array(-0.268),
        CYp=jnp.array(0.0),  # not tabulated
        CYr=jnp.array(0.0),  # not tabulated
        CYdr=jnp.array(0.0),  # zeroed -- see the rudder note above
        Clb=at_reference("Clb"),
        Clp=at_reference("Clp"),
        Clr=at_reference("Clr"),
        # Control power comes from the source's derived linear point, which its
        # author rates MEDIUM confidence; unlike the tables above these are not
        # published numbers. Clda 0.42 is high against the 747's 0.014 and is
        # the least certain number in this definition.
        Clda=jnp.array(Clda),
        Cldr=jnp.array(0.0),  # zeroed -- see the rudder note above
        Cnb=jnp.array(0.0126),
        Cnp=at_reference("Cnp"),
        Cnr=at_reference("Cnr"),
        Cnda=jnp.array(Cnda),
        Cndr=jnp.array(0.0),  # zeroed -- see the rudder note above
        # MODELLING CHOICE, as for the Cherokee. Lycoming O-320, 150 hp sea-level
        # rating at the 2,300 lb gross weight this data is quoted for, 80%
        # propeller efficiency, evaluated at cruise. thrust_lapse handles
        # altitude, so cruise trims near 80% throttle at 5,000 ft.
        max_thrust=jnp.array(0.8 * 150.0 * HP2W / U0),
        thrust_lapse=jnp.array(1.0),
        # MODELLING CHOICE -- deflection limits are not in the source.
        elevator_limit=jnp.array(25.0 * DEG2RAD),
        aileron_limit=jnp.array(20.0 * DEG2RAD),
        rudder_limit=jnp.array(25.0 * DEG2RAD),
    )


def _boeing_737() -> Aircraft:
    """Boeing 737, linearised from JSBSim 1.3.1's own 737 model at cruise.

    *** NOT A QUALIFIED SOURCE. NOT VALID AWAY FROM CRUISE. ***

    737.xml's own fileheader says the model was built from public data,
    technical reports, textbooks "and guesses"; that validation extends only to
    the extent that it "seems to fly right"; and that it is for "educational and
    entertainment purposes only". Nothing here supports any claim about a real
    737, and none is made anywhere. This entry exists so that flightsim's solver
    can be checked against an independent, mature engine fed the SAME
    coefficients -- a disagreement is then a defect in one of the two
    implementations, whatever the numbers describe. See
    docs/superpowers/specs/2026-08-20-jsbsim-737-verification-design.md.

    VALIDITY BAND: linearised about 30,000 ft, M 0.78, alpha 1.965 deg. Unlike
    every other entry in REGISTRY -- linear derivative sets from CR-2144 and
    Nelson, valid across the ordinary linear range -- this one is a local fit to
    a NONLINEAR model, and it degrades away from that point in known ways:

      - JSBSim's CL(alpha) is a table peaking at 1.20 near 13 deg and falling.
        CL0 + CLa*alpha keeps climbing. Above about 10 deg this entry does not
        merely lose accuracy: it has NO stall behaviour and reports lift the
        source model does not have.
      - CD0 here is the value at the trim alpha of a table running 0.021 at
        0 deg to 0.042 at 15 deg, so drag is progressively under-predicted as
        alpha departs from cruise. It also absorbs JSBSim's CDde term
        (0.059*|de|), frozen at its trim elevator, which flightsim has no home
        for -- so moving the elevator changes no drag here.
      - max_thrust is NOT a sea-level static rating. JSBSim blends idle and
        military thrust nonlinearly with throttle (thrust/throttle varies 4.3x
        between throttle 0.2 and 1.0) while this model is linear in throttle,
        so max_thrust and thrust_lapse are fitted AT THE TRIM THROTTLE over
        25,000-35,000 ft. They reproduce JSBSim to 1.8% there and are wrong
        outside it. mach_ram is fitted over M 0.6-1.0 and is wrong at low speed.
      - sweep, t_over_c and kappa_airfoil are NOT 737 geometry -- 737.xml gives
        neither sweep nor thickness. They are chosen to place this project's
        Korn/Lock drag rise at the Mach where JSBSim's CDmach table leaves zero
        (0.79), because a conventional kappa of 0.87 would otherwise add 0.0086
        of wave drag where JSBSim has exactly none, a 32% error on this CD0.
        Wave drag is UNTESTED by the comparison: at M 0.78 both engines give
        exactly zero.

    Nothing in the code prevents use outside the band. To work at another
    condition, re-run scripts/gen_jsbsim_reference.py there; it is parameterised
    for exactly that.

    MOMENTS ARE REFERRED TO THE AERORP, not the CG -- see the aero_ref field.
    That is why Cma is -0.6, Clb is -0.09 and Cnb is +0.26: they are 737.xml's
    own constants, and the CG offset is generated by r x F at run time instead
    of being baked into them. Referred to the CG they would be -1.13, -0.144 and
    +0.273, and which values you got would depend on the fuel state.

    The derivatives are still RECOVERED from the running engine rather than read
    from the file -- the moments are referred back with M_arp = M_cg - r x F
    before differencing -- and landing on the file's own constants at BOTH
    recovery conditions is the check that the referencing is right.

    The pitch axis needs a least-squares fit rather than central differences,
    because setting alpha away from trim also sets alphadot: measured,
    d(alphadot)/d(alpha) = -0.529 /s, which contaminates a differenced Cma by
    +0.067 and was what made the pre-AERORP recovery read -1.0637 instead of the
    alphadot-free -1.1309.

    Cmq (-27.000) and Cmadot (-16.000) go in SEPARATELY, as 737.xml defines
    them and as the fit recovers them. They were folded into a single q term
    while flightsim applied alphadot for the wind only; dynamics.derivatives now
    resolves the aircraft's own alphadot as well, so folding would apply Cmadot
    twice. Note the fit pins their SUM to machine precision and the split only to
    about 0.04, because alphadot and q are nearly collinear in any reachable
    state -- so the split is the softer of the two numbers.

    Every literal below is reproduced by scripts/gen_jsbsim_reference.py and
    cross-checked against the frozen reference in
    test_jsbsim_737.test_737_matches_the_recovered_reference_entry.
    """
    # -- geometry and mass, from 737.xml's <metrics> and the engine's own
    #    mass properties with the modelled fuel load aboard --
    S = 1171.0 * FT2M**2  # ft^2
    b = 94.7 * FT2M  # ft
    c = 12.31 * FT2M  # ft
    # slug-ft^2, as the engine reports them with fuel aboard -- NOT the bare
    # <mass_balance> figures, which exclude the tanks. The Ixz sign convention
    # is confirmed rather than assumed: 737.xml carries
    # negated_crossproduct_inertia="true", the engine reports +19109.13, and
    # inertia_tensor's own positive-forward-up convention negates it to the
    # -25908.5 the reference records. Asserted in
    # test_737_mass_and_inertia_match_the_engine.
    inertia = inertia_tensor(
        *(v * SLUG_FT2_TO_KG_M2 for v in (
            591572.3456383009, 1539552.6887960227,
            1986235.3649231757, 19109.131861384914,
        ))
    )
    return Aircraft(
        mass=jnp.array(107000.0 * LBF2N / G0),  # lb, engine total with fuel
        inertia=inertia,
        inertia_inv=jnp.linalg.inv(inertia),
        S=jnp.array(S),
        b=jnp.array(b),
        c=jnp.array(c),
        # CD0 absorbs everything at trim that is not induced drag; wave drag is
        # zero at M 0.78 by construction.
        CD0=jnp.array(0.02423738849922),
        # The profile-drag rise 737.xml carries as its CD0(alpha) table,
        # recovered from the engine as dCD/dalpha minus the induced part.
        CD_alpha=jnp.array(0.08465218977423),
        # Chosen so CL^2/(pi e AR) reproduces JSBSim's CDi = 0.043 CL^2 exactly.
        e=jnp.array(0.966581789644),
        AR=jnp.array(b * b / S),
        # FITTED, not geometry -- see the docstring.
        sweep=jnp.array(25.0 * DEG2RAD),
        t_over_c=jnp.array(0.12),
        kappa_airfoil=jnp.array(0.9872558124305),
        # Intercepts solved so the model reproduces JSBSim's own coefficients
        # AT the reference point, which is what "linearised about cruise" means.
        CL0=jnp.array(0.1999999945286),
        CLa=jnp.array(4.347826086957),
        CLq=jnp.array(0.0),  # 737.xml defines none; measured 5.5e-4, the drift floor
        CLde=jnp.array(0.2),
        Cm0=jnp.array(-2.956611522467e-08),
        Cma=jnp.array(-0.599999073831),
        Cmq=jnp.array(-27.00022042486),  # Cmq + Cmadot; see the docstring
        Cmde=jnp.array(-0.8490000019392),
        CYb=jnp.array(-1.0),
        CYp=jnp.array(0.0),  # 737.xml defines none
        CYr=jnp.array(0.0),  # 737.xml defines none
        CYdr=jnp.array(0.0),  # 737.xml defines none: rudder makes no side force
        Clb=jnp.array(-0.08999979648072),
        Clp=jnp.array(-0.4000000495119),
        Clr=jnp.array(0.09000127553653),
        Clda=jnp.array(0.07387000014111),  # Mach-scheduled in JSBSim; M 0.78 value
        Cldr=jnp.array(0.01),
        Cnb=jnp.array(0.2599999111519),
        Cnp=jnp.array(0.0),  # 737.xml defines none
        Cnr=jnp.array(-0.350004258434),
        Cnda=jnp.array(0.0),  # 737.xml defines none
        Cndr=jnp.array(-0.2),
        # FITTED at the trim throttle over the cruise band -- see the docstring.
        max_thrust=jnp.array(101375.4010746),
        thrust_lapse=jnp.array(0.7207901171515),
        # Deflection limits are 737.xml's aerosurface_scale ranges, so trim
        # bounds here are the same bounds JSBSim's own FCS enforces.
        elevator_limit=jnp.array(0.3),
        aileron_limit=jnp.array(0.35),
        rudder_limit=jnp.array(0.35),
        mach_ram=jnp.array(0.2510939315992),
        # Sideslip drag. 737.xml's CDbeta table gives 0.05 at beta = 0.26 rad;
        # 0.05/0.26^2 puts the quadratic through that breakpoint, which is where
        # the table's author presumably placed a real number. Below it this
        # gives LESS drag than the table -- 20% of it at 3 deg -- and that is
        # deliberate: linear interpolation through zero makes the table's
        # near-origin behaviour proportional to |beta|, which no symmetric
        # airframe can produce. See the CD_beta field comment.
        CD_beta=jnp.array(0.05 / 0.26**2),
        # 737.xml PITCH/Cmadot. Applies to the WIND-induced alphadot only, so it
        # is exactly zero in still air and does not disturb any result above;
        # Cmq stays at the folded -43.0, which is what the aircraft's own motion
        # needs. JSBSim's 737 defines no CLadot, so that stays zero -- physically
        # inconsistent with carrying a Cmadot, but it is what the model says.
        Cmadot=jnp.array(-15.99977957545),
        # Body-axis vector from the CG to 737.xml's AERORP: 1.183 ft aft and
        # 4.925 ft above, read from the engine rather than transcribed.
        aero_ref=jnp.array([-0.3603476635514, 0.0, -1.500261682243]),
    )


def _boeing_737_approach() -> Aircraft:
    """The same JSBSim 737, recovered at 5,000 ft and M 0.40 instead of cruise.

    READ _boeing_737's DOCSTRING FIRST. Every caveat there applies here, with
    the band moved: this set is linearised about alpha 3.63 deg rather than
    1.97 deg, and is no more valid outside its own neighbourhood than that one
    is outside its.

    It exists because one reference point cannot distinguish a solver that is
    correct from one that is correct in a single place. PROJECT.md section 4
    records this project's phugoid error falling from 17.8% at M 0.80 to 0.4% at
    M 0.25 with identical code, because compressibility drives the terms the
    model omits. Comparing at a second, much lower Mach turns that from an
    inference into a measurement against an independent engine.

    What moves between the two sets is itself informative, and all of it is the
    nonlinearity the linearisation is hiding:

      Cmde   -0.894 -> -1.067   JSBSim schedules it on Mach
      Clda   +0.0739 -> +0.0866  likewise
      CD0    +0.0271 -> +0.0312  the CD0(alpha) table, read at a larger alpha
      Cm0    -0.0107 -> -0.0141  the AERORP offset moment, at a larger alpha

    What does NOT move is the check: CLa, CLde, CYb, Clp, Clr, Cnr, Cldr and
    Cndr are identical to seven figures at both conditions, because 737.xml
    defines them as constants. A solver bug that depended on flight condition
    could not leave those unchanged while moving the others by the amounts the
    tables predict.

    M 0.35 was tried first and is NOT trimmable -- JSBSim reports "wdot doesn't
    appear to be trimmable", because the lift needed there runs the aircraft up
    the nonlinear part of its CL table.
    """
    S = 1171.0 * FT2M**2
    b = 94.7 * FT2M
    c = 12.31 * FT2M
    inertia = inertia_tensor(
        *(v * SLUG_FT2_TO_KG_M2 for v in (
            591572.3456383009, 1539552.6887960227,
            1986235.3649231757, 19109.131861384914,
        ))
    )
    return Aircraft(
        mass=jnp.array(107000.0 * LBF2N / G0),
        inertia=inertia,
        inertia_inv=jnp.linalg.inv(inertia),
        S=jnp.array(S),
        b=jnp.array(b),
        c=jnp.array(c),
        CD0=jnp.array(0.02570062073226),
        # The profile-drag rise 737.xml carries as its CD0(alpha) table,
        # recovered from the engine as dCD/dalpha minus the induced part.
        CD_alpha=jnp.array(0.08641089758596),
        e=jnp.array(0.966581789644),
        AR=jnp.array(b * b / S),
        sweep=jnp.array(25.0 * DEG2RAD),
        t_over_c=jnp.array(0.12),
        kappa_airfoil=jnp.array(1.002098775681),
        CL0=jnp.array(0.199999990422),
        CLa=jnp.array(4.347826086957),
        CLq=jnp.array(0.0),
        CLde=jnp.array(0.2),
        Cm0=jnp.array(-2.057942546435e-05),
        Cma=jnp.array(-0.5996582506232),
        Cmq=jnp.array(-27.03541066755),
        Cmde=jnp.array(-1.020000001599),
        CYb=jnp.array(-1.0),
        CYp=jnp.array(0.0),
        CYr=jnp.array(0.0),
        CYdr=jnp.array(0.0),
        Clb=jnp.array(-0.08999973475265),
        Clp=jnp.array(-0.4000000514816),
        Clr=jnp.array(0.09000062168166),
        Clda=jnp.array(0.08660000011697),
        Cldr=jnp.array(0.01),
        Cnb=jnp.array(0.2599998844166),
        Cnp=jnp.array(0.0),
        Cnr=jnp.array(-0.3500024048283),
        Cnda=jnp.array(0.0),
        Cndr=jnp.array(-0.2),
        max_thrust=jnp.array(82578.56222904),
        thrust_lapse=jnp.array(0.9581403997535),
        elevator_limit=jnp.array(0.3),
        aileron_limit=jnp.array(0.35),
        rudder_limit=jnp.array(0.35),
        mach_ram=jnp.array(0.3345799618749),
        CD_beta=jnp.array(0.05 / 0.26**2),
        Cmadot=jnp.array(-15.96458939685),
        # Body-axis vector from the CG to 737.xml's AERORP: 1.183 ft aft and
        # 4.925 ft above, read from the engine rather than transcribed.
        aero_ref=jnp.array([-0.3603476635514, 0.0, -1.500261682243]),
    )


REGISTRY: dict[str, Aircraft] = {
    "boeing747": _boeing_747(),
    "boeing747_approach": _boeing_747_approach(),
    "boeing737": _boeing_737(),
    "boeing737_approach": _boeing_737_approach(),
    "cherokee": _cherokee_pa28_180(),
    "cessna172": _cessna_172(),
}

# Reference trim conditions, for the trim solver and for tests. SI.
CRUISE: dict[str, dict[str, float]] = {
    "boeing747": {"altitude": 40000.0 * FT2M, "airspeed": 774.0 * FT2M},
    # Table IX-2 is AT sea level. Flown a little above it for the microburst
    # work, which is a 3% density extrapolation rather than the 40,000 ft one.
    "boeing747_approach": {"altitude": 0.0, "airspeed": 165.0 * KT2MS},
    # The condition the 737's derivatives were recovered at, and the ONLY one it
    # is valid near -- see _boeing_737's docstring.
    #
    # The altitude is DENSITY-MATCHED, not nominal: JSBSim flies this at
    # 30,000 ft, but flightsim's ISA uses geometric altitude where the standard
    # uses geopotential, so its density there is 0.159% below JSBSim's. Since
    # qbar is proportional to rho, running at a nominal 30,000 ft would put that
    # bias on every force in the comparison. 9130.83 m is 43.22 ft lower and
    # matches JSBSim's density to 1e-16 relative.
    "boeing737": {"altitude": 9130.825908961, "airspeed": 236.5191917152},
    # The second recovery condition, 5,000 ft and M 0.40. Density-matched for
    # the same reason, though the shift is only -1.45 ft this low down.
    "boeing737_approach": {"altitude": 1523.558080263, "airspeed": 133.7577996784},
    "cherokee": {"altitude": 4920.0 * FT2M, "airspeed": 50.0},
    "cessna172": {"altitude": 5000.0 * FT2M, "airspeed": 60.0},
}
