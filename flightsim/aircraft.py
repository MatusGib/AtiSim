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
from flightsim.units import DEG2RAD, FT2M, HP2W, LB2KG, LBF2N, SLUG_FT2_TO_KG_M2


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

    # Propulsion: thrust = throttle * max_thrust * (rho/rho0)^thrust_lapse
    max_thrust: Array  # N, sea-level static
    thrust_lapse: Array  # density-ratio exponent

    # Control deflection limits (rad), used by trim bounds and the autopilot
    elevator_limit: Array
    aileron_limit: Array
    rudder_limit: Array


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
# stability-to-body rotation is applied. The recovered values agree closely with
# the report's own landing-configuration table (Cmde -1.44 vs -1.40, CLde 0.364
# vs 0.396, Clb -0.294 vs -0.281, Cnb +0.172 vs +0.184), which is the check that
# the conversion chain is right.
# ---------------------------------------------------------------------------

_B747_G = 32.174  # ft/s^2, used only to turn the tabulated weight into mass


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
# Piper PA-28-180 Cherokee
#
# Source: McCormick, "Aerodynamics, Aeronautics, and Flight Mechanics", worked
# example for the Cherokee 180, as collated in aircraft_data_validated.py. Two
# independent transcriptions of the same textbook table agree on every digit.
# This is a published worked example, not a flight-test report.
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
# Source: Roskam and USAF DATCOM as transcribed in PyFME (AeroPython/PyFME,
# MIT), collated in aircraft_data_validated.py. Mass, inertia and geometry are
# independently corroborated by a second citation of the same Roskam table, and
# the inertia ordering Izz > Iyy > Ixx is the physically expected one.
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


REGISTRY: dict[str, Aircraft] = {
    "boeing747": _boeing_747(),
    "cherokee": _cherokee_pa28_180(),
    "cessna172": _cessna_172(),
}

# Reference trim conditions, for the trim solver and for tests. SI.
CRUISE: dict[str, dict[str, float]] = {
    "boeing747": {"altitude": 40000.0 * FT2M, "airspeed": 774.0 * FT2M},
    "cherokee": {"altitude": 4920.0 * FT2M, "airspeed": 50.0},
    "cessna172": {"altitude": 5000.0 * FT2M, "airspeed": 60.0},
}
