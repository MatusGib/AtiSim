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

from flightsim.units import DEG2RAD, FT2M, LB2KG, LBF2N, SLUG_FT2_TO_KG_M2


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
    Xw, Zw, Zq, Mw, Mq, Zde, Mde = 0.0389, -0.317, -5.16, -0.00105, -0.330, -17.9, -1.16
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


REGISTRY: dict[str, Aircraft] = {
    "boeing747": _boeing_747(),
}

# Reference trim conditions, for the trim solver and for tests. SI.
CRUISE: dict[str, dict[str, float]] = {
    "boeing747": {"altitude": 40000.0 * FT2M, "airspeed": 774.0 * FT2M},
}
