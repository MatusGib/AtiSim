"""Aerodynamic force and moment build-up.

The one rule that matters: this module never sees inertial velocity. It is given
velocity and angular rate *relative to the air mass*, so turbulence is a matter
of what the caller passes in, not a change to anything here.

Sign conventions:
  alpha  positive nose-up relative to the relative wind
  beta   positive nose-left (relative wind from the right)
  de     positive trailing-edge down, so CLde > 0 and Cmde < 0
  da     positive gives positive (right-wing-down) roll, so Clda > 0
  dr     positive trailing-edge left, so CYdr > 0 and Cndr < 0
"""

import jax.numpy as jnp
from jax import Array

from flightsim.aircraft import Aircraft
from flightsim.atmosphere import RHO0
from flightsim.state import Controls

# Airspeed floor. Purely a NaN guard, and it is applied ONLY where something
# divides by V: beta, and the three non-dimensional rates. Far below any flight
# speed, so it never binds in normal operation.
#
# It is deliberately NOT applied to dynamic pressure or to Mach. Neither divides
# by V and both are finite at V = 0, so flooring them would report a force the
# aircraft does not have -- a stationary airframe used to produce 1.4 to 171 N
# out of still air through the residual CL0 term, and free fall did not read
# n_z = 0. `jnp.maximum(x, 1.0)` returns x exactly for x >= 1, so confining the
# floor this way changes nothing at or above 1 m/s, bit for bit.
#
# `air_data` still REPORTS the floored airspeed, which is what keeps alpha, beta
# and the rates finite at rest and is what `sensors.sense` shows on the ASI.
# That residue is bounded, tested, and separate from the force path.
V_MIN = 1.0  # m/s

# Lock's fourth-power drag-rise law is anchored on the definition of drag
# divergence, dCD/dM = 0.1 at M_dd. With CD_wave = 20 (M - M_crit)^4 that fixes
# M_crit = M_dd - (0.1/80)^(1/3).
_MDD_OFFSET = (0.1 / 80.0) ** (1.0 / 3.0)  # 0.10772


def drag_divergence_mach(CL: Array, ac: Aircraft) -> Array:
    """Korn equation.

    M_dd = kappa/cos(L) - (t/c)/cos^2(L) - CL/(10 cos^3(L))

    kappa is the airfoil technology factor: ~0.87 conventional, ~0.95
    supercritical. The CL term is what makes the drag rise lift-dependent, which
    is the whole point -- it couples wave drag to angle of attack.
    """
    cos_sweep = jnp.cos(ac.sweep)
    return (
        ac.kappa_airfoil / cos_sweep
        - ac.t_over_c / cos_sweep**2
        - CL / (10.0 * cos_sweep**3)
    )


def wave_drag(mach: Array, CL: Array, ac: Aircraft) -> Array:
    """Compressibility drag rise above the critical Mach number.

    Identically zero for the light aircraft, which never approach M_crit, so
    this needs no special-casing per aircraft.
    """
    m_crit = drag_divergence_mach(CL, ac) - _MDD_OFFSET
    return 20.0 * jnp.maximum(mach - m_crit, 0.0) ** 4


def air_data(vel_rel: Array) -> tuple[Array, Array, Array]:
    """(true airspeed, alpha, beta) from body-axis velocity relative to air."""
    u, v, w = vel_rel
    V = jnp.maximum(jnp.linalg.norm(vel_rel), V_MIN)
    alpha = jnp.arctan2(w, u)
    beta = jnp.arcsin(jnp.clip(v / V, -1.0, 1.0))
    return V, alpha, beta


def coefficients(
    vel_rel: Array,
    omega_rel: Array,
    controls: Controls,
    ac: Aircraft,
    a_sound: Array,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """(CL, CD, CY, Cl, Cm, Cn). Lift and drag wind-axis, the rest body-axis.

    Speed of sound is passed rather than Mach so that the Mach used for wave
    drag is built from the same airspeed as dynamic pressure -- the TRUE one,
    not the floored one the rates below need.
    """
    V, alpha, beta = air_data(vel_rel)
    p, q, r = omega_rel

    # Non-dimensional rates. Span for the lateral pair, chord for pitch. These
    # are three of the four divisions the V_MIN floor exists for, so they take
    # the floored V that `air_data` returns.
    p_hat = p * ac.b / (2.0 * V)
    q_hat = q * ac.c / (2.0 * V)
    r_hat = r * ac.b / (2.0 * V)

    de, da, dr = controls.elevator, controls.aileron, controls.rudder

    CL = ac.CL0 + ac.CLa * alpha + ac.CLq * q_hat + ac.CLde * de
    Cm = ac.Cm0 + ac.Cma * alpha + ac.Cmq * q_hat + ac.Cmde * de
    # Parabolic core plus a lift-dependent compressibility rise.
    # Mach divides by the speed of sound, not by V, so it takes the true
    # airspeed. Below the floor this is unobservable either way -- M_crit is
    # above 0.5 for every aircraft here and `wave_drag` is identically zero at
    # walking pace -- but the floor has no defence here and does not belong.
    CD = (
        ac.CD0
        + CL**2 / (jnp.pi * ac.e * ac.AR)
        + wave_drag(jnp.linalg.norm(vel_rel) / a_sound, CL, ac)
    )

    CY = ac.CYb * beta + ac.CYp * p_hat + ac.CYr * r_hat + ac.CYdr * dr
    Cl = (
        ac.Clb * beta
        + ac.Clp * p_hat
        + ac.Clr * r_hat
        + ac.Clda * da
        + ac.Cldr * dr
    )
    Cn = (
        ac.Cnb * beta
        + ac.Cnp * p_hat
        + ac.Cnr * r_hat
        + ac.Cnda * da
        + ac.Cndr * dr
    )
    return CL, CD, CY, Cl, Cm, Cn


def aero_forces_moments(
    vel_rel: Array,
    omega_rel: Array,
    controls: Controls,
    ac: Aircraft,
    rho: Array,
    a_sound: Array,
    increment=None,
) -> tuple[Array, Array]:
    """Body-axis aerodynamic force (N) and moment (N.m).

    `increment` is a `loads.CoeffIncrement`: an additive set of coefficients
    computed OUTSIDE this module, from the wind field across the airframe. This
    module still never sees the field itself -- it receives four numbers and
    adds them after the build-up, which is what preserves the rule that a
    zero-strength wind is bit-identical to still air.

    Deliberately unannotated. Annotating it would need `from flightsim.loads
    import CoeffIncrement`, and `loads` imports `air_data` from here, so a
    module-level import either way closes a cycle. The type is documented
    instead, which costs a checker and buys a one-directional dependency.
    """
    _, alpha, beta = air_data(vel_rel)
    # The fourth quantity that used to be floored, and the only one that put a
    # force on the airframe. qbar has no division by V and is finite at V = 0.
    # Written as norm(...)**2 rather than dot(...) so that above the floor it is
    # the identical expression to the one this replaced, to the last bit.
    qbar = 0.5 * rho * jnp.linalg.norm(vel_rel) ** 2
    CL, CD, CY, Cl, Cm, Cn = coefficients(vel_rel, omega_rel, controls, ac, a_sound)
    if increment is not None:
        CL = CL + increment.CL
        Cl = Cl + increment.Cl
        Cm = Cm + increment.Cm
        Cn = Cn + increment.Cn

    lift = qbar * ac.S * CL
    drag = qbar * ac.S * CD
    side = qbar * ac.S * CY

    # Lift and drag are wind-axis; rotate into body axes through alpha and beta.
    # Side force is already body-axis, as the lateral derivatives are defined.
    ca, sa = jnp.cos(alpha), jnp.sin(alpha)
    cb, sb = jnp.cos(beta), jnp.sin(beta)
    force = jnp.array(
        [
            -drag * ca * cb + lift * sa,
            -drag * sb + side,
            -drag * sa * cb - lift * ca,
        ]
    )

    moment = qbar * ac.S * jnp.array([ac.b * Cl, ac.c * Cm, ac.b * Cn])
    return force, moment


def thrust_force(
    controls: Controls, ac: Aircraft, rho: Array, mach: Array = 0.0
) -> Array:
    """Body-axis thrust, assumed aligned with the body x axis.

    Deliberately not an engine model: throttle times maximum thrust, with a
    density lapse and a ram term. The exponent is 1 for a normally-aspirated
    piston and around 0.7-0.8 for a high-bypass turbofan.

    `mach` is optional and defaults to zero, which switches the ram term off
    entirely, so callers written before it existed keep their exact behaviour.
    It is separate from `ac.mach_ram`: an aircraft with no ram coefficient is
    unaffected at any Mach, and a caller with no Mach is unaffected by any
    coefficient. Both have to be supplied for the term to act.
    """
    magnitude = (
        controls.throttle
        * ac.max_thrust
        * (rho / RHO0) ** ac.thrust_lapse
        * (1.0 + ac.mach_ram * mach**2)
    )
    return jnp.array([magnitude, 0.0, 0.0])
