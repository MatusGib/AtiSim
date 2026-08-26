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

from atisim.aircraft import Aircraft
from atisim.atmosphere import RHO0
from atisim.state import Controls

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

# Prandtl-Glauert is a SUBSONIC LINEARISED result and diverges at M = 1, where
# the physics it comes from has already stopped applying. These two numbers stop
# that divergence reaching the integrator, and neither is a modelling claim.
#
# The cap is placed at M 0.90 because it is above every condition this project
# flies -- the fastest is the 747's M 0.80 -- so it is a GUARD that never
# actually binds rather than a transonic model, which this project does not have
# and must not pretend to. If a run ever reaches it, the aircraft is outside the
# band `checks.recovery_band` already gates on.
#
# The floor keeps the denominator away from zero so `jacfwd` sees a finite
# derivative even on a trajectory that a solver briefly probes past the cap.
PG_MACH_MAX = 0.90
PG_FLOOR = 1.0 - PG_MACH_MAX**2  # 0.19, the smallest 1 - M^2 the factor can see


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
    alphadot_gust: Array = 0.0,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """(CL, CD, CY, Cl, Cm, Cn). Lift and drag wind-axis, the rest body-axis.

    `alphadot_gust` is an angle-of-attack rate in rad/s. The name records where
    the term came from, not what reaches it: dynamics.derivatives passes the
    wind's contribution PLUS the aircraft's own, which it resolves in a pass.
    It defaults to zero, so every caller written before it existed is unchanged
    and still air performs no extra arithmetic. See the Aircraft fields CLadot
    and Cmadot, which explain why an aircraft using them needs a bare Cmq.

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

    # Same non-dimensionalisation as q_hat: Stengel Eq. (3.4-25), (3.4-26).
    alphadot_hat = alphadot_gust * ac.c / (2.0 * V)

    # Lift from incidence: a table if the aircraft carries one, the linear form
    # otherwise. `.size` is a property of the SHAPE, so this branch is resolved
    # at trace time and costs nothing under jit. The other CL terms stay
    # additive either way, exactly as JSBSim keeps CLalpha and CLde separate.
    # True-airspeed Mach, shared by wave drag, the scheduled control
    # derivatives and the Prandtl-Glauert factor. Built from the unfloored
    # airspeed for the same reason wave drag is: Mach is finite at V = 0.
    mach = jnp.linalg.norm(vel_rel) / a_sound

    # Prandtl-Glauert, RELATIVE to the Mach the entry's data was taken at, so an
    # aircraft flown at its own reference condition is unchanged. See the
    # `pg_mach_ref` field. Undeclared (negative) leaves the factor at exactly 1.
    #
    # Applied to the WHOLE longitudinal lift-slope family rather than to CLa
    # alone. That is not tidiness: PROJECT.md section 7 records CLa(M) applied
    # BY ITSELF making the 747's phugoid WORSE (17.8% -> 19.4%), which is the
    # signature of a partial correction. Every derivative below is proportional
    # to the same two-dimensional section lift slope, so thin-aerofoil theory
    # scales them together or not at all.
    #
    # Lateral derivatives are deliberately NOT scaled. Clb, Clp, Cnb and the
    # rest mix section lift slope with dihedral, fin geometry and sidewash in
    # proportions this project has no source for, so scaling them would be
    # inventing a correction rather than applying one.
    pg = jnp.where(
        ac.pg_mach_ref < 0.0,
        1.0,
        jnp.sqrt(jnp.maximum(1.0 - jnp.minimum(ac.pg_mach_ref, PG_MACH_MAX) ** 2, 0.0))
        / jnp.sqrt(jnp.maximum(1.0 - jnp.minimum(mach, PG_MACH_MAX) ** 2, PG_FLOOR)),
    )

    if ac.CL_table_alpha.size:
        CL_alpha_part = jnp.interp(alpha, ac.CL_table_alpha, ac.CL_table_CL)
    else:
        CL_alpha_part = ac.CL0 + ac.CLa * alpha
    # `CL0` is an intercept, not a slope, so only the alpha-dependent part is
    # scaled. For a table entry the intercept is the table's own value at
    # alpha = 0 and the rest of the curve stretches about it -- the same
    # construction scripts/vortex_diagnose.py uses for its CLa sweep, and the
    # only one that is a pure lift-SLOPE change.
    #
    # Written as `x + (pg - 1)(x - CL0)` rather than the algebraically identical
    # `CL0 + pg(x - CL0)`, because the two differ IN FLOATING POINT at pg = 1:
    # the second round-trips through CL0 and back, which is not the identity,
    # while the first multiplies an exact zero. Every aircraft that has not
    # declared a reference Mach must be bit-for-bit unmoved by this whole block,
    # and test_aero asserts it is.
    if ac.CL_table_alpha.size:
        CL0 = jnp.interp(jnp.array(0.0), ac.CL_table_alpha, ac.CL_table_CL)
    else:
        CL0 = ac.CL0
    CL_alpha_part = CL_alpha_part + (pg - 1.0) * (CL_alpha_part - CL0)

    # Scheduled control derivatives. `.size` is a shape property, so these
    # branches resolve at trace time and an aircraft without a table performs no
    # extra arithmetic. `jnp.interp` clamps outside its range, which is exactly
    # what JSBSim's own <table> blocks do.
    if ac.Cmde_table_mach.size:
        Cmde = jnp.interp(mach, ac.Cmde_table_mach, ac.Cmde_table_Cmde)
    else:
        Cmde = ac.Cmde

    # Scale the COEFFICIENTS, then sum in the original order. Grouping the terms
    # instead -- `Cm0 + pg*(a + b + c + d)` -- is algebraically identical and
    # NOT bit-identical: floating-point addition is not associative, so
    # re-associating a five-term sum moves the last bits even at pg = 1.0. That
    # is enough to break `test_extracting_rk4_step_did_not_move_a_single_bit`
    # and the Fig. 8 pin, both of which assert exact equality on purpose.
    # At pg = 1.0, `1.0 * x` is exactly `x`, so this form leaves every aircraft
    # without a declared reference Mach bit-for-bit unmoved.
    CLq, CLde, CLadot = pg * ac.CLq, pg * ac.CLde, pg * ac.CLadot
    Cma, Cmq, Cmadot = pg * ac.Cma, pg * ac.Cmq, pg * ac.Cmadot
    Cmde = pg * Cmde

    CL = (CL_alpha_part + CLq * q_hat + CLde * de
          + CLadot * alphadot_hat)
    Cm = (ac.Cm0 + Cma * alpha + Cmq * q_hat + Cmde * de
          + Cmadot * alphadot_hat)
    # Parabolic core plus a lift-dependent compressibility rise.
    # Mach divides by the speed of sound, not by V, so it takes the true
    # airspeed. Below the floor this is unobservable either way -- M_crit is
    # above 0.5 for every aircraft here and `wave_drag` is identically zero at
    # walking pace -- but the floor has no defence here and does not belong.
    CD = (
        ac.CD0
        + CL**2 / (jnp.pi * ac.e * ac.AR)
        + wave_drag(jnp.linalg.norm(vel_rel) / a_sound, CL, ac)
        # Sideslip drag. QUADRATIC, and deliberately so: drag is an EVEN
        # function of sideslip for a laterally symmetric airframe, so dCD/dbeta
        # is identically zero at beta = 0 and the leading term is second order.
        #
        # That evenness is also why linear small-perturbation theory has no such
        # derivative at all, and why no formula for it appears in this project's
        # theory backbone -- adding it is a deliberate step outside the linear
        # framework, not the filling of a gap in it. See the design spec's
        # section 1, which records that the reference search came up empty and
        # justifies the form by symmetry rather than by citation.
        #
        # JSBSim's 737 carries the same physics as a five-point table
        # interpolated linearly through zero, which makes CD proportional to
        # |beta| near the origin -- a kink, with a discontinuous slope at
        # beta = 0. Measured: 1.007e-2 at 3 deg and 5.035e-3 at 1.5 deg, exactly
        # half rather than a quarter. That is a coarse-table artifact and is NOT
        # reproduced here, which is the one place this work declines to follow
        # JSBSim.
        + ac.CD_beta * beta**2
        # Profile drag with incidence, linearised about the reference
        # condition; CD0 carries the value at alpha = 0. See the field.
        + ac.CD_alpha * alpha
    )

    if ac.Clda_table_mach.size:
        Clda = jnp.interp(mach, ac.Clda_table_mach, ac.Clda_table_Clda)
    else:
        Clda = ac.Clda

    CY = ac.CYb * beta + ac.CYp * p_hat + ac.CYr * r_hat + ac.CYdr * dr
    Cl = (
        ac.Clb * beta
        + ac.Clp * p_hat
        + ac.Clr * r_hat
        + Clda * da
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
    alphadot_gust: Array = 0.0,
) -> tuple[Array, Array]:
    """Body-axis aerodynamic force (N) and moment (N.m).

    `increment` is a `loads.CoeffIncrement`: an additive set of coefficients
    computed OUTSIDE this module, from the wind field across the airframe. This
    module still never sees the field itself -- it receives four numbers and
    adds them after the build-up, which is what preserves the rule that a
    zero-strength wind is bit-identical to still air.

    Deliberately unannotated. Annotating it would need `from atisim.loads
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
    CL, CD, CY, Cl, Cm, Cn = coefficients(
        vel_rel, omega_rel, controls, ac, a_sound, alphadot_gust
    )
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

    # About ac.aero_ref, which is the CG for every aircraft that leaves the
    # field at its default zero.
    moment = qbar * ac.S * jnp.array([ac.b * Cl, ac.c * Cm, ac.b * Cn])
    # Transfer to the CG: Stengel Eq. (2.4-68). The cross product is with the
    # AERODYNAMIC force only -- thrust is added downstream in dynamics.py and
    # carries its own line of action, which this model still places at the CG.
    #
    # jnp.cross of a zero vector is exact zero, so the default path adds an
    # exact zero rather than a rounded one, and every pre-existing aircraft is
    # bit-identical. Asserted in test_aerodynamic_reference_point.py.
    moment = moment + jnp.cross(ac.aero_ref, force)
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
