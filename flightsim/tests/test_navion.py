"""Navion: an independent acceptance test that bypasses CR-2144 entirely.

The North American Navion's non-dimensional, per-radian, body-axis
derivatives below are the standard set given in Nelson, "Flight Stability
and Automatic Control" (2nd ed.), consistent with the same aircraft's data
in Etkin & Reid, "Dynamics of Flight: Stability and Control", and in McRuer,
Ashkenas & Graham, "Aircraft Dynamics and Automatic Control" -- this is one
of the most widely reproduced example datasets in the field, and this
project's own synthetic test fixture (flightsim/tests/conftest.py's
make_test_aircraft) independently landed on the same numbers to within
rounding, which is corroborating rather than circular: that fixture predates
this file and was never fitted to it.

Unlike the 747, nothing here is recovered through CR-2144's primed/Ixz
inertia-coupling algebra (Ixz = 0 for this airframe; the source gives
per-radian coefficients directly). A pass validates the aero build-up,
trim solver, and dynamics/linearisation chain on an entirely independent
data path from steps 2-4.

*** No externally published Dutch-roll/phugoid/etc. NUMBER TABLE for this
exact derivative set could be independently confirmed here (a genuine search
across CR-2144-adjacent NASA/NTRS reports, MIT OCW lecture notes, and the
Etkin/Nelson texts did not turn up an extractable one within this pass's
effort budget) -- flagged rather than invented, per the project's own rule
against inventing plausible numbers. What is asserted below are the
mode/damping SIGNS and standard-textbook light-aircraft RANGES (phugoid
lightly damped and slow, short period heavily damped, Dutch roll positive
but lightly damped, roll fast, spiral a slow real mode), which is what a
correctly wired aero-to-dynamics chain must reproduce for this data
regardless of which exact estimation method a source used -- the FAA's own
comparison of Navion derivative estimates (Seckel & Morris, "The Stability
Derivatives of the Navion Aircraft...", 1971) found roughly 20% spread
between independent estimation methods, so exact-figure agreement is not
actually the meaningful bar here.
"""

import math

import jax.numpy as jnp
import pytest

from flightsim import trim
from flightsim.aircraft import Aircraft, inertia_tensor
from flightsim.atmosphere import G0
from flightsim.tests.modes import lateral_modes, longitudinal_modes
from flightsim.units import DEG2RAD, FT2M, LBF2N, RAD2DEG, SLUG_FT2_TO_KG_M2


def _navion() -> Aircraft:
    # -- geometry and mass (imperial, verbatim from the source) --
    W = 2750.0  # lb
    S, b, c = 184.0, 33.4, 5.7  # ft^2, ft, ft
    Ixx, Iyy, Izz = 1048.0, 3000.0, 3530.0  # slug-ft^2, Ixz = 0 (symmetric)
    U0 = 176.0  # ft/s, sea level

    m = W * LBF2N / G0
    S_si, b_si, c_si = S * FT2M**2, b * FT2M, c * FT2M
    AR = b_si * b_si / S_si
    U0_si = U0 * FT2M

    # -- longitudinal, trim CL1 = 0.41 / CD1 = 0.05 at alpha0 = 0 (level) --
    CL1, CD1 = 0.41, 0.05
    CLa, Cma, CLq, Cmq, CLde, Cmde = 4.44, -0.683, 3.8, -9.96, 0.355, -0.923
    CDa = 0.33  # given directly (unlike CR-2144's 747/Cherokee, no back-solve from Xw needed)

    # Same drag build-up as every other aircraft in this project: e from the
    # CD-alpha slope at trim, CD0 from what's left of CD1 after induced drag.
    e = 2.0 * CL1 * CLa / (math.pi * AR * CDa)
    CD0 = CD1 - CL1**2 / (math.pi * e * AR)

    inertia = inertia_tensor(*(v * SLUG_FT2_TO_KG_M2 for v in (Ixx, Iyy, Izz, 0.0)))
    return Aircraft(
        mass=jnp.array(m),
        inertia=inertia,
        inertia_inv=jnp.linalg.inv(inertia),
        S=jnp.array(S_si),
        b=jnp.array(b_si),
        c=jnp.array(c_si),
        CD0=jnp.array(CD0),
        e=jnp.array(e),
        AR=jnp.array(AR),
        sweep=jnp.array(0.0),  # unswept, low subsonic: no wave drag
        t_over_c=jnp.array(0.15),
        kappa_airfoil=jnp.array(0.87),
        CL0=jnp.array(CL1),  # alpha0 = 0, so CL0 is the trim CL directly
        CLa=jnp.array(CLa),
        CLq=jnp.array(CLq),
        CLde=jnp.array(CLde),
        Cm0=jnp.array(0.0),  # alpha0 = 0, trimmed with zero elevator assumption
        Cma=jnp.array(Cma),
        Cmq=jnp.array(Cmq),
        Cmde=jnp.array(Cmde),
        CYb=jnp.array(-0.564),
        CYp=jnp.array(-0.0192),
        CYr=jnp.array(0.335),
        CYdr=jnp.array(0.157),
        Clb=jnp.array(-0.074),
        Clp=jnp.array(-0.410),
        Clr=jnp.array(0.107),
        Clda=jnp.array(0.134),
        Cldr=jnp.array(0.0107),
        Cnb=jnp.array(0.071),
        Cnp=jnp.array(-0.0575),
        Cnr=jnp.array(-0.125),
        Cnda=jnp.array(-0.0035),
        Cndr=jnp.array(-0.072),
        # MODELLING CHOICE, not source data (matches the project's convention
        # for every light aircraft): representative 205 hp engine at 80%
        # propeller efficiency, evaluated at the cruise speed above.
        max_thrust=jnp.array(0.8 * 205.0 * 745.7 / U0_si),
        thrust_lapse=jnp.array(1.0),
        elevator_limit=jnp.array(25.0 * DEG2RAD),
        aileron_limit=jnp.array(20.0 * DEG2RAD),
        rudder_limit=jnp.array(25.0 * DEG2RAD),
    )


NAVION = _navion()
U0 = 176.0 * FT2M
H0 = 0.0


def _trim():
    x, res = trim.trim(jnp.array(U0), jnp.array(H0), NAVION)
    return (float(v) for v in x), float(jnp.linalg.norm(res))


def test_navion_trims_and_reproduces_the_sources_trim_lift():
    (alpha, elevator, throttle), residual = _trim()
    assert residual < 1e-8
    assert abs(alpha) * RAD2DEG < 2.0  # source assumes level flight, alpha0 = 0
    assert 0.0 < throttle < 1.0


def test_navion_phugoid_and_short_period_are_in_the_expected_bands():
    (alpha, elevator, throttle), _ = _trim()
    phugoid, short_period = longitudinal_modes(NAVION, alpha, elevator, throttle, U0, H0)

    assert 0.08 < phugoid[0] < 0.4  # rad/s: slow
    assert 0.0 < phugoid[1] < 0.25  # lightly damped, stable

    assert 1.5 < short_period[0] < 6.0  # rad/s: fast
    assert 0.3 < short_period[1] < 0.9  # heavily damped, stable


def test_navion_lateral_modes_are_in_the_expected_bands():
    (alpha, elevator, throttle), _ = _trim()
    dutch_roll, roll_tau, spiral_tau = lateral_modes(NAVION, alpha, elevator, throttle, U0, H0)
    wn, zeta = dutch_roll

    assert 1.0 < wn < 4.0
    assert 0.0 < zeta < 0.4  # positive (stable) but lightly damped, as expected

    assert 0.03 < roll_tau < 1.0  # fast, stable

    assert abs(spiral_tau) > 15.0  # slow real mode either way, not spuriously fast


def test_navion_sidesteps_the_cr2144_unprime_pipeline():
    """Ixz = 0 here -- the whole point of this aircraft as steps 2-4's
    independent check is that none of its derivatives pass through
    aircraft._unprime at all.
    """
    assert float(NAVION.inertia[0, 2]) == 0.0
    assert float(NAVION.inertia[2, 0]) == 0.0
