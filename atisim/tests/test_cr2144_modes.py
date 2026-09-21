"""747 lateral and longitudinal modes against CR-2144's own tabulated values.

Table IX-5 (elevator transfer function, bare airframe) already gives the
longitudinal characteristic equation for FC9 and is checked by
scripts/checkpoint.py and test_trim.py. This module adds:

  * Tables IX-9/IX-10 (aileron/rudder transfer function, SAS off, bare
    airframe) tabulate the SAME lateral-directional characteristic equation --
    the direct lateral analogue of Table IX-5, and the thing Step 2 needed to
    find out existed at all.
  * Table IX-4's speed derivatives (Xu, Zu, Mu) and alpha-dot derivatives
    (Zwdot, Mwdot), which atisim's alpha/q/de-only aero form deliberately
    excludes, added back into a second, standalone linear model used only
    here for mode extraction -- never into the sim itself. (Session 30: the
    SPEED family's Mach content now is in the sim for `boeing747`, sourced from
    CR-2144 printed p. 222; the alpha-dot pair is still only here.)
"""

import jax.numpy as jnp
import numpy as np
import pytest

from atisim import trim
from atisim.aircraft import CRUISE, REGISTRY, boeing747_without_thrust_line
from atisim.tests.modes import lateral_modes, longitudinal_modes
from atisim.units import DEG2RAD

NAME = "boeing747"
AC = REGISTRY[NAME]
V = CRUISE[NAME]["airspeed"]
H = CRUISE[NAME]["altitude"]


def _trim():
    x, _ = trim.trim(jnp.array(V), jnp.array(H), AC)
    return (float(v) for v in x)


# --- Step 2: lateral-directional, against Table IX-9 / IX-10 -----------------
#
# FC9 denominator (identical in both tables -- aileron- and rudder-input
# transfer functions share one characteristic equation, which is itself a
# check on the source): 1/T(DET)1 = .00730 (spiral), 1/T(DET)2 = .562 (roll),
# Z(DET)1 = .0349 (Dutch roll zeta), W(DET)1 = .947 rad/s (Dutch roll wn).
CR2144_DUTCH_ROLL_WN = 0.947
CR2144_DUTCH_ROLL_ZETA = 0.0349
CR2144_ROLL_TAU = 1.0 / 0.562
CR2144_SPIRAL_TAU = 1.0 / 0.00730


def test_lateral_modes_match_cr2144_tables_ix9_and_ix10():
    alpha, elevator, throttle = _trim()
    dutch_roll, roll_tau, spiral_tau = lateral_modes(AC, alpha, elevator, throttle, V, H)
    wn, zeta = dutch_roll

    assert wn == pytest.approx(CR2144_DUTCH_ROLL_WN, rel=0.02)
    assert zeta == pytest.approx(CR2144_DUTCH_ROLL_ZETA, rel=0.1)
    assert roll_tau == pytest.approx(CR2144_ROLL_TAU, rel=0.05)
    # Spiral is the classic ill-conditioned mode: switching this model's
    # thetadot kinematics from the exact r*cos(phi)*tan(theta0) term to the
    # common p-only approximation moves this figure from ~138s to ~103s (a
    # 34% swing) while barely touching wn/zeta/roll above -- a formulation
    # choice, not a derivative error. A direct nonlinear check (perturb the
    # trim in bank with no control input and fit the late-time decay of phi
    # over 500s) independently gives 138.0s, agreeing with this linearisation
    # to <0.01%, so this is what the real integrator actually does.
    assert spiral_tau == pytest.approx(CR2144_SPIRAL_TAU, rel=0.02)
    assert spiral_tau > 0  # stable, as CR-2144 states


# --- Step 3: longitudinal, augmented model against Table IX-5 ----------------
#
# Table IX-4, FC9 (40,000 ft, M 0.80), all as tabulated (imperial, concise
# derivatives) -- the full set, including the ones aircraft.py's alpha/q/de
# form deliberately leaves out.
_ALPHA0_DEG = 4.60  # Table IX-3's own tabulated trim alpha for FC9
_U0 = 774.0  # ft/s
_XU, _ZU, _MU = -0.00276, -0.0650, 0.000193
_XW, _ZW, _MW = 0.0389, -0.317, -0.00105
_ZQ, _MQ = -5.16, -0.339
_ZWD, _MWD = 0.00556, -0.000116
_G_FTS2 = 32.174

CR2144_PHUGOID = (0.0673, 0.0489)
CR2144_SHORT_PERIOD = (0.964, 0.387)


def _augmented_longitudinal_modes():
    """Classical [u, w, q, theta] state-space model, entirely in the imperial
    units Table IX-4 natively uses (mixing them with the sim's SI trim would
    silently misscale the Xu/Zu/Mu terms).

    This is deliberately NOT built from atisim.dynamics: it exists only to
    show what CR-2144's own derivative set implies via the textbook
    small-perturbation equations, side by side with the sim's own exact
    (jax.jacfwd) linearisation.
    """
    theta0 = _ALPHA0_DEG * DEG2RAD
    w0 = _U0 * np.sin(theta0)
    denom = 1.0 - _ZWD

    A = np.zeros((4, 4))
    A[0, :] = [_XU, _XW, -w0, -_G_FTS2 * np.cos(theta0)]
    A[1, :] = np.array([_ZU, _ZW, _ZQ + _U0, -_G_FTS2 * np.sin(theta0)]) / denom
    A[2, :] = np.array([_MU, _MW, _MQ, 0.0]) + _MWD * A[1, :]
    A[3, :] = [0.0, 0.0, 1.0, 0.0]

    eig = np.linalg.eigvals(A)
    modes = [(abs(lam), -lam.real / abs(lam)) for lam in eig if lam.imag > 1e-9]
    return sorted(modes)


def test_augmented_longitudinal_model_closes_the_phugoid_and_short_period_gap():
    """Table IX-4's excluded Xu/Zu/Mu/Zwdot/Mwdot, added back in a model built
    only for this test, should reproduce CR-2144's own Table IX-5 closely --
    confirming their exclusion from the sim (not a bug in what remains) is
    the fully attributed explanation for the sim's own phugoid/short-period
    gap asserted below.
    """
    phugoid, short_period = _augmented_longitudinal_modes()

    assert phugoid[0] == pytest.approx(CR2144_PHUGOID[0], rel=0.03)
    assert phugoid[1] == pytest.approx(CR2144_PHUGOID[1], rel=0.03)
    assert short_period[0] == pytest.approx(CR2144_SHORT_PERIOD[0], rel=0.03)
    assert short_period[1] == pytest.approx(CR2144_SHORT_PERIOD[1], rel=0.03)


def test_the_sims_own_unaugmented_longitudinal_modes_are_the_documented_gap():
    """Locks in today's measured un-augmented figures as a regression guard.

    This is the sim exactly as shipped (alpha/q/de aero only, no Xu/Zu/Mu/
    ~~Zwdot/Mwdot~~ Zwdot): phugoid wn 0.0554 rad/s (18% low vs CR-2144's
    0.0673) and short-period zeta ~~0.338 (13% low vs CR-2144's 0.387)~~
    0.3895. See the augmented comparison above for why the phugoid is low:
    that half is attributed, not a bug.

    THE SHORT-PERIOD HALF IS NO LONGER A GAP. `boeing747` declares Table IX-4's
    Mwdot as `Cmadot`, so the family this figure was pinning the ABSENCE of is
    half restored, and the damping it guarded is gone by design. Zwdot is still
    out, and deliberately -- its tabulated sign gives a negative CL_alphadot
    (aircraft.py says why) -- so the phugoid assertions below are untouched.
    """
    # SESSION 30: THE WORLD CHANGED, NOT THE TOLERANCE. `boeing747` now declares
    # CR-2144's speed derivatives (PROJECT.md section 4), so the shipped entry is
    # no longer the alpha/q/de-only form this test documents: its phugoid is +4%
    # against Table IX-5, not -18%. The figures below are that form's, measured
    # on the same entry with the Mach seam shut, and every band is as it was.
    bare = boeing747_without_thrust_line()._replace(mach_deriv_ref=jnp.array(-1.0))
    x, _ = trim.trim(jnp.array(V), jnp.array(H), bare)
    alpha, elevator, throttle = (float(v) for v in x)
    phugoid, short_period = longitudinal_modes(bare, alpha, elevator, throttle, V, H)

    assert phugoid[0] == pytest.approx(0.0554, rel=0.05)
    assert phugoid[1] == pytest.approx(0.0559, rel=0.1)
    assert short_period[0] == pytest.approx(0.949, rel=0.03)
    # RE-PINNED with the Cmadot declaration: 0.338 -> 0.3895, measured on this
    # same bare entry. THE WORLD CHANGED, NOT THE TOLERANCE -- `rel` is still
    # 0.05 and only the pinned value moved. Isolating the declaration on this
    # entry gives 0.3431 without it and 0.3895 with it, and the three
    # assertions above did not move at all, which is what says the change is
    # confined to the axis it was made on. CR-2144 Table IX-5 says 0.387.
    assert short_period[1] == pytest.approx(0.3895, rel=0.05)
