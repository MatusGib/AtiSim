"""Where on the airframe the wind field is sampled.

The model has always evaluated the wind at one point -- the CG -- and taken the
analytic gradient there. That is exact while the field is linear across the
aircraft and progressively wrong as the field's scale approaches a wingspan,
which is the regime this project is moving into. This module supplies the
geometry needed to sample across the airframe instead.

It needs an extent in each direction. Laterally that is the span, which is
tabulated. Longitudinally it is a tail arm, which is NOT tabulated for any
aircraft this project holds -- so it is recovered from two derivatives that
are.

Provenance for every constant here is in flightsim/provenance.py, and a test
asserts the two agree.
"""

import jax.numpy as jnp
from jax import Array

from flightsim.aircraft import Aircraft

# Plausibility band on the derived tail arm, in mean chords. DECLARED, not
# sourced -- see provenance.LEDGER["airframe.tail_arm_band"]. It brackets
# conventional tail-aft configurations and exists only to reject derivative
# sets whose CLq and Cmq disagree about what aircraft they describe.
TAIL_ARM_BAND = (2.0, 6.0)


def effective_tail_arm(ac: Aircraft) -> Array:
    """Distance from the CG to the effective tail centre of pressure, in chords.

    Stengel, Flight Dynamics 2nd ed. From eq. 3.4-7 the tail's lift response to
    pitch rate is CL_q,ht = CL_a,ht * (l_ht / V); non-dimensionalised with
    q_hat = q*c/2V (eq. 3.4-10) that is CL_qhat = 2 * CL_a,ht * (l_ht/c). From
    eq. 3.4-12 the moment derivative is Cm_qhat = -2 * (l_ht/c)^2 * CL_a,ht.
    The ratio eliminates the tail lift slope, which the project does not hold:

        l_eff / c = -Cm_qhat / CL_qhat

    ATTRIBUTES BOTH DERIVATIVES TO THE TAIL. Stengel eq. 3.4-19 notes Cmq also
    carries wing and fuselage contributions. The obvious correction --
    subtracting the wing's share via eqs. 3.4-13/3.4-14 with the sourced
    h_cm = 0.25 -- was tried and REJECTED: it makes the wing 83% of CLq, which
    implies a 23-chord tail arm. Those are Etkin's two-dimensional
    infinite-aspect-ratio results and they do not transfer to a swept wing with
    a tail. Do not re-attempt it; see design section 3d.
    """
    return -ac.Cmq / ac.CLq


def tail_arm_is_plausible(ac: Aircraft) -> bool:
    """Whether this aircraft's derivative set is self-consistent enough to sample.

    A Python bool, not a traced array: it is a data-quality gate evaluated once
    when a run is set up, never inside a jitted step.
    """
    low, high = TAIL_ARM_BAND
    return bool(low <= float(effective_tail_arm(ac)) <= high)
