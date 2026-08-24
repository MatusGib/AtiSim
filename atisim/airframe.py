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

Provenance for every constant here is in atisim/provenance.py, and a test
asserts the two agree.
"""

import contextlib
from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from atisim.aircraft import Aircraft

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


# Sample counts. DECLARED -- see provenance.LEDGER["strip.n_stations"]. Odd, so
# a station sits exactly on the centreline and the symmetric pair cancels
# exactly rather than to round-off.
N_SPAN = 9
N_LON = 9


class Stations(NamedTuple):
    """Body-axis offsets from the CG at which the wind field is evaluated.

    Two one-dimensional sets rather than one cloud of points, because the three
    gradients the aero model consumes are each a slope along a single axis:
    roll from vertical gust varying across the span, pitch and yaw from gusts
    varying along the fuselage. Sampling a full grid would cost N^2 field
    evaluations to produce the same three numbers.
    """

    span: Array  # (N,) m, body y, positive right
    longitudinal: Array  # (M,) m, body x, positive forward


def stations(ac: Aircraft, n_span: int = N_SPAN, n_lon: int = N_LON) -> Stations:
    """Sample stations for an aircraft.

    Lateral extent is the span, which is SOURCED. Longitudinal extent is the
    derived tail arm, running aft from the CG -- negative x, since body x is
    positive forward.
    """
    half_span = ac.b / 2.0
    arm = effective_tail_arm(ac) * ac.c
    return Stations(
        span=jnp.linspace(-half_span, half_span, n_span),
        longitudinal=jnp.linspace(-arm, 0.0, n_lon),
    )


# ---------------------------------------------------------------------------
# Spanwise loading
#
# The strip integration needs to know how lift is distributed across the span.
# That distribution is NOT available from any source this project holds: taper
# ratio is not tabulated in CR-2144, is not in the Boeing simulation data the
# report itself cites, and is not recoverable from S, b and cbar -- the shape
# factor those three demand is 0.72873 against a trapezoidal minimum of 0.75,
# because the 747 planform is cranked. See the design document, section 3f.
#
# So the shape is DECLARED and the magnitude is CALIBRATED to a sourced number.
# The three shapes below all enclose the sourced wing area, so the spread
# between them measures the shape assumption and nothing else.
# ---------------------------------------------------------------------------

LOADING_SHAPES = ("elliptic", "uniform", "tapered")

# Taper ratio used ONLY by the `tapered` sensitivity alternative. It is NOT the
# 747's taper ratio -- that is not tabulated and not recoverable -- it is a
# representative transport value whose only job is to give the sweep a third,
# differently-shaped member. No result may quote it as 747 geometry.
_SENSITIVITY_TAPER = 0.3

_active_shape = "elliptic"


@contextlib.contextmanager
def loading_shape(name: str):
    """Temporarily select a spanwise loading shape, for the sensitivity sweep.

    A context manager over module state rather than a parameter threaded through
    every call, because the shape is a project-level declaration that must be the
    same everywhere within one run. Threading it would invite two call sites
    disagreeing, which is exactly the failure the ledger exists to prevent.
    """
    global _active_shape
    if name not in LOADING_SHAPES:
        raise ValueError(f"unknown loading shape {name!r}; expected one of {LOADING_SHAPES}")
    previous = _active_shape
    _active_shape = name
    try:
        yield
    finally:
        _active_shape = previous


def elliptic_chord(y: Array, ac: Aircraft) -> Array:
    """Elliptic spanwise chord distribution, scaled to the sourced wing area.

        c(y) = c0 * sqrt(1 - (2y/b)^2),    c0 = 4S/(pi*b)

    Elliptic is the default because it needs no taper ratio at all, so the shape
    introduces exactly one assumption rather than one assumption plus an
    unsourced number.

    The argument of the square root is clamped: `stations` places points exactly
    at +-b/2 where it is analytically zero, and round-off can make it slightly
    negative, which conftest's jax_debug_nans would trip on.
    """
    c0 = 4.0 * ac.S / (jnp.pi * ac.b)
    normalised = 2.0 * y / ac.b
    return c0 * jnp.sqrt(jnp.maximum(1.0 - normalised * normalised, 0.0))


def chord_distribution(y: Array, ac: Aircraft, name: str | None = None) -> Array:
    """Spanwise chord for the named shape, scaled to the sourced wing area."""
    name = _active_shape if name is None else name
    if name == "elliptic":
        return elliptic_chord(y, ac)
    if name == "uniform":
        return jnp.full_like(y, ac.S / ac.b)
    if name == "tapered":
        # Straight taper: c(y) = c_root * (1 - (1-L)*|2y/b|). The area of that
        # shape is b*c_root*(1+L)/2, so c_root = 2S/(b*(1+L)) encloses S.
        lam = _SENSITIVITY_TAPER
        c_root = 2.0 * ac.S / (ac.b * (1.0 + lam))
        return c_root * (1.0 - (1.0 - lam) * jnp.abs(2.0 * y / ac.b))
    raise ValueError(f"unknown loading shape {name!r}")


def calibrated_lift_slope(ac: Aircraft) -> Array:
    """Effective section lift slope, pinned so the strip integral returns Clp.

    Strip theory gives the rolling moment from a roll rate p as

        L = -integral( y * qbar * c(y) * a0 * (p*y/V) dy )

    since a station at y moves down at p*y and therefore sees an incidence
    increment p*y/V (Stengel Flight Dynamics 2nd ed eq. 3.4-39), and the
    resulting lift acts at moment arm y. Non-dimensionalising with
    p_hat = pb/2V:

        Clp_hat = -(2*a0 / (S*b^2)) * integral( y^2 * c(y) dy )

    For the elliptic distribution, integral(y^2 c dy) = c0*b^3*pi/64 with
    c0 = 4S/(pi*b), which collapses to

        Clp_hat = -a0/8      hence      a0 = -8 * Clp

    CALIBRATED, not sourced. This a0 is an EFFECTIVE value: it absorbs sweep,
    the tail's share of Clp, and the difference between elliptic strip theory
    and the real cranked wing. It is not an airfoil property and must never be
    quoted as one. For the 747 it comes out at 2.80 against a thin-airfoil
    6.28 -- low precisely BECAUSE it is effective. A value near 6.28 would mean
    the calibration had not absorbed those effects and would be the surprising
    outcome.

    WHAT IT GUARANTEES, AND AT WHICH STATION COUNT. `Clp_hat = -a0/8` is the
    CONTINUUM integral, so a rigid roll rate through the strip sum reproduces the
    tabulated Clp exactly only in the limit. This docstring used to say "exactly"
    without that qualification. At the shipped N_SPAN = 9 the trapezoidal sum
    returns 82.6% of it -- a -17.4% understatement -- because the elliptic chord
    has a sqrt singularity at the tips that the rule cannot resolve. Measured
    convergence order is 1.50, stable across every refinement tried, so it takes
    ~19 stations for 5% and ~56 for 1%. Both the count and this calibration basis
    were deliberately left as they are; see docs/ASSUMPTIONS.md F5 for why, and
    provenance.LEDGER["strip.n_stations"].

    What it does not guarantee at any station count is that the spanwise SHAPE is
    right, which is why the sensitivity sweep is mandatory.
    """
    return -8.0 * ac.Clp
