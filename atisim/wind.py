"""Wind and turbulence.

Today this returns zero wind. It exists now so that the interfaces around it are
already the right shape when Dryden or von Karman turbulence arrives.

Three things a turbulence model needs, all present here already:

  wind_ned    translational gust velocity, subtracted from inertial velocity
              before any angle of attack is computed
  omega_gust  angular-rate perturbation (p_g, q_g, r_g); a gust gradient across
              the span and chord is a rate, not just a velocity
  wind_state  the shaping filters that turn white noise into Dryden/von Karman
              spectra are dynamic systems and carry state between steps
  alphadot_gust  the WIND-INDUCED angle-of-attack rate, rad/s. A gust changes
              alpha without any pitch rate, and the tail's downwash lag makes
              that a pitching moment (Stengel Eq. 3.4-26). Computed from the
              same field Jacobian that produces omega_gust

`wind_state` is not in the original plan's signature. It is here because a
shaped-noise turbulence model cannot work without somewhere to keep its filter
states, and adding the slot later would mean changing the signature of `step`
-- exactly the retrofit these hooks exist to avoid. It costs an empty tuple
today.

`state` is passed in because Dryden scale lengths and intensities are functions
of altitude, and the filter time constants are functions of true airspeed.
"""

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from atisim import airframe
from atisim.aero import V_MIN
from atisim.aircraft import Aircraft
from atisim.state import State, quat_to_dcm
from atisim.units import FT2M


class WindState(NamedTuple):
    """Empty until a turbulence model needs filter states.

    Deterministic fields -- the vortex array below, and the gust/wave/updraft
    models that will join it -- are pure functions of position and need no
    state at all, so this stays empty for them. It gains fields when a shaped
    -noise model (Dryden) arrives, and `integrate.batch_sim` broadcasts every
    leaf, so every field added here must be a `jnp` array: a Python scalar has
    no `.shape` and raises there.
    """


def zero_wind_state() -> WindState:
    return WindState()


def zero_wind(
    wind_state: WindState, state: State, key: Array, dt: float
) -> tuple[Array, Array, WindState, Array, Array]:
    """Still air. (wind_ned, omega_gust, wind_state, key, alphadot_gust)."""
    del state, dt
    return jnp.zeros(3), jnp.zeros(3), wind_state, key, jnp.array(0.0)


# ---------------------------------------------------------------------------
# Kelvin-Helmholtz vortex array
#
# Source: E. K. Parks, R. C. Wingrove, R. E. Bach, R. S. Mehta, "Identification
# of Vortex-Induced Clear Air Turbulence Using Airline Flight Records",
# J. Aircraft 22(2), Feb 1985, pp. 124-129. Section "Vortex Modeling", Eqs.
# (3)-(6).
#
# The model is a Rankine vortex: "a rotational (solid-body) core embedded in an
# irrotational flow", axis horizontal and perpendicular to the wind vector.
# With l the along-track separation, d the vertical separation (aircraft above
# core), dpsi the angle between the wind vector and the flightpath, and
# r = (l^2 cos^2 dpsi + d^2)^(1/2):
#
#   outside (r >= r0):  w_xy = V0 r0 d / r^2      w_z = -V0 r0 l cos(dpsi) / r^2
#   inside  (r <  r0):  w_xy = V0 d / r0          w_z = -V0 l cos(dpsi) / r0
#
# The two forms agree at r = r0, so the field is continuous. Arrays are built by
# LINEAR SUPERPOSITION, which the source states explicitly and which is what
# makes summing this with other components (wave, updraft, Dryden) legitimate.
#
# This module DEFAULTS to dpsi = 0 -- vortex axes perpendicular to the track --
# and `VortexArray.cos_dpsi` carries the oblique case. It used to refuse the
# oblique case outright, on the grounds that no source held here constrained it.
#
# *** THAT CHANGED IN SESSION 23, AND ONLY BECAUSE A SOURCE ARRIVED. ***
# Mehta 1987 (see MEHTA_HANNIBAL_1987) prints the same Eqs. (1)-(4) AND an
# identified case flown at psi = 31 deg, so cos(dpsi) is now transcribed from a
# document rather than invented. The default is 1.0 and `1.0 * x` is exact in
# IEEE-754, so every baseline frozen before this change is byte-for-byte intact
# -- asserted in test_wind.py rather than assumed.
#
# WHY IT COULD NOT BE FAKED BY MOVING THE CORES. An oblique traverse compresses
# the along-track coordinate by cos(dpsi), so placing cores at x*cos(dpsi) with
# dpsi = 0 reproduces the wind FIELD exactly -- and then gets the TIME base
# wrong by 1/cos(dpsi), because the aircraft covers that compressed geometry at
# the same airspeed. At psi = 31 deg that is a 17% error in traverse time, and
# traverse time divided by the short period is precisely the quantity
# TM-102186 Fig. 8 says governs the response. So the shortcut breaks the one
# thing the Mehta case exists to test.
# ---------------------------------------------------------------------------


class VortexArray(NamedTuple):
    """A row of co-rotating Rankine vortices, all with the same core.

    Parks identifies exactly this: an array on the downslope of a standing
    wave, rotating in one sense. `north`/`down` are the NED coordinates of each
    core, so N is fixed by their shape -- changing the count recompiles, which
    is correct, and vmapping over encounter geometry batches these leaves.

    Identified values, both DC-10s near the tropopause:
      Case 1, Hannibal MO, 37,000 ft: r0 = 500 ft, V0 = 85 ft/s, spacing 3500 ft
      Case 2, Morton WY,   39,000 ft: r0 = 450 ft, V0 = 70 ft/s, spacing 3200 ft

    Hannibal's radius is Wingrove & Bach 1994 Fig. 4's, NOT Parks'; the strength
    and both spacings are Parks pp. 127-128. See PARKS_CASES for why.
    """

    north: Array  # (N,) m, NED north of each core
    down: Array  # (N,) m, NED down of each core
    r0: Array  # m, solid-body core radius
    v0: Array  # m/s, tangential velocity at the core edge
    # cos(dpsi), dpsi being the angle between the wind vector and the
    # flightpath. 1.0 is the perpendicular traverse every case flown before
    # session 23 used, and is exact, so it is the default. Trailing and
    # defaulted so the ~50 existing keyword constructions are untouched.
    cos_dpsi: Array = 1.0
    # The OTHER half of the same angle, needed only by `line_vortex_wind` to
    # build the vortex line's direction in three dimensions. `vortex_wind` never
    # reads it: on the flight path the sine drops out of Parks' expression
    # entirely, which is why the field could be written without it for so long.
    # 0.0 is due east -- perpendicular to a northerly path, and the geometry
    # every case before session 24 flew. Trailing and defaulted for the same
    # reason `cos_dpsi` is: no existing construction changes.
    sin_dpsi: Array = 0.0


# The two cases Parks et al. 1985 identifies, J. Aircraft 22(2) pp. 127-128.
# They live here rather than in a script because more than one entry point needs
# them, and a sourced number restated in two places is a number that will
# eventually disagree with itself.
#
# *** HANNIBAL'S RADIUS IS NOT PARKS' -- IT IS WINGROVE & BACH Fig. 4's. ***
# Decided session 22, and the citation moves with the number rather than being
# left pointing at a document that says something else.
#
# This entry read 600 ft, attributed to Parks 1985, until Wingrove & Bach 1994
# was obtained. That paper's Fig. 4 gives the same Hannibal vortex a 1000 ft
# core DIAMETER -- a 500 ft radius. The two disagree by 20% and Parks 1985 has
# never been retrieved, so the conflict could not be arbitrated on the documents.
# It was resolved in favour of the source actually held and read:
#
#   - Fig. 4 is a table of identified values in a paper that IS in hand, whose
#     Morton row (900 ft diameter -> 450 ft radius) reproduces this dict's
#     Morton radius to the digit. That agreement is what establishes the column
#     as a diameter, and it makes Fig. 4 a checked source rather than a guess.
#   - The 600 ft was a transcription from a paper nobody here has read.
#
# The superseded value is recorded rather than erased: Parks 1985 as transcribed
# gave r0 = 600 ft, and if that document is ever retrieved this is the line to
# revisit. `WINGROVE_FIG4_CASES` below still holds Fig. 4's numbers separately,
# so the two sources remain distinguishable even though they now agree.
#
# *** SESSION 23: THE DISAGREEMENT IS REAL, BOTH TRANSCRIPTIONS ARE FAITHFUL,
# AND THEY ARE TWO DIFFERENT FITS OF ONE ENCOUNTER. ***
#
# Session 23 first concluded that 600 ft was a pre-fit guess mistaken for a
# result -- Mehta's manual startup estimate is exactly 600 ft, which made a
# tidy story. THAT WAS WRONG, and the arithmetic that kills it is Parks' own
# Scorer check:
#
#     3500 ft spacing / 1200 ft diameter = 2.917,  and Parks quotes 2.92.
#     3500 ft spacing / 1000 ft diameter = 3.500,  which he does not.
#
# Parks' radius, his spacing and his published ratio are SELF-CONSISTENT to
# three figures at 600 ft. A transcription error would have broken that. So
# 600 ft is Parks' genuine identified value and the transcription is faithful.
#
# What the two new sources establish is therefore narrower and more useful:
#
#   - Mehta 1987 refits THE SAME ENCOUNTER with five vortices by modified
#     Newton-Raphson, cost falling 482 -> 214, and converges to r0 = 500.5 ft,
#     V0 = 86.8 ft/s. His startup estimate was 600 ft, read off the data by
#     inspection; the fit moved it.
#   - NASA TM-102186 p. 3-4 reports Mehta's converged answer in words: "a
#     diameter of 1,000 ft and a circumferential velocity of 87 ft/sec".
#
# Parks was presented as AIAA 84-0270 (January 1984) and Mehta as AIAA 84-2083
# (August 1984), and Mehta cites Parks. So the ordering is: Parks fits it,
# Mehta refits it with more vortices and a documented cost history, TM-102186
# reports Mehta's numbers.
#
# *** THE PRACTICAL CONSEQUENCE. *** 500 ft is the later and better-converged
# value and is what this project flies. 600 ft is not an error to be corrected
# but an earlier answer to be superseded, and Parks' Scorer ratio belongs to
# it -- which is why that check no longer reproduces here and why
# test_wind.py::test_the_spacing_to_core_diameter_ratio_and_what_session_22_cost_it
# must keep recording the loss rather than being retuned.
#
# Nothing about this rescues the Scorer check at either radius, and Mehta makes
# that plain: his five cores sit at perpendicular spacings of 5179, 5695, 3522
# and 7562 ft, so his own array's spacing-to-diameter ratios run 3.5 to 7.6.
# A uniform KH billow train is a Parks-shaped idealisation of a field that is
# not uniform.
#
# *** WHAT IS STILL A HYBRID, AND DELIBERATELY LEFT SO. *** This entry now
# pairs Fig. 4's RADIUS (500 ft) with Parks' STRENGTH (85 ft/s). Mehta and
# TM-102186 pair 500 ft with 87 ft/s, so no single source states the pair
# below. It is left alone because it is upstream of frozen PROJECT.md section 4
# baselines and moving it silently would invalidate them. The coherent
# single-source pair lives in MEHTA_HANNIBAL_1987 and is flown beside this one;
# what the 2.1% strength difference costs is MEASURED, in
# test_wind.py::test_the_mehta_and_parks_hannibal_strengths_bracket_the_load,
# rather than argued about here.
#
# `spacing` is still Parks': Fig. 4 gives core size and strength and says
# nothing about array spacing, so that number has not moved and cannot.
PARKS_CASES: dict[str, dict[str, float]] = {
    "hannibal": {"r0": 500.0 * FT2M, "v0": 85.0 * FT2M, "spacing": 3500.0 * FT2M},
    "morton": {"r0": 450.0 * FT2M, "v0": 70.0 * FT2M, "spacing": 3200.0 * FT2M},
}

# What Hannibal's radius was before session 22, and where it came from. Kept so
# the change is visible in the code and not only in the history.
HANNIBAL_R0_SUPERSEDED = 600.0 * FT2M  # Parks 1985 as transcribed; see above.


# ---------------------------------------------------------------------------
# The same vortices, as the LATER paper reports them.
#
# Source: R. C. Wingrove, R. E. Bach Jr., "Severe Turbulence and Maneuvering
# from Airline Flight Records", J. Aircraft 31(4), Jul-Aug 1994, pp. 753-760.
# Fig. 4, p. 755, "Models for vortex-induced turbulence". Obtained session 21;
# AUDIT.md had this source down as `unverifiable -- source not available`.
#
# A SEPARATE dict from PARKS_CASES on purpose. These are different numbers from
# a different paper, and merging them would put two citations on one dict and
# lose which value came from where -- the drift PARKS_CASES' own comment exists
# to prevent.
#
# Fig. 4's columns are headed "Vortex diameter (feet)" and "Tangential velocity
# (ft/sec)", reading 1000/85, 900/70 and 900/50. Stored here HALVED, as radii,
# so this dict carries the same quantity in the same units as PARKS_CASES.
#
# MORTON FIXES THE INTERPRETATION. 900 ft of diameter is 450 ft of radius, and
# PARKS_CASES["morton"]["r0"] is 450 ft to the digit. That agreement is the only
# thing distinguishing a diameter column from a radius column; without it every
# core here would risk being a factor of two out with nothing to catch it.
#
# ~~*** HANNIBAL DISAGREES, AND THAT IS NOT RESOLVED HERE. ***~~ RESOLVED IN
# SESSION 23. Fig. 4's 1000 ft diameter is a 500 ft radius, and Mehta 1987's
# converged fit (500.5 ft) plus TM-102186's prose ("a diameter of 1,000 ft")
# now agree with it independently. See the PARKS_CASES comment above for why
# 600 ft is Parks' own earlier fit, superseded rather than mistaken.
# The core STRENGTH is where the sources now split: Fig. 4 and Parks both say
# 85 ft/s, Mehta says 86.8 and TM-102186 says 87.
#
# CIMARRON APPEARS ONLY HERE. Parks identifies two cases; this paper adds a
# third, and it is the one with published time histories (Fig. 3 and Fig. 6a)
# and a published model-against-data overlay (Fig. 4) -- which is why it is the
# case the 737 flies.
#
# `spacing` is deliberately ABSENT. Fig. 4 gives core size and strength and says
# nothing about array spacing, so any value here would be invented. Callers that
# need an array take the spacing from PARKS_CASES and say that they did.
# ---------------------------------------------------------------------------
WINGROVE_FIG4_CASES: dict[str, dict[str, float]] = {
    "hannibal": {"r0": 500.0 * FT2M, "v0": 85.0 * FT2M},
    "morton": {"r0": 450.0 * FT2M, "v0": 70.0 * FT2M},
    "cimarron": {"r0": 450.0 * FT2M, "v0": 50.0 * FT2M},
}

# The altitude each incident was flown at. Wingrove & Bach Table 1, p. 754,
# which quotes them in hundreds of feet: 370, 390 and 330.
#
# Here because these three cases are NOT at one altitude, and density drives
# every aerodynamic force in the comparison. Flying all three at one nominal
# cruise would put a same-signed bias on every result.
WINGROVE_CASE_ALTITUDE: dict[str, float] = {
    "hannibal": 37000.0 * FT2M,
    "morton": 39000.0 * FT2M,
    "cimarron": 33000.0 * FT2M,
}


# ---------------------------------------------------------------------------
# Mehta 1987's converged Hannibal solution, as a COMPLETE field.
#
# Source: R. S. Mehta, "Modeling Clear-Air Turbulence with Vortices Using
# Parameter-Identification Techniques", J. Guidance, Control & Dynamics 10(1),
# Jan-Feb 1987, pp. 27-31. Presented as AIAA 84-2083. The five-vortex result is
# on p. 30, immediately after Fig. 9; psi and the altitude are on p. 29.
#
# THIS IS THE ONLY FIELD IN THIS MODULE THAT DECLARES NOTHING. Every other
# vortex case here takes a core size and a strength from a paper and then needs
# an array spacing, a core count and a traverse geometry supplied from
# somewhere else. Mehta prints all of it: five cores, their individual
# positions, one shared radius, one shared strength, the traverse angle and the
# altitude. Nothing below is a modelling choice.
#
# WHAT IS DROPPED, AND WHY IT IS FREE. Mehta's Eq. (4) also carries bias and
# trend terms (b_xy = 149.8 kt, b_z = 0, C_xy = C_z = 0) modelling the
# non-vortex wind. b_z and both trends are zero. b_xy is a UNIFORM horizontal
# wind, and this simulator's aero is air-relative by construction
# (dynamics.py forms vel_rel before aero.py sees anything), so a uniform wind
# moves the ground track and changes no force, no moment and no angle. Dropping
# it is exact for everything this module measures, and is NOT exact for a
# ground-referenced quantity such as the F-factor -- do not reuse this field
# for one without restoring b_xy.
#
# SIGN OF z. Mehta p. 28 defines x and z as "the horizontal separation along
# the flight path and the vertical separation of the airplane from the vortex",
# and fixes the sign on p. 29: a NEGATIVE horizontal perturbation gives z < 0.
# In Parks' Eq. (3) w_xy goes as +d with d the aircraft ABOVE the core, so
# z has the same sign as d and z is the aircraft's height above the core.
# `vortex_wind` computes `above = down - pos_ned[2]`, which is that same
# quantity, so a core sits at altitude MEHTA_HANNIBAL_ALTITUDE - z.
# ---------------------------------------------------------------------------

# ft, along the flight path. Vortices 3 and 4 are the pair the aircraft flew
# the cores of -- they are the two spikes TM-102186 Fig. 7 calls "significant".
MEHTA_HANNIBAL_X_FT = (-12384.0, -6669.0, -343.0, 3761.0, 12272.0)
# ft, aircraft above core (see SIGN OF z above).
MEHTA_HANNIBAL_Z_FT = (-3516.0, -1836.0, -94.0, -254.0, 1738.0)
MEHTA_HANNIBAL_R0 = 500.5 * FT2M  # m, converged core radius
MEHTA_HANNIBAL_V0 = 86.8 * FT2M  # m/s, converged tangential velocity
MEHTA_HANNIBAL_PSI_DEG = 31.0  # deg, wind vector to flightpath, p. 29
MEHTA_HANNIBAL_ALTITUDE = 37000.0 * FT2M  # m, "straight and level at 37,000 ft"


def mehta_hannibal_array(altitude: float = MEHTA_HANNIBAL_ALTITUDE) -> VortexArray:
    """Mehta 1987 p. 30's five-vortex Hannibal solution, as a `VortexArray`.

    `altitude` is the aircraft's nominal altitude, which is what Mehta's z is
    measured from. It is an argument rather than a constant because the cores
    have to be placed at the altitude the aircraft is actually trimmed at, and
    a run at any other altitude would be a different encounter -- callers that
    change it are declaring that, not tuning it.
    """
    return VortexArray(
        north=jnp.array([x * FT2M for x in MEHTA_HANNIBAL_X_FT]),
        down=jnp.array([-(altitude - z * FT2M) for z in MEHTA_HANNIBAL_Z_FT]),
        r0=jnp.array(MEHTA_HANNIBAL_R0),
        v0=jnp.array(MEHTA_HANNIBAL_V0),
        cos_dpsi=jnp.array(math.cos(math.radians(MEHTA_HANNIBAL_PSI_DEG))),
        # BOTH halves of the angle, so the array is self-consistent. `vortex_wind`
        # never reads the sine -- it drops out of Parks' expression on the flight
        # path -- so setting it moves no existing baseline by a bit. But
        # `line_vortex_wind` builds the vortex line's DIRECTION from the pair,
        # and a cosine without its sine is not a unit vector. Leaving it
        # defaulted made the line form silently wrong, which is how this was
        # found: `scripts/lateral.py` reported a 7 m/s disagreement where the
        # two forms are supposed to be identical.
        sin_dpsi=jnp.array(math.sin(math.radians(MEHTA_HANNIBAL_PSI_DEG))),
    )


# The two vortices whose cores the aircraft actually penetrated, as indices into
# the arrays above. Named rather than written as `2, 3` at each use site because
# the spacing reconciliation in test_wind.py and the analysis window in
# scripts/mehta_hannibal.py must agree on which pair they mean.
MEHTA_HANNIBAL_CORE_PAIR = (2, 3)

# What the measured DC-10 did in this encounter, from NASA TM-102186 p. 3-4:
# "the wide fluctuations in the normal acceleration from +1.7 to -1.0 g".
#
# A BAND TO BE REPORTED AGAINST, NOT A TARGET TO BE HIT. The aircraft is a
# DC-10 at 37,000 ft; this project's 747 has roughly 0.8x the wing loading, and
# PROJECT.md section 5 rules absolute load agreement structurally out of reach.
# Quoted here so a run can be drawn against it without the number being
# retyped into a plotting script.
TM102186_HANNIBAL_NZ = (-1.0, 1.7)  # g, measured min and max
TM102186_HANNIBAL_GUST_PERIOD = 5.0  # s, "sharp up-and-down gusts about 5 sec apart"

# ---------------------------------------------------------------------------
# HOW WELL MEHTA'S FIT ACTUALLY FITS, from his own Appendix.
#
# Eq. (A3) defines the cost he quotes at every array size:
#
#     J = (1/N) * sum_{j=1..N} e^T(j) B e(j),   e(j) = W_actual(j) - W_model(j)
#
# with B the identity matrix for every number below (p. 29, "with B an identity
# matrix"). THE 1/N IS THE LOAD-BEARING PART. J is a MEAN square, not a sum, so
# it converts to an RMS wind residual WITHOUT knowing N -- which the paper never
# states, and which no source held here supplies. Read as a sum it would be
# meaningless: at any plausible N the implied residual falls well below the
# reconstruction error of the data being fitted -- a factor of 4.1 even at N = 30
# -- which no honest fit can do.
#
# UNITS ARE ASSUMED, AND THE ASSUMPTION IS BOUNDED. Mehta never labels J. Its
# natural unit is (ft/s)^2: e is a difference of winds from Eq. (4), which is
# homogeneous in V0, and V0 is quoted in ft/s throughout. Fig. 5 nevertheless
# plots the HORIZONTAL wind in knots, so a mixed-unit e cannot be ruled out.
# `mehta_residual_ceiling` is immune to that -- B = I makes both terms
# non-negative, so either one alone is bounded by J whatever the other's scale.
# `mehta_unmodelled_wind` is NOT immune and says so.
MEHTA_COST_STARTUP = 482.0  # p. 29, the MANUAL startup estimate, n = 2
MEHTA_COST = {2: 355.0, 3: 303.0, 4: 226.0, 5: 214.0}  # p. 29-30, converged

# Mehta p. 30: "Further increases in the number of vortices (n = 6,7, etc.) do
# not result in decreases in the cost. In fact, the algorithm 'pushes' the extra
# vortices away from the flight path". So MEHTA_COST[5] is a FLOOR for this
# model family and not merely where the author stopped -- which is what makes it
# usable as a bound on the field form rather than on one author's patience.
MEHTA_COST_SATURATES_AT = 5


def vortex_wind(pos_ned: Array, array: VortexArray) -> Array:
    """Wind velocity (NED, m/s) induced by the array at a point.

    Superposition of Parks Eqs. (3)-(6) over the cores.
    """

    def one(north: Array, down: Array) -> Array:
        # `along` is Parks' l, the raw along-track separation. Every use of it
        # below is through `l_eff = l * cos(dpsi)`, which is what the source's
        # r = (l^2 cos^2 dpsi + d^2)^(1/2) and its w_z numerator both carry.
        # w_xy has NO cos(dpsi) factor of its own -- it depends on l only
        # through r -- so the two components are not scaled alike and the
        # substitution has to be made per term, not once on `along`.
        along = pos_ned[0] - north  # l, aircraft beyond the core
        along_eff = along * array.cos_dpsi  # l cos(dpsi)
        above = down - pos_ned[2]  # d, aircraft above the core
        r2 = along_eff**2 + above**2

        # Both branches are evaluated, so the outside form's divisor is clamped
        # away from zero. It only ever contributes where r2 >= r0^2 > 0, but an
        # unguarded 1/r2 would produce a NaN at the core centre that
        # jax_debug_nans (conftest.py) would trip on.
        r2_safe = jnp.where(r2 < array.r0**2, array.r0**2, r2)
        inside = r2 < array.r0**2

        w_horizontal = jnp.where(
            inside,
            array.v0 * above / array.r0,
            array.v0 * array.r0 * above / r2_safe,
        )
        w_up = jnp.where(
            inside,
            -array.v0 * along_eff / array.r0,
            -array.v0 * array.r0 * along_eff / r2_safe,
        )
        # The source's w_z is positive UP; NED z is positive DOWN.
        return jnp.array([w_horizontal, 0.0, -w_up])

    return jax.vmap(one)(array.north, array.down).sum(axis=0)


def gust_rates(pos_ned: Array, quat: Array, field) -> Array:
    """Body-axis (p, q, r) gust rates from the gradient of a wind field.

    A gust that varies across the span is a rolling input and one that varies
    along the fuselage is a pitching input -- for a vortex whose core is only a
    few spans across, these are not small. Ignoring them deletes the largest
    lateral input in the encounter.

    With (u_g, v_g, w_g) the BODY-axis components of the gust and body x
    forward / y right / z down, matching `omega_rel = omega - omega_gust` in
    dynamics.py:

        p_gust = +d(w_g)/dy      q_gust = -d(w_g)/dx      r_gust = +d(v_g)/dx

    Derivation of the q sign, which is the one that is easy to get backwards:
    a body rate q puts z-velocity -q*x at station x, so the relative z-velocity
    varies as -x*(q + dw_g/dx). The aero model sees an effective rate
    q_eff = q + dw_g/dx, and since it is handed omega - omega_gust, the gust
    rate must be the negative of the gradient.
    """
    dcm = quat_to_dcm(quat)  # body -> NED
    jac_ned = jax.jacfwd(field)(pos_ned)  # d(wind_ned)_i / d(pos_ned)_j
    grad_body = dcm.T @ jac_ned @ dcm  # d(gust_body)_i / d(pos_body)_j
    return jnp.array([grad_body[2, 1], -grad_body[2, 0], grad_body[1, 0]])


def _slope(coords: Array, values: Array) -> Array:
    """Least-squares slope of `values` against `coords`.

    Exact for a linear profile, which is what makes `sampled_rates` reduce to
    `gust_rates` whenever the field has no curvature across the airframe. The
    denominator cannot vanish for a station set with more than one distinct
    coordinate, which `airframe.stations` guarantees by construction.
    """
    centred = coords - coords.mean()
    return (centred * (values - values.mean())).sum() / (centred * centred).sum()


def sampled_rates(pos_ned: Array, quat: Array, field, stations) -> Array:
    """Body-axis (p, q, r) gust rates from a fit across the airframe.

    Same three quantities as `gust_rates` and the same sign convention -- this
    is a better ESTIMATOR of them, not a different quantity. `gust_rates` takes
    the tangent at the CG; this takes the secant across the extent the
    aerodynamics actually integrate over. For a field that is linear across the
    aircraft the two are identical, and `test_wind.py` asserts it.

    Deliberately does NOT add the three equivalences Stengel lists that the
    model omits (his eqs. 3.4-49, 3.4-51, 3.4-53). Combining each pair into one
    effective rate needs a weighting that his eq. 3.4-55 gets wrong -- it fails
    its own rigid-rotation self-check by a factor of -2 -- and that question is
    left to the strip integration, which never forms an equivalent rate at all.
    See the design document, section 2.
    """
    dcm = quat_to_dcm(quat)  # body -> NED

    def gust_body(offset_body: Array) -> Array:
        """Gust in BODY axes at a body-frame offset from the CG."""
        return dcm.T @ field(pos_ned + dcm @ offset_body)

    span_gusts = jax.vmap(
        lambda y: gust_body(jnp.array([0.0, y, 0.0]))
    )(stations.span)
    lon_gusts = jax.vmap(
        lambda x: gust_body(jnp.array([x, 0.0, 0.0]))
    )(stations.longitudinal)

    # Same three components, same signs, as gust_rates:
    #   p = +d(w_g)/dy    q = -d(w_g)/dx    r = +d(v_g)/dx
    p_gust = _slope(stations.span, span_gusts[:, 2])
    q_gust = -_slope(stations.longitudinal, lon_gusts[:, 2])
    r_gust = _slope(stations.longitudinal, lon_gusts[:, 1])
    return jnp.array([p_gust, q_gust, r_gust])


# ---------------------------------------------------------------------------
# Thunderstorm updraft column
#
# Source for the MAGNITUDES and the DURATION: Wingrove & Bach 1994, p. 756.
# Updrafts of 50 ft/s in the initial stage of thunderstorm development, rising
# to 100 ft/s as the storm builds; the Bermuda 12 Oct 1983 case measured over
# 80 ft/s and took 20 s to traverse, producing a 5.2 deg pitch variation.
#
# *** THE EDGE SHARPNESS IS NOT IN ANY SOURCE. *** The paper constrains the core
# magnitude and the traverse duration and says nothing about how abruptly the
# updraft begins. That matters more than the magnitude does: a smooth column at
# these dimensions is traversed in about three short periods, so the aircraft
# simply climbs with the air and the load factor barely moves. The paper's own
# +0.66 / -1.58 g comes from sharp edges, not from the 80 ft/s.
#
# So `sharpness` is a DECLARED MODELLING PARAMETER, not source data, and it is
# named as one here rather than buried in a default. sharpness = 2 is a plain
# Gaussian; larger values approach a top hat with a correspondingly steeper
# edge. Any result that depends on it must say which value was used.
# ---------------------------------------------------------------------------


# Wingrove & Bach 1994 p. 756, Bermuda 12 Oct 1983: over 80 ft/s, 20 s traverse.
# Here for the same reason as PARKS_CASES above.
UPDRAFT_W0 = 80.0 * FT2M  # m/s, peak updraft
UPDRAFT_SECONDS = 20.0  # s. A TRAVERSE time -- it fixes a diameter only with a speed.


class UpdraftColumn(NamedTuple):
    """An axisymmetric vertical column, super-Gaussian in horizontal radius.

        w_up(r) = w0 * exp(-(r / radius) ** sharpness)

    `radius` is where the updraft has fallen to w0/e. The paper's "20 s
    encounter" is a traverse time, so at a given flight speed it fixes the
    diameter: 20 s at the 747's 236 m/s cruise is about 4.7 km, hence a radius
    of order 2.4 km.
    """

    north: Array  # m, NED north of the column axis
    east: Array  # m, NED east of the column axis
    w0: Array  # m/s, peak updraft (positive UP)
    radius: Array  # m
    sharpness: Array  # DECLARED, not sourced. 2 = Gaussian; larger = sharper edge.


def updraft_wind(pos_ned: Array, column: UpdraftColumn) -> Array:
    """Wind velocity (NED, m/s) of an updraft column at a point."""
    offset = jnp.hypot(pos_ned[0] - column.north, pos_ned[1] - column.east)
    # Guard the fractional power at r = 0: for sharpness < 2 the derivative of
    # r**sharpness is singular there, and jax.grad would produce a NaN that
    # conftest's jax_debug_nans would trip on.
    scaled = jnp.maximum(offset / column.radius, 1e-12)
    w_up = column.w0 * jnp.exp(-(scaled**column.sharpness))
    return jnp.array([0.0, 0.0, -w_up])  # NED z is DOWN; an updraft is negative


# ---------------------------------------------------------------------------
# Mountain lee wave
#
# Source: J. D. Doyle, Q. Jiang, R. B. Smith, V. Grubisic, "Three-Dimensional
# Characteristics of Stratospheric Mountain Waves during T-REX", Mon. Wea. Rev.
# 139 (Jan 2011), 3-23, DOI 10.1175/2010MWR3466.1.
#
# Chosen over a textbook treatment because its measurements are at the right
# ALTITUDE. The NSF/NCAR Gulfstream V flew legs at 11.3 km and 13.1 km over the
# Sierra Nevada during T-REX (March-April 2006); this project's 747 cruises at
# 12.192 km, between them. Everything else in this file is DC-10-class data near
# the tropopause, so the altitudes stay comparable across the whole module.
#
# What the paper supplies, IOP 4 (14 March 2006), G-V at 13.1 km, quoted:
#   "The wave amplitude (crest to trough) of the primary wave is 12 m s-1 to the
#    south and 6 m s-1 to the north."
#   "maximum vertical velocities in excess of 6 m s-1"
#   "a tropospheric lee wave characterized by relatively long wavelengths
#    (~20-35 km)"
#
# What it does NOT supply, and is therefore DECLARED below: a stratospheric
# wavelength. The 20-35 km band is the paper's TROPOSPHERIC figure, and the same
# paragraph warns "Shorter wavelengths are apparent in the stratosphere" without
# giving a number. See LEE_WAVE_WAVELENGTH.
# ---------------------------------------------------------------------------

# m/s, zero-to-peak, i.e. half the paper's crest-to-trough figures. Two entries
# because they are the two legs of ONE flight and they straddle the 747's thrust
# authority -- which is the result, not a coincidence to be averaged away.
LEE_WAVE_AMPLITUDE: dict[str, float] = {"north": 3.0, "south": 6.0}

# m. DECLARED, not sourced: the paper's 20-35 km is tropospheric and it says
# stratospheric wavelengths are shorter without quantifying them. 25 km is the
# middle of the band the paper does give. It does not affect the F-factor peak
# at all -- with no horizontal perturbation the index is -w/V, independent of
# wavelength -- but it sets the encounter duration and the pitching gust rate,
# so any result that depends on those must say which value was used.
LEE_WAVE_WAVELENGTH = 25_000.0

# m. SOURCED, and it is the number the declared one above should be checked
# against rather than replaced by.
#
# Source: P. F. Lester, O. Sen, R. E. Bach Jr., "The Use of DFDR Information in
# the Analysis of a Turbulence Incident over Greenland", Mon. Wea. Rev. 117
# (May 1989), 1103-1107. p. 1106: the mesoscale wavelike variation dominating
# the derived vertical motion has a "wavelength about 22 km", and the paper
# reads the pattern as the aircraft traversing "the trough of a mountain lee
# wave over the western slopes of the Greenland icecap".
#
# WHY IT IS THE RIGHT COMPARISON AND NOT A REPLACEMENT. It is measured from
# DFDR data at 10 km (33,000 ft), which the same paper places about a kilometre
# above the tropopause -- so it is a wavelength in the regime Doyle et al.
# declined to quantify, which is exactly the gap LEE_WAVE_WAVELENGTH was
# declared to fill. But it is ONE case over Greenland, not the Sierra Nevada
# campaign LEE_WAVE_AMPLITUDE comes from, and this project's 747 cruises at
# 12.192 km rather than 10. Substituting it would trade a declared number
# inside a measured band for a measured number from a different mountain range
# at a different altitude, which is not obviously an improvement.
#
# WHAT IT DOES SETTLE. 22 km sits inside Doyle's 20-35 km band, so the declared
# 25 km is no longer merely the midpoint of a band known to be wrong for the
# altitude -- an independent measurement near that altitude lands 12% from it.
# PROJECT.md section 5's "the wavelength is declared, not sourced" stands, but
# the bound on the declaration is now measured rather than absent.
LESTER_LEE_WAVE_WAVELENGTH = 22_000.0

# The Greenland incident itself, for reporting a lee-wave run against. Same
# paper, p. 1105: a B-747 at 33,000 ft (10 km) MSL over southern Greenland at
# 62 N 48 W, 1654 UTC 22 January 1985, which "culminated in a sudden altitude
# gain of 1000 feet (300 m)" with "vertical accelerations reached +2.7g, -1.0g".
#
# A BAND TO REPORT AGAINST, NOT A TARGET. Same standing as
# TM102186_HANNIBAL_NZ, with one difference in this case's favour: the aircraft
# type is a 747, which is the type this project models. The altitude is not --
# 10 km against the modelled 12.192 km.
LESTER_GREENLAND_NZ = (-1.0, 2.7)  # g, measured min and max
LESTER_GREENLAND_ALTITUDE_GAIN = 300.0  # m, "a sudden altitude gain of 1000 feet"
LESTER_GREENLAND_ALTITUDE = 33000.0 * FT2M  # m

# m/s, RMS error of a DFDR-plus-radar wind reconstruction in level flight at
# V = 250 m/s. Lester et al. Table 1, p. 1105, which gives the contributions
# rather than the totals:
#
#   horizontal   dV_xy 1.0,  dV 1.0,  V d(psi + beta) 2.0   -> RSS 2.449
#   vertical     dh_dot 1.0,          V d(Theta - alpha) 2.0 -> RSS 2.236
#
# THIS IS THE BOUND ON EVERY IDENTIFIED VORTEX PARAMETER IN THIS MODULE, and it
# replaces the order-of-magnitude "+/-25%" that docs/ASSUMPTIONS.md carried.
# Note what the vertical term implies about the paper's own assumed flow-angle
# error: V d(Theta - alpha) = 2.0 at V = 250 gives d(Theta - alpha) = 0.0080 rad
# = 0.46 deg, about half the 1 deg that estimate assumed. The paper adds that
# these are "40%-50% greater than those estimated for NCAR aircraft".
DFDR_WIND_RMS_ERROR = {"horizontal": 2.449, "vertical": 2.236}
DFDR_WIND_RMS_ERROR_SPEED = 250.0  # m/s, the level-flight speed Table 1 assumes


def mehta_residual_ceiling(n: int = MEHTA_COST_SATURATES_AT) -> float:
    """RMS of Mehta's total fit residual, m/s. A CEILING on either component.

    `sqrt(J)` with J from `MEHTA_COST`. Because B is the identity, J is the mean
    of `e_xy**2 + e_z**2`, i.e. the sum of two non-negative means -- so either
    component's own mean square is bounded by J and neither component's RMS can
    exceed `sqrt(J)`. That includes the vertical one, which is what a gust model
    cares about, and it holds whatever unit the horizontal term is in. This is
    the one number here that survives that ambiguity.
    """
    return math.sqrt(MEHTA_COST[n]) * FT2M


def mehta_unmodelled_wind(n: int = MEHTA_COST_SATURATES_AT) -> float:
    """RMS per-component wind Mehta's fit does not represent, m/s.

    His residual is the model's error against RECONSTRUCTED winds, so it already
    contains the reconstruction error `DFDR_WIND_RMS_ERROR`. Removing that in
    quadrature leaves the physical fluctuation the vortex array omits -- which
    Mehta names on p. 30: "the small, random fluctuations that are part of the
    overall turbulence".

    A LOWER BOUND, not an estimate, and it is worth being clear which way each
    caveat pushes. Both push the same way, UP:

      - Independence is assumed in subtracting the squares. Mehta fits bias and
        trend terms explicitly, so the correlated part of the reconstruction
        error is partly absorbed into those and is not in the residual.
      - Lester's table is a DIFFERENT encounter, one with no ATC radar fixes
        (PROJECT.md section 3), so its errors are if anything an overestimate
        for Hannibal, which had them.

    So the true value lies between this and `mehta_residual_ceiling`.

    UNLIKE THE CEILING, THIS IS NOT UNIT-ROBUST. Splitting J evenly between the
    two components assumes both are in the same unit, and Mehta never labels J
    -- see the block above MEHTA_COST_STARTUP, and PROJECT.md section 8. If his
    horizontal residual is in knots the even split is wrong and only the ceiling
    survives.
    """
    measured = sum((v / FT2M) ** 2 for v in DFDR_WIND_RMS_ERROR.values())
    return math.sqrt(max(MEHTA_COST[n] - measured, 0.0) / 2.0) * FT2M


class LeeWave(NamedTuple):
    """A horizontally periodic vertical-velocity field: the downstream train.

        w_up(north) = -w0 * cos(2*pi*(north - north0) / wavelength)

    `north` marks a TROUGH, because the downdraft is the half that costs an
    aircraft energy and the whole point of this field is the F-factor.

    DECLARED SIMPLIFICATION -- purely vertical, with no vertical variation over
    the aircraft's altitude band. That is divergence-free, so it is an
    admissible incompressible flow rather than a convenient fiction, and it is
    the same shape as `UpdraftColumn`. What it omits is real: a lee wave also
    has a HORIZONTAL perturbation, in quadrature with the vertical one, whose
    amplitude ratio is the ratio of vertical to horizontal wavenumber. Building
    it needs a stratification N and an ambient cross-mountain wind speed, and no
    source held by this project supplies either at 12 km. The omission is
    recorded in PROJECT.md section 5 rather than papered over with a guess.
    """

    w0: Array  # m/s, zero-to-peak vertical velocity
    wavelength: Array  # m, crest to crest
    north: Array  # m, NED north of a trough


def lee_wave_wind(pos_ned: Array, wave: LeeWave) -> Array:
    """Wind velocity (NED, m/s) of a lee wave train at a point."""
    phase = 2.0 * jnp.pi * (pos_ned[0] - wave.north) / wave.wavelength
    w_up = -wave.w0 * jnp.cos(phase)
    return jnp.array([0.0, 0.0, -w_up])  # NED z is DOWN; an updraft is negative


def along_track_shear(
    pos_ned: Array, vel_ned: Array, accel_ned: Array, field
) -> Array:
    """dU_x/dt experienced by the aircraft, including the turn of its own track.

    `U_x` is the horizontal wind resolved along the ground track, POSITIVE FOR A
    TAILWIND, which is the sign convention Proctor et al. Eq. (3) requires.

    DERIVATION. Write the ground position p(t), the horizontal inertial velocity
    v_h = (v_N, v_E), the horizontal wind field W_h(p), and the unit track
    direction

        h = v_h / |v_h| = (cos psi, sin psi),   psi = atan2(v_E, v_N).

    Eq. (3) needs U_x(t) = W_h(p(t)) . h(t), and BOTH factors depend on time:

        dU_x/dt = [ (v.grad) W_h + dW_h/dt ] . h  +  W_h . dh/dt
                  |______________________________|    |____________|
                          Proctor Eq. (4)             heading rotation

    The first group IS Eq. (4), term for term. Its two spatial pieces are that
    equation's along-track shear times ground speed and vertical shear times
    ascent rate -- together the gradient of `U_x` contracted with the ground
    velocity -- and dW_h/dt is its local time derivative, zero for every field
    in this module because they are all steady in the earth frame.

    The second group is what this function used to omit, and it is a DERIVATION
    rather than a transcription: Eq. (4) is written for a straight track, and
    the paper does not extend it. Differentiating h,

        dh/dt = psi_dot * (-sin psi, cos psi) = psi_dot * n,   n = (-h_E, h_N),

    with n the track normal pointing 90 deg to the RIGHT (at psi = 0, h is north
    and n is east). So the omitted term is

        W_h . dh/dt = psi_dot * (W_h . n),

    the cross-track wind times the rate at which the along-track direction
    sweeps through it. Writing u_perp = -(W_h . n) for the crosswind FROM the
    right gives the equivalent form -u_perp * psi_dot.

    REDUCTION TO THE SOURCE. psi_dot = 0 annihilates the second group and leaves
    Eq. (4) untouched. That reduction is the check that this EXTENDS Proctor et
    al. rather than replacing them, and it is asserted as BIT equality against
    the straight-track expression this replaced, by
    test_audit_regression.py::test_along_track_shear_reduces_to_proctor_eq_4
    _when_the_track_is_straight.

    WHY THE ACCELERATION IS AN ARGUMENT. A position and a velocity do not
    determine a turn rate -- the track's rotation is a property of the
    trajectory, not of the field or of the instantaneous state. From the
    inertial acceleration,

        psi_dot = (v_N a_E - v_E a_N) / |v_h|^2,

    which is computed here so the formula lives in one place rather than at
    every call site. A caller whose ground track is genuinely straight passes
    zeros, and that is then a stated assumption rather than a silent one.

    SIZE OF THE OMISSION. Exactly zero along every run this project reports:
    they fly due north, and every field in this module has zero east wind on the
    north axis, so the cross-track wind and psi_dot both vanish. It reaches
    dF = 0.1423 at a standard-rate turn one core radius above a Parks core,
    where the Rankine tangential velocity is fully horizontal -- the whole of
    the FAA's 1 km alerting threshold, and measured rather than estimated. See
    `docs/ASSUMPTIONS.md` E7.
    """
    track = vel_ned[:2]
    speed = jnp.maximum(jnp.linalg.norm(track), 1e-9)
    heading = track / speed

    def u_x(p: Array) -> Array:
        return jnp.dot(field(p)[:2], heading)

    # Proctor Eq. (4): the field's own variation, seen along a frozen heading.
    frozen_heading = jnp.dot(jax.grad(u_x)(pos_ned), vel_ned)

    # The heading is not frozen. Same 1e-9 guard, squared, so a degenerate
    # ground track gives 0/1e-18 = 0 rather than a NaN, exactly as above.
    psi_dot = (track[0] * accel_ned[1] - track[1] * accel_ned[0]) / speed**2
    normal = jnp.array([-heading[1], heading[0]])
    return frozen_heading + psi_dot * jnp.dot(field(pos_ned)[:2], normal)


# ---------------------------------------------------------------------------
# Microburst
#
# Source: R. M. Oseguera and R. L. Bowles, "A Simple, Analytic 3-Dimensional
# Downburst Model Based on Boundary Layer Stagnation Flow", NASA TM-100632, July
# 1988. Velocity profiles taken from the TASS numerical model, itself built on
# the Joint Airport Weather Studies (JAWS) field data. Bowles is also the author
# of the F-factor in dynamics.py, so the field and the index it is measured with
# come from the same group.
#
# An axisymmetric stagnation-point flow: air descends on the axis, turns, and
# runs out radially. Both components go to zero AT THE GROUND, which the paper's
# introduction singles out as the thing earlier analytic models got wrong.
#
# Two shaping functions (paper's eqs. 5 and 6), with r the radius from the axis
# and z the height above ground:
#
#   u(r,z) = (lambda R^2 / 2r) [1 - exp(-(r/R)^2)] [exp(-z/z*) - exp(-z/eps)]
#   w(r,z) = -lambda exp(-(r/R)^2) [z*(1 - exp(-z/z*)) - eps(1 - exp(-z/eps))]
#
# These satisfy mass continuity exactly, which is asserted in the tests rather
# than taken on trust. The paper states four constants, and they are what pins
# the transcription:
#
#   peak outflow at r/R = 1.1212, z_m/z* = 0.22, z*/eps = 12.5,
#   u_max = 0.2357 * lambda * R
#
# THEY ARE NOT FOUR INDEPENDENT CHECKS, and this comment used to say they were.
# The paper's own order is the other way round: "Analysis of TASS data indicated
# ... the ratio z_m/z* = 0.22", and then "Recalling that z_m/z* = 0.22, the
# values 1.1212 and 12.5 were obtained from iteration for the ratios r/R and
# z*/eps", and 0.2357 follows from those. So 0.22 is an EMPIRICAL INPUT from the
# TASS model and 12.5 and 0.2357 are its consequences. The arithmetic the old
# comment gave is correct -- 0.22 is indeed ln(12.5)/11.5 -- but that identity
# is the relation 12.5 was solved FROM, so reading it backwards turns one
# empirical number into an apparent agreement between two.
#
# 1.1212 is the one that IS independent: it solves exp(-x^2)(2x^2+1) = 1, which
# involves no z at all. Treat the set as one empirical input, one independent
# root, and two consequences. Checked against the paper, printed pp. 4-5 and the
# appendix's "From TASS" block, held at refs/NASA-TM-100632-Oseguera-Bowles-1988.
# ---------------------------------------------------------------------------

# Oseguera & Bowles, from iteration on their own equations.
MICROBURST_PEAK_RADIUS_RATIO = 1.1212  # r/R at maximum outflow
MICROBURST_ZM_OVER_ZSTAR = 0.22  # altitude of maximum outflow, over z*
MICROBURST_ZSTAR_OVER_EPS = 12.5  # out-of-boundary-layer over in-boundary-layer
MICROBURST_UMAX_COEFF = 0.2357  # u_max = 0.2357 * lambda * R


class Microburst(NamedTuple):
    """An axisymmetric downburst. `z` is height above ground, so the ground is
    the NED plane z = 0 and this field is the only one here that has one."""

    north: Array  # m, NED north of the axis
    east: Array  # m, NED east of the axis
    lam: Array  # 1/s, the paper's scaling factor lambda
    radius: Array  # m, R, radius of the downdraft shaft
    z_star: Array  # m, characteristic height, out of boundary layer
    epsilon: Array  # m, characteristic height, in boundary layer


def microburst(
    *, u_max: float, radius: float, z_m: float, north: float = 0.0, east: float = 0.0
) -> Microburst:
    """Build a `Microburst` from the three quantities a source actually reports.

    A paper measures peak outflow, downdraft size and the height the outflow
    peaks at; it does not report `lambda`, `z*` or `epsilon`. Those are internal
    to the model and are recovered here through the paper's own three relations,
    so a caller states cited quantities and never has to invent a scale factor.
    """
    z_star = z_m / MICROBURST_ZM_OVER_ZSTAR
    return Microburst(
        north=jnp.array(north),
        east=jnp.array(east),
        lam=jnp.array(u_max / (MICROBURST_UMAX_COEFF * radius)),
        radius=jnp.array(radius),
        z_star=jnp.array(z_star),
        epsilon=jnp.array(z_star / MICROBURST_ZSTAR_OVER_EPS),
    )


def microburst_wind(pos_ned: Array, burst: Microburst) -> Array:
    """Wind velocity (NED, m/s) of a microburst at a point."""
    north = pos_ned[0] - burst.north
    east = pos_ned[1] - burst.east
    # Clamped at the ground. The shaping function contains exp(-z/epsilon) with
    # epsilon of order 50 m, so a few hundred metres of negative altitude
    # overflows to infinity -- and an aircraft flown into a microburst on fixed
    # controls DOES reach the ground, which is the result rather than an edge
    # case. Holding the ground value keeps such a run finite so the analysis can
    # find the impact point instead of returning NaN for the whole flight.
    altitude = jnp.maximum(-pos_ned[2], 0.0)
    # Squared radius first, then a floored sqrt: hypot's derivative is singular
    # at the axis and conftest turns a NaN into a failure. The floor never bites
    # on the value, because the bracket below vanishes like r^2 there.
    radius_sq = north * north + east * east
    scaled = radius_sq / burst.radius**2
    decay = jnp.exp(-scaled)
    shape = jnp.exp(-altitude / burst.z_star) - jnp.exp(-altitude / burst.epsilon)

    # The paper writes the outflow as (lam R^2 / 2r)[1 - exp(-(r/R)^2)], which
    # is 0/0 on the axis. Factoring the direction cosine n/r back in leaves a
    # function of r^2 alone with a REMOVABLE singularity, and that form has a
    # correct derivative on the axis where the literal one does not. This is not
    # cosmetic: `field_model` differentiates the field to get `omega_gust`, so a
    # value that is right while its gradient is wrong would give a silently
    # wrong rotational gust to anything flying through the core.
    safe = jnp.where(scaled > 1e-8, scaled, 1.0)  # keeps the unused branch finite
    ratio = jnp.where(
        scaled > 1e-8,
        -jnp.expm1(-safe) / safe,
        1.0 - 0.5 * scaled,  # the same function's series, to O(scaled^2)
    )
    horizontal = 0.5 * burst.lam * ratio * shape

    w_up = -burst.lam * decay * (
        burst.z_star * (1.0 - jnp.exp(-altitude / burst.z_star))
        - burst.epsilon * (1.0 - jnp.exp(-altitude / burst.epsilon))
    )
    return jnp.array(
        [horizontal * north, horizontal * east, -w_up]
    )  # NED z is DOWN


# ---------------------------------------------------------------------------
# Composition
#
# Every model here is a velocity field, and aero.py sees only vel_rel and
# omega_rel, so summing fields is exact within the model's own linearisation.
# That is what makes "a vortex array sitting in background turbulence" cost
# nothing beyond the two components themselves.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Dryden vertical gust, as a FROZEN SPATIAL FIELD.
#
# Source: MIL-F-8785C, the vertical-component spatial power spectral density
#
#     Phi_w(Omega) = sigma_w^2 (L_w/pi) (1 + 3 (L_w Omega)^2) / (1 + (L_w Omega)^2)^2
#
# integrated 0 -> infinity, which is what makes it a ONE-SIDED spectrum:
# substituting u = L_w Omega gives (sigma_w^2/pi) * integral of
# (1+3u^2)/(1+u^2)^2, and that integral is pi over the half line and 2pi over
# the whole line. Getting that convention wrong is a factor of 2 in variance and
# sqrt(2) in every gust, so `test_cat_validation.py` measures the realised
# variance rather than trusting this comment.
#
# *** WHY A FIELD AND NOT A SHAPING FILTER. *** The textbook implementation is a
# state-space filter driven by white noise, which needs filter states in
# `WindState`, a PRNG threaded through `step`, and a discretisation whose
# variance depends on dt. None of that is wanted here, because
# docs/ASSUMPTIONS.md E3 already commits this project to a FROZEN field -- wind
# depends on position, not time. Under that assumption the physically correct
# object IS a spatial realisation, and building one:
#
#   - composes with `superpose` like every other field, so a Dryden layer on top
#     of the Parks vortex is one call and not a new code path;
#   - is smooth and analytically differentiable, so `field_model`'s jacfwd still
#     produces a real gust-rate gradient. A grid realisation with interpolation
#     would give a piecewise-constant q_gust, which is exactly the channel the
#     vortex work cares about;
#   - has no dt dependence at all, so it cannot quietly change when a run is
#     refined -- see the E4 wind-hold entry for why that matters here.
#
# It is a SUM OF SINUSOIDS with Dryden-distributed amplitudes and uniform random
# phase. Each realisation is one sample, not an ensemble: a peak-load question
# needs several seeds, and the scripts that ask one run several.
#
# WHAT IS SOURCED AND WHAT IS NOT. L_w = 1750 ft is the spec's own constant
# above 2000 ft and is transcribed. sigma_w is NOT: MIL-F-8785C gives it as a
# chart against altitude and exceedance probability, PROJECT.md section 3
# records that chart as un-digitised, and nothing here invents a value. Callers
# pass sigma_w and say where it came from -- which for the CAT work means
# SWEEPING it and reporting what value would be needed, rather than asserting
# one.
# ---------------------------------------------------------------------------

DRYDEN_LW = 1750.0 * FT2M  # m, MIL-F-8785C scale length above 2000 ft
DRYDEN_ALTITUDE_FLOOR = 2000.0 * FT2M  # m, below which L_w is NOT this constant

# ISOTROPY ABOVE THE FLOOR. MIL-F-8785C makes the turbulence isotropic above
# 2000 ft: L_u = L_v = L_w and sigma_u = sigma_v = sigma_w. Below it the three
# diverge and none of these constants applies -- the same restriction L_w
# already carried. Named separately rather than everyone reusing DRYDEN_LW so
# that a future low-altitude case has three places to change and not one place
# to get wrong.
DRYDEN_LU = DRYDEN_LW
DRYDEN_LV = DRYDEN_LW


def dryden_spectrum(omega: Array, sigma_w: float, L_w: float = DRYDEN_LW) -> Array:
    """MIL-F-8785C vertical spatial PSD at spatial frequency `omega` (rad/m)."""
    u2 = (L_w * omega) ** 2
    return sigma_w**2 * (L_w / jnp.pi) * (1.0 + 3.0 * u2) / (1.0 + u2) ** 2


def dryden_vertical_field(
    sigma_w: float,
    seed: int,
    *,
    L_w: float = DRYDEN_LW,
    n_components: int = 400,
    wavelength_min: float = 20.0,
    wavelength_max: float = 40_000.0,
):
    """A frozen along-track Dryden vertical gust, as a position-only field.

    `wavelength_min` bounds the smallest structure represented. 20 m is a third
    of a 747 span, which is already below the scale at which this project's
    point-sampled gust means anything (`docs/ASSUMPTIONS.md` E2) -- going finer
    would add variance the aircraft model cannot legitimately respond to.

    `wavelength_max` bounds the largest. 40 km is 75 scale lengths, far enough
    out that the omitted low-wavenumber tail is a fraction of a per cent of the
    variance -- which `test_cat_validation.py` measures rather than assumes.

    Components are LOG-SPACED, so the resolution follows the spectrum's own
    shape instead of wasting most of the sum on the flat high-wavenumber tail.
    Each carries amplitude sqrt(2 Phi dOmega) and a uniform random phase, which
    makes the realised variance sum(Phi dOmega) -- a Riemann sum of the integral
    that defines sigma_w^2.
    """
    key = jax.random.PRNGKey(seed)
    omega = jnp.geomspace(
        2.0 * jnp.pi / wavelength_max, 2.0 * jnp.pi / wavelength_min, n_components
    )
    # Trapezoidal widths on a log grid: each component owns half the gap to
    # each neighbour, and the two ends own their single half-gap.
    edges = jnp.concatenate([
        omega[:1], jnp.sqrt(omega[1:] * omega[:-1]), omega[-1:]
    ])
    d_omega = edges[1:] - edges[:-1]
    amplitude = jnp.sqrt(2.0 * dryden_spectrum(omega, sigma_w, L_w) * d_omega)
    phase = jax.random.uniform(key, (n_components,), maxval=2.0 * jnp.pi)

    def field(pos_ned: Array) -> Array:
        w_up = jnp.sum(amplitude * jnp.cos(omega * pos_ned[0] + phase))
        # NED z is DOWN; an updraft is negative. Same convention as LeeWave.
        return jnp.array([0.0, 0.0, -w_up])

    return field


# ---------------------------------------------------------------------------
# THE OTHER TWO DRYDEN COMPONENTS, added session 24 (phase 1).
#
# `dryden_spectrum` above is the TRANSVERSE form, and it serves BOTH the
# vertical and the lateral component -- in isotropic turbulence v and w have
# the same one-dimensional spectrum. The longitudinal component does not:
#
#     Phi_u(Omega) = sigma_u^2 (2 L_u/pi) / (1 + (L_u Omega)^2)
#
# WHY THIS IS MORE THAN A TRANSCRIPTION. MIL-F-8785C is not in the folder --
# PROJECT.md section 3 records only its sigma chart as un-digitised, but the
# spectral forms are second-hand here too, and the vertical one already was.
# What can be checked without the document is that the two forms belong to the
# SAME isotropic field, through the standard relation
#
#     Phi_transverse = 0.5 * (Phi_long - Omega dPhi_long/dOmega)
#
# and they do, exactly: substituting the longitudinal form gives
# sigma^2 (L/pi) (1 + 3 L^2 Omega^2)/(1 + L^2 Omega^2)^2, which is
# `dryden_spectrum` term for term. `test_lateral.py` verifies it numerically
# rather than trusting this comment. That check is worth more than the
# transcription: a wrong pair of forms would almost certainly fail it.
# ---------------------------------------------------------------------------


def dryden_longitudinal_spectrum(
    omega: Array, sigma_u: float, L_u: float = DRYDEN_LU
) -> Array:
    """MIL-F-8785C longitudinal spatial PSD at spatial frequency `omega`.

    One-sided, matching `dryden_spectrum`: the integral over the half line is
    sigma_u^2, not 2 sigma_u^2. Same factor-of-two trap, same reason it is
    measured in a test rather than asserted here.
    """
    return sigma_u**2 * (2.0 * L_u / jnp.pi) / (1.0 + (L_u * omega) ** 2)


def _dryden_component(spectrum, sigma, key, L, n_components,
                      wavelength_min, wavelength_max):
    """One frozen along-track component. The body of `dryden_vertical_field`.

    Factored out when the other two components arrived, so all three are one
    piece of arithmetic rather than three that could drift. Returns a scalar
    function of along-track distance.
    """
    omega = jnp.geomspace(
        2.0 * jnp.pi / wavelength_max, 2.0 * jnp.pi / wavelength_min, n_components
    )
    edges = jnp.concatenate([
        omega[:1], jnp.sqrt(omega[1:] * omega[:-1]), omega[-1:]
    ])
    d_omega = edges[1:] - edges[:-1]
    amplitude = jnp.sqrt(2.0 * spectrum(omega, sigma, L) * d_omega)
    phase = jax.random.uniform(key, (n_components,), maxval=2.0 * jnp.pi)
    return lambda x: jnp.sum(amplitude * jnp.cos(omega * x + phase))


def dryden_field(
    sigma: float,
    seed: int,
    *,
    L: float = DRYDEN_LW,
    n_components: int = 400,
    wavelength_min: float = 20.0,
    wavelength_max: float = 40_000.0,
):
    """All three Dryden components as one frozen position-only field.

    Isotropic: one `sigma` and one `L` serve all three, which is what
    MIL-F-8785C specifies above 2000 ft. The three components get INDEPENDENT
    phase sets from one seed, so they are uncorrelated -- which isotropic
    turbulence requires and which a shared phase set would silently violate.

    *** AXES, AND THIS IS AN APPROXIMATION WORTH STATING. *** Dryden's u, v, w
    are the aircraft's own axes; this returns NED. The two coincide for wings
    level on a northerly heading, which is every run in this project. A turning
    or crabbing run would need the field rotated into the body frame, and
    nothing here does that -- so do not fly this on a manoeuvring case without
    fixing it first.

    *** THIS FIELD STILL HAS NO SPANWISE VARIATION. *** All three components are
    functions of along-track distance alone, so every strip of the wing sees the
    same gust and `strip_roll_moment` still integrates to zero on it. The
    lateral excitation it provides is SIDESLIP from the v component, not a
    rolling gust. `line_vortex_wind` is the field that varies across the span.
    """
    ku, kv, kw = jax.random.split(jax.random.PRNGKey(seed), 3)
    u = _dryden_component(dryden_longitudinal_spectrum, sigma, ku, L,
                          n_components, wavelength_min, wavelength_max)
    v = _dryden_component(dryden_spectrum, sigma, kv, L,
                          n_components, wavelength_min, wavelength_max)
    w = _dryden_component(dryden_spectrum, sigma, kw, L,
                          n_components, wavelength_min, wavelength_max)

    def field(pos_ned: Array) -> Array:
        x = pos_ned[0]
        # NED z is DOWN; an updraft is negative. Same convention as everywhere.
        return jnp.array([u(x), v(x), -w(x)])

    return field


# ---------------------------------------------------------------------------
# THE VORTEX AS A LINE IN SPACE, added session 24 (phase 1).
#
# `vortex_wind` above evaluates Parks' r = (l^2 cos^2 dpsi + d^2)^(1/2), which
# is the perpendicular distance to a vortex LINE -- but only for a point on the
# flight path. It has no `y` dependence at all, so every strip of the wing sees
# the same gust and the strip roll integral returns exactly zero on it. That is
# the gap PROJECT.md section 5 records: the strip path has never moved a number.
#
# WHAT THIS ADDS IS GEOMETRY, NOT PHYSICS. The same line vortex, written in
# three dimensions:
#
#     delta = pos - core
#     r_vec = delta - (delta . t) t          t = the vortex line's direction
#     v     = V(|r_vec|) * (delta x t) / |r_vec|
#
# with Parks' own V(r) -- solid body inside r0, potential outside. On the flight
# path (y = 0) this reduces to his formula EXACTLY: with t = (sin dpsi, cos
# dpsi, 0) and delta = (l, 0, -d), the perpendicular distance squared works out
# to l^2 cos^2 dpsi + d^2, which is Parks Eq. (2) term for term.
#
# WRITTEN BESIDE `vortex_wind` RATHER THAN REPLACING IT. Two reasons, and the
# second is the real one. First, every section-4 vortex baseline was frozen
# against `vortex_wind` and re-deriving them through different floating-point
# arithmetic would move digits for no physical reason. Second, and better: two
# independent implementations of one field that must agree is the pattern this
# project already uses against JSBSim, and it catches what one implementation
# cannot. `test_lateral.py` reconciles them along the whole flight path.
#
# So: `vortex_wind` stays the point model and keeps its baselines;
# `line_vortex_wind` is what a run flies when it wants a spanwise gradient.
# ---------------------------------------------------------------------------


def vortex_axis(array: VortexArray) -> Array:
    """Unit vector along the vortex lines, NED.

    Built from the array's own `cos_dpsi` and `sin_dpsi`. The default
    (1.0, 0.0) gives due east -- perpendicular to a northerly flight path,
    which is the geometry `vortex_wind` assumes and Parks draws.
    """
    axis = jnp.array([array.sin_dpsi, array.cos_dpsi, 0.0])
    # NORMALISED, and the reason is a bug this caught. `cos_dpsi` predates
    # `sin_dpsi` by a session, so an array can carry a cosine with the sine
    # still at its 0.0 default -- a vector of length cos(dpsi), not 1. The
    # induced velocity then comes out scaled by that length and the field is
    # quietly wrong rather than loudly broken. Normalising makes the failure a
    # wrong ANGLE, which `test_lateral.py` can see, instead of a wrong
    # MAGNITUDE, which looks like physics. Constructors that set both -- which
    # is all of them now -- are unaffected: the norm is already 1.
    return axis / jnp.linalg.norm(axis)


def line_vortex_wind(pos_ned: Array, array: VortexArray) -> Array:
    """Wind from the array, as lines in three dimensions. Varies across the span.

    Reduces to `vortex_wind` on the flight path; differs off it, which is the
    entire point. See the block comment above for the geometry and for why both
    forms exist.
    """
    t = vortex_axis(array)

    def one(core_north: Array, core_down: Array) -> Array:
        delta = pos_ned - jnp.array([core_north, 0.0, core_down])
        perpendicular = delta - jnp.dot(delta, t) * t
        r2 = jnp.dot(perpendicular, perpendicular)
        r = jnp.sqrt(jnp.maximum(r2, 1e-12))
        # Direction: the tangential unit vector, |delta x t| = r by construction
        # since the component of delta along t contributes nothing to the cross
        # product. Sign matches `vortex_wind` -- checked, not assumed, by
        # test_the_two_vortex_forms_agree_along_the_whole_flight_path.
        direction = jnp.cross(delta, t) / r
        # Parks' V(r): solid body inside the core, potential outside. `<` rather
        # than `<=` matches `vortex_wind`'s tie-break at exactly r = r0
        # (docs/ASSUMPTIONS.md E9).
        speed = jnp.where(r2 < array.r0**2,
                          array.v0 * r / array.r0,
                          array.v0 * array.r0 / r)
        return speed * direction

    return jnp.sum(jax.vmap(one)(array.north, array.down), axis=0)


def line_vortex_model(array: VortexArray):
    """`wind_model` for an array flown as lines. Sibling of `vortex_model`."""
    return field_model(lambda pos_ned: line_vortex_wind(pos_ned, array))


def superpose(*fields):
    """Sum wind fields. Parks et al. 1985 builds its vortex arrays this way.

    THE EMPTY CASE IS HANDLED EXPLICITLY. `sum(...)` over no terms is the Python
    integer 0, so `superpose()` used to return a value where its caller expected
    a callable -- reachable from `superpose(*chosen)` whenever the filter that
    built `chosen` selected nothing. A sum of no fields is the zero field, which
    is the identity this function's own algebra requires, so that is what comes
    back.

    Written as a separate branch rather than as a `start=` argument so that the
    non-empty path performs exactly the arithmetic it always did, down to the
    signed zeros.
    """
    if not fields:

        def zero_field(pos_ned: Array) -> Array:
            del pos_ned
            return jnp.zeros(3)

        return zero_field

    def combined(pos_ned: Array) -> Array:
        return sum(field(pos_ned) for field in fields)

    return combined


def gust_alphadot(pos_ned: Array, quat: Array, vel_body: Array, field) -> Array:
    """Wind-induced angle-of-attack rate, rad/s, from the field's own gradient.

    The aircraft flying through a frozen field sees the wind change at a rate
    given by the material derivative under the Taylor hypothesis,

        d(wind_ned)/dt = J @ vel_ned,    J = d(wind_ned)/d(pos_ned)

    which is the SAME Jacobian `gust_rates` already forms to produce
    `omega_gust`. No new differentiation, no finite differencing, and a field
    whose gradient is wrong is caught by the existing rate tests rather than
    silently producing a wrong alphadot here.

    Only the WIND part of alphadot is returned. The aircraft's own contribution
    is implicit -- alphadot depends on wdot depends on the forces depend on
    alphadot -- and stays folded into Cmq, which is exact whenever alphadot = q.
    What is returned is precisely the part the aircraft's own motion cannot
    produce: Stengel, Flight Dynamics 2nd ed. p.227, "a plunging aircraft
    experiences non-zero alphadot with zero q".

    Exact rather than small-angle: alpha = arctan2(w_rel, u_rel), so

        alphadot = (u_rel * wdot_rel - w_rel * udot_rel) / (u_rel^2 + w_rel^2)

    and the gust contributes -(d(wind_body)/dt) to the relative velocity.
    """
    dcm = quat_to_dcm(quat)  # body -> NED
    vel_ned = dcm @ vel_body
    wind_rate_ned = jax.jacfwd(field)(pos_ned) @ vel_ned
    # Relative velocity falls as the wind rises, hence the sign.
    rel_rate_body = -(dcm.T @ wind_rate_ned)

    vel_rel = vel_body - dcm.T @ field(pos_ned)
    u_rel, w_rel = vel_rel[0], vel_rel[2]
    # aero.V_MIN, squared. The SAME constant rather than a second one with the
    # same value: one NaN guard, one provenance entry, and it cannot drift.
    denominator = jnp.maximum(u_rel**2 + w_rel**2, V_MIN**2)
    return (u_rel * rel_rate_body[2] - w_rel * rel_rate_body[0]) / denominator


def field_model(field):
    """Turn a position-only wind field into a `wind_model`.

    Returned closure matches the `zero_wind` signature, so it drops straight
    into `integrate.step`/`rollout` and `autopilot.closed_loop_rollout`.
    `omega_gust` is the analytic gradient of the field, so a component cannot
    contribute a translational gust while silently omitting its rotational one.

    The key is returned UNTOUCHED. That is what makes a batch of PRNG keys vary
    only the stochastic components of a composed field, so every member of a
    Monte Carlo ensemble meets the same vortex at the same place -- which is the
    experiment design an error bar on a deterministic encounter needs.
    """

    def model(
        wind_state: WindState, state: State, key: Array, dt: float
    ) -> tuple[Array, Array, WindState, Array]:
        del dt
        wind_ned = field(state.pos_ned)
        omega_gust = gust_rates(state.pos_ned, state.quat, field)
        alphadot = gust_alphadot(state.pos_ned, state.quat, state.vel_body, field)
        return wind_ned, omega_gust, wind_state, key, alphadot

    return model


def sampled_field_model(field, stations):
    """`field_model`, but with the gust rates fitted across the airframe.

    Identical contract to `field_model` -- same signature, same returned tuple,
    key untouched -- so it is a drop-in wherever a wind model is accepted. The
    only difference is which estimator produces `omega_gust`.

    `wind_ned` is still the CG sample. Averaging the translational gust over
    the airframe is a separate change with its own weighting question, and it
    belongs to the strip integration rather than here: this stage changes the
    estimator for quantities already in use and introduces no new constants.
    """

    def model(
        wind_state: WindState, state: State, key: Array, dt: float
    ) -> tuple[Array, Array, WindState, Array]:
        del dt
        wind_ned = field(state.pos_ned)
        omega_gust = sampled_rates(state.pos_ned, state.quat, field, stations)
        alphadot = gust_alphadot(state.pos_ned, state.quat, state.vel_body, field)
        return wind_ned, omega_gust, wind_state, key, alphadot

    return model


# ---------------------------------------------------------------------------
# Strip integration
#
# `sampled_rates` improves the ESTIMATOR for three numbers, but it still
# collapses the field to three numbers, so a profile that varies non-linearly
# across the span is still not represented. Only integrating the field per
# strip carries that. Stengel (Flight Dynamics 2nd ed, p. 217) is explicit that
# below rotor scale the rotary derivatives stop being adequate and strip theory
# or CFD is required; this is the strip-theory half of that.
# ---------------------------------------------------------------------------


def _strip_rolling_coefficient(ac: Aircraft, stations, incidence: Array) -> Array:
    """Rolling-moment coefficient from a spanwise incidence distribution.

        Cl = -(1/(S*b)) * integral( y * c(y) * a0 * dalpha(y) dy )

    The leading minus sign is the body-axis convention: extra lift on the right
    wing (y > 0) acts in -z, and the moment about x is y*F_z, so more lift to
    starboard rolls the aircraft to port. That is what makes roll damping oppose
    roll rate, and it is the sign most easily got backwards.

    Trapezoidal rather than a fixed-order quadrature because the elliptic chord
    has infinite slope at the tips, where a low-order rule does noticeably worse
    than simply using more stations.
    """
    y = stations.span
    chord = airframe.chord_distribution(y, ac)
    a0 = airframe.calibrated_lift_slope(ac)
    return -jnp.trapezoid(y * chord * a0 * incidence, y) / (ac.S * ac.b)


def strip_clp_from_rate(ac: Aircraft, stations, p_hat: Array) -> Array:
    """Rolling-moment coefficient produced by a rigid roll rate.

    The calibration check: with `airframe.calibrated_lift_slope` pinned to the
    tabulated Clp, this returns `Clp * p_hat` -- IN THE CONTINUUM LIMIT, which is
    where the identity a0 = -8*Clp holds. At the shipped `airframe.N_SPAN = 9` it
    returns 82.6% of that, converging at order 1.50; see
    `airframe.calibrated_lift_slope` and docs/ASSUMPTIONS.md F5. Stengel
    eq. 3.4-39 gives the spanwise incidence a roll rate induces, dalpha = p*y/V,
    which in terms of p_hat = pb/2V is dalpha = 2*p_hat*y/b.
    """
    incidence = 2.0 * p_hat * stations.span / ac.b
    return _strip_rolling_coefficient(ac, stations, incidence)


def strip_roll_moment(
    pos_ned: Array, quat: Array, field, ac: Aircraft, stations, airspeed: Array
) -> Array:
    """Rolling-moment coefficient from a wind field, integrated across the span.

    Each strip is given the gust at ITS OWN position rather than the CG's, so a
    profile that varies non-linearly across the span produces the moment it
    physically would. That is the whole point of the strip treatment and the one
    thing an equivalent rate cannot reproduce.

    Returns a coefficient, not a moment, so it composes with `aero.py`'s
    coefficient build-up rather than bypassing it.

    SIGN, and it is the one to be careful about because the two halves point
    opposite ways. A wing moving DOWN meets the air from below and gains
    incidence: that is the +p*y/V of `strip_clp_from_rate`. Air moving DOWN past
    a stationary wing arrives from above and LOSES incidence. So the gust
    increment is -w_g/V, not +w_g/V.

    This is the same convention as everywhere else in the package, and it is
    forced by it: `dynamics.relative_velocity` forms `vel_body - dcm.T @
    wind_ned`, so a larger downward gust reduces the relative w and therefore
    reduces alpha. Getting this backwards produces a model that rolls the right
    way for its own motion and the wrong way for every gust, which no test of
    rigid rotation alone would catch -- which is why
    `test_a_linear_gust_gradient_matches_the_equivalent_rate_answer` compares
    the two against each other.
    """
    dcm = quat_to_dcm(quat)

    def gust_w(y: Array) -> Array:
        offset = jnp.array([0.0, y, 0.0])
        return (dcm.T @ field(pos_ned + dcm @ offset))[2]

    w_gust = jax.vmap(gust_w)(stations.span)
    return _strip_rolling_coefficient(ac, stations, -w_gust / airspeed)


def vortex_model(array: VortexArray):
    """`wind_model` for a vortex array."""
    return field_model(lambda pos_ned: vortex_wind(pos_ned, array))


def updraft_model(column: UpdraftColumn):
    """`wind_model` for an updraft column."""
    return field_model(lambda pos_ned: updraft_wind(pos_ned, column))


def lee_wave_model(wave: LeeWave):
    """`wind_model` for a mountain lee wave train."""
    return field_model(lambda pos_ned: lee_wave_wind(pos_ned, wave))


def microburst_model(burst: Microburst):
    """`wind_model` for a microburst."""
    return field_model(lambda pos_ned: microburst_wind(pos_ned, burst))
