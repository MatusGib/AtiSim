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
# This module fixes dpsi = 0 -- vortex axes perpendicular to the track, which is
# the case the source's own data constrains (it reports vertical-wind traces and
# no lateral results at all). An oblique traverse would carry cos(dpsi) through
# and is deliberately not offered rather than shipped unvalidated.
# ---------------------------------------------------------------------------


class VortexArray(NamedTuple):
    """A row of co-rotating Rankine vortices, all with the same core.

    Parks identifies exactly this: an array on the downslope of a standing
    wave, rotating in one sense. `north`/`down` are the NED coordinates of each
    core, so N is fixed by their shape -- changing the count recompiles, which
    is correct, and vmapping over encounter geometry batches these leaves.

    Identified values (Parks pp. 127-128), both DC-10s near the tropopause:
      Case 1, Hannibal MO,  37,000 ft: r0 = 600 ft, V0 = 85 ft/s, spacing 3500 ft
      Case 2, Morton WY,    39,000 ft: r0 = 450 ft, V0 = 70 ft/s, spacing 3200 ft
    """

    north: Array  # (N,) m, NED north of each core
    down: Array  # (N,) m, NED down of each core
    r0: Array  # m, solid-body core radius
    v0: Array  # m/s, tangential velocity at the core edge


# The two cases Parks et al. 1985 identifies, J. Aircraft 22(2) pp. 127-128.
# They live here rather than in a script because more than one entry point needs
# them, and a sourced number restated in two places is a number that will
# eventually disagree with itself.
PARKS_CASES: dict[str, dict[str, float]] = {
    "hannibal": {"r0": 600.0 * FT2M, "v0": 85.0 * FT2M, "spacing": 3500.0 * FT2M},
    "morton": {"r0": 450.0 * FT2M, "v0": 70.0 * FT2M, "spacing": 3200.0 * FT2M},
}


def vortex_wind(pos_ned: Array, array: VortexArray) -> Array:
    """Wind velocity (NED, m/s) induced by the array at a point.

    Superposition of Parks Eqs. (3)-(6) over the cores.
    """

    def one(north: Array, down: Array) -> Array:
        along = pos_ned[0] - north  # l, aircraft beyond the core
        above = down - pos_ned[2]  # d, aircraft above the core
        r2 = along**2 + above**2

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
            -array.v0 * along / array.r0,
            -array.v0 * array.r0 * along / r2_safe,
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
