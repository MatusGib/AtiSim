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

from flightsim.state import State, quat_to_dcm
from flightsim.units import FT2M


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
) -> tuple[Array, Array, WindState, Array]:
    """Still air. Returns (wind_ned, omega_gust, wind_state, key)."""
    del state, dt
    return jnp.zeros(3), jnp.zeros(3), wind_state, key


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


def along_track_shear(pos_ned: Array, vel_ned: Array, field) -> Array:
    """dU_x/dt experienced by the aircraft. Proctor et al. Eq. (4), steady field.

    `U_x` is the horizontal wind resolved along the ground track, POSITIVE FOR A
    TAILWIND, which is the sign convention Eq. (3) requires. Eq. (4) splits the
    rate into three terms -- along-track shear times ground speed, vertical
    shear times ascent rate, and the local time derivative. The first two are
    exactly the gradient of `U_x` contracted with the ground velocity, which is
    what this computes; the third is zero for every field in this module,
    because they are all steady in the earth frame.
    """
    track = vel_ned[:2]
    heading = track / jnp.maximum(jnp.linalg.norm(track), 1e-9)

    def u_x(p: Array) -> Array:
        return jnp.dot(field(p)[:2], heading)

    return jnp.dot(jax.grad(u_x)(pos_ned), vel_ned)


# ---------------------------------------------------------------------------
# Composition
#
# Every model here is a velocity field, and aero.py sees only vel_rel and
# omega_rel, so summing fields is exact within the model's own linearisation.
# That is what makes "a vortex array sitting in background turbulence" cost
# nothing beyond the two components themselves.
# ---------------------------------------------------------------------------


def superpose(*fields):
    """Sum wind fields. Parks et al. 1985 builds its vortex arrays this way."""

    def combined(pos_ned: Array) -> Array:
        return sum(field(pos_ned) for field in fields)

    return combined


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
        return wind_ned, omega_gust, wind_state, key

    return model


def vortex_model(array: VortexArray):
    """`wind_model` for a vortex array."""
    return field_model(lambda pos_ned: vortex_wind(pos_ned, array))


def updraft_model(column: UpdraftColumn):
    """`wind_model` for an updraft column."""
    return field_model(lambda pos_ned: updraft_wind(pos_ned, column))


def lee_wave_model(wave: LeeWave):
    """`wind_model` for a mountain lee wave train."""
    return field_model(lambda pos_ned: lee_wave_wind(pos_ned, wave))
