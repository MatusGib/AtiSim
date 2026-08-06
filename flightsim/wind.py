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


def vortex_model(array: VortexArray):
    """Build a `wind_model` for a vortex array.

    Returned closure matches the `zero_wind` signature, so it drops straight
    into `integrate.step`/`rollout` and `autopilot.closed_loop_rollout`.

    The key is returned UNTOUCHED. That is what makes a batch of PRNG keys vary
    only the stochastic components of a composed field, so every member of a
    Monte Carlo ensemble meets the same vortex at the same place -- which is the
    experiment design an error bar on a deterministic encounter needs.
    """

    def field(pos_ned: Array) -> Array:
        return vortex_wind(pos_ned, array)

    def model(
        wind_state: WindState, state: State, key: Array, dt: float
    ) -> tuple[Array, Array, WindState, Array]:
        del dt
        wind_ned = field(state.pos_ned)
        omega_gust = gust_rates(state.pos_ned, state.quat, field)
        return wind_ned, omega_gust, wind_state, key

    return model
