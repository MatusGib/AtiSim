"""Aerodynamic coefficient increments from distributed loads.

`aero.py` builds coefficients from air-relative velocity and rate, and its
standing rule is that it never sees the wind field. That rule is what makes a
zero-strength wind bit-identical to still air, and it is asserted. So a load
computed by integrating the field ACROSS the airframe -- which necessarily does
see the field -- cannot be built there.

It enters instead as an increment summed into the coefficients inside
`dynamics.derivatives`, at the same seam where wind already enters and nowhere
else. That keeps `aero.py` untouched, and it means the increment composes with
dynamic pressure and reference area exactly as every other coefficient does.

Four coefficients, named rather than positional. A bare (4,) array would let a
caller put a rolling moment in the yaw slot and produce a model that flies
plausibly and wrongly.

Side force and drag are deliberately absent. A spanwise incidence distribution
produces lift and the three moments at first order; it does not produce side
force, and its drag contribution is second order in the incidence perturbation.
Adding empty channels would invite someone to fill them without deriving them.
"""

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from flightsim.aero import air_data
from flightsim.aircraft import Aircraft
from flightsim.state import State


class CoeffIncrement(NamedTuple):
    """Additions to the aerodynamic coefficients, body/wind axes as `aero.py`."""

    CL: Array
    Cl: Array
    Cm: Array
    Cn: Array


def zero_increment() -> CoeffIncrement:
    """The default. Every field is a jnp array, not a Python float.

    `integrate.batch_sim` broadcasts every leaf of the carry and a Python scalar
    has no `.shape`, which raises there -- the same constraint `WindState`
    documents for the same reason.
    """
    return CoeffIncrement(
        CL=jnp.array(0.0), Cl=jnp.array(0.0), Cm=jnp.array(0.0), Cn=jnp.array(0.0)
    )


def add(a: CoeffIncrement, b: CoeffIncrement) -> CoeffIncrement:
    """Sum two increments. Superposition, the same property that licenses
    `wind.superpose`: two fields acting at once contribute independently."""
    return CoeffIncrement(
        CL=a.CL + b.CL, Cl=a.Cl + b.Cl, Cm=a.Cm + b.Cm, Cn=a.Cn + b.Cn
    )


def strip_increment(state: State, field, ac: Aircraft, stations) -> CoeffIncrement:
    """Coefficient increment from integrating a wind field across the airframe.

    ROLL ONLY. `wind.strip_roll_moment` is the one strip integral this project
    has validated -- against the tabulated Clp for a rigid roll rate, against
    Stengel eq. 3.4-40's closed form for a rectangular wing, and against the
    equivalent-rate treatment for a linear gradient. There is no validated
    pitch or yaw integral, so those channels stay at exact zero rather than
    being filled with something plausible. Shipping an unvalidated pitch
    integral would be worse than shipping none, because it would look like
    increased fidelity.

    Airspeed is AIR-RELATIVE, taken through the same `relative_velocity` the
    rest of the model uses. Using ground speed would reintroduce precisely the
    error the air-relative design exists to prevent, and it would only reveal
    itself in a wind with a significant along-track component.

    Both imports below are local, and for two different reasons. `dynamics`
    imports `CoeffIncrement` from this module, so importing it back at module
    level closes a cycle -- and it closes it in the unrecoverable direction,
    because `relative_velocity` is defined AFTER that import in `dynamics.py`.
    `wind` imports `airframe`, which imports `aircraft`; that chain is
    currently acyclic but only by accident, and keeping it local costs nothing.
    """
    from flightsim import wind
    from flightsim.dynamics import relative_velocity

    wind_at_cg = field(state.pos_ned)
    vel_rel = relative_velocity(state.vel_body, state.quat, wind_at_cg)
    airspeed, _, _ = air_data(vel_rel)

    roll = wind.strip_roll_moment(
        state.pos_ned, state.quat, field, ac, stations, airspeed
    )
    return zero_increment()._replace(Cl=roll)
