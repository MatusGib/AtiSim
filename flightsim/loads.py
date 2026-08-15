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
