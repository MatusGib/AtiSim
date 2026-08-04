"""Aircraft definitions.

Everything stored here is SI, per-radian, and body-axis. Conversions from the
source documents happen at the point of definition using flightsim.units, each
with a comment naming the table it came from.

`Aircraft` is a NamedTuple so it is a JAX pytree: it can be passed straight
through jit and vmap without static_argnums. The name is deliberately NOT a
field -- a string leaf would break tracing -- so aircraft are looked up in the
REGISTRY dict at the bottom of this module.
"""

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array


class Aircraft(NamedTuple):
    # Mass and geometry
    mass: Array  # kg
    inertia: Array  # (3,3) kg.m^2, body axes, including Ixz
    inertia_inv: Array  # (3,3), precomputed
    S: Array  # m^2, reference wing area
    b: Array  # m, span
    c: Array  # m, mean aerodynamic chord

    # Drag polar: CD = CD0 + CL^2 / (pi * e * AR)
    CD0: Array
    e: Array  # Oswald efficiency
    AR: Array  # b^2 / S

    # Longitudinal
    CL0: Array
    CLa: Array
    CLq: Array
    CLde: Array
    Cm0: Array
    Cma: Array
    Cmq: Array
    Cmde: Array

    # Lateral-directional
    CYb: Array
    CYp: Array
    CYr: Array
    CYdr: Array
    Clb: Array
    Clp: Array
    Clr: Array
    Clda: Array
    Cldr: Array
    Cnb: Array
    Cnp: Array
    Cnr: Array
    Cnda: Array
    Cndr: Array

    # Propulsion: thrust = throttle * max_thrust * (rho/rho0)^thrust_lapse
    max_thrust: Array  # N, sea-level static
    thrust_lapse: Array  # density-ratio exponent

    # Control deflection limits (rad), used by trim bounds and the autopilot
    elevator_limit: Array
    aileron_limit: Array
    rudder_limit: Array


def inertia_tensor(Ixx, Iyy, Izz, Ixz) -> Array:
    """Body-axis inertia tensor.

    Ixz is the product of inertia in its usual positive-forward-up sense, so it
    enters the tensor negated. It is not negligible for the 747.
    """
    return jnp.array(
        [
            [Ixx, 0.0, -Ixz],
            [0.0, Iyy, 0.0],
            [-Ixz, 0.0, Izz],
        ]
    )


def stability_to_body(Cl_s, Cn_s, alpha_ref):
    """Rotate rolling/yawing moment derivatives from stability to body axes.

    Stability axes are body axes rotated about y by the reference angle of
    attack, so the x and z components mix.
    """
    ca, sa = jnp.cos(alpha_ref), jnp.sin(alpha_ref)
    return Cl_s * ca - Cn_s * sa, Cn_s * ca + Cl_s * sa


REGISTRY: dict[str, Aircraft] = {}
