"""Shared test fixtures."""

import jax
import jax.numpy as jnp
import pytest

import flightsim  # noqa: F401  -- enables x64 before any array is made
from flightsim.aircraft import Aircraft, inertia_tensor

# NaN guard. A NaN produced inside jit is silent and propagates for thousands of
# steps before anything looks wrong. On for the whole suite: it costs a little
# runtime and buys back hours.
jax.config.update("jax_debug_nans", True)


def make_test_aircraft(Ixz: float = 0.0) -> Aircraft:
    """A SYNTHETIC light-aircraft-shaped set for exercising the physics plumbing.

    These are NOT researched values and must never be used for results. The real,
    cited aircraft live in flightsim/aircraft.py. This exists so that the aero
    and dynamics tests do not depend on the aircraft data being finished, and so
    that a later correction to the real data cannot silently break them.

    Signs follow the conventions documented in flightsim/aero.py.
    """
    b, S = 10.2, 17.1
    inertia = inertia_tensor(1420.0, 4070.0, 4780.0, Ixz)
    return Aircraft(
        mass=jnp.array(1250.0),
        inertia=inertia,
        inertia_inv=jnp.linalg.inv(inertia),
        S=jnp.array(S),
        b=jnp.array(b),
        c=jnp.array(1.74),
        CD0=jnp.array(0.05),
        e=jnp.array(0.8),
        AR=jnp.array(b * b / S),
        sweep=jnp.array(0.0),  # unswept: wave drag stays zero at light-aircraft Mach
        t_over_c=jnp.array(0.12),
        kappa_airfoil=jnp.array(0.87),
        CL0=jnp.array(0.41),
        CLa=jnp.array(4.44),
        CLq=jnp.array(3.8),
        CLde=jnp.array(0.355),
        Cm0=jnp.array(0.05),
        Cma=jnp.array(-0.683),
        Cmq=jnp.array(-9.96),
        Cmde=jnp.array(-0.923),
        CYb=jnp.array(-0.564),
        CYp=jnp.array(-0.0192),
        CYr=jnp.array(0.335),
        CYdr=jnp.array(0.157),
        Clb=jnp.array(-0.074),
        Clp=jnp.array(-0.410),
        Clr=jnp.array(0.107),
        Clda=jnp.array(0.134),
        Cldr=jnp.array(0.0107),
        Cnb=jnp.array(0.071),
        Cnp=jnp.array(-0.0575),
        Cnr=jnp.array(-0.125),
        Cnda=jnp.array(-0.0035),
        Cndr=jnp.array(-0.072),
        max_thrust=jnp.array(2800.0),
        thrust_lapse=jnp.array(1.0),
        elevator_limit=jnp.array(0.35),
        aileron_limit=jnp.array(0.35),
        rudder_limit=jnp.array(0.35),
    )


@pytest.fixture
def test_aircraft() -> Aircraft:
    return make_test_aircraft()
