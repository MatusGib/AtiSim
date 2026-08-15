"""Aerodynamic coefficient increments from distributed loads.

The seam that lets strip-integrated loads reach the equations of motion. It is
deliberately a NamedTuple of named coefficients rather than a bare array: the
four entries are not interchangeable, and a caller that swaps roll for yaw
would otherwise produce a model that flies plausibly and wrongly.

The zero increment is the important case. It is what every existing run uses,
and `test_dynamics.py` asserts that applying it is bit-identical to not
applying one at all.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import flightsim  # noqa: F401  -- enables x64 before any array is made
from flightsim import loads


def test_the_zero_increment_is_all_zeros_and_correctly_shaped():
    """Every field must be a jnp array, not a Python float. integrate.batch_sim
    broadcasts every leaf of the carry and a Python scalar has no .shape, which
    raises there -- the same constraint WindState documents."""
    zero = loads.zero_increment()
    for name, value in zero._asdict().items():
        assert hasattr(value, "shape"), f"{name} is not a jnp array"
        assert float(value) == 0.0, f"{name} is not zero"


def test_the_increment_names_the_four_coefficients_it_carries():
    """Named, not positional. Lift, roll, pitch, yaw -- the four a spanwise and
    longitudinal load distribution can produce that this model has channels for.
    Side force and drag are omitted deliberately: strip theory over a spanwise
    incidence distribution does not produce them at first order."""
    assert loads.CoeffIncrement._fields == ("CL", "Cl", "Cm", "Cn")


def test_increments_add():
    """Superposition. Two fields acting at once contribute independently, the
    same property that makes wind.superpose legitimate."""
    a = loads.CoeffIncrement(
        CL=jnp.array(0.1), Cl=jnp.array(0.2), Cm=jnp.array(0.3), Cn=jnp.array(0.4)
    )
    b = loads.CoeffIncrement(
        CL=jnp.array(0.5), Cl=jnp.array(-0.2), Cm=jnp.array(0.0), Cn=jnp.array(0.1)
    )
    total = loads.add(a, b)
    assert float(total.CL) == pytest.approx(0.6)
    assert float(total.Cl) == pytest.approx(0.0)
    assert float(total.Cm) == pytest.approx(0.3)
    assert float(total.Cn) == pytest.approx(0.5)


def test_adding_zero_changes_nothing_exactly():
    """Not approximately. This is what makes the default path bit-identical."""
    a = loads.CoeffIncrement(
        CL=jnp.array(0.123), Cl=jnp.array(-0.456), Cm=jnp.array(0.789), Cn=jnp.array(0.0)
    )
    total = loads.add(a, loads.zero_increment())
    for name in a._fields:
        assert np.asarray(getattr(total, name)) == np.asarray(getattr(a, name))
