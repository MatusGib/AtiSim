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

import atisim  # noqa: F401  -- enables x64 before any array is made
from atisim import earth, loads

# 47N is the latitude the rest of this project's Earth-rotation work uses, and
# the anchor sits AT the flight altitude -- so a state built at `pos_ned = 0` is
# level flight at 11,278 m and the spanwise fields below are sampled on the
# aircraft's own plane rather than a cruise altitude away from it.
ANCHOR = earth.anchor_at(np.radians(47.0), 0.0, 11278.0)


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


# --- building an increment from a field ------------------------------------


def test_the_strip_increment_carries_roll_only_for_now():
    """Only the rolling moment has a validated strip integral. Pitch and yaw
    are left at exact zero rather than filled with something plausible -- the
    plan that added the roll integral deliberately stopped there, and an
    unvalidated pitch integral would be worse than none.

    Asserted so that if someone later fills those channels, they have to change
    this test and therefore have to justify it.

    `strip_roll_moment` takes a body -> NED MATRIX now. `state.quat` is
    body -> ECEF and both are valid rotations, so passing it would have carried
    the span stations into the wrong frame and returned a plausible wrong
    moment rather than raising.
    """
    from atisim import airframe, wind
    from atisim.aircraft import REGISTRY
    from atisim.state import dcm_body_to_ned, euler_to_quat, pos_ned, state_from_ned

    ac = REGISTRY["boeing747"]
    st = airframe.stations(ac, n_span=201, n_lon=9)
    field = lambda p: jnp.array([0.0, 0.0, 1e-7 * p[1] ** 3])  # noqa: E731
    state = state_from_ned(
        jnp.zeros(3),
        jnp.array([236.0, 0.0, 0.0]),
        euler_to_quat(jnp.array(0.0), jnp.array(0.0), jnp.array(0.0)),
        jnp.zeros(3),
        ANCHOR,
    )

    inc = loads.strip_increment(state, field, ac, st, ANCHOR)
    expected = wind.strip_roll_moment(
        pos_ned(state, ANCHOR), dcm_body_to_ned(state, ANCHOR), field, ac, st, 236.0
    )
    assert float(inc.Cl) == pytest.approx(float(expected), rel=1e-9)
    assert float(inc.CL) == 0.0
    assert float(inc.Cm) == 0.0
    assert float(inc.Cn) == 0.0


def test_the_strip_increment_uses_air_relative_speed_not_ground_speed():
    """The incidence a strip sees is set by the speed of the air over it. Using
    inertial speed would reintroduce exactly the error the whole air-relative
    design exists to avoid, and it would only show up in a headwind."""
    from atisim import airframe
    from atisim.aircraft import REGISTRY
    from atisim.state import euler_to_quat, state_from_ned

    ac = REGISTRY["boeing747"]
    st = airframe.stations(ac, n_span=201, n_lon=9)
    field = lambda p: jnp.array([50.0, 0.0, 1e-7 * p[1] ** 3])  # noqa: E731
    base = state_from_ned(
        jnp.zeros(3),
        jnp.array([236.0, 0.0, 0.0]),
        euler_to_quat(jnp.array(0.0), jnp.array(0.0), jnp.array(0.0)),
        jnp.zeros(3),
        ANCHOR,
    )
    # A 50 m/s tailwind component leaves ground speed alone and reduces
    # airspeed, so an air-relative implementation must give a LARGER incidence
    # and therefore a larger rolling moment than a ground-speed one would.
    inc = loads.strip_increment(base, field, ac, st, ANCHOR)
    still = loads.strip_increment(
        base, lambda p: jnp.array([0.0, 0.0, 1e-7 * p[1] ** 3]), ac, st, ANCHOR
    )
    assert abs(float(inc.Cl)) > abs(float(still.Cl))


# --- the convenience wrapper, and the gate it enforces ---------------------


def test_the_strip_model_refuses_an_aircraft_that_fails_the_tail_arm_gate():
    """The gate exists to stop the strip path being used where its key input is
    not trustworthy. It must fire at construction, loudly, rather than silently
    producing numbers -- a run that quietly used a 0.856-chord tail arm would be
    very hard to spot afterwards."""
    from atisim import airframe
    from atisim.aircraft import REGISTRY

    for name in ("cessna172", "cherokee"):
        ac = REGISTRY[name]
        assert not airframe.tail_arm_is_plausible(ac)
        with pytest.raises(ValueError, match="tail arm"):
            loads.strip_model(lambda p: p * 0.0, ac, ANCHOR)


def test_the_strip_model_accepts_both_747_configurations():
    from atisim.aircraft import REGISTRY

    for name in ("boeing747", "boeing747_approach"):
        model = loads.strip_model(lambda p: p * 0.0, REGISTRY[name], ANCHOR)
        assert callable(model)
