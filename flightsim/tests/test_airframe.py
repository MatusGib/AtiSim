"""Airframe sampling geometry.

The one new relation here is the effective tail arm, l_eff/c = -Cmq/CLq. It
comes from Stengel Flight Dynamics 2nd ed eqs. 3.4-7 and 3.4-12, whose ratio
cancels the tail lift slope. Its justification is empirical -- the value it
returns for the 747 falls inside the real aircraft's tail arm -- so that
check is asserted here rather than described in a comment.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import flightsim  # noqa: F401  -- enables x64 before any array is made
from flightsim import airframe
from flightsim.aircraft import REGISTRY
from flightsim.units import FT2M


def test_the_derived_tail_arm_matches_the_hand_computation():
    """CR-2144 FC9 gives CLq = 5.9450 and Cmq = -23.9232, so -Cmq/CLq = 4.0241
    chords. Asserted as a value so a change to either derivative shows up here
    rather than silently moving every sampled gradient."""
    ac = REGISTRY["boeing747"]
    assert float(airframe.effective_tail_arm(ac)) == pytest.approx(4.0241, rel=1e-4)


def test_the_derived_tail_arm_lands_on_the_real_aircraft_geometry():
    """THE justification for the whole relation. The 747-100's centre of gravity
    sits roughly 100-110 ft ahead of the tailplane, and this recovers 109.9 ft
    having been told nothing about 747 geometry beyond the mean chord. If this
    stops holding, l_eff is no longer defensible and the strip path loses its
    only empirical support -- see design section 7d."""
    ac = REGISTRY["boeing747"]
    arm_ft = float(airframe.effective_tail_arm(ac) * ac.c) / FT2M
    assert 100.0 <= arm_ft <= 110.0, f"derived arm {arm_ft:.1f} ft is outside the real aircraft's"


def test_only_the_two_747_configurations_pass_the_plausibility_gate():
    """The gate must be able to fire, and it fires on half the registry.

    Measured across every aircraft the project holds:

        boeing747           4.0241 chords = 109.90 ft   PASS
        boeing747_approach  3.8519 chords = 105.19 ft   PASS
        cessna172           0.8558 chords =   4.19 ft   FAIL
        cherokee            1.2802 chords =   6.72 ft   FAIL

    Both 747 sets come from CR-2144, whose transcription has been verified
    element by element against the source document. Both light-aircraft sets
    return arms far shorter than those airframes physically have.

    WHAT THIS DOES NOT ESTABLISH is which side is at fault. It may be the source
    data -- PROJECT.md section 3 already records the Cessna's rudder set as
    inconsistent and the Cherokee's Izz < Iyy as flagged by its own author -- or
    it may be that the tail-dominated reading behind l_eff does not transfer to
    a light aircraft with a short tail and a large wing. Either way the strip
    path must not be used for them, which is exactly what the gate enforces.
    """
    passes = {name: airframe.tail_arm_is_plausible(REGISTRY[name]) for name in REGISTRY}
    assert passes == {
        "boeing747": True,
        "boeing747_approach": True,
        "cessna172": False,
        "cherokee": False,
    }, f"gate outcome changed: {passes}"


def test_the_derived_arms_take_their_recorded_values():
    """Asserted per aircraft so a change to any CLq or Cmq surfaces here rather
    than silently moving every sampled gradient downstream."""
    expected = {
        "boeing747": 4.0241,
        "boeing747_approach": 3.8519,
        "cessna172": 0.8558,
        "cherokee": 1.2802,
    }
    for name, arm in expected.items():
        assert float(airframe.effective_tail_arm(REGISTRY[name])) == pytest.approx(
            arm, rel=1e-3
        ), f"{name} derived arm moved"
