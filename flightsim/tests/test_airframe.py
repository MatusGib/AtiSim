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


def test_span_stations_cover_the_whole_span_symmetrically():
    """The lateral extent is the span, which is sourced. Symmetry matters: an
    asymmetric station set would give a non-zero fitted roll gradient in a
    uniform field, which is the first reduction property Task 5 asserts."""
    ac = REGISTRY["boeing747"]
    st = airframe.stations(ac)
    span = np.asarray(st.span)
    assert span.min() == pytest.approx(-float(ac.b) / 2.0)
    assert span.max() == pytest.approx(float(ac.b) / 2.0)
    assert np.allclose(span, -span[::-1]), "span stations must be symmetric about the centreline"


def test_longitudinal_stations_run_from_the_tail_to_the_cg():
    """Body x is positive forward, so the tail is at NEGATIVE x. Pitch damping
    comes overwhelmingly from the tail, so the fit is taken over the CG-to-tail
    interval rather than symmetrically about the CG -- that is the interval the
    aerodynamics actually integrate over."""
    ac = REGISTRY["boeing747"]
    st = airframe.stations(ac)
    lon = np.asarray(st.longitudinal)
    arm = float(airframe.effective_tail_arm(ac) * ac.c)
    assert lon.min() == pytest.approx(-arm)
    assert lon.max() == pytest.approx(0.0)


def test_station_counts_are_configurable_for_the_convergence_study():
    """The count is DECLARED and needs a refinement study, so it must be a knob."""
    ac = REGISTRY["boeing747"]
    st = airframe.stations(ac, n_span=21, n_lon=15)
    assert np.asarray(st.span).shape == (21,)
    assert np.asarray(st.longitudinal).shape == (15,)


def test_the_default_station_count_has_converged():
    """N_SPAN and N_LON are DECLARED, and the ledger says they are chosen by
    convergence rather than taste. This is that study, run as an assertion so
    the claim cannot rot.

    The field is the Parks Hannibal core, which is the smallest-scale field the
    project holds and therefore the hardest case. Doubling the station count
    must move the fitted rates by less than 0.1%.
    """
    from flightsim import wind
    from flightsim.state import State, euler_to_quat

    ac = REGISTRY["boeing747"]
    array = wind.VortexArray(
        north=jnp.array([0.0]),
        down=jnp.array([-11278.0]),
        r0=jnp.array(600.0 * FT2M),
        v0=jnp.array(85.0 * FT2M),
    )
    field = lambda p: wind.vortex_wind(p, array)  # noqa: E731
    # Half a core radius downstream and half a radius above: inside the core but
    # off-centre, so every gradient component is non-zero.
    state = State(
        pos_ned=jnp.array([0.5 * 600.0 * FT2M, 0.0, -(11278.0 + 300.0 * FT2M)]),
        vel_body=jnp.array([236.0, 0.0, 0.0]),
        quat=euler_to_quat(jnp.array(0.0), jnp.array(0.0), jnp.array(0.0)),
        omega=jnp.zeros(3),
    )

    coarse = wind.sampled_rates(
        state.pos_ned, state.quat, field, airframe.stations(ac, 9, 9)
    )
    fine = wind.sampled_rates(
        state.pos_ned, state.quat, field, airframe.stations(ac, 18, 18)
    )
    q_coarse, q_fine = float(coarse[1]), float(fine[1])
    movement = abs(q_fine - q_coarse) / abs(q_fine)
    assert movement < 1e-3, (
        f"pitch rate moves {movement:.2%} between 9 and 18 stations; "
        "the default count has not converged"
    )
