"""The frozen JSBSim vortex reference, and whether it can be compared at all.

Most of this file is not about the answer. It is about the preconditions: that
both engines flew the same air, that neither had drifted out of trim before it
reached the core, and that the wind actually reached the aerodynamics. A
comparison that fails any of those produces numbers that look exactly like a
result and are not one.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from atisim import jsbsim_vortex_ref, wind
from atisim.aircraft import REGISTRY
from atisim.units import FT2M

EXPECTED_KEYS = {
    ("cimarron", "wingrove"),
    ("hannibal", "wingrove"), ("hannibal", "parks"),
    ("morton", "wingrove"), ("morton", "parks"),
}


@pytest.fixture(scope="module")
def reference():
    return jsbsim_vortex_ref.load()


def test_every_expected_encounter_is_present(reference):
    """Cimarron has no `parks` run because Parks 1985 does not contain it."""
    assert set(reference.encounters) == EXPECTED_KEYS


def test_each_case_flew_the_airframe_whose_band_contains_it(reference):
    """The whole reason the cases are split across two airframes.

    Cimarron at 33,000 ft is inside the 737 entry's declared band and the other
    two are not; Hannibal and Morton are inside boeing747_jsbsim's. A case flown
    by the wrong airframe would be extrapolating a local fit and the numbers
    would be wrong in a way nothing else here would catch.
    """
    for key, enc in reference.encounters.items():
        ac = REGISTRY[enc.aircraft]
        low, high = (float(v) for v in ac.valid_altitude)
        altitude = enc.values["altitude"]
        assert low <= altitude <= high, (
            f"{key}: {enc.aircraft} flew at {altitude / FT2M:.0f} ft, outside "
            f"its band of [{low / FT2M:.0f}, {high / FT2M:.0f}] ft"
        )
    assert reference.encounters[("cimarron", "wingrove")].aircraft == "boeing737"
    assert reference.encounters[("hannibal", "wingrove")].aircraft == "boeing747_jsbsim"


def test_the_gradient_was_not_injected_into_jsbsim(reference):
    """JSBSim has no writable gust-rate input, and the reference records it.

    atmosphere/{p,q,r}-turb-rad_sec are READ-ONLY in JSBSim 1.3.1's property
    catalog and a write to q-turb-rad_sec reads back 0.0 after a step. If this
    flag is ever true, the comparison's central asymmetry has changed and the
    report's claim about it is stale.
    """
    assert reference.gradient_injected is False


def test_the_wind_actually_reached_the_aerodynamics(reference):
    """The silent failure mode: a run where the injection quietly did nothing.

    Everything downstream would still produce plausible-looking time histories,
    of an aircraft flying through still air.
    """
    for key, enc in reference.encounters.items():
        peak = max(abs(s.wind[2]) for s in enc.samples)
        assert peak > 0.9 * enc.values["v0"], (
            f"{key}: peak |w_down| {peak:.2f} m/s against a core strength of "
            f"{enc.values['v0']:.2f} m/s -- the aircraft never reached the core"
        )
        span = max(s.Nz for s in enc.samples) - min(s.Nz for s in enc.samples)
        assert span > 0.2, f"{key}: Nz moved only {span:.4f} g"


def test_injected_field_matches_atisims_field_at_every_sample(reference):
    """*** The gate the whole comparison rests on: both engines flew ONE field.

    The generator recomputes the Rankine field in plain numpy rather than
    calling atisim.wind.vortex_wind, precisely so that a bug in that function
    cannot hide by being present on both sides of the comparison. This is where
    the two independent implementations are reconciled, and if it fails nothing
    downstream means anything.
    """
    worst_overall, worst_key = 0.0, None
    for key, enc in reference.encounters.items():
        array = wind.VortexArray(
            north=jnp.array([enc.values["core_north"]]),
            down=jnp.array([-enc.values["altitude"]]),
            r0=jnp.array(enc.values["r0"]),
            v0=jnp.array(enc.values["v0"]),
        )
        for s in enc.samples:
            mine = np.asarray(
                wind.vortex_wind(jnp.array([s.north, 0.0, -s.altitude]), array)
            )
            worst = float(np.abs(mine - s.wind).max())
            if worst > worst_overall:
                worst_overall, worst_key = worst, key
    assert worst_overall < 1e-9, (
        f"the two independently-written vortex fields disagree by "
        f"{worst_overall:.3e} m/s, worst at {worst_key}"
    )


def test_far_field_preconditioning_is_bounded_and_is_not_trim_decay(reference):
    """How much the run-in changes the state before the core is reached.

    *** This test was first written as a 0.5% "drift" bound, on the assumption
    that fixed-control flight SHEDS airspeed and that any loss was a trim
    imbalance to be minimised. Measurement says otherwise and the test was
    changed to match. *** Along hannibal/wingrove the true airspeed runs
    236.055 -> 236.334 -> 236.814 -> 236.942 -> 237.632 m/s: it RISES,
    monotonically, and it rises fastest as the core is approached.

    That is the vortex's own 1/r far field doing work on the aircraft, not the
    engine failing to hold trim. It is physical, it is present in BOTH engines,
    and shortening the lead-in would not remove it -- it would only expose less
    of it while also pushing the start into the region where scripts/vortex.py
    records the far field launching the aircraft out of equilibrium.

    So the quantity is bounded and recorded rather than driven to zero. The real
    question -- do the two engines arrive at the core in the SAME state -- is an
    inter-engine comparison and cannot be asked of this file, which holds only
    one engine's run. scripts/vortex_compare.py asks it.
    """
    worst = {}
    for key, enc in reference.encounters.items():
        change = enc.entry_speed_drift()
        worst[key] = change
        assert change < 0.01, (
            f"{key}: the run-in moved true airspeed by {change:.2%}, more than "
            f"the 1% these encounters were characterised at. Either the lead-in "
            f"of {reference.lead_in_radii:.0f} core radii changed or the field did."
        )
    # The far field is stronger for a bigger core at the same strength, so the
    # 600 ft Hannibal must precondition more than the 500 ft one. If that
    # ordering ever inverts, the effect is not the far field.
    assert worst[("hannibal", "parks")] > worst[("hannibal", "wingrove")]


def test_the_encounter_is_a_single_core_as_fig_4_draws_it(reference):
    """One core, not the Parks array, and that is a deliberate choice.

    Fig. 4 draws a single circle with the flight path through it and shows one
    up-spike followed by one down-spike -- a single-core traverse. Parks' own
    identification is an ARRAY, and scripts/vortex.py flies two cores at the
    identified spacing; that is the right model for comparing against Parks and
    the wrong one here, because a second core adds a second event whose overlap
    with the first depends on a spacing Fig. 4 does not supply, and Cimarron has
    no published spacing at all.

    Asserted through the wind trace: a single Rankine core gives exactly one
    sign change in the vertical gust across the traverse.
    """
    for key, enc in reference.encounters.items():
        w = np.array([s.wind[2] for s in enc.window()])
        crossings = int(np.sum(np.diff(np.sign(w)) != 0))
        assert crossings == 1, f"{key}: {crossings} sign changes, expected 1"


def test_morton_is_a_control_because_both_sources_agree_on_it(reference):
    """Morton's two runs must be IDENTICAL, and that is the point of running it.

    Fig. 4 and PARKS_CASES both give Morton a 450 ft radius, so its `wingrove`
    and `parks` encounters are the same physical case flown twice. Any
    difference between them would be numerical noise in the harness rather than
    a radius effect -- which is exactly the null this comparison needs, because
    Hannibal's two runs DO differ and something has to establish that the
    difference is the radius and not the machinery.
    """
    a = reference.encounters[("morton", "wingrove")]
    b = reference.encounters[("morton", "parks")]
    assert a.values["r0"] == b.values["r0"]
    assert a.load_increments() == b.load_increments()
    assert a.pitch_increments() == b.pitch_increments()


def test_hannibals_two_radii_give_genuinely_different_answers(reference):
    """The unresolved conflict, measured rather than argued.

    Fig. 4 gives Hannibal a 500 ft radius and PARKS_CASES 600 ft, and Parks 1985
    has never been obtained to settle it. This asserts the disagreement MATTERS
    -- if the two radii ever produced the same load increment, the conflict
    would be reportable as harmless, and it is not.
    """
    wingrove = reference.encounters[("hannibal", "wingrove")]
    parks = reference.encounters[("hannibal", "parks")]
    assert wingrove.values["r0"] == pytest.approx(500.0 * FT2M)
    assert parks.values["r0"] == pytest.approx(600.0 * FT2M)
    _, wingrove_min = wingrove.load_increments()
    _, parks_min = parks.load_increments()
    spread = abs(wingrove_min - parks_min) / abs(parks_min)
    assert spread > 0.05, (
        f"the two Hannibal radii differ by only {spread:.1%} in peak negative "
        "load; the conflict would then be reportable as immaterial"
    )
