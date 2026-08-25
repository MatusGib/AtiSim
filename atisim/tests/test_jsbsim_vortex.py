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
    # Hannibal's two runs are now the same radius, so they must precondition
    # identically. Before session 22 this asserted an ORDERING -- the 600 ft
    # core preconditioning more than the 500 ft one, which it did -- and that
    # ordering was the evidence the effect really is the far field. What
    # survives of it is the equality.
    assert worst[("hannibal", "parks")] == worst[("hannibal", "wingrove")]


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


def test_both_sources_now_agree_on_every_case_so_every_pair_is_identical(reference):
    """After session 22 the two radius sources give the same number everywhere.

    Morton always agreed -- Fig. 4's 900 ft diameter halves to PARKS_CASES'
    450 ft. Hannibal did not, until session 22 adopted Fig. 4's 500 ft there
    too. So the `radius_source` dimension is now degenerate, and every pair of
    runs is the same case flown twice.

    That is worth keeping rather than deleting, for two reasons. It records that
    two sources were consulted and what each said. And a pair of runs that must
    be bit-identical is a determinism check on the whole harness -- generator,
    frozen XML, parser and analysis -- which nothing else here provides.
    """
    for case in ("hannibal", "morton"):
        a = reference.encounters[(case, "wingrove")]
        b = reference.encounters[(case, "parks")]
        assert a.values["r0"] == b.values["r0"], case
        assert a.load_increments() == b.load_increments(), case
        assert a.pitch_increments() == b.pitch_increments(), case
        assert a.core_response() == b.core_response(), case


def test_hannibal_now_flies_fig_4s_radius(reference):
    """The decision of session 22, pinned where the runs can see it.

    Fig. 4 gives Hannibal a 1000 ft core diameter, so 500 ft of radius; the
    superseded transcription from Parks 1985 said 600 ft. The reference must
    have been regenerated at the new value -- an XML still carrying 600 ft would
    make every number downstream describe a vortex the project no longer flies,
    and nothing else here would notice.
    """
    for source in ("wingrove", "parks"):
        enc = reference.encounters[("hannibal", source)]
        assert enc.values["r0"] == pytest.approx(500.0 * FT2M), source
        assert enc.values["r0"] == pytest.approx(
            wind.PARKS_CASES["hannibal"]["r0"]
        ), source
        assert enc.values["r0"] != pytest.approx(wind.HANNIBAL_R0_SUPERSEDED)
