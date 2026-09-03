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
    # ONE run per case, from the ONE paper that states its parameters.
    # `radius_source` names that paper. Until session 26 every case was flown at
    # BOTH radii; see the generator's CASE_SOURCE comment for why that stopped.
    ("cimarron", "wingrove"),
    ("hannibal", "parks"),
    ("morton", "parks"),
    # Session 23d. A different KIND of encounter from the four above: Mehta's
    # converged FIVE-vortex array with an oblique traverse, which is the field
    # the headline atisim result flies. Several tests below characterise a
    # SINGLE core and are scoped to `cores is None` rather than relaxed.
    ("mehta", "mehta"),
}

SINGLE_CORE_KEYS = EXPECTED_KEYS - {("mehta", "mehta")}


@pytest.fixture(scope="module")
def reference():
    return jsbsim_vortex_ref.load()


def test_every_expected_encounter_is_present(reference):
    """One run per case, and `radius_source` names the paper it came from.

    Cimarron is `wingrove` because Parks 1985 does not contain it; Hannibal and
    Morton are `parks` because he identifies both and states a coherent triple
    for each. A second run per case would have to cross two papers to exist.
    """
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
    assert reference.encounters[("hannibal", "parks")].aircraft == "boeing747_jsbsim"


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
        if enc.cores is None:
            north = jnp.array([enc.values["core_north"]])
            down = jnp.array([-enc.values["altitude"]])
        else:
            north, down = jnp.array(enc.cores[0]), jnp.array(enc.cores[1])
        array = wind.VortexArray(
            north=north, down=down,
            r0=jnp.array(enc.values["r0"]),
            v0=jnp.array(enc.values["v0"]),
            cos_dpsi=jnp.array(enc.values.get("cos_dpsi", 1.0)),
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
    changed to match. *** Along hannibal/parks the true airspeed RISES
    monotonically from 236.055 m/s, and it rises fastest as the core is
    approached.

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
    for key in sorted(SINGLE_CORE_KEYS):
        enc = reference.encounters[key]
        change = enc.entry_speed_drift()
        worst[key] = change
        assert change < 0.01, (
            f"{key}: the run-in moved true airspeed by {change:.2%}, more than "
            f"the 1% these encounters were characterised at. Either the lead-in "
            f"of {reference.lead_in_radii:.0f} core radii changed or the field did."
        )

    # The array encounter is NOT held to that bound, and the bound is not
    # widened to admit it. 1% characterises a 15-radius run-in to a single
    # core; the Mehta window opens at the edge of a five-core array, so the
    # aircraft has flown through four more far fields before it. Its own value
    # is recorded here so a change in it is still visible, and it is the reason
    # scripts/vortex_compare.py reports the Mehta case from the window edge.
    mehta = reference.encounters[("mehta", "mehta")].entry_speed_drift()
    assert 0.02 < mehta < 0.03, f"array preconditioning moved to {mehta:.2%}"
    # Hannibal's preconditioning at Parks' own 600 ft core, recorded so a change
    # in the field or the lead-in is visible. It was 0.70% here; the ORDERING
    # this once asserted -- a 600 ft core preconditioning MORE than a 500 ft one,
    # which was the evidence that the effect really is the far field -- needed
    # two runs of one case at two radii, and those crossed two papers to exist.
    # The evidence is in the session-22 history; the bound is here.
    assert worst[("hannibal", "parks")] == pytest.approx(0.00703, abs=1e-4)


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
    for key in sorted(SINGLE_CORE_KEYS):
        enc = reference.encounters[key]
        w = np.array([s.wind[2] for s in enc.window()])
        crossings = int(np.sum(np.diff(np.sign(w)) != 0))
        assert crossings == 1, f"{key}: {crossings} sign changes, expected 1"

    # And the negative control the four cases above could never provide: the
    # array encounter must show MORE than one, or `cores` is not being flown.
    arr = reference.encounters[("mehta", "mehta")]
    w = np.array([s.wind[2] for s in arr.window()])
    assert int(np.sum(np.diff(np.sign(w)) != 0)) > 1


def test_hannibal_flies_parks_own_radius(reference):
    """The reversal of session 22, pinned where the runs can see it.

    Parks et al. 1985 was obtained in session 26 and states r0 = 600 ft with
    V0 = 85 ft/s and 3500 ft of spacing -- one coherent identification. The
    project flew 500 ft from session 22 to 25, taken from Wingrove & Bach Fig.
    4's 1000 ft diameter while keeping Parks' strength and spacing: a vortex no
    paper states.

    The frozen XML must have been regenerated at 600 ft. One still carrying 500
    would make every number downstream describe a vortex the project no longer
    flies, and nothing else here would notice.
    """
    enc = reference.encounters[("hannibal", "parks")]
    assert enc.values["r0"] == pytest.approx(600.0 * FT2M)
    assert enc.values["r0"] == pytest.approx(wind.PARKS_CASES["hannibal"]["r0"])
    # The companion: it is NOT Fig. 4's radius, and the two dicts still differ.
    assert enc.values["r0"] != pytest.approx(
        wind.WINGROVE_FIG4_CASES["hannibal"]["r0"])
    # Morton is where the papers agree, so its run cannot distinguish them.
    morton = reference.encounters[("morton", "parks")]
    assert morton.values["r0"] == pytest.approx(
        wind.WINGROVE_FIG4_CASES["morton"]["r0"])

