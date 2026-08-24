"""Checks on the aircraft data itself, not on the physics that consumes it.

The important ones are the cross-checks: where a source gives both dimensional
and non-dimensional forms of the same derivative, this module asserts that the
conversion in aircraft.py reproduces the source's own numbers. That is the only
thing standing between a transcription error and a simulator that flies
confidently and wrongly.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from atisim import trim
from atisim.aircraft import CESSNA172_TABLES, CRUISE, REGISTRY
from atisim.sensors import sense
from atisim.atmosphere import G0
from atisim.units import RAD2DEG

EVERY = sorted(REGISTRY)


@pytest.fixture(params=EVERY)
def named(request):
    name = request.param
    return name, REGISTRY[name], CRUISE[name]["airspeed"], CRUISE[name]["altitude"]


# --- the source cross-checks ------------------------------------------------


def test_cherokee_conversion_reproduces_the_sources_own_coefficients():
    """McCormick gives both forms. The dimensional set is what aircraft.py
    converts; the non-dimensional set is the independent answer it must land on.
    A sign slip or a wrong reference length shows up here and nowhere cheaper.
    """
    ac = REGISTRY["cherokee"]
    for name, derived, source in [
        ("CLa", float(ac.CLa), 4.68),  # source tabulates CZa = -4.68
        ("Cma", float(ac.Cma), -0.741),
        ("Cmq", float(ac.Cmq), -7.42),
        ("CLde", float(ac.CLde), 0.934),  # source tabulates CZde = -0.934
        ("Cmde", float(ac.Cmde), -2.4),
    ]:
        assert derived == pytest.approx(source, rel=0.01), name


def test_cherokee_weight_coefficient_matches_the_sources_trim_lift():
    """The source states CL0 = 0.543 at its condition. Level flight there needs
    mg/qS. Agreement ties mass, area, density and speed together in one check.
    """
    ac = REGISTRY["cherokee"]
    qS = 0.5 * 1.06 * 50.0**2 * float(ac.S)
    assert float(ac.mass) * G0 / qS == pytest.approx(0.543, rel=0.01)


def test_cherokee_parasite_drag_lands_in_the_band_the_source_predicts():
    """CD0 is back-solved from Xw, never read. The source independently says
    parasite CD0 for this airframe is about 0.03-0.04. Nothing forces the
    back-solve to agree, so agreement is evidence the method is right.
    """
    assert 0.030 <= float(REGISTRY["cherokee"].CD0) <= 0.040


def test_cessna_lift_and_moment_fits_track_the_source_table():
    ac = REGISTRY["cessna172"]
    alpha = np.array(CESSNA172_TABLES["alpha_deg"])
    linear = alpha <= 10.0
    a = np.deg2rad(alpha[linear])
    CL_fit = float(ac.CL0) + float(ac.CLa) * a
    Cm_fit = float(ac.Cm0) + float(ac.Cma) * a
    assert np.abs(CL_fit - np.array(CESSNA172_TABLES["CL"])[linear]).max() < 0.05
    assert np.abs(Cm_fit - np.array(CESSNA172_TABLES["Cm"])[linear]).max() < 0.05


def test_cessna_polar_reproduces_the_source_drag_table():
    ac = REGISTRY["cessna172"]
    CL = np.array(CESSNA172_TABLES["CL"])
    CD = np.array(CESSNA172_TABLES["CD"])
    linear = np.array(CESSNA172_TABLES["alpha_deg"]) <= 10.0
    fitted = float(ac.CD0) + CL[linear] ** 2 / (np.pi * float(ac.e) * float(ac.AR))
    assert np.abs(fitted - CD[linear]).max() < 0.002


def test_cessna_control_derivatives_use_the_sources_own_dynamic_pressure():
    """The dimensional control derivatives belong to the source's 67 m/s
    linearisation, not to the 60 m/s cruise chosen for this package. Recovering
    them at the wrong dynamic pressure inflates all four by 25%.
    """
    ac = REGISTRY["cessna172"]
    qS = 0.5 * 1.055 * 67.0**2 * float(ac.S)
    assert float(ac.CLde) == pytest.approx(17.19 * float(ac.mass) / qS, rel=1e-6)
    assert float(ac.Cmde) == pytest.approx(
        -36.23 * float(ac.inertia[1, 1]) / (qS * float(ac.c)), rel=1e-6
    )


def test_the_cessna_rudder_is_deliberately_inert():
    """Not an oversight. The source omits Cldr and Cndr, and its Ydr is
    inconsistent in sign and magnitude, so the whole set is zeroed rather than
    shipping a side force with no matching yawing moment.
    """
    ac = REGISTRY["cessna172"]
    assert float(ac.CYdr) == 0.0
    assert float(ac.Cldr) == 0.0
    assert float(ac.Cndr) == 0.0


def test_the_cessna_stall_table_is_kept_even_though_the_model_cannot_use_it():
    """atisim.aero is linear in alpha. The table runs to CLmax at 19.5 deg
    and is retained whole so a nonlinear path can use it without re-sourcing.
    """
    CL = np.array(CESSNA172_TABLES["CL"])
    assert CL.max() == pytest.approx(1.889)
    assert len(CESSNA172_TABLES["alpha_deg"]) == len(CL)
    for key, values in CESSNA172_TABLES.items():
        assert len(values) == len(CESSNA172_TABLES["alpha_deg"]), key


# --- physical sanity, applied to every aircraft -----------------------------


def test_signs_are_physically_correct(named):
    _, ac, _, _ = named
    assert float(ac.CLa) > 0.0
    assert float(ac.Cma) < 0.0  # statically stable in pitch
    assert float(ac.Cmq) < 0.0  # pitch damping
    assert float(ac.Cmde) < 0.0  # trailing edge down pitches nose down
    assert float(ac.CYb) < 0.0
    assert float(ac.Clb) < 0.0  # dihedral effect
    assert float(ac.Cnb) > 0.0  # directional stability
    assert float(ac.Clp) < 0.0  # roll damping
    assert float(ac.Cnr) < 0.0  # yaw damping
    assert float(ac.Clda) > 0.0
    assert 0.0 < float(ac.e) <= 1.0
    assert float(ac.CD0) > 0.0


def test_inertia_tensor_is_symmetric_positive_definite(named):
    _, ac, _, _ = named
    inertia = np.asarray(ac.inertia)
    assert np.allclose(inertia, inertia.T)
    assert (np.linalg.eigvalsh(inertia) > 0.0).all()
    assert np.allclose(inertia @ np.asarray(ac.inertia_inv), np.eye(3), atol=1e-9)


def test_geometry_is_self_consistent(named):
    _, ac, _, _ = named
    assert float(ac.AR) == pytest.approx(float(ac.b) ** 2 / float(ac.S))


def test_trim_converges_and_leaves_throttle_in_range(named):
    name, ac, V, H = named
    x, residual = trim.trim(jnp.array(V), jnp.array(H), ac)
    assert float(jnp.linalg.norm(residual)) < 1e-8, name
    alpha, elevator, throttle = (float(v) for v in x)
    assert 0.0 < throttle < 1.0, name
    assert abs(alpha) < np.deg2rad(15.0), name
    assert abs(elevator) < float(ac.elevator_limit), name


def test_the_approach_747_mass_and_aerodynamics_are_the_same_condition():
    """Two entries on different pages of CR-2144, checked against each other.

    Table IX-2 states CL = 1.11 for the power-approach point. Table IX-3's
    flight condition 2 states the mass, 564,032 lb, on a different page and in a
    different form. Neither was derived from the other, so W/qS reproducing 1.11
    is what says the mass and the aerodynamics belong to the SAME flight
    condition -- the failure this catches is pairing the approach derivative set
    with the CRUISE weight, which would otherwise trim happily and be 13% wrong.
    """
    from atisim.atmosphere import RHO0

    ac = REGISTRY["boeing747_approach"]
    V = CRUISE["boeing747_approach"]["airspeed"]
    qS = 0.5 * float(RHO0) * V * V * float(ac.S)
    CL = float(ac.mass) * 9.80665 / qS
    assert CL == pytest.approx(1.11, rel=0.005)  # table value; measured 1.1126


def test_the_approach_747_reproduces_table_ix_2_where_it_is_stated_directly():
    """Table IX-2 is already non-dimensional, so most of it is stored verbatim.

    The two that are NOT stored verbatim are the drag split: CD0 and e are solved
    from the table's CD = 0.102 and CDa = 0.66, so they are the only place a
    slip could hide. Both are recovered here from the stored values and compared
    back against the table.
    """
    ac = REGISTRY["boeing747_approach"]
    alpha0 = 5.7 * math.pi / 180.0

    for name, stored, table in [
        ("CLa", float(ac.CLa), 5.70),
        ("Cma", float(ac.Cma), -1.26),
        ("Cmq", float(ac.Cmq), -20.8),
        ("Cmde", float(ac.Cmde), -1.34),
        ("CLde", float(ac.CLde), 0.338),
        ("Clb", float(ac.Clb), -0.221),
        ("Cnb", float(ac.Cnb), 0.150),
        ("Cnr", float(ac.Cnr), -0.30),
    ]:
        assert stored == pytest.approx(table), name

    # CL and CD rebuilt at the tabulated alpha0 must return the table's values.
    pi_e_AR = math.pi * float(ac.e) * float(ac.AR)
    CL = float(ac.CL0) + float(ac.CLa) * alpha0
    assert CL == pytest.approx(1.11, rel=1e-6)
    assert float(ac.CD0) + CL**2 / pi_e_AR == pytest.approx(0.102, rel=1e-6)
    assert 2.0 * CL * float(ac.CLa) / pi_e_AR == pytest.approx(0.66, rel=1e-6)
    # Trim moment is zero at alpha0 with no elevator: the stabiliser carries it.
    assert float(ac.Cm0) + float(ac.Cma) * alpha0 == pytest.approx(0.0, abs=1e-12)


def test_the_approach_747_is_a_lighter_aeroplane_than_the_cruise_one():
    """Max landing weight against max-zero-fuel: it must not be the same set."""
    approach, cruise = REGISTRY["boeing747_approach"], REGISTRY["boeing747"]
    assert float(approach.mass) < float(cruise.mass)
    assert float(approach.mass) / float(cruise.mass) == pytest.approx(
        564032.0 / 636636.0, rel=1e-6
    )
    # Same airframe, so the geometry must be identical, not merely close.
    for field in ("S", "b", "c", "AR"):
        assert float(getattr(approach, field)) == float(getattr(cruise, field)), field


def test_the_light_aircraft_trim_below_their_linear_range_limit():
    """The Cessna table is only linear to about 10 deg, and the Cherokee has no
    stall data at all. Cruise must sit well inside that.
    """
    for name in ("cherokee", "cessna172"):
        ac = REGISTRY[name]
        x, _ = trim.trim(
            jnp.array(CRUISE[name]["airspeed"]), jnp.array(CRUISE[name]["altitude"]), ac
        )
        assert float(x[0]) * RAD2DEG < 8.0, name


def test_cruise_is_above_the_minimum_drag_speed(named):
    """Below V_md the throttle-to-speed, elevator-to-altitude pairing inverts.
    The Cherokee has only 2.7 m/s of margin, which is why its speed loop
    degrades on a large deceleration -- that is the airframe, not the gains.

    The power-approach 747 is excluded because it is not a cruise condition and
    is BELOW V_md on purpose -- see the next test, which asserts that rather than
    leaving it as a silent exemption.
    """
    name, ac, V, H = named
    if name == "boeing747_approach":
        pytest.skip("approach condition; asserted below instead")
    V_md = float(trim.minimum_drag_speed(ac, jnp.array(H)))
    assert V > V_md, name


def test_the_approach_condition_is_below_the_minimum_drag_speed():
    """The back side of the drag curve, and it is meant to be there.

    CR-2144's power-approach point is 1.4 Vs at max landing weight with 20 deg
    of flap, and 165 KTAS comes out 12 m/s BELOW this aeroplane's minimum-drag
    speed. That is not a transcription error, it is what an approach is: an
    airliner on final is flown on the back side, which is exactly why speed
    control there is a thrust job and why a windshear encounter is dangerous at
    approach speed and merely uncomfortable at cruise.

    The consequence is recorded rather than worked around: the autopilot's
    throttle-to-speed / elevator-to-altitude pairing is inverted for this entry,
    so it holds trim but should not be trusted through a large speed excursion.
    The microburst work flies it open loop, which sidesteps the question.
    """
    ac = REGISTRY["boeing747_approach"]
    V = CRUISE["boeing747_approach"]["airspeed"]
    H = CRUISE["boeing747_approach"]["altitude"]
    V_md = float(trim.minimum_drag_speed(ac, jnp.array(H)))
    assert V < V_md
    assert V_md - V == pytest.approx(12.2, abs=0.5)  # measured 12.18 m/s


def test_wave_drag_is_inactive_for_the_light_aircraft():
    from atisim.aero import wave_drag
    from atisim.atmosphere import speed_of_sound

    for name in ("cherokee", "cessna172"):
        ac = REGISTRY[name]
        V, H = CRUISE[name]["airspeed"], CRUISE[name]["altitude"]
        mach = V / float(speed_of_sound(jnp.array(H)))
        assert mach < 0.25, name
        assert float(wave_drag(jnp.array(mach), jnp.array(1.0), ac)) == 0.0, name


def test_every_aircraft_has_its_own_gains():
    """A 747's gains will not fly a light aircraft, so every entry in REGISTRY
    needs both sets or fly.py raises KeyError on selection.
    """
    from atisim.autopilot import GAINS
    from atisim.manual import MANUAL_GAINS

    assert set(GAINS) == set(REGISTRY)
    assert set(MANUAL_GAINS) == set(REGISTRY)


def test_each_aircraft_is_flown_by_its_own_gains(named):
    """Engage at the trimmed condition and hold it: the end-to-end check that
    the data, the trim solve and the gain set agree with each other.
    """
    import jax.numpy as jnp

    from atisim import autopilot as ap_mod
    from atisim import integrate

    name, ac, V, H = named
    gains = ap_mod.GAINS[name]
    x, _ = trim.trim(jnp.array(V), jnp.array(H), ac)
    state = trim.trimmed_state(x[0], jnp.array(V), jnp.array(H))
    controls = trim.trimmed_controls(x[1], x[2])
    targets = ap_mod.Targets(
        altitude=jnp.array(H), heading=jnp.array(0.0), airspeed=jnp.array(V)
    )
    ap = ap_mod.engage(sense(state), controls, targets, gains, ac)
    (_, _), (hist, ctrl) = ap_mod.closed_loop_rollout(
        integrate.init_sim(state, jax.random.PRNGKey(0)),
        ap, targets, gains, jnp.array(0.02), ac, 3000,
    )
    # Bumpless: the first output is the trimmed deflection to the last bit.
    assert float(ctrl.elevator[0]) == pytest.approx(float(x[1]), abs=1e-12), name
    altitude = -np.asarray(hist.pos_ned)[:, 2]
    assert np.abs(altitude - H).max() < 1.0, name


def test_each_aircraft_captures_an_altitude_step(named):
    import jax.numpy as jnp

    from atisim import autopilot as ap_mod
    from atisim import integrate

    name, ac, V, H = named
    gains = ap_mod.GAINS[name]
    step = 300.0 if name == "boeing747" else 150.0
    x, _ = trim.trim(jnp.array(V), jnp.array(H), ac)
    state = trim.trimmed_state(x[0], jnp.array(V), jnp.array(H))
    controls = trim.trimmed_controls(x[1], x[2])
    targets = ap_mod.Targets(
        altitude=jnp.array(H + step), heading=jnp.array(0.0), airspeed=jnp.array(V)
    )
    ap = ap_mod.engage(sense(state), controls, targets, gains, ac)
    (_, _), (hist, _) = ap_mod.closed_loop_rollout(
        integrate.init_sim(state, jax.random.PRNGKey(0)),
        ap, targets, gains, jnp.array(0.02), ac, int(300.0 / 0.02),
    )
    altitude = -np.asarray(hist.pos_ned)[:, 2]
    assert abs(altitude[-1] - (H + step)) < 10.0, name
    assert altitude.max() < H + step + 40.0, name


def test_each_aircraft_captures_a_heading_step(named):
    import jax.numpy as jnp

    from atisim import autopilot as ap_mod
    from atisim import integrate
    from atisim.state import quat_to_euler

    name, ac, V, H = named
    gains = ap_mod.GAINS[name]
    x, _ = trim.trim(jnp.array(V), jnp.array(H), ac)
    state = trim.trimmed_state(x[0], jnp.array(V), jnp.array(H))
    controls = trim.trimmed_controls(x[1], x[2])
    targets = ap_mod.Targets(
        altitude=jnp.array(H),
        heading=jnp.array(np.deg2rad(30.0)),
        airspeed=jnp.array(V),
    )
    ap = ap_mod.engage(sense(state), controls, targets, gains, ac)
    (_, _), (hist, _) = ap_mod.closed_loop_rollout(
        integrate.init_sim(state, jax.random.PRNGKey(0)),
        ap, targets, gains, jnp.array(0.02), ac, int(300.0 / 0.02),
    )
    euler = np.asarray(jax.vmap(quat_to_euler)(hist.quat))
    altitude = -np.asarray(hist.pos_ned)[:, 2]
    assert abs(euler[-1, 2] * RAD2DEG - 30.0) < 1.5, name
    assert np.abs(euler[:, 0]).max() <= float(gains.phi_limit) + 0.05, name
    assert np.abs(altitude - H).max() < 45.0, name


def test_every_aircraft_holds_its_trimmed_condition_for_60_s(named):
    from atisim import integrate

    name, ac, V, H = named
    x, _ = trim.trim(jnp.array(V), jnp.array(H), ac)
    state = trim.trimmed_state(x[0], jnp.array(V), jnp.array(H))
    controls = trim.trimmed_controls(x[1], x[2])
    _, hist = integrate.rollout(
        integrate.init_sim(state, jax.random.PRNGKey(0)),
        controls,
        jnp.array(0.02),
        ac,
        3000,
    )
    altitude = -np.asarray(hist.pos_ned)[:, 2]
    speed = np.linalg.norm(np.asarray(hist.vel_body), axis=1)
    assert np.abs(altitude - H).max() < 1.0, name
    assert np.abs(speed - V).max() < 0.5, name
