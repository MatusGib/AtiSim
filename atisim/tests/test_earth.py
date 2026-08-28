"""WGS-84 geodesy, gravity, and the rotating-Earth frames.

Every constant here was recovered from the running JSBSim binary, not quoted
from a table -- see the design doc section 2. The tests below assert the
DEFINING values; the term-by-term agreement with JSBSim is asserted separately
against the frozen reference.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import atisim  # noqa: F401  -- enables x64 before any array is made
from atisim import earth, earth_ref


def test_the_defining_wgs84_constants_are_exact():
    """a and f are DEFINING; b and e2 are derived from them and never quoted.

    JSBSim reports a semi-minor axis of 6356752.314186 m, which differs from
    a(1-f) by 5.87e-5 m. That is JSBSim's internal storage in feet
    round-tripping, not a different ellipsoid, so this model carries the
    defining pair and derives the rest. Asserted here so nobody "fixes" b to
    JSBSim's reported value later.
    """
    assert earth.A_WGS84 == 6378137.0
    assert earth.F_WGS84 == 1.0 / 298.257223563
    assert earth.GM_WGS84 == 3.986004418e14
    assert earth.J2_WGS84 == 1.08262982e-3
    assert earth.OMEGA_WGS84 == 7.292115e-5

    assert earth.B_WGS84 == pytest.approx(6356752.314245179, abs=1e-6)
    assert earth.E2_WGS84 == pytest.approx(earth.F_WGS84 * (2.0 - earth.F_WGS84), rel=1e-15)
    # The recorded difference against JSBSim's reported value, so a future
    # reader meets it as a number rather than as a surprise.
    assert abs(earth.B_WGS84 - 6356752.314186481) == pytest.approx(5.87e-5, rel=0.02)


def test_geodetic_ecef_roundtrip_reaches_the_float64_floor_including_the_poles():
    """Bowring at a FIXED 3 iterations, measured over the whole latitude range.

    Fixed rather than converged because it must stay jittable and
    differentiable. Three is chosen because it reaches the float64 floor: the
    ulp of an ECEF coordinate is 9.3e-10 m, and measured worst round-trip error
    is 3.5e-6 m at 1 iteration, 1.4e-8 at 2, and 3.7e-9 at 3. A fourth buys
    nothing.

    The poles are INCLUDED. The textbook altitude form h = p/cos(lat) - N is
    singular there; this module uses the non-singular
    h = p cos(lat) + z sin(lat) - a sqrt(1 - e2 sin^2 lat) instead, and this
    test is what holds it to that.

    Vectorised over the whole grid with a single jit+vmap compile rather than
    one Python-level call per point. The loop it replaced made 2,884 unjitted
    calls to a `lax.scan`-based function, each re-tracing and re-compiling,
    which cost three minutes and bought nothing -- the property is elementwise,
    so batching computes exactly the same arithmetic. The argmax reporting is a
    genuine gain: the loop could only say HOW BIG the worst error was, and this
    says WHERE it is.
    """
    lats = np.radians(np.linspace(-90.0, 90.0, 721))
    hs = np.array([0.0, 5.0e3, 12192.0, 2.0e4])
    lat_grid, h_grid = (g.ravel() for g in np.meshgrid(lats, hs, indexing="ij"))
    lon = 0.7

    to_ecef = jax.jit(jax.vmap(earth.geodetic_to_ecef, in_axes=(0, None, 0)))
    to_geodetic = jax.jit(jax.vmap(earth.ecef_to_geodetic))

    r = to_ecef(lat_grid, lon, h_grid)
    back_lat, back_lon, back_h = (np.asarray(x) for x in to_geodetic(r))

    assert np.allclose(back_lon, lon, atol=1e-12)

    lat_err = np.abs(back_lat - lat_grid)
    h_err = np.abs(back_h - h_grid)
    i_lat, i_h = np.argmax(lat_err), np.argmax(h_err)

    assert h_err[i_h] < 1.0e-8, (
        f"altitude round-trip {h_err[i_h]:.2e} m at "
        f"lat={np.degrees(lat_grid[i_h]):.4f} deg, h={h_grid[i_h]:.1f} m"
    )
    assert np.degrees(lat_err[i_lat]) * 3600 < 1.0e-9, (
        f"latitude round-trip {np.degrees(lat_err[i_lat]) * 3600:.2e} arcsec at "
        f"lat={np.degrees(lat_grid[i_lat]):.4f} deg, h={h_grid[i_lat]:.1f} m"
    )


def test_geodetic_to_ecef_puts_the_reference_points_where_they_belong():
    """Three points whose ECEF coordinates are the defining constants themselves."""
    equator = earth.geodetic_to_ecef(0.0, 0.0, 0.0)
    assert np.allclose(np.asarray(equator), [earth.A_WGS84, 0.0, 0.0], atol=1e-6)

    quarter = earth.geodetic_to_ecef(0.0, np.pi / 2, 0.0)
    assert np.allclose(np.asarray(quarter), [0.0, earth.A_WGS84, 0.0], atol=1e-6)

    pole = earth.geodetic_to_ecef(np.pi / 2, 0.0, 0.0)
    assert np.allclose(np.asarray(pole), [0.0, 0.0, earth.B_WGS84], atol=1e-6)


def test_ecef_to_ned_matrix_is_orthonormal_and_points_down_along_the_ellipsoid_normal():
    """Down must be the GEODETIC normal, not the geocentric radius.

    These differ by up to 0.19 degrees at 45 latitude. Using the geocentric
    direction costs 65 m/s in v_north at 47 degrees, measured against JSBSim,
    which is why this is asserted rather than assumed.
    """
    for lat_deg in (-89.0, -45.0, 0.0, 12.5, 47.0, 89.0):
        lat, lon = np.radians(lat_deg), 0.4
        m = np.asarray(earth.ecef_to_ned_matrix(lat, lon))

        assert np.allclose(m @ m.T, np.eye(3), atol=1e-13), f"not orthonormal at {lat_deg}"
        assert np.linalg.det(m) == pytest.approx(1.0, abs=1e-13)

        # `down` must be the inward geodetic normal: moving 1 m along -down
        # from the surface must raise geodetic altitude by exactly 1 m.
        surface = earth.geodetic_to_ecef(lat, lon, 0.0)
        up_ecef = -m[2]
        _, _, h = earth.ecef_to_geodetic(surface + np.asarray(up_ecef))
        assert float(h) == pytest.approx(1.0, abs=1e-6), f"down is not the normal at {lat_deg}"

        # North at the pole-facing side must raise latitude.
        north_ecef = m[0]
        lat2, _, _ = earth.ecef_to_geodetic(surface + np.asarray(north_ecef))
        assert float(lat2) > lat


def test_j2_gravity_matches_the_values_read_off_the_jsbsim_binary():
    """The two anchor values, measured from JSBSim v1.3.1 at sea level.

    These are GRAVITATION, not apparent gravity: JSBSim reports 9.8142 at the
    equator, not the 9.7803 that includes the centrifugal term. The centrifugal
    term is a SEPARATE term in the equation of motion, and folding it into
    gravity here would double-count it.
    """
    equator = earth.gravitation(earth.geodetic_to_ecef(0.0, 0.0, 0.0), earth.WGS84_J2)
    pole = earth.gravitation(earth.geodetic_to_ecef(np.pi / 2, 0.0, 0.0), earth.WGS84_J2)

    assert float(np.linalg.norm(np.asarray(equator))) == pytest.approx(
        9.814197353250055, rel=1e-11
    )
    assert float(np.linalg.norm(np.asarray(pole))) == pytest.approx(
        9.832066846743299, rel=1e-9
    )
    # It points inward.
    assert float(np.asarray(equator)[0]) < 0.0
    assert float(np.asarray(pole)[2]) < 0.0


def test_j2_is_what_separates_the_two_gravity_models():
    """The inverse-square model is JSBSim's `gravity-model = 0`, and it differs.

    Carried so a test can isolate what J2 alone is worth. At the pole the two
    differ by roughly 3 J2 = 0.32%, which is far above anything this project
    calls agreement, so the choice of model is not a detail.
    """
    r = earth.geodetic_to_ecef(np.pi / 2, 0.0, 0.0)
    j2 = float(np.linalg.norm(np.asarray(earth.gravitation(r, earth.WGS84_J2))))
    inv = float(np.linalg.norm(np.asarray(earth.gravitation(r, earth.WGS84_INVERSE_SQUARE))))
    assert abs(j2 / inv - 1.0) == pytest.approx(3.0 * earth.J2_WGS84 * (earth.A_WGS84 / earth.B_WGS84) ** 2, rel=0.02)


def test_flat_is_constant_g_along_the_local_vertical_and_does_not_rotate():
    """FLAT is a CONFIGURATION of the one plant, not a second implementation.

    THE DIRECTION IS THE WHOLE TEST, and this assertion used to enforce the bug
    it now catches. It read "FLAT is spherical, so down is -r_hat" and compared
    against the GEOCENTRIC radial -- while the local NED frame everything else
    uses is built on the GEODETIC normal. The two differ by up to 11.5 arcmin,
    which put 0.0329 m/s^2 of NORTHWARD gravity into FLAT at 45 deg: 3.35e-3 g
    of systematic bias.

    That matters because FLAT exists to reproduce the pre-Earth plant, whose
    gravity was exactly `dcm.T @ [0, 0, G0]` with no horizontal component at
    all, and because a later task asserts FLAT differs from that plant only by
    round-off. 3.35e-3 g is not round-off.

    So the assertion is now the one the name always claimed: resolved into the
    local NED frame, FLAT gravity is [0, 0, G0] exactly.
    """
    from atisim.atmosphere import G0

    for lat_deg in (0.0, 30.0, 45.0, 60.0, 89.0):
        lat = np.radians(lat_deg)
        r = earth.geodetic_to_ecef(lat, 0.3, 9144.0)
        g_ned = np.asarray(earth.ecef_to_ned_matrix(lat, 0.3)) @ np.asarray(
            earth.gravitation(r, earth.FLAT)
        )
        assert float(np.linalg.norm(g_ned)) == pytest.approx(G0, rel=1e-14)
        assert abs(g_ned[0]) < 1e-9, (
            f"FLAT has {g_ned[0]:.4e} m/s^2 of northward gravity at {lat_deg} deg; "
            "the geocentric form gave 0.0329 at 45 deg"
        )
        assert abs(g_ned[1]) < 1e-9
        assert g_ned[2] == pytest.approx(G0, rel=1e-14)

    assert earth.FLAT.rotation_rate == 0.0
    assert earth.WGS84_J2.rotation_rate == earth.OMEGA_WGS84


def test_an_anchor_carries_its_own_ecef_position_and_frame():
    anchor = earth.anchor_at(np.radians(47.0), np.radians(11.0), 0.0)
    assert np.allclose(
        np.asarray(anchor.r_ecef),
        np.asarray(earth.geodetic_to_ecef(np.radians(47.0), np.radians(11.0), 0.0)),
        atol=1e-9,
    )
    m = np.asarray(anchor.T_e2l)
    assert np.allclose(m @ m.T, np.eye(3), atol=1e-13)


def test_the_geodesy_is_jittable_and_differentiable():
    """The stated reason BOWRING_ITERATIONS is fixed rather than converged.

    A while-loop-to-convergence would be neither, so this is the property that
    justifies the design -- and it was asserted in a comment and tested
    nowhere. `gravitation` is included because the equations of motion will
    differentiate through it when the trim solver runs.
    """
    lat, lon, h = np.radians(47.0), np.radians(11.0), 9144.0

    round_trip = jax.jit(lambda a, o, z: earth.ecef_to_geodetic(
        earth.geodetic_to_ecef(a, o, z)
    ))
    back_lat, _, back_h = round_trip(lat, lon, h)
    assert float(back_lat) == pytest.approx(lat, abs=1e-12)
    assert float(back_h) == pytest.approx(h, abs=1e-8)

    # d(altitude)/d(altitude) is exactly 1: the round trip is the identity, so
    # this checks the derivative flows through the whole scan rather than
    # merely that grad() returns without raising.
    d_h = jax.grad(lambda z: round_trip(lat, lon, z)[2])(h)
    assert float(d_h) == pytest.approx(1.0, abs=1e-6)

    # Gravity must differentiate w.r.t. position for the same reason.
    g_mag = jax.jit(lambda r: jnp.linalg.norm(earth.gravitation(r, earth.WGS84_J2)))
    r0 = earth.geodetic_to_ecef(lat, lon, h)
    assert np.all(np.isfinite(np.asarray(jax.grad(g_mag)(r0))))


def test_every_wgs84_constant_is_in_the_provenance_ledger():
    """SOURCED, with the JSBSim recovery named as the CHECK, not as the source.

    The distinction matters and the ledger's own rules enforce it: SOURCED means
    read from a cited table. WGS-84 is that table. The agreement with JSBSim is
    evidence that this model and that one read the same table, which is a
    verification, not a provenance.
    """
    from atisim.provenance import LEDGER

    for name in ("earth.A_WGS84", "earth.F_WGS84", "earth.GM_WGS84",
                 "earth.J2_WGS84", "earth.OMEGA_WGS84"):
        assert name in LEDGER, f"{name} missing from the provenance ledger"
        assert LEDGER[name].category == "SOURCED"
        assert "WGS-84" in LEDGER[name].detail

    for name in ("earth.B_WGS84", "earth.E2_WGS84"):
        assert LEDGER[name].category == "DERIVED"
        assert LEDGER[name].inputs, f"{name} claims DERIVED with no inputs"


def _probes():
    return earth_ref.load()


def test_our_geodesy_reproduces_jsbsims_own_position_and_frame():
    """Round-trip JSBSim's ECEF through our geodesy and back to its own frame.

    This is the check that would have failed had we used geocentric latitude:
    the t_e2l comparison is the one that costs 65 m/s in v_north at 47 degrees.

    THE 1e-11 TOLERANCE IS THE SEMI-MINOR AXIS, AND IT IS NOT NOISE. The plan
    specified 1e-12 and that fails, at 9.255e-12 rad on the 47 deg probe. The
    cause was found rather than absorbed:

        worst latitude residual   9.255e-12 rad = 5.923e-05 m on the ground
        our b minus JSBSim's b                    5.870e-05 m

    Those are the same number to 1%. `earth.B_WGS84` is DERIVED as a(1-f) =
    6356752.314245 m; JSBSim reports 6356752.314186 m, its own value round-
    tripped through feet. Re-running this inversion with JSBSim's b instead
    drops the residual by 19x to 153x per probe, which is what identifies the
    term. So this is a disagreement about the fifth decimal of one constant,
    showing up as an angle, and it is bounded by that constant.

    IT IS DELIBERATELY NOT FIXED. Adopting JSBSim's b to close a tolerance
    would be picking a constant to make a test pass, and it would contradict
    test_the_defining_wgs84_constants_are_exact, which exists to stop exactly
    that. 0.06 mm of ground position is far below anything this project
    reports.

    THE TOLERANCE IS SET AGAINST THE BOUND, NOT AGAINST THE OBSERVATION.
    Measured worst case is 9.255e-12 rad in lat, 3.100e-13 in lon and 7.050e-12
    in t_e2l. 1e-11 would clear the latitude term by 1.08x -- shaved to just
    above what happened to be observed, which is the kind of tolerance that
    fails spuriously the first time the reference is regenerated and then gets
    loosened by someone who does not know why it was tight. 2e-11 is about
    twice the b-discrepancy bound above, which is the quantity that actually
    limits the agreement.

    Nothing is weakened by that. The mistake this test exists to catch --
    geocentric latitude in place of geodetic -- moves t_e2l by 2.57e-3, which
    is EIGHT ORDERS clear of either tolerance.
    """
    for p in _probes():
        lat, lon, h = earth.ecef_to_geodetic(p.r_ecef)
        assert float(lat) == pytest.approx(p.lat, abs=2e-11)
        assert float(lon) == pytest.approx(p.lon, abs=2e-11)

        back = earth.geodetic_to_ecef(lat, lon, h)
        assert np.allclose(np.asarray(back), p.r_ecef, atol=1e-6)

        assert np.allclose(np.asarray(earth.ecef_to_ned_matrix(lat, lon)), p.t_e2l, atol=2e-11)

        # And the frame actually maps JSBSim's ECEF velocity onto its own NED.
        assert np.allclose(p.t_e2l @ p.vel_ecef, p.vel_ned, atol=1e-9)


def test_our_j2_gravity_reproduces_jsbsims_at_every_probe():
    """Worst 4.599e-13 relative, when the position is read back from the engine.

    THE READ-BACK RULE IS WHAT THIS MEASURES. Feeding the NOMINAL commanded
    position into the gravity formula instead of the one JSBSim actually
    reached puts the worst error at 4.833e-06 -- seven orders worse -- because
    the aircraft drifts up to 448 m during the two-second settle. The rel=1e-10
    tolerance sits 217x above the read-back figure and four orders BELOW the
    nominal-position mistake, so it passes the right one and fails the wrong
    one with room on both sides.

    Both numbers are measured across all six probes of the frozen reference.
    They supersede the 3.6e-13 and 2.4e-6 quoted while the design was being
    written, which came from a single probe at 47N.
    """
    for p in _probes():
        ours = np.asarray(earth.gravitation(p.r_ecef, earth.WGS84_J2))
        assert float(np.linalg.norm(ours)) == pytest.approx(p.gravity_magnitude, rel=1e-10)

        # Direction too, via JSBSim's own weight force.
        theirs_body = p.weight / p.mass
        t_b2e = (p.t_l2b @ p.t_e2l).T
        assert np.allclose(t_b2e @ theirs_body, ours, atol=1e-7)


def test_derivatives_reproduce_every_jsbsim_eom_term():
    """Verification check 1, the EOM rows. This is the heart of the work.

    Each probe carries JSBSim's own forces, moments, inertia, velocities and
    rates, plus the accelerations it computed from them. Feeding OUR equations
    the same inputs must give the same accelerations.

    Tolerances are tight because the design measured these relations closing to
    machine zero. A loose tolerance here would pass with a wrong sign on a small
    term, which is exactly the failure the design's frame-transfer probe hit.
    """
    from atisim import earth as E
    from atisim.dynamics import earth_acceleration_terms

    for p in _probes():
        t_b2e = (p.t_l2b @ p.t_e2l).T
        omega_earth_body = t_b2e.T @ np.array([0.0, 0.0, E.OMEGA_WGS84])

        # The frame-transfer relation, which JSBSim states directly.
        assert np.allclose(p.pqri - p.pqr, omega_earth_body, atol=1e-12)

        accel, omega_dot = earth_acceleration_terms(
            force=jnp.asarray(p.force),
            moment=jnp.asarray(p.moment),
            mass=p.mass,
            inertia=jnp.asarray(p.inertia),
            inertia_inv=jnp.asarray(np.linalg.inv(p.inertia)),
            vel_body=jnp.asarray(p.vel_body),
            omega=jnp.asarray(p.pqr),
            quat_body_to_ecef=jnp.asarray(_quat_from(t_b2e)),
            r_ecef=jnp.asarray(p.r_ecef),
            model=E.WGS84_J2,
        )
        assert np.allclose(np.asarray(accel), p.uvwdot, atol=1e-9), (
            f"translational EOM at lat {np.degrees(p.lat):.1f}: "
            f"{np.asarray(accel)} vs {p.uvwdot}"
        )
        assert np.allclose(np.asarray(omega_dot), p.pqrdot, atol=1e-12), (
            f"rotational EOM at lat {np.degrees(p.lat):.1f}: "
            f"{np.asarray(omega_dot)} vs {p.pqrdot}"
        )


def _quat_from(m):
    """numpy Shepperd, so the test does not depend on the code under test."""
    c = np.array([
        1 + m[0, 0] + m[1, 1] + m[2, 2], 1 + m[0, 0] - m[1, 1] - m[2, 2],
        1 - m[0, 0] + m[1, 1] - m[2, 2], 1 - m[0, 0] - m[1, 1] + m[2, 2],
    ])
    i = int(np.argmax(c)); s = np.sqrt(c[i]) * 2.0
    q = [
        np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]),
        np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]),
        np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]),
        np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]),
    ][i]
    return q / np.linalg.norm(q)


def test_the_frame_transfer_sign_is_plus_and_a_wings_level_probe_cannot_tell():
    """The design's own near-miss, pinned so it cannot come back.

    Near wings-level, `omega_be x Omega_b` is about 1e-5 and the WRONG sign's
    residual is the same order -- both signs "fit". Only the high-rate probes
    separate them: measured 6.9e-18 for plus against 1.9e-5 for minus. This test
    asserts that the high-rate probes are what does the separating, so nobody
    later trims the probe grid down to level flight and destroys the evidence.

    MEASURED ON THE FROZEN GRID, AND IT REFINES THAT CLAIM RATHER THAN
    CONFIRMING IT. Every probe here carries |pqr| >= 0.0293 rad/s -- the grid was
    regenerated with controls deflected, so it contains no genuinely wings-level
    case any more -- and against a reference that is itself exact, ALL SIX
    separate the signs at the absolute 1e-9 asserted below:

        lat     |pqr|     resid +     resid -    resid - , relative to |pqrdot|
        0.0    0.05824   3.99e-15    8.47e-06    3.03e-04
       47.0    0.03167   3.89e-16    3.20e-06    5.04e-04
      -33.0    0.02930   2.85e-16    1.70e-06    3.31e-04
       60.0    0.15084   8.33e-15    1.74e-05    3.88e-04
      -70.0    0.11456   7.87e-15    1.25e-05    1.51e-04
       10.0    0.61732   6.81e-15    3.19e-05    1.29e-03

    So the `> 0.05` filter is not what is doing the work at this tolerance, and
    the design's near-miss is visible in the LAST column instead: the wrong sign
    costs 1.5e-4 to 5.0e-4 of `pqrdot` at the low-rate probes and 1.3e-3 at the
    highest-rate one. An ordinary rel=1e-3 comparison -- which is what the first
    probe was judged by -- accepts the minus sign everywhere except the 10 deg
    probe. The absolute threshold is what makes it visible everywhere, and the
    minus residual scales as 2|pqr|*Omega, so the signs would only become
    genuinely indistinguishable at 1e-9 below |pqr| ~ 6.9e-6 rad/s. The filter is
    kept as the guard it was written to be: it fails loudly if the grid is ever
    regenerated at true wings-level, which is the state in which this evidence
    really does vanish.
    """
    high_rate = [p for p in _probes() if np.linalg.norm(p.pqr) > 0.05]
    assert high_rate, "the probe grid no longer contains a high-body-rate case"

    for p in high_rate:
        t_b2e = (p.t_l2b @ p.t_e2l).T
        omega_earth_body = t_b2e.T @ np.array([0.0, 0.0, earth.OMEGA_WGS84])
        term = np.cross(p.pqr, omega_earth_body)
        assert np.allclose(p.pqridot + term, p.pqrdot, atol=1e-14)
        assert not np.allclose(p.pqridot - term, p.pqrdot, atol=1e-9)


# ---------------------------------------------------------------------------
# FLAT against the pre-change model
# ---------------------------------------------------------------------------
def _fly_flat_from_the_frozen_condition(frozen, lon):
    """The frozen pre-Earth initial condition, re-flown on `earth.FLAT`.

    THE AIRCRAFT IS PLACED AT THE ANCHOR, which is where `trim.trimmed_state`
    puts it and what keeps `pos_ecef` a few-kilometre offset instead of a
    6390 km absolute coordinate. `state.py`'s header explains why that matters
    and F4 records what it is worth.

    `lon` is a free parameter because under FLAT -- `rotation_rate` 0, and a
    gravity magnitude that depends on neither longitude nor latitude -- the
    problem is exactly invariant under a rotation about the Earth's axis. Two
    longitudes therefore run identical physics through different ECEF numbers,
    which is what isolates round-off below.
    """
    from atisim import state as st
    from atisim.aircraft import REGISTRY
    from atisim.integrate import init_sim, logged_rollout
    from atisim.trim import longitudinal_controls

    ac = REGISTRY["boeing747"]
    dt, n_steps = float(frozen["dt"]), int(frozen["n_steps"])
    alpha, elevator, throttle = (float(v) for v in frozen["trim"])
    airspeed, alt = float(frozen["airspeed"]), float(frozen["altitude"])

    anchor = earth.anchor_at(0.0, lon, alt)
    start = st.state_from_ned(
        jnp.zeros(3),
        airspeed * jnp.array([np.cos(alpha), 0.0, np.sin(alpha)]),
        st.euler_to_quat(jnp.array(0.0), jnp.array(alpha), jnp.array(0.0)),
        jnp.zeros(3),
        anchor,
    )
    # NOT `trimmed_controls`: that takes a six-element trim SOLUTION now, and
    # the frozen run was flown by the three-unknown solver with aileron and
    # rudder identically zero. `longitudinal_controls` is the name for a
    # hand-built setting with no lateral half to drop.
    _, log = logged_rollout(
        init_sim(start, jax.random.PRNGKey(0)),
        longitudinal_controls(jnp.array(elevator), jnp.array(throttle)),
        dt, ac, n_steps, anchor=anchor, earth_model=earth.FLAT,
    )
    ours = np.asarray(jax.vmap(lambda s: st.pos_ned(s, anchor))(log.state))
    return ours, dt * np.arange(1, n_steps + 1)


def test_flat_is_a_curved_earth_and_its_cost_is_the_local_verticals_rotation():
    """0.489 m over 20 s, and it is PHYSICS rather than the round-off predicted.

    *** THIS TEST ASSERTS THE OPPOSITE OF WHAT THE PLAN AND THE DESIGN EXPECTED,
    AND THE MEASUREMENT IS WHY. ***

    Design section 8 and the `FLAT` docstring both say the FLAT configuration
    differs from the pre-Earth plant only by ECEF round-off, and the plan
    predicted ~3.6e-8 m over this 20 s run. MEASURED: 4.892e-01 m. The
    prediction was out by seven orders, and the reason is not arithmetic.

    `FLAT` IS A CURVED EARTH. `earth.gravitation`'s "constant" branch returns
    `G0 * ecef_to_ned_matrix(lat, lon)[2]` at the AIRCRAFT'S OWN position, so
    the direction of gravity rotates by `V t / R` as the aeroplane flies. The
    pre-Earth plant had one global NED frame and one fixed gravity vector. That
    difference is a term, not a rounding, and it is what this run measures:

        gravity tilt      g V t / R           (a horizontal acceleration)
        displacement      g V t^3 / (6 R)

    Measured against that closed form, with R = A_WGS84 + 12192 m:

        t (s)      divergence        g V t^3 / 6R      ratio
        0.52       8.533e-06         8.484e-06         1.00572
        1.02       6.435e-05         6.403e-05         1.00490
        2.02       4.991e-04         4.973e-04         1.00358
        6.02       1.317e-02         1.316e-02         1.00045
        10.02      6.072e-02         6.070e-02         1.00037
        20.00      4.892e-01         4.827e-01         1.01351

    The closed form accounts for the whole of it to better than 1.4% across the
    entire window, so nothing else of consequence is in there -- and the growth
    is CUBIC (1009.9x from 2 s to 20 s, against 1000 for t^3), the signature of
    an acceleration that grows linearly with time. Round-off would have been
    flat, and a constant acceleration difference would have been 100x.

    So the honest statement about FLAT is the one ASSUMPTIONS.md F4 now records:
    it reproduces the pre-Earth PHYSICS -- gravity magnitude, and no rotation --
    and it does not reproduce the pre-Earth GEOMETRY, because there is no flat
    Earth left underneath it. The round-off floor the design meant to claim is
    real, and is measured separately below at 1.15e-11 m.

    Both assertions are PREDICTIONS rather than allowances: a missing or extra
    Earth term would have to disguise itself as this exact cubic to survive
    them. A rotation leaking into FLAT, in particular, would add a Coriolis
    contribution that does not fit `g V t^3 / 6R`.
    """
    from atisim.atmosphere import G0

    frozen = np.load(Path(__file__).parent / "data" / "pre_earth_trajectory.npz")
    ours, t = _fly_flat_from_the_frozen_condition(frozen, 0.0)

    # The frozen run's origin was sea level; this one's is the anchor at cruise
    # altitude. Same trajectory and same axes, origin shifted by the anchor.
    theirs = frozen["pos_ned"] + np.array([0.0, 0.0, float(frozen["altitude"])])
    divergence = np.linalg.norm(ours - theirs, axis=1)

    early = float(divergence[t <= 2.0][-1])
    late = float(divergence[-1])
    print(f"\nFLAT vs pre-Earth: {early:.3e} m at 2 s, {late:.3e} m at 20 s")

    radius = float(earth.A_WGS84 + frozen["altitude"])
    predicted = float(G0) * float(frozen["airspeed"]) * t**3 / (6.0 * radius)
    settled = t >= 0.5          # below this both quantities are under 1e-5 m
    ratio = divergence[settled] / predicted[settled]
    assert np.all(np.abs(ratio - 1.0) < 0.02), (
        f"the divergence is {ratio.min():.5f}-{ratio.max():.5f} of the local "
        "vertical's closed form, so something other than the curvature of "
        "gravity is now in it"
    )
    assert 900.0 < late / early < 1100.0, (
        f"divergence grew {late / early:.1f}x from 2 s to 20 s; t^3 is 1000x. "
        "A constant acceleration difference would be 100x and round-off ~1x, "
        "so the growth law itself has changed"
    )


def test_the_round_off_floor_on_an_anchor_relative_ecef_trajectory():
    """1.15e-11 m over 20 s: what F4 measures now that position is an offset.

    THE FLOOR HAS TO BE ISOLATED, because the FLAT comparison above is dominated
    by a physical term ten orders of magnitude larger. The isolation is an exact
    symmetry: under FLAT the Earth does not rotate and gravity's magnitude
    depends on neither latitude nor longitude, so the whole problem is invariant
    under a rotation about the Earth's axis. Flying the same case from anchors
    at longitude 0 and 137 deg therefore runs IDENTICAL PHYSICS through
    DIFFERENT ECEF NUMBERS, and every metre of the difference is round-off.

    Measured over the frozen 747 run:

        divergence between the two longitudes    1.153e-11 m at 20 s
        ground-track offset reached              4717.87 m
        ulp of that offset                       9.095e-13 m
        floor, in ulp of the offset              12.7

    That is the number ASSUMPTIONS.md F4 now records, and it supersedes session
    11's 7e-11 m rather than contradicting it: F4's figure was for a `pos_ned`
    that CARRIED the 12,184 m altitude, and this state does not. Position is an
    offset from the anchor, so what gets differenced is the ground track.

    The ulp ratio quoted in `state.py`'s header is offset-dependent and both
    readings are right: float64 ulp doubles at each power of two, so a 6390 km
    absolute coordinate is 512x coarser than a 12 km offset and 1024x coarser
    than the 4.7 km one this run reaches.
    """
    frozen = np.load(Path(__file__).parent / "data" / "pre_earth_trajectory.npz")
    a, _ = _fly_flat_from_the_frozen_condition(frozen, 0.0)
    b, _ = _fly_flat_from_the_frozen_condition(frozen, np.radians(137.0))

    floor = float(np.linalg.norm(a - b, axis=1).max())
    offset = float(np.linalg.norm(a[-1]))
    print(f"\nround-off floor: {floor:.3e} m over a {offset:.1f} m offset "
          f"({floor / np.spacing(offset):.1f} ulp)")

    assert floor < 100.0 * np.spacing(offset), (
        f"round-off reached {floor / np.spacing(offset):.1f} ulp of the "
        f"{offset:.0f} m offset; it was 12.7 ulp when F4 was measured"
    )
    # The load-bearing half: the floor must stay far below anything the FLAT
    # comparison above is trying to see. It is 4.2e10 times smaller today.
    assert floor < 1.0e-9
