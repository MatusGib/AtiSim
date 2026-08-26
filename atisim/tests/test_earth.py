"""WGS-84 geodesy, gravity, and the rotating-Earth frames.

Every constant here was recovered from the running JSBSim binary, not quoted
from a table -- see the design doc section 2. The tests below assert the
DEFINING values; the term-by-term agreement with JSBSim is asserted separately
against the frozen reference.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import atisim  # noqa: F401  -- enables x64 before any array is made
from atisim import earth


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
    """FLAT is a CONFIGURATION of the one plant, not a second implementation."""
    from atisim.atmosphere import G0

    r = earth.geodetic_to_ecef(np.radians(47.0), 0.3, 9144.0)
    g = np.asarray(earth.gravitation(r, earth.FLAT))
    assert float(np.linalg.norm(g)) == pytest.approx(G0, rel=1e-14)
    # Along the inward radius: FLAT is spherical, so down is -r_hat.
    assert np.allclose(g / np.linalg.norm(g), -np.asarray(r) / np.linalg.norm(np.asarray(r)), atol=1e-14)
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
