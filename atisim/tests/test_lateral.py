"""The lateral dimension — session 24, phase 1.

WHAT THIS FILE IS FOR. Until this session every wind field in the project was a
function of along-track distance alone. Nothing varied across the span, so
`wind.strip_roll_moment` integrated to exactly zero on every field, the lateral
modes were validated as eigenvalues and never excited, and
`vortex_viz.Encounter` had no channel that could have recorded a roll if one had
happened. The model was longitudinal by construction rather than by choice, and
no document said so.

THE FAILURE MODE THIS FILE IS WRITTEN AGAINST. PROJECT.md section 6 records two
bugs that survived three sessions and a 209-test suite because *every test was
still air*, and still air cannot distinguish airspeed from groundspeed. A
longitudinal-only suite hides lateral bugs in exactly the same way. So every
assertion below that a lateral quantity is NON-zero has a companion asserting it
IS zero where it must be — the negative control is the point, not the decoration.

Sources, in the order they are first used:

  MIL-F-8785C  the Dryden spectral forms. NOT in the folder; the vertical form
               was already second-hand, and these two carry the same standing.
               What can be checked without the document is that they belong to
               one isotropic field, and that is the first test here.
  PARKS        E. K. Parks et al., J. Aircraft 22(2), 1985, 124-129. Eqs.
               (3)-(6) and the r = (l^2 cos^2 dpsi + d^2)^(1/2) geometry.
  MEHTA        R. S. Mehta, J. Guidance 10(1), 1987, 27-31. The five-vortex
               Hannibal field and its psi = 31 deg traverse.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from atisim import vortex_viz, wind
from atisim.aircraft import CRUISE, REGISTRY
from atisim.units import RAD2DEG

_PSI = math.radians(wind.MEHTA_HANNIBAL_PSI_DEG)


def _oblique():
    """Mehta's field with its traverse angle carried as a real 3-D direction."""
    return wind.mehta_hannibal_array()._replace(sin_dpsi=jnp.array(math.sin(_PSI)))


def _perpendicular():
    """The same array with the vortex lines square to the flight path."""
    return wind.mehta_hannibal_array()._replace(
        cos_dpsi=jnp.array(1.0), sin_dpsi=jnp.array(0.0)
    )


# ---------------------------------------------------------------------------
# The two new Dryden components
# ---------------------------------------------------------------------------


def test_both_dryden_spectra_match_the_specification_now_that_it_is_held():
    """MIL-F-8785C 3.7.1.2, printed p. 47, against the two implemented forms.

    *** THIS TEST COULD NOT BE WRITTEN UNTIL SESSION 25. *** The spec was not in
    the folder when these functions landed, so the check below it -- that the two
    forms belong to one isotropic field -- was the best available and is kept.
    `refs/MIL-F-8785C.pdf` is now held and section 3.7.1.2 prints:

        Phi_u = sigma_u^2 (2 L_u/pi) / [1 + (L_u Om)^2]
        Phi_v = sigma_v^2 (L_v/pi) [1 + 3(L_v Om)^2] / [1 + (L_v Om)^2]^2
        Phi_w = sigma_w^2 (L_w/pi) [1 + 3(L_w Om)^2] / [1 + (L_w Om)^2]^2

    Asserted at three arguments whose values are arithmetic rather than a
    re-typing of the code, which is what makes this a check and not a mirror.
    Between them they pin the three things that can be wrong: the factor of 2 on
    the longitudinal form, the 3 in the transverse numerator, and the SQUARE on
    the transverse denominator.

    What it does NOT close: Figure 7, the sigma chart on p. 49, is still
    un-digitised. Holding the document closes the FORMS, not the intensity.
    """
    sigma, L = 3.0, 500.0
    base = sigma ** 2 * L / np.pi

    # Om = 0. Longitudinal is exactly TWICE transverse -- the factor of 2.
    assert float(wind.dryden_longitudinal_spectrum(jnp.array(0.0), sigma, L))         == pytest.approx(2.0 * base, rel=1e-12)
    assert float(wind.dryden_spectrum(jnp.array(0.0), sigma, L))         == pytest.approx(base, rel=1e-12)

    # L*Om = 1. Phi_u = 2/(1+1) = 1 and Phi_w = (1+3)/(1+1)^2 = 1, both in
    # units of base -- so the two forms CROSS here, and they must cross exactly.
    one = jnp.array(1.0 / L)
    assert float(wind.dryden_longitudinal_spectrum(one, sigma, L))         == pytest.approx(base, rel=1e-12)
    assert float(wind.dryden_spectrum(one, sigma, L))         == pytest.approx(base, rel=1e-12)

    # L*Om = 2. Phi_u = 2/5 = 0.4, Phi_w = (1+12)/25 = 0.52. Past the crossing
    # the transverse form is the LARGER, which is the square in the denominator
    # losing to the 3 in the numerator -- get either wrong and this flips.
    two = jnp.array(2.0 / L)
    assert float(wind.dryden_longitudinal_spectrum(two, sigma, L))         == pytest.approx(0.4 * base, rel=1e-12)
    assert float(wind.dryden_spectrum(two, sigma, L))         == pytest.approx(0.52 * base, rel=1e-12)

    # The spec prints v and w with identical right-hand sides. `dryden_spectrum`
    # serves both, so that is true here by construction -- asserted anyway,
    # because the construction is the claim.
    assert wind.DRYDEN_LV == wind.DRYDEN_LW


def test_the_two_dryden_spectra_belong_to_the_same_isotropic_field():
    """*** The check that substitutes for a document this project does not hold.

    MIL-F-8785C is not in `Reference_papers`. The vertical form was already
    transcribed second-hand and the longitudinal one now is too, so a
    transcription error in either would be invisible to any amount of reading.

    What is checkable without the spec is that the pair is INTERNALLY
    CONSISTENT. For an isotropic field the one-dimensional transverse and
    longitudinal spectra are related by

        Phi_transverse = 0.5 * (Phi_long - Omega dPhi_long/dOmega)

    and a wrong pair of forms would essentially have to be wrong in exactly
    compensating ways to survive it. Differentiated with `jax.grad` rather than a
    finite difference, so the tolerance can be machine precision instead of a
    step-size compromise.
    """
    sigma, L = 3.0, wind.DRYDEN_LW
    omega = jnp.geomspace(1e-6, 1e2, 60)

    d_phi = jax.vmap(jax.grad(
        lambda o: wind.dryden_longitudinal_spectrum(o, sigma, L)
    ))(omega)
    transverse = np.asarray(wind.dryden_spectrum(omega, sigma, L))
    predicted = 0.5 * (
        np.asarray(wind.dryden_longitudinal_spectrum(omega, sigma, L))
        - np.asarray(omega) * np.asarray(d_phi)
    )
    assert np.abs(transverse - predicted).max() / transverse.max() < 1e-12

    # Negative control: the relation is not something any smooth pair satisfies.
    # Halving the length scale on one side must break it.
    wrong = 0.5 * (
        np.asarray(wind.dryden_longitudinal_spectrum(omega, sigma, 0.5 * L))
        - np.asarray(omega) * np.asarray(jax.vmap(jax.grad(
            lambda o: wind.dryden_longitudinal_spectrum(o, sigma, 0.5 * L)))(omega))
    )
    assert np.abs(transverse - wrong).max() / transverse.max() > 0.1


def test_both_spectra_are_one_sided_and_carry_the_variance_they_claim():
    """The factor-of-two trap, for the new component as well as the old.

    A two-sided reading integrates to 2 sigma^2 and makes every gust sqrt(2)
    too large. `dryden_spectrum`'s own comment records the trap; this asserts it
    for both forms rather than for one.
    """
    si = pytest.importorskip("scipy.integrate")
    sigma, L = 2.5, wind.DRYDEN_LW
    for spectrum in (wind.dryden_longitudinal_spectrum, wind.dryden_spectrum):
        area, _ = si.quad(
            lambda o: float(spectrum(jnp.array(o), sigma, L)), 0, np.inf, limit=400
        )
        assert area == pytest.approx(sigma**2, rel=1e-6)


def test_the_three_component_field_realises_its_sigma_on_every_axis():
    """And the components must be INDEPENDENT, which is the easy thing to break.

    One shared phase set across the three components would give the right
    variance on each axis and a perfectly correlated field -- isotropic
    turbulence is neither. `dryden_field` splits one seed three ways; this is
    what checks that it did.
    """
    sigma = 3.0
    field = wind.dryden_field(sigma, seed=3)
    x = jnp.linspace(0.0, 120_000.0, 40_000)
    w = np.asarray(jax.vmap(
        lambda s: field(jnp.array([s, 0.0, -11_000.0]))
    )(x))

    for axis in range(3):
        assert w[:, axis].std() == pytest.approx(sigma, rel=0.10)
    for a, b in ((0, 1), (0, 2), (1, 2)):
        assert abs(np.corrcoef(w[:, a], w[:, b])[0, 1]) < 0.15


def test_the_dryden_field_still_has_no_spanwise_variation_and_says_so():
    """A guard on the docstring, because the claim is easy to misread.

    Adding u and v gives the model SIDESLIP, not a rolling gust: all three
    components are functions of along-track distance, so every strip still sees
    the same vertical gust. If this ever starts failing, the field has gained a
    transverse structure and `dryden_field`'s docstring is lying.
    """
    field = wind.dryden_field(3.0, seed=1)
    on = np.asarray(field(jnp.array([500.0, 0.0, -11_000.0])))
    off = np.asarray(field(jnp.array([500.0, 30.0, -11_000.0])))
    assert np.abs(on - off).max() == 0.0


# ---------------------------------------------------------------------------
# The vortex as a line in space
# ---------------------------------------------------------------------------


def test_the_line_vortex_reproduces_parks_along_the_whole_flight_path():
    """*** The gate the lateral vortex work rests on.

    `line_vortex_wind` is a second implementation of a field the project already
    had, and the reason it is allowed to exist at all is that it must reproduce
    the first one where the first one is defined -- on the flight path. Checked
    for BOTH geometries, because they exercise different halves of the
    arithmetic: at dpsi = 0 the axis is the east unit vector and the projection
    term vanishes; at 31 deg it does not.
    """
    H = wind.MEHTA_HANNIBAL_ALTITUDE
    x = jnp.linspace(-6000.0, 6000.0, 1201)
    pos = jnp.stack([x, jnp.zeros_like(x), jnp.full_like(x, -H)], axis=1)

    for array, label in ((_perpendicular(), "dpsi=0"), (_oblique(), "dpsi=31")):
        parks = np.asarray(jax.vmap(lambda p: wind.vortex_wind(p, array))(pos))
        lines = np.asarray(jax.vmap(lambda p: wind.line_vortex_wind(p, array))(pos))
        # The VERTICAL component is what every result in the project is built
        # from, and it must agree to machine precision in both geometries.
        assert np.abs(parks[:, 2] - lines[:, 2]).max() < 1e-12, label
        # So must the horizontal MAGNITUDE. Its direction is the one thing that
        # differs, and only when the traverse is oblique -- see below.
        assert np.abs(
            np.hypot(parks[:, 0], parks[:, 1]) - np.hypot(lines[:, 0], lines[:, 1])
        ).max() < 1e-12, label


def test_the_point_model_discards_the_east_wind_of_an_oblique_vortex():
    """A simplification in `vortex_wind` that nothing had measured.

    Parks' model is two-dimensional in the plane PERPENDICULAR TO THE VORTEX
    LINES. `vortex_wind` returns his horizontal magnitude along NORTH -- along
    the flight path -- which is exact only when the lines are square to it. At
    Mehta's 31 deg the true horizontal perturbation is that same magnitude
    rotated by dpsi, so it splits into cos(dpsi) along the path and sin(dpsi)
    across it, and the across-path half has been dropped.

    That discarded half is a SIDESLIP input, and it is the only one this field
    has ever had. Which is why the point model's beta is identically zero.
    """
    H = wind.MEHTA_HANNIBAL_ALTITUDE
    array = _oblique()
    p = jnp.array([600.0, 0.0, -H])
    parks = np.asarray(wind.vortex_wind(p, array))
    lines = np.asarray(wind.line_vortex_wind(p, array))

    assert parks[1] == 0.0, "the point model has no east component at all"
    assert abs(lines[1]) > 1.0, "the line form must have a real one"

    # The rotation is exactly by dpsi, which is what makes this a geometry
    # correction rather than a new model.
    assert lines[0] / parks[0] == pytest.approx(math.cos(_PSI), rel=1e-9)
    assert lines[1] / -parks[0] == pytest.approx(math.sin(_PSI), rel=1e-9)


def test_the_traverse_angle_is_carried_as_a_consistent_pair():
    """A bug this file caught, kept as the test that would catch it again.

    `cos_dpsi` predates `sin_dpsi` by a session, so an array can carry a cosine
    with the sine still at its 0.0 default. `vortex_wind` never notices -- the
    sine drops out of Parks' expression on the flight path -- but
    `line_vortex_wind` builds the vortex line's DIRECTION from the pair, and a
    cosine without its sine is a vector of length cos(dpsi) rather than 1. The
    induced velocity then comes out scaled by that length: quietly wrong,
    looking exactly like physics.

    Found by `scripts/lateral.py` reporting a 7 m/s disagreement between the two
    forms in the one geometry where they are supposed to be identical. Two
    things now stop it: `mehta_hannibal_array` sets both halves, and
    `vortex_axis` normalises so a mismatched pair is a wrong ANGLE rather than a
    wrong MAGNITUDE.
    """
    array = wind.mehta_hannibal_array()
    assert float(array.cos_dpsi) == pytest.approx(math.cos(_PSI))
    assert float(array.sin_dpsi) == pytest.approx(math.sin(_PSI))
    assert math.hypot(float(array.cos_dpsi), float(array.sin_dpsi)) ==         pytest.approx(1.0, abs=1e-12)

    # The axis is a unit vector even when the pair is not, which is what turns
    # a silent magnitude error into a visible angle one.
    broken = array._replace(sin_dpsi=jnp.array(0.0))
    assert float(jnp.linalg.norm(wind.vortex_axis(broken))) == pytest.approx(1.0)
    # And normalising a cosine-only pair lands on due east -- i.e. it is read as
    # perpendicular, which is at least a geometry someone meant.
    np.testing.assert_allclose(
        np.asarray(wind.vortex_axis(broken)), [0.0, 1.0, 0.0], atol=1e-12
    )


def test_only_an_oblique_array_varies_across_the_span():
    """The positive result and its negative control, in one test.

    A vortex line parallel to the span cannot produce a rolling gust however it
    is written -- every point of the wing is the same distance from it. An
    oblique one must. Getting the first half wrong would manufacture a rolling
    moment out of nothing.
    """
    H = wind.MEHTA_HANNIBAL_ALTITUDE
    tip = 0.5 * 59.64  # half a 747 span, m

    def across(array):
        left = np.asarray(wind.line_vortex_wind(jnp.array([1000.0, -tip, -H]), array))
        right = np.asarray(wind.line_vortex_wind(jnp.array([1000.0, tip, -H]), array))
        return abs(left[2] - right[2])

    assert across(_perpendicular()) == 0.0
    assert across(_oblique()) > 0.1


# ---------------------------------------------------------------------------
# The response, and the channels that can finally see it
# ---------------------------------------------------------------------------


def _fly(field, *, strip=False, dt=0.05):
    ac = REGISTRY["boeing747"]
    V = CRUISE["boeing747"]["airspeed"]
    H = wind.MEHTA_HANNIBAL_ALTITUDE
    r0 = float(wind.MEHTA_HANNIBAL_R0)
    x0 = float(wind.MEHTA_HANNIBAL_X_FT[0]) * 0.3048
    x1 = float(wind.MEHTA_HANNIBAL_X_FT[-1]) * 0.3048
    start = x0 - 12.0 * r0
    return vortex_viz.fly(
        ac, field, V, H, label="lateral", start_north=start,
        seconds=(x1 + 12.0 * r0 - start) / V, dt=dt,
        window=(x0 - 2.0 * r0, x1 + 2.0 * r0), window_name="array", strip=strip,
    )


def test_a_longitudinal_field_produces_exactly_zero_lateral_response():
    """The negative control on everything below, and the more important half.

    PROJECT.md section 6's lesson is that a suite which only ever exercises the
    easy case cannot see the bug. Here the easy case is a field with no
    spanwise structure, and the assertion is EXACT zero rather than a
    tolerance: the vortex equations have no `y` in them, so anything non-zero
    would mean the new channels are picking up something they invented.
    """
    enc = _fly(lambda p: wind.vortex_wind(p, _oblique()))
    w = enc.window
    for channel in (enc.beta, enc.p_gust, enc.p, enc.r, enc.phi):
        assert channel is not None
        assert np.abs(channel[w]).max() == 0.0


def test_the_line_vortex_rolls_the_aircraft_and_the_strip_path_changes_it():
    """*** The phase-1 gate, and the number it produces is not small.

    Two things happen here for the first time in this project:

      1. A wind field rolls the aircraft. Through Mehta's own identified field
         at his own altitude, the 747 reaches a bank angle of over ten degrees
         -- a response the analysis pipeline could not previously represent, let
         alone report.

      2. `loads.strip_increment` moves a reported number. Built in session 14,
         it had until now changed every result by exactly 0.000000 because no
         field varied across the span. Switching it on moves the peak bank by
         more than a tenth of itself.

    The longitudinal answer barely moves, which matters: it means none of the
    project's existing conclusions were resting on the missing dimension.
    """
    point = _fly(lambda p: wind.vortex_wind(p, _oblique()))
    line = _fly(lambda p: wind.line_vortex_wind(p, _oblique()))
    strip = _fly(lambda p: wind.line_vortex_wind(p, _oblique()), strip=True)

    def peak(enc, ch):
        return float(np.abs(ch(enc)[enc.window]).max())

    # 1. The field now rolls the aircraft, and it did not before.
    assert peak(point, lambda e: e.phi) == 0.0
    assert peak(line, lambda e: e.phi) * RAD2DEG > 10.0
    assert peak(line, lambda e: e.beta) * RAD2DEG > 2.0
    assert peak(line, lambda e: e.p_gust) > 0.05

    # 2. The strip path is no longer a no-op.
    bank_line = peak(line, lambda e: e.phi)
    bank_strip = peak(strip, lambda e: e.phi)
    assert abs(bank_strip - bank_line) / bank_line > 0.10

    # 3. And the longitudinal result is essentially untouched, so nothing that
    #    was concluded from it is put at risk by any of the above.
    for enc in (line, strip):
        moved = abs(enc.n_z[enc.window].max() - point.n_z[point.window].max())
        assert moved / abs(point.n_z[point.window].max() - 1.0) < 0.10
