"""MIL-F-8785C's rotational gust spectra, and the q roll-off option in gust.py.

Design phase V5(a), docs/design/specs/2026-09-23-turbulence-validation-2-design.md
2.3 and 4 (0.4). Session 34 recorded these spectra as absent from MIL-F-8785C.
They are in section 3.7.5, printed p. 58, read from the page image because the
PDF's OCR layer is unusable for equations.
"""

import math

import jax.numpy as jnp
import numpy as np
import pytest

from atisim import wind

SPAN = 59.643264  # boeing747's b, m


def test_q_is_the_w_spectrum_through_the_specifications_filter():
    omega = jnp.geomspace(1e-5, 0.3, 50)
    ratio = wind.dryden_q_spectrum(omega, 1.0, SPAN) / wind.dryden_spectrum(omega, 1.0)
    expected = omega**2 / (1.0 + (4.0 * SPAN * omega / math.pi) ** 2)
    assert np.allclose(np.asarray(ratio), np.asarray(expected), rtol=1e-12)


def test_q_tends_to_the_pure_gradient_at_low_frequency():
    """Printed p. 58: q_g = dw_g/dx 'precisely at very low frequencies only'."""
    omega = jnp.array(1e-6)
    ratio = float(wind.dryden_q_spectrum(omega, 1.0, SPAN)
                  / (omega**2 * wind.dryden_spectrum(omega, 1.0)))
    assert ratio == pytest.approx(1.0, abs=1e-6)


def test_the_pure_gradient_overstates_phi_q_by_14_percent_at_the_747_response_peak():
    """1 + (4 b Omega/pi)^2 at 0.1874 Hz and 236.056 m/s: the 747's V1 response
    peak at M 0.80 / 37,000 ft (PROJECT.md section 4), not its short period,
    0.16404 Hz. The flown field carries a further cos^4(theta)."""
    omega = jnp.array(2.0 * math.pi * 0.1874 / 236.056)
    excess = float(omega**2 * wind.dryden_spectrum(omega, 1.0)
                   / wind.dryden_q_spectrum(omega, 1.0, SPAN))
    assert excess == pytest.approx(1.1434867716747008, rel=1e-9)


def test_p_at_zero_frequency_is_the_specifications_constant():
    """sigma_w^2/L_w * 0.8 (pi L_w / 4b)^(1/3) for the 747's span, by hand."""
    assert float(wind.dryden_p_spectrum(jnp.array(0.0), 1.0, SPAN)) == pytest.approx(
        0.0028723066635546165, rel=1e-12)


def test_p_halves_at_the_span_corner():
    corner = jnp.array(math.pi / (4.0 * SPAN))
    ratio = float(wind.dryden_p_spectrum(corner, 1.0, SPAN)
                  / wind.dryden_p_spectrum(jnp.array(0.0), 1.0, SPAN))
    assert ratio == pytest.approx(0.5, rel=1e-12)


def test_r_uses_the_three_b_over_pi_filter_on_the_lateral_component():
    omega = jnp.array(0.01)
    got = float(wind.dryden_r_spectrum(omega, 2.0, SPAN))
    expected = (0.01**2 / (1.0 + (3.0 * SPAN * 0.01 / math.pi) ** 2)
                * float(wind.dryden_spectrum(omega, 2.0, wind.DRYDEN_LV)))
    assert got == pytest.approx(expected, rel=1e-12)


def test_p_scales_as_sigma_squared_and_rolls_off_as_the_square():
    two_corners = jnp.array(2.0 * math.pi / (4.0 * SPAN))
    ratio = float(wind.dryden_p_spectrum(two_corners, 2.0, SPAN)
                  / wind.dryden_p_spectrum(jnp.array(0.0), 1.0, SPAN))
    assert ratio == pytest.approx(4.0 / 5.0, rel=1e-12)


def test_q_scales_as_sigma_squared():
    om = jnp.array(0.01)
    assert float(wind.dryden_q_spectrum(om, 2.0, SPAN)) == pytest.approx(
        4.0 * float(wind.dryden_q_spectrum(om, 1.0, SPAN)), rel=1e-12)


from atisim import gust, trim  # noqa: E402
from atisim.aircraft import REGISTRY  # noqa: E402
from atisim.atmosphere import speed_of_sound  # noqa: E402


@pytest.fixture(scope="module")
def cruise747():
    ac = REGISTRY["boeing747"]
    H = float(wind.MEHTA_HANNIBAL_ALTITUDE)
    return ac, 0.80 * float(speed_of_sound(H)), H


def test_the_rolloff_enters_as_one_common_factor(cruise747):
    """H is affine in the roll-off factor f = 1/(1 + i(4b/pi)Omega):
    H = H_0 + f H_q. Three spans give three factors, and the three H's must be
    collinear in f. Which channel carries f is pinned by
    test_the_rolloff_is_the_q_channels_own_contribution_times_f_minus_one."""
    ac, V, H = cruise747
    om = 5e-3
    Hs, fs = [], []
    for span in (None, 20.0, 80.0):
        Hs.append(complex(gust.gust_transfer(ac, V, H, om, q_rolloff_span=span)))
        fs.append(1.0 if span is None
                  else 1.0 / (1.0 + 1j * (4.0 * span / math.pi) * om))
    slope_a = (Hs[1] - Hs[0]) / (fs[1] - fs[0])
    slope_b = (Hs[2] - Hs[0]) / (fs[2] - fs[0])
    assert abs(slope_a - slope_b) < 1e-9 * abs(slope_a)


def test_a_vanishing_span_leaves_the_transfer_alone(cruise747):
    ac, V, H = cruise747
    om = np.array([2e-3, 2e-2])
    assert np.allclose(gust.gust_transfer(ac, V, H, om, q_rolloff_span=1e-9),
                       gust.gust_transfer(ac, V, H, om), rtol=1e-9, atol=0.0)


def test_mean_square_ratio_passes_the_rolloff_through(cruise747):
    """Equal to the trapezoid of |H|^2 Phi on mean_square_ratio's own grid, for
    both phases -- which a wrong span or a dropped keyword would break."""
    ac, V, H = cruise747
    b = float(ac.b)
    Om = np.geomspace(2.0 * np.pi / 40_000.0, 2.0 * np.pi / 20.0, 400)
    Phi = np.asarray(wind.dryden_spectrum(jnp.asarray(Om), 1.0), dtype=float)
    for phase in ("minimum", "zero"):
        Hm2 = np.abs(gust.gust_transfer(ac, V, H, Om, q_rolloff_span=b,
                                        q_rolloff_phase=phase)) ** 2
        assert gust.mean_square_ratio(
            ac, V, H, wind.dryden_spectrum, n_points=400, q_rolloff_span=b,
            q_rolloff_phase=phase) == pytest.approx(
                float(np.trapezoid(Hm2 * Phi, Om)), rel=1e-12)


@pytest.mark.parametrize("phase", ["minimum", "zero"])
def test_the_rolloff_is_the_q_channels_own_contribution_times_f_minus_one(
        cruise747, phase):
    """H(span) - H(none) = (f - 1) H_q, with H_q the pitching channel's own
    contribution -- so a filter on any other channel, or on two, fails."""
    ac, V, H = cruise747
    om, span = 5e-3, 50.0
    x, _ = trim.trim(jnp.array(V), jnp.array(H), ac)
    alpha = float(x[0])
    A, B, C, D = gust._linearise(ac, alpha, float(x[1]), float(x[2]), V, H)
    ch = gust.frozen_sinusoid_channels(alpha)
    H_q = ((C @ np.linalg.solve(1j * om * V * np.eye(5) - A, B[:, 1]) + D[1])
           * (ch[1, 0] + 1j * om * ch[1, 1]))
    corner = 1.0 + 1j * (4.0 * span / math.pi) * om
    f = 1.0 / corner if phase == "minimum" else 1.0 / abs(corner)
    got = (complex(gust.gust_transfer(ac, V, H, om, q_rolloff_span=span,
                                      q_rolloff_phase=phase))
           - complex(gust.gust_transfer(ac, V, H, om)))
    assert abs(got - (f - 1.0) * H_q) < 1e-9 * abs((f - 1.0) * H_q)


def test_an_unknown_rolloff_phase_is_refused(cruise747):
    ac, V, H = cruise747
    with pytest.raises(ValueError):
        gust.gust_transfer(ac, V, H, 5e-3, q_rolloff_span=50.0,
                           q_rolloff_phase="linear")
