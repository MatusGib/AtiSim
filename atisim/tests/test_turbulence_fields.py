"""The von Karman vertical field, and a check that it leaves the Dryden field alone.

It changes no default; dryden_vertical_field is what every published turbulence
result was flown through, and stays so.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import integrate

from atisim import wind


def _sample(field, span_m=2_000_000.0, step_m=50.0):
    """w_up along a long straight path, vectorised."""
    x = jnp.arange(0.0, span_m, step_m)
    return np.asarray(jax.vmap(lambda xi: -field(jnp.array([xi, 0.0, 0.0]))[2])(x))


def test_von_karman_constants_are_the_specifications():
    """Printed p. 48: 2,500 ft using the von Karman form. Printed p. 47: 1.339."""
    assert wind.VON_KARMAN_LW == pytest.approx(2500.0 * 0.3048, rel=1e-15)
    assert wind.VON_KARMAN_SCALE == 1.339


def test_von_karman_is_one_sided_like_dryden():
    """Its integral over the half line is sigma^2, not 2 sigma^2."""
    L = float(wind.VON_KARMAN_LW)
    val, _ = integrate.quad(
        lambda om: float(wind.von_karman_spectrum(jnp.array(om), 2.0)),
        0.0, np.inf, limit=400)
    assert val == pytest.approx(4.0, rel=2e-3), L


def test_von_karman_falls_as_minus_five_thirds():
    L = float(wind.VON_KARMAN_LW)
    om = np.array([100.0, 1000.0]) / L
    p = np.asarray(wind.von_karman_spectrum(jnp.asarray(om), 1.0))
    slope = math.log(p[1] / p[0]) / math.log(om[1] / om[0])
    assert slope == pytest.approx(-5.0 / 3.0, abs=2e-3)


def test_von_karman_field_has_about_the_variance_it_asks_for():
    """A fixed-amplitude sum realises sum(A^2/2) exactly over a long path. That
    misses the tail beyond 20 m (~2%) and the log grid's Riemann shortfall
    (~1.6%), so the band is one-sided."""
    var = float(np.var(_sample(wind.von_karman_vertical_field(2.0, 7), 4_000_000.0)))
    assert 0.94 * 4.0 < var < 1.01 * 4.0


def test_the_dryden_field_is_untouched():
    """What every section-4 turbulence number was flown through."""
    a = wind.dryden_vertical_field(3.0, 11)
    for xi in (0.0, 1234.5, 98765.4):
        p = jnp.array([xi, 0.0, 0.0])
        omega, amp, phase = wind.dryden_vertical_components(3.0, 11)
        expected = -float(jnp.sum(amp * jnp.cos(omega * xi + phase)))
        assert float(a(p)[2]) == expected

