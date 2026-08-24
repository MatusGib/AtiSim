"""The Oseguera & Bowles microburst, against its own source's stated constants.

Source: R. M. Oseguera and R. L. Bowles, "A Simple, Analytic 3-Dimensional
Downburst Model Based on Boundary Layer Stagnation Flow", NASA TM-100632, July
1988. Its velocity profiles come from the TASS numerical model, which is itself
built on the Joint Airport Weather Studies (JAWS) field data.

The paper states four constants that it derives by iteration from its own
equations, and they are the sharpest available check that the model has been
transcribed correctly rather than merely plausibly -- each is a DIFFERENT
consequence of the same two shaping functions, so a transcription error cannot
satisfy all four:

    peak outflow at r/R = 1.1212        <- solves exp(-x^2)(2x^2 + 1) = 1
    z_m / z*      = 0.22                <- equals ln(12.5)/11.5 = 0.2196
    z* / epsilon  = 12.5                <- the boundary-layer scale separation
    u_max         = 0.2357 * lambda * R <- the product of the two above

The same paper also gives w_max = lambda * z* * (exp(-z_h/z*) - 0.92), which is
this module's vertical equation evaluated on the axis with epsilon = z*/12.5 and
the boundary-layer term dropped. That is asserted too, because it is the one
statement that pins the VERTICAL equation independently of the horizontal one.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import atisim  # noqa: F401  -- enables x64
from atisim import wind

R = 1000.0
U_MAX = 19.03  # m/s, the paper's own example figure: 37 kt of peak outflow
Z_M = 150.0  # m, midpoint of the paper's "100 - 200 meters above the ground"


def a_burst(u_max=U_MAX, radius=R, z_m=Z_M):
    return wind.microburst(u_max=u_max, radius=radius, z_m=z_m)


def _wind(burst, north, altitude, east=0.0):
    return wind.microburst_wind(jnp.array([north, east, -altitude]), burst)


# --- the four constants the paper states -------------------------------------


def test_peak_outflow_is_at_1_1212_downdraft_radii():
    burst = a_burst()
    radii = np.linspace(1.0, 4.0 * R, 4000)
    speed = np.array([float(_wind(burst, r, Z_M)[0]) for r in radii])
    assert radii[int(np.argmax(speed))] / R == pytest.approx(1.1212, abs=0.005)


def test_the_altitude_of_maximum_outflow_is_0_22_of_the_characteristic_height():
    burst = a_burst()
    altitudes = np.linspace(1.0, 5.0 * Z_M, 4000)
    peak_radius = 1.1212 * R
    speed = np.array([float(_wind(burst, peak_radius, z)[0]) for z in altitudes])
    z_m = altitudes[int(np.argmax(speed))]
    assert z_m / float(burst.z_star) == pytest.approx(0.22, abs=0.005)
    assert z_m == pytest.approx(Z_M, rel=0.01)  # and it is where we asked for it


def test_the_boundary_layer_scale_separation_is_12_5():
    burst = a_burst()
    assert float(burst.z_star) / float(burst.epsilon) == pytest.approx(12.5)


def test_peak_outflow_equals_0_2357_lambda_R():
    burst = a_burst()
    assert float(_wind(burst, 1.1212 * R, Z_M)[0]) == pytest.approx(U_MAX, rel=0.002)
    assert U_MAX == pytest.approx(0.2357 * float(burst.lam) * R, rel=0.002)


def test_the_axis_downflow_matches_the_papers_closed_form():
    """w_max = lambda*z*(exp(-z_h/z*) - 0.92), the vertical equation pinned alone.

    The paper's closed form DROPS the in-boundary-layer term, and the amount it
    drops is exactly lambda*epsilon*exp(-z/epsilon) -- so this asserts the
    identity rather than a tolerance, which pins this module's equation and the
    paper's simplification of it at the same time. Substituting
    epsilon = z*/12.5 into the full expression is where the 0.92 comes from:
    1 - 1/12.5 = 0.92.
    """
    burst = a_burst()
    lam, z_star, eps = float(burst.lam), float(burst.z_star), float(burst.epsilon)
    for z_h in (200.0, 400.0, 600.0):
        stated = lam * z_star * (np.exp(-z_h / z_star) - 0.92)
        dropped = lam * eps * np.exp(-z_h / eps)
        w_up = -float(_wind(burst, 0.0, z_h)[2])  # NED down -> up positive
        assert w_up == pytest.approx(stated - dropped, rel=1e-9)
        assert w_up < 0.0  # it is a DOWNdraft


# --- the properties the model exists to have ---------------------------------


def test_both_components_vanish_at_the_ground():
    """The boundary-layer behaviour the paper says earlier models got wrong.

    Its introduction singles this out: "boundary layer effects should cause
    radial velocity to decay vertically to zero at the ground, as does the
    vertical velocity".
    """
    burst = a_burst()
    for r in (0.0, 0.5 * R, 1.1212 * R, 3.0 * R):
        at_ground = _wind(burst, r, 0.0)
        assert float(at_ground[0]) == pytest.approx(0.0, abs=1e-9)
        assert float(at_ground[2]) == pytest.approx(0.0, abs=1e-9)


def test_the_field_is_divergence_free():
    """It satisfies mass continuity -- the paper's central claim for it."""
    burst = a_burst()

    def divergence(p):
        return jnp.trace(jax.jacobian(lambda q: wind.microburst_wind(q, burst))(p))

    for north, alt in ((0.0, 200.0), (500.0, 300.0), (1121.0, 150.0), (3000.0, 80.0)):
        d = float(divergence(jnp.array([north, 0.0, -alt])))
        assert d == pytest.approx(0.0, abs=1e-9)


def test_the_outflow_is_radial_and_the_core_descends():
    burst = a_burst()
    east_side = _wind(burst, 0.0, 250.0, east=1.1212 * R)
    north_side = _wind(burst, 1.1212 * R, 250.0)
    assert float(east_side[1]) == pytest.approx(float(north_side[0]), rel=1e-9)
    assert float(east_side[0]) == pytest.approx(0.0, abs=1e-9)
    # On the axis: pure downdraft, no outflow.
    axis = _wind(burst, 0.0, 250.0)
    assert float(axis[0]) == pytest.approx(0.0, abs=1e-9)
    assert float(axis[2]) > 0.0  # NED down positive == descending air


def test_the_outflow_reverses_across_the_centre():
    """The headwind-to-tailwind change that makes a microburst lethal."""
    burst = a_burst()
    upwind = float(_wind(burst, -1.1212 * R, Z_M)[0])
    downwind = float(_wind(burst, +1.1212 * R, Z_M)[0])
    assert upwind == pytest.approx(-downwind, rel=1e-9)
    assert downwind > 0.0


def test_the_divergence_meets_the_definition_of_a_microburst():
    """Wilson et al. 1984, quoted by Proctor et al.: a surface outflow counts as
    a microburst when peak horizontal divergence exceeds 10 m/s over 1-4 km."""
    burst = a_burst()
    span = 2.0 * 1.1212 * R
    change = float(_wind(burst, 1.1212 * R, Z_M)[0]) - float(
        _wind(burst, -1.1212 * R, Z_M)[0]
    )
    assert 1000.0 <= span <= 4000.0
    assert change / (span / 1000.0) > 10.0


def test_the_microburst_satisfies_the_wind_model_contract():
    from atisim import trim

    model = wind.microburst_model(a_burst())
    state = trim.trimmed_state(jnp.array(0.05), jnp.array(50.0), jnp.array(300.0))
    key = jax.random.PRNGKey(0)
    wind_ned, omega_gust, _, out_key, _ = model(
        wind.zero_wind_state(), state, key, jnp.array(0.01)
    )
    assert wind_ned.shape == (3,)
    assert omega_gust.shape == (3,)
    assert np.array_equal(np.asarray(out_key), np.asarray(key))
