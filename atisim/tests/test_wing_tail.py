"""The wing-tail gust delay, opt-in since v1.2 (Physics and Assumptions E13).

The pitching gust is built from the gust at the CG minus the gust at the tail,
over the tail arm: `wind.sampled_rates` with two longitudinal stations. In a
frozen field that IS the tail meeting the gust l_t/V after the wing. The
point-gust model takes the gradient at the CG instead, which is this in the
limit l_t -> 0.

What is asserted:
  - JSBSim's tail arms are what its files print, and the CR-2144 747's derived
    arm agrees with JSBSim's B747 to 5% -- an independent check of the
    derivation;
  - two stations give the closed-form secant cos(t) (1 - e^(-i W l cos t)) / l,
    and a vanishing arm gives the point gradient back;
  - the flown response with the wing-tail delay matches its linearisation at
    test_gust.py's gates, alone and together with the gust lag.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from atisim import airframe, gust, trim, wind
from atisim.aircraft import REGISTRY
from atisim.atmosphere import speed_of_sound
from atisim.state import euler_to_quat
from atisim.units import FT2M

AMPLITUDE_GATE = 0.005  # test_gust.py's, unchanged
PHASE_GATE_DEG = 0.5


@pytest.fixture(scope="module")
def cruise747():
    ac = REGISTRY["boeing747"]
    H = float(wind.MEHTA_HANNIBAL_ALTITUDE)
    return ac, 0.80 * float(speed_of_sound(H)), H, float(airframe.effective_tail_arm(ac) * ac.c)


@pytest.fixture(scope="module")
def cruise737():
    ac = REGISTRY["boeing737"]
    H = 30000.0 * FT2M
    return ac, 0.75 * float(speed_of_sound(H)), H, airframe.JSBSIM_HTAILARM_FT["boeing737"] * FT2M


def test_jsbsim_tail_arms_are_what_the_files_print():
    assert airframe.JSBSIM_HTAILARM_FT == {"boeing737": 48.04, "boeing747_jsbsim": 106.6}


def test_the_derived_747_arm_agrees_with_jsbsims_to_five_percent():
    """-Cmq/CLq * c for CR-2144's 747 against JSBSim B747's <htailarm>: two
    sources that share nothing. Measured 33.5 m against 32.5 m, +3.1%."""
    ac = REGISTRY["boeing747"]
    derived = float(airframe.effective_tail_arm(ac) * ac.c)
    jsbsim = airframe.JSBSIM_HTAILARM_FT["boeing747_jsbsim"] * FT2M
    assert derived / jsbsim == pytest.approx(1.0, abs=0.05)


def test_a_given_tail_arm_needs_no_derivable_one():
    """The 737 defines no CLq, so its derived arm is undefined and `stations`
    refuses it -- unless the arm is supplied."""
    ac = REGISTRY["boeing737"]
    with pytest.raises(ValueError):
        airframe.stations(ac)
    st = airframe.stations(ac, n_lon=2, tail_arm=14.64)
    np.testing.assert_allclose(np.asarray(st.longitudinal), [-14.64, 0.0])


def test_two_stations_give_the_closed_form_secant(cruise747):
    ac, V, H, arm = cruise747
    theta, lam, north = 0.04, 900.0, 123.4
    Om = 2.0 * np.pi / lam
    field = wind.sinusoidal_vertical_field(1.0, lam)
    st = airframe.stations(ac, n_lon=2, tail_arm=arm)
    quat = euler_to_quat(jnp.array(0.0), jnp.array(theta), jnp.array(0.0))
    q = float(wind.sampled_rates(jnp.array([north, 0.0, -H]), quat, field, st)[1])
    gain = np.cos(theta) * (1.0 - np.exp(-1j * Om * arm * np.cos(theta))) / arm
    assert q == pytest.approx((gain * np.exp(1j * Om * north)).real, rel=1e-9, abs=1e-12)


def test_a_vanishing_arm_is_the_point_gradient(cruise747):
    ac, V, H, _ = cruise747
    om = 2.0 * np.pi * 0.5 / V
    np.testing.assert_allclose(gust.gust_transfer(ac, V, H, om, tail_arm=1e-6),
                               gust.gust_transfer(ac, V, H, om), rtol=1e-5)


@pytest.mark.parametrize("which", ["747", "737"])
@pytest.mark.parametrize("f_hz", [0.1874, 1.0])
def test_the_flown_delayed_response_matches_its_linearisation(cruise747, cruise737,
                                                             which, f_hz):
    ac, V, H, arm = cruise747 if which == "747" else cruise737
    r = gust.measure_gust_transfer(ac, V, H, V / f_hz, tail_arm=arm)
    theory = gust.gust_transfer(ac, V, H, r["spatial_frequency"],
                                ground_speed=r["ground_speed"], tail_arm=arm)
    assert abs(abs(r["H"]) / abs(theory) - 1.0) < AMPLITUDE_GATE
    assert abs(np.degrees(np.angle(r["H"] / theory))) < PHASE_GATE_DEG


def test_the_delay_and_the_lag_together_match_their_linearisation(cruise747):
    ac, V, H, arm = cruise747
    r = gust.measure_gust_transfer(ac, V, H, V / 0.5, tail_arm=arm, gust_lag=True)
    theory = gust.gust_transfer(ac, V, H, r["spatial_frequency"],
                                ground_speed=r["ground_speed"], tail_arm=arm,
                                gust_lag=True)
    assert abs(abs(r["H"]) / abs(theory) - 1.0) < AMPLITUDE_GATE
    assert abs(np.degrees(np.angle(r["H"] / theory))) < PHASE_GATE_DEG
