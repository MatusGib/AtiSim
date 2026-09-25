"""The Kussner gust lag, opt-in since v1.2 (Physics and Assumptions C12).

The lag is Jones's two-exponential approximation to Kussner's indicial function
on the VERTICAL gust, integrated as two extra states on `SimState` by the same
RK4 stages as the aircraft. Off by default, and then nothing changes.

What is asserted, and why each:
  - Jones's transfer function is what the constants say, and tracks Sears'
    magnitude to 0.980-1.051 over the reduced frequencies this project
    forces -- the DECLARED approximation, priced rather than assumed;
  - a uniform wind produces no lag transient when the lag starts at the wind
    it meets, so the lag is a response to a CHANGE in the gust and not to the
    presence of wind;
  - the flown, lagged response to a single gust sinusoid matches the lagged
    linearisation, amplitude and phase, at the gates the unlagged path passes
    (test_gust.py) -- the verification that makes the lag's number usable;
  - what the lag does to the load, which is NOT the uniform reduction C12's
    |S|-on-|H| bound implied: with phase, it raises the load slightly near
    the resonances and cuts it only above ~0.5 Hz.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from atisim import gust, integrate, trim, vortex_viz, wind
from atisim.aircraft import REGISTRY
from atisim.atmosphere import speed_of_sound

AMPLITUDE_GATE = 0.005  # test_gust.py's, unchanged
PHASE_GATE_DEG = 0.5
MACH = 0.80


@pytest.fixture(scope="module")
def cruise():
    """test_gust.py's condition: the 747 at the Hannibal altitude."""
    ac = REGISTRY["boeing747"]
    H = float(wind.MEHTA_HANNIBAL_ALTITUDE)
    return ac, MACH * float(speed_of_sound(H)), H


def test_jones_transfer_is_what_its_constants_say():
    k = np.array([0.0, 0.05, 0.3, 1.0])
    p = 1j * k
    expected = 1.0 - 0.5 * p / (p + 0.13) - 0.5 * p / (p + 1.0)
    np.testing.assert_allclose(wind.kussner_jones(k), expected, rtol=0, atol=1e-15)
    assert wind.kussner_jones(0.0) == 1.0  # a steady gust is not attenuated


def test_jones_tracks_sears_magnitude_to_within_six_percent():
    """The DECLARED approximation, priced. Measured 0.980-1.051 over k 0.005-1
    on this grid; first written as 5% from a nine-point look, which the grid
    showed was 0.001 too tight -- a first estimate corrected, not a loosening.

    The worst place is not the high-k end but k ~ 0.02, the transports' short
    period: Jones gives |psi| 0.991 where Sears gives 0.969, because a sum of
    exponentials cannot follow Sears' k ln k behaviour at low k."""
    k = np.geomspace(0.005, 1.0, 60)
    ratio = np.abs(wind.kussner_jones(k)) / np.abs(gust.sears(k))
    assert ratio.min() > 0.97 and ratio.max() < 1.06, (ratio.min(), ratio.max())


def test_a_uniform_wind_gives_no_lag_transient(cruise):
    """The lag starts at the wind it meets, so a uniform wind changes nothing."""
    ac, V, H = cruise
    field = lambda p: jnp.array([3.0, -1.0, -2.0])  # includes a vertical part  # noqa: E731
    x, _ = trim.trim(jnp.array(V), jnp.array(H), ac)
    state = trim.trimmed_state(x[0], jnp.array(V), jnp.array(H))
    controls = trim.trimmed_controls(x[1], x[2])
    runs = []
    for lag in (False, True):
        enc = vortex_viz.fly_from_state(
            ac, field, state, controls, label="uniform", seconds=20.0, dt=0.02,
            window=(-np.inf, np.inf), window_name="whole run",
            stage_sampled=True, gust_lag=lag)
        runs.append(enc)
    np.testing.assert_allclose(runs[1].n_z, runs[0].n_z, rtol=0, atol=1e-12)
    np.testing.assert_allclose(runs[1].theta, runs[0].theta, rtol=0, atol=1e-12)


def test_the_lag_is_absent_by_default():
    """None, an empty pytree leaf, so a run without the lag carries exactly
    the SimState it carried before the lag existed."""
    sim = integrate.init_sim(trim.trimmed_state(0.05, 230.0, 10000.0),
                             jax.random.PRNGKey(0))
    assert sim.gust_lag is None
    # The same leaves as the six fields that existed before the lag.
    assert len(jax.tree.leaves(sim)) == len(jax.tree.leaves(tuple(sim)[:-1]))


@pytest.mark.parametrize("f_hz", [0.1874, 1.0])
def test_the_flown_lagged_response_matches_its_linearisation(cruise, f_hz):
    """Amplitude AND phase, at the short period and at 1 Hz, where k is
    0.021 and 0.111 for the 747 and |psi| is 0.991 and 0.840."""
    ac, V, H = cruise
    r = gust.measure_gust_transfer(ac, V, H, V / f_hz, gust_lag=True)
    theory = gust.gust_transfer(ac, V, H, r["spatial_frequency"],
                                ground_speed=r["ground_speed"], gust_lag=True)
    amplitude_error = abs(abs(r["H"]) / abs(theory) - 1.0)
    phase_error = abs(np.degrees(np.angle(r["H"] / theory)))
    assert amplitude_error < AMPLITUDE_GATE, (f_hz, amplitude_error)
    assert phase_error < PHASE_GATE_DEG, (f_hz, phase_error)


def test_what_the_lag_does_to_the_load(cruise):
    """FIRST WRITTEN as "the lag reduces the load at the short period", C12's
    sign, and it failed: the lagged/unlagged |H| there is 1.013. C12's bound
    applied |S| to |H| and so dropped phase; with phase the lagged vertical
    channel interacts with the UNLAGGED pitching channel, as E13's roll-off
    did. What is pinned instead is what the linearisation says, measured:
    +1.3% at the short period, -9% at 1 Hz, -17% at 2 Hz, and sigma_nz over
    the Dryden band -0.56% -- against C12's -6.78%."""
    ac, V, H = cruise

    def ratio(f_hz):
        om = 2.0 * np.pi * f_hz / V
        return (abs(gust.gust_transfer(ac, V, H, om, gust_lag=True))
                / abs(gust.gust_transfer(ac, V, H, om)))

    assert 1.0 < ratio(0.1874) < 1.03
    assert 0.88 < ratio(1.0) < 0.94
    assert ratio(2.0) < 0.86
    Om = np.geomspace(2.0 * np.pi / 40_000.0, 2.0 * np.pi / 20.0, 2000)
    phi = np.asarray(wind.dryden_spectrum(jnp.asarray(Om), 1.0))
    lagged = np.trapezoid(np.abs(gust.gust_transfer(ac, V, H, Om, gust_lag=True)) ** 2 * phi, Om)
    plain = np.trapezoid(np.abs(gust.gust_transfer(ac, V, H, Om)) ** 2 * phi, Om)
    assert 0.98 < np.sqrt(lagged / plain) < 1.0


def test_the_lag_refuses_the_held_wind(cruise):
    """The lag's input is re-evaluated per RK4 stage, so the held path -- which
    has no field to re-evaluate -- must refuse it rather than lag nothing."""
    ac, V, H = cruise
    state = trim.trimmed_state(0.05, jnp.array(V), jnp.array(H))
    controls = trim.trimmed_controls(0.0, 0.5)
    with pytest.raises(ValueError, match="stage_sampled"):
        vortex_viz.fly_from_state(
            ac, wind.sinusoidal_vertical_field(0.5, 2000.0), state, controls,
            label="held", seconds=1.0, dt=0.02, window=(-np.inf, np.inf),
            window_name="whole run", stage_sampled=False, gust_lag=True)


def test_the_lag_refuses_a_step_its_fast_pole_cannot_take(cruise):
    """RK4 on the fast pole is within 0.2% to 2.5 Hz only while lambda dt <= 1.5
    (wind.KUSSNER_RK4_LIMIT); a longer step is refused, not silently wrong."""
    ac, V, H = cruise
    dt_max = wind.kussner_max_dt(V, ac.c)
    assert 0.02 < dt_max < 0.03  # 0.026 s for the 747 at M 0.80
    state = trim.trimmed_state(0.05, jnp.array(V), jnp.array(H))
    controls = trim.trimmed_controls(0.0, 0.5)
    with pytest.raises(ValueError, match="too long"):
        vortex_viz.fly_from_state(
            ac, wind.sinusoidal_vertical_field(0.5, 2000.0), state, controls,
            label="long step", seconds=1.0, dt=0.05, window=(-np.inf, np.inf),
            window_name="whole run", stage_sampled=True, gust_lag=True)
