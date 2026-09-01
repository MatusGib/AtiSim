"""The clear-air-turbulence source work of session 23.

ONE FILE RATHER THAN THREE, deliberately. What is under test here is not a
module -- it is a set of claims that arrived together from a reading of the
`Reference_papers` folder, and every one of them spans `wind`, `checks` and
`validation` at once. Splitting them by the module they happen to touch would
put the evidence for a single finding in three places and leave nothing saying
they stand or fall together.

Sources, in the order they are first used:

  MEHTA    R. S. Mehta, "Modeling Clear-Air Turbulence with Vortices Using
           Parameter-Identification Techniques", J. Guidance, Control &
           Dynamics 10(1), Jan-Feb 1987, 27-31. AIAA 84-2083.
  TM       R. C. Wingrove, R. E. Bach Jr., T. A. Schultz, "Analysis of Severe
           Atmospheric Disturbances From Airline Flight Records", NASA
           TM-102186, June 1989.
  LESTER   P. F. Lester, O. Sen, R. E. Bach Jr., "The Use of DFDR Information
           in the Analysis of a Turbulence Incident over Greenland", Mon. Wea.
           Rev. 117, May 1989, 1103-1107.
  MISAKA   T. Misaka, S. Obayashi, E. Endo, "Measurement-Integrated Simulation
           of Clear Air Turbulence Using a Four-Dimensional Variational
           Method", J. Aircraft 45(4), 2008, 1217-1229.
  Y22      R. Yoshimura et al., "Large-Eddy and Flight Simulations of a
           Clear-Air Turbulence Event over Tokyo on 16 December 2014",
           J. Appl. Meteor. Climatol. 61(5), 2022, 503-519.
"""

import math

import jax.numpy as jnp
import numpy as np
import pytest

from atisim import checks, trim, validation, viz, wind
from atisim.aircraft import REGISTRY
from atisim.atmosphere import speed_of_sound
from atisim.units import FT2M

# ---------------------------------------------------------------------------
# The oblique traverse. cos_dpsi is new; every baseline predating it is not.
# ---------------------------------------------------------------------------

_R0 = 500.0 * FT2M
_V0 = 85.0 * FT2M
_ALT = 11278.0


def _single(r0=_R0, v0=_V0, altitude=_ALT, cos_dpsi=None):
    kw = {} if cos_dpsi is None else {"cos_dpsi": jnp.array(cos_dpsi)}
    return wind.VortexArray(
        north=jnp.array([0.0]), down=jnp.array([-altitude]),
        r0=jnp.array(r0), v0=jnp.array(v0), **kw,
    )


def test_the_default_traverse_is_perpendicular_and_bit_exact():
    """cos_dpsi defaults to 1.0 and moves not one bit of the old arithmetic.

    This is the entire licence for adding a field to `VortexArray`. Every
    PROJECT.md section 4 vortex baseline was frozen before `cos_dpsi` existed,
    and `1.0 * x == x` exactly in IEEE-754, so those baselines must come back
    to the BIT and not to a tolerance. A tolerance here would let a real change
    through unnoticed, which is the one thing this test exists to prevent.
    """
    array = _single()
    assert float(array.cos_dpsi) == 1.0
    for offset in (-3000.0, -400.0, -50.0, 0.0, 50.0, 400.0, 3000.0):
        got = np.asarray(
            wind.vortex_wind(jnp.array([offset, 0.0, -_ALT - 90.0]), array)
        )
        # The pre-change expression longhand, so a later edit to `vortex_wind`
        # cannot silently redefine what this is checking against.
        along, above = offset, 90.0
        r2 = along**2 + above**2
        r2_safe = max(r2, _R0**2)
        if r2 < _R0**2:
            w_h, w_u = _V0 * above / _R0, -_V0 * along / _R0
        else:
            w_h, w_u = _V0 * _R0 * above / r2_safe, -_V0 * _R0 * along / r2_safe
        assert got[0] == w_h, offset
        assert got[2] == -w_u, offset


def test_an_oblique_traverse_compresses_the_along_track_coordinate():
    """Parks r = (l^2 cos^2 dpsi + d^2)^(1/2) -- dpsi enters through l alone.

    An oblique traverse at along-track separation l must give exactly the
    perpendicular answer at separation l*cos(dpsi). That identity is what makes
    `cos_dpsi` a transcription of the source's equation rather than a knob.
    """
    cd = math.cos(math.radians(31.0))
    oblique, perpendicular = _single(cos_dpsi=cd), _single()
    for l in (-4000.0, -900.0, -120.0, 120.0, 900.0, 4000.0):
        d = -_ALT - 60.0
        got = wind.vortex_wind(jnp.array([l, 0.0, d]), oblique)
        want = wind.vortex_wind(jnp.array([l * cd, 0.0, d]), perpendicular)
        np.testing.assert_allclose(np.asarray(got), np.asarray(want),
                                   rtol=0, atol=1e-13)


def test_the_oblique_traverse_actually_changes_the_field():
    """The negative control on the test above.

    Two identities that both hold trivially when nothing happens are not
    evidence. At 31 deg the field must MOVE, or `cos_dpsi` is being ignored.
    """
    p = jnp.array([900.0, 0.0, -_ALT - 60.0])
    flat = np.asarray(wind.vortex_wind(p, _single()))
    tilted = np.asarray(
        wind.vortex_wind(p, _single(cos_dpsi=math.cos(math.radians(31.0))))
    )
    assert abs(tilted[2] - flat[2]) / abs(flat[2]) > 0.05


# ---------------------------------------------------------------------------
# MEHTA: the one vortex field in the project that declares nothing.
# ---------------------------------------------------------------------------


def test_mehta_prints_every_parameter_his_field_needs():
    """Five cores, their positions, one radius, one strength, psi, altitude.

    All on Mehta pp. 29-30. If a future edit slips a modelling choice into this
    field, this is where it surfaces.
    """
    assert len(wind.MEHTA_HANNIBAL_X_FT) == 5
    assert len(wind.MEHTA_HANNIBAL_Z_FT) == 5
    assert wind.MEHTA_HANNIBAL_R0 == pytest.approx(500.5 * FT2M)
    assert wind.MEHTA_HANNIBAL_V0 == pytest.approx(86.8 * FT2M)
    assert wind.MEHTA_HANNIBAL_PSI_DEG == 31.0
    assert wind.MEHTA_HANNIBAL_ALTITUDE == pytest.approx(37000.0 * FT2M)

    array = wind.mehta_hannibal_array()
    assert array.north.shape == (5,)
    assert float(array.cos_dpsi) == pytest.approx(math.cos(math.radians(31.0)))
    # z is the aircraft ABOVE the core, so a core sits at altitude H - z.
    for got, z in zip(-np.asarray(array.down) / FT2M, wind.MEHTA_HANNIBAL_Z_FT):
        assert got == pytest.approx(37000.0 - z, abs=1e-6)


def test_mehta_and_parks_agree_on_the_spacing_once_psi_is_applied():
    """Which spacing Parks' 3500 ft actually is -- and it is not along-track.

    Mehta's x are along the FLIGHT PATH. The vortex lines run perpendicular to
    the WIND, 31 deg off that path, so the separation measured perpendicular to
    the lines is the x difference times cos(31 deg), with the vertical offset in
    quadrature. Applied to the two cores the aircraft penetrated that lands
    within 1% of PARKS_CASES' 3500 ft.

    The discrimination is in the second half: read as an ALONG-TRACK spacing the
    same 3500 ft would place the cores 15% closer than Mehta puts them, and no
    other check in the project would notice.
    """
    i, j = wind.MEHTA_HANNIBAL_CORE_PAIR
    dx = wind.MEHTA_HANNIBAL_X_FT[j] - wind.MEHTA_HANNIBAL_X_FT[i]
    dz = wind.MEHTA_HANNIBAL_Z_FT[j] - wind.MEHTA_HANNIBAL_Z_FT[i]
    assert dx == pytest.approx(4104.0)

    perpendicular = math.hypot(dx * math.cos(math.radians(31.0)), dz)
    parks_ft = wind.PARKS_CASES["hannibal"]["spacing"] / FT2M
    assert parks_ft == pytest.approx(3500.0)
    assert perpendicular == pytest.approx(3521.5, abs=1.0)
    assert abs(perpendicular - parks_ft) / parks_ft < 0.01
    assert abs(dx - parks_ft) / parks_ft > 0.15


def test_two_cores_are_penetrated_and_three_are_not():
    """TM Fig. 7 calls two spikes "significant"; Mehta's z say which two.

    A core is penetrated when |z| < r0. Against a 500.5 ft radius, -3516,
    -1836, -94, -254 and +1738 ft give exactly two -- the pair
    MEHTA_HANNIBAL_CORE_PAIR names.
    """
    r0_ft = wind.MEHTA_HANNIBAL_R0 / FT2M
    hit = tuple(k for k, z in enumerate(wind.MEHTA_HANNIBAL_Z_FT) if abs(z) < r0_ft)
    assert hit == wind.MEHTA_HANNIBAL_CORE_PAIR


def test_the_two_source_strengths_scale_the_wind_exactly():
    """What the 2.1% strength disagreement between the sources is worth.

    PARKS_CASES pairs Fig. 4's 500 ft radius with Parks' 85 ft/s; Mehta and TM
    pair 500 ft with 87. `vortex_wind` multiplies by v0 exactly once on every
    branch, so the induced wind scales exactly and the spread on any
    wind-derived quantity is that same 2.1%. (TM-102186 rounds Mehta's 86.8
    to 87, which against Parks is 2.35% -- so the sources span 2.1-2.4%
    depending on which pair is taken, and the tests pin the Mehta figure.)
    """
    p = jnp.array([300.0, 0.0, -_ALT - 200.0])
    parks = np.asarray(wind.vortex_wind(p, _single(v0=85.0 * FT2M)))
    mehta = np.asarray(wind.vortex_wind(p, _single(v0=86.8 * FT2M)))
    np.testing.assert_allclose(mehta[[0, 2]] / parks[[0, 2]], 86.8 / 85.0,
                               rtol=1e-12)
    assert 86.8 / 85.0 == pytest.approx(1.0212, abs=1e-4)


# ---------------------------------------------------------------------------
# LESTER: a measured bound on every identified parameter in the module.
# ---------------------------------------------------------------------------


def test_the_dfdr_wind_error_is_the_rss_of_lester_table_1():
    """Table 1 p. 1105 gives contributions; `wind` stores the totals."""
    assert wind.DFDR_WIND_RMS_ERROR["horizontal"] == pytest.approx(
        math.sqrt(1.0**2 + 1.0**2 + 2.0**2), abs=1e-3)
    assert wind.DFDR_WIND_RMS_ERROR["vertical"] == pytest.approx(
        math.sqrt(1.0**2 + 2.0**2), abs=1e-3)


def test_the_implied_flow_angle_error_is_half_a_degree():
    """What Table 1's vertical term says about the paper's own alpha error.

    V*d(Theta - alpha) = 2.0 m/s at V = 250 m/s is 0.0080 rad. That is the
    number docs/ASSUMPTIONS.md E2 now carries in place of a 1 deg guess, and
    it halves the band that guess produced.
    """
    implied = math.degrees(2.0 / wind.DFDR_WIND_RMS_ERROR_SPEED)
    assert implied == pytest.approx(0.4584, abs=1e-3)
    assert implied < 0.5


def test_the_measured_error_is_a_smaller_fraction_of_hannibal_than_25_percent():
    """The bound that replaces the order-of-magnitude +/-25% band.

    Against Mehta's converged V0 the measured vertical wind error is about 8%,
    a factor of three tighter. Asserted as an inequality, not a value: the
    claim being made is that the sourced bound is TIGHTER than the guess, and
    that is what has to keep holding.
    """
    frac = wind.DFDR_WIND_RMS_ERROR["vertical"] / wind.MEHTA_HANNIBAL_V0
    assert frac == pytest.approx(0.0845, abs=0.002)
    assert frac < 0.25 / 2.0


def test_the_lester_wavelength_sits_inside_doyles_band():
    """22 km measured near the tropopause against Doyle's tropospheric 20-35 km.

    This is why the declared LEE_WAVE_WAVELENGTH survives rather than being
    replaced: an independent measurement at roughly the right altitude lands
    inside the band the declaration came from, 12% away from it.
    """
    assert 20_000.0 <= wind.LESTER_LEE_WAVE_WAVELENGTH <= 35_000.0
    gap = abs(wind.LESTER_LEE_WAVE_WAVELENGTH - wind.LEE_WAVE_WAVELENGTH)
    assert gap / wind.LEE_WAVE_WAVELENGTH == pytest.approx(0.12, abs=0.005)


# ---------------------------------------------------------------------------
# MISAKA: the severity index that is defined at cruise altitude.
# ---------------------------------------------------------------------------


def _flat_traj(n=2000, dt=0.01, n_z=1.0):
    """A synthetic straight-and-level trajectory at a chosen constant load.

    Built rather than flown: `rms_normal_load` reads only the load-factor
    series, and a real rollout would make a units test depend on trim.
    """
    t = np.arange(1, n + 1) * dt
    return viz.Trajectory(
        t=t,
        pos_ned=np.column_stack([t * 235.9, np.zeros(n), np.full(n, -12192.0)]),
        vel_body=np.column_stack([np.full(n, 235.9), np.zeros(n), np.zeros(n)]),
        quat=np.column_stack([np.ones(n), np.zeros((n, 3))]),
        omega=np.zeros((n, 3)),
        controls=np.zeros((n, 4)),
        wind_ned=np.zeros((n, 3)),
        omega_gust=np.zeros((n, 3)),
        mode=np.zeros(n, dtype=int),
    ), n_z


def test_still_air_reads_as_smooth_not_severe():
    """The index is on (n_z - 1), which is the whole point.

    Taking the RMS of n_z itself would report ~1.0 g -- "severe" by Misaka's
    bands -- for an aircraft sitting in still air. This is the negative control
    that says the deviation is being taken.
    """
    sigma = _sigma_of(np.ones(2000), dt=0.01)
    assert sigma == pytest.approx(0.0, abs=1e-12)


def _sigma_of(n_z, dt):
    """Run the index's own arithmetic over a given load series.

    `rms_normal_load` needs an `Aircraft` and a `Trajectory` to recover n_z;
    these unit tests care about the windowing, so they drive the same maths on
    a series directly. The identity between the two paths is asserted in
    `test_the_index_matches_a_direct_windowed_rms`.
    """
    width = int(round(checks.RMS_NORMAL_LOAD_WINDOW / dt))
    d = np.asarray(n_z) - 1.0
    c = np.concatenate(([0.0], np.cumsum(d * d)))
    return float(np.sqrt((c[width:] - c[:-width]) / width).max())


def test_a_known_sine_lands_where_its_rms_says_it_should():
    """A 0.4 g amplitude oscillation has RMS 0.4/sqrt(2) = 0.283 g.

    Chosen to land in Misaka's MODERATE band (0.2-0.3 g), so the test pins the
    band boundary as well as the arithmetic.
    """
    dt = 0.01
    t = np.arange(4000) * dt
    n_z = 1.0 + 0.4 * np.sin(2.0 * np.pi * t / 2.0)
    sigma = _sigma_of(n_z, dt)
    assert sigma == pytest.approx(0.4 / math.sqrt(2.0), rel=0.02)
    assert checks.RMS_NORMAL_LOAD_BANDS["moderate"] <= sigma
    assert sigma < checks.RMS_NORMAL_LOAD_BANDS["severe"]


def test_the_bands_are_misakas():
    """0.2 g moderate, 0.3 g severe, over a 5 s window. Section IV.B."""
    assert checks.RMS_NORMAL_LOAD_WINDOW == 5.0
    assert checks.RMS_NORMAL_LOAD_BANDS == {"moderate": 0.2, "severe": 0.3}


def test_the_index_declines_to_answer_on_a_run_shorter_than_its_window():
    """Misaka's Figs. 26-27 say the peaks do not survive the 5 s average.

    A 1.5 s core penetration is shorter than one window, and the honest answer
    there is "not computed" rather than a number. `kind` stays `report` so a UI
    cannot colour the refusal green.
    """
    traj, _ = _flat_traj(n=100, dt=0.01)  # 1.0 s
    c = checks.rms_normal_load(traj, REGISTRY["boeing747"], dt=0.01)
    assert c.kind == "report"
    assert c.passed is None
    assert math.isnan(c.value)
    assert "shorter than" in c.detail


def test_the_index_matches_a_direct_windowed_rms():
    """The full `rms_normal_load` path against the arithmetic used above.

    Ties the helper the other tests drive to the function the project ships, so
    they cannot drift apart.
    """
    traj, _ = _flat_traj(n=2000, dt=0.01)
    ac = REGISTRY["boeing747"]
    c = checks.rms_normal_load(traj, ac, dt=0.01)
    n_z = checks.load_factor_series(traj, ac)
    assert c.value == pytest.approx(_sigma_of(n_z, 0.01), rel=1e-12)


# ---------------------------------------------------------------------------
# Y22: a third CR-2144 flight condition, and what the model does at it.
# ---------------------------------------------------------------------------

_FL200_H = 6096.0


def _fl200_speed():
    return 0.8 * float(speed_of_sound(jnp.array(_FL200_H)))


def test_the_yoshimura_747_table_closes_on_itself():
    """Table A5's modes must follow from Table A2's derivatives.

    If they do not, the two tables came from different places and neither can
    be used. Recomputed rather than believed:
        wn^2  = Z_alpha*M_q/U0 - M_alpha
        zeta  = -(Z_alpha/U0 + M_q + M_alphadot) / (2 wn)
    """
    R = validation.REFERENCES
    Za = R["747fl200_Zalpha"].value
    Ma = R["747fl200_Malpha"].value
    Mad = R["747fl200_Malphadot"].value
    Mq = R["747fl200_Mq"].value
    U0 = R["747fl200_U0"].value

    wn = math.sqrt(Za * Mq / U0 - Ma)
    zeta = -(Za / U0 + Mq + Mad) / (2.0 * wn)
    assert wn == pytest.approx(R["747fl200_short_period_wn"].value, rel=0.005)
    assert zeta == pytest.approx(R["747fl200_short_period_zeta"].value, rel=0.005)


def test_table_a2_is_this_projects_own_747():
    """Table A2's dimensional column follows from THIS aircraft's geometry.

    The strongest available check that Yoshimura's reference and
    `aircraft.boeing747` are the same aeroplane: take Table A2's
    non-dimensional coefficients, apply the mass, inertia, wing area and MAC
    that `aircraft.py` transcribed from CR-2144 independently, and the
    dimensional column has to come back. Every entry lands within 0.5%.

    Note the two non-dimensionalisations are NOT alike -- Yoshimura's Eqs.
    (A11) and (A15) carry one power of U0 for the rate derivatives and
    `q_bar` (two powers) for the rest. Getting that wrong puts Z_q out by a
    factor of 250, which is how this test was debugged.
    """
    ac = REGISTRY["boeing747"]
    R = validation.REFERENCES
    V, rho = R["747fl200_U0"].value, 0.653  # Table A3's own values
    S, c = float(ac.S), float(ac.c)
    m, Iyy = float(ac.mass), float(ac.inertia[1, 1])
    q_bar = 0.5 * rho * V * V

    for name, computed in (
        ("747fl200_Zalpha", q_bar * S * R["747fl200_CZalpha"].value / m),
        ("747fl200_Malpha", q_bar * S * c * R["747fl200_Cmalpha"].value / Iyy),
        ("747fl200_Mq", rho * V * S * c * c * R["747fl200_Cmq"].value / (4 * Iyy)),
    ):
        assert computed == pytest.approx(R[name].value, rel=0.005), name

    # Same aeroplane by mass and inertia too -- Table A3 gives 2.89e5 kg and
    # 4.49e7 kg m^2, to three figures.
    assert m == pytest.approx(2.89e5, rel=0.005)
    assert Iyy == pytest.approx(4.49e7, rel=0.005)


def test_the_model_reproduces_the_published_lift_coefficient():
    """CL = 0.266 at M 0.8 / 6096 m, Table A3.

    A condition this project has never flown. Mass, wing area and the ISA
    atmosphere together have to land on the source's own CL, and they do to
    0.2%. This is the part of the comparison that is NOT circular: it uses no
    derivative from either document.
    """
    ac = REGISTRY["boeing747"]
    from atisim.atmosphere import G0, density

    V = _fl200_speed()
    rho = float(density(jnp.array(_FL200_H)))
    CL = float(ac.mass) * G0 / (0.5 * rho * V * V * float(ac.S))
    assert CL == pytest.approx(validation.REFERENCES["747fl200_CL"].value, rel=0.01)


def test_the_short_period_gap_at_fl200_is_the_derivative_set_not_the_solver():
    """Where the 23% frequency error at M 0.8 / 6096 m actually lives.

    `boeing747` carries CR-2144 flight condition 9 -- M 0.8 at 40,000 ft. Run
    at 6,096 m it is applying those derivatives 2.48x outside the dynamic
    pressure they were tabulated at, and the short-period frequency comes out
    23% high.

    The decomposition: swap Table A2's own coefficients for this condition into
    the SAME linearisation, change nothing else, and the frequency error
    collapses to under 1%. So the gap is the data, not the code -- which is a
    statement about provenance that no amount of staring at the solver could
    have produced.

    The residual damping error is separately attributable and separately
    checked below.
    """
    ac = REGISTRY["boeing747"]
    V = _fl200_speed()
    R = validation.REFERENCES

    def short_period(aircraft):
        x, _ = trim.trim(jnp.array(V), jnp.array(_FL200_H), aircraft)
        a, e, t = (float(v) for v in x)
        _, sp = validation.longitudinal_modes(aircraft, a, e, t, V, _FL200_H)
        return sp

    wn_ref = R["747fl200_short_period_wn"].value
    wn_own, _ = short_period(ac)
    swapped = ac._replace(
        CLa=jnp.array(-R["747fl200_CZalpha"].value),
        Cma=jnp.array(R["747fl200_Cmalpha"].value),
        Cmq=jnp.array(R["747fl200_Cmq"].value),
        CLq=jnp.array(-R["747fl200_CZq"].value),
    )
    wn_swapped, _ = short_period(swapped)

    assert abs(wn_own - wn_ref) / wn_ref == pytest.approx(0.235, abs=0.02)
    assert abs(wn_swapped - wn_ref) / wn_ref < 0.01


def test_the_residual_damping_gap_is_the_missing_alpha_dot_term():
    """And it is predictable in closed form, which is what makes it attributed.

    With Table A2's derivatives in place the damping is still ~12% low. This
    model has no C_m_alphadot -- PROJECT.md section 5 lists it among the terms
    genuinely absent -- and Table A2 gives M_alphadot = -0.176. Deleting that
    one term from the reference's own damping formula predicts the shortfall to
    better than a percentage point, so nothing else is contributing.
    """
    R = validation.REFERENCES
    Za, Ma = R["747fl200_Zalpha"].value, R["747fl200_Malpha"].value
    Mq, U0 = R["747fl200_Mq"].value, R["747fl200_U0"].value
    zeta_ref = R["747fl200_short_period_zeta"].value

    wn = math.sqrt(Za * Mq / U0 - Ma)
    zeta_without = -(Za / U0 + Mq) / (2.0 * wn)  # M_alphadot deleted
    predicted_shortfall = (zeta_without - zeta_ref) / zeta_ref
    assert predicted_shortfall == pytest.approx(-0.122, abs=0.01)

    ac = REGISTRY["boeing747"]
    V = _fl200_speed()
    swapped = ac._replace(
        CLa=jnp.array(-R["747fl200_CZalpha"].value),
        Cma=jnp.array(R["747fl200_Cmalpha"].value),
        Cmq=jnp.array(R["747fl200_Cmq"].value),
        CLq=jnp.array(-R["747fl200_CZq"].value),
    )
    x, _ = trim.trim(jnp.array(V), jnp.array(_FL200_H), swapped)
    a, e, t = (float(v) for v in x)
    _, (_, zeta_model) = validation.longitudinal_modes(swapped, a, e, t, V, _FL200_H)
    measured_shortfall = (zeta_model - zeta_ref) / zeta_ref

    # The prediction and the measurement agree to under one point of damping.
    assert abs(measured_shortfall - predicted_shortfall) < 0.015


def test_cr2144s_own_747_derivatives_move_with_altitude_at_constant_mach():
    """The flexible-airframe gap, finally with a number on it.

    PROJECT.md section 5 records that CR-2144's 747 derivative plots are
    labelled "Flexible" while `dynamics.py` integrates a rigid body, and says
    the mismatch is "Not quantified: doing so needs a rigid derivative set the
    project does not hold."

    It does not need one. For a RIGID aircraft the non-dimensional derivatives
    depend on Mach and geometry, not on altitude -- so at constant M 0.8 they
    should be identical at 40,000 ft and 20,000 ft. CR-2144's own are not, and
    the direction is the one aeroelastic relief predicts: at 2.48x the dynamic
    pressure every coefficient is LESS stiff.

    This is a bound on the flexible content, not a clean measurement of it. The
    weight matches to 0.07%, but CR-2144's CG for flight condition 9 is not in
    the material held here, and Table A3's is 25% MAC -- so a CG difference
    cannot be excluded, and it would land almost entirely on Cma, which is the
    largest mover. Recorded as "at most this much" for that reason.
    """
    ac = REGISTRY["boeing747"]
    R = validation.REFERENCES
    for name, own, ref in (
        ("CZalpha", -float(ac.CLa), R["747fl200_CZalpha"].value),
        ("Cmalpha", float(ac.Cma), R["747fl200_Cmalpha"].value),
        ("Cmq", float(ac.Cmq), R["747fl200_Cmq"].value),
        ("CZq", -float(ac.CLq), R["747fl200_CZq"].value),
    ):
        # Every one is stiffer at altitude, which is the sign that matters.
        assert abs(own) > abs(ref), name

    assert abs(float(ac.CLa) - 4.24) / 4.24 == pytest.approx(0.166, abs=0.02)
    assert abs(float(ac.Cma) - (-0.629)) / 0.629 == pytest.approx(0.609, abs=0.03)
    assert abs(float(ac.Cmq) - (-20.5)) / 20.5 == pytest.approx(0.167, abs=0.02)


def test_every_new_reference_carries_its_table():
    """PROJECT.md's standing rule, applied to the rows added this session."""
    for key, ref in validation.REFERENCES.items():
        if key.startswith("747fl200_"):
            assert "Yoshimura" in ref.source
            assert "Table" in ref.source


# ---------------------------------------------------------------------------
# The artifact round-trip. A new field on a serialised type is a silent-loss
# hazard until something asserts it survives.
# ---------------------------------------------------------------------------


def test_an_oblique_array_survives_the_artifact_round_trip():
    """`rebuild_field` must not quietly return a perpendicular traverse.

    An oblique array rebuilt without `cos_dpsi` is wrong by 1/cos(dpsi) in
    traverse time -- 17% for the Mehta case -- and the analysis UI would draw
    that silently, as a plausible run rather than as a failure. The default has
    to be there for artifacts written before session 23, so the only thing that
    can catch a missing read is a test.
    """
    from atisim.analysis import artifact

    array = wind.mehta_hannibal_array()
    meta = {"wind_field": {"kind": "VortexArray", "params": {
        "north": [float(v) for v in array.north],
        "down": [float(v) for v in array.down],
        "r0": float(array.r0), "v0": float(array.v0),
        "cos_dpsi": float(array.cos_dpsi),
    }}}
    field = artifact.rebuild_field(meta)

    for north in (-4000.0, -900.0, 0.0, 900.0, 4000.0):
        p = jnp.array([north, 0.0, -wind.MEHTA_HANNIBAL_ALTITUDE])
        np.testing.assert_allclose(
            np.asarray(field(p)),
            np.asarray(wind.vortex_wind(p, array)),
            rtol=0, atol=1e-13,
        )


def test_an_artifact_without_cos_dpsi_rebuilds_as_perpendicular():
    """Every artifact written before session 23 has to come back unchanged.

    The negative control on the test above: the default must be exactly 1.0, so
    an older `params` block with no `cos_dpsi` key reproduces the field it was
    written from rather than raising or drifting.
    """
    from atisim.analysis import artifact

    meta = {"wind_field": {"kind": "VortexArray", "params": {
        "north": [0.0, 1066.8], "down": [-11278.0, -11278.0],
        "r0": _R0, "v0": _V0,
    }}}
    field = artifact.rebuild_field(meta)
    plain = wind.VortexArray(
        north=jnp.array([0.0, 1066.8]), down=jnp.array([-11278.0, -11278.0]),
        r0=jnp.array(_R0), v0=jnp.array(_V0),
    )
    for north in (-500.0, 0.0, 500.0, 1500.0):
        p = jnp.array([north, 0.0, -11278.0 - 80.0])
        got = np.asarray(field(p))
        want = np.asarray(wind.vortex_wind(p, plain))
        assert got.tolist() == want.tolist(), north
