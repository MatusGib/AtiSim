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

    The papers agree on the RADIUS nowhere and on the STRENGTH nearly: Parks
    says 85 ft/s with a 600 ft core, Mehta 86.8 with 500.5, TM-102186 87 with
    500. `vortex_wind` multiplies by v0 exactly once on every branch, so the
    induced wind scales exactly and the spread on any wind-derived quantity is
    that same 2.1%. (TM-102186 rounds Mehta's 86.8 to 87, which against Parks
    is 2.35% -- so the sources span 2.1-2.4% depending on which pair is taken,
    and the tests pin the Mehta figure.)

    THIS TEST IS ABOUT THE STRENGTH ALONE and deliberately holds the radius
    fixed, which is what makes it a clean scaling check rather than a mixture
    of the two disagreements. The radius disagreement is `ASSUMPTIONS.md` E12.
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


# ---------------------------------------------------------------------------
# Session 23, follow-up: the four bounding experiments.
#
# MIL-F-8785C for the Dryden form; the rest is measured by
# scripts/cat_bounds.py and pinned here.
# ---------------------------------------------------------------------------


def test_the_dryden_spectrum_is_one_sided():
    """Integrated 0 -> infinity it must give sigma^2, not 2 sigma^2.

    The convention is the whole ballgame: getting it wrong is a factor of two
    in variance and sqrt(2) in every gust, and nothing downstream would look
    obviously wrong. Checked by quadrature against the closed form rather than
    by reading the constant off the source.
    """
    from scipy import integrate as si

    sigma = 3.0
    value, _ = si.quad(
        lambda w: float(wind.dryden_spectrum(jnp.array(w), sigma)), 0.0, np.inf,
        limit=400,
    )
    assert value == pytest.approx(sigma**2, rel=1e-6)


def test_a_dryden_realisation_has_the_variance_it_claims():
    """The realised standard deviation must come back as sigma_w.

    This is what makes the sum-of-sinusoids construction a Dryden field rather
    than merely a field with Dryden's shape. Sampled over 60 scale lengths so
    the low-wavenumber components are exercised.

    98% rather than 100%: `wavelength_max` truncates the low-wavenumber tail,
    and that truncation is the only reason this is not exact. Asserted as a
    band with a floor, so a future change that quietly loses variance fails.
    """
    sigma = 3.0
    field = wind.dryden_vertical_field(sigma, seed=0)
    x = np.linspace(0.0, 60.0 * wind.DRYDEN_LW, 6000)
    w = np.array([float(field(jnp.array([xi, 0.0, -11278.0]))[2]) for xi in x])
    assert w.std() == pytest.approx(sigma, rel=0.06)
    assert w.std() < sigma  # truncated tail can only remove variance
    assert abs(w.mean()) < 0.1 * sigma


def test_dryden_realisations_differ_by_seed_and_repeat_by_seed():
    """A realisation is one sample. Two seeds must differ; one seed must not.

    The negative control on the ensemble runs: if seeds did nothing, the spread
    reported over them would be an artefact and the sweep would be one run
    quoted six times.
    """
    p = jnp.array([1234.0, 0.0, -11278.0])
    a = float(wind.dryden_vertical_field(3.0, seed=0)(p)[2])
    b = float(wind.dryden_vertical_field(3.0, seed=1)(p)[2])
    again = float(wind.dryden_vertical_field(3.0, seed=0)(p)[2])
    assert a != b
    assert a == again


def test_the_dryden_field_is_differentiable():
    """`field_model` takes jacfwd of the field for the gust rates.

    A grid realisation with interpolation would give a piecewise-constant
    q_gust -- the very channel the vortex work is about -- which is why this
    field is a sum of sinusoids. If it ever stops being analytically
    differentiable, this is where it shows.
    """
    import jax

    field = wind.dryden_vertical_field(3.0, seed=0)
    jac = jax.jacfwd(field)(jnp.array([500.0, 0.0, -11278.0]))
    assert np.isfinite(np.asarray(jac)).all()
    assert abs(float(jac[2, 0])) > 0.0


def test_the_747_declares_the_band_its_derivatives_were_tabulated_in():
    """CR-2144 flight condition 9 is M 0.80 at 40,000 ft, and it is now said.

    Until session 23 this entry declared nothing, so `checks.recovery_band`
    degraded to a report and a run at 20,000 ft -- 23.5% wrong in short-period
    frequency -- passed in silence.
    """
    ac = REGISTRY["boeing747"]
    lo, hi = (float(v) for v in ac.valid_altitude)
    assert lo == pytest.approx(35000.0 * FT2M)
    assert hi == pytest.approx(45000.0 * FT2M)
    m_lo, m_hi = (float(v) for v in ac.valid_mach)
    assert (m_lo, m_hi) == (0.70, 0.90)
    # The tabulated condition must be inside the band it declares.
    assert lo < 40000.0 * FT2M < hi
    assert m_lo < 0.80 < m_hi


def test_the_band_admits_the_mehta_run_and_refuses_the_fl200_one():
    """The band has to earn its place: it must pass what the project flies and
    fail the case that motivated it.

    Mehta's encounter is at 37,000 ft, inside. The Yoshimura comparison
    condition is 6,096 m, outside by a wide margin. A band that did not
    separate those two would be decoration.
    """
    ac = REGISTRY["boeing747"]
    lo, hi = (float(v) for v in ac.valid_altitude)
    assert lo <= wind.MEHTA_HANNIBAL_ALTITUDE <= hi
    assert not (lo <= _FL200_H <= hi)
    # Lester's Greenland case is outside too, and that is reported rather than
    # hidden -- see scripts/cat_bounds.py section C.
    assert not (lo <= wind.LESTER_GREENLAND_ALTITUDE <= hi)


def test_restoring_the_alpha_dot_term_closes_the_fl200_damping_gap():
    """The prediction that made C_m_alphadot worth chasing, run forward.

    With Table A2's other derivatives in place the damping is 11.7% low and the
    shortfall is attributable in closed form to the missing M_alphadot. Put
    Table A2's own C_m_alphadot in and the gap has to close -- if it does not,
    the attribution was wrong.
    """
    ac = REGISTRY["boeing747"]
    R = validation.REFERENCES
    V = _fl200_speed()
    swapped = ac._replace(
        CLa=jnp.array(-R["747fl200_CZalpha"].value),
        Cma=jnp.array(R["747fl200_Cmalpha"].value),
        Cmq=jnp.array(R["747fl200_Cmq"].value),
        CLq=jnp.array(-R["747fl200_CZq"].value),
    )
    restored = swapped._replace(Cmadot=jnp.array(R["747fl200_Cmalphadot"].value))

    def zeta(a):
        x, _ = trim.trim(jnp.array(V), jnp.array(_FL200_H), a)
        al, e, t = (float(v) for v in x)
        return validation.longitudinal_modes(a, al, e, t, V, _FL200_H)[1][1]

    ref = R["747fl200_short_period_zeta"].value
    assert abs(zeta(swapped) - ref) / ref == pytest.approx(0.117, abs=0.02)
    assert abs(zeta(restored) - ref) / ref < 0.01


def test_strip_loads_are_a_no_op_on_a_spanwise_uniform_field():
    """Measured, not assumed -- and it bounds what item 5 could ever have found.

    `loads.strip_increment` is roll-only, and `vortex_wind` has no y dependence
    at all, so integrating across the span must return exactly the point value.
    That is why the point-gust assumption on the Mehta run had to be bounded
    through the GUST RATES instead: the strip path cannot see this field.
    """
    from atisim import loads

    ac = REGISTRY["boeing747"]
    array = wind.mehta_hannibal_array()
    field = lambda p: wind.vortex_wind(p, array)  # noqa: E731
    st = _stations(ac)
    state = trim.trimmed_state(
        jnp.array(0.05), jnp.array(235.9), jnp.array(wind.MEHTA_HANNIBAL_ALTITUDE)
    )._replace(pos_ned=jnp.array([300.0, 0.0, -wind.MEHTA_HANNIBAL_ALTITUDE]))
    inc = loads.strip_increment(state, field, ac, st)
    assert float(inc.Cl) == pytest.approx(0.0, abs=1e-12)


def _stations(ac):
    from atisim import airframe

    return airframe.stations(ac)


def test_the_approach_747s_tabulated_alpha_dot_terms_are_a_trade_not_a_win():
    """Why Table IX-2's C_L_alphadot and C_m_alphadot are NOT applied.

    They are tabulated -- aircraft.py's own comment records CL_alphadot -6.7
    and Cm_alphadot -3.2 -- and the model has had the fields to hold them since
    the alpha-dot work. Session 23 measured what applying them would do, and
    the answer is a trade rather than an improvement:

        short period wn   +1.4%  ->  -0.3%      against Caughey Eq. (5.54)
        short period z    -5.5%  ->  +0.1%
        phugoid wn        -0.2%  ->  -0.2%
        phugoid z         -4.6%  ->  +8.4%      tolerance asserted is 5%

    The mode alpha-dot physically governs improves markedly; the mode it does
    not touch degrades past the tolerance test_validation.py asserts. Adopting
    them would mean re-pinning that tolerance to let the change through, which
    this project does not do -- see docs/ASSUMPTIONS.md B5 for the precedent.

    THE SIGN IS RESOLVED, AND EMPIRICALLY. The comment transcribes -6.7 but
    Caughey uses +6.7 for the same CR-2144 case, and +6.7 is what a conventional
    aft tail must have -- downwash lag makes the tail see MORE incidence as
    alpha rises. Flown both ways, +6.7 takes short-period wn to -0.3% and -6.7
    takes it to +3.2%, so the data agrees with the physics.

    This test exists so the measurement survives the decision not to act on it.
    """
    ac = REGISTRY["boeing747_approach"]
    R = validation.REFERENCES
    from atisim.aircraft import CRUISE

    V = CRUISE["boeing747_approach"]["airspeed"]
    H = CRUISE["boeing747_approach"]["altitude"]

    def modes(a):
        x, _ = trim.trim(jnp.array(V), jnp.array(H), a)
        al, e, t = (float(v) for v in x)
        return validation.longitudinal_modes(a, al, e, t, V, H)

    with_adot = ac._replace(Cmadot=jnp.array(-3.2), CLadot=jnp.array(6.7))
    wrong_sign = ac._replace(Cmadot=jnp.array(-3.2), CLadot=jnp.array(-6.7))

    (_, ph_z0), (sp_w0, sp_z0) = modes(ac)
    (_, ph_z1), (sp_w1, sp_z1) = modes(with_adot)
    (_, _), (sp_w2, _) = modes(wrong_sign)

    sp_w_ref = R["747pa_short_period_wn"].value
    sp_z_ref = R["747pa_short_period_zeta"].value
    ph_z_ref = R["747pa_phugoid_zeta"].value

    # The short period improves on both counts.
    assert abs(sp_w1 - sp_w_ref) < abs(sp_w0 - sp_w_ref)
    assert abs(sp_z1 - sp_z_ref) < abs(sp_z0 - sp_z_ref)
    # Caughey's sign beats the transcribed one on the frequency.
    assert abs(sp_w1 - sp_w_ref) < abs(sp_w2 - sp_w_ref)
    # And the phugoid damping goes the other way, past its asserted 5%.
    assert abs(ph_z0 - ph_z_ref) / ph_z_ref < 0.05
    assert abs(ph_z1 - ph_z_ref) / ph_z_ref > 0.05

    # The shipped entry still carries neither. If that changes, this test is
    # the thing that has to be revisited first.
    assert float(ac.CLadot) == 0.0
    assert float(ac.Cmadot) == 0.0


# ---------------------------------------------------------------------------
# Session 23c. What the Hannibal comparison is worth once the inputs carry
# error, and the one channel the identification did not set.
#
# The standing problem with every load comparison in this project is that the
# vortex parameters were identified FROM the recorded accelerations, through
# somebody else's aircraft model. Predicting accelerations from them therefore
# partly re-derives the fit. These tests attack that from both ends: a channel
# outside the loop (gust TIMING), and a bound on what the loop's inputs are
# actually known to (Mehta's own residual, Lester's reconstruction error).
# ---------------------------------------------------------------------------

_MEHTA_LEAD_R0 = 12.0


def _mehta_run(*, v0_scale=1.0, r0_scale=1.0, dt=0.02):
    """The headline Mehta run with the identified parameters perturbed.

    dt 0.02 rather than the script's 0.01: PROJECT.md section 4 measures the
    peak load as converged to 0.14% over an eightfold step range, so the
    coarsest measured step is the right one to pay for in a test suite.
    """
    from atisim import vortex_viz
    from atisim.aircraft import CRUISE

    ac = REGISTRY["boeing747"]
    V = CRUISE["boeing747"]["airspeed"]
    H = wind.MEHTA_HANNIBAL_ALTITUDE
    array = wind.mehta_hannibal_array(H)
    array = array._replace(v0=array.v0 * v0_scale, r0=array.r0 * r0_scale)
    r0 = float(wind.MEHTA_HANNIBAL_R0)
    x0, x1 = float(array.north.min()), float(array.north.max())
    start = x0 - _MEHTA_LEAD_R0 * r0
    enc = vortex_viz.fly_in_moving_air(
        ac, lambda p: wind.vortex_wind(p, array), V, H, label="mehta",
        start_north=start, seconds=(x1 + _MEHTA_LEAD_R0 * r0 - start) / V,
        dt=dt, window=(x0 - 2.0 * r0, x1 + 2.0 * r0), window_name="array",
    )
    w = enc.window
    return enc, (float(enc.n_z[w].min()), float(enc.n_z[w].max()))


def _gust_times(t, w_up, floor):
    """Times of the sharp gust extrema -- turning points reaching `floor` m/s.

    `floor` is ABSOLUTE rather than a fraction, because the sweeps below scale
    the field: a fixed fraction of the unscaled V0 would select a different set
    of extrema at each scale and the comparison would be of thresholds rather
    than of gusts. At three times V0 it picks up the outer two vortices, which
    are not the "sharp" gusts the record is about.
    """
    d = np.sign(np.diff(w_up))
    turn = np.where(d[:-1] * d[1:] < 0)[0] + 1
    keep = [k for k in turn if abs(w_up[k]) >= floor]
    return np.asarray([t[k] for k in keep])


def _sampled_gust_times(*, v0_scale=1.0, r0_scale=1.0, n=40001):
    """The same extrema, read off the FIELD along the nominal straight path.

    No integration: this isolates the field's own geometry from the aircraft's
    response to it, which is what the invariance claim below is about.
    """
    import jax

    from atisim.aircraft import CRUISE

    V = CRUISE["boeing747"]["airspeed"]
    H = wind.MEHTA_HANNIBAL_ALTITUDE
    array = wind.mehta_hannibal_array(H)
    array = array._replace(v0=array.v0 * v0_scale, r0=array.r0 * r0_scale)
    r0 = float(wind.MEHTA_HANNIBAL_R0)
    x = np.linspace(float(array.north.min()) - 2.0 * r0,
                    float(array.north.max()) + 2.0 * r0, n)
    pos = jnp.stack([jnp.asarray(x), jnp.zeros(n), jnp.full(n, -H)], axis=1)
    w_ned = jax.vmap(lambda p: wind.vortex_wind(p, array))(pos)
    floor = 0.40 * wind.MEHTA_HANNIBAL_V0 * v0_scale
    return _gust_times(x / V, -np.asarray(w_ned[:, 2]), floor)


def test_the_gust_spacing_is_untouched_by_the_strength_that_was_fitted():
    """Why timing is evidence where amplitude is partly circular.

    V0 was identified by minimising a residual against winds derived from the
    recorded accelerations. Predicting those accelerations back therefore tests
    the composition of two aircraft models rather than this one alone.

    The gust SPACING is outside that loop, and this test says so exactly rather
    than by assertion: the field is linear in V0, so scaling it moves every
    wind value and no turning point at all. r0 does move them -- a Rankine
    extremum sits on the core boundary -- but only within a core, so the
    spacing between two cores barely notices.
    """
    base = _sampled_gust_times()
    assert len(base) == 4, "expected two up-and-down pairs, one per core"

    # V0: exactly invariant, to the resolution of the sampling grid.
    for scale in (0.5, 1.5, 3.0):
        assert _sampled_gust_times(v0_scale=scale) == pytest.approx(base)

    # r0: the extrema move, and the SPACING still does not, to under 2%.
    span = base[2] - base[0]
    for scale in (0.85, 1.15):
        moved = _sampled_gust_times(r0_scale=scale)
        assert not np.allclose(moved, base), "r0 must move the peaks themselves"
        assert abs((moved[2] - moved[0]) - span) / span < 0.02


def test_the_flown_gust_spacing_reproduces_the_recorded_five_seconds():
    """TM-102186's "about 5 sec apart", against the flown vertical wind.

    Stated in the paper's PROSE (p. 3-4), not read off a figure, so there is no
    digitisation error on the reference -- only the author's "about".

    The residual is a speed difference, not a field error. Exactly 5.0 s needs
    250 m/s over the 4,104 ft core separation; this 747 flies 236 m/s because
    that is the Mach its derivative set is tabulated at. The DC-10's own true
    airspeed appears in no source held here, and 6% between two transports at
    37,000 ft is unremarkable.
    """
    enc, _ = _mehta_run()
    w = enc.window
    times = _gust_times(enc.t[w], enc.w_up[w], 0.40 * wind.MEHTA_HANNIBAL_V0)
    assert len(times) == 4

    up1, dn1, up2, dn2 = times
    readings = {
        "peak-to-peak": up2 - up1,
        "trough-to-trough": dn2 - dn1,
        "centre-to-centre": (up2 + dn2) / 2 - (up1 + dn1) / 2,
    }
    # The claim must not depend on which reading of "apart" was taken.
    assert max(readings.values()) - min(readings.values()) < 0.20

    for name, s in readings.items():
        rel = abs(s - wind.TM102186_HANNIBAL_GUST_PERIOD) / \
            wind.TM102186_HANNIBAL_GUST_PERIOD
        assert rel < 0.10, f"{name} {s:.3f} s"

    # And the model is SLOW, not fast -- which is the direction a 747 at M 0.80
    # against a faster DC-10 has to be. A fast model would need explaining.
    assert min(readings.values()) > wind.TM102186_HANNIBAL_GUST_PERIOD


def test_mehtas_cost_converts_without_n_and_beats_nothing_it_should_not():
    """Eq. (A3) carries a 1/N, so J is a mean square and N is not needed.

    That is the whole reason this bound exists. Read as a SUM the cost would be
    uninterpretable: at any plausible N -- the record is about a minute of
    1-to-4 Hz DFDR data -- the implied residual lands well below the error of the
    winds being fitted, by a factor of 4.1 even at N = 30, which no honest fit
    can do.

    So the mean-square reading is checked here the only way it can be, by its
    consequence: the residual must EXCEED the reconstruction error of its own
    data, and it does, by a margin that leaves room for real unmodelled wind.
    """
    resid = wind.mehta_residual_ceiling()
    measured = math.sqrt(sum(v ** 2 for v in wind.DFDR_WIND_RMS_ERROR.values()))

    assert resid == pytest.approx(math.sqrt(wind.MEHTA_COST[5]) * FT2M)
    assert resid > measured, "a fit cannot track data better than the data are"
    # Read as a sum over even 30 points the residual would be 0.81 m/s, i.e.
    # a quarter of the data error. This is that reading failing.
    assert math.sqrt(wind.MEHTA_COST[5] / 30.0) * FT2M < measured


def test_the_five_vortex_array_is_where_the_model_family_runs_out():
    """The converged costs fall monotonically and then stop paying.

    Mehta p. 30 states n = 6, 7 do not lower the cost at all -- the algorithm
    pushes the extra vortices out of the flight path. So J(5) is a FLOOR for a
    Rankine array against this record, which is what lets the residual bound
    the field FORM rather than one author's stopping rule.
    """
    ns = sorted(wind.MEHTA_COST)
    assert ns == [2, 3, 4, 5]
    costs = [wind.MEHTA_COST[n] for n in ns]
    assert costs == sorted(costs, reverse=True)

    # The last vortex buys under 3% of RMS, against 14% for the first refit.
    assert math.sqrt(wind.MEHTA_COST[5] / wind.MEHTA_COST[4]) > 0.97
    assert wind.MEHTA_COST_SATURATES_AT == max(ns)

    # 482 is the MANUAL startup estimate and is kept out of the fitted series.
    # Pairing it with a converged cost is the error this project caught
    # TM-102186 making with Schultz's Table 1 -- see PROJECT.md section 5.
    assert wind.MEHTA_COST_STARTUP not in costs
    assert wind.MEHTA_COST_STARTUP > costs[0]


def test_the_unmodelled_wind_is_bracketed_by_two_sourced_bounds():
    """What the vortex array leaves in the air, from two independent papers.

    Mehta's residual is against RECONSTRUCTED winds, so it contains Lester's
    reconstruction error. Removing that in quadrature leaves the physical
    fluctuation the model omits -- which Mehta names on p. 30 as "the small,
    random fluctuations that are part of the overall turbulence".

    Both ends are needed. The lower bound assumes independence and takes
    Lester's errors at face value, and both assumptions push the true value UP;
    the ceiling assumes the entire residual is vertical and unmodelled, which
    pushes it as far up as it can go.
    """
    floor = wind.mehta_unmodelled_wind()
    ceiling = wind.mehta_residual_ceiling()
    assert 0.0 < floor < ceiling

    measured = sum((v / FT2M) ** 2 for v in wind.DFDR_WIND_RMS_ERROR.values())
    assert floor == pytest.approx(
        math.sqrt((wind.MEHTA_COST[5] - measured) / 2.0) * FT2M)

    # The startup fit had further to fall, so it must bound looser at both ends.
    assert wind.mehta_unmodelled_wind(2) > floor
    assert wind.mehta_residual_ceiling(2) > ceiling


def test_the_propagated_input_band_does_not_reach_the_recorded_load():
    """The 32% shortfall is not inside the uncertainty of the inputs.

    Flown at the most favourable corner of both sweeps -- V0 at the top of
    Lester's error and r0 at the bottom of a declared +/-15% -- the run still
    reaches under three quarters of the DC-10's recorded peak-to-peak. That
    turns "68% of the record" from a number with no error bar into a bounded
    statement: no admissible choice of the identified parameters closes it.

    r0's range is DECLARED, not sourced. Mehta's sensitivity study reports
    convergence from initial guesses of 100-1300 ft, which is a statement about
    his algorithm and not about how well r0 is known.
    """
    lo_rec, hi_rec = wind.TM102186_HANNIBAL_NZ
    recorded = hi_rec - lo_rec
    frac = wind.DFDR_WIND_RMS_ERROR["vertical"] / wind.MEHTA_HANNIBAL_V0
    assert frac == pytest.approx(0.0845, abs=0.0005)

    _, (lo, hi) = _mehta_run(v0_scale=1.0 + frac, r0_scale=0.85)
    assert (hi - lo) / recorded < 0.75
    # And it is an improvement on the unperturbed run, so the corner really is
    # the favourable one rather than merely a different one.
    _, (lo0, hi0) = _mehta_run()
    assert (hi - lo) > (hi0 - lo0)


def test_no_gust_strength_reaches_the_recorded_peak_inside_the_linear_range():
    """The peak load is saturated: the aircraft pitches away and sheds the gust.

    This is TM-102186 Fig. 8's incidence-gain mechanism seen from the inside,
    and it is why the load shortfall cannot be an amplitude error. TRIPLING the
    identified V0 leaves the up-increment near half the recorded one and peak
    |alpha| within a degree of where it started -- the relief scales with the
    gust.

    THE TWO BOUNDARIES COINCIDE, which is the sharp form of the claim and the
    reason this test brackets rather than sampling one point. The peak first
    reaches the recorded +1.7 g between 3.25 and 3.5 times V0, and |alpha|
    leaves the 10 deg linear range in the SAME interval:

        x3.25   n_z max 1.642   |alpha| 9.12 deg    short,   inside
        x3.50   n_z max 1.787   |alpha| 10.35 deg   reaches, OUTSIDE

    So there is no gust strength at which this model both reaches the record and
    may be believed. A coarser sweep would have supported only "the shortfall is
    large", which is a weaker statement and would not have excluded amplitude.
    """
    from atisim.units import RAD2DEG

    def peak_alpha(enc):
        return float(np.abs(enc.alpha_air[enc.window]).max() * RAD2DEG)

    _, (_, hi0) = _mehta_run()
    recorded = wind.TM102186_HANNIBAL_NZ[1]

    # Tripling barely moves the peak: a response that tracked the gust would
    # have elasticity ~1, and this one is under 0.2 and changes sign en route.
    enc3, (_, hi3) = _mehta_run(v0_scale=3.0)
    assert hi3 - 1.0 < 0.85 * (recorded - 1.0)
    assert abs(((hi3 - 1.0) / (hi0 - 1.0) - 1.0) / 2.0) < 0.20
    assert peak_alpha(enc3) < 10.0

    # The bracket. Below the crossing the model is believable and short; above
    # it the model reaches and is outside its own validity.
    enc_lo, (_, hi_lo) = _mehta_run(v0_scale=3.25)
    enc_hi, (_, hi_hi) = _mehta_run(v0_scale=3.5)

    assert hi_lo < recorded and peak_alpha(enc_lo) < 10.0
    assert hi_hi >= recorded and peak_alpha(enc_hi) > 10.0


# ---------------------------------------------------------------------------
# Session 23d. The headline field flown by a second engine, and the Fig. 8
# categories given error bars.
#
# The cross-code arm exists because "both engines under-predict by the same
# amount, so the shortfall is in the inputs" was measured on the PARKS single
# cores and then applied to the MEHTA array. These tests are what happens when
# that inference is checked on the field it was applied to.
# ---------------------------------------------------------------------------


def _mehta_cross_code():
    """(JSBSim encounter, atisim run) on Mehta's field, one state, one field.

    Rebuilds what `scripts/vortex_compare.py` does, because the test suite may
    not import from `scripts/`. Translation-only on the atisim side: JSBSim has
    no writable gust-rate input, so the gradient terms have no counterpart and
    including them would not be a comparison.
    """
    from atisim import jsbsim_vortex_ref, vortex_viz
    from atisim.state import State, euler_to_quat
    from atisim.aircraft import REGISTRY as REG
    from atisim.dynamics import Controls

    enc = jsbsim_vortex_ref.load().encounters[("mehta", "mehta")]
    v = enc.values
    altitude = v["matched_altitude"]
    z = v["altitude"] + enc.cores[1]
    array = wind.VortexArray(
        north=jnp.array(enc.cores[0]), down=jnp.array(-(altitude - z)),
        r0=jnp.array(v["r0"]), v0=jnp.array(v["v0"]),
        cos_dpsi=jnp.array(v["cos_dpsi"]),
    )
    field = lambda p: wind.vortex_wind(p, array)  # noqa: E731
    state = State(
        pos_ned=jnp.array([0.0, 0.0, -altitude]),
        vel_body=jnp.array(enc.initial.vel_body),
        quat=euler_to_quat(*(jnp.array(x) for x in enc.initial.euler)),
        omega=jnp.array(enc.initial.omega),
    )
    controls = Controls(
        elevator=jnp.array(enc.initial.controls[0]),
        aileron=jnp.array(enc.initial.controls[1]),
        rudder=jnp.array(enc.initial.controls[2]),
        throttle=jnp.array(enc.initial.throttle),
    )
    # Translation-only, which is the strictly like-for-like arm: JSBSim samples
    # wind at one point and its gust-rate inputs are read-only, so it carries no
    # gradient at all. Same shape as scripts/vortex_compare.partial_field_model
    # with both terms off.
    def translational(wind_state, st, key, dt):
        del dt
        return field(st.pos_ned), jnp.zeros(3), wind_state, key, jnp.array(0.0)

    run = vortex_viz.fly_from_state(
        REG[enc.aircraft], field, state, controls, label="mehta-xcode",
        seconds=v["duration"], dt=0.02,
        window=enc.window_bounds(), window_name="array",
        wind_model=translational,
    )
    return enc, run


def test_the_two_field_implementations_agree_on_the_oblique_array():
    """The gate the array comparison rests on, and it is a NEW gate.

    `test_jsbsim_vortex.py` already reconciles the numpy and JAX fields for the
    single cores. The array adds two things neither had: superposition of five
    cores, and an oblique traverse at 31 deg. A bug in either would sit on both
    sides of the load comparison below and be invisible to it.
    """
    from atisim import jsbsim_vortex_ref

    enc = jsbsim_vortex_ref.load().encounters[("mehta", "mehta")]
    v = enc.values
    assert enc.cores is not None and len(enc.cores[0]) == 5
    assert v["cos_dpsi"] == pytest.approx(
        math.cos(math.radians(wind.MEHTA_HANNIBAL_PSI_DEG)))

    array = wind.VortexArray(
        north=jnp.array(enc.cores[0]), down=jnp.array(enc.cores[1]),
        r0=jnp.array(v["r0"]), v0=jnp.array(v["v0"]),
        cos_dpsi=jnp.array(v["cos_dpsi"]),
    )
    worst = 0.0
    for s in enc.samples:
        mine = np.asarray(
            wind.vortex_wind(jnp.array([s.north, 0.0, -s.altitude]), array))
        worst = max(worst, float(np.abs(mine - s.wind).max()))
    assert worst < 1e-9, f"the two array implementations disagree by {worst:.3e}"


def test_neither_engine_reaches_the_record_on_the_field_it_was_fitted_to():
    """*** The result step 5 existed to get. ***

    The load shortfall could always have been a solver defect. It is not: an
    independent flight-dynamics engine, given the SAME five-vortex field, the
    SAME aircraft (boeing747_jsbsim, recovered from JSBSim's own B747) and the
    SAME starting state, also falls well short of the DC-10's recorded
    peak-to-peak. JSBSim reaches about three quarters of it; atisim about two
    thirds. Neither reaches.

    That is what closes the solver as an explanation on the headline field
    rather than on the adjacent Parks cases it was previously inferred from.
    """
    from atisim import jsbsim_vortex_ref

    enc = jsbsim_vortex_ref.load().encounters[("mehta", "mehta")]
    lo, hi = enc.window_bounds()
    js = np.array([s.Nz for s in enc.samples if lo <= s.north <= hi])
    recorded = wind.TM102186_HANNIBAL_NZ[1] - wind.TM102186_HANNIBAL_NZ[0]

    assert 0.70 < np.ptp(js) / recorded < 0.80
    assert np.ptp(js) < recorded, "JSBSim must fall short too, or step 5 is moot"


def test_the_two_engines_part_on_the_array_where_they_did_not_on_one_core():
    """And this is the part that QUALIFIES a previous session's conclusion.

    Session 22 measured both engines under-predicting the DFDR "by the same
    amount" and read that as the shortfall being in the inputs. It was measured
    on single Parks cores, where the window is 1.3 s -- a fifth of a short
    period -- and the two engines put their load extremes at the same point to
    within a couple of metres.

    On the five-core array the window is 34 s, five short periods, and they do
    not: the engines select DIFFERENT cores as the worst one. The peak agrees
    (atisim is 2% higher); the trough does not, and the trough is the channel
    that has been short against the DFDR all along.

    So "the same amount" was a property of the short single-core window, not a
    general result, and the array comparison must be quoted on its own terms.
    """
    from atisim import jsbsim_vortex_ref

    ref = jsbsim_vortex_ref.load()
    arr = ref.encounters[("mehta", "mehta")]
    one = ref.encounters[("hannibal", "parks")]

    lo, hi = arr.window_bounds()

    def extremes(enc):
        lo, hi = enc.window_bounds()
        w = [s for s in enc.samples if lo <= s.north <= hi]
        nz = np.array([s.Nz for s in w])
        north = np.array([s.north for s in w])
        return north[nz.argmax()], north[nz.argmin()], np.ptp(nz)

    _, _, span_one = extremes(one)
    n_max, n_min, span_arr = extremes(arr)

    # JSBSim puts both of its own extremes on one core; atisim puts both of its
    # own on a DIFFERENT one. That is the cross-engine claim, and it needs
    # atisim actually flown -- the frozen reference alone cannot make it.
    cores = arr.cores[0]
    _, run = _mehta_cross_code()
    w = run.window
    a_north, a_nz = np.asarray(run.north[w]), np.asarray(run.n_z[w])
    js_core = int(np.abs(cores - n_max).argmin())
    at_core = int(np.abs(cores - a_north[a_nz.argmax()]).argmin())
    assert js_core == int(np.abs(cores - n_min).argmin())
    assert at_core != js_core, (
        f"both engines chose core {js_core}; the divergence this test records "
        f"is gone and the docstring above no longer describes the tree"
    )

    # The peak agrees far better than the span does: the disagreement is in the
    # TROUGH, which is the channel short against the DFDR in the first place.
    assert abs(a_nz.max() - np.array([s.Nz for s in arr.samples
                                      if lo <= s.north <= hi]).max()) < 0.06
    assert a_nz.min() > np.array([s.Nz for s in arr.samples
                                  if lo <= s.north <= hi]).min() + 0.15

    # The window that produced the earlier "same amount" reading is a fraction
    # of a short period; this one is several. That is the difference, stated in
    # the quantity that causes it rather than in the conclusion it changes.
    V = arr.values["airspeed"]
    assert (one.window_bounds()[1] - one.window_bounds()[0]) / V < 2.0
    assert (arr.window_bounds()[1] - arr.window_bounds()[0]) / V > 30.0
    assert span_arr > span_one


def test_the_fig8_ordering_survives_a_sourced_random_layer():
    """PROJECT.md section 7, step 6 -- its own stated verify criterion.

    Step 6 has waited on step 4 (Dryden) since session 3. The criterion written
    there is that the vortex < updraft < manoeuvre ordering holds across the
    ensemble, and it does, at the top of the sigma_w range session 23c derived
    from Mehta's residual.

    WHAT THE SCRIPT REPORTS THAT THIS DOES NOT: the pitch clouds nearly touch at
    that intensity -- a 0.06 deg gap, down from 0.886 at the low end -- so the
    discriminator survives on its LOAD axis rather than its pitch axis. Two
    seeds cannot establish that, so it is measured in scripts/cat_ensemble.py
    and only the ordering is gated here.
    """
    from atisim import vortex_viz
    from atisim.aircraft import CRUISE
    from atisim.wind import PARKS_CASES, UPDRAFT_SECONDS, UPDRAFT_W0

    ac = REGISTRY["boeing747"]
    V, H = CRUISE["boeing747"]["airspeed"], CRUISE["boeing747"]["altitude"]
    sigma = wind.mehta_residual_ceiling()
    case = PARKS_CASES["hannibal"]
    r0, spacing = case["r0"], case["spacing"]
    radius = 0.5 * UPDRAFT_SECONDS * V

    for seed in (0, 1):
        extra = wind.dryden_vertical_field(sigma, seed)

        array = wind.VortexArray(
            north=jnp.array([0.0, spacing]), down=jnp.array([-H, -H]),
            r0=jnp.array(r0), v0=jnp.array(case["v0"]),
        )
        vortex = vortex_viz.fly(
            ac, wind.superpose(lambda p: wind.vortex_wind(p, array), extra),
            V, H, label="v", start_north=-40.0 * r0,
            seconds=(spacing + 40.0 * r0 + 6.0 * r0) / V, dt=0.05,
            window=(-r0, r0), window_name="first core")

        column = wind.UpdraftColumn(
            north=jnp.array(0.0), east=jnp.array(0.0),
            w0=jnp.array(UPDRAFT_W0), radius=jnp.array(radius),
            sharpness=jnp.array(6.0))
        updraft = vortex_viz.fly(
            ac, wind.superpose(lambda p: wind.updraft_wind(p, column), extra),
            V, H, label="u", start_north=-2.0 * radius,
            seconds=4.0 * radius / V, dt=0.05,
            window=(-radius, radius), window_name="column")

        pitch_v = vortex_viz.fig8_point(vortex)[0]
        pitch_u = vortex_viz.fig8_point(updraft)[0]
        assert pitch_v < pitch_u, f"seed {seed}: {pitch_v:.3f} !< {pitch_u:.3f}"
