"""The run checks, and whether each of them can fail.

PROJECT.md section 3's rule -- a check which can only pass shows nothing -- is
what shapes this file. Every check here gets a NEGATIVE CONTROL: an input that
must make it fail. Four of the checks are tripwires that have never fired on any
field the project holds, and for those the negative control is the only evidence
they work at all.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import atisim  # noqa: F401  -- enables x64 before any array is made
from atisim import checks, integrate, trim, viz, wind
from atisim.aircraft import CRUISE, REGISTRY
from atisim.manual import Mode
from atisim.units import FT2M, KT2MS
from atisim.wind import PARKS_CASES

AIRCRAFT = "boeing747"


@pytest.fixture(scope="module")
def parks_run():
    """A Parks Hannibal traverse: trimmed lead-in, both cores, fixed controls.

    THE 40 CORE RADII ARE NOT COPIED FROM scripts/vortex.py, THEY ARE FORCED.
    This fixture was first written with a 14 r0 lead-in, on the reading that
    PROJECT.md section 9 session 3's "below ~12 the far field contaminates the
    first core" made 14 safe. `trimmed_start` failed it immediately at
    n_z[0] = 1.0968 against cos(theta0) = 0.9967 -- 0.100 g out of equilibrium
    before the run begins, which is half the 0.20 g that section records for the
    -6 r0 lead-in that understated first-core d(theta) by 15%.

    The arithmetic agrees: at 14 r0 the 1/r far field is still V0*r0/r = 1.85 m/s,
    which is 0.45 deg of alpha and about 0.08 g of lift. So 12 r0 is where the
    far field stops contaminating the CORE, not where the aircraft is trimmed.

    Recorded here because it is the check's first use catching the exact defect
    it was written for, in this file's own fixture.
    """
    ac = REGISTRY[AIRCRAFT]
    V = CRUISE[AIRCRAFT]["airspeed"]
    H = CRUISE[AIRCRAFT]["altitude"]
    case = PARKS_CASES["hannibal"]
    r0, v0, spacing = case["r0"], case["v0"], case["spacing"]
    array = wind.VortexArray(
        north=jnp.array([0.0, spacing]), down=jnp.array([-H, -H]),
        r0=jnp.array(r0), v0=jnp.array(v0),
    )
    field = lambda p: wind.vortex_wind(p, array)  # noqa: E731
    model = wind.field_model(field)

    x, _ = trim.trim(jnp.array(V), jnp.array(H), ac)
    controls = trim.trimmed_controls(x[1], x[2])
    state = trim.trimmed_state(jnp.array(float(x[0])), jnp.array(V), jnp.array(H))
    lead = 40.0 * r0
    state = state._replace(pos_ned=jnp.array([-lead, 0.0, -H]))

    dt = 0.01
    n = int(round(((spacing + lead + 3.0 * r0) / V) / dt))
    sim = integrate.init_sim(state, jax.random.PRNGKey(0))
    recorder = viz.Recorder()
    for i in range(n):
        sim = integrate.step(sim, controls, jnp.array(dt), ac, wind_model=model)
        recorder.append((i + 1) * dt, sim, controls, Mode.MANUAL)
    traj = recorder.trajectory()
    return dict(traj=traj, ac=ac, field=field, model=model, controls=controls,
                dt=dt, r0=r0, v0=v0, spacing=spacing, V=V, H=H)


# --- C1 quaternion norm ------------------------------------------------------


def test_the_quaternion_norm_holds(parks_run):
    c = checks.quaternion_norm(parks_run["traj"])
    assert c.passed, c
    assert c.value < 1e-12
    assert c.kind == "tripwire"  # it has never fired; say so rather than tick it


def test_the_quaternion_norm_check_can_fail(parks_run):
    """The negative control. Without it this check could `return True`."""
    traj = parks_run["traj"]
    bad = traj._replace(quat=traj.quat * 1.0001)
    assert not checks.quaternion_norm(bad).passed


# --- C2 field divergence -----------------------------------------------------


def test_every_field_in_the_project_is_solenoidal(parks_run):
    c = checks.field_divergence(parks_run["field"], parks_run["traj"].pos_ned)
    assert c.passed, c
    assert c.value < 1e-12


def test_the_divergence_check_can_fail(parks_run):
    """A radially expanding field. Not solenoidal, and the check must say so.

    This is the whole value of C2: it fires on a field NOBODY HAS WRITTEN YET.
    Every field the project currently holds passes it at round-off.
    """
    def diverging(pos_ned):
        return 0.1 * pos_ned

    c = checks.field_divergence(diverging, parks_run["traj"].pos_ned)
    assert not c.passed
    assert c.value == pytest.approx(0.3, rel=1e-6)  # trace of 0.1 * I


# --- C3/C4 energy closure ----------------------------------------------------


def test_the_energy_budget_closes(parks_run):
    c = checks.energy_closure(
        parks_run["traj"], parks_run["ac"], parks_run["field"]
    )
    assert c.passed, c
    assert c.value < 2e-3


def test_the_energy_residual_is_largest_at_the_core_boundaries(parks_run):
    """C4. The Rankine gradient discontinuity, located from an energy diagnostic.

    ASSUMPTIONS.md E2 records that the one-sided derivatives at r = r0 differ by
    2*V0/r0 with opposite signs. That kink shows up as a spike in the per-step
    closure residual, and this asserts the spike is really at the boundary rather
    than somewhere the eye happened to land.
    """
    run = parks_run
    profile = checks.energy_residual_profile(run["traj"], run["ac"], run["field"])
    north, per_step = profile.north, profile.per_step
    median = float(np.median(per_step))

    edges = [-run["r0"], run["r0"],
             run["spacing"] - run["r0"], run["spacing"] + run["r0"]]
    for edge in edges:
        k = int(np.argmin(np.abs(north - edge)))
        assert per_step[k] > 10.0 * median, (
            f"core edge at north={edge:.1f} m is only "
            f"{per_step[k] / median:.1f}x the run median"
        )


# --- C5 trimmed start --------------------------------------------------------


def _short_lead_in_run(run, lead_radii, n_steps=400):
    """Fly the same field from `lead_radii` core radii out."""
    ac, V, H, r0 = run["ac"], run["V"], run["H"], run["r0"]
    x, _ = trim.trim(jnp.array(V), jnp.array(H), ac)
    controls = trim.trimmed_controls(x[1], x[2])
    state = trim.trimmed_state(jnp.array(float(x[0])), jnp.array(V), jnp.array(H))
    state = state._replace(pos_ned=jnp.array([-lead_radii * r0, 0.0, -H]))
    sim = integrate.init_sim(state, jax.random.PRNGKey(0))
    recorder = viz.Recorder()
    for i in range(n_steps):
        sim = integrate.step(sim, controls, jnp.array(0.01), ac, wind_model=run["model"])
        recorder.append((i + 1) * 0.01, sim, controls, Mode.MANUAL)
    return recorder.trajectory(), controls


def test_the_start_offset_is_reported_not_gated_in_a_wind_field(parks_run):
    """A 1/r field has no trimmed start, so there is no honest threshold.

    At scripts/vortex.py's own 40 core radii the far field is still 0.648 m/s and
    the run begins 0.0384 g off cos(theta0). Gating on cos(theta0) would fail the
    project's canonical run; loosening the tolerance until it passed would be
    choosing a number to make a check succeed.
    """
    run = parks_run
    c = checks.trimmed_start(run["traj"], run["ac"], run["controls"], run["field"])
    assert c.kind == "report"
    assert c.passed is None
    assert c.value < 0.10, f"40 r0 should start well inside 10% of the excursion: {c}"


def test_a_shorter_lead_in_starts_further_out_of_equilibrium(parks_run):
    """The check must ORDER the two lead-ins, which is what makes it a diagnostic.

    Section 9 session 3's defect was a -6 r0 lead-in. This asserts the direction
    of the effect rather than an invented threshold: less lead-in, more offset.
    """
    run = parks_run
    far = checks.trimmed_start(run["traj"], run["ac"], run["controls"], run["field"])
    traj_short, controls_short = _short_lead_in_run(run, 6.0)
    near = checks.trimmed_start(traj_short, run["ac"], controls_short, run["field"])
    assert near.value > 3.0 * far.value, (
        f"6 r0 ({near.value:.3f}) should be far worse than 40 r0 ({far.value:.3f})"
    )


def test_the_trimmed_start_check_gates_in_still_air(parks_run):
    """In still air the right answer exists and is exactly cos(theta0).

    PROJECT.md section 4 records n_z[0] = 0.9967 for the manoeuvring case as
    "the trimmed value, i.e. the lead-in worked". That IS a gate.
    """
    run = parks_run
    ac, V, H = run["ac"], run["V"], run["H"]
    x, _ = trim.trim(jnp.array(V), jnp.array(H), ac)
    controls = trim.trimmed_controls(x[1], x[2])
    state = trim.trimmed_state(jnp.array(float(x[0])), jnp.array(V), jnp.array(H))
    sim = integrate.init_sim(state, jax.random.PRNGKey(0))
    recorder = viz.Recorder()
    for i in range(50):
        sim = integrate.step(sim, controls, jnp.array(0.01), ac, wind_model=wind.zero_wind)
        recorder.append((i + 1) * 0.01, sim, controls, Mode.MANUAL)

    c = checks.trimmed_start(recorder.trajectory(), ac, controls, None)
    assert c.kind == "gate"
    assert c.passed, c
    assert c.value < 1e-3


# --- C8 alpha band -----------------------------------------------------------


def test_the_alpha_band_is_reported_for_the_window_and_the_whole_run(parks_run):
    """Both numbers, because they can land in different bands.

    Measured on the full scripts/vortex.py run: 8.27 deg inside the first core
    (green) against 11.39 deg over the whole run (amber). A whole-run-only gate
    would flag a run whose reported result is entirely valid.
    """
    run = parks_run
    north = run["traj"].pos_ned[:, 0]
    window = (north >= -run["r0"]) & (north <= run["r0"])
    c = checks.alpha_band(run["traj"], run["field"], window)

    assert c.window_deg <= c.whole_run_deg
    assert c.window_band in ("linear", "MARGINAL", "INVALID")
    assert c.value == pytest.approx(c.window_deg)


def test_the_alpha_band_reports_invalid_past_the_linear_range(parks_run):
    """Negative control: force alpha past 12 deg and the band must change."""
    run = parks_run
    traj = run["traj"]
    # Add 15 deg of downward relative flow: |alpha| leaves the declared range.
    vel = np.array(traj.vel_body, copy=True)
    vel[:, 2] += np.tan(np.radians(15.0)) * vel[:, 0]
    bad = traj._replace(vel_body=vel)
    window = np.ones(len(traj.t), dtype=bool)
    band = checks.alpha_band(bad, run["field"], window)
    assert band.window_band == "INVALID"
    assert not band.as_check().passed


def test_the_amber_band_is_flagged_but_does_not_condemn_the_run(parks_run):
    """MARGINAL must survive the gate, because a published result sits there.

    The manoeuvring Fig. 8 point is flown at |alpha| 10.31 deg -- amber -- and
    PROJECT.md section 4 reports it. A gate collapsing three bands to
    `band == "linear"` would fail the project's own table, which is how this was
    found: `scripts/vortex.py --artifacts` printed "CHECKS FAILED: alpha band"
    for the manoeuvre on the first run.
    """
    run = parks_run
    traj = run["traj"]
    # 10.5 deg: inside the amber band, outside the linear one. Set in STILL AIR,
    # because alpha here is AIR-relative -- leaving the vortex gust in would add
    # several degrees on top and land in the red band instead.
    vel = np.array(traj.vel_body, copy=True)
    vel[:, 2] = np.tan(np.radians(10.5)) * vel[:, 0]
    still = traj._replace(vel_body=vel, wind_ned=np.zeros_like(traj.wind_ned))
    amber = checks.alpha_band(still, run["field"], np.ones(len(traj.t), dtype=bool))
    assert amber.window_band == "MARGINAL"
    assert amber.as_check().passed, "amber must not condemn the run"


# --- C9 lateral symmetry -----------------------------------------------------


def test_a_symmetric_field_produces_no_lateral_response(parks_run):
    """The Parks vortex has NO east variation, so beta and p must be exactly zero.

    ASSUMPTIONS.md E2: every strip on the span sees the same vertical gust, which
    is why flying the strip path moves this encounter by 0.000000 m. The same
    fact makes any lateral response a defect.
    """
    c = checks.lateral_symmetry(parks_run["traj"])
    assert c.passed, c
    assert c.value == 0.0  # exactly, not approximately


def test_the_lateral_symmetry_check_can_fail(parks_run):
    traj = parks_run["traj"]
    omega = np.array(traj.omega, copy=True)
    omega[100, 0] = 1e-6  # a roll rate a symmetric field cannot produce
    assert not checks.lateral_symmetry(traj._replace(omega=omega)).passed


# --- C11 recorded vs analytic wind -------------------------------------------


def test_the_recorded_wind_is_the_analytic_field_one_step_earlier(parks_run):
    """SimState caches the wind the PREVIOUS step applied, and a panel that
    overlays the two unshifted shows a lag that is correct behaviour.

    Measured on the full run: 0.3315 m/s unshifted, 1.13% of the peak gust, and
    7.1e-15 m/s after a one-step shift.
    """
    run = parks_run
    m = checks.recorded_wind_matches_field(run["traj"], run["field"])
    assert m.as_check().passed, m
    assert m.shifted < 1e-12
    assert m.unshifted > 100.0 * max(m.shifted, 1e-15)  # and the lag is real


def test_the_recorded_wind_check_can_fail(parks_run):
    """Negative control: a run whose recorded wind is not the field it flew."""
    run = parks_run
    traj = run["traj"]
    bad = traj._replace(wind_ned=traj.wind_ned + 1.0)
    assert not checks.recorded_wind_matches_field(bad, run["field"]).as_check().passed


# --- the collected report ----------------------------------------------------


def test_run_checks_returns_every_check_with_a_kind(parks_run):
    run = parks_run
    north = run["traj"].pos_ned[:, 0]
    window = (north >= -run["r0"]) & (north <= run["r0"])
    report = checks.run_checks(
        run["traj"], run["ac"], run["controls"], run["field"], window
    )
    assert len(report) >= 6
    for c in report:
        assert c.kind in ("gate", "tripwire", "report")
        assert c.name
        # A `report` check has no threshold and therefore no verdict.
        if c.kind == "report":
            assert c.passed is None
        else:
            assert c.passed is not None


def test_the_report_is_json_serialisable(parks_run):
    """The UI reads checks from a file, never recomputes them (design section 3.1)."""
    import json

    run = parks_run
    north = run["traj"].pos_ned[:, 0]
    window = (north >= -run["r0"]) & (north <= run["r0"])
    report = checks.run_checks(
        run["traj"], run["ac"], run["controls"], run["field"], window
    )
    text = json.dumps([c.as_dict() for c in report])
    assert json.loads(text)[0]["name"]


# ---------------------------------------------------------------------------
# C10 -- the recovery band
# ---------------------------------------------------------------------------
def _level_run(altitude_m, airspeed_ms, n=40):
    """A synthetic straight-and-level run at one altitude and speed.

    Built rather than flown because the band check reads only position and
    air-relative speed, and a real rollout would make the test depend on trim
    converging at conditions deliberately chosen to be far outside the model's
    range -- which is the thing under test, not a precondition of it.
    """
    quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
    return viz.Trajectory(
        t=np.linspace(0.0, 1.0, n),
        pos_ned=np.stack([np.zeros(n), np.zeros(n),
                          np.full(n, -altitude_m)], axis=1),
        vel_body=np.stack([np.full(n, airspeed_ms), np.zeros(n), np.zeros(n)], axis=1),
        quat=quat,
        omega=np.zeros((n, 3)),
        controls=np.zeros((n, 4)),
        mode=np.zeros(n, dtype=int),
        wind_ned=np.zeros((n, 3)),
        omega_gust=np.zeros((n, 3)),
    )


def test_an_aircraft_flown_at_its_recovery_point_is_inside_its_band():
    ac = REGISTRY["boeing737"]
    c = checks.recovery_band(
        _level_run(CRUISE["boeing737"]["altitude"], CRUISE["boeing737"]["airspeed"]), ac
    )
    assert c.kind == "gate"
    assert c.passed is True
    assert c.value == 0.0


def test_the_recovery_band_condemns_a_run_outside_it():
    """Negative control: Limitation 1's own example, 5,000 ft and 200 kt.

    The 737 docstring says flying it there "produces numbers that are wrong
    without anything failing, warning or logging". This is the thing that
    logs.
    """
    ac = REGISTRY["boeing737"]
    c = checks.recovery_band(_level_run(5000.0 * FT2M, 200.0 * KT2MS), ac)
    assert c.kind == "gate"
    assert c.passed is False
    assert c.value > 1.0


def test_an_aircraft_with_no_declared_band_is_reported_not_passed():
    """A green tick for an unchecked thing is the failure this module exists to
    avoid -- the same reasoning as the tripwire kind."""
    ac = REGISTRY["boeing747"]
    c = checks.recovery_band(
        _level_run(CRUISE["boeing747"]["altitude"], CRUISE["boeing747"]["airspeed"]), ac
    )
    assert c.kind == "report"
    assert c.passed is None
