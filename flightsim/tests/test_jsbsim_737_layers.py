"""The four comparison layers against JSBSim's 737.

Split from test_jsbsim_737.py, which establishes that the registry entry is the
one the generator recovered. This file asks the actual question: does flightsim
compute the same answers as an independent engine given the same coefficients?

Every tolerance here is DERIVED and its derivation is in the docstring beside
it. Where the two models genuinely differ, the difference is predicted from
JSBSim's own table constants and the prediction is asserted -- which is a
stronger statement than allowing a tolerance, because a real defect would have
to disguise itself as a known missing term to survive.

Design: docs/superpowers/specs/2026-08-20-jsbsim-737-verification-design.md
"""

import numpy as np
import pytest

from flightsim import jsbsim_ref
from flightsim.aircraft import REGISTRY

REF = jsbsim_ref.load()
CRUISE_COND = REF.condition["cruise"]
TRIM = REF.trim["longitudinal"]

# The two recovery conditions. One reference point cannot distinguish a solver
# that is right from one that is right in a single place -- PROJECT.md section 4
# records this project's phugoid error moving from 17.8% to 0.4% between two
# conditions with identical code -- so layers 1 and 2 run at both.
CASES = {
    "cruise": ("boeing737", jsbsim_ref.REFERENCE, "cruise"),
    "approach": ("boeing737_approach",
                 jsbsim_ref.REFERENCE.parent / "jsbsim_737_approach_reference.xml",
                 "approach"),
}
_LOADED = {name: jsbsim_ref.load(path) for name, (_, path, _c) in CASES.items()}


def case(name):
    """(reference, condition, trim, aircraft) for one recovery condition."""
    aircraft_name, _path, condition_name = CASES[name]
    ref = _LOADED[name]
    return (ref, ref.condition[condition_name], ref.trim["longitudinal"],
            REGISTRY[aircraft_name])


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _flightsim_coefficients(point, ac):
    import jax.numpy as jnp

    from flightsim import aero
    from flightsim.state import Controls

    controls = Controls(
        elevator=jnp.array(point.controls[0]),
        aileron=jnp.array(point.controls[1]),
        rudder=jnp.array(point.controls[2]),
        throttle=jnp.array(0.0),
    )
    return [
        float(v)
        for v in aero.coefficients(
            jnp.array(point.vel_body),
            jnp.array(point.rates),
            controls,
            ac,
            jnp.array(point.sound_speed),
        )
    ]


def _flightsim_trim(condition="cruise"):
    import jax.numpy as jnp

    from flightsim import trim as trim_mod

    _ref, cond, _trim, ac = case(condition)
    x, residual = trim_mod.trim(
        jnp.array(cond.airspeed), jnp.array(cond.matched_altitude), ac
    )
    return [float(v) for v in x], float(np.max(np.abs(np.asarray(residual))))


def _modes_from(A):
    """(wn, zeta) for each complex pair of a plant matrix, sorted low wn first."""
    out = []
    for lam in np.linalg.eigvals(A):
        if lam.imag > 1e-12:
            wn = abs(lam)
            out.append((wn, -lam.real / wn))
    return sorted(out)


# ---------------------------------------------------------------------------
# Layer 1 -- the aerodynamic build-up
# ---------------------------------------------------------------------------
# The 737.xml drag terms flightsim has no home for. Its CD0 is a single
# constant, frozen at the TRIM value of all three, so the predicted
# disagreement at any other swept point is exactly their departure from trim.
_CD0_ALPHA = ([-1.57, -0.26, 0.0, 0.26, 1.57], [1.5, 0.042, 0.021, 0.042, 1.5])
_CD_BETA = ([-1.57, -0.26, 0.0, 0.26, 1.57], [1.23, 0.05, 0.0, 0.05, 1.23])
_CD_DE = 0.059
# The AERORP sits 4.925 ft above the CG on a 12.31 ft chord, so a drag
# difference carries a pitching-moment difference in that ratio.
_Z_ARM_OVER_CHORD = 4.925 / 12.31


def _missing_drag(point, trim, ac):
    """The CD difference flightsim's model cannot represent.

    The sideslip term is no longer wholly missing -- flightsim now carries a
    quadratic CD_beta -- so what remains is the difference between JSBSim's
    linearly interpolated table and that quadratic. The two agree at the table's
    0.26 rad breakpoint and diverge below it, which is deliberate: see the
    CD_beta comment in aircraft.py.
    """
    return (
        (np.interp(point.alpha, *_CD0_ALPHA) - np.interp(trim.alpha, *_CD0_ALPHA))
        + (np.interp(point.beta, *_CD_BETA) - float(ac.CD_beta) * point.beta**2)
        + _CD_DE * (abs(point.controls[0]) - abs(trim.elevator))
    )


@pytest.mark.parametrize("condition", list(CASES))
@pytest.mark.parametrize(
    "name,index,tol",
    [("CL", 0, 1e-7), ("CY", 2, 1e-10), ("Cl", 3, 5e-5), ("Cn", 5, 5e-5)],
)
def test_layer1_coefficients_with_no_model_difference_agree(name, index, tol, condition):
    """The four coefficients both engines model the same way.

    These need no real tolerance beyond round-off. JSBSim's CL is a table, but
    it is LINEAR on the segment swept here, and CY, Cl and Cn are linear in
    every swept variable in both engines. Worst measured across the 33 sweep
    points: CL 5.5e-9, CY 1.6e-14, Cl 7.0e-6, Cn 1.7e-6 -- so the tolerances
    above sit one to three orders above the measurement and would still catch a
    sign error or a bad non-dimensionalisation.
    """
    ref, _cond, _trim, ac = case(condition)
    worst, where = 0.0, None
    for i, point in enumerate(ref.sweep):
        err = abs(_flightsim_coefficients(point, ac)[index] - point.coefficients[index])
        if err > worst:
            worst, where = err, i
    assert worst <= tol, (
        f"{condition}/{name}: worst |difference| {worst:.3e} > {tol:.1e} "
        f"at sweep point {where}"
    )


@pytest.mark.parametrize("condition", list(CASES))
def test_layer1_drag_difference_is_exactly_the_terms_flightsim_lacks(condition):
    """CD disagrees, and the disagreement is accounted for rather than allowed.

    flightsim has no CD0(alpha) variation, no CDbeta and no CDde. Subtracting
    their predicted departure from trim leaves 3.6e-4 worst case. At the
    sideslip points, where the raw difference is largest at 1.0e-2, the
    prediction accounts for it to 2.9e-10; what remains anywhere is the
    second-order induced-drag difference.
    """
    ref, _cond, trim, ac = case(condition)
    worst, where = 0.0, None
    for i, point in enumerate(ref.sweep):
        got = _flightsim_coefficients(point, ac)[1]
        unexplained = abs((point.coefficients[1] - got) - _missing_drag(point, trim, ac))
        if unexplained > worst:
            worst, where = unexplained, i
    assert worst <= 1e-3, (
        f"{condition}: CD difference beyond the known missing terms "
        f"{worst:.3e} at point {where}"
    )


@pytest.mark.parametrize("condition", list(CASES))
def test_layer1_pitching_moment_is_exact_at_the_reference_point(condition):
    """Cm is a linearisation about trim, so at trim it must be exact.

    Sweep point 2 IS the trim condition. Measured difference there: 5.9e-7.
    This is the test that would catch a wrong Cm0 or Cma intercept, which the
    quadratic bound below deliberately cannot.
    """
    ref, _cond, trim, ac = case(condition)
    at_trim = min(ref.sweep, key=lambda p: abs(p.alpha - trim.alpha) + abs(p.beta))
    difference = abs(at_trim.coefficients[4] - _flightsim_coefficients(at_trim, ac)[4])
    assert difference < 1e-6, f"Cm at the reference point is off by {difference:.3e}"


@pytest.mark.parametrize("condition", list(CASES))
def test_layer1_pitching_moment_difference_stays_within_its_two_mechanisms(condition):
    """Cm's disagreement is second-order, and it has exactly two sources.

    Measured, it vanishes at trim (5.9e-7) and grows symmetrically either side
    -- -1.66e-3 at alpha 0 against -1.78e-3 at alpha 4 deg -- which is the
    signature of a quadratic residual, not a wrong slope.

    Two mechanisms produce it, both of them force differences acting on the
    AERORP-to-CG offset:

      1. The drag terms flightsim has no home for (above), acting on the
         0.400-chord vertical arm.
      2. The LINEARISATION residual. JSBSim's Cm curves with alpha because the
         lift vector rotates into the body x axis as alpha changes and that
         rotated component acts on the same arm; to second order the curvature
         is (z_arm/c) * CLa, giving (z_arm/c) * CLa * dalpha^2.

    The bound is their SUM, which is an upper bound rather than a prediction:
    the two can partly cancel -- and do, which is why the measured values come
    in below it -- but neither can make the total exceed the sum. Asserting the
    sum is therefore honest about what is derived and what is not, where
    predicting the exact value would mean reimplementing JSBSim's moment
    build-up here and claiming more precision than is warranted.
    """
    ref, _cond, trim, ac = case(condition)
    CLa = float(ac.CLa)
    for i, point in enumerate(ref.sweep):
        measured = abs(point.coefficients[4] - _flightsim_coefficients(point, ac)[4])
        bound = _Z_ARM_OVER_CHORD * (
            abs(_missing_drag(point, trim, ac)) + CLa * (point.alpha - trim.alpha) ** 2
        ) + 1e-5
        assert measured <= bound, (
            f"Cm difference {measured:.3e} exceeds the {bound:.3e} its two "
            f"mechanisms allow, at sweep point {i}"
        )


# ---------------------------------------------------------------------------
# Layer 2 -- trim
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("condition", list(CASES))
def test_layer2_longitudinal_trim_matches(condition):
    """Both engines' own trim algorithms, at the same condition.

    Measured: alpha 1.980 deg against JSBSim's 1.965, elevator -0.05311 against
    -0.05192 rad, thrust +1.04%. The thrust difference is the linear-throttle
    approximation plus the 0.19% ram fit residual, both documented in
    _boeing_737's docstring; the tolerance is set just above it at 2%.
    """
    from flightsim.atmosphere import RHO0, density

    _ref, cond, expected, ac = case(condition)
    (alpha, elevator, throttle), residual = _flightsim_trim(condition)
    assert residual < 1e-8, f"flightsim's own trim did not converge: {residual:.2e}"
    assert abs(alpha - expected.alpha) < 5e-4, (
        f"{condition}: alpha {np.degrees(alpha):.5f} deg vs "
        f"{np.degrees(expected.alpha):.5f} deg"
    )
    assert abs(elevator - expected.elevator) < 2e-3

    rho = float(density(cond.matched_altitude))
    mach = cond.airspeed / cond.sound_speed
    thrust = (
        throttle
        * float(ac.max_thrust)
        * (rho / RHO0) ** float(ac.thrust_lapse)
        * (1.0 + float(ac.mach_ram) * mach**2)
    )
    assert abs(thrust - expected.thrust) / expected.thrust < 0.02


def test_layer2_turn_trim_is_recorded_but_not_yet_comparable():
    """JSBSim's 30 deg banked trim is in the reference; flightsim has no banked trim.

    trim.trim solves the WINGS-LEVEL problem only -- its unknowns are
    [alpha, elevator, throttle], with no bank and no aileron or rudder. So the
    turn case cannot be compared today. It is recorded rather than dropped: the
    reference already holds JSBSim's converged 30 deg solution, so the
    comparison is one banked-trim solver away.

    This test asserts the gap rather than skipping silently, so that adding a
    banked trim makes it fail and forces the comparison to be written.
    """
    from flightsim import trim as trim_mod

    turn = REF.trim["turn"]
    assert turn.bank == pytest.approx(np.radians(30.0), abs=1e-6)
    assert turn.alpha > TRIM.alpha, "a banked trim must carry more alpha than level"
    assert not hasattr(trim_mod, "trim_banked"), (
        "a banked trim solver now exists -- write the turn comparison against "
        "REF.trim['turn'] and delete this assertion"
    )


# ---------------------------------------------------------------------------
# Layer 3 -- the linearisation
# ---------------------------------------------------------------------------
def test_layer3_longitudinal_modes_match():
    """Phugoid and short period against JSBSim's exported linearisation.

    Compared on EIGENVALUES, which are invariant under the similarity transform
    between JSBSim's [vt, alpha, theta, q] basis and this project's
    [u, w, q, theta] one. So no basis conversion is needed and none can be got
    wrong -- the transform was in the plan and turned out to be unnecessary.

    Measured: short period wn 1.76847 against 1.76918 (0.04%), zeta 0.39301
    against 0.39294 (0.02%). Phugoid wn within 3.3%, zeta within 1.1%; it is
    the looser of the two because it is a slow drag-and-thrust energy exchange,
    and drag and thrust are exactly where the two models differ.
    """
    from flightsim import validation

    (alpha, elevator, throttle), _ = _flightsim_trim()
    got = validation.longitudinal_modes(
        REGISTRY["boeing737"], alpha, elevator, throttle,
        CRUISE_COND.airspeed, CRUISE_COND.matched_altitude,
    )
    want = _modes_from(REF.linearization.longitudinal)
    for (wn, zeta), (wn_ref, zeta_ref), name, tol in zip(
        got, want, ("phugoid", "short period"), (5e-2, 1e-3)
    ):
        assert abs(wn - wn_ref) / wn_ref < tol, f"{name} wn {wn:.6f} vs {wn_ref:.6f}"
        assert abs(zeta - zeta_ref) / zeta_ref < tol, (
            f"{name} zeta {zeta:.6f} vs {zeta_ref:.6f}"
        )


def test_layer3_lateral_modes_match_once_the_yaw_damper_is_accounted_for():
    """JSBSim's linearisation is CLOSED-LOOP: its FCS is inside it.

    This is the most misleading thing about comparing against do_linearization,
    and nothing in its output says so. 737.xml's yaw damper feeds yaw rate to
    the rudder with unit gain above M 0.11, and an aerosurface_scale gears that
    by 0.35 rad. Against the BARE airframe the lateral comparison looks
    catastrophic -- Dutch roll zeta 0.101 against 0.344, spiral 127.6 s against
    16.7 s, a 664% disagreement -- and a reader would reasonably conclude this
    project's lateral dynamics were broken.

    They are not. The damper adds, non-dimensionally,

        dCnr = Cndr * 0.35 * 2V/b = -1.147   (against a bare Cnr of -0.350)
        dClr = Cldr * 0.35 * 2V/b = +0.057

    and folding those in gives zeta 0.338 against 0.344 and spiral 16.61 s
    against 16.69 s. The pitch and roll channels have no such feedback -- they
    are summers and gearing only -- which is why the longitudinal comparison
    needs no correction at all and lands at 0.04%.
    """
    import jax.numpy as jnp

    from flightsim import validation

    ac = REGISTRY["boeing737"]
    gain = 0.35 * 2.0 * CRUISE_COND.airspeed / float(ac.b)
    closed_loop = ac._replace(
        Cnr=jnp.array(float(ac.Cnr) + float(ac.Cndr) * gain),
        Clr=jnp.array(float(ac.Clr) + float(ac.Cldr) * gain),
        CYr=jnp.array(float(ac.CYr) + float(ac.CYdr) * gain),
    )
    (alpha, elevator, throttle), _ = _flightsim_trim()
    (wn, zeta), roll_tc, spiral_tc = validation.lateral_modes(
        closed_loop, alpha, elevator, throttle,
        CRUISE_COND.airspeed, CRUISE_COND.matched_altitude,
    )

    eigenvalues = np.linalg.eigvals(REF.linearization.lateral)
    eigenvalues = eigenvalues[np.abs(eigenvalues) > 1e-8]  # drop the psi integrator
    pair = eigenvalues[np.abs(eigenvalues.imag) > 1e-9][0]
    reals = np.sort(eigenvalues[np.abs(eigenvalues.imag) <= 1e-9].real)
    wn_ref, zeta_ref = abs(pair), -pair.real / abs(pair)
    roll_ref, spiral_ref = -1.0 / reals[0], -1.0 / reals[-1]

    assert abs(wn - wn_ref) / wn_ref < 5e-2, f"dutch roll wn {wn:.5f} vs {wn_ref:.5f}"
    assert abs(zeta - zeta_ref) / zeta_ref < 5e-2, (
        f"dutch roll zeta {zeta:.5f} vs {zeta_ref:.5f}"
    )
    assert abs(roll_tc - roll_ref) / roll_ref < 5e-2, (
        f"roll TC {roll_tc:.5f} s vs {roll_ref:.5f} s"
    )
    assert abs(spiral_tc - spiral_ref) / spiral_ref < 5e-2, (
        f"spiral TC {spiral_tc:.4f} s vs {spiral_ref:.4f} s"
    )


def test_layer3_bare_airframe_would_fail_without_the_damper_correction():
    """The correction above is load-bearing, and this proves it.

    Without it the spiral disagrees by a factor of 7.6. If a future change made
    the correction unnecessary -- JSBSim linearising open-loop, say -- this test
    fails and the one above becomes wrong in a way nobody would otherwise see.
    """
    from flightsim import validation

    (alpha, elevator, throttle), _ = _flightsim_trim()
    _, _, spiral_tc = validation.lateral_modes(
        REGISTRY["boeing737"], alpha, elevator, throttle,
        CRUISE_COND.airspeed, CRUISE_COND.matched_altitude,
    )
    eigenvalues = np.linalg.eigvals(REF.linearization.lateral)
    eigenvalues = eigenvalues[np.abs(eigenvalues) > 1e-8]
    reals = np.sort(eigenvalues[np.abs(eigenvalues.imag) <= 1e-9].real)
    assert spiral_tc / (-1.0 / reals[-1]) > 5.0


# ---------------------------------------------------------------------------
# Layer 4 -- trajectory
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "case,vel_tol,rate_tol",
    [("elevator_doublet", 0.5, 0.01), ("rudder_kick", 2.0, 0.02)],
)
def test_layer4_trajectory_tracks(case, vel_tol, rate_tol):
    """Integrate 20 s on JSBSim's own prescribed surface history.

    The surfaces are the ACHIEVED deflections JSBSim recorded, not commands, so
    its FCS cannot contribute to any difference. Throttle is set to reproduce
    JSBSim's recorded thrust at the initial condition and then held, as JSBSim
    holds it.

    The two cases differ by 3.6x, and they differ because they excite different
    physics rather than because one is worse. Measured per component:

        elevator doublet   u 0.296   v 0.012   w 0.410     max |beta| 0.003 deg
        rudder kick        u 1.462   v 1.234   w 0.522     max |beta| 2.909 deg

    The doublet is purely longitudinal -- it produces essentially no sideslip,
    so every lateral model difference is inert and the residual is a transient
    in w. The kick excites sideslip and the Dutch roll, which is where the model
    differences live: its v divergence is transient (1.234 peak, 0.105 by t=20)
    and is the Dutch roll frequency difference, while its u divergence is
    SECULAR, still growing at t=20, and is the sideslip drag flightsim
    under-models.

    NOT the Earth-rotation floor, which an earlier version of this docstring
    claimed. Running the same JSBSim case at latitude 0 against 47 deg moves it
    by u 0.403, v 0.012, w 0.067 -- so the floor is in u, and the doublet's
    0.410 is in w. Those are different quantities that happen to be numerically
    close, and matching them was a mistake. Compared per component, the
    doublet's u divergence of 0.296 is BELOW the 0.403 floor, and its w
    divergence of 0.410 is six times the 0.067 floor in that component, so the w
    residual is real and needs its own explanation.

    That explanation is the alphadot fold. Cmq carries Cmq + Cmadot, which is
    exact only when alphadot = q; during the doublet the recorded histories give
    max |alphadot - q| = 0.0122 rad/s, worth up to |dCm| = 0.0015, against
    0.0186 for one degree of alpha. flightsim applies no alphadot term in still
    air by design -- the term added here is wind-driven only -- so that
    difference is unmodelled.

    An earlier 0.25 s sampling of the reference gave 5.43 m/s here purely
    because the replay flew a stale rudder between samples while the yaw damper
    moved continuously; the reference is sampled at 0.05 s for that reason.
    """
    import jax
    import jax.numpy as jnp

    from flightsim import integrate
    from flightsim.atmosphere import RHO0, density, speed_of_sound
    from flightsim.state import Controls, State, euler_to_quat

    ac = REGISTRY["boeing737"]
    samples = REF.trajectory[case]
    first = samples[0]
    # The same density-matching shift the rest of the comparison uses.
    altitude = first.altitude + (CRUISE_COND.matched_altitude - CRUISE_COND.altitude)
    rho = float(density(altitude))
    mach = float(np.linalg.norm(first.vel_body)) / float(speed_of_sound(altitude))
    throttle = first.thrust / (
        float(ac.max_thrust)
        * (rho / RHO0) ** float(ac.thrust_lapse)
        * (1.0 + float(ac.mach_ram) * mach**2)
    )

    sim = integrate.init_sim(
        State(
            pos_ned=jnp.array([0.0, 0.0, -altitude]),
            vel_body=jnp.array(first.vel_body),
            quat=euler_to_quat(*(jnp.array(v) for v in first.euler)),
            omega=jnp.array(first.omega),
        ),
        jax.random.PRNGKey(0),
    )
    worst_vel = worst_rate = 0.0
    for previous, current in zip(samples, samples[1:]):
        controls = Controls(
            elevator=jnp.array(previous.controls[0]),
            aileron=jnp.array(previous.controls[1]),
            rudder=jnp.array(previous.controls[2]),
            throttle=jnp.array(throttle),
        )
        sim = integrate.step(sim, controls, jnp.array(current.t - previous.t), ac)
        worst_vel = max(worst_vel, float(
            np.abs(np.asarray(sim.state.vel_body) - current.vel_body).max()))
        worst_rate = max(worst_rate, float(
            np.abs(np.asarray(sim.state.omega) - current.omega).max()))

    coriolis = REF.diagnostics["coriolis_velocity_m_s"]
    assert worst_vel <= vel_tol, (
        f"{case}: worst velocity divergence {worst_vel:.4f} m/s > {vel_tol} "
        f"(the Earth-rotation floor alone is {coriolis:.4f} m/s)"
    )
    assert worst_rate <= rate_tol, (
        f"{case}: worst angular-rate divergence {worst_rate:.6f} rad/s > {rate_tol}"
    )
