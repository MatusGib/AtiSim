"""The four comparison layers against JSBSim's 737.

Split from test_jsbsim_737.py, which establishes that the registry entry is the
one the generator recovered. This file asks the actual question: does atisim
compute the same answers as an independent engine given the same coefficients?

Every tolerance here carries its reasoning in the docstring beside it, and says
which kind it is. Layer 1's CD and Cm bounds are PREDICTIONS, computed from
737.xml's own table constants -- a stronger statement than an allowance, because
a real defect would have to disguise itself as a known missing term to survive.
Layer 4's backstops are ALLOWANCES and are labelled as such rather than dressed
up; the predictive statements about layer 4 live in the two tests that separate
the replay artifact from the model difference.

Design: docs/superpowers/specs/2026-08-20-jsbsim-737-verification-design.md
"""

import numpy as np
import pytest

from atisim import jsbsim_ref
from atisim.aircraft import REGISTRY

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
def _atisim_coefficients(point, ac):
    import jax.numpy as jnp

    from atisim import aero
    from atisim.state import Controls

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


def _atisim_trim(condition="cruise"):
    import jax.numpy as jnp

    from atisim import trim as trim_mod

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


def _replay(ac, ref, cond, manoeuvre, decimate=1):
    """Fly JSBSim's recorded surface history and return the worst divergence.

    Returns (per-component worst |dv| as a (3,) array, worst |domega|).

    `decimate` thins the reference before replaying it. The consumer holds each
    control sample until the next one, so decimating coarsens the zero-order
    hold and nothing else -- which is what makes the sampling artifact separable
    from the model difference. See
    test_layer4_what_survives_the_hold_is_two_residuals_and_no_more.

    *** THE YAW DAMPER IS SWITCHED OFF HERE, AND THAT KEEPS LAYER 4 HONEST. ***
    Layer 4's whole design is that SURFACE POSITIONS ARE PRESCRIBED TO BOTH
    ENGINES: the generator pre-compensates JSBSim's rudder command each step by
    the yaw-damper output so that `fcs/rudder-pos-rad` follows the commanded
    history, and asserts it hits target. JSBSim's damper is therefore NEUTRALISED
    in this reference by construction.

    Since session 23 atisim's 737 carries a damper of its own in the plant. Left
    on, it would ADD `0.35 r` on top of the prescribed rudder -- so atisim would
    fly a surface JSBSim did not, and the comparison would no longer be
    like-for-like. Measured, that is worth 1.83 m/s of spurious `v` divergence on
    the approach rudder kick against a 0.013 m/s Earth-rotation floor, i.e. it
    would swamp the residual this layer exists to measure.

    Zeroing the gain restores the prescribed-surface premise on both sides. It
    does NOT weaken the test: layer 3 is where the damper is compared, and it now
    matches JSBSim's closed-loop linearisation directly.
    """
    import jax
    import jax.numpy as jnp

    from atisim import integrate
    from atisim.atmosphere import RHO0, density, speed_of_sound
    from atisim.state import Controls, State, euler_to_quat

    ac = ac._replace(yaw_damper_gain=jnp.array(0.0))

    samples = ref.trajectory[manoeuvre][::decimate]
    first = samples[0]
    # The same density-matching shift the rest of the comparison uses.
    altitude = first.altitude + (cond.matched_altitude - cond.altitude)
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
    worst_vel, worst_rate = np.zeros(3), 0.0
    for previous, current in zip(samples, samples[1:]):
        controls = Controls(
            elevator=jnp.array(previous.controls[0]),
            aileron=jnp.array(previous.controls[1]),
            rudder=jnp.array(previous.controls[2]),
            throttle=jnp.array(throttle),
        )
        sim = integrate.step(sim, controls, jnp.array(current.t - previous.t), ac)
        worst_vel = np.maximum(worst_vel, np.abs(
            np.asarray(sim.state.vel_body) - current.vel_body))
        worst_rate = max(worst_rate, float(
            np.abs(np.asarray(sim.state.omega) - current.omega).max()))
    return worst_vel, worst_rate


# ---------------------------------------------------------------------------
# Layer 0 -- the inputs, checked against the engine rather than against
#            another copy of the same decision
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("condition", list(CASES))
def test_inertia_cross_product_sign_matches_the_engines_own_coupling(condition):
    """The Ixz SIGN, established by the engine's behaviour instead of assumed.

    `test_737_mass_and_inertia_match_the_engine` compares atisim's tensor to
    the reference XML's -- but both descend from one line of
    gen_jsbsim_reference.py, so it catches a transcription slip and cannot catch
    a wrong convention. Nothing else in the comparison could either: the sign
    moves the lateral modes by only 1.6-3.2%, well inside layer 3's tolerance.

    JSBSim's own linearisation settles it, because 737.xml defines neither Cnp
    nor CYp. With no yaw moment from roll rate and no side force from it either
    -- so the CG and AERORP references agree on L_p -- the engine's yaw-rate
    response to roll rate is PURE INERTIA COUPLING:

        d(rdot)/dp = -Jxz * L_p / (Ixx*Izz - Jxz^2)

    where Jxz is the off-diagonal of the tensor as atisim actually flies it, so
    this reads the convention off the matrix rather than off a spelling of it.
    The p column is also immune to the yaw damper, which feeds r, not p.

    Measured: JSBSim's A[rdot, p] is +1.180789e-02 at cruise; the shipped tensor
    predicted -1.180789e-02, right in magnitude to seven digits and wrong in
    sign. A[pdot, p] agrees at -1.227332e+00 either way, which is what localises
    the disagreement to the sign and not to L_p or the determinant.
    """
    ref, cond, _trim, ac = case(condition)
    # The identity holds only because both of these are absent from 737.xml.
    assert abs(ref.absent["Cnp"]) < 1e-5, "a real Cnp would break the identity"
    assert abs(ref.absent["CYp"]) < 1e-5, "a real CYp would move L_p to the CG"

    P, R = 6, 8  # [vt, alpha, theta, q, beta, phi, p, psi, r, lat, lon, h]
    inertia = np.asarray(ac.inertia)
    Ixx, Izz, Jxz = inertia[0, 0], inertia[2, 2], inertia[0, 2]
    qbar = 0.5 * cond.density * cond.airspeed**2
    Lp = (float(ac.Clp) * qbar * float(ac.S) * float(ac.b)
          * (float(ac.b) / (2.0 * cond.airspeed)))

    predicted = -Jxz * Lp / (Ixx * Izz - Jxz**2)
    assert predicted == pytest.approx(ref.linearization.A[R][P], rel=1e-6), (
        f"{condition}: atisim's inertia predicts d(rdot)/dp = {predicted:.6e}, "
        f"JSBSim's linearisation says {ref.linearization.A[R][P]:.6e}"
    )


# ---------------------------------------------------------------------------
# Layer 1 -- the aerodynamic build-up
# ---------------------------------------------------------------------------
# The 737.xml drag terms atisim has no home for. Its CD0 is a single
# constant, frozen at the TRIM value of all three, so the predicted
# disagreement at any other swept point is exactly their departure from trim.
_CD0_ALPHA = ([-1.57, -0.26, 0.0, 0.26, 1.57], [1.5, 0.042, 0.021, 0.042, 1.5])
_CD_BETA = ([-1.57, -0.26, 0.0, 0.26, 1.57], [1.23, 0.05, 0.0, 0.05, 1.23])
_CD_DE = 0.059
# The AERORP sits 4.925 ft above the CG on a 12.31 ft chord, so a drag
# difference carries a pitching-moment difference in that ratio.
_Z_ARM_OVER_CHORD = 4.925 / 12.31


def _missing_drag(point, trim, ac):
    """The CD difference atisim's model cannot represent.

    The sideslip term is no longer wholly missing -- atisim now carries a
    quadratic CD_beta -- so what remains is the difference between JSBSim's
    linearly interpolated table and that quadratic. The two agree at the table's
    0.26 rad breakpoint and diverge below it, which is deliberate: see the
    CD_beta comment in aircraft.py.
    """
    return (
        # The alpha channel is now MOSTLY covered: atisim carries a linear
        # CD_alpha, so what is left is the table's departure from that line.
        (np.interp(point.alpha, *_CD0_ALPHA) - np.interp(trim.alpha, *_CD0_ALPHA))
        - float(ac.CD_alpha) * (point.alpha - trim.alpha)
        + (np.interp(point.beta, *_CD_BETA) - float(ac.CD_beta) * point.beta**2)
        + _CD_DE * (abs(point.controls[0]) - abs(trim.elevator))
    )


@pytest.mark.parametrize("condition", list(CASES))
@pytest.mark.parametrize(
    "name,index,tol",
    # CL is 1e-12 rather than 1e-7 since the lift table went in: CL0 used to be
    # an intercept SOLVED to reproduce lift at the reference point, and carried
    # that solve's residual; the table is 737.xml's own 0.20. Worst measured fell
    # from 5.5e-09 to 7.1e-14 at cruise and 9.6e-09 to 1.3e-13 at approach.
    [("CL", 0, 1e-12), ("CY", 2, 1e-10), ("Cl", 3, 1e-7), ("Cn", 5, 1e-7)],
)
def test_layer1_coefficients_with_no_model_difference_agree(name, index, tol, condition):
    """The four coefficients both engines model the same way.

    These need no real tolerance beyond round-off. JSBSim's CL is a table, but
    it is LINEAR on the segment swept here, and CY, Cl and Cn are linear in
    every swept variable in both engines. Worst measured across the sweep:
    CL 7.1e-14, CY 1.6e-14, Cl 1.1e-8, Cn 1.0e-8.

    Cl and Cn were 7.0e-6 and 1.7e-6 until the moment coefficients were referred
    to the AERORP instead of the CG. The residual was the side force acting on
    the CG offset, folded into constant Clb and Cnb where it belongs in r x F;
    referring them properly dropped both by well over two orders, to round-off.
    """
    ref, _cond, _trim, ac = case(condition)
    worst, where = 0.0, None
    for i, point in enumerate(ref.sweep):
        err = abs(_atisim_coefficients(point, ac)[index] - point.coefficients[index])
        if err > worst:
            worst, where = err, i
    assert worst <= tol, (
        f"{condition}/{name}: worst |difference| {worst:.3e} > {tol:.1e} "
        f"at sweep point {where}"
    )


@pytest.mark.parametrize("condition", list(CASES))
def test_layer1_drag_difference_is_exactly_the_terms_atisim_lacks(condition):
    """CD disagrees, and the disagreement is accounted for rather than allowed.

    atisim now carries a linear CD_alpha and a quadratic CD_beta, so the
    prediction is what those two forms do NOT cover plus the CDde term it has no
    home for at all:

      - alpha: the table's departure from the straight line CD_alpha fits. The
        table is linear in alpha above zero, so this is nearly nothing there,
        and grows where the sweep crosses toward the kink at the origin.
      - beta: JSBSim's linearly interpolated table against the quadratic, which
        agree at the 0.26 rad breakpoint and diverge below it by design.
      - elevator: CDde, which atisim freezes into CD0 at the trim deflection.

    The induced term needs no prediction: e is set so CL^2/(pi e AR) reproduces
    JSBSim's CDi = 0.043 CL^2 exactly.
    """
    ref, _cond, trim, ac = case(condition)
    worst, where = 0.0, None
    for i, point in enumerate(ref.sweep):
        got = _atisim_coefficients(point, ac)[1]
        unexplained = abs((point.coefficients[1] - got) - _missing_drag(point, trim, ac))
        if unexplained > worst:
            worst, where = unexplained, i
    assert worst <= 1e-3, (
        f"{condition}: CD difference beyond the known missing terms "
        f"{worst:.3e} at point {where}"
    )


@pytest.mark.parametrize("condition", list(CASES))
def test_layer1_pitching_moment_difference_is_exactly_the_alphadot_term(condition):
    """Cm agrees to round-off once one term is accounted for.

    Referring the moments to the AERORP removed the whole quadratic residual the
    constant-Cma fold used to produce. What is left is not a linearisation error
    but a difference in WHERE the two engines are asked for alphadot.

    `aero.coefficients` is the raw build-up and takes alphadot as an argument;
    this test calls it with none, because the aircraft's own alphadot is
    resolved a level up in `dynamics.derivatives`, not here. JSBSim's reference
    was recorded at states that carry a real alphadot. So

        JSBSim - atisim = Cmadot * alphadot_hat

    exactly, and subtracting it leaves 2.5e-11 at cruise and 1.5e-9 at approach.

    This is an equality, not a tolerance: a defect in the pitch build-up could
    not hide inside it.
    """
    ref, _cond, _trim, ac = case(condition)
    worst, where = 0.0, None
    for i, point in enumerate(ref.sweep):
        got = _atisim_coefficients(point, ac)[4]
        difference = point.coefficients[4] - got
        expected = float(ac.Cmadot) * point.alphadot * point.ci2vel
        residual = abs(difference - expected)
        if residual > worst:
            worst, where = residual, i
    assert worst <= 1e-7, (
        f"{condition}: Cm difference beyond the alphadot term "
        f"{worst:.3e} at sweep point {where}"
    )


# ---------------------------------------------------------------------------
# Layer 2 -- trim
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("condition", list(CASES))
def test_layer2_longitudinal_trim_matches(condition):
    """Both engines' own trim algorithms, at the same condition.

    Measured: alpha 1.9807 deg against JSBSim's 1.9650, elevator -0.053362
    against -0.051922 rad, thrust +1.12% at cruise and +1.56% at approach. The
    thrust difference is the linear-throttle approximation plus the ram fit
    residual, both documented in _boeing_737's docstring; the tolerance is 2%.

    The thrust LEVEL did not move when the ram fit was re-banded, and could not
    have: max_thrust is solved so the model reproduces JSBSim's thrust AT the
    condition, so re-fitting mach_ram moves the two together and leaves their
    product at the trim point alone. What re-banding changed is the SLOPE with
    Mach, which is what the phugoid damping reads -- see layer 3.
    """
    from atisim.atmosphere import RHO0, density

    _ref, cond, expected, ac = case(condition)
    (alpha, elevator, throttle), residual = _atisim_trim(condition)
    assert residual < 1e-8, f"atisim's own trim did not converge: {residual:.2e}"
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
    """JSBSim's 30 deg banked trim is in the reference; atisim has no banked trim.

    trim.trim solves the WINGS-LEVEL problem only -- its unknowns are
    [alpha, elevator, throttle], with no bank and no aileron or rudder. So the
    turn case cannot be compared today. It is recorded rather than dropped: the
    reference already holds JSBSim's converged 30 deg solution, so the
    comparison is one banked-trim solver away.

    This test asserts the gap rather than skipping silently, so that adding a
    banked trim makes it fail and forces the comparison to be written.
    """
    from atisim import trim as trim_mod

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

    Measured: short period wn 1.76841 against 1.76918 (0.04%), zeta 0.39305
    against 0.39294 (0.03%). At approach, 0.08% and 0.08%.

    THE HISTORY OF THIS NUMBER IS THE POINT, because it went
    0.04% -> 3.95% -> 1.30% -> 0.04% and only the last is honest.

    The FIRST 0.04% was two errors cancelling. Cma was recovered by a central
    difference about the CG, which is contaminated by Cmadot because setting
    alpha away from trim also sets alphadot (d(alphadot)/d(alpha) = -0.529 /s);
    that put it at -1.0637 instead of the alphadot-free -1.1309. atisim then
    had no aircraft-motion alphadot coupling at all, so its linearisation was
    missing exactly the term the contamination stood in for.

    Referring the moments to the AERORP fixed the coefficient and left the
    missing term exposed: 3.95%. Resolving the aircraft's own alphadot in
    dynamics.derivatives supplied the term: 1.30%. Giving the drag its CD_alpha
    slope closed the rest: 0.04%, now at BOTH conditions rather than one.

    The last step is the one that shows why AERORP mattered. Referring moments
    to the AERORP makes the pitching moment inherit the force error through
    r x F instead of absorbing it into a fitted Cma -- so a drag slope that was
    0.1267 against JSBSim's 0.2113 could no longer hide, and fixing it moved the
    short period by 1.3%. The CG-referenced model would have shown nothing.

    THE PHUGOID IS TWO DIFFERENT PROBLEMS, and this docstring used to name only
    one of them -- "a slow drag-and-thrust energy exchange" is the DAMPING half
    and says nothing about the frequency, which is the larger error.

      frequency, 6.58%: M_u, the pitching moment due to speed. atisim has none
      beyond an alphadot coupling; JSBSim's Cmde is a Mach table, -1.20 at M 0
      and -0.30 at M 2, and delta_e times that 0.45 slope accounts for 90.5% of
      the gap at cruise and 83.1% at approach. Structural -- a
      constant-coefficient model cannot carry a Mach-tuck term. Asserted in
      test_layer3_phugoid_frequency_gap_is_the_pitching_moment_speed_derivative.

      damping, 3.12%: X_u. This WAS 3.44% at cruise and 13.72% at approach,
      until the generator's thrust fit was re-banded to bracket each condition
      rather than sampling M 0.60-0.95 regardless. The approach entry had been
      carrying a ram coefficient fitted entirely above its own flight condition;
      re-fitting from JSBSim's own engine table put it at -0.2949 and took the
      approach damping error to 1.53%.

    The frequency numbers barely moved across that refit -- 6.582% to 6.582% at
    cruise -- which is the cross-check that the two halves really are separate
    mechanisms rather than one error split two ways.
    """
    from atisim import validation

    (alpha, elevator, throttle), _ = _atisim_trim()
    got = validation.longitudinal_modes(
        REGISTRY["boeing737"], alpha, elevator, throttle,
        CRUISE_COND.airspeed, CRUISE_COND.matched_altitude,
    )
    want = _modes_from(REF.linearization.longitudinal)
    for (wn, zeta), (wn_ref, zeta_ref), name, tol in zip(
        got, want, ("phugoid", "short period"), (1e-1, 2e-2)
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

    and folding those in gives zeta 0.34402 against 0.34410 and spiral 16.653 s
    against 16.691 s. The pitch and roll channels have no such feedback -- they
    are summers and gearing only -- which is why the longitudinal comparison
    needs no correction at all and lands at 0.04%.

    Those figures are AFTER the Ixz sign correction. With the sign as originally
    shipped they read 0.33803 and 16.647 -- a 1.76% Dutch-roll damping error
    that looked like an ordinary model difference and was in fact the tensor
    being fed to the two engines differing. See
    test_inertia_cross_product_sign_matches_the_engines_own_coupling.

    *** SESSION 23: THE FOLD IS GONE AND THE DAMPER IS IN THE PLANT. ***
    `boeing737` now carries `yaw_damper_gain = 0.35` and `dynamics.derivatives`
    adds `0.35 * r` to the rudder before the aerodynamics see it, so this test
    uses the REGISTRY ENTRY UNMODIFIED. Applying the old `_replace` fold on top
    would count the damper twice.

    The plant form subsumes all THREE folds at once, and that is not a
    coincidence: `delta_r = 0.35 r` reaches `CY`, `Cl` and `Cn` through the real
    `CYdr`, `Cldr` and `Cndr`, which is algebraically what adding
    `C*dr * 0.35 * 2V/b` to `C*r` did by hand. The measured result is closer
    than the fold was -- zeta 0.34410 against JSBSim's 0.34410, where the fold
    gave 0.34402 -- because the fold linearised about one airspeed and the plant
    term uses the aircraft's actual yaw rate at every step.
    """
    from atisim import validation

    ac = REGISTRY["boeing737"]
    assert float(ac.yaw_damper_gain) == 0.35, (
        "this test compares against JSBSim's CLOSED-LOOP linearisation, so the "
        "entry must carry the damper JSBSim's FCS carries")
    (alpha, elevator, throttle), _ = _atisim_trim()
    (wn, zeta), roll_tc, spiral_tc = validation.lateral_modes(
        ac, alpha, elevator, throttle,
        CRUISE_COND.airspeed, CRUISE_COND.matched_altitude,
    )

    eigenvalues = np.linalg.eigvals(REF.linearization.lateral)
    eigenvalues = eigenvalues[np.abs(eigenvalues) > 1e-8]  # drop the psi integrator
    pair = eigenvalues[np.abs(eigenvalues.imag) > 1e-9][0]
    reals = np.sort(eigenvalues[np.abs(eigenvalues.imag) <= 1e-9].real)
    wn_ref, zeta_ref = abs(pair), -pair.real / abs(pair)
    roll_ref, spiral_ref = -1.0 / reals[0], -1.0 / reals[-1]

    # These were 5e-2 across the board, and that is how a wrong Ixz sign worth
    # 1.6-3.2% survived a whole comparison. With the sign taken from the
    # engine's own coupling instead, three of the four land at round-off and the
    # tolerances can say so. Measured: 0.022%, 0.025%, 0.0034%, 0.228%.
    #
    # 1e-3 for the oscillatory pair and the roll mode: both are dominated by
    # terms the two engines now share exactly, so what is left is the difference
    # between JSBSim's finite-differenced linearisation and this project's
    # jacfwd one.
    assert abs(wn - wn_ref) / wn_ref < 1e-3, f"dutch roll wn {wn:.5f} vs {wn_ref:.5f}"
    assert abs(zeta - zeta_ref) / zeta_ref < 1e-3, (
        f"dutch roll zeta {zeta:.5f} vs {zeta_ref:.5f}"
    )
    assert abs(roll_tc - roll_ref) / roll_ref < 1e-3, (
        f"roll TC {roll_tc:.5f} s vs {roll_ref:.5f} s"
    )
    # The spiral gets 5e-3, five times the others, because it is the one mode
    # where the two reductions are not the same problem: this project takes a
    # 4-state [v, p, r, phi] split holding u, w and theta fixed, JSBSim's block
    # is [beta, phi, p, psi, r] carved out of the full 12-state model, and the
    # spiral is a near-zero eigenvalue where that difference has room to show.
    assert abs(spiral_tc - spiral_ref) / spiral_ref < 5e-3, (
        f"spiral TC {spiral_tc:.4f} s vs {spiral_ref:.4f} s"
    )


def test_layer3_bare_airframe_would_fail_without_the_damper_correction():
    """The damper above is load-bearing, and this proves it.

    Without it the spiral disagrees by a factor of 7.6. If a future change made
    the damper unnecessary -- JSBSim linearising open-loop, say -- this test
    fails and the one above becomes wrong in a way nobody would otherwise see.

    Session 23: the bare airframe is now constructed by zeroing the entry's own
    `yaw_damper_gain` rather than by using the entry as shipped, because as
    shipped it is no longer bare. Same statement, same number, different route.
    """
    import jax.numpy as jnp

    from atisim import validation

    bare = REGISTRY["boeing737"]._replace(yaw_damper_gain=jnp.array(0.0))
    (alpha, elevator, throttle), _ = _atisim_trim()
    _, _, spiral_tc = validation.lateral_modes(
        bare, alpha, elevator, throttle,
        CRUISE_COND.airspeed, CRUISE_COND.matched_altitude,
    )
    eigenvalues = np.linalg.eigvals(REF.linearization.lateral)
    eigenvalues = eigenvalues[np.abs(eigenvalues) > 1e-8]
    reals = np.sort(eigenvalues[np.abs(eigenvalues.imag) <= 1e-9].real)
    assert spiral_tc / (-1.0 / reals[-1]) > 5.0


# ---------------------------------------------------------------------------
# Layer 4 -- trajectory
# ---------------------------------------------------------------------------
# Worst |dv| and |domega| allowed per (manoeuvre, condition), m/s and rad/s.
# These are BACKSTOPS, and are labelled as such rather than dressed up as
# predictions: a coarse guard against a regression in the integrator, set at the
# measured worst plus about a fifth. The sharp statements about what layer 4
# actually measures are the two tests below this one, which separate the replay
# artifact from the model difference and bound each on its own terms.
#
# The cruise velocity pair keeps the values it already had rather than the ones
# the rule above would give -- 0.60 is tighter than 0.558 plus a fifth, and a
# tolerance is not widened just to make it uniform. Every rate tolerance came
# DOWN: the old 0.01 and 0.02 admitted twice what either manoeuvre produces, and
# the rate divergence is now known to be entirely replay artifact.
_LAYER4_BACKSTOP = {
    ("elevator_doublet", "cruise"): (0.60, 0.006),
    ("elevator_doublet", "approach"): (0.55, 0.005),
    ("rudder_kick", "cruise"): (2.00, 0.016),
    ("rudder_kick", "approach"): (1.00, 0.008),
}


@pytest.mark.parametrize("condition", list(CASES))
@pytest.mark.parametrize("manoeuvre", ("elevator_doublet", "rudder_kick"))
def test_layer4_trajectory_tracks(manoeuvre, condition):
    """Integrate 20 s on JSBSim's own prescribed surface history.

    The surfaces are the ACHIEVED deflections JSBSim recorded, not commands, so
    its FCS cannot contribute to any difference. Throttle is set to reproduce
    JSBSim's recorded thrust at the initial condition and then held, as JSBSim
    holds it.

    The two manoeuvres differ because they excite different physics, not because
    one is worse. Measured per component, at both recovery conditions:

        cruise   doublet  u 0.507  v 0.012  w 0.558   max |beta| 0.003 deg
        cruise   kick     u 1.556  v 0.933  w 0.573   max |beta| 2.909 deg
        approach doublet  u 0.437  v 0.008  w 0.261
        approach kick     u 0.818  v 0.345  w 0.252

    THIS TEST IS A BACKSTOP, and its tolerances are allowances rather than
    predictions -- see _LAYER4_BACKSTOP. What these numbers mean is settled by
    the two tests below, which split each of them into the part that is the
    replay's zero-order hold and the part that is the model. Before that split
    the doublet's w was attributed to the alphadot fold and the kick's v to a
    Dutch-roll phase difference; both attributions were of numbers that are
    mostly sampling, and both are withdrawn there.

    The kick's cruise v figure also moved from 1.245 to 0.933 when the Ixz sign
    was corrected, which is a quarter of it, and was being reported as a model
    difference before that.

    An earlier 0.25 s sampling gave 5.43 m/s on the kick purely because the
    replay flew a stale rudder between samples while the yaw damper
    moved continuously; the reference is sampled at 0.05 s for that reason.
    """
    ref, cond, _trim, ac = case(condition)
    vel_tol, rate_tol = _LAYER4_BACKSTOP[(manoeuvre, condition)]
    worst, worst_rate = _replay(ac, ref, cond, manoeuvre)

    coriolis = ref.diagnostics["coriolis_velocity_m_s"]
    assert worst.max() <= vel_tol, (
        f"{manoeuvre}/{condition}: worst velocity divergence {worst.max():.4f} m/s "
        f"> {vel_tol} (u {worst[0]:.4f}, v {worst[1]:.4f}, w {worst[2]:.4f}; the "
        f"Earth-rotation floor alone is {coriolis:.4f} m/s)"
    )
    assert worst_rate <= rate_tol, (
        f"{manoeuvre}/{condition}: worst angular-rate divergence "
        f"{worst_rate:.6f} rad/s > {rate_tol}"
    )


@pytest.mark.parametrize("condition", list(CASES))
@pytest.mark.parametrize("manoeuvre", ("elevator_doublet", "rudder_kick"))
def test_layer4_hold_error_is_first_order_in_the_sample_interval(manoeuvre, condition):
    """The reference is replayed zero-order-hold, and that costs something.

    JSBSim ran at 1/120 s with the yaw damper moving the rudder continuously;
    the reference is sampled at 0.05 s and the replay holds each sample until
    the next. So part of what layer 4 reports is the hold, not the model. This
    was known -- 0.25 s sampling put the rudder kick at 5.43 m/s -- but the
    sampling was then raised until the number looked acceptable rather than
    until the two parts were separated.

    Decimating the reference separates them, because decimation coarsens the
    hold and changes nothing else. A zero-order hold is first order in the
    interval, so the artifact must halve when the interval halves while a
    genuine model difference must not move at all.

    Measured order on the most sensitive component, from 0.05 / 0.10 / 0.20 s:
    1.04, 1.01 at cruise and 1.02, 1.02 at approach. Asserting 1.8-2.2 instead
    fails all four, which is what shows this measures the hold rather than
    restating arithmetic.

    This test is what licenses the Richardson extrapolation in
    test_layer4_model_difference_is_confined_to_forward_speed. If the order
    stopped being one, that extrapolation would stop being valid and this fails
    first.
    """
    ref, cond, _trim, ac = case(condition)
    f = {k: _replay(ac, ref, cond, manoeuvre, decimate=k)[0] for k in (1, 2, 4)}
    # The component the hold actually moves; for the doublet that is w, for the
    # kick v. Chosen by measurement rather than named, so it cannot go stale.
    i = int(np.argmax(np.abs(f[2] - f[1])))
    order = np.log2((f[4][i] - f[2][i]) / (f[2][i] - f[1][i]))
    assert 0.8 < order < 1.4, (
        f"{manoeuvre}/{condition}: hold error scales as dt^{order:.2f}, not dt. "
        "The Richardson extrapolation in the next test is no longer valid."
    )


@pytest.mark.parametrize("condition", list(CASES))
@pytest.mark.parametrize("manoeuvre", ("elevator_doublet", "rudder_kick"))
def test_layer4_what_survives_the_hold_is_two_residuals_and_no_more(manoeuvre, condition):
    """Extrapolate the hold away and layer 4 resolves into four statements.

    This is what layer 4 actually measures, and it is sharper than the single
    worst-component number it used to report. Extrapolating the first-order hold
    to dt -> 0 with 2*f(h) - f(2h), per component, at both conditions:

        cruise   doublet  u 0.507 -> 0.507   v  0.012   w 0.558 -> 0.172
        cruise   kick     u 1.556 -> 1.561   v  0.933 -> -0.125   w 0.573 -> 0.530
        approach doublet  u 0.437 -> 0.435   v  0.008   w 0.261 -> 0.092
        approach kick     u 0.818 -> 0.820   v  0.345 -> -0.057   w 0.252 -> 0.253

    ARTIFACT, not model: the v divergence and the angular rates extrapolate to
    zero or past it. The rudder kick's headline was mostly the replay flying a
    stale rudder while JSBSim's yaw damper moved continuously; underneath it the
    two engines agree on the lateral channel to within the replay's own
    resolution. Reporting 0.93 m/s of Dutch-roll disagreement would have been
    reporting the sampling.

    REAL, and secular: u. It does not move with the interval and it is still
    growing at t = 20 s. That is the drag-and-thrust difference the ledger
    already names -- part of it is the 0.409 m/s Earth-rotation floor, which
    sits in u too.

    REAL, and transient: w. It peaks with the sideslip excursion -- the kick
    reaches -0.573 m/s at t = 2.66 s against a beta peak of 2.9 deg -- and
    decays to a third of that by t = 20, so it is not an integration defect.
    Divided by airspeed it is nearly the SAME ANGLE at both conditions,

        doublet  0.0416 deg / 0.0395 deg      kick  0.128 deg / 0.108 deg

    across a 1.77x change in speed and a 6x change in altitude, which is the
    signature of a coefficient-level difference rather than anything that
    accumulates. It is bounded here and named, not explained: no term has been
    identified that predicts it, and this test is the record of that.

    The transverse bounds are set against the Earth-rotation floor the reference
    records for each component, because that floor is what atisim cannot
    reproduce even in principle, being flat-Earth and non-rotating. The
    multiples are margin, and are called margin.
    """
    ref, cond, _trim, ac = case(condition)
    f = {k: _replay(ac, ref, cond, manoeuvre, decimate=k) for k in (1, 2)}
    extrapolated = 2.0 * f[1][0] - f[2][0]
    rate = 2.0 * f[1][1] - f[2][1]
    floor = np.array([ref.diagnostics[f"coriolis_{a}_m_s"] for a in "uvw"])

    # -- the two that must vanish with the hold --
    assert extrapolated[1] <= 4.0 * floor[1], (
        f"{manoeuvre}/{condition}: the v divergence extrapolates to "
        f"{extrapolated[1]:.4f} m/s rather than to the {floor[1]:.4f} m/s "
        "Earth-rotation floor, so the lateral channel now carries a model "
        "difference the replay artifact used to hide"
    )
    assert rate <= 1e-3, (
        f"{manoeuvre}/{condition}: angular-rate divergence extrapolates to "
        f"{rate:.5f} rad/s rather than to zero"
    )

    # -- the two that must not --
    assert extrapolated[0] > 0.5 * f[1][0][0], (
        f"{manoeuvre}/{condition}: the u divergence moved with the sample "
        "interval, so it is not the secular difference it is reported as"
    )
    equivalent_alpha = np.degrees(extrapolated[2] / cond.airspeed)
    assert equivalent_alpha <= 0.2, (
        f"{manoeuvre}/{condition}: the w residual is {equivalent_alpha:.4f} deg "
        f"of equivalent alpha ({extrapolated[2]:.4f} m/s), past the 0.2 deg this "
        "comparison has measured at both conditions"
    )


def _longitudinal_pair(ac, condition):
    """(AtiSim, JSBSim) longitudinal plant matrices in ONE basis and ONE unit system.

    JSBSim's block is [vt, alpha, theta, q] with vt in ft/s;
    validation.longitudinal_matrix is [u, w, q, theta] in metres. Eigenvalues
    are invariant under both transforms -- which is why the mode comparison
    needs neither -- but an entry-by-entry statement needs both undone.
    """
    from atisim import validation
    from atisim.units import FT2M

    ref, cond, _trim, _ac = case(condition)
    (alpha, elevator, throttle), _ = _atisim_trim(condition)
    V = cond.airspeed
    s, c = np.sin(alpha), np.cos(alpha)
    P = np.array([[c, s, 0.0, 0.0], [-s / V, c / V, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]])
    ati = P @ np.asarray(validation.longitudinal_matrix(
        ac, alpha, elevator, throttle, V, cond.matched_altitude)) @ np.linalg.inv(P)
    scale = np.array([FT2M, 1.0, 1.0, 1.0])
    js = ref.linearization.A[:4, :4] * (scale[:, None] / scale[None, :])
    return ati, js


def _phugoid_wn(A):
    return _modes_from(A)[0][0]


@pytest.mark.parametrize("condition", list(CASES))
def test_layer3_phugoid_frequency_gap_is_the_pitching_moment_speed_derivative(condition):
    """The phugoid is PREDICTED here, not allowed a tolerance.

    Sessions 17 and 18 put the whole phugoid disagreement down to "a slow
    drag-and-thrust energy exchange". That is the DAMPING story. The FREQUENCY
    error is one entry of the plant matrix -- M_u, the pitching moment due to
    speed -- and substituting JSBSim's value for it alone collapses the gap:

        substituting          cruise wn         approach wn
        nothing               0.05581  6.58%    0.09370  3.42%
        d(qdot)/d(vt) alone   0.05251  0.27%    0.09063  0.03%
        JSBSim                0.05237           0.09060

    Every other entry of the 4x4 agrees to within 2.7% and moves the phugoid not
    at all -- asserted below by requiring a TENFOLD reduction, which substituting
    X_u instead does not produce (it moves the error by less than 1e-4 of
    itself). So this localises the disagreement rather than bounding it.

    WHY atisim's M_u differs is a structural limit, not a defect: a
    constant-coefficient model cannot carry a Mach-tuck term, and the phugoid is
    the only mode slow enough for speed derivatives to dominate -- which is
    exactly why the short period agrees to 0.04% and this does not. See
    test_atisim_has_no_aerodynamic_speed_derivative_of_pitching_moment.

    THE 0.5% BOUND IS AN ALLOWANCE, and is labelled one. What is left after the
    substitution is the other entries' disagreement leaking in, which this test
    does not compute. The tenfold assertion is the predictive half.
    """
    ac = REGISTRY[CASES[condition][0]]
    ati, js = _longitudinal_pair(ac, condition)
    want = _phugoid_wn(js)
    before = abs(_phugoid_wn(ati) - want) / want

    swapped = ati.copy()
    swapped[3, 0] = js[3, 0]          # M_u = d(qdot)/d(vt)
    after = abs(_phugoid_wn(swapped) - want) / want

    assert after < before / 10.0, (
        f"{condition}: M_u no longer explains the phugoid frequency -- "
        f"substituting it moved the error {before:.4%} -> {after:.4%}, less than "
        "the tenfold this predicts. Another derivative now dominates."
    )
    assert after < 5e-3, (
        f"{condition}: {after:.4%} left after substituting M_u, past the 0.5% "
        "the other entries' disagreement accounts for"
    )


def test_atisims_speed_derivative_of_pitching_moment_is_the_cmde_mach_schedule():
    """*** THIS TEST WAS INVERTED IN SESSION 23, AND THAT IS THE POINT. ***

    It used to be `test_atisim_has_no_aerodynamic_speed_derivative_of_pitching_
    moment`, and it PINNED AN ABSENCE: at fixed alpha the build-up had no Mach
    dependence at all except wave drag, so M_u existed only as an alphadot
    coupling and zeroing Cmadot/CLadot sent it to machine zero. Session 19 used
    that to prove the phugoid frequency gap was structural -- "there is no
    coefficient here that could be adjusted to close it without inventing a Mach
    schedule the source would have to supply."

    The source supplied one. 737.xml schedules Cmde on Mach, -1.20 at M 0 to
    -0.30 at M 2, and session 23 read it out of the file and put it in. So the
    absence is gone, deliberately, and this test now pins its REPLACEMENT: the
    aerodynamic speed derivative exists, and it comes from that table.

    What it bought, measured: the cruise phugoid frequency error went 6.58% ->
    0.45% and the approach 3.39% -> 0.49%, and M_u changed SIGN at cruise to
    agree with JSBSim's. Session 19 predicted exactly this -- substituting
    JSBSim's M_u alone cut the error tenfold -- and supplying the mechanism
    rather than the number reproduces it.
    """
    import jax.numpy as jnp

    for condition in CASES:
        ac = REGISTRY[CASES[condition][0]]
        assert ac.Cmde_table_mach.size, f"{condition}: entry lost its Cmde table"
        ati, js = _longitudinal_pair(ac, condition)

        # The alphadot coupling is still there and still real.
        bare = ac._replace(Cmadot=jnp.array(0.0), CLadot=jnp.array(0.0))
        without, _ = _longitudinal_pair(bare, condition)
        assert abs(ati[3, 0]) > 1e-5, f"{condition}: M_u vanished; the probe is vacuous"
        assert abs(without[3, 0]) > 1e-5, (
            f"{condition}: M_u is {without[3, 0]:.3e} with the alphadot terms "
            "removed. It should now be NON-zero -- the Cmde Mach schedule is an "
            "aerodynamic speed dependence and this is where it shows."
        )

        # And the schedule is identified as the SOURCE by probing the build-up
        # directly rather than through the plant matrix. At fixed alpha and fixed
        # controls, dCm/dV is EXACTLY zero without the table and non-zero with
        # it -- an exact statement, where the same claim read off M_u carries
        # ~8e-10 of residue from assembling a 4x4 by jacfwd (against a real M_u
        # of -8.7e-05, so under 1e-5 of it). The direct probe is the sharper
        # instrument and the one that cannot drift.
        flat = ac._replace(
            Cmde_table_mach=jnp.zeros(0), Cmde_table_Cmde=jnp.zeros(0),
            Clda_table_mach=jnp.zeros(0), Clda_table_Clda=jnp.zeros(0),
        )
        from atisim import aero
        from atisim.state import Controls

        (alpha, elevator, throttle), _ = _atisim_trim(condition)
        _ref, cond, _t, _ac = case(condition)
        V, a_sound = cond.airspeed, jnp.array(cond.sound_speed)

        def dCm_dV(entry, dv=1.0):
            def Cm(Vx):
                vel = jnp.array([Vx * np.cos(alpha), 0.0, Vx * np.sin(alpha)])
                ctl = Controls(elevator=jnp.array(elevator), aileron=jnp.array(0.0),
                               rudder=jnp.array(0.0), throttle=jnp.array(throttle))
                return aero.coefficients(vel, jnp.zeros(3), ctl, entry, a_sound)[4]
            return float(Cm(jnp.array(V + dv)) - Cm(jnp.array(V - dv))) / (2.0 * dv)

        assert dCm_dV(flat) == 0.0, (
            f"{condition}: dCm/dV is {dCm_dV(flat):.3e} with the Cmde schedule "
            "removed, so a SECOND aerodynamic speed dependence has appeared -- "
            "find it before trusting the attribution above")
        assert abs(dCm_dV(ac)) > 1e-6, (
            f"{condition}: dCm/dV is {dCm_dV(ac):.3e} WITH the schedule, which "
            "is too small to be the mechanism the phugoid result rests on")

        # The sign now agrees with JSBSim's, which it did not before: atisim read
        # +1.114e-04 at cruise against JSBSim's -1.024e-04.
        assert np.sign(ati[3, 0]) == np.sign(js[3, 0]), (
            f"{condition}: M_u sign {ati[3, 0]:+.3e} against JSBSim's "
            f"{js[3, 0]:+.3e}")


@pytest.mark.parametrize("condition", list(CASES))
def test_the_wave_drag_onset_sits_above_the_recovery_mach(condition):
    """A tripwire on the one Mach term atisim does have.

    kappa_airfoil is FITTED to place this project's Korn/Lock rise at the Mach
    where JSBSim's CDmach table leaves zero, so wave drag is exactly zero at both
    recovery points and contributes nothing to any layer -- including nothing to
    M_u, which is what lets the alphadot attribution above stand.

    The cruise margin is thin: onset at M 0.78998 against a trim M 0.78000, which
    is 0.010 Mach, about 3 m/s of airspeed. A change to kappa, t/c or sweep that
    moved the onset below the trim point would switch wave drag on inside the
    linearisation and move the phugoid for a reason nobody was looking for. This
    fails first if that happens.
    """
    import jax.numpy as jnp

    from atisim import aero

    ref, cond, _trim, ac = case(condition)
    CL = ref.derivatives["CL_trim"]
    onset = float(aero.drag_divergence_mach(jnp.array(CL), ac)) - float(aero._MDD_OFFSET)
    mach = cond.airspeed / cond.sound_speed
    assert mach < onset, (
        f"{condition}: flown at M {mach:.5f}, at or above the M {onset:.5f} wave-drag "
        "onset -- wave drag is now inside the linearisation and the phugoid "
        "attribution in this file no longer holds"
    )
    assert float(aero.wave_drag(jnp.array(mach), jnp.array(CL), ac)) == 0.0


@pytest.mark.parametrize("condition", list(CASES))
def test_layer1_lift_matches_through_the_stall(condition):
    """CL against the engine across the WHOLE table, not just its linear part.

    The ordinary sweep stays inside +/- 4 deg, where 737.xml's CL table is a
    straight line -- so layer 1's 5.5e-09 was a statement about one segment. This
    runs the whole curve: past the break at 0.23 rad, out to both clamped
    endpoints, and into the falling branch beyond.

    Before the table, atisim reported 2.5x the source's lift at alpha 20 deg and
    6.9x at 25 deg, because CL0 + CLa*alpha keeps climbing where the table falls.
    That is the "reports lift the source model does not have" warning in
    _boeing_737's docstring, and it is now closed rather than documented.
    """
    ref, _cond, _trim, ac = case(condition)
    assert len(ref.stall_sweep) >= 14, "the stall sweep did not load"

    import jax.numpy as jnp

    from atisim import aero
    from atisim.state import Controls

    zero = Controls(elevator=jnp.array(0.0), aileron=jnp.array(0.0),
                    rudder=jnp.array(0.0), throttle=jnp.array(0.0))
    V, a_snd = _cond.airspeed, _cond.sound_speed
    worst, where = 0.0, None
    for point in ref.stall_sweep:
        vel = jnp.array([V * np.cos(point.alpha), 0.0, V * np.sin(point.alpha)])
        got = float(aero.coefficients(vel, jnp.zeros(3), zero, ac, jnp.array(a_snd))[0])
        if abs(got - point.CL) > worst:
            worst, where = abs(got - point.CL), np.degrees(point.alpha)
    assert worst < 1e-9, (
        f"{condition}: CL differs from the engine by {worst:.3e} at "
        f"alpha {where:.2f} deg, across the full table"
    )
