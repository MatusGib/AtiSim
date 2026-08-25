import jax.numpy as jnp
import numpy as np
import pytest

from atisim import aero
from atisim.atmosphere import RHO0, speed_of_sound
from atisim.state import Controls

A0 = float(speed_of_sound(0.0))  # sea-level speed of sound


def coef(vel, omega, controls, ac):
    return aero.coefficients(vel, omega, controls, ac, A0)


def fam(vel, omega, controls, ac, rho=RHO0):
    return aero.aero_forces_moments(vel, omega, controls, ac, rho, A0)


ZERO_CONTROLS = Controls(
    elevator=jnp.array(0.0),
    aileron=jnp.array(0.0),
    rudder=jnp.array(0.0),
    throttle=jnp.array(0.0),
)


def test_alpha_positive_for_upward_relative_flow():
    """w > 0 is flow from below, i.e. nose-up angle of attack."""
    V, alpha, beta = aero.air_data(jnp.array([50.0, 0.0, 5.0]))
    assert float(alpha) == pytest.approx(np.arctan2(5.0, 50.0))
    assert float(alpha) > 0
    assert float(beta) == pytest.approx(0.0)


def test_beta_positive_for_flow_from_the_right():
    V, alpha, beta = aero.air_data(jnp.array([50.0, 5.0, 0.0]))
    assert float(beta) == pytest.approx(np.arcsin(5.0 / np.sqrt(2525.0)))
    assert float(beta) > 0


def test_airspeed_floor_prevents_nan_at_zero_velocity():
    V, alpha, beta = aero.air_data(jnp.zeros(3))
    assert np.isfinite([float(V), float(alpha), float(beta)]).all()
    assert float(V) == aero.V_MIN


def test_force_directions_in_straight_and_level(test_aircraft):
    """At alpha = beta = 0 the force is -drag forward, -lift down, no side."""
    vel = jnp.array([50.0, 0.0, 0.0])
    force, moment = fam(
        vel, jnp.zeros(3), ZERO_CONTROLS, test_aircraft, RHO0
    )
    qbar = 0.5 * RHO0 * 50.0**2
    CL, CD, _, _, _, _ = coef(
        vel, jnp.zeros(3), ZERO_CONTROLS, test_aircraft
    )
    assert float(force[0]) == pytest.approx(-qbar * float(test_aircraft.S) * float(CD))
    assert float(force[1]) == pytest.approx(0.0)
    assert float(force[2]) == pytest.approx(-qbar * float(test_aircraft.S) * float(CL))
    assert float(force[2]) < 0  # lift is up, i.e. negative z


def test_lift_rotates_forward_at_positive_alpha(test_aircraft):
    """At positive alpha the lift vector tilts and contributes +x in body axes."""
    fwd_level = fam(
        jnp.array([50.0, 0.0, 0.0]), jnp.zeros(3), ZERO_CONTROLS, test_aircraft, RHO0
    )[0][0]
    fwd_alpha = fam(
        jnp.array([50.0, 0.0, 8.0]), jnp.zeros(3), ZERO_CONTROLS, test_aircraft, RHO0
    )[0][0]
    assert float(fwd_alpha) > float(fwd_level)


def test_static_pitch_stability(test_aircraft):
    """Cma < 0: increasing alpha must produce a more nose-down moment."""
    low = coef(
        jnp.array([50.0, 0.0, 0.0]), jnp.zeros(3), ZERO_CONTROLS, test_aircraft
    )[4]
    high = coef(
        jnp.array([50.0, 0.0, 10.0]), jnp.zeros(3), ZERO_CONTROLS, test_aircraft
    )[4]
    assert float(high) < float(low)


def test_control_sign_conventions(test_aircraft):
    """Each surface must move its moment the documented way."""
    vel = jnp.array([50.0, 0.0, 0.0])
    base = coef(vel, jnp.zeros(3), ZERO_CONTROLS, test_aircraft)

    up_elev = ZERO_CONTROLS._replace(elevator=jnp.array(0.1))
    right_ail = ZERO_CONTROLS._replace(aileron=jnp.array(0.1))
    left_rud = ZERO_CONTROLS._replace(rudder=jnp.array(0.1))

    # Positive elevator is trailing-edge down: more lift, nose-down moment.
    e = coef(vel, jnp.zeros(3), up_elev, test_aircraft)
    assert float(e[0]) > float(base[0])  # CL up
    assert float(e[4]) < float(base[4])  # Cm down

    # Positive aileron rolls right.
    a = coef(vel, jnp.zeros(3), right_ail, test_aircraft)
    assert float(a[3]) > float(base[3])  # Cl up

    # Positive rudder is trailing-edge left: side force right, nose-left yaw.
    r = coef(vel, jnp.zeros(3), left_rud, test_aircraft)
    assert float(r[2]) > float(base[2])  # CY up
    assert float(r[5]) < float(base[5])  # Cn down


def test_damping_derivatives_oppose_rotation(test_aircraft):
    """Clp, Cmq, Cnr are all negative, so each rate opposes itself."""
    vel = jnp.array([50.0, 0.0, 0.0])
    for idx, omega in [
        (3, jnp.array([0.2, 0.0, 0.0])),  # roll rate -> rolling moment
        (4, jnp.array([0.0, 0.2, 0.0])),  # pitch rate -> pitching moment
        (5, jnp.array([0.0, 0.0, 0.2])),  # yaw rate -> yawing moment
    ]:
        base = coef(vel, jnp.zeros(3), ZERO_CONTROLS, test_aircraft)[idx]
        rotating = coef(vel, omega, ZERO_CONTROLS, test_aircraft)[idx]
        assert float(rotating) < float(base)


def test_dynamic_pressure_scaling(test_aircraft):
    """Force scales with V^2 at fixed alpha."""
    f1 = fam(
        jnp.array([50.0, 0.0, 0.0]), jnp.zeros(3), ZERO_CONTROLS, test_aircraft, RHO0
    )[0]
    f2 = fam(
        jnp.array([100.0, 0.0, 0.0]), jnp.zeros(3), ZERO_CONTROLS, test_aircraft, RHO0
    )[0]
    np.testing.assert_allclose(np.asarray(f2), 4.0 * np.asarray(f1), rtol=1e-12)


def test_thrust_lapses_with_density(test_aircraft):
    full = ZERO_CONTROLS._replace(throttle=jnp.array(1.0))
    sea_level = aero.thrust_force(full, test_aircraft, RHO0)
    altitude = aero.thrust_force(full, test_aircraft, jnp.array(0.5 * RHO0))
    assert float(sea_level[0]) == pytest.approx(float(test_aircraft.max_thrust))
    assert float(altitude[0]) == pytest.approx(0.5 * float(test_aircraft.max_thrust))
    assert float(sea_level[1]) == 0.0 and float(sea_level[2]) == 0.0


def test_mach_ram_defaults_to_neutral(test_aircraft):
    """Aircraft defined before the ram term exists must be unaffected by it.

    The field carries a default so that adding it changed no existing aircraft.
    If that default ever moves off zero, every result in PROJECT.md section 4
    shifts silently, so it is asserted rather than trusted.
    """
    assert float(test_aircraft.mach_ram) == 0.0
    full = ZERO_CONTROLS._replace(throttle=jnp.array(1.0))
    still = aero.thrust_force(full, test_aircraft, RHO0, jnp.array(0.0))
    fast = aero.thrust_force(full, test_aircraft, RHO0, jnp.array(0.8))
    assert float(still[0]) == float(fast[0])


def test_mach_ram_raises_thrust_with_mach(test_aircraft):
    """thrust = throttle * Fmax * (rho/rho0)^n * (1 + mach_ram * M^2)."""
    ac = test_aircraft._replace(mach_ram=jnp.array(0.2))
    full = ZERO_CONTROLS._replace(throttle=jnp.array(1.0))
    still = aero.thrust_force(full, ac, RHO0, jnp.array(0.0))
    fast = aero.thrust_force(full, ac, RHO0, jnp.array(0.5))
    assert float(still[0]) == pytest.approx(float(ac.max_thrust))
    # 1 + 0.2 * 0.25 = 1.05
    assert float(fast[0]) == pytest.approx(1.05 * float(ac.max_thrust), rel=1e-12)


def test_mach_ram_defaults_to_zero_mach_when_not_passed(test_aircraft):
    """The argument is optional so pre-existing callers keep working unchanged."""
    ac = test_aircraft._replace(mach_ram=jnp.array(0.2))
    full = ZERO_CONTROLS._replace(throttle=jnp.array(1.0))
    assert float(aero.thrust_force(full, ac, RHO0)[0]) == pytest.approx(
        float(ac.max_thrust)
    )


# ---------------------------------------------------------------------------
# Nonlinear CL(alpha): the stall table
# ---------------------------------------------------------------------------
_737_CL_TABLE = ((-0.20, -0.68), (0.00, 0.20), (0.23, 1.20), (0.46, 0.20))


def _CL_at(ac, alpha_rad):
    import numpy as np

    from atisim import aero
    from atisim.state import Controls

    V = 236.5191917152
    vel = jnp.array([V * np.cos(alpha_rad), 0.0, V * np.sin(alpha_rad)])
    zero = Controls(elevator=jnp.array(0.0), aileron=jnp.array(0.0),
                    rudder=jnp.array(0.0), throttle=jnp.array(0.0))
    return float(aero.coefficients(vel, jnp.zeros(3), zero, ac,
                                   jnp.array(303.2297329682))[0])


@pytest.mark.parametrize("alpha_deg,expected", [
    (13.18, 1.2000), (16.0, 0.9859), (20.0, 0.6823), (26.36, 0.2000),
])
def test_the_737_lift_curve_breaks_where_jsbsims_table_breaks(alpha_deg, expected):
    """737.xml's CL(alpha) peaks at 1.20 near 13 deg and falls. Ours must too."""
    import numpy as np

    from atisim.aircraft import REGISTRY

    got = _CL_at(REGISTRY["boeing737"], np.radians(alpha_deg))
    assert got == pytest.approx(expected, abs=5e-4), (
        f"alpha {alpha_deg} deg: CL {got:.4f}, table says {expected:.4f}"
    )


def test_the_lift_table_is_bit_identical_to_the_linear_form_where_it_is_linear():
    """The table must change NOTHING the comparison already verified.

    737.xml's segment two IS CL0 + CLa*alpha -- slope to 4.8e-13, intercept to
    5.5e-9 -- so across 0 to 13.18 deg the two forms are the same function. If
    they ever stop being, every layer result at the recovery points moves and
    this says so before they do.
    """
    import numpy as np

    from atisim.aircraft import REGISTRY

    ac = REGISTRY["boeing737"]
    linear = ac._replace(CL_table_alpha=jnp.zeros(0), CL_table_CL=jnp.zeros(0))
    worst = 0.0
    for alpha_deg in np.linspace(0.0, 13.0, 40):
        a = np.radians(alpha_deg)
        worst = max(worst, abs(_CL_at(ac, a) - _CL_at(linear, a)))
    assert worst < 1e-8, f"table and linear form differ by {worst:.3e} inside the segment"


def test_the_lift_table_is_not_linearised_at_a_breakpoint():
    """jacfwd at a knot returns a one-sided slope, so the modes would be an
    artifact of where the breakpoints sit. Both 737 entries must trim well
    inside a segment.

    The cruise margin is the tighter one: alpha 1.98 deg against a knot at
    0.00, so 1.98 deg of room. A model change that moved the trim toward zero
    incidence -- more flap, a forward CG, a heavier fuel load -- would land on it.
    """
    import numpy as np

    from atisim.aircraft import CRUISE, REGISTRY
    from atisim.tests.test_jsbsim_737_layers import _atisim_trim

    for name, condition in (("boeing737", "cruise"), ("boeing737_approach", "approach")):
        ac = REGISTRY[name]
        (alpha, _de, _th), _ = _atisim_trim(condition)
        knots = np.asarray(ac.CL_table_alpha)
        gap = np.min(np.abs(knots - alpha))
        assert gap > np.radians(1.0), (
            f"{name} trims at alpha {np.degrees(alpha):.3f} deg, only "
            f"{np.degrees(gap):.3f} deg from a table breakpoint"
        )


def test_only_the_jsbsim_recovered_entries_carry_a_lift_table():
    """Every other aircraft keeps the linear form, so nothing else moved.

    The qualifying set is aircraft.RECOVERED_FROM_JSBSIM, not a name prefix.
    A table belongs to an entry that LINEARISES a nonlinear source model, and
    that is what membership means; it was `startswith("boeing737")` only while
    those were the only such entries.
    """
    from atisim.aircraft import RECOVERED_FROM_JSBSIM, REGISTRY

    for name, ac in REGISTRY.items():
        has_table = bool(ac.CL_table_alpha.size)
        assert has_table == (name in RECOVERED_FROM_JSBSIM), name
