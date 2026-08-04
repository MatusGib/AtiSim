"""Korn-equation critical Mach and Lock fourth-power drag rise."""

import jax.numpy as jnp
import numpy as np
import pytest

from flightsim import aero
from flightsim.tests.conftest import make_test_aircraft


def swept_wing(sweep_deg=37.5, t_over_c=0.09, kappa=0.87):
    """A 747-like wing bolted onto the synthetic test aircraft."""
    return make_test_aircraft()._replace(
        sweep=jnp.array(np.deg2rad(sweep_deg)),
        t_over_c=jnp.array(t_over_c),
        kappa_airfoil=jnp.array(kappa),
    )


def test_no_wave_drag_below_critical_mach():
    ac = swept_wing()
    m_crit = float(aero.drag_divergence_mach(jnp.array(0.654), ac)) - aero._MDD_OFFSET
    assert float(aero.wave_drag(jnp.array(m_crit - 0.05), jnp.array(0.654), ac)) == 0.0
    assert float(aero.wave_drag(jnp.array(0.2), jnp.array(0.654), ac)) == 0.0


def test_drag_rise_slope_is_one_tenth_at_drag_divergence():
    """The definition M_dd is anchored on: dCD/dM = 0.1 there.

    If this holds, the offset relating M_crit to M_dd is consistent with the
    fourth-power law rather than an arbitrary constant.
    """
    ac = swept_wing()
    CL = jnp.array(0.654)
    m_dd = float(aero.drag_divergence_mach(CL, ac))
    h = 1e-5
    slope = (
        float(aero.wave_drag(jnp.array(m_dd + h), CL, ac))
        - float(aero.wave_drag(jnp.array(m_dd - h), CL, ac))
    ) / (2 * h)
    assert slope == pytest.approx(0.1, rel=1e-3)


def test_higher_lift_lowers_the_critical_mach():
    """The CL term is what couples wave drag to angle of attack."""
    ac = swept_wing()
    assert float(aero.drag_divergence_mach(jnp.array(0.9), ac)) < float(
        aero.drag_divergence_mach(jnp.array(0.3), ac)
    )


def test_sweep_and_thinness_raise_the_critical_mach():
    cl = jnp.array(0.5)
    unswept = float(aero.drag_divergence_mach(cl, swept_wing(sweep_deg=0.0)))
    swept = float(aero.drag_divergence_mach(cl, swept_wing(sweep_deg=35.0)))
    thick = float(aero.drag_divergence_mach(cl, swept_wing(t_over_c=0.15)))
    thin = float(aero.drag_divergence_mach(cl, swept_wing(t_over_c=0.06)))
    assert swept > unswept
    assert thin > thick


def test_boeing_747_cruise_point():
    """Sweep 37.5 deg, t/c 0.09, CL 0.654 must put M 0.80 just past drag rise.

    The 747 cruises marginally beyond M_crit, so a small but non-zero wave drag
    is the correct answer. A model giving zero here would be wrong.
    """
    ac = swept_wing()
    CL = jnp.array(0.654)
    m_dd = float(aero.drag_divergence_mach(CL, ac))
    cd_wave = float(aero.wave_drag(jnp.array(0.80), CL, ac))
    assert m_dd == pytest.approx(0.823, abs=0.005)
    assert 0.0005 < cd_wave < 0.002


def test_unswept_light_aircraft_never_sees_wave_drag():
    """The same code path must be inert for the Navion and C172."""
    ac = make_test_aircraft()  # sweep = 0, t/c = 0.12
    for mach in (0.1, 0.2, 0.3, 0.4):
        assert float(aero.wave_drag(jnp.array(mach), jnp.array(0.5), ac)) == 0.0


def test_wave_drag_is_continuous_and_flat_at_onset():
    """C1 continuity at M_crit: a kink would upset the trim Jacobian."""
    ac = swept_wing()
    CL = jnp.array(0.654)
    m_crit = float(aero.drag_divergence_mach(CL, ac)) - aero._MDD_OFFSET
    below = float(aero.wave_drag(jnp.array(m_crit - 1e-6), CL, ac))
    above = float(aero.wave_drag(jnp.array(m_crit + 1e-6), CL, ac))
    assert below == 0.0
    assert above == pytest.approx(0.0, abs=1e-20)
