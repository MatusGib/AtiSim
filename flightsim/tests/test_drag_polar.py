"""The 747 drag polar off its single fitted point (Figure IX-6, 40,000 ft).

CD0 and e were both back-solved from ONE (M, CD) reading at trim (M 0.80),
so the model reproducing that point is circular -- it proves nothing about
the polar's shape anywhere else. This digitizes nine more (M, CD) points
along the same 40,000 ft curve (read directly off the log-log figure, pixel
positions calibrated against its own axis tick labels, cross-checked at
M 0.80 against the CD 0.043 already cited in aircraft.py) and checks the
sim's CD(M) against all of them.

Digitizing off a log-scale hand-drawn 1972 figure carries real reading
uncertainty, roughly +/-0.003 per aircraft.py's own comment on the original
point. What is actually measured here is bigger and systematic, not noise:
- Near the fitted point (M 0.75-0.88) the model tracks the figure to within
  a couple thousandths -- the parabolic-polar-plus-Korn-wave-drag form is a
  good local fit.
- Below M ~0.75 the figure's CD rises faster than the model's as Mach drops
  (more induced drag at the higher CL slower flight needs, beyond what a
  constant-e parabolic polar captures) -- the model under-predicts by up to
  0.014.
- Above M ~0.88 the model's CD rises faster than the figure's -- the Korn
  wave-drag law was anchored at the single M 0.80 point and is not a fit to
  the 747's actual high-Mach rise -- the model over-predicts by up to 0.006.
Both are genuine, attributed limitations of the drag model's functional
form, not integrator or transcription bugs, so this test uses two tolerance
bands rather than one: tight near the fit, and a documented, wider band
(loose enough to pass today, tight enough to catch a real regression) away
from it.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from flightsim import trim
from flightsim.aero import coefficients
from flightsim.aircraft import CRUISE, REGISTRY
from flightsim.atmosphere import speed_of_sound

AC = REGISTRY["boeing747"]
H = CRUISE["boeing747"]["altitude"]
A_SOUND = float(speed_of_sound(jnp.array(H)))

# Digitized from Figure IX-6's 40,000 ft curve (dash-dot line), reading its
# own log-CD / linear-Mach axes: y = -376.8 - 997.0*log10(CD) (row per pixel,
# fit from the .01/.02/.05/.1/.2 tick labels), x = 1339.6 + 997.3*log10(M)
# (fit from the .1/.2/.5/1.0 tick labels). M 0.80 (FC9, the point already
# used for the CD0/e back-solve) reads 0.0439 here against the 0.043
# aircraft.py cites -- consistent to within this digitization's own noise.
FIGURE_IX6_40KFT = {
    0.70: 0.0747,
    0.72: 0.0691,
    0.78: 0.0459,
    0.80: 0.0439,
    0.82: 0.0427,
    0.85: 0.0418,
    0.88: 0.0424,
    0.90: 0.0427,
    0.92: 0.0431,
}
_NEAR_FIT_TOL = 0.004
_OFF_FIT_TOL = 0.02  # documented drag-bucket / wave-drag extrapolation gap


def _sim_cd(mach: float) -> float:
    V = mach * A_SOUND
    x, _ = trim.trim(jnp.array(V), jnp.array(H), AC)
    alpha, elevator, throttle = (float(v) for v in x)
    vel_body = jnp.array([V * np.cos(alpha), 0.0, V * np.sin(alpha)])
    controls = trim.trimmed_controls(jnp.array(elevator), jnp.array(throttle))
    _, CD, _, _, _, _ = coefficients(vel_body, jnp.zeros(3), controls, AC, jnp.array(A_SOUND))
    return float(CD)


@pytest.mark.parametrize("mach", sorted(FIGURE_IX6_40KFT))
def test_drag_polar_matches_figure_ix6_off_the_fitted_point(mach):
    cd_figure = FIGURE_IX6_40KFT[mach]
    cd_sim = _sim_cd(mach)
    residual = cd_sim - cd_figure
    tol = _NEAR_FIT_TOL if 0.75 <= mach <= 0.88 else _OFF_FIT_TOL
    assert abs(residual) < tol, (
        f"M={mach}: sim CD={cd_sim:.4f} vs figure CD={cd_figure:.4f}, "
        f"residual={residual:+.4f}, tolerance={tol}"
    )
