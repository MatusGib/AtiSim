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

FLOWN OVER `earth.FLAT`, NOT THE SHIPPED WGS84_J2, and the reason is that
this file measures a DRAG MODEL against a 1972 figure -- the Earth it is
flown over is not part of the claim. Trim still sets the CL the polar is
evaluated at, so a rotating, J2 Earth does move CD a little: measured
-2.1e-4 at M 0.80 and -3.2e-4 at M 0.70, because gravitation at 47N and
12,192 m is 0.36% below the constant G0 and the aeroplane therefore needs
0.36% less lift. That is 0.5% of CD, an order below this figure's own
+/-0.003 reading noise, and every point below passes under either Earth.
What it would NOT leave alone are the two residuals quoted above: they are
-0.0144 and +0.0060 over `FLAT`, which is what the text says, and -0.0147
and +0.0057 over `WGS84_J2`. Those numbers describe the polar's functional
form, so they are held fixed rather than made to drift with the gravity
model.

`FLAT` also makes this file's use of `trimmed_controls` honest. It carries
`x[1], x[2]` and drops the bank, aileron and rudder that `trim` now solves
for -- over `FLAT` those three come back identically zero (measured
-1.7e-34, 4.9e-37, -3.4e-35 at M 0.80), so nothing is being discarded. Over
`WGS84_J2` they are not zero, and the hand-built `vel_body` below would then
describe a state the solver did not produce.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from atisim import earth, trim
from atisim.aero import coefficients
from atisim.aircraft import CRUISE, REGISTRY
from atisim.atmosphere import speed_of_sound

AC = REGISTRY["boeing747"]
H = CRUISE["boeing747"]["altitude"]
A_SOUND = float(speed_of_sound(jnp.array(H)))
# 47N is the latitude the rest of this project's Earth-rotation work uses, and
# the anchor sits AT the figure's 40,000 ft, because `trimmed_state` places the
# aircraft at the anchor and this polar is a 40,000 ft curve.
ANCHOR = earth.anchor_at(np.radians(47.0), 0.0, H)
EARTH = earth.FLAT  # see the module docstring: this is a drag-model claim

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
    x, _ = trim.trim(jnp.array(V), jnp.array(H), AC, ANCHOR, EARTH)
    alpha = float(x[0])
    vel_body = jnp.array([V * np.cos(alpha), 0.0, V * np.sin(alpha)])
    controls = trim.trimmed_controls(x)
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
