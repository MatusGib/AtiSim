"""The Hannibal horizontal wind: the check the vertical wind could never make.

Data: `atisim/data/tm102186_fig7_winds.csv`, digitised by
`scripts/digitise_hannibal_horizontal_wind.py` from NASA TM-102186 Fig. 7
(printed p. 3-5) -- the horizontal and vertical winds the DC-10 met over
Hannibal, MO, as MEASURED and as Mehta 1987's five-vortex MODEL.

A core the same distance above or below the flight path gives the SAME vertical
wind; only the horizontal perturbation changes sign. So which side of the path
each core sits on -- `wind.py`'s sign of z, argued from Mehta's prose -- is
checkable only here. Bands and orderings, per CLAUDE.md rule 6.
"""

import csv
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import atisim
from atisim import wind
from atisim.units import FT2M

DATA = Path(atisim.__file__).parent / "data" / "tm102186_fig7_winds.csv"
KT2MS = 0.514444
BIAS_KT = 149.8  # Mehta 1987 p. 30, b_xy


def _load():
    out = {}
    with open(DATA, newline="") as f:
        for r in csv.DictReader(f):
            out.setdefault((r["panel"], r["curve"]), []).append((float(r["x_kft"]), float(r["value"])))
    return {k: np.array(sorted(v)) for k, v in out.items()}


def _along_path(x_kft, flip_z=False):
    """AtiSim's field along Mehta's straight path: (horizontal kt incl. bias, vertical ft/s up)."""
    alt = wind.MEHTA_HANNIBAL_ALTITUDE
    arr = wind.mehta_hannibal_array(alt)
    if flip_z:
        arr = arr._replace(down=-2.0 * alt - arr.down)
    pos = jnp.stack([jnp.asarray(x_kft) * 1000.0 * FT2M, jnp.zeros(len(x_kft)),
                     jnp.full(len(x_kft), -alt)], axis=1)
    w = np.asarray(jax.vmap(lambda p: wind.vortex_wind(p, arr))(pos))
    return w[:, 0] / KT2MS + BIAS_KT, -w[:, 2] / FT2M


@pytest.fixture(scope="module")
def data():
    return _load()


def _rms(a):
    return float(np.sqrt(np.mean(np.asarray(a) ** 2)))


def test_the_digitised_table_carries_both_panels_and_both_curves(data):
    for panel in ("horizontal", "vertical"):
        model, actual = data[(panel, "model")], data[(panel, "actual")]
        assert model.shape[0] > 500, "the solid model line, column by column"
        assert actual.shape[0] > 50, "the dotted measured curve, dot by dot"
        assert model[0, 0] < -23.0 and model[-1, 0] > 24.0


def test_the_vertical_panel_calibrates_the_reading(data):
    """THE CONTROL. AtiSim's vertical wind is pinned elsewhere, so the digitised
    vertical MODEL must land on it -- measured 1.35 ft/s RMS on a +-100 ft/s axis.
    A failure here is the digitisation, and nothing below may be believed."""
    m = data[("vertical", "model")]
    assert _rms(m[:, 1] - _along_path(m[:, 0])[1]) < 3.0


def test_the_vertical_wind_cannot_see_which_side_a_core_is_on(data):
    """The physical fact the sign test rests on, asserted exactly: flipping every
    core to the other side of the path leaves the vertical wind unchanged."""
    x = data[("vertical", "model")][:, 0]
    assert np.allclose(_along_path(x)[1], _along_path(x, flip_z=True)[1], rtol=0.0, atol=1e-9)


def test_the_horizontal_panel_confirms_which_side_of_the_path_each_core_is_on(data):
    """THE SIGN TEST. As transcribed, AtiSim's horizontal wind lands on Mehta's
    drawn model (measured 1.58 kt RMS); with every core flipped it misses by
    15.4 kt. `wind.py`'s convention, argued from prose until session 30, is
    measured here."""
    m = data[("horizontal", "model")]
    as_built = _rms(m[:, 1] - _along_path(m[:, 0])[0])
    flipped = _rms(m[:, 1] - _along_path(m[:, 0], flip_z=True)[0])
    assert as_built < 3.0
    assert flipped > 10.0
    assert flipped > 5.0 * as_built


@pytest.mark.parametrize("lo,hi,kind", [(-10, -4, "min"), (1.5, 5, "min"), (8, 18, "max")])
def test_the_horizontal_extremes_fall_where_mehta_drew_them(data, lo, hi, kind):
    """Measured: -6.69 vs -6.76 kft, +3.44 vs +3.26, +12.13 vs +12.28; values within
    1.6 kt. Location within half a thousand feet and value within 3 kt."""
    m = data[("horizontal", "model")]
    sel = (m[:, 0] >= lo) & (m[:, 0] < hi)
    f = np.argmin if kind == "min" else np.argmax
    i = f(m[sel, 1])
    xs = np.linspace(lo, hi, 2001)
    h = _along_path(xs)[0]
    j = f(h)
    assert abs(m[sel, 0][i] - xs[j]) < 0.5
    assert abs(m[sel, 1][i] - h[j]) < 3.0


def test_the_fit_misses_a_sustained_tailwind_after_the_vortex_pair(data):
    """THE FIELD AGAINST THE RECORD. Measured minus AtiSim, horizontal: -3.5 kt
    before the pair, -9.7 kt through it, +12.1 kt after it. The ordering is the
    claim -- the real wind rose through the encounter by ~15 kt more than the
    five-vortex fit carries -- and the bands are only tripwires."""
    a = data[("horizontal", "actual")]
    r = a[:, 1] - _along_path(a[:, 0])[0]

    def mean(lo, hi):
        return r[(a[:, 0] >= lo) & (a[:, 0] < hi)].mean()

    before, through, after = mean(-25, -5), mean(-5, 6), mean(6, 26)
    assert after > 8.0 and through < -5.0
    assert through < before < after
