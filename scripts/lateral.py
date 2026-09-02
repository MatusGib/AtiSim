"""The lateral dimension the model did not have — session 24, phase 1.

Until this session every wind field in the project was a function of along-track
distance alone. Three things followed, and they compounded:

  * `wind.strip_roll_moment` integrated to EXACTLY zero on every field, so the
    strip load path built in session 14 had never moved a reported number;
  * the lateral modes were validated as eigenvalues and never once excited;
  * `vortex_viz.Encounter` carried no roll, sideslip or rate channel, so the
    pipeline could not have reported a rolling response if one had occurred.

The model was longitudinal BY CONSTRUCTION rather than by choice, and no
document said so. This script is the measurement that changes that.

WHAT IS NEW HERE, AND WHAT IS ONLY GEOMETRY. Nothing below adds physics. Parks'
own model is two-dimensional in the plane perpendicular to the vortex LINES;
`wind.vortex_wind` evaluates it on the flight path, where the third dimension
drops out. `wind.line_vortex_wind` is the same equations written as lines in
space -- it reproduces the point model exactly along the flight path and differs
off it, which is the whole point.

Run: PYTHONPATH=<abs worktree root> .venv/Scripts/python.exe scripts/lateral.py --outdir runs/cat
"""

import argparse
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import atisim  # noqa: F401  -- enables x64
from atisim import vortex_viz, wind
from atisim.aircraft import CRUISE, REGISTRY
from atisim.units import FT2M, RAD2DEG

PALETTE = {
    "point": "#7E8D93", "line": "#1D5D77", "strip": "#A9501C",
    "wind": "#3E6A48", "grid": "#D6DCD8",
}
LEAD_R0 = 12.0
PSI = math.radians(wind.MEHTA_HANNIBAL_PSI_DEG)


def _style(ax, xlabel=None, ylabel=None, title=None):
    ax.grid(True, color=PALETTE["grid"], lw=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    if title:
        ax.set_title(title, fontsize=10, loc="left", fontweight="bold")
    ax.tick_params(labelsize=8)


def oblique():
    """Mehta's array. Its 31 deg traverse is now a real 3-D direction.

    `mehta_hannibal_array` sets both halves of the angle since this script found
    that setting only the cosine makes `line_vortex_wind` silently wrong, so
    this is the constructor unchanged -- kept as a named function because the
    obliquity is the whole subject here and deserves to be visible.
    """
    return wind.mehta_hannibal_array()


def fly(field, *, strip=False, dt=0.01):
    ac = REGISTRY["boeing747"]
    V = CRUISE["boeing747"]["airspeed"]
    H = wind.MEHTA_HANNIBAL_ALTITUDE
    r0 = float(wind.MEHTA_HANNIBAL_R0)
    x0 = wind.MEHTA_HANNIBAL_X_FT[0] * FT2M
    x1 = wind.MEHTA_HANNIBAL_X_FT[-1] * FT2M
    start = x0 - LEAD_R0 * r0
    return vortex_viz.fly(
        ac, field, V, H, label="lateral", start_north=start,
        seconds=(x1 + LEAD_R0 * r0 - start) / V, dt=dt,
        window=(x0 - 2.0 * r0, x1 + 2.0 * r0), window_name="array", strip=strip,
    )


def peak(enc, channel):
    return float(np.abs(channel[enc.window]).max())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=Path, default=Path("runs/cat"))
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    print(f"atisim imported from: {atisim.__file__}\n")

    array = oblique()
    H = wind.MEHTA_HANNIBAL_ALTITUDE

    # -- A ------------------------------------------------------------------
    print("=" * 78)
    print("A. THE TWO VORTEX FORMS, RECONCILED AND SEPARATED")
    print("=" * 78)
    print("  `line_vortex_wind` is a second implementation of a field the")
    print("  project already had. It earns its place only by reproducing the")
    print("  first one where the first one is defined -- on the flight path.\n")

    x = jnp.linspace(-6000.0, 6000.0, 2401)
    pos = jnp.stack([x, jnp.zeros_like(x), jnp.full_like(x, -H)], axis=1)
    perpendicular = wind.mehta_hannibal_array()._replace(
        cos_dpsi=jnp.array(1.0), sin_dpsi=jnp.array(0.0)
    )
    for arr, label in ((perpendicular, "perpendicular, dpsi = 0"),
                       (array, "oblique,      dpsi = 31 deg")):
        a = np.asarray(jax.vmap(lambda q: wind.vortex_wind(q, arr))(pos))
        b = np.asarray(jax.vmap(lambda q: wind.line_vortex_wind(q, arr))(pos))
        print(f"  {label}")
        print(f"      vertical component      max |diff| "
              f"{np.abs(a[:, 2] - b[:, 2]).max():.3e} m/s")
        print(f"      horizontal MAGNITUDE    max |diff| "
              f"{np.abs(np.hypot(*a[:, :2].T) - np.hypot(*b[:, :2].T)).max():.3e} m/s")
        print(f"      full vector             max |diff| "
              f"{np.abs(a - b).max():.3e} m/s")

    q = jnp.array([600.0, 0.0, -H])
    a = np.asarray(wind.vortex_wind(q, array))
    b = np.asarray(wind.line_vortex_wind(q, array))
    print(f"\n  WHERE THE OBLIQUE DIFFERENCE IS, at north = 600 m:")
    print(f"      vortex_wind      NED = [{a[0]:+8.4f} {a[1]:+8.4f} {a[2]:+8.4f}]")
    print(f"      line_vortex_wind NED = [{b[0]:+8.4f} {b[1]:+8.4f} {b[2]:+8.4f}]")
    print(f"      north ratio {b[0] / a[0]:.4f} = cos(31 deg) "
          f"{math.cos(PSI):.4f}; east/(-north) {b[1] / -a[0]:.4f} = sin "
          f"{math.sin(PSI):.4f}")
    print(f"\n  Parks' model is 2-D in the plane PERPENDICULAR TO THE LINES.")
    print(f"  The point model returns his horizontal magnitude along the FLIGHT")
    print(f"  PATH, which is exact only at dpsi = 0. At 31 deg the true")
    print(f"  perturbation is that magnitude rotated by dpsi, so {math.sin(PSI):.3f} of it")
    print(f"  is across the path -- and that is the ONLY sideslip input this")
    print(f"  field has ever had. It has been discarded since session 3.")

    # -- B ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("B. FLOWN: WHAT THE MISSING DIMENSION WAS WORTH")
    print("=" * 78)
    runs = {
        "point (vortex_wind)": fly(lambda q: wind.vortex_wind(q, array)),
        "line (line_vortex_wind)": fly(lambda q: wind.line_vortex_wind(q, array)),
        "line + STRIP loads": fly(lambda q: wind.line_vortex_wind(q, array),
                                  strip=True),
    }
    print(f"  {'run':<26}{'|phi|':>9}{'|beta|':>9}{'|p|':>10}"
          f"{'|p_gust|':>11}{'n_z max':>10}")
    print(f"  {'':<26}{'deg':>9}{'deg':>9}{'rad/s':>10}{'rad/s':>11}{'g':>10}")
    for name, enc in runs.items():
        print(f"  {name:<26}{peak(enc, enc.phi) * RAD2DEG:>9.3f}"
              f"{peak(enc, enc.beta) * RAD2DEG:>9.3f}"
              f"{peak(enc, enc.p):>10.4f}{peak(enc, enc.p_gust):>11.4f}"
              f"{enc.n_z[enc.window].max():>10.4f}")

    point, line, strip = runs.values()
    bank_l = peak(line, line.phi) * RAD2DEG
    bank_s = peak(strip, strip.phi) * RAD2DEG
    print(f"\n  TWO FIRSTS.")
    print(f"    1. A wind field rolls the aircraft: {bank_l:.2f} deg of bank")
    print(f"       through Mehta's own field at his own altitude. The point")
    print(f"       model gives exactly 0.000 -- not small, ZERO, because the")
    print(f"       equations have no y in them.")
    print(f"    2. The strip path moves a reported number for the first time:")
    print(f"       {bank_l:.2f} -> {bank_s:.2f} deg, "
          f"{100 * (bank_s / bank_l - 1):+.1f}%. Built in session 14, it had")
    print(f"       changed every result by exactly 0.000000 until now.")

    up_p = point.n_z[point.window].max() - 1.0
    print(f"\n  AND THE LONGITUDINAL ANSWER BARELY MOVES: up-increment "
          f"{up_p:.4f} g ->")
    for name, enc in (("line", line), ("strip", strip)):
        up = enc.n_z[enc.window].max() - 1.0
        print(f"    {up:.4f} g ({name}), {100 * (up / up_p - 1):+.2f}%")
    print(f"  So nothing this project has concluded was resting on the missing")
    print(f"  dimension -- which is the reassuring half of the result.")

    # -- figure -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
    w = line.window

    ax = axes[0]
    a = np.asarray(jax.vmap(lambda q: wind.vortex_wind(q, array))(pos))
    b = np.asarray(jax.vmap(lambda q: wind.line_vortex_wind(q, array))(pos))
    xs = np.asarray(x) / 1000.0
    ax.plot(xs, a[:, 1], color=PALETTE["point"], lw=1.6,
            label="point model: identically 0")
    ax.plot(xs, b[:, 1], color=PALETTE["line"], lw=1.6,
            label="line form: $-\\sin\\Delta\\psi$ of the horizontal")
    _style(ax, "north, km", "east wind, m/s",
           "A. The sideslip input the point model drops")
    ax.legend(fontsize=8, frameon=False, loc="lower right")

    ax = axes[1]
    for name, enc, key in (("point", point, "point"), ("line", line, "line"),
                           ("line + strip", strip, "strip")):
        ax.plot(enc.t[enc.window], enc.phi[enc.window] * RAD2DEG,
                color=PALETTE[key], lw=1.6, label=name)
    _style(ax, "time, s", "bank angle, deg",
           "B. The roll nothing could previously see")
    ax.legend(fontsize=8, frameon=False, loc="upper left")

    ax = axes[2]
    ax.plot(line.t[w], line.p_gust[w], color=PALETTE["line"], lw=1.6,
            label="rolling gust rate, line form")
    ax.plot(point.t[point.window], point.p_gust[point.window],
            color=PALETTE["point"], lw=1.6, label="point model: identically 0")
    _style(ax, "time, s", "$p_{gust}$, rad/s",
           "C. The gradient across the span")
    ax.legend(fontsize=8, frameon=False, loc="upper left")

    fig.tight_layout()
    fig.savefig(args.outdir / "08-lateral.png", dpi=150, bbox_inches="tight")
    print(f"\nfigure -> {args.outdir / '08-lateral.png'}")


if __name__ == "__main__":
    main()
