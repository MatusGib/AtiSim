"""Does the strip load path count the gust's rolling moment twice?

`claude/linearisation-verification-bounds-b73868` (commit cf538dd, found in the
session-32 branch harvest) reported that `wind.field_model` and
`loads.strip_model` BOTH own the spanwise rolling moment and that
`integrate.step` applies both -- "both = point + strip to every digit". It could
not show on the fields of its day, all of which had dw/dy identically zero along
their tracks. Session 24 then added line vortices, which do not, and published
the strip path's effect on Mehta's field as "+22.9% of peak bank", later +17.2%
on session 30's 747 (PROJECT.md section 4, "The lateral dimension").

Read in the code, session 32: with strip=True, `vortex_viz.fly_from_state` still
builds the default `wind.field_model(field)`, whose `omega_gust[0]` is the point
model's equivalent roll rate from dw/dy, and `loads.strip_increment` adds the
strip-integrated rolling moment without removing it. The strip kernel is
verified against "the equivalent-rate treatment for a linear gradient" to 1e-6
-- the same physics the point path already applies.

This flies Mehta's field four ways on the SHIPPED `boeing747`, exactly as
`scripts/lateral.py` does:

  line            point path only                    (section 4's "line")
  line + strip    both paths -- what strip=True does (section 4's "line + strip")
  strip owns roll strip path, point roll rate removed -- what strip=True SHOULD do
  no gust roll    point roll rate removed, no strip  -- how much bank is NOT gust roll

The first two must reproduce section 4's 18.418 deg and 21.579 deg before the
others mean anything.

Run from the repository root, with an ABSOLUTE PYTHONPATH from a worktree:
    PYTHONPATH=<abs worktree root> .venv/Scripts/python.exe scripts/strip_roll_double_count.py
"""
import jax.numpy as jnp
import numpy as np

import atisim  # noqa: F401  -- enables x64
from atisim import loads, trim, vortex_viz, wind
from atisim.aircraft import CRUISE, REGISTRY
from atisim.units import FT2M, RAD2DEG

LEAD_R0 = 12.0   # scripts/lateral.py's lead-in, so the runs are the same runs
SECTION4 = {"line": 18.418, "line + strip": 21.579}   # session-30 rows, degrees


def point_roll_removed(field):
    """`wind.field_model`, with only the ROLL component of `omega_gust` zeroed.

    Pitch and yaw gust rates are kept, because the strip path is roll only and
    has nothing to put in their place. The two marks `integrate.step` reads are
    carried over, so the run is re-sampled per RK4 stage exactly as the shipped
    model is.
    """
    base = wind.field_model(field)

    def model(wind_state, state, key, dt):
        wind_ned, omega_gust, wind_state, key, alphadot = base(wind_state, state, key, dt)
        return wind_ned, omega_gust.at[0].set(0.0), wind_state, key, alphadot

    model.stage_sampled = True
    model.field = field
    return model


def fly(field, *, wind_model=None, load_model=None, dt=0.01):
    """`vortex_viz.fly`, which cannot take a wind model, unrolled so one can be
    passed. Same trim, same start, same window as `scripts/lateral.py`."""
    ac = REGISTRY["boeing747"]
    V = CRUISE["boeing747"]["airspeed"]
    H = wind.MEHTA_HANNIBAL_ALTITUDE
    r0 = float(wind.MEHTA_HANNIBAL_R0)
    x0 = wind.MEHTA_HANNIBAL_X_FT[0] * FT2M
    x1 = wind.MEHTA_HANNIBAL_X_FT[-1] * FT2M
    start = x0 - LEAD_R0 * r0
    x, _ = trim.trim(jnp.array(V), jnp.array(H), ac)
    controls = trim.trimmed_controls(x[1], x[2])
    state = trim.trimmed_state(jnp.array(float(x[0])), jnp.array(V), jnp.array(H))
    state = state._replace(pos_ned=jnp.array([start, 0.0, -H]))
    return vortex_viz.fly_from_state(
        ac, field, state, controls, label="roll", seconds=(x1 + LEAD_R0 * r0 - start) / V,
        dt=dt, window=(x0 - 2.0 * r0, x1 + 2.0 * r0), window_name="array",
        load_model=load_model, wind_model=wind_model,
    )


def peak_bank(enc):
    return float(np.abs(enc.phi[enc.window]).max()) * RAD2DEG


def main():
    ac = REGISTRY["boeing747"]
    array = wind.mehta_hannibal_array()
    field = lambda q: wind.line_vortex_wind(q, array)  # noqa: E731
    strip = loads.strip_model(field, ac)

    runs = {
        "line": fly(field),
        "line + strip": fly(field, load_model=strip),
        "strip owns roll": fly(field, wind_model=point_roll_removed(field), load_model=strip),
        "no gust roll": fly(field, wind_model=point_roll_removed(field)),
    }
    bank = {k: peak_bank(v) for k, v in runs.items()}
    nz = {k: float(v.n_z[v.window].max()) for k, v in runs.items()}

    print("Mehta's five-core field, line vortices, shipped boeing747, 37,000 ft\n")
    print(f"{'run':18}{'peak |phi| deg':>16}{'vs line':>10}{'n_z max':>10}{'section 4':>12}")
    for k in runs:
        rel = 100.0 * (bank[k] / bank["line"] - 1.0)
        ref = f"{SECTION4[k]:.3f}" if k in SECTION4 else ""
        print(f"{k:18}{bank[k]:16.3f}{rel:+9.1f}%{nz[k]:10.4f}{ref:>12}")

    print("\nharness check, against section 4's session-30 rows:")
    for k, v in SECTION4.items():
        print(f"  {k:14} {bank[k]:.3f} against {v:.3f}   diff {bank[k] - v:+.4f} deg")

    published = 100.0 * (bank["line + strip"] / bank["line"] - 1.0)
    corrected = 100.0 * (bank["strip owns roll"] / bank["line"] - 1.0)
    print(f"\nstrip path's effect on peak bank, as published: {published:+.1f}%")
    print(f"                  with roll owned by one path:  {corrected:+.1f}%")
    print(f"bank from sources other than gust roll (no gust roll run): "
          f"{bank['no gust roll']:.3f} deg, "
          f"{100.0 * bank['no gust roll'] / bank['line']:.0f}% of the line run")


if __name__ == "__main__":
    main()
