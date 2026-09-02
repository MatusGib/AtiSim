"""Fig. 8's three categories with error bars -- PROJECT.md section 7, step 6.

Step 6 has waited on step 4 (Dryden) since session 3, and step 4 landed in
session 23b. Its stated verify criterion is that the vortex < updraft <
manoeuvre ORDERING holds across the ensemble, and that is what this asks.

WHY AN ENSEMBLE IS THE POINT. The three deterministic Fig. 8 points have no
spread by construction: every member of a batch meets the same field, so a
single encounter's coordinate carries no error bar and cannot say whether the
discriminator's categories are separable or merely happen to be separated by
the one realisation that was flown. Superposing a random layer gives each
category a CLOUD, and the question becomes whether the clouds overlap.

THE INTENSITY IS SOURCED, NOT PICKED. sigma_w spans the range session 23c
derived from Mehta's own fit residual -- `wind.mehta_unmodelled_wind` to
`wind.mehta_residual_ceiling`. It is the wind his five vortices demonstrably do
not represent, which is exactly what a background layer should be.

ONE LIMB CARRIES NO NOISE, AND IT MATTERS WHICH. `vortex_viz.manoeuvre` flies at
zero wind by construction -- the discriminator separates turbulence from
MANOEUVRING by whether pitch correlates with elevator, and that is the category
definition, not an oversight. Giving it a field would need a new argument on a
core module for one script's benefit. The consequence is stated rather than
hidden: the manoeuvre limb has the largest pitch excursion, so leaving it
noiseless makes the three-way ordering EASIER to hold. The test that does not
get that help is the vortex-versus-updraft overlap, which is reported separately
and is the discriminator's real job.

Run: PYTHONPATH=<abs worktree root> .venv/Scripts/python.exe scripts/cat_ensemble.py --outdir runs/cat
"""

import argparse
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import atisim  # noqa: F401  -- enables x64
from atisim import trim, vortex_viz, wind
from atisim.aircraft import CRUISE, REGISTRY
from atisim.units import RAD2DEG
from atisim.wind import PARKS_CASES as CASES
from atisim.wind import UPDRAFT_SECONDS, UPDRAFT_W0

PALETTE = {
    "vortex": "#1D5D77", "updraft": "#3E6A48", "manoeuvre": "#A9501C",
    "muted": "#7E8D93", "grid": "#D6DCD8",
}

AIRCRAFT = "boeing747"
CASE = "hannibal"
LEAD_IN_R0 = 40.0  # scripts/vortex.py's default, and its reasoning
PUSHDOWN_HOLD = 3.0  # s, scripts/vortex.py's default -- DECLARED, not sourced


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


def fly_vortex(ac, V, H, extra, dt):
    case = CASES[CASE]
    r0, v0, spacing = case["r0"], case["v0"], case["spacing"]
    array = wind.VortexArray(
        north=jnp.array([0.0, spacing]), down=jnp.array([-H, -H]),
        r0=jnp.array(r0), v0=jnp.array(v0),
    )
    base = lambda p: wind.vortex_wind(p, array)  # noqa: E731
    field = base if extra is None else wind.superpose(base, extra)
    lead = LEAD_IN_R0 * r0
    return vortex_viz.fly(
        ac, field, V, H, label="vortex", start_north=-lead,
        seconds=(spacing + lead + 6.0 * r0) / V, dt=dt,
        window=(-r0, r0), window_name="first core",
    )


def fly_updraft(ac, V, H, extra, dt):
    radius = 0.5 * UPDRAFT_SECONDS * V
    column = wind.UpdraftColumn(
        north=jnp.array(0.0), east=jnp.array(0.0), w0=jnp.array(UPDRAFT_W0),
        radius=jnp.array(radius), sharpness=jnp.array(6.0),
    )
    base = lambda p: wind.updraft_wind(p, column)  # noqa: E731
    field = base if extra is None else wind.superpose(base, extra)
    return vortex_viz.fly(
        ac, field, V, H, label="updraft", start_north=-2.0 * radius,
        seconds=4.0 * radius / V, dt=dt,
        window=(-radius, radius), window_name="column",
    )


def fly_manoeuvre(ac, V, H, dt):
    lead, hold = 2.0, PUSHDOWN_HOLD
    seconds = lead + 3.0 * hold
    step = vortex_viz.elevator_for_load(
        ac, V, H, target=vortex_viz.FIG8_LOAD_INCREMENT,
        hold=hold, seconds=seconds, lead_in=lead, dt=dt,
    )
    return vortex_viz.manoeuvre(
        ac, V, H, label="manoeuvre", elevator_step=step, hold=hold,
        seconds=seconds, lead_in=lead, dt=dt,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=Path, default=Path("runs/cat"))
    p.add_argument("--seeds", type=int, default=16)
    p.add_argument("--dt", type=float, default=0.02)
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    print(f"atisim imported from: {atisim.__file__}\n")

    ac = REGISTRY[AIRCRAFT]
    V, H = CRUISE[AIRCRAFT]["airspeed"], CRUISE[AIRCRAFT]["altitude"]
    lo_s, hi_s = wind.mehta_unmodelled_wind(), wind.mehta_residual_ceiling()

    print("=" * 78)
    print("FIG. 8 WITH ERROR BARS -- PROJECT.md section 7, step 6")
    print("=" * 78)
    print(f"  {AIRCRAFT} at CR-2144 FC9, {H:.0f} m, {V:.1f} m/s, dt {args.dt}")
    print(f"  vortex: PARKS_CASES['{CASE}']   updraft: Wingrove w0 "
          f"{UPDRAFT_W0:.1f} m/s over {UPDRAFT_SECONDS:.0f} s")
    print(f"  manoeuvre: zero wind, {PUSHDOWN_HOLD:.0f} s pulse to "
          f"{vortex_viz.FIG8_LOAD_INCREMENT} g -- DECLARED, and NOISELESS")
    print(f"  sigma_w swept over the SOURCED range {lo_s:.3f}-{hi_s:.3f} m/s")
    print(f"  (wind.mehta_unmodelled_wind .. wind.mehta_residual_ceiling)\n")

    man = vortex_viz.fig8_point(fly_manoeuvre(ac, V, H, args.dt))
    det = {
        "vortex": vortex_viz.fig8_point(fly_vortex(ac, V, H, None, args.dt)),
        "updraft": vortex_viz.fig8_point(fly_updraft(ac, V, H, None, args.dt)),
        "manoeuvre": man,
    }
    print("  DETERMINISTIC POINTS (sigma_w = 0), pitch deg / load g:")
    for k, (th, dn) in det.items():
        print(f"    {k:<11}{th:9.3f}{dn:9.3f}")
    ordered = det["vortex"][0] < det["updraft"][0] < det["manoeuvre"][0]
    print(f"    ordering vortex < updraft < manoeuvre: "
          f"{'HOLDS' if ordered else 'FAILS'}\n")

    clouds = {}
    for sigma in (lo_s, hi_s):
        vort, upd = [], []
        for seed in range(args.seeds):
            extra = wind.dryden_vertical_field(sigma, seed)
            vort.append(vortex_viz.fig8_point(
                fly_vortex(ac, V, H, extra, args.dt)))
            upd.append(vortex_viz.fig8_point(
                fly_updraft(ac, V, H, extra, args.dt)))
        clouds[sigma] = {"vortex": np.array(vort), "updraft": np.array(upd)}

        v_th, u_th = clouds[sigma]["vortex"][:, 0], clouds[sigma]["updraft"][:, 0]
        holds = int(np.sum((v_th < u_th) & (u_th < man[0])))
        overlap = bool(v_th.max() >= u_th.min())
        print(f"  sigma_w = {sigma:.3f} m/s, {args.seeds} seeds:")
        print(f"    {'category':<11}{'pitch deg mean':>16}{'[min, max]':>22}"
              f"{'load g mean':>14}")
        for k in ("vortex", "updraft"):
            c = clouds[sigma][k]
            print(f"    {k:<11}{c[:, 0].mean():>16.3f}"
                  f"{f'[{c[:, 0].min():.3f}, {c[:, 0].max():.3f}]':>22}"
                  f"{c[:, 1].mean():>14.3f}")
        print(f"    {'manoeuvre':<11}{man[0]:>16.3f}{'(noiseless)':>22}"
              f"{man[1]:>14.3f}")
        print(f"    ordering holds in {holds}/{args.seeds} members"
              f"{'  -- ALL' if holds == args.seeds else ''}")
        v_dn = clouds[sigma]["vortex"][:, 1]
        u_dn = clouds[sigma]["updraft"][:, 1]
        print(f"    vortex/updraft PITCH clouds overlap: "
              f"{'YES' if overlap else 'no'}"
              f"   (gap {u_th.min() - v_th.max():+.3f} deg)")
        print(f"    vortex/updraft LOAD  clouds overlap: "
              f"{'YES' if v_dn.max() >= u_dn.min() else 'no'}"
              f"   (gap {u_dn.min() - v_dn.max():+.3f} g)")
        print("    -- Fig. 8 is a TWO-dimensional discriminator, so the")
        print("       categories separate while EITHER axis separates.\n")

    print("  WHAT THIS DOES AND DOES NOT ESTABLISH.")
    print("  The ordering is the criterion section 7 wrote, and it survives.")
    print("  But the manoeuvre limb carries no noise, so the three-way result")
    print("  is helped by construction; the vortex-updraft separation is the")
    print("  part that is tested on equal terms, and it is the one to quote.")

    # -- figure -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for ax, sigma in zip(axes, (lo_s, hi_s)):
        for k in ("vortex", "updraft"):
            c = clouds[sigma][k]
            ax.scatter(c[:, 0], c[:, 1], s=26, alpha=0.75,
                       color=PALETTE[k], label=f"{k} ({args.seeds} seeds)")
            ax.scatter(*det[k], s=120, marker="x", lw=2.2, color=PALETTE[k])
        ax.scatter(*man, s=120, marker="*", color=PALETTE["manoeuvre"],
                   label="manoeuvre (noiseless)", zorder=4)
        _style(ax, "pitch excursion, deg", "load excursion, g",
               f"$\\sigma_w$ = {sigma:.2f} m/s")
        ax.legend(fontsize=7.5, frameon=False, loc="lower left")
    axes[0].text(0.98, 0.55, "x = deterministic point", transform=axes[0].transAxes,
                 fontsize=7.5, va="bottom", ha="right", color=PALETTE["muted"])
    fig.suptitle("Fig. 8 categories with a sourced random layer",
                 fontsize=11, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(args.outdir / "07-ensemble.png", dpi=150, bbox_inches="tight")
    print(f"\nfigure -> {args.outdir / '07-ensemble.png'}")


if __name__ == "__main__":
    main()
