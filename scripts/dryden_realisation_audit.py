"""V5 -- auditing the Dryden REALISATION, not the Dryden spectrum.

Design: docs/design/specs/2026-09-21-turbulence-response-validation-design.md,
phase V5. Every turbulence result this project has published was flown through
`wind.dryden_vertical_field`, and what has been checked about it is that it has
the variance and the spectrum it claims. Two things had not been:

  A  THE ROTATIONAL GUST. `gust_rates` turns the field's gradient into a
     pitching and a rolling input. For a field that varies along track only,
     the rolling gust is IDENTICALLY ZERO and the pitching gust has an exact
     closed form. Limb A measures both against that form. This is expected to
     fail on the rolling half and the failure IS the result -- `ASSUMPTIONS`
     E10 already records that no field varies across the span; what it does not
     record is what the realised spectrum therefore is.

  B  THE PEAK FACTOR. `dryden_vertical_field` is a Shinozuka construction --
     FIXED component amplitudes, random phases. It reproduces the target PSD
     exactly and it does not reproduce a Gaussian process's higher moments:
     every realisation has the same amplitude spectrum. Every peak and
     exceedance claim in §4 is downstream of that and none of it had been
     measured. `wind.gaussian_vertical_field` is the control -- same spectrum,
     same grid, amplitudes DRAWN -- so the comparison isolates the construction
     rather than the spectrum.

     Limb B settles `predictions.the_shinozuka_realisation_is_peak_poor`, which
     was sealed at db4eadf, before this script existed.

WHAT THE TEXTBOOK FORMULA CANNOT DO HERE, said before someone reaches for it.
The asymptotic Gaussian peak factor sqrt(2 ln(nu T)) is a limit for a narrowband
process observed over many cycles. It is not what this measures and it is not
the reference: the control is a REALISATION, flown through the same aircraft
over the same record length, so both arms carry the same finite-record bias and
the difference between them is the construction alone.

Run: PYTHONPATH=<abs worktree root> .venv/bin/python scripts/dryden_realisation_audit.py
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np

import atisim  # noqa: F401  -- enables x64
from atisim import predictions, response, trim, vortex_viz, wind
from atisim.aircraft import REGISTRY
from atisim.atmosphere import speed_of_sound

AIRCRAFT = "boeing747"
MACH = 0.80

# Matches scripts/gust_psd_identity.py, and for the same reason: 1200 s is
# seven periods of the 40 km longest component at this speed.
RECORD_SECONDS = 1200.0
SETTLE_SECONDS = 100.0
DT = 0.02

# The prediction's own floor. Sealed as "at least 48 members per arm", so
# running fewer would settle nothing.
ENSEMBLE = 48


def condition():
    H = float(wind.MEHTA_HANNIBAL_ALTITUDE)
    return REGISTRY[AIRCRAFT], MACH * float(speed_of_sound(H)), H


# ---------------------------------------------------------------------------
# Limb A -- the rotational gust the realisation actually produces
# ---------------------------------------------------------------------------


def rotational_check(ac, V, H, sigma_w, seed=0, n=200_000, span=400_000.0):
    """The realised (p_gust, q_gust) against their exact closed form.

    For a frozen field varying along north only, with the aircraft wings level
    at pitch attitude theta, `wind.gust_rates` returns

        p_gust = 0                       exactly, for every field in this project
        q_gust = -cos^2(theta) d(wind_ned[2])/dN

    (derived in `atisim/gust.py`'s header and checked there against a single
    sinusoid). Applied to a sum of sinusoids the second gives a spectrum

        Phi_q(Omega) = cos^4(theta) Omega^2 Phi_w(Omega)

    which is what this measures. It is NOT MIL-HDBK-1797's Phi_q, and the
    comparison the design asked for -- against the specification's own analytic
    rotational spectra -- IS NOT MADE HERE: neither MIL-F-8785C nor
    MIL-HDBK-1797 is in this repository (`refs/` is gitignored) and the
    specification's rotational forms are transcribed nowhere in the tree, so
    writing them from memory is exactly what `docs/DEVELOPMENT.md` rule 2
    forbids. PROJECT.md §0 carries it as the acquisition it is.
    """
    x, _ = trim.trim(jnp.array(V), jnp.array(H), ac)
    alpha = float(x[0])
    state = trim.trimmed_state(jnp.array(alpha), jnp.array(V), jnp.array(H))
    field = wind.dryden_vertical_field(sigma_w, seed)

    north = jnp.linspace(0.0, span, n)

    def one(x_n):
        pos = jnp.array([x_n, 0.0, -H])
        rates = wind.gust_rates(pos, state.quat, field)
        return jnp.array([rates[0], rates[1], field(pos)[2]])

    rows = np.asarray(jax.lax.map(one, north, batch_size=2048))
    p_gust, q_gust, w_z = rows[:, 0], rows[:, 1], rows[:, 2]

    dx = float(span / (n - 1))
    dw = np.gradient(w_z, dx)
    exact_q = -np.cos(alpha) ** 2 * dw

    print("A  the rotational gust the realisation produces")
    print(f"   rolling gust p_gust: max |p| = {np.max(np.abs(p_gust)):.3e} rad/s")
    print(f"   -- identically zero, and no amount of intensity changes it. The")
    print(f"      field varies along track only, so every strip of the wing sees")
    print(f"      the same gust (ASSUMPTIONS E10). A specification Phi_p is")
    print(f"      non-zero at every frequency, so the realised one cannot match")
    print(f"      it at ANY scaling -- this is a structural miss, not a small one.")
    err = np.max(np.abs(q_gust - exact_q)) / np.max(np.abs(exact_q))
    print(f"   pitching gust q_gust: rms = {np.std(q_gust):.6e} rad/s")
    print(f"      against its exact form -cos^2(theta) dw_z/dN, worst relative")
    print(f"      difference {err:.3e} (a central-difference gradient against")
    print(f"      an analytic one, so this is the DIFFERENCE SCHEME's error)")
    # Phi_q = cos^4(theta) Omega^2 Phi_w, checked as a ratio at three scales.
    print(f"   Phi_q / (Omega^2 Phi_w) should be cos^4(theta) = "
          f"{np.cos(alpha) ** 4:.6f}")
    f_hz, psd_q = response.spectrum(q_gust, dx)
    _, psd_w = response.spectrum(w_z, dx)
    Omega = 2.0 * np.pi * f_hz
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = psd_q / (Omega ** 2 * psd_w)
    for target_lambda in (10_000.0, 1_000.0, 100.0):
        i = int(np.argmin(np.abs(Omega - 2.0 * np.pi / target_lambda)))
        print(f"      at lambda = {target_lambda:8.0f} m: {ratio[i]:.6f}")
    return float(np.max(np.abs(p_gust)))


# ---------------------------------------------------------------------------
# Limb B -- the peak factor, Shinozuka against a Gaussian control
# ---------------------------------------------------------------------------


def peak_factors(ac, V, H, sigma_w, maker, seeds, seconds, dt, head):
    """Per-seed (up, down, sigma_nz, field peak factor) for one construction."""
    out = []
    for seed in seeds:
        field = maker(sigma_w, seed)
        enc = vortex_viz.fly_in_moving_air(
            ac, field, V, H, label=f"seed {seed}", start_north=0.0,
            seconds=seconds, dt=dt, window=(-np.inf, np.inf),
            window_name="whole run", stage_sampled=True,
        )
        keep = enc.t >= head
        n_z = enc.n_z[keep]
        w = enc.w_up[keep]
        d = n_z - np.mean(n_z)
        sigma = float(np.std(d))
        wd = w - np.mean(w)
        out.append((
            float(np.max(d) / sigma),
            float(-np.min(d) / sigma),
            sigma,
            float(np.max(np.abs(wd)) / np.std(wd)),
        ))
    return np.array(out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-n", "--seeds", type=int, default=ENSEMBLE)
    p.add_argument("--seconds", type=float, default=RECORD_SECONDS)
    p.add_argument("--dt", type=float, default=DT)
    p.add_argument("--head", type=float, default=SETTLE_SECONDS)
    p.add_argument("--skip-a", action="store_true")
    args = p.parse_args()

    ac, V, H = condition()
    sigma_w = float(wind.mehta_residual_ceiling())
    print(f"{AIRCRAFT} at M {MACH}, {H / 0.3048:.0f} ft, "
          f"sigma_w = {sigma_w:.4f} m/s\n")

    if not args.skip_a:
        rotational_check(ac, V, H, sigma_w)
        print()

    seeds = range(args.seeds)
    arms = {}
    for name, maker in (("Shinozuka", wind.dryden_vertical_field),
                        ("Gaussian ", wind.gaussian_vertical_field)):
        arms[name] = peak_factors(ac, V, H, sigma_w, maker, seeds,
                                  args.seconds, args.dt, args.head)

    print(f"B  the peak factor, N = {args.seeds} per arm, "
          f"{args.seconds:.0f} s records at dt = {args.dt}, "
          f"first {args.head:.0f} s discarded")
    print("   WINDOW DEFINITION: sigma and the extreme are BOTH taken over the")
    print("   whole retained record. That is stated because it is a choice and")
    print("   because TPAWS' own 5 s sigma window is a different one -- see")
    print("   predictions.the_model_peak_factor_lands_below_tpaws.\n")
    print(f"   {'arm':11} {'up max/sigma':>16} {'down |min|/sigma':>20} "
          f"{'sigma_nz (g)':>16} {'field max|w|/sigma':>20}")
    stats = {}
    for name, a in arms.items():
        m = a.mean(axis=0)
        se = a.std(axis=0, ddof=1) / np.sqrt(len(a))
        stats[name] = (m, se)
        print(f"   {name:11} {m[0]:9.4f}+/-{se[0]:.4f} "
              f"{m[1]:13.4f}+/-{se[1]:.4f} {m[2]:9.5f}+/-{se[2]:.5f} "
              f"{m[3]:13.4f}+/-{se[3]:.4f}")

    sh, ga = arms["Shinozuka"], arms["Gaussian "]
    for col, label in ((0, "up  "), (1, "down"), (3, "field")):
        diff = sh[:, col].mean() - ga[:, col].mean()
        se = np.sqrt(sh[:, col].var(ddof=1) / len(sh)
                     + ga[:, col].var(ddof=1) / len(ga))
        print(f"\n   {label} peak factor, Shinozuka - Gaussian = "
              f"{diff:+.4f} +/- {se:.4f}  ({abs(diff) / se:.2f} se)")

    diff = sh[:, 0].mean() - ga[:, 0].mean()
    se = np.sqrt(sh[:, 0].var(ddof=1) / len(sh) + ga[:, 0].var(ddof=1) / len(ga))
    pred = predictions.BY_NAME["the_shinozuka_realisation_is_peak_poor"]
    print(f"\n   {pred.name} (sealed {pred.sealed_at}):")
    if abs(diff) < 2.0 * se:
        print(f"      NOT SETTLED at N = {args.seeds}. The difference is "
              f"{abs(diff) / se:.2f} se, inside the 2 se the seal requires.")
    elif diff < 0:
        print(f"      RIGHT. Shinozuka is lower by {abs(diff):.4f}, "
              f"{abs(diff) / se:.2f} se.")
    else:
        print(f"      WRONG. Shinozuka is HIGHER by {diff:.4f}, "
              f"{abs(diff) / se:.2f} se.")

    print(f"\n   For the TPAWS comparison, which cannot be made in this "
          f"container:\n      the Shinozuka arm's up peak factor is "
          f"{sh[:, 0].mean():.4f} +/- "
          f"{sh[:, 0].std(ddof=1) / np.sqrt(len(sh)):.4f} (N = {len(sh)}).")


if __name__ == "__main__":
    main()
