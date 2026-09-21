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
from atisim import predictions, trim, vortex_viz, wind
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


def rotational_check(ac, V, H, sigma_w, seed=0, n=40_000, span=400_000.0):
    """The realised (p_gust, q_gust) against their EXACT closed form.

    For a frozen field varying along north only, with the aircraft wings level
    at pitch attitude theta, `wind.gust_rates` returns

        p_gust = 0                       exactly, for every field in this project
        q_gust = -cos^2(theta) d(wind_ned[2])/dN

    (derived in `atisim/gust.py`'s header and checked there against a single
    sinusoid). `wind.dryden_vertical_components` gives the sum the field is
    built from, so the derivative is available in closed form too --

        q_gust(x) = -cos^2(theta) sum_k A_k Omega_k sin(Omega_k x + phi_k)

    -- and this comparison carries NO DIFFERENCE SCHEME AT ALL. An earlier draft
    compared against `np.gradient` and read 2.9% at 2 m spacing, which is the
    central difference's own error on a 20 m shortest wavelength and says
    nothing about the field. The spectral form of the same statement,

        Phi_q(Omega) = cos^4(theta) Omega^2 Phi_w(Omega),

    is reported as a variance ratio computed from the components rather than
    from a periodogram, for the same reason: at a 20 m shortest wavelength a
    periodogram ratio carries more leakage than signal.

    **THE COMPARISON THE DESIGN ASKED FOR IS NOT MADE HERE.** It wanted the
    realised rotational spectra against MIL-HDBK-1797's analytic `Phi_p` and
    `Phi_q`. Neither MIL-F-8785C nor MIL-HDBK-1797 is in this repository --
    `refs/` is gitignored -- and the specification's ROTATIONAL forms are
    transcribed nowhere in the tree; `wind.py` carries only Phi_u, Phi_v and
    Phi_w, verbatim from p. 47. Writing the rotational ones from memory is
    exactly what `docs/DEVELOPMENT.md` rule 2 forbids, so it is recorded as an
    acquisition in PROJECT.md section 0 instead.
    """
    x, _ = trim.trim(jnp.array(V), jnp.array(H), ac)
    alpha = float(x[0])
    state = trim.trimmed_state(jnp.array(alpha), jnp.array(V), jnp.array(H))
    field = wind.dryden_vertical_field(sigma_w, seed)
    omega_k, amplitude, phase = wind.dryden_vertical_components(sigma_w, seed)

    north = jnp.linspace(0.0, span, n)

    def one(x_n):
        return wind.gust_rates(jnp.array([x_n, 0.0, -H]), state.quat, field)

    rates = np.asarray(jax.lax.map(one, north, batch_size=1024))
    p_gust, q_gust = rates[:, 0], rates[:, 1]

    ok = np.asarray(omega_k, dtype=float)
    ak = np.asarray(amplitude, dtype=float)
    ph = np.asarray(phase, dtype=float)
    xs = np.asarray(north, dtype=float)
    exact_q = -np.cos(alpha) ** 2 * (
        ak * ok * np.sin(ok * xs[:, None] + ph)).sum(axis=1)

    print("A  the rotational gust the realisation produces")
    print(f"   rolling gust p_gust: max |p| = {np.max(np.abs(p_gust)):.3e} rad/s")
    print("   -- identically zero, and no intensity changes it. The field varies")
    print("      along track only, so every strip of the wing sees the same gust")
    print("      (ASSUMPTIONS E10). A specification Phi_p is non-zero at every")
    print("      frequency, so the realised one cannot match it at ANY scaling:")
    print("      a STRUCTURAL miss, not a small one.")
    worst = float(np.max(np.abs(q_gust - exact_q)) / np.max(np.abs(exact_q)))
    print(f"   pitching gust q_gust: rms = {np.std(q_gust):.6e} rad/s, and it")
    print(f"      matches -cos^2(theta) sum A_k Omega_k sin(.) to {worst:.2e}")
    print("      relative -- an EXACT comparison, no difference scheme in it.")
    power_w = ak ** 2 / 2.0
    power_q = np.cos(alpha) ** 4 * ok ** 2 * power_w
    print(f"   variance ratio sigma_q^2 / sigma_w^2 = "
          f"{power_q.sum() / power_w.sum():.6e} (rad/s per m/s)^2,")
    print(f"      i.e. Phi_q = cos^4(theta) Omega^2 Phi_w with "
          f"cos^4(theta) = {np.cos(alpha) ** 4:.6f}")
    print("   The rotational gust is therefore FULLY DETERMINED by the")
    print("   translational one here. That is the finding: a real atmosphere's")
    print("   p and q spectra are independent quantities, and this field has")
    print("   one degree of freedom where the specification has three.")
    return float(np.max(np.abs(p_gust))), worst


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
