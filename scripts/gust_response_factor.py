"""The gust response factor: the turbulence comparison that CAN fail.

WHY THIS EXISTS. S2 compared a peak factor -- a load divided by its own rms --
and PROJECT.md section 4 records that it cannot discriminate: `boeing737` and
`boeing747` differ by 3.1% on it where the measured population's spread is ten
times that. The reason is structural. A peak factor normalises away exactly the
quantity that distinguishes aeroplanes, which is the GAIN from gust to load.

This measures the gain itself. Quasi-steadily

    sigma_nz / sigma_w  =  rho * V * S * CLa / (2 * m * g)      [g per m/s]

which is lift slope over wing loading: pure aerodynamics, and nothing about it
cancels. The measured ratio divided by that quasi-steady value is the
ATTENUATION -- what unsteady lift, pitch response and above all the aeroplane's
own plunge take out. Its reciprocal 1/(g * quasi) is the PLUNGE TIME CONSTANT,
and a lighter, more heavily loaded wing escapes more of the gust.

THE MEASURED SIDE IS REAL FLIGHT DATA, assembled from three sources now held:

  sigma_dn, weight, TAS, altitude  TPAWS Table 1        atisim/data/tpaws_...csv
  sigma_w per event                TPAWS Figure 133     atisim/data/tpaws_fig133...
  the B-757's own C_Nalpha(M, h)   Stewart Table 1      atisim/data/stewart_...csv
  S = 1951 ft^2                    Stewart, printed p. 2

THE POPULATION MISMATCH IS BRACKETED, NEVER PAPERED OVER. Figure 133 carries 78
points for "all 2002 turbulence events"; Table 1 carries 49 from 2002, the
significant ones (sigma_dn >= 0.2 g) -- and 49 is the paper's own count on
printed p. 5, which is what says the subset was identified correctly. The
scatter is unlabelled so no marker can be paired to a row. The attenuation is
therefore quoted as a BRACKET: against all 78, and against the top 49 by
sigma_w, which is the rank-matched stand-in. Both ends are reported and the
truth is between them.

A SECOND CAVEAT that no arithmetic here removes. Both sigma_dn and sigma_w are
PEAKS of a 5 s running standard deviation, and Figure 134 shows their peaks are
NOT co-located -- up to 2 km apart, 85% within 900 m. A ratio of two separately
located maxima is not exactly a transfer gain. It is quoted as an estimate of
one, and that is a limit on the precision of everything below.

AND THE REAL AEROPLANE IS FLOWN, not substituted for. Section C builds a B-757
from the SOURCED quantities -- mass, S, c, CLa -- on top of an existing entry's
structure, because no source held here gives a 757's pitch derivatives,
inertia, span or drag polar, and rule 2 forbids inventing them. Which entry is
borrowed from is a DECLARED choice, so it is made TWICE, on the 737 and on the
747 frame. If the two agree, the borrowed dynamics do not matter for this
quantity and the answer stands. If they disagree, this test needs a real 757
longitudinal set and that is the finding.

Run: PYTHONPATH=<abs worktree root> .venv/Scripts/python.exe -u \\
         scripts/gust_response_factor.py [-n 4]
"""

import argparse
import csv
import math
import statistics as st
import sys
from pathlib import Path

import numpy as np

import atisim  # noqa: F401  -- enables x64

print(f"atisim imported from: {atisim.__file__}")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from atisim import vortex_viz, wind  # noqa: E402
from atisim.aircraft import REGISTRY  # noqa: E402
from atisim.atmosphere import density, speed_of_sound  # noqa: E402
from tpaws_peak_factor import running_std  # noqa: E402

FT2M = 0.3048
G = 9.80665
LB2KG = 0.45359237
S_757 = 1951.0 * FT2M ** 2          # Stewart, printed p. 2
C_757 = 16.64 * FT2M                # Stewart, printed p. 2
TAU, DT, SECONDS, HEAD = 5.0, 0.02, 1200.0, 100.0
SEEDS = 4

D = Path(atisim.__file__).parent / "data"
MODEL_CASES = (("boeing737", 30.0, 0.75), ("boeing747", 37.0, 0.80))


def eta(h_ft):
    """Stewart Eq. (8), printed p. 8."""
    if h_ft <= 36089.0:
        return (1.0 - 6.87535e-6 * h_ft) ** 5.2561
    return 0.22336 * math.exp(-4.80634e-5 * (h_ft - 36089.0))


def cna_table():
    grid = {}
    with open(D / "stewart_tm2003_212666_table1.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["coefficient"] == "CNalpha":
                grid.setdefault(float(r["altitude_kft"]), []).append(
                    (float(r["q_psf"]), float(r["value"])))
    for k in grid:
        grid[k].sort()
    return grid


def cna(grid, alt_kft, q_psf):
    alts = sorted(grid)
    lo = max([a for a in alts if a <= alt_kft], default=alts[0])
    hi = min([a for a in alts if a >= alt_kft], default=alts[-1])

    def at(a):
        qs, vs = zip(*grid[a])
        return float(np.interp(q_psf, qs, vs))

    if lo == hi:
        return at(lo)
    return at(lo) + (at(hi) - at(lo)) * (alt_kft - lo) / (hi - lo)


def quasi_steady(rho, V, S, cla, m):
    """g per m/s. Delta_n = rho V S CLa w / (2 m g)."""
    return rho * V * S * cla / (2.0 * m * G)


def measured_side(grid):
    rows = [r for r in csv.DictReader(open(D / "tpaws_tm2012_217337_table1.csv"))
            if not r["event"].startswith(("190", "191"))]
    implied, quasis = [], []
    for r in rows:
        alt = min(float(r["altitude_kft_first"]), float(r["altitude_kft_last"]))
        H = alt * 1000.0 * FT2M
        V = float(r["tas_ms"])
        mach = V / float(speed_of_sound(H))
        q = 1481.0 * eta(alt * 1000.0) * mach * mach
        m = float(r["weight_klb"]) * 1000.0 * LB2KG
        qz = quasi_steady(float(density(H)), V, S_757, cna(grid, alt, q), m)
        quasis.append(qz)
        implied.append(float(r["sigma_dn_g"]) / qz)
    obs = np.array(sorted(float(r["sigma_w_ms"]) for r in
                          csv.DictReader(open(D / "tpaws_fig133_sigma_uw.csv"))))
    return rows, np.array(implied), np.array(quasis), obs


def fly(ac, alt_kft, mach, sigma_w, seeds):
    """Both statistics from the same flights, which is the whole argument.

    `pf_load` and `pf_field` are S2's peak factor -- max|d| over the maximum of
    a 5 s running standard deviation -- applied to the LOAD and to the GUST
    that drove it. If the two are close, the aeroplane is barely altering the
    extreme-value structure and S2 was testing the field, not the aeroplane.
    """
    H = alt_kft * 1000.0 * FT2M
    V = mach * float(speed_of_sound(H))
    win = int(round(TAU / DT))
    sn, sw, pf_l, pf_f = [], [], [], []
    for seed in seeds:
        enc = vortex_viz.fly_in_moving_air(
            ac, wind.dryden_vertical_field(sigma_w, seed), V, H,
            label=f"seed {seed}", start_north=0.0, seconds=SECONDS, dt=DT,
            window=(-np.inf, np.inf), window_name="whole run",
            stage_sampled=True,
        )
        keep = enc.t >= HEAD
        d = np.asarray(enc.n_z[keep])
        d = d - d.mean()
        wd = np.asarray(enc.w_up[keep])
        wd = wd - wd.mean()
        sn.append(float(d.std()))
        sw.append(float(wd.std()))
        pf_l.append(float(np.max(d) / np.max(running_std(d, win))))
        pf_f.append(float(np.max(wd) / np.max(running_std(wd, win))))
    ratio = float(np.mean(sn) / np.mean(sw))
    qz = quasi_steady(float(density(H)), V, float(ac.S), float(ac.CLa),
                      float(ac.mass))
    return ratio, qz, ratio / qz, float(np.mean(pf_l)), float(np.mean(pf_f))


def build_757(template_name, mass_kg, cla):
    """The SOURCED 757 on a borrowed frame. Every borrowed field is DECLARED.

    `AR` is recomputed because it is b^2/S and S changes -- `sensitivity.py`'s
    COUPLED_FIELDS names exactly this trap. The span itself is the template's
    and is NOT a 757's: no source held here gives one.
    """
    import jax.numpy as jnp

    t = REGISTRY[template_name]
    b = float(t.b)
    # INERTIA MUST SCALE WITH THE MASS IT BELONGS TO, and `inertia_inv` is its
    # precomputed inverse -- `sensitivity.py`'s COUPLED_FIELDS names both. A
    # first version of this replaced `mass` alone and left the template's
    # inertia, giving a 757's mass with a 747's pitch inertia; that build
    # reported an "attenuation" of 3.019, an aeroplane amplifying the gust
    # threefold, which is what an incoherent pair produces.
    # The scaling I = m * l^2 with l the mean chord is a DECLARED choice, not a
    # source. It is applied to both frames so the two-frame test still asks
    # whether the borrowed structure decides the answer.
    scale = (mass_kg / float(t.mass)) * (C_757 / float(t.c)) ** 2
    inertia = jnp.asarray(np.asarray(t.inertia) * scale)
    return t._replace(
        mass=jnp.asarray(mass_kg), S=jnp.asarray(S_757), c=jnp.asarray(C_757),
        CLa=jnp.asarray(cla), AR=jnp.asarray(b * b / S_757),
        inertia=inertia, inertia_inv=jnp.linalg.inv(inertia),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("-n", "--seeds", type=int, default=SEEDS)
    args = p.parse_args()
    seeds = range(args.seeds)
    grid = cna_table()
    sigma_w_model = float(wind.mehta_residual_ceiling())

    rows, implied, quasis, obs = measured_side(grid)
    print(f"\nA  MEASURED -- the B-757, from TPAWS Table 1 + Figure 133 + "
          f"Stewart Table 1")
    print(f"   {len(rows)} Table 1 rows from 2002 (the paper's own count on "
          f"printed p. 5), {len(obs)} Figure 133 events")
    print(f"   quasi-steady factor   {st.mean(quasis):.5f} g per m/s  "
          f"-> plunge tau {1.0 / (G * st.mean(quasis)):.2f} s")
    print(f"   implied sigma_w at attenuation 1.0: mean {implied.mean():.3f} m/s")
    bracket = []
    for label, o in (("all 78 events", obs), ("top 49 by sigma_w", obs[-49:])):
        k = implied.mean() / o.mean()
        bracket.append(k)
        print(f"   vs {label:20} observed mean {o.mean():.3f} m/s  "
              f"-> ATTENUATION {k:.3f}")
    lo, hi = min(bracket), max(bracket)
    print(f"   MEASURED B-757 ATTENUATION: {lo:.3f} to {hi:.3f}")

    print(f"\nB  THE MODEL'S OWN AIRCRAFT, sigma_w = {sigma_w_model:.4f} m/s, "
          f"N = {args.seeds}")
    print("   the last two columns are S2's statistic, applied to the LOAD and "
          "to the GUST that drove it")
    print(f"   {'aircraft':12} {'tau (s)':>8} {'sig_nz/sig_w':>13} "
          f"{'quasi':>9} {'attenuation':>12} {'PF load':>9} {'PF field':>9}")
    pf = {}
    for name, alt, mach in MODEL_CASES:
        r, qz, k, pl, pfd = fly(REGISTRY[name], alt, mach, sigma_w_model, seeds)
        pf[name] = (k, pl, pfd)
        print(f"   {name:12} {1.0 / (G * qz):8.2f} {r:13.5f} {qz:9.5f} "
              f"{k:12.3f} {pl:9.4f} {pfd:9.4f}", flush=True)
    ks = [v[0] for v in pf.values()]
    pls = [v[1] for v in pf.values()]
    print(f"\n   SPREAD BETWEEN THE TWO AIRCRAFT: attenuation "
          f"{100 * abs(ks[0] - ks[1]) / st.mean(ks):.1f}%, peak factor "
          f"{100 * abs(pls[0] - pls[1]) / st.mean(pls):.1f}%. That ratio is "
          f"the whole reason this test exists.")

    print(f"\nC  THE REAL AEROPLANE -- a B-757 built from SOURCED mass, S, c "
          f"and CLa on two different borrowed frames")
    alt = 25.0
    mach = 0.70
    H = alt * 1000.0 * FT2M
    q = 1481.0 * eta(alt * 1000.0) * mach * mach
    cla757 = cna(grid, alt, q)
    mass757 = st.mean([float(r["weight_klb"]) for r in rows]) * 1000.0 * LB2KG
    print(f"   at {alt:.0f} kft M {mach:.2f}: CLa {cla757:.4f} per rad "
          f"(Stewart, interpolated), mass {mass757:.0f} kg "
          f"(TPAWS mean weight), S {S_757:.1f} m2")
    print(f"   {'frame borrowed from':22} {'tau (s)':>8} {'sig_nz/sig_w':>13} "
          f"{'quasi':>9} {'attenuation':>12}")
    got = {}
    for tmpl in ("boeing737", "boeing747"):
        ac = build_757(tmpl, mass757, cla757)
        r, qz, k, _, _ = fly(ac, alt, mach, sigma_w_model, seeds)
        got[tmpl] = k
        print(f"   {tmpl:22} {1.0 / (G * qz):8.2f} {r:13.5f} {qz:9.5f} "
              f"{k:12.3f}", flush=True)

    spread = abs(got["boeing737"] - got["boeing747"]) / st.mean(list(got.values()))
    print(f"\n   FRAME SENSITIVITY: {100 * spread:.1f}%. "
          + ("The borrowed dynamics do not decide the answer, so the 757 result "
             "stands on its SOURCED terms." if spread < 0.05 else
             "The borrowed dynamics DO decide the answer -- this test needs a "
             "real 757 longitudinal set, and that is the finding."))

    mid = st.mean(list(got.values()))
    verdict = ("INSIDE" if lo <= mid <= hi else "OUTSIDE")
    print(f"\nVERDICT  modelled 757 attenuation {mid:.3f} against the measured "
          f"bracket {lo:.3f}-{hi:.3f}: {verdict}")


if __name__ == "__main__":
    main()
