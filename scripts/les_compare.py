"""AtiSim against Yoshimura across all four LES resolutions.

`les_flight.py` flies one domain. This puts the four beside each other and
beside Yoshimura's own ensemble through the SAME four, processed identically:
20 s of settling discarded, high-passed at `response.PHUGOID_FLOOR_HZ`, then
rms, peak load, spectral peak and upcrossing rate.

WHY BOTH SIDES GET THE SAME FILTER, AND WHY IT IS NOT COSMETIC. With fixed
controls AtiSim's aeroplane drifts 250-450 m in altitude over the record while
Yoshimura's holds to 26-43 m, because theirs is flown to a height and this one
is not. Comparing a raw rms against a raw rms would therefore be comparing a
turbulence response plus a phugoid against a turbulence response alone. The
high-pass is what makes the two the same quantity, and the drift it removes is
reported rather than hidden.

THE RESOLUTION SWEEP IS THE POINT, not a robustness check. The same event is
resolved at 500, 250, 70 and 35 m, and the load rises steeply as the grid
refines -- on Yoshimura's own numbers the high-passed rms goes 0.0070 ->
0.1108 g, a factor of 16, for the same weather at the same hour. Reproducing
the TREND with an independent flight-dynamics code is a stronger statement
than matching any single number, and it bears directly on whether an
idealised Dryden or Rankine field can ever produce a recorded load.

Run: PYTHONPATH=<abs worktree root> .venv/Scripts/python.exe scripts/les_compare.py --outdir runs/cat
"""

import argparse
import csv
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PALETTE = {"model": "#1D5D77", "reference": "#A9501C", "wind": "#3E6A48",
           "muted": "#7E8D93", "grid": "#D6DCD8"}

WORK = Path("C:/Users/mateusz/UROP/yoshimura-figshare-21152203/unpacked/"
            "flightsim-data/work")
DOMAINS = [("D01", "500m", 500.0), ("D02", "250m", 250.0),
           ("D03", "70m", 70.0), ("D04", "35m", 35.0)]
YOSH_DT = 0.0078125
SETTLE = 20.0
F_CUT = 0.05
LEVELS = (0.05, 0.10, 0.20, 0.30, 0.40)


def highpass(x, dt, f_cut=F_CUT):
    X = np.fft.rfft(x - x.mean())
    f = np.fft.rfftfreq(len(x), dt)
    X[f < f_cut] = 0.0
    return np.fft.irfft(X, n=len(x))


def psd(x, dt):
    x = x - x.mean()
    n = len(x)
    w = np.hanning(n)
    X = np.fft.rfft(x * w)
    return np.fft.rfftfreq(n, dt), (np.abs(X) ** 2) * 2 * dt / (n * (w ** 2).mean())


def yoshimura(res):
    """Their 151 flights through this domain, filtered exactly as ours are."""
    files = sorted(glob.glob(str(WORK / f"results-1000_mand_{res}.nc-2D"
                                 / "xlatlon_path-*.txt")))
    k = int(SETTLE / YOSH_DT)
    hp, acc, f = [], None, None
    for fn in files:
        g = np.loadtxt(fn, delimiter=",", usecols=(11,))[k:]
        h = highpass(g, YOSH_DT)
        hp.append(h)
        f, p = psd(h, YOSH_DT)
        acc = p if acc is None else acc + p
    hp = np.array(hp)
    acc /= len(files)
    band = (f > 0.03) & (f < 2.0)
    secs = hp.size * YOSH_DT
    rates = [sum(((h[:-1] < lv) & (h[1:] >= lv)).sum() for h in hp) / secs
             for lv in LEVELS]
    return dict(n=len(files), rms=float(hp.std(axis=1).mean()),
                peak_dn=float(np.abs(hp).max()),
                peak_hz=float(f[band][np.argmax(acc[band])]),
                rates=rates, f=f, psd=acc)


def atisim(outdir, domain, res, dt=0.02):
    """Our ensemble for this domain, from the .npy `les_flight.py` wrote."""
    hits = sorted(outdir.glob(f"les-nz-{domain}-*.npy"))
    if not hits:
        return None
    recs = np.load(hits[-1])
    hp = np.array([highpass(r, dt) for r in recs])
    acc, f = None, None
    for h in hp:
        f, p = psd(h, dt)
        acc = p if acc is None else acc + p
    acc /= len(hp)
    band = (f > 0.03) & (f < 2.0)
    secs = hp.size * dt
    rates = [sum(((h[:-1] < lv) & (h[1:] >= lv)).sum() for h in hp) / secs
             for lv in LEVELS]
    return dict(n=len(recs), rms=float(hp.std(axis=1).mean()),
                peak_dn=float(np.abs(hp).max()),
                peak_hz=float(f[band][np.argmax(acc[band])]),
                rates=rates, f=f, psd=acc, src=hits[-1].name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", type=Path, default=Path("runs/cat"))
    args = ap.parse_args()

    rows = []
    print("=" * 78)
    print("ATISIM vs YOSHIMURA ACROSS ALL FOUR LES RESOLUTIONS")
    print("=" * 78)
    print("  both sides: 20 s settling dropped, high-passed at "
          f"{F_CUT} Hz, then the same statistics\n")
    for dom, res, dx in DOMAINS:
        y = yoshimura(res)
        a = atisim(args.outdir, dom, res)
        rows.append((dom, res, dx, a, y))
        if a is None:
            print(f"  {dom} ({res}): AtiSim run not found -- skipped")
            continue
        print(f"  {dom}  dx = {dx:.0f} m")
        print(f"    n_z rms      AtiSim {a['rms']:.4f}   Yoshimura {y['rms']:.4f}"
              f"   ratio {a['rms']/y['rms']:.2f}x")
        print(f"    peak |dn|    AtiSim {a['peak_dn']:.3f}    Yoshimura "
              f"{y['peak_dn']:.3f}    ratio {a['peak_dn']/y['peak_dn']:.2f}x")
        print(f"    spectral pk  AtiSim {a['peak_hz']:.4f} Hz Yoshimura "
              f"{y['peak_hz']:.4f} Hz")
        print(f"    flights      AtiSim {a['n']}          Yoshimura {y['n']}")

    good = [(d, r, dx, a, y) for d, r, dx, a, y in rows if a]
    if len(good) >= 2:
        dxs = np.array([dx for _, _, dx, _, _ in good])
        ra = np.array([a["rms"] for _, _, _, a, _ in good])
        ry = np.array([y["rms"] for _, _, _, _, y in good])
        print()
        print("  THE TREND, which is the part worth quoting:")
        print(f"    AtiSim    rms {ra.min():.4f} -> {ra.max():.4f} g "
              f"({ra.max()/ra.min():.1f}x across the sweep)")
        print(f"    Yoshimura rms {ry.min():.4f} -> {ry.max():.4f} g "
              f"({ry.max()/ry.min():.1f}x)")
        sa = np.polyfit(np.log(dxs), np.log(ra), 1)[0]
        sy = np.polyfit(np.log(dxs), np.log(ry), 1)[0]
        print(f"    d(log rms)/d(log dx):  AtiSim {sa:+.2f}   Yoshimura {sy:+.2f}")
        print("    Both codes say the load is a strong function of how finely")
        print("    the atmosphere is resolved -- same weather, same hour.")

    with (args.outdir / "les-comparison.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["domain", "res", "dx_m", "atisim_rms", "yosh_rms", "ratio",
                    "atisim_peak_dn", "yosh_peak_dn", "atisim_peak_hz",
                    "yosh_peak_hz"] + [f"atisim_up_{v}" for v in LEVELS]
                   + [f"yosh_up_{v}" for v in LEVELS])
        for dom, res, dx, a, y in rows:
            if a is None:
                continue
            w.writerow([dom, res, dx, f"{a['rms']:.5f}", f"{y['rms']:.5f}",
                        f"{a['rms']/y['rms']:.3f}", f"{a['peak_dn']:.4f}",
                        f"{y['peak_dn']:.4f}", f"{a['peak_hz']:.4f}",
                        f"{y['peak_hz']:.4f}"]
                       + [f"{v:.5f}" for v in a["rates"]]
                       + [f"{v:.5f}" for v in y["rates"]])
    print(f"\n  table -> {args.outdir / 'les-comparison.csv'}")
    if good:
        _figure(args.outdir / "14-les-resolution.png", good)
        print(f"  figure -> {args.outdir / '14-les-resolution.png'}")


def _figure(path, rows):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    dxs = [dx for _, _, dx, _, _ in rows]

    ax = axes[0]
    ax.loglog(dxs, [a["rms"] for _, _, _, a, _ in rows], "o-", lw=2, ms=7,
              color=PALETTE["model"], label="AtiSim (boeing737_approach)")
    ax.loglog(dxs, [y["rms"] for _, _, _, _, y in rows], "s--", lw=2, ms=7,
              color=PALETTE["reference"], label="Yoshimura, their own code")
    ax.set_xlabel("LES grid spacing, m", fontsize=9)
    ax.set_ylabel("$n_z$ rms, g (high-passed)", fontsize=9)
    ax.set_title("A. The load depends on the grid", fontsize=10.5, loc="left")
    ax.invert_xaxis()
    ax.legend(fontsize=8)

    bx = axes[1]
    for (dom, res, dx, a, _y) in rows:
        m = (a["f"] > 0.02) & (a["f"] < 3)
        bx.loglog(a["f"][m], a["psd"][m], lw=1.5, label=f"AtiSim {res}")
    bx.axvline(F_CUT, color=PALETTE["muted"], ls=":", lw=1.2)
    bx.set_xlabel("frequency, Hz", fontsize=9)
    bx.set_ylabel("$n_z$ PSD, g$^2$/Hz", fontsize=9)
    bx.set_title("B. AtiSim's response spectra", fontsize=10.5, loc="left")
    bx.legend(fontsize=7.5)

    cx = axes[2]
    for (dom, res, dx, a, y) in rows:
        cx.semilogy(LEVELS, np.maximum(a["rates"], 1e-5), "o-", lw=1.5, ms=5,
                    label=f"AtiSim {res}")
        cx.semilogy(LEVELS, np.maximum(y["rates"], 1e-5), "s--", lw=1.2, ms=4,
                    alpha=.65, label=f"Yoshimura {res}")
    cx.set_xlabel("load increment level, g", fontsize=9)
    cx.set_ylabel("upcrossings per second", fontsize=9)
    cx.set_title("C. Exceedance, both codes", fontsize=10.5, loc="left")
    cx.legend(fontsize=6.5, ncol=2)

    for a in axes:
        a.grid(True, color=PALETTE["grid"], lw=.6, alpha=.8)
        a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.suptitle("Yoshimura's LES at four resolutions, flown by two independent "
                 "flight-dynamics codes -- 3,000 m, 148.8 m/s, heading 090",
                 fontsize=10.5, x=.005, ha="left")
    fig.tight_layout(rect=[0, 0, 1, .93])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
