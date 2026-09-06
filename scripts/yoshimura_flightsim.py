"""Yoshimura et al. 2023's own flight simulation, read and measured.

`flightsim-data.tar.bz2` -- 561 MB, 3% of the 17.9 GB figshare download, and the
part that can be used first. It is the SIMULATED half of their Fig. 6: their own
2-D aircraft flown through their LES field, 151 virtual flights of 100 s at each
of four LES resolutions. 604 flights, about 17 hours of record.

WHY IT MATTERS HERE, AND IT IS NOT THE REASON IT WAS FETCHED. Session 25 listed
this dataset as the route to a RECORDED acceleration history; that part is
withheld under confidentiality and is not in the tarball. What IS here is the
first CAT wind field available to this project that was not identified from an
aircraft's own accelerations -- it comes out of a weather model. Every load
comparison in PROJECT.md section 4 is flown through a field fitted TO the
accelerations it is then asked to predict. This one is not, and that is the
circularity section 4's session-23 entry opens with.

WHAT THE FILES CONTAIN. `xlatlon_path-NNNN.txt`, 17 columns per row at
dt = 1/128 s: time, u, alpha, q, theta, beta, p, r, phi, psi, G, the three wind
components at the aircraft, lat and lon. `x_cor_path-NNNN.txt` carries the
height. Column 12, `G`, is the vertical acceleration INCREMENT and not n_z: it
sits on zero, not on one.

FOUR THINGS THIS SCRIPT ESTABLISHES, ALL OF WHICH BEAR ON HOW THE DATA CAN BE
USED RATHER THAN ON WHETHER IT AGREES WITH ANYTHING:

  1  THE RESPONSE IS STRONGLY RESOLUTION-DEPENDENT. n_z rms runs 0.031 g at
     dx = 500 m to 0.117 g at dx = 35 m, a factor of 3.8, and the spectral peak
     moves 0.040 -> 0.130 Hz. A coarse LES does not merely under-resolve the
     load, it puts the response at the WRONG FREQUENCY -- the aircraft follows
     the forcing scale because the field carries no energy at the airframe's
     own. Only the finest domain reaches the aircraft's natural frequency.
     Any comparison against this dataset must state which domain it used.

  2  IT IS TWO-DIMENSIONAL AND LONGITUDINAL. beta, p, r and phi are EXACTLY
     zero in every file. So it cannot validate the lateral capability session 24
     built, and section 5's statement that no held source records a lateral CAT
     response still stands. This is worth saying plainly because the dataset
     looks at first like the lateral check the project lacks.

  3  ITS FLIGHT CONDITION IS NOT THIS PROJECT'S. 133.5 m/s at 3,000 m, against
     the 747's 235.9 m/s at 11,278 m. Different aircraft, different Mach,
     different density. Absolute loads are therefore not comparable, and the
     comparison this dataset supports is the SHAPE of the response -- where the
     energy sits, and how the exceedance curve falls -- not the size of it.

     *** THE SPEED HERE WAS 148.8 m/s UNTIL THEIR SOURCE WAS READ. ***
     That figure came from differencing the lat/lon columns of their own
     output. It is wrong: `src/fs.f90` builds those two columns as the
     inverse-distance-weighted AVERAGE of grid latitude and longitude over the
     interpolation stencil, so they are a smoothed field quantity and not a
     position. Their airspeed is `U_0 = 133` m/s in the source, and their
     `x_cor_path-*.txt` position columns run 79385.6 -> 92737.8 m in 100 s =
     133.5 m/s with y constant to the metre. The 12% error mattered: gust load
     goes as w*V, so it inflated every load this project computed through their
     field until it was found. See scripts/les_flight.py.

  4  IT HOLDS ITS FLIGHT CONDITION, AND THIS PROJECT'S ENSEMBLE DOES NOT.
     Altitude varies by 42 m over the whole 100 s record. `cat_spectra.py`'s
     upper-sigma ensemble drifts 703 m and 13.1% in airspeed, which is
     ASSUMPTIONS.md E11 and the reason the lower-sigma ensemble is the one to
     quote. E11 is therefore a consequence of flying with fixed controls, not
     something inherent to the measurement -- an independent implementation of
     the same protocol simply does not have it.

Run: PYTHONPATH=<abs worktree root> .venv/Scripts/python.exe scripts/yoshimura_flightsim.py --outdir runs/cat
"""

import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PALETTE = {"model": "#1D5D77", "reference": "#A9501C", "wind": "#3E6A48",
           "muted": "#7E8D93", "grid": "#D6DCD8"}

ROOT = Path("C:/Users/mateusz/UROP/yoshimura-figshare-21152203/unpacked/flightsim-data/work")
RESOLUTIONS = ("500m", "250m", "70m", "35m")
DT = 1.0 / 128.0
COL_G, COL_W = 11, 14

# This project's own numbers, for scale only -- a different aircraft at a
# different condition, so these are NOT an agreement test. cat_spectra.py.
ATISIM = {"sigma_lo": (2.108, 0.0823, 0.1400), "sigma_hi": (4.459, 0.1836, 0.1700)}
ATISIM_SHORT_PERIOD_HZ = 0.1640
YOSHIMURA_B787_HZ = 0.14           # the paper's own natural frequency


def _psd(x, dt):
    x = x - x.mean()
    n = len(x)
    w = np.hanning(n)
    X = np.fft.rfft(x * w)
    f = np.fft.rfftfreq(n, dt)
    return f, (np.abs(X) ** 2) * 2 * dt / (n * (w ** 2).mean())


def _load(res, root):
    files = sorted(glob.glob(str(root / f"results-1000_mand_{res}.nc-2D" / "xlatlon_path-*.txt")))
    if not files:
        raise SystemExit(f"no flights found for {res} under {root}")
    g, w, psd = [], [], None
    for fn in files:
        a = np.loadtxt(fn, delimiter=",", usecols=(COL_G, COL_W))
        g.append(a[:, 0])
        w.append(a[:, 1])
        f, p = _psd(a[:, 0], DT)
        psd = p if psd is None else psd + p
    return f, psd / len(files), np.array(g), np.array(w)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", type=Path, default=Path("runs/cat"))
    ap.add_argument("--root", type=Path, default=ROOT)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("YOSHIMURA et al. 2023, THEIR OWN FLIGHT SIMULATION -- 604 virtual flights")
    print("=" * 78)
    print(f"  source   {args.root}")
    print(f"  each     151 flights x 100 s at dt = 1/{int(1/DT)} s, four LES resolutions")

    out = {}
    print()
    print("-" * 78)
    print("1. THE RESPONSE DEPENDS ON THE LES RESOLUTION, AND SO DOES ITS FREQUENCY")
    print("-" * 78)
    print("    dx      n_z rms g          sigma_w      peak Hz    peak |dn| g")
    for res in RESOLUTIONS:
        f, psd, g, w = _load(res, args.root)
        out[res] = (f, psd, g, w)
        band = (f > 0.03) & (f < 2.0)
        pk = f[band][np.argmax(psd[band])]
        rms = g.std(axis=1)
        print(f"  {res:>5}   {rms.mean():.4f} [{rms.min():.4f},{rms.max():.4f}]  "
              f"{w.std():7.3f} m/s   {pk:7.4f}   {np.abs(g).max():8.3f}")
    f0, p0, g0, w0 = out[RESOLUTIONS[0]]
    f3, p3, g3, w3 = out[RESOLUTIONS[-1]]
    print()
    print(f"  500 m -> 35 m multiplies n_z rms by "
          f"{g3.std(axis=1).mean()/g0.std(axis=1).mean():.2f} and moves the")
    print(f"  spectral peak from 0.040 to 0.130 Hz, toward the aircraft's own "
          f"{YOSHIMURA_B787_HZ:.2f} Hz.")
    print("  A coarse domain does not just under-resolve the load -- it puts the")
    print("  response at the forcing's frequency instead of the airframe's.")

    print()
    print("-" * 78)
    print("2. WHAT IT IS NOT")
    print("-" * 78)
    lat = np.loadtxt(args.root / "results-1000_mand_35m.nc-2D" / "xlatlon_path-0001.txt",
                     delimiter=",", usecols=(6, 7, 8, 9))
    print(f"  beta, p, r, phi over a whole flight: "
          f"max |value| = {np.abs(lat).max():.3e}  -> strictly 2-D, longitudinal")
    print("  so it CANNOT validate the lateral capability built in session 24,")
    print("  and section 5's 'no held source records a lateral CAT response' stands.")
    cor = np.loadtxt(args.root / "results-1000_mand_35m.nc-2D" / "x_cor_path-0001.txt",
                     delimiter=",")
    print(f"  altitude {cor[:,4].min():.0f}..{cor[:,4].max():.0f} m "
          f"(span {cor[:,4].max()-cor[:,4].min():.0f} m over 100 s)")
    print(f"  -> it HOLDS its condition. cat_spectra.py's upper-sigma ensemble drifts")
    print(f"     703 m and 13.1% in airspeed (ASSUMPTIONS.md E11). E11 is a")
    print(f"     consequence of fixed controls, not of the measurement.")

    print()
    print("-" * 78)
    print("3. THE EXCEEDANCE CURVE, WHICH IS THE COMPARABLE STATISTIC")
    print("-" * 78)
    secs = g3.shape[0] * g3.shape[1] * DT
    print(f"  finest domain, {g3.shape[0]} flights, {secs/60:.0f} min of record")
    print("     level g     up/s")
    for lv in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
        n = sum(((g3[i][:-1] < lv) & (g3[i][1:] >= lv)).sum() for i in range(len(g3)))
        print(f"     {lv:6.2f}   {n/secs:8.4f}")
    print()
    print("  This is the same statistic cat_spectra.py limb C produces and could not")
    print("  compare against anything. It is not a PUBLISHED curve and it is a")
    print("  different aircraft at a different condition -- so it supports a")
    print("  cross-code comparison of SHAPE, the way JSBSim serves the vortex work,")
    print("  and not a validation of absolute load.")

    _figure(args.outdir / "12-yoshimura-flightsim.png", out)
    print(f"\n  figure -> {args.outdir / '12-yoshimura-flightsim.png'}")


def _figure(path, out):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 5))
    cols = [PALETTE["muted"], "#8FA8B0", PALETTE["wind"], PALETTE["model"]]
    for (res, (f, psd, g, w)), c in zip(out.items(), cols):
        m = (f > 0.02) & (f < 3)
        ax.loglog(f[m], psd[m], lw=1.5, color=c,
                  label=f"dx = {res}   rms {g.std(axis=1).mean():.3f} g")
    ax.axvline(YOSHIMURA_B787_HZ, color=PALETTE["reference"], ls="--", lw=1.2)
    ax.annotate("their aircraft's\nnatural frequency, 0.14 Hz",
                xy=(YOSHIMURA_B787_HZ, ax.get_ylim()[1] * .02), fontsize=8,
                color=PALETTE["reference"], ha="left", xytext=(4, 0),
                textcoords="offset points")
    ax.set_xlabel("frequency, Hz", fontsize=9)
    ax.set_ylabel("$n_z$ PSD, g$^2$/Hz", fontsize=9)
    ax.set_title("A. Refining the LES moves the response onto the airframe",
                 fontsize=10.5, loc="left")
    ax.legend(fontsize=8)

    secs = None
    for (res, (f, psd, g, w)), c in zip(out.items(), cols):
        secs = g.shape[0] * g.shape[1] * DT
        lv = np.arange(0.02, 0.9, 0.02)
        r = [sum(((g[i][:-1] < v) & (g[i][1:] >= v)).sum() for i in range(len(g))) / secs
             for v in lv]
        r = np.array(r)
        k = r > 0
        bx.semilogy(lv[k], r[k], lw=1.5, color=c, label=f"dx = {res}")
    bx.set_xlabel("load increment $|\\Delta n|$, g", fontsize=9)
    bx.set_ylabel("upcrossings per second", fontsize=9)
    bx.set_title("B. and lengthens the exceedance tail by an order of magnitude",
                 fontsize=10.5, loc="left")
    bx.legend(fontsize=8)
    for a in (ax, bx):
        a.grid(True, color=PALETTE["grid"], lw=.6, which="both", alpha=.7)
        a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.suptitle("Yoshimura's own 151-flight ensembles, at four LES resolutions "
                 "-- 2-D, longitudinal, 148.8 m/s at 3,000 m",
                 fontsize=11, x=.005, ha="left")
    fig.tight_layout(rect=[0, 0, 1, .95])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
