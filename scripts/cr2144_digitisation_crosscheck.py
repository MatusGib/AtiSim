"""CR-2144 pp. 220-222 read twice, compared: the hand reading the model uses
against an independent automated trace of the same pages.

Settles the sealed prediction `cr2144_two_readings_agree_on_the_good_panels`
(atisim/predictions.py). That seal was committed together with this script and
before `atisim.cr2144_mach.crosscheck` existed, so no run of it can precede the
seal.

Run from the repository root:
    python scripts/cr2144_digitisation_crosscheck.py

Source: NASA CR-2144, printed pp. 220-222 (PDF pp. 225-227). Reads neither the
PDF nor the scans; both inputs are tracked derived data --
atisim/data/cr2144_p220_222_digitised.csv and atisim/data/cr2144_trace/.
"""
import math

import numpy as np

from atisim import cr2144_mach

# Reference_papers/CR-2144/README.md, "Per-panel confidence"
CONFIDENCE = {"cl_alpha": "good", "cd_alpha": "good", "cm_alpha": "good",
              "cm_m": "good", "cd_m": "fair", "cl_m": "poor",
              "cm_alpha_dot": "poor"}


def adjudicate():
    """Where the two readings disagree, which one is off? Both against the
    value Table IX-4 implies at each circled flight condition, through Appendix
    A -- `cr2144_mach.backsolve`, the reference session 30 checked the hand
    reading against. Both readings are read the same way: linear interpolation
    of the curve at the condition's Mach, never extrapolated."""
    auto = cr2144_mach.automated_curves()
    print("\nBoth readings against Table IX-4 at the circled conditions (reading - table)")
    print(f"{'quantity':14}{'FC':>4}{'alt':>5}{'Mach':>7}{'table':>10}{'hand':>10}{'auto':>10}"
          f"{'hand-tab':>10}{'auto-tab':>10}")
    summary = {}
    for q in cr2144_mach.AUTO_NAME:
        for fc, (alt, mach, *_rest) in cr2144_mach.IX3.items():
            table = cr2144_mach.backsolve(fc)[q]
            hand = cr2144_mach.value(q, alt, mach)
            if (q, alt) not in auto:
                continue
            am, av = auto[(q, alt)]
            if not (am[0] <= mach <= am[-1]) or not math.isfinite(hand):
                continue
            a = float(np.interp(mach, am, av))
            summary.setdefault(q, []).append((hand - table, a - table))
            print(f"{q:14}{fc:>4}{alt:>5}{mach:>7.3f}{table:>+10.4f}{hand:>+10.4f}{a:>+10.4f}"
                  f"{hand - table:>+10.4f}{a - table:>+10.4f}")
    print(f"\n{'quantity':14}{'n':>4}{'RMS hand-table':>16}{'RMS auto-table':>16}  closer to the table")
    for q, rs in summary.items():
        h = np.sqrt(np.mean([x[0] ** 2 for x in rs]))
        a = np.sqrt(np.mean([x[1] ** 2 for x in rs]))
        print(f"{q:14}{len(rs):>4}{h:>16.4f}{a:>16.4f}  {'hand' if h < a else 'automated'}")


def main():
    print(f"{'quantity':14}{'alt':>5}{'conf':>6}{'n':>5}{'med %FS':>9}"
          f"{'max %FS':>9}{'med px':>8}{'bias %FS':>10}  sits on")
    for r in cr2144_mach.crosscheck():
        swap = "" if r.best_altitude == r.altitude else "   <-- altitude?"
        print(f"{r.quantity:14}{r.altitude:>5}{CONFIDENCE[r.quantity]:>6}{r.n:>5}"
              f"{r.median_pct_fs:>9.2f}{r.max_pct_fs:>9.2f}{r.median_px:>8.1f}"
              f"{r.bias_pct_fs:>+10.2f}  {r.best_altitude}{swap}")
    adjudicate()


if __name__ == "__main__":
    main()
