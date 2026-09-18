"""CR-2144 pp. 220-222 read twice, compared: session 30's hand reading against
the automated trace found untracked in session 32.

Settles the sealed prediction `cr2144_two_readings_agree_on_the_good_panels`
(atisim/predictions.py). That seal was committed together with this script and
before `atisim.cr2144_mach.crosscheck` existed, so no run of it can precede the
seal.

Run from the repository root:
    .venv/Scripts/python.exe scripts/cr2144_digitisation_crosscheck.py

Source: NASA CR-2144, printed pp. 220-222 (PDF pp. 225-227). Reads neither the
PDF nor the scans; both inputs are tracked derived data --
atisim/data/cr2144_p220_222_digitised.csv and Reference_papers/CR-2144/csv/.
"""
from atisim import cr2144_mach

# Reference_papers/CR-2144/README.md, "Per-panel confidence"
CONFIDENCE = {"cl_alpha": "good", "cd_alpha": "good", "cm_alpha": "good",
              "cm_m": "good", "cd_m": "fair", "cl_m": "poor",
              "cm_alpha_dot": "poor"}


def main():
    print(f"{'quantity':14}{'alt':>5}{'conf':>6}{'n':>5}{'med %FS':>9}"
          f"{'max %FS':>9}{'med px':>8}{'bias %FS':>10}  sits on")
    for r in cr2144_mach.crosscheck():
        swap = "" if r.best_altitude == r.altitude else "   <-- altitude?"
        print(f"{r.quantity:14}{r.altitude:>5}{CONFIDENCE[r.quantity]:>6}{r.n:>5}"
              f"{r.median_pct_fs:>9.2f}{r.max_pct_fs:>9.2f}{r.median_px:>8.1f}"
              f"{r.bias_pct_fs:>+10.2f}  {r.best_altitude}{swap}")


if __name__ == "__main__":
    main()
