# Reference documents — what they are, and where to get them

Every number in AtiSim carries the table it came from (`docs/DEVELOPMENT.md` rule 2, and
`atisim/provenance.py`). This file is the index of the documents those tables live in: what
each one is, what the project reads from it, its md5 so you can confirm you have the same
copy, and **where to fetch it**.

## Why some of these are not in the repository

`.gitignore` has always said this of `refs/`:

> Third-party reference PDFs … deliberately kept out of history: they are re-downloadable and
> mostly not ours to redistribute.

That reasoning is right and `Reference_papers/` did not follow it. **Session 32 adjudicated all
24 documents against their own printed copyright statements** and split them in two.

**Documents that stay in the repository** are US Government works, or carry the Title 17
statement that no US copyright is asserted, or say in terms that they are not subject to US
copyright protection. Redistributing those is permitted, and two of them are what the
digitisation scripts actually read, so a fresh clone can reproduce those results.

**Documents that were removed** are held by AIAA, the American Meteorological Society or
Wiley/AGU, or could not be established. They stay on the maintainer's disk and are listed here
with a permanent locator. **Nothing about the project's results depends on having the PDF
rather than the citation** — every result is a `PROJECT.md` §4 row naming the table it came
from.

Scripts that read a PDF all take a `--pdf`-style path argument for exactly this reason
(`docs/DEVELOPMENT.md` rule 5), so they work against a copy you fetch yourself, wherever you put it.

---

## In the repository — verified redistributable

| Document | md5 | Licence basis | What the project reads |
|---|---|---|---|
| `19890016606.pdf` — NASA TM-102186, Wingrove & Bach, *Severe turbulence and maneuvering from airline flight records* | `51dc842f…927c` | NASA, US Government work | **Fig. 6** the recorded g trace and vertical wind; **Fig. 7** the horizontal wind; **Fig. 8** the three-aircraft ordering. Read by `digitise_tm102186_fig6.py` and `digitise_hannibal_horizontal_wind.py` |
| `parks-1985-identification-of-vortex-induced-clear-air-turbulence-JA22-2.pdf` — Parks et al., *J. Aircraft* **22**(2) 124–129, DOI 10.2514/3.45095 | `43e8bc0e…ed8` | Printed: "U.S. Government and therefore is in the public domain" | r₀ = 600 ft, V₀ = 85 ft/s, spacing 3500 ft (`ASSUMPTIONS.md` E12); **Fig. 6** the DC-10's altitude, pitch and true airspeed. Read by `digitise_parks_fig6_altitude.py` |
| `mehta-2012-modeling-clear-air-turbulence-with-vortices-using-parameter-identification-techniques.pdf` — Mehta, 1987 (copyright 1986) | `5a2f2409…521e` | Printed: "no copyright is asserted in the United States under Title 17, U.S. Code" | The five-vortex Hannibal field; the 2-vortex fit, p. 29; Eq. (A3)'s cost, which becomes the wind residual in `cat_uncertainty.py` |
| `bach-parks-1987-angle-of-attack-estimation-JA24-11.pdf` — Bach & Parks, *J. Aircraft* **24**(11) | `d8ee89b8…2858` | Printed: "No copyright is asserted in the United States under Title 17, U.S. Code" | The error budget: Eq. (2) shows `m` and `S` enter only as `m/S`; Eq. (4) gives ~0.05° of α per 1% of `C_L` |
| `AFFDL-TR-70-101-Ashburn-Waco-Melvin-1970-HICAT-AD878415.pdf` — HICAT, measured RMS gust exceedances from U-2 flights | `79935dc8…6e5b` | Printed: "Approved for public release, distribution unlimited". USAF | The published exceedance data. **Band mismatch is the catch**: 45,000–70,000 ft against this project's 33,000–41,000 |
| `Ger/schultz-2012-multiple-vortex-ring-model-of-the-dfw-microburst.pdf` | `26631f9d…2fcf` | Printed: "and is not subject to copyright protection in the United States" | Microburst vortex-ring structure |
| `Xiao_ger/19910009769.pdf` — NASA, NTRS 19910009769 | `0f817b5d…6408` | NASA, US Government work | Background reading |

## Removed from the repository — fetch these yourself

| Document | md5 | Held by | Where to get it |
|---|---|---|---|
| Yoshimura et al. 2023, *Clear air turbulence resolved by numerical weather prediction model*, **GRL** 50(12) | `ccf79db9…c65f` | **AGU/Wiley, CC BY-NonCommercial.** The NC clause, and the institutional-access watermark on the maintainer's copy, both rule out redistribution | DOI [10.1029/2022GL101286](https://doi.org/10.1029/2022GL101286). The **dataset** is separately CC BY 4.0: figshare 21152203 |
| `measurement-integrated-simulation-of-clear-air-turbulence-using-a-four-dimensional-variational-method` | `67fae15b…d2e` | **AIAA, 2007.** Printed: "All rights reserved" | Search AIAA ARC by title |
| `apme-JAMC-D-21-0071.1.pdf`, *J. Appl. Meteor. Climatol.* | `50d33838…69f5` | **© 2022 American Meteorological Society** | DOI 10.1175/JAMC-D-21-0071.1 |
| `apme-JAMC-D-21-0071.1 (1).pdf` — **byte-for-byte a second copy of the above** under a different name; 4.1 MB of duplicate | `18cb1d90…c2e9` | as above | as above |
| `loving-2012-clear-air-turbulence-(cat)-measurement-for-structural-design-criteria.pdf` | `2d0de052…6c2c` | **AIAA**, no Title 17 exemption printed | Search AIAA ARC by title |
| `1520-0493_1989_117_1103_tuodii_2_0_co_2.pdf`, *Mon. Wea. Rev.* **117** 1103 | `a6bf6e1f…d688` | **American Meteorological Society** | DOI prefix 10.1175, *Mon. Wea. Rev.* 117(5) |
| `Xiao_ger/dynamics_of_rigid_aircraft_in_a_nonstationary_atmosphere.pdf` | `2001f303…8484` | **Not established.** Excluded on that basis rather than on a finding — rule 2 says flag, never invent, and that applies to a licence as much as to a coefficient | — |
| `Flight_Dynamics_-_Second_Edition.pdf` — Stengel, 42.8 MB | `fffdb51b…ea4a` | **Textbook, copyrighted.** Already `.gitignore`d before this audit | Purchase |

## Never in the repository — on disk, listed for completeness

These are US Government works and could be redistributed, but they are large, permanently
available from NTRS, and **nothing is digitised from them yet** (`PROJECT.md` §7). The
locator is worth more than 31 MB in a clone.

| Document | md5 | NTRS ID | Status in the project |
|---|---|---|---|
| NASA CR-3677, Shollenberger et al. 1983, *Results of winglet development studies for DC-10 derivatives* | `a984a78c…b160c0` | 19850002628 | **PARTLY closes the DC-10 acquisition** (§7). Has Fig. 23 tail-off C_L vs α at M 0.82, Fig. 5 C_mCL vs Mach, Figs. 7–9 C_m vs α, Figs. 10/14/17 C_lβ/C_nβ/C_Yβ. **Lacks** C_mq, C_mα̇, the cruise drag polar, mass, inertias, CG and the reference S and c̄. Rigid model at model Reynolds number, no aeroelastic correction. **Not digitised; nothing flown** |
| NASA CR-3748, Taylor, *DC-10 winglet flight evaluation* | `f8fb0080…c8708` | 19870008261 | Not assessed |
| NASA TM-4745, Burcham 1996, *MD-11 augmented thrust-only flight control* | `c9458148…5d491` | 19960047451 | Not assessed |
| NASA TM-1998-206552, Burcham, *MD-11/B-747 engine thrust emergency control* | `8e3f1eb3…f9fc` | 19980148010 | Not assessed |
| Taylor 1978, *DC-10 derivatives: winglet and long-duct nacelle aero development* | `8524299a…1451` | 19780021106 | Not assessed |
| `CR-2144/` — NASA CR-2144 page scans, digitised points and a verification sheet, 3.6 MB | see `CR-2144/manifest.csv` | — | The 747's derivative source. **The digitised points that matter are tracked** in `atisim/data/cr2144_dig/` and `atisim/data/cr2144_p220_222_digitised.csv`, so the scans are not needed to reproduce anything. **The automated trace in `CR-2144/csv/` was scored against Table IX-4 in session 32** (`PROJECT.md` §4, "Two readings of CR-2144"): **do not use its `CL_M` or `Cm_adot`** — 4.9× and 9.5× worse than the hand reading, wrong curves — and prefer the hand reading for `Cm_M`. It is the better reading of `CL_alpha` only |

Also on disk and gitignored since before this audit: `refs/` — NASA CR-2144, NASA CR-114494
(Hanke & Nordwall, Boeing D6-30643), Caughey MAE 5070, Doyle et al. 2011, Oseguera & Bowles
1988, Proctor/Hinton/Bowles 2000, and MIL-F-8785C.
