# References

The numbers in AtiSim come from these documents. In `atisim/aircraft.py`, each aircraft
constant has a reference to its document, table and page. `atisim.provenance` gives the category
of each constant. The repository does not include the documents. A script that reads a document
takes its path as the argument `--pdf`.

## Documentation standards

- ASD-STE100, *Simplified Technical English*, Issue 9 (2025). The writing rules of this documentation.
- ISO/IEC/IEEE 26514:2022, *Design and development of information for users*. The information types of the User Manual.
- NASA-STD-7009, *Standard for Models and Simulations*. The content of the Physics and Assumptions manual.
- JSBSim Reference Manual (jsbsim-team.github.io/jsbsim-reference-manual). The structure of the manuals.

## Aircraft data

- Heffley, R. K. & Jewell, W. F. (1972). *Aircraft Handling Qualities Data.* NASA CR-2144.
  Section IX: Boeing 747 geometry, inertia, stability derivatives (Tables IX-3, IX-4), modes
  (Table IX-5) and Mach derivative charts (pp. 218–228).
- Hanke, C. R. & Nordwall, D. R. (1970). *The Simulation of a Jumbo Jet Transport Aircraft,
  Vol. II: Modeling Data.* NASA CR-114494 (Boeing D6-30643). The 747 thrust line.
- Caughey, D. A. *Introduction to Aircraft Stability and Control.* Cornell University MAE
  5070 course notes, Ch. 5. An independent implementation of the CR-2144 power-approach case.
- JSBSim (github.com/JSBSim-Team/jsbsim). The 737 and 747 aircraft models, and the
  cross-code reference engine.
- McCormick, B. W., Roskam, J. with USAF DATCOM, Nelson, R. C., Etkin, B., and McRuer, D. The
  derivative sets of the light aircraft (Cherokee, Cessna 172, Navion).

## Turbulence encounters

- Parks, E. K., Wingrove, R. C., Bach, R. E. & Mehta, R. S. (1985). Identification of
  vortex-induced clear air turbulence using airline flight records. *J. Aircraft* 22(2),
  124–129. DOI [10.2514/3.45095](https://doi.org/10.2514/3.45095). The Rankine vortex-array
  model and the Hannibal and Morton cases.
- Mehta, R. S. (1987). Modeling clear-air turbulence with vortices using
  parameter-identification techniques. *J. Guidance, Control, and Dynamics* 10(1), 27–31. The
  five-vortex Hannibal field.
- Wingrove, R. C., Bach, R. E. & Schultz, T. A. (1989). *Analysis of Severe Atmospheric
  Disturbances from Airline Flight Records.* NASA TM-102186 (NTRS 19890016606). The recorded
  Hannibal load (Fig. 6), the horizontal wind (Fig. 7) and the three-aircraft comparison
  (Fig. 8).
- Wingrove, R. C. & Bach, R. E. (1994). Severe turbulence and maneuvering from airline
  flight records. *J. Aircraft* 31(4), 753–760. Updraft magnitudes and the
  vortex–updraft–manoeuvre discriminator.
- Bach, R. E. & Parks, E. K. (1987). *J. Aircraft* 24(11), 789–792. Angle-of-attack
  estimation from flight records, and the error budget on the identified fields.
- Lester, P. F., Sen, O. & Bach, R. E. (1989). The use of DFDR information in the analysis
  of a turbulence incident over Greenland. *Mon. Wea. Rev.* 117, 1103–1107. The RMS error of a
  flight-recorder wind reconstruction.

## Wind and turbulence models

- Oseguera, R. M. & Bowles, R. L. (1988). *A Simple, Analytic 3-Dimensional Downburst Model
  Based on Boundary Layer Stagnation Flow.* NASA TM-100632. The microburst.
- Schultz, T. A. Multiple vortex-ring model of the DFW microburst. The microburst's
  vortex-ring structure.
- Proctor, F. H., Hinton, D. A. & Bowles, R. L. (2000). A windshear hazard index. *9th Conf.
  on Aviation, Range and Aerospace Meteorology*, paper 7.7, 482–487. The F-factor.
- Doyle, J. D., Jiang, Q., Smith, R. B. & Grubišić, V. (2011). Three-dimensional
  characteristics of stratospheric mountain waves during T-REX. *Mon. Wea. Rev.* 139, 3–23.
  DOI [10.1175/2010MWR3466.1](https://doi.org/10.1175/2010MWR3466.1). Lee-wave amplitudes.
- MIL-F-8785C (1980). *Military Specification: Flying Qualities of Piloted Airplanes.* The
  Dryden spectral forms (§3.7.1.2) and turbulence exceedance probability (Fig. 7).
- Misaka, T., Obayashi, S. & Endo, E. (2008). Measurement-integrated simulation of
  clear-air turbulence using a four-dimensional variational method. *J. Aircraft* 45(4),
  1217–1229. The RMS normal-load severity index.
- Yoshimura, R. et al. (2022). *J. Appl. Meteor. Climatol.* 61, 503–519. DOI
  [10.1175/JAMC-D-21-0071.1](https://doi.org/10.1175/JAMC-D-21-0071.1). A third CR-2144 747
  flight condition.
- Yoshimura, R. et al. (2023). Clear air turbulence resolved by numerical weather prediction
  model. *Geophys. Res. Lett.* 50(12). DOI
  [10.1029/2022GL101286](https://doi.org/10.1029/2022GL101286). The large-eddy simulation
  comparison, and the 787 model from the authors' own simulation code (dataset: figshare
  21152203).
- Ashburn, E. V., Waco, D. E. & Melvin, C. A. (1970). AFFDL-TR-70-101, the HICAT
  program (DTIC AD878415). Measured high-altitude gust exceedances.

## Flight dynamics and verification

- Stengel, R. F. *Flight Dynamics*, 2nd ed. Princeton University Press. The gust-rate signs
  (eqs. 3.4-48, 3.4-50 and 3.4-52), which AtiSim re-derives rather than transcribes.
- Roache, P. J. *Verification and Validation in Computational Science and Engineering*. Also
  AIAA G-077-1998, *Guide for the Verification and Validation of Computational Fluid Dynamics
  Simulations.* The verification/validation split.
