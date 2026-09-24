# Changelog

This file gives the changes in each version of AtiSim. The format is
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The version numbers use
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- The documentation has three manuals: the User Manual, Physics and Assumptions, and the
  Development Manual. The structure is that of the JSBSim Reference Manual. The text uses
  ASD-STE100 Simplified Technical English.
- The automated trace of CR-2144 is now in the package, at `atisim/data/cr2144_trace/`. Thus
  `cr2144_mach.crosscheck` reads only data in the package.

### Fixed

- The version of the package was 0.1.0. It is now 1.1.0.
- An installation that is not editable now includes all the data that the package reads. This
  data includes the JSBSim reference files.

## [1.1.0] — 2026-09-21

### Changed

- The Boeing 747 has the pitching derivative for the rate of change of angle of attack, from
  CR-2144 Table IX-4 (`Cmadot` = −6.336). The error in short-period damping against Table IX-5
  decreases from −11.5% to +0.6%. The error in phugoid damping decreases from +2.8% to +1.4%.
- The table value of `Zwd` is not in the model. It gives a negative `CL_α̇`, but the downwash lag
  makes this derivative positive.
- The load in the Hannibal encounter decreases from 75.3% to 70.2% of the recorded
  peak-to-peak value. The new term damps the response that the encounter causes.

### Fixed

- The autopilot does not jump when you engage it during a pitch rate. The pitch-rate term in
  `autopilot.engage` had the incorrect sign.
- A uniform wind does not change the attitude. The rate of change of angle of attack now
  includes the transport term of the wind in body axes.

## [1.0.0] — 2026-09-20

The first public version.

### Flight dynamics

- A rigid-body model with six degrees of freedom, quaternion attitude and fixed-step RK4
  integration. JAX compiles the rollouts, and they operate on ensembles with `vmap`. All
  calculations use 64-bit numbers.
- Air-relative aerodynamics. The wind changes only the relative velocity and the gust rates.
- A Newton trim for steady level flight, and the linear modes about the trim.
- Gravity that changes with altitude, and the International Standard Atmosphere.
- A cascaded PID autopilot, and manual flight with a cockpit display.

### Aircraft

- The Boeing 747, at cruise and in power approach, from NASA CR-2144. The model includes the
  speed derivatives and the thrust line.
- The Boeing 737, at cruise and in approach, from the JSBSim 737. Use it only for comparison
  with JSBSim.
- The Boeing 787 model from the code of Yoshimura et al., and the Piper Cherokee and the Cessna
  172 for manual flight.
- A category for each coefficient: sourced, derived, calibrated or declared
  (`atisim.provenance`). The test suite checks these categories.

### Wind and turbulence

- Rankine vortex arrays: the Hannibal fields of Parks et al. (1985) and Mehta (1987), as points
  and as lines in space.
- An updraft, a microburst from Oseguera and Bowles (1988), and a mountain lee wave from Doyle
  et al. (2011).
- Dryden turbulence in three components.
- Rolling loads that the model integrates across the span.

### Verification and validation

- Checks of the solver against exact mathematics.
- Linear modes against CR-2144 Tables IX-4 and IX-5, and against the example of Caughey.
- Comparisons with JSBSim for the 737, the 747 and vortex encounters.
- The Hannibal encounter against the recorded load, and the order of loads in TM-102186
  Figure 8.
- Response spectra and exceedance statistics of Dryden ensembles.
- Two notebooks that do the validation again.

### Tools

- Run files with provenance and check results, and a Dash analysis application.
- Command-line scripts for encounters, manual flight and reference data.
- A documentation site, and continuous integration for the tests, the notebooks and the
  documentation.

### Known problems

- The strip load path counts the rolling moment of the gust two times, when you set
  `vortex_viz.fly(strip=True)`. This affects only lateral results.
- The Earth is flat and does not turn.
- The lift model has no stall. Results above about 10° angle of attack are not valid.
- AtiSim does not predict absolute loads.

[Unreleased]: https://github.com/MatusGib/AtiSim/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/MatusGib/AtiSim/releases/tag/v1.1.0
[1.0.0]: https://github.com/MatusGib/AtiSim/releases/tag/v1.0.0
