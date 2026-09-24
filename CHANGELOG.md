# Changelog

All notable changes to AtiSim are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- The documentation is rewritten around using the tool. It now has a user guide with worked
  examples, a scripts reference, and a page setting out the model and its assumptions.
- The automated CR-2144 trace moved into the package, at `atisim/data/cr2144_trace/`, so
  `cr2144_mach.crosscheck` no longer reads outside the package.

### Fixed

- The package version was still 0.1.0. It now reads 1.1.0.
- A regular (non-editable) install now includes all the data the package reads at runtime,
  including the JSBSim reference XML.

## [1.1.0] — 2026-09-21

### Changed

- The Boeing 747 now carries the angle-of-attack-rate pitching derivative from CR-2144 Table
  IX-4 (`Cmadot` = −6.336, converted from the tabulated `Mwd`). Against Table IX-5, the error in
  short-period damping falls from −11.5% to +0.6%, and the error in phugoid damping from +2.8% to
  +1.4%. The table's `Zwd` is not used: it converts to a negative `CL_α̇`, and downwash lag makes
  that derivative positive.
- As a result, the Hannibal encounter load falls from 75.3% to **70.2%** of the recorded
  peak-to-peak value, because the new term damps the response the encounter excites.

### Fixed

- The autopilot no longer lurches when it is engaged during a pitch rate. `autopilot.engage` had
  the wrong sign on its pitch-rate seed, which caused a transient of `2·q_d·q`.
- A uniform wind no longer changes the attitude. `wind.gust_alphadot` had dropped the transport
  term of `d(wind_body)/dt`, so a wind that was constant in NED still produced an α̇ as the
  aircraft rotated.

## [1.0.0] — 2026-09-20

The first public release.

### Flight dynamics

- Six-degree-of-freedom rigid-body dynamics with quaternion attitude, fixed-step RK4, `lax.scan`
  rollouts, `jit` and `vmap` over ensembles, and float64 throughout.
- Air-relative aerodynamics. Wind enters only through the relative velocity and the gust rates.
- Newton trim for steady level flight, and linearisation about the trim point for modal
  analysis.
- Gravity that varies with altitude, and the International Standard Atmosphere on geopotential
  altitude.
- Prandtl–Glauert compressibility, with Mach-scheduled control derivatives.
- A cascaded PID autopilot, and a manual flying interface with a cockpit display.

### Aircraft

- Boeing 747, in cruise and power approach, from NASA CR-2144, including its speed
  derivatives and thrust line.
- Boeing 737, in cruise and approach, linearised from JSBSim's 737 model. It exists for
  cross-code verification and is not valid away from cruise.
- The Boeing 787 model from Yoshimura et al.'s simulation code, and the Piper Cherokee and
  Cessna 172 for manual flying.
- Every coefficient is classed as sourced, derived, calibrated or declared
  (`atisim.provenance`), and the test suite checks those classes.

### Wind and turbulence

- Rankine vortex arrays: the fields from Parks et al. (1985) and Mehta (1987) for the Hannibal
  encounter, as point fields and as line vortices, with a replay of the field along its
  identified path.
- Updraft, Oseguera–Bowles microburst and Doyle et al. mountain lee wave.
- Dryden continuous turbulence in all three components.
- Rolling loads integrated strip by strip across the span.

### Validation and verification

- Solver verification against exact mathematics: order of accuracy, conservation, and
  torque-free rotation in closed form.
- Linear modes against CR-2144 Tables IX-4 and IX-5, and against Caughey's worked example.
- Cross-code comparison with JSBSim for the 737, the 747 and vortex encounters.
- The Hannibal encounter flown against the recorded load, and the three-aircraft ordering in
  TM-102186 Fig. 8 reproduced.
- Response spectra and load-exceedance statistics over Dryden ensembles.
- Two executed notebooks: `validation-ladder.ipynb` and `solver-validation.ipynb`.

### Tools

- Run artifacts (Parquet time series with provenance and check results) and a Dash analysis
  app.
- Command-line scripts for encounters, flying by hand and reference-data generation.
- A Sphinx documentation site, and CI that runs the tests, the notebooks and the docs build.

### Known issues

- When the strip load path is enabled (`vortex_viz.fly(strip=True)`), it counts the gust's
  rolling moment twice. This affects lateral results only.
- The Earth is flat and non-rotating.
- The lift model has no stall, so results above about 10° angle of attack are outside the model.
- The model is validated for comparing encounters and explaining mechanisms, not for
  predicting absolute loads.

[Unreleased]: https://github.com/MatusGib/AtiSim/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/MatusGib/AtiSim/releases/tag/v1.1.0
[1.0.0]: https://github.com/MatusGib/AtiSim/releases/tag/v1.0.0
