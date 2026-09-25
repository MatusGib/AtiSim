# Changelog

This file gives the changes in each version of AtiSim. The format is
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The version numbers use
[Semantic Versioning](https://semver.org/).

## [1.2.0] — unreleased

### Added

- `atisim.gust`: the linear transfer function from a vertical gust to the load factor, the
  random-process identity, Sears' function for the gust lag, and the Pratt–Walker discrete gust
  formula. `gust.measure_gust_transfer` flies a gust sinusoid and reads the transfer function
  from the run.
- `atisim.insitu`: the 5 s running-σ reductions of NASA/TM-2012-217337 (TPAWS), for a
  comparison of a simulated record with measured encounters.
- `atisim/data/tpaws_tm2012_217337_table1.csv`: the 53 measured B-757 encounters of TPAWS
  Table 1.
- New fields in `atisim.wind`: `von_karman_vertical_field`, `one_minus_cosine_gust`,
  `sinusoidal_vertical_field`, and `gaussian_vertical_field` as a control. The rotational gust
  spectra of MIL-F-8785C §3.7.5: `dryden_p_spectrum`, `dryden_q_spectrum` and
  `dryden_r_spectrum`.
- `vortex_viz.fly`, `fly_in_moving_air` and `fly_from_state` accept `stage_sampled` and
  `gust_lag`. `fly_in_moving_air` also accepts `wind_model`, and `fly_mehta` accepts
  `stage_sampled`, `gust_lag` and `tail_arm`.
- An optional gust lag, `gust_lag=True`, in `integrate.step`, the rollouts and the
  `vortex_viz` fly functions. It applies Jones's approximation to the Küssner function to the
  vertical gust, and `gust.gust_transfer(gust_lag=True)` gives its linear response. The flown
  response agrees with the linear response to 0.01% in amplitude and 0.001° in phase, from
  0.035 Hz to 2.5 Hz. The time step must be less than `wind.kussner_max_dt`. The default is
  `False`.
- A wing–tail gust delay. `airframe.stations(ac, n_lon=2, tail_arm=l)` gives a CG and a tail
  station, and `wind.sampled_field_model` on them makes the pitching gust the difference between
  the two. `airframe.JSBSIM_HTAILARM_FT` gives the tail arm of the 737 and of the JSBSim 747.
  `gust.gust_transfer(tail_arm=l)` gives the linear response, and the flown response agrees with
  it to 0.004% and 0.001°. The default is the point gradient, as before.
- User Manual procedure 4.5: fly with the gust lag and the wing–tail delay.

### Changed

- The documentation has three manuals: the User Manual, Physics and Assumptions, and the
  Development Manual. The structure is that of the JSBSim Reference Manual. The text uses
  ASD-STE100 Simplified Technical English.
- The automated trace of CR-2144 is now in the package, at `atisim/data/cr2144_trace/`. Thus
  `cr2144_mach.crosscheck` reads only data in the package.
- Physics and Assumptions gives the validation against 53 measured B-757 encounters (section
  10.4). In random turbulence, the peak loads of the model are approximately 20% too small
  relative to their rms. The gust response gain is not validated against the 757.
- New assumptions C12 (no gust lag), E13 (no rolling gust, and no pitching-gust roll-off) and
  E14 (the random field is not Gaussian), each with its measured effect.
- **The wind is sampled at each RK4 stage by default** (assumption E4) for a field that is a function
  of position only. Before, the wind was constant during each time step, and the method was first
  order in a field that changes in space. A random filter model still keeps its wind constant during
  the step. Set `stage_sampled=False` to get a version 1.1 result. These values change:

  | Value | 1.1 | 1.2 |
  |---|---|---|
  | Hannibal peak-to-peak load, 747 on the identified path | 1.8955 g (70.2%) | 1.8905 g (70.0%) |
  | Fig. 8 vortex point, pitch change | 1.8397° | 1.8258° |
  | Fig. 8 vortex point, load change | −1.1963 g | −1.1930 g |

- Physics and Assumptions gives what the two optional corrections do, and their limits. With both,
  the Hannibal load is 65.7% of the recorded value. The shape of the vortex core has a larger
  effect: a smooth Lamb–Oseen core gives 76.9%. With von Kármán turbulence and both corrections,
  the peak loads are still approximately 20% too small relative to their rms.
- The test suite runs in approximately 15 minutes, from 45. Approximately fifteen tests that
  calculated one point at a time now make one vectorised call, with the same values. One test
  that was a copy of another is removed.

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

[1.2.0]: https://github.com/MatusGib/AtiSim/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/MatusGib/AtiSim/releases/tag/v1.1.0
[1.0.0]: https://github.com/MatusGib/AtiSim/releases/tag/v1.0.0
