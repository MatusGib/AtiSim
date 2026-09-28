# Changelog

This file gives the changes in each version of AtiSim. The format is
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The version numbers use
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [1.3.1] — 2026-09-28

### Added

- The application saves each test card flight as a run: `testcard-{point}-{aircraft}-{commit}`.
  `cockpit.save_run` writes it when the debrief opens. The run has no `spec.json`, because the
  controls are the inputs of the pilot. Engineering mode and the Lab open it as a run.
- After a flight, **Engineering** in the mode switch opens that flight in Results. The debrief
  and each row of the test card also give a link to the flight.
- The paper record: `atisim.records` finds what the source measured in the same encounter and
  puts it on the time axis of a run. A Hannibal run gets the recorded DC-10 load of Parks et al.
  1985 Fig. 6 and the measured band of TM-102186. The other vortex runs, the updraft and the
  manoeuvre get the load band of Wingrove and Bach Fig. 8. The record is moved in time so that
  its deepest downdraft is at the deepest downdraft of the run. This alignment is declared.
- The Results page, the Lab result card and the debrief of the test card show the paper record
  on the load factor. On the Results page, the **Paper record** switch shows or hides it.
- `panel.placed_field` and `panel.field_ahead_meta`: one placement of the field of a test point,
  for the flight and for its run file.
- `run.validate` gives more errors: a run of fewer than two steps or more than 1,000,000 steps, a
  time step longer than 0.5 s, a field that the time step cannot resolve, an airspeed of Mach 1
  or more, an altitude above 20,000 m, and a run name of more than 100 characters. It gives more
  warnings: a time step longer than 0.05 s, overlapping vortex cores, and a gust angle of attack
  past 10°.
- `studies.split_arguments`: it splits the arguments of a study as the shell of the computer
  does. `studies.INTERACTIVE` names the scripts that cannot run as a study.
- A banner on the Results page tells you when a run diverged.

### Changed

- Wingrove and Bach Fig. 8 in Results: the legend has one entry for each category, and each
  category has its own marker shape. Only the run on the screen has a label. The label of a run
  is its case, for example `wingrove-morton`, not its first word. The category comes from the
  wind field, so all the vortex runs are in the vortex category. A run of a different field is
  not on the figure.
- The ordering panel uses one run for each category. The run on the screen represents its
  category.
- The debrief and the Lab result card show the larger change from level flight first: the
  lowest load factor when it is further from +1 g than the peak.
- In still air, the debrief, the card and the Lab say "still air", not a turbulence band. The
  load then comes from the controls.
- The Validity row of the debrief agrees with the angle of attack row.
- The number fields of Setup, the Lab and the analyses accept `1e-3` and a decimal comma. A value
  that is not a number gives an error that names the field.
- The Lab compare table gives the wind values of the two runs.
- The Lab says so when a case or a result does not exist, or when A and B are the same flight.
- The trim error at a high altitude tells you to decrease the altitude. It does not give the top
  of the search as the minimum-drag speed.
- The core caveat of a Parks case changes with the core radius and the aircraft.

### Fixed

- The Progress tab of Setup stopped at "Write the artifact" when the status bar stopped the poll
  first. The poll now continues for 2 s after the last job.
- A failure to fly A at a different commit showed on the Setup page as a failure of its run.
- An analysis page showed the progress of a different analysis.
- **Re-fly at High** gives a link to the new run when it is written.
- Compare offered reports and run A in **Add a run to compare**.
- The `fly` script ran as a study and waited until its time limit.
- The arguments of a study lost the backslashes of a Windows path.
- The energy closure failed in each still-air run: the check divided by zero energy change.
- The hollow markers of Fig. 8 had the default plotly colors.
- In Results, the 2D cross-section and the load factor against the angle of attack were empty.
  Plotly drew them in a box of 37 px, before their height was set, and their color bars stopped
  the draw. The two graphs now have their height from the start.
- The start label of the Setup preview used a different point than the **Start** row.

## [1.3.0] — 2026-09-27

### Added

- The application (`atisim ui`) has three modes. A switch at the top of each page selects the
  mode: the test card, the Lab or engineering mode.
- The test card: the first page of the application. It gives four test points: calm air, the
  Hannibal and Morton vortex pairs, and a thunderstorm updraft. Each row shows the vertical wind
  ahead of the aircraft, calculated from the wind field. **FLY** opens a cockpit that flies the
  747 in the web browser. The debrief gives the peak load factor from each physics step, and the
  card keeps the result.
- `atisim.cockpit`: the test points, and a flight from the keys that a web page sends. It uses
  the loop of `scripts/fly.py` (`panel.LiveSim`) with no change. `panel.Keys` and
  `panel.warm_up` come from `panel.Panel`, so that the web page flies the same loop.
- The Lab: a mode between the test card and engineering mode. It flies each preset with a few
  changed values: the aircraft, the airspeed, the altitude, and the strength and size of the
  field. A result card gives the peak load factor, the turbulence severity, the checks and a
  recorder chart. The Lab also runs four analyses with their defaults, and it compares two
  results. `atisim.lab` gives its content.
- Engineering mode has an explorer on the left. It gives the presets, the analyses, the research
  scripts by topic and the results by kind, with a filter. The header shows the location of the
  page. The model tree of Setup is in the explorer.
- A page for each research script: its docstring, a run form and its earlier results. An index
  gives all the scripts in their topics. `studies.TOPICS`, `studies.by_topic` and
  `studies.about` give the topics and the docstrings.
- `checks.severity_band`: the band of the RMS normal load as one word. The cockpit and the Lab
  use it.
- `runs.RunRow.analysis`, `runs.RunRow.script` and `runs.RunRow.group`: the analysis that wrote a
  report, the script of a study, and the group of a result.
- `atisim.run`: a run as data. `RunSpec` and `WindSpec` round-trip through JSON. `PRESETS`
  gives the sourced cases and a blank run. `validate` gives the errors and the warnings of a
  spec. `fly` and `save` fly a spec and write its run file. Each value is Sourced or Declared,
  and a changed Sourced value becomes Declared.
- The `atisim` command: `atisim run`, `atisim presets`, `atisim list` and `atisim ui`.
- Setup: a model tree, a property grid with a Sourced or Declared marker on each value, a
  geometry preview, and a dock with Messages, Progress, Log, Checks and Script.
- `figures.field_preview`: the wind field and the planned flight path, before the flight. The
  view centers on the structure of the field, not on the path.
- The application flies each wind field of the engine. The run kinds are a vortex array, a
  single vortex core, the Mehta Hannibal field, an updraft, a lee wave, a microburst, a gust
  sinusoid, the 1 − cosine gust, and Dryden, von Kármán and Gaussian turbulence. A vortex can
  have a Lamb–Oseen core. Any run can add turbulence to its field (`RunSpec.overlay`). The Solver
  item sets the stage sampling, the gust lag and the wing–tail delay. `atisim.coverage` and
  `test_coverage.py` fail when an engine function has no run kind or analysis.
- Analyses: `atisim.analyses`, the commands `atisim analyses` and `atisim analyse`, and a page for
  each analysis. The 13 analyses are seed ensembles, the autopilot, the trim, the modes and their
  sensitivities, a coefficient sweep, the gust transfer function, the discrete gust, a
  convergence study, the JSBSim cross-code check and the verification suite. Each writes a
  report (`atisim.analysis.report`) that Results shows. Each gives the numbers of the script that
  it replaces.
- High fidelity: `RunSpec.fidelity="high"` and `atisim run --fidelity high`. The run also writes
  `diagnostics.parquet`, with each term of each coefficient, the forces and moments by source,
  the derivatives and the energy at each sample. The flight does not change.
  `aero.coefficient_terms` gives the terms of `aero.coefficients`. Each run also writes
  `spec.json`.
- The Diagnostics view of a run: the channels, the coefficient, force and energy budgets, the
  check profiles, the step inspector and a detailed scene at the cursor.
- The Compare view: the values that differ between runs, and each channel with its difference.
  From run A, it flies one change, a convergence study, or the engine of a different commit
  (`atisim.analysis.commits`).
- Engine-development mode, `atisim ui --dev`. The runs fly in a worker process, and
  **Reload engine** loads a changed engine. The JAX compilation cache makes the first run after a
  reload faster.
- Studies: `atisim.studies`, `atisim studies` and `atisim study NAME`. Each script in `scripts/`
  runs with no change, and Results shows its log and the files that it wrote.
- User Manual sections 4.10.6, 4.17 and 4.18: the explorer, the test card and its cockpit, and
  the Lab. Development Manual section 6.7: the procedure to update the application after new
  tests, validation or cases.

### Changed

- The README gives only the installation and the three modes of the application. The User
  Manual gives the procedures.
- `atisim ui` opens on the test card. It compiles the flight of each test point in the
  background, so that the first flight starts quickly.
- `scripts/vortex.py --artifacts` uses `atisim.run`. The run files are the same as before,
  except for the time of the run.
- `python -m atisim.apps.sweep` opens the application on the Results page.
- `figures.field_3d` frames its scene. The box contains the flight path and the field, with the
  same scale on each axis, and the view is orthographic from the south-east. Before, the default
  camera showed a long path as one diagonal line, partly off the panel.
- The `ui` part also installs `dash-mantine-components` and `dash-iconify`.
- The application holds its icons (`atisim/apps/assets/icons.js`) and its fonts
  (`atisim/apps/assets/fonts/`). Thus it operates with no internet connection.
- The Development Manual tells you to write all text in Simplified Technical English: the
  manuals, the README, the changelog and each pull request.

### Fixed

- A study records the paths of the files that it keeps with forward slashes. Before, on Windows,
  the paths had backslashes, and the links to the files of a study did not operate.

## [1.2.0] — 2026-09-25

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

[Unreleased]: https://github.com/MatusGib/AtiSim/compare/v1.3.1...HEAD
[1.3.1]: https://github.com/MatusGib/AtiSim/compare/v1.3.0...v1.3.1
[1.3.0]: https://github.com/MatusGib/AtiSim/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/MatusGib/AtiSim/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/MatusGib/AtiSim/releases/tag/v1.1.0
[1.0.0]: https://github.com/MatusGib/AtiSim/releases/tag/v1.0.0
