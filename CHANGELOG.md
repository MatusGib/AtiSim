# Changelog

AtiSim was developed between August and September 2026 over 32 working sessions. This file
lists what the release contains, grouped by capability. The session-by-session account — what
was measured, what was wrong, and what was corrected — is kept in full in
[`docs/PROJECT.md`](docs/PROJECT.md) §9, and every result named here is a §4 entry there with
the tolerance it was measured to.

The repository's history was rewritten before this release, to take publisher-held reference
documents out of every commit. No code, result or record entry changed;
[`docs/design/commit-map.txt`](docs/design/commit-map.txt) maps the commit ids quoted in the
record to the ones that exist now.

## [1.0.0] — 2026-09-20

### Flight dynamics core

- Six-degree-of-freedom rigid-body dynamics: quaternion attitude, fixed-step fourth-order
  Runge–Kutta, `lax.scan` rollouts, `jit` and `vmap` over ensembles, float64 throughout.
- Air-relative aerodynamics: wind enters **only** through the relative velocity and the gust
  rates, so any wind field drops into the same integrator.
- Newton trim for steady level flight in angle of attack, elevator and throttle; linearisation
  about the trimmed state for modal analysis.
- Gravity varying with altitude, and the International Standard Atmosphere read on
  geopotential altitude.
- Prandtl–Glauert compressibility across the longitudinal lift-slope family, with Mach-scheduled
  control derivatives.
- A cascaded PID autopilot, and a manual flying interface with a cockpit display.

### Aircraft

- **Boeing 747**, cruise and power approach, from NASA CR-2144 — including the report's speed
  derivatives and a declared thrust line, which close the phugoid against Table IX-5.
- **Boeing 737**, cruise and approach, linearised from JSBSim's own 737 model. **Not a qualified
  source and not valid away from cruise**: it exists so AtiSim's solver can be checked against an
  independent engine fed the same coefficients, and no claim about a real 737 rests on it.
- **Boeing 787**, rebuilt from Yoshimura et al.'s own simulation code, for the LES comparison.
- Piper Cherokee and Cessna 172, for manual flying; outside the validated scope.
- Every coefficient carries a provenance category — sourced, derived, calibrated or declared —
  checked by the test suite (`atisim/provenance.py`).

### Wind and turbulence

- Rankine vortex arrays: Parks et al. (1985) and Mehta's (1987) five-core field identified from
  the Hannibal, Missouri DC-10 encounter, as point fields and as line vortices in three
  dimensions, with the along-track wind and the replay of the field on its identified path.
- Updraft, Oseguera–Bowles microburst, and a Doyle et al. mountain lee wave.
- Dryden continuous turbulence, vertical, lateral and longitudinal, for Monte Carlo ensembles.
- Strip-integrated rolling loads across the span (see *Known issues*).

### Validation and verification

- Solver verification against exact mathematics: integration order, conservation, and a
  closed-form torque-free rotation.
- Modes against CR-2144 Tables IX-4 and IX-5, and against Caughey's published worked example.
- Cross-code comparison against JSBSim for the 737 and the 747, including the vortex
  encounters.
- The Hannibal encounter flown against the recorded load, with an uncertainty band; TM-102186
  Fig. 8's three-aircraft load ordering reproduced; response spectra and load-exceedance
  distributions over ensembles.
- A comparison against a published large-eddy simulation of clear-air turbulence, with the
  matched aircraft.
- Digitisations of TM-102186 Figs. 6 and 7, Parks et al. Fig. 6, MIL-F-8785C Fig. 7 and CR-2144
  pp. 220–222, each checked against the document's own printed values — the last one also
  against an independent trace of the same pages.
- A sensitivity study that differentiates the model in its own coefficients.
- Sealed predictions, committed before the run that decides them (`atisim/predictions.py`).
- Two executed notebooks: `notebooks/validation-ladder.ipynb` re-runs the evidence behind the
  validation claim, from hand-derived checks to the recorded encounter and the published
  orderings; `notebooks/solver-validation.ipynb` verifies the solver.

### Analysis and tooling

- Run artifacts (Parquet series with provenance and check results) and a Dash analysis app
  with a shared time cursor across every panel.
- 44 scripts, each documented in [`docs/PROJECT.md`](docs/PROJECT.md) §10, and a Sphinx
  documentation site built from the same record.
- Continuous integration on a clean Linux runner: the full test suite and both notebooks on
  every pull request, and the documentation site built with warnings treated as errors.

### Known issues

- **The strip load path counts the gust's rolling moment twice** when enabled
  (`vortex_viz.fly(strip=True)`). Lateral results only; no longitudinal result depends on it.
  `docs/PROJECT.md` §6(h).
- **The Earth is flat and non-rotating** (`ASSUMPTIONS.md` A1). The rotating WGS-84 model is
  complete on the `wgs84-earth` branch and not yet merged.
- **No stall.** The lift model is linear in angle of attack; results past about 10° are outside
  the model.
- **No absolute load prediction.** The model is validated for comparison and mechanism, not for
  a load value. The documentation's *Validation* page states the claim and its envelope.

### Parked for a later release

- The rotating WGS-84 Earth (`wgs84-earth` branch), with its rebase cost mapped.
- Table IX-4's α̇ pitching derivative on the 747, reviewed and ready.
- The repair of the strip path's roll accounting.
- The full list, with the route for each, is the status table at the head of
  `docs/PROJECT.md` §5.
