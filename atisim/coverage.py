"""What the application covers of the engine, as data a test can hold it to.

The owner's requirement is that the application runs every simulation the
engine can. A list in a document goes stale the day the engine gains a field;
this module is the list, and `tests/test_coverage.py` fails, naming the
function, when an engine entry point is neither reached by a run kind nor
listed here as not a run with the reason why.

The maps:

- `REACHED`: an engine function and the run kind or option that flies it.
- `NOT_A_RUN`: an engine function that is not a simulation by itself, and why.
- `STEP_OPTIONS`: each `integrate.step` option and the `RunSpec` field for it.
- `SIMULATIONS`: every engine function outside `atisim.wind` that runs a
  simulation or reduces one, and the analysis (`atisim.analyses`) or run kind
  that reaches it. `NOT_YET` lists the ones deliberately not reached, and why.
"""

# `atisim.wind` functions named *_wind, *_field, *_model or *_gust, and how a
# run reaches each one.
REACHED = {
    "vortex_wind": "run kind VortexArray (and SingleVortex, MehtaHannibal), Rankine core",
    "vortex_model": "run kind VortexArray: wind.field_model of the same field",
    "lamb_oseen_wind": "core profile lamb-oseen on VortexArray, SingleVortex, MehtaHannibal",
    "line_vortex_wind": "MehtaHannibal, vortex form line",
    "line_vortex_model": "MehtaHannibal, vortex form line: wind.field_model of it",
    "updraft_wind": "run kind UpdraftColumn",
    "updraft_model": "run kind UpdraftColumn",
    "lee_wave_wind": "run kind LeeWave",
    "lee_wave_model": "run kind LeeWave",
    "microburst_wind": "run kind Microburst",
    "microburst_model": "run kind Microburst",
    "sinusoidal_vertical_field": "run kind Sinusoid",
    "one_minus_cosine_gust": "run kind OneMinusCosine",
    "dryden_vertical_field": "run kind Dryden, components vertical; and the overlay",
    "dryden_field": "run kind Dryden, components three-axis",
    "von_karman_vertical_field": "run kind VonKarman; and the overlay",
    "gaussian_vertical_field": "run kind GaussianDryden; and the overlay",
    "field_model": "every run: the default wind model",
    "sampled_field_model": "solver option wing_tail",
    "lagged_wind": "solver option gust_lag (integrate.step applies it)",
    "mehta_unmodelled_wind": "preset mehta-hannibal-turbulent: the overlay's sigma_w",
}

NOT_A_RUN = {
    "zero_wind": "the still-air wind model: run kind none flies it",
}

# `integrate.step` keyword options, and the `RunSpec` field that sets each.
STEP_OPTIONS = {
    "stage_sampled": "stage_sampled",
    "gust_lag": "gust_lag",
    "load_model": "strip",
    "wind_model": "wing_tail",
}

# Engine functions outside `atisim.wind` that fly or reduce a simulation, as
# "module.function", and what reaches each. An analysis is named by its key in
# `atisim.analyses.ANALYSES`; the test checks both ends exist.
SIMULATIONS = {
    "vortex_viz.fly": "every run kind",
    "vortex_viz.fly_in_moving_air": "run kind MehtaHannibal",
    "vortex_viz.fly_from_state": "every run kind, through vortex_viz.fly; analysis cross-code",
    "vortex_viz.manoeuvre": "run kind manoeuvre",
    "vortex_viz.elevator_for_load": "run kind manoeuvre",
    "autopilot.closed_loop_rollout": "analyses step-response, closed-loop",
    "autopilot.engage": "analyses step-response, closed-loop",
    "gust.measure_gust_transfer": "analysis gust-transfer",
    "gust.gust_transfer": "analysis gust-transfer",
    "gust.pratt_walker": "analysis pratt-walker",
    "trim.minimum_drag_speed": "analysis trim",
    "validation.longitudinal_modes": "analyses modes, coefficient-sweep",
    "validation.lateral_modes": "analyses modes, coefficient-sweep",
    "validation.sweep": "analysis coefficient-sweep",
    "sensitivity.mode_sensitivity": "analysis mode-sensitivity",
    "sensitivity.load_elasticities": "analysis load-sensitivity",
    "verification.newton_residual_history": "analysis trim",
    "verification.fitted_order": "analysis convergence",
    "verification.oscillator_refinement": "analysis verification",
    "verification.fixed_control_refinement": "analysis verification",
    "verification.free_fall_through_a_swinging_wind": "analysis verification",
    "verification.torque_free_omega": "analysis verification",
    "insitu.encounter_peak_factors": "analysis ensemble",
    "response.spectrum": "analysis ensemble",
    "response.exceedance": "analysis ensemble",
    "jsbsim_vortex_ref.load": "analysis cross-code",
}

NOT_YET = {
    "vortex_viz.fly_mehta": "the scripts' wrapper; run kind MehtaHannibal flies the same "
                            "field through run.fly",
    "integrate.batched_rollout": "an ensemble flies its members one after another through "
                                 "run.fly, so each is an ordinary run with its own checks; "
                                 "vmapping them is a speed-up, not a new simulation",
    "panel.run_live": "live flying in the browser: plan phase F, optional (decision D6)",
}
