"""The run API: specs, presets, validation, and the artifact they fly to.

Every validation rule is given a crafted spec it must fire on, and every preset
is a spec it must stay silent on -- except the presets where a rule is meant
to fire (the microburst, flown 1.2 km below the Cherokee's CRUISE altitude on
purpose, `scripts/microburst.py` says why; and the Mehta and Wingrove cases,
flown at the altitudes their papers identified the field at, not at CR-2144's
40,000 ft).
"""

import json

import pytest

import atisim  # noqa: F401
from atisim import run
from atisim.analysis import artifact

# (preset, rule field) pairs where the warning is the truth about the preset.
EXPECTED = {("microburst", "altitude_m"),
            ("mehta-hannibal", "altitude_m"), ("mehta-hannibal-turbulent", "altitude_m"),
            ("wingrove-hannibal", "altitude_m"), ("wingrove-cimarron", "altitude_m")}


@pytest.mark.parametrize("name", list(run.PRESETS))
def test_every_preset_round_trips_through_json(name):
    spec = run.PRESETS[name]
    back = run.RunSpec.from_json(spec.to_json())
    assert back == spec
    # And through a plain dict, which is what the UI's store holds.
    assert run.RunSpec.from_json(json.loads(spec.to_json())) == spec


def test_blank_round_trips_and_is_valid():
    assert run.RunSpec.from_json(run.BLANK.to_json()) == run.BLANK
    assert run.validate(run.BLANK) == []


@pytest.mark.parametrize("name", list(run.PRESETS))
def test_presets_are_silent_except_where_a_rule_is_meant_to_fire(name):
    issues = run.validate(run.PRESETS[name])
    assert not [i for i in issues if i.level == "error"], issues
    unexpected = [i for i in issues if (name, i.field) not in EXPECTED]
    assert unexpected == []


def test_the_microburst_preset_is_flagged_as_far_from_cruise():
    issues = run.validate(run.PRESETS["microburst"])
    assert [(i.level, i.field) for i in issues] == [("warning", "altitude_m")]
    assert "CRUISE" in issues[0].message


def _fields(spec, level):
    return {i.field for i in run.validate(spec) if i.level == level}


def test_a_light_aircraft_in_a_parks_case_is_not_a_sourced_result():
    spec = run.PRESETS["vortex-hannibal"]._replace(aircraft="cherokee")
    messages = [i.message for i in run.validate(spec) if i.field == "aircraft"]
    assert messages and "not a sourced result" in messages[0]
    # Also for the updraft and the manoeuvre, the other two Fig. 8 categories.
    for name in ("updraft", "manoeuvre"):
        assert "aircraft" in _fields(run.PRESETS[name]._replace(aircraft="cessna172"),
                                     "warning")
    # And not for the lee wave, which is not a Parks or Wingrove case.
    assert "aircraft" not in _fields(
        run.PRESETS["lee-wave"]._replace(aircraft="boeing737"), "warning")


def test_a_short_lead_in_warns_and_twelve_does_not():
    base = run.PRESETS["vortex-hannibal"]
    assert "lead_in" in _fields(base._replace(lead_in=8.0), "warning")
    assert "lead_in" not in _fields(base._replace(lead_in=12.0), "warning")


def test_a_flight_condition_far_from_cruise_warns():
    base = run.PRESETS["lee-wave"]
    assert "airspeed_mps" in _fields(base._replace(airspeed_mps=180.0), "warning")
    assert "altitude_m" in _fields(base._replace(altitude_m=6000.0), "warning")


def test_strip_loads_carry_the_loading_shape_caveat():
    spec = run.PRESETS["vortex-hannibal"]._replace(strip=True)
    messages = [i.message for i in run.validate(spec) if i.field == "strip"]
    assert messages == [run.STRIP_CAVEAT]


@pytest.mark.parametrize("change, field", [
    (dict(airspeed_mps=0.0), "airspeed_mps"),
    (dict(altitude_m=-5.0), "altitude_m"),
    (dict(dt=0.0), "dt"),
    (dict(name=""), "name"),
    (dict(name="a/b"), "name"),
    (dict(aircraft="concorde"), "aircraft"),
    (dict(seconds=-1.0), "seconds"),
    (dict(lead_in=0.0), "lead_in"),
])
def test_values_the_simulator_cannot_fly_are_errors(change, field):
    spec = run.PRESETS["vortex-hannibal"]._replace(**change)
    assert field in _fields(spec, "error")


def test_a_non_positive_wind_parameter_is_an_error():
    spec = run.with_param(run.PRESETS["vortex-hannibal"], "r0", -1.0)
    assert "r0" in _fields(spec, "error")


def test_strip_loads_on_the_manoeuvre_are_an_error():
    assert "strip" in _fields(run.PRESETS["manoeuvre"]._replace(strip=True), "error")


def test_a_microburst_within_a_wingspan_of_the_ground_is_an_error():
    spec = run.PRESETS["microburst"]._replace(altitude_m=5.0)
    assert "altitude_m" in _fields(spec, "error")


def test_editing_a_sourced_value_declares_it_and_reset_restores_it():
    spec = run.PRESETS["vortex-hannibal"]
    assert run.provenance(spec)["r0"].status == "sourced"
    edited = run.with_param(spec, "r0", 150.0)
    assert run.provenance(edited)["r0"].status == "declared"
    assert "not the source's number" in run.provenance(edited)["r0"].note
    # Typing the source's number back makes it the source's number again.
    assert run.provenance(run.with_param(edited, "r0", spec.wind.params["r0"]))["r0"].status == "sourced"
    assert run.reset_param(edited, "r0") == spec


def test_a_declared_parameter_keeps_its_reason_when_edited():
    spec = run.PRESETS["updraft"]
    reason = spec.wind.declared["sharpness"]
    edited = run.with_param(spec, "sharpness", 4.0)
    assert run.provenance(edited)["sharpness"] == run.Provenance("declared", reason)


def test_flight_condition_provenance_follows_the_cruise_entry():
    spec = run.PRESETS["vortex-hannibal"]
    prov = run.provenance(spec)
    assert prov["airspeed_mps"].status == "sourced"
    assert prov["airspeed_mps"].note == "NASA CR-2144 flight condition 9"
    assert run.provenance(spec._replace(airspeed_mps=200.0))["airspeed_mps"].status == "declared"
    assert run.provenance(run.PRESETS["microburst"])["altitude_m"].status == "declared"


@pytest.mark.parametrize("name", list(run.PRESETS))
def test_the_recorded_field_rebuilds_to_the_field_that_is_flown(name):
    """`wind_meta` is `artifact.rebuild_field`'s input; they must agree."""
    import jax.numpy as jnp
    import numpy as np

    spec = run.PRESETS[name]
    flown = run.build_field(spec)
    rebuilt = artifact.rebuild_field({"wind_field": run.wind_meta(spec)})
    geo = run.geometry(spec)
    for north in np.linspace(geo.start_north, geo.end_north, 7):
        pos = jnp.array([north, 0.0, -spec.altitude_m + 40.0])
        assert np.array_equal(np.asarray(flown(pos)), np.asarray(rebuilt(pos)))


def test_set_edits_run_and_wind_fields():
    spec = run.apply_set(run.PRESETS["vortex-hannibal"], "lead_in=20")
    assert spec.lead_in == 20
    spec = run.apply_set(spec, "wind.r0=150")
    assert spec.wind.params["r0"] == 150 and "r0" in spec.wind.declared
    with pytest.raises(ValueError):
        run.apply_set(spec, "caveats=x")


def test_run_directories_never_overwrite(tmp_path):
    spec = run.PRESETS["updraft"]
    first = run.run_directory(tmp_path, spec)
    first.mkdir()
    second = run.run_directory(tmp_path, spec)
    assert second != first and second.name == first.name + "-2"


pyarrow = pytest.importorskip("pyarrow", reason="saving needs the `ui` extra")


@pytest.fixture(scope="module")
def short_vortex():
    """A short, coarse-dt vortex run: the first core and nothing after it."""
    base = run.PRESETS["vortex-hannibal"]
    r0 = base.wind.params["r0"]
    seconds = (base.lead_in * r0 + 2.0 * r0) / base.airspeed_mps
    spec = base._replace(dt=0.05, seconds=seconds)
    stages = []
    return run.fly(spec, on_stage=stages.append), stages


def test_fly_reports_its_stages_in_order(short_vortex):
    _, stages = short_vortex
    assert stages == ["trimming", "flying", "checks"]


def test_fly_writes_an_artifact_that_reads_and_rebuilds(tmp_path, short_vortex):
    flown, _ = short_vortex
    path = run.save(flown, tmp_path)
    back = artifact.read_run(path)
    assert back.meta["config_hash"] == flown.meta["config_hash"]
    assert back.meta["wind_field"]["kind"] == "VortexArray"
    assert back.meta["declared_parameters"]["seconds_provenance"]
    assert len(back.checks) == len(flown.report)
    field = artifact.rebuild_field(back.meta)
    import jax.numpy as jnp
    assert float(jnp.linalg.norm(field(jnp.array([0.0, 0.0, -back.meta["flight_condition"]["altitude_m"] + 100.0])))) > 0


def test_an_untrimmable_condition_names_the_fix():
    spec = run.PRESETS["vortex-hannibal"]._replace(airspeed_mps=60.0)
    with pytest.raises(run.TrimError) as err:
        run.fly(spec)
    message = str(err.value)
    assert "boeing747 at 60.0 m/s" in message
    assert "minimum-drag speed" in message


def test_a_spec_with_errors_is_refused_before_anything_is_flown():
    with pytest.raises(run.SpecError):
        run.fly(run.PRESETS["vortex-hannibal"]._replace(dt=0.0))


# ---------------------------------------------------------------------------
# Phase 2: the new field kinds, the overlay and the solver options
# ---------------------------------------------------------------------------


def test_the_original_presets_record_no_phase2_keys():
    """A parameter at its default is not written, so the four original kinds'
    wind blocks -- and their config_hash -- are what they were."""
    for name in ("vortex-hannibal", "vortex-morton", "updraft", "lee-wave", "microburst"):
        meta = run.wind_meta(run.PRESETS[name])
        assert "overlay" not in meta and "profile" not in meta["params"]
        assert meta["model"] == "wind.field_model"
    assert run.solver_meta(run.PRESETS["vortex-hannibal"]) == {}
    assert run.solver_caveats(run.PRESETS["vortex-hannibal"]) == []


def test_a_choice_parameter_is_validated_against_its_choices():
    spec = run.with_param(run.PRESETS["vortex-hannibal"], "profile", "gaussian")
    assert ("error", "profile") in {(i.level, i.field) for i in run.validate(spec)}


def test_changing_the_core_profile_declares_it():
    spec = run.with_param(run.PRESETS["vortex-hannibal"], "profile", "lamb-oseen")
    prov = run.provenance(spec)["profile"]
    assert prov.status == "declared" and "rankine" in prov.note
    assert run.wind_meta(spec)["params"]["profile"] == "lamb-oseen"
    back = run.reset_param(spec, "profile")
    assert run.provenance(back)["profile"].status == "sourced"


def test_a_seed_must_be_a_whole_number():
    spec = run.with_param(run.PRESETS["dryden"], "seed", 1.5)
    assert ("error", "seed") in {(i.level, i.field) for i in run.validate(spec)}


def test_the_line_form_refuses_a_lamb_oseen_core():
    spec = run.with_param(run.with_param(run.PRESETS["mehta-hannibal"], "form", "line"),
                          "profile", "lamb-oseen")
    assert "wind" in {i.field for i in run.errors(spec)}


def test_an_overlay_is_added_edited_recorded_and_removed():
    spec = run.with_overlay_kind(run.PRESETS["vortex-hannibal"], "VonKarman")
    assert spec.overlay.kind == "VonKarman"
    spec = run.with_overlay_param(spec, "sigma_w", 2.0)
    assert run.provenance(spec)["overlay.sigma_w"].status == "declared"
    assert run.provenance(spec)["overlay.L_w"].status == "sourced"
    meta = run.wind_meta(spec)
    assert meta["overlay"]["params"]["sigma_w"] == 2.0
    assert run.RunSpec.from_json(spec.to_json()) == spec
    assert run.with_overlay_kind(spec, "none").overlay is None


def test_an_overlay_on_the_manoeuvre_is_an_error():
    spec = run.with_overlay_kind(run.PRESETS["manoeuvre"], "Dryden")
    assert "overlay" in {i.field for i in run.errors(spec)}


def test_the_overlay_is_in_the_flown_and_the_rebuilt_field():
    import jax.numpy as jnp

    spec = run.PRESETS["mehta-hannibal-turbulent"]
    meta = {"wind_field": run.wind_meta(spec)}
    flown, rebuilt = run.build_field(spec), artifact.rebuild_field(meta)
    bare = run.build_field(spec._replace(overlay=None))
    pos = jnp.array([100.0, 0.0, -spec.altitude_m])
    assert jnp.allclose(flown(pos), rebuilt(pos), atol=0, rtol=0)
    assert not jnp.allclose(flown(pos), bare(pos))


def test_the_gust_lag_needs_a_short_step_and_stage_sampling():
    spec = run.PRESETS["vortex-hannibal"]._replace(gust_lag=True, dt=0.5)
    assert "dt" in {i.field for i in run.errors(spec)}
    spec = run.PRESETS["vortex-hannibal"]._replace(gust_lag=True, stage_sampled=False)
    assert "gust_lag" in {i.field for i in run.errors(spec)}


def test_the_solver_options_are_recorded_only_when_on():
    spec = run.PRESETS["vortex-hannibal"]._replace(gust_lag=True, wing_tail=True,
                                                   stage_sampled=False, dt=0.01)
    meta = run.solver_meta(spec)
    assert meta["gust_lag"] == "kussner_jones" and meta["stage_sampled"] is False
    assert meta["wing_tail"]["tail_arm_m"] > 0
    assert len(run.solver_caveats(spec)) == 3
    assert run.wind_meta(spec)["model"] == "wind.sampled_field_model"


def test_the_jsbsim_entries_take_their_tail_arm_from_jsbsim():
    assert run.tail_arm("boeing737") == pytest.approx(48.04 * 0.3048)
    assert run.tail_arm("boeing747") is None


def test_set_reaches_the_overlay_and_the_solver_options():
    spec = run.apply_set(run.PRESETS["vortex-hannibal"], "overlay=Dryden")
    spec = run.apply_set(spec, "overlay.seed=7")
    spec = run.apply_set(spec, "gust_lag=true")
    assert spec.overlay.params["seed"] == 7 and spec.gust_lag is True
    command = run.command(spec)
    assert "--set gust_lag=True" in command and "--set overlay=Dryden" in command
