"""High fidelity: the probe pass, `diagnostics.parquet`, and `RunSpec.fidelity`.

The probe re-evaluates the plant over the logged states after the flight (plan
decision D3). Two things are held here. The flight is untouched: a High run's
`run.parquet`, checks and config hash are the Standard run's, byte for byte.
And the probe is the plant: its accelerations are `dynamics.derivatives` at the
inputs `integrate.step` gives its first stage, its n_z is the run's, and the
coefficient terms sum to the totals.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from atisim import dynamics, loads, run, wind
from atisim.aircraft import REGISTRY
from atisim.analysis import diagnostics
from atisim.state import Controls, State

pytest.importorskip("pyarrow")

SECONDS = 3.0


def _spec(**changes):
    return run.preset("updraft")._replace(seconds=SECONDS, **changes)


def _probe(spec):
    flown = run.fly(spec)
    ac = REGISTRY[spec.aircraft]
    field = run.build_field(spec)
    columns, channels = diagnostics.probe(
        flown.trajectory, ac, field, model=run.wind_model(spec, field),
        load_model=loads.strip_model(field, ac) if spec.strip else None,
        lag=flown.encounter.gust_lag)
    return flown, ac, field, columns, channels


def _stage_one(spec, ac, field, traj, lag):
    """`dynamics.derivatives` at each sample, fed as `integrate.step` feeds its
    first stage -- built here from the step's own pieces, not from the probe."""
    model = run.wind_model(spec, field)
    stations = getattr(model, "stations", None)
    load_model = loads.strip_model(field, ac) if spec.strip else None

    def one(pos, vel, quat, omega, c, lag_state):
        s = State(pos_ned=pos, vel_body=vel, quat=quat, omega=omega)
        controls = Controls(*c)
        w = field(pos)
        og = (wind.gust_rates(pos, quat, field) if stations is None
              else wind.sampled_rates(pos, quat, field, stations))
        increment = None if load_model is None else load_model(s)
        if lag is None:
            ad = wind.gust_alphadot(pos, quat, vel, field)
            return dynamics.derivatives(s, controls, ac, w, og, increment=increment,
                                        alphadot_gust=ad)
        airspeed = jnp.linalg.norm(dynamics.relative_velocity(vel, quat, w))
        lag_dot = wind.kussner_lag_rate(lag_state, w[2], airspeed, ac.c)
        ad = wind.lagged_gust_alphadot(pos, quat, vel, field, lag_state, lag_dot)
        return dynamics.derivatives(s, controls, ac, wind.lagged_wind(w, lag_state), og,
                                    increment=increment, alphadot_gust=ad)

    lags = jnp.zeros((len(traj.t), 2)) if lag is None else jnp.asarray(lag)
    return jax.vmap(one)(jnp.asarray(traj.pos_ned), jnp.asarray(traj.vel_body),
                         jnp.asarray(traj.quat), jnp.asarray(traj.omega),
                         jnp.asarray(traj.controls), lags)


@pytest.mark.parametrize("changes", [
    {}, {"gust_lag": True}, {"wing_tail": True}, {"strip": True, "stage_sampled": False},
], ids=["default", "gust-lag", "wing-tail", "strip-held"])
def test_the_probe_is_the_plant(changes):
    spec = _spec(**changes)
    flown, ac, field, cols, channels = _probe(spec)
    enc = flown.encounter

    # The run's own n_z and air-relative alpha, from `vortex_viz._measure`.
    np.testing.assert_allclose(cols["n_z"], enc.n_z, rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.radians(cols["alpha"]), enc.alpha_air, rtol=0, atol=1e-12)

    # The derivatives are the ones the next step's first stage computes.
    d = _stage_one(spec, ac, field, flown.trajectory, enc.gust_lag)
    for i, name in enumerate(("udot", "vdot", "wdot")):
        np.testing.assert_allclose(cols[name], np.asarray(d.vel_body)[:, i],
                                   rtol=0, atol=1e-12)
    for i, name in enumerate(("pdot", "qdot", "rdot")):
        np.testing.assert_allclose(np.radians(cols[name]), np.asarray(d.omega)[:, i],
                                   rtol=0, atol=1e-12)

    # The mirror's forces rebuild those accelerations, and its moments the rates.
    accel = cols["_accel_from_forces"]
    for i, name in enumerate(("udot", "vdot", "wdot")):
        np.testing.assert_allclose(accel[:, i], cols[name], rtol=0, atol=1e-12)
    moment = np.stack([cols["L_moment"], cols["M_moment"], cols["N_moment"]], axis=1)
    omega = np.asarray(flown.trajectory.omega)
    inertia, inertia_inv = np.asarray(ac.inertia), np.asarray(ac.inertia_inv)
    rates = (moment - np.cross(omega, omega @ inertia.T)) @ inertia_inv.T
    np.testing.assert_allclose(rates, np.asarray(d.omega), rtol=0, atol=1e-9)

    # Every coefficient is the sum of its terms (and the strip increment).
    for coeff in ("CL", "CD", "CY", "Cl", "Cm", "Cn"):
        parts = [k for k in cols if k.startswith(coeff + ".")]
        assert parts, coeff
        np.testing.assert_allclose(sum(cols[k] for k in parts), cols[coeff],
                                   rtol=0, atol=1e-12)

    # Every declared channel has a column, and only the checking column is extra.
    names = [c.name for c in channels]
    assert len(names) == len(set(names))
    assert set(cols) - set(names) == {"_accel_from_forces"}
    assert ("CL.strip" in names) == spec.strip


def test_the_energy_columns_are_what_the_closure_gate_measures():
    from atisim import checks

    spec = _spec()
    flown, ac, field, cols, _ = _probe(spec)
    energy, power = checks._energy_and_power(flown.trajectory, ac, field)
    np.testing.assert_array_equal(cols["energy"], np.asarray(energy))
    t = np.asarray(flown.trajectory.t)
    work = np.trapezoid(np.asarray(power), t)
    assert cols["work"][-1] == pytest.approx(work, rel=1e-12)
    assert cols["energy_residual"][0] == 0.0 and cols["work"][0] == 0.0


def _files(directory):
    return {p.name: p.read_bytes() for p in directory.iterdir()}


def test_high_fidelity_leaves_the_run_bit_identical(tmp_path):
    standard = run.fly(_spec())
    high = run.fly(_spec(fidelity="high"))
    assert standard.diagnostics is None and high.diagnostics is not None
    assert standard.meta["config_hash"] == high.meta["config_hash"]

    dir_a, dir_b = run.save(standard, tmp_path / "a"), run.save(high, tmp_path / "b")
    a, b = _files(dir_a), _files(dir_b)
    assert set(a) == {"run.parquet", "meta.json", "checks.json", run.SPEC_FILE}
    assert set(b) == set(a) | {diagnostics.FILENAME}
    assert a["run.parquet"] == b["run.parquet"]
    assert a["checks.json"] == b["checks.json"]
    # The spec travels with the run, so the app can fly it again.
    assert run.spec_of(dir_a) == standard.spec
    assert run.spec_of(dir_b).fidelity == "high"


def test_diagnostics_read_back_with_units_and_timing(tmp_path):
    flown = run.fly(_spec(fidelity="high"))
    directory = run.save(flown, tmp_path)
    assert diagnostics.is_high(directory)
    back = diagnostics.read(directory)
    by_name = {c.name: c for c in back.channels}
    assert by_name["n_z"].unit == "g" and by_name["n_z"].group == "load"
    assert by_name["CL.alpha"].group == "coefficients: CL"
    assert "_accel_from_forces" not in back.columns
    np.testing.assert_array_equal(back.columns["n_z"], flown.encounter.n_z)
    assert set(back.timing) >= {"trimming", "flying", "checks", "diagnostics"}
    assert all(v >= 0 for v in back.timing.values())
    assert len(back.columns["t"]) == len(flown.trajectory.t)


def test_a_standard_run_has_no_diagnostics(tmp_path):
    directory = run.save(run.fly(_spec()), tmp_path)
    assert diagnostics.read(directory) is None
    assert not diagnostics.is_high(directory)


def test_fidelity_is_validated_settable_and_reproduced():
    spec = run.preset("updraft")
    assert spec.fidelity == "standard"
    assert [i.field for i in run.validate(spec._replace(fidelity="ultra"))
            if i.level == "error"] == ["fidelity"]
    high = run.apply_set(spec, "fidelity=high")
    assert high.fidelity == "high"
    assert "--set fidelity=high" in run.command(high)
    assert run.RunSpec.from_json(high.to_json()) == high
    # A spec written before the field existed reads as Standard.
    raw = spec.to_dict()
    raw.pop("fidelity")
    assert run.RunSpec.from_json(raw).fidelity == "standard"


def test_the_cli_writes_diagnostics_at_high(tmp_path, capsys):
    from atisim import cli

    assert cli.main(["run", "--preset", "updraft", "--fidelity", "high",
                     "--set", f"seconds={SECONDS}", "--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "diagnostics ..." in out
    (directory,) = tmp_path.iterdir()
    assert diagnostics.is_high(directory)


def test_a_cut_run_keeps_the_lag_in_step():
    enc = run.fly(_spec(gust_lag=True)).encounter
    assert enc.gust_lag.shape == (len(enc.t), 2)
    cut = run._cut(enc, 10)
    assert cut.gust_lag.shape == (10, 2)
    np.testing.assert_array_equal(cut.gust_lag, enc.gust_lag[:10])


def test_setup_offers_fidelity_and_shows_the_probe_stage():
    from pathlib import Path
    from types import SimpleNamespace

    from atisim.apps import setup

    ws = SimpleNamespace(root=Path("."))
    panel = str(setup.settings_panel("solver", run.preset("updraft"), ws))
    assert "'name': 'fidelity'" in panel
    job = {"state": "done", "elapsed": 1.0, "error": None,
           "spec": run.preset("updraft")._replace(fidelity="high").to_json(),
           "stages": []}
    assert "Probe the diagnostics" in str(setup.progress(job))
    job["spec"] = run.preset("updraft").to_json()
    assert "Probe the diagnostics" not in str(setup.progress(job))
