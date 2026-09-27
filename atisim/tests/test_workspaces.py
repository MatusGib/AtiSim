"""The development workspaces: check profiles, the step inspector, Diagnostics,
Compare, and flying a spec at another commit (plan sections 3.3 and 3.4)."""

import json
import os

import numpy as np
import pytest

from atisim import run
from atisim.analysis import artifact, commits, devfigures, diagnostics, profiles
from atisim.analysis import step_inspector

pytest.importorskip("pyarrow")

SECONDS = 3.0


def _spec(**changes):
    return run.preset("updraft")._replace(**{"seconds": SECONDS, **changes})


@pytest.fixture(scope="module")
def flown(tmp_path_factory):
    """Four short High runs, one per way the step can be flown, and a Standard one."""
    root = tmp_path_factory.mktemp("workspaces")
    out = {}
    for key, changes in {
        "default": {}, "gust-lag": {"gust_lag": True}, "wing-tail": {"wing_tail": True},
        "strip-held": {"strip": True, "stage_sampled": False},
    }.items():
        out[key] = run.save(run.fly(_spec(fidelity="high", name=f"w-{key}", **changes)), root)
    out["standard"] = run.save(run.fly(_spec(name="w-standard")), root)
    out["manoeuvre"] = run.save(run.fly(run.preset("manoeuvre")._replace(
        fidelity="high", seconds=6.0)), root)
    return root, out


# --- check profiles -----------------------------------------------------------


def test_each_profile_peaks_at_its_checks_value(flown):
    from atisim.aircraft import REGISTRY

    _, runs = flown
    r = artifact.read_run(runs["default"])
    built = profiles.build(r.trajectory, REGISTRY[r.meta["aircraft"]["key"]],
                           artifact.rebuild_field(r.meta), r.checks)
    assert set(built) == {c["name"] for c in r.checks}
    for check in r.checks:
        p = built[check["name"]]
        if isinstance(p, str):
            assert check["name"] == "field divergence"
            continue
        if check["name"] in ("energy residual peak", "trimmed start", "rms normal load"):
            continue  # the check reduces the line further (a ratio or an offset)
        peak = max(float(np.nanmax(y)) for _, y in p.lines)
        assert peak == pytest.approx(check["value"], rel=1e-9, abs=1e-15), check["name"]


# --- the step inspector -------------------------------------------------------


@pytest.mark.parametrize("key", ["default", "gust-lag", "wing-tail", "strip-held",
                                 "manoeuvre"])
def test_the_inspector_re_steps_the_log_exactly(flown, key):
    _, runs = flown
    r = artifact.read_run(runs[key])
    d = diagnostics.read(runs[key])
    lag = (np.stack([d.columns["gust_lag_1"], d.columns["gust_lag_2"]], axis=1)
           if "gust_lag_1" in d.columns else None)
    n = len(r.trajectory.t)
    index = n // 2 if key != "manoeuvre" else int(np.argmax(
        np.abs(np.diff(r.trajectory.controls[:, 0])))) - 1
    detail = step_inspector.detail(r.trajectory, r.meta, index, lag=lag)
    assert all(v == pytest.approx(0.0, abs=1e-12) for v in detail.restep_error.values())
    assert max(detail.refine_difference.values()) < 1e-6
    assert [s.fraction for s in detail.stages] == [0.0, 0.5, 0.5, 1.0]

    # The stages it shows combine to the step `integrate.step` takes.
    import jax.numpy as jnp

    from atisim.state import Controls

    fl = step_inspector.flight(r.meta)
    sim = step_inspector._sim(r.trajectory, detail.index, lag)
    controls = Controls(*(jnp.asarray(v) for v in r.trajectory.controls[detail.index + 1]))
    _, mine = step_inspector.stages(fl, sim, controls)
    ref = step_inspector._step(fl, sim, controls, fl.dt).state
    assert max(step_inspector._groups(mine, ref).values()) < 1e-12


def test_a_gust_lag_step_needs_the_lag_states(flown):
    _, runs = flown
    r = artifact.read_run(runs["gust-lag"])
    with pytest.raises(ValueError, match="High"):
        step_inspector.detail(r.trajectory, r.meta, 5)


def test_held_stages_use_the_first_sample_and_sampled_stages_their_own(flown):
    _, runs = flown
    held = step_inspector.detail(*_run(runs["strip-held"]), 10)
    assert all(np.array_equal(s.wind_used, held.stages[0].wind_used) for s in held.stages)
    sampled = step_inspector.detail(*_run(runs["default"]), 10)
    assert all(np.array_equal(s.wind_used, s.wind_field) for s in sampled.stages)


def _run(path):
    r = artifact.read_run(path)
    return r.trajectory, r.meta


# --- figures -------------------------------------------------------------------


def test_extreme_indices_keep_every_bucket_extreme():
    y = np.sin(np.linspace(0, 40, 10_001)) + np.linspace(0, 1, 10_001)
    y[1234] = 9.0
    at = devfigures.extreme_indices(y, 200)
    assert 1234 in at and 0 in at and len(y) - 1 in at
    assert len(at) <= 402 and np.all(np.diff(at) > 0)
    assert np.array_equal(devfigures.extreme_indices(y[:50], 200), np.arange(50))


def test_the_budget_stacks_every_term_at_the_same_instants(flown):
    _, runs = flown
    d = diagnostics.read(runs["default"])
    fig = devfigures.budget(d.columns, "Cm", cursor_t=1.0)
    stacked = [tr for tr in fig.data if tr.stackgroup]
    assert stacked and all(np.array_equal(tr.x, stacked[0].x) for tr in stacked)
    term, value, share = devfigures.dominant(d.columns, "Cm", len(d.columns["t"]) - 1)
    assert term in {k.split(".", 1)[1] for k in devfigures.terms_of(d.columns, "Cm")}
    assert 0.0 < share <= 1.0


def test_every_workspace_figure_builds(flown):
    _, runs = flown
    d = diagnostics.read(runs["default"])
    r = artifact.read_run(runs["default"])
    devfigures.channel_strips(d.columns, d.channels, ["n_z", "Cm.alpha", "nonsense"], 1.0,
                              (0.5, 2.0))
    forces = devfigures.force_budget(d.columns, 1.0)
    devfigures.energy_budget(d.columns, 1.0)
    # The application moves the cursor in the browser by its name: one line per strip.
    cursor = [sh for sh in forces.layout.shapes if sh.name == "cursor"]
    assert len(cursor) == 3 and all(sh.x0 == sh.x1 == 1.0 for sh in cursor)
    fig = devfigures.scene_high(r.trajectory, d.columns, "alpha",
                                artifact.rebuild_field(r.meta), cursor_index=100)
    assert any(tr.type == "cone" for tr in fig.data)
    devfigures.compare_strips([(d.columns["t"], d.columns["n_z"]),
                               (d.columns["t"][::2], d.columns["n_z"][::2])], "n_z",
                              ["A", "B"], "g")


# --- the pages -----------------------------------------------------------------


def test_the_diagnostics_page_builds_for_high_and_standard(flown):
    pytest.importorskip("dash")
    from atisim.apps import results, shell
    from atisim.tests.test_app import _find, _text

    root, runs = flown
    ws = shell.Workspace(root)
    page = results.layout(ws, runs["default"].name, view="diagnostics")
    assert _find(page, "diag-run") is not None and _find(page, "run") is None
    assert _find(page, "diag-strips") is not None
    assert "High fidelity" in _text(page)
    page = results.layout(ws, runs["standard"].name, view="diagnostics")
    text = _text(page)
    assert "Standard fidelity" in text and "Re-fly at High" in text
    assert _find(page, "diag-refly").disabled is False  # it kept its spec
    # The Overview carries the switch to it.
    page = results.layout(ws, runs["default"].name)
    assert _find(page, "view-diagnostics").href == (
        f"/results?run={runs['default'].name}&view=diagnostics")


def test_the_compare_page_lists_what_differs(flown):
    pytest.importorskip("dash")
    from atisim.apps import compare, results, shell
    from atisim.tests.test_app import _find, _text

    root, runs = flown
    ws = shell.Workspace(root)
    a, b = runs["default"].name, runs["gust-lag"].name
    page = results.layout(ws, a, view="compare", extra=b)
    text = _text(page)
    assert "integrator.gust_lag" in text and "What differs" in text
    assert _find(page, "compare-channel").value == "n_z"
    assert [o["value"] for o in _find(page, "compare-channel").data
            if o["value"].startswith("diag:")], "both are High: their channels are offered"
    assert compare.parse_runs(a, f"{b},{a},") == [a, b]
    assert compare.href([a, b]) == f"/results?view=compare&run={a}&runs={b}"
    rows = dict(compare.differences([ws.loaded(a).run.meta, ws.loaded(b).run.meta]))
    assert rows["integrator.gust_lag"] == ["(absent)", "kussner_jones"]
    assert "created" not in rows
    # Lists that begin alike still read differently: the item one run adds shows.
    one, two = json.dumps(["The headline form"]), json.dumps(["The headline form", "Lag on"])
    assert compare.shown(one, [two]) == ["1 item"]
    assert compare.shown(two, [one]) == ["2 items, 1 not in every other run", "+ Lag on"]
    assert compare.shown("f" * 64, ["e" * 64]) == ["f" * 64]


def test_fly_again_with_one_change_names_and_validates_the_change():
    from atisim.apps import compare

    spec = _spec()
    changed = compare._changed(spec, "gust_lag", "true")
    assert changed.gust_lag is True and changed.name == f"{spec.name}-gust_lag-true"
    with pytest.raises(ValueError):
        compare._changed(spec, "aircraft", "boeing737")


# --- flying at another commit ------------------------------------------------------


def test_an_older_engine_gets_the_spec_it_can_read():
    spec = _spec()
    fields = set(run.RunSpec._fields) - {"fidelity", "wing_tail"}
    data = commits.spec_for_old_engine(spec, fields)
    assert "fidelity" not in data and "wing_tail" not in data
    with pytest.raises(commits.CommitError, match="fidelity"):
        commits.spec_for_old_engine(spec._replace(fidelity="high"), fields)


def test_a_ref_that_is_not_a_commit_is_refused(tmp_path):
    with pytest.raises(commits.CommitError):
        commits.resolve("no-such-ref-anywhere")
    with pytest.raises(commits.CommitError):
        commits.resolve("main; rm -rf /")


def test_a_ref_before_the_cli_is_refused_by_name(tmp_path):
    added = commits._git("log", "--diff-filter=A", "--format=%H", "--", "atisim/cli.py")
    if not added:
        pytest.skip("this clone does not hold the commit that added the CLI")
    before = added.splitlines()[-1] + "~1"
    try:
        commits.resolve(before)
    except commits.CommitError:
        pytest.skip("this clone is too shallow to hold the commit before the CLI")
    with pytest.raises(commits.CommitError, match="predates atisim/cli.py"):
        commits.fly_at(_spec(), before, tmp_path)


def test_a_spec_flies_at_head_in_a_worktree_and_the_worktree_goes(tmp_path):
    lines = []
    path = commits.fly_at(_spec(seconds=1.0), "HEAD", tmp_path, on_line=lines.append)
    assert (path / "run.parquet").exists() and path.parent == tmp_path.resolve()
    sha = commits.resolve("HEAD")[:7]
    assert path.name.endswith(sha) and artifact.read_run(path).meta["git_sha"].startswith(sha)
    assert not any((tmp_path / ".worktrees").iterdir())
    assert any(line.startswith("wrote ") for line in lines)


# --- the engine worker (development mode) --------------------------------------------


def test_the_engine_worker_flies_reloads_and_keeps_the_compile_cache(tmp_path, monkeypatch):
    from atisim.apps import jobs, worker
    from atisim.tests.test_app import _finished

    runner = jobs.JobRunner(tmp_path, worker=True)
    assert runner.engine() == {"mode": "subprocess", "alive": False, "changed": False}
    try:
        job = _finished(runner, runner.submit(_spec(seconds=1.0, name="w-worker")),
                        timeout=300)
        assert job["state"] == "done", job["log"]
        assert (tmp_path / job["path"].split("/")[-1] / "run.parquet").exists()
        assert job["checks"] and job["timing"]["flying"] > 0
        assert [s["name"] for s in job["stages"]][:2] == ["validating", "validating"]
        first = runner.engine()
        assert first["alive"] and not first["changed"] and first["pid"] != os.getpid()
        assert any(worker.cache_dir(tmp_path).iterdir()), "the compile went to the cache"

        # An edit to the engine shows as a change; Reload stops the worker, and
        # the next job starts a new one.
        monkeypatch.setattr(jobs, "engine_stamp", lambda: first["stamp"] + 1.0)
        assert runner.engine()["changed"]
        runner.reload()
        assert runner.engine()["alive"] is False
        job = _finished(runner, runner.submit(_spec(seconds=1.0, name="w-worker-2")),
                        timeout=300)
        assert job["state"] == "done", job["log"]
        assert runner.engine()["pid"] != first["pid"]

        # A job the worker cannot fly fails the job, not the worker.
        job = _finished(runner, runner.submit(_spec(dt=-1.0, name="w-worker-3")))
        assert job["state"] == "failed" and job["error_field"] == "dt"
        assert runner.engine()["alive"]
    finally:
        runner._stop_worker()
