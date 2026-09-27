"""The study adapter: the research scripts in `scripts/`, run as they are, their
output kept as a report (plan section 3.5, task P12)."""

import json
import textwrap
from pathlib import Path

import pytest

from atisim import analyses, studies
from atisim.analysis import artifact, report

pytest.importorskip("plotly")

# A 1x1 PNG, so the fake study writes a real image.
PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000d49444154789c6360000002000154a24f5d0000000049454e44ae426082")


def test_every_script_is_a_study():
    """A new helper module, or a script that silently does nothing when run,
    fails here by name."""
    assert studies.helpers() == ()
    found = {s.name: s for s in studies.scripts()}
    assert len(found) == len(list(studies.SCRIPTS.glob("*.py")))
    assert found["checkpoint"].description.startswith("Trim the 747")
    # sanity.py has no docstring: its header comment describes it.
    assert found["sanity"].description.startswith("Quick checks to make sure")
    assert analyses.SCRIPT.choices == studies.names()


def _fake_repo(tmp_path) -> Path:
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "out").mkdir()
    (repo / "out" / "old.csv").write_text("a,b\n1,2\n")  # there before: not kept
    (repo / "scripts" / "writes.py").write_text(textwrap.dedent(f'''
        """Write one of each kind a study keeps."""
        import sys
        from pathlib import Path

        out = Path("out")
        (out / "figure.png").write_bytes({PNG!r})
        (out / "table.csv").write_text("x,y\\n" + "".join(f"{{i}},{{i * i}}\\n" for i in range(40)))
        (out / "numbers.json").write_text('{{"k": 1}}')
        (out / "ignored.npz").write_bytes(b"not kept")
        print("arguments:", sys.argv[1:])
        print("done")
        '''))
    (repo / "scripts" / "fails.py").write_text(
        "import sys\nprint('needs data the public tree lacks', file=sys.stderr)\nsys.exit(3)\n")
    (repo / "scripts" / "sleeps.py").write_text("import time\nprint('start')\ntime.sleep(60)\n")
    (repo / "scripts" / "helper.py").write_text(
        '"""A helper."""\nimport sys\nsys.path.insert(0, ".")\nX = 1\n')
    return repo


def test_a_study_keeps_what_its_script_wrote_and_only_that(tmp_path):
    repo = _fake_repo(tmp_path)
    assert studies.names(repo / "scripts") == ("fails", "sleeps", "writes")
    assert studies.helpers(repo / "scripts") == ("helper",)
    lines = []
    out = studies.run_script(studies.find("writes", repo / "scripts"), tmp_path / "study",
                             on_line=lines.append, args=["--quick", "a b"], repo=repo)
    assert out.returncode == 0 and lines[-1] == "done"
    assert "arguments: ['--quick', 'a b']" in out.log
    assert {rel for rel, _ in out.outputs} == {"out/figure.png", "out/table.csv",
                                               "out/numbers.json"}
    assert (tmp_path / "study" / "files" / "out" / "figure.png").read_bytes() == PNG


def test_a_failing_script_is_recorded_with_its_own_message(tmp_path, monkeypatch):
    repo = _fake_repo(tmp_path)
    monkeypatch.setattr(studies, "SCRIPTS", repo / "scripts")
    monkeypatch.setattr(studies, "REPO", repo)
    directory = tmp_path / "study-fails"
    directory.mkdir()
    spec = analyses.default("study")._replace(params={"script": "fails", "arguments": "",
                                                      "minutes": 1.0})
    r = analyses.study(spec, directory, analyses.Progress())
    assert r.summary.startswith("Exited with status 3")
    assert "needs data the public tree lacks" in r.summary
    assert r.caveats and "did not finish cleanly" in r.caveats[0]


def test_a_script_is_stopped_at_its_time_limit(tmp_path):
    repo = _fake_repo(tmp_path)
    out = studies.run_script(studies.find("sleeps", repo / "scripts"), tmp_path / "s",
                             minutes=0.03, repo=repo)
    assert out.returncode is None and out.elapsed_s < 30 and "start" in out.log


def test_the_report_shows_images_tables_files_and_the_log(tmp_path, monkeypatch):
    repo = _fake_repo(tmp_path)
    monkeypatch.setattr(studies, "SCRIPTS", repo / "scripts")
    monkeypatch.setattr(studies, "REPO", repo)
    directory = tmp_path / "runs" / "study-writes-x"
    directory.mkdir(parents=True)
    spec = analyses.default("study")._replace(params={"script": "writes", "arguments": "",
                                                      "minutes": 1.0})
    report.write_report(directory, analyses.study(spec, directory, analyses.Progress()))
    loaded = report.read_report(directory)
    by_title = {s["title"]: s for s in loaded.sections}
    assert by_title["out/figure.png"]["image"] == "files/out/figure.png"
    csv = by_title["out/table.csv"]
    assert csv["table"]["headers"] == ["x", "y"] and len(csv["table"]["rows"]) == 30
    assert csv["table"]["rows"][3] == [3.0, 9.0] and "30 of 40" in csv["text"]
    assert by_title["Other files it wrote"]["files"] == ["files/out/numbers.json"]
    assert by_title["What it printed"]["log"].endswith("done")
    assert (directory / "log.txt").read_text().endswith("done\n")
    assert loaded.meta["result_kind"] == "report" and loaded.meta["analysis"] == "study"
    # The sections of every other analysis are written as before: no new keys.
    assert set(by_title["The script"]) == {"title", "text", "figure", "table"}

    pytest.importorskip("dash")
    from atisim.apps import results, shell
    from atisim.tests.test_app import _text

    app = shell.build_app(tmp_path / "runs")
    page = results.layout(shell.Workspace(tmp_path / "runs"), directory.name)
    assert "Study: writes.py" in _text(page)
    assert f"/files/{directory.name}/files/out/figure.png" in json.dumps(
        page, default=lambda c: c.to_plotly_json())
    client = app.server.test_client()
    got = client.get(f"/files/{directory.name}/files/out/figure.png")
    assert got.status_code == 200 and got.data == PNG
    assert client.get(f"/files/{directory.name}/log.txt").status_code == 200
    assert client.get(f"/files/{directory.name}/meta.json").status_code == 200
    assert client.get("/files/../repo/scripts/writes.py").status_code == 404
    assert client.get(f"/files/{directory.name}/../../repo/out/table.csv").status_code == 404


def test_sanity_and_checkpoint_run_as_studies(tmp_path):
    """The plan's acceptance: both run as studies, and what they print is the
    report."""
    from atisim import cli

    assert cli.main(["study", "sanity", "--out", str(tmp_path)]) == 0
    sha = artifact.git_sha()[:7] or "nogit"
    loaded = report.read_report(tmp_path / f"study-sanity-{sha}")
    assert loaded.meta["summary"].endswith("11/11 checks passed")
    assert loaded.meta["spec"]["params"]["script"] == "sanity"

    lines = []
    path = analyses.perform(analyses.default("study")._replace(
        params={"script": "checkpoint", "arguments": "", "minutes": 10.0}), tmp_path,
        on_log=lines.append)
    assert path.name == f"study-checkpoint-{sha}"
    assert any("short period  wn 0.9513" in line for line in lines)
    log = {s["title"]: s for s in report.read_report(path).sections}["What it printed"]["log"]
    assert "phugoid       wn 0.0684" in log


def test_the_study_form_offers_every_script(tmp_path):
    pytest.importorskip("dash")
    from atisim.apps import analyses as page_mod
    from atisim.apps import shell
    from atisim.tests.test_app import _find, _text

    page = page_mod.layout(shell.Workspace(tmp_path), "study")
    select = _find(page, {"type": "aparam", "name": "script"})
    assert select.searchable and len(select.data) == len(studies.names())
    assert "scripts/sanity.py" in _text(page) and "Studies" in _text(page)
    assert analyses.validate(analyses.default("study")._replace(
        params={"script": "sanity", "arguments": "--x 'unclosed", "minutes": 1.0}))
    aspec = analyses.apply_set(analyses.default("study"), "arguments=3")
    assert aspec.params["arguments"] == "3"


def test_the_study_command_passes_everything_after_the_double_dash(monkeypatch):
    from atisim import cli

    seen = []
    monkeypatch.setattr(cli, "_cmd_analyse", lambda args: seen.append(args.set) or 0)
    cli.main(["study", "analyse", "--out", "runs", "--", "runs/x", "a b", "--quick"])
    cli.main(["study", "sanity"])
    assert seen[0][:2] == ["script=analyse", "arguments=runs/x 'a b' --quick"]
    assert seen[1] == ["script=sanity", "arguments="]


# --- topics: where a reader finds a study ----------------------------------------


def test_every_script_sits_under_one_topic_and_the_topics_name_only_real_scripts():
    """A renamed or deleted script fails here by name; a new one is listed under
    OTHER until a topic names it, so it is never hidden."""
    named = [n for t in studies.TOPICS for n in t.names]
    assert len(named) == len(set(named)), "a script is in two topics"
    have = set(studies.names())
    assert set(named) - have == set(), sorted(set(named) - have)
    listed = [s.name for _, group in studies.by_topic() for s in group]
    assert sorted(listed) == sorted(have)
    assert studies.topic_of("cat_validation").key == "cat"
    assert studies.topic_of("no_such_script") is studies.OTHER


def test_a_new_script_is_listed_under_other(tmp_path):
    (tmp_path / "brand_new.py").write_text('"""A new study."""\nprint("hi")\n')
    (tmp_path / "sanity.py").write_text('"""The sanity checks."""\nprint("ok")\n')
    groups = studies.by_topic(tmp_path)
    assert [(t.key, [s.name for s in g]) for t, g in groups] == [
        ("tools", ["sanity"]), ("other", ["brand_new"])]


def test_about_is_the_docstring_after_its_first_line_with_list_lines_kept(tmp_path):
    path = tmp_path / "study.py"
    path.write_text(textwrap.dedent('''
        """One line.

        What it flies, and against
        what source.

        1 The first question,
          wrapped.
        2 The second.
        """
        print("x")
    '''))
    script = studies.Script("study", path, "One line.")
    assert studies.about(script) == ["What it flies, and against what source.",
                                     "1 The first question, wrapped.\n2 The second."]
    assert studies.about(studies.Script("none", tmp_path / "missing.py", "")) == []
