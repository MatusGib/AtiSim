"""The `atisim` command, called as `main([...])`."""

import builtins
import json

import pytest

import atisim  # noqa: F401
from atisim import cli, run


def test_presets_lists_every_preset_with_its_source(capsys):
    assert cli.main(["presets"]) == 0
    out = capsys.readouterr().out
    for name, spec in run.PRESETS.items():
        assert name in out
        assert spec.wind.source in out
    assert "blank" in out


def test_list_on_a_missing_directory_says_how_to_make_a_run(tmp_path, capsys):
    assert cli.main(["list", str(tmp_path / "nothing")]) == 0
    assert "atisim run --preset vortex-hannibal" in capsys.readouterr().out


def test_list_survives_a_broken_meta_json(tmp_path, capsys):
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "meta.json").write_text("{not json")
    assert cli.main(["list", str(tmp_path)]) == 0
    assert "broken  ERROR: meta.json cannot be read" in capsys.readouterr().out


def test_run_refuses_a_spec_with_errors(tmp_path, capsys):
    code = cli.main(["run", "--preset", "updraft", "--out", str(tmp_path),
                     "--set", "dt=0"])
    assert code == 2
    assert "error: dt:" in capsys.readouterr().err


def test_run_refuses_an_unknown_preset(tmp_path, capsys):
    assert cli.main(["run", "--preset", "nope", "--out", str(tmp_path)]) == 2
    assert "unknown preset" in capsys.readouterr().err


def test_ui_without_dash_prints_the_install_hint(monkeypatch, capsys):
    real = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name.split(".")[0] in ("dash", "dash_mantine_components", "dash_iconify"):
            raise ImportError(name)
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    assert cli.main(["ui", "--no-browser"]) != 0
    assert 'pip install -e ".[ui]"' in capsys.readouterr().err


pytest.importorskip("pyarrow", reason="writing a run needs the `ui` extra")


def test_run_flies_a_short_preset_and_list_shows_it(tmp_path, capsys):
    code = cli.main(["run", "--preset", "updraft", "--out", str(tmp_path),
                     "--set", "seconds=2", "--set", "dt=0.05"])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "trimming ..." in out and "flying ..." in out and "wrote " in out
    written = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert len(written) == 1 and written[0].name.startswith("updraft-boeing747-")
    meta = json.loads((written[0] / "meta.json").read_text())
    assert meta["declared_parameters"]["seconds"] == 2.0

    assert cli.main(["list", str(tmp_path)]) == 0
    assert written[0].name in capsys.readouterr().out


def test_run_from_a_spec_file(tmp_path, capsys):
    spec = run.PRESETS["manoeuvre"]._replace(dt=0.05)
    path = tmp_path / "spec.json"
    path.write_text(spec.to_json())
    # A spec file round-trips; flying it is covered above, so only parse here.
    args = cli.build_parser().parse_args(["run", str(path), "--set", "name=m2"])
    assert cli._spec_from_args(args) == spec._replace(name="m2")
