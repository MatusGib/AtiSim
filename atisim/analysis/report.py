"""A report artifact: what an analysis writes where a flight writes `run.parquet`.

A flown encounter is a time series and `artifact.write_run` stores it. The
phase-2 analyses (an ensemble, the linear modes, a gust transfer sweep, a
convergence study, a script run as a study) produce figures and tables instead,
so they need a second artifact shape. It lives in the same runs directory, with
the same `meta.json` that `analysis.runs.scan` lists, and one more file:

    meta.json     schema_version, result_kind, analysis, title, the spec, git_sha,
                  created, caveats, and the member runs if there are any
    report.json   the sections: each a title, text, and a Plotly figure and/or a
                  table; a study's may also carry an image, a log and files
    files/        a study's copies of what its script wrote (`atisim.studies`)
    log.txt       a study's whole log
    checks.json   [] -- a report is not judged by the flight checks; its
                  sections say what was compared with what

`SCHEMA_VERSION` stays 1: a report is a new file in a new directory, and a
reader that does not know `result_kind` never sees one, because it has no
`run.parquet`.

Figures are stored as Plotly JSON, so the page draws the figure the analysis
made rather than re-deriving it. Needs the `ui` extra for plotly, like
`write_run` needs it for pyarrow.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

from atisim.analysis import artifact


class Section(NamedTuple):
    title: str
    text: str = ""
    figure: object = None  # a plotly Figure, or its JSON dict
    table: dict | None = None  # {"headers": [...], "rows": [[...], ...]}
    # A study's: an image and files the report directory holds (paths relative
    # to it), and a block of printed text.
    image: str | None = None
    log: str | None = None
    files: tuple = ()


class Report(NamedTuple):
    analysis: str
    title: str
    spec: dict
    sections: list
    caveats: tuple = ()
    aircraft: str | None = None
    members: tuple = ()  # member run directories, relative to the report's own
    summary: str = ""


def table(headers, rows) -> dict:
    """A table as the page draws it. Numbers stay numbers; the page formats them."""
    return {"headers": list(headers), "rows": [list(r) for r in rows]}


def _figure_json(figure):
    if figure is None or isinstance(figure, dict):
        return figure
    return json.loads(figure.to_json())


def is_report(directory) -> bool:
    try:
        meta = json.loads((Path(directory) / "meta.json").read_text())
    except (OSError, ValueError):
        return False
    return isinstance(meta, dict) and "result_kind" in meta


def write_report(directory, report: Report) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    meta = {
        "schema_version": artifact.SCHEMA_VERSION,
        "result_kind": "report",
        "analysis": report.analysis,
        "title": report.title,
        "summary": report.summary,
        "git_sha": artifact.git_sha(),
        "created": datetime.now(timezone.utc).isoformat(),
        "aircraft": {"key": report.aircraft} if report.aircraft else {},
        "spec": report.spec,
        "caveats": list(report.caveats),
        "members": list(report.members),
    }
    (directory / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    sections = []
    for s in report.sections:
        section = {"title": s.title, "text": s.text, "figure": _figure_json(s.figure),
                   "table": s.table}
        # Only when set, so every earlier analysis writes the report it wrote.
        section.update({k: v for k, v in (("image", s.image), ("log", s.log),
                                          ("files", list(s.files))) if v})
        sections.append(section)
    (directory / "report.json").write_text(json.dumps({"sections": sections},
                                                      default=_plain))
    (directory / "checks.json").write_text("[]")
    return directory


def _plain(value):
    """JSON for the numpy scalars and arrays a table may carry."""
    import numpy as np

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


class Loaded(NamedTuple):
    meta: dict
    sections: list


def read_report(directory) -> Loaded:
    directory = Path(directory)
    meta = json.loads((directory / "meta.json").read_text())
    body = json.loads((directory / "report.json").read_text())
    return Loaded(meta, body.get("sections", []))
