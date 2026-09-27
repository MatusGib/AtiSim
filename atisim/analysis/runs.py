"""A runs directory, summarised for a list: one row per artifact, never a crash.

`artifact.list_runs` parses every `meta.json` to sort by `created`, so one
unreadable file stops the whole listing. A list a user browses has to survive
that: a broken run becomes an error row naming the file and the fault, and the
rest of the directory still shows.

Reads `meta.json` and `checks.json` only -- a few kB per run -- so fifty runs
list without touching a Parquet file. Pure Python: no Dash, no pyarrow.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

from atisim.analysis import artifact


class RunRow(NamedTuple):
    path: Path
    name: str
    kind: str
    aircraft: str
    created: str | None  # ISO UTC, as meta.json stores it
    created_local: str  # for display, local time, minutes
    verdict: str  # "pass" | "warning" | "fail" | "no checks" | "error" | "report"
    stale: bool
    error: str | None
    checks: list
    # For a report: the analysis that wrote it, and for a study its script.
    # Trailing and defaulted, so every existing construction is unchanged.
    analysis: str | None = None
    script: str | None = None

    @property
    def group(self) -> str:
        """Which list a browser puts the row in: a flight, a report or a study."""
        if self.analysis is None:
            return "flight"
        return "study" if self.analysis == "study" else "report"


def verdict(checks: list[dict]) -> str:
    """One word for a run's checks: fail, warning or pass.

    `fail` when a gate failed or a tripwire fired. `warning` when the alpha band
    is MARGINAL: that gate passes an amber run on purpose (`checks.AlphaBand`),
    so the word says it rather than hiding it. A `report` never contributes: it
    has no threshold, and colouring it would invent a verdict.
    """
    judged = [c for c in checks if c.get("kind") in ("gate", "tripwire")]
    if any(c.get("passed") is False for c in judged):
        return "fail"
    if any(c.get("name") == "alpha band" and "(MARGINAL)" in c.get("detail", "")
           for c in checks):
        return "warning"
    return "pass" if judged else "no checks"


def worst_gate_index(checks: list[dict]) -> int | None:
    """Where "Open in Results" puts the cursor: the first failing gate's worst sample.

    `run_checks` lists the gates in the order a reader should meet them, so the
    first failing gate is the one that condemns the run. None when no gate
    failed, or the failing gate is not per-sample.
    """
    for c in checks:
        if c.get("kind") == "gate" and c.get("passed") is False:
            if c.get("worst_index") is not None:
                return int(c["worst_index"])
    return None


def _local(created: str | None) -> str:
    if not created:
        return "unknown"
    try:
        return datetime.fromisoformat(created).astimezone().strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return created


def _kind(meta: dict) -> str:
    if "result_kind" in meta:  # an analysis report (`analysis.report`)
        if meta.get("analysis") == "study":
            script = ((meta.get("spec") or {}).get("params") or {}).get("script", "")
            return f"study of {script}.py" if script else "study"
        return f"{meta.get('analysis', 'analysis')} report"
    kind = meta.get("wind_field", {}).get("kind", "unknown")
    case = meta.get("wind_field", {}).get("case")
    return f"{kind} ({case})" if case else kind


def read_row(path: Path, here_sha: str | None = None) -> RunRow:
    """One run's row. A fault becomes an error row, never an exception."""
    path = Path(path)
    here = artifact.git_sha() if here_sha is None else here_sha
    try:
        meta = json.loads((path / "meta.json").read_text())
        if not isinstance(meta, dict):
            raise ValueError("meta.json is not a JSON object")
    except (OSError, ValueError) as exc:
        return RunRow(path, path.name, "", "", None, "", "error", False,
                      f"meta.json cannot be read: {exc}", [])
    checks_path = path / "checks.json"
    try:
        report = json.loads(checks_path.read_text()) if checks_path.exists() else []
    except (OSError, ValueError) as exc:
        return RunRow(path, path.name, _kind(meta), meta.get("aircraft", {}).get("key", ""),
                      meta.get("created"), _local(meta.get("created")), "error", False,
                      f"checks.json cannot be read: {exc}", [])
    sha = meta.get("git_sha") or ""
    return RunRow(
        path=path,
        name=path.name,
        kind=_kind(meta),
        aircraft=meta.get("aircraft", {}).get("key", ""),
        created=meta.get("created"),
        created_local=_local(meta.get("created")),
        verdict="report" if "result_kind" in meta else verdict(report),
        stale=bool(here and sha and sha != here),
        error=None,
        checks=report,
        analysis=(meta.get("analysis") or "analysis") if "result_kind" in meta else None,
        script=((meta.get("spec") or {}).get("params") or {}).get("script") or None
        if meta.get("analysis") == "study" else None,
    )


def scan(root) -> list[RunRow]:
    """Every run under `root`, newest first; unreadable runs last."""
    root = Path(root)
    if not root.is_dir():
        return []
    here = artifact.git_sha()
    rows = [read_row(p.parent, here) for p in sorted(root.glob("*/meta.json"))]
    good = sorted((r for r in rows if r.error is None),
                  key=lambda r: r.created or "", reverse=True)
    return good + [r for r in rows if r.error is not None]
