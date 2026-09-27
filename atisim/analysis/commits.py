"""Fly a spec with the engine at another commit (plan section 3.4, A/B by commit).

"I changed the engine: what moved?" The answer needs the same spec flown by
the old engine. `fly_at` makes a git worktree at the ref, under the runs
directory's `.worktrees/`, runs `atisim run SPEC --out RUNS` there in a
subprocess with the worktree's own package first on the path, and removes the
worktree. The run lands beside the others; its name and `meta.json` carry that
commit's SHA, so Compare shows the two engines side by side.

A ref that predates `atisim/cli.py` cannot fly a spec, and this says so and
stops: there is no fallback to guess at.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import atisim

REPO = Path(atisim.__file__).resolve().parent.parent


class CommitError(RuntimeError):
    """The ref cannot fly the spec; the message says why."""


def _git(*args, cwd=REPO) -> str:
    done = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if done.returncode != 0:
        raise CommitError(done.stderr.strip() or f"git {' '.join(args)} failed")
    return done.stdout.strip()


def resolve(ref: str) -> str:
    """The full SHA of `ref`, or CommitError."""
    if not re.fullmatch(r"[\w./~^@{}-]+", ref or ""):
        raise CommitError(f"{ref!r} is not a git ref")
    try:
        return _git("rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}")
    except CommitError:
        raise CommitError(f"{ref} is not a commit in this repository") from None


def spec_for_old_engine(spec, fields: set[str]) -> dict:
    """The spec as JSON an older `RunSpec` can read: a field that engine does
    not know is dropped when it holds its default, and refused otherwise."""
    from atisim.run import RunSpec

    data = spec.to_dict()
    defaults = RunSpec._field_defaults
    for name in list(data):
        if name in fields:
            continue
        if name in defaults and data[name] == defaults[name]:
            data.pop(name)
        else:
            raise CommitError(f"that commit's engine has no {name!r}, and this spec sets "
                              f"it to {data[name]!r}")
    return data


def _fields_at(worktree: Path) -> set[str]:
    """The RunSpec field names the worktree's engine knows."""
    code = ("import json, atisim.run as r; print(json.dumps(list(r.RunSpec._fields)))")
    done = subprocess.run([sys.executable, "-c", code], cwd=worktree, capture_output=True,
                          text=True, env={**os.environ, "PYTHONPATH": str(worktree)})
    if done.returncode != 0:
        raise CommitError("that commit's engine does not import: "
                          + (done.stderr.strip().splitlines() or ["no output"])[-1])
    return set(json.loads(done.stdout.strip().splitlines()[-1]))


def fly_at(spec, ref: str, root, on_line=None, on_stage=None) -> Path:
    """Fly `spec` at `ref`; return the run directory it wrote under `root`."""
    on_line = on_line or (lambda line: None)
    on_stage = on_stage or (lambda message: None)
    root = Path(root).resolve()
    sha = resolve(ref)
    try:
        _git("cat-file", "-e", f"{sha}:atisim/cli.py")
    except CommitError:
        added = _git("log", "--diff-filter=A", "--reverse", "--format=%h", "--",
                     "atisim/cli.py").split()
        after = f" Choose a ref at or after {added[0]}, which added it." if added else ""
        raise CommitError(f"{ref} ({sha[:7]}) predates atisim/cli.py, so it has no "
                          f"atisim run command to fly a spec with.{after}") from None
    worktree = root / ".worktrees" / sha[:12]
    worktree.parent.mkdir(parents=True, exist_ok=True)
    if worktree.exists():
        _git("worktree", "remove", "--force", str(worktree))
    _git("worktree", "add", "--detach", str(worktree), sha)
    try:
        on_stage(f"flying at {sha[:7]}")
        spec_path = worktree / "flown-spec.json"
        spec_path.write_text(json.dumps(spec_for_old_engine(spec, _fields_at(worktree)),
                                        indent=2))
        proc = subprocess.Popen(
            [sys.executable, "-m", "atisim.cli", "run", str(spec_path), "--out", str(root)],
            cwd=worktree, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            env={**os.environ, "PYTHONPATH": str(worktree)})
        wrote = None
        lines = []
        for line in proc.stdout:
            line = line.rstrip()
            lines.append(line)
            on_line(line)
            if line.startswith("wrote "):
                wrote = line[len("wrote "):].strip()
        proc.wait()
        if proc.returncode != 0 or wrote is None:
            last = next((ln for ln in reversed(lines) if ln.strip()), "no output")
            raise CommitError(f"the run at {sha[:7]} failed: {last}")
        path = Path(wrote)
        return path if path.is_absolute() else (worktree / path).resolve()
    finally:
        try:
            _git("worktree", "remove", "--force", str(worktree))
        except CommitError:
            pass
