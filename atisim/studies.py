"""The research scripts in `scripts/`, run as they are: the study adapter.

Each script is its own program, and the application does not rebuild them as
run kinds. This module lists them and runs one as a
subprocess, unchanged; `atisim.analyses` wraps that as the `study` analysis, so
the application, the CLI and the job runner run a study the way they run any
other analysis, and Results shows what it printed and what it wrote.

A script is a STUDY when running it does something: it has an
`if __name__ == "__main__":` guard, or it runs statements at its top level
other than putting a directory on `sys.path`. The others are helper modules
that the studies import, and
running one would do nothing.

What a study wrote is found by looking: every file under the repository with a
kept suffix whose size or modification time changed while the script ran is
copied into the study's directory, under `files/` at its path relative to the
repository. The script writes where it always writes; the copy is what the
study keeps. A script that needs data the public tree does not hold fails with
its own message, and the study records that failure as it is.

No Dash here and no JAX: the script is another process.
"""

import ast
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Callable, NamedTuple

import atisim

REPO = Path(atisim.__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"

# What a study keeps of what it wrote. Images are shown, tables previewed, and
# the rest listed.
IMAGES = (".png", ".svg")
KEPT = IMAGES + (".csv", ".json", ".md", ".txt")
MAX_FILE_BYTES = 50 * 1024 * 1024

# Never looked into for outputs: version control, caches, and tool folders.
SKIPPED_DIRS = {".git", "__pycache__", "node_modules", ".worktrees", ".jax-cache",
                ".pytest_cache", ".ruff_cache", ".venv", ".impeccable", ".claude"}


class Script(NamedTuple):
    name: str  # the file's stem: `sanity` for scripts/sanity.py
    path: Path
    description: str  # the docstring's first line, or the header comment's


def _does_something(tree: ast.Module) -> bool:
    for node in tree.body:
        if isinstance(node, ast.If) and "__main__" in ast.unparse(node.test):
            return True
        if isinstance(node, ast.Expr):
            if isinstance(node.value, ast.Constant):
                continue  # a docstring
            if "sys.path" in ast.unparse(node.value):
                continue  # a helper putting its directory on the path
            return True
        if isinstance(node, (ast.For, ast.While, ast.With, ast.Try)):
            return True
    return False


def _description(text: str, tree: ast.Module) -> str:
    doc = ast.get_docstring(tree)
    if doc:
        line = doc.strip().splitlines()[0].strip()
    else:
        comments = []
        for raw in text.splitlines():
            if not raw.startswith("#"):
                break
            comments.append(raw.lstrip("#").strip())
        line = next((c for c in comments if c and not c.startswith("!")), "")
    return line[:1].upper() + line[1:] if line else "No description."


@lru_cache(maxsize=4)
def _scan(directory: Path) -> tuple:
    found = []
    for path in sorted(directory.glob("*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        if _does_something(tree):
            found.append(Script(path.stem, path, _description(text, tree)))
    return tuple(found)


def scripts(directory: Path | None = None) -> tuple:
    """Every study in `scripts/`, by name."""
    return _scan(Path(directory or SCRIPTS))


def names(directory: Path | None = None) -> tuple:
    return tuple(s.name for s in scripts(directory))


def helpers(directory: Path | None = None) -> tuple:
    """The modules in `scripts/` that are not studies."""
    directory = Path(directory or SCRIPTS)
    studies = set(names(directory))
    return tuple(p.stem for p in sorted(directory.glob("*.py")) if p.stem not in studies)


def find(name: str, directory: Path | None = None) -> Script:
    for s in scripts(directory):
        if s.name == name:
            return s
    raise KeyError(name)


def about(script: Script, limit: int = 1200) -> list[str]:
    """The docstring's paragraphs after its first line, up to `limit` characters.

    What a reader needs before running a script they did not write: what it
    flies, against what, and what it prints. Empty when the docstring is one
    line or missing.
    """
    try:
        doc = ast.get_docstring(ast.parse(script.path.read_text(encoding="utf-8",
                                                                  errors="replace")))
    except (OSError, SyntaxError):
        return []
    if not doc:
        return []
    paragraphs = [_joined(p) for p in doc.strip().split("\n\n")]
    out, used = [], 0
    for p in paragraphs[1:]:
        if not p or used + len(p) > limit:
            break
        out.append(p)
        used += len(p)
    return out


_ITEM = re.compile(r"^\s*(\d+[.)]?|[-*\u2022])\s")


def _joined(paragraph: str) -> str:
    """Re-flow a docstring paragraph, keeping a line that starts a list item."""
    lines = []
    for raw in paragraph.splitlines():
        text = " ".join(raw.split())
        if not text:
            continue
        if lines and not _ITEM.match(raw):
            lines[-1] += " " + text
        else:
            lines.append(text)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Topics: where a reader finds a study
# ---------------------------------------------------------------------------


class Topic(NamedTuple):
    key: str
    title: str
    blurb: str
    names: tuple  # script stems, in reading order


# The scripts by topic, in reading order. A script named nowhere here is listed
# under OTHER, so a new script is never hidden.
TOPICS: tuple = (
    Topic("cat", "Clear-air turbulence cases",
          "The published encounters flown, set beside the papers' records",
          ("cat_validation", "cat_ensemble", "cat_spectra", "vortex", "lateral")),
    Topic("intensity", "Turbulence intensity",
          "MIL-F-8785C: turbulence intensity against altitude and exceedance probability",
          ("digitise_mil_f_8785c_fig7",)),
    Topic("jsbsim", "JSBSim cross-code",
          "AtiSim against JSBSim: the reference data and the vortex comparison",
          ("vortex_compare", "gen_jsbsim_747", "gen_jsbsim_reference",
           "gen_jsbsim_vortex_reference")),
    Topic("aircraft", "Aircraft data",
          "The 747's derivatives from CR-2144: digitised and checked",
          ("cr2144_digitisation_crosscheck", "checkpoint")),
    Topic("fields", "Other wind fields", "Mountain lee waves and microbursts",
          ("leewave", "microburst")),
    Topic("tools", "Tools and quick checks",
          "The sanity checks, re-analysis, gain tuning and interactive flying",
          ("sanity", "analyse", "tune", "fly")),
)
OTHER = Topic("other", "Other scripts", "Scripts no topic names yet", ())
_TOPIC_OF = {name: t for t in TOPICS for name in t.names}


def topic_of(name: str) -> Topic:
    return _TOPIC_OF.get(name, OTHER)


def by_topic(directory: Path | None = None) -> list[tuple[Topic, list[Script]]]:
    """Every study under its topic, in the topics' order; OTHER last, if used."""
    found = {s.name: s for s in scripts(directory)}
    out = []
    for t in TOPICS:
        listed = [found[n] for n in t.names if n in found]
        if listed:
            out.append((t, listed))
    rest = [s for n, s in found.items() if n not in _TOPIC_OF]
    if rest:
        out.append((OTHER, rest))
    return out


# ---------------------------------------------------------------------------
# Running one
# ---------------------------------------------------------------------------


class Outcome(NamedTuple):
    returncode: int | None  # None: stopped at the time limit
    elapsed_s: float
    log: str
    outputs: tuple  # (relative path, bytes) of every file copied, repository-relative
    skipped: tuple  # (relative path, reason) of what was written but not kept


def _snapshot(repo: Path, exclude: tuple) -> dict:
    seen = {}
    for folder, dirs, files in os.walk(repo):
        here = Path(folder)
        dirs[:] = [d for d in dirs if d not in SKIPPED_DIRS
                   and not any((here / d) == e for e in exclude)]
        for f in files:
            if not f.lower().endswith(KEPT):
                continue
            path = here / f
            try:
                st = path.stat()
            except OSError:
                continue
            seen[path] = (st.st_mtime_ns, st.st_size)
    return seen


def run_script(script: Script, directory, on_line: Callable[[str], None] | None = None,
               minutes: float = 30.0, args=(), repo: Path | None = None) -> Outcome:
    """Run `script` with `args` and the repository as its working directory; copy
    what it wrote into `directory/files/`. Stops it after `minutes`."""
    on_line = on_line or (lambda line: None)
    repo = Path(repo or REPO).resolve()
    directory = Path(directory).resolve()
    before = _snapshot(repo, (directory,))
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(
        [str(repo)] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]),
        "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1"}
    started = time.time()
    proc = subprocess.Popen([sys.executable, "-u", str(script.path), *args], cwd=repo, env=env,
                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, errors="replace")
    stopped = threading.Event()

    def stop():
        stopped.set()
        proc.kill()

    timer = threading.Timer(minutes * 60.0, stop)
    timer.start()
    lines = []
    try:
        for line in proc.stdout:
            line = line.rstrip("\n")
            lines.append(line)
            on_line(line)
        proc.wait()
    finally:
        timer.cancel()
    elapsed = time.time() - started
    after = _snapshot(repo, (directory,))
    outputs, skipped = [], []
    for path, stamp in sorted(after.items()):
        if before.get(path) == stamp:
            continue
        rel = path.relative_to(repo).as_posix()
        if stamp[1] > MAX_FILE_BYTES:
            skipped.append((rel, f"{stamp[1] / 1e6:.0f} MB, over the "
                                 f"{MAX_FILE_BYTES // 1024 ** 2} MB a study keeps"))
            continue
        target = directory / "files" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        outputs.append((rel, stamp[1]))
    return Outcome(None if stopped.is_set() else proc.returncode, elapsed, "\n".join(lines),
                   tuple(outputs), tuple(skipped))
