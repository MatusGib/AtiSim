"""The engine worker: runs and analyses flown in a process of their own.

`atisim ui --dev` starts the application in engine-development mode. Its jobs
run here, in one long-lived subprocess, not in the application's thread, so
that editing the engine and pressing **Reload engine** takes effect without
restarting the application: the worker is stopped, and the next job starts a
new one that imports the engine from disk again (plan section 3.4).

JAX's persistent compilation cache is on (in the runs directory's
`.jax-cache/`), so a new worker loads the compiled flight from disk instead of
compiling it again. Every compile is cached, the small ones too: together they
cost as much as the flight's own. Measured on `vortex-hannibal`: 14.5 s for
the first job, 5.4 s for the first job after a reload. The cache is keyed on
the traced program, so an engine edit that changes it compiles afresh.

The protocol is one JSON request per line on stdin, `{"kind": "run" |
"analysis", "spec": {...}, "root": "..."}`, and events on stdout, each a line
starting with `MARK`: `ready`, `stage`, `log`, and one `finish` per request.
Anything else the engine prints is passed on as a log line.
"""

import json
import os
import sys
import time
import traceback
from pathlib import Path

MARK = "@@atisim "


def cache_dir(root) -> Path:
    return Path(root) / ".jax-cache"


def enable_cache(root) -> None:
    import jax

    path = cache_dir(root)
    path.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(path))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)


class _Emitter:
    def __init__(self, out):
        self.out = out

    def send(self, **event):
        self.out.write(MARK + json.dumps(event, default=str) + "\n")
        self.out.flush()

    def stage(self, name, message=None):
        self.send(event="stage", name=name, message=message)

    def log(self, message):
        self.send(event="log", message=message)

    def finish(self, state, **fields):
        self.send(event="finish", state=state, fields=fields)


def main(root: str) -> None:
    out = sys.stdout
    # The engine's own prints must not interleave with the events: they go to
    # stderr, which the application reads as log lines.
    sys.stdout = sys.stderr
    emit = _Emitter(out)
    enable_cache(root)

    from atisim.analysis import artifact
    from atisim.apps import jobs

    emit.send(event="ready", pid=os.getpid(), sha=artifact.git_sha(), started=time.time())
    for line in sys.stdin:
        if not line.strip():
            continue
        request = json.loads(line)
        try:
            perform = jobs.perform_analysis if request["kind"] == "analysis" \
                else jobs.perform_run
            perform(request["spec"], Path(request["root"]), emit)
        except Exception as exc:  # the worker outlives any one job, as the thread does
            emit.log(traceback.format_exc().rstrip())
            emit.finish("failed", error=f"{type(exc).__name__}: {exc}",
                        message=f"failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs")
