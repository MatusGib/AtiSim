"""One background worker thread that flies runs and runs analyses, and a
registry of their state.

A thread, not Dash background callbacks. Those need `diskcache` plus a process
pool, which adds dependencies and re-imports JAX in every process; a thread
shares the already-compiled functions, so the second run of a spec skips the
JAX compile and is fast. One worker and a FIFO queue: runs never compete for
the CPU, and a second Run press queues behind the first.

There is no cancel and no percentage. The first run includes the JAX compile
and can take tens of seconds; the UI shows the stage and the elapsed time,
which are true, rather than a progress bar, which would be invented.

An analysis (`atisim.analyses`) is a job of the second kind. It shares the one
worker, so a run and an analysis never compete for the CPU either. Its stages
are the analysis's own words ("flying member 3 of 8"), recorded under the
state "running".
"""

import itertools
import json
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

from atisim import run
from atisim.analysis import runs as runs_mod
from atisim.units import RAD2DEG

LIVE = ("queued", "validating", "trimming", "flying", "checks", "diagnostics", "writing",
        "running")


class JobRunner:
    """`worker=True` (engine-development mode, `atisim ui --dev`) flies runs and
    analyses in the engine worker subprocess (`atisim.apps.worker`) instead of
    this thread, so `reload()` can pick up an edited engine."""

    def __init__(self, root: Path, worker: bool = False):
        self.root = Path(root)
        self.worker = worker
        self._jobs: dict[str, dict] = {}
        self._queue: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._ids = itertools.count(1)
        self._thread: threading.Thread | None = None
        self._proc: subprocess.Popen | None = None
        self._engine: dict = {}  # the running worker: pid, sha, started, stamp
        self._reload = False
        self._using_worker = threading.Lock()  # held while a job talks to the worker

    # -- public ------------------------------------------------------------
    def submit(self, spec: run.RunSpec) -> str:
        job_id = f"job-{next(self._ids)}"
        now = time.time()
        with self._lock:
            self._jobs[job_id] = {
                "id": job_id, "kind": "run", "spec": spec.to_dict(), "state": "queued",
                "stages": [], "log": [f"{_clock(now)} queued {spec.name}"],
                "error": None, "error_field": None, "path": None, "checks": [],
                "worst_index": None,
                "submitted": now, "finished": None,
            }
        self._queue.put(job_id)
        self._ensure_thread()
        return job_id

    def submit_analysis(self, aspec) -> str:
        """Queue an `analyses.AnalysisSpec`. Returns the job id."""
        job_id = f"job-{next(self._ids)}"
        now = time.time()
        with self._lock:
            self._jobs[job_id] = {
                "id": job_id, "kind": "analysis", "spec": aspec.to_dict(),
                "state": "queued", "stages": [],
                "log": [f"{_clock(now)} queued {aspec.name}"],
                "error": None, "error_field": None, "path": None, "summary": "",
                "checks": [], "worst_index": None,
                "submitted": now, "finished": None,
            }
        self._queue.put(job_id)
        self._ensure_thread()
        return job_id

    def submit_commit(self, spec: run.RunSpec, ref: str) -> str:
        """Queue `spec` to fly at git `ref` (A/B by commit). Returns the job id."""
        job_id = f"job-{next(self._ids)}"
        now = time.time()
        with self._lock:
            self._jobs[job_id] = {
                "id": job_id, "kind": "commit", "spec": spec.to_dict(), "ref": ref,
                "state": "queued", "stages": [],
                "log": [f"{_clock(now)} queued {spec.name} at {ref}"],
                "error": None, "error_field": None, "path": None, "checks": [],
                "worst_index": None, "submitted": now, "finished": None,
            }
        self._queue.put(job_id)
        self._ensure_thread()
        return job_id

    def get(self, job_id: str | None) -> dict | None:
        """A snapshot, safe to hand to a callback. Elapsed times are filled in."""
        with self._lock:
            job = self._jobs.get(job_id or "")
            if job is None:
                return None
            snap = {**job, "stages": [dict(s) for s in job["stages"]],
                    "log": list(job["log"])}
        now = time.time()
        for stage in snap["stages"]:
            stage["elapsed"] = (stage["ended"] or now) - stage["started"]
        snap["elapsed"] = (snap["finished"] or now) - snap["submitted"]
        return snap

    def live(self) -> bool:
        with self._lock:
            return any(j["state"] in LIVE for j in self._jobs.values())

    def latest(self) -> dict | None:
        with self._lock:
            ids = list(self._jobs)
        return self.get(ids[-1]) if ids else None

    # -- worker ------------------------------------------------------------
    def _ensure_thread(self):
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._work, name="atisim-jobs",
                                            daemon=True)
            self._thread.start()

    def _stage(self, job_id: str, name: str, message: str | None = None):
        now = time.time()
        with self._lock:
            job = self._jobs[job_id]
            if job["stages"] and job["stages"][-1]["ended"] is None:
                job["stages"][-1]["ended"] = now
            job["state"] = name
            if name not in ("done", "failed"):
                job["stages"].append({"name": name, "message": message or name,
                                      "started": now, "ended": None})
            job["log"].append(f"{_clock(now)} {message or name}")

    def _log(self, job_id: str, message: str):
        with self._lock:
            self._jobs[job_id]["log"].append(f"{_clock(time.time())} {message}")

    def _finish(self, job_id: str, state: str, **fields):
        self._stage(job_id, state, fields.pop("message", state))
        with self._lock:
            self._jobs[job_id].update(fields, finished=time.time())

    def _work(self):
        while True:
            job_id = self._queue.get()
            try:
                with self._lock:
                    kind = self._jobs[job_id].get("kind", "run")
                if self.worker and kind in ("run", "analysis"):
                    self._remote(job_id, kind)
                else:
                    {"analysis": self._analyse, "commit": self._at_commit}.get(
                        kind, self._fly)(job_id)
            except Exception as exc:  # the worker must outlive any one run
                self._log(job_id, traceback.format_exc().rstrip())
                self._finish(job_id, "failed", error=f"{type(exc).__name__}: {exc}",
                             message=f"failed: {type(exc).__name__}: {exc}")
            finally:
                self._queue.task_done()

    def _fly(self, job_id: str):
        with self._lock:
            spec = self._jobs[job_id]["spec"]
        perform_run(spec, self.root, _Local(self, job_id))

    # -- the engine worker (development mode) -------------------------------
    def engine(self) -> dict:
        """What the status bar says about the engine: where jobs run, the
        worker's commit and start time, and whether the engine's source has
        changed on disk since it started."""
        if not self.worker:
            return {"mode": "in-process"}
        with self._lock:
            info = dict(self._engine)
            alive = self._proc is not None and self._proc.poll() is None
        info.update(mode="subprocess", alive=alive)
        info["changed"] = bool(alive and engine_stamp() > info.get("stamp", 0.0))
        return info

    def reload(self) -> None:
        """Stop the worker; the next job starts a new one on the engine as it is
        on disk now. A job in flight is not interrupted: its worker is stopped
        when it finishes."""
        with self._lock:
            self._reload = True
        if self._using_worker.acquire(blocking=False):
            try:
                self._stop_worker()
            finally:
                self._using_worker.release()

    def _stop_worker(self):
        with self._lock:
            proc, self._proc, self._reload = self._proc, None, False
            self._engine = {}
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    def _start_worker(self, job_id: str | None = None):
        """Start the worker; what it prints before it is ready (an import error
        in an edited engine, say) goes to the job's log."""
        import atisim

        repo = Path(atisim.__file__).resolve().parent.parent
        stamp = engine_stamp()
        proc = subprocess.Popen(
            [sys.executable, "-u", "-m", "atisim.apps.worker", str(self.root)],
            cwd=repo, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
            env={**os.environ, "PYTHONPATH": str(repo)})
        ready = self._read_event(proc, job_id)
        if ready is None or ready.get("event") != "ready":
            proc.kill()
            proc.wait()
            raise RuntimeError("the engine worker did not start; its output is in the log")
        with self._lock:
            self._proc = proc
            self._engine = {"pid": ready["pid"], "sha": ready["sha"],
                            "started": ready["started"], "stamp": stamp}

    def _read_event(self, proc, job_id):
        """The next event from the worker; its other output goes to the job's log."""
        from atisim.apps.worker import MARK

        for line in proc.stdout:
            line = line.rstrip("\n")
            if line.startswith(MARK):
                return json.loads(line[len(MARK):])
            if job_id is not None and line.strip():
                self._log(job_id, line)
        return None

    def _remote(self, job_id: str, kind: str):
        with self._using_worker:
            self._remote_locked(job_id, kind)

    def _remote_locked(self, job_id: str, kind: str):
        with self._lock:
            pending, spec = self._reload, self._jobs[job_id]["spec"]
        if pending:
            self._stop_worker()
        if self._proc is None or self._proc.poll() is not None:
            self._stage(job_id, "validating", "starting the engine worker")
            self._start_worker(job_id)
        proc = self._proc
        proc.stdin.write(json.dumps({"kind": kind, "spec": spec,
                                     "root": str(self.root)}) + "\n")
        proc.stdin.flush()
        while True:
            event = self._read_event(proc, job_id)
            if event is None:
                self._stop_worker()
                self._finish(job_id, "failed", error="the engine worker stopped",
                             message="failed: the engine worker stopped")
                return
            if event["event"] == "stage":
                self._stage(job_id, event["name"], event.get("message"))
            elif event["event"] == "log":
                self._log(job_id, event["message"])
            elif event["event"] == "finish":
                fields = dict(event.get("fields") or {})
                self._finish(job_id, event["state"], **fields)
                with self._lock:
                    reload_now = self._reload
                if reload_now:
                    self._stop_worker()
                return

    def _analyse(self, job_id: str):
        with self._lock:
            spec = self._jobs[job_id]["spec"]
        perform_analysis(spec, self.root, _Local(self, job_id))

    def _at_commit(self, job_id: str):
        """Fly the job's spec with the engine at another commit.

        A git worktree at the ref, under the runs directory's `.worktrees/`, runs
        `atisim run` in a subprocess with the worktree's own package first on the
        path; the run lands in the runs directory beside the others, named with
        that commit's SHA. The worktree is removed afterwards, pass or fail.
        """
        from atisim.analysis import commits

        with self._lock:
            spec = run.RunSpec.from_json(self._jobs[job_id]["spec"])
            ref = self._jobs[job_id]["ref"]
        self._stage(job_id, "running", f"checking out {ref} in a worktree")
        try:
            path = commits.fly_at(spec, ref, self.root,
                                  on_line=lambda line: self._log(job_id, line),
                                  on_stage=lambda m: self._stage(job_id, "running", m))
        except commits.CommitError as exc:
            self._finish(job_id, "failed", error=str(exc), message=f"failed: {exc}")
            return
        self._finish(job_id, "done", path=str(path), message=f"wrote {path.name}")


class _Local:
    """Where `perform_run` and `perform_analysis` report, for an in-process job."""

    def __init__(self, runner: "JobRunner", job_id: str):
        self.runner, self.job_id = runner, job_id

    def stage(self, name: str, message: str | None = None):
        self.runner._stage(self.job_id, name, message)

    def log(self, message: str):
        self.runner._log(self.job_id, message)

    def finish(self, state: str, **fields):
        self.runner._finish(self.job_id, state, **fields)


def perform_run(spec_data, root: Path, report) -> None:
    """Validate, fly and write one run, reporting its stages to `report`.

    The same code in both places a job can run: the app's own thread, and the
    engine worker (`atisim.apps.worker`), which reports by printing events.
    """
    spec = run.RunSpec.from_json(spec_data)
    report.stage("validating")
    issues = run.validate(spec)
    for issue in issues:
        report.log(f"{issue.level}: {issue.message}")
    problems = [i for i in issues if i.level == "error"]
    if problems:
        report.finish("failed", error=problems[0].message, error_field=problems[0].field,
                      message=f"failed: {problems[0].message}")
        return
    stage_messages = {
        "trimming": f"trimming {spec.aircraft} at {spec.airspeed_mps:.1f} m/s, "
                    f"{spec.altitude_m:.0f} m",
        "flying": f"flying {run.KIND_LABELS.get(spec.wind.kind, spec.wind.kind)}, "
                  f"dt {spec.dt:g} s (the first run includes the JAX compile)",
        "checks": "running the checks",
        "diagnostics": "high fidelity: probing every sample for the diagnostics",
    }
    try:
        flown = run.fly(spec, on_stage=lambda name: report.stage(
            name, stage_messages.get(name)))
    except (run.TrimError, run.SpecError) as exc:
        # A trim failure is the flight condition's: that is where the fix goes.
        field = "airspeed_mps" if isinstance(exc, run.TrimError) else None
        report.finish("failed", error=str(exc), error_field=field, message=f"failed: {exc}")
        return
    trim = flown.meta["trim"]
    report.log(f"trim alpha {trim['alpha_rad'] * RAD2DEG:.3f} deg, "
               f"elevator {trim['elevator_rad'] * RAD2DEG:.3f} deg, "
               f"throttle {trim['throttle']:.4f}")
    report.log(f"flew {flown.meta['integrator']['n_steps']} steps")
    report.stage("writing", f"writing the artifact under {root}")
    path = run.save(flown, root)
    checks = [c.as_dict() for c in flown.report]
    failed = [c["name"] for c in checks if c["passed"] is False]
    report.log("checks: " + (f"FAILED {', '.join(failed)}" if failed else "no gate failed"))
    report.finish("done", path=str(path), checks=checks,
                  worst_index=runs_mod.worst_gate_index(checks),
                  timing={k: round(v, 3) for k, v in flown.timing.items()},
                  message=f"wrote {path.name}")


def perform_analysis(spec_data, root: Path, report) -> None:
    """Validate and perform one analysis, reporting to `report` (see `perform_run`)."""
    from atisim import analyses
    from atisim.analysis import report as report_mod

    aspec = analyses.AnalysisSpec.from_json(spec_data)
    report.stage("validating")
    issues = analyses.validate(aspec)
    for issue in issues:
        report.log(f"{issue.level}: {issue.message}")
    problems = [i for i in issues if i.level == "error"]
    if problems:
        report.finish("failed", error=problems[0].message, error_field=problems[0].field,
                      message=f"failed: {problems[0].message}")
        return
    try:
        path = analyses.perform(aspec, root, on_stage=lambda message: report.stage(
            "running", message), on_log=report.log)
    except (run.TrimError, run.SpecError, analyses.AnalysisError) as exc:
        report.finish("failed", error=str(exc), message=f"failed: {exc}")
        return
    summary = report_mod.read_report(path).meta.get("summary", "")
    report.log(summary)
    report.finish("done", path=str(path), summary=summary, message=f"wrote {path.name}")


def engine_stamp() -> float:
    """The newest modification time of the engine's source: `atisim/**/*.py`
    outside the application and the tests, which a reload cannot change."""
    import atisim

    package = Path(atisim.__file__).resolve().parent
    newest = 0.0
    for path in package.rglob("*.py"):
        rel = path.relative_to(package).parts
        if rel and rel[0] in ("apps", "tests"):
            continue
        newest = max(newest, path.stat().st_mtime)
    return newest


def _clock(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S")
