"""The `atisim` command: fly a run, run an analysis, list presets and runs, open
the app.

    atisim run --preset vortex-hannibal --out runs/
    atisim run spec.json --set lead_in=20 --set wind.r0=150
    atisim run --preset updraft --fidelity high
    atisim presets
    atisim analyses
    atisim analyse modes --set aircraft=boeing737
    atisim analyse ensemble --base dryden --set members=16 --out runs/
    atisim studies
    atisim study sanity --out runs/
    atisim list runs/
    atisim ui runs/

`main(argv)` takes its arguments as a list so the tests call it directly. Dash
is imported inside `ui` only: every other subcommand works without the `ui`
extra, and `run` needs only pyarrow, to write the artifact.
"""

import argparse
import sys
from pathlib import Path

UI_HINT = 'The app needs the `ui` extra. Install it with: pip install -e ".[ui]"'


def _spec_from_args(args):
    from atisim import run

    if args.preset:
        spec = run.preset(args.preset)
    else:
        spec = run.RunSpec.from_json(Path(args.spec).read_text())
    for assignment in args.set or []:
        spec = run.apply_set(spec, assignment)
    if getattr(args, "fidelity", None):
        spec = spec._replace(fidelity=args.fidelity)
    return spec


def _cmd_run(args) -> int:
    from atisim import run

    try:
        spec = _spec_from_args(args)
    except (KeyError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    issues = run.validate(spec)
    for issue in issues:
        print(f"{issue.level}: {issue.field}: {issue.message}", file=sys.stderr)
    if any(i.level == "error" for i in issues):
        return 2
    try:
        flown = run.fly(spec, on_stage=lambda name: print(f"{name} ...", flush=True))
        print("writing ...", flush=True)
    except run.TrimError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    try:
        path = run.save(flown, args.out)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    failed = [c.name for c in flown.report if c.passed is False]
    print(f"wrote {path}")
    print(f"checks: {'FAILED ' + ', '.join(failed) if failed else 'no gate failed'}")
    return 0


def _cmd_presets(args) -> int:
    from atisim import run

    width = max(len(name) for name in run.PRESETS)
    for name, spec in run.PRESETS.items():
        print(f"{name:<{width}}  {spec.aircraft:<10}  {spec.wind.source}")
    print(f"{'blank':<{width}}  {run.BLANK.aircraft:<10}  still air; set every parameter yourself")
    return 0


def _cmd_analyses(args) -> int:
    from atisim import analyses

    width = max(len(key) for key in analyses.ANALYSES)
    for family, _ in analyses.FAMILIES:
        print(family)
        for key, a in analyses.ANALYSES.items():
            if a.family == family:
                base = "  (on a run: --base)" if a.needs_base else ""
                print(f"  {key:<{width}}  {a.description}{base}")
    return 0


def _cmd_analyse(args) -> int:
    from atisim import analyses, run

    try:
        if args.spec:
            aspec = analyses.AnalysisSpec.from_json(Path(args.spec).read_text())
        else:
            if args.analysis not in analyses.ANALYSES:
                raise ValueError(f"unknown analysis {args.analysis!r}; see `atisim analyses`")
            base = None
            if args.base:
                base = (run.preset(args.base) if args.base in run.PRESETS
                        else run.RunSpec.from_json(Path(args.base).read_text()))
            aspec = analyses.default(args.analysis, base)
        for assignment in args.set or []:
            aspec = analyses.apply_set(aspec, assignment)
    except (KeyError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    issues = analyses.validate(aspec)
    for issue in issues:
        print(f"{issue.level}: {issue.field}: {issue.message}", file=sys.stderr)
    if any(i.level == "error" for i in issues):
        return 2
    try:
        path = analyses.perform(aspec, args.out,
                                on_stage=lambda name: print(f"{name} ...", flush=True))
    except (run.TrimError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    from atisim.analysis import report

    print(f"wrote {path}")
    print(report.read_report(path).meta["summary"])
    return 0


def _cmd_studies(args) -> int:
    from atisim import studies

    found = studies.scripts()
    width = max(len(s.name) for s in found)
    for s in found:
        print(f"  {s.name:<{width}}  {s.description}")
    print(f"\n{len(found)} studies. Run one: atisim study NAME [-- ARGUMENTS]")
    return 0


def _cmd_study(args) -> int:
    import shlex

    sets = [f"script={args.script}", f"arguments={shlex.join(args.arguments or [])}"]
    if args.minutes is not None:
        sets.append(f"minutes={args.minutes}")
    return _cmd_analyse(argparse.Namespace(
        spec=None, analysis="study", base=None, out=args.out,
        set=sets + list(args.set or [])))


def _cmd_list(args) -> int:
    from atisim.analysis import runs as runs_mod

    rows = runs_mod.scan(args.runs)
    if not rows:
        print(f"no runs under {args.runs}. Fly one with: "
              f"atisim run --preset vortex-hannibal --out {args.runs}")
        return 0
    for row in rows:
        if row.error:
            print(f"{row.name}  ERROR: {row.error}")
            continue
        stale = "  stale" if row.stale else ""
        print(f"{row.name}  {row.kind}  {row.aircraft}  {row.created_local}  "
              f"{row.verdict}{stale}")
    return 0


def _cmd_ui(args) -> int:
    try:
        import dash  # noqa: F401
        import dash_iconify  # noqa: F401
        import dash_mantine_components  # noqa: F401
    except ImportError:
        print(UI_HINT, file=sys.stderr)
        return 2
    from atisim.apps import shell

    app = shell.build_app(Path(args.runs), dev=args.dev, warm=True)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"AtiSim is at {url}  (Ctrl+C to stop)"
          + ("; engine-development mode: jobs run in the engine worker" if args.dev else ""))
    if not args.no_browser:
        import threading
        import webbrowser

        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=args.port, debug=False)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="atisim", description="Fly, list and inspect AtiSim turbulence-encounter runs.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="fly one run and write its artifact")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--preset", help="a preset name (see `atisim presets`)")
    source.add_argument("spec", nargs="?", help="a RunSpec JSON file")
    p.add_argument("--out", default="runs", help="runs directory (default: runs)")
    p.add_argument("--set", action="append", metavar="KEY=VALUE",
                   help="change one field, e.g. lead_in=20 or wind.r0=150; repeatable")
    p.add_argument("--fidelity", choices=("standard", "high"),
                   help="high also writes diagnostics.parquet: every coefficient term, "
                        "force, moment and derivative per sample (the flight is the same)")
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser("presets", help="list the presets and their sources")
    p.set_defaults(func=_cmd_presets)

    p = sub.add_parser("analyses", help="list the analyses: ensembles, autopilot, modes, "
                                        "gust response, verification")
    p.set_defaults(func=_cmd_analyses)

    p = sub.add_parser("analyse", help="run one analysis and write its report")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("analysis", nargs="?", help="an analysis key (see `atisim analyses`)")
    source.add_argument("--spec", help="an AnalysisSpec JSON file")
    p.add_argument("--base", help="the run it analyses: a preset name or a RunSpec JSON file")
    p.add_argument("--out", default="runs", help="runs directory (default: runs)")
    p.add_argument("--set", action="append", metavar="KEY=VALUE",
                   help="change one parameter, e.g. members=16 or base.dt=0.005; repeatable")
    p.set_defaults(func=_cmd_analyse)

    p = sub.add_parser("studies", help="list the research scripts that run as studies")
    p.set_defaults(func=_cmd_studies)

    p = sub.add_parser("study", help="run one script from scripts/ as a study and keep "
                                     "its output")
    p.add_argument("script", help="a script's name without .py (see `atisim studies`)")
    p.add_argument("arguments", nargs="*",
                   help="the script's own arguments, after --")
    p.add_argument("--out", default="runs", help="runs directory (default: runs)")
    p.add_argument("--minutes", type=float, help="the time limit (default 30)")
    p.add_argument("--set", action="append", metavar="KEY=VALUE", help=argparse.SUPPRESS)
    p.set_defaults(func=_cmd_study)

    p = sub.add_parser("list", help="list the runs in a directory, newest first")
    p.add_argument("runs", nargs="?", default="runs")
    p.set_defaults(func=_cmd_list)

    p = sub.add_parser("ui", help="open the app in a browser (needs the `ui` extra)")
    p.add_argument("runs", nargs="?", default="runs")
    p.add_argument("--port", type=int, default=8050)
    p.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    p.add_argument("--dev", action="store_true",
                   help="engine-development mode: fly in a worker process that "
                        "Reload engine restarts on the edited engine")
    p.set_defaults(func=_cmd_ui)
    return parser


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = []
    if argv[:1] == ["study"] and "--" in argv:
        # Everything after `--` is the script's own, wherever the options stand.
        cut = argv.index("--")
        argv, extra = argv[:cut], argv[cut + 1:]
    args = build_parser().parse_args(argv)
    if extra:
        args.arguments = list(args.arguments or []) + extra
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
