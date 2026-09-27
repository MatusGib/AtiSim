"""The analysis UI's original entry point, kept: it opens the app on Results.

    .venv/Scripts/python.exe -m atisim.apps.sweep runs/analysis

The deep dive this command used to build on its own now lives in
`atisim.apps.results`, inside the app shell (`atisim.apps.shell`), with its
behaviour and every component id unchanged. `atisim ui runs/analysis` opens the
same app on Start.
"""

import argparse
from pathlib import Path

from atisim.apps.results import Loaded  # noqa: F401  -- the name this module exported
from atisim.apps.shell import build_app as _build_app


def build_app(root: Path):
    """The whole app, landing on Results."""
    return _build_app(root, landing="/results")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="?", default=Path("runs/analysis"),
                        help="directory of run artifacts")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    build_app(args.runs).run(host="127.0.0.1", port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
