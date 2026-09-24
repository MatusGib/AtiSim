"""Sphinx configuration for the AtiSim documentation site.

Build from the repository root:

    python -m pip install -e .[docs]
    python -m sphinx -b html docs docs/_build/html
"""
import os
import sys

# Import `atisim` from THIS tree, not from wherever an editable install points,
# so the API reference always documents the code beside it.
sys.path.insert(0, os.path.abspath(".."))

project = "AtiSim"
author = "Mateusz Wozniak"
copyright = "2026, Mateusz Wozniak"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
]

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
root_doc = "index"

myst_enable_extensions = ["dollarmath", "colon_fence", "deflist"]
myst_heading_anchors = 3

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_default_options = {"members": True, "show-inheritance": True}
# Optional extras: the analysis UI (dash, plotly, pyarrow) and the JSBSim
# reference engine. Mocked so the API reference builds from the runtime install.
autodoc_mock_imports = ["dash", "plotly", "pyarrow", "jsbsim"]


html_theme = "furo"
html_title = "AtiSim"
html_static_path = []


# The package's docstrings are written as plain prose with ASCII maths -- `|v|`
# for a magnitude, `|alpha|`, names ending in an underscore -- not as
# reStructuredText. Read as reST, `|v|` becomes an undefined substitution and
# `c_` a link target, and both render as red error text. Escape exactly those
# two constructs, OUTSIDE backtick spans, so the docstrings render as written
# without rewriting 31,000 lines of source into a markup they were not meant for.
import re

_BACKTICKS = re.compile(r"(``.*?``|`[^`]*`)")
_TRAILING_UNDERSCORE = re.compile(r"(?<=\w)_(?=\W|$)")


def _escape_prose(text: str) -> str:
    return _TRAILING_UNDERSCORE.sub(r"\\_", text.replace("|", r"\|"))


def _escape_line(line: str) -> str:
    parts = _BACKTICKS.split(line)
    return "".join(p if _BACKTICKS.fullmatch(p) else _escape_prose(p) for p in parts)


# The docstrings also align small tables and equations by column with a hanging
# indent -- "Frames   NED ..." -- which reST reads as an unexpected indentation.
# Such a block becomes a LITERAL block, so its columns stay aligned: always when
# it follows a paragraph line directly (the reST error case), and when it follows
# a blank line only if it is column-aligned (otherwise it is a valid block quote
# and is left alone). A list item's continuation lines are never touched.
_ALIGNED = re.compile(r"\S\s{2,}\S")
_BULLET = re.compile(r"^\s*([-*+]|\d+[.)])\s")


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _plain_docstrings(app, what, name, obj, options, lines):
    out = []
    literal_at = None   # indentation that opened the current literal block
    prev = ""           # the last non-blank line seen
    for line in lines:
        text = line.strip()
        if literal_at is not None:
            if not text or _indent(line) > literal_at:
                out.append(line)
                prev = line if text else prev
                continue
            literal_at = None
        opens = (text and prev.strip() and _indent(line) > _indent(prev)
                 and not _BULLET.match(prev))
        direct = bool(out) and out[-1].strip() != ""
        if opens and (direct or _ALIGNED.search(text)):
            if direct:
                out.append("")
            out += [" " * _indent(prev) + "::", "", line]
            literal_at, prev = _indent(prev), line
            continue
        out.append(_escape_line(line))
        if text:
            if text.endswith("::"):
                literal_at = _indent(line)
            prev = line
    lines[:] = out


def setup(app):
    app.connect("autodoc-process-docstring", _plain_docstrings)
