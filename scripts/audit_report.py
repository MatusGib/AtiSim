"""Generate the audit PDF FROM AUDIT.md.

`AUDIT_PROMPT.md` requires the report to be *generated from* the markdown, not
written directly, so that the PDF cannot drift from the document the findings
actually live in. This script is the only thing that writes the PDF, and it
reads nothing else.

    python scripts/audit_report.py [--out docs/summary/audit-report.pdf]

Deliberately a small hand-rolled markdown subset rather than a dependency: the
project's runtime list is four packages and `pyproject.toml` explains at length
why it stays that way. `pymupdf` is already in the `dev` extra (it is how the
source PDFs are read), and `fitz.Story` renders the HTML.
"""

from __future__ import annotations

import argparse
import html
import pathlib
import re

import fitz

CSS = """
body   { font-family: sans-serif; font-size: 9.5px; line-height: 1.45; }
h1     { font-size: 20px; margin-top: 14px; margin-bottom: 6px; }
h2     { font-size: 15px; margin-top: 14px; margin-bottom: 5px; }
h3     { font-size: 12px; margin-top: 11px; margin-bottom: 4px; }
h4     { font-size: 10px; margin-top: 9px;  margin-bottom: 3px; }
p      { margin-top: 3px; margin-bottom: 3px; }
li     { margin-top: 2px; margin-bottom: 2px; }
code   { font-family: monospace; font-size: 8.5px; }
pre    { font-family: monospace; font-size: 8px; margin-top: 4px; margin-bottom: 4px; }
table  { font-size: 8px; margin-top: 5px; margin-bottom: 7px; }
th     { font-family: sans-serif; font-size: 8px; text-align: left; }
td     { font-family: sans-serif; font-size: 8px; }
blockquote { margin-left: 12px; font-size: 9px; }
hr     { margin-top: 8px; margin-bottom: 8px; }
"""

_INLINE = (
    (re.compile(r"`([^`]+)`"), r"<code>\1</code>"),
    (re.compile(r"\*\*([^*]+)\*\*"), r"<b>\1</b>"),
    (re.compile(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])"), r"<i>\1</i>"),
    (re.compile(r"~~([^~]+)~~"), r"<i>\1</i>"),
    (re.compile(r"\[([^\]]+)\]\(([^)]+)\)"), r"\1"),
)


def inline(text: str) -> str:
    """Escape, then apply the inline markers. Order matters: escaping first
    means a literal `<` in the source cannot open a tag."""
    out = html.escape(text, quote=False)
    for pattern, repl in _INLINE:
        out = pattern.sub(repl, out)
    return out


def _split_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def to_html(md: str) -> str:
    """Markdown subset -> HTML: headings, tables, fenced code, lists,
    blockquotes, rules, paragraphs."""
    lines = md.splitlines()
    out: list[str] = []
    i, n = 0, len(lines)
    list_open = False

    def close_list() -> None:
        nonlocal list_open
        if list_open:
            out.append("</ul>")
            list_open = False

    while i < n:
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):
            close_list()
            i += 1
            body = []
            while i < n and not lines[i].strip().startswith("```"):
                body.append(html.escape(lines[i], quote=False))
                i += 1
            i += 1
            out.append("<pre>" + "<br/>".join(body) + "</pre>")
            continue

        # A table needs a header row and a |---|---| separator underneath.
        if (stripped.startswith("|") and i + 1 < n
                and re.fullmatch(r"\|[\s:|-]+\|", lines[i + 1].strip())):
            close_list()
            header = _split_row(stripped)
            i += 2
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                rows.append(_split_row(lines[i].strip()))
                i += 1
            out.append("<table border='1' cellpadding='3' cellspacing='0'>")
            out.append("<tr>" + "".join(f"<th>{inline(c)}</th>" for c in header) + "</tr>")
            for row in rows:
                row = (row + [""] * len(header))[: len(header)]
                out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in row) + "</tr>")
            out.append("</table>")
            continue

        if not stripped:
            close_list()
            i += 1
            continue

        if re.fullmatch(r"(-{3,}|\*{3,}|_{3,})", stripped):
            close_list()
            out.append("<hr/>")
            i += 1
            continue

        m = re.match(r"(#{1,6})\s+(.*)", stripped)
        if m:
            close_list()
            level = min(len(m.group(1)), 4)
            out.append(f"<h{level}>{inline(m.group(2))}</h{level}>")
            i += 1
            continue

        if stripped.startswith(">"):
            close_list()
            # Consecutive `>` lines are ONE quote. Treated line-by-line they
            # render as a stack of separate blocks with a paragraph gap between
            # each, which reads as a list rather than as a quotation.
            block: list[str] = []
            while i < n and lines[i].strip().startswith(">"):
                text = lines[i].strip().lstrip(">").strip()
                if not text:
                    block.append("<br/>")
                else:
                    block.append(inline(text))
                i += 1
            out.append("<blockquote>" + " ".join(block) + "</blockquote>")
            continue

        m = re.match(r"[-*+]\s+(.*)", stripped) or re.match(r"\d+[.)]\s+(.*)", stripped)
        if m:
            if not list_open:
                out.append("<ul>")
                list_open = True
            out.append(f"<li>{inline(m.group(1))}</li>")
            i += 1
            continue

        close_list()
        para = [stripped]
        i += 1
        while i < n and lines[i].strip() and not re.match(
                r"(#{1,6}\s|\||```|>|[-*+]\s|\d+[.)]\s|-{3,}$)", lines[i].strip()):
            para.append(lines[i].strip())
            i += 1
        out.append(f"<p>{inline(' '.join(para))}</p>")

    close_list()
    return "<html><body>" + "\n".join(out) + "</body></html>"


def render(html_text: str, out_path: pathlib.Path) -> int:
    story = fitz.Story(html=html_text, user_css=CSS)
    writer = fitz.DocumentWriter(str(out_path))
    page = fitz.paper_rect("a4")
    frame = page + (40, 44, -40, -44)
    pages = 0
    more = 1
    while more:
        device = writer.begin_page(page)
        more, _ = story.place(frame)
        story.draw(device)
        writer.end_page()
        pages += 1
        if pages > 400:
            raise RuntimeError("runaway pagination")
    writer.close()
    return pages


def main() -> None:
    root = pathlib.Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=str(root / "audit" / "AUDIT.md"))
    ap.add_argument("--out", default=str(root / "docs" / "summary" / "audit-report.pdf"))
    args = ap.parse_args()

    src = pathlib.Path(args.source)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pages = render(to_html(src.read_text(encoding="utf-8")), out)
    print(f"{src.name} -> {out}  ({pages} pages, {out.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
