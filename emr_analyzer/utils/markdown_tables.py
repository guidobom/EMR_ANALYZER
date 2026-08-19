"""Markdown pipe-tables → HTML, for the chat panel (QTextBrowser).

The clinical-query chat renders HTML: LLM answers formatted as Markdown
tables would show up as raw ``| ... |`` text.  This helper converts
consecutive pipe-table lines into real ``<table>`` blocks while the rest
of the text keeps flowing as escaped plain text.
"""

from __future__ import annotations

import html
import re

_SEPARATOR_CELL = re.compile(r"^:?-{3,}:?$")


def _cells(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _table_to_html(table_lines: list[str]) -> str:
    rows = [_cells(line) for line in table_lines]
    # Drop pure separator rows (| --- | --- |).
    rows = [
        row for row in rows
        if not all(_SEPARATOR_CELL.match(cell) for cell in row)
    ]
    if not rows:
        return ""
    parts = ["<table border='1' cellspacing='0' cellpadding='4'>"]
    for index, row in enumerate(rows):
        tag = "th" if index == 0 else "td"
        parts.append(
            "<tr>"
            + "".join(
                f"<{tag}>{html.escape(cell)}</{tag}>" for cell in row
            )
            + "</tr>"
        )
    parts.append("</table>")
    return "".join(parts)


def markdown_tables_to_html(text: str) -> str:
    """Convert Markdown pipe-tables in *text* to HTML ``<table>`` blocks.

    Non-table lines are returned untouched (the caller escapes them).
    """
    lines = text.split("\n")
    output: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.lstrip()
        if stripped.startswith("|") and line.rstrip().endswith("|"):
            table_lines = [line.rstrip()]
            index += 1
            while index < len(lines) and lines[index].lstrip().startswith("|"):
                table_lines.append(lines[index].rstrip())
                index += 1
            table_html = _table_to_html(table_lines)
            if table_html:
                output.append(table_html)
        else:
            output.append(line)
            index += 1
    return "\n".join(output)


def render_markdown_to_html(text: str) -> str:
    """Render chat content for QTextBrowser: tables as HTML, text escaped.

    The chat panel feeds raw HTML to ``QTextBrowser``; this produces safe
    output where pipe-tables become real ``<table>`` blocks and every
    other line is HTML-escaped (newlines become ``<br>``).
    """
    lines = str(text or "").split("\n")
    output: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.lstrip().startswith("|") and line.rstrip().endswith("|"):
            table_lines = [line.rstrip()]
            index += 1
            while index < len(lines) and lines[index].lstrip().startswith("|"):
                table_lines.append(lines[index].rstrip())
                index += 1
            table_html = _table_to_html(table_lines)
            if table_html:
                output.append(table_html)
        else:
            output.append(html.escape(line))
            index += 1
    return "<br>".join(output)
