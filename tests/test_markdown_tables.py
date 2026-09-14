"""Tests for the Markdown pipe-table → HTML rendering of the chat panel."""

from __future__ import annotations

import unittest

from emr_analyzer.utils.markdown_tables import (
    markdown_tables_to_html,
    render_markdown_to_html,
)


class MarkdownTablesTest(unittest.TestCase):
    def test_simple_table(self):
        text = (
            "| ID | Evento | Grado |\n"
            "|---|---|---|\n"
            "| 1 | Colite | G2 |"
        )
        html = render_markdown_to_html(text)
        self.assertIn("<table", html)
        self.assertIn("<th>ID</th>", html)
        self.assertIn("<td>Colite</td>", html)
        self.assertNotIn("|---", html)

    def test_heading_between_tables_kept(self):
        text = (
            "| A | B |\n|---|---|\n| 1 | 2 |\n"
            "### Sotto-tabella\n"
            "| C |\n|---|\n| 3 |"
        )
        html = render_markdown_to_html(text)
        self.assertEqual(html.count("<table"), 2)
        self.assertIn("### Sotto-tabella", html)

    def test_plain_text_escaped(self):
        html = render_markdown_to_html("testo <b>non</b> formattato")
        self.assertIn("&lt;b&gt;", html)
        self.assertNotIn("<b>non</b>", html)

    def test_newlines_become_breaks(self):
        html = render_markdown_to_html("riga1\nriga2")
        self.assertIn("riga1<br>riga2", html)

    def test_malformed_table_line_left_as_text(self):
        text = "| cella senza chiusura"
        html = render_markdown_to_html(text)
        self.assertNotIn("<table", html)
        self.assertIn("| cella senza chiusura", html)

    def test_empty_text(self):
        self.assertEqual(render_markdown_to_html(""), "")

    def test_separator_only_rows_dropped(self):
        text = "| A | B |\n|---|---|\n| :---: | --- |"
        html = markdown_tables_to_html(text)
        self.assertNotIn(":---:", html)

    def test_strikethrough_spans_become_s_tags(self):
        html = render_markdown_to_html("~~rimosso~~ testo")
        self.assertIn("<s>rimosso</s> testo", html)
        self.assertNotIn("~~", html)

    def test_strikethrough_escapes_inner_content(self):
        html = render_markdown_to_html("~~<b>x</b>~~")
        self.assertIn("<s>&lt;b&gt;x&lt;/b&gt;</s>", html)

    def test_lone_tildes_left_as_text(self):
        html = render_markdown_to_html("~~ aperta")
        self.assertIn("~~ aperta", html)


if __name__ == "__main__":
    unittest.main()
