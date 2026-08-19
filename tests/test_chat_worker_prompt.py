"""Tests for the clinical query prompt builder (pure, no Qt event loop)."""

from __future__ import annotations

import unittest

from emr_analyzer.gui.workers import build_query_prompt

PROFILE = "Profilo narrativo del paziente."
ENTRIES = "[2020-05-01] [diagnosis] Melanoma dorsale"
QUESTION = "Quando è iniziata la terapia?"


def make_conversation(count: int) -> list[dict]:
    messages = []
    for i in range(count):
        messages.append({"role": "user", "content": f"domanda {i}"})
        messages.append(
            {"role": "assistant", "content": f"risposta {i}"}
        )
    return messages


class TestBuildQueryPrompt(unittest.TestCase):
    def test_without_conversation_prompt_is_classic(self):
        prompt = build_query_prompt(PROFILE, ENTRIES, QUESTION)
        self.assertIn("PROFILO CLINICO:\n" + PROFILE, prompt)
        self.assertIn("REGISTRO CRONOLOGICO:\n" + ENTRIES, prompt)
        self.assertIn(f"DOMANDA: {QUESTION}", prompt)
        self.assertNotIn("CONVERSAZIONE PRECEDENTE", prompt)

    def test_toggle_off_ignores_conversation(self):
        prompt = build_query_prompt(
            PROFILE, ENTRIES, QUESTION,
            conversation=make_conversation(2),
            use_conversation_context=False,
        )
        self.assertNotIn("CONVERSAZIONE PRECEDENTE", prompt)
        self.assertNotIn("domanda 0", prompt)

    def test_toggle_on_without_messages_no_section(self):
        prompt = build_query_prompt(
            PROFILE, ENTRIES, QUESTION,
            conversation=[],
            use_conversation_context=True,
        )
        self.assertNotIn("CONVERSAZIONE PRECEDENTE", prompt)

    def test_toggle_on_embeds_conversation(self):
        prompt = build_query_prompt(
            PROFILE, ENTRIES, QUESTION,
            conversation=make_conversation(2),
            use_conversation_context=True,
        )
        self.assertIn("CONVERSAZIONE PRECEDENTE:", prompt)
        self.assertIn("[UTENTE] domanda 1", prompt)
        self.assertIn("[ASSISTENTE] risposta 1", prompt)
        # Section sits before the question.
        self.assertLess(prompt.index("CONVERSAZIONE PRECEDENTE"),
                        prompt.index("DOMANDA:"))

    def test_cap_keeps_last_messages_only(self):
        conversation = make_conversation(25)  # 50 messages, 25 pairs
        prompt = build_query_prompt(
            PROFILE, ENTRIES, QUESTION,
            conversation=conversation,
            use_conversation_context=True,
        )
        # Only the last 20 messages (10 pairs) survive.
        self.assertIn("[UTENTE] domanda 24", prompt)
        self.assertIn("[ASSISTENTE] risposta 24", prompt)
        self.assertNotIn("domanda 10", prompt)   # earliest kept is 15
        self.assertIn("domanda 15", prompt)

    def test_cap_truncates_long_messages(self):
        conversation = [
            {"role": "user", "content": "x" * 2000},
            {"role": "assistant", "content": "y" * 2000},
        ]
        prompt = build_query_prompt(
            PROFILE, ENTRIES, QUESTION,
            conversation=conversation,
            use_conversation_context=True,
        )
        self.assertNotIn("x" * 400, prompt)
        self.assertIn("x" * 300 + "…", prompt)


if __name__ == "__main__":
    unittest.main()
