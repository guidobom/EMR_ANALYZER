"""Tests for the per-patient clinical chat repository."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import tempfile
import unittest

from emr_analyzer.database.chat_repo import ChatRepository
from emr_analyzer.database.engine import DatabaseEngine
from emr_analyzer.database.migrations import init_database, SCHEMA_VERSION
from emr_analyzer.database.patient_repo import PatientRepository
from emr_analyzer.models import Patient
from emr_analyzer.models.chat_message import ChatMessage


def make_message(patient_id="P001", role="user", content="domanda?",
                 model_used="", context_mode=0, created_at=None) -> ChatMessage:
    return ChatMessage(
        id=ChatRepository.new_id(),
        patient_id=patient_id,
        role=role,
        content=content,
        model_used=model_used,
        context_mode=context_mode,
        created_at=created_at or datetime.now().isoformat(),
    )


class ChatRepositoryTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.db = DatabaseEngine(root / "registry.db")
        init_database(self.db)
        patient_repo = PatientRepository(self.db)
        patient_repo.insert(Patient(id="P001", pseudonym="001", sex="M"))
        patient_repo.insert(Patient(id="P002", pseudonym="002", sex="F"))
        self.repo = ChatRepository(self.db)

    def tearDown(self):
        self._tmp.cleanup()

    def test_schema_version_11(self):
        row = self.db.execute(
            "SELECT MAX(version) AS v FROM schema_version"
        ).fetchone()
        self.assertEqual(int(row["v"]), SCHEMA_VERSION)

    def test_add_and_get_ordered(self):
        first = make_message(created_at="2026-08-19T10:00:00.000001")
        second = make_message(
            role="assistant", content="risposta",
            created_at="2026-08-19T10:00:01.000001",
        )
        self.repo.add_message(second)
        self.repo.add_message(first)
        messages = self.repo.get_by_patient("P001")
        self.assertEqual([m.id for m in messages], [first.id, second.id])
        self.assertEqual(messages[0].content, "domanda?")
        self.assertEqual(messages[1].role, "assistant")

    def test_per_patient_isolation(self):
        self.repo.add_message(make_message(patient_id="P001"))
        self.repo.add_message(make_message(patient_id="P002", content="altro"))
        self.assertEqual(len(self.repo.get_by_patient("P001")), 1)
        self.assertEqual(len(self.repo.get_by_patient("P002")), 1)
        self.assertEqual(
            self.repo.get_by_patient("P002")[0].content, "altro"
        )

    def test_metadata_round_trip(self):
        message = make_message(
            role="assistant", content="risposta clinica",
            model_used="qwen3-14b", context_mode=1,
        )
        self.repo.add_message(message)
        loaded = self.repo.get_by_patient("P001")[0]
        self.assertEqual(loaded.role, "assistant")
        self.assertEqual(loaded.model_used, "qwen3-14b")
        self.assertEqual(loaded.context_mode, 1)
        self.assertEqual(loaded.created_at, message.created_at)

    def test_count_and_clear(self):
        self.repo.add_message(make_message())
        self.repo.add_message(make_message(role="assistant"))
        self.assertEqual(self.repo.count_by_patient("P001"), 2)
        self.repo.clear_for_patient("P001")
        self.assertEqual(self.repo.count_by_patient("P001"), 0)

    def test_fk_cascade_on_patient_delete(self):
        self.repo.add_message(make_message())
        self.db.execute("DELETE FROM patients WHERE id = 'P001'")
        self.db.commit()
        self.assertEqual(self.repo.count_by_patient("P001"), 0)

    def test_drop_all_tables_includes_chat(self):
        from emr_analyzer.database.migrations import drop_all_tables
        drop_all_tables(self.db)
        rows = self.db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name='clinical_chat'"
        ).fetchall()
        self.assertEqual(len(rows), 0)


if __name__ == "__main__":
    unittest.main()
