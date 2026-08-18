"""Tests for stored document path relocation against the workspace root."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from emr_analyzer.utils.document_paths import resolve_document_path


class TestResolveDocumentPath(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _make_file(self, relative: str) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"%PDF-fake")
        return path

    def test_stored_path_wins_when_it_exists(self):
        existing = self._make_file("outside/keepme.pdf")
        doc = {
            "original_path": str(existing),
            "filename": "keepme.pdf",
            "patient_id": "P001",
        }
        self.assertEqual(
            resolve_document_path(doc, self.root), str(existing)
        )

    def test_dead_absolute_path_rebuilt_from_workspace_layout(self):
        # Registry built on another machine: the Linux path is dead, the
        # file now lives under <root>/<patient>/documents/original/.
        real = self._make_file("P001/documents/original/doc1.pdf")
        doc = {
            "original_path": "/home/utente/Desktop/MELANOMA/P001/"
                             "documents/original/doc1.pdf",
            "filename": "doc1.pdf",
            "patient_id": "P001",
        }
        self.assertEqual(
            resolve_document_path(doc, self.root), str(real)
        )

    def test_capitalized_layout_fallback(self):
        self._make_file("P002/Documents/Original/doc2.pdf")
        doc = {
            "original_path": "/elsewhere/doc2.pdf",
            "filename": "doc2.pdf",
            "patient_id": "P002",
        }
        resolved = resolve_document_path(doc, self.root)
        # On case-insensitive filesystems (macOS default) the lowercase
        # candidate resolves to the same file: check existence, not case.
        import os
        self.assertTrue(os.path.isfile(resolved))
        self.assertEqual(Path(resolved).name, "doc2.pdf")

    def test_relative_stored_path_joined_to_root(self):
        real = self._make_file("P003/scan.pdf")
        doc = {
            "original_path": "P003/scan.pdf",
            "filename": "scan.pdf",
            "patient_id": "P003",
        }
        self.assertEqual(
            resolve_document_path(doc, self.root), str(real)
        )

    def test_missing_file_falls_back_to_stored_path(self):
        stored = "/home/utente/Desktop/MELANOMA/P001/documents/original/gone.pdf"
        doc = {
            "original_path": stored,
            "filename": "gone.pdf",
            "patient_id": "P001",
        }
        self.assertEqual(resolve_document_path(doc, self.root), stored)

    def test_accepts_model_objects(self):
        real = self._make_file("P004/documents/original/doc4.pdf")
        doc = SimpleNamespace(
            original_path="/dead/doc4.pdf",
            filename="doc4.pdf",
            patient_id="P004",
        )
        self.assertEqual(
            resolve_document_path(doc, self.root), str(real)
        )

    def test_empty_doc_returns_empty(self):
        self.assertEqual(resolve_document_path({}, self.root), "")


if __name__ == "__main__":
    unittest.main()
