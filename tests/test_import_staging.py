"""Tests for the import inbox lifecycle (privacy fix P3).

The staging inbox holds raw identity copies (name, CF, …) in ``_inbox``.
These must never survive an import, neither on the happy path nor on a
catastrophic staging failure.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from emr_analyzer.config import active_workspace
from emr_analyzer.models.patient_identity import PatientIdentityEvidence
from emr_analyzer.pipeline.import_staging import ImportStagingService


class _StubIdentityExtractor:
    """Extracts a bare evidence object; can be told to fail on some names."""

    def __init__(self, raise_on=None):
        self.raise_on = set(raise_on or [])

    def extract(self, path):
        if Path(path).name in self.raise_on:
            raise ValueError("identity extraction failed")
        return PatientIdentityEvidence(source_path=str(path))


class _StubDocumentRepo:
    def get_by_hash_global(self, file_hash):
        return None


class ImportStagingTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="emr_staging_")
        self._root = Path(self._tmp)
        self._orig_workspace_path = active_workspace.path
        active_workspace.set_path(self._root / "workspaces")
        self._inbox = active_workspace.path / "_inbox"
        self._service = ImportStagingService(
            identity_extractor=_StubIdentityExtractor(),
            document_repo=_StubDocumentRepo(),
        )

    def tearDown(self):
        active_workspace.set_path(self._orig_workspace_path)
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _make_png(self, name="ref.png") -> Path:
        path = self._root / "src" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png-content")
        return path

    def _inbox_dirs(self):
        return [p for p in self._inbox.iterdir()] if self._inbox.exists() else []

    def test_stage_copies_files_and_cleanup_removes_inbox(self):
        source = self._make_png("a.png")
        batch = self._service.stage([str(source)])

        self.assertEqual(len(batch.documents), 1)
        self.assertIsNone(batch.documents[0].error)
        self.assertTrue(Path(batch.documents[0].staged_path).is_file())
        # The raw identity copy sits in _inbox until cleanup.
        self.assertEqual(len(self._inbox_dirs()), 1)

        batch.cleanup()
        self.assertEqual(self._inbox_dirs(), [])

    def test_per_file_error_keeps_batch_without_raising(self):
        service = ImportStagingService(
            identity_extractor=_StubIdentityExtractor(raise_on=["bad.png"]),
            document_repo=_StubDocumentRepo(),
        )
        good = self._make_png("good.png")
        bad = self._make_png("bad.png")

        batch = service.stage([str(bad), str(good)])

        by_name = {Path(d.staged_path).name: d for d in batch.documents}
        self.assertIsNotNone(by_name["bad.png"].error)
        self.assertIsNone(by_name["good.png"].error)
        # The per-file failure does not abort the batch: inbox still exists.
        self.assertEqual(len(self._inbox_dirs()), 1)

    def test_catastrophic_failure_cleans_inbox(self):
        source = self._make_png("a.png")

        def exploding_progress(*args, **kwargs):
            raise RuntimeError("progress crashed")

        with self.assertRaises(RuntimeError):
            self._service.stage(
                [str(source)], progress_callback=exploding_progress
            )

        # The raw identity copy must not survive a catastrophic failure.
        self.assertEqual(self._inbox_dirs(), [])

    def test_cancel_check_stops_before_copy(self):
        source = self._make_png("a.png")
        batch = self._service.stage(
            [str(source)], cancel_check=lambda: True
        )
        self.assertEqual(batch.documents, [])
        self.assertEqual(len(self._inbox_dirs()), 1)
        batch.cleanup()
        self.assertEqual(self._inbox_dirs(), [])


if __name__ == "__main__":
    unittest.main()
