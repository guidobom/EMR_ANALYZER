"""Fase 4 — registry integration: SNOMED variant selection and persistence.

Covers the registry-side knob policy:

- ``ClinicalPipelinePolicy.atomic_extraction_variant`` selects the constrained
  ``SnomedAtomicEvidenceExtractor``; the default stays the base extractor.
- A missing RF2 release with ``variant == "snomed"`` is a configuration error
  (``FileNotFoundError``), never a silent fallback.
- Checkpoint constants are variant-aware and include the release digest.
- ``extract_atomic_evidence`` persists atoms tagged ``llm_atomic_snomed_v1``
  with the SNOMED CT terminology payload.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from emr_analyzer.clinical.atomic_evidence import (
    ATOMIC_PIPELINE_VERSION,
    ATOMIC_PROMPT_DIGEST,
    ATOMIC_PROMPT_VERSION,
    AtomicEvidenceExtractor,
)
from emr_analyzer.clinical.registry_builder import ClinicalRegistryBuilder
from emr_analyzer.clinical.snomed import (
    SNOMED_ATOMIC_PIPELINE_VERSION,
    SNOMED_ATOMIC_PROMPT_VERSION,
    load_snapshot,
)
from emr_analyzer.clinical.snomed.extractor import (
    SNOMED_ATOMIC_PROMPT_DIGEST,
    SnomedAtomicEvidenceExtractor,
)
from emr_analyzer.config import active_workspace
from emr_analyzer.database.document_repo import DocumentRepository
from emr_analyzer.database.engine import DatabaseEngine
from emr_analyzer.database.evidence_repo import EvidenceRepository
from emr_analyzer.database.lab_repo import LabRepository
from emr_analyzer.database.migrations import init_database
from emr_analyzer.database.overlay_repo import DocumentTextOverlayRepository
from emr_analyzer.database.processing_repo import ProcessingRepository
from emr_analyzer.database.registry_repo import ClinicalRegistryRepository
from emr_analyzer.database.timeline_repo import TimelineRepository
from emr_analyzer.settings import ClinicalPipelinePolicy

FIXTURE_DIR = (
    Path(__file__).resolve().parent / "fixtures" / "snomed_rf2" / "Snapshot_20260101"
)


class _SnomedLlm:
    """Fake LLM emitting a valid constrained SNOMED code for the chunk."""

    model = "fake-snomed-medical"
    context_length = 32768
    max_output_tokens = 4096
    is_available = True

    def __init__(self, response=None):
        self._response = response or {
            "diagnosis": [
                {
                    "snomed_code": "44054006",
                    "polarity": "present",
                    "source_refs": [1],
                },
            ],
        }

    def generate_structured(self, prompt, system, schema, **kwargs):
        return self._response


def _seed(db: DatabaseEngine) -> None:
    db.execute(
        """INSERT INTO patients
           (id, pseudonym, created_at, updated_at)
           VALUES ('P001', 'PAZIENTE_001', '2026-01-01', '2026-01-01')"""
    )
    db.execute(
        """INSERT INTO documents
           (id, patient_id, filename, original_path, file_hash,
            document_date, document_type, import_date)
           VALUES ('D1', 'P001', 'D1.pdf', '/D1.pdf', 'hash-D1',
                   '2025-01-10', 'referto', '2026-01-01')"""
    )
    db.commit()


def _repos(db: DatabaseEngine):
    return dict(
        registry_repo=ClinicalRegistryRepository(db),
        evidence_repo=EvidenceRepository(db),
        processing_repo=ProcessingRepository(db),
        timeline_repo=TimelineRepository(db),
        document_repo=DocumentRepository(db),
        lab_repo=LabRepository(db),
        overlay_repo=DocumentTextOverlayRepository(db),
        db=db,
    )


@pytest.fixture(scope="module")
def snapshot():
    return load_snapshot(FIXTURE_DIR)


@pytest.fixture()
def db():
    with tempfile.TemporaryDirectory() as tmp:
        engine = DatabaseEngine(Path(tmp) / "emr.sqlite")
        init_database(engine)
        _seed(engine)
        yield engine
        engine.close()


def _snomed_policy(**overrides) -> ClinicalPipelinePolicy:
    kwargs = dict(
        atomic_extraction_variant="snomed",
        snomed_release_dir=str(FIXTURE_DIR),
        snomed_languages=("en", "it"),
    )
    kwargs.update(overrides)
    return ClinicalPipelinePolicy(**kwargs)


class TestVariantSelection:
    def test_default_variant_uses_base_extractor(self, db):
        builder = ClinicalRegistryBuilder(
            atomic_llm_client=_SnomedLlm(),
            pipeline_policy=ClinicalPipelinePolicy(),
            **_repos(db),
        )
        assert isinstance(builder.atomic_extractor, AtomicEvidenceExtractor)
        assert not isinstance(
            builder.atomic_extractor, SnomedAtomicEvidenceExtractor
        )

    def test_snomed_variant_selects_constrained_extractor(self, db, snapshot):
        builder = ClinicalRegistryBuilder(
            atomic_llm_client=_SnomedLlm(),
            pipeline_policy=_snomed_policy(),
            **_repos(db),
        )
        assert isinstance(builder.atomic_extractor, SnomedAtomicEvidenceExtractor)
        # The retriever is wired to the licensed release digest so checkpoint
        # invalidation is keyed on the actual RF2 content.
        assert builder.atomic_extractor.retriever.release_digest == snapshot.digest

    def test_snomed_variant_missing_release_is_config_error(self, db):
        with pytest.raises(FileNotFoundError):
            ClinicalRegistryBuilder(
                atomic_llm_client=_SnomedLlm(),
                pipeline_policy=_snomed_policy(
                    snomed_release_dir="/nonexistent/snomed/release"
                ),
                **_repos(db),
            )

    def test_snomed_variant_without_llm_keeps_extractor_none(self, db):
        builder = ClinicalRegistryBuilder(
            atomic_llm_client=None,
            pipeline_policy=_snomed_policy(),
            **_repos(db),
        )
        assert builder.atomic_extractor is None


class TestVariantConstants:
    def test_snomed_constants_carry_release_digest(self, db, snapshot):
        builder = ClinicalRegistryBuilder(
            atomic_llm_client=_SnomedLlm(),
            pipeline_policy=_snomed_policy(),
            **_repos(db),
        )
        assert builder._atomic_prompt_constants() == (
            SNOMED_ATOMIC_PIPELINE_VERSION,
            SNOMED_ATOMIC_PROMPT_VERSION,
            SNOMED_ATOMIC_PROMPT_DIGEST,
            snapshot.digest,
        )

    def test_standard_constants_match_base(self, db):
        builder = ClinicalRegistryBuilder(
            atomic_llm_client=_SnomedLlm(),
            pipeline_policy=ClinicalPipelinePolicy(),
            **_repos(db),
        )
        assert builder._atomic_prompt_constants() == (
            ATOMIC_PIPELINE_VERSION,
            ATOMIC_PROMPT_VERSION,
            ATOMIC_PROMPT_DIGEST,
            None,
        )


def _run_snomed_extraction(db, *, text, response=None):
    """Run the atomic stage through the registry with the SNOMED variant."""
    old_workspace = active_workspace.path
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp) / "workspace"
        extraction = workspace / "P001" / "extraction"
        extraction.mkdir(parents=True)
        (extraction / "D1.md").write_text(text, encoding="utf-8")
        active_workspace.set_path(workspace)
        builder = ClinicalRegistryBuilder(
            atomic_llm_client=_SnomedLlm(response=response),
            pipeline_policy=_snomed_policy(),
            **_repos(db),
        )
        try:
            result = builder.extract_atomic_evidence(
                "P001", incremental=False, num_workers=1
            )
        finally:
            active_workspace.set_path(old_workspace)
    return builder, result


class TestAtomicExtractionPersistence:
    def test_snomed_atoms_persist_with_terminology_payload(
        self, db, snapshot
    ):
        builder, result = _run_snomed_extraction(
            db, text="Il paziente ha diabete mellito di tipo 2.",
        )

        assert result["total_entries"] >= 1
        evidence = builder.evidence_repo.get_by_patient("P001")
        snomed_atoms = [
            e for e in evidence if e.terminology_system == "SNOMED CT"
        ]
        assert len(snomed_atoms) == 1
        atom = snomed_atoms[0]
        assert atom.extraction_method == "llm_atomic_snomed_v1"
        assert atom.terminology_code == "44054006"
        assert atom.mapping_status == "resolved_llm"
        assert atom.mapping_confidence == 0.90
        assert atom.canonical_label == "Diabete mellito"
        assert atom.data["snomed"]["code"] == "44054006"
        assert atom.data["snomed"]["release_digest"] == snapshot.digest
        # The run was tagged with the SNOMED variant and release digest.
        row = db.execute(
            """SELECT parameters_json FROM processing_runs
               WHERE patient_id=? AND stage=? ORDER BY started_at DESC LIMIT 1""",
            ("P001", "atomic_evidence_v3"),
        ).fetchone()
        assert row is not None
        parameters = json.loads(row["parameters_json"])
        assert parameters["atomic_variant"] == "snomed"
        assert parameters["snomed_release_digest"] == snapshot.digest

    def test_abbreviation_fallback_resolves_through_registry(self, db):
        # ``HT`` is not a literal token in the fixture index, so the curated
        # abbreviation fallback is what lets the retriever recall hypertension.
        # This exercises the fallback end-to-end through the registry builder.
        builder, result = _run_snomed_extraction(
            db,
            text="Il paziente presenta HT.",
            response={
                "diagnosis": [
                    {
                        "snomed_code": "38341003",
                        "polarity": "present",
                        "source_refs": [1],
                    },
                ],
            },
        )

        assert result["total_entries"] >= 1
        evidence = builder.evidence_repo.get_by_patient("P001")
        snomed_atoms = [
            e for e in evidence if e.terminology_system == "SNOMED CT"
        ]
        assert len(snomed_atoms) == 1
        atom = snomed_atoms[0]
        assert atom.terminology_code == "38341003"
        assert atom.canonical_label == "Ipertensione arteriosa"
        assert atom.extraction_method == "llm_atomic_snomed_v1"
        assert atom.mapping_status == "resolved_llm"
