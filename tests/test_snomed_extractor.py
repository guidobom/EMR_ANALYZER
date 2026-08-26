"""Tests for the SNOMED-constrained atomic extractor (Percorso A)."""

from __future__ import annotations

from pathlib import Path

import pytest

from emr_analyzer.clinical.atomic_evidence import (
    AtomicEvidenceExtractor,
    SentenceSpan,
    TextChunk,
    _canonical_wire_item,
    build_atomic_evidence_schema,
    build_atomic_prompt,
)
from emr_analyzer.clinical.snomed import (
    ReleaseSnapshot,
    SnomedIndex,
    SnomedCandidateRetriever,
    load_snapshot,
)
from emr_analyzer.clinical.snomed.extractor import (
    SNOMED_ATOMIC_PROMPT_DIGEST,
    SnomedAtomicEvidenceExtractor,
)
from emr_analyzer.settings import ClinicalPipelinePolicy

FIXTURE_DIR = (
    Path(__file__).resolve().parent / "fixtures" / "snomed_rf2" / "Snapshot_20260101"
)


@pytest.fixture(scope="module")
def snapshot() -> ReleaseSnapshot:
    return load_snapshot(FIXTURE_DIR)


@pytest.fixture(scope="module")
def retriever(snapshot: ReleaseSnapshot) -> SnomedCandidateRetriever:
    return SnomedCandidateRetriever(
        SnomedIndex(snapshot.concepts.values()),
        release_digest=snapshot.digest,
    )


class _FakeLLM:
    model = "test-model"
    context_length = 32768
    max_output_tokens = 4096

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate_structured(self, prompt, system, schema, **kwargs):
        self.calls.append({"prompt": prompt, "system": system, "schema": schema})
        if len(self.responses) == 1:
            return self.responses[0]
        return self.responses.pop(0)


def _extractor(retriever, llm, *, adaptive=False):
    return SnomedAtomicEvidenceExtractor(
        llm,
        retriever=retriever,
        policy=ClinicalPipelinePolicy(adaptive_specialized_retry=adaptive),
    )


def _diabete_set(retriever):
    return retriever.candidates_for_text("diabete")


class TestSchemaShape:
    def test_enum_matches_candidate_set(self, retriever):
        codes = ("44054006", "93655004")
        schema = build_atomic_evidence_schema(snomed_codes=codes)
        for fact_type, bucket in schema["properties"].items():
            item = bucket["items"]
            assert item["properties"]["snomed_code"]["enum"] == list(codes)
            assert "concept" not in item["properties"]
            assert item["required"] == ["snomed_code", "polarity", "source_refs"]

    def test_standard_path_keeps_concept(self):
        schema = build_atomic_evidence_schema()
        item = schema["properties"]["diagnosis"]["items"]
        assert "concept" in item["properties"]
        assert "snomed_code" not in item["properties"]
        assert "concept" in item["required"]

    def test_codes_deduplicated_and_ordered(self):
        schema = build_atomic_evidence_schema(
            snomed_codes=("93655004", "44054006", "93655004")
        )
        item = schema["properties"]["diagnosis"]["items"]
        assert item["properties"]["snomed_code"]["enum"] == [
            "93655004", "44054006",
        ]


class TestCanonicalWireItem:
    def test_valid_code_resolves_pt(self, retriever):
        candidates = _diabete_set(retriever)
        item, reasons, changed = _canonical_wire_item(
            {"snomed_code": "44054006", "polarity": "present",
             "source_refs": [1]},
            "diagnosis",
            [SentenceSpan(sentence_id=1, start=0, end=10, text="diabete")],
            snomed_candidates=candidates,
        )
        assert item is not None
        assert not reasons
        assert item["concept"] == "Diabete mellito"
        assert item["_snomed"]  # _expand_atomic_item maps it to "snomed" later
        meta = item["_snomed"]
        assert meta["code"] == "44054006"
        assert meta["system"] == "SNOMED CT"
        assert meta["status"] == "resolved_llm"
        assert meta["confidence"] == 0.90
        assert meta["release_digest"] == retriever.release_digest
        assert item["fact_type"] == "diagnosis"

    def test_code_outside_set_is_failure(self, retriever):
        candidates = _diabete_set(retriever)
        item, reasons, _ = _canonical_wire_item(
            {"snomed_code": "99999999", "polarity": "present",
             "source_refs": [1]},
            "diagnosis",
            [SentenceSpan(sentence_id=1, start=0, end=10, text="diabete")],
            snomed_candidates=candidates,
        )
        assert item is None
        assert any("set candidato" in reason for reason in reasons)

    def test_missing_code_is_failure(self, retriever):
        candidates = _diabete_set(retriever)
        item, reasons, _ = _canonical_wire_item(
            {"polarity": "present", "source_refs": [1]},
            "diagnosis",
            [SentenceSpan(sentence_id=1, start=0, end=10, text="diabete")],
            snomed_candidates=candidates,
        )
        assert item is None
        assert any("snomed_code" in reason for reason in reasons)

    def test_single_domain_reclassifies(self, retriever):
        candidates = _diabete_set(retriever)
        # 44054006 maps only to diagnosis; the LLM emitted it in medication.
        item, reasons, _ = _canonical_wire_item(
            {"snomed_code": "44054006", "polarity": "present",
             "source_refs": [1]},
            "medication",
            [SentenceSpan(sentence_id=1, start=0, end=10, text="diabete")],
            snomed_candidates=candidates,
        )
        assert item is not None
        assert item["fact_type"] == "diagnosis"
        assert item["_snomed"]["reclassified_from"] == "medication"
        assert item["_snomed"]["domain_mismatch"] is False

    def test_multi_domain_in_domain_stays(self, retriever, snapshot):
        candidates = retriever.candidates_for_concepts([
            c for c in [snapshot.concept("372250003")] if c is not None
        ])
        item, reasons, _ = _canonical_wire_item(
            {"snomed_code": "372250003", "polarity": "present",
             "source_refs": [1]},
            "medication",
            [SentenceSpan(sentence_id=1, start=0, end=10, text="metformina")],
            snomed_candidates=candidates,
        )
        assert item is not None
        assert item["fact_type"] == "medication"
        assert item["_snomed"]["domain_mismatch"] is False

    def test_multi_domain_off_domain_audits_not_fails(self, retriever, snapshot):
        candidates = retriever.candidates_for_concepts([
            c for c in [snapshot.concept("372250003")] if c is not None
        ])
        item, reasons, _ = _canonical_wire_item(
            {"snomed_code": "372250003", "polarity": "present",
             "source_refs": [1]},
            "diagnosis",
            [SentenceSpan(sentence_id=1, start=0, end=10, text="metformina")],
            snomed_candidates=candidates,
        )
        assert item is not None
        assert item["fact_type"] == "diagnosis"
        assert item["_snomed"]["domain_mismatch"] is True


class TestSchemaCacheKey:
    def test_different_candidate_sets_yield_different_schemas(
        self, retriever
    ):
        extractor = _extractor(
            retriever, _FakeLLM([{}]), adaptive=False
        )
        diabete = _diabete_set(retriever)
        melanoma = retriever.candidates_for_text("melanoma")
        assert diabete.digest != melanoma.digest
        schema_a = extractor._schema_for(
            1, snomed_candidates=diabete
        )
        schema_b = extractor._schema_for(
            1, snomed_candidates=melanoma
        )
        assert schema_a is not schema_b
        assert (
            schema_a["properties"]["diagnosis"]["items"]["properties"][
                "snomed_code"
            ]["enum"]
            == ["44054006"]
        )
        assert schema_b["properties"]["diagnosis"]["items"]["properties"][
            "snomed_code"
        ]["enum"] == ["104910003", "93655004"]

    def test_same_candidate_set_is_cached(self, retriever):
        extractor = _extractor(retriever, _FakeLLM([{}]), adaptive=False)
        diabete = _diabete_set(retriever)
        first = extractor._schema_for(1, snomed_candidates=diabete)
        second = extractor._schema_for(1, snomed_candidates=diabete)
        assert first is second

    def test_candidate_digest_participates_in_key(self, retriever):
        # Same sentence count and fact types, different candidate set, must
        # NOT reuse a cached schema.
        extractor = _extractor(retriever, _FakeLLM([{}]), adaptive=False)
        a = _diabete_set(retriever)
        b = retriever.candidates_for_text("ipertensione")
        assert a.digest != b.digest
        schema_a = extractor._schema_for(1, snomed_candidates=a)
        schema_b = extractor._schema_for(1, snomed_candidates=b)
        assert schema_a is not schema_b


class TestPromptCatalog:
    def test_prompt_includes_candidate_catalog(self, retriever):
        diabete = _diabete_set(retriever)
        chunk = TextChunk(index=0, text="Il paziente ha diabete mellito.",
                          page_start=1)
        prompt = build_atomic_prompt(
            chunk, document_type="lettera_di_dimissione",
            document_date="2026-01-01",
            candidate_catalog=diabete.to_catalog(),
        )
        assert "CANDIDATI" in prompt
        assert "44054006 | Diabete mellito" in prompt
        assert "[domini: diagnosis]" in prompt

    def test_prompt_without_catalog_matches_base(self):
        chunk = TextChunk(index=0, text="Il paziente ha diabete mellito.",
                          page_start=1)
        with_catalog = build_atomic_prompt(
            chunk, document_type="x", document_date=None,
            candidate_catalog="CAT",
        )
        without = build_atomic_prompt(
            chunk, document_type="x", document_date=None,
            candidate_catalog=None,
        )
        assert without == build_atomic_prompt(
            chunk, document_type="x", document_date=None,
        )
        assert "CAT" in with_catalog
        assert "CANDIDATI" not in without

    def test_snomed_prompt_digest_is_stable(self):
        assert len(SNOMED_ATOMIC_PROMPT_DIGEST) == 64
        assert SNOMED_ATOMIC_PROMPT_DIGEST.isalnum()


class TestExtractChunkIntegration:
    def test_constrained_generation_emits_snomed_atom(self, retriever):
        llm = _FakeLLM([{
            "diagnosis": [
                {"snomed_code": "44054006", "polarity": "present",
                 "source_refs": [1]},
            ],
        }])
        extractor = _extractor(retriever, llm, adaptive=False)
        chunk = TextChunk(index=0, text="Il paziente ha diabete mellito.",
                          page_start=1)
        items, spans = extractor._extract_chunk(
            chunk, document_type="lettera_di_dimissione",
            document_date="2026-01-01",
        )
        assert len(items) == 1
        item = items[0]
        assert item["fact_type"] == "diagnosis"
        assert item["concept"] == "Diabete mellito"
        assert item["_snomed"]["code"] == "44054006"
        # The wire schema sent to the LLM never exposed a free-text concept.
        assert "concept" not in llm.calls[0]["schema"]["properties"][
            "diagnosis"
        ]["items"]["properties"]
        assert llm.calls[0]["schema"]["properties"]["diagnosis"]["items"][
            "properties"
        ]["snomed_code"]["enum"] == ["44054006"]
        assert "CANDIDATI" in llm.calls[0]["prompt"]

    def test_code_outside_set_is_repaired_then_dropped(self, retriever):
        llm = _FakeLLM([
            {
                "diagnosis": [
                    {"snomed_code": "99999999", "polarity": "present",
                     "source_refs": [1]},
                ],
            },
            {"diagnosis": []},
        ])
        extractor = _extractor(retriever, llm, adaptive=True)
        chunk = TextChunk(index=0, text="Il paziente ha diabete mellito.",
                          page_start=1)
        items, spans = extractor._extract_chunk(
            chunk, document_type="lettera_di_dimissione",
            document_date="2026-01-01",
        )
        assert items == []
        metrics = extractor.last_extraction_metrics()
        assert metrics["invalid_items"] >= 1
        assert metrics["validation_retries"] >= 1

    def test_code_outside_set_repair_recovers_valid_code(self, retriever):
        llm = _FakeLLM([
            {
                "diagnosis": [
                    {"snomed_code": "99999999", "polarity": "present",
                     "source_refs": [1]},
                ],
            },
            {
                "diagnosis": [
                    {"snomed_code": "44054006", "polarity": "present",
                     "source_refs": [1]},
                ],
            },
        ])
        extractor = _extractor(retriever, llm, adaptive=True)
        chunk = TextChunk(index=0, text="Il paziente ha diabete mellito.",
                          page_start=1)
        items, spans = extractor._extract_chunk(
            chunk, document_type="lettera_di_dimissione",
            document_date="2026-01-01",
        )
        assert len(items) == 1
        assert items[0]["_snomed"]["code"] == "44054006"

    def test_empty_candidate_set_degrades_to_standard(self, retriever):
        llm = _FakeLLM([{
            "diagnosis": [
                {"concept": "Diabete mellito", "polarity": "present",
                 "source_refs": [1]},
            ],
        }])
        extractor = _extractor(retriever, llm, adaptive=False)
        chunk = TextChunk(index=0, text="zzzqwerty",
                          page_start=1)
        items, spans = extractor._extract_chunk(
            chunk, document_type="lettera_di_dimissione",
            document_date="2026-01-01",
        )
        assert len(items) == 1
        # Degraded path: no snomed metadata, free-text concept kept.
        assert "_snomed" not in items[0]
        assert items[0]["concept"] == "Diabete mellito"
        metrics = extractor.last_extraction_metrics()
        assert metrics["snomed_empty_candidate_chunks"] >= 1

    def test_release_digest_stamped_on_candidates(self, retriever, snapshot):
        candidates = _diabete_set(retriever)
        assert candidates.release_digest == snapshot.digest

    def test_extraction_method_label_is_snomed_variant(self):
        assert (
            SnomedAtomicEvidenceExtractor._atomic_extraction_method
            == "llm_atomic_snomed_v1"
        )
        assert AtomicEvidenceExtractor._atomic_extraction_method == "llm_atomic_v2"


class TestExtractDocumentIntegration:
    def test_synonym_duplicates_collapse_to_one_coded_atom(self, retriever):
        llm = _FakeLLM([{
            "diagnosis": [
                {"snomed_code": "44054006", "polarity": "present",
                 "source_refs": [1]},
                {"snomed_code": "44054006", "polarity": "present",
                 "source_refs": [1]},
            ],
        }])
        extractor = _extractor(retriever, llm, adaptive=False)
        evidence = extractor.extract_document(
            patient_id="P1", document_id="D1",
            document_type="lettera_di_dimissione",
            document_date="2026-01-01",
            text="Il paziente ha diabete mellito.",
        )
        coded = [
            e for e in evidence
            if e.terminology_system == "SNOMED CT"
        ]
        assert len(coded) == 1
        atom = coded[0]
        assert atom.terminology_code == "44054006"
        assert atom.canonical_label == "Diabete mellito"
        assert atom.mapping_status == "resolved_llm"
        assert atom.mapping_confidence == 0.90
        assert atom.extraction_method == "llm_atomic_snomed_v1"
        assert atom.data["snomed"]["code"] == "44054006"
        metrics = extractor.last_extraction_metrics()
        assert metrics["snomed_coded_atoms"] >= 1

    def test_empty_candidate_chunk_counts_unmapped_atom(self, retriever):
        llm = _FakeLLM([{
            "diagnosis": [
                {"concept": "Reperto non codificabile", "polarity": "present",
                 "source_refs": [1]},
            ],
        }])
        extractor = _extractor(retriever, llm, adaptive=False)
        evidence = extractor.extract_document(
            patient_id="P1", document_id="D1",
            document_type="lettera_di_dimissione",
            document_date="2026-01-01",
            text="zzzqwerty",
        )
        metrics = extractor.last_extraction_metrics()
        assert metrics["snomed_empty_candidate_chunks"] >= 1
        assert metrics["snomed_unmapped_atoms"] >= 1
        assert metrics["snomed_coded_atoms"] == 0
        # Degraded atom carries no terminology payload.
        assert not any(
            e.terminology_system == "SNOMED CT" for e in evidence
        )
