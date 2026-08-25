"""Regression tests for the evidence-first clinical registry."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import re
import tempfile
import threading
import time
import unittest

from emr_analyzer.clinical.atomic_evidence import (
    ATOMIC_EVIDENCE_SCHEMA,
    ATOMIC_PROMPT_DIGEST,
    ATOMIC_PROMPT_VERSION,
    AtomicExtractionCancelled,
    AtomicEvidenceExtractor,
    TextChunk,
    build_atomic_evidence_schema,
    build_atomic_prompt,
    deduplicate_atomic_evidence,
    infer_document_content_type,
    split_sentence_spans,
    split_text_chunks,
)
from emr_analyzer.clinical.consolidation import (
    FUSION_PROMPT_DIGEST,
    FUSION_PROMPT_VERSION,
    FUSION_SCHEMA,
    ClinicalConsolidator,
    evidence_for_prompt,
)
from emr_analyzer.clinical.block_reuse import (
    build_targeted_reuse_text,
    clone_reused_evidence,
    evidence_matching_reuse_block,
    plan_exact_block_reuse,
    plan_reuse_verification_batches,
)
from emr_analyzer.clinical.correlation import ClinicalCorrelationBuilder
from emr_analyzer.clinical.registry_builder import ClinicalRegistryBuilder
from emr_analyzer.clinical.temporal import normalize_clinical_date
from emr_analyzer.config import active_workspace
from emr_analyzer.database.audit_repo import AuditRepository
from emr_analyzer.database.clinical_state_repo import ClinicalStateRepository
from emr_analyzer.database.document_repo import DocumentRepository
from emr_analyzer.database.engine import DatabaseEngine
from emr_analyzer.database.evidence_repo import EvidenceRepository
from emr_analyzer.database.lab_repo import LabRepository
from emr_analyzer.database.migrations import init_database
from emr_analyzer.database.overlay_repo import DocumentTextOverlayRepository
from emr_analyzer.database.registry_repo import ClinicalRegistryRepository
from emr_analyzer.database.processing_repo import ProcessingRepository
from emr_analyzer.database.review_repo import ReviewDecisionRepository
from emr_analyzer.database.timeline_repo import TimelineRepository
from emr_analyzer.export.registry_export import ClinicalRegistryExporter
from emr_analyzer.models.clinical_evidence import ClinicalEvidence
from emr_analyzer.models.clinical_state import ClinicalState
from emr_analyzer.models.clinical_registry import (
    ClinicalEvent,
    ClinicalEventRelation,
    EventEvidenceLink,
    ProcessingManifestItem,
    ProcessingRun,
)
from emr_analyzer.extraction.llm_client import OutputLimitError
from emr_analyzer.settings import ClinicalPipelinePolicy


def _seed(db: DatabaseEngine) -> None:
    db.execute(
        """INSERT INTO patients
           (id, pseudonym, created_at, updated_at)
           VALUES ('P001', 'PAZIENTE_001', '2026-01-01', '2026-01-01')"""
    )
    for document_id, date in (("D1", "2025-01-10"), ("D2", "2025-01-12")):
        db.execute(
            """INSERT INTO documents
               (id, patient_id, filename, original_path, file_hash,
                document_date, document_type, import_date)
               VALUES (?, 'P001', ?, ?, ?, ?, 'referto', '2026-01-01')""",
            (document_id, f"{document_id}.pdf", f"/{document_id}.pdf",
             f"hash-{document_id}", date),
        )
    db.commit()


class _AtomicLlm:
    model = "fake-medical"
    context_length = 4096
    max_output_tokens = 512

    def generate_structured(self, prompt, system, schema):
        return {
            "evidence": [{
                "category": "symptom",
                "normalized_entity": "dispnea",
                "source_text": "dispnea da sforzo",
                "source_page": 2,
                "assertion": "present",
                "certainty": "patient_reported",
                "date_precision": "day",
                "observed_date": "2025-01-10",
                "significance": "clinically_relevant",
                "confidence": 0.9,
            }]
        }


class _StructuredCaptureLlm:
    model = "fake-medical"
    context_length = 4096
    max_output_tokens = 512
    temperature = 0.1
    top_p = 0.9
    top_k = 40
    seed = 42

    def __init__(self, response):
        self.response = response
        self.calls = []

    def generate_structured(self, prompt, system, schema):
        self.calls.append({"prompt": prompt, "system": system, "schema": schema})
        return self.response


class _SequencedStructuredLlm(_StructuredCaptureLlm):
    def __init__(self, responses):
        super().__init__({})
        self.responses = list(responses)

    def generate_structured(
        self, prompt, system, schema, *, max_tokens=None
    ):
        self.calls.append({
            "prompt": prompt, "system": system, "schema": schema,
            "max_tokens": max_tokens,
        })
        return self.responses.pop(0) if self.responses else {}


class ClinicalRegistryV2Test(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = DatabaseEngine(Path(self.tmp.name) / "emr.sqlite")
        init_database(self.db)
        _seed(self.db)

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def test_partial_and_retroactive_dates_preserve_precision(self):
        month = normalize_clinical_date("marzo 2024")
        self.assertEqual((month.start, month.precision), ("2024-03", "month"))
        relative = normalize_clinical_date(
            "da due mesi", document_date="2025-05-31"
        )
        self.assertEqual(relative.start, "2025-03-31")
        self.assertEqual(relative.precision, "approximate")
        self.assertEqual(relative.source, "retrospective_duration")
        day_without_year = normalize_clinical_date(
            "15 febbraio", document_date="2024-03-20"
        )
        self.assertEqual(
            (day_without_year.start, day_without_year.precision),
            ("2024-02-15", "day"),
        )
        undated = normalize_clinical_date(
            None, document_date="2025-05-31"
        )
        self.assertIsNone(undated.start)
        self.assertEqual(undated.source, "unknown")

    def test_undated_event_keeps_documentation_date_as_separate_basis(self):
        evidence = ClinicalEvidence(
            evidence_id="E_UNDATED", patient_id="P001", document_id="D1",
            category="diagnosis", normalized_entity="ipertensione",
            source_text="Ipertensione arteriosa in anamnesi",
            observed_date=None, document_date="2025-01-10",
        )
        event = ClinicalConsolidator().consolidate(
            "P001", [evidence]
        )[0].event
        self.assertEqual(event.first_evidence_date, "2025-01-10")
        self.assertEqual(event.first_documented_date, "2025-01-10")
        self.assertEqual(event.date_precision, "day")
        self.assertEqual(
            event.structured_data["first_evidence_basis"],
            "first_documented_date",
        )

    def test_chunker_does_not_mistake_standalone_numbers_for_pages(self):
        text = "Dose\n\n10\n\n--- PAGINA 2 ---\nDispnea\n\n<!-- page:3 -->\nTAC"
        chunks = split_text_chunks(text, 1000)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].page_start, 2)
        self.assertEqual(chunks[0].page_end, 3)
        self.assertIn("10", chunks[0].text)

    def test_chunker_keeps_date_heading_with_following_clinical_section(self):
        text = (
            ("Anamnesi remota invariata. " * 34)
            + "\n\n10/01/2025:\n\n"
            + ("Riferisce dispnea e tosse. " * 18)
        )
        chunks = split_text_chunks(text, 1000)
        self.assertEqual(len(chunks), 2)
        self.assertNotIn("10/01/2025", chunks[0].text)
        self.assertTrue(chunks[1].text.startswith("10/01/2025:"))
        self.assertIn("dispnea", chunks[1].text)

    def test_atomic_extraction_keeps_exact_quote_and_page(self):
        extractor = AtomicEvidenceExtractor(_AtomicLlm())
        evidence = extractor.extract_document(
            patient_id="P001", document_id="D1",
            document_type="visita", document_date="2025-01-10",
            text="--- PAGINA 2 ---\nLa paziente riferisce dispnea da sforzo.",
        )
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].source_page, 2)
        self.assertTrue(evidence[0].data["quote_verified"])
        self.assertIn("dispnea da sforzo", evidence[0].source_text)

    def test_atomic_dedup_collapses_same_fact_across_documents(self):
        quote = "Pregresso melanoma metastatico con localizzazioni polmonari."
        first = ClinicalEvidence(
            evidence_id="E_FIRST", patient_id="P001", document_id="D1",
            category="diagnosis", normalized_entity="Melanoma_metastatico",
            source_text=quote, document_date="2025-01-10",
            assertion="present", certainty="suspected", confidence=0.7,
            data={"quote_verified": True},
        )
        copied = ClinicalEvidence(
            evidence_id="E_COPY", patient_id="P001", document_id="D2",
            category="diagnosis", normalized_entity="melanoma metastatico",
            source_text=quote.lower(), document_date="2025-01-12",
            assertion="present", certainty="confirmed", confidence=0.9,
            data={"quote_verified": True},
        )

        unique = deduplicate_atomic_evidence([copied, first])

        self.assertEqual(len(unique), 1)
        self.assertEqual(unique[0].evidence_id, "E_FIRST")
        self.assertEqual(unique[0].document_id, "D1")
        self.assertEqual(unique[0].certainty, "confirmed")
        self.assertEqual(
            unique[0].data["duplicate_source_evidence_ids"], ["E_COPY"]
        )
        self.assertEqual(len(unique[0].data["source_occurrences"]), 2)

    def test_atomic_dedup_preserves_true_new_occurrences(self):
        common = dict(
            patient_id="P001", category="laboratory_finding",
            normalized_entity="tsh", source_text="TSH rilevato al controllo",
            assertion="present", unit="mUI/L",
            data={"quote_verified": True},
        )
        evidence = [
            ClinicalEvidence(
                evidence_id="E_DATE_1", document_id="D1",
                observed_date="2025-01-10", numeric_value=4.2, **common,
            ),
            ClinicalEvidence(
                evidence_id="E_DATE_2", document_id="D2",
                observed_date="2025-02-10", numeric_value=4.2, **common,
            ),
            ClinicalEvidence(
                evidence_id="E_VALUE", document_id="D2",
                observed_date="2025-01-10", numeric_value=7.8, **common,
            ),
        ]

        self.assertEqual(len(deduplicate_atomic_evidence(evidence)), 3)

    def test_atomic_dedup_does_not_redate_copied_relative_history(self):
        common = dict(
            patient_id="P001", category="symptom",
            normalized_entity="tosse", source_text="Tosse da circa due mesi",
            assertion="present", certainty="patient_reported",
            date_precision="approximate",
            date_source="retrospective_duration",
            data={
                "quote_verified": True,
                "date_original_text": "da circa due mesi",
            },
        )
        first = ClinicalEvidence(
            evidence_id="E_REL_1", document_id="D1",
            document_date="2025-01-10", observed_date="2024-11-10",
            **common,
        )
        copied = ClinicalEvidence(
            evidence_id="E_REL_2", document_id="D2",
            document_date="2025-03-10", observed_date="2025-01-10",
            **common,
        )

        unique = deduplicate_atomic_evidence([first, copied])

        self.assertEqual(len(unique), 1)
        self.assertEqual(unique[0].observed_date, "2024-11-10")

    def test_atomic_v6_schema_is_explicit_strict_and_excludes_labs_from_llm(self):
        prompt = build_atomic_prompt(
            TextChunk(0, "Dispnea.", 1, 1),
            document_type="visita", document_date="2025-01-10",
        )
        fixed_prompt = prompt.removesuffix("Dispnea.")
        self.assertLess(len(fixed_prompt), 180)
        self.assertIn("[S1] Dispnea.", prompt)
        self.assertNotIn("1 item=1 concetto", prompt)
        item_schema = ATOMIC_EVIDENCE_SCHEMA["properties"][
            "radiology_finding"
        ]["items"]
        properties = item_schema["properties"]
        self.assertEqual(item_schema["type"], "object")
        self.assertFalse(item_schema["additionalProperties"])
        self.assertEqual(
            set(item_schema["required"]),
            {"concept", "polarity", "source_refs"},
        )
        buckets = ATOMIC_EVIDENCE_SCHEMA["properties"]
        self.assertIn("medication", buckets)
        self.assertIn("radiology_finding", buckets)
        self.assertIn("clinical_decision", buckets)
        self.assertNotIn("laboratory_test", buckets)
        self.assertEqual(
            properties["polarity"]["enum"],
            ["present", "negated", "suspected"],
        )
        medication_properties = buckets["medication"]["items"]["properties"]
        self.assertIn(
            "active_ingredient",
            medication_properties["medication"]["properties"],
        )
        self.assertNotIn("source_text", properties)
        self.assertNotIn("document_date", properties)
        self.assertNotIn("confidence", properties)
        self.assertTrue(ATOMIC_PROMPT_VERSION.startswith("atomic_evidence_it_v10"))
        self.assertEqual(len(ATOMIC_PROMPT_DIGEST), 64)

    def test_atomic_v8_dynamic_schema_prevents_invalid_citation_ids(self):
        schema = build_atomic_evidence_schema(3)
        refs = schema["properties"]["diagnosis"]["items"]["properties"][
            "source_refs"
        ]
        self.assertEqual(refs["items"]["enum"], [1, 2, 3])
        self.assertEqual(
            set(schema["required"]), set(schema["properties"])
        )

    def test_atomic_v10_one_pass_repairs_transitions_dates_and_negatives(self):
        llm = _StructuredCaptureLlm({
            "medication": [{
                "concept": "nivolumab", "polarity": "present",
                "source_refs": [1, 2],
            }],
            "radiology_finding": [{
                "concept": "opacità a vetro smerigliato",
                "polarity": "negated", "source_refs": [3],
                "payload": {"modality": "TC", "comparison": "riduzione"},
            }],
            "histopathology": [{
                "concept": "melanoma ulcerato", "polarity": "present",
                "source_refs": [4],
                "payload": {"biomarkers": ["Breslow 3,2 mm"]},
            }],
            "biomarker": [{
                "concept": "BRAF", "polarity": "negated",
                "source_refs": [5],
                "payload": {"method": "genetico", "specimen": "cute"},
            }],
            "instrumental_finding": [{
                "concept": "broncoscopia con BAL", "polarity": "present",
                "source_refs": [6], "value_text": "negativa per infezioni",
            }, {
                "concept": "TSH", "polarity": "present",
                "source_refs": [7], "numeric_value": 0.02,
                "unit": "mUI/L",
            }],
        })
        extractor = AtomicEvidenceExtractor(llm)
        items = extractor.extract_document(
            patient_id="P001", document_id="D1", document_type="oncologia",
            document_date="2024-03-20",
            text=(
                "Il 10 gennaio 2024 è stato iniziato nivolumab. "
                "Il 16 febbraio nivolumab è stato sospeso. "
                "La TC del 5 marzo mostrava riduzione delle opacità a vetro "
                "smerigliato, senza embolia polmonare. "
                "La biopsia originaria mostrava melanoma ulcerato con "
                "Breslow 3,2 mm. BRAF non mutato. Il 20 marzo è stata "
                "eseguita broncoscopia con BAL, negativa per infezioni. "
                "Gli esami mostravano TSH 0,02 mUI/L, sotto il range."
            ),
        )
        self.assertEqual(len(llm.calls), 1)
        medication = [item for item in items if item.category == "medication"]
        self.assertEqual(
            [(item.observed_date, item.clinical_status) for item in medication],
            [("2024-01-10", "started"), ("2024-02-16", "suspended")],
        )
        improved = next(
            item for item in items
            if item.normalized_entity == "opacità a vetro smerigliato"
        )
        self.assertEqual((improved.assertion, improved.clinical_status), (
            "present", "improved"
        ))
        embolism = next(
            item for item in items if "embolia" in item.normalized_entity
        )
        self.assertEqual((embolism.assertion, embolism.certainty), (
            "absent", "excluded"
        ))
        pathology = next(
            item for item in items if item.category == "histopathology"
        )
        self.assertIsNone(pathology.observed_date)
        self.assertEqual(
            pathology.typed_payload["histopathology"]["measurement"],
            "Breslow 3,2 mm",
        )
        biomarker = next(
            item for item in items if item.category == "biomarker"
        )
        self.assertEqual(
            (biomarker.assertion, biomarker.certainty, biomarker.value_text),
            ("present", "confirmed", "non mutato"),
        )
        self.assertEqual(biomarker.typed_payload, {})
        procedures = [item for item in items if item.category == "procedure"]
        self.assertEqual(
            [(item.normalized_entity, item.observed_date) for item in procedures],
            [("broncoscopia con BAL", "2024-03-20")],
        )
        self.assertFalse(any(item.normalized_entity == "TSH" for item in items))
        self.assertEqual(
            extractor.last_extraction_metrics()["coverage_retries"], 0
        )

    def test_atomic_v10_preserves_planned_followup_target_and_timing(self):
        llm = _StructuredCaptureLlm({
            "clinical_decision": [{
                "concept": "rivalutazione pneumologica",
                "polarity": "present", "source_refs": [1],
            }],
        })
        item = AtomicEvidenceExtractor(
            llm,
            policy=ClinicalPipelinePolicy(adaptive_specialized_retry=False),
        ).extract_document(
            patient_id="P001", document_id="D1", document_type="visita",
            document_date="2024-03-20",
            text=(
                "È stata programmata rivalutazione pneumologica con nuova "
                "TC dopo quattro settimane."
            ),
        )[0]
        self.assertEqual(item.clinical_status, "planned")
        self.assertEqual(
            item.typed_payload["clinical_decision"]["target"], "nuova TC"
        )
        self.assertEqual(
            item.typed_payload["clinical_decision"]["timing"],
            "dopo quattro settimane",
        )

    def test_atomic_v8_normalizes_harmless_wire_variants_without_retry(self):
        llm = _StructuredCaptureLlm({
            "radiology_finding": [{
                "concept": "nodulo polmonare",
                "polarity": "presente",
                "source_refs": ["S1"],
                "numeric_value": "8,0",
                "unit": "mm",
                "payload": {
                    "modality": "TC", "measurement": "8 mm",
                    "signal_characteristics": "non specificato nel testo",
                },
            }],
        })
        extractor = AtomicEvidenceExtractor(llm)
        items = extractor.extract_document(
            patient_id="P001", document_id="D1",
            document_type="visita_oncologica",
            document_date="2025-01-10",
            text="TC: nodulo polmonare di 8 mm.",
        )
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].numeric_value, 8.0)
        self.assertTrue(items[0].data["wire_normalized"])
        self.assertEqual(
            items[0].typed_payload["radiology_finding"]["measurement"],
            "8 mm",
        )
        self.assertNotIn(
            "signal_characteristics",
            items[0].typed_payload["radiology_finding"],
        )

    def test_atomic_v9_recovers_uncovered_named_medication(self):
        llm = _SequencedStructuredLlm([
            {"diagnosis": [{
                "concept": "melanoma metastatico", "polarity": "present",
                "source_refs": [1],
            }]},
            {"medication": [{
                "concept": "Nivolumab", "polarity": "present",
                "source_refs": [1],
                "clinical_status": "started",
                "medication": {"lifecycle_status": "started"},
            }]},
        ])
        items = AtomicEvidenceExtractor(llm).extract_document(
            patient_id="P001", document_id="D1",
            document_type="visita_oncologica",
            document_date="2025-07-29",
            text=(
                "Paziente con melanoma metastatico: il 29.07.25 inizia "
                "rechallenge con Nivolumab."
            ),
        )
        medication = [item for item in items if item.category == "medication"]
        self.assertEqual(len(llm.calls), 2)
        self.assertEqual([item.normalized_entity for item in medication], [
            "Nivolumab"
        ])
        self.assertEqual(medication[0].clinical_status, "started")
        self.assertEqual(
            AtomicEvidenceExtractor(llm).policy.adaptive_specialized_retry,
            True,
        )

    def test_atomic_v9_repairs_medication_bucket_and_rejects_numeric_vital(self):
        llm = _StructuredCaptureLlm({
            "procedure": [{
                "concept": "terapia", "polarity": "present",
                "source_refs": [1],
            }],
            "vital_sign": [{
                "concept": "3X 2,5", "polarity": "present",
                "source_refs": [2],
            }],
        })
        items = AtomicEvidenceExtractor(
            llm,
            policy=ClinicalPipelinePolicy(adaptive_specialized_retry=False),
        ).extract_document(
            patient_id="P001", document_id="D1",
            document_type="visita_oncologica",
            document_date="2026-03-02",
            text=(
                "18.03.26 Terapia con Nivolumab. "
                "Linfoadenomegalia inguinale destra di 3X 2,5 cm."
            ),
        )
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].category, "medication")
        self.assertEqual(items[0].normalized_entity.casefold(), "nivolumab")

    def test_content_hint_follows_radiology_text_not_wrong_declared_label(self):
        text = (
            "Area iperintensa in T2 con restrizione del segnale in "
            "diffusione e potenziamento contrastografico."
        )
        self.assertEqual(infer_document_content_type(text), "radiology")
        prompt = build_atomic_prompt(
            TextChunk(0, text), document_type="visita_oncologica",
            document_date="2025-01-10",
        )
        self.assertIn("tipo_dichiarato=visita_oncologica", prompt)
        self.assertIn("contenuto_probabile=radiology", prompt)

    def test_atomic_cancel_stops_before_calling_the_model(self):
        llm = _StructuredCaptureLlm({})
        extractor = AtomicEvidenceExtractor(llm)
        with self.assertRaises(AtomicExtractionCancelled):
            extractor.extract_document(
                patient_id="P001", document_id="D1",
                document_type="visita", document_date="2025-01-10",
                text="Dispnea.", cancel_check=lambda: True,
            )
        self.assertEqual(llm.calls, [])

    def test_normal_variant_is_archived_but_excluded_from_registry(self):
        llm = _StructuredCaptureLlm({
            "radiology_finding": [{
                "concept": "utero antiversoflesso",
                "polarity": "present", "source_refs": [1],
            }],
        })
        item = AtomicEvidenceExtractor(llm).extract_document(
            patient_id="P001", document_id="D1",
            document_type="radiologia", document_date="2025-01-10",
            text=(
                "Utero antiversoflesso con diametro di 7 cm. "
                "Piccola cisti cervicale di 5 mm."
            ),
        )[0]
        self.assertEqual(item.clinical_relevance, "excluded_non_informative")
        self.assertEqual(
            item.data["registry_role_reason"], "routine_normal_finding"
        )

    def test_direct_negative_is_grounded_as_polarity_not_absence_concept(self):
        llm = _StructuredCaptureLlm({
            "radiology_finding": [{
                "concept": "assenza di ulteriori lesioni ossee",
                "polarity": "present", "source_refs": [1],
            }],
        })
        item = AtomicEvidenceExtractor(llm).extract_document(
            patient_id="P001", document_id="D1",
            document_type="radiologia", document_date="2025-01-10",
            text="Attualmente non si dimostrano ulteriori lesioni ossee.",
        )[0]
        self.assertEqual(item.normalized_entity, "lesioni ossee")
        self.assertEqual(item.assertion, "absent")
        self.assertEqual(item.data["polarity"], "negated")
        self.assertEqual(item.data["registry_role"], "contextual")

    def test_referential_imaging_attribute_stays_in_the_same_atom(self):
        llm = _StructuredCaptureLlm({
            "radiology_finding": [
                {
                    "concept": "formazione glutea",
                    "polarity": "present", "source_refs": [1],
                    "payload": {"measurement": "13 x 8 mm"},
                },
                {
                    "concept": "rapporto con il muscolo gluteo",
                    "polarity": "present", "source_refs": [2],
                    "payload": {
                        "relation_to_adjacent_structures": "piano di clivaggio"
                    },
                },
            ],
        })
        items = AtomicEvidenceExtractor(llm).extract_document(
            patient_id="P001", document_id="D1",
            document_type="radiologia", document_date="2025-01-10",
            text=(
                "Formazione glutea di 13 x 8 mm. "
                "Essa è in rapporto con il muscolo gluteo con piano di "
                "clivaggio conservato."
            ),
        )
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].data["sentence_refs"], [1, 2])
        relation = items[0].typed_payload["radiology_finding"][
            "relation_to_adjacent_structures"
        ]
        self.assertIn("muscolo gluteo", relation)

    def test_unemitted_referential_sentence_extends_previous_atom_source(self):
        llm = _StructuredCaptureLlm({
            "radiology_finding": [{
                "concept": "formazione glutea",
                "polarity": "present", "source_refs": [1],
                "payload": {"measurement": "13 x 8 mm"},
            }],
        })
        items = AtomicEvidenceExtractor(llm).extract_document(
            patient_id="P001", document_id="D1",
            document_type="radiologia", document_date="2025-01-10",
            text=(
                "Formazione glutea di 13 x 8 mm. "
                "Essa è in rapporto con il muscolo gluteo con piano di "
                "clivaggio conservato."
            ),
        )
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].data["sentence_refs"], [1, 2])
        self.assertIn("piano di clivaggio", items[0].source_text)

    def test_atomic_v6_maps_polarity_dates_and_exact_source_reference(self):
        llm = _StructuredCaptureLlm({
            "evidence": [{
                "fact_type": "diagnosis",
                "concept": "polmonite immuno-mediata",
                "polarity": "suspected",
                "observation_date": "13/03/2025",
                "source_refs": [2],
                "severity": "grado 2",
            }],
        })
        item = AtomicEvidenceExtractor(llm).extract_document(
            patient_id="P001", document_id="D1",
            document_type="visita", document_date="2025-03-20",
            text=(
                "Controllo oncologico.\n"
                "Il 13/03/2025 sospetta polmonite immuno-mediata di grado 2."
            ),
        )[0]

        self.assertEqual(item.assertion, "present")
        self.assertEqual(item.certainty, "suspected")
        self.assertEqual(item.observed_date, "2025-03-13")
        self.assertEqual(item.document_date, "2025-03-20")
        self.assertEqual(item.data["polarity"], "suspected")
        self.assertEqual(item.data["report_date"], "2025-03-20")
        self.assertEqual(
            item.data["source_reference"]["passage"],
            "Il 13/03/2025 sospetta polmonite immuno-mediata di grado 2.",
        )
        self.assertEqual(item.schema_version, "3.0")

    def test_atomic_v6_rejects_payload_outside_the_rigid_schema(self):
        llm = _StructuredCaptureLlm({
            "evidence": [{
                "fact_type": "diagnosis",
                "concept": "melanoma",
                "polarity": "present",
                "source_refs": [1],
                "invented_field": "not allowed",
            }],
        })
        items = AtomicEvidenceExtractor(llm).extract_document(
            patient_id="P001", document_id="D1",
            document_type="visita", document_date="2025-03-20",
            text="Diagnosi di melanoma.",
        )
        self.assertEqual(items, [])

    def test_atomic_extractor_rejects_llm_generated_laboratory_values(self):
        llm = _StructuredCaptureLlm({
            "evidence": [{
                "category": "laboratory_finding",
                "normalized_entity": "PCR",
                "source_text": "PCR 12 mg/dL",
                "assertion": "present",
                "certainty": "confirmed",
                "numeric_value": 12,
                "unit": "mg/dL",
            }],
        })
        items = AtomicEvidenceExtractor(llm).extract_document(
            patient_id="P001", document_id="D1",
            document_type="visita", document_date="2025-03-20",
            text="PCR 12 mg/dL",
        )
        self.assertEqual(items, [])

    def test_atomic_v5_resolves_sentence_refs_to_exact_source(self):
        llm = _StructuredCaptureLlm({
            "evidence": [[
                "symptom", "dispnea", [2], "present",
                "patient_reported", "clinically_relevant", {},
            ]],
        })
        item = AtomicEvidenceExtractor(llm).extract_document(
            patient_id="P001", document_id="D1",
            document_type="visita", document_date="2025-01-10",
            text="Obiettività invariata.\nLa paziente riferisce dispnea da sforzo.",
        )[0]
        self.assertEqual(
            item.source_text, "La paziente riferisce dispnea da sforzo."
        )
        self.assertEqual(item.data["sentence_refs"], [2])
        self.assertTrue(item.data["quote_verified"])
        self.assertIn("[S2] La paziente", llm.calls[0]["prompt"])

    def test_atomic_v5_compact_wire_preserves_nested_clinical_fields(self):
        llm = _StructuredCaptureLlm({
            "evidence": [[
                "medication", "nivolumab", [1], "present", "confirmed",
                "high", {
                    "d": "10/01/2025", "s": "started",
                    "m": {
                    "n": "Nivolumab", "i": "nivolumab", "l": "started",
                    "d": "240 mg", "r": "EV", "f": "ogni 2 settimane",
                    },
                    "o": {
                    "l": "prima linea", "r": ["nivolumab"], "c": "1",
                    "s": "metastatico", "i": "palliativo",
                    },
                },
            ]],
        })
        item = AtomicEvidenceExtractor(llm).extract_document(
            patient_id="P001", document_id="D1",
            document_type="oncologia", document_date="2025-01-10",
            text="Il 10/01/2025 avviato nivolumab 240 mg EV ogni 2 settimane.",
        )[0]
        self.assertEqual(item.normalized_entity, "nivolumab")
        self.assertEqual(item.observed_date, "2025-01-10")
        self.assertEqual(item.data["therapy"]["lifecycle_status"], "started")
        self.assertEqual(item.data["therapy"]["dose"], "240 mg")
        self.assertEqual(item.data["oncology"]["line_label"], "prima linea")
        self.assertEqual(item.data["oncology"]["regimen"], ["nivolumab"])

    def test_atomic_v5_tuple_materially_reduces_repeated_json(self):
        legacy = {
            "category": "imaging_finding",
            "entity": "nodulo polmonare",
            "refs": [2, 3],
            "assertion": "present",
            "certainty": "confirmed",
            "significance": "potentially_relevant",
        }
        compact = [
            "imaging_finding", "nodulo polmonare", [2, 3], "present",
            "confirmed", "potentially_relevant", {},
        ]
        old_size = len(json.dumps(legacy, separators=(",", ":")))
        new_size = len(json.dumps(compact, separators=(",", ":")))
        self.assertLess(new_size, old_size * 0.7)

    def test_long_source_is_split_before_any_output_limit_failure(self):
        class DenseSafeLlm:
            model = "fake-medical"
            context_length = 32768
            max_output_tokens = 6144

            def __init__(self):
                self.calls = []

            def generate_structured(
                self, prompt, system, schema, *, max_tokens=None
            ):
                source = prompt.split("TESTO:\n", 1)[-1]
                self.calls.append((len(source), max_tokens))
                if len(source) > 4500:
                    raise OutputLimitError(max_tokens or 6144)
                return {"evidence": []}

        llm = DenseSafeLlm()
        extractor = AtomicEvidenceExtractor(llm)
        text = "\n\n".join(
            f"Sezione {index}. " + ("Dato clinico documentato. " * 55)
            for index in range(4)
        )
        extractor.extract_document(
            patient_id="P001", document_id="D1",
            document_type="visita", document_date="2025-01-10", text=text,
        )
        metrics = extractor.last_extraction_metrics()
        self.assertGreater(metrics["source_chunks"], 1)
        self.assertEqual(metrics["llm_calls"], metrics["source_chunks"])
        self.assertEqual(metrics["output_limit_retries"], 0)
        self.assertTrue(all(length <= 4500 for length, _ in llm.calls))

    def test_atomic_output_limit_bisects_and_retries_without_losing_spans(self):
        class AdaptiveLlm:
            model = "fake-medical"
            context_length = 4096
            max_output_tokens = 1024

            def __init__(self):
                self.calls = []

            def generate_structured(
                self, prompt, system, schema, *, max_tokens=None
            ):
                self.calls.append(prompt)
                refs = re.findall(r"\[S(\d+)\]", prompt)
                if len(refs) > 1:
                    raise OutputLimitError(max_tokens or 1024)
                return {"evidence": [{
                    "category": "symptom", "entity": "dispnea",
                    "refs": [1], "assertion": "present",
                    "certainty": "patient_reported",
                    "significance": "clinically_relevant",
                }]}

        llm = AdaptiveLlm()
        extractor = AtomicEvidenceExtractor(llm)
        items = extractor.extract_document(
            patient_id="P001", document_id="D1",
            document_type="visita", document_date="2025-01-10",
            text="Riferisce dispnea. Riferisce tosse.",
        )
        self.assertEqual(len(llm.calls), 3)
        self.assertEqual(len(items), 2)
        self.assertTrue(all(
            item.data["adaptive_retry_depth"] == 1 for item in items
        ))
        self.assertEqual(extractor.last_extraction_metrics()["llm_calls"], 3)
        self.assertEqual(
            extractor.last_extraction_metrics()["output_limit_retries"], 1
        )

    def test_atomic_post_validation_fixes_real_model_edge_cases(self):
        llm = _StructuredCaptureLlm({
            "evidence": [
                {
                    "category": "vital_sign", "entity": "SpO2",
                    "refs": [2], "assertion": "absent",
                    "certainty": "confirmed", "number": 88, "unit": "%",
                },
                {
                    "category": "vital_sign", "entity": "SpO2",
                    "refs": [2], "assertion": "present",
                    "certainty": "confirmed", "number": 88, "unit": "%",
                },
                {
                    "category": "diagnosis",
                    "entity": "polmonite immuno-mediata", "refs": [3],
                    "assertion": "present", "certainty": "confirmed",
                },
                {
                    "category": "diagnosis",
                    "entity": "polmonite immuno-mediata", "refs": [4],
                    "assertion": "present", "certainty": "confirmed",
                    "extra": {"grade": "2"},
                },
                {
                    "category": "medication", "entity": "nivolumab",
                    "refs": [5], "assertion": "present",
                    "certainty": "confirmed",
                    "drug": {"name": "nivolumab", "lifecycle": "suspended"},
                },
            ],
        })
        text = (
            "10/03/2025 Valutazione clinica.\n"
            "SpO2 88% in aria ambiente.\n"
            "13/03/2025: quadro compatibile con polmonite;\n"
            "immuno-mediata grado 2 durante terapia con nivolumab.\n"
            "Nivolumab sospeso."
        )
        items = AtomicEvidenceExtractor(llm).extract_document(
            patient_id="P001", document_id="D1",
            document_type="visita", document_date="2025-03-25", text=text,
        )
        spo2 = [item for item in items if item.normalized_entity == "SpO2"]
        self.assertEqual(len(spo2), 1)
        self.assertEqual(spo2[0].assertion, "present")
        self.assertEqual(spo2[0].observed_date, "2025-03-10")
        self.assertEqual(spo2[0].data["date_context_sentence_ref"], 1)
        toxicity = [item for item in items if item.category == "toxicity"]
        self.assertEqual(len(toxicity), 1)
        self.assertEqual(toxicity[0].certainty, "suspected")
        self.assertEqual(toxicity[0].severity, "grado 2")
        self.assertEqual(toxicity[0].data["sentence_refs"], [3, 4])
        medication = [item for item in items if item.category == "medication"]
        self.assertEqual(medication[0].observed_date, "2025-03-13")
        self.assertEqual(medication[0].clinical_status, "suspended")

    def test_sentence_spans_preserve_numbered_lines(self):
        spans = split_sentence_spans("Prima frase.\nSeconda frase; terza.")
        self.assertEqual([span.sentence_id for span in spans], [1, 2, 3])
        self.assertEqual(spans[1].text, "Seconda frase;")

    def test_sentence_spans_do_not_treat_pdf_line_wrap_as_new_sentence(self):
        spans = split_sentence_spans(
            "Riferisce dispnea da sforzo\ninsorta da circa cinque giorni."
        )
        self.assertEqual(len(spans), 1)
        self.assertIn("sforzo\ninsorta", spans[0].text)

    def test_retrospective_duration_uses_local_event_date_as_anchor(self):
        llm = _StructuredCaptureLlm({
            "evidence": [{
                "category": "symptom", "entity": "dispnea", "refs": [1],
                "assertion": "present", "certainty": "patient_reported",
                "date": "da circa 5 giorni",
            }],
        })
        item = AtomicEvidenceExtractor(llm).extract_document(
            patient_id="P001", document_id="D1",
            document_type="visita", document_date="2025-03-25",
            text=(
                "10/03/2025: riferisce dispnea da sforzo insorta "
                "da circa 5 giorni."
            ),
        )[0]
        self.assertEqual(item.observed_date, "2025-03-05")
        self.assertEqual(item.date_precision, "approximate")

    def test_explicit_symptom_resolution_is_recovered_deterministically(self):
        llm = _StructuredCaptureLlm({
            "evidence": [{
                "category": "vital_sign", "entity": "SpO2", "refs": [1],
                "assertion": "present", "certainty": "confirmed",
                "number": 96, "unit": "%",
            }],
        })
        items = AtomicEvidenceExtractor(llm).extract_document(
            patient_id="P001", document_id="D1",
            document_type="controllo", document_date="2025-03-25",
            text=(
                "Al controllo del 25/03/2025 la dispnea è risolta, "
                "SpO2 96% in aria ambiente."
            ),
        )
        resolution = [
            item for item in items
            if item.normalized_entity == "dispnea" and item.assertion == "absent"
        ]
        self.assertEqual(len(resolution), 1)
        self.assertEqual(resolution[0].clinical_status, "resolved")
        self.assertEqual(resolution[0].observed_date, "2025-03-25")
        self.assertTrue(
            resolution[0].data["deterministic_explicit_resolution"]
        )

    def test_grade_is_removed_from_toxicity_entity_and_kept_as_severity(self):
        llm = _StructuredCaptureLlm({
            "evidence": [{
                "category": "toxicity",
                "entity": "polmonite immuno-mediata di grado 2",
                "refs": [1], "assertion": "present",
                "certainty": "suspected",
            }],
        })
        item = AtomicEvidenceExtractor(llm).extract_document(
            patient_id="P001", document_id="D1",
            document_type="visita", document_date="2025-03-13",
            text=(
                "Quadro compatibile con polmonite immuno-mediata "
                "di grado 2 durante terapia."
            ),
        )[0]
        self.assertEqual(item.normalized_entity, "polmonite immuno-mediata")
        self.assertEqual(item.severity, "grado 2")

    def test_new_processing_run_marks_stale_work_as_interrupted(self):
        repo = ProcessingRepository(self.db)
        old = ProcessingRun(patient_id="P001", stage="clinical_registry_v2")
        repo.start_run(old)
        manifest = ProcessingManifestItem(
            patient_id="P001", document_id="D1", stage="atomic_evidence",
            input_hash="h", pipeline_version="v", run_id=old.run_id,
            status="running",
        )
        repo.upsert_manifest(manifest)

        new = ProcessingRun(patient_id="P001", stage="clinical_registry_v2")
        repo.start_run(new)

        old_row = self.db.execute(
            "SELECT status, completed_at FROM processing_runs WHERE run_id=?",
            (old.run_id,),
        ).fetchone()
        manifest_row = self.db.execute(
            "SELECT status, error_message FROM processing_manifest "
            "WHERE manifest_id=?",
            (manifest.manifest_id,),
        ).fetchone()
        self.assertEqual(old_row["status"], "interrupted")
        self.assertTrue(old_row["completed_at"])
        self.assertEqual(manifest_row["status"], "failed")
        self.assertIn("ripresa", manifest_row["error_message"])

    def test_completed_v3_manifest_can_resume_under_compatible_v4_pipeline(self):
        repo = ProcessingRepository(self.db)
        run = ProcessingRun(patient_id="P001", stage="clinical_registry_v2")
        repo.start_run(run)
        manifest = ProcessingManifestItem(
            patient_id="P001", document_id="D1", stage="atomic_evidence",
            input_hash="same-input", pipeline_version="registry_pipeline_v3",
            prompt_version="atomic_evidence_it_v5", model_digest="model-a",
            run_id=run.run_id, status="running",
        )
        repo.upsert_manifest(manifest)
        repo.mark_result(manifest.manifest_id, status="completed")

        self.assertFalse(repo.is_current(
            "D1", "atomic_evidence", "same-input", "registry_pipeline_v4",
            "atomic_evidence_it_v5", "model-a",
        ))
        self.assertTrue(repo.is_current(
            "D1", "atomic_evidence", "same-input", "registry_pipeline_v4",
            "atomic_evidence_it_v5", "model-a",
            compatible_pipeline_versions=("registry_pipeline_v3",),
        ))
        self.assertFalse(repo.is_current(
            "D1", "atomic_evidence", "same-input", "registry_pipeline_v4",
            "different-prompt", "model-a",
            compatible_pipeline_versions=("registry_pipeline_v3",),
        ))

    def test_exact_block_reuse_removes_only_later_same_section_and_type(self):
        repeated = (
            "Anamnesi oncologica con melanoma metastatico trattato con "
            "pembrolizumab in prima linea e successiva risposta parziale."
        )
        docs = [
            type("Doc", (), {"id": "D1", "document_type": "visita"})(),
            type("Doc", (), {"id": "D2", "document_type": "visita"})(),
            type("Doc", (), {"id": "D3", "document_type": "radiologia"})(),
        ]
        plans = plan_exact_block_reuse([
            (docs[0], f"ANAMNESI:\n{repeated}\nDato nuovo A", "h1"),
            (docs[1], f"ANAMNESI:\n{repeated}\nDato nuovo B", "h2"),
            (docs[2], f"ANAMNESI:\n{repeated}", "h3"),
        ])
        self.assertEqual(len(plans["D2"].reuse_links), 1)
        self.assertNotIn(repeated, plans["D2"].extraction_text)
        self.assertIn("Dato nuovo B", plans["D2"].extraction_text)
        self.assertEqual(len(plans["D3"].reuse_links), 0)
        self.assertIn(repeated, plans["D3"].extraction_text)

    def test_reuse_verification_deduplicates_fingerprint_and_keeps_context(self):
        repeated = (
            "Anamnesi oncologica con melanoma metastatico trattato con "
            "pembrolizumab e successiva risposta parziale documentata."
        )
        docs = [
            type("Doc", (), {"id": name, "document_type": "visita"})()
            for name in ("D1", "D2", "D3")
        ]
        plans = plan_exact_block_reuse([
            (docs[0], f"ANAMNESI:\n{repeated}", "h1"),
            (docs[1], f"ANAMNESI:\n{repeated}", "h2"),
            (docs[2], f"ANAMNESI:\n{repeated}", "h3"),
        ])
        links = [
            *plans["D2"].reuse_links, *plans["D3"].reuse_links,
        ]
        batches = plan_reuse_verification_batches(links)
        self.assertEqual(len(batches), 1)
        self.assertEqual(len(batches[0].links), 1)
        self.assertEqual(batches[0].source_document_id, "D1")
        self.assertEqual(batches[0].text.count(repeated), 1)
        self.assertTrue(batches[0].text.startswith("ANAMNESI:"))

    def test_contextual_verified_quote_can_be_cloned_when_target_contains_it(self):
        repeated = (
            "Melanoma metastatico trattato con pembrolizumab in prima linea "
            "con risposta parziale confermata radiologicamente."
        )
        source_full = f"ANAMNESI:\n{repeated}"
        doc1 = type("Doc", (), {"id": "D1", "document_type": "visita"})()
        doc2 = type("Doc", (), {"id": "D2", "document_type": "visita"})()
        link = plan_exact_block_reuse([
            (doc1, source_full, "h1"), (doc2, source_full, "h2"),
        ])["D2"].reuse_links[0]
        source = ClinicalEvidence(
            evidence_id="E_CONTEXT", patient_id="P001", document_id="D1",
            category="diagnosis", normalized_entity="melanoma metastatico",
            source_text=source_full, document_date="2025-01-10",
        )
        self.assertEqual(
            evidence_matching_reuse_block([source], link), [source]
        )
        clones = clone_reused_evidence(
            [source], link, patient_id="P001",
            target_document_date="2025-01-12", target_full_text=source_full,
        )
        self.assertEqual(len(clones), 1)
        self.assertEqual(clones[0].document_id, "D2")

    def test_targeted_reuse_text_keeps_page_and_deduplicates_occurrence(self):
        repeated = (
            "Melanoma metastatico in trattamento oncologico con risposta "
            "parziale stabile all'ultima rivalutazione disponibile."
        )
        doc1 = type("Doc", (), {"id": "D1", "document_type": "visita"})()
        doc2 = type("Doc", (), {"id": "D2", "document_type": "visita"})()
        text = f"--- PAGINA 2 ---\nANAMNESI:\n{repeated}\n{repeated}"
        links = plan_exact_block_reuse([
            (doc1, f"ANAMNESI:\n{repeated}", "h1"),
            (doc2, text, "h2"),
        ])["D2"].reuse_links
        targeted = build_targeted_reuse_text(links)
        self.assertIn("--- PAGINA 2 ---", targeted)
        self.assertIn("ANAMNESI:", targeted)
        self.assertEqual(targeted.count(repeated), 1)

    def test_registry_verifies_unique_missing_block_once_without_full_fallback(self):
        repeated = (
            "Melanoma metastatico trattato con pembrolizumab in prima linea "
            "con risposta parziale confermata alla rivalutazione radiologica."
        )

        class FocusedVerifierLlm:
            model = "fake-medical"
            context_length = 4096
            max_output_tokens = 1024
            temperature = 0.0
            top_p = 0.9
            top_k = 40
            seed = 42
            is_available = True

            def __init__(self):
                self.atomic_calls = []

            def generate_structured(
                self, prompt, system, schema, *, max_tokens=None
            ):
                if "claims" in schema.get("properties", {}):
                    return {"claims": [], "conflicting_evidence_ids": []}
                self.atomic_calls.append(prompt)
                # Simulate a miss in the full source document.  The focused
                # second pass sees the repeated block without unrelated text.
                if repeated not in prompt or "Dato nuovo sorgente" in prompt:
                    return {"evidence": []}
                match = re.search(
                    rf"\[S(\d+)\][^\n]*{re.escape(repeated[:35])}", prompt
                )
                ref = int(match.group(1)) if match else 1
                return {"evidence": [[
                    "diagnosis", "melanoma metastatico", [ref], "present",
                    "confirmed", "high", {},
                ]]}

        old_workspace = active_workspace.path
        workspace = Path(self.tmp.name) / "workspace"
        extraction = workspace / "P001" / "extraction"
        extraction.mkdir(parents=True)
        (extraction / "D1.md").write_text(
            f"ANAMNESI:\n{repeated}\nDato nuovo sorgente.",
            encoding="utf-8",
        )
        (extraction / "D2.md").write_text(
            f"ANAMNESI:\n{repeated}\nDato nuovo target.",
            encoding="utf-8",
        )
        active_workspace.set_path(workspace)
        llm = FocusedVerifierLlm()
        evidence_repo = EvidenceRepository(self.db)
        try:
            builder = ClinicalRegistryBuilder(
                registry_repo=ClinicalRegistryRepository(self.db),
                evidence_repo=evidence_repo,
                processing_repo=ProcessingRepository(self.db),
                timeline_repo=TimelineRepository(self.db),
                document_repo=DocumentRepository(self.db),
                lab_repo=LabRepository(self.db),
                overlay_repo=DocumentTextOverlayRepository(self.db),
                llm_client=llm,
                pipeline_policy=ClinicalPipelinePolicy(
                    adaptive_specialized_retry=False
                ),
                db=self.db,
            )
            atomic_result = builder.extract_atomic_evidence(
                "P001", incremental=False, num_workers=2,
            )
            self.assertEqual(
                ClinicalRegistryRepository(self.db).get_events("P001"), []
            )
            result = builder.build_structured_events("P001")
            calls_before_validation = len(llm.atomic_calls)
            validation_result = builder.prepare_validation("P001")
            self.assertEqual(len(llm.atomic_calls), calls_before_validation)
        finally:
            active_workspace.set_path(old_workspace)

        evidence = [
            item for item in evidence_repo.get_by_patient("P001")
            if item.normalized_entity == "melanoma metastatico"
        ]
        self.assertEqual({item.document_id for item in evidence}, {"D1", "D2"})
        self.assertEqual(result["atomic_duplicates_suppressed"], 1)
        self.assertEqual(result["total_entries"], 1)
        self.assertGreaterEqual(result["duplicate_source_links"], 1)
        registry_repo = ClinicalRegistryRepository(self.db)
        events = registry_repo.get_events("P001")
        self.assertTrue(events)
        detail = registry_repo.get_event_detail(events[0].event_id)
        relations = {
            item["relation"] for item in (detail or {}).get("evidence", [])
        }
        self.assertIn("duplicate_source", relations)
        self.assertEqual(atomic_result["stage"], "atomic")
        self.assertEqual(result["stage"], "events")
        self.assertEqual(validation_result["stage"], "validation")
        self.assertGreaterEqual(validation_result["validation_pending"], 1)
        self.assertEqual(atomic_result["unique_reuse_blocks_verified"], 1)
        self.assertGreaterEqual(atomic_result["reused_evidence"], 1)
        self.assertEqual(atomic_result["targeted_reuse_verifications"], 0)
        self.assertEqual(atomic_result["full_document_fallbacks"], 0)

    def test_reused_relative_evidence_is_redated_and_keeps_provenance(self):
        source_text = (
            "Da circa due mesi presenta tosse secca persistente senza febbre, "
            "con obiettività toracica sostanzialmente invariata."
        )
        doc1 = type("Doc", (), {"id": "D1", "document_type": "visita"})()
        doc2 = type("Doc", (), {"id": "D2", "document_type": "visita"})()
        plans = plan_exact_block_reuse([
            (doc1, source_text, "h1"), (doc2, source_text, "h2"),
        ])
        link = plans["D2"].reuse_links[0]
        source = ClinicalEvidence(
            evidence_id="E1", patient_id="P001", document_id="D1",
            category="symptom", normalized_entity="tosse",
            source_text=source_text, observed_date="2024-11-10",
            document_date="2025-01-10", date_precision="approximate",
            date_source="retrospective_duration",
            certainty="patient_reported",
            data={"date_original_text": "da circa due mesi"},
        )
        clones = clone_reused_evidence(
            [source], link, patient_id="P001",
            target_document_date="2025-03-10", target_full_text=source_text,
        )
        self.assertEqual(len(clones), 1)
        self.assertEqual(clones[0].document_id, "D2")
        self.assertEqual(clones[0].observed_date, "2025-01-10")
        self.assertTrue(clones[0].data["reused_exact_block"])
        self.assertEqual(
            clones[0].data["reused_from_document_id"], "D1"
        )

    def test_atomic_payload_discards_out_of_schema_nested_metadata(self):
        llm = _StructuredCaptureLlm({
            "evidence": [{
                "category": "oncology_treatment_line",
                "normalized_entity": "pembrolizumab",
                "source_text": "Avviato pembrolizumab 200 mg ev",
                "assertion": "present",
                "certainty": "confirmed",
                "clinical_status": "started",
                "date_precision": "day",
                "observed_date": "2025-01-10",
                "significance": "clinically_relevant",
                "confidence": 0.99,
                "therapy": {
                    "original_name": "pembrolizumab",
                    "active_ingredient": "pembrolizumab",
                    "dose": "200 mg",
                    "route": "ev",
                    "invented": "da eliminare",
                },
                "oncology": {
                    "line_label": "prima linea",
                    "regimen": "pembrolizumab",
                    "unexpected": "da eliminare",
                },
                "additional_data": {
                    "stage": "IV",
                    "unexpected": "da eliminare",
                },
            }],
        })
        item = AtomicEvidenceExtractor(llm).extract_document(
            patient_id="P001", document_id="D1",
            document_type="oncologia", document_date="2025-01-10",
            text="Avviato pembrolizumab 200 mg ev",
        )[0]
        self.assertEqual(item.category, "medication")
        self.assertEqual(item.confidence, 0.75)
        self.assertEqual(item.data["oncology"]["regimen"], ["pembrolizumab"])
        self.assertNotIn("invented", item.data["therapy"])
        self.assertNotIn("unexpected", item.data["oncology"])
        self.assertNotIn("unexpected", item.data)
        self.assertEqual(item.data["stage"], "IV")

    def test_atomic_post_validation_recovers_grounded_date_value_and_lifecycle(self):
        llm = _StructuredCaptureLlm({
            "evidence": [
                {
                    "category": "clinical_sign", "normalized_entity": "SpO2 88%",
                    "source_text": "Da circa 5 giorni SpO2 88% in aria ambiente",
                    "assertion": "present", "certainty": "confirmed",
                    "observed_date": "non specificata",
                    "date_precision": "unknown",
                    "significance": "clinically_relevant",
                },
                {
                    "category": "medication", "normalized_entity": "pembrolizumab",
                    "source_text": "si sospende pembrolizumab",
                    "assertion": "absent", "certainty": "confirmed",
                    "date_precision": "unknown",
                    "significance": "clinically_relevant",
                    "therapy": {
                        "original_name": "pembrolizumab",
                        "intent": "sospeso",
                    },
                },
            ],
        })
        items = AtomicEvidenceExtractor(llm).extract_document(
            patient_id="P001", document_id="D1",
            document_type="visita", document_date="2025-02-03",
            text=(
                "Da circa 5 giorni SpO2 88% in aria ambiente. "
                "Nel sospetto di tossicità si sospende pembrolizumab."
            ),
        )
        vital = next(item for item in items if item.category == "vital_sign")
        self.assertEqual(vital.normalized_entity, "SpO2")
        self.assertEqual((vital.numeric_value, vital.unit), (88.0, "%"))
        self.assertEqual(vital.observed_date, "2025-01-29")
        self.assertEqual(vital.date_source, "retrospective_duration")
        medication = next(item for item in items if item.category == "medication")
        self.assertEqual(medication.assertion, "present")
        self.assertEqual(medication.clinical_status, "suspended")
        self.assertEqual(
            medication.data["therapy"]["lifecycle_status"], "suspended"
        )
        self.assertNotIn("intent", medication.data["therapy"])
        event = ClinicalConsolidator().consolidate(
            "P001", [medication]
        )[0].event
        self.assertEqual(event.status, "suspended")

    def test_drug_administration_date_does_not_date_diagnosis_in_same_quote(self):
        quote = (
            "Pembrolizumab in prima linea per melanoma metastatico; "
            "ultima somministrazione il 27/01/2025."
        )
        llm = _StructuredCaptureLlm({
            "evidence": [{
                "category": "diagnosis",
                "normalized_entity": "melanoma metastatico",
                "source_text": quote,
                "assertion": "present", "certainty": "confirmed",
                "observed_date": "27/01/2025", "date_precision": "day",
                "significance": "high",
                "oncology": {"line_label": "prima linea"},
            }],
        })
        item = AtomicEvidenceExtractor(llm).extract_document(
            patient_id="P001", document_id="D1",
            document_type="oncologia", document_date="2025-02-03",
            text=quote,
        )[0]
        self.assertIsNone(item.observed_date)
        self.assertEqual(item.date_source, "unknown")
        self.assertNotIn("oncology", item.data)

    def test_undocumented_infectious_cause_is_not_a_confirmed_lab_result(self):
        quote = "Non è documentata una causa infettiva"
        llm = _StructuredCaptureLlm({
            "evidence": [{
                "category": "diagnosis",
                "normalized_entity": "infezione", "source_text": quote,
                "assertion": "absent", "certainty": "confirmed",
                "date_precision": "unknown",
                "significance": "clinically_relevant",
            }],
        })
        item = AtomicEvidenceExtractor(llm).extract_document(
            patient_id="P001", document_id="D1",
            document_type="visita", document_date="2025-02-05", text=quote,
        )[0]
        self.assertEqual(item.category, "diagnosis")
        self.assertEqual(item.certainty, "unknown")
        event = ClinicalConsolidator().consolidate(
            "P001", [item]
        )[0].event
        self.assertEqual(event.status, "unknown")

    def test_dedup_fuses_sources_and_splits_recurrence_after_resolution(self):
        evidence = [
            ClinicalEvidence(
                evidence_id="E1", patient_id="P001", document_id="D1",
                category="symptom", normalized_entity="dispnea",
                source_text="Dispnea da sforzo", observed_date="2025-01-10",
                document_date="2025-01-10", certainty="patient_reported",
            ),
            ClinicalEvidence(
                evidence_id="E2", patient_id="P001", document_id="D2",
                category="symptom", normalized_entity="dispnea",
                source_text="Dispnea associata a tosse", observed_date="2025-01-12",
                document_date="2025-01-12", certainty="confirmed",
            ),
            ClinicalEvidence(
                evidence_id="E3", patient_id="P001", document_id="D2",
                category="symptom", normalized_entity="dispnea",
                source_text="Dispnea risolta", observed_date="2025-01-20",
                document_date="2025-01-12", clinical_status="resolved",
            ),
            ClinicalEvidence(
                evidence_id="E4", patient_id="P001", document_id="D2",
                category="symptom", normalized_entity="dispnea",
                source_text="Nuova dispnea", observed_date="2025-03-01",
                document_date="2025-01-12",
            ),
        ]
        bundles = ClinicalConsolidator().consolidate("P001", evidence)
        self.assertEqual(len(bundles), 2)
        self.assertEqual(len(bundles[0].links), 3)
        self.assertEqual(bundles[0].event.first_evidence_date, "2025-01-10")
        self.assertIn("tosse", bundles[0].event.summary_detail.lower())
        self.assertEqual(bundles[1].episode.recurrence_index, 2)

    def test_fusion_v3_generates_cited_prose_once_and_compacts_input(self):
        response = {
            "status": "active",
            "certainty": "confirmed",
            "severity": None,
            "significance": "clinically_relevant",
            "claims": [{
                "text": "Dispnea da sforzo persistente, successivamente associata a tosse.",
                "evidence_ids": ["E1", "E2"],
                "certainty": "confirmed",
            }],
            "conflicting_evidence_ids": [],
            # A backend that ignores the v3 schema must not be able to inject
            # a second, uncited rendering of the note.
            "summary_short": "Testo non supportato da ignorare",
        }
        llm = _StructuredCaptureLlm(response)
        evidence = [
            ClinicalEvidence(
                evidence_id="E1", patient_id="P001", document_id="D1",
                category="symptom", normalized_entity="dispnea",
                source_text="Dispnea da sforzo", observed_date="2025-01-10",
                document_date="2025-01-10", source_page=2,
                data={"quote_verified": True, "chunk_index": 1},
            ),
            ClinicalEvidence(
                evidence_id="E2", patient_id="P001", document_id="D2",
                category="symptom", normalized_entity="dispnea",
                source_text="Dispnea associata a tosse",
                observed_date="2025-01-12", document_date="2025-01-12",
                data={"quote_verified": True, "chunk_index": 0},
            ),
        ]
        bundle = ClinicalConsolidator(fusion_llm=llm).consolidate(
            "P001", evidence
        )[0]
        expected = response["claims"][0]["text"]
        self.assertEqual(bundle.event.summary_short, expected)
        self.assertEqual(bundle.event.summary_detail, expected)
        self.assertNotIn("Testo non supportato", bundle.event.summary_short)
        sent_item = evidence_for_prompt(evidence[0])
        self.assertNotIn("document_id", sent_item)
        self.assertNotIn("source_page", sent_item)
        self.assertNotIn("data", sent_item)
        self.assertNotIn("severity", sent_item)
        self.assertLess(len(json.dumps(sent_item)), 300)
        self.assertNotIn("summary_short", FUSION_SCHEMA["properties"])
        self.assertEqual(
            set(FUSION_SCHEMA["properties"]),
            {"claims", "conflicting_evidence_ids"},
        )
        self.assertEqual(len(FUSION_PROMPT_DIGEST), 64)
        self.assertTrue(FUSION_PROMPT_VERSION.startswith("clinical_fusion_it_v3"))

    def test_identical_sources_skip_llm_and_cite_every_evidence(self):
        llm = _StructuredCaptureLlm({})
        evidence = [
            ClinicalEvidence(
                evidence_id=evidence_id, patient_id="P001",
                document_id=document_id, category="diagnosis",
                normalized_entity="ipertensione",
                source_text="Ipertensione arteriosa in anamnesi",
                document_date=date,
            )
            for evidence_id, document_id, date in (
                ("E1", "D1", "2025-01-10"),
                ("E2", "D2", "2025-01-12"),
            )
        ]
        bundle = ClinicalConsolidator(fusion_llm=llm).consolidate(
            "P001", evidence
        )[0]
        self.assertEqual(llm.calls, [])
        self.assertEqual(len(bundle.event.structured_data["claims"]), 1)
        self.assertEqual(
            set(bundle.event.structured_data["claims"][0]["evidence_ids"]),
            {"E1", "E2"},
        )
        self.assertTrue(all(link.included_in_summary for link in bundle.links))

    def test_independent_fusion_clusters_use_parallel_workers(self):
        class ParallelFusionLlm:
            model = "fake-medical"
            context_length = 4096
            max_output_tokens = 1024
            temperature = 0.1
            top_p = 0.9
            top_k = 40
            seed = 42

            def __init__(self):
                self.active = 0
                self.maximum = 0
                self.lock = threading.Lock()

            def generate_structured(
                self, prompt, system, schema, *, max_tokens=None
            ):
                with self.lock:
                    self.active += 1
                    self.maximum = max(self.maximum, self.active)
                try:
                    time.sleep(0.03)
                    ids = sorted(set(re.findall(r"E_[A-Z]+\d", prompt)))
                    return {
                        "claims": [{
                            "text": "Sintesi clinica verificata",
                            "evidence_ids": ids,
                            "certainty": "confirmed",
                        }],
                        "conflicting_evidence_ids": [],
                    }
                finally:
                    with self.lock:
                        self.active -= 1

        llm = ParallelFusionLlm()
        evidence = []
        for entity, prefix in (("dispnea", "E_D"), ("tosse", "E_T")):
            evidence.extend([
                ClinicalEvidence(
                    evidence_id=f"{prefix}{index}", patient_id="P001",
                    document_id=f"D{index}", category="symptom",
                    normalized_entity=entity,
                    source_text=f"{entity} descrizione {index}",
                    document_date=f"2025-01-1{index}",
                )
                for index in (1, 2)
            ])
        bundles = ClinicalConsolidator(fusion_llm=llm).consolidate(
            "P001", evidence, num_workers=2
        )
        self.assertEqual(len(bundles), 2)
        self.assertEqual(llm.maximum, 2)

    def test_fusion_cache_depends_on_prompt_model_and_sampling_signature(self):
        evidence = [
            ClinicalEvidence(
                evidence_id="E1", patient_id="P001", document_id="D1",
                category="symptom", normalized_entity="dispnea",
                source_text="Dispnea", observed_date="2025-01-10",
                document_date="2025-01-10",
            ),
            ClinicalEvidence(
                evidence_id="E2", patient_id="P001", document_id="D2",
                category="symptom", normalized_entity="dispnea",
                source_text="Dispnea persistente", observed_date="2025-01-12",
                document_date="2025-01-12",
            ),
        ]

        def response(text):
            return {
                "status": "active", "certainty": "confirmed",
                "severity": None, "significance": "clinically_relevant",
                "claims": [{
                    "text": text, "evidence_ids": ["E1", "E2"],
                    "certainty": "confirmed",
                }],
                "conflicting_evidence_ids": [],
            }

        first_llm = _StructuredCaptureLlm(response("Prima sintesi"))
        first = ClinicalConsolidator(fusion_llm=first_llm).consolidate(
            "P001", evidence
        )[0].event
        same_llm = _StructuredCaptureLlm(response("Non deve essere usata"))
        cached = ClinicalConsolidator(
            fusion_llm=same_llm, existing_events=[first]
        ).consolidate("P001", evidence)[0].event
        self.assertEqual(cached.summary_short, "Prima sintesi")
        self.assertEqual(len(same_llm.calls), 0)

        changed_llm = _StructuredCaptureLlm(response("Sintesi rigenerata"))
        changed_llm.temperature = 0.2
        regenerated = ClinicalConsolidator(
            fusion_llm=changed_llm, existing_events=[first]
        ).consolidate("P001", evidence)[0].event
        self.assertEqual(regenerated.summary_short, "Sintesi rigenerata")
        self.assertEqual(len(changed_llm.calls), 1)

    def test_fusion_keeps_conflicting_sources_visible_in_neutral_claim(self):
        llm = _StructuredCaptureLlm({
            "status": "unknown", "certainty": "unknown",
            "severity": None, "significance": "clinically_relevant",
            "claims": [{
                "text": (
                    "Alla stessa data una fonte documenta dispnea, mentre "
                    "un'altra ne riferisce l'assenza."
                ),
                "evidence_ids": ["E_PRESENT", "E_ABSENT"],
                "certainty": "unknown",
            }],
            "conflicting_evidence_ids": ["E_PRESENT", "E_ABSENT"],
        })
        evidence = [
            ClinicalEvidence(
                evidence_id="E_PRESENT", patient_id="P001", document_id="D1",
                category="symptom", normalized_entity="dispnea",
                source_text="Presente dispnea", assertion="present",
                observed_date="2025-01-10", document_date="2025-01-10",
            ),
            ClinicalEvidence(
                evidence_id="E_ABSENT", patient_id="P001", document_id="D2",
                category="symptom", normalized_entity="dispnea",
                source_text="Nega dispnea", assertion="absent",
                observed_date="2025-01-10", document_date="2025-01-10",
            ),
        ]
        bundle = ClinicalConsolidator(fusion_llm=llm).consolidate(
            "P001", evidence
        )[0]
        self.assertTrue(bundle.event.structured_data["fusion_validated"])
        self.assertEqual(bundle.event.review_status, "pending")
        self.assertEqual(
            {link.relation for link in bundle.links}, {"contradicts"}
        )
        self.assertTrue(all(link.included_in_summary for link in bundle.links))

    def test_fusion_rejects_technical_evidence_ids_inside_clinical_prose(self):
        llm = _StructuredCaptureLlm({
            "status": "active", "certainty": "confirmed",
            "severity": None, "significance": "clinically_relevant",
            "claims": [{
                "text": "E1 ed E2 documentano dispnea persistente.",
                "evidence_ids": ["E1", "E2"], "certainty": "confirmed",
            }],
            "conflicting_evidence_ids": [],
        })
        evidence = [
            ClinicalEvidence(
                evidence_id=evidence_id, patient_id="P001",
                document_id=f"D{index}", category="symptom",
                normalized_entity="dispnea", source_text=source,
                document_date=f"2025-01-1{index}",
            )
            for index, (evidence_id, source) in enumerate(
                (("E1", "Dispnea"), ("E2", "Dispnea persistente")), start=1
            )
        ]
        event = ClinicalConsolidator(fusion_llm=llm).consolidate(
            "P001", evidence
        )[0].event
        self.assertFalse(event.structured_data["fusion_validated"])
        self.assertEqual(event.review_status, "pending")
        self.assertNotIn("E1", event.summary_short)
        self.assertNotIn("E2", event.summary_short)

    def test_retrodated_acute_episode_does_not_shift_reviewed_event_id(self):
        reviewed = ClinicalEvent(
            event_id="EVT_REVIEWED_MARCH", patient_id="P001",
            episode_id="EPI_REVIEWED_MARCH", category="symptom",
            canonical_entity="dispnea", summary_short="Dispnea di marzo",
            first_evidence_date="2025-03-01", review_status="accepted",
            structured_data={"evidence_ids": ["E_MARCH"]},
        )
        evidence = [
            ClinicalEvidence(
                evidence_id="E_JAN", patient_id="P001", document_id="D1",
                category="symptom", normalized_entity="dispnea",
                source_text="Dispnea a gennaio", observed_date="2025-01-01",
                document_date="2025-01-10",
            ),
            ClinicalEvidence(
                evidence_id="E_MARCH", patient_id="P001", document_id="D2",
                category="symptom", normalized_entity="dispnea",
                source_text="Dispnea a marzo", observed_date="2025-03-01",
                document_date="2025-01-12",
            ),
        ]
        bundles = ClinicalConsolidator(
            existing_events=[reviewed]
        ).consolidate("P001", evidence)
        january, march = bundles
        self.assertNotEqual(january.event.event_id, reviewed.event_id)
        self.assertEqual(march.event.event_id, reviewed.event_id)
        self.assertEqual(march.episode.episode_id, reviewed.episode_id)
        self.assertEqual(
            march.episode.previous_episode_id, january.episode.episode_id
        )

    def test_review_correction_survives_automatic_rebuild(self):
        event = ClinicalEvent(
            event_id="EVT_TEST", patient_id="P001", category="diagnosis",
            canonical_entity="ipotiroidismo",
            summary_short="Sintesi automatica", first_evidence_date="2025-01-10",
        )
        repo = ClinicalRegistryRepository(self.db)
        repo.save_event(event)
        ReviewDecisionRepository(self.db).decide_event(
            "P001", "EVT_TEST", "corrected",
            corrected_value={"summary_short": "Ipotiroidismo confermato"},
            reason="Conferma specialistica",
        )
        repo.save_event(ClinicalEvent(
            event_id="EVT_TEST", patient_id="P001", category="diagnosis",
            canonical_entity="ipotiroidismo",
            summary_short="Nuova sintesi automatica",
        ))
        persisted = repo.get_event_detail("EVT_TEST")
        self.assertEqual(
            persisted["event"]["summary_short"], "Ipotiroidismo confermato"
        )
        self.assertEqual(persisted["event"]["review_status"], "corrected")
        self.assertEqual(len(persisted["reviews"]), 1)

    def test_reviewed_event_relation_survives_automatic_rebuild(self):
        repo = ClinicalRegistryRepository(self.db)
        for event_id in ("EVT_SOURCE", "EVT_TARGET"):
            repo.save_event(ClinicalEvent(
                event_id=event_id, patient_id="P001", category="diagnosis",
                canonical_entity=event_id.casefold(),
                summary_short=event_id,
            ))
        original = ClinicalEventRelation(
            relation_id="REL_REVIEWED", patient_id="P001",
            source_event_id="EVT_SOURCE", target_event_id="EVT_TARGET",
            relation_type="related_episode", confidence=0.95,
            rationale="Collegamento confermato", review_status="accepted",
        )
        repo.replace_generated_relations("P001", [original])
        replacement = ClinicalEventRelation(
            relation_id="REL_AUTO", patient_id="P001",
            source_event_id="EVT_SOURCE", target_event_id="EVT_TARGET",
            relation_type="related_episode", confidence=0.4,
            rationale="Nuova ipotesi automatica", review_status="auto",
        )
        repo.replace_generated_relations("P001", [replacement])
        relation = repo.get_event_detail("EVT_SOURCE")["relations"][0]
        self.assertEqual(relation["relation_id"], "REL_REVIEWED")
        self.assertEqual(relation["review_status"], "accepted")
        self.assertEqual(relation["rationale"], "Collegamento confermato")

    def test_changed_registry_invalidates_only_stale_narrative_profile(self):
        state_repo = ClinicalStateRepository(self.db)
        state_repo.save(ClinicalState(
            patient_id="P001", clinical_profile="Vecchio profilo",
            open_problems=["problema preservato"],
        ))
        builder = object.__new__(ClinicalRegistryBuilder)
        builder.db = self.db
        self.assertTrue(builder._invalidate_clinical_profile("P001"))
        updated = state_repo.load("P001")
        self.assertEqual(updated.clinical_profile, "")
        self.assertEqual(updated.open_problems, ["problema preservato"])

    def test_legacy_timeline_sync_excludes_administrative_only_event(self):
        evidence = ClinicalEvidence(
            evidence_id="E_APPOINTMENT", patient_id="P001",
            document_id="D1", category="follow_up",
            normalized_entity="PET FDG",
            source_text="Prossimi appuntamenti: PET il 26/09 ore 09:00",
        )
        EvidenceRepository(self.db).insert_batch([evidence])
        event = ClinicalEvent(
            event_id="EVT_APPOINTMENT", patient_id="P001",
            category="follow_up", canonical_entity="pet_fdg",
            summary_short="Prossimo appuntamento PET",
        )
        registry_repo = ClinicalRegistryRepository(self.db)
        registry_repo.save_event(event, [EventEvidenceLink(
            link_id="LNK_APPOINTMENT", event_id=event.event_id,
            evidence_id=evidence.evidence_id,
        )])
        builder = object.__new__(ClinicalRegistryBuilder)
        builder.registry_repo = registry_repo
        builder.timeline_repo = TimelineRepository(self.db)
        builder._sync_legacy_timeline("P001", [], [])
        self.assertEqual(builder.timeline_repo.count_by_patient("P001"), 0)

    def test_cross_document_thyroid_and_respiratory_correlations_are_cautious(self):
        thyroid = [
            ClinicalEvidence(
                evidence_id="ET1", patient_id="P001", document_id="D1",
                category="laboratory_finding", normalized_entity="TSH",
                source_text="TSH soppresso 0,01 mUI/L",
                numeric_value=0.01, unit="mUI/L", observed_date="2025-01-10",
                document_date="2025-01-10",
            ),
            ClinicalEvidence(
                evidence_id="ET2", patient_id="P001", document_id="D2",
                category="diagnosis", normalized_entity="ipotiroidismo",
                source_text="Insorgenza di ipotiroidismo",
                observed_date="2025-01-12", document_date="2025-01-12",
            ),
        ]
        respiratory = [
            ClinicalEvidence(
                evidence_id="ER1", patient_id="P001", document_id="D1",
                category="symptom", normalized_entity="dispnea_tosse",
                source_text="Tosse e dispnea", observed_date="2025-02-01",
                document_date="2025-02-01",
            ),
            ClinicalEvidence(
                evidence_id="ER2", patient_id="P001", document_id="D1",
                category="vital_sign", normalized_entity="SpO2",
                source_text="SpO2 88%", numeric_value=88, unit="%",
                observed_date="2025-02-01", document_date="2025-02-01",
            ),
            ClinicalEvidence(
                evidence_id="ER3", patient_id="P001", document_id="D2",
                category="imaging_finding", normalized_entity="addensamento_polmonare",
                source_text="TAC: addensamento polmonare basale",
                observed_date="2025-02-03", document_date="2025-02-03",
            ),
        ]
        bundles = ClinicalCorrelationBuilder().build(
            "P001", thyroid + respiratory
        )
        by_system = {
            bundle.event.structured_data["correlation_system"]: bundle
            for bundle in bundles
        }
        self.assertEqual(set(by_system), {"tiroideo", "respiratorio"})
        for bundle in by_system.values():
            self.assertEqual(bundle.event.certainty, "inferred")
            self.assertEqual(bundle.event.review_status, "pending")
            self.assertFalse(bundle.event.structured_data["causality_asserted"])
            self.assertIn("non è dimostrata", bundle.event.summary_detail)
        self.assertEqual(len(by_system["respiratorio"].links), 3)
        self.assertNotIn(
            "88% 88 %", by_system["respiratorio"].event.summary_detail
        )

    def test_full_provenance_export_and_fts(self):
        evidence = ClinicalEvidence(
            evidence_id="E_EXPORT", patient_id="P001", document_id="D1",
            category="imaging_finding", normalized_entity="nodulo_polmonare",
            source_text="Nodulo polmonare di 8 mm", source_page=3,
            observed_date="2025-01-10", document_date="2025-01-10",
        )
        EvidenceRepository(self.db).insert_batch([evidence])
        bundle = ClinicalConsolidator().consolidate("P001", [evidence])[0]
        repo = ClinicalRegistryRepository(self.db)
        repo.save_episode(bundle.episode)
        repo.save_event(bundle.event, bundle.links, bundle.updates)
        self.assertEqual(repo.search_events("P001", "nodulo polmonare")[0].event_id,
                         bundle.event.event_id)

        destination = Path(self.tmp.name) / "registry.json"
        ClinicalRegistryExporter({"registry_repo": repo}).export(
            "P001", destination, include_sources=True
        )
        exported = json.loads(destination.read_text(encoding="utf-8"))
        source = exported["events"][0]["evidence"][0]
        self.assertEqual(source["source_page"], 3)
        self.assertEqual(source["source_text"], "Nodulo polmonare di 8 mm")
        self.assertNotIn("filename", source)

    def test_csv_export_keeps_all_selected_record_types(self):
        destination = Path(self.tmp.name) / "registry.csv"
        data = {
            "events": [{
                "event": {"event_id": "EV1", "summary_short": "Dispnea"},
                "episode": {"episode_id": "EP1"},
                "evidence": [{"evidence_id": "E1", "source_text": "Dispnea"}],
                "updates": [], "relations": [], "reviews": [],
            }],
            "lab_values": [{"normalized_name": "tsh", "value": 0.01}],
            "medication_courses": [{"course_id": "M1"}],
            "oncology_lines": [{"line_id": "O1"}],
            "lab_trends": [{"trend_id": "T1"}],
            "clinical_profile": "Profilo sintetico",
        }
        ClinicalRegistryExporter({})._write_csv(destination, data)
        with destination.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(
            {row["record_type"] for row in rows},
            {
                "clinical_event", "clinical_episode", "event_evidence",
                "lab_value", "medication_course", "oncology_line",
                "lab_trend", "clinical_profile",
            },
        )
        evidence = next(row for row in rows if row["record_type"] == "event_evidence")
        self.assertEqual(evidence["parent_event_id"], "EV1")

    def test_audit_redacts_identity_and_is_hash_chained(self):
        repo = AuditRepository(self.db)
        repo.log("P001", "route", details={
            "llm_identity": {
                "name": "Mario Rossi", "fiscal_code": "RSSMRA80A01H501U",
                "confidence": 0.92,
            }
        })
        payload = repo.get_by_patient("P001")[0]["details"]
        self.assertTrue(payload["llm_identity"]["redacted"])
        self.assertNotIn("Mario", json.dumps(payload))
        self.assertEqual(repo.verify_chain("P001"), (True, []))


if __name__ == "__main__":
    unittest.main()
