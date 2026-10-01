"""Synthetic project builders shared by the tests (invented data only)."""

from __future__ import annotations

from emr_analyzer.clinical.evidence_utils import content_hash
from emr_analyzer.clinical.grounded_sources import METHOD
from emr_analyzer.database.document_repo import DocumentRepository
from emr_analyzer.database.engine import DatabaseEngine
from emr_analyzer.database.evidence_repo import EvidenceRepository
from emr_analyzer.database.lab_repo import LabRepository
from emr_analyzer.database.migrations import init_database
from emr_analyzer.database.overlay_repo import DocumentTextOverlayRepository
from emr_analyzer.database.patient_repo import PatientRepository
from emr_analyzer.database.processing_repo import ProcessingRepository
from emr_analyzer.models import Patient
from emr_analyzer.models.clinical_evidence import ClinicalEvidence
from emr_analyzer.models.document import DocumentRecord


class Project:
    """A temporary project database with patients and normalized texts."""

    def __init__(self, root):
        self.root = root
        self.db = DatabaseEngine(root / "emr_registry.db")
        init_database(self.db)
        self.patients = PatientRepository(self.db)
        self.documents = DocumentRepository(self.db)
        self.labs = LabRepository(self.db)
        self.evidence = EvidenceRepository(self.db)
        self.processing = ProcessingRepository(self.db)
        self.overlays = DocumentTextOverlayRepository(self.db)

    def add_patient(self, patient_id: str) -> None:
        self.patients.insert(Patient(id=patient_id, pseudonym="SINTETICO"))
        (self.root / patient_id / "extraction").mkdir(parents=True, exist_ok=True)

    def add_document(self, patient_id: str, document_id: str, text: str, *,
                     date: str = "2026-09-01", document_type: str = "visita_oncologica"):
        original = self.root / patient_id / f"{document_id}.txt"
        original.write_text(text, encoding="utf-8")
        self.documents.insert(DocumentRecord(
            id=document_id, patient_id=patient_id, filename=original.name,
            original_path=str(original), file_hash=document_id,
            document_date=date, document_type=document_type))
        (self.root / patient_id / "extraction" / f"{document_id}.md").write_text(text, encoding="utf-8")

    def pipeline(self, llm=None, extractor=None):
        from emr_analyzer.clinical.extraction_pipeline import ExtractionPipeline

        pipeline = ExtractionPipeline(
            evidence_repo=self.evidence, processing_repo=self.processing,
            document_repo=self.documents, lab_repo=self.labs,
            overlay_repo=self.overlays, llm_client=llm, db=self.db)
        if extractor is not None:
            pipeline.make_extractor = lambda: extractor
            pipeline.extractor = extractor
        return pipeline

    def close(self):
        self.db.close()


class FakeLlm:
    model = "fake-model"
    is_available = True
    parallel_workers = 1


class KeywordExtractor:
    """Deterministic stand-in for EventExtractor: one event per keyword."""

    prompt_version = "test-extractor-v1"
    prompt_digest = "test-digest"
    model_name = "fake-model"

    def __init__(self, keywords=("tosse", "febbre", "nivolumab")):
        self.keywords = keywords
        self.calls = []

    @property
    def model_digest(self):
        return content_hash("fake", self.prompt_version)

    def last_extraction_metrics(self):
        return {"llm_calls": 1}

    def guidance_digest(self, text):
        return ""

    def extract_document(self, *, patient_id, document_id, document_type, document_date,
                         text, evidence_ready_callback=None, cancel_check=None,
                         statement_roles=None, **_):
        from emr_analyzer.clinical.sentence_groups import clinical_sentences
        import re as _re

        self.calls.append(document_id)
        spans = clinical_sentences(text)
        rows = []
        for keyword in self.keywords:
            start = text.find(keyword)
            if start < 0:
                continue
            sentence = next((span for span in spans if span.start <= start < span.end), None)
            if sentence is None:
                continue
            if statement_roles and statement_roles.get(sentence.sentence_id) == "copy":
                continue
            # Like the real reader: the fragment is the whole-word interval.
            tokens = list(_re.finditer(r"\S+", sentence.text))
            keyword_end = start + len(keyword)
            first_index = next(i for i, match in enumerate(tokens)
                               if sentence.start + match.end() > start)
            last_index = max(i for i, match in enumerate(tokens)
                             if sentence.start + match.start() < keyword_end)
            first, last = first_index + 1, last_index + 1
            start = sentence.start + tokens[first_index].start()
            end = sentence.start + tokens[last_index].end()
            rows.append(ClinicalEvidence(
                patient_id=patient_id, document_id=document_id,
                evidence_id="EVD_" + content_hash(patient_id, document_id, start, end, keyword),
                category="symptom", fact_type="medication" if keyword == "nivolumab" else "symptom",
                normalized_entity=keyword, canonical_label=keyword, concept_original=keyword,
                source_text=text[start:end], assertion="present", certainty="confirmed",
                document_date=document_date, extraction_method=METHOD, status="proposed",
                data={"fhir_pipeline": True, "experiencer": "patient", "attributes": {},
                      "event_kind": "continuous", "lexicon_term_id": "test_" + keyword,
                      "source_spans": [{"start": start, "end": end, "text": text[start:end],
                                        "sentence_id": sentence.sentence_id}],
                      "source_sentence": {"start": sentence.start, "end": sentence.end,
                                          "text": sentence.text},
                      "source_selection": {"sentence": sentence.sentence_id,
                                           "first_word": first, "last_word": last},
                      "source_version": content_hash(text), "date_provenance": {"mode": "unknown"}}))
        if evidence_ready_callback:
            evidence_ready_callback(rows)
        return rows
