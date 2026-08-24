"""Persistence for immutable document-level clinical evidence."""

from __future__ import annotations

import json

from .engine import DatabaseEngine
from .pipeline_repo import ClinicalPipelineRepository
from ..clinical.terminology import DeterministicTerminologyResolver
from ..models.clinical_evidence import ClinicalEvidence


_EVIDENCE_COLUMNS = (
    "evidence_id", "patient_id", "document_id", "category",
    "normalized_entity", "fact_type", "concept_original",
    "canonical_label", "terminology_system", "terminology_code",
    "mapping_status", "mapping_confidence", "clinical_relevance",
    "typed_payload_json",
    "assertion", "temporality", "clinical_status", "observed_date",
    "observed_date_end", "document_date", "date_precision", "date_source",
    "anatomical_site", "laterality", "severity", "significance",
    "certainty", "value_text", "numeric_value", "unit", "source_page",
    "source_text", "bbox_json", "confidence", "extraction_method",
    "model_name", "prompt_version", "schema_version", "status",
    "data_json", "created_at",
)
_INSERT_EVIDENCE_SQL = (
    "INSERT INTO clinical_evidence (" + ",".join(_EVIDENCE_COLUMNS) + ") "
    "VALUES (" + ",".join("?" for _ in _EVIDENCE_COLUMNS) + ")"
)
_UPSERT_EVIDENCE_SQL = (
    _INSERT_EVIDENCE_SQL
    + " ON CONFLICT(evidence_id) DO UPDATE SET "
    + ",".join(
        f"{column}=excluded.{column}"
        for column in _EVIDENCE_COLUMNS
        if column not in {
            "evidence_id", "patient_id", "document_id", "created_at",
        }
    )
)


class EvidenceRepository:
    def __init__(self, db: DatabaseEngine):
        self.db = db
        self.pipeline = ClinicalPipelineRepository(db)
        self.terminology = DeterministicTerminologyResolver(self.pipeline)

    def insert_batch(self, evidence: list[ClinicalEvidence]) -> None:
        evidence = self.terminology.resolve_all(evidence)
        if not evidence:
            return
        self.db.executemany(
            _INSERT_EVIDENCE_SQL,
            [self._params(item) for item in evidence],
        )
        for item in evidence:
            self.pipeline.ensure_primary_source(item)
        self.db.commit()

    def replace_document(self, document_id: str,
                         evidence: list[ClinicalEvidence]) -> None:
        evidence = self.terminology.resolve_all(evidence)
        self._assert_document_scope(document_id, evidence)
        self._assert_identity_scope(document_id, evidence)
        with self.db:
            if evidence:
                self.db.executemany(
                    _UPSERT_EVIDENCE_SQL,
                    [self._params(item) for item in evidence],
                )
                for item in evidence:
                    self.pipeline.ensure_primary_source(item)
            self._delete_stale_document_ids(
                document_id, {item.evidence_id for item in evidence}
            )

    def replace_document_method(self, document_id: str, method: str,
                                evidence: list[ClinicalEvidence]) -> None:
        evidence = self.terminology.resolve_all(evidence)
        self._assert_document_scope(document_id, evidence)
        self._assert_identity_scope(document_id, evidence)
        if any(item.extraction_method != method for item in evidence):
            raise ValueError("Metodo di estrazione non coerente con il replace")
        with self.db:
            if evidence:
                self.db.executemany(
                    _UPSERT_EVIDENCE_SQL,
                    [self._params(item) for item in evidence],
                )
                for item in evidence:
                    self.pipeline.ensure_primary_source(item)
            current_ids = {
                row["evidence_id"] for row in self.db.execute(
                    """SELECT evidence_id FROM clinical_evidence
                       WHERE document_id=? AND extraction_method=?""",
                    (document_id, method),
                ).fetchall()
            }
            self._delete_ids(current_ids - {
                item.evidence_id for item in evidence
            })

    def get_by_document(self, document_id: str) -> list[ClinicalEvidence]:
        rows = self.db.execute(
            """SELECT * FROM clinical_evidence
               WHERE document_id=? ORDER BY source_page, id""",
            (document_id,),
        ).fetchall()
        return [self._row_to_evidence(row) for row in rows]

    def get_by_patient(self, patient_id: str) -> list[ClinicalEvidence]:
        rows = self.db.execute(
            """SELECT * FROM clinical_evidence
               WHERE patient_id=? ORDER BY observed_date, source_page, id""",
            (patient_id,),
        ).fetchall()
        return [self._row_to_evidence(row) for row in rows]

    def delete_by_document(self, document_id: str) -> None:
        self.db.execute(
            "DELETE FROM clinical_evidence WHERE document_id=?", (document_id,)
        )
        self.db.commit()

    def _delete_stale_document_ids(
        self, document_id: str, retained_ids: set[str]
    ) -> None:
        current = {
            row["evidence_id"] for row in self.db.execute(
                "SELECT evidence_id FROM clinical_evidence WHERE document_id=?",
                (document_id,),
            ).fetchall()
        }
        self._delete_ids(current - retained_ids)

    def _delete_ids(self, evidence_ids: set[str]) -> None:
        if not evidence_ids:
            return
        placeholders = ",".join("?" for _ in evidence_ids)
        self.db.execute(
            f"DELETE FROM clinical_evidence WHERE evidence_id IN ({placeholders})",
            tuple(sorted(evidence_ids)),
        )

    @staticmethod
    def _assert_document_scope(
        document_id: str, evidence: list[ClinicalEvidence]
    ) -> None:
        if any(item.document_id != document_id for item in evidence):
            raise ValueError("Evidenza associata a un documento diverso")

    def _assert_identity_scope(
        self, document_id: str, evidence: list[ClinicalEvidence]
    ) -> None:
        ids = {item.evidence_id for item in evidence}
        if not ids:
            return
        placeholders = ",".join("?" for _ in ids)
        collision = self.db.execute(
            f"""SELECT evidence_id FROM clinical_evidence
                WHERE evidence_id IN ({placeholders}) AND document_id<>?
                LIMIT 1""",
            (*sorted(ids), document_id),
        ).fetchone()
        if collision is not None:
            raise ValueError(
                "Identificativo evidenza già associato a un altro documento"
            )

    @staticmethod
    def _params(item: ClinicalEvidence) -> tuple:
        return (
            item.evidence_id, item.patient_id, item.document_id, item.category,
            item.normalized_entity, item.fact_type, item.concept_original,
            item.canonical_label, item.terminology_system,
            item.terminology_code, item.mapping_status,
            item.mapping_confidence,
            item.clinical_relevance,
            json.dumps(item.typed_payload, ensure_ascii=False),
            item.assertion, item.temporality, item.clinical_status,
            item.observed_date, item.observed_date_end, item.document_date,
            item.date_precision,
            item.date_source, item.anatomical_site, item.laterality,
            item.severity, item.significance, item.certainty, item.value_text,
            item.numeric_value, item.unit, item.source_page, item.source_text,
            json.dumps(item.bbox) if item.bbox else None, item.confidence,
            item.extraction_method, item.model_name, item.prompt_version,
            item.schema_version, item.status,
            json.dumps(item.data, ensure_ascii=False), item.created_at,
        )

    @staticmethod
    def _row_to_evidence(row) -> ClinicalEvidence:
        return ClinicalEvidence(
            evidence_id=row["evidence_id"],
            patient_id=row["patient_id"],
            document_id=row["document_id"],
            category=row["category"],
            normalized_entity=row["normalized_entity"],
            fact_type=row["fact_type"],
            concept_original=row["concept_original"],
            canonical_label=row["canonical_label"],
            terminology_system=row["terminology_system"],
            terminology_code=row["terminology_code"],
            mapping_status=row["mapping_status"] or "unmapped",
            mapping_confidence=row["mapping_confidence"],
            clinical_relevance=(
                row["clinical_relevance"] or "accepted_clinical"
            ),
            typed_payload=json.loads(row["typed_payload_json"] or "{}"),
            assertion=row["assertion"],
            temporality=row["temporality"],
            clinical_status=row["clinical_status"],
            observed_date=row["observed_date"],
            observed_date_end=row["observed_date_end"],
            document_date=row["document_date"],
            date_precision=row["date_precision"] or "unknown",
            date_source=row["date_source"],
            anatomical_site=row["anatomical_site"],
            laterality=row["laterality"],
            severity=row["severity"],
            significance=row["significance"] or "clinically_relevant",
            certainty=row["certainty"] or "confirmed",
            value_text=row["value_text"],
            numeric_value=row["numeric_value"],
            unit=row["unit"],
            source_page=row["source_page"],
            source_text=row["source_text"],
            bbox=tuple(json.loads(row["bbox_json"])) if row["bbox_json"] else None,
            confidence=row["confidence"] or 0.0,
            extraction_method=row["extraction_method"],
            model_name=row["model_name"],
            prompt_version=row["prompt_version"],
            schema_version=row["schema_version"],
            status=row["status"],
            data=json.loads(row["data_json"] or "{}"),
            created_at=row["created_at"],
        )
