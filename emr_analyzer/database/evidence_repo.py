"""Persistence for immutable document-level clinical evidence."""

from __future__ import annotations

import json

from .engine import DatabaseEngine
from ..models.clinical_evidence import ClinicalEvidence


class EvidenceRepository:
    def __init__(self, db: DatabaseEngine):
        self.db = db

    def insert_batch(self, evidence: list[ClinicalEvidence]) -> None:
        if not evidence:
            return
        self.db.executemany(
            """INSERT INTO clinical_evidence
               (evidence_id, patient_id, document_id, category,
                normalized_entity, assertion, temporality, clinical_status,
                observed_date, observed_date_end, document_date,
                date_precision, date_source, anatomical_site, laterality,
                severity, significance, certainty, value_text, numeric_value,
                unit, source_page, source_text, bbox_json, confidence,
                extraction_method, model_name, prompt_version, schema_version,
                status, data_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?,
                       ?, ?, ?, ?, ?, ?, ?, ?,
                       ?, ?, ?, ?, ?, ?, ?, ?,
                       ?, ?, ?, ?, ?, ?, ?, ?)""",
            [self._params(item) for item in evidence],
        )
        self.db.commit()

    def replace_document(self, document_id: str,
                         evidence: list[ClinicalEvidence]) -> None:
        with self.db:
            self.db.execute(
                "DELETE FROM clinical_evidence WHERE document_id=?", (document_id,)
            )
            if evidence:
                self.db.executemany(
                    """INSERT INTO clinical_evidence
                       (evidence_id, patient_id, document_id, category,
                        normalized_entity, assertion, temporality, clinical_status,
                        observed_date, observed_date_end, document_date,
                        date_precision, date_source, anatomical_site, laterality,
                        severity, significance, certainty, value_text,
                        numeric_value, unit, source_page, source_text, bbox_json,
                        confidence, extraction_method, model_name,
                        prompt_version, schema_version, status, data_json,
                        created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?,
                               ?, ?, ?, ?, ?, ?, ?, ?,
                               ?, ?, ?, ?, ?, ?, ?, ?,
                               ?, ?, ?, ?, ?, ?, ?, ?)""",
                    [self._params(item) for item in evidence],
                )

    def replace_document_method(self, document_id: str, method: str,
                                evidence: list[ClinicalEvidence]) -> None:
        with self.db:
            self.db.execute(
                """DELETE FROM clinical_evidence
                   WHERE document_id=? AND extraction_method=?""",
                (document_id, method),
            )
            if evidence:
                self.db.executemany(
                    """INSERT INTO clinical_evidence
                       (evidence_id, patient_id, document_id, category,
                        normalized_entity, assertion, temporality, clinical_status,
                        observed_date, observed_date_end, document_date,
                        date_precision, date_source, anatomical_site, laterality,
                        severity, significance, certainty, value_text,
                        numeric_value, unit, source_page, source_text, bbox_json,
                        confidence, extraction_method, model_name,
                        prompt_version, schema_version, status, data_json,
                        created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?,
                               ?, ?, ?, ?, ?, ?, ?, ?,
                               ?, ?, ?, ?, ?, ?, ?, ?,
                               ?, ?, ?, ?, ?, ?, ?, ?)""",
                    [self._params(item) for item in evidence],
                )

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

    @staticmethod
    def _params(item: ClinicalEvidence) -> tuple:
        return (
            item.evidence_id, item.patient_id, item.document_id, item.category,
            item.normalized_entity, item.assertion, item.temporality,
            item.clinical_status, item.observed_date, item.observed_date_end,
            item.document_date, item.date_precision,
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
