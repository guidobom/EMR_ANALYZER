"""Persistence for one consolidated clinical projection per document."""

from __future__ import annotations

from datetime import datetime
import json

from .engine import DatabaseEngine
from ..models.document_projection import DocumentClinicalProjection


class DocumentProjectionRepository:
    def __init__(self, db: DatabaseEngine):
        self.db = db

    def replace(self, projection: DocumentClinicalProjection) -> None:
        projection.updated_at = datetime.now().isoformat()
        self.db.execute(
            """INSERT INTO document_clinical_projections
               (projection_id, patient_id, document_id, projection_json,
                evidence_count, observation_count, consolidation_method,
                model_name, prompt_version, schema_version, status,
                created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(document_id) DO UPDATE SET
                 projection_id=excluded.projection_id,
                 patient_id=excluded.patient_id,
                 projection_json=excluded.projection_json,
                 evidence_count=excluded.evidence_count,
                 observation_count=excluded.observation_count,
                 consolidation_method=excluded.consolidation_method,
                 model_name=excluded.model_name,
                 prompt_version=excluded.prompt_version,
                 schema_version=excluded.schema_version,
                 status=excluded.status,
                 updated_at=excluded.updated_at""",
            (
                projection.projection_id, projection.patient_id,
                projection.document_id,
                json.dumps(projection.to_dict(), ensure_ascii=False),
                len(projection.evidence_ids), len(projection.observations),
                projection.consolidation_method, projection.model_name,
                projection.prompt_version, projection.schema_version,
                projection.status, projection.created_at, projection.updated_at,
            ),
        )
        self.db.commit()

    def get_by_document(
        self, document_id: str
    ) -> DocumentClinicalProjection | None:
        row = self.db.execute(
            """SELECT projection_json FROM document_clinical_projections
               WHERE document_id=?""",
            (document_id,),
        ).fetchone()
        if not row:
            return None
        return DocumentClinicalProjection.from_dict(
            json.loads(row["projection_json"])
        )

    def get_by_patient(self, patient_id: str) -> list[DocumentClinicalProjection]:
        rows = self.db.execute(
            """SELECT projection_json FROM document_clinical_projections
               WHERE patient_id=? ORDER BY updated_at, document_id""",
            (patient_id,),
        ).fetchall()
        return [
            DocumentClinicalProjection.from_dict(json.loads(row["projection_json"]))
            for row in rows
        ]

    def delete_by_document(self, document_id: str) -> None:
        self.db.execute(
            "DELETE FROM document_clinical_projections WHERE document_id=?",
            (document_id,),
        )
        self.db.commit()
