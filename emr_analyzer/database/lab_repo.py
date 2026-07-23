"""Lab values repository — CRUD operations for lab_values table."""

import json
from typing import Optional

from .engine import DatabaseEngine
from ..models.lab_result import LabValue


class LabRepository:
    """Data access for laboratory values."""

    def __init__(self, db: DatabaseEngine):
        self.db = db

    def insert(self, lab: LabValue) -> int:
        cursor = self.db.execute(
            """INSERT INTO lab_values (patient_id, document_id,
               parameter_name, normalized_name, value, operator, unit,
               reference_low, reference_high, reference_text, is_abnormal,
               flag, sample_date, biological_material, lab_name, page,
               source_text, confidence, validated_by_user)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (lab.patient_id, lab.document_id, lab.parameter_name,
             lab.normalized_name, lab.value, lab.operator, lab.unit,
             lab.reference_low, lab.reference_high, lab.reference_text,
             1 if lab.is_abnormal else 0, lab.flag, lab.sample_date,
             lab.biological_material, lab.lab_name, lab.page,
             lab.source_text, lab.confidence, 1 if lab.validated_by_user else 0),
        )
        self.db.commit()
        return cursor.lastrowid

    def insert_batch(self, lab_values: list[LabValue]) -> None:
        params = [
            (lv.patient_id, lv.document_id, lv.parameter_name,
             lv.normalized_name, lv.value, lv.operator, lv.unit,
             lv.reference_low, lv.reference_high, lv.reference_text,
             1 if lv.is_abnormal else 0, lv.flag, lv.sample_date,
             lv.biological_material, lv.lab_name, lv.page,
             lv.source_text, lv.confidence,
             1 if lv.validated_by_user else 0)
            for lv in lab_values
        ]
        self.db.executemany(
            """INSERT INTO lab_values (patient_id, document_id,
               parameter_name, normalized_name, value, operator, unit,
               reference_low, reference_high, reference_text, is_abnormal,
               flag, sample_date, biological_material, lab_name, page,
               source_text, confidence, validated_by_user)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            params,
        )
        self.db.commit()

    def get_by_patient(self, patient_id: str) -> list[LabValue]:
        cursor = self.db.execute(
            """SELECT * FROM lab_values WHERE patient_id=?
               ORDER BY sample_date ASC, normalized_name ASC""",
            (patient_id,),
        )
        return [self._row_to_lab(r) for r in cursor.fetchall()]

    def get_by_document(self, document_id: str) -> list[LabValue]:
        cursor = self.db.execute(
            "SELECT * FROM lab_values WHERE document_id=? ORDER BY id",
            (document_id,),
        )
        return [self._row_to_lab(r) for r in cursor.fetchall()]

    def get_by_parameter(self, patient_id: str,
                         normalized_name: str) -> list[LabValue]:
        cursor = self.db.execute(
            """SELECT * FROM lab_values
               WHERE patient_id=? AND normalized_name=?
               ORDER BY sample_date ASC""",
            (patient_id, normalized_name),
        )
        return [self._row_to_lab(r) for r in cursor.fetchall()]

    def get_distinct_parameters(self, patient_id: str) -> list[str]:
        cursor = self.db.execute(
            """SELECT DISTINCT normalized_name FROM lab_values
               WHERE patient_id=? ORDER BY normalized_name""",
            (patient_id,),
        )
        return [r["normalized_name"] for r in cursor.fetchall()]

    def get_abnormal_count(self, patient_id: str) -> int:
        cursor = self.db.execute(
            "SELECT COUNT(*) FROM lab_values WHERE patient_id=? AND is_abnormal=1",
            (patient_id,),
        )
        return cursor.fetchone()[0]

    def update_validation(self, lab_id: int, validated: bool) -> None:
        self.db.execute(
            "UPDATE lab_values SET validated_by_user=? WHERE id=?",
            (1 if validated else 0, lab_id),
        )
        self.db.commit()

    def delete_by_document(self, document_id: str) -> None:
        self.db.execute(
            "DELETE FROM lab_values WHERE document_id=?", (document_id,)
        )
        self.db.commit()

    def _row_to_lab(self, row) -> LabValue:
        return LabValue(
            patient_id=row["patient_id"],
            document_id=row["document_id"],
            parameter_name=row["parameter_name"],
            normalized_name=row["normalized_name"],
            value=row["value"],
            operator=row["operator"],
            unit=row["unit"],
            reference_low=row["reference_low"],
            reference_high=row["reference_high"],
            reference_text=row["reference_text"] or "",
            is_abnormal=bool(row["is_abnormal"]),
            flag=row["flag"],
            sample_date=row["sample_date"],
            biological_material=row["biological_material"],
            lab_name=row["lab_name"],
            page=row["page"],
            source_text=row["source_text"] or "",
            confidence=row["confidence"] or 1.0,
            validated_by_user=bool(row["validated_by_user"]),
        )
