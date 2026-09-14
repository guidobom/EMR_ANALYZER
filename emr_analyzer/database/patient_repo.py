"""Patient repository — CRUD operations for patients table."""

from typing import Optional

from .engine import DatabaseEngine
from ..models import Patient


class PatientRepository:
    """Data access for patients."""

    def __init__(self, db: DatabaseEngine):
        self.db = db

    def insert(self, patient: Patient) -> None:
        self.db.execute(
            """INSERT INTO patients (id, pseudonym, initials, sex, birth_year,
               main_pathology, center, notes, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (patient.id, patient.pseudonym, patient.initials,
             patient.sex, patient.birth_year,
             patient.main_pathology, patient.center, patient.notes,
             patient.created_at, patient.updated_at),
        )
        self.db.commit()

    def update(self, patient: Patient) -> None:
        self.db.execute(
            """UPDATE patients SET pseudonym=?, initials=?, sex=?, birth_year=?,
               main_pathology=?, center=?, notes=?, updated_at=?
               WHERE id=?""",
            (patient.pseudonym, patient.initials, patient.sex,
             patient.birth_year,
             patient.main_pathology, patient.center, patient.notes,
             patient.updated_at, patient.id),
        )
        self.db.commit()

    def get_by_id(self, patient_id: str) -> Optional[Patient]:
        cursor = self.db.execute(
            "SELECT * FROM patients WHERE id=?", (patient_id,)
        )
        row = cursor.fetchone()
        if row:
            return Patient(
                id=row["id"],
                pseudonym=row["pseudonym"],
                initials=row["initials"],
                sex=row["sex"],
                birth_year=row["birth_year"],
                main_pathology=row["main_pathology"],
                center=row["center"],
                notes=row["notes"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
        return None

    def list_all(self) -> list[Patient]:
        cursor = self.db.execute(
            "SELECT * FROM patients ORDER BY created_at DESC"
        )
        return [
            Patient(
                id=r["id"], pseudonym=r["pseudonym"],
                initials=r["initials"], sex=r["sex"],
                birth_year=r["birth_year"],
                main_pathology=r["main_pathology"],
                center=r["center"], notes=r["notes"],
                created_at=r["created_at"], updated_at=r["updated_at"],
            )
            for r in cursor.fetchall()
        ]

    def delete(self, patient_id: str) -> None:
        self.db.execute("DELETE FROM patients WHERE id=?", (patient_id,))
        self.db.commit()

    def get_next_id(self) -> str:
        """Generate the next patient ID (P001, P002, ...)."""
        cursor = self.db.execute(
            """SELECT id FROM patients
               WHERE id GLOB 'P[0-9]*' AND substr(id, 2) NOT GLOB '*[^0-9]*'
               ORDER BY CAST(substr(id, 2) AS INTEGER) DESC LIMIT 1"""
        )
        row = cursor.fetchone()
        if row:
            last_num = int(row["id"][1:]) if row["id"][1:].isdigit() else 0
            return f"P{last_num + 1:03d}"
        return "P001"

    def search(self, query: str) -> list[Patient]:
        """Search patients by pseudonym or notes."""
        q = f"%{query}%"
        cursor = self.db.execute(
            """SELECT * FROM patients
               WHERE pseudonym LIKE ? OR notes LIKE ? OR main_pathology LIKE ?
               ORDER BY created_at DESC""",
            (q, q, q),
        )
        return [
            Patient(
                id=r["id"], pseudonym=r["pseudonym"],
                initials=r["initials"], sex=r["sex"],
                birth_year=r["birth_year"],
                main_pathology=r["main_pathology"],
                center=r["center"], notes=r["notes"],
                created_at=r["created_at"], updated_at=r["updated_at"],
            )
            for r in cursor.fetchall()
        ]
