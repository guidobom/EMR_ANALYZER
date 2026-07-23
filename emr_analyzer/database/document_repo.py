"""Document repository — CRUD operations for documents table."""

import json
from typing import Optional

from .engine import DatabaseEngine
from ..models.document import DocumentRecord


class DocumentRepository:
    """Data access for documents."""

    def __init__(self, db: DatabaseEngine):
        self.db = db

    def insert(self, doc: DocumentRecord) -> None:
        self.db.execute(
            """INSERT INTO documents (id, patient_id, filename, original_path,
               file_hash, page_count, document_date, document_type, import_date,
               parsing_status, extraction_status, validation_status,
               event_count, lab_value_count, error_message, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (doc.id, doc.patient_id, doc.filename, doc.original_path,
             doc.file_hash, doc.page_count, doc.document_date, doc.document_type,
             doc.import_date, doc.parsing_status, doc.extraction_status,
             doc.validation_status, doc.event_count, doc.lab_value_count,
             doc.error_message, doc.metadata_json),
        )
        self.db.commit()

    def update(self, doc: DocumentRecord) -> None:
        self.db.execute(
            """UPDATE documents SET document_date=?, document_type=?,
               parsing_status=?, extraction_status=?, validation_status=?,
               event_count=?, lab_value_count=?, error_message=?, metadata_json=?
               WHERE id=?""",
            (doc.document_date, doc.document_type, doc.parsing_status,
             doc.extraction_status, doc.validation_status,
             doc.event_count, doc.lab_value_count,
             doc.error_message, doc.metadata_json, doc.id),
        )
        self.db.commit()

    def get_by_id(self, doc_id: str) -> Optional[DocumentRecord]:
        cursor = self.db.execute(
            "SELECT * FROM documents WHERE id=?", (doc_id,)
        )
        row = cursor.fetchone()
        if row:
            return self._row_to_doc(row)
        return None

    def get_by_hash(self, patient_id: str, file_hash: str) -> Optional[DocumentRecord]:
        cursor = self.db.execute(
            "SELECT * FROM documents WHERE patient_id=? AND file_hash=?",
            (patient_id, file_hash),
        )
        row = cursor.fetchone()
        if row:
            return self._row_to_doc(row)
        return None

    def get_by_hash_global(self, file_hash: str) -> Optional[DocumentRecord]:
        """Find a duplicate independently of the currently selected patient."""
        cursor = self.db.execute(
            "SELECT * FROM documents WHERE file_hash=? ORDER BY import_date LIMIT 1",
            (file_hash,),
        )
        row = cursor.fetchone()
        return self._row_to_doc(row) if row else None

    def list_by_patient(self, patient_id: str) -> list[DocumentRecord]:
        cursor = self.db.execute(
            """SELECT * FROM documents WHERE patient_id=?
               ORDER BY document_date DESC, import_date DESC""",
            (patient_id,),
        )
        return [self._row_to_doc(r) for r in cursor.fetchall()]

    def list_by_type(self, patient_id: str, doc_type: str) -> list[DocumentRecord]:
        cursor = self.db.execute(
            "SELECT * FROM documents WHERE patient_id=? AND document_type=?",
            (patient_id, doc_type),
        )
        return [self._row_to_doc(r) for r in cursor.fetchall()]

    def update_parsing_status(self, doc_id: str, status: str,
                              error: Optional[str] = None) -> None:
        self.db.execute(
            "UPDATE documents SET parsing_status=?, error_message=? WHERE id=?",
            (status, error, doc_id),
        )
        self.db.commit()

    def update_extraction_status(self, doc_id: str, status: str) -> None:
        self.db.execute(
            "UPDATE documents SET extraction_status=? WHERE id=?",
            (status, doc_id),
        )
        self.db.commit()

    def update_counts(self, doc_id: str, event_count: int, lab_count: int) -> None:
        self.db.execute(
            "UPDATE documents SET event_count=?, lab_value_count=? WHERE id=?",
            (event_count, lab_count, doc_id),
        )
        self.db.commit()

    def delete(self, doc_id: str) -> None:
        self.db.execute("DELETE FROM documents WHERE id=?", (doc_id,))
        self.db.commit()

    def get_next_id(self) -> str:
        """Generate the next document ID (DOC_000001, DOC_000002, ...)."""
        cursor = self.db.execute(
            "SELECT id FROM documents ORDER BY id DESC LIMIT 1"
        )
        row = cursor.fetchone()
        if row:
            last_id = row["id"]
            last_num = int(last_id.split("_")[1])
            return f"DOC_{last_num + 1:06d}"
        return "DOC_000001"

    def get_count_by_patient(self, patient_id: str) -> int:
        cursor = self.db.execute(
            "SELECT COUNT(*) FROM documents WHERE patient_id=?", (patient_id,)
        )
        return cursor.fetchone()[0]

    def _row_to_doc(self, row) -> DocumentRecord:
        return DocumentRecord(
            id=row["id"],
            patient_id=row["patient_id"],
            filename=row["filename"],
            original_path=row["original_path"],
            file_hash=row["file_hash"],
            page_count=row["page_count"] or 0,
            document_date=row["document_date"],
            document_type=row["document_type"],
            import_date=row["import_date"],
            parsing_status=row["parsing_status"],
            extraction_status=row["extraction_status"],
            validation_status=row["validation_status"],
            event_count=row["event_count"] or 0,
            lab_value_count=row["lab_value_count"] or 0,
            error_message=row["error_message"],
            metadata_json=row["metadata_json"],
        )
