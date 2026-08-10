"""Import patients from another EMR Analyzer project."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

from ..config import active_workspace
from ..database.engine import DatabaseEngine
from ..database.migrations import init_database


class PatientImportService:
    """Copy patient data from a source project into the active project."""

    def __init__(self):
        self._id_maps: dict[str, dict[str, str]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_source_patients(self, source_path: str | Path) -> list[dict]:
        """Return a summary of every patient in the source project."""
        source = Path(source_path)
        db_path = source / "emr_registry.db"
        if not db_path.exists():
            raise FileNotFoundError(f"Progetto non trovato: {db_path}")

        db = DatabaseEngine(db_path)
        patients = []
        try:
            rows = db.execute(
                "SELECT * FROM patients ORDER BY created_at"
            ).fetchall()
            for row in rows:
                doc_count = db.execute(
                    "SELECT COUNT(*) FROM documents WHERE patient_id=?",
                    (row["id"],),
                ).fetchone()[0]
                timeline_count = db.execute(
                    "SELECT COUNT(*) FROM clinical_timeline WHERE patient_id=?",
                    (row["id"],),
                ).fetchone()[0]
                cs_row = db.execute(
                    "SELECT clinical_profile FROM clinical_state WHERE patient_id=?",
                    (row["id"],),
                ).fetchone()
                has_profile = bool(cs_row and cs_row["clinical_profile"])
                patients.append({
                    "id": row["id"],
                    "pseudonym": row["pseudonym"],
                    "initials": row["initials"],
                    "sex": row["sex"],
                    "birth_year": row["birth_year"],
                    "document_count": doc_count,
                    "timeline_entries": timeline_count,
                    "has_profile": has_profile,
                })
        finally:
            db.close()
        return patients

    def import_patients(
        self,
        source_path: str | Path,
        patient_ids: list[str],
        progress_callback=None,
    ) -> dict:
        """Import selected patients into the active project.

        Returns a summary dict with counts of imported items.
        """
        source = Path(source_path)
        target = active_workspace.path
        source_db_path = source / "emr_registry.db"
        target_db_path = target / "emr_registry.db"

        if not source_db_path.exists():
            raise FileNotFoundError(f"Progetto sorgente non trovato: {source_db_path}")

        source_db = DatabaseEngine(source_db_path)
        target_db = DatabaseEngine(target_db_path)
        init_database(target_db)

        self._load_target_ids(target_db)
        stats = {"patients": 0, "documents": 0, "timeline": 0, "profiles": 0}

        try:
            for i, patient_id in enumerate(patient_ids):
                if progress_callback:
                    progress_callback(
                        int((i / len(patient_ids)) * 100),
                        f"Importazione {patient_id}...",
                    )
                self._import_one(
                    source, source_db, target, target_db,
                    patient_id, stats,
                )
        finally:
            source_db.close()
            target_db.close()

        return stats

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _import_one(self, source, source_db, target, target_db,
                    patient_id, stats):
        """Import a single patient and all its data."""
        # 1. Create new patient record
        new_pid = self._next_id(target_db, "patients", "P")
        patient_row = source_db.execute(
            "SELECT * FROM patients WHERE id=?", (patient_id,)
        ).fetchone()
        if not patient_row:
            return

        now = datetime.now().isoformat()
        target_db.execute(
            """INSERT INTO patients (id, pseudonym, initials, sex, birth_year,
               main_pathology, center, notes, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (new_pid, patient_row["pseudonym"],
             patient_row["initials"], patient_row["sex"],
             patient_row["birth_year"], patient_row["main_pathology"],
             patient_row["center"], patient_row["notes"],
             now, now),
        )
        target_db.commit()
        self._id_maps.setdefault(patient_id, {})["patient"] = new_pid
        stats["patients"] += 1

        # 2. Copy document files and records
        doc_map: dict[str, str] = {}
        doc_rows = source_db.execute(
            "SELECT * FROM documents WHERE patient_id=? ORDER BY document_date",
            (patient_id,),
        ).fetchall()

        # Create target directories
        target_docs = target / new_pid / "documents" / "original"
        target_extraction = target / new_pid / "extraction"
        target_docs.mkdir(parents=True, exist_ok=True)
        target_extraction.mkdir(parents=True, exist_ok=True)

        for doc_row in doc_rows:
            new_doc_id = self._next_id(target_db, "documents", "DOC_")
            doc_map[doc_row["id"]] = new_doc_id

            target_db.execute(
                """INSERT INTO documents
                   (id, patient_id, filename, original_path, file_hash,
                    page_count, document_date, document_type, import_date,
                    parsing_status, extraction_status, validation_status,
                    event_count, lab_value_count, error_message, metadata_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (new_doc_id, new_pid, doc_row["filename"],
                 str(target_docs / Path(doc_row["original_path"]).name),
                 doc_row["file_hash"], doc_row["page_count"],
                 doc_row["document_date"], doc_row["document_type"], now,
                 doc_row["parsing_status"], doc_row["extraction_status"],
                 doc_row["validation_status"], doc_row["event_count"],
                 doc_row["lab_value_count"], doc_row["error_message"],
                 doc_row["metadata_json"]),
            )

            # Copy physical files
            src_doc = Path(doc_row["original_path"])
            if src_doc.exists():
                shutil.copy2(src_doc, target_docs / src_doc.name)

            # Copy extraction files
            for suffix in [".md", "_raw.md", "_source.txt", "_cleaned_source.md",
                           ".json", "_tables.json", "_pages.jsonl", "_words.jsonl"]:
                src_file = source / patient_id / "extraction" / f"{doc_row['id']}{suffix}"
                if src_file.exists():
                    shutil.copy2(src_file, target_extraction / f"{new_doc_id}{suffix}")

            stats["documents"] += 1

        # 3. Copy lab values
        lab_rows = source_db.execute(
            "SELECT * FROM lab_values WHERE patient_id=?",
            (patient_id,),
        ).fetchall()
        for lab_row in lab_rows:
            new_doc_id = doc_map.get(lab_row["document_id"], lab_row["document_id"])
            target_db.execute(
                """INSERT INTO lab_values
                   (patient_id, document_id, parameter_name, normalized_name,
                    value, operator, unit, reference_low, reference_high,
                    reference_text, is_abnormal, flag, sample_date,
                    biological_material, lab_name, page, source_text,
                    confidence, validated_by_user)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (new_pid, new_doc_id, lab_row["parameter_name"],
                 lab_row["normalized_name"], lab_row["value"], lab_row["operator"],
                 lab_row["unit"], lab_row["reference_low"], lab_row["reference_high"],
                 lab_row["reference_text"], lab_row["is_abnormal"], lab_row["flag"],
                 lab_row["sample_date"], lab_row["biological_material"],
                 lab_row["lab_name"], lab_row["page"], lab_row["source_text"],
                 lab_row["confidence"], lab_row["validated_by_user"]),
            )

        # 4. Copy timeline entries
        timeline_rows = source_db.execute(
            "SELECT * FROM clinical_timeline WHERE patient_id=? ORDER BY entry_id",
            (patient_id,),
        ).fetchall()
        for tl_row in timeline_rows:
            new_entry_id = self._next_id(target_db, "clinical_timeline", "CTL_")
            # Remap document IDs in source_document_ids
            src_doc_ids = json.loads(tl_row["source_document_ids"] or "[]")
            new_doc_ids = [doc_map.get(d, d) for d in src_doc_ids]
            target_db.execute(
                """INSERT INTO clinical_timeline
                   (entry_id, patient_id, date_observed, date_resolved,
                    category, description, source_document_ids, source_texts,
                    status, confidence, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (new_entry_id, new_pid, tl_row["date_observed"],
                 tl_row["date_resolved"], tl_row["category"],
                 tl_row["description"],
                 json.dumps(new_doc_ids, ensure_ascii=False),
                 tl_row["source_texts"], tl_row["status"],
                 tl_row["confidence"], now, now),
            )
            stats["timeline"] += 1

        # 5. Copy clinical profile
        cs_row = source_db.execute(
            "SELECT * FROM clinical_state WHERE patient_id=?",
            (patient_id,),
        ).fetchone()
        if cs_row and cs_row.get("clinical_profile"):
            state_json = json.loads(cs_row["state_json"])
            state_json["patient_id"] = new_pid
            target_db.execute(
                """INSERT INTO clinical_state
                   (patient_id, state_json, updated_at, version)
                   VALUES (?, ?, ?, ?)""",
                (new_pid, json.dumps(state_json, ensure_ascii=False),
                 now, 1),
            )
            stats["profiles"] += 1

        target_db.commit()

    def _next_id(self, db, table: str, prefix: str) -> str:
        """Generate the next sequential prefixed ID."""
        col = "entry_id" if table == "clinical_timeline" else "id"
        cursor = db.execute(
            f"SELECT {col} FROM {table} ORDER BY {col} DESC LIMIT 1"
        )
        row = cursor.fetchone()
        if row:
            last = row[0]
            num_part = last[len(prefix):]
            if num_part.isdigit():
                return f"{prefix}{int(num_part) + 1:0{len(num_part)}d}"
        return f"{prefix}001"

    def _load_target_ids(self, db) -> None:
        """Pre-load existing IDs to avoid conflicts (no-op, _next_id handles it)."""
        pass
