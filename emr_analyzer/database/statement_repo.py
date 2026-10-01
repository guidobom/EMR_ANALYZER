"""Statement index and certified annotations (one row per occurrence / statement).

The index is rebuilt from text and replaces itself: it is a derived view, never
edited by hand and never a source of clinical truth.  The annotation table keeps
the canonical payload certified for a statement, so a repeated statement is
analysed once and projected onto its copies.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json

from ..clinical.statement_index import (PAYLOAD_VERSION, StatementOccurrence,
                                        assign_roles, content_hash, document_digest,
                                        index_document)


class StatementIndexRepository:
    def __init__(self, db):
        self.db = db

    def rebuild(self, patient_id: str, documents, run_id: str | None = None
                ) -> dict[str, list[StatementOccurrence]]:
        """Recompute the whole patient index from its narrative documents.

        ``documents`` is an iterable of ``(document_id, document_date, text)``.
        Returns the occurrences grouped by document.
        """
        occurrences: list[StatementOccurrence] = []
        for document_id, document_date, text in documents:
            occurrences.extend(index_document(document_id, document_date, text))
        occurrences = assign_roles(occurrences)
        now = datetime.now(timezone.utc).isoformat()
        with self.db:
            self.db.execute("DELETE FROM statement_index WHERE patient_id=?", (patient_id,))
            self.db.executemany(
                """INSERT INTO statement_index (occurrence_id, patient_id, document_id,
                       document_date, sentence_ordinal, sentence_start, sentence_end,
                       sentence_text, normalized_text, previous_text, heading_text,
                       statement_key, role, carrier_occurrence_id, carrier_document_id,
                       carrier_document_date, key_version, source_version, run_id,
                       created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                [(item.occurrence_id, patient_id, item.document_id, item.document_date,
                  item.ordinal, item.start, item.end, item.text, item.normalized_text,
                  item.previous_text, item.heading_text, item.statement_key, item.role,
                  item.carrier_occurrence_id, item.carrier_document_id,
                  item.carrier_document_date, "statement-v1", item.source_version,
                  run_id, now, now) for item in occurrences])
        grouped: dict[str, list[StatementOccurrence]] = {}
        for item in occurrences:
            grouped.setdefault(item.document_id, []).append(item)
        for items in grouped.values():
            items.sort(key=lambda item: item.ordinal)
        return grouped

    def for_document(self, document_id: str) -> list[StatementOccurrence]:
        rows = self.db.execute(
            """SELECT * FROM statement_index WHERE document_id=? ORDER BY sentence_ordinal""",
            (document_id,)).fetchall()
        return [_occurrence(row) for row in rows]

    def digest(self, document_id: str) -> str:
        """Digest of a document's statements and their carriers, for ``input_hash``."""
        return document_digest(self.for_document(document_id))

    def counts(self, patient_id: str) -> dict:
        row = self.db.execute(
            """SELECT COUNT(*) AS occurrences,
                      SUM(role='origin') AS origins, SUM(role='copy') AS copies,
                      COALESCE(SUM(CASE WHEN role='copy' THEN LENGTH(sentence_text) END),0) AS copy_chars
               FROM statement_index WHERE patient_id=?""", (patient_id,)).fetchone()
        return {key: row[key] or 0 for key in ("occurrences", "origins", "copies", "copy_chars")}

    def delete_documents(self, document_ids) -> int:
        ids = list(document_ids)
        if not ids:
            return 0
        marks = ",".join("?" * len(ids))
        with self.db:
            return self.db.execute(
                f"DELETE FROM statement_index WHERE document_id IN ({marks})", tuple(ids)).rowcount


class StatementAnnotationRepository:
    def __init__(self, db):
        self.db = db

    def save(self, *, patient_id: str, statement_key: str, carrier_occurrence_id: str,
             carrier_document_id: str, carrier_document_date: str | None, carrier_sentence: str,
             status: str, payload: list, model_digest: str, prompt_version: str,
             model_name: str | None = None, raw: dict | None = None,
             metrics: dict | None = None, payload_version: str = PAYLOAD_VERSION) -> str:
        annotation_key = annotation_key_for(statement_key, payload_version, model_digest,
                                            prompt_version)
        now = datetime.now(timezone.utc).isoformat()
        with self.db:
            self.db.execute(
                """INSERT INTO statement_annotations (patient_id, statement_key, annotation_key,
                       payload_version, carrier_occurrence_id, carrier_document_id,
                       carrier_document_date, carrier_sentence, model_name, model_digest,
                       prompt_version, status, payload_json, raw_json, metrics_json,
                       created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(patient_id, statement_key, annotation_key) DO UPDATE SET
                       carrier_occurrence_id=excluded.carrier_occurrence_id,
                       carrier_document_id=excluded.carrier_document_id,
                       carrier_document_date=excluded.carrier_document_date,
                       carrier_sentence=excluded.carrier_sentence,
                       status=excluded.status, payload_json=excluded.payload_json,
                       raw_json=excluded.raw_json, metrics_json=excluded.metrics_json,
                       updated_at=excluded.updated_at""",
                (patient_id, statement_key, annotation_key, payload_version,
                 carrier_occurrence_id, carrier_document_id, carrier_document_date,
                 carrier_sentence, model_name, model_digest, prompt_version, status,
                 json.dumps(payload, ensure_ascii=False),
                 json.dumps(raw, ensure_ascii=False) if raw is not None else None,
                 json.dumps(metrics or {}, ensure_ascii=False), now, now))
        return annotation_key

    def get(self, patient_id: str, statement_key: str, model_digest: str,
            prompt_version: str, payload_version: str = PAYLOAD_VERSION) -> dict | None:
        row = self.db.execute(
            """SELECT * FROM statement_annotations
               WHERE patient_id=? AND statement_key=? AND annotation_key=?""",
            (patient_id, statement_key,
             annotation_key_for(statement_key, payload_version, model_digest, prompt_version))
        ).fetchone()
        return _annotation(row) if row else None

    def by_statement(self, patient_id: str, statement_key: str) -> list[dict]:
        rows = self.db.execute(
            """SELECT * FROM statement_annotations WHERE patient_id=? AND statement_key=?
               ORDER BY updated_at DESC""", (patient_id, statement_key)).fetchall()
        return [_annotation(row) for row in rows]

    def by_carrier_document(self, document_id: str) -> list[dict]:
        rows = self.db.execute(
            "SELECT * FROM statement_annotations WHERE carrier_document_id=?",
            (document_id,)).fetchall()
        return [_annotation(row) for row in rows]

    def delete_carrier_documents(self, document_ids) -> int:
        ids = list(document_ids)
        if not ids:
            return 0
        marks = ",".join("?" * len(ids))
        with self.db:
            return self.db.execute(
                f"DELETE FROM statement_annotations WHERE carrier_document_id IN ({marks})",
                tuple(ids)).rowcount


def annotation_key_for(statement_key: str, payload_version: str, model_digest: str,
                       prompt_version: str) -> str:
    return content_hash(statement_key, payload_version, model_digest, prompt_version)


def _occurrence(row) -> StatementOccurrence:
    return StatementOccurrence(
        occurrence_id=row["occurrence_id"], document_id=row["document_id"],
        document_date=row["document_date"], ordinal=row["sentence_ordinal"],
        start=row["sentence_start"], end=row["sentence_end"], text=row["sentence_text"],
        normalized_text=row["normalized_text"], previous_text=row["previous_text"],
        heading_text=row["heading_text"], statement_key=row["statement_key"],
        source_version=row["source_version"], role=row["role"],
        carrier_occurrence_id=row["carrier_occurrence_id"],
        carrier_document_id=row["carrier_document_id"],
        carrier_document_date=row["carrier_document_date"])


def _annotation(row) -> dict:
    data = dict(row)
    data["payload"] = json.loads(data.pop("payload_json") or "[]")
    data["raw"] = json.loads(data.pop("raw_json") or "null")
    data["metrics"] = json.loads(data.pop("metrics_json") or "{}")
    return data


def invalidate_documents(db, document_ids, *, stage: str = "atomic_evidence") -> None:
    """Forget everything a document contributed to the statements of a patient.

    Used when a report changes hands: its index rows, the payloads it carried,
    the events derived from it and its extraction state must not follow it into
    another patient's history.  The statement index is rebuilt from text on the
    next run, so a statement whose carrier moved promotes a new carrier.
    """
    ids = list(document_ids)
    if not ids:
        return
    marks = ",".join("?" * len(ids))
    with db:
        db.execute(f"DELETE FROM statement_index WHERE document_id IN ({marks})", tuple(ids))
        db.execute(f"DELETE FROM statement_annotations WHERE carrier_document_id IN ({marks})",
                   tuple(ids))
        db.execute(f"DELETE FROM clinical_evidence WHERE document_id IN ({marks}) "
                   f"AND extraction_method='fhir_events_v1'", tuple(ids))
        db.execute(f"DELETE FROM processing_manifest WHERE document_id IN ({marks}) "
                   f"AND stage=?", (*ids, stage))
