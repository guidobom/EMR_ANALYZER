"""Shared concept → SNOMED CT mappings (label and event type, no patient data)."""

from __future__ import annotations

from datetime import datetime, timezone
import json

# proposed      model choice among catalog candidates, to be reviewed
# confirmed     reviewer decision (a code, or deliberately no code)
# needs_review  no adequate candidate or the model abstained
STATUSES = ("proposed", "confirmed", "needs_review")

SCHEMA = """CREATE TABLE IF NOT EXISTS concept_mappings (
    label_key TEXT NOT NULL,
    fact_type TEXT NOT NULL,
    label TEXT NOT NULL,
    status TEXT NOT NULL,
    code TEXT,
    fsn TEXT,
    term TEXT,
    release TEXT,
    source TEXT NOT NULL,
    english TEXT,
    candidates_json TEXT NOT NULL DEFAULT '[]',
    examples_json TEXT NOT NULL DEFAULT '[]',
    model_digest TEXT,
    note TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    reviewed_at TEXT,
    PRIMARY KEY(label_key, fact_type)
)"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ConceptMappingRepository:
    def __init__(self, db):
        self.db = db
        with self.db:
            self.db.execute(SCHEMA)

    def get(self, label_key: str, fact_type: str) -> dict | None:
        row = self.db.execute(
            "SELECT * FROM concept_mappings WHERE label_key=? AND fact_type=?",
            (label_key, fact_type)).fetchone()
        return _row(row) if row else None

    def get_many(self, keys) -> dict[tuple[str, str], dict]:
        keys = list(dict.fromkeys(keys))
        result = {}
        for start in range(0, len(keys), 400):
            chunk = keys[start:start + 400]
            clause = " OR ".join("(label_key=? AND fact_type=?)" for _ in chunk)
            for row in self.db.execute(f"SELECT * FROM concept_mappings WHERE {clause}",
                                       tuple(value for key in chunk for value in key)):
                result[(row["label_key"], row["fact_type"])] = _row(row)
        return result

    def all(self, status: str | None = None) -> list[dict]:
        query, args = "SELECT * FROM concept_mappings", ()
        if status:
            query, args = query + " WHERE status=?", (status,)
        return [_row(row) for row in self.db.execute(query + " ORDER BY label_key, fact_type", args)]

    def propose(self, label_key, fact_type, label, *, status, code=None, concept=None,
                release=None, source="llm", english=None, candidates=(), examples=(),
                model_digest=None, note=None) -> bool:
        """Store an automatic outcome; a reviewer's decision is never overwritten."""
        if status not in ("proposed", "needs_review"):
            raise ValueError("Stato automatico non valido")
        now = _now()
        with self.db:
            cursor = self.db.execute(
                """INSERT INTO concept_mappings (label_key, fact_type, label, status, code, fsn, term,
                       release, source, english, candidates_json, examples_json, model_digest, note,
                       created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(label_key, fact_type) DO UPDATE SET
                       label=excluded.label, status=excluded.status, code=excluded.code,
                       fsn=excluded.fsn, term=excluded.term, release=excluded.release,
                       source=excluded.source, english=excluded.english,
                       candidates_json=excluded.candidates_json, examples_json=excluded.examples_json,
                       model_digest=excluded.model_digest, note=excluded.note,
                       updated_at=excluded.updated_at
                   WHERE concept_mappings.status <> 'confirmed'""",
                (label_key, fact_type, label, status, code,
                 (concept or {}).get("fsn"), (concept or {}).get("term"), release, source, english,
                 json.dumps(list(candidates), ensure_ascii=False),
                 json.dumps(list(examples), ensure_ascii=False), model_digest, note, now, now))
        return cursor.rowcount > 0

    def confirm(self, label_key, fact_type, label, *, code=None, concept=None, release=None,
                note=None) -> None:
        """Reviewer decision for every occurrence of the concept (code may be None)."""
        now = _now()
        with self.db:
            self.db.execute(
                """INSERT INTO concept_mappings (label_key, fact_type, label, status, code, fsn, term,
                       release, source, note, created_at, updated_at, reviewed_at)
                   VALUES (?,?,?,'confirmed',?,?,?,?,'manual',?,?,?,?)
                   ON CONFLICT(label_key, fact_type) DO UPDATE SET
                       status='confirmed', code=excluded.code, fsn=excluded.fsn, term=excluded.term,
                       release=excluded.release, source='manual', note=excluded.note,
                       updated_at=excluded.updated_at, reviewed_at=excluded.reviewed_at""",
                (label_key, fact_type, label, code, (concept or {}).get("fsn"),
                 (concept or {}).get("term"), release, note, now, now, now))

    def reset(self, keys=None, *, status: str | None = None) -> int:
        """Forget automatic outcomes so they are recomputed (confirmed rows stay)."""
        with self.db:
            if keys is None:
                query = "DELETE FROM concept_mappings WHERE status<>'confirmed'"
                args = ()
                if status:
                    query, args = query + " AND status=?", (status,)
                return self.db.execute(query, args).rowcount
            removed = 0
            for label_key, fact_type in keys:
                removed += self.db.execute(
                    """DELETE FROM concept_mappings WHERE label_key=? AND fact_type=?
                       AND status<>'confirmed'""", (label_key, fact_type)).rowcount
            return removed


def _row(row) -> dict:
    data = dict(row)
    data["candidates"] = json.loads(data.pop("candidates_json") or "[]")
    data["examples"] = json.loads(data.pop("examples_json") or "[]")
    return data
