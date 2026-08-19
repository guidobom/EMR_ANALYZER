"""Per-patient clinical query chat repository."""

import uuid
from datetime import datetime

from .engine import DatabaseEngine
from ..models.chat_message import ChatMessage


class ChatRepository:
    """Data access for the clinical_chat table."""

    def __init__(self, db: DatabaseEngine):
        self.db = db

    def add_message(self, message: ChatMessage) -> None:
        """Persist one prompt or response."""
        self.db.execute(
            """INSERT INTO clinical_chat
               (id, patient_id, role, content, model_used, context_mode,
                created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                message.id,
                message.patient_id,
                message.role,
                message.content,
                message.model_used,
                int(message.context_mode),
                message.created_at or datetime.now().isoformat(),
            ),
        )
        self.db.commit()

    def get_by_patient(self, patient_id: str) -> list[ChatMessage]:
        """All messages of one patient, chronological."""
        rows = self.db.execute(
            """SELECT id, patient_id, role, content, model_used,
                      context_mode, created_at
               FROM clinical_chat
               WHERE patient_id = ?
               ORDER BY created_at ASC, rowid ASC""",
            (patient_id,),
        ).fetchall()
        return [
            ChatMessage(
                id=row["id"],
                patient_id=row["patient_id"],
                role=row["role"],
                content=row["content"],
                model_used=row["model_used"] or "",
                context_mode=int(row["context_mode"] or 0),
                created_at=row["created_at"] or "",
            )
            for row in rows
        ]

    def count_by_patient(self, patient_id: str) -> int:
        row = self.db.execute(
            "SELECT COUNT(*) AS n FROM clinical_chat WHERE patient_id = ?",
            (patient_id,),
        ).fetchone()
        return int(row["n"]) if row else 0

    def clear_for_patient(self, patient_id: str) -> None:
        """Remove every message of one patient."""
        self.db.execute(
            "DELETE FROM clinical_chat WHERE patient_id = ?", (patient_id,)
        )
        self.db.commit()

    @staticmethod
    def new_id() -> str:
        """Deterministic-format unique id for a message."""
        return "CHAT_" + uuid.uuid4().hex
