"""One persisted chat message of the clinical query history."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"


@dataclass
class ChatMessage:
    """A single prompt or response of the per-patient query chat."""

    id: str
    patient_id: str
    role: str
    content: str
    model_used: str = ""
    context_mode: int = 0
    created_at: str = field(default="")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ChatMessage":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
