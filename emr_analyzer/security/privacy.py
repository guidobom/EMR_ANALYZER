"""Privacy-safe serialization for logs and review payloads.

Direct identifiers may be used transiently for routing but must never be
persisted in generic JSON audit or validation fields.  This module keeps only
field-presence and confidence metadata for identity verdicts and applies the
same deterministic redactor used for clinical prose to free-text payloads.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..pipeline.sensitive_data import SensitiveDataSanitizer


_IDENTITY_CONTAINER_KEYS = {
    "identity", "llm_identity", "patient_identity", "raw_identity",
}
_DIRECT_IDENTIFIER_KEYS = {
    "name", "full_name", "patient_name", "surname", "given_name",
    "fiscal_code", "codice_fiscale", "birth_date", "date_of_birth",
    "hospital_patient_id", "email", "phone", "address",
}


def identity_metadata(identity: Any) -> dict[str, Any]:
    """Return non-identifying metadata for an in-memory identity result."""
    if not isinstance(identity, Mapping):
        return {"fields_present": [], "redacted": True}
    if identity.get("redacted") is True and isinstance(
        identity.get("fields_present"), list
    ):
        result: dict[str, Any] = {
            "fields_present": sorted({
                str(field) for field in identity["fields_present"]
            }),
            "redacted": True,
        }
        confidence = identity.get("confidence")
        if isinstance(confidence, (int, float)):
            result["confidence"] = max(0.0, min(float(confidence), 1.0))
        return result
    fields = sorted(
        str(key) for key, value in identity.items()
        if value not in (None, "", [], {}) and key != "confidence"
    )
    result: dict[str, Any] = {
        "fields_present": fields,
        "redacted": True,
    }
    confidence = identity.get("confidence")
    if isinstance(confidence, (int, float)):
        result["confidence"] = max(0.0, min(float(confidence), 1.0))
    return result


def sanitize_for_persistence(value: Any, *, _key: str = "") -> Any:
    """Recursively remove direct identifiers from a serializable payload."""
    key = str(_key or "").lower()
    if key in _IDENTITY_CONTAINER_KEYS:
        return identity_metadata(value)
    if key in _DIRECT_IDENTIFIER_KEYS:
        return "[DATO IDENTIFICATIVO RIMOSSO]"
    if isinstance(value, Mapping):
        return {
            str(child_key): sanitize_for_persistence(
                child_value, _key=str(child_key)
            )
            for child_key, child_value in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [sanitize_for_persistence(item) for item in value]
    if isinstance(value, str):
        return SensitiveDataSanitizer().sanitize(value).text
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return sanitize_for_persistence(str(value))
