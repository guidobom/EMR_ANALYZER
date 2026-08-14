"""Persistence and matching for pseudonymized patient identity keys."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets

from .engine import DatabaseEngine
from ..config import IDENTITY_KEY_PATH
from ..models.patient_identity import PatientIdentityEvidence


@dataclass(frozen=True)
class IdentityMatch:
    patient_id: str | None = None
    confidence: float = 0.0
    reason: str = ""
    conflict: bool = False


class IdentityKeyService:
    """Create deterministic HMAC keys without persisting raw identifiers."""

    def __init__(self, key_path: str | Path = IDENTITY_KEY_PATH):
        self._key_path = Path(key_path)
        self._key: bytes | None = None

    def digest(self, field_name: str, normalized_value: str) -> str:
        if not normalized_value:
            return ""
        payload = f"{field_name}\0{normalized_value}".encode("utf-8")
        return hmac.new(self._load_key(), payload, hashlib.sha256).hexdigest()

    def _load_key(self) -> bytes:
        if self._key is not None:
            return self._key
        self._key_path.parent.mkdir(parents=True, exist_ok=True)
        if self._key_path.exists():
            self._key = self._key_path.read_bytes()
        else:
            self._key = secrets.token_bytes(32)
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            descriptor = os.open(self._key_path, flags, 0o600)
            try:
                os.write(descriptor, self._key)
            finally:
                os.close(descriptor)
        if len(self._key) < 32:
            raise ValueError("La chiave locale per le identità non è valida")
        return self._key


class PatientIdentityRepository:
    """Store identity fingerprints and resolve incoming documents."""

    def __init__(self, db: DatabaseEngine, key_service: IdentityKeyService | None = None):
        self.db = db
        self.keys = key_service or IdentityKeyService()

    def has_identity(self, patient_id: str) -> bool:
        row = self.db.execute(
            "SELECT 1 FROM patient_identities WHERE patient_id=?", (patient_id,)
        ).fetchone()
        return row is not None

    def upsert(
        self,
        patient_id: str,
        evidence: PatientIdentityEvidence,
        *,
        source_document_id: str | None = None,
        status: str = "auto",
    ) -> None:
        now = datetime.now().isoformat()
        field_keys = self._evidence_keys(evidence)
        self.db.execute(
            """INSERT INTO patient_identities
               (patient_id, fiscal_code_key, normalized_name_key,
                birth_date_key, sex, hospital_patient_id_key, confidence,
                status, source_document_id, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(patient_id) DO UPDATE SET
                 fiscal_code_key=COALESCE(excluded.fiscal_code_key, fiscal_code_key),
                 normalized_name_key=COALESCE(excluded.normalized_name_key, normalized_name_key),
                 birth_date_key=COALESCE(excluded.birth_date_key, birth_date_key),
                 sex=COALESCE(excluded.sex, sex),
                 hospital_patient_id_key=COALESCE(excluded.hospital_patient_id_key, hospital_patient_id_key),
                 confidence=MAX(confidence, excluded.confidence),
                 status=excluded.status,
                 source_document_id=COALESCE(excluded.source_document_id, source_document_id),
                 updated_at=excluded.updated_at""",
            (
                patient_id,
                field_keys.get("fiscal_code"),
                field_keys.get("name"),
                field_keys.get("birth_date"),
                evidence.sex.normalized if evidence.sex else None,
                field_keys.get("hospital_patient_id"),
                evidence.confidence,
                status,
                source_document_id,
                now,
                now,
            ),
        )
        hospital_id_key = field_keys.get("hospital_patient_id")
        if hospital_id_key:
            self.db.execute(
                """INSERT OR IGNORE INTO patient_hospital_ids
                   (patient_id, hospital_patient_id_key, created_at, updated_at)
                   VALUES (?, ?, ?, ?)""",
                (patient_id, hospital_id_key, now, now),
            )
        self.db.commit()

    def find_match(self, evidence: PatientIdentityEvidence) -> IdentityMatch:
        """Match only strong anchored evidence; surface cross-field conflicts."""

        keys = self._evidence_keys(evidence)
        cf_patients = self._patients_for_key(
            "fiscal_code_key", keys.get("fiscal_code")
        )
        if len(cf_patients) > 1:
            return IdentityMatch(
                reason="Il codice fiscale risulta associato a più workspace",
                conflict=True,
            )
        cf_patient = cf_patients[0] if cf_patients else None

        hp_patients = self._patients_for_hospital_id(
            keys.get("hospital_patient_id")
        )
        if len(hp_patients) > 1:
            return IdentityMatch(
                reason="L'ID paziente ospedaliero risulta associato a più workspace",
                conflict=True,
            )
        hp_patient = hp_patients[0] if hp_patients else None

        pair_patient = None
        if keys.get("name") and keys.get("birth_date"):
            row = self.db.execute(
                """SELECT patient_id FROM patient_identities
                   WHERE normalized_name_key=? AND birth_date_key=?""",
                (keys["name"], keys["birth_date"]),
            ).fetchone()
            pair_patient = row["patient_id"] if row else None

        if cf_patient and pair_patient and cf_patient != pair_patient:
            return IdentityMatch(
                reason="Codice fiscale e nome/data corrispondono a workspace diversi",
                conflict=True,
            )
        if hp_patient and pair_patient and hp_patient != pair_patient:
            return IdentityMatch(
                reason="ID ospedaliero e nome/data corrispondono a workspace diversi",
                conflict=True,
            )
        if cf_patient and hp_patient and cf_patient != hp_patient:
            return IdentityMatch(
                reason="Codice fiscale e ID ospedaliero corrispondono a workspace diversi",
                conflict=True,
            )

        patient_id = cf_patient or hp_patient or pair_patient
        if not patient_id:
            return IdentityMatch(reason="Nessuna identità registrata compatibile")

        row = self.db.execute(
            "SELECT * FROM patient_identities WHERE patient_id=?", (patient_id,)
        ).fetchone()
        conflicts = []
        incoming_cf = keys.get("fiscal_code")
        incoming_hpid = keys.get("hospital_patient_id")
        incoming_name = keys.get("name")
        incoming_birth = keys.get("birth_date")
        if incoming_cf and row and row["fiscal_code_key"] \
                and incoming_cf != row["fiscal_code_key"]:
            conflicts.append("fiscal_code")
        # A patient legitimately holds several hospital patient IDs (one per
        # hospital unit): only a value absent from the registered set is a
        # conflict, not a different one of the patient's own IDs.
        registered_hpids = set(self._hospital_patient_ids(patient_id))
        if row and row["hospital_patient_id_key"]:
            registered_hpids.add(row["hospital_patient_id_key"])
        if incoming_hpid and registered_hpids \
                and incoming_hpid not in registered_hpids:
            conflicts.append("hospital_patient_id")
        if incoming_name and row and row["normalized_name_key"] \
                and incoming_name != row["normalized_name_key"]:
            conflicts.append("name")
        if incoming_birth and row and row["birth_date_key"] \
                and incoming_birth != row["birth_date_key"]:
            conflicts.append("birth_date")
        if conflicts:
            return IdentityMatch(
                patient_id=patient_id,
                reason="Conflitto negli identificatori: " + ", ".join(conflicts),
                conflict=True,
            )
        if cf_patient:
            reason = "codice fiscale"
            confidence = 0.99
        elif hp_patient:
            reason = "ID paziente ospedaliero"
            confidence = 0.98
        else:
            reason = "nome e data di nascita"
            confidence = min(evidence.confidence, 0.96)
        return IdentityMatch(patient_id, confidence, f"Corrispondenza per {reason}")

    def add_document_evidence(
        self, document_id: str, patient_id: str, evidence: PatientIdentityEvidence
    ) -> None:
        now = datetime.now().isoformat()
        rows = []
        for field_name, field in evidence.fields.items():
            rows.append((
                document_id,
                patient_id,
                field_name,
                self.keys.digest(field_name, field.normalized),
                field.page,
                json.dumps(field.bbox) if field.bbox else None,
                field.method,
                field.confidence,
                now,
            ))
        if rows:
            self.db.executemany(
                """INSERT INTO document_identity_evidence
                   (document_id, patient_id, field_name, value_key, page,
                    bbox_json, extraction_method, confidence, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
            self.db.commit()

    def _evidence_keys(self, evidence: PatientIdentityEvidence) -> dict[str, str]:
        keys = {}
        for name, field in evidence.fields.items():
            normalized = field.normalized
            if name == "name":
                # Explicit headers may use either NOME COGNOME or COGNOME NOME.
                # Birth date/CF provide the disambiguating second identifier.
                normalized = " ".join(sorted(normalized.split()))
            keys[name] = self.keys.digest(name, normalized)
        return keys

    def _patients_for_key(self, column: str, value: str | None) -> list[str]:
        if not value:
            return []
        if column not in {"fiscal_code_key", "hospital_patient_id_key"}:
            raise ValueError("Campo identità non consentito")
        rows = self.db.execute(
            f"SELECT patient_id FROM patient_identities WHERE {column}=?", (value,)
        ).fetchall()
        return [row["patient_id"] for row in rows]

    def _hospital_patient_ids(self, patient_id: str) -> list[str]:
        """All hospital-patient-id digests registered for a patient."""
        rows = self.db.execute(
            "SELECT hospital_patient_id_key FROM patient_hospital_ids "
            "WHERE patient_id=?",
            (patient_id,),
        ).fetchall()
        return [row["hospital_patient_id_key"] for row in rows]

    def add_hospital_patient_id(self, patient_id: str, value: str) -> None:
        """Register one more hospital patient ID for a patient (idempotent).

        A person holds several hospital IDs across the units of a trust, so
        the registry keeps a set per patient rather than a single column.
        Only the keyed digest is persisted.
        """
        if not value:
            return
        now = datetime.now().isoformat()
        self.db.execute(
            """INSERT OR IGNORE INTO patient_hospital_ids
               (patient_id, hospital_patient_id_key, created_at, updated_at)
               VALUES (?, ?, ?, ?)""",
            (patient_id, self.keys.digest("hospital_patient_id", value),
             now, now),
        )
        self.db.commit()

    def _patients_for_hospital_id(self, value: str | None) -> list[str]:
        """Patients holding this hospital-patient-id digest.

        The multi-valued table is the source of truth; the legacy single
        column is still consulted so databases that predate the table match
        the same way.
        """
        if not value:
            return []
        rows = self.db.execute(
            """SELECT DISTINCT patient_id FROM (
                 SELECT patient_id FROM patient_hospital_ids
                  WHERE hospital_patient_id_key=?
                 UNION
                 SELECT patient_id FROM patient_identities
                  WHERE hospital_patient_id_key=?
               )""",
            (value, value),
        ).fetchall()
        return [row["patient_id"] for row in rows]
