"""Resolve staged documents to existing or new patient workspaces."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from ..models.patient_identity import PatientIdentityEvidence
from .import_staging import StagedDocument


@dataclass
class RoutingGroup:
    key: str
    documents: list[StagedDocument] = field(default_factory=list)
    evidence: PatientIdentityEvidence | None = None
    patient_id: str | None = None
    create_new: bool = False
    needs_review: bool = False
    conflict: bool = False
    reason: str = ""


class PatientRoutingService:
    """Group documents by identity and match them against the registry."""

    def __init__(self, identity_repo, identity_extractor, patient_repo, document_repo,
                 audit_repo=None):
        self.identity_repo = identity_repo
        self.identity_extractor = identity_extractor
        self.patient_repo = patient_repo
        self.document_repo = document_repo
        self.audit_repo = audit_repo

    def bootstrap_existing_identities(self, sample_size: int = 3) -> int:
        """Lazily register strong identities for pre-existing workspaces."""

        created = 0
        for patient in self.patient_repo.list_all():
            if self.identity_repo.has_identity(patient.id):
                continue
            documents = self.document_repo.list_by_patient(patient.id)[:sample_size]
            samples = []
            for document in documents:
                evidence = self.identity_extractor.extract(document.original_path)
                if evidence.is_strong and evidence.group_key:
                    samples.append((document, evidence))
            if not samples:
                continue
            group_keys = {evidence.group_key for _, evidence in samples}
            if len(group_keys) != 1:
                continue
            document, evidence = samples[0]
            self.identity_repo.upsert(
                patient.id, evidence, source_document_id=document.id,
                status="bootstrap",
            )
            if self.audit_repo:
                self.audit_repo.log(
                    patient.id,
                    "identity_bootstrap",
                    "patient_identity",
                    patient.id,
                    {
                        "source_document_id": document.id,
                        "sample_count": len(samples),
                        "confidence": evidence.confidence,
                    },
                )
            created += 1
        return created

    def resolve(self, documents: list[StagedDocument]) -> list[RoutingGroup]:
        grouped: dict[str, list[StagedDocument]] = defaultdict(list)
        for index, document in enumerate(documents):
            if document.is_duplicate or document.error:
                continue
            key = document.evidence.group_key or f"unresolved:{index}"
            grouped[key].append(document)

        result = []
        for key, members in grouped.items():
            evidence = members[0].evidence
            group = RoutingGroup(key=key, documents=members, evidence=evidence)
            conflict_fields = self._conflicting_fields(members)
            if conflict_fields:
                group.needs_review = True
                group.conflict = True
                group.reason = "Valori discordanti nel gruppo: " + ", ".join(conflict_fields)
                result.append(group)
                continue
            match = self.identity_repo.find_match(evidence)
            if match.conflict:
                group.patient_id = match.patient_id
                group.needs_review = True
                group.conflict = True
                group.reason = match.reason
            elif match.patient_id:
                group.patient_id = match.patient_id
                group.reason = match.reason
            elif evidence.is_strong:
                group.create_new = True
                group.reason = "Nuova identità forte"
            else:
                group.needs_review = True
                group.reason = "Identità insufficiente per l'attribuzione automatica"
            result.append(group)
        return result

    @staticmethod
    def _conflicting_fields(documents: list[StagedDocument]) -> list[str]:
        conflicts = []
        for field_name in ("name", "fiscal_code", "birth_date", "sex"):
            values = set()
            for document in documents:
                field = getattr(document.evidence, field_name)
                if field is None:
                    continue
                value = field.normalized
                if field_name == "name":
                    value = " ".join(sorted(value.split()))
                values.add(value)
            if len(values) > 1:
                conflicts.append(field_name)
        return conflicts
