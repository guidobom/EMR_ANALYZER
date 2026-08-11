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

        initial = [
            RoutingGroup(key=key, documents=members, evidence=members[0].evidence)
            for key, members in grouped.items()
        ]
        # Documents describing the same person may land in different groups
        # (e.g. one report carries a fiscal code, another only name+birth).
        # No patient is created until resolve() returns, so these groups could
        # never see each other through the persistent registry and would each
        # spawn a new workspace.  Merge them up-front by shared strong signals.
        merged_groups = self._merge_batch_identities(initial)

        result = []
        for group in merged_groups:
            members = group.documents
            evidence = group.evidence
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
    def _identity_fingerprints(
        evidence: PatientIdentityEvidence | None,
    ) -> tuple[tuple, ...]:
        """Strong identity signals used to merge groups within one batch.

        Mirrors the matching rules of the identity repository: a fiscal code
        alone is authoritative, otherwise name+birth date together.  Only the
        ordering-insensitive name is used, so inverted initials still match.
        """
        if evidence is None:
            return ()
        fingerprints = []
        if evidence.fiscal_code:
            fingerprints.append(("cf", evidence.fiscal_code.normalized))
        if evidence.name and evidence.birth_date:
            canonical_name = " ".join(sorted(evidence.name.normalized.split()))
            fingerprints.append(("nb", canonical_name, evidence.birth_date.normalized))
        return tuple(fingerprints)

    @classmethod
    def _merge_batch_identities(
        cls, groups: list[RoutingGroup]
    ) -> list[RoutingGroup]:
        """Merge groups whose evidence identifies the same person.

        Two groups describe the same person when they share a fiscal code or
        share name+birth date.  The representative evidence kept is the most
        identifying one (fiscal code preferred), so the created workspace
        registers the richest identity available in the batch.
        """
        merged: list[RoutingGroup] = []
        by_fingerprint: dict[tuple, RoutingGroup] = {}
        for group in groups:
            if not group.documents:
                continue
            fingerprints = cls._identity_fingerprints(group.evidence)
            target = next(
                (by_fingerprint[fp] for fp in fingerprints if fp in by_fingerprint),
                None,
            )
            if target is None:
                target = RoutingGroup(
                    key=group.key,
                    documents=list(group.documents),
                    evidence=group.evidence,
                )
                merged.append(target)
            else:
                target.documents.extend(group.documents)
                target.evidence = cls._prefer_evidence(target.evidence, group.evidence)
            for fp in fingerprints:
                by_fingerprint.setdefault(fp, target)
        return merged

    @staticmethod
    def _prefer_evidence(
        current: PatientIdentityEvidence | None,
        candidate: PatientIdentityEvidence | None,
    ) -> PatientIdentityEvidence | None:
        """Keep the most identifying evidence when merging groups."""
        if candidate is None:
            return current
        if current is None:
            return candidate

        def rank(evidence: PatientIdentityEvidence) -> int:
            if evidence.fiscal_code:
                return 3
            if evidence.name and evidence.birth_date:
                return 2
            if evidence.name:
                return 1
            return 0

        candidate_rank = rank(candidate)
        current_rank = rank(current)
        if candidate_rank > current_rank:
            return candidate
        if candidate_rank < current_rank:
            return current
        return candidate if candidate.confidence > current.confidence else current

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
