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
    partial_matches: int = 0


@dataclass
class RoutingPlan:
    """Conservative import plan built from resolved identity groups."""

    matched_groups: list[RoutingGroup] = field(default_factory=list)
    new_groups: list[RoutingGroup] = field(default_factory=list)
    blocked_groups: list[RoutingGroup] = field(default_factory=list)

    @property
    def importable_count(self) -> int:
        return sum(
            len(group.documents)
            for group in self.matched_groups + self.new_groups
        )


def build_routing_plan(groups: list[RoutingGroup]) -> RoutingPlan:
    """Keep strong identities separate and quarantine ambiguous documents.

    A missing or conflicting identity is never inferred from another group in
    the same batch. This prevents a large mixed-patient import from collapsing
    into the largest detected workspace.
    """

    return RoutingPlan(
        matched_groups=[
            group
            for group in groups
            if group.patient_id and not group.needs_review
        ],
        new_groups=[group for group in groups if group.create_new],
        blocked_groups=[
            group
            for group in groups
            if group.needs_review or (not group.patient_id and not group.create_new)
        ],
    )


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
        eligible = [
            document
            for document in documents
            if not document.is_duplicate and not document.error
        ]
        grouped = self._identity_components(eligible)

        result = []
        for key, members in grouped.items():
            representative = max(
                members,
                key=lambda document: (
                    document.evidence.is_strong,
                    document.evidence.confidence,
                    len(document.evidence.fields),
                ),
            )
            evidence = representative.evidence
            group = RoutingGroup(
                key=key,
                documents=members,
                evidence=evidence,
                partial_matches=sum(
                    not document.evidence.is_strong for document in members
                ),
            )
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

    @classmethod
    def _identity_components(
        cls, documents: list[StagedDocument]
    ) -> dict[str, list[StagedDocument]]:
        """Build connected identity groups using every strong identifier.

        A document containing both CF and name/date acts as a bridge between
        reports that expose only one of those identifiers. If the batch has
        exactly one unambiguous strong component, reports with compatible
        partial evidence can join it; this is never done in a multi-patient
        batch.
        """

        if not documents:
            return {}

        parents = list(range(len(documents)))

        def find(index: int) -> int:
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        def union(first: int, second: int) -> None:
            first_root = find(first)
            second_root = find(second)
            if first_root != second_root:
                parents[second_root] = first_root

        key_owner: dict[str, int] = {}
        for index, document in enumerate(documents):
            for key in cls._evidence_link_keys(document.evidence):
                owner = key_owner.setdefault(key, index)
                union(index, owner)

        strong_roots = {
            find(index)
            for index, document in enumerate(documents)
            if document.evidence.is_strong
        }
        if len(strong_roots) == 1:
            anchor_root = next(iter(strong_roots))
            anchor_evidence = [
                document.evidence
                for index, document in enumerate(documents)
                if find(index) == anchor_root
            ]
            anchor_values = cls._consistent_values(anchor_evidence)
            if anchor_values is not None:
                for index, document in enumerate(documents):
                    if find(index) == anchor_root:
                        continue
                    if cls._compatible_partial_evidence(
                        document.evidence, anchor_values
                    ):
                        union(anchor_root, index)

        components: dict[int, list[tuple[int, StagedDocument]]] = defaultdict(list)
        for index, document in enumerate(documents):
            components[find(index)].append((index, document))

        grouped: dict[str, list[StagedDocument]] = {}
        for items in sorted(components.values(), key=lambda value: value[0][0]):
            members = [document for _, document in items]
            representative = max(
                members,
                key=lambda document: (
                    document.evidence.is_strong,
                    document.evidence.confidence,
                    len(document.evidence.fields),
                ),
            )
            key = (
                representative.evidence.group_key
                or f"unresolved:{items[0][0]}"
            )
            grouped[key] = members
        return grouped

    @staticmethod
    def _evidence_link_keys(evidence: PatientIdentityEvidence) -> list[str]:
        keys = []
        if evidence.fiscal_code:
            keys.append(f"cf:{evidence.fiscal_code.normalized}")
        if evidence.name and evidence.birth_date:
            canonical_name = " ".join(
                sorted(evidence.name.normalized.split())
            )
            keys.append(
                f"name_birth:{canonical_name}|"
                f"{evidence.birth_date.normalized}"
            )
        return keys

    @staticmethod
    def _consistent_values(
        evidence_items: list[PatientIdentityEvidence],
    ) -> dict[str, set[str]] | None:
        values = {
            "fiscal_code": {
                evidence.fiscal_code.normalized
                for evidence in evidence_items if evidence.fiscal_code
            },
            "name": {
                " ".join(sorted(evidence.name.normalized.split()))
                for evidence in evidence_items if evidence.name
            },
            "birth_date": {
                evidence.birth_date.normalized
                for evidence in evidence_items if evidence.birth_date
            },
            "sex": {
                evidence.sex.normalized
                for evidence in evidence_items if evidence.sex
            },
        }
        if any(len(field_values) > 1 for field_values in values.values()):
            return None
        return values

    @staticmethod
    def _compatible_partial_evidence(
        evidence: PatientIdentityEvidence,
        anchor_values: dict[str, set[str]],
    ) -> bool:
        incoming = {
            "fiscal_code": (
                evidence.fiscal_code.normalized
                if evidence.fiscal_code else None
            ),
            "name": (
                " ".join(sorted(evidence.name.normalized.split()))
                if evidence.name else None
            ),
            "birth_date": (
                evidence.birth_date.normalized
                if evidence.birth_date else None
            ),
            "sex": evidence.sex.normalized if evidence.sex else None,
        }
        matched_identity_field = False
        for field_name in ("fiscal_code", "name", "birth_date", "sex"):
            value = incoming[field_name]
            expected = anchor_values[field_name]
            if value is None or not expected:
                continue
            if value not in expected:
                return False
            if field_name in {"fiscal_code", "name", "birth_date"}:
                matched_identity_field = True
        return matched_identity_field

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
