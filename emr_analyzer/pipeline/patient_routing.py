"""Resolve staged documents to existing or new patient workspaces."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

from ..models.patient_identity import PatientIdentityEvidence
from .import_staging import StagedDocument
from .patient_identity import normalize_text


def _extract_document_text(path: str) -> str:
    """Raw text of the first pages, for the surname/birth-date search.

    The coordinate-aware extractor may miss the demographic header when the
    label/value order differs (value above the label, two-column layouts),
    yet the text still carries the patient's name.  This returns the raw
    page text so a second pass can search it directly.
    """
    import fitz

    try:
        document = fitz.open(str(path))
    except Exception:
        return ""
    try:
        chunks = []
        for index in range(min(document.page_count, 3)):
            chunks.append(document[index].get_text())
        return " ".join(chunks)
    finally:
        document.close()


def _birth_date_in_text(text: str, birth_iso: str) -> bool:
    """True if the dd/mm/yyyy form of the birth date appears in the text."""
    parts = birth_iso.split("-")
    if len(parts) != 3:
        return False
    year, month, day = parts

    def variants(value: str) -> set[str]:
        number = int(value)
        return {str(number), f"{number:02d}"}

    alternatives = [
        f"{d}[./\\-]{m}[./\\-]{year}"
        for d in variants(day)
        for m in variants(month)
    ]
    pattern = re.compile(
        r"(?<!\d)(?:%s)(?!\d)" % "|".join(alternatives)
    )
    return pattern.search(text) is not None


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
        self._auto_assign_unresolved(result)
        return result

    @staticmethod
    def _identity_fingerprints(
        evidence: PatientIdentityEvidence | None,
    ) -> tuple[tuple, ...]:
        """Strong identity signals used to merge groups within one batch.

        Mirrors the matching rules of the identity repository: a fiscal code
        or hospital patient ID alone is authoritative, otherwise name+birth
        date together.  Only the ordering-insensitive name is used, so
        inverted initials still match.
        """
        if evidence is None:
            return ()
        fingerprints = []
        if evidence.fiscal_code:
            fingerprints.append(("cf", evidence.fiscal_code.normalized))
        if evidence.hospital_patient_id:
            fingerprints.append(("hp", evidence.hospital_patient_id.normalized))
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

        # A name that wraps onto a second line may still yield a truncated
        # extraction (``GUERRA VILLIAM``) next to the full one (``GUERRA
        # VILLIAM SILVESTRO``).  Those share the birth date and differ only
        # by a token subset, so fold them together before matching.
        index = 0
        while index < len(merged):
            other = index + 1
            while other < len(merged):
                if cls._token_subset_merge(merged[index], merged[other]):
                    merged[index].documents.extend(merged[other].documents)
                    merged[index].evidence = cls._prefer_evidence(
                        merged[index].evidence, merged[other].evidence
                    )
                    merged.pop(other)
                else:
                    other += 1
            index += 1
        return merged

    @staticmethod
    def _token_subset_merge(
        first: RoutingGroup, second: RoutingGroup
    ) -> bool:
        """True when both groups name the same person by birth + subset name."""
        first_evidence = first.evidence
        second_evidence = second.evidence
        if not (first_evidence and second_evidence
                and first_evidence.name and second_evidence.name
                and first_evidence.birth_date and second_evidence.birth_date):
            return False
        if (first_evidence.birth_date.normalized
                != second_evidence.birth_date.normalized):
            return False
        first_tokens = set(first_evidence.name.normalized.split())
        second_tokens = set(second_evidence.name.normalized.split())
        return first_tokens <= second_tokens or second_tokens <= first_tokens

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

        def score(evidence: PatientIdentityEvidence) -> int:
            total = 0
            if evidence.fiscal_code:
                total += 8
            if evidence.hospital_patient_id:
                total += 8
            if evidence.name and evidence.birth_date:
                total += 4
            elif evidence.name or evidence.birth_date:
                total += 1
            return total

        candidate_score = score(candidate)
        current_score = score(current)
        if candidate_score > current_score:
            return candidate
        if candidate_score < current_score:
            return current
        return candidate if candidate.confidence > current.confidence else current

    @staticmethod
    def _names_compatible(values: set[str]) -> bool:
        """True when every name is a token-subset of the longest one.

        A report whose name wraps onto a second line yields a truncated
        extraction (``GUERRA VILLIAM``) next to the full one (``GUERRA
        VILLIAM SILVESTRO``).  Those describe the same person and are not a
        genuine conflict; only truly disjoint names are discordant.  The
        full name is not a subset of the truncated one, so compatibility is
        judged against the longest token set, not pairwise.
        """
        token_sets = [set(value.split()) for value in values]
        if not token_sets:
            return True
        longest = max(token_sets, key=len)
        return all(tokens <= longest for tokens in token_sets)

    @classmethod
    def _conflicting_fields(cls, documents: list[StagedDocument]) -> list[str]:
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
                if field_name == "name" and cls._names_compatible(values):
                    continue
                conflicts.append(field_name)
        return conflicts

    # --- best-effort auto-assignment of unresolved groups ------------------

    def _auto_assign_unresolved(self, groups: list[RoutingGroup]) -> None:
        """Absorb needs_review groups whose text names a batch candidate.

        A batch often resolves a few solid identities (new workspaces to
        create, or existing patients).  A needs_review group that the
        coordinate-aware extractor could not anchor may still carry the
        candidate's surname or birth date in its text.  When every document
        of the group votes for the same candidate, and no candidate ties,
        the group is merged into it; whatever remains is genuinely
        unassigned and is surfaced to the user instead of being dropped.
        """
        candidates = [
            (index, group)
            for index, group in enumerate(groups)
            if (group.create_new or group.patient_id) and group.evidence
        ]
        if not candidates:
            return
        signals = [
            (index, self._candidate_surname(group),
             self._candidate_birth(group))
            for index, group in candidates
        ]

        absorbed: dict[int, list[StagedDocument]] = defaultdict(list)
        survivors: list[RoutingGroup] = []
        for group in groups:
            if group.patient_id or group.create_new or not group.needs_review:
                survivors.append(group)
                continue
            target = self._vote_target(group, signals)
            if target is None:
                survivors.append(group)
            else:
                absorbed[target].extend(group.documents)
        for index, extra in absorbed.items():
            groups[index].documents.extend(extra)
        groups[:] = survivors

    @staticmethod
    def _candidate_surname(group: RoutingGroup) -> str | None:
        """The surname token of a candidate identity, if trustworthy.

        ``SURNAME, FIRSTNAME`` headers (``GUERRA, VILLIAM SILVESTRO``) name
        the surname before the comma; otherwise the last token is kept, as
        in the plain ``NOME COGNOME`` format.
        """
        evidence = group.evidence
        if not evidence or not evidence.name or not evidence.name.normalized:
            return None
        raw = evidence.name.value or ""
        if "," in raw:
            before = raw.split(",")[0].strip()
            tokens = [token for token in before.split() if len(token) >= 4]
            if tokens:
                return tokens[-1]
        tokens = [
            token for token in evidence.name.normalized.split()
            if len(token) >= 4
        ]
        return tokens[-1] if tokens else None

    @staticmethod
    def _candidate_birth(group: RoutingGroup) -> str | None:
        evidence = group.evidence
        if not evidence or not evidence.birth_date \
                or not evidence.birth_date.normalized:
            return None
        return evidence.birth_date.normalized

    def _vote_target(self, group: RoutingGroup, signals) -> int | None:
        """Index of the candidate every document in the group votes for."""
        targets: list[int] = []
        for document in group.documents:
            target = self._document_target(document, signals)
            if target is None:
                return None
            targets.append(target)
        if targets and len(set(targets)) == 1:
            return targets[0]
        return None

    def _document_target(self, document: StagedDocument, signals) -> int | None:
        """Best candidate for one document, or None when ambiguous/weak."""
        raw_text = _extract_document_text(document.original_path)
        if not raw_text:
            return None
        words = set(normalize_text(raw_text).split())

        best_score = 0
        best_index = None
        ambiguous = False
        for index, surname, birth in signals:
            score = 0
            if surname and surname in words:
                score += 2
            if birth and _birth_date_in_text(raw_text, birth):
                score += 3
            if score == 0:
                continue
            if score > best_score:
                best_score = score
                best_index = index
                ambiguous = False
            elif score == best_score:
                ambiguous = True
        if ambiguous or best_score < 2:
            return None
        return best_index
