"""Reviewed events: extracted occurrences ⊕ concept codes ⊕ reviewer decisions.

Machine occurrences stay as extracted; a reviewer decision is stored apart,
keyed by the source occurrence, with a snapshot of the whole reviewed event.
A new extraction therefore never discards a decision: it is re-attached to
the same occurrence, or shown on its own when that occurrence disappeared.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import uuid

from .document_text import effective_text
from .evidence_utils import content_hash
from .grounded_sources import METHOD
from .snomed_coding import concept_key
from ..models.clinical_evidence import ClinicalEvidence

EDITABLE = ("label", "fact_type", "assertion", "certainty", "subject", "temporality", "state",
            "observed_date", "observed_date_end", "date_precision", "attributes",
            "numeric_value", "unit", "value_text")
SNAPSHOT = EDITABLE + ("document_id", "document_date", "start", "end", "quote", "page", "code")
REVIEWED = ("confirmed", "corrected", "added")


@dataclass
class EffectiveEvent:
    key: str
    patient_id: str
    document_id: str
    document_date: str | None
    label: str
    fact_type: str
    assertion: str = "present"
    certainty: str = "confirmed"
    subject: str = "patient"
    temporality: str | None = None
    state: str | None = None
    observed_date: str | None = None
    observed_date_end: str | None = None
    date_precision: str = "unknown"
    attributes: dict = field(default_factory=dict)
    numeric_value: float | None = None
    unit: str | None = None
    value_text: str | None = None
    start: int | None = None
    end: int | None = None
    quote: str = ""
    page: int | None = None
    # proposed | needs_review | confirmed | corrected | added | rejected
    status: str = "proposed"
    stale: bool = False
    note: str | None = None
    # Occurrence-level SNOMED decision ({"code": ...}; code None = no code).
    code: dict | None = None
    coding: dict = field(default_factory=dict)
    machine: ClinicalEvidence | None = None
    override: dict | None = None
    extraction_method: str = METHOD
    #: identity of the repeated statement this event belongs to, if any
    statement_key: str | None = None

    # Attribute names shared with ClinicalEvidence (concept collection).
    @property
    def normalized_entity(self) -> str:
        return self.label

    @property
    def source_text(self) -> str:
        return self.quote

    @property
    def concept(self) -> tuple[str, str]:
        return concept_key(self.label, self.fact_type)

    @property
    def reviewed(self) -> bool:
        return self.status in REVIEWED or self.status == "rejected"


def occurrence_key(item: ClinicalEvidence) -> str:
    """Identity of a source occurrence, stable across extractions."""
    span = (item.data.get("source_spans") or [{}])[0]
    return content_hash("occurrence-v1", item.document_id, span.get("start"), span.get("end"),
                        *concept_key(item.normalized_entity, item.fact_type))


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def from_machine(item: ClinicalEvidence) -> EffectiveEvent:
    data = item.data or {}
    span = (data.get("source_spans") or [{}])[0]
    provenance = data.get("date_provenance") or {}
    return EffectiveEvent(
        key=occurrence_key(item), patient_id=item.patient_id, document_id=item.document_id,
        document_date=item.document_date, label=item.normalized_entity, fact_type=item.fact_type,
        assertion=item.assertion, certainty=item.certainty, subject=data.get("experiencer", "patient"),
        temporality=item.temporality, state=item.clinical_status, observed_date=item.observed_date,
        observed_date_end=item.observed_date_end, date_precision=item.date_precision,
        attributes=dict(data.get("attributes") or {}), numeric_value=item.numeric_value,
        unit=item.unit, value_text=item.value_text, start=span.get("start"), end=span.get("end"),
        quote=item.source_text or "", page=item.source_page,
        status="needs_review" if item.status == "needs_review" else "proposed",
        note=provenance.get("needs_review") or data.get("value_review"), machine=item,
        statement_key=(data.get("statement_reuse") or {}).get("statement_key"))


class EventReviewService:
    def __init__(self, evidence_repo, overrides, coder=None, *, overlay_repo=None,
                 audit_repo=None, snomed=None):
        self.evidence_repo = evidence_repo
        self.overrides = overrides
        self.coder = coder
        self.overlay_repo = overlay_repo
        self.audit = audit_repo
        self.snomed = snomed

    # ------------------------------------------------------------------ view
    def text(self, patient_id: str, document_id: str) -> str:
        return effective_text(patient_id, document_id, self.overlay_repo)

    def events(self, patient_id: str, *, include_rejected: bool = True) -> list[EffectiveEvent]:
        overrides = self.overrides.by_patient(patient_id)
        texts: dict[str, str] = {}
        result, seen = [], set()
        for item in self.evidence_repo.get_by_patient(patient_id):
            if item.extraction_method != METHOD:
                continue
            event = from_machine(item)
            if event.key in seen:
                continue
            seen.add(event.key)
            override = overrides.get(event.key)
            if override is not None:
                event = self._apply(event, override, texts)
            result.append(event)
        for key, override in overrides.items():
            if key in seen or override["action"] == "reject":
                continue        # a rejected occurrence that is no longer extracted
            event = self._from_snapshot(patient_id, key, override)
            result.append(self._check_source(event, override, texts))
        self._attach_coding(result)
        return result if include_rejected else [event for event in result if event.status != "rejected"]

    def _apply(self, event, override, texts) -> EffectiveEvent:
        event.override = override
        action = override["action"]
        if action == "reject":
            event.status = "rejected"
        elif action == "confirm":
            snapshot = override["fields"]
            changed = [name for name in EDITABLE
                       if name in snapshot and snapshot[name] != getattr(event, name)]
            event.status = "needs_review" if changed else "confirmed"
            if changed:
                event.code = snapshot.get("code")
                event.note = ("Evento modificato da una nuova estrazione dopo la conferma: "
                              + ", ".join(changed))
                return event
        else:
            snapshot = override["fields"]
            for name in SNAPSHOT:
                if name in snapshot:
                    setattr(event, name, snapshot[name])
            event.status = "corrected"
            event = self._check_source(event, override, texts)
        event.code = override["fields"].get("code")
        event.note = override.get("note") or (event.note if action == "confirm" else None)
        return event

    def _from_snapshot(self, patient_id, key, override) -> EffectiveEvent:
        snapshot = override["fields"]
        event = EffectiveEvent(key=key, patient_id=patient_id, document_id=override["document_id"],
                               document_date=snapshot.get("document_date"),
                               label=snapshot.get("label", ""), fact_type=snapshot.get("fact_type", ""))
        for name in SNAPSHOT:
            if name in snapshot:
                setattr(event, name, snapshot[name])
        event.override = override
        event.code = snapshot.get("code")
        event.note = override.get("note")
        event.status = {"add": "added", "confirm": "confirmed"}.get(override["action"], "corrected")
        return event

    def _check_source(self, event, override, texts) -> EffectiveEvent:
        """Keep the reviewed interval valid on the current text, or flag it."""
        if event.document_id not in texts:
            texts[event.document_id] = self.text(event.patient_id, event.document_id)
        text = texts[event.document_id]
        if override.get("text_hash") in (None, text_hash(text)):
            return event
        if event.start is not None and text[event.start:event.end] == event.quote:
            return event
        if event.quote and text.count(event.quote) == 1:
            event.start = text.index(event.quote)
            event.end = event.start + len(event.quote)
            return event
        event.stale = True
        event.note = "Il testo del referto è cambiato: verifica il frammento di questo evento."
        return event

    def _attach_coding(self, events) -> None:
        mappings = self.coder.mappings if self.coder is not None else None
        concepts = mappings.get_many({event.concept for event in events}) if mappings is not None else {}
        for event in events:
            if event.code is not None:
                event.coding = {**event.code, "status": "confirmed", "source": "occurrence"}
            else:
                event.coding = dict(concepts.get(event.concept) or {}, source="concept")

    # --------------------------------------------------------------- actions
    def statement_copies(self, event: EffectiveEvent) -> list[EffectiveEvent]:
        """The other occurrences of the same statement, in this patient."""
        if not event.statement_key:
            return [event]
        siblings = [other for other in self.events(event.patient_id, include_rejected=True)
                    if other.statement_key == event.statement_key]
        return siblings or [event]

    def _fanout(self, event: EffectiveEvent) -> list[EffectiveEvent]:
        """The event itself plus every copy of the same statement."""
        return self.statement_copies(event)

    def confirm(self, event: EffectiveEvent, *, note=None, reviewer=None, fanout=False) -> None:
        for target in (self._fanout(event) if fanout else [event]):
            if target.status == "added":
                self._save(target, "add", note=note, reviewer=reviewer)
                continue
            action = "correct" if target.status == "corrected" else "confirm"
            self._save(target, action, note=note, reviewer=reviewer)

    def correct(self, event: EffectiveEvent, changes: dict, *, note=None, reviewer=None,
                fanout=False) -> EffectiveEvent:
        """Apply field, interval or occurrence-code changes to one event.

        With ``fanout`` the same changes reach every copy of the statement.
        A fragment belongs to one report only, so an interval change never
        propagates: it would point at different words elsewhere.
        """
        unknown = set(changes) - set(EDITABLE) - {"start", "end", "code"}
        if unknown:
            raise ValueError("Campi non modificabili: " + ", ".join(sorted(unknown)))
        if fanout and ({"start", "end"} & set(changes)):
            raise ValueError("Il frammento vale per un solo referto: correggilo su ogni copia")
        updated = self._apply_changes(event, changes)
        self._save(updated, "add" if updated.status == "added" else "correct",
                   note=note, reviewer=reviewer)
        if fanout:
            for sibling in self.statement_copies(event):
                if sibling.key == event.key:
                    continue
                self._apply_changes(sibling, changes)
                self._save(sibling, "add" if sibling.status == "added" else "correct",
                           note=note, reviewer=reviewer)
        return updated

    def _apply_changes(self, event: EffectiveEvent, changes: dict) -> EffectiveEvent:
        for name in EDITABLE:
            if name in changes:
                setattr(event, name, changes[name])
        if "code" in changes:
            event.code = changes["code"]
        if "start" in changes or "end" in changes:
            text = self.text(event.patient_id, event.document_id)
            start, end = int(changes.get("start", event.start)), int(changes.get("end", event.end))
            if not 0 <= start < end <= len(text) or not text[start:end].strip():
                raise ValueError("Intervallo di testo non valido per questo referto")
            event.start, event.end, event.quote = start, end, text[start:end]
            event.stale = False
        if not (event.label or "").strip():
            raise ValueError("L'evento deve avere un'etichetta")
        return event

    def reject(self, event: EffectiveEvent, *, note=None, reviewer=None, fanout=False) -> None:
        for target in (self._fanout(event) if fanout else [event]):
            if target.status == "added":
                self.restore(target)
                continue
            self._save(target, "reject", note=note, reviewer=reviewer)

    def restore(self, event: EffectiveEvent, *, fanout=False) -> None:
        """Drop the decision: back to the extracted occurrence (or delete a manual event)."""
        for target in (self._fanout(event) if fanout else [event]):
            if self.overrides.delete(target.key):
                self._audit(target, "restore")

    def add(self, patient_id: str, document_id: str, start: int, end: int, *, label: str,
            fact_type: str, document_date: str | None = None, note=None, reviewer=None,
            **fields) -> EffectiveEvent:
        text = self.text(patient_id, document_id)
        if not 0 <= start < end <= len(text) or not text[start:end].strip():
            raise ValueError("Seleziona un frammento del referto")
        if not (label or "").strip():
            raise ValueError("L'evento deve avere un'etichetta")
        unknown = set(fields) - set(EDITABLE)
        if unknown:
            raise ValueError("Campi non modificabili: " + ", ".join(sorted(unknown)))
        event = EffectiveEvent(key="manual:" + uuid.uuid4().hex, patient_id=patient_id,
                               document_id=document_id, document_date=document_date,
                               label=label.strip(), fact_type=fact_type, start=start, end=end,
                               quote=text[start:end], status="added", **fields)
        self._save(event, "add", note=note, reviewer=reviewer)
        return event

    def confirm_concept(self, label: str, fact_type: str, code: str | None, *, note=None) -> None:
        """Reviewer coding for every occurrence of the concept (code None = no code)."""
        if self.coder is None or self.coder.mappings is None:
            raise RuntimeError("Archivio delle codifiche non disponibile")
        concept = self.snomed.lookup(code) if (self.snomed is not None and code) else None
        if code and (concept is None or not concept["active"]):
            raise ValueError("Seleziona un concetto SNOMED CT attivo")
        release = None
        if self.snomed is not None:
            meta = self.snomed.metadata()
            release = meta.get("version_uri") or meta.get("release")
        self.coder.mappings.confirm(*concept_key(label, fact_type), label, code=code,
                                    concept=concept, release=release, note=note)
        if code and self.snomed is not None:
            try:
                self.snomed.add_alias(label, code)
            except ValueError:
                pass

    def _save(self, event, action, *, note=None, reviewer=None) -> None:
        snapshot = {name: getattr(event, name) for name in SNAPSHOT}
        text = self.text(event.patient_id, event.document_id)
        self.overrides.save(event.key, patient_id=event.patient_id, document_id=event.document_id,
                            action=action, fields=snapshot,
                            base_evidence_id=event.machine.evidence_id if event.machine else None,
                            text_hash=text_hash(text), note=note, reviewer=reviewer)
        self._audit(event, action)

    def _audit(self, event, action) -> None:
        if self.audit is not None:
            self.audit.log(event.patient_id, f"event_review_{action}", "clinical_event", event.key,
                           {"document_id": event.document_id, "label": event.label,
                            "fact_type": event.fact_type}, actor_id="local_user",
                           actor_role="clinician")


def to_evidence(event: EffectiveEvent) -> ClinicalEvidence:
    """ClinicalEvidence view of a reviewed event, for the FHIR exporter."""
    machine_data = dict(event.machine.data) if event.machine is not None else {}
    if event.status in ("corrected", "added"):
        # A corrected copy is its own fact from now on: it must not fold back
        # into the shared resource while the other copies stay unchanged.
        machine_data.pop("historical_reuse_id", None)
        machine_data.pop("statement_reuse", None)
    data = {**machine_data, "fhir_pipeline": True, "experiencer": event.subject,
            "attributes": dict(event.attributes or {}),
            "source_spans": [{"start": event.start, "end": event.end, "text": event.quote}],
            "date_provenance": machine_data.get("date_provenance") or {"mode": "manual"},
            "review_status": event.status, "review_note": event.note}
    return ClinicalEvidence(
        patient_id=event.patient_id, document_id=event.document_id,
        evidence_id="EVD_" + content_hash("reviewed", event.key),
        category=event.fact_type, fact_type=event.fact_type, normalized_entity=event.label,
        canonical_label=event.label, concept_original=event.label, source_text=event.quote,
        assertion=event.assertion, certainty=event.certainty, temporality=event.temporality,
        clinical_status=event.state, observed_date=event.observed_date,
        observed_date_end=event.observed_date_end, date_precision=event.date_precision or "unknown",
        document_date=event.document_date, numeric_value=event.numeric_value, unit=event.unit,
        value_text=event.value_text, source_page=event.page, extraction_method=METHOD,
        status="needs_review" if (event.stale or event.status == "needs_review") else "proposed",
        data=data,
    )
