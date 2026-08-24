"""Persistence and evaluation services for blinded clinical gold sets."""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Iterable
import uuid

from .engine import DatabaseEngine
from ..evaluation.metrics import evaluate_patient_events
from ..evaluation.runner import EvaluationRunner
from ..models.clinical_registry import (
    ASSERTION_TYPES,
    CERTAINTY_LEVELS,
    DATE_PRECISIONS,
    EVIDENCE_RELATIONS,
    EVENT_CATEGORIES,
    EVENT_STATUSES,
    SIGNIFICANCE_LEVELS,
)
from ..models.gold_set import (
    GOLD_REVIEWER_SLOTS,
    GOLD_SPLITS,
    GoldAnnotation,
    GoldAtomicAnnotation,
    GoldSetCase,
)
from ..models.clinical_pipeline import ATOMIC_FACT_TYPES, EVIDENCE_DISPOSITIONS


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _loads(value, fallback):
    try:
        return json.loads(value) if value else fallback
    except (TypeError, ValueError):
        return fallback


def manual_evidence_id(
    document_id: str,
    page: int | None,
    quote: str,
    *,
    category: str = "",
    entity: str = "",
    date: str = "",
) -> str:
    """Deterministic identifier for a clinician-selected source passage."""
    canonical = _json({
        "document_id": str(document_id),
        "page": int(page) if page else None,
        "quote": " ".join(str(quote).casefold().split()),
        "category": str(category).casefold(),
        "entity": str(entity).casefold(),
        "date": str(date),
    })
    return "GEV_" + uuid.uuid5(uuid.NAMESPACE_URL, canonical).hex


class GoldSetRepository:
    """Manage independent annotations, adjudication and locked exports."""

    def __init__(self, db: DatabaseEngine, registry_repo=None, audit_repo=None):
        self.db = db
        self.registry_repo = registry_repo
        self.audit = audit_repo

    def ensure_case(self, patient_id: str) -> GoldSetCase:
        now = _now()
        with self.db:
            self.db.execute(
                """INSERT OR IGNORE INTO gold_set_cases
                   (patient_id, included, split, status, reviewer_a_id,
                    reviewer_b_id, adjudicator_id, reviewer_a_status,
                    reviewer_b_status, notes, locked_at, created_at, updated_at)
                   VALUES (?, 0, 'pilot', 'draft', '', '', '', 'draft',
                           'draft', '', NULL, ?, ?)""",
                (patient_id, now, now),
            )
        return self.get_case(patient_id)

    def get_case(self, patient_id: str) -> GoldSetCase | None:
        row = self.db.execute(
            "SELECT * FROM gold_set_cases WHERE patient_id=?", (patient_id,)
        ).fetchone()
        return self._row_to_case(row) if row else None

    def configure_case(
        self,
        patient_id: str,
        *,
        included: bool,
        split: str,
        reviewer_a_id: str,
        reviewer_b_id: str,
        adjudicator_id: str,
        notes: str = "",
    ) -> GoldSetCase:
        case = self.ensure_case(patient_id)
        if case.status == "locked":
            raise ValueError("Il gold set è bloccato e non può essere riconfigurato")
        if split not in GOLD_SPLITS:
            raise ValueError(f"Split gold set non valido: {split}")
        assigned = [
            value.strip() for value in (
                reviewer_a_id, reviewer_b_id, adjudicator_id
            ) if value.strip()
        ]
        if included and len(assigned) != 3:
            raise ValueError(
                "Per includere il caso sono obbligatori revisore A, "
                "revisore B e adjudicatore"
            )
        if len(assigned) != len({value.casefold() for value in assigned}):
            raise ValueError(
                "Revisori A, B e adjudicatore devono avere identificativi distinti"
            )
        if case.reviewer_a_status == "submitted" and (
            reviewer_a_id.strip() != case.reviewer_a_id
        ):
            raise ValueError("L'assegnazione del revisore A è già stata consegnata")
        if case.reviewer_b_status == "submitted" and (
            reviewer_b_id.strip() != case.reviewer_b_id
        ):
            raise ValueError("L'assegnazione del revisore B è già stata consegnata")
        status = "excluded"
        if included:
            status = (
                "adjudication"
                if case.reviewer_a_status == "submitted"
                and case.reviewer_b_status == "submitted"
                else "annotating"
            )
        with self.db:
            self.db.execute(
                """UPDATE gold_set_cases SET included=?, split=?, status=?,
                   reviewer_a_id=?, reviewer_b_id=?, adjudicator_id=?, notes=?,
                   updated_at=? WHERE patient_id=?""",
                (
                    1 if included else 0, split, status,
                    reviewer_a_id.strip(), reviewer_b_id.strip(),
                    adjudicator_id.strip(), notes.strip(), _now(), patient_id,
                ),
            )
        self._audit(
            patient_id, "gold_case_configured", "gold_set_case", patient_id,
            {"included": included, "split": split},
            actor_id=adjudicator_id or "local_user", actor_role="coordinator",
        )
        return self.get_case(patient_id)

    def list_cases(
        self, *, included_only: bool = False, locked_only: bool = False
    ) -> list[GoldSetCase]:
        clauses = []
        if included_only:
            clauses.append("included=1")
        if locked_only:
            clauses.append("status='locked'")
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.db.execute(
            "SELECT * FROM gold_set_cases" + where
            + " ORDER BY split, patient_id"
        ).fetchall()
        return [self._row_to_case(row) for row in rows]

    def list_annotations(
        self, patient_id: str, reviewer_slot: str
    ) -> list[GoldAnnotation]:
        self._validate_slot(reviewer_slot)
        rows = self.db.execute(
            """SELECT * FROM gold_annotations
               WHERE patient_id=? AND reviewer_slot=?
               ORDER BY COALESCE(first_evidence_date, '9999'),
                        category, canonical_entity, annotation_id""",
            (patient_id, reviewer_slot),
        ).fetchall()
        return [self._row_to_annotation(row) for row in rows]

    def get_annotation(self, annotation_id: str) -> GoldAnnotation | None:
        row = self.db.execute(
            "SELECT * FROM gold_annotations WHERE annotation_id=?",
            (annotation_id,),
        ).fetchone()
        return self._row_to_annotation(row) if row else None

    def list_atomic_annotations(
        self, patient_id: str, reviewer_slot: str
    ) -> list[GoldAtomicAnnotation]:
        self._validate_slot(reviewer_slot)
        rows = self.db.execute(
            """SELECT * FROM gold_atomic_annotations
               WHERE patient_id=? AND reviewer_slot=?
               ORDER BY document_id, source_page, annotation_id""",
            (patient_id, reviewer_slot),
        ).fetchall()
        return [self._row_to_atomic_annotation(row) for row in rows]

    def save_atomic_annotation(
        self, annotation: GoldAtomicAnnotation
    ) -> GoldAtomicAnnotation:
        self._validate_slot(annotation.reviewer_slot)
        case = self.ensure_case(annotation.patient_id)
        self._assert_editable(case, annotation.reviewer_slot)
        self._assert_assigned_reviewer(case, annotation)
        if annotation.annotation_kind not in {
            "evidence", "exclusion", "duplicate", "invalid"
        }:
            raise ValueError("Tipo di annotazione atomica non valido")
        if annotation.disposition not in EVIDENCE_DISPOSITIONS:
            raise ValueError("Disposizione atomica non valida")
        if annotation.fact_type and annotation.fact_type not in ATOMIC_FACT_TYPES:
            raise ValueError("Tipo di fatto atomico non valido")
        if not annotation.source_text.strip():
            raise ValueError("Il passaggio sorgente è obbligatorio")
        if annotation.annotation_kind == "evidence" and (
            not annotation.fact_type or not str(annotation.concept_original or "").strip()
        ):
            raise ValueError("Tipo di fatto e concetto sono obbligatori")
        if annotation.annotation_kind in {"exclusion", "invalid"} and not (
            annotation.exclusion_reason or ""
        ).strip():
            raise ValueError("Il motivo di esclusione è obbligatorio")
        if annotation.annotation_kind == "duplicate":
            duplicate_id = str(
                annotation.duplicate_of_annotation_id or ""
            ).strip()
            if not duplicate_id or duplicate_id == annotation.annotation_id:
                raise ValueError(
                    "Una duplicazione deve indicare un'altra annotazione"
                )
            duplicate = self.db.execute(
                """SELECT patient_id, reviewer_slot
                   FROM gold_atomic_annotations WHERE annotation_id=?""",
                (duplicate_id,),
            ).fetchone()
            if duplicate is None:
                raise ValueError("Annotazione atomica originale non trovata")
            if (
                duplicate["patient_id"] != annotation.patient_id
                or duplicate["reviewer_slot"] != annotation.reviewer_slot
            ):
                raise ValueError(
                    "Il duplicato deve riferirsi allo stesso caso e revisore"
                )
        now = _now()
        annotation.updated_at = now
        with self.db:
            self.db.execute(
                """INSERT INTO gold_atomic_annotations
                   (annotation_id, patient_id, document_id, reviewer_slot,
                    reviewer_id, annotation_kind, fact_type, concept_original,
                    canonical_label, observation_date, date_precision, polarity,
                    disposition, exclusion_reason, source_page, bbox_json,
                    sentence_refs_json, source_text, value_json,
                    duplicate_of_annotation_id, status, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                           ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(annotation_id) DO UPDATE SET
                     fact_type=excluded.fact_type,
                     concept_original=excluded.concept_original,
                     canonical_label=excluded.canonical_label,
                     observation_date=excluded.observation_date,
                     date_precision=excluded.date_precision,
                     polarity=excluded.polarity,
                     disposition=excluded.disposition,
                     exclusion_reason=excluded.exclusion_reason,
                     source_page=excluded.source_page,
                     bbox_json=excluded.bbox_json,
                     sentence_refs_json=excluded.sentence_refs_json,
                     source_text=excluded.source_text,
                     value_json=excluded.value_json,
                     duplicate_of_annotation_id=excluded.duplicate_of_annotation_id,
                     status=excluded.status,
                     updated_at=excluded.updated_at""",
                (
                    annotation.annotation_id, annotation.patient_id,
                    annotation.document_id, annotation.reviewer_slot,
                    annotation.reviewer_id.strip(), annotation.annotation_kind,
                    annotation.fact_type, annotation.concept_original,
                    annotation.canonical_label, annotation.observation_date,
                    annotation.date_precision, annotation.polarity,
                    annotation.disposition, annotation.exclusion_reason,
                    annotation.source_page,
                    _json(annotation.bbox) if annotation.bbox else None,
                    _json(annotation.sentence_refs), annotation.source_text.strip(),
                    _json(annotation.value),
                    annotation.duplicate_of_annotation_id, annotation.status,
                    annotation.created_at, annotation.updated_at,
                ),
            )
        self._audit(
            annotation.patient_id, "gold_atomic_annotation_saved",
            "gold_atomic_annotation", annotation.annotation_id,
            {
                "reviewer_slot": annotation.reviewer_slot,
                "annotation_kind": annotation.annotation_kind,
                "fact_type": annotation.fact_type,
            },
            actor_id=annotation.reviewer_id,
            actor_role=annotation.reviewer_slot,
        )
        return annotation

    def delete_atomic_annotation(
        self, annotation_id: str, *, actor_id: str = "local_user"
    ) -> None:
        row = self.db.execute(
            "SELECT * FROM gold_atomic_annotations WHERE annotation_id=?",
            (annotation_id,),
        ).fetchone()
        if row is None:
            return
        annotation = self._row_to_atomic_annotation(row)
        case = self.ensure_case(annotation.patient_id)
        self._assert_editable(case, annotation.reviewer_slot)
        with self.db:
            self.db.execute(
                "DELETE FROM gold_atomic_annotations WHERE annotation_id=?",
                (annotation_id,),
            )
        self._audit(
            annotation.patient_id, "gold_atomic_annotation_deleted",
            "gold_atomic_annotation", annotation_id, {},
            actor_id=actor_id, actor_role=annotation.reviewer_slot,
        )

    def save_annotation(self, annotation: GoldAnnotation) -> GoldAnnotation:
        annotation.reviewer_id = annotation.reviewer_id.strip()
        annotation.evidence_ids = list(dict.fromkeys(
            str(reference.get("evidence_id") or "")
            for reference in annotation.source_refs
            if reference.get("evidence_id")
        ))
        self._validate_annotation(annotation)
        case = self.ensure_case(annotation.patient_id)
        self._assert_editable(case, annotation.reviewer_slot)
        self._assert_assigned_reviewer(case, annotation)
        existing = self.get_annotation(annotation.annotation_id)
        if existing and (
            existing.patient_id != annotation.patient_id
            or existing.reviewer_slot != annotation.reviewer_slot
        ):
            raise ValueError(
                "Un'annotazione esistente non può cambiare paziente o ruolo"
            )
        now = _now()
        if not annotation.created_at:
            annotation.created_at = now
        annotation.updated_at = now
        with self.db:
            self.db.execute(
                """INSERT INTO gold_annotations
                   (annotation_id, patient_id, reviewer_slot, reviewer_id,
                    category, canonical_entity, summary_short, summary_detail,
                    first_evidence_date, first_documented_date, date_end,
                    date_precision, status, certainty, assertion,
                    anatomical_site, laterality, severity, significance,
                    episode_key, recurrence_index, evidence_ids_json,
                    source_refs_json, structured_data_json,
                    source_annotation_ids_json, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                           ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(annotation_id) DO UPDATE SET
                     reviewer_id=excluded.reviewer_id,
                     category=excluded.category,
                     canonical_entity=excluded.canonical_entity,
                     summary_short=excluded.summary_short,
                     summary_detail=excluded.summary_detail,
                     first_evidence_date=excluded.first_evidence_date,
                     first_documented_date=excluded.first_documented_date,
                     date_end=excluded.date_end,
                     date_precision=excluded.date_precision,
                     status=excluded.status,
                     certainty=excluded.certainty,
                     assertion=excluded.assertion,
                     anatomical_site=excluded.anatomical_site,
                     laterality=excluded.laterality,
                     severity=excluded.severity,
                     significance=excluded.significance,
                     episode_key=excluded.episode_key,
                     recurrence_index=excluded.recurrence_index,
                     evidence_ids_json=excluded.evidence_ids_json,
                     source_refs_json=excluded.source_refs_json,
                     structured_data_json=excluded.structured_data_json,
                     source_annotation_ids_json=excluded.source_annotation_ids_json,
                     updated_at=excluded.updated_at""",
                (
                    annotation.annotation_id, annotation.patient_id,
                    annotation.reviewer_slot, annotation.reviewer_id.strip(),
                    annotation.category, annotation.canonical_entity.strip(),
                    annotation.summary_short.strip(),
                    annotation.summary_detail.strip(),
                    annotation.first_evidence_date,
                    annotation.first_documented_date, annotation.date_end,
                    annotation.date_precision, annotation.status,
                    annotation.certainty, annotation.assertion,
                    annotation.anatomical_site, annotation.laterality,
                    annotation.severity, annotation.significance,
                    annotation.episode_key, max(1, annotation.recurrence_index),
                    _json(annotation.evidence_ids),
                    _json(annotation.source_refs),
                    _json(annotation.structured_data),
                    _json(annotation.source_annotation_ids),
                    annotation.created_at, annotation.updated_at,
                ),
            )
            # Keep a normalized citation index in addition to the JSON
            # snapshot.  The foreign key prevents a cited source document
            # from being deleted or moved underneath the gold standard.
            self.db.execute(
                "DELETE FROM gold_annotation_sources WHERE annotation_id=?",
                (annotation.annotation_id,),
            )
            for reference in annotation.source_refs:
                evidence_id = str(reference["evidence_id"])
                source_id = "GSRC_" + uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{annotation.annotation_id}:{evidence_id}",
                ).hex
                self.db.execute(
                    """INSERT INTO gold_annotation_sources
                       (source_id, annotation_id, patient_id, evidence_id,
                        document_id, source_page, source_text, relation,
                        created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        source_id, annotation.annotation_id,
                        annotation.patient_id, evidence_id,
                        str(reference["document_id"]), reference.get("page"),
                        str(reference["quote"]),
                        str(reference.get("relation") or "supports"), now,
                    ),
                )
        self._audit(
            annotation.patient_id, "gold_annotation_saved",
            "gold_annotation", annotation.annotation_id,
            {
                "reviewer_slot": annotation.reviewer_slot,
                "category": annotation.category,
                "source_count": len(annotation.source_refs),
            },
            actor_id=annotation.reviewer_id, actor_role=annotation.reviewer_slot,
        )
        return annotation

    def delete_annotation(
        self, annotation_id: str, *, actor_id: str = "local_user"
    ) -> None:
        annotation = self.get_annotation(annotation_id)
        if annotation is None:
            return
        case = self.ensure_case(annotation.patient_id)
        self._assert_editable(case, annotation.reviewer_slot)
        with self.db:
            self.db.execute(
                "DELETE FROM gold_annotations WHERE annotation_id=?",
                (annotation_id,),
            )
        self._audit(
            annotation.patient_id, "gold_annotation_deleted",
            "gold_annotation", annotation_id,
            {"reviewer_slot": annotation.reviewer_slot},
            actor_id=actor_id, actor_role=annotation.reviewer_slot,
        )

    def submit_reviewer(
        self, patient_id: str, reviewer_slot: str, reviewer_id: str
    ) -> GoldSetCase:
        if reviewer_slot not in {"reviewer_a", "reviewer_b"}:
            raise ValueError("Solo i revisori A e B possono consegnare")
        case = self.ensure_case(patient_id)
        if not case.included:
            raise ValueError("Il paziente non è incluso nel gold set")
        self._assert_editable(case, reviewer_slot)
        assigned = getattr(case, f"{reviewer_slot}_id")
        reviewer_id = reviewer_id.strip()
        if not reviewer_id:
            raise ValueError("Identificativo del revisore obbligatorio")
        if assigned and assigned != reviewer_id:
            raise ValueError("Il revisore non corrisponde all'assegnazione")
        if not self.list_annotations(patient_id, reviewer_slot):
            raise ValueError("Inserire almeno un evento prima della consegna")
        status_column = f"{reviewer_slot}_status"
        id_column = f"{reviewer_slot}_id"
        other_status = (
            case.reviewer_b_status
            if reviewer_slot == "reviewer_a" else case.reviewer_a_status
        )
        case_status = "adjudication" if other_status == "submitted" else "annotating"
        with self.db:
            self.db.execute(
                f"""UPDATE gold_set_cases SET {status_column}='submitted',
                    {id_column}=?, status=?, updated_at=? WHERE patient_id=?""",
                (reviewer_id, case_status, _now(), patient_id),
            )
        self._audit(
            patient_id, "gold_annotation_submitted", "gold_set_case", patient_id,
            {"reviewer_slot": reviewer_slot},
            actor_id=reviewer_id, actor_role=reviewer_slot,
        )
        return self.get_case(patient_id)

    def reopen_reviewer(
        self, patient_id: str, reviewer_slot: str, adjudicator_id: str
    ) -> GoldSetCase:
        if reviewer_slot not in {"reviewer_a", "reviewer_b"}:
            raise ValueError(reviewer_slot)
        case = self.ensure_case(patient_id)
        if case.status == "locked":
            raise ValueError("Il gold set bloccato non può essere riaperto")
        self._assert_adjudicator(case, adjudicator_id)
        with self.db:
            # Any final event may depend on the annotation being reopened.
            # Invalidate the whole adjudication layer while preserving A/B.
            self.db.execute(
                "DELETE FROM gold_adjudication_decisions WHERE patient_id=?",
                (patient_id,),
            )
            self.db.execute(
                """DELETE FROM gold_annotations
                   WHERE patient_id=? AND reviewer_slot='adjudicated'""",
                (patient_id,),
            )
            self.db.execute(
                f"""UPDATE gold_set_cases SET {reviewer_slot}_status='draft',
                    status='annotating', updated_at=? WHERE patient_id=?""",
                (_now(), patient_id),
            )
        self._audit(
            patient_id, "gold_annotation_reopened", "gold_set_case", patient_id,
            {"reviewer_slot": reviewer_slot},
            actor_id=adjudicator_id, actor_role="adjudicator",
        )
        return self.get_case(patient_id)

    def copy_to_adjudicated(
        self, annotation_id: str, adjudicator_id: str,
        *, related_annotation_ids: Iterable[str] = (),
        rationale: str = "",
    ) -> GoldAnnotation:
        source = self.get_annotation(annotation_id)
        if source is None or source.reviewer_slot not in {
            "reviewer_a", "reviewer_b"
        }:
            raise ValueError("Annotazione sorgente non valida")
        case = self.ensure_case(source.patient_id)
        if not self.can_adjudicate(case):
            raise ValueError("Entrambe le annotazioni devono essere consegnate")
        source_ids = list(dict.fromkeys([
            source.annotation_id, *[str(value) for value in related_annotation_ids]
        ]))
        for source_id in source_ids:
            related = self.get_annotation(source_id)
            if (
                related is None or related.patient_id != source.patient_id
                or related.reviewer_slot not in {"reviewer_a", "reviewer_b"}
            ):
                raise ValueError("Annotazione correlata non valida")
        resolved_adjudicator = adjudicator_id.strip() or case.adjudicator_id
        self._assert_adjudicator(case, resolved_adjudicator)
        copied = replace(
            source,
            annotation_id=f"GOLD_{uuid.uuid4().hex}",
            reviewer_slot="adjudicated",
            reviewer_id=resolved_adjudicator,
            source_annotation_ids=source_ids,
            created_at=_now(),
            updated_at=_now(),
        )
        final = self.save_annotation(copied)
        for source_id in source_ids:
            self.set_adjudication_decision(
                source.patient_id, source_id,
                decision="included", adjudicator_id=resolved_adjudicator,
                final_annotation_id=final.annotation_id,
                rationale=rationale or "Incluso nell'evento finale",
            )
        return final

    def exclude_from_gold(
        self,
        patient_id: str,
        source_annotation_ids: Iterable[str],
        adjudicator_id: str,
        rationale: str,
    ) -> None:
        if not rationale.strip():
            raise ValueError("La motivazione dell'esclusione è obbligatoria")
        case = self.ensure_case(patient_id)
        if not self.can_adjudicate(case) or case.status == "locked":
            raise ValueError("Adjudication non disponibile")
        self._assert_adjudicator(case, adjudicator_id)
        source_ids = list(dict.fromkeys(str(value) for value in source_annotation_ids))
        if not source_ids:
            raise ValueError("Nessuna annotazione selezionata")
        for source_id in source_ids:
            annotation = self.get_annotation(source_id)
            if (
                annotation is None or annotation.patient_id != patient_id
                or annotation.reviewer_slot not in {"reviewer_a", "reviewer_b"}
            ):
                raise ValueError("Annotazione sorgente non valida")
            self.set_adjudication_decision(
                patient_id, source_id, decision="excluded",
                adjudicator_id=adjudicator_id, rationale=rationale,
            )

    def set_adjudication_decision(
        self,
        patient_id: str,
        source_annotation_id: str,
        *,
        decision: str,
        adjudicator_id: str,
        final_annotation_id: str | None = None,
        rationale: str = "",
    ) -> None:
        if decision not in {"included", "excluded"}:
            raise ValueError("Decisione di adjudication non valida")
        if decision == "included" and not final_annotation_id:
            raise ValueError("Una decisione inclusa richiede l'evento finale")
        case = self.ensure_case(patient_id)
        if case.status == "locked" or not self.can_adjudicate(case):
            raise ValueError("Adjudication non modificabile")
        self._assert_adjudicator(case, adjudicator_id)
        now = _now()
        decision_id = "GADJ_" + uuid.uuid5(
            uuid.NAMESPACE_URL, f"{patient_id}:{source_annotation_id}"
        ).hex
        with self.db:
            self.db.execute(
                """INSERT INTO gold_adjudication_decisions
                   (decision_id, patient_id, source_annotation_id,
                    final_annotation_id, decision, rationale, adjudicator_id,
                    created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(source_annotation_id) DO UPDATE SET
                     final_annotation_id=excluded.final_annotation_id,
                     decision=excluded.decision,
                     rationale=excluded.rationale,
                     adjudicator_id=excluded.adjudicator_id,
                     updated_at=excluded.updated_at""",
                (
                    decision_id, patient_id, source_annotation_id,
                    final_annotation_id, decision, rationale.strip(),
                    adjudicator_id.strip(), now, now,
                ),
            )
        self._audit(
            patient_id, "gold_adjudication_decided", "gold_annotation",
            source_annotation_id,
            {"decision": decision, "final_annotation_id": final_annotation_id},
            actor_id=adjudicator_id, actor_role="adjudicator",
        )

    def list_adjudication_decisions(self, patient_id: str) -> list[dict]:
        rows = self.db.execute(
            """SELECT * FROM gold_adjudication_decisions
               WHERE patient_id=? ORDER BY created_at, decision_id""",
            (patient_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def lock_case(self, patient_id: str, adjudicator_id: str) -> GoldSetCase:
        case = self.ensure_case(patient_id)
        if not self.can_adjudicate(case):
            raise ValueError("Servono le consegne indipendenti di A e B")
        self._assert_adjudicator(case, adjudicator_id)
        final = self.list_annotations(patient_id, "adjudicated")
        if not final:
            raise ValueError("Nessun evento adjudicato")
        incomplete = [
            item.annotation_id for item in final
            if not item.source_refs or not item.evidence_ids
        ]
        if incomplete:
            raise ValueError(
                "Ogni evento finale deve avere almeno una fonte: "
                + ", ".join(incomplete)
            )
        source_ids = {
            item.annotation_id
            for slot in ("reviewer_a", "reviewer_b")
            for item in self.list_annotations(patient_id, slot)
        }
        decisions = self.list_adjudication_decisions(patient_id)
        decided_ids = {item["source_annotation_id"] for item in decisions}
        missing = sorted(source_ids - decided_ids)
        invalid_included = [
            item["source_annotation_id"] for item in decisions
            if item["decision"] == "included" and not item["final_annotation_id"]
        ]
        if missing or invalid_included:
            raise ValueError(
                "Adjudication incompleta; annotazioni senza decisione: "
                + ", ".join(missing + invalid_included)
            )
        locked_at = _now()
        with self.db:
            self.db.execute(
                """UPDATE gold_set_cases SET status='locked', locked_at=?,
                   adjudicator_id=?, updated_at=? WHERE patient_id=?""",
                (locked_at, adjudicator_id.strip(), locked_at, patient_id),
            )
        digest = hashlib.sha256(
            _json([item.to_evaluation_dict() for item in final]).encode("utf-8")
        ).hexdigest()
        self._audit(
            patient_id, "gold_case_locked", "gold_set_case", patient_id,
            {"event_count": len(final), "gold_digest": digest},
            actor_id=adjudicator_id, actor_role="adjudicator",
        )
        return self.get_case(patient_id)

    @staticmethod
    def can_adjudicate(case: GoldSetCase) -> bool:
        return (
            case.included
            and case.reviewer_a_status == "submitted"
            and case.reviewer_b_status == "submitted"
        )

    def reviewer_agreement(self, patient_id: str) -> dict:
        reviewer_a = [
            item.to_evaluation_dict()
            for item in self.list_annotations(patient_id, "reviewer_a")
        ]
        reviewer_b = [
            item.to_evaluation_dict()
            for item in self.list_annotations(patient_id, "reviewer_b")
        ]
        return evaluate_patient_events(reviewer_a, reviewer_b)

    def evaluation_record(self, patient_id: str) -> dict:
        case = self.ensure_case(patient_id)
        gold = [
            self._gold_event_for_evaluation(item)
            for item in self.list_annotations(patient_id, "adjudicated")
        ]
        predicted = []
        if self.registry_repo:
            for event in self.registry_repo.get_events(patient_id):
                detail = self.registry_repo.get_event_detail(event.event_id) or {}
                item = asdict(event)
                item["evidence_ids"] = [
                    str(source.get("evidence_id"))
                    for source in detail.get("evidence", [])
                    if source.get("evidence_id")
                ]
                predicted.append(item)
        return {
            "patient_id": patient_id,
            "gold_events": gold,
            "predicted_events": predicted,
            "gold_metadata": {
                "split": case.split,
                "status": case.status,
                "reviewer_a_id": case.reviewer_a_id,
                "reviewer_b_id": case.reviewer_b_id,
                "adjudicator_id": case.adjudicator_id,
                "locked_at": case.locked_at,
            },
        }

    def _gold_event_for_evaluation(
        self, annotation: GoldAnnotation
    ) -> dict:
        """Map manual passages to atomic IDs only when the source agrees.

        Manual ``GEV_*`` identifiers remain the expected citation when the
        extractor missed the passage.  When an atomic evidence quote from the
        same document matches, its stable ``EVD_*`` ID is used so citation
        completeness measures the registry link rather than an ID namespace.
        """
        event = annotation.to_evaluation_dict()
        mapped_ids: list[str] = []
        for reference in annotation.source_refs:
            document_id = str(reference.get("document_id") or "")
            quote = " ".join(str(reference.get("quote") or "").casefold().split())
            matches = []
            if document_id and quote:
                rows = self.db.execute(
                    """SELECT evidence_id, source_text FROM clinical_evidence
                       WHERE patient_id=? AND document_id=?""",
                    (annotation.patient_id, document_id),
                ).fetchall()
                for row in rows:
                    source = " ".join(
                        str(row["source_text"] or "").casefold().split()
                    )
                    if source and (
                        source == quote or source in quote or quote in source
                    ):
                        matches.append(str(row["evidence_id"]))
            mapped_ids.extend(
                matches or [str(reference.get("evidence_id") or "")]
            )
        event["evidence_ids"] = list(dict.fromkeys(
            value for value in mapped_ids if value
        ))
        event["manual_evidence_ids"] = list(annotation.evidence_ids)
        return event

    def evaluate_patient(self, patient_id: str) -> dict:
        record = self.evaluation_record(patient_id)
        return evaluate_patient_events(
            record["gold_events"], record["predicted_events"]
        )

    def evaluate_project(self, *, locked_only: bool = True) -> dict:
        cases = self.list_cases(included_only=True, locked_only=locked_only)
        return EvaluationRunner().evaluate_records(
            self.evaluation_record(case.patient_id) for case in cases
        )

    def export_jsonl(
        self,
        path: str | Path,
        *,
        patient_ids: Iterable[str] | None = None,
        locked_only: bool = True,
    ) -> int:
        selected = set(patient_ids or [])
        cases = self.list_cases(included_only=True, locked_only=locked_only)
        if selected:
            cases = [case for case in cases if case.patient_id in selected]
        destination = Path(path)
        with destination.open("w", encoding="utf-8") as stream:
            for case in cases:
                stream.write(
                    json.dumps(
                        self.evaluation_record(case.patient_id),
                        ensure_ascii=False, sort_keys=True,
                    ) + "\n"
                )
        return len(cases)

    def _assert_editable(self, case: GoldSetCase, reviewer_slot: str) -> None:
        if not case.included:
            raise ValueError("Il paziente non è incluso nel gold set")
        if case.status == "locked":
            raise ValueError("Il gold set è bloccato")
        if reviewer_slot == "reviewer_a" and case.reviewer_a_status == "submitted":
            raise ValueError("L'annotazione del revisore A è già consegnata")
        if reviewer_slot == "reviewer_b" and case.reviewer_b_status == "submitted":
            raise ValueError("L'annotazione del revisore B è già consegnata")
        if reviewer_slot == "adjudicated" and not self.can_adjudicate(case):
            raise ValueError("L'adjudication non è ancora disponibile")

    @staticmethod
    def _validate_slot(reviewer_slot: str) -> None:
        if reviewer_slot not in GOLD_REVIEWER_SLOTS:
            raise ValueError(f"Ruolo revisore non valido: {reviewer_slot}")

    def _validate_annotation(self, annotation: GoldAnnotation) -> None:
        self._validate_slot(annotation.reviewer_slot)
        if annotation.category not in EVENT_CATEGORIES:
            raise ValueError(f"Categoria non valida: {annotation.category}")
        if annotation.date_precision not in DATE_PRECISIONS:
            raise ValueError("Precisione temporale non valida")
        if annotation.status not in EVENT_STATUSES:
            raise ValueError("Stato clinico non valido")
        if annotation.certainty not in CERTAINTY_LEVELS:
            raise ValueError("Certezza non valida")
        if annotation.assertion not in ASSERTION_TYPES:
            raise ValueError("Asserzione non valida")
        if annotation.significance not in SIGNIFICANCE_LEVELS:
            raise ValueError("Significatività clinica non valida")
        if not annotation.canonical_entity.strip() or not annotation.summary_short.strip():
            raise ValueError("Entità e sintesi breve sono obbligatorie")
        if not annotation.reviewer_id.strip():
            raise ValueError("Identificativo del revisore obbligatorio")
        if not annotation.source_refs:
            raise ValueError("Ogni evento annotato richiede almeno una fonte")
        seen_evidence_ids: set[str] = set()
        for reference in annotation.source_refs:
            if not all(reference.get(key) for key in (
                "evidence_id", "document_id", "quote"
            )):
                raise ValueError(
                    "Ogni fonte richiede evidence_id, document_id e passaggio"
                )
            evidence_id = str(reference["evidence_id"])
            if evidence_id in seen_evidence_ids:
                raise ValueError("La stessa evidenza è citata più di una volta")
            seen_evidence_ids.add(evidence_id)
            relation = str(reference.get("relation") or "supports")
            if relation not in EVIDENCE_RELATIONS:
                raise ValueError("Relazione della fonte non valida")
            page = reference.get("page")
            if page is not None and (
                not isinstance(page, int) or isinstance(page, bool) or page < 1
            ):
                raise ValueError("La pagina della fonte deve essere positiva")
            document = self.db.execute(
                "SELECT patient_id FROM documents WHERE id=?",
                (str(reference["document_id"]),),
            ).fetchone()
            if document is None:
                raise ValueError("Documento citato non trovato")
            if document["patient_id"] != annotation.patient_id:
                raise ValueError(
                    "La fonte citata appartiene a un altro paziente"
                )

    def _assert_assigned_reviewer(
        self, case: GoldSetCase, annotation: GoldAnnotation
    ) -> None:
        if annotation.reviewer_slot == "reviewer_a":
            assigned = case.reviewer_a_id
        elif annotation.reviewer_slot == "reviewer_b":
            assigned = case.reviewer_b_id
        else:
            self._assert_adjudicator(case, annotation.reviewer_id)
            return
        if not assigned or annotation.reviewer_id != assigned:
            raise ValueError("Il revisore non corrisponde all'assegnazione")

    @staticmethod
    def _assert_adjudicator(case: GoldSetCase, adjudicator_id: str) -> None:
        adjudicator_id = str(adjudicator_id or "").strip()
        if not adjudicator_id:
            raise ValueError("Identificativo dell'adjudicatore obbligatorio")
        if case.adjudicator_id and adjudicator_id != case.adjudicator_id:
            raise ValueError("L'adjudicatore non corrisponde all'assegnazione")

    def _audit(
        self, patient_id, action, target_type, target_id, details,
        *, actor_id, actor_role,
    ) -> None:
        if self.audit:
            self.audit.log(
                patient_id, action, target_type, target_id, details,
                actor_id=actor_id or "local_user", actor_role=actor_role,
            )

    @staticmethod
    def _row_to_case(row) -> GoldSetCase:
        return GoldSetCase(
            patient_id=row["patient_id"], included=bool(row["included"]),
            split=row["split"], status=row["status"],
            reviewer_a_id=row["reviewer_a_id"],
            reviewer_b_id=row["reviewer_b_id"],
            adjudicator_id=row["adjudicator_id"],
            reviewer_a_status=row["reviewer_a_status"],
            reviewer_b_status=row["reviewer_b_status"], notes=row["notes"],
            locked_at=row["locked_at"], created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _row_to_annotation(row) -> GoldAnnotation:
        return GoldAnnotation(
            annotation_id=row["annotation_id"], patient_id=row["patient_id"],
            reviewer_slot=row["reviewer_slot"], reviewer_id=row["reviewer_id"],
            category=row["category"], canonical_entity=row["canonical_entity"],
            summary_short=row["summary_short"], summary_detail=row["summary_detail"],
            first_evidence_date=row["first_evidence_date"],
            first_documented_date=row["first_documented_date"],
            date_end=row["date_end"], date_precision=row["date_precision"],
            status=row["status"], certainty=row["certainty"],
            assertion=row["assertion"], anatomical_site=row["anatomical_site"],
            laterality=row["laterality"], severity=row["severity"],
            significance=row["significance"], episode_key=row["episode_key"],
            recurrence_index=row["recurrence_index"],
            evidence_ids=_loads(row["evidence_ids_json"], []),
            source_refs=_loads(row["source_refs_json"], []),
            structured_data=_loads(row["structured_data_json"], {}),
            source_annotation_ids=_loads(
                row["source_annotation_ids_json"], []
            ),
            created_at=row["created_at"], updated_at=row["updated_at"],
        )

    @staticmethod
    def _row_to_atomic_annotation(row) -> GoldAtomicAnnotation:
        bbox = _loads(row["bbox_json"], None)
        return GoldAtomicAnnotation(
            annotation_id=row["annotation_id"],
            patient_id=row["patient_id"],
            document_id=row["document_id"],
            reviewer_slot=row["reviewer_slot"],
            reviewer_id=row["reviewer_id"],
            annotation_kind=row["annotation_kind"],
            fact_type=row["fact_type"],
            concept_original=row["concept_original"],
            canonical_label=row["canonical_label"],
            observation_date=row["observation_date"],
            date_precision=row["date_precision"],
            polarity=row["polarity"],
            disposition=row["disposition"],
            exclusion_reason=row["exclusion_reason"],
            source_page=row["source_page"],
            bbox=tuple(bbox) if bbox else None,
            sentence_refs=_loads(row["sentence_refs_json"], []),
            source_text=row["source_text"],
            value=_loads(row["value_json"], {}),
            duplicate_of_annotation_id=row["duplicate_of_annotation_id"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
