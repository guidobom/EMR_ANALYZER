"""SQLite persistence for clinical-pipeline v3 derived objects."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import uuid

from .engine import DatabaseEngine
from ..models.clinical_pipeline import (
    ClinicalHypothesis,
    EventClaim,
    EvidenceRelation,
    EvidenceSourceReference,
    ExcludedEvidence,
    TerminologyMapping,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _loads(value, fallback):
    try:
        return json.loads(value) if value else fallback
    except (TypeError, ValueError, json.JSONDecodeError):
        return fallback


def _source_rank(source: EvidenceSourceReference) -> tuple[int, int, int]:
    """Prefer primary citations and the representation with more geometry."""
    return (
        1 if source.source_role == "primary" else 0,
        len(source.sentence_refs or ()),
        1 if source.bbox else 0,
    )


class ClinicalPipelineRepository:
    """Store provenance, exclusions, graph edges, claims and hypotheses."""

    def __init__(self, db: DatabaseEngine):
        self.db = db

    # ------------------------------------------------------------- sources

    def replace_aggregation_coverage(
        self,
        patient_id: str,
        run_id: str,
        rows,
    ) -> None:
        """Persist one immutable audit snapshot of evidence-event coverage."""
        with self.db:
            self.db.execute(
                "DELETE FROM clinical_aggregation_coverage WHERE run_id=?",
                (run_id,),
            )
            for row in rows:
                self.db.execute(
                    """INSERT INTO clinical_aggregation_coverage
                       (run_id, patient_id, evidence_id, disposition, status,
                        event_ids_json, reason, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        run_id,
                        patient_id,
                        row.evidence_id,
                        row.disposition,
                        row.status,
                        _json(list(row.event_ids)),
                        row.reason,
                        _now(),
                    ),
                )

    def list_aggregation_coverage(
        self,
        patient_id: str,
        *,
        run_id: str | None = None,
        status: str | None = None,
    ) -> list[dict]:
        clauses = ["patient_id=?"]
        params: list[object] = [patient_id]
        if run_id:
            clauses.append("run_id=?")
            params.append(run_id)
        if status:
            clauses.append("status=?")
            params.append(status)
        rows = self.db.execute(
            "SELECT * FROM clinical_aggregation_coverage WHERE "
            + " AND ".join(clauses)
            + " ORDER BY created_at DESC, evidence_id",
            tuple(params),
        ).fetchall()
        return [{
            **dict(row),
            "event_ids": _loads(row["event_ids_json"], []),
        } for row in rows]

    def replace_evidence_sources(
        self, evidence_id: str, sources: list[EvidenceSourceReference]
    ) -> None:
        # A copied passage can reach the canonical atom through more than one
        # deduplication path.  SQLite correctly rejects that duplicate source,
        # but provenance replacement must itself be idempotent.  Prefer the
        # primary/richest representation of each literal occurrence.
        unique_sources: dict[tuple, EvidenceSourceReference] = {}
        for source in sources:
            if source.evidence_id != evidence_id:
                raise ValueError("Fonte associata a una evidenza diversa")
            key = (
                source.document_id, source.source_page, source.passage,
            )
            current = unique_sources.get(key)
            if current is None or _source_rank(source) > _source_rank(current):
                unique_sources[key] = source
        with self.db:
            self.db.execute(
                "DELETE FROM evidence_source_refs WHERE evidence_id=?",
                (evidence_id,),
            )
            for source in unique_sources.values():
                self.db.execute(
                    """INSERT INTO evidence_source_refs
                       (source_ref_id, evidence_id, document_id, source_page,
                        bbox_json, sentence_refs_json, passage, source_role,
                        created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        source.source_ref_id, source.evidence_id,
                        source.document_id, source.source_page,
                        _json(source.bbox) if source.bbox else None,
                        _json(source.sentence_refs), source.passage,
                        source.source_role, source.created_at,
                    ),
                )

    def ensure_primary_source(self, evidence) -> None:
        """Persist the compatibility source columns as a normalized row."""
        if not evidence.source_text:
            return
        payload = "|".join((
            evidence.evidence_id, evidence.document_id,
            str(evidence.source_page or ""), evidence.source_text,
        ))
        source_id = "SRC_" + uuid.uuid5(uuid.NAMESPACE_URL, payload).hex
        sentence_refs = (evidence.data or {}).get("sentence_refs") or []
        self.db.execute(
            """INSERT OR IGNORE INTO evidence_source_refs
               (source_ref_id, evidence_id, document_id, source_page,
                bbox_json, sentence_refs_json, passage, source_role, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'primary', ?)
            """,
            (
                source_id, evidence.evidence_id, evidence.document_id,
                evidence.source_page,
                _json(evidence.bbox) if evidence.bbox else None,
                _json(sentence_refs), evidence.source_text,
                evidence.created_at,
            ),
        )
        # The same literal source may predate the deterministic source ID.
        # Refresh that existing row rather than inserting a second identity.
        self.db.execute(
            """UPDATE evidence_source_refs
               SET bbox_json=?, sentence_refs_json=?, source_role='primary'
               WHERE source_ref_id=? OR (
                 evidence_id=? AND document_id=?
                 AND source_page IS ? AND passage=?
               )""",
            (
                _json(evidence.bbox) if evidence.bbox else None,
                _json(sentence_refs), source_id, evidence.evidence_id,
                evidence.document_id, evidence.source_page,
                evidence.source_text,
            ),
        )

    def get_evidence_sources(self, evidence_id: str) -> list[dict]:
        rows = self.db.execute(
            """SELECT * FROM evidence_source_refs WHERE evidence_id=?
               ORDER BY CASE source_role WHEN 'primary' THEN 0 ELSE 1 END,
                        document_id, source_page""",
            (evidence_id,),
        ).fetchall()
        return [
            {
                **dict(row),
                "bbox": _loads(row["bbox_json"], None),
                "sentence_refs": _loads(row["sentence_refs_json"], []),
            }
            for row in rows
        ]

    # ----------------------------------------------------------- exclusions

    def replace_excluded_document(
        self, document_id: str, items: list[ExcludedEvidence]
    ) -> None:
        with self.db:
            locked = {
                row["excluded_id"] for row in self.db.execute(
                    """SELECT excluded_id FROM excluded_evidence
                       WHERE document_id=? AND review_status IN
                       ('accepted', 'corrected', 'rejected', 'locked')""",
                    (document_id,),
                ).fetchall()
            }
            if locked:
                placeholders = ",".join("?" for _ in locked)
                self.db.execute(
                    "DELETE FROM excluded_evidence WHERE document_id=? "
                    f"AND excluded_id NOT IN ({placeholders})",
                    (document_id, *sorted(locked)),
                )
            else:
                self.db.execute(
                    "DELETE FROM excluded_evidence WHERE document_id=?",
                    (document_id,),
                )
            for item in items:
                if item.document_id != document_id:
                    raise ValueError("Esclusione associata a un documento diverso")
                self.db.execute(
                    """INSERT INTO excluded_evidence
                       (excluded_id, patient_id, document_id, disposition,
                        reason_code, fact_type, concept, source_page, bbox_json,
                        sentence_refs_json, source_text, extraction_method,
                        model_name, prompt_version, review_status, data_json,
                        created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(document_id, disposition, source_text)
                       DO UPDATE SET
                         disposition=excluded.disposition,
                         reason_code=excluded.reason_code,
                         fact_type=excluded.fact_type,
                         concept=excluded.concept,
                         source_page=excluded.source_page,
                         bbox_json=excluded.bbox_json,
                         sentence_refs_json=excluded.sentence_refs_json,
                         source_text=excluded.source_text,
                         data_json=excluded.data_json
                       WHERE excluded_evidence.review_status NOT IN
                         ('accepted','corrected','rejected','locked')""",
                    (
                        item.excluded_id, item.patient_id, item.document_id,
                        item.disposition, item.reason_code, item.fact_type,
                        item.concept, item.source_page,
                        _json(item.bbox) if item.bbox else None,
                        _json(item.sentence_refs), item.source_text,
                        item.extraction_method, item.model_name,
                        item.prompt_version, item.review_status,
                        _json(item.data), item.created_at,
                    ),
                )

    def list_excluded(
        self, patient_id: str, *, pending_only: bool = False
    ) -> list[dict]:
        suffix = " AND review_status IN ('auto','pending')" if pending_only else ""
        rows = self.db.execute(
            "SELECT * FROM excluded_evidence WHERE patient_id=?" + suffix
            + " ORDER BY document_id, source_page, created_at",
            (patient_id,),
        ).fetchall()
        return [
            {
                **dict(row),
                "bbox": _loads(row["bbox_json"], None),
                "sentence_refs": _loads(row["sentence_refs_json"], []),
                "data": _loads(row["data_json"], {}),
            }
            for row in rows
        ]

    def review_exclusion(
        self, excluded_id: str, decision: str, *, reason: str = ""
    ) -> None:
        if decision not in {"accepted", "rejected", "pending"}:
            raise ValueError("Decisione di esclusione non valida")
        row = self.db.execute(
            "SELECT data_json FROM excluded_evidence WHERE excluded_id=?",
            (excluded_id,),
        ).fetchone()
        if row is None:
            raise ValueError("Evidenza esclusa non trovata")
        data = _loads(row["data_json"], {})
        data["review_reason"] = str(reason or "").strip()
        data["reviewed_at"] = _now()
        with self.db:
            self.db.execute(
                """UPDATE excluded_evidence
                   SET review_status=?, data_json=? WHERE excluded_id=?""",
                (decision, _json(data), excluded_id),
            )

    # --------------------------------------------------------- terminology

    def save_mapping(self, mapping: TerminologyMapping) -> None:
        self.db.execute(
            """INSERT INTO terminology_mappings
               (mapping_id, normalized_concept, fact_type, canonical_label,
                terminology_system, terminology_code, original_unit,
                canonical_unit, multiplier, offset, mapping_status, source,
                mapping_confidence, review_status, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(normalized_concept, fact_type, original_unit)
               DO UPDATE SET
                 canonical_label=excluded.canonical_label,
                 terminology_system=excluded.terminology_system,
                 terminology_code=excluded.terminology_code,
                 canonical_unit=excluded.canonical_unit,
                 multiplier=excluded.multiplier,
                 offset=excluded.offset,
                 mapping_status=excluded.mapping_status,
                 mapping_confidence=excluded.mapping_confidence,
                 source=excluded.source,
                 review_status=excluded.review_status,
                 updated_at=excluded.updated_at""",
            (
                mapping.mapping_id, mapping.normalized_concept,
                mapping.fact_type or "", mapping.canonical_label,
                mapping.terminology_system, mapping.terminology_code,
                mapping.original_unit or "", mapping.canonical_unit,
                mapping.multiplier, mapping.offset, mapping.mapping_status,
                mapping.source, mapping.mapping_confidence,
                mapping.review_status, mapping.created_at,
                mapping.updated_at,
            ),
        )
        self.db.commit()

    def find_mapping(
        self, normalized_concept: str, fact_type: str | None,
        original_unit: str | None = None,
    ) -> dict | None:
        row = self.db.execute(
            """SELECT * FROM terminology_mappings
               WHERE normalized_concept=?
                 AND (fact_type=? OR fact_type='')
                 AND (original_unit=? OR original_unit='')
               ORDER BY CASE WHEN fact_type=? THEN 0 ELSE 1 END,
                        CASE WHEN original_unit=? THEN 0 ELSE 1 END,
                        CASE review_status WHEN 'accepted' THEN 0 ELSE 1 END
               LIMIT 1""",
            (
                normalized_concept, fact_type or "", original_unit or "",
                fact_type or "", original_unit or "",
            ),
        ).fetchone()
        return dict(row) if row else None

    # ---------------------------------------------------------- duplicates

    def replace_duplicate_groups(
        self,
        patient_id: str,
        canonical_evidence,
        uncertain_pairs: list[tuple[str, str, float]] = (),
    ) -> list[str]:
        """Persist exact copy groups plus conservative near-copy candidates."""
        reviewed = {
            row["duplicate_group_id"] for row in self.db.execute(
                """SELECT duplicate_group_id FROM evidence_duplicate_groups
                   WHERE patient_id=? AND review_status IN
                   ('accepted','corrected','rejected','locked')""",
                (patient_id,),
            ).fetchall()
        }
        pending_ids = []
        with self.db:
            if reviewed:
                placeholders = ",".join("?" for _ in reviewed)
                self.db.execute(
                    "DELETE FROM evidence_duplicate_groups WHERE patient_id=? "
                    f"AND duplicate_group_id NOT IN ({placeholders})",
                    (patient_id, *sorted(reviewed)),
                )
            else:
                self.db.execute(
                    "DELETE FROM evidence_duplicate_groups WHERE patient_id=?",
                    (patient_id,),
                )

            groups = []
            for item in canonical_evidence:
                duplicate_ids = list(dict.fromkeys(
                    (item.data or {}).get("duplicate_source_evidence_ids") or []
                ))
                if not duplicate_ids:
                    continue
                members = [item.evidence_id, *duplicate_ids]
                key = "exact:" + ":".join(sorted(members))
                groups.append((item.evidence_id, members, key,
                               "exact_cross_document_copy", "auto"))
            for left, right, similarity in uncertain_pairs:
                members = sorted({left, right})
                key = "candidate:" + ":".join(members)
                groups.append((members[0], members, key,
                               f"near_copy_similarity={similarity:.3f}", "pending"))

            for canonical_id, members, key, reason, status in groups:
                group_id = "DUP_" + uuid.uuid5(
                    uuid.NAMESPACE_URL, f"{patient_id}:{key}"
                ).hex
                now = _now()
                self.db.execute(
                    """INSERT INTO evidence_duplicate_groups
                       (duplicate_group_id, patient_id, canonical_evidence_id,
                        occurrence_key, decision_reason, review_status,
                        created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(patient_id, occurrence_key) DO UPDATE SET
                         canonical_evidence_id=excluded.canonical_evidence_id,
                         decision_reason=excluded.decision_reason,
                         updated_at=excluded.updated_at
                       WHERE evidence_duplicate_groups.review_status NOT IN
                         ('accepted','corrected','rejected','locked')""",
                    (
                        group_id, patient_id, canonical_id, key, reason,
                        status, now, now,
                    ),
                )
                for evidence_id in members:
                    self.db.execute(
                        """INSERT OR IGNORE INTO evidence_duplicate_members
                           (duplicate_group_id, evidence_id, source_role,
                            created_at) VALUES (?, ?, ?, ?)""",
                        (
                            group_id, evidence_id,
                            "canonical" if evidence_id == canonical_id
                            else "duplicate_candidate", now,
                        ),
                    )
                stored_status = self.db.execute(
                    """SELECT review_status FROM evidence_duplicate_groups
                       WHERE patient_id=? AND occurrence_key=?""",
                    (patient_id, key),
                ).fetchone()
                if stored_status and stored_status["review_status"] == "pending":
                    pending_ids.append(group_id)
        return pending_ids

    def review_duplicate_group(self, group_id: str, decision: str) -> None:
        if decision not in {"accepted", "rejected", "pending"}:
            raise ValueError("Decisione di duplicazione non valida")
        cursor = self.db.execute(
            """UPDATE evidence_duplicate_groups
               SET review_status=?, updated_at=? WHERE duplicate_group_id=?""",
            (decision, _now(), group_id),
        )
        if cursor.rowcount != 1:
            raise ValueError("Gruppo di duplicati non trovato")
        self.db.commit()

    def list_accepted_duplicate_groups(self, patient_id: str) -> list[dict]:
        rows = self.db.execute(
            """SELECT * FROM evidence_duplicate_groups
               WHERE patient_id=? AND review_status='accepted'
               ORDER BY created_at, duplicate_group_id""",
            (patient_id,),
        ).fetchall()
        result = []
        for row in rows:
            members = self.db.execute(
                """SELECT evidence_id, source_role
                   FROM evidence_duplicate_members
                   WHERE duplicate_group_id=? ORDER BY evidence_id""",
                (row["duplicate_group_id"],),
            ).fetchall()
            result.append({
                **dict(row), "members": [dict(member) for member in members],
            })
        return result

    # -------------------------------------------------------------- graph

    def replace_evidence_relations(
        self, patient_id: str, relations: list[EvidenceRelation]
    ) -> None:
        with self.db:
            self.db.execute(
                """DELETE FROM evidence_relations WHERE patient_id=?
                   AND review_status NOT IN
                   ('accepted','corrected','rejected','locked')""",
                (patient_id,),
            )
            for relation in relations:
                if relation.patient_id != patient_id:
                    raise ValueError("Relazione associata a un altro paziente")
                self.db.execute(
                    """INSERT INTO evidence_relations
                       (relation_id, patient_id, source_evidence_id,
                        target_evidence_id, relation_type, direction, weight,
                        cluster_effect, rationale, rule_features_json,
                        model_votes_json, generation_method, review_status,
                        created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(source_evidence_id, target_evidence_id,
                                   relation_type) DO UPDATE SET
                         direction=excluded.direction,
                         weight=excluded.weight,
                         cluster_effect=excluded.cluster_effect,
                         rationale=excluded.rationale,
                         rule_features_json=excluded.rule_features_json,
                         model_votes_json=excluded.model_votes_json,
                         generation_method=excluded.generation_method
                       WHERE evidence_relations.review_status NOT IN
                         ('accepted','corrected','rejected','locked')""",
                    (
                        relation.relation_id, relation.patient_id,
                        relation.source_evidence_id,
                        relation.target_evidence_id, relation.relation_type,
                        relation.direction, relation.weight,
                        relation.cluster_effect, relation.rationale,
                        _json(relation.rule_features),
                        _json(relation.model_votes),
                        relation.generation_method, relation.review_status,
                        relation.created_at,
                    ),
                )

    def list_evidence_relations(self, patient_id: str) -> list[dict]:
        rows = self.db.execute(
            """SELECT * FROM evidence_relations WHERE patient_id=?
               ORDER BY weight DESC, relation_id""",
            (patient_id,),
        ).fetchall()
        return [
            {
                **dict(row),
                "rule_features": _loads(row["rule_features_json"], {}),
                "model_votes": _loads(row["model_votes_json"], []),
            }
            for row in rows
        ]

    def list_relation_adjudication_cache(
        self, patient_id: str, namespace: str
    ) -> dict[str, dict]:
        """Return validated-at-read candidates checkpointed for one model.

        The caller still validates the decision against the current JSON
        contract.  ``cache_key`` contains the complete candidate payload, so
        changed evidence cannot accidentally reuse a stale model answer.
        """
        rows = self.db.execute(
            """SELECT cache_key, decision_json
               FROM evidence_relation_adjudication_cache
               WHERE patient_id=? AND namespace=?""",
            (patient_id, namespace),
        ).fetchall()
        return {
            str(row["cache_key"]): _loads(row["decision_json"], {})
            for row in rows
        }

    def save_relation_adjudication_batch(
        self,
        patient_id: str,
        namespace: str,
        round_index: int,
        records: list[dict],
    ) -> None:
        """Durably checkpoint one completed LLM relation batch."""
        if not records:
            return
        now = _now()
        with self.db:
            for record in records:
                self.db.execute(
                    """INSERT INTO evidence_relation_adjudication_cache
                       (patient_id, namespace, candidate_id, round_index,
                        cache_key, source_evidence_id, target_evidence_id,
                        decision_json, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(patient_id, namespace, candidate_id,
                                   round_index) DO UPDATE SET
                         cache_key=excluded.cache_key,
                         source_evidence_id=excluded.source_evidence_id,
                         target_evidence_id=excluded.target_evidence_id,
                         decision_json=excluded.decision_json,
                         updated_at=excluded.updated_at""",
                    (
                        patient_id, namespace, record["candidate_id"],
                        int(round_index), record["cache_key"],
                        record["source_evidence_id"],
                        record["target_evidence_id"],
                        _json(record["decision"]), now, now,
                    ),
                )

    def review_evidence_relation(self, relation_id: str, decision: str) -> None:
        if decision not in {"accepted", "rejected", "pending"}:
            raise ValueError("Decisione sulla relazione non valida")
        cursor = self.db.execute(
            """UPDATE evidence_relations SET review_status=?
               WHERE relation_id=?""",
            (decision, relation_id),
        )
        if cursor.rowcount != 1:
            raise ValueError("Relazione fra evidenze non trovata")
        self.db.commit()

    # -------------------------------------------------------------- claims

    def replace_event_claims(
        self, event_id: str, claims: list[EventClaim]
    ) -> None:
        with self.db:
            locked = self.db.execute(
                """SELECT 1 FROM clinical_events WHERE event_id=?
                   AND review_status IN ('accepted','corrected','locked')""",
                (event_id,),
            ).fetchone()
            if locked:
                return
            self.db.execute(
                "DELETE FROM clinical_event_claims WHERE event_id=?",
                (event_id,),
            )
            for claim in claims:
                if claim.event_id != event_id:
                    raise ValueError("Claim associato a un evento diverso")
                self.db.execute(
                    """INSERT INTO clinical_event_claims
                       (claim_id, event_id, claim_type, text, certainty,
                        review_status, position, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        claim.claim_id, claim.event_id, claim.claim_type,
                        claim.text, claim.certainty, claim.review_status,
                        claim.position, claim.created_at, claim.updated_at,
                    ),
                )
                source_rows = [
                    (claim.claim_id, "evidence", source_id, "supports", _now())
                    for source_id in claim.evidence_ids
                ] + [
                    (claim.claim_id, "lab_observation", str(source_id),
                     "supports", _now())
                    for source_id in claim.lab_observation_ids
                ]
                for row in source_rows:
                    self._validate_claim_source(row[1], row[2])
                    self.db.execute(
                        """INSERT INTO clinical_event_claim_sources
                           (claim_id, source_type, source_id, source_role,
                            created_at) VALUES (?, ?, ?, ?, ?)""",
                        row,
                    )

    def list_event_claims(self, event_id: str) -> list[dict]:
        claims = self.db.execute(
            """SELECT * FROM clinical_event_claims WHERE event_id=?
               ORDER BY position, claim_id""",
            (event_id,),
        ).fetchall()
        result = []
        for claim in claims:
            sources = self.db.execute(
                """SELECT source_type, source_id, source_role
                   FROM clinical_event_claim_sources WHERE claim_id=?
                   ORDER BY source_type, source_id""",
                (claim["claim_id"],),
            ).fetchall()
            result.append({**dict(claim), "sources": [dict(row) for row in sources]})
        return result

    def _validate_claim_source(self, source_type: str, source_id: str) -> None:
        if source_type == "evidence":
            row = self.db.execute(
                "SELECT 1 FROM clinical_evidence WHERE evidence_id=?",
                (source_id,),
            ).fetchone()
        elif source_type == "lab_observation":
            row = self.db.execute(
                "SELECT 1 FROM lab_values WHERE id=?", (source_id,)
            ).fetchone()
        else:
            raise ValueError(f"Tipo fonte claim non valido: {source_type}")
        if row is None:
            raise ValueError(f"Fonte claim inesistente: {source_type}:{source_id}")

    # ----------------------------------------------------------- hypotheses

    def save_hypotheses(
        self, patient_id: str, hypotheses: list[ClinicalHypothesis]
    ) -> None:
        with self.db:
            for item in hypotheses:
                if item.patient_id != patient_id:
                    raise ValueError("Ipotesi associata a un altro paziente")
                self.db.execute(
                    """INSERT INTO clinical_hypotheses
                       (hypothesis_id, patient_id, source_event_id,
                        target_event_id, hypothesis_type, strength,
                        known_mechanism, rationale, confounders_json,
                        survival_score, status, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(source_event_id, target_event_id,
                                   hypothesis_type) DO UPDATE SET
                         strength=excluded.strength,
                         known_mechanism=excluded.known_mechanism,
                         rationale=excluded.rationale,
                         confounders_json=excluded.confounders_json,
                         survival_score=excluded.survival_score,
                         updated_at=excluded.updated_at
                       WHERE clinical_hypotheses.status='needs_review'""",
                    (
                        item.hypothesis_id, item.patient_id,
                        item.source_event_id, item.target_event_id,
                        item.hypothesis_type, item.strength,
                        1 if item.known_mechanism else 0, item.rationale,
                        _json(item.confounders), item.survival_score,
                        item.status, item.created_at, item.updated_at,
                    ),
                )

    def replace_pending_hypotheses(
        self, patient_id: str, hypotheses: list[ClinicalHypothesis]
    ) -> None:
        """Refresh generated hypotheses while preserving reviewed decisions."""
        with self.db:
            self.db.execute(
                """DELETE FROM clinical_hypotheses
                   WHERE patient_id=? AND status='needs_review'""",
                (patient_id,),
            )
            self.save_hypotheses(patient_id, hypotheses)

    def review_hypothesis(self, hypothesis_id: str, decision: str) -> None:
        if decision not in {"accepted", "rejected", "needs_review"}:
            raise ValueError("Decisione sull'ipotesi non valida")
        cursor = self.db.execute(
            """UPDATE clinical_hypotheses SET status=?, updated_at=?
               WHERE hypothesis_id=?""",
            (decision, _now(), hypothesis_id),
        )
        if cursor.rowcount != 1:
            raise ValueError("Ipotesi non trovata")
        self.db.commit()

    def list_hypotheses(
        self, patient_id: str, *, accepted_only: bool = False
    ) -> list[dict]:
        suffix = " AND status='accepted'" if accepted_only else ""
        rows = self.db.execute(
            "SELECT * FROM clinical_hypotheses WHERE patient_id=?" + suffix
            + " ORDER BY status, survival_score DESC, hypothesis_id",
            (patient_id,),
        ).fetchall()
        return [
            {
                **dict(row),
                "known_mechanism": bool(row["known_mechanism"]),
                "confounders": _loads(row["confounders_json"], []),
            }
            for row in rows
        ]

    # ----------------------------------------------------------- categories

    def list_categories(self, object_level: str | None = None) -> list[dict]:
        if object_level:
            rows = self.db.execute(
                """SELECT * FROM clinical_category_registry
                   WHERE enabled=1 AND object_level IN (?, 'both')
                   ORDER BY display_name""",
                (object_level,),
            ).fetchall()
        else:
            rows = self.db.execute(
                """SELECT * FROM clinical_category_registry
                   WHERE enabled=1 ORDER BY object_level, display_name"""
            ).fetchall()
        return [{**dict(row), "data": _loads(row["data_json"], {})}
                for row in rows]

    def register_category(
        self,
        category_id: str,
        display_name: str,
        *,
        object_level: str = "event",
        data: dict | None = None,
        enabled: bool = True,
    ) -> None:
        """Register a project category without requiring a schema migration."""
        category_id = str(category_id or "").strip().casefold()
        if not category_id or not all(
            character.isalnum() or character == "_" for character in category_id
        ):
            raise ValueError("Identificativo categoria non valido")
        if object_level not in {"evidence", "event", "both"}:
            raise ValueError("Livello categoria non valido")
        now = _now()
        with self.db:
            self.db.execute(
                """INSERT INTO clinical_category_registry
                   (category,object_level,display_name,enabled,is_builtin,
                    data_json,created_at,updated_at)
                   VALUES (?, ?, ?, ?, 0, ?, ?, ?)
                   ON CONFLICT(category) DO UPDATE SET
                     object_level=excluded.object_level,
                     display_name=excluded.display_name,
                     enabled=excluded.enabled,
                     data_json=excluded.data_json,
                     updated_at=excluded.updated_at
                   WHERE clinical_category_registry.is_builtin=0""",
                (
                    category_id, object_level, display_name.strip(),
                    1 if enabled else 0, _json(data or {}), now, now,
                ),
            )
