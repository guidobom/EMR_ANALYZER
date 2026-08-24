"""Persistence and retrieval for the evidence-based clinical registry."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import uuid
from typing import Iterable, Optional

from .engine import DatabaseEngine
from ..models.clinical_registry import (
    ClinicalEpisode,
    ClinicalEvent,
    ClinicalEventRelation,
    EventEvidenceLink,
    EventUpdate,
    LabTrend,
    MedicationCourse,
    OncologyLine,
)


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _loads(value, fallback):
    try:
        parsed = json.loads(value) if value else fallback
    except (TypeError, ValueError):
        return fallback
    return parsed


class ClinicalRegistryRepository:
    """Database access for events, episodes and their complete evidence tree."""

    def __init__(self, db: DatabaseEngine):
        self.db = db

    def save_episode(self, episode: ClinicalEpisode) -> None:
        self.db.execute(
            """INSERT INTO clinical_episodes
               (episode_id, patient_id, category, canonical_entity,
                onset_date, onset_date_end, onset_precision,
                first_documented_date, resolution_date, status,
                recurrence_index, previous_episode_id, data_json,
                created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(episode_id) DO UPDATE SET
                 category=excluded.category,
                 canonical_entity=excluded.canonical_entity,
                 onset_date=excluded.onset_date,
                 onset_date_end=excluded.onset_date_end,
                 onset_precision=excluded.onset_precision,
                 first_documented_date=excluded.first_documented_date,
                 resolution_date=excluded.resolution_date,
                 status=excluded.status,
                 recurrence_index=excluded.recurrence_index,
                 previous_episode_id=excluded.previous_episode_id,
                 data_json=excluded.data_json,
                 updated_at=excluded.updated_at""",
            (
                episode.episode_id, episode.patient_id, episode.category,
                episode.canonical_entity, episode.onset_date,
                episode.onset_date_end, episode.onset_precision,
                episode.first_documented_date, episode.resolution_date,
                episode.status, episode.recurrence_index,
                episode.previous_episode_id, _json(episode.data),
                episode.created_at, episode.updated_at,
            ),
        )

    def save_event(
        self,
        event: ClinicalEvent,
        links: Iterable[EventEvidenceLink] = (),
        updates: Iterable[EventUpdate] = (),
    ) -> None:
        """Upsert one event and atomically refresh its generated projection.

        Human-reviewed events are never overwritten by an automatic rebuild.
        Their evidence links may still be augmented, preserving review intent.
        """
        links = list(links)
        updates = list(updates)
        with self.db:
            self._save_event_row(event)
            existing = self.db.execute(
                "SELECT review_status FROM clinical_events WHERE event_id=?",
                (event.event_id,),
            ).fetchone()
            human_locked = bool(
                existing and existing["review_status"] in {
                    "accepted", "corrected", "rejected"
                }
            )
            if not human_locked:
                self.db.execute(
                    "DELETE FROM clinical_event_evidence WHERE event_id=?",
                    (event.event_id,),
                )
                self.db.execute(
                    "DELETE FROM clinical_event_updates WHERE event_id=?",
                    (event.event_id,),
                )
            for link in links:
                self._save_link(link)
            if not human_locked:
                for update in updates:
                    self._save_update(update)

    def replace_generated_registry(
        self,
        patient_id: str,
        episodes: Iterable[ClinicalEpisode],
        event_bundles: Iterable[
            tuple[ClinicalEvent, Iterable[EventEvidenceLink], Iterable[EventUpdate]]
        ],
    ) -> None:
        """Replace automatic rows while preserving every human decision."""
        episodes = list(episodes)
        bundles = [
            (event, list(links), list(updates))
            for event, links, updates in event_bundles
        ]
        incoming_event_ids = {event.event_id for event, _, _ in bundles}
        incoming_episode_ids = {episode.episode_id for episode in episodes}
        with self.db:
            stale = self.db.execute(
                """SELECT event_id FROM clinical_events
                   WHERE patient_id=?
                     AND review_status NOT IN ('accepted','corrected','rejected')""",
                (patient_id,),
            ).fetchall()
            for row in stale:
                if row["event_id"] not in incoming_event_ids:
                    self.db.execute(
                        "DELETE FROM clinical_events WHERE event_id=?",
                        (row["event_id"],),
                    )

            for episode in episodes:
                self.save_episode(episode)
            for event, links, updates in bundles:
                self.save_event(event, links, updates)

            old_episodes = self.db.execute(
                "SELECT episode_id FROM clinical_episodes WHERE patient_id=?",
                (patient_id,),
            ).fetchall()
            for row in old_episodes:
                if row["episode_id"] in incoming_episode_ids:
                    continue
                used = self.db.execute(
                    "SELECT 1 FROM clinical_events WHERE episode_id=? LIMIT 1",
                    (row["episode_id"],),
                ).fetchone()
                if not used:
                    self.db.execute(
                        "DELETE FROM clinical_episodes WHERE episode_id=?",
                        (row["episode_id"],),
                    )

    def _save_event_row(self, event: ClinicalEvent) -> None:
        existing = self.db.execute(
            "SELECT review_status FROM clinical_events WHERE event_id=?",
            (event.event_id,),
        ).fetchone()
        if existing and existing["review_status"] in {
            "accepted", "corrected", "rejected"
        }:
            return
        self.db.execute(
            """INSERT INTO clinical_events
               (event_id, patient_id, episode_id, category, canonical_entity,
                summary_short, summary_detail, anatomical_site, laterality,
                severity, significance, status, certainty, assertion,
                first_evidence_date, first_documented_date, date_end,
                date_precision, confidence, review_status,
                structured_data_json, model_name, prompt_version,
                schema_version, version, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                       ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(event_id) DO UPDATE SET
                 episode_id=excluded.episode_id,
                 category=excluded.category,
                 canonical_entity=excluded.canonical_entity,
                 summary_short=excluded.summary_short,
                 summary_detail=excluded.summary_detail,
                 anatomical_site=excluded.anatomical_site,
                 laterality=excluded.laterality,
                 severity=excluded.severity,
                 significance=excluded.significance,
                 status=excluded.status,
                 certainty=excluded.certainty,
                 assertion=excluded.assertion,
                 first_evidence_date=excluded.first_evidence_date,
                 first_documented_date=excluded.first_documented_date,
                 date_end=excluded.date_end,
                 date_precision=excluded.date_precision,
                 confidence=excluded.confidence,
                 review_status=excluded.review_status,
                 structured_data_json=excluded.structured_data_json,
                 model_name=excluded.model_name,
                 prompt_version=excluded.prompt_version,
                 schema_version=excluded.schema_version,
                 version=clinical_events.version + 1,
                 updated_at=excluded.updated_at""",
            (
                event.event_id, event.patient_id, event.episode_id,
                event.category, event.canonical_entity, event.summary_short,
                event.summary_detail, event.anatomical_site, event.laterality,
                event.severity, event.significance, event.status,
                event.certainty, event.assertion, event.first_evidence_date,
                event.first_documented_date, event.date_end,
                event.date_precision, event.confidence, event.review_status,
                _json(event.structured_data), event.model_name,
                event.prompt_version, event.schema_version, event.version,
                event.created_at, event.updated_at,
            ),
        )

    def _save_link(self, link: EventEvidenceLink) -> None:
        self.db.execute(
            """INSERT INTO clinical_event_evidence
               (link_id, event_id, evidence_id, relation,
                role, relation_confidence, rationale, included_in_summary,
                created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(event_id, evidence_id, relation) DO UPDATE SET
                 relation_confidence=excluded.relation_confidence,
                 role=excluded.role,
                 rationale=excluded.rationale,
                 included_in_summary=excluded.included_in_summary""",
            (
                link.link_id, link.event_id, link.evidence_id, link.relation,
                link.role, link.relation_confidence, link.rationale,
                1 if link.included_in_summary else 0, link.created_at,
            ),
        )

    def _save_update(self, update: EventUpdate) -> None:
        self.db.execute(
            """INSERT OR REPLACE INTO clinical_event_updates
               (update_id, event_id, update_date, date_precision, summary,
                status_after, evidence_ids_json, structured_data_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                update.update_id, update.event_id, update.update_date,
                update.date_precision, update.summary, update.status_after,
                _json(update.evidence_ids), _json(update.structured_data),
                update.created_at,
            ),
        )

    def replace_generated_relations(
        self,
        patient_id: str,
        relations: Iterable[ClinicalEventRelation],
    ) -> None:
        """Replace automatic episode links while preserving reviewed links."""
        relations = list(relations)
        with self.db:
            self.db.execute(
                """DELETE FROM clinical_event_relations
                   WHERE patient_id=?
                     AND review_status NOT IN ('accepted','corrected','rejected')""",
                (patient_id,),
            )
            for relation in relations:
                self.db.execute(
                    """INSERT INTO clinical_event_relations
                       (relation_id, patient_id, source_event_id,
                        target_event_id, relation_type, confidence, rationale,
                        review_status, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(source_event_id, target_event_id,
                                   relation_type) DO UPDATE SET
                         confidence=excluded.confidence,
                         rationale=excluded.rationale,
                         review_status=excluded.review_status
                       WHERE clinical_event_relations.review_status
                         NOT IN ('accepted','corrected','rejected')""",
                    (
                        relation.relation_id, relation.patient_id,
                        relation.source_event_id, relation.target_event_id,
                        relation.relation_type, relation.confidence,
                        relation.rationale, relation.review_status,
                        relation.created_at,
                    ),
                )

    def merge_events_manual(
        self,
        patient_id: str,
        survivor_event_id: str,
        absorbed_event_id: str,
        *,
        summary_short: str,
    ) -> None:
        """Merge two events without deleting either audit identity."""
        if survivor_event_id == absorbed_event_id:
            raise ValueError("Seleziona due eventi diversi")
        rows = self.db.execute(
            """SELECT * FROM clinical_events
               WHERE patient_id=? AND event_id IN (?, ?)""",
            (patient_id, survivor_event_id, absorbed_event_id),
        ).fetchall()
        if len(rows) != 2:
            raise ValueError("Uno degli eventi da unire non esiste")
        survivor = next(row for row in rows if row["event_id"] == survivor_event_id)
        data = _loads(survivor["structured_data_json"], {})
        merged = list(dict.fromkeys([
            *(data.get("manually_merged_event_ids") or []), absorbed_event_id,
        ]))
        data["manually_merged_event_ids"] = merged
        now = datetime.now(timezone.utc).isoformat()
        with self.db:
            links = self.db.execute(
                """SELECT * FROM clinical_event_evidence WHERE event_id=?""",
                (absorbed_event_id,),
            ).fetchall()
            for link in links:
                self.db.execute(
                    """INSERT OR IGNORE INTO clinical_event_evidence
                       (link_id,event_id,evidence_id,relation,role,
                        relation_confidence,rationale,included_in_summary,created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        "LNK_" + uuid.uuid4().hex, survivor_event_id,
                        link["evidence_id"], link["relation"], link["role"],
                        link["relation_confidence"],
                        link["rationale"] or "Unione manuale di eventi",
                        link["included_in_summary"], now,
                    ),
                )
            evidence_ids = [
                row["evidence_id"] for row in self.db.execute(
                    """SELECT evidence_id FROM clinical_event_evidence
                       WHERE event_id=?""", (survivor_event_id,)
                ).fetchall()
            ]
            data["evidence_ids"] = list(dict.fromkeys(evidence_ids))
            self.db.execute(
                """UPDATE clinical_events SET summary_short=?, summary_detail=?,
                   structured_data_json=?, review_status='corrected',
                   version=version+1, updated_at=? WHERE event_id=?""",
                (
                    summary_short.strip(), summary_short.strip(), _json(data),
                    now, survivor_event_id,
                ),
            )
            self.db.execute(
                """UPDATE clinical_events SET review_status='rejected',
                   version=version+1, updated_at=? WHERE event_id=?""",
                (now, absorbed_event_id),
            )
            self._replace_manual_claim(
                survivor_event_id, summary_short, evidence_ids, now
            )

    def split_event_manual(
        self,
        patient_id: str,
        event_id: str,
        evidence_ids: list[str],
        *,
        original_summary: str,
        new_summary: str,
    ) -> str:
        """Split selected evidence into a new locked event and episode."""
        row = self.db.execute(
            "SELECT * FROM clinical_events WHERE patient_id=? AND event_id=?",
            (patient_id, event_id),
        ).fetchone()
        if row is None:
            raise ValueError("Evento da dividere non trovato")
        all_links = self.db.execute(
            "SELECT * FROM clinical_event_evidence WHERE event_id=?",
            (event_id,),
        ).fetchall()
        selected = set(evidence_ids)
        available = {link["evidence_id"] for link in all_links}
        if not selected or not selected < available:
            raise ValueError(
                "Lo split richiede almeno una evidenza per ciascun nuovo evento"
            )
        if not selected <= available:
            raise ValueError("Una evidenza selezionata non appartiene all'evento")
        new_event_id = "EVT_" + uuid.uuid4().hex
        new_episode_id = "EPI_" + uuid.uuid4().hex
        now = datetime.now(timezone.utc).isoformat()
        selected_rows = [link for link in all_links if link["evidence_id"] in selected]
        evidence_dates = self.db.execute(
            """SELECT observed_date, document_date FROM clinical_evidence
               WHERE evidence_id IN ({})""".format(
                ",".join("?" for _ in selected)
            ), tuple(sorted(selected)),
        ).fetchall()
        selected_date = min((
            value for evidence in evidence_dates
            for value in (evidence["observed_date"] or evidence["document_date"],)
            if value
        ), default=row["first_evidence_date"])
        structured = _loads(row["structured_data_json"], {})
        structured.update({
            "evidence_ids": sorted(selected),
            "manual_split_from_event_id": event_id,
            "claims": [],
        })
        original_data = _loads(row["structured_data_json"], {})
        original_data.update({
            "evidence_ids": sorted(available - selected),
            "manual_split_event_ids": list(dict.fromkeys([
                *(original_data.get("manual_split_event_ids") or []), new_event_id,
            ])),
            "claims": [],
        })
        with self.db:
            self.db.execute(
                """INSERT INTO clinical_episodes
                   (episode_id,patient_id,category,canonical_entity,onset_date,
                    onset_date_end,onset_precision,first_documented_date,
                    resolution_date,status,recurrence_index,previous_episode_id,
                    data_json,created_at,updated_at)
                   VALUES (?, ?, ?, ?, ?, NULL, ?, ?, NULL, ?, 1, NULL, '{}', ?, ?)""",
                (
                    new_episode_id, patient_id, row["category"],
                    row["canonical_entity"], selected_date,
                    row["date_precision"], row["first_documented_date"],
                    row["status"], now, now,
                ),
            )
            self.db.execute(
                """INSERT INTO clinical_events
                   (event_id,patient_id,episode_id,category,canonical_entity,
                    summary_short,summary_detail,anatomical_site,laterality,
                    severity,significance,status,certainty,assertion,
                    first_evidence_date,first_documented_date,date_end,
                    date_precision,confidence,review_status,structured_data_json,
                    model_name,prompt_version,schema_version,version,created_at,updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                           ?, 'corrected', ?, ?, ?, ?, 1, ?, ?)""",
                (
                    new_event_id, patient_id, new_episode_id, row["category"],
                    row["canonical_entity"], new_summary.strip(),
                    new_summary.strip(), row["anatomical_site"], row["laterality"],
                    row["severity"], row["significance"], row["status"],
                    row["certainty"], row["assertion"], selected_date,
                    row["first_documented_date"], row["date_end"],
                    row["date_precision"], row["confidence"], _json(structured),
                    row["model_name"], row["prompt_version"],
                    row["schema_version"], now, now,
                ),
            )
            for link in selected_rows:
                self.db.execute(
                    """INSERT INTO clinical_event_evidence
                       (link_id,event_id,evidence_id,relation,role,
                        relation_confidence,rationale,included_in_summary,created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        "LNK_" + uuid.uuid4().hex, new_event_id,
                        link["evidence_id"], link["relation"], link["role"],
                        link["relation_confidence"], "Split manuale",
                        link["included_in_summary"], now,
                    ),
                )
            self.db.execute(
                """DELETE FROM clinical_event_evidence WHERE event_id=? AND
                   evidence_id IN ({})""".format(
                    ",".join("?" for _ in selected)
                ), (event_id, *sorted(selected)),
            )
            self.db.execute(
                "DELETE FROM clinical_event_claims WHERE event_id IN (?, ?)",
                (event_id, new_event_id),
            )
            self.db.execute(
                """UPDATE clinical_events SET summary_short=?, summary_detail=?,
                   structured_data_json=?, review_status='corrected',
                   version=version+1, updated_at=? WHERE event_id=?""",
                (
                    original_summary.strip(), original_summary.strip(),
                    _json(original_data), now, event_id,
                ),
            )
            self._replace_manual_claim(
                event_id, original_summary, sorted(available - selected), now
            )
            self._replace_manual_claim(
                new_event_id, new_summary, sorted(selected), now
            )
        return new_event_id

    def _replace_manual_claim(
        self,
        event_id: str,
        text: str,
        evidence_ids: Iterable[str],
        now: str,
    ) -> None:
        """Create a fully cited claim after a human structural correction."""
        source_ids = list(dict.fromkeys(str(item) for item in evidence_ids))
        self.db.execute(
            "DELETE FROM clinical_event_claims WHERE event_id=?", (event_id,)
        )
        claim_text = " ".join(str(text or "").split())
        if not claim_text or not source_ids:
            return
        claim_id = "CLM_" + uuid.uuid4().hex
        self.db.execute(
            """INSERT INTO clinical_event_claims
               (claim_id,event_id,claim_type,text,certainty,review_status,
                position,created_at,updated_at)
               VALUES (?, ?, 'clinical_observation', ?, 'confirmed',
                       'corrected', 0, ?, ?)""",
            (claim_id, event_id, claim_text, now, now),
        )
        for evidence_id in source_ids:
            exists = self.db.execute(
                "SELECT 1 FROM clinical_evidence WHERE evidence_id=?",
                (evidence_id,),
            ).fetchone()
            if exists is None:
                continue
            self.db.execute(
                """INSERT INTO clinical_event_claim_sources
                   (claim_id,source_type,source_id,source_role,created_at)
                   VALUES (?, 'evidence', ?, 'supports', ?)""",
                (claim_id, evidence_id, now),
            )

    def get_events(
        self,
        patient_id: str,
        *,
        category: Optional[str] = None,
        status: Optional[str] = None,
        include_rejected: bool = False,
    ) -> list[ClinicalEvent]:
        clauses = ["patient_id=?"]
        params: list[object] = [patient_id]
        if category:
            clauses.append("category=?")
            params.append(category)
        if status:
            clauses.append("status=?")
            params.append(status)
        if not include_rejected:
            clauses.append("review_status<>'rejected'")
        rows = self.db.execute(
            "SELECT * FROM clinical_events WHERE "
            + " AND ".join(clauses)
            + " ORDER BY COALESCE(first_evidence_date, '9999'), event_id",
            tuple(params),
        ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def get_event_detail(self, event_id: str) -> dict | None:
        row = self.db.execute(
            "SELECT * FROM clinical_events WHERE event_id=?", (event_id,)
        ).fetchone()
        if row is None:
            return None
        event = asdict(self._row_to_event(row))
        episode = None
        if row["episode_id"]:
            episode_row = self.db.execute(
                "SELECT * FROM clinical_episodes WHERE episode_id=?",
                (row["episode_id"],),
            ).fetchone()
            if episode_row:
                episode = dict(episode_row)
                episode["data"] = _loads(episode.pop("data_json"), {})
        evidence_rows = self.db.execute(
            """SELECT l.link_id, l.relation, l.role, l.relation_confidence,
                      l.rationale, l.included_in_summary,
                      e.*, d.document_date AS source_document_date
               FROM clinical_event_evidence l
               JOIN clinical_evidence e ON e.evidence_id=l.evidence_id
               JOIN documents d ON d.id=e.document_id
               WHERE l.event_id=?
               ORDER BY COALESCE(e.observed_date, d.document_date),
                        e.document_id, e.source_page, e.id""",
            (event_id,),
        ).fetchall()
        evidences = []
        for evidence_row in evidence_rows:
            item = dict(evidence_row)
            item["data"] = _loads(item.pop("data_json"), {})
            item["bbox"] = _loads(item.pop("bbox_json"), None)
            source_rows = self.db.execute(
                """SELECT * FROM evidence_source_refs WHERE evidence_id=?
                   ORDER BY CASE source_role WHEN 'primary' THEN 0 ELSE 1 END,
                            document_id, source_page""",
                (item["evidence_id"],),
            ).fetchall()
            item["source_refs"] = [{
                **dict(source),
                "bbox": _loads(source["bbox_json"], None),
                "sentence_refs": _loads(source["sentence_refs_json"], []),
            } for source in source_rows]
            evidences.append(item)
        update_rows = self.db.execute(
            """SELECT * FROM clinical_event_updates WHERE event_id=?
               ORDER BY COALESCE(update_date, '9999'), update_id""",
            (event_id,),
        ).fetchall()
        updates = []
        for update_row in update_rows:
            item = dict(update_row)
            item["evidence_ids"] = _loads(item.pop("evidence_ids_json"), [])
            item["structured_data"] = _loads(
                item.pop("structured_data_json"), {}
            )
            updates.append(item)
        relation_rows = self.db.execute(
            """SELECT r.*,
                      source.summary_short AS source_summary,
                      target.summary_short AS target_summary,
                      source.category AS source_category,
                      target.category AS target_category
               FROM clinical_event_relations r
               JOIN clinical_events source
                 ON source.event_id=r.source_event_id
               JOIN clinical_events target
                 ON target.event_id=r.target_event_id
               WHERE r.source_event_id=? OR r.target_event_id=?
               ORDER BY r.created_at, r.relation_id""",
            (event_id, event_id),
        ).fetchall()
        review_rows = self.db.execute(
            """SELECT * FROM review_decisions
               WHERE target_type='clinical_event' AND target_id=?
               ORDER BY created_at, decision_id""",
            (event_id,),
        ).fetchall()
        reviews = []
        for review_row in review_rows:
            item = dict(review_row)
            item["previous_value"] = _loads(
                item.pop("previous_value_json"), {}
            )
            item["corrected_value"] = _loads(
                item.pop("corrected_value_json"), {}
            )
            reviews.append(item)
        claim_rows = self.db.execute(
            """SELECT * FROM clinical_event_claims WHERE event_id=?
               ORDER BY position, claim_id""",
            (event_id,),
        ).fetchall()
        claims = []
        for claim in claim_rows:
            sources = self.db.execute(
                """SELECT source_type, source_id, source_role
                   FROM clinical_event_claim_sources WHERE claim_id=?
                   ORDER BY source_type, source_id""",
                (claim["claim_id"],),
            ).fetchall()
            claims.append({
                **dict(claim), "sources": [dict(source) for source in sources]
            })
        return {
            "event": event,
            "episode": episode,
            "evidence": evidences,
            "updates": updates,
            "relations": [dict(item) for item in relation_rows],
            "reviews": reviews,
            "claims": claims,
        }

    def search_events(
        self,
        patient_id: str,
        query: str,
        *,
        limit: int = 200,
        category: Optional[str] = None,
        status: Optional[str] = None,
    ) -> list[ClinicalEvent]:
        """Local FTS5 search with a safe LIKE fallback."""
        query = " ".join(str(query or "").split()).strip()
        if not query:
            return self.get_events(
                patient_id, category=category, status=status
            )[:limit]
        match_query = _fts_match_query(query)
        filters = ["e.patient_id=?", "e.review_status<>'rejected'"]
        params: list[object] = [match_query, patient_id]
        if category:
            filters.append("e.category=?")
            params.append(category)
        if status:
            filters.append("e.status=?")
            params.append(status)
        params.append(max(1, min(int(limit), 1000)))
        try:
            rows = self.db.execute(
                """SELECT e.* FROM clinical_events_fts f
                   JOIN clinical_events e ON e.rowid=f.rowid
                   WHERE clinical_events_fts MATCH ? AND """
                + " AND ".join(filters)
                + " ORDER BY bm25(clinical_events_fts), "
                  "COALESCE(e.first_evidence_date, '9999') LIMIT ?",
                tuple(params),
            ).fetchall()
        except Exception as exc:
            message = str(exc).lower()
            if not any(token in message for token in (
                "no such table", "fts5", "syntax error", "unterminated"
            )):
                raise
            pattern = f"%{query}%"
            like_filters = [
                "patient_id=?", "review_status<>'rejected'",
                "(canonical_entity LIKE ? OR summary_short LIKE ? "
                "OR summary_detail LIKE ?)",
            ]
            like_params: list[object] = [
                patient_id, pattern, pattern, pattern,
            ]
            if category:
                like_filters.append("category=?")
                like_params.append(category)
            if status:
                like_filters.append("status=?")
                like_params.append(status)
            like_params.append(max(1, min(int(limit), 1000)))
            rows = self.db.execute(
                "SELECT * FROM clinical_events WHERE "
                + " AND ".join(like_filters)
                + " ORDER BY COALESCE(first_evidence_date, '9999') LIMIT ?",
                tuple(like_params),
            ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def count_by_patient(self, patient_id: str) -> int:
        return self.db.execute(
            """SELECT COUNT(*) FROM clinical_events
               WHERE patient_id=? AND review_status<>'rejected'""",
            (patient_id,),
        ).fetchone()[0]

    def save_medication_courses(
        self, patient_id: str, courses: Iterable[MedicationCourse]
    ) -> None:
        courses = list(courses)
        with self.db:
            self.db.execute(
                "DELETE FROM medication_courses WHERE patient_id=?",
                (patient_id,),
            )
            for course in courses:
                self.db.execute(
                    """INSERT INTO medication_courses
                       (course_id, patient_id, normalized_name,
                        original_names_json, indication, intent,
                        lifecycle_status, start_date, end_date, dose, route,
                        frequency, adherence, episode_id, event_ids_json,
                        data_json, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        course.course_id, course.patient_id,
                        course.normalized_name, _json(course.original_names),
                        course.indication, course.intent,
                        course.lifecycle_status, course.start_date,
                        course.end_date, course.dose, course.route,
                        course.frequency, course.adherence, course.episode_id,
                        _json(course.event_ids), _json(course.data),
                        course.created_at, course.updated_at,
                    ),
                )

    def save_oncology_lines(
        self, patient_id: str, lines: Iterable[OncologyLine]
    ) -> None:
        lines = list(lines)
        with self.db:
            self.db.execute(
                "DELETE FROM oncology_lines WHERE patient_id=?", (patient_id,)
            )
            for line in lines:
                self.db.execute(
                    """INSERT INTO oncology_lines
                       (line_id, patient_id, line_label, regimen_json, setting,
                        intent, start_date, end_date, status, cycles_json,
                        modifications_json, toxicities_json, responses_json,
                        progression_event_id, event_ids_json, created_at,
                        updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        line.line_id, line.patient_id, line.line_label,
                        _json(line.regimen), line.setting, line.intent,
                        line.start_date, line.end_date, line.status,
                        _json(line.cycles), _json(line.modifications),
                        _json(line.toxicities), _json(line.responses),
                        line.progression_event_id, _json(line.event_ids),
                        line.created_at, line.updated_at,
                    ),
                )

    def save_lab_trends(
        self, patient_id: str, trends: Iterable[LabTrend]
    ) -> None:
        trends = list(trends)
        with self.db:
            self.db.execute(
                "DELETE FROM lab_trends WHERE patient_id=?", (patient_id,)
            )
            for trend in trends:
                self.db.execute(
                    """INSERT INTO lab_trends
                       (trend_id, patient_id, normalized_name, summary,
                        start_date, end_date, direction, severity, resolved,
                        lab_value_ids_json, event_id, data_json, created_at,
                        updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        trend.trend_id, trend.patient_id,
                        trend.normalized_name, trend.summary,
                        trend.start_date, trend.end_date, trend.direction,
                        trend.severity, 1 if trend.resolved else 0,
                        _json(trend.lab_value_ids), trend.event_id,
                        _json(trend.data), trend.created_at, trend.updated_at,
                    ),
                )

    @staticmethod
    def _row_to_event(row) -> ClinicalEvent:
        return ClinicalEvent(
            event_id=row["event_id"],
            patient_id=row["patient_id"],
            episode_id=row["episode_id"],
            category=row["category"],
            canonical_entity=row["canonical_entity"],
            summary_short=row["summary_short"],
            summary_detail=row["summary_detail"],
            anatomical_site=row["anatomical_site"],
            laterality=row["laterality"],
            severity=row["severity"],
            significance=row["significance"],
            status=row["status"],
            certainty=row["certainty"],
            assertion=row["assertion"],
            first_evidence_date=row["first_evidence_date"],
            first_documented_date=row["first_documented_date"],
            date_end=row["date_end"],
            date_precision=row["date_precision"],
            confidence=row["confidence"],
            review_status=row["review_status"],
            structured_data=_loads(row["structured_data_json"], {}),
            model_name=row["model_name"],
            prompt_version=row["prompt_version"],
            schema_version=row["schema_version"],
            version=row["version"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


def _fts_match_query(query: str) -> str:
    stopwords = {
        "alla", "alle", "dalla", "delle", "della", "degli", "quale",
        "quali", "quando", "come", "sono", "stato", "stata", "paziente",
        "elenca", "descrivi", "tutti", "tutte", "clinica", "clinico",
    }
    tokens = [
        token for token in __import__("re").findall(
            r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+", query.casefold()
        )
        if len(token) >= 3 and token not in stopwords
    ]
    if not tokens:
        tokens = [
            token for token in __import__("re").findall(r"[A-Za-z0-9]+", query)
            if token
        ]
    # Quoted tokens prevent FTS operators in user text from changing syntax.
    return " OR ".join(f'"{token.replace(chr(34), chr(34) * 2)}"*' for token in tokens)
