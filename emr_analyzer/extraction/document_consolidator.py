"""Consolidate atomic evidence into one validated document-level block."""

from __future__ import annotations

import json
import re

from ..models.clinical_evidence import ClinicalEvidence
from ..models.document_projection import (
    DocumentClinicalProjection,
    DocumentObservation,
    DocumentObservationRelationship,
    DocumentProjectionConflict,
    DocumentSourceSpan,
)


PROMPT_VERSION = "document_consolidation_v1"
SCHEMA_VERSION = "1.0"

RELATIONSHIP_TYPES = {
    "treated_with", "response_to", "caused_by", "follow_up_of",
    "supports", "conflicts_with", "temporally_associated_with",
}

CONSOLIDATION_SCHEMA = {
    "type": "object",
    "properties": {
        "observation_groups": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "group_key": {"type": "string"},
                    "evidence_ids": {
                        "type": "array", "items": {"type": "string"},
                    },
                    "canonical_evidence_id": {"type": "string"},
                    "group_relation": {
                        "type": "string",
                        "enum": ["same_entity", "synonym", "evolution", "duplicate"],
                    },
                },
                "required": [
                    "group_key", "evidence_ids", "canonical_evidence_id",
                    "group_relation",
                ],
            },
        },
        "relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "from_group_key": {"type": "string"},
                    "to_group_key": {"type": "string"},
                    "relationship_type": {
                        "type": "string",
                        "enum": sorted(RELATIONSHIP_TYPES),
                    },
                    "evidence_ids": {
                        "type": "array", "items": {"type": "string"},
                    },
                    "confidence": {"type": "number"},
                },
                "required": [
                    "from_group_key", "to_group_key", "relationship_type",
                    "evidence_ids", "confidence",
                ],
            },
        },
        "conflicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "evidence_ids": {
                        "type": "array", "items": {"type": "string"},
                    },
                    "conflict_type": {
                        "type": "string",
                        "enum": [
                            "opposite_assertion", "incompatible_status",
                            "incompatible_value", "incompatible_date",
                        ],
                    },
                },
                "required": ["evidence_ids", "conflict_type"],
            },
        },
    },
    "required": ["observation_groups", "relationships", "conflicts"],
}


class DocumentClinicalConsolidator:
    """Build a lossless document projection, optionally assisted by Ollama."""

    def __init__(self, llm_client=None):
        self.llm = llm_client

    def consolidate(
        self,
        evidence: list[ClinicalEvidence],
        patient_id: str,
        document_id: str,
        document_date: str | None = None,
        document_type: str | None = None,
    ) -> DocumentClinicalProjection:
        ordered = sorted(evidence, key=self._evidence_sort_key)
        warnings = []
        if self._can_use_llm(ordered):
            try:
                observations, relationships, conflicts = self._llm_consolidate(
                    ordered
                )
                method = "llm_validated"
            except Exception as exc:
                warnings.append(
                    "Consolidamento LLM non disponibile; applicato fallback "
                    f"deterministico ({type(exc).__name__})"
                )
                observations = self._deterministic_observations(ordered)
                relationships, conflicts = [], self._deterministic_conflicts(
                    observations
                )
                method = "deterministic_fallback"
        else:
            observations = self._deterministic_observations(ordered)
            relationships, conflicts = [], self._deterministic_conflicts(
                observations
            )
            method = "deterministic"

        return DocumentClinicalProjection(
            patient_id=patient_id,
            document_id=document_id,
            document_date=document_date,
            document_type=document_type,
            observations=observations,
            relationships=relationships,
            conflicts=conflicts,
            consolidation_method=method,
            model_name=(self.llm.model if method == "llm_validated" else None),
            prompt_version=(PROMPT_VERSION if method == "llm_validated" else None),
            schema_version=SCHEMA_VERSION,
            warnings=warnings,
        )

    def _can_use_llm(self, evidence: list[ClinicalEvidence]) -> bool:
        if not self.llm or not self.llm.is_available:
            return False
        candidates = [item for item in evidence if self._is_llm_candidate(item)]
        return len(candidates) >= 2

    @staticmethod
    def _is_llm_candidate(item: ClinicalEvidence) -> bool:
        return not (
            item.category == "laboratory_finding"
            and item.clinical_status == "within_range"
        )

    def _llm_consolidate(self, evidence: list[ClinicalEvidence]):
        candidates = [item for item in evidence if self._is_llm_candidate(item)]
        compact = [self._compact_item(item) for item in candidates]
        prompt = f"""Raggruppa le evidenze cliniche di UN SOLO documento.

Vincoli assoluti:
- usa esclusivamente gli evidence_id forniti;
- ogni evidence_id deve comparire in un solo observation_group;
- raggruppa solo evidenze della stessa categoria che descrivono lo stesso
  fenomeno clinico, sinonimi oppure la sua evoluzione tra pagine diverse;
- terapie, sintomi, reperti e diagnosi distinti devono restare in gruppi distinti;
- usa relationships per collegare gruppi differenti (per esempio un sintomo
  trattato con un farmaco o un reperto che evolve nel tempo);
- non creare diagnosi, eventi, valori, date o interpretazioni nuove;
- segnala come conflict solo contraddizioni reali, non una normale evoluzione;
- restituisci esclusivamente JSON conforme allo schema.

EVIDENZE:
{json.dumps(compact, ensure_ascii=False, separators=(",", ":"))}
"""
        data = self.llm.generate_structured(
            prompt,
            (
                "Sei un consolidatore clinico documentale. Puoi soltanto "
                "raggruppare o collegare evidenze già verificate tramite i "
                "loro ID; non puoi introdurre nuovi fatti."
            ),
            CONSOLIDATION_SCHEMA,
        )
        if not isinstance(data, dict):
            raise ValueError("Risposta di consolidamento non strutturata")

        all_by_id = {item.evidence_id: item for item in evidence}
        candidate_ids = {item.evidence_id for item in candidates}
        used_ids = set()
        observations = []
        key_to_observation = {}

        for group in data.get("observation_groups", []):
            group_key = str(group.get("group_key") or "").strip()
            ids = list(dict.fromkeys(
                evidence_id for evidence_id in group.get("evidence_ids", [])
                if evidence_id in candidate_ids and evidence_id not in used_ids
            ))
            canonical_id = group.get("canonical_evidence_id")
            if (not group_key or group_key in key_to_observation or
                    not ids or canonical_id not in ids):
                continue
            items = [all_by_id[evidence_id] for evidence_id in ids]
            # Cross-category facts are linked, never collapsed into one fact.
            if len({item.category for item in items}) != 1:
                continue
            observation = self._observation_from_items(
                items,
                canonical_id=canonical_id,
                grouping_method="llm_validated",
                group_relation=group.get("group_relation"),
            )
            if len({
                self._normalize_entity(item.normalized_entity) for item in items
            }) > 1:
                observation.requires_review = True
            observations.append(observation)
            key_to_observation[group_key] = observation
            used_ids.update(ids)

        # Invalid, omitted and deterministic-laboratory evidence must survive.
        leftovers = [
            item for item in evidence if item.evidence_id not in used_ids
        ]
        observations.extend(self._deterministic_observations(leftovers))

        relationships = []
        seen_relationships = set()
        for raw in data.get("relationships", []):
            source = key_to_observation.get(raw.get("from_group_key"))
            target = key_to_observation.get(raw.get("to_group_key"))
            relation = raw.get("relationship_type")
            if not source or not target or source is target:
                continue
            if relation not in RELATIONSHIP_TYPES:
                continue
            relationship_evidence = set(
                source.evidence_ids + target.evidence_ids
            )
            valid_evidence_ids = list(dict.fromkeys(
                evidence_id for evidence_id in raw.get("evidence_ids", [])
                if evidence_id in relationship_evidence
            ))
            if not valid_evidence_ids:
                continue
            key = (source.observation_id, target.observation_id, relation)
            if key in seen_relationships:
                continue
            seen_relationships.add(key)
            relationships.append(DocumentObservationRelationship(
                from_observation_id=source.observation_id,
                to_observation_id=target.observation_id,
                relationship_type=relation,
                evidence_ids=valid_evidence_ids,
                confidence=self._confidence(raw.get("confidence")),
            ))

        conflicts = []
        for raw in data.get("conflicts", []):
            ids = list(dict.fromkeys(
                evidence_id for evidence_id in raw.get("evidence_ids", [])
                if evidence_id in all_by_id
            ))
            conflict_type = raw.get("conflict_type")
            if len(ids) < 2 or conflict_type not in {
                "opposite_assertion", "incompatible_status",
                "incompatible_value", "incompatible_date",
            }:
                continue
            conflicts.append(DocumentProjectionConflict(ids, conflict_type))
            for observation in observations:
                if set(observation.evidence_ids).intersection(ids):
                    observation.requires_review = True

        # No valid grouping means the response cannot influence persistence.
        if candidates and not key_to_observation:
            raise ValueError("Nessun gruppo LLM valido")
        return observations, relationships, conflicts

    def _deterministic_observations(
        self, evidence: list[ClinicalEvidence]
    ) -> list[DocumentObservation]:
        groups = {}
        for item in evidence:
            key = (
                item.category,
                self._normalize_entity(item.normalized_entity),
            )
            groups.setdefault(key, []).append(item)
        return [
            self._observation_from_items(
                items, grouping_method="deterministic_exact"
            )
            for items in groups.values()
        ]

    def _observation_from_items(
        self,
        items: list[ClinicalEvidence],
        canonical_id: str | None = None,
        grouping_method: str = "deterministic_exact",
        group_relation: str | None = None,
    ) -> DocumentObservation:
        items = sorted(items, key=self._evidence_sort_key)
        canonical = next(
            (item for item in items if item.evidence_id == canonical_id),
            items[-1],
        )
        dates = [item.observed_date for item in items if item.observed_date]
        values = [item.value_text for item in items if item.value_text is not None]
        numeric_values = [
            item.numeric_value for item in items if item.numeric_value is not None
        ]
        units = [item.unit for item in items if item.unit]
        statuses = [item.clinical_status for item in items if item.clinical_status]
        spans = [DocumentSourceSpan(
            evidence_id=item.evidence_id,
            page=item.source_page,
            text=item.source_text,
            bbox=item.bbox,
        ) for item in items]
        return DocumentObservation(
            category=canonical.category,
            normalized_entity=canonical.normalized_entity,
            evidence_ids=[item.evidence_id for item in items],
            source_spans=spans,
            observed_start_date=min(dates) if dates else None,
            observed_end_date=max(dates) if dates else None,
            assertions=list(dict.fromkeys(item.assertion for item in items)),
            temporalities=list(dict.fromkeys(item.temporality for item in items)),
            clinical_status=statuses[-1] if statuses else None,
            value_text=values[-1] if values else None,
            numeric_value=numeric_values[-1] if numeric_values else None,
            unit=units[-1] if units else None,
            confidence=(
                sum(item.confidence for item in items) / len(items)
                if items else 0.0
            ),
            data={
                "grouping_method": grouping_method,
                "group_relation": group_relation,
            },
        )

    @staticmethod
    def _deterministic_conflicts(
        observations: list[DocumentObservation]
    ) -> list[DocumentProjectionConflict]:
        conflicts = []
        negative = {"absent", "negated", "not_present"}
        for observation in observations:
            assertions = set(observation.assertions)
            if (assertions.intersection(negative) and
                    assertions.difference(negative) and
                    observation.observed_start_date == observation.observed_end_date):
                observation.requires_review = True
                conflicts.append(DocumentProjectionConflict(
                    evidence_ids=observation.evidence_ids,
                    conflict_type="opposite_assertion",
                ))
        return conflicts

    @staticmethod
    def _compact_item(item: ClinicalEvidence) -> dict:
        return {
            "evidence_id": item.evidence_id,
            "category": item.category,
            "entity": item.normalized_entity,
            "assertion": item.assertion,
            "temporality": item.temporality,
            "clinical_status": item.clinical_status,
            "date": item.observed_date,
            "value": item.value_text,
            "numeric_value": item.numeric_value,
            "unit": item.unit,
            "page": item.source_page,
            "source_text": item.source_text[:400],
        }

    @staticmethod
    def _normalize_entity(value: str) -> str:
        return re.sub(r"[^a-z0-9à-ÿ]+", " ", value.casefold()).strip()

    @staticmethod
    def _evidence_sort_key(item: ClinicalEvidence):
        return (
            item.observed_date or "",
            item.source_page or 0,
            item.evidence_id,
        )

    @staticmethod
    def _confidence(value) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.0
