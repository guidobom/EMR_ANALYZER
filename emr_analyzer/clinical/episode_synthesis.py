"""LLM-assisted assembly of atomic draft events into clinical episodes.

The model never receives the whole patient corpus and never decides which
sources exist.  Deterministic blocking creates small, disjoint candidate
groups; the model may absorb observations into a problem episode or link an
autonomous treatment/procedure episode.  Every generated claim must cite an
immutable evidence id and all output is validated before it can alter a draft.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import inspect
import json
from typing import Any, Iterable

from .consolidation import build_updates, canonicalize_entity, stable_id
from .correlation import clinical_system
from .temporal import date_sort_key, temporal_distance_days
from ..models.clinical_evidence import ClinicalEvidence
from ..models.clinical_registry import ClinicalEventRelation, EventEvidenceLink
from ..prompt_catalog import load_prompt, prompts_digest


EPISODE_ASSEMBLY_PROMPT_VERSION = "episode_assembly_it_v3"

_SYSTEM_PROMPT = load_prompt("episode_assembly_system")
_TASK = load_prompt(
    "episode_assembly_task",
    required_markers=(
        "candidate_group_id", "absorbed_event_ids", "related_event_ids",
        "evidence_ids",
    ),
)

_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "drafts": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "anchor_event_id": {"type": "string"},
                    "absorbed_event_ids": {
                        "type": "array", "items": {"type": "string"},
                    },
                    "related_event_ids": {
                        "type": "array", "items": {"type": "string"},
                    },
                    "problem_label": {"type": "string"},
                    "category": {
                        "type": "string",
                        "enum": [
                            "diagnosis", "symptom", "clinical_sign",
                            "clinical_syndrome", "toxicity", "adverse_event",
                            "comorbidity", "laboratory_finding",
                            "laboratory_trend", "imaging_finding",
                            "histopathology", "biomarker", "other",
                        ],
                    },
                    "claims": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "text": {"type": "string"},
                                "evidence_ids": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": ["text", "evidence_ids"],
                        },
                    },
                },
                "required": [
                    "anchor_event_id", "absorbed_event_ids",
                    "related_event_ids", "problem_label", "category", "claims",
                ],
            },
        },
    },
    "required": ["drafts"],
}
EPISODE_ASSEMBLY_PROMPT_DIGEST = prompts_digest(
    _SYSTEM_PROMPT, _TASK, schema=_SCHEMA
)

_AUTONOMOUS_CATEGORIES = {
    "medication", "oncology_treatment_line", "procedure", "surgery",
    "hospitalization", "discharge", "response", "progression",
    "recommendation", "care_plan", "follow_up", "allergy", "vaccination",
    "family_history", "risk_factor", "functional_status", "other",
}
_ANCHOR_CATEGORIES = {
    "diagnosis", "symptom", "clinical_sign", "clinical_syndrome",
    "toxicity", "adverse_event", "comorbidity", "laboratory_finding",
    "laboratory_trend", "imaging_finding", "histopathology", "biomarker",
}
_HUMAN_LOCKED = {"accepted", "corrected", "rejected"}

_SYSTEM_WINDOWS = {
    "respiratorio": 21,
    "tiroideo": 60,
    "epatico": 30,
    "renale": 30,
    "ematologico": 45,
    "gastrointestinale": 21,
    "neurologico": 21,
    "cardiovascolare": 14,
    "cutaneo": 21,
    "non_classificato": 30,
}


@dataclass(frozen=True, slots=True)
class EpisodeSynthesisStats:
    candidate_groups: int = 0
    llm_calls: int = 0
    cached_groups: int = 0
    episodes_absorbed: int = 0
    autonomous_links: int = 0


def _bundle_items(bundle, evidence_by_id) -> list[ClinicalEvidence]:
    return [
        evidence_by_id[link.evidence_id]
        for link in bundle.links if link.evidence_id in evidence_by_id
    ]


def _bundle_system(bundle, evidence_by_id) -> str:
    systems = [
        system for item in _bundle_items(bundle, evidence_by_id)
        if (system := clinical_system(item))
    ]
    if systems:
        return Counter(systems).most_common(1)[0][0]
    entity = canonicalize_entity(bundle.event.canonical_entity)
    return entity.split("_", 1)[0] if entity else "non_classificato"


def _bundle_documents(bundle, evidence_by_id) -> set[str]:
    return {
        item.document_id for item in _bundle_items(bundle, evidence_by_id)
    }


def plan_candidate_groups(
    bundles: Iterable,
    *,
    evidence_by_id: dict[str, ClinicalEvidence],
    maximum_group_size: int = 24,
    window_days: int | None = None,
    maximum_span_days: int = 180,
) -> list[list]:
    """Create disjoint clinical/time blocks; unrelated events never reach LLM."""
    by_system: dict[str, list] = defaultdict(list)
    for bundle in bundles:
        by_system[_bundle_system(bundle, evidence_by_id)].append(bundle)
    result: list[list] = []
    for related in by_system.values():
        related.sort(key=lambda bundle: (
            date_sort_key(
                bundle.event.first_evidence_date
                or bundle.event.first_documented_date
            ),
            bundle.event.event_id,
        ))
        current: list = []
        anchor_date: str | None = None
        last_date: str | None = None
        documents: set[str] = set()
        system_window = int(
            window_days
            if window_days is not None
            else _SYSTEM_WINDOWS.get(
                _bundle_system(related[0], evidence_by_id), 30
            )
        )
        for bundle in related:
            date = (
                bundle.event.first_evidence_date
                or bundle.event.first_documented_date
            )
            bundle_docs = _bundle_documents(bundle, evidence_by_id)
            gap = temporal_distance_days(last_date, date)
            span = temporal_distance_days(anchor_date, date)
            same_document = bool(documents & bundle_docs)
            must_split = bool(
                current and (
                    len(current) >= maximum_group_size
                    or (
                        gap is not None and gap > system_window
                        and not same_document
                    )
                    or (
                        span is not None and span > maximum_span_days
                        and not same_document
                    )
                )
            )
            if must_split:
                if _useful_candidate_group(current):
                    result.append(current)
                current, anchor_date, last_date, documents = (
                    [], None, None, set()
                )
            current.append(bundle)
            documents.update(bundle_docs)
            if anchor_date is None and date:
                anchor_date = date
            if date:
                last_date = date
        if _useful_candidate_group(current):
            result.append(current)
    return result


def _useful_candidate_group(group: list) -> bool:
    if len(group) < 2:
        return False
    categories = {bundle.event.category for bundle in group}
    return bool(categories & _ANCHOR_CATEGORIES) and (
        len(categories) >= 2 or len(group) >= 3
    )


def _candidate_payload(group: list, evidence_by_id) -> list[dict]:
    payload = []
    for bundle in group:
        included_ids = {
            link.evidence_id for link in bundle.links
            if link.included_in_summary
        }
        items = sorted(
            _bundle_items(bundle, evidence_by_id),
            key=lambda item: (
                item.evidence_id not in included_ids,
                canonicalize_entity(item.normalized_entity)
                != canonicalize_entity(bundle.event.canonical_entity),
                -(item.confidence or 0.0),
                item.evidence_id,
            ),
        )[:4]
        payload.append({
            "event_id": bundle.event.event_id,
            "date": bundle.event.first_evidence_date,
            "category": bundle.event.category,
            "entity": bundle.event.canonical_entity,
            "summary": str(bundle.event.summary_short or "")[:240],
            "status": bundle.event.status,
            "assertion": bundle.event.assertion,
            "evidence": [{
                "evidence_id": item.evidence_id,
                "entity": item.normalized_entity,
                "date": item.observed_date or item.document_date,
                "assertion": item.assertion,
                "certainty": item.certainty,
                "status": item.clinical_status,
                "value": (
                    item.value_text
                    if item.value_text is not None else item.numeric_value
                ),
                "unit": item.unit,
                "site": item.anatomical_site,
                "side": item.laterality,
                "source": " ".join(item.source_text.split())[:180],
            } for item in items],
            "derived_features": {
                "lab_phases": (
                    bundle.event.structured_data.get("phases") or []
                ),
                "therapy_status_history": (
                    bundle.event.structured_data.get("status_history") or []
                ),
            },
        })
    return payload


def _fingerprint(payload: list[dict], model_identity: str) -> str:
    return hashlib.sha256(json.dumps(
        {
            "prompt": EPISODE_ASSEMBLY_PROMPT_VERSION,
            "prompt_digest": EPISODE_ASSEMBLY_PROMPT_DIGEST,
            "model": model_identity,
            "events": payload,
        },
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _pack_uncached_groups(
    entries: list[tuple[list, list[dict], str]],
    *,
    maximum_events: int = 32,
    maximum_groups: int = 6,
) -> list[list[tuple[list, list[dict], str]]]:
    """Pack disjoint groups into fewer LLM calls without merging contexts."""
    batches: list[list[tuple[list, list[dict], str]]] = []
    current: list[tuple[list, list[dict], str]] = []
    event_count = 0
    for entry in entries:
        group_size = len(entry[0])
        if current and (
            event_count + group_size > maximum_events
            or len(current) >= maximum_groups
        ):
            batches.append(current)
            current = []
            event_count = 0
        current.append(entry)
        event_count += group_size
    if current:
        batches.append(current)
    return batches


class ClinicalEpisodeSynthesizer:
    """Validated LLM enrichment of deterministic draft event blocks."""

    def __init__(self, llm=None, existing_events=()):
        self.llm = llm
        self.existing_events = list(existing_events or [])
        self._cache = {
            str(event.structured_data.get("episode_assembly_fingerprint")): (
                event.structured_data.get("episode_assembly_decision")
            )
            for event in self.existing_events
            if event.structured_data.get("episode_assembly_fingerprint")
            and isinstance(
                event.structured_data.get("episode_assembly_decision"), dict
            )
        }

    def synthesize(
        self,
        patient_id: str,
        bundles: Iterable,
        *,
        evidence_by_id: dict[str, ClinicalEvidence],
        num_workers: int = 1,
        progress_callback=None,
    ) -> tuple[list, list[ClinicalEventRelation], EpisodeSynthesisStats]:
        bundles = list(bundles)
        if not self.llm or not getattr(self.llm, "is_available", False):
            return bundles, [], EpisodeSynthesisStats()
        groups = plan_candidate_groups(
            bundles, evidence_by_id=evidence_by_id
        )
        if not groups:
            return bundles, [], EpisodeSynthesisStats()

        model_identity = json.dumps({
            "model": str(getattr(self.llm, "model", "") or ""),
            "temperature": getattr(self.llm, "temperature", None),
            "top_p": getattr(self.llm, "top_p", None),
            "top_k": getattr(self.llm, "top_k", None),
            "seed": getattr(self.llm, "seed", None),
        }, sort_keys=True, separators=(",", ":"))
        entries = []
        cached_resolved = []
        uncached = []
        cached = 0
        for group in groups:
            payload = _candidate_payload(group, evidence_by_id)
            fingerprint = _fingerprint(payload, model_identity)
            entry = (group, payload, fingerprint)
            entries.append(entry)
            cached_decision = self._cache.get(fingerprint)
            if cached_decision is not None:
                cached += 1
                cached_resolved.append((
                    group,
                    fingerprint,
                    validate_episode_decision(
                        cached_decision,
                        group,
                        evidence_by_id=evidence_by_id,
                        allowed_evidence_ids=_payload_evidence_ids(payload),
                    ),
                ))
            else:
                uncached.append(entry)

        call_jobs = _pack_uncached_groups(uncached)

        def resolve_batch(batch):
            raw = self._resolve(batch)
            drafts = raw.get("drafts", []) if isinstance(raw, dict) else []
            if not isinstance(drafts, list):
                drafts = []
            result = []
            for group, payload, fingerprint in batch:
                event_ids = {bundle.event.event_id for bundle in group}
                group_raw = {
                    "drafts": [
                        draft for draft in drafts
                        if isinstance(draft, dict)
                        and draft.get("anchor_event_id") in event_ids
                    ]
                }
                result.append((
                    group,
                    fingerprint,
                    validate_episode_decision(
                        group_raw,
                        group,
                        evidence_by_id=evidence_by_id,
                        allowed_evidence_ids=_payload_evidence_ids(payload),
                    ),
                ))
            return result

        resolved = list(cached_resolved)
        workers = max(1, min(int(num_workers or 1), len(call_jobs) or 1))
        if workers == 1:
            completed_groups = cached
            for batch in call_jobs:
                batch_result = resolve_batch(batch)
                resolved.extend(batch_result)
                completed_groups += len(batch_result)
                if progress_callback:
                    progress_callback(completed_groups, len(groups))
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                batch_results = list(pool.map(resolve_batch, call_jobs))
            resolved.extend(
                item for batch_result in batch_results for item in batch_result
            )
            if progress_callback:
                progress_callback(len(groups), len(groups))

        decisions = {
            fingerprint: decision
            for _, fingerprint, decision in resolved
        }

        active = {bundle.event.event_id: bundle for bundle in bundles}
        relations: list[ClinicalEventRelation] = []
        absorbed_count = link_count = 0
        for group, _, fingerprint in entries:
            decision = decisions.get(fingerprint, {"drafts": []})
            absorbed, linked, new_relations = _apply_decision(
                patient_id,
                active,
                decision,
                fingerprint=fingerprint,
                evidence_by_id=evidence_by_id,
            )
            absorbed_count += absorbed
            link_count += linked
            relations.extend(new_relations)
            # Cache negative/no-op decisions as well.  Candidate groups are
            # disjoint, therefore one surviving generated event is a safe
            # durable cache owner for the whole group.
            cache_owner = next((
                active[bundle.event.event_id]
                for bundle in group
                if bundle.event.event_id in active
                and active[bundle.event.event_id].event.review_status
                not in _HUMAN_LOCKED
            ), None)
            if cache_owner is not None:
                cache_owner.event.structured_data.update({
                    "episode_assembly_fingerprint": fingerprint,
                    "episode_assembly_decision": decision,
                    "episode_assembly_prompt_version": (
                        EPISODE_ASSEMBLY_PROMPT_VERSION
                    ),
                })
        relations = [
            relation for relation in relations
            if relation.source_event_id in active
            and relation.target_event_id in active
        ]
        final = [
            bundle for bundle in bundles
            if bundle.event.event_id in active
        ]
        return final, relations, EpisodeSynthesisStats(
            candidate_groups=len(groups),
            llm_calls=len(call_jobs),
            cached_groups=cached,
            episodes_absorbed=absorbed_count,
            autonomous_links=link_count,
        )

    def _resolve(
        self, batch: list[tuple[list, list[dict], str]]
    ) -> dict:
        candidate_groups = [{
            "candidate_group_id": fingerprint[:16],
            "events": payload,
        } for _, payload, fingerprint in batch]
        event_count = sum(len(payload) for _, payload, _ in batch)
        prompt = (
            _TASK + "\n\nGRUPPI CANDIDATI:\n"
            + json.dumps(
                candidate_groups,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
        try:
            generator = self.llm.generate_structured
            parameters = inspect.signature(generator).parameters
            kwargs = {}
            if "max_tokens" in parameters or any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            ):
                kwargs["max_tokens"] = min(
                    int(getattr(self.llm, "max_output_tokens", 4096) or 4096),
                    max(1536, event_count * 180),
                )
            result = generator(prompt, _SYSTEM_PROMPT, _SCHEMA, **kwargs)
            return result if isinstance(result, dict) else {}
        except Exception:
            return {}


def _payload_evidence_ids(payload: list[dict]) -> set[str]:
    return {
        str(item.get("evidence_id"))
        for event in payload
        for item in event.get("evidence", [])
        if item.get("evidence_id")
    }


def validate_episode_decision(
    result: Any,
    group: list,
    *,
    evidence_by_id: dict[str, ClinicalEvidence],
    allowed_evidence_ids: set[str] | None = None,
) -> dict:
    """Reject hallucinated IDs, autonomous absorption and uncited prose."""
    if not isinstance(result, dict) or not isinstance(result.get("drafts"), list):
        return {"drafts": []}
    by_event = {bundle.event.event_id: bundle for bundle in group}
    known_events = set(by_event)
    known_evidence = {
        link.evidence_id for bundle in group for link in bundle.links
        if link.evidence_id in evidence_by_id
    }
    consumed: set[str] = set()
    drafts = []
    for draft in result["drafts"]:
        if not isinstance(draft, dict):
            continue
        anchor_id = str(draft.get("anchor_event_id") or "")
        if anchor_id not in known_events or anchor_id in consumed:
            continue
        anchor = by_event[anchor_id]
        if (
            anchor.event.category not in _ANCHOR_CATEGORIES
            or anchor.event.review_status in _HUMAN_LOCKED
        ):
            continue
        absorbed_ids = draft.get("absorbed_event_ids", [])
        if not isinstance(absorbed_ids, list):
            continue
        absorbed = []
        for event_id in absorbed_ids:
            if event_id not in known_events or event_id in consumed or event_id == anchor_id:
                continue
            candidate = by_event[event_id]
            if candidate.event.category in _AUTONOMOUS_CATEGORIES:
                continue
            if candidate.event.review_status in _HUMAN_LOCKED:
                continue
            absorbed.append(event_id)
        if not absorbed:
            continue
        member_ids = [anchor_id] + absorbed
        member_evidence = {
            link.evidence_id
            for event_id in member_ids
            for link in by_event[event_id].links
            if link.evidence_id in known_evidence
        }
        if allowed_evidence_ids is not None:
            member_evidence &= allowed_evidence_ids
        related_ids = draft.get("related_event_ids", [])
        related = [
            event_id for event_id in (
                related_ids if isinstance(related_ids, list) else []
            )
            if event_id in known_events
            and event_id != anchor_id and event_id not in absorbed
        ]
        claims = []
        cited: set[str] = set()
        raw_claims = draft.get("claims", [])
        for claim in raw_claims if isinstance(raw_claims, list) else []:
            if not isinstance(claim, dict):
                continue
            text = " ".join(str(claim.get("text") or "").split())
            raw_ids = claim.get("evidence_ids", [])
            if (
                not isinstance(raw_ids, list)
                or not raw_ids
                or any(evidence_id not in member_evidence for evidence_id in raw_ids)
            ):
                continue
            ids = list(dict.fromkeys(raw_ids))
            if text and ids and not any(evidence_id in text for evidence_id in known_evidence):
                claims.append({"text": text, "evidence_ids": ids})
                cited.update(ids)
        every_member_cited = True
        for event_id in member_ids:
            member_links = by_event[event_id].links
            preferred = {
                link.evidence_id for link in member_links
                if link.included_in_summary
            } or {link.evidence_id for link in member_links}
            if not (preferred & cited):
                every_member_cited = False
                break
        if not claims or not every_member_cited:
            continue
        consumed.update(member_ids)
        drafts.append({
            "anchor_event_id": anchor_id,
            "absorbed_event_ids": absorbed,
            "related_event_ids": list(dict.fromkeys(related)),
            "problem_label": " ".join(
                str(draft.get("problem_label") or anchor.event.canonical_entity).split()
            )[:240],
            "category": (
                draft.get("category")
                if draft.get("category") in _ANCHOR_CATEGORIES | {"other"}
                else anchor.event.category
            ),
            "claims": claims,
        })
    return {"drafts": drafts}


def _apply_decision(
    patient_id: str,
    active: dict[str, Any],
    decision: dict,
    *,
    fingerprint: str,
    evidence_by_id: dict[str, ClinicalEvidence],
) -> tuple[int, int, list[ClinicalEventRelation]]:
    absorbed_count = linked_count = 0
    relations: list[ClinicalEventRelation] = []
    for draft in decision.get("drafts", []):
        anchor_id = draft["anchor_event_id"]
        anchor = active.get(anchor_id)
        if anchor is None:
            continue
        absorbed = [
            active[event_id] for event_id in draft["absorbed_event_ids"]
            if event_id in active
        ]
        if not absorbed:
            continue
        group = [anchor] + absorbed
        all_links = {link.evidence_id: link for bundle in group for link in bundle.links}
        anchor.links = []
        for evidence_id, old_link in all_links.items():
            relation = old_link.relation if old_link.relation in {
                "supports", "contradicts", "excluded", "correlated", "updates"
            } else "supports"
            anchor.links.append(EventEvidenceLink(
                link_id=stable_id("LNK", anchor_id, evidence_id, relation),
                event_id=anchor_id,
                evidence_id=evidence_id,
                relation=relation,
                role=old_link.role,
                relation_confidence=old_link.relation_confidence,
                rationale=old_link.rationale or "Evidenza incorporata nell'episodio",
                included_in_summary=old_link.included_in_summary,
            ))
        evidence_items = [
            evidence_by_id[evidence_id]
            for evidence_id in all_links if evidence_id in evidence_by_id
        ]
        dates = [
            item.observed_date or item.document_date for item in evidence_items
            if item.observed_date or item.document_date
        ]
        earliest = min(dates, key=date_sort_key) if dates else anchor.event.first_evidence_date
        claims = draft["claims"]
        cited_evidence_ids = {
            evidence_id for claim in claims
            for evidence_id in claim["evidence_ids"]
        }
        for link in anchor.links:
            if link.evidence_id in cited_evidence_ids:
                link.included_in_summary = True
        summary = "; ".join(claim["text"] for claim in claims)
        event = anchor.event
        event.canonical_entity = canonicalize_entity(draft["problem_label"]) or event.canonical_entity
        event.category = draft["category"]
        event.summary_short = summary[:500]
        event.summary_detail = summary
        event.first_evidence_date = earliest
        event.review_status = "pending"
        event.structured_data.update({
            "evidence_ids": list(all_links),
            "claims": claims,
            "absorbed_event_ids": [bundle.event.event_id for bundle in absorbed],
            "episode_assembly_fingerprint": fingerprint,
            "episode_assembly_decision": decision,
            "episode_assembly_prompt_version": EPISODE_ASSEMBLY_PROMPT_VERSION,
        })
        anchor.episode.canonical_entity = event.canonical_entity
        anchor.episode.category = event.category
        anchor.episode.onset_date = earliest
        anchor.updates = build_updates(anchor_id, evidence_items, earliest)
        for bundle in absorbed:
            active.pop(bundle.event.event_id, None)
            absorbed_count += 1
        for related_id in draft["related_event_ids"]:
            if related_id not in active:
                continue
            relation_type = "related_episode"
            relations.append(ClinicalEventRelation(
                relation_id=stable_id("REL", anchor_id, related_id, relation_type),
                patient_id=patient_id,
                source_event_id=anchor_id,
                target_event_id=related_id,
                relation_type=relation_type,
                confidence=0.8,
                rationale="Episodio autonomo clinicamente collegato dal modello",
                review_status="pending",
            ))
            linked_count += 1
    return absorbed_count, linked_count, relations
