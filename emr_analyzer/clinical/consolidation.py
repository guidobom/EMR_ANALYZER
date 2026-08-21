"""Evidence clustering, episode reconstruction and cautious clinical fusion."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import hashlib
import inspect
import json
import re
import unicodedata
from typing import Any, Iterable
import uuid

from .temporal import date_sort_key, precision_for_iso, temporal_distance_days
from ..models.clinical_evidence import ClinicalEvidence
from ..models.clinical_registry import (
    CERTAINTY_LEVELS,
    ClinicalEpisode,
    ClinicalEvent,
    EventEvidenceLink,
    EventUpdate,
)


_FUSION_SYSTEM_PROMPT = (
    "Fondi evidenze dello stesso possibile evento clinico. Usa solo i dati "
    "forniti; non inventare diagnosi o causalità. Ogni claim cita evidence_ids "
    "validi. Solo JSON."
)

_FUSION_TASK = """Scrivi claim clinici concisi.
- Copri ogni ID in un claim o nei conflitti; ID solo in evidence_ids, mai nel testo.
- Unisci duplicati/complementi; conserva evoluzione e stati terapeutici.
- Non aumentare certezza/gravità. Attivo poi sospeso/risolto è evoluzione:
  usa lo stato più recente, non un conflitto.
- Per vere divergenze descrivi entrambe senza scegliere e marcane gli ID.
- Un sintomo resta soggettivo; reperto != diagnosi; temporalità != causalità;
  non creare date."""

FUSION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "text": {"type": "string"},
                    "evidence_ids": {
                        "type": "array", "items": {"type": "string"},
                    },
                    "certainty": {
                        "type": "string", "enum": list(CERTAINTY_LEVELS),
                    },
                },
                "required": ["text", "evidence_ids", "certainty"],
            },
        },
        "conflicting_evidence_ids": {
            "type": "array", "items": {"type": "string"},
        },
    },
    "required": ["claims", "conflicting_evidence_ids"],
}

FUSION_PROMPT_VERSION = "clinical_fusion_it_v3"
FUSION_PROMPT_DIGEST = hashlib.sha256(
    (
        _FUSION_SYSTEM_PROMPT
        + "\x1f"
        + _FUSION_TASK
        + "\x1f"
        + json.dumps(FUSION_SCHEMA, sort_keys=True)
    ).encode("utf-8")
).hexdigest()


@dataclass(slots=True)
class EvidenceCluster:
    patient_id: str
    category: str
    canonical_entity: str
    evidence: list[ClinicalEvidence] = field(default_factory=list)
    episode_index: int = 1
    conflicts: list[str] = field(default_factory=list)
    event_id: str | None = None
    episode_id: str | None = None
    previous_episode_id: str | None = None


@dataclass(slots=True)
class ConsolidatedBundle:
    episode: ClinicalEpisode
    event: ClinicalEvent
    links: list[EventEvidenceLink]
    updates: list[EventUpdate]


_CATEGORY_ALIASES = {
    "laboratory": "laboratory_finding",
    "treatment": "medication",
    "treatment_interruption": "medication",
    "treatment_completed": "medication",
    "imaging": "imaging_finding",
    "histology": "histopathology",
    "consultation": "care_plan",
}

_ACUTE_CATEGORIES = {
    "symptom", "clinical_sign", "vital_sign", "adverse_event",
    "hospitalization", "discharge", "procedure", "surgery",
}

_WINDOW_DAYS = {
    "symptom": 14,
    "clinical_sign": 14,
    "vital_sign": 7,
    "laboratory_finding": 14,
    "imaging_finding": 30,
    "toxicity": 42,
    "adverse_event": 14,
    "procedure": 3,
    "surgery": 3,
    "hospitalization": 3,
    "discharge": 3,
}

_RESOLUTION_STATUSES = {
    "resolved", "completed", "stopped", "suspended", "cancelled",
    "within_range", "normal", "negative",
}

_ACTIVE_STATUSES = {
    "active", "ongoing", "present", "abnormal", "started", "prescribed",
    "taken", "administered", "positive",
}

_HIGH_RISK_CATEGORIES = {
    "diagnosis", "toxicity", "adverse_event", "progression",
    "oncology_treatment_line", "allergy",
}


class ClinicalConsolidator:
    """Build episodes/events without deleting or rewriting source evidence."""

    def __init__(self, fusion_llm=None, existing_events=None):
        self.fusion = ClinicalFusionEngine(fusion_llm) if fusion_llm else None
        self._existing_event_list = list(existing_events or [])
        self.existing_events = {
            event.event_id: event for event in self._existing_event_list
        }

    def consolidate(
        self,
        patient_id: str,
        evidence: Iterable[ClinicalEvidence],
        *,
        num_workers: int = 1,
        progress_callback=None,
    ) -> list[ConsolidatedBundle]:
        usable = [item for item in evidence if item.patient_id == patient_id]
        clusters = self._cluster_same_events(usable)
        self._assign_stable_ids(clusters)
        workers = max(1, min(int(num_workers or 1), len(clusters) or 1))
        if workers == 1:
            bundles = []
            for index, cluster in enumerate(clusters, start=1):
                bundles.append(self._bundle(cluster))
                if progress_callback:
                    progress_callback(index, len(clusters))
            return bundles

        # Cluster construction and stable IDs remain deterministic.  Only
        # independent prose fusion calls are concurrent; results are restored
        # to cluster order before persistence.
        ordered: list[ConsolidatedBundle | None] = [None] * len(clusters)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_map = {
                pool.submit(self._bundle, cluster): index
                for index, cluster in enumerate(clusters)
            }
            completed = 0
            for future in as_completed(future_map):
                ordered[future_map[future]] = future.result()
                completed += 1
                if progress_callback:
                    progress_callback(completed, len(clusters))
        return [bundle for bundle in ordered if bundle is not None]

    def _assign_stable_ids(self, clusters: list[EvidenceCluster]) -> None:
        """Keep a reviewed episode attached when earlier evidence is added.

        Recurrence numbers are presentation metadata and may change when a
        later document retrodates an older episode.  Existing events are
        therefore matched by shared immutable evidence IDs.  New acute
        episodes use their anchor evidence, while non-acute entities retain
        the historical single-episode identifier for compatibility with
        medication projections.
        """
        existing_by_category: dict[str, list[ClinicalEvent]] = {}
        for event in self._existing_event_list:
            category = _CATEGORY_ALIASES.get(event.category, event.category)
            existing_by_category.setdefault(category, []).append(event)

        used_existing: set[str] = set()
        by_identity: dict[tuple[str, str], list[EvidenceCluster]] = {}
        for cluster in clusters:
            by_identity.setdefault(
                (cluster.category, cluster.canonical_entity), []
            ).append(cluster)

        for identity, related_clusters in by_identity.items():
            previous_episode_id = None
            for cluster in related_clusters:
                evidence_ids = {item.evidence_id for item in cluster.evidence}
                candidates = []
                for event in existing_by_category.get(identity[0], []):
                    if event.event_id in used_existing:
                        continue
                    previous_ids = set(
                        event.structured_data.get("evidence_ids", [])
                    )
                    overlap = len(evidence_ids & previous_ids)
                    if overlap:
                        same_entity = int(
                            canonicalize_entity(event.canonical_entity)
                            == identity[1]
                        )
                        candidates.append((overlap, same_entity, event))
                matched = max(
                    candidates,
                    key=lambda item: (item[0], item[1], item[2].event_id),
                    default=None,
                )
                if matched:
                    existing = matched[2]
                    used_existing.add(existing.event_id)
                    cluster.event_id = existing.event_id
                    cluster.episode_id = existing.episode_id or stable_id(
                        "EPI", existing.event_id
                    )
                else:
                    identity_token = str(cluster.episode_index)
                    if cluster.category in _ACUTE_CATEGORIES:
                        anchor = min(
                            cluster.evidence,
                            key=lambda item: (
                                date_sort_key(
                                    item.observed_date or item.document_date
                                ),
                                item.document_id,
                                item.evidence_id,
                            ),
                        )
                        identity_token = anchor.evidence_id
                    cluster.event_id = stable_id(
                        "EVT", cluster.patient_id, cluster.category,
                        cluster.canonical_entity, identity_token,
                    )
                    cluster.episode_id = stable_id(
                        "EPI", cluster.patient_id, cluster.category,
                        cluster.canonical_entity, identity_token,
                    )
                cluster.previous_episode_id = previous_episode_id
                previous_episode_id = cluster.episode_id

    def _cluster_same_events(
        self, evidence: list[ClinicalEvidence]
    ) -> list[EvidenceCluster]:
        blocks: dict[tuple[str, str], list[ClinicalEvidence]] = {}
        for item in evidence:
            category = _CATEGORY_ALIASES.get(item.category, item.category)
            entity = canonicalize_entity(item.normalized_entity)
            if not entity:
                entity = canonicalize_entity(item.source_text[:120]) or "evento"
            blocks.setdefault((category, entity), []).append(item)

        clusters: list[EvidenceCluster] = []
        for (category, entity), items in sorted(blocks.items()):
            items.sort(key=lambda item: (
                date_sort_key(item.observed_date or item.document_date),
                item.document_id, item.evidence_id,
            ))
            current: EvidenceCluster | None = None
            resolved = False
            episode_index = 1
            for item in items:
                if current is None:
                    current = EvidenceCluster(
                        patient_id=item.patient_id, category=category,
                        canonical_entity=entity, episode_index=episode_index,
                    )
                elif self._starts_new_episode(
                    current, item, resolved=resolved
                ):
                    self._detect_conflicts(current)
                    clusters.append(current)
                    episode_index += 1
                    current = EvidenceCluster(
                        patient_id=item.patient_id, category=category,
                        canonical_entity=entity, episode_index=episode_index,
                    )
                    resolved = False
                current.evidence.append(item)
                if _is_resolution(item):
                    resolved = True
            if current is not None:
                self._detect_conflicts(current)
                clusters.append(current)
        return clusters

    @staticmethod
    def _starts_new_episode(
        cluster: EvidenceCluster,
        item: ClinicalEvidence,
        *,
        resolved: bool,
    ) -> bool:
        if not cluster.evidence:
            return False
        if resolved and not _is_resolution(item) and item.assertion != "absent":
            return True
        category = cluster.category
        if category not in _ACUTE_CATEGORIES:
            return False
        previous = cluster.evidence[-1]
        distance = temporal_distance_days(
            previous.observed_date or previous.document_date,
            item.observed_date or item.document_date,
        )
        window = _WINDOW_DAYS.get(category, 14)
        return distance is not None and distance > window

    @staticmethod
    def _detect_conflicts(cluster: EvidenceCluster) -> None:
        by_date: dict[str, list[ClinicalEvidence]] = {}
        for item in cluster.evidence:
            key = item.observed_date or item.document_date or "unknown"
            by_date.setdefault(key, []).append(item)
        for key, dated in by_date.items():
            assertions = {item.assertion for item in dated}
            if "present" in assertions and "absent" in assertions:
                cluster.conflicts.append(
                    f"Affermazioni discordanti alla data {key}"
                )
            numeric = {
                round(item.numeric_value, 8)
                for item in dated if item.numeric_value is not None
            }
            units = {item.unit for item in dated if item.unit}
            if len(numeric) > 1 and len(units) <= 1:
                cluster.conflicts.append(
                    f"Valori numerici discordanti alla data {key}"
                )

    def _bundle(self, cluster: EvidenceCluster) -> ConsolidatedBundle:
        observed = [
            item.observed_date for item in cluster.evidence
            if item.observed_date
        ]
        documented = [
            item.document_date for item in cluster.evidence if item.document_date
        ]
        first_evidence = min(observed, key=date_sort_key) if observed else (
            min(documented, key=date_sort_key) if documented else None
        )
        first_documented = (
            min(documented, key=date_sort_key) if documented else None
        )
        resolution_dates = [
            item.observed_date or item.document_date
            for item in cluster.evidence if _is_resolution(item)
            and (item.observed_date or item.document_date)
        ]
        resolution_date = (
            max(resolution_dates, key=date_sort_key)
            if resolution_dates else None
        )
        episode_id = cluster.episode_id or stable_id(
            "EPI", cluster.patient_id, cluster.category,
            cluster.canonical_entity, str(cluster.episode_index),
        )
        event_id = cluster.event_id or stable_id(
            "EVT", cluster.patient_id, cluster.category,
            cluster.canonical_entity, str(cluster.episode_index),
        )
        evidence_fingerprint = hashlib.sha256(
            "\x1f".join(sorted(
                _evidence_semantic_token(item)
                for item in cluster.evidence
            )).encode("utf-8")
        ).hexdigest()
        fusion_signature = self.fusion.signature if self.fusion else None
        fusion_fingerprint = (
            hashlib.sha256(
                f"{evidence_fingerprint}\x1f{fusion_signature}".encode("utf-8")
            ).hexdigest()
            if fusion_signature else None
        )
        status = infer_event_status(cluster.evidence)
        precision = (
            best_date_precision(cluster.evidence, first_evidence)
            if observed else precision_for_iso(first_documented)
        )
        first_evidence_basis = (
            "clinical_event_date" if observed else "first_documented_date"
        )
        certainty = infer_certainty(cluster.evidence)
        review_status = "pending" if (
            cluster.conflicts
            or certainty == "inferred"
            or any(item.status == "needs_review" for item in cluster.evidence)
        ) else "auto"

        existing = self.existing_events.get(event_id)
        same_evidence = bool(
            existing
            and existing.structured_data.get("evidence_fingerprint")
            == evidence_fingerprint
        )
        reusable_fusion = bool(
            same_evidence
            and (
                not self.fusion
                or existing.structured_data.get("fusion_fingerprint")
                == fusion_fingerprint
            )
        )
        if existing and reusable_fusion:
            fused = {
                "canonical_entity": existing.canonical_entity,
                "summary_short": existing.summary_short,
                "summary_detail": existing.summary_detail,
                "status": existing.status,
                "certainty": existing.certainty,
                "severity": existing.severity,
                "significance": existing.significance,
                "claims": existing.structured_data.get("claims", []),
                "conflicting_evidence_ids": existing.structured_data.get(
                    "conflicting_evidence_ids", []
                ),
                "valid": existing.structured_data.get(
                    "fusion_validated", True
                ),
            }
        else:
            fused = (
                self.fusion.fuse(cluster)
                if self.fusion and _requires_llm_fusion(cluster)
                else deterministic_fusion(cluster)
            )
        if not fused.get("valid", True):
            review_status = "pending"
        summary_short = fused.get("summary_short") or display_entity(
            cluster.canonical_entity
        )
        summary_detail = fused.get("summary_detail") or summary_short
        canonical_entity = canonicalize_entity(
            fused.get("canonical_entity") or cluster.canonical_entity
        ) or cluster.canonical_entity
        status = normalize_status(fused.get("status") or status)
        certainty = normalize_certainty(fused.get("certainty") or certainty)
        severity = fused.get("severity") or _most_informative(
            item.severity for item in cluster.evidence
        )
        significance = fused.get("significance") or infer_significance(
            cluster.evidence
        )

        episode = ClinicalEpisode(
            episode_id=episode_id,
            patient_id=cluster.patient_id,
            category=cluster.category,
            canonical_entity=canonical_entity,
            onset_date=first_evidence,
            onset_date_end=_earliest_date_end(cluster.evidence, first_evidence),
            onset_precision=precision,
            first_documented_date=first_documented,
            resolution_date=resolution_date,
            status=status,
            recurrence_index=cluster.episode_index,
            previous_episode_id=cluster.previous_episode_id,
            data={
                "conflicts": cluster.conflicts,
                "first_evidence_basis": first_evidence_basis,
            },
        )
        event = ClinicalEvent(
            event_id=event_id,
            patient_id=cluster.patient_id,
            episode_id=episode_id,
            category=cluster.category,
            canonical_entity=canonical_entity,
            summary_short=summary_short[:500],
            summary_detail=summary_detail,
            anatomical_site=_most_informative(
                item.anatomical_site for item in cluster.evidence
            ),
            laterality=_most_informative(
                item.laterality for item in cluster.evidence
            ),
            severity=severity,
            significance=significance,
            status=status,
            certainty=certainty,
            assertion=infer_assertion(cluster.evidence),
            first_evidence_date=first_evidence,
            first_documented_date=first_documented,
            date_end=resolution_date,
            date_precision=precision,
            confidence=calibrated_proxy(cluster.evidence, cluster.conflicts),
            review_status=review_status,
            structured_data={
                "evidence_count": len(cluster.evidence),
                "source_document_count": len({
                    item.document_id for item in cluster.evidence
                }),
                "first_evidence_basis": first_evidence_basis,
                "conflicts": cluster.conflicts,
                "claims": fused.get("claims", []),
                "conflicting_evidence_ids": fused.get(
                    "conflicting_evidence_ids", []
                ),
                "evidence_ids": [item.evidence_id for item in cluster.evidence],
                "evidence_fingerprint": evidence_fingerprint,
                "fusion_fingerprint": (
                    fusion_fingerprint
                    or (
                        existing.structured_data.get("fusion_fingerprint")
                        if existing and reusable_fusion else None
                    )
                ),
                "fusion_signature": (
                    fusion_signature
                    or (
                        existing.structured_data.get("fusion_signature")
                        if existing and reusable_fusion else None
                    )
                ),
                "fusion_validated": bool(fused.get("valid", True)),
            },
            model_name=(
                getattr(self.fusion.llm, "model", None) if self.fusion
                else (existing.model_name if existing and reusable_fusion else None)
            ),
            prompt_version=(
                FUSION_PROMPT_VERSION if self.fusion
                else (
                    existing.prompt_version
                    if existing and reusable_fusion else None
                )
            ),
        )

        conflicting_ids = set(fused.get("conflicting_evidence_ids", []))
        cited_ids = {
            evidence_id
            for claim in fused.get("claims", [])
            for evidence_id in claim.get("evidence_ids", [])
        }
        links = []
        for item in cluster.evidence:
            relation = (
                "contradicts" if item.evidence_id in conflicting_ids
                else "supports"
            )
            links.append(EventEvidenceLink(
                link_id=stable_id(
                    "LNK", event_id, item.evidence_id, relation
                ),
                event_id=event_id,
                evidence_id=item.evidence_id,
                relation=relation,
                relation_confidence=calibrated_proxy([item], []),
                rationale=(
                    "Evidenza discordante da sottoporre a revisione"
                    if relation == "contradicts" else "Evidenza del cluster"
                ),
                included_in_summary=(item.evidence_id in cited_ids),
            ))
        updates = build_updates(event_id, cluster.evidence, first_evidence)
        return ConsolidatedBundle(episode, event, links, updates)


class ClinicalFusionEngine:
    """Cited fusion: every generated claim must name supporting evidence."""

    _SCHEMA = FUSION_SCHEMA

    def __init__(self, llm):
        self.llm = llm

    @property
    def signature(self) -> str:
        """Invalidate cached prose when prompt, model or sampling changes."""
        model_payload = {
            "model": getattr(self.llm, "model", None),
            "temperature": getattr(self.llm, "temperature", None),
            "top_p": getattr(self.llm, "top_p", None),
            "top_k": getattr(self.llm, "top_k", None),
            "seed": getattr(self.llm, "seed", None),
        }
        try:
            info = self.llm.backend.model_info(model_payload["model"]) or {}
        except Exception:
            info = {}
        model_payload.update({
            "file": info.get("file"),
            "size_bytes": info.get("size_bytes"),
            "architecture": info.get("architecture"),
        })
        return hashlib.sha256(
            (
                FUSION_PROMPT_DIGEST
                + "\x1f"
                + json.dumps(model_payload, sort_keys=True)
            ).encode("utf-8")
        ).hexdigest()

    def fuse(self, cluster: EvidenceCluster) -> dict[str, Any]:
        items = [evidence_for_prompt(item) for item in cluster.evidence]
        if len(json.dumps(items, ensure_ascii=False)) <= self._input_budget():
            return self._fuse_payload(cluster, items)

        # Large chronic-event clusters are reduced hierarchically. Every
        # partial claim retains original evidence IDs and the final validator
        # still requires complete coverage of the source cluster.
        batches: list[list[ClinicalEvidence]] = []
        current: list[ClinicalEvidence] = []
        size = 0
        for evidence, item in zip(cluster.evidence, items):
            item_size = len(json.dumps(item, ensure_ascii=False))
            if current and size + item_size > self._input_budget():
                batches.append(current)
                current, size = [], 0
            current.append(evidence)
            size += item_size
        if current:
            batches.append(current)

        partial_claims = []
        partial_conflicts = []
        for batch in batches:
            partial_cluster = EvidenceCluster(
                patient_id=cluster.patient_id,
                category=cluster.category,
                canonical_entity=cluster.canonical_entity,
                evidence=batch,
                episode_index=cluster.episode_index,
            )
            partial = self._fuse_payload(
                partial_cluster,
                [evidence_for_prompt(item) for item in batch],
            )
            if not partial.get("valid", False):
                fallback = deterministic_fusion(cluster)
                fallback["valid"] = False
                return fallback
            partial_claims.extend(partial.get("claims", []))
            partial_conflicts.extend(
                partial.get("conflicting_evidence_ids", [])
            )
        reduced = [
            {
                "claim": claim["text"],
                "evidence_ids": claim["evidence_ids"],
                "certainty": claim["certainty"],
            }
            for claim in partial_claims
        ]
        result = self._generate(
            cluster,
            reduced,
            heading=(
                "CLAIM PARZIALI GIÀ VERIFICATI; riconciliali senza perdere "
                "alcun evidence_id"
            ),
        )
        if isinstance(result, dict):
            result["conflicting_evidence_ids"] = list(dict.fromkeys(
                list(result.get("conflicting_evidence_ids", []))
                + partial_conflicts
            ))
        return validate_fusion_result(result, cluster)

    def _fuse_payload(
        self, cluster: EvidenceCluster, items: list[dict[str, Any]]
    ) -> dict[str, Any]:
        return validate_fusion_result(
            self._generate(cluster, items, heading="EVIDENZE"), cluster
        )

    def _generate(
        self,
        cluster: EvidenceCluster,
        items: list[dict[str, Any]],
        *,
        heading: str,
    ) -> dict[str, Any]:
        prompt = build_fusion_prompt(cluster, items, heading=heading)
        try:
            generator = self.llm.generate_structured
            parameters = inspect.signature(generator).parameters
            if "max_tokens" in parameters or any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            ):
                return generator(
                    prompt, _FUSION_SYSTEM_PROMPT, self._SCHEMA,
                    max_tokens=self._output_budget(items),
                )
            return generator(prompt, _FUSION_SYSTEM_PROMPT, self._SCHEMA)
        except Exception:
            return {}

    def _output_budget(self, items: list[dict[str, Any]]) -> int:
        configured = max(
            256, int(getattr(self.llm, "max_output_tokens", 4096) or 4096)
        )
        # Fusion emits short claims and evidence IDs, never a second narrative.
        estimated = max(768, 384 + len(items) * 96)
        return min(configured, estimated)

    def _input_budget(self) -> int:
        context = int(getattr(self.llm, "context_length", 32768) or 32768)
        output = int(getattr(self.llm, "max_output_tokens", 4096) or 4096)
        usable_tokens = max(3000, context - min(output, context // 2) - 3500)
        return max(8000, int(usable_tokens * 2.2))


def build_fusion_prompt(
    cluster: EvidenceCluster,
    items: list[dict[str, Any]],
    *,
    heading: str = "EVIDENZE",
) -> str:
    """Return the compact prompt; the JSON schema carries enum definitions."""
    return (
        f"EVENTO: categoria={cluster.category}; "
        f"entità={cluster.canonical_entity}; elementi={len(items)}\n\n"
        f"{_FUSION_TASK}\n\n{heading}:\n"
        + json.dumps(items, ensure_ascii=False, separators=(",", ":"))
    )


def validate_fusion_result(
    result: Any, cluster: EvidenceCluster
) -> dict[str, Any]:
    fallback = deterministic_fusion(cluster)
    if not isinstance(result, dict):
        fallback["valid"] = False
        return fallback
    known = {item.evidence_id for item in cluster.evidence}
    claims = []
    all_cited: set[str] = set()
    for claim in result.get("claims", []):
        if not isinstance(claim, dict):
            continue
        text = " ".join(str(claim.get("text") or "").split())
        if any(evidence_id in text for evidence_id in known):
            continue
        ids = [
            evidence_id for evidence_id in claim.get("evidence_ids", [])
            if evidence_id in known
        ] if isinstance(claim.get("evidence_ids"), list) else []
        if not text or not ids:
            continue
        all_cited.update(ids)
        claims.append({
            "text": text,
            "evidence_ids": list(dict.fromkeys(ids)),
            "certainty": normalize_certainty(claim.get("certainty")),
        })
    conflicts = list(dict.fromkeys([
        evidence_id for evidence_id in result.get(
            "conflicting_evidence_ids", []
        ) if evidence_id in known
    ])) if isinstance(result.get("conflicting_evidence_ids"), list) else []
    conflict_set = set(conflicts)
    valid = bool(claims) and all_cited.union(conflict_set) == known
    if not valid:
        fallback["valid"] = False
        fallback["claims_from_failed_fusion"] = claims
        return fallback
    detail = " ".join(claim["text"] for claim in claims)
    return {
        "canonical_entity": cluster.canonical_entity,
        # Both renderings are assembled solely from cited claims. This avoids
        # asking the model to generate duplicate, potentially unsupported prose.
        "summary_short": detail[:500],
        "summary_detail": detail,
        "status": infer_event_status(cluster.evidence),
        "certainty": infer_certainty(cluster.evidence),
        "severity": _most_informative(
            item.severity for item in cluster.evidence
        ),
        "significance": infer_significance(cluster.evidence),
        "claims": claims,
        "conflicting_evidence_ids": conflicts,
        "valid": True,
    }


def _evidence_semantic_token(item: ClinicalEvidence) -> str:
    """Fingerprint only information that can change the fused clinical note."""
    payload = {
        "evidence_id": item.evidence_id,
        "category": item.category,
        "entity": canonicalize_entity(item.normalized_entity),
        "fusion_input": evidence_for_prompt(item),
    }
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _requires_llm_fusion(cluster: EvidenceCluster) -> bool:
    """Avoid an LLM call when sources repeat the exact same clinical claim."""
    if len(cluster.evidence) <= 1:
        return False
    if cluster.conflicts:
        return True
    signatures = set()
    for item in cluster.evidence:
        payload = evidence_for_prompt(item)
        # Repeated documentation dates are represented by EventUpdate; they
        # do not require regenerating identical prose.
        payload.pop("evidence_id", None)
        payload.pop("observed_date", None)
        payload.pop("documented_date", None)
        signatures.add(json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ))
    return len(signatures) > 1


def deterministic_fusion(cluster: EvidenceCluster) -> dict[str, Any]:
    claims = []
    by_text: dict[str, dict[str, Any]] = {}
    for item in cluster.evidence:
        text = evidence_claim(item)
        key = " ".join(text.casefold().split())
        if not text:
            continue
        existing = by_text.get(key)
        if existing is not None:
            if item.evidence_id not in existing["evidence_ids"]:
                existing["evidence_ids"].append(item.evidence_id)
            continue
        claim = {
            "text": text,
            "evidence_ids": [item.evidence_id],
            "certainty": normalize_certainty(item.certainty),
        }
        claims.append(claim)
        by_text[key] = claim
    summary = "; ".join(claim["text"] for claim in claims)
    return {
        "canonical_entity": cluster.canonical_entity,
        "summary_short": summary[:500] or display_entity(
            cluster.canonical_entity
        ),
        "summary_detail": summary,
        "status": infer_event_status(cluster.evidence),
        "certainty": infer_certainty(cluster.evidence),
        "severity": _most_informative(
            item.severity for item in cluster.evidence
        ),
        "significance": infer_significance(cluster.evidence),
        "claims": claims,
        "conflicting_evidence_ids": [],
        "valid": True,
    }


def build_updates(
    event_id: str,
    evidence: list[ClinicalEvidence],
    first_date: str | None,
) -> list[EventUpdate]:
    grouped: dict[str, list[ClinicalEvidence]] = {}
    for item in evidence:
        date = item.observed_date or item.document_date
        if not date or date == first_date:
            continue
        grouped.setdefault(date, []).append(item)
    updates = []
    for update_date, items in sorted(grouped.items(), key=lambda pair: date_sort_key(pair[0])):
        summary = "; ".join(
            dict.fromkeys(evidence_claim(item) for item in items)
        )
        updates.append(EventUpdate(
            update_id=stable_id("UPD", event_id, update_date),
            event_id=event_id,
            update_date=update_date,
            date_precision=best_date_precision(items, update_date),
            summary=summary,
            status_after=infer_event_status(items),
            evidence_ids=[item.evidence_id for item in items],
        ))
    return updates


def evidence_claim(item: ClinicalEvidence) -> str:
    quoted = " ".join(str(item.source_text or "").split()).strip(" .;:")
    parts = [quoted or display_entity(item.normalized_entity)]
    quoted_folded = quoted.casefold()
    if item.value_text:
        if str(item.value_text).casefold() not in quoted_folded:
            parts.append(item.value_text)
    elif item.numeric_value is not None:
        rendered = f"{item.numeric_value:g}{(' ' + item.unit) if item.unit else ''}"
        if re.sub(r"\s+", "", rendered.casefold()) not in re.sub(
            r"\s+", "", quoted_folded
        ):
            parts.append(rendered)
    if item.clinical_status and str(item.clinical_status).casefold() not in quoted_folded:
        parts.append(f"({item.clinical_status})")
    if item.severity and str(item.severity).casefold() not in quoted_folded:
        parts.append(f"gravità {item.severity}")
    if item.assertion == "absent" and not re.search(
        r"\b(?:no|non|assenza|nega|negativo)\b", quoted_folded
    ):
        parts.insert(0, "Assenza di")
    if item.certainty == "suspected" and not re.search(
        r"\b(?:sospett|possibil|probabil)\w*", quoted_folded
    ):
        parts.insert(0, "Sospetto di")
    elif item.certainty == "patient_reported" and not re.search(
        r"\b(?:rifer|lament|segnal)\w*", quoted_folded
    ):
        parts.insert(0, "Riferito:")
    return " ".join(part for part in parts if part).strip()


def evidence_for_prompt(item: ClinicalEvidence) -> dict[str, Any]:
    payload = {
        "evidence_id": item.evidence_id,
        "observed_date": item.observed_date,
        "assertion": item.assertion,
        "certainty": item.certainty,
        "status": item.clinical_status,
        "site": item.anatomical_site,
        "laterality": item.laterality,
        "severity": item.severity,
        "value_text": item.value_text,
        "numeric_value": item.numeric_value,
        "unit": item.unit,
        "source_text": item.source_text,
    }
    if not item.observed_date and item.document_date:
        payload["documented_date"] = item.document_date
    therapy = item.data.get("therapy")
    oncology = item.data.get("oncology")
    if isinstance(therapy, dict) and therapy:
        payload["therapy"] = therapy
    if isinstance(oncology, dict) and oncology:
        payload["oncology"] = oncology
    return {
        key: value for key, value in payload.items()
        if value not in (None, "", [], {})
    }


def stable_id(prefix: str, *parts: str) -> str:
    value = "\x1f".join(str(part or "") for part in parts)
    return f"{prefix}_{uuid.uuid5(uuid.NAMESPACE_URL, value).hex}"


def canonicalize_entity(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value or "").casefold())
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^a-z0-9%+]+", " ", text)
    stop = {"di", "del", "della", "con", "da", "in", "il", "la", "un", "una"}
    tokens = [token for token in text.split() if token not in stop]
    return "_".join(tokens[:20])


def display_entity(value: str) -> str:
    return " ".join(str(value or "evento").replace("_", " ").split()).strip().capitalize()


def infer_event_status(evidence: Iterable[ClinicalEvidence]) -> str:
    items = list(evidence)
    if not items:
        return "unknown"
    last = max(items, key=lambda item: (
        date_sort_key(item.observed_date or item.document_date),
        item.document_id, item.evidence_id,
    ))
    raw = str(last.clinical_status or "").casefold()
    if raw in {"planned", "proposed"}:
        return "planned"
    if raw == "completed":
        return "completed"
    if raw in {"suspended", "stopped", "interrupted"}:
        return "suspended"
    if raw == "cancelled":
        return "cancelled"
    if last.assertion == "absent":
        earlier_present = any(
            item is not last and item.assertion == "present" for item in items
        )
        return "resolved" if earlier_present else "unknown"
    if raw in _RESOLUTION_STATUSES:
        return "resolved"
    if raw in _ACTIVE_STATUSES:
        return "active"
    if any(not _is_resolution(item) for item in items):
        return "active"
    return "unknown"


def normalize_status(value: object) -> str:
    raw = str(value or "unknown").strip().casefold()
    aliases = {
        "present": "active", "in_corso": "ongoing", "stopped": "suspended",
        "interrupted": "suspended", "negative": "resolved",
    }
    raw = aliases.get(raw, raw)
    return raw if raw in {
        "proposed", "planned", "active", "ongoing", "completed",
        "resolved", "suspended", "cancelled", "unknown",
    } else "unknown"


def infer_certainty(evidence: Iterable[ClinicalEvidence]) -> str:
    levels = [normalize_certainty(item.certainty) for item in evidence]
    if "confirmed" in levels:
        return "confirmed"
    if "patient_reported" in levels:
        return "patient_reported"
    if "suspected" in levels:
        return "suspected"
    if "inferred" in levels:
        return "inferred"
    if "excluded" in levels:
        return "excluded"
    return "unknown"


def normalize_certainty(value: object) -> str:
    raw = str(value or "unknown").strip().casefold()
    aliases = {"reported": "patient_reported", "possible": "suspected"}
    raw = aliases.get(raw, raw)
    return raw if raw in {
        "confirmed", "suspected", "patient_reported", "inferred",
        "excluded", "unknown",
    } else "unknown"


def infer_assertion(evidence: Iterable[ClinicalEvidence]) -> str:
    assertions = {item.assertion for item in evidence}
    if assertions == {"absent"}:
        return "absent"
    if "present" in assertions:
        return "present"
    return next(iter(assertions), "unknown")


def infer_significance(evidence: Iterable[ClinicalEvidence]) -> str:
    priority = {
        "critical": 5, "high": 4, "clinically_relevant": 3,
        "potentially_relevant": 2, "uncertain": 1,
    }
    values = [normalize_significance(item.significance) for item in evidence]
    return max(values, key=lambda value: priority[value], default="uncertain")


def normalize_significance(value: object) -> str:
    raw = str(value or "uncertain").strip().casefold()
    return raw if raw in {
        "critical", "high", "clinically_relevant", "potentially_relevant",
        "uncertain",
    } else "uncertain"


def calibrated_proxy(
    evidence: Iterable[ClinicalEvidence], conflicts: Iterable[str]
) -> float:
    """Transparent heuristic, not an LLM self-confidence calibration."""
    items = list(evidence)
    if not items:
        return 0.0
    score = 0.55
    if all(item.data.get("quote_verified", True) for item in items):
        score += 0.15
    if len({item.document_id for item in items}) > 1:
        score += 0.1
    if any(item.extraction_method == "deterministic_lab" for item in items):
        score += 0.1
    if list(conflicts):
        score -= 0.25
    if any(item.certainty in {"inferred", "suspected"} for item in items):
        score -= 0.1
    return round(max(0.0, min(score, 0.95)), 3)


def best_date_precision(
    evidence: Iterable[ClinicalEvidence], date_value: str | None
) -> str:
    priority = {"day": 5, "month": 4, "year": 3, "interval": 2,
                "approximate": 1, "unknown": 0}
    candidates = [
        item.date_precision for item in evidence
        if item.observed_date == date_value
    ]
    return max(candidates, key=lambda value: priority.get(value, 0),
               default="unknown")


def _is_resolution(item: ClinicalEvidence) -> bool:
    status = str(item.clinical_status or "").casefold()
    return item.assertion == "absent" or status in _RESOLUTION_STATUSES


def _earliest_date_end(
    evidence: Iterable[ClinicalEvidence], first_date: str | None
) -> str | None:
    candidates = [
        item.observed_date_end for item in evidence
        if item.observed_date == first_date and item.observed_date_end
    ]
    return min(candidates, key=date_sort_key) if candidates else None


def _most_informative(values: Iterable[str | None]) -> str | None:
    candidates = [" ".join(str(value).split()) for value in values if value]
    return max(candidates, key=len) if candidates else None
