"""Two-scale candidate generation, relation adjudication and graph splitting."""

from __future__ import annotations

from collections import defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import hashlib
import inspect
import json
import uuid

from .temporal import date_bounds, temporal_distance_days
from .terminology import normalize_concept as _fold
from ..models.clinical_pipeline import (
    CLUSTER_EFFECTS,
    EVIDENCE_RELATION_TYPES,
    EvidenceRelation,
)
from ..settings import ClinicalPipelinePolicy
from ..prompt_catalog import load_prompt, prompts_digest


_RELATION_SYSTEM_PROMPT = load_prompt("evidence_relations_system")
_RELATION_TASK = load_prompt(
    "evidence_relations_task",
    required_markers=("must_link", "context_only", "cannot_link"),
)

RELATION_PROMPT_VERSION = "evidence-relations-v2"

_RELATION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "candidate_id": {"type": "string"},
                    "linked": {"type": "boolean"},
                    "relation_type": {
                        "type": "string", "enum": list(EVIDENCE_RELATION_TYPES),
                    },
                    "cluster_effect": {
                        "type": "string", "enum": list(CLUSTER_EFFECTS),
                    },
                    "weight": {"type": "number", "minimum": 0, "maximum": 1},
                    "rationale": {"type": "string"},
                },
                "required": [
                    "candidate_id", "linked", "relation_type",
                    "cluster_effect", "weight", "rationale",
                ],
            },
        },
    },
    "required": ["decisions"],
}
RELATION_PROMPT_DIGEST = prompts_digest(
    _RELATION_SYSTEM_PROMPT, _RELATION_TASK, schema=_RELATION_SCHEMA
)

_STOPWORDS = {
    "della", "delle", "dello", "degli", "alla", "alle", "con", "per",
    "che", "non", "sono", "come", "nel", "nella", "dei", "dal", "una",
    "the", "and", "from", "present", "presente", "paziente", "esame",
}

_SINGLE_EVENT_EXCEPTIONS = {
    "diagnosis", "procedure", "hospitalization", "surgery", "histopathology",
}

_OBJECTIVE = {
    "laboratory_finding", "imaging_finding", "instrumental_finding",
    "histopathology", "clinical_sign", "vital_sign", "biomarker",
}

_MANIFESTATION = {"symptom", "clinical_sign", "vital_sign", "adverse_event"}


@dataclass(slots=True)
class CandidatePair:
    candidate_id: str
    source_id: str
    target_id: str
    scales: set[str] = field(default_factory=set)
    shared_terms: set[str] = field(default_factory=set)
    temporal_distance: int | None = None


@dataclass(slots=True)
class EvidenceGraphCluster:
    evidence_ids: list[str]
    roles: dict[str, str]
    relation_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class EvidenceGraphResult:
    relations: list[EvidenceRelation]
    clusters: list[EvidenceGraphCluster]
    candidate_count: int
    llm_calls: int
    split_count: int
    cache_hits: int = 0
    auto_resolved_count: int = 0


class EvidenceGraphCancelled(RuntimeError):
    """Raised after a cooperative stop between relation LLM requests."""


class EvidenceGraphBuilder:
    """Avoid all-pairs comparison and form events only from accepted links."""

    def __init__(self, llm=None, *, policy: ClinicalPipelinePolicy | None = None):
        self.llm = llm
        self.policy = policy or ClinicalPipelinePolicy()
        self._voting_errors: list[str] = []

    def build(
        self,
        patient_id: str,
        evidence,
        *,
        reviewed_relations=(),
        anchor_evidence_ids: set[str] | None = None,
        num_workers: int = 1,
        cancel_check=None,
        progress_callback=None,
        cache_repository=None,
    ) -> EvidenceGraphResult:
        items = [item for item in evidence if item.patient_id == patient_id]
        by_id = {item.evidence_id: item for item in items}
        candidates = self._candidates(items)
        provisional = [
            self._rule_relation(patient_id, pair, by_id) for pair in candidates
        ]
        relations, llm_calls, cache_hits, auto_resolved = self._adjudicate(
            patient_id,
            candidates,
            provisional,
            by_id,
            num_workers=num_workers,
            cancel_check=cancel_check,
            progress_callback=progress_callback,
            cache_repository=cache_repository,
        )
        relations = _apply_reviewed_relations(
            patient_id, relations, reviewed_relations, set(by_id)
        )
        clusters, split_count = self._clusters(
            items,
            relations,
            anchor_evidence_ids=(
                set(by_id) if anchor_evidence_ids is None
                else set(anchor_evidence_ids) & set(by_id)
            ),
        )
        return EvidenceGraphResult(
            relations=relations,
            clusters=clusters,
            candidate_count=len(candidates),
            llm_calls=llm_calls,
            split_count=split_count,
            cache_hits=cache_hits,
            auto_resolved_count=auto_resolved,
        )

    # ------------------------------------------------------ candidate stage

    def _candidates(self, items) -> list[CandidatePair]:
        if len(items) < 2:
            return []
        top_k = max(1, self.policy.semantic_top_k)
        pairs: dict[tuple[str, str], CandidatePair] = {}
        ordered = sorted(items, key=_evidence_sort_key)

        # Local scale: only bounded chronological neighbours, never a dense
        # all-pairs window.
        for index, item in enumerate(ordered):
            for other in ordered[max(0, index - top_k):index]:
                distance = temporal_distance_days(
                    _clinical_date(item), _clinical_date(other)
                )
                if distance is not None and distance <= self.policy.local_window_days:
                    self._add_pair(pairs, item, other, "local", distance)

        # Longitudinal scale: bounded postings for exact concepts and salient
        # semantic terms. Each atom sees at most top_k predecessors per key.
        entity_postings: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=top_k)
        )
        term_postings: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=top_k)
        )
        for item in ordered:
            entity = _entity_key(item)
            if entity:
                for other in entity_postings[entity]:
                    distance = temporal_distance_days(
                        _clinical_date(item), _clinical_date(other)
                    )
                    if distance is None or distance <= self.policy.longitudinal_window_days:
                        self._add_pair(
                            pairs, item, other, "longitudinal", distance,
                            shared_terms={entity},
                        )
                entity_postings[entity].append(item)
            for term in sorted(_semantic_terms(item))[:12]:
                for other in term_postings[term]:
                    distance = temporal_distance_days(
                        _clinical_date(item), _clinical_date(other)
                    )
                    if distance is None or distance <= self.policy.longitudinal_window_days:
                        self._add_pair(
                            pairs, item, other, "longitudinal", distance,
                            shared_terms={term},
                        )
                term_postings[term].append(item)

        # Bound the final workload per node by a deterministic preliminary
        # relevance score, even when a very common term creates many pairs.
        ranked = sorted(
            pairs.values(),
            key=lambda pair: (
                -_candidate_priority(pair, by_id={
                    item.evidence_id: item for item in items
                }),
                pair.candidate_id,
            ),
        )
        counts: dict[str, int] = defaultdict(int)
        kept = []
        for pair in ranked:
            if counts[pair.source_id] >= top_k or counts[pair.target_id] >= top_k:
                continue
            kept.append(pair)
            counts[pair.source_id] += 1
            counts[pair.target_id] += 1
        return sorted(kept, key=lambda pair: pair.candidate_id)

    @staticmethod
    def _add_pair(
        pairs, left, right, scale: str, distance: int | None,
        shared_terms: set[str] | None = None,
    ) -> None:
        source_id, target_id = sorted((left.evidence_id, right.evidence_id))
        key = (source_id, target_id)
        pair = pairs.get(key)
        if pair is None:
            token = "\x1f".join(key)
            pair = CandidatePair(
                candidate_id="CAN_" + uuid.uuid5(
                    uuid.NAMESPACE_URL, token
                ).hex,
                source_id=source_id,
                target_id=target_id,
                temporal_distance=distance,
            )
            pairs[key] = pair
        pair.scales.add(scale)
        pair.shared_terms.update(shared_terms or set())
        if distance is not None and (
            pair.temporal_distance is None or distance < pair.temporal_distance
        ):
            pair.temporal_distance = distance

    # --------------------------------------------------------- edge stage

    def _rule_relation(self, patient_id, pair, by_id) -> EvidenceRelation:
        left, right = by_id[pair.source_id], by_id[pair.target_id]
        same_entity = _entity_key(left) == _entity_key(right)
        opposite = {left.assertion, right.assertion} >= {"present", "absent"}
        different_side = (
            left.laterality and right.laterality
            and _fold(left.laterality) != _fold(right.laterality)
        )
        relation_type = "temporally_associated_with"
        effect = "uncertain"
        weight = 0.5
        rationale = "Candidato generato a due scale"
        if same_entity and opposite and _is_later_resolution(left, right):
            relation_type, effect, weight = (
                "documents_resolution", "context_only", 0.94
            )
            rationale = (
                "Negazione o normalizzazione successiva compatibile con "
                "la risoluzione del concetto precedente"
            )
        elif same_entity and opposite:
            relation_type, effect, weight = "contradicts", "cannot_link", 0.98
            rationale = "Stesso concetto con polarità temporalmente incompatibile"
        elif same_entity and different_side:
            relation_type, effect, weight = "contradicts", "cannot_link", 0.92
            rationale = "Stesso concetto con lateralità incompatibile"
        elif same_entity:
            relation_type, effect = "same_process", "cohesive"
            weight = 0.93 if "local" in pair.scales else 0.86
            rationale = "Stesso concetto clinico in finestra compatibile"
        elif {left.category, right.category} & {"histopathology"} and (
            {left.category, right.category} & {"diagnosis", "procedure"}
        ):
            relation_type, effect, weight = "same_process", "cohesive", 0.84
            rationale = "Istologia e diagnosi/procedura semanticamente collegate"
        elif (
            left.category in _MANIFESTATION and right.category in _OBJECTIVE
        ) or (
            right.category in _MANIFESTATION and left.category in _OBJECTIVE
        ):
            relation_type, effect = "manifestation_of", "cohesive"
            weight = 0.79 if "local" in pair.scales else 0.68
            rationale = "Manifestazione e reperto obiettivo candidati"
        elif "medication" in {left.category, right.category}:
            relation_type = "treats"
            effect = "context_only"
            weight = 0.72 if "local" in pair.scales else 0.62
            rationale = (
                "Terapia autonoma potenzialmente collegata al contesto clinico"
            )
        elif "procedure" in {left.category, right.category}:
            relation_type, effect = "evaluates", "context_only"
            weight = 0.72 if "local" in pair.scales else 0.62
            rationale = (
                "Procedura autonoma potenzialmente collegata al contesto clinico"
            )
        elif pair.shared_terms and "local" in pair.scales:
            relation_type, effect, weight = (
                "temporally_associated_with", "uncertain", 0.64
            )
            rationale = "Termini clinici condivisi e prossimità temporale"
        relation_id = "ERL_" + uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"{patient_id}:{pair.source_id}:{pair.target_id}:{relation_type}",
        ).hex
        return EvidenceRelation(
            relation_id=relation_id,
            patient_id=patient_id,
            source_evidence_id=pair.source_id,
            target_evidence_id=pair.target_id,
            relation_type=relation_type,
            direction="undirected",
            weight=weight,
            cluster_effect=effect,
            rationale=rationale,
            rule_features={
                "scales": sorted(pair.scales),
                "shared_terms": sorted(pair.shared_terms),
                "temporal_distance_days": pair.temporal_distance,
                "same_entity": same_entity,
            },
            generation_method="candidate_rules",
            review_status="pending",
        )

    def _adjudicate(
        self,
        patient_id,
        candidates,
        provisional,
        by_id,
        *,
        num_workers=1,
        cancel_check=None,
        progress_callback=None,
        cache_repository=None,
    ):
        if not provisional:
            return [], 0, 0, 0
        available = bool(self.llm and getattr(self.llm, "is_available", True))
        if not available:
            for relation in provisional:
                relation.generation_method = "rule_fallback"
            return provisional, 0, 0, 0

        profile = self.policy.consensus_profile
        vote_rounds = {"fast": 1, "selective": 1, "robust": 2, "research": 3}[
            profile
        ]
        candidate_by_id = {pair.candidate_id: pair for pair in candidates}
        relation_by_pair = {
            (item.source_evidence_id, item.target_evidence_id): item
            for item in provisional
        }
        votes: dict[str, list[dict]] = defaultdict(list)
        calls = 0
        cache_hits = 0

        # These constraints are authoritative later in this very method, so
        # asking the model to adjudicate them cannot change the result.  Skip
        # those calls without changing clinical behaviour.
        hard_candidate_ids = {
            pair.candidate_id
            for pair in candidates
            if relation_by_pair[(pair.source_id, pair.target_id)].cluster_effect
            == "cannot_link"
        }
        adjudicated_candidates = [
            pair for pair in candidates
            if pair.candidate_id not in hard_candidate_ids
        ]
        auto_resolved = len(hard_candidate_ids)
        total_work = len(candidates) * vote_rounds
        completed_work = auto_resolved * vote_rounds

        namespace = _relation_cache_namespace(self.llm)
        cached = {}
        if cache_repository is not None:
            try:
                cached = cache_repository.list_relation_adjudication_cache(
                    patient_id, namespace
                )
            except Exception:
                # Cache failure must never change clinical output.
                cached = {}

        def notify_progress() -> None:
            if progress_callback is not None:
                progress_callback(
                    completed_work, total_work, cache_hits, auto_resolved
                )

        def check_cancelled() -> None:
            if cancel_check is not None and cancel_check():
                raise EvidenceGraphCancelled(
                    "Elaborazione delle relazioni interrotta su richiesta "
                    "dell'utente"
                )

        notify_progress()
        workers = max(1, min(int(num_workers or 1), 32))
        for round_index in range(vote_rounds):
            check_cancelled()
            pending = []
            cache_keys = {}
            for pair in adjudicated_candidates:
                row = _candidate_prompt_row(pair, by_id)
                cache_key = _relation_cache_key(
                    namespace, round_index, row
                )
                cache_keys[pair.candidate_id] = cache_key
                stored = cached.get(cache_key)
                cached_vote = _validate_cached_vote(stored, pair.candidate_id)
                if cached_vote is not None:
                    votes[pair.candidate_id].append(cached_vote)
                    cache_hits += 1
                    completed_work += 1
                else:
                    pending.append(pair)
            notify_progress()

            batch_size = _vote_batch_size(
                int(getattr(self.llm, "context_length", 32768) or 32768)
            )
            batches = [
                pending[index:index + batch_size]
                for index in range(0, len(pending), batch_size)
            ]
            if not batches:
                continue

            if workers == 1:
                for batch in batches:
                    check_cancelled()
                    payload = self._classify_batch(batch, by_id)
                    calls += 1
                    self._record_batch_votes(
                        patient_id, namespace, round_index, batch, payload,
                        votes, candidate_by_id, cache_keys,
                        cache_repository, cached,
                    )
                    completed_work += len(batch)
                    notify_progress()
                continue

            executor = ThreadPoolExecutor(
                max_workers=min(workers, len(batches)),
                thread_name_prefix="evidence-relations",
            )
            futures: dict[Future, list[CandidatePair]] = {}
            try:
                for batch in batches:
                    check_cancelled()
                    futures[executor.submit(
                        self._classify_batch, batch, by_id
                    )] = batch
                    calls += 1
                for future in as_completed(futures):
                    batch = futures[future]
                    payload = future.result()
                    self._record_batch_votes(
                        patient_id, namespace, round_index, batch, payload,
                        votes, candidate_by_id, cache_keys,
                        cache_repository, cached,
                    )
                    completed_work += len(batch)
                    notify_progress()
                    check_cancelled()
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
            finally:
                executor.shutdown(wait=True, cancel_futures=True)

        if calls and not any(
            votes.get(pair.candidate_id) for pair in candidates
        ):
            detail = (
                "; ".join(dict.fromkeys(self._voting_errors[:3]))
                if self._voting_errors else "nessun voto valido ricevuto"
            )
            raise RuntimeError(
                "Votazione relazioni fallita: il modello locale non ha "
                f"prodotto alcun voto valido. ({detail})"
            )

        result = []
        for pair in candidates:
            base = relation_by_pair[(pair.source_id, pair.target_id)]
            pair_votes = votes.get(pair.candidate_id, [])
            if base.cluster_effect == "cannot_link":
                base.model_votes = pair_votes
                base.generation_method = "hybrid_hard_constraint"
                base.review_status = "auto"
                result.append(base)
                continue
            linked = [vote for vote in pair_votes if vote["linked"]]
            required = len(pair_votes) // 2 + 1
            if len(linked) < required:
                base.cluster_effect = "uncertain"
                base.weight = min(base.weight, 0.49)
                base.model_votes = pair_votes
                base.generation_method = "hybrid_adjudicated"
                base.rationale = "Il modello non conferma il collegamento candidato"
                result.append(base)
                continue
            chosen = max(
                linked,
                key=lambda vote: (
                    sum(v["relation_type"] == vote["relation_type"] for v in linked),
                    vote["weight"], vote["candidate_id"],
                ),
            )
            base.relation_type = chosen["relation_type"]
            base.cluster_effect = chosen["cluster_effect"]
            base.weight = sum(v["weight"] for v in linked) / len(linked)
            base.rationale = chosen["rationale"]
            base.model_votes = pair_votes
            base.generation_method = "hybrid_adjudicated"
            base.review_status = (
                "auto" if len(linked) == len(pair_votes) else "pending"
            )
            result.append(base)
        return result, calls, cache_hits, auto_resolved

    @staticmethod
    def _record_batch_votes(
        patient_id,
        namespace,
        round_index,
        batch,
        payload,
        votes,
        candidate_by_id,
        cache_keys,
        cache_repository,
        cached,
    ) -> None:
        valid = []
        batch_ids = {pair.candidate_id for pair in batch}
        pair_by_id = {pair.candidate_id: pair for pair in batch}
        for vote in payload:
            candidate_id = vote["candidate_id"]
            if candidate_id not in candidate_by_id or candidate_id not in batch_ids:
                continue
            votes[candidate_id].append(vote)
            pair = pair_by_id[candidate_id]
            record = {
                "cache_key": cache_keys[candidate_id],
                "candidate_id": candidate_id,
                "source_evidence_id": pair.source_id,
                "target_evidence_id": pair.target_id,
                "decision": vote,
            }
            valid.append(record)
            cached[record["cache_key"]] = vote
        if cache_repository is not None and valid:
            try:
                cache_repository.save_relation_adjudication_batch(
                    patient_id, namespace, round_index, valid
                )
            except Exception:
                # A cache/checkpoint is an acceleration layer.  The votes
                # already in memory remain authoritative for this run.
                pass

    def _classify_batch(self, batch, by_id) -> list[dict]:
        rows = [_candidate_prompt_row(pair, by_id) for pair in batch]
        prompt = (
            _RELATION_TASK + "\n\nCANDIDATI:\n"
            + json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
        )
        try:
            generator = self.llm.generate_structured
            parameters = inspect.signature(generator).parameters
            if "max_tokens" in parameters or any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            ):
                data = generator(
                    prompt, _RELATION_SYSTEM_PROMPT, _RELATION_SCHEMA,
                    max_tokens=min(
                        int(getattr(self.llm, "max_output_tokens", 4096) or 4096),
                        max(1024, 160 * len(batch)),
                    ),
                )
            else:
                data = generator(prompt, _RELATION_SYSTEM_PROMPT, _RELATION_SCHEMA)
        except Exception as exc:
            self._voting_errors.append(
                f"{type(exc).__name__}: {str(exc)[:200]}"
            )
            return []
        return _validate_votes(data, {pair.candidate_id for pair in batch})

    # ------------------------------------------------------- cluster stage

    def _clusters(self, items, relations, *, anchor_evidence_ids: set[str]):
        by_id = {item.evidence_id: item for item in items}
        cannot = {
            frozenset((rel.source_evidence_id, rel.target_evidence_id))
            for rel in relations if rel.cluster_effect == "cannot_link"
        }
        positive = [
            rel for rel in relations
            if rel.cluster_effect in {"must_link", "cohesive"}
            and rel.weight >= self.policy.cohesive_threshold
        ]
        adjacency: dict[str, list[EvidenceRelation]] = defaultdict(list)
        for rel in positive:
            adjacency[rel.source_evidence_id].append(rel)
            adjacency[rel.target_evidence_id].append(rel)

        components = _connected_components(adjacency)
        split_count = 0
        split_components = []
        for component in components:
            separated = _split_component(
                component, adjacency, cannot,
                bridge_threshold=self.policy.bridge_split_threshold,
            )
            split_count += max(0, len(separated) - 1)
            split_components.extend(separated)

        clusters: list[EvidenceGraphCluster] = []
        assigned_core: set[str] = set()
        for component in split_components:
            if len(component) < 2:
                continue
            assigned_core.update(component)
            roles = _roles_for_component(component, by_id)
            relation_ids = [
                rel.relation_id for rel in positive
                if rel.source_evidence_id in component
                and rel.target_evidence_id in component
            ]
            clusters.append(EvidenceGraphCluster(
                evidence_ids=sorted(component), roles=roles,
                relation_ids=sorted(relation_ids),
            ))

        # Every clinically eligible primary atom must remain visible even when
        # it has no accepted graph edge.  Contextual normal/negative atoms are
        # deliberately omitted here and may only attach to an existing event.
        for item in items:
            if item.evidence_id in assigned_core:
                continue
            if (
                item.evidence_id in anchor_evidence_ids
                or item.category in _SINGLE_EVENT_EXCEPTIONS
            ):
                clusters.append(EvidenceGraphCluster(
                    evidence_ids=[item.evidence_id],
                    roles={item.evidence_id: "core"},
                ))

        # context_only links can attach one atom to multiple already formed
        # events. They never create an event on their own.
        for relation in relations:
            if relation.cluster_effect != "context_only" or relation.weight < 0.55:
                continue
            endpoints = (
                relation.source_evidence_id, relation.target_evidence_id
            )
            for cluster in clusters:
                present = [endpoint in cluster.evidence_ids for endpoint in endpoints]
                if present.count(True) != 1:
                    continue
                missing = endpoints[0] if not present[0] else endpoints[1]
                # A primary medication/procedure/clinical fact retains its own
                # autonomous event.  context_only is allowed to absorb only a
                # non-anchoring observation such as a later negative finding.
                if missing in anchor_evidence_ids:
                    continue
                cluster.evidence_ids.append(missing)
                cluster.evidence_ids = sorted(set(cluster.evidence_ids))
                cluster.roles[missing] = _role_for(by_id[missing], core=False)
                cluster.relation_ids.append(relation.relation_id)
        return clusters, split_count


def _vote_batch_size(context_length: int) -> int:
    """Pairs per voting call, sized so prompt + output fit the context.

    Measured on the real corpus: ~360 prompt tokens per pair plus ~160
    output tokens per pair.  A 15% margin keeps the request safely inside
    smaller contexts (the 4B voting model runs at 8k).
    """
    if context_length < 4096:
        return 4
    return max(4, min(32, int(context_length * 0.85 / 520)))


def _is_later_resolution(left, right) -> bool:
    """True only for an unambiguously later absent/resolved observation."""
    absent = left if left.assertion == "absent" else right
    present = right if absent is left else left
    if absent.assertion != "absent" or present.assertion != "present":
        return False
    absent_bounds = date_bounds(_clinical_date(absent))
    present_bounds = date_bounds(_clinical_date(present))
    if not absent_bounds or not present_bounds:
        return False
    if absent_bounds[0] <= present_bounds[1]:
        return False
    return True


def _apply_reviewed_relations(
    patient_id: str, generated, reviewed_rows, evidence_ids: set[str]
):
    """Make human pair decisions authoritative across incremental rebuilds."""
    reviewed_by_pair = {}
    for row in reviewed_rows or ():
        status = str(row.get("review_status") or "")
        source_id = str(row.get("source_evidence_id") or "")
        target_id = str(row.get("target_evidence_id") or "")
        if (
            status not in {"accepted", "rejected"}
            or source_id not in evidence_ids or target_id not in evidence_ids
        ):
            continue
        pair = frozenset((source_id, target_id))
        previous = reviewed_by_pair.get(pair)
        if previous is None or status == "accepted":
            reviewed_by_pair[pair] = row

    result = []
    used_pairs = set()
    for relation in generated:
        pair = frozenset((
            relation.source_evidence_id, relation.target_evidence_id,
        ))
        reviewed = reviewed_by_pair.get(pair)
        if reviewed is None:
            result.append(relation)
            continue
        used_pairs.add(pair)
        override = _reviewed_relation(patient_id, reviewed)
        if override is None:
            result.append(relation)
        elif override.review_status == "rejected":
            override.cluster_effect = "uncertain"
            override.weight = 0.0
            override.rationale = "Collegamento rifiutato in revisione umana"
            override.generation_method = "human_rejected_override"
            result.append(override)
        else:
            override.generation_method = "human_accepted_override"
            result.append(override)

    # An accepted edge remains authoritative even if a changed blocker no
    # longer generates that pair automatically.
    for pair, row in reviewed_by_pair.items():
        if pair in used_pairs or row.get("review_status") != "accepted":
            continue
        override = _reviewed_relation(patient_id, row)
        if override is not None:
            override.generation_method = "human_accepted_override"
            result.append(override)
    return sorted(result, key=lambda item: item.relation_id)


def _reviewed_relation(patient_id: str, row: dict) -> EvidenceRelation | None:
    try:
        return EvidenceRelation(
            relation_id=str(row["relation_id"]),
            patient_id=patient_id,
            source_evidence_id=str(row["source_evidence_id"]),
            target_evidence_id=str(row["target_evidence_id"]),
            relation_type=str(row["relation_type"]),
            direction=str(row.get("direction") or "directed"),
            weight=float(row["weight"]),
            cluster_effect=str(row["cluster_effect"]),
            rationale=str(row.get("rationale") or ""),
            rule_features=dict(row.get("rule_features") or {}),
            model_votes=list(row.get("model_votes") or []),
            generation_method=str(row.get("generation_method") or "human"),
            review_status=str(row["review_status"]),
            created_at=str(row.get("created_at") or ""),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _validate_votes(data, known_ids: set[str]) -> list[dict]:
    if not isinstance(data, dict) or not isinstance(data.get("decisions"), list):
        return []
    result = []
    for raw in data["decisions"]:
        if not isinstance(raw, dict) or raw.get("candidate_id") not in known_ids:
            continue
        if raw.get("relation_type") not in EVIDENCE_RELATION_TYPES:
            continue
        if raw.get("cluster_effect") not in CLUSTER_EFFECTS:
            continue
        try:
            weight = min(1.0, max(0.0, float(raw.get("weight"))))
        except (TypeError, ValueError):
            continue
        result.append({
            "candidate_id": raw["candidate_id"],
            "linked": bool(raw.get("linked")),
            "relation_type": raw["relation_type"],
            "cluster_effect": raw["cluster_effect"],
            "weight": weight,
            "rationale": " ".join(str(raw.get("rationale") or "").split())[:500],
        })
    return result


def _connected_components(adjacency) -> list[set[str]]:
    unseen = set(adjacency)
    result = []
    while unseen:
        start = min(unseen)
        component = set()
        stack = [start]
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            unseen.discard(node)
            for edge in adjacency.get(node, []):
                other = (
                    edge.target_evidence_id
                    if edge.source_evidence_id == node
                    else edge.source_evidence_id
                )
                if other not in component:
                    stack.append(other)
        result.append(component)
    return result


def _split_component(component, adjacency, cannot, *, bridge_threshold: float):
    blocked_edges = set()
    for edge in {
        relation.relation_id: relation
        for node in component for relation in adjacency.get(node, [])
    }.values():
        normalized_strength = max(0.0, min(1.0, (edge.weight - 0.5) * 2))
        if normalized_strength < bridge_threshold and _is_bridge(
            component, adjacency, edge
        ):
            blocked_edges.add(edge.relation_id)
    # If a prohibited pair survived through different strong paths, cut the
    # weakest path edge deterministically and recompute components.
    for pair in cannot:
        if pair <= component:
            path = _path_between(component, adjacency, *sorted(pair), blocked_edges)
            if path:
                weakest = min(path, key=lambda edge: (edge.weight, edge.relation_id))
                blocked_edges.add(weakest.relation_id)
    reduced: dict[str, list] = defaultdict(list)
    for node in component:
        reduced[node]
        for edge in adjacency.get(node, []):
            if edge.relation_id not in blocked_edges:
                reduced[node].append(edge)
    return _connected_components(reduced) if reduced else [{node} for node in component]


def _is_bridge(component, adjacency, candidate) -> bool:
    return not _path_between(
        component, adjacency, candidate.source_evidence_id,
        candidate.target_evidence_id, {candidate.relation_id},
    )


def _path_between(component, adjacency, start, target, blocked):
    queue = deque([(start, [])])
    seen = {start}
    while queue:
        node, path = queue.popleft()
        if node == target:
            return path
        for edge in adjacency.get(node, []):
            if edge.relation_id in blocked:
                continue
            other = edge.target_evidence_id if edge.source_evidence_id == node else edge.source_evidence_id
            if other in component and other not in seen:
                seen.add(other)
                queue.append((other, path + [edge]))
    return []


def _roles_for_component(component, by_id):
    anchor = min(
        (by_id[evidence_id] for evidence_id in component),
        key=lambda item: (_anchor_rank(item), _evidence_sort_key(item)),
    )
    return {
        evidence_id: (
            "core" if evidence_id == anchor.evidence_id
            else _role_for(by_id[evidence_id], core=False)
        )
        for evidence_id in component
    }


def _role_for(item, *, core: bool) -> str:
    if core:
        return "core"
    if item.assertion == "absent" or item.certainty == "excluded":
        return "negative_context"
    if item.category in _MANIFESTATION:
        return "manifestation"
    if item.category in {"histopathology", "imaging_finding", "instrumental_finding"}:
        return "diagnostic_support"
    if item.category == "medication" or item.category in {"procedure", "surgery"}:
        return "treatment"
    if item.category in {"laboratory_finding", "vital_sign"}:
        return "monitoring"
    if item.category in {"response", "progression", "discharge"}:
        return "outcome"
    return "core"


def _anchor_rank(item) -> int:
    return {
        "diagnosis": 0, "histopathology": 1, "procedure": 2,
        "hospitalization": 3, "symptom": 4, "clinical_sign": 5,
        "imaging_finding": 6, "instrumental_finding": 7,
        "medication": 8, "laboratory_finding": 9,
    }.get(item.category, 10)


def _candidate_priority(pair, by_id) -> float:
    left, right = by_id[pair.source_id], by_id[pair.target_id]
    score = 2.0 if _entity_key(left) == _entity_key(right) else 0.0
    score += min(2.0, len(pair.shared_terms) * 0.35)
    score += 0.7 if "local" in pair.scales else 0.0
    if pair.temporal_distance is not None:
        score += 1.0 / (1.0 + pair.temporal_distance)
    return score


def _clinical_date(item):
    return item.observed_date or item.document_date


def _evidence_sort_key(item):
    value = str(_clinical_date(item) or "9999-99-99")
    return value, item.document_id, item.evidence_id


def _entity_key(item):
    return _fold(item.canonical_label or item.normalized_entity)


def _semantic_terms(item) -> set[str]:
    value = " ".join((
        item.canonical_label or "", item.normalized_entity or "",
        item.anatomical_site or "",
    ))
    return {
        token for token in _fold(value).split("_")
        if len(token) >= 4 and token not in _STOPWORDS
    }


def _prompt_evidence(item) -> dict:
    return {
        "evidence_id": item.evidence_id,
        "fact_type": item.fact_type,
        "category": item.category,
        "concept": item.canonical_label or item.normalized_entity,
        "date": _clinical_date(item),
        "polarity": (item.data or {}).get("polarity"),
        "status": item.clinical_status,
        "site": item.anatomical_site,
        "laterality": item.laterality,
        "value": item.value_text if item.value_text is not None else item.numeric_value,
        "unit": item.unit,
        "source": " ".join(str(item.source_text or "").split())[:600],
    }


def _candidate_prompt_row(pair: CandidatePair, by_id) -> dict:
    return {
        "candidate_id": pair.candidate_id,
        "scales": sorted(pair.scales),
        "temporal_distance_days": pair.temporal_distance,
        "shared_terms": sorted(pair.shared_terms),
        "a": _prompt_evidence(by_id[pair.source_id]),
        "b": _prompt_evidence(by_id[pair.target_id]),
    }


def _relation_cache_namespace(llm) -> str:
    """Fingerprint everything that can alter one relation-model decision."""
    model_info = None
    try:
        backend = getattr(llm, "backend", None)
        resolver = getattr(backend, "model_info", None)
        if callable(resolver):
            model_info = resolver(str(getattr(llm, "model", "")))
    except Exception:
        model_info = None
    payload = {
        "prompt_version": RELATION_PROMPT_VERSION,
        "prompt_digest": RELATION_PROMPT_DIGEST,
        "schema": _RELATION_SCHEMA,
        "llm_class": (
            f"{llm.__class__.__module__}.{llm.__class__.__qualname__}"
            if llm is not None else ""
        ),
        "model": getattr(llm, "model", None),
        "backend": getattr(llm, "backend_type", None),
        "temperature": getattr(llm, "temperature", None),
        "top_p": getattr(llm, "top_p", None),
        "top_k": getattr(llm, "top_k", None),
        "seed": getattr(llm, "seed", None),
        "model_info": model_info,
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _relation_cache_key(namespace: str, round_index: int, row: dict) -> str:
    encoded = json.dumps(
        {
            "namespace": namespace,
            "round_index": int(round_index),
            "candidate": row,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_cached_vote(value, candidate_id: str) -> dict | None:
    if isinstance(value, dict) and "decision" in value:
        value = value.get("decision")
    validated = _validate_votes(
        {"decisions": [value]}, {candidate_id}
    ) if isinstance(value, dict) else []
    return validated[0] if validated else None
