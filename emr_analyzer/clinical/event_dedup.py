"""Deterministic content cleanup and cross-event deduplication.

Complementary to :class:`ClinicalConsolidator`, which only clusters evidence
by exact (category, canonical entity).  This module runs inside
``ClinicalRegistryBuilder.build`` *after* clustering and fusion and addresses
two registry-quality problems:

1. Non-clinical content leaks into summaries: report headers, exam-execution
   metadata, tracer administration notes and full instrument measurement
   tables.  ``strip_header_boilerplate`` and ``compress_measurement_table``
   clean every bundle summary deterministically (no LLM call).
2. The same clinical fact restated across documents/dates becomes several
   events (historical reprints, cross-category doubles).  ``deduplicate_bundles``
   merges bundles whose *cleaned* summaries describe the same event, while
   keeping reviewed events untouched and distinct measurement parameters apart.

Order matters: summaries are cleaned *before* similarity is computed, so two
echo parameters sharing one identical raw measurement table no longer look
like duplicates.
"""

from __future__ import annotations

from difflib import SequenceMatcher
import re
import unicodedata
from typing import Iterable

from .consolidation import (
    build_updates,
    canonicalize_entity,
    display_entity,
    stable_id,
)
from .temporal import date_sort_key, temporal_distance_days
from ..models.clinical_evidence import ClinicalEvidence
from .evidence_relevance import is_administrative_evidence
from ..models.clinical_registry import (
    ClinicalEpisode,
    ClinicalEvent,
    EventEvidenceLink,
)

# -- Header / boilerplate stripping ------------------------------------------

# Header labels are removed independently from their values.  A broad
# ``[^.;]{0,100}`` suffix used here previously swallowed the first real
# measurement whenever a PDF placed it on the same line as ``Paziente:``.
_HEADER_FIELD = re.compile(
    r"(?i)\b(?:paziente|quesito\s+clinico|acc\s+number|n[°º]\s*esame|"
    r"richiesto\s+da|id\s+paziente|provenienza|classe\s+dose|"
    r"data\s+esame|ora\s+esame|data\s+di\s+nascita|"
    r"n[°º]\s*referto)\s*[:：]"
)

_HEADER_SCALAR_VALUE = re.compile(
    r"(?i)\b(?:data\s+esame|ora\s+esame|data\s+di\s+nascita|"
    r"acc\s+number|id\s+paziente|n[°º]\s*esame|n[°º]\s*referto)"
    r"\s*[:：]\s*(?:\[[^\]]+\]|[A-Z0-9/*:.\-]+)"
)

_QUESTION_VALUE = re.compile(
    r"(?i)\bquesito\s+clinico\s*[:：]\s*.*?"
    r"(?=(?:prestazioni\s+eseguite|classe\s+dose|data\s+esame)\b|[.;\n]|$)"
)

_HEADER_PHRASES = (
    r"(?i)\bprestazioni\s+eseguite(?:\s+e\s+indicazione\s+di\s+dose)?"
    r"(?:\s+secondo\s+l['’]?\s*art\.?\s*\d+)?[^;\n]*",
    r"(?i)\besame\s+confrontato\s+con\s+precedent\w*[^.;]*",
    r"(?i)(?:\bdi\s+18f[- ]?fdg\s*\(?\s*)?\bsomministrazione\s+"
    r"(?:e\.v\.\s+)?di\s+18f[- ]?fdg[^;\n]*",
    r"(?i)\bsomministrazione\s+del\s+tracciante[^.;]*",
    r"(?i)\b(?:prossim[oaie]?\s+)?appuntament\w*[^.;\n]*",
    r"(?i)\bprenotazion\w*[^.;\n]*",
)

_PLACEHOLDER = re.compile(r"\[[^\]]*(?:PAZIENTE|RIMOSS[OA]|ANAGRAFICA|TELEFONO|NASCITA)[^\]]*\]")

# Headers are replaced by a single control character so neighbouring claim
# separators ("; ", ". ") can be re-joined without ever touching the periods
# and decimals of the surviving clinical text.
_GAP = ""
_GAP_JOIN = re.compile(r"\s*[.;,]*\s*\s*[.;,]*\s*")


def _join_gap(match: re.Match) -> str:
    return "; " if any(char in match.group(0) for char in ";,.") else " "


def strip_header_boilerplate(text: str) -> str:
    """Remove report headers, exam-execution metadata and anonymization
    placeholders from a summary, keeping the clinical claims intact."""
    value = str(text or "")
    for pattern in _HEADER_PHRASES:
        value = re.sub(pattern, _GAP, value)
    value = _QUESTION_VALUE.sub(_GAP, value)
    value = _HEADER_SCALAR_VALUE.sub(_GAP, value)
    value = _HEADER_FIELD.sub(_GAP, value)
    value = _PLACEHOLDER.sub(_GAP, value)
    value = _GAP_JOIN.sub(_join_gap, value)
    value = " ".join(value.split()).strip(" ;,:-")
    return " ".join(value.split())


# -- Measurement-table compression -------------------------------------------

_MEASUREMENT_CATEGORIES = {"imaging_finding", "vital_sign", "clinical_sign"}

_TABLE_MARKERS = (
    re.compile(r"(?i)\bb[- ]?mode\b"),
    re.compile(r"(?i)\bsimpson\s+monoplan\w*\b"),
    re.compile(r"(?i)\btdi\b"),
    re.compile(r"(?i)\bdoppler\s+velocit\w*\b"),
    re.compile(r"(?i)\bgr[aà]bas\b"),
    re.compile(r"(?i)\b(?:vod|vos|boo|too)\s*:"),
    re.compile(r"(?i)\b4ch\b"),
    re.compile(r"\[\s*\d+(?:[.,]\d+)?\s*[-–—]\s*\d+(?:[.,]\d+)?\s*\]"),
)


def _table_marker_count(text: str) -> int:
    return sum(1 for pattern in _TABLE_MARKERS if pattern.search(text))


def _entity_label(event: ClinicalEvent, item: ClinicalEvidence | None) -> str:
    """Prefer the original casing of the measured parameter; fall back to the
    canonical form rendered for display."""
    raw = item.normalized_entity if item and item.normalized_entity else None
    if not raw:
        raw = str(event.canonical_entity or "")
    label = " ".join(str(raw).replace("_", " ").split()).strip()
    return label or display_entity(event.canonical_entity)


def _measurement_value(item: ClinicalEvidence | None) -> str:
    if item is None:
        return ""
    if item.value_text:
        return str(item.value_text)
    if item.numeric_value is not None:
        rendered = f"{item.numeric_value:g}"
        if item.unit:
            rendered += f" {item.unit}"
        return rendered
    return ""


def compress_measurement_table(event: ClinicalEvent, evidence) -> str:
    """Replace a full instrument measurement dump with a single
    ``parameter + measured value`` summary.

    The ten distinct echo parameters share one ~300-character table; keeping
    the table verbatim both bloats the registry and makes every parameter look
    identical.  Using the evidence's own value keeps each parameter
    distinguishable (e.g. "Frazione Eiezione 4 Ch Simpson MonoPlano 68 %").
    """
    value = _measurement_value(evidence)
    label = _entity_label(event, evidence)
    if value:
        return f"{label}: {value}"
    return label


def atomic_event_summary(
    event: ClinicalEvent,
    evidence: ClinicalEvidence | None,
) -> str:
    """Render the clinical atom, never its entire report-sized citation."""
    label = _entity_label(event, evidence)
    parts = []
    assertion = str(event.assertion or "present").casefold()
    if assertion == "absent":
        parts.append(f"Assenza di {label}")
    elif assertion in {"conditional", "hypothetical"}:
        parts.append(f"Ipotesi: {label}")
    else:
        parts.append(label)
    value = _measurement_value(evidence)
    if value and value.casefold() not in parts[0].casefold():
        parts.append(value)
    site = (evidence.anatomical_site if evidence else None) or event.anatomical_site
    laterality = (evidence.laterality if evidence else None) or event.laterality
    severity = (evidence.severity if evidence else None) or event.severity
    if site and str(site).casefold() not in " ".join(parts).casefold():
        parts.append(str(site))
    if laterality and str(laterality).casefold() not in " ".join(parts).casefold():
        parts.append(str(laterality))
    if severity and str(severity).casefold() not in " ".join(parts).casefold():
        parts.append(str(severity))
    return "; ".join(parts)


def clean_event_summary(event: ClinicalEvent, evidence) -> str:
    """Header-free, table-compressed summary used for both display and
    deduplication similarity."""
    text = strip_header_boilerplate(event.summary_short or "")
    marker_count = _table_marker_count(text)
    if event.category in _MEASUREMENT_CATEGORIES and (
        marker_count >= 3 or (len(text) > 150 and marker_count >= 2)
    ):
        compressed = compress_measurement_table(event, evidence)
        if compressed and len(compressed) < len(text):
            text = compressed
    claims = event.structured_data.get("claims", []) if (
        isinstance(event.structured_data, dict)
    ) else []
    source = " ".join(str(evidence.source_text or "").split()) if evidence else ""
    rendered = " ".join(str(event.summary_short or "").split())
    source_dominated = bool(
        evidence
        and len(text) > 280
        and (
            source.casefold().startswith(rendered.casefold()[:160])
            or rendered.casefold().startswith(source.casefold()[:160])
            or marker_count >= 2
        )
    )
    if source_dominated and len(claims) <= 1:
        focused = atomic_event_summary(event, evidence)
        if focused:
            text = focused
    return text.strip(" ;,:-")


def _best_summary_evidence(
    bundle, evidence_by_id: dict[str, ClinicalEvidence]
) -> ClinicalEvidence | None:
    """Pick the atom closest to the event identity, preferring a value."""
    candidates = []
    for link in bundle.links:
        item = evidence_by_id.get(link.evidence_id)
        if item is None or not link.included_in_summary:
            continue
        exact_entity = int(
            canonicalize_entity(item.normalized_entity)
            == canonicalize_entity(bundle.event.canonical_entity)
        )
        has_value = int(
            bool(item.value_text) or item.numeric_value is not None
        )
        candidates.append((
            exact_entity, has_value, item.confidence,
            -len(item.source_text or ""), item.evidence_id, item,
        ))
    return max(candidates, default=(0, 0, 0, 0, "", None))[-1]


def clean_bundle_summaries(
    bundles: Iterable, evidence_by_id: dict[str, ClinicalEvidence]
) -> dict[str, str]:
    """Return one concise, distinguishable display summary per bundle.

    Different atoms can legitimately cite the same long source sentence.  If
    their cleaned display text is still identical, render their individual
    entities instead of showing the same sentence four or twenty times.
    """
    bundles = list(bundles)
    result: dict[str, str] = {}
    selected: dict[str, ClinicalEvidence | None] = {}
    for bundle in bundles:
        item = _best_summary_evidence(bundle, evidence_by_id)
        selected[bundle.event.event_id] = item
        cleaned = clean_event_summary(bundle.event, item)
        result[bundle.event.event_id] = (
            cleaned or atomic_event_summary(bundle.event, item)
        )

    visual_groups: dict[tuple[str, str, str], list] = {}
    for bundle in bundles:
        event_id = bundle.event.event_id
        visual_groups.setdefault((
            bundle.event.category,
            bundle.event.first_evidence_date or "",
            _normalize_for_sim(result[event_id]),
        ), []).append(bundle)
    for group in visual_groups.values():
        if len(group) <= 1:
            continue
        entities = {
            canonicalize_entity(bundle.event.canonical_entity)
            for bundle in group
        }
        if len(entities) <= 1:
            continue
        for bundle in group:
            event_id = bundle.event.event_id
            focused = atomic_event_summary(
                bundle.event, selected.get(event_id)
            )
            if focused:
                result[event_id] = focused
    return result


# -- Deduplication -----------------------------------------------------------

_REVIEW_RANK = {
    "accepted": 3, "corrected": 3, "pending": 2,
    "deferred": 1, "auto": 1,
}
_HUMAN_ACCEPTED = {"accepted", "corrected"}
_HUMAN_FLAGGED = {"rejected", "deferred"}

_CATEGORY_PRIORITY = {
    "diagnosis": 1, "comorbidity": 2, "clinical_sign": 3,
    "imaging_finding": 4, "laboratory_finding": 5, "histopathology": 6,
    "toxicity": 7, "adverse_event": 8, "symptom": 9, "procedure": 10,
    "surgery": 11, "medication": 12, "vital_sign": 13, "follow_up": 14,
    "other": 15,
}

_RESOLUTION_STATUSES = {
    "resolved", "completed", "suspended", "cancelled",
}

# Same-date tiers dominate clinical deduplication; buckets are normally small
# and compared exhaustively.  Larger buckets and all cross-date pairs fall
# back to rare-word blocking (two summaries with ratio >= 0.9 necessarily
# share at least one of their five least frequent words).
_DATE_BUCKET_EXHAUSTIVE = 200
_RARE_WORD_MAX_POSTINGS = 60


def _normalize_for_sim(text: str) -> str:
    value = unicodedata.normalize("NFKD", str(text or ""))
    value = "".join(char for char in value if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", value.casefold()).strip()


def _entity_similarity(a, b) -> float:
    ca = canonicalize_entity(a.event.canonical_entity)
    cb = canonicalize_entity(b.event.canonical_entity)
    if ca == cb:
        return 1.0
    if not ca or not cb:
        return 0.0
    return SequenceMatcher(None, ca, cb).ratio()


def _is_resolution_marked(event: ClinicalEvent) -> bool:
    return bool(
        event.date_end
        or str(event.status or "").strip().casefold() in _RESOLUTION_STATUSES
    )


def _numeric_values(
    bundle, evidence_by_id: dict[str, ClinicalEvidence]
) -> set[float]:
    values = set()
    for link in bundle.links:
        item = evidence_by_id.get(link.evidence_id)
        if item is not None and item.numeric_value is not None:
            values.add(round(item.numeric_value, 6))
    return values


def _pair_mergeable(
    a,
    b,
    *,
    clean_a: str,
    clean_b: str,
    evidence_by_id: dict[str, ClinicalEvidence],
) -> bool:
    na, nb = _normalize_for_sim(clean_a), _normalize_for_sim(clean_b)
    if not na or not nb:
        return False
    # Polarità: never fuse presence with absence, nor events carrying a
    # resolution marker (they document the end of a fact, not a restatement).
    if _is_resolution_marked(a.event) or _is_resolution_marked(b.event):
        return False
    if (a.event.assertion or "present") != (b.event.assertion or "present"):
        return False
    # Numeric conflict: distinct measured values for the same entity are
    # separate observations (protects the echo parameters whose only
    # difference is the value itself).
    values_a = _numeric_values(a, evidence_by_id)
    values_b = _numeric_values(b, evidence_by_id)
    if values_a and values_b and values_a != values_b:
        return False

    sim_summary = SequenceMatcher(None, na, nb).ratio()
    sim_entity = _entity_similarity(a, b)
    date_a = a.event.first_evidence_date
    date_b = b.event.first_evidence_date
    same_date = bool(date_a and date_b and date_a == date_b)
    if same_date:
        if a.event.category == b.event.category:
            return sim_summary >= 0.90 and sim_entity >= 0.85
        return sim_summary >= 0.90 and sim_entity >= 0.60

    threshold = 0.95
    if (
        a.event.category in {"surgery", "procedure"}
        or b.event.category in {"surgery", "procedure"}
    ):
        gap = temporal_distance_days(date_a, date_b)
        if gap is not None and gap > 30:
            threshold = 0.98
    return sim_summary >= threshold and sim_entity >= 0.90


def _rare_word_pairs(indices: list[int], cleaned: dict[int, str]) -> set[tuple[int, int]]:
    postings: dict[str, list[int]] = {}
    for index in indices:
        for word in set(str(cleaned.get(index, "")).split()):
            postings.setdefault(word, []).append(index)
    pairs: set[tuple[int, int]] = set()
    for word, positions in postings.items():
        if len(positions) > _RARE_WORD_MAX_POSTINGS or len(positions) < 2:
            continue
        for first in range(len(positions)):
            for second in range(first + 1, len(positions)):
                left, right = positions[first], positions[second]
                pairs.add((left, right) if left < right else (right, left))
    return pairs


def _candidate_pairs(
    bundles: list, cleaned: dict[int, str]
) -> list[tuple[int, int]]:
    """Cheap pre-filter: exhaustive inside small date buckets, rare-word
    blocking across dates.  Never rejects a pair whose summaries are close."""
    by_date: dict[str | None, list[int]] = {}
    for index, bundle in enumerate(bundles):
        by_date.setdefault(bundle.event.first_evidence_date, []).append(index)
    pairs: set[tuple[int, int]] = set()
    for positions in by_date.values():
        if len(positions) <= _DATE_BUCKET_EXHAUSTIVE:
            for first in range(len(positions)):
                for second in range(first + 1, len(positions)):
                    left, right = positions[first], positions[second]
                    pairs.add((left, right) if left < right else (right, left))
        else:
            pairs |= _rare_word_pairs(positions, cleaned)
    pairs |= _rare_word_pairs(list(range(len(bundles))), cleaned)
    return sorted(pairs)


def _survivor_key(bundle, cleaned: str, persisted_review_status: dict[str, str]):
    review = persisted_review_status.get(
        bundle.event.event_id, bundle.event.review_status
    )
    rank = _REVIEW_RANK.get(review, 1)
    priority = _CATEGORY_PRIORITY.get(bundle.event.category, 16)
    year, month, day = date_sort_key(bundle.event.first_evidence_date)
    return (
        rank,
        -priority,
        len(cleaned),
        bundle.event.confidence or 0.0,
        (-year, -month, -day),  # earliest date wins under max()
        bundle.event.event_id,
    )


def _persisted_review(
    bundle, persisted_review_status: dict[str, str]
) -> str:
    return persisted_review_status.get(
        bundle.event.event_id, bundle.event.review_status
    )


def _apply_clean_summary(
    bundle, cleaned: str, persisted_review_status: dict[str, str]
) -> None:
    if _persisted_review(bundle, persisted_review_status) in _HUMAN_ACCEPTED:
        return  # a clinician reviewed this prose; do not rewrite it
    bundle.event.summary_short = cleaned[:500]
    bundle.event.summary_detail = cleaned


def _merge_group(
    group: list[int],
    bundles: list,
    *,
    cleaned_by_event: dict[str, str],
    evidence_by_id: dict[str, ClinicalEvidence],
    persisted_review_status: dict[str, str],
) -> tuple:
    """Fold ``group`` (bundle indices) into its survivor; return
    ``(survivor_bundle, survivor_index)``."""

    def survivor_key(index: int):
        bundle = bundles[index]
        return _survivor_key(
            bundle,
            cleaned_by_event.get(bundle.event.event_id, ""),
            persisted_review_status,
        )

    survivor_index = max(group, key=survivor_key)
    survivor = bundles[survivor_index]
    others = [
        bundles[index] for index in group if index != survivor_index
    ]
    group_bundles = others + [survivor]
    survivor_event = survivor.event

    merged_into = [bundle.event.event_id for bundle in others]
    survivor_event.structured_data.setdefault("evidence_ids", [])
    known_ids = set(survivor_event.structured_data["evidence_ids"])
    for bundle in others:
        for evidence_id in bundle.event.structured_data.get("evidence_ids", []):
            if evidence_id not in known_ids:
                known_ids.add(evidence_id)
                survivor_event.structured_data["evidence_ids"].append(evidence_id)
    survivor_event.structured_data["merged_into_ids"] = merged_into

    evidence_dates = [
        bundle.event.first_evidence_date for bundle in group_bundles
        if bundle.event.first_evidence_date
    ]
    if evidence_dates:
        earliest = min(evidence_dates, key=date_sort_key)
        survivor_event.first_evidence_date = earliest
        survivor.episode.onset_date = earliest
    documented = [
        bundle.event.first_documented_date for bundle in group_bundles
        if bundle.event.first_documented_date
    ]
    if documented:
        earliest_documented = min(documented, key=date_sort_key)
        survivor_event.first_documented_date = earliest_documented
        survivor.episode.first_documented_date = earliest_documented
    survivor_event.confidence = max(
        bundle.event.confidence or 0.0 for bundle in group_bundles
    ) or survivor_event.confidence

    _apply_clean_summary(
        survivor,
        cleaned_by_event[survivor.event.event_id],
        persisted_review_status,
    )

    cross_category = len({
        bundle.event.category for bundle in group_bundles
    }) > 1
    cross_date = len({
        bundle.event.first_evidence_date for bundle in group_bundles
    }) > 1
    if (cross_date or cross_category) and _persisted_review(
        survivor, persisted_review_status
    ) not in _HUMAN_ACCEPTED:
        survivor_event.review_status = "pending"

    # Fold links: keep the survivor's own untouched, append one link per newly
    # merged evidence with a deterministic id.
    own_links = {link.evidence_id for link in survivor.links}
    for bundle in others:
        for link in bundle.links:
            if link.evidence_id in own_links:
                continue
            relation = link.relation if link.relation in {
                "supports", "contradicts", "excluded", "correlated", "updates"
            } else "supports"
            confidence = link.relation_confidence or 0.6
            survivor.links.append(EventEvidenceLink(
                link_id=stable_id(
                    "LNK", survivor.event.event_id, link.evidence_id, relation
                ),
                event_id=survivor.event.event_id,
                evidence_id=link.evidence_id,
                relation=relation,
                role=link.role,
                relation_confidence=confidence,
                rationale=(
                    link.rationale or "Evidenza unificata per evento duplicato"
                ),
                included_in_summary=link.included_in_summary,
            ))
            own_links.add(link.evidence_id)

    # Rebuild updates from the full merged evidence set so no documented
    # change is lost; update ids are deterministic and cannot collide with the
    # survivor's previous ones.
    merged_evidence = [
        evidence_by_id[evidence_id]
        for evidence_id in own_links if evidence_id in evidence_by_id
    ]
    survivor.updates = build_updates(
        survivor.event.event_id,
        merged_evidence,
        survivor_event.first_evidence_date,
    )
    return survivor, survivor_index


def deduplicate_bundles(
    bundles: list,
    *,
    evidence_by_id: dict[str, ClinicalEvidence],
    persisted_review_status: dict[str, str],
    evidence_relations=None,
) -> tuple[list, int]:
    """Clean every bundle summary and merge cross-document duplicates.

    Returns ``(final_bundles, merged_count)``.  ``final_bundles`` contains one
    bundle per survivor (its identity and event id preserved) plus every
    bundle that was not part of a merge group.  Merged-away bundles disappear,
    so ``replace_generated_registry`` deletes their stale rows.
    """
    bundles = list(bundles)
    graph_pairs = None
    if evidence_relations is not None:
        graph_pairs = {
            frozenset((
                relation.source_evidence_id,
                relation.target_evidence_id,
            ))
            for relation in evidence_relations
            if relation.cluster_effect in {"must_link", "cohesive"}
        }
    cleaned_by_event: dict[str, str] = clean_bundle_summaries(
        bundles, evidence_by_id
    )
    cleaned = {
        index: cleaned_by_event[bundle.event.event_id]
        for index, bundle in enumerate(bundles)
    }

    valid_pairs: set[tuple[int, int]] = set()
    for left, right in _candidate_pairs(bundles, cleaned):
        if graph_pairs is not None and not _bundles_linked_by_graph(
            bundles[left], bundles[right], graph_pairs
        ):
            continue
        if _pair_mergeable(
            bundles[left], bundles[right],
            clean_a=cleaned[left], clean_b=cleaned[right],
            evidence_by_id=evidence_by_id,
        ):
            valid_pairs.add((left, right))

    parent = list(range(len(bundles)))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_left] = root_right

    for left, right in valid_pairs:
        union(left, right)

    components: dict[int, list[int]] = {}
    for index in range(len(bundles)):
        components.setdefault(find(index), []).append(index)

    merged_away: list[int] = []
    final: list = []
    for component in components.values():
        if len(component) == 1:
            bundle = bundles[component[0]]
            _apply_clean_summary(
                bundle, cleaned[component[0]], persisted_review_status
            )
            final.append(bundle)
            continue
        # Transitivity guard: every member must be mergeable with every other,
        # otherwise one duplicate would drag in unrelated facts.
        consistent = all(
            (left, right) in valid_pairs
            for first, left in enumerate(component)
            for right in component[first + 1:]
        )
        if not consistent:
            for index in component:
                bundle = bundles[index]
                _apply_clean_summary(
                    bundle, cleaned[index], persisted_review_status
                )
                final.append(bundle)
            continue
        reviewed = [
            _persisted_review(bundles[index], persisted_review_status)
            for index in component
        ]
        if any(status in _HUMAN_FLAGGED for status in reviewed):
            for index in component:
                bundle = bundles[index]
                _apply_clean_summary(
                    bundle, cleaned[index], persisted_review_status
                )
                final.append(bundle)
            continue
        if sum(1 for status in reviewed if status in _HUMAN_ACCEPTED) > 1:
            for index in component:
                bundle = bundles[index]
                _apply_clean_summary(
                    bundle, cleaned[index], persisted_review_status
                )
                final.append(bundle)
            continue
        survivor, survivor_index = _merge_group(
            component,
            bundles,
            cleaned_by_event=cleaned_by_event,
            evidence_by_id=evidence_by_id,
            persisted_review_status=persisted_review_status,
        )
        final.append(survivor)
        merged_away.extend(
            index for index in component if index != survivor_index
        )

    return final, len(merged_away)


def _bundles_linked_by_graph(left, right, graph_pairs: set[frozenset]) -> bool:
    """Require source identity or an explicit cohesive evidence edge."""
    left_ids = {link.evidence_id for link in left.links}
    right_ids = {link.evidence_id for link in right.links}
    if left_ids & right_ids:
        return True
    return any(
        frozenset((left_id, right_id)) in graph_pairs
        for left_id in left_ids for right_id in right_ids
    )
