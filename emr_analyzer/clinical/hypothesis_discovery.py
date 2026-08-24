"""Bounded, review-only discovery of hypotheses between clinical events."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import inspect
import json
import uuid

from ..models.clinical_pipeline import ClinicalHypothesis


_SYSTEM_PROMPT = (
    "Proponi soltanto collegamenti clinici plausibili da sottoporre a revisione "
    "umana. Non trasformare una semplice vicinanza temporale in causalità e "
    "non aggiungere fatti assenti. Rispondi esclusivamente con JSON conforme "
    "allo schema."
)

_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "hypotheses": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "candidate_id": {"type": "string"},
                    "plausible": {"type": "boolean"},
                    "hypothesis_type": {"type": "string"},
                    "strength": {
                        "type": "string",
                        "enum": ["weak", "moderate", "strong"],
                    },
                    "known_mechanism": {"type": "boolean"},
                    "rationale": {"type": "string"},
                    "confounders": {
                        "type": "array", "items": {"type": "string"},
                    },
                    "survival_score": {
                        "type": ["number", "null"],
                        "minimum": 0,
                        "maximum": 1,
                    },
                },
                "required": [
                    "candidate_id", "plausible", "hypothesis_type",
                    "strength", "known_mechanism", "rationale",
                    "confounders", "survival_score",
                ],
            },
        },
    },
    "required": ["hypotheses"],
}


@dataclass(slots=True)
class HypothesisCandidate:
    candidate_id: str
    source_event_id: str
    target_event_id: str
    temporal_distance_days: int | None


class HypothesisDiscovery:
    """Generate a bounded candidate set and ask an LLM only to review it."""

    def __init__(self, registry_repo, pipeline_repo, llm_client):
        self.registry_repo = registry_repo
        self.pipeline_repo = pipeline_repo
        self.llm = llm_client

    def run(self, patient_id: str, *, top_k: int = 4) -> dict:
        if not self.llm or not getattr(self.llm, "is_available", True):
            raise RuntimeError("Modello per gli eventi clinici non disponibile")
        events = self.registry_repo.get_events(patient_id)
        candidates = self._candidates(events, top_k=max(1, min(12, top_k)))
        by_id = {event.event_id: event for event in events}
        hypotheses = []
        calls = 0
        for offset in range(0, len(candidates), 24):
            batch = candidates[offset:offset + 24]
            if not batch:
                continue
            calls += 1
            payload = self._classify(batch, by_id)
            hypotheses.extend(self._validated(patient_id, batch, payload))
        self.pipeline_repo.replace_pending_hypotheses(patient_id, hypotheses)
        return {
            "candidate_count": len(candidates),
            "hypothesis_count": len(hypotheses),
            "llm_calls": calls,
        }

    @staticmethod
    def _candidates(events, *, top_k: int) -> list[HypothesisCandidate]:
        ordered = sorted(events, key=lambda event: (
            event.first_evidence_date or event.first_documented_date or "9999",
            event.event_id,
        ))
        pairs = {}
        for index, event in enumerate(ordered):
            for other in ordered[max(0, index - top_k):index]:
                if event.category == other.category and (
                    event.canonical_entity == other.canonical_entity
                ):
                    continue
                source, target = sorted((event.event_id, other.event_id))
                token = f"{source}:{target}"
                pairs[(source, target)] = HypothesisCandidate(
                    candidate_id="HCAN_" + uuid.uuid5(
                        uuid.NAMESPACE_URL, token
                    ).hex,
                    source_event_id=source,
                    target_event_id=target,
                    temporal_distance_days=_distance(event, other),
                )
        return [pairs[key] for key in sorted(pairs)]

    def _classify(self, batch, by_id) -> dict:
        rows = []
        for item in batch:
            rows.append({
                "candidate_id": item.candidate_id,
                "temporal_distance_days": item.temporal_distance_days,
                "source": _event_payload(by_id[item.source_event_id]),
                "target": _event_payload(by_id[item.target_event_id]),
            })
        prompt = (
            "Valuta i candidati seguenti. plausible=true anche per una "
            "relazione clinicamente plausibile ma non documentata, purché "
            "l'incertezza sia esplicita. Ogni output deve riusare esattamente "
            "un candidate_id fornito. Le ipotesi saranno escluse dal RAG fino "
            "alla revisione umana.\n\nCANDIDATI:\n"
            + json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
        )
        generator = self.llm.generate_structured
        parameters = inspect.signature(generator).parameters
        if "max_tokens" in parameters or any(
            value.kind == inspect.Parameter.VAR_KEYWORD
            for value in parameters.values()
        ):
            return generator(
                prompt, _SYSTEM_PROMPT, _SCHEMA,
                max_tokens=min(
                    int(getattr(self.llm, "max_output_tokens", 4096) or 4096),
                    max(1024, 180 * len(batch)),
                ),
            )
        return generator(prompt, _SYSTEM_PROMPT, _SCHEMA)

    @staticmethod
    def _validated(patient_id, batch, payload) -> list[ClinicalHypothesis]:
        if not isinstance(payload, dict):
            return []
        candidates = {item.candidate_id: item for item in batch}
        seen = set()
        result = []
        for raw in payload.get("hypotheses") or []:
            if not isinstance(raw, dict) or raw.get("plausible") is not True:
                continue
            candidate_id = str(raw.get("candidate_id") or "")
            candidate = candidates.get(candidate_id)
            hypothesis_type = " ".join(
                str(raw.get("hypothesis_type") or "").split()
            )[:120]
            rationale = " ".join(
                str(raw.get("rationale") or "").split()
            )[:1200]
            key = (candidate_id, hypothesis_type.casefold())
            if (
                candidate is None or key in seen or not hypothesis_type
                or not rationale
                or raw.get("strength") not in {"weak", "moderate", "strong"}
                or not isinstance(raw.get("known_mechanism"), bool)
            ):
                continue
            score = raw.get("survival_score")
            if score is not None and (
                isinstance(score, bool) or not isinstance(score, (int, float))
                or not 0 <= float(score) <= 1
            ):
                continue
            confounders = [
                " ".join(str(item).split())[:300]
                for item in (raw.get("confounders") or [])
                if " ".join(str(item).split())
            ][:12]
            seen.add(key)
            stable = (
                f"{patient_id}:{candidate.source_event_id}:"
                f"{candidate.target_event_id}:{hypothesis_type.casefold()}"
            )
            result.append(ClinicalHypothesis(
                hypothesis_id="HYP_" + uuid.uuid5(
                    uuid.NAMESPACE_URL, stable
                ).hex,
                patient_id=patient_id,
                source_event_id=candidate.source_event_id,
                target_event_id=candidate.target_event_id,
                hypothesis_type=hypothesis_type,
                strength=str(raw["strength"]),
                known_mechanism=raw["known_mechanism"],
                rationale=rationale,
                confounders=confounders,
                survival_score=float(score) if score is not None else None,
                status="needs_review",
            ))
        return result


def _event_payload(event) -> dict:
    return {
        "event_id": event.event_id,
        "category": event.category,
        "entity": event.canonical_entity,
        "summary": event.summary_short,
        "date": event.first_evidence_date,
        "certainty": event.certainty,
        "assertion": event.assertion,
    }


def _distance(left, right) -> int | None:
    try:
        left_date = date.fromisoformat(
            (left.first_evidence_date or left.first_documented_date)[:10]
        )
        right_date = date.fromisoformat(
            (right.first_evidence_date or right.first_documented_date)[:10]
        )
    except (TypeError, ValueError):
        return None
    return abs((left_date - right_date).days)
