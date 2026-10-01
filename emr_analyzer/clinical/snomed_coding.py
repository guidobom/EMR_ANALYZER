"""SNOMED CT coding of clinical concepts, once per label and event type.

Occurrences of the same concept share one mapping, stored in the shared
Lexicon database: coding is consistent across documents and patients, a new
concept costs model calls once, and a reviewer decision applies everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
import time
import unicodedata

import jsonschema

from .evidence_utils import content_hash
from .grounded_sources import METHOD
from ..prompt_catalog import load_prompt

# SNOMED hierarchies (semantic tags) admissible for each extracted event type.
FACT_TYPE_TAGS = {
    "diagnosis": ("disorder", "finding", "morphologic abnormality", "situation"),
    "symptom": ("finding", "disorder"),
    "clinical_sign": ("finding", "disorder", "observable entity"),
    "histopathology": ("morphologic abnormality", "disorder", "finding"),
    "radiology_finding": ("finding", "disorder", "morphologic abnormality"),
    "instrumental_finding": ("finding", "disorder", "observable entity", "morphologic abnormality"),
    "laboratory_test": ("observable entity", "procedure", "finding"),
    "vital_sign": ("observable entity", "finding"),
    "biomarker": ("observable entity", "finding", "substance", "procedure"),
    "medication": ("substance", "medicinal product", "clinical drug", "medicinal product form", "product"),
    "procedure": ("procedure", "regime/therapy"),
    "clinical_decision": ("procedure", "regime/therapy", "situation"),
    "hospitalization": ("procedure", "event", "situation"),
    "discharge": ("procedure", "event", "situation"),
}


class CodingCancelled(RuntimeError):
    pass


def concept_key(label, fact_type) -> tuple[str, str]:
    """Identity of a concept: normalized label and event type."""
    text = unicodedata.normalize("NFKC", str(label or "")).casefold()
    text = " ".join(text.split()).strip(" .,;:")
    return text, str(fact_type or "")


@dataclass
class Concept:
    key: tuple[str, str]
    label: str
    fact_type: str
    count: int = 0
    examples: list = field(default_factory=list)


def collect_concepts(events) -> dict[tuple[str, str], Concept]:
    concepts: dict[tuple[str, str], Concept] = {}
    for item in events:
        if item.extraction_method != METHOD or not (item.normalized_entity or "").strip():
            continue
        key = concept_key(item.normalized_entity, item.fact_type)
        concept = concepts.setdefault(key, Concept(key, item.normalized_entity.strip(), item.fact_type))
        concept.count += 1
        quote = " ".join(str(item.source_text or "").split())[:300]
        if quote and quote not in concept.examples and len(concept.examples) < 2:
            concept.examples.append(quote)
    return concepts


class ConceptCoder:
    def __init__(self, llm, snomed, mappings, *, coding_system=None, translation_system=None,
                 batch_size=8, candidate_limit=10):
        self.llm = llm
        self.snomed = snomed
        self.mappings = mappings
        self.coding_system = coding_system or load_prompt("snomed_concept_coding_system")
        self.translation_system = translation_system or load_prompt("snomed_query_translation_system")
        self.batch_size = batch_size
        self.candidate_limit = candidate_limit
        self.metrics: dict = {}

    @property
    def available(self) -> bool:
        return bool(self.snomed is not None and self.snomed.available and self.mappings is not None)

    @property
    def model_digest(self) -> str:
        return content_hash(self.coding_system, self.translation_system, self.candidate_limit,
                            json.dumps(FACT_TYPE_TAGS, sort_keys=True),
                            json.dumps({key: getattr(self.llm, key, None) for key in (
                                "model", "model_path", "temperature", "seed", "top_p", "top_k",
                                "thinking_enabled")}, sort_keys=True))

    def resolve(self, events) -> dict[tuple[str, str], dict]:
        """Current mapping of each concept occurring in ``events``."""
        if self.mappings is None:
            return {}
        return self.mappings.get_many(collect_concepts(events).keys())

    def code_events(self, events, *, cancel_check=None, progress=None) -> dict:
        """Code every concept of ``events`` that has no mapping yet."""
        concepts = collect_concepts(events)
        self.metrics = dict(concepts=len(concepts), cached=0, aliases=0, coded=0, abstained=0,
                            no_candidates=0, errors=0, llm_calls=0)
        if not concepts or not self.available:
            return dict(self.metrics)
        known = self.mappings.get_many(concepts.keys())
        pending = [concept for key, concept in concepts.items() if key not in known]
        self.metrics["cached"] = len(concepts) - len(pending)
        if not pending or self.llm is None or not getattr(self.llm, "is_available", True):
            return dict(self.metrics)

        def check():
            if cancel_check and cancel_check():
                raise CodingCancelled("Codifica SNOMED interrotta")

        release = (self.snomed.metadata().get("version_uri") or self.snomed.metadata().get("release"))
        remaining = []
        for concept in pending:
            alias = self._alias(concept)
            if alias:
                self.mappings.confirm(*concept.key, concept.label, code=alias["code"], concept=alias,
                                      release=release, note="Alias italiano confermato nel catalogo")
                self.metrics["aliases"] += 1
            else:
                remaining.append(concept)
        english = {} if self.snomed.metadata().get("embedding_model") else self._translate(remaining, check)
        candidates = {}
        for concept in remaining:
            check()
            found = self.snomed.search(concept.label, english.get(concept.key, ""),
                                       limit=self.candidate_limit,
                                       tags=FACT_TYPE_TAGS.get(concept.fact_type))
            if found:
                candidates[concept.key] = found
            else:
                self.mappings.propose(*concept.key, concept.label, status="needs_review",
                                      release=release, english=english.get(concept.key),
                                      examples=concept.examples, model_digest=self.model_digest,
                                      note="Nessun candidato nel catalogo per questa gerarchia")
                self.metrics["no_candidates"] += 1
        queue = [concept for concept in remaining if concept.key in candidates]
        done = 0
        while queue:
            check()
            batch = queue[:self.batch_size]
            del queue[:len(batch)]
            while len(batch) > 1 and not self._fits(batch, candidates):
                queue.insert(0, batch.pop())
            self._select(batch, candidates, english, release)
            done += len(batch)
            if progress:
                progress(done, len(candidates))
        return dict(self.metrics)

    # ---------------------------------------------------------------- steps
    def _alias(self, concept) -> dict | None:
        from .snomed_catalog import words
        row = self.snomed.db.execute("SELECT code FROM sct_aliases WHERE text=?",
                                     (" ".join(words(concept.label)),)).fetchone()
        found = self.snomed.lookup(row[0]) if row else None
        tags = FACT_TYPE_TAGS.get(concept.fact_type)
        if found and found["active"] and (not tags or found["tag"] in tags):
            return found
        return None

    def _translate(self, concepts, check) -> dict:
        result = {}
        for start in range(0, len(concepts), 16):
            check()
            batch = concepts[start:start + 16]
            payload = [{"id": str(index), "label": concept.label, "type": concept.fact_type,
                        "context": concept.examples[:1]} for index, concept in enumerate(batch)]
            contract = {"type": "object", "additionalProperties": False, "required": ["queries"],
                        "properties": {"queries": {"type": "array", "minItems": len(batch),
                                                   "maxItems": len(batch), "items": {
                            "type": "object", "additionalProperties": False, "required": ["id", "english"],
                            "properties": {"id": {"type": "string", "enum": [p["id"] for p in payload]},
                                           "english": {"type": "string", "minLength": 1, "maxLength": 160}}}}}}
            try:
                answer = self._call(json.dumps(payload, ensure_ascii=False), self.translation_system,
                                    contract, max_tokens=1024)
                for item in answer["queries"]:
                    result[batch[int(item["id"])].key] = item["english"]
            except CodingCancelled:
                raise
            except Exception:
                self.metrics["errors"] += 1      # the Italian label is still searched
        return result

    def _payload(self, batch, candidates):
        return [{"id": str(index), "etichetta": concept.label, "tipo": concept.fact_type,
                 "esempi": concept.examples,
                 "candidati": [{"code": row["code"], "fsn": row["fsn"],
                                "parents": [parent["fsn"] for parent in row.get("parents", [])[:2]]}
                               for row in candidates[concept.key]]}
                for index, concept in enumerate(batch)]

    def _contract(self, payload):
        codes = sorted({row["code"] for item in payload for row in item["candidati"]})
        return {"type": "object", "additionalProperties": False, "required": ["mappings"],
                "properties": {"mappings": {"type": "array", "items": {
                    "type": "object", "additionalProperties": False, "required": ["id", "code"],
                    "properties": {"id": {"type": "string", "enum": [item["id"] for item in payload]},
                                   "code": {"type": ["string", "null"], "enum": codes + [None]}}}}}}

    def _fits(self, batch, candidates) -> bool:
        payload = self._payload(batch, candidates)
        prompt = json.dumps(payload, ensure_ascii=False) + json.dumps(self._contract(payload))
        counter = getattr(self.llm, "count_prompt_tokens", None)
        try:
            size = counter(prompt, self.coding_system) if callable(counter) \
                else len((prompt + self.coding_system).encode())
        except Exception:
            size = len((prompt + self.coding_system).encode())
        output = min(1024, int(getattr(self.llm, "max_output_tokens", 4096)))
        return size + output + 256 <= int(getattr(self.llm, "context_length", 32768))

    def _select(self, batch, candidates, english, release) -> None:
        payload = self._payload(batch, candidates)
        contract = self._contract(payload)
        try:
            answer = self._call(json.dumps(payload, ensure_ascii=False), self.coding_system, contract,
                                max_tokens=min(1024, int(getattr(self.llm, "max_output_tokens", 4096))))
            choices = {item["id"]: item["code"] for item in answer["mappings"]}
            if set(choices) != {item["id"] for item in payload}:
                raise ValueError("Risposta incompleta o identificativi duplicati")
            for index, concept in enumerate(batch):
                if choices[str(index)] not in {row["code"] for row in candidates[concept.key]} | {None}:
                    raise ValueError("Codice non presente nei candidati del concetto")
        except CodingCancelled:
            raise
        except Exception:
            self.metrics["errors"] += 1          # retried at the next run
            return
        for index, concept in enumerate(batch):
            code = choices[str(index)]
            chosen = next((row for row in candidates[concept.key] if row["code"] == code), None)
            stored = [{"code": row["code"], "fsn": row["fsn"], "term": row["term"]}
                      for row in candidates[concept.key]]
            self.mappings.propose(
                *concept.key, concept.label,
                status="proposed" if chosen else "needs_review", code=code, concept=chosen,
                release=release, english=english.get(concept.key), candidates=stored,
                examples=concept.examples, model_digest=self.model_digest,
                note=None if chosen else "Il modello non ha trovato un candidato adeguato")
            self.metrics["coded" if chosen else "abstained"] += 1

    def _call(self, prompt, system, contract, *, max_tokens):
        started = time.monotonic()
        try:
            answer = self.llm.generate_structured(prompt, system, contract, max_tokens=max_tokens)
        finally:
            self.metrics["llm_calls"] += 1
            self.metrics["elapsed_seconds"] = self.metrics.get("elapsed_seconds", 0) + time.monotonic() - started
        jsonschema.validate(answer, contract)
        return answer


class MemoryConceptMappings:
    """Mapping store that keeps results in memory (prompt previews, dry runs)."""

    def __init__(self):
        self.rows: dict[tuple[str, str], dict] = {}

    def get_many(self, keys):
        return {key: self.rows[key] for key in keys if key in self.rows}

    def propose(self, label_key, fact_type, label, *, status, code=None, concept=None, **fields):
        self.rows[(label_key, fact_type)] = {
            "label_key": label_key, "fact_type": fact_type, "label": label, "status": status,
            "code": code, "fsn": (concept or {}).get("fsn"), "term": (concept or {}).get("term"),
            "source": fields.get("source", "llm"), "english": fields.get("english"),
            "candidates": list(fields.get("candidates", ())), "examples": list(fields.get("examples", ())),
            "note": fields.get("note"), "release": fields.get("release")}
        return True

    def confirm(self, label_key, fact_type, label, *, code=None, concept=None, release=None, note=None):
        self.rows[(label_key, fact_type)] = {
            "label_key": label_key, "fact_type": fact_type, "label": label, "status": "confirmed",
            "code": code, "fsn": (concept or {}).get("fsn"), "term": (concept or {}).get("term"),
            "source": "manual", "note": note, "release": release}
