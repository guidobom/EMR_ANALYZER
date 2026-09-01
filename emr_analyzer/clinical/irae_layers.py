"""Structured 3-layer irAE analysis shared by the headless tool and the GUI.

Layers 1-2 are the deterministic lexicon scan in ``irae_prototype`` (temporal
anchoring to the first checkpoint-inhibitor dose, then an organ toxicity
lexicon over the atomic evidence).  Layer 3 is one structured LLM call per
organ over the pre-filtered candidates (citable by ``[#id]``), with the
constrained JSON schema below; the model never free-generates codes.

Both ``tools/irae_headless.py`` and the GUI batch queue ("Coda registri →
Analisi irAE", structured method) delegate here, so the refined NCTCAE prompt
and schema stay in one place.
"""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from typing import Any, Iterable

from .irae_prototype import (
    IciAnchor,
    ToxicityCandidate,
    find_ici_anchor,
    scan_for_irae,
    summarize_candidates,
)

DEFAULT_MAX_CANDIDATES = 60
DEFAULT_MAX_TOKENS = 8192
DEFAULT_PARALLEL = 4
DEFAULT_MAX_CONSOLIDATION_FINDINGS = 200

# System prompt for the structured layer: the NCI CTCAE is the single
# reference for adverse-event detection and grading (the user's hard rule).
SYSTEM_PROMPT = (
    "Sei un oncologo medico esperto nella diagnosi, classificazione e "
    "gestione delle tossicità immuno-correlate (immune-related adverse "
    "events, irAE) associate alle immunoterapie oncologiche. Analizza i "
    "candidati pre-filtrati forniti e restituisci SOLO un oggetto JSON "
    "valido che rispetta esattamente lo schema richiesto, senza testo "
    "prima o dopo. Per ogni irAE identifica: tipo; grado CTCAE; data di "
    "prima insorgenza; probabilità di origine immuno-correlata. "
    "Riferimento unico per la rilevazione e la classificazione degli "
    "eventi avversi: NCTCAE (NCI Common Terminology Criteria for Adverse "
    "Events), versione 6.0: confronta sistematicamente i candidati con i "
    "capitoli CTCAE di tossicità immuno-correlata — cute (rash, prurito, "
    "vitiligine), gastrointestinali (colite, diarrea, epatite, "
    "pancreatite, mucosite), endocrine (tiroidite, ipofisite, "
    "insufficienza surrenalica, diabete), polmonari (polmonite, "
    "interstiziopatia), cardiache (miocardite, troponina elevata, "
    "aritmia), muscoloscheletriche (artralgia, miosite, CPK elevata), "
    "renali (nefrite, creatinina elevata), neurologiche (polineuropatia, "
    "miastenia, encefalite, mielite, diplopia), oculari (uveite, "
    "cheratite, sclerite), ematologiche (neutropenia, anemia emolitica, "
    "trombocitopenia), urologiche (cistite). Distingui esplicitamente i "
    "reperti legati alla malattia tumorale (es. lesioni cutanee del "
    "melanoma, captazioni surrenaliche da metastasi) da una vera tossicità "
    "immuno-correlata. Privilegia la precisione: riporta un irAE SOLO quando "
    "i dati lo supportano; non segnalare automaticamente come "
    "immuno-correlato ogni evento avvenuto durante l'immunoterapia. Un "
    "singolo valore di laboratorio borderline o un sintomo aspecifico senza "
    "conferma — seconda rilevazione o trend, oppure diagnosi clinica scritta "
    "nel referto — non è sufficiente per un irAE PROBABILE o "
    "CERTA_CONFERMATA. Usa POSSIBILE quando i dati sono insufficienti e "
    "IMPROBABILE per i reperti della malattia tumorale, motivando ogni scelta "
    "con diagnostic_support. Eventi già presenti prima dell'inizio "
    "dell'immunoterapia (baseline) o spiegabili da cause alternative meglio "
    "supportate dai dati (progressione, infezione, chemioterapia) non sono "
    "irAE."
)

# Constrained JSON schema: forces llama-server GBNF decoding and keeps the
# output small (synthesis per organ, never per candidate).
LAYER3_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "iraes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "organ": {"type": "string"},
                    "irAE_type": {
                        "type": "string",
                        "description": "Tipologia dell'irAE (es. epatite immunomediata, colite, polmonite, artralgia).",
                    },
                    "ctcae_grade": {
                        "type": "string",
                        "enum": [
                            "G1", "G2", "G3", "G4", "G5",
                            "non_determinabile",
                        ],
                    },
                    "first_onset_date": {
                        "type": "string",
                        "description": "Data di prima insorgenza YYYY-MM-DD o YYYY-MM.",
                    },
                    "probability_immune": {
                        "type": "string",
                        "enum": [
                            "CERTA_CONFERMATA", "PROBABILE", "POSSIBILE",
                            "IMPROBABILE", "INDETERMINATA",
                        ],
                    },
                    "new_onset_vs_exacerbation": {
                        "type": "string",
                        "description": "insorgenza nuova | riacutizzazione di condizione preesistente",
                    },
                    "alternative_causes": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "key_evidence_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Identificatori [#id] delle evidenze chiave.",
                    },
                    "diagnostic_support": {
                        "type": "string",
                        "enum": [
                            "conferma_clinica_scritta",
                            "trend_evidenze_multiple",
                            "risposta_a_steroidi_o_sospensione",
                            "valore_isolato",
                            "preesistente_baseline",
                            "non_specificato",
                        ],
                        "description": "Base diagnostica dell'irAE: diagnosi clinica scritta nel referto, più rilevazioni/trend, risposta a steroidi o sospensione dell'immunoterapia, singolo valore/sintomo isolato, preesistente all'immunoterapia, o non specificata.",
                    },
                },
                "required": [
                    "organ", "irAE_type", "ctcae_grade", "first_onset_date",
                    "probability_immune", "diagnostic_support",
                ],
            },
        }
    },
    "required": ["iraes"],
}

# Layer 4 (final consolidation): one structured call over the per-organ
# irAEs.  Merges the same clinical event detected in different organs and
# deepens each final irAE's characterization.
SYSTEM_PROMPT_CONSOLIDATION = (
    "Sei un oncologo medico esperto nella diagnosi, classificazione e "
    "gestione delle tossicità immuno-correlate (irAE) associate alle "
    "immunoterapie oncologiche. Ti vengono presentati gli irAE identificati "
    "per singolo organo dalle analisi precedenti: la tua FASE FINALE è "
    "produrre la lista consolidata e approfondita, unendo i duplicati che "
    "descrivono lo stesso evento clinico anche se rilevati in organi "
    "diversi. Riferimento unico per la rilevazione e la classificazione "
    "degli eventi avversi: NCTCAE (NCI CTCAE), versione 6.0. Restituisci "
    "SOLO un oggetto JSON valido che rispetta esattamente lo schema "
    "richiesto, senza testo prima o dopo."
)

LAYER4_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "iraes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "irAE_type": {"type": "string"},
                    "ctcae_grade": {
                        "type": "string",
                        "enum": [
                            "G1", "G2", "G3", "G4", "G5",
                            "non_determinabile",
                        ],
                    },
                    "first_onset_date": {"type": "string"},
                    "probability_immune": {
                        "type": "string",
                        "enum": [
                            "CERTA_CONFERMATA", "PROBABILE", "POSSIBILE",
                            "IMPROBABILE", "INDETERMINATA",
                        ],
                    },
                    "new_onset_vs_exacerbation": {"type": "string"},
                    "alternative_causes": {"type": "string"},
                    "confidence": {
                        "type": "number", "minimum": 0, "maximum": 1,
                    },
                    "key_evidence_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "source_organs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Organi in cui lo stesso irAE è stato rilevato.",
                    },
                    "notes": {
                        "type": "string",
                        "description": "Approfondimento clinico: caratterizzazione, decorso, correlazione temporale con l'immunoterapia.",
                    },
                },
                "required": [
                    "irAE_type", "ctcae_grade", "first_onset_date",
                    "probability_immune", "key_evidence_ids",
                ],
            },
        }
    },
    "required": ["iraes"],
}


def evidence_rows_from_models(
    evidence: Iterable[Any],
) -> list[dict[str, Any]]:
    """Bridge ``ClinicalEvidence`` objects into the row dicts the prototype
    lexicon expects.

    ``ClinicalEvidence.to_dict()`` keeps the extraction payload under ``data``
    while the deterministic layer reads ``data_json``; this normalizes the
    key and carries the quantitative lab fields forward.
    """
    def _get(obj: Any, key: str, default: Any = "") -> Any:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    rows: list[dict[str, Any]] = []
    for item in evidence:
        data = _get(item, "data", None)
        if not isinstance(data, dict):
            data = _get(item, "data_json", None)
        if not isinstance(data, dict):
            data = {}
        bbox = _get(item, "bbox", None)
        rows.append({
            "evidence_id": _get(item, "evidence_id", ""),
            "category": _get(item, "category", ""),
            "normalized_entity": _get(item, "normalized_entity", ""),
            "observed_date": _get(item, "observed_date", ""),
            "data_json": data,
            "value_text": _get(item, "value_text", ""),
            "numeric_value": _get(item, "numeric_value", None),
            "unit": _get(item, "unit", ""),
            "document_id": _get(item, "document_id", ""),
            "source_page": _get(item, "source_page", None),
            "bbox": list(bbox) if bbox else None,
            "source_text": _get(item, "source_text", ""),
        })
    return rows


def _compact_evidence(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Slim evidence rows for the report, without the extraction payload.

    The report carries both the Layer 2 ``candidates`` and this compact list
    so the inspector can render the evidence cited by the model even when it
    is not a deterministic candidate (``key_evidence_ids`` may point to any
    atomic evidence).  Only provenance and a short value are kept — never
    ``data_json``.
    """
    compact: list[dict[str, Any]] = []
    for row in rows:
        bbox = row.get("bbox")
        compact.append({
            "evidence_id": row.get("evidence_id") or "",
            "document_id": row.get("document_id") or "",
            "source_page": row.get("source_page"),
            "bbox": list(bbox) if bbox else None,
            "source_text": row.get("source_text") or "",
            "normalized_entity": row.get("normalized_entity") or "",
            "category": row.get("category") or "",
            "observed_date": row.get("observed_date") or "",
            "value_text": row.get("value_text") or "",
        })
    return compact


def format_candidate_line(candidate: ToxicityCandidate) -> str:
    offset = (
        f"d={candidate.offset_days:+d}g"
        if candidate.offset_days is not None
        else "data non risolta"
    )
    line = (
        f"[#{candidate.evidence_id}] [{candidate.observed_raw}] "
        f"[{candidate.band} {offset}] [{candidate.organ}] "
        f"[{candidate.category}] {candidate.entity}"
    )
    if candidate.value:
        line += f" — VALORE: {candidate.value}"
    if candidate.reference:
        line += f" — REFERENZA: {candidate.reference}"
    if candidate.quote:
        line += f"\n    citazione: “{candidate.quote[:200]}”"
    return line


def build_organ_prompt(
    organ: str,
    candidates: list[ToxicityCandidate],
    anchor: IciAnchor | None,
    max_candidates: int,
) -> tuple[str, int]:
    """Layer 3 prompt for ONE organ over its pre-filtered candidates."""
    dropped = 0
    kept = candidates
    if len(kept) > max_candidates:
        dropped = len(kept) - max_candidates
        kept = kept[:max_candidates]

    lines = []
    if anchor is not None:
        lines.append(
            f"ESPOSIZIONE ALL'IMMUNOTERAPIA: {anchor.first_drug} dal "
            f"{anchor.first_raw} ({anchor.first_date.isoformat()}); ultima "
            f"dose {anchor.last_drug} il {anchor.last_raw} "
            f"({anchor.last_date.isoformat()})."
        )
    else:
        lines.append("ESPOSIZIONE ALL'IMMUNOTERAPIA: non determinata.")
    lines.append("")
    lines.append(
        f"CANDIDATI PRE-FILTRATI PER L'ORGANO «{organ}» (screening "
        "deterministico; ogni voce è citabile con il suo [#id]; offset in "
        "giorni dal primo checkpoint; le voci senza data non vanno scartate "
        "ma ancorate tramite il contesto):"
    )
    for candidate in kept:
        lines.append(format_candidate_line(candidate))
    if dropped:
        lines.append(
            f"(Nota: {dropped} candidati aggiuntivi oltre il tetto di "
            f"{max_candidates} sono stati omessi per vincolo di contesto; "
            "non altera la valutazione dei primi per ordine clinico.)"
        )
    lines.append("")
    lines.append(
        "Valuta SOLO i candidati elencati e identifica i potenziali irAE "
        "dell'organo. SINTETIZZA: raggruppa le voci ridondanti o che "
        "descrivono lo stesso evento clinico; NON produrre una voce per "
        "ogni candidato. Per ciascun irAE identifica tipo, grado CTCAE, "
        "data di prima insorgenza, probabilità di origine immuno-correlata, "
        "insorgenza nuova vs riacutizzazione di condizione preesistente, "
        "cause alternative e gli [#id] delle evidenze chiave. Indica per ogni "
        "irAE il campo obbligatorio diagnostic_support: "
        "conferma_clinica_scritta | trend_evidenze_multiple | "
        "risposta_a_steroidi_o_sospensione | valore_isolato | "
        "preesistente_baseline | non_specificato. Se l'organo "
        "non presenta veri irAE (es. reperti della malattia tumorale), "
        "restituisci una lista vuota o voci con probabilità IMPROBABILE "
        "motivando le cause alternative."
    )
    return "\n".join(lines), dropped


def run_organ_call(
    client: Any,
    organ: str,
    prompt: str,
    max_tokens: int,
) -> dict:
    """One structured LLM call for one organ; returns the parsed JSON dict."""
    try:
        data = client.generate_structured(
            prompt, SYSTEM_PROMPT, LAYER3_SCHEMA, max_tokens=max_tokens
        )
        if not isinstance(data, dict) or not isinstance(
            data.get("iraes"), list
        ):
            raise ValueError("schema violato: manca la lista 'iraes'")
        return {"organ": organ, "iraes": data["iraes"]}
    except Exception as exc:
        return {
            "organ": organ,
            "iraes": [],
            "error": f"{type(exc).__name__}: {exc}",
        }


def _evidence_ref(value) -> str:
    """Normalize a model-cited evidence reference (``#EVD_x`` → ``EVD_x``).

    The registry stores ids without the leading ``#`` that the model echoes
    (the prompt cites ``[#id]``); every lookup and display must strip it so
    the cited evidence resolves against ``clinical_evidence.evidence_id``.
    """
    return str(value or "").strip().lstrip("#")


def _finding_label(item: dict[str, Any]) -> str:
    """One-line summary of a per-organ (Layer 3) finding for Layer 4."""
    grade = item.get("ctcae_grade", "?")
    onset = item.get("first_onset_date", "?")
    prob = item.get("probability_immune", "?")
    ev = item.get("key_evidence_ids") or []
    label = (
        f"{item.get('irAE_type', '?')} · {grade} · insorgenza {onset} · {prob}"
    )
    if ev:
        label += (
            " · evidenze [#"
            + " #".join(_evidence_ref(e) for e in ev[:8])
            + "]"
        )
    return label


# Probability levels kept in the definitive list vs the "suspects" watchlist.
_FINAL_PROBS = {"CERTA_CONFERMATA", "PROBABILE"}
_SUSPECT_PROBS = {"POSSIBILE"}
_EXCLUDED_PROBS = {"IMPROBABILE", "INDETERMINATA"}


def _final_partition(
    findings: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split per-organ findings into the definitive list and the suspects.

    ``definitive`` = CERTA_CONFERMATA / PROBABILE (a missing or empty
    ``probability_immune`` is treated as definitive — conservative, so
    findings with an anomalous schema are never lost); ``suspects`` =
    POSSIBILE.  IMPROBABILE / INDETERMINATA are excluded from both.
    """
    definitive = [
        f for f in findings
        if (not f.get("probability_immune")
            or f.get("probability_immune") in _FINAL_PROBS)
    ]
    suspects = [
        f for f in findings
        if f.get("probability_immune") in _SUSPECT_PROBS
    ]
    return definitive, suspects


_CITED_ID_RE = re.compile(r"#?\bEVD_[0-9a-f]+", re.IGNORECASE)


def _backfill_cited_ids(
    items: list[dict[str, Any]],
    findings_by_organ: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Ensure every consolidated irAE carries ``key_evidence_ids``.

    The model occasionally omits the structured citations and writes them as
    ``#EVD_...`` tokens inside ``notes`` instead (see ``_evidence_ref``); when
    neither the structured field nor the notes carry ids, fall back to the
    union of the per-organ Layer 3 findings' citations for the entry's
    ``source_organs``.  Ids are normalized and deduplicated preserving order;
    ids the model did provide are never removed.
    """
    for item in items:
        ids = [
            _evidence_ref(e)
            for e in (item.get("key_evidence_ids") or [])
        ]
        if not ids:
            ids = [
                match.group(0).lstrip("#")
                for match in _CITED_ID_RE.finditer(str(item.get("notes") or ""))
            ]
        if not ids:
            for organ in set(str(o) for o in (item.get("source_organs") or [])):
                for finding in findings_by_organ.get(organ, []):
                    ids.extend(
                        _evidence_ref(e)
                        for e in (finding.get("key_evidence_ids") or [])
                    )
        ids = list(dict.fromkeys(i for i in ids if i))
        if ids:
            item["key_evidence_ids"] = ids
    return items


def build_consolidation_prompt(
    findings: list[dict[str, Any]],
    anchor: dict[str, Any] | None,
    *,
    max_findings: int = DEFAULT_MAX_CONSOLIDATION_FINDINGS,
) -> tuple[str, int]:
    """Layer 4 prompt: the consolidated, final list over the per-organ
    findings.  Findings are listed in three blocks — definitive, suspects,
    excluded — so the model merges the definitive ones and only promotes a
    suspect with confirmatory evidence.  Returns ``(prompt, truncated)`` where
    ``truncated`` counts the findings omitted over ``max_findings``."""
    truncated = 0
    if len(findings) > max_findings:
        truncated = len(findings) - max_findings
        findings = findings[:max_findings]

    definitive, suspects = _final_partition(findings)

    lines = []
    if anchor is not None:
        lines.append(
            f"ESPOSIZIONE ALL'IMMUNOTERAPIA: {anchor.get('first_drug')} dal "
            f"{anchor.get('first_raw')} ({anchor.get('first_date')}); ultima "
            f"dose {anchor.get('last_drug')} il {anchor.get('last_raw')} "
            f"({anchor.get('last_date')})."
        )
    else:
        lines.append("ESPOSIZIONE ALL'IMMUNOTERAPIA: non determinata.")
    lines.append("")
    lines.append(
        f"IRAE IDENTIFICATI PER SINGOLO ORGANO (FASE FINALE): "
        f"{len(definitive)} definitivi, {len(suspects)} sospetti, "
        f"{len(findings) - len(definitive) - len(suspects)} esclusi."
    )

    if definitive:
        lines.append("REPERTI DEFINITIVI (CERTA_CONFERMATA / PROBABILE):")
        for index, item in enumerate(definitive, start=1):
            organ = item.get("organ", "?")
            lines.append(
                f"[{index}] Organo: {organ} — {_finding_label(item)}"
            )
    if suspects:
        lines.append("SOSPETTI (POSSIBILE — da confermare):")
        for index, item in enumerate(suspects, start=1):
            organ = item.get("organ", "?")
            lines.append(
                f"[S{index}] Organo: {organ} — {_finding_label(item)}"
            )
    excluded = [
        f for f in findings
        if f.get("probability_immune") in _EXCLUDED_PROBS
    ]
    if excluded:
        lines.append("ESCLUSI (IMPROBABILE / INDETERMINATA):")
        for index, item in enumerate(excluded, start=1):
            organ = item.get("organ", "?")
            lines.append(
                f"[X{index}] Organo: {organ} — {_finding_label(item)}"
            )
    if truncated:
        lines.append(
            f"(Nota: {truncated} reperti aggiuntivi oltre il tetto di "
            f"{max_findings} sono stati omessi per vincolo di contesto.)"
        )
    lines.append("")
    lines.append(
        "FASE FINALE: produci la lista DEFINITIVA e approfondita degli irAE "
        "del paziente. La lista contiene SOLO irAE confermati "
        "(CERTA_CONFERMATA o PROBABILE). UNISCI i duplicati: due voci che "
        "descrivono lo stesso evento clinico anche in organi diversi (es. "
        "«Elevazione della troponina» in Miocardite e «Miocardite da ICI» in "
        "Cardiotossicità) devono diventare UNA sola voce, indicando in "
        "source_organs tutti gli organi in cui è stato rilevato. NON "
        "inventare irAE nuovi e NON riprodurre voci come doppioni. Un "
        "sospetto POSSIBILE è promuovibile solo con evidenza di conferma "
        "(seconda rilevazione o trend, risposta a steroidi o sospensione "
        "dell'immunoterapia); altrimenti OMETTILO dalla lista finale. I "
        "reperti esclusi (IMPROBABILE / INDETERMINATA) rappresentano "
        "tipicamente la malattia tumorale: non riportarli, motivando "
        "l'esclusione in alternative_causes. Approfondisci ogni voce residua "
        "con notes: caratterizzazione clinica, decorso, correlazione "
        "temporale con l'immunoterapia. Mantieni grado CTCAE, data di prima "
        "insorgenza, probabilità immuno-correlata, insorgenza nuova vs "
        "riacutizzazione e gli [#id] delle evidenze chiave, aggregando quelli "
        "delle voci unite."
    )
    return "\n".join(lines), truncated


def consolidate_iraes(
    client: Any,
    findings: list[dict[str, Any]],
    anchor: dict[str, Any] | None,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_findings: int = DEFAULT_MAX_CONSOLIDATION_FINDINGS,
) -> dict[str, Any]:
    """Layer 4: one structured call that merges cross-organ duplicates and
    deepens each final irAE's characterization.

    The returned ``iraes`` are only the definitive findings
    (CERTA_CONFERMATA / PROBABILE); ``suspects`` carries the POSSIBILE ones
    that lacked confirmation.  Always returns a dict ``{iraes, suspects,
    applied, error, input_count, output_count}``; on failure ``iraes`` falls
    back to the definitive partition of the input findings so the analysis is
    never lost.
    """
    if not findings:
        return {
            "iraes": [],
            "suspects": [],
            "applied": False,
            "error": None,
            "input_count": 0,
            "output_count": 0,
        }
    prompt, _truncated = build_consolidation_prompt(
        findings, anchor, max_findings=max_findings
    )
    try:
        data = client.generate_structured(
            prompt, SYSTEM_PROMPT_CONSOLIDATION, LAYER4_SCHEMA,
            max_tokens=max_tokens,
        )
        if not isinstance(data, dict) or not isinstance(
            data.get("iraes"), list
        ):
            raise ValueError("schema violato: manca la lista 'iraes'")
        consolidated = data["iraes"]
        definitive, suspects = _final_partition(consolidated)
        findings_by_organ: dict[str, list[dict[str, Any]]] = {}
        for finding in findings:
            findings_by_organ.setdefault(
                str(finding.get("organ") or ""), []
            ).append(finding)
        _backfill_cited_ids(definitive, findings_by_organ)
        _backfill_cited_ids(suspects, findings_by_organ)
        return {
            "iraes": definitive,
            "suspects": suspects,
            "applied": True,
            "error": None,
            "input_count": len(findings),
            "output_count": len(definitive),
        }
    except Exception as exc:
        definitive, suspects = _final_partition(findings)
        return {
            "iraes": definitive,
            "suspects": suspects,
            "applied": False,
            "error": f"{type(exc).__name__}: {exc}",
            "input_count": len(findings),
            "output_count": len(definitive),
        }


def analyze_irae(
    rows: Iterable[dict[str, Any]],
    client: Any,
    *,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    parallel: int = DEFAULT_PARALLEL,
) -> dict[str, Any]:
    """Full 3-layer irAE analysis for one patient.

    ``rows`` are the atomic-evidence dicts (see ``evidence_rows_from_models``
    or ``irae_headless.load_evidence`` with ``data_json`` already parsed).
    Layers 1-2 are deterministic; Layer 3 runs one structured call per organ
    in a bounded thread pool.  Returns the structured report dict (same shape
    as the headless tool's JSON), including ``timing_seconds``.
    """
    started = time.perf_counter()
    anchor = find_ici_anchor(rows)
    candidates = scan_for_irae(rows, anchor)

    baseline_excluded = 0
    if anchor is not None:
        baseline_excluded = sum(
            1
            for c in scan_for_irae(rows, anchor, drop_baseline=False)
            if c.band == "pre_ici"
        )

    report: dict[str, Any] = {
        "anchor": (
            {
                "first_drug": anchor.first_drug,
                "first_date": anchor.first_date.isoformat(),
                "first_raw": anchor.first_raw,
                "last_drug": anchor.last_drug,
                "last_date": anchor.last_date.isoformat(),
                "last_raw": anchor.last_raw,
                "occurrences": anchor.occurrences,
            }
            if anchor is not None
            else None
        ),
        "immunotherapy_start": (
            {
                "date": anchor.first_date.isoformat(),
                "raw": anchor.first_raw,
                "drug": anchor.first_drug,
            }
            if anchor is not None
            else None
        ),
        "candidates_total": len(candidates),
        "candidates_by_organ": summarize_candidates(candidates),
        "baseline_excluded": baseline_excluded,
        "timing_seconds": {},
    }

    by_organ: dict[str, list[ToxicityCandidate]] = {}
    for candidate in candidates:
        by_organ.setdefault(candidate.organ, []).append(candidate)
    organ_order = [s["organ"] for s in summarize_candidates(candidates)]

    prompts = [
        (organ, build_organ_prompt(organ, by_organ[organ], anchor, max_candidates)[0])
        for organ in organ_order
    ]

    organ_results: dict[str, dict] = {}
    if prompts:
        t_llm_start = time.perf_counter()
        max_workers = max(1, min(parallel, len(prompts)))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(run_organ_call, client, organ, prompt, max_tokens): organ
                for organ, prompt in prompts
            }
            for future in as_completed(futures):
                result = future.result()
                organ_results[result["organ"]] = result
        report["timing_seconds"]["llm_calls"] = round(
            time.perf_counter() - t_llm_start, 1
        )
    else:
        report["timing_seconds"]["llm_calls"] = 0.0

    findings = [
        item
        for result in organ_results.values()
        for item in result.get("iraes", [])
    ]

    anchor_dict = report["anchor"]
    t_consolidation_start = time.perf_counter()
    consolidation = consolidate_iraes(
        client, findings, anchor_dict, max_tokens=max_tokens
    )
    report["timing_seconds"]["layer4"] = round(
        time.perf_counter() - t_consolidation_start, 1
    )
    report["iraes"] = consolidation["iraes"]
    report["consolidation"] = consolidation
    report["organ_results"] = organ_results

    # Inspection payload: the Layer 2 candidates (bounded, in clinical order)
    # plus the compact provenance of every evidence the model cited, so the
    # inspector can show and open them regardless of whether they were
    # deterministic candidates.
    report["candidates"] = [asdict(c) for c in candidates[:max_candidates]]
    cited_ids: list[str] = []
    for item in findings:
        cited_ids.extend(_evidence_ref(eid) for eid in (item.get("key_evidence_ids") or []))
    for item in consolidation.get("iraes", []) + consolidation.get("suspects", []):
        cited_ids.extend(_evidence_ref(eid) for eid in (item.get("key_evidence_ids") or []))
    evidence_by_id = {str(row.get("evidence_id")): row for row in rows}
    report["evidence"] = _compact_evidence([
        evidence_by_id[eid] for eid in cited_ids if eid in evidence_by_id
    ])

    report["timing_seconds"]["total"] = round(time.perf_counter() - started, 1)
    return report


def _append_per_organ(lines: list[str], report: dict[str, Any]) -> None:
    """Per-organ (Layer 3) section: one heading per organ."""
    for organ, result in report.get("organ_results", {}).items():
        lines.append(f"## {organ}")
        if result.get("error"):
            lines.append(f"⚠️ Errore: {result['error']}")
            lines.append("")
            continue
        iraes = result.get("iraes", [])
        if not iraes:
            lines.append("_Nessun irAE probabile identificato._")
            lines.append("")
            continue
        for item in iraes:
            grade = item.get("ctcae_grade", "?")
            prob = item.get("probability_immune", "?")
            onset = item.get("first_onset_date", "?")
            lines.append(
                f"- **{item.get('irAE_type', '?')}** · {grade} · "
                f"insorgenza {onset} · {prob}"
            )
            if item.get("new_onset_vs_exacerbation"):
                lines.append(f"  - {item['new_onset_vs_exacerbation']}")
            if item.get("alternative_causes"):
                lines.append(f"  - Cause alt.: {item['alternative_causes']}")
            if item.get("diagnostic_support"):
                lines.append(f"  - Supporto: {item['diagnostic_support']}")
            ev = item.get("key_evidence_ids", [])
            if ev:
                lines.append(
                f"  - Evidenze: [#{' #'.join(_evidence_ref(e) for e in ev[:8])}]"
            )
        lines.append("")


def _append_consolidated(lines: list[str], report: dict[str, Any]) -> None:
    """Layer 4 section: the final, duplicate-free irAE list.  Shown only when
    the consolidation call actually applied (per-organ findings otherwise)."""
    consolidation = report.get("consolidation")
    if not isinstance(consolidation, dict) or not consolidation.get("applied"):
        return
    consolidated = consolidation.get("iraes") or []
    lines.append("## 🔬 Analisi finale consolidata (Layer 4)")
    if consolidation.get("error"):
        lines.append(f"⚠️ Errore: {consolidation['error']}")
        lines.append("")
        return
    if not consolidated:
        lines.append("_Nessun irAE confermato dopo la verifica finale._")
        lines.append("")
        return
    if consolidation.get("input_count") != consolidation.get("output_count"):
        lines.append(
            f"_Uniti {consolidation['input_count']} reperti per organo in "
            f"{consolidation['output_count']} irAE definitivi._"
        )
        lines.append("")
    for item in consolidated:
        grade = item.get("ctcae_grade", "?")
        prob = item.get("probability_immune", "?")
        onset = item.get("first_onset_date", "?")
        organs = item.get("source_organs") or []
        organ_txt = ", ".join(organs) if organs else item.get("organ", "?")
        lines.append(
            f"- **{item.get('irAE_type', '?')}** · {grade} · "
            f"insorgenza {onset} · {prob} · organi: {organ_txt}"
        )
        if item.get("new_onset_vs_exacerbation"):
            lines.append(f"  - {item['new_onset_vs_exacerbation']}")
        if item.get("alternative_causes"):
            lines.append(f"  - Cause alt.: {item['alternative_causes']}")
        if item.get("notes"):
            lines.append(f"  - Approfondimento: {item['notes']}")
        ev = item.get("key_evidence_ids", [])
        if ev:
            lines.append(
                f"  - Evidenze: [#{' #'.join(_evidence_ref(e) for e in ev[:8])}]"
            )

    suspects = consolidation.get("suspects") or []
    if suspects:
        lines.append("### 🔎 Sospetti da monitorare (non confermati)")
        for item in suspects:
            grade = item.get("ctcae_grade", "?")
            prob = item.get("probability_immune", "?")
            onset = item.get("first_onset_date", "?")
            organs = item.get("source_organs") or []
            organ_txt = ", ".join(organs) if organs else item.get("organ", "?")
            lines.append(
                f"- **{item.get('irAE_type', '?')}** · {grade} · "
                f"insorgenza {onset} · {prob} · organi: {organ_txt}"
            )
            if item.get("alternative_causes"):
                lines.append(f"  - Cause alt.: {item['alternative_causes']}")
            ev = item.get("key_evidence_ids", [])
            if ev:
                lines.append(
                f"  - Evidenze: [#{' #'.join(_evidence_ref(e) for e in ev[:8])}]"
            )
    lines.append("")


def render_irae_markdown(report: dict[str, Any]) -> str:
    """Render the structured report as the Markdown the queue dialog shows."""
    lines = ["# Analisi irAE strutturata — NCTCAE 6.0", ""]
    anchor = report.get("anchor")
    if anchor is not None:
        lines.append(
            f"**Inizio immunoterapia**: {anchor['first_drug']} dal "
            f"{anchor['first_raw']} ({anchor['first_date']})"
        )
        lines.append(
            f"**Ultima dose**: {anchor['last_drug']} il {anchor['last_raw']} "
            f"({anchor['last_date']})"
        )
    else:
        lines.append("**Inizio immunoterapia**: non determinato")
    lines.append(f"**Candidati pre-filtrati (Layer 2)**: {report.get('candidates_total', 0)}")
    lines.append("")

    _append_per_organ(lines, report)
    _append_consolidated(lines, report)
    return "\n".join(lines).rstrip() + "\n"


def parallel_instance_count(client, *, override=0) -> int:
    """How many llama-server instances the irAE queue may run in parallel.

    ``override > 0`` pins the count (``0``/default = auto from the memory
    budget).  Multi-instance is only possible when the client can clone
    itself (:meth:`~emr_analyzer.extraction.llm_client.LlmClient.for_instance`)
    on the ``llama_cpp`` backend; vLLM has no instance dimension and fakes
    used by tests return 1.  The model's on-disk size and the *real*
    (memory-capped) worker count are read through the backend so the budget
    reflects the actual per-server KV cache.
    """
    if not hasattr(client, "for_instance"):
        return 1
    if getattr(client, "backend_type", "llama_cpp") != "llama_cpp":
        return 1
    if override > 0:
        return max(1, int(override))
    try:
        size = int(
            (client.backend.model_info(client.model) or {}).get("size_bytes") or 0
        )
        if size <= 0:
            # Unknown model: a size of 0 would over-allocate instances (the
            # budget would omit the weights); stay conservative.
            return 1
        workers = client.backend.key_for(client).np
    except Exception:
        return 1
    from ..utils import hardware

    return max(
        1,
        hardware.max_llm_instances(
            size, client.context_length, workers
        ),
    )
