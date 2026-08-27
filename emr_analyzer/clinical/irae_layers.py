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

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Iterable

from .irae_prototype import (
    IciAnchor,
    ToxicityCandidate,
    find_ici_anchor,
    scan_for_irae,
    summarize_candidates,
)

DEFAULT_MAX_CANDIDATES = 60
DEFAULT_MAX_TOKENS = 4096
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
    "melanoma, captazioni surrenaliche da metastasi) da una vera "
    "tossicità immuno-correlata. Privilegia la sensibilità, ma non "
    "considerare automaticamente immuno-correlato qualsiasi evento "
    "verificatosi durante immunoterapia."
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
                },
                "required": [
                    "organ", "irAE_type", "ctcae_grade", "first_onset_date",
                    "probability_immune",
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
                    "probability_immune",
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
        rows.append({
            "evidence_id": _get(item, "evidence_id", ""),
            "category": _get(item, "category", ""),
            "normalized_entity": _get(item, "normalized_entity", ""),
            "observed_date": _get(item, "observed_date", ""),
            "data_json": data,
            "value_text": _get(item, "value_text", ""),
            "numeric_value": _get(item, "numeric_value", None),
            "unit": _get(item, "unit", ""),
        })
    return rows


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
        "cause alternative e gli [#id] delle evidenze chiave. Se l'organo "
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
        label += f" · evidenze [#{' #'.join(ev[:8])}]"
    return label


def build_consolidation_prompt(
    findings: list[dict[str, Any]],
    anchor: dict[str, Any] | None,
    *,
    max_findings: int = DEFAULT_MAX_CONSOLIDATION_FINDINGS,
) -> tuple[str, int]:
    """Layer 4 prompt: the consolidated, final list over the per-organ
    findings.  Returns ``(prompt, truncated)`` where ``truncated`` counts the
    findings omitted over ``max_findings``."""
    truncated = 0
    kept = findings
    if len(kept) > max_findings:
        truncated = len(kept) - max_findings
        kept = kept[:max_findings]

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
        f"{len(kept)} reperto/i da consolidare."
    )
    for index, item in enumerate(kept, start=1):
        organ = item.get("organ", "?")
        lines.append(
            f"[{index}] Organo: {organ} — {_finding_label(item)}"
        )
    if truncated:
        lines.append(
            f"(Nota: {truncated} reperti aggiuntivi oltre il tetto di "
            f"{max_findings} sono stati omessi per vincolo di contesto.)"
        )
    lines.append("")
    lines.append(
        "FASE FINALE: produci la lista DEFINITIVA e approfondita degli irAE "
        "del paziente. UNISCI i duplicati: due voci che descrivono lo stesso "
        "evento clinico anche in organi diversi (es. «Elevazione della "
        "troponina» in Miocardite e «Miocardite da ICI» in "
        "Cardiotossicità) devono diventare UNA sola voce, indicando in "
        "source_organs tutti gli organi in cui è stato rilevato. NON "
        "inventare irAE nuovi e NON riprodurre voci come doppioni. "
        "Escludi dalla lista definitiva gli irAE con probabilità IMPROBABILE "
        "che rappresentano reperti della malattia tumorale (es. lesioni "
        "cutanee del melanoma), motivandone l'esclusione in alternative_causes. "
        "Approfondisci ogni voce residua con notes: caratterizzazione clinica, "
        "decorso, correlazione temporale con l'immunoterapia. Mantieni grado "
        "CTCAE, data di prima insorgenza, probabilità immuno-correlata, "
        "insorgenza nuova vs riacutizzazione e gli [#id] delle evidenze "
        "chiave, aggregando quelli delle voci unite."
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

    Always returns a dict ``{iraes, applied, error, input_count,
    output_count}``; on failure ``iraes`` falls back to the input findings so
    the analysis is never lost.
    """
    if not findings:
        return {
            "iraes": [],
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
        return {
            "iraes": consolidated,
            "applied": True,
            "error": None,
            "input_count": len(findings),
            "output_count": len(consolidated),
        }
    except Exception as exc:
        return {
            "iraes": findings,
            "applied": False,
            "error": f"{type(exc).__name__}: {exc}",
            "input_count": len(findings),
            "output_count": len(findings),
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
            ev = item.get("key_evidence_ids", [])
            if ev:
                lines.append(f"  - Evidenze: [#{' #'.join(ev[:8])}]")
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
            lines.append(f"  - Evidenze: [#{' #'.join(ev[:8])}]")
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
