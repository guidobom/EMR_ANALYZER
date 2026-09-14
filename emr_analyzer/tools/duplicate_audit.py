"""Misura la duplicazione delle evidenze atomiche per causa (read-only).

Usage::

    python -m emr_analyzer.tools.duplicate_audit --project /path/to/MELANOMA

Classifica i potenziali accorpamenti per causa:
  - ``exact``: chiave di identità identica (il dedup attuale li fonde);
  - ``typo``: entità quasi identiche (refusi: ipililumab/ipilimumab);
  - ``subset``: una entità è sottoinsieme di token dell'altra;
  - ``attribute``: stessa entità, attributi (severità/sede) che
    normalizzati coincidono;
  - ``english``: entità in inglese (candidata a traduzione/normalizzazione).

Scrive un report Markdown nel progetto. Non modifica il database.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

from ..clinical.concept_canonicalization import (
    canonical_concept,
    canonical_severity,
)

_ID_COLS = (
    "patient_id", "category", "normalized_entity", "observed_date",
    "observed_date_end", "assertion", "value_text", "numeric_value", "unit",
    "anatomical_site", "laterality", "severity", "clinical_status",
)

_FILLER = {"none", "n.d.", "n.d", "nd", "n/a", "na", "non specificato",
           "non specificata", "non disponibile", "unknown", "sconosciuto",
           "sconosciuta", "-", "0"}

_EN_TOKENS = {
    "of", "upper", "limb", "including", "malignant", "left", "right",
    "neck", "head", "shoulder", "trunk", "back", "skin", "lymph",
    "node", "nodes", "metastatic", "melanoma", "cancer", "cell", "cells",
    "clear", "basal", "squamous", "disease", "secondary", "tumor",
    "tumors", "tumour", "axillary", "sentinel", "biopsy", "excision",
}


def _norm(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or "").casefold())
    without_marks = "".join(
        char for char in text if not unicodedata.combining(char)
    )
    return " ".join(
        "".join(char if char.isalnum() else " " for char in without_marks)
        .split()
    )


def _severity_key(value: object) -> str:
    text = _norm(value)
    text = re.sub(r"\b(?:stadio|stage|st)\b", " ", text)
    return " ".join(text.split())


def _site_key(value: object) -> str:
    text = _norm(value)
    if text in _FILLER:
        return ""
    return text


def _exact_key(row: dict, canonical: bool = False) -> tuple:
    therapy = {}
    oncology = {}
    try:
        data = json.loads(row.get("data_json") or "{}")
        therapy = data.get("therapy") or {}
        oncology = data.get("oncology") or {}
    except (TypeError, ValueError):
        pass
    entity = _norm(row["normalized_entity"])
    severity = _severity_key(row["severity"])
    if canonical:
        entity = canonical_concept(row["category"], entity)
        severity = canonical_severity(row["severity"])
    return (
        row["patient_id"],
        _norm(row["category"]),
        entity,
        row["observed_date"] or "",
        row["observed_date_end"] or "",
        _norm(row["assertion"]),
        _norm(row["value_text"]),
        row["numeric_value"],
        _norm(row["unit"]),
        _site_key(row["anatomical_site"]),
        _norm(row["laterality"]),
        severity,
        _norm(row["clinical_status"]),
        _norm(therapy.get("lifecycle_status")),
        _norm(therapy.get("dose")),
        _norm(therapy.get("route")),
        _norm(therapy.get("frequency")),
        _norm(oncology.get("line_label") or oncology.get("line")),
        _norm(json.dumps(oncology.get("regimen") or [], sort_keys=True)),
        _norm(oncology.get("cycle")),
        _norm(oncology.get("modification")),
    )


def _is_english(entity: str) -> bool:
    tokens = [_norm(token) for token in re.split(r"[^a-z0-9]+", _norm(entity)) if token]
    if not tokens:
        return False
    english = sum(1 for token in tokens if token in _EN_TOKENS)
    return english >= len(tokens) * 0.6 and english >= 2


def _load_rows(db_path: Path) -> dict[str, list[dict]]:
    source = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    source.row_factory = sqlite3.Row
    try:
        rows = source.execute(
            "SELECT evidence_id, patient_id, document_id, document_date, "
            "category, normalized_entity, assertion, observed_date, "
            "observed_date_end, value_text, numeric_value, unit, "
            "anatomical_site, laterality, severity, clinical_status, "
            "source_text, data_json FROM clinical_evidence"
        ).fetchall()
    finally:
        source.close()
    by_patient: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_patient[row["patient_id"]].append(dict(row))
    return by_patient


def _audit_patient(rows: list[dict]) -> dict:
    findings: dict[str, list[dict]] = defaultdict(list)
    used: set[str] = set()

    # --- unique-atom counts under both keys -------------------------------
    legacy_groups: dict[tuple, list[dict]] = defaultdict(list)
    canonical_groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        legacy_groups[_exact_key(row, canonical=False)].append(row)
        canonical_groups[_exact_key(row, canonical=True)].append(row)
    findings["legacy_unique"] = len(legacy_groups)
    findings["canonical_unique"] = len(canonical_groups)

    # --- exact: rows that the current dedup key already merges -----------
    groups = canonical_groups
    for key, members in groups.items():
        docs = {row["document_id"] for row in members}
        if len(members) > 1 and len(docs) > 1:
            canonical = min(members, key=lambda r: (r["document_date"] or "9999", r["evidence_id"]))
            extra = [r for r in members if r["evidence_id"] != canonical["evidence_id"]]
            findings["exact"].append({
                "entity": canonical["normalized_entity"],
                "category": canonical["category"],
                "occurrences": len(members),
                "suppressible": len(extra),
                "documents": len(docs),
                "sample_dates": sorted({r["document_date"] or "?" for r in members})[:4],
                "evidence_ids": [r["evidence_id"] for r in extra][:6],
            })
            used.update(r["evidence_id"] for r in extra)

    # --- typo / subset / english: per (patient, category) entity pairs ---
    by_category: dict[str, dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        by_category[row["category"]][_norm(row["normalized_entity"])].append(row)

    for category, entity_map in by_category.items():
        entities = sorted(entity_map)
        by_first: dict[str, list[str]] = defaultdict(list)
        for entity in entities:
            by_first[entity[:1]].append(entity)
        seen_pairs: set[tuple[str, str]] = set()
        for first_char, bucket in by_first.items():
            for i, left in enumerate(bucket):
                left_tokens = set(left.split())
                for right in bucket[i + 1:]:
                    pair = (left, right)
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    left_rows = entity_map[left]
                    right_rows = entity_map[right]
                    right_tokens = set(right.split())
                    ratio = SequenceMatcher(
                        None, left, right, autojunk=False
                    ).ratio()
                    if 0.86 <= ratio < 1.0 and min(len(left), len(right)) >= 4:
                        extra = [
                            r for r in right_rows
                            if r["evidence_id"] not in used
                        ]
                        findings["typo"].append({
                            "entity": f"{left} ≈ {right}",
                            "category": category,
                            "occurrences": len(left_rows) + len(right_rows),
                            "suppressible": len(extra),
                            "documents": len({r["document_id"] for r in left_rows + right_rows}),
                            "sample_dates": sorted({r["document_date"] or "?" for r in left_rows + right_rows})[:4],
                            "evidence_ids": [r["evidence_id"] for r in extra][:6],
                        })
                        used.update(r["evidence_id"] for r in extra)
                        continue
                    subset = (
                        left_tokens < right_tokens
                        or right_tokens < left_tokens
                    )
                    same_head = (
                        left.split()[:1] == right.split()[:1]
                        if left.split() and right.split() else False
                    )
                    if (
                        subset and same_head
                        and min(len(left_tokens), len(right_tokens)) >= 2
                        and len(max(left_tokens, right_tokens, key=len))
                        <= len(min(left_tokens, right_tokens, key=len)) + 4
                    ):
                        extra = [
                            r for r in right_rows
                            if r["evidence_id"] not in used
                        ]
                        findings["subset"].append({
                            "entity": f"{left} ⊂ {right}",
                            "category": category,
                            "occurrences": len(left_rows) + len(right_rows),
                            "suppressible": len(extra),
                            "documents": len({r["document_id"] for r in left_rows + right_rows}),
                            "sample_dates": sorted({r["document_date"] or "?" for r in left_rows + right_rows})[:4],
                            "evidence_ids": [r["evidence_id"] for r in extra][:6],
                        })
                        used.update(r["evidence_id"] for r in extra)
        for entity, entity_rows in entity_map.items():
            if _is_english(entity) and len(entity_rows) >= 2:
                extra = [r for r in entity_rows if r["evidence_id"] not in used]
                findings["english"].append({
                    "entity": entity,
                    "category": category,
                    "occurrences": len(entity_rows),
                    "suppressible": len(extra),
                    "documents": len({r["document_id"] for r in entity_rows}),
                    "sample_dates": sorted({r["document_date"] or "?" for r in entity_rows})[:4],
                    "evidence_ids": [r["evidence_id"] for r in extra][:6],
                })
                used.update(r["evidence_id"] for r in extra)

    # --- attribute: same entity, severity/site raw differ but normalize
    #     equal (or one side missing) -------------------------------------
    for category, entity_map in by_category.items():
        for entity, entity_rows in entity_map.items():
            severity_variants = defaultdict(list)
            for row in entity_rows:
                severity_variants[_severity_key(row["severity"])].append(row)
            if len(severity_variants) > 1:
                extra = [
                    r for members in severity_variants.values() for r in members
                    if r["evidence_id"] not in used
                ][1:]
                findings["attribute"].append({
                    "entity": f"{entity} [severità: "
                              + " | ".join(sorted(severity_variants)) + "]",
                    "category": category,
                    "occurrences": len(entity_rows),
                    "suppressible": len(extra),
                    "documents": len({r["document_id"] for r in entity_rows}),
                    "sample_dates": sorted({r["document_date"] or "?" for r in entity_rows})[:4],
                    "evidence_ids": [r["evidence_id"] for r in extra][:6],
                })
                used.update(r["evidence_id"] for r in extra)
            site_variants = defaultdict(list)
            for row in entity_rows:
                site_variants[_site_key(row["anatomical_site"])].append(row)
            if len(site_variants) > 1 and "" in site_variants:
                extra = [
                    r for members in site_variants.values() for r in members
                    if r["evidence_id"] not in used
                ][1:]
                findings["attribute"].append({
                    "entity": f"{entity} [sede: mancante vs presente]",
                    "category": category,
                    "occurrences": len(entity_rows),
                    "suppressible": len(extra),
                    "documents": len({r["document_id"] for r in entity_rows}),
                    "sample_dates": sorted({r["document_date"] or "?" for r in entity_rows})[:4],
                    "evidence_ids": [r["evidence_id"] for r in extra][:6],
                })
                used.update(r["evidence_id"] for r in extra)
    return dict(findings)


def _render_report(
    project: Path, per_patient: dict[str, dict], total_atoms: int
) -> Path:
    lines: list[str] = []
    lines.append("# Audit duplicazioni evidenze atomiche")
    lines.append("")
    lines.append(f"Progetto: `{project.name}` — generato "
                 f"{datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append(f"Atomi totali: **{total_atoms}** — "
                 f"pazienti con evidenze: **{len(per_patient)}**")
    lines.append("")
    if "legacy_unique" in per_patient.get(next(iter(per_patient)), {}):
        legacy_unique = sum(
            patient["legacy_unique"] for patient in per_patient.values()
        )
        canonical_unique = sum(
            patient["canonical_unique"] for patient in per_patient.values()
        )
        lines.append("| Vista | Atomi unici dopo dedup |")
        lines.append("|---|---:|")
        lines.append(f"| chiave precedente | {legacy_unique} |")
        lines.append(f"| chiave canonica | **{canonical_unique}** "
                     f"({canonical_unique - legacy_unique:+d}) |")
        lines.append("")
    totals: Counter = Counter()
    for findings in per_patient.values():
        for cause, items in findings.items():
            if isinstance(items, list):
                totals[cause] += len(items)
    suppressed: Counter = Counter()
    for findings in per_patient.values():
        for cause, items in findings.items():
            if isinstance(items, list):
                suppressed[cause] += sum(item["suppressible"] for item in items)
    lines.append("| Causa | Gruppi | Atomi sopprimibili |")
    lines.append("|---|---:|---:|")
    for cause in ("exact", "typo", "subset", "attribute", "english"):
        lines.append(f"| {cause} | {totals[cause]} | {suppressed[cause]} |")
    total_suppressible = sum(suppressed.values())
    lines.append(f"| **Totale** | {sum(totals.values())} | "
                 f"**{total_suppressible}** ({100.0 * total_suppressible / max(total_atoms, 1):.1f}%) |")
    lines.append("")
    lines.append("## Esempi per causa")
    lines.append("")
    for cause, label in (
        ("exact", "Ripetizioni a chiave esatta (già sopprimibili dal dedup attuale)"),
        ("typo", "Varianti per refuso (fuzzy)"),
        ("subset", "Varianti per sottoinsieme di token"),
        ("attribute", "Stessa entità, attributi (severità/sede) che normalizzati coincidono"),
        ("english", "Entità in inglese"),
    ):
        lines.append(f"### {label}")
        examples: Counter = Counter()
        for patient_id, findings in per_patient.items():
            for item in findings.get(cause, []):
                examples[(item["entity"], item["category"])] += item["suppressible"]
        if not examples:
            lines.append("_nessun candidato_")
            lines.append("")
            continue
        for (entity, category), count in examples.most_common(15):
            lines.append(f"- `{entity}` ({category}) — {count} atomi")
        lines.append("")
    report = project / f"duplicate_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Misura le duplicazioni delle evidenze atomiche."
    )
    parser.add_argument("--project", required=True, type=Path,
                        help="Directory del progetto (contiene emr_registry.db)")
    args = parser.parse_args(argv)

    db_path = args.project / "emr_registry.db"
    if not db_path.is_file():
        parser.error(f"database non trovato: {db_path}")
    by_patient = _load_rows(db_path)
    total_atoms = sum(len(rows) for rows in by_patient.values())
    print(f"Pazienti: {len(by_patient)} — atomi: {total_atoms}")
    per_patient = {}
    for index, (patient_id, rows) in enumerate(
        sorted(by_patient.items()), start=1
    ):
        per_patient[patient_id] = _audit_patient(rows)
        if index % 10 == 0:
            print(f"  {index}/{len(by_patient)} pazienti analizzati")
    report = _render_report(args.project, per_patient, total_atoms)
    print(f"Report: {report.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
