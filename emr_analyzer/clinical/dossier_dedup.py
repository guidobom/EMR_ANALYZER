"""Conservative grouping of repeated observations from a single dossier.

No semantic similarity threshold can prove event identity. Merge only the
same source occurrence, copies of the same report, or identical assertions
and quotes with a verified explicit event date. Keep every original finding.
"""
from __future__ import annotations

import hashlib
import json
import re

from .temporal import normalize_clinical_date

VERSION = "dossier-dedup-v1"
_RELATIVE = re.compile(
    r"\b(?:oggi|ieri|domani|attual\w*|ora|nuov\w*|ancora|persiste\w*|"
    r"ricompar\w*|recidiv\w*|ripres\w*|successiv\w*|precedent\w*)\b", re.I,
)


def normalize_text(text):
    # Do not remove negations, punctuation, numbers, units or accents.
    return " ".join(str(text or "").split())


def verified_event_date(item):
    value, anchor = item.get("event_date"), item.get("event_date_quote")
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return ""
    if not isinstance(anchor, str) or not normalize_text(anchor):
        return ""
    if normalize_text(anchor) not in normalize_text(item.get("quote")):
        return ""
    parsed = normalize_clinical_date(anchor)
    if parsed.precision != "day" or parsed.start != value or parsed.approximate:
        return ""
    return value


def group_findings(patient_id, findings, sources):
    """Return auditable groups and statistics; never mutate source findings."""
    groups, by_key = [], {}
    for index, finding in enumerate(findings):
        source = sources.get(finding["source_id"], {})
        quote = normalize_text(finding["quote"])
        statement = normalize_text(finding["statement"])
        event_date = verified_event_date(finding)
        keys = []
        if finding.get("quote_start") is not None and finding.get("quote_end") is not None:
            keys.append(("same_source_span", source.get("document_id"),
                         source.get("page_segment"), source.get("page"),
                         finding["quote_start"], finding["quote_end"], quote, statement))
        if source.get("document_hash"):
            keys.append(("identical_report", source["document_hash"],
                         source.get("document_date"), source.get("page_segment"),
                         source.get("page"), quote, statement))
        # Unknown dates never inherit the report date. Ambiguous relative or
        # recurrence wording remains separate, even if the text is identical.
        if (event_date and not finding.get("relative_context")
                and not _RELATIVE.search(quote + " " + statement)):
            keys.append(("dated_exact_copy", event_date, quote, statement))
        match = next(((by_key[key], key[0]) for key in keys if key in by_key), None)
        if match:
            group, reason = match
            if reason not in group["merge_reasons"]:
                group["merge_reasons"].append(reason)
        else:
            group = {
                "group_id": "OBS_" + hashlib.sha256(json.dumps(
                    [patient_id, finding["source_id"], index], ensure_ascii=False,
                ).encode()).hexdigest()[:20],
                "statement": finding["statement"], "quote": finding["quote"],
                "event_date": event_date, "source_ids": [],
                "finding_indices": [], "occurrences": [], "merge_reasons": [],
            }
            groups.append(group)
        for key in keys:
            by_key.setdefault(key, group)
        group["finding_indices"].append(index)
        if finding["source_id"] not in group["source_ids"]:
            group["source_ids"].append(finding["source_id"])
        group["occurrences"].append({
            "finding_index": index, "source_id": finding["source_id"],
            "document_id": source.get("document_id"),
            "document_date": source.get("document_date"), "page": source.get("page"),
            "quote_start": finding.get("quote_start"), "quote_end": finding.get("quote_end"),
        })
    # Identical passages that could represent different visits/interpretations
    # are flagged, not silently turned into a single clinical event.
    passages = {}
    for group in groups:
        passages.setdefault(normalize_text(group["quote"]), []).append(group["group_id"])
    candidates = [ids for ids in passages.values() if len(ids) > 1]
    return groups, {
        "version": VERSION, "raw_findings": len(findings),
        "grouped_findings": len(groups), "duplicates_merged": len(findings) - len(groups),
        "unmerged_repeated_passages": candidates,
        "note": "Gruppi di osservazioni, non conteggio validato di eventi clinici indipendenti.",
    }


def format_group(group):
    citations = " ".join(f"[{source}]" for source in group["source_ids"])
    dates = sorted({str(item["document_date"]) for item in group["occurrences"]
                    if item.get("document_date")})
    return (
        f"{citations} {group['statement']}\nCitazione: {group['quote']}\n"
        f"Data evento esplicita: {group['event_date'] or 'non determinata'}; "
        f"date dei referti: {', '.join(dates) or 'non disponibili'}; "
        f"occorrenze testuali: {len(group['occurrences'])}. "
        "Le ripetizioni non sono conferme indipendenti."
    )
