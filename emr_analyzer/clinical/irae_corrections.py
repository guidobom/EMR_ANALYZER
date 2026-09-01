"""Persistent manual corrections for the structured irAE report.

Each correction is a small intent record stored per patient in
``<workspace>/<patient_id>/irae_corrections.json``.  Corrections are applied
ONCE over a fresh, uncorrected report (``apply_irae_corrections``), so they
are idempotent across re-analyses: the same raw report plus the same file
always yields the same corrected report.  The raw report is never mutated —
``apply_irae_corrections`` works on annotated copies.

Correction actions:

* ``remove`` — drop a finding (definitive or suspect); sticky, restoring it
  means adding it back manually.
* ``edit`` — merge whitelisted fields and re-partition the finding when the
  edited ``probability_immune`` crosses the definitive/suspect boundary.
* ``add`` — introduce a manually identified irAE, placed in the definitive or
  suspect list according to its probability (duplicates are refused).

A correction targets a finding by ``finding_id`` first (stable, assigned by
``annotate_report``), then by ``(irAE_type, first_onset_date)``, then by
``irAE_type`` alone (first match, logged as ambiguous when more than one
finding matches).
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ..config import active_workspace

CORRECTIONS_FILENAME = "irae_corrections.json"

_VALID_ACTIONS = frozenset({"remove", "edit", "add"})

# Probability levels kept in the definitive list vs the suspects watchlist
# (mirrors ``irae_layers._final_partition``).
_FINAL_PROBS = frozenset({"CERTA_CONFERMATA", "PROBABILE"})
_SUSPECT_PROBS = frozenset({"POSSIBILE"})
_EXCLUDED_PROBS = frozenset({"IMPROBABILE", "INDETERMINATA"})
_VALID_PROBS = _FINAL_PROBS | _SUSPECT_PROBS | _EXCLUDED_PROBS

# Fields a manual edit may change; everything else is protected.
EDITABLE_FIELDS = frozenset({
    "irAE_type",
    "ctcae_grade",
    "first_onset_date",
    "probability_immune",
    "new_onset_vs_exacerbation",
    "alternative_causes",
    "notes",
    "confidence",
})


@dataclass
class IraeCorrection:
    """One manual intent over a single irAE finding."""

    action: str  # "remove" | "edit" | "add"
    irAE_type: str
    first_onset_date: str = ""
    finding_id: str = ""
    fields: dict = field(default_factory=dict)
    reason: str = ""
    applied_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "IraeCorrection | None":
        action = str(data.get("action") or "")
        if action not in _VALID_ACTIONS:
            return None
        irAE_type = str(data.get("irAE_type") or "")
        if not irAE_type:
            return None
        fields = data.get("fields")
        if not isinstance(fields, dict):
            fields = {}
        return cls(
            action=action,
            irAE_type=irAE_type,
            first_onset_date=str(data.get("first_onset_date") or ""),
            finding_id=str(data.get("finding_id") or ""),
            fields=dict(fields),
            reason=str(data.get("reason") or ""),
            applied_at=str(data.get("applied_at") or ""),
        )


def corrections_path(patient_id: str) -> Path:
    """Per-patient correction file inside the active workspace."""
    return active_workspace.path / patient_id / CORRECTIONS_FILENAME


def load_corrections(patient_id: str) -> list[IraeCorrection]:
    """Corrections persisted for *patient_id*; ``[]`` when missing/corrupt."""
    path = corrections_path(patient_id)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    corrections: list[IraeCorrection] = []
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            correction = IraeCorrection.from_dict(item)
            if correction is not None:
                corrections.append(correction)
    return corrections


def save_corrections(patient_id: str, corrections: list[IraeCorrection]) -> None:
    """Atomically persist *corrections* for *patient_id* (tmp + replace)."""
    path = corrections_path(patient_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [correction.to_dict() for correction in corrections]
    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent), prefix=".irae_corr_", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def add_correction(patient_id: str, correction: IraeCorrection) -> list[IraeCorrection]:
    """Append *correction* to the patient's file and persist it."""
    if not correction.applied_at:
        correction.applied_at = datetime.now().isoformat(timespec="seconds")
    corrections = load_corrections(patient_id)
    corrections.append(correction)
    save_corrections(patient_id, corrections)
    return corrections


def annotate_report(report: dict[str, Any]) -> dict[str, Any]:
    """Return an editable copy of *report* with a stable ``finding_id`` on
    every definitive irAE and suspect (``irAE-1``, ``irAE-2`` ...).

    Only the copied report is annotated: the input report (and its nested
    lists/dicts) is never mutated.  The definitive list stays the same object
    under both ``report["iraes"]`` and ``report["consolidation"]["iraes"]``
    (as the pipeline builds it), so edits keep both views consistent.
    """
    corrected = dict(report)
    consolidation = corrected.get("consolidation")
    if isinstance(consolidation, dict):
        consolidation = dict(consolidation)
        corrected["consolidation"] = consolidation
    else:
        consolidation = {}

    organ_results = corrected.get("organ_results")
    if isinstance(organ_results, dict):
        corrected["organ_results"] = {
            organ: dict(result)
            for organ, result in organ_results.items()
        }

    iraes = list(corrected.get("iraes") or [])
    suspects = list(consolidation.get("suspects") or [])
    if not iraes and not suspects:
        return corrected

    iraes = [dict(item) for item in iraes]
    suspects = [dict(item) for item in suspects]
    corrected["iraes"] = iraes
    consolidation["iraes"] = iraes
    consolidation["suspects"] = suspects
    for index, item in enumerate(iraes + suspects, start=1):
        item.setdefault("finding_id", f"irAE-{index}")
    return corrected


def _partition_for(probability: Any) -> str | None:
    """Target list for a probability: 'iraes', 'suspects' or None (excluded)."""
    prob = str(probability or "") if probability else ""
    if not prob or prob in _FINAL_PROBS:
        return "iraes"
    if prob in _SUSPECT_PROBS:
        return "suspects"
    return None


def _same_finding(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Same clinical finding when type and first-onset date coincide."""
    return (
        str(a.get("irAE_type") or "") == str(b.get("irAE_type") or "")
        and str(a.get("first_onset_date") or "")
        == str(b.get("first_onset_date") or "")
    )


def _match_in(
    findings: list[dict[str, Any]], correction: IraeCorrection
) -> tuple[int | None, bool]:
    """Index of the first finding matched by *correction* and whether the
    match was ambiguous (type-only fallback over multiple candidates)."""
    if correction.finding_id:
        for index, item in enumerate(findings):
            if item.get("finding_id") == correction.finding_id:
                return index, False
    if correction.first_onset_date:
        by_type_date = [
            (index, item)
            for index, item in enumerate(findings)
            if str(item.get("irAE_type") or "") == correction.irAE_type
            and str(item.get("first_onset_date") or "")
            == correction.first_onset_date
        ]
        if by_type_date:
            return by_type_date[0][0], False
    by_type = [
        (index, item)
        for index, item in enumerate(findings)
        if str(item.get("irAE_type") or "") == correction.irAE_type
    ]
    if by_type:
        return by_type[0][0], len(by_type) > 1
    return None, False


def _locate(
    iraes: list[dict[str, Any]],
    suspects: list[dict[str, Any]],
    correction: IraeCorrection,
) -> tuple[str | None, int | None, bool]:
    """Find the target finding across both lists; ``(kind, index, ambiguous)``."""
    for kind, findings in (("iraes", iraes), ("suspects", suspects)):
        index, ambiguous = _match_in(findings, correction)
        if index is not None:
            return kind, index, ambiguous
    return None, None, False


def _target_organs(item: dict[str, Any]) -> list[str]:
    """Organs to mirror for *item* (source_organs, else its single organ)."""
    organs = item.get("source_organs") or ([item.get("organ")] if item.get("organ") else [])
    return [str(organ) for organ in organs if organ]


def _mirror_remove(corrected: dict[str, Any], target: dict[str, Any]) -> None:
    """Drop the finding from the per-organ Layer 3 sections (best effort)."""
    for organ in _target_organs(target):
        result = corrected.get("organ_results", {}).get(organ)
        if not isinstance(result, dict):
            continue
        entries = result.get("iraes") or []
        result["iraes"] = [
            item for item in entries if not _same_finding(item, target)
        ]


def _mirror_edit(
    corrected: dict[str, Any], target: dict[str, Any], merged: dict[str, Any]
) -> None:
    """Replace the finding in the per-organ Layer 3 sections (best effort)."""
    for organ in _target_organs(target):
        result = corrected.get("organ_results", {}).get(organ)
        if not isinstance(result, dict):
            continue
        entries = result.get("iraes") or []
        replacement = dict(merged)
        replacement.setdefault("organ", organ)
        result["iraes"] = [
            replacement if _same_finding(item, target) else item
            for item in entries
        ]


def _dup_exists(corrected: dict[str, Any], new_item: dict[str, Any]) -> bool:
    consolidation = corrected.get("consolidation") or {}
    for item in list(corrected.get("iraes") or []) + list(
        consolidation.get("suspects") or []
    ):
        if _same_finding(item, new_item):
            return True
    return False


def _apply_one(
    corrected: dict[str, Any], correction: IraeCorrection
) -> dict[str, Any] | None:
    """Apply a single correction to the corrected report; None = no target."""
    iraes = corrected.get("iraes") or []
    consolidation = corrected.get("consolidation") or {}
    suspects = consolidation.get("suspects") or []

    if correction.action == "add":
        return _apply_add(corrected, correction)

    kind, index, ambiguous = _locate(iraes, suspects, correction)
    if kind is None:
        return None
    findings = iraes if kind == "iraes" else suspects
    target = findings[index]

    if correction.action == "remove":
        del findings[index]
        _mirror_remove(corrected, target)
        return {
            "action": "remove",
            "irAE_type": correction.irAE_type,
            "kind": kind,
            "finding_id": target.get("finding_id", ""),
            "ambiguous": ambiguous,
            "applied": True,
        }

    if correction.action == "edit":
        merged = dict(target)
        for key, value in (correction.fields or {}).items():
            if key in EDITABLE_FIELDS:
                merged[key] = value
        if correction.irAE_type:
            merged["irAE_type"] = correction.irAE_type
        if correction.first_onset_date:
            merged["first_onset_date"] = correction.first_onset_date
        target_partition = _partition_for(merged.get("probability_immune"))
        del findings[index]
        if target_partition is None:
            # Edited into an excluded probability: not shown anywhere.
            _mirror_remove(corrected, target)
            return {
                "action": "edit",
                "irAE_type": correction.irAE_type,
                "kind": kind,
                "finding_id": target.get("finding_id", ""),
                "moved_to": "excluded",
                "ambiguous": ambiguous,
                "applied": True,
            }
        if target_partition == kind:
            findings.insert(index, merged)
        else:
            other = iraes if target_partition == "iraes" else suspects
            other.append(merged)
        _mirror_edit(corrected, target, merged)
        return {
            "action": "edit",
            "irAE_type": correction.irAE_type,
            "kind": kind,
            "finding_id": target.get("finding_id", ""),
            "moved_to": target_partition if target_partition != kind else None,
            "ambiguous": ambiguous,
            "applied": True,
        }

    return None


def _apply_add(corrected: dict[str, Any], correction: IraeCorrection) -> dict[str, Any]:
    """Introduce a manually identified irAE (definitive or suspect by
    probability); refuses duplicates of an existing finding."""
    new_item: dict[str, Any] = {}
    for key, value in (correction.fields or {}).items():
        if key in EDITABLE_FIELDS:
            new_item[key] = value
    if correction.irAE_type:
        new_item["irAE_type"] = correction.irAE_type
    if correction.first_onset_date:
        new_item["first_onset_date"] = correction.first_onset_date
    probability = str(new_item.get("probability_immune") or "PROBABILE")
    new_item["probability_immune"] = probability
    partition = _partition_for(probability)
    if partition is None:
        return {
            "action": "add",
            "irAE_type": correction.irAE_type,
            "moved_to": "excluded",
            "applied": True,
        }
    if _dup_exists(corrected, new_item):
        return {
            "action": "add",
            "irAE_type": correction.irAE_type,
            "applied": False,
            "reason": "duplicato",
        }
    consolidation = corrected.get("consolidation")
    if not isinstance(consolidation, dict):
        consolidation = {}
        corrected["consolidation"] = consolidation
    if partition == "iraes":
        iraes = corrected.setdefault("iraes", [])
        new_item["finding_id"] = f"irAE-{len(iraes) + len(consolidation.get('suspects') or []) + 1}"
        iraes.append(new_item)
    else:
        suspects = consolidation.setdefault("suspects", [])
        new_item["finding_id"] = f"irAE-{len(corrected.get('iraes') or []) + len(suspects) + 1}"
        suspects.append(new_item)
    return {
        "action": "add",
        "irAE_type": correction.irAE_type,
        "moved_to": partition,
        "applied": True,
    }


def apply_irae_corrections(
    report: dict[str, Any],
    corrections: list[IraeCorrection],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Apply *corrections* to a fresh *report* (pure, idempotent).

    Returns ``(corrected_report, applied_log)``; the input report and its
    nested structures are never mutated.  ``corrected_report`` carries stable
    ``finding_id`` values on every visible finding.
    """
    corrected = annotate_report(report)
    log: list[dict[str, Any]] = []
    if not corrections:
        return corrected, log
    for correction in corrections:
        result = _apply_one(corrected, correction)
        if result is None:
            log.append({
                "action": correction.action,
                "irAE_type": correction.irAE_type,
                "applied": False,
                "reason": "nessuna corrispondenza",
            })
        else:
            log.append(result)
    return corrected, log
