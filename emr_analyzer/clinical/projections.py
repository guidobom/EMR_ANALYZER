"""Deterministic laboratory, medication and oncology projections."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .consolidation import (
    ConsolidatedBundle,
    best_date_precision,
    canonicalize_entity,
    display_entity,
    stable_id,
)
from .temporal import date_sort_key
from ..models.clinical_evidence import ClinicalEvidence
from ..models.clinical_registry import (
    ClinicalEpisode,
    ClinicalEvent,
    EventEvidenceLink,
    LabTrend,
    MedicationCourse,
    OncologyLine,
)


class LabTrendBuilder:
    def __init__(self, db):
        self.db = db

    def build(
        self,
        patient_id: str,
        lab_evidence: Iterable[ClinicalEvidence],
    ) -> tuple[list[LabTrend], list[ConsolidatedBundle]]:
        evidence_by_parameter: dict[str, list[ClinicalEvidence]] = defaultdict(list)
        for item in lab_evidence:
            if item.category in {"laboratory", "laboratory_finding"}:
                evidence_by_parameter[item.normalized_entity].append(item)
        rows = self.db.execute(
            """SELECT * FROM lab_values WHERE patient_id=?
               ORDER BY normalized_name, COALESCE(sample_date, '9999'), id""",
            (patient_id,),
        ).fetchall()
        grouped = defaultdict(list)
        for row in rows:
            grouped[row["normalized_name"]].append(row)

        trends: list[LabTrend] = []
        bundles: list[ConsolidatedBundle] = []
        for parameter, values in grouped.items():
            abnormal = [row for row in values if row["is_abnormal"]]
            if not abnormal:
                continue
            last = values[-1]
            resolved = bool(
                not last["is_abnormal"]
                and date_sort_key(last["sample_date"])
                >= date_sort_key(abnormal[-1]["sample_date"])
            )
            if len(abnormal) < 2 and not resolved:
                # The atomic abnormal event already represents a single value.
                continue
            units = {row["unit"] for row in values if row["unit"]}
            compatible_units = len(units) <= 1
            numeric = [row for row in values if row["value"] is not None]
            direction = _numeric_direction(numeric) if compatible_units else "incompatibile"
            first_date = next(
                (row["sample_date"] for row in abnormal if row["sample_date"]),
                None,
            )
            end_date = last["sample_date"]
            summary = _lab_summary(
                parameter, values, direction=direction, resolved=resolved,
                compatible_units=compatible_units,
            )
            trend_id = stable_id("TRD", patient_id, parameter, first_date or "unknown")
            event_id = stable_id(
                "EVT", patient_id, "laboratory_trend", parameter,
                first_date or "unknown",
            )
            episode_id = stable_id(
                "EPI", patient_id, "laboratory_trend", parameter,
                first_date or "unknown",
            )
            trend = LabTrend(
                trend_id=trend_id, patient_id=patient_id,
                normalized_name=parameter, summary=summary,
                start_date=first_date, end_date=end_date,
                direction=direction, resolved=resolved,
                lab_value_ids=[int(row["id"]) for row in values],
                event_id=event_id,
                data={
                    "units": sorted(units),
                    "unit_compatible": compatible_units,
                    "abnormal_count": len(abnormal),
                    "value_count": len(values),
                },
            )
            trends.append(trend)
            relevant_evidence = evidence_by_parameter.get(parameter, [])
            episode = ClinicalEpisode(
                episode_id=episode_id, patient_id=patient_id,
                category="laboratory_trend", canonical_entity=parameter,
                onset_date=first_date,
                onset_precision=best_date_precision(relevant_evidence, first_date),
                resolution_date=end_date if resolved else None,
                first_documented_date=min(
                    (item.document_date for item in relevant_evidence if item.document_date),
                    key=date_sort_key, default=None,
                ),
                status="resolved" if resolved else "active",
            )
            event = ClinicalEvent(
                event_id=event_id, patient_id=patient_id,
                episode_id=episode_id, category="laboratory_trend",
                canonical_entity=parameter, summary_short=summary[:500],
                summary_detail=summary,
                significance="clinically_relevant",
                status="resolved" if resolved else "active",
                certainty="confirmed", assertion="present",
                first_evidence_date=first_date,
                first_documented_date=episode.first_documented_date,
                date_end=end_date if resolved else None,
                date_precision=episode.onset_precision,
                confidence=0.85 if compatible_units else 0.55,
                review_status="auto" if compatible_units else "pending",
                structured_data=trend.data,
            )
            links = [
                EventEvidenceLink(
                    link_id=stable_id(
                        "LNK", event_id, item.evidence_id, "supports"
                    ),
                    event_id=event_id, evidence_id=item.evidence_id,
                    relation="supports", relation_confidence=0.9,
                    rationale="Punto della serie laboratoristica",
                )
                for item in relevant_evidence
            ]
            bundles.append(ConsolidatedBundle(episode, event, links, []))
        return trends, bundles


class TherapyProjectionBuilder:
    def build(
        self,
        patient_id: str,
        evidence: Iterable[ClinicalEvidence],
    ) -> tuple[
        list[MedicationCourse], list[OncologyLine], list[ConsolidatedBundle]
    ]:
        items = list(evidence)
        medication_items = [
            item for item in items
            if item.category in {"medication", "treatment"}
            or isinstance(item.data.get("therapy"), dict)
        ]
        medications = self._medications(patient_id, medication_items)
        oncology_items = [
            item for item in items
            if item in medication_items or isinstance(item.data.get("oncology"), dict)
        ]
        lines, bundles = self._oncology_lines(patient_id, oncology_items)
        return medications, lines, bundles

    @staticmethod
    def _medications(
        patient_id: str, items: list[ClinicalEvidence]
    ) -> list[MedicationCourse]:
        grouped: dict[str, list[ClinicalEvidence]] = defaultdict(list)
        for item in items:
            therapy = item.data.get("therapy") or {}
            name = therapy.get("active_ingredient") or item.normalized_entity
            normalized = canonicalize_entity(name)
            if normalized:
                grouped[normalized].append(item)
        courses = []
        for name, evidence in grouped.items():
            evidence.sort(key=lambda item: date_sort_key(
                item.observed_date or item.document_date
            ))
            first, last = evidence[0], evidence[-1]
            latest = last.data.get("therapy") or {}
            statuses = [
                str((item.data.get("therapy") or {}).get("lifecycle_status")
                    or item.clinical_status or "unknown").casefold()
                for item in evidence
            ]
            status = _medication_status(statuses)
            originals = []
            for item in evidence:
                original = (item.data.get("therapy") or {}).get(
                    "original_name"
                ) or item.normalized_entity
                if original and original not in originals:
                    originals.append(str(original))
            courses.append(MedicationCourse(
                course_id=stable_id("MED", patient_id, name),
                patient_id=patient_id, normalized_name=name,
                original_names=originals,
                indication=_first_nonempty(
                    (item.data.get("therapy") or {}).get("indication")
                    for item in evidence
                ),
                intent=_first_nonempty(
                    (item.data.get("therapy") or {}).get("intent")
                    for item in evidence
                ),
                lifecycle_status=status,
                start_date=first.observed_date or first.document_date,
                end_date=(
                    last.observed_date or last.document_date
                    if status in {"completed", "suspended", "cancelled"}
                    else None
                ),
                dose=latest.get("dose"), route=latest.get("route"),
                frequency=latest.get("frequency"),
                adherence=latest.get("adherence"),
                event_ids=[stable_id(
                    "EVT", patient_id, "medication", name, "1"
                )],
                data={
                    "evidence_ids": [item.evidence_id for item in evidence],
                    "status_history": statuses,
                },
            ))
        return courses

    @staticmethod
    def _oncology_lines(
        patient_id: str, items: list[ClinicalEvidence]
    ) -> tuple[list[OncologyLine], list[ConsolidatedBundle]]:
        grouped: dict[str, list[ClinicalEvidence]] = defaultdict(list)
        for item in items:
            oncology = item.data.get("oncology") or {}
            line = oncology.get("line") or oncology.get("line_label")
            regimen = oncology.get("regimen")
            if not line and not regimen:
                continue
            if isinstance(regimen, list):
                regimen_key = "+".join(canonicalize_entity(name) for name in regimen)
            else:
                regimen_key = canonicalize_entity(regimen or item.normalized_entity)
            key = canonicalize_entity(str(line or "linea_non_specificata"))
            if key == "linea_non_specificata":
                key += "_" + regimen_key
            grouped[key].append(item)
        lines: list[OncologyLine] = []
        bundles: list[ConsolidatedBundle] = []
        for key, evidence in grouped.items():
            evidence.sort(key=lambda item: date_sort_key(
                item.observed_date or item.document_date
            ))
            first, last = evidence[0], evidence[-1]
            oncology_payloads = [item.data.get("oncology") or {} for item in evidence]
            label = _first_nonempty(
                payload.get("line_label") or payload.get("line")
                for payload in oncology_payloads
            ) or display_entity(key)
            regimen = []
            cycles, modifications, toxicities, responses = [], [], [], []
            for item, payload in zip(evidence, oncology_payloads):
                raw_regimen = payload.get("regimen")
                names = raw_regimen if isinstance(raw_regimen, list) else [
                    raw_regimen or item.normalized_entity
                ]
                for name in names:
                    if name and str(name) not in regimen:
                        regimen.append(str(name))
                for target, field_name in (
                    (cycles, "cycle"), (modifications, "modification"),
                    (toxicities, "toxicity"), (responses, "response"),
                ):
                    value = payload.get(field_name)
                    if value:
                        target.append({
                            "date": item.observed_date or item.document_date,
                            "value": value,
                            "evidence_id": item.evidence_id,
                        })
            start_date = first.observed_date or first.document_date
            end_status = _medication_status([
                str(item.clinical_status or "unknown") for item in evidence
            ])
            end_date = (
                last.observed_date or last.document_date
                if end_status in {"completed", "suspended", "cancelled"}
                else None
            )
            line_id = stable_id("ONC", patient_id, key)
            event_id = stable_id(
                "EVT", patient_id, "oncology_treatment_line", key
            )
            episode_id = stable_id(
                "EPI", patient_id, "oncology_treatment_line", key
            )
            summary = (
                f"{label}: schema {', '.join(regimen)}"
                + (f"; inizio {start_date}" if start_date else "")
                + (f"; conclusione/sospensione {end_date}" if end_date else "")
            )
            detail_parts = [summary]
            if modifications:
                detail_parts.append(
                    "Modifiche: " + "; ".join(
                        f"{item['date'] or 'data n.d.'}: {item['value']}"
                        for item in modifications
                    )
                )
            if toxicities:
                detail_parts.append(
                    "Tossicità: " + "; ".join(
                        f"{item['date'] or 'data n.d.'}: {item['value']}"
                        for item in toxicities
                    )
                )
            if responses:
                detail_parts.append(
                    "Risposta/progressione: " + "; ".join(
                        f"{item['date'] or 'data n.d.'}: {item['value']}"
                        for item in responses
                    )
                )
            line = OncologyLine(
                line_id=line_id, patient_id=patient_id, line_label=str(label),
                regimen=regimen,
                setting=_first_nonempty(p.get("setting") for p in oncology_payloads),
                intent=_first_nonempty(p.get("intent") for p in oncology_payloads),
                start_date=start_date, end_date=end_date, status=end_status,
                cycles=cycles, modifications=modifications,
                toxicities=[str(item["value"]) for item in toxicities],
                responses=[str(item["value"]) for item in responses],
                event_ids=[event_id],
            )
            lines.append(line)
            episode = ClinicalEpisode(
                episode_id=episode_id, patient_id=patient_id,
                category="oncology_treatment_line", canonical_entity=key,
                onset_date=start_date,
                onset_precision=best_date_precision(evidence, start_date),
                resolution_date=end_date, status=end_status,
                first_documented_date=min(
                    (item.document_date for item in evidence if item.document_date),
                    key=date_sort_key, default=None,
                ),
            )
            event = ClinicalEvent(
                event_id=event_id, patient_id=patient_id,
                episode_id=episode_id, category="oncology_treatment_line",
                canonical_entity=key, summary_short=summary[:500],
                summary_detail=". ".join(detail_parts), status=end_status,
                certainty="confirmed", assertion="present",
                first_evidence_date=start_date,
                first_documented_date=episode.first_documented_date,
                date_end=end_date, date_precision=episode.onset_precision,
                confidence=0.8, review_status="auto",
                structured_data={
                    "line_id": line_id, "regimen": regimen,
                    "cycles": cycles, "modifications": modifications,
                    "toxicities": toxicities, "responses": responses,
                },
            )
            links = [
                EventEvidenceLink(
                    link_id=stable_id(
                        "LNK", event_id, item.evidence_id, "supports"
                    ),
                    event_id=event_id, evidence_id=item.evidence_id,
                    relation="supports", relation_confidence=0.8,
                    rationale="Evidenza relativa alla linea oncologica",
                )
                for item in evidence
            ]
            bundles.append(ConsolidatedBundle(episode, event, links, []))
        return lines, bundles


def _numeric_direction(rows: list) -> str:
    if len(rows) < 2:
        return "non_valutabile"
    first, last = float(rows[0]["value"]), float(rows[-1]["value"])
    scale = max(abs(first), 1e-9)
    change = (last - first) / scale
    if change > 0.05:
        return "aumento"
    if change < -0.05:
        return "riduzione"
    return "stabile"


def _lab_summary(
    parameter: str,
    values: list,
    *,
    direction: str,
    resolved: bool,
    compatible_units: bool,
) -> str:
    abnormal = [row for row in values if row["is_abnormal"]]
    first = abnormal[0]
    last = values[-1]
    first_value = _format_lab_value(first)
    last_value = _format_lab_value(last)
    status = "con successiva normalizzazione" if resolved else "persistente"
    if not compatible_units:
        status = "con unità non direttamente confrontabili"
    return (
        f"{display_entity(parameter)}: alterazione {status}; "
        f"prima evidenza {first['sample_date'] or 'data n.d.'} "
        f"({first_value}), ultimo controllo "
        f"{last['sample_date'] or 'data n.d.'} ({last_value}); "
        f"andamento {direction}."
    )


def _format_lab_value(row) -> str:
    if row["value_text"]:
        return str(row["value_text"])
    if row["value"] is None:
        return "valore n.d."
    operator = row["operator"] or ""
    unit = f" {row['unit']}" if row["unit"] else ""
    return f"{operator}{row['value']:g}{unit}"


def _medication_status(statuses: list[str]) -> str:
    if not statuses:
        return "unknown"
    latest = statuses[-1].casefold()
    if any(token in latest for token in ("sosp", "interrott", "stopped")):
        return "suspended"
    if any(token in latest for token in ("complet", "terminat")):
        return "completed"
    if any(token in latest for token in ("cancell", "non assunt")):
        return "cancelled"
    if any(token in latest for token in ("propost", "planned")):
        return "planned"
    if any(token in latest for token in (
        "active", "ongoing", "assunt", "somministr", "prescritt", "inizi"
    )):
        return "active"
    return "unknown"


def _first_nonempty(values: Iterable):
    return next((value for value in values if value not in (None, "", [], {})), None)
