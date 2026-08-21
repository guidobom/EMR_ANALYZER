"""Run reproducible evaluation from patient-level JSONL gold records."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Iterable

from .metrics import aggregate_patient_metrics, evaluate_patient_events


class EvaluationRunner:
    def __init__(self, registry_repo=None):
        self.registry_repo = registry_repo

    def evaluate_records(self, records: Iterable[dict]) -> dict:
        patient_results = []
        for record in records:
            patient_id = str(record.get("patient_id") or "")
            gold = list(record.get("gold_events") or [])
            predicted = record.get("predicted_events")
            if predicted is None:
                if self.registry_repo is None:
                    raise ValueError(
                        "predicted_events assenti e repository non configurato"
                    )
                predicted = []
                for event in self.registry_repo.get_events(patient_id):
                    item = asdict(event)
                    detail = self.registry_repo.get_event_detail(event.event_id) or {}
                    item["evidence_ids"] = [
                        evidence.get("evidence_id")
                        for evidence in detail.get("evidence", [])
                    ]
                    predicted.append(item)
            metrics = evaluate_patient_events(gold, list(predicted))
            patient_results.append({"patient_id": patient_id, **metrics})
        return {
            "schema": "emr_analyzer.evaluation.v1",
            "aggregate": aggregate_patient_metrics(patient_results),
            "patients": patient_results,
        }

    def evaluate_jsonl(self, path: str | Path) -> dict:
        records = []
        with Path(path).open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                item = json.loads(line)
                if not isinstance(item, dict) or not item.get("patient_id"):
                    raise ValueError(f"Record JSONL non valido alla riga {line_number}")
                records.append(item)
        return self.evaluate_records(records)

    @staticmethod
    def save_report(report: dict, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
