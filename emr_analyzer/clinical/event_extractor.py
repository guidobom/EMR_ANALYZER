"""Event Extractor — orchestrates clinical event extraction from documents."""

from ..extraction.qwen_client import QwenClient
from ..extraction.lab_parser import LabParser
from ..extraction.radiology_extractor import RadiologyExtractor
from .event_store import EventStore
from ..models.clinical_event import ClinicalEvent, EventType
from ..models.lab_result import LabValue


class EventExtractor:
    """
    Orchestrates the extraction of clinical events from a document.
    Combines deterministic parsers (lab, radiology) with LLM extraction.
    """

    def __init__(self, qwen_client: QwenClient,
                 lab_parser: LabParser,
                 radiology_extractor: RadiologyExtractor,
                 event_store: EventStore):
        self._qwen = qwen_client
        self._lab_parser = lab_parser
        self._rad_extractor = radiology_extractor
        self._event_store = event_store

    def extract_all(self, text: str, patient_id: str,
                    document_id: str,
                    document_date: str = None) -> dict:
        """
        Run all extractors on a document and return combined results.
        """
        result = {
            "patient_id": patient_id,
            "document_id": document_id,
            "events": [],
            "lab_values": [],
            "radiology_report": None,
        }

        # 1. Lab values (deterministic)
        lab_values = self._lab_parser.parse(
            text, patient_id=patient_id, document_id=document_id
        )
        result["lab_values"] = lab_values

        # Convert significant lab alterations to events
        for lv in lab_values:
            if lv.is_abnormal and lv.confidence >= 0.7:
                event = ClinicalEvent(
                    event_id="",  # Will be assigned by EventStore
                    patient_id=patient_id,
                    event_date=lv.sample_date or document_date or "",
                    event_type=EventType.LAB_ALTERATION.value,
                    entity=f"{lv.parameter_name} ({lv.flag})",
                    value=lv.value,
                    unit=lv.unit,
                    source_document_id=document_id,
                    source_text=lv.source_text,
                    confidence=lv.confidence,
                )
                result["events"].append(event)

        # 2. Radiology report (if applicable)
        rad_report = self._rad_extractor.extract(text, document_id)
        if rad_report.conclusions or rad_report.findings:
            result["radiology_report"] = rad_report

            # Convert radiology findings to events
            if rad_report.conclusions:
                result["events"].append(ClinicalEvent(
                    event_id="",
                    patient_id=patient_id,
                    event_date=document_date or "",
                    event_type=EventType.RADIOLOGY_FINDING.value,
                    entity=rad_report.conclusions[:200],
                    source_document_id=document_id,
                    source_text=rad_report.conclusions,
                    confidence=0.85,
                ))

        # 3. Qwen3-14B clinical event extraction
        if self._qwen and self._qwen.is_available:
            llm_events = self._qwen.extract_clinical_events(
                text, patient_id, document_id
            )
            result["events"].extend(llm_events)

        return result
