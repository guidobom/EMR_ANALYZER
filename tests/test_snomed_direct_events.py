"""Percorso B — direct events from RAG + LLM (experimental).

``DirectSnomedEventBuilder`` produces document-level ``SnomedDirectEvent``
records with a closed-set wire contract: ``snomed_code`` is an ``enum`` over
the chunk's event-like candidate set and ``observed_date`` is an ``enum`` over
the dates grounded in the source.  Lab events are deterministic and never call
the LLM.  The path is experimental and never the default.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from emr_analyzer.clinical.snomed import (
    SnomedCandidateRetriever,
    SnomedIndex,
    load_snapshot,
)
from emr_analyzer.clinical.snomed.direct_events import (
    DirectSnomedEventBuilder,
    build_direct_event_schema,
    grounded_dates,
)
from emr_analyzer.models.lab_result import LabValue

FIXTURE_DIR = (
    Path(__file__).resolve().parent / "fixtures" / "snomed_rf2" / "Snapshot_20260101"
)


class _DirectLlm:
    model = "fake-direct-medical"
    context_length = 32768
    max_output_tokens = 4096
    is_available = True

    def __init__(self, responses=None):
        self._responses = list(responses or [])
        self.calls = 0
        self.last_schema = None
        self._usage = {"prompt_tokens": 100, "completion_tokens": 20}

    def _valid_from_schema(self, schema):
        item_schema = schema["properties"]["events"]["items"]["properties"]
        codes = item_schema["snomed_code"]["enum"]
        dates = item_schema["observed_date"]["enum"]
        refs = item_schema["source_refs"]["items"]["enum"]
        return {
            "events": [
                {
                    "snomed_code": codes[0],
                    "polarity": "present",
                    "certainty": "definitive",
                    "observed_date": next((d for d in dates if d), ""),
                    "source_refs": [refs[0]],
                }
            ]
        }

    def generate_structured(self, prompt, system, schema, *, max_tokens=None):
        self.calls += 1
        self.last_schema = schema
        self._usage["prompt_tokens"] = 100 + self.calls
        if self._responses:
            return self._responses.pop(0)
        return self._valid_from_schema(schema)

    def last_generation_metadata(self):
        return {"usage": dict(self._usage)}


@pytest.fixture(scope="module")
def snapshot():
    return load_snapshot(FIXTURE_DIR)


def _builder(snapshot, llm, **kwargs):
    retriever = SnomedCandidateRetriever(
        SnomedIndex(snapshot.concepts.values()),
        release_digest=snapshot.digest,
    )
    return DirectSnomedEventBuilder(llm, retriever=retriever, **kwargs)


class TestGroundedDates:
    def test_iso_and_dmy_normalized(self):
        assert grounded_dates(
            "Ricovero 2025-01-10; controllo il 15/03/2025."
        ) == ("2025-01-10", "2025-03-15")

    def test_document_date_merged(self):
        assert grounded_dates(
            "Alcuni esami.", document_date="2025-01-10"
        ) == ("2025-01-10",)

    def test_invalid_dates_ignored(self):
        assert grounded_dates("33/13/2025 e 2025-99-99") == ()

    def test_empty(self):
        assert grounded_dates("") == ()

    def test_cap_keeps_most_recent(self):
        text = " ".join(f"2025-01-{d:02d}" for d in range(1, 20))
        dates = grounded_dates(text, cap=5)
        assert len(dates) == 5
        assert dates[0] == "2025-01-15"


class TestSchemaShape:
    def test_enums_are_closed(self):
        schema = build_direct_event_schema(
            codes=["44054006", "38341003"],
            sentence_count=3,
            dates=["2025-03-15"],
        )
        items = schema["properties"]["events"]["items"]
        assert items["properties"]["snomed_code"]["enum"] == [
            "44054006", "38341003",
        ]
        assert items["properties"]["observed_date"]["enum"] == [
            "", "2025-03-15",
        ]
        assert items["properties"]["source_refs"]["items"]["enum"] == [1, 2, 3]
        assert set(items["required"]) == {
            "snomed_code", "polarity", "certainty",
            "observed_date", "source_refs",
        }


class TestTextEvents:
    def test_event_is_grounded_and_coded(self, snapshot):
        llm = _DirectLlm()
        builder = _builder(snapshot, llm)
        events = builder.build_events(
            patient_id="P001",
            document_id="D1",
            document_type="referto",
            document_date=None,
            text="Il paziente presenta diabete mellito in data 15/03/2025.",
        )
        assert len(events) == 1
        event = events[0]
        assert event.snomed_code == "44054006"
        assert event.label == "Diabete mellito"
        assert event.observed_date == "2025-03-15"
        assert event.polarity == "present"
        assert event.certainty == "definitive"
        assert event.origin == "llm_text"
        assert event.experimental is True
        assert event.release_digest == snapshot.digest
        assert event.source_refs == (1,)
        assert event.source_passages
        assert event.lab_value_ids == ()
        assert llm.calls == 1

    def test_event_like_filter_excludes_observables(self, snapshot):
        # ``glicemia`` (observable) and ``diabete`` (disorder) are both recalled;
        # the event-level enum keeps only event-like codes.
        llm = _DirectLlm()
        builder = _builder(snapshot, llm)
        builder.build_events(
            patient_id="P001",
            document_id="D1",
            document_type="referto",
            document_date=None,
            text="Il paziente presenta glicemia elevata e diabete mellito.",
        )
        enum = llm.last_schema["properties"]["events"]["items"][
            "properties"
        ]["snomed_code"]["enum"]
        assert "44054006" in enum
        assert "33747003" not in enum

    def test_invalid_item_is_repaired_within_closed_enum(self, snapshot):
        # First response violates the closed set (free code + invented date);
        # the repair pass re-serves the same enum and the fake emits a valid one.
        llm = _DirectLlm(responses=[
            {
                "events": [
                    {
                        "snomed_code": "99999999",
                        "polarity": "present",
                        "certainty": "definitive",
                        "observed_date": "1999-01-01",
                        "source_refs": [1],
                    }
                ]
            }
        ])
        builder = _builder(snapshot, llm)
        events = builder.build_events(
            patient_id="P001",
            document_id="D1",
            document_type="referto",
            document_date=None,
            text="Il paziente presenta diabete mellito.",
        )
        assert len(events) == 1
        assert events[0].snomed_code == "44054006"
        metrics = builder.last_metrics()
        assert metrics["validation_retries"] == 1
        assert metrics["repaired_items"] == 1
        assert metrics["llm_calls"] == 2

    def test_empty_candidate_chunk_is_recorded(self, snapshot):
        llm = _DirectLlm()
        builder = _builder(snapshot, llm)
        events = builder.build_events(
            patient_id="P001",
            document_id="D1",
            document_type="referto",
            document_date=None,
            text="zzzqwerty",
        )
        assert events == []
        metrics = builder.last_metrics()
        assert metrics["empty_candidate_chunks"] == 1
        assert llm.calls == 0


class TestLabEvents:
    def test_deterministic_lab_event_no_llm(self, snapshot):
        llm = _DirectLlm()
        builder = _builder(snapshot, llm)
        lab = LabValue(
            patient_id="P001",
            document_id="D1",
            parameter_name="Glicemia",
            normalized_name="glicemia",
            value=180.0,
            reference_low=70.0,
            reference_high=115.0,
            sample_date="2025-02-01",
            source_text="Glicemia 180 mg/dl",
        )
        events = builder.build_events(
            patient_id="P001",
            document_id="D1",
            document_type="laboratorio",
            document_date=None,
            text="",
            lab_values=[lab],
        )
        assert len(events) == 1
        event = events[0]
        assert event.snomed_code == "33747003"
        assert event.label == "Glicemia"
        assert event.observed_date == "2025-02-01"
        assert event.polarity == "present"
        assert event.certainty == "definitive"
        assert event.origin == "deterministic_lab"
        assert event.lab_value_ids
        assert llm.calls == 0

    def test_normal_lab_value_produces_no_event(self, snapshot):
        llm = _DirectLlm()
        builder = _builder(snapshot, llm)
        lab = LabValue(
            patient_id="P001",
            document_id="D1",
            parameter_name="Glicemia",
            normalized_name="glicemia",
            value=95.0,
            reference_low=70.0,
            reference_high=115.0,
            sample_date="2025-02-01",
            source_text="Glicemia 95 mg/dl",
        )
        events = builder.build_events(
            patient_id="P001",
            document_id="D1",
            document_type="laboratorio",
            document_date=None,
            text="",
            lab_values=[lab],
        )
        assert events == []
