"""Speed-related behaviour: stable caches, shared pool, skipped lab reports."""

import json
import uuid

import pytest

from emr_analyzer.clinical.extraction_pipeline import ExtractionCancelled
from emr_analyzer.models.lab_result import LabValue
from tests.helpers import FakeLlm, KeywordExtractor, Project


class ExtractorLlm(FakeLlm):
    model_path = None
    temperature = 0.0
    seed = 42
    top_p = 0.9
    top_k = 40
    context_length = 32768
    max_output_tokens = 4096
    thinking_enabled = False


SYMPTOMS = ["tosse", "febbre", "dispnea", "astenia", "nausea", "vomito", "cefalea",
            "diarrea", "prurito", "edema", "anemia", "rash"]


def test_unrelated_lexicon_example_and_snomed_alias_keep_the_fingerprint(tmp_path, workspace):
    from emr_analyzer.clinical.event_extraction import EventExtractor
    from emr_analyzer.clinical.grounded_sources import catalog_snapshot
    from emr_analyzer.database.engine import DatabaseEngine
    from emr_analyzer.database.migrations import init_database
    from emr_analyzer.database.shared_lexicon_repo import SharedLexiconRepository

    project_db = DatabaseEngine(tmp_path / "ws.db")
    init_database(project_db)
    shared = SharedLexiconRepository(project_db, workspace, tmp_path / "shared_lexicon.db")
    ids = {}
    with shared.db:
        for label in SYMPTOMS + ["ittero"]:
            ids[label] = uuid.uuid4().hex
            shared.db.execute("INSERT INTO local_lexicon_terms VALUES (?,?,?)", (ids[label], label, label))
    snomed = shared.snomed_catalog
    with snomed.db:
        snomed.db.execute("INSERT INTO sct_meta VALUES ('format','RF2 International Snapshot')")
        snomed.db.execute("INSERT INTO sct_meta VALUES ('sha256','synthetic')")
        snomed.db.execute("INSERT INTO sct_concepts VALUES ('18165001',1,'Jaundice (finding)','Jaundice','finding')")
    # Twelve terms occur in the text and fill all guidance cards; "ittero" does not.
    text = "Riferisce " + ", ".join(SYMPTOMS) + "."
    extractor = lambda: EventExtractor(ExtractorLlm(), snomed_catalog=snomed, catalog=catalog_snapshot(shared))

    before = extractor()
    shared.save_example(ids["ittero"], "Cute itterica.")
    snomed.add_alias("ittero", "18165001")
    after = extractor()
    assert before.model_digest == after.model_digest
    assert before.guidance_digest(text) == after.guidance_digest(text)
    # Guidance that is actually sent still makes the document stale.
    shared.save_example(ids["tosse"], "Tosse secca da una settimana.")
    assert extractor().guidance_digest(text) != before.guidance_digest(text)
    shared.close()
    project_db.close()


@pytest.fixture
def project(workspace):
    project = Project(workspace)
    for patient in ("P001", "P002"):
        project.add_patient(patient)
    project.add_document("P001", "DOC_001", "Riferisce tosse.")
    project.add_document("P002", "DOC_002", "Prosegue nivolumab. Febbre.")
    project.add_document("P002", "DOC_LAB", "Glucosio 90 mg/dL (70-110)", document_type="laboratorio")
    project.labs.insert(LabValue(patient_id="P002", document_id="DOC_LAB", parameter_name="Glucosio",
                                 normalized_name="glucosio", value=90, unit="mg/dL"))
    yield project
    project.close()


def test_shared_pool_finalizes_each_patient_and_skips_parsed_lab_reports(project, workspace):
    extractor = KeywordExtractor()
    finished = []
    plans = project.pipeline(FakeLlm(), extractor).extract_patients(
        ["P001", "P002"], patient_finished=lambda pid, result: finished.append((pid, result)))

    assert [pid for pid, _ in finished] == ["P001", "P002"]
    assert sorted(extractor.calls) == ["DOC_001", "DOC_002"]          # DOC_LAB not sent to the model
    assert plans["P002"].result["laboratory_documents"] == 1
    for patient in ("P001", "P002"):
        assert (workspace / patient / "clinical_events.fhir.json").is_file()
    bundle = json.loads((workspace / "P002" / "clinical_events.fhir.json").read_text(encoding="utf-8"))
    assert any(entry["resource"].get("valueQuantity", {}).get("value") == 90 for entry in bundle["entry"])

    extractor.calls.clear()
    plans = project.pipeline(FakeLlm(), extractor).extract_patients(["P001", "P002"])
    assert extractor.calls == [] and plans["P002"].result["documents_skipped"] == 2


def test_cancellation_keeps_completed_documents_and_interrupts_runs(project):
    extractor = KeywordExtractor()
    pipeline = project.pipeline(FakeLlm(), extractor)
    with pytest.raises(ExtractionCancelled):
        pipeline.extract_patients(["P001", "P002"], cancel_check=lambda: bool(extractor.calls))
    assert extractor.calls == ["DOC_001"]
    assert [item.normalized_entity for item in project.evidence.get_by_patient("P001")] == ["tosse"]
    statuses = {row[0] for row in project.db.execute("SELECT status FROM processing_runs")}
    assert "running" not in statuses


def test_loinc_proposals_are_cached(tmp_path):
    from emr_analyzer.clinical.loinc_catalog import LoincCatalog

    catalog = LoincCatalog(tmp_path / "loinc.db")
    axes = {"COMPONENT": "Glucose", "PROPERTY": "MCnc", "TIME_ASPCT": "Pt", "SYSTEM": "Ser/Plas",
            "SCALE_TYP": "Qn", "METHOD_TYP": "", "EXAMPLE_UCUM_UNITS": "mg/dL", "EXAMPLE_UNITS": "",
            "CLASSTYPE": "1", "ORDER_OBS": "Both"}
    with catalog.db:
        catalog.db.execute("INSERT INTO loinc_terms VALUES ('2345-7','ACTIVE','Glucose [Mass/volume]',?)",
                           (json.dumps(axes),))
        catalog.db.execute("INSERT INTO loinc_components VALUES ('glucosio','2345-7')")

    class Llm(ExtractorLlm):
        calls = 0

        def generate_structured(self, prompt, system, schema, max_tokens=None):
            Llm.calls += 1
            return {"mappings": [{"id": "0", "code": "2345-7"}]}

    lab = LabValue(patient_id="P", document_id="D", parameter_name="Glucosio", normalized_name="glucosio",
                   value=90, unit="mg/dL", biological_material="siero")
    first = catalog.propose([lab], Llm())
    second = catalog.propose([lab], Llm())
    assert first == second and next(iter(first.values()))["code"] == "2345-7"
    assert Llm.calls == 1
