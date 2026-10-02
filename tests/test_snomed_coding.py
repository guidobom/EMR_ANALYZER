"""Concept-level SNOMED CT coding with a synthetic catalog and a scripted model."""

import json

import pytest

from emr_analyzer.clinical.evidence_utils import content_hash
from emr_analyzer.clinical.grounded_sources import METHOD
from emr_analyzer.clinical.snomed_catalog import SnomedCatalog
from emr_analyzer.clinical.snomed_coding import ConceptCoder, concept_key
from emr_analyzer.database.concept_mapping_repo import ConceptMappingRepository
from emr_analyzer.database.engine import DatabaseEngine
from emr_analyzer.models.clinical_evidence import ClinicalEvidence

CONCEPTS = [
    ("38341003", "Hypertensive disorder, systemic arterial (disorder)", "Hypertension", "disorder"),
    ("271649006", "Systolic blood pressure (observable entity)", "Systolic blood pressure", "observable entity"),
    ("387458008", "Aspirin (substance)", "Aspirin", "substance"),
    ("267036007", "Dyspnea (finding)", "Dyspnea", "finding"),
]
ENGLISH = {"Ipertensione arteriosa": "arterial hypertension", "Aspirina": "aspirin",
           "Pressione sistolica": "systolic blood pressure", "Dispnea": "dyspnea",
           "Concetto ignoto": "unknown thing"}


def make_catalog(path):
    catalog = SnomedCatalog(path)
    with catalog.db:
        for key, value in (("format", "RF2 International Snapshot"), ("sha256", "synthetic"),
                           ("release", "20260901")):
            catalog.db.execute("INSERT INTO sct_meta VALUES (?,?)", (key, value))
        for table, columns in (("sct_rf2_description", ("id", "conceptId", "term", "active", "languageCode", "typeId")),
                               ("sct_rf2_language", ("referencedComponentId", "active", "refsetId", "acceptabilityId")),
                               ("sct_rf2_relationship", ("sourceId", "destinationId", "active", "typeId",
                                                         "characteristicTypeId", "relationshipGroup"))):
            catalog.db.execute(f"CREATE TABLE {table} ({', '.join(columns)})")
        for code, fsn, term, tag in CONCEPTS:
            catalog.db.execute("INSERT INTO sct_concepts VALUES (?,?,?,?,?)", (code, 1, fsn, term, tag))
            catalog.db.execute("INSERT INTO sct_search VALUES (?,?)", (code, term))
            catalog.db.execute("INSERT INTO sct_search VALUES (?,?)", (code, fsn))
    return catalog


class ScriptedLlm:
    model = "scripted"
    is_available = True
    context_length = 32768
    max_output_tokens = 4096

    def __init__(self, abstain=()):
        self.calls = []
        self.abstain = set(abstain)

    def generate_structured(self, prompt, system, schema, max_tokens=None):
        payload = json.loads(prompt)
        properties = schema["properties"]
        if "queries" in properties:
            self.calls.append("translate")
            return {"queries": [{"id": item["id"], "english": ENGLISH[item["label"]]} for item in payload]}
        self.calls.append("select")
        return {"mappings": [{"id": item["id"],
                              "code": None if item["etichetta"] in self.abstain else item["candidati"][0]["code"]}
                             for item in payload]}


def event(label, fact_type, document="DOC_1", quote=None):
    quote = quote or label.lower()
    return ClinicalEvidence(patient_id="P001", document_id=document,
                            evidence_id="EVD_" + content_hash(document, label, fact_type),
                            category=fact_type, fact_type=fact_type, normalized_entity=label,
                            source_text=quote, extraction_method=METHOD, data={"fhir_pipeline": True})


@pytest.fixture
def coder(tmp_path):
    catalog = make_catalog(tmp_path / "snomed_ct.db")
    db = DatabaseEngine(tmp_path / "shared.db")
    mappings = ConceptMappingRepository(db)
    yield lambda llm: ConceptCoder(llm, catalog, mappings)
    catalog.db.close()
    db.close()


EVENTS = [event("Ipertensione arteriosa", "diagnosis"), event("Ipertensione arteriosa", "diagnosis", "DOC_2"),
          event("Aspirina", "medication"), event("Pressione sistolica", "vital_sign"),
          event("Concetto ignoto", "diagnosis")]


def test_each_concept_is_coded_once_within_its_hierarchy(coder):
    llm = ScriptedLlm()
    stats = coder(llm).code_events(EVENTS)
    assert stats["concepts"] == 4 and llm.calls == ["translate", "select"]
    mappings = coder(llm).resolve(EVENTS)
    assert mappings[concept_key("Ipertensione arteriosa", "diagnosis")]["code"] == "38341003"
    assert mappings[concept_key("Aspirina", "medication")]["code"] == "387458008"
    assert mappings[concept_key("Pressione sistolica", "vital_sign")]["code"] == "271649006"
    unknown = mappings[concept_key("Concetto ignoto", "diagnosis")]
    assert unknown["status"] == "needs_review" and unknown["code"] is None

    llm.calls.clear()
    assert coder(llm).code_events(EVENTS + [event("Aspirina", "medication", "DOC_9")])["cached"] == 4
    assert llm.calls == []


def test_candidates_are_restricted_to_the_event_hierarchy(coder):
    llm = ScriptedLlm()
    seen = []
    original = llm.generate_structured

    def spy(prompt, system, schema, max_tokens=None):
        if "mappings" in schema["properties"]:
            seen.extend(row["code"] for item in json.loads(prompt) for row in item["candidati"])
        return original(prompt, system, schema, max_tokens)

    llm.generate_structured = spy
    coder(llm).code_events([event("Aspirina", "medication")])
    assert seen and set(seen) == {"387458008"}


def test_model_abstention_and_reviewer_decisions(coder):
    llm = ScriptedLlm(abstain={"Dispnea"})
    instance = coder(llm)
    instance.code_events([event("Dispnea", "symptom")])
    key = concept_key("Dispnea", "symptom")
    assert instance.resolve([event("Dispnea", "symptom")])[key]["status"] == "needs_review"

    instance.mappings.confirm(*key, "Dispnea", code="267036007", concept={"fsn": "Dyspnea (finding)"})
    instance.mappings.reset()                           # automatic outcomes only
    instance.code_events([event("Dispnea", "symptom")])
    row = instance.resolve([event("Dispnea", "symptom")])[key]
    assert (row["status"], row["code"]) == ("confirmed", "267036007")


def test_fhir_uses_concept_codes(coder, workspace):
    from emr_analyzer.clinical.fhir_registry import FhirRegistry

    llm = ScriptedLlm()
    instance = coder(llm)
    instance.code_events(EVENTS)
    mappings = instance.resolve(EVENTS)
    registry = FhirRegistry("P001")
    for item in EVENTS:
        registry.clinical(item, mappings.get(concept_key(item.normalized_entity, item.fact_type)))
    codings = [r["code"]["coding"][0]["code"] for r in registry.resources.values()
               if r.get("code", {}).get("coding")]
    assert codings.count("38341003") == 2 and registry.unmapped == 1


def test_patient_processing_codes_concepts_and_exports_them(tmp_path, workspace):
    from tests.helpers import KeywordExtractor, Project

    catalog = make_catalog(tmp_path / "snomed_ct.db")
    with catalog.db:
        for code, fsn, term, tag in (("49727002", "Cough (finding)", "Cough", "finding"),
                                     ("386661006", "Fever (finding)", "Fever", "finding"),
                                     ("704191007", "Nivolumab (substance)", "Nivolumab", "substance")):
            catalog.db.execute("INSERT INTO sct_concepts VALUES (?,?,?,?,?)", (code, 1, fsn, term, tag))
            catalog.db.execute("INSERT INTO sct_search VALUES (?,?)", (code, term))
    ENGLISH.update({"tosse": "cough", "febbre": "fever", "nivolumab": "nivolumab"})
    shared_db = DatabaseEngine(tmp_path / "shared.db")

    class Shared:
        snomed_catalog = catalog
        concept_mappings = ConceptMappingRepository(shared_db)
        loinc_catalog = None
        terms = staticmethod(lambda: [])
        extraction_examples = staticmethod(lambda: [])

    project = Project(workspace)
    project.add_patient("P001")
    project.add_document("P001", "DOC_001", "Riferisce tosse e febbre. Prosegue nivolumab.")
    pipeline = project.pipeline(ScriptedLlm())
    pipeline.shared_lexicon_repo = Shared()
    extractor = KeywordExtractor()
    pipeline.make_extractor = lambda: extractor
    result = pipeline.extract_patient("P001")

    assert result["coding"]["coded"] == 3
    bundle = json.loads((workspace / "P001" / "clinical_events.fhir.json").read_text(encoding="utf-8"))
    codes = {coding["code"] for entry in bundle["entry"]
             for key in ("code", "medicationCodeableConcept")
             for coding in entry["resource"].get(key, {}).get("coding", [])}
    assert {"49727002", "386661006", "704191007"} <= codes
    project.close()
    catalog.db.close()
    shared_db.close()


class FakeEncoder:
    """Deterministic stand-in for a multilingual entity encoder."""
    AXES = (("hyperten", "ipertens"), ("aspirin", "aspirin"), ("dyspn", "dispn"), ("systolic", "sistolic"))

    def __init__(self, *args, **kwargs):
        pass

    def encode(self, texts, normalize_embeddings=True, **kwargs):
        import numpy as np
        rows = []
        for text in texts:
            text = text.casefold()
            vector = np.array([1.0 if any(key in text for key in axis) else 0.0 for axis in self.AXES] + [0.1])
            rows.append(vector / np.linalg.norm(vector))
        return np.array(rows, dtype="float32")


def test_vector_index_links_italian_labels_without_translation(tmp_path, monkeypatch):
    import sentence_transformers

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeEncoder)
    catalog = make_catalog(tmp_path / "snomed_ct.db")
    model_dir = tmp_path / "encoder"
    model_dir.mkdir()
    catalog.build_vectors(model_dir)
    assert catalog.metadata()["embedding_model"] == str(model_dir.resolve())
    found = catalog.search("ipertensione arteriosa", "", limit=3, tags=("disorder", "finding"))
    assert found[0]["code"] == "38341003"
    assert all(row["tag"] in ("disorder", "finding") for row in found)

    db = DatabaseEngine(tmp_path / "shared.db")
    llm = ScriptedLlm()
    ConceptCoder(llm, catalog, ConceptMappingRepository(db)).code_events(
        [event("Ipertensione arteriosa", "diagnosis"), event("Dispnea", "symptom")])
    assert llm.calls == ["select"]                     # no translation call
    catalog.db.close()
    db.close()


def test_stale_index_degrades_instead_of_stopping(tmp_path, monkeypatch):
    """A new SNOMED release must not break coding: it falls back and asks a rebuild."""
    import sentence_transformers

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeEncoder)
    catalog = make_catalog(tmp_path / "snomed_ct.db")
    model_dir = tmp_path / "encoder"
    model_dir.mkdir()
    assert catalog.vector_status()["state"] == "missing"
    catalog.build_vectors(model_dir)
    assert catalog.vector_status()["state"] == "ready"

    # Importing another release changes the catalogue fingerprint.
    with catalog.db:
        catalog.db.execute("INSERT OR REPLACE INTO sct_meta VALUES ('sha256','release-2027')")
    stato = catalog.vector_status()
    assert stato["state"] == "stale" and "ricostruito" in stato["reason"]
    assert isinstance(catalog.search("ipertensione arteriosa", "", limit=3,
                                     tags=("disorder", "finding")), list)

    db = DatabaseEngine(tmp_path / "shared.db")
    llm = ScriptedLlm()
    metrics = ConceptCoder(llm, catalog, ConceptMappingRepository(db)).code_events(
        [event("Ipertensione arteriosa", "diagnosis")])
    assert metrics["vectors"] == "stale"
    assert "translate" in llm.calls          # without usable vectors the label is translated
    catalog.db.close()
    db.close()
