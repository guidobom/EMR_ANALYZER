"""Contract tests for the end-to-end clinical pipeline v3."""

from datetime import date, timedelta
import json
from pathlib import Path
import threading
import time

import pytest

from emr_analyzer.clinical.evidence_graph import (
    EvidenceGraphBuilder,
    EvidenceGraphCancelled,
)
from emr_analyzer.clinical.atomic_evidence import AtomicEvidenceExtractor
from emr_analyzer.clinical.lab_evidence import abnormal_lab_evidence
from emr_analyzer.clinical.hypothesis_discovery import HypothesisDiscovery
from emr_analyzer.clinical.registry_builder import (
    _apply_reviewed_duplicate_groups,
    _excluded_record,
)
from emr_analyzer.database.engine import DatabaseEngine
from emr_analyzer.database.evidence_repo import EvidenceRepository
from emr_analyzer.database.gold_set_repo import GoldSetRepository
from emr_analyzer.database.migrations import init_database
from emr_analyzer.database.pipeline_repo import ClinicalPipelineRepository
from emr_analyzer.database.registry_repo import ClinicalRegistryRepository
from emr_analyzer.models.clinical_evidence import ClinicalEvidence
from emr_analyzer.models.clinical_registry import ClinicalEvent, EventEvidenceLink
from emr_analyzer.models.clinical_pipeline import EvidenceSourceReference
from emr_analyzer.models.gold_set import GoldAtomicAnnotation
from emr_analyzer.models.lab_result import LabValue
from emr_analyzer.settings import (
    ClinicalPipelinePolicy,
    LabEvidencePolicy,
    load_pipeline_policy,
    save_pipeline_policy,
)


def _seed(db):
    db.execute(
        """INSERT INTO patients(id,pseudonym,created_at,updated_at)
           VALUES ('P1','P1','2026-01-01','2026-01-01')"""
    )
    for document_id, document_date in (("D1", "2025-01-01"), ("D2", "2025-01-03")):
        db.execute(
            """INSERT INTO documents
               (id,patient_id,filename,original_path,file_hash,document_date,
                document_type,import_date)
               VALUES (?, 'P1', ?, ?, ?, ?, 'referto', '2026-01-01')""",
            (
                document_id, f"{document_id}.pdf", f"/{document_id}.pdf",
                f"hash-{document_id}", document_date,
            ),
        )
    db.commit()


def _evidence(evidence_id, category, entity, *, assertion="present", day="2025-01-01"):
    return ClinicalEvidence(
        evidence_id=evidence_id, patient_id="P1", document_id="D1",
        category=category, normalized_entity=entity,
        source_text=f"{entity} documentata", observed_date=day,
        document_date=day, assertion=assertion,
    )


class _RelationLLM:
    is_available = True
    model = "relation-test-model"
    backend_type = "test"
    temperature = 0.1
    top_p = 0.9
    top_k = 40
    seed = 42
    max_output_tokens = 4096

    def __init__(self, delay=0.0):
        self.delay = delay
        self.calls = 0
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def generate_structured(self, prompt, _system, _schema, **_kwargs):
        with self._lock:
            self.calls += 1
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                time.sleep(self.delay)
            rows = json.loads(prompt.split("CANDIDATI:\n", 1)[1])
            return {
                "decisions": [{
                    "candidate_id": row["candidate_id"],
                    "linked": True,
                    "relation_type": "same_process",
                    "cluster_effect": "cohesive",
                    "weight": 0.91,
                    "rationale": "Stesso processo clinico",
                } for row in rows]
            }
        finally:
            with self._lock:
                self.active -= 1


def test_pipeline_policy_round_trip(tmp_path):
    path = tmp_path / "settings.json"
    policy = ClinicalPipelinePolicy(
        consensus_profile="research", local_window_days=12,
        semantic_top_k=7,
        lab=LabEvidencePolicy(
            significant_delta_within_range=True,
            default_relative_delta=0.4,
            analyzer_rules={"tsh": {"relative_delta": 0.3}},
        ),
    )
    save_pipeline_policy(policy, path)
    loaded = load_pipeline_policy(path)
    assert loaded == policy


def test_event_category_catalog_is_extensible_without_a_migration(tmp_path):
    db = DatabaseEngine(tmp_path / "categories.sqlite")
    init_database(db)
    repository = ClinicalPipelineRepository(db)
    repository.register_category(
        "patient_reported_outcome", "Esito riferito dal paziente",
        object_level="event", data={"color": "blue"},
    )
    categories = {
        item["category"]: item for item in repository.list_categories("event")
    }
    assert "adverse_event" in categories
    assert categories["patient_reported_outcome"]["data"] == {"color": "blue"}


def test_lab_delta_policy_can_create_an_atom_without_range_violation():
    previous = LabValue(
        patient_id="P1", document_id="D1", parameter_name="TSH",
        normalized_name="ormone_tireostimolante", value=1.0, unit="mIU/L",
        reference_low=0.2, reference_high=4.0, sample_date="2025-01-01",
        source_text="TSH 1.0 mIU/L",
    )
    current = LabValue(
        patient_id="P1", document_id="D2", parameter_name="TSH",
        normalized_name="ormone_tireostimolante", value=2.0, unit="mIU/L",
        reference_low=0.2, reference_high=4.0, sample_date="2025-01-20",
        source_text="TSH 2.0 mIU/L",
    )
    atoms = abnormal_lab_evidence(
        patient_id="P1", document_id="D2", document_date="2025-01-20",
        lab_values=[current], prior_values=[previous],
        policy=LabEvidencePolicy(
            significant_delta_within_range=True,
            default_relative_delta=0.5,
        ),
    )
    assert len(atoms) == 1
    assert atoms[0].typed_payload["inclusion_reasons"] == [
        "significant_delta_within_range"
    ]


def test_deterministic_loinc_resolution_is_persisted(tmp_path):
    db = DatabaseEngine(tmp_path / "pipeline.sqlite")
    init_database(db)
    _seed(db)
    repository = EvidenceRepository(db)
    item = ClinicalEvidence(
        evidence_id="E_LAB", patient_id="P1", document_id="D1",
        category="laboratory_finding", fact_type="laboratory_test",
        normalized_entity="TSH", concept_original="TSH",
        numeric_value=8.0, unit="mIU/L", source_text="TSH 8.0 mIU/L",
    )
    repository.insert_batch([item])
    stored = repository.get_by_patient("P1")[0]
    assert stored.terminology_system == "LOINC"
    assert stored.terminology_code == "3016-3"
    assert stored.mapping_status == "resolved_deterministic"
    assert stored.mapping_confidence == 0.9


def test_evidence_source_persistence_is_idempotent_for_duplicate_paths(tmp_path):
    db = DatabaseEngine(tmp_path / "source-idempotency.sqlite")
    init_database(db)
    _seed(db)
    evidence = _evidence("E_SOURCE", "diagnosis", "polmonite")
    evidence.source_page = 1
    evidence_repo = EvidenceRepository(db)
    evidence_repo.insert_batch([evidence])
    pipeline = ClinicalPipelineRepository(db)
    duplicate = EvidenceSourceReference(
        source_ref_id="SRC_DUPLICATE",
        evidence_id=evidence.evidence_id,
        document_id=evidence.document_id,
        source_page=1,
        passage=evidence.source_text,
        source_role="duplicate_source",
    )
    primary = EvidenceSourceReference(
        source_ref_id="SRC_LEGACY_PRIMARY",
        evidence_id=evidence.evidence_id,
        document_id=evidence.document_id,
        source_page=1,
        passage=evidence.source_text,
        source_role="primary",
        sentence_refs=[1],
    )

    pipeline.replace_evidence_sources(
        evidence.evidence_id, [duplicate, primary, duplicate]
    )
    pipeline.ensure_primary_source(evidence)

    rows = pipeline.get_evidence_sources(evidence.evidence_id)
    assert len(rows) == 1
    assert rows[0]["source_ref_id"] == "SRC_LEGACY_PRIMARY"
    assert rows[0]["source_role"] == "primary"


def test_graph_clusters_multimodal_evidence_and_blocks_contradictions():
    symptom = _evidence("E1", "symptom", "dispnea")
    imaging = _evidence("E2", "imaging_finding", "infiltrati polmonari")
    positive = _evidence("E3", "diagnosis", "polmonite")
    negative = _evidence(
        "E4", "diagnosis", "polmonite", assertion="absent"
    )
    result = EvidenceGraphBuilder(
        policy=ClinicalPipelinePolicy(cohesive_threshold=0.72)
    ).build("P1", [symptom, imaging, positive, negative])
    assert len(result.relations) <= 4 * 12 // 2
    assert any(
        relation.cluster_effect == "cannot_link"
        and {relation.source_evidence_id, relation.target_evidence_id}
        == {"E3", "E4"}
        for relation in result.relations
    )
    assert any(
        {"E1", "E2"} <= set(cluster.evidence_ids)
        for cluster in result.clusters
    )
    assert not any(
        {"E3", "E4"} <= set(cluster.evidence_ids)
        for cluster in result.clusters
    )


def test_graph_candidate_generation_is_bounded_not_quadratic():
    items = []
    start = date(2025, 1, 1)
    for index in range(180):
        day = (start + timedelta(days=index)).isoformat()
        item = _evidence(
            f"E{index}", "symptom", f"sintomo comune {index % 4}", day=day
        )
        item.document_id = "D1" if index % 2 == 0 else "D2"
        items.append(item)
    policy = ClinicalPipelinePolicy(
        semantic_top_k=6, local_window_days=10,
        longitudinal_window_days=90,
    )
    result = EvidenceGraphBuilder(policy=policy).build("P1", items)
    assert result.candidate_count <= len(items) * policy.semantic_top_k // 2
    assert result.candidate_count < len(items) * (len(items) - 1) // 10


def test_rejected_graph_relation_remains_authoritative_on_rebuild():
    first = _evidence("E1", "diagnosis", "polmonite")
    second = _evidence("E2", "diagnosis", "polmonite")
    initial = EvidenceGraphBuilder().build("P1", [first, second])
    relation = initial.relations[0]
    reviewed = [{
        "relation_id": relation.relation_id,
        "source_evidence_id": relation.source_evidence_id,
        "target_evidence_id": relation.target_evidence_id,
        "relation_type": relation.relation_type,
        "direction": relation.direction,
        "weight": relation.weight,
        "cluster_effect": relation.cluster_effect,
        "rationale": relation.rationale,
        "rule_features": relation.rule_features,
        "model_votes": relation.model_votes,
        "generation_method": relation.generation_method,
        "review_status": "rejected",
        "created_at": relation.created_at,
    }]
    rebuilt = EvidenceGraphBuilder().build(
        "P1", [first, second], reviewed_relations=reviewed
    )
    assert rebuilt.relations[0].review_status == "rejected"
    assert rebuilt.relations[0].weight == 0.0
    assert not any(
        {"E1", "E2"} <= set(cluster.evidence_ids)
        for cluster in rebuilt.clusters
    )


def test_graph_relation_batches_use_configured_parallel_slots():
    start = date(2025, 1, 1)
    items = [
        _evidence(
            f"E{index}", "symptom", "dispnea",
            day=(start + timedelta(days=index % 8)).isoformat(),
        )
        for index in range(96)
    ]
    policy = ClinicalPipelinePolicy(
        semantic_top_k=12, local_window_days=10,
        longitudinal_window_days=90,
    )
    parallel_llm = _RelationLLM(delay=0.01)
    parallel = EvidenceGraphBuilder(
        parallel_llm, policy=policy
    ).build("P1", items, num_workers=4)
    serial_llm = _RelationLLM()
    serial = EvidenceGraphBuilder(
        serial_llm, policy=policy
    ).build("P1", items, num_workers=1)

    assert parallel_llm.max_active > 1
    assert parallel.llm_calls == serial.llm_calls
    assert [
        (item.relation_id, item.relation_type, item.cluster_effect, item.weight)
        for item in parallel.relations
    ] == [
        (item.relation_id, item.relation_type, item.cluster_effect, item.weight)
        for item in serial.relations
    ]


def test_graph_hard_constraints_do_not_consume_an_llm_call():
    present = _evidence("E1", "diagnosis", "polmonite")
    absent = _evidence(
        "E2", "diagnosis", "polmonite", assertion="absent"
    )
    llm = _RelationLLM()
    result = EvidenceGraphBuilder(llm).build("P1", [present, absent])

    assert llm.calls == 0
    assert result.llm_calls == 0
    assert result.auto_resolved_count == 1
    assert result.relations[0].cluster_effect == "cannot_link"


def test_graph_relation_cache_resumes_without_repeating_llm_calls(tmp_path):
    db = DatabaseEngine(tmp_path / "relation-cache.sqlite")
    init_database(db)
    _seed(db)
    evidence_repo = EvidenceRepository(db)
    evidence_repo.insert_batch([
        _evidence("E1", "symptom", "dispnea"),
        _evidence("E2", "imaging_finding", "infiltrato polmonare"),
    ])
    items = evidence_repo.get_by_patient("P1")
    cache = ClinicalPipelineRepository(db)

    first_llm = _RelationLLM()
    first = EvidenceGraphBuilder(first_llm).build(
        "P1", items, cache_repository=cache
    )
    second_llm = _RelationLLM()
    second = EvidenceGraphBuilder(second_llm).build(
        "P1", items, cache_repository=cache
    )

    assert first.llm_calls == 1
    assert second.llm_calls == 0
    assert second.cache_hits == 1
    assert second.relations[0].relation_type == first.relations[0].relation_type
    assert db.get_table_count("evidence_relation_adjudication_cache") == 1


def test_graph_cancellation_is_checked_between_parallel_batches():
    items = [
        _evidence(f"E{index}", "symptom", "dispnea")
        for index in range(96)
    ]
    llm = _RelationLLM(delay=0.02)
    cancelled = threading.Event()

    def progress(completed, total, _cached, _automatic):
        if 0 < completed < total:
            cancelled.set()

    with pytest.raises(EvidenceGraphCancelled):
        EvidenceGraphBuilder(llm).build(
            "P1", items, num_workers=4,
            cancel_check=cancelled.is_set,
            progress_callback=progress,
        )
    assert llm.max_active > 1


def test_atomic_gold_annotations_are_source_first_and_persisted(tmp_path):
    db = DatabaseEngine(tmp_path / "gold.sqlite")
    init_database(db)
    _seed(db)
    repository = GoldSetRepository(db)
    repository.configure_case(
        "P1", included=True, split="pilot", reviewer_a_id="A",
        reviewer_b_id="B", adjudicator_id="C",
    )
    annotation = GoldAtomicAnnotation(
        patient_id="P1", document_id="D1", reviewer_slot="reviewer_a",
        reviewer_id="A", fact_type="symptom", concept_original="dispnea",
        polarity="present", source_page=2,
        source_text="La paziente riferisce dispnea.",
    )
    repository.save_atomic_annotation(annotation)
    loaded = repository.list_atomic_annotations("P1", "reviewer_a")
    assert len(loaded) == 1
    assert loaded[0].source_text == annotation.source_text
    assert loaded[0].fact_type == "symptom"


def test_duplicate_groups_retain_exact_sources_and_pending_candidates(tmp_path):
    db = DatabaseEngine(tmp_path / "duplicates.sqlite")
    init_database(db)
    _seed(db)
    evidence_repo = EvidenceRepository(db)
    first = _evidence("E1", "diagnosis", "melanoma")
    copied = _evidence("E2", "diagnosis", "melanoma")
    copied.document_id = "D2"
    evidence_repo.insert_batch([first, copied])
    first.data["duplicate_source_evidence_ids"] = ["E2"]
    pipeline = ClinicalPipelineRepository(db)
    pending = pipeline.replace_duplicate_groups(
        "P1", [first], [("E1", "E2", 0.87)]
    )
    assert len(pending) == 1
    rows = db.execute(
        "SELECT review_status FROM evidence_duplicate_groups WHERE patient_id='P1'"
    ).fetchall()
    assert {row["review_status"] for row in rows} == {"auto", "pending"}


def test_accepted_near_duplicate_is_applied_and_does_not_reenter_queue(tmp_path):
    db = DatabaseEngine(tmp_path / "reviewed-duplicates.sqlite")
    init_database(db)
    _seed(db)
    evidence_repo = EvidenceRepository(db)
    first = _evidence("E1", "diagnosis", "melanoma")
    second = _evidence("E2", "diagnosis", "melanoma cutaneo")
    second.document_id = "D2"
    evidence_repo.insert_batch([first, second])
    pipeline = ClinicalPipelineRepository(db)
    pending = pipeline.replace_duplicate_groups(
        "P1", [first, second], [("E1", "E2", 0.91)]
    )
    pipeline.review_duplicate_group(pending[0], "accepted")
    pending_again = pipeline.replace_duplicate_groups(
        "P1", [first, second], [("E1", "E2", 0.91)]
    )
    assert pending_again == []
    collapsed = _apply_reviewed_duplicate_groups(
        [first, second], pipeline.list_accepted_duplicate_groups("P1")
    )
    assert len(collapsed) == 1
    assert collapsed[0].data["duplicate_source_evidence_ids"] == ["E2"]


def test_excluded_projection_keeps_reason_and_exact_source():
    item = _evidence("E_ADMIN", "other", "prossimo appuntamento")
    item.clinical_relevance = "excluded_administrative"
    item.source_text = "Prossimo appuntamento il 3 marzo."
    item.data = {"sentence_refs": [2], "registry_role": "administrative"}
    excluded = _excluded_record(item, "scheduling")
    assert excluded.reason_code == "scheduling"
    assert excluded.source_text == item.source_text
    assert excluded.sentence_refs == [2]


def test_incremental_evidence_upsert_preserves_reviewed_event_links(tmp_path):
    db = DatabaseEngine(tmp_path / "incremental.sqlite")
    init_database(db)
    _seed(db)
    evidence_repo = EvidenceRepository(db)
    atom = _evidence("E_STABLE", "diagnosis", "melanoma")
    evidence_repo.insert_batch([atom])
    registry = ClinicalRegistryRepository(db)
    event = ClinicalEvent(
        event_id="EVT_LOCKED", patient_id="P1", category="diagnosis",
        canonical_entity="melanoma", summary_short="Melanoma",
        review_status="corrected",
    )
    registry.save_event(event, [EventEvidenceLink(
        event_id=event.event_id, evidence_id=atom.evidence_id,
    )])
    original_created_at = atom.created_at
    refreshed = _evidence("E_STABLE", "diagnosis", "melanoma cutaneo")
    refreshed.created_at = "2099-01-01T00:00:00"
    evidence_repo.replace_document_method(
        "D1", refreshed.extraction_method, [refreshed]
    )
    detail = registry.get_event_detail(event.event_id)
    assert [row["evidence_id"] for row in detail["evidence"]] == ["E_STABLE"]
    stored = evidence_repo.get_by_document("D1")[0]
    assert stored.normalized_entity == "melanoma cutaneo"
    assert stored.created_at == original_created_at


def test_atomic_validation_problem_triggers_specialized_retry_and_typed_payload():
    class RepairingLlm:
        model = "fake"
        context_length = 4096
        max_output_tokens = 2048

        def __init__(self):
            self.calls = 0

        def generate_structured(self, prompt, system, schema, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return {"evidence": [{
                    "fact_type": "radiology_finding", "concept": "nodulo",
                    "polarity": "present", "source_refs": [99],
                    "campo_inventato": "forza retry mirato",
                }]}
            return {"evidence": [{
                "fact_type": "radiology_finding", "concept": "nodulo",
                "polarity": "present", "source_refs": [1],
                "payload": {
                    "modality": "TC", "body_region": "polmone",
                    "measurement": "8 mm",
                },
            }]}

    llm = RepairingLlm()
    extractor = AtomicEvidenceExtractor(
        llm,
        policy=ClinicalPipelinePolicy(
            adaptive_specialized_retry=True, max_specialized_retries=1
        ),
    )
    result = extractor.extract_document(
        patient_id="P1", document_id="D1", document_type="radiologia",
        document_date="2025-01-01", text="TC: nodulo polmonare di 8 mm.",
    )
    assert llm.calls == 2
    assert len(result) == 1
    assert result[0].typed_payload["radiology_finding"]["measurement"] == "8 mm"
    assert extractor.last_extraction_metrics()["validation_retries"] == 1


def test_manual_split_and_merge_preserve_event_identities_and_sources(tmp_path):
    db = DatabaseEngine(tmp_path / "manual.sqlite")
    init_database(db)
    _seed(db)
    evidence_repo = EvidenceRepository(db)
    atoms = [
        _evidence("E1", "diagnosis", "polmonite"),
        _evidence("E2", "imaging_finding", "infiltrati"),
        _evidence("E3", "symptom", "dispnea"),
    ]
    evidence_repo.insert_batch(atoms)
    repository = ClinicalRegistryRepository(db)
    first = ClinicalEvent(
        event_id="EVT_1", patient_id="P1", category="diagnosis",
        canonical_entity="polmonite", summary_short="Polmonite",
    )
    second = ClinicalEvent(
        event_id="EVT_2", patient_id="P1", category="symptom",
        canonical_entity="dispnea", summary_short="Dispnea",
    )
    repository.save_event(first, [
        EventEvidenceLink(event_id="EVT_1", evidence_id="E1"),
        EventEvidenceLink(event_id="EVT_1", evidence_id="E2"),
    ])
    repository.save_event(second, [
        EventEvidenceLink(event_id="EVT_2", evidence_id="E3"),
    ])
    new_id = repository.split_event_manual(
        "P1", "EVT_1", ["E2"], original_summary="Polmonite clinica",
        new_summary="Reperto radiologico autonomo",
    )
    assert repository.get_event_detail(new_id)["event"]["review_status"] == "corrected"
    assert {
        item["evidence_id"] for item in repository.get_event_detail(new_id)["evidence"]
    } == {"E2"}
    repository.merge_events_manual(
        "P1", "EVT_1", "EVT_2", summary_short="Polmonite con dispnea"
    )
    assert repository.get_event_detail("EVT_2")["event"]["review_status"] == "rejected"
    assert {
        item["evidence_id"] for item in repository.get_event_detail("EVT_1")["evidence"]
    } == {"E1", "E3"}
    claims = repository.get_event_detail("EVT_1")["claims"]
    assert len(claims) == 1
    assert {
        source["source_id"] for source in claims[0]["sources"]
    } == {"E1", "E3"}


def test_hypothesis_discovery_is_separate_bounded_and_review_preserving(tmp_path):
    class PlausibilityLlm:
        is_available = True
        max_output_tokens = 2048

        def generate_structured(self, prompt, system, schema, **kwargs):
            rows = json.loads(prompt.split("CANDIDATI:\n", 1)[1])
            return {"hypotheses": [{
                "candidate_id": rows[0]["candidate_id"],
                "plausible": True,
                "hypothesis_type": "possibile relazione terapeutica",
                "strength": "weak",
                "known_mechanism": False,
                "rationale": "Sequenza compatibile, non documentata.",
                "confounders": ["patologia concomitante"],
                "survival_score": 0.42,
            }]}

    db = DatabaseEngine(tmp_path / "hypotheses.sqlite")
    init_database(db)
    _seed(db)
    registry = ClinicalRegistryRepository(db)
    registry.save_event(ClinicalEvent(
        event_id="EVT_T", patient_id="P1", category="medication",
        canonical_entity="nivolumab", summary_short="Avvio nivolumab",
        first_evidence_date="2025-01-01",
    ))
    registry.save_event(ClinicalEvent(
        event_id="EVT_A", patient_id="P1", category="adverse_event",
        canonical_entity="rash", summary_short="Comparsa di rash",
        first_evidence_date="2025-01-03",
    ))
    pipeline = ClinicalPipelineRepository(db)
    discovery = HypothesisDiscovery(registry, pipeline, PlausibilityLlm())
    result = discovery.run("P1", top_k=2)
    assert result == {
        "candidate_count": 1, "hypothesis_count": 1, "llm_calls": 1,
    }
    item = pipeline.list_hypotheses("P1")[0]
    assert item["status"] == "needs_review"
    pipeline.review_hypothesis(item["hypothesis_id"], "accepted")
    discovery.run("P1", top_k=2)
    assert pipeline.list_hypotheses("P1")[0]["status"] == "accepted"
    assert pipeline.list_hypotheses("P1", accepted_only=True)


def test_method_and_tracer_boilerplate_are_archived_before_llm_extraction():
    class CaptureLlm:
        model = "fake"
        context_length = 4096
        max_output_tokens = 2048

        def __init__(self):
            self.prompt = ""

        def generate_structured(self, prompt, system, schema, **kwargs):
            self.prompt = prompt
            return {"evidence": [{
                "fact_type": "radiology_finding",
                "concept": "nodulo polmonare", "polarity": "present",
                "source_refs": [1],
            }]}

    llm = CaptureLlm()
    result = AtomicEvidenceExtractor(llm).extract_document(
        patient_id="P1", document_id="D1", document_type="PET",
        document_date="2025-01-01",
        text=(
            "Prestazioni eseguite e indicazione di dose secondo l'art. 1.\n"
            "18F-FDG (somministrazione e.v. di 18F-FDG).\n"
            "Si evidenzia nodulo polmonare."
        ),
    )
    assert "Prestazioni eseguite" not in llm.prompt
    assert "18F-FDG" not in llm.prompt
    excluded = [item for item in result if item.clinical_relevance.startswith("excluded_")]
    clinical = [item for item in result if item.clinical_relevance == "accepted_clinical"]
    assert len(excluded) == 2
    assert len(clinical) == 1
    assert all(
        item.extraction_method == "deterministic_nonclinical"
        for item in excluded
    )
