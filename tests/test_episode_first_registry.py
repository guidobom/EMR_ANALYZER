"""Episode-first relevance, ownership and provenance tests."""

import unittest

from PyQt5.QtWidgets import QApplication

from emr_analyzer.clinical.consolidation import ConsolidatedBundle
from emr_analyzer.clinical.episode_assembler import (
    attach_contextual_evidence,
    build_event_relations,
)
from emr_analyzer.clinical.episode_synthesis import (
    ClinicalEpisodeSynthesizer,
    validate_episode_decision,
)
from emr_analyzer.clinical.event_dedup import deduplicate_bundles
from emr_analyzer.clinical.evidence_relevance import (
    ADMINISTRATIVE_OR_METHODOLOGICAL,
    CONTEXTUAL,
    PRIMARY,
    classify_evidence,
    is_administrative_mapping,
    partition_evidence,
)
from emr_analyzer.clinical.query_service import format_event_detail
from emr_analyzer.gui.pdf_viewer import highlighted_normalized_html
from emr_analyzer.gui.event_quick_view import EventQuickViewDialog
from emr_analyzer.models.clinical_evidence import ClinicalEvidence
from emr_analyzer.models.clinical_registry import (
    ClinicalEpisode,
    ClinicalEvent,
    EventEvidenceLink,
)


def atom(
    evidence_id,
    entity,
    *,
    category="diagnosis",
    source=None,
    assertion="present",
    certainty="confirmed",
    document_id="D1",
    date="2025-01-10",
):
    return ClinicalEvidence(
        evidence_id=evidence_id,
        patient_id="P001",
        document_id=document_id,
        category=category,
        normalized_entity=entity,
        source_text=source or entity,
        assertion=assertion,
        certainty=certainty,
        observed_date=date,
        document_date=date,
        confidence=0.9,
    )


def draft(event_id, item, *, category=None):
    category = category or item.category
    episode = ClinicalEpisode(
        episode_id=f"EPI_{event_id}", patient_id="P001",
        category=category, canonical_entity=item.normalized_entity,
        onset_date=item.observed_date,
    )
    event = ClinicalEvent(
        event_id=event_id, episode_id=episode.episode_id,
        patient_id="P001", category=category,
        canonical_entity=item.normalized_entity,
        summary_short=item.normalized_entity,
        first_evidence_date=item.observed_date,
        structured_data={"evidence_ids": [item.evidence_id]},
    )
    link = EventEvidenceLink(
        link_id=f"LNK_{event_id}_{item.evidence_id}",
        event_id=event_id, evidence_id=item.evidence_id,
    )
    return ConsolidatedBundle(episode, event, [link], [])


class EvidenceRelevanceTest(unittest.TestCase):
    def test_methodological_atoms_are_retained_but_not_registry_eligible(self):
        methodological = atom(
            "E1", "somministrazione e.v. di 18F-FDG", category="procedure"
        )
        clinical = atom(
            "E2", "ipercaptazione patologica 18F-FDG",
            category="imaging_finding",
            source=(
                "somministrazione e.v. di 18F-FDG; "
                "ipercaptazione patologica in sede glutea"
            ),
        )
        self.assertEqual(
            classify_evidence(methodological).role,
            ADMINISTRATIVE_OR_METHODOLOGICAL,
        )
        self.assertEqual(classify_evidence(clinical).role, PRIMARY)
        primary, contextual, excluded = partition_evidence(
            [methodological, clinical]
        )
        self.assertEqual(primary, [clinical])
        self.assertEqual(contextual, [])
        self.assertEqual(excluded, [methodological])
        self.assertEqual(
            methodological.data["registry_role"],
            ADMINISTRATIVE_OR_METHODOLOGICAL,
        )

    def test_negative_atom_is_context_not_standalone_anchor(self):
        negative = atom(
            "E1", "versamento pleurico", category="imaging_finding",
            source="Non si evidenzia versamento pleurico", assertion="absent",
        )
        self.assertEqual(classify_evidence(negative).role, CONTEXTUAL)

    def test_atom_named_as_exam_is_excluded_when_source_is_only_appointment(self):
        scheduled_pet = atom(
            "E1", "PET FDG", category="procedure",
            source=(
                "Prossimi appuntamenti:\n"
                "- 26/09 Medicina Nucleare per PET ore 09:00"
            ),
        )
        disposition = classify_evidence(scheduled_pet)
        self.assertEqual(
            disposition.role, ADMINISTRATIVE_OR_METHODOLOGICAL
        )
        self.assertEqual(disposition.reason, "scheduling")
        self.assertTrue(is_administrative_mapping({
            "normalized_entity": scheduled_pet.normalized_entity,
            "source_text": scheduled_pet.source_text,
        }))


class EpisodeOwnershipTest(unittest.TestCase):
    def test_contextual_atom_attaches_to_matching_episode(self):
        positive = atom(
            "E1", "versamento pleurico", category="imaging_finding"
        )
        negative = atom(
            "E2", "versamento pleurico", category="imaging_finding",
            source="Non si evidenzia versamento pleurico", assertion="absent",
        )
        bundle = draft("EVT1", positive)
        linked, unassigned = attach_contextual_evidence(
            [bundle], [negative], evidence_by_id={
                positive.evidence_id: positive,
                negative.evidence_id: negative,
            },
        )
        self.assertEqual((linked, unassigned), (1, 0))
        contextual_link = next(
            link for link in bundle.links if link.evidence_id == "E2"
        )
        self.assertEqual(contextual_link.relation, "excluded")
        self.assertFalse(contextual_link.included_in_summary)

    def test_unrelated_normal_atom_does_not_create_or_attach_event(self):
        diagnosis = atom("E1", "melanoma", category="diagnosis")
        normal = atom(
            "E2", "creatinina", category="laboratory_finding",
            source="Creatinina nei limiti", assertion="present",
        )
        bundle = draft("EVT1", diagnosis)
        linked, unassigned = attach_contextual_evidence(
            [bundle], [normal], evidence_by_id={"E1": diagnosis, "E2": normal}
        )
        self.assertEqual((linked, unassigned), (0, 1))
        self.assertEqual(len(bundle.links), 1)

    def test_autonomous_derived_episode_is_linked_by_shared_evidence(self):
        evidence = atom("E1", "dispnea", category="symptom")
        symptom = draft("EVT_SYMPTOM", evidence)
        syndrome = draft(
            "EVT_SYNDROME", evidence, category="clinical_syndrome"
        )
        relations = build_event_relations("P001", [symptom, syndrome])
        relation_types = {relation.relation_type for relation in relations}
        self.assertIn("aggregates", relation_types)
        self.assertIn("component_of", relation_types)

    def test_dedup_preserves_contextual_relation_and_summary_exclusion(self):
        first_atom = atom(
            "E1", "versamento pleurico", category="imaging_finding",
            document_id="D1",
        )
        second_atom = atom(
            "E2", "versamento pleurico", category="imaging_finding",
            document_id="D2",
        )
        negative = atom(
            "E3", "versamento pleurico", category="imaging_finding",
            source="Non si evidenzia versamento pleurico",
            assertion="absent", document_id="D1",
        )
        merged_away = draft("EVT_Z", first_atom)
        merged_away.event.confidence = 0.2
        survivor = draft("EVT_A", second_atom)
        survivor.event.confidence = 0.95
        evidence_by_id = {
            item.evidence_id: item
            for item in (first_atom, second_atom, negative)
        }
        attach_contextual_evidence(
            [merged_away, survivor], [negative],
            evidence_by_id=evidence_by_id,
        )
        final, count = deduplicate_bundles(
            [merged_away, survivor],
            evidence_by_id=evidence_by_id,
            persisted_review_status={},
        )
        self.assertEqual(count, 1)
        contextual_link = next(
            link for link in final[0].links if link.evidence_id == "E3"
        )
        self.assertEqual(contextual_link.relation, "excluded")
        self.assertFalse(contextual_link.included_in_summary)


class ProvenanceRenderingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_normalized_text_highlight_escapes_source_markup(self):
        rendered = highlighted_normalized_html(
            "TSH\n  < 0,01 e dispnea", ["TSH < 0,01"]
        )
        self.assertIn("<mark", rendered)
        self.assertIn("TSH\n  &lt; 0,01", rendered)
        self.assertNotIn("< 0,01", rendered)

    def test_normalized_text_highlights_every_occurrence(self):
        rendered = highlighted_normalized_html(
            "Dispnea iniziale. Dispnea risolta.", ["dispnea"]
        )
        self.assertEqual(rendered.count("<mark"), 2)

    def test_llm_context_omits_administrative_source_but_keeps_clinical(self):
        rendered = format_event_detail({
            "event": {
                "event_id": "EVT_1", "first_evidence_date": "2025-01-01",
                "category": "diagnosis", "status": "active",
                "certainty": "confirmed", "summary_short": "Melanoma",
            },
            "evidence": [{
                "evidence_id": "E_ADMIN", "document_id": "D1",
                "normalized_entity": "PET FDG",
                "source_text": "Prossimi appuntamenti: PET il 26/09",
            }, {
                "evidence_id": "E_CLIN", "document_id": "D2",
                "normalized_entity": "melanoma",
                "source_text": "Diagnosi istologica di melanoma",
            }],
            "updates": [],
        })
        self.assertNotIn("Prossimi appuntamenti", rendered)
        self.assertIn("Diagnosi istologica di melanoma", rendered)

    def test_event_quick_view_builds_report_and_atomic_evidence_tree(self):
        class EmptyDocumentRepository:
            @staticmethod
            def get_by_id(_document_id):
                return None

        dialog = EventQuickViewDialog({
            "event": {
                "event_id": "EVT_1", "summary_short": "Dispnea < grado 2",
                "first_evidence_date": "2025-01-01", "status": "active",
                "certainty": "confirmed",
            },
            "episode": {"recurrence_index": 1},
            "evidence": [{
                "evidence_id": "E1", "document_id": "D1",
                "normalized_entity": "dispnea", "source_text": "Dispnea",
                "source_page": 2, "relation": "supports",
                "included_in_summary": 1,
            }],
            "relations": [],
        }, {"document_repo": EmptyDocumentRepository()})
        try:
            self.assertEqual(dialog._evidence_tree.topLevelItemCount(), 1)
            root = dialog._evidence_tree.topLevelItem(0)
            self.assertEqual(root.childCount(), 1)
            self.assertIn("Dispnea < grado 2", dialog._summary.toPlainText())
            self.assertEqual(
                dialog._preview._text_toggle.text(),
                "Testo clinico normalizzato",
            )
        finally:
            dialog.close()


class _EpisodeLlm:
    is_available = True
    model = "test-model"
    max_output_tokens = 4096

    def __init__(self, result):
        self.result = result
        self.calls = 0

    def generate_structured(self, _prompt, _system, _schema, **_kwargs):
        self.calls += 1
        return self.result


class LlmEpisodeSynthesisTest(unittest.TestCase):
    def test_absorbs_observation_and_links_autonomous_therapy(self):
        symptom_atom = atom(
            "E1", "dispnea", category="symptom",
            source="Insorgenza di dispnea durante immunoterapia",
        )
        sign_atom = atom(
            "E2", "SpO2 ridotta", category="clinical_sign",
            source="Dispnea con SpO2 ridotta durante immunoterapia",
        )
        therapy_atom = atom(
            "E3", "pembrolizumab", category="medication",
            source="Durante pembrolizumab insorge dispnea",
        )
        bundles = [
            draft("EVT_SYM", symptom_atom),
            draft("EVT_SIGN", sign_atom),
            draft("EVT_MED", therapy_atom),
        ]
        result = {"drafts": [{
            "anchor_event_id": "EVT_SYM",
            "absorbed_event_ids": ["EVT_SIGN"],
            "related_event_ids": ["EVT_MED"],
            "problem_label": "episodio respiratorio",
            "category": "clinical_syndrome",
            "claims": [{
                "text": "Episodio di dispnea con riduzione della SpO2",
                "evidence_ids": ["E1", "E2"],
            }],
        }]}
        llm = _EpisodeLlm(result)
        evidence_by_id = {
            item.evidence_id: item
            for item in (symptom_atom, sign_atom, therapy_atom)
        }
        final, relations, stats = ClinicalEpisodeSynthesizer(llm).synthesize(
            "P001", bundles, evidence_by_id=evidence_by_id
        )
        self.assertEqual(llm.calls, 1)
        self.assertEqual(stats.episodes_absorbed, 1)
        self.assertEqual({bundle.event.event_id for bundle in final}, {
            "EVT_SYM", "EVT_MED",
        })
        episode = next(
            bundle for bundle in final if bundle.event.event_id == "EVT_SYM"
        )
        self.assertEqual(episode.event.category, "clinical_syndrome")
        self.assertEqual({link.evidence_id for link in episode.links}, {"E1", "E2"})
        self.assertEqual(relations[0].relation_type, "related_episode")

    def test_validation_never_absorbs_autonomous_procedure(self):
        symptom_atom = atom(
            "E1", "dispnea", category="symptom", source="Dispnea"
        )
        procedure_atom = atom(
            "E2", "broncoscopia", category="procedure",
            source="Broncoscopia per dispnea",
        )
        group = [
            draft("EVT_SYM", symptom_atom),
            draft("EVT_PROC", procedure_atom),
        ]
        decision = validate_episode_decision(
            {"drafts": [{
                "anchor_event_id": "EVT_SYM",
                "absorbed_event_ids": ["EVT_PROC"],
                "related_event_ids": [],
                "problem_label": "dispnea",
                "category": "symptom",
                "claims": [{"text": "Dispnea", "evidence_ids": ["E1", "E2"]}],
            }]},
            group,
            evidence_by_id={"E1": symptom_atom, "E2": procedure_atom},
        )
        self.assertEqual(decision, {"drafts": []})

    def test_claim_cannot_cite_evidence_from_unabsorbed_event(self):
        symptom_atom = atom("E1", "dispnea", category="symptom")
        sign_atom = atom("E2", "SpO2 ridotta", category="clinical_sign")
        unrelated_atom = atom(
            "E3", "pembrolizumab", category="medication"
        )
        group = [
            draft("EVT_SYM", symptom_atom),
            draft("EVT_SIGN", sign_atom),
            draft("EVT_MED", unrelated_atom),
        ]
        decision = validate_episode_decision(
            {"drafts": [{
                "anchor_event_id": "EVT_SYM",
                "absorbed_event_ids": ["EVT_SIGN"],
                "related_event_ids": ["EVT_MED"],
                "problem_label": "episodio respiratorio",
                "category": "clinical_syndrome",
                "claims": [{
                    "text": "Dispnea attribuita a pembrolizumab",
                    "evidence_ids": ["E1", "E2", "E3"],
                }],
            }]},
            group,
            evidence_by_id={
                "E1": symptom_atom, "E2": sign_atom,
                "E3": unrelated_atom,
            },
        )
        self.assertEqual(decision, {"drafts": []})

    def test_batches_independent_candidate_groups_in_one_llm_call(self):
        respiratory = [
            atom("E1", "dispnea", category="symptom"),
            atom("E2", "SpO2 ridotta", category="clinical_sign"),
        ]
        thyroid = [
            atom("E3", "TSH elevato", category="laboratory_finding"),
            atom("E4", "ipotiroidismo", category="diagnosis"),
        ]
        bundles = [
            draft(f"EVT{index}", item)
            for index, item in enumerate(respiratory + thyroid, start=1)
        ]
        llm = _EpisodeLlm({"drafts": []})
        _, _, stats = ClinicalEpisodeSynthesizer(llm).synthesize(
            "P001",
            bundles,
            evidence_by_id={
                item.evidence_id: item for item in respiratory + thyroid
            },
        )
        self.assertEqual(stats.candidate_groups, 2)
        self.assertEqual(stats.llm_calls, 1)
        self.assertEqual(llm.calls, 1)

    def test_noop_decision_is_cached_on_a_surviving_event(self):
        symptom_atom = atom("E1", "dispnea", category="symptom")
        sign_atom = atom("E2", "SpO2 ridotta", category="clinical_sign")
        evidence_by_id = {"E1": symptom_atom, "E2": sign_atom}
        first_bundles = [
            draft("EVT_SYM", symptom_atom),
            draft("EVT_SIGN", sign_atom),
        ]
        first_llm = _EpisodeLlm({"drafts": []})
        first, _, first_stats = ClinicalEpisodeSynthesizer(
            first_llm
        ).synthesize(
            "P001", first_bundles, evidence_by_id=evidence_by_id
        )
        self.assertEqual(first_stats.llm_calls, 1)

        rebuilt = [
            draft("EVT_SYM", symptom_atom),
            draft("EVT_SIGN", sign_atom),
        ]
        second_llm = _EpisodeLlm({"drafts": []})
        _, _, second_stats = ClinicalEpisodeSynthesizer(
            second_llm,
            existing_events=[bundle.event for bundle in first],
        ).synthesize(
            "P001", rebuilt, evidence_by_id=evidence_by_id
        )
        self.assertEqual(second_llm.calls, 0)
        self.assertEqual(second_stats.cached_groups, 1)

    def test_contextual_evidence_enters_summary_only_when_llm_cites_it(self):
        symptom_atom = atom("E1", "dispnea", category="symptom")
        sign_atom = atom("E2", "SpO2 ridotta", category="clinical_sign")
        contextual = atom(
            "E3", "dispnea", category="symptom",
            source="Al controllo dispnea assente", assertion="absent",
        )
        bundles = [
            draft("EVT_SYM", symptom_atom),
            draft("EVT_SIGN", sign_atom),
        ]
        evidence_by_id = {
            item.evidence_id: item
            for item in (symptom_atom, sign_atom, contextual)
        }
        attach_contextual_evidence(
            bundles, [contextual], evidence_by_id=evidence_by_id
        )
        llm = _EpisodeLlm({"drafts": [{
            "anchor_event_id": "EVT_SYM",
            "absorbed_event_ids": ["EVT_SIGN"],
            "related_event_ids": [],
            "problem_label": "episodio respiratorio",
            "category": "clinical_syndrome",
            "claims": [{
                "text": "Dispnea con desaturazione, successivamente assente",
                "evidence_ids": ["E1", "E2", "E3"],
            }],
        }]})
        final, _, _ = ClinicalEpisodeSynthesizer(llm).synthesize(
            "P001", bundles, evidence_by_id=evidence_by_id
        )
        episode = final[0]
        contextual_link = next(
            link for link in episode.links if link.evidence_id == "E3"
        )
        self.assertTrue(contextual_link.included_in_summary)


if __name__ == "__main__":
    unittest.main()
