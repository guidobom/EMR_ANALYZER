"""Tests for deterministic cleanup and cross-event deduplication."""

import unittest

from emr_analyzer.clinical.consolidation import ConsolidatedBundle
from emr_analyzer.clinical.event_dedup import (
    atomic_event_summary,
    clean_bundle_summaries,
    clean_event_summary,
    compress_measurement_table,
    deduplicate_bundles,
    is_administrative_evidence,
    strip_header_boilerplate,
)
from emr_analyzer.models.clinical_evidence import ClinicalEvidence
from emr_analyzer.models.clinical_registry import (
    ClinicalEpisode,
    ClinicalEvent,
    EventEvidenceLink,
)


def evidence(
    eid,
    entity,
    *,
    category="diagnosis",
    date="2025-01-01",
    source="Reperto clinico documentato.",
    value_text=None,
    numeric_value=None,
    unit=None,
    assertion="present",
):
    return ClinicalEvidence(
        evidence_id=eid,
        patient_id="P001",
        document_id="D1",
        category=category,
        normalized_entity=entity,
        source_text=source,
        observed_date=date,
        document_date=date,
        date_precision="day",
        value_text=value_text,
        numeric_value=numeric_value,
        unit=unit,
        confidence=0.8,
        assertion=assertion,
    )


def bundle(
    event_id,
    *,
    category="diagnosis",
    entity="diabete_tipo_2",
    summary="Paziente affetto da diabete di tipo 2 in terapia orale.",
    date="2025-01-01",
    assertion="present",
    confidence=0.8,
    review_status="auto",
    status="active",
    date_end=None,
    evidence_items=None,
):
    evidence_items = evidence_items or [evidence(f"{event_id}_E1", entity)]
    event = ClinicalEvent(
        event_id=event_id,
        patient_id="P001",
        episode_id=f"EPI_{event_id}",
        category=category,
        canonical_entity=entity,
        summary_short=summary,
        summary_detail=summary,
        first_evidence_date=date,
        first_documented_date=date,
        assertion=assertion,
        confidence=confidence,
        review_status=review_status,
        status=status,
        date_end=date_end,
        structured_data={"evidence_ids": [e.evidence_id for e in evidence_items]},
    )
    episode = ClinicalEpisode(
        episode_id=f"EPI_{event_id}",
        patient_id="P001",
        category=category,
        canonical_entity=entity,
        onset_date=date,
        first_documented_date=date,
    )
    links = [
        EventEvidenceLink(
            event_id=event_id,
            evidence_id=e.evidence_id,
            relation="supports",
            link_id=f"LNK_{event_id}_{e.evidence_id}",
        )
        for e in evidence_items
    ]
    return ConsolidatedBundle(episode=episode, event=event, links=links, updates=[])


def evidence_map(*items):
    return {item.evidence_id: item for item in items}


class TestIsAdministrativeEvidence(unittest.TestCase):
    def test_drops_tracer_administration(self):
        item = evidence(
            "E1", "18F-FDG somministrazione e.v. di 18F-FDG",
            category="procedure", source="di 18F-FDG (somministrazione e.v. di 18F-FDG)",
        )
        self.assertTrue(is_administrative_evidence(item))

    def test_drops_report_headers(self):
        self.assertTrue(is_administrative_evidence(
            evidence("E1", "Quesito Clinico", category="other")
        ))
        self.assertTrue(is_administrative_evidence(
            evidence("E1", "Paziente: [PAZIENTE]", category="other")
        ))
        self.assertTrue(is_administrative_evidence(
            evidence(
                "E1",
                "Prestazioni eseguite e indicazione di dose secondo l'art.161",
                category="other",
            )
        ))
        self.assertTrue(is_administrative_evidence(
            evidence("E1", "Classe dose", category="other")
        ))
        self.assertTrue(is_administrative_evidence(
            evidence("E1", "Acc Number: 123456", category="other")
        ))

    def test_drops_appointments(self):
        self.assertTrue(is_administrative_evidence(
            evidence("E1", "Appuntamento di controllo", category="follow_up")
        ))

    def test_keeps_real_finding_with_bloated_source(self):
        item = evidence(
            "E1",
            "Frazione Eiezione 4 Ch Simpson MonoPlano",
            category="imaging_finding",
            source=(
                "4CH B-Mode [35 - 75] Frazione Eiezione 55 %; "
                "Simpson MonoPlano B-Mode [35 - 75] Frazione Eiezione 55 %; "
                "TDI S 8 cm/s; Doppler Velocità E/A 1.2"
            ),
        )
        self.assertFalse(is_administrative_evidence(item))

    def test_keeps_fdg_finding_without_administration_verb(self):
        self.assertFalse(is_administrative_evidence(
            evidence("E1", "Ipercaptazione 18F-FDG", category="imaging_finding")
        ))


class TestStripHeaderBoilerplate(unittest.TestCase):
    def test_removes_header_block_before_clinical_claim(self):
        cleaned = strip_header_boilerplate(
            "Paziente: [PAZIENTE] Data Esame: 24/02/2025 Quesito Clinico: "
            "melanoma. Presenza di dispnea da sforzo."
        )
        self.assertIn("Presenza di dispnea da sforzo", cleaned)
        self.assertNotIn("Paziente", cleaned)
        self.assertNotIn("Quesito Clinico", cleaned)

    def test_removes_exam_execution_boilerplate(self):
        cleaned = strip_header_boilerplate(
            "Prestazioni eseguite e indicazione di dose secondo l'art.161 del "
            "D.Lgs 101/2020; esame confrontato con precedente. Reperto stabile."
        )
        self.assertIn("Reperto stabile", cleaned)
        self.assertNotIn("Prestazioni eseguite", cleaned)
        self.assertNotIn("confrontato", cleaned)

    def test_removes_tracer_fragment(self):
        cleaned = strip_header_boilerplate(
            "di 18F-FDG (somministrazione e.v. di 18F-FDG); captazione patologica"
        )
        self.assertIn("captazione patologica", cleaned)
        self.assertNotIn("somministrazione", cleaned)

    def test_keeps_clinical_prose_without_colon(self):
        text = "La paziente riferisce dolore al petto da una settimana."
        self.assertEqual(strip_header_boilerplate(text), text)


class TestCompressMeasurementTable(unittest.TestCase):
    def test_compresses_to_parameter_plus_value(self):
        item = evidence(
            "E1", "Frazione Eiezione 4 Ch Simpson MonoPlano",
            category="imaging_finding", value_text="68 %",
            source="4CH B-Mode [35 - 75] ...",
        )
        event = bundle("EVT1", entity="frazione_eiezione_4_ch_simpson_monoplano").event
        compressed = compress_measurement_table(event, item)
        self.assertEqual(
            compressed, "Frazione Eiezione 4 Ch Simpson MonoPlano: 68 %"
        )

    def test_uses_numeric_value_with_unit(self):
        item = evidence(
            "E1", "PAPs", category="vital_sign", numeric_value=38.0, unit="mmHg"
        )
        event = bundle("EVT1", entity="paps", category="vital_sign").event
        # original casing of the measured parameter is preserved
        self.assertEqual(compress_measurement_table(event, item), "PAPs: 38 mmHg")

    def test_clean_event_summary_strips_and_compresses(self):
        item = evidence(
            "E1", "Frazione Eiezione 4 Ch Simpson MonoPlano",
            category="imaging_finding", value_text="68 %",
            source="4CH B-Mode [35 - 75] Frazione Eiezione 68 %",
        )
        event = bundle(
            "EVT1", entity="frazione_eiezione_4_ch_simpson_monoplano",
            category="imaging_finding",
            summary=(
                "Paziente: [PAZIENTE] 4CH B-Mode [35 - 75] Frazione Eiezione "
                "68 %; Simpson MonoPlano B-Mode [35 - 75] Frazione Eiezione "
                "68 %; TDI S 8 cm/s; Doppler Velocità E/A 1.2; VOD: normale"
            ),
        ).event
        cleaned = clean_event_summary(event, item)
        self.assertNotIn("Paziente", cleaned)
        self.assertEqual(cleaned, "Frazione Eiezione 4 Ch Simpson MonoPlano: 68 %")

    def test_long_shared_report_is_rendered_as_atomic_concept(self):
        source = (
            "Paziente: [PAZIENTE] Prestazioni eseguite e indicazione di dose "
            "secondo l'art. 161 del D.Lgs 101/2020; "
            + "testo tecnico e reperti multipli " * 30
        )
        item = evidence(
            "E1", "versamento pleurico", category="clinical_sign", source=source
        )
        event = bundle(
            "EVT1", category="clinical_sign", entity="versamento_pleurico",
            summary=source, evidence_items=[item],
        ).event
        self.assertEqual(
            clean_event_summary(event, item), "versamento pleurico"
        )

    def test_atomic_summary_keeps_value_and_site(self):
        item = evidence(
            "E1", "SpO2", category="vital_sign", numeric_value=88, unit="%"
        )
        item.anatomical_site = "aria ambiente"
        event = bundle(
            "EVT1", category="vital_sign", entity="spo2",
            evidence_items=[item],
        ).event
        self.assertEqual(
            atomic_event_summary(event, item), "SpO2; 88 %; aria ambiente"
        )


class TestDeduplicateBundles(unittest.TestCase):
    def test_merges_same_date_cross_category_double(self):
        summary = "Sospetto danno miocardico con troponina aumentata"
        a = bundle(
            "EVT_A", category="diagnosis", entity="sospetto_danno_miocardico",
            summary=summary,
        )
        b = bundle(
            "EVT_B", category="procedure", entity="sospetto_danno_miocardico",
            summary=summary,
        )
        final, merged = deduplicate_bundles(
            [a, b], evidence_by_id=evidence_map(),
            persisted_review_status={},
        )
        self.assertEqual(merged, 1)
        self.assertEqual(len(final), 1)
        survivor = final[0]
        self.assertEqual(
            sorted(survivor.event.structured_data["merged_into_ids"]),
            ["EVT_A"] if survivor.event.event_id == "EVT_B" else ["EVT_B"],
        )

    def test_merges_cross_date_verbatim_restatement(self):
        summary = (
            "Allargamento chirurgico ed exeresi di linfonodi sentinella "
            "negativi per il melanoma"
        )
        e1 = evidence("E1", "exeresi linfonodi sentinella", date="2019-05-01")
        e2 = evidence("E2", "exeresi linfonodi sentinella", date="2024-06-01")
        a = bundle(
            "EVT_A", category="surgery", entity="exeresi_linfonodi_sentinella",
            summary=summary, date="2019-05-01", evidence_items=[e1],
        )
        b = bundle(
            "EVT_B", category="surgery", entity="exeresi_linfonodi_sentinella",
            summary=summary, date="2024-06-01", evidence_items=[e2],
        )
        final, merged = deduplicate_bundles(
            [a, b], evidence_by_id=evidence_map(e1, e2),
            persisted_review_status={},
        )
        self.assertEqual(merged, 1)
        self.assertEqual(len(final), 1)
        survivor = final[0]
        # earliest date wins on both event and episode
        self.assertEqual(survivor.event.first_evidence_date, "2019-05-01")
        self.assertEqual(survivor.episode.onset_date, "2019-05-01")
        # merged evidence folded and later date becomes an update
        self.assertEqual(
            len(survivor.event.structured_data["evidence_ids"]), 2
        )
        self.assertEqual(len(survivor.updates), 1)
        self.assertEqual(survivor.updates[0].update_date, "2024-06-01")

    def test_does_not_merge_distinct_measurement_values(self):
        raw_table = (
            "4CH B-Mode [35 - 75] Frazione Eiezione 68 %; Simpson MonoPlano "
            "B-Mode [35 - 75] Frazione Eiezione 68 %; TDI S 8 cm/s; "
            "Doppler Velocità E/A 1.2; VOD: normale; VOS: normale; "
            "BOO: normale; TOO: normale"
        )
        e1 = evidence(
            "E1", "Frazione Eiezione 4 Ch Simpson MonoPlano",
            category="imaging_finding", date="2025-01-01",
            numeric_value=68.0, unit="%",
        )
        e2 = evidence(
            "E2", "Frazione Eiezione 4 Ch Simpson MonoPlano",
            category="imaging_finding", date="2025-01-01",
            numeric_value=72.0, unit="%",
        )
        a = bundle(
            "EVT_A", category="imaging_finding",
            entity="frazione_eiezione_4_ch_simpson_monoplano",
            summary=raw_table.replace("68", "68"), date="2025-01-01",
            evidence_items=[e1],
        )
        b = bundle(
            "EVT_B", category="imaging_finding",
            entity="frazione_eiezione_4_ch_simpson_monoplano",
            summary=raw_table.replace("68", "72"), date="2025-01-01",
            evidence_items=[e2],
        )
        final, merged = deduplicate_bundles(
            [a, b], evidence_by_id=evidence_map(e1, e2),
            persisted_review_status={},
        )
        self.assertEqual(merged, 0)
        self.assertEqual(len(final), 2)
        # distinct summaries preserved after compression
        summaries = {bundle.event.summary_short for bundle in final}
        self.assertTrue(any("68" in summary for summary in summaries))
        self.assertTrue(any("72" in summary for summary in summaries))

    def test_does_not_merge_present_with_absent(self):
        summary = "Dispnea da sforzo"
        a = bundle(
            "EVT_A", category="symptom", entity="dispnea",
            summary=summary, assertion="present",
        )
        b = bundle(
            "EVT_B", category="symptom", entity="dispnea",
            summary=summary, assertion="absent",
        )
        final, merged = deduplicate_bundles(
            [a, b], evidence_by_id=evidence_map(),
            persisted_review_status={},
        )
        self.assertEqual(merged, 0)
        self.assertEqual(len(final), 2)

    def test_does_not_merge_resolution_marked_event(self):
        summary = "Ipertensione in terapia"
        a = bundle(
            "EVT_A", category="diagnosis", entity="ipertensione",
            summary=summary, status="resolved", date_end="2025-01-31",
        )
        b = bundle(
            "EVT_B", category="diagnosis", entity="ipertensione",
            summary=summary,
        )
        final, merged = deduplicate_bundles(
            [a, b], evidence_by_id=evidence_map(),
            persisted_review_status={},
        )
        self.assertEqual(merged, 0)
        self.assertEqual(len(final), 2)

    def test_skips_group_with_two_human_locked(self):
        summary = "Nefropatia da mezzo di contrasto"
        a = bundle(
            "EVT_A", category="diagnosis", entity="nefropatia_mdc",
            summary=summary, review_status="accepted",
        )
        b = bundle(
            "EVT_B", category="diagnosis", entity="nefropatia_mdc",
            summary=summary, review_status="corrected",
        )
        c = bundle(
            "EVT_C", category="diagnosis", entity="nefropatia_mdc",
            summary=summary, review_status="auto",
        )
        final, merged = deduplicate_bundles(
            [a, b, c],
            evidence_by_id=evidence_map(),
            persisted_review_status={
                "EVT_A": "accepted", "EVT_B": "corrected",
            },
        )
        self.assertEqual(merged, 0)
        self.assertEqual(len(final), 3)

    def test_skips_group_with_rejected_member(self):
        summary = "Melanoma metastatico"
        a = bundle(
            "EVT_A", category="diagnosis", entity="melanoma_metastatico",
            summary=summary,
        )
        b = bundle(
            "EVT_B", category="diagnosis", entity="melanoma_metastatico",
            summary=summary,
        )
        final, merged = deduplicate_bundles(
            [a, b],
            evidence_by_id=evidence_map(),
            persisted_review_status={"EVT_B": "rejected"},
        )
        self.assertEqual(merged, 0)
        self.assertEqual(len(final), 2)

    def test_survivor_keeps_identity_and_review_rank_wins(self):
        summary = "Trattamento di prima linea con immunoterapia"
        e1 = evidence("E1", "immunoterapia", date="2023-01-10")
        e2 = evidence("E2", "immunoterapia", date="2023-01-10")
        pending = bundle(
            "EVT_PENDING", category="diagnosis", entity="immunoterapia",
            summary=summary, review_status="pending", confidence=0.5,
            evidence_items=[e2],
        )
        auto = bundle(
            "EVT_AUTO", category="procedure", entity="immunoterapia",
            summary=summary, review_status="auto", confidence=0.9,
            evidence_items=[e1],
        )
        final, merged = deduplicate_bundles(
            [auto, pending], evidence_by_id=evidence_map(e1, e2),
            persisted_review_status={"EVT_PENDING": "pending"},
        )
        self.assertEqual(merged, 1)
        self.assertEqual(len(final), 1)
        survivor = final[0]
        # pending outranks auto even though auto has higher confidence
        self.assertEqual(survivor.event.event_id, "EVT_PENDING")
        self.assertEqual(survivor.event.category, "diagnosis")
        # evidence folded, merged-away excluded, confidence maxed
        self.assertEqual(
            len(survivor.event.structured_data["evidence_ids"]), 2
        )
        self.assertIn("EVT_AUTO", survivor.event.structured_data["merged_into_ids"])
        self.assertEqual(survivor.event.confidence, 0.9)

    def test_human_accepted_summary_is_kept_verbatim(self):
        summary = "Paziente affetto da diabete di tipo 2 in terapia orale."
        a = bundle(
            "EVT_A", category="diagnosis", entity="diabete_tipo_2",
            summary="Header: " + summary, review_status="accepted",
        )
        final, merged = deduplicate_bundles(
            [a], evidence_by_id=evidence_map(),
            persisted_review_status={"EVT_A": "accepted"},
        )
        self.assertEqual(merged, 0)
        self.assertEqual(final[0].event.summary_short, "Header: " + summary)

    def test_skips_transitive_ambiguity(self):
        # A and B are within 0.95 similarity, B and C within 0.95, but A and C
        # fall below the cross-date threshold: merging the connected component
        # would drag unrelated content in, so the whole group is kept apart.
        core = " ".join(["nucleo"] * 30)
        a = bundle(
            "EVT_A", category="diagnosis", entity="diabete_tipo_2",
            summary=core + " alfa", date="2024-01-01",
        )
        b = bundle(
            "EVT_B", category="diagnosis", entity="diabete_tipo_2",
            summary=core + " bravo charlie", date="2024-03-01",
        )
        c = bundle(
            "EVT_C", category="diagnosis", entity="diabete_tipo_2",
            summary=core + " bravo charlie delta echo", date="2024-05-01",
        )
        final, merged = deduplicate_bundles(
            [a, b, c], evidence_by_id=evidence_map(),
            persisted_review_status={},
        )
        self.assertEqual(merged, 0)
        self.assertEqual(len(final), 3)


if __name__ == "__main__":
    unittest.main()
