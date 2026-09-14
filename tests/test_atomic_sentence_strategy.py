"""Sentence-granular extraction strategy (challenger B, pipeline v13)."""

import re

from types import SimpleNamespace

from emr_analyzer.clinical.atomic_evidence import (
    AtomicEvidenceExtractor,
    AtomicExtractionCancelled,
    SentenceSpan,
    build_atomic_sentence_prompt,
    build_sentence_windows,
    _sentence_has_clinical_signal,
)


def _spans(*texts):
    return [
        SentenceSpan(index, 0, len(text), text)
        for index, text in enumerate(texts, start=1)
    ]


class _SentenceLlm:
    """Emits one diagnosis atom per sentence whose text mentions 'dispnea'."""

    model = "test-model"
    max_output_tokens = 2048

    def __init__(self):
        self.calls = []

    def generate_structured(
        self, prompt, system, schema, *, max_tokens=None
    ):
        self.calls.append({"prompt": prompt, "system": system, "schema": schema})
        match = re.search(r"FRASE OBIETTIVO \[S(\d+)\] (.+)", prompt)
        if match is None or "dispnea" not in match.group(2):
            return {
                "diagnosis": [],
                "symptom": [],
                "medication": [],
                "laboratory_test": [],
                "radiology_finding": [],
                "instrumental_finding": [],
                "procedure": [],
                "clinical_decision": [],
                "hospitalization": [],
                "discharge": [],
                "clinical_sign": [],
                "vital_sign": [],
                "histopathology": [],
                "biomarker": [],
            }
        return {
            "diagnosis": [],
            "symptom": [{
                "concept": "dispnea",
                "polarity": "present",
                "source_refs": [int(match.group(1))],
            }],
            "medication": [],
            "laboratory_test": [],
            "radiology_finding": [],
            "instrumental_finding": [],
            "procedure": [],
            "clinical_decision": [],
            "hospitalization": [],
            "discharge": [],
            "clinical_sign": [],
            "vital_sign": [],
            "histopathology": [],
            "biomarker": [],
        }


def test_build_sentence_windows_carries_plain_context():
    spans = _spans("a", "b", "c", "d")
    windows = build_sentence_windows(spans, 1)
    assert [target.text for target, _ in windows] == ["a", "b", "c", "d"]
    assert [len(context) for _, context in windows] == [0, 1, 1, 1]
    assert windows[3][1][0].text == "c"


def test_build_sentence_windows_carries_temporal_heading():
    spans = _spans("Anamnesi", "01/02/2025", "Riferisce dispnea.")
    windows = build_sentence_windows(spans, 1)
    # A sentence following a dated section header carries the header plus
    # one extra preceding sentence, so the governing date stays resolvable.
    assert [len(context) for _, context in windows] == [0, 1, 2]
    assert [span.text for span in windows[2][1]] == [
        "Anamnesi", "01/02/2025",
    ]


def test_sentence_signal_gate_skips_only_empty():
    assert _sentence_has_clinical_signal(
        SentenceSpan(1, 0, 5, "TC total body.")
    )
    assert not _sentence_has_clinical_signal(SentenceSpan(1, 0, 0, "   "))


def test_atomic_sentence_prompt_marks_target_and_numbers_context():
    target = SentenceSpan(2, 0, 20, "Persiste dispnea da sforzo.")
    context = [SentenceSpan(1, 0, 12, "In data odierna")]
    prompt = build_atomic_sentence_prompt(
        target, context, document_type="visita", document_date="2025-02-03"
    )
    assert "FRASI DI CONTESTO" in prompt
    assert "[S1] In data odierna" in prompt
    assert "FRASE OBIETTIVO [S2] Persiste dispnea da sforzo." in prompt
    assert "usa solo l'ID 2" in prompt
    assert "data_documento=2025-02-03" in prompt


def test_sentence_window_schema_cites_only_target():
    extractor = AtomicEvidenceExtractor(SimpleNamespace(model="test-model"))
    schema = extractor._sentence_window_schema(2)
    for bucket in schema["properties"].values():
        refs = bucket["items"]["properties"]["source_refs"]["items"]
        assert refs["enum"] == [3]


def test_extract_document_sentencewise_calls_once_per_sentence():
    llm = _SentenceLlm()
    extractor = AtomicEvidenceExtractor(
        llm, strategy="sentence",
    )
    text = (
        "Il paziente giunge in visita di controllo.\n"
        "Persiste dispnea da sforzo di grado lieve.\n"
        "Si conferma la terapia in corso.\n"
        "Riferisce inoltre dispnea da sforzo invariata."
    )
    evidence = extractor.extract_document(
        patient_id="P1", document_id="D1", document_type="visita",
        document_date="2025-02-03", text=text,
    )
    # One call per clinically relevant sentence (all four here).
    assert len(llm.calls) == 4
    metrics = extractor.last_extraction_metrics()
    assert metrics["llm_calls"] == 4
    assert metrics["sentence_calls"] == 4
    # Both restatements of the same fact collapse into one atom.
    assert len(evidence) == 1
    assert evidence[0].normalized_entity == "dispnea"
    assert evidence[0].assertion == "present"


def test_extract_document_sentencewise_honours_cancel():
    extractor = AtomicEvidenceExtractor(
        _SentenceLlm(), strategy="sentence",
    )
    try:
        extractor.extract_document(
            patient_id="P1", document_id="D1", document_type="visita",
            document_date="2025-02-03", text="Dispnea da sforzo.",
            cancel_check=lambda: True,
        )
    except AtomicExtractionCancelled:
        return
    raise AssertionError("la cancellazione non è stata rispettata")


def test_policy_strategy_field_selects_sentence_mode():
    from emr_analyzer.settings import ClinicalPipelinePolicy

    policy = ClinicalPipelinePolicy.from_dict({"atomic_strategy": "sentence"})
    assert policy.atomic_strategy == "sentence"
    extractor = AtomicEvidenceExtractor(
        _SentenceLlm(), policy=policy,
    )
    assert extractor.strategy == "sentence"
    extractor = AtomicEvidenceExtractor(_SentenceLlm())
    assert extractor.strategy == "chunked"
