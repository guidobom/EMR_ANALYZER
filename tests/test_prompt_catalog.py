from pathlib import Path

import pytest

from emr_analyzer import prompt_catalog


def test_all_bundled_prompt_files_are_nonempty_and_external():
    paths = prompt_catalog.available_prompts()
    names = {path.stem for path in paths}
    assert {
        "clinical_text_system", "clinical_text_instructions",
        "atomic_evidence_it", "atomic_evidence_system",
        "clinical_fusion_system", "clinical_fusion_task",
        "evidence_relations_system", "evidence_relations_task",
        "episode_assembly_system", "episode_assembly_task",
        "clinical_query_system", "narrative_profile_system",
    }.issubset(names)
    assert all(path.is_file() and path.read_text(encoding="utf-8").strip()
               for path in paths)


def test_prompt_loader_rejects_missing_required_contract(monkeypatch, tmp_path):
    monkeypatch.setattr(prompt_catalog, "PROMPT_DIR", Path(tmp_path))
    (tmp_path / "atomic_evidence_it.txt").write_text(
        "testo abbastanza lungo ma senza il marcatore richiesto",
        encoding="utf-8",
    )
    with pytest.raises(prompt_catalog.PromptConfigurationError):
        prompt_catalog.load_prompt(
            "atomic_evidence_it",
            required_markers=("source_refs",),
            minimum_length=20,
        )


def test_prompt_digest_changes_when_only_prompt_text_changes():
    first = prompt_catalog.prompts_digest("istruzione A", schema={"x": 1})
    second = prompt_catalog.prompts_digest("istruzione B", schema={"x": 1})
    assert first != second


def test_prompt_name_cannot_escape_catalog_directory():
    with pytest.raises(prompt_catalog.PromptConfigurationError):
        prompt_catalog.prompt_path("../segreto")


def test_custom_prompt_can_be_saved_and_activated(monkeypatch, tmp_path):
    custom_dir = tmp_path / "custom"
    selection_path = tmp_path / "active_versions.json"
    monkeypatch.setattr(prompt_catalog, "CUSTOM_PROMPT_DIR", custom_dir)
    monkeypatch.setattr(
        prompt_catalog, "PROMPT_SELECTION_PATH", selection_path
    )
    institutional = prompt_catalog.load_prompt("clinical_query_system")
    custom = prompt_catalog.save_custom_prompt(
        "clinical_query_system",
        "accuracy-v2",
        institutional + "\nMantieni separati fatti e interpretazioni.",
    )
    assert custom.path.is_file()
    assert prompt_catalog.active_prompt_selection(
        "clinical_query_system"
    ) == ("institutional", "institutional-v1")

    prompt_catalog.activate_prompt(
        "clinical_query_system", custom.version, custom.origin
    )

    assert prompt_catalog.active_prompt_selection(
        "clinical_query_system"
    ) == ("custom", "accuracy-v2")
    assert prompt_catalog.load_prompt("clinical_query_system").endswith(
        "Mantieni separati fatti e interpretazioni."
    )


def test_institutional_prompt_cannot_be_overwritten(monkeypatch, tmp_path):
    monkeypatch.setattr(prompt_catalog, "CUSTOM_PROMPT_DIR", tmp_path)
    with pytest.raises(prompt_catalog.PromptConfigurationError):
        prompt_catalog.save_custom_prompt(
            "clinical_query_system",
            "institutional-v1",
            prompt_catalog.load_prompt("clinical_query_system"),
        )
