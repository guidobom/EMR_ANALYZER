"""The application keeps only the extraction → SNOMED → review → FHIR path."""

import importlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULES = sorted(
    ".".join(path.relative_to(ROOT).with_suffix("").parts).removesuffix(".__init__")
    for path in (ROOT / "emr_analyzer").rglob("*.py")
)
REMOVED = (
    "clinical.registry_builder", "clinical.clinical_history_builder", "clinical.evidence_graph",
    "clinical.consolidation", "clinical.irae_layers", "clinical.dossier_query",
    "clinical.event_relations", "gui.clinical_history_tab", "gui.gold_set_tab",
    "database.registry_repo", "database.timeline_repo", "database.pipeline_repo",
)


@pytest.mark.parametrize("module", MODULES)
def test_every_module_imports(module):
    importlib.import_module(module)


@pytest.mark.parametrize("module", REMOVED)
def test_removed_modules_are_gone(module):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("emr_analyzer." + module)


def test_only_extraction_and_document_llm_roles():
    from emr_analyzer.settings import MODEL_ROLES, default_llm_configs

    assert MODEL_ROLES == ("document", "atomic_evidence")
    assert set(default_llm_configs()) == set(MODEL_ROLES)


def test_extraction_preset_uses_the_measured_sampling():
    from emr_analyzer.settings import default_llm_configs

    extraction = default_llm_configs()["atomic_evidence"]
    assert (extraction.temperature, extraction.top_p, extraction.top_k) == (0.7, 0.8, 20)


def test_legacy_extraction_sampling_is_migrated_once(tmp_path):
    import json

    from emr_analyzer.settings import (LLMRoleConfig, default_llm_configs,
                                       load_pipeline_llm_config)

    default = default_llm_configs()["atomic_evidence"]
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"pipeline_llm": {"atomic": {
        **default.to_dict(), "temperature": 0.0, "top_p": 0.9, "top_k": 40}}}),
        encoding="utf-8")
    migrated = load_pipeline_llm_config("atomic", default, path)
    assert (migrated.temperature, migrated.top_p, migrated.top_k) == (0.7, 0.8, 20)
    # A deliberate choice is never overwritten.
    path.write_text(json.dumps({"pipeline_llm": {"atomic": {
        **default.to_dict(), "temperature": 0.0, "top_p": 0.9, "top_k": 20}}}),
        encoding="utf-8")
    kept = load_pipeline_llm_config("atomic", default, path)
    assert kept.temperature == 0.0


def test_prompt_manifest_matches_prompt_files():
    import json

    folder = ROOT / "emr_analyzer" / "resources" / "prompts"
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    keys = {item["key"] for item in manifest["prompts"]}
    files = {path.stem for path in folder.glob("*.txt")}
    assert keys == files
