from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication, QMessageBox

from emr_analyzer.gui.model_manager_dialog import ModelManagerDialog
from emr_analyzer.llm_backend.model_catalog import (
    MODEL_CATALOG,
    builtin_catalog_manifest,
    cache_catalog_manifest,
    filter_catalog,
    hardware_fit,
    load_catalog_snapshot,
    validate_catalog_manifest,
)
from emr_analyzer.utils.hardware import HardwareProfile


def _hardware(total_ram_gb: float = 120.0) -> HardwareProfile:
    return HardwareProfile(
        total_ram_gb=total_ram_gb,
        available_ram_gb=total_ram_gb - 8,
        cpu_cores_physical=20,
        cpu_cores_logical=20,
        has_apple_silicon=False,
        gpu_name="NVIDIA test GPU",
    )


def test_catalog_entries_are_unique_installable_ollama_models():
    assert len(MODEL_CATALOG) >= 7
    assert len({model.catalog_id for model in MODEL_CATALOG}) == len(
        MODEL_CATALOG
    )
    assert len({model.ollama_tag for model in MODEL_CATALOG}) == len(
        MODEL_CATALOG
    )
    for model in MODEL_CATALOG:
        assert ":" in model.ollama_tag
        assert model.source_url.startswith("https://ollama.com/library/")
        assert model.download_size_gb > 0
        assert model.context_length >= 32_768
        assert model.recommended_ram_gb >= model.minimum_ram_gb
        assert set(model.roles) <= {"document", "clinical_state"}

    resource = Path("emr_analyzer/resources/model_catalog.json")
    payload = json.loads(resource.read_text(encoding="utf-8"))
    snapshot = validate_catalog_manifest(payload)
    assert snapshot.models == MODEL_CATALOG


def test_catalog_prioritizes_role_and_filters_for_ram():
    clinical = filter_catalog(
        role="clinical_state", total_ram_gb=120, compatible_only=True
    )
    assert clinical[0].catalog_id == "qwen3-30b-a3b-instruct"
    assert all("clinical_state" in model.roles for model in clinical)

    events = filter_catalog(
        role="clinical_events", total_ram_gb=120, compatible_only=True
    )
    assert events[0].catalog_id == "qwen3-30b-a3b-instruct"
    assert all("clinical_state" in model.roles for model in events)

    atomic = filter_catalog(
        role="atomic_evidence", total_ram_gb=120, compatible_only=True
    )
    assert any(model.catalog_id == "qwen3-4b" for model in atomic)

    small_document = filter_catalog(
        role="document", total_ram_gb=8, compatible_only=True
    )
    assert [model.catalog_id for model in small_document] == ["qwen3-4b"]
    assert hardware_fit(small_document[0], 8).level == "compatible"

    medical = filter_catalog(query="clinico-specializzato", total_ram_gb=64)
    assert [model.catalog_id for model in medical] == ["medgemma-27b"]


def test_catalog_cache_is_atomic_validated_and_cannot_downgrade(tmp_path):
    cache = tmp_path / "catalog.json"
    manifest = builtin_catalog_manifest()
    manifest["catalog_version"] = 2
    manifest["review_date"] = "2026-09-01"
    raw = json.dumps(manifest, ensure_ascii=False).encode("utf-8")

    cached = cache_catalog_manifest(raw, cache)

    assert cached.catalog_version == 2
    assert load_catalog_snapshot(cache).source == "cache"
    downgrade = builtin_catalog_manifest()
    with pytest.raises(ValueError, match="meno recente"):
        cache_catalog_manifest(
            json.dumps(downgrade).encode("utf-8"), cache
        )
    assert load_catalog_snapshot(cache).catalog_version == 2


def test_invalid_cached_catalog_falls_back_to_integrated(tmp_path):
    cache = tmp_path / "catalog.json"
    cache.write_text('{"models": [{"source_url": "https://evil.test"}]}')

    snapshot = load_catalog_snapshot(cache)

    assert snapshot.source == "integrated"
    assert snapshot.models == MODEL_CATALOG


def test_visual_catalog_filters_and_prefills_one_click_install():
    app = QApplication.instance() or QApplication([])
    with (
        patch(
            "emr_analyzer.gui.model_manager_dialog.HardwareProfile.capture",
            return_value=_hardware(),
        ),
        patch(
            "emr_analyzer.gui.model_manager_dialog.cleanup_abandoned_partials",
            return_value=[],
        ),
    ):
        dialog = ModelManagerDialog(["gemma3-12b"])

    assert "qwen3-30b-a3b-instruct" in dialog._catalog_cards
    assert dialog._catalog_update_button.text() == "↻ Aggiorna catalogo"
    assert not dialog._catalog_cards["gemma3-12b"].install_button.isEnabled()
    role_index = dialog._catalog_role.findData("clinical_state")
    dialog._catalog_role.setCurrentIndex(role_index)
    assert "qwen3-4b" not in dialog._catalog_cards
    assert dialog._catalog_role.findData("atomic_evidence") >= 0
    assert dialog._catalog_role.findData("clinical_events") >= 0

    selected = next(
        model for model in MODEL_CATALOG
        if model.catalog_id == "qwen3-30b-a3b-instruct"
    )
    with (
        patch.object(QMessageBox, "question", return_value=QMessageBox.Yes),
        patch.object(dialog, "_start_install") as start_install,
    ):
        dialog._install_catalog_model(selected)

    assert dialog._source.currentData() == "ollama"
    assert dialog._ollama_tag.currentText() == selected.ollama_tag
    assert dialog._tabs.currentWidget() is dialog._manual_page
    start_install.assert_called_once_with()

    remote_manifest = builtin_catalog_manifest()
    remote_manifest["catalog_version"] = 2
    remote_manifest["review_date"] = "2026-09-01"
    remote_snapshot = replace(
        validate_catalog_manifest(remote_manifest), source="cache"
    )
    with (
        patch(
            "emr_analyzer.gui.model_manager_dialog.load_catalog_snapshot",
            return_value=remote_snapshot,
        ),
        patch.object(QMessageBox, "information"),
    ):
        dialog._on_catalog_updated({
            "catalog_version": 2,
            "review_date": "2026-09-01",
            "model_count": len(remote_snapshot.models),
        })
    assert "cache locale v2" in dialog._catalog_header.text()
    dialog.deleteLater()
    app.processEvents()
