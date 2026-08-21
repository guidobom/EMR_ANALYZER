from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock, patch

from emr_analyzer.clinical.atomic_evidence import AtomicEvidenceExtractor
from emr_analyzer.extraction.llm_client import LlmClient
from emr_analyzer.gui.main_window import MainWindow
from emr_analyzer.settings import LLMRoleConfig


def _config(*, context=32768, workers=3, temperature=0.1, output=6144):
    return LLMRoleConfig(
        model="qwen3-14b",
        context_length=context,
        parallel_workers=workers,
        temperature=temperature,
        max_output_tokens=output,
    )


def test_propagate_state_llm_updates_registry_extractor_too():
    history = SimpleNamespace(_llm=None)
    registry = SimpleNamespace(llm=None, atomic_extractor=None)
    window = SimpleNamespace(
        _services={
            "clinical_history_builder": history,
            "registry_builder": registry,
        }
    )
    client = SimpleNamespace(model="qwen3-14b")

    MainWindow._propagate_state_llm(window, client)

    assert history._llm is client
    assert registry.llm is client
    assert isinstance(registry.atomic_extractor, AtomicEvidenceExtractor)
    assert registry.atomic_extractor.llm is client


def test_apply_stops_only_obsolete_physical_runtime_shapes():
    old = {
        "document": _config(context=16384, workers=1, temperature=0.0),
        "clinical_state": _config(context=16384, workers=1),
    }
    new = {
        "document": _config(context=32768, workers=3, temperature=0.0),
        "clinical_state": _config(context=32768, workers=3),
    }
    statusbar = SimpleNamespace(showMessage=Mock())
    window = SimpleNamespace(
        _services={"llm_configs": old},
        _propagate_document_llm=Mock(),
        _propagate_state_llm=Mock(),
        update_model_status=Mock(),
        statusbar=statusbar,
        _ollama_available=False,
    )

    def identity(client):
        return (
            "/models/qwen3-14b.gguf",
            client.context_length,
            client.parallel_workers,
        )

    with (
        patch("emr_analyzer.gui.main_window.save_llm_configs"),
        patch.object(LlmClient, "runtime_identity", new=identity),
        patch.object(
            LlmClient, "is_available", new_callable=PropertyMock,
            return_value=True,
        ),
        patch.object(
            LlmClient, "server_available", new_callable=PropertyMock,
            return_value=True,
        ),
        patch.object(
            LlmClient,
            "unload_runtimes",
            return_value={"errors": {}, "unloaded": ["old"]},
        ) as unload,
    ):
        MainWindow._apply_llm_configs(window, new)

    unload.assert_called_once()
    obsolete = unload.call_args.args[0]
    assert len(obsolete) == 1
    assert obsolete[0].context_length == 16384
    assert window._services["llm_configs"] == new


def test_request_only_changes_keep_shared_runtime_loaded():
    old = {
        "document": _config(temperature=0.0, output=8192),
        "clinical_state": _config(temperature=0.1, output=6144),
    }
    new = {
        "document": _config(temperature=0.2, output=12288),
        "clinical_state": _config(temperature=0.0, output=10240),
    }
    window = SimpleNamespace(
        _services={"llm_configs": old},
        _propagate_document_llm=Mock(),
        _propagate_state_llm=Mock(),
        update_model_status=Mock(),
        statusbar=SimpleNamespace(showMessage=Mock()),
        _ollama_available=False,
    )

    def identity(client):
        return (
            "/models/qwen3-14b.gguf",
            client.context_length,
            client.parallel_workers,
        )

    with (
        patch("emr_analyzer.gui.main_window.save_llm_configs"),
        patch.object(LlmClient, "runtime_identity", new=identity),
        patch.object(
            LlmClient, "is_available", new_callable=PropertyMock,
            return_value=True,
        ),
        patch.object(
            LlmClient, "server_available", new_callable=PropertyMock,
            return_value=True,
        ),
        patch.object(LlmClient, "unload_runtimes") as unload,
    ):
        MainWindow._apply_llm_configs(window, new)

    unload.assert_not_called()


def test_close_is_refused_while_an_llm_operation_is_running():
    event = SimpleNamespace(ignore=Mock(), accept=Mock())
    window = SimpleNamespace(
        workspace_tabs=SimpleNamespace(llm_operation_running=lambda: True),
        _services={},
    )
    with patch("emr_analyzer.gui.main_window.QMessageBox.warning") as warning:
        MainWindow.closeEvent(window, event)

    warning.assert_called_once()
    event.ignore.assert_called_once()
    event.accept.assert_not_called()
