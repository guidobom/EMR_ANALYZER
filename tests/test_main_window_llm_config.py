from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock, patch

from emr_analyzer.clinical.atomic_evidence import AtomicEvidenceExtractor
from emr_analyzer.clinical.registry_builder import ClinicalRegistryBuilder
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


def test_each_live_client_is_propagated_only_to_its_pipeline_stage():
    history = SimpleNamespace(_llm=None)
    registry = SimpleNamespace(
        llm=None, atomic_llm=None, event_llm=None, atomic_extractor=None
    )
    window = SimpleNamespace(
        _services={
            "clinical_history_builder": history,
            "registry_builder": registry,
        }
    )
    atomic = SimpleNamespace(model="atomic-model")
    events = SimpleNamespace(model="event-model")
    analysis = SimpleNamespace(model="analysis-model")

    MainWindow._propagate_atomic_llm(window, atomic)
    MainWindow._propagate_event_llm(window, events)
    MainWindow._propagate_state_llm(window, analysis)

    assert history._llm is analysis
    assert registry.atomic_llm is atomic
    assert registry.event_llm is events
    assert registry.llm is events
    assert isinstance(registry.atomic_extractor, AtomicEvidenceExtractor)
    assert registry.atomic_extractor.llm is atomic


def test_registry_builder_keeps_atomic_and_event_models_independent():
    atomic_backend = SimpleNamespace(stop_config=Mock())
    atomic = SimpleNamespace(
        model="atomic-model", backend=atomic_backend,
        runtime_identity=lambda: ("atomic.gguf", 32768, 4),
    )
    events = SimpleNamespace(
        model="event-model",
        runtime_identity=lambda: ("events.gguf", 65536, 2),
    )
    builder = ClinicalRegistryBuilder(
        registry_repo=None,
        evidence_repo=None,
        processing_repo=None,
        timeline_repo=None,
        document_repo=None,
        lab_repo=None,
        overlay_repo=None,
        atomic_llm_client=atomic,
        event_llm_client=events,
        db=object(),
    )

    assert builder.atomic_extractor.llm is atomic
    assert builder.event_llm is events
    assert builder.llm is events

    builder._release_atomic_runtime_before_events(enabled=True)

    atomic_backend.stop_config.assert_called_once_with(atomic)


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
    assert window._services["llm_configs"]["document"] == new["document"]
    assert window._services["llm_configs"]["clinical_state"] == new["clinical_state"]
    assert window._services["llm_configs"]["atomic_evidence"] == new["clinical_state"]
    assert window._services["llm_configs"]["clinical_events"] == new["clinical_state"]


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
