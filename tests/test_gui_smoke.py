"""Build the real services and main window on a temporary project."""

from emr_analyzer.settings import LLMRoleConfig, MODEL_ROLES


def test_main_window_builds_with_patient_tabs(workspace, qapp, tmp_path, monkeypatch):
    import emr_analyzer.app as app_module

    monkeypatch.setattr(app_module, "SHARED_LEXICON_PATH", tmp_path / "shared_lexicon.db")
    configs = {role: LLMRoleConfig(model="") for role in MODEL_ROLES}
    app = object.__new__(app_module.EMRAnalyzerApp)
    app._services = {}
    app._llm_configs = configs
    app._qapp = qapp
    app._init_dirs()
    app._init_services()
    services = app._services
    assert services["extraction_pipeline"].llm is None
    assert services["atomic_evidence_llm_client"] is None

    from emr_analyzer.gui.main_window import MainWindow

    window = MainWindow()
    window.set_services(services)
    tabs = window.workspace_tabs
    titles = [tabs.tabText(index) for index in range(tabs.count())]
    assert titles == ["📄 Documenti", "🔬 Laboratorio", "🧬 Eventi SNOMED",
                      "✓ Attribuzioni", "Lessico condiviso"]
    assert not tabs.llm_operation_running()
    window._closing = True
    services["shared_lexicon_repo"].close()
    services["db"].close()
