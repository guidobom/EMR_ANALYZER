from types import SimpleNamespace

from emr_analyzer import prompt_catalog
from emr_analyzer.config import active_workspace
from emr_analyzer.gui.prompt_manager_dialog import PromptManagerDialog
from emr_analyzer.gui.prompt_manager_dialog import PromptPreviewWorker


def test_prompt_manager_lists_tasks_and_active_versions(monkeypatch, tmp_path):
    monkeypatch.setattr(
        prompt_catalog, "CUSTOM_PROMPT_DIR", tmp_path / "custom"
    )
    monkeypatch.setattr(
        prompt_catalog,
        "PROMPT_SELECTION_PATH",
        tmp_path / "active_versions.json",
    )
    dialog = PromptManagerDialog()
    try:
        assert dialog.task_combo.count() == len(
            prompt_catalog.prompt_definitions()
        )
        assert dialog.version_combo.count() >= 1
        assert "ATTIVO" in dialog.version_combo.currentText()
        assert dialog._current_version.origin == "institutional"
        assert not dialog.save_button.isEnabled()
        assert dialog.save_as_button.isEnabled()
    finally:
        dialog.close()


class _PatientRepo:
    def list_all(self):
        return [SimpleNamespace(id="P001", pseudonym="TEST")]


class _DocumentRepo:
    def __init__(self, document):
        self.document = document

    def list_by_patient(self, patient_id):
        return [self.document] if patient_id == "P001" else []


class _PreviewLlm:
    is_available = True
    model = "preview-model"

    def generate_text(self, prompt, system="", **_kwargs):
        assert "DOCUMENTO NORMALIZZATO" in prompt
        assert "PROMPT MODIFICATO" in system
        return "risultato di prova"

    def last_generation_metadata(self):
        return {"prompt_tokens": 42, "completion_tokens": 7}


def test_preview_uses_unsaved_editor_text_without_writing_database(
    monkeypatch, tmp_path
):
    del monkeypatch
    previous_workspace = active_workspace.path
    active_workspace.set_path(tmp_path)
    document = SimpleNamespace(
        id="DOC_000001",
        patient_id="P001",
        filename="referto.pdf",
        document_date="2024-01-01",
        document_type="visita_oncologica",
    )
    extraction = tmp_path / "P001" / "extraction"
    extraction.mkdir(parents=True)
    (extraction / "DOC_000001.md").write_text(
        "DOCUMENTO NORMALIZZATO", encoding="utf-8"
    )
    llm = _PreviewLlm()
    services = {
        "patient_repo": _PatientRepo(),
        "document_repo": _DocumentRepo(document),
        "clinical_state_llm_client": llm,
    }
    dialog = PromptManagerDialog(services=services, patient_id="P001")
    try:
        index = next(
            row for row in range(dialog.task_combo.count())
            if dialog.task_combo.itemData(row) == "clinical_query_system"
        )
        dialog.task_combo.setCurrentIndex(index)
        dialog.editor.setPlainText("PROMPT MODIFICATO NON SALVATO")
        request = dialog._build_preview_request(
            "clinical_query_system", document
        )
        assert request["system_prompt"] == "PROMPT MODIFICATO NON SALVATO"

        completed = []
        worker = PromptPreviewWorker(llm, request)
        worker.completed.connect(completed.append)
        worker.run()

        assert completed[0]["output"] == "risultato di prova"
        assert completed[0]["generation"]["prompt_tokens"] == 42
        assert not (tmp_path / "active_versions.json").exists()
    finally:
        dialog.editor.document().setModified(False)
        dialog.close()
        active_workspace.set_path(previous_workspace)
