"""Review tab and concept dialog driven offscreen on a synthetic project."""

import pytest
from PyQt5.QtGui import QTextCursor

from emr_analyzer.gui.qt_utils import qt_offset
from tests.helpers import FakeLlm, KeywordExtractor, Project

TEXT = "Riferisce tosse da tre giorni. Nega febbre. Prosegue nivolumab."


@pytest.fixture
def tab(workspace, qapp):
    from emr_analyzer.gui.event_review_tab import EventReviewTab

    project = Project(workspace)
    project.add_patient("P001")
    project.add_document("P001", "DOC_001", TEXT)
    pipeline = project.pipeline(FakeLlm(), KeywordExtractor())
    pipeline.extract_patient("P001")
    services = {"db": project.db, "extraction_pipeline": pipeline, "overlay_repo": project.overlays,
                "document_repo": project.documents}
    widget = EventReviewTab()
    widget.set_services(services)
    widget.load_patient("P001")
    yield widget, project, pipeline, services
    widget.request_shutdown()
    project.close()


def select(widget, label):
    rows = widget._model.rows
    widget._table.selectRow(next(index for index, event in enumerate(rows) if event.label == label))
    widget._on_selection()
    return widget._current


def test_table_editor_and_source(tab):
    widget, project, pipeline, services = tab
    assert sorted(event.label for event in widget._model.rows) == ["febbre", "nivolumab", "tosse"]
    event = select(widget, "tosse")
    assert widget._label.text() == "tosse"
    assert widget._source.text[event.start:event.end] == "tosse"
    assert widget._source.extraSelections()[0].cursor.selectedText() == "tosse"


def test_confirm_correct_fragment_and_reject(tab, workspace):
    widget, project, pipeline, services = tab
    select(widget, "tosse")
    widget._on_confirm()
    assert widget._current.status == "confirmed"

    select(widget, "febbre")
    widget._label.setText("Febbre")
    widget._on_save()
    assert widget._current.label == "Febbre" and widget._current.status == "corrected"

    start = TEXT.index("tosse da tre giorni")
    select(widget, "tosse")
    cursor = widget._source.textCursor()
    cursor.setPosition(qt_offset(TEXT, start))
    cursor.setPosition(qt_offset(TEXT, start + len("tosse da tre giorni")), QTextCursor.KeepAnchor)
    widget._source.setTextCursor(cursor)
    widget._on_use_selection()
    assert widget._current.quote == "tosse da tre giorni"

    select(widget, "nivolumab")
    widget._on_reject()
    assert widget._current.status == "rejected"                 # still listed under "Tutti", greyed
    widget._filter.setCurrentText("Da revisionare")
    assert "nivolumab" not in [event.label for event in widget._model.rows]
    widget._filter.setCurrentText("Scartati")
    assert [event.label for event in widget._model.rows] == ["nivolumab"]

    widget._auto_export()
    assert (workspace / "P001" / "clinical_events.fhir.json").is_file()


def test_concept_dialog_lists_project_concepts(tab, qapp):
    from emr_analyzer.gui.concept_review_dialog import ConceptReviewDialog

    widget, project, pipeline, services = tab
    dialog = ConceptReviewDialog(services)
    labels = {dialog._table.item(row, 0).text() for row in range(dialog._table.rowCount())}
    assert labels == {"tosse", "febbre", "nivolumab"}
    assert "senza codifica" in dialog._summary.text()
    dialog.reject()
