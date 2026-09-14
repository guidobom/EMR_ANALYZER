from pathlib import Path
from unittest.mock import Mock

import fitz
from PyQt5.QtCore import Qt

from emr_analyzer.gui.document_comparison_dialog import DocumentComparisonDialog
from emr_analyzer.gui.documents_tab import DocumentsTab
from emr_analyzer.models.document import DocumentRecord


def make_dialog(tmp_path, monkeypatch, *, markdown=True, pdf=True):
    original = tmp_path/'source.pdf'
    if pdf:
        document = fitz.open()
        for text in ('Pagina uno: referto originale.', 'Pagina due: terapia.'):
            page = document.new_page()
            page.insert_text((40,40),text)
        document.save(original)
        document.close()
    text_path = tmp_path/'DOC_1.md'
    if markdown:
        text_path.write_text('# Referto\nPrima terapia.\nSeconda terapia.\n<script>testo letterale</script>')
    monkeypatch.setattr('emr_analyzer.gui.document_comparison_dialog.normalized_text_path', lambda data: text_path if text_path.exists() else None)
    data=dict(id='DOC_1',patient_id='P001',filename='source.pdf',original_path=str(original))
    return DocumentComparisonDialog(data,{}), original, text_path


def test_side_by_side_original_and_literal_markdown_and_search(tmp_path,monkeypatch):
    dialog,original,text_path=make_dialog(tmp_path,monkeypatch)
    before=(original.read_bytes(),text_path.read_bytes())
    assert dialog._splitter.count()==2
    assert dialog._splitter.orientation()==Qt.Horizontal
    assert dialog._pdf._doc.page_count==2
    assert dialog._pdf._text_toggle.isHidden()
    assert dialog._text.isReadOnly()
    assert dialog._text.toPlainText()==text_path.read_text()
    dialog._pdf._next_page()
    assert dialog._pdf._current_page==1
    dialog._search.setText('terapia')
    dialog._find()
    first=dialog._text.textCursor().selectionStart()
    dialog._find()
    assert dialog._text.textCursor().selectionStart()>first
    dialog._find()
    assert dialog._text.textCursor().selectionStart()==first
    dialog._find(True)
    assert dialog._text.textCursor().selectionStart()>first
    dialog._search.setText('inesistente')
    dialog._find()
    assert 'Nessun risultato' in dialog._status.text()
    dialog.reject()
    assert dialog._pdf._doc is None
    assert (original.read_bytes(),text_path.read_bytes())==before


def test_missing_markdown_can_be_loaded_later_without_losing_pdf(tmp_path,monkeypatch):
    dialog,_,text_path=make_dialog(tmp_path,monkeypatch,markdown=False)
    assert dialog._pdf._doc is not None
    assert 'non disponibile' in dialog._status.text()
    text_path.write_text('Nuovo testo clinico.')
    dialog._load_markdown()
    assert dialog._text.toPlainText()=='Nuovo testo clinico.'
    text_path.write_bytes(b'\xff')
    dialog._load_markdown()
    assert 'Impossibile leggere' in dialog._status.text()
    assert not dialog._text.toPlainText()
    dialog.reject()


def test_missing_pdf_does_not_prevent_reading_markdown(tmp_path,monkeypatch):
    dialog,_,text_path=make_dialog(tmp_path,monkeypatch,pdf=False)
    assert dialog._pdf._doc is None
    assert dialog._text.toPlainText()==text_path.read_text()
    dialog.reject()


def test_table_button_opens_comparison_for_selected_document():
    doc=DocumentRecord('DOC_1','P001','source.pdf','/nonexistent/source.pdf','hash')
    repo=Mock();repo.list_by_patient.return_value=[doc]
    tab=DocumentsTab();tab.set_services({'document_repo':repo});tab.load_patient('P001')
    assert not tab._compare_btn.isEnabled()
    observed=[]
    tab._view_extracted_text=lambda doc_id,data:observed.append((doc_id,data['original_path']))
    tab._table.selectRow(0)
    assert tab._compare_btn.isEnabled()
    tab._compare_btn.click()
    assert observed==[('DOC_1','/nonexistent/source.pdf')]
    tab.close()
