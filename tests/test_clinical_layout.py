from emr_analyzer.pipeline.pdf_extractor import PdfPage, PdfExtractionResult
from emr_analyzer.extraction.clinical_layout import detect_sidebar, clinical_page_text
from emr_analyzer.extraction.document_normalization import normalize_document


def word(text, x, y, width=None):
    return dict(text=text, x0=x, x1=x+(width or len(text)*3), top=y, bottom=y+8)


def page(directory=True):
    words = []
    if directory:
        for y, role in ((100,'Direttore'),(220,'Segreteria'),(400,'Coordinatrice')):
            words += [word(role,10,y), word('Tel.',10,y+12),word('Mario',10,y+24),word('Rossi',32,y+24)]
    words += [word('Anamnesi',120,130),word('Nega febbre.',120,145),
              word('Diagnosi',120,250),word('Lesione benigna.',120,265),
              word('[',115,410,width=2),word(']',120,410,width=2),
              word('ENOXAPARINA',128,410),word('4000UI',240,410),
              word('1 fiala sottocute',330,410),word('ore 20 per 10 giorni',330,422)]
    return PdfPage(1,600,800,'SOURCE WITH INTERLEAVED COLUMNS',words)


def test_geometric_directory_removed_preserving_all_clinical_cells_and_checkboxes():
    p = page()
    ratio=detect_sidebar([p])
    assert ratio is not None
    text,audit=clinical_page_text(p,ratio)
    assert 'Mario' not in text and 'Rossi' not in text
    for value in ('Nega febbre.', 'Lesione benigna.', '[', ']', 'ENOXAPARINA', '4000UI', '1 fiala sottocute', 'ore 20 per 10 giorni'):
        assert value in text
    assert audit['source_words'] == audit['retained_words']+len(audit['excluded_word_indices'])
    assert 'Mario' not in str(audit)
    result=normalize_document(None,'',parsing_result=PdfExtractionResult('test.pdf',[p]))
    assert result.retention_audit['pages'][0]['layout_selection']['source_words'] == len(p.words)
    assert '<!-- page:' not in result.text
    assert result.retention_audit['output_pages'][0]['page'] == 1


def test_no_directory_evidence_leaves_native_text_unchanged():
    p = page(False)
    ratio=detect_sidebar([p])
    assert ratio is None
    assert clinical_page_text(p,ratio) == (p.text,{})
