"""Normalize in memory and retain deterministic page provenance."""
from collections import Counter

from .clinical_text_result import ClinicalTextIsolationError, ClinicalTextIsolationResult
from .clinical_text_filter import ClinicalTextFilter
from .clinical_layout import detect_sidebar, clinical_page_text
from .continuous_markdown import merge_pages, source_pages
from ..pipeline.staff_identity import signature_staff_names, redact_known_staff_names


def normalize_document(isolator, text, document_date=None, parsing_result=None,
                       sensitive_identity=None, *, cancel_check=None):
    """Never return a partial document; metadata is added after validation.

    An absent filter selects the deterministic clinical-text filter. Pages are
    processed separately, then joined with a database map of their output spans.
    """
    pages = getattr(parsing_result, "pages", None)
    layout_audits = {}
    if isinstance(pages, (list, tuple)) and pages and all(hasattr(p, "text") for p in pages):
        sidebar = detect_sidebar(pages)
        units = []
        for page in pages:
            page_text, layout_audit = clinical_page_text(page, sidebar)
            units.append((page.page, page_text))
            if layout_audit:
                layout_audits[page.page] = layout_audit
        numbers = [number for number, _ in units]
        if numbers != list(range(1, len(units) + 1)):
            raise ClinicalTextIsolationError("Sequenza delle pagine sorgenti non valida")
    elif isinstance(getattr(getattr(parsing_result, 'document', None), 'pages', None), dict):
        from ..pipeline.converter import docling_source_markdown
        document = parsing_result.document
        numbers = sorted(document.pages)
        if not numbers or numbers != list(range(1, len(numbers)+1)):
            raise ClinicalTextIsolationError("Sequenza delle pagine Docling non valida")
        units = [(number, docling_source_markdown(document, page_no=number)) for number in numbers]
    else:
        try:
            units = source_pages(text)
        except ValueError as exc:
            raise ClinicalTextIsolationError(str(exc)) from exc
    parts, warnings, counts = [], [], Counter()
    staff_names = set()
    for _, source in units:
        staff_names.update(signature_staff_names(source))
    page_audits = []
    chunks = 0
    last = None
    for number, source in units:
        if cancel_check and cancel_check():
            raise ClinicalTextIsolationError("Normalizzazione interrotta")
        if not source or not source.strip():
            raise ClinicalTextIsolationError(f"Pagina {number or 'n.d.'} senza testo: verificare estrazione/OCR")
        selector = isolator if isolator is not None else ClinicalTextFilter()
        source, staff_count = redact_known_staff_names(source, staff_names)
        counts['staff_name'] += staff_count
        last = selector.isolate(source, document_date=document_date,
                                sensitive_identity=sensitive_identity, cancel_check=cancel_check)
        audit = getattr(last, "retention_audit", {})
        if audit:
            page_audits.append({"page": number, **audit,
                                "layout_selection": layout_audits.get(number, {})})
        if not last.text.strip():
            raise ClinicalTextIsolationError("Normalizzazione vuota")
        parts.append((number, last.text))
        counts.update(last.redaction_counts)
        warnings.extend(f"Pagina {number or 'n.d.'}: {warning}" for warning in last.warnings)
        chunks += last.chunk_count
    if cancel_check and cancel_check():
        raise ClinicalTextIsolationError("Normalizzazione interrotta")
    body, provenance = merge_pages(parts)
    return ClinicalTextIsolationResult(
        text=body, model_name=last.model_name,
        prompt_version=last.prompt_version, chunk_count=chunks,
        warnings=warnings, redaction_counts=dict(counts),
        deidentification_version=last.deidentification_version,
        retention_audit={"pages": page_audits, **provenance,
                         "review_required": any(p.get("review_required") for p in page_audits)},
    )
