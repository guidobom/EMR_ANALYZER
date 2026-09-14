"""Deterministic selection of source lines, never generative rewriting."""
from collections import Counter
import hashlib
import re

from .clinical_text_result import ClinicalTextIsolationResult, ClinicalTextIsolationError
from .administrative_templates import administrative_cell, layout_cells, tidy_layout
from .administrative_blocks import administrative_spans
from ..pipeline.sensitive_data import SensitiveDataSanitizer

VERSION = "clinical-markdown-filter-v4.5"

# A clinical mention wins over an administrative-looking line. This deliberately
# favours review over silently deleting mixed-content text.
CLINICAL = re.compile(
    r"\b(?:diagnos\w*|anamnes\w*|terap\w*|assume\w*|assunzion\w*|riferisc\w*|"
    r"nega\w*|febbr\w*|dolor\w*|sintom\w*|allerg\w*|sospend\w*|iniziat\w*|"
    r"ricover\w*|dimess\w*|dimission\w*|intervent\w*|biops\w*|metasta\w*|"
    r"melanom\w*|carcinom\w*|progression\w*|remission\w*|recidiv\w*|"
    r"valutazion\w*|controll\w*|follow[ -]?up|consigli\w*|prescri\w*|"
    r"somministra\w*|posologi\w*|effett\w*\s+avvers\w*|materiale|siero|urina|"
    r"emoglobin\w*|creatinin\w*|linfocit\w*|neutrofil\w*|mg|mcg|µg|mmol|g/dl|ml)\b",
    re.I,
)

RULES = [
    ("administrative_label", r"(?:nome e cognome|cognome e nome|luogo e data di nascita|data di nascita|codice fiscale|codice u\.o\.|data/ora accettazione|indirizzo|telefono|sesso|ente|ospedale|reparto/ambulatorio)\s*:?"),
    ("administrative_heading", r"(?:dati anagrafici(?: del paziente)?|richiedenti|prestazioni erogate|recapiti|informazioni amministrative|firma digitale)\s*:?"),
    ("identity_field", r"(?:nome e cognome|cognome e nome|data di nascita|luogo e data di nascita|codice fiscale|codice assistito|id paziente|nosologico|numero cartella|data/ora accettazione|numero prenotazione|numero fattura|indirizzo|residenza|telefono|tel\.?|cellulare|fax|e-?mail)\s*:\s*.+"),
    ("redacted_identity", r"(?:\[(?:PAZIENTE|MEDICO|[^\]]*RIMOSS[OA])\][\s,;:.-]*)+"),
    ("institutional_header", r"(?:azienda ospedalier[ao](?:[ -]universitaria)?|azienda usl|ausl|servizio sanitario (?:regionale|nazionale)|arcispedale|policlinico|ospedale|unità operativa|unita operativa|u\.?o\.?|dipartimento)\s+[^.!?;]+"),
    ("digital_signature", r"(?:documento (?:informatico |provvisto di )?firmato digitalmente|firmato digitalmente|copia informatica del referto|firma digitale|documento informatico ai sensi)\b.*"),
    ("privacy_notice", r"(?:ai sensi (?:del|dell[’'])|informativa (?:sulla |sul trattamento dei dati |ai sensi )?privacy|trattamento dei dati personali|regolamento (?:ue|europeo)|gdpr|documento riservato|titolare del trattamento)\b.*"),
    ("page_counter", r"(?:pag(?:ina)?\.?\s*:?\s*\d+\s*(?:/|di)\s*\d+|_{3,})"),
]
RULES = [(name, re.compile(pattern, re.I)) for name, pattern in RULES]
SUSPECT = re.compile(r"\b(?:anagrafic\w*|privacy|gdpr|firma\w*|prenotazion\w*|recapit\w*|amministrativ\w*|indirizzo|telefono|codice fiscale)\b", re.I)


class ClinicalTextFilter:
    def __init__(self, llm_client=None):
        # Retained for role propagation; text filtering never calls the model.
        self.llm = llm_client

    def isolate(self, text, document_date=None, sensitive_identity=None, cancel_check=None):
        # Semicolons often separate identifying fields from clinical notes.
        # Keep that boundary so address redaction cannot consume the next note.
        safe_parts, redactions = [], Counter()
        for part in text.split(';'):
            safe = SensitiveDataSanitizer().sanitize(part, sensitive_identity, preserve_layout=True)
            safe_parts.append(safe.text)
            redactions.update(safe.counts)
        source = ';'.join(safe_parts)
        kept, decisions, warnings = [], [], []
        offset = 0
        def segments():
            cursor = 0
            for start, end, reason in administrative_spans(source) + [(len(source), len(source), None)]:
                for match in re.finditer(r"[^;\n]*[;\n]|[^;\n]+$", source[cursor:start]):
                    for cell in layout_cells(match.group()):
                        yield cell, None
                if end > start:
                    yield source[start:end], reason
                cursor = end

        for line, block_rule in segments():
            if cancel_check and cancel_check():
                raise ClinicalTextIsolationError("Filtraggio interrotto")
            value = line.strip().rstrip(';').lstrip('#').strip()
            rule = next((name for name, pattern in RULES if pattern.fullmatch(value)), None)
            guarded = bool(CLINICAL.search(value))
            template_rule = block_rule or administrative_cell(value)
            remove = bool(template_rule) or (bool(rule) and not guarded)
            if template_rule:
                rule = template_rule
            review = bool(value) and not remove and bool(SUSPECT.search(value))
            if review:
                warnings.append(f"Blocco misto o amministrativo dubbio conservato: caratteri {offset}-{offset + len(line)}")
            decisions.append({"start": offset, "end": offset + len(line),
                              "action": "remove" if remove else "keep",
                              "reason": rule if remove else ("review" if review else "preserve")})
            if not remove:
                kept.append(line)
            offset += len(line)
        selected = ''.join(kept)
        body = tidy_layout(selected)
        return ClinicalTextIsolationResult(
            text=body if body.strip() else "[NESSUN CONTENUTO CLINICO]",
            model_name="deterministic", prompt_version=VERSION,
            chunk_count=1, warnings=warnings, redaction_counts=dict(redactions),
            retention_audit={"version": VERSION, "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                             "source_characters": len(source), "retained_characters": len(selected),
                             "removed_characters": len(source) - len(selected), "decisions": decisions,
                             "formatting": "trim_line_margins_and_blank_padding",
                             "output_characters": len(body),
                             "formatting_removed_characters": len(selected) - len(body),
                             "review_required": bool(warnings)},
        )
