"""Contextual administrative blocks, independent of hospital names.

Only bounded, recognized form paragraphs are excluded. Clinical section titles,
exam dates, specimen data, doses and patient-specific instructions are retained.
"""
import re
from ..pipeline.staff_identity import STAFF_SIGNATURE_HEADING

# Wrapped legal notices must be recognized as a paragraph, before line filtering
# can leave their continuations behind. Never cross a page marker.
_BLOCKS = [
    ('legal_notice', r'["“]?Referto firmato digitalmente ai sensi delle norme vigenti:[\s\S]{0,1200}?82/2005["”]?'),
    ('legal_notice', r'["“]?Copia del referto informatico predisposto e conservato presso[^\n]*(?:\n[^\n]*){0,3}?82/2005["”]?'),
    ('formulary_notice', r"Si autorizza l.erogazione del medicinale/equivalente presente nel Prontuario[\s\S]{0,350}?attivo,\s+dosaggio e forma farmaceutica\."),
    ('formulary_notice', r"Sono erogabili dalla farmacia ospedaliera,[\s\S]{0,650}?corretto proseguimento della\s+terapia\."),
    ('medication_reconciliation_notice', r"In base alla terapia assunta dal paziente al momento del ricovero,\s+rilevata nell.ambito dell.attività di ricognizione\s+farmacologica,[\s\S]{0,600}?familiari \(care\s+giver\)[\"”]?"),
]
_BLOCKS = [(name, re.compile(r'^[ \t]*' + pattern + r'[ \t]*(?:\n|$)', re.I | re.M)) for name, pattern in _BLOCKS]
_SECTION = re.compile(r'^\s*(?:Anamnesi|Conclusioni|Diagnosi|Terapia:|Referto\s*$|Esame\s+Esito)', re.I | re.M)
_SIGNATURE_HEADING = STAFF_SIGNATURE_HEADING
_NAME_LINE = re.compile(r"[A-ZÀ-Ü][A-Za-zÀ-ÖØ-öø-ÿ'.’\-]+(?:[ \t]+[A-ZÀ-Ü][A-Za-zÀ-ÖØ-öø-ÿ'.’\-]+){0,4}")
# Exclude only names following an explicit signature heading. Uppercase clinical
# headings/medications must never be treated as anonymous signature candidates.
_NOT_NAME = re.compile(r'\b(?:TERAPIA|DIAGNOSI|REFERTO|NOTE|CONTROLLO|ESAME|SODICA|CLAVULANICO|NEGATIVO|POSITIVO|MATERIALE|CONCLUSIONI|ANAMNESI|DOLORE|FEBBRE)\b', re.I)


def administrative_spans(source):
    spans = []
    for reason, pattern in _BLOCKS:
        for match in pattern.finditer(source):
            value = match.group()
            if '<!--' not in value and not _SECTION.search(value):
                spans.append((match.start(), match.end(), reason))
    lines = list(re.finditer(r'[^\n]*(?:\n|$)', source))
    # Earlier filter versions could leave isolated continuations of a legal
    # paragraph. Match them only in the immediate vicinity of a legal notice.
    legal_fragment = re.compile(r'(?:del\s+)?23/01/2002\.|82/2005["”]?|conformità alle regole tecniche di cui all.art\. 71 del D\. Lgs 82/2005["”]?', re.I)
    for index, match in enumerate(lines):
        if legal_fragment.fullmatch(match.group().strip()):
            vicinity = ''.join(m.group() for m in lines[max(0,index-3):index+4])
            if re.search(r'firma|Copia del referto|normativa', vicinity, re.I):
                spans.append((match.start(), match.end(), 'legal_continuation'))
        # Address fragments can be left by wrapped labels in imaging forms.
        # Require a patient field immediately above and a complete postal form.
        if re.fullmatch(r"[A-ZÀ-Ü .’'-]+,\s*\d{1,8}\s*\([A-Z]{2}\)", match.group().strip()):
            preceding = ''.join(m.group() for m in lines[max(0,index-3):index])
            if re.search(r'Paziente\s*:|\[INDIRIZZO RIMOSSO\]', preceding, re.I):
                spans.append((match.start(), match.end(), 'postal_form_context'))
                if index+1 < len(lines) and re.fullmatch(r'[A-ZÀ-Ü ]{3,40}', lines[index+1].group().strip()) and not _NOT_NAME.search(lines[index+1].group()):
                    spans.append((lines[index+1].start(), lines[index+1].end(), 'postal_city_context'))
    signature_budget = 0
    for match in lines:
        value = match.group().strip()
        if '<!--' in value:
            signature_budget = 0
        elif _SIGNATURE_HEADING.fullmatch(value) and not _NOT_NAME.search(value):
            signature_budget = 3
        elif signature_budget and value:
            if value == '[MEDICO]':
                signature_budget -= 1
            elif _NAME_LINE.fullmatch(value) and not _NOT_NAME.search(value):
                spans.append((match.start(), match.end(), 'signature_name_context'))
                signature_budget -= 1
            else:
                signature_budget = 0
    # Earlier/larger spans take precedence; audit decisions partition the source.
    result = []
    for span in sorted(spans, key=lambda s: (s[0], -s[1])):
        if not result or span[0] >= result[-1][1]:
            result.append(span)
    return result
