"""Redact explicitly attributed staff names without deleting clinical prose."""
import re

_TOKEN = r"(?:[A-ZÀ-Ü]\.|[A-ZÀ-Ü][A-Za-zÀ-ÖØ-öø-ÿ'’\-]*|(?:de|di|del|della|van|von))"
_PATTERN = re.compile(
    r"(?<!\w)(?P<title>(?i:(?:dott\.?|dr\.?|prof\.?)(?:ssa|\.ssa)?\.?|"
    r"(?:medico(?: refertante)?|refertato da|tsrm)\s*:))[ \t]*"
    r"(?P<name>" + _TOKEN + r"(?:[ \t]" + _TOKEN + r"){0,5})(?!\w)"
)
_STOP = {'si', 'non', 'nega', 'nessuna', 'nessun', 'terapia', 'diagnosi', 'anamnesi',
         'referto', 'controllo', 'conclusioni', 'ecog', 'ps', 'eo', 'tc', 'rm', 'pet',
         'consiglia', 'prescrive', 'riferisce', 'materiale', 'dose', 'ritmo',
         'negativo', 'positivo', 'sodica', 'clavulanico', 'il', 'la', 'in', 'per'}
_PARTICLES = {'de', 'di', 'del', 'della', 'van', 'von'}
_ROLE = (
    r'(?:(?:IL|LA)\s+)?(?:MEDICO(?:\s+(?:(?:NEURO)?RADIOLOGO(?:\s+SPECIALIZZANDO)?|REFERTANTE|NUCLEARE|SPECIALIZZANDO|IN FORMAZIONE(?: SPECIALISTICA)?))?|'
    r'PROFESSORE|PATOLOGO|DIRIGENTE MEDICO|CPSI(?:\s+TRIAGISTA)?|INFERMIER[EA](?:\s+TRIAGISTA)?|'
    r'TECNICO(?:\s+DI)?\s+RADIOLOGIA|T\.?S\.?R\.?M\.?)'
    r'(?:[ \t]+(?:DR\.|DOTT\.|DOTT\.SSA|PROF\.))?')
STAFF_SIGNATURE_HEADING = re.compile(_ROLE + r'(?:[ \t]+' + _ROLE + r')*[ \t]*:?', re.I)
_FULL_NAME = re.compile(_TOKEN + r'(?:[ \t]+' + _TOKEN + r'){1,5}')
_NOT_PERSON = _STOP | {'scala', 'numerica', 'valore', 'data', 'ora', 'prognosi',
    'pronto', 'soccorso', 'sodio', 'potassio', 'emocromo', 'formula', 'paziente',
    'azotemia', 'glucosio', 'acido', 'acetilsalicilico', 'cloruro', 'calcio',
    'risonanza', 'magnetica', 'mezzo', 'contrasto', 'pressione', 'arteriosa'}


def signature_staff_names(text):
    """Learn full names only from explicit staff signature fields, in memory."""
    names = set()
    awaiting = False
    for line in text.splitlines():
        value = line.strip()
        if not value:
            continue
        if STAFF_SIGNATURE_HEADING.fullmatch(value):
            awaiting = True
            continue
        if awaiting:
            for cell in re.split(r'[ \t]{2,}', value):
                if _FULL_NAME.fullmatch(cell):
                    tokens = {m.group().casefold() for m in re.finditer(_TOKEN, cell)}
                    if not tokens & _NOT_PERSON:
                        names.add(' '.join(cell.split()))
        awaiting = False
    return names


def redact_known_staff_names(text, names):
    """Replace verified full names across pages without altering surrounding data."""
    if not names:
        return text, 0
    alternatives = [r'[ \t]+'.join(re.escape(t) for t in name.split())
                    for name in sorted(names, key=len, reverse=True)]
    pattern = re.compile(r'(?<!\w)(?:' + '|'.join(alternatives) + r')(?!\w)', re.I)
    return pattern.subn('[MEDICO]', text)


def redact_staff_names(text):
    text, count = redact_known_staff_names(text, signature_staff_names(text))
    def replace(match):
        nonlocal count
        end = 0
        name = match.group('name')
        for token in re.finditer(_TOKEN, name):
            value = token.group().casefold()
            if value in _STOP:
                break
            if value not in _PARTICLES:
                end = token.end()
        if not end:
            return match.group()
        count += 1
        return '[MEDICO]' + name[end:]
    return _PATTERN.sub(replace, text), count
