"""Administrative form fields observed in local hospital report layouts.

Match complete cells, never unbounded sections or arbitrary repeated lines.
"""
import re
from ..pipeline.staff_identity import STAFF_SIGNATURE_HEADING

_ACTION = re.compile(r"\b(?:nega\w*|riferisc\w*|assum\w*|prescri\w*|consigli\w*|"
                     r"sospes\w*|sospend\w*|riprend\w*|somministra\w*|diagnos\w*|"
                     r"dolor\w*|febbr\w*|ricover\w*|dimetti\w*|metastas\w*|"
                     r"melanom\w*|carcinom\w*|biops\w*|materiale|posologi\w*|stadio|ECOG|"
                     r"negativ\w*|positiv\w*|allerg\w*|sintom\w*|emoglobin\w*|creatinin\w*|"
                     r"non|si|terapia|controllo|evidenz\w*|versamento|rilevat\w*)\b|"
                     r"\d\s*(?:mg|mcg|mmol|ml)\b", re.I)

_SPECIALTY = (
    r"(?:RADIOTERAPIA(?: ONCOLOGICA)?|ONCOLOGIA(?: CLINICA)?|ONCOLOGICO|"
    r"ENDOCRINOLOGIA(?: GENERALE)?|CARDIOLOGIA|NEUROLOGIA|PNEUMOLOGIA|"
    r"OCULISTICA|ORTOPEDIA|ORTOPEDICO(?: DIVISIONALE)?|ECOCOLORDOPPLER|"
    r"CHIRURGIA(?: PLASTICA| GENERALE)?|CHIR\. PLAS PREOP|"
    r"MEDICINA NUCLEARE|LABORATORIO(?: ANALISI)?|"
    r"RADIOLOGIA(?: DIAGNOSTICA(?: ED INTERVENTISTICA)?(?: INTERAZIENDALE)?)?)"
)
_SERVICE = re.compile(
    r"(?:SERVIZIO DI|AMBULATORIO(?: DI)?|D\.?S\.?A\.?|D\.?H\.?|"
    r"DAY (?:SERVICE|HOSPITAL)|DEG\.|DEGENZA|S\.?C\.?\s+DI)\s+"
    + _SPECIALTY + r"(?:\s+(?:\([A-Z0-9.]+\)|\d{2,}[A-Z]?))?"
    r"(?:\s*[-–]\s*SEDE DI\s+[A-ZÀ-Ü'’ -]+)?\s*:?", re.I
)
_TIMESTAMP_FIELD = re.compile(
    r"\d{1,2}[./-]\d{1,2}[./-]\d{4}[ \t]+\d{1,2}:\d{2}(?::\d{2})?[ \t]+(.+)"
)
_ACCESSION = r'(?:Acc\.?\s*(?:Number|No\.?)|Accession\s*(?:Number|No\.?)?|Numero\s+di\s+accesso)\s*:\s*(?:\[IDENTIFICATIVO RIMOSSO\]|[A-Z0-9][A-Z0-9._/-]*)'

_PATTERNS = [
    _ACCESSION,
    r'Provenienza\s*:\s*[^:;\n!?]+(?:\s+' + _ACCESSION + r')?',
    r"AMBULATORIO\s+N[°º.]?\s*\d+",
    r"U\.O\.A\.R\.[OU]\.?",
    r"(?:DATA|\[PAZIENTE\])/ORA ACCETTAZIONE\s+PROVENIENZA",
    r"DH ONCOLOGICO\s*\[TELEFONO RIMOSSO\]",
    r"AMBULATORIO\s+[A-ZÀ-Ü .’'-]+\s+N[°º.]?\s*\d+\s*,?\s*(?:IN\s+)?AREA\s+[A-Z0-9]+",
    r"(?:(?:Direttore|Responsabile|Dirigente|Medico|Professore|Patologo)(?:\s+(?:sanitario|F\.?F\.?))?\s*:?\s*)?\[MEDICO\](?:\s*[,/]\s*\[MEDICO\])*(?:\s*\([A-Z]{2,5}\))?[:.]?",
    r"(?:ASL|ASST|AUSL|IRCCS|AOU|ATS|A\.O\.U\.|Casa di cura|Istituto di ricovero e cura a carattere scientifico)\s+[^;!?]+",
    r"Segreteria Direzione",
    r"(?:TEL\.?SEGRETERIA:?|TEL DIRETTI DELL.AMBULATORIO)\s*\[TELEFONO RIMOSSO\]",
    r"Firma (?:\d° Chirurgo|dell.Anestesista)(?:\s*\.{3,})?",
    r"Referto firmato da [^\n]+ il \d{1,2}[/.-]\d{1,2}[/.-]\d{4}\.?",
    # Generic Italian report forms: no hospital/city/name is required.
    r"(?:Professore|Medici|Coordinatric[ei]|Coordinatore(?: Infermieristico)?|Cordiali saluti|Egr\.?\s*Collega\s*,?|Alla cortese atten[sz]ione del Medico Curante\s*,?)",
    r"(?:Direttore|Direttore sanitario|Responsabile)(?:\s+F\.?F\.?)?\s*:?\s*(?:Prof\.?|Dott\.?|Dr\.?)(?:ssa)?\.?\s+[^;\n]+",
    r"(?:T\.?S\.?R\.?M\.?|(?:IL\s+)?MEDICO (?:NEURO)?RADIOLOGO|IL MEDICO (?:NUCLEARE|SPECIALIZZANDO))(?:\s+(?:T\.?S\.?R\.?M\.?|(?:IL\s+)?MEDICO (?:NEURO)?RADIOLOGO|IL MEDICO (?:NUCLEARE|SPECIALIZZANDO)))*",
    r"(?:Id\.?\s*Paz\.?|Id Paziente|Nosologico|Acc\.? Number|Richiesta)\s*:\s*(?:\[IDENTIFICATIVO RIMOSSO\]|[A-Z0-9-]+)(?:\s+Nosologico:\s*\[IDENTIFICATIVO RIMOSSO\])?",
    r"Paziente\s*:\s*(?:\[PAZIENTE\](?: [A-ZÀ-Ü]+){0,3})?(?:\s+Data di Nascita:\s*\[DATA ANAGRAFICA RIMOSSA\]\s+Sesso:\s*[FM])?",
    r"(?:Figli[oa]|Familiare|Coniuge)\s*:\s*\[TELEFONO RIMOSSO\]",
    r"[,\s]*SEGRETERIA[^\n]*\[TELEFONO RIMOSSO\][^\n]*",
    r"(?:DH Oncologia|TEL\. Inferm\. Amb\.)\s*\[TELEFONO RIMOSSO\]",
    r"In caso di necessità\s*:?\s*(?:contattare\s+)?\[TELEFONO RIMOSSO\][^\n.]*",
    r"(?:Pagina\s+\d+\s+di\s+\d+\s+)?Documento provvisto di firma digitale[^\n]*",
    r"(?:Rappresentazione di un referto firmato elettronicamente|Num\. Certificato|Firmatario|Data firma digitale:|Versione (?:N°|digitale del referto))[^\n]*",
    r"Il referto n\.\s*\S+\s+è conservato secondo la normativa in vigore\.",
    r"(?:il massimario di scarto\.\s*)?Per ogni informazione o chiarimento sugli aspetti medici, pu[oò]['’]? rivolgersi al suo medico curante\.",
    r"(?:Risultati validati\.?|LEGENDA PER (?:ESAMI|ANALISI) ESEGUIT[IE] PRESSO\s*:?)",
    r"\[\d+\]\s+Laboratorio Analisi\s+[^\n]+",
    r"(?:LABORATORIO DI ANALISI UNICO PROVINCIALE\s*-\s*Direttore.*|SC LAB\.ANALISI CHIM\.CLIN\.e MICROBIOLOGIA)",
    r"(?:STRUTTURA COMPLESSA DI|U\.?O\.?|S\.?C\.?|S\.?S\.?D\.?)\s+(?:DI\s+)?[A-ZÀ-Üa-zà-ü .’'–-]+",
    r"(?:DEG\.|DEGENZA |DAY SERVICE )[A-ZÀ-Ü .]+(?:\([\d.]+\))?",
    r"(?:Prestazioni eseguite e indicazione|Indicazioni) di dose secondo l.?\s*art\.161 del D\.\s*Lgs\.?\s*101/2020\s*:?",
    r"(?:DATA ORA RICHIESTA\s+MEDICO INVIANTE|DATA ORA REFERTO|ORDINE\s+REFERTANTE)",
    r"(?:NOME E COGNOME|LUOGO E DATA DI NASCITA|SESSO|CF:|CODICE FISCALE(?::\s*\[IDENTIFICATIVO RIMOSSO\])?|A\d{6,})",
    r"(?:Tel\.?|Telefono|Fax|Cell\.?)\s*:\s*\[TELEFONO RIMOSSO\]",
    r"(?:Segreteria|Coordinatore Infermieristico|Infermieri (?:Degenza|Day Surgery)|Studio Medici Degenza|DEGENZA|DAY SURGERY|AMBULATORI)",
    r"(?:Dirigente Medico|DIRETTORE F\.F\.:.*|MEDICI:\s*(?:DR\.|DOTT\.).*|N°VERDE VISITE,CONTROLLI E MEDICAZIONI\s+\d+)",
    r"(?:[A-Z]{2}-\d+\s+\(\d+\)\s+AMB\..*|Azienda Ospedaliero - Universitaria di Ferrara)",
    r"(?:Segreteria:\s*(?:tel\.:\s*\[TELEFONO RIMOSSO\])(?:\s+Fax:\s*\[TELEFONO RIMOSSO\])?|spazio valrifdescr|Versione del referto firmato digitalmente N\.\s*\d+)",
    r"(?:\[\d\],?)+\s+UOC Patologia Clinica - Direttore .*",
    r"(?:NOME E COGNOME\s+SESSO|LUOGO E DATA DI NASCITA\s+CODICE FISCALE|REFERTO\s+CODICE U\.O\.|DATA/ORA ACCETTAZIONE\s+PROVENIENZA)",
    r"(?:Ente|Ospedale|Reparto/Ambulatorio|Luogo di nascita|Data Nascita|Data di Nascita|Sede legale|Partita IVA)\s*:\s*.*",
    r"(?:Partita IVA|Sede legale)\s+.*",
    r"(?:Universitaria di Ferrara|Arcispedale S\.?\s*Anna|ANATOMIA PATOLOGICA|LABORATORIO DI ANALISI UNICO PROVINCIALE|ANALISI BIOCHIMICO-CLINICHE E MICROBIOLOGIA)",
    r"(?:Direttore|Dirigenti? Medici?|Medico in Formazione(?: Specialistica)?|Il medico in formazione specialistica|Patologo|Medico in Formazione\s+Dirigente Medico|Risultati validati da)\s*:?",
    r"(?:Referto del\s+\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\s+delle ore\s+\d{1,2}:\d{2}:\d{2}|Referto di\s+\[PAZIENTE\]\s+n\..*|Referto (?:n\.|id\.)\s+\S+\s*-?\s*pag\..*)",
    r"(?:Questo documento è firmato digitalmente\.|Referto (?:id\..*|firmato digitalmente.*)|Refertato da:.*In Data:.*)",
    r"Ogni informazione clinica contenuta in questo documento è stata redatta sotto la responsabilità dei rispettivi validatori\.",
    r"Per ogni informazione o chiarimento sugli aspetti medici, può rivolgersi al suo medico curante\.",
    r"(?:LEGENDA PER ANALISI ESEGUITE PRESSO:|Centro Prelievi San Rocco.*|(?:\[\d\],?)+\s+(?:LUP Laboratorio|Laboratorio|Servizio di Immunoematologia).*Direttore.*)",
    r"(?:DAY SERVICE ONC\.CLIN\.\([\d.]+\)|Referto Completo)",
    r"(?:\[(?:EMAIL|TELEFONO) RIMOSS[OA]\]|[TFtf]\.\s*\[TELEFONO RIMOSSO\])(?:\s*[-–,]?\s*(?:www\.[^\s]+|[TFtf]\.\s*\[TELEFONO RIMOSSO\]|\[TELEFONO RIMOSSO\]|A\d{6,}))*",
]
_PATTERNS = [re.compile(p, re.I) for p in _PATTERNS]
_NAME = r"[A-ZÀ-Ü][A-Za-zÀ-ÖØ-öø-ÿ’'.-]*(?: [A-ZÀ-Ü][A-Za-zÀ-ÖØ-öø-ÿ’'.-]*){0,5}"
_ROSTER_ENTRY = (r"(?:(?i:infermieri?|infermiere|anestesista|chirurgo)(?i: di sala)?\s*:\s*)?"
                 r"(?:\[MEDICO\]|" + _NAME + r")")
_STAFF_ROSTER = re.compile(
    r"(?i:NOME OPERATORI|OPERATORI|MEDICI|INFERMIERI|INFERMIERE DI SALA)\s*:\s*"
    + _ROSTER_ENTRY + r"(?:\s*[,;/–.-]\s*" + _ROSTER_ENTRY + r")*[.;]?"
)
_DOCTOR_PART = r"(?i:(?:(?:Direttore|Patologo|Analisi effettuata dalla|Medico in formazione specialistica)\s+)?(?:Prof\.?|Dott\.?|Dr\.?)(?:ssa)?\.?\s+)" + _NAME
_DOCTOR = re.compile(_DOCTOR_PART + r"(?:\s*[,/]\s*" + _DOCTOR_PART + r")*(?:\s*\([A-Z]{2,5}\))?[:.]?")
_CAPITAL_HEADER = re.compile(r"(?:U\.?O\.?|UNITA[’À']? OPERATIVA|DIP\.)\s+[A-ZÀ-Ü0-9\s./()’'&,-]+")
_PATIENT_FIELD = re.compile(r"\[PAZIENTE\](?: [A-ZÀ-Ü][A-ZÀ-Ü’'-]*){0,3}(?:[ \t]+[FM])?")
_BIRTH_FIELD = re.compile(r"[A-ZÀ-Ü’' ,.-]+\[DATA ANAGRAFICA RIMOSSA\](?:\s*\[CODICE FISCALE RIMOSSO\])?")
_FIELD_PREFIX = re.compile(r"[ \t]*(?:(?:CODICE FISCALE|CF):[ \t]*\[(?:IDENTIFICATIVO|CODICE FISCALE) RIMOSSO\]|(?:Tel\.?|Telefono|Fax):[ \t]*\[TELEFONO RIMOSSO\])", re.I)


def _field_parts(value):
    prefix = _FIELD_PREFIX.match(value)
    if prefix:
        yield prefix.group()
        if prefix.end() < len(value):
            yield value[prefix.end():]
    else:
        yield value


def administrative_cell(value):
    if STAFF_SIGNATURE_HEADING.fullmatch(value):
        return 'staff_signature_heading'
    if (_SERVICE.fullmatch(re.sub(r'[ \t]+', ' ', value))
            and not _ACTION.search(re.sub(r'\bdiagnostica\b', '', value, flags=re.I))):
        return "service_header"
    # A timestamp is removable only with a wholly administrative companion.
    # Standalone times and clinical timelines (e.g. pain scores) remain intact.
    stamp = _TIMESTAMP_FIELD.fullmatch(value)
    if stamp and administrative_cell(stamp[1]):
        return "timestamped_administrative_field"
    if _ACTION.search(value):
        return None
    if _STAFF_ROSTER.fullmatch(value):
        return "staff_roster"
    if any(pattern.fullmatch(value) for pattern in _PATTERNS):
        return "hospital_form_field"
    if _DOCTOR.fullmatch(value):
        return "staff_signature"
    if _CAPITAL_HEADER.fullmatch(value):
        return "department_header"
    if _PATIENT_FIELD.fullmatch(value) or _BIRTH_FIELD.fullmatch(value):
        return "anonymized_identity_field"
    return None


def layout_cells(source):
    """Split space-aligned columns only when a recognized admin cell exists."""
    for line in source.splitlines(keepends=True):
        if administrative_cell(line.strip().rstrip(';')):
            yield line
            continue
        # Explicit form labels can share a line with the clinical question even
        # with a single space. Preserve the clinical field as a separate cell.
        boundary = re.search(r'(?i)(?<=\S)[ \t]+(?=(?:Quesito Clinico|Anamnesi|Data[- ]Ora Esame|Data Esame|Conclusioni|Referto)\s*:)', line)
        if boundary and administrative_cell(line[:boundary.start()].strip()):
            yield line[:boundary.end()]
            yield line[boundary.end():]
            continue
        accession = re.match(r'[ \t]*' + _ACCESSION, line, re.I)
        if accession:
            yield accession.group()
            yield line[accession.end():]
            continue
        # Some PDF renderers concatenate a form field and an exam title with
        # a single space. Remove only the recognized field, keeping its tail.
        prefix = _FIELD_PREFIX.match(line)
        if prefix:
            yield prefix.group()
            line = line[prefix.end():]
        pieces = list(re.finditer(r"\S(?:.*?\S)?(?=[ \t]{2,}|\r?\n|$)", line))
        if len(pieces) > 1 and any(administrative_cell(p.group().strip()) for p in pieces):
            end = 0
            for piece in pieces:
                if piece.start() > end:
                    yield line[end:piece.start()]
                yield from _field_parts(piece.group())
                end = piece.end()
            if end < len(line):
                yield line[end:]
        else:
            yield line


def tidy_layout(text):
    """Trim page-padding whitespace without changing internal table spacing."""
    lines = [line.strip() for line in text.splitlines()]
    result = re.sub(r"\n{3,}", "\n\n", '\n'.join(lines)).strip()
    return result + ('\n' if result and text.endswith('\n') else '')
