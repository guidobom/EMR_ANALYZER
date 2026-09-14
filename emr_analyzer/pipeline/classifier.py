"""Document classifier — determines the document type."""

import re
from typing import Optional

from ..models.document import DocumentType
from ..config import (
    RADIOLOGY_KEYWORDS, LAB_KEYWORDS,
    HOSPITALIZATION_KEYWORDS, VISIT_KEYWORDS,
)


class DocumentClassifier:
    """
    Multi-signal document classifier.
    Uses filename patterns, lexical rules, and structural cues.
    """

    # Imaging units issue imaging reports: the department is the document's
    # own letterhead, while the provenance names the *requesting* unit.
    _RADIOLOGY_UNIT = re.compile(
        r"\b(?:RADIOLOGIA|NEURORADIOLOGIA|RADIODIAGNOSTICA|"
        r"DIAGNOSTICA\s+PER\s+IMMAGINI)\b",
        re.IGNORECASE,
    )
    _NUCLEAR_UNIT = re.compile(r"\bMEDICINA\s+NUCLEARE\b", re.IGNORECASE)
    # The same units, recognized as letterhead lines inside the document
    # text itself ("Dipartimento ... di Radiologia", "STRUTTURA COMPLESSA DI
    # MEDICINA NUCLEARE").  This covers documents whose stored header
    # metadata is missing or partial; a quoted exam inside an oncology
    # visit has no such line.
    _RADIOLOGY_UNIT_LINE = re.compile(
        r"^\s*(?:DIPARTIMENTO|U\.?O\.?|UNIT[AÀ]\s+OPERATIVA|"
        r"STRUTTURA\s+(?:COMPLESSA|SEMPLICE)|S\.?C\.?)\s*"
        r"[^.\n]{0,90}\b(?:RADIOLOGIA|NEURORADIOLOGIA|RADIODIAGNOSTICA|"
        r"DIAGNOSTICA\s+PER\s+IMMAGINI)\b",
        re.IGNORECASE | re.MULTILINE,
    )
    _NUCLEAR_UNIT_LINE = re.compile(
        r"^\s*(?:DIPARTIMENTO|U\.?O\.?|UNIT[AÀ]\s+OPERATIVA|"
        r"STRUTTURA\s+(?:COMPLESSA|SEMPLICE)|S\.?C\.?)\s*"
        r"[^.\n]{0,90}\bMEDICINA\s+NUCLEARE\b",
        re.IGNORECASE | re.MULTILINE,
    )
    _IMAGING_MODALITY = re.compile(
        r"\b(?:TC|TAC|RMN?|RX|ECOGRAFIA|ECOTOMOGRAFIA|MAMMOGRAFIA|"
        r"TOMOGRAFIA\s+COMPUTERIZZATA|RISONANZA\s+MAGNETICA|"
        r"RADIOGRAFIA|ANGIOGRAFIA|PIELOGRAFIA|UROGRAFIA|"
        r"COLANGIOGRAFIA|FLEBOGRAFIA|FLUOROSCOPIC\w*)\b",
        re.IGNORECASE,
    )
    # A radiotherapy unit writes radiotherapy notes: the unit name is
    # prefixed (UO RADIOTERAPIA, AMB. RADIOTERAPICO, DSA RADIOTERAPIA), so
    # a visit that merely *schedules* radiotherapy does not match.
    _RADIOTHERAPY_UNIT = re.compile(
        r"\b(?:U\.?O\.?|AMB\.?|DAY\s+SERVICE|DIPARTIMENTO|"
        r"UNIT[AÀ]\s+OPERATIVA)\s*[^.\n]{0,60}\bRADIOTERAPIA\b|"
        r"\bAMB\.\s+RADIOTERAPICO\b|\bDSA\s+RADIOTERAPIA\b",
        re.IGNORECASE,
    )
    # Letterhead of a non-oncology clinical unit: an ambulatory visit or
    # exam issued by that unit must not become an oncological visit just
    # because the patient's history is full of oncology terms or the
    # trust's oncology department name appears in the letterhead line.
    _SPECIALIST_UNIT = re.compile(
        r"\b(?:U\.?O\.?|AMB\.?|UNIT[AÀ]\s+OPERATIVA|DAY\s+SERVICE|"
        r"DIPARTIMENTO)\s*[^.\n]{0,60}\b(?:DERMATOLOGIA|"
        r"VIDEODERMATOSCOPIA|CHIRURGICO\s+DERMATOLOGICO|"
        r"CHIRURGIA|SENOLOGIC\w*|"
        r"CARDIOLOGIA|ECOCARDIOGRAFIA|ECOCARDIACA|"
        r"O\.?R\.?L\.?)\b|"
        r"\bECO(?:COLOR)?DOPPLERGRAFIA\s+CARDIACA\b|"
        r"\bAMB\.\s+VIDEODERMATOSCOPIA\b|\bAMB\.\s+CHIRURGICO\b|"
        r"\bVALUTAZIONE\s+ANESTESIOLOGICA\s+PREOPERATORIA\b",
        re.IGNORECASE,
    )
    # Documents that must keep their own type even when a specialist unit
    # line appears in the head (a discharge letter from a surgical ward
    # mentions the operating room in its decorso, an histology report names
    # the requesting surgical unit, ...).
    _LETTERHEAD_GUARD = re.compile(
        r"\bLETTERA\s+DI\s+DIMISSIONE\b|\bSCHEDA\s+DI\s+DIMISSIONE\s+"
        r"OSPEDALIERA\b|\bCARTELLA\s+CLINICA\b|\bMOTIVO\s+DEL\s+RICOVERO\b|"
        r"\bDIARIO\s+(?:MEDICO|INFERMIERISTICO)\b|"
        r"\bANATOMIA\s+PATOLOGICA\b|\bESAME\s+ISTOLOGICO\b|"
        r"\bLABORATORIO\s+DI\s+ANALISI\b|\bREFERTO\s+ANATOMO.?PATOLOGICO\b",
        re.IGNORECASE,
    )
    # Emergency-department sheets: the triage block is unmistakable.
    _PS_SHEET = re.compile(
        r"\bPRONTO\s+SOCCORSO\b",
        re.IGNORECASE,
    )
    _PS_SHEET_SUPPORT = re.compile(
        r"\bDATI\s+ACCETTAZIONE\b|\bTRIAGE\b|\bMEZZO\s+TRASPORTO\b|"
        r"\bDIPARTIMENTO\s+DI\s+EMERGENZA\b|\bDATI\s+EPISODIO\b|"
        r"\bMED\.?\s+D['\s]URGENZA\b",
        re.IGNORECASE,
    )
    # Surgical intervention sheets ("Blocco Operatorio", "Sala Operatoria").
    _OP_BLOCK = re.compile(
        r"\b(?:BLOCCO\s+OPERATORIO|SALA\s+OPERATORIA)\b",
        re.IGNORECASE,
    )
    # Specialist visit markers that a letterhead unit document carries.
    _SPECIALIST_VISIT_MARKER = re.compile(
        r"\b(?:VISITA|AMBULATORIALE|PRESTAZIONI\s+EROGATE|REFERTO|"
        r"MEDICAZIONE|CONTROLLO\s+AMBULATORIALE|ESAME\s+OBIETTIVO)\b",
        re.IGNORECASE,
    )
    _NUCLEAR_MODALITY = re.compile(
        r"\b(?:PET(?:[-\s]?(?:TC|FDG))?|FDG|SCINTIGRAFIA)\b",
        re.IGNORECASE,
    )
    # Report-structural phrases that accompany the exam title on the first
    # page of a radiology report.  Deliberately strict: a visit note that
    # quotes a past exam ("TC total body co mdc: ...") carries the modality
    # and the contrast abbreviation but none of these phrases.
    _IMAGING_REPORT_SUPPORT = re.compile(
        r"\bEsame eseguito\b|\bId\s+Dicom\b|\bDICOM\b|\btomografo\b|"
        r"\bPrestazioni eseguite\b",
        re.IGNORECASE,
    )

    def classify(
        self,
        text: str,
        filename: str = "",
        header_metadata: Optional[dict] = None,
    ) -> str:
        """
        Classify a document into one or more DocumentType categories.
        Returns the primary type as a string.
        """
        header_metadata = header_metadata or {}

        # The provenance names the requesting unit ("DAY SERVICE
        # ONCOLOGIA"), not the unit that produced the document — it must
        # not feed the lexical scores, or every imaging report ordered by
        # oncology would gain 5 points for `visita_oncologica`.
        header_text = "\n".join(
            str(value)
            for value in (
                header_metadata.get("department"),
                *(header_metadata.get("services") or []),
            )
            if value
        )

        combined_text = (
            f"{header_text}\n{text}" if header_text else text
        )

        # A document that is structurally a dedicated lab result sheet wins
        # over a "VISITA DI CONTROLLO" header hint — otherwise the lab parser
        # never runs on the values. The detector is conservative: it backs
        # off on discharge letters, clinical charts, and specialist visits.
        if self._looks_like_lab_result_sheet(combined_text):
            return DocumentType.LABORATORIO.value

        # The issuing unit's letterhead decides for specialist units: a
        # radiotherapy note or a dermatology visit keeps its type even
        # when the header hint (which trusts the stored specialty) would
        # say "oncological visit" — the trust's oncology department name
        # inside the letterhead line inflates that specialty.
        letterhead_type = self._letterhead_specialist_type(
            combined_text, header_metadata
        )
        if letterhead_type:
            return letterhead_type

        # An explicit performance listed in the first-page header is more
        # authoritative than diagnoses mentioned in the body. For example,
        # an oncological history must not turn a cardiology visit into an
        # oncology visit.
        header_hint = header_metadata.get("document_type_hint")
        if header_hint in {item.value for item in DocumentType}:
            return header_hint

        # A report issued by a radiology / nuclear-medicine unit is an
        # imaging report even when the patient's oncology keywords dominate
        # the body — a CT scan of a melanoma patient is not an oncological
        # visit. The unit's letterhead is decisive; the exam modality picks
        # between radiology and nuclear medicine.
        imaging_type = self._imaging_report_type(combined_text, header_metadata)
        if imaging_type:
            return imaging_type

        # Accumulate weighted scores from every signal source.
        scores: dict[str, float] = {}

        # Signal 1: Filename-based (fixed weight per match)
        if filename:
            fn_type = self._classify_filename(filename)
            if fn_type:
                scores[fn_type] = scores.get(fn_type, 0) + 4.0

        # Signal 2: Lexical keywords — use the raw weighted scores
        # returned by _classify_text_scored (not just presence/absence).
        for doc_type, weight in self._classify_text_scored(combined_text):
            scores[doc_type] = scores.get(doc_type, 0) + weight

        # Signal 3: Structural cues (tables, sections)
        struct_type = self._classify_structure(text)
        if struct_type:
            scores[struct_type] = scores.get(struct_type, 0) + 3.0

        if not scores:
            return DocumentType.NON_CLASSIFICATO.value

        # Return the type with the highest weighted score
        primary = max(scores, key=scores.get)
        return primary

    @classmethod
    def _letterhead_specialist_type(
        cls, text: str, header_metadata: dict
    ) -> Optional[str]:
        """The issuing unit's letterhead decides for specialist units.

        A radiotherapy session note, a dermatology / surgery / cardiology
        ambulatory visit, an emergency sheet or an operating-block sheet
        must not become an oncological visit just because the patient's
        history is full of oncology terms.  The guard (checked on the
        letterhead window only, so quoted histology reports in a visit's
        anamnesis do not block the rule) keeps discharge letters, clinical
        charts, histology and lab documents with their own type.
        """
        head = text[:800]
        if cls._RADIOLOGY_UNIT_LINE.search(text[:500]) or cls._NUCLEAR_UNIT_LINE.search(text[:500]):
            # The imaging short-circuit below owns radiology units.
            return None
        if cls._LETTERHEAD_GUARD.search(text[:300]):
            return None
        department = str(header_metadata.get("department") or "")
        if (
            cls._RADIOTHERAPY_UNIT.search(head)
            or "radioterap" in department.casefold()
        ):
            return DocumentType.RADIOTERAPIA.value
        if cls._OP_BLOCK.search(head):
            return DocumentType.VERBALE_OPERATORIO.value
        if cls._PS_SHEET.search(head) and cls._PS_SHEET_SUPPORT.search(head):
            return DocumentType.PRONTO_SOCCORSO.value
        if cls._SPECIALIST_UNIT.search(head):
            return DocumentType.VISITA_SPECIALISTICA.value
        # Fallback on the stored specialty when the letterhead line in the
        # text was not matched (line splits, OCR noise).  Documents with
        # an imaging exam title (ultrasound ambulatories, ...) keep the
        # imaging type: the title boost below handles them.
        non_oncology_specialties = {
            "dermatologia", "chirurgia", "cardiologia", "urologia",
            "ortopedia", "oculistica", "endocrinologia",
            "gastroenterologia", "pneumologia", "neurologia",
            "reumatologia", "otorinolaringoiatria", "ginecologia",
        }
        _first_1500 = text[:1500] if len(text) >= 1500 else text
        is_imaging = (
            cls._IMAGING_MODALITY.search(_first_1500)
            and cls._IMAGING_REPORT_SUPPORT.search(_first_1500)
        )
        if (
            header_metadata.get("specialty") in non_oncology_specialties
            and cls._SPECIALIST_VISIT_MARKER.search(head)
            and not is_imaging
        ):
            return DocumentType.VISITA_SPECIALISTICA.value
        return None

    @classmethod
    def _imaging_report_type(
        cls, text: str, header_metadata: dict
    ) -> Optional[str]:
        """Return the imaging type when the issuing unit is a radiology /
        nuclear-medicine department and the text names an exam modality.

        The unit is read from the stored header metadata first; when it is
        missing there, the document's own letterhead lines are scanned
        ("Dipartimento ... di Radiologia", "STRUTTURA COMPLESSA DI MEDICINA
        NUCLEARE") so partial metadata does not hide the strongest signal.
        """
        department = " ".join(
            str(value)
            for value in (
                header_metadata.get("department"),
                header_metadata.get("specialty"),
            )
            if value
        )
        head = text[:800]
        unit_is_radiology = bool(
            cls._RADIOLOGY_UNIT.search(department)
            or cls._RADIOLOGY_UNIT_LINE.search(head)
            or header_metadata.get("specialty") == "radiologia"
        )
        unit_is_nuclear = bool(
            cls._NUCLEAR_UNIT.search(department)
            or cls._NUCLEAR_UNIT_LINE.search(head)
            or header_metadata.get("specialty") == "medicina_nucleare"
        )
        if not (unit_is_radiology or unit_is_nuclear):
            return None
        has_imaging = bool(cls._IMAGING_MODALITY.search(text))
        has_nuclear = bool(cls._NUCLEAR_MODALITY.search(text))
        if not (has_imaging or has_nuclear):
            return None
        if unit_is_nuclear and has_nuclear and not unit_is_radiology:
            return DocumentType.MEDICINA_NUCLEARE.value
        return DocumentType.RADIOLOGIA.value

    @staticmethod
    def _looks_like_lab_result_sheet(text: str) -> bool:
        """Recognize a dedicated result sheet even if ordered by Oncology.

        The short-circuit is intentionally conservative: it backs off when the
        document contains clear signals of another document type (discharge
        letter, clinical chart, specialist visit, etc.) so that a discharge
        letter that merely *includes* lab results is not misclassified.
        """
        # ---- Blocking check: bail out when the text is clearly NOT a
        #      standalone lab sheet -------------------------------------------
        _non_lab_headers = (
            r"\bLETTERA\s+DI\s+DIMISSIONE\b",
            r"\bDIMISSIONE\s+(?:PROTETTA|ORDINARIA|OSPEDALIERA)\b",
            r"\bMOTIVO\s+DEL\s+RICOVERO\b",
            r"\bDECORSO\s+CLINICO\b",
            r"\bDIAGNOSI\s+(?:DI\s+)?(?:INGRESSO|DIMISSIONE)\b",
            r"\bTERAPIA\s+(?:ALLA|PRESCRITTA\s+ALLA)\s+DIMISSIONE\b",
            r"\bANAMNESI\s+(?:REMOTA|PATOLOGICA|FARMACOLOGICA|ONCOLOGICA)\b",
            r"\bESAME\s+OBIETTIVO\b",
            r"\bCONSULENZA\s+SPECIALISTICA\b",
            r"\bPIANO\s+DI\s+FOLLOW[-\s]?UP\b",
            r"\bCARTELLA\s+CLINICA\b",
            r"\bREPARTO\s+(?:DI\s+)?(?:DEGENZA|POST[-\s]?ACUTI|LUNGODEGENZA|RIABILITAZIONE)\b",
        )
        non_lab_hits = sum(
            bool(re.search(pattern, text, re.IGNORECASE))
            for pattern in _non_lab_headers
        )
        if non_lab_hits >= 2:
            return False

        # ---- Lab sheet markers ---------------------------------------------
        report_markers = (
            r"\bESAME\s+ESITO\s+U\.?M\.?\s+INTERVALLI?\s+RIFERIMENTO\b",
            r"\bRISULTATI?\s+VALIDAT[IO]\b",
            r"\bREFERTO\s+COMPLETO\b",
            r"\bMATERIALE\s*:\s*(?:SIERO|PLASMA|SANGUE|URIN[AE])\b",
            r"\bLABORATORIO\s+(?:UNICO|DI\s+ANALISI)\b",
        )
        marker_count = sum(
            bool(re.search(pattern, text, re.IGNORECASE))
            for pattern in report_markers
        )
        # Require at least one marker — the blocking check above already
        # prevents discharge letters and clinical charts from reaching
        # this point, so a single marker is sufficient evidence.
        if marker_count < 1:
            return False

        # When only one marker is present, require at least two result
        # lines — a single numeric line could be an incidental match.
        min_result_lines = 1 if marker_count >= 2 else 2
        result_lines = len(re.findall(
            r"(?im)^\s*(?:\[\d+\]\s*)?"
            r"[A-ZÀ-Ü][A-ZÀ-Ü0-9 .()/%+\-:]{1,60}:?\s+"
            r"(?:\d+(?:[.,]\d+)?\s*(?:\*+\s*)?"
            r"(?:mg/dl|g/dl|mmol/l|u/l|ng/ml|ng/l|pg/ml|fl|"
            r"%|x10\^?[36]/[µμu]l|inr|ratio)\b|"
            r"(?:NEGATIV[OA]|POSITIV[OA]|ASSENTE|PRESENTE|"
            r"NON\s+RIVELAT[OA]|DEBOLE)\b)",
            text,
            re.IGNORECASE,
        ))
        return result_lines >= min_result_lines

    def _classify_filename(self, filename: str) -> Optional[str]:
        """Classify by filename patterns."""
        fn = filename.lower()

        patterns = [
            (["lab", "laboratorio", "esami", "analisi", "emocromo",
              "biochimica", "ematologia", "coagulazione", "sierologia",
              "immunoenzimatica", "elettroforesi"],
             DocumentType.LABORATORIO),
            (["tac", "rm", "rx", "eco", "radiologia", "mammo", "tomografia",
              "risonanza", "scintigrafia", "pet", "angiografia"],
             DocumentType.RADIOLOGIA),
            (["dimissione", "lettera_dimissione", "lettera di dimissione",
              "dimissione_protetta", "dimissione_ordinaria",
              "post_acuti", "pre_acuti", "postacuti", "preacuti",
              "lungodegenza", "lungo_degenza",
              "riabilitazione", "cure_intermedie"],
             DocumentType.LETTERA_DIMISSIONE),
            (["sdo", "scheda_dimissione"],
             DocumentType.SDO),
            (["cartella", "ricovero", "clinica", "degenza"],
             DocumentType.CARTELLA_CLINICA),
            (["diario_medico", "diario medico", "decorso"],
             DocumentType.DIARIO_MEDICO),
            (["diario_inf", "infermieristic", "note_inf"],
             DocumentType.DIARIO_INFERMIERISTICO),
            (["terapi", "piano_terapeutico", "prescrizione"],
             DocumentType.PIANO_TERAPEUTICO),
            (["visita_onco", "oncology", "oncologica"],
             DocumentType.VISITA_ONCOLOGICA),
            (["visita", "ambulatori", "specialist", "consulenza"],
             DocumentType.VISITA_SPECIALISTICA),
            (["verbale_op", "operatorio", "chirurgia", "intervento"],
             DocumentType.VERBALE_OPERATORIO),
            (["pronto_soccorso", "ps ", "triage"],
             DocumentType.PRONTO_SOCCORSO),
            (["consulenza", "parere"],
             DocumentType.CONSULENZA),
            (["anatomia_patologica", "istologico", "biopsia", "citologico"],
             DocumentType.ANATOMIA_PATOLOGICA),
        ]

        for keywords, doc_type in patterns:
            if any(kw in fn for kw in keywords):
                return doc_type.value

        return None

    def _classify_text_scored(self, text: str) -> list[tuple[str, float]]:
        """Classify by lexical patterns — returns (type, score) pairs."""
        # Define keyword groups with their document type
        keyword_groups = [
            (RADIOLOGY_KEYWORDS, DocumentType.RADIOLOGIA.value, 1),
            (LAB_KEYWORDS, DocumentType.LABORATORIO.value, 1),
            (HOSPITALIZATION_KEYWORDS, DocumentType.CARTELLA_CLINICA.value, 1),
            (VISIT_KEYWORDS, DocumentType.VISITA_SPECIALISTICA.value, 1),
            # Microbiology / lab tests
            ([r'\bANTIBIOGRAMMA\b', r'\bCOLTURA\b', r'\bCOLTURALE\b',
              r'\bMICROBIOLOGIA\b', r'\bBATTERIOLOGICO\b', r'\bESAME\s+COLTURALE\b',
              r'\bTAMPONE\b', r'\bURINOCOLTURA\b', r'\bEMOCOLTURA\b', r'\bCOPROCOLTURA\b',
              r'\bANTIBIOTICO\b', r'\bSENSIBILITÀ\b', r'\bSENSIBILE\b', r'\bRESISTENTE\b',
              r'\bMIC\b', r'\bCARICA\s+BATTERICA\b', r'\bGERME\s+ISOLATO\b',
              r'\bSTAFILOCOCCO\b', r'\bSTREPTOCOCCO\b', r'\bESCHERICHIA\b',
              r'\bPSEUDOMONAS\b', r'\bCANDIDA\b', r'\bENTEROBATTERI\b',
              r'\bGRAM\s*[+-]\b', r'\bSIEROLOGIA\b', r'\bIMMUNOENZIMATICA\b',
              r'\bELETTROFORESI\b', r'\bPCR\b'],
             DocumentType.LABORATORIO.value, 4),   # HIGH WEIGHT - very specific

            # Additional fine-grained groups
            ([r'\bVISITA\s+ONCOLOGICA\b', r'\bONCOLOGIA\b', r'\bMELANOMA\b',
              r'\bCARCINOMA\b', r'\bTUMORE\b', r'\bNEOPLASIA\b', r'\bMETASTASI\b',
              r'\bSTADIAZIONE\b', r'\bTERAPIA\s+ONCOLOGICA\b', r'\bCHEMIOTERAPIA\b',
              r'\bIMMUNOTERAPIA\b', r'\bNIVOLUMAB\b', r'\bPEMBROLIZUMAB\b',
              r'\bIPILIMUMAB\b', r'\bTARGET\s+THERAPY\b', r'\bBIOLOGICO\b',
              r'\bONCOLOGICO\b', r'\bONCO\s+EMATOLOGIA\b', r'\bDAY\s+SERVICE\s+ONCOLOGIA\b',
              r'\bDH\s+ONCOLOGICO\b', r'\bSTADIO\s+III', r'\bSTADIO\s+IV\b',
              r'\bMALATTIA\s+ONCOLOGICA\b', r'\bNEOPLASTICO\b', r'\bNEOPLASTICA\b'],
             DocumentType.VISITA_ONCOLOGICA.value, 5),   # HIGHER weight to beat pathology
            ([r'\bDIMISSIONE\b', r'\bDIMESSO\b', r'\bLETTERA\s+DI\s+DIMISSIONE\b',
              r'\bDIMISSIONE\s+PROTETTA\b', r'\bDIMISSIONE\s+ORDINARIA\b',
              r'\bDIMISSIONE\s+DA\s+REPARTO\b',
              r'\bMOTIVO\s+DEL\s+RICOVERO\b', r'\bDECORSO\s+CLINICO\b',
              r'\bDIAGNOSI\s+(?:DI\s+)?(?:INGRESSO|DIMISSIONE)\b',
              r'\bTERAPIA\s+(?:ALLA|PRESCRITTA\s+ALLA)\s+DIMISSIONE\b',
              r'\bOUTCOME\s+ALLA\s+DIMISSIONE\b',
              r'\bDESTINAZIONE\s+ALLA\s+DIMISSIONE\b',
              r'\bPIANO\s+DI\s+FOLLOW[-\s]?UP\b'],
             DocumentType.LETTERA_DIMISSIONE.value, 3),  # Raised from 2
            # Pre-acute / post-acute / rehabilitation wards
            ([r'\bREPARTO\s+(?:POST[-\s]?ACUTI|PRE[-\s]?ACUTI)\b',
              r'\bLUNGODEGENZA\b', r'\bLUNGO[-\s]?DEGENZA\b',
              r'\bRIABILITAZIONE\s+(?:INTENSIVA|ESTENSIVA|MOTORIA|CARDIOLOGICA|'
              r'NEUROLOGICA|RESPIRATORIA|GERIATRICA)\b',
              r'\bPOST[-\s]?ACUTI\b', r'\bPRE[-\s]?ACUTI\b',
              r'\bDEGENZA\s+(?:POST[-\s]?ACUTI|PRE[-\s]?ACUTI)\b',
              r'\bCURE\s+INTERMEDIE\b', r'\bREPARTO\s+RIABILITATIVO\b',
              r'\bMEDICINA\s+RIABILITATIVA\b',
              r'\bUNIT[AÀ]\s+SPINALE\b', r'\bUNIT[AÀ]\s+SPINALI\b',
              r'\bUNIT[AÀ]\s+GRAVI\s+CEREBROLESIONI\b'],
             DocumentType.LETTERA_DIMISSIONE.value, 3),
            ([r'\bSDO\b', r'\bSCHEDA\s+DI\s+DIMISSIONE\s+OSPEDALIERA\b'],
             DocumentType.SDO.value, 3),
            ([r'\bDIARIO\s+MEDICO\b', r'\bDECORSO\s+CLINICO\b'],
             DocumentType.DIARIO_MEDICO.value, 2),
            ([r'\bDIARIO\s+INFERMIERISTICO\b', r'\bNOTE\s+INFERMIERISTICHE\b'],
             DocumentType.DIARIO_INFERMIERISTICO.value, 2),
            ([r'\bPIANO\s+TERAPEUTICO\b', r'\bPROGRAMMA\s+TERAPEUTICO\b',
              r'\bSCHEMA\s+TERAPEUTICO\b'],
             DocumentType.PIANO_TERAPEUTICO.value, 2),
            ([r'\bPRONTO\s+SOCCORSO\b', r'\bTRIAGE\b', r'\bCODICE\s+(?:BIANCO|VERDE|GIALLO|ROSSO)\b'],
             DocumentType.PRONTO_SOCCORSO.value, 3),
            ([r'\bVERBALE\s+OPERATORIO\b', r'\bINTERVENTO\s+CHIRURGICO\b',
              r'\bCHIRURGIA\b', r'\bOPERAZIONE\b'],
             DocumentType.VERBALE_OPERATORIO.value, 2),
            ([r'\bCONSULENZA\b', r'\bPARERE\s+SPECIALISTICO\b',
              r'\bCONSULENZA\s+SPECIALISTICA\b'],
             DocumentType.CONSULENZA.value, 2),
            ([r'\bANATOMIA\s+PATOLOGICA\b', r'\bISTOLOGICO\b', r'\bBIOPTICO\b',
              r'\bBIOPSIA\b', r'\bCITOLOGICO\b', r'\bPUNCH\s+BIOPSY\b',
              r'\bANALISI\s+IMMUNOISTOCHIMICA\b', r'\bIMMUNOISTOCHIMICA\b',
              r'\bANALISI\s+MUTAZIONALE\b', r'\bMUTAZIONE\s+(?:GENICA|DEL\s+GENE)\b',
              r'\bSEQUENZIAMENTO\b', r'\bBIOLOGIA\s+MOLECOLARE\b',
              r'\bREFERTO\s+ANATOMO.?PATOLOGICO\b', r'\bESAME\s+ISTOLOGICO\b',
              r'\bESAME\s+CITOLOGICO\b', r'\bAGOASPIRATO\b', r'\bAGO\s+ASPIRATO\b',
              r'\bPUNCH\b', r'\bESCISSIONE\b', r'\bRESEZIONE\b',
              r'\bIMMUNOFENOTIPO\b', r'\bIMMUNOISTOCHIMICO\b',
              r'\bBRAF\b', r'\bPD.?L1\b', r'\bPDL1\b', r'\bEGFR\b',
              r'\bKRAS\b', r'\bNRAS\b', r'\bHER2\b', r'\bALK\b',
              r'\bROS1\b', r'\bMICROSATELLITI\b', r'\bMSI\b',
              r'\bDESCRIZIONE\s+MACROSCOPICA\b', r'\bDESCRIZIONE\s+MICROSCOPICA\b',
              r'\bDIAGNOSI\s+ISTOLOGICA\b', r'\bDIAGNOSI\s+ANATOMO.?PATOLOGICA\b',
              r'\bCLASSIFICAZIONE\s+WHO\b', r'\bGRADING\b', r'\bSTADIAZIONE\s+PT\b',
              r'\bMARGINI\s+DI\s+RESEZIONE\b', r'\bLINFONODI\s+ESAMINATI\b',
              r'\bEMATOSSILINA\b', r'\bEOSINA\b', r'\bCOLORAZIONE\b',
              r'\bCAMPIONE\s+ISTOLOGICO\b', r'\bCAMPIONE\s+CITOLOGICO\b',
              r'\bTESSUTO\s+(?:FISSATO|INCLUSO|IN\s+PARAFFINA)\b',
              r'\bCELLULE\s+(?:TUMORALI|NEOPLASTICHE|ATIPICHE)\b'],
             DocumentType.ANATOMIA_PATOLOGICA.value, 4),  # HIGH WEIGHT - very specific
            ([r'\bMEDICAZIONE\b', r'\bUSTIONI\b', r'\bMEDICAZIONE\s+USTIONI\b',
              r'\bAMBULATORIO\b', r'\bDERMATOLOGIA\b', r'\bAMBULATORIALE\b',
              r'\bVISITA\s+(?:OCULISTICA|DERMATOLOGICA|CARDIOLOGICA|ORTOPEDICA|'
              r'NEUROLOGICA|UROLOGICA|GINECOLOGICA|OTORINO|PNEUMOLOGICA|'
              r'ENDOCRINOLOGICA|GASTROENTEROLOGICA|DIETOLOGICA|FISIATRICA|'
              r'DI\s+CONTROLLO)\b',
              r'\bOCULISTICA\b', r'\bFLUORANGIOGRAFIA\b', r'\bOCT\b', r'\bFOO\b',
              r'\bCONTROLLO\s+AMBULATORIALE\b', r'\bPRESTAZIONI\s+EROGATE\b',
              r'\bPRIMA\s+VISITA\b', r'\bVISITA\s+DI\s+CONTROLLO\b'],
             DocumentType.VISITA_SPECIALISTICA.value, 2),
        ]

        # Score each document type
        scores: dict[str, float] = {}
        for keywords, doc_type, weight in keyword_groups:
            count = sum(
                1 for kw in keywords
                if re.search(kw, text, re.IGNORECASE)
            )
            if count > 0:
                scores[doc_type] = scores.get(doc_type, 0) + count * weight

        # ---- Heading bonus: when the document opens with a clear type
        #      marker, that type gets a large fixed boost.  This prevents
        #      incidental body-text keywords from overriding the document's
        #      own heading (e.g. a discharge letter that happens to mention
        #      oncology drugs must still be classified as a discharge letter).
        _first_300 = text[:300] if len(text) >= 300 else text
        _heading_boosts = [
            (r"\bLETTERA\s+DI\s+DIMISSIONE\b", DocumentType.LETTERA_DIMISSIONE.value, 60),
            (r"\bDIMISSIONE\s+(?:PROTETTA|ORDINARIA)\b", DocumentType.LETTERA_DIMISSIONE.value, 60),
            (r"\bSCHEDA\s+DI\s+DIMISSIONE\s+OSPEDALIERA\b", DocumentType.SDO.value, 60),
            (r"\bREFERTO\s+(?:RADIOLOGICO|RADIOLOGIA)\b", DocumentType.RADIOLOGIA.value, 40),
            (r"\bESAME\s+ESITO\s+U\.?M\.?\s+INTERVALLI?\s+RIFERIMENTO\b", DocumentType.LABORATORIO.value, 40),
            (r"\bVERBALE\s+OPERATORIO\b", DocumentType.VERBALE_OPERATORIO.value, 40),
            (r"\bPRONTO\s+SOCCORSO\b", DocumentType.PRONTO_SOCCORSO.value, 40),
            (r"\bDIARIO\s+(?:MEDICO|INFERMIERISTICO)\b", DocumentType.DIARIO_MEDICO.value, 40),
            (r"\bCARTELLA\s+CLINICA\b", DocumentType.CARTELLA_CLINICA.value, 40),
        ]
        for pattern, doc_type, boost in _heading_boosts:
            if re.search(pattern, _first_300, re.IGNORECASE):
                scores[doc_type] = scores.get(doc_type, 0) + boost
                break  # Only the first matching heading counts

        # ---- Imaging exam-title boost: the exam title often sits after a
        #      long institutional letterhead, well beyond the 300-char
        #      heading window. A modality keyword plus a report-structural
        #      phrase on the first page marks a radiology / nuclear-medicine
        #      report; a visit note that merely mentions an upcoming exam
        #      has the modality without the structural phrase.
        _first_1500 = text[:1500] if len(text) >= 1500 else text
        if self._IMAGING_MODALITY.search(
            _first_1500
        ) and self._IMAGING_REPORT_SUPPORT.search(_first_1500):
            scores[DocumentType.RADIOLOGIA.value] = (
                scores.get(DocumentType.RADIOLOGIA.value, 0) + 40
            )
        elif self._NUCLEAR_MODALITY.search(
            _first_1500
        ) and self._IMAGING_REPORT_SUPPORT.search(_first_1500):
            scores[DocumentType.MEDICINA_NUCLEARE.value] = (
                scores.get(DocumentType.MEDICINA_NUCLEARE.value, 0) + 40
            )

        if not scores:
            return []

        # Return types sorted by score (highest first)
        sorted_types = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Only return types that are clearly above noise
        # (at least 2 points or the top result)
        threshold = max(2, sorted_types[0][1] * 0.5)
        return [(t, s) for t, s in sorted_types if s >= threshold]

    def _classify_text(self, text: str) -> list[str]:
        """Backward-compatible wrapper — returns only type names."""
        return [t for t, _s in self._classify_text_scored(text)]

    def _classify_structure(self, text: str) -> Optional[str]:
        """Classify by structural properties of the text."""
        # Lab reports: have many numeric patterns with units
        numeric_unit_pattern = re.compile(
            r'[\d.,]+\s*(?:mg/dL|g/dL|mmol/L|U/L|UI/L|mEq/L|ng/mL|fL|%|mm/h)',
            re.IGNORECASE,
        )
        numeric_count = len(numeric_unit_pattern.findall(text))

        # Lab reports: typical ratio of numeric lines
        lines = text.split('\n')
        numeric_lines = sum(
            1 for line in lines
            if re.search(r'\b\d{1,4}[.,]\d{1,3}\b', line)
        )

        # If >30% of lines contain numbers with units, likely a lab report
        if len(lines) > 5 and numeric_lines / max(len(lines), 1) > 0.25:
            return DocumentType.LABORATORIO.value

        return None

    def get_all_candidates(self, text: str, filename: str = "") -> list[tuple[str, float]]:
        """
        Return all possible document types with confidence scores.
        Used for the import dialog to suggest types.
        """
        # Reuse the scored approach from classify()
        text_types = self._classify_text(text)
        filename_types = []
        if filename:
            fn_result = self._classify_filename(filename)
            if fn_result:
                filename_types = [fn_result]

        # Combine: text-based has more weight
        all_types = text_types + filename_types
        if not all_types:
            return [(DocumentType.NON_CLASSIFICATO.value, 0.3)]

        # Remove duplicates while preserving order
        seen = set()
        unique = []
        for t in all_types:
            if t not in seen:
                seen.add(t)
                unique.append(t)

        # Score: if text analysis found it, higher confidence
        result = []
        for i, t in enumerate(unique):
            if i == 0:
                conf = 0.85
            elif i == 1:
                conf = 0.65
            elif i == 2:
                conf = 0.50
            else:
                conf = 0.35
            # Top text-based result gets boost
            if t in text_types and t == text_types[0] if text_types else False:
                conf = 0.90
            result.append((t, conf))

        return result

    # ------------------------------------------------------------------
    # Pre-acute / post-acute discharge letter detection
    # ------------------------------------------------------------------

    _PRE_ACUTE_PATTERNS: list[re.Pattern] = [
        re.compile(pat, re.IGNORECASE)
        for pat in [
            r"\bREPARTO\s+(?:POST[-\s]?ACUTI|PRE[-\s]?ACUTI)\b",
            r"\bLUNGODEGENZA\b", r"\bLUNGO[-\s]?DEGENZA\b",
            r"\bRIABILITAZIONE\s+(?:INTENSIVA|ESTENSIVA|MOTORIA|"
            r"CARDIOLOGICA|NEUROLOGICA|RESPIRATORIA|GERIATRICA)\b",
            r"\bPOST[-\s]?ACUTI\b", r"\bPRE[-\s]?ACUTI\b",
            r"\bDEGENZA\s+(?:POST[-\s]?ACUTI|PRE[-\s]?ACUTI)\b",
            r"\bCURE\s+INTERMEDIE\b", r"\bREPARTO\s+RIABILITATIVO\b",
            r"\bMEDICINA\s+RIABILITATIVA\b",
            r"\bUNIT[AÀ]\s+SPINALE\b",
            r"\bUNIT[AÀ]\s+GRAVI\s+CEREBROLESIONI\b",
        ]
    ]

    @classmethod
    def is_pre_acute_discharge(cls, text: str) -> bool:
        """Return True when *text* looks like a discharge letter from a
        pre-acute / post-acute / rehabilitation ward."""
        return any(
            pat.search(text) for pat in cls._PRE_ACUTE_PATTERNS
        )
