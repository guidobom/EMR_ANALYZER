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
        signals = []
        header_metadata = header_metadata or {}

        # An explicit performance listed in the first-page header is more
        # authoritative than diagnoses mentioned in the body. For example,
        # an oncological history must not turn a cardiology visit into an
        # oncology visit.
        header_hint = header_metadata.get("document_type_hint")
        if header_hint in {item.value for item in DocumentType}:
            return header_hint

        header_text = "\n".join(
            str(value)
            for value in (
                header_metadata.get("department"),
                header_metadata.get("provenance"),
                *(header_metadata.get("services") or []),
            )
            if value
        )

        combined_text = (
            f"{header_text}\n{text}" if header_text else text
        )
        if self._looks_like_lab_result_sheet(combined_text):
            return DocumentType.LABORATORIO.value

        # Signal 1: Filename-based
        if filename:
            fn_type = self._classify_filename(filename)
            if fn_type:
                signals.append(("filename", fn_type))

        # Signal 2: Lexical keywords in text
        text_types = self._classify_text(combined_text)
        signals.extend(("text", t) for t in text_types)

        # Signal 3: Structural cues (tables, sections)
        struct_type = self._classify_structure(text)
        if struct_type:
            signals.append(("structure", struct_type))

        # Determine primary type
        if not signals:
            return DocumentType.NON_CLASSIFICATO.value

        # Count occurrences of each type
        type_counts = {}
        for source, doc_type in signals:
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1

        # Return the most frequent type
        primary = max(type_counts, key=type_counts.get)
        return primary

    @staticmethod
    def _looks_like_lab_result_sheet(text: str) -> bool:
        """Recognize a dedicated result sheet even if ordered by Oncology."""
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
        result_line = re.search(
            r"(?im)^\s*(?:\[\d+\]\s*)?"
            r"[A-ZÀ-Ü][A-ZÀ-Ü0-9 .()/%+-]{1,60}:?\s+"
            r"\d+(?:[.,]\d+)?\s*(?:\*+\s*)?"
            r"(?:mg/dl|g/dl|mmol/l|u/l|ng/ml|ng/l|pg/ml|fl|"
            r"%|x10\^?[36]/[µμu]l|inr|ratio)\b",
            text,
            re.IGNORECASE,
        )
        return marker_count >= 2 and result_line is not None

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
            (["dimissione", "lettera_dimissione", "lettera di dimissione"],
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

    def _classify_text(self, text: str) -> list[str]:
        """Classify by lexical patterns in the text content — scored approach."""
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
            ([r'\bDIMISSIONE\b', r'\bDIMESSO\b', r'\bLETTERA\s+DI\s+DIMISSIONE\b'],
             DocumentType.LETTERA_DIMISSIONE.value, 2),
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
        scores = {}
        for keywords, doc_type, weight in keyword_groups:
            count = sum(
                1 for kw in keywords
                if re.search(kw, text, re.IGNORECASE)
            )
            if count > 0:
                scores[doc_type] = scores.get(doc_type, 0) + count * weight

        if not scores:
            return []

        # Return types sorted by score (highest first)
        sorted_types = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Only return types that are clearly above noise
        # (at least 2 points or the top result)
        threshold = max(2, sorted_types[0][1] * 0.5)
        return [t for t, s in sorted_types if s >= threshold]

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
