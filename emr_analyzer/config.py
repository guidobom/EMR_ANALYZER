"""Centralized configuration for EMR Analyzer."""

from pathlib import Path

# --- Application paths ---
APP_NAME = "EMR Analyzer"
APP_VERSION = "0.7.0"
OFFLINE_MODE = True
# Verify each document's workspace attribution with the LLM before its
# clinical text is normalized. The check runs on the pre-anonymization
# text and blocks extraction of a document whose LLM identity points to a
# different workspace. Set to False to skip the per-document identity call.
ATTRIBUTION_VERIFICATION_ENABLED = True
# Inietta nel prompt di estrazione un few-shot di voci canoniche tratte dalle
# voci timeline golden (confermate dall'utente) di ALTRI pazienti, così le
# nuove estrazioni replicano lo stile di descrizione confermato. Gli esempi
# sono contenuti clinici de-identificati già presenti nel DB locale.
# False disabilita la lettura DB e la sezione nel prompt.
GOLDEN_FEWSHOT_ENABLED = True
# Numero massimo di esempi golden iniettati in un singolo prompt di estrazione.
GOLDEN_FEWSHOT_MAX_EXAMPLES = 10
BASE_DIR = Path.home() / ".emr_analyzer"
_workspace_path: Path = BASE_DIR / "workspaces"


class _ActiveWorkspace:
    """Mutable singleton holding the current project workspace root.

    Call ``set_path()`` to switch projects.  All code that previously
    imported ``active_workspace.path`` now references ``active_workspace.path``
    so the change propagates everywhere immediately.
    """
    path: Path = _workspace_path

    @classmethod
    def set_path(cls, new_path: Path) -> None:
        cls.path = Path(new_path)
        cls.path.mkdir(parents=True, exist_ok=True)


active_workspace = _ActiveWorkspace()

CACHE_DIR = BASE_DIR / "cache"
LOG_DIR = BASE_DIR / "logs"
IDENTITY_KEY_PATH = BASE_DIR / "identity.key"

# --- Supported file types ---
SUPPORTED_EXTENSIONS = (
    ".pdf",
    ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp",
    ".txt", ".md", ".csv", ".hl7",
    ".xml", ".cda", ".json",
    ".docx", ".doc", ".xlsx",
)

# --- Local models via llama.cpp (app-managed llama-server) ---
# The app spawns its own llama-server child processes; the only external
# requirement is the binary (brew install llama.cpp) and GGUF model files
# registered in the local model index.
LLAMA_SERVER_BINARY = ""  # resolved at runtime: PATH, then brew prefixes
LLAMA_SERVER_HOST = "127.0.0.1"
# Base port for the app-owned servers; allocated upward from here, avoiding
# a possibly still-running Ollama daemon on 11434.
LLAMA_SERVER_BASE_PORT = 11435
# Deadline for a freshly spawned server to finish loading its model.
LLAMA_SERVER_LOAD_TIMEOUT = 180
# Client-side timeout (seconds) for generation requests.  Clinical prompts
# can reach 10-15K input tokens and the 14B model generates slowly under
# load: 5 minutes proved too tight, so be generous (the old Ollama client
# had no practical timeout at all).
LLAMA_SERVER_HTTP_TIMEOUT = 1800
# GGUF models directory and its metadata index (see llm_backend/model_store.py).
LLM_MODELS_DIR = BASE_DIR / "models"
LLM_MODEL_INDEX_PATH = LLM_MODELS_DIR / "index.json"

# The document model can be smaller/faster; the Clinical State model can be
# larger because it is used after the evidence has already been normalized.
# Names are friendly GGUF-index names (<family>-<tag>, e.g. "qwen3-14b");
# legacy Ollama names like "qwen3:14b" are resolved at runtime.
DOCUMENT_LLM_MODEL_NAME = "qwen3-14b"
CLINICAL_STATE_LLM_MODEL_NAME = "qwen3-14b"
# Default model for LlmClient when no LLMRoleConfig is provided.
DEFAULT_LLM_MODEL_NAME = DOCUMENT_LLM_MODEL_NAME
LLM_DEFAULT_CONTEXT_LENGTH = 32768

# Legacy Ollama constants, kept only for old persisted settings compatibility.
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_CONTEXT_LENGTH = 32768

# Lab parameter synonyms (Italian clinical abbreviations -> canonical name)
LAB_SYNONYMS = {
    "hb": "emoglobina",
    "hgb": "emoglobina",
    "plt": "piastrine",
    "got": "ast",
    "gpt": "alt",
    "ast": "aspartato_aminotransferasi",
    "alt": "alanina_aminotransferasi",
    "creat.": "creatinina",
    "glicemia": "glucosio",
    "azotemia": "urea",
    "ves": "velocita_eritrosedimentazione",
    "gb": "globuli_bianchi",
    "wbc": "globuli_bianchi",
    "gr": "globuli_rossi",
    "rbc": "globuli_rossi",
    "hct": "ematocrito",
    "mcv": "volume_corpuscolare_medio",
    "mch": "contenuto_emoglobinico_medio",
    "mchc": "concentrazione_emoglobinica_media",
    "rdw": "ampiezza_distribuzione_eritrocitaria",
    "neutrofili": "neutrofili",
    "linfociti": "linfociti",
    "monociti": "monociti",
    "eosinofili": "eosinofili",
    "basofili": "basofili",
    "pt": "tempo_protrombina",
    "pt (inr)": "tempo_protrombina_inr",
    "pt(inr)": "tempo_protrombina_inr",
    "pt (ratio)": "tempo_protrombina_ratio",
    "pt(ratio)": "tempo_protrombina_ratio",
    "ptt": "tempo_tromboplastina_parziale",
    "aptt": "tempo_tromboplastina_parziale_attivata",
    "inr": "rapporto_normalizzato_internazionale",
    "fibrinogeno": "fibrinogeno",
    "d-dimero": "d_dimero",
    "pcr": "proteina_c_reattiva",
    "proteina_c_reattiva": "proteina_c_reattiva",
    "tsh": "ormone_tireostimolante",
    "ft3": "triiodotironina_libera",
    "ft4": "tiroxina_libera",
    "t3": "triiodotironina",
    "t4": "tiroxina",
    "ldh": "lattato_deidrogenasi",
    "cpk": "creatinfosfochinasi",
    "ck": "creatinchinasi",
    "troponina": "troponina",
    "nt-probnp": "nt_pro_bnp",
    "pro_bnp": "pro_bnp",
    "ca125": "ca_125",
    "ca 125": "ca_125",
    "cea": "antigene_carcinoembrionario",
    "afp": "alfa_fetoproteina",
    "psa": "antigene_prostatico_specifico",
    "ca19-9": "ca_19_9",
    "ca 19.9": "ca_19_9",
    "bilirubina_totale": "bilirubina_totale",
    "bilirubina_diretta": "bilirubina_diretta",
    "bilirubina_indiretta": "bilirubina_indiretta",
    "colesterolo_totale": "colesterolo_totale",
    "hdl": "colesterolo_hdl",
    "ldl": "colesterolo_ldl",
    "trigliceridi": "trigliceridi",
    "proteine_totali": "proteine_totali",
    "albumina": "albumina",
    "calcio": "calcio",
    "potassio": "potassio",
    "sodio": "sodio",
    "cloro": "cloro",
    "magnesio": "magnesio",
    "ferro": "ferro",
    "ferritina": "ferritina",
    "transferrina": "transferrina",
    "acido_urico": "acido_urico",
    "uricemia": "acido_urico",
    "amilasi": "amilasi",
    "lipasi": "lipasi",
    "pct": "procalcitonina",
    "egfr": "egfr",
    "vel. filtr. glomerulare (egfr)": "egfr",
    "iga": "immunoglobulina_a",
    "igg": "immunoglobulina_g",
    "igm": "immunoglobulina_m",
    "ige": "immunoglobulina_e",
}

# Unit canonicalization
UNIT_MAPPING = {
    "mg/dl": "mg/dL",
    "mg/100ml": "mg/dL",
    "mg/100 ml": "mg/dL",
    "g/dl": "g/dL",
    "g/100ml": "g/dL",
    "ng/ml": "ng/mL",
    "ng/l": "ng/L",
    "pg/ml": "pg/mL",
    "mg/l": "mg/L",
    "g/l": "g/L",
    "μg/ml": "μg/mL",
    "µg/ml": "μg/mL",
    "ug/ml": "μg/mL",
    "μg/dl": "μg/dL",
    "µg/dl": "μg/dL",
    "ug/dl": "μg/dL",
    "u/l": "U/L",
    "ui/l": "U/L",
    "iu/l": "U/L",
    "meq/l": "mEq/L",
    "mmol/l": "mmol/L",
    "μmol/l": "μmol/L",
    "umol/l": "μmol/L",
    "ml/min": "mL/min",
    "μu/ml": "μU/mL",
    "µu/ml": "μU/mL",
    "uu/ml": "μU/mL",
    "pmol/l": "pmol/L",
    "nmol/l": "nmol/L",
    "ratio": "Ratio",
    "inr": "INR",
    "fl": "fL",
    "pg": "pg",
    "%": "%",
    "mm/h": "mm/h",
    "mm3": "mm³",
    "μl": "/μL",
    "ul": "/μL",
    "103/μl": "×10³/μL",
    "106/μl": "×10⁶/μL",
}


# --- Document classification keywords ---
RADIOLOGY_KEYWORDS = [
    r"\bTECNICA\s+(?:DI\s+)?ESAME\b", r"\bMETODICA\b",
    r"\bREPERTO\s+RADIOLOGICO\b", r"\bQUESITO\s+DIAGNOSTICO\b",
    r"\bRX\b", r"\bTC\b", r"\bTAC\b", r"\bRM\b", r"\bRMN\b",
    r"\bECOGRAFIA\b", r"\bECOGRAFO\b", r"\bRADIOGRAFIA\b",
    r"\bRADIOLOGIA\b", r"\bMAMMOGRAFIA\b", r"\bTOMOGRAFIA\b",
    r"\bRISONANZA\s+MAGNETICA\b", r"\bANGIOGRAFIA\b",
    r"\bSCINTIGRAFIA\b", r"\bPET\b", r"\bPET-TC\b",
    r"\bRADIOLOGO\b", r"\bMEZZO\s+DI\s+CONTRASTO\b", r"\bMDC\b",
    r"\bRADIODIAGNOSTICA\b", r"\bECOGRAFICO\b", r"\bECOTOMOGRAFIA\b",
    r"\bSTUDIO\s+RADIOLOGICO\b", r"\bRX-TORACE\b",
    r"\bTOMOGRAFIA\s+COMPUTERIZZATA\b",
]

LAB_KEYWORDS = [
    r"\bmg/dL\b", r"\bg/dL\b", r"\bmmol/L\b", r"\bU/L\b", r"\bUI/L\b",
    r"\bmEq/L\b", r"\bng/mL\b", r"\bpg/mL\b", r"\bfL\b",
    r"\bEMOCROMO\b", r"\bGLICEMIA\b", r"\bCREATININA\b",
    r"\bAZOTEMIA\b", r"\bTRANSAMINASI\b", r"\bCOLESTEROLO\b",
    r"\bTRIGLICERIDI\b", r"\bELETTROLITI\b", r"\bEMOGAS\b",
    r"\bESAMI\s+DI\s+LABORATORIO\b", r"\bLABORATORIO\b",
    r"\bANALISI\b", r"\bESAMI\s+EMATOCHIMICI\b", r"\bBIOCHIMICA\b",
    r"\bEMATOLOGIA\b", r"\bCOAGULAZIONE\b", r"\bSIEROLOGIA\b",
    r"\bIMMUNOENZIMATICA\b", r"\bELETTROFORESI\b",
    # Microbiology
    r"\bANTIBIOGRAMMA\b", r"\bCOLTURA\b", r"\bCOLTURALE\b",
    r"\bMICROBIOLOGIA\b", r"\bMICROBIOLOGICO\b", r"\bBATTERIOLOGICO\b",
    r"\bESAME\s+COLTURALE\b", r"\bTAMPONE\b", r"\bURINOCOLTURA\b",
    r"\bCOPROCOLTURA\b", r"\bEMOCOLTURA\b", r"\bBATTERICO\b",
    r"\bANTIBIOTICO\b", r"\bSENSIBILITÀ\b", r"\bSENSIBILE\b",
    r"\bRESISTENTE\b", r"\bMIC\b", r"\bCARICA\s+BATTERICA\b",
    r"\bGERME\b", r"\bBATTERIO\b", r"\bSTAFILOCOCCO\b", r"\bSTREPTOCOCCO\b",
    r"\bESCHERICHIA\b", r"\bPSEUDOMONAS\b", r"\bCANDIDA\b",
    r"\bENTEROBATTERI\b", r"\bGRAM\s*[+-]\b", r"\bGRAM\s*POSITIVI\b",
    r"\bGRAM\s*NEGATIVI\b",
]

HOSPITALIZATION_KEYWORDS = [
    r"\bCARTELLA\s+CLINICA\b", r"\bLETTERA\s+DI\s+DIMISSIONE\b",
    r"\bSDO\b", r"\bSCHEDA\s+DI\s+DIMISSIONE\b", r"\bRICOVERO\b",
    r"\bDIARIO\s+MEDICO\b", r"\bDIARIO\s+INFERMIERISTICO\b",
    r"\bDEGENZA\b", r"\bREPARTO\b", r"\bDIMISSIONE\b",
    r"\bTRASFERIMENTO\b", r"\bACCETTAZIONE\b",
]

VISIT_KEYWORDS = [
    r"\bVISITA\b", r"\bESAME\s+OBIETTIVO\b", r"\bANAMNESI\b",
    r"\bMOTIVO\s+DELLA\s+VISITA\b", r"\bCONSULENZA\b",
    r"\bSPECIALISTICA\b", r"\bONCOLOGICA\b", r"\bAMBULATORIALE\b",
]

# --- Clinical sections ---
CLINICAL_SECTIONS = {
    "dati_identificativi": [
        r"\bDATI\s+IDENTIFICATIVI\b", r"\bDATI\s+DEL\s+PAZIENTE\b",
        r"\bIDENTIFICAZIONE\b",
    ],
    "motivo_visita": [
        r"\bMOTIVO\s+(?:DELLA\s+)?VISITA\b", r"\bMOTIVO\s+DEL\s+RICOVERO\b",
        r"\bQUESITO\s+(?:DIAGNOSTICO|CLINICO)\b",
    ],
    "anamnesi_remota": [
        r"\bANAMNESI\s+REMOTA\b", r"\bANAMNESI\s+PATOLOGICA\b",
        r"\bSTORIA\s+CLINICA\b",
    ],
    "anamnesi_oncologica": [
        r"\bANAMNESI\s+ONCOLOGICA\b", r"\bSTORIA\s+ONCOLOGICA\b",
        r"\bDIAGNOSI\s+ONCOLOGICA\b",
    ],
    "anamnesi_farmacologica": [
        r"\bANAMNESI\s+FARMACOLOGICA\b", r"\bFARMACI\s+IN\s+CORSO\b",
        r"\bTERAPIA\s+DOMICILIARE\b",
    ],
    "allergie": [
        r"\bALLERGIE\b", r"\bINTOLLERANZE\b", r"\bREAZIONI\s+AVVERSE\b",
    ],
    "terapie_in_corso": [
        r"\bTERAPIA\s+(?:IN\s+CORSO|ATTUALE|IN\s+ATTO)\b",
        r"\bTRATTAMENTO\s+(?:IN\s+CORSO|ATTUALE)\b",
    ],
    "esame_obiettivo": [
        r"\bESAME\s+OBIETTIVO\b", r"\bEO\b",
        r"\bCONDIZIONI\s+(?:GENERALI|CLINICHE)\b",
    ],
    "laboratorio": [
        r"\bESAMI\s+(?:DI\s+)?LABORATORIO\b", r"\bLABORATORIO\b",
        r"\bESAMI\s+EMATOCHIMICI\b", r"\bEMOCROMO\b",
    ],
    "imaging": [
        r"\bIMAGING\b", r"\bDIAGNOSTICA\s+PER\s+IMMAGINI\b",
        r"\bREFERTI\s+(?:RADIOLOGICI|STRUMENTALI)\b",
    ],
    "diagnosi": [
        r"\bDIAGNOSI\b", r"\bCONCLUSIONI\s+DIAGNOSTICHE\b",
        r"\bDIAGNOSI\s+(?:PRINCIPALE|SECONDARIA|DI\s+INGRESSO|DI\s+DIMISSIONE)\b",
    ],
    "valutazione": [
        r"\bVALUTAZIONE\b", r"\bCONSIDERAZIONI\b", r"\bCOMMENTO\b",
        r"\bGIUDIZIO\b",
    ],
    "piano_terapeutico": [
        r"\bPIANO\s+TERAPEUTICO\b", r"\bPROGRAMMA\s+TERAPEUTICO\b",
        r"\bINDICAZIONI\s+TERAPEUTICHE\b",
    ],
    "raccomandazioni": [
        r"\bRACCOMANDAZIONI\b", r"\bCONSIGLI\b", r"\bINDICAZIONI\b",
    ],
    "follow_up": [
        r"\bFOLLOW.?UP\b", r"\bFOLLOW\s*UP\b", r"\bCONTROLLO\b",
        r"\bPROSSIM[AO]\s+(?:VISITA|CONTROLLO|APPUNTAMENTO)\b",
    ],
    "conclusioni": [
        r"\bCONCLUSIONI\b", r"\bIN\s+CONCLUSIONE\b",
        r"\bIN\s+SINTESI\b", r"\bRIASSUNTO\b",
    ],
}

# Hospitalization-specific sections
ADMISSION_SECTIONS = {
    "accesso": [
        r"\bACCESSO\b", r"\bINGRESSO\b", r"\bACCOGLIENZA\b",
        r"\bACCETTAZIONE\b",
    ],
    "diagnosi_ingresso": [
        r"\bDIAGNOSI\s+(?:DI|D['’])INGRESSO\b",
        r"\bDIAGNOSI\s+INIZIALE\b",
    ],
    "diario_medico": [
        r"\bDIARIO\s+MEDICO\b", r"\bDECORSO\s+CLINICO\b",
        r"\bEVOLUZIONE\s+CLINICA\b",
    ],
    "diario_infermieristico": [
        r"\bDIARIO\s+INFERMIERISTICO\b", r"\bNOTE\s+INFERMIERISTICHE\b",
        r"\bSCHEDA\s+INFERMIERISTICA\b",
    ],
    "terapia": [
        r"\bTERAPIA\b", r"\bSCHEDA\s+TERAPEUTICA\b",
        r"\bSOMMINISTRAZIONE\b", r"\bPRESCRIZIONI\b",
    ],
    "consulenze": [
        r"\bCONSULENZ[AE]\b", r"\bPARERE\s+SPECIALISTICO\b",
    ],
    "procedure": [
        r"\bPROCEDURE\b", r"\bINTERVENTI\b", r"\bOPERAZIONI\b",
    ],
    "complicanze": [
        r"\bCOMPLICANZ[AE]\b", r"\bEVENTI\s+AVVERSI\b",
        r"\bPROBLEMI\s+INSORTI\b",
    ],
    "dimissione": [
        r"\bDIMISSIONE\b", r"\bLETTERA\s+DI\s+DIMISSIONE\b",
        r"\bCONCLUSIONE\s+DEL\s+RICOVERO\b",
    ],
    "sdo": [
        r"\bSDO\b", r"\bSCHEDA\s+DI\s+DIMISSIONE\s+OSPEDALIERA\b",
    ],
}
