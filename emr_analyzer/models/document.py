"""Document data models."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class DocumentType(str, Enum):
    LABORATORIO = "laboratorio"
    RADIOLOGIA = "radiologia"
    MEDICINA_NUCLEARE = "medicina_nucleare"
    RADIOTERAPIA = "radioterapia"
    ANATOMIA_PATOLOGICA = "anatomia_patologica"
    VISITA_ONCOLOGICA = "visita_oncologica"
    VISITA_SPECIALISTICA = "visita_specialistica"
    PIANO_TERAPEUTICO = "piano_terapeutico"
    LETTERA_DIMISSIONE = "lettera_dimissione"
    SDO = "sdo"
    CARTELLA_CLINICA = "cartella_clinica"
    DIARIO_MEDICO = "diario_medico"
    DIARIO_INFERMIERISTICO = "diario_infermieristico"
    TERAPIA_SOMMINISTRATA = "terapia_somministrata"
    CONSULENZA = "consulenza"
    VERBALE_OPERATORIO = "verbale_operatorio"
    PRONTO_SOCCORSO = "pronto_soccorso"
    DOCUMENTO_AMMINISTRATIVO = "documento_amministrativo"
    NON_CLASSIFICATO = "non_classificato"


class ParsingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    COMPLETED_WITH_WARNINGS = "completed_with_warnings"
    ERROR = "error"


class ExtractionStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    ERROR = "error"


class ValidationStatus(str, Enum):
    PENDING = "pending"
    IN_REVIEW = "in_review"
    VALIDATED = "validated"
    REJECTED = "rejected"


@dataclass
class DocumentRecord:
    """Metadata record for an imported document."""
    id: str                                          # DOC_000123
    patient_id: str
    filename: str
    original_path: str
    file_hash: str
    page_count: int = 0
    document_date: Optional[str] = None
    document_type: str = DocumentType.NON_CLASSIFICATO.value
    import_date: str = field(default_factory=lambda: datetime.now().isoformat())
    parsing_status: str = ParsingStatus.PENDING.value
    extraction_status: str = ExtractionStatus.PENDING.value
    validation_status: str = ValidationStatus.PENDING.value
    event_count: int = 0
    lab_value_count: int = 0
    error_message: Optional[str] = None
    metadata_json: Optional[str] = None              # JSON blob for extended metadata

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "patient_id": self.patient_id,
            "filename": self.filename,
            "original_path": self.original_path,
            "file_hash": self.file_hash,
            "page_count": self.page_count,
            "document_date": self.document_date,
            "document_type": self.document_type,
            "import_date": self.import_date,
            "parsing_status": self.parsing_status,
            "extraction_status": self.extraction_status,
            "validation_status": self.validation_status,
            "event_count": self.event_count,
            "lab_value_count": self.lab_value_count,
            "error_message": self.error_message,
            "metadata_json": self.metadata_json,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DocumentRecord":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ClinicalSection:
    """A semantically coherent section within a document."""
    section_type: str    # anamnesi, diagnosi, terapia, etc.
    header: str
    text: str
    page_start: int
    page_end: int
    confidence: float = 1.0
