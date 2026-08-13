"""Clinical State data models."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Diagnosis:
    """A diagnosis in the Clinical State."""
    name: str
    icd_code: Optional[str] = None
    date: Optional[str] = None
    status: str = "active"               # active, resolved, in_remission
    source_event_id: Optional[str] = None
    source_document_id: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "icd_code": self.icd_code,
            "date": self.date,
            "status": self.status,
            "source_event_id": self.source_event_id,
            "source_document_id": self.source_document_id,
            "notes": self.notes,
        }


@dataclass
class Treatment:
    """A treatment in the Clinical State."""
    name: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    status: str = "active"               # active, completed, interrupted
    line: Optional[int] = None           # 1st line, 2nd line, ...
    setting: Optional[str] = None        # adjuvant, neoadjuvant, palliative, ...
    source_event_id: Optional[str] = None
    source_document_id: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "status": self.status,
            "line": self.line,
            "setting": self.setting,
            "source_event_id": self.source_event_id,
            "source_document_id": self.source_document_id,
            "notes": self.notes,
        }


@dataclass
class Toxicity:
    """A toxicity/adverse event record."""
    name: str
    grade: Optional[int] = None          # CTCAE grade 1-5
    date: Optional[str] = None
    status: str = "active"
    related_to: Optional[str] = None     # Treatment it's related to
    source_event_id: Optional[str] = None
    source_document_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "grade": self.grade,
            "date": self.date,
            "status": self.status,
            "related_to": self.related_to,
            "source_event_id": self.source_event_id,
            "source_document_id": self.source_document_id,
        }


@dataclass
class Procedure:
    """A medical procedure in the Clinical State."""
    name: str
    date: Optional[str] = None
    outcome: Optional[str] = None
    source_event_id: Optional[str] = None
    source_document_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "date": self.date,
            "outcome": self.outcome,
            "source_event_id": self.source_event_id,
            "source_document_id": self.source_document_id,
        }


@dataclass
class Biomarker:
    """A clinically relevant biomarker."""
    name: str
    value: Optional[str] = None
    date: Optional[str] = None
    interpretation: Optional[str] = None
    source_event_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "value": self.value,
            "date": self.date,
            "interpretation": self.interpretation,
            "source_event_id": self.source_event_id,
        }


@dataclass
class ClinicalState:
    """Consolidated, deduplicated clinical picture of a patient."""
    patient_id: str
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    version: int = 1

    # Clinical profile
    clinical_profile: Optional[str] = None          # Narrative summary

    # Lists
    active_diagnoses: list[Diagnosis] = field(default_factory=list)
    past_diagnoses: list[Diagnosis] = field(default_factory=list)
    staging: Optional[str] = None                   # TNM or similar
    biomarkers: list[Biomarker] = field(default_factory=list)
    active_treatments: list[Treatment] = field(default_factory=list)
    completed_treatments: list[Treatment] = field(default_factory=list)
    toxicities: list[Toxicity] = field(default_factory=list)
    allergies: list[str] = field(default_factory=list)
    comorbidities: list[str] = field(default_factory=list)
    procedures: list[Procedure] = field(default_factory=list)
    hospitalizations: list[dict] = field(default_factory=list)
    imaging_findings: list[dict] = field(default_factory=list)
    lab_trends: list[dict] = field(default_factory=list)
    # Complete, temporally ordered projection of every normalized evidence
    # item. The specialized lists above are convenient views, not the source
    # of truth and not a filter on clinical completeness.
    observations: list[dict] = field(default_factory=list)
    # Document-level grouping and cross-page relationships used to interpret
    # the atomic observations without discarding their provenance.
    document_projections: list[dict] = field(default_factory=list)
    open_problems: list[str] = field(default_factory=list)
    follow_up: list[dict] = field(default_factory=list)
    functional_status: Optional[str] = None
    last_known_status: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "patient_id": self.patient_id,
            "updated_at": self.updated_at,
            "version": self.version,
            "clinical_profile": self.clinical_profile,
            "active_diagnoses": [d.to_dict() for d in self.active_diagnoses],
            "past_diagnoses": [d.to_dict() for d in self.past_diagnoses],
            "staging": self.staging,
            "biomarkers": [b.to_dict() for b in self.biomarkers],
            "active_treatments": [t.to_dict() for t in self.active_treatments],
            "completed_treatments": [t.to_dict() for t in self.completed_treatments],
            "toxicities": [t.to_dict() for t in self.toxicities],
            "allergies": self.allergies,
            "comorbidities": self.comorbidities,
            "procedures": [p.to_dict() for p in self.procedures],
            "hospitalizations": self.hospitalizations,
            "imaging_findings": self.imaging_findings,
            "lab_trends": self.lab_trends,
            "observations": self.observations,
            "document_projections": self.document_projections,
            "open_problems": self.open_problems,
            "follow_up": self.follow_up,
            "functional_status": self.functional_status,
            "last_known_status": self.last_known_status,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ClinicalState":
        state = cls(
            patient_id=data.get("patient_id", ""),
            updated_at=data.get("updated_at", ""),
            version=data.get("version", 1),
            clinical_profile=data.get("clinical_profile"),
            staging=data.get("staging"),
            allergies=data.get("allergies", []),
            comorbidities=data.get("comorbidities", []),
            hospitalizations=data.get("hospitalizations", []),
            imaging_findings=data.get("imaging_findings", []),
            lab_trends=data.get("lab_trends", []),
            observations=data.get("observations", []),
            document_projections=data.get("document_projections", []),
            open_problems=data.get("open_problems", []),
            follow_up=data.get("follow_up", []),
            functional_status=data.get("functional_status"),
            last_known_status=data.get("last_known_status"),
        )
        state.active_diagnoses = [Diagnosis(**d) for d in data.get("active_diagnoses", [])]
        state.past_diagnoses = [Diagnosis(**d) for d in data.get("past_diagnoses", [])]
        state.biomarkers = [Biomarker(**b) for b in data.get("biomarkers", [])]
        state.active_treatments = [Treatment(**t) for t in data.get("active_treatments", [])]
        state.completed_treatments = [Treatment(**t) for t in data.get("completed_treatments", [])]
        state.toxicities = [Toxicity(**t) for t in data.get("toxicities", [])]
        state.procedures = [Procedure(**p) for p in data.get("procedures", [])]
        return state
