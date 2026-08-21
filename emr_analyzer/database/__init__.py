"""Database layer for EMR Analyzer."""

from .engine import DatabaseEngine
from .patient_identity_repo import PatientIdentityRepository
from .evidence_repo import EvidenceRepository
from .registry_repo import ClinicalRegistryRepository
from .processing_repo import ProcessingRepository
from .overlay_repo import DocumentTextOverlayRepository
from .review_repo import ReviewDecisionRepository
from .gold_set_repo import GoldSetRepository

__all__ = [
    "DatabaseEngine", "PatientIdentityRepository", "EvidenceRepository",
    "ClinicalRegistryRepository", "ProcessingRepository",
    "DocumentTextOverlayRepository", "ReviewDecisionRepository",
    "GoldSetRepository",
]
