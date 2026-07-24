"""Longitudinal Clinical State builder from normalized clinical texts.

Processes all normalized clinical text documents for a patient and extracts
structured clinical entities (diagnoses, treatments, toxicities, symptoms,
biomarkers, staging, procedures, etc.) using a local LLM, then merges them
into a deduplicated ClinicalState.
"""

import json
import logging
from pathlib import Path
from typing import Callable, Optional

from ..config import WORKSPACES_DIR
from ..extraction.qwen_client import QwenClient
from ..models.clinical_state import (
    ClinicalState, ClinicalStateDelta,
    Diagnosis, Treatment, Toxicity, Procedure, Biomarker,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ollama JSON schema — the LLM follows this to produce structured output.
# Each field is constrained by the schema so that Ollama's ``format``
# parameter enforces valid JSON with the expected shape.
# ---------------------------------------------------------------------------

CLINICAL_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "diagnoses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "icd_code": {"type": "string"},
                    "date": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": [
                            "active", "resolved", "in_remission",
                            "suspected", "ruled_out",
                        ],
                    },
                    "notes": {"type": "string"},
                },
                "required": ["name"],
            },
        },
        "treatments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": [
                            "active", "completed", "interrupted", "planned",
                        ],
                    },
                    "line": {"type": "string"},
                    "setting": {"type": "string"},
                    "dose": {"type": "string"},
                    "frequency": {"type": "string"},
                    "notes": {"type": "string"},
                },
                "required": ["name"],
            },
        },
        "toxicities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "grade": {"type": "integer"},
                    "date": {"type": "string"},
                    "status": {"type": "string"},
                    "related_to": {"type": "string"},
                    "notes": {"type": "string"},
                },
                "required": ["name"],
            },
        },
        "symptoms": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "severity": {"type": "string"},
                    "date": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": [
                            "active", "resolved", "improving", "worsening",
                        ],
                    },
                    "attribution": {"type": "string"},
                },
                "required": ["description"],
            },
        },
        "biomarkers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "value": {"type": "string"},
                    "date": {"type": "string"},
                    "interpretation": {"type": "string"},
                    "unit": {"type": "string"},
                },
                "required": ["name", "value"],
            },
        },
        "staging": {
            "type": "object",
            "properties": {
                "tnm": {"type": "string"},
                "stage_group": {"type": "string"},
                "date": {"type": "string"},
                "classification_system": {"type": "string"},
            },
        },
        "procedures": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "date": {"type": "string"},
                    "outcome": {"type": "string"},
                    "notes": {"type": "string"},
                },
                "required": ["name"],
            },
        },
        "allergies": {
            "type": "array",
            "items": {"type": "string"},
        },
        "comorbidities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "condition": {"type": "string"},
                    "date": {"type": "string"},
                    "status": {"type": "string"},
                },
                "required": ["condition"],
            },
        },
        "hospitalizations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string"},
                    "admission_date": {"type": "string"},
                    "discharge_date": {"type": "string"},
                    "department": {"type": "string"},
                    "notes": {"type": "string"},
                },
            },
        },
        "performance_status": {
            "type": "object",
            "properties": {
                "ecog_score": {"type": "integer"},
                "karnofsky_score": {"type": "integer"},
                "date": {"type": "string"},
            },
        },
        "imaging_findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "exam_type": {"type": "string"},
                    "date": {"type": "string"},
                    "findings": {"type": "string"},
                    "conclusion": {"type": "string"},
                },
            },
        },
        "follow_up": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "recommendation": {"type": "string"},
                    "date": {"type": "string"},
                    "specialty": {"type": "string"},
                },
            },
        },
        "clinical_profile": {"type": "string"},
    },
}

# ---------------------------------------------------------------------------
# Prompts — in Italian because the clinical texts are Italian.
# ---------------------------------------------------------------------------

CLINICAL_EXTRACTION_SYSTEM_PROMPT = """Sei un assistente clinico specializzato in oncologia medica. Analizzi testi clinici normalizzati ed estrai informazioni strutturate in formato JSON.

REGOLE FONDAMENTALI:
1. Estrai SOLO ciò che è esplicitamente menzionato nel testo — non inventare nulla
2. Date sempre in formato ISO (AAAA-MM-GG). Se presente solo l'anno, usa AAAA-06-15. Se il giorno non è specificato, usa 15
3. Per tossicità: grading CTCAE (1-5) se menzionato; altrimenti ometti il campo
4. Staging TNM: formato standard (es. cT2N1M0, pT3N0)
5. Dosaggi farmaci con unità (es. "5 mg/kg", "1000 mg/m²")
6. Categorie non trovate → array vuoto [], mai null. Staging e performance_status non trovati → ignora il campo
7. Diagnosi in italiano clinico standard (es. "Adenocarcinoma polmonare" non "Ca polmone dx"), con codice ICD se presente
8. Sintomi con severità (lieve/moderato/grave) e attribuzione (malattia/trattamento/altra)
9. Normalizza i nomi delle entità: usa la forma più completa e standard, non abbreviazioni
10. Quando un farmaco ha dosaggio e frequenza, includili sempre

RISPETTA STRETTAMENTE LO SCHEMA JSON FORNITO. Non aggiungere campi non previsti dallo schema.

ESEMPIO di estrazione attesa (da un testo fittizio):

Input: "Il paziente Giovanni Rossi, 65 anni, con diagnosi di adenocarcinoma polmonare sinistro cT2N1M0 (stadio IIIA) diagnosticato a marzo 2024. In trattamento con pembrolizumab 200 mg ogni 3 settimane dal 15/04/2024. Riferisce astenia G2 e nausea G1. ECOG PS 1. Allergia alla penicillina."

Output atteso:
{
  "diagnoses": [{"name": "Adenocarcinoma polmonare sinistro", "icd_code": "", "date": "2024-03-15", "status": "active", "notes": "cT2N1M0 stadio IIIA"}],
  "treatments": [{"name": "Pembrolizumab", "start_date": "2024-04-15", "end_date": "", "status": "active", "line": "", "setting": "", "dose": "200 mg", "frequency": "ogni 3 settimane", "notes": ""}],
  "toxicities": [{"name": "Astenia", "grade": 2, "date": "", "status": "active", "related_to": "Pembrolizumab", "notes": ""}, {"name": "Nausea", "grade": 1, "date": "", "status": "active", "related_to": "Pembrolizumab", "notes": ""}],
  "symptoms": [{"description": "Astenia", "severity": "moderato", "date": "", "status": "active", "attribution": "trattamento"}, {"description": "Nausea", "severity": "lieve", "date": "", "status": "active", "attribution": "trattamento"}],
  "biomarkers": [],
  "staging": {"tnm": "cT2N1M0", "stage_group": "IIIA", "date": "2024-03-15", "classification_system": "AJCC"},
  "procedures": [],
  "allergies": ["Penicillina"],
  "comorbidities": [],
  "hospitalizations": [],
  "performance_status": {"ecog_score": 1, "karnofsky_score": null, "date": ""},
  "imaging_findings": [],
  "follow_up": [],
  "clinical_profile": "Paziente con adenocarcinoma polmonare sinistro stadio IIIA in trattamento con pembrolizumab. Performance status ECOG 1. Tossicità attuali: astenia G2 e nausea G1."
}"""

CLINICAL_EXTRACTION_USER_PROMPT = """Analizza il seguente testo clinico normalizzato ed estrai TUTTE le informazioni clinicamente rilevanti seguendo lo schema JSON.

Documento: {filename}
Data documento: {doc_date}

Testo clinico:
{text}

Istruzioni specifiche per ogni categoria (rispetta lo schema JSON — i campi obbligatori vanno sempre compilati, per quelli opzionali usa stringa vuota se non presenti):

1. **DIAGNOSI** (priorità ALTA)
   - Neoplasie maligne, condizioni benigne, sindromi, patologie croniche
   - Normalizza il nome: "Carcinoma polmonare a piccole cellule" non "Ca polmone SCLC"
   - Stato: active se in corso o non specificato, resolved se guarito, in_remission se in remissione
   - Includi codice ICD solo se esplicitamente menzionato

2. **TERAPIE** (priorità ALTA)
   - Farmaci antitumorali, ormonoterapie, immunoterapie, chemioterapia, terapie di supporto
   - IMPORTANTE: includi dose (es. "5 mg/kg", "1000 mg/m²", "200 mg") e frequenza (es. "ogni 3 settimane", "die", "bis in die")
   - Stato: active se in corso, completed se terminato, interrupted se sospeso, planned se pianificato
   - Setting: adjuvant, neoadjuvant, palliative, curative, maintenance — se chiaro dal contesto
   - Line: numero romano (I, II, III) se indicato

3. **TOSSICITÀ** (priorità ALTA)
   - Grading CTCAE numerico se presente (es. "G2", "grado 2", "grade 2" → grade: 2)
   - Se il grado non è specificato ma la tossicità è descritta come "lieve" → grade 1, "moderato" → grade 2, "grave" → grade 3
   - related_to: nome del farmaco sospetto (se menzionato)

4. **SINTOMI** (priorità MEDIA)
   - Sintomi riferiti dal paziente, non eventi avversi da farmaci (quelli vanno in toxicities)
   - Severità: lieve/moderato/grave
   - Attribuzione: malattia/trattamento/altra

5. **BIOMARCATORI** (priorità MEDIA)
   - Mutazioni (EGFR, KRAS, BRAF, ecc.), recettori (ER, PR, HER2), PD-L1, marker sierici (CEA, CA19-9, ecc.)
   - Valore: riporta il risultato esatto (es. "EGFR esone 19 delezione", "PD-L1 60%", "ER 8/8")

6. **STADIAZIONE** (priorità ALTA se presente)
   - TNM completo (es. "cT2N1M0", "ypT3N0")
   - Stadio clinico o patologico
   - Data e sistema (AJCC 8ª ed, UICC, ecc.)

7. **PROCEDURE** (priorità MEDIA)
   - Solo interventi e procedure invasive rilevanti per la storia oncologica
   - outcome: esito se menzionato (es. "completo", "complicato da...", "R0")

8. **ALLERGIE** (priorità BASSA)
   - Sostanze con reazione allergica documentata

9. **COMORBIDITÀ** (priorità BASSA)
   - Condizioni preesistenti rilevanti (ipertensione, diabete, cardiopatie, BPCO, insufficienza renale, ecc.)

10. **RICOVERI** (priorità MEDIA)
    - Episode acute, ricoveri programmati, day hospital con pernottamento
    - admission_date e discharge_date se disponibili

11. **PERFORMANCE STATUS** (priorità ALTA se presente)
    - ECOG 0-5 o Karnofsky 0-100
    - Data della valutazione

12. **IMAGING** (priorità MEDIA)
    - Esami radiologici con referto: TC, RMN, PET-TC, RX, ecografia, mammografia
    - findings: reperti principali (es. "nodulo polmonare di 2.3 cm", "lesioni epatiche")
    - conclusion: impressione diagnostica finale

13. **FOLLOW-UP** (priorità BASSA)
    - Prossimi controlli, esami pianificati, visite di follow-up

14. **PROFILO CLINICO** — riassunto di 2-4 frasi che sintetizzi il quadro complessivo: neoplasia e stadio, terapia in corso, tossicità rilevanti, performance status. Scrivilo anche se gli altri campi sono già compilati."""


def enrich_existing(existing, new_item):
    """Copy non-empty fields from *new_item* to *existing* if the
    corresponding field on *existing* is empty or less detailed.
    Works with dataclass instances and plain dicts."""
    if isinstance(existing, dict) and isinstance(new_item, dict):
        for key in new_item:
            if key == "source_document_id":
                continue  # keep the first source
            new_val = new_item.get(key)
            old_val = existing.get(key)
            if new_val and not old_val:
                existing[key] = new_val
            elif new_val and old_val and isinstance(new_val, str) and isinstance(old_val, str):
                if len(new_val) > len(old_val):
                    existing[key] = new_val
        return
    # Dataclass objects: iterate over fields
    for field_name in (f.name for f in getattr(existing, '__dataclass_fields__', {}).values()):
        if field_name == "source_document_id":
            continue
        new_val = getattr(new_item, field_name, None)
        old_val = getattr(existing, field_name, None)
        if new_val and not old_val:
            setattr(existing, field_name, new_val)
        elif new_val and old_val and isinstance(new_val, str) and isinstance(old_val, str):
            if len(new_val) > len(old_val):
                setattr(existing, field_name, new_val)


class ClinicalStateBuilder:
    """Build a ClinicalState by extracting structured information from all
    normalized clinical texts of a patient via a local LLM."""

    def __init__(self, cs_repo,
                 llm_client: QwenClient):
        self._cs_repo = cs_repo
        self.llm = llm_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_for_patient(
        self,
        patient_id: str,
        progress_callback: Callable[[int, str], None] | None = None,
    ) -> ClinicalState:
        """Build a complete ClinicalState for *patient_id* by processing
        every normalized clinical text document through the LLM.

        Parameters
        ----------
        patient_id:
            The patient identifier.
        progress_callback:
            Optional ``fn(percent, message)`` called after each document
            (and during deduplication).

        Returns
        -------
        The saved, deduplicated ClinicalState.
        """
        texts = list(self._collect_normalized_texts(patient_id))
        total = len(texts)
        if total == 0:
            raise ValueError(
                f"Nessun testo clinico normalizzato trovato per {patient_id}"
            )

        # Start with a fresh state.
        state = ClinicalState(patient_id=patient_id)

        for idx, (doc_id, text, filename, doc_date) in enumerate(texts, start=1):
            if progress_callback:
                progress_callback(
                    int((idx - 1) * 100 / total),
                    f"Estrazione dati dal documento {idx}/{total} — {filename}",
                )

            extraction = self._extract_from_text(doc_id, text, filename, doc_date)
            if extraction is None:
                logger.warning("Estrazione fallita per %s (%s)", doc_id, filename)
                continue
            self._merge_into_state(state, extraction, doc_id)

        if progress_callback:
            progress_callback(95, "Deduplicazione e salvataggio...")

        state = self._deduplicate(state)
        self._cs_repo.save(state)

        if progress_callback:
            progress_callback(100, "Clinical State ricostruito con successo.")

        return state

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------

    def _collect_normalized_texts(
        self, patient_id: str,
    ):
        """Return ``(doc_id, text, filename, document_date)`` for every
        normalized clinical text, sorted chronologically (oldest first).

        Chronological order lets the merge phase build the Clinical State
        progressively: earlier documents establish the baseline, later ones
        add detail or updates.
        """
        extraction_dir = WORKSPACES_DIR / patient_id / "extraction"
        if not extraction_dir.is_dir():
            return []

        entries = []

        for md_path in extraction_dir.glob("*.md"):
            doc_id = md_path.stem

            # Skip internal files (pages, words, tables, cleaned_source, raw)
            if doc_id.endswith(("_pages", "_words", "_tables",
                               "_cleaned_source", "_source", "_raw")):
                continue

            # Only active clinical-text files are meaningful.
            if not (extraction_dir / f"{doc_id}_source.txt").exists():
                continue

            text = md_path.read_text(encoding="utf-8").strip()
            if not text:
                continue

            # Try to extract a document date from the metadata JSON stored
            # alongside the text.
            doc_date = self._guess_doc_date(patient_id, doc_id)

            entries.append((doc_id, text, md_path.name, doc_date))

        # Sort by document date (oldest first).  Documents without a known
        # date are placed at the end, preserving their relative alphabetical
        # order among themselves.
        def _sort_key(entry):
            date = entry[3]  # doc_date
            if date and len(date) >= 10:
                return date  # ISO string sorts lexicographically
            return "9999-99-99"  # push undated docs to the end

        entries.sort(key=_sort_key)
        return entries

    @staticmethod
    def _guess_doc_date(patient_id: str, doc_id: str) -> str:
        """Best-effort attempt at reading the document date from metadata."""
        import json
        extraction_dir = WORKSPACES_DIR / patient_id / "extraction"
        json_path = extraction_dir / f"{doc_id}.json"
        if not json_path.exists():
            return ""
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            doc_date = data.get("document_date") or ""
            if isinstance(doc_date, str) and len(doc_date) >= 10:
                return doc_date[:10]
        except Exception:
            pass
        return ""

    # ------------------------------------------------------------------
    # LLM extraction
    # ------------------------------------------------------------------

    def _extract_from_text(
        self,
        doc_id: str,
        text: str,
        filename: str,
        doc_date: str,
    ) -> dict | None:
        """Call the LLM with the structured schema and return the parsed
        JSON dict, or ``None`` on failure."""
        user_prompt = CLINICAL_EXTRACTION_USER_PROMPT.format(
            filename=filename,
            doc_date=doc_date or "(non disponibile)",
            text=text,
        )

        for attempt in range(2):
            try:
                result = self.llm.generate_structured(
                    prompt=user_prompt,
                    system=CLINICAL_EXTRACTION_SYSTEM_PROMPT,
                    schema=CLINICAL_EXTRACTION_SCHEMA,
                )
                if result and isinstance(result, dict):
                    return result
            except Exception as exc:
                logger.warning(
                    "Tentativo %d fallito per %s: %s",
                    attempt + 1, doc_id, exc,
                )
        return None

    # ------------------------------------------------------------------
    # Merge extracted data into a ClinicalState
    # ------------------------------------------------------------------

    @staticmethod
    def _get_name(item, name_attr: str) -> str:
        """Get the value of *name_attr* from a dataclass or dict."""
        if isinstance(item, dict):
            return str(item.get(name_attr, "") or "")
        return str(getattr(item, name_attr, "") or "")

    @staticmethod
    def _merge_or_enrich(target_list: list, new_item, name_attr: str = "name"):
        """Append *new_item* if no similar item exists; otherwise merge
        richer fields into the existing one (keep the longer name, add
        missing ICD codes, dates, grades, etc.)."""
        from difflib import SequenceMatcher

        new_name = ClinicalStateBuilder._get_name(new_item, name_attr)
        new_name_lower = new_name.lower().strip()
        if not new_name_lower or len(new_name_lower) < 3:
            return

        for existing in target_list:
            ename = ClinicalStateBuilder._get_name(existing, name_attr)
            ename_lower = ename.lower().strip()
            if ename_lower == new_name_lower:
                # Exact match — enrich existing with any new detail.
                enrich_existing(existing, new_item)
                return
            if SequenceMatcher(None, new_name_lower, ename_lower).ratio() > 0.80:
                # Similar match — keep the longer/more detailed name,
                # then enrich with new fields.
                if len(new_name) > len(ename):
                    if isinstance(existing, dict):
                        existing[name_attr] = new_name
                    else:
                        setattr(existing, name_attr, new_name)
                enrich_existing(existing, new_item)
                return

        target_list.append(new_item)

    def _merge_into_state(
        self, state: ClinicalState, extraction: dict, doc_id: str,
    ) -> None:
        """Merge a single document's extracted data into *state*."""
        # --- Diagnoses ---
        for d in extraction.get("diagnoses", []):
            diagnosis = Diagnosis(
                name=d.get("name", ""),
                icd_code=d.get("icd_code"),
                date=d.get("date"),
                status=d.get("status", "active"),
                source_document_id=doc_id,
                notes=d.get("notes"),
            )
            target = (state.past_diagnoses if diagnosis.status
                      in ("resolved", "ruled_out", "in_remission")
                      else state.active_diagnoses)
            self._merge_or_enrich(target, diagnosis)

        # --- Treatments ---
        for t in extraction.get("treatments", []):
            treatment = Treatment(
                name=t.get("name", ""),
                start_date=t.get("start_date"),
                end_date=t.get("end_date"),
                status=t.get("status", "active"),
                line=t.get("line"),
                setting=t.get("setting"),
                source_document_id=doc_id,
                notes=t.get("notes"),
            )
            target = (state.completed_treatments if treatment.status
                      in ("completed", "interrupted")
                      else state.active_treatments)
            self._merge_or_enrich(target, treatment)

        # --- Toxicities ---
        for t in extraction.get("toxicities", []):
            toxicity = Toxicity(
                name=t.get("name", ""),
                grade=t.get("grade"),
                date=t.get("date"),
                status=t.get("status", "active"),
                related_to=t.get("related_to"),
                source_document_id=doc_id,
            )
            self._merge_or_enrich(state.toxicities, toxicity)

        # --- Symptoms ---
        for s in extraction.get("symptoms", []):
            symptom = {
                "description": s.get("description", ""),
                "severity": s.get("severity"),
                "date": s.get("date"),
                "status": s.get("status", "active"),
                "attribution": s.get("attribution"),
                "source_document_id": doc_id,
            }
            self._merge_or_enrich(state.symptoms, symptom, name_attr="description")

        # --- Biomarkers ---
        for b in extraction.get("biomarkers", []):
            biomarker = Biomarker(
                name=b.get("name", ""),
                value=b.get("value"),
                date=b.get("date"),
                interpretation=b.get("interpretation"),
                source_event_id=doc_id,
            )
            self._merge_or_enrich(state.biomarkers, biomarker)

        # --- Staging ---
        st = extraction.get("staging")
        if st and isinstance(st, dict):
            tnm = st.get("tnm") or st.get("stage_group") or ""
            if tnm:
                # Keep the most recent staging when a date is available.
                existing_date = getattr(state, "_staging_date", "")
                new_date = st.get("date", "")
                if new_date >= existing_date:
                    state.staging = tnm + (f" ({st.get('classification_system', '')})" if st.get('classification_system') else "")
                    state._staging_date = new_date  # type: ignore[attr-defined]

        # --- Procedures ---
        for p in extraction.get("procedures", []):
            procedure = Procedure(
                name=p.get("name", ""),
                date=p.get("date"),
                outcome=p.get("outcome"),
                source_document_id=doc_id,
            )
            self._merge_or_enrich(state.procedures, procedure)

        # --- Allergies ---
        for a in extraction.get("allergies", []):
            if isinstance(a, str) and a not in state.allergies:
                state.allergies.append(a)

        # --- Comorbidities ---
        for c in extraction.get("comorbidities", []):
            if isinstance(c, dict):
                cond = c.get("condition", "")
                if cond and not any(
                    existing.lower() == cond.lower()
                    for existing in state.comorbidities
                ):
                    state.comorbidities.append(cond)
            elif isinstance(c, str) and c not in state.comorbidities:
                state.comorbidities.append(c)

        # --- Hospitalizations ---
        for h in extraction.get("hospitalizations", []):
            if isinstance(h, dict):
                state.hospitalizations.append({
                    "reason": h.get("reason", ""),
                    "admission_date": h.get("admission_date"),
                    "discharge_date": h.get("discharge_date"),
                    "department": h.get("department"),
                    "notes": h.get("notes"),
                    "source_document_id": doc_id,
                })

        # --- Performance status ---
        ps = extraction.get("performance_status")
        if ps and isinstance(ps, dict):
            existing_ps = state.performance_status or {}
            existing_date = existing_ps.get("date", "")
            new_date = ps.get("date", "")
            if new_date >= existing_date:
                state.performance_status = {
                    "ecog_score": ps.get("ecog_score"),
                    "karnofsky_score": ps.get("karnofsky_score"),
                    "date": new_date,
                }

        # --- Imaging findings ---
        for img in extraction.get("imaging_findings", []):
            if isinstance(img, dict):
                state.imaging_findings.append({
                    "exam_type": img.get("exam_type", ""),
                    "date": img.get("date"),
                    "findings": img.get("findings"),
                    "conclusion": img.get("conclusion"),
                    "source_document_id": doc_id,
                })

        # --- Follow-up ---
        for f in extraction.get("follow_up", []):
            if isinstance(f, dict):
                state.follow_up.append({
                    "recommendation": f.get("recommendation", ""),
                    "date": f.get("date"),
                    "specialty": f.get("specialty"),
                    "source_document_id": doc_id,
                })

        # --- Clinical profile ---
        profile = extraction.get("clinical_profile")
        if profile and isinstance(profile, str) and profile.strip():
            existing = state.clinical_profile or ""
            if existing:
                state.clinical_profile = existing + "\n\n---\n\n" + profile.strip()
            else:
                state.clinical_profile = profile.strip()

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def _deduplicate(self, state: ClinicalState) -> ClinicalState:
        """Reuse the ClinicalStateManager's deduplication logic."""
        # Import the manager at runtime to avoid circular imports.
        from .clinical_state import ClinicalStateManager

        return ClinicalStateManager._deduplicate_state(state)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _already_in(item_list: list, name: str) -> bool:
        """Return True if an item with a similar name already exists."""
        from difflib import SequenceMatcher

        name_lower = name.lower().strip()
        if not name_lower or len(name_lower) < 3:
            return True
        for existing in item_list:
            ename = (
                existing.name.lower().strip()
                if hasattr(existing, "name")
                else str(existing).lower().strip()
            )
            if ename == name_lower:
                return True
            if SequenceMatcher(None, name_lower, ename).ratio() > 0.75:
                return True
        return False
