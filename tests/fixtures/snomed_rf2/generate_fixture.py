#!/usr/bin/env python3
"""Regenerate the synthetic RF2 fixture used by the SNOMED tests.

The fixture is a self-contained, deterministic RF2 release snapshot built from
the compact spec below.  Codes are representative (some match real SNOMED CT
ids, some are approximations); the fixture is a *test corpus*, not a real
release, so no value here should be treated as clinically authoritative.

Run from the repository root:

    python tests/fixtures/snomed_rf2/generate_fixture.py

The generator overwrites the RF2 files and the ``concepts.csv`` manifest; both
are committed so tests never need to invoke it.
"""

from __future__ import annotations

import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
RELEASE_DIR = HERE / "Snapshot_20260101"
CONTENT_DIR = RELEASE_DIR / "Snapshot" / "Content"
LANG_DIR = RELEASE_DIR / "Snapshot" / "Refset" / "Language"
RELEASE_DATE = "20260101"

# RF2 component ids (international core module; stable reference values).
MODULE_ID = "900000000000207008"
DEFINITION_PRIMITIVE = "900000000000074008"
FSN_TYPE = "900000000000003001"
SYNONYM_TYPE = "900000000000013009"
IS_A_TYPE = "116680003"
CHARACTERISTIC_INFERRED = "900000000000011006"
MODIFIER_SOME = "900000000000451002"
PREFERRED_ACCEPTABILITY = "900000000000548007"
ACCEPTABLE_ACCEPTABILITY = "900000000000549004"
LANG_REFSET_EN = "900000000000509007"
LANG_REFSET_IT = "11000284103"

# ---------------------------------------------------------------------------
# Compact concept spec.  ``parents`` are IS-A parents; ``syn`` entries become
# synonym descriptions with ``acceptable`` acceptability, PTs get ``preferred``.
# ---------------------------------------------------------------------------

CONCEPT_SPECS: dict[str, dict] = {
    "138875005": {
        "fsn_it": "Concetto SNOMED CT (concetto)",
        "fsn_en": "SNOMED CT Concept (concept)",
        "pt_it": "Concetto SNOMED CT", "pt_en": "SNOMED CT Concept",
    },
    "404684003": {
        "fsn_it": "Reperto clinico (reperto)",
        "fsn_en": "Clinical finding (finding)",
        "pt_it": "Reperto clinico", "pt_en": "Clinical finding",
        "parents": ["138875005"],
    },
    "64572001": {
        "fsn_it": "Malattia (disturbo)",
        "fsn_en": "Disease (disorder)",
        "pt_it": "Malattia", "pt_en": "Disease",
        "parents": ["404684003"],
    },
    "71388002": {
        "fsn_it": "Procedura (procedura)",
        "fsn_en": "Procedure (procedure)",
        "pt_it": "Procedura", "pt_en": "Procedure",
        "parents": ["138875005"],
    },
    "363787002": {
        "fsn_it": "Entità osservabile (entità osservabile)",
        "fsn_en": "Observable entity (observable entity)",
        "pt_it": "Entità osservabile", "pt_en": "Observable entity",
        "parents": ["138875005"],
    },
    "105590001": {
        "fsn_it": "Sostanza (sostanza)",
        "fsn_en": "Substance (substance)",
        "pt_it": "Sostanza", "pt_en": "Substance",
        "parents": ["138875005"],
    },
    "279125003": {
        "fsn_it": "Struttura morfologicamente anomala (struttura morfologicamente anomala)",
        "fsn_en": "Morphologically abnormal structure (morphologic abnormality)",
        "pt_it": "Struttura morfologicamente anomala",
        "pt_en": "Morphologically abnormal structure",
        "parents": ["404684003"],
    },
    # --- Diagnosis (disorder) -------------------------------------------
    "44054006": {
        "fsn_it": "Diabete mellito (disturbo)",
        "fsn_en": "Diabetes mellitus (disorder)",
        "pt_it": "Diabete mellito", "pt_en": "Diabetes mellitus",
        "syn_it": ["diabete"], "syn_en": ["diabetes", "DM"],
        "parents": ["64572001"], "expected_fact_types": ["diagnosis"],
    },
    "38341003": {
        "fsn_it": "Disturbo ipertensivo (disturbo)",
        "fsn_en": "Hypertensive disorder (disorder)",
        "pt_it": "Ipertensione arteriosa", "pt_en": "Hypertensive disorder",
        "syn_it": ["ipertensione"], "syn_en": ["hypertension", "high blood pressure"],
        "parents": ["64572001"], "expected_fact_types": ["diagnosis"],
    },
    "42343007": {
        "fsn_it": "Scompenso cardiaco congestizio (disturbo)",
        "fsn_en": "Congestive heart failure (disorder)",
        "pt_it": "Scompenso cardiaco congestizio", "pt_en": "Congestive heart failure",
        "syn_it": ["scompenso cardiaco"], "syn_en": ["heart failure", "CHF"],
        "parents": ["64572001"], "expected_fact_types": ["diagnosis"],
    },
    "271737000": {
        "fsn_it": "Anemia (disturbo)",
        "fsn_en": "Anemia (disorder)",
        "pt_it": "Anemia", "pt_en": "Anemia",
        "syn_en": ["anaemia"],
        "parents": ["64572001"], "expected_fact_types": ["diagnosis"],
    },
    "93655004": {
        "fsn_it": "Melanoma maligno (disturbo)",
        "fsn_en": "Malignant melanoma (disorder)",
        "pt_it": "Melanoma maligno", "pt_en": "Malignant melanoma",
        "syn_it": ["melanoma"], "syn_en": ["melanoma"],
        "parents": ["64572001"], "expected_fact_types": ["diagnosis"],
    },
    "13645005": {
        "fsn_it": "Broncopneumopatia cronica ostruttiva (disturbo)",
        "fsn_en": "Chronic obstructive pulmonary disease (disorder)",
        "pt_it": "Broncopneumopatia cronica ostruttiva", "pt_en": "COPD",
        "syn_it": ["BPCO"], "syn_en": ["COPD"],
        "parents": ["64572001"], "expected_fact_types": ["diagnosis"],
    },
    # --- Symptoms (finding) ---------------------------------------------
    "386661006": {
        "fsn_it": "Febbre (reperto)",
        "fsn_en": "Fever (finding)",
        "pt_it": "Febbre", "pt_en": "Fever",
        "syn_it": ["iperpiressia"], "syn_en": ["pyrexia"],
        "parents": ["404684003"], "expected_fact_types": ["symptom"],
    },
    "22253000": {
        "fsn_it": "Dolore (reperto)",
        "fsn_en": "Pain (finding)",
        "pt_it": "Dolore", "pt_en": "Pain",
        "parents": ["404684003"], "expected_fact_types": ["symptom"],
    },
    "49727002": {
        "fsn_it": "Tosse (reperto)",
        "fsn_en": "Cough (finding)",
        "pt_it": "Tosse", "pt_en": "Cough",
        "parents": ["404684003"], "expected_fact_types": ["symptom"],
    },
    "267036007": {
        "fsn_it": "Dispnea (reperto)",
        "fsn_en": "Dyspnea (finding)",
        "pt_it": "Dispnea", "pt_en": "Dyspnea",
        "syn_it": ["affanno"], "syn_en": ["shortness of breath"],
        "parents": ["404684003"], "expected_fact_types": ["symptom"],
    },
    "52778003": {
        "fsn_it": "Astenia (reperto)",
        "fsn_en": "Asthenia (finding)",
        "pt_it": "Astenia", "pt_en": "Asthenia",
        "syn_it": ["affaticamento"], "syn_en": ["fatigue"],
        "parents": ["404684003"], "expected_fact_types": ["symptom"],
    },
    "422587007": {
        "fsn_it": "Nausea (reperto)",
        "fsn_en": "Nausea (finding)",
        "pt_it": "Nausea", "pt_en": "Nausea",
        "parents": ["404684003"], "expected_fact_types": ["symptom"],
    },
    "418363000": {
        "fsn_it": "Prurito (reperto)",
        "fsn_en": "Pruritus (finding)",
        "pt_it": "Prurito", "pt_en": "Pruritus",
        "parents": ["404684003"], "expected_fact_types": ["symptom"],
    },
    # --- Signs (finding) ------------------------------------------------
    "79654002": {
        "fsn_it": "Edema (reperto)",
        "fsn_en": "Edema (finding)",
        "pt_it": "Edema", "pt_en": "Edema",
        "syn_en": ["oedema"],
        "parents": ["404684003"], "expected_fact_types": ["clinical_sign"],
    },
    "18165001": {
        "fsn_it": "Ittero (reperto)",
        "fsn_en": "Jaundice (finding)",
        "pt_it": "Ittero", "pt_en": "Jaundice",
        "syn_en": ["icterus"],
        "parents": ["404684003"], "expected_fact_types": ["clinical_sign"],
    },
    # --- Radiology findings (finding) -----------------------------------
    "300928002": {
        "fsn_it": "Nodulo polmonare (reperto)",
        "fsn_en": "Nodule of lung (finding)",
        "pt_it": "Nodulo polmonare", "pt_en": "Nodule of lung",
        "syn_it": ["nodulo"],
        "parents": ["404684003"], "expected_fact_types": ["radiology_finding"],
    },
    "384715000": {
        "fsn_it": "Versamento pleurico (reperto)",
        "fsn_en": "Pleural effusion (finding)",
        "pt_it": "Versamento pleurico", "pt_en": "Pleural effusion",
        "syn_it": ["versamento"],
        "parents": ["404684003"], "expected_fact_types": ["radiology_finding"],
    },
    "271513008": {
        "fsn_it": "Opacità (reperto)",
        "fsn_en": "Opacity (finding)",
        "pt_it": "Opacità", "pt_en": "Opacity",
        "syn_it": ["opacità polmonare"],
        "parents": ["404684003"], "expected_fact_types": ["radiology_finding"],
    },
    "385443000": {
        "fsn_it": "Linfoadenopatia (reperto)",
        "fsn_en": "Lymphadenopathy (finding)",
        "pt_it": "Linfoadenopatia", "pt_en": "Lymphadenopathy",
        "syn_it": ["adenopatia"], "syn_en": ["lymph node enlargement"],
        "parents": ["404684003"], "expected_fact_types": ["radiology_finding"],
    },
    "300773005": {
        "fsn_it": "Lesione epatica (reperto)",
        "fsn_en": "Lesion of liver (finding)",
        "pt_it": "Lesione epatica", "pt_en": "Lesion of liver",
        "syn_it": ["lesione"],
        "parents": ["404684003"], "expected_fact_types": ["radiology_finding"],
    },
    # --- Procedures -----------------------------------------------------
    "86273004": {
        "fsn_it": "Biopsia (procedura)",
        "fsn_en": "Biopsy (procedure)",
        "pt_it": "Biopsia", "pt_en": "Biopsy",
        "syn_it": ["biopsia"],
        "parents": ["71388002"], "expected_fact_types": ["procedure", "clinical_decision"],
    },
    "77477000": {
        "fsn_it": "Tomografia computerizzata (procedura)",
        "fsn_en": "Computed tomography (procedure)",
        "pt_it": "Tomografia computerizzata", "pt_en": "Computed tomography",
        "syn_it": ["TC", "TAC"], "syn_en": ["CT scan", "CAT scan"],
        "parents": ["71388002"], "expected_fact_types": ["procedure", "clinical_decision"],
    },
    "312250001": {
        "fsn_it": "Risonanza magnetica (procedura)",
        "fsn_en": "Magnetic resonance imaging (procedure)",
        "pt_it": "Risonanza magnetica", "pt_en": "Magnetic resonance imaging",
        "syn_it": ["RM", "RMN"], "syn_en": ["MRI", "MR"],
        "parents": ["71388002"], "expected_fact_types": ["procedure", "clinical_decision"],
    },
    "105328002": {
        "fsn_it": "Tomografia a emissione di positroni (procedura)",
        "fsn_en": "Positron emission tomography (procedure)",
        "pt_it": "Tomografia a emissione di positroni", "pt_en": "PET scan",
        "syn_it": ["PET"], "syn_en": ["PET"],
        "parents": ["71388002"], "expected_fact_types": ["procedure", "clinical_decision"],
    },
    "73761001": {
        "fsn_it": "Colonscopia (procedura)",
        "fsn_en": "Colonoscopy (procedure)",
        "pt_it": "Colonscopia", "pt_en": "Colonoscopy",
        "parents": ["71388002"], "expected_fact_types": ["procedure", "clinical_decision"],
    },
    "387713003": {
        "fsn_it": "Resezione chirurgica (procedura)",
        "fsn_en": "Surgical resection (procedure)",
        "pt_it": "Resezione chirurgica", "pt_en": "Surgical resection",
        "syn_it": ["resezione"], "syn_en": ["surgical excision"],
        "parents": ["71388002"], "expected_fact_types": ["procedure", "clinical_decision"],
    },
    # --- Vital signs (observable entity) --------------------------------
    "400975006": {
        "fsn_it": "Pressione arteriosa (entità osservabile)",
        "fsn_en": "Blood pressure (observable entity)",
        "pt_it": "Pressione arteriosa", "pt_en": "Blood pressure",
        "syn_it": ["PA"], "syn_en": ["blood pressure"],
        "parents": ["363787002"], "expected_fact_types": ["vital_sign"],
    },
    "364075005": {
        "fsn_it": "Frequenza cardiaca (entità osservabile)",
        "fsn_en": "Heart rate (observable entity)",
        "pt_it": "Frequenza cardiaca", "pt_en": "Heart rate",
        "syn_it": ["FC"], "syn_en": ["heart rate"],
        "parents": ["363787002"], "expected_fact_types": ["vital_sign"],
    },
    "386725007": {
        "fsn_it": "Temperatura corporea (entità osservabile)",
        "fsn_en": "Body temperature (observable entity)",
        "pt_it": "Temperatura corporea", "pt_en": "Body temperature",
        "syn_it": ["temperatura"],
        "parents": ["363787002"], "expected_fact_types": ["vital_sign"],
    },
    "431314004": {
        "fsn_it": "Saturazione periferica di ossigeno (entità osservabile)",
        "fsn_en": "Peripheral oxygen saturation (observable entity)",
        "pt_it": "Saturazione periferica di ossigeno", "pt_en": "Oxygen saturation",
        "syn_it": ["SpO2", "saturazione"], "syn_en": ["SpO2", "oxygen saturation"],
        "parents": ["363787002"], "expected_fact_types": ["vital_sign"],
    },
    "27113001": {
        "fsn_it": "Peso corporeo (entità osservabile)",
        "fsn_en": "Body weight (observable entity)",
        "pt_it": "Peso corporeo", "pt_en": "Body weight",
        "syn_it": ["peso"],
        "parents": ["363787002"], "expected_fact_types": ["vital_sign"],
    },
    # --- Laboratory tests (observable entity) ---------------------------
    "102299008": {
        "fsn_it": "Creatinina sierica (entità osservabile)",
        "fsn_en": "Serum creatinine (observable entity)",
        "pt_it": "Creatinina sierica", "pt_en": "Serum creatinine",
        "syn_it": ["creatinina"], "syn_en": ["creatinine"],
        "parents": ["363787002"], "expected_fact_types": ["laboratory_test"],
    },
    "33747003": {
        "fsn_it": "Glicemia (entità osservabile)",
        "fsn_en": "Glucose measurement (observable entity)",
        "pt_it": "Glicemia", "pt_en": "Glucose measurement",
        "syn_it": ["glucosio", "glicemia"], "syn_en": ["blood glucose", "glucose"],
        "parents": ["363787002"], "expected_fact_types": ["laboratory_test"],
    },
    "407042007": {
        "fsn_it": "Emoglobina glicata (entità osservabile)",
        "fsn_en": "Glycated hemoglobin measurement (observable entity)",
        "pt_it": "Emoglobina glicata", "pt_en": "HbA1c",
        "syn_it": ["HbA1c", "emoglobina glicosilata"],
        "syn_en": ["glycated hemoglobin", "HbA1c"],
        "parents": ["363787002"], "expected_fact_types": ["laboratory_test"],
        "note": "codice sintetico",
    },
    "53477007": {
        "fsn_it": "Ormone tireostimolante (entità osservabile)",
        "fsn_en": "Thyroid stimulating hormone measurement (observable entity)",
        "pt_it": "Ormone tireostimolante", "pt_en": "Thyrotropin",
        "syn_it": ["TSH"], "syn_en": ["TSH"],
        "parents": ["363787002"], "expected_fact_types": ["laboratory_test"],
    },
    "300151006": {
        "fsn_it": "Emoglobina (entità osservabile)",
        "fsn_en": "Hemoglobin measurement (observable entity)",
        "pt_it": "Emoglobina", "pt_en": "Hemoglobin",
        "syn_it": ["Hb"], "syn_en": ["haemoglobin"],
        "parents": ["363787002"], "expected_fact_types": ["laboratory_test"],
    },
    "88480006": {
        "fsn_it": "Potassio (entità osservabile)",
        "fsn_en": "Potassium measurement (observable entity)",
        "pt_it": "Potassio", "pt_en": "Potassium",
        "parents": ["363787002"], "expected_fact_types": ["laboratory_test"],
    },
    "303796003": {
        "fsn_it": "Velocità di filtrazione glomerulare (entità osservabile)",
        "fsn_en": "Glomerular filtration rate (observable entity)",
        "pt_it": "Velocità di filtrazione glomerulare", "pt_en": "eGFR",
        "syn_it": ["GFR", "eGFR"], "syn_en": ["glomerular filtration rate", "eGFR"],
        "parents": ["363787002"], "expected_fact_types": ["laboratory_test"],
    },
    "31728002": {
        "fsn_it": "Velocità di eritrosedimentazione (entità osservabile)",
        "fsn_en": "Erythrocyte sedimentation rate measurement (observable entity)",
        "pt_it": "Velocità di eritrosedimentazione", "pt_en": "ESR",
        "syn_it": ["VES"], "syn_en": ["erythrocyte sedimentation rate", "ESR"],
        "parents": ["363787002"], "expected_fact_types": ["laboratory_test"],
    },
    "40228008": {
        "fsn_it": "Proteina C reattiva (entità osservabile)",
        "fsn_en": "C reactive protein measurement (observable entity)",
        "pt_it": "Proteina C reattiva", "pt_en": "C-reactive protein",
        "syn_it": ["PCR"], "syn_en": ["C-reactive protein", "CRP"],
        "parents": ["363787002"], "expected_fact_types": ["laboratory_test"],
    },
    # --- Drugs (substance) ----------------------------------------------
    "372250003": {
        "fsn_it": "Metformina (sostanza)",
        "fsn_en": "Metformin (substance)",
        "pt_it": "Metformina", "pt_en": "Metformin",
        "parents": ["105590001"], "expected_fact_types": ["medication", "clinical_decision"],
    },
    "372723003": {
        "fsn_it": "Ramipril (sostanza)",
        "fsn_en": "Ramipril (substance)",
        "pt_it": "Ramipril", "pt_en": "Ramipril",
        "parents": ["105590001"], "expected_fact_types": ["medication", "clinical_decision"],
    },
    "372681007": {
        "fsn_it": "Atorvastatina (sostanza)",
        "fsn_en": "Atorvastatin (substance)",
        "pt_it": "Atorvastatina", "pt_en": "Atorvastatin",
        "parents": ["105590001"], "expected_fact_types": ["medication", "clinical_decision"],
    },
    "429040005": {
        "fsn_it": "Pembrolizumab (sostanza)",
        "fsn_en": "Pembrolizumab (substance)",
        "pt_it": "Pembrolizumab", "pt_en": "Pembrolizumab",
        "syn_it": ["anti-PD-1"], "syn_en": ["anti-PD-1"],
        "parents": ["105590001"], "expected_fact_types": ["medication", "clinical_decision"],
    },
    "389219000": {
        "fsn_it": "Nivolumab (sostanza)",
        "fsn_en": "Nivolumab (substance)",
        "pt_it": "Nivolumab", "pt_en": "Nivolumab",
        "parents": ["105590001"], "expected_fact_types": ["medication", "clinical_decision"],
    },
    # --- Histopathology -------------------------------------------------
    "104910003": {
        "fsn_it": "Melanoma maligno (struttura morfologicamente anomala)",
        "fsn_en": "Malignant melanoma (morphologic abnormality)",
        "pt_it": "Melanoma maligno", "pt_en": "Malignant melanoma",
        "parents": ["279125003"],
        "expected_fact_types": ["histopathology", "diagnosis"],
    },
    "81917007": {
        "fsn_it": "Spessore di Breslow (entità osservabile)",
        "fsn_en": "Breslow thickness measurement (observable entity)",
        "pt_it": "Spessore di Breslow", "pt_en": "Breslow thickness",
        "syn_it": ["Breslow"], "syn_en": ["Breslow"],
        "parents": ["363787002"], "expected_fact_types": ["histopathology"],
        "note": "codice sintetico",
    },
    "285278005": {
        "fsn_it": "Livello di Clark (entità osservabile)",
        "fsn_en": "Clark level (observable entity)",
        "pt_it": "Livello di Clark", "pt_en": "Clark level",
        "syn_it": ["Clark"], "syn_en": ["Clark"],
        "parents": ["363787002"], "expected_fact_types": ["histopathology"],
        "note": "codice sintetico",
    },
    # --- Biomarkers (observable entity) ---------------------------------
    "447206003": {
        "fsn_it": "Mutazione del gene BRAF (entità osservabile)",
        "fsn_en": "BRAF gene mutation (observable entity)",
        "pt_it": "Mutazione del gene BRAF", "pt_en": "BRAF mutation",
        "syn_it": ["BRAF"], "syn_en": ["BRAF"],
        "parents": ["363787002"], "expected_fact_types": ["biomarker"],
    },
    "443761006": {
        "fsn_it": "Espressione del ligando PD-L1 (entità osservabile)",
        "fsn_en": "Programmed death-ligand 1 expression (observable entity)",
        "pt_it": "Espressione del ligando PD-L1", "pt_en": "PD-L1 expression",
        "syn_it": ["PD-L1"], "syn_en": ["PD-L1"],
        "parents": ["363787002"], "expected_fact_types": ["biomarker"],
    },
    "34803002": {
        "fsn_it": "Antigene carcinoembrionario (entità osservabile)",
        "fsn_en": "Carcinoembryonic antigen measurement (observable entity)",
        "pt_it": "Antigene carcinoembrionario", "pt_en": "CEA",
        "syn_it": ["CEA"], "syn_en": ["carcinoembryonic antigen", "CEA"],
        "parents": ["363787002"], "expected_fact_types": ["biomarker"],
    },
}

CONCEPT_HEADER = [
    "id", "effectiveTime", "active", "moduleId", "definitionStatusId",
]
DESCRIPTION_HEADER = [
    "id", "effectiveTime", "active", "moduleId", "conceptId", "languageCode",
    "typeId", "term", "caseSignificanceId",
]
RELATIONSHIP_HEADER = [
    "id", "effectiveTime", "active", "moduleId", "sourceId", "destinationId",
    "relationshipGroup", "typeId", "characteristicTypeId", "modifierId",
]
LANGUAGE_REFSET_HEADER = [
    "id", "effectiveTime", "active", "moduleId", "refsetId",
    "referencedComponentId", "acceptabilityId",
]


def build() -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    concept_rows: list[dict] = []
    description_rows: list[dict] = []
    relationship_rows: list[dict] = []
    lang_rows: list[dict] = []
    description_counter = 0
    relationship_counter = 0

    for concept_id in sorted(CONCEPT_SPECS, key=lambda c: int(c)):
        spec = CONCEPT_SPECS[concept_id]
        concept_rows.append({
            "id": concept_id,
            "effectiveTime": RELEASE_DATE,
            "active": "1",
            "moduleId": MODULE_ID,
            "definitionStatusId": DEFINITION_PRIMITIVE,
        })
        for lang, lang_code in (("it", "it"), ("en", "en")):
            fsn = spec.get(f"fsn_{lang}")
            pt = spec.get(f"pt_{lang}")
            syns = spec.get(f"syn_{lang}", []) or []
            if fsn is None or pt is None:
                continue
            # FSN description.
            description_counter += 1
            description_id = str(100000000 + description_counter)
            description_rows.append({
                "id": description_id,
                "effectiveTime": RELEASE_DATE,
                "active": "1",
                "moduleId": MODULE_ID,
                "conceptId": concept_id,
                "languageCode": lang_code,
                "typeId": FSN_TYPE,
                "term": fsn,
                "caseSignificanceId": "900000000000448009",
            })
            # Preferred-term synonym.
            description_counter += 1
            pt_id = str(100000000 + description_counter)
            description_rows.append({
                "id": pt_id,
                "effectiveTime": RELEASE_DATE,
                "active": "1",
                "moduleId": MODULE_ID,
                "conceptId": concept_id,
                "languageCode": lang_code,
                "typeId": SYNONYM_TYPE,
                "term": pt,
                "caseSignificanceId": "900000000000448009",
            })
            lang_rows.append({
                "id": f"6{pt_id}",
                "effectiveTime": RELEASE_DATE,
                "active": "1",
                "moduleId": MODULE_ID,
                "refsetId": LANG_REFSET_IT if lang == "it" else LANG_REFSET_EN,
                "referencedComponentId": pt_id,
                "acceptabilityId": PREFERRED_ACCEPTABILITY,
            })
            # Acceptable synonyms.
            for syn in syns:
                description_counter += 1
                syn_id = str(100000000 + description_counter)
                description_rows.append({
                    "id": syn_id,
                    "effectiveTime": RELEASE_DATE,
                    "active": "1",
                    "moduleId": MODULE_ID,
                    "conceptId": concept_id,
                    "languageCode": lang_code,
                    "typeId": SYNONYM_TYPE,
                    "term": syn,
                    "caseSignificanceId": "900000000000448009",
                })
                lang_rows.append({
                    "id": f"6{syn_id}",
                    "effectiveTime": RELEASE_DATE,
                    "active": "1",
                    "moduleId": MODULE_ID,
                    "refsetId": LANG_REFSET_IT if lang == "it" else LANG_REFSET_EN,
                    "referencedComponentId": syn_id,
                    "acceptabilityId": ACCEPTABLE_ACCEPTABILITY,
                })
        for parent in spec.get("parents", []):
            relationship_counter += 1
            relationship_rows.append({
                "id": f"3{relationship_counter:08d}",
                "effectiveTime": RELEASE_DATE,
                "active": "1",
                "moduleId": MODULE_ID,
                "sourceId": concept_id,
                "destinationId": parent,
                "relationshipGroup": "0",
                "typeId": IS_A_TYPE,
                "characteristicTypeId": CHARACTERISTIC_INFERRED,
                "modifierId": MODIFIER_SOME,
            })
    return concept_rows, description_rows, lang_rows, relationship_rows


def _write_rf2(path: Path, header: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("|".join(header) + "\n")
        for row in rows:
            handle.write("|".join(str(row[column]) for column in header) + "\n")


def write_manifest() -> None:
    columns = [
        "code", "fsn_it", "fsn_en", "pt_it", "pt_en", "syn_it", "syn_en",
        "parents", "expected_fact_types", "note",
    ]
    manifest_path = HERE / "concepts.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for code in sorted(CONCEPT_SPECS, key=lambda c: int(c)):
            spec = CONCEPT_SPECS[code]
            writer.writerow([
                code,
                spec.get("fsn_it", ""), spec.get("fsn_en", ""),
                spec.get("pt_it", ""), spec.get("pt_en", ""),
                ";".join(spec.get("syn_it", []) or []),
                ";".join(spec.get("syn_en", []) or []),
                ";".join(spec.get("parents", [])),
                ";".join(spec.get("expected_fact_types", [])),
                spec.get("note", ""),
            ])


def write_readme() -> None:
    (HERE / "README.md").write_text(
        f"""# Fixture RF2 sintetica — SNOMED CT

Fixture **sintetica** di un release SNOMED CT (edizione International) usata
dai test della pipeline `snomed`.  Non è un release reale: i codici sono
rappresentativi (alcuni coincidono con codici SNOMED veri, altri sono
approssimazioni) e **nessun valore va trattato come clinicamente
autorevole**.  Per una run reale usare la guida Fase 0 (acquisto release via
MLDS) e puntare `EMR_ANALYZER_SNOMED_RELEASE_DIR` alla directory del release.

Struttura generata da `generate_fixture.py` (eseguilo per rigenerare):

```
Snapshot_20260101/
  Snapshot/
    Content/
      sct2_Concept_Snapshot_INT_{RELEASE_DATE}.txt
      sct2_Description_Snapshot-en_INT_{RELEASE_DATE}.txt
      sct2_Description_Snapshot-it_INT_{RELEASE_DATE}.txt
      sct2_Relationship_Snapshot_INT_{RELEASE_DATE}.txt
    Refset/Language/
      der2_cRefset_LanguageSnapshot-en_INT_{RELEASE_DATE}.txt
      der2_cRefset_LanguageSnapshot-it_INT_{RELEASE_DATE}.txt
concepts.csv     # manifest di asserzione letto dai test
README.md
```

I file RF2 hanno header standard; il loader applica la semantica snapshot
"max effectiveTime + active==1 per id".  `concepts.csv` è l'oracolo dei test:
colonna `expected_fact_types` = bucket atomici attesi da `fact_types_for_concept`.

{len(CONCEPT_SPECS)} concetti: ancore semantiche, diagnosi, sintomi, segni,
reperti radiologici, procedure, parametri vitali, esami di laboratorio,
farmaci, istologia e biomarcatori.
""".strip() + "\n",
        encoding="utf-8",
    )


def main() -> None:
    concept_rows, description_rows, lang_rows, relationship_rows = build()
    _write_rf2(
        CONTENT_DIR / f"sct2_Concept_Snapshot_INT_{RELEASE_DATE}.txt",
        CONCEPT_HEADER, concept_rows,
    )
    _write_rf2(
        CONTENT_DIR / f"sct2_Description_Snapshot-en_INT_{RELEASE_DATE}.txt",
        DESCRIPTION_HEADER,
        [row for row in description_rows if row["languageCode"] == "en"],
    )
    _write_rf2(
        CONTENT_DIR / f"sct2_Description_Snapshot-it_INT_{RELEASE_DATE}.txt",
        DESCRIPTION_HEADER,
        [row for row in description_rows if row["languageCode"] == "it"],
    )
    _write_rf2(
        CONTENT_DIR / f"sct2_Relationship_Snapshot_INT_{RELEASE_DATE}.txt",
        RELATIONSHIP_HEADER, relationship_rows,
    )
    _write_rf2(
        LANG_DIR / f"der2_cRefset_LanguageSnapshot-en_INT_{RELEASE_DATE}.txt",
        LANGUAGE_REFSET_HEADER,
        [row for row in lang_rows if row["refsetId"] == LANG_REFSET_EN],
    )
    _write_rf2(
        LANG_DIR / f"der2_cRefset_LanguageSnapshot-it_INT_{RELEASE_DATE}.txt",
        LANGUAGE_REFSET_HEADER,
        [row for row in lang_rows if row["refsetId"] == LANG_REFSET_IT],
    )
    write_manifest()
    write_readme()
    print(f"Fixture rigenerata: {len(CONCEPT_SPECS)} concetti in {RELEASE_DIR}")


if __name__ == "__main__":
    main()
