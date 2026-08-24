"""Database schema creation and additive migrations for EMR Analyzer."""

import re

from .engine import DatabaseEngine


SCHEMA_VERSION = 15

CREATE_TABLES_SQL = [
    # Patients
    """
    CREATE TABLE IF NOT EXISTS patients (
        id TEXT PRIMARY KEY,
        pseudonym TEXT NOT NULL,
        initials TEXT,
        sex TEXT,
        birth_year INTEGER,
        main_pathology TEXT,
        center TEXT,
        notes TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    # Keyed patient identity fingerprints used only for document routing.
    # Raw names and fiscal codes remain in the source PDF and are not stored
    # in this table.
    """
    CREATE TABLE IF NOT EXISTS patient_identities (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id TEXT NOT NULL UNIQUE REFERENCES patients(id) ON DELETE CASCADE,
        fiscal_code_key TEXT,
        normalized_name_key TEXT,
        birth_date_key TEXT,
        sex TEXT,
        hospital_patient_id_key TEXT,
        confidence REAL DEFAULT 0.0,
        status TEXT DEFAULT 'proposed',
        source_document_id TEXT REFERENCES documents(id) ON DELETE SET NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    # Documents
    """
    CREATE TABLE IF NOT EXISTS documents (
        id TEXT PRIMARY KEY,
        patient_id TEXT NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
        filename TEXT NOT NULL,
        original_path TEXT NOT NULL,
        file_hash TEXT NOT NULL,
        page_count INTEGER DEFAULT 0,
        document_date TEXT,
        document_type TEXT NOT NULL,
        import_date TEXT NOT NULL,
        parsing_status TEXT DEFAULT 'pending',
        extraction_status TEXT DEFAULT 'pending',
        validation_status TEXT DEFAULT 'pending',
        event_count INTEGER DEFAULT 0,
        lab_value_count INTEGER DEFAULT 0,
        error_message TEXT,
        metadata_json TEXT
    )
    """,
    # Provenance for every identifier used to route an imported document.
    """
    CREATE TABLE IF NOT EXISTS document_identity_evidence (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        patient_id TEXT NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
        field_name TEXT NOT NULL,
        value_key TEXT NOT NULL,
        page INTEGER DEFAULT 1,
        bbox_json TEXT,
        extraction_method TEXT NOT NULL,
        confidence REAL DEFAULT 0.0,
        created_at TEXT NOT NULL
    )
    """,
    # Immutable document-level facts from deterministic parsers and the local
    # document model. The Clinical State is derived from these records.
    """
    CREATE TABLE IF NOT EXISTS clinical_evidence (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        evidence_id TEXT NOT NULL UNIQUE,
        patient_id TEXT NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
        document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        category TEXT NOT NULL,
        normalized_entity TEXT NOT NULL,
        fact_type TEXT,
        concept_original TEXT,
        canonical_label TEXT,
        terminology_system TEXT,
        terminology_code TEXT,
        mapping_status TEXT NOT NULL DEFAULT 'unmapped',
        mapping_confidence REAL,
        clinical_relevance TEXT NOT NULL DEFAULT 'accepted_clinical',
        typed_payload_json TEXT NOT NULL DEFAULT '{}',
        assertion TEXT DEFAULT 'present',
        temporality TEXT DEFAULT 'current',
        clinical_status TEXT,
        observed_date TEXT,
        value_text TEXT,
        numeric_value REAL,
        unit TEXT,
        source_page INTEGER,
        source_text TEXT NOT NULL,
        bbox_json TEXT,
        confidence REAL DEFAULT 0.0,
        extraction_method TEXT NOT NULL,
        model_name TEXT,
        prompt_version TEXT,
        schema_version TEXT NOT NULL,
        status TEXT DEFAULT 'proposed',
        data_json TEXT,
        created_at TEXT NOT NULL
    )
    """,
    # Lab values
    """
    CREATE TABLE IF NOT EXISTS lab_values (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id TEXT NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
        document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        parameter_name TEXT NOT NULL,
        normalized_name TEXT NOT NULL,
        value REAL,
        value_text TEXT,
        operator TEXT,
        unit TEXT,
        reference_low REAL,
        reference_high REAL,
        reference_text TEXT,
        is_abnormal INTEGER DEFAULT 0,
        flag TEXT,
        sample_date TEXT,
        biological_material TEXT,
        lab_name TEXT,
        page INTEGER,
        source_text TEXT,
        confidence REAL DEFAULT 1.0,
        validated_by_user INTEGER DEFAULT 0
    )
    """,
    # Clinical State
    """
    CREATE TABLE IF NOT EXISTS clinical_state (
        patient_id TEXT PRIMARY KEY REFERENCES patients(id) ON DELETE CASCADE,
        state_json TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        version INTEGER DEFAULT 1
    )
    """,
    # Validation queue
    """
    CREATE TABLE IF NOT EXISTS validation_queue (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id TEXT NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
        item_type TEXT NOT NULL,
        item_id TEXT NOT NULL,
        issue TEXT NOT NULL,
        severity TEXT DEFAULT 'medium',
        status TEXT DEFAULT 'pending',
        original_value TEXT,
        corrected_value TEXT,
        user_notes TEXT,
        created_at TEXT NOT NULL,
        resolved_at TEXT
    )
    """,
    # Audit log
    """
    CREATE TABLE IF NOT EXISTS audit_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id TEXT NOT NULL,
        action TEXT NOT NULL,
        target_type TEXT,
        target_id TEXT,
        details_json TEXT,
        model_used TEXT,
        model_version TEXT,
        timestamp TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS audit_chain_heads (
        patient_id TEXT PRIMARY KEY,
        last_hash TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    # Clinical Timeline — strictly temporal clinical registry
    """
    CREATE TABLE IF NOT EXISTS clinical_timeline (
        entry_id TEXT PRIMARY KEY,
        patient_id TEXT NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
        date_observed TEXT NOT NULL,
        date_resolved TEXT,
        category TEXT NOT NULL DEFAULT 'other',
        description TEXT NOT NULL,
        source_document_ids TEXT NOT NULL DEFAULT '[]',
        source_texts TEXT NOT NULL DEFAULT '[]',
        status TEXT NOT NULL DEFAULT 'active',
        confidence REAL DEFAULT 0.5,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    # Evidence-based registry v2.  The existing clinical_timeline remains a
    # backwards-compatible projection; these tables are the downstream source
    # of truth and never rewrite normalized document text or laboratory rows.
    """
    CREATE TABLE IF NOT EXISTS clinical_episodes (
        episode_id TEXT PRIMARY KEY,
        patient_id TEXT NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
        category TEXT NOT NULL,
        canonical_entity TEXT NOT NULL,
        onset_date TEXT,
        onset_date_end TEXT,
        onset_precision TEXT NOT NULL DEFAULT 'unknown',
        first_documented_date TEXT,
        resolution_date TEXT,
        status TEXT NOT NULL DEFAULT 'active',
        recurrence_index INTEGER NOT NULL DEFAULT 1,
        previous_episode_id TEXT REFERENCES clinical_episodes(episode_id)
            ON DELETE SET NULL,
        data_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS clinical_events (
        event_id TEXT PRIMARY KEY,
        patient_id TEXT NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
        episode_id TEXT REFERENCES clinical_episodes(episode_id)
            ON DELETE SET NULL,
        category TEXT NOT NULL,
        canonical_entity TEXT NOT NULL,
        summary_short TEXT NOT NULL,
        summary_detail TEXT NOT NULL DEFAULT '',
        anatomical_site TEXT,
        laterality TEXT,
        severity TEXT,
        significance TEXT NOT NULL DEFAULT 'clinically_relevant',
        status TEXT NOT NULL DEFAULT 'active',
        certainty TEXT NOT NULL DEFAULT 'confirmed',
        assertion TEXT NOT NULL DEFAULT 'present',
        first_evidence_date TEXT,
        first_documented_date TEXT,
        date_end TEXT,
        date_precision TEXT NOT NULL DEFAULT 'unknown',
        confidence REAL,
        review_status TEXT NOT NULL DEFAULT 'auto',
        structured_data_json TEXT NOT NULL DEFAULT '{}',
        model_name TEXT,
        prompt_version TEXT,
        schema_version TEXT NOT NULL,
        version INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS clinical_event_evidence (
        link_id TEXT PRIMARY KEY,
        event_id TEXT NOT NULL REFERENCES clinical_events(event_id)
            ON DELETE CASCADE,
        evidence_id TEXT NOT NULL REFERENCES clinical_evidence(evidence_id)
            ON DELETE CASCADE,
        relation TEXT NOT NULL DEFAULT 'supports',
        role TEXT NOT NULL DEFAULT 'core',
        relation_confidence REAL,
        rationale TEXT NOT NULL DEFAULT '',
        included_in_summary INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL,
        UNIQUE(event_id, evidence_id, relation)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS clinical_event_updates (
        update_id TEXT PRIMARY KEY,
        event_id TEXT NOT NULL REFERENCES clinical_events(event_id)
            ON DELETE CASCADE,
        update_date TEXT,
        date_precision TEXT NOT NULL DEFAULT 'unknown',
        summary TEXT NOT NULL,
        status_after TEXT,
        evidence_ids_json TEXT NOT NULL DEFAULT '[]',
        structured_data_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS clinical_event_relations (
        relation_id TEXT PRIMARY KEY,
        patient_id TEXT NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
        source_event_id TEXT NOT NULL REFERENCES clinical_events(event_id)
            ON DELETE CASCADE,
        target_event_id TEXT NOT NULL REFERENCES clinical_events(event_id)
            ON DELETE CASCADE,
        relation_type TEXT NOT NULL,
        confidence REAL,
        rationale TEXT NOT NULL DEFAULT '',
        review_status TEXT NOT NULL DEFAULT 'auto',
        created_at TEXT NOT NULL,
        UNIQUE(source_event_id, target_event_id, relation_type)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS processing_runs (
        run_id TEXT PRIMARY KEY,
        patient_id TEXT NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
        stage TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'running',
        model_name TEXT,
        model_digest TEXT,
        prompt_version TEXT,
        schema_version TEXT NOT NULL,
        parameters_json TEXT NOT NULL DEFAULT '{}',
        started_at TEXT NOT NULL,
        completed_at TEXT,
        error_message TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS processing_manifest (
        manifest_id TEXT PRIMARY KEY,
        run_id TEXT REFERENCES processing_runs(run_id) ON DELETE SET NULL,
        patient_id TEXT NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
        document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        stage TEXT NOT NULL,
        input_hash TEXT NOT NULL,
        pipeline_version TEXT NOT NULL,
        prompt_version TEXT NOT NULL DEFAULT '',
        model_digest TEXT NOT NULL DEFAULT '',
        status TEXT NOT NULL DEFAULT 'pending',
        output_hash TEXT,
        output_count INTEGER NOT NULL DEFAULT 0,
        attempts INTEGER NOT NULL DEFAULT 0,
        error_message TEXT,
        processed_at TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE(document_id, stage, input_hash, pipeline_version,
               prompt_version, model_digest)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS review_decisions (
        decision_id TEXT PRIMARY KEY,
        patient_id TEXT NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
        target_type TEXT NOT NULL,
        target_id TEXT NOT NULL,
        decision TEXT NOT NULL,
        reviewer_id TEXT NOT NULL,
        reviewer_role TEXT NOT NULL,
        reason TEXT NOT NULL DEFAULT '',
        previous_value_json TEXT NOT NULL DEFAULT '{}',
        corrected_value_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS document_text_overlays (
        overlay_id TEXT PRIMARY KEY,
        patient_id TEXT NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
        document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        version INTEGER NOT NULL,
        base_text_hash TEXT NOT NULL,
        corrected_text TEXT NOT NULL,
        reason TEXT NOT NULL DEFAULT '',
        author_id TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'active',
        created_at TEXT NOT NULL,
        UNIQUE(document_id, version)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS medication_courses (
        course_id TEXT PRIMARY KEY,
        patient_id TEXT NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
        normalized_name TEXT NOT NULL,
        original_names_json TEXT NOT NULL DEFAULT '[]',
        indication TEXT,
        intent TEXT,
        lifecycle_status TEXT NOT NULL DEFAULT 'unknown',
        start_date TEXT,
        end_date TEXT,
        dose TEXT,
        route TEXT,
        frequency TEXT,
        adherence TEXT,
        episode_id TEXT REFERENCES clinical_episodes(episode_id)
            ON DELETE SET NULL,
        event_ids_json TEXT NOT NULL DEFAULT '[]',
        data_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS oncology_lines (
        line_id TEXT PRIMARY KEY,
        patient_id TEXT NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
        line_label TEXT NOT NULL,
        regimen_json TEXT NOT NULL DEFAULT '[]',
        setting TEXT,
        intent TEXT,
        start_date TEXT,
        end_date TEXT,
        status TEXT NOT NULL DEFAULT 'unknown',
        cycles_json TEXT NOT NULL DEFAULT '[]',
        modifications_json TEXT NOT NULL DEFAULT '[]',
        toxicities_json TEXT NOT NULL DEFAULT '[]',
        responses_json TEXT NOT NULL DEFAULT '[]',
        progression_event_id TEXT REFERENCES clinical_events(event_id)
            ON DELETE SET NULL,
        event_ids_json TEXT NOT NULL DEFAULT '[]',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS lab_trends (
        trend_id TEXT PRIMARY KEY,
        patient_id TEXT NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
        normalized_name TEXT NOT NULL,
        summary TEXT NOT NULL,
        start_date TEXT,
        end_date TEXT,
        direction TEXT NOT NULL DEFAULT 'variable',
        severity TEXT,
        resolved INTEGER NOT NULL DEFAULT 0,
        lab_value_ids_json TEXT NOT NULL DEFAULT '[]',
        event_id TEXT REFERENCES clinical_events(event_id) ON DELETE SET NULL,
        data_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    # Gold-set validation v1. Reviewer A and B annotations remain independent;
    # only adjudicated rows form the final reference standard.
    """
    CREATE TABLE IF NOT EXISTS gold_set_cases (
        patient_id TEXT PRIMARY KEY REFERENCES patients(id) ON DELETE CASCADE,
        included INTEGER NOT NULL DEFAULT 0,
        split TEXT NOT NULL DEFAULT 'pilot',
        status TEXT NOT NULL DEFAULT 'draft',
        reviewer_a_id TEXT NOT NULL DEFAULT '',
        reviewer_b_id TEXT NOT NULL DEFAULT '',
        adjudicator_id TEXT NOT NULL DEFAULT '',
        reviewer_a_status TEXT NOT NULL DEFAULT 'draft',
        reviewer_b_status TEXT NOT NULL DEFAULT 'draft',
        notes TEXT NOT NULL DEFAULT '',
        locked_at TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS gold_annotations (
        annotation_id TEXT PRIMARY KEY,
        patient_id TEXT NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
        reviewer_slot TEXT NOT NULL,
        reviewer_id TEXT NOT NULL,
        category TEXT NOT NULL,
        canonical_entity TEXT NOT NULL,
        summary_short TEXT NOT NULL,
        summary_detail TEXT NOT NULL DEFAULT '',
        first_evidence_date TEXT,
        first_documented_date TEXT,
        date_end TEXT,
        date_precision TEXT NOT NULL DEFAULT 'unknown',
        status TEXT NOT NULL DEFAULT 'active',
        certainty TEXT NOT NULL DEFAULT 'confirmed',
        assertion TEXT NOT NULL DEFAULT 'present',
        anatomical_site TEXT,
        laterality TEXT,
        severity TEXT,
        significance TEXT NOT NULL DEFAULT 'clinically_relevant',
        episode_key TEXT,
        recurrence_index INTEGER NOT NULL DEFAULT 1,
        evidence_ids_json TEXT NOT NULL DEFAULT '[]',
        source_refs_json TEXT NOT NULL DEFAULT '[]',
        structured_data_json TEXT NOT NULL DEFAULT '{}',
        source_annotation_ids_json TEXT NOT NULL DEFAULT '[]',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS gold_annotation_sources (
        source_id TEXT PRIMARY KEY,
        annotation_id TEXT NOT NULL REFERENCES gold_annotations(annotation_id)
            ON DELETE CASCADE,
        patient_id TEXT NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
        evidence_id TEXT NOT NULL,
        document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE RESTRICT,
        source_page INTEGER,
        source_text TEXT NOT NULL,
        relation TEXT NOT NULL DEFAULT 'supports',
        created_at TEXT NOT NULL,
        UNIQUE(annotation_id, evidence_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS gold_adjudication_decisions (
        decision_id TEXT PRIMARY KEY,
        patient_id TEXT NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
        source_annotation_id TEXT NOT NULL REFERENCES gold_annotations(annotation_id)
            ON DELETE CASCADE,
        final_annotation_id TEXT REFERENCES gold_annotations(annotation_id)
            ON DELETE SET NULL,
        decision TEXT NOT NULL,
        rationale TEXT NOT NULL DEFAULT '',
        adjudicator_id TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE(source_annotation_id)
    )
    """,
    # Pipeline v3: one evidence may retain several exact source passages after
    # cross-document deduplication.  The historical source columns on
    # clinical_evidence remain the primary citation for compatibility.
    """
    CREATE TABLE IF NOT EXISTS evidence_source_refs (
        source_ref_id TEXT PRIMARY KEY,
        evidence_id TEXT NOT NULL REFERENCES clinical_evidence(evidence_id)
            ON DELETE CASCADE,
        document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE RESTRICT,
        source_page INTEGER,
        bbox_json TEXT,
        sentence_refs_json TEXT NOT NULL DEFAULT '[]',
        passage TEXT NOT NULL,
        source_role TEXT NOT NULL DEFAULT 'primary',
        created_at TEXT NOT NULL,
        UNIQUE(evidence_id, document_id, source_page, passage)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS excluded_evidence (
        excluded_id TEXT PRIMARY KEY,
        patient_id TEXT NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
        document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        disposition TEXT NOT NULL,
        reason_code TEXT NOT NULL,
        fact_type TEXT,
        concept TEXT,
        source_page INTEGER,
        bbox_json TEXT,
        sentence_refs_json TEXT NOT NULL DEFAULT '[]',
        source_text TEXT NOT NULL,
        extraction_method TEXT NOT NULL,
        model_name TEXT,
        prompt_version TEXT,
        review_status TEXT NOT NULL DEFAULT 'auto',
        data_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        UNIQUE(document_id, disposition, source_text)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS terminology_mappings (
        mapping_id TEXT PRIMARY KEY,
        normalized_concept TEXT NOT NULL,
        fact_type TEXT NOT NULL DEFAULT '',
        canonical_label TEXT NOT NULL,
        terminology_system TEXT,
        terminology_code TEXT,
        original_unit TEXT NOT NULL DEFAULT '',
        canonical_unit TEXT,
        multiplier REAL,
        offset REAL,
        mapping_status TEXT NOT NULL DEFAULT 'candidate',
        mapping_confidence REAL,
        source TEXT NOT NULL DEFAULT 'local_dictionary',
        review_status TEXT NOT NULL DEFAULT 'auto',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE(normalized_concept, fact_type, original_unit)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS evidence_duplicate_groups (
        duplicate_group_id TEXT PRIMARY KEY,
        patient_id TEXT NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
        canonical_evidence_id TEXT NOT NULL
            REFERENCES clinical_evidence(evidence_id) ON DELETE CASCADE,
        occurrence_key TEXT NOT NULL,
        decision_reason TEXT NOT NULL DEFAULT '',
        review_status TEXT NOT NULL DEFAULT 'auto',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE(patient_id, occurrence_key)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS evidence_duplicate_members (
        duplicate_group_id TEXT NOT NULL
            REFERENCES evidence_duplicate_groups(duplicate_group_id)
            ON DELETE CASCADE,
        evidence_id TEXT NOT NULL REFERENCES clinical_evidence(evidence_id)
            ON DELETE CASCADE,
        source_role TEXT NOT NULL DEFAULT 'duplicate_source',
        created_at TEXT NOT NULL,
        PRIMARY KEY(duplicate_group_id, evidence_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS evidence_relations (
        relation_id TEXT PRIMARY KEY,
        patient_id TEXT NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
        source_evidence_id TEXT NOT NULL
            REFERENCES clinical_evidence(evidence_id) ON DELETE CASCADE,
        target_evidence_id TEXT NOT NULL
            REFERENCES clinical_evidence(evidence_id) ON DELETE CASCADE,
        relation_type TEXT NOT NULL,
        direction TEXT NOT NULL DEFAULT 'directed',
        weight REAL NOT NULL,
        cluster_effect TEXT NOT NULL,
        rationale TEXT NOT NULL DEFAULT '',
        rule_features_json TEXT NOT NULL DEFAULT '{}',
        model_votes_json TEXT NOT NULL DEFAULT '[]',
        generation_method TEXT NOT NULL DEFAULT 'hybrid',
        review_status TEXT NOT NULL DEFAULT 'auto',
        created_at TEXT NOT NULL,
        UNIQUE(source_evidence_id, target_evidence_id, relation_type)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS evidence_relation_adjudication_cache (
        patient_id TEXT NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
        namespace TEXT NOT NULL,
        candidate_id TEXT NOT NULL,
        round_index INTEGER NOT NULL DEFAULT 0,
        cache_key TEXT NOT NULL UNIQUE,
        source_evidence_id TEXT NOT NULL
            REFERENCES clinical_evidence(evidence_id) ON DELETE CASCADE,
        target_evidence_id TEXT NOT NULL
            REFERENCES clinical_evidence(evidence_id) ON DELETE CASCADE,
        decision_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY(patient_id, namespace, candidate_id, round_index)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS clinical_event_claims (
        claim_id TEXT PRIMARY KEY,
        event_id TEXT NOT NULL REFERENCES clinical_events(event_id)
            ON DELETE CASCADE,
        claim_type TEXT NOT NULL,
        text TEXT NOT NULL,
        certainty TEXT NOT NULL DEFAULT 'confirmed',
        review_status TEXT NOT NULL DEFAULT 'auto',
        position INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS clinical_event_claim_sources (
        claim_id TEXT NOT NULL REFERENCES clinical_event_claims(claim_id)
            ON DELETE CASCADE,
        source_type TEXT NOT NULL,
        source_id TEXT NOT NULL,
        source_role TEXT NOT NULL DEFAULT 'supports',
        created_at TEXT NOT NULL,
        PRIMARY KEY(claim_id, source_type, source_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS clinical_hypotheses (
        hypothesis_id TEXT PRIMARY KEY,
        patient_id TEXT NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
        source_event_id TEXT NOT NULL REFERENCES clinical_events(event_id)
            ON DELETE CASCADE,
        target_event_id TEXT NOT NULL REFERENCES clinical_events(event_id)
            ON DELETE CASCADE,
        hypothesis_type TEXT NOT NULL,
        strength TEXT NOT NULL DEFAULT 'weak',
        known_mechanism INTEGER NOT NULL DEFAULT 0,
        rationale TEXT NOT NULL,
        confounders_json TEXT NOT NULL DEFAULT '[]',
        survival_score REAL,
        status TEXT NOT NULL DEFAULT 'needs_review',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE(source_event_id, target_event_id, hypothesis_type)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS clinical_category_registry (
        category TEXT PRIMARY KEY,
        display_name TEXT NOT NULL,
        object_level TEXT NOT NULL DEFAULT 'event',
        enabled INTEGER NOT NULL DEFAULT 1,
        is_builtin INTEGER NOT NULL DEFAULT 0,
        data_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS gold_atomic_annotations (
        annotation_id TEXT PRIMARY KEY,
        patient_id TEXT NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
        document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE RESTRICT,
        reviewer_slot TEXT NOT NULL,
        reviewer_id TEXT NOT NULL,
        annotation_kind TEXT NOT NULL DEFAULT 'evidence',
        fact_type TEXT,
        concept_original TEXT,
        canonical_label TEXT,
        observation_date TEXT,
        date_precision TEXT NOT NULL DEFAULT 'unknown',
        polarity TEXT,
        disposition TEXT NOT NULL DEFAULT 'accepted_clinical',
        exclusion_reason TEXT,
        source_page INTEGER,
        bbox_json TEXT,
        sentence_refs_json TEXT NOT NULL DEFAULT '[]',
        source_text TEXT NOT NULL,
        value_json TEXT NOT NULL DEFAULT '{}',
        duplicate_of_annotation_id TEXT
            REFERENCES gold_atomic_annotations(annotation_id)
            ON DELETE SET NULL,
        status TEXT NOT NULL DEFAULT 'draft',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    # Hospital patient IDs are many-to-one: a person legitimately holds
    # several IDs across the units of a hospital trust (one per referral
    # path), so a single column cannot represent them.  Only keyed digests
    # are stored; the raw values stay in the source PDFs.
    """
    CREATE TABLE IF NOT EXISTS patient_hospital_ids (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id TEXT NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
        hospital_patient_id_key TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE(patient_id, hospital_patient_id_key)
    )
    """,
    # Schema version tracking
    """
    CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER PRIMARY KEY,
        applied_at TEXT NOT NULL
    )
    """,
    # Clinical query chat — per-patient trace of prompts and responses,
    # like a ChatGPT-style conversation history.  role is 'user' or
    # 'assistant'; context_mode marks answers generated with the previous
    # Q&A embedded as conversational context (1) or not (0).
    """
    CREATE TABLE IF NOT EXISTS clinical_chat (
        id TEXT PRIMARY KEY,
        patient_id TEXT NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        model_used TEXT,
        context_mode INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL
    )
    """,
]

INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_docs_patient ON documents(patient_id)",
    "CREATE INDEX IF NOT EXISTS idx_docs_type ON documents(document_type)",
    "CREATE INDEX IF NOT EXISTS idx_docs_status ON documents(parsing_status)",
    "CREATE INDEX IF NOT EXISTS idx_identity_patient ON patient_identities(patient_id)",
    "CREATE INDEX IF NOT EXISTS idx_identity_cf ON patient_identities(fiscal_code_key)",
    "CREATE INDEX IF NOT EXISTS idx_identity_name_birth ON patient_identities(normalized_name_key, birth_date_key)",
    "CREATE INDEX IF NOT EXISTS idx_doc_identity_doc ON document_identity_evidence(document_id)",
    "CREATE INDEX IF NOT EXISTS idx_evidence_patient ON clinical_evidence(patient_id)",
    "CREATE INDEX IF NOT EXISTS idx_evidence_doc ON clinical_evidence(document_id)",
    "CREATE INDEX IF NOT EXISTS idx_evidence_category ON clinical_evidence(category)",
    "CREATE INDEX IF NOT EXISTS idx_evidence_date ON clinical_evidence(observed_date)",
    "CREATE INDEX IF NOT EXISTS idx_lab_patient ON lab_values(patient_id)",
    "CREATE INDEX IF NOT EXISTS idx_lab_doc ON lab_values(document_id)",
    "CREATE INDEX IF NOT EXISTS idx_lab_name ON lab_values(normalized_name)",
    "CREATE INDEX IF NOT EXISTS idx_lab_date ON lab_values(sample_date)",
    "CREATE INDEX IF NOT EXISTS idx_valid_patient ON validation_queue(patient_id)",
    "CREATE INDEX IF NOT EXISTS idx_valid_status ON validation_queue(status)",
    "CREATE INDEX IF NOT EXISTS idx_audit_patient ON audit_log(patient_id)",
    "CREATE INDEX IF NOT EXISTS idx_timeline_patient ON clinical_timeline(patient_id)",
    "CREATE INDEX IF NOT EXISTS idx_timeline_date ON clinical_timeline(date_observed)",
    "CREATE INDEX IF NOT EXISTS idx_timeline_category ON clinical_timeline(category)",
    "CREATE INDEX IF NOT EXISTS idx_timeline_status ON clinical_timeline(status)",
    "CREATE INDEX IF NOT EXISTS idx_episode_patient ON clinical_episodes(patient_id)",
    "CREATE INDEX IF NOT EXISTS idx_episode_entity ON clinical_episodes(patient_id, category, canonical_entity)",
    "CREATE INDEX IF NOT EXISTS idx_event_patient_date ON clinical_events(patient_id, first_evidence_date)",
    "CREATE INDEX IF NOT EXISTS idx_event_patient_category ON clinical_events(patient_id, category)",
    "CREATE INDEX IF NOT EXISTS idx_event_patient_status ON clinical_events(patient_id, status)",
    "CREATE INDEX IF NOT EXISTS idx_event_review ON clinical_events(review_status)",
    "CREATE INDEX IF NOT EXISTS idx_event_evidence_event ON clinical_event_evidence(event_id)",
    "CREATE INDEX IF NOT EXISTS idx_event_evidence_evidence ON clinical_event_evidence(evidence_id)",
    "CREATE INDEX IF NOT EXISTS idx_event_update_event_date ON clinical_event_updates(event_id, update_date)",
    "CREATE INDEX IF NOT EXISTS idx_event_relation_patient ON clinical_event_relations(patient_id)",
    "CREATE INDEX IF NOT EXISTS idx_run_patient_stage ON processing_runs(patient_id, stage, started_at)",
    "CREATE INDEX IF NOT EXISTS idx_manifest_doc_stage ON processing_manifest(document_id, stage)",
    "CREATE INDEX IF NOT EXISTS idx_manifest_patient_status ON processing_manifest(patient_id, stage, status)",
    "CREATE INDEX IF NOT EXISTS idx_review_patient_target ON review_decisions(patient_id, target_type, target_id)",
    "CREATE INDEX IF NOT EXISTS idx_overlay_document ON document_text_overlays(document_id, version)",
    "CREATE INDEX IF NOT EXISTS idx_medication_patient ON medication_courses(patient_id, normalized_name)",
    "CREATE INDEX IF NOT EXISTS idx_oncology_line_patient ON oncology_lines(patient_id, start_date)",
    "CREATE INDEX IF NOT EXISTS idx_lab_trend_patient ON lab_trends(patient_id, normalized_name)",
    "CREATE INDEX IF NOT EXISTS idx_gold_case_split ON gold_set_cases(included, split, status)",
    "CREATE INDEX IF NOT EXISTS idx_gold_annotation_patient_slot ON gold_annotations(patient_id, reviewer_slot)",
    "CREATE INDEX IF NOT EXISTS idx_gold_annotation_entity ON gold_annotations(patient_id, category, canonical_entity)",
    "CREATE INDEX IF NOT EXISTS idx_gold_source_document ON gold_annotation_sources(document_id, patient_id)",
    "CREATE INDEX IF NOT EXISTS idx_gold_decision_patient ON gold_adjudication_decisions(patient_id, decision)",
    "CREATE INDEX IF NOT EXISTS idx_hpid_patient ON patient_hospital_ids(patient_id)",
    "CREATE INDEX IF NOT EXISTS idx_hpid_key ON patient_hospital_ids(hospital_patient_id_key)",
    "CREATE INDEX IF NOT EXISTS idx_chat_patient ON clinical_chat(patient_id)",
    "CREATE INDEX IF NOT EXISTS idx_chat_patient_created ON clinical_chat(patient_id, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_evidence_relevance ON clinical_evidence(patient_id, clinical_relevance)",
    "CREATE INDEX IF NOT EXISTS idx_evidence_fact_type ON clinical_evidence(patient_id, fact_type)",
    "CREATE INDEX IF NOT EXISTS idx_source_ref_evidence ON evidence_source_refs(evidence_id)",
    "CREATE INDEX IF NOT EXISTS idx_source_ref_document ON evidence_source_refs(document_id)",
    "CREATE INDEX IF NOT EXISTS idx_excluded_patient_status ON excluded_evidence(patient_id, disposition, review_status)",
    "CREATE INDEX IF NOT EXISTS idx_excluded_document ON excluded_evidence(document_id)",
    "CREATE INDEX IF NOT EXISTS idx_mapping_lookup ON terminology_mappings(normalized_concept, fact_type)",
    "CREATE INDEX IF NOT EXISTS idx_duplicate_patient ON evidence_duplicate_groups(patient_id, review_status)",
    "CREATE INDEX IF NOT EXISTS idx_evidence_relation_patient ON evidence_relations(patient_id, cluster_effect)",
    "CREATE INDEX IF NOT EXISTS idx_relation_cache_patient ON evidence_relation_adjudication_cache(patient_id, namespace)",
    "CREATE INDEX IF NOT EXISTS idx_claim_event ON clinical_event_claims(event_id, position)",
    "CREATE INDEX IF NOT EXISTS idx_claim_source ON clinical_event_claim_sources(source_type, source_id)",
    "CREATE INDEX IF NOT EXISTS idx_hypothesis_patient ON clinical_hypotheses(patient_id, status)",
    "CREATE INDEX IF NOT EXISTS idx_gold_atomic_patient ON gold_atomic_annotations(patient_id, reviewer_slot, status)",
    "CREATE INDEX IF NOT EXISTS idx_gold_atomic_document ON gold_atomic_annotations(document_id)",
]


def init_database(db: DatabaseEngine) -> None:
    """Create all tables and indexes if they don't exist."""
    from datetime import datetime

    with db:
        for sql in CREATE_TABLES_SQL:
            db.execute(sql)

        # ---- Schema migrations (safe ALTER TABLE) -------------------------
        # Add columns that may be missing in databases created by older versions.
        _safe_add_column(db, "patients", "initials", "TEXT")
        # v7: support textual lab results (e.g. "NEGATIVO", "POSITIVO")
        _safe_add_column(db, "lab_values", "value_text", "TEXT")
        # v8: provenance of canonical dedup — entry_ids merged into a survivor
        _safe_add_column(db, "clinical_timeline", "merged_into_ids", "TEXT")
        # v9: user-confirmed timeline entries for the golden validation set
        _safe_add_column(db, "clinical_timeline", "is_golden", "INTEGER DEFAULT 0")

        # v12: evidence-level clinical structure and tamper-evident audit
        # metadata. Existing rows remain valid and are not rewritten.
        for column, col_type in (
            ("document_date", "TEXT"),
            ("observed_date_end", "TEXT"),
            ("date_precision", "TEXT DEFAULT 'unknown'"),
            ("date_source", "TEXT"),
            ("anatomical_site", "TEXT"),
            ("laterality", "TEXT"),
            ("severity", "TEXT"),
            ("significance", "TEXT DEFAULT 'clinically_relevant'"),
            ("certainty", "TEXT DEFAULT 'confirmed'"),
            ("fact_type", "TEXT"),
            ("concept_original", "TEXT"),
            ("canonical_label", "TEXT"),
            ("terminology_system", "TEXT"),
            ("terminology_code", "TEXT"),
            ("mapping_status", "TEXT DEFAULT 'unmapped'"),
            ("mapping_confidence", "REAL"),
            (
                "clinical_relevance",
                "TEXT DEFAULT 'accepted_clinical'",
            ),
            ("typed_payload_json", "TEXT DEFAULT '{}'"),
        ):
            _safe_add_column(db, "clinical_evidence", column, col_type)
        _safe_add_column(
            db, "terminology_mappings", "mapping_confidence", "REAL"
        )
        _safe_add_column(
            db, "clinical_event_evidence", "role", "TEXT DEFAULT 'core'"
        )
        for column, col_type in (
            ("actor_id", "TEXT DEFAULT 'system'"),
            ("actor_role", "TEXT DEFAULT 'system'"),
            ("run_id", "TEXT"),
            ("input_hash", "TEXT"),
            ("prompt_hash", "TEXT"),
            ("previous_hash", "TEXT"),
            ("entry_hash", "TEXT"),
        ):
            _safe_add_column(db, "audit_log", column, col_type)

        # v10: a person may hold several hospital patient IDs (one per unit).
        # Fold any legacy single-column value into the multi-valued table so
        # both storage forms are consistent; idempotent (UNIQUE + OR IGNORE).
        db.execute(
            """INSERT OR IGNORE INTO patient_hospital_ids
               (patient_id, hospital_patient_id_key, created_at, updated_at)
               SELECT patient_id, hospital_patient_id_key, created_at, updated_at
               FROM patient_identities WHERE hospital_patient_id_key IS NOT NULL"""
        )

        _seed_clinical_categories(db)

        # Indexes that target columns introduced by a migration must only be
        # created after the corresponding ALTER TABLE statements.  This keeps
        # upgrades from pre-v14 project databases safe and idempotent.
        for sql in INDEXES_SQL:
            db.execute(sql)

        # Set schema version
        cursor = db.execute("SELECT MAX(version) FROM schema_version")
        current = cursor.fetchone()[0]

        if current is None or current < SCHEMA_VERSION:
            db.execute(
                "INSERT OR REPLACE INTO schema_version (version, applied_at) VALUES (?, ?)",
                (SCHEMA_VERSION, datetime.now().isoformat()),
            )

        _init_event_fts(db)

        # Audit records may be deleted only by the explicit patient-retention
        # workflow; ordinary application code cannot silently rewrite them.
        db.execute(
            """CREATE TRIGGER IF NOT EXISTS audit_log_no_update
               BEFORE UPDATE ON audit_log
               BEGIN
                   SELECT RAISE(ABORT, 'audit_log is append-only');
               END"""
        )


def _safe_add_column(db: DatabaseEngine, table: str, column: str,
                     col_type: str) -> None:
    """Add a column if it doesn't already exist (SQLite doesn't have
    ``ALTER TABLE … ADD COLUMN IF NOT EXISTS``)."""
    identifier = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
    if not identifier.fullmatch(table) or not identifier.fullmatch(column):
        raise ValueError("Nome tabella/colonna non valido")
    columns = {
        row["name"] for row in db.execute(
            f'PRAGMA table_info("{table}")'
        ).fetchall()
    }
    if column in columns:
        return
    db.execute(f'ALTER TABLE "{table}" ADD COLUMN "{column}" {col_type}')


def _seed_clinical_categories(db: DatabaseEngine) -> None:
    """Seed the extensible category catalog without overwriting user rows."""
    from datetime import datetime
    from ..models.clinical_registry import EVENT_CATEGORIES

    now = datetime.now().isoformat()
    evidence_categories = (
        ("medication", "Farmaco/terapia", "evidence"),
        ("laboratory_test", "Esame di laboratorio", "evidence"),
        ("radiology_finding", "Reperto radiologico", "evidence"),
        ("instrumental_finding", "Reperto strumentale", "evidence"),
        ("diagnosis", "Diagnosi", "both"),
        ("symptom", "Sintomo", "both"),
        ("clinical_decision", "Decisione clinica", "evidence"),
        ("procedure", "Procedura", "both"),
        ("clinical_sign", "Segno clinico", "evidence"),
        ("vital_sign", "Parametro vitale", "evidence"),
        ("histopathology", "Istologia", "both"),
    )
    categories = {
        category: [label, level]
        for category, label, level in evidence_categories
    }
    for category in EVENT_CATEGORIES:
        if category in categories:
            categories[category][1] = "both"
        else:
            categories[category] = [
                category.replace("_", " ").capitalize(), "event",
            ]
    db.executemany(
        """INSERT INTO clinical_category_registry
           (category, display_name, object_level, enabled, is_builtin,
            data_json, created_at, updated_at)
           VALUES (?, ?, ?, 1, 1, '{}', ?, ?)
           ON CONFLICT(category) DO UPDATE SET
             display_name=excluded.display_name,
             object_level=excluded.object_level,
             updated_at=excluded.updated_at
           WHERE clinical_category_registry.is_builtin=1""",
        [(category, label, level, now, now)
         for category, (label, level) in categories.items()],
    )


def _init_event_fts(db: DatabaseEngine) -> None:
    """Create the local full-text event index when SQLite supports FTS5."""
    try:
        existed = db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            ("clinical_events_fts",),
        ).fetchone() is not None
        db.execute(
            """CREATE VIRTUAL TABLE IF NOT EXISTS clinical_events_fts
               USING fts5(event_id UNINDEXED, patient_id UNINDEXED,
                          canonical_entity, summary_short, summary_detail,
                          content='clinical_events', content_rowid='rowid',
                          tokenize='unicode61 remove_diacritics 2')"""
        )
        db.execute(
            """CREATE TRIGGER IF NOT EXISTS clinical_events_fts_ai
               AFTER INSERT ON clinical_events BEGIN
                 INSERT INTO clinical_events_fts(
                   rowid, event_id, patient_id, canonical_entity,
                   summary_short, summary_detail
                 ) VALUES (
                   new.rowid, new.event_id, new.patient_id,
                   new.canonical_entity, new.summary_short, new.summary_detail
                 );
               END"""
        )
        db.execute(
            """CREATE TRIGGER IF NOT EXISTS clinical_events_fts_ad
               AFTER DELETE ON clinical_events BEGIN
                 INSERT INTO clinical_events_fts(
                   clinical_events_fts, rowid, event_id, patient_id,
                   canonical_entity, summary_short, summary_detail
                 ) VALUES (
                   'delete', old.rowid, old.event_id, old.patient_id,
                   old.canonical_entity, old.summary_short, old.summary_detail
                 );
               END"""
        )
        db.execute(
            """CREATE TRIGGER IF NOT EXISTS clinical_events_fts_au
               AFTER UPDATE ON clinical_events BEGIN
                 INSERT INTO clinical_events_fts(
                   clinical_events_fts, rowid, event_id, patient_id,
                   canonical_entity, summary_short, summary_detail
                 ) VALUES (
                   'delete', old.rowid, old.event_id, old.patient_id,
                   old.canonical_entity, old.summary_short, old.summary_detail
                 );
                 INSERT INTO clinical_events_fts(
                   rowid, event_id, patient_id, canonical_entity,
                   summary_short, summary_detail
                 ) VALUES (
                   new.rowid, new.event_id, new.patient_id,
                   new.canonical_entity, new.summary_short, new.summary_detail
                 );
               END"""
        )
        # Populate only an index created over an existing registry. Rebuilding
        # on every startup scales poorly with large multi-patient projects.
        if not existed:
            db.execute(
                "INSERT INTO clinical_events_fts(clinical_events_fts) "
                "VALUES('rebuild')"
            )
    except Exception as exc:
        # FTS is an optional acceleration layer. Only suppress the known
        # feature-availability case; schema or trigger defects must surface.
        message = str(exc).lower()
        if "no such module: fts5" not in message:
            raise


def reset_stale_processing(db: DatabaseEngine) -> None:
    """Reset 'processing' document statuses left by an interrupted run.

    Called at application startup, when no extraction queue can be
    running: every 'processing' flag is stale by definition and would
    otherwise block those documents in the queue forever.
    """
    with db:
        db.execute(
            "UPDATE documents SET parsing_status='pending' "
            "WHERE parsing_status='processing'"
        )
        db.execute(
            "UPDATE documents SET extraction_status='pending' "
            "WHERE extraction_status='processing'"
        )


def drop_all_tables(db: DatabaseEngine) -> None:
    """Drop all tables (for testing/reset). Use with caution."""
    tables = [
        "clinical_events_fts", "clinical_event_relations",
        "clinical_event_claim_sources", "clinical_event_claims",
        "clinical_hypotheses", "evidence_relations",
        "evidence_relation_adjudication_cache",
        "evidence_duplicate_members", "evidence_duplicate_groups",
        "evidence_source_refs", "excluded_evidence",
        "terminology_mappings", "gold_atomic_annotations",
        "clinical_category_registry",
        "clinical_event_updates", "clinical_event_evidence",
        "gold_adjudication_decisions", "gold_annotation_sources",
        "gold_annotations", "gold_set_cases",
        "lab_trends", "oncology_lines", "medication_courses",
        "document_text_overlays", "review_decisions",
        "processing_manifest", "processing_runs", "clinical_events",
        "clinical_episodes", "audit_log", "audit_chain_heads",
        "validation_queue",
        "clinical_chat", "clinical_state", "lab_values", "clinical_timeline",
        "clinical_evidence", "document_identity_evidence",
        "patient_hospital_ids", "patient_identities",
        "documents", "patients", "schema_version",
    ]
    with db:
        for table in tables:
            db.execute(f"DROP TABLE IF EXISTS {table}")
