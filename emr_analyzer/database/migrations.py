"""Database schema creation and migration for EMR Analyzer."""

from .engine import DatabaseEngine


SCHEMA_VERSION = 8

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
    # Docling elements
    """
    CREATE TABLE IF NOT EXISTS docling_elements (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        element_id TEXT NOT NULL,
        page INTEGER,
        type TEXT NOT NULL,
        text TEXT,
        section TEXT,
        confidence REAL DEFAULT 1.0,
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
    # One document-level block that groups the atomic evidence spans without
    # replacing them as the auditable source of truth.
    """
    CREATE TABLE IF NOT EXISTS document_clinical_projections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        projection_id TEXT NOT NULL UNIQUE,
        patient_id TEXT NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
        document_id TEXT NOT NULL UNIQUE REFERENCES documents(id) ON DELETE CASCADE,
        projection_json TEXT NOT NULL,
        evidence_count INTEGER DEFAULT 0,
        observation_count INTEGER DEFAULT 0,
        consolidation_method TEXT NOT NULL,
        model_name TEXT,
        prompt_version TEXT,
        schema_version TEXT NOT NULL,
        status TEXT DEFAULT 'proposed',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    # Clinical Events (Event Store)
    """
    CREATE TABLE IF NOT EXISTS clinical_events (
        event_id TEXT PRIMARY KEY,
        patient_id TEXT NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
        event_date TEXT NOT NULL,
        event_type TEXT NOT NULL,
        entity TEXT NOT NULL,
        value REAL,
        unit TEXT,
        status TEXT DEFAULT 'proposed',
        source_document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        page INTEGER,
        source_text TEXT NOT NULL,
        confidence REAL DEFAULT 0.0,
        validated_by_user INTEGER DEFAULT 0,
        validated_at TEXT,
        user_notes TEXT,
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
    # Clinical State deltas (history)
    """
    CREATE TABLE IF NOT EXISTS clinical_state_deltas (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id TEXT NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
        delta_json TEXT NOT NULL,
        document_id TEXT REFERENCES documents(id) ON DELETE SET NULL,
        applied_at TEXT NOT NULL,
        auto_applied INTEGER DEFAULT 0
    )
    """,
    # One auditable execution of the longitudinal Clinical State builder.
    # Previous runs are retained and marked as superseded; this makes a full
    # reconstruction reproducible without mutating its source ledger.
    """
    CREATE TABLE IF NOT EXISTS clinical_state_runs (
        run_id TEXT PRIMARY KEY,
        patient_id TEXT NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
        model_name TEXT NOT NULL,
        prompt_version TEXT NOT NULL,
        schema_version TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending',
        source_signature TEXT NOT NULL,
        document_count INTEGER DEFAULT 0,
        processed_count INTEGER DEFAULT 0,
        current_document_id TEXT REFERENCES documents(id) ON DELETE SET NULL,
        capability_json TEXT,
        error_message TEXT,
        started_at TEXT NOT NULL,
        completed_at TEXT
    )
    """,
    # Immutable bitemporal ledger emitted by the second LLM. valid_* describes
    # when a fact was clinically true; asserted_at records when it entered the
    # documentary record. Corrections link to, but never erase, prior evidence.
    """
    CREATE TABLE IF NOT EXISTS longitudinal_evidence (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        evidence_id TEXT NOT NULL UNIQUE,
        run_id TEXT NOT NULL REFERENCES clinical_state_runs(run_id) ON DELETE CASCADE,
        patient_id TEXT NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
        document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        operation TEXT NOT NULL,
        target_evidence_id TEXT REFERENCES longitudinal_evidence(evidence_id) ON DELETE SET NULL,
        category TEXT NOT NULL,
        normalized_entity TEXT NOT NULL,
        assertion TEXT DEFAULT 'present',
        clinical_status TEXT,
        valid_start_date TEXT,
        valid_end_date TEXT,
        asserted_at TEXT NOT NULL,
        date_precision TEXT DEFAULT 'unknown',
        original_date_expression TEXT,
        anchor_date TEXT,
        date_derivation_rule TEXT,
        value_text TEXT,
        numeric_value REAL,
        unit TEXT,
        grade INTEGER,
        grading_system TEXT,
        source_page INTEGER,
        source_text TEXT NOT NULL,
        confidence REAL DEFAULT 0.0,
        objective_basis INTEGER DEFAULT 0,
        correction_explicit INTEGER DEFAULT 0,
        canonical_status TEXT NOT NULL DEFAULT 'active',
        model_name TEXT NOT NULL,
        prompt_version TEXT NOT NULL,
        schema_version TEXT NOT NULL,
        data_json TEXT,
        created_at TEXT NOT NULL
    )
    """,
    # Per-document checkpoint and raw delta. A stopped run can resume without
    # repeating already completed LLM calls.
    """
    CREATE TABLE IF NOT EXISTS clinical_state_checkpoints (
        run_id TEXT NOT NULL REFERENCES clinical_state_runs(run_id) ON DELETE CASCADE,
        document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        sequence_index INTEGER NOT NULL,
        temporal_group TEXT,
        document_date TEXT,
        date_source TEXT,
        status TEXT NOT NULL DEFAULT 'pending',
        delta_json TEXT,
        processed_at TEXT,
        error_message TEXT,
        PRIMARY KEY (run_id, document_id)
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
    # Schema version tracking
    """
    CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER PRIMARY KEY,
        applied_at TEXT NOT NULL
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
    "CREATE INDEX IF NOT EXISTS idx_projection_patient ON document_clinical_projections(patient_id)",
    "CREATE INDEX IF NOT EXISTS idx_projection_document ON document_clinical_projections(document_id)",
    "CREATE INDEX IF NOT EXISTS idx_events_patient ON clinical_events(patient_id)",
    "CREATE INDEX IF NOT EXISTS idx_events_date ON clinical_events(event_date)",
    "CREATE INDEX IF NOT EXISTS idx_events_type ON clinical_events(event_type)",
    "CREATE INDEX IF NOT EXISTS idx_events_doc ON clinical_events(source_document_id)",
    "CREATE INDEX IF NOT EXISTS idx_lab_patient ON lab_values(patient_id)",
    "CREATE INDEX IF NOT EXISTS idx_lab_doc ON lab_values(document_id)",
    "CREATE INDEX IF NOT EXISTS idx_lab_name ON lab_values(normalized_name)",
    "CREATE INDEX IF NOT EXISTS idx_lab_date ON lab_values(sample_date)",
    "CREATE INDEX IF NOT EXISTS idx_valid_patient ON validation_queue(patient_id)",
    "CREATE INDEX IF NOT EXISTS idx_valid_status ON validation_queue(status)",
    "CREATE INDEX IF NOT EXISTS idx_audit_patient ON audit_log(patient_id)",
    "CREATE INDEX IF NOT EXISTS idx_elements_doc ON docling_elements(document_id)",
    "CREATE INDEX IF NOT EXISTS idx_state_runs_patient ON clinical_state_runs(patient_id)",
    "CREATE INDEX IF NOT EXISTS idx_state_runs_status ON clinical_state_runs(status)",
    "CREATE INDEX IF NOT EXISTS idx_longitudinal_patient ON longitudinal_evidence(patient_id)",
    "CREATE INDEX IF NOT EXISTS idx_longitudinal_run ON longitudinal_evidence(run_id)",
    "CREATE INDEX IF NOT EXISTS idx_longitudinal_document ON longitudinal_evidence(document_id)",
    "CREATE INDEX IF NOT EXISTS idx_longitudinal_valid_date ON longitudinal_evidence(valid_start_date)",
    "CREATE INDEX IF NOT EXISTS idx_longitudinal_entity ON longitudinal_evidence(normalized_entity)",
    "CREATE INDEX IF NOT EXISTS idx_checkpoints_run ON clinical_state_checkpoints(run_id)",
    "CREATE INDEX IF NOT EXISTS idx_timeline_patient ON clinical_timeline(patient_id)",
    "CREATE INDEX IF NOT EXISTS idx_timeline_date ON clinical_timeline(date_observed)",
    "CREATE INDEX IF NOT EXISTS idx_timeline_category ON clinical_timeline(category)",
    "CREATE INDEX IF NOT EXISTS idx_timeline_status ON clinical_timeline(status)",
]


def init_database(db: DatabaseEngine) -> None:
    """Create all tables and indexes if they don't exist."""
    from datetime import datetime

    with db:
        for sql in CREATE_TABLES_SQL:
            db.execute(sql)

        for sql in INDEXES_SQL:
            db.execute(sql)

        # ---- Schema migrations (safe ALTER TABLE) -------------------------
        # Add columns that may be missing in databases created by older versions.
        _safe_add_column(db, "patients", "initials", "TEXT")
        # v7: support textual lab results (e.g. "NEGATIVO", "POSITIVO")
        _safe_add_column(db, "lab_values", "value_text", "TEXT")
        # v8: provenance of canonical dedup — entry_ids merged into a survivor
        _safe_add_column(db, "clinical_timeline", "merged_into_ids", "TEXT")

        # Set schema version
        cursor = db.execute("SELECT MAX(version) FROM schema_version")
        current = cursor.fetchone()[0]

        if current is None or current < SCHEMA_VERSION:
            db.execute(
                "INSERT OR REPLACE INTO schema_version (version, applied_at) VALUES (?, ?)",
                (SCHEMA_VERSION, datetime.now().isoformat()),
            )


def _safe_add_column(db: DatabaseEngine, table: str, column: str,
                     col_type: str) -> None:
    """Add a column if it doesn't already exist (SQLite doesn't have
    ``ALTER TABLE … ADD COLUMN IF NOT EXISTS``)."""
    try:
        db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
    except Exception:
        pass  # Column already exists


def drop_all_tables(db: DatabaseEngine) -> None:
    """Drop all tables (for testing/reset). Use with caution."""
    tables = [
        "audit_log", "validation_queue", "clinical_state_checkpoints",
        "longitudinal_evidence", "clinical_state_runs",
        "clinical_state_deltas",
        "clinical_state", "lab_values", "clinical_events",
        "clinical_timeline",
        "document_clinical_projections", "clinical_evidence",
        "document_identity_evidence", "patient_identities",
        "docling_elements", "documents", "patients", "schema_version",
    ]
    with db:
        for table in tables:
            db.execute(f"DROP TABLE IF EXISTS {table}")
