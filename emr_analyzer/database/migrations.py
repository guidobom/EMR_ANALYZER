"""Database schema creation and migration for EMR Analyzer."""

from .engine import DatabaseEngine


SCHEMA_VERSION = 10

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
    "CREATE INDEX IF NOT EXISTS idx_hpid_patient ON patient_hospital_ids(patient_id)",
    "CREATE INDEX IF NOT EXISTS idx_hpid_key ON patient_hospital_ids(hospital_patient_id_key)",
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
        # v9: user-confirmed timeline entries for the golden validation set
        _safe_add_column(db, "clinical_timeline", "is_golden", "INTEGER DEFAULT 0")

        # v10: a person may hold several hospital patient IDs (one per unit).
        # Fold any legacy single-column value into the multi-valued table so
        # both storage forms are consistent; idempotent (UNIQUE + OR IGNORE).
        db.execute(
            """INSERT OR IGNORE INTO patient_hospital_ids
               (patient_id, hospital_patient_id_key, created_at, updated_at)
               SELECT patient_id, hospital_patient_id_key, created_at, updated_at
               FROM patient_identities WHERE hospital_patient_id_key IS NOT NULL"""
        )

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
        "audit_log", "validation_queue",
        "clinical_state", "lab_values", "clinical_timeline",
        "clinical_evidence", "document_identity_evidence",
        "patient_hospital_ids", "patient_identities",
        "documents", "patients", "schema_version",
    ]
    with db:
        for table in tables:
            db.execute(f"DROP TABLE IF EXISTS {table}")
