"""One per-user terminology/example registry, independent of project databases."""
from pathlib import Path
import uuid

from .engine import DatabaseEngine
from .local_lexicon_repo import LocalLexiconRepository, SCHEMA, label_key, source_hash


class SharedLexiconRepository(LocalLexiconRepository):
    def __init__(self, workspace_db, workspace_path, registry_path):
        self.workspace_db = workspace_db
        self.workspace_path = str(Path(workspace_path).resolve())
        self.registry_path = Path(registry_path)
        # Identity lives with the project and survives a directory move.
        with workspace_db:
            workspace_db.execute('CREATE TABLE IF NOT EXISTS local_lexicon_identity (id TEXT PRIMARY KEY)')
            row = workspace_db.execute('SELECT id FROM local_lexicon_identity LIMIT 1').fetchone()
            self.workspace_id = row[0] if row else uuid.uuid4().hex
            if row is None:
                workspace_db.execute('INSERT INTO local_lexicon_identity VALUES (?)', (self.workspace_id,))
        super().__init__(DatabaseEngine(self.registry_path))
        with self.db:
            self.db.execute('''CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY, workspace_id TEXT NOT NULL, workspace_path TEXT NOT NULL,
                original_document_id TEXT NOT NULL, patient_id TEXT NOT NULL,
                filename TEXT NOT NULL, document_date TEXT,
                UNIQUE(workspace_id,original_document_id))''')
            for sql in SCHEMA:
                self.db.execute(sql)
            self.db.execute('''CREATE TABLE IF NOT EXISTS annotated_source_texts (
                document_id TEXT NOT NULL REFERENCES documents(id), source_hash TEXT NOT NULL,
                text TEXT NOT NULL, PRIMARY KEY(document_id,source_hash))''')
            self.db.execute('''CREATE TABLE IF NOT EXISTS imported_local_annotations (
                workspace_id TEXT NOT NULL, annotation_id TEXT NOT NULL,
                PRIMARY KEY(workspace_id,annotation_id))''')
            self.db.execute('UPDATE documents SET workspace_path=? WHERE workspace_id=?',
                            (self.workspace_path,self.workspace_id))
            self._import_existing()
            self._backfill_source_texts()
        self._undo.clear()

    @property
    def snomed_catalog(self):
        if not hasattr(self, '_snomed_catalog'):
            from ..clinical.snomed_catalog import SnomedCatalog
            self._snomed_catalog = SnomedCatalog(self.registry_path.with_name('snomed_ct.db'))
        return self._snomed_catalog

    @property
    def loinc_catalog(self):
        if not hasattr(self, '_loinc_catalog'):
            from ..clinical.loinc_catalog import LoincCatalog
            self._loinc_catalog = LoincCatalog(self.registry_path.with_name('loinc.db'))
        return self._loinc_catalog

    def _document_key(self, document_id):
        return self.workspace_id + ':' + document_id

    def _register_document(self, document_id):
        row = self.workspace_db.execute('SELECT patient_id,filename,document_date FROM documents WHERE id=?', (document_id,)).fetchone()
        if row is None:
            raise ValueError('Documento sorgente non disponibile nel progetto corrente.')
        key = self._document_key(document_id)
        self.db.execute('''INSERT INTO documents VALUES (?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET
            workspace_path=excluded.workspace_path, patient_id=excluded.patient_id,
            filename=excluded.filename,document_date=excluded.document_date''',
            (key,self.workspace_id,self.workspace_path,document_id,*tuple(row)))
        return key

    def _import_existing(self):
        # Import once per annotation; do not resurrect examples deleted centrally.
        local = LocalLexiconRepository(self.workspace_db)
        for term in local.terms():
            receipt = 'term:' + term['id']
            if self.db.execute('SELECT 1 FROM imported_local_annotations WHERE workspace_id=? AND annotation_id=?',
                               (self.workspace_id,receipt)).fetchone():
                continue
            self.db.execute('INSERT OR IGNORE INTO local_lexicon_terms VALUES (?,?,?)',
                            (uuid.uuid4().hex,term['label'],term['label_key']))
            self.db.execute('INSERT INTO imported_local_annotations VALUES (?,?)',(self.workspace_id,receipt))
        for row in local.annotations():
            if self.db.execute('SELECT 1 FROM imported_local_annotations WHERE workspace_id=? AND annotation_id=?',
                               (self.workspace_id,row['id'])).fetchone():
                continue
            key = self._register_document(row['document_id'])
            normalized = label_key(row['label'])
            self.db.execute('INSERT OR IGNORE INTO local_lexicon_terms VALUES (?,?,?)',
                            (uuid.uuid4().hex,row['label'],normalized))
            term_id = self.db.execute('SELECT id FROM local_lexicon_terms WHERE label_key=?',(normalized,)).fetchone()[0]
            self.db.execute('''INSERT OR IGNORE INTO local_lexicon_annotations VALUES (?,?,?,?,?,?,?,?,?)''',
                            (uuid.uuid4().hex,key,term_id,row['source_hash'],row['start'],row['end'],row['quote'],row['context'],row['assertion']))
            self.db.execute('INSERT INTO imported_local_annotations VALUES (?,?)',(self.workspace_id,row['id']))

        for row in local.examples():
            if not row.get('manual'):
                continue
            if self.db.execute('SELECT 1 FROM imported_local_annotations WHERE workspace_id=? AND annotation_id=?',
                               (self.workspace_id,row['id'])).fetchone():
                continue
            normalized = label_key(row['label'])
            self.db.execute('INSERT OR IGNORE INTO local_lexicon_terms VALUES (?,?,?)',
                            (uuid.uuid4().hex,row['label'],normalized))
            term_id = self.db.execute('SELECT id FROM local_lexicon_terms WHERE label_key=?', (normalized,)).fetchone()[0]
            self.db.execute('INSERT INTO local_lexicon_examples VALUES (?,?,?,?,?)',
                            (uuid.uuid4().hex,term_id,row['quote'],row['assertion'],row['role']))
            self.db.execute('INSERT INTO imported_local_annotations VALUES (?,?)', (self.workspace_id,row['id']))

    def annotations(self, document_id=None, term_id=None):
        rows = super().annotations(self._document_key(document_id) if document_id is not None else None, term_id)
        sources = {r['id']: dict(r) for r in self.db.execute('SELECT * FROM documents')}
        for row in rows:
            source = sources[row['document_id']]
            row['document_id'] = source['original_document_id']
            row['workspace_id'] = source['workspace_id']
            row['workspace_path'] = source['workspace_path']
            row['workspace_name'] = Path(source['workspace_path']).name
            row['current_workspace'] = source['workspace_id'] == self.workspace_id
        return rows

    def save(self, document_id, text, start, end, label, assertion='Non specificato', annotation_id=None):
        with self.db:
            key = self._register_document(document_id)
            ident = super().save(key,text,start,end,label,assertion,annotation_id)
            self.db.execute('INSERT OR IGNORE INTO annotated_source_texts VALUES (?,?,?)',
                            (key,source_hash(text),text))
            return ident

    def annotated_text(self, annotation_id):
        """Return the immutable full text, if available; never read a live project."""
        row = self.db.execute('''SELECT s.text FROM annotated_source_texts s
            JOIN local_lexicon_annotations a ON a.document_id=s.document_id AND a.source_hash=s.source_hash
            WHERE a.id=?''', (annotation_id,)).fetchone()
        return row[0] if row else None

    def _backfill_source_texts(self):
        # Earlier annotations stored excerpts only. Do not attach a newer document
        # to old coordinates merely because its document identifier is the same.
        from .overlay_repo import DocumentTextOverlayRepository
        rows = self.db.execute('''SELECT DISTINCT a.document_id,a.source_hash,d.original_document_id
            FROM local_lexicon_annotations a JOIN documents d ON d.id=a.document_id
            LEFT JOIN annotated_source_texts s ON s.document_id=a.document_id AND s.source_hash=a.source_hash
            WHERE d.workspace_id=? AND s.document_id IS NULL''', (self.workspace_id,)).fetchall()
        for row in rows:
            doc = self.workspace_db.execute('SELECT patient_id FROM documents WHERE id=?',
                                            (row['original_document_id'],)).fetchone()
            if doc is None:
                continue
            for folder in ('extraction','docling'):
                path = Path(self.workspace_path)/doc[0]/folder/(row['original_document_id']+'.md')
                if not path.is_file():
                    continue
                try:
                    text = path.read_text(encoding='utf-8')
                    text = DocumentTextOverlayRepository(self.workspace_db).effective_text(row['original_document_id'],text)
                    text = text.replace('\r\n','\n').replace('\r','\n').replace('\u2029','\n').replace('\u2028','\n')
                    if source_hash(text) == row['source_hash']:
                        self.db.execute('INSERT OR IGNORE INTO annotated_source_texts VALUES (?,?,?)',
                                        (row['document_id'],row['source_hash'],text))
                except (OSError,UnicodeError):
                    pass  # Excerpt and surrounding context remain available.
                break

    def close(self):
        if hasattr(self, "_loinc_catalog"):
            self._loinc_catalog.db.close()
        if hasattr(self, '_snomed_catalog'):
            self._snomed_catalog.db.close()
        self.db.close()
