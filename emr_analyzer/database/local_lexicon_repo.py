"""Workspace-local, human-authored examples; never automatic clinical evidence."""
from contextlib import contextmanager
import hashlib
import json
import unicodedata
import uuid

CONTEXTS = ('Non specificato', 'Presente', 'Negato', 'Possibile', 'Familiare', 'Altro', 'Assente', 'Non determinato')
SCHEMA = [
    '''CREATE TABLE IF NOT EXISTS local_lexicon_terms (
       id TEXT PRIMARY KEY, label TEXT NOT NULL, label_key TEXT NOT NULL UNIQUE)''',
    '''CREATE TABLE IF NOT EXISTS local_lexicon_annotations (
       id TEXT PRIMARY KEY,
       document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
       term_id TEXT NOT NULL REFERENCES local_lexicon_terms(id),
       source_hash TEXT NOT NULL, start INTEGER NOT NULL, end INTEGER NOT NULL,
       quote TEXT NOT NULL, context TEXT NOT NULL, assertion TEXT NOT NULL,
       CHECK(start >= 0 AND end > start),
       UNIQUE(document_id,source_hash,start,end,term_id))''',
    '''CREATE TABLE IF NOT EXISTS local_event_definitions (
       term_id TEXT PRIMARY KEY REFERENCES local_lexicon_terms(id) ON DELETE CASCADE,
       definition TEXT NOT NULL, fields_json TEXT NOT NULL)''',
    '''CREATE TABLE IF NOT EXISTS local_example_roles (
       annotation_id TEXT PRIMARY KEY REFERENCES local_lexicon_annotations(id) ON DELETE CASCADE,
       role TEXT NOT NULL CHECK(role IN ('example','counterexample')))''',
    '''CREATE TABLE IF NOT EXISTS local_lexicon_examples (
       id TEXT PRIMARY KEY,
       term_id TEXT NOT NULL REFERENCES local_lexicon_terms(id) ON DELETE CASCADE,
       quote TEXT NOT NULL, assertion TEXT NOT NULL,
       role TEXT NOT NULL CHECK(role IN ('example','counterexample')))''',
    'CREATE INDEX IF NOT EXISTS local_annotations_document ON local_lexicon_annotations(document_id)',
]


def source_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def label_key(text):
    return unicodedata.normalize('NFKC', ' '.join(text.split())).casefold()


class LocalLexiconRepository:
    def __init__(self, db):
        self.db = db
        self._undo = []

    def terms(self):
        return [dict(r) for r in self.db.execute('''SELECT t.*, COUNT(a.id) + (SELECT COUNT(*) FROM local_lexicon_examples e WHERE e.term_id=t.id) AS examples
            FROM local_lexicon_terms t LEFT JOIN local_lexicon_annotations a ON a.term_id=t.id
            GROUP BY t.id ORDER BY t.label_key''')]

    def annotations(self, document_id=None, term_id=None):
        query = '''SELECT a.*, t.label, d.patient_id, d.filename, d.document_date
                   FROM local_lexicon_annotations a JOIN local_lexicon_terms t ON t.id=a.term_id
                   JOIN documents d ON d.id=a.document_id WHERE 1=1'''
        args = []
        for name, value in [('document_id', document_id), ('term_id', term_id)]:
            if value is not None:
                query += f' AND a.{name}=?'
                args.append(value)
        return [dict(r) for r in self.db.execute(query+' ORDER BY d.patient_id,d.document_date,a.start', tuple(args))]

    def _snapshot(self):
        return {table: [dict(r) for r in self.db.execute(f'SELECT * FROM {table} ORDER BY 1')]
                for table in ('local_lexicon_terms', 'local_lexicon_annotations', 'local_event_definitions', 'local_example_roles', 'local_lexicon_examples')}

    @contextmanager
    def _change(self):
        with self.db:
            before = self._snapshot()
            yield
            after = self._snapshot()
        if before != after:
            self._undo.append((before, after))
            self._undo = self._undo[-20:]

    def undo(self):
        if not self._undo:
            return False
        before, after = self._undo[-1]
        with self.db:
            if self._snapshot() != after:
                raise ValueError('Il lessico è cambiato altrove: annullamento non applicabile.')
            self.db.execute('DELETE FROM local_lexicon_examples')
            self.db.execute('DELETE FROM local_example_roles')
            self.db.execute('DELETE FROM local_event_definitions')
            self.db.execute('DELETE FROM local_lexicon_annotations')
            self.db.execute('DELETE FROM local_lexicon_terms')
            for table, rows in before.items():
                for row in rows:
                    self.db.execute(f"INSERT INTO {table} ({','.join(row)}) VALUES ({','.join('?' for _ in row)})", tuple(row.values()))
        self._undo.pop()
        return True

    def save(self, document_id, text, start, end, label, assertion='Non specificato', annotation_id=None):
        label = ' '.join(label.split())
        if not label or len(label) > 200:
            raise ValueError('Inserisci un termine di 1–200 caratteri.')
        if not 0 <= start < end <= len(text) or not text[start:end].strip():
            raise ValueError('Seleziona un frammento significativo del testo.')
        if assertion not in CONTEXTS:
            raise ValueError('Contesto non valido.')
        with self._change():
            if annotation_id:
                old = self.db.execute('SELECT document_id FROM local_lexicon_annotations WHERE id=?', (annotation_id,)).fetchone()
                if old is None or old[0] != document_id:
                    raise ValueError('Annotazione non disponibile per questo documento.')
            key = label_key(label)
            self.db.execute('INSERT OR IGNORE INTO local_lexicon_terms VALUES (?,?,?)', (uuid.uuid4().hex, label, key))
            term_id = self.db.execute('SELECT id FROM local_lexicon_terms WHERE label_key=?', (key,)).fetchone()[0]
            digest = source_hash(text)
            duplicate = self.db.execute('''SELECT id FROM local_lexicon_annotations
                WHERE document_id=? AND source_hash=? AND start=? AND end=? AND term_id=?''',
                (document_id,digest,start,end,term_id)).fetchone()
            if duplicate and duplicate[0] != annotation_id:
                raise ValueError('Questo frammento è già associato al termine.')
            ident = annotation_id or uuid.uuid4().hex
            self.db.execute('''INSERT INTO local_lexicon_annotations VALUES (?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET term_id=excluded.term_id,source_hash=excluded.source_hash,
                start=excluded.start,end=excluded.end,quote=excluded.quote,context=excluded.context,assertion=excluded.assertion''',
                (ident,document_id,term_id,digest,start,end,text[start:end],text[max(0,start-180):min(len(text),end+180)],assertion))
        return ident

    def delete(self, annotation_id):
        with self._change():
            self.db.execute('DELETE FROM local_lexicon_annotations WHERE id=?', (annotation_id,))
            self.db.execute('DELETE FROM local_lexicon_examples WHERE id=?', (annotation_id,))

    def rename(self, term_id, label):
        label = ' '.join(label.split())
        if not label or len(label)>200:
            raise ValueError('Inserisci un termine di 1–200 caratteri.')
        with self._change():
            found = self.db.execute('SELECT id FROM local_lexicon_terms WHERE label_key=?', (label_key(label),)).fetchone()
            if found and found[0] != term_id:
                raise ValueError('Il termine esiste già. Usa Unisci termini.')
            self.db.execute('UPDATE local_lexicon_terms SET label=?,label_key=? WHERE id=?', (label,label_key(label),term_id))

    def merge(self, source_id, target_id):
        if source_id == target_id:
            return
        with self._change():
            if not self.db.execute('SELECT 1 FROM local_lexicon_terms WHERE id=?', (target_id,)).fetchone():
                raise ValueError('Termine di destinazione mancante.')
            # Keep conflicting annotations rather than silently lose context.
            overlap = self.db.execute('''SELECT 1 FROM local_lexicon_annotations a JOIN local_lexicon_annotations b
                ON a.document_id=b.document_id AND a.source_hash=b.source_hash AND a.start=b.start AND a.end=b.end
                WHERE a.term_id=? AND b.term_id=?''', (source_id,target_id)).fetchone()
            if overlap:
                raise ValueError('Alcuni frammenti hanno entrambi i termini: risolvi le annotazioni duplicate prima di unirli.')
            source_card = self.event_definition(source_id)
            target_card = self.event_definition(target_id)
            if any(source_card.values()) and any(target_card.values()) and source_card != target_card:
                raise ValueError('Le schede evento differiscono: uniformale prima di unire i termini.')
            if any(source_card.values()) and not any(target_card.values()):
                self.db.execute('INSERT OR REPLACE INTO local_event_definitions VALUES (?,?,?)',
                                (target_id,source_card['definition'],json.dumps({k:v for k,v in source_card.items() if k != 'definition'})))
            self.db.execute('UPDATE local_lexicon_annotations SET term_id=? WHERE term_id=?', (target_id,source_id))
            self.db.execute('UPDATE local_lexicon_examples SET term_id=? WHERE term_id=?', (target_id,source_id))
            self.db.execute('DELETE FROM local_lexicon_terms WHERE id=?', (source_id,))

    def event_definition(self, term_id):
        row = self.db.execute('SELECT definition,fields_json FROM local_event_definitions WHERE term_id=?',(term_id,)).fetchone()
        if not row:
            return {'definition':'', 'fields':[]}
        stored = json.loads(row[1])
        return {'definition':row[0], **(stored if isinstance(stored, dict) else {'fields':stored})}

    def set_event_definition(self, term_id, definition, fields, *, structure=None):
        fields = list(dict.fromkeys(f.strip() for f in fields if f.strip()))
        if len(definition)>2000 or len(fields)>20 or any(len(f)>100 for f in fields):
            raise ValueError('Massimo 2000 caratteri di definizione e 20 campi di 100 caratteri.')
        payload = fields
        if structure is not None:
            from ..clinical.lexicon_structure import normalize_structure
            payload = dict(fields=fields, **normalize_structure(structure))
            payload['fields'] = [f['name'] for f in payload['field_definitions']]
        with self._change():
            self.db.execute('INSERT INTO local_event_definitions VALUES (?,?,?) ON CONFLICT(term_id) DO UPDATE SET definition=excluded.definition,fields_json=excluded.fields_json',
                            (term_id,definition.strip(),json.dumps(payload,ensure_ascii=False)))

    def example_role(self, annotation_id):
        manual = self.db.execute('SELECT role FROM local_lexicon_examples WHERE id=?', (annotation_id,)).fetchone()
        if manual:
            return manual[0]
        row = self.db.execute('SELECT role FROM local_example_roles WHERE annotation_id=?',(annotation_id,)).fetchone()
        return row[0] if row else 'example'

    def set_example_role(self, annotation_id, role):
        if role not in {'example','counterexample'}:
            raise ValueError('Tipo di esempio non valido.')
        with self._change():
            if self.db.execute('SELECT 1 FROM local_lexicon_examples WHERE id=?', (annotation_id,)).fetchone():
                self.db.execute('UPDATE local_lexicon_examples SET role=? WHERE id=?', (role,annotation_id))
                return
            self.db.execute('INSERT INTO local_example_roles VALUES (?,?) ON CONFLICT(annotation_id) DO UPDATE SET role=excluded.role', (annotation_id,role))

    def extraction_examples(self):
        return [dict(id=r['id'], term_id=r['term_id'], label=r['label'],
                     quote=r['quote'], context=r['context'], assertion=r['assertion'],
                     role=self.example_role(r['id']), **self.event_definition(r['term_id']))
                for r in self.examples()]


    def examples(self, term_id=None):
        """Standalone teaching text, whether selected from a document or typed."""
        selected = self.annotations(term_id=term_id)
        query = "SELECT e.*, t.label FROM local_lexicon_examples e JOIN local_lexicon_terms t ON t.id=e.term_id"
        args = ()
        if term_id is not None:
            query += " WHERE e.term_id=?"
            args = (term_id,)
        manual = [dict(row, context=row['quote'], manual=True) for row in
                  self.db.execute(query + ' ORDER BY e.id', args)]
        return selected + manual

    def save_example(self, term_id, text, assertion='Non specificato', role='example', example_id=None):
        """No document, patient or fabricated source record is needed."""
        if not text.strip():
            raise ValueError('Scrivi una frase di esempio.')
        if assertion not in CONTEXTS or role not in {'example', 'counterexample'}:
            raise ValueError('Stato o tipo di esempio non valido.')
        with self._change():
            if not self.db.execute('SELECT 1 FROM local_lexicon_terms WHERE id=?', (term_id,)).fetchone():
                raise ValueError('Termine non disponibile.')
            if example_id:
                old = self.db.execute('SELECT term_id FROM local_lexicon_examples WHERE id=?', (example_id,)).fetchone()
                if old is None or old[0] != term_id:
                    raise ValueError('Esempio non disponibile per questo termine.')
            duplicate = self.db.execute('SELECT id FROM local_lexicon_examples WHERE term_id=? AND quote=? AND assertion=? AND role=?',
                                        (term_id,text,assertion,role)).fetchone()
            if duplicate and duplicate[0] != example_id:
                raise ValueError('Questa frase è già presente fra gli esempi del termine.')
            ident = example_id or uuid.uuid4().hex
            self.db.execute("INSERT INTO local_lexicon_examples VALUES (?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET quote=excluded.quote,assertion=excluded.assertion,role=excluded.role",
                            (ident,term_id,text,assertion,role))
        return ident
