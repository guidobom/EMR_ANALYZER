"""SNOMED CT International RF2 lookup, English synonyms and inferred hierarchy."""
import hashlib
import json
from pathlib import Path
import re
import threading
import unicodedata

from ..database.engine import DatabaseEngine


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def words(text):
    text = unicodedata.normalize('NFKD', text.casefold())
    return re.findall(r'[a-z0-9]+', ''.join(c for c in text if not unicodedata.combining(c)))


class SnomedCatalog:
    def __init__(self, path):
        self.path = Path(path)
        self.db = DatabaseEngine(self.path)
        self._vectors = None
        self._encoder = None
        self._lock = threading.RLock()
        with self.db:
            self.db.execute('CREATE TABLE IF NOT EXISTS sct_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
            self.db.execute('''CREATE TABLE IF NOT EXISTS sct_concepts (
                code TEXT PRIMARY KEY, active INTEGER NOT NULL, fsn TEXT NOT NULL,
                term TEXT NOT NULL, tag TEXT NOT NULL)''')
            self.db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS sct_search USING fts5(code UNINDEXED, term, tokenize='unicode61 remove_diacritics 2')")
            self.db.execute('''CREATE TABLE IF NOT EXISTS sct_aliases (
                text TEXT PRIMARY KEY, code TEXT NOT NULL)''')

    def metadata(self):
        return {r['key']: r['value'] for r in self.db.execute('SELECT * FROM sct_meta')}

    @property
    def digest(self):
        meta = self.metadata()
        aliases = [tuple(r) for r in self.db.execute('SELECT * FROM sct_aliases ORDER BY text')]
        return hashlib.sha256(json.dumps([meta, aliases], sort_keys=True).encode()).hexdigest()

    @property
    def available(self):
        return self.metadata().get('format') == 'RF2 International Snapshot' and bool(self.metadata().get('sha256'))

    def lookup(self, code):
        row = self.db.execute('SELECT * FROM sct_concepts WHERE code=?', (code,)).fetchone()
        return dict(row) if row else None

    def import_file(self, source, progress=None, cancelled=None):
        from .snomed_rf2 import import_snapshot
        return import_snapshot(self, source, progress, cancelled)

    def synonyms(self, code):
        if not self.available:
            return []
        return [r[0] for r in self.db.execute(
            "SELECT DISTINCT term FROM sct_rf2_description WHERE conceptId=? AND active='1' "
            "AND languageCode='en' AND typeId='900000000000013009' ORDER BY term", (str(code),))]

    def parents(self, code):
        if not self.available:
            return []
        return [dict(r) for r in self.db.execute(
            "SELECT DISTINCT c.code,c.fsn,c.term FROM sct_rf2_relationship r "
            "JOIN sct_concepts c ON c.code=r.destinationId AND c.active=1 "
            "WHERE r.sourceId=? AND r.active='1' AND r.typeId='116680003' "
            "AND r.characteristicTypeId='900000000000011006' ORDER BY c.code", (str(code),))]

    def is_descendant_of(self, code, ancestor):
        if not self.available:
            return False
        concept, parent = self.lookup(str(code)), self.lookup(str(ancestor))
        if not concept or not parent or not concept['active'] or not parent['active']:
            return False
        return bool(self.db.execute(
            "WITH RECURSIVE ancestors(id) AS (SELECT destinationId FROM sct_rf2_relationship "
            "WHERE sourceId=? AND active='1' AND typeId='116680003' AND characteristicTypeId='900000000000011006' "
            "UNION SELECT r.destinationId FROM sct_rf2_relationship r JOIN ancestors a ON r.sourceId=a.id "
            "WHERE r.active='1' AND r.typeId='116680003' AND r.characteristicTypeId='900000000000011006') "
            "SELECT 1 FROM ancestors WHERE id=? LIMIT 1", (str(code),str(ancestor))).fetchone())

    def relationships(self, code):
        if not self.available:
            return []
        return [dict(r) for r in self.db.execute(
            "SELECT r.typeId,r.destinationId,r.relationshipGroup,t.term AS type_term,c.term AS destination_term "
            "FROM sct_rf2_relationship r JOIN sct_concepts t ON t.code=r.typeId "
            "JOIN sct_concepts c ON c.code=r.destinationId "
            "WHERE r.sourceId=? AND r.active='1' AND r.characteristicTypeId='900000000000011006' "
            "ORDER BY r.relationshipGroup,r.typeId,r.destinationId", (str(code),))]

    def add_alias(self, text, code):
        concept = self.lookup(code)
        if not concept or not concept['active'] or not text.strip():
            raise ValueError('Seleziona un concetto attivo e una espressione italiana.')
        with self.db:
            self.db.execute('INSERT OR REPLACE INTO sct_aliases VALUES (?,?)', (' '.join(words(text)), code))

    def vector_status(self):
        """State of the multilingual index for the catalogue in use.

        The index carries the fingerprint of the catalogue it was built from:
        importing a new release (or re-importing the same one) makes it stale.
        """
        if not self.available:
            return {'state': 'unavailable', 'model': None, 'reason': 'nessun catalogo importato'}
        meta = self.metadata()
        model = meta.get('embedding_model')
        if not model:
            return {'state': 'missing', 'model': None,
                    'reason': 'indice vettoriale non creato'}
        index = self.path.with_suffix('.vectors.npz')
        if not Path(index).exists():
            return {'state': 'stale', 'model': model, 'reason': 'file dell’indice assente'}
        try:
            import numpy as np
            with np.load(index, allow_pickle=False) as data:
                fingerprint = str(data['fingerprint'])
        except Exception:
            return {'state': 'stale', 'model': model, 'reason': 'indice illeggibile'}
        if fingerprint != meta.get('sha256'):
            return {'state': 'stale', 'model': model,
                    'reason': 'il catalogo è cambiato: l’indice va ricostruito'}
        return {'state': 'ready', 'model': model, 'reason': ''}

    def _ready_vectors(self, meta):
        """Vectors for the current catalogue revision, or (None, None) if unusable.

        A stale index degrades to text search instead of stopping the coding
        run: the next build restores it.
        """
        if not meta.get('embedding_model'):
            return None, None
        revision = (meta.get('sha256'), meta.get('embedding_revision'))
        with self._lock:
            if getattr(self, '_loaded_revision', None) != revision:
                self._vectors = self._encoder = None
            if getattr(self, '_stale_revision', None) == revision:
                return None, None
            if self._vectors is None:
                import numpy as np
                from sentence_transformers import SentenceTransformer
                try:
                    with np.load(self.path.with_suffix('.vectors.npz'), allow_pickle=False) as data:
                        if str(data['fingerprint']) != meta.get('sha256'):
                            self._stale_revision = revision
                            return None, None
                        self._vectors = (data['codes'].copy(), data['vectors'].astype('float32'))
                except (OSError, ValueError, KeyError):
                    self._stale_revision = revision
                    return None, None
                self._encoder = SentenceTransformer(meta['embedding_model'], local_files_only=True)
                self._loaded_revision = revision
            return self._vectors, self._encoder

    def build_vectors(self, model_path, progress=None, cancelled=None, *, tags=None, device=None,
                      batch_size=256):
        """Multilingual vector index of the concepts used to code clinical events.

        One vector per active concept, from its preferred term (the semantic
        tag of the FSN adds no meaning to an entity name), restricted by
        default to the hierarchies admissible for extracted events. Vectors
        are stored as float16 next to the catalog.
        """
        import numpy as np
        from sentence_transformers import SentenceTransformer
        if not self.available:
            raise ValueError('Importa prima la International Edition RF2.')
        model_path = str(Path(model_path).resolve())
        if not Path(model_path).is_dir():
            raise ValueError('Seleziona una cartella locale con un modello SentenceTransformer multilingue.')
        if tags is None:
            from .snomed_coding import FACT_TYPE_TAGS
            tags = sorted({tag for values in FACT_TYPE_TAGS.values() for tag in values})
        encoder = SentenceTransformer(model_path, local_files_only=True, device=device)
        fingerprint = self.metadata()['sha256']
        marks = ','.join('?' for _ in tags)
        rows = self.db.execute(
            f'SELECT code,term,fsn FROM sct_concepts WHERE active=1 AND tag IN ({marks}) ORDER BY code',
            tuple(tags)).fetchall()
        codes, parts = [], []
        for start in range(0, len(rows), batch_size):
            if cancelled and cancelled():
                raise InterruptedError('Indicizzazione annullata.')
            batch = rows[start:start+batch_size]
            names = [r['term'] or re.sub(r'\s*\([^()]*\)$', '', r['fsn']) for r in batch]
            parts.append(encoder.encode(names, normalize_embeddings=True, show_progress_bar=False,
                                        batch_size=batch_size).astype('float16'))
            codes.extend(r['code'] for r in batch)
            if progress:
                progress(f'Indicizzati {len(codes):,}/{len(rows):,} concetti…')
        vectors = np.concatenate(parts)
        index = self.path.with_suffix('.vectors.npz')
        temporary = index.with_suffix('.tmp.npz')
        np.savez(temporary, codes=np.array(codes), vectors=vectors, fingerprint=np.array(fingerprint))
        with self._lock, self.db:
            if self.metadata()['sha256'] != fingerprint:
                temporary.unlink(missing_ok=True)
                raise ValueError('Catalogo cambiato durante indicizzazione: ripetere.')
            temporary.replace(index)
            self.db.execute('INSERT OR REPLACE INTO sct_meta VALUES (?,?)', ('embedding_model', model_path))
            self.db.execute('INSERT OR REPLACE INTO sct_meta VALUES (?,?)', ('embedding_revision', file_digest(index)))
            self._vectors = (np.array(codes), vectors.astype('float32'))
            self._encoder = encoder
            self._loaded_revision = (fingerprint, self.metadata()['embedding_revision'])
        return self.metadata()

    def search(self, text, english='', limit=12, tags=None):
        """RRF merge of lexical and optional multilingual vector candidates.

        ``tags`` restricts candidates to SNOMED hierarchies (semantic tags such
        as ``disorder`` or ``substance``) inside every query, so concepts of
        other hierarchies cannot crowd out the relevant ones.
        """
        if not self.available or limit <= 0:
            return []
        tags = tuple(sorted(tags)) if tags else ()
        tag_sql = f" AND c.tag IN ({','.join('?' for _ in tags)})" if tags else ""
        allowed = lambda concept: bool(concept and concept['active'] and (not tags or concept['tag'] in tags))
        rankings = []
        exact_codes = set()
        alias = self.db.execute('SELECT code FROM sct_aliases WHERE text=?', (' '.join(words(text)),)).fetchone()
        if alias and allowed(self.lookup(alias[0])):
            rankings.append([alias[0]])
        else:
            alias = None
        for query in (text, english):
            if query.strip():
                exact = [r[0] for r in self.db.execute(
                    "SELECT DISTINCT d.conceptId FROM sct_rf2_description d JOIN sct_concepts c ON c.code=d.conceptId "
                    "WHERE d.term=? COLLATE NOCASE AND d.active='1' AND c.active=1 AND d.languageCode='en' "
                    "AND EXISTS (SELECT 1 FROM sct_rf2_language l WHERE l.referencedComponentId=d.id "
                    "AND l.active='1' AND l.refsetId IN ('900000000000509007','900000000000508004') "
                    "AND l.acceptabilityId IN ('900000000000548007','900000000000549004'))" + tag_sql + " LIMIT 40",
                    (query.strip(), *tags))]
                exact_codes.update(exact)
                rankings.append(exact)
            tokens = list(dict.fromkeys(words(query)))[:24]
            if tokens:
                expressions = ['"'+' '.join(tokens)+'"', ' AND '.join('"'+t+'"' for t in tokens),
                               ' OR '.join('"'+t+'"' for t in tokens)]
                for expression in dict.fromkeys(expressions):
                    found = [r[0] for r in self.db.execute(
                        'SELECT sct_search.code FROM sct_search JOIN sct_concepts c ON c.code=sct_search.code '
                        'WHERE sct_search MATCH ? AND c.active=1' + tag_sql
                        + ' ORDER BY bm25(sct_search) LIMIT 200', (expression, *tags))]
                    rankings.append(list(dict.fromkeys(found))[:40])
        vectors, encoder = self._ready_vectors(self.metadata())
        if encoder is not None:
            import numpy as np
            codes, matrix = vectors
            vector = encoder.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
            scores = matrix @ vector
            count = min(200 if tags else 40, len(scores))
            best = np.argpartition(-scores, count-1)[:count]
            ranked = [str(codes[i]) for i in sorted(best, key=lambda i: -scores[i])]
            if tags:
                ranked = [code for code in ranked if allowed(self.lookup(code))]
            rankings.append(ranked[:40])
        scores = {}
        for ranking in rankings:
            for rank, code in enumerate(ranking):
                scores[code] = scores.get(code, 0) + 1/(60+rank)
        if alias and alias[0] in scores:
            scores[alias[0]] += 1
        for code in exact_codes:
            scores[code] += .15
        return [{**self.lookup(code), 'parents': self.parents(code)[:4]}
                for code in sorted(scores, key=lambda c: (-scores[c], c))[:limit]]
