"""Bounded example retrieval. Examples guide extraction but never constitute evidence."""
from functools import lru_cache
import hashlib
import json
import re
import threading
import unicodedata


def words(text):
    text = unicodedata.normalize('NFKD',text.casefold())
    text = ''.join(c for c in text if not unicodedata.combining(c))
    return set(re.findall(r'[a-z0-9]{3,}',text)) - {'della','delle','con','per','del','nel','alla','una','non','dei','che'}


class LexiconGuidance:
    def __init__(self, rows, encoder=None):
        self.rows = sorted(rows,key=lambda r:r['id'])
        self.encoder = encoder
        self.lock = threading.Lock()
        self.texts = [' '.join((r['label'],r['definition'],r['quote'],*r['fields'])) for r in self.rows]
        self.tokens = [words(t) for t in self.texts]
        self.vectors = None
        self.digest = hashlib.sha256(json.dumps(self.rows,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        if encoder is not None and self.rows:
            try:
                self.vectors = encoder.encode(self.texts,normalize_embeddings=True,show_progress_bar=False)
            except Exception:
                self.encoder = None
        self.digest += ':semantic' if self.vectors is not None else ':lexical'

    def retrieve(self, text, limit=4):
        tokens = words(text)
        scores = [len(tokens & t)/max(1,len(t)) for t in self.tokens]
        if self.vectors is not None:
            import numpy as np
            with self.lock:
                vector = self.encoder.encode([text],normalize_embeddings=True,show_progress_bar=False)[0]
            similarities = np.asarray(self.vectors) @ vector
            scores = [max(score,float(sim) if sim>=0.45 else 0) for score,sim in zip(scores,similarities)]
        ranked = [i for i in sorted(range(len(scores)),key=lambda i:(-scores[i],self.rows[i]['id'])) if scores[i]>0]
        picked = ranked[:limit]
        # Include a contrasting example for retrieved concepts when available.
        for i in list(picked):
            opposite = next((j for j,r in enumerate(self.rows) if r['term_id']==self.rows[i]['term_id'] and r['role']!=self.rows[i]['role']),None)
            if opposite is not None and opposite not in picked:
                picked.append(opposite)
        return [self.rows[i] for i in picked[:limit+2]]

    def prompt(self,text,max_chars=5000):
        rows=[]
        for r in self.retrieve(text):
            item={k:r[k] for k in ('id','label','role','assertion','definition','fields','quote')}
            encoded=json.dumps(rows+[item],ensure_ascii=False)
            if len(encoded)<=max_chars:
                rows.append(item)
        if not rows:
            return '',[]
        return ('\nESEMPI DIDATTICI — NON SONO FONTI DEL DOCUMENTO CORRENTE:\n'+json.dumps(rows,ensure_ascii=False)+
                '\nUsa le schede per riconoscere eventi analoghi. Non trasferire farmaci, date, persone o altri fatti dagli esempi. '
                'Un controesempio non attesta quel tipo di evento, ma può descriverne altri. Estrai anche eventi senza esempi. '
                'Riporta solo attributi espliciti nella fonte corrente; ometti quelli mancanti. Campi personalizzati in attributes.details.\n',
                [r['id'] for r in rows])


@lru_cache(maxsize=1)
def local_encoder():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2',local_files_only=True)
    except Exception:
        return None


def build_guidance(repo):
    if repo is None:
        return LexiconGuidance([])
    rows=repo.extraction_examples()
    return LexiconGuidance(rows,local_encoder() if rows else None)
