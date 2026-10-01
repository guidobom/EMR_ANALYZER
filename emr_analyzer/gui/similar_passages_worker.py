"""Background search limited to the captured current-workspace document list."""
from PyQt5.QtCore import QThread, pyqtSignal
from ..clinical.lexicon_guidance import local_encoder
from ..clinical.similar_passages import passage_spans, score_passages
from ..database.local_lexicon_repo import source_hash


class SimilarPassagesWorker(QThread):
    completed = pyqtSignal(list, str, int)
    failed = pyqtSignal(str)
    progress = pyqtSignal(int,int)

    def __init__(self, documents, query, label, known, threshold, overlay=None, parent=None):
        super().__init__(parent)
        self.documents,self.query,self.label=documents,query,label
        self.known,self.threshold,self.overlay=known,threshold,overlay

    def run(self):
        try:
            encoder=local_encoder()
            mode='semantica' if encoder is not None else 'testuale approssimata (modello semantico locale non disponibile)'
            if self.isInterruptionRequested():
                return
            query_vector=encoder.encode([self.query],normalize_embeddings=True,show_progress_bar=False)[0] if encoder is not None else None
            best=[]
            unreadable=0
            for number,(doc,path) in enumerate(self.documents,1):
                if self.isInterruptionRequested():
                    return
                if path is None:
                    unreadable+=1
                    continue
                try:
                    text=path.read_text(encoding='utf-8')
                    if self.overlay is not None:
                        text=self.overlay.effective_text(doc.id,text)
                    text=text.replace('\r\n','\n').replace('\r','\n').replace('\u2029','\n').replace('\u2028','\n')
                except (OSError,UnicodeError):
                    unreadable+=1
                    continue
                digest=source_hash(text)
                spans=iter(passage_spans(text))
                while True:
                    if self.isInterruptionRequested():
                        return
                    batch=[]
                    for _ in range(32):
                        span=next(spans,None)
                        if span is None:
                            break
                        start,end,quote=span
                        if any(d==doc.id and h==digest and start<b and end>a for d,h,a,b in self.known):
                            continue
                        batch.append(span)
                    if not batch:
                        if span is None:
                            break
                        continue
                    scores=score_passages(self.query,[s[2] for s in batch],encoder,query_vector)
                    for (start,end,quote),score in zip(batch,scores):
                        if score>=self.threshold:
                            best.append(dict(document_id=doc.id,patient_id=doc.patient_id,filename=doc.filename,
                                source_hash=digest,start=start,end=end,quote=quote,label=self.label,
                                score=score,context=text[max(0,start-100):end+100]))
                    best=sorted(best,key=lambda r:(-r['score'],r['document_id'],r['start']))[:100]
                    if span is None:
                        break
                self.progress.emit(number,len(self.documents))
            # Collapse overlapping windows from the same source; keep best-ranked.
            results=[]
            for row in best:
                if not any(r['document_id']==row['document_id'] and row['start']<r['end'] and row['end']>r['start'] for r in results):
                    results.append(row)
            if not self.isInterruptionRequested():
                self.completed.emit(results,mode,unreadable)
        except Exception as exc:
            if not self.isInterruptionRequested():
                self.failed.emit(f'Ricerca non completata: {exc}')
        finally:
            if self.overlay is not None:
                self.overlay.db.close()  # Only this worker's thread-local connection.
