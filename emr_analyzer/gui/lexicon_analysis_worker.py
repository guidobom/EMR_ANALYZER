"""Read-only LLM analysis: returns drafts, never writes examples or evidence."""
from PyQt5.QtCore import QThread, pyqtSignal
from ..clinical.event_extraction import EventExtractor
from ..clinical.grounded_sources import IncompleteAtomicExtraction, AtomicExtractionCancelled
from ..database.local_lexicon_repo import source_hash


class LexiconAnalysisWorker(QThread):
    completed = pyqtSignal(list,str)
    failed = pyqtSignal(str)
    progress = pyqtSignal(int,int)
    is_annotation_analysis = True

    def __init__(self, llm, document, text, examples, policy, parent=None):
        super().__init__(parent)
        self.llm,self.document,self.text,self.examples,self.policy=llm,document,text,examples,policy

    def run(self):
        try:
            cards={}
            for example in self.examples:
                card=cards.setdefault(example['term_id'],dict(term_id=example['term_id'],label=example['label'],
                    definition=example.get('definition',''),fields=example.get('fields',[]),examples=[]))
                card['examples'].append(example)
            extractor=EventExtractor(self.llm,policy=self.policy,catalog=list(cards.values()))
            extractor.system += ('\nQuesta è una proposta di annotazione umana. Identifica anche eventi non rappresentati '
                                 'nel lessico e proponi nuovi termini sintetici italiani. Gli esempi non limitano gli eventi da cercare. '
                                 'Mantieni asserzione, certezza e soggetto distinti. Non trasformare un dubbio in una presenza certa.')
            warning=''
            try:
                atoms=extractor.extract_document(patient_id=self.document.patient_id,document_id=self.document.id,
                    document_type=self.document.document_type,document_date=self.document.document_date,text=self.text,
                    cancel_check=self.isInterruptionRequested,
                    chunk_progress_callback=lambda n,total:self.progress.emit(n,total))
            except IncompleteAtomicExtraction as exc:
                atoms=exc.evidence
                warning=f'Analisi parziale: {len(exc.issues)} gruppi/problemi non risolti. Le proposte disponibili richiedono revisione.'
            if self.isInterruptionRequested():
                return
            rows=[]
            for atom in atoms:
                spans=atom.data.get('source_spans',[])
                if not spans:
                    continue
                start,end=min(s['start'] for s in spans),max(s['end'] for s in spans)
                if not 0<=start<end<=len(self.text):
                    continue
                state='Non determinato'
                if atom.data.get('experiencer')=='patient' and atom.certainty=='confirmed' and atom.temporality!='hypothetical':
                    state={'present':'Presente','absent':'Assente'}.get(atom.assertion,'Non determinato')
                rows.append(dict(id=atom.evidence_id,document_id=self.document.id,source_hash=source_hash(self.text),
                    start=start,end=end,quote=self.text[start:end],label=atom.normalized_entity,state=state,
                    details=atom.to_atomic_dict(),review='pending'))
            self.completed.emit(rows,warning)
        except AtomicExtractionCancelled:
            pass
        except Exception as exc:
            self.failed.emit(f'Analisi LLM non completata: {exc}')
