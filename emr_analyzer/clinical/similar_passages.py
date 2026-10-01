"""Read-only passage similarity search; scores are not clinical probabilities."""
import re
import unicodedata
from difflib import SequenceMatcher
from .sentence_groups import clinical_sentences


def normalized(text):
    return ' '.join(''.join(c for c in unicodedata.normalize('NFKD',text.casefold())
                           if not unicodedata.combining(c)).split())


def passage_spans(text, max_chars=700):
    """Bound encoder inputs while retaining exact character positions."""
    for sentence in clinical_sentences(text):
        start=sentence.start
        while start<sentence.end:
            end=min(sentence.end,start+max_chars)
            if end<sentence.end:
                boundary=text.rfind(' ',start+max_chars//2,end)
                if boundary>start:
                    end=boundary
            yield start,end,text[start:end]
            if end==sentence.end:
                break
            # Overlap preserves expressions at window boundaries.
            following=text.find(' ',max(start+1,end-100),end)
            start=following+1 if following>=0 else end
            while start<sentence.end and text[start].isspace():
                start+=1


def score_passages(query, passages, encoder=None, query_vector=None):
    if not passages:
        return []
    if encoder is not None:
        import numpy as np
        q=query_vector if query_vector is not None else encoder.encode([query],normalize_embeddings=True,show_progress_bar=False)[0]
        vectors=encoder.encode(passages,normalize_embeddings=True,show_progress_bar=False)
        return [float(v) for v in np.asarray(vectors) @ q]
    q=normalized(query)
    qw=set(re.findall(r'\w+',q))
    scores=[]
    for passage in passages:
        p=normalized(passage)
        pw=set(re.findall(r'\w+',p))
        token_overlap=len(qw & pw)/max(1,len(qw | pw))
        scores.append(max(token_overlap,SequenceMatcher(None,q,p,autojunk=False).ratio()))
    return scores
