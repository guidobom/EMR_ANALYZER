"""Join clinical pages and keep their provenance out of the readable Markdown."""
from collections import Counter
import hashlib
import re

_TIMESTAMP = re.compile(r'^\d{1,2}[./-]\d{1,2}[./-]\d{4}[ \t]+\d{1,2}:\d{2}(?::\d{2})?$')
_EMPTY = '[NESSUN CONTENUTO CLINICO]'
_PAGE_MARKER = re.compile(r'^[ \t]*<!--[ \t]*page:[ \t]*(\d+)[ \t]*-->[ \t]*\r?$', re.M | re.I)


def source_pages(text):
    """Recover explicit legacy page boundaries before filtering their contents.

    Never guess page numbers or discard text before the first marker. Invalid
    sequences fail rather than create misleading PDF citations.
    """
    markers = list(_PAGE_MARKER.finditer(text))
    if not markers:
        return [(None, text)]
    numbers = [int(m[1]) for m in markers]
    if numbers != list(range(1, len(numbers) + 1)):
        raise ValueError('Sequenza dei separatori di pagina non valida')
    prefix = text[:markers[0].start()]
    pages = []
    for i, marker in enumerate(markers):
        end = markers[i + 1].start() if i + 1 < len(markers) else len(text)
        body = text[marker.end():end]
        if i == 0:
            body = prefix + body
        pages.append((numbers[i], body))
    return pages


def merge_pages(pages):
    # Only exact standalone timestamps repeated in the first three nonempty
    # lines of different pages are page furniture. Narrative dates stay intact.
    headers = Counter()
    for _, text in pages:
        top = [l.strip() for l in text.splitlines() if l.strip()][:3]
        headers.update(set(l for l in top if _TIMESTAMP.fullmatch(l)))
    repeated = {value for value,n in headers.items() if n > 1}
    output, mapping, removals = '', [], []
    for number, text in pages:
        kept, ordinal = [], 0
        for match in re.finditer(r'[^\n]*(?:\n|$)', text):
            value = match.group().strip()
            if value:
                ordinal += 1
            if ordinal <= 3 and value in repeated:
                removals.append({'page':number,'start':match.start(),'end':match.end(),
                                 'reason':'repeated_page_timestamp'})
            else:
                kept.append(match.group())
        body = ''.join(kept).strip()
        empty = body == _EMPTY or not body
        if empty:
            body = ''
        if body and output:
            output += '\n'
        start = len(output)
        output += body
        mapping.append({'page':number,'start':start,'end':len(output), 'administrative_only':empty})
    if not output:
        output = _EMPTY
        mapping[0]['end'] = len(output)
        for entry in mapping[1:]:
            entry['start'] = entry['end'] = len(output)
    return output, {'output_pages':mapping,
                    'output_sha256':hashlib.sha256(output.encode()).hexdigest(),
                    'page_join_version':'continuous-markdown-v1',
                    'join_removals':removals}
