"""Recover the main text column of reports with an explicit staff directory.

Uses source word geometry, not patient names, hospital names or fixed page crops.
The exclusion is enabled only with multiple clinical alignment anchors and a
vertically extended directory containing both staff roles and contact labels.
"""
from collections import Counter
import re

_CLINICAL_ANCHOR = re.compile(r'^(?:Anamnesi|Dimettiamo|Decorso|Indagini|Controlli|Diagnosi)$', re.I)
_ROLE = re.compile(r'^(?:Direttore|Dirigenti|Segreteria|Coordinatrice|Coordinatore)$', re.I)
_PHONE = re.compile(r'^(?:Tel\.?|Fax\.?)$', re.I)


def _words(page):
    words = getattr(page, 'words', None)
    if not isinstance(words, list):
        return []
    return [w for w in words if isinstance(w, dict) and w.get('text') and
            all(isinstance(w.get(k), (float, int)) for k in ('x0', 'x1', 'top', 'bottom'))]


def detect_sidebar(pages):
    """Return a relative column boundary only for strongly supported layouts."""
    for page in pages:
        words = _words(page)
        width, height = getattr(page, 'width', 0), getattr(page, 'height', 0)
        if not words or not width or not height:
            continue
        roles = [w for w in words if w['x0'] < width * .15 and _ROLE.fullmatch(w['text'])]
        phones = [w for w in words if w['x0'] < width * .15 and _PHONE.fullmatch(w['text'])]
        if len(roles) < 3 or len(phones) < 3:
            continue
        if max(w['top'] for w in roles) - min(w['top'] for w in roles) < height * .2:
            continue
        anchors = [w for w in words if width * .15 < w['x0'] < width * .4 and
                   _CLINICAL_ANCHOR.fullmatch(w['text'])]
        counts = Counter(round(w['x0'] / 3) for w in anchors)
        if not counts or counts.most_common(1)[0][1] < 2:
            continue
        cluster = counts.most_common(1)[0][0]
        # Bullets and unchecked medication boxes can sit a few points to the
        # left of the prose margin. Keep a margin around the clinical column.
        boundary = min(w['x0'] for w in anchors if round(w['x0']/3) == cluster) - 8
        return boundary / width
    return None


def clinical_page_text(page, boundary_ratio):
    """Exclude a staff directory, preserving every other native word."""
    words = _words(page)
    width, height = getattr(page, 'width', 0), getattr(page, 'height', 0)
    if boundary_ratio is None or not words or not width or not height:
        return page.text, {}
    boundary = boundary_ratio * width
    roles = [w for w in words if w['x1'] < boundary and _ROLE.fullmatch(w['text'])]
    phones = [w for w in words if w['x1'] < boundary and _PHONE.fullmatch(w['text'])]
    if len(roles) < 3 or len(phones) < 3:
        return page.text, {}
    top = min(w['top'] for w in roles) - 1
    # The directory ends before the page footer; only the narrow left column
    # is excluded, never a whole line that may also contain a drug or a dose.
    bottom = min(height * .94, max(w['bottom'] for w in roles + phones) + 30)
    removed = [i for i, w in enumerate(words) if w['x1'] <= boundary and top <= w['top'] <= bottom]
    form_zones = []
    titles = [w for w in words if w['text'].upper() == 'LETTERA' and w['top'] < height*.25]
    episodes = [w for w in words if w['text'].upper() == 'EPISODIO' and w['top'] < height*.3]
    if titles and episodes:
        first, last = min(w['bottom'] for w in titles)+2, max(w['bottom'] for w in episodes)+2
        form_zones.append([0, first, width, last])
        for i, word in enumerate(words):
            if first <= word['top'] <= last:
                if word['text'].upper() in {'DATA', 'ACCETTAZIONE', 'DIMISSIONE'} or re.fullmatch(r'\d{1,2}[./-]\d{1,2}[./-]\d{2,4}', word['text']):
                    continue
                removed.append(i)
        identity = [w for w in words if w['text'].upper() == 'ANAGRAFICI' and last < w['top'] < height*.4]
        labels = [w for w in words if w['text'].upper() in {'TELEFONO', 'INDIRIZZO', 'FISCALE'} and last < w['top'] < height*.4]
        if identity and labels:
            first, last = min(w['top'] for w in identity)-2, max(w['bottom'] for w in labels)+2
            form_zones.append([boundary, first, width, last])
            removed.extend(i for i,w in enumerate(words) if w['x0'] >= boundary and first <= w['top'] <= last)
    removed = sorted(set(removed))
    excluded = set(removed)
    kept = [w for i, w in enumerate(words) if i not in excluded]
    rows = []
    for word in sorted(kept, key=lambda w: (w['bottom'], w['x0'])):
        if not rows or abs(rows[-1][0]['bottom'] - word['bottom']) > 2:
            rows.append([word])
        else:
            rows[-1].append(word)
    lines = []
    for row in rows:
        row.sort(key=lambda w: w['x0'])
        pieces = [row[0]['text']]
        for previous, word in zip(row, row[1:]):
            gap = word['x0'] - previous['x1']
            pieces.append(' ' * max(1, min(40, round(gap / 4))) + word['text'])
        lines.append(''.join(pieces))
    return '\n'.join(lines), {
        'version': 'staff-column-v1', 'reason': 'staff_directory_geometry',
        'excluded_bbox': [0, top, boundary, bottom],
        'form_zones': form_zones,
        'source_words': len(words), 'retained_words': len(kept),
        'excluded_word_indices': removed,
    }
