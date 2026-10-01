"""Source-preserving sentence groups. No generated summaries or clinical filtering."""
from dataclasses import dataclass
import re
from .evidence_utils import SentenceSpan

_ABBREVIATION = re.compile(
    r"(?i)(?:\b(?:dott|dr|prof|sig|sigg|pz|ca|es|ecc|cfr|n|art|fig|vol|vs|ev|im|sc)|\b[A-Z])\.$"
)


def clinical_sentences(text):
    """Keep offsets into the original text; do not split PDF line wraps or decimals."""
    cuts = [0]
    for match in re.finditer(r"\n\s*\n+|(?<=[.!?;])\s+", text):
        prefix = text[max(0, match.start() - 30):match.start()]
        if not re.search(r'\n\s*\n', match.group()) and _ABBREVIATION.search(prefix):
            continue
        cuts.append(match.end())
    cuts.append(len(text))
    result = []
    for start, end in zip(cuts, cuts[1:]):
        while start < end and text[start].isspace(): start += 1
        while end > start and text[end - 1].isspace(): end -= 1
        if start < end:
            result.append(SentenceSpan(len(result) + 1, start, end, text[start:end]))
    return result


@dataclass(frozen=True)
class SentenceGroup:
    targets: tuple
    context: tuple = ()
    #: Sentences already certified as a statement of this patient: visible to
    #: the model as context, never source of a new fact.
    copies: tuple = ()

    @property
    def spans(self):
        return tuple(sorted((*self.context, *self.targets, *self.copies),
                            key=lambda s: (s.start, s.end)))

    @property
    def numbered(self):
        return [SentenceSpan(i, s.start, s.end, s.text)
                for i, s in enumerate(self.spans, 1)]

    @property
    def target_ids(self):
        positions = {(s.start, s.end) for s in self.targets}
        return {i for i, s in enumerate(self.spans, 1) if (s.start, s.end) in positions}

    @property
    def copy_ids(self):
        positions = {(s.start, s.end) for s in self.copies}
        return {i for i, s in enumerate(self.spans, 1) if (s.start, s.end) in positions}

    def prompt(self, document_type, document_date):
        lines = [f"CONTESTO: tipo_dichiarato={document_type}; data_documento={document_date or 'non disponibile'}; pagine=n.d.",
                 "Estrai solo fatti delle frasi OBIETTIVO. Ogni oggetto deve citarne almeno una.",
                 "Puoi citare anche il CONTESTO se sostiene soggetto, data o attributi; non estrarne altri fatti."]
        if self.copies:
            lines.append("Le frasi COPIA ripetono un fatto già annotato altrove per questo "
                         "paziente: non estrarne nulla, servono solo a capire il contesto.")
        for s in self.numbered:
            if s.sentence_id in self.target_ids:
                role = 'OBIETTIVO'
            elif s.sentence_id in self.copy_ids:
                role = 'COPIA'
            else:
                role = 'CONTESTO'
            lines.append(f"{role} [S{s.sentence_id}] {s.text}")
        return '\n'.join(lines)


def plan_sentence_groups(text, count_tokens, target_tokens=1000):
    spans = clinical_sentences(text)
    if not spans: return []
    costs = [count_tokens(s.text) + 10 for s in spans]
    groups, start, size = [], 0, 0
    for i, cost in enumerate(costs):
        if i > start and size + cost > target_tokens:
            groups.append(_group(spans, start, i))
            start, size = i, 0
        size += cost
    groups.append(_group(spans, start, len(spans)))
    return groups


def _group(spans, start, end):
    context = list(spans[max(0, start - 2):start])
    # A following sentence can resolve an antecedent/abbreviation too.
    context += spans[end:end + 1]
    # Carry the nearest short section/date heading without a summary.
    for prior in reversed(spans[:start]):
        if len(prior.text) <= 120 and (prior.text.rstrip().endswith(':') or
                re.fullmatch(r'[\d/ .:-]+', prior.text)):
            if prior not in context: context.append(prior)
            break
    return SentenceGroup(tuple(spans[start:end]), tuple(context))


def plan_statement_groups(text, roles, count_tokens, target_tokens=1000):
    """Groups whose targets are only the origin sentences of the patient.

    ``roles`` maps the sentence id to ``'origin'`` or ``'copy'``.  Copy
    sentences stay visible — they carry the context a model needs — but they
    are never a target: their statement is certified once, elsewhere.  Groups
    made only of copies produce no call at all.
    """
    groups, _ = statement_group_plan(text, roles, count_tokens, target_tokens)
    return groups


def statement_group_plan(text, roles, count_tokens, target_tokens=1000):
    """The planned groups and how many groups were skipped as all-copies."""
    spans = clinical_sentences(text)
    if not spans:
        return [], 0
    costs = [count_tokens(s.text) + 10 for s in spans]
    groups, skipped, start, size = [], 0, 0, 0
    for i, cost in enumerate(costs):
        if i > start and size + cost > target_tokens:
            group = _statement_group(spans, start, i, roles)
            if group.targets:
                groups.append(group)
            else:
                skipped += 1
            start, size = i, 0
        size += cost
    group = _statement_group(spans, start, len(spans), roles)
    if group.targets:
        groups.append(group)
    else:
        skipped += 1
    return groups, skipped


def _statement_group(spans, start, end, roles):
    body = spans[start:end]
    copies = tuple(s for s in body if roles.get(s.sentence_id) == 'copy')
    targets = tuple(s for s in body if roles.get(s.sentence_id) != 'copy')
    base = _group(spans, start, end)
    copies += tuple(s for s in base.context
                    if roles.get(s.sentence_id) == 'copy' and s not in copies)
    context = tuple(s for s in base.context if s not in copies and s not in targets)
    return SentenceGroup(targets, context, copies)


def split_group(group):
    if len(group.targets) > 1:
        mid = len(group.targets) // 2
        left, right = group.targets[:mid], group.targets[mid:]
        # A copy belongs to one half only; context stays fully visible.
        left_copies = tuple(s for s in group.copies if s.start < right[0].start)
        right_copies = tuple(s for s in group.copies if s.start >= right[0].start)
        return (SentenceGroup(left, tuple(dict.fromkeys((*group.context, right[0]))), left_copies),
                SentenceGroup(right, tuple(dict.fromkeys((*group.context, left[-1]))), right_copies))
    # A sentence is the smallest semantic unit. Splitting it at arbitrary
    # whitespace can detach negation, subject or dose from the fact. Surface
    # an explicit review condition when even one intact sentence cannot fit.
    return ()
