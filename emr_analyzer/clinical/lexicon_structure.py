"""User-defined event cards; occurrence values stay separate from the catalog."""
import math
import re

VALUE_TYPES = {'none': 'Nessun valore', 'number': 'Numero', 'composite': 'Valore composto',
               'text': 'Testo', 'choice': 'Scelta da elenco'}
FIELD_TYPES = {'text': 'Testo', 'number': 'Numero', 'choice': 'Scelta da elenco'}
DEDUP_RULES = {'auto': 'Automatica (compatibilità precedente)',
              'condition': 'Condizione: entro 15 giorni',
              'measurement': 'Misurazione: solo stessa osservazione',
              'action': 'Azione: solo stesso episodio',
              'therapy': 'Terapia: stesso farmaco, dose e stato'}
CATEGORIES = ['Parametri vitali', 'Sintomi', 'Diagnosi', 'Terapie', 'Procedure',
              'Esami di laboratorio', 'Reperti strumentali', 'Altro']


def normalize_structure(value):
    result = dict(category=str(value.get('category', '')).strip(),
                  value_type=value.get('value_type', 'none'),
                  dedup_rule=value.get('dedup_rule', 'auto'), field_definitions=[])
    if len(result['category']) > 100 or result['value_type'] not in VALUE_TYPES or result['dedup_rule'] not in DEDUP_RULES:
        raise ValueError('Categoria, tipo di valore o regola non validi.')
    names = set()
    for field in value.get('field_definitions', []):
        name = str(field.get('name', '')).strip()
        kind = field.get('type', 'text')
        unit = str(field.get('unit', '')).strip()
        choices = list(dict.fromkeys(str(c).strip() for c in field.get('choices', []) if str(c).strip()))
        if not name or len(name)>100 or name.casefold() in names or kind not in FIELD_TYPES or len(unit)>40:
            raise ValueError('Ogni campo richiede un nome univoco e un tipo valido.')
        if kind == 'choice' and not choices:
            raise ValueError('Inserisci le opzioni per i campi a scelta.')
        if len(choices)>50 or any(len(c)>100 for c in choices):
            raise ValueError('Massimo 50 opzioni di 100 caratteri per campo.')
        names.add(name.casefold())
        result['field_definitions'].append(dict(name=name, type=kind, unit=unit, choices=choices))
    fields = result['field_definitions']
    if len(fields)>20:
        raise ValueError('Massimo 20 campi per termine.')
    main = result['value_type']
    if main in ('number', 'text', 'choice') and (not fields or fields[0]['type'] != main):
        raise ValueError('Il primo campo deve avere il tipo del valore principale.')
    if main == 'composite' and len(fields)<2:
        raise ValueError('Un valore composto richiede almeno due campi.')
    return result


def occurrence_values(card, attributes):
    """Parse configured fields, without guessing units, defaults or missing values."""
    values, units = {}, {}
    for field in card.get('field_definitions', []):
        name = field['name']
        raw = attributes.get(name)
        if raw is None or raw == '':
            continue
        if field['type'] == 'number':
            text = str(raw).strip().replace(',', '.')
            if not re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)', text):
                raise ValueError(f'Valore numerico non valido: {name}')
            raw = float(text)
            if not math.isfinite(raw):
                raise ValueError(f'Valore numerico non finito: {name}')
        elif field['type'] == 'choice' and raw not in field['choices']:
            raise ValueError(f'Valore fuori elenco: {name}')
        values[name] = raw
        if field['unit']:
            # Unit is the declared extraction contract, never converted here.
            units[name] = field['unit']
    return values, units
