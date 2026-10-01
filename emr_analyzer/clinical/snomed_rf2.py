"""Transactional import of an unpacked International Edition RF2 Snapshot.

Retains original RF2 fields, including inactive components. OWL is stored, not
classified; hierarchy queries use the release's inferred relationship snapshot.
"""
import csv
import hashlib
import json
from pathlib import Path
import re

BASE = 'id effectiveTime active moduleId '
COMPONENTS = {
    'concept': ('sct2_Concept_Snapshot_INT_', BASE+'definitionStatusId'),
    'description': ('sct2_Description_Snapshot-en_INT_', BASE+'conceptId languageCode typeId term caseSignificanceId'),
    'language': ('der2_cRefset_LanguageSnapshot-en_INT_', BASE+'refsetId referencedComponentId acceptabilityId'),
    'relationship': ('sct2_Relationship_Snapshot_INT_', BASE+'sourceId destinationId relationshipGroup typeId characteristicTypeId modifierId'),
    'concrete': ('sct2_RelationshipConcreteValues_Snapshot_INT_', BASE+'sourceId value relationshipGroup typeId characteristicTypeId modifierId'),
    'owl': ('sct2_sRefset_OWLExpressionSnapshot_INT_', BASE+'refsetId referencedComponentId owlExpression'),
    'association': ('der2_cRefset_AssociationSnapshot_INT_', BASE+'refsetId referencedComponentId targetComponentId'),
    'definition': ('sct2_TextDefinition_Snapshot-en_INT_', BASE+'conceptId languageCode typeId term caseSignificanceId'),
}
FSN='900000000000003001'
SYN='900000000000013009'
PREFERRED='900000000000548007'
US='900000000000509007'
GB='900000000000508004'


def import_snapshot(catalog, source, progress=None, cancelled=None):
    from .snomed_catalog import file_digest
    root=Path(source)
    if not root.is_dir():
        raise ValueError('Seleziona la cartella estratta della International Edition RF2, non un file GPS.')
    info=json.loads((root/'release_package_information.json').read_text(encoding='utf-8-sig'))
    release=info.get('effectiveTime','')
    if not re.fullmatch(r'20\d{6}',release):
        raise ValueError('Data della release RF2 non valida.')
    files={}
    for name,(prefix,columns) in COMPONENTS.items():
        found=list((root/'Snapshot').rglob(prefix+release+'.txt'))
        if len(found)!=1:
            raise ValueError('Componente Snapshot International mancante o ambiguo: '+prefix)
        files[name]=found[0]
        with found[0].open(encoding='utf-8-sig') as f:
            if f.readline().strip().split('\t')!=columns.split():
                raise ValueError('Intestazione RF2 inattesa: '+found[0].name)
    def check():
        if cancelled and cancelled():
            raise InterruptedError('Importazione annullata; catalogo precedente conservato.')
    def say(message):
        if progress:progress(message)
    digests={}
    for name,path in files.items():
        check();say('Verifica '+path.name);digests[name]=file_digest(path)
    counts={};db=catalog.db
    with db:
        for name,path in files.items():
            check();say('Importazione '+name+'…')
            columns=COMPONENTS[name][1].split();table='sct_rf2_'+name
            db.execute(f'CREATE TABLE IF NOT EXISTS {table} ('+', '.join('"'+c+'" TEXT NOT NULL'+(' PRIMARY KEY' if c=='id' else '') for c in columns)+')')
            db.execute(f'DELETE FROM {table}')
            sql=f'INSERT INTO {table} VALUES ('+','.join('?' for _ in columns)+')'
            count=0;batch=[]
            with path.open(encoding='utf-8-sig',newline='') as f:
                # RF2 is tab-delimited, not CSV: quotation marks in terms are literal.
                reader=csv.reader(f,delimiter='\t',quoting=csv.QUOTE_NONE);next(reader)
                for row in reader:
                    if len(row)!=len(columns) or row[2] not in ('0','1') or not re.fullmatch(r'\d{8}',row[1]) or row[1]>release:
                        raise ValueError(f'Riga RF2 non valida: {path.name}:{count+2}')
                    batch.append(row);count+=1
                    if len(batch)>=10000:
                        check();db.executemany(sql,batch);batch=[]
                        if count%100000==0:say(f'{name}: {count:,} righe')
                if batch:db.executemany(sql,batch)
            counts[name]=count
        say('Indicizzazione delle descrizioni e delle relazioni…');check()
        db.execute('CREATE INDEX IF NOT EXISTS sct_description_concept ON sct_rf2_description(conceptId,active,typeId)')
        db.execute('CREATE INDEX IF NOT EXISTS sct_description_term ON sct_rf2_description(term COLLATE NOCASE,active)')
        db.execute('CREATE INDEX IF NOT EXISTS sct_language_description ON sct_rf2_language(referencedComponentId,active,refsetId,acceptabilityId)')
        db.execute('CREATE INDEX IF NOT EXISTS sct_relationship_source ON sct_rf2_relationship(sourceId,active,typeId)')
        db.execute('CREATE INDEX IF NOT EXISTS sct_relationship_destination ON sct_rf2_relationship(destinationId,active,typeId)')
        db.execute('CREATE INDEX IF NOT EXISTS sct_concrete_source ON sct_rf2_concrete(sourceId,active)')
        db.execute('DELETE FROM sct_concepts')
        # US preferred synonym, GB fallback, then active description. FSN is
        # distinct from the preferred synonym; neither is guessed by the model.
        db.execute('''INSERT INTO sct_concepts(code,active,fsn,term,tag)
          SELECT c.id,CAST(c.active AS INTEGER),COALESCE(d.fsn,''),
                 COALESCE(d.us_term,d.gb_term,d.synonym,d.fsn,''),''
          FROM sct_rf2_concept c LEFT JOIN (
            SELECT d.conceptId,
              MIN(CASE WHEN d.typeId=? THEN d.term END) fsn,
              MIN(CASE WHEN d.typeId=? AND us.acceptabilityId=? THEN d.term END) us_term,
              MIN(CASE WHEN d.typeId=? AND gb.acceptabilityId=? THEN d.term END) gb_term,
              MIN(CASE WHEN d.typeId=? THEN d.term END) synonym
            FROM sct_rf2_description d
            LEFT JOIN sct_rf2_language us ON us.referencedComponentId=d.id AND us.active='1' AND us.refsetId=?
            LEFT JOIN sct_rf2_language gb ON gb.referencedComponentId=d.id AND gb.active='1' AND gb.refsetId=?
            WHERE d.active='1' AND d.languageCode='en' GROUP BY d.conceptId
          ) d ON d.conceptId=c.id''',(FSN,SYN,PREFERRED,SYN,PREFERRED,SYN,US,GB))
        if db.execute("SELECT COUNT(*) FROM sct_concepts WHERE active=1 AND (fsn='' OR term='')").fetchone()[0]:
            raise ValueError('Concetti attivi senza descrizione inglese: importazione rifiutata.')
        updates=[]
        for row in db.execute('SELECT code,fsn FROM sct_concepts'):
            tag=re.search(r'\(([^()]*)\)$',row['fsn'])
            updates.append((tag.group(1) if tag else '',row['code']))
            if len(updates)>=10000:check();db.executemany('UPDATE sct_concepts SET tag=? WHERE code=?',updates);updates=[]
        if updates:db.executemany('UPDATE sct_concepts SET tag=? WHERE code=?',updates)
        say('Indice di ricerca sui termini e sinonimi inglesi…');check()
        db.execute('DELETE FROM sct_search')
        db.execute('''INSERT INTO sct_search(code,term)
          SELECT DISTINCT d.conceptId,d.term FROM sct_rf2_description d
          JOIN sct_concepts c ON c.code=d.conceptId AND c.active=1
          WHERE d.active='1' AND d.languageCode='en' AND d.typeId IN (?,?) AND EXISTS (
            SELECT 1 FROM sct_rf2_language l WHERE l.referencedComponentId=d.id AND l.active='1'
            AND l.refsetId IN (?,?) AND l.acceptabilityId IN (?, '900000000000549004'))''',
            (FSN,SYN,US,GB,PREFERRED))
        active=db.execute('SELECT COUNT(*) FROM sct_concepts WHERE active=1').fetchone()[0]
        if not active:raise ValueError('Catalogo RF2 privo di concetti attivi.')
        # Keep user aliases only if their destination still exists and is active.
        db.execute('DELETE FROM sct_aliases WHERE code NOT IN (SELECT code FROM sct_concepts WHERE active=1)')
        metadata=dict(format='RF2 International Snapshot',release=release,
            version_uri='http://snomed.info/sct/900000000000207008/version/'+release,
            sha256=hashlib.sha256(json.dumps(digests,sort_keys=True).encode()).hexdigest(),
            total=str(counts['concept']),active=str(active),language='en-US; en-GB fallback',
            source=str(root.resolve()),components=json.dumps(counts),file_digests=json.dumps(digests,sort_keys=True))
        check();db.execute('DELETE FROM sct_meta')
        db.executemany('INSERT INTO sct_meta VALUES (?,?)',list(metadata.items()))
    with catalog._lock:
        catalog._vectors=catalog._encoder=None
    say(f'International Edition {release}: {active:,} concetti attivi.')
    return catalog.metadata()
