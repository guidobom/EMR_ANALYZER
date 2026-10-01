"""FHIR R4 collection generated deterministically from source-grounded occurrences."""
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import uuid

# Application extensions are explicit, not additional JSON fields on FHIR resources.
EXT='urn:emr-analyzer:fhir:StructureDefinition:'
SNOMED='http://snomed.info/sct'
LOINC='http://loinc.org'


def ident(*parts):
    return str(uuid.uuid5(uuid.NAMESPACE_URL,json.dumps(parts,sort_keys=True,ensure_ascii=False,default=str)))


def reference(resource):
    return {'reference':'urn:uuid:'+resource['id']}


def extension(name,value):
    return {'url':EXT+name,'valueString':str(value) if value not in (None,'') else 'unknown'}


def clinical_date(value):
    if not value:return None
    value=str(value)
    if re.fullmatch(r'\d{4}(?:-\d{2}(?:-\d{2})?)?',value):
        try:
            datetime.strptime(value,{4:'%Y',7:'%Y-%m',10:'%Y-%m-%d'}[len(value)])
            return value
        except ValueError:pass
    return None


def code(text,system=None,number=None,display=None,version=None):
    result={'text':text or 'Evento clinico'}
    if number:
        result['coding']=[{'system':system,'code':number,'display':display or text}]
        if version:result['coding'][0]['version']=version
    return result


def quantity(value,unit=None,operator=None):
    if not math.isfinite(float(value)):raise ValueError('Valore non finito nel laboratorio.')
    result={'value':value}
    if unit:result['unit']=unit
    if operator in ('<','<=','>=','>'):result['comparator']=operator
    # Only units whose UCUM representation is unambiguous are asserted as coded.
    ucum={'mg/dL':'mg/dL','g/dL':'g/dL','g/L':'g/L','mmol/L':'mmol/L','U/L':'U/L','%':'%','mmHg':'mm[Hg]','°C':'Cel'}
    if unit in ucum:result.update(system='http://unitsofmeasure.org',code=ucum[unit])
    return result


class FhirRegistry:
    def __init__(self,patient_id,loinc=None,lab_proposals=None):
        self.patient_id=patient_id
        self.loinc=loinc
        self.lab_proposals=lab_proposals or {}
        self.resources={}
        self.patient={'resourceType':'Patient','id':ident('patient',patient_id),
                      'identifier':[{'system':'urn:emr-analyzer:patient','value':patient_id}]}
        self.add(self.patient)
        self.agent={'resourceType':'Device','id':ident('emr-analyzer-event-extractor'),
                    'deviceName':[{'name':'EMR Analyzer — estrazione automatica','type':'user-friendly-name'}]}
        self.add(self.agent)
        self.documents={}
        self.event_count=0
        self.unmapped=0

    def add(self,resource):
        self.resources[resource['id']]=resource
        return resource

    def document(self,document_id):
        if document_id not in self.documents:
            self.documents[document_id]=self.add({'resourceType':'DocumentReference','id':ident(self.patient_id,'source',document_id),
                'status':'current','subject':reference(self.patient),
                'identifier':[{'system':'urn:emr-analyzer:document','value':document_id}],
                'content':[{'attachment':{'contentType':'text/plain','title':'Documento anonimizzato '+document_id}}]})
        return self.documents[document_id]

    def provenance(self,event,document_id,quote,model=None,source_context=None):
        source=self.document(document_id)
        record={'resourceType':'Provenance','id':ident('provenance',event['id'],document_id,quote),
            'target':[reference(event)],'recorded':datetime.now(timezone.utc).isoformat(),
            'agent':[{'who':reference(self.agent)}],
            'entity':[{'role':'source','what':reference(source)}],
            'extension':[extension('source-quote',quote)]}
        if not quote:record['extension']=[]
        if model:record['extension'].append(extension('extraction-model',model))
        if source_context:
            record['extension'].append(extension('source-context',json.dumps(source_context,ensure_ascii=False)))
        if not record['extension']:record.pop('extension')
        self.add(record)

    def clinical(self,item):
        data=item.data or {}
        from .historical_reuse import historical_identity
        # No temporal clustering here: only identical source occurrences share an id.
        rid=ident(self.patient_id,item.document_id,item.fact_type,item.normalized_entity,
            data.get('experiencer'),item.certainty,item.source_text,
            data.get('source_spans'),item.observed_date,item.assertion,item.clinical_status,item.typed_payload)
        shared = historical_identity(item)
        if shared:
            rid = ident(self.patient_id, 'historical-event', shared)
        if rid in self.resources:
            event = self.resources[rid]
            self.provenance(event,item.document_id,item.source_text,item.model_name,
                {k:data.get(k) for k in ('source_spans','date_provenance','source_relations','relation_review')})
            return
        concept=code(item.normalized_entity,SNOMED,data.get('snomed_concept_id'),data.get('snomed_term'),data.get('snomed_release'))
        date=clinical_date(item.observed_date)
        end=clinical_date(item.observed_date_end)
        details={'assertion':item.assertion,'certainty':item.certainty,'subject':data.get('experiencer','unknown'),
            'temporality':item.temporality,'state':item.clinical_status,'attributes':data.get('attributes',{}),
            'date_provenance':data.get('date_provenance',{}),'date_precision':item.date_precision,
            'date_proposal':data.get('date_proposal'), 'value_review':data.get('value_review'),
            'intermediate_repairs':data.get('intermediate_repairs',[]),
            'mapping_status':data.get('snomed_mapping_status','needs_review'),
            'mapping_reason':data.get('snomed_mapping_reason',''),'fact_type':item.fact_type,
            'source_relations':data.get('source_relations',[]),
            'relation_review':data.get('relation_review',{})}
        base={'id':rid,'extension':[extension('extraction-context',json.dumps(details,ensure_ascii=False)),
            extension('review-status','needs-review' if item.status=='needs_review' or not concept.get('coding') else 'proposed')]}
        subject=data.get('experiencer')
        if subject=='family' and item.assertion=='present':
            event={**base,'resourceType':'FamilyMemberHistory','status':'partial','patient':reference(self.patient),
                'relationship':{'text':'Familiare; relazione non determinata'},'condition':[{'code':concept}]}
        elif subject not in ('patient',):
            # Never silently attribute another person's finding to this patient.
            event={**base,'resourceType':'Observation','status':'unknown','code':concept,
                   'note':[{'text':'Soggetto: '+str(subject or 'non determinato')}]}
            if date:event['effectiveDateTime']=date
        elif item.fact_type=='diagnosis' and item.assertion=='present':
            verification='confirmed' if item.certainty=='confirmed' else 'provisional' if item.certainty in ('possible','suspected') else 'unconfirmed'
            event={**base,'resourceType':'Condition','subject':reference(self.patient),'code':concept,
                'verificationStatus':code(verification,'http://terminology.hl7.org/CodeSystem/condition-ver-status',verification)}
            if date:event['extension'].append(extension('event-date', date))
            if end:event['extension'].append(extension('event-date-end', end))
        elif (item.fact_type=='medication' and item.assertion=='present'
              and data.get('attributes',{}).get('action')=='prescribed'):
            event={**base,'resourceType':'MedicationRequest','subject':reference(self.patient),
                'status':'unknown','intent':'order','medicationCodeableConcept':concept}
            if date:event['authoredOn']=date
            if data.get('attributes',{}).get('dose'):
                event['dosageInstruction']=[{'text':data['attributes']['dose']}]
        elif (item.fact_type=='medication' and item.assertion=='present' and date
              and data.get('attributes',{}).get('action')=='administered'):
            event={**base,'resourceType':'MedicationAdministration','subject':reference(self.patient),
                'status':'completed' if item.certainty=='confirmed' else 'unknown',
                'medicationCodeableConcept':concept,'effectivePeriod':{'start':date,**({'end':end} if end else {})}}
            if data.get('attributes',{}).get('dose'):
                event['dosage']={'text':data['attributes']['dose']}
        elif item.fact_type=='medication' and item.assertion!='unknown':
            state={'active':'active','in corso':'active','stopped':'stopped','sospeso':'stopped','completed':'completed'}.get((item.clinical_status or '').casefold(),'unknown')
            event={**base,'resourceType':'MedicationStatement','subject':reference(self.patient),
                'status':'not-taken' if item.assertion=='absent' else state,'medicationCodeableConcept':concept}
            if date:event['effectivePeriod']={'start':date,**({'end':end} if end else {})}
            dose=data.get('attributes',{}).get('dose')
            if dose:event['dosage']=[{'text':dose}]
        elif item.fact_type=='procedure':
            event={**base,'resourceType':'Procedure','subject':reference(self.patient),'code':concept,
                   'status':'not-done' if item.assertion=='absent' else 'unknown'}
            if date:event['performedPeriod']={'start':date,**({'end':end} if end else {})}
        else:
            event={**base,'resourceType':'Observation','subject':reference(self.patient),'status':'unknown','code':concept}
            if date:event['effectivePeriod']={'start':date,**({'end':end} if end else {})}
            if item.assertion=='absent':event['valueBoolean']=False
            elif item.numeric_value is not None:event['valueQuantity']=quantity(item.numeric_value,item.unit,data.get('attributes',{}).get('comparator'))
            elif item.value_text:event['valueString']=item.value_text
            elif item.assertion=='present' and item.certainty=='confirmed' and item.fact_type!='laboratory_test':event['valueBoolean']=True
            else:event['dataAbsentReason']=code('Non determinato','http://terminology.hl7.org/CodeSystem/data-absent-reason','unknown')
        self.add(event)
        self.event_count+=1
        self.unmapped+=int(not concept.get('coding'))
        self.provenance(event,item.document_id,item.source_text,item.model_name,
            {k:data.get(k) for k in ('source_spans','date_provenance','source_relations','relation_review')})

    def laboratory(self,lab,key):
        matched=self.loinc.resolve(lab) if self.loinc else None
        if not matched and self.loinc:
            matched=self.lab_proposals.get(self.loinc.signature(lab.normalized_name or lab.parameter_name,lab.biological_material,lab.unit))
        concept=code(lab.parameter_name,LOINC,matched['code'] if matched else None,
            matched['label'] if matched else None,self.loinc.metadata().get('release') if matched else None)
        event={'resourceType':'Observation','id':ident(self.patient_id,'lab',key),
            'status':'unknown','subject':reference(self.patient),'code':concept,
            'category':[code('Laboratorio','http://terminology.hl7.org/CodeSystem/observation-category','laboratory')],
            'extension':[extension('review-status','validated' if matched and lab.validated_by_user and matched.get('mapping_review')!='proposed' else 'needs-review'),
                extension('mapping-status',matched.get('mapping_review','reviewed') if matched else 'unmapped'),
                extension('original-unit',lab.unit or ''),extension('original-analyte',lab.parameter_name)]}
        if lab.biological_material:
            specimen=self.add({'resourceType':'Specimen','id':ident(event['id'],'specimen'),
                'subject':reference(self.patient),'type':{'text':lab.biological_material}})
            event['specimen']=reference(specimen)
        date=clinical_date(lab.sample_date)
        if date:event['effectiveDateTime']=date
        elif lab.sample_date:event['extension'].append(extension('original-date',lab.sample_date))
        if lab.value is not None:event['valueQuantity']=quantity(lab.value,lab.unit,lab.operator)
        elif lab.value_text:event['valueString']=lab.value_text
        else:event['dataAbsentReason']=code('Risultato mancante','http://terminology.hl7.org/CodeSystem/data-absent-reason','unknown')
        interval={}
        if lab.reference_low is not None:interval['low']=quantity(lab.reference_low,lab.unit)
        if lab.reference_high is not None:interval['high']=quantity(lab.reference_high,lab.unit)
        if lab.reference_text:interval['text']=lab.reference_text
        if interval:event['referenceRange']=[interval]
        direction=lab.flag if lab.flag in ('H','L') else 'A' if lab.is_abnormal else None
        if direction:event['interpretation']=[code(direction,'http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation',direction)]
        if lab.page is not None:event['extension'].append(extension('source-page',lab.page))
        self.add(event);self.event_count+=1;self.unmapped+=int(not matched)
        self.provenance(event,lab.document_id,lab.source_text)

    def bundle(self,coverage):
        self.add({'resourceType':'Basic','id':ident(self.patient_id,'coverage'),
            'code':{'text':'Manifesto di estrazione'},'subject':reference(self.patient),
            'extension':[extension('extraction-coverage',json.dumps(coverage,ensure_ascii=False)),
                         extension('uncoded-event-count',self.unmapped),
                         extension('validation-level','local-structural-checks; not HL7 profile certification')]})
        return {'resourceType':'Bundle','id':ident(self.patient_id,'clinical-events'),
            'type':'collection','timestamp':datetime.now(timezone.utc).isoformat(),
            'meta':{'tag':[{'system':'urn:emr-analyzer:pipeline','code':'fhir-events-v1'}]},
            'entry':[{'fullUrl':'urn:uuid:'+r['id'],'resource':r} for r in self.resources.values()]}

    def write(self,path,coverage):
        bundle=self.bundle(coverage)
        validate_bundle(bundle)
        target=Path(path);target.parent.mkdir(parents=True,exist_ok=True)
        fd,temp=tempfile.mkstemp(prefix='.clinical-events-',suffix='.json',dir=target.parent)
        try:
            with os.fdopen(fd,'w',encoding='utf-8') as handle:
                json.dump(bundle,handle,ensure_ascii=False,indent=2,allow_nan=False)
                handle.flush();os.fsync(handle.fileno())
            os.replace(temp,target)
        finally:
            if os.path.exists(temp):os.unlink(temp)
        return str(target)


def validate_bundle(bundle):
    """Operational referential/shape checks; full HL7 profile validation remains separate."""
    if bundle.get('resourceType')!='Bundle' or bundle.get('type')!='collection':raise ValueError('Bundle non valido.')
    entries=bundle.get('entry',[])
    urls={e['fullUrl'] for e in entries}
    if len(urls)!=len(entries):raise ValueError('Identificativi FHIR duplicati.')
    def walk(value):
        if isinstance(value,dict):
            if 'reference' in value and value['reference'].startswith('urn:uuid:') and value['reference'] not in urls:
                raise ValueError('Riferimento FHIR non risolto.')
            for v in value.values():walk(v)
        elif isinstance(value,list):
            for v in value:walk(v)
    required={'Specimen':('type',),'Patient':('identifier',),'Device':('deviceName',),'Basic':('code',),
        'Observation':('status','code'),'Condition':('subject',),
        'Procedure':('status','subject'),'MedicationStatement':('status','subject','medicationCodeableConcept'),
        'FamilyMemberHistory':('status','patient','relationship'),
        'MedicationRequest':('status','intent','subject','medicationCodeableConcept'),
        'MedicationAdministration':('status','subject','medicationCodeableConcept','effectivePeriod'),
        'DocumentReference':('status','content'),'Provenance':('target','recorded','agent')}
    for entry in entries:
        resource=entry['resource'];kind=resource['resourceType']
        if kind not in required or any(not resource.get(k) for k in required[kind]):
            raise ValueError('Risorsa FHIR incompleta: '+kind)
        if entry['fullUrl']!='urn:uuid:'+resource['id']:raise ValueError('Identità FHIR incoerente.')
    walk(bundle)
    json.dumps(bundle,allow_nan=False)
    # R4 models validate known fields, primitive types and required elements when installed.
    try:
        from fhirclient.models.bundle import Bundle
    except ImportError:
        return
    Bundle(bundle,strict=True)


def lab_occurrence_keys(labs):
    """Stable identity of each parsed result, in input order.

    Only raw report content counts: review flags, derived interpretation and
    results of other documents never change it. Identical rows of the same
    document are told apart by a counter.
    """
    seen, keys = {}, []
    for lab in labs:
        content = (lab.document_id, lab.parameter_name, lab.value, lab.value_text,
                   lab.operator, lab.unit, lab.reference_text, lab.sample_date,
                   lab.page, lab.source_text)
        count = seen.get(content, 0)
        seen[content] = count + 1
        keys.append(json.dumps([*content, count], ensure_ascii=False, default=str))
    return keys
