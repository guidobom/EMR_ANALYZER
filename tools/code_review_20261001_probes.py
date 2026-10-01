"""Synthetic, isolated reproductions for the code-review report.

Never accesses patient workspaces. All databases and artifacts use TemporaryDirectory.
Run with the project's Python environment and a JSON output path.
"""
import importlib
import json
from pathlib import Path
import sys
import tempfile
from dataclasses import replace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emr_analyzer.database.engine import DatabaseEngine
from emr_analyzer.database.migrations import init_database
from emr_analyzer.database.patient_repo import PatientRepository
from emr_analyzer.database.document_repo import DocumentRepository
from emr_analyzer.database.lab_repo import LabRepository
from emr_analyzer.database.audit_repo import AuditRepository
from emr_analyzer.database.local_lexicon_repo import LocalLexiconRepository
from emr_analyzer.database.shared_lexicon_repo import SharedLexiconRepository
from emr_analyzer.models import Patient
from emr_analyzer.models.document import DocumentRecord
from emr_analyzer.models.lab_result import LabValue
from emr_analyzer.extraction.normalizer import LabNormalizer
from emr_analyzer.evaluation.metrics import evaluate_patient_events
from emr_analyzer.clinical.fhir_registry import FhirRegistry
from emr_analyzer.clinical.document_deletion import DocumentDeletionService
from emr_analyzer.clinical.patient_deletion import PatientWorkspaceDeletionService


def main():
    results = {}
    with tempfile.TemporaryDirectory(prefix='emr-source-review-') as temp:
        root = Path(temp)
        workspace = root / 'workspace'
        db = DatabaseEngine(root / 'review.db')
        init_database(db)
        init_database(db)
        results['migration_idempotence'] = {'foreign_key_errors': [list(r) for r in db.execute('PRAGMA foreign_key_check')],
            'integrity': db.execute('PRAGMA integrity_check').fetchone()[0]}
        patients = PatientRepository(db)
        documents = DocumentRepository(db)
        patients.insert(Patient(id='P001', pseudonym='SINTETICO'))
        patient_root = workspace / 'P001'
        patient_root.mkdir(parents=True)
        original = patient_root / 'synthetic.txt'
        original.write_text('Esempio sintetico: tosse.', encoding='utf-8')
        documents.insert(DocumentRecord(id='DOC_001',patient_id='P001',filename=original.name,
            original_path=str(original),file_hash='synthetic',document_date='2026-09-01'))

        lab = LabValue(patient_id='P001',document_id='DOC_001',parameter_name='Glucosio',
            normalized_name='glucosio',value=100,unit='mg/dL',confidence=0.0)
        labs = LabRepository(db)
        labs.insert_batch([lab])
        results['zero_lab_confidence'] = {'input':0.0,'persisted_reload':labs.get_by_patient('P001')[0].confidence}
        normalizer = LabNormalizer()
        results['lab_synonyms'] = {s:normalizer.normalize_parameter(s) for s in ('GOT','AST','GPT','ALT','creat.')}
        results['signed_numeric_values'] = {s:normalizer.normalize_value(s) for s in ('-3.2','-3,2','+3,2','1.234,56')}
        results['censored_lab_flags'] = {'less_than_100_range_0_50':normalizer.is_abnormal(100,0,50,'<'),
                                        'greater_than_10_range_20_100':normalizer.is_abnormal(10,20,100,'>')}
        from emr_analyzer.extraction.lab_parser import LabParser
        results['censored_lab_active_parser']=[v.to_dict() for v in LabParser().parse(
            'Glucosio <100 mg/dL (0-50)',patient_id='P001',document_id='DOC_001')]

        gold = dict(category='diagnosis',canonical_entity='diabete',first_evidence_date='2026-09-01',
                    assertion='present',certainty='confirmed',status='active',experiencer='patient')
        opposite = dict(gold,assertion='absent',experiencer='family')
        results['opposite_assertion_evaluation'] = evaluate_patient_events([gold],[opposite])

        fhir = FhirRegistry('P001')
        fhir.laboratory(lab,0)
        first_id = next(r['id'] for r in fhir.resources.values() if r['resourceType']=='Observation')
        other = FhirRegistry('P001')
        other.laboratory(replace(lab,validated_by_user=True),0)
        second_id = next(r['id'] for r in other.resources.values() if r['resourceType']=='Observation')
        results['fhir_lab_identity_on_review'] = {'before':first_id,'after':second_id,'changed':first_id!=second_id}
        reordered = FhirRegistry('P001')
        reordered.laboratory(lab,1)
        results['fhir_lab_identity_on_order'] = {'changed':first_id!=next(r['id'] for r in reordered.resources.values() if r['resourceType']=='Observation')}

        audit = AuditRepository(db)
        audit.log('P001','first')
        audit.log('P001','second')
        before = audit.verify_chain('P001')
        db.execute("DELETE FROM audit_log WHERE id=(SELECT MAX(id) FROM audit_log WHERE patient_id=?)",('P001',))
        db.commit()
        results['audit_tail_removal'] = {'before':before,'after_tail_removal':audit.verify_chain('P001'),
            'head_still_present':bool(db.execute('SELECT last_hash FROM audit_chain_heads WHERE patient_id=?',('P001',)).fetchone()[0])}

        from emr_analyzer.clinical.patient_import import PatientImportService
        from emr_analyzer.database.timeline_repo import TimelineRepository
        from emr_analyzer.models.clinical_timeline import ClinicalTimelineEntry
        from emr_analyzer.database.evidence_repo import EvidenceRepository
        from emr_analyzer.models.clinical_evidence import ClinicalEvidence
        labs.insert(replace(lab,value=None,value_text='NEGATIVO'))
        EvidenceRepository(db).insert_batch([ClinicalEvidence(patient_id='P001',document_id='DOC_001',
            category='symptom',normalized_entity='tosse',source_text='tosse')])
        from emr_analyzer.database.registry_repo import ClinicalRegistryRepository
        from emr_analyzer.models.clinical_registry import ClinicalEvent, EventEvidenceLink
        from emr_analyzer.export.registry_export import ClinicalRegistryExporter
        registry=ClinicalRegistryRepository(db)
        event=ClinicalEvent(patient_id='P001',category='symptom',canonical_entity='tosse',summary_short='Sintomo')
        registry.save_event(event)
        registry._save_link(EventEvidenceLink(event_id=event.event_id,
            evidence_id=EvidenceRepository(db).get_by_patient('P001')[0].evidence_id))
        db.commit()
        exported=ClinicalRegistryExporter({'registry_repo':registry}).collect('P001',include_sources=False)
        exported_evidence=exported['events'][0]['evidence'][0]
        results['export_sources_disabled']={'top_level_source_text_present':'source_text' in exported_evidence,
            'source_ref_passages':[r['passage'] for r in exported_evidence['source_refs']]}
        import_db = DatabaseEngine(root/'import.db')
        init_database(import_db)
        stats = dict(patients=0,documents=0,timeline=0,profiles=0)
        PatientImportService()._import_one(workspace,db,root/'import',import_db,'P001',stats)
        results['patient_import_textual_lab'] = [dict(r) for r in import_db.execute('SELECT value,value_text FROM lab_values')]
        results['patient_import_evidence'] = {'source':db.execute('SELECT COUNT(*) FROM clinical_evidence').fetchone()[0],
            'target':import_db.execute('SELECT COUNT(*) FROM clinical_evidence').fetchone()[0]}
        import_db.close()
        TimelineRepository(db).save_batch([ClinicalTimelineEntry(entry_id='CTL_001',patient_id='P001',
            date_observed='2026-09-01',category='symptom',description='tosse'),
            ClinicalTimelineEntry(entry_id='CTL_002',patient_id='P001',date_observed='2026-09-02',category='symptom',description='astenia')])
        collision_db=DatabaseEngine(root/'import-collision.db')
        init_database(collision_db)
        try:
            PatientImportService()._import_one(workspace,db,root/'import-collision',collision_db,'P001',
                dict(patients=0,documents=0,timeline=0,profiles=0))
            results['patient_import_timeline_collision']={'error':None}
        except Exception as exc:
            collision_db.rollback()
            results['patient_import_timeline_collision']={'error':type(exc).__name__+': '+str(exc),
                'patients_left_after_rollback':collision_db.execute('SELECT COUNT(*) FROM patients').fetchone()[0],
                'copied_files_left':len(list((root/'import-collision').rglob('synthetic.txt')))}
        collision_db.close()

        local = LocalLexiconRepository(db)
        annotation = local.save('DOC_001','Tosse persistente.',0,5,'Tosse')
        term = local.terms()[0]['id']
        local.set_event_definition(term,'Sintomo respiratorio',['durata'])
        local.set_example_role(annotation,'counterexample')
        shared = SharedLexiconRepository(db,workspace,root/'shared.db')
        imported = shared.annotations()[0]
        results['shared_migration_semantics'] = {'local_definition':local.event_definition(term),
            'shared_definition':shared.event_definition(imported['term_id']),
            'local_role':local.example_role(annotation),'shared_role':shared.example_role(imported['id'])}

        path = patient_root/'clinical_events.fhir.json'
        fhir.write(path,{'complete':True})
        deletion = DocumentDeletionService(db,documents,workspaces_dir=workspace).delete('DOC_001')
        results['document_delete_with_evidence']={'deleted':deletion.deleted,'error':deletion.error,
            'original_restored':original.exists()}
        if not deletion.deleted:
            EvidenceRepository(db).delete_by_document('DOC_001')
            deletion = DocumentDeletionService(db,documents,workspaces_dir=workspace).delete('DOC_001')
        results['document_delete_residues'] = {'deleted':deletion.deleted,'error':deletion.error,
            'fhir_exists':path.exists(),'shared_annotations':len(shared.annotations()),
            'shared_full_texts':shared.db.execute('SELECT COUNT(*) FROM annotated_source_texts').fetchone()[0]}
        # A new centrally saved annotation demonstrates persistence of full text.
        original.write_text('Sintetico',encoding='utf-8')
        documents.insert(DocumentRecord(id='DOC_002',patient_id='P001',filename=original.name,
            original_path=str(original),file_hash='synthetic2'))
        shared.save('DOC_002','Testo clinico sintetico: tosse.',24,29,'Tosse')
        patient_deletion = PatientWorkspaceDeletionService(db,patients,workspaces_dir=workspace,cache_dir=root/'cache').delete('P001')
        results['patient_delete_shared_residues'] = {'deleted':patient_deletion.deleted,'error':patient_deletion.error,
            'shared_annotations':len(shared.annotations()),'shared_full_texts':shared.db.execute('SELECT COUNT(*) FROM annotated_source_texts').fetchone()[0]}

        # No deletion is executed: demonstrate only the unvalidated path plan.
        outsider = root/'outside-project'
        outsider.mkdir()
        patients.insert(Patient(id='../outside-project',pseudonym='SINTETICO'))
        service = PatientWorkspaceDeletionService(db,patients,workspaces_dir=workspace,cache_dir=root/'cache')
        planned = service._managed_paths('../outside-project',set(),set(),workspace/'../outside-project')
        results['patient_deletion_path_guard'] = {'external_directory_planned': any(p.resolve()==outsider.resolve() for p in planned)}
        shared.close()
        db.close()

    imports = []
    for path in sorted(Path('emr_analyzer').rglob('*.py')):
        module = '.'.join(path.with_suffix('').parts).removesuffix('.__init__')
        try:
            importlib.import_module(module)
        except Exception as exc:
            imports.append({'module':module,'error':type(exc).__name__+': '+str(exc)})
    results['application_module_imports'] = {'failures':imports}
    Path(sys.argv[1]).write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(results,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
