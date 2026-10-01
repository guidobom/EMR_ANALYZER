# Inventario del codice — 1 ottobre 2026

Stato corrente dei file; esclusi gli strumenti creati per questa revisione. I dettagli di tutte le 2.385 funzioni/metodi, con posizione iniziale/finale e hash SHA-256 dei file, sono nel [JSON](CODE_REVIEW_2026-10-01_INVENTORY.json). Le descrizioni qui sotto derivano dalle docstring e non attestano che ogni funzione sia raggiungibile dalla GUI.

| Area | File | Righe | Funzioni/metodi |
|---|---:|---:|---:|
| . | 1 | 25 | 1 |
| emr_analyzer | 5 | 1737 | 50 |
| emr_analyzer/clinical | 55 | 18179 | 582 |
| emr_analyzer/database | 21 | 6416 | 257 |
| emr_analyzer/evaluation | 4 | 260 | 11 |
| emr_analyzer/export | 3 | 483 | 16 |
| emr_analyzer/extraction | 15 | 4037 | 103 |
| emr_analyzer/gui | 48 | 21510 | 804 |
| emr_analyzer/llm_backend | 11 | 4564 | 174 |
| emr_analyzer/models | 12 | 1699 | 47 |
| emr_analyzer/pipeline | 12 | 4704 | 164 |
| emr_analyzer/security | 3 | 229 | 12 |
| emr_analyzer/tools | 3 | 598 | 16 |
| emr_analyzer/utils | 9 | 1667 | 59 |
| tools | 15 | 2881 | 73 |
| tools/clinical_light | 3 | 330 | 16 |

## .

| File | Righe | Funzioni | Scopo dichiarato |
|---|---:|---:|---|
| [run.py](../run.py) | 25 | 1 | EMR Analyzer — Entry point for the clinical document analysis system. |

## emr_analyzer

| File | Righe | Funzioni | Scopo dichiarato |
|---|---:|---:|---|
| [__init__.py](../emr_analyzer/__init__.py) | 39 | 1 | EMR Analyzer package bootstrap. |
| [app.py](../emr_analyzer/app.py) | 376 | 9 | Application orchestrator for EMR Analyzer. |
| [config.py](../emr_analyzer/config.py) | 443 | 1 | Centralized configuration for EMR Analyzer. |
| [prompt_catalog.py](../emr_analyzer/prompt_catalog.py) | 325 | 15 | Versioned external prompt catalog for every local LLM task. |
| [settings.py](../emr_analyzer/settings.py) | 554 | 24 | Local application settings store (never contains clinical data). |

## emr_analyzer/clinical

| File | Righe | Funzioni | Scopo dichiarato |
|---|---:|---:|---|
| [__init__.py](../emr_analyzer/clinical/__init__.py) | 1 | 0 | Clinical reasoning layer. |
| [aggregate_v4.py](../emr_analyzer/clinical/aggregate_v4.py) | 267 | 9 | Fast deterministic front-end for aggregate clinical-event synthesis. |
| [clinical_history_builder.py](../emr_analyzer/clinical/clinical_history_builder.py) | 753 | 26 | Clinical history facade: delegates extraction to the FHIR event registry. |
| [compact_annotations.py](../emr_analyzer/clinical/compact_annotations.py) | 243 | 10 | Compact source annotations: the program expands provenance, never the LLM. |
| [concept_canonicalization.py](../emr_analyzer/clinical/concept_canonicalization.py) | 186 | 6 | Deterministic concept canonicalization for atomic-evidence identity. |
| [consolidation.py](../emr_analyzer/clinical/consolidation.py) | 1224 | 40 | Evidence clustering, episode reconstruction and cautious clinical fusion. |
| [correlation.py](../emr_analyzer/clinical/correlation.py) | 237 | 4 | Cautious cross-document, cross-modality clinical correlations. |
| [document_deletion.py](../emr_analyzer/clinical/document_deletion.py) | 186 | 4 | Safe deletion of a report and all data derived from it. |
| [document_reattribution.py](../emr_analyzer/clinical/document_reattribution.py) | 325 | 5 | Re-attribute a single document to another patient after human review. |
| [dossier_dedup.py](../emr_analyzer/clinical/dossier_dedup.py) | 113 | 4 | Conservative grouping of repeated observations from a single dossier. |
| [dossier_query.py](../emr_analyzer/clinical/dossier_query.py) | 287 | 8 | Exhaustive local queries over active clinical Markdown, independent of irAE. |
| [embedding_candidates.py](../emr_analyzer/clinical/embedding_candidates.py) | 202 | 6 | Embedding-based near-miss duplicate candidates for atomic evidence. |
| [episode_assembler.py](../emr_analyzer/clinical/episode_assembler.py) | 227 | 7 | Episode-first reconciliation over atomic evidence and draft events. |
| [episode_synthesis.py](../emr_analyzer/clinical/episode_synthesis.py) | 743 | 15 | LLM-assisted assembly of atomic draft events into clinical episodes. |
| [event_dedup.py](../emr_analyzer/clinical/event_dedup.py) | 727 | 26 | Deterministic content cleanup and cross-event deduplication. |
| [event_extraction.py](../emr_analyzer/clinical/event_extraction.py) | 276 | 12 | Compact source annotation followed by targeted terminology linking. |
| [event_relations.py](../emr_analyzer/clinical/event_relations.py) | 120 | 1 | Optional, independently checkpointed relations between existing source events. |
| [evidence_deletion.py](../emr_analyzer/clinical/evidence_deletion.py) | 196 | 3 | Reset of a patient's atomic evidence and everything derived from it. |
| [evidence_graph.py](../emr_analyzer/clinical/evidence_graph.py) | 1085 | 33 | Two-scale candidate generation, relation adjudication and graph splitting. |
| [evidence_relevance.py](../emr_analyzer/clinical/evidence_relevance.py) | 254 | 10 | Registry disposition of immutable atomic clinical evidence. |
| [evidence_utils.py](../emr_analyzer/clinical/evidence_utils.py) | 205 | 14 | Source identity and deduplication utilities, independent of extraction models. |
| [fhir_registry.py](../emr_analyzer/clinical/fhir_registry.py) | 294 | 17 | FHIR R4 collection generated deterministically from source-grounded occurrences. |
| [grounded_sources.py](../emr_analyzer/clinical/grounded_sources.py) | 449 | 21 | Internal source grounding, deterministic clinical dates and resumable LLM calls. |
| [historical_reuse.py](../emr_analyzer/clinical/historical_reuse.py) | 223 | 11 | Conservative exact historical-block reuse, with independently verified sources. |
| [hypothesis_discovery.py](../emr_analyzer/clinical/hypothesis_discovery.py) | 233 | 7 | Bounded, review-only discovery of hypotheses between clinical events. |
| [irae_analysis.py](../emr_analyzer/clinical/irae_analysis.py) | 119 | 6 | Full-registry immune-related adverse event (irAE) analysis. |
| [irae_corrections.py](../emr_analyzer/clinical/irae_corrections.py) | 510 | 21 | Persistent manual corrections for the structured irAE report. |
| [irae_evidence.py](../emr_analyzer/clinical/irae_evidence.py) | 77 | 3 | Read atomic evidence rows from the project registry (``emr_registry.db``). |
| [irae_export.py](../emr_analyzer/clinical/irae_export.py) | 130 | 4 | Excel export of the structured irAE analysis results. |
| [irae_layers.py](../emr_analyzer/clinical/irae_layers.py) | 1058 | 22 | Structured 3-layer irAE analysis shared by the headless tool and the GUI. |
| [irae_prototype.py](../emr_analyzer/clinical/irae_prototype.py) | 568 | 11 | Deterministic pre-filtering for immune-related adverse event (irAE) analysis. |
| [irae_reconsolidate.py](../emr_analyzer/clinical/irae_reconsolidate.py) | 116 | 3 | Re-run only the Layer 4 consolidation of a saved irAE report. |
| [irae_reports.py](../emr_analyzer/clinical/irae_reports.py) | 71 | 4 | Persistent raw structured irAE reports, one file per patient. |
| [lab_evidence.py](../emr_analyzer/clinical/lab_evidence.py) | 441 | 17 | Deterministic conversion of abnormal laboratory rows into atomic evidence. |
| [laboratory_support.py](../emr_analyzer/clinical/laboratory_support.py) | 66 | 3 | Laboratory observations support documented events; they never diagnose one. |
| [lexicon_dedup.py](../emr_analyzer/clinical/lexicon_dedup.py) | 119 | 5 | Lossless, patient-scoped projection of lexicon occurrences into dated events. |
| [lexicon_guidance.py](../emr_analyzer/clinical/lexicon_guidance.py) | 79 | 6 | Bounded example retrieval. Examples guide extraction but never constitute evidence. |
| [lexicon_structure.py](../emr_analyzer/clinical/lexicon_structure.py) | 69 | 2 | User-defined event cards; occurrence values stay separate from the catalog. |
| [loinc_catalog.py](../emr_analyzer/clinical/loinc_catalog.py) | 181 | 10 | Local official LOINC table and explicitly reviewed, specimen/unit-bound mappings. |
| [patient_deletion.py](../emr_analyzer/clinical/patient_deletion.py) | 269 | 9 | Complete, reversible deletion of one patient workspace. |
| [patient_import.py](../emr_analyzer/clinical/patient_import.py) | 278 | 5 | Import patients from another EMR Analyzer project. |
| [projections.py](../emr_analyzer/clinical/projections.py) | 573 | 12 | Deterministic laboratory, medication and oncology projections. |
| [query_context.py](../emr_analyzer/clinical/query_context.py) | 75 | 3 | Shared, conservative context budgeting for local clinical queries. |
| [query_service.py](../emr_analyzer/clinical/query_service.py) | 259 | 9 | Structured + full-text local retrieval for clinical questions. |
| [referenced_annotations.py](../emr_analyzer/clinical/referenced_annotations.py) | 198 | 6 | Source-addressed intermediate events: the LLM selects, Python quotes. |
| [registry_builder.py](../emr_analyzer/clinical/registry_builder.py) | 2082 | 47 | Orchestrator for the versioned evidence → event → episode registry. |
| [report_metadata.py](../emr_analyzer/clinical/report_metadata.py) | 31 | 3 | Deterministic report metadata, kept separate from clinical event dates. |
| [sentence_groups.py](../emr_analyzer/clinical/sentence_groups.py) | 94 | 8 | Source-preserving sentence groups. No generated summaries or clinical filtering. |
| [similar_passages.py](../emr_analyzer/clinical/similar_passages.py) | 49 | 3 | Read-only passage similarity search; scores are not clinical probabilities. |
| [snomed_catalog.py](../emr_analyzer/clinical/snomed_catalog.py) | 203 | 15 | SNOMED CT International RF2 lookup, English synonyms and inferred hierarchy. |
| [snomed_rf2.py](../emr_analyzer/clinical/snomed_rf2.py) | 132 | 3 | Transactional import of an unpacked International Edition RF2 Snapshot. |
| [temporal.py](../emr_analyzer/clinical/temporal.py) | 278 | 8 | Conservative temporal normalization for Italian clinical evidence. |
| [terminology.py](../emr_analyzer/clinical/terminology.py) | 186 | 6 | Deterministic terminology and UCUM-like unit resolution after extraction. |
| [timeline_serializer.py](../emr_analyzer/clinical/timeline_serializer.py) | 216 | 8 | Deterministic chronological registry serialized from atomic evidence. |
| [workspace_merge.py](../emr_analyzer/clinical/workspace_merge.py) | 596 | 21 | Merge one patient workspace into another within the same project. |

## emr_analyzer/database

| File | Righe | Funzioni | Scopo dichiarato |
|---|---:|---:|---|
| [__init__.py](../emr_analyzer/database/__init__.py) | 17 | 0 | Database layer for EMR Analyzer. |
| [atomic_group_repo.py](../emr_analyzer/database/atomic_group_repo.py) | 54 | 5 | Durable results and per-call metrics for independently resumable extraction. |
| [audit_repo.py](../emr_analyzer/database/audit_repo.py) | 160 | 4 | Audit log repository. |
| [chat_repo.py](../emr_analyzer/database/chat_repo.py) | 75 | 6 | Per-patient clinical query chat repository. |
| [clinical_state_repo.py](../emr_analyzer/database/clinical_state_repo.py) | 57 | 5 | Clinical State repository. |
| [document_repo.py](../emr_analyzer/database/document_repo.py) | 158 | 16 | Document repository — CRUD operations for documents table. |
| [engine.py](../emr_analyzer/database/engine.py) | 108 | 12 | SQLite database engine for EMR Analyzer. |
| [evidence_repo.py](../emr_analyzer/database/evidence_repo.py) | 257 | 14 | Persistence for immutable document-level clinical evidence. |
| [gold_set_repo.py](../emr_analyzer/database/gold_set_repo.py) | 1014 | 39 | Persistence and evaluation services for blinded clinical gold sets. |
| [lab_repo.py](../emr_analyzer/database/lab_repo.py) | 131 | 11 | Lab values repository — CRUD operations for lab_values table. |
| [local_lexicon_repo.py](../emr_analyzer/database/local_lexicon_repo.py) | 238 | 19 | Workspace-local, human-authored examples; never automatic clinical evidence. |
| [migrations.py](../emr_analyzer/database/migrations.py) | 1184 | 6 | Database schema creation and additive migrations for EMR Analyzer. |
| [overlay_repo.py](../emr_analyzer/database/overlay_repo.py) | 88 | 6 | Versioned correction overlays for immutable normalized documents. |
| [patient_identity_repo.py](../emr_analyzer/database/patient_identity_repo.py) | 313 | 13 | Persistence and matching for pseudonymized patient identity keys. |
| [patient_repo.py](../emr_analyzer/database/patient_repo.py) | 111 | 8 | Patient repository — CRUD operations for patients table. |
| [pipeline_repo.py](../emr_analyzer/database/pipeline_repo.py) | 876 | 34 | SQLite persistence for clinical-pipeline v3 derived objects. |
| [processing_repo.py](../emr_analyzer/database/processing_repo.py) | 211 | 9 | Versioned processing runs and document-stage manifest. |
| [registry_repo.py](../emr_analyzer/database/registry_repo.py) | 884 | 22 | Persistence and retrieval for the evidence-based clinical registry. |
| [review_repo.py](../emr_analyzer/database/review_repo.py) | 110 | 4 | Persistent human review decisions that survive registry rebuilds. |
| [shared_lexicon_repo.py](../emr_analyzer/database/shared_lexicon_repo.py) | 168 | 11 | One per-user terminology/example registry, independent of project databases. |
| [timeline_repo.py](../emr_analyzer/database/timeline_repo.py) | 202 | 13 | Clinical Timeline repository. |

## emr_analyzer/evaluation

| File | Righe | Funzioni | Scopo dichiarato |
|---|---:|---:|---|
| [__init__.py](../emr_analyzer/evaluation/__init__.py) | 6 | 0 | Patient-level evaluation utilities for the clinical registry. |
| [__main__.py](../emr_analyzer/evaluation/__main__.py) | 25 | 1 | Modulo senza docstring. |
| [metrics.py](../emr_analyzer/evaluation/metrics.py) | 168 | 6 | Clinically oriented metrics for event extraction and consolidation. |
| [runner.py](../emr_analyzer/evaluation/runner.py) | 61 | 4 | Run reproducible evaluation from patient-level JSONL gold records. |

## emr_analyzer/export

| File | Righe | Funzioni | Scopo dichiarato |
|---|---:|---:|---|
| [__init__.py](../emr_analyzer/export/__init__.py) | 1 | 0 | Export layer — CSV, JSON, XLSX export of clinical data. |
| [golden_set.py](../emr_analyzer/export/golden_set.py) | 125 | 4 | Golden validation set export. |
| [registry_export.py](../emr_analyzer/export/registry_export.py) | 357 | 12 | Complete, provenance-preserving exports of the clinical registry. |

## emr_analyzer/extraction

| File | Righe | Funzioni | Scopo dichiarato |
|---|---:|---:|---|
| [__init__.py](../emr_analyzer/extraction/__init__.py) | 14 | 0 | Extraction layer — information extraction from clinical text. |
| [administrative_blocks.py](../emr_analyzer/extraction/administrative_blocks.py) | 71 | 1 | Contextual administrative blocks, independent of hospital names. |
| [administrative_templates.py](../emr_analyzer/extraction/administrative_templates.py) | 184 | 4 | Administrative form fields observed in local hospital report layouts. |
| [clinical_layout.py](../emr_analyzer/extraction/clinical_layout.py) | 105 | 3 | Recover the main text column of reports with an explicit staff directory. |
| [clinical_text_filter.py](../emr_analyzer/extraction/clinical_text_filter.py) | 98 | 3 | Deterministic selection of source lines, never generative rewriting. |
| [clinical_text_isolator.py](../emr_analyzer/extraction/clinical_text_isolator.py) | 879 | 28 | Plain-text clinical isolation from deterministic PDF text. |
| [clinical_text_result.py](../emr_analyzer/extraction/clinical_text_result.py) | 22 | 1 | Shared result types without model or prompt initialization. |
| [continuous_markdown.py](../emr_analyzer/extraction/continuous_markdown.py) | 71 | 2 | Join clinical pages and keep their provenance out of the readable Markdown. |
| [document_normalization.py](../emr_analyzer/extraction/document_normalization.py) | 80 | 1 | Normalize in memory and retain deterministic page provenance. |
| [golden_fewshot.py](../emr_analyzer/extraction/golden_fewshot.py) | 111 | 3 | Golden few-shot examples for the timeline-extraction prompts. |
| [lab_parser.py](../emr_analyzer/extraction/lab_parser.py) | 1071 | 17 | Deterministic lab value parser. |
| [llm_client.py](../emr_analyzer/extraction/llm_client.py) | 889 | 29 | Client for configurable local models served by llama.cpp or vLLM. |
| [normalizer.py](../emr_analyzer/extraction/normalizer.py) | 235 | 7 | Lab parameter name and unit normalization. |
| [patterns.py](../emr_analyzer/extraction/patterns.py) | 115 | 1 | Regex patterns for Italian medical text extraction. |
| [specimen.py](../emr_analyzer/extraction/specimen.py) | 92 | 3 | Specimen (biological material) detection for laboratory reports. |

## emr_analyzer/gui

| File | Righe | Funzioni | Scopo dichiarato |
|---|---:|---:|---|
| [__init__.py](../emr_analyzer/gui/__init__.py) | 6 | 0 | GUI components for EMR Analyzer. |
| [application_shutdown.py](../emr_analyzer/gui/application_shutdown.py) | 118 | 6 | Application-wide shutdown coordination for background work. |
| [atomic_gold_widget.py](../emr_analyzer/gui/atomic_gold_widget.py) | 356 | 15 | Source-first, prediction-blinded editor for atomic-evidence gold labels. |
| [batch_import_dialog.py](../emr_analyzer/gui/batch_import_dialog.py) | 363 | 13 | Batch import dialog — one window to confirm imports for several patients. |
| [clinical_history_tab.py](../emr_analyzer/gui/clinical_history_tab.py) | 2502 | 77 | Clinical History tab — chronological timeline view with query capability. |
| [context_panel.py](../emr_analyzer/gui/context_panel.py) | 359 | 10 | Right panel showing context/details of the selected item. |
| [document_comparison_dialog.py](../emr_analyzer/gui/document_comparison_dialog.py) | 105 | 4 | Read-only, side-by-side inspection of the original PDF and active Markdown. |
| [documents_tab.py](../emr_analyzer/gui/documents_tab.py) | 1742 | 56 | Documents tab with drag-and-drop zone and document list. |
| [dossier_query_dialog.py](../emr_analyzer/gui/dossier_query_dialog.py) | 208 | 13 | Run any clinical prompt over one patient or a selected cohort. |
| [event_quick_view.py](../emr_analyzer/gui/event_quick_view.py) | 189 | 5 | Source-rich Quick View for one clinical episode. |
| [event_structure_dialog.py](../emr_analyzer/gui/event_structure_dialog.py) | 73 | 4 | Manual split editor for one evidence-backed clinical event. |
| [excluded_evidence_dialog.py](../emr_analyzer/gui/excluded_evidence_dialog.py) | 137 | 6 | Review queue and source Quick View for excluded clinical evidence. |
| [export_dialog.py](../emr_analyzer/gui/export_dialog.py) | 129 | 4 | Export dialog — select format and content for data export. |
| [gold_set_tab.py](../emr_analyzer/gui/gold_set_tab.py) | 1092 | 39 | Blinded double-annotation and adjudication workspace for clinical gold sets. |
| [hypothesis_dialog.py](../emr_analyzer/gui/hypothesis_dialog.py) | 180 | 10 | Separate discovery and human review of clinical hypotheses. |
| [import_dialog.py](../emr_analyzer/gui/import_dialog.py) | 469 | 14 | Import dialog — shows file checks before processing. |
| [import_patient_dialog.py](../emr_analyzer/gui/import_patient_dialog.py) | 202 | 11 | Dialog for importing patients from another project. |
| [irae_evidence_inspector.py](../emr_analyzer/gui/irae_evidence_inspector.py) | 509 | 22 | Evidence inspector for one selected irAE finding. |
| [irae_patient_tab.py](../emr_analyzer/gui/irae_patient_tab.py) | 161 | 12 | One patient's irAE report as a selectable, inspectable tab. |
| [irae_queue_dialog.py](../emr_analyzer/gui/irae_queue_dialog.py) | 164 | 7 | Selection dialog for the multi-patient irAE analysis queue. |
| [irae_queue_result_dialog.py](../emr_analyzer/gui/irae_queue_result_dialog.py) | 222 | 6 | Summary dialog of a multi-patient irAE analysis queue. |
| [irae_result_dialog.py](../emr_analyzer/gui/irae_result_dialog.py) | 257 | 12 | Result dialog for the full-registry irAE analysis. |
| [laboratory_tab.py](../emr_analyzer/gui/laboratory_tab.py) | 347 | 16 | Laboratory tab — lab values table with temporal charts. |
| [lexicon_analysis_worker.py](../emr_analyzer/gui/lexicon_analysis_worker.py) | 58 | 2 | Read-only LLM analysis: returns drafts, never writes examples or evidence. |
| [llm_config_dialog.py](../emr_analyzer/gui/llm_config_dialog.py) | 2471 | 64 | Configuration dialog for independent llama.cpp and vLLM roles. |
| [local_lexicon_tab.py](../emr_analyzer/gui/local_lexicon_tab.py) | 933 | 56 | Select source text, assign a short label, and curate workspace-local examples. |
| [loinc_catalog_dialog.py](../emr_analyzer/gui/loinc_catalog_dialog.py) | 72 | 8 | Official LOINC import and review of specimen/unit-specific laboratory mappings. |
| [main_window.py](../emr_analyzer/gui/main_window.py) | 916 | 33 | Main window for EMR Analyzer. |
| [merge_workspace_dialog.py](../emr_analyzer/gui/merge_workspace_dialog.py) | 321 | 14 | Dialog to merge one or more patient workspaces into other workspaces. |
| [model_manager_dialog.py](../emr_analyzer/gui/model_manager_dialog.py) | 763 | 26 | Download and selectively import local GGUF models from the LLM dialog. |
| [normalization_dialog.py](../emr_analyzer/gui/normalization_dialog.py) | 307 | 15 | Dialog listing documents that still need clinical-text normalization. |
| [patient_panel.py](../emr_analyzer/gui/patient_panel.py) | 269 | 12 | Patient panel — left sidebar for patient list and management. |
| [pdf_viewer.py](../emr_analyzer/gui/pdf_viewer.py) | 347 | 21 | PDF and normalized-text evidence preview widgets. |
| [pipeline_config_dialog.py](../emr_analyzer/gui/pipeline_config_dialog.py) | 303 | 10 | Configuration dialog for the versioned clinical-pipeline policy. |
| [pipeline_llm.py](../emr_analyzer/gui/pipeline_llm.py) | 33 | 2 | Explicit launch-time LLM selection, with independent pipeline presets. |
| [progress_dialog.py](../emr_analyzer/gui/progress_dialog.py) | 91 | 9 | Progress dialog for long-running processing tasks. |
| [project_dialog.py](../emr_analyzer/gui/project_dialog.py) | 209 | 12 | Project selection/creation dialog shown at application startup. |
| [prompt_manager_dialog.py](../emr_analyzer/gui/prompt_manager_dialog.py) | 865 | 30 | Editor and per-task version selector for external LLM prompts. |
| [qt_utils.py](../emr_analyzer/gui/qt_utils.py) | 22 | 1 | Small Qt runtime helpers shared by synchronous GUI workflows. |
| [quick_look.py](../emr_analyzer/gui/quick_look.py) | 304 | 16 | macOS-style Quick Look preview for in-app PDF inspection. |
| [registry_queue_dialog.py](../emr_analyzer/gui/registry_queue_dialog.py) | 271 | 9 | Selection dialog for sequential multi-patient registry generation. |
| [registry_queue_result_dialog.py](../emr_analyzer/gui/registry_queue_result_dialog.py) | 73 | 1 | Final summary for multi-patient chronological-registry generation. |
| [similar_passages_worker.py](../emr_analyzer/gui/similar_passages_worker.py) | 81 | 2 | Background search limited to the captured current-workspace document list. |
| [snomed_catalog_dialog.py](../emr_analyzer/gui/snomed_catalog_dialog.py) | 160 | 14 | Local RF2 International import, retrieval inspection and reviewed Italian aliases. |
| [styles.py](../emr_analyzer/gui/styles.py) | 288 | 0 | QSS stylesheets for EMR Analyzer. |
| [validation_tab.py](../emr_analyzer/gui/validation_tab.py) | 599 | 18 | Validation tab — review queue for human validation of extractions. |
| [workers.py](../emr_analyzer/gui/workers.py) | 884 | 35 | QThread workers for background processing. |
| [workspace_tabs.py](../emr_analyzer/gui/workspace_tabs.py) | 980 | 32 | Central QTabWidget workspace for EMR Analyzer. |

## emr_analyzer/llm_backend

| File | Righe | Funzioni | Scopo dichiarato |
|---|---:|---:|---|
| [__init__.py](../emr_analyzer/llm_backend/__init__.py) | 81 | 3 | Local LLM backends managed by EMR Analyzer. |
| [backend.py](../emr_analyzer/llm_backend/backend.py) | 361 | 25 | High-level llama.cpp backend consumed by :class:`LlmClient`. |
| [diagnostics.py](../emr_analyzer/llm_backend/diagnostics.py) | 589 | 11 | Non-invasive diagnostics for llama.cpp hardware acceleration. |
| [model_catalog.py](../emr_analyzer/llm_backend/model_catalog.py) | 619 | 12 | Curated, offline model catalogue for EMR Analyzer. |
| [model_download_helper.py](../emr_analyzer/llm_backend/model_download_helper.py) | 77 | 3 | Network-capable helper used only for explicit model installations. |
| [model_installer.py](../emr_analyzer/llm_backend/model_installer.py) | 960 | 32 | Install GGUF models without exposing clinical application state. |
| [model_store.py](../emr_analyzer/llm_backend/model_store.py) | 220 | 6 | Local GGUF model directory and its JSON index. |
| [server_manager.py](../emr_analyzer/llm_backend/server_manager.py) | 473 | 23 | Lifecycle management for the app's own llama-server child processes. |
| [server_runtime.py](../emr_analyzer/llm_backend/server_runtime.py) | 434 | 17 | Verified, application-managed llama-server runtime storage. |
| [vllm_backend.py](../emr_analyzer/llm_backend/vllm_backend.py) | 431 | 25 | Optional vLLM backend for Linux/CUDA systems such as DGX Spark. |
| [vllm_server_manager.py](../emr_analyzer/llm_backend/vllm_server_manager.py) | 319 | 17 | Lifecycle for optional app-managed vLLM OpenAI servers. |

## emr_analyzer/models

| File | Righe | Funzioni | Scopo dichiarato |
|---|---:|---:|---|
| [__init__.py](../emr_analyzer/models/__init__.py) | 78 | 3 | Patient and Workspace data models. |
| [chat_message.py](../emr_analyzer/models/chat_message.py) | 29 | 2 | One persisted chat message of the clinical query history. |
| [clinical_evidence.py](../emr_analyzer/models/clinical_evidence.py) | 133 | 2 | Normalized clinical evidence linked to exact document provenance. |
| [clinical_pipeline.py](../emr_analyzer/models/clinical_pipeline.py) | 187 | 7 | Versioned contracts for the end-to-end clinical pipeline v3. |
| [clinical_registry.py](../emr_analyzer/models/clinical_registry.py) | 354 | 2 | Versioned downstream models for the evidence-based clinical registry. |
| [clinical_state.py](../emr_analyzer/models/clinical_state.py) | 209 | 7 | Clinical State data models. |
| [clinical_timeline.py](../emr_analyzer/models/clinical_timeline.py) | 145 | 2 | Clinical Timeline data models — strictly temporal clinical registry. |
| [document.py](../emr_analyzer/models/document.py) | 106 | 2 | Document data models. |
| [gold_set.py](../emr_analyzer/models/gold_set.py) | 126 | 2 | Human-annotated clinical gold-set domain models. |
| [lab_result.py](../emr_analyzer/models/lab_result.py) | 154 | 10 | Lab result data models. |
| [patient_identity.py](../emr_analyzer/models/patient_identity.py) | 114 | 6 | Patient identity evidence extracted from report headers. |
| [validation.py](../emr_analyzer/models/validation.py) | 64 | 2 | Validation queue data model. |

## emr_analyzer/pipeline

| File | Righe | Funzioni | Scopo dichiarato |
|---|---:|---:|---|
| [__init__.py](../emr_analyzer/pipeline/__init__.py) | 0 | 0 | Modulo senza docstring. |
| [classifier.py](../emr_analyzer/pipeline/classifier.py) | 686 | 10 | Document classifier — determines the document type. |
| [cleaner.py](../emr_analyzer/pipeline/cleaner.py) | 632 | 12 | Text cleaner — removes boilerplate and artifacts from clinical text. |
| [converter.py](../emr_analyzer/pipeline/converter.py) | 100 | 10 | Docling standard converter. |
| [deterministic_attribution.py](../emr_analyzer/pipeline/deterministic_attribution.py) | 132 | 3 | Deterministic (non-LLM) document attribution. |
| [header_metadata.py](../emr_analyzer/pipeline/header_metadata.py) | 238 | 11 | Conservative metadata extraction from the first-page report header. |
| [import_staging.py](../emr_analyzer/pipeline/import_staging.py) | 128 | 4 | Temporary inbox used before a document is assigned to a workspace. |
| [patient_identity.py](../emr_analyzer/pipeline/patient_identity.py) | 915 | 33 | Fast, coordinate-aware patient identity extraction. |
| [patient_routing.py](../emr_analyzer/pipeline/patient_routing.py) | 630 | 24 | Resolve staged documents to existing or new patient workspaces. |
| [pdf_extractor.py](../emr_analyzer/pipeline/pdf_extractor.py) | 780 | 40 | Deterministic extraction of electronic PDFs with pdfplumber. |
| [sensitive_data.py](../emr_analyzer/pipeline/sensitive_data.py) | 388 | 13 | Deterministic removal of direct identifiers from clinical prose. |
| [staff_identity.py](../emr_analyzer/pipeline/staff_identity.py) | 75 | 4 | Redact explicitly attributed staff names without deleting clinical prose. |

## emr_analyzer/security

| File | Righe | Funzioni | Scopo dichiarato |
|---|---:|---:|---|
| [__init__.py](../emr_analyzer/security/__init__.py) | 15 | 0 | Security controls used by the local clinical application. |
| [offline.py](../emr_analyzer/security/offline.py) | 136 | 10 | Process-level offline policy for clinical processing. |
| [privacy.py](../emr_analyzer/security/privacy.py) | 78 | 2 | Privacy-safe serialization for logs and review payloads. |

## emr_analyzer/tools

| File | Righe | Funzioni | Scopo dichiarato |
|---|---:|---:|---|
| [__init__.py](../emr_analyzer/tools/__init__.py) | 1 | 0 | Maintenance tools that run outside the GUI (one-off passes, rechecks). |
| [duplicate_audit.py](../emr_analyzer/tools/duplicate_audit.py) | 401 | 9 | Misura la duplicazione delle evidenze atomiche per causa (read-only). |
| [reclassify_documents.py](../emr_analyzer/tools/reclassify_documents.py) | 196 | 7 | One-off document-type reclassification for an existing project database. |

## emr_analyzer/utils

| File | Righe | Funzioni | Scopo dichiarato |
|---|---:|---:|---|
| [__init__.py](../emr_analyzer/utils/__init__.py) | 35 | 0 | Utility functions and constants for EMR Analyzer. |
| [date_utils.py](../emr_analyzer/utils/date_utils.py) | 81 | 2 | Italian date parsing utilities. |
| [document_paths.py](../emr_analyzer/utils/document_paths.py) | 62 | 2 | Resolution of stored document paths against the current workspace. |
| [file_utils.py](../emr_analyzer/utils/file_utils.py) | 115 | 8 | File utility functions: hashing, validation, type detection. |
| [hardware.py](../emr_analyzer/utils/hardware.py) | 608 | 22 | Hardware resource detection for dynamic worker pool sizing and auto-config. |
| [markdown_tables.py](../emr_analyzer/utils/markdown_tables.py) | 115 | 5 | Markdown pipe-tables → HTML, for the chat panel (QTextBrowser). |
| [nvidia.py](../emr_analyzer/utils/nvidia.py) | 206 | 9 | Read-only NVIDIA/DGX hardware probing. |
| [slot_benchmark.py](../emr_analyzer/utils/slot_benchmark.py) | 324 | 6 | Measured llama.cpp slot tuning for the atomic-evidence workload. |
| [text_utils.py](../emr_analyzer/utils/text_utils.py) | 121 | 5 | Italian text normalization utilities. |

## tools

| File | Righe | Funzioni | Scopo dichiarato |
|---|---:|---:|---|
| [add_report_dates.py](../tools/add_report_dates.py) | 87 | 3 | Add report dates to normalized Markdown; dry run unless --apply is given. |
| [audit_code.py](../tools/audit_code.py) | 47 | 1 | Read-only Python inventory and exact-body duplicate/dead-code candidates. |
| [cleanup_empty_workspaces.py](../tools/cleanup_empty_workspaces.py) | 114 | 0 | Delete every empty patient workspace from the open project registry. |
| [clinical_light_prototype.py](../tools/clinical_light_prototype.py) | 137 | 4 | Isolated P068 experiment. Never writes patient workspace or application settings. |
| [clinical_sample_review.py](../tools/clinical_sample_review.py) | 279 | 14 | Isolated, reviewed clinical sample: prepare, run, propose review matches. |
| [correct_queue_workspaces.py](../tools/correct_queue_workspaces.py) | 150 | 2 | Correct the empty queue workspaces (P046-P075) created with the OLD extractor. |
| [export_patient_md.py](../tools/export_patient_md.py) | 290 | 9 | Export per-patient clinical texts and blood chemistry, anonymized. |
| [import_snomed_rf2.py](../tools/import_snomed_rf2.py) | 25 | 1 | Import an unpacked International RF2 Snapshot into a local SNOMED catalog. |
| [irae_headless.py](../tools/irae_headless.py) | 311 | 5 | Headless immune-related adverse event (irAE) analysis for one patient. |
| [irae_rebuild_from_xlsx.py](../tools/irae_rebuild_from_xlsx.py) | 355 | 5 | Rebuild persistent irAE reports from an Excel export + the project registry. |
| [monitor_lung_extraction.py](../tools/monitor_lung_extraction.py) | 175 | 3 | Monitor dello stadio ``atomic_evidence_v3`` sul progetto LUNG. |
| [redact_legacy_identity_payloads.py](../tools/redact_legacy_identity_payloads.py) | 206 | 6 | Audit and optionally redact legacy direct identifiers downstream. |
| [reset_project_to_pdfs.py](../tools/reset_project_to_pdfs.py) | 160 | 5 | Reset a closed project to original PDFs and minimal patient/document records. |
| [setup_llama_backend.py](../tools/setup_llama_backend.py) | 355 | 11 | One-time setup of the llama.cpp backend for EMR Analyzer. |
| [setup_vllm_backend.py](../tools/setup_vllm_backend.py) | 190 | 4 | Diagnose or explicitly install the optional vLLM backend. |

## tools/clinical_light

| File | Righe | Funzioni | Scopo dichiarato |
|---|---:|---:|---|
| [__init__.py](../tools/clinical_light/__init__.py) | 1 | 0 | Experimental extraction only. Never imported by the application. |
| [engine.py](../tools/clinical_light/engine.py) | 257 | 14 | Candidate verification + open discovery, isolated from production entry points. |
| [linking.py](../tools/clinical_light/linking.py) | 72 | 2 | Local RF2 retrieval; short English queries, candidate-index decisions, no codes generated. |

## Candidati statici da verificare

I nomi usati via Qt, proprietà, stringhe o `getattr` possono risultare falsamente inutilizzati. Non eliminare automaticamente questi simboli.

- Import candidati: 43.
- Funzioni candidate: 80.

### Import candidati

| File:riga | Nome |
|---|---|
| emr_analyzer/clinical/clinical_history_builder.py:4 | `itertools` |
| emr_analyzer/clinical/clinical_history_builder.py:6 | `threading` |
| emr_analyzer/clinical/correlation.py:9 | `display_entity` |
| emr_analyzer/clinical/event_dedup.py:37 | `is_administrative_evidence` |
| emr_analyzer/clinical/event_dedup.py:38 | `ClinicalEpisode` |
| emr_analyzer/clinical/event_extraction.py:7 | `GroundedSourceReader` |
| emr_analyzer/clinical/event_extraction.py:7 | `schema` |
| emr_analyzer/clinical/fhir_registry.py:3 | `hashlib` |
| emr_analyzer/clinical/registry_builder.py:7 | `replace` |
| emr_analyzer/clinical/registry_builder.py:16 | `locate_quote` |
| emr_analyzer/clinical/registry_builder.py:34 | `load_document_geometry` |
| emr_analyzer/database/lab_repo.py:3 | `json` |
| emr_analyzer/database/lab_repo.py:4 | `Optional` |
| emr_analyzer/extraction/lab_parser.py:9 | `LAB_TABULAR_PATTERN` |
| emr_analyzer/extraction/lab_parser.py:9 | `LAB_INLINE_PATTERN` |
| emr_analyzer/extraction/lab_parser.py:9 | `build_known_param_pattern` |
| emr_analyzer/extraction/llm_client.py:18 | `DISCHARGE_ALLOWED_CATEGORIES` |
| emr_analyzer/extraction/llm_client.py:18 | `format_examples_section` |
| emr_analyzer/gui/context_panel.py:191 | `Path` |
| emr_analyzer/gui/export_dialog.py:8 | `Qt` |
| emr_analyzer/gui/import_dialog.py:13 | `QThread` |
| emr_analyzer/gui/import_patient_dialog.py:10 | `Qt` |
| emr_analyzer/gui/irae_patient_tab.py:13 | `QFont` |
| emr_analyzer/gui/main_window.py:3 | `os` |
| emr_analyzer/gui/main_window.py:6 | `QApplication` |
| emr_analyzer/gui/pipeline_config_dialog.py:8 | `QFileDialog` |
| emr_analyzer/gui/pipeline_config_dialog.py:31 | `ClinicalPipelinePolicy` |
| emr_analyzer/gui/progress_dialog.py:7 | `Qt` |
| emr_analyzer/gui/project_dialog.py:16 | `active_workspace` |
| emr_analyzer/gui/validation_tab.py:14 | `ValidationStatus` |
| emr_analyzer/gui/validation_tab.py:14 | `Severity` |
| emr_analyzer/gui/workspace_tabs.py:5 | `QWidget` |
| emr_analyzer/gui/workspace_tabs.py:5 | `QVBoxLayout` |
| emr_analyzer/gui/workspace_tabs.py:5 | `QLabel` |
| emr_analyzer/gui/workspace_tabs.py:8 | `Qt` |
| emr_analyzer/llm_backend/model_installer.py:27 | `LLM_MODEL_INDEX_PATH` |
| emr_analyzer/llm_backend/model_store.py:19 | `LLM_MODELS_DIR` |
| emr_analyzer/models/chat_message.py:6 | `Optional` |
| emr_analyzer/pipeline/cleaner.py:7 | `Optional` |
| emr_analyzer/pipeline/converter.py:35 | `docling` |
| emr_analyzer/utils/file_utils.py:7 | `Optional` |
| tools/clinical_light_prototype.py:5 | `hashlib` |
| tools/setup_llama_backend.py:24 | `os` |

### Funzioni candidate

| File:riga | Nome |
|---|---|
| emr_analyzer/clinical/aggregate_v4.py:138 | `build_coverage_ledger` |
| emr_analyzer/clinical/clinical_history_builder.py:707 | `_get_normalized_text_path` |
| emr_analyzer/clinical/clinical_history_builder.py:711 | `_format_registry_context` |
| emr_analyzer/clinical/clinical_history_builder.py:733 | `_load_golden_examples` |
| emr_analyzer/clinical/compact_annotations.py:174 | `_recover_partial` |
| emr_analyzer/clinical/consolidation.py:152 | `consolidate` |
| emr_analyzer/clinical/dossier_dedup.py:41 | `group_findings` |
| emr_analyzer/clinical/episode_assembler.py:89 | `attach_contextual_evidence` |
| emr_analyzer/clinical/episode_synthesis.py:344 | `synthesize` |
| emr_analyzer/clinical/evidence_relevance.py:188 | `classify_nonclinical_passage` |
| emr_analyzer/clinical/evidence_relevance.py:47 | `can_anchor_episode` |
| emr_analyzer/clinical/historical_reuse.py:107 | `_load_historical_group` |
| emr_analyzer/clinical/historical_reuse.py:124 | `_save_historical_group` |
| emr_analyzer/clinical/irae_evidence.py:34 | `registry_db_path` |
| emr_analyzer/clinical/irae_prototype.py:514 | `build_layer3_prompt` |
| emr_analyzer/clinical/lab_evidence.py:57 | `is_out_of_range` |
| emr_analyzer/clinical/lab_evidence.py:67 | `abnormal_lab_evidence` |
| emr_analyzer/clinical/lexicon_guidance.py:75 | `build_guidance` |
| emr_analyzer/clinical/registry_builder.py:1310 | `rebuild_from_evidence` |
| emr_analyzer/clinical/snomed_catalog.py:62 | `synonyms` |
| emr_analyzer/clinical/snomed_catalog.py:78 | `is_descendant_of` |
| emr_analyzer/clinical/timeline_serializer.py:88 | `filter_entries` |
| emr_analyzer/database/audit_repo.py:120 | `verify_chain` |
| emr_analyzer/database/document_repo.py:52 | `get_by_hash` |
| emr_analyzer/database/document_repo.py:79 | `list_by_type` |
| emr_analyzer/database/document_repo.py:109 | `update_counts` |
| emr_analyzer/database/engine.py:97 | `vacuum` |
| emr_analyzer/database/engine.py:101 | `get_table_count` |
| emr_analyzer/database/evidence_repo.py:62 | `replace_document` |
| emr_analyzer/database/gold_set_repo.py:822 | `evaluate_project` |
| emr_analyzer/database/lab_repo.py:71 | `get_by_parameter` |
| emr_analyzer/database/lab_repo.py:81 | `get_distinct_parameters` |
| emr_analyzer/database/lab_repo.py:89 | `get_abnormal_count` |
| emr_analyzer/database/migrations.py:1153 | `drop_all_tables` |
| emr_analyzer/database/overlay_repo.py:12 | `text_hash` |
| emr_analyzer/database/overlay_repo.py:72 | `history` |
| emr_analyzer/database/pipeline_repo.py:52 | `replace_aggregation_coverage` |
| emr_analyzer/database/pipeline_repo.py:82 | `list_aggregation_coverage` |
| emr_analyzer/database/pipeline_repo.py:187 | `get_evidence_sources` |
| emr_analyzer/database/pipeline_repo.py:714 | `list_event_claims` |
| emr_analyzer/database/processing_repo.py:203 | `processed_document_ids` |
| emr_analyzer/database/review_repo.py:94 | `get_for_target` |
| emr_analyzer/database/timeline_repo.py:17 | `save_batch` |
| emr_analyzer/database/timeline_repo.py:168 | `get_next_entry_id` |
| emr_analyzer/extraction/clinical_text_isolator.py:490 | `_numbers` |
| emr_analyzer/extraction/lab_parser.py:677 | `_parse_markdown_tables` |
| emr_analyzer/extraction/lab_parser.py:880 | `_match_to_lab_value` |
| emr_analyzer/extraction/llm_client.py:510 | `last_response_text` |
| emr_analyzer/extraction/llm_client.py:517 | `extract_patient_identity` |
| emr_analyzer/gui/clinical_history_tab.py:2162 | `_on_irae_progress` |
| emr_analyzer/gui/clinical_history_tab.py:2168 | `_on_irae_finished` |
| emr_analyzer/gui/documents_tab.py:76 | `dragEnterEvent` |
| emr_analyzer/gui/documents_tab.py:84 | `dragLeaveEvent` |
| emr_analyzer/gui/documents_tab.py:90 | `dropEvent` |
| emr_analyzer/gui/documents_tab.py:1682 | `_delete_document` |
| emr_analyzer/gui/local_lexicon_tab.py:538 | `_open_example` |
| emr_analyzer/gui/main_window.py:822 | `_propagate_atomic_llm` |
| emr_analyzer/gui/main_window.py:831 | `_propagate_event_llm` |
| emr_analyzer/gui/quick_look.py:256 | `wheelEvent` |
| emr_analyzer/gui/workspace_tabs.py:758 | `_run_irae_classic_queue` |
| emr_analyzer/llm_backend/model_catalog.py:385 | `builtin_catalog_manifest` |
| emr_analyzer/llm_backend/model_installer.py:111 | `search_huggingface_gguf` |
| emr_analyzer/llm_backend/model_installer.py:178 | `list_huggingface_gguf_files` |
| emr_analyzer/models/lab_result.py:104 | `baseline` |
| emr_analyzer/models/lab_result.py:111 | `min_value` |
| emr_analyzer/models/lab_result.py:118 | `max_value` |
| emr_analyzer/models/lab_result.py:125 | `absolute_change` |
| emr_analyzer/models/lab_result.py:134 | `percent_change` |
| emr_analyzer/models/patient_identity.py:106 | `masked_label` |
| emr_analyzer/pipeline/classifier.py:616 | `get_all_candidates` |
| emr_analyzer/pipeline/classifier.py:681 | `is_pre_acute_discharge` |
| emr_analyzer/pipeline/header_metadata.py:26 | `classification_text` |
| emr_analyzer/pipeline/patient_identity.py:158 | `decode_sex_from_cf` |
| emr_analyzer/pipeline/patient_routing.py:93 | `bootstrap_existing_identities` |
| emr_analyzer/pipeline/sensitive_data.py:197 | `sanitize_payload` |
| emr_analyzer/prompt_catalog.py:320 | `available_prompts` |
| emr_analyzer/settings.py:314 | `load_model_assignments` |
| emr_analyzer/settings.py:358 | `save_model_assignment` |
| emr_analyzer/utils/markdown_tables.py:49 | `markdown_tables_to_html` |
| emr_analyzer/utils/nvidia.py:37 | `cuda_architecture` |
