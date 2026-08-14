#!/usr/bin/env python
"""Diagnostic: name variants behind duplicate workspaces.

Re-routes one queue folder with the NEW code and prints, for every resulting
group, the distinct name/birth values extracted from its documents.  Two
groups that describe the same person but did NOT merge show up as the same
birth date across groups — that split is the root cause of "one patient,
two folders".  Read-only, no LLM.
"""
from __future__ import annotations

import hashlib
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from emr_analyzer.database.document_repo import DocumentRepository
from emr_analyzer.database.engine import DatabaseEngine
from emr_analyzer.database.patient_identity_repo import PatientIdentityRepository
from emr_analyzer.database.patient_repo import PatientRepository
from emr_analyzer.pipeline.import_staging import StagedDocument
from emr_analyzer.pipeline.patient_identity import PatientIdentityExtractor
from emr_analyzer.pipeline.patient_routing import PatientRoutingService

DB = Path("/home/utente/Desktop/PietroB/emr_registry.db")
SRC = Path(sys.argv[1] if len(sys.argv) > 1 else
           "/home/utente/Desktop/DATI con CARTELLE SINGOLO PZ/MELANOMA/CODA MASSARENTI A  ZORZAN M")

db = DatabaseEngine(DB)
extractor = PatientIdentityExtractor()
router = PatientRoutingService(
    id_repo := PatientIdentityRepository(db), extractor,
    PatientRepository(db), DocumentRepository(db),
)
known = {r["file_hash"] for r in db.execute("SELECT file_hash FROM documents").fetchall()}

t0 = time.time()
staged = []
for f in sorted(SRC.glob("*.pdf")):
    h = hashlib.sha256(f.read_bytes()).hexdigest()
    if h in known:
        continue
    ev = extractor.extract(f)
    staged.append(
        StagedDocument(original_path=str(f), staged_path=str(f),
                       original_name=f.name, file_hash=h, check={}, evidence=ev)
    )
print(f"extract: {len(staged)} staged di {SRC} ({time.time()-t0:.0f}s)", flush=True)

t1 = time.time()
groups = router.resolve(staged)
print(f"routing: {len(groups)} groups ({time.time()-t1:.0f}s)\n", flush=True)

# birth date -> list of (group_label, name) — groups sharing birth are the split candidates
by_birth: dict[str, list[tuple[str, str]]] = defaultdict(list)
for i, g in enumerate(groups, 1):
    names = Counter()
    births = Counter()
    for d in g.documents:
        if d.evidence and d.evidence.name:
            names[d.evidence.name.normalized] += 1
        if d.evidence and d.evidence.birth_date:
            births[d.evidence.birth_date.normalized] += 1
    label = f"G{i}({len(g.documents)}doc)"
    top_name = names.most_common(1)[0] if names else ("-", 0)
    print(f"{label:16s} nome[{top_name[1]}]: '{top_name[0]}'  birth: {dict(births)}",
          flush=True)
    for b, _n in births.items():
        by_birth[b].append((label, top_name[0]))

print("\n=== GRUPPI CHE CONDIVIDONO LA DATA DI NASCITA (possibili doppioni) ===", flush=True)
for b, items in sorted(by_birth.items()):
    if len(items) > 1:
        print(f"  nascita {b}  <- {items}", flush=True)
