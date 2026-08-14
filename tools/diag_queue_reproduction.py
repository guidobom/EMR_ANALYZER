#!/usr/bin/env python
"""Diagnostic: reproduce the routing of one queue folder with the NEW code.

Re-extracts and re-routes the PDFs of a single queue folder against the real
registry (read-only), then reports every resulting group with its evidence so
the corrected workspace set can be compared against the current empty shells
(P046-P075).  No LLM calls, nothing written to the registry.
"""
from __future__ import annotations

import hashlib
import sys
import time
from collections import Counter
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
id_repo = PatientIdentityRepository(db)
router = PatientRoutingService(
    id_repo, extractor, PatientRepository(db), DocumentRepository(db)
)
known = {r["file_hash"] for r in db.execute("SELECT file_hash FROM documents").fetchall()}

files = sorted(SRC.glob("*.pdf"))
print(f"FILES: {len(files)} in {SRC}", flush=True)

t0 = time.time()
staged = []
dup = 0
sex_total = 0
sex_values = Counter()
for f in files:
    h = hashlib.sha256(f.read_bytes()).hexdigest()
    if h in known:
        dup += 1
        continue
    ev = extractor.extract(f)
    staged.append(
        StagedDocument(original_path=str(f), staged_path=str(f),
                       original_name=f.name, file_hash=h, check={}, evidence=ev)
    )
    if ev and ev.sex and ev.sex.normalized:
        sex_total += 1
        sex_values[ev.sex.normalized] += 1
print(f"EXTRACT: {len(files)} totali, {dup} gia' noti, {len(staged)} nuovi "
      f"({time.time()-t0:.0f}s)", flush=True)
print(f"SEX: {sex_total} docs con sesso ({dict(sex_values)})", flush=True)

t1 = time.time()
groups = router.resolve(staged)
print(f"ROUTE: {len(groups)} gruppi in {time.time()-t1:.0f}s", flush=True)

cat = Counter()
reason = Counter()
print("\n=== GRUPPI ===", flush=True)
for g in groups:
    ev = g.evidence
    name = ev.name.normalized if ev and ev.name else "-"
    sex = ev.sex.normalized if ev and ev.sex else "-"
    birth = ev.birth_date.normalized if ev and ev.birth_date else "-"
    cf = ev.fiscal_code.normalized if ev and ev.fiscal_code else "-"
    hpid = ev.hospital_patient_id.normalized if ev and ev.hospital_patient_id else "-"
    if g.patient_id:
        cat["matched"] += len(g.documents)
        outcome = f"MATCH {g.patient_id}"
    elif g.create_new:
        cat["create_new"] += len(g.documents)
        outcome = "CREATE_NEW"
    else:
        cat["needs_review"] += len(g.documents)
        outcome = "REVIEW"
    print(f"{outcome:22s} n={len(g.documents):5d}  {name:28s} s={sex} "
          f"nasc={birth} cf={cf} hpid={hpid}  [{g.key[:40]}]", flush=True)
    if g.needs_review:
        reason[g.reason] += len(g.documents)

print("\n=== ESITI ===", flush=True)
for k, n in cat.most_common():
    print(f"  {n:6d}  {k}", flush=True)
if reason:
    print("=== MOTIVI REVIEW ===", flush=True)
    for k, n in reason.most_common():
        print(f"  {n:6d}  {k}", flush=True)
