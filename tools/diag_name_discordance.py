#!/usr/bin/env python
"""Diagnostic: why do CODA documents land in "Da assegnare"?

Re-runs the full routing of the CODA/aa-fatti backlog against the real
registry with the CURRENT extractor (sex fix included), then dissects the
groups flagged "Valori discordanti nel gruppo: name" to characterize the
actual name variants the extractor produces for the same person.
Read-only: nothing is written to the registry.
"""
from __future__ import annotations

import hashlib
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
db = DatabaseEngine(DB)
extractor = PatientIdentityExtractor()
id_repo = PatientIdentityRepository(db)
router = PatientRoutingService(
    id_repo, extractor, PatientRepository(db), DocumentRepository(db)
)
known = {r["file_hash"] for r in db.execute("SELECT file_hash FROM documents").fetchall()}

SRC = Path("/home/utente/Desktop/DATI con CARTELLE SINGOLO PZ")
coda_files = []
for f in SRC.glob("*/*/*.pdf"):
    parts = f.parts
    idx = parts.index("DATI con CARTELLE SINGOLO PZ")
    sub = parts[idx + 2] if len(parts) > idx + 2 else ""
    if sub.upper().startswith("CODA") or sub.lower().startswith("aa fatti"):
        coda_files.append(f)

t0 = time.time()
staged = []
dup = 0
sex_values = Counter()      # normalized sex values extracted now
sex_total = 0
for f in coda_files:
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
print(f"STEP1 done: {len(coda_files)} totali, {dup} gia' nel registro, "
      f"{len(staged)} nuovi  ({time.time()-t0:.0f}s)", flush=True)
print(f"SEX estratti: {sex_total} docs (di cui {sex_values['F']} F, "
      f"{sex_values['M']} M, altro {dict(sex_values)} )", flush=True)

t1 = time.time()
groups = router.resolve(staged)
print(f"STEP2 done: {len(groups)} gruppi in {time.time()-t1:.0f}s", flush=True)

cat = Counter()
reason = Counter()
name_conflict_groups = []
for g in groups:
    n = len(g.documents)
    if g.patient_id and not g.conflict:
        cat["assegnato"] += n
    elif g.create_new:
        cat["nuovo_paziente"] += n
    elif g.needs_review:
        cat["da_assegnare"] += n
        reason[g.reason] += n
        if "name" in g.reason and "Valori discordanti" in g.reason:
            name_conflict_groups.append(g)
print("=== DOCUMENTI PER ESITO ===", flush=True)
for k, n in cat.most_common():
    print(f"  {n:6d}  {k}", flush=True)
print("=== MOTIVI 'DA ASSEGNARE' ===", flush=True)
for k, n in reason.most_common():
    print(f"  {n:6d}  {k}", flush=True)

# ---- Dissect the name-discordant groups -------------------------------
print(f"\n=== GRUPPI CONFLITTO NOME: {len(name_conflict_groups)} gruppi ===", flush=True)
size_hist = Counter()
total_docs_in = 0
for g in name_conflict_groups:
    size_hist[len(g.documents)] += 1
    total_docs_in += len(g.documents)
print(f"documenti coinvolti: {total_docs_in}", flush=True)
print("distribuzione dimensione gruppo:", dict(sorted(size_hist.items())), flush=True)

name_value_sets = Counter()
for g in name_conflict_groups:
    names = set()
    for d in g.documents:
        if d.evidence and d.evidence.name and d.evidence.name.normalized:
            names.add(d.evidence.name.normalized)
    name_value_sets[len(names)] += 1
print("distinct name VALUES per group:", dict(sorted(name_value_sets.items())), flush=True)

print("\n=== VARIANTI NOME PER GRUPPO (valore -> n documenti) ===", flush=True)
for g in sorted(name_conflict_groups, key=lambda g: len(g.documents), reverse=True):
    counter = Counter()
    for d in g.documents:
        if d.evidence and d.evidence.name and d.evidence.name.normalized:
            counter[d.evidence.name.normalized] += 1
    print(f"\n-- gruppo {g.key[:60]} ({len(g.documents)} docs, {len(counter)} varianti):", flush=True)
    for value, n in counter.most_common():
        print(f"     {n:5d}  '{value}'", flush=True)
