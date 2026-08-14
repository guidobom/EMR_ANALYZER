#!/usr/bin/env python
"""Correct the empty queue workspaces (P046-P075) created with the OLD extractor.

The queue folder was routed before the extractor fixes (sex spelled out,
name contamination), so the 30 workspaces it pre-created carry incomplete
or wrong initials/sex/birth_year and registered identities that no longer
match the cleaned extraction.  Re-routing the same files with the NEW code
shows which workspace each group now resolves to:

- workspaces still matched (CF/hpid unchanged) get their patients row
  refreshed from the corrected evidence (majority vote across the group);
- workspaces no longer matched by any group are empty orphans whose files
  would create fresh workspaces on the next run: they are deleted.

Read-only dry-run by default; pass --apply to write (after backing up the
registry DB).  No LLM calls.
"""
from __future__ import annotations

import hashlib
import shutil
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from emr_analyzer.database.document_repo import DocumentRepository
from emr_analyzer.database.engine import DatabaseEngine
from emr_analyzer.database.patient_identity_repo import PatientIdentityRepository
from emr_analyzer.database.patient_repo import PatientRepository
from emr_analyzer.pipeline.import_staging import StagedDocument
from emr_analyzer.pipeline.patient_identity import PatientIdentityExtractor
from emr_analyzer.pipeline.patient_routing import PatientRoutingService

APPLY = "--apply" in sys.argv
DB = Path("/home/utente/Desktop/PietroB/emr_registry.db")
SRC = Path("/home/utente/Desktop/DATI con CARTELLE SINGOLO PZ/MELANOMA/CODA MASSARENTI A  ZORZAN M")

SHELLS = {f"P0{i}" for i in range(46, 76)}  # P046..P075

db = DatabaseEngine(DB)
extractor = PatientIdentityExtractor()
id_repo = PatientIdentityRepository(db)
router = PatientRoutingService(
    id_repo, extractor, PatientRepository(db), DocumentRepository(db)
)
known = {r["file_hash"] for r in db.execute("SELECT file_hash FROM documents").fetchall()}

t0 = time.time()
files = sorted(SRC.glob("*.pdf"))
staged = []
for f in files:
    h = hashlib.sha256(f.read_bytes()).hexdigest()
    if h in known:
        continue
    ev = extractor.extract(f)
    staged.append(
        StagedDocument(original_path=str(f), staged_path=str(f),
                       original_name=f.name, file_hash=h, check={}, evidence=ev)
    )
print(f"extract+route: {len(files)} files -> {len(staged)} staged "
      f"({time.time()-t0:.0f}s)", flush=True)
t1 = time.time()
groups = router.resolve(staged)
print(f"routing: {len(groups)} groups ({time.time()-t1:.0f}s)", flush=True)

# ---- map: shell -> docs from every group that matched it ------------------
shell_docs: dict[str, list] = defaultdict(list)
matched_shells: set[str] = set()
review_docs = 0
create_new_docs = 0
for g in groups:
    if g.patient_id:
        matched_shells.add(g.patient_id)
        if g.patient_id in SHELLS:
            shell_docs[g.patient_id].extend(g.documents)
    elif g.create_new:
        create_new_docs += len(g.documents)
    else:
        review_docs += len(g.documents)


def _majority(entries, transform):
    counts = Counter()
    for entry in entries:
        value = transform(entry)
        if value:
            counts[value] += 1
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def _initials_from(name: str | None) -> str | None:
    if not name:
        return None
    return "".join(p[0].upper() for p in name.split() if p)[:4]


updates = []   # (shell, initials, sex, birth_year, ndocs)
for shell in sorted(SHELLS):
    docs = shell_docs.get(shell)
    if not docs:
        continue
    name = _majority(docs, lambda d: d.evidence.name.normalized if d.evidence and d.evidence.name else None)
    sex = _majority(docs, lambda d: d.evidence.sex.normalized if d.evidence and d.evidence.sex else None)
    birth = _majority(docs, lambda d: d.evidence.birth_date.normalized[:4] if d.evidence and d.evidence.birth_date else None)
    birth_year = int(birth) if birth and birth.isdigit() else None
    updates.append((shell, _initials_from(name), sex, birth_year, len(docs)))

deletes = sorted(SHELLS - matched_shells)

print("\n=== PIANO CORREZIONE ===", flush=True)
print(f"gruppi: {len(groups)} | doc matched-schermi: {sum(len(v) for v in shell_docs.values())} | "
      f"create_new: {create_new_docs} | review: {review_docs}", flush=True)
print(f"\n-- AGGIORNA {len(updates)} workspace (nomi/sesso/anno ricalcolati dal gruppo): --", flush=True)
for shell, initials, sex, birth_year, ndocs in updates:
    print(f"  {shell:5s} -> iniziali={initials!s:6s} sesso={sex!s:2s} anno={birth_year!s:5s} "
          f"({ndocs} doc del gruppo)", flush=True)
print(f"\n-- ELIMINA {len(deletes)} workspace orfani (nessun gruppo li usa piu'): --", flush=True)
for shell in deletes:
    print(f"  {shell:5s} (0 documenti, nessuna cartella su disco)", flush=True)

if not APPLY:
    print("\nDRY-RUN: nessuna modifica applicata. Rilanciare con --apply per scrivere "
          "(richiede app chiusa: backup incluso).", flush=True)
    sys.exit(0)

# ---- apply ---------------------------------------------------------------
backup_path = DB.parent / f"backup_emr_registry_queuefix_{datetime.now():%Y%m%d_%H%M%S}.db"
shutil.copy2(DB, backup_path)
print(f"\nBACKUP: {backup_path}", flush=True)

con = db.connection
for shell, initials, sex, birth_year, _ndocs in updates:
    con.execute(
        "UPDATE patients SET initials=?, sex=?, birth_year=?, updated_at=? WHERE id=?",
        (initials, sex, birth_year, datetime.now().isoformat(), shell),
    )
    con.execute(
        "UPDATE patient_identities SET sex=? WHERE patient_id=?",
        (sex, shell),
    )
for shell in deletes:
    con.execute("DELETE FROM patient_identities WHERE patient_id=?", (shell,))
    con.execute("DELETE FROM patient_hospital_ids WHERE patient_id=?", (shell,))
    con.execute("DELETE FROM patients WHERE id=?", (shell,))
con.commit()
print(f"APPLICATO: {len(updates)} workspace aggiornati, {len(deletes)} eliminati.", flush=True)
