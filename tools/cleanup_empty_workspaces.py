#!/usr/bin/env python
"""Delete every empty patient workspace from the open project registry.

"Empty" = a patient row with zero documents (no files imported).  Such
workspaces are pre-created shells that either never received files or whose
files now route elsewhere; removing them gives a clean slate so a new batch
can re-route from scratch.  All dependent rows (identities, hospital IDs,
audit, validation, clinical tables) are removed too.

Read-only dry-run by default; pass --apply to write after backing up the DB.
"""
from __future__ import annotations

import shutil
import sys
from datetime import datetime
from pathlib import Path

from emr_analyzer.database.engine import DatabaseEngine

APPLY = "--apply" in sys.argv
DB = Path("/home/utente/Desktop/PietroB/emr_registry.db")

# Tables with a patient_id column; the documents table is what defines
# "empty" so it is checked, not deleted here.
PATIENT_TABLES = [
    "clinical_state_checkpoints",   # no patient_id: cleaned via run_ids below
    "clinical_state_deltas",
    "clinical_state_runs",
    "document_clinical_projections",
    "validation_queue",
    "audit_log",
    "patient_identities",
    "patient_hospital_ids",
    "longitudinal_evidence",
    "clinical_evidence",
    "clinical_timeline",
    "lab_values",
    "clinical_events",
    "clinical_state",
    "document_identity_evidence",
]

db = DatabaseEngine(DB)
empty = db.execute(
    """SELECT p.id FROM patients p
       WHERE (SELECT COUNT(*) FROM documents d WHERE d.patient_id = p.id) = 0
       ORDER BY p.id"""
).fetchall()
pids = [r["id"] for r in empty]
print(f"Workspace vuoti (0 documenti): {len(pids)}", flush=True)
for pid in pids:
    print(f"  {pid}", flush=True)
if not pids:
    print("Nessun workspace vuoto: niente da fare.", flush=True)
    sys.exit(0)

# Show how many rows exist per dependent table, to keep the report honest.
print("\nRighe dipendenti da rimuovere:", flush=True)
total = 0
for table in PATIENT_TABLES:
    if table == "clinical_state_checkpoints":
        continue
    n = db.execute(
        f"SELECT COUNT(*) AS n FROM {table} WHERE patient_id IN "
        f"({','.join('?' * len(pids))})", pids
    ).fetchone()["n"]
    if n:
        print(f"  {table}: {n}", flush=True)
        total += n
# checkpoints via run_ids of the deleted patients
runs = db.execute(
    f"SELECT run_id FROM clinical_state_runs WHERE patient_id IN "
    f"({','.join('?' * len(pids))})", pids
).fetchall()
run_ids = [r["run_id"] for r in runs]
if run_ids:
    n = db.execute(
        f"SELECT COUNT(*) AS n FROM clinical_state_checkpoints WHERE run_id IN "
        f"({','.join('?' * len(run_ids))})", run_ids
    ).fetchone()["n"]
    if n:
        print(f"  clinical_state_checkpoints: {n}", flush=True)
        total += n
print(f"  TOTALE righe dipendenti: {total}", flush=True)

if not APPLY:
    print("\nDRY-RUN: nessuna modifica. Rilanciare con --apply per scrivere "
          "(backup incluso, richiede app chiusa).", flush=True)
    sys.exit(0)

backup_path = DB.parent / f"backup_emr_registry_empty_{datetime.now():%Y%m%d_%H%M%S}.db"
shutil.copy2(DB, backup_path)
print(f"\nBACKUP: {backup_path}", flush=True)

con = db.connection
con.execute("BEGIN")
for table in PATIENT_TABLES:
    if table == "clinical_state_checkpoints":
        continue
    con.execute(
        f"DELETE FROM {table} WHERE patient_id IN ({','.join('?' * len(pids))})",
        pids,
    )
if run_ids:
    con.execute(
        f"DELETE FROM clinical_state_checkpoints WHERE run_id IN "
        f"({','.join('?' * len(run_ids))})", run_ids,
    )
con.execute(
    f"DELETE FROM patients WHERE id IN ({','.join('?' * len(pids))})", pids,
)
con.commit()
print(f"APPLICATO: {len(pids)} workspace vuoti eliminati.", flush=True)
