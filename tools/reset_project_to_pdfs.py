"""Reset a closed project to original PDFs and minimal patient/document records.

Dry run by default. --apply permanently removes derived files and clinical data.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sqlite3


def quoted(name):
    return '"' + name.replace('"', '""') + '"'


def plan(root):
    root = root.resolve(strict=True)
    db = root / "emr_registry.db"
    con = sqlite3.connect(db.as_uri() + "?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        patients = [dict(r) for r in con.execute("SELECT * FROM patients")]
        documents = [dict(r) for r in con.execute("SELECT * FROM documents")]
        versions = [dict(r) for r in con.execute("SELECT * FROM schema_version")]
        shadows = {r[1] for r in con.execute("PRAGMA table_list") if r[2] == "shadow"}
        schema = [dict(r) for r in con.execute(
            "SELECT type,name,sql FROM sqlite_master WHERE sql IS NOT NULL "
            "AND name NOT LIKE 'sqlite_%' ORDER BY CASE type WHEN 'table' THEN 0 "
            "WHEN 'index' THEN 1 ELSE 2 END") if r['name'] not in shadows]
    finally:
        con.close()
    patient_ids = {p["id"] for p in patients}
    if not patient_ids or not documents:
        raise ValueError("Progetto vuoto o non riconosciuto")
    if not all(re.fullmatch(r"[A-Za-z0-9_-]+", p) for p in patient_ids):
        raise ValueError("Identificativo paziente non sicuro")
    originals, remove, directories = {}, [], []
    for base, dirs, files in os.walk(root, followlinks=False):
        directory = Path(base)
        for name in dirs:
            child = directory / name
            if child.is_symlink():
                raise ValueError(f"Directory simbolica da verificare: {child}")
            directories.append(child)
        for name in files:
            path = directory / name
            if path.is_symlink():
                raise ValueError(f"File simbolico da verificare: {path}")
            rel = path.relative_to(root)
            is_original = (len(rel.parts) >= 4 and rel.parts[0] in patient_ids
                           and rel.parts[1:3] in (("documents", "original"), ("Documents", "Original"))
                           and path.suffix.lower() == ".pdf")
            if is_original:
                stat = path.stat()
                originals[path] = (stat.st_size, stat.st_mtime_ns, stat.st_ino)
            elif path.suffix.lower() == ".pdf":
                raise ValueError(f"PDF fuori dalle cartelle originali: {rel}; nessuna cancellazione")
            elif path != db:
                remove.append(path)
    original_by_inode = {p.stat().st_ino: p for p in originals}
    for doc in documents:
        stored = Path(doc["original_path"])
        candidates = []
        for layout in (("documents", "original"), ("Documents", "Original")):
            candidates.append(root / doc["patient_id"] / Path(*layout) / doc["filename"])
        candidates.append(stored if stored.is_absolute() else root / stored)
        source = next((original_by_inode.get(p.stat().st_ino) for p in candidates
                       if p.is_file() and p.resolve().is_relative_to(root)
                       and p.stat().st_ino in original_by_inode), None)
        if source is None or source.relative_to(root).parts[0] != doc["patient_id"]:
            raise ValueError(f"PDF originale non verificato: {doc['id']}; nessuna cancellazione")
        doc["original_path"] = str(source)
    return dict(root=root, db=db, patients=patients, documents=documents,
                versions=versions, schema=schema, originals=originals,
                remove=remove, directories=directories)


def insert_rows(con, table, rows):
    for row in rows:
        keys = list(row)
        con.execute(f"INSERT INTO {quoted(table)} ({','.join(map(quoted, keys))}) "
                    f"VALUES ({','.join('?' for _ in keys)})", [row[k] for k in keys])


def apply_reset(data):
    root, db = data["root"], data["db"]
    staging = root / "emr_registry.reset-new.db"
    if staging.exists():
        raise ValueError("Database temporaneo già presente: verificare prima di riprendere")
    now = datetime.now(timezone.utc).isoformat()
    patients = [dict(p, pseudonym=p['id'], initials=None, sex=None, birth_year=None,
                     main_pathology=None, center=None, notes=None, updated_at=now)
                for p in data["patients"]]
    documents = [dict(d, page_count=0, document_date=None, document_type="non_classificato",
                      parsing_status="pending", extraction_status="pending",
                      validation_status="pending", event_count=0, lab_value_count=0,
                      error_message=None, metadata_json=None) for d in data["documents"]]
    con = sqlite3.connect(staging)
    try:
        for item in data['schema']:
            if item['type'] == 'table':
                con.execute(item['sql'])
        insert_rows(con, 'patients', patients)
        insert_rows(con, 'documents', documents)
        insert_rows(con, 'schema_version', data['versions'])
        for item in data['schema']:
            if item['type'] != 'table':
                con.execute(item['sql'])
        con.commit()
        if con.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
            raise ValueError('Integrità del nuovo database non verificata')
        if con.execute('PRAGMA foreign_key_check').fetchall():
            raise ValueError('Collegamenti del nuovo database non validi')
    finally:
        con.close()
    # The old database must not have a live WAL when replaced.
    old = sqlite3.connect(db, timeout=2)
    try:
        if old.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchone()[0] != 0:
            raise ValueError('Database in uso; chiudere l’app')
    finally:
        old.close()
    for path, expected in data['originals'].items():
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns, stat.st_ino) != expected:
            raise ValueError('Un PDF originale è cambiato durante la preparazione')
    staging.replace(db)
    for index, path in enumerate(data['remove'], 1):
        path.unlink(missing_ok=True)
        if index % 10000 == 0:
            print(json.dumps({'removed': index, 'total': len(data['remove'])}), flush=True)
    for suffix in ('-wal', '-shm'):
        db.with_name(db.name + suffix).unlink(missing_ok=True)
    for directory in sorted(data['directories'], key=lambda p: len(p.parts), reverse=True):
        if not any(directory.iterdir()):
            directory.rmdir()
    for path, expected in data['originals'].items():
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns, stat.st_ino) != expected:
            raise ValueError('Verifica finale dei PDF non riuscita')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workspace', type=Path)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    data = plan(args.workspace)
    summary = {'patients': len(data['patients']), 'documents': len(data['documents']),
               'original_pdfs': len(data['originals']), 'derived_files_to_remove': len(data['remove'])}
    print(json.dumps(summary), flush=True)
    if args.apply:
        apply_reset(data)
        print(json.dumps(dict(summary, state='reset_complete')), flush=True)


if __name__ == '__main__':
    main()
