"""One-off document-type reclassification for an existing project database.

Usage::

    python -m emr_analyzer.tools.reclassify_documents --project /path/to/MELANOMA
    python -m emr_analyzer.tools.reclassify_documents --project /path/to/MELANOMA --apply

Runs the current :class:`DocumentClassifier` over every document with the
same inputs the extraction pass uses (raw source text, filename, stored
header metadata) and writes a preview CSV of the proposed changes.  With
``--apply`` the database is backed up first and the changes are applied,
one ``audit_log`` row per document.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

from ..database.engine import DatabaseEngine
from ..pipeline.classifier import DocumentClassifier

_PAGE_MARKER = re.compile(r"^--- PAGINA \d+ ---\s*", re.MULTILINE)


def _source_text(project: Path, document: dict) -> str:
    from ..pipeline.pdf_extractor import PdfPlumberExtractor
    from ..utils.document_paths import resolve_document_path
    parser = PdfPlumberExtractor()
    result = parser.convert(resolve_document_path(document, project))
    return _PAGE_MARKER.sub("", parser.export_text(result))


def _header_metadata(metadata_json: str | None) -> dict:
    if not metadata_json:
        return {}
    try:
        return json.loads(metadata_json).get("header") or {}
    except (TypeError, ValueError):
        return {}


def _preview_rows(project: Path, db: DatabaseEngine,
                  classifier: DocumentClassifier) -> list[dict]:
    rows = db.execute(
        "SELECT id, patient_id, filename, original_path, document_type, metadata_json "
        "FROM documents ORDER BY id"
    ).fetchall()
    changes = []
    for row in rows:
        text = _source_text(project, dict(row))
        new_type = classifier.classify(
            text,
            row["filename"],
            header_metadata=_header_metadata(row["metadata_json"]),
        )
        if new_type == "non_classificato" or new_type == row["document_type"]:
            continue
        head = " ".join(text.split())[:160]
        changes.append({
            "id": row["id"],
            "patient_id": row["patient_id"],
            "old_type": row["document_type"],
            "new_type": new_type,
            "filename": row["filename"],
            "text_head": head,
        })
    return changes


def _write_preview(project: Path, changes: list[dict]) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = project / f"reclassifica_preview_{stamp}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(changes[0].keys()))
        writer.writeheader()
        writer.writerows(changes)
    return path


def _backup_db(db_path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = db_path.with_name(f"{db_path.name}.bak-{stamp}-reclassifica")
    source = sqlite3.connect(db_path)
    try:
        target = sqlite3.connect(backup)
        try:
            source.backup(target)
        finally:
            target.close()
    finally:
        source.close()
    return backup


def _apply(db: DatabaseEngine, changes: list[dict]) -> None:
    now = datetime.now().isoformat()
    for change in changes:
        db.execute(
            "UPDATE documents SET document_type=? WHERE id=?",
            (change["new_type"], change["id"]),
        )
        db.execute(
            """INSERT INTO audit_log
               (patient_id, action, target_type, target_id, details_json,
                actor_id, actor_role, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                change["patient_id"],
                "reclassify_document_type",
                "document",
                change["id"],
                json.dumps({
                    "old_type": change["old_type"],
                    "new_type": change["new_type"],
                    "reason": "riclassificazione deterministica (imaging)",
                }),
                "system",
                "system",
                now,
            ),
        )
    db.commit()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Riclassifica i tipi documento di un progetto con il "
                    "classificatore corrente."
    )
    parser.add_argument("--project", required=True, type=Path,
                        help="Directory del progetto (contiene emr_registry.db)")
    parser.add_argument("--apply", action="store_true",
                        help="Applica le modifiche dopo il backup del database")
    parser.add_argument("--from-type", dest="from_type", default="",
                        help="Filtra per tipo documento attuale "
                             "(valori separati da virgola)")
    parser.add_argument("--to-type", dest="to_type", default="",
                        help="Filtra per tipo documento proposto "
                             "(valori separati da virgola)")
    args = parser.parse_args(argv)

    db_path = args.project / "emr_registry.db"
    if not db_path.is_file():
        parser.error(f"database non trovato: {db_path}")
    from_types = {value.strip() for value in args.from_type.split(",") if value.strip()}
    to_types = {value.strip() for value in args.to_type.split(",") if value.strip()}
    db = DatabaseEngine(db_path)
    try:
        changes = _preview_rows(args.project, db, DocumentClassifier())
    finally:
        db.close()

    if from_types:
        changes = [c for c in changes if c["old_type"] in from_types]
    if to_types:
        changes = [c for c in changes if c["new_type"] in to_types]

    if not changes:
        print("Nessuna modifica proposta: tutti i tipi restano invariati.")
        return 0

    by_move: dict[tuple[str, str], int] = {}
    for change in changes:
        key = (change["old_type"], change["new_type"])
        by_move[key] = by_move.get(key, 0) + 1
    print(f"{len(changes)} documenti da riclassificare:")
    for (old, new), count in sorted(by_move.items()):
        print(f"  {old} -> {new}: {count}")

    preview_path = _write_preview(args.project, changes)
    print(f"Preview: {preview_path.name}")

    if args.apply:
        backup = _backup_db(db_path)
        print(f"Backup creato: {backup.name}")
        db = DatabaseEngine(db_path)
        try:
            _apply(db, changes)
        finally:
            db.close()
        print(f"Applicate {len(changes)} modifiche con righe di audit_log.")
    else:
        print("Modalità preview: nessuna modifica al database "
              "(usa --apply per applicare).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
