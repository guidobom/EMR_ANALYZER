"""Audit and optionally redact legacy direct identifiers downstream.

The command is dry-run by default. ``--apply`` first creates a SQLite backup
and a ZIP archive of every parser JSON that is about to change. Normalized
Markdown/text and laboratory rows are never opened for writing.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sqlite3
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emr_analyzer.pipeline.sensitive_data import SensitiveDataSanitizer
from emr_analyzer.security.privacy import identity_metadata


AUDIT_TRIGGER = """CREATE TRIGGER IF NOT EXISTS audit_log_no_update
BEFORE UPDATE ON audit_log BEGIN
  SELECT RAISE(ABORT, 'audit_log is append-only');
END"""


_IDENTITY_KEYS = {"identity", "llm_identity", "patient_identity", "raw_identity"}
_DIRECT_KEYS = {
    "name", "full_name", "patient_name", "surname", "given_name",
    "fiscal_code", "codice_fiscale", "birth_date", "date_of_birth",
    "hospital_patient_id", "email", "phone", "address",
}


def _strict_sanitize(value, *, key: str = ""):
    """Redact only detected identifiers; retain unrelated whitespace/text."""
    lowered = str(key or "").casefold()
    if lowered in _IDENTITY_KEYS:
        return identity_metadata(value)
    if lowered in _DIRECT_KEYS:
        return "[DATO IDENTIFICATIVO RIMOSSO]"
    if isinstance(value, dict):
        return {
            str(child_key): _strict_sanitize(child, key=str(child_key))
            for child_key, child in value.items()
        }
    if isinstance(value, list):
        return [_strict_sanitize(child) for child in value]
    if isinstance(value, str):
        result = SensitiveDataSanitizer().sanitize(value)
        return result.text if result.counts else value
    return value


def _sanitize_json_text(value: str | None) -> str | None:
    if not value:
        return value
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return SensitiveDataSanitizer().sanitize(str(value)).text
    sanitized = _strict_sanitize(parsed)
    if sanitized == parsed:
        return value
    return json.dumps(sanitized, ensure_ascii=False, sort_keys=True)


def inspect_database(path: Path) -> tuple[list[tuple], list[tuple]]:
    audit_changes, validation_changes = [], []
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        tables = {
            row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if "audit_log" in tables:
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(audit_log)")
            }
            hash_filter = "AND entry_hash IS NULL" if "entry_hash" in columns else ""
            for row in connection.execute(
                f"SELECT id, details_json FROM audit_log "
                f"WHERE details_json IS NOT NULL {hash_filter}"
            ):
                sanitized = _sanitize_json_text(row["details_json"])
                if sanitized != row["details_json"]:
                    audit_changes.append((sanitized, row["id"]))
        if "validation_queue" in tables:
            for row in connection.execute(
                """SELECT id, original_value, corrected_value
                   FROM validation_queue
                   WHERE item_type='attribution' OR original_value LIKE '%identity%'"""
            ):
                original = _sanitize_json_text(row["original_value"])
                corrected = _sanitize_json_text(row["corrected_value"])
                if (original, corrected) != (
                    row["original_value"], row["corrected_value"]
                ):
                    validation_changes.append((original, corrected, row["id"]))
    finally:
        connection.close()
    return audit_changes, validation_changes


def inspect_parser_json(workspace: Path) -> list[tuple[Path, dict]]:
    changes = []
    for path in workspace.glob("*/extraction/*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        sanitized = _strict_sanitize(payload)
        if sanitized != payload:
            changes.append((path, sanitized))
    return changes


def apply_changes(
    database: Path,
    workspace: Path,
    audit_changes: list[tuple],
    validation_changes: list[tuple],
    json_changes: list[tuple[Path, dict]],
) -> tuple[Path, Path | None]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = database.with_name(f"{database.name}.pre_redaction_{stamp}.bak")
    source = sqlite3.connect(database)
    destination = sqlite3.connect(backup)
    try:
        source.backup(destination)
    finally:
        destination.close()
        source.close()

    archive_path = None
    if json_changes:
        archive_path = workspace / f"parser_json_pre_redaction_{stamp}.zip"
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for path, _ in json_changes:
                archive.write(path, path.relative_to(workspace))

    connection = sqlite3.connect(database)
    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("DROP TRIGGER IF EXISTS audit_log_no_update")
        connection.executemany(
            "UPDATE audit_log SET details_json=? WHERE id=?", audit_changes
        )
        connection.executemany(
            """UPDATE validation_queue
               SET original_value=?, corrected_value=? WHERE id=?""",
            validation_changes,
        )
        connection.execute(AUDIT_TRIGGER)
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

    for path, payload in json_changes:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return backup, archive_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    workspace = args.workspace.expanduser().resolve()
    database = (args.database or workspace / "emr_registry.db").resolve()
    if not workspace.is_dir() or not database.is_file():
        parser.error("workspace o database non trovato")
    audit, validation = inspect_database(database)
    parser_json = inspect_parser_json(workspace)
    report = {
        "mode": "apply" if args.apply else "dry-run",
        "database": str(database),
        "legacy_audit_rows": len(audit),
        "validation_rows": len(validation),
        "parser_json_files": len(parser_json),
        "normalized_texts_modified": 0,
        "lab_rows_modified": 0,
    }
    if args.apply:
        backup, archive = apply_changes(
            database, workspace, audit, validation, parser_json
        )
        report["database_backup"] = str(backup)
        report["parser_json_backup"] = str(archive) if archive else None
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
