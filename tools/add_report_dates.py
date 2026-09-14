"""Add report dates to normalized Markdown; dry run unless --apply is given."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import signal
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emr_analyzer.clinical.report_metadata import with_report_date


def read_local_file(path):
    def expired(*_):
        raise TimeoutError("Lettura del file oltre 5 secondi")
    previous = signal.signal(signal.SIGALRM, expired)
    signal.alarm(5)
    try:
        return path.read_bytes()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    root = args.workspace.resolve()
    with sqlite3.connect((root / "emr_registry.db").as_uri() + "?mode=ro", uri=True) as con:
        rows = con.execute("SELECT id, patient_id, document_date, metadata_json FROM documents").fetchall()
    changes = []
    for index, (did, pid, value, metadata) in enumerate(rows, 1):
        if index % 1000 == 0:
            print(json.dumps({"scanned": index, "total": len(rows)}), flush=True)
        if not all(re.fullmatch(r"[A-Za-z0-9_-]+", v or "") for v in (did, pid)):
            continue
        try:
            clinical = json.loads(metadata or "{}").get("clinical_text", {})
        except (ValueError, AttributeError):
            continue
        if not isinstance(clinical, dict) or not clinical.get("deidentification_version"):
            continue
        for folder in ("extraction", "docling"):
            path = root / pid / folder / f"{did}.md"
            if not path.is_file() or not path.resolve().is_relative_to(root / pid):
                continue
            try:
                original = read_local_file(path)
            except OSError as exc:
                print(json.dumps({"skipped": str(path.relative_to(root)), "error": str(exc)}), flush=True)
                continue
            updated = with_report_date(original.decode("utf-8"), value).encode("utf-8")
            if updated != original:
                changes.append((path, original, updated))
    print(json.dumps({"files_to_update": len(changes), "apply": args.apply, "backup_bytes": sum(len(c[1]) for c in changes)}))
    if not args.apply or not changes:
        return
    backup = root / "metadata_backups" / datetime.now().strftime("report_dates_%Y%m%d_%H%M%S_%f")
    backup.mkdir(parents=True, exist_ok=False)
    journal = backup / "manifest.jsonl"
    for index, (path, original, updated) in enumerate(changes, 1):
        if index % 1000 == 0:
            print(json.dumps({"updating": index, "total": len(changes)}), flush=True)
        if read_local_file(path) != original:
            raise RuntimeError(f"File modificato durante l'aggiornamento: {path}")
        relative = path.relative_to(root)
        saved = backup / relative
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_bytes(original)
        with journal.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"file": str(relative),
                                     "original_sha256": hashlib.sha256(original).hexdigest(),
                                     "updated_sha256": hashlib.sha256(updated).hexdigest()}) + "\n")
        temporary = path.with_name(path.name + ".report-date.tmp")
        with temporary.open("xb") as stream:
            stream.write(updated)
        temporary.replace(path)
    print(json.dumps({"updated": len(changes), "backup": str(backup)}))


if __name__ == "__main__":
    main()
