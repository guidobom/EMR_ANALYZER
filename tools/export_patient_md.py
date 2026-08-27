#!/usr/bin/env python3
"""Export per-patient clinical texts and blood chemistry, anonymized.

For each project (MELANOMA, RENE, LUNG) on the Desktop this script builds a
``md <PROJECT>`` folder and, inside it, one folder per patient (the patient
pseudonym, e.g. ``P070``) containing:

- every normalized clinical text (``extraction/DOC_XXXX.md``), copied with an
  extra anonymization pass that redacts residual person names (doctor names
  and the like) to ``[NOME RIMOSSO]``;
- ``esami_ematochimici.md`` — the blood chemistry from the project registry,
  grouped by sample date.

Usage::

    python tools/export_patient_md.py                    # all three projects
    python tools/export_patient_md.py --projects MELANOMA
    python tools/export_patient_md.py --verify-only      # rescan an existing export

The patient code (folder name) is the pseudonym already used across the app;
the export never leaves the local disk.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

DEFAULT_PROJECTS_ROOT = Path("/home/utente/Desktop")
DEFAULT_OUTPUT_ROOT = Path("/home/utente/Desktop")

PROJECTS = ("MELANOMA", "RENE", "LUNG")

# -- Anonymization -----------------------------------------------------------
# Redact person names that follow a title, keeping the title.  IGNORECASE so
# all-caps medical letters are handled too; the name part still must start
# with a capital letter, which avoids prose false positives ("il dottore dice"
# stays untouched, "il dottor Bianchi" is redacted).
_NAME_RE = re.compile(
    r"(?P<title>(?i:\b(?:"
    r"Dott(?:or)?(?:e|\.)?|Dott(?:or)?(?:e|\.)?ssa|"
    r"Dr\.?|"
    r"Prof(?:essor)?(?:e|\.)?|Prof(?:essor)?(?:e|\.)?ssa|"
    r"Sig(?:\.|nor)?(?:a)?"
    r")\s+))"
    # Names: optional initials ("A.", "P.L.") then one or two capitalised
    # tokens ("Borghi", "DE GIORGI").  Capital letters stay case-sensitive so
    # prose ("il dottore dice") is never touched.
    r"(?P<names>(?:[A-ZÀ-Ý]\.(?:[A-ZÀ-Ý]\.)*\s*)?"
    r"[A-ZÀ-Ý][\wÀ-ÿ'’\-]*(?:\s+[A-ZÀ-Ý][\wÀ-ÿ'’\-]*){0,1})",
)

# Known redaction placeholders the pipeline already emits.
_KNOWN_PLACEHOLDERS = {
    "[PAZIENTE]",
    "[TELEFONO RIMOSSO]",
    "[INDIRIZZO RIMOSSO]",
    "[NOME RIMOSSO]",
    "[DATA RIMOSSA]",
    "[NUMERO RIMOSSO]",
}

# Residual-PII scanners, used both on source and on exported files.
_PII_PATTERNS = {
    "nomi_medici": _NAME_RE,
    "codice_fiscale": re.compile(r"\b[A-Z]{6}\d{2}[A-Z]\d{2}[A-Z]\d{3}[A-Z]\b"),
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"),
}


def redact_pii(text: str) -> str:
    """Replace person names after a title with ``[NOME RIMOSSO]``."""
    return _NAME_RE.sub(r"\g<title>[NOME RIMOSSO]", text)


def scan_pii(text: str) -> dict[str, list[str]]:
    """Return matched PII per category (for verification)."""
    hits: dict[str, list[str]] = {}
    for name, pattern in _PII_PATTERNS.items():
        found = [match.group(0) for match in pattern.finditer(text)]
        if found:
            hits[name] = found
    return hits


# -- Registry access ---------------------------------------------------------
def _normalized_md_files(patient_dir: Path) -> list[Path]:
    extraction = patient_dir / "extraction"
    if not extraction.exists():
        return []
    files = []
    for path in sorted(extraction.glob("DOC_*.md")):
        name = path.name
        if name.endswith("_raw.md") or name.endswith("_cleaned_source.md"):
            continue
        files.append(path)
    return files


def _lab_markdown(db_path: Path, patient_id: str) -> str:
    """Build the blood-chemistry markdown for one patient."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            "SELECT parameter_name, value, value_text, operator, unit, "
            "       reference_low, reference_high, reference_text, "
            "       is_abnormal, flag, sample_date "
            "FROM lab_values WHERE patient_id = ? "
            "ORDER BY sample_date, parameter_name",
            (patient_id,),
        ).fetchall()
    finally:
        con.close()

    if not rows:
        return ""

    lines = [f"# Esami ematochimici — {patient_id}", ""]
    by_date: dict[str, list[sqlite3.Row]] = {}
    for row in rows:
        by_date.setdefault(row["sample_date"] or "data non specificata", []).append(row)

    for date_label, group in by_date.items():
        lines.append(f"## {date_label}")
        lines.append("")
        lines.append("| Parametro | Valore | Unità | Riferimento | Anomalia |")
        lines.append("|---|---|---|---|---|")
        for row in group:
            value = _format_lab_value(row)
            unit = _escape_cell(row["unit"] or "")
            reference = _escape_cell(row["reference_text"] or "")
            anomaly = "⚠" if row["is_abnormal"] else ""
            parameter = redact_pii(row["parameter_name"] or "")
            lines.append(
                f"| {_escape_cell(parameter)} | {value} "
                f"| {unit} | {reference} | {anomaly} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _format_lab_value(row) -> str:
    operator = row["operator"] or ""
    if row["value"] is not None:
        raw = f"{row['value']:g}"
    elif row["value_text"]:
        raw = redact_pii(row["value_text"])
    else:
        raw = "—"
    return f"{operator}{raw}" if operator else raw


def _escape_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


# -- Export ------------------------------------------------------------------
def export_patient(
    project_dir: Path, output_root: Path, patient_id: str
) -> dict:
    """Copy anonymized md texts and write blood chemistry for one patient."""
    patient_dir = project_dir / patient_id
    db_path = project_dir / "emr_registry.db"

    target_dir = output_root / patient_id
    target_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    redacted_names = 0
    for source in _normalized_md_files(patient_dir):
        text = source.read_text(encoding="utf-8", errors="replace")
        before = text
        redacted = redact_pii(text)
        if redacted != before:
            redacted_names += len(
                scan_pii(before).get("nomi_medici", [])
            ) - len(scan_pii(redacted).get("nomi_medici", []))
        (target_dir / source.name).write_text(redacted, encoding="utf-8")
        copied += 1

    if db_path.exists():
        lab_md = _lab_markdown(db_path, patient_id)
        if lab_md:
            (target_dir / "esami_ematochimici.md").write_text(
                lab_md, encoding="utf-8"
            )

    return {"patient": patient_id, "md": copied, "names_redacted": redacted_names}


def verify_export(export_dir: Path) -> dict:
    """Scan an exported patient folder for residual PII."""
    hits: dict[str, list[str]] = {}
    for path in sorted(export_dir.rglob("*.md")):
        text = path.read_text(encoding="utf-8", errors="replace")
        for name, matched in scan_pii(text).items():
            hits.setdefault(name, []).extend(matched[:10])
    return hits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--projects-root",
        type=Path,
        default=DEFAULT_PROJECTS_ROOT,
        help="Directory containing the project folders (default: Desktop).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Where to create the 'md <PROJECT>' folders (default: Desktop).",
    )
    parser.add_argument(
        "--projects",
        nargs="*",
        default=list(PROJECTS),
        choices=PROJECTS,
        help="Projects to export (default: all three).",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Rescan an existing export for residual PII and exit.",
    )
    args = parser.parse_args(argv)

    if args.verify_only:
        total: dict[str, int] = Counter()
        for project in args.projects:
            export_dir = args.output_root / f"md {project}"
            if not export_dir.exists():
                print(f"[skip] {export_dir} non esiste")
                continue
            hits = verify_export(export_dir)
            for name, matched in hits.items():
                total[name] += len(matched)
            status = "OK" if not hits else f"{len(hits)} categorie PII residue"
            print(f"[verify] {project}: {status}")
            for name, matched in hits.items():
                print(f"    {name}: {len(matched)} esempi: {sorted(set(matched))[:4]}")
        if total:
            print(f"\nTOTALE PII residue: {sum(total.values())}")
            return 1
        print("\nNessuna PII residua.")
        return 0

    grand = {"patients": 0, "md": 0, "names_redacted": 0}
    for project in args.projects:
        project_dir = args.projects_root / project
        if not project_dir.is_dir():
            print(f"[skip] {project_dir} non esiste")
            continue
        output_dir = args.output_root / f"md {project}"
        output_dir.mkdir(parents=True, exist_ok=True)

        patient_ids = sorted(
            p.name
            for p in project_dir.iterdir()
            if p.is_dir() and re.fullmatch(r"P\d{3}", p.name)
        )
        project_totals = {"md": 0, "names_redacted": 0}
        for patient_id in patient_ids:
            result = export_patient(project_dir, output_dir, patient_id)
            project_totals["md"] += result["md"]
            project_totals["names_redacted"] += result["names_redacted"]
        grand["patients"] += len(patient_ids)
        grand["md"] += project_totals["md"]
        grand["names_redacted"] += project_totals["names_redacted"]
        print(
            f"[export] {project}: {len(patient_ids)} pazienti, "
            f"{project_totals['md']} md, "
            f"{project_totals['names_redacted']} nomi redatti → {output_dir}"
        )

    print(
        f"\nFatto: {grand['patients']} pazienti, {grand['md']} file md, "
        f"{grand['names_redacted']} nomi redatti."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
