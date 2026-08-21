"""Complete, provenance-preserving exports of the clinical registry."""

from __future__ import annotations

import csv
from dataclasses import asdict
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import textwrap
import zipfile


class ClinicalRegistryExporter:
    """Export the v2 registry without truncating sources or updates."""

    def __init__(self, services: dict):
        self.services = services

    def collect(
        self,
        patient_id: str,
        include_sources: bool = True,
        *,
        include_events: bool = True,
        include_labs: bool = True,
        include_profile: bool = True,
    ) -> dict:
        registry = self.services.get("registry_repo")
        if registry is None:
            raise RuntimeError("Repository del registro clinico non disponibile")
        events = []
        for event in registry.get_events(patient_id) if include_events else []:
            detail = registry.get_event_detail(event.event_id) or {
                "event": asdict(event), "evidence": [], "updates": [],
                "relations": [], "reviews": [], "episode": None,
            }
            if not include_sources:
                for evidence in detail.get("evidence", []):
                    evidence.pop("source_text", None)
                    evidence.pop("bbox", None)
            events.append(detail)

        result = {
            "schema": "emr_analyzer.clinical_registry_export.v2",
            "patient_id": patient_id,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "events": events,
            "lab_values": [],
            "medication_courses": (
                self._rows("medication_courses", patient_id) if include_events else []
            ),
            "oncology_lines": (
                self._rows("oncology_lines", patient_id) if include_events else []
            ),
            "lab_trends": (
                self._rows("lab_trends", patient_id) if include_events else []
            ),
        }
        lab_repo = self.services.get("lab_repo")
        if lab_repo and include_labs:
            result["lab_values"] = [
                value.to_dict() for value in lab_repo.get_by_patient(patient_id)
            ]
            if not include_sources:
                for value in result["lab_values"]:
                    value.pop("source_text", None)
        state_repo = self.services.get("cs_repo")
        state = state_repo.load(patient_id) if state_repo else None
        result["clinical_profile"] = (
            state.clinical_profile
            if include_profile and state and state.clinical_profile else ""
        )
        return result

    def _rows(self, table: str, patient_id: str) -> list[dict]:
        registry = self.services["registry_repo"]
        allowed = {"medication_courses", "oncology_lines", "lab_trends"}
        if table not in allowed:
            raise ValueError(table)
        rows = registry.db.execute(
            f'SELECT * FROM "{table}" WHERE patient_id=? ORDER BY 1',
            (patient_id,),
        ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            for key in list(item):
                if not key.endswith("_json"):
                    continue
                value = item.pop(key)
                try:
                    item[key[:-5]] = json.loads(value or "null")
                except (TypeError, ValueError):
                    item[key[:-5]] = value
            result.append(item)
        return result

    def export(
        self, patient_id: str, destination: str | Path,
        *, include_sources: bool = True, include_events: bool = True,
        include_labs: bool = True, include_profile: bool = True,
    ) -> None:
        path = Path(destination)
        data = self.collect(
            patient_id, include_sources=include_sources,
            include_events=include_events, include_labs=include_labs,
            include_profile=include_profile,
        )
        suffix = path.suffix.lower()
        if suffix == ".json":
            path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        elif suffix == ".csv":
            self._write_csv(path, data)
        elif suffix == ".xlsx":
            self._write_xlsx(path, data)
        elif suffix in {".md", ".txt"}:
            path.write_text(self._markdown(data), encoding="utf-8")
        elif suffix == ".pdf":
            self._write_pdf(path, self._markdown(data))
        elif suffix == ".docx":
            self._write_docx(path, self._markdown(data))
        else:
            raise ValueError(f"Formato di esportazione non supportato: {suffix}")

    @staticmethod
    def _event_rows(data: dict):
        for detail in data["events"]:
            event = detail["event"]
            evidence = detail.get("evidence", [])
            yield {
                "event_id": event.get("event_id"),
                "prima_evidenza": event.get("first_evidence_date"),
                "prima_documentazione": event.get("first_documented_date"),
                "precisione_data": event.get("date_precision"),
                "categoria": event.get("category"),
                "entita": event.get("canonical_entity"),
                "sintesi": event.get("summary_short"),
                "dettaglio": event.get("summary_detail"),
                "stato": event.get("status"),
                "certezza": event.get("certainty"),
                "gravita": event.get("severity"),
                "sede": event.get("anatomical_site"),
                "revisione": event.get("review_status"),
                "evidenze": len(evidence),
                "documenti": ", ".join(dict.fromkeys(
                    str(item.get("document_id") or "") for item in evidence
                    if item.get("document_id")
                )),
            }

    def _write_csv(self, path: Path, data: dict) -> None:
        rows = list(self._csv_rows(data))
        fields = list(dict.fromkeys(key for row in rows for key in row))
        if not fields:
            fields = ["record_type"]
        with path.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (dict, list)) else value
                    for key, value in row.items()
                })

    @staticmethod
    def _csv_rows(data: dict):
        """Yield a lossless, typed flat representation for single-file CSV.

        JSON and XLSX naturally preserve separate collections.  A CSV has one
        table, therefore every row carries ``record_type`` and, where
        applicable, ``event_id``.  This prevents the old behaviour in which a
        CSV silently omitted laboratory values and derived projections.
        """
        for detail in data.get("events", []):
            event = dict(detail.get("event") or {})
            event_id = event.get("event_id")
            yield {"record_type": "clinical_event", **event}
            episode = detail.get("episode")
            if episode:
                yield {
                    "record_type": "clinical_episode",
                    "parent_event_id": event_id,
                    **episode,
                }
            for collection, record_type in (
                ("evidence", "event_evidence"),
                ("updates", "event_update"),
                ("relations", "event_relation"),
                ("reviews", "event_review"),
            ):
                for item in detail.get(collection, []):
                    yield {
                        "record_type": record_type,
                        "parent_event_id": event_id,
                        **item,
                    }
        for collection, record_type in (
            ("lab_values", "lab_value"),
            ("medication_courses", "medication_course"),
            ("oncology_lines", "oncology_line"),
            ("lab_trends", "lab_trend"),
        ):
            for item in data.get(collection, []):
                yield {"record_type": record_type, **item}
        if data.get("clinical_profile"):
            yield {
                "record_type": "clinical_profile",
                "clinical_profile": data["clinical_profile"],
            }

    def _write_xlsx(self, path: Path, data: dict) -> None:
        from openpyxl import Workbook

        workbook = Workbook()
        first = workbook.active
        first.title = "Eventi"
        self._append_dicts(first, list(self._event_rows(data)))
        evidence_rows = []
        update_rows = []
        review_rows = []
        for detail in data["events"]:
            event_id = detail["event"]["event_id"]
            evidence_rows.extend(
                {"event_id": event_id, **item} for item in detail.get("evidence", [])
            )
            update_rows.extend(
                {"event_id": event_id, **item} for item in detail.get("updates", [])
            )
            review_rows.extend(
                {"event_id": event_id, **item} for item in detail.get("reviews", [])
            )
        for name, rows in (
            ("Evidenze", evidence_rows), ("Aggiornamenti", update_rows),
            ("Revisioni", review_rows), ("Laboratorio", data["lab_values"]),
            ("Farmaci", data["medication_courses"]),
            ("Linee oncologiche", data["oncology_lines"]),
            ("Trend laboratorio", data["lab_trends"]),
        ):
            sheet = workbook.create_sheet(name[:31])
            self._append_dicts(sheet, rows)
        profile = workbook.create_sheet("Profilo")
        profile.append(["Profilo clinico narrativo"])
        profile.append([data.get("clinical_profile", "")])
        workbook.save(path)

    @staticmethod
    def _append_dicts(sheet, rows: list[dict]) -> None:
        if not rows:
            sheet.append(["Nessun dato"])
            return
        columns = list(dict.fromkeys(key for row in rows for key in row))
        sheet.append(columns)
        for row in rows:
            sheet.append([
                json.dumps(row.get(key), ensure_ascii=False)
                if isinstance(row.get(key), (dict, list)) else row.get(key)
                for key in columns
            ])
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions

    @staticmethod
    def _markdown(data: dict) -> str:
        lines = [
            f"# Registro clinico — {data['patient_id']}", "",
            f"Esportato: {data['exported_at']}", "",
        ]
        if data.get("clinical_profile"):
            lines.extend(["## Profilo clinico", "", data["clinical_profile"], ""])
        lines.extend(["## Registro cronologico", ""])
        for detail in data["events"]:
            event = detail["event"]
            lines.extend([
                f"### {event.get('first_evidence_date') or 'Data n.d.'} — "
                f"{event.get('summary_short') or event.get('canonical_entity')}",
                "",
                f"ID: `{event['event_id']}` · categoria: {event.get('category')} · "
                f"certezza: {event.get('certainty')} · stato: {event.get('status')}",
                "",
            ])
            if event.get("summary_detail"):
                lines.extend([event["summary_detail"], ""])
            if detail.get("evidence"):
                lines.extend(["Evidenze:", ""])
                for evidence in detail["evidence"]:
                    citation = f"doc {evidence.get('document_id') or '?'}"
                    if evidence.get("source_page"):
                        citation += f", p. {evidence['source_page']}"
                    source = evidence.get("source_text") or "passaggio omesso"
                    lines.append(
                        f"- [{citation}; {evidence.get('relation') or 'supports'}] {source}"
                    )
                lines.append("")
            if detail.get("updates"):
                lines.extend(["Aggiornamenti:", ""])
                for update in detail["updates"]:
                    lines.append(
                        f"- {update.get('update_date') or 'data n.d.'}: "
                        f"{update.get('summary') or ''}"
                    )
                lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _write_pdf(path: Path, content: str) -> None:
        import fitz

        document = fitz.open()
        lines = []
        for raw_line in content.replace("**", "").replace("`", "").splitlines():
            lines.extend(textwrap.wrap(raw_line, width=105) or [""])
        for offset in range(0, len(lines), 53):
            page = document.new_page(width=595, height=842)
            page.insert_text(
                (42, 48), "\n".join(lines[offset:offset + 53]),
                fontsize=9, fontname="helv", lineheight=1.25,
            )
        if not document.page_count:
            document.new_page()
        document.save(path)
        document.close()

    @staticmethod
    def _write_docx(path: Path, content: str) -> None:
        paragraphs = "".join(
            "<w:p><w:r><w:t xml:space=\"preserve\">"
            + escape(line) + "</w:t></w:r></w:p>"
            for line in content.replace("**", "").replace("`", "").splitlines()
        )
        document = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            f"<w:body>{paragraphs}<w:sectPr/></w:body></w:document>"
        )
        content_types = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
            '</Types>'
        )
        relationships = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
            '</Relationships>'
        )
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("[Content_Types].xml", content_types)
            archive.writestr("_rels/.rels", relationships)
            archive.writestr("word/document.xml", document)
