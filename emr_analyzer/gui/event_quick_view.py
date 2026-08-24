"""Source-rich Quick View for one clinical episode."""

from __future__ import annotations

from collections import defaultdict
import html

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QSplitter, QTextBrowser, QTreeWidget,
    QTreeWidgetItem, QVBoxLayout, QWidget, QLabel,
)

from .pdf_viewer import DocumentEvidencePreview


class EventQuickViewDialog(QDialog):
    """Display an event and jump from each atom to its highlighted source."""

    def __init__(
        self,
        detail: dict,
        services: dict,
        parent=None,
        *,
        selected_evidence_id: str | None = None,
    ):
        super().__init__(parent)
        self._detail = detail or {}
        self._services = services
        self._selected_evidence_id = selected_evidence_id
        event = self._detail.get("event") or {}
        self.setWindowTitle(
            "Quick View evento — " + str(event.get("summary_short") or "")[:80]
        )
        self.resize(1450, 900)
        self._setup_ui()
        self._populate()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(QLabel("<b>Episodio clinico e provenienza</b>"))
        self._summary = QTextBrowser()
        self._summary.setMaximumHeight(250)
        left_layout.addWidget(self._summary)
        self._evidence_tree = QTreeWidget()
        self._evidence_tree.setHeaderLabels(
            ["Data", "Relazione", "Evidenza atomica"]
        )
        self._evidence_tree.setColumnWidth(0, 105)
        self._evidence_tree.setColumnWidth(1, 140)
        self._evidence_tree.itemSelectionChanged.connect(
            self._show_selected_evidence
        )
        left_layout.addWidget(self._evidence_tree, stretch=1)
        splitter.addWidget(left)
        self._preview = DocumentEvidencePreview(self._services, self)
        splitter.addWidget(self._preview)
        splitter.setSizes([520, 930])
        layout.addWidget(splitter)

    def _populate(self) -> None:
        event = self._detail.get("event") or {}
        episode = self._detail.get("episode") or {}
        relations = self._detail.get("relations") or []
        escaped_summary = html.escape(
            str(event.get("summary_short") or "Evento clinico")
        )
        escaped_start = html.escape(
            str(event.get("first_evidence_date") or "n.d.")
        )
        escaped_end = html.escape(str(event.get("date_end") or ""))
        escaped_status = html.escape(str(event.get("status") or "n.d."))
        escaped_certainty = html.escape(
            str(event.get("certainty") or "n.d.")
        )
        lines = [
            f"<h3>{escaped_summary}</h3>",
            f"<p><b>Periodo:</b> {escaped_start}"
            + (f" → {escaped_end}" if escaped_end else "")
            + f" &nbsp; <b>Stato:</b> {escaped_status}"
            + f" &nbsp; <b>Certezza:</b> {escaped_certainty}</p>",
        ]
        detail_text = event.get("summary_detail") or ""
        if detail_text and detail_text != event.get("summary_short"):
            lines.append(f"<p>{html.escape(str(detail_text))}</p>")
        if episode.get("recurrence_index", 1) > 1:
            lines.append(f"<p><b>Ricorrenza:</b> {episode['recurrence_index']}</p>")
        if relations:
            lines.append("<p><b>Episodi collegati:</b></p><ul>")
            event_id = event.get("event_id")
            for relation in relations:
                outgoing = relation.get("source_event_id") == event_id
                other = relation.get("target_summary") if outgoing else relation.get("source_summary")
                lines.append(
                    f"<li>{html.escape(str(relation.get('relation_type') or 'collegato'))}: "
                    f"{html.escape(str(other or 'evento'))}</li>"
                )
            lines.append("</ul>")
        self._summary.setHtml("".join(lines))

        document_repo = self._services.get("document_repo")
        grouped = defaultdict(list)
        for evidence in self._detail.get("evidence", []):
            grouped[str(evidence.get("document_id") or "")].append(evidence)
        first_child = selected_child = None
        for document_id, evidences in grouped.items():
            document = document_repo.get_by_id(document_id) if document_repo else None
            filename = document.filename if document else document_id
            document_date = (
                document.document_date if document else evidences[0].get("source_document_date")
            )
            root = QTreeWidgetItem([
                str(document_date or "n.d."), "Referto",
                f"{filename} — {len(evidences)} evidenze",
            ])
            root.setData(0, Qt.UserRole, {"document_id": document_id})
            self._evidence_tree.addTopLevelItem(root)
            for evidence in evidences:
                relation = evidence.get("relation") or "supports"
                if relation == "duplicate_source":
                    included = "copia · solo fonte"
                    relation_label = "copia documentale"
                else:
                    included = (
                        "in sintesi"
                        if evidence.get("included_in_summary") else "contesto"
                    )
                    relation_label = relation
                child = QTreeWidgetItem([
                    str(evidence.get("observed_date") or evidence.get("source_document_date") or "n.d."),
                    f"{relation_label} · {included}",
                    str(evidence.get("normalized_entity") or evidence.get("source_text") or ""),
                ])
                child.setToolTip(2, str(evidence.get("source_text") or ""))
                child.setData(0, Qt.UserRole, evidence)
                root.addChild(child)
                first_child = first_child or child
                if evidence.get("evidence_id") == self._selected_evidence_id:
                    selected_child = child
            root.setExpanded(True)
        target = selected_child or first_child
        if target:
            self._evidence_tree.setCurrentItem(target)

    def _show_selected_evidence(self) -> None:
        item = self._evidence_tree.currentItem()
        evidence = item.data(0, Qt.UserRole) if item else None
        if not isinstance(evidence, dict) or not evidence.get("evidence_id"):
            return
        document_id = str(evidence.get("document_id") or "")
        document_repo = self._services.get("document_repo")
        document = document_repo.get_by_id(document_id) if document_repo else None
        if not document:
            return
        same_document = [
            source for source in self._detail.get("evidence", [])
            if str(source.get("document_id") or "") == document_id
        ]
        highlights = [{
            "evidence_id": source.get("evidence_id"),
            "page_number": source.get("source_page"),
            "bbox": source.get("bbox"),
            "source_text": source.get("source_text"),
            "normalized_entity": source.get("normalized_entity"),
        } for source in same_document]
        self._preview.set_document(
            document.to_dict(),
            highlights=highlights,
            selected_evidence_id=evidence.get("evidence_id"),
        )

    def closeEvent(self, event) -> None:
        self._preview.close_document()
        event.accept()
