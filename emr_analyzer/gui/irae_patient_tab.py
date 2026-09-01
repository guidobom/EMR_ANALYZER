"""One patient's irAE report as a selectable, inspectable tab.

Left: a list of every finding (definitive irAEs then suspects) from the
CORRECTED report; right: the rendered Markdown of the same corrected report.
Double-clicking a finding opens the evidence inspector, where manual
corrections are persisted and re-applied; the tab refreshes itself and emits
``report_updated`` so the owning dialog stays coherent.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QListWidget, QListWidgetItem, QSplitter, QTextBrowser, QVBoxLayout,
    QWidget,
)

from ..clinical.irae_corrections import apply_irae_corrections, load_corrections
from ..clinical.irae_layers import render_irae_markdown
from ..utils.markdown_tables import render_markdown_to_html


class IraePatientTab(QWidget):
    """Selectable irAE findings on the left, rendered Markdown on the right."""

    report_updated = pyqtSignal(object)  # corrected report dict

    def __init__(
        self,
        report: dict,
        patient_id: str,
        services: dict,
        parent=None,
    ):
        super().__init__(parent)
        self._raw_report = report
        self._patient_id = patient_id
        self._services = services
        self._corrected: dict = {}
        self._setup_ui()
        self.refresh()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def current_report(self) -> dict:
        """The corrected report (raw + persisted corrections), recomputed."""
        return apply_irae_corrections(
            self._raw_report, load_corrections(self._patient_id)
        )[0]

    def set_report(self, report: dict) -> None:
        """Replace the raw report and rebuild the view.

        Persisted corrections are re-applied over the new raw report by
        ``refresh`` (see ``apply_irae_corrections``), so a re-consolidated
        report renders exactly like a freshly analysed one.
        """
        self._raw_report = report
        self.refresh()

    def markdown(self) -> str:
        """Markdown of the currently displayed (corrected) report."""
        return render_irae_markdown(self._corrected)

    def markdown_html(self) -> str:
        return render_markdown_to_html(self.markdown())

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        self._list = QListWidget()
        self._list.setToolTip(
            "Doppio click su un irAE per ispezionarne le evidenze e correggerlo"
        )
        self._list.itemDoubleClicked.connect(self._on_double_click)
        self._browser = QTextBrowser()
        self._browser.setOpenExternalLinks(False)
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._list)
        splitter.addWidget(self._browser)
        splitter.setSizes([420, 680])
        layout.addWidget(splitter)

    def refresh(self) -> None:
        """Rebuild the finding list and the rendered report."""
        self._corrected = self.current_report()
        self._list.clear()
        consolidation = self._corrected.get("consolidation") or {}
        definitive = list(self._corrected.get("iraes") or [])
        suspects = list(consolidation.get("suspects") or [])
        if not definitive and not suspects:
            placeholder = QListWidgetItem("Nessun irAE nel report.")
            placeholder.setFlags(Qt.NoItemFlags)
            self._list.addItem(placeholder)
        else:
            for item in definitive:
                self._add_item(item, definitive=True)
            for item in suspects:
                self._add_item(item, definitive=False)
        self._render_markdown()

    def _add_item(self, item: dict, definitive: bool) -> None:
        grade = item.get("ctcae_grade") or "?"
        onset = item.get("first_onset_date") or "?"
        prob = item.get("probability_immune") or "?"
        tag = "irAE" if definitive else "sospetto"
        list_item = QListWidgetItem(
            f"[{tag}] {item.get('irAE_type')} · {grade} · insorgenza {onset} · {prob}"
        )
        list_item.setData(Qt.UserRole, item)
        self._list.addItem(list_item)

    def _render_markdown(self) -> None:
        self._browser.setHtml(self.markdown_html())

    def _on_double_click(self, list_item: QListWidgetItem) -> None:
        finding = list_item.data(Qt.UserRole)
        if not isinstance(finding, dict):
            return
        from .irae_evidence_inspector import IraeEvidenceInspector

        inspector = IraeEvidenceInspector(
            self._raw_report,
            finding,
            self._patient_id,
            self._services,
            parent=self,
        )
        inspector.report_updated.connect(self._on_report_updated)
        inspector.exec_()

    def _on_report_updated(self, report: dict) -> None:
        self.refresh()
        self.report_updated.emit(report)
