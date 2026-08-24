"""Configuration dialog for the versioned clinical-pipeline policy."""

from __future__ import annotations

import json

from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QSpinBox,
    QPushButton,
    QHBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..settings import (
    ClinicalPipelinePolicy,
    LabEvidencePolicy,
    load_pipeline_policy,
    save_pipeline_policy,
)


class PipelineConfigDialog(QDialog):
    """Edit extraction, laboratory and graph policies without clinical data."""

    def __init__(self, parent=None, *, pipeline_repo=None):
        super().__init__(parent)
        self._pipeline_repo = pipeline_repo
        self._pending_categories: list[tuple[str, str, str]] = []
        self.setWindowTitle("Configura pipeline clinica")
        self.resize(720, 610)
        self._policy = load_pipeline_policy()
        root = QVBoxLayout(self)
        notice = QLabel(
            "Le modifiche si applicano alle prossime costruzioni o "
            "ricostruzioni del registro. I dati sorgente non vengono modificati."
        )
        notice.setWordWrap(True)
        root.addWidget(notice)
        tabs = QTabWidget()
        tabs.addTab(self._extraction_page(), "Estrazione")
        tabs.addTab(self._laboratory_page(), "Laboratorio")
        tabs.addTab(self._graph_page(), "Eventi e consenso")
        if self._pipeline_repo is not None:
            tabs.addTab(self._categories_page(), "Categorie")
        root.addWidget(tabs)
        buttons = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _extraction_page(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        self._adaptive_retry = QCheckBox(
            "Ripeti solo le porzioni che non superano la validazione"
        )
        self._adaptive_retry.setChecked(
            self._policy.adaptive_specialized_retry
        )
        self._max_retries = QSpinBox()
        self._max_retries.setRange(0, 10)
        self._max_retries.setValue(self._policy.max_specialized_retries)
        form.addRow("Retry adattivo", self._adaptive_retry)
        form.addRow("Tentativi specializzati massimi", self._max_retries)
        return page

    def _laboratory_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        box = QGroupBox("Quando creare una evidenza atomica di laboratorio")
        form = QFormLayout(box)
        lab = self._policy.lab
        self._lab_flag = QCheckBox("Flag esplicito H/L/*/!")
        self._lab_flag.setChecked(lab.explicit_abnormal_flag)
        self._lab_range = QCheckBox("Valore fuori intervallo di riferimento")
        self._lab_range.setChecked(lab.outside_reference_range)
        self._lab_text = QCheckBox("Risultato testuale esplicitamente patologico")
        self._lab_text.setChecked(lab.textual_abnormality)
        self._lab_delta = QCheckBox(
            "Variazione significativa anche se ancora nel range"
        )
        self._lab_delta.setChecked(lab.significant_delta_within_range)
        self._delta_window = QSpinBox()
        self._delta_window.setRange(1, 3650)
        self._delta_window.setSuffix(" giorni")
        self._delta_window.setValue(lab.delta_window_days)
        self._delta_relative = QDoubleSpinBox()
        self._delta_relative.setRange(0.0, 10.0)
        self._delta_relative.setDecimals(2)
        self._delta_relative.setSingleStep(0.05)
        self._delta_relative.setValue(lab.default_relative_delta)
        for widget in (
            self._lab_flag, self._lab_range, self._lab_text, self._lab_delta,
        ):
            form.addRow(widget)
        form.addRow("Finestra variazione", self._delta_window)
        form.addRow("Variazione relativa minima", self._delta_relative)
        layout.addWidget(box)
        layout.addWidget(QLabel(
            "Regole per analizzatore/parametro (JSON). Le chiavi possono "
            "sovrascrivere relative_delta e window_days."
        ))
        self._analyzer_rules = QPlainTextEdit(json.dumps(
            lab.analyzer_rules, ensure_ascii=False, indent=2, sort_keys=True
        ))
        layout.addWidget(self._analyzer_rules)
        return page

    def _graph_page(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        self._consensus = QComboBox()
        for label, value in (
            ("Veloce — una valutazione per candidato", "fast"),
            ("Selettivo — una valutazione, profilo predefinito", "selective"),
            ("Robusto — più verifiche sui collegamenti", "robust"),
            ("Ricerca — massimo consenso e revisione", "research"),
        ):
            self._consensus.addItem(label, value)
        self._consensus.setCurrentIndex(max(
            0, self._consensus.findData(self._policy.consensus_profile)
        ))
        self._local_window = self._day_spin(
            self._policy.local_window_days, 0, 365
        )
        self._long_window = self._day_spin(
            self._policy.longitudinal_window_days, 1, 3650
        )
        self._long_step = self._day_spin(
            self._policy.longitudinal_step_days, 1, 3650
        )
        self._top_k = QSpinBox()
        self._top_k.setRange(1, 100)
        self._top_k.setValue(self._policy.semantic_top_k)
        self._cohesive = self._probability(self._policy.cohesive_threshold)
        self._split = self._probability(self._policy.bridge_split_threshold)
        form.addRow("Profilo di consenso", self._consensus)
        form.addRow("Finestra locale", self._local_window)
        form.addRow("Finestra longitudinale", self._long_window)
        form.addRow("Passo longitudinale", self._long_step)
        form.addRow("Candidati semantici massimi per evidenza", self._top_k)
        form.addRow("Soglia arco coesivo", self._cohesive)
        form.addRow("Soglia split dei ponti deboli", self._split)
        return page

    def _categories_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        label = QLabel(
            "Il catalogo è conservato in SQLite. Le categorie personalizzate "
            "sono disponibili per annotazione e correzione manuale senza una "
            "migrazione dello schema."
        )
        label.setWordWrap(True)
        layout.addWidget(label)
        self._category_table = QTableWidget(0, 3)
        self._category_table.setHorizontalHeaderLabels([
            "Identificativo", "Nome", "Livello",
        ])
        self._category_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self._category_table)
        for item in self._pipeline_repo.list_categories():
            self._append_category_row(
                item["category"], item["display_name"], item["object_level"]
            )

        row = QHBoxLayout()
        self._category_id = QLineEdit()
        self._category_id.setPlaceholderText("es. patient_reported_outcome")
        self._category_name = QLineEdit()
        self._category_name.setPlaceholderText("Nome visualizzato")
        self._category_level = QComboBox()
        self._category_level.addItem("Evento", "event")
        self._category_level.addItem("Evidenza", "evidence")
        self._category_level.addItem("Entrambi", "both")
        add = QPushButton("Aggiungi")
        add.clicked.connect(self._queue_category)
        row.addWidget(self._category_id)
        row.addWidget(self._category_name)
        row.addWidget(self._category_level)
        row.addWidget(add)
        layout.addLayout(row)
        return page

    def _queue_category(self) -> None:
        category = self._category_id.text().strip().casefold()
        name = self._category_name.text().strip()
        level = str(self._category_level.currentData())
        if (
            not category or not name
            or not all(char.isalnum() or char == "_" for char in category)
        ):
            QMessageBox.warning(
                self, "Categoria non valida",
                "Inserisci ID (lettere, numeri, underscore) e nome."
            )
            return
        existing = {
            self._category_table.item(row, 0).text()
            for row in range(self._category_table.rowCount())
        }
        if category in existing:
            QMessageBox.information(
                self, "Categoria esistente", "L'identificativo è già presente."
            )
            return
        self._pending_categories.append((category, name, level))
        self._append_category_row(category, name, level)
        self._category_id.clear()
        self._category_name.clear()

    def _append_category_row(self, category: str, name: str, level: str) -> None:
        row = self._category_table.rowCount()
        self._category_table.insertRow(row)
        for column, value in enumerate((category, name, level)):
            self._category_table.setItem(
                row, column, QTableWidgetItem(str(value))
            )

    @staticmethod
    def _day_spin(value: int, minimum: int, maximum: int) -> QSpinBox:
        widget = QSpinBox()
        widget.setRange(minimum, maximum)
        widget.setSuffix(" giorni")
        widget.setValue(value)
        return widget

    @staticmethod
    def _probability(value: float) -> QDoubleSpinBox:
        widget = QDoubleSpinBox()
        widget.setRange(0.0, 1.0)
        widget.setDecimals(2)
        widget.setSingleStep(0.05)
        widget.setValue(value)
        return widget

    def _save(self) -> None:
        try:
            rules = json.loads(self._analyzer_rules.toPlainText() or "{}")
            if not isinstance(rules, dict):
                raise ValueError("Le regole di laboratorio devono essere un oggetto JSON")
            policy = ClinicalPipelinePolicy(
                adaptive_specialized_retry=self._adaptive_retry.isChecked(),
                max_specialized_retries=self._max_retries.value(),
                consensus_profile=str(self._consensus.currentData()),
                local_window_days=self._local_window.value(),
                longitudinal_window_days=self._long_window.value(),
                longitudinal_step_days=self._long_step.value(),
                semantic_top_k=self._top_k.value(),
                cohesive_threshold=self._cohesive.value(),
                bridge_split_threshold=self._split.value(),
                lab=LabEvidencePolicy(
                    explicit_abnormal_flag=self._lab_flag.isChecked(),
                    outside_reference_range=self._lab_range.isChecked(),
                    textual_abnormality=self._lab_text.isChecked(),
                    significant_delta_within_range=self._lab_delta.isChecked(),
                    delta_window_days=self._delta_window.value(),
                    default_relative_delta=self._delta_relative.value(),
                    analyzer_rules=rules,
                ),
            )
            save_pipeline_policy(policy)
            if self._pipeline_repo is not None:
                for category, name, level in self._pending_categories:
                    self._pipeline_repo.register_category(
                        category, name, object_level=level
                    )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            QMessageBox.warning(self, "Configurazione non salvata", str(exc))
            return
        self.accept()
