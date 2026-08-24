"""Download and selectively import local GGUF models from the LLM dialog."""

from __future__ import annotations

import html
import json
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..config import LLM_MODELS_DIR
from ..llm_backend.model_installer import (
    ModelInstallError,
    cleanup_abandoned_partials,
    discover_ollama_models,
    model_name_from_ollama_tag,
    model_name_from_url,
    normalize_model_name,
)
from ..llm_backend.model_catalog import (
    CatalogModel,
    HardwareFit,
    filter_catalog,
    hardware_fit,
    load_catalog_snapshot,
)
from ..utils.hardware import HardwareProfile


class _CatalogCard(QFrame):
    """Compact visual summary of one installable recommended model."""

    install_requested = pyqtSignal(object)

    _FIT_STYLE = {
        "optimal": "background:#d5f5e3;color:#196f3d;",
        "compatible": "background:#fcf3cf;color:#7d6608;",
        "insufficient": "background:#fadbd8;color:#922b21;",
    }

    def __init__(
        self,
        model: CatalogModel,
        fit: HardwareFit,
        *,
        installed: bool,
        parent=None,
    ):
        super().__init__(parent)
        self.model = model
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(
            "QFrame {background:#ffffff;border:1px solid #d5d8dc;"
            "border-radius:7px;} QLabel {border:none;background:transparent;}"
        )
        layout = QVBoxLayout(self)
        header = QHBoxLayout()
        title = QLabel(
            f"<b>{html.escape(model.display_name)}</b> · "
            f"{html.escape(model.provider)}"
        )
        title.setTextInteractionFlags(Qt.TextSelectableByMouse)
        header.addWidget(title, stretch=1)
        profile = QLabel(html.escape(model.profile))
        profile.setStyleSheet(
            "background:#eaf2f8;color:#1b4f72;padding:3px 7px;"
            "border-radius:8px;"
        )
        header.addWidget(profile)
        fit_badge = QLabel(fit.label)
        fit_badge.setToolTip(fit.explanation)
        fit_badge.setStyleSheet(
            self._FIT_STYLE[fit.level] + "padding:3px 7px;border-radius:8px;"
        )
        header.addWidget(fit_badge)
        layout.addLayout(header)

        description = QLabel(html.escape(model.description))
        description.setWordWrap(True)
        layout.addWidget(description)
        roles = []
        if "document" in model.roles:
            roles.append("Documenti/evidenze")
        if "clinical_state" in model.roles:
            roles.append("Evidenze/eventi/analisi")
        metrics = QLabel(
            f"<b>Ruoli:</b> {' + '.join(roles)} &nbsp; · &nbsp; "
            f"<b>Pesi:</b> {html.escape(model.parameter_label)}, "
            f"{html.escape(model.quantization)} "
            f"({model.download_size_gb:g} GiB) &nbsp; · &nbsp; "
            f"<b>Contesto:</b> {_format_context(model.context_length)} &nbsp; · "
            f"&nbsp; <b>RAM:</b> minimo {model.minimum_ram_gb:g} GiB, "
            f"consigliata {model.recommended_ram_gb:g} GiB"
        )
        metrics.setWordWrap(True)
        layout.addWidget(metrics)

        details = QLabel(
            f"<b>Punti di forza:</b> {html.escape(model.strengths)}<br>"
            f"<span style='color:#7f6000'><b>Nota:</b> "
            f"{html.escape(model.caution)}</span>"
        )
        details.setWordWrap(True)
        layout.addWidget(details)

        footer = QHBoxLayout()
        source = QLabel(
            f"Licenza: {html.escape(model.license_name)} · "
            f"<a href='{html.escape(model.source_url, quote=True)}'>"
            "scheda ufficiale</a>"
        )
        source.setOpenExternalLinks(True)
        footer.addWidget(source, stretch=1)
        self.install_button = QPushButton(
            "✓ Già installato" if installed else "⬇ Installa"
        )
        self.install_button.setEnabled(not installed)
        self.install_button.clicked.connect(
            lambda: self.install_requested.emit(self.model)
        )
        footer.addWidget(self.install_button)
        layout.addLayout(footer)


def _format_context(tokens: int) -> str:
    if tokens >= 1024 and tokens % 1024 == 0:
        return f"{tokens // 1024}K"
    return f"{tokens:,}".replace(",", ".")


class ModelManagerDialog(QDialog):
    """Explicit model installer that never receives clinical data."""

    model_installed = pyqtSignal(str)

    def __init__(self, installed_models: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Scarica o importa modelli GGUF")
        self.setMinimumWidth(720)
        self.resize(940, 860)
        self._installed_models = set(installed_models)
        self._worker: _ModelInstallWorker | None = None
        self._hardware = HardwareProfile.capture()
        self._catalog_snapshot = load_catalog_snapshot()
        self._catalog_cards: dict[str, _CatalogCard] = {}
        cleanup_abandoned_partials()

        layout = QVBoxLayout(self)
        explanation = QLabel(
            "Il modello viene salvato in <b>~/.emr_analyzer/models/</b>, "
            "validato come GGUF e registrato automaticamente. Il download "
            "è eseguito da un processo separato: EMR Analyzer e i documenti "
            "clinici restano offline."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        license_notice = QLabel(
            "Prima dell'uso verifica licenza, condizioni cliniche e limiti "
            "del modello scelto. L'installazione non aggira modelli gated o "
            "autenticazioni richieste dal fornitore."
        )
        license_notice.setWordWrap(True)
        license_notice.setStyleSheet("color: #7f6000;")
        layout.addWidget(license_notice)

        self._tabs = QTabWidget()
        # Build manual controls first because catalogue buttons prefill them.
        self._manual_page = self._build_manual_install_page()
        self._catalog_page = self._build_catalog_page()
        self._tabs.addTab(self._catalog_page, "★ Catalogo consigliato")
        self._tabs.addTab(self._manual_page, "Installazione manuale")
        layout.addWidget(self._tabs, stretch=1)

        self._status = QLabel("Pronto.")
        self._status.setWordWrap(True)
        self._status.setStyleSheet("color: #5d6d7e;")
        layout.addWidget(self._status)
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setFormat("%p%")
        layout.addWidget(self._progress)

        buttons = QHBoxLayout()
        self._install_button = QPushButton("⬇ Scarica e registra")
        self._install_button.clicked.connect(self._start_install)
        buttons.addWidget(self._install_button)
        self._cancel_button = QPushButton("Interrompi")
        self._cancel_button.setEnabled(False)
        self._cancel_button.clicked.connect(self._cancel_install)
        buttons.addWidget(self._cancel_button)
        buttons.addStretch()
        self._close_button = QPushButton("Chiudi")
        self._close_button.clicked.connect(self.reject)
        buttons.addWidget(self._close_button)
        layout.addLayout(buttons)

    def _build_manual_install_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("Origine:"))
        self._source = QComboBox()
        self._source.addItem("Ollama — modello locale o nuovo", "ollama")
        self._source.addItem("URL HTTPS diretto a un file GGUF", "url")
        source_row.addWidget(self._source, stretch=1)
        layout.addLayout(source_row)

        self._source_pages = QStackedWidget()
        self._source_pages.addWidget(self._build_ollama_page())
        self._source_pages.addWidget(self._build_url_page())
        layout.addWidget(self._source_pages)
        self._source.currentIndexChanged.connect(
            self._source_pages.setCurrentIndex
        )

        common = QGridLayout()
        self._local_name = QLineEdit()
        self._local_name.setPlaceholderText(
            "Automatico, oppure un nome come qwen3-30b-a3b-q4"
        )
        self._local_name.setToolTip(
            "Nome mostrato in Configura LLM e usato per il file GGUF locale."
        )
        common.addWidget(QLabel("Nome locale (opzionale):"), 0, 0)
        common.addWidget(self._local_name, 0, 1)
        layout.addLayout(common)
        layout.addStretch()
        return page

    def _build_catalog_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        self._catalog_header = QLabel()
        self._catalog_header.setWordWrap(True)
        self._update_catalog_header()
        layout.addWidget(self._catalog_header)

        filters = QHBoxLayout()
        self._catalog_role = QComboBox()
        self._catalog_role.addItem("Tutti i ruoli", "")
        self._catalog_role.addItem("LLM per i documenti", "document")
        self._catalog_role.addItem(
            "LLM per le evidenze atomiche", "atomic_evidence"
        )
        self._catalog_role.addItem(
            "LLM per gli eventi clinici", "clinical_events"
        )
        self._catalog_role.addItem(
            "LLM per analisi e interrogazione", "clinical_state"
        )
        filters.addWidget(self._catalog_role)
        self._catalog_search = QLineEdit()
        self._catalog_search.setPlaceholderText(
            "Cerca per nome, produttore o profilo…"
        )
        filters.addWidget(self._catalog_search, stretch=1)
        self._catalog_compatible = QCheckBox("Solo compatibili con la RAM")
        self._catalog_compatible.setChecked(True)
        filters.addWidget(self._catalog_compatible)
        self._catalog_update_button = QPushButton("↻ Aggiorna catalogo")
        self._catalog_update_button.setToolTip(
            "Verifica esplicitamente il manifesto del catalogo. Se la fonte "
            "remota non è pubblicamente accessibile, conserva il catalogo "
            "validato incluso nell'app."
        )
        self._catalog_update_button.clicked.connect(
            self._start_catalog_update
        )
        filters.addWidget(self._catalog_update_button)
        layout.addLayout(filters)

        self._catalog_scroll = QScrollArea()
        self._catalog_scroll.setWidgetResizable(True)
        self._catalog_container = QWidget()
        self._catalog_layout = QVBoxLayout(self._catalog_container)
        self._catalog_layout.setAlignment(Qt.AlignTop)
        self._catalog_scroll.setWidget(self._catalog_container)
        layout.addWidget(self._catalog_scroll, stretch=1)

        self._catalog_role.currentIndexChanged.connect(
            self._refresh_catalog
        )
        self._catalog_search.textChanged.connect(self._refresh_catalog)
        self._catalog_compatible.toggled.connect(self._refresh_catalog)
        self._refresh_catalog()
        return page

    def _update_catalog_header(self) -> None:
        hardware = self._hardware
        snapshot = self._catalog_snapshot
        source = (
            f"cache locale v{snapshot.catalog_version}"
            if snapshot.source == "cache"
            else f"integrato v{snapshot.catalog_version}"
        )
        gpu = (
            f" · {html.escape(hardware.gpu_name)}"
            if hardware.gpu_name else ""
        )
        self._catalog_header.setText(
            f"<b>Sistema rilevato:</b> {hardware.total_ram_gb:.1f} GiB RAM"
            f"{gpu}<br><span style='color:#5d6d7e'>Catalogo {source}, "
            f"verificato il {html.escape(snapshot.review_date)}; "
            "l'ordinamento privilegia i modelli adatti alla memoria "
            "disponibile.</span>"
        )

    def _refresh_catalog(self, *_args) -> None:
        while self._catalog_layout.count():
            item = self._catalog_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._catalog_cards = {}
        role = str(self._catalog_role.currentData() or "") or None
        models = filter_catalog(
            role=role,
            query=self._catalog_search.text(),
            total_ram_gb=self._hardware.total_ram_gb,
            compatible_only=self._catalog_compatible.isChecked(),
            models=self._catalog_snapshot.models,
        )
        if not models:
            empty = QLabel(
                "Nessun modello corrisponde ai filtri. Disattiva il filtro "
                "RAM oppure usa l'installazione manuale."
            )
            empty.setWordWrap(True)
            self._catalog_layout.addWidget(empty)
            return
        for model in models:
            local_name = model_name_from_ollama_tag(model.ollama_tag)
            installed = local_name.casefold() in {
                name.casefold() for name in self._installed_models
            }
            card = _CatalogCard(
                model,
                hardware_fit(model, self._hardware.total_ram_gb),
                installed=installed,
            )
            card.install_requested.connect(self._install_catalog_model)
            self._catalog_cards[model.catalog_id] = card
            self._catalog_layout.addWidget(card)

    def _install_catalog_model(self, model: CatalogModel) -> None:
        fit = hardware_fit(model, self._hardware.total_ram_gb)
        warning = ""
        if fit.level == "insufficient":
            warning = (
                "\n\n⚠ La RAM rilevata è inferiore al minimo stimato; "
                "il modello potrebbe non avviarsi."
            )
        reply = QMessageBox.question(
            self,
            "Conferma download modello",
            f"Scaricare {model.display_name}?\n\n"
            f"Tag: {model.ollama_tag}\n"
            f"Download stimato: {model.download_size_gb:g} GiB\n"
            f"RAM consigliata: {model.recommended_ram_gb:g} GiB\n"
            f"Licenza: {model.license_name}"
            f"{warning}",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self._source.setCurrentIndex(self._source.findData("ollama"))
        self._ollama_tag.setCurrentText(model.ollama_tag)
        self._local_name.clear()
        self._tabs.setCurrentWidget(self._manual_page)
        self._start_install()

    def _start_catalog_update(self) -> None:
        if self._worker is not None:
            return
        self._set_running(True)
        self._progress.setRange(0, 0)
        self._status.setText(
            "Connessione al repository ufficiale del catalogo. Nessun dato "
            "clinico viene trasmesso."
        )
        worker = _ModelInstallWorker(["--update-catalog"])
        self._worker = worker
        worker.progress_changed.connect(self._on_progress)
        worker.catalog_updated.connect(self._on_catalog_updated)
        worker.failed.connect(self._on_failure)
        worker.cancelled.connect(self._on_cancelled)
        worker.finished.connect(self._release_worker)
        worker.start()

    def _on_catalog_updated(self, result: dict) -> None:
        self._catalog_snapshot = load_catalog_snapshot()
        self._update_catalog_header()
        self._refresh_catalog()
        self._progress.setRange(0, 100)
        self._progress.setValue(100)
        count = int(result.get("model_count") or 0)
        version = int(result.get("catalog_version") or 0)
        review_date = str(result.get("review_date") or "n.d.")
        if result.get("updated", True) is False:
            notice = str(result.get("notice") or (
                "Il catalogo remoto non è disponibile; resta attivo il "
                "catalogo validato incluso nell'applicazione."
            ))
            self._progress.setFormat("Catalogo locale attivo")
            self._status.setText(
                f"Catalogo integrato v{version} attivo: {count} modelli, "
                f"revisione {review_date}."
            )
            QMessageBox.information(
                self,
                "Catalogo integrato attivo",
                f"{notice}\n\nVersione: {version}\n"
                f"Modelli disponibili: {count}\n"
                f"Ultima revisione: {review_date}.",
            )
            return
        self._progress.setFormat("Catalogo aggiornato")
        self._status.setText(
            f"✓ Catalogo v{version} aggiornato: {count} modelli, "
            f"revisione {review_date}."
        )
        QMessageBox.information(
            self,
            "Catalogo aggiornato",
            f"Catalogo v{version} validato e salvato nella cache locale.\n"
            f"Modelli disponibili: {count}.\n"
            f"Ultima revisione: {review_date}.",
        )

    def _build_ollama_page(self) -> QWidget:
        page = QWidget()
        grid = QGridLayout(page)
        self._ollama_tag = QComboBox()
        self._ollama_tag.setEditable(True)
        self._ollama_tag.setInsertPolicy(QComboBox.NoInsert)
        local_models = discover_ollama_models()
        for model in local_models:
            self._ollama_tag.addItem(model.tag)
        self._ollama_tag.setCurrentIndex(-1)
        self._ollama_tag.setEditText("")
        self._ollama_tag.lineEdit().setPlaceholderText(
            "Per esempio qwen3:30b-a3b"
        )
        grid.addWidget(QLabel("Nome/tag Ollama:"), 0, 0)
        grid.addWidget(self._ollama_tag, 0, 1)
        local_summary = QLabel(
            f"{len(local_models)} modelli rilevati nell'archivio Ollama. "
            "Se il tag è già presente verrà soltanto importato; altrimenti "
            "Ollama lo scaricherà prima."
        )
        local_summary.setWordWrap(True)
        local_summary.setStyleSheet("color: #5d6d7e;")
        grid.addWidget(local_summary, 1, 0, 1, 2)
        return page

    def _build_url_page(self) -> QWidget:
        page = QWidget()
        grid = QGridLayout(page)
        self._url = QLineEdit()
        self._url.setPlaceholderText(
            "https://…/modello.Q4_K_M.gguf"
        )
        self._sha256 = QLineEdit()
        self._sha256.setPlaceholderText(
            "64 caratteri esadecimali (consigliata, non obbligatoria)"
        )
        grid.addWidget(QLabel("URL GGUF:"), 0, 0)
        grid.addWidget(self._url, 0, 1)
        grid.addWidget(QLabel("SHA-256 attesa:"), 1, 0)
        grid.addWidget(self._sha256, 1, 1)
        warning = QLabel(
            "Usa il collegamento diretto a un singolo file .gguf. I modelli "
            "GGUF suddivisi in più file non sono ancora installabili da "
            "questa finestra."
        )
        warning.setWordWrap(True)
        warning.setStyleSheet("color: #7f6000;")
        grid.addWidget(warning, 2, 0, 1, 2)
        return page

    def _start_install(self) -> None:
        if self._worker is not None:
            return
        source = str(self._source.currentData())
        local_name = self._local_name.text().strip()
        try:
            if local_name:
                local_name = normalize_model_name(local_name)
            if source == "ollama":
                tag = self._ollama_tag.currentText().strip()
                inferred = local_name or model_name_from_ollama_tag(tag)
                arguments = ["--ollama-tag", tag]
            else:
                url = self._url.text().strip()
                inferred = local_name or model_name_from_url(url)
                arguments = ["--url", url]
                checksum = self._sha256.text().strip()
                if checksum:
                    if len(checksum) != 64 or any(
                        character not in "0123456789abcdefABCDEF"
                        for character in checksum
                    ):
                        raise ModelInstallError(
                            "La checksum SHA-256 deve contenere 64 caratteri "
                            "esadecimali."
                        )
                    arguments.extend(["--sha256", checksum])
            if inferred in self._installed_models:
                raise ModelInstallError(
                    f"Il modello {inferred} è già disponibile in EMR Analyzer."
                )
            if local_name:
                arguments.extend(["--name", local_name])
        except ModelInstallError as exc:
            QMessageBox.warning(self, "Dati non validi", str(exc))
            return

        self._set_running(True)
        self._progress.setRange(0, 0)
        self._status.setText(
            "Avvio del processo di installazione. Nessun dato clinico viene "
            "trasmesso."
        )
        worker = _ModelInstallWorker(arguments)
        self._worker = worker
        worker.progress_changed.connect(self._on_progress)
        worker.succeeded.connect(self._on_success)
        worker.failed.connect(self._on_failure)
        worker.cancelled.connect(self._on_cancelled)
        worker.finished.connect(self._release_worker)
        worker.start()

    def _on_progress(
        self, completed: int, total: int, message: str
    ) -> None:
        if total > 0:
            self._progress.setRange(0, 1000)
            self._progress.setValue(min(1000, int(completed * 1000 / total)))
            if total < 10 * 1024 * 1024:
                self._progress.setFormat(
                    f"{completed / 1024:.1f} / {total / 1024:.1f} KiB — %p%"
                )
            else:
                self._progress.setFormat(
                    f"{completed / 1024 ** 3:.2f} / "
                    f"{total / 1024 ** 3:.2f} GiB — %p%"
                )
        else:
            self._progress.setRange(0, 0)
            self._progress.setFormat(message)
        self._status.setText(message)

    def _on_success(self, name: str, entry: dict) -> None:
        self._installed_models.add(name)
        self._refresh_catalog()
        self._progress.setRange(0, 100)
        self._progress.setValue(100)
        self._progress.setFormat("Installazione completata")
        size = int(entry.get("size_bytes") or 0) / 1024 ** 3
        self._status.setText(
            f"✓ {name} registrato correttamente ({size:.2f} GiB)."
        )
        self.model_installed.emit(name)
        QMessageBox.information(
            self,
            "Modello installato",
            f"{name} è ora disponibile nei selettori LLM.\n\n"
            f"File: {entry.get('file', '')}",
        )

    def _on_failure(self, error: str) -> None:
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setFormat("Operazione non riuscita")
        self._status.setText(f"✗ {error}")
        QMessageBox.warning(self, "Operazione non riuscita", error)

    def _on_cancelled(self) -> None:
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setFormat("Annullata")
        self._status.setText(
            "Operazione annullata; eventuali file parziali sono stati rimossi."
        )

    def _release_worker(self) -> None:
        worker = self._worker
        self._worker = None
        if worker is not None:
            worker.deleteLater()
        self._set_running(False)

    def _cancel_install(self) -> None:
        if self._worker is None:
            return
        self._cancel_button.setEnabled(False)
        self._status.setText("Interruzione e pulizia del file parziale…")
        self._worker.cancel()

    def _set_running(self, running: bool) -> None:
        self._source.setEnabled(not running)
        self._source_pages.setEnabled(not running)
        self._local_name.setEnabled(not running)
        self._catalog_page.setEnabled(not running)
        self._install_button.setEnabled(not running)
        self._cancel_button.setEnabled(running)
        self._close_button.setEnabled(not running)

    def reject(self) -> None:
        if self._worker is not None:
            if QMessageBox.question(
                self,
                "Download in corso",
                "Interrompere il download del modello? Il file parziale "
                "verrà eliminato.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            ) == QMessageBox.Yes:
                self._cancel_install()
            return
        super().reject()


class _ModelInstallWorker(QThread):
    """Run the network-capable helper outside the offline clinical process."""

    progress_changed = pyqtSignal(int, int, str)
    succeeded = pyqtSignal(str, object)
    catalog_updated = pyqtSignal(object)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, arguments: list[str]):
        super().__init__()
        self.arguments = list(arguments)
        self._process: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._cancel_requested = False

    def run(self) -> None:
        helper = Path(__file__).resolve().parents[1] / (
            "llm_backend/model_download_helper.py"
        )
        command = [sys.executable, str(helper), *self.arguments]
        options: dict = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
            "bufsize": 1,
        }
        if os.name == "posix":
            options["start_new_session"] = True
        elif os.name == "nt":  # pragma: no cover - exercised on Windows
            options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

        success: tuple[str, dict] | None = None
        catalog_success: dict | None = None
        reported_error = ""
        try:
            process = subprocess.Popen(command, **options)
            with self._lock:
                self._process = process
                cancel_now = self._cancel_requested
            if cancel_now:
                self._terminate_process(process)
            assert process.stdout is not None
            for raw_line in process.stdout:
                try:
                    event = json.loads(raw_line)
                except ValueError:
                    continue
                event_type = event.get("type")
                if event_type == "progress":
                    self.progress_changed.emit(
                        int(event.get("completed") or 0),
                        int(event.get("total") or 0),
                        str(event.get("message") or "Installazione…"),
                    )
                elif event_type == "success":
                    success = (
                        str(event.get("name") or ""),
                        dict(event.get("entry") or {}),
                    )
                elif event_type == "catalog_success":
                    catalog_success = {
                        key: value for key, value in event.items()
                        if key != "type"
                    }
                elif event_type == "error":
                    reported_error = str(event.get("message") or "")
            stderr = process.stderr.read() if process.stderr is not None else ""
            return_code = process.wait()
            if self._cancel_requested:
                self._remove_process_partials(process.pid)
                self.cancelled.emit()
            elif return_code == 0 and success is not None:
                self.succeeded.emit(*success)
            elif return_code == 0 and catalog_success is not None:
                self.catalog_updated.emit(catalog_success)
            else:
                self.failed.emit(
                    reported_error
                    or stderr.strip()
                    or f"Il processo di installazione è terminato con codice "
                       f"{return_code}."
                )
        except Exception as exc:
            if self._cancel_requested:
                self.cancelled.emit()
            else:
                self.failed.emit(str(exc))
        finally:
            with self._lock:
                self._process = None

    def cancel(self) -> None:
        self._cancel_requested = True
        with self._lock:
            process = self._process
        if process is not None and process.poll() is None:
            self._terminate_process(process)

    @staticmethod
    def _terminate_process(process: subprocess.Popen) -> None:
        try:
            if os.name == "posix":
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            else:  # pragma: no cover - exercised on Windows
                process.terminate()
        except (OSError, ProcessLookupError):
            pass

    @staticmethod
    def _remove_process_partials(process_id: int) -> None:
        for path in LLM_MODELS_DIR.glob(f".*.{process_id}.*.part"):
            path.unlink(missing_ok=True)
