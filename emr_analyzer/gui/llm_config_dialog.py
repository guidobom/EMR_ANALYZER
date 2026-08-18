"""Single configuration dialog for the two local llama.cpp roles."""

from __future__ import annotations

from PyQt5.QtCore import QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from ..extraction.llm_client import LlmClient
from ..settings import LLMRoleConfig


ROLE_DEFINITIONS = {
    "document": (
        "LLM per i documenti",
        "Isola e normalizza il contenuto clinico estratto dai singoli PDF.",
    ),
    "clinical_state": (
        "LLM per il Clinical State",
        "Ricostruisce e interroga la storia clinica longitudinale.",
    ),
}


class LLMConfigDialog(QDialog):
    """Edit, validate and test both model assignments in one place."""

    def __init__(
        self,
        configs: dict[str, LLMRoleConfig],
        available_models: list[str],
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Configura LLM locali")
        self.setMinimumWidth(860)
        self._initial_configs = dict(configs)
        self._installed_models = set(available_models)
        self._models = sorted(
            self._installed_models
            | {config.model for config in configs.values() if config.model}
        )
        self._widgets: dict[str, dict] = {}
        self._capability_cache: dict[str, dict] = {}
        self._workers: dict[str, _ModelWarmupWorker] = {}
        self._unload_worker: _ModelUnloadWorker | None = None

        layout = QVBoxLayout(self)
        if self._models:
            intro = QLabel(
                "I parametri sono indipendenti per le due funzioni. Il limite "
                "massimo del contesto viene letto dai metadati del modello "
                "locale. Il motore è un processo llama.cpp avviato e fermato "
                "dall'applicazione stessa."
            )
        else:
            intro = QLabel(
                "⚠️ Nessun modello locale registrato in "
                "~/.emr_analyzer/models.\n"
                "Esegui tools/setup_llama_backend.py per copiare i GGUF "
                "già scaricati da Ollama (nessun download)."
            )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        for role in ("document", "clinical_state"):
            layout.addWidget(self._build_role_group(role, configs[role]))

        gpu_actions = QHBoxLayout()
        gpu_actions.addStretch()
        self._unload_all_button = QPushButton("■ Libera tutta la GPU")
        self._unload_all_button.setToolTip(
            "Ferma i processi llama-server avviati dall'applicazione, "
            "scaricando dalla memoria tutti i modelli residenti."
        )
        self._unload_all_button.clicked.connect(self._unload_all_models)
        gpu_actions.addWidget(self._unload_all_button)

        # Effective llama-server slots: one server per role, spawned with
        # -np = min(parallel_workers, RAM-safe maximum).
        gpu_actions.addSpacing(20)
        self._slots_label = QLabel("Slot paralleli (llama-server): —")
        self._slots_label.setStyleSheet("color: #5d6d7e;")
        gpu_actions.addWidget(self._slots_label)
        layout.addLayout(gpu_actions)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel
        )
        self._buttons.button(QDialogButtonBox.Save).setText("Salva e applica")
        self._buttons.button(QDialogButtonBox.Cancel).setText("Annulla")
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

        for role in self._widgets:
            self._on_model_changed(role, initial=True)

        self._update_slots_label()

        self._runtime_timer = QTimer(self)
        self._runtime_timer.setInterval(5000)
        self._runtime_timer.timeout.connect(self._refresh_runtime_statuses)
        self._runtime_timer.start()

    def _build_role_group(
        self, role: str, config: LLMRoleConfig
    ) -> QGroupBox:
        title, description = ROLE_DEFINITIONS[role]
        group = QGroupBox(title)
        grid = QGridLayout(group)
        description_label = QLabel(description)
        description_label.setWordWrap(True)
        grid.addWidget(description_label, 0, 0, 1, 6)

        model = QComboBox()
        model.addItem("(nessun modello)", "")
        for name in self._models:
            model.addItem(name, name)
        index = model.findData(config.model)
        model.setCurrentIndex(index if index >= 0 else 0)

        status = QLabel("● verifica…")
        status.setMinimumWidth(112)
        test_button = QPushButton("▶ Carica e testa")
        optimize_button = QPushButton("⚡ Ottimizza")
        optimize_button.setToolTip(
            "Analizza l'hardware e il modello per suggerire "
            "i parametri ottimali (contesto, token, worker)."
        )
        unload_button = QPushButton("■ Scarica dalla GPU")
        unload_button.setEnabled(False)

        grid.addWidget(QLabel("Modello:"), 1, 0)
        grid.addWidget(model, 1, 1, 1, 3)
        grid.addWidget(status, 1, 4)
        grid.addWidget(test_button, 1, 5)
        grid.addWidget(optimize_button, 1, 6)

        max_context = QLabel("Contesto massimo: verifica in corso…")
        max_context.setStyleSheet("color: #5d6d7e;")
        grid.addWidget(max_context, 2, 1, 1, 5)

        temperature = QDoubleSpinBox()
        temperature.setRange(0.0, 2.0)
        temperature.setDecimals(2)
        temperature.setSingleStep(0.05)
        temperature.setValue(config.temperature)
        temperature.setToolTip(
            "0 produce risposte più deterministiche; valori maggiori "
            "aumentano la variabilità."
        )

        context = QSpinBox()
        context.setRange(512, 2_000_000)
        context.setSingleStep(1024)
        context.setValue(config.context_length)
        context.setGroupSeparatorShown(True)
        context.setToolTip(
            "Numero massimo di token disponibili tra prompt e risposta."
        )

        output = QSpinBox()
        output.setRange(1, 262_144)
        output.setSingleStep(256)
        output.setValue(config.max_output_tokens)
        output.setGroupSeparatorShown(True)
        output.setToolTip("Numero massimo di token generabili nella risposta.")

        top_p = QDoubleSpinBox()
        top_p.setRange(0.0, 1.0)
        top_p.setDecimals(2)
        top_p.setSingleStep(0.05)
        top_p.setValue(config.top_p)
        top_p.setToolTip("Campionamento nucleus; 0,9 è un valore conservativo.")

        top_k = QSpinBox()
        top_k.setRange(0, 1000)
        top_k.setValue(config.top_k)
        top_k.setToolTip("Limita i token candidati a ogni passo di generazione.")

        seed = QSpinBox()
        seed.setRange(-1, 2_147_483_647)
        seed.setValue(config.seed)
        seed.setToolTip(
            "Seed fisso per favorire la riproducibilità; -1 usa un seed casuale."
        )

        grid.addWidget(QLabel("Temperatura:"), 3, 0)
        grid.addWidget(temperature, 3, 1)
        grid.addWidget(QLabel("Contesto:"), 3, 3)
        grid.addWidget(context, 3, 4, 1, 2)
        grid.addWidget(QLabel("Token risposta:"), 4, 0)
        grid.addWidget(output, 4, 1)
        grid.addWidget(QLabel("Top-p:"), 4, 3)
        grid.addWidget(top_p, 4, 4, 1, 2)
        grid.addWidget(QLabel("Top-k:"), 5, 0)
        grid.addWidget(top_k, 5, 1)
        grid.addWidget(QLabel("Seed:"), 5, 3)
        grid.addWidget(seed, 5, 4, 1, 2)
        resident_note = QLabel(
            "Il modello resta in memoria finché non lo scarichi "
            "(nessuna scadenza automatica)."
        )
        resident_note.setStyleSheet("color: #5d6d7e;")
        grid.addWidget(resident_note, 6, 0, 1, 3)
        grid.addWidget(unload_button, 6, 4, 1, 2)

        # Worker selector — parallel document/text processing
        workers_combo = None
        workers_info = None
        if role in ("document", "clinical_state"):
            workers_combo = QComboBox()
            workers_combo.setToolTip(
                "Numero di documenti analizzati in parallelo dal LLM. "
                "Il valore massimo dipende dalla RAM disponibile e dal modello."
            )
            # Pre-seed with the configured value so _refresh_worker_options
            # can restore it after populating the full list.
            configured = getattr(config, "parallel_workers", 1)
            workers_combo.addItem(str(configured), configured)
            workers_info = QLabel("")
            workers_info.setStyleSheet("color: #5d6d7e; font-size: 11px;")
            grid.addWidget(QLabel("Worker paralleli:"), 7, 0)
            grid.addWidget(workers_combo, 7, 1)
            grid.addWidget(workers_info, 7, 3, 1, 3)

        # Recommendation label (shown below the parameter grid)
        rec_label = QLabel("")
        rec_label.setWordWrap(True)
        rec_label.setStyleSheet(
            "color: #2c3e50; background: #eaf2f8; "
            "border: 1px solid #aed6f1; border-radius: 4px; "
            "padding: 6px; margin-top: 4px;"
        )
        rec_label.setVisible(False)
        grid.addWidget(rec_label, 8, 0, 1, 7)

        self._widgets[role] = {
            "model": model,
            "status": status,
            "test": test_button,
            "optimize": optimize_button,
            "rec_label": rec_label,
            "unload": unload_button,
            "max_context": max_context,
            "temperature": temperature,
            "context_length": context,
            "max_output_tokens": output,
            "top_p": top_p,
            "top_k": top_k,
            "seed": seed,
            "workers_combo": workers_combo,
            "workers_info": workers_info,
        }
        model.currentIndexChanged.connect(
            lambda _index, selected_role=role: self._on_model_changed(
                selected_role
            )
        )
        test_button.clicked.connect(
            lambda _checked=False, selected_role=role: self._test_model(
                selected_role
            )
        )
        optimize_button.clicked.connect(
            lambda _checked=False, selected_role=role: self._optimize_params(
                selected_role
            )
        )
        unload_button.clicked.connect(
            lambda _checked=False, selected_role=role: self._unload_model(
                selected_role
            )
        )
        if workers_combo is not None:
            workers_combo.currentIndexChanged.connect(
                lambda _index: self._update_slots_label()
            )
        return group

    def configurations(self) -> dict[str, LLMRoleConfig]:
        return {
            role: self._collect_config(role) for role in self._widgets
        }

    def _collect_config(self, role: str) -> LLMRoleConfig:
        widgets = self._widgets[role]
        workers = 1
        if widgets.get("workers_combo") is not None:
            workers_data = widgets["workers_combo"].currentData()
            workers = int(workers_data) if workers_data is not None else 1
        return LLMRoleConfig(
            model=str(widgets["model"].currentData() or ""),
            temperature=widgets["temperature"].value(),
            context_length=widgets["context_length"].value(),
            max_output_tokens=widgets["max_output_tokens"].value(),
            top_p=widgets["top_p"].value(),
            top_k=widgets["top_k"].value(),
            seed=widgets["seed"].value(),
            # Deprecated with the llama.cpp backend; kept for compatibility.
            keep_alive_minutes=10,
            parallel_workers=workers,
        )

    def _on_model_changed(self, role: str, initial: bool = False) -> None:
        widgets = self._widgets[role]
        model = str(widgets["model"].currentData() or "")
        if not model:
            widgets["max_context"].setText("Contesto massimo: —")
            widgets["context_length"].setMaximum(2_000_000)
            widgets["test"].setEnabled(False)
            widgets["optimize"].setEnabled(False)
            widgets["unload"].setEnabled(False)
            widgets["rec_label"].setVisible(False)
            self._set_status(role, "non_selezionato")
            return
        if model not in self._installed_models:
            widgets["max_context"].setText(
                "Contesto massimo: modello non installato"
            )
            widgets["context_length"].setMaximum(2_000_000)
            widgets["test"].setEnabled(False)
            widgets["optimize"].setEnabled(False)
            widgets["unload"].setEnabled(False)
            widgets["rec_label"].setVisible(False)
            self._set_status(role, "errore", "Modello non installato")
            return

        widgets["test"].setEnabled(role not in self._workers)
        widgets["optimize"].setEnabled(True)
        widgets["unload"].setEnabled(False)
        try:
            capabilities = self._capability_cache.get(model)
            if capabilities is None:
                capabilities = LlmClient(model=model).model_capabilities()
                self._capability_cache[model] = capabilities
            maximum = capabilities.get("max_context_length")
            if maximum:
                widgets["context_length"].setMaximum(int(maximum))
                widgets["max_context"].setText(
                    "Contesto massimo dichiarato dal modello: "
                    f"{self._format_integer(maximum)} token"
                )
            else:
                widgets["context_length"].setMaximum(2_000_000)
                widgets["max_context"].setText(
                    "Contesto massimo: non dichiarato nei metadati del modello"
                )
        except Exception as exc:
            widgets["context_length"].setMaximum(2_000_000)
            widgets["max_context"].setText(
                "Contesto massimo: impossibile leggere i metadati"
            )
            widgets["optimize"].setEnabled(False)
            self._set_status(role, "errore", str(exc))
            return
        self._refresh_runtime_status(role)
        self._refresh_worker_options(role)

        # Auto-optimize when a model is first selected (only if the user
        # hasn't already manually changed parameters).
        if not initial and model:
            self._optimize_params(role, silent=True)

    def _refresh_worker_options(self, role: str) -> None:
        """Populate the parallel-workers combo based on hardware limits."""
        widgets = self._widgets[role]
        combo = widgets.get("workers_combo")
        info_label = widgets.get("workers_info")
        if combo is None:
            return

        model = str(widgets["model"].currentData() or "")
        context = widgets["context_length"].value()

        from ..utils.hardware import (
            get_worker_options,
            get_safe_max_workers,
            get_model_size_gb,
            get_available_ram_gb,
            estimate_per_request_ram_gb,
        )

        combo.blockSignals(True)
        current = combo.currentData()
        combo.clear()

        if not model:
            combo.addItem("1 (nessun modello)", 1)
            if info_label:
                info_label.setText("")
            combo.blockSignals(False)
            return

        max_safe = get_safe_max_workers(model, context)
        model_size = get_model_size_gb(model)
        available = get_available_ram_gb()
        per_req = estimate_per_request_ram_gb(context)

        for n in get_worker_options(model, context):
            label = str(n)
            if n > max_safe:
                label = f"{n} (RAM insufficiente)"
            combo.addItem(label, n)
            if n > max_safe:
                # Disable unsafe options via a greyed-out model
                model_idx = combo.count() - 1
                combo.model().item(model_idx).setEnabled(False)

        # Restore previous selection if still valid
        idx = combo.findData(current)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        else:
            combo.setCurrentIndex(0)

        combo.blockSignals(False)

        if info_label:
            if model_size:
                estimated = model_size + (per_req * min(max_safe, 8)) + 2.0
                info_label.setText(
                    f"RAM disp: {available:.1f} GB — "
                    f"Stima {max_safe} worker: ~{estimated:.1f} GB "
                    f"(modello {model_size:.1f} GB)"
                )
            else:
                info_label.setText(
                    f"RAM disp: {available:.1f} GB — max stimato: {max_safe} worker"
                )

    def _optimize_params(self, role: str, silent: bool = False) -> None:
        """Analyse hardware + model and fill recommended parameters.

        When *silent* is True the recommendation label is updated without
        a popup dialog (used on initial model selection).
        """
        widgets = self._widgets[role]
        model = str(widgets["model"].currentData() or "")
        if not model:
            return

        from ..utils.hardware import recommend_all

        try:
            rec = recommend_all(model, role)
        except Exception:
            if not silent:
                QMessageBox.warning(
                    self,
                    "Ottimizzazione non disponibile",
                    "Impossibile analizzare l'hardware o il modello.\n"
                    "Verifica che llama.cpp sia installato e il modello sia registrato.",
                )
            return

        if rec is None:
            if not silent:
                QMessageBox.warning(
                    self,
                    "Modello non trovato",
                    f"Impossibile leggere i metadati di {model}.\n"
                    "Assicurati che il modello sia registrato in ~/.emr_analyzer/models (tools/setup_llama_backend.py).",
                )
            return

        # Apply recommended values
        widgets["context_length"].setValue(rec.context_length)
        widgets["max_output_tokens"].setValue(rec.max_output_tokens)

        # Workers: pick the recommended value if it's in the combo
        combo = widgets.get("workers_combo")
        if combo is not None and combo.count() > 0:
            idx = combo.findData(rec.parallel_workers)
            if idx >= 0:
                combo.setCurrentIndex(idx)

        # Build recommendation text
        lines = [
            "<b>⚡ Parametri ottimizzati automaticamente</b>",
            f"• Contesto: {self._format_integer(rec.context_length)} token — "
            f"{rec.context_rationale}",
            f"• Token risposta: {self._format_integer(rec.max_output_tokens)} — "
            f"{rec.output_rationale}",
            f"• Worker paralleli: {rec.parallel_workers} — "
            f"{rec.workers_rationale}",
        ]
        widgets["rec_label"].setText(
            "<br>".join(lines)
        )
        widgets["rec_label"].setVisible(True)

        if not silent:
            QMessageBox.information(
                self,
                "Parametri ottimizzati",
                f"I parametri per {model} ({role}) sono stati "
                f"configurati in base all'hardware disponibile.\n\n"
                f"Contesto: {self._format_integer(rec.context_length)} token\n"
                f"Token risposta: {self._format_integer(rec.max_output_tokens)}\n"
                f"Worker paralleli: {rec.parallel_workers}",
            )

    def _refresh_runtime_statuses(self) -> None:
        for role in self._widgets:
            if role not in self._workers:
                self._refresh_runtime_status(role)

    def _refresh_runtime_status(self, role: str) -> None:
        unload_button = self._widgets[role]["unload"]
        test_button = self._widgets[role]["test"]
        model = str(self._widgets[role]["model"].currentData() or "")
        if not model:
            test_button.setEnabled(False)
            unload_button.setEnabled(False)
            self._set_status(role, "non_selezionato")
            return
        if model not in self._installed_models:
            test_button.setEnabled(False)
            unload_button.setEnabled(False)
            self._set_status(role, "errore", "Modello non installato")
            return
        test_button.setEnabled(
            role not in self._workers and self._unload_worker is None
        )
        try:
            # The role config decides the server key (model + ctx + slots):
            # probing with a bare model name would target the wrong server.
            runtime = LlmClient(
                config=self._collect_config(role)
            ).loaded_model_info()
        except Exception as exc:
            unload_button.setEnabled(False)
            self._set_status(role, "errore", str(exc))
            return
        if runtime:
            unload_button.setEnabled(True)
            self._set_status(role, "caricato", self._runtime_tooltip(runtime))
        else:
            unload_button.setEnabled(False)
            self._set_status(
                role,
                "disponibile",
                "Modello registrato, ma il suo server llama.cpp non è in esecuzione",
            )

    def _test_model(self, role: str) -> None:
        if self._workers or self._unload_worker is not None:
            QMessageBox.information(
                self,
                "Operazione LLM in corso",
                "Attendi il completamento dell'operazione corrente.",
            )
            return
        config = self._collect_config(role)
        if not config.model or not self._validate_role(role, config):
            return
        self._set_status(role, "caricamento")
        self._set_role_enabled(role, False)
        worker = _ModelWarmupWorker(role, config)
        self._workers[role] = worker
        self._unload_all_button.setEnabled(False)
        worker.succeeded.connect(self._on_warmup_success)
        worker.failed.connect(self._on_warmup_failure)
        worker.finished.connect(
            lambda selected_role=role: self._release_worker(selected_role)
        )
        worker.start()

    def _on_warmup_success(self, role: str, model: str, runtime: dict) -> None:
        self._set_role_enabled(role, True)
        self._set_status(role, "caricato", self._runtime_tooltip(runtime))
        elapsed = float(runtime.get("elapsed_seconds") or 0)
        memory = self._format_bytes(runtime.get("size_vram"))
        memory_line = f"\nMemoria modello: {memory}" if memory else ""
        QMessageBox.information(
            self,
            "Modello pronto",
            f"{model} è caricato e ha risposto correttamente.\n"
            f"Tempo di caricamento/test: {elapsed:.1f} s{memory_line}",
        )

    def _on_warmup_failure(self, role: str, model: str, error: str) -> None:
        self._set_role_enabled(role, True)
        # A server can remain resident even when its warm-up fails (for
        # example after a Metal out-of-memory error), so keep recovery
        # available directly from the dialog.
        try:
            resident = LlmClient(
                config=self._collect_config(role)
            ).loaded_model_info() is not None
        except Exception:
            resident = False
        self._widgets[role]["unload"].setEnabled(resident)
        self._set_status(role, "errore", error)
        QMessageBox.warning(
            self,
            "Test modello fallito",
            f"{model} non è stato caricato o non ha risposto.\n\n{error}",
        )

    def _release_worker(self, role: str) -> None:
        worker = self._workers.pop(role, None)
        if worker is not None:
            worker.deleteLater()
        if not self._workers and self._unload_worker is None:
            self._unload_all_button.setEnabled(True)

    def _unload_model(self, role: str) -> None:
        model = str(self._widgets[role]["model"].currentData() or "")
        if not model:
            return
        self._start_unload(role, [model])

    def _update_slots_label(self) -> None:
        """Show the effective llama-server slot count for each role.

        Every role gets its own server process spawned with
        ``-np = min(parallel_workers, RAM-safe maximum)``; the label makes
        the real concurrency visible without a server-side setting.
        """
        parts = []
        for role in ("document", "clinical_state"):
            widgets = self._widgets.get(role)
            if widgets is None:
                continue
            model = str(widgets["model"].currentData() or "")
            if not model:
                continue
            combo = widgets.get("workers_combo")
            workers = int(combo.currentData() or 1) if combo else 1
            context = widgets["context_length"].value()
            try:
                from ..utils.hardware import get_safe_max_workers
                slots = min(workers, get_safe_max_workers(model, context))
            except Exception:
                slots = workers
            parts.append(f"{role}: {slots}")
        text = "Slot paralleli (llama-server): " + (
            " · ".join(parts) if parts else "—"
        )
        self._slots_label.setText(text)

    def _unload_all_models(self) -> None:
        self._start_unload("__all__", None)

    def _start_unload(
        self, operation: str, model_names: list[str] | None
    ) -> None:
        if self._workers or self._unload_worker is not None:
            QMessageBox.information(
                self,
                "Operazione LLM in corso",
                "Attendi il completamento dell'operazione corrente.",
            )
            return

        self._runtime_timer.stop()
        self._unload_all_button.setEnabled(False)
        if operation == "__all__":
            for role in self._widgets:
                self._set_role_enabled(role, False)
                if self._widgets[role]["model"].currentData():
                    self._set_status(role, "scaricamento")
        else:
            self._set_role_enabled(operation, False)
            self._set_status(operation, "scaricamento")

        worker = _ModelUnloadWorker(operation, model_names)
        self._unload_worker = worker
        worker.succeeded.connect(self._on_unload_success)
        worker.failed.connect(self._on_unload_failure)
        worker.finished.connect(self._release_unload_worker)
        worker.start()

    def _on_unload_success(self, operation: str, result: dict) -> None:
        unloaded = result.get("unloaded", [])
        errors = result.get("errors", {})
        if errors:
            details = "\n".join(
                f"- {model}: {error}" for model, error in errors.items()
            )
            QMessageBox.warning(
                self,
                "GPU liberata parzialmente",
                f"Modelli scaricati: {len(unloaded)}.\n\n"
                f"Errori:\n{details}",
            )
        elif unloaded:
            QMessageBox.information(
                self,
                "GPU liberata",
                "Modelli scaricati dalla memoria:\n- "
                + "\n- ".join(unloaded),
            )
        else:
            QMessageBox.information(
                self,
                "GPU già libera",
                "Nessuno dei modelli selezionati era caricato in memoria.",
            )

    def _on_unload_failure(self, operation: str, error: str) -> None:
        QMessageBox.warning(
            self,
            "Impossibile liberare la GPU",
            f"llama-server non ha completato lo scaricamento.\n\n{error}",
        )

    def _release_unload_worker(self) -> None:
        worker = self._unload_worker
        self._unload_worker = None
        if worker is not None:
            worker.deleteLater()
        for role in self._widgets:
            self._set_role_enabled(role, True)
        self._unload_all_button.setEnabled(True)
        self._refresh_runtime_statuses()
        self._runtime_timer.start()

    def _set_role_enabled(self, role: str, enabled: bool) -> None:
        for key, widget in self._widgets[role].items():
            if key in {"status", "max_context", "rec_label"}:
                continue
            if widget is not None:
                widget.setEnabled(enabled)

    def _set_status(self, role: str, state: str, tooltip: str = "") -> None:
        states = {
            "non_selezionato": ("● non selezionato", "#7f8c8d"),
            "disponibile": ("● disponibile", "#d68910"),
            "caricamento": ("● caricamento…", "#2980b9"),
            "scaricamento": ("● scaricamento…", "#2980b9"),
            "caricato": ("● caricato", "#27ae60"),
            "errore": ("● errore", "#c0392b"),
        }
        text, color = states[state]
        label = self._widgets[role]["status"]
        label.setText(text)
        label.setStyleSheet(f"color: {color}; font-weight: bold;")
        label.setToolTip(tooltip or text.lstrip("● "))

    def _validate_role(self, role: str, config: LLMRoleConfig) -> bool:
        maximum = (
            self._capability_cache.get(config.model, {}).get(
                "max_context_length"
            )
            if config.model else None
        )
        title = ROLE_DEFINITIONS[role][0]
        if maximum and config.context_length > int(maximum):
            QMessageBox.warning(
                self,
                "Contesto non valido",
                f"{title}: il contesto impostato supera il massimo dichiarato "
                f"di {self._format_integer(maximum)} token.",
            )
            return False
        if config.max_output_tokens > config.context_length:
            QMessageBox.warning(
                self,
                "Lunghezza non valida",
                f"{title}: i token massimi della risposta non possono "
                "superare la lunghezza del contesto.",
            )
            return False
        return True

    def accept(self) -> None:
        if self._workers or self._unload_worker is not None:
            QMessageBox.information(
                self,
                "Operazione in corso",
                "Attendi il completamento dell'operazione prima di salvare.",
            )
            return
        configs = self.configurations()
        if not all(
            self._validate_role(role, config)
            for role, config in configs.items()
        ):
            return
        self._runtime_timer.stop()
        super().accept()

    def reject(self) -> None:
        if self._workers or self._unload_worker is not None:
            QMessageBox.information(
                self,
                "Operazione in corso",
                "Attendi il completamento dell'operazione prima di chiudere.",
            )
            return
        self._runtime_timer.stop()
        super().reject()

    @staticmethod
    def _format_integer(value) -> str:
        return f"{int(value):,}".replace(",", ".")

    @staticmethod
    def _format_bytes(value) -> str:
        if value is None:
            return ""
        try:
            size = float(value)
        except (TypeError, ValueError):
            return ""
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if size < 1024 or unit == "TB":
                return f"{size:.1f} {unit}"
            size /= 1024
        return ""

    @classmethod
    def _runtime_tooltip(cls, runtime: dict) -> str:
        lines = ["Modello caricato nel server llama.cpp dell'applicazione"]
        memory = cls._format_bytes(runtime.get("size_vram"))
        if memory:
            lines.append(f"Memoria acceleratore/unificata: {memory}")
        if runtime.get("context_length"):
            lines.append(
                "Contesto attualmente caricato: "
                f"{cls._format_integer(runtime['context_length'])} token"
            )
        if runtime.get("expires_at"):
            lines.append(f"Scadenza: {runtime['expires_at']}")
        return "\n".join(lines)


class _ModelWarmupWorker(QThread):
    """Load and verify one configured model without clinical data."""

    succeeded = pyqtSignal(str, str, object)
    failed = pyqtSignal(str, str, str)

    def __init__(self, role: str, config: LLMRoleConfig):
        super().__init__()
        self.role = role
        self.config = config

    def run(self) -> None:
        try:
            runtime = LlmClient(config=self.config).warmup()
            self.succeeded.emit(self.role, self.config.model, runtime)
        except Exception as exc:
            self.failed.emit(self.role, self.config.model, str(exc))


class _ModelUnloadWorker(QThread):
    """Unload selected or all resident llama.cpp models without blocking Qt."""

    succeeded = pyqtSignal(str, object)
    failed = pyqtSignal(str, str)

    def __init__(
        self, operation: str, model_names: list[str] | None
    ):
        super().__init__()
        self.operation = operation
        self.model_names = model_names

    def run(self) -> None:
        try:
            result = LlmClient.unload_models(self.model_names)
            self.succeeded.emit(self.operation, result)
        except Exception as exc:
            self.failed.emit(self.operation, str(exc))
