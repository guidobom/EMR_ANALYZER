"""Configuration dialog for independent llama.cpp and vLLM roles."""

from __future__ import annotations

from PyQt5.QtCore import QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QCheckBox,
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
    QTabWidget,
    QVBoxLayout,
)

from ..extraction.llm_client import LlmClient
from ..settings import LLMRoleConfig, MODEL_ROLES


ROLE_DEFINITIONS = {
    "document": (
        "LLM per i documenti",
        "Isola e normalizza il contenuto clinico estratto dai singoli PDF.",
    ),
    "atomic_evidence": (
        "LLM per le evidenze atomiche",
        "Estrae fatti clinici aderenti al testo, citati e strutturati in JSON.",
    ),
    "clinical_events": (
        "LLM per gli eventi clinici",
        "Fonde, deduplica e assembla le evidenze in problemi ed episodi.",
    ),
    "clinical_state": (
        "LLM per analisi e interrogazione",
        "Interroga il registro e genera analisi longitudinali, incluso irAE.",
    ),
}

ROLE_TAB_LABELS = {
    "document": "Documenti",
    "atomic_evidence": "Evidenze atomiche",
    "clinical_events": "Eventi clinici",
    "clinical_state": "Analisi",
}


class LLMConfigDialog(QDialog):
    """Edit, validate and test all model assignments in one place."""

    def __init__(
        self,
        configs: dict[str, LLMRoleConfig],
        available_models: list[str],
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Configura LLM locali")
        self.setMinimumWidth(900)
        self.resize(980, 850)
        configs = dict(configs)
        for role in ("atomic_evidence", "clinical_events"):
            configs.setdefault(role, configs["clinical_state"])
        self._initial_configs = dict(configs)
        self._installed_models = set(available_models)  # GGUF / llama.cpp
        try:
            self._vllm_models = set(
                LlmClient.list_available_models(backend="vllm")
            )
        except Exception:
            self._vllm_models = set()
        configured_by_backend = {"llama_cpp": set(), "vllm": set()}
        for config in configs.values():
            if config.model:
                configured_by_backend.setdefault(config.backend, set()).add(
                    config.model
                )
        self._model_choices = {
            "llama_cpp": self._installed_models | configured_by_backend["llama_cpp"],
            "vllm": self._vllm_models | configured_by_backend["vllm"],
        }
        self._models = sorted(
            self._model_choices["llama_cpp"] | self._model_choices["vllm"]
        )
        self._widgets: dict[str, dict] = {}
        self._capability_cache: dict[str, dict] = {}
        self._workers: dict[str, _ModelWarmupWorker] = {}
        self._warmup_was_loaded: dict[str, bool] = {}
        self._runtimes_started_in_dialog: dict[
            tuple, LLMRoleConfig
        ] = {}
        self._unload_worker: _ModelUnloadWorker | None = None
        self._unload_affected_roles: set[str] = set()

        layout = QVBoxLayout(self)
        if self._models:
            intro = QLabel(
                "Le quattro funzioni hanno modelli e parametri indipendenti. "
                "Quando modello, contesto e slot coincidono condividono un "
                "solo processo locale e una sola copia dei pesi. "
                "Temperatura e token di risposta possono invece differire "
                "senza duplicare il modello."
            )
        else:
            intro = QLabel(
                "⚠️ Nessun modello locale registrato in "
                "~/.emr_analyzer/models.\n"
                "Usa “Scarica o importa modelli” per aggiungere un GGUF."
            )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        acceleration_box = QGroupBox("Accelerazione CUDA / Metal e vLLM")
        acceleration_layout = QHBoxLayout(acceleration_box)
        self._acceleration_status = QLabel(
            "Accelerazione non ancora verificata. Il controllo non carica "
            "modelli e non interrompe i server in esecuzione."
        )
        self._acceleration_status.setWordWrap(True)
        self._acceleration_status.setStyleSheet("color: #5d6d7e;")
        acceleration_layout.addWidget(self._acceleration_status, stretch=1)
        self._acceleration_probe_button = QPushButton(
            "Verifica CUDA / Metal / vLLM"
        )
        self._acceleration_probe_button.setToolTip(
            "Verifica llama-server, vLLM e i dispositivi disponibili senza "
            "caricare modelli o interrompere server."
        )
        self._acceleration_probe_button.clicked.connect(
            self._start_acceleration_probe
        )
        acceleration_layout.addWidget(self._acceleration_probe_button)
        layout.addWidget(acceleration_box)
        self._acceleration_worker: _AccelerationProbeWorker | None = None

        model_actions = QHBoxLayout()
        self._model_count_label = QLabel()
        self._model_count_label.setStyleSheet("color: #5d6d7e;")
        model_actions.addWidget(self._model_count_label)
        model_actions.addStretch()
        self._manage_models_button = QPushButton(
            "⬇ Scarica o importa modelli…"
        )
        self._manage_models_button.setToolTip(
            "Scarica un nuovo GGUF tramite Ollama o URL HTTPS, oppure importa "
            "selettivamente un modello già presente nell'archivio Ollama."
        )
        self._manage_models_button.clicked.connect(self._open_model_manager)
        model_actions.addWidget(self._manage_models_button)
        layout.addLayout(model_actions)
        self._update_model_count_label()

        self._role_tabs = QTabWidget()
        for role in MODEL_ROLES:
            self._role_tabs.addTab(
                self._build_role_group(role, configs[role]),
                ROLE_TAB_LABELS[role],
            )
        layout.addWidget(self._role_tabs, stretch=1)

        runtime_box = QGroupBox("Server LLM locali fisici")
        runtime_layout = QHBoxLayout(runtime_box)
        self._runtime_summary = QLabel("Verifica runtime in corso…")
        self._runtime_summary.setWordWrap(True)
        runtime_layout.addWidget(self._runtime_summary, stretch=1)
        self._align_runtime_button = QPushButton("Allinea e condividi runtime")
        self._align_runtime_button.setToolTip(
            "Usa per i ruoli con lo stesso modello il contesto e il numero "
            "di slot più "
            "capienti, mantenendo indipendenti i parametri di generazione."
        )
        self._align_runtime_button.clicked.connect(self._align_shared_runtime)
        self._align_runtime_button.setVisible(False)
        runtime_layout.addWidget(self._align_runtime_button)
        layout.addWidget(runtime_box)

        gpu_actions = QHBoxLayout()
        gpu_actions.addStretch()
        self._unload_all_button = QPushButton("■ Libera tutti i modelli")
        self._unload_all_button.setToolTip(
            "Ferma i processi llama-server e vLLM avviati dall'applicazione, "
            "scaricando dalla memoria unificata/GPU tutti i modelli residenti."
        )
        self._unload_all_button.clicked.connect(self._unload_all_models)
        gpu_actions.addWidget(self._unload_all_button)

        gpu_actions.addSpacing(20)
        self._slots_label = QLabel("Slot reali: —")
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

    def _start_acceleration_probe(self) -> None:
        if (
            self._acceleration_worker is not None
            and self._acceleration_worker.isRunning()
        ):
            return
        self._acceleration_status.setText(
            "Verifica dei backend locali llama.cpp e vLLM in corso…"
        )
        self._acceleration_status.setStyleSheet(
            "color: #2980b9; font-weight: bold;"
        )
        self._acceleration_probe_button.setEnabled(False)
        worker = _AccelerationProbeWorker(self)
        self._acceleration_worker = worker
        worker.completed.connect(self._show_acceleration_diagnostic)
        worker.finished.connect(self._release_acceleration_worker)
        worker.start()

    def _show_acceleration_diagnostic(self, diagnostic) -> None:
        colors = {
            "accelerated": "#1e8449",
            "cpu_only": "#d68910",
            "binary_missing": "#c0392b",
            "backend_missing": "#c0392b",
            "backend_unavailable": "#c0392b",
            "probe_unsupported": "#d68910",
            "error": "#c0392b",
        }
        self._acceleration_status.setText(diagnostic.summary)
        self._acceleration_status.setStyleSheet(
            f"color: {colors.get(diagnostic.status, '#5d6d7e')}; "
            "font-weight: bold;"
        )
        self._acceleration_status.setToolTip(diagnostic.details)
        self._acceleration_probe_button.setText("Verifica di nuovo")

    def _release_acceleration_worker(self) -> None:
        worker = self._acceleration_worker
        self._acceleration_worker = None
        self._acceleration_probe_button.setEnabled(True)
        if worker is not None:
            worker.deleteLater()

    def _open_model_manager(self) -> None:
        from .model_manager_dialog import ModelManagerDialog

        manager = ModelManagerDialog(sorted(self._installed_models), self)
        manager.model_installed.connect(self._register_installed_model)
        manager.exec_()
        # Also catches an installation completed immediately before a window
        # manager close or a future non-signal installation path.
        from ..llm_backend.model_store import list_models
        for name in list_models():
            if name not in self._installed_models:
                self._register_installed_model(name)

    def _register_installed_model(self, name: str) -> None:
        """Expose a newly registered GGUF without restarting the app."""
        clean_name = str(name or "").strip()
        if not clean_name:
            return
        selections = {
            role: str(widgets["model"].currentData() or "")
            for role, widgets in self._widgets.items()
        }
        self._installed_models.add(clean_name)
        self._model_choices["llama_cpp"].add(clean_name)
        self._models = sorted(
            self._model_choices["llama_cpp"] | self._model_choices["vllm"]
        )
        self._capability_cache.pop(("llama_cpp", clean_name), None)
        for role, widgets in self._widgets.items():
            if widgets["backend"].currentData() != "llama_cpp":
                continue
            combo = widgets["model"]
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("(nessun modello)", "")
            for model_name in sorted(self._model_choices["llama_cpp"]):
                combo.addItem(model_name, model_name)
            index = combo.findData(selections[role])
            combo.setCurrentIndex(index if index >= 0 else 0)
            combo.blockSignals(False)
            self._on_model_changed(role, initial=True)
        self._update_model_count_label()

    def _update_model_count_label(self) -> None:
        count = len(self._installed_models)
        self._model_count_label.setText(
            f"GGUF registrati: {count} · modelli vLLM locali: "
            f"{len(self._vllm_models)}"
        )

    def _build_role_group(
        self, role: str, config: LLMRoleConfig
    ) -> QGroupBox:
        title, description = ROLE_DEFINITIONS[role]
        group = QGroupBox(title)
        grid = QGridLayout(group)
        description_label = QLabel(description)
        description_label.setWordWrap(True)
        grid.addWidget(description_label, 0, 0, 1, 6)

        backend = QComboBox()
        backend.addItem("llama.cpp · GGUF (macOS / CUDA)", "llama_cpp")
        backend.addItem("vLLM · CUDA (DGX / Linux)", "vllm")
        backend_index = backend.findData(config.backend)
        backend.setCurrentIndex(backend_index if backend_index >= 0 else 0)
        backend.setToolTip(
            "llama.cpp usa GGUF ed è compatibile con Metal e CUDA; vLLM "
            "usa modelli Hugging Face già locali ed è destinato a CUDA."
        )

        model = QComboBox()
        # Create the line editor even when the initial backend is llama.cpp;
        # it will become visible if the user later switches this role to
        # vLLM and wants to enter an explicit local model directory.
        model.setEditable(True)
        model.addItem("(nessun modello)", "")
        for name in sorted(self._model_choices.get(config.backend, set())):
            model.addItem(name, name)
        index = model.findData(config.model)
        model.setCurrentIndex(index if index >= 0 else 0)
        model.setEditable(config.backend == "vllm")

        status = QLabel("● verifica…")
        status.setMinimumWidth(112)
        test_button = QPushButton("▶ Carica e testa")
        optimize_button = QPushButton("⚡ Ottimizza")
        optimize_button.setToolTip(
            "Analizza l'hardware e il modello per suggerire "
            "i parametri ottimali (contesto, token, worker)."
        )
        unload_button = QPushButton("■ Scarica dalla memoria")
        unload_button.setEnabled(False)

        grid.addWidget(QLabel("Backend:"), 1, 0)
        grid.addWidget(backend, 1, 1, 1, 3)
        grid.addWidget(QLabel("Modello:"), 2, 0)
        grid.addWidget(model, 2, 1, 1, 3)
        grid.addWidget(status, 2, 4)
        grid.addWidget(test_button, 2, 5)
        grid.addWidget(optimize_button, 2, 6)

        max_context = QLabel("Contesto massimo: verifica in corso…")
        max_context.setStyleSheet("color: #5d6d7e;")
        grid.addWidget(max_context, 3, 1, 1, 5)

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
            "Contesto massimo per ogni richiesta/slot, condiviso tra prompt "
            "e risposta. Non è la somma dei contesti di tutti gli slot."
        )

        output = QSpinBox()
        output.setRange(1, 262_144)
        output.setSingleStep(256)
        output.setValue(config.max_output_tokens)
        output.setGroupSeparatorShown(True)
        output.setToolTip(
            "Tetto massimo della risposta, incluso nel contesto della singola "
            "richiesta. Se il modello raggiunge questo valore la risposta può "
            "risultare troncata."
        )

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

        grid.addWidget(QLabel("Temperatura:"), 4, 0)
        grid.addWidget(temperature, 4, 1)
        grid.addWidget(QLabel("Contesto:"), 4, 3)
        grid.addWidget(context, 4, 4, 1, 2)
        grid.addWidget(QLabel("Token risposta (massimo):"), 5, 0)
        grid.addWidget(output, 5, 1)
        grid.addWidget(QLabel("Top-p:"), 5, 3)
        grid.addWidget(top_p, 5, 4, 1, 2)
        grid.addWidget(QLabel("Top-k:"), 6, 0)
        grid.addWidget(top_k, 6, 1)
        grid.addWidget(QLabel("Seed:"), 6, 3)
        grid.addWidget(seed, 6, 4, 1, 2)
        output_warning = QLabel("")
        output_warning.setWordWrap(True)
        output_warning.setStyleSheet(
            "color: #7f6000; background: #fff4ce; "
            "border: 1px solid #e5c365; border-radius: 4px; padding: 4px;"
        )
        output_warning.setVisible(False)
        grid.addWidget(output_warning, 7, 0, 1, 7)

        resident_note = QLabel(
            "Il server resta residente finché non viene scaricato. Se è "
            "condiviso, lo scaricamento interessa tutti i ruoli associati."
        )
        resident_note.setStyleSheet("color: #5d6d7e;")
        grid.addWidget(resident_note, 8, 0, 1, 3)
        grid.addWidget(unload_button, 8, 4, 1, 2)

        # Worker selector — parallel document/text processing
        workers_combo = None
        workers_info = None
        if role in MODEL_ROLES:
            workers_combo = QComboBox()
            workers_combo.setToolTip(
                "Numero di richieste simultanee gestibili dal server. "
                "La memoria del contesto viene riservata per ogni slot."
            )
            # Pre-seed with the configured value so _refresh_worker_options
            # can restore it after populating the full list.
            configured = getattr(config, "parallel_workers", 1)
            workers_combo.addItem(str(configured), configured)
            workers_info = QLabel("")
            workers_info.setWordWrap(True)
            workers_info.setStyleSheet("color: #5d6d7e; font-size: 11px;")
            grid.addWidget(QLabel("Richieste parallele (slot):"), 9, 0)
            grid.addWidget(workers_combo, 9, 1)
            grid.addWidget(workers_info, 9, 3, 1, 3)

        speculative = QCheckBox(
            "Decodifica speculativa n-gram (sperimentale)"
        )
        speculative.setChecked(bool(config.speculative_decoding))
        speculative.setToolTip(
            "llama.cpp propone sequenze ripetitive e il modello principale "
            "verifica ogni token prima di accettarlo. Può accelerare il JSON "
            "clinico, ma va misurato sul computer locale."
        )
        grid.addWidget(speculative, 10, 0, 1, 6)

        vllm_group = QGroupBox("Parametri motore vLLM")
        vllm_grid = QGridLayout(vllm_group)
        vllm_dtype = QComboBox()
        for value in ("auto", "bfloat16", "float16", "float32"):
            vllm_dtype.addItem(value, value)
        dtype_index = vllm_dtype.findData(config.vllm_dtype)
        vllm_dtype.setCurrentIndex(dtype_index if dtype_index >= 0 else 0)
        vllm_gpu_memory = QDoubleSpinBox()
        vllm_gpu_memory.setRange(0.05, 0.99)
        vllm_gpu_memory.setDecimals(2)
        vllm_gpu_memory.setSingleStep(0.05)
        vllm_gpu_memory.setValue(config.vllm_gpu_memory_utilization)
        vllm_tensor_parallel = QSpinBox()
        vllm_tensor_parallel.setRange(1, 16)
        vllm_tensor_parallel.setValue(config.vllm_tensor_parallel_size)
        vllm_quantization = QComboBox()
        vllm_quantization.setEditable(True)
        for value in ("", "awq", "gptq", "bitsandbytes", "fp8", "compressed-tensors"):
            vllm_quantization.addItem(value or "auto / dal modello", value)
        quant_index = vllm_quantization.findData(config.vllm_quantization)
        if quant_index >= 0:
            vllm_quantization.setCurrentIndex(quant_index)
        elif config.vllm_quantization:
            vllm_quantization.setEditText(config.vllm_quantization)
        vllm_trust_remote_code = QCheckBox("Consenti codice remoto già locale")
        vllm_trust_remote_code.setChecked(config.vllm_trust_remote_code)
        vllm_trust_remote_code.setToolTip(
            "Abilitare solo per modelli verificati: permette al repository "
            "locale di eseguire codice Python personalizzato."
        )
        vllm_enforce_eager = QCheckBox("Forza modalità eager")
        vllm_enforce_eager.setChecked(config.vllm_enforce_eager)
        vllm_grid.addWidget(QLabel("Precisione:"), 0, 0)
        vllm_grid.addWidget(vllm_dtype, 0, 1)
        vllm_grid.addWidget(QLabel("Quota memoria GPU:"), 0, 2)
        vllm_grid.addWidget(vllm_gpu_memory, 0, 3)
        vllm_grid.addWidget(QLabel("Tensor parallel GPU:"), 1, 0)
        vllm_grid.addWidget(vllm_tensor_parallel, 1, 1)
        vllm_grid.addWidget(QLabel("Quantizzazione:"), 1, 2)
        vllm_grid.addWidget(vllm_quantization, 1, 3)
        vllm_grid.addWidget(vllm_trust_remote_code, 2, 0, 1, 2)
        vllm_grid.addWidget(vllm_enforce_eager, 2, 2, 1, 2)
        grid.addWidget(vllm_group, 11, 0, 1, 7)

        # Recommendation label (shown below the parameter grid)
        rec_label = QLabel("")
        rec_label.setWordWrap(True)
        rec_label.setStyleSheet(
            "color: #2c3e50; background: #eaf2f8; "
            "border: 1px solid #aed6f1; border-radius: 4px; "
            "padding: 6px; margin-top: 4px;"
        )
        rec_label.setVisible(False)
        grid.addWidget(rec_label, 12, 0, 1, 7)

        self._widgets[role] = {
            "backend": backend,
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
            "output_warning": output_warning,
            "top_p": top_p,
            "top_k": top_k,
            "seed": seed,
            "workers_combo": workers_combo,
            "workers_info": workers_info,
            "speculative_decoding": speculative,
            "vllm_group": vllm_group,
            "vllm_dtype": vllm_dtype,
            "vllm_gpu_memory_utilization": vllm_gpu_memory,
            "vllm_tensor_parallel_size": vllm_tensor_parallel,
            "vllm_quantization": vllm_quantization,
            "vllm_trust_remote_code": vllm_trust_remote_code,
            "vllm_enforce_eager": vllm_enforce_eager,
        }
        backend.currentIndexChanged.connect(
            lambda _index, selected_role=role: self._on_backend_changed(
                selected_role
            )
        )
        model.currentIndexChanged.connect(
            lambda _index, selected_role=role: self._on_model_changed(
                selected_role
            )
        )
        model.editTextChanged.connect(
            lambda _text, selected_role=role: self._on_model_changed(
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
                lambda _index, selected_role=role: (
                    self._refresh_worker_info(selected_role),
                    self._update_runtime_summary(),
                )
            )
        context.valueChanged.connect(
            lambda _value, selected_role=role: self._on_runtime_shape_changed(
                selected_role
            )
        )
        speculative.stateChanged.connect(
            lambda _value: self._update_runtime_summary()
        )
        output.valueChanged.connect(
            lambda _value, selected_role=role: (
                self._refresh_output_guidance(selected_role)
            )
        )
        for vllm_widget in (
            vllm_dtype,
            vllm_gpu_memory,
            vllm_tensor_parallel,
            vllm_quantization,
            vllm_trust_remote_code,
            vllm_enforce_eager,
        ):
            signal = getattr(vllm_widget, "currentIndexChanged", None)
            if signal is None:
                signal = getattr(vllm_widget, "valueChanged", None)
            if signal is None:
                signal = getattr(vllm_widget, "stateChanged", None)
            if signal is not None:
                signal.connect(
                    lambda _value, selected_role=role: (
                        self._refresh_worker_info(selected_role),
                        self._refresh_runtime_statuses(),
                    )
                )
        return group

    def configurations(self) -> dict[str, LLMRoleConfig]:
        return {
            role: self._collect_config(role) for role in self._widgets
        }

    @staticmethod
    def _selected_model(widgets: dict) -> str:
        combo = widgets["model"]
        data = combo.currentData()
        if data not in (None, ""):
            return str(data).strip()
        text = str(combo.currentText() or "").strip()
        return "" if text == "(nessun modello)" else text

    def _collect_config(self, role: str) -> LLMRoleConfig:
        widgets = self._widgets[role]
        backend = str(widgets["backend"].currentData() or "llama_cpp")
        workers = 1
        if widgets.get("workers_combo") is not None:
            workers_data = widgets["workers_combo"].currentData()
            workers = int(workers_data) if workers_data is not None else 1
        return LLMRoleConfig(
            model=self._selected_model(widgets),
            backend=backend,
            temperature=widgets["temperature"].value(),
            context_length=widgets["context_length"].value(),
            max_output_tokens=widgets["max_output_tokens"].value(),
            top_p=widgets["top_p"].value(),
            top_k=widgets["top_k"].value(),
            seed=widgets["seed"].value(),
            # Deprecated with the llama.cpp backend; kept for compatibility.
            keep_alive_minutes=10,
            parallel_workers=workers,
            speculative_decoding=(
                widgets["speculative_decoding"].isChecked()
                if backend == "llama_cpp" else False
            ),
            vllm_dtype=str(widgets["vllm_dtype"].currentData() or "auto"),
            vllm_gpu_memory_utilization=(
                widgets["vllm_gpu_memory_utilization"].value()
            ),
            vllm_tensor_parallel_size=(
                widgets["vllm_tensor_parallel_size"].value()
            ),
            vllm_quantization=str(
                widgets["vllm_quantization"].currentData()
                if widgets["vllm_quantization"].currentData() is not None
                else widgets["vllm_quantization"].currentText()
            ).strip(),
            vllm_trust_remote_code=(
                widgets["vllm_trust_remote_code"].isChecked()
            ),
            vllm_enforce_eager=(
                widgets["vllm_enforce_eager"].isChecked()
            ),
        )

    def _on_backend_changed(self, role: str) -> None:
        """Swap the selector between GGUF and cached Hugging Face models."""
        widgets = self._widgets[role]
        backend = str(widgets["backend"].currentData() or "llama_cpp")
        combo = widgets["model"]
        previous = self._selected_model(widgets)
        choices = sorted(self._model_choices.get(backend, set()))
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("(nessun modello)", "")
        for name in choices:
            combo.addItem(name, name)
        index = combo.findData(previous)
        if index < 0 and choices:
            index = 1
        combo.setCurrentIndex(max(0, index))
        combo.setEditable(backend == "vllm")
        combo.blockSignals(False)
        widgets["vllm_group"].setVisible(backend == "vllm")
        widgets["speculative_decoding"].setVisible(backend == "llama_cpp")
        self._on_model_changed(role)

    def _on_model_changed(self, role: str, initial: bool = False) -> None:
        widgets = self._widgets[role]
        backend = str(widgets["backend"].currentData() or "llama_cpp")
        widgets["vllm_group"].setVisible(backend == "vllm")
        widgets["speculative_decoding"].setVisible(backend == "llama_cpp")
        model = self._selected_model(widgets)
        if not model:
            widgets["max_context"].setText("Contesto massimo: —")
            widgets["context_length"].setMaximum(2_000_000)
            widgets["test"].setEnabled(False)
            widgets["optimize"].setEnabled(False)
            widgets["unload"].setEnabled(False)
            widgets["rec_label"].setVisible(False)
            self._set_status(role, "non_selezionato")
            self._refresh_worker_options(role)
            self._refresh_output_guidance(role)
            self._refresh_runtime_statuses()
            return
        config = self._collect_config(role)
        try:
            model_available = LlmClient(config=config).is_available
        except Exception:
            model_available = False
        if not model_available:
            widgets["max_context"].setText(
                "Contesto massimo: modello non installato"
            )
            widgets["context_length"].setMaximum(2_000_000)
            widgets["test"].setEnabled(False)
            widgets["optimize"].setEnabled(False)
            widgets["unload"].setEnabled(False)
            widgets["rec_label"].setVisible(False)
            location = (
                "cache Hugging Face locale"
                if backend == "vllm" else "archivio GGUF locale"
            )
            self._set_status(
                role, "errore", f"Modello non presente nella {location}"
            )
            self._refresh_worker_options(role)
            self._refresh_output_guidance(role)
            self._refresh_runtime_statuses()
            return

        widgets["test"].setEnabled(role not in self._workers)
        widgets["optimize"].setEnabled(backend == "llama_cpp")
        widgets["unload"].setEnabled(False)
        try:
            cache_key = (backend, model)
            capabilities = self._capability_cache.get(cache_key)
            if capabilities is None:
                capabilities = LlmClient(config=config).model_capabilities()
                self._capability_cache[cache_key] = capabilities
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
            self._refresh_worker_options(role)
            self._refresh_output_guidance(role)
            self._refresh_runtime_statuses()
            return
        self._refresh_runtime_status(role)
        self._refresh_worker_options(role)
        self._refresh_output_guidance(role)

        # Auto-optimize when a model is first selected (only if the user
        # hasn't already manually changed parameters).
        if not initial and model and backend == "llama_cpp":
            self._optimize_params(role, silent=True)

    def _refresh_worker_options(self, role: str) -> None:
        """Populate slot choices using cold-start capacity, not free RAM.

        Free RAM already excludes a resident model and therefore cannot be
        used to subtract the same model weights a second time.
        """
        widgets = self._widgets[role]
        combo = widgets.get("workers_combo")
        if combo is None:
            return

        model = self._selected_model(widgets)
        context = widgets["context_length"].value()

        from ..utils.hardware import (
            get_worker_options,
            get_safe_max_workers,
        )

        combo.blockSignals(True)
        current = combo.currentData()
        combo.clear()

        if not model:
            combo.addItem("1 (nessun modello)", 1)
            combo.blockSignals(False)
            self._refresh_worker_info(role)
            return

        backend = str(widgets["backend"].currentData() or "llama_cpp")
        if backend == "vllm":
            for n in range(1, 9):
                combo.addItem(str(n), n)
            idx = combo.findData(current)
            combo.setCurrentIndex(idx if idx >= 0 else 0)
            combo.blockSignals(False)
            self._refresh_worker_info(role)
            return

        max_safe = get_safe_max_workers(model, context)

        for n in get_worker_options(model, context):
            label = str(n)
            if n > max_safe:
                label = f"{n} (oltre capacità stimata)"
            combo.addItem(label, n)
            if n > max_safe:
                model_idx = combo.count() - 1
                combo.model().item(model_idx).setEnabled(False)

        # Restore previous selection if still valid
        idx = combo.findData(current)
        if idx >= 0 and combo.model().item(idx).isEnabled():
            combo.setCurrentIndex(idx)
        else:
            safe_idx = combo.findData(max_safe)
            combo.setCurrentIndex(safe_idx if safe_idx >= 0 else 0)

        combo.blockSignals(False)
        self._refresh_worker_info(role)

    def _refresh_worker_info(self, role: str) -> None:
        widgets = self._widgets[role]
        info_label = widgets.get("workers_info")
        combo = widgets.get("workers_combo")
        if info_label is None or combo is None:
            return
        model = self._selected_model(widgets)
        if not model:
            info_label.setText("")
            return
        context = widgets["context_length"].value()
        workers = int(combo.currentData() or 1)
        backend = str(widgets["backend"].currentData() or "llama_cpp")
        runtime = getattr(self, "_last_runtime_by_role", {}).get(role)
        if backend == "vllm":
            text = (
                f"vLLM gestirà fino a {workers} richieste concorrenti · "
                f"quota memoria GPU "
                f"{widgets['vllm_gpu_memory_utilization'].value():.0%} · "
                f"tensor parallel {widgets['vllm_tensor_parallel_size'].value()}"
            )
            if runtime:
                text += (
                    f"\nRuntime caricato: {runtime.get('slots') or '?'} "
                    "sequenze massime"
                )
            info_label.setText(text)
            info_label.setToolTip(
                "vLLM pianifica dinamicamente le sequenze entro la quota di "
                "memoria GPU configurata."
            )
            return
        from ..utils.hardware import (
            estimate_server_ram_gb,
            get_available_ram_gb,
            get_model_size_gb,
            get_total_ram_gb,
        )

        total = get_total_ram_gb()
        available = get_available_ram_gb()
        model_size = get_model_size_gb(model)
        estimated = estimate_server_ram_gb(model, context, workers)
        text = (
            f"Capacità {total:.1f} GiB · liberi/reclamabili ora "
            f"{available:.1f} GiB (dopo i processi residenti)"
        )
        if model_size is not None:
            text += (
                f" · requisito macchina a freddo ~{estimated:.1f} GiB "
                f"(pesi {model_size:.1f} GiB, {workers} slot e riserva sistema)"
            )
        if runtime:
            text += (
                f"\nRuntime caricato: {runtime.get('slots') or '?'} slot × "
                f"{self._format_integer(runtime.get('context_length') or context)}"
            )
        info_label.setText(text)
        info_label.setToolTip(
            "La RAM disponibile è una misura istantanea e include già il "
            "costo dei modelli residenti; non viene usata per sottrarre "
            "nuovamente gli stessi pesi."
        )

    def _on_runtime_shape_changed(self, role: str) -> None:
        self._refresh_worker_options(role)
        self._refresh_output_guidance(role)
        self._refresh_runtime_statuses()

    def _refresh_output_guidance(self, role: str) -> None:
        """Warn visibly when likely long outputs have too little headroom."""
        widgets = self._widgets[role]
        warning = widgets["output_warning"]
        model = self._selected_model(widgets)
        if not model:
            warning.setVisible(False)
            return
        from ..utils.hardware import recommend_output_tokens

        recommended, _ = recommend_output_tokens(
            widgets["context_length"].value(), role
        )
        configured = widgets["max_output_tokens"].value()
        if configured >= recommended:
            warning.setVisible(False)
            return
        workload = {
            "document": "normalizzazioni documentali lunghe",
            "atomic_evidence": "estrazioni JSON con molte evidenze",
            "clinical_events": "fusioni e assemblaggi di episodi complessi",
            "clinical_state": "registri estesi e analisi irAE complete",
        }[role]
        warning.setText(
            f"⚠ Limite inferiore al valore consigliato "
            f"({self._format_integer(recommended)} token): {workload} "
            "possono essere troncati. Usa “Ottimizza” oppure aumentalo "
            "manualmente."
        )
        warning.setVisible(True)

    def _runtime_identity(self, config: LLMRoleConfig) -> tuple:
        if not config.model:
            return (
                "", config.context_length, config.parallel_workers,
                "ngram-cache" if config.speculative_decoding else "none",
            )
        try:
            return LlmClient(config=config).runtime_identity()
        except Exception:
            # Keeps the dialog usable for a configured but missing model.
            return (
                f"{config.backend}:{config.model}",
                config.context_length,
                config.parallel_workers,
                (
                    "ngram-cache" if config.speculative_decoding else
                    ("vllm" if config.backend == "vllm" else "none")
                ),
            )

    def _runtime_snapshot(self) -> tuple[dict[str, tuple], dict[str, dict | None]]:
        identities: dict[str, tuple] = {}
        runtime_cache: dict[tuple, dict | None] = {}
        runtime_by_role: dict[str, dict | None] = {}
        for role in self._widgets:
            config = self._collect_config(role)
            if not config.model:
                continue
            try:
                if not LlmClient(config=config).is_available:
                    continue
            except Exception:
                continue
            identity = self._runtime_identity(config)
            identities[role] = identity
            if identity not in runtime_cache:
                try:
                    runtime_cache[identity] = LlmClient(
                        config=config
                    ).loaded_model_info()
                except Exception:
                    runtime_cache[identity] = None
            runtime_by_role[role] = runtime_cache[identity]
        return identities, runtime_by_role

    def _topology_ram_estimate(
        self, configs: dict[str, LLMRoleConfig] | None = None
    ) -> tuple[float, float, float]:
        """Return LLM runtime estimate, system reserve and total capacity."""
        from ..utils.hardware import (
            estimate_runtime_ram_gb,
            get_system_ram_reserve_gb,
            get_total_ram_gb,
        )

        unique: dict[tuple, LLMRoleConfig] = {}
        for config in (configs or self.configurations()).values():
            if config.model:
                unique.setdefault(self._runtime_identity(config), config)
        # The GGUF estimator cannot infer VRAM requirements for arbitrary
        # Hugging Face checkpoints.  vLLM enforces its configured GPU quota
        # itself; only llama.cpp runtimes participate in this RAM warning.
        llm_ram = sum(
            estimate_runtime_ram_gb(
                config.model, identity[1], identity[2]
            )
            for identity, config in unique.items()
            if config.backend == "llama_cpp"
        )
        return llm_ram, get_system_ram_reserve_gb(), get_total_ram_gb()

    def _update_runtime_summary(
        self,
        identities: dict[str, tuple] | None = None,
        runtime_by_role: dict[str, dict | None] | None = None,
    ) -> None:
        if identities is None or runtime_by_role is None:
            identities, runtime_by_role = self._runtime_snapshot()
        groups: dict[tuple, list[str]] = {}
        for role, identity in identities.items():
            groups.setdefault(identity, []).append(role)
        role_label = ROLE_TAB_LABELS
        lines = []
        actual_slot_parts = []
        for identity, roles in groups.items():
            runtime = runtime_by_role.get(roles[0])
            group_config = self._collect_config(roles[0])
            model = group_config.model
            backend_label = (
                "vLLM" if group_config.backend == "vllm" else "llama.cpp"
            )
            shared = len(roles) > 1
            slots = int(runtime.get("slots") or identity[2]) if runtime else identity[2]
            context = int(
                runtime.get("context_length") or identity[1]
            ) if runtime else identity[1]
            active = int(runtime.get("active_slots") or 0) if runtime else 0
            state = "caricato" if runtime else "configurato, non caricato"
            sharing = "server condiviso" if shared else "server dedicato"
            speculation = (
                ", speculazione n-gram"
                if len(identity) > 3 and identity[3] == "ngram-cache" else ""
            )
            lines.append(
                f"<b>{model}</b> [{backend_label}]: {sharing}, {slots} slot × "
                f"{self._format_integer(context)} token, {state}{speculation}"
                + (f", {active} in uso" if active else "")
                + " — " + ", ".join(role_label[role] for role in roles)
            )
            if runtime:
                actual_slot_parts.append(
                    f"{model}: {slots} reali"
                    + (f" ({active} occupati)" if active else "")
                )

        # Editing model/context/slots does not reshape an already running
        # process until "Salva e applica". Surface that previous runtime so
        # the form never suggests that its memory has already been released.
        current_identities = set(identities.values())
        previous_seen: set[tuple] = set()
        has_previous_runtime = False
        previous_runtime_ram = 0.0
        for config in self._initial_configs.values():
            if not config.model:
                continue
            previous_identity = self._runtime_identity(config)
            if (
                previous_identity in current_identities
                or previous_identity in previous_seen
            ):
                continue
            previous_seen.add(previous_identity)
            try:
                previous_runtime = LlmClient(
                    config=config
                ).loaded_model_info()
            except Exception:
                previous_runtime = None
            if not previous_runtime:
                continue
            has_previous_runtime = True
            previous_slots = int(
                previous_runtime.get("slots") or previous_identity[2]
            )
            previous_context = int(
                previous_runtime.get("context_length") or previous_identity[1]
            )
            from ..utils.hardware import estimate_runtime_ram_gb
            previous_runtime_ram += estimate_runtime_ram_gb(
                config.model, previous_context, previous_slots
            )
            previous_active = int(
                previous_runtime.get("active_slots") or 0
            )
            lines.append(
                f"<b>{config.model}</b>: runtime precedente ancora residente, "
                f"{previous_slots} slot × "
                f"{self._format_integer(previous_context)} token"
                + (f", {previous_active} in uso" if previous_active else "")
                + "; verrà sostituito applicando la configurazione"
            )
            actual_slot_parts.append(
                f"{config.model} (precedente): {previous_slots} reali"
                + (f" ({previous_active} occupati)" if previous_active else "")
            )

        configured_ram, system_reserve, total_capacity = (
            self._topology_ram_estimate()
        )
        estimated_requirement = (
            configured_ram + previous_runtime_ram + system_reserve
            if configured_ram or previous_runtime_ram else 0.0
        )
        capacity_exceeded = estimated_requirement > total_capacity
        if estimated_requirement:
            qualifier = "temporaneo" if has_previous_runtime else "configurato"
            lines.append(
                f"Stima complessiva {qualifier}: "
                f"<b>~{estimated_requirement:.1f} GiB</b> inclusa la riserva "
                f"per sistema/app, su {total_capacity:.1f} GiB fisici."
            )

        same_weights_duplicated = (
            len(groups) > 1
            and len({identity[0] for identity in groups}) < len(groups)
        )
        if capacity_exceeded:
            heading = (
                "⚠️ <b>Fabbisogno stimato oltre la capacità fisica</b>: "
                "riduci contesto/slot o evita server distinti."
            )
            color = "#8b1e1e; background:#fdecec; border:1px solid #e5a5a5;"
        elif same_weights_duplicated:
            duplicated_kind = (
                "GGUF"
                if all(
                    self._collect_config(role).backend == "llama_cpp"
                    for roles in groups.values() for role in roles
                )
                else "modello"
            )
            heading = (
                f"⚠️ <b>Lo stesso {duplicated_kind} richiede più server "
                "fisici</b>: contesto "
                "o slot non coincidono e i pesi verrebbero caricati più volte."
            )
            color = "#7f6000; background:#fff4ce; border:1px solid #e5c365;"
        elif has_previous_runtime:
            heading = (
                "⚠️ <b>Modifiche non ancora applicate</b>: il runtime "
                "precedente è tuttora residente in memoria."
            )
            color = "#7f6000; background:#fff4ce; border:1px solid #e5c365;"
        elif len(groups) == 1 and groups:
            only_roles = next(iter(groups.values()))
            heading = (
                "✓ <b>Un solo server fisico condiviso</b>."
                if len(only_roles) > 1
                else "<b>Un solo server fisico configurato</b>."
            )
            color = "#1e6b3a; background:#eaf7ee; border:1px solid #9bd3aa;"
        elif groups:
            heading = f"<b>{len(groups)} server fisici distinti</b>."
            color = "#2c3e50; background:#eef3f7; border:1px solid #bdcbd6;"
        else:
            heading = "Nessun server configurato."
            color = "#5d6d7e; background:#f4f6f7; border:1px solid #d5dbdb;"
        self._runtime_summary.setText(
            heading + ("<br>" + "<br>".join(lines) if lines else "")
        )
        self._runtime_summary.setStyleSheet(
            f"color:{color} border-radius:4px; padding:6px;"
        )
        self._align_runtime_button.setVisible(same_weights_duplicated)
        self._slots_label.setText(
            "Slot reali: " + (
                " · ".join(actual_slot_parts)
                if actual_slot_parts else "nessun server caricato"
            )
        )

    def _align_shared_runtime(self) -> None:
        configs = self.configurations()
        active = [role for role, config in configs.items() if config.model]
        if len(active) < 2:
            return
        identities = {role: self._runtime_identity(configs[role]) for role in active}
        if len({identity[0] for identity in identities.values()}) != 1:
            return
        target_context = max(configs[role].context_length for role in active)
        target_workers = max(configs[role].parallel_workers for role in active)
        target_speculation = any(
            configs[role].speculative_decoding for role in active
        )
        for role in active:
            self._widgets[role]["context_length"].setValue(target_context)
            self._refresh_worker_options(role)
            combo = self._widgets[role]["workers_combo"]
            index = combo.findData(target_workers)
            if index >= 0 and combo.model().item(index).isEnabled():
                combo.setCurrentIndex(index)
            self._widgets[role]["speculative_decoding"].setChecked(
                target_speculation
            )
        self._refresh_runtime_statuses()

    def _optimize_params(self, role: str, silent: bool = False) -> None:
        """Analyse hardware + model and fill recommended parameters.

        When *silent* is True the recommendation label is updated without
        a popup dialog (used on initial model selection).
        """
        widgets = self._widgets[role]
        model = self._selected_model(widgets)
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

        # The same GGUF should keep one physical runtime. Synchronize only
        # context and slots; generation parameters remain role-specific.
        role_identity = self._runtime_identity(self._collect_config(role))[0]
        synchronized = []
        for other_role in self._widgets:
            if other_role == role:
                continue
            other = self._collect_config(other_role)
            if not other.model or self._runtime_identity(other)[0] != role_identity:
                continue
            self._widgets[other_role]["context_length"].setValue(
                rec.context_length
            )
            self._refresh_worker_options(other_role)
            other_combo = self._widgets[other_role]["workers_combo"]
            other_index = other_combo.findData(rec.parallel_workers)
            if (
                other_index >= 0
                and other_combo.model().item(other_index).isEnabled()
            ):
                other_combo.setCurrentIndex(other_index)
            synchronized.append(ROLE_DEFINITIONS[other_role][0])

        # Build recommendation text
        lines = [
            "<b>⚡ Parametri ottimizzati automaticamente</b>",
            f"• Contesto: {self._format_integer(rec.context_length)} token — "
            f"{rec.context_rationale}",
            f"• Token risposta: {self._format_integer(rec.max_output_tokens)} — "
            f"{rec.output_rationale}",
            f"• Slot paralleli: {rec.parallel_workers} — "
            f"{rec.workers_rationale}",
        ]
        if synchronized:
            lines.append(
                "• Runtime condiviso allineato con: "
                + ", ".join(synchronized)
            )
        widgets["rec_label"].setText(
            "<br>".join(lines)
        )
        widgets["rec_label"].setVisible(True)

        if not silent:
            QMessageBox.information(
                self,
                "Parametri ottimizzati",
                f"I parametri per {model} ({ROLE_DEFINITIONS[role][0]}) "
                "sono stati configurati in base all'hardware rilevato.\n\n"
                f"Contesto: {self._format_integer(rec.context_length)} token\n"
                f"Token risposta: {self._format_integer(rec.max_output_tokens)}\n"
                f"Slot paralleli: {rec.parallel_workers}"
                + (
                    "\nRuntime condiviso allineato con: "
                    + ", ".join(synchronized)
                    if synchronized else ""
                ),
            )
        self._refresh_runtime_statuses()

    def _refresh_runtime_statuses(self) -> None:
        identities, runtime_by_role = self._runtime_snapshot()
        self._last_runtime_by_role = runtime_by_role
        groups: dict[tuple, list[str]] = {}
        for role, identity in identities.items():
            groups.setdefault(identity, []).append(role)
        for role, widgets in self._widgets.items():
            model = self._selected_model(widgets)
            unload_button = widgets["unload"]
            test_button = widgets["test"]
            if not model:
                test_button.setEnabled(False)
                unload_button.setEnabled(False)
                self._set_status(role, "non_selezionato")
                continue
            config = self._collect_config(role)
            try:
                available = LlmClient(config=config).is_available
            except Exception:
                available = False
            if not available:
                test_button.setEnabled(False)
                unload_button.setEnabled(False)
                self._set_status(role, "errore", "Modello non installato")
                continue
            identity = identities.get(role)
            runtime = runtime_by_role.get(role)
            shared_roles = groups.get(identity, []) if identity else []
            shared = len(shared_roles) > 1
            processing = bool(runtime and runtime.get("processing"))
            test_button.setEnabled(
                role not in self._workers
                and self._unload_worker is None
                and not processing
            )
            if runtime:
                unload_button.setEnabled(not processing)
                unload_button.setText(
                    "■ Scarica server condiviso"
                    if shared else "■ Scarica questo server"
                )
                tooltip = self._runtime_tooltip(runtime)
                if shared:
                    tooltip += "\nCondiviso con: " + ", ".join(
                        ROLE_DEFINITIONS[item][0] for item in shared_roles
                        if item != role
                    )
                state = (
                    "in_uso_condiviso" if processing and shared
                    else "in_uso" if processing
                    else "caricato_condiviso" if shared
                    else "caricato"
                )
                self._set_status(role, state, tooltip)
            else:
                unload_button.setEnabled(False)
                unload_button.setText("■ Scarica dalla memoria")
                self._set_status(
                    role, "disponibile",
                    "Modello registrato; il runtime configurato non è caricato.",
                )
            self._refresh_worker_info(role)
        self._update_runtime_summary(identities, runtime_by_role)

    def _refresh_runtime_status(self, role: str) -> None:
        self._refresh_runtime_statuses()

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
        try:
            self._warmup_was_loaded[role] = (
                LlmClient(config=config).loaded_model_info() is not None
            )
        except Exception:
            self._warmup_was_loaded[role] = False
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
        if not self._warmup_was_loaded.pop(role, False):
            config = self._collect_config(role)
            self._runtimes_started_in_dialog[
                self._runtime_identity(config)
            ] = config
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
        self._refresh_runtime_statuses()

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
        if resident and not self._warmup_was_loaded.pop(role, False):
            config = self._collect_config(role)
            self._runtimes_started_in_dialog[
                self._runtime_identity(config)
            ] = config
        else:
            self._warmup_was_loaded.pop(role, None)
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
        self._refresh_runtime_statuses()

    def _unload_model(self, role: str) -> None:
        config = self._collect_config(role)
        if not config.model:
            return
        identities, runtime_by_role = self._runtime_snapshot()
        identity = identities.get(role)
        affected = {
            other_role for other_role, other_identity in identities.items()
            if other_identity == identity
        } or {role}
        runtime = runtime_by_role.get(role)
        if runtime and runtime.get("processing"):
            QMessageBox.warning(
                self, "Server in uso",
                "Il server sta elaborando richieste cliniche e non può essere "
                "scaricato. Attendi il completamento dell'operazione.",
            )
            return
        if len(affected) > 1 and QMessageBox.question(
            self,
            "Server condiviso",
            "Questo è un unico server condiviso da:\n- "
            + "\n- ".join(ROLE_DEFINITIONS[item][0] for item in sorted(affected))
            + "\n\nScaricandolo tutti questi ruoli risulteranno non caricati. "
              "Continuare?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        ) != QMessageBox.Yes:
            return
        self._start_unload(role, [config], affected)

    def _update_slots_label(self) -> None:
        """Compatibility wrapper; the summary now uses real /slots data."""
        self._update_runtime_summary()

    def _unload_all_models(self) -> None:
        _, runtime_by_role = self._runtime_snapshot()
        if any(
            runtime and runtime.get("processing")
            for runtime in runtime_by_role.values()
        ):
            QMessageBox.warning(
                self, "Server in uso",
                "Almeno un server sta elaborando richieste cliniche. Attendi "
                "il completamento prima di liberare la memoria.",
            )
            return
        self._start_unload("__all__", None, set(self._widgets))

    def _start_unload(
        self,
        operation: str,
        configs: list[LLMRoleConfig] | None,
        affected_roles: set[str],
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
        self._unload_affected_roles = set(affected_roles)
        for role in affected_roles:
            self._set_role_enabled(role, False)
            if self._widgets[role]["model"].currentData():
                self._set_status(role, "scaricamento")

        worker = _ModelUnloadWorker(operation, configs)
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
                "Memoria liberata parzialmente",
                f"Modelli scaricati: {len(unloaded)}.\n\n"
                f"Errori:\n{details}",
            )
        elif unloaded:
            QMessageBox.information(
                self,
                "Memoria liberata",
                "Modelli scaricati dalla memoria:\n- "
                + "\n- ".join(unloaded),
            )
        else:
            QMessageBox.information(
                self,
                "Memoria già libera",
                "Nessuno dei modelli selezionati era caricato in memoria.",
            )

    def _on_unload_failure(self, operation: str, error: str) -> None:
        QMessageBox.warning(
            self,
            "Impossibile liberare la memoria",
            f"llama-server non ha completato lo scaricamento.\n\n{error}",
        )

    def _release_unload_worker(self) -> None:
        worker = self._unload_worker
        self._unload_worker = None
        if worker is not None:
            worker.deleteLater()
        for role in self._unload_affected_roles:
            self._set_role_enabled(role, True)
        self._unload_affected_roles.clear()
        self._unload_all_button.setEnabled(True)
        self._refresh_runtime_statuses()
        self._runtime_timer.start()

    def _set_role_enabled(self, role: str, enabled: bool) -> None:
        for key, widget in self._widgets[role].items():
            if key in {
                "status", "max_context", "output_warning", "rec_label",
            }:
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
            "caricato_condiviso": ("● caricato · condiviso", "#27ae60"),
            "in_uso": ("● in uso", "#2980b9"),
            "in_uso_condiviso": ("● in uso · condiviso", "#2980b9"),
            "errore": ("● errore", "#c0392b"),
        }
        text, color = states[state]
        label = self._widgets[role]["status"]
        label.setText(text)
        label.setStyleSheet(f"color: {color}; font-weight: bold;")
        label.setToolTip(tooltip or text.lstrip("● "))

    def _validate_role(self, role: str, config: LLMRoleConfig) -> bool:
        maximum = (
            self._capability_cache.get(
                (config.backend, config.model), {}
            ).get(
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
        if (
            self._acceleration_worker is not None
            and self._acceleration_worker.isRunning()
        ):
            QMessageBox.information(
                self,
                "Diagnostica in corso",
                "Attendi pochi secondi il completamento della verifica "
                "CUDA/Metal.",
            )
            return
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
        # Probe both edited and initially loaded runtime shapes. A user may
        # have changed context in the form while the old server is still
        # processing a registry build.
        configs_to_probe = list(configs.values()) + list(
            self._initial_configs.values()
        )
        probed: set[tuple] = set()
        busy = []
        for config in configs_to_probe:
            if not config.model:
                continue
            identity = self._runtime_identity(config)
            if identity in probed:
                continue
            probed.add(identity)
            try:
                runtime = LlmClient(config=config).loaded_model_info()
            except Exception:
                runtime = None
            if runtime and runtime.get("processing"):
                busy.append(
                    f"{config.model}: {runtime.get('active_slots') or '?'} "
                    f"slot in uso"
                )
        if busy:
            QMessageBox.warning(
                self, "Elaborazione LLM in corso",
                "Non è sicuro applicare o riavviare i runtime mentre sono "
                "attive richieste cliniche:\n- " + "\n- ".join(busy)
                + "\n\nAttendi il completamento e riapri questa finestra.",
            )
            return

        identities = {
            role: self._runtime_identity(config)
            for role, config in configs.items() if config.model
        }
        duplicate_weights = (
            len(set(identities.values())) > 1
            and len({identity[0] for identity in identities.values()})
            < len(set(identities.values()))
        )
        if duplicate_weights and QMessageBox.question(
            self,
            "Duplicazione del modello",
            "Lo stesso modello è configurato con parametri motore diversi. "
            "Verranno quindi avviati più server e i pesi saranno caricati "
            "più volte in memoria.\n\nVuoi salvare comunque?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        ) != QMessageBox.Yes:
            return
        llm_ram, system_reserve, total_capacity = (
            self._topology_ram_estimate(configs)
        )
        estimated_requirement = llm_ram + system_reserve if llm_ram else 0.0
        if estimated_requirement > total_capacity and QMessageBox.question(
            self,
            "Memoria probabilmente insufficiente",
            f"La configurazione richiede circa "
            f"{estimated_requirement:.1f} GiB inclusa la riserva per "
            f"sistema/app, ma la macchina dispone di {total_capacity:.1f} "
            "GiB fisici. Il caricamento potrebbe fallire.\n\n"
            "Vuoi salvare comunque?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        ) != QMessageBox.Yes:
            return
        selected_identities = set(identities.values())
        extra_test_configs = [
            config
            for identity, config in self._runtimes_started_in_dialog.items()
            if identity not in selected_identities
        ]
        if extra_test_configs:
            cleanup = LlmClient.unload_runtimes(extra_test_configs)
            errors = cleanup.get("errors") or {}
            if errors:
                QMessageBox.warning(
                    self,
                    "Runtime di test non scaricato",
                    "La configurazione verrà applicata, ma non è stato "
                    "possibile fermare un runtime di test non selezionato:\n- "
                    + "\n- ".join(
                        f"{name}: {error}" for name, error in errors.items()
                    ),
                )
        self._runtime_timer.stop()
        super().accept()

    def reject(self) -> None:
        if (
            self._acceleration_worker is not None
            and self._acceleration_worker.isRunning()
        ):
            QMessageBox.information(
                self,
                "Diagnostica in corso",
                "Attendi pochi secondi il completamento della verifica "
                "CUDA/Metal.",
            )
            return
        if self._workers or self._unload_worker is not None:
            QMessageBox.information(
                self,
                "Operazione in corso",
                "Attendi il completamento dell'operazione prima di chiudere.",
            )
            return
        # "Carica e testa" may have spawned a different context/slot shape.
        # Cancelling must not leave that unselected runtime resident.
        if self._runtimes_started_in_dialog:
            result = LlmClient.unload_runtimes(
                list(self._runtimes_started_in_dialog.values())
            )
            errors = result.get("errors") or {}
            if errors:
                QMessageBox.warning(
                    self,
                    "Runtime di test non scaricato",
                    "La finestra verrà chiusa, ma non è stato possibile "
                    "fermare un runtime avviato solo per il test:\n- "
                    + "\n- ".join(
                        f"{name}: {error}" for name, error in errors.items()
                    ),
                )
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
        backend = str(runtime.get("backend") or "llama_cpp")
        engine = "vLLM" if backend == "vllm" else "llama.cpp"
        lines = [f"Modello caricato nel server {engine} dell'applicazione"]
        memory = cls._format_bytes(runtime.get("size_vram"))
        if memory:
            lines.append(f"Memoria acceleratore/unificata: {memory}")
        if runtime.get("context_length"):
            lines.append(
                "Contesto attualmente caricato: "
                f"{cls._format_integer(runtime['context_length'])} token"
            )
        if runtime.get("slots"):
            lines.append(
                f"Slot reali: {runtime['slots']}"
                + (
                    f" ({runtime.get('active_slots', 0)} attualmente occupati)"
                    if runtime.get("active_slots") else ""
                )
            )
        if runtime.get("expires_at"):
            lines.append(f"Scadenza: {runtime['expires_at']}")
        return "\n".join(lines)


class _AccelerationProbeWorker(QThread):
    """Probe local engines without loading or stopping a model."""

    completed = pyqtSignal(object)

    def run(self) -> None:
        from ..llm_backend.diagnostics import diagnose_local_acceleration

        self.completed.emit(diagnose_local_acceleration())


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
    """Unload selected or all resident local models without blocking Qt."""

    succeeded = pyqtSignal(str, object)
    failed = pyqtSignal(str, str)

    def __init__(
        self, operation: str, configs: list[LLMRoleConfig] | None
    ):
        super().__init__()
        self.operation = operation
        self.configs = configs

    def run(self) -> None:
        try:
            result = (
                LlmClient.unload_models(None)
                if self.configs is None
                else LlmClient.unload_runtimes(self.configs)
            )
            self.succeeded.emit(self.operation, result)
        except Exception as exc:
            self.failed.emit(self.operation, str(exc))
