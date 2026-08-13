"""Client for configurable local Ollama models."""

import json
import re
import time

from ..config import OLLAMA_BASE_URL, DEFAULT_LLM_MODEL_NAME, OLLAMA_CONTEXT_LENGTH
from ..config import OFFLINE_MODE
from ..settings import LLMRoleConfig
from ..security.offline import require_loopback_url
from ..utils.text_utils import fuzzy_find


class LlmClient:
    """
    Client for configurable local models served by Ollama.

    The document-normalization path uses ``generate_text``; the structured
    methods are retained only for the separate Clinical State layer.
    """

    def __init__(self, base_url: str = OLLAMA_BASE_URL,
                 model: str = DEFAULT_LLM_MODEL_NAME,
                 config: LLMRoleConfig | None = None):
        self.base_url = require_loopback_url(base_url) if OFFLINE_MODE else base_url.rstrip("/")
        self.model = config.model if config is not None else model
        self.temperature = config.temperature if config is not None else 0.1
        self.context_length = (
            config.context_length if config is not None else OLLAMA_CONTEXT_LENGTH
        )
        self.max_output_tokens = (
            config.max_output_tokens if config is not None else 4096
        )
        self.top_p = config.top_p if config is not None else 0.9
        self.top_k = config.top_k if config is not None else 40
        self.seed = config.seed if config is not None else 42
        self.keep_alive_minutes = (
            config.keep_alive_minutes if config is not None else 10
        )
        self._available = None  # Lazy check

    @property
    def keep_alive(self) -> str:
        return f"{self.keep_alive_minutes}m"

    @classmethod
    def list_available_models(cls, base_url: str = OLLAMA_BASE_URL) -> list[str]:
        """Return locally installed model names without loading them."""
        import ollama

        host = require_loopback_url(base_url) if OFFLINE_MODE else base_url.rstrip("/")
        response = ollama.Client(host=host).list()
        if hasattr(response, "models"):
            models = response.models
        elif isinstance(response, dict):
            models = response.get("models", [])
        elif isinstance(response, list):
            models = response
        else:
            models = []
        names = []
        for item in models:
            if isinstance(item, dict):
                name = item.get("model") or item.get("name") or ""
            else:
                name = getattr(item, "model", None) or getattr(
                    item, "name", ""
                )
            if name:
                names.append(str(name))
        return sorted(set(names))

    @property
    def is_available(self) -> bool:
        """Check if Ollama server is reachable and model is available."""
        if self._available is None:
            try:
                import ollama
                client = ollama.Client(host=self.base_url)
                resp = client.list()
                # Handle both old (dict) and new (ListResponse) API formats
                if hasattr(resp, 'models'):
                    model_objs = resp.models
                    model_names = [m.model for m in model_objs]
                elif isinstance(resp, dict):
                    model_names = [m.get("name", "") for m in resp.get("models", [])]
                else:
                    model_names = []
                requested = self.model.removesuffix(":latest")
                self._available = any(
                    name.removesuffix(":latest") == requested
                    or name.startswith(f"{requested}:")
                    for name in model_names
                )
            except Exception:
                self._available = False
        return self._available

    @property
    def server_available(self) -> bool:
        """Return whether the local Ollama service is reachable."""
        try:
            import ollama
            ollama.Client(host=self.base_url).list()
            return True
        except Exception:
            return False

    def loaded_model_info(self) -> dict | None:
        """Return runtime information if this model is loaded by Ollama."""
        import ollama

        response = ollama.Client(host=self.base_url).ps()
        return self._find_loaded_model(response, self.model)

    @classmethod
    def unload_models(
        cls,
        model_names: list[str] | tuple[str, ...] | None = None,
        base_url: str = OLLAMA_BASE_URL,
    ) -> dict:
        """Unload selected (or all) resident Ollama models.

        Ollama unloads a runner when it receives an empty generation request
        with ``keep_alive=0``.  The current resident list is read first so this
        method is idempotent and never loads a model merely to unload it.
        """
        import ollama

        host = (
            require_loopback_url(base_url)
            if OFFLINE_MODE else base_url.rstrip("/")
        )
        client = ollama.Client(host=host)
        loaded = cls._loaded_model_names(client.ps())

        if model_names is None:
            targets = loaded
        else:
            requested = {
                cls._normalize_model_name(name)
                for name in model_names if str(name or "").strip()
            }
            targets = [
                name for name in loaded
                if cls._normalize_model_name(name) in requested
            ]

        unloaded = []
        errors = {}
        for model_name in targets:
            try:
                # Use the CLI stop command — it's ~10x faster than the
                # keep_alive=0 generation trick through the Python client.
                import subprocess as _sp
                _sp.run(
                    ["ollama", "stop", model_name],
                    capture_output=True,
                    timeout=10,
                    check=True,
                )
                unloaded.append(model_name)
            except Exception as exc:
                # Fallback: keep_alive=0 generation
                try:
                    client.generate(
                        model=model_name, prompt="", keep_alive=0,
                    )
                    unloaded.append(model_name)
                except Exception as exc2:
                    errors[model_name] = str(exc2)

        return {
            "unloaded": unloaded,
            "not_loaded": (
                []
                if model_names is None else [
                    name for name in model_names
                    if cls._normalize_model_name(name)
                    not in {
                        cls._normalize_model_name(loaded_name)
                        for loaded_name in loaded
                    }
                ]
            ),
            "errors": errors,
        }

    def unload(self) -> bool:
        """Unload this client's model, returning whether it was resident."""
        result = self.unload_models([self.model], self.base_url)
        if result["errors"]:
            error = next(iter(result["errors"].values()))
            raise RuntimeError(error)
        return bool(result["unloaded"])

    def model_capabilities(self) -> dict:
        """Read model metadata, including its declared maximum context."""
        import ollama

        response = ollama.Client(host=self.base_url).show(self.model)
        if hasattr(response, "modelinfo"):
            model_info = response.modelinfo or {}
            capabilities = getattr(response, "capabilities", None) or []
        elif isinstance(response, dict):
            model_info = (
                response.get("model_info")
                or response.get("modelinfo")
                or {}
            )
            capabilities = response.get("capabilities") or []
        else:
            model_info, capabilities = {}, []
        if not isinstance(model_info, dict):
            model_info = dict(model_info)

        architecture = str(model_info.get("general.architecture") or "")
        preferred_key = (
            f"{architecture}.context_length" if architecture else ""
        )
        raw_context = model_info.get(preferred_key) if preferred_key else None
        if raw_context is None:
            candidates = [
                value for key, value in model_info.items()
                if str(key).endswith(".context_length")
            ]
            raw_context = candidates[0] if candidates else None
        maximum_context = self._as_int(raw_context)
        return {
            "model": self.model,
            "architecture": architecture,
            "max_context_length": maximum_context,
            "capabilities": [str(item) for item in capabilities],
        }

    @classmethod
    def _find_loaded_model(cls, response, requested_model: str) -> dict | None:
        """Normalize both object and legacy-dict responses from ``/api/ps``."""
        if hasattr(response, "models"):
            models = response.models
        elif isinstance(response, dict):
            models = response.get("models", [])
        else:
            models = []
        requested = cls._normalize_model_name(requested_model)
        for item in models:
            if isinstance(item, dict):
                getter = item.get
            else:
                getter = lambda key, default=None, obj=item: getattr(
                    obj, key, default
                )
            model_name = getter("model") or getter("name") or ""
            if cls._normalize_model_name(model_name) != requested:
                continue
            expires_at = getter("expires_at")
            return {
                "model": model_name,
                "size": cls._as_int(getter("size")),
                "size_vram": cls._as_int(getter("size_vram")),
                "context_length": cls._as_int(getter("context_length")),
                "expires_at": (
                    expires_at.isoformat()
                    if hasattr(expires_at, "isoformat") else str(expires_at or "")
                ),
            }
        return None

    @classmethod
    def _loaded_model_names(cls, response) -> list[str]:
        """Normalize the resident model names returned by ``/api/ps``."""
        if hasattr(response, "models"):
            models = response.models
        elif isinstance(response, dict):
            models = response.get("models", [])
        else:
            models = []

        names = []
        for item in models:
            if isinstance(item, dict):
                name = item.get("model") or item.get("name")
            else:
                name = (
                    getattr(item, "model", None)
                    or getattr(item, "name", None)
                )
            if name and name not in names:
                names.append(str(name))
        return names

    def warmup(self, keep_alive: str | None = None) -> dict:
        """Load and test the model with a tiny, non-clinical request."""
        import ollama

        active_keep_alive = keep_alive or self.keep_alive
        started = time.perf_counter()
        client = ollama.Client(host=self.base_url)
        response = client.chat(
            model=self.model,
            messages=[{
                "role": "user",
                "content": "Test tecnico di disponibilità. Rispondi soltanto OK.",
            }],
            think=False,
            options={
                "temperature": 0,
                "num_predict": 8,
                # Use the same context configured for real processing, so the
                # first clinical call does not have to reload a different runner.
                "num_ctx": self.context_length,
            },
            keep_alive=active_keep_alive,
        )
        if hasattr(response, "message"):
            content = response.message.content or ""
        elif isinstance(response, dict):
            message = response.get("message", {})
            content = message.get("content", "") if isinstance(message, dict) else ""
        else:
            content = ""
        if not str(content).strip():
            raise RuntimeError("Il modello ha restituito una risposta di test vuota")
        runtime = self.loaded_model_info()
        if runtime is None:
            raise RuntimeError(
                "Il test ha risposto, ma Ollama non segnala il modello in memoria"
            )
        runtime["elapsed_seconds"] = time.perf_counter() - started
        runtime["test_response"] = str(content).strip()[:100]
        runtime["keep_alive"] = active_keep_alive
        return runtime

    @staticmethod
    def _normalize_model_name(value: str) -> str:
        return str(value or "").strip().removesuffix(":latest")

    @staticmethod
    def _as_int(value) -> int | None:
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def _ollama_generate(self, prompt: str, system: str = "",
                         stream: bool = False,
                         format_schema: dict | str | None = None) -> str:
        """Internal: call Ollama Chat API (disables thinking/reasoning tokens)."""
        import ollama
        client = ollama.Client(host=self.base_url)

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        request = {
            "model": self.model,
            "messages": messages,
            # Reasoning text is not part of the clinical document and wastes
            # context on models that support a separate thinking channel.
            "think": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_output_tokens,
                "num_ctx": self.context_length,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "seed": self.seed,
            },
            "keep_alive": self.keep_alive,
        }
        if format_schema is not None:
            request["format"] = format_schema
        response = client.chat(**request)
        done_reason = (
            getattr(response, "done_reason", None)
            if not isinstance(response, dict)
            else response.get("done_reason")
        )
        if done_reason == "length":
            raise RuntimeError(
                "Ollama ha raggiunto il limite di token in output "
                f"(max_output_tokens={self.max_output_tokens}); "
                "aumentalo in Configura LLM"
            )
        # Handle both old (dict) and new (ChatResponse) API
        if hasattr(response, 'message'):
            return response.message.content or ""
        if isinstance(response, dict):
            msg = response.get("message", {})
            if isinstance(msg, dict):
                return msg.get("content", "")
        return ""

    def generate_structured(self, prompt: str, system: str,
                            schema: dict) -> dict:
        """Generate locally with an Ollama-enforced JSON schema."""

        response = self._ollama_generate(
            prompt, system, format_schema=schema
        )
        return json.loads(response)

    def extract_patient_identity(self, text: str) -> dict:
        """Extract the patient's identity fields from raw document text.

        Runs on the pre-anonymization text: the model must be able to read
        the actual name/CF/birth date so the result can confirm (or refute)
        the workspace attribution made by the deterministic routing.

        Returns a dict with keys ``name``, ``birth_date``, ``fiscal_code``
        and ``confidence``; an empty dict when the call fails or the model
        could not produce valid JSON.
        """
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "birth_date": {"type": "string"},
                "fiscal_code": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": ["name", "birth_date", "fiscal_code"],
        }
        system_prompt = (
            "Sei un assistente che estrae l'identità anagrafica del paziente "
            "da un documento clinico. Rispondi SOLO con JSON valido, senza "
            "altro testo."
        )
        user_prompt = f"""Estrai l'identità del PAZIENTE a cui si riferisce il documento.

Campi:
- name: nome e cognome completi del paziente, come scritti nel documento
- birth_date: data di nascita in formato YYYY-MM-DD, se presente
- fiscal_code: codice fiscale a 16 caratteri, se presente; altrimenti stringa vuota
- confidence: la tua confidenza (0.0-1.0)

Regole:
- È rilevante SOLO il paziente: ignora medici, infermieri, referenti e
  ogni altra persona citata nel testo.
- Se non puoi determinare il paziente con sicurezza, lascia i campi vuoti.
- Non inventare o correggere alcun valore.

TESTO:
{text[:4000]}
"""
        try:
            data = self.generate_structured(
                user_prompt, system_prompt, schema
            )
        except Exception:
            return {}
        if not isinstance(data, dict):
            return {}
        try:
            confidence = float(data.get("confidence", 0.5) or 0.5)
        except (TypeError, ValueError):
            confidence = 0.5
        return {
            "name": (data.get("name") or "").strip(),
            "birth_date": (data.get("birth_date") or "").strip(),
            "fiscal_code": (data.get("fiscal_code") or "").strip().upper(),
            "confidence": confidence,
        }

    def generate_text(self, prompt: str, system: str = "") -> str:
        """Generate plain text without JSON/schema constraints."""
        response = self._ollama_generate(prompt, system, format_schema=None)
        if not str(response or "").strip():
            raise ValueError("Ollama ha restituito una risposta vuota")
        return str(response)

    def extract_clinical_events(self, text: str, patient_id: str,
                                document_id: str) -> list:
        """
        Extract clinical events from text using the configured LLM.
        Returns a list of ClinicalEvent-compatible dicts.
        """
        from ..models.clinical_event import ClinicalEvent, EventType

        system_prompt = (
            "Sei un assistente clinico specializzato nell'estrazione di "
            "informazioni da documentazione medica in lingua italiana. "
            "Rispondi SOLO con JSON valido, senza testo aggiuntivo."
        )

        user_prompt = f"""Analizza il seguente testo clinico ed estrai tutti gli eventi clinici in formato JSON.

Classifica ogni evento secondo questi tipi:
- diagnosis: diagnosi formulate
- treatment_started: inizio di una terapia
- treatment_ended: fine di una terapia
- treatment_modified: modifica di una terapia
- progression: progressione di malattia
- response: risposta a terapia
- hospitalization: ricovero ospedaliero
- discharge: dimissione
- procedure: procedura medica
- surgery: intervento chirurgico
- toxicity: tossicità / effetto avverso
- adverse_event: evento avverso
- symptom: sintomo rilevante
- lab_alteration: alterazione significativa di esami di laboratorio
- radiology_finding: reperto radiologico significativo
- consultation: consulenza specialistica
- death: decesso
- follow_up: follow-up programmato
- prescription: prescrizione
- other: altro evento clinico

Per ogni evento includi OBBLIGATORIAMENTE:
- event_type: uno dei tipi sopra
- event_date: in formato YYYY-MM-DD (o la data più precisa disponibile)
- entity: il nome del farmaco, diagnosi, procedura, etc.
- source_text: il testo ESATTO dal documento che supporta questo evento

Campi opzionali:
- value: valore numerico se applicabile
- unit: unità di misura se applicabile
- confidence: la tua confidenza (0.0-1.0)

Esempio di output:
```json
{{
  "events": [
    {{
      "event_type": "treatment_started",
      "event_date": "2026-04-12",
      "entity": "nivolumab",
      "value": null,
      "unit": null,
      "source_text": "Si avvia trattamento con nivolumab 240 mg ogni 2 settimane",
      "confidence": 0.95
    }}
  ]
}}
```

TESTO DA ANALIZZARE:
{text[:5000]}
"""

        try:
            response = self._ollama_generate(user_prompt, system_prompt)

            # Extract JSON from response
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                response = json_match.group(1)

            data = json.loads(response)
            raw_events = data.get("events", [])

        except Exception:
            return []

        # Convert to ClinicalEvent objects with validation
        events = []
        for i, ev in enumerate(raw_events):
            source_text = ev.get("source_text", "")

            # Anti-hallucination: verify source text exists in original
            if source_text and not fuzzy_find(text, source_text, threshold=0.7):
                continue  # Skip hallucinated events

            event_date = ev.get("event_date", "")
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(event_date)
                if dt.year < 1900 or dt.year > 2100:
                    continue
            except (ValueError, TypeError):
                pass  # Keep even approximate dates

            event = ClinicalEvent(
                event_id=f"EVT_{i + 1:06d}",  # Will be replaced by EventStore
                patient_id=patient_id,
                event_date=event_date,
                event_type=ev.get("event_type", "other"),
                entity=ev.get("entity", ""),
                value=ev.get("value"),
                unit=ev.get("unit"),
                status="proposed",
                source_document_id=document_id,
                source_text=source_text,
                confidence=float(ev.get("confidence", 0.5)),
            )
            events.append(event)

        return events

    def propose_clinical_state_delta(self, current_state: dict,
                                     new_events: list[dict]) -> dict:
        """
        Ask the Clinical State LLM to propose a delta based on new events.
        """
        system_prompt = (
            "Sei un assistente clinico. Confronta lo stato clinico attuale "
            "con nuovi eventi e proponi modifiche. "
            "Rispondi SOLO con JSON valido."
        )

        user_prompt = f"""Confronta lo stato clinico attuale con i nuovi eventi e proponi un delta JSON.

Stato attuale:
{json.dumps(current_state, ensure_ascii=False, indent=2)[:2000]}

Nuovi eventi:
{json.dumps(new_events, ensure_ascii=False, indent=2)[:2000]}

Produci un JSON con questa struttura:
```json
{{
  "delta": {{
    "add": [],        // nuovi elementi
    "update": [],     // elementi da modificare
    "close": [],      // elementi da chiudere
    "confirm": [],    // elementi confermati
    "conflict": [],   // conflitti da risolvere
    "ignore_as_duplicate": []  // duplicati
  }}
}}
```
"""

        try:
            response = self._ollama_generate(user_prompt, system_prompt)
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                response = json_match.group(1)
            data = json.loads(response)
            return data.get("delta", {})
        except Exception:
            return {"add": [], "update": [], "close": [],
                    "confirm": [], "conflict": [], "ignore_as_duplicate": []}

    def query_clinical_state(self, clinical_state: dict,
                             events: list[dict], question: str) -> str:
        """Query specialized state views plus provenance-bearing evidence."""
        system_prompt = (
            "Sei un assistente clinico esperto. Rispondi alla domanda "
            "basandoti ESCLUSIVAMENTE sui dati forniti. "
            "Se un dato non è disponibile, dichiaralo esplicitamente. "
            "Non inventare informazioni. Cita le date quando disponibili."
        )

        state_views = dict(clinical_state)
        observations = state_views.pop("observations", [])
        document_projections = state_views.pop("document_projections", [])
        selected_observations = self._select_relevant_observations(
            observations, question
        )
        selected_document_context = self._select_document_context(
            document_projections, selected_observations
        )
        # lab_trends duplicates deterministic evidence already present in the
        # observation projection and needlessly consumes model context.
        state_views.pop("lab_trends", None)
        context = {
            "clinical_state_views": state_views,
            "selected_evidence": selected_observations,
            "document_level_context": selected_document_context,
            "recent_clinical_events": events[-100:],
        }
        serialized_context = json.dumps(
            context, ensure_ascii=False, separators=(",", ":")
        )

        user_prompt = f"""Basandoti sui seguenti dati clinici, rispondi alla domanda.

DATI CLINICI:
{serialized_context}

DOMANDA: {question}

Rispondi in modo chiaro e conciso, citando date e fonti quando disponibili.
"""

        try:
            return self._ollama_generate(user_prompt, system_prompt)
        except Exception as e:
            return f"❌ Errore nell'interrogazione: {e}"

    @staticmethod
    def _select_relevant_observations(
        observations: list[dict], question: str, max_chars: int = 65000
    ) -> list[dict]:
        """Deterministic first-pass retrieval without external embeddings."""
        normalized_question = question.lower()
        query_terms = {
            token for token in re.findall(r"[a-zà-ÿ0-9_-]{3,}", normalized_question)
            if token not in {
                "che", "con", "dei", "del", "della", "delle", "gli", "nel",
                "nella", "nelle", "per", "tutti", "tutte", "una", "uno",
                "come", "sono", "stato", "durante", "effettuate", "riporta",
            }
        }
        hinted_categories = set()
        category_hints = {
            "avvers": {
                "toxicity", "adverse_event", "symptom", "laboratory_finding",
                "radiology_finding", "medication_current", "treatment_started",
                "treatment_modified", "treatment_ended", "prescription",
            },
            "tossicit": {"toxicity", "adverse_event", "laboratory_finding"},
            "immun": {
                "toxicity", "adverse_event", "symptom", "laboratory_finding",
                "medication_current", "treatment_started", "treatment_modified",
            },
            "radiolog": {"radiology_finding", "response", "progression"},
            "strument": {"radiology_finding", "response", "progression", "procedure"},
            "recist": {"radiology_finding", "response", "progression"},
            "terapi": {
                "medication_current", "treatment_started", "treatment_modified",
                "treatment_ended", "prescription", "response", "progression",
            },
            "farmac": {
                "medication_current", "treatment_started", "treatment_modified",
                "treatment_ended", "prescription",
            },
            "laborator": {"laboratory_finding", "lab_alteration"},
            "ematochim": {"laboratory_finding", "lab_alteration"},
            "specialist": {"consultation", "physical_exam", "care_plan"},
        }
        for stem, categories in category_hints.items():
            if stem in normalized_question:
                hinted_categories.update(categories)

        scored = []
        for index, observation in enumerate(observations):
            searchable = " ".join(str(observation.get(field) or "").lower() for field in (
                "category", "entity", "source_text", "clinical_status",
                "value_text", "date",
            ))
            term_matches = sum(term in searchable for term in query_terms)
            category_match = observation.get("category") in hinted_categories
            score = (4 if category_match else 0) + term_matches
            scored.append((score, index, observation))

        relevant = [item for item in scored if item[0] > 0]
        if not relevant:
            # A generic question receives the latest evidence first while
            # preserving stable source order for equal dates.
            relevant = scored[-250:]
        relevant.sort(key=lambda item: (-item[0], item[1]))

        selected = []
        consumed = 0
        for _, _, observation in relevant:
            size = len(json.dumps(observation, ensure_ascii=False))
            if selected and consumed + size > max_chars:
                break
            selected.append(observation)
            consumed += size
        return sorted(
            selected,
            key=lambda item: (
                item.get("date") or "", item.get("source_document_id") or "",
                item.get("source_page") or 0,
            ),
        )

    @staticmethod
    def _select_document_context(
        projections: list[dict], selected_evidence: list[dict]
    ) -> list[dict]:
        """Keep cross-page groups and relations for retrieved evidence only."""
        selected_ids = {
            item.get("evidence_id") for item in selected_evidence
            if item.get("evidence_id")
        }
        context = []
        for projection in projections:
            all_observations = projection.get("observations", [])
            observations = [
                observation for observation in all_observations
                if selected_ids.intersection(observation.get("evidence_ids", []))
            ]
            if not observations:
                continue
            observation_ids = {
                item.get("observation_id") for item in observations
            }
            relationships = [
                relationship
                for relationship in projection.get("relationships", [])
                if (relationship.get("from_observation_id") in observation_ids or
                    relationship.get("to_observation_id") in observation_ids)
            ]
            connected_ids = set(observation_ids)
            for relationship in relationships:
                connected_ids.add(relationship.get("from_observation_id"))
                connected_ids.add(relationship.get("to_observation_id"))
            observations = [
                observation for observation in all_observations
                if observation.get("observation_id") in connected_ids
            ]
            compact_observations = [{
                key: observation.get(key) for key in (
                    "observation_id", "category", "normalized_entity",
                    "evidence_ids", "observed_start_date", "observed_end_date",
                    "assertions", "clinical_status", "value_text",
                    "numeric_value", "unit", "requires_review",
                )
            } for observation in observations]
            conflicts = [
                conflict for conflict in projection.get("conflicts", [])
                if selected_ids.intersection(conflict.get("evidence_ids", []))
            ]
            context.append({
                "document_id": projection.get("document_id"),
                "document_date": projection.get("document_date"),
                "document_type": projection.get("document_type"),
                "observations": compact_observations,
                "relationships": relationships,
                "conflicts": conflicts,
            })
        return context

    # ------------------------------------------------------------------
    # Clinical Timeline — strictly temporal extraction & deduplication
    # ------------------------------------------------------------------

    _TIMELINE_EXTRACTION_SCHEMA = {
        "type": "object",
        "properties": {
            "entries": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "date_observed": {
                            "type": "string",
                            "description": "Data di prima osservazione YYYY-MM-DD o YYYY-MM",
                        },
                        "date_resolved": {
                            "type": ["string", "null"],
                            "description": "Data di risoluzione se nota, altrimenti null",
                        },
                        "category": {
                            "type": "string",
                            "enum": [
                                "diagnosis", "treatment", "procedure",
                                "surgery", "toxicity", "adverse_event",
                                "imaging_finding", "laboratory", "symptom",
                                "hospitalization", "discharge", "follow_up",
                                "other",
                            ],
                        },
                        "description": {
                            "type": "string",
                            "description": "Descrizione clinica concisa e accurata in italiano",
                        },
                        "status": {
                            "type": "string",
                            "enum": ["active", "resolved", "ongoing"],
                            "description": "Stato dell'osservazione",
                        },
                        "source_text": {
                            "type": "string",
                            "description": "Testo originale dal documento che supporta questa osservazione",
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Confidenza dell'estrazione (0.0-1.0)",
                        },
                    },
                    "required": [
                        "date_observed", "category", "description",
                        "status", "source_text", "confidence",
                    ],
                },
            }
        },
        "required": ["entries"],
    }

    _TIMELINE_DEDUP_SCHEMA = {
        "type": "object",
        "properties": {
            "deduplicated_entries": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "entry_id": {"type": "string"},
                        "date_observed": {"type": "string"},
                        "date_resolved": {"type": ["string", "null"]},
                        "category": {"type": "string"},
                        "description": {"type": "string"},
                        "status": {"type": "string"},
                        "source_document_ids": {
                            "type": "array", "items": {"type": "string"},
                        },
                        "source_texts": {
                            "type": "array", "items": {"type": "string"},
                        },
                        "confidence": {"type": "number"},
                        "enrichment_text": {
                            "type": ["string", "null"],
                            "description": "Dettagli aggiuntivi dalle voci duplicate, o null",
                        },
                    },
                    "required": [
                        "entry_id", "date_observed", "category",
                        "description", "status", "confidence",
                    ],
                },
            },
            "removed_entry_ids": {
                "type": "array",
                "items": {"type": "string"},
            },
            "merge_map": {
                "type": "object",
                "additionalProperties": {"type": "string"},
            },
        },
        "required": ["deduplicated_entries", "removed_entry_ids", "merge_map"],
    }

    def extract_timeline_entries(
        self,
        normalized_text: str,
        registry_summary: str,
        document_date: str | None = None,
        max_text_chars: int | None = None,
    ) -> dict:
        """Extract clinical timeline entries from one normalized document.

        Uses plain-text generation (no JSON schema enforcement) for broader
        model compatibility, then parses the JSON block from the response.
        Falls back gracefully on any parsing error.

        The *max_text_chars* parameter controls how much text is sent to the
        LLM.  When ``None`` (default), a budget is computed from the model's
        ``context_length`` so the full prompt fits in the context window.
        """
        system_prompt = (
            "Sei un assistente clinico specializzato nell'estrazione di "
            "informazioni cliniche rilevanti da documentazione medica in "
            "lingua italiana. Lavori in modo conservativo: estrai solo "
            "informazioni esplicitamente presenti nel testo, senza inferire, "
            "interpretare o dedurre. Rispondi SOLO con un array JSON "
            "circondato da ```json ... ```, senza altro testo."
        )

        doc_date_info = (
            f"Il documento ha data {document_date}. "
            if document_date else ""
        )

        # Compute a context-aware text budget when the caller doesn't
        # specify an explicit limit.
        if max_text_chars is None:
            max_text_chars = self._compute_text_budget()

        user_prompt = f"""Analizza il seguente testo clinico ed estrai le osservazioni clinicamente rilevanti.

{doc_date_info}

REGISTRO CLINICO ATTUALE (per contesto — NON ri-estrarre osservazioni già presenti):
{registry_summary if registry_summary else "(Nessuna informazione pregressa registrata)"}

REGOLE GENERALI:
1. Estrai SOLO informazioni esplicitamente presenti nel testo. Non dedurre, non interpretare.
2. Ogni osservazione DEVE includere: date_observed, category, description, status, source_text, confidence.
3. date_resolved e' opzionale (solo se il testo dice esplicitamente che la condizione si e' risolta).
4. status: "active" (in corso), "resolved" (risolto), "ongoing" (cronico).
5. NON aggiungere osservazioni gia' presenti nel registro.
6. Descrizione concisa ma clinicamente precisa (1-3 frasi), includendo dettagli quantitativi quando disponibili.

CATEGORIE SPECIFICHE:
- diagnosis: diagnosi oncologiche e non, con stadio/grading se noto (es. "Melanoma dorsale, Breslow 2.7mm, Clark IV, BRAF mutato")
- histopathology: referti istopatologici (esame istologico, biopsia, immunoistochimica, marcatori molecolari)
- treatment: terapie farmacologiche con farmaco, dose, via, frequenza, data inizio (es. "Pembrolizumab 180 mg ev ogni 3 settimane dal 24/10/2017")
- treatment_interruption: sospensione/interruzione di una terapia con motivazione
- procedure: procedure diagnostiche o terapeutiche (es. EGDS, endoscopia)
- surgery: interventi chirurgici con data e tipo (es. "Asportazione melanoma dorsale + dissezione ascellare, 2013")
- toxicity: tossicità da trattamento con grado CTCAE se noto
- adverse_event: eventi avversi non necessariamente correlati al trattamento
- imaging_finding: referti radiologici (TC, PET, RM, eco) con sede, esito, RECIST se riportato (es. "TC total-body: progressione, settembre 2017")
- laboratory: SOLO valori di laboratorio ALTERATI (fuori range), includere parametro, valore, unità e range se disponibili
- biomarker: biomarcatori molecolari e loro valore/presenza (es. "BRAF V600E mutato", "PD-L1 60%")
- symptom: sintomi clinicamente rilevanti
- hospitalization: ricoveri ospedalieri con motivo e reparto
- discharge: lettere di dimissione
- follow_up: appuntamenti di controllo programmati
- progression: progressione di malattia documentata
- response: risposta a trattamento (risposta completa, parziale, stabilità, progressione)
- other: altre informazioni clinicamente rilevanti

FORMATO RISPOSTA (ESATTAMENTE così):
```json
[
  {{
    "date_observed": "2024-03-15",
    "date_resolved": null,
    "category": "treatment",
    "description": "Pembrolizumab 180 mg ev ogni 3 settimane",
    "status": "active",
    "source_text": "Dal 24.10.2017 inizia PEMBROLIZUMAB 180 mg ogni 3 settimane",
    "confidence": 0.9
  }}
]
```

TESTO DA ANALIZZARE:
{normalized_text[:max_text_chars]}
"""
        try:
            raw = self.generate_text(user_prompt, system_prompt)
            entries = self._parse_json_entries(raw)
            if not entries and raw.strip():
                # Log the raw response when parsing produced nothing
                import sys as _sys
                preview = raw[:500].replace("\n", "\\n")
                print(
                    f"[extract_timeline_entries] LLM returned "
                    f"{len(raw)} chars but parsed 0 entries. "
                    f"Raw preview: {preview}",
                    file=_sys.stderr,
                )
            return {"entries": entries}
        except Exception as exc:
            import sys as _sys
            print(
                f"[extract_timeline_entries] Error: {exc}",
                file=_sys.stderr,
            )
            return {"entries": []}

    def extract_from_discharge_letter(
        self,
        normalized_text: str,
        document_date: str | None = None,
        registry_summary: str = "",
        max_text_chars: int = 50000,
    ) -> dict:
        """Extract clinical events from a pre-acute discharge letter.

        Uses a specialised prompt that understands the typical structure of
        an Italian discharge letter from a post-acute / rehabilitation ward:
        admission diagnosis, clinical course, therapies, consultations,
        adverse events, discharge outcome, and follow-up plan.

        Returns a dict with key ``"entries"`` containing timeline-ready dicts.
        """
        import re as _re

        text_budget = max(
            4000, min(max_text_chars, len(normalized_text))
        )
        doc_date_info = (
            f"Il documento ha data {document_date}. "
            if document_date else ""
        )
        system_prompt = (
            "Sei un medico che estrae eventi clinici strutturati da una "
            "LETTERA DI DIMISSIONE da un reparto di degenza pre-acuti o "
            "post-acuti (lungodegenza, riabilitazione). Lavori in modo "
            "conservativo: estrai solo informazioni esplicitamente presenti "
            "nel testo. Rispondi SOLO con un array JSON circondato da "
            "```json ... ```, senza altro testo."
        )

        user_prompt = f"""Analizza la seguente LETTERA DI DIMISSIONE ed estrai gli eventi clinicamente rilevanti.

{doc_date_info}

REGISTRO CLINICO ATTUALE (per contesto — NON ri-estrarre osservazioni già presenti):
{registry_summary if registry_summary else "(Nessuna informazione pregressa registrata)"}

STRUTTURA TIPICA DI UNA LETTERA DI DIMISSIONE PRE-ACUTI:
1. Diagnosi di ingresso e diagnosi alla dimissione
2. Motivo del ricovero
3. Decorso clinico (sintesi)
4. Terapie somministrate durante il ricovero
5. Eventi avversi o complicanze insorte
6. Consulenze specialistiche richieste
7. Outcome alla dimissione (migliorato, stabile, trasferito)
8. Terapia prescritta alla dimissione
9. Piano di follow-up

REGOLE OBBLIGATORIE:
- Estrai SOLO informazioni esplicitamente presenti nel testo. Non dedurre, non interpretare.
- Per il DECORSO CLINICO: NON estrarre ogni giorno come evento separato.
  Sintetizza in 1-3 eventi clinicamente significativi (es. "Miglioramento
  progressivo delle condizioni generali", "Complicanza respiratoria in
  giornata X con avvio ossigenoterapia", "Trasferimento in riabilitazione").
- Per le TERAPIE: includi farmaco, dose, via di somministrazione, data inizio
  e data fine/modifica se disponibili. Distingui tra terapia somministrata
  durante il ricovero e terapia prescritta alla dimissione.
- Per gli EVENTI AVVERSI: includi tipo, data, gestione intrapresa.
- Per le CONSULENZE: includi specialità, data, quesito e risposta se presenti.
- Per le DIAGNOSI: distinguere diagnosi di ingresso, diagnosi alla dimissione
  e diagnosi secondarie. Includere stadio/grading se noto.
- Ogni evento DEVE includere source_text (citazione letterale dal documento).
- NON aggiungere osservazioni già presenti nel registro clinico.
- Descrizione concisa ma clinicamente precisa (1-3 frasi).

CATEGORIE (usa una di queste per ogni evento):
- diagnosis: diagnosi (ingresso, dimissione, secondaria)
- treatment: terapia farmacologica iniziata o somministrata
- treatment_completed: terapia completata o sospesa
- procedure: procedura diagnostica o terapeutica
- surgery: intervento chirurgico
- toxicity: tossicità da trattamento
- adverse_event: evento avverso o complicanza
- consultation: consulenza specialistica
- imaging_finding: reperto radiologico significativo
- laboratory: alterazione significativa di esami di laboratorio
- hospitalization: ricovero (data ingresso, reparto, motivo)
- discharge: dimissione (data, outcome, destinazione)
- follow_up: appuntamento di controllo o follow-up programmato
- other: altro evento clinicamente rilevante

FORMATO RISPOSTA (ESATTAMENTE così):
```json
[
  {{
    "date_observed": "2024-03-15",
    "date_resolved": null,
    "category": "discharge",
    "description": "Dimissione con outcome migliorato. Destinazione: domicilio con attivazione ADI.",
    "status": "resolved",
    "source_text": "Il paziente viene dimesso in data 15/03/2024 con outcome migliorato",
    "confidence": 0.95
  }}
]
```

TESTO DA ANALIZZARE:
{normalized_text[:text_budget]}
"""
        try:
            raw = self.generate_text(user_prompt, system_prompt)
            entries = self._parse_json_entries(raw)
            if not entries and raw.strip():
                import sys as _sys
                preview = raw[:500].replace("\n", "\\n")
                print(
                    f"[extract_from_discharge_letter] LLM returned "
                    f"{len(raw)} chars but parsed 0 entries. "
                    f"Raw preview: {preview}",
                    file=_sys.stderr,
                )
            return {"entries": entries}
        except Exception as exc:
            import sys as _sys
            print(
                f"[extract_from_discharge_letter] Error: {exc}",
                file=_sys.stderr,
            )
            return {"entries": []}

    def _compute_text_budget(self) -> int:
        """Estimate how many characters of source text fit in the context.

        Reserve tokens for the system prompt, the user-prompt template,
        the registry summary, and the output (max_output_tokens).  The
        remainder is converted to characters using a conservative 2.5
        chars-per-token estimate.
        """
        ctx = self.context_length or 32768
        output_reserve = min(
            self.max_output_tokens or 4096, max(512, ctx // 2)
        )
        # Prompt template overhead (system + instructions + registry):
        # empirically ~2 500 tokens for the full timeline-extraction template.
        prompt_overhead_tokens = 2500
        available_tokens = max(
            480, ctx - output_reserve - prompt_overhead_tokens
        )
        chars_per_token = 2.5
        return max(4000, int(available_tokens * chars_per_token))

    @staticmethod
    def _parse_json_entries(raw: str) -> list[dict]:
        """Extract a JSON array from an LLM response.

        Handles ```json fences, stray whitespace, and missing outer brackets.
        Returns an empty list on any parse failure.
        """
        import json as _json
        import re as _re

        if not raw or not raw.strip():
            return []

        text = raw.strip()

        # Try to extract from ```json ... ``` fence
        m = _re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if m:
            text = m.group(1).strip()

        # Remove any leading/trailing non-JSON text
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]

        try:
            parsed = _json.loads(text)
            if isinstance(parsed, list):
                # Validate each entry has required fields
                valid = []
                for item in parsed:
                    if isinstance(item, dict) and "description" in item:
                        valid.append(item)
                return valid
            elif isinstance(parsed, dict) and "entries" in parsed:
                return parsed["entries"]
        except (_json.JSONDecodeError, ValueError):
            pass

        return []

    _DEDUP_BATCH_SIZE = 40
    _DEDUP_OVERLAP = 10

    def deduplicate_timeline(self, entries: list[dict]) -> dict:
        """Semantic dedup with sliding window for large registries.

        Sends ALL entries to the LLM using a compact format so the full
        registry fits in the context window (32K tokens ≈ 64K chars).
        The LLM identifies semantically equivalent clinical events and
        synthesises the most complete description from all sources.
        """
        if not entries:
            return {"removed_entry_ids": [], "enrichments": {}}

        if len(entries) <= self._DEDUP_BATCH_SIZE:
            return self._dedup_batch(entries)

        # Sliding-window dedup for large registries.
        # Each batch reuses the deduplicated output of the previous batch
        # so the LLM never sees entries that have already been removed.
        all_removed: set[str] = set()
        all_enrichments: dict[str, str] = {}
        working = list(entries)  # mutable copy — updated after each batch
        cursor = 0  # first entry that still needs dedup

        while cursor < len(working):
            batch_end = min(cursor + self._DEDUP_BATCH_SIZE, len(working))
            batch = working[cursor:batch_end]
            result = self._dedup_batch(batch)

            batch_removed = set(result.get("removed_entry_ids", []))
            all_removed |= batch_removed
            for eid, enrichment in result.get("enrichments", {}).items():
                if not isinstance(enrichment, str):
                    enrichment = str(enrichment)
                all_enrichments[eid] = (
                    all_enrichments.get(eid, "") + " | " + enrichment
                    if eid in all_enrichments else enrichment
                )

            # Rebuild working list: keep-only entries, apply enrichments
            kept = []
            for e in working[:batch_end]:
                if e.get("entry_id") in batch_removed:
                    continue
                enrichment = all_enrichments.get(e.get("entry_id"))
                if enrichment:
                    e = dict(e)
                    e["description"] = (
                        str(e.get("description", ""))
                        + "\n\n[Integrazione: " + enrichment + "]"
                    )
                kept.append(e)
            kept.extend(working[batch_end:])
            working = kept

            # Advance cursor: skip the first (batch_size - overlap) kept
            # entries, which are now considered fully deduplicated.
            new_done = max(1, len(batch) - self._DEDUP_OVERLAP)
            # Adjust for entries that were removed from the batch
            removed_in_batch = sum(
                1 for e in batch if e.get("entry_id") in batch_removed
            )
            cursor += max(1, new_done - removed_in_batch)
            if cursor >= len(working):
                break

        return {
            "removed_entry_ids": sorted(all_removed),
            "enrichments": all_enrichments,
        }

    def _dedup_batch(self, entries: list[dict]) -> dict:
        """Deduplicate a single batch that fits in the context window."""
        import json as _json

        system_prompt = (
            "Sei un medico esperto di oncologia. Il tuo compito e' analizzare "
            "un registro clinico temporale ed eliminare le voci ridondanti "
            "che descrivono lo STESSO evento clinico. "
            "Rispondi SOLO con JSON valido."
        )

        compact = []
        for e in entries:
            compact.append(_json.dumps({
                "id": e.get("entry_id", ""),
                "d": e.get("date_observed", ""),
                "dr": e.get("date_resolved"),
                "c": e.get("category", ""),
                "t": e.get("description", ""),
                "s": e.get("status", ""),
            }, ensure_ascii=False))
        entries_json = "[\n" + ",\n".join(compact) + "\n]"

        user_prompt = f"""Elimina i duplicati semantici da questo registro ({len(entries)} voci).

REGOLE: Due voci sono DUPLICATE se descrivono lo STESSO evento clinico.
NON unire: eventi diversi, sospensione/ripresa di trattamento, diagnosi vs progressione.
Per ogni gruppo tieni la voce con descrizione PIU' COMPLETA e data PIU' PRECISA.

REGISTRO:
{entries_json}

Restituisci SOLO: ```json {{"removed_entry_ids": [...], "enrichments": {{...}}}} ```
"""
        try:
            raw = self.generate_text(user_prompt, system_prompt)
            return self._parse_json_dedup(raw, entries)
        except Exception:
            return {"removed_entry_ids": [], "enrichments": {}}

    @staticmethod
    def _parse_json_dedup(raw: str, fallback_entries: list[dict]) -> dict:
        """Parse the LLM dedup response (compact format: only removed IDs
        and optional enrichments).  Falls back gracefully on any error."""
        import json as _json
        import re as _re

        if not raw or not raw.strip():
            return {
                "removed_entry_ids": [],
                "enrichments": {},
            }

        text = raw.strip()

        # Extract from ```json fence
        m = _re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if m:
            text = m.group(1).strip()

        # Find JSON object boundaries
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]

        try:
            parsed = _json.loads(text)
            if isinstance(parsed, dict):
                removed = parsed.get("removed_entry_ids", [])
                enrichments = parsed.get("enrichments", {})
                # Also support old format for backward compat
                if "deduplicated_entries" in parsed:
                    return parsed
                return {
                    "removed_entry_ids": (
                        list(removed) if isinstance(removed, list) else []
                    ),
                    "enrichments": (
                        enrichments if isinstance(enrichments, dict) else {}
                    ),
                }
        except (_json.JSONDecodeError, ValueError):
            pass

        return {
            "removed_entry_ids": [],
            "enrichments": {},
        }
