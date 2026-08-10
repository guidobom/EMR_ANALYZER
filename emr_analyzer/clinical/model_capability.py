"""Local model capability checks for longitudinal Clinical State building."""

from __future__ import annotations

from ..models.longitudinal import (
    DOCUMENT_DELTA_JSON_SCHEMA,
    ModelCapabilityResult,
)


class ClinicalStateModelCapabilityChecker:
    """Verify both JSON-schema support and a minimal clinical delta task."""

    def __init__(self):
        self._cache: dict[tuple[str, str], ModelCapabilityResult] = {}

    def check(self, llm_client, force: bool = False) -> ModelCapabilityResult:
        model_name = getattr(llm_client, "model", "") if llm_client else ""
        base_url = getattr(llm_client, "base_url", "") if llm_client else ""
        key = (base_url, model_name)
        if not force and key in self._cache:
            return self._cache[key]

        if not llm_client or not llm_client.is_available:
            return self._remember(key, ModelCapabilityResult(
                model_name=model_name or "non selezionato",
                available=False,
                json_schema_supported=False,
                clinical_delta_supported=False,
                message="Ollama o il modello non sono disponibili.",
            ))

        try:
            schema_response = llm_client.generate_structured(
                (
                    "Test tecnico locale. Restituisci status='ok' e number=7. "
                    "Non aggiungere altri campi."
                ),
                "Rispondi esclusivamente secondo lo schema JSON richiesto.",
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["status", "number"],
                    "properties": {
                        "status": {"type": "string", "enum": ["ok"]},
                        "number": {"type": "integer", "enum": [7]},
                    },
                },
            )
            if (
                not isinstance(schema_response, dict)
                or schema_response.get("status") != "ok"
                or schema_response.get("number") != 7
            ):
                raise ValueError("risposta JSON formalmente valida ma non conforme")
        except Exception as exc:
            return self._remember(key, ModelCapabilityResult(
                model_name=model_name,
                available=True,
                json_schema_supported=False,
                clinical_delta_supported=False,
                message=(
                    "Il modello non ha superato il test di output JSON "
                    f"strutturato: {type(exc).__name__}: {exc}"
                ),
            ))

        source_quote = (
            "Correzione: l'evento del 10/01/2024 era rash CTCAE grado 2, "
            "non grado 1."
        )
        prompt = f"""Test clinico sintetico di capacità.

STATO PRECEDENTE:
- evidence_id: EVIDENCE_TEST_1
- categoria: toxicity
- entità: rash
- data clinica: 2024-01-10
- grading: CTCAE grado 1

DOCUMENTO SUCCESSIVO, data 2024-02-01:
[PAGINA 1] {source_quote}

Produci un solo delta. Devi riconoscere una CORREZIONE retrospettiva,
indirizzata a EVIDENCE_TEST_1, mantenere la data clinica 2024-01-10 e
registrare CTCAE grado 2. source_text deve essere una citazione letterale.
Compila tutti i campi richiesti; usa null quando un dato non è disponibile.
"""
        try:
            response = llm_client.generate_structured(
                prompt,
                (
                    "Sei un motore di riconciliazione clinica. Distingui una "
                    "correzione retrospettiva da un'evoluzione nel tempo."
                ),
                DOCUMENT_DELTA_JSON_SCHEMA,
            )
            self._validate_semantic_probe(response, source_quote)
        except Exception as exc:
            return self._remember(key, ModelCapabilityResult(
                model_name=model_name,
                available=True,
                json_schema_supported=True,
                clinical_delta_supported=False,
                message=(
                    "Il JSON è supportato, ma il modello non ha superato il "
                    "test clinico correzione/evoluzione: "
                    f"{type(exc).__name__}: {exc}"
                ),
            ))

        return self._remember(key, ModelCapabilityResult(
            model_name=model_name,
            available=True,
            json_schema_supported=True,
            clinical_delta_supported=True,
            message=(
                "Modello compatibile: JSON strutturato e riconciliazione "
                "clinica incrementale verificati."
            ),
        ))

    @staticmethod
    def _validate_semantic_probe(response: dict, source_quote: str) -> None:
        if not isinstance(response, dict):
            raise ValueError("risposta non oggetto")
        operations = response.get("operations")
        if not isinstance(operations, list) or len(operations) != 1:
            raise ValueError("era attesa una singola operazione")
        operation = operations[0]
        expected = {
            "operation": "correction",
            "target_evidence_id": "EVIDENCE_TEST_1",
            "valid_start_date": "2024-01-10",
            "grade": 2,
        }
        for field, value in expected.items():
            if operation.get(field) != value:
                raise ValueError(
                    f"{field}={operation.get(field)!r}, atteso {value!r}"
                )
        if "rash" not in str(operation.get("normalized_entity", "")).casefold():
            raise ValueError("entità clinica non riconosciuta")
        quote = " ".join(str(operation.get("source_text") or "").split())
        if quote not in " ".join(source_quote.split()):
            raise ValueError("source_text non è una citazione letterale")
        if not operation.get("correction_explicit"):
            raise ValueError("correzione esplicita non riconosciuta")

    def invalidate(self, llm_client=None) -> None:
        if llm_client is None:
            self._cache.clear()
            return
        key = (
            getattr(llm_client, "base_url", ""),
            getattr(llm_client, "model", ""),
        )
        self._cache.pop(key, None)

    def _remember(self, key, result):
        self._cache[key] = result
        return result
