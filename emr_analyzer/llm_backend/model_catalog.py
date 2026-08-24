"""Curated, offline model catalogue for EMR Analyzer.

The catalogue is intentionally shipped with the application.  Merely opening
the LLM settings must never contact a remote service while a clinical
workspace is active.  Network access starts only after the user explicitly
chooses to install one of the verified Ollama tags below.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
import re
from typing import Iterable
from urllib.parse import urlparse

from ..config import LLM_MODEL_CATALOG_CACHE_PATH


CATALOG_REVIEW_DATE = "2026-08-21"
CATALOG_SCHEMA_VERSION = 1
CATALOG_VERSION = 1
_OLLAMA_TAG_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,119}:[A-Za-z0-9][A-Za-z0-9._-]{0,119}$"
)
_MAX_MANIFEST_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class CatalogModel:
    catalog_id: str
    display_name: str
    ollama_tag: str
    provider: str
    parameter_label: str
    quantization: str
    download_size_gb: float
    context_length: int
    minimum_ram_gb: float
    recommended_ram_gb: float
    roles: tuple[str, ...]
    profile: str
    description: str
    strengths: str
    caution: str
    license_name: str
    source_url: str
    document_rank: int
    clinical_state_rank: int
    medically_specialized: bool = False

    def supports_role(self, role: str | None) -> bool:
        """Map the fine-grained app roles onto catalogue capabilities."""
        if not role:
            return True
        if role == "atomic_evidence":
            # Atomic extraction combines document fidelity with structured
            # clinical output; either catalogue capability is relevant.
            return bool({"document", "clinical_state"} & set(self.roles))
        if role == "clinical_events":
            return "clinical_state" in self.roles
        return role in self.roles

    def rank_for(self, role: str | None) -> int:
        if role == "document":
            return self.document_rank
        if role in {"clinical_events", "clinical_state"}:
            return self.clinical_state_rank
        if role == "atomic_evidence":
            return min(self.document_rank, self.clinical_state_rank)
        return min(self.document_rank, self.clinical_state_rank)


@dataclass(frozen=True, slots=True)
class HardwareFit:
    level: str
    label: str
    explanation: str


@dataclass(frozen=True, slots=True)
class CatalogSnapshot:
    models: tuple[CatalogModel, ...]
    catalog_version: int
    review_date: str
    source: str


MODEL_CATALOG: tuple[CatalogModel, ...] = (
    CatalogModel(
        catalog_id="qwen3-30b-a3b-instruct",
        display_name="Qwen3 30B-A3B Instruct 2507",
        ollama_tag="qwen3:30b-a3b-instruct-2507-q4_K_M",
        provider="Qwen",
        parameter_label="30B totali · 3B attivi",
        quantization="Q4_K_M",
        download_size_gb=19.0,
        context_length=262_144,
        minimum_ram_gb=28,
        recommended_ram_gb=40,
        roles=("document", "clinical_state"),
        profile="Qualità consigliata",
        description=(
            "Scelta predefinita per registri longitudinali, assemblaggio di "
            "episodi e analisi strutturate in italiano."
        ),
        strengths=(
            "Istruzioni robuste, ragionamento efficiente MoE, contesto molto "
            "ampio e buona resa multilingue."
        ),
        caution="Non è un dispositivo medico: ogni risultato richiede verifica.",
        license_name="Apache 2.0",
        source_url="https://ollama.com/library/qwen3/tags",
        document_rank=2,
        clinical_state_rank=1,
    ),
    CatalogModel(
        catalog_id="medgemma-27b",
        display_name="MedGemma 27B",
        ollama_tag="medgemma:27b",
        provider="Google Health AI",
        parameter_label="27B",
        quantization="Q4_K_M",
        download_size_gb=17.0,
        context_length=131_072,
        minimum_ram_gb=26,
        recommended_ram_gb=40,
        roles=("clinical_state",),
        profile="Clinico-specializzato",
        description=(
            "Modello specializzato per comprensione di testo medico; utile "
            "come alternativa per Clinical State e analisi cliniche."
        ),
        strengths=(
            "Vocabolario e ragionamento biomedico dedicati, con lunga finestra "
            "di contesto."
        ),
        caution=(
            "Licenza sanitaria specifica; la specializzazione non sostituisce "
            "la validazione clinica né garantisce il miglior JSON."
        ),
        license_name="Health AI Developer Foundations",
        source_url="https://ollama.com/library/medgemma/tags",
        document_rank=6,
        clinical_state_rank=2,
        medically_specialized=True,
    ),
    CatalogModel(
        catalog_id="gpt-oss-20b",
        display_name="gpt-oss 20B",
        ollama_tag="gpt-oss:20b",
        provider="OpenAI",
        parameter_label="20.9B · MoE",
        quantization="MXFP4",
        download_size_gb=14.0,
        context_length=131_072,
        minimum_ram_gb=24,
        recommended_ram_gb=32,
        roles=("clinical_state",),
        profile="Ragionamento",
        description=(
            "Alternativa orientata a ragionamento e output strutturati per "
            "interrogazioni complesse del registro."
        ),
        strengths="Ragionamento configurabile, structured output e licenza permissiva.",
        caution=(
            "Richiede una versione recente di llama.cpp; non è specializzato "
            "sulla lingua o sulla pratica clinica italiana."
        ),
        license_name="Apache 2.0",
        source_url="https://ollama.com/library/gpt-oss/tags",
        document_rank=7,
        clinical_state_rank=3,
    ),
    CatalogModel(
        catalog_id="gemma3-12b",
        display_name="Gemma 3 12B",
        ollama_tag="gemma3:12b",
        provider="Google",
        parameter_label="12B",
        quantization="Q4_K_M",
        download_size_gb=8.1,
        context_length=131_072,
        minimum_ram_gb=16,
        recommended_ram_gb=24,
        roles=("document", "clinical_state"),
        profile="Documenti · bilanciato",
        description=(
            "Modello bilanciato per isolamento e normalizzazione dei referti, "
            "adatto anche a registri di dimensioni moderate."
        ),
        strengths="Multilingue, contesto 128K e buon rapporto qualità/memoria.",
        caution="Per episodi complessi è meno capace dei modelli da 27–30B.",
        license_name="Gemma Terms",
        source_url="https://ollama.com/library/gemma3/tags",
        document_rank=1,
        clinical_state_rank=5,
    ),
    CatalogModel(
        catalog_id="qwen3-14b",
        display_name="Qwen3 14B",
        ollama_tag="qwen3:14b-q4_K_M",
        provider="Qwen",
        parameter_label="14B",
        quantization="Q4_K_M",
        download_size_gb=9.3,
        context_length=40_960,
        minimum_ram_gb=16,
        recommended_ram_gb=24,
        roles=("document", "clinical_state"),
        profile="Bilanciato",
        description=(
            "Scelta compatta e affidabile quando si desidera usare lo stesso "
            "modello per documenti e Clinical State."
        ),
        strengths="Buona comprensione dell'italiano e istruzioni strutturate.",
        caution="La finestra 40K limita i prompt longitudinali più grandi.",
        license_name="Apache 2.0",
        source_url="https://ollama.com/library/qwen3/tags",
        document_rank=3,
        clinical_state_rank=4,
    ),
    CatalogModel(
        catalog_id="ministral3-14b",
        display_name="Ministral 3 14B Instruct",
        ollama_tag="ministral-3:14b-instruct-2512-q4_K_M",
        provider="Mistral AI",
        parameter_label="14B",
        quantization="Q4_K_M",
        download_size_gb=9.1,
        context_length=262_144,
        minimum_ram_gb=18,
        recommended_ram_gb=28,
        roles=("document", "clinical_state"),
        profile="Contesto esteso",
        description=(
            "Alternativa europea a contesto molto ampio per referti lunghi e "
            "output JSON."
        ),
        strengths="Italiano, system prompt, JSON e finestra 256K.",
        caution="Richiede Ollama e llama.cpp recenti.",
        license_name="Apache 2.0",
        source_url="https://ollama.com/library/ministral-3/tags",
        document_rank=4,
        clinical_state_rank=6,
    ),
    CatalogModel(
        catalog_id="qwen3-4b",
        display_name="Qwen3 4B",
        ollama_tag="qwen3:4b-q4_K_M",
        provider="Qwen",
        parameter_label="4B",
        quantization="Q4_K_M",
        download_size_gb=2.5,
        context_length=262_144,
        minimum_ram_gb=8,
        recommended_ram_gb=12,
        roles=("document",),
        profile="Rapido · memoria ridotta",
        description=(
            "Modello leggero per prove, macchine con poca RAM e prima "
            "normalizzazione documentale."
        ),
        strengths="Download piccolo e latenza contenuta.",
        caution=(
            "Non consigliato per assemblaggio clinico definitivo o inferenze "
            "longitudinali complesse."
        ),
        license_name="Apache 2.0",
        source_url="https://ollama.com/library/qwen3/tags",
        document_rank=5,
        clinical_state_rank=9,
    ),
    CatalogModel(
        catalog_id="llama33-70b-q4",
        display_name="Llama 3.3 70B Instruct",
        ollama_tag="llama3.3:70b-instruct-q4_K_M",
        provider="Meta",
        parameter_label="70B",
        quantization="Q4_K_M",
        download_size_gb=43.0,
        context_length=131_072,
        minimum_ram_gb=56,
        recommended_ram_gb=80,
        roles=("clinical_state",),
        profile="Alta qualità · hardware capiente",
        description=(
            "Opzione densa di grandi dimensioni per macchine con molta memoria "
            "quando la qualità conta più della velocità."
        ),
        strengths="Modello multilingue con supporto esplicito dell'italiano.",
        caution="Download, memoria e tempi di elaborazione molto elevati.",
        license_name="Llama 3.3 Community License",
        source_url="https://ollama.com/library/llama3.3/tags",
        document_rank=8,
        clinical_state_rank=7,
    ),
)


def hardware_fit(model: CatalogModel, total_ram_gb: float) -> HardwareFit:
    total = max(0.0, float(total_ram_gb or 0.0))
    if total < model.minimum_ram_gb:
        return HardwareFit(
            "insufficient",
            "RAM insufficiente",
            f"Richiede almeno {model.minimum_ram_gb:g} GiB; rilevati {total:.1f} GiB.",
        )
    if total < model.recommended_ram_gb:
        return HardwareFit(
            "compatible",
            "Compatibile con limiti",
            "Ridurre contesto o slot concorrenti per mantenere margine di memoria.",
        )
    return HardwareFit(
        "optimal",
        "Adatto a questo sistema",
        f"RAM consigliata ≥ {model.recommended_ram_gb:g} GiB; rilevati {total:.1f} GiB.",
    )


def filter_catalog(
    *,
    role: str | None = None,
    query: str = "",
    total_ram_gb: float = 0.0,
    compatible_only: bool = False,
    models: Iterable[CatalogModel] | None = None,
) -> list[CatalogModel]:
    needle = " ".join(str(query or "").casefold().split())
    result = []
    for model in models if models is not None else MODEL_CATALOG:
        if role and not model.supports_role(role):
            continue
        if compatible_only and hardware_fit(model, total_ram_gb).level == "insufficient":
            continue
        searchable = " ".join((
            model.display_name, model.provider, model.profile,
            model.description, model.strengths, model.ollama_tag,
        )).casefold()
        if needle and needle not in searchable:
            continue
        result.append(model)
    fit_order = {"optimal": 0, "compatible": 1, "insufficient": 2}
    return sorted(result, key=lambda model: (
        fit_order[hardware_fit(model, total_ram_gb).level],
        model.rank_for(role),
        model.download_size_gb,
        model.display_name.casefold(),
    ))


def builtin_catalog_manifest() -> dict:
    """Return the serializable manifest shipped with this application."""
    return {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "catalog_version": CATALOG_VERSION,
        "review_date": CATALOG_REVIEW_DATE,
        "models": [asdict(model) for model in MODEL_CATALOG],
    }


def validate_catalog_manifest(payload: object) -> CatalogSnapshot:
    """Validate untrusted remote JSON before it can enter the GUI/cache."""
    if not isinstance(payload, dict):
        raise ValueError("Il catalogo remoto non è un oggetto JSON.")
    allowed_root = {
        "schema_version", "catalog_version", "review_date", "models",
    }
    if set(payload) != allowed_root:
        raise ValueError("Il catalogo remoto contiene campi radice non validi.")
    schema_version = _strict_int(payload.get("schema_version"), "schema_version")
    if schema_version != CATALOG_SCHEMA_VERSION:
        raise ValueError(
            f"Schema catalogo non supportato: {schema_version}."
        )
    catalog_version = _strict_int(
        payload.get("catalog_version"), "catalog_version"
    )
    if catalog_version < 1:
        raise ValueError("La versione del catalogo deve essere positiva.")
    review_date = _safe_manifest_text(
        payload.get("review_date"), "review_date", maximum=10
    )
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", review_date):
        raise ValueError("La data di revisione del catalogo non è ISO.")
    raw_models = payload.get("models")
    if not isinstance(raw_models, list) or not 1 <= len(raw_models) <= 100:
        raise ValueError("Il catalogo deve contenere da 1 a 100 modelli.")

    expected_fields = {field.name for field in fields(CatalogModel)}
    models = []
    ids: set[str] = set()
    tags: set[str] = set()
    for index, raw in enumerate(raw_models, start=1):
        if not isinstance(raw, dict) or set(raw) != expected_fields:
            raise ValueError(
                f"La voce catalogo {index} ha campi mancanti o inattesi."
            )
        model = _validated_catalog_model(raw, index)
        if model.catalog_id.casefold() in ids:
            raise ValueError("Il catalogo contiene catalog_id duplicati.")
        if model.ollama_tag.casefold() in tags:
            raise ValueError("Il catalogo contiene tag Ollama duplicati.")
        ids.add(model.catalog_id.casefold())
        tags.add(model.ollama_tag.casefold())
        models.append(model)
    return CatalogSnapshot(
        models=tuple(models),
        catalog_version=catalog_version,
        review_date=review_date,
        source="remote",
    )


def load_catalog_snapshot(
    cache_path: str | Path = LLM_MODEL_CATALOG_CACHE_PATH,
) -> CatalogSnapshot:
    """Load a validated cache, falling back to the built-in catalogue."""
    fallback = CatalogSnapshot(
        models=MODEL_CATALOG,
        catalog_version=CATALOG_VERSION,
        review_date=CATALOG_REVIEW_DATE,
        source="integrated",
    )
    try:
        path = Path(cache_path)
        if not path.is_file() or path.stat().st_size > _MAX_MANIFEST_BYTES:
            return fallback
        payload = json.loads(path.read_text(encoding="utf-8"))
        cached = validate_catalog_manifest(payload)
        if cached.catalog_version < fallback.catalog_version:
            return fallback
        return CatalogSnapshot(
            models=cached.models,
            catalog_version=cached.catalog_version,
            review_date=cached.review_date,
            source="cache",
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return fallback


def cache_catalog_manifest(
    raw_payload: bytes,
    cache_path: str | Path = LLM_MODEL_CATALOG_CACHE_PATH,
) -> CatalogSnapshot:
    """Validate remote bytes and atomically replace the local cache."""
    if not raw_payload or len(raw_payload) > _MAX_MANIFEST_BYTES:
        raise ValueError("Dimensione del catalogo remoto non valida.")
    try:
        payload = json.loads(raw_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Il catalogo remoto non è JSON UTF-8 valido.") from exc
    snapshot = validate_catalog_manifest(payload)
    active = load_catalog_snapshot(cache_path)
    if snapshot.catalog_version < active.catalog_version:
        raise ValueError(
            "Aggiornamento rifiutato: la versione remota è meno recente "
            "del catalogo attivo."
        )
    if (
        snapshot.catalog_version == active.catalog_version
        and (
            snapshot.models != active.models
            or snapshot.review_date != active.review_date
        )
    ):
        raise ValueError(
            "Aggiornamento rifiutato: il contenuto è cambiato senza "
            "incrementare catalog_version."
        )
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_bytes(raw_payload)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return CatalogSnapshot(
        models=snapshot.models,
        catalog_version=snapshot.catalog_version,
        review_date=snapshot.review_date,
        source="cache",
    )


def _validated_catalog_model(raw: dict, index: int) -> CatalogModel:
    text_limits = {
        "catalog_id": 120, "display_name": 160, "ollama_tag": 240,
        "provider": 100, "parameter_label": 100, "quantization": 50,
        "profile": 100, "description": 700, "strengths": 700,
        "caution": 700, "license_name": 160, "source_url": 500,
    }
    cleaned = {
        key: _safe_manifest_text(raw.get(key), key, maximum=maximum)
        for key, maximum in text_limits.items()
    }
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{1,119}", cleaned["catalog_id"]):
        raise ValueError(f"catalog_id non valido nella voce {index}.")
    if not _OLLAMA_TAG_RE.fullmatch(cleaned["ollama_tag"]):
        raise ValueError(f"Tag Ollama non valido nella voce {index}.")
    parsed_source = urlparse(cleaned["source_url"])
    if (
        parsed_source.scheme != "https"
        or parsed_source.hostname != "ollama.com"
        or not parsed_source.path.startswith("/library/")
        or parsed_source.username
        or parsed_source.password
    ):
        raise ValueError(f"Fonte non consentita nella voce {index}.")

    roles = raw.get("roles")
    if (
        not isinstance(roles, (list, tuple))
        or not roles
        or len(roles) > 2
        or len(set(roles)) != len(roles)
        or not set(roles) <= {"document", "clinical_state"}
    ):
        raise ValueError(f"Ruoli non validi nella voce {index}.")
    medically_specialized = raw.get("medically_specialized")
    if not isinstance(medically_specialized, bool):
        raise ValueError(f"Flag medico non valido nella voce {index}.")

    download_size = _strict_number(
        raw.get("download_size_gb"), "download_size_gb", 0.05, 1000
    )
    context = _strict_int(raw.get("context_length"), "context_length")
    if not 2048 <= context <= 2_097_152:
        raise ValueError(f"Contesto non valido nella voce {index}.")
    minimum_ram = _strict_number(
        raw.get("minimum_ram_gb"), "minimum_ram_gb", 1, 2048
    )
    recommended_ram = _strict_number(
        raw.get("recommended_ram_gb"), "recommended_ram_gb", 1, 4096
    )
    if recommended_ram < minimum_ram:
        raise ValueError(f"RAM consigliata incoerente nella voce {index}.")
    document_rank = _strict_int(raw.get("document_rank"), "document_rank")
    clinical_rank = _strict_int(
        raw.get("clinical_state_rank"), "clinical_state_rank"
    )
    if not 1 <= document_rank <= 100 or not 1 <= clinical_rank <= 100:
        raise ValueError(f"Priorità non valida nella voce {index}.")
    return CatalogModel(
        **cleaned,
        download_size_gb=download_size,
        context_length=context,
        minimum_ram_gb=minimum_ram,
        recommended_ram_gb=recommended_ram,
        roles=tuple(roles),
        document_rank=document_rank,
        clinical_state_rank=clinical_rank,
        medically_specialized=medically_specialized,
    )


def _safe_manifest_text(value: object, field: str, *, maximum: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Il campo {field} deve essere testuale.")
    cleaned = " ".join(value.split())
    if not cleaned or len(cleaned) > maximum:
        raise ValueError(f"Lunghezza non valida per il campo {field}.")
    if any(ord(character) < 32 for character in cleaned):
        raise ValueError(f"Caratteri di controllo nel campo {field}.")
    return cleaned


def _strict_int(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"Il campo {field} deve essere intero.")
    return value


def _strict_number(
    value: object, field: str, minimum: float, maximum: float
) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not minimum <= float(value) <= maximum
    ):
        raise ValueError(f"Valore numerico non valido per {field}.")
    return float(value)
