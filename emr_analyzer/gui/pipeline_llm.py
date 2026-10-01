"""Explicit launch-time LLM selection, with independent pipeline presets."""

PIPELINES = {
    "lexicon": ("atomic_evidence", "Lessico condiviso — analisi del documento"),
    "atomic": ("atomic_evidence", "Estrazione e codifica SNOMED CT degli eventi clinici"),
    "documents": ("document", "Estrazione e normalizzazione dei documenti"),
}


def pipeline_definition(key):
    if key.startswith("prompt:"):
        from .prompt_manager_dialog import _PROMPT_ROLE
        prompt = key.split(":", 1)[1]
        return _PROMPT_ROLE.get(prompt, "atomic_evidence"), f"Prova del prompt — {prompt}"
    return PIPELINES[key]


def prepare_pipeline(services, key, parent):
    """Return false on cancel; the application installs the UI coordinator.

    Isolated widgets and headless consumers may continue using injected clients.
    No worker is started and no settings are changed by this helper itself.
    """
    prepare = services.get("prepare_pipeline_llm")
    return bool(prepare(key, parent)) if callable(prepare) else True
