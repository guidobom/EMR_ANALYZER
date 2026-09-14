"""Shared, conservative context budgeting for local clinical queries.

Character budgets are estimates, not a substitute for backend tokenization.
Never force two oversized summaries into the same model request.
"""


def context_budget(llm_client, *, overhead: str = "") -> int:
    context = int(getattr(llm_client, "context_length", 32768) or 32768)
    output = int(getattr(llm_client, "max_output_tokens", 4096) or 4096)
    budget = int((context - output - 1024) * 2) - len(overhead)
    if budget < 512:
        raise ValueError(
            "Contesto insufficiente per domanda, istruzioni e risposta: "
            "aumentare il contesto o ridurre l'output/la conversazione."
        )
    return budget


def split_text(text: str, max_chars: int) -> list[str]:
    """Split without discarding any character; prefer line boundaries."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    chunks = []
    while text:
        end = min(len(text), max_chars)
        if end < len(text):
            boundary = text.rfind("\n", 0, end)
            if boundary > max_chars // 2:
                end = boundary + 1
        chunks.append(text[:end])
        text = text[end:]
    return chunks


def reconcile_partials(llm_client, partials, question, system_prompt, *, max_chars,
                       check_cancel=None):
    current = list(partials)
    if not current:
        return "Nessuna evidenza disponibile per la domanda."
    prefix = f"DOMANDA: {question}\n\nSINTESI PARZIALI DA RICONCILIARE:\n"
    suffix = (
        "\n\nProduci una risposta unica, elimina ripetizioni, conserva "
        "discordanze e tutte le citazioni verificabili. Non introdurre "
        "affermazioni nuove. L'assenza di un dato in una parte non prova "
        "l'assenza nel dossier."
    )
    available = max_chars - len(prefix) - len(suffix)
    while len(current) > 1:
        if check_cancel:
            check_cancel()
        groups, group, size = [], [], 0
        for part in current:
            if group and size + len(part) + 2 > available:
                groups.append(group)
                group, size = [], 0
            group.append(part)
            size += len(part) + 2
        if group:
            groups.append(group)
        if len(groups) == len(current):
            return (
                "Sintesi unica non prodotta: le risposte parziali superano "
                "il budget di contesto. Risultati separati da riconciliare:\n\n"
                + "\n\n".join(current)
            )
        reduced = []
        for group in groups:
            if check_cancel:
                check_cancel()
            reduced.append(group[0] if len(group) == 1 else llm_client.generate_text(
                prefix + "\n\n".join(group) + suffix, system_prompt
            ))
        current = reduced
    return current[0]
