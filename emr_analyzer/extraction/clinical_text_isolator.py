"""Plain-text clinical isolation from deterministic PDF text."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
import re
from typing import Iterable

from ..pipeline.sensitive_data import (
    DEIDENTIFICATION_VERSION,
    SensitiveDataSanitizer,
)

PROMPT_VERSION = "clinical_text_isolation_v5"
DEFAULT_CONTEXT_LENGTH = 16_384
DEFAULT_MAX_OUTPUT_TOKENS = 4_096
PROMPT_TOKEN_RESERVE = 1_000
CHARS_PER_TOKEN_ESTIMATE = 2.5
MIN_CHUNK_CHARS = 1_200
MAX_VALIDATION_ATTEMPTS = 2
FALLBACK_SECTION_CHARS = 4_500
MIN_FALLBACK_SOURCE_CHARS = 1_800

_PAGE_COMMENT_RE = re.compile(
    r"<!--\s*page\s*:\s*(\d+)\s*-->", re.IGNORECASE
)
_PAGE_MARKER_RE = re.compile(r"\[PAGINA\s+\d+\]", re.IGNORECASE)
_PLACEHOLDER_RE = re.compile(r"\[\[(?:VALORE|PAGINA)_[A-Z]+\]\]")
_MALFORMED_PLACEHOLDER_RE = re.compile(
    r"\[\[?\s*(?:VALORE|PAGINA)[\s_-]+[A-Z]+", re.IGNORECASE
)
_NUMERIC_LITERAL_RE = re.compile(
    r"(?:"
    r"\d{1,2}[./-]\d{1,2}(?:[./-]\d{2,4})?"
    r"|\d{1,2}:\d{2}"
    r"|[+-]?\d+(?:[.,]\d+)*(?:\s?%|\s?°C)?"
    r")",
    re.IGNORECASE,
)

_SYSTEM_PROMPT = (
    "Sei un estrattore clinico conservativo. Filtra il testo supportato dalla "
    "sorgente mantenendo rigorosamente ordine, paragrafi e associazioni "
    "originali. Non riorganizzare, parafrasare, interpretare o inferire. "
    "Restituisci esclusivamente testo clinico in italiano; non produrre JSON, "
    "spiegazioni o commenti."
)


@dataclass
class ClinicalTextIsolationResult:
    text: str
    model_name: str
    prompt_version: str = PROMPT_VERSION
    chunk_count: int = 0
    warnings: list[str] = field(default_factory=list)
    redaction_counts: dict[str, int] = field(default_factory=dict)
    deidentification_version: str = DEIDENTIFICATION_VERSION


class ClinicalTextIsolationError(RuntimeError):
    """The document model did not produce a complete safe text."""

    def __init__(self, message: str, *, systemic: bool = False):
        super().__init__(message)
        self.systemic = systemic


class ClinicalTextValidationError(ValueError):
    """All validation attempts for one source block failed."""

    def __init__(self, details: list[str], categories: list[str]):
        self.details = details
        self.categories = categories
        message = "; ".join(
            f"tentativo {index}: {detail}"
            for index, detail in enumerate(details, start=1)
        )
        super().__init__(message)


class ClinicalTextIsolator:
    """Produce one normalized clinical text, never a structured payload."""

    def __init__(self, llm_client=None, sanitizer=None):
        self.llm = llm_client
        self.sanitizer = sanitizer or SensitiveDataSanitizer()

    def isolate(
        self,
        text: str,
        document_date: str | None = None,
        parsing_result=None,
        sensitive_identity=None,
    ) -> ClinicalTextIsolationResult:
        if not self.llm or not self.llm.is_available:
            raise ClinicalTextIsolationError(
                "Il motore locale o il modello documentale non sono disponibili",
                systemic=True,
            )
        source = self._prepare_source(text, parsing_result)
        source_sanitization = self.sanitizer.sanitize(
            source, sensitive_identity
        )
        source = source_sanitization.text
        chunks = list(self._chunks(source))
        if not chunks:
            raise ClinicalTextIsolationError(
                "Il testo sorgente del documento è vuoto"
            )

        normalized_parts = []
        errors = []
        systemic_errors = []
        warnings = []
        processing_units = 0
        for index, chunk in enumerate(chunks, start=1):
            try:
                normalized, corrections = self._normalize_chunk(
                    chunk, document_date
                )
                normalized_parts.append(normalized)
                processing_units += 1
                if corrections:
                    warnings.append(
                        f"chunk {index}: retry correttivo riuscito dopo "
                        f"{corrections[-1]}"
                    )
            except ClinicalTextValidationError as primary_error:
                try:
                    normalized, fallback_warnings, section_count = (
                        self._normalize_with_section_fallback(
                            chunk, document_date, primary_error
                        )
                    )
                    normalized_parts.append(normalized)
                    processing_units += section_count
                    warnings.extend(
                        f"chunk {index}: {warning}"
                        for warning in fallback_warnings
                    )
                except Exception as fallback_error:
                    message = " ".join(
                        str(fallback_error).split()
                    )[:700]
                    errors.append(
                        f"chunk {index}: {type(fallback_error).__name__}: "
                        f"{message}"
                    )
                    systemic_errors.append(
                        self._is_systemic_error(fallback_error)
                    )
            except Exception as exc:
                message = " ".join(str(exc).split())[:500]
                errors.append(f"chunk {index}: {type(exc).__name__}: {message}")
                systemic_errors.append(self._is_systemic_error(exc))

        # A partial normalized document is unsafe: do not save any output if
        # even one source chunk was not processed successfully.
        if errors:
            raise ClinicalTextIsolationError(
                f"Isolamento testuale incompleto ({len(errors)}/{len(chunks)} "
                f"chunk falliti). Primo errore: {errors[0]}",
                systemic=any(systemic_errors),
            )

        final_sanitization = self.sanitizer.sanitize(
            "\n\n".join(normalized_parts).strip(),
            sensitive_identity,
        )
        redaction_counts = Counter(source_sanitization.counts)
        redaction_counts.update(final_sanitization.counts)
        return ClinicalTextIsolationResult(
            text=final_sanitization.text,
            model_name=self.llm.model,
            chunk_count=processing_units,
            warnings=warnings,
            redaction_counts=dict(redaction_counts),
        )

    def _normalize_chunk(
        self, source: str, document_date: str | None
    ) -> tuple[str, list[str]]:
        """Normalize one chunk while keeping every numeric literal immutable."""
        protected_source, replacements = self._protect_numeric_literals(source)
        validation_details = []
        validation_categories = []

        for attempt in range(MAX_VALIDATION_ATTEMPTS):
            correction = (
                self._corrective_instruction(validation_categories[-1])
                if validation_categories else None
            )
            # Corrective retries vary the sampling slightly so a
            # deterministic model failure (fixed seed + low temperature)
            # does not repeat identically forever.  The FIRST attempt
            # always uses the configured parameters untouched.
            overrides = {}
            if attempt > 0 and getattr(self.llm, "seed", None) is not None:
                overrides["seed"] = self.llm.seed + attempt
                overrides["temperature"] = max(
                    0.4,
                    float(getattr(self.llm, "temperature", 0.1) or 0.1),
                )
            try:
                response = self.llm.generate_text(
                    self._prompt(
                        protected_source,
                        document_date,
                        corrective_instruction=correction,
                    ),
                    _SYSTEM_PROMPT,
                    **overrides,
                )
            except RuntimeError as exc:
                # Output-length errors are transient — treat as a validation
                # failure so the corrective-retry → section-fallback chain
                # can shrink the chunk and retry.
                if "limite di token" in str(exc):
                    # Surface as a validation failure so the caller's
                    # section-fallback chain shrinks the chunk and retries,
                    # instead of failing the whole document.
                    raise ClinicalTextValidationError(
                        ["il modello ha raggiunto il limite di token in output"],
                        ["limite token output"],
                    ) from exc
                if validation_categories:
                    raise RuntimeError(
                        "chiamata di retry fallita dopo "
                        f"{validation_categories[-1]}: {exc}"
                    ) from exc
                raise
            except Exception as exc:
                if validation_categories:
                    raise RuntimeError(
                        "chiamata di retry fallita dopo "
                        f"{validation_categories[-1]}: {exc}"
                    ) from exc
                raise
            protected_output = self._clean_response(response)
            try:
                normalized = self._restore_numeric_literals(
                    protected_output, replacements
                )
                self._validate_output(normalized, source, document_date)
                normalized = self._strip_page_markers(normalized)
                return normalized, validation_categories
            except ValueError as exc:
                validation_details.append(" ".join(str(exc).split())[:300])
                validation_categories.append(
                    self._validation_category(exc)
                )

        raise ClinicalTextValidationError(
            validation_details, validation_categories
        )

    @staticmethod
    def _prompt(
        text: str,
        document_date: str | None,
        corrective_instruction: str | None = None,
    ) -> str:
        del document_date  # The administrative date must not leak into output.
        retry_instruction = (
            f"\nCORREZIONE OBBLIGATORIA DEL TENTATIVO PRECEDENTE:\n"
            f"{corrective_instruction}\n"
            if corrective_instruction else ""
        )
        return f"""Filtra dal documento seguente il contenuto clinicamente rilevante conservando l'ordine originale.

REGOLE OBBLIGATORIE:
- restituisci soltanto il testo clinico, senza JSON, premesse, conclusioni o
  commenti sul lavoro svolto;
- elimina anagrafica, intestazioni amministrative, recapiti, codici, firme,
  prenotazioni, privacy e piè di pagina, anche quando nome del paziente,
  telefono, e-mail, indirizzo o identificativi compaiono nel corpo del referto;
- conserva integralmente diagnosi, anamnesi, sintomi, negazioni, esame
  obiettivo, valutazioni specialistiche, terapie con dose/via/frequenza e
  relative modifiche, procedure, ricoveri, radiologia, laboratorio,
  biomarcatori, eventi avversi, risposta/progressione e piani di cura;
- non parafrasare, non riassumere, non correggere, non interpretare, non
  classificare e non dedurre;
- conserva rigorosamente la sequenza dei paragrafi e delle osservazioni così
  come appare nella sorgente;
- non raggruppare informazioni simili, non costruire una nuova cronologia,
  non spostare contenuti tra sezioni e non ripetere paragrafi;
- non aggiungere diagnosi, causalità, grading, criteri RECIST/CTCAE o date;
- conserva letteralmente le formulazioni cliniche, le unità, le negazioni e
  le espressioni temporali originali;
- ogni token [[VALORE_X]] rappresenta una data o un numero: copialo
  esattamente, senza modificarlo, duplicarlo o sostituirlo e senza scrivere
  cifre direttamente; ogni segnaposto [[VALORE_X]] deve comparire
  ESATTAMENTE UNA VOLTA nella risposta;
- ogni token [[PAGINA_X]] delimita internamente una pagina della sorgente:
  non riportarlo nella risposta;
- conserva l'ordine originale dei segnaposto [[VALORE_X]];
- puoi soltanto regolarizzare spazi e interruzioni di riga, senza creare o
  rinominare titoli di sezione;
- se non esiste contenuto clinico, restituisci soltanto:
  [NESSUN CONTENUTO CLINICO]
{retry_instruction}

TESTO SORGENTE:
{text}
"""

    @staticmethod
    def _validation_category(error: Exception) -> str:
        message = str(error).lower()
        if "riordinato" in message:
            return "segnaposti riordinati"
        if "duplicat" in message:
            return "segnaposti duplicati"
        if "alterato" in message:
            return "segnaposti alterati"
        if "non presenti nella sorgente" in message:
            return "segnaposti sconosciuti"
        if "prodotti direttamente" in message:
            return "numeri o date aggiunti"
        if "non presenti esattamente" in message:
            return "numeri o date non fedeli"
        if "json" in message:
            return "formato JSON non consentito"
        if "vuoto" in message:
            return "risposta vuota"
        if "limite di token" in message or "token" in message:
            return "risposta troppo lunga"
        return "errore di validazione"

    @staticmethod
    def _corrective_instruction(category: str) -> str:
        instructions = {
            "segnaposti riordinati": (
                "Hai cambiato l'ordine dei segnaposti. Riproduci i paragrafi "
                "nello stesso ordine della sorgente; non raggruppare gli "
                "eventi cronologicamente e non spostare alcuna osservazione."
            ),
            "segnaposti duplicati": (
                "Hai duplicato uno o più segnaposti. Ogni segnaposto può "
                "comparire al massimo una volta; non ripetere paragrafi o "
                "valori già riportati."
            ),
            "segnaposti alterati": (
                "Hai modificato la sintassi di uno o più segnaposti. Copia "
                "ogni token letteralmente, incluse parentesi e underscore."
            ),
            "segnaposti sconosciuti": (
                "Hai prodotto segnaposti non presenti nella sorgente. Usa "
                "soltanto i token effettivamente forniti."
            ),
            "numeri o date aggiunti": (
                "Hai scritto cifre direttamente. Non produrre alcuna cifra: "
                "copia esclusivamente i segnaposti forniti."
            ),
            "numeri o date non fedeli": (
                "Hai alterato un numero o una data. Mantieni esclusivamente "
                "i segnaposti originali senza trasformarli."
            ),
            "formato JSON non consentito": (
                "Non restituire JSON, liste di oggetti o metadati: produci "
                "soltanto testo clinico."
            ),
            "risposta troppo lunga": (
                "La risposta precedente ha superato il limite di token. Sii "
                "più conciso: rimuovi il testo amministrativo ridondante e "
                "conserva soltanto il contenuto clinico essenziale."
            ),
            "risposta vuota": (
                "La risposta precedente era vuota. Riporta tutto il contenuto "
                "clinicamente rilevante mantenendo l'ordine della sorgente."
            ),
        }
        return instructions.get(
            category,
            "Ripeti l'estrazione rispettando letteralmente tutte le regole.",
        )

    def _normalize_with_section_fallback(
        self,
        source: str,
        document_date: str | None,
        primary_error: ClinicalTextValidationError,
    ) -> tuple[str, list[str], int]:
        sections = list(self._fallback_sections(source))
        if len(sections) <= 1:
            raise primary_error

        normalized_parts = []
        warnings = [
            f"fallback per {len(sections)} sezioni dopo "
            + ", ".join(dict.fromkeys(primary_error.categories))
        ]
        for index, section in enumerate(sections, start=1):
            try:
                normalized, corrections = self._normalize_chunk(
                    section, document_date
                )
            except ClinicalTextValidationError as exc:
                raise ClinicalTextValidationError(
                    primary_error.details + [
                        f"fallback sezione {index}/{len(sections)}: {exc}"
                    ],
                    primary_error.categories + [
                        "fallback per sezione fallito"
                    ],
                ) from exc
            except Exception as exc:
                raise RuntimeError(
                    f"fallback sezione {index}/{len(sections)}: {exc}"
                ) from exc
            if normalized != "[NESSUN CONTENUTO CLINICO]":
                normalized_parts.append(normalized)
            if corrections:
                warnings.append(
                    f"sezione {index}/{len(sections)}: retry riuscito dopo "
                    f"{corrections[-1]}"
                )

        return (
            "\n\n".join(normalized_parts).strip()
            or "[NESSUN CONTENUTO CLINICO]",
            warnings,
            len(sections),
        )

    def _fallback_sections(self, source: str) -> Iterable[str]:
        """Split a failed document by pages/paragraph groups, never by tokens."""
        page_blocks = self._page_blocks(source)
        sections = []
        for block in page_blocks:
            sections.extend(
                self._split_oversized_block(
                    block, FALLBACK_SECTION_CHARS
                )
            )

        if len(sections) <= 1 and len(source) >= MIN_FALLBACK_SOURCE_CHARS:
            target = max(MIN_CHUNK_CHARS, min(
                FALLBACK_SECTION_CHARS, len(source) // 2
            ))
            sections = list(self._split_oversized_block(source, target))

        yield from (section for section in sections if section.strip())

    @staticmethod
    def _is_systemic_error(error: Exception) -> bool:
        """Identify failures for which continuing the batch is unsafe."""
        if isinstance(error, ClinicalTextValidationError):
            return False
        message = str(error).lower()
        systemic_signals = (
            "llama-server",
            "ollama",
            "compute error",
            "server_error",
            "status code: 500",
            "status code 500",
            "connection",
            "connessione",
            "timed out",
            "timeout",
            "runner stopped",
            "insufficient memory",
            "out of memory",
            "modello documentale non",
        )
        return any(signal in message for signal in systemic_signals)

    @staticmethod
    def _clean_response(response: str) -> str:
        value = str(response or "").strip()
        if value.startswith("```") and value.endswith("```"):
            lines = value.splitlines()
            value = "\n".join(lines[1:-1]).strip()
        return value

    @classmethod
    def _validate_output(
        cls, output: str, source: str, document_date: str | None
    ) -> None:
        del document_date
        if not output:
            raise ValueError("il modello ha restituito testo vuoto")
        if cls._is_json(output):
            raise ValueError("il modello ha restituito JSON invece di testo")
        if output == "[NESSUN CONTENUTO CLINICO]":
            return

        source_numbers = Counter(cls._numeric_literals(source))
        output_numbers = Counter(cls._numeric_literals(output))
        unsupported = []
        for literal, count in output_numbers.items():
            excess = count - source_numbers.get(literal, 0)
            if excess > 0:
                unsupported.extend([literal] * excess)
        if unsupported:
            raise ValueError(
                "numeri/date non presenti esattamente nella sorgente: "
                + ", ".join(sorted(unsupported)[:8])
            )

    @staticmethod
    def _is_json(value: str) -> bool:
        if not value.lstrip().startswith(("{", "[")):
            return False
        try:
            parsed = json.loads(value)
            return isinstance(parsed, (dict, list))
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _numbers(value: str) -> list[str]:
        """Backward-compatible alias for exact numeric literal extraction."""
        return ClinicalTextIsolator._numeric_literals(value)

    @staticmethod
    def _numeric_literals(value: str) -> list[str]:
        without_page_markers = _PAGE_MARKER_RE.sub("", value or "")
        return [
            match.group(0).strip()
            for match in _NUMERIC_LITERAL_RE.finditer(
                without_page_markers
            )
        ]

    @classmethod
    def _protect_numeric_literals(
        cls, source: str
    ) -> tuple[str, dict[str, str]]:
        """Replace page markers and numeric literals with alphabetic tokens."""
        combined = re.compile(
            rf"{_PAGE_MARKER_RE.pattern}|{_NUMERIC_LITERAL_RE.pattern}",
            re.IGNORECASE,
        )
        replacements: dict[str, str] = {}
        position = 0

        def replace(match: re.Match) -> str:
            nonlocal position
            kind = (
                "PAGINA"
                if _PAGE_MARKER_RE.fullmatch(match.group(0))
                else "VALORE"
            )
            placeholder = (
                f"[[{kind}_{cls._alphabetic_id(position)}]]"
            )
            position += 1
            replacements[placeholder] = match.group(0)
            return placeholder

        return combined.sub(replace, source), replacements

    @classmethod
    def _restore_numeric_literals(
        cls, output: str, replacements: dict[str, str]
    ) -> str:
        if not output:
            raise ValueError("il modello ha restituito testo vuoto")
        if cls._is_json(output):
            raise ValueError("il modello ha restituito JSON invece di testo")
        if output == "[NESSUN CONTENUTO CLINICO]":
            return output

        placeholders = _PLACEHOLDER_RE.findall(output)
        unknown = sorted(set(placeholders) - set(replacements))
        if unknown:
            raise ValueError(
                "segnaposto non presenti nella sorgente: "
                + ", ".join(unknown[:4])
            )
        duplicated = sorted(
            token for token, count in Counter(placeholders).items()
            if count > 1
        )
        if duplicated:
            raise ValueError(
                "segnaposto duplicati dal modello: "
                + ", ".join(duplicated[:4])
            )
        source_order = {
            placeholder: index
            for index, placeholder in enumerate(replacements)
        }
        observed_order = [source_order[token] for token in placeholders]
        if observed_order != sorted(observed_order):
            raise ValueError("il modello ha riordinato i segnaposto")

        output_without_placeholders = _PLACEHOLDER_RE.sub("", output)
        raw_numbers = cls._numeric_literals(output_without_placeholders)
        if raw_numbers:
            raise ValueError(
                "numeri/date prodotti direttamente dal modello: "
                + ", ".join(sorted(raw_numbers)[:8])
            )

        restored = output
        for placeholder, original in replacements.items():
            restored = restored.replace(placeholder, original)
        if _MALFORMED_PLACEHOLDER_RE.search(restored):
            raise ValueError("il modello ha alterato uno o più segnaposto")
        return restored

    @staticmethod
    def _alphabetic_id(index: int) -> str:
        """Return A..Z, AA..AZ... without introducing digits."""
        value = index + 1
        letters = []
        while value:
            value, remainder = divmod(value - 1, 26)
            letters.append(chr(ord("A") + remainder))
        return "".join(reversed(letters))

    @staticmethod
    def _strip_page_markers(output: str) -> str:
        """Remove internal provenance labels from user-facing clinical text."""
        value = re.sub(
            r"\[(?:PAGINA\s+\d+|PAGINE\s+SORGENTE\s*:[^\]]+)\]",
            "",
            output,
            flags=re.IGNORECASE,
        )
        value = _PAGE_COMMENT_RE.sub("", value)
        value = re.sub(r"[ \t]+\n", "\n", value)
        value = re.sub(r"\n{3,}", "\n\n", value)
        return value.strip()

    @staticmethod
    def _prepare_source(text: str, parsing_result) -> str:
        source = _PAGE_COMMENT_RE.sub(
            lambda match: f"[PAGINA {match.group(1)}]",
            str(text or ""),
        ).strip()
        if _PAGE_MARKER_RE.search(source):
            return source
        if parsing_result and getattr(parsing_result, "pages", None):
            return "\n\n".join(
                f"[PAGINA {page.page}]\n{page.text.strip()}"
                for page in parsing_result.pages if page.text.strip()
            ).strip()
        return source

    def _source_character_budget(self) -> int:
        context = int(
            getattr(self.llm, "context_length", DEFAULT_CONTEXT_LENGTH)
            or DEFAULT_CONTEXT_LENGTH
        )
        configured_output = int(
            getattr(
                self.llm,
                "max_output_tokens",
                DEFAULT_MAX_OUTPUT_TOKENS,
            )
            or DEFAULT_MAX_OUTPUT_TOKENS
        )
        # The filtered output can approach the source size for dense
        # clinical text.  Bound the source so the model's response
        # never exceeds ``max_output_tokens``.  2× is conservative
        # (filtering typically removes ≥50 % of administrative text).
        max_source_for_output = configured_output * 2
        # Also honour the model's total context window.
        max_source_for_context = max(
            480, context - configured_output - PROMPT_TOKEN_RESERVE
        )
        available_tokens = min(max_source_for_context, max_source_for_output)
        return max(
            MIN_CHUNK_CHARS,
            int(available_tokens * CHARS_PER_TOKEN_ESTIMATE),
        )

    def _chunks(
        self,
        text: str,
        parsing_result=None,
        max_chars: int | None = None,
    ) -> Iterable[str]:
        """Use the whole document first; split semantically only if required."""
        source = self._prepare_source(text, parsing_result)
        if not source:
            return
        limit = max_chars or self._source_character_budget()
        if len(source) <= limit:
            yield source
            return

        blocks = self._page_blocks(source)
        buffer = []
        buffer_size = 0
        for block in blocks:
            pieces = list(self._split_oversized_block(block, limit))
            for piece in pieces:
                separator_size = 2 if buffer else 0
                if (
                    buffer
                    and buffer_size + separator_size + len(piece) > limit
                ):
                    yield "\n\n".join(buffer)
                    buffer = []
                    buffer_size = 0
                buffer.append(piece)
                buffer_size += separator_size + len(piece)
        if buffer:
            yield "\n\n".join(buffer)

    @staticmethod
    def _page_blocks(source: str) -> list[str]:
        matches = list(_PAGE_MARKER_RE.finditer(source))
        if not matches:
            return [source]
        blocks = []
        prefix = source[:matches[0].start()].strip()
        for index, match in enumerate(matches):
            end = (
                matches[index + 1].start()
                if index + 1 < len(matches) else len(source)
            )
            block = source[match.start():end].strip()
            if index == 0 and prefix:
                block = f"{prefix}\n{block}"
            if block:
                blocks.append(block)
        return blocks

    @classmethod
    def _split_oversized_block(
        cls, block: str, max_chars: int
    ) -> Iterable[str]:
        if len(block) <= max_chars:
            yield block
            return

        marker_match = _PAGE_MARKER_RE.match(block)
        marker = marker_match.group(0) if marker_match else ""
        payload = (
            block[marker_match.end():].strip()
            if marker_match else block
        )
        capacity = max(
            MIN_CHUNK_CHARS,
            max_chars - len(marker) - (1 if marker else 0),
        )
        while payload:
            if len(payload) <= capacity:
                piece, payload = payload, ""
            else:
                candidates = [
                    payload.rfind("\n\n", 0, capacity),
                    payload.rfind("\n", 0, capacity),
                    payload.rfind(". ", 0, capacity),
                    payload.rfind(" ", 0, capacity),
                ]
                boundary = max(candidates)
                if boundary < capacity // 2:
                    boundary = capacity
                elif payload[boundary:boundary + 2] == ". ":
                    boundary += 1
                piece = payload[:boundary]
                payload = payload[boundary:].lstrip()
            piece = piece.strip()
            if piece:
                yield f"{marker}\n{piece}".strip() if marker else piece
