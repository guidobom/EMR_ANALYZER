"""Local GGUF model directory and its JSON index.

The setup script (``tools/setup_llama_backend.py``) copies GGUF files into
``~/.emr_analyzer/models/`` and writes ``index.json`` with the metadata the
runtime needs (file path, size, architecture, maximum context).  The runtime
never parses GGUF files nor shells out for metadata: everything it needs is a
lookup on this index, which keeps the app offline-safe and fast.

The GGUF metadata reader lives here as well and is used only by the setup
script (and by tests).
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

from ..config import LLM_MODELS_DIR, LLM_MODEL_INDEX_PATH

# ---------------------------------------------------------------------------
# GGUF metadata parsing (pure Python, setup-time only)
# ---------------------------------------------------------------------------

_GGUF_MAGIC = b"GGUF"

# GGUF value types (little-endian uint32 tags).
_GGUF_UINT8 = 0
_GGUF_INT8 = 1
_GGUF_UINT16 = 2
_GGUF_INT16 = 3
_GGUF_UINT32 = 4
_GGUF_INT32 = 5
_GGUF_FLOAT32 = 6
_GGUF_BOOL = 7
_GGUF_STRING = 8
_GGUF_ARRAY = 9
_GGUF_UINT64 = 10
_GGUF_INT64 = 11
_GGUF_FLOAT64 = 12

_SKIP_SIZES = {
    _GGUF_UINT8: 1, _GGUF_INT8: 1,
    _GGUF_UINT16: 2, _GGUF_INT16: 2,
    _GGUF_UINT32: 4, _GGUF_INT32: 4,
    _GGUF_FLOAT32: 4, _GGUF_BOOL: 1,
    _GGUF_UINT64: 8, _GGUF_INT64: 8,
    _GGUF_FLOAT64: 8,
}

_INT_TYPES = {_GGUF_UINT32, _GGUF_INT32, _GGUF_UINT64, _GGUF_INT64}


def read_gguf_metadata(path: str | Path) -> dict | None:
    """Read the few metadata fields the app needs from a GGUF header.

    Returns ``{"architecture": str, "max_context_length": int}`` or ``None``
    when the file is not a valid GGUF.  Never raises on malformed input.
    """
    try:
        with open(path, "rb") as handle:
            if handle.read(4) != _GGUF_MAGIC:
                return None
            header = handle.read(4 + 8 + 8)
            if len(header) < 20:
                return None
            _, tensor_count, kv_count = struct.unpack("<IQQ", header)
            if kv_count > 100_000:  # sanity bound on malformed headers
                return None

            architecture = None
            context_length = None
            context_key = None
            for _ in range(kv_count):
                raw_len = handle.read(8)
                if len(raw_len) < 8:
                    return None
                (key_len,) = struct.unpack("<Q", raw_len)
                if key_len > 4096:
                    return None
                key = handle.read(key_len).decode("utf-8", errors="replace")
                (raw_type,) = struct.unpack("<I", handle.read(4))
                value = _read_value(handle, raw_type)
                if key == "general.architecture" and isinstance(value, str):
                    architecture = value
                elif key.endswith(".context_length") and isinstance(value, int):
                    if key.startswith(architecture or ""):
                        # Exact <arch>.context_length wins over any other match,
                        # even when a different one was seen earlier.
                        context_key = key
                        context_length = value
                    elif context_key is None:
                        context_length = value
                if architecture and context_key is not None:
                    break

        if architecture is None and context_length is None:
            return None
        return {
            "architecture": architecture or "",
            "max_context_length": context_length,
        }
    except (OSError, struct.error, ValueError):
        return None


def _read_value(handle, value_type: int):
    """Read one GGUF value; returns str for strings, int for ints, else None.

    Short reads raise :class:`ValueError` so the caller can treat the file
    as truncated instead of silently decoding partial values.
    """
    if value_type == _GGUF_STRING:
        raw = handle.read(8)
        if len(raw) < 8:
            raise ValueError("truncated string length")
        (raw_len,) = struct.unpack("<Q", raw)
        if raw_len > 64 * 1024 * 1024:  # sanity bound
            return None
        payload = handle.read(raw_len)
        if len(payload) < raw_len:
            raise ValueError("truncated string payload")
        return payload.decode("utf-8", errors="replace")
    if value_type == _GGUF_ARRAY:
        raw = handle.read(4)
        if len(raw) < 4:
            raise ValueError("truncated array type")
        (element_type,) = struct.unpack("<I", raw)
        raw = handle.read(8)
        if len(raw) < 8:
            raise ValueError("truncated array count")
        (count,) = struct.unpack("<Q", raw)
        if count > 1_000_000:  # sanity bound
            return None
        element_size = _SKIP_SIZES.get(element_type)
        for _ in range(count):
            if element_size is not None:
                chunk = handle.read(element_size)
                if len(chunk) < element_size:
                    raise ValueError("truncated array element")
            else:
                _read_value(handle, element_type)
        return None
    size = _SKIP_SIZES.get(value_type)
    if size is not None:
        raw = handle.read(size)
        if len(raw) < size:
            raise ValueError("truncated value")
        if value_type in _INT_TYPES:
            return int.from_bytes(raw, "little", signed=value_type in
                                  (_GGUF_INT32, _GGUF_INT64))
    return None


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------


def load_index(path: str | Path = LLM_MODEL_INDEX_PATH) -> dict:
    """Load the model index; ``{}`` when absent or unreadable."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def save_index(index: dict, path: str | Path = LLM_MODEL_INDEX_PATH) -> None:
    """Atomically persist the model index."""
    index_path = Path(path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = index_path.with_suffix(index_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(index_path)


def list_models(path: str | Path = LLM_MODEL_INDEX_PATH) -> list[str]:
    """Friendly names of the locally available GGUF models, sorted."""
    return sorted(name for name in load_index(path))


def resolve(
    name: str, path: str | Path = LLM_MODEL_INDEX_PATH
) -> dict | None:
    """Resolve a (possibly legacy) model name to an index entry.

    Accepts ``family-tag`` style names; legacy Ollama names like
    ``qwen3:14b`` are normalized first (``:`` → ``-``).
    """
    raw = str(name or "").strip()
    if not raw:
        return None
    index = load_index(path)
    normalized = raw.removesuffix(":latest").replace(":", "-")

    if normalized in index:
        return index[normalized]
    # Case-insensitive exact match on the friendly name.
    folded = normalized.lower()
    for candidate in index:
        if candidate.lower() == folded:
            return index[candidate]
    # Fuzzy: family prefix match, prefer the largest model file.
    family = folded.split("-", 1)[0]
    matches = [
        candidate for candidate in index
        if candidate.lower().startswith(f"{family}-")
    ]
    if not matches and family:
        matches = [
            candidate for candidate in index
            if candidate.lower().startswith(family)
        ]
    if not matches:
        return None
    best = max(matches, key=lambda candidate: index[candidate].get("size_bytes", 0))
    return index[best]
