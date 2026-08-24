"""Network-capable helper used only for explicit model installations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Also works when launched by absolute filename from a source checkout.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from emr_analyzer.llm_backend.model_installer import (  # noqa: E402
    ModelInstallError,
    install_from_ollama,
    install_from_url,
    update_model_catalog,
)


def _event(kind: str, **payload) -> None:
    print(
        json.dumps({"type": kind, **payload}, ensure_ascii=False),
        flush=True,
    )


def _progress(completed: int, total: int | None, message: str) -> None:
    _event(
        "progress",
        completed=int(completed),
        total=int(total) if total is not None else None,
        message=message,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--ollama-tag")
    source.add_argument("--url")
    source.add_argument("--update-catalog", action="store_true")
    parser.add_argument("--name", default="")
    parser.add_argument("--sha256", default="")
    arguments = parser.parse_args(argv)

    try:
        if arguments.update_catalog:
            result = update_model_catalog(progress=_progress)
            _event("catalog_success", **result)
            return 0
        if arguments.ollama_tag:
            name, entry = install_from_ollama(
                arguments.ollama_tag,
                local_name=arguments.name or None,
                progress=_progress,
            )
        else:
            name, entry = install_from_url(
                arguments.url,
                local_name=arguments.name or None,
                expected_sha256=arguments.sha256 or None,
                progress=_progress,
            )
        _event("success", name=name, entry=entry)
        return 0
    except ModelInstallError as exc:
        _event("error", message=str(exc))
        return 2
    except Exception as exc:  # pragma: no cover - last-resort process boundary
        _event("error", message=f"Errore inatteso: {exc}")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
