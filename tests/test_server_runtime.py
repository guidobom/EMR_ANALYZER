"""Managed llama-server installation tests (synthetic executables only)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from emr_analyzer.llm_backend.server_manager import find_server_binary
from emr_analyzer.llm_backend.server_runtime import (
    ServerRuntimeError,
    active_managed_server,
    deactivate_managed_server,
    install_managed_server,
)


def _fake_server(path: Path, *, device: str = "Metal: Apple Test GPU") -> Path:
    path.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = \"--version\" ]; then\n"
        "  echo 'version: 10470 built for Darwin arm64'\n"
        "  exit 0\n"
        "fi\n"
        "echo 'Available devices:'\n"
        f"echo '  {device}'\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def test_install_copies_activates_and_rechecks_checksum():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = _fake_server(root / "source-server")
        runtime_dir = root / "managed"
        index_path = runtime_dir / "index.json"

        installed = install_managed_server(
            source,
            runtime_dir=runtime_dir,
            index_path=index_path,
            system_name="Darwin",
            machine="arm64",
        )
        active = active_managed_server(
            runtime_dir=runtime_dir,
            index_path=index_path,
            system_name="Darwin",
            machine="arm64",
        )

        assert active is not None
        assert active.binary_path == installed.binary_path
        assert active.binary_path != source
        assert active.backend == "METAL"
        assert active.sha256 == installed.sha256

        active.binary_path.write_text("tampered", encoding="utf-8")
        assert active_managed_server(
            runtime_dir=runtime_dir,
            index_path=index_path,
            system_name="Darwin",
            machine="arm64",
        ) is None


def test_install_rejects_macos_binary_without_exposed_metal_device():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = _fake_server(root / "source-server", device="BLAS: Accelerate")

        with pytest.raises(ServerRuntimeError, match="non espone un dispositivo METAL"):
            install_managed_server(
                source,
                runtime_dir=root / "managed",
                index_path=root / "managed/index.json",
                system_name="Darwin",
                machine="arm64",
            )


def test_install_checks_expected_checksum_before_execution():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = _fake_server(root / "source-server")

        with pytest.raises(ServerRuntimeError, match="Checksum SHA-256"):
            install_managed_server(
                source,
                expected_sha256="0" * 64,
                runtime_dir=root / "managed",
                index_path=root / "managed/index.json",
                system_name="Darwin",
                machine="arm64",
            )


def test_deactivate_keeps_installation_but_clears_active_selection():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = _fake_server(root / "source-server")
        runtime_dir = root / "managed"
        index_path = runtime_dir / "index.json"
        installed = install_managed_server(
            source,
            runtime_dir=runtime_dir,
            index_path=index_path,
            system_name="Darwin",
            machine="arm64",
        )

        assert deactivate_managed_server(index_path=index_path)
        assert installed.binary_path.exists()
        assert active_managed_server(
            runtime_dir=runtime_dir,
            index_path=index_path,
            system_name="Darwin",
            machine="arm64",
        ) is None


def test_server_discovery_prefers_active_managed_runtime():
    runtime = SimpleNamespace(binary_path=Path("/managed/llama-server"))
    with (
        patch(
            "emr_analyzer.llm_backend.server_runtime.active_managed_server",
            return_value=runtime,
        ),
        patch(
            "emr_analyzer.llm_backend.server_manager."
            "find_external_server_binary",
            return_value="/external/llama-server",
        ),
    ):
        assert find_server_binary() == "/managed/llama-server"
