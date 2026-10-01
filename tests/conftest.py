"""Isolated test environment: no user project, shared lexicon or LLM server."""

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Point the active project to a temporary folder for one test."""
    from emr_analyzer.config import active_workspace

    previous = active_workspace.path
    active_workspace.set_path(tmp_path / "project")
    yield active_workspace.path
    active_workspace.__class__.path = previous


@pytest.fixture(scope="session")
def qapp():
    from PyQt5.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])
