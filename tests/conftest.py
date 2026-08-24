"""Shared test process fixtures."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

if __name__ == "__main__":
    raise SystemExit(
        "tests/conftest.py non è uno script eseguibile: viene caricato "
        "automaticamente da pytest.\n"
        "Installa le dipendenze di sviluppo con:\n"
        "  python -m pip install -r requirements-dev.txt\n"
        "Poi esegui la suite dalla radice del progetto con:\n"
        "  python -m pytest -q"
    )

import pytest

# Importing the package configures a valid conda Qt plugin root before the
# first QApplication. A session-owned reference prevents native teardown and
# recreation between unittest-style GUI classes in the same pytest process.
import emr_analyzer  # noqa: F401,E402
from PyQt5.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def qt_application():
    application = QApplication.instance() or QApplication([])
    yield application
