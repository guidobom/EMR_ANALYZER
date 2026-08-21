"""Shared test process fixtures."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

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
