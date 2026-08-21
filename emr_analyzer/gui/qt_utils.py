"""Small Qt runtime helpers shared by synchronous GUI workflows."""

from __future__ import annotations

import os

from PyQt5.QtWidgets import QApplication


def process_gui_events() -> None:
    """Pump GUI events only on an interactive Qt platform.

    Offscreen/minimal plugins are used by tests and have no window server to
    refresh; repeatedly pumping them can abort natively after widget teardown.
    """
    if os.environ.get("QT_QPA_PLATFORM", "").casefold() in {
        "offscreen", "minimal",
    }:
        return
    application = QApplication.instance()
    if application is not None:
        application.processEvents()
