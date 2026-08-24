"""Tests for application-wide cancellation of GUI background workers."""

from types import SimpleNamespace
from unittest.mock import patch

from PyQt5.QtCore import QThread

from emr_analyzer.gui import application_shutdown


class _CooperativeWorker(QThread):
    def __init__(self):
        super().__init__()
        self.cancel_called = False

    def cancel(self) -> None:
        self.cancel_called = True
        self.requestInterruption()

    def run(self) -> None:
        while not self.isInterruptionRequested():
            self.msleep(5)


def test_running_dialog_worker_is_discovered_and_cancelled():
    worker = _CooperativeWorker()
    worker.start()
    try:
        widget = SimpleNamespace(_worker=worker)
        app = SimpleNamespace(allWidgets=lambda: [widget])
        with patch.object(
            application_shutdown.QApplication, "instance", return_value=app
        ):
            threads = application_shutdown.running_qthreads()
            assert threads == [worker]
            assert application_shutdown.request_qthread_shutdown(threads) == 1

        assert worker.cancel_called
        assert worker.wait(1000)
    finally:
        worker.requestInterruption()
        worker.wait(1000)
