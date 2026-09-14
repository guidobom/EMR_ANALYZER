"""Large GGUF counters must survive Qt's queued signal transport."""
from PyQt5.QtCore import QObject, Qt, pyqtSlot
from PyQt5.QtWidgets import QApplication, QLabel, QProgressBar

from emr_analyzer.gui.model_manager_dialog import ModelManagerDialog, _ModelInstallWorker


class Receiver(QObject):
    def __init__(self):
        super().__init__()
        self.received = []
        self._progress = QProgressBar()
        self._status = QLabel()

    @pyqtSlot(object, object, str)
    def receive(self, completed, total, message):
        self.received.append((completed, total))
        ModelManagerDialog._on_progress(self, completed, total, message)


def test_queued_progress_preserves_byte_counts_above_2_and_16_gib():
    total = 20 * 1024**3
    counts = [2**31 - 1, 2**31 + 1, 2**32 + 1, 16 * 1024**3, total]

    class Worker(_ModelInstallWorker):
        def run(self):
            for completed in counts:
                self.progress_changed.emit(completed, total, 'Download GGUF…')

    receiver = Receiver()
    worker = Worker([])
    worker.progress_changed.connect(receiver.receive, Qt.QueuedConnection)
    worker.start()
    assert worker.wait(5000)
    QApplication.processEvents()
    assert receiver.received == [(completed, total) for completed in counts]
    assert receiver._progress.value() == 1000
    assert receiver._progress.format() == '20.00 / 20.00 GiB — %p%'


def test_large_incomplete_download_is_not_shown_as_complete():
    receiver = Receiver()
    receiver.receive(16 * 1024**3, 20 * 1024**3, 'Download GGUF…')
    assert receiver._progress.value() == 800
    assert receiver._progress.format() == '16.00 / 20.00 GiB — %p%'
