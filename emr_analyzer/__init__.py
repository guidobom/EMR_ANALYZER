"""EMR Analyzer package bootstrap."""

from pathlib import Path
import os
import sys


__version__ = "0.7.0"


def _configure_qt_plugins() -> None:
    """Use conda's Qt plugins when PyQt reports a stale bundled path.

    A mixed pip/conda repair can leave ``QLibraryInfo`` pointing at the
    removed pip runtime even though the conda Qt libraries are valid. Setting
    the plugin root before QApplication is constructed is deterministic and
    avoids a native abort.
    """
    if os.environ.get("QT_PLUGIN_PATH"):
        return
    for candidate in (
        Path(sys.prefix) / "plugins",
        Path(os.environ.get("CONDA_PREFIX", "")) / "plugins",
    ):
        if (candidate / "platforms").is_dir():
            os.environ["QT_PLUGIN_PATH"] = str(candidate)
            return


_configure_qt_plugins()
