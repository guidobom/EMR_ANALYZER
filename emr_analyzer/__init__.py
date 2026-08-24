"""EMR Analyzer package bootstrap."""

from pathlib import Path
import os
import sys


__version__ = "0.7.0"


def _configure_qt_plugins() -> None:
    """Keep PyQt and its platform plugins in the same Python environment.

    A parent shell may expose the *base* ``CONDA_PREFIX`` while executing the
    Python binary of a named environment.  Loading base Qt plugins beside the
    pip PyQt frameworks of that environment creates duplicate Objective-C
    classes and aborts QApplication.  Prefer the plugins shipped beside the
    imported PyQt package; only then consider an environment-level fallback.
    """
    if os.environ.get("QT_PLUGIN_PATH"):
        return
    python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidates = (
        Path(sys.prefix) / "lib" / python_version
        / "site-packages" / "PyQt5" / "Qt5" / "plugins",
        Path(sys.prefix) / "Lib" / "site-packages"
        / "PyQt5" / "Qt5" / "plugins",
        Path(sys.prefix) / "plugins",
    )
    for candidate in candidates:
        if (candidate / "platforms").is_dir():
            os.environ["QT_PLUGIN_PATH"] = str(candidate)
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(
                candidate / "platforms"
            )
            return


_configure_qt_plugins()
