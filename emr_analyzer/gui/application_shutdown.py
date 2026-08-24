"""Application-wide shutdown coordination for background work.

Qt refuses to destroy a running ``QThread`` and a local LLM HTTP request can
legitimately have a very long timeout.  On an explicit application exit we
therefore request cooperative cancellation first and arm a short, daemon
timer as a last-resort process exit.  SQLite transactions remain atomic; the
startup recovery already resets document/run states left as ``processing``.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterable

from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import QApplication


_shutdown_requested = threading.Event()
_timer_lock = threading.Lock()
_emergency_timer: threading.Timer | None = None


def shutdown_requested() -> bool:
    """Whether the user has confirmed application shutdown."""

    return _shutdown_requested.is_set()


def mark_shutdown_requested() -> None:
    """Make long-running synchronous loops stop at their next safe point."""

    _shutdown_requested.set()


def running_qthreads() -> list[QThread]:
    """Return running QThreads referenced by any currently open widget.

    Dialog-specific workers are commonly kept in ``_worker`` attributes or
    dictionaries.  Looking only at the central workspace would miss model
    downloads, model warm-up, hypothesis discovery and workspace merges.
    """

    app = QApplication.instance()
    if app is None:
        return []
    found: list[QThread] = []
    seen: set[int] = set()
    for widget in app.allWidgets():
        for value in getattr(widget, "__dict__", {}).values():
            for thread in _threads_in_value(value):
                identity = id(thread)
                if identity in seen:
                    continue
                seen.add(identity)
                try:
                    is_running = thread.isRunning()
                except RuntimeError:
                    is_running = False
                if is_running:
                    found.append(thread)
    return found


def request_qthread_shutdown(threads: Iterable[QThread] | None = None) -> int:
    """Request cancellation/interruption of all running GUI QThreads."""

    active = list(threads) if threads is not None else running_qthreads()
    for thread in active:
        cancel = getattr(thread, "cancel", None)
        if callable(cancel):
            try:
                cancel()
            except Exception:
                pass
        try:
            thread.requestInterruption()
        except RuntimeError:
            pass
    return len(active)


def schedule_emergency_exit(
    *, delay_seconds: float = 5.0, exit_code: int = 0
) -> threading.Timer:
    """Guarantee exit if cooperative cleanup becomes stuck.

    ``os._exit`` is deliberately confined to this explicit, user-confirmed
    path.  The timer is a daemon and is harmless when normal process shutdown
    completes before the deadline.
    """

    global _emergency_timer
    with _timer_lock:
        if _emergency_timer is not None and _emergency_timer.is_alive():
            return _emergency_timer
        timer = threading.Timer(
            max(0.1, float(delay_seconds)), os._exit, args=(int(exit_code),)
        )
        timer.daemon = True
        timer.name = "emr-analyzer-emergency-exit"
        _emergency_timer = timer
        timer.start()
        return timer


def _threads_in_value(value) -> Iterable[QThread]:
    if isinstance(value, QThread):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            if isinstance(item, QThread):
                yield item
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            if isinstance(item, QThread):
                yield item


def _reset_for_tests() -> None:
    """Reset global state; intended only for isolated unit tests."""

    global _emergency_timer
    _shutdown_requested.clear()
    with _timer_lock:
        if _emergency_timer is not None:
            _emergency_timer.cancel()
        _emergency_timer = None
