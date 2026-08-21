"""Process-level offline policy for clinical processing.

The policy blocks network connections initiated by this Python process except
for loopback connections (for example llama-server on localhost).  An explicit
model installation may launch the dedicated ``model_download_helper`` process;
that helper receives only a model tag/URL and destination, never a workspace or
clinical content.  The clinical process itself remains loopback-only.
"""

from __future__ import annotations

import ipaddress
import os
import socket
import threading
from dataclasses import dataclass
from typing import Any, ClassVar
from urllib.parse import urlparse


class OfflineViolation(RuntimeError):
    """Raised when an outbound operation violates the offline policy."""


def is_loopback_host(host: Any) -> bool:
    if isinstance(host, bytes):
        try:
            host = host.decode("ascii")
        except UnicodeDecodeError:
            return False
    if not isinstance(host, str):
        return False
    normalized = host.strip().lower().rstrip(".")
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def require_loopback_url(url: str) -> str:
    """Return a normalized URL or reject any non-loopback endpoint."""

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise OfflineViolation("Only HTTP(S) loopback endpoints are allowed")
    if not parsed.hostname or not is_loopback_host(parsed.hostname):
        raise OfflineViolation("Clinical offline mode only permits loopback endpoints")
    if parsed.username or parsed.password:
        raise OfflineViolation("Credentials in endpoint URLs are not allowed")
    return url.rstrip("/")


@dataclass
class OfflinePolicy:
    """Configure library offline flags and optionally block process networking."""

    strict_network: bool = True

    _lock: ClassVar = threading.Lock()
    _installed: ClassVar[bool] = False
    _original_connect: ClassVar = None
    _original_connect_ex: ClassVar = None
    _original_create_connection: ClassVar = None
    _original_getaddrinfo: ClassVar = None

    def activate(self) -> None:
        self.configure_environment()
        if self.strict_network:
            self.install_network_guard()

    @staticmethod
    def configure_environment() -> None:
        # Prevent common model/document libraries from contacting remote hubs
        # or sending telemetry during a clinical processing session.
        for name, value in {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "DO_NOT_TRACK": "1",
        }.items():
            os.environ[name] = value

    @classmethod
    def install_network_guard(cls) -> None:
        """Install an idempotent loopback-only guard for the Python process."""

        with cls._lock:
            if cls._installed:
                return
            cls._original_connect = socket.socket.connect
            cls._original_connect_ex = socket.socket.connect_ex
            cls._original_create_connection = socket.create_connection
            cls._original_getaddrinfo = socket.getaddrinfo

            original_connect = cls._original_connect
            original_connect_ex = cls._original_connect_ex
            original_create_connection = cls._original_create_connection
            original_getaddrinfo = cls._original_getaddrinfo

            def guarded_connect(sock, address):
                cls._require_allowed_address(address)
                return original_connect(sock, address)

            def guarded_connect_ex(sock, address):
                cls._require_allowed_address(address)
                return original_connect_ex(sock, address)

            def guarded_create_connection(address, *args, **kwargs):
                cls._require_allowed_address(address)
                return original_create_connection(address, *args, **kwargs)

            def guarded_getaddrinfo(host, *args, **kwargs):
                if host is not None and not is_loopback_host(host):
                    raise OfflineViolation(
                        "DNS resolution is blocked by clinical offline mode"
                    )
                return original_getaddrinfo(host, *args, **kwargs)

            socket.socket.connect = guarded_connect
            socket.socket.connect_ex = guarded_connect_ex
            socket.create_connection = guarded_create_connection
            socket.getaddrinfo = guarded_getaddrinfo
            cls._installed = True

    @classmethod
    def _require_allowed_address(cls, address) -> None:
        # AF_UNIX sockets use a filesystem path rather than a (host, port)
        # tuple and never leave the machine.
        if isinstance(address, (str, bytes)):
            return
        if not isinstance(address, tuple) or not address:
            raise OfflineViolation("Unsupported network address in offline mode")
        if not is_loopback_host(address[0]):
            raise OfflineViolation("Outbound network access is blocked in clinical mode")

    @classmethod
    def _reset_for_tests(cls) -> None:
        """Restore socket functions. Intended only for isolated tests."""

        with cls._lock:
            if not cls._installed:
                return
            socket.socket.connect = cls._original_connect
            socket.socket.connect_ex = cls._original_connect_ex
            socket.create_connection = cls._original_create_connection
            socket.getaddrinfo = cls._original_getaddrinfo
            cls._installed = False
