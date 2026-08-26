"""Process-wide SNOMED release and index cache.

The registered RF2 release is never committed to the repository; it lives in a
user-supplied directory.  This module keeps one immutable snapshot + index per
``(release_dir, languages)`` pair, shared by every parallel registry worker,
and re-loads transparently when the release directory content changes (the
snapshot digest is recomputed on demand).
"""

from __future__ import annotations

import threading
from typing import Iterable

from .gps_loader import load_gps
from .index import SnomedIndex
from .rf2_loader import ReleaseSnapshot, load_snapshot

#: Source kinds the manager can load.  ``"rf2"`` is the licensed release; the
#: ``"gps"`` interim kind loads the CC BY-ND Global Patient Set freeset.  The
#: RF2 path is byte-identical to the pre-GPS behaviour (default kind).
_SOURCE_KINDS = ("rf2", "gps")


class _ReleaseManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshots: dict[
            tuple[str, tuple[str, ...], str], ReleaseSnapshot
        ] = {}
        self._indexes: dict[
            tuple[str, tuple[str, ...], str], SnomedIndex
        ] = {}

    def snapshot(
        self,
        release_dir: str,
        languages: Iterable[str] = ("en", "it"),
        kind: str = "rf2",
    ) -> ReleaseSnapshot:
        if kind not in _SOURCE_KINDS:
            raise ValueError(
                f"kind sconosciuto: {kind!r} (attesi {', '.join(_SOURCE_KINDS)})"
            )
        key = (str(release_dir), tuple(languages), kind)
        with self._lock:
            cached = self._snapshots.get(key)
            if cached is not None:
                # Recompute the digest to detect an on-disk change without
                # re-parsing; a changed digest forces a reload.
                if cached.digest == _snapshot_digest_from_disk(cached):
                    return cached
            snapshot = (
                load_gps(release_dir)
                if kind == "gps"
                else load_snapshot(release_dir, languages=languages)
            )
            self._snapshots[key] = snapshot
            self._indexes.pop(key, None)
            return snapshot

    def index(
        self,
        release_dir: str,
        languages: Iterable[str] = ("en", "it"),
        kind: str = "rf2",
    ) -> SnomedIndex:
        key = (str(release_dir), tuple(languages), kind)
        snapshot = self.snapshot(release_dir, languages, kind)
        with self._lock:
            cached = self._indexes.get(key)
            if cached is None:
                cached = SnomedIndex(snapshot.iter_concepts())
                self._indexes[key] = cached
            return cached

    def clear(self) -> None:
        with self._lock:
            self._snapshots.clear()
            self._indexes.clear()


# Default process-wide manager.  Tests that need isolation can construct their
# own instance.
manager = _ReleaseManager()


def _snapshot_digest_from_disk(snapshot: ReleaseSnapshot) -> str:
    """Cheap fingerprint of the loaded release used to detect a swap."""
    from .rf2_loader import _release_digest

    return _release_digest(snapshot.concepts)
