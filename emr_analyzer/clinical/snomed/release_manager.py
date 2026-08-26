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

from .index import SnomedIndex
from .rf2_loader import ReleaseSnapshot, load_snapshot


class _ReleaseManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshots: dict[tuple[str, tuple[str, ...]], ReleaseSnapshot] = {}
        self._indexes: dict[tuple[str, tuple[str, ...]], SnomedIndex] = {}

    def snapshot(
        self,
        release_dir: str,
        languages: Iterable[str] = ("en", "it"),
    ) -> ReleaseSnapshot:
        key = (str(release_dir), tuple(languages))
        with self._lock:
            cached = self._snapshots.get(key)
            if cached is not None:
                # Recompute the digest to detect an on-disk change without
                # re-parsing; a changed digest forces a reload.
                if cached.digest == _snapshot_digest_from_disk(cached):
                    return cached
            snapshot = load_snapshot(release_dir, languages=languages)
            self._snapshots[key] = snapshot
            self._indexes.pop(key, None)
            return snapshot

    def index(
        self,
        release_dir: str,
        languages: Iterable[str] = ("en", "it"),
    ) -> SnomedIndex:
        key = (str(release_dir), tuple(languages))
        snapshot = self.snapshot(release_dir, languages)
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
