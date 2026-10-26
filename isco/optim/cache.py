"""Persistent evaluation cache (one JSON per evaluated key) - makes searches resumable
and lets experiments share evaluations (plan: cache key = config x variant set x sensor x
process x detector x seed)."""
from __future__ import annotations

from pathlib import Path

from ..utils.hashing import content_hash
from ..utils.io import data_root, read_json, write_json


class EvalCache:
    def __init__(self, root: str | Path | None = None):
        self.root = Path(root) if root else data_root() / "evalcache"

    def key(self, **parts) -> str:
        return content_hash(parts, length=16)

    def path(self, part_id: str, key: str) -> Path:
        return self.root / part_id / f"{key}.json"

    def get(self, part_id: str, key: str):
        p = self.path(part_id, key)
        return read_json(p) if p.exists() else None

    def put(self, part_id: str, key: str, value: dict) -> None:
        write_json(self.path(part_id, key), value, indent=None)
