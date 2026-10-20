"""Content hashing for configs / cache keys."""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


def _normalise(obj: Any, ndigits: int = 6) -> Any:
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, (np.floating,)):
        return round(float(obj), ndigits)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _normalise(obj.tolist(), ndigits)
    if isinstance(obj, dict):
        return {str(k): _normalise(v, ndigits) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_normalise(v, ndigits) for v in obj]
    if hasattr(obj, "model_dump"):
        return _normalise(obj.model_dump(mode="json"), ndigits)
    return obj


def canonical_json(obj: Any, ndigits: int = 6) -> str:
    return json.dumps(_normalise(obj, ndigits), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def content_hash(obj: Any, length: int = 10, ndigits: int = 6) -> str:
    return hashlib.sha1(canonical_json(obj, ndigits).encode()).hexdigest()[:length]
