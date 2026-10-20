"""JSON / YAML helpers with atomic writes."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml


class _Enc(json.JSONEncoder):
    def default(self, o: Any) -> Any:  # noqa: D401
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        if hasattr(o, "model_dump"):
            return o.model_dump(mode="json")
        return super().default(o)


def atomic_write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def write_json(path: str | Path, obj: Any, indent: int | None = 2) -> None:
    atomic_write_text(path, json.dumps(obj, cls=_Enc, indent=indent, ensure_ascii=False))


def read_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def append_jsonl(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj, cls=_Enc, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[Any]:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def read_yaml(path: str | Path) -> Any:
    with open(path) as f:
        return yaml.safe_load(f)


def write_yaml(path: str | Path, obj: Any) -> None:
    atomic_write_text(path, yaml.safe_dump(json.loads(json.dumps(obj, cls=_Enc)), sort_keys=False, allow_unicode=True))


def repo_root() -> Path:
    """The ISCO repository root (directory holding pyproject.toml)."""
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "pyproject.toml").exists():
            return p
    return Path.cwd()


def data_root() -> Path:
    """Root of generated/downloaded data: ``$ISCO_DATA_ROOT`` or ``<repo>/data``."""
    import os

    env = os.environ.get("ISCO_DATA_ROOT")
    return Path(env) if env else repo_root() / "data"


def data_rel(path: str | Path) -> str:
    """Path stored in manifests: relative to ``data_root()``."""
    return str(Path(path).resolve().relative_to(data_root().resolve()))


def resolve_data_path(p: str | Path) -> Path:
    """Inverse of ``data_rel``; also accepts absolute paths and legacy ``data/...``
    paths (relative to the repository root)."""
    p = Path(p)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] == "data" and not (data_root() / p).exists():
        return repo_root() / p
    return data_root() / p
