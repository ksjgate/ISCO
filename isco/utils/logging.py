"""Console + file logging (rich if available)."""
from __future__ import annotations

import logging
from pathlib import Path

_CONFIGURED = False


def get_logger(name: str = "isco") -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        try:
            from rich.logging import RichHandler

            handler: logging.Handler = RichHandler(show_path=False, rich_tracebacks=False, markup=False)
            fmt = "%(message)s"
        except Exception:  # pragma: no cover
            handler = logging.StreamHandler()
            fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        root = logging.getLogger("isco")
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        root.propagate = False
        _CONFIGURED = True
    return logging.getLogger(name)


def add_file_handler(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(path)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger("isco").addHandler(fh)
