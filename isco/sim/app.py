"""SimulationApp lifecycle (Isaac Sim 5.1).

``SimulationApp`` must exist before any ``isaacsim.*`` / ``omni.*`` / ``pxr`` module is
imported, and only one may exist per process. Every other ``isco.sim`` module imports
those packages lazily (inside functions), so ``import isco.sim`` works without Isaac.

``create_new_stage=False`` (as in the PoC): with the default ``True`` the GUI app opens
an empty stage asynchronously right after start-up, which can invalidate the scene we
build (Isaac Lab's AppLauncher disables it for the same reason); ``World()`` creates
the stage instead.
"""
from __future__ import annotations

import os
import time
from typing import Any

_APP: Any = None
_INFO: dict = {}


def launch(headless: bool = True, kit_args: list[str] | None = None, **cfg: Any):
    """Create the process-wide ``SimulationApp`` once and return it (idempotent).

    ``SimulationApp`` forwards ``sys.argv[1:]`` verbatim to Kit, which aborts on foreign
    flags (e.g. pytest's ``-m isaac``: "Ill formed parameter" -> segfault) and exits on
    ``-h``. Kit therefore only sees ``kit_args`` (default: the ``--/carb/setting=value``
    overrides found in ``sys.argv``); ``sys.argv`` is restored afterwards."""
    global _APP
    if _APP is not None:
        return _APP
    import sys

    from isaacsim import SimulationApp  # noqa: WPS433 - must be the first Isaac import

    config = {"headless": bool(headless), "create_new_stage": False}
    config.update(cfg)
    if kit_args is None:
        kit_args = [a for a in sys.argv[1:] if a.startswith("--/")]
    saved_argv = list(sys.argv)
    sys.argv = [saved_argv[0] if saved_argv else "isco"] + list(kit_args)
    t0 = time.perf_counter()
    try:
        _APP = SimulationApp(config)
    finally:
        sys.argv = saved_argv
    _INFO.update(startup_s=round(time.perf_counter() - t0, 3), headless=bool(headless), pid=os.getpid(),
                 isaac_version=isaac_version())
    return _APP


def get_app():
    """The running ``SimulationApp`` or ``None``."""
    return _APP


def is_launched() -> bool:
    return _APP is not None


def info() -> dict:
    """Start-up information (``startup_s``, ``isaac_version``, ``pid``, ``headless``)."""
    return dict(_INFO)


def isaac_version() -> str:
    try:
        from importlib.metadata import version

        return version("isaacsim")
    except Exception:  # pragma: no cover
        return "unknown"


def close(skip_cleanup: bool = False) -> None:
    """Close the app (call at the very end of a script; the process cannot relaunch)."""
    global _APP
    if _APP is None:
        return
    app, _APP = _APP, None
    app.close(skip_cleanup=skip_cleanup)
