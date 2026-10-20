"""Only isco.sim may import Isaac Sim / Omniverse / USD modules (ADR-0002)."""
import subprocess
import sys

CODE = r"""
import builtins, importlib, pkgutil
import isco
real_import = builtins.__import__
def guard(name, *a, **k):
    if name.split('.')[0] in ('isaacsim', 'omni', 'pxr', 'carb'):
        raise ImportError('blocked: ' + name)
    return real_import(name, *a, **k)
builtins.__import__ = guard
mods = [m.name for m in pkgutil.walk_packages(isco.__path__, 'isco.') if not m.name.startswith('isco.sim')]
for m in mods:
    importlib.import_module(m)
import isco.sim  # the package itself must also import lazily
print(len(mods))
"""


def test_non_sim_modules_import_without_isaac():
    r = subprocess.run([sys.executable, "-c", CODE], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    assert int(r.stdout.strip().splitlines()[-1]) > 30
