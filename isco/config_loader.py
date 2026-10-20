"""Load named YAML configs from ``configs/<group>/<name>.yaml`` (or explicit paths)."""
from __future__ import annotations

from pathlib import Path

from .utils.io import read_yaml, repo_root


def config_path(group: str, name_or_path: str) -> Path:
    p = Path(name_or_path)
    if p.suffix in (".yaml", ".yml") and p.exists():
        return p
    return repo_root() / "configs" / group / f"{name_or_path}.yaml"


def load(group: str, name_or_path: str) -> dict:
    return read_yaml(config_path(group, name_or_path)) or {}


def noise(name: str = "N2"):
    from .schemas.sensor import NoiseModel

    return NoiseModel.model_validate(load("noise", name))


def process(name: str = "inline_default"):
    from .schemas.process import ProcessParams

    return ProcessParams.model_validate(load("process", name))


def sensor(name: str = "sl_default"):
    from .schemas.sensor import SensorModel

    return SensorModel.model_validate(load("sensor", name))


def grid(name: str = "default"):
    from .geometry.camera_model import SlotGrid

    return SlotGrid.from_dict(load("grid", name))


def detector(name: str = "fpfh_bank", **overrides):
    from .detect.pipeline import DetectorConfig

    return DetectorConfig.model_validate({**load("detector", name), **overrides})


def dictionary(name: str = "default_dictionary"):
    from .datagen.defects import DefectDictionary

    return DefectDictionary.model_validate(load("defects", name))


def variant_spec(name: str = "mvp", dictionary_name: str = "default_dictionary"):
    from .datagen.variants import VariantSetSpec

    d = load("variants", name)
    d.setdefault("dictionary", load("defects", dictionary_name))
    return VariantSetSpec.model_validate(d)


def search(name: str = "smoke") -> dict:
    return load("search", name)
