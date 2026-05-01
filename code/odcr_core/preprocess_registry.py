from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass
from typing import Any

from odcr_core.preprocess_schema import PreprocessConfig, validate_preprocess_config


@dataclass(frozen=True)
class PreprocessPresetRegistration:
    name: str
    stage: str
    description: str
    entry_module: str
    entry_function: str = "build_experiment"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_PREPROCESS_PRESET_REGISTRY: dict[str, PreprocessPresetRegistration] = {
    "preprocess_a_default": PreprocessPresetRegistration(
        name="preprocess_a_default",
        stage="preprocess_a",
        description="Canonical CPU preprocess preset for materializing the preprocess CSV asset contract.",
        entry_module="configs.experiments.preprocess_a_default",
    ),
    "preprocess_b_a100_2gpu": PreprocessPresetRegistration(
        name="preprocess_b_a100_2gpu",
        stage="preprocess_b",
        description=(
            "Canonical A100 2-GPU preprocess_b preset. Builds dual-channel user/item profiles from canonical assets."
        ),
        entry_module="configs.experiments.preprocess_b_a100_2gpu",
    ),
    "preprocess_c_a100_2gpu": PreprocessPresetRegistration(
        name="preprocess_c_a100_2gpu",
        stage="preprocess_c",
        description=(
            "Canonical A100 2-GPU preprocess_c preset. Builds dual-channel domain semantics from canonical assets."
        ),
        entry_module="configs.experiments.preprocess_c_a100_2gpu",
    ),
}


def list_preprocess_presets() -> list[PreprocessPresetRegistration]:
    return [_PREPROCESS_PRESET_REGISTRY[name] for name in sorted(_PREPROCESS_PRESET_REGISTRY)]


def get_preprocess_preset_registration(name: str) -> PreprocessPresetRegistration:
    try:
        return _PREPROCESS_PRESET_REGISTRY[name]
    except KeyError as exc:
        known = ", ".join(sorted(_PREPROCESS_PRESET_REGISTRY))
        raise KeyError(f"Unknown preprocess preset {name!r}. Known presets: {known}") from exc


def instantiate_preprocess_preset(name: str) -> PreprocessConfig:
    reg = get_preprocess_preset_registration(name)
    module = importlib.import_module(reg.entry_module)
    try:
        build_fn = getattr(module, reg.entry_function)
    except AttributeError as exc:
        raise AttributeError(
            f"Preprocess preset module {reg.entry_module!r} is missing {reg.entry_function!r}"
        ) from exc
    config = build_fn()
    return validate_preprocess_config(config)


def render_preprocess_preset(name: str) -> dict[str, Any]:
    reg = get_preprocess_preset_registration(name)
    config = instantiate_preprocess_preset(name)
    return {
        "name": reg.name,
        "stage": reg.stage,
        "description": reg.description,
        "entry_module": reg.entry_module,
        "config": config.to_dict(),
    }
