"""Internal preprocess config fixtures for tests and schema checks.

These registrations are not a user-visible entrypoint. Operators run the
current preprocess stages through ./odcr preprocess a, ./odcr preprocess b, or
./odcr preprocess c; this module only materializes internal config examples.
"""

from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass
from typing import Any

from odcr_core.preprocess_schema import PreprocessConfig, validate_preprocess_config


@dataclass(frozen=True)
class InternalPreprocessConfigRegistration:
    name: str
    stage: str
    description: str
    entry_module: str
    entry_function: str = "build_experiment"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_INTERNAL_PREPROCESS_CONFIG_REGISTRY: dict[str, InternalPreprocessConfigRegistration] = {
    "preprocess_a_default": InternalPreprocessConfigRegistration(
        name="preprocess_a_default",
        stage="preprocess_a",
        description="Internal CPU preprocess config fixture for materializing the preprocess CSV asset contract.",
        entry_module="configs.experiments.preprocess_a_default",
    ),
    "preprocess_b_a100_2gpu": InternalPreprocessConfigRegistration(
        name="preprocess_b_a100_2gpu",
        stage="preprocess_b",
        description=(
            "Internal A100 2-GPU preprocess_b config fixture. Builds dual-channel user/item profiles from canonical assets."
        ),
        entry_module="configs.experiments.preprocess_b_a100_2gpu",
    ),
    "preprocess_c_a100_2gpu": InternalPreprocessConfigRegistration(
        name="preprocess_c_a100_2gpu",
        stage="preprocess_c",
        description=(
            "Internal A100 2-GPU preprocess_c config fixture. Builds dual-channel domain semantics from canonical assets."
        ),
        entry_module="configs.experiments.preprocess_c_a100_2gpu",
    ),
}


def list_internal_preprocess_configs() -> list[InternalPreprocessConfigRegistration]:
    return [_INTERNAL_PREPROCESS_CONFIG_REGISTRY[name] for name in sorted(_INTERNAL_PREPROCESS_CONFIG_REGISTRY)]


def get_internal_preprocess_config_registration(name: str) -> InternalPreprocessConfigRegistration:
    try:
        return _INTERNAL_PREPROCESS_CONFIG_REGISTRY[name]
    except KeyError as exc:
        known = ", ".join(sorted(_INTERNAL_PREPROCESS_CONFIG_REGISTRY))
        raise KeyError(f"Unknown internal preprocess config {name!r}. Known configs: {known}") from exc


def instantiate_internal_preprocess_config(name: str) -> PreprocessConfig:
    reg = get_internal_preprocess_config_registration(name)
    module = importlib.import_module(reg.entry_module)
    try:
        build_fn = getattr(module, reg.entry_function)
    except AttributeError as exc:
        raise AttributeError(
            f"Internal preprocess config module {reg.entry_module!r} is missing {reg.entry_function!r}"
        ) from exc
    config = build_fn()
    return validate_preprocess_config(config)


def render_internal_preprocess_config(name: str) -> dict[str, Any]:
    reg = get_internal_preprocess_config_registration(name)
    config = instantiate_internal_preprocess_config(name)
    return {
        "name": reg.name,
        "stage": reg.stage,
        "description": reg.description,
        "entry_module": reg.entry_module,
        "config": config.to_dict(),
    }
