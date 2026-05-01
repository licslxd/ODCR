from __future__ import annotations

from configs.base.hardware import preprocess_cpu_serial
from configs.base.preprocess_common import (
    all_preprocess_datasets,
    default_preprocess_paths,
    default_runtime_options,
)
from odcr_core.preprocess_schema import PreprocessAConfig


def build_preprocess_a_stage(*, preset_name: str, description: str) -> PreprocessAConfig:
    _ = preprocess_cpu_serial()
    return PreprocessAConfig(
        preset_name=preset_name,
        description=description,
        datasets=all_preprocess_datasets(),
        paths=default_preprocess_paths("preprocess_a"),
        runtime=default_runtime_options(workers=1),
    )
