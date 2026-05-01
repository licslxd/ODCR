from __future__ import annotations

from configs.preprocess.preprocess_a import build_preprocess_a_stage

PRESET_NAME = "preprocess_a_default"
PRESET_DESCRIPTION = "Canonical CPU preprocess preset for preprocess_a."


def build_experiment():
    return build_preprocess_a_stage(
        preset_name=PRESET_NAME,
        description=PRESET_DESCRIPTION,
    )
