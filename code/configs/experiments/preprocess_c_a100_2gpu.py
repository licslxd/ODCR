from __future__ import annotations

from configs.preprocess.preprocess_c import build_preprocess_c_stage

PRESET_NAME = "preprocess_c_a100_2gpu"
PRESET_DESCRIPTION = (
    "Internal A100 2-GPU preprocess_c config fixture. Uses token-aware chunking and attention-mask-aware mean pooling."
)


def build_experiment():
    return build_preprocess_c_stage(
        preset_name=PRESET_NAME,
        description=PRESET_DESCRIPTION,
    )
