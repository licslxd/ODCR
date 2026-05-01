from __future__ import annotations

from odcr_core.preprocess_schema import PreprocessHardwareConfig


def preprocess_cpu_serial() -> PreprocessHardwareConfig:
    return PreprocessHardwareConfig(gpu_ids=())


def preprocess_gpu_a100_2gpu() -> PreprocessHardwareConfig:
    return PreprocessHardwareConfig(gpu_ids=(0, 1))
