from __future__ import annotations

from configs.base.hardware import preprocess_gpu_a100_2gpu
from configs.base.preprocess_common import (
    all_preprocess_datasets,
    default_preprocess_paths,
    default_runtime_options,
)
from odcr_core.preprocess_schema import PreprocessBConfig


def build_preprocess_b_stage(*, preset_name: str, description: str) -> PreprocessBConfig:
    hardware = preprocess_gpu_a100_2gpu()
    return PreprocessBConfig(
        preset_name=preset_name,
        description=description,
        datasets=all_preprocess_datasets(),
        paths=default_preprocess_paths("preprocess_b"),
        runtime=default_runtime_options(workers=len(hardware.gpu_ids)),
        hardware=hardware,
        embed_batch_size=512,
        read_chunk_rows=100_000,
        group_shard_size=4_096,
        grouped_text_cache_enabled=True,
        grouped_text_cache_dir="cache/preprocess_b",
        grouped_text_cache_version="preprocess_b_grouped_text_cache_v1",
        bf16_enabled=True,
        tf32_enabled=True,
        verify_sample_size=8,
        verify_seed=7,
    )
