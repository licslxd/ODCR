from __future__ import annotations

from configs.base.hardware import preprocess_gpu_a100_2gpu
from configs.base.preprocess_common import (
    all_preprocess_datasets,
    default_preprocess_paths,
    default_runtime_options,
)
from odcr_core.preprocess_schema import PreprocessCConfig


def build_preprocess_c_stage(*, preset_name: str, description: str) -> PreprocessCConfig:
    hardware = preprocess_gpu_a100_2gpu()
    return PreprocessCConfig(
        preset_name=preset_name,
        description=description,
        datasets=all_preprocess_datasets(),
        paths=default_preprocess_paths("preprocess_c"),
        runtime=default_runtime_options(workers=len(hardware.gpu_ids)),
        hardware=hardware,
        chunk_batch_size=512,
        tokenizer_parallelism_enabled=True,
        tokenizer_threads_per_worker=4,
        tokenizer_total_threads=8,
        prefetch_batches=2,
        pin_memory=True,
        non_blocking_h2d=True,
        async_prefetch_enabled=True,
        scheduling_policy="lpt_by_token_windows",
        cpu_cores_reserved=2,
        cpu_cores_available=12,
        bf16_enabled=True,
        tf32_enabled=True,
        tokenizer_hotpath_enabled=True,
        token_window_cache_enabled=True,
        token_window_cache_dir="cache/preprocess_c",
        token_window_cache_version="preprocess_c_token_windows_v3",
        token_window_cache_shard_size=4096,
    )
