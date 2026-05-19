from __future__ import annotations

from configs.preprocess.preprocess_b import build_preprocess_b_stage

PRESET_NAME = "preprocess_b_a100_2gpu"
PRESET_DESCRIPTION = (
    "Internal A100 2-GPU preprocess_b config fixture. Keeps legacy compute_embeddings numeric semantics; "
    "default operational path is verify-only reuse before any full recompute, with grouped-text shard cache and "
    "preprocess_b-local BF16/TF32 steady-state enabled."
)


def build_experiment():
    return build_preprocess_b_stage(
        preset_name=PRESET_NAME,
        description=PRESET_DESCRIPTION,
    )
