from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Tuple


TOP_LEVEL_BLOCKS: tuple[str, ...] = (
    "project",
    "env",
    "hardware",
    "tasks",
    "preprocess",
    "step3",
    "step4",
    "step5",
    "eval",
)

HARDWARE_PROFILE_REQUIRED_KEYS: tuple[str, ...] = (
    "ddp_world_size",
    "num_proc",
    "max_num_proc",
    "reserved_cpu",
    "max_parallel_cpu",
    "dataloader_num_workers_train",
    "dataloader_num_workers_valid",
    "dataloader_num_workers_test",
    "dataloader_prefetch_factor_train",
    "dataloader_prefetch_factor_valid",
    "dataloader_prefetch_factor_test",
    "pin_memory",
    "persistent_workers",
    "non_blocking_h2d",
)
"""Required One-Control child hardware payload fields."""

HARDWARE_PROFILE_THREAD_ENV_KEYS: tuple[str, ...] = (
    "omp_num_threads",
    "mkl_num_threads",
    "tokenizers_parallelism",
)
"""Hardware profile fields transported as process/thread launcher env."""

TRAIN_PRECISION_CHOICES: tuple[str, ...] = ("bf16", "fp16", "fp32")
"""Resolver-owned train precision values transported to children."""

PREPROCESS_CPU_GPU_ONE_CONTROL_KEYS: dict[str, tuple[str, ...]] = {
    "preprocess.b": (
        "tokenizer_parallelism_enabled",
        "tokenizer_threads_per_worker",
        "tokenizer_total_threads",
        "prefetch_batches",
        "pin_memory",
        "non_blocking_h2d",
        "async_prefetch_enabled",
        "token_aware_batching_enabled",
        "max_tokens_per_gpu_batch",
        "cpu_cores_reserved",
        "cpu_cores_available",
    ),
    "preprocess.c": (
        "tokenizer_parallelism_enabled",
        "tokenizer_threads_per_worker",
        "tokenizer_total_threads",
        "prefetch_batches",
        "pin_memory",
        "non_blocking_h2d",
        "async_prefetch_enabled",
        "scheduling_policy",
        "cpu_cores_reserved",
        "cpu_cores_available",
    ),
}
"""One-Control schema anchor for preprocess_b/c CPU tokenizer and GPU transfer controls."""

SAFE_DECODE_PLACEHOLDER: dict[str, Any] = {
    "decode_strategy": "greedy",
    "decode_seed": None,
    "max_explanation_length": 25,
    "label_smoothing": 0.1,
    "repetition_penalty": 1.15,
    "generate_temperature": 0.8,
    "generate_top_p": 0.9,
    "no_repeat_ngram_size": None,
    "min_len": None,
    "domain_fusion_mode": "gate_cross_attn",
}
"""Resolver-owned decode placeholder for stages that do not consume generation."""


class OneControlConfigError(ValueError):
    """Raised when the one-control configuration cannot be resolved safely."""


@dataclass(frozen=True)
class SourceRecord:
    key: str
    value: Any
    source: str


@dataclass(frozen=True)
class ResolvedConfig:
    command: str
    repo_root: Path
    code_dir: Path

    task_id: int
    auxiliary: str
    target: str
    scenario: str
    direction: str
    task_profile_id: str
    task_profile_key: str
    profile_isolation_hash: str

    preset_name: str
    run_name: Optional[str]
    from_run: Optional[str]
    step5_run: Optional[str]
    step4_run: Optional[str]
    step3_checkpoint_dir: Optional[str]

    train_csv: Optional[str]
    model_path: Optional[str]

    learning_rate: float
    coef: float
    explainer_loss_weight: float

    train_batch_size: int
    global_batch_size: int
    per_device_train_batch_size: int
    per_gpu_batch_size: int
    effective_global_batch_size: int
    batch_semantics_version: str
    grad_accum_removed: bool
    epochs: int
    max_epochs: int
    min_epochs: int
    early_stop_patience: int
    validate_every_epochs: int
    max_grad_norm: float
    tokenizer_max_length: int
    evidence_max_length: int
    valid_batch_size: int
    valid_micro_batch_size: int
    num_proc: int
    ddp_world_size: int
    seed: int

    checkpoint_dir: str
    log_dir: str
    iteration_root_dir: str
    iteration_id: str
    manifest_dir: str
    eval_run_dir: Optional[str]

    label_smoothing: float
    repetition_penalty: float
    generate_temperature: float
    generate_top_p: float

    decode_strategy: str
    decode_seed: Optional[int]
    max_explanation_length: int
    train_label_max_length: int
    no_repeat_ngram_size: Optional[int]
    min_len: Optional[int]
    domain_fusion_mode: str

    step3_mode: str
    step5_train_only: bool

    hardware_preset_id: str
    decode_preset_id: str

    num_return_sequences: int
    rerank_method: str
    rerank_top_k: int
    rerank_weight_logprob: float
    rerank_weight_length: float
    rerank_weight_repeat: float
    rerank_weight_dirty: float
    rerank_target_len_ratio: float
    export_examples_mode: str
    export_full_rerank_examples: bool
    rerank_malformed_tail_penalty: float
    rerank_malformed_token_penalty: float

    decode_profile_json: str
    rerank_profile_json: str
    rerank_preset_id: str

    hardware_profile_json: str
    optimizer_config_json: str
    precision_config_json: str
    tokenizer_config_json: str
    evidence_config_json: str
    scheduler_config_json: str
    valid_batch_config_json: str
    scenario_profile_json: str
    task_profile_config_json: str
    backup_profiles_config_json: str
    exploration_profiles_config_json: str
    worker_profiles_config_json: str
    prefetcher_config_json: str
    checkpoint_policy_config_json: str
    quality_gate_config_json: str
    grad_finite_config_json: str
    diagnostic_eval_config_json: str
    cross_rank_structured_gather_config_json: str
    memory_config_json: str
    timing_config_json: str
    performance_candidates_config_json: str
    cache_policy_config_json: str
    objective_drift_config_json: str
    recovery_config_json: str
    phase_loss_schedule_config_json: str
    conflict_aware_config_json: str
    loss_gradient_conflict_probe_config_json: str
    adapter_gating_config_json: str
    paper_candidate_selection_config_json: str
    checkpoint_averaging_config_json: str
    omp_num_threads: int
    mkl_num_threads: int
    tokenizers_parallelism: bool
    thread_env_requested_json: str
    thread_env_effective_json: str
    launcher_env_requested_json: str
    launcher_env_effective_json: str
    training_preset_train_batch_size: int
    global_eval_batch_size: Optional[int]
    eval_per_gpu_batch_size: Optional[int]
    eval_profile_id: str

    consumed_presets_json: str
    config_before_cli_json: str
    matrix_session_id: Optional[str]
    matrix_cell_id: Optional[str]
    invoked_command: str
    resolved_command_kind: str
    cell_command: Optional[str]

    effective_training_payload_json: str
    training_semantic_fingerprint: str
    generation_semantic_fingerprint: str
    runtime_diagnostics_fingerprint: str
    config_field_sources_json: str
    eval_profile_resolution_json: str
    upstream_resolution_json: str = ""
    step4_rcr_config_json: str = "{}"
    step4_runtime_config_json: str = "{}"
    step5_innovation_config_json: str = ""
    ddp_find_unused_parameters: bool = True
    ddp_find_unused_false_preflight: str = "synthetic_one_batch"
    ddp_static_graph: bool = False
    ddp_graph_safety_preflight: bool = True
    step3_loss_semantics_json: str = ""
    data_dir: str = ""
    merged_dir: str = ""
    runs_dir: str = ""
    cache_dir: str = ""
    models_dir: str = ""
    step5_text_model: str = ""
    sentence_embed_model: str = ""
    embed_dim: int = 1024
    offline: bool = True
    local_files_only: bool = True
    checkpoint_policy: str = "best"
    full_bleu_eval_resolved: Optional[dict[str, Any]] = None
    full_bleu_decode_strategy: str = "inherit"
    step3_eval_protocol: str = "minimal_eval"
    step3_eval_split: str = "valid"
    step3_eval_batch_candidates_json: str = "[]"
    step3_eval_protocol_config_json: str = "{}"
    train_mode: str = "full"
    train_precision: str = "bf16"
    allow_tf32: bool = True
    amp_autocast: bool = True
    grad_scaler: bool = False
    pin_memory: bool = True
    persistent_workers: bool = True
    non_blocking_h2d: bool = True
    per_device_eval_batch_size: int = 2
    lora_r: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.05
    lora_target_modules: Tuple[str, ...] = ()
    nlayers: int = 2
    nhead: int = 2
    nhid: int = 2048
    dropout: float = 0.2


def json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, default=str)


def fingerprint(data: Any) -> str:
    raw = json_dumps(data)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def as_plain_dict(cfg: ResolvedConfig) -> dict[str, Any]:
    out = asdict(cfg)
    out["repo_root"] = str(cfg.repo_root)
    out["code_dir"] = str(cfg.code_dir)
    return out
