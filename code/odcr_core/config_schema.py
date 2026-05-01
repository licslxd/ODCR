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
    adv: float
    eta: float
    explainer_loss_weight: float

    train_batch_size: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    effective_global_batch_size: int
    epochs: int
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
    step4_rcr_config_json: str = "{}"
    step5_innovation_config_json: str = ""
    ddp_find_unused_parameters: bool = True
    ddp_find_unused_false_preflight: str = "synthetic_one_batch"
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
    train_mode: str = "full"
    train_precision: str = "bf16"
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
