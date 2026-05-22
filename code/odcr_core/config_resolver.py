from __future__ import annotations

import json
import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from odcr_core import path_layout, run_naming
from odcr_core.config_schema import (
    HARDWARE_PROFILE_REQUIRED_KEYS,
    OneControlConfigError,
    ResolvedConfig,
    SAFE_DECODE_PLACEHOLDER,
    SourceRecord,
    TOP_LEVEL_BLOCKS,
    TRAIN_PRECISION_CHOICES,
    fingerprint,
    json_dumps,
)
from odcr_core.rating_source import (
    RATING_SOURCE_TYPE,
    resolve_rating_source_config,
    validate_rating_source,
)
from odcr_core.training_diagnostics import runtime_diagnostics_fingerprint_source
from odcr_core.step3_quality import (
    DIAGNOSTIC_PROTOCOLS,
    MEMORY_REQUIRED_FIELDS,
    MEMORY_PHASES,
    PREFETCH_EVIDENCE_FIELDS,
    STEP3_CHECKPOINT_POLICY_VERSION,
    STEP3_DIAGNOSTIC_PROTOCOL_VERSION,
    STEP3_PERFORMANCE_CANDIDATE_SCHEMA_VERSION,
    STEP3_QUALITY_GATE_VERSION,
    TIMING_REQUIRED_FIELDS,
    default_a100_candidate_matrix,
)
from odcr_core.step3_eval_protocol import (
    MINIMAL_EVAL,
    ODCR_STEP3_DIAGNOSTIC,
    PAPER_TARGET_ONLY_EVAL,
    normalize_eval_protocol,
    step3_eval_protocol_spec,
)
from odcr_core.upstream_resolver import UpstreamResolutionError, resolve_latest, resolve_run, resolve_upstream

_CODE_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _CODE_DIR.parent
_STEP5_HEAD_AWARE_LORA_TARGET_SENTINEL = "__HEAD_AWARE_STEP5_DEFAULT__"
_STEP5_LORA_TARGET_POLICY_ID = "step5_explanation_lora_allowlist/1"
_STEP5_DELETED_LEGACY_MODULES = ("recommender", "flan_soft_prompt_stack", "hidden2token")


def _step5_forbidden_lora_targets_from_model_config(nlayers: Any) -> list[str]:
    try:
        layer_count = int(nlayers)
    except (TypeError, ValueError):
        layer_count = 0
    targets = ["domain_cross_attn.out_proj"]
    targets.extend(
        f"transformer_encoder.layers.{idx}.self_attn.out_proj"
        for idx in range(max(layer_count, 0))
    )
    return targets


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise OneControlConfigError("PyYAML is required to read configs/odcr.yaml") from exc
    p = Path(path)
    if not p.is_absolute():
        p = (_REPO_ROOT / p).resolve()
    if not p.is_file():
        raise OneControlConfigError(f"config file not found: {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise OneControlConfigError(f"{p} must contain a mapping at the top level")
    missing = [k for k in TOP_LEVEL_BLOCKS if k not in raw]
    if missing:
        raise OneControlConfigError(f"{p} is missing required top-level blocks: {missing}")
    extra = sorted(set(raw) - set(TOP_LEVEL_BLOCKS))
    if extra:
        raise OneControlConfigError(f"{p} has unsupported top-level blocks: {extra}")
    return raw


def parse_set_value(raw: str) -> Any:
    s = str(raw).strip()
    lower = s.lower()
    if lower in ("true", "false"):
        return lower == "true"
    if lower in ("none", "null", "~"):
        return None
    if s.startswith("[") or s.startswith("{"):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
    try:
        if "." not in s and "e" not in lower:
            return int(s)
        return float(s)
    except ValueError:
        return raw


def apply_cli_sets(cfg: dict[str, Any], raw_sets: Iterable[str]) -> tuple[dict[str, Any], dict[str, str]]:
    out = deepcopy(cfg)
    sources: dict[str, str] = {}
    for item in raw_sets:
        if "=" not in item:
            raise OneControlConfigError(f"--set expects key=value, got {item!r}")
        key, value_raw = item.split("=", 1)
        key = key.strip()
        if not key:
            raise OneControlConfigError("--set key cannot be empty")
        _reject_retired_accum_name(key)
        parts = key.split(".")
        cur: Any = out
        for part in parts[:-1]:
            if not isinstance(cur, dict):
                raise OneControlConfigError(f"--set cannot descend through non-object key {part!r} in {key!r}")
            if part not in cur or cur[part] is None:
                cur[part] = {}
            cur = cur[part]
        if not isinstance(cur, dict):
            raise OneControlConfigError(f"--set target parent is not an object for {key!r}")
        cur[parts[-1]] = parse_set_value(value_raw)
        sources[key] = "cli --set"
    return out, sources


def _get(obj: Mapping[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = obj
    for part in path.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _merge_dicts(*items: Mapping[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items:
        if not item:
            continue
        for key, value in item.items():
            if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
                out[key] = _merge_dicts(out[key], value)
            else:
                out[key] = deepcopy(value)
    return out


NO_ACCUM_BATCH_SEMANTICS_VERSION = "odcr_no_accum/1"
NO_ACCUM_REMOVED_MESSAGE = (
    "grad_accum has been removed in ODCR no-accum architecture; use per_gpu_batch_size "
    "and global_batch_size = per_gpu_batch_size * ddp_world_size."
)
_RETIRED_ACCUM_FIELDS = frozenset(
    {
        "grad_accum",
        "gradient_accumulation_steps",
        "accumulate_grad_batches",
        "accum_steps",
        "accumulation_steps",
    }
)
_RETIRED_ACCUM_ENV = frozenset(
    {
        "ODCR_GRAD_ACCUM",
        "ODCR_GRADIENT_ACCUMULATION_STEPS",
        "ODCR_ACCUMULATE_GRAD_BATCHES",
        "ODCR_ACCUM_STEPS",
        "ODCR_ACCUMULATION_STEPS",
    }
)

_STEP5_RETIRED_TRAIN_FIELDS = frozenset({"adv", "eta", "train_label_max_length"})
_STEP5_FORMAL_RATIO_ID = "STEP5_RATIO_0"
_STEP5_FORMAL_CF_MIX_ID = "STEP5_CF_MIX_FORMAL_HIGH_MEDIUM"
_STEP5_HISTORICAL_CF_MIX_SEMANTICS: dict[tuple[str, str], dict[str, float]] = {}


def _reject_retired_accum_name(name: str) -> None:
    if str(name).split(".")[-1] in _RETIRED_ACCUM_FIELDS:
        raise OneControlConfigError(NO_ACCUM_REMOVED_MESSAGE)


def _reject_retired_accum_keys(mapping: Mapping[str, Any], context: str) -> None:
    retired = sorted(str(key) for key in mapping if str(key) in _RETIRED_ACCUM_FIELDS)
    if retired:
        raise OneControlConfigError(f"{context} contains retired accumulation key(s) {retired}. {NO_ACCUM_REMOVED_MESSAGE}")


def _reject_retired_accum_env() -> None:
    present = sorted(name for name in _RETIRED_ACCUM_ENV if (os.environ.get(name) or "").strip())
    if present:
        raise OneControlConfigError(f"Retired accumulation environment variable(s) set: {present}. {NO_ACCUM_REMOVED_MESSAGE}")


def _reject_unknown_keys(mapping: Mapping[str, Any], allowed: set[str], context: str) -> None:
    _reject_retired_accum_keys(mapping, context)
    extra = sorted(str(key) for key in mapping if str(key) not in allowed)
    if extra:
        raise OneControlConfigError(f"{context} has unsupported key(s) under strict One-Control schema: {extra}")


def _validate_step3_profile_shape(item: Mapping[str, Any], context: str) -> None:
    _reject_unknown_keys(
        item,
        {
            "task_id",
            "source",
            "target",
            "scenario",
            "direction",
            "active_profile",
            "profile_id",
            "candidate",
            "formal_allowed",
            "profile_ready",
            "formal_ready",
            "probe_only",
            "note",
            "train",
            "tokenizer",
            "evidence",
            "scheduler",
            "cross_rank_structured_gather",
            "effective_pool_expected",
            "memory",
            "optimizer",
            "precision",
        },
        context,
    )
    train = item.get("train")
    if isinstance(train, Mapping):
        _reject_unknown_keys(train, {"batch_size", "per_gpu_batch_size", "lr", "max_grad_norm", "backend"}, f"{context}.train")
        backend = train.get("backend")
        if isinstance(backend, Mapping):
            _reject_unknown_keys(
                backend,
                {
                    "train_precision",
                    "allow_tf32",
                    "amp_autocast",
                    "grad_scaler",
                },
                f"{context}.train.backend",
            )
    for block, keys in (
        ("tokenizer", {"max_length"}),
        ("evidence", {"max_evidence_length"}),
        ("scheduler", {"name", "warmup_ratio", "min_lr_ratio", "validation_aware_lr_damping"}),
        ("cross_rank_structured_gather", {"enabled", "mode"}),
        ("memory", {"activation_checkpointing", "profile_buffer_policy"}),
    ):
        value = item.get(block)
        if isinstance(value, Mapping):
            _reject_unknown_keys(value, keys, f"{context}.{block}")


def _validate_step3_named_runtime_profiles(raw: Mapping[str, Any], context: str, *, kind: str) -> None:
    if kind == "backup":
        allowed = {
            "base_task_profile",
            "task_profile_id",
            "candidate",
            "batch_size",
            "per_gpu_batch_size",
            "ddp_world_size",
            "cross_rank_structured_gather",
            "gather_mode",
            "effective_pool_expected",
            "activation_checkpointing",
            "profile_buffer_policy",
            "backup_only",
            "manual_selection_required",
            "not_default",
            "formal_allowed",
            "probe_only",
        }
    else:
        allowed = {
            "base_task_profile",
            "task_profile_id",
            "candidate",
            "batch_size",
            "per_gpu_batch_size",
            "ddp_world_size",
            "cross_rank_structured_gather",
            "gather_mode",
            "effective_pool_expected",
            "activation_checkpointing",
            "profile_buffer_policy",
            "exploration_only",
            "formal_allowed",
            "probe_only",
            "replacement_gate_status",
            "promotion_policy",
        }
    for name, item in raw.items():
        if isinstance(item, Mapping):
            _reject_unknown_keys(item, allowed, f"{context}.{name}")


def _validate_step3_config_shape(cfg: Mapping[str, Any]) -> None:
    raw = _mapping(_get(cfg, "step3"), "step3")
    _reject_unknown_keys(
        raw,
        {
            "ddp",
            "loss_semantics",
            "structured_losses",
            "scenario_profiles",
            "task_profiles",
            "backup_profiles",
            "exploration_profiles",
            "optimizer",
            "tokenizer",
            "evidence",
            "scheduler",
            "eval",
            "cache",
            "prefetcher",
            "checkpoint_policy",
            "quality_gate",
            "grad_finite",
            "diagnostic_eval",
            "cross_rank_structured_gather",
            "memory",
            "timing",
            "performance_candidates",
            "worker_profiles",
            "objective_drift",
            "recovery",
            "phase_loss_schedule",
            "conflict_aware",
            "loss_gradient_conflict_probe",
            "adapter_gating",
            "paper_candidate_selection",
            "checkpoint_averaging",
            "train",
        },
        "step3",
    )
    for name, item in _mapping(raw.get("scenario_profiles"), "step3.scenario_profiles").items():
        if isinstance(item, Mapping):
            _reject_unknown_keys(item, {"tokenizer", "evidence", "train"}, f"step3.scenario_profiles.{name}")
            for block, keys in (
                ("tokenizer", {"max_length"}),
                ("evidence", {"max_evidence_length"}),
                ("train", {"lr"}),
            ):
                value = item.get(block)
                if isinstance(value, Mapping):
                    _reject_unknown_keys(value, keys, f"step3.scenario_profiles.{name}.{block}")
    for name, item in _mapping(raw.get("task_profiles"), "step3.task_profiles").items():
        if isinstance(item, Mapping):
            _validate_step3_profile_shape(item, f"step3.task_profiles.{name}")
    _validate_step3_named_runtime_profiles(
        _mapping(raw.get("backup_profiles"), "step3.backup_profiles"),
        "step3.backup_profiles",
        kind="backup",
    )
    _validate_step3_named_runtime_profiles(
        _mapping(raw.get("exploration_profiles"), "step3.exploration_profiles"),
        "step3.exploration_profiles",
        kind="exploration",
    )
    train = _mapping(raw.get("train"), "step3.train")
    _reject_unknown_keys(
        train,
        {
            "batch_size",
            "per_gpu_batch_size",
            "max_epochs",
            "min_epochs",
            "early_stop_patience",
            "validate_every_epochs",
            "max_grad_norm",
            "train_label_max_length",
            "checkpoint_metric",
            "full_bleu_eval",
            "backend",
        },
        "step3.train",
    )
    backend = _mapping(train.get("backend"), "step3.train.backend")
    _reject_unknown_keys(
        backend,
        {
            "train_precision",
            "allow_tf32",
            "amp_autocast",
            "grad_scaler",
            "train_dynamic_padding",
            "ema_enabled",
            "ema_decay",
            "generate_during_train",
            "full_bleu_decode_strategy",
        },
        "step3.train.backend",
    )
    cache = _mapping(raw.get("cache"), "step3.cache")
    _reject_unknown_keys(cache, {"tokenizer_schema_version", "formal_cache_namespace"}, "step3.cache")


def _validate_config_shape(cfg: Mapping[str, Any]) -> None:
    tasks = _mapping(_get(cfg, "tasks"), "tasks")
    for task_id, raw in tasks.items():
        if isinstance(raw, Mapping):
            _reject_unknown_keys(raw, {"source", "target", "scenario", "direction", "auxiliary", "lr"}, f"tasks.{task_id}")
    _validate_step3_config_shape(cfg)


def _reject_step5_retired_controls(train: Mapping[str, Any]) -> None:
    bad = sorted(k for k in _STEP5_RETIRED_TRAIN_FIELDS if k in train)
    if bad:
        raise OneControlConfigError(
            "step5.train contains retired ambiguous controls "
            f"{bad}; use step5.train.explainer_loss_weight and step5.train.label_max_length for active Step5 controls."
        )


def _task_row(cfg: Mapping[str, Any], task_id: int) -> dict[str, Any]:
    tasks = _get(cfg, "tasks", {})
    raw = None
    if isinstance(tasks, Mapping):
        raw = tasks.get(str(task_id), tasks.get(task_id))
    if not isinstance(raw, Mapping):
        raise OneControlConfigError(f"tasks.{task_id} must be configured in configs/odcr.yaml")
    source = raw.get("source", raw.get("auxiliary"))
    target = raw.get("target")
    if not source or not target:
        raise OneControlConfigError(f"tasks.{task_id} must define source and target")
    return dict(raw)


def _stage_task_override(stage_cfg: Mapping[str, Any], task_id: int) -> dict[str, Any]:
    raw = stage_cfg.get("tasks", {})
    if not isinstance(raw, Mapping):
        return {}
    item = raw.get(str(task_id), raw.get(task_id, {}))
    return dict(item) if isinstance(item, Mapping) else {}


def _set_nested(obj: dict[str, Any], dotted: str, value: Any) -> None:
    cur = obj
    parts = dotted.split(".")
    for part in parts[:-1]:
        if part not in cur or not isinstance(cur[part], dict):
            cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = deepcopy(value)


def _apply_train_cli_overrides(
    *,
    cfg: Mapping[str, Any],
    cli_sources: Mapping[str, str],
    stage: str,
    train: dict[str, Any],
) -> None:
    prefix = f"{stage}.train."
    for key in cli_sources:
        if key.startswith(prefix):
            _set_nested(train, key[len(prefix) :], _get(cfg, key))


def _nonnegative_int(value: Any, key: str) -> int:
    try:
        out = int(value)
    except Exception as exc:
        raise OneControlConfigError(f"{key} must be an integer, got {value!r}") from exc
    if out < 0:
        raise OneControlConfigError(f"{key} must be >= 0, got {out}")
    return out


def _validate_active_hardware_profile(stem: str, profile: Mapping[str, Any]) -> dict[str, Any]:
    missing = [key for key in HARDWARE_PROFILE_REQUIRED_KEYS if key not in profile]
    if missing:
        raise OneControlConfigError(
            f"hardware.profiles.{stem} missing required child payload fields: {missing}"
        )

    normalized = dict(profile)
    ddp_world_size = _positive_int(
        normalized.get("ddp_world_size"),
        f"hardware.profiles.{stem}.ddp_world_size",
    )
    max_parallel_cpu = _positive_int(
        normalized.get("max_parallel_cpu"),
        f"hardware.profiles.{stem}.max_parallel_cpu",
    )
    max_num_proc = _positive_int(normalized.get("max_num_proc"), f"hardware.profiles.{stem}.max_num_proc")
    reserved_cpu = _positive_int(normalized.get("reserved_cpu"), f"hardware.profiles.{stem}.reserved_cpu")
    configured_num_proc = normalized.get("num_proc")
    if isinstance(configured_num_proc, str) and configured_num_proc.strip().lower() == "auto":
        available = max_parallel_cpu - reserved_cpu
        if available < 1:
            raise OneControlConfigError(
                f"hardware.profiles.{stem}.reserved_cpu={reserved_cpu} leaves no CPU for cold Step3 tokenization "
                f"under max_parallel_cpu={max_parallel_cpu}."
            )
        num_proc = min(max_num_proc, available)
        num_proc_source = "auto"
        num_proc_formula = (
            f"min(max_num_proc({max_num_proc}), max_parallel_cpu({max_parallel_cpu}) - "
            f"reserved_cpu({reserved_cpu}))"
        )
    else:
        num_proc = _positive_int(configured_num_proc, f"hardware.profiles.{stem}.num_proc")
        if num_proc > max_num_proc:
            raise OneControlConfigError(
                f"hardware.profiles.{stem}.num_proc={num_proc} exceeds max_num_proc={max_num_proc}."
            )
        num_proc_source = "fixed"
        num_proc_formula = f"fixed num_proc({num_proc})"
    normalized["ddp_world_size"] = ddp_world_size
    normalized["num_proc"] = num_proc
    normalized["num_proc_configured"] = configured_num_proc
    normalized["tokenization_num_proc"] = num_proc
    normalized["tokenization_num_proc_source"] = num_proc_source
    normalized["tokenization_num_proc_formula"] = num_proc_formula
    normalized["max_num_proc"] = max_num_proc
    normalized["reserved_cpu"] = reserved_cpu
    normalized["max_parallel_cpu"] = max_parallel_cpu

    if num_proc > max_parallel_cpu:
        raise OneControlConfigError(
            f"hardware.profiles.{stem}.num_proc={num_proc} exceeds "
            f"max_parallel_cpu={max_parallel_cpu}; keep dataset map processes within the CPU budget."
        )
    tokenization_active = num_proc + reserved_cpu
    if tokenization_active > max_parallel_cpu:
        raise OneControlConfigError(
            f"hardware.profiles.{stem}.num_proc={num_proc} with "
            f"and reserved_cpu={reserved_cpu} exceeds max_parallel_cpu={max_parallel_cpu}; "
            "Step3 tokenizer cache construction uses pre-DDP datasets.map workers and must stay within the CPU budget."
        )

    for split in ("train", "valid", "test"):
        key = f"dataloader_num_workers_{split}"
        workers = _nonnegative_int(normalized.get(key), f"hardware.profiles.{stem}.{key}")
        normalized[key] = workers
        active_workers = workers * ddp_world_size
        active_with_ranks = active_workers + reserved_cpu
        if active_with_ranks > max_parallel_cpu:
            raise OneControlConfigError(
                f"hardware.profiles.{stem}.{key}={workers} with "
                f"ddp_world_size={ddp_world_size} plus reserved_cpu={reserved_cpu} exceeds "
                f"max_parallel_cpu={max_parallel_cpu}; Step3 dataloader workers are per rank."
            )
        pf_key = f"dataloader_prefetch_factor_{split}"
        prefetch = _positive_int(normalized.get(pf_key), f"hardware.profiles.{stem}.{pf_key}")
        normalized[pf_key] = prefetch

    normalized["worker_budget_formula"] = {
        "semantics": "Step3 dataloader_num_workers_* are per rank; num_proc is pre-DDP datasets.map/tokenizer process count.",
        "reserved_cpu": reserved_cpu,
        "train_active_processes": normalized["dataloader_num_workers_train"] * ddp_world_size + reserved_cpu,
        "valid_active_processes": normalized["dataloader_num_workers_valid"] * ddp_world_size + reserved_cpu,
        "test_active_processes": normalized["dataloader_num_workers_test"] * ddp_world_size + reserved_cpu,
        "tokenization_active_processes": num_proc + reserved_cpu,
        "tokenization_num_proc": num_proc,
        "tokenization_num_proc_source": num_proc_source,
        "tokenization_num_proc_formula": num_proc_formula,
        "max_parallel_cpu": max_parallel_cpu,
    }

    for key in ("omp_num_threads", "mkl_num_threads"):
        if key in normalized:
            normalized[key] = _positive_int(normalized[key], f"hardware.profiles.{stem}.{key}")

    for key in ("pin_memory", "persistent_workers", "non_blocking_h2d"):
        if key not in normalized:
            raise OneControlConfigError(f"hardware.profiles.{stem}.{key} must be configured in configs/odcr.yaml")
        normalized[key] = _bool(normalized[key])

    return normalized


def _active_hardware(cfg: Mapping[str, Any], stage_profile: str | None = None) -> tuple[str, dict[str, Any]]:
    hardware = _get(cfg, "hardware", {})
    if not isinstance(hardware, Mapping):
        raise OneControlConfigError("hardware must be a mapping")
    stem = str(stage_profile or hardware.get("active") or "default")
    profiles = hardware.get("profiles")
    if not isinstance(profiles, Mapping) or stem not in profiles:
        raise OneControlConfigError(f"hardware profile {stem!r} not found under hardware.profiles")
    profile = profiles[stem]
    if not isinstance(profile, Mapping):
        raise OneControlConfigError(f"hardware.profiles.{stem} must be a mapping")
    return stem, _validate_active_hardware_profile(stem, profile)


def _positive_int(value: Any, key: str) -> int:
    try:
        out = int(value)
    except Exception as exc:
        raise OneControlConfigError(f"{key} must be an integer, got {value!r}") from exc
    if out < 1:
        raise OneControlConfigError(f"{key} must be >= 1, got {out}")
    return out


def _positive_float(value: Any, key: str) -> float:
    try:
        out = float(value)
    except Exception as exc:
        raise OneControlConfigError(f"{key} must be a number, got {value!r}") from exc
    return out


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in ("", "none", "null"):
        return None
    return int(value)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _resolve_repo_path(value: Any, key: str, repo_root: Path) -> str:
    raw = str(value or "").strip()
    if not raw:
        raise OneControlConfigError(f"{key} must be configured in configs/odcr.yaml")
    p = Path(os.path.expanduser(raw))
    if not p.is_absolute():
        p = repo_root / p
    return str(p.resolve())


def _resolve_global_runtime_roots(cfg: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    project = _mapping(_get(cfg, "project"), "project")
    env = _mapping(_get(cfg, "env"), "env")
    roots = {
        "runs_dir": _resolve_repo_path(project.get("run_root"), "project.run_root", repo_root),
        "cache_dir": _resolve_repo_path(project.get("cache_dir"), "project.cache_dir", repo_root),
        "data_dir": _resolve_repo_path(project.get("data_dir"), "project.data_dir", repo_root),
        "merged_dir": _resolve_repo_path(project.get("merged_dir"), "project.merged_dir", repo_root),
        "models_dir": _resolve_repo_path(env.get("models_dir"), "env.models_dir", repo_root),
        "step5_text_model": _resolve_repo_path(env.get("step5_text_model"), "env.step5_text_model", repo_root),
        "sentence_embed_model": _resolve_repo_path(env.get("sentence_embed_model"), "env.sentence_embed_model", repo_root),
        "embed_dim": _positive_int(env.get("embed_dim"), "env.embed_dim"),
        "offline": _bool(env.get("offline", True)),
        "local_files_only": _bool(env.get("local_files_only", True)),
    }
    expected_by_env = {
        "ODCR_DATA_DIR": ("data_dir", roots["data_dir"]),
        "ODCR_MERGED_DATA_DIR": ("merged_dir", roots["merged_dir"]),
        "ODCR_MODELS_DIR": ("models_dir", roots["models_dir"]),
        "ODCR_STEP5_TEXT_MODEL": ("step5_text_model", roots["step5_text_model"]),
        "ODCR_SENTENCE_EMBED_MODEL": ("sentence_embed_model", roots["sentence_embed_model"]),
    }
    conflicts: list[str] = []
    for env_name, (root_key, expected) in expected_by_env.items():
        raw = (os.environ.get(env_name) or "").strip()
        if not raw:
            continue
        observed = str(Path(os.path.expanduser(raw)).resolve())
        if observed != str(expected):
            conflicts.append(f"{env_name}={observed} conflicts with configs/odcr.yaml {root_key}={expected}")
    raw_embed = (os.environ.get("ODCR_EMBED_DIM") or "").strip()
    if raw_embed:
        try:
            observed_embed = int(raw_embed)
        except ValueError as exc:
            raise OneControlConfigError("ODCR_EMBED_DIM must be an integer when present") from exc
        if observed_embed != int(roots["embed_dim"]):
            conflicts.append(
                f"ODCR_EMBED_DIM={observed_embed} conflicts with configs/odcr.yaml env.embed_dim={roots['embed_dim']}"
            )
    if conflicts:
        raise OneControlConfigError(
            "Legacy ODCR_* root/model/embed_dim environment variables cannot override One-Control config: "
            + "; ".join(conflicts)
        )
    return roots


_STEP4_RCR_REQUIRED_EXPORT_FIELDS = (
    "train_keep",
    "sample_weight_hint",
    "route_scorer",
    "route_explainer",
    "route_reason_scorer",
    "route_reason_explainer",
    "content_retention_score",
    "style_shift_score",
    "rating_stability_score",
    "cf_reliability_score",
    "uncertainty_score",
    "entropy_score",
    "text_quality_score",
    "confidence_bucket",
    "preprocess_route_scorer_prior",
    "preprocess_route_explainer_prior",
)


def _mapping(value: Any, key: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OneControlConfigError(f"{key} must be a mapping")
    return value


def _rcr_float(value: Any, key: str, *, min_value: float | None = None, max_value: float | None = None) -> float:
    out = _positive_float(value, key)
    if min_value is not None and out < min_value:
        raise OneControlConfigError(f"{key} must be >= {min_value}, got {out}")
    if max_value is not None and out > max_value:
        raise OneControlConfigError(f"{key} must be <= {max_value}, got {out}")
    return float(out)


def _rcr_int(value: Any, key: str, *, min_value: int = 0) -> int:
    out = int(_positive_int(value, key))
    if out < min_value:
        raise OneControlConfigError(f"{key} must be >= {min_value}, got {out}")
    return out


def _resolve_rcr_weights(raw: Mapping[str, Any], key: str, names: tuple[str, ...]) -> dict[str, float]:
    obj = _mapping(raw.get(key), f"step4.rcr.{key}")
    missing = [name for name in names if name not in obj]
    extra = sorted(set(obj) - set(names))
    if missing or extra:
        raise OneControlConfigError(f"step4.rcr.{key} mismatch; missing={missing}, extra={extra}")
    out = {name: _rcr_float(obj[name], f"step4.rcr.{key}.{name}", min_value=0.0, max_value=1.0) for name in names}
    total = sum(out.values())
    if abs(total - 1.0) > 1e-6:
        raise OneControlConfigError(f"step4.rcr.{key} weights must sum to 1.0, got {total}")
    return out


def _resolve_step4_rcr_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    raw = _mapping(_get(cfg, "step4.rcr"), "step4.rcr")
    scorer = _mapping(raw.get("route_scorer"), "step4.rcr.route_scorer")
    explainer = _mapping(raw.get("route_explainer"), "step4.rcr.route_explainer")
    bucket = _mapping(raw.get("confidence_bucket"), "step4.rcr.confidence_bucket")
    high = _mapping(bucket.get("high"), "step4.rcr.confidence_bucket.high")
    medium = _mapping(bucket.get("medium"), "step4.rcr.confidence_bucket.medium")
    train_keep = _mapping(raw.get("train_keep"), "step4.rcr.train_keep")
    sample_weight = _mapping(raw.get("sample_weight_hint"), "step4.rcr.sample_weight_hint")
    export = _mapping(raw.get("export"), "step4.rcr.export")
    required = export.get("required_fields")
    if not isinstance(required, list) or not all(isinstance(x, str) and x for x in required):
        raise OneControlConfigError("step4.rcr.export.required_fields must be a non-empty string list")
    missing_required = [x for x in _STEP4_RCR_REQUIRED_EXPORT_FIELDS if x not in required]
    if missing_required:
        raise OneControlConfigError(
            "step4.rcr.export.required_fields must include the Step4 RCR contract columns: "
            + ", ".join(missing_required)
        )
    return {
        "cf_reliability_weights": _resolve_rcr_weights(
            raw,
            "cf_reliability_weights",
            ("content_retention", "rating_stability", "style_shift", "text_quality"),
        ),
        "uncertainty_weights": _resolve_rcr_weights(
            raw,
            "uncertainty_weights",
            ("rating_instability", "content_weakness", "text_quality_weakness", "entropy"),
        ),
        "rating_delta_soft_cap": _rcr_float(
            raw.get("rating_delta_soft_cap"), "step4.rcr.rating_delta_soft_cap", min_value=1e-6
        ),
        "route_scorer": {
            "min_reliability": _rcr_float(scorer.get("min_reliability"), "step4.rcr.route_scorer.min_reliability", min_value=0.0, max_value=1.0),
            "min_content_retention": _rcr_float(scorer.get("min_content_retention"), "step4.rcr.route_scorer.min_content_retention", min_value=0.0, max_value=1.0),
            "min_rating_stability": _rcr_float(scorer.get("min_rating_stability"), "step4.rcr.route_scorer.min_rating_stability", min_value=0.0, max_value=1.0),
            "max_uncertainty": _rcr_float(scorer.get("max_uncertainty"), "step4.rcr.route_scorer.max_uncertainty", min_value=0.0, max_value=1.0),
            "max_rating_delta": _rcr_float(scorer.get("max_rating_delta"), "step4.rcr.route_scorer.max_rating_delta", min_value=0.0),
            "min_text_quality": _rcr_float(scorer.get("min_text_quality"), "step4.rcr.route_scorer.min_text_quality", min_value=0.0, max_value=1.0),
        },
        "route_explainer": {
            "min_reliability": _rcr_float(explainer.get("min_reliability"), "step4.rcr.route_explainer.min_reliability", min_value=0.0, max_value=1.0),
            "relaxed_min_reliability": _rcr_float(explainer.get("relaxed_min_reliability"), "step4.rcr.route_explainer.relaxed_min_reliability", min_value=0.0, max_value=1.0),
            "min_content_retention": _rcr_float(explainer.get("min_content_retention"), "step4.rcr.route_explainer.min_content_retention", min_value=0.0, max_value=1.0),
            "min_style_shift": _rcr_float(explainer.get("min_style_shift"), "step4.rcr.route_explainer.min_style_shift", min_value=0.0, max_value=1.0),
            "max_uncertainty": _rcr_float(explainer.get("max_uncertainty"), "step4.rcr.route_explainer.max_uncertainty", min_value=0.0, max_value=1.0),
            "min_text_quality": _rcr_float(explainer.get("min_text_quality"), "step4.rcr.route_explainer.min_text_quality", min_value=0.0, max_value=1.0),
        },
        "confidence_bucket": {
            "high": {
                "bucket": _rcr_int(high.get("bucket"), "step4.rcr.confidence_bucket.high.bucket"),
                "min_reliability": _rcr_float(high.get("min_reliability"), "step4.rcr.confidence_bucket.high.min_reliability", min_value=0.0, max_value=1.0),
                "max_uncertainty": _rcr_float(high.get("max_uncertainty"), "step4.rcr.confidence_bucket.high.max_uncertainty", min_value=0.0, max_value=1.0),
                "min_rating_stability": _rcr_float(high.get("min_rating_stability"), "step4.rcr.confidence_bucket.high.min_rating_stability", min_value=0.0, max_value=1.0),
                "min_content_retention": _rcr_float(high.get("min_content_retention"), "step4.rcr.confidence_bucket.high.min_content_retention", min_value=0.0, max_value=1.0),
            },
            "medium": {
                "bucket": _rcr_int(medium.get("bucket"), "step4.rcr.confidence_bucket.medium.bucket"),
                "min_reliability": _rcr_float(medium.get("min_reliability"), "step4.rcr.confidence_bucket.medium.min_reliability", min_value=0.0, max_value=1.0),
                "max_uncertainty": _rcr_float(medium.get("max_uncertainty"), "step4.rcr.confidence_bucket.medium.max_uncertainty", min_value=0.0, max_value=1.0),
                "min_content_retention": _rcr_float(medium.get("min_content_retention"), "step4.rcr.confidence_bucket.medium.min_content_retention", min_value=0.0, max_value=1.0),
            },
            "low_bucket": int(bucket.get("low_bucket", 0)),
        },
        "train_keep": {
            "reject_when_both_routes_zero": _bool(train_keep.get("reject_when_both_routes_zero", True)),
            "reject_reason": str(train_keep.get("reject_reason", "rcr_route_reject")),
        },
        "sample_weight_hint": {
            "scorer_route_multiplier": _rcr_float(sample_weight.get("scorer_route_multiplier"), "step4.rcr.sample_weight_hint.scorer_route_multiplier", min_value=0.0),
            "explainer_only_route_multiplier": _rcr_float(sample_weight.get("explainer_only_route_multiplier"), "step4.rcr.sample_weight_hint.explainer_only_route_multiplier", min_value=0.0),
            "reliability_floor": _rcr_float(sample_weight.get("reliability_floor"), "step4.rcr.sample_weight_hint.reliability_floor", min_value=0.0, max_value=1.0),
            "reliability_scale": _rcr_float(sample_weight.get("reliability_scale"), "step4.rcr.sample_weight_hint.reliability_scale", min_value=0.0, max_value=1.0),
            "uncertainty_base": _rcr_float(sample_weight.get("uncertainty_base"), "step4.rcr.sample_weight_hint.uncertainty_base", min_value=0.0, max_value=1.0),
            "uncertainty_scale": _rcr_float(sample_weight.get("uncertainty_scale"), "step4.rcr.sample_weight_hint.uncertainty_scale", min_value=0.0, max_value=1.0),
        },
        "export": {
            "required_fields": list(required),
        },
    }


def _resolve_step4_runtime_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    raw = _mapping(_get(cfg, "step4.runtime"), "step4.runtime")
    fmt = str(raw.get("partial_format", "auto")).strip().lower()
    if fmt not in {"auto", "csv", "parquet"}:
        raise OneControlConfigError("step4.runtime.partial_format must be one of auto/csv/parquet")
    decode_threads = int(raw.get("decode_threads", 0))
    if decode_threads < 0:
        raise OneControlConfigError("step4.runtime.decode_threads must be >= 0")
    return {
        "decode_threads": decode_threads,
        "decode_chunk": _rcr_int(raw.get("decode_chunk", 4096), "step4.runtime.decode_chunk", min_value=1),
        "partial_format": fmt,
        "perf_log_interval": _rcr_int(raw.get("perf_log_interval", 10), "step4.runtime.perf_log_interval", min_value=1),
        "preflight_default_max_samples": _rcr_int(
            raw.get("preflight_default_max_samples", 128),
            "step4.runtime.preflight_default_max_samples",
            min_value=1,
        ),
        "partial_wait_timeout_seconds": _rcr_int(
            raw.get("partial_wait_timeout_seconds", 600),
            "step4.runtime.partial_wait_timeout_seconds",
            min_value=1,
        ),
    }


def _resolve_step4_step5_dedicated_exports_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    raw = _mapping(_get(cfg, "step4.step5_dedicated_exports"), "step4.step5_dedicated_exports")
    allowed = {
        "enabled",
        "output_dir_name",
        "full_audit_format",
        "scorer_train_format",
        "explainer_train_format",
        "scorer_filter",
        "explainer_filter",
        "write_gold_cf_subsplits",
        "full_audit_role",
        "atomic_write",
        "validate_after_write",
        "chunk_rows",
    }
    _reject_unknown_keys(raw, allowed, "step4.step5_dedicated_exports")

    def _fmt(key: str) -> str:
        value = str(raw.get(key) or "").strip().lower()
        if value != "parquet":
            raise OneControlConfigError(f"step4.step5_dedicated_exports.{key} must be parquet")
        return value

    def _filter(name: str, route_key: str) -> dict[str, Any]:
        obj = _mapping(raw.get(name), f"step4.step5_dedicated_exports.{name}")
        _reject_unknown_keys(
            obj,
            {"train_keep", route_key, "min_sample_weight_hint"},
            f"step4.step5_dedicated_exports.{name}",
        )
        if _bool(obj.get("train_keep")) is not True:
            raise OneControlConfigError(f"step4.step5_dedicated_exports.{name}.train_keep must be true")
        if _bool(obj.get(route_key)) is not True:
            raise OneControlConfigError(f"step4.step5_dedicated_exports.{name}.{route_key} must be true")
        min_weight = _rcr_float(
            obj.get("min_sample_weight_hint"),
            f"step4.step5_dedicated_exports.{name}.min_sample_weight_hint",
            min_value=0.0,
        )
        return {
            "train_keep": True,
            route_key: True,
            "min_sample_weight_hint": float(min_weight),
        }

    output_dir_name = str(raw.get("output_dir_name") or "").strip()
    if not output_dir_name or "/" in output_dir_name or output_dir_name in {".", ".."}:
        raise OneControlConfigError("step4.step5_dedicated_exports.output_dir_name must be a simple directory name")
    full_audit_role = str(raw.get("full_audit_role") or "").strip()
    if full_audit_role != "audit_only":
        raise OneControlConfigError("step4.step5_dedicated_exports.full_audit_role must be audit_only")
    return {
        "enabled": _bool(raw.get("enabled")),
        "output_dir_name": output_dir_name,
        "full_audit_format": _fmt("full_audit_format"),
        "scorer_train_format": _fmt("scorer_train_format"),
        "explainer_train_format": _fmt("explainer_train_format"),
        "scorer_filter": _filter("scorer_filter", "route_scorer"),
        "explainer_filter": _filter("explainer_filter", "route_explainer"),
        "write_gold_cf_subsplits": _bool(raw.get("write_gold_cf_subsplits")),
        "full_audit_role": full_audit_role,
        "atomic_write": _bool(raw.get("atomic_write")),
        "validate_after_write": _bool(raw.get("validate_after_write")),
        "chunk_rows": _rcr_int(raw.get("chunk_rows", 100000), "step4.step5_dedicated_exports.chunk_rows", min_value=1),
    }


def _resolve_step4_step5_pool_exports_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    raw = _mapping(_get(cfg, "step4.step5_pool_exports"), "step4.step5_pool_exports")
    allowed = {
        "enabled",
        "output_dir_name",
        "full_audit_role",
        "legacy_dedicated_exports_role",
        "chunk_rows",
    }
    _reject_unknown_keys(raw, allowed, "step4.step5_pool_exports")
    output_dir_name = str(raw.get("output_dir_name") or "").strip()
    if output_dir_name != "step5_pools":
        raise OneControlConfigError("step4.step5_pool_exports.output_dir_name must be step5_pools")
    full_audit_role = str(raw.get("full_audit_role") or "").strip()
    if full_audit_role != "audit_only":
        raise OneControlConfigError("step4.step5_pool_exports.full_audit_role must be audit_only")
    legacy_role = str(raw.get("legacy_dedicated_exports_role") or "").strip()
    if legacy_role != "legacy_old_filter_exports":
        raise OneControlConfigError(
            "step4.step5_pool_exports.legacy_dedicated_exports_role must be legacy_old_filter_exports"
        )
    return {
        "enabled": _bool(raw.get("enabled")),
        "output_dir_name": output_dir_name,
        "full_audit_role": full_audit_role,
        "legacy_dedicated_exports_role": legacy_role,
        "chunk_rows": _rcr_int(raw.get("chunk_rows", 100000), "step4.step5_pool_exports.chunk_rows", min_value=1),
    }


def _resolve_step4_gold_quality_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    raw = _mapping(_get(cfg, "step4.gold_quality"), "step4.gold_quality")
    _reject_unknown_keys(
        raw,
        {
            "schema_version",
            "high_min_score",
            "medium_min_score",
            "hard_reject_min_words",
            "hard_reject_max_words",
            "max_repeat_ngram_ratio",
            "good_repeat_ngram_ratio",
            "user_head_cutoff",
            "user_mid_cutoff",
            "item_head_cutoff",
            "item_mid_cutoff",
            "proxy_weights",
            "sampling_weight",
            "sanity",
        },
        "step4.gold_quality",
    )
    if str(raw.get("schema_version") or "") != "odcr_gold_quality_score/1":
        raise OneControlConfigError("step4.gold_quality.schema_version must be odcr_gold_quality_score/1")
    weights = _mapping(raw.get("proxy_weights"), "step4.gold_quality.proxy_weights")
    _reject_unknown_keys(
        weights,
        {"text_quality", "consistency", "uncertainty", "control_coverage", "evidence_alignment", "coverage_diversity"},
        "step4.gold_quality.proxy_weights",
    )
    weight_out = {
        key: _rcr_float(weights.get(key), f"step4.gold_quality.proxy_weights.{key}", min_value=0.0, max_value=1.0)
        for key in ("text_quality", "consistency", "uncertainty", "control_coverage", "evidence_alignment", "coverage_diversity")
    }
    if abs(sum(weight_out.values()) - 1.0) > 1e-6:
        raise OneControlConfigError("step4.gold_quality.proxy_weights must sum to 1.0")
    sampling = _mapping(raw.get("sampling_weight"), "step4.gold_quality.sampling_weight")
    _reject_unknown_keys(sampling, {"high", "medium", "reject", "aux_domain_multiplier"}, "step4.gold_quality.sampling_weight")
    sanity = _mapping(raw.get("sanity"), "step4.gold_quality.sanity")
    _reject_unknown_keys(
        sanity,
        {
            "medium_min_ratio",
            "high_max_ratio",
            "reject_warn_ratio",
            "target_high_range",
            "target_medium_range",
            "target_reject_range",
            "aux_high_range",
            "aux_medium_range",
            "aux_reject_range",
        },
        "step4.gold_quality.sanity",
    )

    def _range(name: str) -> list[float]:
        vals = sanity.get(name)
        if not isinstance(vals, list) or len(vals) != 2:
            raise OneControlConfigError(f"step4.gold_quality.sanity.{name} must be [min, max]")
        lo = _rcr_float(vals[0], f"step4.gold_quality.sanity.{name}[0]", min_value=0.0, max_value=1.0)
        hi = _rcr_float(vals[1], f"step4.gold_quality.sanity.{name}[1]", min_value=0.0, max_value=1.0)
        if lo > hi:
            raise OneControlConfigError(f"step4.gold_quality.sanity.{name} min must be <= max")
        return [lo, hi]

    high_min = _rcr_float(raw.get("high_min_score"), "step4.gold_quality.high_min_score", min_value=0.0, max_value=1.0)
    medium_min = _rcr_float(raw.get("medium_min_score"), "step4.gold_quality.medium_min_score", min_value=0.0, max_value=1.0)
    if medium_min > high_min:
        raise OneControlConfigError("step4.gold_quality.medium_min_score must be <= high_min_score")
    return {
        "schema_version": "odcr_gold_quality_score/1",
        "high_min_score": high_min,
        "medium_min_score": medium_min,
        "hard_reject_min_words": _rcr_int(raw.get("hard_reject_min_words"), "step4.gold_quality.hard_reject_min_words", min_value=1),
        "hard_reject_max_words": _rcr_int(raw.get("hard_reject_max_words"), "step4.gold_quality.hard_reject_max_words", min_value=1),
        "max_repeat_ngram_ratio": _rcr_float(raw.get("max_repeat_ngram_ratio"), "step4.gold_quality.max_repeat_ngram_ratio", min_value=0.0, max_value=1.0),
        "good_repeat_ngram_ratio": _rcr_float(raw.get("good_repeat_ngram_ratio"), "step4.gold_quality.good_repeat_ngram_ratio", min_value=0.0, max_value=1.0),
        "user_head_cutoff": _rcr_int(raw.get("user_head_cutoff"), "step4.gold_quality.user_head_cutoff", min_value=1),
        "user_mid_cutoff": _rcr_int(raw.get("user_mid_cutoff"), "step4.gold_quality.user_mid_cutoff", min_value=1),
        "item_head_cutoff": _rcr_int(raw.get("item_head_cutoff"), "step4.gold_quality.item_head_cutoff", min_value=1),
        "item_mid_cutoff": _rcr_int(raw.get("item_mid_cutoff"), "step4.gold_quality.item_mid_cutoff", min_value=1),
        "proxy_weights": weight_out,
        "sampling_weight": {
            "high": _rcr_float(sampling.get("high"), "step4.gold_quality.sampling_weight.high", min_value=0.0),
            "medium": _rcr_float(sampling.get("medium"), "step4.gold_quality.sampling_weight.medium", min_value=0.0),
            "reject": _rcr_float(sampling.get("reject"), "step4.gold_quality.sampling_weight.reject", min_value=0.0),
            "aux_domain_multiplier": _rcr_float(sampling.get("aux_domain_multiplier"), "step4.gold_quality.sampling_weight.aux_domain_multiplier", min_value=0.0, max_value=1.0),
        },
        "sanity": {
            "medium_min_ratio": _rcr_float(sanity.get("medium_min_ratio"), "step4.gold_quality.sanity.medium_min_ratio", min_value=0.0, max_value=1.0),
            "high_max_ratio": _rcr_float(sanity.get("high_max_ratio"), "step4.gold_quality.sanity.high_max_ratio", min_value=0.0, max_value=1.0),
            "reject_warn_ratio": _rcr_float(sanity.get("reject_warn_ratio"), "step4.gold_quality.sanity.reject_warn_ratio", min_value=0.0, max_value=1.0),
            "target_high_range": _range("target_high_range"),
            "target_medium_range": _range("target_medium_range"),
            "target_reject_range": _range("target_reject_range"),
            "aux_high_range": _range("aux_high_range"),
            "aux_medium_range": _range("aux_medium_range"),
            "aux_reject_range": _range("aux_reject_range"),
        },
    }


def _resolve_step4_cf_tiers_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    raw = _mapping(_get(cfg, "step4.cf_tiers"), "step4.cf_tiers")
    _reject_unknown_keys(raw, {"schema_version", "hard_reject", "explanation", "sampling_weight"}, "step4.cf_tiers")
    if str(raw.get("schema_version") or "") != "odcr_cf_quality_tier/1":
        raise OneControlConfigError("step4.cf_tiers.schema_version must be odcr_cf_quality_tier/1")
    hard = _mapping(raw.get("hard_reject"), "step4.cf_tiers.hard_reject")
    _reject_unknown_keys(
        hard,
        {"max_uncertainty", "min_text_quality", "min_rating_stability", "min_content_retention", "min_words", "max_repeat_ngram_ratio"},
        "step4.cf_tiers.hard_reject",
    )

    def _tier_block(head: str) -> dict[str, Any]:
        head_raw = _mapping(raw.get(head), f"step4.cf_tiers.{head}")
        _reject_unknown_keys(head_raw, {"high", "medium", "low_weighted"}, f"step4.cf_tiers.{head}")
        out: dict[str, Any] = {}
        for tier in ("high", "medium", "low_weighted"):
            item = _mapping(head_raw.get(tier), f"step4.cf_tiers.{head}.{tier}")
            allowed = {
                "require_route_explainer",
                "min_confidence_bucket",
                "min_style_shift",
                "min_reliability",
                "max_uncertainty",
                "min_text_quality",
            }
            _reject_unknown_keys(item, allowed, f"step4.cf_tiers.{head}.{tier}")
            row: dict[str, Any] = {}
            for key, value in item.items():
                if key.startswith("require_route_"):
                    row[key] = _bool(value)
                elif key == "min_confidence_bucket":
                    row[key] = _rcr_int(value, f"step4.cf_tiers.{head}.{tier}.{key}", min_value=0)
                else:
                    row[key] = _rcr_float(value, f"step4.cf_tiers.{head}.{tier}.{key}", min_value=0.0, max_value=1.0)
            out[tier] = row
        return out

    weights = _mapping(raw.get("sampling_weight"), "step4.cf_tiers.sampling_weight")
    _reject_unknown_keys(weights, {"explanation"}, "step4.cf_tiers.sampling_weight")

    def _weights(head: str) -> dict[str, float]:
        item = _mapping(weights.get(head), f"step4.cf_tiers.sampling_weight.{head}")
        _reject_unknown_keys(item, {"high", "medium", "low_weighted", "reject"}, f"step4.cf_tiers.sampling_weight.{head}")
        return {
            key: _rcr_float(item.get(key), f"step4.cf_tiers.sampling_weight.{head}.{key}", min_value=0.0)
            for key in ("high", "medium", "low_weighted", "reject")
        }

    return {
        "schema_version": "odcr_cf_quality_tier/1",
        "hard_reject": {
            "max_uncertainty": _rcr_float(hard.get("max_uncertainty"), "step4.cf_tiers.hard_reject.max_uncertainty", min_value=0.0, max_value=1.0),
            "min_text_quality": _rcr_float(hard.get("min_text_quality"), "step4.cf_tiers.hard_reject.min_text_quality", min_value=0.0, max_value=1.0),
            "min_rating_stability": _rcr_float(hard.get("min_rating_stability"), "step4.cf_tiers.hard_reject.min_rating_stability", min_value=0.0, max_value=1.0),
            "min_content_retention": _rcr_float(hard.get("min_content_retention"), "step4.cf_tiers.hard_reject.min_content_retention", min_value=0.0, max_value=1.0),
            "min_words": _rcr_int(hard.get("min_words"), "step4.cf_tiers.hard_reject.min_words", min_value=1),
            "max_repeat_ngram_ratio": _rcr_float(hard.get("max_repeat_ngram_ratio"), "step4.cf_tiers.hard_reject.max_repeat_ngram_ratio", min_value=0.0, max_value=1.0),
        },
        "explanation": _tier_block("explanation"),
        "sampling_weight": {"explanation": _weights("explanation")},
    }


def _resolve_step5_sampler_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    raw = _mapping(_get(cfg, "step5.sampler"), "step5.sampler")
    allowed = {
        "enabled",
        "contract_source",
        "effective_epoch_enabled",
        "seed",
        "rotate_across_epochs",
        "full_audit_default_allowed",
        "legacy_gold_heavy_exports_allowed",
        "auto_budget",
        "explanation",
        "epochs",
    }
    _reject_unknown_keys(raw, allowed, "step5.sampler")
    if str(raw.get("contract_source") or "") != "step4_pool_manifest":
        raise OneControlConfigError("step5.sampler.contract_source must be step4_pool_manifest")
    if _bool(raw.get("full_audit_default_allowed")):
        raise OneControlConfigError("step5.sampler.full_audit_default_allowed must be false")
    if _bool(raw.get("legacy_gold_heavy_exports_allowed")):
        raise OneControlConfigError("step5.sampler.legacy_gold_heavy_exports_allowed must be false")

    def _explanation(name: str = "explanation") -> dict[str, Any]:
        obj = _mapping(raw.get(name), f"step5.sampler.{name}")
        _reject_unknown_keys(
            obj,
            {
                "default_candidate",
                "target_gold_ratio",
                "aux_gold_ratio",
                "cf_ratio",
                "target_gold_tier_mix",
                "aux_gold_tier_mix",
                "cf_tier_mix",
                "aux_gold_weight",
                "cf_high_weight",
                "cf_medium_weight",
                "cf_low_weight",
            },
            f"step5.sampler.{name}",
        )
        default_candidate = str(obj.get("default_candidate") or "").strip()
        if default_candidate not in {"small", "medium", "full", "large"}:
            raise OneControlConfigError(f"step5.sampler.{name}.default_candidate must be small, medium, full, or large")
        tg = _rcr_float(obj.get("target_gold_ratio"), f"step5.sampler.{name}.target_gold_ratio", min_value=0.0, max_value=1.0)
        ag = _rcr_float(obj.get("aux_gold_ratio"), f"step5.sampler.{name}.aux_gold_ratio", min_value=0.0, max_value=1.0)
        cf = _rcr_float(obj.get("cf_ratio"), f"step5.sampler.{name}.cf_ratio", min_value=0.0, max_value=1.0)
        if abs((tg + ag + cf) - 1.0) > 1e-6:
            raise OneControlConfigError(f"step5.sampler.{name} ratios must sum to 1.0")
        for value, (lo, hi), key in zip(
            (tg, ag, cf),
            ((0.30, 0.45), (0.10, 0.25), (0.34, 0.50)),
            ("target_gold_ratio", "aux_gold_ratio", "cf_ratio"),
        ):
            if not (lo <= value <= hi):
                raise OneControlConfigError(f"step5.sampler.{name}.{key} must be in [{lo}, {hi}]")
        def _gold_mix(mix_name: str) -> dict[str, float]:
            mix_raw = _mapping(obj.get(mix_name), f"step5.sampler.{name}.{mix_name}")
            _reject_unknown_keys(mix_raw, {"high", "medium"}, f"step5.sampler.{name}.{mix_name}")
            out_mix = {
                "high": _rcr_float(mix_raw.get("high"), f"step5.sampler.{name}.{mix_name}.high", min_value=0.0, max_value=1.0),
                "medium": _rcr_float(mix_raw.get("medium"), f"step5.sampler.{name}.{mix_name}.medium", min_value=0.0, max_value=1.0),
            }
            if abs(sum(out_mix.values()) - 1.0) > 1e-6:
                raise OneControlConfigError(f"step5.sampler.{name}.{mix_name} values must sum to 1.0")
            return out_mix
        mix = _mapping(obj.get("cf_tier_mix"), f"step5.sampler.{name}.cf_tier_mix")
        mix_out = {
            "high": _rcr_float(mix.get("high"), f"step5.sampler.{name}.cf_tier_mix.high", min_value=0.0, max_value=1.0),
            "medium": _rcr_float(mix.get("medium"), f"step5.sampler.{name}.cf_tier_mix.medium", min_value=0.0, max_value=1.0),
            "low_weighted": _rcr_float(mix.get("low_weighted"), f"step5.sampler.{name}.cf_tier_mix.low_weighted", min_value=0.0, max_value=1.0),
        }
        if abs(sum(mix_out.values()) - 1.0) > 1e-6:
            raise OneControlConfigError(f"step5.sampler.{name}.cf_tier_mix values must sum to 1.0")
        return {
            "default_candidate": default_candidate,
            "target_gold_ratio": tg,
            "aux_gold_ratio": ag,
            "cf_ratio": cf,
            "target_gold_tier_mix": _gold_mix("target_gold_tier_mix"),
            "aux_gold_tier_mix": _gold_mix("aux_gold_tier_mix"),
            "cf_tier_mix": mix_out,
            "aux_gold_weight": _rcr_float(obj.get("aux_gold_weight"), f"step5.sampler.{name}.aux_gold_weight", min_value=0.0),
            "cf_high_weight": _rcr_float(obj.get("cf_high_weight"), f"step5.sampler.{name}.cf_high_weight", min_value=0.0),
            "cf_medium_weight": _rcr_float(obj.get("cf_medium_weight"), f"step5.sampler.{name}.cf_medium_weight", min_value=0.0),
            "cf_low_weight": _rcr_float(obj.get("cf_low_weight"), f"step5.sampler.{name}.cf_low_weight", min_value=0.0),
        }

    epochs = _mapping(raw.get("epochs"), "step5.sampler.epochs")
    pilot = epochs.get("pilot_fraction_candidates")
    if not isinstance(pilot, list) or not pilot:
        raise OneControlConfigError("step5.sampler.epochs.pilot_fraction_candidates must be a non-empty list")
    auto_budget = _mapping(raw.get("auto_budget"), "step5.sampler.auto_budget")
    _reject_unknown_keys(
        auto_budget,
        {
            "enabled",
            "capacity_basis",
            "budget_multipliers",
            "min_steps_per_effective_epoch",
            "preferred_steps_per_effective_epoch",
            "max_steps_per_effective_epoch",
            "max_replacement_rate",
        },
        "step5.sampler.auto_budget",
    )
    if _bool(auto_budget.get("enabled")) is not True:
        raise OneControlConfigError("step5.sampler.auto_budget.enabled must be true")
    if str(auto_budget.get("capacity_basis") or "") != "balanced_capacity":
        raise OneControlConfigError("step5.sampler.auto_budget.capacity_basis must be balanced_capacity")
    multipliers_raw = _mapping(auto_budget.get("budget_multipliers"), "step5.sampler.auto_budget.budget_multipliers")
    _reject_unknown_keys(multipliers_raw, {"small", "medium", "full", "large"}, "step5.sampler.auto_budget.budget_multipliers")
    multiplier_out = {
        key: _rcr_float(
            multipliers_raw.get(key),
            f"step5.sampler.auto_budget.budget_multipliers.{key}",
            min_value=0.0,
        )
        for key in ("small", "medium", "full", "large")
    }
    preferred = auto_budget.get("preferred_steps_per_effective_epoch")
    if not isinstance(preferred, list) or len(preferred) != 2:
        raise OneControlConfigError("step5.sampler.auto_budget.preferred_steps_per_effective_epoch must be a two-value list")
    preferred_out = [
        _rcr_int(preferred[0], "step5.sampler.auto_budget.preferred_steps_per_effective_epoch[0]", min_value=1),
        _rcr_int(preferred[1], "step5.sampler.auto_budget.preferred_steps_per_effective_epoch[1]", min_value=1),
    ]
    if preferred_out[0] > preferred_out[1]:
        raise OneControlConfigError("step5.sampler.auto_budget preferred step bounds must be ordered")
    min_steps = _rcr_int(auto_budget.get("min_steps_per_effective_epoch"), "step5.sampler.auto_budget.min_steps_per_effective_epoch", min_value=1)
    max_steps = _rcr_int(auto_budget.get("max_steps_per_effective_epoch"), "step5.sampler.auto_budget.max_steps_per_effective_epoch", min_value=1)
    if min_steps > preferred_out[0] or preferred_out[1] > max_steps:
        raise OneControlConfigError("step5.sampler.auto_budget step bounds must satisfy min <= preferred_low <= preferred_high <= max")
    return {
        "enabled": _bool(raw.get("enabled")),
        "contract_source": "step4_pool_manifest",
        "effective_epoch_enabled": _bool(raw.get("effective_epoch_enabled")),
        "seed": _positive_int(raw.get("seed"), "step5.sampler.seed"),
        "rotate_across_epochs": _bool(raw.get("rotate_across_epochs")),
        "full_audit_default_allowed": False,
        "legacy_gold_heavy_exports_allowed": False,
        "auto_budget": {
            "enabled": True,
            "capacity_basis": "balanced_capacity",
            "budget_multipliers": multiplier_out,
            "min_steps_per_effective_epoch": min_steps,
            "preferred_steps_per_effective_epoch": preferred_out,
            "max_steps_per_effective_epoch": max_steps,
            "max_replacement_rate": _rcr_float(
                auto_budget.get("max_replacement_rate"),
                "step5.sampler.auto_budget.max_replacement_rate",
                min_value=0.0,
                max_value=1.0,
            ),
        },
        "mode": "explanation_only",
        "route_primary": "route_explainer",
        "components": {
            "target_anchor": "optional",
            "aux_gold": "enabled",
            "cf": "enabled",
            "aux_cf": "enabled",
        },
        "explanation": _explanation(),
        "epochs": {
            "max_effective_epochs": _rcr_int(epochs.get("max_effective_epochs"), "step5.sampler.epochs.max_effective_epochs", min_value=1),
            "early_stopping_patience": _rcr_int(epochs.get("early_stopping_patience"), "step5.sampler.epochs.early_stopping_patience", min_value=0),
            "pilot_fraction_candidates": [
                _rcr_float(x, "step5.sampler.epochs.pilot_fraction_candidates", min_value=0.0, max_value=1.0)
                for x in pilot
            ],
        },
    }


def _resolve_step5_task_decoupled_policy_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    raw = _mapping(_get(cfg, "step5.task_decoupled_policy"), "step5.task_decoupled_policy")
    _reject_unknown_keys(raw, {"schema_version", "enabled", "explanation"}, "step5.task_decoupled_policy")
    explanation = _mapping(raw.get("explanation"), "step5.task_decoupled_policy.explanation")
    _reject_unknown_keys(
        explanation,
        {"branch", "train_components", "use_big_model", "allow_target_anchor", "target_anchor_role"},
        "step5.task_decoupled_policy.explanation",
    )
    components = _mapping(explanation.get("train_components"), "step5.task_decoupled_policy.explanation.train_components")
    _reject_unknown_keys(components, {"target_gold", "aux_gold", "cf"}, "step5.task_decoupled_policy.explanation.train_components")
    if str(explanation.get("branch") or "") != "explainer_rich":
        raise OneControlConfigError("step5.task_decoupled_policy.explanation.branch must be explainer_rich")
    if _bool(explanation.get("use_big_model")) is not True:
        raise OneControlConfigError("Step5 explanation must use the large explainer model")
    return {
        "schema_version": str(raw.get("schema_version") or "odcr_step5_explanation_policy/1"),
        "enabled": _bool(raw.get("enabled", True)),
        "mode": "explanation_only",
        "explanation": {
            "branch": "explainer_rich",
            "train_components": dict(components),
            "use_big_model": True,
            "allow_target_anchor": _bool(explanation.get("allow_target_anchor", True)),
            "target_anchor_role": str(
                explanation.get("target_anchor_role") or "optional_target_explanation_anchor_not_rating_supervision"
            ),
        },
        "rating_training": {"enabled": False, "source": RATING_SOURCE_TYPE},
    }

def resolve_step4_step5_dedicated_exports_config(
    *,
    config_path: str | Path,
    set_overrides: Iterable[str],
) -> tuple[dict[str, Any], dict[str, str]]:
    """Resolve the Step4 dedicated-export block without planning a formal run."""
    _reject_retired_accum_env()
    base = load_yaml_config(config_path)
    cfg, cli_sources = apply_cli_sets(base, set_overrides)
    _validate_config_shape(cfg)
    return _resolve_step4_step5_dedicated_exports_config(cfg), cli_sources


def resolve_step4_step5_pool_exports_config(
    *,
    config_path: str | Path,
    set_overrides: Iterable[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str]]:
    """Resolve Step4 pool/gold/CF and Step5 sampler blocks without planning a run."""
    _reject_retired_accum_env()
    base = load_yaml_config(config_path)
    cfg, cli_sources = apply_cli_sets(base, set_overrides)
    _validate_config_shape(cfg)
    return (
        _resolve_step4_step5_pool_exports_config(cfg),
        _resolve_step4_gold_quality_config(cfg),
        _resolve_step4_cf_tiers_config(cfg),
        _resolve_step5_sampler_config(cfg),
        cli_sources,
    )


def _resolve_step3_structured_losses_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    raw = _mapping(_get(cfg, "step3.structured_losses"), "step3.structured_losses")
    orth = _mapping(raw.get("orthogonal"), "step3.structured_losses.orthogonal")
    return {
        "orthogonal": {
            "weight": _rcr_float(
                orth.get("weight"),
                "step3.structured_losses.orthogonal.weight",
                min_value=0.0,
            ),
            "xcov_weight": _rcr_float(
                orth.get("xcov_weight"),
                "step3.structured_losses.orthogonal.xcov_weight",
                min_value=0.0,
            ),
            "cosine_weight": _rcr_float(
                orth.get("cosine_weight"),
                "step3.structured_losses.orthogonal.cosine_weight",
                min_value=0.0,
            ),
        },
        "variance_weight": _rcr_float(raw.get("variance_weight"), "step3.structured_losses.variance_weight", min_value=0.0),
        "shared_invariance_weight": _rcr_float(raw.get("shared_invariance_weight"), "step3.structured_losses.shared_invariance_weight", min_value=0.0),
        "specific_separation_weight": _rcr_float(raw.get("specific_separation_weight"), "step3.structured_losses.specific_separation_weight", min_value=0.0),
        "anchor_alignment_weight": _rcr_float(raw.get("anchor_alignment_weight"), "step3.structured_losses.anchor_alignment_weight", min_value=0.0),
        "content_alignment_weight": _rcr_float(raw.get("content_alignment_weight"), "step3.structured_losses.content_alignment_weight", min_value=0.0),
        "style_alignment_weight": _rcr_float(raw.get("style_alignment_weight"), "step3.structured_losses.style_alignment_weight", min_value=0.0),
        "shared_prototype_weight": _rcr_float(raw.get("shared_prototype_weight"), "step3.structured_losses.shared_prototype_weight", min_value=0.0),
        "domain_style_alignment_weight": _rcr_float(raw.get("domain_style_alignment_weight"), "step3.structured_losses.domain_style_alignment_weight", min_value=0.0),
        "local_style_alignment_weight": _rcr_float(raw.get("local_style_alignment_weight"), "step3.structured_losses.local_style_alignment_weight", min_value=0.0),
        "polarity_alignment_weight": _rcr_float(raw.get("polarity_alignment_weight"), "step3.structured_losses.polarity_alignment_weight", min_value=0.0),
        "residual_specific_weight": _rcr_float(raw.get("residual_specific_weight"), "step3.structured_losses.residual_specific_weight", min_value=0.0),
        "prototype_separation_weight": _rcr_float(raw.get("prototype_separation_weight"), "step3.structured_losses.prototype_separation_weight", min_value=0.0),
        "light_explainer_weight": _rcr_float(raw.get("light_explainer_weight"), "step3.structured_losses.light_explainer_weight", min_value=0.0),
    }


def _resolve_step3_ddp_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    raw = _mapping(_get(cfg, "step3.ddp"), "step3.ddp")
    return {
        "ddp_find_unused_parameters": _bool(raw.get("find_unused_parameters", False)),
        "ddp_static_graph": _bool(raw.get("static_graph", False)),
        "ddp_graph_safety_preflight": _bool(raw.get("graph_safety_preflight", True)),
    }


def _resolve_step3_loss_semantics_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    raw = _mapping(_get(cfg, "step3.loss_semantics"), "step3.loss_semantics")
    quality = _mapping(raw.get("quality_weight"), "step3.loss_semantics.quality_weight")
    return {
        "specific_separation_margin": _rcr_float(
            raw.get("specific_separation_margin"),
            "step3.loss_semantics.specific_separation_margin",
            min_value=0.0,
        ),
        "variance_target_std": _rcr_float(
            raw.get("variance_target_std"),
            "step3.loss_semantics.variance_target_std",
            min_value=0.0,
        ),
        "variance_eps": _rcr_float(raw.get("variance_eps"), "step3.loss_semantics.variance_eps", min_value=0.0),
        "orthogonal_eps": _rcr_float(raw.get("orthogonal_eps"), "step3.loss_semantics.orthogonal_eps", min_value=0.0),
        "cosine_eps": _rcr_float(raw.get("cosine_eps"), "step3.loss_semantics.cosine_eps", min_value=0.0),
        "sample_weight_eps": _rcr_float(raw.get("sample_weight_eps"), "step3.loss_semantics.sample_weight_eps", min_value=0.0),
        "prototype_separation_eps": _rcr_float(
            raw.get("prototype_separation_eps"),
            "step3.loss_semantics.prototype_separation_eps",
            min_value=0.0,
        ),
        "quality_weight": {
            "evidence_base": _rcr_float(
                quality.get("evidence_base"),
                "step3.loss_semantics.quality_weight.evidence_base",
                min_value=0.0,
            ),
            "evidence_scale": _rcr_float(
                quality.get("evidence_scale"),
                "step3.loss_semantics.quality_weight.evidence_scale",
                min_value=0.0,
            ),
            "anchor_base": _rcr_float(
                quality.get("anchor_base"),
                "step3.loss_semantics.quality_weight.anchor_base",
                min_value=0.0,
            ),
            "anchor_scale": _rcr_float(
                quality.get("anchor_scale"),
                "step3.loss_semantics.quality_weight.anchor_scale",
                min_value=0.0,
            ),
        },
    }


_STEP5_CCV_REQUIRED_CONTROL_FIELDS = (
    "content_evidence",
    "style_evidence",
    "domain_style_anchor",
    "local_style_residual_hint",
    "polarity_anchor",
    "cf_reliability_score",
    "content_retention_score",
    "style_shift_score",
    "rating_stability_score",
    "uncertainty_score",
    "confidence_bucket",
    "route_explainer",
    "route_scorer",
    "sample_weight_hint",
    "content_anchor_score",
    "style_anchor_score",
    "evidence_quality_prior",
)
_STEP5_RETIRED_BACKEND_KEYS = frozenset(
    {
        "lambda_lci",
        "lambda_fca",
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "lora_target_modules",
    }
)


def _resolve_named_float_map(
    raw: Mapping[str, Any],
    key: str,
    names: tuple[str, ...],
    *,
    min_value: float = 0.0,
    max_value: float | None = None,
) -> dict[str, float]:
    lookup_key = key.split(".")[-1]
    obj = _mapping(raw.get(lookup_key), key)
    missing = [name for name in names if name not in obj]
    extra = sorted(set(obj) - set(names))
    if missing or extra:
        raise OneControlConfigError(f"{key} mismatch; missing={missing}, extra={extra}")
    return {
        name: _rcr_float(obj[name], f"{key}.{name}", min_value=min_value, max_value=max_value)
        for name in names
    }


def _resolve_step5_innovation_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    raw = _mapping(_get(cfg, "step5"), "step5")
    train = raw.get("train")
    backend = train.get("backend") if isinstance(train, Mapping) else None
    if isinstance(backend, Mapping):
        retired = sorted(k for k in _STEP5_RETIRED_BACKEND_KEYS if k in backend)
        if retired:
            raise OneControlConfigError(
                "step5.train.backend contains retired Step5 LCI/FCA/LoRA keys "
                f"{retired}; use step5.lci / step5.fca / step5.ccv.native_lora instead."
            )
    lci = _mapping(raw.get("lci"), "step5.lci")
    uci = _mapping(raw.get("uci"), "step5.uci")
    explainer_gate = _mapping(raw.get("explainer_gate"), "step5.explainer_gate")
    ccv = _mapping(raw.get("ccv"), "step5.ccv")
    fca = _mapping(raw.get("fca"), "step5.fca")
    native_lora = _mapping(ccv.get("native_lora"), "step5.ccv.native_lora")
    control_fields = ccv.get("control_fields")
    if not isinstance(control_fields, list) or not all(isinstance(x, str) and x for x in control_fields):
        raise OneControlConfigError("step5.ccv.control_fields must be a non-empty string list")
    missing_control = [x for x in _STEP5_CCV_REQUIRED_CONTROL_FIELDS if x not in control_fields]
    if missing_control:
        raise OneControlConfigError(
            "step5.ccv.control_fields missing required Step5 explanation control fields: "
            + ", ".join(missing_control)
        )
    mode = str(fca.get("evidence_alignment_mode", "")).strip().lower()
    if mode not in ("evidence_basis",):
        raise OneControlConfigError("step5.fca.evidence_alignment_mode must be 'evidence_basis'")
    packet_policy = str(ccv.get("control_packet_field_policy", "")).strip().lower()
    if packet_policy not in ("strict_required",):
        raise OneControlConfigError("step5.ccv.control_packet_field_policy must be 'strict_required'")
    adapter_policy = str(ccv.get("verbalizer_adapter_policy", "")).strip().lower()
    if adapter_policy not in ("ccv_control_adapter",):
        raise OneControlConfigError("step5.ccv.verbalizer_adapter_policy must be 'ccv_control_adapter'")
    soft_prompt_len = _positive_int(ccv.get("soft_prompt_len"), "step5.ccv.soft_prompt_len")
    numeric_control_dim = _positive_int(ccv.get("numeric_control_dim"), "step5.ccv.numeric_control_dim")
    if numeric_control_dim != 13:
        raise OneControlConfigError("step5.ccv.numeric_control_dim must be 13 for the current CCVControlPacket")
    control_adapter_blocks = _positive_int(
        ccv.get("control_adapter_input_blocks"),
        "step5.ccv.control_adapter_input_blocks",
    )
    if control_adapter_blocks != 6:
        raise OneControlConfigError("step5.ccv.control_adapter_input_blocks must be 6 for the current CCV adapter")
    lora_targets = native_lora.get("target_modules", [])
    if not isinstance(lora_targets, list) or not all(isinstance(x, str) for x in lora_targets):
        raise OneControlConfigError("step5.ccv.native_lora.target_modules must be a string list")
    lora_targets_clean = [x.strip() for x in lora_targets if x.strip()]
    if not lora_targets_clean:
        raise OneControlConfigError(
            "step5.ccv.native_lora.target_modules=[] is retired; use "
            f"[{_STEP5_HEAD_AWARE_LORA_TARGET_SENTINEL!r}] or an explicit head-aware allowlist subset."
        )
    if _STEP5_HEAD_AWARE_LORA_TARGET_SENTINEL in lora_targets_clean and lora_targets_clean != [
        _STEP5_HEAD_AWARE_LORA_TARGET_SENTINEL
    ]:
        raise OneControlConfigError(
            f"{_STEP5_HEAD_AWARE_LORA_TARGET_SENTINEL} must be the only item when selecting the default head-aware LoRA allowlist."
        )
    lora_policy_id = str(native_lora.get("target_policy_id") or _STEP5_LORA_TARGET_POLICY_ID).strip()
    if lora_policy_id != _STEP5_LORA_TARGET_POLICY_ID:
        raise OneControlConfigError(
            f"step5.ccv.native_lora.target_policy_id must be {_STEP5_LORA_TARGET_POLICY_ID!r}"
        )
    explainer_min_weight = _rcr_float(
        explainer_gate.get("min_weight"),
        "step5.explainer_gate.min_weight",
        min_value=0.0,
    )
    explainer_max_weight = _rcr_float(
        explainer_gate.get("max_weight"),
        "step5.explainer_gate.max_weight",
        min_value=0.0,
    )
    if explainer_max_weight < explainer_min_weight:
        raise OneControlConfigError("step5.explainer_gate.max_weight must be >= min_weight")
    return {
        "lci": {
            "enabled": _bool(lci.get("enabled", True)),
            "weight": _rcr_float(lci.get("weight"), "step5.lci.weight", min_value=0.0),
            "confidence_schedule": _resolve_named_float_map(
                lci,
                "step5.lci.confidence_schedule",
                ("high", "medium", "low"),
                min_value=0.0,
            ),
            "min_reliability": _rcr_float(lci.get("min_reliability"), "step5.lci.min_reliability", min_value=0.0, max_value=1.0),
            "max_uncertainty": _rcr_float(lci.get("max_uncertainty"), "step5.lci.max_uncertainty", min_value=0.0, max_value=1.0),
            "perturb_std": _rcr_float(lci.get("perturb_std"), "step5.lci.perturb_std", min_value=0.0),
            "counterfactual_label_weight": _rcr_float(lci.get("counterfactual_label_weight"), "step5.lci.counterfactual_label_weight", min_value=0.0),
            "robustness_weight": _rcr_float(lci.get("robustness_weight"), "step5.lci.robustness_weight", min_value=0.0),
        },
        "uci": {
            "enabled": _bool(uci.get("enabled", True)),
            "bucket_weights": _resolve_named_float_map(
                uci,
                "step5.uci.bucket_weights",
                ("high", "medium", "low"),
                min_value=0.0,
            ),
            "uncertainty_temperature": _rcr_float(uci.get("uncertainty_temperature"), "step5.uci.uncertainty_temperature", min_value=1e-6),
            "low_confidence_floor": _rcr_float(uci.get("low_confidence_floor", 0.0), "step5.uci.low_confidence_floor", min_value=0.0, max_value=1.0),
        },
        "explainer_gate": {
            "bucket_weights": _resolve_named_float_map(
                explainer_gate,
                "step5.explainer_gate.bucket_weights",
                ("high", "medium", "low"),
                min_value=0.0,
            ),
            "uncertainty_exponent": _rcr_float(
                explainer_gate.get("uncertainty_exponent"),
                "step5.explainer_gate.uncertainty_exponent",
                min_value=1e-6,
            ),
            "style_shift_diversity_boost": _rcr_float(
                explainer_gate.get("style_shift_diversity_boost"),
                "step5.explainer_gate.style_shift_diversity_boost",
                min_value=0.0,
            ),
            "min_weight": explainer_min_weight,
            "max_weight": explainer_max_weight,
            "explainer_only_multiplier": _rcr_float(
                explainer_gate.get("explainer_only_multiplier"),
                "step5.explainer_gate.explainer_only_multiplier",
                min_value=0.0,
            ),
        },
        "ccv": {
            "enabled": _bool(ccv.get("enabled", True)),
            "control_fields": list(control_fields),
            "uncertainty_tone_control": _bool(ccv.get("uncertainty_tone_control", True)),
            "route_conditioning": _bool(ccv.get("route_conditioning", True)),
            "numeric_control_weight": _rcr_float(ccv.get("numeric_control_weight"), "step5.ccv.numeric_control_weight", min_value=0.0),
            "control_packet_field_policy": packet_policy,
            "verbalizer_adapter_policy": adapter_policy,
            "soft_prompt_len": soft_prompt_len,
            "numeric_control_dim": numeric_control_dim,
            "control_adapter_input_blocks": control_adapter_blocks,
            "native_lora": {
                "enabled": _bool(native_lora.get("enabled", True)),
                "target_policy_id": lora_policy_id,
                "r": _positive_int(native_lora.get("r"), "step5.ccv.native_lora.r"),
                "alpha": _rcr_float(native_lora.get("alpha"), "step5.ccv.native_lora.alpha", min_value=0.0),
                "dropout": _rcr_float(
                    native_lora.get("dropout"),
                    "step5.ccv.native_lora.dropout",
                    min_value=0.0,
                    max_value=1.0,
                ),
                "target_modules": lora_targets_clean,
            },
        },
        "fca": {
            "enabled": _bool(fca.get("enabled", True)),
            "weight": _rcr_float(fca.get("weight"), "step5.fca.weight", min_value=0.0),
            "min_reliability": _rcr_float(fca.get("min_reliability"), "step5.fca.min_reliability", min_value=0.0, max_value=1.0),
            "max_uncertainty": _rcr_float(fca.get("max_uncertainty"), "step5.fca.max_uncertainty", min_value=0.0, max_value=1.0),
            "evidence_alignment_mode": mode,
        },
    }


def _validate_train_batch(stage: str, train: Mapping[str, Any], ddp_world_size: int) -> tuple[int, int, int]:
    _reject_retired_accum_keys(train, f"{stage}.train")
    batch_size = _positive_int(train.get("batch_size"), f"{stage}.train.batch_size")
    per_gpu = _positive_int(train.get("per_gpu_batch_size"), f"{stage}.train.per_gpu_batch_size")
    expected = per_gpu * int(ddp_world_size)
    if batch_size != expected:
        raise OneControlConfigError(
            f"{stage}.train batch formula failed: batch_size={batch_size} but "
            f"per_gpu_batch_size({per_gpu}) * ddp_world_size({ddp_world_size}) = {expected}. "
            f"ODCR uses {NO_ACCUM_BATCH_SEMANTICS_VERSION} semantics."
        )
    return batch_size, per_gpu, expected


def _validate_eval_batch(eval_batch_size: int | None, ddp_world_size: int) -> int | None:
    if eval_batch_size is None:
        return None
    ebs = _positive_int(eval_batch_size, "eval_batch_size")
    if ebs % int(ddp_world_size) != 0:
        raise OneControlConfigError(
            f"eval_batch_size={ebs} must be divisible by ddp_world_size={ddp_world_size}"
        )
    return ebs // int(ddp_world_size)


def _resolve_step5_lifecycle_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    raw = _mapping(_mapping(cfg.get("step5"), "step5").get("lifecycle"), "step5.lifecycle")
    allowed = {
        "schema_version",
        "formal_default_phase",
        "embedded_final_eval_default",
        "allow_embedded_final_eval_diagnostic",
        "explanation_handoff_required_for_downstream",
        "checkpoint_load_policy",
        "cpu_staged_checkpoint_load_required",
        "write_latest_after_train_only",
        "e5_post_train_lifecycle_required",
    }
    _reject_unknown_keys(raw, allowed, "step5.lifecycle")
    phase = str(raw.get("formal_default_phase") or "train_only").strip()
    if phase not in {"train_only", "eval_only", "full"}:
        raise OneControlConfigError("step5.lifecycle.formal_default_phase must be train_only, eval_only, or full")
    policy = str(raw.get("checkpoint_load_policy") or "cpu_staged").strip()
    if policy != "cpu_staged":
        raise OneControlConfigError("step5.lifecycle.checkpoint_load_policy must be cpu_staged")
    return {
        "schema_version": str(raw.get("schema_version") or "odcr_step5_lifecycle/1"),
        "formal_default_phase": phase,
        "embedded_final_eval_default": _bool(raw.get("embedded_final_eval_default", False)),
        "allow_embedded_final_eval_diagnostic": _bool(raw.get("allow_embedded_final_eval_diagnostic", False)),
        "explanation_handoff_required_for_downstream": _bool(raw.get("explanation_handoff_required_for_downstream", True)),
        "checkpoint_load_policy": policy,
        "cpu_staged_checkpoint_load_required": _bool(raw.get("cpu_staged_checkpoint_load_required", True)),
        "write_latest_after_train_only": _bool(raw.get("write_latest_after_train_only", False)),
        "e5_post_train_lifecycle_required": _bool(raw.get("e5_post_train_lifecycle_required", True)),
    }


def needs_decode_layer(command: str, *, step5_train_only: bool = False) -> bool:
    """Return whether the resolver command consumes generation/decode config."""
    if command == "step5" and step5_train_only:
        return False
    return command in ("step4", "step5", "eval")


def _latest_run(repo_root: Path, task_id: int, stage: str, *, dry_run: bool) -> str:
    try:
        return resolve_upstream(repo_root=repo_root, task=int(task_id), stage=stage, mode="formal", repair=True).run_id
    except UpstreamResolutionError as exc:
        raise OneControlConfigError(str(exc)) from exc


def _alloc_run(repo_root: Path, task_id: int, stage: str, requested: str | None, *, dry_run: bool) -> str:
    if dry_run:
        return run_naming.parse_run_id(requested) if requested and requested not in ("auto", "") else "dry_run"
    parent = path_layout.get_stage_task_root(repo_root, stage, task_id)
    parent.mkdir(parents=True, exist_ok=True)
    return run_naming.allocate_child_dir(parent, requested=requested, kind="run")


def _alloc_step4_run(repo_root: Path, task_id: int, step3_run: str, requested: str | None, *, dry_run: bool) -> str:
    parent = path_layout.get_stage_task_root(repo_root, "step4", task_id)
    if requested and requested not in ("", "auto"):
        rid = run_naming.parse_run_id(requested)
        if (parent / rid).exists():
            raise OneControlConfigError(f"Step4 run id already exists and will not be overwritten: {parent / rid}")
        return rid
    if dry_run:
        return run_naming.next_run_id(parent)
    parent.mkdir(parents=True, exist_ok=True)
    return run_naming.allocate_child_dir(parent, requested=None, kind="run")


def _stage_root(repo_root: Path, task_id: int, stage: str, run_id: str) -> Path:
    return path_layout.get_stage_run_root(repo_root, task_id, "v1", stage, run_id).resolve()


def _default_decode_placeholder() -> dict[str, Any]:
    return deepcopy(SAFE_DECODE_PLACEHOLDER)


def _resolve_eval_profile(cfg: Mapping[str, Any], profile_name: str | None) -> tuple[str, dict[str, Any]]:
    eval_cfg = _get(cfg, "eval", {})
    if not isinstance(eval_cfg, Mapping):
        raise OneControlConfigError("eval must be a mapping")
    name = str(profile_name or eval_cfg.get("profile") or "balanced_2gpu")
    profiles = eval_cfg.get("profiles", {})
    if not isinstance(profiles, Mapping) or name not in profiles:
        raise OneControlConfigError(f"eval profile {name!r} not found under eval.profiles")
    raw = profiles[name]
    if not isinstance(raw, Mapping):
        raise OneControlConfigError(f"eval.profiles.{name} must be a mapping")
    return name, dict(raw)


def _resolve_decode(cfg: Mapping[str, Any], name: str | None, *, need_decode: bool) -> tuple[str, dict[str, Any]]:
    if not need_decode:
        return "", _default_decode_placeholder()
    eval_cfg = _get(cfg, "eval", {})
    decode_cfg = eval_cfg.get("decode", {}) if isinstance(eval_cfg, Mapping) else {}
    if not isinstance(decode_cfg, Mapping):
        raise OneControlConfigError("eval.decode must be a mapping")
    base = decode_cfg.get("default", {})
    if not isinstance(base, Mapping):
        raise OneControlConfigError("eval.decode.default must be a mapping")
    stem = str(name or "mainline")
    overlay = decode_cfg.get(stem, {})
    if stem != "default" and not isinstance(overlay, Mapping):
        raise OneControlConfigError(f"eval.decode.{stem} must be a mapping")
    merged = _merge_dicts(base, overlay if stem != "default" else {})
    return stem, merged


def _resolve_rerank(cfg: Mapping[str, Any], name: str | None, *, need_rerank: bool) -> tuple[str, dict[str, Any]]:
    if not need_rerank:
        return "", {}
    eval_cfg = _get(cfg, "eval", {})
    rerank_cfg = eval_cfg.get("rerank", {}) if isinstance(eval_cfg, Mapping) else {}
    if not isinstance(rerank_cfg, Mapping):
        raise OneControlConfigError("eval.rerank must be a mapping")
    base = rerank_cfg.get("default", {})
    if not isinstance(base, Mapping):
        raise OneControlConfigError("eval.rerank.default must be a mapping")
    stem = str(name or "quality")
    overlay = rerank_cfg.get(stem, {})
    if stem != "default" and not isinstance(overlay, Mapping):
        raise OneControlConfigError(f"eval.rerank.{stem} must be a mapping")
    return stem, _merge_dicts(base, overlay if stem != "default" else {})


def _resolve_step3_scenario_profile(
    cfg: Mapping[str, Any],
    task: Mapping[str, Any],
    *,
    task_id: int,
) -> tuple[str, str, dict[str, Any]]:
    scenario = str(task.get("scenario") or "legacy_scenario").strip()
    direction = str(task.get("direction") or "unspecified").strip()
    if not scenario:
        raise OneControlConfigError(f"tasks.{task_id}.scenario must be non-empty for Step3 resolution")
    profiles = _mapping(_get(cfg, "step3.scenario_profiles"), "step3.scenario_profiles")
    raw = profiles.get(scenario)
    if not isinstance(raw, Mapping):
        raise OneControlConfigError(
            f"tasks.{task_id}.scenario={scenario!r} has no step3.scenario_profiles entry"
        )
    return scenario, direction, dict(raw)


def _select_step3_task_profile(
    cfg: Mapping[str, Any],
    task: Mapping[str, Any],
    *,
    task_id: int,
) -> tuple[str, dict[str, Any]]:
    profiles = _mapping(_get(cfg, "step3.task_profiles"), "step3.task_profiles")
    matched: list[tuple[str, Mapping[str, Any]]] = []
    for key, raw in profiles.items():
        if isinstance(raw, Mapping) and int(raw.get("task_id") or -1) == int(task_id):
            matched.append((str(key), raw))
    if len(matched) != 1:
        raise OneControlConfigError(
            f"Step3 requires exactly one isolated task profile for task{task_id}; found {len(matched)}."
        )
    key, raw = matched[0]
    source = str(raw.get("source") or "")
    target = str(raw.get("target") or "")
    if source != str(task.get("source")) or target != str(task.get("target")):
        raise OneControlConfigError(
            f"step3.task_profiles.{key} must not remap task{task_id}: "
            f"profile={source}->{target}, tasks.{task_id}={task.get('source')}->{task.get('target')}"
        )
    for required in ("scenario", "direction", "active_profile", "profile_id", "train"):
        if required not in raw:
            raise OneControlConfigError(f"step3.task_profiles.{key}.{required} is required.")
    if str(raw.get("scenario")) != str(task.get("scenario")) or str(raw.get("direction")) != str(task.get("direction")):
        raise OneControlConfigError(
            f"step3.task_profiles.{key} scenario/direction must match tasks.{task_id}; task id remap is forbidden."
        )
    return key, dict(raw)


def _profile_activation_checkpointing(value: Any, *, context: str) -> str:
    if isinstance(value, bool):
        return "selective" if value else "off"
    text = str(value or "off").strip()
    if text not in ("off", "selective"):
        raise OneControlConfigError(f"{context}.activation_checkpointing must be off/false or selective/true.")
    return text


def _resolve_step3_task_profile_config(
    *,
    key: str,
    raw: Mapping[str, Any],
    task_id: int,
    task: Mapping[str, Any],
    ddp_world_size: int,
    train: Mapping[str, Any],
    tokenizer_config: Mapping[str, Any],
    evidence_config: Mapping[str, Any],
    scheduler_config: Mapping[str, Any],
    gather_config: Mapping[str, Any],
    memory_config: Mapping[str, Any],
) -> dict[str, Any]:
    profile_id = str(raw.get("profile_id") or "").strip()
    active_profile = str(raw.get("active_profile") or "").strip()
    if not profile_id or not active_profile:
        raise OneControlConfigError(f"step3.task_profiles.{key} must define profile_id and active_profile.")
    if int(task_id) == 2:
        if str(raw.get("candidate") or "") != "G1S":
            raise OneControlConfigError("task2 Step3 formal profile must expose candidate=G1S.")
        if profile_id != "task2_strong_forward_g1s":
            raise OneControlConfigError("task2 Step3 formal profile must resolve to task2_strong_forward_g1s.")
        if not _bool(raw.get("formal_allowed", False)) or _bool(raw.get("probe_only", False)):
            raise OneControlConfigError("task2 Step3 G1S must be formal_allowed=true and probe_only=false.")
    else:
        if _bool(raw.get("formal_allowed", False)):
            raise OneControlConfigError(f"step3.task_profiles.{key} is isolated/profile-ready only, not formal-ready.")
    profile_expected_pool = _positive_int(raw.get("effective_pool_expected"), f"step3.task_profiles.{key}.effective_pool_expected")
    per_gpu = _positive_int(train.get("per_gpu_batch_size"), "step3.task_profile.resolved.per_gpu_batch_size")
    batch = _positive_int(train.get("batch_size"), "step3.task_profile.resolved.batch_size")
    resolved_effective_pool = per_gpu * int(ddp_world_size)
    if batch != resolved_effective_pool:
        raise OneControlConfigError(
            f"step3.task_profiles.{key} resolved batch_size={batch} must equal "
            f"per_gpu_batch_size({per_gpu}) * ddp_world_size({ddp_world_size})."
        )
    profile = {
        "key": key,
        "task_id": int(task_id),
        "source": str(task["source"]),
        "target": str(task["target"]),
        "scenario": str(raw.get("scenario")),
        "direction": str(raw.get("direction")),
        "active_profile": active_profile,
        "profile_id": profile_id,
        "candidate": str(raw.get("candidate") or ""),
        "formal_allowed": _bool(raw.get("formal_allowed", True)),
        "probe_only": _bool(raw.get("probe_only", False)),
        "train": {
            "batch_size": batch,
            "per_gpu_batch_size": per_gpu,
            "micro_batch_size_alias": per_gpu,
            "ddp_world_size": int(ddp_world_size),
            "lr": float(train.get("lr")),
            "max_grad_norm": float(train.get("max_grad_norm")),
            "backend": dict(train.get("backend") or {}),
        },
        "tokenizer": dict(tokenizer_config),
        "evidence": dict(evidence_config),
        "scheduler": dict(scheduler_config),
        "cross_rank_structured_gather": {
            "enabled": bool(gather_config.get("enabled")),
            "mode": str(gather_config.get("mode")),
            "allowed_tensors": list(gather_config.get("allowed_tensors") or []),
            "forbidden_tensors": list(gather_config.get("forbidden_tensors") or []),
        },
        "effective_structured_pool": {
            "local_per_gpu_batch": per_gpu,
            "local_micro_batch_alias": per_gpu,
            "ddp_world_size": int(ddp_world_size),
            "effective_pool_expected": resolved_effective_pool,
            "task_profile_default_effective_pool_expected": profile_expected_pool,
            "matches_task_profile_default": profile_expected_pool == resolved_effective_pool,
            "formula": "effective_pool_expected == per_gpu_batch_size * ddp_world_size",
            "gathered_tensor_names": list(gather_config.get("allowed_tensors") or []),
            "remote_tensors_detached": True,
        },
        "memory": dict(memory_config),
        "isolation_contract": {
            "task_id_source_target_bound": True,
            "does_not_mutate_other_task_profiles": True,
            "tokenizer_cache_namespace_excludes_training_profile_parameters": True,
        },
    }
    profile["profile_isolation_hash"] = fingerprint(
        {
            "task": {
                "task_id": profile["task_id"],
                "source": profile["source"],
                "target": profile["target"],
                "scenario": profile["scenario"],
                "direction": profile["direction"],
            },
            "profile_id": profile_id,
            "active_profile": active_profile,
            "train": profile["train"],
            "tokenizer": profile["tokenizer"],
            "evidence": profile["evidence"],
            "scheduler": profile["scheduler"],
            "gather": profile["cross_rank_structured_gather"],
            "memory": profile["memory"],
        }
    )
    return profile


def _resolve_step3_optimizer_config(cfg: Mapping[str, Any], task_profile: Mapping[str, Any] | None = None) -> dict[str, Any]:
    raw = _mapping(_get(cfg, "step3.optimizer"), "step3.optimizer")
    profile_optimizer = task_profile.get("optimizer") if isinstance(task_profile, Mapping) else None
    if profile_optimizer is not None and str(profile_optimizer).strip().lower() not in ("adamw",):
        raise OneControlConfigError("Step3 task profile optimizer must be AdamW.")
    name = str(raw.get("name") or "").strip().lower()
    if name != "adamw":
        raise OneControlConfigError("step3.optimizer.name must be 'adamw'.")
    betas_raw = raw.get("betas")
    if not isinstance(betas_raw, (list, tuple)) or len(betas_raw) != 2:
        raise OneControlConfigError("step3.optimizer.betas must be a two-item list")
    betas = [float(betas_raw[0]), float(betas_raw[1])]
    if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
        raise OneControlConfigError("step3.optimizer.betas must be in [0, 1)")
    groups = _mapping(raw.get("param_groups"), "step3.optimizer.param_groups")
    return {
        "name": name,
        "betas": betas,
        "eps": _rcr_float(raw.get("eps"), "step3.optimizer.eps", min_value=0.0),
        "param_groups": {
            "dense_weight_decay": _rcr_float(
                groups.get("dense_weight_decay"),
                "step3.optimizer.param_groups.dense_weight_decay",
                min_value=0.0,
            ),
            "special_weight_decay": _rcr_float(
                groups.get("special_weight_decay"),
                "step3.optimizer.param_groups.special_weight_decay",
                min_value=0.0,
            ),
            "no_decay": _rcr_float(
                groups.get("no_decay"),
                "step3.optimizer.param_groups.no_decay",
                min_value=0.0,
            ),
        },
        "exclude_frozen_evidence_buffers": _bool(raw.get("exclude_frozen_evidence_buffers", True)),
    }


def _resolve_step3_backend_config(train: Mapping[str, Any]) -> dict[str, Any]:
    backend = _mapping(train.get("backend"), "step3.train.backend")
    out = {
        "train_precision": str(backend.get("train_precision") or "").strip().lower(),
        "allow_tf32": _bool(backend.get("allow_tf32", False)),
        "amp_autocast": _bool(backend.get("amp_autocast", False)),
        "grad_scaler": _bool(backend.get("grad_scaler", True)),
    }
    if out["train_precision"] != "bf16":
        raise OneControlConfigError("step3.train.backend.train_precision must be bf16 for ODCR Step3 v0.")
    if not out["allow_tf32"]:
        raise OneControlConfigError("step3.train.backend.allow_tf32 must be true for ODCR Step3 v0.")
    if not out["amp_autocast"]:
        raise OneControlConfigError("step3.train.backend.amp_autocast must be true for ODCR Step3 v0.")
    if out["grad_scaler"]:
        raise OneControlConfigError("step3.train.backend.grad_scaler must be false for bf16 Step3 v0.")
    return out


def _resolve_step3_tokenizer_evidence_config(
    cfg: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    base_tok = _mapping(_get(cfg, "step3.tokenizer"), "step3.tokenizer")
    base_evd = _mapping(_get(cfg, "step3.evidence"), "step3.evidence")
    prof_tok = profile.get("tokenizer") if isinstance(profile.get("tokenizer"), Mapping) else {}
    prof_evd = profile.get("evidence") if isinstance(profile.get("evidence"), Mapping) else {}
    tok = _merge_dicts(base_tok, prof_tok)
    evd = _merge_dicts(base_evd, prof_evd)
    tok_len = _positive_int(tok.get("max_length"), "step3.tokenizer.max_length")
    evd_len = _positive_int(evd.get("max_evidence_length"), "step3.evidence.max_evidence_length")
    return {"max_length": tok_len}, {"max_evidence_length": evd_len}


def _resolve_step3_scheduler_config(cfg: Mapping[str, Any], task_profile: Mapping[str, Any] | None = None) -> dict[str, Any]:
    raw = _merge_dicts(
        _mapping(_get(cfg, "step3.scheduler"), "step3.scheduler"),
        task_profile.get("scheduler") if isinstance(task_profile, Mapping) else None,
    )
    name = str(raw.get("name") or "").strip().lower()
    if name not in {"warmup_cosine", "safe_damping_v2"}:
        raise OneControlConfigError(
            "step3.scheduler.name must be warmup_cosine or safe_damping_v2 for Step3 V3."
        )
    warmup_ratio = _rcr_float(raw.get("warmup_ratio"), "step3.scheduler.warmup_ratio", min_value=0.0)
    min_lr_ratio = _rcr_float(raw.get("min_lr_ratio"), "step3.scheduler.min_lr_ratio", min_value=0.0)
    if warmup_ratio <= 0.0 or warmup_ratio >= 1.0:
        raise OneControlConfigError("step3.scheduler.warmup_ratio must be in (0,1)")
    if min_lr_ratio < 0.0 or min_lr_ratio >= 1.0:
        raise OneControlConfigError("step3.scheduler.min_lr_ratio must be in [0,1)")
    damping_raw = _mapping(raw.get("validation_aware_lr_damping"), "step3.scheduler.validation_aware_lr_damping")
    damping_enabled = _bool(damping_raw.get("enabled", False))
    damping = {
        "enabled": damping_enabled,
        "name": str(damping_raw.get("name") or "safe_damping_v2"),
        "base_scheduler": "warmup_cosine",
        "formal_allowed": _bool(damping_raw.get("formal_allowed", False)),
        "probe_only": _bool(damping_raw.get("probe_only", True)),
        "monitor_metric": str(damping_raw.get("monitor_metric") or "valid_loss"),
        "direction": str(damping_raw.get("direction") or "min").strip().lower(),
        "worsen_abs_threshold": _rcr_float(
            damping_raw.get("worsen_abs_threshold", 0.25),
            "step3.scheduler.validation_aware_lr_damping.worsen_abs_threshold",
            min_value=0.0,
        ),
        "worsen_ratio_threshold": _rcr_float(
            damping_raw.get("worsen_ratio_threshold", 0.10),
            "step3.scheduler.validation_aware_lr_damping.worsen_ratio_threshold",
            min_value=0.0,
        ),
        "worsen_patience": _positive_int(
            damping_raw.get("worsen_patience", 3),
            "step3.scheduler.validation_aware_lr_damping.worsen_patience",
        ),
        "lr_decay_factor": _rcr_float(
            damping_raw.get("lr_decay_factor", 0.5),
            "step3.scheduler.validation_aware_lr_damping.lr_decay_factor",
            min_value=0.0,
        ),
        "min_lr_ratio": _rcr_float(
            damping_raw.get("min_lr_ratio", min_lr_ratio),
            "step3.scheduler.validation_aware_lr_damping.min_lr_ratio",
            min_value=0.0,
        ),
        "cooldown_epochs": _nonnegative_int(
            damping_raw.get("cooldown_epochs", 3),
            "step3.scheduler.validation_aware_lr_damping.cooldown_epochs",
        ),
        "max_damping_events": _positive_int(
            damping_raw.get("max_damping_events", 2),
            "step3.scheduler.validation_aware_lr_damping.max_damping_events",
        ),
        "start_epoch": _positive_int(
            damping_raw.get("start_epoch", 4),
            "step3.scheduler.validation_aware_lr_damping.start_epoch",
        ),
        "recent_trend_epochs": _positive_int(
            damping_raw.get("recent_trend_epochs", 2),
            "step3.scheduler.validation_aware_lr_damping.recent_trend_epochs",
        ),
        "recent_recovery_tolerance": _rcr_float(
            damping_raw.get("recent_recovery_tolerance", 1.0e-3),
            "step3.scheduler.validation_aware_lr_damping.recent_recovery_tolerance",
            min_value=0.0,
        ),
        "effective_lr_floor_abs": _rcr_float(
            damping_raw.get("effective_lr_floor_abs", 0.0),
            "step3.scheduler.validation_aware_lr_damping.effective_lr_floor_abs",
            min_value=0.0,
        ),
        "effective_lr_floor_ratio": _rcr_float(
            damping_raw.get("effective_lr_floor_ratio", 0.25),
            "step3.scheduler.validation_aware_lr_damping.effective_lr_floor_ratio",
            min_value=0.0,
        ),
        "effective_min_lr_policy": str(
            damping_raw.get("effective_min_lr_policy")
            or "safe_floor_no_halve_to_zero"
        ),
        "action_on_max_events": str(damping_raw.get("action_on_max_events") or "stop_and_select_candidate"),
        "action_on_low_lr_no_progress": str(
            damping_raw.get("action_on_low_lr_no_progress") or "stop_and_select_candidate"
        ),
    }
    if damping["monitor_metric"] != "valid_loss" or damping["direction"] != "min":
        raise OneControlConfigError("Step3 validation-aware damping currently supports valid_loss/min only.")
    if not (0.0 < float(damping["lr_decay_factor"]) < 1.0):
        raise OneControlConfigError("step3.scheduler.validation_aware_lr_damping.lr_decay_factor must be in (0,1).")
    if int(damping["worsen_patience"]) < 3:
        raise OneControlConfigError("safe_damping_v2 requires worsen_patience >= 3.")
    if int(damping["cooldown_epochs"]) < 3:
        raise OneControlConfigError("safe_damping_v2 requires cooldown_epochs >= 3.")
    if int(damping["max_damping_events"]) > 3:
        raise OneControlConfigError("safe_damping_v2 requires max_damping_events <= 3.")
    if float(damping["effective_lr_floor_ratio"]) < 0.25:
        raise OneControlConfigError("safe_damping_v2 requires effective_lr_floor_ratio >= 0.25.")
    if float(damping["min_lr_ratio"]) < float(min_lr_ratio):
        damping["min_lr_ratio"] = float(min_lr_ratio)
    if name == "warmup_cosine" and damping_enabled:
        raise OneControlConfigError(
            "hidden Step3 LR damping is forbidden: step3.scheduler.name=warmup_cosine requires "
            "step3.scheduler.validation_aware_lr_damping.enabled=false."
        )
    if name == "safe_damping_v2" and not damping_enabled:
        raise OneControlConfigError(
            "step3.scheduler.name=safe_damping_v2 requires "
            "step3.scheduler.validation_aware_lr_damping.enabled=true."
        )
    if name == "safe_damping_v2" and bool(damping["formal_allowed"]):
        raise OneControlConfigError("safe_damping_v2 is probe-only and cannot be formal_allowed.")
    return {
        "name": name,
        "base_scheduler": "warmup_cosine",
        "damping_enabled": bool(damping_enabled),
        "warmup_ratio": float(warmup_ratio),
        "min_lr_ratio": float(min_lr_ratio),
        "base_min_lr_ratio": float(min_lr_ratio),
        "validation_aware_lr_damping": damping,
    }


def _resolve_step3_objective_drift_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    raw = _mapping(_get(cfg, "step3.objective_drift"), "step3.objective_drift")
    return {
        "enabled": _bool(raw.get("enabled", True)),
        "valid_loss_abs_threshold": _rcr_float(raw.get("valid_loss_abs_threshold", 0.25), "step3.objective_drift.valid_loss_abs_threshold", min_value=0.0),
        "valid_loss_ratio_threshold": _rcr_float(raw.get("valid_loss_ratio_threshold", 0.10), "step3.objective_drift.valid_loss_ratio_threshold", min_value=0.0),
        "severe_valid_loss_abs_threshold": _rcr_float(raw.get("severe_valid_loss_abs_threshold", 0.75), "step3.objective_drift.severe_valid_loss_abs_threshold", min_value=0.0),
        "severe_valid_loss_ratio_threshold": _rcr_float(raw.get("severe_valid_loss_ratio_threshold", 0.20), "step3.objective_drift.severe_valid_loss_ratio_threshold", min_value=0.0),
        "component_weighted_delta_threshold": _rcr_float(raw.get("component_weighted_delta_threshold", 0.01), "step3.objective_drift.component_weighted_delta_threshold", min_value=0.0),
        "severe_component_count": _positive_int(raw.get("severe_component_count", 3), "step3.objective_drift.severe_component_count"),
        "statuses": list(raw.get("statuses") or ["none", "warning", "objective_drift", "severe_objective_drift"]),
        "actions": dict(raw.get("actions") or {}),
    }


def _resolve_step3_recovery_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    raw = _mapping(_get(cfg, "step3.recovery"), "step3.recovery")
    source_scope = str(raw.get("source_checkpoint_scope") or "best_observed")
    if source_scope != "best_observed":
        raise OneControlConfigError("step3.recovery.source_checkpoint_scope must be best_observed.")
    scheduler = str(raw.get("recovery_scheduler") or "short_cosine")
    if scheduler != "short_cosine":
        raise OneControlConfigError("step3.recovery.recovery_scheduler must be short_cosine.")
    return {
        "enabled": _bool(raw.get("enabled", True)),
        "trigger": str(raw.get("trigger") or "severe_objective_drift"),
        "restart_lr_ratio": _rcr_float(raw.get("restart_lr_ratio", 0.25), "step3.recovery.restart_lr_ratio", min_value=0.0),
        "recovery_epochs": _positive_int(raw.get("recovery_epochs", 8), "step3.recovery.recovery_epochs"),
        "max_recoveries": _positive_int(raw.get("max_recoveries", 1), "step3.recovery.max_recoveries"),
        "source_checkpoint_scope": source_scope,
        "save_drift_checkpoint": _bool(raw.get("save_drift_checkpoint", True)),
        "recovery_scheduler": scheduler,
        "formal_allowed": _bool(raw.get("formal_allowed", True)),
    }


def _resolve_step3_phase_loss_schedule_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    raw = _mapping(_get(cfg, "step3.phase_loss_schedule"), "step3.phase_loss_schedule")
    phases_raw = raw.get("phases") or []
    if not isinstance(phases_raw, list) or not phases_raw:
        raise OneControlConfigError("step3.phase_loss_schedule.phases must be a non-empty list.")
    phases: list[dict[str, Any]] = []
    for idx, item in enumerate(phases_raw):
        if not isinstance(item, Mapping):
            raise OneControlConfigError(f"step3.phase_loss_schedule.phases[{idx}] must be an object.")
        multipliers_raw = item.get("loss_multipliers") or {}
        if not isinstance(multipliers_raw, Mapping):
            raise OneControlConfigError(f"step3.phase_loss_schedule.phases[{idx}].loss_multipliers must be an object.")
        phases.append(
            {
                "name": str(item.get("name") or ""),
                "start_epoch": _positive_int(item.get("start_epoch", 1), f"step3.phase_loss_schedule.phases[{idx}].start_epoch"),
                "end_epoch": None if item.get("end_epoch") in (None, "") else _positive_int(item.get("end_epoch"), f"step3.phase_loss_schedule.phases[{idx}].end_epoch"),
                "loss_multipliers": {
                    str(key): _rcr_float(value, f"step3.phase_loss_schedule.phases[{idx}].loss_multipliers.{key}", min_value=0.0)
                    for key, value in multipliers_raw.items()
                },
            }
        )
    names = {phase["name"] for phase in phases}
    required = {"alignment_warmup", "task_refinement", "light_regularization"}
    if not required.issubset(names):
        raise OneControlConfigError("step3.phase_loss_schedule must define alignment_warmup, task_refinement, and light_regularization.")
    return {
        "enabled": _bool(raw.get("enabled", True)),
        "transition": str(raw.get("transition") or "epoch_or_objective_drift"),
        "phases": phases,
    }


def _resolve_step3_conflict_aware_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    raw = _mapping(_get(cfg, "step3.conflict_aware"), "step3.conflict_aware")
    mode = str(raw.get("mode") or "off")
    allowed = list(raw.get("allowed_modes") or ["off", "dynamic_weighting", "pcgrad", "gradnorm", "uncertainty_weighting"])
    if mode not in allowed:
        raise OneControlConfigError("step3.conflict_aware.mode must be one of step3.conflict_aware.allowed_modes.")
    if mode in {"pcgrad", "gradnorm", "uncertainty_weighting"} and _bool(raw.get("formal_allowed", False)):
        raise OneControlConfigError("conflict-aware optimizer modes require probe evidence before formal_allowed=true.")
    return {
        "enabled": _bool(raw.get("enabled", False)),
        "mode": mode,
        "allowed_modes": allowed,
        "default_mode": str(raw.get("default_mode") or "off"),
        "formal_allowed": _bool(raw.get("formal_allowed", False)),
        "requires_gradient_probe": _bool(raw.get("requires_gradient_probe", True)),
        "ddp_graph_safe_zero_required": _bool(raw.get("ddp_graph_safe_zero_required", True)),
    }


def _resolve_step3_loss_gradient_conflict_probe_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    raw = _mapping(_get(cfg, "step3.loss_gradient_conflict_probe"), "step3.loss_gradient_conflict_probe")
    if not _bool(raw.get("real_data_only", True)):
        raise OneControlConfigError("step3.loss_gradient_conflict_probe.real_data_only must be true.")
    return {
        "enabled": _bool(raw.get("enabled", True)),
        "real_data_only": True,
        "synthetic_benchmark_forbidden": _bool(raw.get("synthetic_benchmark_forbidden", True)),
        "bounded_max_batches": _positive_int(raw.get("bounded_max_batches", 4), "step3.loss_gradient_conflict_probe.bounded_max_batches"),
        "output_dir": str(raw.get("output_dir") or "AI_analysis/01_raw_logs/step3_loss_gradient_conflict_probe"),
        "write_formal_checkpoint": _bool(raw.get("write_formal_checkpoint", False)),
    }


def _resolve_step3_adapter_gating_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    raw = _mapping(_get(cfg, "step3.adapter_gating"), "step3.adapter_gating")
    enabled = _bool(raw.get("enabled", False))
    formal_allowed = _bool(raw.get("formal_allowed", False))
    if enabled and formal_allowed:
        raise OneControlConfigError("step3.adapter_gating cannot be enabled for formal use before gradient-probe evidence.")
    return {
        "enabled": enabled,
        "formal_allowed": formal_allowed,
        "requires_gradient_probe": _bool(raw.get("requires_gradient_probe", True)),
        "rating_adapter": str(raw.get("rating_adapter") or "off"),
        "shared_specific_gate": str(raw.get("shared_specific_gate") or "off"),
        "explainer_adapter": str(raw.get("explainer_adapter") or "off"),
        "style_content_residual_gate": str(raw.get("style_content_residual_gate") or "off"),
        "checkpoint_compatibility": str(raw.get("checkpoint_compatibility") or "disabled_no_state_change"),
    }


def _resolve_step3_paper_candidate_selection_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    raw = _mapping(_get(cfg, "step3.paper_candidate_selection"), "step3.paper_candidate_selection")
    rating_guard = _mapping(raw.get("rating_guard"), "step3.paper_candidate_selection.rating_guard")
    diversity_guard = _mapping(raw.get("diversity_guard"), "step3.paper_candidate_selection.diversity_guard")
    weights = _mapping(raw.get("composite_weights"), "step3.paper_candidate_selection.composite_weights")
    outputs = _mapping(raw.get("outputs"), "step3.paper_candidate_selection.outputs")
    return {
        "enabled": _bool(raw.get("enabled", True)),
        "eval_protocol": str(raw.get("eval_protocol") or "paper_target_only_eval"),
        "bounded_eval_required": _bool(raw.get("bounded_eval_required", True)),
        "candidate_pool": list(raw.get("candidate_pool") or []),
        "no_paper_eval_no_selection": _bool(raw.get("no_paper_eval_no_selection", True)),
        "scorer_explainer_split": _bool(raw.get("scorer_explainer_split", True)),
        "rating_guard": {
            "max_mae": _rcr_float(rating_guard.get("max_mae", 0.65), "step3.paper_candidate_selection.rating_guard.max_mae", min_value=0.0),
            "max_rmse": _rcr_float(rating_guard.get("max_rmse", 0.95), "step3.paper_candidate_selection.rating_guard.max_rmse", min_value=0.0),
            "max_mae_worse_abs": _rcr_float(rating_guard.get("max_mae_worse_abs", 0.02), "step3.paper_candidate_selection.rating_guard.max_mae_worse_abs", min_value=0.0),
            "max_rmse_worse_abs": _rcr_float(rating_guard.get("max_rmse_worse_abs", 0.03), "step3.paper_candidate_selection.rating_guard.max_rmse_worse_abs", min_value=0.0),
        },
        "diversity_guard": {
            "dist1_floor": _rcr_float(diversity_guard.get("dist1_floor", 0.05), "step3.paper_candidate_selection.diversity_guard.dist1_floor", min_value=0.0),
            "dist2_floor": _rcr_float(diversity_guard.get("dist2_floor", 0.20), "step3.paper_candidate_selection.diversity_guard.dist2_floor", min_value=0.0),
            "collapse_penalty": _rcr_float(diversity_guard.get("collapse_penalty", 100.0), "step3.paper_candidate_selection.diversity_guard.collapse_penalty", min_value=0.0),
        },
        "composite_weights": {str(key): _rcr_float(value, f"step3.paper_candidate_selection.composite_weights.{key}", min_value=0.0) for key, value in weights.items()},
        "outputs": {
            "paper_candidate_selection": str(outputs.get("paper_candidate_selection") or "paper_candidate_selection.json"),
            "candidate_eval_registry": str(outputs.get("candidate_eval_registry") or "candidate_eval_registry.jsonl"),
        },
    }


def _resolve_step3_checkpoint_averaging_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    raw = _mapping(_get(cfg, "step3.checkpoint_averaging"), "step3.checkpoint_averaging")
    return {
        "enabled": _bool(raw.get("enabled", False)),
        "candidate_only": _bool(raw.get("candidate_only", True)),
        "never_overwrite_best_observed": _bool(raw.get("never_overwrite_best_observed", True)),
        "requires_lineage": _bool(raw.get("requires_lineage", True)),
        "requires_paper_eval_before_downstream": _bool(raw.get("requires_paper_eval_before_downstream", True)),
    }


def _resolve_step3_eval_config(train: Mapping[str, Any], stage_cfg: Mapping[str, Any], ddp_world_size: int) -> dict[str, Any]:
    raw = _mapping(stage_cfg.get("eval"), "step3.eval")
    protocol = normalize_eval_protocol(str(raw.get("protocol") or MINIMAL_EVAL))
    split = str(raw.get("split") or "valid").strip().lower()
    if split not in {"valid", "test"}:
        raise OneControlConfigError("step3.eval.split must be valid or test.")
    paper_splits_raw = raw.get("paper_splits") or ["valid", "test"]
    if not isinstance(paper_splits_raw, list):
        raise OneControlConfigError("step3.eval.paper_splits must be a list.")
    paper_splits = [str(item).strip().lower() for item in paper_splits_raw]
    bad_splits = sorted(set(paper_splits) - {"valid", "test"})
    if bad_splits:
        raise OneControlConfigError(f"step3.eval.paper_splits has unsupported split(s): {bad_splits}")
    batch_candidates_raw = raw.get("batch_candidates") or []
    if not isinstance(batch_candidates_raw, list) or not batch_candidates_raw:
        raise OneControlConfigError("step3.eval.batch_candidates must be a non-empty list.")
    batch_candidates = [
        _positive_int(value, "step3.eval.batch_candidates[]")
        for value in batch_candidates_raw
    ]
    optional_batch_candidates_raw = raw.get("optional_batch_candidates") or []
    if not isinstance(optional_batch_candidates_raw, list):
        raise OneControlConfigError("step3.eval.optional_batch_candidates must be a list.")
    optional_batch_candidates = [
        _positive_int(value, "step3.eval.optional_batch_candidates[]")
        for value in optional_batch_candidates_raw
    ]
    for candidate in [*batch_candidates, *optional_batch_candidates]:
        if candidate % int(ddp_world_size) != 0:
            raise OneControlConfigError(
                f"step3.eval batch candidate {candidate} is not divisible by ddp_world_size={ddp_world_size}."
            )
    derive = _bool(raw.get("derive_from_train", True))
    valid_batch = raw.get("valid_batch_size")
    valid_micro = raw.get("valid_micro_batch_size")
    paper_cfg = _mapping(raw.get(PAPER_TARGET_ONLY_EVAL), f"step3.eval.{PAPER_TARGET_ONLY_EVAL}")
    diag_cfg = _mapping(raw.get(ODCR_STEP3_DIAGNOSTIC), f"step3.eval.{ODCR_STEP3_DIAGNOSTIC}")
    paper_ref = _positive_int(paper_cfg.get("max_ref_len", 25), f"step3.eval.{PAPER_TARGET_ONLY_EVAL}.max_ref_len")
    paper_dec = _positive_int(paper_cfg.get("max_decode_len", 25), f"step3.eval.{PAPER_TARGET_ONLY_EVAL}.max_decode_len")
    diag_ref = _positive_int(diag_cfg.get("max_ref_len", 48), f"step3.eval.{ODCR_STEP3_DIAGNOSTIC}.max_ref_len")
    diag_dec = _positive_int(diag_cfg.get("max_decode_len", 48), f"step3.eval.{ODCR_STEP3_DIAGNOSTIC}.max_decode_len")
    if paper_ref != 25 or paper_dec != 25:
        raise OneControlConfigError("paper_target_only_eval must use max_ref_len=max_decode_len=25.")
    if bool(paper_cfg.get("berts_score_enabled", paper_cfg.get("bert_score_enabled", False))):
        raise OneControlConfigError("paper_target_only_eval must keep BERTScore disabled.")
    protocol_config = step3_eval_protocol_spec(
        protocol,
        split=split,
        diagnostic_text_len=diag_ref,
        paper_text_len=paper_ref,
    )
    if protocol == PAPER_TARGET_ONLY_EVAL:
        protocol_config["max_decode_len"] = paper_dec
    if protocol == ODCR_STEP3_DIAGNOSTIC:
        protocol_config["max_decode_len"] = diag_dec
    if derive:
        if valid_batch is not None or valid_micro is not None:
            raise OneControlConfigError(
                "step3.eval.derive_from_train=true requires valid_batch_size and valid_micro_batch_size to be null."
            )
        resolved_micro = _positive_int(train.get("per_gpu_batch_size"), "step3.train.per_gpu_batch_size")
        resolved_global = resolved_micro * int(ddp_world_size)
        return {
            "protocol": protocol,
            "split": split,
            "paper_splits": paper_splits,
            "derive_from_train": True,
            "valid_batch_size": resolved_global,
            "valid_micro_batch_size": resolved_micro,
            "batch_candidates": batch_candidates,
            "optional_batch_candidates": optional_batch_candidates,
            "selected_eval_batch": str(raw.get("selected_eval_batch") or "auto_largest_safe"),
            "invariance_required": _bool(raw.get("invariance_required", True)),
            "protocol_config": protocol_config,
            "minimal_eval": dict(_mapping(raw.get("minimal_eval"), "step3.eval.minimal_eval")),
            ODCR_STEP3_DIAGNOSTIC: dict(diag_cfg),
            PAPER_TARGET_ONLY_EVAL: dict(paper_cfg),
            "full_pipeline_final_eval": dict(_mapping(raw.get("full_pipeline_final_eval"), "step3.eval.full_pipeline_final_eval")),
            "source": "resolver-derived from step3.train.per_gpu_batch_size * ddp_world_size",
        }
    resolved_batch = _positive_int(valid_batch, "step3.eval.valid_batch_size")
    resolved_micro = _positive_int(valid_micro, "step3.eval.valid_micro_batch_size")
    if resolved_batch != resolved_micro * int(ddp_world_size):
        raise OneControlConfigError(
            "step3.eval valid batch formula failed: "
            f"{resolved_batch} != {resolved_micro} * {ddp_world_size}"
        )
    return {
        "protocol": protocol,
        "split": split,
        "paper_splits": paper_splits,
        "derive_from_train": False,
        "valid_batch_size": resolved_batch,
        "valid_micro_batch_size": resolved_micro,
        "batch_candidates": batch_candidates,
        "optional_batch_candidates": optional_batch_candidates,
        "selected_eval_batch": str(raw.get("selected_eval_batch") or "auto_largest_safe"),
        "invariance_required": _bool(raw.get("invariance_required", True)),
        "protocol_config": protocol_config,
        "minimal_eval": dict(_mapping(raw.get("minimal_eval"), "step3.eval.minimal_eval")),
        ODCR_STEP3_DIAGNOSTIC: dict(diag_cfg),
        PAPER_TARGET_ONLY_EVAL: dict(paper_cfg),
        "full_pipeline_final_eval": dict(_mapping(raw.get("full_pipeline_final_eval"), "step3.eval.full_pipeline_final_eval")),
        "source": "step3.eval",
    }


def _resolve_step3_backup_profiles_config(cfg: Mapping[str, Any], ddp_world_size: int) -> dict[str, Any]:
    raw = _mapping(_get(cfg, "step3.backup_profiles"), "step3.backup_profiles")
    out: dict[str, Any] = {}
    for name, item in raw.items():
        if not isinstance(item, Mapping):
            raise OneControlConfigError(f"step3.backup_profiles.{name} must be a mapping")
        batch = _positive_int(item.get("batch_size"), f"step3.backup_profiles.{name}.batch_size")
        micro = _positive_int(item.get("per_gpu_batch_size"), f"step3.backup_profiles.{name}.per_gpu_batch_size")
        if batch != micro * int(ddp_world_size):
            raise OneControlConfigError(f"step3.backup_profiles.{name} batch formula failed.")
        if not _bool(item.get("cross_rank_structured_gather", False)):
            raise OneControlConfigError(f"step3.backup_profiles.{name} requires cross_rank_structured_gather=true.")
        if not _bool(item.get("backup_only", False)) or not _bool(item.get("manual_selection_required", False)):
            raise OneControlConfigError(f"step3.backup_profiles.{name} must be backup_only and manual_selection_required.")
        if _bool(item.get("formal_allowed", False)):
            raise OneControlConfigError(f"step3.backup_profiles.{name} must not be formal_allowed.")
        out[str(name)] = {
            "base_task_profile": str(item.get("base_task_profile") or ""),
            "task_profile_id": str(item.get("task_profile_id") or ""),
            "candidate": str(item.get("candidate") or ""),
            "batch_size": batch,
            "per_gpu_batch_size": micro,
            "micro_batch_size_alias": micro,
            "ddp_world_size": int(ddp_world_size),
            "cross_rank_structured_gather": True,
            "gather_mode": str(item.get("gather_mode") or "local_gradient_context"),
            "effective_pool_expected": _positive_int(
                item.get("effective_pool_expected", batch),
                f"step3.backup_profiles.{name}.effective_pool_expected",
            ),
            "activation_checkpointing": _profile_activation_checkpointing(
                item.get("activation_checkpointing", "off"),
                context=f"step3.backup_profiles.{name}",
            ),
            "profile_buffer_policy": str(item.get("profile_buffer_policy") or "gpu_resident"),
            "backup_only": True,
            "manual_selection_required": True,
            "not_default": _bool(item.get("not_default", True)),
            "formal_allowed": False,
            "probe_only": _bool(item.get("probe_only", False)),
        }
    return out


def _resolve_step3_exploration_profiles_config(cfg: Mapping[str, Any], ddp_world_size: int) -> dict[str, Any]:
    raw = _mapping(_get(cfg, "step3.exploration_profiles"), "step3.exploration_profiles")
    out: dict[str, Any] = {}
    for name, item in raw.items():
        if not isinstance(item, Mapping):
            raise OneControlConfigError(f"step3.exploration_profiles.{name} must be a mapping")
        batch = _positive_int(item.get("batch_size"), f"step3.exploration_profiles.{name}.batch_size")
        micro = _positive_int(item.get("per_gpu_batch_size"), f"step3.exploration_profiles.{name}.per_gpu_batch_size")
        if batch != micro * int(ddp_world_size):
            raise OneControlConfigError(f"step3.exploration_profiles.{name} batch formula failed.")
        if not _bool(item.get("cross_rank_structured_gather", False)):
            raise OneControlConfigError(f"step3.exploration_profiles.{name} requires cross_rank_structured_gather=true.")
        formal_allowed = _bool(item.get("formal_allowed", False))
        probe_only = _bool(item.get("probe_only", True))
        if formal_allowed or not probe_only:
            raise OneControlConfigError(f"step3.exploration_profiles.{name} must be probe_only=true and formal_allowed=false.")
        if not _bool(item.get("exploration_only", False)):
            raise OneControlConfigError(f"step3.exploration_profiles.{name} must be exploration_only=true.")
        replacement_status = str(item.get("replacement_gate_status") or "").strip()
        if replacement_status not in ("failed_or_not_passed",):
            raise OneControlConfigError(
                f"step3.exploration_profiles.{name}.replacement_gate_status must be failed_or_not_passed."
            )
        out[str(name)] = {
            "base_task_profile": str(item.get("base_task_profile") or ""),
            "task_profile_id": str(item.get("task_profile_id") or ""),
            "candidate": str(item.get("candidate") or name),
            "batch_size": batch,
            "per_gpu_batch_size": micro,
            "micro_batch_size_alias": micro,
            "ddp_world_size": int(ddp_world_size),
            "cross_rank_structured_gather": True,
            "gather_mode": str(item.get("gather_mode") or "local_gradient_context"),
            "effective_pool_expected": _positive_int(
                item.get("effective_pool_expected", batch),
                f"step3.exploration_profiles.{name}.effective_pool_expected",
            ),
            "activation_checkpointing": _profile_activation_checkpointing(
                item.get("activation_checkpointing", "off"),
                context=f"step3.exploration_profiles.{name}",
            ),
            "profile_buffer_policy": str(item.get("profile_buffer_policy") or "gpu_resident"),
            "formal_allowed": False,
            "probe_only": True,
            "exploration_only": True,
            "replacement_gate_status": replacement_status,
            "promotion_policy": str(item.get("promotion_policy") or "manual_only"),
        }
    return out


def _resolve_step3_batch_candidate_role(
    train: Mapping[str, Any],
    *,
    active_task_profile_id: str,
    active_candidate: str = "",
) -> str:
    profile_id = str(active_task_profile_id or "").strip()
    candidate = str(active_candidate or "").strip()
    if profile_id == "task2_strong_forward_g1s" and candidate == "G1S":
        return "G1S"
    if profile_id == "task2_strong_forward_g1" and candidate == "G1":
        return "G1_backup"
    return f"{profile_id}:{candidate}" if candidate else profile_id


def _resolve_step3_worker_profiles_config(
    cfg: Mapping[str, Any],
    *,
    ddp_world_size: int,
    max_parallel_cpu: int,
) -> dict[str, Any]:
    raw = _mapping(_get(cfg, "step3.worker_profiles"), "step3.worker_profiles")
    out: dict[str, Any] = {}
    reserved_cpu = 2
    for name, item in raw.items():
        if not isinstance(item, Mapping):
            raise OneControlConfigError(f"step3.worker_profiles.{name} must be a mapping")
        workers = _positive_int(
            item.get("train_workers_per_rank"),
            f"step3.worker_profiles.{name}.train_workers_per_rank",
        )
        pf = _positive_int(item.get("prefetch_factor"), f"step3.worker_profiles.{name}.prefetch_factor")
        budget = workers * int(ddp_world_size) + reserved_cpu
        if budget > int(max_parallel_cpu):
            raise OneControlConfigError(
                f"step3.worker_profiles.{name} exceeds CPU budget: "
                f"{workers} * {ddp_world_size} + {reserved_cpu} = {budget} > {max_parallel_cpu}"
            )
        out[str(name)] = {
            "train_workers_per_rank": workers,
            "prefetch_factor": pf,
            "reserved_cpu": reserved_cpu,
            "cpu_budget": budget,
            "max_parallel_cpu": int(max_parallel_cpu),
            "role": str(item.get("role") or "performance_candidate"),
        }
    return out


def _resolve_step3_prefetcher_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    raw = _mapping(_get(cfg, "step3.prefetcher"), "step3.prefetcher")
    fallback_policy = str(raw.get("fallback_policy") or "fail_fast").strip().lower()
    if fallback_policy not in ("fail_fast", "record_warning"):
        raise OneControlConfigError("step3.prefetcher.fallback_policy must be fail_fast or record_warning.")
    return {
        "enabled": _bool(raw.get("enabled", True)),
        "double_buffer": _bool(raw.get("double_buffer", True)),
        "diagnostic_cpu_mode": _bool(raw.get("diagnostic_cpu_mode", False)),
        "measure_cuda_events": _bool(raw.get("measure_cuda_events", False)),
        "fallback_policy": fallback_policy,
        "evidence_schema_version": "odcr_step3_prefetch_evidence/1",
        "evidence_fields": list(PREFETCH_EVIDENCE_FIELDS),
    }


def _resolve_step3_checkpoint_policy_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    raw = _mapping(_get(cfg, "step3.checkpoint_policy"), "step3.checkpoint_policy")
    direction = str(raw.get("selection_direction") or "min").strip().lower()
    if direction != "min":
        raise OneControlConfigError("step3.checkpoint_policy.selection_direction must be min.")
    top_k = _positive_int(raw.get("top_k", 3), "step3.checkpoint_policy.top_k")
    if top_k < 3:
        raise OneControlConfigError("step3.checkpoint_policy.top_k must be >= 3.")
    per_epoch = _mapping(raw.get("per_epoch"), "step3.checkpoint_policy.per_epoch")
    downstream_scope = str(raw.get("downstream_default_scope") or "best_observed").strip()
    if downstream_scope != "best_observed":
        raise OneControlConfigError("Step3 downstream_default_scope must be best_observed.")
    return {
        "schema_version": str(raw.get("schema_version") or STEP3_CHECKPOINT_POLICY_VERSION),
        "selection_metric": str(raw.get("selection_metric") or "valid_loss"),
        "selection_direction": direction,
        "downstream_default_scope": downstream_scope,
        "best_alias": str(raw.get("best_alias") or "best_observed"),
        "keep_best_pth_alias": _bool(raw.get("keep_best_pth_alias", True)),
        "top_k": top_k,
        "per_epoch": {
            "enabled": _bool(per_epoch.get("enabled", False)),
            "keep_interval": _positive_int(per_epoch.get("keep_interval", 1), "step3.checkpoint_policy.per_epoch.keep_interval"),
        },
        "save_optimizer_state": _bool(raw.get("save_optimizer_state", True)),
    }


def _resolve_step3_quality_gate_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    raw = _mapping(_get(cfg, "step3.quality_gate"), "step3.quality_gate")
    missing_policy = str(raw.get("missing_quality_sidecar_policy") or "block").strip().lower()
    if missing_policy not in ("block", "warning"):
        raise OneControlConfigError("step3.quality_gate.missing_quality_sidecar_policy must be block or warning.")
    return {
        "schema_version": str(raw.get("schema_version") or STEP3_QUALITY_GATE_VERSION),
        "missing_quality_sidecar_policy": missing_policy,
        "grad_inf_count_block_threshold": _nonnegative_int(raw.get("grad_inf_count_block_threshold", 0), "step3.quality_gate.grad_inf_count_block_threshold"),
        "continuous_nonfinite_block_threshold": _nonnegative_int(raw.get("continuous_nonfinite_block_threshold", 3), "step3.quality_gate.continuous_nonfinite_block_threshold"),
        "post_clip_zero_block_ratio": _rcr_float(raw.get("post_clip_zero_block_ratio", 0.20), "step3.quality_gate.post_clip_zero_block_ratio", min_value=0.0),
        "empty_pred_rate_block_threshold": _rcr_float(raw.get("empty_pred_rate_block_threshold", 0.50), "step3.quality_gate.empty_pred_rate_block_threshold", min_value=0.0, max_value=1.0),
        "distinct_zero_blocks": _bool(raw.get("distinct_zero_blocks", True)),
        "valid_loss_deterioration_ratio_block_threshold": _rcr_float(raw.get("valid_loss_deterioration_ratio_block_threshold", 0.25), "step3.quality_gate.valid_loss_deterioration_ratio_block_threshold", min_value=0.0),
        "timing_unknown_ratio_warn_threshold": _rcr_float(raw.get("timing_unknown_ratio_warn_threshold", 0.05), "step3.quality_gate.timing_unknown_ratio_warn_threshold", min_value=0.0),
        "timing_unknown_ratio_block_threshold": _rcr_float(raw.get("timing_unknown_ratio_block_threshold", 0.50), "step3.quality_gate.timing_unknown_ratio_block_threshold", min_value=0.0),
        "block_on_timing_not_closed": _bool(raw.get("block_on_timing_not_closed", False)),
        "run_summary_fields": ["quality_status", "downstream_ready", "quality_block_reasons", "quality_warnings"],
    }


def _resolve_step3_grad_finite_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    raw = _mapping(_get(cfg, "step3.grad_finite"), "step3.grad_finite")
    return {
        "enabled": _bool(raw.get("enabled", True)),
        "skip_optimizer_on_nonfinite": _bool(raw.get("skip_optimizer_on_nonfinite", True)),
        "scheduler_step_on_skipped_optimizer": _bool(raw.get("scheduler_step_on_skipped_optimizer", False)),
        "continuous_nonfinite_abort_threshold": _nonnegative_int(raw.get("continuous_nonfinite_abort_threshold", 3), "step3.grad_finite.continuous_nonfinite_abort_threshold"),
        "monitor_interval_steps": _positive_int(raw.get("monitor_interval_steps", 50), "step3.grad_finite.monitor_interval_steps"),
        "anomaly_topk": _positive_int(raw.get("anomaly_topk", 5), "step3.grad_finite.anomaly_topk"),
        "full_scan_on_anomaly": _bool(raw.get("full_scan_on_anomaly", True)),
    }


def _resolve_step3_diagnostic_eval_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    raw = _mapping(_get(cfg, "step3.diagnostic_eval"), "step3.diagnostic_eval")
    protocols = _mapping(raw.get("protocols"), "step3.diagnostic_eval.protocols")
    out_protocols = dict(DIAGNOSTIC_PROTOCOLS)
    for name, value in protocols.items():
        if isinstance(value, Mapping):
            out_protocols[str(name)] = _merge_dicts(out_protocols.get(str(name), {}), value)
    required = set(DIAGNOSTIC_PROTOCOLS)
    missing = sorted(required - set(out_protocols))
    if missing:
        raise OneControlConfigError(f"step3.diagnostic_eval.protocols missing required protocols: {missing}")
    return {
        "schema_version": STEP3_DIAGNOSTIC_PROTOCOL_VERSION,
        "sample_schema_version": str(raw.get("sample_schema_version") or "odcr_step3_diagnostic_sample/1"),
        "collapse_schema_version": str(raw.get("collapse_schema_version") or "odcr_step3_collapse_stats/1"),
        "samples_required": _bool(raw.get("samples_required", True)),
        "max_samples": _positive_int(raw.get("max_samples", 256), "step3.diagnostic_eval.max_samples"),
        "protocols": out_protocols,
    }


def _resolve_step3_cross_rank_gather_config(
    cfg: Mapping[str, Any],
    task_profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    raw = _merge_dicts(
        task_profile.get("cross_rank_structured_gather") if isinstance(task_profile, Mapping) else None,
        _mapping(_get(cfg, "step3.cross_rank_structured_gather"), "step3.cross_rank_structured_gather"),
    )
    mode = str(raw.get("mode") or "local_gradient_context").strip()
    if mode != "local_gradient_context":
        raise OneControlConfigError("step3.cross_rank_structured_gather.mode must be local_gradient_context.")
    allowed = [str(x) for x in (raw.get("allowed_tensors") or [])]
    forbidden = [str(x) for x in (raw.get("forbidden_tensors") or [])]
    forbidden_required = {
        "raw_text",
        "token_ids",
        "profile_matrices",
        "domain_profile_matrices",
        "full_profile_matrices",
        "large_vocab_logits",
        "checkpoint_state_tensors",
    }
    if not forbidden_required.issubset(set(forbidden)):
        raise OneControlConfigError(
            "step3.cross_rank_structured_gather.forbidden_tensors must include raw_text, token_ids, "
            "profile_matrices, domain_profile_matrices, full_profile_matrices, large_vocab_logits, "
            "and checkpoint_state_tensors."
        )
    enabled = _bool(raw.get("enabled", False))
    diagnostic_allow_disabled = _bool(raw.get("diagnostic_allow_disabled", False))
    if not enabled and not diagnostic_allow_disabled:
        raise OneControlConfigError(
            "Step3 formal ODCR v0 requires step3.cross_rank_structured_gather.enabled=true; "
            "use diagnostic_allow_disabled=true only for explicit diagnostics."
        )
    return {
        "enabled": enabled,
        "mode": mode,
        "diagnostic_allow_disabled": diagnostic_allow_disabled,
        "allowed_tensors": allowed,
        "forbidden_tensors": forbidden,
    }


def _resolve_step3_memory_config(cfg: Mapping[str, Any], task_profile: Mapping[str, Any] | None = None) -> dict[str, Any]:
    profile_memory = task_profile.get("memory") if isinstance(task_profile, Mapping) else None
    if isinstance(profile_memory, Mapping):
        profile_memory = dict(profile_memory)
        if "activation_checkpointing" in profile_memory and isinstance(profile_memory["activation_checkpointing"], bool):
            profile_memory["activation_checkpointing"] = {
                "enabled": bool(profile_memory["activation_checkpointing"]),
                "policy": "selective",
            }
    raw = _merge_dicts(profile_memory, _mapping(_get(cfg, "step3.memory"), "step3.memory"))
    ckpt = _mapping(raw.get("activation_checkpointing"), "step3.memory.activation_checkpointing")
    policy = str(ckpt.get("policy") or "selective").strip()
    if policy != "selective":
        raise OneControlConfigError("step3.memory.activation_checkpointing.policy must be selective.")
    modules = [str(x).strip() for x in (ckpt.get("modules") or []) if str(x).strip()]
    profile_policy = str(raw.get("profile_buffer_policy") or "gpu_resident").strip()
    if profile_policy not in ("gpu_resident", "cpu_pinned_batch_gather"):
        raise OneControlConfigError("step3.memory.profile_buffer_policy must be gpu_resident or cpu_pinned_batch_gather.")
    phase_profiler = _mapping(raw.get("phase_profiler"), "step3.memory.phase_profiler")
    allocator_candidates = _mapping(raw.get("allocator_candidates"), "step3.memory.allocator_candidates")
    return {
        "activation_checkpointing": {
            "enabled": _bool(ckpt.get("enabled", False)),
            "policy": policy,
            "modules": modules,
        },
        "profile_buffer_policy": profile_policy,
        "phase_profiler": {
            "enabled": _bool(phase_profiler.get("enabled", True)),
            "sample_interval_steps": _positive_int(
                phase_profiler.get("sample_interval_steps", 50),
                "step3.memory.phase_profiler.sample_interval_steps",
            ),
            "phases": list(phase_profiler.get("phases") or MEMORY_PHASES),
            "required_fields": list(MEMORY_REQUIRED_FIELDS) if "MEMORY_REQUIRED_FIELDS" in globals() else [],
            "empty_cache_policy": str(phase_profiler.get("empty_cache_policy") or "phase_boundary_only"),
        },
        "allocator_candidates": {
            "expandable_segments": _bool(allocator_candidates.get("expandable_segments", False)),
            "max_split_size_mb": allocator_candidates.get("max_split_size_mb"),
            "garbage_collection_threshold": allocator_candidates.get("garbage_collection_threshold"),
            "formal_default": False,
            "runtime_verified": False,
        },
    }


def _resolve_step3_timing_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    raw = _mapping(_get(cfg, "step3.timing"), "step3.timing")
    fields = raw.get("fields") or []
    if not isinstance(fields, list) or not all(isinstance(x, str) for x in fields):
        raise OneControlConfigError("step3.timing.fields must be a string list")
    required = set(TIMING_REQUIRED_FIELDS)
    merged_fields = list(dict.fromkeys(list(fields) + list(TIMING_REQUIRED_FIELDS)))
    return {
        "enabled": _bool(raw.get("enabled", True)),
        "startup_steady_state_split": _bool(raw.get("startup_steady_state_split", True)),
        "sample_interval_steps": _positive_int(raw.get("sample_interval_steps", 50), "step3.timing.sample_interval_steps"),
        "unknown_ratio_warn_threshold": _rcr_float(raw.get("unknown_ratio_warn_threshold", 0.05), "step3.timing.unknown_ratio_warn_threshold", min_value=0.0),
        "unknown_ratio_block_threshold": _rcr_float(raw.get("unknown_ratio_block_threshold", 0.50), "step3.timing.unknown_ratio_block_threshold", min_value=0.0),
        "fields": merged_fields,
        "required_fields": sorted(required),
        "rank_timing_required": True,
    }


def _resolve_step3_performance_candidates_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    raw = _mapping(_get(cfg, "step3.performance_candidates"), "step3.performance_candidates")
    if not raw:
        return default_a100_candidate_matrix()
    matrix = default_a100_candidate_matrix()
    for key, value in raw.items():
        if key in matrix and isinstance(matrix.get(key), Mapping) and isinstance(value, Mapping):
            matrix[key] = _merge_dicts(matrix[key], value)
        else:
            matrix[str(key)] = deepcopy(value)
    batch_ladder = _mapping(matrix.get("batch_ladder"), "step3.performance_candidates.batch_ladder")
    selected_candidate = str(matrix.get("selected_candidate") or "").strip()
    for name, item in batch_ladder.items():
        if not isinstance(item, Mapping):
            raise OneControlConfigError(f"step3.performance_candidates.batch_ladder.{name} must be a mapping")
        formal_allowed = _bool(item.get("formal_allowed", False))
        probe_only = _bool(item.get("probe_only", True))
        if formal_allowed and str(name) != selected_candidate:
            raise OneControlConfigError(f"batch ladder candidate {name} must not be formal_allowed before runtime evidence.")
        if str(name) == selected_candidate:
            if not formal_allowed or probe_only:
                raise OneControlConfigError(
                    f"selected formal batch ladder candidate {name} must be formal_allowed=true and probe_only=false."
                )
            continue
        if not probe_only:
            raise OneControlConfigError(f"batch ladder candidate {name} must remain probe_only.")
    matrix["schema_version"] = STEP3_PERFORMANCE_CANDIDATE_SCHEMA_VERSION
    matrix["formal_default_unchanged"] = False
    return matrix


def _resolve_step3_cache_policy_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    raw = _mapping(_get(cfg, "step3.cache"), "step3.cache")
    schema = str(raw.get("tokenizer_schema_version") or "").strip()
    if schema != "odcr_step3_tokenizer_cache/2":
        raise OneControlConfigError("step3.cache.tokenizer_schema_version must be odcr_step3_tokenizer_cache/2")
    return {
        "tokenizer_schema_version": schema,
        "formal_cache_namespace": str(raw.get("formal_cache_namespace") or "cache/step3/tokenizer"),
    }


def _training_row(stage: str, train: Mapping[str, Any], task: Mapping[str, Any], *, eval_batch_size: int | None) -> dict[str, Any]:
    _reject_retired_accum_keys(train, f"{stage}.train")
    backend = train.get("backend", {})
    if backend is not None and not isinstance(backend, Mapping):
        raise OneControlConfigError(f"{stage}.train.backend must be a mapping")
    if stage == "step5":
        _reject_step5_retired_controls(train)
    row = dict(backend or {})
    public_keys = {"batch_size", "per_gpu_batch_size", "backend", "mode", "label_max_length"}
    for key, value in train.items():
        if key not in public_keys:
            row[key] = deepcopy(value)
    if stage == "step5":
        if "label_max_length" not in train:
            raise OneControlConfigError("step5.train.label_max_length must be configured; train_label_max_length is retired for Step5.")
        train_label_length = int(train["label_max_length"])
    else:
        train_label_length = int(train.get("train_label_max_length", 64))
    row.update(
        {
            "train_batch_size": int(train["batch_size"]),
            "per_device_train_batch_size": int(train["per_gpu_batch_size"]),
            "per_gpu_batch_size": int(train["per_gpu_batch_size"]),
            "global_batch_size": int(train["batch_size"]),
            "batch_semantics_version": NO_ACCUM_BATCH_SEMANTICS_VERSION,
            "batch_formula": "global_batch_size = per_gpu_batch_size * ddp_world_size",
            "grad_accum_removed": True,
            "train_label_max_length": train_label_length,
            "lr": float(train.get("lr", task.get("lr", 1e-3))),
        }
    )
    if stage == "step3":
        max_epochs = _positive_int(train.get("max_epochs"), "step3.train.max_epochs")
        row["max_epochs"] = max_epochs
        row["epochs"] = max_epochs
        row["min_epochs"] = _positive_int(train.get("min_epochs"), "step3.train.min_epochs")
        row["early_stop_patience"] = _positive_int(
            train.get("early_stop_patience"),
            "step3.train.early_stop_patience",
        )
        row["validate_every_epochs"] = _positive_int(
            train.get("validate_every_epochs"),
            "step3.train.validate_every_epochs",
        )
        row["max_grad_norm"] = _rcr_float(train.get("max_grad_norm"), "step3.train.max_grad_norm", min_value=0.0)
        if row["max_grad_norm"] <= 0.0:
            raise OneControlConfigError("step3.train.max_grad_norm must be > 0")
    else:
        row["epochs"] = int(train["epochs"])
        row["coef"] = float(train.get("coef", task.get("coef", 0.5)))
    if eval_batch_size is not None:
        row["eval_batch_size"] = int(eval_batch_size)
    return row


def _train_precision_source(stage: str, train: Mapping[str, Any]) -> str | None:
    backend = train.get("backend", {})
    if isinstance(backend, Mapping) and "train_precision" in backend:
        return f"{stage}.train.backend.train_precision"
    if "train_precision" in train:
        return f"{stage}.train.train_precision"
    return None


def _resolve_train_precision(stage: str, train: Mapping[str, Any], row: Mapping[str, Any]) -> tuple[str, str]:
    source = _train_precision_source(stage, train)
    if "train_precision" not in row:
        if stage == "step3":
            raise OneControlConfigError(
                "step3.train.backend.train_precision must be configured in configs/odcr.yaml; "
                "Step3 precision may not come from runtime env, child argparse, or helper defaults."
            )
        value = "bf16"
        source = "resolver schema default"
    else:
        value = str(row["train_precision"]).strip().lower()
    if value not in TRAIN_PRECISION_CHOICES:
        raise OneControlConfigError(
            f"{stage}.train.backend.train_precision must be one of {TRAIN_PRECISION_CHOICES}, got {value!r}"
        )
    if stage == "step3" and value != "bf16":
        raise OneControlConfigError("Step3 v0 active precision must be bf16.")
    if not source:
        source = f"{stage}.train.backend.train_precision"
    return value, source


def _apply_step5_native_lora_row(row: dict[str, Any], step5_innovation_config: Mapping[str, Any]) -> None:
    ccv = _mapping(step5_innovation_config.get("ccv"), "step5.ccv")
    native_lora = _mapping(ccv.get("native_lora"), "step5.ccv.native_lora")
    lci = _mapping(step5_innovation_config.get("lci"), "step5.lci")
    fca = _mapping(step5_innovation_config.get("fca"), "step5.fca")
    row["step5_lci_weight"] = _rcr_float(lci.get("weight"), "step5.lci.weight", min_value=0.0)
    row["step5_fca_weight"] = _rcr_float(fca.get("weight"), "step5.fca.weight", min_value=0.0)
    enabled = _bool(native_lora.get("enabled", True))
    row["train_mode"] = "lora" if enabled else "full"
    row["lora_r"] = _positive_int(native_lora.get("r"), "step5.ccv.native_lora.r")
    row["lora_alpha"] = _rcr_float(native_lora.get("alpha"), "step5.ccv.native_lora.alpha", min_value=0.0)
    row["lora_dropout"] = _rcr_float(
        native_lora.get("dropout"),
        "step5.ccv.native_lora.dropout",
        min_value=0.0,
        max_value=1.0,
    )
    row["lora_target_modules"] = list(native_lora.get("target_modules", []) or [])
    row["lora_target_policy_id"] = str(native_lora.get("target_policy_id") or _STEP5_LORA_TARGET_POLICY_ID)
    row["deleted_legacy_modules"] = list(_STEP5_DELETED_LEGACY_MODULES)
    row["retired_combined_formal_enabled"] = False
    row["all_trainable_grad_required"] = True


def _resolve_step5_model_config(cfg: Mapping[str, Any], runtime_roots: Mapping[str, Any]) -> dict[str, Any]:
    raw = _mapping(_get(cfg, "step5.model"), "step5.model")
    out = {
        "emsize": _positive_int(raw.get("emsize"), "step5.model.emsize"),
        "nlayers": _positive_int(raw.get("nlayers"), "step5.model.nlayers"),
        "nhead": _positive_int(raw.get("nhead"), "step5.model.nhead"),
        "nhid": _positive_int(raw.get("nhid"), "step5.model.nhid"),
        "dropout": _rcr_float(raw.get("dropout"), "step5.model.dropout", min_value=0.0, max_value=1.0),
    }
    if int(out["emsize"]) != int(runtime_roots["embed_dim"]):
        raise OneControlConfigError(
            "step5.model.emsize must equal env.embed_dim; "
            f"got step5.model.emsize={out['emsize']} env.embed_dim={runtime_roots['embed_dim']}"
        )
    if int(out["emsize"]) % int(out["nhead"]) != 0:
        raise OneControlConfigError("step5.model.emsize must be divisible by step5.model.nhead")
    return out


def _resolve_step5_ddp_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    raw = _mapping(_get(cfg, "step5.ddp"), "step5.ddp")
    find_unused = _bool(raw.get("find_unused_parameters", False))
    static_graph = _bool(raw.get("static_graph", False))
    if find_unused and static_graph:
        raise OneControlConfigError(
            "step5.ddp.static_graph=true requires step5.ddp.find_unused_parameters=false"
        )
    preflight = str(raw.get("find_unused_false_preflight", "real_sample_plan_one_batch")).strip().lower()
    real_policies = ("real_sample_plan_one_batch", "real_batch_one_step")
    if preflight not in (*real_policies, "fail_fast"):
        raise OneControlConfigError(
            "step5.ddp.find_unused_false_preflight must be 'real_sample_plan_one_batch', "
            "'real_batch_one_step', or 'fail_fast'"
        )
    if not find_unused and preflight not in real_policies:
        raise OneControlConfigError(
            "step5.ddp.find_unused_parameters=false requires a real-data formal preflight "
            "(step5.ddp.find_unused_false_preflight=real_sample_plan_one_batch or real_batch_one_step)."
        )
    return {
        "ddp_find_unused_parameters": find_unused,
        "ddp_static_graph": static_graph,
        "ddp_find_unused_false_preflight": preflight,
        "formal_preflight_uses_real_data": preflight in real_policies,
    }


def _resolve_step5_export_loader_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    raw = _mapping(_get(cfg, "step5.export_loader"), "step5.export_loader")
    required = (
        "cache_enabled",
        "cache_namespace",
        "chunk_rows",
        "validate_sample_rows",
        "bounded_max_rows",
        "stale_policy",
    )
    missing = [key for key in required if key not in raw]
    if missing:
        raise OneControlConfigError(
            "step5.export_loader missing required One-Control keys: " + ", ".join(missing)
        )
    cache_namespace = str(raw.get("cache_namespace") or "").strip().strip("/")
    if not cache_namespace or any(part in ("", ".", "..") for part in cache_namespace.split("/")):
        raise OneControlConfigError("step5.export_loader.cache_namespace must be a safe cache/<producer> path segment")
    stale_policy = str(raw.get("stale_policy") or "").strip().lower()
    if stale_policy not in ("rebuild", "fail_fast"):
        raise OneControlConfigError("step5.export_loader.stale_policy must be 'rebuild' or 'fail_fast'")
    return {
        "cache_enabled": _bool(raw.get("cache_enabled")),
        "cache_namespace": cache_namespace,
        "chunk_rows": _positive_int(raw.get("chunk_rows"), "step5.export_loader.chunk_rows"),
        "validate_sample_rows": _positive_int(
            raw.get("validate_sample_rows"),
            "step5.export_loader.validate_sample_rows",
        ),
        "bounded_max_rows": _positive_int(raw.get("bounded_max_rows"), "step5.export_loader.bounded_max_rows"),
        "stale_policy": stale_policy,
    }


def _resolve_step5_data_pipeline_config(
    cfg: Mapping[str, Any],
    *,
    ddp_world_size: int,
) -> dict[str, Any]:
    raw = _mapping(_get(cfg, "step5.data_pipeline"), "step5.data_pipeline")
    required = (
        "sample_plan_enabled",
        "token_cache_enabled",
        "bounded_token_cache_enabled",
        "dataloader_queue_size",
        "workers_per_rank_candidates",
        "prefetch_factor_candidates",
        "max_parallel_cpu",
        "reserved_cpu",
        "cpu_budget_guard",
        "pipeline_timing_enabled",
        "gpu_util_sampling_enabled",
    )
    missing = [key for key in required if key not in raw]
    if missing:
        raise OneControlConfigError("step5.data_pipeline missing required One-Control keys: " + ", ".join(missing))
    _reject_unknown_keys(raw, set(required), "step5.data_pipeline")

    def _int_list(key: str) -> list[int]:
        values = raw.get(key)
        if not isinstance(values, list) or not values:
            raise OneControlConfigError(f"step5.data_pipeline.{key} must be a non-empty list")
        out: list[int] = []
        seen: set[int] = set()
        for idx, value in enumerate(values):
            item = _positive_int(value, f"step5.data_pipeline.{key}[{idx}]")
            if item in seen:
                raise OneControlConfigError(f"step5.data_pipeline.{key} contains duplicate value {item}")
            seen.add(item)
            out.append(item)
        return out

    workers = _int_list("workers_per_rank_candidates")
    prefetch = _int_list("prefetch_factor_candidates")
    max_cpu = _positive_int(raw.get("max_parallel_cpu"), "step5.data_pipeline.max_parallel_cpu")
    reserved = _rcr_int(raw.get("reserved_cpu"), "step5.data_pipeline.reserved_cpu", min_value=0)
    guard = _bool(raw.get("cpu_budget_guard"))
    formulas = []
    for worker in workers:
        active = int(worker) * int(ddp_world_size) + int(reserved)
        ok = active <= int(max_cpu)
        formulas.append(
            {
                "workers_per_rank": int(worker),
                "ddp_world_size": int(ddp_world_size),
                "reserved_cpu": int(reserved),
                "max_parallel_cpu": int(max_cpu),
                "active_processes": int(active),
                "ok": bool(ok),
                "formula": f"{int(worker)} * {int(ddp_world_size)} + {int(reserved)} <= {int(max_cpu)}",
            }
        )
        if guard and not ok:
            raise OneControlConfigError(
                "step5.data_pipeline workers_per_rank_candidates exceed CPU budget: "
                f"{int(worker)} * {int(ddp_world_size)} + {int(reserved)} > {int(max_cpu)}"
            )
    return {
        "sample_plan_enabled": _bool(raw.get("sample_plan_enabled")),
        "token_cache_enabled": _bool(raw.get("token_cache_enabled")),
        "bounded_token_cache_enabled": _bool(raw.get("bounded_token_cache_enabled")),
        "dataloader_queue_size": _positive_int(raw.get("dataloader_queue_size"), "step5.data_pipeline.dataloader_queue_size"),
        "workers_per_rank_candidates": workers,
        "prefetch_factor_candidates": prefetch,
        "max_parallel_cpu": int(max_cpu),
        "reserved_cpu": int(reserved),
        "cpu_budget_guard": bool(guard),
        "cpu_budget_formulas": formulas,
        "pipeline_timing_enabled": _bool(raw.get("pipeline_timing_enabled")),
        "gpu_util_sampling_enabled": _bool(raw.get("gpu_util_sampling_enabled")),
    }


def _resolve_step5_e4_bounded_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    raw = _mapping(_get(cfg, "step5.e4_bounded"), "step5.e4_bounded")
    required = (
        "enabled",
        "evidence_level",
        "namespace_root",
        "max_runtime_seconds",
        "max_samples_guard",
        "oom_policy",
        "formal_namespace_policy",
        "batch_candidates",
        "dataloader_candidates",
        "row_candidates",
        "long_window_candidates",
    )
    missing = [key for key in required if key not in raw]
    if missing:
        raise OneControlConfigError("step5.e4_bounded missing required One-Control keys: " + ", ".join(missing))
    evidence_level = str(raw.get("evidence_level") or "").strip()
    if evidence_level != "E4_gpu_shard_forward_bounded_formal_entry_with_validation":
        raise OneControlConfigError(
            "step5.e4_bounded.evidence_level must be E4_gpu_shard_forward_bounded_formal_entry_with_validation"
        )
    namespace_root = str(raw.get("namespace_root") or "").strip().strip("/")
    if namespace_root not in {"AI_analysis", "runs/step5/task2/bounded_validation", "test_artifacts"}:
        raise OneControlConfigError(
            "step5.e4_bounded.namespace_root must be AI_analysis, runs/step5/task2/bounded_validation, or test_artifacts"
        )
    oom_policy = str(raw.get("oom_policy") or "").strip().lower()
    if oom_policy not in {"reject_candidate"}:
        raise OneControlConfigError("step5.e4_bounded.oom_policy must be reject_candidate")
    formal_namespace_policy = str(raw.get("formal_namespace_policy") or "").strip().lower()
    if formal_namespace_policy != "forbid":
        raise OneControlConfigError("step5.e4_bounded.formal_namespace_policy must be forbid")

    def _candidate_list(name: str, keys: tuple[str, ...]) -> list[dict[str, Any]]:
        items = raw.get(name)
        if not isinstance(items, list) or not items:
            raise OneControlConfigError(f"step5.e4_bounded.{name} must be a non-empty list")
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for idx, item in enumerate(items):
            if not isinstance(item, Mapping):
                raise OneControlConfigError(f"step5.e4_bounded.{name}[{idx}] must be a mapping")
            cid = str(item.get("id") or "").strip()
            if not cid:
                raise OneControlConfigError(f"step5.e4_bounded.{name}[{idx}].id must be non-empty")
            if cid in seen:
                raise OneControlConfigError(f"duplicate step5.e4_bounded.{name} id: {cid}")
            seen.add(cid)
            row: dict[str, Any] = {"id": cid}
            for key in keys:
                row[key] = _positive_int(item.get(key), f"step5.e4_bounded.{name}.{cid}.{key}")
            out.append(row)
        return out

    batches = _candidate_list("batch_candidates", ("per_gpu_batch_size", "global_batch_size"))
    for item in batches:
        if int(item["global_batch_size"]) != int(item["per_gpu_batch_size"]) * 2:
            raise OneControlConfigError(
                "step5.e4_bounded batch candidate global_batch_size must equal per_gpu_batch_size * ddp_world_size(2)"
            )
    dataloaders = _candidate_list("dataloader_candidates", ("workers_per_rank", "prefetch_factor"))
    rows = _candidate_list("row_candidates", ("bounded_rows", "chunk_rows"))
    long_windows = _candidate_list("long_window_candidates", ("bounded_rows", "min_steps"))
    max_samples_guard = _positive_int(raw.get("max_samples_guard"), "step5.e4_bounded.max_samples_guard")
    for item in rows:
        if int(item["bounded_rows"]) > max_samples_guard:
            raise OneControlConfigError(
                "step5.e4_bounded row candidate bounded_rows exceeds step5.e4_bounded.max_samples_guard"
            )
    for item in long_windows:
        if int(item["bounded_rows"]) > max_samples_guard:
            raise OneControlConfigError(
                "step5.e4_bounded long_window candidate bounded_rows exceeds step5.e4_bounded.max_samples_guard"
            )
    return {
        "enabled": _bool(raw.get("enabled")),
        "evidence_level": evidence_level,
        "namespace_root": namespace_root,
        "max_runtime_seconds": _positive_int(raw.get("max_runtime_seconds"), "step5.e4_bounded.max_runtime_seconds"),
        "max_samples_guard": max_samples_guard,
        "oom_policy": oom_policy,
        "formal_namespace_policy": formal_namespace_policy,
        "batch_candidates": batches,
        "dataloader_candidates": dataloaders,
        "row_candidates": rows,
        "long_window_candidates": long_windows,
    }


def _resolve_step5_eval_config(
    cfg: Mapping[str, Any],
    *,
    ddp_world_size: int,
    train_per_gpu_batch_size: int,
) -> dict[str, Any]:
    raw = _mapping(_get(cfg, "step5.eval"), "step5.eval")
    _reject_unknown_keys(
        raw,
        {
            "valid_per_gpu_batch_size",
            "valid_batch_size",
            "valid_forward_micro_batch_size",
            "test_per_gpu_batch_size",
            "test_forward_micro_batch_size",
            "metric_accumulation",
            "validation_memory_policy",
            "validation_mode",
            "formal_entry_E4_validation_required",
            "old_eval_batch_2048_retired",
            "valid_loss_components",
        },
        "step5.eval",
    )
    valid_per_gpu = _rcr_int(raw.get("valid_per_gpu_batch_size"), "step5.eval.valid_per_gpu_batch_size", min_value=1)
    valid_global = _rcr_int(raw.get("valid_batch_size"), "step5.eval.valid_batch_size", min_value=1)
    valid_forward = _rcr_int(raw.get("valid_forward_micro_batch_size"), "step5.eval.valid_forward_micro_batch_size", min_value=1)
    test_per_gpu = _rcr_int(raw.get("test_per_gpu_batch_size"), "step5.eval.test_per_gpu_batch_size", min_value=1)
    test_forward = _rcr_int(raw.get("test_forward_micro_batch_size"), "step5.eval.test_forward_micro_batch_size", min_value=1)
    if valid_global != valid_per_gpu * ddp_world_size:
        raise OneControlConfigError("step5.eval.valid_batch_size must equal valid_per_gpu_batch_size * ddp_world_size")
    if valid_forward > valid_per_gpu:
        raise OneControlConfigError("step5.eval.valid_forward_micro_batch_size must be <= valid_per_gpu_batch_size")
    if test_forward > test_per_gpu:
        raise OneControlConfigError("step5.eval.test_forward_micro_batch_size must be <= test_per_gpu_batch_size")
    if str(raw.get("validation_memory_policy") or "") != "microbatch_accumulate":
        raise OneControlConfigError("step5.eval.validation_memory_policy must be microbatch_accumulate")
    if str(raw.get("validation_mode") or "") != "explanation_only":
        raise OneControlConfigError("step5.eval.validation_mode must be explanation_only")
    components = _mapping(raw.get("valid_loss_components"), "step5.eval.valid_loss_components")
    _reject_unknown_keys(components, {"explanation"}, "step5.eval.valid_loss_components")
    explanation_components = list(components.get("explanation") or [])
    if "scorer_rating_mse" in explanation_components:
        raise OneControlConfigError("Step5 explanation validation must not include rating MSE")
    return {
        "valid_per_gpu_batch_size": valid_per_gpu,
        "valid_batch_size": valid_global,
        "valid_global_batch_size": valid_global,
        "valid_forward_micro_batch_size": valid_forward,
        "test_per_gpu_batch_size": test_per_gpu,
        "test_forward_micro_batch_size": test_forward,
        "validation_microbatch_accumulation": _bool(raw.get("metric_accumulation")),
        "validation_memory_policy": "microbatch_accumulate",
        "step5_validation_mode": "explanation_only",
        "formal_entry_E4_validation_required": _bool(raw.get("formal_entry_E4_validation_required")),
        "old_eval_batch_2048_retired": _bool(raw.get("old_eval_batch_2048_retired")),
        "valid_loss_components": {"explanation": explanation_components},
        "valid_loss_components_json": json_dumps({"explanation": explanation_components}),
    }


def _resolve_step5_valid_loss_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    raw = _mapping(_get(cfg, "step5.valid_loss"), "step5.valid_loss")
    _reject_unknown_keys(raw, {"label_max_length"}, "step5.valid_loss")
    label_max = _rcr_int(raw.get("label_max_length"), "step5.valid_loss.label_max_length", min_value=8)
    if label_max > 512:
        raise OneControlConfigError("step5.valid_loss.label_max_length must be <= 512")
    return {
        "schema_version": "odcr_step5_valid_loss_config/1",
        "label_max_length": int(label_max),
    }


def _resolve_step5_final_eval_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    raw = _mapping(_get(cfg, "step5.final_eval"), "step5.final_eval")
    _reject_unknown_keys(
        raw,
        {
            "prediction_max_length",
            "reference_max_length",
            "official_profile",
            "metric_input_builder",
            "metrics_implementation",
            "test_once",
        },
        "step5.final_eval",
    )
    pred_max = _rcr_int(raw.get("prediction_max_length"), "step5.final_eval.prediction_max_length", min_value=1)
    ref_max = _rcr_int(raw.get("reference_max_length"), "step5.final_eval.reference_max_length", min_value=1)
    if pred_max != 25 or ref_max != 25:
        raise OneControlConfigError("Step5 official final eval requires prediction/reference max length 25")
    profile = str(raw.get("official_profile") or "").strip()
    if profile != "paper_greedy_25":
        raise OneControlConfigError("step5.final_eval.official_profile must be paper_greedy_25")
    if str(raw.get("metric_input_builder") or "") != "build_paper_metric_inputs":
        raise OneControlConfigError("step5.final_eval.metric_input_builder must be build_paper_metric_inputs")
    if str(raw.get("metrics_implementation") or "") != "official_paper_metrics":
        raise OneControlConfigError("step5.final_eval.metrics_implementation must be official_paper_metrics")
    return {
        "schema_version": "odcr_step5_final_eval_config/1",
        "prediction_max_length": int(pred_max),
        "reference_max_length": int(ref_max),
        "official_profile": profile,
        "metric_input_builder": "build_paper_metric_inputs",
        "metrics_implementation": "official_paper_metrics",
        "test_once": _bool(raw.get("test_once", True)),
    }


def _step5_official_eval_batch_layout(
    step5_eval_config: Mapping[str, Any],
    *,
    eval_split: str,
    ddp_world_size: int,
) -> tuple[int, int]:
    split = str(eval_split or "valid").strip().lower()
    if split == "test":
        per_gpu = _rcr_int(
            step5_eval_config.get("test_per_gpu_batch_size"),
            "step5.eval.test_per_gpu_batch_size",
            min_value=1,
        )
        return per_gpu * int(ddp_world_size), per_gpu
    per_gpu = _rcr_int(
        step5_eval_config.get("valid_per_gpu_batch_size"),
        "step5.eval.valid_per_gpu_batch_size",
        min_value=1,
    )
    global_batch = _rcr_int(
        step5_eval_config.get("valid_global_batch_size", step5_eval_config.get("valid_batch_size")),
        "step5.eval.valid_batch_size",
        min_value=1,
    )
    return global_batch, per_gpu


def _resolve_step5_prompt_templates_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    raw = _mapping(_get(cfg, "step5.prompt_templates"), "step5.prompt_templates")
    _reject_unknown_keys(
        raw,
        {
            "schema_version",
            "allowed_template_count",
            "train_policy",
            "valid_test_policy",
            "input_formatting_only",
            "explanation_only",
            "active_templates",
        },
        "step5.prompt_templates",
    )
    if str(raw.get("schema_version") or "") != "odcr_step5_prompt_template_registry/1":
        raise OneControlConfigError("step5.prompt_templates.schema_version must be odcr_step5_prompt_template_registry/1")
    if _rcr_int(raw.get("allowed_template_count"), "step5.prompt_templates.allowed_template_count", min_value=1) != 3:
        raise OneControlConfigError("step5.prompt_templates.allowed_template_count must be 3")
    if str(raw.get("train_policy") or "") != "controlled_canonical_deterministic":
        raise OneControlConfigError("step5.prompt_templates.train_policy must be controlled_canonical_deterministic")
    if str(raw.get("valid_test_policy") or "") != "fixed_canonical":
        raise OneControlConfigError("step5.prompt_templates.valid_test_policy must be fixed_canonical")
    if _bool(raw.get("input_formatting_only")) is not True or _bool(raw.get("explanation_only")) is not True:
        raise OneControlConfigError("step5.prompt_templates must be formatting-only and explanation-only")
    templates = [str(item) for item in (raw.get("active_templates") or [])]
    required = {
        "Step5_target_anchor_explainer_v1",
        "Step5_aux_gold_explainer_v1",
        "Step5_aux_cf_explainer_v1",
    }
    if set(templates) != required:
        raise OneControlConfigError("step5.prompt_templates.active_templates must name the three Step5 explanation templates")
    return {
        "schema_version": "odcr_step5_prompt_template_registry/1",
        "allowed_template_count": 3,
        "train_policy": "controlled_canonical_deterministic",
        "valid_test_policy": "fixed_canonical",
        "input_formatting_only": True,
        "explanation_only": True,
        "active_templates": templates,
    }

def _resolve_step5_effective_epoch_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    raw = _mapping(_get(cfg, "step5.effective_epoch"), "step5.effective_epoch")
    _reject_unknown_keys(
        raw,
        {
            "enabled",
            "max_effective_epochs",
            "early_stopping_patience",
            "retired_full_table_epochs",
            "retired_full_table_policy",
        },
        "step5.effective_epoch",
    )
    if str(raw.get("retired_full_table_policy") or "") != "fail_fast":
        raise OneControlConfigError("step5.effective_epoch.retired_full_table_policy must be fail_fast")
    return {
        "enabled": _bool(raw.get("enabled")),
        "max_effective_epochs": _rcr_int(raw.get("max_effective_epochs"), "step5.effective_epoch.max_effective_epochs", min_value=1),
        "early_stopping_patience": _rcr_int(raw.get("early_stopping_patience"), "step5.effective_epoch.early_stopping_patience", min_value=0),
        "retired_full_table_epochs": _rcr_int(raw.get("retired_full_table_epochs"), "step5.effective_epoch.retired_full_table_epochs", min_value=1),
        "retired_full_table_policy": "fail_fast",
    }


def _resolve_step5_batch_candidates_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    raw = _mapping(_get(cfg, "step5.batch_candidates"), "step5.batch_candidates")
    _reject_unknown_keys(raw, {"ddp_world_size", "fsdp_zero_policy", "selected_default", "candidates"}, "step5.batch_candidates")
    world_size = _rcr_int(raw.get("ddp_world_size"), "step5.batch_candidates.ddp_world_size", min_value=1)
    if str(raw.get("fsdp_zero_policy") or "") != "not_introduced":
        raise OneControlConfigError("step5.batch_candidates.fsdp_zero_policy must be not_introduced")
    candidates_raw = raw.get("candidates")
    if not isinstance(candidates_raw, list) or not candidates_raw:
        raise OneControlConfigError("step5.batch_candidates.candidates must be a non-empty list")
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for idx, item in enumerate(candidates_raw):
        if not isinstance(item, Mapping):
            raise OneControlConfigError(f"step5.batch_candidates.candidates[{idx}] must be a mapping")
        _reject_unknown_keys(item, {"id", "per_gpu_batch_size", "global_batch_size", "role"}, f"step5.batch_candidates.candidates[{idx}]")
        cid = str(item.get("id") or "").strip()
        if cid in seen or not cid:
            raise OneControlConfigError(f"step5.batch_candidates.candidates[{idx}].id must be unique and non-empty")
        seen.add(cid)
        per_gpu = _rcr_int(item.get("per_gpu_batch_size"), f"step5.batch_candidates.{cid}.per_gpu_batch_size", min_value=1)
        global_batch = _rcr_int(item.get("global_batch_size"), f"step5.batch_candidates.{cid}.global_batch_size", min_value=1)
        if global_batch != per_gpu * world_size:
            raise OneControlConfigError(
                f"step5.batch_candidates.{cid}.global_batch_size must equal per_gpu_batch_size * ddp_world_size"
            )
        out.append(
            {
                "id": cid,
                "per_gpu_batch_size": per_gpu,
                "global_batch_size": global_batch,
                "role": str(item.get("role") or ""),
            }
        )
    selected = str(raw.get("selected_default") or "").strip()
    if selected not in seen:
        raise OneControlConfigError("step5.batch_candidates.selected_default must name a candidate id")
    return {
        "ddp_world_size": world_size,
        "fsdp_zero_policy": "not_introduced",
        "selected_default": selected,
        "candidates": out,
    }


def _resolve_step5_tuning_config(
    cfg: Mapping[str, Any],
    batch_candidates_config: Mapping[str, Any],
) -> dict[str, Any]:
    raw = _mapping(_get(cfg, "step5.tuning"), "step5.tuning")
    _reject_unknown_keys(
        raw,
        {
            "enabled",
            "selected_tuning_candidate",
            "fallback_tuning_candidate",
            "effective_samples",
            "optimizer_steps",
            "batch_candidate",
            "fallback_batch_candidate",
            "selected_budget_candidate",
            "pilot_fraction_candidates",
            "lr_candidates",
            "warmup_fraction_candidates",
            "innovation_weight_candidates",
            "ratio_candidates",
            "cf_tier_mix_candidates",
            "gold_tier_mix_candidates",
            "search_strategy",
        },
        "step5.tuning",
    )
    batch_ids = {str(item.get("id")) for item in batch_candidates_config.get("candidates", []) if isinstance(item, Mapping)}
    batch_candidate = str(raw.get("batch_candidate") or "").strip()
    fallback_batch_candidate = str(raw.get("fallback_batch_candidate") or "").strip()
    for key, value in (("batch_candidate", batch_candidate), ("fallback_batch_candidate", fallback_batch_candidate)):
        if value not in batch_ids:
            raise OneControlConfigError(f"step5.tuning.{key} must name a resolved step5.batch_candidates id")
    selected_budget = str(raw.get("selected_budget_candidate") or "medium").strip()
    if selected_budget not in {"small", "medium", "full", "large"}:
        raise OneControlConfigError("step5.tuning.selected_budget_candidate must be small, medium, full, or large")
    selected_tuning_candidate = str(raw.get("selected_tuning_candidate") or "").strip()
    fallback_tuning_candidate = str(raw.get("fallback_tuning_candidate") or "").strip()
    if not selected_tuning_candidate or not fallback_tuning_candidate:
        raise OneControlConfigError("step5.tuning selected and fallback candidates must be configured")

    def _single_positive_int_map(key: str) -> dict[str, int]:
        values = _mapping(raw.get(key), f"step5.tuning.{key}")
        _reject_unknown_keys(values, {"explanation"}, f"step5.tuning.{key}")
        return {"explanation": _rcr_int(values.get("explanation"), f"step5.tuning.{key}.explanation", min_value=1)}

    def _fraction_list(key: str) -> list[float]:
        values = raw.get(key)
        if not isinstance(values, list) or not values:
            raise OneControlConfigError(f"step5.tuning.{key} must be a non-empty list")
        return [_rcr_float(value, f"step5.tuning.{key}", min_value=0.0, max_value=1.0) for value in values]

    def _ratio_candidates() -> list[dict[str, Any]]:
        values = _mapping(_mapping(raw.get("ratio_candidates"), "step5.tuning.ratio_candidates").get("explanation"), "step5.tuning.ratio_candidates.explanation")
        out: list[dict[str, Any]] = []
        for cid, item in values.items():
            if not isinstance(item, Mapping):
                raise OneControlConfigError(f"step5.tuning.ratio_candidates.explanation.{cid} must be a mapping")
            _reject_unknown_keys(item, {"target_gold", "aux_gold", "cf"}, f"step5.tuning.ratio_candidates.explanation.{cid}")
            tg = _rcr_float(item.get("target_gold"), f"step5.tuning.ratio_candidates.explanation.{cid}.target_gold", min_value=0.30, max_value=0.45)
            ag = _rcr_float(item.get("aux_gold"), f"step5.tuning.ratio_candidates.explanation.{cid}.aux_gold", min_value=0.10, max_value=0.25)
            cf = _rcr_float(item.get("cf"), f"step5.tuning.ratio_candidates.explanation.{cid}.cf", min_value=0.34, max_value=0.50)
            if abs((tg + ag + cf) - 1.0) > 1e-6:
                raise OneControlConfigError(f"step5.tuning.ratio_candidates.explanation.{cid} ratios must sum to 1.0")
            out.append({"id": str(cid), "target_gold": tg, "aux_gold": ag, "cf": cf})
        if not out:
            raise OneControlConfigError("step5.tuning.ratio_candidates.explanation must not be empty")
        return out

    def _mix_candidates(group_key: str, fields: set[str], subkey: str = "explanation") -> list[dict[str, Any]]:
        values = _mapping(_mapping(raw.get(group_key), f"step5.tuning.{group_key}").get(subkey), f"step5.tuning.{group_key}.{subkey}")
        rows: list[dict[str, Any]] = []
        for cid, item in values.items():
            if not isinstance(item, Mapping):
                raise OneControlConfigError(f"step5.tuning.{group_key}.{subkey}.{cid} must be a mapping")
            _reject_unknown_keys(item, fields, f"step5.tuning.{group_key}.{subkey}.{cid}")
            row = {"id": str(cid)}
            total = 0.0
            for field in sorted(fields):
                value = _rcr_float(item.get(field), f"step5.tuning.{group_key}.{subkey}.{cid}.{field}", min_value=0.0, max_value=1.0)
                row[field] = value
                total += value
            if abs(total - 1.0) > 1e-6:
                raise OneControlConfigError(f"step5.tuning.{group_key}.{subkey}.{cid} values must sum to 1.0")
            rows.append(row)
        if not rows:
            raise OneControlConfigError(f"step5.tuning.{group_key}.{subkey} must not be empty")
        return rows

    innovation = raw.get("innovation_weight_candidates")
    if not isinstance(innovation, list) or not innovation:
        raise OneControlConfigError("step5.tuning.innovation_weight_candidates must be a non-empty list")
    innovation_out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for idx, item in enumerate(innovation):
        if not isinstance(item, Mapping):
            raise OneControlConfigError(f"step5.tuning.innovation_weight_candidates[{idx}] must be a mapping")
        _reject_unknown_keys(item, {"id", "fca", "explainer_loss_weight", "ccv_numeric_control_weight"}, f"step5.tuning.innovation_weight_candidates[{idx}]")
        cid = str(item.get("id") or "").strip()
        if not cid or cid in seen:
            raise OneControlConfigError("step5.tuning.innovation_weight_candidates ids must be unique and non-empty")
        seen.add(cid)
        innovation_out.append({
            "id": cid,
            "fca": _rcr_float(item.get("fca"), f"step5.tuning.innovation_weight_candidates.{cid}.fca", min_value=0.0),
            "explainer_loss_weight": _rcr_float(item.get("explainer_loss_weight"), f"step5.tuning.innovation_weight_candidates.{cid}.explainer_loss_weight", min_value=0.0),
            "ccv_numeric_control_weight": _rcr_float(item.get("ccv_numeric_control_weight"), f"step5.tuning.innovation_weight_candidates.{cid}.ccv_numeric_control_weight", min_value=0.0),
        })

    strategy = _mapping(raw.get("search_strategy"), "step5.tuning.search_strategy")
    _reject_unknown_keys(strategy, {"stage_a_fraction", "stage_b_fraction", "extended_primary_fraction", "extended_backup_fraction"}, "step5.tuning.search_strategy")
    lr_values = raw.get("lr_candidates")
    if not isinstance(lr_values, list) or not lr_values:
        raise OneControlConfigError("step5.tuning.lr_candidates must be a non-empty list")
    return {
        "enabled": _bool(raw.get("enabled")),
        "selected_tuning_candidate": selected_tuning_candidate,
        "fallback_tuning_candidate": fallback_tuning_candidate,
        "effective_samples": _single_positive_int_map("effective_samples"),
        "optimizer_steps": _single_positive_int_map("optimizer_steps"),
        "batch_candidate": batch_candidate,
        "fallback_batch_candidate": fallback_batch_candidate,
        "selected_budget_candidate": selected_budget,
        "pilot_fraction_candidates": _fraction_list("pilot_fraction_candidates"),
        "lr_candidates": [_rcr_float(value, "step5.tuning.lr_candidates", min_value=0.0) for value in lr_values],
        "warmup_fraction_candidates": _fraction_list("warmup_fraction_candidates"),
        "innovation_weight_candidates": innovation_out,
        "ratio_candidates": {"explanation": _ratio_candidates()},
        "cf_tier_mix_candidates": {"explanation": _mix_candidates("cf_tier_mix_candidates", {"high", "medium", "low_weighted"})},
        "gold_tier_mix_candidates": {
            "target_gold": _mix_candidates("gold_tier_mix_candidates", {"high", "medium"}, "target_gold"),
            "aux_gold": _mix_candidates("gold_tier_mix_candidates", {"high", "medium"}, "aux_gold"),
        },
        "search_strategy": {
            "stage_a_fraction": _rcr_float(strategy.get("stage_a_fraction"), "step5.tuning.search_strategy.stage_a_fraction", min_value=0.0, max_value=1.0),
            "stage_b_fraction": _rcr_float(strategy.get("stage_b_fraction"), "step5.tuning.search_strategy.stage_b_fraction", min_value=0.0, max_value=1.0),
            "extended_primary_fraction": _rcr_float(strategy.get("extended_primary_fraction"), "step5.tuning.search_strategy.extended_primary_fraction", min_value=0.0, max_value=1.0),
            "extended_backup_fraction": _rcr_float(strategy.get("extended_backup_fraction"), "step5.tuning.search_strategy.extended_backup_fraction", min_value=0.0, max_value=1.0),
        },
    }


def _step5_candidate_tokens(candidate: str) -> dict[str, str]:
    tokens = {str(part).strip() for part in str(candidate or "").split("+") if str(part).strip()}
    out: dict[str, str] = {}
    for prefixes, key, label in (
        (("STEP5_RATIO_",), "ratio", "STEP5_RATIO_"),
        (("STEP5_CF_MIX_",), "cf_mix", "STEP5_CF_MIX_"),
        (("TG_MIX_",), "target_gold_mix", "TG_MIX_"),
        (("AG_MIX_",), "aux_gold_mix", "AG_MIX_"),
        (("LR_",), "lr", "LR_"),
        (("W",), "weight", "W"),
    ):
        matches = sorted(token for token in tokens if any(token.startswith(prefix) for prefix in prefixes))
        if len(matches) != 1:
            raise OneControlConfigError(
                f"step5.tuning.selected_tuning_candidate must contain exactly one {label} token; got {candidate!r}"
            )
        out[key] = matches[0]
    return out


def _candidate_row_by_id(rows: Any, wanted: str, label: str) -> Mapping[str, Any]:
    if not isinstance(rows, list):
        raise OneControlConfigError(f"{label} must be a resolved candidate list")
    for row in rows:
        if isinstance(row, Mapping) and str(row.get("id")) == wanted:
            return row
    raise OneControlConfigError(f"selected Step5 candidate id {wanted!r} missing from {label}")


def _assert_close_mapping(actual: Mapping[str, Any], expected: Mapping[str, Any], *, fields: tuple[str, ...], label: str) -> None:
    for field in fields:
        a = _rcr_float(actual.get(field), f"{label}.{field}", min_value=0.0)
        e = _rcr_float(expected.get(field), f"{label}.selected.{field}", min_value=0.0)
        if abs(a - e) > 1e-6:
            raise OneControlConfigError(f"{label}.{field}={a} does not match selected Step5 tuning candidate value {e}")


def _assert_step5_selected_candidate_consistency(
    *,
    row: Mapping[str, Any],
    sampler_config: Mapping[str, Any],
    tuning_config: Mapping[str, Any],
    innovation_config: Mapping[str, Any],
) -> None:
    tokens = _step5_candidate_tokens(str(tuning_config.get("selected_tuning_candidate") or ""))
    if tokens.get("ratio") != _STEP5_FORMAL_RATIO_ID:
        raise OneControlConfigError(f"Step5 explanation formal mainline requires {_STEP5_FORMAL_RATIO_ID}")
    if tokens.get("cf_mix") != _STEP5_FORMAL_CF_MIX_ID:
        raise OneControlConfigError(f"Step5 explanation formal mainline requires {_STEP5_FORMAL_CF_MIX_ID}")
    explanation_cfg = _mapping(sampler_config.get("explanation"), "step5.sampler.explanation")
    ratio = _candidate_row_by_id(
        (tuning_config.get("ratio_candidates") or {}).get("explanation"),
        tokens["ratio"],
        "step5.tuning.ratio_candidates.explanation",
    )
    _assert_close_mapping(
        {
            "target_gold": explanation_cfg.get("target_gold_ratio"),
            "aux_gold": explanation_cfg.get("aux_gold_ratio"),
            "cf": explanation_cfg.get("cf_ratio"),
        },
        ratio,
        fields=("target_gold", "aux_gold", "cf"),
        label="step5.sampler.explanation",
    )
    cf_mix = _candidate_row_by_id(
        (tuning_config.get("cf_tier_mix_candidates") or {}).get("explanation"),
        tokens["cf_mix"],
        "step5.tuning.cf_tier_mix_candidates.explanation",
    )
    _assert_close_mapping(
        _mapping(explanation_cfg.get("cf_tier_mix"), "step5.sampler.explanation.cf_tier_mix"),
        cf_mix,
        fields=("high", "medium", "low_weighted"),
        label="step5.sampler.explanation.cf_tier_mix",
    )
    selected_lr = float(str(tokens["lr"])[len("LR_") :])
    if abs(float(row.get("lr")) - selected_lr) > 1e-12:
        raise OneControlConfigError(f"step5.train.lr={row.get('lr')} does not match selected candidate {tokens['lr']}")
    weights = _candidate_row_by_id(tuning_config.get("innovation_weight_candidates"), tokens["weight"], "step5.tuning.innovation_weight_candidates")
    fca = _mapping(innovation_config.get("fca"), "step5.fca")
    ccv = _mapping(innovation_config.get("ccv"), "step5.ccv")
    checks = (
        ("step5.fca.weight", fca.get("weight"), weights.get("fca")),
        ("step5.train.explainer_loss_weight", row.get("explainer_loss_weight"), weights.get("explainer_loss_weight")),
        ("step5.ccv.numeric_control_weight", ccv.get("numeric_control_weight"), weights.get("ccv_numeric_control_weight")),
    )
    for label, actual, expected in checks:
        if abs(float(actual) - float(expected)) > 1e-12:
            raise OneControlConfigError(f"{label}={actual} does not match selected candidate {tokens['weight']} value {expected}")


def _step5_formal_active_candidate_payload(
    *,
    sampler_config: Mapping[str, Any],
    tuning_config: Mapping[str, Any],
    row: Mapping[str, Any],
) -> dict[str, Any]:
    tokens = _step5_candidate_tokens(str(tuning_config.get("selected_tuning_candidate") or ""))
    explanation_cfg = _mapping(sampler_config.get("explanation"), "step5.sampler.explanation")
    explanation_mix = dict(_mapping(explanation_cfg.get("cf_tier_mix"), "step5.sampler.explanation.cf_tier_mix"))
    low_weighted_disabled = float(explanation_mix.get("low_weighted") or 0.0) == 0.0
    return {
        "schema_version": "odcr_step5_formal_active_candidate/2",
        "mode": "explanation_only",
        "selected_tuning_candidate": str(tuning_config.get("selected_tuning_candidate") or ""),
        "candidate_parts": {
            "ratio_id": tokens["ratio"],
            "cf_mix_id": tokens["cf_mix"],
            "target_gold_mix_id": tokens["target_gold_mix"],
            "aux_gold_mix_id": tokens["aux_gold_mix"],
            "lr_id": tokens["lr"],
            "weights_id": tokens["weight"],
        },
        "explanation_cf_mix_id": tokens["cf_mix"],
        "explanation_cf_mix": explanation_mix,
        "lr": float(row.get("lr") or 0.0),
        "weights_id": tokens["weight"],
        "batch_candidate": str(tuning_config.get("batch_candidate") or ""),
        "effective_samples": dict(tuning_config.get("effective_samples") or {}),
        "optimizer_steps": dict(tuning_config.get("optimizer_steps") or {}),
        "low_weighted_policy": "disabled_for_mainline" if low_weighted_disabled else "active",
        "full_audit_default_forbidden": bool(sampler_config.get("full_audit_default_allowed") is False),
        "old_dedicated_default_forbidden": bool(sampler_config.get("legacy_gold_heavy_exports_allowed") is False),
        "step4_sampling_contract_role": "pool_lineage_only",
        "active_sampler_source": "configs/odcr.yaml:step5.sampler + configs/odcr.yaml:step5.tuning.selected_tuning_candidate",
        "ai_analysis_runtime_config_source": "evidence_only_not_runtime_config",
    }

def _resolve_step5_memory_truth_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    raw = _mapping(_get(cfg, "step5.memory_truth"), "step5.memory_truth")
    required = (
        "schema_version",
        "reserved_diagnostic_only",
        "reject_on_reserved",
        "reject_on_oom",
        "reject_on_allocated_ratio",
        "allocated_warning_ratio",
        "nvidia_smi_instability_ratio",
        "memory_creep_delta_gb",
        "short_window_steps",
        "long_window_steps",
        "track_param_memory",
        "track_optimizer_memory",
        "track_sequence_lengths",
        "gradient_checkpointing_enabled",
        "gradient_checkpointing_reentrant_policy",
        "disable_use_cache_during_training",
        "empty_cache_for_measurement_only",
    )
    missing = [key for key in required if key not in raw]
    if missing:
        raise OneControlConfigError("step5.memory_truth missing required One-Control keys: " + ", ".join(missing))
    schema_version = str(raw.get("schema_version") or "").strip()
    if schema_version != "odcr_step5_memory_truth/1":
        raise OneControlConfigError("step5.memory_truth.schema_version must be odcr_step5_memory_truth/1")
    if not _bool(raw.get("reserved_diagnostic_only")):
        raise OneControlConfigError("step5.memory_truth.reserved_diagnostic_only must be true")
    if _bool(raw.get("reject_on_reserved")):
        raise OneControlConfigError("step5.memory_truth.reject_on_reserved must be false")
    reentrant_policy = str(raw.get("gradient_checkpointing_reentrant_policy") or "").strip().lower()
    if reentrant_policy not in {"non_reentrant", "reentrant", "disabled"}:
        raise OneControlConfigError(
            "step5.memory_truth.gradient_checkpointing_reentrant_policy must be "
            "non_reentrant, reentrant, or disabled"
        )
    if not _bool(raw.get("gradient_checkpointing_enabled")) and reentrant_policy != "disabled":
        raise OneControlConfigError(
            "step5.memory_truth.gradient_checkpointing_reentrant_policy must be disabled "
            "when gradient_checkpointing_enabled=false"
        )
    if _bool(raw.get("gradient_checkpointing_enabled")) and reentrant_policy == "disabled":
        raise OneControlConfigError(
            "step5.memory_truth.gradient_checkpointing_reentrant_policy=disabled requires "
            "gradient_checkpointing_enabled=false"
        )

    def _ratio_or_none(key: str) -> float | None:
        value = raw.get(key)
        if value is None:
            return None
        ratio = _rcr_float(value, f"step5.memory_truth.{key}", min_value=0.0, max_value=1.0)
        if ratio <= 0.0:
            return None
        return ratio

    memory_creep_delta = _rcr_float(raw.get("memory_creep_delta_gb"), "step5.memory_truth.memory_creep_delta_gb", min_value=0.0)
    if memory_creep_delta <= 0.0:
        raise OneControlConfigError("step5.memory_truth.memory_creep_delta_gb must be > 0")
    return {
        "schema_version": schema_version,
        "reserved_diagnostic_only": True,
        "reject_on_reserved": False,
        "reject_on_oom": _bool(raw.get("reject_on_oom")),
        "reject_on_allocated_ratio": _ratio_or_none("reject_on_allocated_ratio"),
        "allocated_warning_ratio": _ratio_or_none("allocated_warning_ratio"),
        "nvidia_smi_instability_ratio": _ratio_or_none("nvidia_smi_instability_ratio"),
        "memory_creep_delta_gb": memory_creep_delta,
        "short_window_steps": _positive_int(raw.get("short_window_steps"), "step5.memory_truth.short_window_steps"),
        "long_window_steps": _positive_int(raw.get("long_window_steps"), "step5.memory_truth.long_window_steps"),
        "track_param_memory": _bool(raw.get("track_param_memory")),
        "track_optimizer_memory": _bool(raw.get("track_optimizer_memory")),
        "track_sequence_lengths": _bool(raw.get("track_sequence_lengths")),
        "gradient_checkpointing_enabled": _bool(raw.get("gradient_checkpointing_enabled")),
        "gradient_checkpointing_reentrant_policy": reentrant_policy,
        "disable_use_cache_during_training": _bool(raw.get("disable_use_cache_during_training")),
        "empty_cache_for_measurement_only": _bool(raw.get("empty_cache_for_measurement_only")),
    }


def _lineage_for_step5(step4_run: str) -> str:
    if step4_run == "latest":
        return "latest"
    parts = run_naming.parse_run_id(step4_run).split("_")
    return parts[0]


def _lineage_for_eval(step5_run: str) -> tuple[str, str]:
    if step5_run == "latest":
        return "latest", "latest"
    step4_run = run_naming.step4_slug_from_step5_slug(step5_run)
    parts = run_naming.step5_numeric_slug(step5_run).split("_")
    return parts[0], step4_run


def resolve_config(
    *,
    config_path: str | Path,
    command: str,
    task_id: int | None,
    set_overrides: Iterable[str],
    dry_run: bool,
    run_id: str | None = None,
    from_step3: str | None = None,
    from_step4: str | None = None,
    from_step5: str | None = None,
    step5_head: str | None = None,
    checkpoint: str | None = None,
    eval_profile: str | None = None,
    mode: str | None = None,
) -> tuple[ResolvedConfig, list[SourceRecord], dict[str, Any]]:
    _reject_retired_accum_env()
    base = load_yaml_config(config_path)
    cfg, cli_sources = apply_cli_sets(base, set_overrides)
    _validate_config_shape(cfg)
    project = cfg["project"]
    if not isinstance(project, Mapping):
        raise OneControlConfigError("project must be a mapping")
    repo_root = _REPO_ROOT
    runtime_roots = _resolve_global_runtime_roots(cfg, repo_root)
    tid = int(task_id or project.get("default_task") or 2)
    task = _task_row(cfg, tid)
    auxiliary = str(task["source"])
    target = str(task["target"])
    scenario = str(task.get("scenario") or "legacy_scenario")
    direction = str(task.get("direction") or "unspecified")

    stage_for_train = "step5" if command == "eval" else command
    step5_head_norm = run_naming.parse_step5_head(step5_head if command == "step5" else "explanation")
    stage_cfg = _get(cfg, stage_for_train, {})
    if not isinstance(stage_cfg, Mapping):
        stage_cfg = {}
    train_base = stage_cfg.get("train", {})
    if train_base is None:
        train_base = {}
    if not isinstance(train_base, Mapping):
        raise OneControlConfigError(f"{stage_for_train}.train must be a mapping")
    if stage_for_train == "step3":
        base_backend = _mapping(train_base.get("backend"), "step3.train.backend")
        if "train_precision" not in base_backend:
            raise OneControlConfigError(
                "step3.train.backend.train_precision must be configured in configs/odcr.yaml; "
                "task profiles may isolate values but must not replace the Step3 control-plane field."
            )
    step3_scenario_profile: dict[str, Any] = {}
    step3_task_profile_key = ""
    step3_task_profile_raw: dict[str, Any] = {}
    if stage_for_train == "step3":
        scenario, direction, step3_scenario_profile = _resolve_step3_scenario_profile(cfg, task, task_id=tid)
        step3_task_profile_key, step3_task_profile_raw = _select_step3_task_profile(cfg, task, task_id=tid)
    train = _merge_dicts(
        train_base,
        step3_scenario_profile.get("train") if stage_for_train == "step3" else None,
        step3_task_profile_raw.get("train") if stage_for_train == "step3" else None,
        _stage_task_override(stage_cfg, tid),
    )
    _apply_train_cli_overrides(cfg=cfg, cli_sources=cli_sources, stage=stage_for_train, train=train)
    step5_lifecycle_config = _resolve_step5_lifecycle_config(cfg) if stage_for_train == "step5" else {}
    step5_lifecycle_phase = ""
    step5_allow_embedded_final_eval = False
    step5_train_only_resolved = False
    if command == "step5":
        requested_phase = str(mode or "").strip()
        if not requested_phase:
            requested_phase = str(step5_lifecycle_config.get("formal_default_phase") or "train_only")
        if requested_phase not in {"train_only", "full"}:
            raise OneControlConfigError("Step5 lifecycle mode must be train_only or full")
        step5_lifecycle_phase = requested_phase
        step5_train_only_resolved = requested_phase == "train_only"
        step5_allow_embedded_final_eval = (
            requested_phase == "full"
            and bool(step5_lifecycle_config.get("embedded_final_eval_default"))
            and bool(step5_lifecycle_config.get("allow_embedded_final_eval_diagnostic"))
        )
        if requested_phase == "full" and not step5_allow_embedded_final_eval:
            raise OneControlConfigError(
                "Step5 embedded final eval is retired from the formal default lifecycle; "
                "use train_only plus a fresh eval/recovery handoff, or explicitly enable the diagnostic "
                "step5.lifecycle embedded-final-eval controls."
            )
    elif stage_for_train == "step5":
        step5_lifecycle_phase = "eval_only"

    eval_profile_name = ""
    eval_profile_obj: dict[str, Any] = {}
    need_eval = command in ("step4", "step5", "eval")
    if need_eval:
        eval_profile_name, eval_profile_obj = _resolve_eval_profile(cfg, eval_profile)
    eval_cfg_top = _get(cfg, "eval", {})
    eval_split = str((eval_cfg_top.get("split") if isinstance(eval_cfg_top, Mapping) else None) or "valid").strip().lower()
    if eval_split not in {"valid", "test"}:
        raise OneControlConfigError("eval.split must be either valid or test")
    hw_name_from_profile = eval_profile_obj.get("hardware") if eval_profile_obj else None
    hw_name, hw = _active_hardware(cfg, str(hw_name_from_profile) if hw_name_from_profile else None)
    ddp_world_size = _positive_int(hw.get("ddp_world_size", 1), "hardware.ddp_world_size")
    num_proc = _positive_int(hw.get("num_proc", 1), "hardware.num_proc")
    batch_size, per_gpu, eff = _validate_train_batch(stage_for_train, train, ddp_world_size)
    step3_optimizer_config: dict[str, Any] = {}
    step3_backend_config: dict[str, Any] = {}
    step3_tokenizer_config: dict[str, Any] = {}
    step3_evidence_config: dict[str, Any] = {}
    step3_scheduler_config: dict[str, Any] = {}
    step3_eval_config: dict[str, Any] = {}
    step3_task_profile_config: dict[str, Any] = {}
    step3_backup_profiles_config: dict[str, Any] = {}
    step3_exploration_profiles_config: dict[str, Any] = {}
    step3_worker_profiles_config: dict[str, Any] = {}
    step3_prefetcher_config: dict[str, Any] = {}
    step3_checkpoint_policy_config: dict[str, Any] = {}
    step3_quality_gate_config: dict[str, Any] = {}
    step3_grad_finite_config: dict[str, Any] = {}
    step3_diagnostic_eval_config: dict[str, Any] = {}
    step3_cross_rank_gather_config: dict[str, Any] = {}
    step3_memory_config: dict[str, Any] = {}
    step3_timing_config: dict[str, Any] = {}
    step3_performance_candidates_config: dict[str, Any] = {}
    step3_cache_policy_config: dict[str, Any] = {}
    step3_objective_drift_config: dict[str, Any] = {}
    step3_recovery_config: dict[str, Any] = {}
    step3_phase_loss_schedule_config: dict[str, Any] = {}
    step3_conflict_aware_config: dict[str, Any] = {}
    step3_loss_gradient_conflict_probe_config: dict[str, Any] = {}
    step3_adapter_gating_config: dict[str, Any] = {}
    step3_paper_candidate_selection_config: dict[str, Any] = {}
    step3_checkpoint_averaging_config: dict[str, Any] = {}
    step3_batch_candidate_role = ""
    if stage_for_train == "step3":
        step3_optimizer_config = _resolve_step3_optimizer_config(cfg, step3_task_profile_raw)
        step3_backend_config = _resolve_step3_backend_config(train)
        step3_tokenizer_config, step3_evidence_config = _resolve_step3_tokenizer_evidence_config(
            cfg,
            _merge_dicts(step3_scenario_profile, step3_task_profile_raw),
        )
        step3_scheduler_config = _resolve_step3_scheduler_config(cfg, step3_task_profile_raw)
        step3_eval_config = _resolve_step3_eval_config(train, stage_cfg, ddp_world_size)
        step3_worker_profiles_config = _resolve_step3_worker_profiles_config(
            cfg,
            ddp_world_size=ddp_world_size,
            max_parallel_cpu=_positive_int(hw.get("max_parallel_cpu", 1), "hardware.max_parallel_cpu"),
        )
        step3_prefetcher_config = _resolve_step3_prefetcher_config(cfg)
        step3_checkpoint_policy_config = _resolve_step3_checkpoint_policy_config(cfg)
        step3_quality_gate_config = _resolve_step3_quality_gate_config(cfg)
        step3_grad_finite_config = _resolve_step3_grad_finite_config(cfg)
        step3_diagnostic_eval_config = _resolve_step3_diagnostic_eval_config(cfg)
        step3_cross_rank_gather_config = _resolve_step3_cross_rank_gather_config(cfg, step3_task_profile_raw)
        step3_memory_config = _resolve_step3_memory_config(cfg, step3_task_profile_raw)
        step3_backup_profiles_config = _resolve_step3_backup_profiles_config(cfg, ddp_world_size)
        step3_exploration_profiles_config = _resolve_step3_exploration_profiles_config(cfg, ddp_world_size)
        step3_timing_config = _resolve_step3_timing_config(cfg)
        step3_performance_candidates_config = _resolve_step3_performance_candidates_config(cfg)
        step3_cache_policy_config = _resolve_step3_cache_policy_config(cfg)
        step3_objective_drift_config = _resolve_step3_objective_drift_config(cfg)
        step3_recovery_config = _resolve_step3_recovery_config(cfg)
        step3_phase_loss_schedule_config = _resolve_step3_phase_loss_schedule_config(cfg)
        step3_conflict_aware_config = _resolve_step3_conflict_aware_config(cfg)
        step3_loss_gradient_conflict_probe_config = _resolve_step3_loss_gradient_conflict_probe_config(cfg)
        step3_adapter_gating_config = _resolve_step3_adapter_gating_config(cfg)
        step3_paper_candidate_selection_config = _resolve_step3_paper_candidate_selection_config(cfg)
        step3_checkpoint_averaging_config = _resolve_step3_checkpoint_averaging_config(cfg)
        step3_task_profile_config = _resolve_step3_task_profile_config(
            key=step3_task_profile_key,
            raw=step3_task_profile_raw,
            task_id=tid,
            task=task,
            ddp_world_size=ddp_world_size,
            train=train,
            tokenizer_config=step3_tokenizer_config,
            evidence_config=step3_evidence_config,
            scheduler_config=step3_scheduler_config,
            gather_config=step3_cross_rank_gather_config,
            memory_config=step3_memory_config,
        )
        step3_batch_candidate_role = _resolve_step3_batch_candidate_role(
            train,
            active_task_profile_id=str(step3_task_profile_config["profile_id"]),
            active_candidate=str(step3_task_profile_config.get("candidate") or ""),
        )
    if stage_for_train != "step3":
        step3_tokenizer_config, step3_evidence_config = _resolve_step3_tokenizer_evidence_config(cfg, {})

    eval_batch_size: int | None = None
    eval_per_gpu: int | None = None
    if need_eval:
        eval_batch_size = _positive_int(eval_profile_obj.get("eval_batch_size"), "eval.profile.eval_batch_size")
        eval_per_gpu = _validate_eval_batch(eval_batch_size, ddp_world_size)

    need_decode = needs_decode_layer(command, step5_train_only=step5_train_only_resolved)
    decode_name = str(eval_profile_obj.get("decode")) if eval_profile_obj.get("decode") else None
    decode_id, decode = _resolve_decode(cfg, decode_name, need_decode=need_decode)
    need_rerank = command == "eval" and bool(eval_profile_obj.get("rerank"))
    rerank_name = str(eval_profile_obj.get("rerank")) if eval_profile_obj.get("rerank") else None
    rerank_id, rerank = _resolve_rerank(cfg, rerank_name, need_rerank=need_rerank)
    if stage_for_train == "step5" and command == "eval":
        if eval_profile_name != "paper_greedy_25":
            raise OneControlConfigError(
                "Step5 official eval requires eval profile paper_greedy_25; "
                f"{eval_profile_name!r} is diagnostic_only and cannot write official paper metrics."
            )
        if not bool(eval_profile_obj.get("official", False)):
            raise OneControlConfigError("eval.profiles.paper_greedy_25 must declare official: true")
        if str(decode_id) != "paper_greedy_25":
            raise OneControlConfigError("Step5 official eval requires eval.decode.paper_greedy_25")
        if bool(eval_profile_obj.get("rerank")):
            raise OneControlConfigError("Step5 official paper_greedy_25 eval forbids rerank")
        if str(decode.get("decode_strategy", "")).strip().lower() != "greedy":
            raise OneControlConfigError("Step5 official paper_greedy_25 decode_strategy must be greedy")
        if int(decode.get("max_explanation_length", 0)) != 25 or int(decode.get("hard_max_len", 0)) != 25:
            raise OneControlConfigError("Step5 official paper_greedy_25 max/hard length must be 25")
        if abs(float(decode.get("repetition_penalty", 1.0)) - 1.0) > 1e-12:
            raise OneControlConfigError("Step5 official paper_greedy_25 repetition_penalty must be 1.0")

    iteration_id = "v1"
    run_name: str | None = None
    from_run: str | None = None
    step4_run: str | None = None
    step5_run: str | None = None
    step3_checkpoint_dir: str | None = None
    eval_run_dir: str | None = None
    model_path: str | None = None
    upstream_resolution_payload: dict[str, Any] = {}
    active_stage_status_payload: dict[str, Any] = {}

    if command == "step3":
        run_name = _alloc_run(repo_root, tid, "step3", run_id or "auto", dry_run=dry_run)
        run_root = _stage_root(repo_root, tid, "step3", run_name)
        try:
            active_stage_status_payload = resolve_latest(
                repo_root=repo_root,
                stage="step3",
                task=tid,
                repair=True,
            ).to_payload(repo_root)
        except UpstreamResolutionError as exc:
            active_stage_status_payload = {
                "schema_version": "odcr_upstream_resolution/1",
                "producer_stage": "step3",
                "consumer_stage": "step4",
                "task": tid,
                "available": False,
                "error": str(exc),
            }
    elif command == "step4":
        src = from_step3 or "latest"
        try:
            upstream = resolve_upstream(
                repo_root=repo_root,
                stage="step3",
                task=tid,
                from_run=None if src == "latest" else src,
                mode="formal",
                consumer_stage="step4",
                repair=True,
            )
        except UpstreamResolutionError as exc:
            raise OneControlConfigError(str(exc)) from exc
        from_run = upstream.run_id
        upstream_resolution_payload = upstream.to_payload(repo_root)
        step4_run = _alloc_step4_run(repo_root, tid, from_run, run_id or "auto", dry_run=dry_run)
        run_root = _stage_root(repo_root, tid, "step4", step4_run)
        step3_checkpoint_dir = str(_stage_root(repo_root, tid, "step3", from_run))
    elif command == "step5":
        if step5_lifecycle_phase == "eval_only":
            raise OneControlConfigError("Step5 eval-only rating handoff is retired; use eval after an explanation handoff.")
        else:
            src = from_step4 or "latest"
            try:
                upstream = resolve_upstream(
                    repo_root=repo_root,
                    stage="step4",
                    task=tid,
                    from_run=None if src == "latest" else src,
                    mode="formal",
                    consumer_stage="step5",
                    repair=True,
                )
            except UpstreamResolutionError as exc:
                raise OneControlConfigError(str(exc)) from exc
            step4_run = upstream.run_id
            upstream_resolution_payload = upstream.to_payload(repo_root)
            upstream_from_step3 = (upstream.stage_status.get("upstream") or {}).get("from_step3")
            from_run = run_naming.parse_run_id(str(upstream_from_step3)) if upstream_from_step3 else _lineage_for_step5(step4_run)
            step5_parent = path_layout.get_stage_task_root(repo_root, "step5", tid)
            if run_id and run_id not in ("", "auto"):
                try:
                    step5_run = run_naming.normalize_step5_run_id_for_step4(
                        str(run_id),
                        step4_run=step4_run,
                        head=step5_head_norm,
                    )
                except ValueError as exc:
                    raise OneControlConfigError(str(exc)) from exc
                if (step5_parent / step5_run).exists():
                    raise FileExistsError(f"已存在目录（禁止覆盖）: {step5_parent / step5_run}")
            else:
                if not dry_run:
                    step5_parent.mkdir(parents=True, exist_ok=True)
                step5_run = run_naming.allocate_step5_run_id(step5_parent, step4_run, head=step5_head_norm)
            run_root = _stage_root(repo_root, tid, "step5", step5_run)
            eval_run_dir = str((run_root / "post_train_eval").resolve())
    elif command == "eval":
        src = from_step5 or "latest"
        try:
            upstream = (
                resolve_latest(repo_root=repo_root, stage="step5", task=tid, repair=True)
                if src == "latest"
                else resolve_run(
                    repo_root=repo_root,
                    stage="step5",
                    task=tid,
                    run_id=run_naming.parse_stage_run_id("step5", str(src)),
                    repair=True,
                    requested_run=str(src),
                )
            )
        except UpstreamResolutionError as exc:
            raise OneControlConfigError(str(exc)) from exc
        status_payload = upstream.stage_status if isinstance(upstream.stage_status, Mapping) else {}
        final_status = str(status_payload.get("final_status") or "").strip().lower()
        artifacts_payload = status_payload.get("artifacts") if isinstance(status_payload.get("artifacts"), Mapping) else {}
        checkpoint_item = artifacts_payload.get("selected_checkpoint") if isinstance(artifacts_payload, Mapping) else {}
        checkpoint_path = str(checkpoint_item.get("path") or "").strip() if isinstance(checkpoint_item, Mapping) else ""
        if not checkpoint_path:
            checkpoint_path = str(status_payload.get("selected_checkpoint") or "").strip()
        if checkpoint_path:
            checkpoint_abs = (
                (repo_root / checkpoint_path).resolve()
                if not Path(checkpoint_path).is_absolute()
                else Path(checkpoint_path).expanduser().resolve()
            )
        else:
            checkpoint_abs = Path()
        if final_status not in {"completed_with_explanation_handoff", "train_completed_no_explanation_handoff"}:
            raise OneControlConfigError(
                f"Step5 run {upstream.run_id} is not eligible for official eval reclosure: final_status={final_status!r}"
            )
        if not checkpoint_path or not checkpoint_abs.is_file():
            raise OneControlConfigError(
                f"Step5 official eval reclosure requires selected checkpoint for run {upstream.run_id}; got {checkpoint_path!r}"
            )
        step5_run = upstream.run_id
        upstream_resolution_payload = upstream.to_payload(repo_root)
        upstream_resolution_payload["eligible_for_eval_reclosure"] = True
        upstream_resolution_payload["needs_explanation_handoff"] = final_status == "train_completed_no_explanation_handoff"
        upstream_resolution_payload["eval_reclosure_reason"] = "one_epoch_post_train_official_eval"
        upstream_payload = upstream.stage_status.get("upstream") if isinstance(upstream.stage_status, Mapping) else {}
        if isinstance(upstream_payload, Mapping):
            upstream_from_step3 = str(upstream_payload.get("from_step3") or "").strip()
            upstream_from_step4 = str(upstream_payload.get("from_step4") or "").strip()
        else:
            upstream_from_step3 = ""
            upstream_from_step4 = ""
        inferred_from_run, inferred_step4 = _lineage_for_eval(step5_run)
        from_run = run_naming.parse_run_id(upstream_from_step3) if upstream_from_step3 else inferred_from_run
        step4_run = run_naming.parse_run_id(upstream_from_step4) if upstream_from_step4 else inferred_step4
        run_root = _stage_root(repo_root, tid, "step5", step5_run)
        if need_rerank:
            eval_stage = "rerank"
            eval_run_id = _alloc_run(repo_root, tid, eval_stage, run_id or "auto", dry_run=dry_run)
            eval_run_dir = str(_stage_root(repo_root, tid, eval_stage, eval_run_id))
        else:
            eval_run_dir = str((run_root / "post_train_eval" / eval_split).resolve())
    else:
        raise OneControlConfigError(f"unsupported command for resolver: {command}")

    log_dir = str((Path(eval_run_dir) if command == "eval" and eval_run_dir else run_root) / "meta")
    manifest_dir = log_dir
    iteration_root = repo_root / "runs" / f"task{tid}"
    if not dry_run:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        stamp = Path(log_dir) / "resolved_config.generated_at"
        stamp.write_text(datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ\n"), encoding="utf-8")

    step4_rcr_config = _resolve_step4_rcr_config(cfg)
    step4_runtime_config = _resolve_step4_runtime_config(cfg)
    step4_step5_dedicated_exports_config = _resolve_step4_step5_dedicated_exports_config(cfg)
    step4_step5_pool_exports_config = _resolve_step4_step5_pool_exports_config(cfg)
    step4_gold_quality_config = _resolve_step4_gold_quality_config(cfg)
    step4_cf_tiers_config = _resolve_step4_cf_tiers_config(cfg)
    rating_source_config = (
        validate_rating_source(resolve_rating_source_config(_get(cfg, "rating_source"), repo_root=repo_root), repo_root=repo_root)
        if stage_for_train == "step5"
        else {}
    )
    step3_structured_losses_config = _resolve_step3_structured_losses_config(cfg)
    step3_ddp_config = _resolve_step3_ddp_config(cfg) if stage_for_train == "step3" else {}
    step3_loss_semantics_config = _resolve_step3_loss_semantics_config(cfg)
    step5_innovation_config = _resolve_step5_innovation_config(cfg)
    step5_task_decoupled_policy_config = (
        _resolve_step5_task_decoupled_policy_config(cfg) if stage_for_train == "step5" else {}
    )
    step5_model_factory_policy = {}
    if stage_for_train == "step5":
        step5_model_factory_policy = {
            "schema_version": "odcr_step5_model_factory_policy/1",
            "mode": "explanation_only",
            "head": "explanation",
            "explanation": {
                "factory": "build_step5_explanation_model",
                "branch": "explainer_rich",
                "uses_big_model": True,
                "uses_tokenizer": True,
                "uses_generation": True,
                "returns_word_dist": True,
                "computes_decoder_ce": True,
                "uses_aux_cf": True,
            },
            "active": {
                "factory": "build_step5_explanation_model",
                "uses_big_model": True,
                "uses_tokenizer": True,
                "uses_generation": True,
                "returns_word_dist": True,
                "trains_rating": False,
            },
        }
    step5_model_config = _resolve_step5_model_config(cfg, runtime_roots)
    step5_ddp_config = _resolve_step5_ddp_config(cfg) if stage_for_train == "step5" else {}
    step5_export_loader_config = _resolve_step5_export_loader_config(cfg) if stage_for_train == "step5" else {}
    step5_data_pipeline_config = _resolve_step5_data_pipeline_config(cfg, ddp_world_size=ddp_world_size) if stage_for_train == "step5" else {}
    step5_sampler_config = _resolve_step5_sampler_config(cfg) if stage_for_train == "step5" else {}
    if stage_for_train == "step5":
        step5_sampler_config["task_decoupled_policy"] = dict(step5_task_decoupled_policy_config)
    step5_prompt_templates_config = _resolve_step5_prompt_templates_config(cfg) if stage_for_train == "step5" else {}
    step5_effective_epoch_config = _resolve_step5_effective_epoch_config(cfg) if stage_for_train == "step5" else {}
    step5_batch_candidates_config = _resolve_step5_batch_candidates_config(cfg) if stage_for_train == "step5" else {}
    step5_tuning_config = (
        _resolve_step5_tuning_config(cfg, step5_batch_candidates_config)
        if stage_for_train == "step5"
        else {}
    )
    step5_e4_bounded_config = _resolve_step5_e4_bounded_config(cfg) if stage_for_train == "step5" else {}
    step5_memory_truth_config = _resolve_step5_memory_truth_config(cfg) if stage_for_train == "step5" else {}
    row = _training_row(stage_for_train, train, task, eval_batch_size=eval_batch_size if command in ("step5", "eval") else None)
    step5_eval_config = (
        _resolve_step5_eval_config(
            cfg,
            ddp_world_size=ddp_world_size,
            train_per_gpu_batch_size=int(row["per_gpu_batch_size"]),
        )
        if stage_for_train == "step5"
        else {}
    )
    step5_valid_loss_config = _resolve_step5_valid_loss_config(cfg) if stage_for_train == "step5" else {}
    step5_final_eval_config = _resolve_step5_final_eval_config(cfg) if stage_for_train == "step5" else {}
    if (
        stage_for_train == "step5"
        and command == "eval"
        and step5_eval_config
        and bool(step5_eval_config.get("old_eval_batch_2048_retired"))
    ):
        eval_batch_size, eval_per_gpu = _step5_official_eval_batch_layout(
            step5_eval_config,
            eval_split=eval_split,
            ddp_world_size=ddp_world_size,
        )
        row["eval_batch_size"] = int(eval_batch_size)
    train_precision, train_precision_source = _resolve_train_precision(stage_for_train, train, row)
    row["train_precision"] = train_precision
    row.setdefault("tokenizer_max_length", int(step3_tokenizer_config["max_length"]))
    row.setdefault("evidence_max_length", int(step3_evidence_config["max_evidence_length"]))
    step5_formal_active_candidate_config: dict[str, Any] = {}
    if stage_for_train == "step3":
        row.update(
            {
                "scenario": scenario,
                "direction": direction,
                "candidate": step3_batch_candidate_role,
                "optimizer": step3_optimizer_config,
                "optimizer_name": step3_optimizer_config["name"],
                "optimizer_betas": list(step3_optimizer_config["betas"]),
                "optimizer_eps": float(step3_optimizer_config["eps"]),
                "optimizer_dense_weight_decay": float(
                    step3_optimizer_config["param_groups"]["dense_weight_decay"]
                ),
                "optimizer_special_weight_decay": float(
                    step3_optimizer_config["param_groups"]["special_weight_decay"]
                ),
                "optimizer_no_decay": float(step3_optimizer_config["param_groups"]["no_decay"]),
                "optimizer_exclude_frozen_evidence_buffers": bool(
                    step3_optimizer_config["exclude_frozen_evidence_buffers"]
                ),
                "allow_tf32": bool(step3_backend_config["allow_tf32"]),
                "amp_autocast": bool(step3_backend_config["amp_autocast"]),
                "grad_scaler": bool(step3_backend_config["grad_scaler"]),
                "tokenizer_max_length": int(step3_tokenizer_config["max_length"]),
                "evidence_max_length": int(step3_evidence_config["max_evidence_length"]),
                "lr_scheduler": str(step3_scheduler_config["name"]),
                "lr_base_scheduler": str(step3_scheduler_config.get("base_scheduler", "warmup_cosine")),
                "lr_damping_enabled": bool(step3_scheduler_config.get("damping_enabled", False)),
                "warmup_ratio": float(step3_scheduler_config["warmup_ratio"]),
                "min_lr_ratio": float(step3_scheduler_config["min_lr_ratio"]),
                "effective_min_lr_policy": str(
                    step3_scheduler_config.get("validation_aware_lr_damping", {}).get(
                        "effective_min_lr_policy",
                        "base_floor",
                    )
                ),
                "validation_aware_lr_damping": bool(
                    step3_scheduler_config.get("validation_aware_lr_damping", {}).get("enabled", False)
                ),
                "step3_objective_drift_enabled": bool(step3_objective_drift_config.get("enabled", False)),
                "step3_recovery_enabled": bool(step3_recovery_config.get("enabled", False)),
                "step3_recovery_scheduler": str(step3_recovery_config.get("recovery_scheduler", "")),
                "step3_phase_loss_schedule_enabled": bool(step3_phase_loss_schedule_config.get("enabled", False)),
                "step3_conflict_aware_mode": str(step3_conflict_aware_config.get("mode", "off")),
                "step3_adapter_gating_enabled": bool(step3_adapter_gating_config.get("enabled", False)),
                "step3_paper_candidate_selection_enabled": bool(step3_paper_candidate_selection_config.get("enabled", False)),
                "step3_eval_protocol": str(step3_eval_config.get("protocol", MINIMAL_EVAL)),
                "step3_eval_split": str(step3_eval_config.get("split", "valid")),
                "step3_eval_batch_candidates": list(step3_eval_config.get("batch_candidates") or []),
                "eval_batch_size": int(step3_eval_config["valid_batch_size"]),
                "valid_batch_size": int(step3_eval_config["valid_batch_size"]),
                "valid_micro_batch_size": int(step3_eval_config["valid_micro_batch_size"]),
                "valid_batch_derive_from_train": bool(step3_eval_config["derive_from_train"]),
                "batch_semantics_version": NO_ACCUM_BATCH_SEMANTICS_VERSION,
                "step3_batch_semantics": NO_ACCUM_BATCH_SEMANTICS_VERSION,
                "step3_batch_formula": "global_batch_size = per_gpu_batch_size * ddp_world_size",
                "grad_accum_removed": True,
                "step3_batch_candidate_role": step3_batch_candidate_role,
                "task_profile_key": step3_task_profile_key,
                "task_profile_id": step3_task_profile_config["profile_id"],
                "active_profile": step3_task_profile_config["active_profile"],
                "profile_isolation_hash": step3_task_profile_config["profile_isolation_hash"],
                "profile_formal_allowed": bool(step3_task_profile_config["formal_allowed"]),
                "profile_probe_only": bool(step3_task_profile_config["probe_only"]),
                "prefetcher_enabled": bool(step3_prefetcher_config["enabled"]),
                "prefetcher_double_buffer": bool(step3_prefetcher_config["double_buffer"]),
                "checkpoint_policy": str(step3_checkpoint_policy_config["downstream_default_scope"]),
                "quality_gate_version": str(step3_quality_gate_config["schema_version"]),
                "grad_finite_enabled": bool(step3_grad_finite_config["enabled"]),
                "diagnostic_eval_protocol": "odcr_step3_diagnostic",
                "cross_rank_structured_gather_enabled": bool(step3_cross_rank_gather_config["enabled"]),
                "gather_mode": str(step3_cross_rank_gather_config["mode"]),
                "local_per_gpu_batch": int(per_gpu),
                "local_micro_batch_alias": int(per_gpu),
                "effective_structured_pool": int(per_gpu) * int(ddp_world_size),
                "effective_pool_expected": int(step3_task_profile_config["effective_structured_pool"]["effective_pool_expected"]),
                "gathered_tensor_names": list(step3_cross_rank_gather_config["allowed_tensors"]),
                "remote_tensors_detached": True,
            }
        )
    if stage_for_train == "step5":
        _apply_step5_native_lora_row(row, step5_innovation_config)
        train_precision, train_precision_source = _resolve_train_precision(stage_for_train, train, row)
        row["train_precision"] = train_precision
        row["step5_mode"] = "explanation_only"
        row["step5_head"] = "explanation"
        row["rating_source"] = dict(rating_source_config)
        row["head_specific_lora_allowlist_id"] = (
            f"{_STEP5_LORA_TARGET_POLICY_ID}:{row['step5_head']}"
        )
        row["head_specific_trainable_policy"] = (
            f"step5_head_specific_trainable_contract/1:{row['step5_head']}"
        )
        row["head_gated_loss_contract"] = {
            "schema_version": "odcr_step5_explanation_loss_contract/1",
            "head": row["step5_head"],
            "mode": "explanation_only",
            "rating_training": False,
            "active_losses": ["explainer_ce", "ccv", "fca"],
        }
        row["final_lora_target_modules"] = list(row.get("lora_target_modules") or [])
        row["forbidden_lora_targets"] = _step5_forbidden_lora_targets_from_model_config(
            step5_model_config.get("nlayers")
        )
        row.update(step5_eval_config)
        row["valid_loss_label_max_length"] = int(step5_valid_loss_config["label_max_length"])
        row["final_eval_prediction_max_length"] = int(step5_final_eval_config["prediction_max_length"])
        row["final_eval_reference_max_length"] = int(step5_final_eval_config["reference_max_length"])
        row["step5_final_eval"] = dict(step5_final_eval_config)
        row["step5_valid_loss"] = dict(step5_valid_loss_config)
        row.update(step5_model_config)
        row.update(step5_ddp_config)
        max_effective_epochs = int((step5_effective_epoch_config or {}).get("max_effective_epochs") or (step5_sampler_config.get("epochs") or {}).get("max_effective_epochs", 1))
        sampler_epochs = int((step5_sampler_config.get("epochs") or {}).get("max_effective_epochs", max_effective_epochs))
        if sampler_epochs != max_effective_epochs:
            raise OneControlConfigError("step5.effective_epoch.max_effective_epochs must match step5.sampler.epochs.max_effective_epochs")
        if int(row.get("epochs", 0)) > max_effective_epochs:
            raise OneControlConfigError(
                "step5.train.epochs must not exceed step5.sampler.epochs.max_effective_epochs; "
                "Step5 uses effective epoch budgets, not a 30-epoch full-table sweep."
            )
        row["effective_epoch_enabled"] = bool(step5_sampler_config.get("effective_epoch_enabled", True))
        row["effective_epoch_max"] = max_effective_epochs
        if "explainer_loss_weight" not in row:
            raise OneControlConfigError("step5.train.explainer_loss_weight must be configured in configs/odcr.yaml")
        row["explainer_loss_weight"] = _rcr_float(
            row["explainer_loss_weight"],
            "step5.train.explainer_loss_weight",
            min_value=0.0,
        )
        _assert_step5_selected_candidate_consistency(
            row=row,
            sampler_config=step5_sampler_config,
            tuning_config=step5_tuning_config,
            innovation_config=step5_innovation_config,
        )
        step5_formal_active_candidate_config = _step5_formal_active_candidate_payload(
            sampler_config=step5_sampler_config,
            tuning_config=step5_tuning_config,
            row=row,
        )
        row["selected_tuning_candidate"] = str(step5_tuning_config.get("selected_tuning_candidate") or "")
        row["fallback_tuning_candidate"] = str(step5_tuning_config.get("fallback_tuning_candidate") or "")
        row["step5_effective_samples"] = dict(step5_tuning_config.get("effective_samples") or {})
        row["step5_optimizer_steps"] = dict(step5_tuning_config.get("optimizer_steps") or {})
        row["step5_formal_active_candidate"] = dict(step5_formal_active_candidate_config)
        row["step5_explanation_cf_mix_id"] = str(step5_formal_active_candidate_config.get("explanation_cf_mix_id") or "")
        row["step5_lifecycle"] = dict(step5_lifecycle_config)
        row["step5_lifecycle_phase"] = step5_lifecycle_phase
        row["step5_train_only"] = bool(step5_train_only_resolved)
        row["step5_allow_embedded_final_eval"] = bool(step5_allow_embedded_final_eval)
        row["step5_checkpoint_load_policy"] = str(step5_lifecycle_config.get("checkpoint_load_policy") or "cpu_staged")
    if stage_for_train == "step3":
        row.update(step3_ddp_config)
    payload = {
        "schema_version": 3,
        "task_id": tid,
        "preset_name": stage_for_train,
        "training_row": row,
        "explainer_loss_weight": 0.0 if stage_for_train == "step3" else float(row.get("explainer_loss_weight", 0.0)),
        "auxiliary": auxiliary,
        "target": target,
        "scenario": scenario,
        "direction": direction,
        "task_profile_id": step3_task_profile_config.get("profile_id", "") if stage_for_train == "step3" else "",
        "task_profile_key": step3_task_profile_key if stage_for_train == "step3" else "",
        "profile_isolation_hash": step3_task_profile_config.get("profile_isolation_hash", "") if stage_for_train == "step3" else "",
        "runtime_roots": runtime_roots,
    }
    payload["step3_structured_losses"] = step3_structured_losses_config
    payload["step3_loss_semantics"] = step3_loss_semantics_config
    if stage_for_train == "step3":
        payload["step3_ddp"] = step3_ddp_config
        payload["step3_scenario_profile"] = step3_scenario_profile
        payload["step3_task_profile"] = step3_task_profile_config
        payload["step3_backup_profiles"] = step3_backup_profiles_config
        payload["step3_exploration_profiles"] = step3_exploration_profiles_config
        payload["step3_optimizer"] = step3_optimizer_config
        payload["step3_precision"] = step3_backend_config
        payload["step3_tokenizer"] = step3_tokenizer_config
        payload["step3_evidence"] = step3_evidence_config
        payload["step3_scheduler"] = step3_scheduler_config
        payload["step3_eval"] = step3_eval_config
        payload["step3_worker_profiles"] = step3_worker_profiles_config
        payload["step3_prefetcher"] = step3_prefetcher_config
        payload["step3_checkpoint_policy"] = step3_checkpoint_policy_config
        payload["step3_quality_gate"] = step3_quality_gate_config
        payload["step3_grad_finite"] = step3_grad_finite_config
        payload["step3_diagnostic_eval"] = step3_diagnostic_eval_config
        payload["step3_cross_rank_structured_gather"] = step3_cross_rank_gather_config
        payload["step3_memory"] = step3_memory_config
        payload["step3_timing"] = step3_timing_config
        payload["step3_performance_candidates"] = step3_performance_candidates_config
        payload["step3_cache_policy"] = step3_cache_policy_config
        payload["step3_objective_drift"] = step3_objective_drift_config
        payload["step3_recovery"] = step3_recovery_config
        payload["step3_phase_loss_schedule"] = step3_phase_loss_schedule_config
        payload["step3_conflict_aware"] = step3_conflict_aware_config
        payload["step3_loss_gradient_conflict_probe"] = step3_loss_gradient_conflict_probe_config
        payload["step3_adapter_gating"] = step3_adapter_gating_config
        payload["step3_paper_candidate_selection"] = step3_paper_candidate_selection_config
        payload["step3_checkpoint_averaging"] = step3_checkpoint_averaging_config
    payload["step4_rcr"] = step4_rcr_config
    payload["step4_runtime"] = step4_runtime_config
    payload["step4_step5_dedicated_exports"] = step4_step5_dedicated_exports_config
    payload["step4_step5_pool_exports"] = step4_step5_pool_exports_config
    payload["step4_gold_quality"] = step4_gold_quality_config
    payload["step4_cf_tiers"] = step4_cf_tiers_config
    payload["runtime_precision"] = {
        "train_precision": train_precision,
        "allow_tf32": bool(step3_backend_config.get("allow_tf32", False)) if stage_for_train == "step3" else None,
        "amp_autocast": bool(step3_backend_config.get("amp_autocast", True)) if stage_for_train == "step3" else None,
        "grad_scaler": bool(step3_backend_config.get("grad_scaler", False)) if stage_for_train == "step3" else None,
        "source": train_precision_source,
        "transport_env": "ODCR_RUNTIME_PRECISION_MODE",
    }
    if stage_for_train == "step5":
        payload["step5_mode"] = "explanation_only"
        payload["step5_head"] = "explanation"
        payload["rating_source"] = dict(rating_source_config)
        payload["selected_tuning_candidate"] = str(step5_tuning_config.get("selected_tuning_candidate") or "")
        payload["fallback_tuning_candidate"] = str(step5_tuning_config.get("fallback_tuning_candidate") or "")
        payload["step5_effective_samples"] = dict(step5_tuning_config.get("effective_samples") or {})
        payload["step5_optimizer_steps"] = dict(step5_tuning_config.get("optimizer_steps") or {})
        payload["step5_innovation"] = step5_innovation_config
        payload["step5_task_decoupled_policy"] = step5_task_decoupled_policy_config
        payload["step5_model_factory_policy"] = step5_model_factory_policy
        payload["step5_model"] = step5_model_config
        payload["lora_target_policy_id"] = row.get("lora_target_policy_id")
        payload["head_specific_lora_allowlist_id"] = row.get("head_specific_lora_allowlist_id")
        payload["final_lora_target_modules"] = list(row.get("final_lora_target_modules") or [])
        payload["forbidden_lora_targets"] = list(row.get("forbidden_lora_targets") or [])
        payload["deleted_legacy_modules"] = list(row.get("deleted_legacy_modules") or [])
        payload["retired_combined_formal_enabled"] = False
        payload["all_trainable_grad_required"] = True
        payload["head_specific_trainable_policy"] = row.get("head_specific_trainable_policy")
        payload["head_gated_loss_contract"] = row.get("head_gated_loss_contract")
        payload["step5_export_loader"] = step5_export_loader_config
        payload["step5_data_pipeline"] = step5_data_pipeline_config
        payload["step5_sampler"] = step5_sampler_config
        payload["step5_prompt_templates"] = step5_prompt_templates_config
        payload["step5_effective_epoch"] = step5_effective_epoch_config
        payload["step5_batch_candidates"] = step5_batch_candidates_config
        payload["step5_tuning"] = step5_tuning_config
        payload["step5_formal_active_candidate"] = step5_formal_active_candidate_config
        payload["step5_e4_bounded"] = step5_e4_bounded_config
        payload["step5_lifecycle"] = step5_lifecycle_config
        payload["step5_memory_truth"] = step5_memory_truth_config
        payload["step5_eval"] = step5_eval_config
        payload["step5_valid_loss"] = step5_valid_loss_config
        payload["step5_final_eval"] = step5_final_eval_config

    decode_strategy = str(decode.get("decode_strategy", "greedy")).strip().lower()
    decode_seed = _optional_int(decode.get("decode_seed"))
    no_repeat = _optional_int(decode.get("no_repeat_ngram_size"))
    min_len = _optional_int(decode.get("min_len"))
    domain_fusion_mode = str(decode.get("domain_fusion_mode", "gate_cross_attn"))
    if decode_strategy not in ("greedy", "nucleus", "uncertainty_low_temp_top_k"):
        raise OneControlConfigError(f"unsupported decode_strategy: {decode_strategy!r}")

    hw_semantic = {
        k: v
        for k, v in hw.items()
        if k not in ("omp_num_threads", "mkl_num_threads", "tokenizers_parallelism", "cuda_visible_devices")
    }
    thread_env = {
        "OMP_NUM_THREADS": str(_positive_int(hw.get("omp_num_threads", 1), "hardware.omp_num_threads")),
        "MKL_NUM_THREADS": str(_positive_int(hw.get("mkl_num_threads", 1), "hardware.mkl_num_threads")),
        "TOKENIZERS_PARALLELISM": "true" if _bool(hw.get("tokenizers_parallelism", False)) else "false",
    }
    launcher_env = {}
    if hw.get("cuda_visible_devices") not in (None, ""):
        launcher_env["CUDA_VISIBLE_DEVICES"] = str(hw["cuda_visible_devices"])

    field_sources = {
        "config": str(Path(config_path)),
        "override_chain": "CLI --set > configs/odcr.yaml > resolver schema defaults",
        "task": f"tasks.{tid}",
        "task.scenario": f"tasks.{tid}.scenario",
        "task.direction": f"tasks.{tid}.direction",
        "hardware": f"hardware.profiles.{hw_name}",
        "train": f"{stage_for_train}.train",
        "eval_profile": f"eval.profiles.{eval_profile_name}" if eval_profile_name else None,
        "eval_split": "eval.split",
        "decode": f"eval.decode.{decode_id}" if decode_id else None,
        "rerank": f"eval.rerank.{rerank_id}" if rerank_id else None,
        "step4_rcr": "step4.rcr",
        "step4_runtime": "step4.runtime",
        "step4_step5_dedicated_exports": "step4.step5_dedicated_exports",
        "step4_step5_pool_exports": "step4.step5_pool_exports",
        "step4_gold_quality": "step4.gold_quality",
        "step4_cf_tiers": "step4.cf_tiers",
        "rating_source": "rating_source" if stage_for_train == "step5" else None,
        "step3_structured_losses": "step3.structured_losses",
        "step3_loss_semantics": "step3.loss_semantics",
        "step3_ddp_find_unused_parameters": "step3.ddp.find_unused_parameters",
        "step3_ddp_static_graph": "step3.ddp.static_graph",
        "step3_ddp_graph_safety_preflight": "step3.ddp.graph_safety_preflight",
        "step3_scenario_profile": f"step3.scenario_profiles.{scenario}" if stage_for_train == "step3" else None,
        "step3_task_profile": (
            f"step3.task_profiles.{step3_task_profile_key}" if stage_for_train == "step3" else None
        ),
        "task_profile_id": (
            f"step3.task_profiles.{step3_task_profile_key}.profile_id" if stage_for_train == "step3" else None
        ),
        "profile_isolation_hash": "resolver-derived from isolated Step3 task profile" if stage_for_train == "step3" else None,
        "step3_optimizer": "step3.optimizer" if stage_for_train == "step3" else None,
        "step3_precision": "step3.train.backend" if stage_for_train == "step3" else None,
        "step3_tokenizer": (
            f"step3.task_profiles.{step3_task_profile_key}.tokenizer over step3.scenario_profiles.{scenario}.tokenizer over step3.tokenizer"
            if stage_for_train == "step3"
            else None
        ),
        "step3_evidence": (
            f"step3.task_profiles.{step3_task_profile_key}.evidence over step3.scenario_profiles.{scenario}.evidence over step3.evidence"
            if stage_for_train == "step3"
            else None
        ),
        "step3_scheduler": "step3.scheduler" if stage_for_train == "step3" else None,
        "step3_scheduler_damping": "step3.scheduler.validation_aware_lr_damping" if stage_for_train == "step3" else None,
        "step3_max_grad_norm": "step3.train.max_grad_norm" if stage_for_train == "step3" else None,
        "step3_batch_semantics": "step3.train global_batch_size/per_gpu_batch_size with hardware.ddp_world_size" if stage_for_train == "step3" else None,
        "step3_batch_candidate_role": "step3.task_profiles candidate" if stage_for_train == "step3" else None,
        "step3_eval_valid_batch": (
            "step3.eval resolver-derived from train per-GPU batch"
            if stage_for_train == "step3" and step3_eval_config.get("derive_from_train")
            else ("step3.eval" if stage_for_train == "step3" else None)
        ),
        "step3_eval_protocol": "step3.eval.protocol" if stage_for_train == "step3" else None,
        "step3_eval_split": "step3.eval.split" if stage_for_train == "step3" else None,
        "step3_eval_batch_candidates": "step3.eval.batch_candidates" if stage_for_train == "step3" else None,
        "step3_eval_paper_protocol": f"step3.eval.{PAPER_TARGET_ONLY_EVAL}" if stage_for_train == "step3" else None,
        "step3_backup_profiles": "step3.backup_profiles" if stage_for_train == "step3" else None,
        "step3_exploration_profiles": "step3.exploration_profiles" if stage_for_train == "step3" else None,
        "step3_worker_profiles": "step3.worker_profiles" if stage_for_train == "step3" else None,
        "step3_prefetcher": "step3.prefetcher" if stage_for_train == "step3" else None,
        "step3_checkpoint_policy": "step3.checkpoint_policy" if stage_for_train == "step3" else None,
        "step3_quality_gate": "step3.quality_gate" if stage_for_train == "step3" else None,
        "step3_grad_finite": "step3.grad_finite" if stage_for_train == "step3" else None,
        "step3_diagnostic_eval": "step3.diagnostic_eval" if stage_for_train == "step3" else None,
        "step3_cross_rank_structured_gather": "step3.cross_rank_structured_gather" if stage_for_train == "step3" else None,
        "step3_memory": "step3.memory" if stage_for_train == "step3" else None,
        "step3_timing": "step3.timing" if stage_for_train == "step3" else None,
        "step3_performance_candidates": "step3.performance_candidates" if stage_for_train == "step3" else None,
        "step3_cache_policy": "step3.cache" if stage_for_train == "step3" else None,
        "step3_objective_drift": "step3.objective_drift" if stage_for_train == "step3" else None,
        "step3_recovery": "step3.recovery" if stage_for_train == "step3" else None,
        "step3_phase_loss_schedule": "step3.phase_loss_schedule" if stage_for_train == "step3" else None,
        "step3_conflict_aware": "step3.conflict_aware" if stage_for_train == "step3" else None,
        "step3_loss_gradient_conflict_probe": "step3.loss_gradient_conflict_probe" if stage_for_train == "step3" else None,
        "step3_adapter_gating": "step3.adapter_gating" if stage_for_train == "step3" else None,
        "step3_paper_candidate_selection": "step3.paper_candidate_selection" if stage_for_train == "step3" else None,
        "step3_checkpoint_averaging": "step3.checkpoint_averaging" if stage_for_train == "step3" else None,
        "step5_lci": "step5.lci",
        "step5_uci": "step5.uci",
        "step5_explainer_gate": "step5.explainer_gate",
        "step5_ccv": "step5.ccv",
        "step5_fca": "step5.fca",
        "step5_native_lora": "step5.ccv.native_lora",
        "lora_target_policy_id": "step5.ccv.native_lora.target_policy_id",
        "head_specific_lora_allowlist_id": "step5.ccv.native_lora.target_policy_id + CLI --head",
        "final_lora_target_modules": "Step5 runtime head-aware LoRA allowlist",
        "forbidden_lora_targets": "Step5 runtime LoRA forbidden target policy",
        "deleted_legacy_modules": "Step5 active production model deletion contract",
        "retired_combined_formal_enabled": "retired head-split marker",
        "all_trainable_grad_required": "Step5 formal/E4 all-trainable-grad gate",
        "head_specific_trainable_policy": "Step5 head-aware trainable contract",
        "head_gated_loss_contract": "Step5 train loop head-gated loss contract",
        "step5_model": "step5.model",
        "step5_export_loader": "step5.export_loader",
        "step5_data_pipeline": "step5.data_pipeline",
        "step5_sampler": "step5.sampler",
        "step5_task_decoupled_policy": "step5.task_decoupled_policy",
        "step5_model_factory_policy": "derived from step5.task_decoupled_policy + CLI --head",
        "step5_prompt_templates": "step5.prompt_templates",
        "step5_effective_epoch": "step5.effective_epoch",
        "step5_batch_candidates": "step5.batch_candidates",
        "step5_tuning": "step5.tuning",
        "step5_formal_active_candidate": (
            "configs/odcr.yaml:step5.tuning.selected_tuning_candidate + configs/odcr.yaml:step5.sampler"
            if stage_for_train == "step5"
            else None
        ),
        "step5_formal_active_candidate.explanation_cf_mix_id": (
            "configs/odcr.yaml:step5.tuning.selected_tuning_candidate"
            if stage_for_train == "step5"
            else None
        ),
        "step5_formal_active_candidate.explanation_cf_mix": (
            "configs/odcr.yaml:step5.sampler.explanation.cf_tier_mix"
            if stage_for_train == "step5"
            else None
        ),
        "step5_formal_active_candidate.active_sampler_source": (
            "configs/odcr.yaml:step5.sampler + configs/odcr.yaml:step5.tuning.selected_tuning_candidate"
            if stage_for_train == "step5"
            else None
        ),
        "step5_formal_active_candidate.step4_sampling_contract_role": (
            "runs/step4/task*/run/step5_pools/step5_sampling_contract.json is pool_lineage_only"
            if stage_for_train == "step5"
            else None
        ),
        "step5_head": "CLI --head" if command == "step5" else None,
        "step5_e4_bounded": "step5.e4_bounded" if stage_for_train == "step5" else None,
        "step5_lifecycle": "step5.lifecycle" if stage_for_train == "step5" else None,
        "step5_lifecycle_phase": (
            "CLI lifecycle flag over configs/odcr.yaml:step5.lifecycle.formal_default_phase"
            if command == "step5"
            else None
        ),
        "step5_train_only": (
            "derived from step5.lifecycle formal_default_phase / CLI lifecycle flag"
            if command == "step5"
            else None
        ),
        "step5_allow_embedded_final_eval": "step5.lifecycle embedded-final-eval diagnostic gates" if command == "step5" else None,
        "step5_checkpoint_load_policy": "step5.lifecycle.checkpoint_load_policy" if stage_for_train == "step5" else None,
        "step5_eval": "step5.eval" if stage_for_train == "step5" else None,
        "step5_valid_loss": "step5.valid_loss" if stage_for_train == "step5" else None,
        "step5_final_eval": "step5.final_eval" if stage_for_train == "step5" else None,
        "train_per_gpu_batch_size": "step5.train.per_gpu_batch_size" if stage_for_train == "step5" else None,
        "valid_per_gpu_batch_size": "step5.eval.valid_per_gpu_batch_size" if stage_for_train == "step5" else None,
        "valid_global_batch_size": "step5.eval.valid_batch_size" if stage_for_train == "step5" else None,
        "valid_forward_micro_batch_size": "step5.eval.valid_forward_micro_batch_size" if stage_for_train == "step5" else None,
        "validation_microbatch_accumulation": "step5.eval.metric_accumulation" if stage_for_train == "step5" else None,
        "validation_memory_policy": "step5.eval.validation_memory_policy" if stage_for_train == "step5" else None,
        "step5_validation_mode": "step5.eval.validation_mode" if stage_for_train == "step5" else None,
        "formal_entry_E4_validation_required": (
            "step5.eval.formal_entry_E4_validation_required" if stage_for_train == "step5" else None
        ),
        "old_eval_batch_2048_retired": "step5.eval.old_eval_batch_2048_retired" if stage_for_train == "step5" else None,
        "valid_loss_components": "step5.eval.valid_loss_components" if stage_for_train == "step5" else None,
        "validation_flans_logits_materialized": "not_applicable_explanation_only" if stage_for_train == "step5" else None,
        "validation_e4_evidence_id": "AI_analysis formal-entry E4 with validation evidence" if stage_for_train == "step5" else None,
        "validation_oom_guard_status": (
            "step5.eval explicit batch is governed by runtime OOM evidence, not train per_gpu cap"
            if stage_for_train == "step5"
            else None
        ),
        "step5_memory_truth": "step5.memory_truth",
        "step5_gradient_checkpointing_reentrant_policy": "step5.memory_truth.gradient_checkpointing_reentrant_policy",
        "step5_ddp_find_unused_parameters": "configs/odcr.yaml:step5.ddp.find_unused_parameters",
        "step5_ddp_static_graph": "configs/odcr.yaml:step5.ddp.static_graph",
        "step5_ddp_find_unused_false_preflight": "configs/odcr.yaml:step5.ddp.find_unused_false_preflight",
        "formal_preflight_uses_real_data": "configs/odcr.yaml:step5.ddp.find_unused_false_preflight real-data policy",
        "step5_ccv.control_fields": "configs/odcr.yaml:step5.ccv.control_fields",
        "step5_ccv.required_control_fields": (
            "code/odcr_core/config_resolver.py:_STEP5_CCV_REQUIRED_CONTROL_FIELDS + "
            "configs/odcr.yaml:step5.ccv.control_fields"
        ),
        "step5_ccv.derived_control_input.polarity_ids": (
            "configs/odcr.yaml:step5.ccv.control_fields.polarity_anchor -> Step5 Processor/control packet"
        ),
        "step5_train_explainer_loss_weight": "step5.train.explainer_loss_weight",
        "hardware.max_parallel_cpu": f"hardware.profiles.{hw_name}.max_parallel_cpu",
        "hardware.num_proc": f"hardware.profiles.{hw_name}.num_proc",
        "hardware.max_num_proc": f"hardware.profiles.{hw_name}.max_num_proc",
        "hardware.reserved_cpu": f"hardware.profiles.{hw_name}.reserved_cpu",
        "hardware.tokenization_num_proc": f"hardware.profiles.{hw_name}.num_proc auto resolver",
        "hardware.omp_num_threads": f"hardware.profiles.{hw_name}.omp_num_threads",
        "hardware.mkl_num_threads": f"hardware.profiles.{hw_name}.mkl_num_threads",
        "hardware.tokenizers_parallelism": f"hardware.profiles.{hw_name}.tokenizers_parallelism",
        "runtime_env.OMP_NUM_THREADS": "hardware.omp_num_threads -> OMP_NUM_THREADS transport",
        "runtime_env.MKL_NUM_THREADS": "hardware.mkl_num_threads -> MKL_NUM_THREADS transport",
        "runtime_env.TOKENIZERS_PARALLELISM": "hardware.tokenizers_parallelism -> TOKENIZERS_PARALLELISM transport",
        "runtime_env.thread_env_requested": "hardware profile thread controls",
        "runtime_env.thread_env_effective": "resolver-owned thread controls",
        "hardware.dataloader_num_workers_train": f"hardware.profiles.{hw_name}.dataloader_num_workers_train",
        "hardware.dataloader_num_workers_valid": f"hardware.profiles.{hw_name}.dataloader_num_workers_valid",
        "hardware.dataloader_num_workers_test": f"hardware.profiles.{hw_name}.dataloader_num_workers_test",
        "hardware.dataloader_prefetch_factor_train": f"hardware.profiles.{hw_name}.dataloader_prefetch_factor_train",
        "hardware.dataloader_prefetch_factor_valid": f"hardware.profiles.{hw_name}.dataloader_prefetch_factor_valid",
        "hardware.dataloader_prefetch_factor_test": f"hardware.profiles.{hw_name}.dataloader_prefetch_factor_test",
        "hardware.pin_memory": f"hardware.profiles.{hw_name}.pin_memory",
        "hardware.persistent_workers": f"hardware.profiles.{hw_name}.persistent_workers",
        "hardware.non_blocking_h2d": f"hardware.profiles.{hw_name}.non_blocking_h2d",
        "train_precision": train_precision_source,
        "runtime_precision_mode": "ResolvedConfig.train_precision -> ODCR_RUNTIME_PRECISION_MODE transport",
        "runs_dir": "project.run_root",
        "cache_dir": "project.cache_dir",
        "data_dir": "project.data_dir",
        "merged_dir": "project.merged_dir",
        "models_dir": "env.models_dir",
        "step5_text_model": "env.step5_text_model",
        "sentence_embed_model": "env.sentence_embed_model",
        "embed_dim": "env.embed_dim",
        "offline": "env.offline",
        "local_files_only": "env.local_files_only",
        "upstream_resolution": (
            "latest.json -> meta/stage_status.json via odcr_core.upstream_resolver"
            if upstream_resolution_payload
            else None
        ),
        "active_stage_status": (
            "latest.json -> meta/stage_status.json via odcr_core.upstream_resolver"
            if active_stage_status_payload
            else None
        ),
        **cli_sources,
    }
    for key in (
        "cf_reliability_weights",
        "uncertainty_weights",
        "rating_delta_soft_cap",
        "route_scorer",
        "route_explainer",
        "confidence_bucket",
        "train_keep",
        "sample_weight_hint",
        "export.required_fields",
    ):
        field_sources[f"step4_rcr.{key}"] = f"step4.rcr.{key}"
    for key in (
        "decode_threads",
        "decode_chunk",
        "partial_format",
        "perf_log_interval",
        "preflight_default_max_samples",
        "partial_wait_timeout_seconds",
    ):
        field_sources[f"step4_runtime.{key}"] = f"step4.runtime.{key}"
    for key in (
        "enabled",
        "output_dir_name",
        "full_audit_format",
        "scorer_train_format",
        "explainer_train_format",
        "scorer_filter",
        "explainer_filter",
        "write_gold_cf_subsplits",
        "full_audit_role",
        "atomic_write",
        "validate_after_write",
        "chunk_rows",
    ):
        field_sources[f"step4_step5_dedicated_exports.{key}"] = f"step4.step5_dedicated_exports.{key}"
    for key in (
        "enabled",
        "output_dir_name",
        "full_audit_role",
        "legacy_dedicated_exports_role",
        "chunk_rows",
    ):
        field_sources[f"step4_step5_pool_exports.{key}"] = f"step4.step5_pool_exports.{key}"
    for key in (
        "schema_version",
        "high_min_score",
        "medium_min_score",
        "hard_reject_min_words",
        "hard_reject_max_words",
        "max_repeat_ngram_ratio",
        "good_repeat_ngram_ratio",
        "proxy_weights",
        "sampling_weight",
        "sanity",
    ):
        field_sources[f"step4_gold_quality.{key}"] = f"step4.gold_quality.{key}"
    for key in ("schema_version", "hard_reject", "explanation", "sampling_weight"):
        field_sources[f"step4_cf_tiers.{key}"] = f"step4.cf_tiers.{key}"
    for key in (
        "cache_enabled",
        "cache_namespace",
        "chunk_rows",
        "validate_sample_rows",
        "bounded_max_rows",
        "stale_policy",
    ):
        field_sources[f"step5_export_loader.{key}"] = f"step5.export_loader.{key}"
    for key in (
        "sample_plan_enabled",
        "token_cache_enabled",
        "bounded_token_cache_enabled",
        "dataloader_queue_size",
        "workers_per_rank_candidates",
        "prefetch_factor_candidates",
        "max_parallel_cpu",
        "reserved_cpu",
        "cpu_budget_guard",
        "pipeline_timing_enabled",
        "gpu_util_sampling_enabled",
    ):
        field_sources[f"step5_data_pipeline.{key}"] = f"step5.data_pipeline.{key}"
    for key in (
        "enabled",
        "contract_source",
        "effective_epoch_enabled",
        "seed",
        "rotate_across_epochs",
        "full_audit_default_allowed",
        "legacy_gold_heavy_exports_allowed",
        "auto_budget",
        "explanation",
        "epochs",
    ):
        field_sources[f"step5_sampler.{key}"] = f"step5.sampler.{key}"
    for key in (
        "schema_version",
        "allowed_template_count",
        "train_policy",
        "valid_test_policy",
        "input_formatting_only",
        "explanation_only",
        "active_templates",
    ):
        field_sources[f"step5_prompt_templates.{key}"] = f"step5.prompt_templates.{key}"
    for key in (
        "enabled",
        "max_effective_epochs",
        "early_stopping_patience",
        "retired_full_table_epochs",
        "retired_full_table_policy",
    ):
        field_sources[f"step5_effective_epoch.{key}"] = f"step5.effective_epoch.{key}"
    for key in ("ddp_world_size", "fsdp_zero_policy", "selected_default", "candidates"):
        field_sources[f"step5_batch_candidates.{key}"] = f"step5.batch_candidates.{key}"
    for key in (
        "enabled",
        "selected_tuning_candidate",
        "fallback_tuning_candidate",
        "effective_samples",
        "optimizer_steps",
        "batch_candidate",
        "fallback_batch_candidate",
        "selected_budget_candidate",
        "pilot_fraction_candidates",
        "lr_candidates",
        "warmup_fraction_candidates",
        "innovation_weight_candidates",
        "ratio_candidates",
        "cf_tier_mix_candidates",
        "gold_tier_mix_candidates",
        "search_strategy",
    ):
        field_sources[f"step5_tuning.{key}"] = f"step5.tuning.{key}"
    for key in (
        "valid_per_gpu_batch_size",
        "valid_batch_size",
        "valid_global_batch_size",
        "valid_forward_micro_batch_size",
        "valid_micro_batch_size",
        "test_per_gpu_batch_size",
        "test_forward_micro_batch_size",
        "metric_accumulation",
        "validation_microbatch_accumulation",
        "validation_memory_policy",
        "step5_validation_mode",
        "formal_entry_E4_validation_required",
        "old_eval_batch_2048_retired",
        "valid_loss_components",
    ):
        field_sources[f"step5_eval.{key}"] = f"step5.eval.{key}"
    for key in (
        "enabled",
        "evidence_level",
        "namespace_root",
        "max_runtime_seconds",
        "max_samples_guard",
        "oom_policy",
        "formal_namespace_policy",
        "batch_candidates",
        "dataloader_candidates",
        "row_candidates",
        "long_window_candidates",
    ):
        field_sources[f"step5_e4_bounded.{key}"] = f"step5.e4_bounded.{key}"
    for key in (
        "schema_version",
        "formal_default_phase",
        "embedded_final_eval_default",
        "allow_embedded_final_eval_diagnostic",
        "explanation_handoff_required_for_downstream",
        "checkpoint_load_policy",
        "cpu_staged_checkpoint_load_required",
        "write_latest_after_train_only",
        "e5_post_train_lifecycle_required",
    ):
        field_sources[f"step5_lifecycle.{key}"] = f"step5.lifecycle.{key}"
    for key in (
        "schema_version",
        "reserved_diagnostic_only",
        "reject_on_reserved",
        "reject_on_oom",
        "reject_on_allocated_ratio",
        "allocated_warning_ratio",
        "nvidia_smi_instability_ratio",
        "memory_creep_delta_gb",
        "short_window_steps",
        "long_window_steps",
        "track_param_memory",
        "track_optimizer_memory",
        "track_sequence_lengths",
        "gradient_checkpointing_enabled",
        "gradient_checkpointing_reentrant_policy",
        "disable_use_cache_during_training",
        "empty_cache_for_measurement_only",
    ):
        field_sources[f"step5_memory_truth.{key}"] = f"step5.memory_truth.{key}"
    consumed = {
        "single_config": str(Path(config_path)),
        "hardware_profile": hw_name,
        "eval_profile": eval_profile_name or None,
        "decode_profile": decode_id or None,
        "rerank_profile": rerank_id or None,
        "step4_rcr": "step4.rcr",
        "step4_runtime": "step4.runtime",
        "step4_step5_dedicated_exports": "step4.step5_dedicated_exports",
        "step4_step5_pool_exports": "step4.step5_pool_exports",
        "rating_source": "rating_source" if stage_for_train == "step5" else None,
        "step5_data_pipeline": "step5.data_pipeline" if stage_for_train == "step5" else None,
        "upstream_resolution": upstream_resolution_payload if upstream_resolution_payload else None,
        "active_stage_status": active_stage_status_payload if active_stage_status_payload else None,
        "step3_structured_losses": "step3.structured_losses",
        "step3_loss_semantics": "step3.loss_semantics",
        "step3_ddp": "step3.ddp",
        "step3_scenario_profile": f"step3.scenario_profiles.{scenario}" if stage_for_train == "step3" else None,
        "step3_task_profile": f"step3.task_profiles.{step3_task_profile_key}" if stage_for_train == "step3" else None,
        "step3_optimizer": "step3.optimizer" if stage_for_train == "step3" else None,
        "step3_precision": "step3.train.backend" if stage_for_train == "step3" else None,
        "step3_tokenizer": "step3.tokenizer + scenario profile tokenizer + isolated task profile tokenizer" if stage_for_train == "step3" else None,
        "step3_evidence": "step3.evidence + scenario profile evidence + isolated task profile evidence" if stage_for_train == "step3" else None,
        "step3_scheduler": "step3.scheduler" if stage_for_train == "step3" else None,
        "step3_eval": "step3.eval" if stage_for_train == "step3" else None,
        "step3_backup_profiles": "step3.backup_profiles" if stage_for_train == "step3" else None,
        "step3_exploration_profiles": "step3.exploration_profiles" if stage_for_train == "step3" else None,
        "step3_worker_profiles": "step3.worker_profiles" if stage_for_train == "step3" else None,
        "step3_prefetcher": "step3.prefetcher" if stage_for_train == "step3" else None,
        "step3_checkpoint_policy": "step3.checkpoint_policy" if stage_for_train == "step3" else None,
        "step3_quality_gate": "step3.quality_gate" if stage_for_train == "step3" else None,
        "step3_grad_finite": "step3.grad_finite" if stage_for_train == "step3" else None,
        "step3_diagnostic_eval": "step3.diagnostic_eval" if stage_for_train == "step3" else None,
        "step3_cross_rank_structured_gather": "step3.cross_rank_structured_gather" if stage_for_train == "step3" else None,
        "step3_memory": "step3.memory" if stage_for_train == "step3" else None,
        "step3_timing": "step3.timing" if stage_for_train == "step3" else None,
        "step3_performance_candidates": "step3.performance_candidates" if stage_for_train == "step3" else None,
        "step3_cache_policy": "step3.cache" if stage_for_train == "step3" else None,
        "step3_objective_drift": "step3.objective_drift" if stage_for_train == "step3" else None,
        "step3_recovery": "step3.recovery" if stage_for_train == "step3" else None,
        "step3_phase_loss_schedule": "step3.phase_loss_schedule" if stage_for_train == "step3" else None,
        "step3_conflict_aware": "step3.conflict_aware" if stage_for_train == "step3" else None,
        "step3_loss_gradient_conflict_probe": "step3.loss_gradient_conflict_probe" if stage_for_train == "step3" else None,
        "step3_adapter_gating": "step3.adapter_gating" if stage_for_train == "step3" else None,
        "step3_paper_candidate_selection": "step3.paper_candidate_selection" if stage_for_train == "step3" else None,
        "step3_checkpoint_averaging": "step3.checkpoint_averaging" if stage_for_train == "step3" else None,
        "runtime_roots": {
            "data_dir": "project.data_dir",
            "merged_dir": "project.merged_dir",
            "runs_dir": "project.run_root",
            "cache_dir": "project.cache_dir",
            "models_dir": "env.models_dir",
            "step5_text_model": "env.step5_text_model",
            "sentence_embed_model": "env.sentence_embed_model",
            "embed_dim": "env.embed_dim",
            "offline": "env.offline",
            "local_files_only": "env.local_files_only",
        },
        "step5_innovation": {
            "lci": "step5.lci",
            "uci": "step5.uci",
            "explainer_gate": "step5.explainer_gate",
            "ccv": "step5.ccv",
            "fca": "step5.fca",
            "native_lora": "step5.ccv.native_lora",
            "lora_target_policy_id": "step5.ccv.native_lora.target_policy_id",
            "head_specific_lora_allowlist_id": "step5.ccv.native_lora.target_policy_id + CLI --head",
            "final_lora_target_modules": "Step5 runtime head-aware LoRA allowlist",
            "forbidden_lora_targets": "Step5 runtime LoRA forbidden target policy",
            "deleted_legacy_modules": "Step5 active production model deletion contract",
            "retired_combined_formal_enabled": "retired head-split marker",
            "all_trainable_grad_required": "Step5 formal/E4 all-trainable-grad gate",
            "head_specific_trainable_policy": "Step5 head-aware trainable contract",
            "head_gated_loss_contract": "Step5 train loop head-gated loss contract",
            "model": "step5.model",
            "ddp": "step5.ddp",
            "explainer_loss_weight": "step5.train.explainer_loss_weight",
        },
        "step5_task_decoupled_policy": (
            "step5.task_decoupled_policy" if stage_for_train == "step5" else None
        ),
        "step5_model_factory_policy": (
            "derived from step5.task_decoupled_policy + CLI --head" if stage_for_train == "step5" else None
        ),
        "step5_export_loader": "step5.export_loader" if stage_for_train == "step5" else None,
        "step5_data_pipeline": "step5.data_pipeline" if stage_for_train == "step5" else None,
        "step5_sampler": "step5.sampler" if stage_for_train == "step5" else None,
        "step5_prompt_templates": "step5.prompt_templates" if stage_for_train == "step5" else None,
        "step5_effective_epoch": "step5.effective_epoch" if stage_for_train == "step5" else None,
        "step5_batch_candidates": "step5.batch_candidates" if stage_for_train == "step5" else None,
        "step5_tuning": "step5.tuning" if stage_for_train == "step5" else None,
        "step5_formal_active_candidate": (
            "configs/odcr.yaml:step5.tuning.selected_tuning_candidate + configs/odcr.yaml:step5.sampler"
            if stage_for_train == "step5"
            else None
        ),
        "step5_head": "CLI --head" if command == "step5" else None,
        "selected_tuning_candidate": "step5.tuning.selected_tuning_candidate" if stage_for_train == "step5" else None,
        "fallback_tuning_candidate": "step5.tuning.fallback_tuning_candidate" if stage_for_train == "step5" else None,
        "step5_effective_samples": "step5.tuning.effective_samples" if stage_for_train == "step5" else None,
        "step5_optimizer_steps": "step5.tuning.optimizer_steps" if stage_for_train == "step5" else None,
        "step5_e4_bounded": "step5.e4_bounded" if stage_for_train == "step5" else None,
        "step5_lifecycle": "step5.lifecycle" if stage_for_train == "step5" else None,
        "step5_lifecycle_phase": (
            "CLI lifecycle flag over configs/odcr.yaml:step5.lifecycle.formal_default_phase"
            if command == "step5"
            else None
        ),
        "step5_train_only": (
            "derived from step5.lifecycle formal_default_phase / CLI lifecycle flag"
            if command == "step5"
            else None
        ),
        "step5_allow_embedded_final_eval": "step5.lifecycle embedded-final-eval diagnostic gates" if command == "step5" else None,
        "step5_checkpoint_load_policy": "step5.lifecycle.checkpoint_load_policy" if stage_for_train == "step5" else None,
        "step5_memory_truth": "step5.memory_truth" if stage_for_train == "step5" else None,
        "step5_valid_loss": "step5.valid_loss" if stage_for_train == "step5" else None,
        "step5_final_eval": "step5.final_eval" if stage_for_train == "step5" else None,
    }
    train_fp = fingerprint({"payload": payload, "hardware": hw_semantic, "ddp_world_size": ddp_world_size})
    gen_fp = fingerprint({"decode": decode, "eval_batch_size": eval_batch_size, "rerank": rerank}) if need_decode else ""
    runtime_fp = fingerprint(runtime_diagnostics_fingerprint_source())
    field_sources = {k: v for k, v in field_sources.items() if v is not None}
    sources = [SourceRecord(k, v, field_sources.get(k, "configs/odcr.yaml")) for k, v in sorted(field_sources.items())]

    resolved_snapshot = {
        "task": {
            "id": tid,
            "source": auxiliary,
            "target": target,
            "scenario": scenario,
            "direction": direction,
            "task_profile_key": step3_task_profile_key if stage_for_train == "step3" else None,
            "task_profile_id": step3_task_profile_config.get("profile_id") if stage_for_train == "step3" else None,
            "active_profile": step3_task_profile_config.get("active_profile") if stage_for_train == "step3" else None,
            "profile_isolation_hash": step3_task_profile_config.get("profile_isolation_hash") if stage_for_train == "step3" else None,
        },
        "hardware": {
            "profile": hw_name,
            **hw_semantic,
            "thread_controls": {
                "omp_num_threads": int(thread_env["OMP_NUM_THREADS"]),
                "mkl_num_threads": int(thread_env["MKL_NUM_THREADS"]),
                "tokenizers_parallelism": thread_env["TOKENIZERS_PARALLELISM"] == "true",
                "source": {
                    "omp_num_threads": f"hardware.profiles.{hw_name}.omp_num_threads",
                    "mkl_num_threads": f"hardware.profiles.{hw_name}.mkl_num_threads",
                    "tokenizers_parallelism": f"hardware.profiles.{hw_name}.tokenizers_parallelism",
                },
            },
        },
        "train": {
            "stage": stage_for_train,
            "global_batch_size": batch_size,
            "batch_size": batch_size,
            "per_gpu_batch_size": per_gpu,
            "micro_batch_size_alias": per_gpu,
            "batch_semantics_version": NO_ACCUM_BATCH_SEMANTICS_VERSION,
            "batch_formula": "global_batch_size = per_gpu_batch_size * ddp_world_size",
            "grad_accum_removed": True,
            **(
                {
                    "step3_batch_semantics": NO_ACCUM_BATCH_SEMANTICS_VERSION,
                    "step3_batch_formula": "global_batch_size = per_gpu_batch_size * ddp_world_size",
                    "step3_batch_candidate_role": step3_batch_candidate_role,
                    "candidate": step3_batch_candidate_role,
                }
                if stage_for_train == "step3"
                else {}
            ),
            "ddp_world_size": ddp_world_size,
            "precision": train_precision,
            "precision_source": train_precision_source,
            "max_epochs": int(row.get("max_epochs", row["epochs"])),
            "epochs": int(row["epochs"]),
            "lr": row["lr"],
            **(
                {
                    "min_epochs": int(row["min_epochs"]),
                    "early_stop_patience": int(row["early_stop_patience"]),
                    "validate_every_epochs": int(row["validate_every_epochs"]),
                    "max_grad_norm": float(row["max_grad_norm"]),
                }
                if stage_for_train == "step3"
                else {"coef": row["coef"]}
            ),
            **(
                {"explainer_loss_weight": float(row["explainer_loss_weight"])}
                if stage_for_train == "step5"
                else {}
            ),
            **(
                {
                    "selected_tuning_candidate": str(step5_tuning_config.get("selected_tuning_candidate") or ""),
                    "fallback_tuning_candidate": str(step5_tuning_config.get("fallback_tuning_candidate") or ""),
                    "effective_samples": dict(step5_tuning_config.get("effective_samples") or {}),
                    "optimizer_steps": dict(step5_tuning_config.get("optimizer_steps") or {}),
                    "batch_candidate": str(step5_tuning_config.get("batch_candidate") or ""),
                    "fallback_batch_candidate": str(step5_tuning_config.get("fallback_batch_candidate") or ""),
                }
                if stage_for_train == "step5"
                else {}
            ),
            **(
                {"step5_head": step5_head_norm}
                if command == "step5"
                else {}
            ),
            **(
                {
                    "ddp_find_unused_parameters": bool(row["ddp_find_unused_parameters"]),
                    "ddp_static_graph": bool(row["ddp_static_graph"]),
                    "ddp_find_unused_false_preflight": str(row["ddp_find_unused_false_preflight"]),
                }
                if stage_for_train == "step5"
                else {}
            ),
            **(
                {
                    "ddp_find_unused_parameters": bool(row["ddp_find_unused_parameters"]),
                    "ddp_static_graph": bool(row["ddp_static_graph"]),
                    "ddp_graph_safety_preflight": bool(row["ddp_graph_safety_preflight"]),
                }
                if stage_for_train == "step3"
                else {}
            ),
        },
        "eval": {
            "profile": eval_profile_name or None,
            "split": eval_split,
            "eval_batch_size": eval_batch_size,
            "eval_per_gpu_batch_size": eval_per_gpu,
            **(
                {
                    "step5_train_validation": {
                        "valid_per_gpu_batch_size": int(step5_eval_config["valid_per_gpu_batch_size"]),
                        "valid_global_batch_size": int(step5_eval_config["valid_global_batch_size"]),
                        "valid_forward_micro_batch_size": int(step5_eval_config["valid_forward_micro_batch_size"]),
                        "validation_microbatch_accumulation": bool(
                            step5_eval_config["validation_microbatch_accumulation"]
                        ),
                        "validation_memory_policy": str(step5_eval_config["validation_memory_policy"]),
                        "step5_validation_mode": str(step5_eval_config["step5_validation_mode"]),
                        "formal_entry_E4_validation_required": bool(
                            step5_eval_config["formal_entry_E4_validation_required"]
                        ),
                        "old_eval_batch_2048_retired": bool(step5_eval_config["old_eval_batch_2048_retired"]),
                    },
                    "eval_batch_size_role_for_step5_train_validation": (
                        "step5_official_eval_uses_step5_eval_batch"
                        if command == "eval"
                        else "not_active"
                    ),
                }
                if stage_for_train == "step5"
                else {}
            ),
            "decode": decode,
            "rerank": rerank if rerank else None,
            "rerank_source_table": (
                {
                    "eval_profile": f"eval.profiles.{eval_profile_name}.rerank",
                    "rerank_profile": f"eval.rerank.{rerank_id}",
                    "transport": "ResolvedConfig.rerank_profile_json + step5 eval-rerank CLI transport",
                }
                if rerank
                else None
            ),
        },
        "step4_rcr": step4_rcr_config if command == "step4" else None,
        "step4_runtime": step4_runtime_config if command == "step4" else None,
        "step4_step5_dedicated_exports": step4_step5_dedicated_exports_config if command == "step4" else None,
        "step4_step5_pool_exports": step4_step5_pool_exports_config if command == "step4" else None,
        "step4_gold_quality": step4_gold_quality_config if command == "step4" else None,
        "step4_cf_tiers": step4_cf_tiers_config if command == "step4" else None,
        "step3_structured_losses": step3_structured_losses_config if stage_for_train == "step3" else None,
        "step3_loss_semantics": step3_loss_semantics_config if stage_for_train == "step3" else None,
        "step3_ddp": step3_ddp_config if stage_for_train == "step3" else None,
        "step3_scenario_profile": step3_scenario_profile if stage_for_train == "step3" else None,
        "step3_task_profile": step3_task_profile_config if stage_for_train == "step3" else None,
        "step3_optimizer": step3_optimizer_config if stage_for_train == "step3" else None,
        "step3_precision": step3_backend_config if stage_for_train == "step3" else None,
        "step3_tokenizer": step3_tokenizer_config if stage_for_train == "step3" else None,
        "step3_evidence": step3_evidence_config if stage_for_train == "step3" else None,
        "step3_scheduler": step3_scheduler_config if stage_for_train == "step3" else None,
        "step3_eval": step3_eval_config if stage_for_train == "step3" else None,
        "step3_backup_profiles": step3_backup_profiles_config if stage_for_train == "step3" else None,
        "step3_exploration_profiles": step3_exploration_profiles_config if stage_for_train == "step3" else None,
        "step3_worker_profiles": step3_worker_profiles_config if stage_for_train == "step3" else None,
        "step3_prefetcher": step3_prefetcher_config if stage_for_train == "step3" else None,
        "step3_checkpoint_policy": step3_checkpoint_policy_config if stage_for_train == "step3" else None,
        "step3_quality_gate": step3_quality_gate_config if stage_for_train == "step3" else None,
        "step3_grad_finite": step3_grad_finite_config if stage_for_train == "step3" else None,
        "step3_diagnostic_eval": step3_diagnostic_eval_config if stage_for_train == "step3" else None,
        "step3_cross_rank_structured_gather": step3_cross_rank_gather_config if stage_for_train == "step3" else None,
        "step3_memory": step3_memory_config if stage_for_train == "step3" else None,
        "step3_timing": step3_timing_config if stage_for_train == "step3" else None,
        "step3_performance_candidates": step3_performance_candidates_config if stage_for_train == "step3" else None,
        "step3_cache_policy": step3_cache_policy_config if stage_for_train == "step3" else None,
        "step3_objective_drift": step3_objective_drift_config if stage_for_train == "step3" else None,
        "step3_recovery": step3_recovery_config if stage_for_train == "step3" else None,
        "step3_phase_loss_schedule": step3_phase_loss_schedule_config if stage_for_train == "step3" else None,
        "step3_conflict_aware": step3_conflict_aware_config if stage_for_train == "step3" else None,
        "step3_loss_gradient_conflict_probe": step3_loss_gradient_conflict_probe_config if stage_for_train == "step3" else None,
        "step3_adapter_gating": step3_adapter_gating_config if stage_for_train == "step3" else None,
        "step3_paper_candidate_selection": step3_paper_candidate_selection_config if stage_for_train == "step3" else None,
        "step3_checkpoint_averaging": step3_checkpoint_averaging_config if stage_for_train == "step3" else None,
        "step5": step5_innovation_config if stage_for_train == "step5" else None,
        "rating_source": rating_source_config if stage_for_train == "step5" else None,
        "step5_task_decoupled_policy": (
            step5_task_decoupled_policy_config if stage_for_train == "step5" else None
        ),
        "step5_model_factory_policy": step5_model_factory_policy if stage_for_train == "step5" else None,
        "lora_target_policy_id": row.get("lora_target_policy_id") if stage_for_train == "step5" else None,
        "head_specific_lora_allowlist_id": row.get("head_specific_lora_allowlist_id") if stage_for_train == "step5" else None,
        "final_lora_target_modules": list(row.get("final_lora_target_modules") or []) if stage_for_train == "step5" else None,
        "forbidden_lora_targets": list(row.get("forbidden_lora_targets") or []) if stage_for_train == "step5" else None,
        "deleted_legacy_modules": list(row.get("deleted_legacy_modules") or []) if stage_for_train == "step5" else None,
        "retired_combined_formal_enabled": False if stage_for_train == "step5" else None,
        "all_trainable_grad_required": True if stage_for_train == "step5" else None,
        "head_specific_trainable_policy": row.get("head_specific_trainable_policy") if stage_for_train == "step5" else None,
        "head_gated_loss_contract": row.get("head_gated_loss_contract") if stage_for_train == "step5" else None,
        "step5_model": step5_model_config if stage_for_train == "step5" else None,
        "step5_export_loader": step5_export_loader_config if stage_for_train == "step5" else None,
        "step5_data_pipeline": step5_data_pipeline_config if stage_for_train == "step5" else None,
        "step5_sampler": step5_sampler_config if stage_for_train == "step5" else None,
        "step5_head": step5_head_norm if command == "step5" else None,
        "step5_prompt_templates": step5_prompt_templates_config if stage_for_train == "step5" else None,
        "step5_effective_epoch": step5_effective_epoch_config if stage_for_train == "step5" else None,
        "step5_batch_candidates": step5_batch_candidates_config if stage_for_train == "step5" else None,
        "step5_tuning": step5_tuning_config if stage_for_train == "step5" else None,
        "step5_eval": step5_eval_config if stage_for_train == "step5" else None,
        "step5_valid_loss": step5_valid_loss_config if stage_for_train == "step5" else None,
        "step5_final_eval": step5_final_eval_config if stage_for_train == "step5" else None,
        "step5_formal_active_candidate": (
            step5_formal_active_candidate_config if stage_for_train == "step5" else None
        ),
        "selected_tuning_candidate": (
            str(step5_tuning_config.get("selected_tuning_candidate") or "") if stage_for_train == "step5" else None
        ),
        "fallback_tuning_candidate": (
            str(step5_tuning_config.get("fallback_tuning_candidate") or "") if stage_for_train == "step5" else None
        ),
        "step5_effective_samples": dict(step5_tuning_config.get("effective_samples") or {}) if stage_for_train == "step5" else None,
        "step5_optimizer_steps": dict(step5_tuning_config.get("optimizer_steps") or {}) if stage_for_train == "step5" else None,
        "step5_e4_bounded": step5_e4_bounded_config if stage_for_train == "step5" else None,
        "step5_lifecycle": step5_lifecycle_config if stage_for_train == "step5" else None,
        "step5_lifecycle_phase": step5_lifecycle_phase if command == "step5" else None,
        "step5_train_only": bool(step5_train_only_resolved) if command == "step5" else None,
        "step5_allow_embedded_final_eval": bool(step5_allow_embedded_final_eval) if command == "step5" else None,
        "step5_checkpoint_load_policy": (
            str(step5_lifecycle_config.get("checkpoint_load_policy") or "cpu_staged")
            if stage_for_train == "step5"
            else None
        ),
        "step5_memory_truth": step5_memory_truth_config if stage_for_train == "step5" else None,
        "step5_ddp": step5_ddp_config if stage_for_train == "step5" else None,
        "run": {
            "stage_run_dir": str(run_root),
            "meta_dir": log_dir,
            "from_step3": from_run,
            "from_step4": step4_run,
            "from_step5": step5_run,
            "step5_head": step5_head_norm if command == "step5" else None,
            "eval_run_dir": eval_run_dir,
        },
        "upstream_resolution": upstream_resolution_payload if upstream_resolution_payload else None,
        "active_stage_status": active_stage_status_payload if active_stage_status_payload else None,
        "runtime_env": {
            "thread_env_requested": dict(thread_env),
            "thread_env_effective": dict(thread_env),
            "launcher_env_requested": dict(launcher_env),
            "launcher_env_effective": dict(launcher_env),
            "num_proc": int(num_proc),
            "max_parallel_cpu": int(hw.get("max_parallel_cpu", 0) or 0),
            "reserved_cpu": int((hw.get("worker_budget_formula") or {}).get("reserved_cpu", 2))
            if isinstance(hw.get("worker_budget_formula"), Mapping)
            else 2,
            "tokenization_formula": (
                f"num_proc({int(num_proc)}) + reserved_cpu("
                f"{int((hw.get('worker_budget_formula') or {}).get('reserved_cpu', 2)) if isinstance(hw.get('worker_budget_formula'), Mapping) else 2}) "
                f"<= max_parallel_cpu({int(hw.get('max_parallel_cpu', 0) or 0)})"
            ),
            "worker_formula": (
                f"dataloader_num_workers_train({int(hw.get('dataloader_num_workers_train', 0) or 0)}) "
                f"* ddp_world_size({int(ddp_world_size)}) + reserved_cpu("
                f"{int((hw.get('worker_budget_formula') or {}).get('reserved_cpu', 2)) if isinstance(hw.get('worker_budget_formula'), Mapping) else 2}) "
                f"<= max_parallel_cpu({int(hw.get('max_parallel_cpu', 0) or 0)})"
            ),
            "source": {
                "num_proc": f"hardware.profiles.{hw_name}.num_proc",
                "tokenization_num_proc": f"hardware.profiles.{hw_name}.num_proc auto resolver",
                "max_parallel_cpu": f"hardware.profiles.{hw_name}.max_parallel_cpu",
                "max_num_proc": f"hardware.profiles.{hw_name}.max_num_proc",
                "reserved_cpu": f"hardware.profiles.{hw_name}.reserved_cpu",
                "omp_num_threads": f"hardware.profiles.{hw_name}.omp_num_threads",
                "mkl_num_threads": f"hardware.profiles.{hw_name}.mkl_num_threads",
                "tokenizers_parallelism": f"hardware.profiles.{hw_name}.tokenizers_parallelism",
            },
        },
        "roots": {
            "runs_dir": runtime_roots["runs_dir"],
            "cache_dir": runtime_roots["cache_dir"],
            "data_dir": runtime_roots["data_dir"],
            "merged_dir": runtime_roots["merged_dir"],
            "models_dir": runtime_roots["models_dir"],
            "source": {
                "runs_dir": "project.run_root",
                "cache_dir": "project.cache_dir",
                "data_dir": "project.data_dir",
                "merged_dir": "project.merged_dir",
                "models_dir": "env.models_dir",
            },
        },
        "models": {
            "step5_text_model": runtime_roots["step5_text_model"],
            "sentence_embed_model": runtime_roots["sentence_embed_model"],
            "source": {
                "step5_text_model": "env.step5_text_model",
                "sentence_embed_model": "env.sentence_embed_model",
            },
        },
        "embed_dim": {
            "value": int(runtime_roots["embed_dim"]),
            "source": "env.embed_dim",
        },
        "offline": {
            "value": bool(runtime_roots["offline"]),
            "source": "env.offline",
        },
        "local_files_only": {
            "value": bool(runtime_roots["local_files_only"]),
            "source": "env.local_files_only",
        },
        "field_sources": field_sources,
    }

    cfg_obj = ResolvedConfig(
        command="eval-rerank" if command == "eval" and need_rerank else command,
        repo_root=repo_root,
        code_dir=_CODE_DIR,
        task_id=tid,
        auxiliary=auxiliary,
        target=target,
        scenario=scenario,
        direction=direction,
        task_profile_id=step3_task_profile_config.get("profile_id", "") if stage_for_train == "step3" else "",
        task_profile_key=step3_task_profile_key if stage_for_train == "step3" else "",
        profile_isolation_hash=step3_task_profile_config.get("profile_isolation_hash", "") if stage_for_train == "step3" else "",
        preset_name=stage_for_train,
        run_name=run_name,
        from_run=from_run,
        step5_run=step5_run,
        step4_run=step4_run,
        step3_checkpoint_dir=step3_checkpoint_dir,
        train_csv=None,
        model_path=model_path,
        learning_rate=float(row["lr"]),
        coef=float(row.get("coef", 0.0)),
        train_batch_size=batch_size,
        global_batch_size=batch_size,
        per_device_train_batch_size=per_gpu,
        per_gpu_batch_size=per_gpu,
        effective_global_batch_size=eff,
        batch_semantics_version=NO_ACCUM_BATCH_SEMANTICS_VERSION,
        grad_accum_removed=True,
        epochs=int(row["epochs"]),
        max_epochs=int(row.get("max_epochs", row["epochs"])),
        min_epochs=int(row.get("min_epochs", 0)),
        early_stop_patience=int(row.get("early_stop_patience", 0)),
        validate_every_epochs=int(row.get("validate_every_epochs", 1)),
        max_grad_norm=float(row.get("max_grad_norm", 0.0)),
        tokenizer_max_length=int(row.get("tokenizer_max_length", 0)),
        evidence_max_length=int(row.get("evidence_max_length", 0)),
        valid_batch_size=int(row.get("valid_batch_size", 0)),
        valid_micro_batch_size=int(row.get("valid_micro_batch_size", 0)),
        valid_per_gpu_batch_size=int(row.get("valid_per_gpu_batch_size", 0)),
        valid_global_batch_size=int(row.get("valid_global_batch_size", row.get("valid_batch_size", 0))),
        valid_forward_micro_batch_size=int(row.get("valid_forward_micro_batch_size", row.get("valid_micro_batch_size", 0))),
        test_per_gpu_batch_size=int(row.get("test_per_gpu_batch_size", 0)),
        test_forward_micro_batch_size=int(row.get("test_forward_micro_batch_size", 0)),
        validation_microbatch_accumulation=bool(row.get("validation_microbatch_accumulation", False)),
        validation_memory_policy=str(row.get("validation_memory_policy", "")),
        step5_validation_mode=str(row.get("step5_validation_mode", "")),
        formal_entry_E4_validation_required=bool(row.get("formal_entry_E4_validation_required", False)),
        old_eval_batch_2048_retired=bool(row.get("old_eval_batch_2048_retired", False)),
        valid_loss_components_json=json_dumps(row.get("valid_loss_components", {})),
        valid_loss_label_max_length=int(row.get("valid_loss_label_max_length", row.get("train_label_max_length", 64))),
        final_eval_prediction_max_length=int(row.get("final_eval_prediction_max_length", 25)),
        final_eval_reference_max_length=int(row.get("final_eval_reference_max_length", 25)),
        num_proc=num_proc,
        ddp_world_size=ddp_world_size,
        seed=int(project.get("seed", 3407)),
        checkpoint_dir=str(run_root),
        log_dir=log_dir,
        iteration_root_dir=str(iteration_root),
        iteration_id=iteration_id,
        manifest_dir=manifest_dir,
        eval_run_dir=eval_run_dir,
        label_smoothing=float(decode["label_smoothing"]),
        repetition_penalty=float(decode["repetition_penalty"]),
        generate_temperature=float(decode["generate_temperature"]),
        generate_top_p=float(decode["generate_top_p"]),
        decode_strategy=decode_strategy,
        decode_seed=decode_seed,
        max_explanation_length=int(decode["max_explanation_length"]),
        train_label_max_length=int(row.get("train_label_max_length", train.get("train_label_max_length", 64))),
        no_repeat_ngram_size=no_repeat,
        min_len=min_len,
        domain_fusion_mode=domain_fusion_mode,
        step3_mode=str(mode or train.get("mode") or "full"),
        step5_train_only=bool(step5_train_only_resolved),
        step5_lifecycle_phase=str(step5_lifecycle_phase or ("eval_only" if command == "eval" else "")),
        step5_allow_embedded_final_eval=bool(step5_allow_embedded_final_eval),
        hardware_preset_id=hw_name,
        decode_preset_id=decode_id,
        num_return_sequences=int(eval_profile_obj.get("num_return_sequences", rerank.get("num_return_sequences", 3) if rerank else 3)),
        rerank_method=str(rerank.get("rerank_method", "rule_v3")),
        rerank_top_k=int(rerank.get("rerank_top_k", 1)),
        rerank_weight_logprob=float(rerank.get("rerank_weight_logprob", 0.35)),
        rerank_weight_length=float(rerank.get("rerank_weight_length", 0.10)),
        rerank_weight_repeat=float(rerank.get("rerank_weight_repeat", 0.16)),
        rerank_weight_dirty=float(rerank.get("rerank_weight_dirty", 0.20)),
        rerank_target_len_ratio=float(rerank.get("rerank_target_len_ratio", 1.10)),
        export_examples_mode=str(rerank.get("export_examples_mode", "head50")),
        export_full_rerank_examples=str(rerank.get("export_examples_mode", "")).lower() == "full",
        rerank_malformed_tail_penalty=float(rerank.get("rerank_malformed_tail_penalty", 0.15)),
        rerank_malformed_token_penalty=float(rerank.get("rerank_malformed_token_penalty", 0.18)),
        decode_profile_json=json_dumps(decode),
        rerank_profile_json=json_dumps(rerank),
        rerank_preset_id=rerank_id,
        hardware_profile_json=json_dumps(hw_semantic),
        optimizer_config_json=json_dumps(step3_optimizer_config),
        precision_config_json=json_dumps(step3_backend_config),
        tokenizer_config_json=json_dumps(step3_tokenizer_config),
        evidence_config_json=json_dumps(step3_evidence_config),
        scheduler_config_json=json_dumps(step3_scheduler_config),
        valid_batch_config_json=json_dumps(step3_eval_config),
        scenario_profile_json=json_dumps(step3_scenario_profile),
        task_profile_config_json=json_dumps(step3_task_profile_config),
        backup_profiles_config_json=json_dumps(step3_backup_profiles_config),
        exploration_profiles_config_json=json_dumps(step3_exploration_profiles_config),
        worker_profiles_config_json=json_dumps(step3_worker_profiles_config),
        prefetcher_config_json=json_dumps(step3_prefetcher_config),
        checkpoint_policy_config_json=json_dumps(step3_checkpoint_policy_config),
        quality_gate_config_json=json_dumps(step3_quality_gate_config),
        grad_finite_config_json=json_dumps(step3_grad_finite_config),
        diagnostic_eval_config_json=json_dumps(step3_diagnostic_eval_config),
        cross_rank_structured_gather_config_json=json_dumps(step3_cross_rank_gather_config),
        memory_config_json=json_dumps(step3_memory_config),
        timing_config_json=json_dumps(step3_timing_config),
        performance_candidates_config_json=json_dumps(step3_performance_candidates_config),
        cache_policy_config_json=json_dumps(step3_cache_policy_config),
        objective_drift_config_json=json_dumps(step3_objective_drift_config),
        recovery_config_json=json_dumps(step3_recovery_config),
        phase_loss_schedule_config_json=json_dumps(step3_phase_loss_schedule_config),
        conflict_aware_config_json=json_dumps(step3_conflict_aware_config),
        loss_gradient_conflict_probe_config_json=json_dumps(step3_loss_gradient_conflict_probe_config),
        adapter_gating_config_json=json_dumps(step3_adapter_gating_config),
        paper_candidate_selection_config_json=json_dumps(step3_paper_candidate_selection_config),
        checkpoint_averaging_config_json=json_dumps(step3_checkpoint_averaging_config),
        omp_num_threads=int(thread_env["OMP_NUM_THREADS"]),
        mkl_num_threads=int(thread_env["MKL_NUM_THREADS"]),
        tokenizers_parallelism=thread_env["TOKENIZERS_PARALLELISM"] == "true",
        thread_env_requested_json=json_dumps(thread_env),
        thread_env_effective_json=json_dumps(thread_env),
        launcher_env_requested_json=json_dumps(launcher_env),
        launcher_env_effective_json=json_dumps(launcher_env),
        training_preset_train_batch_size=batch_size,
        global_eval_batch_size=eval_batch_size,
        eval_per_gpu_batch_size=eval_per_gpu,
        eval_profile_id=eval_profile_name,
        eval_split=eval_split,
        consumed_presets_json=json_dumps(consumed),
        config_before_cli_json=json_dumps({"config_path": str(config_path)}),
        matrix_session_id=None,
        matrix_cell_id=None,
        invoked_command=command,
        resolved_command_kind=command,
        cell_command=None,
        effective_training_payload_json=json_dumps(payload),
        training_semantic_fingerprint=train_fp,
        generation_semantic_fingerprint=gen_fp,
        runtime_diagnostics_fingerprint=runtime_fp,
        config_field_sources_json=json_dumps(field_sources),
        eval_profile_resolution_json=json_dumps(eval_profile_obj),
        upstream_resolution_json=json_dumps(upstream_resolution_payload),
        step4_rcr_config_json=json_dumps(step4_rcr_config),
        step4_runtime_config_json=json_dumps(step4_runtime_config),
        step4_step5_dedicated_exports_config_json=json_dumps(step4_step5_dedicated_exports_config),
        step4_step5_pool_exports_config_json=json_dumps(step4_step5_pool_exports_config),
        step4_gold_quality_config_json=json_dumps(step4_gold_quality_config),
        step4_cf_tiers_config_json=json_dumps(step4_cf_tiers_config),
        step5_innovation_config_json=json_dumps(step5_innovation_config),
        step5_task_decoupled_policy_config_json=json_dumps(step5_task_decoupled_policy_config),
        rating_source_config_json=json_dumps(rating_source_config),
        step5_mode="explanation_only",
        step5_head="explanation",
        lora_target_policy_id=str(row.get("lora_target_policy_id", "")),
        head_specific_lora_allowlist_id=str(row.get("head_specific_lora_allowlist_id", "")),
        final_lora_target_modules=tuple(row.get("final_lora_target_modules", ()) or ()),
        forbidden_lora_targets=tuple(row.get("forbidden_lora_targets", ()) or ()),
        deleted_legacy_modules=tuple(row.get("deleted_legacy_modules", ()) or ()),
        combined_formal_enabled=bool(row.get("retired_combined_formal_enabled", False)),
        all_trainable_grad_required=bool(row.get("all_trainable_grad_required", False)),
        head_specific_trainable_policy=str(row.get("head_specific_trainable_policy", "")),
        head_gated_loss_contract_json=json_dumps(row.get("head_gated_loss_contract", {})),
        step5_selected_tuning_candidate=str(step5_tuning_config.get("selected_tuning_candidate") or "") if stage_for_train == "step5" else "",
        step5_fallback_tuning_candidate=str(step5_tuning_config.get("fallback_tuning_candidate") or "") if stage_for_train == "step5" else "",
        step5_effective_samples_json=json_dumps(step5_tuning_config.get("effective_samples") or {}),
        step5_optimizer_steps_json=json_dumps(step5_tuning_config.get("optimizer_steps") or {}),
        step5_export_loader_config_json=json_dumps(step5_export_loader_config),
        step5_data_pipeline_config_json=json_dumps(step5_data_pipeline_config),
        step5_sampler_config_json=json_dumps(step5_sampler_config),
        step5_prompt_templates_config_json=json_dumps(step5_prompt_templates_config),
        step5_effective_epoch_config_json=json_dumps(step5_effective_epoch_config),
        step5_batch_candidates_config_json=json_dumps(step5_batch_candidates_config),
        step5_tuning_config_json=json_dumps(step5_tuning_config),
        step5_eval_config_json=json_dumps(step5_eval_config),
        step5_e4_bounded_config_json=json_dumps(step5_e4_bounded_config),
        step5_lifecycle_config_json=json_dumps(step5_lifecycle_config),
        step5_memory_truth_config_json=json_dumps(step5_memory_truth_config),
        step5_gradient_checkpointing_reentrant_policy=str(
            step5_memory_truth_config.get("gradient_checkpointing_reentrant_policy", "non_reentrant")
        ),
        ddp_find_unused_parameters=bool(row.get("ddp_find_unused_parameters", False)),
        ddp_find_unused_false_preflight=str(row.get("ddp_find_unused_false_preflight", "real_sample_plan_one_batch")),
        ddp_static_graph=bool(row.get("ddp_static_graph", False)),
        ddp_graph_safety_preflight=bool(row.get("ddp_graph_safety_preflight", True)),
        step3_loss_semantics_json=json_dumps(step3_loss_semantics_config if stage_for_train == "step3" else {}),
        explainer_loss_weight=float(payload["explainer_loss_weight"]),
        data_dir=str(runtime_roots["data_dir"]),
        merged_dir=str(runtime_roots["merged_dir"]),
        runs_dir=str(runtime_roots["runs_dir"]),
        cache_dir=str(runtime_roots["cache_dir"]),
        models_dir=str(runtime_roots["models_dir"]),
        step5_text_model=str(runtime_roots["step5_text_model"]),
        sentence_embed_model=str(runtime_roots["sentence_embed_model"]),
        embed_dim=int(runtime_roots["embed_dim"]),
        offline=bool(runtime_roots["offline"]),
        local_files_only=bool(runtime_roots["local_files_only"]),
        full_bleu_eval_resolved=dict(row.get("full_bleu_eval", {"mode": "off"})),
        full_bleu_decode_strategy=str(row.get("full_bleu_decode_strategy", "inherit")),
        step3_eval_protocol=str(step3_eval_config.get("protocol", MINIMAL_EVAL) if stage_for_train == "step3" else ""),
        step3_eval_split=str(step3_eval_config.get("split", "valid") if stage_for_train == "step3" else ""),
        step3_eval_batch_candidates_json=json_dumps(
            step3_eval_config.get("batch_candidates", []) if stage_for_train == "step3" else []
        ),
        step3_eval_protocol_config_json=json_dumps(
            step3_eval_config.get("protocol_config", {}) if stage_for_train == "step3" else {}
        ),
        train_mode=str(row.get("train_mode", "lora" if stage_for_train == "step5" else "full")),
        train_precision=str(train_precision),
        allow_tf32=bool(row.get("allow_tf32", False)),
        amp_autocast=bool(row.get("amp_autocast", True)),
        grad_scaler=bool(row.get("grad_scaler", False)),
        pin_memory=bool(hw["pin_memory"]),
        persistent_workers=bool(hw["persistent_workers"]),
        non_blocking_h2d=bool(hw["non_blocking_h2d"]),
        per_device_eval_batch_size=int(row.get("per_device_eval_batch_size", 2)),
        lora_r=int(row.get("lora_r", 16)),
        lora_alpha=float(row.get("lora_alpha", 32.0)),
        lora_dropout=float(row.get("lora_dropout", 0.05)),
        lora_target_modules=tuple(row.get("lora_target_modules", ()) or ()),
        nlayers=int(row.get("nlayers", 2)),
        nhead=int(row.get("nhead", 2)),
        nhid=int(row.get("nhid", 2048)),
        dropout=float(row.get("dropout", 0.2)),
    )
    return cfg_obj, sources, resolved_snapshot


def write_resolved_config(cfg: ResolvedConfig, snapshot: Mapping[str, Any], *, dry_run: bool) -> Path | None:
    from odcr_core.manifests import write_resolved_config_artifacts

    if dry_run:
        if getattr(cfg, "command", "") == "step4":
            from odcr_core.step4_runtime import write_step4_dry_run_resolved_artifacts

            return write_step4_dry_run_resolved_artifacts(cfg, snapshot)
        return None

    formal_source_table = str(snapshot.get("train", {}).get("stage") or "") == "step3"
    out, _ = write_resolved_config_artifacts(
        Path(cfg.manifest_dir),
        snapshot,
        formal_only_source_table=formal_source_table,
        write_verbose_source_table=formal_source_table,
    )
    return out


def build_preprocess_config(
    *,
    config_path: str | Path,
    stage_letter: str,
    set_overrides: Iterable[str],
    dry_run: bool,
):
    base = load_yaml_config(config_path)
    cfg, _ = apply_cli_sets(base, set_overrides)
    _validate_config_shape(cfg)
    repo_root = _REPO_ROOT
    runtime_roots = _resolve_global_runtime_roots(cfg, repo_root)
    pp = cfg["preprocess"]
    if not isinstance(pp, Mapping):
        raise OneControlConfigError("preprocess must be a mapping")
    letter = stage_letter.lower()
    if letter not in ("a", "b", "c"):
        raise OneControlConfigError("preprocess stage must be a, b, or c")
    raw = pp.get(letter, {})
    if not isinstance(raw, Mapping):
        raise OneControlConfigError(f"preprocess.{letter} must be a mapping")

    from odcr_core.preprocess_schema import (
        PreprocessAConfig,
        PreprocessBConfig,
        PreprocessCConfig,
        PreprocessHardwareConfig,
        PreprocessPathsConfig,
        PreprocessResolvedPayload,
        PreprocessRuntimeOptions,
        validate_preprocess_config,
    )

    def _preprocess_cache_path(raw_value: Any, *, default_name: str) -> str:
        raw_text = str(raw_value or default_name).strip()
        if not raw_text:
            raise OneControlConfigError("preprocess cache path must be non-empty")
        p = Path(raw_text)
        if p.is_absolute():
            return str(p.resolve())
        parts = p.parts
        if parts and parts[0] == "cache":
            return str((Path(runtime_roots["cache_dir"]) / Path(*parts[1:])).resolve())
        return str((repo_root / p).resolve())

    def _resolved_payload(
        *,
        gpu_ids: tuple[int, ...] = (),
        bf16: bool = False,
        tf32: bool = False,
    ) -> PreprocessResolvedPayload:
        return PreprocessResolvedPayload(
            data_dir=str(runtime_roots["data_dir"]),
            merged_dir=str(runtime_roots["merged_dir"]),
            runs_dir=str(runtime_roots["runs_dir"]),
            cache_dir=str(runtime_roots["cache_dir"]),
            models_dir=str(runtime_roots["models_dir"]),
            step5_text_model=str(runtime_roots["step5_text_model"]),
            sentence_embed_model=str(runtime_roots["sentence_embed_model"]),
            sentence_embed_model_path=str(runtime_roots["sentence_embed_model"]),
            embed_dim=int(runtime_roots["embed_dim"]),
            offline=bool(runtime_roots["offline"]),
            local_files_only=bool(runtime_roots["local_files_only"]),
            gpu_ids=tuple(int(item) for item in gpu_ids),
            bf16=bool(bf16),
            tf32=bool(tf32),
            sources={
                "data_dir": "project.data_dir",
                "merged_dir": "project.merged_dir",
                "runs_dir": "project.run_root",
                "cache_dir": "project.cache_dir",
                "models_dir": "env.models_dir",
                "step5_text_model": "env.step5_text_model",
                "sentence_embed_model": "env.sentence_embed_model",
                "sentence_embed_model_path": "env.sentence_embed_model",
                "embed_dim": "env.embed_dim",
                "offline": "env.offline",
                "local_files_only": "env.local_files_only",
                "gpu_ids": f"preprocess.{letter}.gpu_ids or hardware.preprocess.gpu_ids",
                "bf16": f"preprocess.{letter}.bf16_enabled",
                "tf32": f"preprocess.{letter}.tf32_enabled",
                "tokenizer_parallelism_enabled": f"preprocess.{letter}.tokenizer_parallelism_enabled",
                "tokenizer_threads_per_worker": f"preprocess.{letter}.tokenizer_threads_per_worker",
                "tokenizer_total_threads": f"preprocess.{letter}.tokenizer_total_threads",
                "prefetch_batches": f"preprocess.{letter}.prefetch_batches",
                "pin_memory": f"preprocess.{letter}.pin_memory",
                "non_blocking_h2d": f"preprocess.{letter}.non_blocking_h2d",
                "async_prefetch_enabled": f"preprocess.{letter}.async_prefetch_enabled",
                "cpu_cores_reserved": f"preprocess.{letter}.cpu_cores_reserved",
                "cpu_cores_available": f"preprocess.{letter}.cpu_cores_available",
            },
        )

    datasets = tuple(raw.get("datasets", pp.get("datasets", ())))
    runtime = PreprocessRuntimeOptions(
        python_bin=str(_get(cfg, "env.python_bin", "python")),
        resume=_bool(raw.get("resume", True)),
        skip_completed=_bool(raw.get("skip_completed", True)),
        verify_only=_bool(raw.get("verify_only", False)),
        dry_run=bool(dry_run),
        workers=raw.get("workers"),
        force_datasets=tuple(raw.get("force_datasets", ())),
    )
    if dry_run:
        preprocess_run_id = "dry_run"
    else:
        preprocess_parent = Path(runtime_roots["runs_dir"]) / "preprocess" / letter
        preprocess_parent.mkdir(parents=True, exist_ok=True)
        preprocess_run_id = run_naming.allocate_child_dir(
            preprocess_parent,
            requested="auto",
            kind="run",
        )
    preprocess_run_root = Path(runtime_roots["runs_dir"]) / "preprocess" / letter / preprocess_run_id
    paths = PreprocessPathsConfig(
        meta_root=str((preprocess_run_root / "meta").resolve()),
        shell_log_dir=str((preprocess_run_root / "meta" / "shell_logs").resolve()),
    )
    if letter == "a":
        return validate_preprocess_config(
            PreprocessAConfig(
                preset_name="one_control_preprocess_a",
                description="One-control preprocess A",
                datasets=datasets,
                paths=paths,
                runtime=runtime,
                run_id=preprocess_run_id,
                resolved=_resolved_payload(),
            )
        )
    gpu_ids = tuple(int(x) for x in raw.get("gpu_ids", _get(cfg, "hardware.preprocess.gpu_ids", (0, 1))))
    if letter == "b":
        bf16_enabled = _bool(raw.get("bf16_enabled", True))
        tf32_enabled = _bool(raw.get("tf32_enabled", True))
        max_tokens_per_gpu_batch = raw.get("max_tokens_per_gpu_batch")
        return validate_preprocess_config(
            PreprocessBConfig(
                preset_name="one_control_preprocess_b",
                description="One-control preprocess B",
                datasets=datasets,
                paths=paths,
                runtime=runtime,
                run_id=preprocess_run_id,
                hardware=PreprocessHardwareConfig(gpu_ids=gpu_ids),
                embed_batch_size=int(raw.get("batch_size", 512)),
                read_chunk_rows=int(raw.get("read_chunk_rows", 100_000)),
                group_shard_size=int(raw.get("group_shard_size", 4096)),
                tokenizer_parallelism_enabled=_bool(raw.get("tokenizer_parallelism_enabled", True)),
                tokenizer_threads_per_worker=int(raw.get("tokenizer_threads_per_worker", 4)),
                tokenizer_total_threads=int(raw.get("tokenizer_total_threads", 8)),
                prefetch_batches=int(raw.get("prefetch_batches", 2)),
                pin_memory=_bool(raw.get("pin_memory", True)),
                non_blocking_h2d=_bool(raw.get("non_blocking_h2d", True)),
                async_prefetch_enabled=_bool(raw.get("async_prefetch_enabled", True)),
                token_aware_batching_enabled=_bool(raw.get("token_aware_batching_enabled", False)),
                max_tokens_per_gpu_batch=(
                    None if max_tokens_per_gpu_batch is None else int(max_tokens_per_gpu_batch)
                ),
                cpu_cores_reserved=int(raw.get("cpu_cores_reserved", 2)),
                cpu_cores_available=int(raw.get("cpu_cores_available", 12)),
                grouped_text_cache_enabled=_bool(raw.get("grouped_text_cache_enabled", True)),
                grouped_text_cache_dir=_preprocess_cache_path(
                    raw.get("grouped_text_cache_dir", "cache/preprocess_b"),
                    default_name="cache/preprocess_b",
                ),
                grouped_text_cache_version=str(raw.get("grouped_text_cache_version", "preprocess_b_grouped_text_cache_v1")),
                bf16_enabled=bf16_enabled,
                tf32_enabled=tf32_enabled,
                verify_sample_size=int(raw.get("verify_sample_size", 8)),
                verify_seed=int(raw.get("verify_seed", 7)),
                resolved=_resolved_payload(gpu_ids=gpu_ids, bf16=bf16_enabled, tf32=tf32_enabled),
            )
        )
    bf16_enabled = _bool(raw.get("bf16_enabled", True))
    tf32_enabled = _bool(raw.get("tf32_enabled", True))
    return validate_preprocess_config(
        PreprocessCConfig(
            preset_name="one_control_preprocess_c",
            description="One-control preprocess C",
            datasets=datasets,
            paths=paths,
            runtime=runtime,
            run_id=preprocess_run_id,
            hardware=PreprocessHardwareConfig(gpu_ids=gpu_ids),
            chunk_batch_size=int(raw.get("chunk_batch_size", 512)),
            tokenizer_parallelism_enabled=_bool(raw.get("tokenizer_parallelism_enabled", True)),
            tokenizer_threads_per_worker=int(raw.get("tokenizer_threads_per_worker", 4)),
            tokenizer_total_threads=int(raw.get("tokenizer_total_threads", 8)),
            prefetch_batches=int(raw.get("prefetch_batches", 2)),
            pin_memory=_bool(raw.get("pin_memory", True)),
            non_blocking_h2d=_bool(raw.get("non_blocking_h2d", True)),
            async_prefetch_enabled=_bool(raw.get("async_prefetch_enabled", True)),
            scheduling_policy=str(raw.get("scheduling_policy", "lpt_by_token_windows")),
            cpu_cores_reserved=int(raw.get("cpu_cores_reserved", 2)),
            cpu_cores_available=int(raw.get("cpu_cores_available", 12)),
            bf16_enabled=bf16_enabled,
            tf32_enabled=tf32_enabled,
            tokenizer_hotpath_enabled=_bool(raw.get("tokenizer_hotpath_enabled", True)),
            token_window_cache_enabled=_bool(raw.get("token_window_cache_enabled", True)),
            token_window_cache_dir=_preprocess_cache_path(
                raw.get("token_window_cache_dir", "cache/preprocess_c"),
                default_name="cache/preprocess_c",
            ),
            token_window_cache_version=str(raw.get("token_window_cache_version", "preprocess_c_token_windows_v3")),
            token_window_cache_shard_size=int(raw.get("token_window_cache_shard_size", 4096)),
            resolved=_resolved_payload(gpu_ids=gpu_ids, bf16=bf16_enabled, tf32=tf32_enabled),
        )
    )
