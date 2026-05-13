"""单次运行的复现/排障清单：结构化字段 + 稳定 JSON 路径。"""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from odcr_core.artifacts import train_csv_path
from odcr_core.csb_contract import CSB_ODCR_METHOD_NAME, method_payload
from odcr_core.file_atomic import atomic_write_json
from odcr_core.index_contract import INDEX_CONTRACT_FILENAME
from odcr_core import path_layout
from odcr_core.training_diagnostics import training_diagnostics_snapshot
from odcr_core.generation_semantics import compute_generation_semantic_family_tag
from paths_config import (
    DEFAULT_SENTENCE_EMBED_MODEL_ID,
    DEFAULT_STEP5_TEXT_MODEL_ID,
    get_sentence_embed_model_dir,
    get_step5_text_model_dir,
)

ResolvedConfig = Any

MANIFEST_SCHEMA_VERSION = "4.5"
MANIFEST_FILENAME = "manifest.json"
RESOLVED_CONFIG_FILENAME = "resolved_config.json"
SOURCE_TABLE_FILENAME = "source_table.json"
SOURCE_TABLE_VERBOSE_FILENAME = "source_table_verbose.json"
TRAINING_RUNTIME_CONFIG_FILENAME = "training_runtime_config.json"
RUN_SUMMARY_FILENAME = "run_summary.json"
CONSOLE_LOG_FILENAME = "console.log"
FULL_LOG_FILENAME = "full.log"
DEBUG_LOG_FILENAME = "debug.log"
SAMPLES_LOG_FILENAME = "samples.jsonl"
LATEST_FILENAME = "latest.json"
RUN_SUMMARY_SCHEMA_VERSION = "1.0"
SOURCE_TABLE_SCHEMA_VERSION = "1.0"
FORMAL_VIEW_SCHEMA_VERSION = "odcr_formal_config_view/1"
TRAINING_RUNTIME_CONFIG_SCHEMA_VERSION = "odcr_training_runtime_config/1"
OPTIONAL_ARTIFACT_REASONS = {
    "errors_log": "no_error",
    "debug_log": "debug_disabled",
    "samples_log": "samples_not_requested",
    "training_runtime_config": "optional_missing_with_reason",
}

_STEP3_FORMAL_VIEW_KEYS = (
    "method",
    "task",
    "hardware",
    "train",
    "step3_structured_losses",
    "experiment_profile",
    "step3_csb_odcr",
    "step3_loss_semantics",
    "step3_ddp",
    "step3_task_profile",
    "step3_optimizer",
    "step3_precision",
    "step3_tokenizer",
    "step3_evidence",
    "step3_scheduler",
    "step3_grad_finite",
    "step3_numerical_stability",
    "step3_phase_loss_schedule",
    "step3_eval",
    "step3_worker_profiles",
    "step3_prefetcher",
    "step3_cross_rank_structured_gather",
    "step3_memory",
    "step3_timing",
    "step3_cache_policy",
    "runtime_env",
    "run",
    "active_stage_status",
    "roots",
    "models",
    "embed_dim",
    "offline",
    "local_files_only",
)

_STEP3_FORMAL_SOURCE_EXCLUDE_PARTS = (
    "backup",
    "exploration",
    "step5",
    "decode",
    "rerank",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _repo_relative(repo_root: str | Path, value: str | Path | None) -> str | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    root = Path(repo_root).expanduser().resolve()
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (root / path).resolve()
    else:
        path = path.resolve()
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _artifact_exists(repo_root: Path, value: Any) -> bool:
    if not isinstance(value, (str, Path)):
        return True
    raw = str(value).strip()
    if not raw:
        return False
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    else:
        path = path.resolve()
    return path.exists()


def _artifact_hash(repo_root: Path, value: Any) -> str | None:
    if not isinstance(value, (str, Path)):
        return None
    raw = str(value).strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    else:
        path = path.resolve()
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_text_if_file(path: Path, *, max_chars: int = 2_000_000) -> str:
    if not path.is_file():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    if len(text) > max_chars:
        return text[-max_chars:]
    return text


def _oom_failure_details(
    *,
    text: str,
    latest_error: str | None,
    training_loop_started: bool,
    checkpoint_created: bool,
) -> dict[str, Any] | None:
    combined = "\n".join(part for part in (str(latest_error or ""), text) if part)
    lowered = combined.lower()
    oom_seen = (
        "cuda out of memory" in lowered
        or "torch.outofmemoryerror" in lowered
        or "outofmemoryerror" in lowered
    )
    if not oom_seen:
        return None
    backward_seen = (
        "loss.backward" in lowered
        or "backward oom" in lowered
        or re.search(r"\bbackward\b", lowered) is not None
    )
    torchrun_child = (
        "childfailederror" in lowered
        or "torch.distributed.elastic" in lowered
        or re.search(r"\brank\s*[:=]\s*\d+", combined, flags=re.IGNORECASE) is not None
    )
    epoch_boundary = bool(
        checkpoint_created
        or "after_checkpoint_save" in lowered
        or "after_validation" in lowered
        or "epoch boundary" in lowered
        or "epoch-boundary" in lowered
    )
    failure_phase = "epoch_boundary_backward_oom" if epoch_boundary and backward_seen else "train_backward_oom"
    signature_match = re.search(
        r"(torch\.OutOfMemoryError|CUDA out of memory|loss\.backward\(\)|ChildFailedError)[^\n]*",
        combined,
        flags=re.IGNORECASE,
    )
    return {
        "failure_phase": failure_phase,
        "fatal_source": "torchrun_child_rank_oom" if torchrun_child else "cuda_oom",
        "fatal_signature": signature_match.group(0).strip() if signature_match else "CUDA out of memory during loss.backward",
        "root_cause": "cuda_out_of_memory_during_loss_backward" if backward_seen else "cuda_out_of_memory",
        "oom_detected": True,
        "backward_oom_detected": bool(backward_seen),
        "training_loop_started": bool(training_loop_started),
        "checkpoint_created": bool(checkpoint_created),
    }


def _nonempty_csv(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    text = _read_text_if_file(path, max_chars=32_000)
    return len([line for line in text.splitlines() if line.strip()]) > 1


def _jsonl_has_train_rows(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    for line in _read_text_if_file(path, max_chars=2_000_000).splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if '"split": "train"' in line or '"split":"train"' in line or '"loss_name"' in line:
                return True
            continue
        if not isinstance(row, Mapping):
            continue
        if str(row.get("split") or "").lower() == "train":
            return True
        if row.get("loss_name") and row.get("epoch") is not None:
            return True
    return False


def _training_loop_evidence(meta: Path, text: str, checkpoint_created: bool) -> dict[str, Any]:
    markers = any(token in text for token in ("[Epoch Summary]", "[Train/no_accum]", "n_optimizer_steps="))
    runtime_cfg = meta / TRAINING_RUNTIME_CONFIG_FILENAME
    runtime_started = False
    if runtime_cfg.is_file():
        try:
            runtime_payload = json.loads(runtime_cfg.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            runtime_payload = {}
        if isinstance(runtime_payload, Mapping):
            runtime_started = bool(runtime_payload.get("training_loop_started", False))
    epoch_summary_nonempty = _nonempty_csv(meta / path_layout.metrics_filename("epoch_summary"))
    metrics_train_rows = _jsonl_has_train_rows(meta / path_layout.metrics_filename("metrics"))
    loss_train_rows = _jsonl_has_train_rows(meta / path_layout.metrics_filename("loss_breakdown"))
    return {
        "training_loop_started": bool(
            markers
            or runtime_started
            or checkpoint_created
            or epoch_summary_nonempty
            or metrics_train_rows
            or loss_train_rows
        ),
        "training_marker_seen": bool(markers),
        "training_runtime_config_started": bool(runtime_started),
        "epoch_summary_nonempty": bool(epoch_summary_nonempty),
        "metrics_train_rows": bool(metrics_train_rows),
        "loss_breakdown_train_rows": bool(loss_train_rows),
    }


def _signature_line(pattern: str, text: str, fallback: str) -> str:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return match.group(0).strip() if match else fallback


def _nonfinite_gradient_failure_details(text: str, latest_error: str | None) -> dict[str, Any] | None:
    combined = "\n".join(part for part in (str(latest_error or ""), text) if part)
    lowered = combined.lower()
    if not any(
        token in lowered
        for token in (
            "nonfinite gradient",
            "non-finite gradient",
            "grad finite gate",
            "gradfinite",
            "nonfinite grad",
            "non-finite grad",
        )
    ):
        return None
    signature = _signature_line(
        r"Step3 nonfinite gradient gate aborted after \d+ continuous skipped steps\.?",
        combined,
        "Step3 nonfinite gradient gate aborted after 3 continuous skipped steps",
    )
    return {
        "failure_phase": "training_loop_nonfinite_gradient_gate",
        "fatal_source": "Step3 grad finite gate",
        "fatal_signature": signature,
        "root_cause": "epoch2 post-backward gradient norm explosion / nonfinite grad-norm decision",
        "train_nonfinite_gradient": True,
    }


def _high_grad_norm_failure_details(text: str, latest_error: str | None) -> dict[str, Any] | None:
    combined = "\n".join(part for part in (str(latest_error or ""), text) if part)
    lowered = combined.lower()
    if "high gradient norm gate aborted" not in lowered and "high_grad_norm_abort" not in lowered:
        return None
    signature = _signature_line(
        r"Step3 high gradient norm gate aborted after \d+ continuous skipped steps\.?",
        combined,
        "Step3 high gradient norm gate aborted",
    )
    return {
        "failure_phase": "train_high_grad_norm_abort",
        "fatal_source": "Step3 high finite grad norm gate",
        "fatal_signature": signature,
        "root_cause": "post-backward finite gradient norm exceeded high-grad stability gate",
        "train_high_grad_norm_abort": True,
    }


def _ddp_runtime_failure_details(text: str, latest_error: str | None) -> dict[str, Any] | None:
    combined = "\n".join(part for part in (str(latest_error or ""), text) if part)
    lowered = combined.lower()
    if not any(token in lowered for token in ("childfailederror", "worknccl", "torch.distributed.elastic", "nccl")):
        return None
    return {
        "failure_phase": "train_ddp_runtime_error",
        "fatal_source": "torch.distributed runtime",
        "fatal_signature": _signature_line(
            r"(ChildFailedError|WorkNCCL\([^\n]+|torch\.distributed\.elastic[^\n]+)",
            combined,
            "Step3 DDP runtime error",
        ),
        "root_cause": "distributed training runtime failure after Step3 training started",
    }


def _extract_failure_root_signature(
    *,
    meta: Path,
    latest_error: str | None,
    repo_root: Path,
    checkpoint_path: str | Path | None,
) -> dict[str, Any]:
    text = "\n".join(
        _read_text_if_file(meta / name)
        for name in ("errors.log", FULL_LOG_FILENAME, CONSOLE_LOG_FILENAME, DEBUG_LOG_FILENAME)
    )
    cache_key_match = re.search(r"\[Tokenize\].*?fingerprint=([^|\s]+).*?cache_dir=([^|\n]+)", text)
    nccl_match = re.search(
        r"WorkNCCL\(SeqNum=(?P<seq>\d+),\s*OpType=(?P<op>[^,\)]+),\s*"
        r"NumelIn=(?P<numel_in>\d+),\s*NumelOut=(?P<numel_out>\d+),\s*"
        r"Timeout\(ms\)=(?P<timeout>\d+)\)",
        text,
    )
    rank_match = re.search(r"rank\s*[:=]\s*(\d+).*?local_rank\s*[:=]\s*(\d+)", text, flags=re.IGNORECASE | re.DOTALL)
    progress_matches = re.findall(r"Tokenize \(num_proc=\d+\):\s+\d+%.*?\|\s*(\d+)/(\d+)", text)
    checkpoint_created = bool(checkpoint_path and _artifact_exists(repo_root, checkpoint_path))
    training_evidence = _training_loop_evidence(meta, text, checkpoint_created)
    training_loop_started = bool(training_evidence["training_loop_started"])
    details: dict[str, Any] = {
        "failure_phase": "unknown",
        "fatal_signature": str(latest_error or "").strip(),
        "fatal_source": "latest_error",
        "training_loop_started": training_loop_started,
        "checkpoint_created": checkpoint_created,
        **training_evidence,
    }
    nonfinite_details = _nonfinite_gradient_failure_details(text, latest_error)
    high_grad_details = _high_grad_norm_failure_details(text, latest_error)
    oom_details = _oom_failure_details(
        text=text,
        latest_error=latest_error,
        training_loop_started=training_loop_started,
        checkpoint_created=checkpoint_created,
    )
    ddp_details = _ddp_runtime_failure_details(text, latest_error) if training_loop_started else None
    if training_loop_started and nonfinite_details:
        details.update(nonfinite_details)
    elif training_loop_started and high_grad_details:
        details.update(high_grad_details)
    elif oom_details:
        details.update(oom_details)
    elif training_loop_started and ddp_details:
        details.update(ddp_details)
    elif (not training_loop_started) and ("Tokenize" in text or cache_key_match):
        details["failure_phase"] = "tokenization_cache"
    eval_runtime_seen = any(
        token in text
        for token in (
            "executors/step3_entry.py eval",
            "Step3 eval",
            "Step 3 two-phase eval",
            '"mode": "eval_ddp',
            "eval_ddp_gpu_inference_phase",
        )
    )
    if nccl_match and not (oom_details or nonfinite_details or high_grad_details):
        if training_loop_started:
            details["failure_phase"] = "post_train_eval" if eval_runtime_seen else "train_ddp_runtime_error"
        else:
            details["failure_phase"] = "post_train_eval" if eval_runtime_seen else ("tokenization_cache" if "Tokenize" in text else "ddp_startup")
        details["fatal_source"] = "logs"
        details["fatal_signature"] = nccl_match.group(0)
        details["nccl"] = {
            "seq_num": int(nccl_match.group("seq")),
            "op_type": nccl_match.group("op"),
            "numel_in": int(nccl_match.group("numel_in")),
            "numel_out": int(nccl_match.group("numel_out")),
            "timeout_ms": int(nccl_match.group("timeout")),
        }
    if rank_match:
        details["rank"] = int(rank_match.group(1))
        details["local_rank"] = int(rank_match.group(2))
    if progress_matches:
        cur, total = progress_matches[-1]
        details["tokenization_progress"] = {"current": int(cur), "total": int(total)}
    if cache_key_match:
        details["cache_key"] = cache_key_match.group(1).strip()
        details["cache_dir"] = cache_key_match.group(2).strip()
        cdir = Path(details["cache_dir"]).expanduser()
        details["cache_status"] = "completed" if (cdir / "cache_manifest.json").is_file() else "failed_or_missing"
    startup = meta / "step3_tokenizer_cache_startup.json"
    if startup.is_file():
        try:
            startup_payload = json.loads(startup.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            startup_payload = {}
        if isinstance(startup_payload, Mapping):
            details["cache_status"] = startup_payload.get("status") or details.get("cache_status")
            details["cache_dir"] = startup_payload.get("cache_dir") or details.get("cache_dir")
            details["cache_key"] = startup_payload.get("cache_key") or details.get("cache_key")
            details["startup_cache_payload_path"] = _repo_relative(repo_root, startup)
    return details


def _artifact_optional_record(repo_root: Path, key: str, value: Any, *, latest_error: str | None) -> dict[str, Any]:
    reason = OPTIONAL_ARTIFACT_REASONS.get(key, "missing_optional")
    if key == "errors_log" and latest_error:
        reason = "error_log_not_materialized"
    return {
        "path": _repo_relative(repo_root, value) if isinstance(value, (str, Path)) else value,
        "optional": True,
        "missing_ok": True,
        "reason": reason,
    }


def _duration_seconds(started_at: str | None, finished_at: str | None) -> float | None:
    if not started_at or not finished_at:
        return None
    try:
        start = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        finish = datetime.fromisoformat(finished_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    return max(0.0, round((finish - start).total_seconds(), 3))


def canonical_stage_name(command: str) -> str:
    if command == "eval-rerank":
        return "rerank"
    return str(command)


def resolved_config_path(meta_dir: str | Path) -> Path:
    return Path(meta_dir).expanduser().resolve() / RESOLVED_CONFIG_FILENAME


def source_table_path(meta_dir: str | Path) -> Path:
    return Path(meta_dir).expanduser().resolve() / SOURCE_TABLE_FILENAME


def source_table_verbose_path(meta_dir: str | Path) -> Path:
    return Path(meta_dir).expanduser().resolve() / SOURCE_TABLE_VERBOSE_FILENAME


def training_runtime_config_path(meta_dir: str | Path) -> Path:
    return Path(meta_dir).expanduser().resolve() / TRAINING_RUNTIME_CONFIG_FILENAME


def run_summary_path(meta_dir: str | Path) -> Path:
    return Path(meta_dir).expanduser().resolve() / RUN_SUMMARY_FILENAME


def latest_pointer_path(stage_unit_dir: str | Path) -> Path:
    return Path(stage_unit_dir).expanduser().resolve() / LATEST_FILENAME


def _is_step3_snapshot(snapshot: Mapping[str, Any]) -> bool:
    train = snapshot.get("train")
    if not isinstance(train, Mapping):
        return False
    return str(train.get("stage") or "").strip() == "step3"


def _formal_field_sources(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    raw = snapshot.get("field_sources")
    field_sources = dict(raw) if isinstance(raw, Mapping) else {}
    if not _is_step3_snapshot(snapshot):
        return field_sources
    out: dict[str, Any] = {}
    for key, value in field_sources.items():
        text = f"{key} {value}".lower()
        if any(part in text for part in _STEP3_FORMAL_SOURCE_EXCLUDE_PARTS):
            continue
        out[key] = value
    return out


def _strip_formal_probe_markers(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _strip_formal_probe_markers(item)
            for key, item in value.items()
            if key != "probe_only"
        }
    if isinstance(value, list):
        return [_strip_formal_probe_markers(item) for item in value]
    return value


def formal_snapshot_view(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Return the default user-facing formal view for ``odcr show`` and dry-run."""
    if not _is_step3_snapshot(snapshot):
        return dict(snapshot)
    out: dict[str, Any] = {
        "view_schema_version": FORMAL_VIEW_SCHEMA_VERSION,
        "view": "formal",
    }
    for key in _STEP3_FORMAL_VIEW_KEYS:
        if key in snapshot and snapshot.get(key) is not None:
            if key == "models" and isinstance(snapshot.get(key), Mapping):
                models = dict(snapshot.get(key) or {})
                source = models.get("source") if isinstance(models.get("source"), Mapping) else {}
                out[key] = {
                    "sentence_embed_model": models.get("sentence_embed_model"),
                    "source": {"sentence_embed_model": source.get("sentence_embed_model")},
                }
            elif key == "step3_task_profile" and isinstance(snapshot.get(key), Mapping):
                profile = dict(snapshot.get(key) or {})
                profile.pop("formal_allowed", None)
                profile.pop("probe_only", None)
                out[key] = _strip_formal_probe_markers(profile)
            elif key == "step3_cache_policy" and isinstance(snapshot.get(key), Mapping):
                cache_policy = dict(snapshot.get(key) or {})
                cache_policy.pop("probe_cache_namespace", None)
                out[key] = _strip_formal_probe_markers(cache_policy)
            else:
                out[key] = _strip_formal_probe_markers(snapshot.get(key))
    out["field_sources"] = _formal_field_sources(snapshot)
    return out


def build_source_table_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    raw = snapshot.get("field_sources")
    field_sources = dict(raw) if isinstance(raw, Mapping) else {}
    return {
        "source_table_schema_version": SOURCE_TABLE_SCHEMA_VERSION,
        "view": "verbose",
        "generated_at_utc": _utc_now(),
        "field_sources": field_sources,
        "records": [
            {"key": str(key), "source": value}
            for key, value in sorted(field_sources.items(), key=lambda item: str(item[0]))
        ],
    }


def build_formal_source_table_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    field_sources = _formal_field_sources(snapshot)
    return {
        "source_table_schema_version": SOURCE_TABLE_SCHEMA_VERSION,
        "view": "formal",
        "generated_at_utc": _utc_now(),
        "field_sources": field_sources,
        "records": [
            {"key": str(key), "source": value}
            for key, value in sorted(field_sources.items(), key=lambda item: str(item[0]))
        ],
    }


def write_resolved_config_artifacts(
    meta_dir: str | Path,
    snapshot: Mapping[str, Any],
    *,
    source_table: Mapping[str, Any] | None = None,
    formal_only_source_table: bool = False,
    write_verbose_source_table: bool = False,
) -> tuple[Path, Path]:
    meta = Path(meta_dir).expanduser().resolve()
    config_path = resolved_config_path(meta)
    source_path = source_table_path(meta)
    if source_table is not None:
        source_payload = dict(source_table)
    elif formal_only_source_table:
        source_payload = build_formal_source_table_snapshot(snapshot)
    else:
        source_payload = build_source_table_snapshot(snapshot)
    atomic_write_json(config_path, dict(snapshot))
    atomic_write_json(source_path, source_payload)
    if write_verbose_source_table and formal_only_source_table:
        atomic_write_json(source_table_verbose_path(meta), build_source_table_snapshot(snapshot))
    return config_path, source_path


def write_training_runtime_config_artifact(
    meta_dir: str | Path,
    runtime_snapshot: Mapping[str, Any],
) -> Path:
    meta = Path(meta_dir).expanduser().resolve()
    payload = dict(runtime_snapshot)
    payload.setdefault("training_runtime_config_schema_version", TRAINING_RUNTIME_CONFIG_SCHEMA_VERSION)
    payload.setdefault("generated_at_utc", _utc_now())
    out = training_runtime_config_path(meta)
    atomic_write_json(out, payload)
    return out


def build_run_summary(
    *,
    repo_root: str | Path,
    run_dir: str | Path,
    meta_dir: str | Path,
    run_id: str,
    stage: str,
    status: str,
    started_at: str,
    finished_at: str | None = None,
    command: str | None = None,
    task_id: int | None = None,
    unit: str | None = None,
    source_domain: str | None = None,
    target_domain: str | None = None,
    console_log_path: str | Path | None = None,
    full_log_path: str | Path | None = None,
    errors_log_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
    lineage_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    key_artifacts: Mapping[str, Any] | None = None,
    latest_error: str | None = None,
    validation_status: str | None = None,
    post_edit_scope: str | None = None,
) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve()
    meta = Path(meta_dir).expanduser().resolve()
    run_root = Path(run_dir).expanduser().resolve()
    if manifest_path is None:
        manifest_path = meta / MANIFEST_FILENAME
    if console_log_path is None:
        console_log_path = meta / CONSOLE_LOG_FILENAME
    if full_log_path is None:
        full_log_path = meta / FULL_LOG_FILENAME
    if errors_log_path is None:
        errors_log_path = meta / "errors.log"
    artifact_map: dict[str, Any] = {
        "console_log": console_log_path,
        "full_log": full_log_path,
        "errors_log": errors_log_path,
        "debug_log": meta / DEBUG_LOG_FILENAME,
        "samples_log": meta / SAMPLES_LOG_FILENAME,
        "training_runtime_config": training_runtime_config_path(meta),
    }
    artifact_map.update(dict(key_artifacts or {}))
    key_artifact_payload: dict[str, Any] = {}
    optional_artifact_payload: dict[str, Any] = {}
    for key, value in artifact_map.items():
        if key in OPTIONAL_ARTIFACT_REASONS and not _artifact_exists(root, value):
            optional_artifact_payload[str(key)] = _artifact_optional_record(
                root,
                str(key),
                value,
                latest_error=latest_error,
            )
            continue
        key_artifact_payload[str(key)] = (
            _repo_relative(root, value) if isinstance(value, (str, Path)) else value
        )
    runtime_config = training_runtime_config_path(meta)
    runtime_config_exists = runtime_config.is_file()
    runtime_config_rel = _repo_relative(root, runtime_config) if runtime_config_exists else None
    runtime_config_hash = _artifact_hash(root, runtime_config) if runtime_config_exists else None
    payload = {
        "run_summary_schema_version": RUN_SUMMARY_SCHEMA_VERSION,
        "run_id": str(run_id),
        "stage": canonical_stage_name(stage),
        "task_id": task_id,
        "unit": unit,
        "source_domain": source_domain,
        "target_domain": target_domain,
        "status": status,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_sec": _duration_seconds(started_at, finished_at),
        "command": command,
        "run_dir": _repo_relative(root, run_root),
        "meta_dir": _repo_relative(root, meta),
        "resolved_config_path": _repo_relative(root, resolved_config_path(meta)),
        "resolved_config_hash": _artifact_hash(root, resolved_config_path(meta)),
        "training_runtime_config_path": runtime_config_rel,
        "training_runtime_config_hash": runtime_config_hash,
        "source_table_path": _repo_relative(root, source_table_path(meta)),
        "source_table_hash": _artifact_hash(root, source_table_path(meta)),
        "source_table_verbose_path": _repo_relative(root, source_table_verbose_path(meta)),
        "console_log_path": _repo_relative(root, console_log_path),
        "full_log_path": _repo_relative(root, full_log_path),
        "authoritative_full_log_path": _repo_relative(root, full_log_path),
        "errors_log_path": _repo_relative(root, errors_log_path),
        "debug_log_path": _repo_relative(root, meta / DEBUG_LOG_FILENAME),
        "metrics_path": _repo_relative(root, metrics_path),
        "lineage_path": _repo_relative(root, lineage_path),
        "manifest_path": _repo_relative(root, manifest_path),
        "key_artifacts": key_artifact_payload,
        "optional_artifacts": optional_artifact_payload,
        "latest_error": latest_error,
        "validation_status": validation_status,
        "post_edit_scope": post_edit_scope,
    }
    if canonical_stage_name(stage) == "step3":
        payload.update(
            {
                "quality_status": "not_evaluated",
                "readiness_status": "not_evaluated",
                "downstream_ready": False,
                "quality_block_reasons": [],
                "quality_warnings": [],
                "quality_gate_version": "odcr_step3_upstream_readiness_gate/1",
                "quality_gate_inputs": {},
                "paper_metric_gate": False,
                "paper_metrics_excluded_from_readiness": ["BLEU", "ROUGE", "DIST", "METEOR", "paper_target_only_eval"],
                "selected_downstream_checkpoint": None,
                "selected_downstream_checkpoint_scope": None,
                "selected_downstream_checkpoint_epoch": None,
                "selected_downstream_checkpoint_metric": None,
                "runtime_evidence": {
                    "code_present": True,
                    "active_path": False,
                    "runtime_verified": False,
                    "formal_verified": False,
                },
            }
        )
    if str(status).lower() in {"failed", "partial", "interrupted"}:
        checkpoint_value = key_artifacts.get("model") if isinstance(key_artifacts, Mapping) else None
        failure = _extract_failure_root_signature(
            meta=meta,
            latest_error=latest_error,
            repo_root=root,
            checkpoint_path=checkpoint_value,
        )
        payload["failure_root_signature"] = failure
        payload["failure_phase"] = failure.get("failure_phase")
        payload["fatal_signature"] = failure.get("fatal_signature")
        payload["training_loop_started"] = failure.get("training_loop_started")
        payload["checkpoint_created"] = failure.get("checkpoint_created")
        if failure.get("cache_status") is not None:
            payload["cache_status"] = failure.get("cache_status")
        if failure.get("cache_dir") is not None:
            payload["cache_dir"] = failure.get("cache_dir")
        if failure.get("cache_key") is not None:
            payload["cache_key"] = failure.get("cache_key")
    return payload


def _read_step3_eval_status_sidecar(meta: Path) -> dict[str, Any]:
    path = meta / "step3_eval_status.json"
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def _apply_step3_train_eval_status_split(
    payload: dict[str, Any],
    *,
    cfg: ResolvedConfig,
    meta: Path,
    status: str,
    latest_error: str | None,
    key_artifacts: Mapping[str, Any],
) -> None:
    readiness_path = meta / "readiness_audit.json"
    readiness = {}
    if readiness_path.is_file():
        try:
            readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
        except Exception:
            readiness = {}
    if isinstance(readiness, Mapping) and readiness:
        payload["readiness_audit"] = _repo_relative(Path(cfg.repo_root), readiness_path)
        payload["quality_status"] = str(readiness.get("quality_status") or readiness.get("readiness_status") or "")
        payload["readiness_status"] = str(readiness.get("readiness_status") or readiness.get("quality_status") or "")
        payload["downstream_ready"] = bool(readiness.get("downstream_ready") is True)
        payload["status"] = "step4_ready" if payload["downstream_ready"] else payload.get("status")
        payload["stage_status"] = "step4_ready" if payload["downstream_ready"] else "not_ready"
        payload["paper_metric_gate"] = False
        payload["paper_metrics_excluded_from_readiness"] = list(
            readiness.get("paper_metrics_excluded_from_readiness") or ["BLEU", "ROUGE", "DIST", "METEOR"]
        )
        for key in (
            "selected_downstream_checkpoint",
            "selected_downstream_checkpoint_hash",
            "selected_downstream_checkpoint_scope",
            "selected_downstream_checkpoint_epoch",
            "selected_downstream_checkpoint_metric",
        ):
            if key in readiness:
                payload[key] = readiness[key]
        payload["selected_checkpoint"] = readiness.get("selected_downstream_checkpoint") or payload.get("selected_checkpoint")
        payload["selected_checkpoint_hash"] = readiness.get("selected_downstream_checkpoint_hash") or payload.get("selected_checkpoint_hash")
        payload["status_source"] = "step3_upstream_readiness_gate"
        return
    sidecar = _read_step3_eval_status_sidecar(meta)
    if sidecar:
        for key in (
            "train_status",
            "eval_status",
            "quality_status",
            "downstream_ready",
            "failure_phase",
            "eval_protocol",
            "eval_scope",
            "paper_comparable",
            "selected_checkpoint",
            "selected_checkpoint_scope",
        ):
            if key in sidecar:
                payload[key] = sidecar[key]
        payload["step3_eval_status_sidecar"] = _repo_relative(Path(cfg.repo_root), meta / "step3_eval_status.json")
        return

    mode = str(getattr(cfg, "step3_mode", "") or "full")
    status_l = str(status or "").lower()
    checkpoint_value = key_artifacts.get("model") if isinstance(key_artifacts, Mapping) else None
    checkpoint_exists = bool(checkpoint_value and _artifact_exists(Path(cfg.repo_root), checkpoint_value))
    failure_phase = str(payload.get("failure_phase") or "")
    err = str(latest_error or "")
    eval_failure = (
        failure_phase == "post_train_eval"
        or ("step3_entry.py eval" in err)
        or (" eval " in err and checkpoint_exists)
    )
    payload.setdefault("eval_protocol", getattr(cfg, "step3_eval_protocol", "") or "minimal_eval")
    payload.setdefault("eval_scope", getattr(cfg, "step3_eval_split", "") or "valid")
    payload.setdefault("paper_comparable", False)
    payload.setdefault("selected_checkpoint", _repo_relative(Path(cfg.repo_root), checkpoint_value) if checkpoint_value else None)
    payload.setdefault("selected_checkpoint_scope", "best_observed")
    if status_l == "ok":
        if mode == "train_only":
            payload.update({"train_status": "completed", "eval_status": "not_requested"})
        elif mode == "eval_only":
            payload.update({"train_status": "completed" if checkpoint_exists else "not_run", "eval_status": "completed"})
        else:
            payload.update({"train_status": "completed", "eval_status": "completed"})
    elif status_l in {"failed", "partial", "interrupted"} and eval_failure:
        payload.update(
            {
                "train_status": "completed",
                "eval_status": "failed",
                "quality_status": "not_evaluated",
                "downstream_ready": False,
                "failure_phase": "post_train_eval",
            }
        )
    elif status_l in {"failed", "partial", "interrupted"}:
        payload.update({"train_status": "failed", "eval_status": "not_started"})
    else:
        payload.update({"train_status": "running", "eval_status": "pending" if mode == "full" else "not_requested"})


def write_latest_pointer_json(
    *,
    repo_root: str | Path,
    stage_unit_dir: str | Path,
    run_id: str,
    run_dir: str | Path,
    summary_path: str | Path,
    status: str,
    updated_at: str | None = None,
) -> Path:
    root = Path(repo_root).expanduser().resolve()
    payload = {
        "schema_version": "odcr_latest_pointer/active_stage_status/1",
        "active_run_id": str(run_id),
        "latest_run_id": str(run_id),
        "latest_run_dir": _repo_relative(root, run_dir),
        "latest_summary_path": _repo_relative(root, summary_path),
        "latest_stage_status_path": _repo_relative(root, Path(run_dir) / "meta" / "stage_status.json"),
        "updated_at": updated_at or _utc_now(),
        "status_claim_source": "stage_status_strict_verifier",
    }
    _ = status
    return atomic_write_json(latest_pointer_path(stage_unit_dir), payload)


def write_run_summary_json(
    summary: Mapping[str, Any],
    *,
    repo_root: str | Path,
    update_latest: bool = True,
) -> Path:
    root = Path(repo_root).expanduser().resolve()
    meta_dir_value = summary.get("meta_dir")
    if not meta_dir_value:
        raise ValueError("run_summary requires meta_dir")
    meta = Path(str(meta_dir_value))
    if not meta.is_absolute():
        meta = (root / meta).resolve()
    run_dir_value = summary.get("run_dir")
    run_dir = None
    if run_dir_value:
        run_dir = Path(str(run_dir_value))
        if not run_dir.is_absolute():
            run_dir = (root / run_dir).resolve()
    out = run_summary_path(meta)
    atomic_write_json(out, dict(summary))
    stage = canonical_stage_name(str(summary.get("stage") or ""))
    stage_status_payload = None
    if stage in {"step3", "step4", "step5"} and summary.get("task_id") is not None and run_dir is not None:
        from odcr_core.stage_status import build_and_write_stage_status

        stage_status_payload = build_and_write_stage_status(
            repo_root=root,
            stage=stage,
            task=int(summary.get("task_id")),
            run_id=str(summary.get("run_id") or run_dir.name),
        )
        if stage == "step4" and stage_status_payload.get("downstream_ready") is not True:
            update_latest = False
    if update_latest:
        if run_dir is None:
            raise ValueError("run_summary requires run_dir to update latest.json")
        write_latest_pointer_json(
            repo_root=root,
            stage_unit_dir=run_dir.parent,
            run_id=str(summary.get("run_id") or run_dir.name),
            run_dir=run_dir,
            summary_path=out,
            status=str(summary.get("status") or "pending"),
        )
    return out


def _run_id_for_config(cfg: ResolvedConfig) -> str:
    if cfg.command == "step3" and cfg.run_name is not None:
        return str(cfg.run_name)
    if cfg.command == "step4" and cfg.step4_run is not None:
        return str(cfg.step4_run)
    if cfg.command in ("step5", "eval", "eval-rerank") and cfg.step5_run is not None:
        if cfg.command in ("eval", "eval-rerank") and cfg.eval_run_dir:
            return Path(cfg.eval_run_dir).name
        return str(cfg.step5_run)
    if cfg.eval_run_dir:
        return Path(cfg.eval_run_dir).name
    return Path(cfg.checkpoint_dir).name


def _run_dir_for_config(cfg: ResolvedConfig) -> Path:
    if cfg.command in ("eval", "eval-rerank") and cfg.eval_run_dir:
        return Path(cfg.eval_run_dir).expanduser().resolve()
    return Path(cfg.checkpoint_dir).expanduser().resolve()


def _primary_log_for_config(cfg: ResolvedConfig) -> Path:
    return Path(cfg.manifest_dir).expanduser().resolve() / FULL_LOG_FILENAME


def _console_log_for_config(cfg: ResolvedConfig) -> Path:
    return Path(cfg.manifest_dir).expanduser().resolve() / CONSOLE_LOG_FILENAME


def build_run_summary_for_config(
    cfg: ResolvedConfig,
    *,
    status: str,
    started_at: str,
    finished_at: str | None = None,
    command: str | None = None,
    latest_error: str | None = None,
    validation_status: str | None = None,
    post_edit_scope: str | None = None,
) -> dict[str, Any]:
    run_dir = _run_dir_for_config(cfg)
    meta = Path(cfg.manifest_dir).expanduser().resolve()
    stage = canonical_stage_name(cfg.command)
    if stage in ("eval", "rerank"):
        metrics_path = path_layout.eval_metrics_path(run_dir, rerank=(stage == "rerank"))
    else:
        metrics_path = meta / path_layout.metrics_filename("metrics")
    key_artifacts: dict[str, Any] = {
        "manifest": meta / MANIFEST_FILENAME,
        "resolved_config": resolved_config_path(meta),
        "training_runtime_config": training_runtime_config_path(meta),
        "source_table": source_table_path(meta),
        "console_log": meta / CONSOLE_LOG_FILENAME,
        "full_log": meta / FULL_LOG_FILENAME,
        "authoritative_full_log": meta / FULL_LOG_FILENAME,
        "debug_log": meta / DEBUG_LOG_FILENAME,
        "samples_log": meta / SAMPLES_LOG_FILENAME,
    }
    if cfg.command in ("step3", "step5", "eval", "eval-rerank"):
        key_artifacts["model"] = path_layout.best_model_path(Path(cfg.checkpoint_dir))
    if cfg.command == "step3":
        key_artifacts["source_table_verbose"] = source_table_verbose_path(meta)
        key_artifacts["metrics"] = meta / path_layout.metrics_filename("metrics")
        key_artifacts["loss_breakdown"] = meta / path_layout.metrics_filename("loss_breakdown")
        key_artifacts["timing_profile"] = meta / path_layout.metrics_filename("timing_profile")
        key_artifacts["gpu_profile"] = meta / path_layout.metrics_filename("gpu_profile")
        key_artifacts["epoch_summary"] = meta / path_layout.metrics_filename("epoch_summary")
        key_artifacts["readiness_audit"] = meta / "readiness_audit.json"
    if cfg.command in ("step4", "step5"):
        try:
            key_artifacts["training_csv"] = train_csv_path(cfg)
        except Exception:
            pass
    if stage in ("eval", "rerank"):
        key_artifacts["metrics"] = metrics_path
    payload = build_run_summary(
        repo_root=cfg.repo_root,
        run_dir=run_dir,
        meta_dir=meta,
        run_id=_run_id_for_config(cfg),
        stage=stage,
        task_id=int(cfg.task_id),
        unit=None,
        source_domain=str(getattr(cfg, "auxiliary", "") or "") or None,
        target_domain=str(getattr(cfg, "target", "") or "") or None,
        status=status,
        started_at=started_at,
        finished_at=finished_at,
        command=command,
        console_log_path=_console_log_for_config(cfg),
        full_log_path=_primary_log_for_config(cfg),
        errors_log_path=meta / "errors.log",
        metrics_path=metrics_path,
        lineage_path=path_layout.state_dir(Path(cfg.checkpoint_dir)) / "checkpoint_lineage.json",
        manifest_path=meta / MANIFEST_FILENAME,
        key_artifacts=key_artifacts,
        latest_error=latest_error,
        validation_status=validation_status,
        post_edit_scope=post_edit_scope,
    )
    method_config: dict[str, Any] = {}
    try:
        method_config = json.loads(str(getattr(cfg, "method_config_json", "") or "{}"))
    except json.JSONDecodeError:
        method_config = {}
    if not method_config:
        method_config = method_payload()
    payload["method_name"] = str(method_config.get("method_name") or CSB_ODCR_METHOD_NAME)
    payload["method"] = dict(method_config)
    payload["method_schema"] = str(method_config.get("schema_version") or method_payload()["schema_version"])
    payload["method_schema_hash"] = str(method_config.get("method_schema_hash") or method_payload()["method_schema_hash"])
    if cfg.command == "step3":
        try:
            experiment_profile = json.loads(str(getattr(cfg, "experiment_profile_config_json", "") or "{}"))
        except json.JSONDecodeError:
            experiment_profile = {}
        try:
            csb_config = json.loads(str(getattr(cfg, "csb_odcr_config_json", "") or "{}"))
        except json.JSONDecodeError:
            csb_config = {}
        payload["experiment_profile"] = str(experiment_profile.get("name") or getattr(cfg, "experiment_profile", ""))
        payload["ablation_profile"] = str(
            experiment_profile.get("ablation_profile") or getattr(cfg, "ablation_profile", "")
        )
        payload["csb_contract"] = dict(csb_config.get("contract") or {})
        payload["csb_contract_hash"] = str((csb_config.get("contract") or {}).get("contract_hash") or "")
        try:
            memory_config = json.loads(str(getattr(cfg, "memory_config_json", "") or "{}"))
        except json.JSONDecodeError:
            memory_config = {}
        allocator = memory_config.get("allocator") if isinstance(memory_config, Mapping) else {}
        if isinstance(allocator, Mapping):
            payload["allocator"] = dict(allocator)
        try:
            grad_finite = json.loads(str(getattr(cfg, "grad_finite_config_json", "") or "{}"))
        except json.JSONDecodeError:
            grad_finite = {}
        if isinstance(grad_finite, Mapping):
            payload["grad_finite"] = dict(grad_finite)
        try:
            numerical_stability = json.loads(str(getattr(cfg, "numerical_stability_config_json", "") or "{}"))
        except json.JSONDecodeError:
            numerical_stability = {}
        if isinstance(numerical_stability, Mapping):
            payload["numerical_stability"] = dict(numerical_stability)
        try:
            payload["launcher_env_effective"] = json.loads(str(getattr(cfg, "launcher_env_effective_json", "") or "{}"))
        except json.JSONDecodeError:
            payload["launcher_env_effective"] = {}
    if cfg.command == "step3":
        _apply_step3_train_eval_status_split(
            payload,
            cfg=cfg,
            meta=meta,
            status=status,
            latest_error=latest_error,
            key_artifacts=key_artifacts,
        )
    if cfg.command in {"step4", "step5", "eval", "eval-rerank"}:
        if cfg.from_run is not None:
            payload["from_step3"] = cfg.from_run
        if cfg.step4_run is not None:
            payload["from_step4"] = cfg.step4_run
        if cfg.step5_run is not None:
            payload["from_step5"] = cfg.step5_run
        upstream_resolution = getattr(cfg, "upstream_resolution_json", "") or ""
        if upstream_resolution.strip():
            try:
                payload["upstream_resolution"] = json.loads(upstream_resolution)
            except json.JSONDecodeError:
                payload["upstream_resolution"] = {"unparsed": upstream_resolution}
    return payload


def write_run_summary_for_config(
    cfg: ResolvedConfig,
    *,
    status: str,
    started_at: str,
    finished_at: str | None = None,
    command: str | None = None,
    latest_error: str | None = None,
    validation_status: str | None = None,
    post_edit_scope: str | None = None,
) -> Path:
    summary = build_run_summary_for_config(
        cfg,
        status=status,
        started_at=started_at,
        finished_at=finished_at,
        command=command,
        latest_error=latest_error,
        validation_status=validation_status,
        post_edit_scope=post_edit_scope,
    )
    return write_run_summary_json(summary, repo_root=cfg.repo_root, update_latest=True)


def _stage_label(command: str) -> str:
    return {
        "step3": "step3_structured_disentanglement",
        "step4": "step4_counterfactual_eval_inference",
        "step5": "step5_main_train",
        "eval": "eval_step5_valid",
        "eval-rerank": "eval_step5_valid_rerank",
    }.get(command, command)


def _resolved_train_csv(cfg: ResolvedConfig) -> str | None:
    if cfg.command == "step4":
        if not cfg.from_run:
            return None
        return str(train_csv_path(cfg).resolve())
    if cfg.command == "step5" and cfg.from_run and cfg.step5_run:
        return str(train_csv_path(cfg).resolve())
    if cfg.command in ("eval", "eval-rerank") and cfg.step5_run:
        return str(train_csv_path(cfg).resolve())
    return None


def _resolved_model_weights(cfg: ResolvedConfig) -> str | None:
    if cfg.model_path:
        return str(Path(cfg.model_path).resolve())
    if cfg.command in ("step5", "eval", "eval-rerank"):
        ck = Path(cfg.checkpoint_dir)
        return str(path_layout.best_model_path(ck))
    return None


def _training_row_slice_for_manifest(cfg: ResolvedConfig) -> dict[str, Any]:
    """从 effective payload 取出主线损失权重片段（与 torchrun 子进程所见一致）。"""
    raw = (getattr(cfg, "effective_training_payload_json", "") or "").strip()
    if not raw:
        return {}
    try:
        p = json.loads(raw)
        row = p.get("training_row")
        if not isinstance(row, dict):
            return {}
    except json.JSONDecodeError:
        return {}
    # 仅保留 ODCR 主线可解释损失权重；Step3 不暴露 retired adversarial controls。
    keys = (
        "lambda_ortho",
        "lambda_ortho_xcov",
        "lambda_ortho_cos",
        "lambda_ortho_step5",
        "step5_lci_weight",
        "step5_fca_weight",
    )
    return {k: row[k] for k in keys if k in row}


def _manifest_backbones_block(cfg: ResolvedConfig) -> dict[str, Any]:
    """运行时 backbone 条件（与 index_contract 的「数据/表征」块互补）。"""
    raw_embed_dim = getattr(cfg, "embed_dim", None)
    if raw_embed_dim is None:
        raise RuntimeError(
            "manifest backbones hidden_size requires resolved cfg.embed_dim from One-Control; "
            "bare ODCR_* env and default fallbacks are not allowed."
        )
    try:
        hid = int(raw_embed_dim)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "manifest backbones hidden_size requires positive integer cfg.embed_dim from One-Control."
        ) from exc
    if hid <= 0:
        raise RuntimeError(
            "manifest backbones hidden_size requires positive integer cfg.embed_dim from One-Control."
        )
    return {
        "sentence_embed": {
            "model_id": DEFAULT_SENTENCE_EMBED_MODEL_ID,
            "local_dir": str(Path(get_sentence_embed_model_dir()).resolve()),
            "family": "bge_large_en",
            "hidden_size": hid,
            "dual_channel": True,
            "load_policy": "local_files_only_true_require_dir",
        },
        "text_model": {
            "model_id": DEFAULT_STEP5_TEXT_MODEL_ID,
            "local_dir": str(Path(get_step5_text_model_dir()).resolve()),
            "family": "t5_tokenized_explainer_stack",
            "tokenizer_id": DEFAULT_STEP5_TEXT_MODEL_ID,
            "train_mode": str(getattr(cfg, "train_mode", "full")),
            "load_policy": "local_files_only_true_require_dir",
        },
    }


def _manifest_training_runtime_block(cfg: ResolvedConfig) -> dict[str, Any]:
    out = {
        "precision": str(getattr(cfg, "train_precision", "bf16")),
        "batch_semantics_version": str(getattr(cfg, "batch_semantics_version", "odcr_no_accum/1")),
        "grad_accum_removed": bool(getattr(cfg, "grad_accum_removed", True)),
        "global_batch_size": int(getattr(cfg, "global_batch_size", cfg.train_batch_size)),
        "per_gpu_batch_size": int(getattr(cfg, "per_gpu_batch_size", cfg.per_device_train_batch_size)),
        "per_device_train_batch_size": int(cfg.per_device_train_batch_size),
        "per_device_eval_batch_size": int(getattr(cfg, "per_device_eval_batch_size", 2)),
        "effective_batch_size": int(cfg.effective_global_batch_size),
        "ddp_world_size": int(cfg.ddp_world_size),
    }
    if cfg.command == "step3":
        out["step3_batch_semantics"] = "odcr_no_accum/1"
        try:
            memory_cfg = json.loads(str(getattr(cfg, "memory_config_json", "") or "{}"))
        except json.JSONDecodeError:
            memory_cfg = {}
        allocator = memory_cfg.get("allocator") if isinstance(memory_cfg, Mapping) else {}
        if isinstance(allocator, Mapping):
            out["allocator"] = dict(allocator)
    return out


def _manifest_peft_block(cfg: ResolvedConfig) -> dict[str, Any]:
    tm = str(getattr(cfg, "train_mode", "full")).strip().lower()
    lmods = list(getattr(cfg, "lora_target_modules", ()) or ())
    base = {
        "r": int(getattr(cfg, "lora_r", 16)),
        "alpha": float(getattr(cfg, "lora_alpha", 32.0)),
        "dropout": float(getattr(cfg, "lora_dropout", 0.05)),
        "target_modules": lmods if lmods else None,
    }
    if tm == "lora":
        return {
            "enabled": True,
            "type": "lora",
            "implementation": "odcr_native_linear",
            **base,
        }
    return {
        "enabled": False,
        "type": "none",
        "implementation": "",
        **base,
    }


def _run_lineage(cfg: ResolvedConfig) -> dict[str, Any]:
    """task/iter 与各 stage slug，供实验组脚本单点读取。"""
    out: dict[str, Any] = {
        "task_id": cfg.task_id,
        "iteration_id": cfg.iteration_id,
    }
    if cfg.run_name is not None:
        out["step3_run"] = cfg.run_name
    if cfg.from_run is not None:
        out["step3_run"] = cfg.from_run
    if cfg.step4_run:
        out["step4_run"] = cfg.step4_run
    if cfg.step5_run:
        out["step5_run"] = cfg.step5_run
    if cfg.eval_run_dir:
        er = Path(cfg.eval_run_dir)
        out["eval_run"] = er.name
        out["eval_run_dir"] = str(er.resolve())
        out["metrics_path"] = str(
            path_layout.eval_metrics_path(er, rerank=(cfg.command == "eval-rerank")).resolve()
        )
    if cfg.command == "eval-rerank" and cfg.eval_run_dir:
        out["rerank_run"] = Path(cfg.eval_run_dir).name
        out["rerank_run_dir"] = str(Path(cfg.eval_run_dir).resolve())
    dr = getattr(cfg, "decode_preset_id", "") or ""
    if dr:
        out["decode_preset_id"] = dr
    if cfg.command == "eval-rerank" and cfg.rerank_preset_id:
        out["rerank_preset_id"] = cfg.rerank_preset_id
    return out


def build_run_manifest(cfg: ResolvedConfig, *, cli_invocation: str | None = None) -> dict[str, Any]:
    """
    供 stdout 摘要、JSON 落盘与外部工具解析。
    字段以结构化嵌套为主（manifest_schema_version 2.0 起不再写入与嵌套重复的扁平键）。

    **Schema 4.5**：在 step5 命令下增加 ``backbones`` / ``training_runtime`` / ``peft`` 可审计块；
    运行环境（OMP/MKL/TOKENIZERS/CUDA 等）**仅**出现在顶层 ``runtime_env``；
    ``hyperparameters`` 不含线程或 CUDA 镜像字段。
    """
    if not (cli_invocation or "").strip():
        cli_invocation = (os.environ.get("ODCR_MANIFEST_CLI_INVOCATION") or "").strip() or None

    train_csv_res = _resolved_train_csv(cfg)
    model_res = _resolved_model_weights(cfg)

    _train_fp = getattr(cfg, "training_semantic_fingerprint", "") or ""
    _gen_fp = getattr(cfg, "generation_semantic_fingerprint", "") or ""
    _rd_fp = getattr(cfg, "runtime_diagnostics_fingerprint", "") or ""
    _src_json = getattr(cfg, "config_field_sources_json", "") or "{}"
    try:
        _src_obj = json.loads(_src_json) if _src_json.strip() else {}
    except json.JSONDecodeError:
        _src_obj = {}
    _cp = getattr(cfg, "consumed_presets_json", "") or "{}"
    try:
        _consumed = json.loads(_cp) if _cp.strip() else {}
    except json.JSONDecodeError:
        _consumed = {}
    _bcb = getattr(cfg, "config_before_cli_json", "") or "{}"
    try:
        _before_cli = json.loads(_bcb) if _bcb.strip() else {}
    except json.JSONDecodeError:
        _before_cli = {}
    _treq = getattr(cfg, "thread_env_requested_json", "") or "{}"
    _tee = getattr(cfg, "thread_env_effective_json", "") or "{}"
    _lreq = getattr(cfg, "launcher_env_requested_json", "") or "{}"
    _lee = getattr(cfg, "launcher_env_effective_json", "") or "{}"
    try:
        _treq_o = json.loads(_treq) if _treq.strip() else {}
    except json.JSONDecodeError:
        _treq_o = {}
    try:
        _tee_o = json.loads(_tee) if _tee.strip() else {}
    except json.JSONDecodeError:
        _tee_o = {}
    try:
        _lreq_o = json.loads(_lreq) if _lreq.strip() else {}
    except json.JSONDecodeError:
        _lreq_o = {}
    try:
        _lee_o = json.loads(_lee) if _lee.strip() else {}
    except json.JSONDecodeError:
        _lee_o = {}
    m: dict[str, Any] = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "repo_root": str(cfg.repo_root.resolve()),
        "mainline_command": cfg.command,
        "stage": _stage_label(cfg.command),
        "task_id": cfg.task_id,
        "invoked_command": getattr(cfg, "invoked_command", None) or cfg.command,
        "resolved_command_kind": getattr(cfg, "resolved_command_kind", None) or cfg.command,
        "cell_command": getattr(cfg, "cell_command", None),
        "matrix_session_id": getattr(cfg, "matrix_session_id", None),
        "matrix_cell_id": getattr(cfg, "matrix_cell_id", None),
        "training_semantic_fingerprint": _train_fp or None,
        "generation_semantic_fingerprint": _gen_fp or None,
        "runtime_diagnostics_fingerprint": _rd_fp or None,
        "config_field_sources": _src_obj,
        "consumed_presets": _consumed,
        "config_before_cli": _before_cli,
        "runtime_env": {
            "thread_env_requested": _treq_o,
            "thread_env_effective": _tee_o,
            "launcher_env_requested": _lreq_o,
            "launcher_env_effective": _lee_o,
            "note": (
                "runtime_env 为唯一运行环境记录区（OMP/MKL/TOKENIZERS/CUDA_VISIBLE_DEVICES 等）；"
                "不计入 training_semantic_fingerprint / generation_semantic_fingerprint"
            ),
        },
        "training_preset": cfg.preset_name,
        "hardware_preset": cfg.hardware_preset_id,
        "decode_preset": cfg.decode_preset_id or None,
        "eval_profile": getattr(cfg, "eval_profile_id", "") or None,
        "generation_semantic_resolved": (
            {
                "decode_preset": cfg.decode_preset_id,
                "decode_strategy": cfg.decode_strategy,
                "decode_seed": cfg.decode_seed,
                "max_explanation_length": cfg.max_explanation_length,
                "decode_max_explanation_length": cfg.max_explanation_length,
                "label_smoothing": cfg.label_smoothing,
                "repetition_penalty": cfg.repetition_penalty,
                "generate_temperature": cfg.generate_temperature,
                "generate_top_p": cfg.generate_top_p,
                "no_repeat_ngram_size": cfg.no_repeat_ngram_size,
                "min_len": cfg.min_len,
                "domain_fusion_mode": getattr(cfg, "domain_fusion_mode", "gate_cross_attn"),
                "decode_profile_sha1": hashlib.sha1(
                    (cfg.decode_profile_json or "").encode("utf-8")
                ).hexdigest()[:16],
                "rerank_profile_sha1": hashlib.sha1(
                    (cfg.rerank_profile_json or "").encode("utf-8")
                ).hexdigest()[:16],
                "generation_semantic_family_tag": compute_generation_semantic_family_tag(
                    {
                        "strategy": cfg.decode_strategy,
                        "temperature": cfg.generate_temperature,
                        "top_p": cfg.generate_top_p,
                        "repetition_penalty": cfg.repetition_penalty,
                        "max_explanation_length": cfg.max_explanation_length,
                        "no_repeat_ngram_size": cfg.no_repeat_ngram_size,
                        "min_len": cfg.min_len,
            "domain_fusion_mode": getattr(cfg, "domain_fusion_mode", "gate_cross_attn"),
                    }
                ),
            }
            if (cfg.decode_preset_id or "").strip()
            else None
        ),
        "training_label": {
            "train_label_max_length": getattr(cfg, "train_label_max_length", None),
            "train_dynamic_padding": getattr(cfg, "train_dynamic_padding", None),
            "train_padding_strategy": getattr(cfg, "train_padding_strategy", None),
            "decode_max_explanation_length": cfg.max_explanation_length,
        },
        "domain_auxiliary": cfg.auxiliary,
        "domain_target": cfg.target,
        "run_lineage": _run_lineage(cfg),
        "checkpoint_resolution": (
            {
                "default_checkpoint_policy": "best",
                "best_checkpoint_path": str(
                    path_layout.best_model_path(Path(cfg.checkpoint_dir))
                ),
                "best_event_path": str(path_layout.state_dir(Path(cfg.checkpoint_dir)) / "best_event.json"),
                "checkpoint_selection_metric": "valid_loss",
                "canonical_weight_file": "model/best.pth",
            }
            if cfg.command in ("step3", "step5", "eval", "eval-rerank")
            else None
        ),
        "paths": {
            "stage_run_dir": cfg.checkpoint_dir,
            "log_dir": cfg.log_dir,
            "iteration_root_dir": cfg.iteration_root_dir,
            "manifest_dir": cfg.manifest_dir,
            "eval_run_dir": cfg.eval_run_dir,
            **(
                {
                    "step3_checkpoint_dir": cfg.step3_checkpoint_dir,
                    "step4_run": cfg.step4_run,
                }
                if cfg.command == "step4" and cfg.step3_checkpoint_dir
                else (
                    {"step4_run": cfg.step4_run}
                    if cfg.step4_run
                    else {}
                )
            ),
        },
        "hyperparameters": {
            "learning_rate": cfg.learning_rate,
            **({} if cfg.command == "step3" else {"coef": cfg.coef}),
            **(
                {"explainer_loss_weight": cfg.explainer_loss_weight}
                if cfg.command == "step5"
                else {}
            ),
            **(
                {
                    "optimizer": json.loads(str(getattr(cfg, "optimizer_config_json", "{}") or "{}")),
                    "precision": json.loads(str(getattr(cfg, "precision_config_json", "{}") or "{}")),
                    "tokenizer": json.loads(str(getattr(cfg, "tokenizer_config_json", "{}") or "{}")),
                    "evidence": json.loads(str(getattr(cfg, "evidence_config_json", "{}") or "{}")),
                    "scheduler": json.loads(str(getattr(cfg, "scheduler_config_json", "{}") or "{}")),
                    "valid_batch": json.loads(str(getattr(cfg, "valid_batch_config_json", "{}") or "{}")),
                    "scenario_profile": json.loads(str(getattr(cfg, "scenario_profile_json", "{}") or "{}")),
                    "max_grad_norm": getattr(cfg, "max_grad_norm", None),
                    "validate_every_epochs": getattr(cfg, "validate_every_epochs", None),
                }
                if cfg.command == "step3"
                else {}
            ),
            **(_training_row_slice_for_manifest(cfg)),
            "train_global_batch_size": cfg.train_batch_size,
            "train_per_device_batch_size": cfg.per_device_train_batch_size,
            **(
                {"step3_batch_semantics": "odcr_no_accum/1"}
                if cfg.command == "step3"
                else {}
            ),
            "batch_semantics_version": getattr(cfg, "batch_semantics_version", "odcr_no_accum/1"),
            "grad_accum_removed": bool(getattr(cfg, "grad_accum_removed", True)),
            "global_batch_size": getattr(cfg, "global_batch_size", cfg.train_batch_size),
            "per_gpu_batch_size": getattr(cfg, "per_gpu_batch_size", cfg.per_device_train_batch_size),
            "effective_global_batch_size": cfg.effective_global_batch_size,
            "epochs": cfg.epochs,
            "num_proc": cfg.num_proc,
            "ddp_world_size": cfg.ddp_world_size,
            "seed": cfg.seed,
            "label_smoothing": cfg.label_smoothing,
            "train_label_max_length": getattr(cfg, "train_label_max_length", None),
            "train_dynamic_padding": getattr(cfg, "train_dynamic_padding", None),
            "train_padding_strategy": getattr(cfg, "train_padding_strategy", None),
            "loss_weight_repeat_ul": getattr(cfg, "loss_weight_repeat_ul", None),
            "loss_weight_terminal_clean": getattr(cfg, "loss_weight_terminal_clean", None),
            "repetition_penalty": cfg.repetition_penalty,
            "generate_temperature": cfg.generate_temperature,
            "generate_top_p": cfg.generate_top_p,
            "training_preset_train_batch_size": cfg.training_preset_train_batch_size,
            "ema_enabled": getattr(cfg, "ema_enabled", None),
            "ema_decay": getattr(cfg, "ema_decay", None),
            "generate_during_train": getattr(cfg, "generate_during_train", None),
            "decode_backend": getattr(cfg, "decode_backend", None),
            "decode_backend_fallback_policy": getattr(cfg, "decode_backend_fallback_policy", None),
            "train_time_eval_decode_backend": getattr(cfg, "train_time_eval_decode_backend", None),
            **(
                {
                    "global_eval_batch_size": cfg.global_eval_batch_size,
                    "eval_per_gpu_batch_size": cfg.eval_per_gpu_batch_size,
                }
                if cfg.global_eval_batch_size is not None
                else {}
            ),
            **(
                {"full_bleu_eval_resolved": dict(cfg.full_bleu_eval_resolved)}
                if getattr(cfg, "full_bleu_eval_resolved", None)
                else {}
            ),
            "full_bleu_decode_strategy": getattr(cfg, "full_bleu_decode_strategy", "inherit"),
        },
        "step_modes": {
            "step3_mode": cfg.step3_mode,
            "step5_train_only": cfg.step5_train_only,
        },
        "training_diagnostics": training_diagnostics_snapshot(
            diagnostics_scope="parent",
            effective_training_payload_json=str(
                getattr(cfg, "effective_training_payload_json", "") or ""
            ),
        ),
        "governance_layer": {
            "purpose": "repro_orchestration_audit",
            "note": "manifest/fingerprint/matrix/analysis_pack 属工程治理层，不属于核心建模增强。",
        },
    }
    m["effective_config"] = {
        "hyperparameters": m["hyperparameters"],
        "hardware_preset": cfg.hardware_preset_id,
        "training_preset": cfg.preset_name,
        "decode_preset": cfg.decode_preset_id or None,
        "eval_profile_orchestrator": getattr(cfg, "eval_profile_id", "") or None,
        "rerank_preset": (cfg.rerank_preset_id or None) if cfg.command == "eval-rerank" else None,
        "training_semantic_fingerprint": _train_fp or None,
        "generation_semantic_fingerprint": _gen_fp or None,
    }
    if cfg.command in ("eval", "eval-rerank", "eval-matrix", "eval-rerank-matrix", "step4") and getattr(
        cfg, "eval_profile_id", ""
    ):
        _ej = getattr(cfg, "eval_profile_resolution_json", "") or "{}"
        try:
            _eor = json.loads(_ej) if _ej.strip() else {}
        except json.JSONDecodeError:
            _eor = {}
        m["eval_profile_detail"] = {
            "eval_profile": cfg.eval_profile_id,
            "resolved_hardware_preset": cfg.hardware_preset_id,
            "resolved_decode_preset": cfg.decode_preset_id or None,
            "resolved_rerank_preset": (cfg.rerank_preset_id or None)
            if cfg.command in ("eval-rerank", "eval-rerank-matrix")
            else None,
            "global_eval_batch_size": cfg.global_eval_batch_size,
            "eval_per_gpu_batch_size": cfg.eval_per_gpu_batch_size,
            "ddp_world_size": cfg.ddp_world_size,
            "orchestrator_yaml": _eor if isinstance(_eor, dict) else {},
        }
    if cli_invocation:
        m["invoked_command_line"] = cli_invocation

    ids: dict[str, Any] = {}
    if cfg.run_name is not None:
        ids["run_name"] = cfg.run_name
    if cfg.from_run is not None:
        ids["from_run"] = cfg.from_run
    if cfg.step5_run is not None:
        ids["step5_run"] = cfg.step5_run
    if ids:
        m["run_identifiers"] = ids

    ri: dict[str, Any] = {}
    if cfg.train_csv:
        ri["train_csv_cli"] = cfg.train_csv
    if train_csv_res:
        ri["train_csv_resolved"] = train_csv_res
        ri["index_contract_resolved"] = str(Path(train_csv_res).resolve().parent / INDEX_CONTRACT_FILENAME)
    if model_res:
        ri["model_weights_resolved"] = model_res
    if ri:
        m["resolved_inputs"] = ri

    if cfg.command == "eval-rerank":
        m["rerank"] = {
            "rerank_preset": cfg.rerank_preset_id,
            "num_return_sequences": cfg.num_return_sequences,
            "rerank_method": cfg.rerank_method,
            "rerank_top_k": cfg.rerank_top_k,
            "rerank_weight_logprob": cfg.rerank_weight_logprob,
            "rerank_weight_length": cfg.rerank_weight_length,
            "rerank_weight_repeat": cfg.rerank_weight_repeat,
            "rerank_weight_dirty": cfg.rerank_weight_dirty,
            "rerank_target_len_ratio": cfg.rerank_target_len_ratio,
            "export_examples_mode": cfg.export_examples_mode,
            "export_full_rerank_examples": cfg.export_full_rerank_examples,
            "rerank_malformed_tail_penalty": cfg.rerank_malformed_tail_penalty,
            "rerank_malformed_token_penalty": cfg.rerank_malformed_token_penalty,
        }

    if cfg.command == "step5":
        m["backbones"] = _manifest_backbones_block(cfg)
        m["training_runtime"] = _manifest_training_runtime_block(cfg)
        m["peft"] = _manifest_peft_block(cfg)

    return m


def manifest_json_path(cfg: ResolvedConfig) -> Path:
    """与当次 run 产物同目录的 ``manifest.json``。"""
    return Path(cfg.manifest_dir) / MANIFEST_FILENAME


def write_run_manifest_json(cfg: ResolvedConfig, manifest: Mapping[str, Any] | None = None) -> Path:
    data = dict(manifest) if manifest is not None else build_run_manifest(cfg)
    out = manifest_json_path(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return out


def should_write_manifest_json() -> bool:
    """Run manifests are mandatory One-Control handoff artifacts."""
    return True
