"""单次运行的复现/排障清单：结构化字段 + 稳定 JSON 路径。"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from odcr_core.artifacts import train_csv_path
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
RUN_SUMMARY_FILENAME = "run_summary.json"
CONSOLE_LOG_FILENAME = "console.log"
FULL_LOG_FILENAME = "full.log"
DEBUG_LOG_FILENAME = "debug.log"
SAMPLES_LOG_FILENAME = "samples.jsonl"
LATEST_FILENAME = "latest.json"
RUN_SUMMARY_SCHEMA_VERSION = "1.0"
SOURCE_TABLE_SCHEMA_VERSION = "1.0"
OPTIONAL_ARTIFACT_REASONS = {
    "errors_log": "no_error",
    "debug_log": "debug_disabled",
    "samples_log": "samples_not_requested",
}


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


def run_summary_path(meta_dir: str | Path) -> Path:
    return Path(meta_dir).expanduser().resolve() / RUN_SUMMARY_FILENAME


def latest_pointer_path(stage_unit_dir: str | Path) -> Path:
    return Path(stage_unit_dir).expanduser().resolve() / LATEST_FILENAME


def build_source_table_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    raw = snapshot.get("field_sources")
    field_sources = dict(raw) if isinstance(raw, Mapping) else {}
    return {
        "source_table_schema_version": SOURCE_TABLE_SCHEMA_VERSION,
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
) -> tuple[Path, Path]:
    meta = Path(meta_dir).expanduser().resolve()
    config_path = resolved_config_path(meta)
    source_path = source_table_path(meta)
    source_payload = dict(source_table) if source_table is not None else build_source_table_snapshot(snapshot)
    atomic_write_json(config_path, dict(snapshot))
    atomic_write_json(source_path, source_payload)
    return config_path, source_path


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
        "source_table_path": _repo_relative(root, source_table_path(meta)),
        "console_log_path": _repo_relative(root, console_log_path),
        "full_log_path": _repo_relative(root, full_log_path),
        "errors_log_path": _repo_relative(root, errors_log_path),
        "metrics_path": _repo_relative(root, metrics_path),
        "lineage_path": _repo_relative(root, lineage_path),
        "manifest_path": _repo_relative(root, manifest_path),
        "key_artifacts": key_artifact_payload,
        "optional_artifacts": optional_artifact_payload,
        "latest_error": latest_error,
        "validation_status": validation_status,
        "post_edit_scope": post_edit_scope,
    }
    return payload


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
        "latest_run_id": str(run_id),
        "latest_run_dir": _repo_relative(root, run_dir),
        "latest_summary_path": _repo_relative(root, summary_path),
        "latest_status": status,
        "updated_at": updated_at or _utc_now(),
    }
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
    out = run_summary_path(meta)
    atomic_write_json(out, dict(summary))
    if update_latest:
        run_dir_value = summary.get("run_dir")
        if not run_dir_value:
            raise ValueError("run_summary requires run_dir to update latest.json")
        run_dir = Path(str(run_dir_value))
        if not run_dir.is_absolute():
            run_dir = (root / run_dir).resolve()
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
        "source_table": source_table_path(meta),
        "console_log": meta / CONSOLE_LOG_FILENAME,
        "full_log": meta / FULL_LOG_FILENAME,
        "debug_log": meta / DEBUG_LOG_FILENAME,
        "samples_log": meta / SAMPLES_LOG_FILENAME,
    }
    if cfg.command in ("step3", "step5", "eval", "eval-rerank"):
        key_artifacts["model"] = path_layout.best_model_path(Path(cfg.checkpoint_dir))
    if cfg.command in ("step4", "step5"):
        try:
            key_artifacts["training_csv"] = train_csv_path(cfg)
        except Exception:
            pass
    if stage in ("eval", "rerank"):
        key_artifacts["metrics"] = metrics_path
    return build_run_summary(
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
    return {
        "precision": str(getattr(cfg, "train_precision", "bf16")),
        "per_device_train_batch_size": int(cfg.per_device_train_batch_size),
        "per_device_eval_batch_size": int(getattr(cfg, "per_device_eval_batch_size", 2)),
        "gradient_accumulation_steps": int(cfg.gradient_accumulation_steps),
        "effective_batch_size": int(cfg.effective_global_batch_size),
        "ddp_world_size": int(cfg.ddp_world_size),
    }


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
            "coef": cfg.coef,
            **(
                {"explainer_loss_weight": cfg.explainer_loss_weight}
                if cfg.command == "step5"
                else {}
            ),
            **(_training_row_slice_for_manifest(cfg)),
            "train_global_batch_size": cfg.train_batch_size,
            "train_per_device_batch_size": cfg.per_device_train_batch_size,
            "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
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
