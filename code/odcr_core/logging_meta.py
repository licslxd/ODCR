"""MAINLINE 统一日志块：控制台摘要与 run-meta 文件日志分层。"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from odcr_core import path_layout
from odcr_core.dispatch import print_dispatch_routing, print_dispatch_script_detail
from odcr_core.manifests import (
    MANIFEST_FILENAME,
    build_run_manifest,
    resolved_config_path,
    run_summary_path,
    source_table_path,
    manifest_json_path,
)

ResolvedConfig = Any

CONSOLE_LEVEL_SUMMARY = "summary"
CONSOLE_LEVEL_VERBOSE = "verbose"
CONSOLE_LEVEL_DEBUG = "debug"
CONSOLE_LEVELS = frozenset({CONSOLE_LEVEL_SUMMARY, CONSOLE_LEVEL_VERBOSE, CONSOLE_LEVEL_DEBUG})

CONSOLE_LOG_FILENAME = "console.log"
FULL_LOG_FILENAME = "full.log"
ERRORS_LOG_FILENAME = "errors.log"
DEBUG_LOG_FILENAME = "debug.log"
SAMPLES_LOG_FILENAME = "samples.jsonl"

CONSOLE_POLICY_SUMMARY = (
    "Default console shows stage/task/domains, run_id, status, timing, key train config, "
    "device/speed summaries when emitted, epoch summaries, final metrics, errors, and run_summary path. "
    "Full config, source table, guardrail detail, per-batch/per-rank output, and sample text stay in run-meta files "
    "or require verbose/debug display."
)


def normalize_console_level(value: str | None) -> str:
    raw = (value or CONSOLE_LEVEL_SUMMARY).strip().lower()
    if raw not in CONSOLE_LEVELS:
        return CONSOLE_LEVEL_SUMMARY
    return raw


def console_level_from_flags(*, verbose: bool = False, debug: bool = False) -> str:
    if debug:
        return CONSOLE_LEVEL_DEBUG
    if verbose:
        return CONSOLE_LEVEL_VERBOSE
    return CONSOLE_LEVEL_SUMMARY


def run_log_paths(cfg: ResolvedConfig) -> dict[str, Path]:
    meta = Path(cfg.manifest_dir).expanduser().resolve()
    return {
        "console": meta / CONSOLE_LOG_FILENAME,
        "full": meta / FULL_LOG_FILENAME,
        "errors": meta / ERRORS_LOG_FILENAME,
        "debug": meta / DEBUG_LOG_FILENAME,
        "samples": meta / SAMPLES_LOG_FILENAME,
    }


def _repo_relative(cfg: ResolvedConfig, path: str | Path | None) -> str:
    if path is None:
        return ""
    root = Path(cfg.repo_root).expanduser().resolve()
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = (root / p).resolve()
    else:
        p = p.resolve()
    try:
        return p.relative_to(root).as_posix()
    except ValueError:
        return p.as_posix()


def _append_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for line in lines:
            fh.write(line.rstrip("\n") + "\n")


def emit_console_lines(cfg: ResolvedConfig, lines: list[str]) -> None:
    paths = run_log_paths(cfg)
    _append_lines(paths["console"], lines)
    for line in lines:
        print(line, flush=True)


def append_error_log(cfg: ResolvedConfig, lines: list[str]) -> None:
    paths = run_log_paths(cfg)
    _append_lines(paths["errors"], lines)


def append_debug_log(cfg: ResolvedConfig, lines: list[str]) -> None:
    paths = run_log_paths(cfg)
    _append_lines(paths["debug"], lines)


def append_full_log(cfg: ResolvedConfig, lines: list[str]) -> None:
    paths = run_log_paths(cfg)
    _append_lines(paths["full"], lines)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _fmt_elapsed(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    seconds = max(0.0, float(seconds))
    if seconds >= 3600:
        return f"{seconds / 3600:.2f}h"
    if seconds >= 60:
        return f"{seconds / 60:.1f}m"
    return f"{seconds:.1f}s"


def _run_id(cfg: ResolvedConfig) -> str:
    if cfg.command == "step3" and cfg.run_name:
        return str(cfg.run_name)
    if cfg.command == "step4" and cfg.step4_run:
        return str(cfg.step4_run)
    if cfg.command in ("eval", "eval-rerank") and cfg.eval_run_dir:
        return Path(cfg.eval_run_dir).name
    if cfg.command == "step5" and cfg.step5_run:
        return str(cfg.step5_run)
    return Path(cfg.checkpoint_dir).name


def _device_summary(cfg: ResolvedConfig) -> str:
    cuda_visible = ""
    try:
        launcher = json.loads(getattr(cfg, "launcher_env_effective_json", "") or "{}")
    except json.JSONDecodeError:
        launcher = {}
    if isinstance(launcher, dict):
        cuda_visible = str(launcher.get("CUDA_VISIBLE_DEVICES") or "").strip()
    pieces = [
        f"hardware={getattr(cfg, 'hardware_preset_id', '')}",
        f"ddp_world_size={getattr(cfg, 'ddp_world_size', '')}",
        f"precision={getattr(cfg, 'train_precision', '')}",
    ]
    if cuda_visible:
        pieces.append(f"CUDA_VISIBLE_DEVICES={cuda_visible}")
    return " ".join(piece for piece in pieces if piece and not piece.endswith("="))


def _key_config_summary(cfg: ResolvedConfig) -> str:
    return (
        f"batch_size={cfg.train_batch_size} micro_batch_size={cfg.per_device_train_batch_size} "
        f"grad_accum={cfg.gradient_accumulation_steps} ddp_world_size={cfg.ddp_world_size} "
        f"epochs={cfg.epochs} lr={cfg.learning_rate}"
    )


def console_summary_lines(
    cfg: ResolvedConfig,
    *,
    status: str,
    started_at: str | None = None,
    elapsed_sec: float | None = None,
    finished_at: str | None = None,
    error: str | None = None,
) -> list[str]:
    paths = run_log_paths(cfg)
    started = started_at or _utc_now()
    lines = [
        f"[ODCR] stage={cfg.command} task={cfg.task_id} source={cfg.auxiliary} target={cfg.target}",
        f"[ODCR] run_id={_run_id(cfg)} status={status}",
        f"[ODCR] started_at={started} elapsed={_fmt_elapsed(elapsed_sec)}",
        f"[ODCR] key_config {_key_config_summary(cfg)}",
        f"[ODCR] device {_device_summary(cfg)}",
        (
            "[ODCR] logs "
            f"console={_repo_relative(cfg, paths['console'])} "
            f"full={_repo_relative(cfg, paths['full'])} "
            f"errors={_repo_relative(cfg, paths['errors'])}"
        ),
        f"[ODCR] run_summary={_repo_relative(cfg, run_summary_path(cfg.manifest_dir))}",
    ]
    if finished_at:
        lines.insert(3, f"[ODCR] finished_at={finished_at} total_duration={_fmt_elapsed(elapsed_sec)}")
    if error:
        lines.append(f"[ODCR] error={error}")
    return lines


def initialize_run_log_files(
    cfg: ResolvedConfig,
    snapshot: dict[str, Any],
    *,
    command_line: str,
    started_at: str,
    console_level: str,
) -> dict[str, Path]:
    """Prepare run-meta log files and write detailed handoff context to full.log."""
    paths = run_log_paths(cfg)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)

    manifest = build_run_manifest(cfg, cli_invocation=command_line)
    details = [
        "========== ODCR RUN LOGGING PHASE 2 ==========",
        f"generated_at_utc={_utc_now()}",
        f"command={command_line}",
        f"console_level={normalize_console_level(console_level)}",
        f"console_policy={CONSOLE_POLICY_SUMMARY}",
        f"resolved_config_path={_repo_relative(cfg, resolved_config_path(cfg.manifest_dir))}",
        f"source_table_path={_repo_relative(cfg, source_table_path(cfg.manifest_dir))}",
        f"run_summary_path={_repo_relative(cfg, run_summary_path(cfg.manifest_dir))}",
        f"samples_log_path={_repo_relative(cfg, paths['samples'])}",
        "resolved_snapshot=" + json.dumps(snapshot, ensure_ascii=False, sort_keys=True, default=str),
        "run_manifest_preview=" + json.dumps(manifest, ensure_ascii=False, sort_keys=True, default=str),
        "========== END PRE-RUN CONTEXT ==========",
    ]
    append_full_log(cfg, details)
    append_debug_log(
        cfg,
        [
            "========== ODCR DEBUG LOG ==========",
            "Raw child stdout/stderr captured by the parent launcher is appended here.",
        ],
    )
    append_error_log(
        cfg,
        [
            "========== ODCR ERRORS/WARNINGS LOG ==========",
            "Warnings, errors, and traceback snippets captured by the parent launcher are appended here.",
        ],
    )
    return paths


def _stage_label(command: str) -> str:
    return {
        "step3": "step3（结构化 shared/specific 解耦）",
        "step4": "step4（反事实推理，eval 语义 / eval_profile）",
        "step5": "step5（主模型训练）",
        "eval": "eval（Step5 评测）",
        "eval-rerank": "eval-rerank（Step5 多候选 rerank 评测）",
        "eval-rerank-matrix": "eval-rerank-matrix（多 decode preset × rerank）",
        "rerank-summary": "rerank-summary（Phase2 汇总表）",
        "pipeline": "pipeline（step3→step4→step5）",
    }.get(command, command)


def print_pre_run_banner(
    command: str,
    cfg: ResolvedConfig,
    *,
    cli_invocation: str | None = None,
    console_level: str = CONSOLE_LEVEL_SUMMARY,
    started_at: str | None = None,
) -> None:
    """运行子进程前的主线摘要（JSON 在 ``runners`` torchrun 前落盘，见 ``[Manifest] wrote``）。"""
    level = normalize_console_level(console_level)
    if level == CONSOLE_LEVEL_SUMMARY:
        _ = cli_invocation
        emit_console_lines(cfg, console_summary_lines(cfg, status="starting", started_at=started_at))
        return

    emit_console_lines(cfg, console_summary_lines(cfg, status="starting", started_at=started_at))
    man = build_run_manifest(cfg, cli_invocation=cli_invocation)
    print(f"[ODCR Mainline] command={command}", flush=True)
    _tfp = getattr(cfg, "training_semantic_fingerprint", "") or ""
    _gfp = getattr(cfg, "generation_semantic_fingerprint", "") or ""
    _rd = getattr(cfg, "runtime_diagnostics_fingerprint", "") or ""
    if _tfp.strip():
        print(f"[Semantic] training_semantic_fingerprint={_tfp}", flush=True)
    if _gfp.strip():
        print(f"[Semantic] generation_semantic_fingerprint={_gfp}", flush=True)
    if _rd.strip():
        print(f"[Diagnostics] runtime_diagnostics_fingerprint={_rd}", flush=True)
    print(f"[Stage] {_stage_label(command)}", flush=True)
    print(
        f"[Preset] training={cfg.preset_name!r} hardware={cfg.hardware_preset_id!r} "
        f"decode_preset={cfg.decode_preset_id!r}",
        flush=True,
    )
    if getattr(cfg, "eval_profile_id", "") and cfg.command in ("eval", "eval-rerank", "step4"):
        _rp = cfg.rerank_preset_id if cfg.command == "eval-rerank" else ""
        print(
            f"[Eval profile orchestrator] name={cfg.eval_profile_id!r} hardware={cfg.hardware_preset_id!r} "
            f"decode_preset={cfg.decode_preset_id!r} rerank_preset={_rp!r} "
            f"global_eval_batch_size={cfg.global_eval_batch_size} eval_per_gpu_batch_size={cfg.eval_per_gpu_batch_size} "
            f"ddp_world_size={cfg.ddp_world_size}",
            flush=True,
        )
    print("[Resolved Inputs]", flush=True)
    print(f"  task={cfg.task_id} auxiliary={cfg.auxiliary!r} target={cfg.target!r}", flush=True)
    if cfg.train_csv:
        print(f"  train_csv (CLI)={cfg.train_csv}", flush=True)
    ri = man.get("resolved_inputs") or {}
    if ri.get("train_csv_resolved"):
        print(f"  train_csv (resolved)={ri['train_csv_resolved']}", flush=True)
    if cfg.run_name:
        print(f"  run_name={cfg.run_name!r}", flush=True)
    if cfg.from_run:
        print(f"  from_run={cfg.from_run!r}", flush=True)
    if cfg.step5_run:
        print(f"  step5_run={cfg.step5_run!r}", flush=True)
    if cfg.step4_run:
        print(f"  step4_run={cfg.step4_run!r}", flush=True)
    if cfg.model_path:
        print(f"  model_path={cfg.model_path}", flush=True)
    elif ri.get("model_weights_resolved"):
        print(f"  model_weights (resolved)={ri['model_weights_resolved']}", flush=True)
    print("[Resolved Outputs]", flush=True)
    print(f"  stage_run_dir={cfg.checkpoint_dir}", flush=True)
    print(f"  log_dir={cfg.log_dir}", flush=True)
    print(
        f"  iteration_root_dir={cfg.iteration_root_dir}  # vN root, not a metric-file directory",
        flush=True,
    )
    print(f"  iteration_id={cfg.iteration_id}", flush=True)
    print(f"  manifest_dir={cfg.manifest_dir}", flush=True)
    if cfg.eval_run_dir:
        _er = Path(cfg.eval_run_dir)
        print(f"  eval_run_dir={cfg.eval_run_dir}", flush=True)
        if cfg.command == "eval-rerank":
            print(f"  rerank_run_dir={cfg.eval_run_dir}", flush=True)
        print(
            f"  metrics_path={path_layout.eval_metrics_path(_er, rerank=(cfg.command == 'eval-rerank'))}",
            flush=True,
        )
    if command == "step3":
        print(f"  step3_mode={cfg.step3_mode}", flush=True)
    if command == "step5":
        print(f"  step5_train_only={cfg.step5_train_only}", flush=True)
    hp = man.get("hyperparameters") or {}
    _re = man.get("runtime_env") or {}
    _te = _re.get("thread_env_effective") if isinstance(_re, dict) else {}
    _le = _re.get("launcher_env_effective") if isinstance(_re, dict) else {}
    if not isinstance(_te, dict):
        _te = {}
    if not isinstance(_le, dict):
        _le = {}
    print(
        "  runtime_env: "
        f"thread_env_effective={json.dumps(_te, ensure_ascii=False)} "
        f"launcher_env_effective={json.dumps(_le, ensure_ascii=False)}",
        flush=True,
    )
    if cfg.command == "step4" and cfg.global_eval_batch_size is not None:
        _epg = cfg.eval_per_gpu_batch_size
        _epid = (getattr(cfg, "eval_profile_id", "") or "").strip()
        print(
            f"  step4_eval_inference: eval_profile_name={_epid!r} "
            f"global_eval_batch_size={cfg.global_eval_batch_size} "
            f"eval_per_gpu_batch_size={_epg} num_proc={cfg.num_proc} "
            f"ddp_world_size={cfg.ddp_world_size} seed={cfg.seed}",
            flush=True,
        )
    elif cfg.global_eval_batch_size is not None and cfg.command in ("eval", "eval-rerank", "step5"):
        _epg = cfg.eval_per_gpu_batch_size
        print(
            f"  eval_parallelism: global_eval_batch_size={cfg.global_eval_batch_size} "
            f"eval_per_gpu_batch_size={_epg} train_global_batch_size={cfg.train_batch_size} "
            f"train_per_device_batch_size={cfg.per_device_train_batch_size} gradient_accumulation_steps={cfg.gradient_accumulation_steps} "
            f"effective_global_batch_size={cfg.effective_global_batch_size} num_proc={cfg.num_proc} "
            f"ddp_world_size={cfg.ddp_world_size} seed={cfg.seed}",
            flush=True,
        )
    else:
        print(
            f"  train_parallelism: train_global_batch_size={cfg.train_batch_size} "
            f"train_per_device_batch_size={cfg.per_device_train_batch_size} gradient_accumulation_steps={cfg.gradient_accumulation_steps} "
            f"effective_global_batch_size={cfg.effective_global_batch_size} epochs={cfg.epochs} "
            f"num_proc={cfg.num_proc} ddp_world_size={cfg.ddp_world_size} seed={cfg.seed}",
            flush=True,
        )
    if cfg.command == "step3":
        print(
            f"  train_objective: lr={hp.get('learning_rate')} coef={hp.get('coef')} "
            "objective=structured_disentanglement",
            flush=True,
        )
    else:
        print(
            f"  train_objective: lr={hp.get('learning_rate')} coef={hp.get('coef')} "
            f"explainer_loss_weight={hp.get('explainer_loss_weight')}",
            flush=True,
        )
    dr = man.get("generation_semantic_resolved") or {}
    print(
        f"  decode_preset={dr.get('decode_preset')!r} decode_strategy={dr.get('decode_strategy')!r} "
        f"decode_seed={dr.get('decode_seed')!r} max_explanation_length={dr.get('max_explanation_length')}",
        flush=True,
    )
    print(
        f"  decode (generation): label_smoothing={dr.get('label_smoothing')} "
        f"repetition_penalty={dr.get('repetition_penalty')} "
        f"temperature={dr.get('generate_temperature')} top_p={dr.get('generate_top_p')}",
        flush=True,
    )
    print("[Dispatch Summary]", flush=True)
    print_dispatch_routing(command)
    print_dispatch_script_detail(command)
    mp = manifest_json_path(cfg)
    print(
        f"[Manifest] torchrun 前将写入 {mp}（文件名固定为 {MANIFEST_FILENAME}；"
        "run handoff metadata is mandatory）。复现字段说明见 README。",
        flush=True,
    )


def print_pipeline_opening(*, step3_preset: str) -> None:
    print("[ODCR Mainline] command=pipeline", flush=True)
    print("[Stage] pipeline（step3→step4→step5）", flush=True)
    print(
        f"[Preset] Step3/Step4 使用 CLI --preset={step3_preset!r}；Step5 将强制 preset='step5'；"
        "Step4 须本命令的 --eval-profile（推理 batch 仅来自该 profile 的 eval_batch_size）。",
        flush=True,
    )
    print_dispatch_routing("pipeline")
    print_dispatch_script_detail("pipeline")
    print(
        f"[Manifest] 各段 torchrun 前在各自 manifest_dir 写入 {MANIFEST_FILENAME} "
        "（mandatory run handoff metadata）。",
        flush=True,
    )
