"""torchrun 子进程编排：供 ``odcr.py`` 调用；INTERNAL EXECUTOR 仅出现在本子模块。"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from odcr_core.artifacts import ensure_step5_csv_symlink
from odcr_core.dispatch import TORCHRUN_STEP3_SCRIPT, TORCHRUN_STEP4_SCRIPT, TORCHRUN_STEP5_SCRIPT
from odcr_core.logging_meta import (
    append_debug_log,
    append_error_log,
    append_full_log,
    emit_console_lines,
    normalize_console_level,
    run_log_paths,
)
from odcr_core.manifests import build_run_manifest, write_run_manifest_json, write_training_runtime_config_artifact
from odcr_core import path_layout

ResolvedConfig = Any

_ERROR_LINE_TOKENS = (
    "error",
    "exception",
    "traceback",
    "runtimeerror",
    "valueerror",
    "filenotfounderror",
    "cuda out of memory",
    "warning",
    "warn",
)

_SUMMARY_LINE_TOKENS = (
    "[Epoch Summary]",
    "[Eval Summary]",
    "FINAL RESULTS",
    "[Recommendation]",
    "[Explanation]",
    "MAE =",
    "RMSE =",
    "BLEU:",
    "METEOR:",
    "ROUGE:",
    "Eval elapsed:",
    "[step3 runner]",
    "[step4 runner]",
    "[step5 runner]",
    "[Manifest] wrote",
)


def _torchrun_cmd() -> list[str]:
    if shutil.which("torchrun"):
        return ["torchrun"]
    return [sys.executable, "-m", "torch.distributed.run"]


def _scrub_odcr_env(env: dict[str, str]) -> None:
    for k in list(env.keys()):
        if k.startswith("ODCR_"):
            del env[k]


def _scrub_training_side_env(env: dict[str, str]) -> None:
    """torchrun 子进程启动前清洗环境。

    移除父 shell 中的 ``TRAIN_*``、``EVAL_BATCH_SIZE``、``MAX_PARALLEL_CPU`` 以及**全部** ``ODCR_*``
    （随后由 :func:`_odcr_layout_env` 再注入 resolver-owned ``ODCR_RESOLVED_*`` 白名单变量）。
    因此 **export TRAIN_* / ODCR_QUICK_EVAL_***
    **不会**稳定影响子进程；训练语义请以 **CLI + configs/odcr.yaml** 为准。见 README 与主指南 §3。
    """
    _scrub_odcr_env(env)
    for k in list(env.keys()):
        if k.startswith("TRAIN_"):
            del env[k]
    for k in ("EVAL_BATCH_SIZE", "MAX_PARALLEL_CPU"):
        env.pop(k, None)


def _odcr_layout_env(cfg: ResolvedConfig) -> dict[str, str]:
    meta = str(path_layout.get_task_meta_dir(cfg.repo_root, cfg.task_id, cfg.iteration_id))
    log_paths = run_log_paths(cfg)
    out: dict[str, str] = {
        "ODCR_ROOT": str(cfg.repo_root),
        "ODCR_STAGE_RUN_DIR": str(Path(cfg.checkpoint_dir).resolve()),
        "ODCR_ITERATION_META_DIR": meta,
        "ODCR_MANIFEST_DIR": str(Path(cfg.manifest_dir).resolve()),
        "ODCR_LOG_DIR": str(Path(cfg.manifest_dir).resolve()),
        "ODCR_DUAL_TRAIN_LOG": "1",
        "ODCR_SUMMARY_LOG": str(log_paths["console"]),
        "ODCR_LOG_CONSOLE": "0",
        "ODCR_RESOLVED_DATA_DIR": str(Path(cfg.data_dir).resolve()),
        "ODCR_RESOLVED_MERGED_DIR": str(Path(cfg.merged_dir).resolve()),
        "ODCR_RESOLVED_RUNS_DIR": str(Path(cfg.runs_dir).resolve()),
        "ODCR_RESOLVED_CACHE_DIR": str(Path(cfg.cache_dir).resolve()),
        "ODCR_RESOLVED_MODELS_DIR": str(Path(cfg.models_dir).resolve()),
        "ODCR_RESOLVED_STEP5_TEXT_MODEL": str(Path(cfg.step5_text_model).resolve()),
        "ODCR_RESOLVED_SENTENCE_EMBED_MODEL": str(Path(cfg.sentence_embed_model).resolve()),
        "ODCR_RESOLVED_EMBED_DIM": str(int(cfg.embed_dim)),
    }
    _tp = getattr(cfg, "effective_training_payload_json", "") or ""
    if _tp.strip():
        out["ODCR_EFFECTIVE_TRAINING_PAYLOAD_JSON"] = _tp
    _fs = (getattr(cfg, "config_field_sources_json", "") or "").strip()
    if _fs:
        out["ODCR_CONFIG_FIELD_SOURCES_JSON"] = _fs
    if cfg.eval_run_dir:
        out["ODCR_EVAL_RUN_DIR"] = str(Path(cfg.eval_run_dir).resolve())
    if cfg.command == "step4" and cfg.step3_checkpoint_dir:
        out["ODCR_STEP3_RUN_DIR"] = str(Path(cfg.step3_checkpoint_dir).resolve())
    _ur = (getattr(cfg, "upstream_resolution_json", "") or "").strip()
    if _ur:
        out["ODCR_UPSTREAM_RESOLUTION_JSON"] = _ur
    _tfp = (getattr(cfg, "training_semantic_fingerprint", "") or "").strip()
    if _tfp:
        out["ODCR_TRAINING_SEMANTIC_FINGERPRINT"] = _tfp
    _gfp = (getattr(cfg, "generation_semantic_fingerprint", "") or "").strip()
    if _gfp:
        out["ODCR_GENERATION_SEMANTIC_FINGERPRINT"] = _gfp
    _rd = (getattr(cfg, "runtime_diagnostics_fingerprint", "") or "").strip()
    if _rd:
        out["ODCR_RUNTIME_DIAGNOSTICS_FINGERPRINT"] = _rd
    _tr = (getattr(cfg, "thread_env_requested_json", "") or "").strip()
    if _tr:
        out["ODCR_THREAD_ENV_REQUESTED_JSON"] = _tr
    _te = (getattr(cfg, "thread_env_effective_json", "") or "").strip()
    if _te:
        out["ODCR_THREAD_ENV_EFFECTIVE_JSON"] = _te
    _lr = (getattr(cfg, "launcher_env_requested_json", "") or "").strip()
    if _lr:
        out["ODCR_LAUNCHER_ENV_REQUESTED_JSON"] = _lr
    _le = (getattr(cfg, "launcher_env_effective_json", "") or "").strip()
    if _le:
        out["ODCR_LAUNCHER_ENV_EFFECTIVE_JSON"] = _le
    if cfg.command == "step3":
        out["ODCR_TRAINING_STAGE"] = "step3"
        out["ODCR_STEP3_TOKENIZER_CACHE_STARTUP_JSON"] = str(
            (Path(cfg.manifest_dir).resolve() / "step3_tokenizer_cache_startup.json")
        )
    return out


def _run_torchrun_explicit(
    *,
    code_dir: Path,
    repo_root: Path,
    ddp_world_size: int,
    env_extra: dict[str, str],
    script: str,
    py_args: list[str],
) -> None:
    cmd = [
        *_torchrun_cmd(),
        "--standalone",
        f"--nproc_per_node={ddp_world_size}",
        script,
        *py_args,
    ]
    env = _base_env_raw(repo_root)
    env.update(env_extra)
    _ensure_code_dir_on_pythonpath(env, code_dir)
    print("[odcr] cwd:", code_dir, flush=True)
    print("[odcr] exec:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(code_dir), env=env, check=True)


def _base_env_raw(repo_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    _scrub_training_side_env(env)
    env["ODCR_ROOT"] = str(repo_root)
    return env


def _ensure_code_dir_on_pythonpath(env: dict[str, str], code_dir: Path) -> None:
    """torchrun 以 ``executors/*.py`` 为入口时，sys.path 首项为 ``code/executors``，无法 ``import executors``。"""
    root = str(code_dir.resolve())
    prev = env.get("PYTHONPATH", "").strip()
    env["PYTHONPATH"] = root if not prev else f"{root}{os.pathsep}{prev}"


def _print_startup_runtime_env(cfg: ResolvedConfig) -> None:
    print(
        "[startup_runtime_env] "
        f"launcher_requested={getattr(cfg, 'launcher_env_requested_json', '')} "
        f"launcher_effective={getattr(cfg, 'launcher_env_effective_json', '')} "
        f"thread_requested={getattr(cfg, 'thread_env_requested_json', '')} "
        f"thread_effective={getattr(cfg, 'thread_env_effective_json', '')}",
        flush=True,
    )


def _maybe_write_run_manifest(cfg: ResolvedConfig) -> None:
    """torchrun 前落盘，保证训练崩溃后仍能拿到本次解析结果。"""
    data = build_run_manifest(cfg)
    p = write_run_manifest_json(cfg, data)
    print(f"[Manifest] wrote {p}", flush=True)


def _line_is_error_or_warning(line: str) -> bool:
    low = line.lower()
    return any(token in low for token in _ERROR_LINE_TOKENS)


def _line_is_console_summary(line: str) -> bool:
    return any(token in line for token in _SUMMARY_LINE_TOKENS)


def _record_child_output(cfg: ResolvedConfig, line: str, *, console_level: str) -> None:
    if not line:
        return
    append_debug_log(cfg, [line])
    append_full_log(cfg, [f"[raw child] {line}"])
    if _line_is_error_or_warning(line):
        append_error_log(cfg, [f"[child stream] {line}"])
    level = normalize_console_level(console_level)
    if level in {"verbose", "debug"} or _line_is_error_or_warning(line) or _line_is_console_summary(line):
        emit_console_lines(cfg, [line])


def _run_torchrun(
    cfg: ResolvedConfig,
    *,
    env_extra: dict[str, str],
    script: str,
    py_args: list[str],
    console_level: str = "summary",
) -> None:
    env = dict(_base_env_raw(cfg.repo_root))
    env.update(_odcr_layout_env(cfg))
    env.update(_torchrun_hardware_env(cfg))
    env.update(env_extra)
    _ensure_code_dir_on_pythonpath(env, cfg.code_dir)
    cmd = [
        *_torchrun_cmd(),
        "--standalone",
        f"--nproc_per_node={cfg.ddp_world_size}",
        script,
        *py_args,
    ]
    launcher_lines = [
        f"[odcr launcher] cwd={cfg.code_dir}",
        "[odcr launcher] exec=" + " ".join(cmd),
        (
            "[startup_runtime_env] "
            f"launcher_requested={getattr(cfg, 'launcher_env_requested_json', '')} "
            f"launcher_effective={getattr(cfg, 'launcher_env_effective_json', '')} "
            f"thread_requested={getattr(cfg, 'thread_env_requested_json', '')} "
            f"thread_effective={getattr(cfg, 'thread_env_effective_json', '')}"
        ),
    ]
    append_debug_log(cfg, launcher_lines)
    append_full_log(cfg, ["========== LAUNCHER COMMAND ==========", *launcher_lines])
    if normalize_console_level(console_level) in {"verbose", "debug"}:
        emit_console_lines(cfg, launcher_lines)

    started = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        cwd=str(cfg.code_dir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for raw in proc.stdout:
        _record_child_output(cfg, raw.rstrip("\n"), console_level=console_level)
    returncode = proc.wait()
    elapsed = time.monotonic() - started
    append_debug_log(cfg, [f"[odcr launcher] returncode={returncode} elapsed={elapsed:.3f}s"])
    append_full_log(cfg, [f"[odcr launcher] returncode={returncode} elapsed={elapsed:.3f}s"])
    if returncode != 0:
        append_error_log(cfg, [f"[odcr launcher] failed returncode={returncode} elapsed={elapsed:.3f}s"])
        raise subprocess.CalledProcessError(returncode, cmd)


def _full_log_file(cfg: ResolvedConfig) -> str:
    return str(run_log_paths(cfg)["full"])


@contextmanager
def _patched_env(updates: dict[str, str]):
    old: dict[str, str | None] = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _write_step3_parent_initial_runtime_config(cfg: ResolvedConfig) -> None:
    hardware_profile = json.loads(cfg.hardware_profile_json or "{}")
    worker_budget = hardware_profile.get("worker_budget_formula") if isinstance(hardware_profile, dict) else {}
    reserved_cpu = int(worker_budget.get("reserved_cpu", 2)) if isinstance(worker_budget, dict) else 2
    payload = {
        "phase": "parent_pre_ddp_initial",
        "status": "initial",
        "stage": "step3",
        "task_id": int(cfg.task_id),
        "source_domain": str(cfg.auxiliary),
        "target_domain": str(cfg.target),
        "training_loop_started": False,
        "checkpoint_created": False,
        "cache_status": "not_started",
        "num_proc": int(cfg.num_proc),
        "max_parallel_cpu": int(getattr(cfg, "max_parallel_cpu", 0) or 0),
        "reserved_cpu": reserved_cpu,
        "tokenization_formula": (
            f"num_proc({int(cfg.num_proc)}) + reserved_cpu({reserved_cpu}) "
            f"<= max_parallel_cpu({int(getattr(cfg, 'max_parallel_cpu', 0) or 0)})"
        ),
        "worker_formula": (
            f"dataloader_num_workers_train({int(getattr(cfg, 'dataloader_num_workers_train', 0) or 0)}) "
            f"* ddp_world_size({int(getattr(cfg, 'ddp_world_size', 1) or 1)}) + reserved_cpu({reserved_cpu}) "
            f"<= max_parallel_cpu({int(getattr(cfg, 'max_parallel_cpu', 0) or 0)})"
        ),
        "omp_num_threads": int(cfg.omp_num_threads),
        "mkl_num_threads": int(cfg.mkl_num_threads),
        "tokenizers_parallelism": bool(cfg.tokenizers_parallelism),
        "hardware_profile": hardware_profile,
        "thread_env_requested": json.loads(getattr(cfg, "thread_env_requested_json", "") or "{}"),
        "thread_env_effective": json.loads(getattr(cfg, "thread_env_effective_json", "") or "{}"),
        "launcher_env_requested": json.loads(getattr(cfg, "launcher_env_requested_json", "") or "{}"),
        "launcher_env_effective": json.loads(getattr(cfg, "launcher_env_effective_json", "") or "{}"),
        "training_semantic_fingerprint": getattr(cfg, "training_semantic_fingerprint", "") or None,
        "generation_semantic_fingerprint": getattr(cfg, "generation_semantic_fingerprint", "") or None,
        "runtime_diagnostics_fingerprint": getattr(cfg, "runtime_diagnostics_fingerprint", "") or None,
    }
    write_training_runtime_config_artifact(cfg.manifest_dir, payload)


def _ensure_step3_pre_ddp_tokenizer_cache(cfg: ResolvedConfig, *, log_file: str, model_path: str) -> None:
    from executors.step3_train_core import ensure_step3_tokenizer_cache_ready_pre_ddp
    from train_logging import LOGGER_NAME, setup_train_logging

    env = dict(_odcr_layout_env(cfg))
    env.update(_torchrun_hardware_env(cfg))
    with _patched_env(env):
        _write_step3_parent_initial_runtime_config(cfg)
        setup_train_logging(
            log_file=log_file,
            task_idx=int(cfg.task_id),
            rank=0,
            world_size=1,
            run_id=str(cfg.run_name or ""),
        )
        args = SimpleNamespace(
            auxiliary=cfg.auxiliary,
            target=cfg.target,
            num_proc=cfg.num_proc,
            seed=cfg.seed,
            checkpoint_metric="valid_loss",
            log_file=log_file,
            save_file=model_path,
        )
        try:
            ensure_step3_tokenizer_cache_ready_pre_ddp(
                args,
                rank="parent",
                world_size=int(cfg.ddp_world_size),
                build_allowed=True,
                log_tokenize=True,
                show_datasets_progress=True,
            )
        finally:
            import logging

            logging.getLogger(LOGGER_NAME).handlers.clear()


def _run_step3_train(cfg: ResolvedConfig, *, console_level: str = "summary") -> None:
    assert cfg.run_name is not None
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
    log_file = _full_log_file(cfg)
    model_path = str(path_layout.best_model_path(Path(cfg.checkpoint_dir)))
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    _ensure_step3_pre_ddp_tokenizer_cache(cfg, log_file=log_file, model_path=model_path)

    py_args = [
        "train",
        "--auxiliary",
        cfg.auxiliary,
        "--target",
        cfg.target,
        "--num-proc",
        str(cfg.num_proc),
        "--seed",
        str(cfg.seed),
        "--checkpoint-metric",
        "valid_loss",
        "--log_file",
        log_file,
        "--save_file",
        model_path,
    ]
    _run_torchrun(cfg, env_extra={}, script=TORCHRUN_STEP3_SCRIPT, py_args=py_args, console_level=console_level)


def _run_step3_eval(cfg: ResolvedConfig, *, console_level: str = "summary") -> None:
    assert cfg.run_name is not None
    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
    log_file = _full_log_file(cfg)
    model_path = str(path_layout.best_model_path(Path(cfg.checkpoint_dir)))
    py_args = [
        "eval",
        "--auxiliary",
        cfg.auxiliary,
        "--target",
        cfg.target,
        "--batch-size",
        str(cfg.valid_batch_size),
        "--num-proc",
        str(cfg.num_proc),
        "--seed",
        str(cfg.seed),
        "--log_file",
        log_file,
        "--save_file",
        model_path,
        "--eval-protocol",
        str(getattr(cfg, "step3_eval_protocol", "minimal_eval") or "minimal_eval"),
        "--eval-split",
        str(getattr(cfg, "step3_eval_split", "valid") or "valid"),
    ]
    _run_torchrun(cfg, env_extra={}, script=TORCHRUN_STEP3_SCRIPT, py_args=py_args, console_level=console_level)


def run_step3(cfg: ResolvedConfig, *, console_level: str = "summary") -> None:
    assert cfg.run_name is not None
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
    _maybe_write_run_manifest(cfg)

    mode = cfg.step3_mode
    if mode == "eval_only":
        _run_step3_eval(cfg, console_level=console_level)
        return
    _run_step3_train(cfg, console_level=console_level)
    if mode == "full":
        _run_step3_eval(cfg, console_level=console_level)


def run_step4(cfg: ResolvedConfig, *, console_level: str = "summary") -> None:
    assert cfg.from_run is not None
    if cfg.global_eval_batch_size is None:
        raise RuntimeError(
            "内部错误: step4 缺少 global_eval_batch_size；应在 config_resolver 中由 eval profile 解析 eval_batch_size。"
        )
    if cfg.eval_per_gpu_batch_size is None:
        raise RuntimeError("内部错误: step4 缺少 eval_per_gpu_batch_size（应与 global_eval_batch_size 同时解析）。")
    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
    _maybe_write_run_manifest(cfg)
    log_file = _full_log_file(cfg)

    g_eval = int(cfg.global_eval_batch_size)
    e_per_gpu = int(cfg.eval_per_gpu_batch_size)
    from odcr_core.step4_runtime import (
        prepare_step4_encoded_cache,
        reject_step4_formal_env_overrides,
        step4_runtime_env,
    )

    reject_step4_formal_env_overrides(mode="formal")
    prepare_step4_encoded_cache(cfg, dry_run=False, build_allowed=True)
    env_extra = dict(_odcr_profile_env(cfg))
    # Step4 导出到 numpy 路径不接受 bf16，强制用 fp32。
    env_extra["ODCR_RUNTIME_PRECISION_MODE"] = "fp32"
    env_extra["ODCR_GLOBAL_EVAL_BATCH_SIZE"] = str(g_eval)
    env_extra["ODCR_EVAL_PER_GPU_BATCH_SIZE"] = str(e_per_gpu)
    env_extra["ODCR_STEP4_RCR_CONFIG_JSON"] = str(getattr(cfg, "step4_rcr_config_json", "") or "")
    env_extra.update(step4_runtime_env(cfg, mode="formal"))
    py_args = [
        "--task",
        str(cfg.task_id),
        "--batch-size",
        str(g_eval),
        "--num-proc",
        str(cfg.num_proc),
        "--log_file",
        log_file,
    ]
    _run_torchrun(cfg, env_extra=env_extra, script=TORCHRUN_STEP4_SCRIPT, py_args=py_args, console_level=console_level)


def _step5_decode_cli_args(cfg: ResolvedConfig) -> list[str]:
    """与 One-Control decode 解析结果对齐，传给 step5_entry（train/eval）。"""
    out: list[str] = [
        "--decode-strategy",
        str(cfg.decode_strategy),
        "--max-explanation-length",
        str(cfg.max_explanation_length),
        "--label-smoothing",
        str(cfg.label_smoothing),
        "--repetition-penalty",
        str(cfg.repetition_penalty),
        "--generate-temperature",
        str(cfg.generate_temperature),
        "--generate-top-p",
        str(cfg.generate_top_p),
    ]
    if cfg.decode_seed is not None:
        out.extend(["--decode-seed", str(cfg.decode_seed)])
    if cfg.no_repeat_ngram_size is not None:
        out.extend(["--no-repeat-ngram-size", str(cfg.no_repeat_ngram_size)])
    if cfg.min_len is not None:
        out.extend(["--min-len", str(cfg.min_len)])
    return out


def _odcr_profile_env(cfg: ResolvedConfig) -> dict[str, str]:
    out: dict[str, str] = {
        "ODCR_DECODE_PROFILE_JSON": cfg.decode_profile_json,
        "ODCR_RERANK_PROFILE_JSON": cfg.rerank_profile_json,
        "ODCR_DECODE_PRESET_STEM": str(cfg.decode_preset_id),
        "ODCR_RERANK_PRESET_STEM": str(cfg.rerank_preset_id or ""),
    }
    if getattr(cfg, "eval_profile_id", "") and cfg.command in (
        "eval",
        "eval-rerank",
        "eval-matrix",
        "eval-rerank-matrix",
        "step4",
    ):
        out["ODCR_EVAL_PROFILE_NAME"] = str(cfg.eval_profile_id)
    return out


def _runtime_precision_mode(cfg: ResolvedConfig) -> str:
    precision = str(getattr(cfg, "train_precision", "") or "").strip().lower()
    if precision not in {"bf16", "fp16", "fp32"}:
        raise RuntimeError(
            "ResolvedConfig.train_precision must be one of bf16/fp16/fp32 before torchrun launch; "
            "use configs/odcr.yaml stage.train.backend.train_precision."
        )
    return precision


def _torchrun_hardware_env(cfg: ResolvedConfig) -> dict[str, str]:
    """显式注入子进程：hardware JSON/stem、runtime precision transport、线程/CUDA launcher env。"""
    out: dict[str, str] = {
        "ODCR_HARDWARE_PROFILE_JSON": cfg.hardware_profile_json,
        "ODCR_HARDWARE_PRESET": str(cfg.hardware_preset_id),
        "ODCR_RUNTIME_PRECISION_MODE": _runtime_precision_mode(cfg),
        "ODCR_RUNTIME_ALLOW_TF32": "1" if bool(getattr(cfg, "allow_tf32", False)) else "0",
        "ODCR_RUNTIME_AMP_AUTOCAST": "1" if bool(getattr(cfg, "amp_autocast", True)) else "0",
        "ODCR_RUNTIME_GRAD_SCALER": "1" if bool(getattr(cfg, "grad_scaler", False)) else "0",
        "ODCR_STEP3_TOKENIZER_MAX_LENGTH": str(int(getattr(cfg, "tokenizer_max_length", 0) or 0)),
        "ODCR_STEP3_EVIDENCE_MAX_LENGTH": str(int(getattr(cfg, "evidence_max_length", 0) or 0)),
        "OMP_NUM_THREADS": str(int(cfg.omp_num_threads)),
        "MKL_NUM_THREADS": str(int(cfg.mkl_num_threads)),
        "TOKENIZERS_PARALLELISM": "true" if cfg.tokenizers_parallelism else "false",
    }
    try:
        _le = json.loads(getattr(cfg, "launcher_env_effective_json", "") or "{}")
    except json.JSONDecodeError:
        _le = {}
    if isinstance(_le, dict):
        cvd = _le.get("CUDA_VISIBLE_DEVICES")
        if cvd is not None and str(cvd).strip() != "":
            out["CUDA_VISIBLE_DEVICES"] = str(cvd).strip()
    return out


def run_step5(cfg: ResolvedConfig, *, console_level: str = "summary") -> None:
    assert cfg.from_run is not None and cfg.step5_run is not None
    ensure_step5_csv_symlink(cfg)
    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
    _maybe_write_run_manifest(cfg)
    log_file = _full_log_file(cfg)

    env_extra = dict(_odcr_profile_env(cfg))
    model_path = str(path_layout.best_model_path(Path(cfg.checkpoint_dir)))
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    py_args = [
        "train",
        "--auxiliary",
        cfg.auxiliary,
        "--target",
        cfg.target,
        "--num-proc",
        str(cfg.num_proc),
        "--seed",
        str(cfg.seed),
        "--log_file",
        log_file,
        "--save_file",
        model_path,
        *_step5_decode_cli_args(cfg),
    ]
    if cfg.step5_train_only:
        py_args.append("--train-only")
        stub = run_log_paths(cfg)["debug"]
        stub.parent.mkdir(parents=True, exist_ok=True)
        with stub.open("a", encoding="utf-8") as fh:
            fh.write("step5 --train-only：本次跳过训练后 valid 评估；完整指标请运行: python code/odcr.py eval …\n")
    _run_torchrun(cfg, env_extra=env_extra, script=TORCHRUN_STEP5_SCRIPT, py_args=py_args, console_level=console_level)


def run_eval(cfg: ResolvedConfig, *, console_level: str = "summary") -> None:
    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
    _maybe_write_run_manifest(cfg)
    log_file = _full_log_file(cfg)

    mp = Path(cfg.model_path).expanduser().resolve() if cfg.model_path else None
    if mp is None:
        assert cfg.from_run is not None and cfg.step5_run is not None
        ck = Path(cfg.checkpoint_dir)
        mp = path_layout.best_model_path(ck)

    if not mp.is_file():
        raise FileNotFoundError(f"评测权重不存在: {mp}")

    env_extra: dict[str, str] = dict(_odcr_profile_env(cfg))

    py_args = [
        "eval",
        "--auxiliary",
        cfg.auxiliary,
        "--target",
        cfg.target,
        "--eval-batch-size",
        str(cfg.global_eval_batch_size),
        "--num-proc",
        str(cfg.num_proc),
        "--seed",
        str(cfg.seed),
        "--log_file",
        log_file,
        "--save_file",
        str(mp),
        *_step5_decode_cli_args(cfg),
    ]
    _run_torchrun(cfg, env_extra=env_extra, script=TORCHRUN_STEP5_SCRIPT, py_args=py_args, console_level=console_level)


def _rerank_runner_cli_args(cfg: ResolvedConfig) -> list[str]:
    out = [
        "--num-return-sequences",
        str(cfg.num_return_sequences),
        "--rerank-method",
        str(cfg.rerank_method),
        "--rerank-top-k",
        str(cfg.rerank_top_k),
        "--rerank-weight-logprob",
        str(cfg.rerank_weight_logprob),
        "--rerank-weight-length",
        str(cfg.rerank_weight_length),
        "--rerank-weight-repeat",
        str(cfg.rerank_weight_repeat),
        "--rerank-weight-dirty",
        str(cfg.rerank_weight_dirty),
        "--rerank-target-len-ratio",
        str(cfg.rerank_target_len_ratio),
        "--export-examples-mode",
        str(cfg.export_examples_mode),
        "--rerank-malformed-tail-penalty",
        str(cfg.rerank_malformed_tail_penalty),
        "--rerank-malformed-token-penalty",
        str(cfg.rerank_malformed_token_penalty),
    ]
    if cfg.export_full_rerank_examples:
        out.append("--export-full-rerank-examples")
    return out


def run_eval_rerank(cfg: ResolvedConfig, *, console_level: str = "summary") -> None:
    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
    _maybe_write_run_manifest(cfg)
    log_file = _full_log_file(cfg)

    mp = Path(cfg.model_path).expanduser().resolve() if cfg.model_path else None
    if mp is None:
        assert cfg.from_run is not None and cfg.step5_run is not None
        ck = Path(cfg.checkpoint_dir)
        mp = path_layout.best_model_path(ck)

    if not mp.is_file():
        raise FileNotFoundError(f"评测权重不存在: {mp}")

    env_extra: dict[str, str] = dict(_odcr_profile_env(cfg))

    py_args = [
        "eval-rerank",
        "--auxiliary",
        cfg.auxiliary,
        "--target",
        cfg.target,
        "--eval-batch-size",
        str(cfg.global_eval_batch_size),
        "--num-proc",
        str(cfg.num_proc),
        "--seed",
        str(cfg.seed),
        "--log_file",
        log_file,
        "--save_file",
        str(mp),
        *_step5_decode_cli_args(cfg),
        *_rerank_runner_cli_args(cfg),
    ]
    _run_torchrun(cfg, env_extra=env_extra, script=TORCHRUN_STEP5_SCRIPT, py_args=py_args, console_level=console_level)


    print(f"Step3 产物: {cfg3.checkpoint_dir}", flush=True)
    print(f"Step5 产物: {cfg5.checkpoint_dir}", flush=True)
