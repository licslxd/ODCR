"""Step3 rating paper-eval seed runner.

Public workflows are deliberately small:

* single: evaluate one seed against one existing Step3 checkpoint.
* multi: evaluate the fixed five seeds against one existing Step3 checkpoint.

This runner does not train, does not create formal Step3 run directories, and
does not accept/rewrite formal ``meta/eval_handoff.json``. Its outputs are
overwritten evaluation reports under ``runs/step3/task<T>/eval/{1,5}``.
"""
from __future__ import annotations

import csv
import json
import math
import os
import shutil
import socket
import subprocess
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from odcr_core import path_layout
from odcr_core.config_resolver import resolve_config, write_resolved_config
from odcr_core.file_atomic import atomic_write_json
from odcr_core.step3_eval_handoff import PAPER_TARGET_ONLY_EVAL


DEFAULT_RATING_SEEDS: tuple[int, ...] = (3407, 1337, 1234, 5678, 9012)
EVAL_RUNNER_SCHEMA_VERSION = "odcr_step3_rating_eval_runner/1"
EVAL_MEAN_STD_SCHEMA_VERSION = "odcr_step3_rating_eval_5seed_mean_std/1"
TASK_SAMPLE_COUNTS: dict[int, dict[str, int]] = {
    2: {"valid": 109732, "test": 109720},
}


class Step3RatingSeedRunnerError(RuntimeError):
    """Raised when the Step3 rating eval runner cannot safely proceed."""


@dataclass(frozen=True)
class RatingSeedRun:
    seed: int
    run_id: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _repo_relative(repo_root: Path, path: str | Path) -> str:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = (repo_root / p).resolve()
    else:
        p = p.resolve()
    try:
        return p.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise Step3RatingSeedRunnerError(f"{label} missing: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise Step3RatingSeedRunnerError(f"{label} is not valid JSON: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise Step3RatingSeedRunnerError(f"{label} root must be a JSON object: {path}")
    return data


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Step3RatingSeedRunnerError(message)


def _source_run_root(repo_root: Path, task: int, run_id: str) -> Path:
    return path_layout.get_stage_run_root(repo_root, int(task), "v1", "step3", str(run_id)).resolve()


def _source_checkpoint(repo_root: Path, task: int, run_id: str) -> Path:
    return path_layout.best_model_path(_source_run_root(repo_root, task, run_id)).resolve()


def _eval_count(mode: str) -> int:
    return 1 if str(mode).strip().lower() == "single" else 5


def _eval_root(repo_root: Path, task: int, count: int) -> Path:
    return repo_root / "runs" / "step3" / f"task{int(task)}" / "eval" / str(int(count))


def _eval_seed_dir(repo_root: Path, task: int, count: int, seed: int) -> Path:
    return _eval_root(repo_root, task, count) / f"seed{int(seed)}"


def _eval_split_log_dir(repo_root: Path, task: int, count: int, seed: int, split: str) -> Path:
    return _eval_seed_dir(repo_root, task, count, seed) / split


def _eval_artifact_dir(repo_root: Path, task: int, count: int, seed: int, split: str) -> Path:
    return _eval_split_log_dir(repo_root, task, count, seed, split) / f"eval_{PAPER_TARGET_ONLY_EVAL}_{split}"


def _single_report_paths(repo_root: Path, task: int) -> dict[str, Path]:
    base = _eval_root(repo_root, task, 1) / f"step3_rating_task{int(task)}_eval_1seed"
    return {"json": base.with_suffix(".json"), "md": base.with_suffix(".md")}


def _multi_report_paths(repo_root: Path, task: int) -> dict[str, Path]:
    base = _eval_root(repo_root, task, 5) / f"step3_rating_task{int(task)}_eval_5seed"
    return {
        "csv": Path(str(base) + "_runs.csv"),
        "json": Path(str(base) + "_mean_std.json"),
        "md": Path(str(base) + "_report.md"),
    }


def _driver_slug(task: int, count: int, run_id: str) -> str:
    return f"step3_rating_task{int(task)}_eval_{int(count)}seed_run{run_id}"


def _driver_paths(repo_root: Path, task: int, count: int, run_id: str) -> dict[str, Path]:
    slug = _driver_slug(task, count, run_id)
    launch_dir = repo_root / "test"
    return {
        "driver_log": launch_dir / f"{slug}.driver.nohup.log",
        "driver_pid": launch_dir / f"{slug}.driver.pid",
        "trace_log": launch_dir / f"{slug}.trace.log",
    }


def build_direct_nohup_command(
    repo_root: str | Path,
    *,
    task: int,
    mode: str,
    source_run_id: str,
    seed: int | None = None,
) -> list[str]:
    root = Path(repo_root).expanduser().resolve()
    count = _eval_count(mode)
    driver = _driver_paths(root, int(task), count, str(source_run_id))
    args = ["./odcr", "step3-rating", "--task", str(int(task)), "--mode", str(mode), "--run-id", str(source_run_id)]
    if str(mode).strip().lower() == "single":
        if seed is None:
            raise Step3RatingSeedRunnerError("single direct nohup command requires seed")
        args.extend(["--seed", str(int(seed))])
    return [
        f"cd {root}",
        f"LAUNCH_DIR={root / 'test'}",
        'mkdir -p "$LAUNCH_DIR"',
        f"nohup {' '.join(args)} > \"$LAUNCH_DIR/{driver['driver_log'].name}\" 2>&1 &",
        f'echo $! > "$LAUNCH_DIR/{driver["driver_pid"].name}"',
    ]


def _plan_single(seed: int, run_id: str) -> list[RatingSeedRun]:
    return [RatingSeedRun(seed=int(seed), run_id=str(run_id))]


def _plan_multi(source_run_id: str) -> list[RatingSeedRun]:
    return [RatingSeedRun(seed=seed, run_id=str(source_run_id)) for seed in DEFAULT_RATING_SEEDS]


def build_rating_seed_plan(
    *,
    task: int,
    mode: str,
    seed: int | None = None,
    run_id: str | None = None,
    run_id_start: int | None = None,
) -> list[RatingSeedRun]:
    _ = task
    value = str(mode or "").strip().lower()
    if value == "single":
        if seed is None or run_id in (None, "", "auto"):
            raise Step3RatingSeedRunnerError("single mode requires --seed and source --run-id")
        return _plan_single(int(seed), str(run_id))
    if value == "multi":
        source_run_id = str(run_id if run_id not in (None, "", "auto") else run_id_start if run_id_start is not None else "")
        if not source_run_id:
            raise Step3RatingSeedRunnerError("multi mode requires source --run-id")
        return _plan_multi(source_run_id)
    raise Step3RatingSeedRunnerError(
        f"unsupported Step3 rating mode {mode!r}; use single or multi"
    )


def _validate_source_checkpoint(repo_root: Path, *, task: int, run_id: str) -> dict[str, Any]:
    run_root = _source_run_root(repo_root, task, run_id)
    checkpoint = _source_checkpoint(repo_root, task, run_id)
    _require(run_root.is_dir(), f"source Step3 run dir missing: {_repo_relative(repo_root, run_root)}")
    _require(checkpoint.is_file(), f"source Step3 checkpoint missing: {_repo_relative(repo_root, checkpoint)}")
    status_path = run_root / "meta" / "stage_status.json"
    status = _load_json(status_path, label="source stage_status.json") if status_path.is_file() else {}
    if status:
        _require(str(status.get("stage") or "") == "step3", "source stage_status.stage must be step3")
        _require(int(status.get("task") or status.get("task_id") or -1) == int(task), "source stage_status task mismatch")
        _require(str(status.get("final_status") or "") != "failed", "source Step3 run is failed; cannot evaluate")
    return {
        "source_run_root": _repo_relative(repo_root, run_root),
        "source_checkpoint": _repo_relative(repo_root, checkpoint),
        "source_stage_status": _repo_relative(repo_root, status_path) if status_path.is_file() else None,
    }


def _append_trace(trace_log: Path | None, lines: Iterable[str]) -> None:
    if trace_log is None:
        return
    trace_log.parent.mkdir(parents=True, exist_ok=True)
    with trace_log.open("a", encoding="utf-8") as fh:
        for line in lines:
            fh.write(str(line).rstrip() + "\n")


def _assert_gpu_ready(trace_log: Path | None = None) -> dict[str, Any]:
    host = socket.gethostname()
    handshake = [
        f"[GPU HANDSHAKE] {_now_iso()}",
        f"pwd={Path.cwd()}",
        f"hostname={host}",
        f"TMUX={os.environ.get('TMUX', '')}",
        f"SLURM_JOB_ID={os.environ.get('SLURM_JOB_ID', '')}",
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}",
    ]
    for line in handshake:
        print(line, flush=True)
    _append_trace(trace_log, handshake)
    if host == "admin" or host.startswith("admin"):
        raise Step3RatingSeedRunnerError(
            "Current tmux does not expose CUDA. Please manually run `odcr-enter-gpu <JOBID>` "
            "in this same tmux to enter the GPU node, then rerun the probe."
        )
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    try:
        smi = subprocess.run(["nvidia-smi"], text=True, capture_output=True, check=True)
    except Exception as exc:  # pragma: no cover - depends on runtime env
        raise Step3RatingSeedRunnerError(f"nvidia-smi failed during CUDA handshake: {exc}") from exc
    print(smi.stdout, end="", flush=True)
    _append_trace(trace_log, smi.stdout.splitlines())
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on runtime env
        raise Step3RatingSeedRunnerError(f"torch import failed during CUDA handshake: {exc}") from exc
    available = bool(torch.cuda.is_available())
    count = int(torch.cuda.device_count()) if available else 0
    cuda_lines = [
        f"torch.cuda.is_available = {available}",
        f"torch.cuda.device_count = {count}",
    ]
    if available:
        cuda_lines.append(f"devices = {[torch.cuda.get_device_name(i) for i in range(count)]}")
    for line in cuda_lines:
        print(line, flush=True)
    _append_trace(trace_log, cuda_lines)
    if not cuda_visible or not available or count < 2:
        raise Step3RatingSeedRunnerError(
            "Current tmux does not expose CUDA. Please manually run `odcr-enter-gpu <JOBID>` "
            "in this same tmux to enter the GPU node, then rerun the probe."
        )
    return {
        "hostname": host,
        "cuda_visible_devices": cuda_visible,
        "torch_cuda_available": available,
        "torch_cuda_device_count": count,
    }


def _eval_overrides(seed: int, split: str) -> list[str]:
    return [
        f"project.seed={int(seed)}",
        f"step3.eval.protocol={PAPER_TARGET_ONLY_EVAL}",
        f"step3.eval.split={split}",
    ]


def _run_paper_eval_split(
    *,
    repo_root: Path,
    config_path: str,
    task: int,
    source_run_id: str,
    seed: int,
    count: int,
    split: str,
    console_level: str,
) -> None:
    from odcr_core.runners import _run_step3_eval

    cfg, _sources, snapshot = resolve_config(
        config_path=config_path,
        command="step3",
        task_id=int(task),
        set_overrides=_eval_overrides(seed, split),
        dry_run=True,
        run_id=str(source_run_id),
        mode="eval_only",
    )
    log_dir = _eval_split_log_dir(repo_root, task, count, seed, split)
    eval_cfg = replace(
        cfg,
        log_dir=str(log_dir),
        manifest_dir=str(log_dir),
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    write_resolved_config(eval_cfg, snapshot, dry_run=False)
    with (log_dir / "full.log").open("a", encoding="utf-8") as fh, redirect_stdout(fh), redirect_stderr(fh):
        print(
            f"[STEP3_RATING_EVAL_START] task={task} source_run_id={source_run_id} "
            f"seed={seed} split={split} at={_now_iso()}",
            flush=True,
        )
        _run_step3_eval(eval_cfg, console_level=console_level)
        print(
            f"[STEP3_RATING_EVAL_DONE] task={task} source_run_id={source_run_id} "
            f"seed={seed} split={split} at={_now_iso()}",
            flush=True,
        )


def _metric(metrics: Mapping[str, Any], key: str) -> float:
    rec = metrics.get("recommendation")
    if isinstance(rec, Mapping):
        lookup = {"MAE": "mae", "RMSE": "rmse"}
        value = rec.get(lookup.get(key, key.lower()))
    else:
        value = metrics.get(key) if key in metrics else metrics.get(key.lower())
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise Step3RatingSeedRunnerError(f"metric {key} is missing or not numeric: {value!r}") from exc
    if not math.isfinite(out):
        raise Step3RatingSeedRunnerError(f"metric {key} must be finite: {value!r}")
    return out


def _load_eval_split(repo_root: Path, *, task: int, count: int, seed: int, split: str) -> dict[str, Any]:
    artifact_dir = _eval_artifact_dir(repo_root, task, count, seed, split)
    summary = _load_json(artifact_dir / "eval_summary.json", label=f"{split} eval_summary.json")
    integrity = _load_json(artifact_dir / "sample_integrity_report.json", label=f"{split} sample_integrity_report.json")
    _require(str(summary.get("eval_status")) == "completed", f"{split} eval_status must be completed")
    _require(str(summary.get("eval_protocol")) == PAPER_TARGET_ONLY_EVAL, f"{split} eval protocol mismatch")
    _require(bool(summary.get("target_only")) is True, f"{split} target_only must be true")
    _require(str(integrity.get("status")) == "PASS", f"{split} sample integrity must PASS")
    count_value = int(summary.get("sample_count") or integrity.get("sample_count") or 0)
    expected = TASK_SAMPLE_COUNTS.get(int(task), {}).get(split)
    if expected is not None:
        _require(count_value == int(expected), f"{split} sample_count mismatch: {count_value} != {expected}")
    metrics = summary.get("metrics")
    _require(isinstance(metrics, Mapping), f"{split} metrics missing")
    return {
        "eval_summary": _repo_relative(repo_root, artifact_dir / "eval_summary.json"),
        "sample_integrity": _repo_relative(repo_root, artifact_dir / "sample_integrity_report.json"),
        "sample_count": count_value,
        "mae": _metric(metrics, "MAE"),
        "rmse": _metric(metrics, "RMSE"),
    }


def _write_seed_handoff(
    repo_root: Path,
    *,
    task: int,
    count: int,
    seed: int,
    source_run_id: str,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    valid = _load_eval_split(repo_root, task=task, count=count, seed=seed, split="valid")
    test = _load_eval_split(repo_root, task=task, count=count, seed=seed, split="test")
    payload = {
        "schema_version": "odcr_step3_rating_eval_seed_handoff/1",
        "generated_at": _now_iso(),
        "stage": "step3_rating_eval",
        "task": int(task),
        "seed": int(seed),
        "source_run_id": str(source_run_id),
        "source_checkpoint": source["source_checkpoint"],
        "metric_protocol": PAPER_TARGET_ONLY_EVAL,
        "target_only": True,
        "valid": valid,
        "test": test,
    }
    path = _eval_seed_dir(repo_root, task, count, seed) / "eval_handoff.json"
    atomic_write_json(path, payload)
    payload["eval_handoff"] = _repo_relative(repo_root, path)
    return payload


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _sample_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        raise Step3RatingSeedRunnerError("sample std requires at least two values")
    mu = _mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1))


def _summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid_mae = [float(row["valid"]["mae"]) for row in rows]
    valid_rmse = [float(row["valid"]["rmse"]) for row in rows]
    test_mae = [float(row["test"]["mae"]) for row in rows]
    test_rmse = [float(row["test"]["rmse"]) for row in rows]
    return {
        "valid": {
            "mae_mean": _mean(valid_mae),
            "mae_std": _sample_std(valid_mae),
            "rmse_mean": _mean(valid_rmse),
            "rmse_std": _sample_std(valid_rmse),
        },
        "test": {
            "mae_mean": _mean(test_mae),
            "mae_std": _sample_std(test_mae),
            "rmse_mean": _mean(test_rmse),
            "rmse_std": _sample_std(test_rmse),
        },
    }


def _write_single_report(repo_root: Path, *, task: int, row: Mapping[str, Any]) -> dict[str, Any]:
    paths = _single_report_paths(repo_root, task)
    payload = {
        "schema_version": EVAL_RUNNER_SCHEMA_VERSION,
        "generated_at": _now_iso(),
        "task": int(task),
        "mode": "single",
        "eval_namespace": _repo_relative(repo_root, _eval_root(repo_root, task, 1)),
        "run": row,
    }
    atomic_write_json(paths["json"], payload)
    lines = [
        f"# Step3 Rating Eval 1-Seed: task{int(task)}",
        "",
        f"- source_run_id: {row['source_run_id']}",
        f"- seed: {row['seed']}",
        f"- valid MAE/RMSE: {row['valid']['mae']}/{row['valid']['rmse']}",
        f"- test MAE/RMSE: {row['test']['mae']}/{row['test']['rmse']}",
        f"- eval_handoff: {row['eval_handoff']}",
        "",
    ]
    paths["md"].write_text("\n".join(lines), encoding="utf-8")
    payload["json_path"] = _repo_relative(repo_root, paths["json"])
    payload["md_path"] = _repo_relative(repo_root, paths["md"])
    return payload


def _write_multi_report(repo_root: Path, *, task: int, source_run_id: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    paths = _multi_report_paths(repo_root, task)
    summary = _summarize(rows)
    payload = {
        "schema_version": EVAL_MEAN_STD_SCHEMA_VERSION,
        "generated_at": _now_iso(),
        "task": int(task),
        "mode": "multi",
        "source_run_id": str(source_run_id),
        "eval_namespace": _repo_relative(repo_root, _eval_root(repo_root, task, 5)),
        "seeds": list(DEFAULT_RATING_SEEDS),
        "metric_protocol": PAPER_TARGET_ONLY_EVAL,
        "target_only": True,
        "std_ddof": 1,
        "paper_comparable_mean_std": len(rows) == len(DEFAULT_RATING_SEEDS),
        **summary,
        "runs": list(rows),
    }
    paths["csv"].parent.mkdir(parents=True, exist_ok=True)
    with paths["csv"].open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "seed",
                "source_run_id",
                "eval_handoff",
                "valid_mae",
                "valid_rmse",
                "valid_sample_count",
                "test_mae",
                "test_rmse",
                "test_sample_count",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "seed": row["seed"],
                    "source_run_id": row["source_run_id"],
                    "eval_handoff": row["eval_handoff"],
                    "valid_mae": row["valid"]["mae"],
                    "valid_rmse": row["valid"]["rmse"],
                    "valid_sample_count": row["valid"]["sample_count"],
                    "test_mae": row["test"]["mae"],
                    "test_rmse": row["test"]["rmse"],
                    "test_sample_count": row["test"]["sample_count"],
                }
            )
    atomic_write_json(paths["json"], payload)
    _write_multi_markdown(paths["md"], payload)
    payload["runs_csv"] = _repo_relative(repo_root, paths["csv"])
    payload["mean_std_json"] = _repo_relative(repo_root, paths["json"])
    payload["report_md"] = _repo_relative(repo_root, paths["md"])
    return payload


def _write_multi_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    lines = [
        f"# Step3 Rating Eval task{payload['task']} 5-Seed Mean/Std",
        "",
        f"- source_run_id: {payload['source_run_id']}",
        f"- protocol: {payload['metric_protocol']}",
        f"- target_only: {payload['target_only']}",
        f"- std_ddof: {payload['std_ddof']}",
        "",
        "| metric | mean | std |",
        "| --- | --- | --- |",
        f"| valid MAE | {payload['valid']['mae_mean']} | {payload['valid']['mae_std']} |",
        f"| valid RMSE | {payload['valid']['rmse_mean']} | {payload['valid']['rmse_std']} |",
        f"| test MAE | {payload['test']['mae_mean']} | {payload['test']['mae_std']} |",
        f"| test RMSE | {payload['test']['rmse_mean']} | {payload['test']['rmse_std']} |",
        "",
        "| seed | valid MAE/RMSE | test MAE/RMSE | eval_handoff |",
        "| --- | --- | --- | --- |",
    ]
    for row in payload.get("runs") or []:
        lines.append(
            f"| {row['seed']} | {row['valid']['mae']}/{row['valid']['rmse']} | "
            f"{row['test']['mae']}/{row['test']['rmse']} | {row['eval_handoff']} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def aggregate_step3_rating_eval_five_seed(
    repo_root: str | Path,
    *,
    task: int,
    source_run_id: str,
) -> dict[str, Any]:
    """Aggregate an already-written eval/5 namespace.

    This is intentionally not exposed as a public CLI mode. It exists for tests
    and for the multi runner's internal finalization path.
    """
    root = Path(repo_root).expanduser().resolve()
    source = _validate_source_checkpoint(root, task=int(task), run_id=str(source_run_id))
    rows = [
        _write_seed_handoff(
            root,
            task=int(task),
            count=5,
            seed=seed,
            source_run_id=str(source_run_id),
            source=source,
        )
        for seed in DEFAULT_RATING_SEEDS
    ]
    return _write_multi_report(root, task=int(task), source_run_id=str(source_run_id), rows=rows)


def _dry_run_payload(repo_root: Path, *, task: int, mode: str, runs: Sequence[RatingSeedRun]) -> dict[str, Any]:
    count = _eval_count(mode)
    source_run_id = runs[0].run_id
    commands: list[dict[str, Any]] = []
    for item in runs:
        commands.append(
            {
                "seed": item.seed,
                "source_run_id": item.run_id,
                "paper_eval_valid": {
                    "internal": "odcr_core.runners._run_step3_eval",
                    "set": _eval_overrides(item.seed, "valid"),
                    "log_dir": _repo_relative(repo_root, _eval_split_log_dir(repo_root, task, count, item.seed, "valid")),
                },
                "paper_eval_test": {
                    "internal": "odcr_core.runners._run_step3_eval",
                    "set": _eval_overrides(item.seed, "test"),
                    "log_dir": _repo_relative(repo_root, _eval_split_log_dir(repo_root, task, count, item.seed, "test")),
                },
                "eval_handoff": _repo_relative(repo_root, _eval_seed_dir(repo_root, task, count, item.seed) / "eval_handoff.json"),
            }
        )
    paths = _single_report_paths(repo_root, task) if count == 1 else _multi_report_paths(repo_root, task)
    payload = {
        "schema_version": "odcr_step3_rating_eval_runner_plan/1",
        "dry_run": True,
        "task": int(task),
        "mode": mode,
        "source_run_id": source_run_id,
        "eval_seed_count": count,
        "eval_namespace": _repo_relative(repo_root, _eval_root(repo_root, task, count)),
        "overwrite_eval_namespace_on_run": True,
        "seeds": [item.seed for item in runs],
        "commands": commands,
        "report_paths": {key: _repo_relative(repo_root, value) for key, value in paths.items()},
        "direct_odcr_nohup": {
            "command": build_direct_nohup_command(
                repo_root,
                task=int(task),
                mode=mode,
                source_run_id=source_run_id,
                seed=runs[0].seed if count == 1 else None,
            ),
            "artifacts": {
                key: _repo_relative(repo_root, value)
                for key, value in _driver_paths(repo_root, int(task), count, source_run_id).items()
            },
        },
        "formal_training_executed": False,
    }
    return payload


def run_step3_rating_seed_runner(
    repo_root: str | Path,
    *,
    task: int,
    mode: str,
    config_path: str,
    seed: int | None = None,
    run_id: str | None = None,
    run_id_start: int | None = None,
    dry_run: bool = False,
    command_runner: Any | None = None,
    console_level: str = "summary",
) -> dict[str, Any]:
    _ = command_runner
    root = Path(repo_root).expanduser().resolve()
    mode_norm = str(mode or "").strip().lower()
    runs = build_rating_seed_plan(
        task=int(task),
        mode=mode_norm,
        seed=seed,
        run_id=run_id,
        run_id_start=run_id_start,
    )
    if dry_run:
        return _dry_run_payload(root, task=int(task), mode=mode_norm, runs=runs)

    count = _eval_count(mode_norm)
    source_run_id = runs[0].run_id
    driver = _driver_paths(root, int(task), count, source_run_id)
    source = _validate_source_checkpoint(root, task=int(task), run_id=source_run_id)
    gpu = _assert_gpu_ready(trace_log=driver["trace_log"])

    eval_root = _eval_root(root, int(task), count)
    if eval_root.exists():
        shutil.rmtree(eval_root)
    eval_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for item in runs:
        _run_paper_eval_split(
            repo_root=root,
            config_path=config_path,
            task=int(task),
            source_run_id=item.run_id,
            seed=item.seed,
            count=count,
            split="valid",
            console_level=console_level,
        )
        _run_paper_eval_split(
            repo_root=root,
            config_path=config_path,
            task=int(task),
            source_run_id=item.run_id,
            seed=item.seed,
            count=count,
            split="test",
            console_level=console_level,
        )
        rows.append(
            _write_seed_handoff(
                root,
                task=int(task),
                count=count,
                seed=item.seed,
                source_run_id=item.run_id,
                source=source,
            )
        )

    if count == 1:
        report = _write_single_report(root, task=int(task), row=rows[0])
    else:
        report = _write_multi_report(root, task=int(task), source_run_id=source_run_id, rows=rows)
    return {
        "schema_version": EVAL_RUNNER_SCHEMA_VERSION,
        "generated_at": _now_iso(),
        "task": int(task),
        "mode": mode_norm,
        "source": source,
        "gpu": gpu,
        "eval_namespace": _repo_relative(root, eval_root),
        "formal_training_executed": False,
        "runs": rows,
        "report": report,
    }
