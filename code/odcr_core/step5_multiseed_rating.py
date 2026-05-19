"""Plan and aggregate Step5A rating multi-seed runs."""
from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping, Sequence

from odcr_core.file_atomic import atomic_write_json


MULTISEED_MEAN_STD_SCHEMA_VERSION = "odcr_step5A_rating_multiseed_mean_std/1"
DEFAULT_STEP5A_RATING_SEEDS = (3407, 1337, 1234, 5678, 9012)
MULTISEED_MODES = ("reuse_seed3407", "strict_rerun_all")


class Step5MultiseedRatingError(RuntimeError):
    """Raised when Step5A multi-seed rating planning fails."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _repo_relative(root: Path, path: str | Path | None) -> str | None:
    if path is None:
        return None
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = (root / p).resolve()
    else:
        p = p.resolve()
    try:
        return p.relative_to(root).as_posix()
    except ValueError:
        return p.as_posix()


def parse_seed_list(raw: str | Sequence[int] | None) -> list[int]:
    if raw is None:
        return list(DEFAULT_STEP5A_RATING_SEEDS)
    if isinstance(raw, str):
        items = [item.strip() for item in raw.split(",") if item.strip()]
        if not items:
            raise Step5MultiseedRatingError("--seeds must not be empty")
        return [int(item) for item in items]
    return [int(item) for item in raw]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _completed_step5a_handoffs(root: Path, task: int) -> dict[int, dict[str, Any]]:
    parent = root / "runs" / "step5" / f"task{int(task)}"
    out: dict[int, dict[str, Any]] = {}
    if not parent.is_dir():
        return out
    for run_root in sorted(parent.iterdir()):
        if not run_root.is_dir() or not run_root.name.endswith("_step5A"):
            continue
        handoff = _load_json(run_root / "meta" / "eval_handoff.json")
        if str(handoff.get("status") or "").lower() not in {"ok", "completed", "accepted"}:
            continue
        summary = _load_json(run_root / "meta" / "run_summary.json")
        seed = int(summary.get("seed") or handoff.get("seed") or 3407)
        out[seed] = {
            "run_id": run_root.name,
            "checkpoint_hash": handoff.get("checkpoint_hash"),
            "valid_mae": (handoff.get("valid") or {}).get("mae"),
            "valid_rmse": (handoff.get("valid") or {}).get("rmse"),
            "test_mae": (handoff.get("test") or {}).get("mae"),
            "test_rmse": (handoff.get("test") or {}).get("rmse"),
            "eval_handoff_path": _repo_relative(root, run_root / "meta" / "eval_handoff.json"),
            "rating_quality_diagnostic_path": _repo_relative(root, run_root / "eval" / "rating_quality_diagnostic.json")
            if (run_root / "eval" / "rating_quality_diagnostic.json").is_file()
            else None,
            "status": "completed",
        }
    return out


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return float(values[0]), None
    return float(mean(values)), float(stdev(values))


def build_step5A_multiseed_rating_plan(
    *,
    repo_root: str | Path,
    task: int,
    head: str = "step5A",
    from_step4_run: str = "1",
    seeds: Sequence[int] | None = None,
    mode: str = "reuse_seed3407",
    dry_run: bool = True,
) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve()
    if head != "step5A":
        raise Step5MultiseedRatingError("Step5A multiseed rating supports --head step5A only")
    if mode not in MULTISEED_MODES:
        raise Step5MultiseedRatingError(f"--mode must be one of {', '.join(MULTISEED_MODES)}")
    seed_list = [int(seed) for seed in (seeds or DEFAULT_STEP5A_RATING_SEEDS)]
    completed = _completed_step5a_handoffs(root, int(task))
    rows: list[dict[str, Any]] = []
    launch_plan: list[dict[str, Any]] = []
    for seed in seed_list:
        completed_row = completed.get(seed) if mode == "reuse_seed3407" else None
        if completed_row:
            row = {"seed": seed, **completed_row}
            if seed == 3407 and mode == "reuse_seed3407":
                launch_plan.append(
                    {
                        "seed": seed,
                        "action": "reuse_existing",
                        "run_id": completed_row["run_id"],
                        "finalize_command": (
                            f"./odcr step5 --task {task} --head step5A --from-step5-run "
                            f"{completed_row['run_id']} --finalize-rating-handoff"
                        ),
                    }
                )
        else:
            row = {
                "seed": seed,
                "run_id": None,
                "checkpoint_hash": None,
                "valid_mae": None,
                "valid_rmse": None,
                "test_mae": None,
                "test_rmse": None,
                "eval_handoff_path": None,
                "rating_quality_diagnostic_path": None,
                "status": "planned_not_started",
            }
            launch_plan.append(
                {
                    "seed": seed,
                    "action": "train_eval_handoff",
                    "train_command": (
                        f"./odcr --set project.seed={seed} step5 --task {task} "
                        f"--head step5A --from-step4-run {from_step4_run} --run-id auto"
                    ),
                    "rating_eval_command_template": (
                        f"./odcr --set project.seed={seed} step5 --task {task} --head step5A "
                        "--from-step5-run <RUN_ID> --eval-only --checkpoint "
                        f"runs/step5/task{task}/<RUN_ID>/model/best.pth"
                    ),
                    "finalize_only_command_template": (
                        f"./odcr step5 --task {task} --head step5A --from-step5-run "
                        "<RUN_ID> --finalize-rating-handoff"
                    ),
                    "expected_run_dir": f"runs/step5/task{task}/<auto>_step5A",
                }
            )
        rows.append(row)

    completed_rows = [row for row in rows if row["status"] == "completed"]
    failed_rows = [row for row in rows if str(row["status"]).startswith("failed")]

    def metric_values(key: str) -> list[float]:
        out = []
        for row in completed_rows:
            value = row.get(key)
            if value is not None and not (isinstance(value, float) and math.isnan(value)):
                out.append(float(value))
        return out

    valid_mae_mean, valid_mae_std = _mean_std(metric_values("valid_mae"))
    valid_rmse_mean, valid_rmse_std = _mean_std(metric_values("valid_rmse"))
    test_mae_mean, test_mae_std = _mean_std(metric_values("test_mae"))
    test_rmse_mean, test_rmse_std = _mean_std(metric_values("test_rmse"))
    complete_all = len(completed_rows) == len(seed_list)
    mean_std_payload = {
        "schema_version": MULTISEED_MEAN_STD_SCHEMA_VERSION,
        "task": int(task),
        "head": "step5A",
        "seeds": seed_list,
        "mode": mode,
        "metric_protocol": "code1_compatible_rating_v1",
        "target_only": True,
        "valid_mae_mean": valid_mae_mean,
        "valid_mae_std": valid_mae_std,
        "valid_rmse_mean": valid_rmse_mean,
        "valid_rmse_std": valid_rmse_std,
        "test_mae_mean": test_mae_mean,
        "test_mae_std": test_mae_std,
        "test_rmse_mean": test_rmse_mean,
        "test_rmse_std": test_rmse_std,
        "std_ddof": 1,
        "paper_comparable_mean_std": bool(complete_all),
        "completed_seed_count": len(completed_rows),
        "failed_seed_count": len(failed_rows),
        "failed_seeds": [int(row["seed"]) for row in failed_rows],
        "multiseed_actual_started": False,
        "notes": [
            "This command plans and aggregates Step5A rating seeds; it does not start long GPU training.",
            "paper_comparable_mean_std becomes true only after all seeds have completed eval_handoff.",
        ],
        "created_at": _utc_now(),
    }
    out_dir = root / "runs" / "step5" / f"task{int(task)}" / "multiseed"
    csv_path = out_dir / f"step5A_rating_task{int(task)}_runs.csv"
    mean_std_path = out_dir / f"step5A_rating_task{int(task)}_mean_std.json"
    report_path = out_dir / f"step5A_rating_task{int(task)}_report.md"
    result = {
        "status": "dry_run" if dry_run else "ok",
        "task": int(task),
        "head": "step5A",
        "seeds": seed_list,
        "mode": mode,
        "rows": rows,
        "mean_std": mean_std_payload,
        "launch_plan": launch_plan,
        "artifacts": {
            "runs_csv": _repo_relative(root, csv_path),
            "mean_std": _repo_relative(root, mean_std_path),
            "report": _repo_relative(root, report_path),
        },
        "multiseed_actual_started": False,
    }
    if dry_run:
        return result
    out_dir.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "seed",
                "run_id",
                "checkpoint_hash",
                "valid_mae",
                "valid_rmse",
                "test_mae",
                "test_rmse",
                "eval_handoff_path",
                "rating_quality_diagnostic_path",
                "status",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    atomic_write_json(mean_std_path, mean_std_payload)
    report_lines = [
        "# Step5A Rating Multi-Seed Plan",
        "",
        f"- task: {task}",
        f"- mode: {mode}",
        f"- seeds: {seed_list}",
        f"- completed_seed_count: {len(completed_rows)}",
        f"- paper_comparable_mean_std: {complete_all}",
        "",
        "## Launch Plan",
    ]
    for item in launch_plan:
        report_lines.append(f"- seed {item['seed']}: {item['action']}")
        if item.get("train_command"):
            report_lines.append(f"  train: `{item['train_command']}`")
            report_lines.append(f"  eval: `{item['rating_eval_command_template']}`")
            report_lines.append(f"  finalize: `{item['finalize_only_command_template']}`")
        elif item.get("finalize_command"):
            report_lines.append(f"  finalize: `{item['finalize_command']}`")
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return result
