"""路径解析：新版统一走 runs/task*/vN/…（由 path_layout + run_naming 实现）。"""
from __future__ import annotations

from pathlib import Path

from odcr_core import path_layout, run_naming
from odcr_core.index_contract import ODCR_ROUTING_TRAIN_CSV


def repo_root_from_code_dir(code_dir: Path) -> Path:
    return code_dir.resolve().parent


def resolve_step3_dir(root: Path, task: int, step3_run: str, iteration_id: str = "v1") -> Path:
    rid = run_naming.parse_run_id(step3_run)
    it = run_naming.normalize_iteration_id(iteration_id)
    return path_layout.get_train_step3_run_root(root, task, it, rid)


def resolve_step5_dir(
    root: Path, task: int, step3_run: str, step5_run: str, iteration_id: str = "v1"
) -> Path:
    s3 = run_naming.parse_run_id(step3_run)
    s5 = run_naming.parse_run_id(step5_run)
    it = run_naming.normalize_iteration_id(iteration_id)
    _ = s3  # step5 root 仅由 task/iter/step5_run 定位
    return path_layout.get_train_step5_run_root(root, task, it, s5)


def resolve_train_csv(
    root: Path,
    task: int,
    step3_run: str,
    explicit: str | None,
    iteration_id: str = "v1",
    *,
    step4_run: str | None = None,
    step5_run: str | None = None,
) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    it = run_naming.normalize_iteration_id(iteration_id)
    if step5_run:
        rid4 = run_naming.step4_slug_from_step5_slug(step5_run)
        return path_layout.get_train_step4_run_root(root, task, it, rid4) / ODCR_ROUTING_TRAIN_CSV
    if step4_run:
        rid4 = run_naming.parse_run_id(step4_run)
        return path_layout.get_train_step4_run_root(root, task, it, rid4) / ODCR_ROUTING_TRAIN_CSV
    raise ValueError(
        "resolve_train_csv 需要 --train-csv，或 step5_run（由目录名反推 step4），或 step4_run；"
        "不再从 train/step3 回退。"
    )


def resolve_model_path(
    root: Path,
    task: int,
    step3_run: str | None,
    step5_run: str | None,
    explicit: str | None,
    iteration_id: str = "v1",
) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    if step3_run and step5_run:
        p = resolve_step5_dir(root, task, step3_run, step5_run, iteration_id)
        return path_layout.model_file_path(p)
    if step3_run:
        p = resolve_step3_dir(root, task, step3_run, iteration_id)
        return path_layout.model_file_path(p)
    raise ValueError("model path cannot be resolved: need --model-path or step3/step5 run ids")


def resolve_iteration_root_dir(checkpoint_dir: Path) -> Path:
    """由某次 stage run 根目录解析 ``runs/task{T}/vN/``（迭代根）。

    评测指标文件在 ``<iteration>/eval/<run>/eval_metrics.json``，**不在**本目录下。
    """
    p = checkpoint_dir.resolve()
    parts = p.parts
    if "runs" in parts:
        ix = parts.index("runs")
        if ix + 3 <= len(parts):
            return Path(*parts[: ix + 3])
    return p


def resolve_metrics_dir(checkpoint_dir: Path) -> Path:
    """已弃用别名：请用 :func:`resolve_iteration_root_dir`。"""
    return resolve_iteration_root_dir(checkpoint_dir)
