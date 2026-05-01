"""产物路径：训练 CSV、模型、manifest 相关。"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from odcr_core import path_layout, run_naming
from odcr_core.index_contract import INDEX_CONTRACT_FILENAME, ODCR_ROUTING_TRAIN_CSV
from odcr_core.paths import (
    repo_root_from_code_dir,
    resolve_iteration_root_dir,
    resolve_metrics_dir,
    resolve_step3_dir,
    resolve_step5_dir,
)

ResolvedConfig = Any


def index_contract_path(cfg: ResolvedConfig) -> Path:
    """与 ``train_csv_path`` 解析后的真实 CSV 同目录的 ``index_contract.json``。"""
    return train_csv_path(cfg).resolve().parent / INDEX_CONTRACT_FILENAME


def train_csv_path(cfg: ResolvedConfig) -> Path:
    if cfg.train_csv:
        return Path(cfg.train_csv).expanduser().resolve()
    if cfg.command == "step4":
        return Path(cfg.checkpoint_dir) / ODCR_ROUTING_TRAIN_CSV
    if cfg.step5_run:
        rid4 = run_naming.step4_slug_from_step5_slug(cfg.step5_run)
        return (
            path_layout.get_train_step4_run_root(cfg.repo_root, cfg.task_id, cfg.iteration_id, rid4)
            / ODCR_ROUTING_TRAIN_CSV
        )
    if cfg.step4_run:
        rid4 = run_naming.parse_run_id(cfg.step4_run)
        return (
            path_layout.get_train_step4_run_root(cfg.repo_root, cfg.task_id, cfg.iteration_id, rid4)
            / ODCR_ROUTING_TRAIN_CSV
        )
    raise ValueError(
        "无法解析训练 CSV（正式名 odcr_routing_train.csv）：请指定 --train-csv，或 --step5-run"
        "（如 2_1_1 → train/step4/2_1/），或在 step4 命令下使用当前 checkpoint 目录。"
    )


def model_path_default(cfg: ResolvedConfig) -> Path:
    return path_layout.model_file_path(Path(cfg.checkpoint_dir))


def ensure_step5_csv_symlink(cfg: ResolvedConfig) -> None:
    """在 Step5 run 目录下创建指向 Step4 ``odcr_routing_train.csv`` 的同名软链，供 ``ODCR_STAGE_RUN_DIR`` 内读取。"""
    assert cfg.from_run is not None and cfg.step5_run is not None
    run_root = Path(cfg.checkpoint_dir)
    run_root.mkdir(parents=True, exist_ok=True)
    legacy = run_root / "factuals_counterfactuals.csv"
    if legacy.exists() or legacy.is_symlink():
        if legacy.is_symlink():
            legacy.unlink()
        else:
            raise FileExistsError(
                f"Step5 目录存在遗留实体文件 {legacy}；主线已退役 factuals_counterfactuals.csv，请删除后重试。"
            )
    dest = run_root / ODCR_ROUTING_TRAIN_CSV
    src = train_csv_path(cfg)
    if not src.is_file():
        raise FileNotFoundError(f"缺少 Step4 正式训练表 CSV: {src}")
    if dest.exists() or dest.is_symlink():
        if dest.is_symlink() and dest.resolve() == src.resolve():
            return
        raise FileExistsError(f"已存在且非预期软链或文件: {dest}")
    rel = os.path.relpath(src, dest.parent)
    os.symlink(rel, dest)


__all__ = [
    "index_contract_path",
    "train_csv_path",
    "model_path_default",
    "ensure_step5_csv_symlink",
    "repo_root_from_code_dir",
    "resolve_step3_dir",
    "resolve_step5_dir",
    "resolve_iteration_root_dir",
    "resolve_metrics_dir",
]
