"""
正式基线注册：在 ``runs/task{T}/vN/baselines/`` 登记 baseline_id、源 eval 目录与 metrics 快照，
供后续 summary / compare 统一读取；**不修改**源 ``eval/<run>/`` 内文件。
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from odcr_core import path_layout, run_naming
from odcr_core.phase1_eval_summary import (
    PHASE1_PAPER_COLUMN_KEYS,
    PHASE1_REPO_COLUMN_KEYS,
    row_from_metrics_root,
)

DEFAULT_BASELINE_INDEX_NAME = "default_baseline_index.json"
REGISTRATION_FILENAME = "baseline_registration.json"
METRICS_SNAPSHOT_NAME = "metrics_snapshot.json"

_RE_BASELINE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _task_iter_key(task_id: int, iteration_id: str) -> str:
    it = run_naming.normalize_iteration_id(iteration_id)
    return f"{int(task_id)}|{it}"


def _default_index_path(repo_root: Path, task_id: int, iteration_id: str) -> Path:
    return path_layout.get_baselines_root(repo_root, task_id, iteration_id) / DEFAULT_BASELINE_INDEX_NAME


def default_baseline_index_path(repo_root: Path, task_id: int, iteration_id: str) -> Path:
    """``default_baseline_index.json`` 的绝对路径（文件未必已存在）。"""
    it = run_naming.normalize_iteration_id(iteration_id)
    return _default_index_path(repo_root, task_id, it).resolve()


def _baseline_home(repo_root: Path, task_id: int, iteration_id: str, baseline_id: str) -> Path:
    return path_layout.get_baselines_root(repo_root, task_id, iteration_id) / baseline_id


def _validate_baseline_id(baseline_id: str) -> str:
    s = (baseline_id or "").strip()
    if not s or len(s) > 128 or not _RE_BASELINE_ID.match(s):
        raise ValueError(
            "baseline_id 须为非空 slug：字母数字开头，可含 ._-；长度≤128。"
            f" 当前: {baseline_id!r}"
        )
    return s


def _resolve_source_eval_dir(repo_root: Path, source_eval_dir: str | Path) -> Path:
    raw = Path(source_eval_dir)
    p = (repo_root / raw).resolve() if not raw.is_absolute() else raw.expanduser().resolve()
    if not p.is_dir():
        raise FileNotFoundError(f"source_eval_dir 不存在或不是目录: {p}")
    return p


def register_baseline(
    repo_root: Path,
    task_id: int,
    iteration_id: str,
    baseline_id: str,
    source_eval_dir: str | Path,
    *,
    note: str | None = None,
    purpose: str | None = None,
    set_default: bool = False,
    force: bool = False,
) -> Tuple[Path, Path]:
    """
    登记基线：校验源目录与 ``eval_metrics.json``，写入 ``baselines/<baseline_id>/`` 下登记文件与快照。

    返回 ``(registration_json_path, metrics_snapshot_path)``。
    """
    repo_root = repo_root.resolve()
    bid = _validate_baseline_id(baseline_id)
    it = run_naming.normalize_iteration_id(iteration_id)
    src = _resolve_source_eval_dir(repo_root, source_eval_dir)
    metrics_p = path_layout.eval_metrics_path(src, rerank=False)
    if not metrics_p.is_file():
        metrics_p = src / "metrics.json"
    if not metrics_p.is_file():
        raise FileNotFoundError(f"源 eval 目录缺少 eval_metrics.json: {metrics_p}")

    base_root = path_layout.get_baselines_root(repo_root, task_id, it)
    base_root.mkdir(parents=True, exist_ok=True)
    home = _baseline_home(repo_root, task_id, it, bid)
    reg_p = home / REGISTRATION_FILENAME
    snap_p = home / METRICS_SNAPSHOT_NAME

    if home.exists() and not force:
        raise FileExistsError(
            f"基线目录已存在: {home}（如需覆盖快照与登记，请加 force=True / CLI --force）"
        )
    home.mkdir(parents=True, exist_ok=True)

    with open(metrics_p, "r", encoding="utf-8") as f:
        metrics_doc: Dict[str, Any] = json.load(f)

    try:
        rel_src = src.relative_to(repo_root)
    except ValueError:
        rel_src = src
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    reg_payload = {
        "schema": "odcr_baseline_registration_v1",
        "baseline_id": bid,
        "task_id": int(task_id),
        "iteration_id": it,
        "source_eval_dir": str(rel_src).replace("\\", "/"),
        "source_metrics_path": str((rel_src / metrics_p.name)).replace("\\", "/"),
        "metrics_snapshot_filename": METRICS_SNAPSHOT_NAME,
        "registered_at_utc": now,
        "note": (note or "").strip() or None,
        "purpose": (purpose or "").strip() or None,
    }
    with open(reg_p, "w", encoding="utf-8") as wf:
        json.dump(reg_payload, wf, ensure_ascii=False, indent=2)
        wf.write("\n")
    with open(snap_p, "w", encoding="utf-8") as wf:
        json.dump(metrics_doc, wf, ensure_ascii=False, indent=2)
        wf.write("\n")

    if set_default:
        set_default_baseline(repo_root, task_id, it, bid)

    return reg_p, snap_p


def set_default_baseline(repo_root: Path, task_id: int, iteration_id: str, baseline_id: str) -> Path:
    """将 ``(task_id, iteration_id)`` 的默认基线设为 ``baseline_id``（须已登记）。"""
    bid = _validate_baseline_id(baseline_id)
    it = run_naming.normalize_iteration_id(iteration_id)
    home = _baseline_home(repo_root, task_id, it, bid)
    if not (home / REGISTRATION_FILENAME).is_file():
        raise FileNotFoundError(f"未找到已登记基线: {home / REGISTRATION_FILENAME}")

    idx_p = _default_index_path(repo_root, task_id, it)
    idx_p.parent.mkdir(parents=True, exist_ok=True)
    data: Dict[str, Any] = {"schema": "odcr_default_baseline_index_v1", "by_key": {}}
    if idx_p.is_file():
        with open(idx_p, "r", encoding="utf-8") as rf:
            raw = json.load(rf)
        if isinstance(raw, dict) and isinstance(raw.get("by_key"), dict):
            data["by_key"] = dict(raw["by_key"])

    data["by_key"][_task_iter_key(task_id, it)] = bid
    data["updated_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(idx_p, "w", encoding="utf-8") as wf:
        json.dump(data, wf, ensure_ascii=False, indent=2)
        wf.write("\n")
    return idx_p


def get_default_baseline_id(repo_root: Path, task_id: int, iteration_id: str) -> Optional[str]:
    it = run_naming.normalize_iteration_id(iteration_id)
    idx_p = _default_index_path(repo_root, task_id, it)
    if not idx_p.is_file():
        return None
    with open(idx_p, "r", encoding="utf-8") as rf:
        raw = json.load(rf)
    if not isinstance(raw, dict):
        return None
    bk = raw.get("by_key")
    if not isinstance(bk, dict):
        return None
    v = bk.get(_task_iter_key(task_id, it))
    return str(v).strip() if v is not None else None


def load_baseline_registration(
    repo_root: Path,
    task_id: int,
    iteration_id: str,
    baseline_id: str,
) -> Dict[str, Any]:
    bid = _validate_baseline_id(baseline_id)
    it = run_naming.normalize_iteration_id(iteration_id)
    reg_p = _baseline_home(repo_root, task_id, it, bid) / REGISTRATION_FILENAME
    if not reg_p.is_file():
        raise FileNotFoundError(f"未找到基线登记: {reg_p}")
    with open(reg_p, "r", encoding="utf-8") as rf:
        return json.load(rf)


def get_baseline_metrics_snapshot_path(
    repo_root: Path,
    task_id: int,
    iteration_id: str,
    baseline_id: str,
) -> Path:
    it = run_naming.normalize_iteration_id(iteration_id)
    bid = _validate_baseline_id(baseline_id)
    snap = _baseline_home(repo_root, task_id, it, bid) / METRICS_SNAPSHOT_NAME
    if not snap.is_file():
        raise FileNotFoundError(f"未找到 metrics 快照: {snap}")
    return snap.resolve()


def load_baseline_metrics_document(
    repo_root: Path,
    task_id: int,
    iteration_id: str,
    baseline_id: str | None = None,
) -> Dict[str, Any]:
    """加载基线 **完整** metrics 文档（快照 JSON 根对象）。"""
    it = run_naming.normalize_iteration_id(iteration_id)
    bid = baseline_id or get_default_baseline_id(repo_root, task_id, it)
    if not bid:
        raise FileNotFoundError(
            f"未配置默认基线: {_default_index_path(repo_root, task_id, it)} "
            f"（请 register-baseline --set-default 或传入 baseline_id）"
        )
    snap = get_baseline_metrics_snapshot_path(repo_root, task_id, it, bid)
    with open(snap, "r", encoding="utf-8") as rf:
        return json.load(rf)


def load_baseline_metrics(
    repo_root: Path,
    task_id: int,
    iteration_id: str,
    baseline_id: str | None = None,
) -> Dict[str, Any]:
    """
    加载基线对应的 phase1 风格 **扁平行**（与 ``row_from_metrics_root`` / eval-summary 行字段对齐）。
    """
    root = load_baseline_metrics_document(repo_root, task_id, iteration_id, baseline_id=baseline_id)
    it = run_naming.normalize_iteration_id(iteration_id)
    bid = baseline_id or get_default_baseline_id(repo_root, task_id, it) or ""
    snap_path = get_baseline_metrics_snapshot_path(repo_root, task_id, it, bid)
    return row_from_metrics_root(root, metrics_path=str(snap_path))


def _is_finite_float(x: Any) -> bool:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    return v == v and abs(v) != float("inf")


def delta_metric_keys() -> List[str]:
    """与 phase1 表对齐、用于数值对比的键（repo + paper 列）。"""
    return list(PHASE1_REPO_COLUMN_KEYS) + list(PHASE1_PAPER_COLUMN_KEYS)


def compute_delta_vs_baseline(
    row_metrics: Mapping[str, Any],
    baseline_metrics: Mapping[str, Any],
    *,
    keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    对 ``row_metrics`` 与 ``baseline_metrics``（均为 phase1 扁平行或兼容 dict）做逐键数值差分。

    返回结构::
      { "per_metric": { "<key>": { "row", "baseline", "delta", "rel_delta" } }, "keys_compared": [...] }
    ``rel_delta`` 在 baseline 非零且有限时为 ``delta / baseline``，否则为 ``null``。
    """
    use_keys = list(keys) if keys is not None else delta_metric_keys()
    per: Dict[str, Any] = {}
    compared: List[str] = []
    eps = 1e-12
    for k in use_keys:
        if k not in row_metrics or k not in baseline_metrics:
            continue
        rv, bv = row_metrics[k], baseline_metrics[k]
        if not _is_finite_float(rv) or not _is_finite_float(bv):
            continue
        fr, fb = float(rv), float(bv)
        d = fr - fb
        rel = (d / fb) if abs(fb) > eps else None
        per[k] = {"row": fr, "baseline": fb, "delta": d, "rel_delta": rel}
        compared.append(k)
    return {"per_metric": per, "keys_compared": compared}


__all__ = [
    "DEFAULT_BASELINE_INDEX_NAME",
    "METRICS_SNAPSHOT_NAME",
    "REGISTRATION_FILENAME",
    "compute_delta_vs_baseline",
    "default_baseline_index_path",
    "delta_metric_keys",
    "get_baseline_metrics_snapshot_path",
    "get_default_baseline_id",
    "load_baseline_metrics",
    "load_baseline_metrics_document",
    "load_baseline_registration",
    "register_baseline",
    "set_default_baseline",
]
