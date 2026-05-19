"""Step5A frozen-teacher parity helpers and Step3 tokenizer-cache evidence lookup."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import torch


STEP5A_TEACHER_PARITY_SCHEMA_VERSION = "odcr_step5A_teacher_parity/1"
TOKENIZED_EVIDENCE_COLUMNS = (
    "content_evidence_ids",
    "style_evidence_ids",
    "domain_style_anchor_ids",
    "local_style_hint_ids",
    "polarity_ids",
)


def tensor_distribution(values: torch.Tensor) -> dict[str, float]:
    flat = values.detach().float().view(-1).cpu()
    if flat.numel() == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(flat.mean().item()),
        "std": float(flat.std(unbiased=False).item()),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
    }


def pearson_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    x = a.detach().float().view(-1).cpu()
    y = b.detach().float().view(-1).cpu()
    if x.numel() < 2:
        return 1.0
    x = x - x.mean()
    y = y - y.mean()
    den = torch.sqrt((x * x).sum() * (y * y).sum()).clamp_min(1e-12)
    return float(((x * y).sum() / den).item())


def _rankdata(x: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(x)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(x.numel(), dtype=torch.float32)
    return ranks


def spearman_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    x = a.detach().float().view(-1).cpu()
    y = b.detach().float().view(-1).cpu()
    if x.numel() < 2:
        return 1.0
    return pearson_corr(_rankdata(x), _rankdata(y))


def build_step5a_teacher_parity_report(
    *,
    teacher_pred: torch.Tensor,
    step5a_initial_pred: torch.Tensor,
    gt_rating: torch.Tensor,
    sample_ids: Sequence[int],
    split: str,
    parity_tolerance: float = 1e-6,
) -> dict[str, Any]:
    teacher = teacher_pred.detach().float().view(-1).cpu()
    initial = step5a_initial_pred.detach().float().view(-1).cpu()
    gt = gt_rating.detach().float().view(-1).cpu()
    if not (teacher.numel() == initial.numel() == gt.numel()):
        raise ValueError("teacher, Step5A initial, and gt tensors must have the same sample count")
    err_t = teacher - gt
    err_i = initial - gt
    delta = initial - teacher
    teacher_rmse = float(torch.sqrt((err_t * err_t).mean()).item()) if err_t.numel() else 0.0
    initial_rmse = float(torch.sqrt((err_i * err_i).mean()).item()) if err_i.numel() else 0.0
    delta_rmse = float(torch.sqrt((delta * delta).mean()).item()) if delta.numel() else 0.0
    max_abs_delta = float(delta.abs().max().item()) if delta.numel() else 0.0
    report = {
        "schema_version": STEP5A_TEACHER_PARITY_SCHEMA_VERSION,
        "sample_count": int(teacher.numel()),
        "split": str(split),
        "sample_ids": [int(x) for x in list(sample_ids)[:32]],
        "teacher_pred_distribution": tensor_distribution(teacher),
        "step5A_initial_pred_distribution": tensor_distribution(initial),
        "gt_distribution": tensor_distribution(gt),
        "teacher_mae": float(err_t.abs().mean().item()) if err_t.numel() else 0.0,
        "teacher_rmse": teacher_rmse,
        "step5A_initial_mae": float(err_i.abs().mean().item()) if err_i.numel() else 0.0,
        "step5A_initial_rmse": initial_rmse,
        "teacher_vs_step5A_max_abs_delta": max_abs_delta,
        "teacher_vs_step5A_mae_delta": float(delta.abs().mean().item()) if delta.numel() else 0.0,
        "teacher_vs_step5A_rmse_delta": delta_rmse,
        "pearson": pearson_corr(teacher, initial),
        "spearman_optional": spearman_corr(teacher, initial),
        "out_of_range_rate": float(((initial < 0.0) | (initial > 5.0)).float().mean().item()) if initial.numel() else 0.0,
    }
    report["parity_pass"] = bool(
        report["sample_count"] > 0
        and math.isfinite(max_abs_delta)
        and max_abs_delta <= float(parity_tolerance)
        and report["teacher_vs_step5A_rmse_delta"] <= float(parity_tolerance)
    )
    return report


def _arrow_tables(cache_dir: str | Path, split: str) -> Iterable[Any]:
    import pyarrow.ipc as ipc

    split_dir = Path(cache_dir) / str(split)
    for path in sorted(split_dir.glob("*.arrow")):
        with path.open("rb") as handle:
            yield ipc.RecordBatchStreamReader(handle).read_all()


def collect_step3_target_tokenized_rows(
    *,
    cache_dir: str | Path,
    split: str,
    target_local_ids: Sequence[int],
) -> dict[int, dict[str, Any]]:
    """Collect tokenized evidence rows without constructing a tokenizer."""

    wanted = {int(x) for x in target_local_ids}
    if not wanted:
        return {}
    out: dict[int, dict[str, Any]] = {}
    target_seen = 0
    for table in _arrow_tables(cache_dir, split):
        domains = table.column("domain").to_pylist()
        ratings = table.column("rating").to_pylist()
        users = table.column("user_idx").to_pylist()
        items = table.column("item_idx").to_pylist()
        values = {col: table.column(col).to_pylist() for col in TOKENIZED_EVIDENCE_COLUMNS}
        anchors = {
            "content_anchor_score": table.column("content_anchor_score").to_pylist(),
            "style_anchor_score": table.column("style_anchor_score").to_pylist(),
            "evidence_quality_prior": table.column("evidence_quality_prior").to_pylist(),
        }
        for row_idx, domain in enumerate(domains):
            if str(domain) != "target":
                continue
            local_id = target_seen
            target_seen += 1
            if local_id not in wanted:
                continue
            row = {
                "user_idx_global": int(users[row_idx]),
                "item_idx_global": int(items[row_idx]),
                "domain_idx": 1,
                "rating": float(ratings[row_idx]),
                "step3_target_local_id": int(local_id),
            }
            for col in TOKENIZED_EVIDENCE_COLUMNS:
                raw = values[col][row_idx]
                row[col] = [int(raw)] if col == "polarity_ids" and not isinstance(raw, list) else [int(x) for x in raw]
            for col, raw_values in anchors.items():
                row[col] = float(raw_values[row_idx])
            out[local_id] = row
            if len(out) == len(wanted):
                return out
    missing = sorted(wanted - set(out))
    if missing:
        raise KeyError(f"Step3 tokenizer cache split={split!r} missing target-local ids: {missing[:20]}")
    return out


def attach_step3_tokenized_evidence_columns(
    df: pd.DataFrame,
    *,
    cache_dir: str | Path,
    split: str,
    sample_id_column: str = "sample_id",
) -> pd.DataFrame:
    if df.empty:
        return df
    ids = [int(x) for x in df[sample_id_column].astype("int64").tolist()]
    lookup = collect_step3_target_tokenized_rows(cache_dir=cache_dir, split=split, target_local_ids=ids)
    out = df.copy()
    out["step3_token_cache_split"] = str(split)
    for col in TOKENIZED_EVIDENCE_COLUMNS:
        out[col] = [lookup[int(sample_id)][col] for sample_id in ids]
    for col in ("content_anchor_score", "style_anchor_score", "evidence_quality_prior"):
        out[col] = [lookup[int(sample_id)][col] for sample_id in ids]
    return out


__all__ = [
    "STEP5A_TEACHER_PARITY_SCHEMA_VERSION",
    "TOKENIZED_EVIDENCE_COLUMNS",
    "attach_step3_tokenized_evidence_columns",
    "build_step5a_teacher_parity_report",
    "collect_step3_target_tokenized_rows",
    "pearson_corr",
    "spearman_corr",
    "tensor_distribution",
]
