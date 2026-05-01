"""Phase 2：扫描 runs/.../rerank/*/rerank_summary.json，生成 phase2_rerank_summary（含与同 generation-semantic baseline 的 delta）。

**Phase 2** 表示 rerank/matrix 汇总语义，非旧 pipeline 阶段编号；产物文件名与此一致。
"""
from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from odcr_core import path_layout
from odcr_core.phase1_eval_summary import _load_run_row, apply_phase1_style_scoring


def _collect_eval_metrics_paths(iteration_dir: Path) -> list[Path]:
    ev = iteration_dir / "eval"
    if not ev.is_dir():
        return []
    out: list[Path] = []
    for run in sorted(ev.iterdir()):
        if not run.is_dir():
            continue
        mp = path_layout.eval_metrics_path(run, rerank=False)
        if not mp.is_file():
            mp = run / "metrics.json"
        if mp.is_file():
            out.append(mp)
    return out


def _collect_rerank_metrics_paths(iteration_dir: Path) -> list[Path]:
    rr = iteration_dir / "rerank"
    if not rr.is_dir():
        return []
    out: list[Path] = []
    for run in sorted(rr.iterdir()):
        if not run.is_dir():
            continue
        mp = path_layout.eval_metrics_path(run, rerank=True)
        if not mp.is_file():
            mp = run / "metrics.json"
        if mp.is_file():
            out.append(mp)
    return out


def _f(x: Any, default: float = float("nan")) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _load_phase2_row(metrics_path: Path) -> Optional[Dict[str, Any]]:
    with open(metrics_path, "r", encoding="utf-8") as f:
        root = json.load(f)
    if not root.get("rerank_enabled"):
        return None
    base = _load_run_row(metrics_path)
    rs = root.get("rerank_summary") or {}
    rw = root.get("rerank_weights") or {}
    base.update(
        {
            "rerank_enabled": True,
            "rerank_method": root.get("rerank_method", ""),
            "num_return_sequences": int(root.get("num_return_sequences") or 0),
            "rerank_top_k": int(root.get("rerank_top_k") or 0),
            "rerank_weight_logprob": _f(rw.get("logprob")),
            "rerank_weight_length": _f(rw.get("length")),
            "rerank_weight_repeat": _f(rw.get("repeat")),
            "rerank_weight_dirty": _f(rw.get("dirty")),
            "rerank_target_len_ratio": _f(root.get("rerank_target_len_ratio")),
            "mean_selected_avg_logprob": _f(rs.get("mean_selected_avg_logprob")),
            "mean_selected_repeat_penalty": _f(rs.get("mean_selected_repeat_penalty")),
            "mean_selected_dirty_penalty": _f(rs.get("mean_selected_dirty_penalty")),
            "mean_selected_length_deviation_penalty": _f(rs.get("mean_selected_length_deviation_penalty")),
            "selected_not_best_logprob_rate": _f(rs.get("selected_not_best_logprob_rate")),
            "mean_candidate_rouge_proxy": _f(rs.get("mean_candidate_rouge_proxy")),
            "mean_candidate_count": _f(rs.get("avg_candidate_count")),
            "mean_selected_malformed_tail_penalty": _f(rs.get("mean_selected_malformed_tail_penalty")),
            "mean_selected_malformed_token_penalty": _f(rs.get("mean_selected_malformed_token_penalty")),
            "selected_malformed_tail_hit_rate": _f(rs.get("selected_malformed_tail_hit_rate")),
            "selected_malformed_token_hit_rate": _f(rs.get("selected_malformed_token_hit_rate")),
            "export_examples_mode": str(rs.get("export_examples_mode") or root.get("export_examples_mode") or ""),
            "completion_pass_rate": _f(rs.get("completion_pass_rate")),
            "well_formed_pass_rate": _f(rs.get("well_formed_pass_rate")),
            "source_coverage_mean": _f(rs.get("source_coverage_mean")),
            "entity_drift_hit_rate": _f(rs.get("entity_drift_hit_rate")),
            "generic_template_hit_rate": _f(rs.get("generic_template_hit_rate")),
            "hard_filter_drop_rate": _f(rs.get("hard_filter_drop_rate")),
        }
    )
    return base


def _baseline_by_generation_semantic_iteration(iteration_dir: Path) -> Dict[str, Dict[str, Any]]:
    """同一 generation_semantic_fingerprint 下，从 eval/ 中取非 rerank 且 mtime 最新的 run 作为 baseline。"""
    by_fp: Dict[str, List[tuple[float, Dict[str, Any]]]] = {}
    for mp in _collect_eval_metrics_paths(iteration_dir):
        try:
            with open(mp, "r", encoding="utf-8") as f:
                rootj = json.load(f)
        except OSError:
            continue
        if rootj.get("rerank_enabled"):
            continue
        fp = str(rootj.get("generation_semantic_fingerprint") or "")
        if not fp:
            continue
        row = _load_run_row(mp)
        mtime = os.path.getmtime(mp)
        by_fp.setdefault(fp, []).append((mtime, row))
    out: Dict[str, Dict[str, Any]] = {}
    for fp, lst in by_fp.items():
        lst.sort(key=lambda x: x[0], reverse=True)
        out[fp] = lst[0][1]
    return out


def _delta(cur: float, base: float) -> Optional[float]:
    if cur != cur or base != base:
        return None
    return round(float(cur) - float(base), 6)


def _target_tradeoff_flag(row: Dict[str, Any]) -> str:
    dr = float(row.get("delta_rouge_l_vs_same_decode") or 0.0)
    dm = float(row.get("delta_meteor_vs_same_decode") or 0.0)
    dd1 = float(row.get("delta_dist_1_vs_same_decode") or 0.0)
    dd2 = float(row.get("delta_dist_2_vs_same_decode") or 0.0)
    improved_align = (dr > 0.002) or (dm > 0.002)
    dropped_dist = (dd1 < -0.003) or (dd2 < -0.003)
    return "warn_main_target_tradeoff" if (improved_align and dropped_dist) else "ok"


def generate_phase2_rerank_summary(
    iteration_dir: str,
    *,
    out_dir: str,
) -> Union[List[Dict[str, Any]], Any]:
    root_it = Path(iteration_dir).expanduser().resolve()
    if not root_it.is_dir():
        raise FileNotFoundError(f"迭代目录不存在: {root_it}")
    out = Path(out_dir).expanduser().resolve()

    paths = sorted(_collect_rerank_metrics_paths(root_it))
    rows: List[Dict[str, Any]] = []
    for p in paths:
        r = _load_phase2_row(p)
        if r is not None:
            rows.append(r)
    if not rows:
        raise FileNotFoundError(
            f"未找到 rerank_enabled 的 rerank_summary.json: {root_it}/rerank/*/（请先跑 eval-rerank）"
        )

    baselines = _baseline_by_generation_semantic_iteration(root_it)
    for r in rows:
        fp = str(r.get("generation_semantic_fingerprint") or "")
        b = baselines.get(fp)
        if not b:
            r["delta_rouge_l_vs_same_decode"] = None
            r["delta_meteor_vs_same_decode"] = None
            r["delta_bleu_4_vs_same_decode"] = None
            r["delta_dist_1_vs_same_decode"] = None
            r["delta_dist_2_vs_same_decode"] = None
            r["delta_dirty_hit_rate_vs_same_decode"] = None
            r["delta_pred_len_ratio_vs_same_decode"] = None
            r["same_decode_baseline_run"] = None
            r["target_tradeoff_flag"] = "baseline_missing"
            continue
        r["same_decode_baseline_run"] = b.get("eval_run_dir")
        r["delta_rouge_l_vs_same_decode"] = _delta(r["rouge_l"], b["rouge_l"])
        r["delta_meteor_vs_same_decode"] = _delta(r["meteor"], b["meteor"])
        r["delta_bleu_4_vs_same_decode"] = _delta(r["bleu_4"], b["bleu_4"])
        r["delta_dist_1_vs_same_decode"] = _delta(r["dist_1"], b["dist_1"])
        r["delta_dist_2_vs_same_decode"] = _delta(r["dist_2"], b["dist_2"])
        r["delta_dirty_hit_rate_vs_same_decode"] = _delta(r["dirty_hit_rate"], b["dirty_hit_rate"])
        r["delta_pred_len_ratio_vs_same_decode"] = _delta(r["pred_len_ratio"], b["pred_len_ratio"])
        r["target_tradeoff_flag"] = _target_tradeoff_flag(r)

    apply_phase1_style_scoring(rows)

    stem = "phase2_rerank_summary"
    out.mkdir(parents=True, exist_ok=True)
    csv_p = out / f"{stem}.csv"
    json_p = out / f"{stem}.json"
    fieldnames = list(rows[0].keys()) if rows else []
    with open(csv_p, "w", encoding="utf-8", newline="") as cf:
        w = csv.DictWriter(cf, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    def _json_safe(o: Any) -> Any:
        if isinstance(o, dict):
            return {k: _json_safe(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_json_safe(v) for v in o]
        if isinstance(o, float) and (o != o or o in (float("inf"), float("-inf"))):
            return None
        if isinstance(o, bool):
            return o
        return o

    payload = {
        "iteration_dir": str(root_it),
        "rerank_root": str(root_it / "rerank"),
        "output_stem": stem,
        "metrics_semantics": {
            "base_columns_from_phase1": "与 eval 相同的 rouge_l/meteor/bleu_4 等（来自 eval_metrics.json 嵌套 metrics.*）",
            "paper_columns": "paper_bleu_4 / paper_rouge_l_f / paper_dist_2_pct（来自 eval_metrics.json paper_metrics）",
            "rerank_columns": "rerank_method、num_return_sequences、delta_*_vs_same_decode 等",
            "note": "勿将 phase1 的 repo 列与 paper_metrics 混读；rerank 行仅含 rerank_enabled 的 rerank_summary.json。",
        },
        "rows": _json_safe(rows),
    }
    with open(json_p, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    mm_path = out / "matrix_manifest.json"
    mm_payload = {
        "schema": "odcr_matrix_manifest_v1",
        "phase": "phase2_rerank",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "iteration_dir": str(root_it),
        "matrix_out_dir": str(out),
        "metrics_paths": [str(p.resolve()) for p in paths],
        "primary_summary_files": {
            "csv": str((out / f"{stem}.csv").resolve()),
            "json": str((out / f"{stem}.json").resolve()),
        },
        "metrics_semantics": payload["metrics_semantics"],
    }
    with open(mm_path, "w", encoding="utf-8") as mf:
        json.dump(mm_payload, mf, ensure_ascii=False, indent=2)
        mf.write("\n")

    print(f"[phase2_rerank] wrote {csv_p}", flush=True)
    print(f"[phase2_rerank] wrote {json_p}", flush=True)

    try:
        import pandas as pd

        return pd.DataFrame(rows)
    except ImportError:
        return rows


__all__ = ["generate_phase2_rerank_summary"]
