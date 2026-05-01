"""
Phase 1：扫描 runs/.../eval/*/eval_metrics.json，生成综合排序表与双基线标红/提醒列。
**Phase 1** 表示 matrix/评测汇总语义，与历史 pipeline「阶段编号」无关；磁盘产物名 ``phase1_summary.*`` 与此一致。
仅以 eval_metrics.json 为真相源；历史 metrics.json 只作旧产物读取。
主分 main_score / 辅助 aux_bonus / final_score；composite_score 与 final_score 同值。
实现不依赖 pandas（顶层无 pandas import）；若已安装 pandas，返回 DataFrame。
"""
from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from odcr_core import path_layout
from odcr_core.generation_semantics import compute_generation_semantic_family_tag

# --- 相对 nucleus 基线 (t=0.8, p=0.9) 红色阈值 ---
NUC_REL_ROUGE = 0.03
NUC_REL_METEOR = 0.03
NUC_REL_BLEU = 0.10
NUC_LEN_DEV = 0.30

# phase1_summary 列语义：与 eval_metrics.json 中 repo_metrics / paper_metrics 区分（勿混读）
PHASE1_REPO_COLUMN_KEYS = (
    "mae",
    "rmse",
    "rouge_l",
    "meteor",
    "bleu_4",
    "dist_1",
    "dist_2",
    "collapse_unique_ratio",
    "collapse_top1_ratio",
    "mean_pred_len_words",
    "mean_ref_len_words",
    "pred_len_ratio",
    "dirty_hit_rate",
)
PHASE1_PAPER_COLUMN_KEYS = ("paper_bleu_4", "paper_rouge_l_f", "paper_dist_2_pct")

# --- 主目标 / 辅助 bonus（归一化后线性组合）---
W_MAIN_ROUGE = 0.40
W_MAIN_METEOR = 0.35
W_MAIN_BLEU = 0.25
W_AUX_D1 = 0.04
W_AUX_D2 = 0.06

# --- Gate：推荐 + dirty + 长度 + 塌缩 ---
GATE_REC_REL = 0.08
GATE_LEN_LO = 0.85
GATE_LEN_HI = 1.30
GATE_LEN_SOFT_LO = 0.92
GATE_LEN_SOFT_HI = 1.15
GATE_COLLAPSE_TOP1 = 0.01


def _collect_eval_metrics_paths(iteration_dir: Path) -> List[Path]:
    ev = iteration_dir / "eval"
    if not ev.is_dir():
        return []
    out: List[Path] = []
    for run in sorted(ev.iterdir()):
        if not run.is_dir():
            continue
        mp = path_layout.eval_metrics_path(run, rerank=False)
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


def _decode_int_field(dec_res: Dict[str, Any], dec_raw: Dict[str, Any], key: str) -> Optional[int]:
    v = dec_res[key] if key in dec_res else dec_raw.get(key)
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def row_from_metrics_root(
    root: Dict[str, Any],
    *,
    metrics_path: str = "",
) -> Dict[str, Any]:
    """
    从已解析的 eval_metrics.json 根对象构造 phase1 汇总行（与 ``_load_run_row`` 列一致）。

    ``metrics_path`` 可选；若省略则尽量使用 ``root['eval_run_dir']`` / ``root['metrics_path']``。
    """
    m = root.get("metrics") or {}
    rec = m.get("recommendation") or {}
    expl = m.get("explanation") or {}
    rouge = expl.get("rouge") or {}
    bleu = expl.get("bleu") or {}
    dist = expl.get("dist") or {}
    cs = m.get("collapse_stats") or {}
    tmc = m.get("text_metrics_corpus_and_sentence") or {}
    corp = tmc.get("corpus_level") or {}
    dirty = m.get("dirty_text") or {}
    pm = root.get("paper_metrics") or m.get("paper_metrics") or {}
    pm_bleu = pm.get("bleu") if isinstance(pm, dict) else {}
    pm_rg = pm.get("rouge") if isinstance(pm, dict) else {}
    pm_di = pm.get("distinct_corpus") if isinstance(pm, dict) else {}
    pm_di_pct = pm_di.get("scale_percent_0_100") if isinstance(pm_di, dict) else {}

    dec_res = root.get("generation_semantic_resolved") or {}
    dec_raw = root.get("decode") or {}
    strategy = (dec_res.get("strategy") or dec_raw.get("decode_strategy") or "").strip().lower()
    temp = _f(dec_res.get("temperature"), _f(dec_raw.get("generate_temperature")))
    top_p = _f(dec_res.get("top_p"), _f(dec_raw.get("generate_top_p")))
    rp = _f(dec_res.get("repetition_penalty"), _f(dec_raw.get("repetition_penalty")))
    mxlen = int(dec_res.get("max_explanation_length") or dec_raw.get("max_explanation_length") or 0)
    nr = _decode_int_field(dec_res, dec_raw, "no_repeat_ngram_size")
    mnl = _decode_int_field(dec_res, dec_raw, "min_len")
    dft = (
        dec_res.get("generation_semantic_family_tag")
        if isinstance(dec_res.get("generation_semantic_family_tag"), str)
        else ""
    )
    if not (dft or "").strip():
        dft = compute_generation_semantic_family_tag(
            {
                "strategy": strategy,
                "temperature": temp,
                "top_p": top_p,
                "repetition_penalty": rp,
                "max_explanation_length": mxlen,
                "no_repeat_ngram_size": nr,
                "min_len": mnl,
            }
        )

    pred_len = _f(corp.get("mean_pred_len_words"))
    ref_len = _f(corp.get("mean_ref_len_words"))
    pred_len_ratio = pred_len / ref_len if ref_len and ref_len > 0 else float("nan")

    hit_dirty = _f(dirty.get("hit_rate"), 0.0)

    parent = (root.get("eval_run_dir") or "").strip()
    if not parent and metrics_path:
        parent = str(Path(metrics_path).expanduser().resolve().parent)
    mp_out = (metrics_path or "").strip() or str(root.get("metrics_path") or root.get("metrics_json_path") or "")
    return {
        "eval_run_dir": parent,
        "metrics_path": mp_out,
        "metrics_schema_version": root.get("metrics_schema_version", ""),
        "generation_semantic_fingerprint": root.get("generation_semantic_fingerprint", ""),
        "strategy": strategy,
        "temperature": temp,
        "top_p": top_p,
        "repetition_penalty": rp,
        "max_explanation_length": mxlen,
        "no_repeat_ngram_size": nr,
        "min_len": mnl,
        "generation_semantic_family_tag": dft,
        "mae": _f(rec.get("mae")),
        "rmse": _f(rec.get("rmse")),
        "rouge_l": _f(rouge.get("l")),
        "meteor": _f(expl.get("meteor")),
        "bleu_4": _f(bleu.get("4")),
        "dist_1": _f(dist.get("1")),
        "dist_2": _f(dist.get("2")),
        "collapse_unique_ratio": _f(cs.get("pred_unique_ratio")),
        "collapse_top1_ratio": _f(cs.get("top1_pred_ratio")),
        "mean_pred_len_words": pred_len,
        "mean_ref_len_words": ref_len,
        "pred_len_ratio": pred_len_ratio,
        "dirty_hit_rate": hit_dirty,
        "paper_bleu_4": _f(pm_bleu.get("4")) if isinstance(pm_bleu, dict) else float("nan"),
        "paper_rouge_l_f": _f(pm_rg.get("rouge_l_f")) if isinstance(pm_rg, dict) else float("nan"),
        "paper_dist_2_pct": _f(pm_di_pct.get("2")) if isinstance(pm_di_pct, dict) else float("nan"),
    }


def _load_run_row(metrics_path: Path) -> Dict[str, Any]:
    with open(metrics_path, "r", encoding="utf-8") as f:
        root = json.load(f)
    return row_from_metrics_root(root, metrics_path=str(metrics_path.resolve()))


def _pick_baseline(rows: List[Dict[str, Any]], pred) -> Optional[Dict[str, Any]]:
    cand = [r for r in rows if pred(r)]
    if not cand:
        return None
    cand.sort(key=lambda r: os.path.getmtime(r["metrics_path"]), reverse=True)
    return cand[0]


def _minmax_col(rows: List[Dict[str, Any]], key: str) -> List[float]:
    vals = [float(r[key]) for r in rows if r.get(key) == r.get(key)]  # not nan
    if not vals:
        return [0.5] * len(rows)
    lo, hi = min(vals), max(vals)
    out: List[float] = []
    for r in rows:
        v = r.get(key)
        if v is None or (isinstance(v, float) and v != v):
            out.append(0.5)
            continue
        fv = float(v)
        if hi - lo < 1e-12:
            out.append(0.5)
        else:
            out.append((fv - lo) / (hi - lo))
    return out


def _fmt_red_flags_nucleus(row: Dict[str, Any], nuc: Dict[str, Any]) -> str:
    if not nuc:
        return ""
    parts: List[str] = []
    for name, cur, base, thr in (
        ("rouge_l", row["rouge_l"], nuc["rouge_l"], NUC_REL_ROUGE),
        ("meteor", row["meteor"], nuc["meteor"], NUC_REL_METEOR),
        ("bleu_4", row["bleu_4"], nuc["bleu_4"], NUC_REL_BLEU),
    ):
        if base == base and cur == cur and base > 0:
            drop = (base - cur) / base
            if drop > thr:
                parts.append(f"{name}_drop>{thr:.0%}")
    pr = row["pred_len_ratio"]
    if pr == pr and abs(pr - 1.0) > NUC_LEN_DEV:
        parts.append(f"len_ratio_dev>{NUC_LEN_DEV:.0%}")
    dr = row["dirty_hit_rate"]
    nb = float(nuc.get("dirty_hit_rate", 0.0) or 0.0)
    if dr > nb + 1e-9:
        parts.append("dirty_up")
    return ";".join(parts)


def _compute_gates(row: Dict[str, Any], nucleus_b: Optional[Dict[str, Any]]) -> tuple[bool, str, str]:
    fails: List[str] = []
    warns: List[str] = []

    if nucleus_b:
        nm, nr_ = nucleus_b["mae"], nucleus_b["rmse"]
        if row["mae"] == row["mae"] and nm == nm and row["mae"] > nm * (1.0 + GATE_REC_REL):
            fails.append("rec_mae_worse_8pct_vs_nucleus")
        if row["rmse"] == row["rmse"] and nr_ == nr_ and row["rmse"] > nr_ * (1.0 + GATE_REC_REL):
            fails.append("rec_rmse_worse_8pct_vs_nucleus")

        bd = float(nucleus_b.get("dirty_hit_rate") or 0.0)
        dr = row["dirty_hit_rate"]
        thr_d = max(bd + 0.01, bd * 1.25)
        if dr == dr and dr > thr_d:
            fails.append("dirty_hit_rate_high_vs_nucleus")
        elif dr == dr and dr > bd + 1e-9:
            warns.append("dirty_hit_rate_up_mild")

        nuc_u = nucleus_b.get("collapse_unique_ratio")
        cu = row["collapse_unique_ratio"]
        if nuc_u == nuc_u and cu == cu and cu < float(nuc_u) * 0.92:
            warns.append("collapse_unique_low_vs_nucleus")

    pr = row["pred_len_ratio"]
    if pr != pr:
        fails.append("pred_len_ratio_invalid")
    else:
        if pr < GATE_LEN_LO or pr > GATE_LEN_HI:
            fails.append("pred_len_ratio_out_of_band")
        elif pr < GATE_LEN_SOFT_LO or pr > GATE_LEN_SOFT_HI:
            warns.append("pred_len_ratio_soft_drift")

    t1 = row["collapse_top1_ratio"]
    if t1 == t1 and t1 > GATE_COLLAPSE_TOP1:
        fails.append("collapse_top1_gt_1pct")
    elif t1 != t1:
        warns.append("collapse_top1_ratio_nan")

    return (len(fails) == 0, ";".join(fails), ";".join(warns))


def _fmt_warn_greedy(row: Dict[str, Any], gr: Dict[str, Any]) -> str:
    if not gr:
        return ""
    parts: List[str] = []
    gd1, gd2 = gr.get("dist_1"), gr.get("dist_2")
    if gd1 == gd1 and row["dist_1"] <= float(gd1) * 1.002:
        parts.append("dist1_no_gain_vs_greedy")
    if gd2 == gd2 and row["dist_2"] <= float(gd2) * 1.002:
        parts.append("dist2_no_gain_vs_greedy")
    gt1 = gr.get("collapse_top1_ratio")
    if gt1 == gt1 and row["collapse_top1_ratio"] >= float(gt1) * 0.98:
        parts.append("top1_not_down_vs_greedy")
    gu = gr.get("collapse_unique_ratio")
    if gu == gu and row["collapse_unique_ratio"] <= float(gu) * 1.01:
        parts.append("unique_not_up_vs_greedy")
    return ";".join(parts)


def _sort_key(row: Dict[str, Any]) -> tuple:
    fs = row.get("final_score", 0.0)
    if isinstance(fs, float) and fs != fs:
        fs = -1e9
    gate_ord = 0 if row.get("gate_pass") else 1
    return (gate_ord, -float(fs))


def apply_phase1_style_scoring(rows: List[Dict[str, Any]]) -> None:
    """与 generate_phase1_summary 一致的 minmax、main/aux/final、gate、标红/提醒、排序与 rank_eligible（就地修改）。"""
    if not rows:
        return

    def _is_greedy(r: Dict[str, Any]) -> bool:
        return r["strategy"] == "greedy"

    def _is_nuc_base(r: Dict[str, Any]) -> bool:
        return (
            r["strategy"] == "nucleus"
            and abs(r["temperature"] - 0.8) < 1e-5
            and abs(r["top_p"] - 0.9) < 1e-5
        )

    greedy_b = _pick_baseline(rows, _is_greedy)
    nucleus_b = _pick_baseline(rows, _is_nuc_base)

    n1 = _minmax_col(rows, "rouge_l")
    n2 = _minmax_col(rows, "meteor")
    n3 = _minmax_col(rows, "bleu_4")
    n4 = _minmax_col(rows, "dist_1")
    n5 = _minmax_col(rows, "dist_2")
    for i, r in enumerate(rows):
        r["rouge_l_norm"] = n1[i]
        r["meteor_norm"] = n2[i]
        r["bleu_4_norm"] = n3[i]
        r["dist_1_norm"] = n4[i]
        r["dist_2_norm"] = n5[i]
        main = W_MAIN_ROUGE * n1[i] + W_MAIN_METEOR * n2[i] + W_MAIN_BLEU * n3[i]
        aux = W_AUX_D1 * n4[i] + W_AUX_D2 * n5[i]
        final = main + aux
        r["main_score"] = main
        r["aux_bonus"] = aux
        r["final_score"] = final
        r["composite_score"] = final

    for r in rows:
        gp, gf, gw = _compute_gates(r, nucleus_b)
        r["gate_pass"] = gp
        r["gate_fail_reasons"] = gf
        r["gate_warning_reasons"] = gw

    nuc_snap = (
        {
            "rouge_l": nucleus_b["rouge_l"],
            "meteor": nucleus_b["meteor"],
            "bleu_4": nucleus_b["bleu_4"],
            "dirty_hit_rate": nucleus_b["dirty_hit_rate"],
            "collapse_unique_ratio": nucleus_b["collapse_unique_ratio"],
        }
        if nucleus_b
        else {}
    )

    nuc_dir = nucleus_b["eval_run_dir"] if nucleus_b else None
    gr_dir = greedy_b["eval_run_dir"] if greedy_b else None

    for r in rows:
        if nuc_dir and r["eval_run_dir"] == nuc_dir:
            r["red_vs_nucleus_base"] = ""
        else:
            r["red_vs_nucleus_base"] = _fmt_red_flags_nucleus(r, nuc_snap)
        if gr_dir and r["eval_run_dir"] == gr_dir:
            r["warn_vs_greedy"] = ""
        else:
            r["warn_vs_greedy"] = _fmt_warn_greedy(r, greedy_b or {})

    rows.sort(key=_sort_key)

    rp, rf = 0, 0
    for r in rows:
        if r["gate_pass"]:
            rp += 1
            r["rank_eligible"] = rp
        else:
            rf += 1
            r["rank_eligible"] = -rf


def generate_phase1_summary(
    iteration_dir: str,
    *,
    out_dir: str,
    only_latest_n: Optional[int] = None,
    output_stem: str = "phase1_summary",
) -> Union[List[Dict[str, Any]], Any]:
    """
    扫描 ``<iteration>/eval/<run>/eval_metrics.json``，将 phase1 汇总写入 ``out_dir``（通常为 matrix/<run>/）。

    only_latest_n：若为正整数，仅纳入 eval_metrics.json **mtime 最新** 的 N 个 run。
    """
    root_it = Path(iteration_dir).expanduser().resolve()
    if not root_it.is_dir():
        raise FileNotFoundError(f"迭代目录不存在: {root_it}")
    out = Path(out_dir).expanduser().resolve()

    stem = Path(str(output_stem or "phase1_summary")).name
    if not stem or stem in (".", ".."):
        stem = "phase1_summary"

    paths = sorted(_collect_eval_metrics_paths(root_it))
    if not paths:
        raise FileNotFoundError(f"未找到 eval_metrics.json: {root_it}/eval/*/eval_metrics.json")

    if only_latest_n is not None and only_latest_n > 0:
        paths = sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)[: int(only_latest_n)]
        if not paths:
            raise FileNotFoundError(f"only_latest_n={only_latest_n} 未选中任何 eval_metrics.json")

    rows: List[Dict[str, Any]] = [_load_run_row(p) for p in paths]

    apply_phase1_style_scoring(rows)

    def _is_greedy_meta(r: Dict[str, Any]) -> bool:
        return r["strategy"] == "greedy"

    def _is_nuc_base_meta(r: Dict[str, Any]) -> bool:
        return (
            r["strategy"] == "nucleus"
            and abs(r["temperature"] - 0.8) < 1e-5
            and abs(r["top_p"] - 0.9) < 1e-5
        )

    greedy_b = _pick_baseline(rows, _is_greedy_meta)
    nucleus_b = _pick_baseline(rows, _is_nuc_base_meta)

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
        "eval_metrics_root": str(root_it / "eval"),
        "only_latest_n": int(only_latest_n) if only_latest_n and only_latest_n > 0 else None,
        "output_stem": stem,
        "metrics_semantics": {
            "repo_training_table_columns": list(PHASE1_REPO_COLUMN_KEYS),
            "paper_table_columns": list(PHASE1_PAPER_COLUMN_KEYS),
            "truth_in_eval_metrics_json": {
                "repo_metrics": "eval_metrics.json['repo_metrics'] 与嵌套 metrics.*",
                "paper_metrics": "eval_metrics.json['paper_metrics']",
            },
            "note": "论文/对外表优先对齐 paper_* 列与 paper_metrics；repo 列来自训练侧嵌套 metrics，勿与 paper 混读。",
        },
        "baseline_greedy_run": greedy_b["eval_run_dir"] if greedy_b else None,
        "baseline_nucleus_08_09_run": nucleus_b["eval_run_dir"] if nucleus_b else None,
        "weights": {
            "main": {
                "rouge_l": W_MAIN_ROUGE,
                "meteor": W_MAIN_METEOR,
                "bleu_4": W_MAIN_BLEU,
            },
            "aux_bonus": {"dist_1": W_AUX_D1, "dist_2": W_AUX_D2},
        },
        "rows": _json_safe(rows),
    }
    with open(json_p, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    mm_path = out / "matrix_manifest.json"
    mm_payload = {
        "schema": "odcr_matrix_manifest_v1",
        "phase": "phase1",
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

    print(f"[phase1] wrote {csv_p}", flush=True)
    print(f"[phase1] wrote {json_p}", flush=True)
    if not nucleus_b:
        print("[phase1] 警告: 未找到 nucleus t=0.8 p=0.9 基线 run，gate 与 red_vs_nucleus 部分失效。", flush=True)
    if not greedy_b:
        print("[phase1] 警告: 未找到 greedy 基线 run，warn_vs_greedy 为空。", flush=True)

    try:
        import pandas as pd

        return pd.DataFrame(rows)
    except ImportError:
        return rows


__all__ = [
    "PHASE1_PAPER_COLUMN_KEYS",
    "PHASE1_REPO_COLUMN_KEYS",
    "apply_phase1_style_scoring",
    "generate_phase1_summary",
    "row_from_metrics_root",
]
