# -*- coding: utf-8 -*-
"""Step5 评测：按 sample_id 合并分片结果、扩展文本指标、落盘。"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from collections import Counter, deque
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from paths_config import require_nltk_data_dir

_NLTK_LOCAL = require_nltk_data_dir()
os.environ["NLTK_DATA"] = _NLTK_LOCAL
import nltk

if _NLTK_LOCAL not in nltk.data.path:
    nltk.data.path.insert(0, _NLTK_LOCAL)

from nltk import word_tokenize


def merge_eval_rows_by_sample_id(
    rows_per_rank: Sequence[Sequence[Dict[str, Any]]],
    expected_n: int,
) -> List[Dict[str, Any]]:
    """DDP 各 rank 本地行合并后按 sample_id 排序；校验无重复、无缺失。"""
    flat: List[Dict[str, Any]] = []
    for rows in rows_per_rank:
        flat.extend(rows)
    if len(flat) != expected_n:
        raise RuntimeError(
            f"eval gather 条数不一致: 期望 {expected_n}, 实际 {len(flat)}"
        )
    by_id: Dict[int, Dict[str, Any]] = {}
    for row in flat:
        sid = int(row["sample_id"])
        if sid in by_id:
            raise RuntimeError(f"重复 sample_id={sid}")
        by_id[sid] = row
    want = set(range(expected_n))
    got = set(by_id.keys())
    if got != want:
        raise RuntimeError(
            f"sample_id 集合错误: 缺失={sorted(want - got)} 多余={sorted(got - want)}"
        )
    return [by_id[i] for i in range(expected_n)]


def _tokenize_safe(s: str) -> List[str]:
    try:
        return word_tokenize(s or "")
    except Exception:
        return re.findall(r"\S+", s or "")


def corpus_distinct_n(sentences: Sequence[str], n: int) -> float:
    """语料级：所有句子的 n-gram distinct。"""
    unique: set = set()
    total = 0
    for s in sentences:
        toks = _tokenize_safe(s)
        if n == 1:
            grams = [(t,) for t in toks]
        else:
            grams = [tuple(toks[i : i + n]) for i in range(max(0, len(toks) - n + 1))]
        unique.update(grams)
        total += len(grams)
    if total <= 0:
        return 0.0
    return round(100.0 * len(unique) / total, 4)


def sentence_mean_distinct_n(sentences: Sequence[str], n: int) -> float:
    """句级：每句 distinct_n 再平均（百分比）。"""
    scores: List[float] = []
    for s in sentences:
        toks = _tokenize_safe(s)
        if n == 1:
            grams = [(t,) for t in toks]
        else:
            grams = [tuple(toks[i : i + n]) for i in range(max(0, len(toks) - n + 1))]
        if not grams:
            scores.append(0.0)
            continue
        scores.append(100.0 * len(set(grams)) / len(grams))
    if not scores:
        return 0.0
    return round(float(sum(scores) / len(scores)), 4)


def unigram_repetition_ratio(sentence: str) -> float:
    """句内重复 unigram 比例: 1 - unique/total。"""
    toks = _tokenize_safe(sentence)
    if len(toks) <= 1:
        return 0.0
    return 1.0 - len(set(toks)) / len(toks)


def trigram_repetition_ratio(sentence: str) -> float:
    """连续 trigram 重复率：出现次数>1 的 trigram 数 / trigram 总数。"""
    toks = _tokenize_safe(sentence)
    if len(toks) < 3:
        return 0.0
    trigrams = [tuple(toks[i : i + 3]) for i in range(len(toks) - 2)]
    c = Counter(trigrams)
    rep = sum(cnt for cnt in c.values() if cnt > 1)
    return rep / len(trigrams)


def mean_length_words(sentences: Sequence[str]) -> float:
    lens = [len(_tokenize_safe(s)) for s in sentences]
    if not lens:
        return 0.0
    return round(sum(lens) / len(lens), 4)


def extended_text_metrics_bundle(predictions: Sequence[str], references: Sequence[str]) -> Dict[str, Any]:
    """与 evaluate_text 并存：语料级 + 句级 distinct / 重复 / 长度。

    仅供诊断（塌缩、重复、句内/句间多样性）；论文主表 DIST-1/DIST-2 以 base_utils.evaluate_text 的 dist 为准，
    勿将本 bundle 中的 distinct 与主表横向对比。
    """
    pred_list = list(predictions)
    ref_list = list(references)
    uni_rep = [unigram_repetition_ratio(p) for p in pred_list]
    tri_rep = [trigram_repetition_ratio(p) for p in pred_list]
    return {
        "corpus_level": {
            "distinct_1_pct": corpus_distinct_n(pred_list, 1),
            "distinct_2_pct": corpus_distinct_n(pred_list, 2),
            "mean_pred_len_words": mean_length_words(pred_list),
            "mean_ref_len_words": mean_length_words(ref_list),
        },
        "sentence_level_mean": {
            "distinct_1_pct": sentence_mean_distinct_n(pred_list, 1),
            "distinct_2_pct": sentence_mean_distinct_n(pred_list, 2),
            "unigram_repetition_ratio": round(sum(uni_rep) / max(len(uni_rep), 1), 4),
            "trigram_repetition_ratio": round(sum(tri_rep) / max(len(tri_rep), 1), 4),
        },
    }


def write_predictions_csv(path: str, rows: Iterable[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_predictions_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_last_n_lines(path: str, n: int = 120) -> Tuple[List[str], Optional[str]]:
    """读取文本文件末尾 n 行（整行字符串，保留行末 \\n 若 readlines 提供）。成功时第二项为 None。"""
    if not (path or "").strip():
        return [], "empty path"
    ap = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(ap):
        return [], f"file not found: {ap}"
    if n <= 0:
        return [], None
    try:
        dq: deque[str] = deque(maxlen=n)
        with open(ap, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                dq.append(line)
        return list(dq), None
    except OSError as e:
        return [], str(e)


def _digest_fmt_scalar(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, float) and (v != v):  # NaN
        return "null"
    return str(v)


class _MISSING:
    pass


def _digest_label_missing(v: Any) -> str:
    if v is _MISSING:
        return "missing"
    if v is None:
        return "null"
    return _digest_fmt_scalar(v)


def _digest_truncate_line(s: str, max_len: int = 220) -> str:
    t = (s or "").replace("\r", "").replace("\n", "↵ ")
    if len(t) <= max_len:
        return t
    return t[: max_len - 3] + "..."


def _digest_norm_pred_text(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def write_eval_digest_log(
    *,
    eval_subdir: str,
    metrics_final: Dict[str, Any],
    merged_rows: Sequence[Dict[str, Any]],
    final_cfg: Any,
    decode_cfg: Dict[str, Any],
    active_log_file: Optional[str],
    task_idx: int,
    auxiliary: str,
    target: str,
    eval_export_tag: str,
    command: str,
    eval_timing_summary: Optional[Dict[str, Any]] = None,
) -> str:
    """写入当次 eval/rerank 产物目录下的 eval_digest.log；返回绝对路径。"""
    sub = os.path.abspath(eval_subdir)
    os.makedirs(sub, exist_ok=True)
    out_path = os.path.join(sub, "eval_digest.log")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ckpt = ""
    try:
        ckpt = os.path.abspath(str(getattr(final_cfg, "save_file", "") or ""))
    except Exception:
        ckpt = ""

    lines: List[str] = []

    def _emit(section: str) -> None:
        lines.append(section)
        lines.append("")

    def _kv(key: str, val: str) -> None:
        lines.append(f"{key}={val}")

    # ----- [Eval Digest Meta] -----
    _emit("[Eval Digest Meta]")
    _kv("timestamp", ts)
    _kv("checkpoint", ckpt or "missing")
    _kv("eval_export_dir", sub)
    _kv("decode_strategy", _digest_label_missing(decode_cfg.get("decode_strategy")))
    _kv("generate_temperature", _digest_label_missing(decode_cfg.get("generate_temperature")))
    _kv("generate_top_p", _digest_label_missing(decode_cfg.get("generate_top_p")))
    _kv("repetition_penalty", _digest_label_missing(decode_cfg.get("repetition_penalty")))
    _kv("label_smoothing", _digest_label_missing(decode_cfg.get("label_smoothing")))
    _kv("max_explanation_length", _digest_label_missing(decode_cfg.get("max_explanation_length")))
    for _dk in (
        "candidate_family",
        "soft_max_len",
        "hard_max_len",
        "eos_boost_start",
        "eos_boost_value",
    ):
        if _dk in (decode_cfg or {}):
            _kv(f"decode.{_dk}", _digest_label_missing(decode_cfg.get(_dk)))
    _kv("task_idx", str(int(task_idx)))
    _kv("auxiliary", str(auxiliary))
    _kv("target", str(target))
    _kv("command", str(command))
    _kv("eval_export_tag", str(eval_export_tag))
    lines.append("")

    if eval_timing_summary:
        _emit("[Eval Timing Summary]")
        for _tk in sorted(eval_timing_summary.keys()):
            _kv(f"timing.{_tk}", _digest_fmt_scalar(eval_timing_summary.get(_tk)))
        lines.append("")

    mf = metrics_final if isinstance(metrics_final, dict) else {}

    # ----- [Metrics Summary] -----
    _emit("[Metrics Summary]")
    rec = mf.get("recommendation")
    if isinstance(rec, dict):
        _kv("metrics.recommendation.mae", _digest_label_missing(rec.get("mae")))
        _kv("metrics.recommendation.rmse", _digest_label_missing(rec.get("rmse")))
    else:
        _kv("metrics.recommendation.mae", "missing")
        _kv("metrics.recommendation.rmse", "missing")

    expl = mf.get("explanation")
    if isinstance(expl, dict):
        rouge = expl.get("rouge")
        if isinstance(rouge, dict):
            for k in ("1", "2", "l"):
                kk = f"metrics.explanation.rouge.{k}"
                _kv(kk, _digest_label_missing(rouge.get(k)))
        else:
            _kv("metrics.explanation.rouge", "missing")
        bleu = expl.get("bleu")
        if isinstance(bleu, dict):
            for k in sorted(bleu.keys(), key=lambda x: str(x)):
                _kv(f"metrics.explanation.bleu.{k}", _digest_label_missing(bleu.get(k)))
        else:
            _kv("metrics.explanation.bleu", "missing")
        _kv("metrics.explanation.meteor", _digest_label_missing(expl.get("meteor")))
        _kv(
            "metrics.explanation.dist_note",
            "DIST-1/DIST-2 = evaluate_text corpus-level (paper-compatible); not the ext bundle below",
        )
        dist = expl.get("dist")
        if isinstance(dist, dict):
            for k in sorted(dist.keys(), key=lambda x: str(x)):
                _kv(f"metrics.explanation.dist.{k}", _digest_label_missing(dist.get(k)))
        else:
            _kv("metrics.explanation.dist", "missing")
    else:
        _kv("metrics.explanation", "missing")

    pm = mf.get("paper_metrics")
    if isinstance(pm, dict) and pm:
        _emit("[Paper-comparable metrics]")
        _kv("paper_metrics.schema_version", _digest_label_missing(pm.get("schema_version")))
        tok = pm.get("tokenization")
        if isinstance(tok, dict):
            _kv("paper_metrics.tokenization", str(tok.get("name")))
        pbleu = pm.get("bleu")
        if isinstance(pbleu, dict):
            for k in ("1", "2", "3", "4"):
                _kv(f"paper_metrics.bleu.percent.{k}", _digest_label_missing(pbleu.get(k)))
        prg = pm.get("rouge")
        if isinstance(prg, dict):
            _kv("paper_metrics.rouge.rouge_l_f_pct", _digest_label_missing(prg.get("rouge_l_f")))
        pdi = pm.get("distinct_corpus")
        if isinstance(pdi, dict):
            sp = pdi.get("scale_percent_0_100") or {}
            if isinstance(sp, dict):
                _kv("paper_metrics.distinct.corpus.percent.1", _digest_label_missing(sp.get("1")))
                _kv("paper_metrics.distinct.corpus.percent.2", _digest_label_missing(sp.get("2")))
        lines.append("")

    tmc = mf.get("text_metrics_corpus_and_sentence")
    if isinstance(tmc, dict):
        _emit("[Metrics secondary — debug / non-paper]")
        _kv(
            "metrics.text_metrics.ext_note",
            "extended_text_metrics_bundle：诊断专用；论文主表 DIST-1/DIST-2 仅见上文 metrics.explanation.dist（evaluate_text 语料级）",
        )
        corp = tmc.get("corpus_level")
        if isinstance(corp, dict):
            _kv(
                "metrics.text_metrics.corpus.distinct_1_pct",
                _digest_label_missing(corp.get("distinct_1_pct")),
            )
            _kv(
                "metrics.text_metrics.corpus.distinct_2_pct",
                _digest_label_missing(corp.get("distinct_2_pct")),
            )
            _kv(
                "metrics.text_metrics.corpus.mean_pred_len_words",
                _digest_label_missing(corp.get("mean_pred_len_words")),
            )
            _kv(
                "metrics.text_metrics.corpus.mean_ref_len_words",
                _digest_label_missing(corp.get("mean_ref_len_words")),
            )
        else:
            _kv("metrics.text_metrics.corpus_level", "missing")
        sent = tmc.get("sentence_level_mean")
        if isinstance(sent, dict):
            _kv(
                "metrics.text_metrics.sentence.distinct_1_pct",
                _digest_label_missing(sent.get("distinct_1_pct")),
            )
            _kv(
                "metrics.text_metrics.sentence.distinct_2_pct",
                _digest_label_missing(sent.get("distinct_2_pct")),
            )
            _kv(
                "metrics.text_metrics.sentence.unigram_repetition_ratio",
                _digest_label_missing(sent.get("unigram_repetition_ratio")),
            )
            _kv(
                "metrics.text_metrics.sentence.trigram_repetition_ratio",
                _digest_label_missing(sent.get("trigram_repetition_ratio")),
            )
        else:
            _kv("metrics.text_metrics.sentence_level_mean", "missing")
    else:
        _kv("metrics.text_metrics_corpus_and_sentence", "missing")

    cs = mf.get("collapse_stats")
    if isinstance(cs, dict) and cs:
        _kv(
            "metrics.collapse_stats.unique_count",
            _digest_label_missing(cs.get("pred_unique_count")),
        )
        _kv(
            "metrics.collapse_stats.unique_ratio",
            _digest_label_missing(cs.get("pred_unique_ratio")),
        )
        _kv(
            "metrics.collapse_stats.top1_pred_text",
            _digest_truncate_line(str(cs.get("top1_pred_text") or ""), 300),
        )
        _kv("metrics.collapse_stats.top1_pred_count", _digest_label_missing(cs.get("top1_pred_count")))
        _kv("metrics.collapse_stats.top1_pred_ratio", _digest_label_missing(cs.get("top1_pred_ratio")))
        cw = cs.get("collapse_warnings")
        if cw is None:
            _kv("metrics.collapse_stats.warnings", "null")
        elif isinstance(cw, list):
            _kv("metrics.collapse_stats.warnings", "; ".join(str(x) for x in cw) if cw else "(empty)")
        else:
            _kv("metrics.collapse_stats.warnings", _digest_fmt_scalar(cw))
    else:
        _kv("metrics.collapse_stats", "null / missing")
    lines.append("")

    # ----- prediction stats from merged rows (stripped pred_text) -----
    preds_norm = [_digest_norm_pred_text(r.get("pred_text")) for r in merged_rows]
    n_total = len(preds_norm)
    ctr: Counter[str] = Counter(preds_norm)
    n_unique = len(ctr)
    uniq_ratio = round(n_unique / n_total, 8) if n_total else 0.0
    top1_cnt = ctr.most_common(1)[0][1] if n_total else 0
    top1_ratio = round(top1_cnt / n_total, 8) if n_total else 0.0
    top20 = ctr.most_common(20)

    _emit("[Prediction Collapse Summary]")
    _kv("predictions.total_samples", str(n_total))
    _kv("predictions.pred_text_unique_count", str(n_unique))
    _kv("predictions.pred_text_unique_ratio", str(uniq_ratio))
    _kv("predictions.top1_count", str(top1_cnt))
    _kv("predictions.top1_ratio", str(top1_ratio))
    _kv("predictions.pred_text_normalize_rule", "str(x).strip(); None->empty string")
    lines.append("")

    _emit("[Prediction Top20]")
    if n_total <= 0:
        _kv("predictions.top20", "empty corpus")
    else:
        for i, (txt, cnt) in enumerate(top20, 1):
            ratio = round(cnt / n_total, 8)
            _kv(f"predictions.top20.{i:02d}.count", str(cnt))
            _kv(f"predictions.top20.{i:02d}.ratio", str(ratio))
            lines.append(f"predictions.top20.{i:02d}.text={_digest_truncate_line(txt, 400)}")
    lines.append("")

    # ----- [Prediction Examples] -----
    _emit("[Prediction Examples]")
    ex_rule = "sorted_by_sample_id_ascending"
    ex_rows: List[Dict[str, Any]] = []
    if not merged_rows:
        ex_rule = "no_rows"
    elif all("sample_id" in r for r in merged_rows):
        try:
            ex_rows = sorted(merged_rows, key=lambda r: int(r["sample_id"]))[:10]
        except Exception:
            ex_rule = "fallback_first_10_rows_original_order_bad_sample_id"
            ex_rows = list(merged_rows)[:10]
    else:
        ex_rule = "fallback_first_10_rows_missing_sample_id_field"
        ex_rows = list(merged_rows)[:10]
    _kv("predictions.examples.selection_rule", ex_rule)
    for i, r in enumerate(ex_rows, 1):
        sid = r.get("sample_id", "missing")
        _kv(f"predictions.examples.{i:02d}.sample_id", _digest_fmt_scalar(sid))
        _kv(f"predictions.examples.{i:02d}.pred_rating", _digest_label_missing(r.get("pred_rating")))
        _kv(f"predictions.examples.{i:02d}.gt_rating", _digest_label_missing(r.get("gt_rating")))
        lines.append(
            f"predictions.examples.{i:02d}.pred_text={_digest_truncate_line(_digest_norm_pred_text(r.get('pred_text')), 320)}"
        )
        lines.append(
            f"predictions.examples.{i:02d}.ref_text={_digest_truncate_line(_digest_norm_pred_text(r.get('ref_text')), 320)}"
        )
    lines.append("")

    # ----- [Active Log Tail] -----
    _emit("[Active Log Tail]")
    src = (active_log_file or "").strip()
    if not src:
        src_abs = ""
        _kv("source_log", "null / missing")
    else:
        src_abs = os.path.abspath(os.path.expanduser(src))
        _kv("source_log", src_abs)
    _kv("tail_lines", "120")
    tail, terr = read_last_n_lines(src_abs, 120) if src_abs else ([], "no active log path")
    if terr:
        lines.append("[Log tail unavailable]")
        _kv("log_tail_error", terr)
        lines.append("")
    else:
        lines.append("--- begin log tail (verbatim lines) ---")
        for ln in tail:
            # 保留行内原文；仅去掉 readlines 带来的行尾换行符，避免与外层 join 重复空行
            if ln.endswith("\r\n"):
                lines.append(ln[:-2])
            elif ln.endswith("\n"):
                lines.append(ln[:-1])
            elif ln.endswith("\r"):
                lines.append(ln[:-1])
            else:
                lines.append(ln)
        lines.append("--- end log tail ---")
        lines.append("")

    body = "\n".join(lines)
    if not body.endswith("\n"):
        body += "\n"
    with open(out_path, "w", encoding="utf-8") as wf:
        wf.write(body)
    return out_path


def eval_decode_tag(*, decode_strategy: str, generate_temperature: float, generate_top_p: float) -> str:
    """目录名 / run 标签用：greedy 与 nucleus 不同温、不同 top_p 可区分。"""
    st = (decode_strategy or "").strip().lower()
    if st == "greedy":
        return "greedy"
    # 文件名安全：用小数点替换避免歧义
    ts = str(float(generate_temperature)).replace(".", "p")
    ps = str(float(generate_top_p)).replace(".", "p")
    return f"nucleus_t{ts}_p{ps}"


def compute_collapse_stats(
    predictions: Sequence[str],
    references: Sequence[str],
    *,
    top_k_file: int = 20,
) -> Dict[str, Any]:
    """生成塌缩统计：唯一输出占比、top1、topK 频次、平均词长（NLTK/word_tokenize 口径）。"""
    pred_list = [p if isinstance(p, str) else "" for p in predictions]
    ref_list = [r if isinstance(r, str) else "" for r in references]
    n = len(pred_list)
    if n == 0:
        return {
            "n_samples": 0,
            "pred_unique_count": 0,
            "pred_unique_ratio": 0.0,
            "top1_pred_text": "",
            "top1_pred_count": 0,
            "top1_pred_ratio": 0.0,
            "top10_pred_texts_with_count": [],
            "top20_pred_texts_with_count": [],
            "mean_pred_len_tokens": 0.0,
            "mean_ref_len_tokens": 0.0,
            "collapse_warnings": [],
        }

    ctr: Counter[str] = Counter(pred_list)
    pred_unique_count = len(ctr)
    pred_unique_ratio = round(pred_unique_count / n, 6)
    mc = ctr.most_common(1)[0]
    top1_text, top1_count = mc[0], int(mc[1])
    top1_pred_ratio = round(top1_count / n, 6)
    top10 = [{"text": t[:240], "count": int(c)} for t, c in ctr.most_common(10)]
    top20 = [{"text": t[:500], "count": int(c)} for t, c in ctr.most_common(top_k_file)]

    mean_pred_len_tokens = float(mean_length_words(pred_list))
    mean_ref_len_tokens = float(mean_length_words(ref_list))

    warnings: List[str] = []
    if top1_pred_ratio >= 0.2:
        warnings.append("top1_pred_ratio>=0.2")
    if n >= 100 and pred_unique_ratio <= 0.01:
        warnings.append("pred_unique_ratio<=0.01")
    if n >= 100 and pred_unique_count <= 5:
        warnings.append("pred_unique_count<=5")

    return {
        "n_samples": n,
        "pred_unique_count": pred_unique_count,
        "pred_unique_ratio": pred_unique_ratio,
        "top1_pred_text": top1_text,
        "top1_pred_count": top1_count,
        "top1_pred_ratio": top1_pred_ratio,
        "top10_pred_texts_with_count": top10,
        "top20_pred_texts_with_count": top20,
        "mean_pred_len_tokens": round(mean_pred_len_tokens, 4),
        "mean_ref_len_tokens": round(mean_ref_len_tokens, 4),
        "collapse_warnings": warnings,
    }


def log_sample_id_alignment_snippet(
    rows: Sequence[Dict[str, Any]], k: int = 20, logger=None
) -> None:
    """Sanity：抽样打印 sample_id / 评分 / 文本前缀。"""
    lines = []
    n = min(k, len(rows))
    for i in range(n):
        r = rows[i]
        sid = r.get("sample_id", "")
        pr = r.get("pred_rating", "")
        gr = r.get("gt_rating", "")
        pt = (r.get("pred_text", "") or "")[:80]
        rt = (r.get("ref_text", "") or "")[:80]
        lines.append(f"  [{sid}] pred_r={pr:.4g} gt_r={gr:.4g} | pred={pt!r} | ref={rt!r}")
    msg = "[Eval sanity] sample_id 对齐抽样 (前 %d 条):\n" % n + "\n".join(lines)
    if logger:
        logger.info(msg)
    else:
        print(msg, flush=True)
