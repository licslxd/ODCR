# -*- coding: utf-8 -*-
"""评测用脏文本启发式统计（Phase 1）；与排序/标红解耦，仅产出结构化计数。"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

_HTML_ENTITY_RE = re.compile(r"&#\d{1,8};|&#x[0-9a-fA-F]{1,8};|&[a-zA-Z][a-zA-Z0-9]{1,31};")
_NON_ALNUM_SPACE_RE = re.compile(r"[^\w\s]", re.UNICODE)
# 控制字符与明显乱码片段
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _word_len(s: str) -> int:
    return len(s.split())


def _truncated_quote_or_paren(s: str) -> bool:
    t = (s or "").strip()
    if len(t) < 2:
        return False
    if t[-1] in "\"'":
        dq = t.count('"')
        sq = t.count("'")
        return (dq % 2 == 1) or (sq % 2 == 1)
    if t[-1] in "([{":
        return True
    return False


def _illegal_symbol_density(s: str) -> float:
    if not s:
        return 0.0
    sym = len(_NON_ALNUM_SPACE_RE.findall(s))
    return sym / max(len(s), 1)


def _too_long_vs_ref(s: str, ref_mean: Optional[float]) -> bool:
    if ref_mean is None or ref_mean <= 0:
        return False
    return _word_len(s) > 2.0 * ref_mean


def _repeat_fragment(s: str, min_len: int = 12, cap: int = 400) -> bool:
    t = (s or "").replace(" ", "")
    if len(t) < min_len * 2:
        return False
    t = t[:cap]
    n = len(t)
    for L in (min_len, min_len + 4, min_len + 8):
        if L > n // 2:
            break
        for i in range(0, min(n - L, 120) + 1):
            frag = t[i : i + L]
            if t.count(frag) >= 2:
                return True
    return False


def _has_ctrl(s: str) -> bool:
    return bool(_CTRL_RE.search(s or ""))


def compute_dirty_text_stats(
    pred_texts: Sequence[str],
    *,
    ref_mean_len_words: Optional[float] = None,
) -> Dict[str, Any]:
    """
    对每条 pred 做多规则检测；hit_rate = 至少命中一条规则的样本占比。

    ref_mean_len_words: 语料参考句平均词长（与 text_metrics corpus mean_ref 一致），用于「过长句」规则。
    """
    texts = [str(x) if x is not None else "" for x in pred_texts]
    n = len(texts)
    if n == 0:
        return {
            "hit_rate": 0.0,
            "rules": {},
            "examples": [],
            "n_samples": 0,
        }

    rule_names = (
        "html_entity",
        "truncated_quote",
        "high_punct_density",
        "too_long_vs_ref",
        "repeat_fragment",
        "control_chars",
    )
    per_rule_hits = {r: 0 for r in rule_names}
    sample_hit_any = [False] * n
    examples: List[Dict[str, Any]] = []

    def _push_example(rule: str, idx: int, snippet: str) -> None:
        if len(examples) >= 10:
            return
        examples.append(
            {
                "rule": rule,
                "sample_index": idx,
                "text_preview": (snippet or "")[:240],
            }
        )

    for i, s in enumerate(texts):
        hit_here = False
        if _HTML_ENTITY_RE.search(s):
            per_rule_hits["html_entity"] += 1
            hit_here = True
            _push_example("html_entity", i, s)
        if _truncated_quote_or_paren(s):
            per_rule_hits["truncated_quote"] += 1
            hit_here = True
            _push_example("truncated_quote", i, s)
        if _illegal_symbol_density(s) >= 0.35:
            per_rule_hits["high_punct_density"] += 1
            hit_here = True
            _push_example("high_punct_density", i, s)
        if _too_long_vs_ref(s, ref_mean_len_words):
            per_rule_hits["too_long_vs_ref"] += 1
            hit_here = True
            _push_example("too_long_vs_ref", i, s)
        if _repeat_fragment(s):
            per_rule_hits["repeat_fragment"] += 1
            hit_here = True
            _push_example("repeat_fragment", i, s)
        if _has_ctrl(s):
            per_rule_hits["control_chars"] += 1
            hit_here = True
            _push_example("control_chars", i, s)
        sample_hit_any[i] = hit_here

    any_count = sum(1 for h in sample_hit_any if h)
    rules_out: Dict[str, Any] = {}
    for r in rule_names:
        c = per_rule_hits[r]
        rules_out[r] = {"count": c, "rate": round(c / n, 6)}

    return {
        "hit_rate": round(any_count / n, 6),
        "rules": rules_out,
        "examples": examples,
        "n_samples": n,
    }


def per_text_dirty_rule_hits(s: str, *, ref_mean_len_words: Optional[float] = None) -> Dict[str, bool]:
    """与 compute_dirty_text_stats 规则一致的单条布尔命中（供 rerank / 诊断复用）。"""
    t = str(s) if s is not None else ""
    return {
        "html_entity": bool(_HTML_ENTITY_RE.search(t)),
        "truncated_quote": _truncated_quote_or_paren(t),
        "high_punct_density": _illegal_symbol_density(t) >= 0.35,
        "too_long_vs_ref": _too_long_vs_ref(t, ref_mean_len_words),
        "repeat_fragment": _repeat_fragment(t),
        "control_chars": _has_ctrl(t),
    }


def dirty_penalty_score(s: str, *, ref_mean_len_words: Optional[float] = None) -> float:
    """非负惩罚，约 [0,1]：命中规则数 / 规则总数。"""
    hits = per_text_dirty_rule_hits(s, ref_mean_len_words=ref_mean_len_words)
    n = max(len(hits), 1)
    return float(sum(1 for v in hits.values() if v)) / float(n)
