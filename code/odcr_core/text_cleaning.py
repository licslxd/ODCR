"""
Explanation 清洗与质量标注（Step4 导出 / Step5 审计）。

可单测、规则可审计：每类问题独立 flag，不单给总分。
"""
from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Set

# 英文解释尾部「像被截断」的功能词（轻度坏尾信号）
_FUNCTION_WORD_TAIL: Set[str] = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "if",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "as",
        "at",
        "by",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
    }
)

# bad_tail 中视为「严重」、质量乘子 0
_SEVERE_BAD_TAIL_TYPES: Set[str] = frozenset(
    {
        "unclosed_paren",
        "unclosed_bracket",
        "dangling_ampersand",
        "incomplete_entity_suffix",
    }
)

_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
# 明显 HTML 实体残留（清洗后仍可疑）
_ENTITY_LIKE_RE = re.compile(r"&(?:#\d+|#x[0-9a-fA-F]+|[A-Za-z][A-Za-z0-9]{0,31});")
_INCOMPLETE_ENTITY_TAIL_RE = re.compile(r"&[A-Za-z0-9#]{1,31}$")


@dataclass
class CleanResult:
    clean_text: str
    clean_changed: bool
    clean_failed: bool
    steps: Dict[str, Any] = field(default_factory=dict)


def clean_explanation_text(text: Optional[str]) -> CleanResult:
    """
    html.unescape、控制字符清理、空白归一化、不完整 entity 尾处理、首尾引号/括号修剪。
    """
    steps: Dict[str, Any] = {}
    if text is None or (isinstance(text, float) and str(text) == "nan"):
        return CleanResult("", True, True, {"reason": "null_or_nan"})
    raw = str(text)
    original_for_cmp = raw
    if not raw.strip():
        return CleanResult("", True, True, {"reason": "empty_input"})

    s = html.unescape(raw)
    steps["html_unescape"] = s != raw

    s2 = _CTRL_RE.sub(" ", s)
    steps["stripped_controls"] = s2 != s
    s = s2

    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s).strip()
    steps["ws_normalized"] = True

    # 不完整 HTML entity 尾：删掉从最后一个 & 起的残缺片段
    if re.search(r"&[^;\s]{0,48}$", s) and ";" not in s.split("&")[-1]:
        cut = s.rfind("&")
        if cut >= 0:
            s = s[:cut].rstrip()
            steps["dropped_incomplete_entity_tail"] = True

    s = _strip_outer_quotes(s)
    s = _trim_unmatched_edge_brackets(s)
    steps["trim_quotes_brackets"] = True

    s = re.sub(r"\s+", " ", s).strip()

    clean_failed = False
    if not s:
        clean_failed = True
    # 仍含裸 & 且不像合法实体
    if s.endswith("&") or _INCOMPLETE_ENTITY_TAIL_RE.search(s):
        clean_failed = True

    changed = s != original_for_cmp.strip() or bool(steps.get("html_unescape"))
    return CleanResult(
        clean_text=s,
        clean_changed=changed,
        clean_failed=clean_failed,
        steps=steps,
    )


def _strip_outer_quotes(s: str) -> str:
    t = s.strip()
    if len(t) >= 2 and t[0] == t[-1] and t[0] in "'\"":
        return t[1:-1].strip()
    return t


def _trim_unmatched_edge_brackets(s: str) -> str:
    t = s.strip()
    if not t:
        return t
    # 行首多余闭括号 / 行尾多余开括号（单字符级轻量修复）
    while t.startswith(")") or t.startswith("]"):
        t = t[1:].lstrip()
    while t.endswith("(") or t.endswith("["):
        t = t[:-1].rstrip()
    return t.strip()


def detect_bad_tail(text: str) -> Dict[str, Any]:
    """未闭合括号、尾部的 &、不完整 entity、功能词结尾等。"""
    t = (text or "").strip()
    out: Dict[str, Any] = {
        "bad_tail_hit": False,
        "bad_tail_types": [],
    }
    types: List[str] = []
    if not t:
        out["bad_tail_hit"] = False
        out["bad_tail_types"] = []
        return out

    o_paren = t.count("(") - t.count(")")
    o_brack = t.count("[") - t.count("]")
    if o_paren > 0:
        types.append("unclosed_paren")
    if o_brack > 0:
        types.append("unclosed_bracket")
    if t.endswith("&"):
        types.append("dangling_ampersand")
    if _INCOMPLETE_ENTITY_TAIL_RE.search(t):
        types.append("incomplete_entity_suffix")

    last = re.sub(r"[^\w\s]", "", t.split()[-1]).lower() if t.split() else ""
    if last and last in _FUNCTION_WORD_TAIL:
        types.append("ends_function_word")

    out["bad_tail_types"] = types
    out["bad_tail_hit"] = len(types) > 0
    out["bad_tail_severe"] = bool(set(types) & _SEVERE_BAD_TAIL_TYPES)
    return out


def detect_short_fragment(text: str, *, min_words: int = 4, min_chars: int = 12) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    words = [w for w in re.split(r"\s+", t) if w]
    return len(words) < min_words or len(t) < min_chars


def detect_repeat_tail_hit(text: str, *, n: int = 3, min_repeat: int = 2) -> bool:
    """尾部 n-gram 在句末明显重复（如 '... great album great album'）。"""
    t = (text or "").strip().lower()
    words = [w for w in re.split(r"\s+", t) if w]
    if len(words) < n * min_repeat:
        return False
    tail = tuple(words[-n:])
    # 在末尾 2*n*min_repeat 窗口内数出现次数
    window = words[-(n * min_repeat + n) :]
    if len(window) < n * min_repeat:
        return False
    cnt = 0
    for i in range(len(window) - n + 1):
        if tuple(window[i : i + n]) == tail:
            cnt += 1
    return cnt >= min_repeat


def detect_template_like(
    text: str,
    template_stats: Mapping[str, int],
    *,
    min_count: int = 3,
) -> Dict[str, Any]:
    """基于语料级频次：同一规范化串出现 >= min_count 视为模板句。"""
    key = _template_key(text)
    c = int(template_stats.get(key, 0))
    return {"template_hit": c >= min_count, "template_key": key, "template_count": c}


def _template_key(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def html_entity_hit_raw(text: Optional[str]) -> bool:
    """原始串是否出现 HTML 实体模式（含未闭合 &...）。"""
    if text is None:
        return False
    s = str(text)
    if "&" not in s:
        return False
    if _ENTITY_LIKE_RE.search(s):
        return True
    if re.search(r"&[A-Za-z0-9#]{1,48}", s):
        return True
    return False


def build_sample_quality_flags(
    *,
    raw_explanation: str,
    clean_result: CleanResult,
    template_stats: Mapping[str, int],
    template_min_count: int = 3,
) -> Dict[str, Any]:
    """汇总单行质量 flag + 质量乘子（不含来源权重）。"""
    clean = clean_result.clean_text
    bad = detect_bad_tail(clean)
    tmpl = detect_template_like(clean, template_stats, min_count=template_min_count)
    short = detect_short_fragment(clean)
    rep = detect_repeat_tail_hit(clean)
    html_hit = html_entity_hit_raw(raw_explanation)

    flags: Dict[str, Any] = {
        "clean_text": clean,
        "clean_changed": bool(clean_result.clean_changed),
        "clean_failed": bool(clean_result.clean_failed),
        "clean_steps": dict(clean_result.steps),
        "html_entity_hit": bool(html_hit),
        "bad_tail_hit": bool(bad["bad_tail_hit"]),
        "bad_tail_types": list(bad["bad_tail_types"]),
        "bad_tail_severe": bool(bad.get("bad_tail_severe", False)),
        "template_hit": bool(tmpl["template_hit"]),
        "template_count": int(tmpl["template_count"]),
        "short_fragment_hit": bool(short),
        "repeat_tail_hit": bool(rep),
        "ends_function_word": "ends_function_word" in bad["bad_tail_types"],
    }
    flags["functional_tail_hit"] = bool(flags["ends_function_word"])

    q = _quality_multiplier_from_flags(flags)
    flags["quality_tier"] = q["tier"]
    flags["quality_mult"] = q["mult"]
    return flags


def _quality_multiplier_from_flags(flags: Mapping[str, Any]) -> Dict[str, Any]:
    """严重直接 0；模板频次在 merge_flags_into_row 硬删/强降权；此处仅句面轻症。"""
    if flags.get("clean_failed") or not str(flags.get("clean_text", "")).strip():
        return {"tier": "severe", "mult": 0.0}
    if flags.get("bad_tail_severe"):
        return {"tier": "severe", "mult": 0.0}
    mild = (
        flags.get("short_fragment_hit")
        or flags.get("html_entity_hit")
        or flags.get("functional_tail_hit")
    )
    if mild:
        return {"tier": "mild", "mult": 0.55}
    return {"tier": "ok", "mult": 1.0}


def build_template_stats(clean_texts: Sequence[str]) -> Dict[str, int]:
    """语料级模板统计：规范化串 -> 出现次数。"""
    from collections import Counter

    c: Counter[str] = Counter()
    for t in clean_texts:
        c[_template_key(t)] += 1
    return dict(c)


def merge_flags_into_row(
    row: MutableMapping[str, Any],
    flags: Mapping[str, Any],
    *,
    sample_origin: str,
    is_cf: int,
    origin_weights: Mapping[str, float],
) -> None:
    """写入 CSV 列：质量 flag、train_keep、drop_reason、sample_weight_hint（硬删+强降权策略）。"""
    ow = float(origin_weights.get(sample_origin, 1.0))
    qm = float(flags["quality_mult"])
    hint = round(ow * qm, 6)
    tmpl_count = int(flags.get("template_count", 0) or 0)
    template_hard_drop_min_count = int(
        row.get("template_hard_drop_min_count", 0) or 0
    )
    row.pop("template_hard_drop_min_count", None)
    template_hard_drop_hit = bool(
        flags.get("template_hit")
        and template_hard_drop_min_count > 0
        and tmpl_count >= template_hard_drop_min_count
    )

    row["sample_origin"] = sample_origin
    row["is_counterfactual"] = int(is_cf)
    row["clean_text"] = flags["clean_text"]
    row["clean_changed"] = int(bool(flags["clean_changed"]))
    row["html_entity_hit"] = int(bool(flags["html_entity_hit"]))
    row["bad_tail_hit"] = int(bool(flags["bad_tail_hit"]))
    row["bad_tail_types"] = ";".join(flags["bad_tail_types"]) if flags["bad_tail_types"] else ""
    row["template_hit"] = int(bool(flags["template_hit"]))
    row["template_count"] = tmpl_count
    row["template_hard_drop_hit"] = int(template_hard_drop_hit)
    row["template_downweighted"] = 0
    row["noisy_tail_downweighted"] = 0
    row["short_fragment_hit"] = int(bool(flags["short_fragment_hit"]))
    row["repeat_tail_hit"] = int(bool(flags["repeat_tail_hit"]))
    row["train_drop_reason"] = ""
    row["train_keep"] = 1
    row["sample_weight_hint"] = hint

    if not str(flags["clean_text"]).strip():
        row["train_keep"] = 0
        row["train_drop_reason"] = "empty_clean_text"
        row["sample_weight_hint"] = 0.0
        return
    if flags.get("clean_failed"):
        row["train_keep"] = 0
        row["train_drop_reason"] = "malformed_tail"
        row["sample_weight_hint"] = 0.0
        return
    if flags.get("bad_tail_severe"):
        row["train_keep"] = 0
        row["train_drop_reason"] = "severe_bad_tail"
        row["sample_weight_hint"] = 0.0
        return
    # 严重模板：极高频一律删；高频非 target_gold 删
    if flags.get("template_hit"):
        if tmpl_count >= 22 or template_hard_drop_hit:
            row["train_keep"] = 0
            row["train_drop_reason"] = "severe_template"
            row["sample_weight_hint"] = 0.0
            return
        if tmpl_count >= 10 and sample_origin != "target_gold":
            row["train_keep"] = 0
            row["train_drop_reason"] = "severe_template"
            row["sample_weight_hint"] = 0.0
            return
        # 中度模板：强降权
        if tmpl_count >= 3:
            tmul = 0.42 if sample_origin == "target_gold" else 0.22
            row["sample_weight_hint"] = round(float(row["sample_weight_hint"]) * tmul, 6)
            row["template_downweighted"] = 1
    # 噪尾：重复尾或非严重 bad_tail
    noisy_tail = bool(flags.get("repeat_tail_hit")) or (
        bool(flags.get("bad_tail_hit")) and not bool(flags.get("bad_tail_severe"))
    )
    if noisy_tail:
        row["sample_weight_hint"] = round(float(row["sample_weight_hint"]) * 0.28, 6)
        row["noisy_tail_downweighted"] = 1
    # 轻症 generic 已在 qm；再压低 aux_cf，避免过度稀释 target_gold
    if sample_origin == "aux_cf":
        row["sample_weight_hint"] = round(float(row["sample_weight_hint"]) * 0.72, 6)
