"""Phase 2：rule-based 候选重排序（与 decode fingerprint 解耦，配置写入 metrics 顶层）。

Rerank 为辅助路径，非主表 canonical mainline 默认；template penalty（generic_template）默认权重为 0 且
短路计算路径，避免残留数值风险。主线评估与选模以主生成 + 主表指标为准。
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from odcr_eval_dirty_text import dirty_penalty_score, per_text_dirty_rule_hits
from odcr_eval_metrics import _tokenize_safe, unigram_repetition_ratio

# rule_v2：未闭合尾巴 / 畸形 token 的固定罚分系数（命中即扣满额；由 CLI 覆盖传入 engine）
DEFAULT_MALFORMED_TAIL_COEF = 0.15
DEFAULT_MALFORMED_TOKEN_COEF = 0.18

_TAIL_STOPWORDS = frozenset(
    {
        "and",
        "or",
        "but",
        "the",
        "a",
        "an",
        "to",
        "of",
        "with",
        "for",
        "on",
        "in",
        "at",
        "from",
        "by",
    }
)

_RE_ALNUM_MESS = re.compile(
    r"(?:[a-zA-Z]+\d+[a-zA-Z0-9]{4,}|\d+[a-zA-Z]{2,}\d+[a-zA-Z0-9]*|[a-zA-Z0-9]{12,})"
)
_RE_WORD_ODD_PUNCT = re.compile(r"\b\w+[^\w\s\'\"-]\w+\b")
_RE_RUN_PUNCT = re.compile(r"([^\w\s])\1{2,}")
_RE_HTMLISH = re.compile(
    r"</[a-zA-Z][^>]{0,40}>|<[a-zA-Z][^>]{0,40}|\b&\s*#\d{1,8};|\b&[a-z]{2,12};|&#x[0-9a-fA-F]{1,8};",
    re.I,
)
_RE_PAREN_GLUE = re.compile(r"\b\w{2,}\)\b|\b\w+\s+\(\s*$|\([a-z]{3,}\w*\b|\w+\)\w+")
_RE_SYMBOL_GLUE = re.compile(r"[\)\]\}\"]\s*[\(\[\{\"]|[\w][\)\]\}][\w]")


def _malformed_tail_inspect(pred: str) -> Tuple[float, List[str]]:
    """返回 (0 或 1 的命中强度, 原因列表)。强度供可选加权，当前 engine 用二元 + 固定系数。"""
    s = str(pred) if pred is not None else ""
    reasons: List[str] = []
    t = s.rstrip()
    if not t:
        return 0.0, reasons
    last = t[-1]
    if last in "(['\"":
        reasons.append(f"unclosed_tail_char:{last!r}")
    if last in "-:/":
        reasons.append(f"trailing_punct:{last!r}")
    toks = _tokenize_safe(t)
    if toks:
        lw = toks[-1].lower().strip("\"'")  # noqa: B005
        if lw in _TAIL_STOPWORDS:
            reasons.append(f"tail_stopword:{lw}")
        if len(lw) == 1 and lw.isalpha():
            reasons.append("tail_single_letter_token")
        if lw.endswith("-") or (len(lw) >= 2 and lw[-1] == "-" and lw[-2].isalnum()):
            reasons.append("tail_hyphen_fragment")
    hit = 1.0 if reasons else 0.0
    return hit, reasons


def _malformed_token_inspect(pred: str) -> Tuple[float, List[str]]:
    s = str(pred) if pred is not None else ""
    reasons: List[str] = []
    if not s:
        return 0.0, reasons
    if _RE_PAREN_GLUE.search(s):
        reasons.append("paren_glue_or_split")
    if _RE_ALNUM_MESS.search(s):
        reasons.append("suspicious_alnum_token")
    if _RE_RUN_PUNCT.search(s):
        reasons.append("repeated_punct_run")
    if _RE_HTMLISH.search(s):
        reasons.append("html_or_entity_fragment")
    if _RE_WORD_ODD_PUNCT.search(s):
        reasons.append("intraword_odd_punct")
    if _RE_SYMBOL_GLUE.search(s):
        reasons.append("symbol_glue")
    hit = 1.0 if reasons else 0.0
    return hit, reasons


def build_dirty_detail_v2(pred: str, *, ref_mean_len_words: Optional[float] = None) -> Dict[str, Any]:
    hits = per_text_dirty_rule_hits(pred, ref_mean_len_words=ref_mean_len_words)
    active = [k for k, v in hits.items() if v]
    return {"rule_hits": hits, "active_rules": active}


def _repeated_ngram_ratio(tokens: Sequence[str], n: int) -> float:
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    if not grams:
        return 0.0
    c = Counter(grams)
    rep = sum(1 for g, k in c.items() if k > 1)
    return float(rep) / float(len(grams))


def _repeat_fragment_hit(s: str) -> bool:
    t = (s or "").replace(" ", "")
    if len(t) < 24:
        return False
    t = t[:400]
    n = len(t)
    for L in (12, 16, 20):
        if L > n // 2:
            break
        for i in range(0, min(n - L, 120) + 1):
            frag = t[i : i + L]
            if t.count(frag) >= 2:
                return True
    return False


def extract_rerank_features(
    candidate_text: str,
    reference_text: str,
    *,
    avg_logprob: float,
    ref_mean_len_words: Optional[float] = None,
) -> Dict[str, Any]:
    """单条候选的 rerank 特征（均可导出）。"""
    pred = str(candidate_text) if candidate_text is not None else ""
    ref = str(reference_text) if reference_text is not None else ""
    ptoks = _tokenize_safe(pred)
    rtoks = _tokenize_safe(ref)
    pred_len_words = max(len(ptoks), 1)
    ref_len_words = max(len(rtoks), 1)
    pred_len_ratio = float(pred_len_words) / float(ref_len_words)

    uni_rep = float(unigram_repetition_ratio(pred))
    bi_rep = _repeated_ngram_ratio(ptoks, 2)
    tri_rep = _repeated_ngram_ratio(ptoks, 3)
    frag_hit = _repeat_fragment_hit(pred)
    repeat_penalty = float(
        0.35 * uni_rep + 0.30 * bi_rep + 0.25 * tri_rep + 0.10 * (1.0 if frag_hit else 0.0)
    )

    dirty = float(dirty_penalty_score(pred, ref_mean_len_words=ref_mean_len_words))
    tail_hit_f, tail_rs = _malformed_tail_inspect(pred)
    tok_hit_f, tok_rs = _malformed_token_inspect(pred)
    ddv2 = build_dirty_detail_v2(pred, ref_mean_len_words=ref_mean_len_words)

    return {
        "avg_logprob": float(avg_logprob),
        "pred_len_words": int(pred_len_words),
        "pred_len_ratio": round(pred_len_ratio, 6),
        "repeat_penalty": round(repeat_penalty, 6),
        "dirty_penalty": round(dirty, 6),
        "repeat_detail": {
            "unigram_repetition_ratio": round(uni_rep, 6),
            "bigram_repeat_ratio": round(bi_rep, 6),
            "trigram_repeat_ratio": round(tri_rep, 6),
            "repeat_fragment_hit": frag_hit,
        },
        "malformed_tail_hit": bool(tail_hit_f > 0),
        "malformed_tail_reasons": tail_rs,
        "malformed_token_hit": bool(tok_hit_f > 0),
        "malformed_token_reasons": tok_rs,
        "dirty_detail_v2": ddv2,
    }


def length_deviation_penalty(pred_len_ratio: float, *, target_ratio: float) -> float:
    return float(abs(float(pred_len_ratio) - float(target_ratio)))


def score_candidates_rule_v1(
    features: Mapping[str, Any],
    *,
    weight_logprob: float,
    weight_length: float,
    weight_repeat: float,
    weight_dirty: float,
    target_len_ratio: float,
) -> Tuple[float, float]:
    """
    rule_v1:
      score = w_lp * avg_logprob
            - w_len * |pred_len_ratio - target|
            - w_rep * repeat_penalty
            - w_dirty * dirty_penalty
    返回 (rerank_score, length_deviation_penalty)。
    """
    lp = float(features["avg_logprob"])
    plr = float(features["pred_len_ratio"])
    rep = float(features["repeat_penalty"])
    dty = float(features["dirty_penalty"])
    len_pen = length_deviation_penalty(plr, target_ratio=target_len_ratio)
    score = (
        float(weight_logprob) * lp
        - float(weight_length) * len_pen
        - float(weight_repeat) * rep
        - float(weight_dirty) * dty
    )
    return float(score), float(len_pen)


def score_candidates_rule_v2(
    features: Mapping[str, Any],
    *,
    weight_logprob: float,
    weight_length: float,
    weight_repeat: float,
    weight_dirty: float,
    target_len_ratio: float,
    coef_malformed_tail: float,
    coef_malformed_token: float,
) -> Tuple[float, float, Dict[str, float]]:
    """
    rule_v2:
      score = w_lp*lp - w_len*len_dev - w_rep*rep - w_dirty*dirty
              - (tail_hit ? coef_tail : 0) - (token_hit ? coef_token : 0)
    """
    lp = float(features["avg_logprob"])
    plr = float(features["pred_len_ratio"])
    rep = float(features["repeat_penalty"])
    dty = float(features["dirty_penalty"])
    len_pen = length_deviation_penalty(plr, target_ratio=target_len_ratio)
    tail_hit = bool(features.get("malformed_tail_hit"))
    tok_hit = bool(features.get("malformed_token_hit"))
    tail_applied = float(coef_malformed_tail) if tail_hit else 0.0
    tok_applied = float(coef_malformed_token) if tok_hit else 0.0
    score = (
        float(weight_logprob) * lp
        - float(weight_length) * len_pen
        - float(weight_repeat) * rep
        - float(weight_dirty) * dty
        - tail_applied
        - tok_applied
    )
    breakdown = {
        "logprob_term": float(weight_logprob) * lp,
        "length_term": -float(weight_length) * len_pen,
        "repeat_term": -float(weight_repeat) * rep,
        "dirty_term": -float(weight_dirty) * dty,
        "malformed_tail_term": -tail_applied,
        "malformed_token_term": -tok_applied,
        "final_rerank_score": float(score),
    }
    return float(score), float(len_pen), breakdown


def rouge_l_proxy(candidate: str, reference: str) -> float:
    """轻量 proxy：按词 bigram F1 近似，仅用于 summary 诊断列。"""
    ct = _tokenize_safe(candidate.lower())
    rt = _tokenize_safe(reference.lower())
    if len(ct) < 2 or len(rt) < 2:
        return 0.0
    bg_c = Counter(tuple(ct[i : i + 2]) for i in range(len(ct) - 1))
    bg_r = Counter(tuple(rt[i : i + 2]) for i in range(len(rt) - 1))
    overlap = sum(min(bg_c[g], bg_r[g]) for g in bg_c)
    if overlap <= 0:
        return 0.0
    prec = overlap / max(len(ct) - 1, 1)
    rec = overlap / max(len(rt) - 1, 1)
    if prec + rec <= 0:
        return 0.0
    return round(2.0 * prec * rec / (prec + rec), 6)


def build_rerank_weights_dict(
    *,
    weight_logprob: float,
    weight_length: float,
    weight_repeat: float,
    weight_dirty: float,
) -> Dict[str, float]:
    return {
        "logprob": float(weight_logprob),
        "length": float(weight_length),
        "repeat": float(weight_repeat),
        "dirty": float(weight_dirty),
    }


RERANK_V3_SCHEMA_VERSION = "odcr_rerank_v3/1.0"

_RE_GENERIC_TEMPLATES = re.compile(
    r"(?i)\b(great|good|nice|awesome|amazing)\s+(movie|film|product|item|book)\b|"
    r"\bi\s+(really\s+)?(enjoyed|liked|loved)\s+it\b|"
    r"\bhighly\s+recommend\b|\bworth\s+(watching|buying|reading)\b|"
    r"\bnot\s+bad\b|\bpretty\s+good\b"
)


def keywords_from_source_text(source: str, *, max_kw: int = 48) -> List[str]:
    """从 review 等源文本抽轻量关键词（字母数字长度>=4），不读 reference explanation。"""
    s = str(source or "")
    toks = _tokenize_safe(s)
    out: List[str] = []
    seen = set()
    for t in toks:
        w = t.strip("\"'").lower()
        if len(w) < 4 or not any(c.isalnum() for c in w):
            continue
        if w in seen:
            continue
        seen.add(w)
        out.append(w)
        if len(out) >= max_kw:
            break
    return out


def lexical_coverage_score(pred: str, keywords: Sequence[str]) -> float:
    if not keywords:
        return 0.5
    pt = set(t.lower() for t in _tokenize_safe(pred))
    hit = sum(1 for k in keywords if k.lower() in pt)
    return float(hit) / float(len(keywords))


def generic_template_penalty_score(pred: str) -> float:
    s = str(pred or "")
    if len(s.strip()) < 8:
        return 0.0
    return 1.0 if _RE_GENERIC_TEMPLATES.search(s) else 0.0


def completion_score_basic(pred: str) -> float:
    t = (pred or "").rstrip()
    if not t:
        return 0.0
    last = t[-1]
    if last in ".!?":
        return 1.0
    if last in ",:;":
        return 0.55
    return 0.35


def well_formed_score_v3(
    *,
    malformed_tail_hit: bool,
    malformed_token_hit: bool,
    tail_reasons: Sequence[str],
    tok_reasons: Sequence[str],
) -> float:
    base = 1.0
    if malformed_tail_hit:
        base -= 0.35
    if malformed_token_hit:
        base -= 0.30
    if len(tok_reasons) >= 3:
        base -= 0.15
    return max(0.0, min(1.0, base))


def length_penalty_v2_gaussian(n_words: int, *, target: float, sigma: float) -> float:
    """宽容高斯长度罚（越大越差）。"""
    d = float(n_words) - float(target)
    s = max(float(sigma), 1e-6)
    return float((d / s) ** 2)


def repeat_penalty_v2_from_features(feats: Mapping[str, Any]) -> float:
    v = feats.get("repeat_penalty", 0.0)
    if v is None:
        return 0.0
    return float(v)


DEFAULT_RERANK_V3_PROFILE: Dict[str, Any] = {
    "schema_version": RERANK_V3_SCHEMA_VERSION,
    "hard_min_pred_words": 2,
    "hard_max_malformed_token_reasons": 4,
    "hard_min_source_coverage": 0.0,
    "length_target_words": 18.0,
    "length_sigma_words": 10.0,
    "w_lp_norm": 1.0,
    "w_completion": 0.55,
    "w_well_formed": 0.45,
    "w_source_coverage": 0.5,
    "w_repeat_v2": 0.35,
    "w_malformed_tail": 0.4,
    "w_malformed_token": 0.45,
    "w_generic_template": 0.0,
    "debug_generic_template_penalty": False,
    "w_entity_drift": 0.4,
    "w_length_v2": 0.12,
}


def merge_rerank_v3_profile(overlay: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    m = dict(DEFAULT_RERANK_V3_PROFILE)
    if overlay:
        for k, v in overlay.items():
            if v is not None:
                m[k] = v
    return m


def v3_hard_filter(
    pred: str,
    feats: Mapping[str, Any],
    source_cov: float,
    profile: Mapping[str, Any],
    *,
    w_generic_template: Optional[float] = None,
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    plw = int(feats.get("pred_len_words") or 0)
    if plw < int(profile.get("hard_min_pred_words", 2)):
        reasons.append("too_short")
    tok_rs = list(feats.get("malformed_token_reasons") or [])
    if len(tok_rs) >= int(profile.get("hard_max_malformed_token_reasons", 4)):
        reasons.append("severe_malformed_token")
    if source_cov < float(profile.get("hard_min_source_coverage", 0.0)):
        reasons.append("severe_source_mismatch")
    wg = (
        float(w_generic_template)
        if w_generic_template is not None
        else float(profile.get("w_generic_template", 0.0))
    )
    if wg > 0.0:
        g = generic_template_penalty_score(pred)
        if g > 0 and plw <= 6:
            reasons.append("generic_junk_short")
    return (len(reasons) == 0), reasons


def score_candidates_rule_v3(
    pred: str,
    feats: Mapping[str, Any],
    *,
    review_keywords: Sequence[str],
    lp_norm: float,
    profile: Optional[Mapping[str, Any]] = None,
) -> Tuple[bool, List[str], float, Dict[str, float]]:
    prof = merge_rerank_v3_profile(profile)
    src_cov = lexical_coverage_score(pred, review_keywords)
    drift = max(0.0, 1.0 - src_cov)
    w_gt = float(prof["w_generic_template"])
    debug_gt = bool(prof.get("debug_generic_template_penalty", False))
    ok, hard_rs = v3_hard_filter(pred, feats, src_cov, prof, w_generic_template=w_gt)
    comp = completion_score_basic(pred)
    wf = well_formed_score_v3(
        malformed_tail_hit=bool(feats.get("malformed_tail_hit")),
        malformed_token_hit=bool(feats.get("malformed_token_hit")),
        tail_reasons=list(feats.get("malformed_tail_reasons") or []),
        tok_reasons=list(feats.get("malformed_token_reasons") or []),
    )
    rep = repeat_penalty_v2_from_features(feats)
    mtail = 1.0 if feats.get("malformed_tail_hit") else 0.0
    mtok = 1.0 if feats.get("malformed_token_hit") else 0.0
    generic_soft_disabled = bool(w_gt <= 0.0 and not debug_gt)
    gen = 0.0 if generic_soft_disabled else float(generic_template_penalty_score(pred))
    n_words = max(int(feats.get("pred_len_words") or 1), 1)
    len_pen = length_penalty_v2_gaussian(
        n_words,
        target=float(prof["length_target_words"]),
        sigma=float(prof["length_sigma_words"]),
    )
    soft = (
        float(prof["w_lp_norm"]) * float(lp_norm)
        + float(prof["w_completion"]) * comp
        + float(prof["w_well_formed"]) * wf
        + float(prof["w_source_coverage"]) * src_cov
        - float(prof["w_repeat_v2"]) * rep
        - float(prof["w_malformed_tail"]) * mtail
        - float(prof["w_malformed_token"]) * mtok
        - (0.0 if generic_soft_disabled else w_gt * gen)
        - float(prof["w_entity_drift"]) * drift
        - float(prof["w_length_v2"]) * len_pen
    )
    if not ok:
        soft = -1e6 + float(lp_norm) * 0.01
    breakdown = {
        "lp_norm": round(float(lp_norm), 6),
        "completion_score": round(comp, 6),
        "well_formed_score": round(wf, 6),
        "source_coverage_score": round(src_cov, 6),
        "entity_drift_penalty": round(drift, 6),
        "repeat_penalty_v2": round(rep, 6),
        "malformed_tail_penalty": round(mtail, 6),
        "malformed_token_penalty": round(mtok, 6),
        "generic_template_penalty_raw": (None if generic_soft_disabled else round(gen, 6)),
        "generic_template_weight_effective": round(w_gt, 6),
        "generic_template_soft_term": (0.0 if generic_soft_disabled else round(-w_gt * gen, 6)),
        "generic_template_soft_disabled": bool(generic_soft_disabled),
        "length_penalty_v2": round(len_pen, 6),
        "soft_score_pre_gate": round(float(soft if ok else -1e6), 6),
        "final_rerank_score": round(float(soft), 6),
    }
    return ok, hard_rs, float(soft), breakdown


def extract_rerank_features_for_v3(
    candidate_text: str,
    *,
    avg_logprob: float,
    token_len: int,
    ref_mean_len_words: Optional[float] = None,
) -> Dict[str, Any]:
    """v3 用特征：重复/脏/畸形等；长度字段不依赖 reference explanation。"""
    pred = str(candidate_text) if candidate_text is not None else ""
    ptoks = _tokenize_safe(pred)
    pred_len_words = max(len(ptoks), 1)
    uni_rep = float(unigram_repetition_ratio(pred))
    bi_rep = _repeated_ngram_ratio(ptoks, 2)
    tri_rep = _repeated_ngram_ratio(ptoks, 3)
    frag_hit = _repeat_fragment_hit(pred)
    repeat_penalty = float(
        0.35 * uni_rep + 0.30 * bi_rep + 0.25 * tri_rep + 0.10 * (1.0 if frag_hit else 0.0)
    )
    dirty = float(dirty_penalty_score(pred, ref_mean_len_words=ref_mean_len_words))
    tail_hit_f, tail_rs = _malformed_tail_inspect(pred)
    tok_hit_f, tok_rs = _malformed_token_inspect(pred)
    ddv2 = build_dirty_detail_v2(pred, ref_mean_len_words=ref_mean_len_words)
    ref_anchor = float(ref_mean_len_words) if ref_mean_len_words is not None else float(pred_len_words)
    pred_len_ratio = float(pred_len_words) / max(ref_anchor, 1.0)
    return {
        "avg_logprob": float(avg_logprob),
        "pred_len_words": int(pred_len_words),
        "pred_len_ratio": round(pred_len_ratio, 6),
        "repeat_penalty": round(repeat_penalty, 6),
        "dirty_penalty": round(dirty, 6),
        "dirty_penalty_diagnostic_only": round(dirty, 6),
        "repeat_detail": {
            "unigram_repetition_ratio": round(uni_rep, 6),
            "bigram_repeat_ratio": round(bi_rep, 6),
            "trigram_repeat_ratio": round(tri_rep, 6),
            "repeat_fragment_hit": frag_hit,
        },
        "malformed_tail_hit": bool(tail_hit_f > 0),
        "malformed_tail_reasons": tail_rs,
        "malformed_token_hit": bool(tok_hit_f > 0),
        "malformed_token_reasons": tok_rs,
        "dirty_detail_v2": ddv2,
        "token_len_decode": int(max(1, token_len)),
    }


def compute_lp_norm(avg_logprob: float, token_len: int) -> float:
    """长度归一 log 似然：sum_lp / sqrt(T) ≈ avg * sqrt(T)。"""
    t = max(1, int(token_len))
    return float(avg_logprob) * float(math.sqrt(t))
