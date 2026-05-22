"""Step4-owned quality scoring for Step5 gold and CF pools.

The scoring functions in this module intentionally use deterministic proxies.
They do not quota-split rows, call Step5 losses, or read hidden environment
controls.  Thresholds and weights come from the resolved One-Control payload.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd


GOLD_QUALITY_SCHEMA_VERSION = "odcr_gold_quality_score/1"
CF_TIER_SCHEMA_VERSION = "odcr_cf_quality_tier/1"

GOLD_PROXY_COLUMNS: tuple[str, ...] = (
    "gold_text_quality_proxy",
    "gold_consistency_proxy",
    "gold_uncertainty_proxy",
    "gold_control_coverage_proxy",
    "gold_evidence_alignment_proxy",
    "gold_coverage_diversity_proxy",
)


def _num(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name not in df.columns:
        return pd.Series(float(default), index=df.index, dtype="float64")
    return pd.to_numeric(df[name], errors="coerce").fillna(float(default)).astype(float)


def _text(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df.columns:
        return pd.Series("", index=df.index, dtype="string")
    return df[name].fillna("").astype(str)


def _flag(df: pd.DataFrame, name: str) -> pd.Series:
    return _num(df, name, 0.0).astype(int) == 1


def _clip01(value: pd.Series | np.ndarray | float) -> pd.Series:
    if isinstance(value, pd.Series):
        return value.clip(lower=0.0, upper=1.0).astype(float)
    return pd.Series(np.clip(value, 0.0, 1.0), dtype="float64")


def repeat_bigram_ratio(text: Any) -> float:
    words = [w for w in str(text or "").lower().split() if w]
    if len(words) < 4:
        return 0.0
    grams = list(zip(words, words[1:]))
    if not grams:
        return 0.0
    return float(1.0 - (len(set(grams)) / max(len(grams), 1)))


def rating_bucket(rating: pd.Series) -> pd.Series:
    val = pd.to_numeric(rating, errors="coerce")
    return pd.cut(
        val,
        bins=[-np.inf, 2.0, 3.0, 4.0, np.inf],
        labels=["low_rating", "mid_low_rating", "mid_high_rating", "high_rating"],
    ).astype("string").fillna("invalid_rating")


def length_bucket(words: pd.Series) -> pd.Series:
    val = pd.to_numeric(words, errors="coerce")
    return pd.cut(
        val,
        bins=[-np.inf, 2, 8, 24, 64, 160, np.inf],
        labels=["empty_or_tiny", "short", "medium", "long", "very_long", "extreme_long"],
    ).astype("string").fillna("unknown_len")


def index_bucket(series: pd.Series, prefix: str, *, head_cutoff: int, mid_cutoff: int) -> pd.Series:
    val = pd.to_numeric(series, errors="coerce").fillna(-1).astype(int)
    return pd.Series(
        np.select(
            [val < 0, val < int(head_cutoff), val < int(mid_cutoff)],
            [f"{prefix}_invalid", f"{prefix}_head", f"{prefix}_mid"],
            default=f"{prefix}_tail",
        ),
        index=series.index,
    )


def _bucket_score(bucket: pd.Series, mapping: Mapping[str, float], default: float = 1.0) -> pd.Series:
    return bucket.astype(str).map(lambda x: float(mapping.get(x, default))).astype(float)


def _nonempty_score(df: pd.DataFrame, cols: tuple[str, ...]) -> pd.Series:
    if not cols:
        return pd.Series(1.0, index=df.index, dtype="float64")
    present = pd.Series(0.0, index=df.index, dtype="float64")
    for col in cols:
        present += _text(df, col).str.strip().ne("").astype(float)
    return present / float(len(cols))


def _anchor_score(df: pd.DataFrame, cols: tuple[str, ...]) -> pd.Series:
    if not cols:
        return pd.Series(1.0, index=df.index, dtype="float64")
    total = pd.Series(0.0, index=df.index, dtype="float64")
    for col in cols:
        total += _clip01(_num(df, col, 0.0))
    return total / float(len(cols))


def _append_flag(base: pd.Series, mask: pd.Series, label: str) -> pd.Series:
    mask = mask.fillna(False).astype(bool)
    out = base.astype(str)
    current = out.loc[mask]
    out.loc[mask] = np.where(current.eq("none"), label, current + "|" + label)
    return out


def _length_quality(words: pd.Series) -> pd.Series:
    val = pd.to_numeric(words, errors="coerce").fillna(0).astype(float)
    return pd.Series(
        np.select(
            [
                val < 3,
                val < 8,
                val < 16,
                val <= 120,
                val <= 240,
                val <= 420,
                val <= 800,
            ],
            [0.0, 0.45, 0.72, 1.0, 0.82, 0.65, 0.45],
            default=0.25,
        ),
        index=words.index,
        dtype="float64",
    )


def default_gold_quality_config(raw: Mapping[str, Any] | None = None) -> dict[str, Any]:
    cfg = dict(raw or {})
    cfg.setdefault("schema_version", GOLD_QUALITY_SCHEMA_VERSION)
    cfg.setdefault("high_min_score", 0.955)
    cfg.setdefault("medium_min_score", 0.45)
    cfg.setdefault("hard_reject_min_words", 3)
    cfg.setdefault("hard_reject_max_words", 1400)
    cfg.setdefault("max_repeat_ngram_ratio", 0.62)
    cfg.setdefault("good_repeat_ngram_ratio", 0.35)
    cfg.setdefault("user_head_cutoff", 10_000)
    cfg.setdefault("user_mid_cutoff", 100_000)
    cfg.setdefault("item_head_cutoff", 10_000)
    cfg.setdefault("item_mid_cutoff", 100_000)
    cfg.setdefault(
        "proxy_weights",
        {
            "text_quality": 0.24,
            "consistency": 0.20,
            "uncertainty": 0.14,
            "control_coverage": 0.14,
            "evidence_alignment": 0.16,
            "coverage_diversity": 0.12,
        },
    )
    cfg.setdefault(
        "sampling_weight",
        {"high": 1.0, "medium": 0.55, "reject": 0.0, "aux_domain_multiplier": 0.90},
    )
    cfg.setdefault(
        "sanity",
        {
            "medium_min_ratio": 0.05,
            "high_max_ratio": 0.80,
            "reject_warn_ratio": 0.40,
            "target_high_range": [0.25, 0.45],
            "target_medium_range": [0.45, 0.65],
            "target_reject_range": [0.05, 0.15],
            "aux_high_range": [0.15, 0.35],
            "aux_medium_range": [0.45, 0.70],
            "aux_reject_range": [0.10, 0.25],
        },
    )
    return cfg


def default_cf_tier_config(raw: Mapping[str, Any] | None = None) -> dict[str, Any]:
    cfg = dict(raw or {})
    cfg.setdefault("schema_version", CF_TIER_SCHEMA_VERSION)
    cfg.setdefault(
        "hard_reject",
        {
            "max_uncertainty": 0.90,
            "min_text_quality": 0.20,
            "min_rating_stability": 0.45,
            "min_content_retention": 0.45,
            "min_words": 3,
            "max_repeat_ngram_ratio": 0.72,
        },
    )
    cfg.setdefault(
        "rating_stability_control",
        {
            "high": {
                "require_route_scorer": True,
                "min_confidence_bucket": 1,
                "min_rating_stability": 0.95,
                "min_content_retention": 0.94,
                "max_uncertainty": 0.06,
                "min_text_quality": 0.93,
                "min_reliability": 0.80,
            },
            "medium": {
                "min_confidence_bucket": 1,
                "min_rating_stability": 0.86,
                "min_content_retention": 0.82,
                "max_uncertainty": 0.12,
                "min_text_quality": 0.75,
                "min_reliability": 0.65,
            },
            "low_weighted": {
                "min_rating_stability": 0.65,
                "min_content_retention": 0.60,
                "max_uncertainty": 0.35,
                "min_text_quality": 0.55,
                "min_reliability": 0.45,
            },
        },
    )
    cfg.setdefault(
        "step5_explanation",
        {
            "high": {
                "require_route_explainer": True,
                "min_confidence_bucket": 1,
                "min_style_shift": 0.28,
                "min_reliability": 0.80,
                "max_uncertainty": 0.06,
                "min_text_quality": 0.93,
            },
            "medium": {
                "min_confidence_bucket": 1,
                "min_style_shift": 0.24,
                "min_reliability": 0.74,
                "max_uncertainty": 0.10,
                "min_text_quality": 0.82,
            },
            "low_weighted": {
                "min_style_shift": 0.10,
                "min_reliability": 0.55,
                "max_uncertainty": 0.35,
                "min_text_quality": 0.55,
            },
        },
    )
    cfg.setdefault(
        "sampling_weight",
        {
            "rating_stability_control": {"high": 1.20, "medium": 0.80, "low_weighted": 0.20, "reject": 0.0},
            "step5_explanation": {"high": 1.20, "medium": 0.90, "low_weighted": 0.30, "reject": 0.0},
        },
    )
    return cfg


def score_gold_quality(chunk: pd.DataFrame, cfg: Mapping[str, Any] | None = None) -> pd.DataFrame:
    cfg = default_gold_quality_config(cfg)
    out = chunk.copy()
    clean = _text(out, "clean_text")
    words = clean.str.split().map(len).astype(int)
    repeat = clean.map(repeat_bigram_ratio).astype(float)
    rating = _num(out, "rating", np.nan)
    origin = _text(out, "sample_origin")
    domain = _text(out, "domain")
    text_quality_score = _clip01(_num(out, "text_quality_score", 0.0))
    uncertainty = _clip01(_num(out, "uncertainty_score", 0.0))
    confidence = _clip01(_num(out, "confidence_bucket", 1.0) / 2.0)
    sample_weight = _clip01(_num(out, "sample_weight_hint", 0.0))
    user_idx = _num(out, "user_idx_global", -1.0)
    item_idx = _num(out, "item_idx_global", -1.0)

    out["explanation_length_words"] = words
    out["ngram_repeat_ratio"] = repeat
    out["rating_bucket"] = rating_bucket(out["rating"])
    out["length_bucket"] = length_bucket(words)
    out["explanation_length_bucket"] = out["length_bucket"]
    out["user_coverage_bucket"] = index_bucket(
        out["user_idx_global"],
        "user",
        head_cutoff=int(cfg["user_head_cutoff"]),
        mid_cutoff=int(cfg["user_mid_cutoff"]),
    )
    out["item_coverage_bucket"] = index_bucket(
        out["item_idx_global"],
        "item",
        head_cutoff=int(cfg["item_head_cutoff"]),
        mid_cutoff=int(cfg["item_mid_cutoff"]),
    )
    out["coverage_bucket"] = (
        out["user_coverage_bucket"].astype(str)
        + "|"
        + out["item_coverage_bucket"].astype(str)
        + "|"
        + out["length_bucket"].astype(str)
    )
    out["domain_role"] = np.select(
        [origin.eq("target_gold"), origin.eq("aux_gold"), origin.eq("aux_cf")],
        ["target_gold_supervision", "aux_gold_migration_anchor", "aux_cf_counterfactual"],
        default="unknown",
    )

    hard_bad_text = (
        _flag(out, "bad_tail_hit")
        | _flag(out, "template_hard_drop_hit")
        | _flag(out, "short_fragment_hit")
        | _flag(out, "repeat_tail_hit")
    )
    soft_noise = _flag(out, "template_hit") | _flag(out, "template_downweighted") | _flag(out, "noisy_tail_downweighted")
    valid_origin_domain = ((origin == "target_gold") & (domain == "target")) | (
        (origin == "aux_gold") & (domain == "auxiliary")
    )
    nonempty = clean.str.strip().ne("")
    rating_legal = rating.between(1.0, 5.0).fillna(False)
    min_words = int(cfg["hard_reject_min_words"])
    max_words = int(cfg["hard_reject_max_words"])
    hard_reject = (
        (~nonempty)
        | (words < min_words)
        | (words > max_words)
        | (~rating_legal)
        | user_idx.isna()
        | item_idx.isna()
        | (user_idx < 0)
        | (item_idx < 0)
        | (repeat > float(cfg["max_repeat_ngram_ratio"]))
        | hard_bad_text
        | (~valid_origin_domain)
    )
    hard_flags = pd.Series("none", index=out.index, dtype="object")
    for mask, label in (
        (~nonempty, "empty_text"),
        (words < min_words, "too_short"),
        (words > max_words, "too_long"),
        (~rating_legal, "invalid_rating"),
        ((user_idx < 0) | (item_idx < 0) | user_idx.isna() | item_idx.isna(), "missing_user_item"),
        (repeat > float(cfg["max_repeat_ngram_ratio"]), "repetitive_ngram"),
        (_flag(out, "bad_tail_hit"), "bad_tail"),
        (_flag(out, "template_hard_drop_hit"), "template_hard_drop"),
        (_flag(out, "short_fragment_hit"), "short_fragment"),
        (_flag(out, "repeat_tail_hit"), "repeat_tail"),
        (~valid_origin_domain, "origin_domain_mismatch"),
    ):
        hard_flags = _append_flag(hard_flags, mask, label)

    text_proxy = (
        0.18 * nonempty.astype(float)
        + 0.28 * _length_quality(words)
        + 0.18 * _clip01(1.0 - (repeat / max(float(cfg["good_repeat_ngram_ratio"]), 1e-6)))
        + 0.16 * (~hard_bad_text).astype(float)
        + 0.08 * (~soft_noise).astype(float)
        + 0.12 * text_quality_score
    )

    lower = clean.str.lower()
    positive_hit = lower.str.contains(r"\b(?:great|excellent|love|loved|perfect|amazing|best|wonderful|enjoyed)\b", regex=True)
    negative_hit = lower.str.contains(r"\b(?:bad|terrible|awful|hate|hated|worst|boring|poor|waste|disappoint)\b", regex=True)
    sentiment_conflict = ((rating >= 4.0) & negative_hit & ~positive_hit) | ((rating <= 2.0) & positive_hit & ~negative_hit)
    extreme_has_evidence = (~((rating >= 4.5) | (rating <= 1.5))) | (words >= 12)
    consistency_proxy = (
        0.24 * rating_legal.astype(float)
        + 0.20 * extreme_has_evidence.fillna(False).astype(float)
        + 0.20 * (~sentiment_conflict.fillna(False)).astype(float)
        + 0.16 * valid_origin_domain.astype(float)
        + 0.20 * sample_weight
    )

    completeness = _nonempty_score(
        out,
        (
            "content_evidence",
            "style_evidence",
            "domain_style_anchor",
            "local_style_residual_hint",
            "polarity_anchor",
        ),
    )
    uncertainty_proxy = (
        0.40 * (1.0 - uncertainty)
        + 0.25 * completeness
        + 0.20 * (~(hard_bad_text | soft_noise)).astype(float)
        + 0.15 * confidence
    )
    control_proxy = (
        0.55 * completeness
        + 0.25 * _anchor_score(out, ("content_anchor_score", "style_anchor_score"))
        + 0.20 * valid_origin_domain.astype(float)
    )
    evidence_proxy = (
        0.28 * _clip01(_num(out, "content_anchor_score", 0.0))
        + 0.22 * _clip01(_num(out, "style_anchor_score", 0.0))
        + 0.22 * _clip01(_num(out, "evidence_quality_prior", 0.0))
        + 0.18 * valid_origin_domain.astype(float)
        + 0.10 * np.where(origin.eq("aux_gold"), 0.85, 1.0)
    )
    user_score = _bucket_score(
        out["user_coverage_bucket"],
        {"user_invalid": 0.0, "user_head": 0.55, "user_mid": 0.85, "user_tail": 1.0},
    )
    item_score = _bucket_score(
        out["item_coverage_bucket"],
        {"item_invalid": 0.0, "item_head": 0.65, "item_mid": 0.90, "item_tail": 1.0},
    )
    rating_score = _bucket_score(
        out["rating_bucket"],
        {"invalid_rating": 0.0, "low_rating": 0.88, "mid_low_rating": 1.0, "mid_high_rating": 1.0, "high_rating": 0.82},
    )
    len_score = _bucket_score(
        out["length_bucket"],
        {
            "empty_or_tiny": 0.0,
            "short": 0.70,
            "medium": 0.92,
            "long": 1.0,
            "very_long": 0.82,
            "extreme_long": 0.62,
        },
    )
    diversity_proxy = 0.34 * user_score + 0.24 * item_score + 0.20 * rating_score + 0.22 * len_score

    weights = dict(cfg["proxy_weights"])
    score = (
        float(weights["text_quality"]) * text_proxy
        + float(weights["consistency"]) * consistency_proxy
        + float(weights["uncertainty"]) * uncertainty_proxy
        + float(weights["control_coverage"]) * control_proxy
        + float(weights["evidence_alignment"]) * evidence_proxy
        + float(weights["coverage_diversity"]) * diversity_proxy
    )
    score = _clip01(score)
    high = (~hard_reject) & (score >= float(cfg["high_min_score"]))
    medium = (~hard_reject) & (~high) & (score >= float(cfg["medium_min_score"]))
    out["gold_quality_score"] = score.astype(float)
    out["gold_anchor_quality"] = out["gold_quality_score"]
    out["gold_quality_tier"] = np.select(
        [high.to_numpy(), medium.to_numpy(), hard_reject.to_numpy()],
        ["high", "medium", "reject"],
        default="reject",
    )
    reason = pd.Series("low_score", index=out.index, dtype="object")
    reason.loc[high] = "high_gold_quality_score"
    reason.loc[medium] = "medium_gold_quality_score"
    reason.loc[hard_reject] = hard_flags.loc[hard_reject]
    out["gold_quality_reasons"] = reason
    out["gold_quality_reject_reason"] = reason
    out["hard_reject_flags"] = hard_flags
    out["gold_text_quality_proxy"] = _clip01(text_proxy)
    out["gold_consistency_proxy"] = _clip01(consistency_proxy)
    out["gold_uncertainty_proxy"] = _clip01(uncertainty_proxy)
    out["gold_control_coverage_proxy"] = _clip01(control_proxy)
    out["gold_evidence_alignment_proxy"] = _clip01(evidence_proxy)
    out["gold_coverage_diversity_proxy"] = _clip01(diversity_proxy)

    weight_cfg = dict(cfg["sampling_weight"])
    base_weight = out["gold_quality_tier"].astype(str).map(
        {
            "high": float(weight_cfg["high"]),
            "medium": float(weight_cfg["medium"]),
            "reject": float(weight_cfg["reject"]),
        }
    ).fillna(0.0)
    aux_mult = float(weight_cfg.get("aux_domain_multiplier", 0.90))
    out["recommended_sampling_weight"] = (base_weight * np.where(origin.eq("aux_gold"), aux_mult, 1.0)).astype(float)
    return out


def assign_cf_tiers(chunk: pd.DataFrame, cfg: Mapping[str, Any] | None = None) -> pd.DataFrame:
    cfg = default_cf_tier_config(cfg)
    out = chunk.copy()
    clean = _text(out, "clean_text")
    words = clean.str.split().map(len).astype(int)
    if "ngram_repeat_ratio" in out.columns:
        repeat = _num(out, "ngram_repeat_ratio", 0.0)
    else:
        repeat = clean.map(repeat_bigram_ratio).astype(float)
        out["ngram_repeat_ratio"] = repeat
    text_quality = _clip01(_num(out, "text_quality_score", 0.0))
    unc = _clip01(_num(out, "uncertainty_score", 1.0))
    content = _clip01(_num(out, "content_retention_score", 0.0))
    stability = _clip01(_num(out, "rating_stability_score", 0.0))
    style = _clip01(_num(out, "style_shift_score", 0.0))
    reliability = _clip01(_num(out, "cf_reliability_score", 0.0))
    confidence = _num(out, "confidence_bucket", 0.0)
    rs = _num(out, "route_scorer", 0.0).astype(int) == 1
    re = _num(out, "route_explainer", 0.0).astype(int) == 1

    reject_cfg = dict(cfg["hard_reject"])
    hard_bad = (
        clean.str.strip().eq("")
        | (words < int(reject_cfg["min_words"]))
        | (repeat > float(reject_cfg["max_repeat_ngram_ratio"]))
        | _flag(out, "bad_tail_hit")
        | _flag(out, "template_hard_drop_hit")
        | _flag(out, "short_fragment_hit")
        | (unc >= float(reject_cfg["max_uncertainty"]))
        | (text_quality < float(reject_cfg["min_text_quality"]))
        | (stability < float(reject_cfg["min_rating_stability"]))
        | (content < float(reject_cfg["min_content_retention"]))
    )

    def _tier(head: str, route: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        head_cfg = dict(cfg[head])
        high_cfg = dict(head_cfg["high"])
        medium_cfg = dict(head_cfg["medium"])
        low_cfg = dict(head_cfg["low_weighted"])
        if head == "rating_stability_control":
            high = (
                (~hard_bad)
                & route
                & (confidence >= float(high_cfg["min_confidence_bucket"]))
                & (stability >= float(high_cfg["min_rating_stability"]))
                & (content >= float(high_cfg["min_content_retention"]))
                & (unc <= float(high_cfg["max_uncertainty"]))
                & (text_quality >= float(high_cfg["min_text_quality"]))
                & (reliability >= float(high_cfg["min_reliability"]))
            )
            medium = (
                (~hard_bad)
                & (~high)
                & (route | (confidence >= float(medium_cfg["min_confidence_bucket"])))
                & (stability >= float(medium_cfg["min_rating_stability"]))
                & (content >= float(medium_cfg["min_content_retention"]))
                & (unc <= float(medium_cfg["max_uncertainty"]))
                & (text_quality >= float(medium_cfg["min_text_quality"]))
                & (reliability >= float(medium_cfg["min_reliability"]))
            )
            low = (
                (~hard_bad)
                & (~high)
                & (~medium)
                & (stability >= float(low_cfg["min_rating_stability"]))
                & (content >= float(low_cfg["min_content_retention"]))
                & (unc <= float(low_cfg["max_uncertainty"]))
                & (text_quality >= float(low_cfg["min_text_quality"]))
                & (reliability >= float(low_cfg["min_reliability"]))
            )
            score = (
                0.26 * stability
                + 0.24 * content
                + 0.18 * reliability
                + 0.14 * text_quality
                + 0.10 * (1.0 - unc)
                + 0.08 * route.astype(float)
            )
        else:
            high = (
                (~hard_bad)
                & route
                & (confidence >= float(high_cfg["min_confidence_bucket"]))
                & (style >= float(high_cfg["min_style_shift"]))
                & (reliability >= float(high_cfg["min_reliability"]))
                & (unc <= float(high_cfg["max_uncertainty"]))
                & (text_quality >= float(high_cfg["min_text_quality"]))
            )
            medium = (
                (~hard_bad)
                & (~high)
                & (route | (confidence >= float(medium_cfg["min_confidence_bucket"])))
                & (style >= float(medium_cfg["min_style_shift"]))
                & (reliability >= float(medium_cfg["min_reliability"]))
                & (unc <= float(medium_cfg["max_uncertainty"]))
                & (text_quality >= float(medium_cfg["min_text_quality"]))
            )
            low = (
                (~hard_bad)
                & (~high)
                & (~medium)
                & (style >= float(low_cfg["min_style_shift"]))
                & (reliability >= float(low_cfg["min_reliability"]))
                & (unc <= float(low_cfg["max_uncertainty"]))
                & (text_quality >= float(low_cfg["min_text_quality"]))
            )
            score = (
                0.28 * style
                + 0.24 * reliability
                + 0.16 * text_quality
                + 0.14 * (1.0 - unc)
                + 0.10 * route.astype(float)
                + 0.08 * (confidence / 2.0).clip(0.0, 1.0)
            )
        return _clip01(score), high.fillna(False), medium.fillna(False), low.fillna(False)

    score_a, high_a, med_a, low_a = _tier("rating_stability_control", rs)
    score_b, high_b, med_b, low_b = _tier("step5_explanation", re)
    hard = hard_bad.fillna(False).astype(bool)
    out["cf_quality_score_rating_stability_control"] = score_a.astype(float)
    out["cf_quality_score_step5_explanation"] = score_b.astype(float)
    out["cf_tier_rating_stability_control"] = np.select(
        [hard.to_numpy(), high_a.to_numpy(), med_a.to_numpy(), low_a.to_numpy()],
        ["reject", "high", "medium", "low_weighted"],
        default="reject",
    )
    out["cf_tier_step5_explanation"] = np.select(
        [hard.to_numpy(), high_b.to_numpy(), med_b.to_numpy(), low_b.to_numpy()],
        ["reject", "high", "medium", "low_weighted"],
        default="reject",
    )
    out["cf_tier_reason_rating_stability_control"] = np.select(
        [hard.to_numpy(), high_a.to_numpy(), med_a.to_numpy(), low_a.to_numpy()],
        ["hard_quality_reject", "scorer_high_quality_route", "scorer_medium_quality", "scorer_low_weighted_quality"],
        default="scorer_reject_low_quality",
    )
    out["cf_tier_reason_step5_explanation"] = np.select(
        [hard.to_numpy(), high_b.to_numpy(), med_b.to_numpy(), low_b.to_numpy()],
        ["hard_quality_reject", "explainer_high_quality_route", "explainer_medium_quality", "explainer_low_weighted_quality"],
        default="explainer_reject_low_quality",
    )
    weights = dict(cfg["sampling_weight"])
    for head, col in (("rating_stability_control", "cf_tier_rating_stability_control"), ("step5_explanation", "cf_tier_step5_explanation")):
        mapping = {k: float(v) for k, v in dict(weights[head]).items()}
        out[f"recommended_sampling_weight_{head}"] = out[col].astype(str).map(mapping).fillna(0.0).astype(float)
    out["cf_tier"] = ""
    out["cf_tier_reason"] = ""
    return out


def add_step5_quality_columns(
    chunk: pd.DataFrame,
    *,
    gold_quality_config: Mapping[str, Any] | None = None,
    cf_tier_config: Mapping[str, Any] | None = None,
    pool_contract_version: str = "",
) -> pd.DataFrame:
    out = score_gold_quality(chunk, gold_quality_config)
    out = assign_cf_tiers(out, cf_tier_config)
    out["step5_pool_contract_version"] = str(pool_contract_version)
    return out


__all__ = [
    "CF_TIER_SCHEMA_VERSION",
    "GOLD_PROXY_COLUMNS",
    "GOLD_QUALITY_SCHEMA_VERSION",
    "add_step5_quality_columns",
    "assign_cf_tiers",
    "default_cf_tier_config",
    "default_gold_quality_config",
    "index_bucket",
    "length_bucket",
    "rating_bucket",
    "repeat_bigram_ratio",
    "score_gold_quality",
]
