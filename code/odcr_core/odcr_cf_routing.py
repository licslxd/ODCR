from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd


_EMPTY_EVIDENCE_TOKENS = {"", "none", "nan", "null", "unknown"}


def _tokens(text: str) -> set[str]:
    cleaned = (
        str(text)
        .lower()
        .replace("|", " ")
        .replace(";", " ")
        .replace(":", " ")
        .replace("=", " ")
        .replace(",", " ")
    )
    return {t for t in cleaned.split() if t and t not in _EMPTY_EVIDENCE_TOKENS}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return float(len(a & b) / max(1, len(a | b)))


def _unit(value: object, default: float = 0.0) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(v):
        return float(default)
    return float(max(0.0, min(1.0, v)))


def _finite_float(value: object, default: float = float("nan")) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float(default)
    return float(v) if np.isfinite(v) else float(default)


def _series(df: pd.DataFrame, name: str, default: object) -> pd.Series:
    if name in df.columns:
        return df[name]
    return pd.Series([default] * len(df))


def _require_live_rcr_diagnostics(df: pd.DataFrame) -> None:
    required = ("shared_latent_similarity", "specific_latent_shift", "rating_delta")
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise ValueError(
            "Step4 RCR routing requires live latent/rating diagnostics and refuses proxy fallback: "
            + ", ".join(missing)
        )
    nonfinite: list[str] = []
    for name in required:
        values = pd.to_numeric(df[name], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float, copy=False)).all():
            nonfinite.append(name)
    if nonfinite:
        raise ValueError(
            "Step4 RCR routing requires finite live latent/rating diagnostics and refuses proxy fallback: "
            + ", ".join(nonfinite)
        )


def _evidence_strength(text: str) -> float:
    toks = _tokens(text)
    if not toks:
        return 0.0
    return max(0.0, min(1.0, len(toks) / 8.0))


def _text_quality_score(text: str, entropy_value: float) -> float:
    toks = str(text).split()
    n = len(toks)
    if n <= 0:
        return 0.0
    length_floor = min(1.0, n / 6.0)
    length_ceiling = 1.0 if n <= 45 else max(0.35, 45.0 / float(n))
    unique_ratio = len(set(t.lower() for t in toks)) / max(1, n)
    entropy_conf = float(np.exp(-max(0.0, entropy_value)))
    return _unit(0.35 * length_floor + 0.20 * length_ceiling + 0.25 * unique_ratio + 0.20 * entropy_conf)


def _default_rcr_mapping() -> dict[str, Any]:
    return {
        "cf_reliability_weights": {
            "content_retention": 0.36,
            "rating_stability": 0.30,
            "style_shift": 0.22,
            "text_quality": 0.12,
        },
        "uncertainty_weights": {
            "rating_instability": 0.45,
            "content_weakness": 0.28,
            "text_quality_weakness": 0.17,
            "entropy": 0.10,
        },
        "rating_delta_soft_cap": 1.00,
        "route_scorer": {
            "min_reliability": 0.68,
            "min_content_retention": 0.62,
            "min_rating_stability": 0.62,
            "max_uncertainty": 0.38,
            "max_rating_delta": 0.50,
            "min_text_quality": 0.40,
        },
        "route_explainer": {
            "min_reliability": 0.55,
            "relaxed_min_reliability": 0.48,
            "min_content_retention": 0.42,
            "min_style_shift": 0.50,
            "max_uncertainty": 0.62,
            "min_text_quality": 0.40,
        },
        "confidence_bucket": {
            "high": {
                "bucket": 2,
                "min_reliability": 0.75,
                "max_uncertainty": 0.28,
                "min_rating_stability": 0.70,
                "min_content_retention": 0.65,
            },
            "medium": {
                "bucket": 1,
                "min_reliability": 0.50,
                "max_uncertainty": 0.58,
                "min_content_retention": 0.40,
            },
            "low_bucket": 0,
        },
        "train_keep": {
            "reject_when_both_routes_zero": True,
            "reject_reason": "rcr_route_reject",
        },
        "sample_weight_hint": {
            "scorer_route_multiplier": 1.0,
            "explainer_only_route_multiplier": 0.72,
            "reliability_floor": 0.35,
            "reliability_scale": 0.65,
            "uncertainty_base": 0.75,
            "uncertainty_scale": 0.25,
        },
        "export": {
            "required_fields": [
                "train_keep",
                "sample_weight_hint",
                "route_scorer",
                "route_explainer",
                "route_reason_scorer",
                "route_reason_explainer",
                "content_retention_score",
                "style_shift_score",
                "rating_stability_score",
                "cf_reliability_score",
                "uncertainty_score",
                "entropy_score",
                "text_quality_score",
                "confidence_bucket",
                "preprocess_route_scorer_prior",
                "preprocess_route_explainer_prior",
            ],
        },
    }


def _merge_rcr_mapping(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    out = {k: (dict(v) if isinstance(v, Mapping) else v) for k, v in base.items()}
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            merged = dict(out[key])
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, Mapping) and isinstance(merged.get(sub_key), Mapping):
                    nested = dict(merged[sub_key])
                    nested.update(dict(sub_value))
                    merged[sub_key] = nested
                else:
                    merged[sub_key] = sub_value
            out[key] = merged
        else:
            out[key] = value
    return out


@dataclass(init=False)
class ODCFRoutingConfig:
    """Step4 RCR config. Defaults are only available through explicit test helpers."""

    cf_reliability_weights: dict[str, float] = field(
        default_factory=lambda: {
            "content_retention": 0.36,
            "rating_stability": 0.30,
            "style_shift": 0.22,
            "text_quality": 0.12,
        }
    )
    uncertainty_weights: dict[str, float] = field(
        default_factory=lambda: {
            "rating_instability": 0.45,
            "content_weakness": 0.28,
            "text_quality_weakness": 0.17,
            "entropy": 0.10,
        }
    )
    rating_delta_soft_cap: float = 1.00
    route_scorer: dict[str, float] = field(
        default_factory=lambda: {
            "min_reliability": 0.68,
            "min_content_retention": 0.62,
            "min_rating_stability": 0.62,
            "max_uncertainty": 0.38,
            "max_rating_delta": 0.50,
            "min_text_quality": 0.40,
        }
    )
    route_explainer: dict[str, float] = field(
        default_factory=lambda: {
            "min_reliability": 0.55,
            "relaxed_min_reliability": 0.48,
            "min_content_retention": 0.42,
            "min_style_shift": 0.50,
            "max_uncertainty": 0.62,
            "min_text_quality": 0.40,
        }
    )
    confidence_bucket: dict[str, Any] = field(
        default_factory=lambda: {
            "high": {
                "bucket": 2,
                "min_reliability": 0.75,
                "max_uncertainty": 0.28,
                "min_rating_stability": 0.70,
                "min_content_retention": 0.65,
            },
            "medium": {
                "bucket": 1,
                "min_reliability": 0.50,
                "max_uncertainty": 0.58,
                "min_content_retention": 0.40,
            },
            "low_bucket": 0,
        }
    )
    train_keep: dict[str, Any] = field(
        default_factory=lambda: {
            "reject_when_both_routes_zero": True,
            "reject_reason": "rcr_route_reject",
        }
    )
    sample_weight_hint: dict[str, float] = field(
        default_factory=lambda: {
            "scorer_route_multiplier": 1.0,
            "explainer_only_route_multiplier": 0.72,
            "reliability_floor": 0.35,
            "reliability_scale": 0.65,
            "uncertainty_base": 0.75,
            "uncertainty_scale": 0.25,
        }
    )
    export: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        *,
        cf_reliability_weights: Mapping[str, float] | None = None,
        uncertainty_weights: Mapping[str, float] | None = None,
        rating_delta_soft_cap: float | None = None,
        route_scorer: Mapping[str, float] | None = None,
        route_explainer: Mapping[str, float] | None = None,
        confidence_bucket: Mapping[str, Any] | None = None,
        train_keep: Mapping[str, Any] | None = None,
        sample_weight_hint: Mapping[str, float] | None = None,
        export: Mapping[str, Any] | None = None,
        allow_test_defaults: bool = False,
    ) -> None:
        supplied = {
            "cf_reliability_weights": cf_reliability_weights,
            "uncertainty_weights": uncertainty_weights,
            "rating_delta_soft_cap": rating_delta_soft_cap,
            "route_scorer": route_scorer,
            "route_explainer": route_explainer,
            "confidence_bucket": confidence_bucket,
            "train_keep": train_keep,
            "sample_weight_hint": sample_weight_hint,
            "export": export,
        }
        if allow_test_defaults:
            values = _merge_rcr_mapping(_default_rcr_mapping(), {k: v for k, v in supplied.items() if v is not None})
        else:
            missing = [k for k, v in supplied.items() if v is None]
            if missing:
                raise RuntimeError(
                    "ODCFRoutingConfig active construction requires resolved step4.rcr fields; "
                    f"missing={missing}. Use ODCFRoutingConfig.for_test_default() only in tests."
                )
            values = {k: v for k, v in supplied.items() if v is not None}
        self.cf_reliability_weights = dict(values["cf_reliability_weights"])
        self.uncertainty_weights = dict(values["uncertainty_weights"])
        self.rating_delta_soft_cap = float(values["rating_delta_soft_cap"])
        self.route_scorer = dict(values["route_scorer"])
        self.route_explainer = dict(values["route_explainer"])
        self.confidence_bucket = dict(values["confidence_bucket"])
        self.train_keep = dict(values["train_keep"])
        self.sample_weight_hint = dict(values["sample_weight_hint"])
        self.export = dict(values["export"])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def for_test_default(cls) -> "ODCFRoutingConfig":
        return cls(allow_test_defaults=True)

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any] | None,
        *,
        allow_test_defaults: bool = False,
    ) -> "ODCFRoutingConfig":
        if not raw:
            if allow_test_defaults:
                return cls.for_test_default()
            raise RuntimeError("Step4 active RCR config mapping is required; defaults are test-only.")
        if allow_test_defaults:
            base = _merge_rcr_mapping(_default_rcr_mapping(), raw)
        else:
            required = set(_default_rcr_mapping())
            missing = sorted(required - set(raw))
            if missing:
                raise RuntimeError(
                    "Step4 active RCR config mapping is incomplete; missing="
                    f"{missing}. Use resolver output from configs/odcr.yaml."
                )
            base = dict(raw)
        return cls(**base)

    @classmethod
    def from_json(
        cls,
        raw: str,
        *,
        require: bool = True,
        allow_test_defaults: bool = False,
    ) -> "ODCFRoutingConfig":
        text = str(raw or "").strip()
        if not text:
            if allow_test_defaults:
                return cls.for_test_default()
            if require:
                raise RuntimeError("缺少 ODCR_STEP4_RCR_CONFIG_JSON；Step4 RCR 参数必须由 One-Control 注入。")
            raise RuntimeError("Step4 RCR defaults are test-only; pass allow_test_defaults=True explicitly.")
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"ODCR_STEP4_RCR_CONFIG_JSON 非法 JSON: {exc}") from exc
        if not isinstance(obj, Mapping):
            raise RuntimeError("ODCR_STEP4_RCR_CONFIG_JSON 根须为 object。")
        return cls.from_mapping(obj, allow_test_defaults=allow_test_defaults)

    @classmethod
    def from_env(cls, *, require: bool = True, allow_test_defaults: bool = False) -> "ODCFRoutingConfig":
        return cls.from_json(
            os.environ.get("ODCR_STEP4_RCR_CONFIG_JSON", ""),
            require=require,
            allow_test_defaults=allow_test_defaults,
        )


def _w(weights: Mapping[str, float], key: str) -> float:
    return float(weights[key])


def _cfg_float(obj: Mapping[str, Any], key: str) -> float:
    return float(obj[key])


def _confidence_bucket(row: Mapping[str, Any], conf: ODCFRoutingConfig) -> int:
    rel = _unit(row.get("cf_reliability_score"), 1.0)
    unc = _unit(row.get("uncertainty_score"), 0.0)
    rs = _unit(row.get("rating_stability_score"), 1.0)
    cr = _unit(row.get("content_retention_score"), 1.0)
    high = conf.confidence_bucket["high"]
    medium = conf.confidence_bucket["medium"]
    if (
        rel >= _cfg_float(high, "min_reliability")
        and unc <= _cfg_float(high, "max_uncertainty")
        and rs >= _cfg_float(high, "min_rating_stability")
        and cr >= _cfg_float(high, "min_content_retention")
    ):
        return int(high["bucket"])
    if (
        rel >= _cfg_float(medium, "min_reliability")
        and unc <= _cfg_float(medium, "max_uncertainty")
        and cr >= _cfg_float(medium, "min_content_retention")
    ):
        return int(medium["bucket"])
    return int(conf.confidence_bucket.get("low_bucket", 0))


def attach_odcr_cf_routing(
    target_df: pd.DataFrame,
    cf_df: pd.DataFrame,
    *,
    cfg: ODCFRoutingConfig | None = None,
) -> pd.DataFrame:
    if cfg is None:
        raise RuntimeError("attach_odcr_cf_routing requires resolved step4.rcr config; defaults are test-only.")
    conf = cfg
    merged = cf_df.copy()
    tgt = target_df.reset_index(drop=True)
    merged = merged.reset_index(drop=True)
    _require_live_rcr_diagnostics(merged)

    for prior_col, posterior_col in (
        ("preprocess_route_scorer_prior", "route_scorer"),
        ("preprocess_route_explainer_prior", "route_explainer"),
    ):
        if prior_col not in merged.columns:
            if posterior_col in merged.columns:
                raise ValueError(
                    f"Step4 input contains stale preprocess {posterior_col!r} without {prior_col!r}; "
                    "route_scorer / route_explainer are Step4 posterior-only."
                )
            merged[prior_col] = 0

    base_content = _series(tgt, "content_evidence", "").astype(str)
    base_style = (
        _series(tgt, "style_evidence", "").astype(str)
        + " "
        + _series(tgt, "domain_style_anchor", "").astype(str)
        + " "
        + _series(tgt, "local_style_residual_hint", "").astype(str)
    )
    entropy = _series(merged, "entropy", 0.0).astype(float)
    content_anchor = _series(tgt, "content_anchor_score", 0.0)
    style_anchor = _series(tgt, "style_anchor_score", 0.0)
    shared_sim_col = pd.to_numeric(merged["shared_latent_similarity"], errors="raise")
    specific_shift_col = pd.to_numeric(merged["specific_latent_shift"], errors="raise")
    rating_delta_col = pd.to_numeric(merged["rating_delta"], errors="raise")
    content_retention = []
    style_shift = []
    rating_stability = []
    reliability = []
    uncertainty = []
    entropy_score = []
    text_quality = []
    confidence_bucket = []
    route_scorer = []
    route_explainer = []
    reason_scorer = []
    reason_explainer = []

    for i in range(len(merged)):
        cf_text = str(merged.loc[i, "explanation"])
        cf_tok = _tokens(cf_text)
        base_content_tok = _tokens(base_content.iloc[i] if i < len(base_content) else "")
        base_style_tok = _tokens(base_style.iloc[i] if i < len(base_style) else "")
        entropy_raw = max(0.0, _finite_float(entropy.iloc[i], 0.0))
        entropy_u = _unit(1.0 - float(np.exp(-entropy_raw)))
        c_anchor = _unit(content_anchor.iloc[i] if i < len(content_anchor) else 0.0)
        s_anchor = _unit(style_anchor.iloc[i] if i < len(style_anchor) else 0.0)
        content_text_alignment = _jaccard(cf_tok, base_content_tok)
        style_text_shift = 1.0 - _jaccard(cf_tok, base_style_tok)

        shared_sim = _finite_float(shared_sim_col.iloc[i], np.nan)
        shared_sim = _unit(shared_sim, 0.0)

        specific_shift = _finite_float(specific_shift_col.iloc[i], np.nan)
        specific_shift = _unit(specific_shift, 0.0)

        cr = _unit(0.58 * shared_sim + 0.24 * c_anchor + 0.12 * _evidence_strength(base_content.iloc[i]) + 0.06 * content_text_alignment)
        ss = _unit(0.62 * specific_shift + 0.20 * s_anchor + 0.10 * _evidence_strength(base_style.iloc[i]) + 0.08 * style_text_shift)

        rating_delta = abs(_finite_float(rating_delta_col.iloc[i], np.nan))
        rs = _unit(1.0 - min(rating_delta / max(conf.rating_delta_soft_cap, 1e-6), 1.0))

        tq = _text_quality_score(cf_text, entropy_raw)
        rel = _unit(
            _w(conf.cf_reliability_weights, "content_retention") * cr
            + _w(conf.cf_reliability_weights, "rating_stability") * rs
            + _w(conf.cf_reliability_weights, "style_shift") * ss
            + _w(conf.cf_reliability_weights, "text_quality") * tq
        )
        unc = _unit(
            _w(conf.uncertainty_weights, "rating_instability") * (1.0 - rs)
            + _w(conf.uncertainty_weights, "content_weakness") * (1.0 - cr)
            + _w(conf.uncertainty_weights, "text_quality_weakness") * (1.0 - tq)
            + _w(conf.uncertainty_weights, "entropy") * entropy_u
        )
        scorer_cfg = conf.route_scorer
        explainer_cfg = conf.route_explainer
        rating_delta_ok = rating_delta <= _cfg_float(scorer_cfg, "max_rating_delta")
        s_route = int(
            rel >= _cfg_float(scorer_cfg, "min_reliability")
            and cr >= _cfg_float(scorer_cfg, "min_content_retention")
            and rs >= _cfg_float(scorer_cfg, "min_rating_stability")
            and unc <= _cfg_float(scorer_cfg, "max_uncertainty")
            and rating_delta_ok
            and tq >= _cfg_float(scorer_cfg, "min_text_quality")
        )
        e_route = int(
            tq >= _cfg_float(explainer_cfg, "min_text_quality")
            and cr >= _cfg_float(explainer_cfg, "min_content_retention")
            and unc <= _cfg_float(explainer_cfg, "max_uncertainty")
            and (
                rel >= _cfg_float(explainer_cfg, "min_reliability")
                or (
                    ss >= _cfg_float(explainer_cfg, "min_style_shift")
                    and rel >= _cfg_float(explainer_cfg, "relaxed_min_reliability")
                )
            )
        )
        bucket = _confidence_bucket(
            {
                "cf_reliability_score": rel,
                "uncertainty_score": unc,
                "rating_stability_score": rs,
                "content_retention_score": cr,
            },
            conf,
        )
        content_retention.append(round(cr, 4))
        style_shift.append(round(ss, 4))
        rating_stability.append(round(rs, 4))
        reliability.append(round(rel, 4))
        uncertainty.append(round(unc, 4))
        entropy_score.append(round(entropy_u, 4))
        text_quality.append(round(tq, 4))
        confidence_bucket.append(int(bucket))
        route_scorer.append(s_route)
        route_explainer.append(e_route)
        if s_route:
            reason_scorer.append("rcr_scorer_clean")
        elif not rating_delta_ok:
            reason_scorer.append("rating_delta_gt_0.5")
        elif rs < _cfg_float(scorer_cfg, "min_rating_stability"):
            reason_scorer.append("rating_stability_low")
        elif cr < _cfg_float(scorer_cfg, "min_content_retention"):
            reason_scorer.append("content_retention_low")
        elif unc > _cfg_float(scorer_cfg, "max_uncertainty"):
            reason_scorer.append("posterior_uncertainty_high")
        else:
            reason_scorer.append("reliability_below_scorer")

        if e_route:
            reason_explainer.append("rcr_explainer_rich")
        elif tq < _cfg_float(explainer_cfg, "min_text_quality"):
            reason_explainer.append("text_quality_low")
        elif cr < _cfg_float(explainer_cfg, "min_content_retention"):
            reason_explainer.append("content_retention_too_low")
        elif unc > _cfg_float(explainer_cfg, "max_uncertainty"):
            reason_explainer.append("posterior_uncertainty_high")
        else:
            reason_explainer.append("reliability_or_style_shift_low")

    merged["content_retention_score"] = content_retention
    merged["style_shift_score"] = style_shift
    merged["rating_stability_score"] = rating_stability
    merged["cf_reliability_score"] = reliability
    merged["uncertainty_score"] = uncertainty
    merged["entropy_score"] = entropy_score
    merged["text_quality_score"] = text_quality
    merged["confidence_bucket"] = confidence_bucket
    merged["route_scorer"] = route_scorer
    merged["route_explainer"] = route_explainer
    merged["route_reason_scorer"] = reason_scorer
    merged["route_reason_explainer"] = reason_explainer
    if "train_keep" not in merged.columns:
        merged["train_keep"] = 1
    if "sample_weight_hint" not in merged.columns:
        merged["sample_weight_hint"] = 1.0
    return merged
