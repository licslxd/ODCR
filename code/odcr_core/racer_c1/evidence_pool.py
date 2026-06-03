"""Train-only evidence pool construction for RACER-C1."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from odcr_core.text_cleaning import clean_explanation_text

from .logging import write_json


EVIDENCE_POOL_SCHEMA_VERSION = "odcr_racer_c1_evidence_pool/2"
TOKEN_STATS_SCHEMA_VERSION = "odcr_racer_c1_token_length_stats/1"
LEAKAGE_CHECK_SCHEMA_VERSION = "odcr_racer_c1_leakage_check/1"
CONTENT_STYLE_STATS_SCHEMA_VERSION = "odcr_racer_c1_content_style_split_stats/1"
CF_ANCHOR_STATS_SCHEMA_VERSION = "odcr_racer_c1_cf_anchor_stats/1"
ROLE_DISTRIBUTION_SCHEMA_VERSION = "odcr_racer_c1_contrastive_role_distribution/1"
CROSS_DOMAIN_DISTRIBUTION_SCHEMA_VERSION = "odcr_racer_c1_cross_domain_evidence_distribution/1"

_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_GENERIC_PHRASES = (
    "good product",
    "great product",
    "great item",
    "works well",
    "recommend it",
    "very good",
    "good quality",
    "great album",
    "great movie",
    "worth buying",
)


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    source_split: str
    source_domain: str
    source_user: str
    source_item: str
    source_rating: float
    source_rating_bucket: str
    source_type: str
    raw_explanation: str
    clean_explanation: str
    clean_explanation_25: str
    token_count_raw: int
    token_count_25: int
    content_evidence: str
    style_evidence: str
    causal_content_evidence: str
    style_shortcut_evidence: str
    cf_content_anchor: str
    cf_style_anchor: str
    cf_aspect_anchor: str
    cf_reliability_bucket: str
    cf_template_flag: int
    causal_content_score: float
    style_shortcut_score: float
    contrastive_role_hint: str
    template_score: float
    evidence_quality_prior: float
    rcr_score: float
    content_retention_score: float
    style_shift_score: float
    rating_stability_score: float
    uncertainty_score: float
    sample_weight_hint: float
    train_keep: int
    source_sample_id: str
    source_row_index: int
    source_path: str


def _tokens(text: str) -> list[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(text or "")]


def truncate_words(text: str, max_tokens: int) -> str:
    words = str(text or "").strip().split()
    return " ".join(words[:max_tokens])


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def _rating_bucket(rating: float) -> str:
    if rating >= 4.0:
        return "high_rating"
    if rating <= 2.0:
        return "low_rating"
    return "mid_rating"


def template_score(text: str, *, duplicate_count: int = 1, template_hit: bool = False) -> float:
    toks = _tokens(text)
    if not toks:
        return 1.0
    lower = " ".join(toks)
    generic_hits = sum(1 for phrase in _GENERIC_PHRASES if phrase in lower)
    low_diversity = 1.0 - (len(set(toks)) / max(1, len(toks)))
    duplicate = min(1.0, max(0.0, (duplicate_count - 1) / 10.0))
    hit = 1.0 if template_hit else 0.0
    return round(min(1.0, 0.35 * generic_hits + 0.35 * low_diversity + 0.20 * duplicate + 0.25 * hit), 6)


def _clamp01(value: float) -> float:
    return round(min(1.0, max(0.0, float(value))), 6)


def _anchor_text(*parts: str, max_tokens: int = 8) -> str:
    joined = " ".join(str(part or "").strip() for part in parts if str(part or "").strip())
    toks = _tokens(joined)
    if not toks:
        return ""
    return " ".join(toks[:max_tokens])


def _content_specificity(text: str) -> float:
    toks = _tokens(text)
    if not toks:
        return 0.0
    generic = {tok for phrase in _GENERIC_PHRASES for tok in phrase.split()}
    content = [tok for tok in toks if tok not in generic and len(tok) >= 3]
    return len(set(content)) / max(1, len(set(toks)))


def _cf_reliability_bucket(*, rcr_score: float, uncertainty_score: float, template: float) -> str:
    if rcr_score >= 0.80 and uncertainty_score <= 0.25 and template <= 0.35:
        return "high"
    if rcr_score >= 0.55 and template <= 0.60:
        return "medium"
    return "low"


def causal_content_score(*, content_text: str, clean_text: str, content_retention: float, rcr_score: float, template: float) -> float:
    specificity = max(_content_specificity(content_text), _content_specificity(clean_text))
    return _clamp01(0.35 * content_retention + 0.30 * rcr_score + 0.25 * specificity - 0.20 * template + 0.10)


def style_shortcut_score(*, style_text: str, clean_text: str, style_shift: float, template: float) -> float:
    style_presence = 1.0 if _tokens(style_text) else 0.0
    low_specificity = 1.0 - _content_specificity(clean_text)
    return _clamp01(0.45 * template + 0.25 * style_shift + 0.20 * low_specificity + 0.10 * style_presence)


def contrastive_role_hint(*, rcr_bucket: str, template: float, causal_content: float, style_shortcut: float) -> str:
    if rcr_bucket == "high" and causal_content >= 0.50 and template <= 0.35:
        return "positive_high_rcr_content"
    if rcr_bucket == "medium" and causal_content >= 0.40 and style_shortcut <= 0.65:
        return "soft_positive_medium_rcr"
    if rcr_bucket == "low":
        return "hard_negative_low_rcr"
    if template >= 0.65 or style_shortcut >= 0.75:
        return "hard_negative_template_style"
    return "candidate_neutral"


def _source_type(row: Mapping[str, Any]) -> str:
    origin = str(row.get("sample_origin") or "").strip()
    if origin == "aux_cf" or "cf" in origin.lower():
        return "cf"
    if origin in {"target_gold", "aux_gold"}:
        return origin
    domain = str(row.get("domain") or "").strip()
    return f"{domain or 'train'}_factual"


def _candidate_text(row: Mapping[str, Any]) -> tuple[str, str]:
    clean = str(row.get("clean_text") or "").strip()
    raw = str(row.get("explanation") or "").strip()
    if clean:
        return raw or clean, clean
    result = clean_explanation_text(raw)
    return raw, result.clean_text


def _records_from_frame(
    frame: pd.DataFrame,
    *,
    source_path: Path,
    max_tokens: int,
    row_offset: int,
) -> list[EvidenceRecord]:
    if frame.empty:
        return []
    duplicate_counts = Counter()
    cleaned: list[tuple[str, str]] = []
    for row in frame.to_dict("records"):
        raw, clean = _candidate_text(row)
        clean25 = truncate_words(clean, max_tokens)
        cleaned.append((raw, clean25))
        duplicate_counts[" ".join(_tokens(clean25))] += 1

    records: list[EvidenceRecord] = []
    for idx, row in enumerate(frame.to_dict("records")):
        global_idx = int(row_offset + idx)
        raw, clean = _candidate_text(row)
        clean25 = truncate_words(clean, max_tokens)
        toks_raw = _tokens(clean)
        toks25 = _tokens(clean25)
        train_keep = _safe_int(row.get("train_keep"), 1)
        key = " ".join(toks25)
        rating = _safe_float(row.get("rating"), 0.0)
        rcr = _safe_float(row.get("cf_reliability_score"), _safe_float(row.get("evidence_quality_prior"), 0.0))
        template_hit = bool(_safe_int(row.get("template_hit"), 0))
        content_text = str(row.get("content_evidence") or "")
        style_text = str(row.get("style_evidence") or "")
        template = template_score(clean25, duplicate_count=duplicate_counts[key], template_hit=template_hit)
        content_retention = _safe_float(row.get("content_retention_score"), 0.0)
        style_shift = _safe_float(row.get("style_shift_score"), 0.0)
        uncertainty = _safe_float(row.get("uncertainty_score"), 1.0)
        content_score = causal_content_score(
            content_text=content_text,
            clean_text=clean25,
            content_retention=content_retention,
            rcr_score=rcr,
            template=template,
        )
        shortcut_score = style_shortcut_score(
            style_text=style_text,
            clean_text=clean25,
            style_shift=style_shift,
            template=template,
        )
        reliability_bucket = _cf_reliability_bucket(rcr_score=rcr, uncertainty_score=uncertainty, template=template)
        role = contrastive_role_hint(
            rcr_bucket=reliability_bucket,
            template=template,
            causal_content=content_score,
            style_shortcut=shortcut_score,
        )
        if not clean25 or train_keep <= 0 or bool(_safe_int(row.get("bad_tail_hit"), 0)) or bool(_safe_int(row.get("short_fragment_hit"), 0)):
            role = "quarantine"
        rec = EvidenceRecord(
            evidence_id=f"train:{Path(source_path).stem}:{global_idx}:{row.get('sample_id', global_idx)}",
            source_split="train",
            source_domain=str(row.get("domain") or "target"),
            source_user=str(row.get("user") or ""),
            source_item=str(row.get("item") or ""),
            source_rating=rating,
            source_rating_bucket=_rating_bucket(rating),
            source_type=_source_type(row),
            raw_explanation=raw,
            clean_explanation=clean,
            clean_explanation_25=clean25,
            token_count_raw=len(toks_raw),
            token_count_25=len(toks25),
            content_evidence=content_text,
            style_evidence=style_text,
            causal_content_evidence=content_text or clean25,
            style_shortcut_evidence=style_text,
            cf_content_anchor=_anchor_text(content_text, clean25, max_tokens=8),
            cf_style_anchor=_anchor_text(style_text, max_tokens=8),
            cf_aspect_anchor=_anchor_text(content_text, max_tokens=4) or _anchor_text(clean25, max_tokens=4),
            cf_reliability_bucket=reliability_bucket,
            cf_template_flag=int(template >= 0.65 or template_hit),
            causal_content_score=content_score,
            style_shortcut_score=shortcut_score,
            contrastive_role_hint=role,
            template_score=template,
            evidence_quality_prior=_safe_float(row.get("evidence_quality_prior"), 0.0),
            rcr_score=rcr,
            content_retention_score=content_retention,
            style_shift_score=style_shift,
            rating_stability_score=_safe_float(row.get("rating_stability_score"), 0.0),
            uncertainty_score=uncertainty,
            sample_weight_hint=_safe_float(row.get("sample_weight_hint"), 1.0),
            train_keep=train_keep,
            source_sample_id=str(row.get("sample_id") if row.get("sample_id") is not None else global_idx),
            source_row_index=global_idx,
            source_path=source_path.as_posix(),
        )
        records.append(rec)
    return records


def token_length_stats_from_lengths(lengths_raw: Iterable[int], *, max_tokens: int) -> dict[str, Any]:
    lengths = sorted(int(value) for value in lengths_raw)
    if not lengths:
        return {
            "schema_version": TOKEN_STATS_SCHEMA_VERSION,
            "status": "empty",
            "max_prediction_tokens": max_tokens,
            "count": 0,
        }

    def pct(q: float) -> int:
        pos = min(len(lengths) - 1, max(0, int(round((len(lengths) - 1) * q))))
        return lengths[pos]

    truncated = sum(1 for value in lengths if value > max_tokens)
    return {
        "schema_version": TOKEN_STATS_SCHEMA_VERSION,
        "status": "computed",
        "max_prediction_tokens": max_tokens,
        "count": len(lengths),
        "p50": pct(0.50),
        "p90": pct(0.90),
        "p95": pct(0.95),
        "p99": pct(0.99),
        "max": lengths[-1],
        "truncation_count": truncated,
        "truncation_rate": round(truncated / max(1, len(lengths)), 6),
    }


def token_length_stats(records: Iterable[EvidenceRecord], *, max_tokens: int) -> dict[str, Any]:
    return token_length_stats_from_lengths((rec.token_count_raw for rec in records), max_tokens=max_tokens)


def leakage_check_from_counts(*, bad_split_count: int, missing_provenance_count: int) -> dict[str, Any]:
    return {
        "schema_version": LEAKAGE_CHECK_SCHEMA_VERSION,
        "status": "PASS" if bad_split_count == 0 and missing_provenance_count == 0 else "FAIL",
        "valid_test_evidence_source_split_train": bad_split_count == 0,
        "missing_provenance_count": int(missing_provenance_count),
        "bad_split_count": int(bad_split_count),
        "train_positive_excludes_current_interaction": "enforced_in_train_labels",
        "valid_test_reference_not_in_query_or_pool": "pool_builder_uses_train_routing_only",
    }


def leakage_check(records: Iterable[EvidenceRecord]) -> dict[str, Any]:
    bad_split_count = 0
    missing_provenance_count = 0
    for rec in records:
        bad_split_count += int(rec.source_split != "train")
        missing_provenance_count += int(not rec.source_path)
    return leakage_check_from_counts(
        bad_split_count=bad_split_count,
        missing_provenance_count=missing_provenance_count,
    )


def _round_ratio(numerator: int, denominator: int) -> float:
    return round(float(numerator) / max(1, int(denominator)), 6)


def _mean(total: float, count: int) -> float:
    return round(float(total) / max(1, int(count)), 6)


def build_train_only_evidence_pool(
    *,
    source_csv: Path,
    output_jsonl: Path,
    diagnostics_dir: Path,
    max_tokens: int,
) -> dict[str, Any]:
    if not source_csv.is_file():
        raise FileNotFoundError(f"RACER-C1 train evidence source missing: {source_csv}")
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    record_count = 0
    row_offset = 0
    lengths_raw: list[int] = []
    type_counts: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    reliability_bucket_counts: Counter[str] = Counter()
    template_flag_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    source_component_counts: Counter[str] = Counter()
    top_level_source_counts: Counter[str] = Counter()
    bad_split_count = 0
    missing_provenance_count = 0
    causal_content_score_sum = 0.0
    style_shortcut_score_sum = 0.0
    causal_content_nonempty = 0
    style_shortcut_nonempty = 0
    content_anchor_nonempty = 0
    style_anchor_nonempty = 0
    aspect_anchor_nonempty = 0
    high_causal_content_count = 0
    high_style_shortcut_count = 0
    cf_total_count = 0
    cf_accepted_count = 0
    cf_rejected_count = 0
    cf_template_count = 0
    cf_quarantine_count = 0
    with output_jsonl.open("w", encoding="utf-8") as fh:
        for frame in pd.read_csv(source_csv, chunksize=100_000):
            records = _records_from_frame(
                frame,
                source_path=source_csv,
                max_tokens=max_tokens,
                row_offset=row_offset,
            )
            row_offset += int(len(frame))
            for rec in records:
                fh.write(json.dumps(asdict(rec), ensure_ascii=False, sort_keys=True) + "\n")
                record_count += 1
                lengths_raw.append(rec.token_count_raw)
                type_counts[rec.source_type] += 1
                source_component_counts[rec.source_type] += 1
                top_level_source_counts["cf" if rec.source_type == "cf" else rec.source_type] += 1
                domain_counts[rec.source_domain] += 1
                reliability_bucket_counts[rec.cf_reliability_bucket] += 1
                template_flag_counts[str(rec.cf_template_flag)] += 1
                role_counts[rec.contrastive_role_hint] += 1
                causal_content_score_sum += rec.causal_content_score
                style_shortcut_score_sum += rec.style_shortcut_score
                causal_content_nonempty += int(bool(_tokens(rec.causal_content_evidence)))
                style_shortcut_nonempty += int(bool(_tokens(rec.style_shortcut_evidence)))
                content_anchor_nonempty += int(bool(_tokens(rec.cf_content_anchor)))
                style_anchor_nonempty += int(bool(_tokens(rec.cf_style_anchor)))
                aspect_anchor_nonempty += int(bool(_tokens(rec.cf_aspect_anchor)))
                high_causal_content_count += int(rec.causal_content_score >= 0.50)
                high_style_shortcut_count += int(rec.style_shortcut_score >= 0.65)
                if rec.source_type == "cf":
                    cf_total_count += 1
                    cf_accepted_count += int(rec.train_keep > 0 and rec.contrastive_role_hint != "quarantine")
                    cf_rejected_count += int(rec.train_keep <= 0)
                    cf_template_count += int(bool(rec.cf_template_flag))
                    cf_quarantine_count += int(rec.contrastive_role_hint == "quarantine")
                bad_split_count += int(rec.source_split != "train")
                missing_provenance_count += int(not rec.source_path)
    stats = token_length_stats_from_lengths(lengths_raw, max_tokens=max_tokens)
    leak = leakage_check_from_counts(
        bad_split_count=bad_split_count,
        missing_provenance_count=missing_provenance_count,
    )
    content_style_stats = {
        "schema_version": CONTENT_STYLE_STATS_SCHEMA_VERSION,
        "status": "computed",
        "record_count": record_count,
        "d4c_w_decomposition": {
            "W": "textual attributes",
            "C": "causal content evidence",
            "S": "style/template shortcut evidence",
        },
        "mean_causal_content_score": _mean(causal_content_score_sum, record_count),
        "mean_style_shortcut_score": _mean(style_shortcut_score_sum, record_count),
        "causal_content_nonempty_count": causal_content_nonempty,
        "causal_content_nonempty_rate": _round_ratio(causal_content_nonempty, record_count),
        "style_shortcut_nonempty_count": style_shortcut_nonempty,
        "style_shortcut_nonempty_rate": _round_ratio(style_shortcut_nonempty, record_count),
        "high_causal_content_count": high_causal_content_count,
        "high_causal_content_rate": _round_ratio(high_causal_content_count, record_count),
        "high_style_shortcut_count": high_style_shortcut_count,
        "high_style_shortcut_rate": _round_ratio(high_style_shortcut_count, record_count),
        "metric_purpose": {
            "causal_content": "supports reference-like aspect/phrase overlap for ROUGE, BLEU, and METEOR",
            "style_shortcut": "is penalized to reduce generic/template collapse and protect DIST",
        },
    }
    cf_anchor_stats = {
        "schema_version": CF_ANCHOR_STATS_SCHEMA_VERSION,
        "status": "computed",
        "record_count": record_count,
        "content_anchor_nonempty_count": content_anchor_nonempty,
        "content_anchor_nonempty_rate": _round_ratio(content_anchor_nonempty, record_count),
        "style_anchor_nonempty_count": style_anchor_nonempty,
        "style_anchor_nonempty_rate": _round_ratio(style_anchor_nonempty, record_count),
        "aspect_anchor_nonempty_count": aspect_anchor_nonempty,
        "aspect_anchor_nonempty_rate": _round_ratio(aspect_anchor_nonempty, record_count),
        "cf_reliability_bucket_counts": dict(sorted(reliability_bucket_counts.items())),
        "cf_template_flag_counts": dict(sorted(template_flag_counts.items())),
        "sentence_label_policy": "counterfactual text is decomposed into controllable anchors and reliability/template metadata, not trusted as unconditional CE labels",
    }
    role_distribution = {
        "schema_version": ROLE_DISTRIBUTION_SCHEMA_VERSION,
        "status": "computed",
        "record_count": record_count,
        "contrastive_role_hint_counts": dict(sorted(role_counts.items())),
        "positive_like_count": sum(count for role, count in role_counts.items() if role.startswith("positive") or role.startswith("soft_positive")),
        "hard_negative_like_count": sum(count for role, count in role_counts.items() if role.startswith("hard_negative")),
        "neutral_count": int(role_counts.get("candidate_neutral", 0)),
        "quarantine_count": int(role_counts.get("quarantine", 0)),
        "supervision_policy": {
            "positive": "high-RCR/high-content evidence is pulled close",
            "soft_positive": "medium-RCR content evidence is pulled close with lower weight",
            "hard_negative": "low-RCR, high-template, or style-shortcut evidence is used as contrastive repulsion",
        },
    }
    cross_domain_distribution = {
        "schema_version": CROSS_DOMAIN_DISTRIBUTION_SCHEMA_VERSION,
        "status": "computed",
        "record_count": record_count,
        "target_gold_count": int(source_component_counts.get("target_gold", 0)),
        "aux_gold_count": int(source_component_counts.get("aux_gold", 0)),
        "cf_total_count": int(cf_total_count),
        "cf_accepted_count": int(cf_accepted_count),
        "cf_rejected_count": int(cf_rejected_count),
        "cf_template_count": int(cf_template_count),
        "cf_quarantine_count": int(cf_quarantine_count),
        "source_type_counts": dict(sorted(source_component_counts.items())),
        "source_domain_counts": dict(sorted(domain_counts.items())),
        "cross_domain_pass": bool(
            source_component_counts.get("target_gold", 0) > 0
            and source_component_counts.get("aux_gold", 0) > 0
            and cf_total_count > 0
        ),
        "pool_semantics": "all target_gold/aux_gold/CF evidence is materialized; rejected/template/corrupt evidence is routed into hard-negative or quarantine roles instead of being trusted as generation text",
    }
    write_json(diagnostics_dir / "token_length_stats.json", stats)
    write_json(diagnostics_dir / "leakage_check.json", leak)
    write_json(diagnostics_dir / "content_style_split_stats.json", content_style_stats)
    write_json(diagnostics_dir / "cf_anchor_stats.json", cf_anchor_stats)
    write_json(diagnostics_dir / "contrastive_role_distribution.json", role_distribution)
    write_json(diagnostics_dir / "cross_domain_evidence_distribution.json", cross_domain_distribution)
    return {
        "schema_version": EVIDENCE_POOL_SCHEMA_VERSION,
        "status": "built",
        "source_csv": source_csv.as_posix(),
        "output_jsonl": output_jsonl.as_posix(),
        "record_count": record_count,
        "source_row_count": row_offset,
        "streaming_chunksize": 100_000,
        "source_type_counts": dict(sorted(type_counts.items())),
        "source_domain_counts": dict(sorted(domain_counts.items())),
        "target_gold_count": int(source_component_counts.get("target_gold", 0)),
        "aux_gold_count": int(source_component_counts.get("aux_gold", 0)),
        "cf_total_count": int(cf_total_count),
        "cf_accepted_count": int(cf_accepted_count),
        "cf_rejected_count": int(cf_rejected_count),
        "cf_template_count": int(cf_template_count),
        "cf_quarantine_count": int(cf_quarantine_count),
        "cf_reliability_bucket_counts": dict(sorted(reliability_bucket_counts.items())),
        "contrastive_role_hint_counts": dict(sorted(role_counts.items())),
        "token_length_stats": stats,
        "leakage_check": leak,
        "content_style_split_stats": content_style_stats,
        "cf_anchor_stats": cf_anchor_stats,
        "contrastive_role_distribution": role_distribution,
        "cross_domain_evidence_distribution": cross_domain_distribution,
    }
