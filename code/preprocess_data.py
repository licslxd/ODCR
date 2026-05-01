from __future__ import annotations

import argparse
import os
import pickle
import re
import sys
import time
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_contract import (
    EXPLANATION_PLACEHOLDER,
    PROCESSED_COLUMN_ORDER,
    anchor_visible_explanation,
    write_preprocess_csv,
)

DATASETS = ["AM_Movies", "AM_Electronics", "AM_CDs", "TripAdvisor", "Yelp"]
AMAZON_DATASETS = {"AM_Movies", "AM_Electronics", "AM_CDs"}
CANONICAL_PREPROCESS_CHUNK_SIZE = 50_000


def _install_resolved_data_dir(data_dir: str) -> None:
    raw = str(data_dir or "").strip()
    if not raw:
        raise ValueError("preprocess_data.py requires --data-dir from the resolved One-Control preprocess payload.")
    os.environ["ODCR_RESOLVED_DATA_DIR"] = os.path.abspath(os.path.expanduser(raw))


def _resolved_data_dir() -> str:
    raw = str(os.environ.get("ODCR_RESOLVED_DATA_DIR") or "").strip()
    if not raw:
        raise RuntimeError(
            "preprocess_data.py requires ODCR_RESOLVED_DATA_DIR/--data-dir from ./odcr; "
            "refusing to read configs/odcr.yaml as a child-side fallback."
        )
    return os.path.abspath(os.path.expanduser(raw))

STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "on",
    "for",
    "this",
    "that",
    "it",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "with",
    "as",
    "by",
    "at",
    "from",
    "i",
    "you",
    "he",
    "she",
    "they",
    "we",
    "my",
    "our",
    "your",
    "their",
    "its",
    "but",
    "if",
    "not",
    "just",
    "all",
    "can",
    "will",
    "would",
    "about",
    "into",
    "than",
    "then",
    "there",
    "here",
    "when",
    "while",
    "also",
    "too",
    "such",
    "what",
    "which",
    "where",
    "how",
    "why",
    "because",
    "however",
    "although",
    "really",
}
ASPECT_MAP: Dict[str, List[str]] = {
    "quality": ["quality", "durable", "cheap", "premium", "excellent", "poor"],
    "service": ["service", "staff", "support", "delivery", "waiter"],
    "price": ["price", "cost", "value", "expensive", "affordable"],
    "taste": ["taste", "flavor", "delicious", "bland", "fresh"],
    "design": ["design", "look", "style", "layout", "interface"],
}
POS_WORDS = {"good", "great", "excellent", "amazing", "love", "perfect", "nice"}
NEG_WORDS = {"bad", "poor", "awful", "hate", "terrible", "worse", "worst"}

TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_'-]*")
ENTITY_PATTERN = re.compile(r"\b[A-Z][a-zA-Z0-9]+\b")
FIRST_PERSON_PATTERN = re.compile(r"\bI\b|\bmy\b|\bme\b")
INTENSIFIER_PATTERN = re.compile(r"\bvery\b|\breally\b|\bextremely\b")
CONTRASTIVE_PATTERN = re.compile(r"\bhowever\b|\bbut\b|\balthough\b")
BECAUSE_PATTERN = re.compile(r"\bbecause\b")


@dataclass(frozen=True, slots=True)
class ContentBundle:
    content_keywords: str
    content_aspects: str
    content_entities: str
    content_evidence: str
    content_anchor_score: float


@dataclass(frozen=True, slots=True)
class StyleBundle:
    style_markers: str
    template_family: str
    polarity_anchor: str
    length_style_bucket: str
    domain_style_anchor: str
    local_style_residual_hint: str
    style_evidence: str
    style_anchor_score: float


@dataclass(frozen=True, slots=True)
class QualityRoutingBundle:
    evidence_quality_prior: float
    preprocess_route_scorer_prior: int
    preprocess_route_explainer_prior: int


def _tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(str(text).lower())


def _dedupe_preserve_order(values: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for value in values:
        clean = str(value).strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        out.append(clean)
    return out


def _join_or_default(values: List[str], *, default: str = "none") -> str:
    clean = _dedupe_preserve_order(values)
    return "|".join(clean) if clean else default


def _surface_text(value: str) -> str:
    text = str(value).replace("|", " ").strip()
    return text if text else "none"


def _top_keywords(tokens: List[str], k: int = 6) -> List[str]:
    counts: Dict[str, int] = {}
    for token in tokens:
        if token in STOPWORDS or len(token) < 3:
            continue
        counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [word for word, _ in ranked[:k]]


def _content_aspects(tokens: List[str]) -> List[str]:
    token_set = set(tokens)
    out: List[str] = []
    for aspect, keywords in ASPECT_MAP.items():
        if any(keyword in token_set for keyword in keywords):
            out.append(aspect)
    return out[:4]


def _content_entities(raw: str) -> List[str]:
    entities = set(ENTITY_PATTERN.findall(str(raw)))
    return sorted(list(entities))[:6]


def _content_evidence_text(*, keywords: str, aspects: str, entities: str) -> str:
    return (
        f"keywords {_surface_text(keywords)} ; "
        f"aspects {_surface_text(aspects)} ; "
        f"entities {_surface_text(entities)}"
    )


def _build_content_bundle(review: str, explanation: str) -> ContentBundle:
    joined_text = f"{review} {explanation}".strip()
    tokens = _tokenize(joined_text)
    keywords = _join_or_default(_top_keywords(tokens, k=6))
    aspects = _join_or_default(_content_aspects(tokens))
    entities = _join_or_default(_content_entities(joined_text))
    score = min(
        1.0,
        0.16 * len([x for x in keywords.split("|") if x and x != "none"])
        + 0.18 * len([x for x in aspects.split("|") if x and x != "none"])
        + 0.08 * len([x for x in entities.split("|") if x and x != "none"]),
    )
    return ContentBundle(
        content_keywords=keywords,
        content_aspects=aspects,
        content_entities=entities,
        content_evidence=_content_evidence_text(
            keywords=keywords,
            aspects=aspects,
            entities=entities,
        ),
        content_anchor_score=round(float(score), 4),
    )


def _style_markers(text: str) -> List[str]:
    raw = str(text)
    lowered = raw.lower()
    markers: List[str] = []
    if "!" in raw:
        markers.append("exclaim")
    if "?" in raw:
        markers.append("question")
    if FIRST_PERSON_PATTERN.search(raw):
        markers.append("first_person")
    if INTENSIFIER_PATTERN.search(lowered):
        markers.append("intensifier")
    if CONTRASTIVE_PATTERN.search(lowered):
        markers.append("contrastive")
    return markers[:5]


def _template_family(text: str) -> str:
    raw = str(text).strip()
    lowered = raw.lower()
    if raw.endswith("?"):
        return "question"
    if CONTRASTIVE_PATTERN.search(lowered):
        return "contrastive_statement"
    if BECAUSE_PATTERN.search(lowered):
        return "causal_statement"
    if raw.startswith(("I ", "i ")):
        return "first_person_statement"
    if raw.endswith("!"):
        return "exclamatory_statement"
    return "plain_statement"


def _polarity_anchor_from_tokens(tokens: List[str]) -> str:
    pos = sum(1 for token in tokens if token in POS_WORDS)
    neg = sum(1 for token in tokens if token in NEG_WORDS)
    if pos > neg:
        return "positive"
    if neg > pos:
        return "negative"
    return "neutral"


def _length_bucket_from_tokens(tokens: List[str]) -> str:
    token_count = len(tokens)
    if token_count <= 8:
        return "short"
    if token_count <= 20:
        return "medium"
    return "long"


def _local_style_residual_hint(markers: List[str], raw_text: str) -> str:
    marker_set = set(markers)
    perspective = "first_person" if "first_person" in marker_set else "external"
    intensity = "high" if ("intensifier" in marker_set or "exclaim" in marker_set) else "steady"
    discourse = "contrastive" if "contrastive" in marker_set else "direct"
    punctuation = "question" if "?" in raw_text else "exclaim" if "!" in raw_text else "flat"
    return (
        f"perspective={perspective};"
        f"intensity={intensity};"
        f"discourse={discourse};"
        f"punctuation={punctuation}"
    )


def _style_evidence_text(
    *,
    style_markers: str,
    template_family: str,
    polarity_anchor: str,
    length_style_bucket: str,
    domain_style_anchor: str,
    local_style_residual_hint: str,
) -> str:
    return (
        f"markers {_surface_text(style_markers)} ; "
        f"template_family {_surface_text(template_family)} ; "
        f"polarity {_surface_text(polarity_anchor)} ; "
        f"length {_surface_text(length_style_bucket)} ; "
        f"domain_style_anchor {_surface_text(domain_style_anchor)} ; "
        f"local_style_residual_hint {_surface_text(local_style_residual_hint)}"
    )


def _build_style_bundle(dataset: str, explanation: str) -> StyleBundle:
    raw_text = str(explanation).strip()
    tokens = _tokenize(raw_text)
    markers = _style_markers(raw_text)
    markers_text = _join_or_default(markers)
    template_family = _template_family(raw_text)
    polarity_anchor = _polarity_anchor_from_tokens(tokens)
    length_style_bucket = _length_bucket_from_tokens(tokens)
    domain_style_anchor = f"{dataset}:{template_family}:{length_style_bucket}:{polarity_anchor}"
    local_style_residual_hint = _local_style_residual_hint(markers, raw_text)
    score = min(
        1.0,
        0.18 * len(markers)
        + (0.22 if polarity_anchor != "neutral" else 0.12)
        + (0.22 if template_family != "plain_statement" else 0.08)
        + (0.12 if length_style_bucket != "medium" else 0.06),
    )
    return StyleBundle(
        style_markers=markers_text,
        template_family=template_family,
        polarity_anchor=polarity_anchor,
        length_style_bucket=length_style_bucket,
        domain_style_anchor=domain_style_anchor,
        local_style_residual_hint=local_style_residual_hint,
        style_evidence=_style_evidence_text(
            style_markers=markers_text,
            template_family=template_family,
            polarity_anchor=polarity_anchor,
            length_style_bucket=length_style_bucket,
            domain_style_anchor=domain_style_anchor,
            local_style_residual_hint=local_style_residual_hint,
        ),
        style_anchor_score=round(float(score), 4),
    )


def _build_quality_routing_bundle(content: ContentBundle, style: StyleBundle) -> QualityRoutingBundle:
    evidence_quality_prior = round(
        float(min(1.0, 0.55 * content.content_anchor_score + 0.45 * style.style_anchor_score)),
        4,
    )
    preprocess_route_scorer_prior = int(evidence_quality_prior >= 0.45 and content.content_anchor_score >= 0.25)
    preprocess_route_explainer_prior = int(
        style.style_anchor_score >= 0.35
        or style.template_family != "plain_statement"
        or style.polarity_anchor != "neutral"
    )
    return QualityRoutingBundle(
        evidence_quality_prior=evidence_quality_prior,
        preprocess_route_scorer_prior=preprocess_route_scorer_prior,
        preprocess_route_explainer_prior=preprocess_route_explainer_prior,
    )


def _schema_fail(dataset: str, available_columns: List[str], missing_hint: str) -> None:
    cols = ", ".join(sorted(available_columns)) or "<empty>"
    raise ValueError(
        f"{dataset} 原始 reviews.pickle 字段不满足当前预处理契约：{missing_hint}。\n"
        f"  当前可用字段: {cols}\n"
        "  预处理要求至少能解析出 user / item / rating / review / explanation；"
        "Amazon 域至少需要 text 或 sentence，模板域至少需要 template 或可回退的 explanation/text。"
    )


def _extract_nested_text(value) -> str:
    if value is None:
        return ""
    if pd.isna(value) if not isinstance(value, (list, tuple, dict)) else False:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        if len(value) > 2 and isinstance(value[2], str) and value[2].strip():
            return value[2].strip()
        for item in value:
            text = _extract_nested_text(item)
            if text:
                return text
    return ""


def _pick_series(
    df: pd.DataFrame,
    candidate_columns: List[str],
    *,
    transform=None,
) -> tuple[pd.Series | None, str | None]:
    for column in candidate_columns:
        if column not in df.columns:
            continue
        series = df[column]
        if transform is not None:
            series = series.apply(transform)
        return series.fillna("").astype(str), column
    return None, None


def _ensure_core_columns(dataset: str, df: pd.DataFrame) -> None:
    required = ["user", "item", "rating"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        _schema_fail(dataset, list(df.columns), f"缺少核心字段 {missing}")


def _finalize_review_and_explanation(df: pd.DataFrame, review: pd.Series, explanation: pd.Series) -> pd.DataFrame:
    out = df.copy()
    out["review"] = review.fillna("").astype(str)
    out["explanation"] = explanation.fillna("").astype(str)
    out["review"] = out["review"].str.strip()
    out["explanation"] = out["explanation"].str.strip()
    out["explanation"] = out["explanation"].where(out["explanation"] != "", out["review"])
    out["explanation"] = out["explanation"].where(out["explanation"] != "", EXPLANATION_PLACEHOLDER)
    return out


def _prepare_amazon_dataset(dataset: str, df: pd.DataFrame) -> pd.DataFrame:
    _ensure_core_columns(dataset, df)
    review, review_src = _pick_series(df, ["text", "review"])
    explanation, explanation_src = _pick_series(
        df,
        ["sentence", "explanation"],
        transform=_extract_nested_text,
    )
    if review is None and explanation is None:
        _schema_fail(dataset, list(df.columns), "Amazon 域至少需要 text 或 sentence")
    if review is None:
        review = explanation.copy()
    if explanation is None:
        explanation = review.copy()
    out = _finalize_review_and_explanation(df, review, explanation)
    drop_cols = [column for column in ("sentence", "template", "predicted") if column in out.columns]
    if drop_cols:
        out = out.drop(columns=drop_cols)
    if not review_src and not explanation_src:
        _schema_fail(dataset, list(df.columns), "未能解析 review / explanation")
    return out


def _prepare_template_dataset(dataset: str, df: pd.DataFrame) -> pd.DataFrame:
    _ensure_core_columns(dataset, df)
    if dataset == "Yelp":
        df = df.sample(n=len(df) // 2, random_state=42)
    else:
        df = df.sample(n=int(len(df) * 0.9), random_state=42)
    explanation, explanation_src = _pick_series(
        df,
        ["template", "explanation", "text", "review"],
        transform=_extract_nested_text,
    )
    review, _ = _pick_series(df, ["review", "text"])
    if explanation is None:
        _schema_fail(dataset, list(df.columns), "模板域至少需要 template 或可回退的 explanation/text")
    if review is None:
        review = explanation.copy()
    out = _finalize_review_and_explanation(df, review, explanation)
    if dataset == "Yelp":
        empty_review_mask = out["review"].fillna("").astype(str).str.strip().eq("")
        dropped_empty_review_rows = int(empty_review_mask.sum())
        if dropped_empty_review_rows:
            out = out.loc[~empty_review_mask].copy()
        _log_stage(
            dataset,
            "prepare_raw",
            f"dropped_empty_review_rows={dropped_empty_review_rows}",
        )
    drop_cols = [column for column in ("template", "predicted", "sentence") if column in out.columns]
    if drop_cols:
        out = out.drop(columns=drop_cols)
    if explanation_src is None:
        _schema_fail(dataset, list(df.columns), "未能解析 explanation")
    return out


def _prepare_raw_dataset(dataset: str) -> pd.DataFrame:
    with open(os.path.join(_resolved_data_dir(), dataset, "reviews.pickle"), "rb") as handle:
        data = pickle.load(handle)
    df = pd.DataFrame(data)
    if df.empty:
        raise ValueError(f"{dataset} 原始 reviews.pickle 为空，无法执行预处理。")
    if dataset in AMAZON_DATASETS:
        return _prepare_amazon_dataset(dataset, df)
    return _prepare_template_dataset(dataset, df)


def _log_stage(dataset: str, stage: str, message: str) -> None:
    print(f"[preprocess_data] dataset={dataset} stage={stage} {message}", flush=True)


def _build_canonical_preprocess_assets(dataset: str, df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    reviews = out["review"].fillna("").astype(str).tolist()
    explanations = out["explanation"].fillna("").astype(str).tolist()
    effective_explanations = [
        anchor_visible_explanation(explanation) or str(review).strip()
        for review, explanation in zip(reviews, explanations)
    ]
    total = len(out)
    chunk_size = CANONICAL_PREPROCESS_CHUNK_SIZE

    content_evidence_all: List[str] = []
    content_anchor_score_all: List[float] = []
    polarity_anchor_all: List[str] = []
    domain_style_anchor_all: List[str] = []
    local_style_residual_hint_all: List[str] = []
    style_evidence_all: List[str] = []
    style_anchor_score_all: List[float] = []
    evidence_quality_prior_all: List[float] = []
    preprocess_route_scorer_prior_all: List[int] = []
    preprocess_route_explainer_prior_all: List[int] = []

    t_anchor = time.perf_counter()
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        for review, explanation in zip(reviews[start:end], effective_explanations[start:end]):
            content = _build_content_bundle(review, explanation)
            style = _build_style_bundle(dataset, explanation)
            routing = _build_quality_routing_bundle(content, style)

            content_evidence_all.append(content.content_evidence)
            content_anchor_score_all.append(content.content_anchor_score)
            polarity_anchor_all.append(style.polarity_anchor)
            domain_style_anchor_all.append(style.domain_style_anchor)
            local_style_residual_hint_all.append(style.local_style_residual_hint)
            style_evidence_all.append(style.style_evidence)
            style_anchor_score_all.append(style.style_anchor_score)
            evidence_quality_prior_all.append(routing.evidence_quality_prior)
            preprocess_route_scorer_prior_all.append(routing.preprocess_route_scorer_prior)
            preprocess_route_explainer_prior_all.append(routing.preprocess_route_explainer_prior)

        _log_stage(
            dataset,
            "build_assets",
            f"rows={end}/{total} elapsed_s={time.perf_counter() - t_anchor:.2f}",
        )

    out["content_evidence"] = content_evidence_all
    out["content_anchor_score"] = content_anchor_score_all
    out["polarity_anchor"] = polarity_anchor_all
    out["domain_style_anchor"] = domain_style_anchor_all
    out["local_style_residual_hint"] = local_style_residual_hint_all
    out["style_evidence"] = style_evidence_all
    out["style_anchor_score"] = style_anchor_score_all
    out["evidence_quality_prior"] = evidence_quality_prior_all
    out["preprocess_route_scorer_prior"] = preprocess_route_scorer_prior_all
    out["preprocess_route_explainer_prior"] = preprocess_route_explainer_prior_all
    return out


def _iterative_k_core(df: pd.DataFrame, min_user_item_count: int = 5, rounds: int = 30) -> pd.DataFrame:
    out = df
    for round_idx in range(rounds):
        user_interactions = out["user"].value_counts()
        item_interactions = out["item"].value_counts()
        keep_users = user_interactions[user_interactions >= min_user_item_count].index
        keep_items = item_interactions[item_interactions >= min_user_item_count].index
        next_out = out[out["user"].isin(keep_users) & out["item"].isin(keep_items)]
        if len(next_out) == len(out):
            out = next_out
            break
        out = next_out
        if round_idx == rounds - 1:
            break
    return out.reset_index(drop=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="预处理 reviews.pickle -> processed.csv；支持按数据集子集执行。",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="逗号分隔的数据集名，如 'Yelp' 或 'AM_Movies,TripAdvisor'；不传则处理全部。",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Resolved project.data_dir from ./odcr; direct YAML fallback is forbidden.",
    )
    return parser.parse_args()


def _resolve_datasets(raw: str | None) -> List[str]:
    if raw is None or not str(raw).strip():
        return list(DATASETS)
    datasets = [dataset.strip() for dataset in str(raw).split(",") if dataset.strip()]
    unknown = [dataset for dataset in datasets if dataset not in DATASETS]
    if unknown:
        raise ValueError(f"未知数据集: {unknown}; 可选值: {DATASETS}")
    return datasets


if __name__ == "__main__":
    args = _parse_args()
    _install_resolved_data_dir(args.data_dir)
    for dataset in _resolve_datasets(args.datasets):
        t_dataset = time.perf_counter()
        _log_stage(dataset, "prepare_raw", "start")
        raw_df = _prepare_raw_dataset(dataset)
        _log_stage(
            dataset,
            "prepare_raw",
            f"done rows={len(raw_df)} elapsed_s={time.perf_counter() - t_dataset:.2f}",
        )

        t_kcore = time.perf_counter()
        _log_stage(dataset, "k_core", "start")
        filtered = _iterative_k_core(raw_df)
        _log_stage(
            dataset,
            "k_core",
            f"done rows={len(filtered)} elapsed_s={time.perf_counter() - t_kcore:.2f}",
        )

        t_assets = time.perf_counter()
        _log_stage(dataset, "build_assets", "start")
        canonical = _build_canonical_preprocess_assets(dataset, filtered)
        canonical = canonical.loc[:, list(PROCESSED_COLUMN_ORDER)].copy()
        _log_stage(
            dataset,
            "build_assets",
            f"done rows={len(canonical)} elapsed_s={time.perf_counter() - t_assets:.2f}",
        )

        output_path = os.path.join(_resolved_data_dir(), dataset, "processed.csv")
        write_preprocess_csv(canonical, output_path, column_order=PROCESSED_COLUMN_ORDER)
        _log_stage(
            dataset,
            "write_csv",
            f"done path={output_path} rows={len(canonical)} total_elapsed_s={time.perf_counter() - t_dataset:.2f}",
        )
