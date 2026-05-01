from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal

import pandas as pd

EXPLANATION_PLACEHOLDER = "No explanation provided"
PREPROCESS_CONTRACT_VERSION = "odcr_preprocess_contract/3.1"

ContractStage = Literal["processed", "split", "merged"]
FieldKind = Literal["text", "float", "int"]

CORE_REQUIRED_COLUMNS = ("user", "item", "rating", "review", "explanation")

CANONICAL_PREPROCESS_ASSET_COLUMNS = (
    "content_evidence",
    "content_anchor_score",
    "style_evidence",
    "style_anchor_score",
    "polarity_anchor",
    "domain_style_anchor",
    "local_style_residual_hint",
    "evidence_quality_prior",
    "preprocess_route_scorer_prior",
    "preprocess_route_explainer_prior",
)

DEPRECATED_PREPROCESS_DETAIL_COLUMNS = (
    "content_keywords",
    "content_aspects",
    "content_entities",
    "style_markers",
    "template_family",
    "length_style_bucket",
)

STEP4_POSTERIOR_ROUTE_COLUMNS = ("route_scorer", "route_explainer")

CONTENT_PROFILE_TEXT_COLUMNS = ("review", "content_evidence")
STYLE_PROFILE_TEXT_COLUMNS = (
    "explanation",
    "style_evidence",
    "domain_style_anchor",
    "polarity_anchor",
    "local_style_residual_hint",
)
DOMAIN_CONTENT_TEXT_COLUMNS = CONTENT_PROFILE_TEXT_COLUMNS
DOMAIN_STYLE_TEXT_COLUMNS = STYLE_PROFILE_TEXT_COLUMNS

POLARITY_VALUES = ("positive", "negative", "neutral")
MERGED_DOMAIN_VALUES = ("auxiliary", "target")
UNIT_INTERVAL_FLOAT_COLUMNS = (
    "content_anchor_score",
    "style_anchor_score",
    "evidence_quality_prior",
)
BINARY_INT_COLUMNS = ("preprocess_route_scorer_prior", "preprocess_route_explainer_prior")


@dataclass(frozen=True)
class PreprocessFieldSpec:
    name: str
    kind: FieldKind
    stages: tuple[ContractStage, ...]
    role: str
    description: str
    producer: str
    consumers: tuple[str, ...]
    default: str | float | int | None = None
    allow_empty: bool = False
    allowed_values: tuple[str, ...] | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


PREPROCESS_FIELD_SPECS: tuple[PreprocessFieldSpec, ...] = (
    PreprocessFieldSpec(
        name="user",
        kind="text",
        stages=("processed", "split", "merged"),
        role="core",
        description="Canonical raw user id.",
        producer="preprocess_data.py::_prepare_*_dataset",
        consumers=("split_data.py", "combine_data.py"),
    ),
    PreprocessFieldSpec(
        name="item",
        kind="text",
        stages=("processed", "split", "merged"),
        role="core",
        description="Canonical raw item id.",
        producer="preprocess_data.py::_prepare_*_dataset",
        consumers=("split_data.py", "combine_data.py"),
    ),
    PreprocessFieldSpec(
        name="rating",
        kind="float",
        stages=("processed", "split", "merged"),
        role="core",
        description="Observed supervision rating.",
        producer="preprocess_data.py::_prepare_*_dataset",
        consumers=("split_data.py", "combine_data.py", "step3", "step5"),
    ),
    PreprocessFieldSpec(
        name="review",
        kind="text",
        stages=("processed", "split", "merged"),
        role="core",
        description="Normalized review text used as the base content surface.",
        producer="preprocess_data.py::_finalize_review_and_explanation",
        consumers=("compute_embeddings.py", "infer_domain_semantics.py"),
    ),
    PreprocessFieldSpec(
        name="explanation",
        kind="text",
        stages=("processed", "split", "merged"),
        role="core",
        description="Normalized explanation text used as the base style surface.",
        producer="preprocess_data.py::_finalize_review_and_explanation",
        consumers=("compute_embeddings.py", "infer_domain_semantics.py", "step4", "step5"),
    ),
    PreprocessFieldSpec(
        name="content_evidence",
        kind="text",
        stages=("processed", "split", "merged"),
        role="evidence",
        description="Canonical content-side evidence object consumed by downstream content channels.",
        producer="preprocess_data.py::_build_content_bundle",
        consumers=("compute_embeddings.py", "infer_domain_semantics.py", "step4 routing"),
        default="keywords none ; aspects none ; entities none",
    ),
    PreprocessFieldSpec(
        name="content_anchor_score",
        kind="float",
        stages=("processed", "split", "merged"),
        role="anchor",
        description="Scalar content anchor strength for Step3/Step5 hard consumption.",
        producer="preprocess_data.py::_build_content_bundle",
        consumers=("step3_train_core.py", "step5_engine.py"),
        default=0.0,
    ),
    PreprocessFieldSpec(
        name="polarity_anchor",
        kind="text",
        stages=("processed", "split", "merged"),
        role="anchor",
        description="Canonical polarity anchor consumed by downstream style channels.",
        producer="preprocess_data.py::_build_style_bundle",
        consumers=("compute_embeddings.py", "infer_domain_semantics.py", "future step5 verbalizer"),
        default="neutral",
        allowed_values=POLARITY_VALUES,
    ),
    PreprocessFieldSpec(
        name="domain_style_anchor",
        kind="text",
        stages=("processed", "split", "merged"),
        role="anchor",
        description="Canonical domain-global style anchor for HSS-style downstream consumers.",
        producer="preprocess_data.py::_build_style_bundle",
        consumers=("compute_embeddings.py", "infer_domain_semantics.py", "future step3 HSS consumers"),
        default="unknown:plain_statement:medium:neutral",
    ),
    PreprocessFieldSpec(
        name="local_style_residual_hint",
        kind="text",
        stages=("processed", "split", "merged"),
        role="anchor",
        description="Canonical local style residual hint reserved for future HSS/local residual consumers.",
        producer="preprocess_data.py::_build_style_bundle",
        consumers=("compute_embeddings.py", "infer_domain_semantics.py", "future step3 HSS consumers"),
        default="perspective=external;intensity=steady;discourse=direct;punctuation=flat",
    ),
    PreprocessFieldSpec(
        name="style_evidence",
        kind="text",
        stages=("processed", "split", "merged"),
        role="evidence",
        description="Canonical style-side evidence object consumed by downstream style channels.",
        producer="preprocess_data.py::_build_style_bundle",
        consumers=("compute_embeddings.py", "infer_domain_semantics.py", "step4 routing"),
        default=(
            "markers none ; template_family plain_statement ; polarity neutral ; "
            "length medium ; domain_style_anchor unknown:plain_statement:medium:neutral ; "
            "local_style_residual_hint perspective=external;intensity=steady;discourse=direct;punctuation=flat"
        ),
    ),
    PreprocessFieldSpec(
        name="style_anchor_score",
        kind="float",
        stages=("processed", "split", "merged"),
        role="anchor",
        description="Scalar style anchor strength for Step3/Step5 hard consumption.",
        producer="preprocess_data.py::_build_style_bundle",
        consumers=("step3_train_core.py", "step5_engine.py"),
        default=0.0,
    ),
    PreprocessFieldSpec(
        name="evidence_quality_prior",
        kind="float",
        stages=("processed", "split", "merged"),
        role="quality",
        description="Canonical preprocess-side prior that summarizes evidence quality before Step4 reliability modeling.",
        producer="preprocess_data.py::_build_quality_routing_bundle",
        consumers=("step4_training_export.py", "step5_engine.py"),
        default=0.0,
    ),
    PreprocessFieldSpec(
        name="preprocess_route_scorer_prior",
        kind="int",
        stages=("processed", "split", "merged"),
        role="routing_prior",
        description="Preprocess-side scorer route prior. The route_scorer name is reserved for Step4 posterior exports only.",
        producer="preprocess_data.py::_build_quality_routing_bundle",
        consumers=("step4_training_export.py prior audit",),
        default=0,
    ),
    PreprocessFieldSpec(
        name="preprocess_route_explainer_prior",
        kind="int",
        stages=("processed", "split", "merged"),
        role="routing_prior",
        description="Preprocess-side explainer route prior. The route_explainer name is reserved for Step4 posterior exports only.",
        producer="preprocess_data.py::_build_quality_routing_bundle",
        consumers=("step4_training_export.py prior audit",),
        default=0,
    ),
    PreprocessFieldSpec(
        name="user_idx",
        kind="int",
        stages=("split", "merged"),
        role="transport",
        description="Split-local user index used by preprocess_b/preprocess_c and merged transport.",
        producer="split_data.py::split_func",
        consumers=("combine_data.py", "compute_embeddings.py", "infer_domain_semantics.py"),
        default=0,
    ),
    PreprocessFieldSpec(
        name="item_idx",
        kind="int",
        stages=("split", "merged"),
        role="transport",
        description="Split-local item index used by preprocess_b/preprocess_c and merged transport.",
        producer="split_data.py::split_func",
        consumers=("combine_data.py", "compute_embeddings.py", "infer_domain_semantics.py"),
        default=0,
    ),
    PreprocessFieldSpec(
        name="domain",
        kind="text",
        stages=("merged",),
        role="transport",
        description="Merged-domain transport label; combine_data must materialize it explicitly.",
        producer="combine_data.py::merge_task",
        consumers=("step3", "step4", "step5"),
        default="",
        allowed_values=MERGED_DOMAIN_VALUES,
    ),
)

PREPROCESS_FIELD_SPEC_BY_NAME = {spec.name: spec for spec in PREPROCESS_FIELD_SPECS}
PREPROCESS_TEXT_COLUMNS = tuple(spec.name for spec in PREPROCESS_FIELD_SPECS if spec.kind == "text")
PREPROCESS_FLOAT_COLUMNS = tuple(spec.name for spec in PREPROCESS_FIELD_SPECS if spec.kind == "float")
PREPROCESS_INT_COLUMNS = tuple(spec.name for spec in PREPROCESS_FIELD_SPECS if spec.kind == "int")

PREPROCESS_DERIVED_TEXT_DEFAULTS = {
    spec.name: spec.default
    for spec in PREPROCESS_FIELD_SPECS
    if spec.kind == "text" and spec.name not in CORE_REQUIRED_COLUMNS and spec.default is not None
}
PREPROCESS_DERIVED_FLOAT_DEFAULTS = {
    spec.name: spec.default
    for spec in PREPROCESS_FIELD_SPECS
    if spec.kind == "float" and spec.name != "rating" and spec.default is not None
}
PREPROCESS_DERIVED_INT_DEFAULTS = {
    spec.name: spec.default
    for spec in PREPROCESS_FIELD_SPECS
    if spec.kind == "int" and spec.default is not None
}
PREPROCESS_DERIVED_DEFAULTS = {
    **PREPROCESS_DERIVED_TEXT_DEFAULTS,
    **PREPROCESS_DERIVED_FLOAT_DEFAULTS,
    **PREPROCESS_DERIVED_INT_DEFAULTS,
}

PROCESSED_COLUMN_ORDER = tuple(
    spec.name for spec in PREPROCESS_FIELD_SPECS if "processed" in spec.stages
)
SPLIT_COLUMN_ORDER = tuple(spec.name for spec in PREPROCESS_FIELD_SPECS if "split" in spec.stages)
MERGED_COLUMN_ORDER = tuple(spec.name for spec in PREPROCESS_FIELD_SPECS if "merged" in spec.stages)
PREPROCESS_PROCESSED_ASSET_COLUMNS = tuple(
    column for column in PROCESSED_COLUMN_ORDER if column not in CORE_REQUIRED_COLUMNS
)


def expected_preprocess_column_order(
    *,
    require_split_indices: bool = False,
    require_domain: bool = False,
) -> tuple[str, ...]:
    if require_domain:
        return MERGED_COLUMN_ORDER
    if require_split_indices:
        return SPLIT_COLUMN_ORDER
    return PROCESSED_COLUMN_ORDER


def _expected_field_specs(
    *,
    require_split_indices: bool = False,
    require_domain: bool = False,
) -> tuple[PreprocessFieldSpec, ...]:
    stages = (
        ("merged",)
        if require_domain
        else ("split",)
        if require_split_indices
        else ("processed",)
    )
    return tuple(spec for spec in PREPROCESS_FIELD_SPECS if any(stage in spec.stages for stage in stages))


def render_preprocess_contract_snapshot(
    *,
    require_split_indices: bool = False,
    require_domain: bool = False,
) -> dict[str, object]:
    fields = _expected_field_specs(
        require_split_indices=require_split_indices,
        require_domain=require_domain,
    )
    return {
        "contract_version": PREPROCESS_CONTRACT_VERSION,
        "required_columns": list(
            expected_preprocess_column_order(
                require_split_indices=require_split_indices,
                require_domain=require_domain,
            )
        ),
        "canonical_asset_columns": list(CANONICAL_PREPROCESS_ASSET_COLUMNS),
        "content_channel_text_sources": list(CONTENT_PROFILE_TEXT_COLUMNS),
        "style_channel_text_sources": list(STYLE_PROFILE_TEXT_COLUMNS),
        "fields": [spec.to_dict() for spec in fields],
    }


def preprocess_csv_dtype_map(columns: Iterable[str]) -> dict[str, object]:
    dtype_map: dict[str, object] = {}
    for column in columns:
        spec = PREPROCESS_FIELD_SPEC_BY_NAME.get(column)
        if spec is None:
            continue
        if spec.kind == "text":
            dtype_map[column] = str
        elif spec.kind == "float":
            dtype_map[column] = "float64"
        elif spec.kind == "int":
            dtype_map[column] = "int64"
    return dtype_map


def _read_csv_header(path: str | Path) -> list[str]:
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            return next(reader)
        except StopIteration as exc:
            raise ValueError(f"{path} is empty; expected a preprocess CSV with a header.") from exc


def find_deprecated_preprocess_detail_columns(columns: Iterable[str]) -> list[str]:
    seen = set(DEPRECATED_PREPROCESS_DETAIL_COLUMNS)
    return [str(column) for column in columns if str(column) in seen]


def assert_no_deprecated_preprocess_detail_columns(
    columns: Iterable[str],
    *,
    source_label: str,
) -> None:
    deprecated = find_deprecated_preprocess_detail_columns(columns)
    if deprecated:
        raise ValueError(
            f"{source_label} contains retired preprocess detail columns: {deprecated}. "
            "Rerun preprocess_data.py / preprocess_a so processed, split, and merged CSVs "
            f"materialize {PREPROCESS_CONTRACT_VERSION} canonical evidence columns only."
        )


def find_step4_posterior_route_columns(columns: Iterable[str]) -> list[str]:
    seen = set(STEP4_POSTERIOR_ROUTE_COLUMNS)
    return [str(column) for column in columns if str(column) in seen]


def assert_no_step4_posterior_route_columns(
    columns: Iterable[str],
    *,
    source_label: str,
) -> None:
    stale = find_step4_posterior_route_columns(columns)
    if stale:
        raise ValueError(
            f"{source_label} contains Step4 posterior route columns in a preprocess CSV: {stale}. "
            "Rerun preprocess_a with preprocess_route_scorer_prior / "
            "preprocess_route_explainer_prior; route_scorer / route_explainer are Step4 posterior-only."
        )


def _validate_preprocess_header(
    columns: Iterable[str],
    *,
    require_split_indices: bool,
    require_domain: bool,
    source_label: str,
    require_exact_order: bool,
) -> None:
    actual = [str(column) for column in columns]
    expected = list(
        expected_preprocess_column_order(
            require_split_indices=require_split_indices,
            require_domain=require_domain,
        )
    )
    assert_no_deprecated_preprocess_detail_columns(actual, source_label=source_label)
    assert_no_step4_posterior_route_columns(actual, source_label=source_label)
    missing = [column for column in expected if column not in actual]
    if missing:
        raise ValueError(f"{source_label} missing required contract columns: {missing}")
    extras = [column for column in actual if column not in expected]
    if extras:
        raise ValueError(f"{source_label} contains non-contract columns: {extras}")
    if require_exact_order and actual != expected:
        raise ValueError(
            f"{source_label} column order does not match {PREPROCESS_CONTRACT_VERSION}; "
            f"expected={expected}, actual={actual}"
        )


def _normalize_text_series(series: pd.Series, *, strip: bool) -> pd.Series:
    out = series.fillna("").astype(str)
    if strip:
        out = out.str.strip()
    return out


def _normalize_numeric_series(series: pd.Series, dtype: str) -> pd.Series:
    return pd.to_numeric(series, errors="raise").astype(dtype)


def _first_bad_examples(mask: pd.Series, *, limit: int = 5) -> list[int]:
    return [int(idx) for idx in mask[mask].index[:limit].tolist()]


def _validate_non_empty_text_columns(df: pd.DataFrame, expected_fields: tuple[PreprocessFieldSpec, ...]) -> None:
    for spec in expected_fields:
        if spec.kind != "text" or spec.allow_empty or spec.name not in df.columns:
            continue
        stripped = df[spec.name].fillna("").astype(str).str.strip()
        bad = stripped.eq("")
        if bool(bad.any()):
            raise ValueError(
                f"preprocess column {spec.name!r} contains empty values at rows {_first_bad_examples(bad)}"
            )


def _validate_allowed_text_values(df: pd.DataFrame, expected_fields: tuple[PreprocessFieldSpec, ...]) -> None:
    for spec in expected_fields:
        if spec.kind != "text" or not spec.allowed_values or spec.name not in df.columns:
            continue
        series = df[spec.name].fillna("").astype(str).str.strip()
        invalid = sorted({value for value in series.tolist() if value not in spec.allowed_values})
        if invalid:
            raise ValueError(
                f"preprocess column {spec.name!r} has invalid values {invalid[:5]}; "
                f"allowed={list(spec.allowed_values)}"
            )


def _validate_unit_interval_columns(df: pd.DataFrame, *, expected_columns: tuple[str, ...]) -> None:
    for column in UNIT_INTERVAL_FLOAT_COLUMNS:
        if column not in expected_columns or column not in df.columns:
            continue
        series = df[column].astype("float64")
        invalid = ~series.between(0.0, 1.0, inclusive="both")
        if bool(invalid.any()):
            raise ValueError(
                f"preprocess float column {column!r} must stay within [0, 1]; "
                f"bad rows={_first_bad_examples(invalid)}"
            )


def _validate_binary_int_columns(df: pd.DataFrame, *, expected_columns: tuple[str, ...]) -> None:
    for column in BINARY_INT_COLUMNS:
        if column not in expected_columns or column not in df.columns:
            continue
        series = df[column].astype("int64")
        invalid = ~series.isin((0, 1))
        if bool(invalid.any()):
            raise ValueError(
                f"preprocess int column {column!r} must be binary 0/1; bad rows={_first_bad_examples(invalid)}"
            )


def _validate_index_columns(df: pd.DataFrame, *, expected_columns: tuple[str, ...]) -> None:
    for column in ("user_idx", "item_idx"):
        if column not in expected_columns or column not in df.columns:
            continue
        series = df[column].astype("int64")
        invalid = series < 0
        if bool(invalid.any()):
            raise ValueError(
                f"preprocess index column {column!r} must be non-negative; bad rows={_first_bad_examples(invalid)}"
            )


def normalize_preprocess_dataframe(
    df: pd.DataFrame,
    *,
    require_split_indices: bool = False,
    require_domain: bool = False,
) -> pd.DataFrame:
    out = df.copy()
    expected_columns = expected_preprocess_column_order(
        require_split_indices=require_split_indices,
        require_domain=require_domain,
    )
    expected_fields = _expected_field_specs(
        require_split_indices=require_split_indices,
        require_domain=require_domain,
    )

    _validate_preprocess_header(
        out.columns,
        require_split_indices=require_split_indices,
        require_domain=require_domain,
        source_label="preprocess dataframe",
        require_exact_order=False,
    )

    for spec in PREPROCESS_FIELD_SPECS:
        if spec.name not in out.columns:
            continue
        if spec.kind == "text":
            out[spec.name] = _normalize_text_series(
                out[spec.name],
                strip=spec.name not in ("user", "item"),
            )
        elif spec.kind == "float":
            out[spec.name] = _normalize_numeric_series(out[spec.name], "float64")
        elif spec.kind == "int":
            out[spec.name] = _normalize_numeric_series(out[spec.name], "int64")

    _validate_non_empty_text_columns(out, expected_fields)
    _validate_allowed_text_values(out, expected_fields)
    _validate_unit_interval_columns(out, expected_columns=expected_columns)
    _validate_binary_int_columns(out, expected_columns=expected_columns)
    _validate_index_columns(out, expected_columns=expected_columns)
    return out


def read_preprocess_csv(
    path: str | Path,
    *,
    require_split_indices: bool = False,
    require_domain: bool = False,
) -> pd.DataFrame:
    header = _read_csv_header(path)
    _validate_preprocess_header(
        header,
        require_split_indices=require_split_indices,
        require_domain=require_domain,
        source_label=str(path),
        require_exact_order=True,
    )
    dtype_map = preprocess_csv_dtype_map(header)
    df = pd.read_csv(
        path,
        dtype=dtype_map,
        keep_default_na=False,
        na_values=[],
        low_memory=False,
    )
    return normalize_preprocess_dataframe(
        df,
        require_split_indices=require_split_indices,
        require_domain=require_domain,
    )


def write_preprocess_csv(
    df: pd.DataFrame,
    path: str | Path,
    *,
    require_split_indices: bool = False,
    require_domain: bool = False,
    column_order: Iterable[str] | None = None,
) -> None:
    out = normalize_preprocess_dataframe(
        df,
        require_split_indices=require_split_indices,
        require_domain=require_domain,
    )
    ordered = list(
        column_order
        if column_order is not None
        else expected_preprocess_column_order(
            require_split_indices=require_split_indices,
            require_domain=require_domain,
        )
    )
    missing = [column for column in ordered if column not in out.columns]
    if missing:
        raise ValueError(f"preprocess dataframe missing ordered columns: {missing}")
    extras = [column for column in out.columns if column not in ordered]
    if extras:
        raise ValueError(
            "preprocess dataframe contains non-contract columns that would be silently transported: "
            f"{extras}"
        )
    out = out[ordered]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def anchor_visible_explanation(text: object) -> str:
    normalized = str(text or "").strip()
    if normalized == EXPLANATION_PLACEHOLDER:
        return ""
    return normalized
