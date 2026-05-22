"""Step5 cache-aware reader for the Step4 RCR training export.

The loader is deliberately resolver-facing: callers pass the Step4 export,
manifest, index contract, expected fingerprint, and resolved Step5 loader
configuration.  Dry-run and bounded modes read only header/sample/bounded rows;
formal train builds or reuses a Step5-local filtered cache before pandas ever
loads a train table into memory.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from odcr_core.file_atomic import atomic_write_json
from odcr_core.index_contract import (
    GLOBAL_COL_ITEM,
    GLOBAL_COL_USER,
    INDEX_CONTRACT_FILENAME,
    ODCR_ROUTING_TRAIN_CSV,
    STEP4_RCR_REQUIRED_COLUMNS,
    load_index_contract,
    validate_split_indices,
)
from odcr_core.step4_export_validator import STEP4_EXPORT_MANIFEST
from odcr_core.step5_pool_sampler import (
    STEP5_POOL_MANIFEST,
    STEP5_POOL_SAMPLER_SCHEMA_VERSION,
    STEP5_POOLS_DIRNAME,
    STEP5_SAMPLING_CONTRACT,
    Step5PoolSamplerError,
    resolve_step5_pool_source,
    sample_effective_epochs_from_pools,
)
from odcr_core.training_checkpoint import file_fingerprint, stable_hash


STEP5_EXPORT_LOADER_SCHEMA_VERSION = "odcr_step5_export_loader/1"
STEP5_TRAIN_TABLE_CACHE_SCHEMA_VERSION = "odcr_step5_train_table_cache/1"
STEP5_TRAIN_TABLE_SEMANTIC_SCHEMA_VERSION = "odcr_step5_train_table_semantic/1"
STEP5_POOL_TRAIN_TABLE_CACHE_SCHEMA_VERSION = "odcr_step5_pool_train_table_cache/1"

STEP5_RUNTIME_DIAGNOSTIC_KEYS: frozenset[str] = frozenset(
    {
        "sampler_plan_time_s",
        "parquet_read_time_s",
        "parquet_metadata_time_s",
        "prompt_build_time_s",
        "sampler_compute_time_s",
        "tokenize_wall_time_s",
        "tokenize_time_s",
        "build_wall_time_s",
        "load_wall_time_s",
        "wall_time_s",
        "elapsed_s",
        "duration_sec",
    }
)

STEP5_TRAIN_REQUIRED_COLUMNS: tuple[str, ...] = (
    GLOBAL_COL_USER,
    GLOBAL_COL_ITEM,
    "rating",
    "domain",
    "clean_text",
    "train_keep",
    "sample_weight_hint",
    "route_scorer",
    "route_explainer",
    "cf_reliability_score",
    "content_retention_score",
    "style_shift_score",
    "rating_stability_score",
    "uncertainty_score",
    "confidence_bucket",
    "text_quality_score",
    "content_evidence",
    "style_evidence",
    "domain_style_anchor",
    "local_style_residual_hint",
    "polarity_anchor",
    "content_anchor_score",
    "style_anchor_score",
    "evidence_quality_prior",
)
STEP5_TRAIN_OPTIONAL_COLUMNS: tuple[str, ...] = (
    "sample_origin",
    "entropy_score",
    "route_reason_scorer",
    "route_reason_explainer",
    "preprocess_route_scorer_prior",
    "preprocess_route_explainer_prior",
    "html_entity_hit",
    "bad_tail_hit",
    "template_hit",
    "template_downweighted",
    "noisy_tail_downweighted",
    "repeat_tail_hit",
    "train_drop_reason",
    "step5_control_mode",
    "step5_control_source",
    "step5_control_contract_version",
    "gold_quality_score",
    "gold_quality_tier",
    "gold_quality_reasons",
    "hard_reject_flags",
    "coverage_bucket",
    "rating_bucket",
    "length_bucket",
    "domain_role",
    "recommended_sampling_weight",
    "text_quality_proxy",
    "consistency_proxy",
    "uncertainty_proxy",
    "control_coverage_proxy",
    "evidence_alignment_proxy",
    "coverage_diversity_proxy",
    "cf_tier_rating_stability_control",
    "cf_tier_step5_explanation",
    "cf_quality_score_rating_stability_control",
    "cf_quality_score_step5_explanation",
    "cf_tier_reasons_rating_stability_control",
    "cf_tier_reasons_step5_explanation",
    "cf_sampling_weight_rating_stability_control",
    "cf_sampling_weight_step5_explanation",
)
STEP5_TRAIN_VALIDATION_COLUMNS: tuple[str, ...] = tuple(
    dict.fromkeys((*STEP5_TRAIN_REQUIRED_COLUMNS, *STEP4_RCR_REQUIRED_COLUMNS))
)
STEP5_TRAIN_LOADER_COLUMNS: tuple[str, ...] = tuple(
    dict.fromkeys((*STEP5_TRAIN_VALIDATION_COLUMNS, *STEP5_TRAIN_OPTIONAL_COLUMNS))
)


class Step5ExportLoaderError(RuntimeError):
    """Raised when Step5 cannot safely consume the Step4 RCR export."""


@dataclass(frozen=True)
class Step5ExportSource:
    export_path: Path
    manifest_path: Path
    index_contract_path: Path
    index_contract: Mapping[str, Any]
    manifest: Mapping[str, Any]
    header_columns: tuple[str, ...]
    expected_sha256: str
    actual_sha256: str | None
    source_fingerprint: Mapping[str, Any]
    manifest_sha256: str
    index_contract_sha256: str
    required_columns: tuple[str, ...]
    fingerprint_payload: Mapping[str, Any]

    def to_summary(self) -> dict[str, Any]:
        return {
            "schema_version": STEP5_EXPORT_LOADER_SCHEMA_VERSION,
            "export_path": str(self.export_path),
            "manifest_path": str(self.manifest_path),
            "index_contract_path": str(self.index_contract_path),
            "expected_sha256": self.expected_sha256,
            "actual_sha256": self.actual_sha256,
            "source_size": self.source_fingerprint.get("size"),
            "source_mtime_ns": self.source_fingerprint.get("mtime_ns"),
            "manifest_sha256": self.manifest_sha256,
            "index_contract_sha256": self.index_contract_sha256,
            "required_columns_hash": stable_hash(list(self.required_columns)),
            "header_column_count": len(self.header_columns),
        }


@dataclass(frozen=True)
class Step5TrainTableLoadResult:
    train_df: pd.DataFrame
    audit_raw_df: pd.DataFrame
    source: Step5ExportSource
    cache_dir: Path | None
    cache_manifest_path: Path | None
    cache_hit: bool
    raw_row_count: int
    filtered_row_count: int
    raw_index_min_max: dict[str, list[int]] | None
    stats: Mapping[str, Any]

    def to_summary(self) -> dict[str, Any]:
        return {
            "schema_version": STEP5_TRAIN_TABLE_CACHE_SCHEMA_VERSION,
            "cache_dir": str(self.cache_dir) if self.cache_dir is not None else None,
            "cache_manifest_path": str(self.cache_manifest_path) if self.cache_manifest_path is not None else None,
            "cache_hit": bool(self.cache_hit),
            "raw_row_count": int(self.raw_row_count),
            "filtered_row_count": int(self.filtered_row_count),
            "raw_index_min_max": self.raw_index_min_max,
            "stats": dict(self.stats),
            "source": self.source.to_summary(),
        }

    def sample_plan_hash(self) -> str:
        return _sample_plan_hash(self.train_df)

    def source_table_compatibility_hash(self) -> str:
        return stable_hash(_source_table_semantic_payload(self.source))

    def to_runtime_diagnostics(self) -> dict[str, Any]:
        return _runtime_diagnostics_from_mapping(self.stats)

    def to_semantic_fingerprint_payload(self) -> dict[str, Any]:
        semantic_stats = _strip_runtime_diagnostics(self.stats)
        return {
            "schema_version": STEP5_TRAIN_TABLE_SEMANTIC_SCHEMA_VERSION,
            "source": _source_table_semantic_payload(self.source),
            "source_table_compatibility_hash": self.source_table_compatibility_hash(),
            "raw_row_count": int(self.raw_row_count),
            "filtered_row_count": int(self.filtered_row_count),
            "raw_index_min_max": dict(self.raw_index_min_max or {}),
            "sample_plan_hash": self.sample_plan_hash(),
            "stats": semantic_stats,
        }


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise Step5ExportLoaderError(f"{label} missing for Step5 export loader: {path}")
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise Step5ExportLoaderError(f"{label} invalid JSON for Step5 export loader: {path}: {exc}") from exc
    if not isinstance(obj, dict):
        raise Step5ExportLoaderError(f"{label} JSON root must be object for Step5 export loader: {path}")
    return obj


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _semantic_file_payload(path: Any, fingerprint: Mapping[str, Any] | None = None) -> dict[str, Any]:
    fp = dict(fingerprint or {})
    return {
        "exists": bool(fp.get("exists", True)),
        "is_file": bool(fp.get("is_file", True)),
        "size": int(fp.get("size") or 0),
        "sha256": str(fp.get("sha256") or fp.get("sample_sha256") or ""),
        "name": Path(str(path or fp.get("path") or "")).name,
    }


def _source_table_semantic_payload(source: Any) -> dict[str, Any]:
    manifest_path = getattr(source, "manifest_path", None)
    sampling_contract_path = getattr(source, "sampling_contract_path", None)
    index_contract_path = getattr(source, "index_contract_path", None)
    export_path = getattr(source, "export_path", None) or getattr(source, "source_full_export", None)
    manifest = getattr(source, "manifest", {}) or {}
    sampling_contract = getattr(source, "sampling_contract", {}) or {}
    summary = source.to_summary() if hasattr(source, "to_summary") else {}
    return {
        "schema_version": STEP5_TRAIN_TABLE_SEMANTIC_SCHEMA_VERSION,
        "export": _semantic_file_payload(export_path, file_fingerprint(export_path) if export_path else {}),
        "manifest": _semantic_file_payload(manifest_path, file_fingerprint(manifest_path) if manifest_path else {}),
        "index_contract": _semantic_file_payload(
            index_contract_path,
            file_fingerprint(index_contract_path) if index_contract_path else {},
        ),
        "pool_manifest": _semantic_file_payload(manifest_path, file_fingerprint(manifest_path) if manifest_path else {}),
        "sampling_contract": _semantic_file_payload(
            sampling_contract_path,
            file_fingerprint(sampling_contract_path) if sampling_contract_path else {},
        ),
        "pool_schema_version": summary.get("pool_schema_version") or manifest.get("schema_version"),
        "sampling_contract_schema_version": summary.get("sampling_contract_schema_version")
        or sampling_contract.get("schema_version"),
        "source_row_counts": dict(manifest.get("source_row_counts") or {}),
    }


def _strip_runtime_diagnostics(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(k): _strip_runtime_diagnostics(v)
            for k, v in value.items()
            if str(k) not in STEP5_RUNTIME_DIAGNOSTIC_KEYS
        }
    if isinstance(value, list):
        return [_strip_runtime_diagnostics(v) for v in value]
    if isinstance(value, tuple):
        return [_strip_runtime_diagnostics(v) for v in value]
    return value


def _runtime_diagnostics_from_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, item in value.items():
        if str(key) in STEP5_RUNTIME_DIAGNOSTIC_KEYS:
            out[str(key)] = item
        elif isinstance(item, Mapping):
            nested = _runtime_diagnostics_from_mapping(item)
            if nested:
                out[str(key)] = nested
        elif isinstance(item, list):
            nested_items = [
                _runtime_diagnostics_from_mapping(v)
                for v in item
                if isinstance(v, Mapping)
            ]
            nested_items = [v for v in nested_items if v]
            if nested_items:
                out[str(key)] = nested_items
    return out


def _sample_plan_hash(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return stable_hash({"schema_version": "odcr_step5_sample_plan_hash/1", "rows": 0})
    columns = [
        col
        for col in (
            "step5_plan_index",
            "sample_id",
            GLOBAL_COL_USER,
            GLOBAL_COL_ITEM,
            "task_head",
            "sampler_component",
            "sampler_tier",
            "effective_epoch",
            "route_scorer",
            "route_explainer",
            "step5_prompt_template_id",
            "step5_prompt_version",
            "step5_prompt_instance_id",
        )
        if col in df.columns
    ]
    if not columns:
        return stable_hash({"schema_version": "odcr_step5_sample_plan_hash/1", "rows": int(len(df))})
    hashed = pd.util.hash_pandas_object(df.loc[:, columns], index=True).to_numpy(dtype="uint64", copy=False)
    digest = hashlib.sha256()
    digest.update(str(tuple(columns)).encode("utf-8"))
    digest.update(str(len(df)).encode("utf-8"))
    digest.update(hashed.tobytes())
    return digest.hexdigest()[:32]


def _resolve_source_paths(
    export_path: str | Path,
    *,
    index_contract_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> tuple[Path, Path, Path]:
    export = Path(export_path).expanduser().resolve()
    if not export.is_file():
        raise Step5ExportLoaderError(
            f"Step5 export loader missing Step4 RCR export: {export}. "
            "Use ./odcr step5 through the upstream resolver; do not pass a manual CSV."
        )
    base = export.parent
    contract = Path(index_contract_path).expanduser().resolve() if index_contract_path else base / INDEX_CONTRACT_FILENAME
    manifest = Path(manifest_path).expanduser().resolve() if manifest_path else base / STEP4_EXPORT_MANIFEST
    return export, contract, manifest


def _read_header(export_path: Path) -> tuple[str, ...]:
    try:
        header = pd.read_csv(export_path, nrows=0)
    except Exception as exc:
        raise Step5ExportLoaderError(f"Step5 export loader failed to read CSV header: {export_path}: {exc}") from exc
    return tuple(str(c) for c in header.columns)


def _required_columns(required_columns: Sequence[str] | None) -> tuple[str, ...]:
    required = tuple(dict.fromkeys(str(c) for c in (required_columns or STEP5_TRAIN_VALIDATION_COLUMNS) if str(c)))
    if not required:
        raise Step5ExportLoaderError("Step5 export loader requires a non-empty required column contract.")
    return required


def _check_required_columns(header: Sequence[str], required: Sequence[str], *, ctx: str) -> None:
    available = set(str(c) for c in header)
    missing = [str(c) for c in required if str(c) not in available]
    if missing:
        raise Step5ExportLoaderError(
            f"{ctx} missing required Step4 RCR export fields for Step5: {', '.join(missing)}. "
            "Failure stage=Step5 export loader/header validation; producer must be the canonical "
            "Step4 odcr_routing_train.csv selected by upstream_resolver, not a hand-written CSV."
        )


def _contract_train_fingerprint(index_contract: Mapping[str, Any]) -> Mapping[str, Any]:
    fps = index_contract.get("fingerprints")
    if not isinstance(fps, Mapping):
        raise Step5ExportLoaderError("index_contract missing fingerprints for Step5 export loader.")
    train_fp = fps.get("train_csv")
    if not isinstance(train_fp, Mapping):
        raise Step5ExportLoaderError("index_contract.fingerprints.train_csv missing for Step5 export loader.")
    return train_fp


def validate_step5_export_source(
    export_path: str | Path,
    *,
    index_contract_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    index_contract: Mapping[str, Any] | None = None,
    required_columns: Sequence[str] | None = None,
    verify_sha256: bool = False,
    mode: str = "validate_only",
) -> Step5ExportSource:
    """Validate a Step4 export without materializing the train table."""

    export, contract_path, manifest = _resolve_source_paths(
        export_path,
        index_contract_path=index_contract_path,
        manifest_path=manifest_path,
    )
    contract = dict(index_contract) if index_contract is not None else load_index_contract(str(contract_path))
    manifest_payload = _load_json(manifest, label="Step4 export manifest")
    required = _required_columns(required_columns)
    header = _read_header(export)
    _check_required_columns(header, required, ctx=f"Step5 {mode}")

    train_fp = _contract_train_fingerprint(contract)
    expected_sha = str(train_fp.get("sha256") or "").strip()
    if not expected_sha:
        raise Step5ExportLoaderError(
            "index_contract.fingerprints.train_csv.sha256 missing; Step5 refuses a Step4 export "
            "without sha256 lineage. Rerun Step4 through ./odcr."
        )
    stat = export.stat()
    if train_fp.get("size") is not None and int(train_fp["size"]) != int(stat.st_size):
        raise Step5ExportLoaderError(
            f"Step5 export source stale: size mismatch index_contract={train_fp.get('size')} actual={int(stat.st_size)}."
        )
    if train_fp.get("mtime_ns") is not None:
        actual_mtime_ns = int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9)))
        if int(train_fp["mtime_ns"]) != actual_mtime_ns:
            raise Step5ExportLoaderError("Step5 export source stale: mtime_ns mismatch with index_contract.")

    actual_sha: str | None = None
    if verify_sha256:
        actual_sha = _file_sha256(export)
        if actual_sha != expected_sha:
            raise Step5ExportLoaderError(
                f"Step5 export source sha256 mismatch: index_contract={expected_sha} actual={actual_sha}."
            )

    manifest_sha = _file_sha256(manifest)
    contract_sha = _file_sha256(contract_path)
    source_fp = file_fingerprint(export, sample_only=not verify_sha256)
    if verify_sha256:
        source_fp = {**source_fp, "sha256": actual_sha}
    fingerprint_payload = {
        "schema_version": STEP5_EXPORT_LOADER_SCHEMA_VERSION,
        "mode": str(mode),
        "export_path": str(export),
        "source_size": int(stat.st_size),
        "source_mtime_ns": int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9))),
        "source_sha256": expected_sha,
        "actual_sha256_verified": bool(verify_sha256),
        "manifest_path": str(manifest),
        "manifest_sha256": manifest_sha,
        "index_contract_path": str(contract_path),
        "index_contract_sha256": contract_sha,
        "index_contract_schema_version": contract.get("schema_version"),
        "manifest_schema_version": manifest_payload.get("schema_version"),
        "step4_export_lineage_hash": (
            (contract.get("step4_export_lineage") or manifest_payload.get("step4_export_lineage") or {}).get("lineage_hash")
            if isinstance(contract.get("step4_export_lineage") or manifest_payload.get("step4_export_lineage"), Mapping)
            else None
        ),
        "required_columns_hash": stable_hash(list(required)),
        "required_columns": list(required),
    }
    return Step5ExportSource(
        export_path=export,
        manifest_path=manifest,
        index_contract_path=contract_path,
        index_contract=contract,
        manifest=manifest_payload,
        header_columns=header,
        expected_sha256=expected_sha,
        actual_sha256=actual_sha,
        source_fingerprint=source_fp,
        manifest_sha256=manifest_sha,
        index_contract_sha256=contract_sha,
        required_columns=required,
        fingerprint_payload=fingerprint_payload,
    )


def _available_read_columns(header: Sequence[str]) -> list[str]:
    header_set = set(header)
    return [c for c in STEP5_TRAIN_LOADER_COLUMNS if c in header_set]


def _filter_step5_train_df(df: pd.DataFrame, *, explainer_only_multiplier: float) -> pd.DataFrame:
    for col in ("train_keep", "route_scorer", "route_explainer", "sample_weight_hint", "uncertainty_score", "confidence_bucket"):
        if col not in df.columns:
            raise Step5ExportLoaderError(
                f"Step5 export loader/filter missing {col}; Step4 RCR posterior fields must be present before collate."
            )
    keep = pd.to_numeric(df["train_keep"], errors="raise").astype(int) == 1
    clean = df["clean_text"].fillna("").astype(str).str.strip() != ""
    route_scorer = pd.to_numeric(df["route_scorer"], errors="raise").astype(int)
    route_explainer = pd.to_numeric(df["route_explainer"], errors="raise").astype(int)
    routed = (route_scorer == 1) | (route_explainer == 1)
    out = df.loc[keep & clean & routed].copy()
    if out.empty:
        return out.reset_index(drop=True)
    out["sample_weight_hint"] = pd.to_numeric(out["sample_weight_hint"], errors="raise").astype(float)
    out["route_scorer"] = pd.to_numeric(out["route_scorer"], errors="raise").astype(int)
    out["route_explainer"] = pd.to_numeric(out["route_explainer"], errors="raise").astype(int)
    out["uncertainty_score"] = pd.to_numeric(out["uncertainty_score"], errors="raise").astype(float)
    out["confidence_bucket"] = pd.to_numeric(out["confidence_bucket"], errors="raise").astype(float)
    out["sample_weight_hint"] = out["sample_weight_hint"].where(
        out["route_scorer"] == 1,
        out["sample_weight_hint"] * float(explainer_only_multiplier),
    )
    return out.reset_index(drop=True)


def _idx_minmax_from_df(df: pd.DataFrame) -> dict[str, list[int]] | None:
    if df.empty or GLOBAL_COL_USER not in df.columns or GLOBAL_COL_ITEM not in df.columns:
        return None
    return {
        "user_idx_global": [int(df[GLOBAL_COL_USER].min()), int(df[GLOBAL_COL_USER].max())],
        "item_idx_global": [int(df[GLOBAL_COL_ITEM].min()), int(df[GLOBAL_COL_ITEM].max())],
    }


def _merge_idx_minmax(acc: dict[str, list[int]] | None, cur: dict[str, list[int]] | None) -> dict[str, list[int]] | None:
    if cur is None:
        return acc
    if acc is None:
        return {k: list(v) for k, v in cur.items()}
    for key, val in cur.items():
        acc[key] = [min(int(acc[key][0]), int(val[0])), max(int(acc[key][1]), int(val[1]))]
    return acc


def _cache_manifest_payload(
    *,
    source: Step5ExportSource,
    cache_key: str,
    cache_csv: Path,
    raw_row_count: int,
    filtered_row_count: int,
    raw_index_min_max: Mapping[str, Any] | None,
    explainer_only_multiplier: float,
    stats: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": STEP5_TRAIN_TABLE_CACHE_SCHEMA_VERSION,
        "cache_key": cache_key,
        "cache_csv": str(cache_csv),
        "raw_row_count": int(raw_row_count),
        "filtered_row_count": int(filtered_row_count),
        "raw_index_min_max": dict(raw_index_min_max or {}),
        "explainer_only_multiplier": float(explainer_only_multiplier),
        "source": source.to_summary(),
        "fingerprint_payload": dict(source.fingerprint_payload),
        "stats": dict(stats),
    }


def _cache_key(source: Step5ExportSource, *, explainer_only_multiplier: float) -> str:
    return stable_hash(
        {
            "schema_version": STEP5_TRAIN_TABLE_CACHE_SCHEMA_VERSION,
            "source": dict(source.fingerprint_payload),
            "explainer_only_multiplier": float(explainer_only_multiplier),
            "filter": "train_keep==1 and clean_text nonempty and (route_scorer==1 or route_explainer==1)",
            "sample_id_policy": "sequential_after_filter",
        }
    )


def _pool_sampler_tuning_identity(tuning_config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Content-affecting Step5 pool sampler knobs only.

    Optimizer, loss, decode, and runtime fields are lineage-only for the sampled
    table cache.  They do not alter the selected rows or rendered prompt text.
    """

    tuning = dict(tuning_config or {})
    return {
        "selected_budget_candidate": tuning.get("selected_budget_candidate"),
        "batch_candidate": tuning.get("batch_candidate"),
        "effective_samples": dict(tuning.get("effective_samples") or {})
        if isinstance(tuning.get("effective_samples"), Mapping)
        else {},
    }


def _pool_sample_cache_identity(
    *,
    pool_source: Any,
    sampler_config: Mapping[str, Any] | None,
    batch_candidates_config: Mapping[str, Any] | None,
    tuning_config: Mapping[str, Any] | None,
    mode: str,
    task_head: str,
    bounded_max_rows: int | None,
    columns: Sequence[str] | None,
) -> dict[str, Any]:
    return {
        "schema_version": STEP5_POOL_TRAIN_TABLE_CACHE_SCHEMA_VERSION,
        "source": _source_table_semantic_payload(pool_source),
        "sampler_config": dict(sampler_config or {}),
        "batch_candidates_config": dict(batch_candidates_config or {}),
        "tuning_identity": _pool_sampler_tuning_identity(tuning_config),
        "mode": str(mode),
        "task_head": str(task_head),
        "bounded_max_rows": int(bounded_max_rows) if bounded_max_rows is not None else None,
        "columns": [str(col) for col in columns] if columns is not None else None,
    }


def _pool_sample_cache_lineage(
    *,
    sampler_config: Mapping[str, Any] | None,
    batch_candidates_config: Mapping[str, Any] | None,
    tuning_config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema_version": "odcr_step5_pool_train_table_cache_lineage/1",
        "sampler_config": dict(sampler_config or {}),
        "batch_candidates_config": dict(batch_candidates_config or {}),
        "tuning_config": dict(tuning_config or {}),
    }


def _pool_sample_cache_key(identity: Mapping[str, Any]) -> str:
    return stable_hash(identity)


def _pool_sample_cache_manifest_payload(
    *,
    cache_key: str,
    cache_path: Path,
    identity: Mapping[str, Any],
    lineage: Mapping[str, Any],
    row_count: int,
    raw_row_count: int,
    stats: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": STEP5_POOL_TRAIN_TABLE_CACHE_SCHEMA_VERSION,
        "cache_key": str(cache_key),
        "cache_path": str(cache_path),
        "cache_identity": dict(identity),
        "cache_identity_hash": stable_hash(identity),
        "lineage_metadata": dict(lineage),
        "lineage_metadata_hash": stable_hash(lineage),
        "row_count": int(row_count),
        "raw_row_count": int(raw_row_count),
        "stats": dict(stats),
    }


def _load_pool_sample_cache(
    cache_dir: Path,
    *,
    expected_identity: Mapping[str, Any],
    expected_cache_key: str,
    source: Step5ExportSource,
    pool_source: Any,
    stale_policy: str,
) -> Step5TrainTableLoadResult | None:
    manifest_path = cache_dir / "cache_manifest.json"
    cache_path = cache_dir / "train_sampled.parquet"
    if not manifest_path.is_file() or not cache_path.is_file():
        return None
    policy = str(stale_policy or "rebuild").strip().lower()

    def _stale(message: str) -> Step5TrainTableLoadResult | None:
        if policy == "rebuild":
            return None
        raise Step5ExportLoaderError(message)

    manifest = _load_json(manifest_path, label="Step5 pool train table cache manifest")
    if str(manifest.get("schema_version")) != STEP5_POOL_TRAIN_TABLE_CACHE_SCHEMA_VERSION:
        return _stale(f"Step5 pool train table cache schema mismatch at {manifest_path}.")
    if str(manifest.get("cache_key") or "") != str(expected_cache_key):
        return _stale(f"Step5 pool train table cache key mismatch at {manifest_path}.")
    if dict(manifest.get("cache_identity") or {}) != dict(expected_identity):
        return _stale(f"Step5 pool train table cache identity mismatch at {manifest_path}.")
    df = pd.read_parquet(cache_path)
    _check_required_columns(tuple(df.columns), source.required_columns, ctx="Step5 formal_train pool cache")
    row_count = int(manifest.get("row_count", len(df)))
    if row_count != len(df):
        return _stale("Step5 pool train table cache row count mismatch.")
    stats = dict(manifest.get("stats") or {})
    stats["cache_load_wall_time_s"] = float(stats.get("cache_load_wall_time_s") or 0.0)
    return Step5TrainTableLoadResult(
        train_df=df,
        audit_raw_df=df.head(16).copy(),
        source=pool_source,  # type: ignore[arg-type]
        cache_dir=cache_dir,
        cache_manifest_path=manifest_path,
        cache_hit=True,
        raw_row_count=int(manifest.get("raw_row_count", len(df))),
        filtered_row_count=int(len(df)),
        raw_index_min_max=_idx_minmax_from_df(df),
        stats=stats,
    )


def _load_cache(
    cache_dir: Path,
    *,
    expected_manifest: Mapping[str, Any],
    source: Step5ExportSource,
    stale_policy: str,
) -> Step5TrainTableLoadResult | None:
    manifest_path = cache_dir / "cache_manifest.json"
    cache_csv = cache_dir / "train_filtered.csv"
    if not manifest_path.is_file() or not cache_csv.is_file():
        return None
    policy = str(stale_policy or "rebuild").strip().lower()

    def _stale(message: str) -> Step5TrainTableLoadResult | None:
        if policy == "rebuild":
            return None
        raise Step5ExportLoaderError(message)

    manifest = _load_json(manifest_path, label="Step5 train table cache manifest")
    expected_source = dict(expected_manifest.get("source") or {})
    actual_source = dict(manifest.get("source") or {})
    if actual_source != expected_source:
        return _stale(
            f"Step5 train table cache stale/corrupt: source fingerprint mismatch at {manifest_path}."
        )
    if str(manifest.get("schema_version")) != STEP5_TRAIN_TABLE_CACHE_SCHEMA_VERSION:
        return _stale(f"Step5 train table cache schema mismatch at {manifest_path}.")
    df = pd.read_csv(cache_csv)
    _check_required_columns(tuple(df.columns), source.required_columns, ctx="Step5 formal_train cache")
    filtered_row_count = int(manifest.get("filtered_row_count", len(df)))
    if filtered_row_count != len(df):
        return _stale("Step5 train table cache row count mismatch.")
    audit_raw = pd.read_csv(source.export_path, usecols=_available_read_columns(source.header_columns), nrows=16)
    return Step5TrainTableLoadResult(
        train_df=df,
        audit_raw_df=audit_raw,
        source=source,
        cache_dir=cache_dir,
        cache_manifest_path=manifest_path,
        cache_hit=True,
        raw_row_count=int(manifest.get("raw_row_count", 0)),
        filtered_row_count=filtered_row_count,
        raw_index_min_max=dict(manifest.get("raw_index_min_max") or {}) or None,
        stats=dict(manifest.get("stats") or {}),
    )


def load_step5_train_table(
    export_path: str | Path,
    *,
    cache_root: str | Path | None,
    index_contract_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    index_contract: Mapping[str, Any] | None = None,
    required_columns: Sequence[str] | None = None,
    mode: str = "formal_train",
    explainer_only_multiplier: float = 1.0,
    cache_enabled: bool = True,
    chunk_rows: int = 100_000,
    validate_sample_rows: int = 16,
    bounded_max_rows: int | None = None,
    verify_sha256: bool = True,
    stale_policy: str = "rebuild",
    validation_ctx: Mapping[str, Any] | None = None,
) -> Step5TrainTableLoadResult:
    """Load Step5 train rows without a naked full Step4 ``pd.read_csv``."""

    mode_norm = str(mode).strip().lower()
    if mode_norm not in {"validate_only", "bounded", "formal_prepare", "formal_train"}:
        raise Step5ExportLoaderError(f"unsupported Step5 export loader mode: {mode!r}")
    stale_policy_norm = str(stale_policy or "rebuild").strip().lower()
    if stale_policy_norm not in {"rebuild", "fail_fast"}:
        raise Step5ExportLoaderError(
            f"unsupported step5.export_loader.stale_policy: {stale_policy!r}; expected rebuild or fail_fast."
        )
    source = validate_step5_export_source(
        export_path,
        index_contract_path=index_contract_path,
        manifest_path=manifest_path,
        index_contract=index_contract,
        required_columns=required_columns,
        verify_sha256=bool(verify_sha256 and mode_norm in {"formal_prepare", "formal_train"}),
        mode=mode_norm,
    )
    read_cols = _available_read_columns(source.header_columns)
    sample_rows = max(1, int(validate_sample_rows))
    audit_raw = pd.read_csv(source.export_path, usecols=read_cols, nrows=sample_rows)
    if mode_norm == "validate_only":
        return Step5TrainTableLoadResult(
            train_df=pd.DataFrame(columns=read_cols),
            audit_raw_df=audit_raw,
            source=source,
            cache_dir=None,
            cache_manifest_path=None,
            cache_hit=False,
            raw_row_count=int(((source.manifest.get("row_counts") or {}).get("total_rows") or 0)),
            filtered_row_count=0,
            raw_index_min_max=None,
            stats={"mode": "validate_only", "full_csv_parse": False},
        )

    if mode_norm == "bounded":
        n = max(1, int(bounded_max_rows or sample_rows))
        df = pd.read_csv(source.export_path, usecols=read_cols, nrows=n)
        validate_split_indices(df, source.index_contract, "train", ctx=dict(validation_ctx or {}))
        filtered = _filter_step5_train_df(df, explainer_only_multiplier=float(explainer_only_multiplier))
        filtered["sample_id"] = range(len(filtered))
        return Step5TrainTableLoadResult(
            train_df=filtered,
            audit_raw_df=df.head(sample_rows).copy(),
            source=source,
            cache_dir=None,
            cache_manifest_path=None,
            cache_hit=False,
            raw_row_count=int(len(df)),
            filtered_row_count=int(len(filtered)),
            raw_index_min_max=_idx_minmax_from_df(df),
            stats={"mode": "bounded", "bounded_max_rows": n, "full_csv_parse": False},
        )

    root = Path(cache_root).expanduser().resolve() if cache_root is not None else source.export_path.parent / ".step5_cache"
    key = _cache_key(source, explainer_only_multiplier=float(explainer_only_multiplier))
    cache_dir = root / key
    cache_csv = cache_dir / "train_filtered.csv"
    expected_manifest = _cache_manifest_payload(
        source=source,
        cache_key=key,
        cache_csv=cache_csv,
        raw_row_count=0,
        filtered_row_count=0,
        raw_index_min_max=None,
        explainer_only_multiplier=float(explainer_only_multiplier),
        stats={},
    )
    if cache_enabled:
        cached = _load_cache(
            cache_dir,
            expected_manifest=expected_manifest,
            source=source,
            stale_policy=stale_policy_norm,
        )
        if cached is not None:
            return cached
    elif mode_norm == "formal_train":
        raise Step5ExportLoaderError("step5.export_loader.cache_enabled=false is not allowed for formal_train.")

    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = cache_dir.with_name(cache_dir.name + ".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=False)
    tmp_csv = tmp_dir / "train_filtered.csv"

    raw_rows = 0
    filtered_rows = 0
    raw_index_min_max: dict[str, list[int]] | None = None
    sample_origin_counts: dict[str, int] = {}
    train_keep_zero = 0
    chunks_for_return: list[pd.DataFrame] = []
    first_write = True
    ctx = dict(validation_ctx or {})
    try:
        for chunk in pd.read_csv(source.export_path, usecols=read_cols, chunksize=max(1, int(chunk_rows))):
            raw_rows += int(len(chunk))
            validate_split_indices(chunk, source.index_contract, "train", ctx=ctx)
            raw_index_min_max = _merge_idx_minmax(raw_index_min_max, _idx_minmax_from_df(chunk))
            if "sample_origin" in chunk.columns:
                for key2, val in chunk["sample_origin"].fillna("").astype(str).value_counts().items():
                    sample_origin_counts[str(key2)] = sample_origin_counts.get(str(key2), 0) + int(val)
            if "train_keep" in chunk.columns:
                train_keep_zero += int((pd.to_numeric(chunk["train_keep"], errors="raise").astype(int) == 0).sum())
            filtered = _filter_step5_train_df(
                chunk,
                explainer_only_multiplier=float(explainer_only_multiplier),
            )
            if filtered.empty:
                continue
            filtered["sample_id"] = range(filtered_rows, filtered_rows + len(filtered))
            filtered_rows += int(len(filtered))
            filtered.to_csv(tmp_csv, mode="w" if first_write else "a", header=first_write, index=False)
            first_write = False
            chunks_for_return.append(filtered)
        if filtered_rows <= 0:
            raise Step5ExportLoaderError(
                "Step5 formal_train cache build found zero rows after train_keep/route/clean_text filtering; "
                "check Step4 RCR thresholds and export manifest."
            )
        if not tmp_csv.is_file():
            raise Step5ExportLoaderError("Step5 formal_train cache build did not write train_filtered.csv.")
        os.replace(str(tmp_csv), str(cache_csv))
        stats = {
            "mode": mode_norm,
            "full_csv_parse": False,
            "chunk_rows": int(chunk_rows),
            "sample_origin_counts": sample_origin_counts,
            "train_keep_zero": int(train_keep_zero),
        }
        manifest_payload = _cache_manifest_payload(
            source=source,
            cache_key=key,
            cache_csv=cache_csv,
            raw_row_count=raw_rows,
            filtered_row_count=filtered_rows,
            raw_index_min_max=raw_index_min_max,
            explainer_only_multiplier=float(explainer_only_multiplier),
            stats=stats,
        )
        atomic_write_json(cache_dir / "cache_manifest.json", manifest_payload)
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

    if chunks_for_return:
        train_df = pd.concat(chunks_for_return, ignore_index=True)
    else:
        train_df = pd.read_csv(cache_csv)
    return Step5TrainTableLoadResult(
        train_df=train_df,
        audit_raw_df=audit_raw,
        source=source,
        cache_dir=cache_dir,
        cache_manifest_path=cache_dir / "cache_manifest.json",
        cache_hit=False,
        raw_row_count=raw_rows,
        filtered_row_count=filtered_rows,
        raw_index_min_max=raw_index_min_max,
        stats=stats,
    )


def _parse_sampler_config(raw: Mapping[str, Any] | str | None) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, str):
        if not raw.strip():
            return {}
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise Step5ExportLoaderError("step5.sampler resolved JSON is invalid") from exc
        if not isinstance(obj, dict):
            raise Step5ExportLoaderError("step5.sampler resolved JSON must be an object")
        return obj
    return dict(raw)


def load_step5_pool_train_table(
    export_path: str | Path,
    *,
    cache_root: str | Path | None = None,
    cache_enabled: bool = True,
    index_contract_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    index_contract: Mapping[str, Any] | None = None,
    required_columns: Sequence[str] | None = None,
    mode: str = "formal_train",
    sampler_config: Mapping[str, Any] | str | None = None,
    batch_candidates_config: Mapping[str, Any] | str | None = None,
    tuning_config: Mapping[str, Any] | str | None = None,
    task_head: str = "explanation",
    bounded_max_rows: int | None = None,
    validate_sample_rows: int = 16,
    verify_sha256: bool = True,
    stale_policy: str = "rebuild",
    validation_ctx: Mapping[str, Any] | None = None,
) -> Step5TrainTableLoadResult:
    """Load Step5 train rows from Step4 pools and the sampling contract.

    Full audit and old gold-heavy dedicated exports are intentionally rejected
    as default training inputs by requiring ``step5_pools`` next to the selected
    Step4 export.
    """

    mode_norm = str(mode).strip().lower()
    if mode_norm not in {"validate_only", "bounded", "formal_prepare", "formal_train"}:
        raise Step5ExportLoaderError(f"unsupported Step5 pool loader mode: {mode!r}")
    stale_policy_norm = str(stale_policy or "rebuild").strip().lower()
    if stale_policy_norm not in {"rebuild", "fail_fast"}:
        raise Step5ExportLoaderError(
            f"unsupported step5.export_loader.stale_policy: {stale_policy!r}; expected rebuild or fail_fast."
        )
    source = validate_step5_export_source(
        export_path,
        index_contract_path=index_contract_path,
        manifest_path=manifest_path,
        index_contract=index_contract,
        required_columns=required_columns,
        verify_sha256=bool(verify_sha256 and mode_norm in {"formal_prepare", "formal_train"}),
        mode=mode_norm,
    )
    try:
        pool_source = resolve_step5_pool_source(step4_run_dir=source.export_path.parent)
    except Step5PoolSamplerError as exc:
        raise Step5ExportLoaderError(
            "Step5 default train input now requires Step4 RCR pools: "
            f"{source.export_path.parent / STEP5_POOLS_DIRNAME / STEP5_POOL_MANIFEST} and "
            f"{source.export_path.parent / STEP5_POOLS_DIRNAME / STEP5_SAMPLING_CONTRACT}. "
            "Full audit and legacy gold-heavy dedicated exports are audit/ablation only."
        ) from exc
    sampler = _parse_sampler_config(sampler_config)
    if not sampler:
        raise Step5ExportLoaderError("Step5 pool loader requires resolved step5.sampler config.")
    batch_candidates = _parse_sampler_config(batch_candidates_config)
    tuning = _parse_sampler_config(tuning_config)
    if mode_norm == "validate_only":
        sample = pd.DataFrame()
        stats = {
            "schema_version": STEP5_POOL_SAMPLER_SCHEMA_VERSION,
            "mode": "validate_only",
            "full_csv_parse": False,
            "pool_manifest_required": True,
            "pool_manifest_path": str(pool_source.manifest_path),
            "sampling_contract_path": str(pool_source.sampling_contract_path),
            "full_audit_default_train_forbidden": True,
            "legacy_gold_heavy_exports_rejected_by_default": True,
        }
        return Step5TrainTableLoadResult(
            train_df=pd.DataFrame(columns=_available_read_columns(source.header_columns)),
            audit_raw_df=sample,
            source=pool_source,  # type: ignore[arg-type]
            cache_dir=None,
            cache_manifest_path=None,
            cache_hit=False,
            raw_row_count=int(((pool_source.manifest.get("source_row_counts") or {}).get("total_rows") or 0)),
            filtered_row_count=0,
            raw_index_min_max=None,
            stats=stats,
        )
    # Pool parquet files extend the canonical Step4 CSV contract with derived
    # quality/tier fields, so do not restrict reads to the source CSV header.
    columns = tuple(dict.fromkeys((*STEP5_TRAIN_LOADER_COLUMNS, "sample_id", "task_head", "effective_epoch")))
    cache_dir: Path | None = None
    cache_manifest_path: Path | None = None
    cache_hit = False
    cache_identity: dict[str, Any] | None = None
    cache_lineage: dict[str, Any] | None = None
    cache_key: str | None = None
    if mode_norm == "formal_train" and bool(cache_enabled):
        root = Path(cache_root).expanduser().resolve() if cache_root is not None else source.export_path.parent / ".step5_pool_cache"
        cache_identity = _pool_sample_cache_identity(
            pool_source=pool_source,
            sampler_config=sampler,
            batch_candidates_config=batch_candidates,
            tuning_config=tuning,
            mode=mode_norm,
            task_head=str(task_head),
            bounded_max_rows=bounded_max_rows,
            columns=columns,
        )
        cache_lineage = _pool_sample_cache_lineage(
            sampler_config=sampler,
            batch_candidates_config=batch_candidates,
            tuning_config=tuning,
        )
        cache_key = _pool_sample_cache_key(cache_identity)
        cache_dir = root / str(cache_key)
        cached = _load_pool_sample_cache(
            cache_dir,
            expected_identity=cache_identity,
            expected_cache_key=cache_key,
            source=source,
            pool_source=pool_source,
            stale_policy=stale_policy_norm,
        )
        if cached is not None:
            return cached
    elif mode_norm == "formal_train" and not bool(cache_enabled):
        raise Step5ExportLoaderError("step5.export_loader.cache_enabled=false is not allowed for formal_train.")
    try:
        sampled = sample_effective_epochs_from_pools(
            pool_source,
            sampler_config=sampler,
            batch_candidates_config=batch_candidates,
            tuning_config=tuning,
            mode=mode_norm,
            task_head=str(task_head),
            bounded_max_rows=bounded_max_rows,
            columns=columns,
        )
    except Step5PoolSamplerError as exc:
        raise Step5ExportLoaderError(str(exc)) from exc
    train_df = sampled.train_df.reset_index(drop=True)
    if train_df.empty:
        raise Step5ExportLoaderError("Step5 pool sampler produced zero rows; check pool distribution and sampling contract.")
    validate_split_indices(train_df, source.index_contract, "train", ctx=dict(validation_ctx or {}))
    stats = dict(sampled.stats)
    if cache_dir is not None and cache_identity is not None and cache_lineage is not None and cache_key is not None:
        cache_dir.parent.mkdir(parents=True, exist_ok=True)
        tmp_dir = cache_dir.with_name(f"{cache_dir.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=False)
        tmp_path = tmp_dir / "train_sampled.parquet"
        try:
            train_df.to_parquet(tmp_path, index=False)
            manifest_payload = _pool_sample_cache_manifest_payload(
                cache_key=cache_key,
                cache_path=cache_dir / "train_sampled.parquet",
                identity=cache_identity,
                lineage=cache_lineage,
                row_count=len(train_df),
                raw_row_count=sampled.raw_row_count,
                stats=stats,
            )
            atomic_write_json(tmp_dir / "cache_manifest.json", manifest_payload)
            for attempt in range(2):
                if cache_dir.exists():
                    if cache_dir.is_dir():
                        shutil.rmtree(cache_dir, ignore_errors=True)
                    else:
                        try:
                            cache_dir.unlink()
                        except FileNotFoundError:
                            pass
                try:
                    os.replace(str(tmp_dir), str(cache_dir))
                    break
                except FileExistsError:
                    if attempt >= 1:
                        raise
            cache_manifest_path = cache_dir / "cache_manifest.json"
        finally:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
    return Step5TrainTableLoadResult(
        train_df=train_df,
        audit_raw_df=sampled.audit_raw_df,
        source=pool_source,  # type: ignore[arg-type]
        cache_dir=cache_dir,
        cache_manifest_path=cache_manifest_path,
        cache_hit=cache_hit,
        raw_row_count=int(sampled.raw_row_count),
        filtered_row_count=int(sampled.filtered_row_count),
        raw_index_min_max=_idx_minmax_from_df(train_df),
        stats={**stats, "full_csv_parse": False},
    )


def resolved_step5_export_paths(cfg: Any) -> tuple[Path, Path, Path]:
    """Resolve Step5 export paths from the unified config object."""

    from odcr_core import path_layout

    if getattr(cfg, "step4_run", None):
        export = (
            path_layout.get_train_step4_run_root(
                cfg.repo_root,
                int(cfg.task_id),
                str(getattr(cfg, "iteration_id", "v1") or "v1"),
                str(cfg.step4_run),
            )
            / ODCR_ROUTING_TRAIN_CSV
        )
    else:
        from odcr_core.artifacts import train_csv_path

        export = train_csv_path(cfg)
    export = Path(export).expanduser().resolve()
    return export, export.parent / INDEX_CONTRACT_FILENAME, export.parent / STEP4_EXPORT_MANIFEST


def validate_step5_export_for_resolved_config(
    cfg: Any,
    *,
    mode: str = "validate_only",
    verify_sha256: bool = False,
    validate_sample_rows: int = 16,
) -> dict[str, Any]:
    """Dry-run/show helper: validate Step4 handoff without full CSV parse or writes."""

    export, contract, manifest = resolved_step5_export_paths(cfg)
    source = validate_step5_export_source(
        export,
        index_contract_path=contract,
        manifest_path=manifest,
        required_columns=STEP5_TRAIN_VALIDATION_COLUMNS,
        verify_sha256=verify_sha256,
        mode=mode,
    )
    try:
        pool_source = resolve_step5_pool_source(step4_run_dir=export.parent)
    except Step5PoolSamplerError as exc:
        raise Step5ExportLoaderError(
            "Step5 show/dry-run requires Step4 pool manifest and sampling contract; "
            "legacy full audit or gold-heavy dedicated tables are no longer default train inputs."
        ) from exc
    return {
        **source.to_summary(),
        "pool_source": pool_source.to_summary(),
        "mode": str(mode),
        "sample_rows_read": 0,
        "full_csv_parse": False,
        "formal_namespace_write": False,
        "full_audit_default_train_forbidden": True,
        "legacy_gold_heavy_exports_rejected_by_default": True,
    }


__all__ = [
    "STEP5_EXPORT_LOADER_SCHEMA_VERSION",
    "STEP5_TRAIN_TABLE_CACHE_SCHEMA_VERSION",
    "STEP5_TRAIN_LOADER_COLUMNS",
    "STEP5_TRAIN_REQUIRED_COLUMNS",
    "STEP5_TRAIN_VALIDATION_COLUMNS",
    "Step5ExportLoaderError",
    "Step5ExportSource",
    "Step5TrainTableLoadResult",
    "load_step5_train_table",
    "load_step5_pool_train_table",
    "resolved_step5_export_paths",
    "validate_step5_export_for_resolved_config",
    "validate_step5_export_source",
]
