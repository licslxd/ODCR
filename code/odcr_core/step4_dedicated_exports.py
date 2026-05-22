"""Step4-owned derived exports for audit/control and Step5 explanation tables."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from odcr_core.file_atomic import atomic_write_json
from odcr_core.index_contract import INDEX_CONTRACT_FILENAME, ODCR_ROUTING_TRAIN_CSV
from odcr_core.step4_export_validator import STEP4_EXPORT_MANIFEST, validate_step4_export_ready


STEP4_DEDICATED_EXPORTS_SCHEMA_VERSION = "odcr_step5_dedicated_train_exports/1"
STEP4_ROUTE_INTERSECTION_REPORT_SCHEMA_VERSION = "odcr_step4_route_intersection_report/1"
STEP4_DEDICATED_EXPORTS_STATUS_SCHEMA_VERSION = "odcr_step4_step5_dedicated_exports_status/1"
STEP4_DEDICATED_EXPORTS_STATUS = "step5_dedicated_exports_status.json"
STEP4_DEDICATED_EXPORTS_DIRNAME = "step5_exports"
FULL_AUDIT_PARQUET = "odcr_routing_full_audit.parquet"
STEP5_EXPLANATION_SCORER_TRAIN_PARQUET = "rating_stability_control_scorer_train.parquet"
STEP5_EXPLANATION_EXPLAINER_TRAIN_PARQUET = "step5_explanation_explainer_train.parquet"
STEP5_EXPLANATION_GOLD_ANCHOR_PARQUET = "rating_stability_control_gold_anchor.parquet"
STEP5_EXPLANATION_CF_AUG_PARQUET = "rating_stability_control_cf_aug.parquet"
STEP5_EXPLANATION_GOLD_ANCHOR_PARQUET = "step5_explanation_gold_anchor.parquet"
STEP5_EXPLANATION_CF_AUG_PARQUET = "step5_explanation_cf_aug.parquet"
ROUTE_INTERSECTION_REPORT = "step5_route_intersection_report.json"
STEP5_TRAIN_MANIFEST = "step5_train_manifest.json"

STEP5_DEDICATED_SOURCE_REQUIRED_COLUMNS: tuple[str, ...] = (
    "user",
    "item",
    "rating",
    "review",
    "explanation",
    "clean_text",
    "domain",
    "sample_id",
    "sample_origin",
    "train_keep",
    "route_scorer",
    "route_explainer",
    "sample_weight_hint",
    "confidence_bucket",
    "uncertainty_score",
    "cf_reliability_score",
    "content_retention_score",
    "rating_stability_score",
    "style_shift_score",
    "text_quality_score",
    "route_reason_scorer",
    "route_reason_explainer",
    "preprocess_route_scorer_prior",
    "preprocess_route_explainer_prior",
    "entropy_score",
    "content_evidence",
    "style_evidence",
    "domain_style_anchor",
    "local_style_residual_hint",
    "polarity_anchor",
    "content_anchor_score",
    "style_anchor_score",
    "evidence_quality_prior",
    "user_idx_global",
    "item_idx_global",
)

STEP5_EXPLANATION_SCORER_TRAIN_COLUMNS: tuple[str, ...] = (
    "user",
    "item",
    "rating",
    "review",
    "explanation",
    "clean_text",
    "domain",
    "sample_id",
    "user_idx_global",
    "item_idx_global",
    "sample_origin",
    "train_keep",
    "route_scorer",
    "route_explainer",
    "sample_weight_hint",
    "confidence_bucket",
    "uncertainty_score",
    "cf_reliability_score",
    "content_retention_score",
    "rating_stability_score",
    "style_shift_score",
    "text_quality_score",
    "route_reason_scorer",
    "route_reason_explainer",
    "preprocess_route_scorer_prior",
    "preprocess_route_explainer_prior",
    "entropy_score",
    "content_evidence",
    "content_anchor_score",
    "style_evidence",
    "style_anchor_score",
    "domain_style_anchor",
    "local_style_residual_hint",
    "polarity_anchor",
    "evidence_quality_prior",
)

STEP5_EXPLANATION_EXPLAINER_TRAIN_COLUMNS: tuple[str, ...] = STEP5_EXPLANATION_SCORER_TRAIN_COLUMNS

STRING_COLUMNS = {
    "user",
    "item",
    "review",
    "explanation",
    "content_evidence",
    "polarity_anchor",
    "domain_style_anchor",
    "local_style_residual_hint",
    "style_evidence",
    "domain",
    "route_reason_scorer",
    "route_reason_explainer",
    "sample_origin",
    "clean_text",
    "bad_tail_types",
    "train_drop_reason",
}
INT_COLUMNS = {
    "preprocess_route_scorer_prior",
    "preprocess_route_explainer_prior",
    "sample_id",
    "confidence_bucket",
    "route_scorer",
    "route_explainer",
    "train_keep",
    "is_counterfactual",
    "clean_changed",
    "html_entity_hit",
    "bad_tail_hit",
    "template_hit",
    "template_count",
    "template_hard_drop_hit",
    "template_downweighted",
    "noisy_tail_downweighted",
    "short_fragment_hit",
    "repeat_tail_hit",
    "user_idx_global",
    "item_idx_global",
}
FLOAT_COLUMNS = {
    "rating",
    "content_anchor_score",
    "style_anchor_score",
    "evidence_quality_prior",
    "entropy",
    "rating_target",
    "rating_counterfactual",
    "rating_delta",
    "rating_stability_score",
    "shared_latent_similarity",
    "specific_latent_shift",
    "content_retention_score",
    "style_shift_score",
    "cf_reliability_score",
    "uncertainty_score",
    "entropy_score",
    "text_quality_score",
    "sample_weight_hint",
}

FIELD_REASONS: dict[str, str] = {
    "user": "human-readable source user id for audit and traceability",
    "item": "human-readable source item id for audit and traceability",
    "rating": "rating label retained for Step3 rating-source lineage and audit joins",
    "review": "source review text retained as the text-side evidence field",
    "explanation": "raw Step4 explanation target retained for training/audit comparison",
    "clean_text": "cleaned explanation text used by current Step5 tokenization",
    "domain": "domain label required for Step5 domain conditioning",
    "sample_id": "Step4 row identity for lineage and deterministic joins",
    "user_idx_global": "global user index contract required by Step5 embedding lookup",
    "item_idx_global": "global item index contract required by Step5 embedding lookup",
    "sample_origin": "origin split required for gold-anchor versus CF augmentation accounting",
    "train_keep": "Step4 posterior train gate used by dedicated export filters",
    "route_scorer": "Step4 posterior rating-stability/scorer-safety audit control signal",
    "route_explainer": "Step4 posterior primary Step5 explanation training route decision",
    "sample_weight_hint": "posterior sample weight used by Step5 loss scheduling",
    "confidence_bucket": "posterior confidence group for UCI/CCV control",
    "uncertainty_score": "posterior uncertainty score for UCI and route diagnostics",
    "cf_reliability_score": "RCR posterior reliability basis for LCI/FCA weighting",
    "content_retention_score": "content retention basis for rating-stability audit and FCA",
    "rating_stability_score": "rating-stability basis for scorer-safety audit confidence",
    "style_shift_score": "style-shift basis for explainer routing and FCA diversity",
    "text_quality_score": "text hygiene score for route confidence and validation",
    "route_reason_scorer": "human-readable rating-stability/scorer-safety posterior reason",
    "route_reason_explainer": "human-readable explainer-route posterior reason",
    "preprocess_route_scorer_prior": "preprocess rating-stability prior retained as prior-only audit context",
    "preprocess_route_explainer_prior": "preprocess explainer prior retained as prior-only audit context",
    "entropy_score": "generation entropy feature retained for uncertainty lineage",
    "content_evidence": "CCV/FCA content evidence basis",
    "content_anchor_score": "CCV numeric content anchor control",
    "style_evidence": "CCV/FCA style evidence basis",
    "style_anchor_score": "CCV numeric style anchor control",
    "domain_style_anchor": "CCV domain-style control text",
    "local_style_residual_hint": "CCV local-style control text",
    "polarity_anchor": "CCV polarity control text",
    "evidence_quality_prior": "preprocess evidence prior retained as prior-only control context",
}


class Step4DedicatedExportError(RuntimeError):
    """Raised when Step4 dedicated Step5 exports cannot be produced or trusted."""


@dataclass
class Step4DedicatedExportValidationResult:
    ready: bool
    run_dir: Path
    status_path: Path
    manifest_path: Path
    report_path: Path
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_payload(self, repo_root: str | Path | None = None) -> dict[str, Any]:
        root = Path(repo_root).expanduser().resolve() if repo_root is not None else None

        def rel(path: Path | None) -> str | None:
            if path is None:
                return None
            p = path.resolve()
            if root is None:
                return p.as_posix()
            try:
                return p.relative_to(root).as_posix()
            except ValueError:
                return p.as_posix()

        return {
            "schema_version": STEP4_DEDICATED_EXPORTS_STATUS_SCHEMA_VERSION,
            "ready": bool(self.ready),
            "run_dir": rel(self.run_dir),
            "status_path": rel(self.status_path),
            "manifest_path": rel(self.manifest_path),
            "report_path": rel(self.report_path),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "diagnostics": dict(self.diagnostics),
        }


class _Stats:
    def __init__(self) -> None:
        self.count = 0
        self.sum = 0.0
        self.min: float | None = None
        self.max: float | None = None

    def update(self, values: pd.Series) -> None:
        numeric = pd.to_numeric(values, errors="coerce").dropna().astype(float)
        if numeric.empty:
            return
        arr = numeric.to_numpy(dtype=float, copy=False)
        if not np.isfinite(arr).all():
            raise Step4DedicatedExportError("non-finite value while accumulating dedicated export statistics")
        self.count += int(len(arr))
        self.sum += float(arr.sum())
        mn = float(arr.min())
        mx = float(arr.max())
        self.min = mn if self.min is None else min(self.min, mn)
        self.max = mx if self.max is None else max(self.max, mx)

    def to_payload(self) -> dict[str, Any]:
        if self.count <= 0:
            return {"count": 0, "min": None, "mean": None, "max": None}
        return {
            "count": int(self.count),
            "min": float(self.min if self.min is not None else 0.0),
            "mean": float(self.sum / max(self.count, 1)),
            "max": float(self.max if self.max is not None else 0.0),
        }


class _ParquetSink:
    def __init__(self, path: Path, columns: Sequence[str]) -> None:
        self.path = path
        self.tmp_path = Path(str(path) + f".tmp.{os.getpid()}")
        self.columns = tuple(columns)
        self.writer: pq.ParquetWriter | None = None
        self.row_count = 0
        self.column_count = len(self.columns)
        if self.tmp_path.exists():
            self.tmp_path.unlink()

    def write(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        table = pa.Table.from_pandas(df.loc[:, self.columns], preserve_index=False)
        if self.writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = pq.ParquetWriter(str(self.tmp_path), table.schema, compression="snappy")
        self.writer.write_table(table)
        self.row_count += int(len(df))

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        elif not self.tmp_path.exists():
            empty = pa.Table.from_pandas(pd.DataFrame(columns=list(self.columns)), preserve_index=False)
            self.writer = pq.ParquetWriter(str(self.tmp_path), empty.schema, compression="snappy")
            self.writer.close()
            self.writer = None

    def commit(self, *, require_non_empty: bool) -> dict[str, Any]:
        self.close()
        if require_non_empty and self.row_count <= 0:
            raise Step4DedicatedExportError(f"refusing to publish zero-row parquet: {self.path}")
        meta = pq.ParquetFile(str(self.tmp_path)).metadata
        if int(meta.num_rows) != int(self.row_count):
            raise Step4DedicatedExportError(
                f"parquet row_count mismatch for {self.path.name}: writer={self.row_count} metadata={meta.num_rows}"
            )
        os.replace(str(self.tmp_path), str(self.path))
        _fsync_parent(self.path)
        return {
            "path": self.path,
            "row_count": int(self.row_count),
            "column_count": int(self.column_count),
            "size_bytes": int(self.path.stat().st_size),
            "sha256": _file_sha256(self.path),
        }

    def cleanup(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        if self.tmp_path.exists():
            self.tmp_path.unlink()


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _fsync_parent(path: Path) -> None:
    try:
        fd = os.open(str(path.parent), os.O_DIRECTORY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _repo_relative(repo_root: Path, path: str | Path | None) -> str | None:
    if path is None:
        return None
    p = Path(path).expanduser()
    p = (repo_root / p).resolve() if not p.is_absolute() else p.resolve()
    try:
        return p.relative_to(repo_root).as_posix()
    except ValueError:
        return p.as_posix()


def _repo_path(repo_root: Path, raw: Any, *, context: str) -> Path:
    value = str(raw or "").strip()
    if not value:
        raise Step4DedicatedExportError(f"{context} path is empty")
    path = Path(value).expanduser()
    return (repo_root / path).resolve() if not path.is_absolute() else path.resolve()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise Step4DedicatedExportError(f"{label} missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise Step4DedicatedExportError(f"{label} invalid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise Step4DedicatedExportError(f"{label} JSON root must be an object: {path}")
    return payload


def _format_path_map(repo_root: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    if "path" in out:
        out["path"] = _repo_relative(repo_root, out["path"])
    return out


def _normalize_config(raw: Mapping[str, Any]) -> dict[str, Any]:
    cfg = dict(raw)
    cfg.setdefault("enabled", True)
    cfg.setdefault("output_dir_name", STEP4_DEDICATED_EXPORTS_DIRNAME)
    cfg.setdefault("full_audit_format", "parquet")
    cfg.setdefault("scorer_train_format", "parquet")
    cfg.setdefault("explainer_train_format", "parquet")
    cfg.setdefault("write_gold_cf_subsplits", True)
    cfg.setdefault("full_audit_role", "audit_only")
    cfg.setdefault("atomic_write", True)
    cfg.setdefault("validate_after_write", True)
    cfg.setdefault("chunk_rows", 100_000)
    cfg.setdefault("scorer_filter", {"train_keep": True, "route_scorer": True, "min_sample_weight_hint": 0.0})
    cfg.setdefault("explainer_filter", {"train_keep": True, "route_explainer": True, "min_sample_weight_hint": 0.0})
    return cfg


def _validate_required_columns(columns: Sequence[str], required: Sequence[str], *, context: str) -> None:
    available = {str(c) for c in columns}
    missing = [str(c) for c in required if str(c) not in available]
    if missing:
        raise Step4DedicatedExportError(f"{context} missing required columns: {', '.join(missing)}")


def _read_header(path: Path) -> tuple[str, ...]:
    try:
        return tuple(str(c) for c in pd.read_csv(path, nrows=0).columns)
    except Exception as exc:
        raise Step4DedicatedExportError(f"failed to read source export header: {path}: {exc}") from exc


def _normalize_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    out = chunk.copy()
    for col in out.columns:
        if col in STRING_COLUMNS:
            out[col] = out[col].astype("string").fillna("")
        elif col in INT_COLUMNS:
            out[col] = pd.to_numeric(out[col], errors="raise").astype("Int64")
        elif col in FLOAT_COLUMNS:
            out[col] = pd.to_numeric(out[col], errors="raise").astype("float64")
    return out


def _bool_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="raise").fillna(0).astype(int) == 1


def _weight_series(df: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(df["sample_weight_hint"], errors="raise").fillna(0.0).astype(float)


def _confidence_label(value: Any) -> str:
    try:
        bucket = int(value)
    except Exception:
        bucket = -1
    return {2: "high", 1: "medium", 0: "low"}.get(bucket, f"unknown_{bucket}")


def _route_class(rs: pd.Series, re: pd.Series) -> pd.Series:
    rs_i = rs.astype(int)
    re_i = re.astype(int)
    return pd.Series(
        np.select(
            [(rs_i == 1) & (re_i == 1), (rs_i == 1) & (re_i == 0), (rs_i == 0) & (re_i == 1)],
            ["both", "scorer_only", "explainer_only"],
            default="neither",
        ),
        index=rs.index,
    )


def _increment_counts(target: dict[str, int], values: Mapping[Any, Any]) -> None:
    for key, value in values.items():
        target[str(key)] = int(target.get(str(key), 0)) + int(value)


def _value_counts(series: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in series.value_counts(dropna=False).items()}


def _update_confidence_breakdown(store: dict[str, dict[str, int]], name: str, df: pd.DataFrame) -> None:
    bucket = store.setdefault(name, {"high": 0, "medium": 0, "low": 0})
    if df.empty:
        return
    labels = pd.to_numeric(df["confidence_bucket"], errors="raise").map(_confidence_label)
    for key, value in labels.value_counts().items():
        bucket[str(key)] = int(bucket.get(str(key), 0)) + int(value)


def _update_stats_by_origin_route(
    store: dict[str, dict[str, dict[str, _Stats]]],
    df: pd.DataFrame,
    *,
    value_col: str,
) -> None:
    if df.empty:
        return
    rs = pd.to_numeric(df["route_scorer"], errors="raise").fillna(0).astype(int)
    re = pd.to_numeric(df["route_explainer"], errors="raise").fillna(0).astype(int)
    route = _route_class(rs, re)
    tmp = df[["sample_origin", value_col]].copy()
    tmp["_route_class"] = route
    for (origin, route_class), sub in tmp.groupby(["sample_origin", "_route_class"], dropna=False):
        origin_key = str(origin)
        route_key = str(route_class)
        stats = store.setdefault(origin_key, {}).setdefault(route_key, _Stats())
        stats.update(sub[value_col])


def _stats_payload(store: dict[str, dict[str, dict[str, _Stats]]]) -> dict[str, Any]:
    return {
        origin: {route: stats.to_payload() for route, stats in routes.items()}
        for origin, routes in sorted(store.items())
    }


def _column_source_table(full_columns: Sequence[str]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    full_set = set(full_columns)
    for col in sorted(set(STEP5_EXPLANATION_SCORER_TRAIN_COLUMNS) | set(STEP5_EXPLANATION_EXPLAINER_TRAIN_COLUMNS)):
        fields[col] = {
            "included_in": [
                name
                for name, columns in (
                    ("full_audit", full_columns),
                    ("rating_stability_control_scorer_train", STEP5_EXPLANATION_SCORER_TRAIN_COLUMNS),
                    ("step5_explanation_explainer_train", STEP5_EXPLANATION_EXPLAINER_TRAIN_COLUMNS),
                )
                if col in set(columns)
            ],
            "source_column_present": col in full_set,
            "reason": FIELD_REASONS.get(col, "retained for Step5 dedicated export contract"),
        }
    audit_only = [col for col in full_columns if col not in fields]
    for col in audit_only:
        fields[str(col)] = {
            "included_in": ["full_audit"],
            "source_column_present": True,
            "reason": "audit-only Step4 source column intentionally excluded from narrow RatingStabilityControl/B train tables",
        }
    return {
        "schema_version": "odcr_step4_dedicated_export_source_table/1",
        "fields": fields,
    }


def _relative_export_record(repo_root: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(record)
    out["path"] = _repo_relative(repo_root, out.get("path"))
    return out


def export_step4_dedicated_exports(
    *,
    repo_root: str | Path,
    task: int,
    from_run: str,
    config: Mapping[str, Any],
    dry_run: bool = False,
    update_stage_status: bool = True,
) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve()
    cfg = _normalize_config(config)
    if not bool(cfg.get("enabled", True)):
        raise Step4DedicatedExportError("step4.step5_dedicated_exports.enabled must be true to export dedicated tables")
    run_dir = root / "runs" / "step4" / f"task{int(task)}" / str(from_run)
    source_csv = run_dir / ODCR_ROUTING_TRAIN_CSV
    source_manifest_path = run_dir / STEP4_EXPORT_MANIFEST
    source_index_path = run_dir / INDEX_CONTRACT_FILENAME
    output_dir = run_dir / str(cfg["output_dir_name"])
    header = _read_header(source_csv)
    _validate_required_columns(header, STEP5_DEDICATED_SOURCE_REQUIRED_COLUMNS, context="Step4 full export")
    _validate_required_columns(header, STEP5_EXPLANATION_SCORER_TRAIN_COLUMNS, context="RatingStabilityControl scorer column contract")
    _validate_required_columns(header, STEP5_EXPLANATION_EXPLAINER_TRAIN_COLUMNS, context="Step5 explanation explainer column contract")
    base_validation = validate_step4_export_ready(run_dir, repo_root=root)
    if not base_validation.ready:
        raise Step4DedicatedExportError("source Step4 export is not ready: " + "; ".join(base_validation.errors))
    source_sha = str(base_validation.diagnostics.get("export_sha256") or "")
    source_manifest_sha = str(base_validation.diagnostics.get("manifest_sha256") or _file_sha256(source_manifest_path))
    source_index_sha = str(base_validation.diagnostics.get("index_contract_sha256") or _file_sha256(source_index_path))
    source_size = int(source_csv.stat().st_size)

    scorer_filter = dict(cfg.get("scorer_filter") or {})
    explainer_filter = dict(cfg.get("explainer_filter") or {})
    scorer_min_weight = float(scorer_filter.get("min_sample_weight_hint", 0.0))
    explainer_min_weight = float(explainer_filter.get("min_sample_weight_hint", 0.0))
    chunk_rows = max(1, int(cfg.get("chunk_rows", 100_000)))
    if dry_run:
        return {
            "schema_version": STEP4_DEDICATED_EXPORTS_SCHEMA_VERSION,
            "dry_run": True,
            "task": int(task),
            "step4_run": str(from_run),
            "source_full_export": _repo_relative(root, source_csv),
            "source_full_export_sha256": source_sha,
            "output_dir": _repo_relative(root, output_dir),
            "planned_exports": {
                "full_audit": FULL_AUDIT_PARQUET,
                "rating_stability_control_scorer_train": STEP5_EXPLANATION_SCORER_TRAIN_PARQUET,
                "step5_explanation_explainer_train": STEP5_EXPLANATION_EXPLAINER_TRAIN_PARQUET,
            },
            "filter_rules": {
                "rating_stability_control_scorer_train": f"train_keep==1 AND route_scorer==1 AND sample_weight_hint>{scorer_min_weight:g}",
                "step5_explanation_explainer_train": f"train_keep==1 AND route_explainer==1 AND sample_weight_hint>{explainer_min_weight:g}",
            },
            "full_audit_role": str(cfg.get("full_audit_role", "audit_only")),
            "narrow_column_counts": {
                "full_audit": len(header),
                "rating_stability_control_scorer_train": len(STEP5_EXPLANATION_SCORER_TRAIN_COLUMNS),
                "step5_explanation_explainer_train": len(STEP5_EXPLANATION_EXPLAINER_TRAIN_COLUMNS),
            },
            "source_readiness": base_validation.to_payload(root),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    sinks: dict[str, _ParquetSink] = {
        "full_audit": _ParquetSink(output_dir / FULL_AUDIT_PARQUET, header),
        "rating_stability_control_scorer_train": _ParquetSink(output_dir / STEP5_EXPLANATION_SCORER_TRAIN_PARQUET, STEP5_EXPLANATION_SCORER_TRAIN_COLUMNS),
        "step5_explanation_explainer_train": _ParquetSink(output_dir / STEP5_EXPLANATION_EXPLAINER_TRAIN_PARQUET, STEP5_EXPLANATION_EXPLAINER_TRAIN_COLUMNS),
    }
    if bool(cfg.get("write_gold_cf_subsplits", True)):
        sinks.update(
            {
                "rating_stability_control_gold_anchor": _ParquetSink(output_dir / STEP5_EXPLANATION_GOLD_ANCHOR_PARQUET, STEP5_EXPLANATION_SCORER_TRAIN_COLUMNS),
                "rating_stability_control_cf_aug": _ParquetSink(output_dir / STEP5_EXPLANATION_CF_AUG_PARQUET, STEP5_EXPLANATION_SCORER_TRAIN_COLUMNS),
                "step5_explanation_gold_anchor": _ParquetSink(output_dir / STEP5_EXPLANATION_GOLD_ANCHOR_PARQUET, STEP5_EXPLANATION_EXPLAINER_TRAIN_COLUMNS),
                "step5_explanation_cf_aug": _ParquetSink(output_dir / STEP5_EXPLANATION_CF_AUG_PARQUET, STEP5_EXPLANATION_EXPLAINER_TRAIN_COLUMNS),
            }
        )

    composition: dict[str, int] = {}
    single_counts = {"train_keep": 0, "route_scorer": 0, "route_explainer": 0}
    intersections = {
        "train_keep_and_route_scorer": 0,
        "train_keep_and_route_explainer": 0,
        "train_keep_and_route_scorer_and_route_explainer": 0,
        "train_keep_and_route_scorer_and_aux_cf": 0,
        "train_keep_and_route_explainer_and_aux_cf": 0,
        "train_keep_and_route_scorer_and_target_gold": 0,
        "train_keep_and_route_explainer_and_target_gold": 0,
        "train_keep_and_route_scorer_and_aux_gold": 0,
        "train_keep_and_route_explainer_and_aux_gold": 0,
    }
    confidence_breakdown: dict[str, dict[str, int]] = {}
    origin_breakdown: dict[str, dict[str, int]] = {
        "full_audit": {},
        "rating_stability_control_scorer_train": {},
        "step5_explanation_explainer_train": {},
    }
    sample_weight_stats: dict[str, dict[str, dict[str, _Stats]]] = {}
    uncertainty_stats: dict[str, dict[str, dict[str, _Stats]]] = {}
    text_quality_stats: dict[str, dict[str, dict[str, _Stats]]] = {}
    drop_reason_counts: dict[str, int] = {}
    low_confidence_kept_counts: dict[str, Any] = {"all": 0, "by_origin": {}, "rating_stability_control_scorer_train": 0, "step5_explanation_explainer_train": 0}
    zero_weight_counts: dict[str, Any] = {"all": 0, "by_origin": {}, "rating_stability_control_scorer_train": 0, "step5_explanation_explainer_train": 0}
    total_rows = 0

    dtype = {col: "string" for col in STRING_COLUMNS if col in header}
    try:
        for chunk in pd.read_csv(source_csv, chunksize=chunk_rows, dtype=dtype):
            chunk = _normalize_chunk(chunk)
            total_rows += int(len(chunk))
            train_keep = _bool_series(chunk, "train_keep")
            route_scorer = _bool_series(chunk, "route_scorer")
            route_explainer = _bool_series(chunk, "route_explainer")
            sample_weight = _weight_series(chunk)
            aux_cf = chunk["sample_origin"].astype(str) == "aux_cf"
            target_gold = chunk["sample_origin"].astype(str) == "target_gold"
            aux_gold = chunk["sample_origin"].astype(str) == "aux_gold"
            rating_stability_control_mask = train_keep & route_scorer & (sample_weight > scorer_min_weight)
            step5b_mask = train_keep & route_explainer & (sample_weight > explainer_min_weight)

            sinks["full_audit"].write(chunk.loc[:, header])
            sinks["rating_stability_control_scorer_train"].write(chunk.loc[rating_stability_control_mask, STEP5_EXPLANATION_SCORER_TRAIN_COLUMNS])
            sinks["step5_explanation_explainer_train"].write(chunk.loc[step5b_mask, STEP5_EXPLANATION_EXPLAINER_TRAIN_COLUMNS])
            if "rating_stability_control_gold_anchor" in sinks:
                sinks["rating_stability_control_gold_anchor"].write(chunk.loc[rating_stability_control_mask & (target_gold | aux_gold), STEP5_EXPLANATION_SCORER_TRAIN_COLUMNS])
                sinks["rating_stability_control_cf_aug"].write(chunk.loc[rating_stability_control_mask & aux_cf, STEP5_EXPLANATION_SCORER_TRAIN_COLUMNS])
                sinks["step5_explanation_gold_anchor"].write(chunk.loc[step5b_mask & (target_gold | aux_gold), STEP5_EXPLANATION_EXPLAINER_TRAIN_COLUMNS])
                sinks["step5_explanation_cf_aug"].write(chunk.loc[step5b_mask & aux_cf, STEP5_EXPLANATION_EXPLAINER_TRAIN_COLUMNS])

            _increment_counts(composition, _value_counts(chunk["sample_origin"]))
            _increment_counts(origin_breakdown["full_audit"], _value_counts(chunk["sample_origin"]))
            _increment_counts(origin_breakdown["rating_stability_control_scorer_train"], _value_counts(chunk.loc[rating_stability_control_mask, "sample_origin"]))
            _increment_counts(origin_breakdown["step5_explanation_explainer_train"], _value_counts(chunk.loc[step5b_mask, "sample_origin"]))

            single_counts["train_keep"] += int(train_keep.sum())
            single_counts["route_scorer"] += int(route_scorer.sum())
            single_counts["route_explainer"] += int(route_explainer.sum())
            intersections["train_keep_and_route_scorer"] += int((train_keep & route_scorer).sum())
            intersections["train_keep_and_route_explainer"] += int((train_keep & route_explainer).sum())
            intersections["train_keep_and_route_scorer_and_route_explainer"] += int((train_keep & route_scorer & route_explainer).sum())
            intersections["train_keep_and_route_scorer_and_aux_cf"] += int((train_keep & route_scorer & aux_cf).sum())
            intersections["train_keep_and_route_explainer_and_aux_cf"] += int((train_keep & route_explainer & aux_cf).sum())
            intersections["train_keep_and_route_scorer_and_target_gold"] += int((train_keep & route_scorer & target_gold).sum())
            intersections["train_keep_and_route_explainer_and_target_gold"] += int((train_keep & route_explainer & target_gold).sum())
            intersections["train_keep_and_route_scorer_and_aux_gold"] += int((train_keep & route_scorer & aux_gold).sum())
            intersections["train_keep_and_route_explainer_and_aux_gold"] += int((train_keep & route_explainer & aux_gold).sum())

            _update_confidence_breakdown(confidence_breakdown, "full_audit", chunk)
            _update_confidence_breakdown(confidence_breakdown, "rating_stability_control_scorer_train", chunk.loc[rating_stability_control_mask])
            _update_confidence_breakdown(confidence_breakdown, "step5_explanation_explainer_train", chunk.loc[step5b_mask])
            _update_stats_by_origin_route(sample_weight_stats, chunk, value_col="sample_weight_hint")
            _update_stats_by_origin_route(uncertainty_stats, chunk, value_col="uncertainty_score")
            _update_stats_by_origin_route(text_quality_stats, chunk, value_col="text_quality_score")

            dropped = chunk.loc[~train_keep, "train_drop_reason"].astype("string").fillna("")
            if not dropped.empty:
                _increment_counts(drop_reason_counts, _value_counts(dropped.where(dropped.str.len() > 0, "unspecified")))
            low_kept = train_keep & (pd.to_numeric(chunk["confidence_bucket"], errors="raise").fillna(-1).astype(int) == 0)
            low_confidence_kept_counts["all"] += int(low_kept.sum())
            _increment_counts(low_confidence_kept_counts["by_origin"], _value_counts(chunk.loc[low_kept, "sample_origin"]))
            low_confidence_kept_counts["rating_stability_control_scorer_train"] += int((low_kept & rating_stability_control_mask).sum())
            low_confidence_kept_counts["step5_explanation_explainer_train"] += int((low_kept & step5b_mask).sum())
            zero_weight = sample_weight <= 0.0
            zero_weight_counts["all"] += int(zero_weight.sum())
            _increment_counts(zero_weight_counts["by_origin"], _value_counts(chunk.loc[zero_weight, "sample_origin"]))
            zero_weight_counts["rating_stability_control_scorer_train"] += int((zero_weight & rating_stability_control_mask).sum())
            zero_weight_counts["step5_explanation_explainer_train"] += int((zero_weight & step5b_mask).sum())
    except Exception:
        for sink in sinks.values():
            sink.cleanup()
        raise

    sink_records: dict[str, dict[str, Any]] = {}
    try:
        for name, sink in sinks.items():
            sink_records[name] = sink.commit(require_non_empty=name in {"full_audit", "rating_stability_control_scorer_train", "step5_explanation_explainer_train"})
    except Exception:
        for sink in sinks.values():
            sink.cleanup()
        raise

    if total_rows != base_validation.row_count:
        raise Step4DedicatedExportError(
            f"source scan row count mismatch: export scan {total_rows} != readiness {base_validation.row_count}"
        )

    filter_rules = {
        "rating_stability_control_scorer_train": f"train_keep==1 AND route_scorer==1 AND sample_weight_hint>{scorer_min_weight:g}",
        "step5_explanation_explainer_train": f"train_keep==1 AND route_explainer==1 AND sample_weight_hint>{explainer_min_weight:g}",
    }
    export_row_counts = {name: int(record["row_count"]) for name, record in sink_records.items()}
    report = {
        "schema_version": STEP4_ROUTE_INTERSECTION_REPORT_SCHEMA_VERSION,
        "source_full_export": _repo_relative(root, source_csv),
        "source_full_export_sha256": source_sha,
        "total_rows": int(total_rows),
        "composition": {k: int(v) for k, v in sorted(composition.items())},
        "single_column_counts": {k: int(v) for k, v in single_counts.items()},
        "intersections": {k: int(v) for k, v in intersections.items()},
        "confidence_breakdown": confidence_breakdown,
        "origin_breakdown": origin_breakdown,
        "export_row_counts": export_row_counts,
        "sample_weight_hint_stats_by_origin_and_route": _stats_payload(sample_weight_stats),
        "uncertainty_score_stats_by_origin_and_route": _stats_payload(uncertainty_stats),
        "text_quality_stats": _stats_payload(text_quality_stats),
        "drop_reason_counts": {k: int(v) for k, v in sorted(drop_reason_counts.items())},
        "low_confidence_kept_counts": low_confidence_kept_counts,
        "zero_weight_counts": zero_weight_counts,
        "filter_rules": filter_rules,
        "design_assessment": {
            "concat_only_risk": True,
            "rcr_physically_materialized": True,
            "full_table_is_audit_only": True,
            "gold_rows_default_route_both": True,
            "dedicated_exports_materialize_route_intersections": True,
        },
    }
    report_path = output_dir / ROUTE_INTERSECTION_REPORT
    atomic_write_json(report_path, report)

    source_manifest = _load_json(source_manifest_path, label="source Step4 manifest")
    lineage = source_manifest.get("step4_export_lineage") if isinstance(source_manifest.get("step4_export_lineage"), Mapping) else {}
    frozen = lineage.get("frozen_step3_lineage") if isinstance(lineage, Mapping) else {}
    from_step3 = str((frozen or {}).get("upstream_step3_run_id") or "")
    manifest_exports: dict[str, Any] = {}
    for name, record in sink_records.items():
        role = {
            "full_audit": "audit_only",
            "rating_stability_control_scorer_train": "rating_stability_control_train",
            "step5_explanation_explainer_train": "step5_explanation_train",
            "rating_stability_control_gold_anchor": "rating_stability_control_gold_anchor",
            "rating_stability_control_cf_aug": "rating_stability_control_cf_aug",
            "step5_explanation_gold_anchor": "step5_explanation_gold_anchor",
            "step5_explanation_cf_aug": "step5_explanation_cf_aug",
        }.get(name, name)
        item = {
            "path": _repo_relative(root, record["path"]),
            "row_count": int(record["row_count"]),
            "column_count": int(record["column_count"]),
            "size_bytes": int(record["size_bytes"]),
            "sha256": str(record["sha256"]),
            "role": role,
            "format": "parquet",
        }
        if name == "rating_stability_control_scorer_train":
            item["filter"] = filter_rules["rating_stability_control_scorer_train"]
        elif name == "step5_explanation_explainer_train":
            item["filter"] = filter_rules["step5_explanation_explainer_train"]
        elif name.startswith("rating_stability_control_"):
            item["parent_filter"] = filter_rules["rating_stability_control_scorer_train"]
        elif name.startswith("step5_explanation_"):
            item["parent_filter"] = filter_rules["step5_explanation_explainer_train"]
        manifest_exports[name] = item

    manifest = {
        "schema_version": STEP4_DEDICATED_EXPORTS_SCHEMA_VERSION,
        "producer_stage": "step4",
        "task_id": int(task),
        "step4_run": str(from_run),
        "from_step3": from_step3,
        "source_full_export": _repo_relative(root, source_csv),
        "source_full_export_sha256": source_sha,
        "source_full_export_size_bytes": source_size,
        "source_manifest": _repo_relative(root, source_manifest_path),
        "source_manifest_sha256": source_manifest_sha,
        "source_index_contract": _repo_relative(root, source_index_path),
        "source_index_contract_sha256": source_index_sha,
        "route_intersection_report": _repo_relative(root, report_path),
        "route_intersection_report_sha256": _file_sha256(report_path),
        "exports": manifest_exports,
        "lineage": {
            "source_step4_export_lineage": lineage,
            "source_step4_export_lineage_hash": lineage.get("lineage_hash") if isinstance(lineage, Mapping) else None,
            "source_step4_readiness": base_validation.to_payload(root),
        },
        "column_contract": {
            "full_audit_column_count": len(header),
            "rating_stability_control_scorer_train_column_count": len(STEP5_EXPLANATION_SCORER_TRAIN_COLUMNS),
            "step5_explanation_explainer_train_column_count": len(STEP5_EXPLANATION_EXPLAINER_TRAIN_COLUMNS),
            "required_columns": list(STEP5_DEDICATED_SOURCE_REQUIRED_COLUMNS),
            "rating_stability_control_scorer_train_columns": list(STEP5_EXPLANATION_SCORER_TRAIN_COLUMNS),
            "step5_explanation_explainer_train_columns": list(STEP5_EXPLANATION_EXPLAINER_TRAIN_COLUMNS),
        },
        "source_table": _column_source_table(header),
        "export_config": cfg,
        "created_at_utc": _now(),
        "do_not_use_full_audit_as_default_step5_train": True,
    }
    manifest_path = output_dir / STEP5_TRAIN_MANIFEST
    atomic_write_json(manifest_path, manifest)

    validation = validate_step4_dedicated_exports(
        run_dir,
        repo_root=root,
        expected_source_full_export_sha256=source_sha,
        output_dir_name=str(cfg["output_dir_name"]),
        raise_on_error=True,
    )
    status_payload = write_step5_dedicated_exports_status(
        repo_root=root,
        run_dir=run_dir,
        validation=validation,
        update_stage_status=False,
    )
    manifest = _load_json(manifest_path, label="dedicated export manifest")
    manifest["status_sidecar"] = _repo_relative(root, run_dir / "meta" / STEP4_DEDICATED_EXPORTS_STATUS)
    atomic_write_json(manifest_path, manifest)
    validation = validate_step4_dedicated_exports(
        run_dir,
        repo_root=root,
        expected_source_full_export_sha256=source_sha,
        output_dir_name=str(cfg["output_dir_name"]),
        raise_on_error=True,
    )
    status_payload = write_step5_dedicated_exports_status(
        repo_root=root,
        run_dir=run_dir,
        validation=validation,
        update_stage_status=bool(update_stage_status),
    )

    result_exports = {name: _relative_export_record(root, record) for name, record in sink_records.items()}
    return {
        "schema_version": STEP4_DEDICATED_EXPORTS_SCHEMA_VERSION,
        "dry_run": False,
        "task": int(task),
        "step4_run": str(from_run),
        "output_dir": _repo_relative(root, output_dir),
        "source_full_export": _repo_relative(root, source_csv),
        "source_full_export_sha256": source_sha,
        "exports": result_exports,
        "route_intersection_report": _repo_relative(root, report_path),
        "step5_train_manifest": _repo_relative(root, manifest_path),
        "status_sidecar": _repo_relative(root, run_dir / "meta" / STEP4_DEDICATED_EXPORTS_STATUS),
        "status": status_payload,
        "validation": validation.to_payload(root),
    }


def validate_step4_dedicated_exports(
    run_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
    output_dir_name: str = STEP4_DEDICATED_EXPORTS_DIRNAME,
    expected_source_full_export_sha256: str | None = None,
    raise_on_error: bool = False,
) -> Step4DedicatedExportValidationResult:
    root = Path(repo_root).expanduser().resolve() if repo_root is not None else Path.cwd().resolve()
    run = Path(run_dir).expanduser()
    run = (root / run).resolve() if not run.is_absolute() else run.resolve()
    export_dir = run / str(output_dir_name)
    manifest_path = export_dir / STEP5_TRAIN_MANIFEST
    report_path = export_dir / ROUTE_INTERSECTION_REPORT
    status_path = run / "meta" / STEP4_DEDICATED_EXPORTS_STATUS
    result = Step4DedicatedExportValidationResult(
        ready=False,
        run_dir=run,
        status_path=status_path,
        manifest_path=manifest_path,
        report_path=report_path,
    )
    try:
        manifest = _load_json(manifest_path, label="dedicated export manifest")
        report = _load_json(report_path, label="route intersection report")
        if manifest.get("schema_version") != STEP4_DEDICATED_EXPORTS_SCHEMA_VERSION:
            raise Step4DedicatedExportError("dedicated export manifest schema_version mismatch")
        if report.get("schema_version") != STEP4_ROUTE_INTERSECTION_REPORT_SCHEMA_VERSION:
            raise Step4DedicatedExportError("route intersection report schema_version mismatch")
        if manifest.get("do_not_use_full_audit_as_default_step5_train") is not True:
            raise Step4DedicatedExportError("full audit table must not be labeled as default Step5 train")
        source_csv = _repo_path(root, manifest.get("source_full_export"), context="source_full_export")
        selected_csv = run / ODCR_ROUTING_TRAIN_CSV
        if source_csv.resolve() != selected_csv.resolve():
            raise Step4DedicatedExportError("source_full_export does not match current Step4 selected export path")
        expected_sha = str(expected_source_full_export_sha256 or manifest.get("source_full_export_sha256") or "").strip()
        if not expected_sha:
            raise Step4DedicatedExportError("source_full_export_sha256 missing from dedicated manifest")
        if str(report.get("source_full_export_sha256") or "") != expected_sha:
            raise Step4DedicatedExportError("route report source_full_export_sha256 mismatch")
        current_source_sha = expected_sha
        if expected_source_full_export_sha256 is None:
            current_source_sha = _file_sha256(source_csv)
            if current_source_sha != expected_sha:
                raise Step4DedicatedExportError("source full export sha256 no longer matches dedicated manifest")
        exports = manifest.get("exports")
        if not isinstance(exports, Mapping):
            raise Step4DedicatedExportError("dedicated manifest exports must be an object")
        required_names = ("full_audit", "rating_stability_control_scorer_train", "step5_explanation_explainer_train")
        for name in required_names:
            if name not in exports or not isinstance(exports.get(name), Mapping):
                raise Step4DedicatedExportError(f"dedicated manifest missing export {name}")
        if str(exports["full_audit"].get("role") or "") != "audit_only":
            raise Step4DedicatedExportError("full_audit export role must be audit_only")
        if str(exports["rating_stability_control_scorer_train"].get("role") or "") != "rating_stability_control_train":
            raise Step4DedicatedExportError("rating_stability_control_scorer_train role must be rating_stability_control_train")
        if str(exports["step5_explanation_explainer_train"].get("role") or "") != "step5_explanation_train":
            raise Step4DedicatedExportError("step5_explanation_explainer_train role must be step5_explanation_train")
        diagnostics_exports: dict[str, Any] = {}
        for name, item in exports.items():
            if not isinstance(item, Mapping):
                continue
            path = _repo_path(root, item.get("path"), context=f"exports.{name}.path")
            if not path.is_file():
                raise Step4DedicatedExportError(f"dedicated export missing: {path}")
            if _file_sha256(path) != str(item.get("sha256") or ""):
                raise Step4DedicatedExportError(f"dedicated export sha256 mismatch: {name}")
            pf = pq.ParquetFile(str(path))
            columns = tuple(pf.schema_arrow.names)
            row_count = int(pf.metadata.num_rows)
            if row_count != int(item.get("row_count") or -1):
                raise Step4DedicatedExportError(f"dedicated export row_count mismatch: {name}")
            if name in {"rating_stability_control_scorer_train", "rating_stability_control_gold_anchor", "rating_stability_control_cf_aug"}:
                _validate_required_columns(columns, STEP5_EXPLANATION_SCORER_TRAIN_COLUMNS, context=f"{name} parquet")
            if name in {"step5_explanation_explainer_train", "step5_explanation_gold_anchor", "step5_explanation_cf_aug"}:
                _validate_required_columns(columns, STEP5_EXPLANATION_EXPLAINER_TRAIN_COLUMNS, context=f"{name} parquet")
            if name in {"rating_stability_control_scorer_train", "step5_explanation_explainer_train"} and row_count <= 0:
                raise Step4DedicatedExportError(f"{name} must be non-empty")
            diagnostics_exports[name] = {
                "path": _repo_relative(root, path),
                "row_count": row_count,
                "column_count": len(columns),
                "size_bytes": int(path.stat().st_size),
                "sha256": str(item.get("sha256") or ""),
                "role": str(item.get("role") or ""),
            }
        report_counts = report.get("export_row_counts") if isinstance(report.get("export_row_counts"), Mapping) else {}
        for name in ("full_audit", "rating_stability_control_scorer_train", "step5_explanation_explainer_train"):
            if int(report_counts.get(name) or -1) != int(exports[name].get("row_count") or -2):
                raise Step4DedicatedExportError(f"route report export row_count mismatch for {name}")
        intersections = report.get("intersections") if isinstance(report.get("intersections"), Mapping) else {}
        if int(intersections.get("train_keep_and_route_scorer") or -1) < int(exports["rating_stability_control_scorer_train"].get("row_count") or 0):
            raise Step4DedicatedExportError("RatingStabilityControl row_count exceeds train_keep_and_route_scorer intersection")
        if int(intersections.get("train_keep_and_route_explainer") or -1) < int(exports["step5_explanation_explainer_train"].get("row_count") or 0):
            raise Step4DedicatedExportError("Step5 explanation row_count exceeds train_keep_and_route_explainer intersection")
        zero_weight = report.get("zero_weight_counts") if isinstance(report.get("zero_weight_counts"), Mapping) else {}
        if int(zero_weight.get("rating_stability_control_scorer_train") or 0) != 0:
            raise Step4DedicatedExportError("RatingStabilityControl dedicated export contains zero-weight rows")
        if int(zero_weight.get("step5_explanation_explainer_train") or 0) != 0:
            raise Step4DedicatedExportError("Step5 explanation dedicated export contains zero-weight rows")
        result.ready = True
        result.diagnostics = {
            "source_full_export": _repo_relative(root, source_csv),
            "source_full_export_sha256": current_source_sha,
            "route_intersection_report_sha256": _file_sha256(report_path),
            "step5_train_manifest_sha256": _file_sha256(manifest_path),
            "exports": diagnostics_exports,
            "filter_rules": report.get("filter_rules"),
            "intersections": intersections,
            "full_audit_table_role": "audit_only",
            "step5_train_input_role": "dedicated_split_exports",
        }
    except Exception as exc:
        result.errors.append(str(exc))
        if raise_on_error:
            raise
    return result


def step4_dedicated_stage_status_fields(
    *,
    repo_root: str | Path,
    run_dir: str | Path,
    validation: Step4DedicatedExportValidationResult | None = None,
) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve()
    run = Path(run_dir).expanduser()
    run = (root / run).resolve() if not run.is_absolute() else run.resolve()
    validation = validation or validate_step4_dedicated_exports(run, repo_root=root)
    fields: dict[str, Any] = {
        "step5_dedicated_exports_ready": bool(validation.ready),
        "full_audit_table_role": "audit_only",
        "step5_train_input_role": "dedicated_split_exports" if validation.ready else None,
        "step5_dedicated_exports_status": _repo_relative(root, run / "meta" / STEP4_DEDICATED_EXPORTS_STATUS),
        "step5_train_manifest": _repo_relative(root, validation.manifest_path),
        "route_intersection_report": _repo_relative(root, validation.report_path),
        "dedicated_export_readiness": validation.to_payload(root),
    }
    exports = ((validation.diagnostics or {}).get("exports") or {}) if validation.ready else {}
    if isinstance(exports, Mapping):
        fields["selected_full_audit_export"] = (exports.get("full_audit") or {}).get("path") if isinstance(exports.get("full_audit"), Mapping) else None
        fields["rating_stability_control_scorer_train_export"] = (exports.get("rating_stability_control_scorer_train") or {}).get("path") if isinstance(exports.get("rating_stability_control_scorer_train"), Mapping) else None
        fields["step5_explanation_explainer_train_export"] = (exports.get("step5_explanation_explainer_train") or {}).get("path") if isinstance(exports.get("step5_explanation_explainer_train"), Mapping) else None
    return fields


def write_step5_dedicated_exports_status(
    *,
    repo_root: str | Path,
    run_dir: str | Path,
    validation: Step4DedicatedExportValidationResult,
    update_stage_status: bool = True,
) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve()
    run = Path(run_dir).expanduser()
    run = (root / run).resolve() if not run.is_absolute() else run.resolve()
    meta = run / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    stage_status_path = meta / "stage_status.json"
    previous_stage_status_sha = _file_sha256(stage_status_path) if stage_status_path.is_file() else None
    fields = step4_dedicated_stage_status_fields(repo_root=root, run_dir=run, validation=validation)
    payload = {
        "schema_version": STEP4_DEDICATED_EXPORTS_STATUS_SCHEMA_VERSION,
        "generated_at_utc": _now(),
        "producer_stage": "step4",
        "task_id": int(_infer_task_id_from_run_dir(run)),
        "step4_run": run.name,
        "run_dir": _repo_relative(root, run),
        "previous_stage_status_sha256": previous_stage_status_sha,
        "stage_status_update": "applied_after_sidecar_write" if update_stage_status and stage_status_path.is_file() else "not_requested_or_missing",
        "selected_source_export": _repo_relative(root, run / ODCR_ROUTING_TRAIN_CSV),
        "selected_source_export_sha256": validation.diagnostics.get("source_full_export_sha256"),
        **fields,
        "validation": validation.to_payload(root),
    }
    status_path = meta / STEP4_DEDICATED_EXPORTS_STATUS
    atomic_write_json(status_path, payload)
    if update_stage_status and stage_status_path.is_file() and validation.ready:
        status = _load_json(stage_status_path, label="stage_status")
        status.update(fields)
        status.setdefault("artifacts", {})
        if isinstance(status["artifacts"], dict):
            for key in (
                "selected_full_audit_export",
                "rating_stability_control_scorer_train_export",
                "step5_explanation_explainer_train_export",
                "step5_train_manifest",
                "route_intersection_report",
                "step5_dedicated_exports_status",
            ):
                path_text = status.get(key)
                if path_text:
                    path = _repo_path(root, path_text, context=key)
                    status["artifacts"][key] = {
                        "path": _repo_relative(root, path),
                        "exists": path.is_file(),
                        "is_file": path.is_file(),
                        "sha256": _file_sha256(path) if path.is_file() else None,
                    }
        status["updated_at"] = _now()
        status["updated_at_utc"] = status["updated_at"]
        atomic_write_json(stage_status_path, status)
    return payload


def _infer_task_id_from_run_dir(run_dir: Path) -> int:
    parent = run_dir.parent.name
    if parent.startswith("task"):
        return int(parent[len("task") :])
    return 0


__all__ = [
    "FULL_AUDIT_PARQUET",
    "ROUTE_INTERSECTION_REPORT",
    "STEP4_DEDICATED_EXPORTS_DIRNAME",
    "STEP4_DEDICATED_EXPORTS_SCHEMA_VERSION",
    "STEP4_DEDICATED_EXPORTS_STATUS",
    "STEP4_DEDICATED_EXPORTS_STATUS_SCHEMA_VERSION",
    "STEP4_ROUTE_INTERSECTION_REPORT_SCHEMA_VERSION",
    "STEP5_EXPLANATION_SCORER_TRAIN_COLUMNS",
    "STEP5_EXPLANATION_SCORER_TRAIN_PARQUET",
    "STEP5_EXPLANATION_EXPLAINER_TRAIN_COLUMNS",
    "STEP5_EXPLANATION_EXPLAINER_TRAIN_PARQUET",
    "STEP5_TRAIN_MANIFEST",
    "Step4DedicatedExportError",
    "Step4DedicatedExportValidationResult",
    "export_step4_dedicated_exports",
    "step4_dedicated_stage_status_fields",
    "validate_step4_dedicated_exports",
    "write_step5_dedicated_exports_status",
]
