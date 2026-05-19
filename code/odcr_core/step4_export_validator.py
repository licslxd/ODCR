"""Strict Step4 export readiness gate for Step5 handoff.

This module is deliberately CPU-only.  It treats a Step4 CSV as insufficient
until the export manifest, index contract, frozen lineage, RCR posterior
columns, and partial-artifact accounting all validate from disk.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from odcr_core.index_contract import (
    INDEX_CONTRACT_FILENAME,
    INDEX_CONTRACT_SCHEMA_VERSION,
    ODCR_ROUTING_TRAIN_CSV,
    STEP4_RCR_REQUIRED_COLUMNS,
    load_index_contract,
    validate_step4_export_lineage,
)


STEP4_EXPORT_MANIFEST = "step4_train_table_manifest.json"
STEP4_EXPORT_READINESS_SCHEMA_VERSION = "odcr_step4_export_readiness/1"
STEP4_EXPORT_READY_REQUIRED_COLUMNS = tuple(STEP4_RCR_REQUIRED_COLUMNS) + (
    "cf_reliability_score",
    "uncertainty_score",
)
STEP4_EXPORT_LEGACY_PRIMARY_COLUMNS = (
    "content_keywords",
    "content_aspects",
    "content_entities",
    "style_markers",
    "template_family",
    "length_style_bucket",
)


class Step4ExportValidationError(RuntimeError):
    """Raised when a Step4 export is not safe for downstream consumption."""


@dataclass
class Step4ExportValidationResult:
    ready: bool
    run_dir: Path
    export_path: Path
    manifest_path: Path
    index_contract_path: Path
    row_count: int = 0
    route_scorer_count: int = 0
    route_explainer_count: int = 0
    train_keep_count: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_payload(self, repo_root: str | Path | None = None) -> dict[str, Any]:
        root = Path(repo_root).expanduser().resolve() if repo_root is not None else None

        def rel(path: Path) -> str:
            p = path.resolve()
            if root is None:
                return str(p)
            try:
                return p.relative_to(root).as_posix()
            except ValueError:
                return p.as_posix()

        return {
            "schema_version": STEP4_EXPORT_READINESS_SCHEMA_VERSION,
            "ready": bool(self.ready),
            "run_dir": rel(self.run_dir),
            "export_path": rel(self.export_path),
            "manifest_path": rel(self.manifest_path),
            "index_contract_path": rel(self.index_contract_path),
            "row_count": int(self.row_count),
            "route_scorer_count": int(self.route_scorer_count),
            "route_explainer_count": int(self.route_explainer_count),
            "train_keep_count": int(self.train_keep_count),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "diagnostics": dict(self.diagnostics),
        }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise Step4ExportValidationError(f"{label} missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise Step4ExportValidationError(f"{label} invalid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise Step4ExportValidationError(f"{label} JSON root must be object: {path}")
    return payload


def _require_columns(df: pd.DataFrame, required: tuple[str, ...]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise Step4ExportValidationError("Step4 export missing required posterior columns: " + ", ".join(missing))


def _validate_binary_column(df: pd.DataFrame, col: str) -> int:
    vals = pd.to_numeric(df[col], errors="raise")
    bad = ~vals.isin([0, 1])
    if bool(bad.any()):
        raise Step4ExportValidationError(f"{col} must be parseable binary 0/1")
    return int((vals.astype(int) == 1).sum())


def _validate_numeric_unitish(df: pd.DataFrame, col: str, *, min_value: float = 0.0) -> tuple[float, float, float]:
    vals = pd.to_numeric(df[col], errors="raise").astype(float)
    arr = vals.to_numpy(dtype=float, copy=False)
    if not np.isfinite(arr).all():
        raise Step4ExportValidationError(f"{col} contains non-finite values")
    if float(arr.min(initial=min_value)) < min_value:
        raise Step4ExportValidationError(f"{col} contains values below {min_value}")
    return float(arr.min()), float(arr.mean()), float(arr.max())


def _partial_manifest_paths(run_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    for root in (run_dir / ".step4_partials", run_dir / "step4_partials"):
        if root.is_dir():
            candidates.extend(sorted(root.glob("*.manifest.json")))
    return candidates


def _resolve_partial_path(run_dir: Path, raw: Any) -> Path:
    text = str(raw or "").strip()
    if not text:
        raise Step4ExportValidationError("partial_manifest.path is required")
    path = Path(text).expanduser()
    return (run_dir / path).resolve() if not path.is_absolute() else path.resolve()


def _manifest_target_cf_rows(manifest: Mapping[str, Any]) -> int | None:
    rcr = manifest.get("rcr_routing")
    if not isinstance(rcr, Mapping):
        return None
    raw = rcr.get("n_target_rows_for_cf")
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise Step4ExportValidationError(f"partial_cf_rows_mismatch: invalid n_target_rows_for_cf={raw!r}") from exc
    if value < 0:
        raise Step4ExportValidationError(f"partial_cf_rows_mismatch: invalid n_target_rows_for_cf={value}")
    return value


def _validate_manifest_composition(manifest: Mapping[str, Any], final_rows: int) -> dict[str, Any]:
    row_counts = manifest.get("row_counts")
    if not isinstance(row_counts, Mapping):
        raise Step4ExportValidationError("final_csv_rows_mismatch: manifest.row_counts missing")
    manifest_total = int(row_counts.get("total_rows", -1))
    if manifest_total != int(final_rows):
        raise Step4ExportValidationError(
            f"final_csv_rows_mismatch: csv rows {int(final_rows)} != manifest total_rows {manifest_total}"
        )
    by_origin = row_counts.get("by_sample_origin")
    composition: dict[str, int] = {}
    if isinstance(by_origin, Mapping) and by_origin:
        composition = {str(k): int(v) for k, v in by_origin.items()}
        composed_total = int(sum(composition.values()))
        if composed_total != manifest_total:
            raise Step4ExportValidationError(
                f"composition_rows_mismatch: by_sample_origin sum {composed_total} != total_rows {manifest_total}"
            )
        target_cf_rows = _manifest_target_cf_rows(manifest)
        cf_rows = int(
            sum(
                value
                for key, value in composition.items()
                if key == "cf" or key.endswith("_cf") or "counterfactual" in key
            )
        )
        if target_cf_rows is not None and cf_rows and cf_rows != target_cf_rows:
            raise Step4ExportValidationError(
                f"composition_cf_rows_mismatch: final cf rows {cf_rows} != n_target_rows_for_cf {target_cf_rows}"
            )
    return {
        "status": "ok",
        "total_rows": manifest_total,
        "by_sample_origin": composition,
    }


def _validate_partial_accounting(run_dir: Path, manifest: Mapping[str, Any], final_rows: int) -> dict[str, Any]:
    for root in (run_dir / ".step4_partials", run_dir / "step4_partials"):
        if root.is_dir():
            failed = sorted(root.glob("*.failed"))
            if failed:
                raise Step4ExportValidationError("Step4 partial failed marker present: " + ", ".join(str(p) for p in failed[:5]))
    recorded = manifest.get("partial_artifacts")
    paths = _partial_manifest_paths(run_dir)
    if recorded is None and not paths:
        if _manifest_target_cf_rows(manifest) is not None:
            raise Step4ExportValidationError("partial_cf_rows_mismatch: partial artifacts missing for declared CF inference rows")
        return {"status": "absent_allowed_for_legacy_fixture", "row_count": final_rows}
    deduped: dict[str, Mapping[str, Any]] = {}
    if isinstance(recorded, list):
        for item in recorded:
            if isinstance(item, Mapping):
                key = str(_resolve_partial_path(run_dir, item.get("path")))
                deduped[key] = item
    for path in paths:
        item = _load_json(path, label="partial_manifest")
        key = str(_resolve_partial_path(run_dir, item.get("path")))
        deduped.setdefault(key, item)
    items: list[Mapping[str, Any]] = list(deduped.values())
    if not items:
        raise Step4ExportValidationError("partial artifacts are declared but empty")
    declared_rows = 0
    actual_rows = 0
    schemas: list[tuple[tuple[str, str], ...]] = []
    row_idx_chunks: list[np.ndarray] = []
    manifest_records: list[dict[str, Any]] = []
    for item in items:
        if str(item.get("status") or "ok") != "ok":
            raise Step4ExportValidationError(f"partial artifact not ok: {item}")
        declared_row_count = int(item.get("row_count", -1))
        if declared_row_count < 0:
            raise Step4ExportValidationError(f"partial_cf_rows_mismatch: invalid partial row_count {declared_row_count}")
        declared_rows += declared_row_count
        p = item.get("path")
        expected = str(item.get("sha256") or "")
        path = _resolve_partial_path(run_dir, p)
        if not path.is_file():
            raise Step4ExportValidationError(f"partial artifact missing: {path}")
        if expected and _sha256(path) != expected:
            raise Step4ExportValidationError(f"partial artifact hash mismatch: {path}")
        try:
            partial_df = pd.read_parquet(path)
        except Exception as exc:
            raise Step4ExportValidationError(f"partial artifact unreadable: {path}: {exc}") from exc
        if "row_idx" not in partial_df.columns:
            raise Step4ExportValidationError(f"partial_schema_mismatch: row_idx missing in {path}")
        actual_row_count = int(len(partial_df))
        if actual_row_count != declared_row_count:
            raise Step4ExportValidationError(
                f"partial_cf_rows_mismatch: manifest row_count {declared_row_count} != parquet rows {actual_row_count} for {path}"
            )
        actual_rows += actual_row_count
        schema = tuple((str(col), str(dtype)) for col, dtype in partial_df.dtypes.items())
        schemas.append(schema)
        idx = pd.to_numeric(partial_df["row_idx"], errors="raise").to_numpy(dtype=np.int64, copy=False)
        if len(np.unique(idx)) != len(idx):
            raise Step4ExportValidationError(f"partial_row_idx_overlap: duplicate row_idx inside {path}")
        row_idx_chunks.append(idx)
        manifest_records.append(
            {
                "path": str(path),
                "rank": item.get("rank"),
                "world_size": item.get("world_size"),
                "row_count": actual_row_count,
                "status": item.get("status", "ok"),
            }
        )
    if schemas and any(schema != schemas[0] for schema in schemas):
        raise Step4ExportValidationError("partial_schema_mismatch: partial parquet schemas differ")
    target_cf_rows = _manifest_target_cf_rows(manifest)
    if target_cf_rows is None:
        target_cf_rows = actual_rows
    if declared_rows != actual_rows:
        raise Step4ExportValidationError(
            f"partial_cf_rows_mismatch: manifest rows {declared_rows} != parquet rows {actual_rows}"
        )
    if actual_rows != target_cf_rows:
        raise Step4ExportValidationError(
            f"partial_cf_rows_mismatch: partial rows {actual_rows} != n_target_rows_for_cf {target_cf_rows}"
        )
    if row_idx_chunks:
        union = np.unique(np.concatenate(row_idx_chunks))
    else:
        union = np.asarray([], dtype=np.int64)
    if int(len(union)) != actual_rows:
        raise Step4ExportValidationError(
            f"partial_row_idx_overlap: row_idx union {int(len(union))} != partial rows {actual_rows}"
        )
    if target_cf_rows > 0:
        if int(union[0]) != 0 or int(union[-1]) != target_cf_rows - 1:
            raise Step4ExportValidationError(
                "partial_row_idx_gap: row_idx union must cover contiguous 0..n_target_rows_for_cf-1"
            )
        if not np.array_equal(union, np.arange(target_cf_rows, dtype=np.int64)):
            raise Step4ExportValidationError(
                "partial_row_idx_gap: row_idx union must cover contiguous 0..n_target_rows_for_cf-1"
            )
    return {
        "status": "ok",
        "partial_count": len(items),
        "row_count": actual_rows,
        "declared_row_count": declared_rows,
        "target_cf_rows": target_cf_rows,
        "final_total_rows": int(final_rows),
        "row_idx_union_count": int(len(union)),
        "row_idx_union_min": int(union[0]) if len(union) else None,
        "row_idx_union_max": int(union[-1]) if len(union) else None,
        "schema_all_equal": True,
        "manifests": manifest_records,
        "semantic": "partial parquet rows are CF target inference rows, not final training table rows",
    }


def _numeric_stats_update(stats: dict[str, dict[str, float]], col: str, values: pd.Series, *, min_value: float) -> None:
    vals = pd.to_numeric(values, errors="raise").astype(float)
    arr = vals.to_numpy(dtype=float, copy=False)
    if len(arr) == 0:
        return
    if not np.isfinite(arr).all():
        raise Step4ExportValidationError(f"{col} contains non-finite values")
    if float(arr.min()) < min_value:
        raise Step4ExportValidationError(f"{col} contains values below {min_value}")
    item = stats.setdefault(col, {"min": float("inf"), "sum": 0.0, "max": float("-inf"), "count": 0.0})
    item["min"] = min(float(item["min"]), float(arr.min()))
    item["sum"] = float(item["sum"]) + float(arr.sum())
    item["max"] = max(float(item["max"]), float(arr.max()))
    item["count"] = float(item["count"]) + float(len(arr))


def _validate_export_csv(export_path: Path) -> dict[str, Any]:
    stats: dict[str, dict[str, float]] = {}
    row_count = 0
    route_scorer_count = 0
    route_explainer_count = 0
    train_keep_count = 0
    columns: list[str] | None = None
    try:
        reader = pd.read_csv(export_path, chunksize=200_000)
        for chunk in reader:
            if columns is None:
                columns = [str(c) for c in chunk.columns]
                _require_columns(chunk, STEP4_EXPORT_READY_REQUIRED_COLUMNS)
                for legacy in STEP4_EXPORT_LEGACY_PRIMARY_COLUMNS:
                    if legacy in chunk.columns:
                        raise Step4ExportValidationError(
                            f"legacy retired column present in Step4 primary export: {legacy}"
                        )
            row_count += int(len(chunk))
            route_scorer_count += _validate_binary_column(chunk, "route_scorer")
            route_explainer_count += _validate_binary_column(chunk, "route_explainer")
            train_keep_count += _validate_binary_column(chunk, "train_keep")
            bucket_vals = pd.to_numeric(chunk["confidence_bucket"], errors="raise")
            if not bucket_vals.isin([0, 1, 2]).all():
                raise Step4ExportValidationError("confidence_bucket values must be in {0,1,2}")
            _numeric_stats_update(stats, "sample_weight_hint", chunk["sample_weight_hint"], min_value=0.0)
            _numeric_stats_update(stats, "uncertainty_score", chunk["uncertainty_score"], min_value=0.0)
            _numeric_stats_update(stats, "cf_reliability_score", chunk["cf_reliability_score"], min_value=0.0)
    except pd.errors.EmptyDataError as exc:
        raise Step4ExportValidationError("row_count must be > 0") from exc
    if row_count <= 0:
        raise Step4ExportValidationError("row_count must be > 0")
    if train_keep_count <= 0:
        raise Step4ExportValidationError("train_keep must have at least one kept row")
    if route_scorer_count + route_explainer_count <= 0:
        raise Step4ExportValidationError("scorer-clean / explainer-rich split has no routed rows")
    normalized_stats = {
        col: {
            "min": float(item["min"]),
            "mean": float(item["sum"]) / max(float(item["count"]), 1.0),
            "max": float(item["max"]),
        }
        for col, item in stats.items()
    }
    return {
        "row_count": int(row_count),
        "route_scorer_count": int(route_scorer_count),
        "route_explainer_count": int(route_explainer_count),
        "train_keep_count": int(train_keep_count),
        "columns": columns or [],
        "numeric_stats": normalized_stats,
        "required_columns_missing": [],
    }


def _validate_train_csv_fingerprint(
    *,
    contract: Mapping[str, Any],
    export_path: Path,
    export_sha256: str,
) -> dict[str, Any]:
    fingerprints = contract.get("fingerprints")
    if not isinstance(fingerprints, Mapping):
        raise Step4ExportValidationError("contract_stale: index_contract.fingerprints missing")
    train_fp = fingerprints.get("train_csv")
    if not isinstance(train_fp, Mapping):
        raise Step4ExportValidationError("contract_stale: index_contract.fingerprints.train_csv missing")
    if train_fp.get("exists") is not True:
        raise Step4ExportValidationError("contract_stale: index_contract.fingerprints.train_csv.exists is false")
    if train_fp.get("is_file") is not True:
        raise Step4ExportValidationError("contract_stale: index_contract.fingerprints.train_csv.is_file is not true")
    raw_path = str(train_fp.get("path") or contract.get("train_csv_path") or "").strip()
    if not raw_path:
        raise Step4ExportValidationError("contract_stale: index_contract train_csv path missing")
    fp_path = Path(raw_path).expanduser()
    if not fp_path.is_absolute():
        fp_path = (export_path.parent / fp_path).resolve()
    else:
        fp_path = fp_path.resolve()
    if fp_path != export_path.resolve():
        raise Step4ExportValidationError(
            f"contract_stale: train_csv fingerprint path {fp_path} != export path {export_path.resolve()}"
        )
    stat = export_path.stat()
    if train_fp.get("size") is not None and int(train_fp.get("size")) != int(stat.st_size):
        raise Step4ExportValidationError(
            f"contract_stale: train_csv size {train_fp.get('size')} != actual {int(stat.st_size)}"
        )
    if train_fp.get("mtime_ns") is not None and int(train_fp.get("mtime_ns")) != int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9))):
        raise Step4ExportValidationError("contract_stale: train_csv mtime_ns mismatch")
    if str(train_fp.get("sha256") or "") != str(export_sha256):
        raise Step4ExportValidationError("contract_stale: train_csv sha256 mismatch")
    return {
        "status": "ok",
        "path": str(export_path.resolve()),
        "exists": True,
        "is_file": True,
        "size": int(stat.st_size),
        "mtime_ns": int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9))),
        "sha256": export_sha256,
        "schema_version": str(train_fp.get("schema_version") or "odcr_file_fingerprint/1"),
        "fingerprint_version": str(train_fp.get("fingerprint_version") or train_fp.get("schema_version") or "odcr_file_fingerprint/1"),
    }


def validate_step4_export_ready(
    run_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
    current_step4_rcr_config: Mapping[str, Any] | None = None,
    task_id: int | None = None,
    auxiliary_domain: str | None = None,
    target_domain: str | None = None,
    raise_on_error: bool = False,
) -> Step4ExportValidationResult:
    root = Path(repo_root).expanduser().resolve() if repo_root is not None else Path.cwd().resolve()
    run = Path(run_dir).expanduser()
    if not run.is_absolute():
        run = (root / run).resolve()
    else:
        run = run.resolve()
    export_path = run / ODCR_ROUTING_TRAIN_CSV
    manifest_path = run / STEP4_EXPORT_MANIFEST
    index_contract_path = run / INDEX_CONTRACT_FILENAME
    result = Step4ExportValidationResult(
        ready=False,
        run_dir=run,
        export_path=export_path,
        manifest_path=manifest_path,
        index_contract_path=index_contract_path,
    )
    try:
        if not export_path.is_file():
            raise Step4ExportValidationError(f"final export file missing: {export_path}")
        if not manifest_path.is_file():
            raise Step4ExportValidationError(f"export manifest missing: {manifest_path}")
        if not index_contract_path.is_file():
            raise Step4ExportValidationError(f"index_contract missing: {index_contract_path}")
        manifest = _load_json(manifest_path, label="export_manifest")
        contract = load_index_contract(str(index_contract_path))
        if str(contract.get("schema_version") or "") != INDEX_CONTRACT_SCHEMA_VERSION:
            raise Step4ExportValidationError("index_contract schema_version mismatch")
        csv_stats = _validate_export_csv(export_path)
        result.row_count = int(csv_stats["row_count"])
        result.route_scorer_count = int(csv_stats["route_scorer_count"])
        result.route_explainer_count = int(csv_stats["route_explainer_count"])
        result.train_keep_count = int(csv_stats["train_keep_count"])
        composition = _validate_manifest_composition(manifest, result.row_count)
        export_sha256 = _sha256(export_path)
        train_csv_fingerprint = _validate_train_csv_fingerprint(
            contract=contract,
            export_path=export_path,
            export_sha256=export_sha256,
        )
        lineage = contract.get("step4_export_lineage") or manifest.get("step4_export_lineage")
        if not isinstance(lineage, Mapping):
            raise Step4ExportValidationError("frozen step4_export_lineage missing")
        if current_step4_rcr_config is not None and task_id is not None and auxiliary_domain and target_domain:
            validate_step4_export_lineage(
                {**contract, "step4_export_lineage": lineage},
                current_step4_rcr_config=current_step4_rcr_config,
                task_id=int(task_id),
                auxiliary_domain=str(auxiliary_domain),
                target_domain=str(target_domain),
            )
        frozen = lineage.get("frozen_step3_lineage")
        if not isinstance(frozen, Mapping):
            raise Step4ExportValidationError("frozen Step3 lineage missing from Step4 export lineage")
        frozen_required = (
            "upstream_step3_run_id",
            "step3_checkpoint_path",
            "step3_checkpoint_hash",
            "step3_stage_status_hash",
            "step3_eval_handoff_hash",
        )
        missing_frozen = [key for key in frozen_required if not str(frozen.get(key) or "").strip()]
        if missing_frozen:
            raise Step4ExportValidationError("frozen Step3 lineage missing fields: " + ", ".join(missing_frozen))
        partials = _validate_partial_accounting(run, manifest, result.row_count)
        result.diagnostics = {
            "manifest_schema_version": manifest.get("schema_version"),
            "index_contract_schema_version": contract.get("schema_version"),
            "export_sha256": export_sha256,
            "manifest_sha256": _sha256(manifest_path),
            "index_contract_sha256": _sha256(index_contract_path),
            "sample_weight_hint_stats": csv_stats["numeric_stats"]["sample_weight_hint"],
            "uncertainty_score_stats": csv_stats["numeric_stats"]["uncertainty_score"],
            "cf_reliability_score_stats": csv_stats["numeric_stats"]["cf_reliability_score"],
            "final_export": {
                "row_count": result.row_count,
                "required_columns_missing": csv_stats["required_columns_missing"],
                "route_scorer_count": result.route_scorer_count,
                "route_explainer_count": result.route_explainer_count,
                "train_keep_count": result.train_keep_count,
            },
            "composition": composition,
            "train_csv_fingerprint": train_csv_fingerprint,
            "partial_artifacts": partials,
            "step5_required_fields_precheck": "passed",
        }
        result.ready = True
    except Exception as exc:
        result.errors.append(str(exc))
        if raise_on_error:
            raise
    return result


def require_step4_export_ready(run_dir: str | Path, **kwargs: Any) -> Step4ExportValidationResult:
    result = validate_step4_export_ready(run_dir, **kwargs)
    if not result.ready:
        raise Step4ExportValidationError("; ".join(result.errors) or "Step4 export not ready")
    return result


__all__ = [
    "STEP4_EXPORT_MANIFEST",
    "STEP4_EXPORT_READINESS_SCHEMA_VERSION",
    "Step4ExportValidationError",
    "Step4ExportValidationResult",
    "require_step4_export_ready",
    "validate_step4_export_ready",
]
