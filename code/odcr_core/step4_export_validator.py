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


def _validate_partial_accounting(run_dir: Path, manifest: Mapping[str, Any], final_rows: int) -> dict[str, Any]:
    for root in (run_dir / ".step4_partials", run_dir / "step4_partials"):
        if root.is_dir():
            failed = sorted(root.glob("*.failed"))
            if failed:
                raise Step4ExportValidationError("Step4 partial failed marker present: " + ", ".join(str(p) for p in failed[:5]))
    recorded = manifest.get("partial_artifacts")
    paths = _partial_manifest_paths(run_dir)
    if recorded is None and not paths:
        return {"status": "absent_allowed_for_legacy_fixture", "row_count": final_rows}
    items: list[Mapping[str, Any]] = []
    if isinstance(recorded, list):
        items.extend(x for x in recorded if isinstance(x, Mapping))
    for path in paths:
        items.append(_load_json(path, label="partial_manifest"))
    if not items:
        raise Step4ExportValidationError("partial artifacts are declared but empty")
    rows = 0
    for item in items:
        if str(item.get("status") or "ok") != "ok":
            raise Step4ExportValidationError(f"partial artifact not ok: {item}")
        rows += int(item.get("row_count", -1))
        p = item.get("path")
        expected = str(item.get("sha256") or "")
        if p and expected:
            path = Path(str(p))
            if not path.is_absolute():
                path = run_dir / path
            if not path.is_file():
                raise Step4ExportValidationError(f"partial artifact missing: {path}")
            if _sha256(path) != expected:
                raise Step4ExportValidationError(f"partial artifact hash mismatch: {path}")
    if rows != int(final_rows):
        raise Step4ExportValidationError(f"partial row count {rows} != final row count {final_rows}")
    return {"status": "ok", "partial_count": len(items), "row_count": rows}


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
        df = pd.read_csv(export_path)
        if len(df) <= 0:
            raise Step4ExportValidationError("row_count must be > 0")
        _require_columns(df, STEP4_EXPORT_READY_REQUIRED_COLUMNS)
        for legacy in STEP4_EXPORT_LEGACY_PRIMARY_COLUMNS:
            if legacy in df.columns:
                raise Step4ExportValidationError(f"legacy retired column present in Step4 primary export: {legacy}")
        result.route_scorer_count = _validate_binary_column(df, "route_scorer")
        result.route_explainer_count = _validate_binary_column(df, "route_explainer")
        result.train_keep_count = _validate_binary_column(df, "train_keep")
        if result.train_keep_count <= 0:
            raise Step4ExportValidationError("train_keep must have at least one kept row")
        bucket_vals = pd.to_numeric(df["confidence_bucket"], errors="raise")
        if not bucket_vals.isin([0, 1, 2]).all():
            raise Step4ExportValidationError("confidence_bucket values must be in {0,1,2}")
        sample_weight_stats = _validate_numeric_unitish(df, "sample_weight_hint", min_value=0.0)
        _validate_numeric_unitish(df, "uncertainty_score", min_value=0.0)
        _validate_numeric_unitish(df, "cf_reliability_score", min_value=0.0)
        if result.route_scorer_count + result.route_explainer_count <= 0:
            raise Step4ExportValidationError("scorer-clean / explainer-rich split has no routed rows")
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
        partials = _validate_partial_accounting(run, manifest, len(df))
        result.row_count = int(len(df))
        result.diagnostics = {
            "manifest_schema_version": manifest.get("schema_version"),
            "index_contract_schema_version": contract.get("schema_version"),
            "export_sha256": _sha256(export_path),
            "manifest_sha256": _sha256(manifest_path),
            "index_contract_sha256": _sha256(index_contract_path),
            "sample_weight_hint_stats": {
                "min": sample_weight_stats[0],
                "mean": sample_weight_stats[1],
                "max": sample_weight_stats[2],
            },
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
